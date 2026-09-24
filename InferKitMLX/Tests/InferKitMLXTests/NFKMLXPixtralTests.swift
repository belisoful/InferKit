//
//  NFKMLXPixtralTests.swift
//  InferKitMLXTests
//
//  Pixtral 12B (mistral-experimental/pixtral-12b, Mistral, Apache-2.0), the vision-language model whose
//  novel part is a from-scratch 2D-rotary vision tower. The module evaluates MLX arrays, so these skip
//  without a Metal library for MLX (see Tools/mlx-metallib.sh). Parity is gated on the released weights
//  + the recorded oracle (`IK_VAL_PIXTRAL` = the release directory, `IK_PARITY_PIXTRAL` = the record
//  from `run_reference.py pixtral`). The vision tower and connector are compared in float32 seam by
//  seam; the fused decoder is compared in the release bfloat16 by argmax.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXPixtralTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        eval(mine)
        let a = mine.reshaped([-1]).asType(.float32).asArray(Float.self)
        let b = reference.reshaped([-1]).asType(.float32).asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    /// Casts a module's parameters to float32 for a high-precision comparison against the fp32 oracle.
    private func castToFloat32(_ module: Module) {
        let cast = module.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }
        module.update(parameters: ModuleParameters.unflattened(cast))
        eval(module)
    }

    private static let tinyVision = NFKMLXPixtralVisionConfiguration(
        hiddenSize: 32, depth: 4, headCount: 2, intermediateSize: 64, patchSize: 2, imageSize: 16)

    /// The vision tower embeds each patch, reads it with the 2D rotary, and returns one feature per patch.
    func testTheVisionTowerProducesOneFeaturePerPatch() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(6)
        let net = NFKMLXPixtralVisionNet(Self.tinyVision)
        let pixels = MLXRandom.normal([1, 3, 8, 6])                          // a 4×3 patch grid
        let output = net(pixels)
        eval(output)
        XCTAssertEqual(output.shape, [12, 32], "one feature per patch at the vision width")

        let (cos, sin) = net.rotaryEmbedding(gridH: 4, gridW: 3)
        eval(cos, sin)
        XCTAssertEqual(cos.shape, [1, 12, 16], "the rotary table covers the full head dimension")
    }

    /// The connector projects patch features from the vision width to the decoder width.
    func testTheConnectorProjectsToTheDecoderWidth() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(6)
        let connector = NFKMLXPixtralConnector(visionSize: 32, textSize: 48)
        let projected = connector(MLXRandom.normal([12, 32]))
        eval(projected)
        XCTAssertEqual(projected.shape, [12, 48])
    }

    /// The image processor resizes within the longest edge and rounds each side to a multiple of the patch.
    func testTheProcessorResizesToPatchMultiples() throws {
        let processor = NFKMLXPixtralImageProcessor()
        let (height, width) = processor.resize(height: 2000, width: 1000)
        XCTAssertLessThanOrEqual(max(height, width), 1024, "the longest edge is bounded")
        XCTAssertEqual(height % 16, 0, "each side is a multiple of the patch")
        XCTAssertEqual(width % 16, 0)
    }

    /// PARITY: the released vision tower and connector against the fp32 oracle, seam by seam.
    func testVisionParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_PIXTRAL"], let recordPath = env["IK_PARITY_PIXTRAL"] else {
            throw XCTSkip("set IK_VAL_PIXTRAL (release directory) and IK_PARITY_PIXTRAL (oracle record)")
        }
        let directoryURL = URL(fileURLWithPath: directory)
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        let vision = try NFKMLXPixtral.visionNet(directoryURL: directoryURL)
        let connector = try NFKMLXPixtral.connector(directoryURL: directoryURL)
        castToFloat32(vision)
        castToFloat32(connector)

        let pixelValues = rec["pixel_values"]!                              // [1, 3, H, W]
        let seams = vision.intermediateSeams(pixelValues)
        XCTAssertGreaterThan(cosine(seams.patch, rec["patch_embeds"]![0]), 0.999, "patch embedding diverges")
        XCTAssertGreaterThan(cosine(seams.lnPre, rec["ln_pre"]![0]), 0.999, "ln_pre diverges")
        XCTAssertGreaterThan(cosine(seams.layer0, rec["layer0"]![0]), 0.999, "first block diverges")
        XCTAssertGreaterThan(cosine(seams.output, rec["vision_output"]![0]), 0.999, "vision output diverges")
        XCTAssertGreaterThan(cosine(connector(seams.output), rec["projected"]![0]), 0.999, "connector diverges")
    }

    /// PARITY: the whole pipeline — vision tower, connector, the scatter of projected features into the
    /// `[IMG]` positions, and the Mistral decoder — against a tiny float32 oracle, loaded through the
    /// ordinary release builders. This measures the fusion and the decoder integration the released
    /// weights would, at a size that runs in memory; the released 12B decoder does not fit the fused
    /// pass on a 32 GB machine, so it is measured separately behind an opt-in below.
    func testFusedPipelineParityOnTinyWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_PIXTRAL_TINY"], let recordPath = env["IK_PARITY_PIXTRAL_TINY"] else {
            throw XCTSkip("set IK_VAL_PIXTRAL_TINY and IK_PARITY_PIXTRAL_TINY (run_reference.py pixtral_tiny)")
        }
        let directoryURL = URL(fileURLWithPath: directory)
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        let vision = try NFKMLXPixtral.visionNet(directoryURL: directoryURL)
        let connector = try NFKMLXPixtral.connector(directoryURL: directoryURL)
        let decoder = try NFKMLXPixtral.decoder(directoryURL: directoryURL)

        let visionOutput = vision(rec["pixel_values"]!)
        XCTAssertGreaterThan(cosine(visionOutput, rec["vision_output"]![0]), 0.9999, "vision output diverges")
        let projected = connector(visionOutput)
        XCTAssertGreaterThan(cosine(projected, rec["projected"]![0]), 0.9999, "connector diverges")

        let inputIds = rec["input_ids"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let logits = NFKMLXPixtral.logits(decoder: decoder, inputIds: inputIds, features: projected)[0]
        let reference = rec["output"]!                                      // [sequence, vocabulary]
        XCTAssertGreaterThan(cosine(logits, reference), 0.9999, "the fused logits diverge")

        let sequence = reference.dim(0)
        let mineArgmax = logits.argMax(axis: -1).asArray(Int32.self)
        let referenceArgmax = reference.argMax(axis: -1).asArray(Int32.self)
        for position in 0 ..< sequence {
            XCTAssertEqual(mineArgmax[position], referenceArgmax[position], "argmax differs at \(position)")
        }
    }

    /// PARITY (opt-in, memory-heavy): the fused decoder against the released bfloat16 oracle. The 12B
    /// decoder needs ~24 GB resident for the fused pass, which does not fit alongside other work on a
    /// 32 GB machine, so this runs only when `IK_PIXTRAL_FUSED` is set. The recorded projected features
    /// splice into the `[IMG]` positions, and the decoder's argmax is compared to the reference's.
    func testFusedDecoderMatchesTheReferenceOnReleasedWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard env["IK_PIXTRAL_FUSED"] != nil else {
            throw XCTSkip("set IK_PIXTRAL_FUSED to run the memory-heavy 12B fused-decoder parity")
        }
        guard let directory = env["IK_VAL_PIXTRAL"], let recordPath = env["IK_PARITY_PIXTRAL"] else {
            throw XCTSkip("set IK_VAL_PIXTRAL (release directory) and IK_PARITY_PIXTRAL (oracle record)")
        }
        let directoryURL = URL(fileURLWithPath: directory)
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        let decoder = try NFKMLXPixtral.decoder(directoryURL: directoryURL)
        let inputIds = rec["input_ids"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let features = rec["projected"]![0].asType(.bfloat16)               // [patches, textHidden]

        let logits = NFKMLXPixtral.logits(decoder: decoder, inputIds: inputIds, features: features)[0]
        eval(logits)
        let reference = rec["output"]!
        let sequence = reference.dim(0)
        let mineArgmax = logits.argMax(axis: -1).asArray(Int32.self)
        let referenceArgmax = reference.argMax(axis: -1).asArray(Int32.self)
        var agree = 0
        for position in 0 ..< sequence where mineArgmax[position] == referenceArgmax[position] { agree += 1 }
        XCTAssertGreaterThanOrEqual(agree, sequence - 1,
                                    "the fused decoder disagrees with the reference at \(sequence - agree) "
                                    + "of \(sequence) positions")
    }
}
