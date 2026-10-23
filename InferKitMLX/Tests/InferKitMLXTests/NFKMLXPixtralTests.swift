//
//  NFKMLXPixtralTests.swift
//  InferKitMLXTests
//
//  Pixtral 12B (mistral-experimental/pixtral-12b, Mistral, Apache-2.0), the vision-language model whose
//  novel part is a from-scratch 2D-rotary vision tower. The module evaluates MLX arrays, so these skip
//  without a Metal library for MLX (see Tools/mlx-metallib.sh). Parity is gated on the released weights
//  + the recorded oracle (`IK_VAL_PIXTRAL` = the release directory, `IK_PARITY_PIXTRAL` = the record
//  from `run_reference.py pixtral`). The vision tower and connector are compared in float32 seam by
//  seam. The fused decoder is compared to `run_reference.py pixtral_fused_layerwise` given the same
//  projected features: at float32 state by state (`IK_PARITY_PIXTRAL_FUSED`); at the release's
//  bfloat16 layer by layer against the reference's block floors (`IK_PARITY_PIXTRAL_FUSED_BF16`); and at
//  bfloat16 end to end by a rank test against torch's jittered twins (`IK_PARITY_PIXTRAL_FUSED_TWINS`).
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
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< n {
            let x = Double(a[i]), y = Double(b[i])
            dot += x * y; na += x * x; nb += y * y
        }
        return Float(dot / (na.squareRoot() * nb.squareRoot() + 1e-300))
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

    /// The streamed decoder at `precision` over the fused sequence of a `run_reference.py
    /// pixtral_fused_layerwise` record (`recordKey`), given the `run_pixtral` record's projected features as
    /// the oracle was: the record, the fused input, every layer's state, and the logits.
    private func fusedPass(record recordKey: String, precision: NFKMLXWeightPrecision) throws
        -> (record: [String: MLXArray], embeddings: MLXArray, states: [MLXArray], logits: MLXArray) {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_PIXTRAL"], let featuresPath = env["IK_PARITY_PIXTRAL"],
              let recordPath = env[recordKey] else {
            throw XCTSkip("set IK_VAL_PIXTRAL, IK_PARITY_PIXTRAL, and \(recordKey) "
                          + "(run_reference.py pixtral_fused_layerwise)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let decoder = try NFKMLXPixtral.decoder(directoryURL: URL(fileURLWithPath: directory),
                                                precision: precision, residency: .streamed)
        let inputIds = rec["input_ids"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let projected = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: featuresPath)).arrays["projected"]!
        let features = projected[0].asType(decoder.model.embedTokens.weight.dtype)
        let (embeddings, imageCount) = NFKMLXPixtral.fusedInput(decoder: decoder, inputIds: inputIds, features: features)
        XCTAssertEqual(imageCount, features.dim(0), "one [IMG] position per projected feature")
        let states = decoder.layerStates(fromEmbeddings: embeddings)
        let logits = decoder.logits(fromHidden: states[states.count - 1])[0]
        eval(logits)
        try decoder.verifyStream()
        return (rec, embeddings[0], states.map { $0[0] }, logits)
    }

    /// The cosine between each row of `mine` and of `reference`, `[rows, width]`.
    private func rowCosines(_ mine: MLXArray, _ reference: MLXArray) -> [Float] {
        let a = mine.asType(.float32), b = reference.asType(.float32)
        return ((a * b).sum(axis: -1) / (sqrt((a * a).sum(axis: -1)) * sqrt((b * b).sum(axis: -1)))).asArray(Float.self)
    }

    private func argmaxDifferences(_ mine: MLXArray, _ reference: MLXArray) -> [Int] {
        let a = mine.argMax(axis: -1).asArray(Int32.self), b = reference.argMax(axis: -1).asArray(Int32.self)
        return (0 ..< a.count).filter { a[$0] != b[$0] }
    }

    /// PARITY: the released 12B decoder, streamed at float32, over the fused sequence against the float32
    /// reference decoder given the same projected features, at every position after every layer. The
    /// vision tower and connector are held to their own record in ``testVisionParityOnTheReleasedWeights``.
    func testFusedDecoderMatchesTheReferenceOnReleasedWeights() throws {
        let run = try fusedPass(record: "IK_PARITY_PIXTRAL_FUSED", precision: .float32)
        let rec = run.record
        XCTAssertGreaterThan(rowCosines(run.embeddings, rec["fused_embeds"]!).min()!, 0.99999, "the fused input")
        var worst = (cosine: Float(1), layer: 0)
        for index in 1 ..< run.states.count - 1 {
            let measured = rowCosines(run.states[index], rec["hidden.\(index)"]!).min()!
            if measured < worst.cosine { worst = (measured, index) }
        }
        let logitCosine = cosine(run.logits, rec["output"]!)
        let differing = argmaxDifferences(run.logits, rec["output"]!)
        print("pixtral-12b fused (float32): logit cosine \(logitCosine), argmax \(run.logits.dim(0) - differing.count)/"
              + "\(run.logits.dim(0)), worst position state \(worst.cosine) after layer \(worst.layer)")
        XCTAssertGreaterThan(worst.cosine, 0.9999, "the state after layer \(worst.layer)")
        XCTAssertGreaterThan(logitCosine, 0.9999, "the fused logits")
        XCTAssertLessThanOrEqual(differing.count, 1, "the fused argmax differs at \(differing)")
    }

    /// PARITY: the released 12B decoder streamed at its own bfloat16 over the fused sequence, held as a
    /// distribution. Under 20 one-unit jitters of 16 fused-embedding values, this side's distances from the
    /// float32 reference (`IK_PARITY_PIXTRAL_FUSED`) sit no farther than torch's bfloat16 twins under the
    /// same jitters (`IK_PARITY_PIXTRAL_FUSED_TWINS`, `PIXTRAL_JITTERS=1-20`), by a rank test (z ≤ 2): the
    /// worst position after each layer, and the logits. Image positions such as 71 amplify a one-unit
    /// difference through the stack in torch's own runs, so a single run's distance is no bar.
    func testFusedDecoderAtBfloat16MatchesTheReferenceUnderJitter() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_PIXTRAL"], let featuresPath = env["IK_PARITY_PIXTRAL"],
              let exactPath = env["IK_PARITY_PIXTRAL_FUSED"], let twinsPath = env["IK_PARITY_PIXTRAL_FUSED_TWINS"] else {
            throw XCTSkip("set IK_VAL_PIXTRAL, IK_PARITY_PIXTRAL, IK_PARITY_PIXTRAL_FUSED, and "
                          + "IK_PARITY_PIXTRAL_FUSED_TWINS (run_reference.py pixtral_fused_layerwise)")
        }
        let exact = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: exactPath)).arrays
        let twins = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: twinsPath)).arrays
        let decoder = try NFKMLXPixtral.decoder(directoryURL: URL(fileURLWithPath: directory), precision: .checkpoint,
                                                residency: .streamed)
        let inputIds = exact["input_ids"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let projected = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: featuresPath)).arrays["projected"]!
        let embeddings = NFKMLXPixtral.fusedInput(decoder: decoder, inputIds: inputIds,
                                                  features: projected[0].asType(.bfloat16)).embeddings
        let layers = decoder.configuration.layerCount
        let targets = (0 ..< layers).map { rowsOf(exact["hidden.\($0)"]!) }
        let targetLogits = exact["output"]!.reshaped([-1]).asArray(Float.self).map(Double.init)

        var ours = [[Double]](repeating: [], count: layers + 1), theirs = ours
        for seed in 1 ... 20 {
            let states = decoder.layerStates(fromEmbeddings: jittered(embeddings, seed: seed))
            let logits = decoder.logits(fromHidden: states[layers])[0].asType(.float32)
            eval(logits)
            for index in 1 ..< layers {
                ours[index].append(worstRowDistance(states[index][0], targets[index]))
                theirs[index].append(Double(twins["twin\(seed).distance"]![index].max().item(Float.self)))
            }
            ours[layers].append(distance(logits.reshaped([-1]).asArray(Float.self).map(Double.init), targetLogits))
            theirs[layers].append(Double(twins["twin\(seed).logit_distance"]![0].item(Float.self)))
        }
        try decoder.verifyStream()
        var worst = (z: -Double.infinity, label: "")
        for index in 1 ... layers {
            let label = index == layers ? "the logits" : "the worst position after layer \(index)"
            let z = rankZ(ours[index], theirs[index])
            print(String(format: "pixtral-12b fused (bfloat16) %@ under jitter: ours median %.4g, max %.4g; torch's twins "
                         + "median %.4g, max %.4g; rank z %+.2f", label, median(ours[index]), ours[index].max()!,
                         median(theirs[index]), theirs[index].max()!, z))
            if z > worst.z { worst = (z, label) }
        }
        XCTAssertLessThanOrEqual(worst.z, 2, "\(worst.label): farther from float32 than torch under the same jitters")
    }

    /// `embeddings` with up to 16 values one unit in the last place larger in magnitude, at the positions
    /// `run_reference.py`'s `_jittered_bfloat16(embeddings, seed)` raises.
    private func jittered(_ embeddings: MLXArray, seed: Int) -> MLXArray {
        var state = UInt64(seed) &* 0x9E37_79B9_7F4A_7C15
        var bumps = [UInt16](repeating: 0, count: embeddings.size)
        for _ in 0 ..< 16 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            bumps[Int(state >> 33) % embeddings.size] = 1
        }
        return (embeddings.reshaped([-1]).view(dtype: .uint16) + MLXArray(bumps)).view(dtype: embeddings.dtype)
            .reshaped(embeddings.shape)
    }

    private func rowsOf(_ array: MLXArray) -> [[Double]] {
        let width = array.dim(-1)
        let values = array.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        return stride(from: 0, to: values.count, by: width).map { Array(values[$0 ..< $0 + width]) }
    }

    /// The largest `1 - cosine` between a row of `state` and the same row of `target`, in Double.
    private func worstRowDistance(_ state: MLXArray, _ target: [[Double]]) -> Double {
        zip(rowsOf(state), target).map { distance($0, $1) }.max()!
    }

    private func distance(_ a: [Double], _ b: [Double]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(a.count, b.count) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return 1 - dot / (na.squareRoot() * nb.squareRoot())
    }

    private func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted.count % 2 == 1 ? sorted[sorted.count / 2] : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
    }

    /// The Mann–Whitney rank-sum statistic of `ours` against `theirs` as a z score, ties at mid-rank: positive
    /// where `ours` ranks higher.
    private func rankZ(_ ours: [Double], _ theirs: [Double]) -> Double {
        let pooled = (ours.map { ($0, true) } + theirs.map { ($0, false) }).sorted { $0.0 < $1.0 }
        var ranks = [Double](repeating: 0, count: pooled.count)
        var start = 0
        while start < pooled.count {
            var end = start
            while end + 1 < pooled.count && pooled[end + 1].0 == pooled[start].0 { end += 1 }
            for index in start ... end { ranks[index] = Double(start + end) / 2 + 1 }
            start = end + 1
        }
        let rankSum = zip(pooled, ranks).filter { $0.0.1 }.reduce(0) { $0 + $1.1 }
        let (n1, n2) = (Double(ours.count), Double(theirs.count))
        return (rankSum - n1 * (n1 + 1) / 2 - n1 * n2 / 2) / (n1 * n2 * (n1 + n2 + 1) / 12).squareRoot()
    }

    /// PARITY: each layer of the released 12B decoder at its own bfloat16, run alone on the bfloat16
    /// reference's input to it over the fused sequence, against the reference's output: the distance stays
    /// below the block floor, the rounding the reference's own layer adds (`hidden.<i+1>.floor`, the layer
    /// rerun in float32 on the same input; `IK_PROBE_BLOCK_FLOOR=1`).
    func testFusedDecoderLayersAtBfloat16StayWithinTheirBlockFloors() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_PIXTRAL"], let recordPath = env["IK_PARITY_PIXTRAL_FUSED_BF16"] else {
            throw XCTSkip("set IK_VAL_PIXTRAL and IK_PARITY_PIXTRAL_FUSED_BF16 (run_reference.py pixtral_fused_layerwise)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let decoder = try NFKMLXPixtral.decoder(directoryURL: URL(fileURLWithPath: directory), precision: .checkpoint,
                                                residency: .streamed)
        var worst = (ratio: Float(0), layer: 0)
        for index in 0 ..< decoder.configuration.layerCount - 1 {
            let input = rec["hidden.\(index)"]!.asType(.bfloat16).expandedDimensions(axis: 0)
            let output = try XCTUnwrap(decoder.layerOutput(index, input), "layer \(index) ran")[0]
            let expected = rec["hidden.\(index + 1)"]!
            let floor = try XCTUnwrap(rec["hidden.\(index + 1).floor"], "the block floor of layer \(index)")
            let ours = 1 - cosine(output, expected), reference = 1 - cosine(floor, expected)
            let positions = rowCosines(output, expected)
            let position = positions.indices.min { positions[$0] < positions[$1] }!
            let floorPositions = rowCosines(floor, expected)
            print("pixtral-12b fused (bfloat16) layer \(index) alone: \(ours) against its block floor \(reference) "
                  + "(\(ours / reference) ×); worst position \(position) at \(1 - positions[position]), its floor there "
                  + "\(1 - floorPositions[position])")
            if ours / reference > worst.ratio { worst = (ours / reference, index) }
        }
        try decoder.verifyStream()
        XCTAssertLessThan(worst.ratio, 1, "layer \(worst.layer) on the reference's input: farther from it than its own "
                          + "block's rounding")
    }
}
