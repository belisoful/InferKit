//
//  NFKMLXCosmosTokenizerTests.swift
//  InferKitMLXTests
//
//  The Cosmos Tokenizer (nvidia/Cosmos-0.1-Tokenizer-*): image and causal video autoencoders with a
//  continuous latent or FSQ tokens. The released-weight tests are gated on the directory holding one
//  subdirectory per variant (`IK_VAL_COSMOS_TOKENIZER`: `CI8x8/encoder.jit`, `CI8x8/decoder.jit`, …) and
//  on the per-variant records `run_reference.py cosmos_tokenizer` wrote beside them
//  (`IK_PARITY_COSMOS_TOKENIZER`: `CI8x8/record.safetensors`, …). Every variant present is compared
//  seam by seam, from the wavelet patch through the latent or tokens to the reconstruction.
//

import XCTest
import InferKit
import CoreGraphics
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXCosmosTokenizerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Double {
        let a = mine.asType(.float32).reshaped([-1]).asArray(Float.self)
        let b = reference.asType(.float32).reshaped([-1]).asArray(Float.self)
        XCTAssertEqual(a.count, b.count)
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(a.count, b.count) {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        return dot / ((na * nb).squareRoot() + 1e-30)
    }

    /// The variants whose release and record are both present.
    private func releasedVariants() throws -> [(NFKMLXCosmosTokenizerVariant, URL, [String: MLXArray])] {
        let environment = NFKMLXValidationConfig.environment
        guard let root = environment["IK_VAL_COSMOS_TOKENIZER"],
              let records = environment["IK_PARITY_COSMOS_TOKENIZER"] else {
            throw XCTSkip("set IK_VAL_COSMOS_TOKENIZER and IK_PARITY_COSMOS_TOKENIZER")
        }
        var found: [(NFKMLXCosmosTokenizerVariant, URL, [String: MLXArray])] = []
        for variant in NFKMLXCosmosTokenizerVariant.allCases {
            let directory = URL(fileURLWithPath: root).appendingPathComponent(variant.releaseName)
            let record = URL(fileURLWithPath: records).appendingPathComponent(variant.releaseName)
                .appendingPathComponent("record.safetensors")
            guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("encoder.jit").path),
                  FileManager.default.fileExists(atPath: record.path) else { continue }
            found.append((variant, directory, try NFKMLXWeights.loadCheckpoint(url: record).arrays))
        }
        try XCTSkipIf(found.isEmpty, "no Cosmos tokenizer release with a record under \(root)")
        return found
    }

    private func net(_ variant: NFKMLXCosmosTokenizerVariant, _ directory: URL) throws -> NFKMLXCosmosTokenizerNet {
        try NFKMLXCosmosTokenizer.network(variant: variant, weightsURLs: [
            directory.appendingPathComponent("encoder.jit"), directory.appendingPathComponent("decoder.jit"),
        ])
    }

    func testTheHaarPatchersInvertEachOther() throws {
        try requireMLXRuntime()
        let image = MLXRandom.normal([1, 32, 48, 3], key: MLXRandom.key(1))
        let patched = NFKCosmosPatcher2D.patch(image, levels: 2)
        XCTAssertEqual(patched.shape, [1, 8, 12, 48])
        XCTAssertLessThan(abs(NFKCosmosPatcher2D.unpatch(patched, levels: 2) - image).max().item(Float.self), 1e-5)

        let clip = MLXRandom.normal([1, 9, 32, 32, 3], key: MLXRandom.key(2))
        let patchedClip = NFKCosmosPatcher3D.patch(clip, patchSize: 4, levels: 2)
        XCTAssertEqual(patchedClip.shape, [1, 3, 8, 8, 192])
        let restored = NFKCosmosPatcher3D.unpatch(patchedClip, patchSize: 4, levels: 2)
        XCTAssertEqual(restored.shape, clip.shape)
        XCTAssertLessThan(abs(restored - clip).max().item(Float.self), 1e-5)
    }

    func testFiniteScalarQuantizationIndicesRoundTrip() throws {
        try requireMLXRuntime()
        let quantizer = NFKMLXCosmosFSQ(levels: [8, 8, 8, 5, 5, 5])
        let indices = MLXArray(Array(Int32(0) ..< Int32(64000)))
        let codes = quantizer.codes(indices: indices)
        XCTAssertEqual(codes.shape, [64000, 6])
        XCTAssertEqual(quantizer.indices(codes: codes).asArray(Int32.self), Array(Int32(0) ..< Int32(64000)))
        XCTAssertEqual(quantizer.codes(quantizer.codes(indices: indices) * 0.5).shape, [64000, 6])
    }

    func testEveryVariantBuildsTheReleasedParameterTree() throws {
        try requireMLXRuntime()
        for (variant, directory, _) in try releasedVariants() {
            // `verifyShapes` inside the loader fails any built shape that differs from the release's.
            XCTAssertNoThrow(try net(variant, directory), variant.releaseName)
        }
    }

    func testEveryVariantMatchesTheReferenceSeamBySeam() throws {
        try requireMLXRuntime()
        for (variant, directory, record) in try releasedVariants() {
            try assertSeamParity(try net(variant, directory), record: record, name: variant.releaseName)
        }
    }

    /// Every seam the record holds, from the wavelet patch through the latent or tokens to the
    /// reconstruction.
    private func assertSeamParity(_ net: NFKMLXCosmosTokenizerNet, record: [String: MLXArray], name: String) throws {
        let configuration = net.configuration
        let pixels = try XCTUnwrap(record["clip"], name).expandedDimensions(axis: 0)

        let patched = configuration.isVideo
            ? NFKCosmosPatcher3D.patch(pixels, patchSize: configuration.patchSize, levels: configuration.patchLevels)
            : NFKCosmosPatcher2D.patch(pixels, levels: configuration.patchLevels)
        let encoderOutput = net.encoderOutput(pixels)
        let latent = net.encode(pixels)
        let features = net.decoderFeatures(latent)
        let reconstruction = net.decode(latent)
        eval(patched, encoderOutput, latent, features, reconstruction)

        let seams: [(String, MLXArray, String)] = [
            ("patched", patched, "patched"),
            ("encoder", encoderOutput, "encoder_out"),
        ]
            + (configuration.isDiscrete ? [] : [
                ("latent", latent, "latent"),
                ("decoder", features, "decoder_out"),
                ("reconstruction", reconstruction[0], "output"),
            ])
        for (label, mine, key) in seams {
            let reference = try XCTUnwrap(record[key], "\(name) \(key)")
            XCTAssertEqual(mine.shape.suffix(reference.ndim), reference.shape.suffix(reference.ndim),
                           "\(name) \(label)")
            XCTAssertGreaterThan(cosine(mine, reference), 0.999999999, "\(name) \(label)")
        }
        if configuration.isDiscrete {
            try assertTokenParity(net, pixels: pixels, record: record, name: name)
        }
    }

    /// A discrete tokenizer rounds each latent scalar to its level, so any difference flips a token
    /// whose scalar sits that close to a rounding boundary, and the releases put many scalars near one
    /// (DI8x8: 34 of 1,536 within 1e-3). Parity is stated where it is defined: the latent before
    /// rounding matches, every token clear of a boundary matches exactly, and the reference's own
    /// tokens decode to the reference's reconstruction. At float32 every token of every record matches.
    private func assertTokenParity(_ net: NFKMLXCosmosTokenizerNet, pixels: MLXArray,
                                   record: [String: MLXArray], name: String) throws {
        let referenceInput = try XCTUnwrap(record["quantizer_input"], "\(name) quantizer_input")
        XCTAssertGreaterThan(cosine(net.quantizerInput(pixels), referenceInput), 0.999999999, "\(name) quantizer input")

        let bounded = net.quantizer.bounded(referenceInput)
        let margin = abs(bounded - floor(bounded) - 0.5).min(axis: -1).reshaped([-1]).asArray(Float.self)
        let mine = net.tokens(pixels).reshaped([-1]).asArray(Int32.self)
        let theirs = try XCTUnwrap(record["indices"]).asType(.int32).reshaped([-1]).asArray(Int32.self)
        var clear = 0, clearAgreeing = 0
        for i in theirs.indices where margin[i] > 1e-5 {
            clear += 1
            clearAgreeing += mine[i] == theirs[i] ? 1 : 0
        }
        XCTAssertGreaterThan(clear, theirs.count / 2, "\(name) tokens clear of a boundary")
        XCTAssertEqual(clearAgreeing, clear, "\(name) tokens clear of a boundary")

        let referenceTokens = try XCTUnwrap(record["indices"]).asType(.int32)
        let features = net.decoderFeatures(net.quantizer.codes(indices: referenceTokens))
        XCTAssertGreaterThan(cosine(features, try XCTUnwrap(record["decoder_out"])), 0.999999999, "\(name) decoder")
        let decoded = net.decode(tokens: referenceTokens)
        XCTAssertGreaterThan(cosine(decoded[0], try XCTUnwrap(record["output"])), 0.999999999, "\(name) decode")
    }

    /// The factories and the download take a release's single `autoencoder.jit`. For six releases it
    /// holds the same weights as the `encoder.jit` and `decoder.jit` pair and must load to the identical
    /// network; for DI8x8, DV4x8x8, DV8x8x8, and DV8x16x16 NVIDIA published different weights in it, so
    /// it is held to its own record (`record_autoencoder.safetensors`, `run_reference.py
    /// cosmos_tokenizer --checkpoint <variant>/autoencoder.jit`).
    func testEveryReleasedAutoencoderFileMatchesItsReference() throws {
        try requireMLXRuntime()
        var compared = 0
        for (variant, directory, _) in try releasedVariants() {
            let combined = directory.appendingPathComponent("autoencoder.jit")
            guard FileManager.default.fileExists(atPath: combined.path) else { continue }
            let tokenizer = try NFKMLXCosmosTokenizer.tokenizer(variant: variant, weightsURL: combined)
            let ownRecord = directory.appendingPathComponent("record_autoencoder.safetensors")
            if FileManager.default.fileExists(atPath: ownRecord.path) {
                let record = try NFKMLXWeights.loadCheckpoint(url: ownRecord).arrays
                try assertSeamParity(tokenizer.network, record: record, name: "\(variant.releaseName) autoencoder.jit")
            } else {
                let fromPair = Dictionary(uniqueKeysWithValues: try net(variant, directory).parameters().flattened())
                let fromFile = tokenizer.network.parameters().flattened()
                XCTAssertEqual(fromFile.count, fromPair.count, variant.releaseName)
                let differing = fromFile.filter { key, value in abs(value - fromPair[key]!).max().item(Float.self) > 0 }
                XCTAssertTrue(differing.isEmpty, "\(variant.releaseName): \(differing.count) tensors differ from the pair")
            }
            compared += 1
        }
        try XCTSkipIf(compared == 0, "no autoencoder.jit beside the releases")
    }

    func testTheBackendReconstructsAnImageAtItsOwnSize() throws {
        try requireMLXRuntime()
        guard let (variant, directory, _) = try releasedVariants().first(where: { !$0.0.isVideo }) else {
            throw XCTSkip("no image variant present")
        }
        let tokenizer = try NFKMLXCosmosTokenizer.tokenizer(
            variant: variant, encoderWeightsURL: directory.appendingPathComponent("encoder.jit"),
            decoderWeightsURL: directory.appendingPathComponent("decoder.jit"))
        // A size the wavelet does not divide exercises the reference's centered padding and crop.
        let rows = (0 ..< 72).map { y in (0 ..< 100).map { x in [Float(x) / 100, Float(y) / 72, 0.5] } }
        let image = MLXArray(rows.flatMap { $0.flatMap { $0 } }, [72, 100, 3])
        let output = tokenizer.reconstruct(frames: [image])[0]
        XCTAssertEqual(output.shape, [72, 100, 3])
        XCTAssertGreaterThan(cosine(output, image), 0.98)

        let backend = NFKMLXCosmosTokenizerBackend(tokenizer: tokenizer)
        let picture = try NFKMLXImageBridge.cgImage(from: image, options: NFKMLXImageOptions())
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: picture]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage))

        let aligned = try NFKMLXImageBridge.cgImage(from: image[0 ..< 64, 0 ..< 96], options: NFKMLXImageOptions())
        let code = try tokenizer.code(forImage: aligned)
        let factor = NFKMLXCosmosTokenizerConfiguration.variant(variant).spatialCompression
        let grid = [64 / factor, 96 / factor]
        XCTAssertEqual(code.shape.map(\.intValue), variant.isDiscrete ? grid : grid + [16])
        let frames = try tokenizer.frames(for: code).map { $0 as! CGImage }
        XCTAssertEqual(frames.count, 1)
        XCTAssertEqual(frames[0].width, 96)
        XCTAssertEqual(frames[0].height, 64)
    }

    // MARK: - Customization

    private func objective() throws -> NFKMLXCosmosTokenizerObjective {
        guard let vgg = NFKMLXValidationConfig.environment["IK_VAL_VGG16"] else {
            throw XCTSkip("set IK_VAL_VGG16 to the VGG-16 ImageNet weights (timm/vgg16.tv_in1k model.safetensors)")
        }
        return try NFKMLXCosmosTokenizerObjective(vggWeightsURL: URL(fileURLWithPath: vgg))
    }

    func testTheObjectiveMatchesTheReferenceOnIdenticalTensors() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_COSMOS_TOKENIZER_LOSS"] else {
            throw XCTSkip("set IK_PARITY_COSMOS_TOKENIZER_LOSS (run_reference.py cosmos_tokenizer_loss)")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        var objective = try objective()
        objective.gramWeight = 1
        for label in ["image", "clip"] {
            let parts = objective.components(reconstruction: try XCTUnwrap(record["\(label)_reconstruction"]),
                                             target: try XCTUnwrap(record["\(label)_target"]))
            for (term, mine) in [("color", parts.color), ("lpips", parts.perceptual), ("gram", parts.gram)] {
                let reference = try XCTUnwrap(record["\(label)_\(term)"]).item(Float.self)
                XCTAssertEqual(mine.item(Float.self), reference, accuracy: abs(reference) * 1e-5, "\(label) \(term)")
            }
        }
    }

    /// A small network of each kind, so a fine-tune runs in seconds.
    private func tinyNetwork(video: Bool) -> NFKMLXCosmosTokenizerNet {
        var configuration = video
            ? NFKMLXCosmosTokenizerConfiguration(isVideo: true, isDiscrete: false, spatialCompression: 8,
                                                 temporalCompression: 4)
            : NFKMLXCosmosTokenizerConfiguration(isVideo: false, isDiscrete: true, spatialCompression: 8)
        configuration.channels = 32
        configuration.encoderChannelMultipliers = [1, 1]
        configuration.decoderChannelMultipliers = [1, 1]
        configuration.zChannels = 32
        if !video {
            configuration.latentChannels = configuration.levels.count
        } else {
            configuration.latentChannels = 4
        }
        return NFKMLXCosmosTokenizerNet(configuration: configuration)
    }

    func testAFineTuneLowersTheLossOfBothKinds() throws {
        try requireMLXRuntime()
        let objective = try objective()
        for video in [true, false] {
            let net = tinyNetwork(video: video)
            let shape = video ? [1, 5, 32, 32, 3] : [2, 32, 32, 3]
            let batch = MLX.clip(MLXRandom.normal(shape, key: MLXRandom.key(3)) * 0.5, min: -1, max: 1)
            let losses = try NFKMLXCosmosTokenizer.fineTune(
                net, examples: { _ in batch }, objective: objective,
                optimizer: AdamW(learningRate: 1e-3, biasCorrection: true), steps: 6)
            XCTAssertEqual(losses.count, 6)
            XCTAssertLessThan(losses[5], losses[0], video ? "video" : "image")
        }
    }

    func testADecoderFineTuneLeavesTheEncoderAndItsTokensUnchanged() throws {
        try requireMLXRuntime()
        let objective = try objective()
        let net = tinyNetwork(video: false)
        let batch = MLX.clip(MLXRandom.normal([2, 32, 32, 3], key: MLXRandom.key(4)) * 0.5, min: -1, max: 1)
        let tokensBefore = net.tokens(batch)
        let encoderBefore = net.encoder.parameters().flattened().map { ($0.0, $0.1 * 1) }
        let decoderBefore = net.decoder.parameters().flattened().map { ($0.0, $0.1 * 1) }
        eval(tokensBefore)
        try NFKMLXCosmosTokenizer.fineTune(net, examples: { _ in batch }, trainable: .decoder,
                                           objective: objective,
                                           optimizer: AdamW(learningRate: 1e-3, biasCorrection: true), steps: 3)
        let encoderAfter = Dictionary(uniqueKeysWithValues: net.encoder.parameters().flattened())
        for (key, value) in encoderBefore {
            XCTAssertEqual(abs(value - encoderAfter[key]!).max().item(Float.self), 0, "encoder \(key)")
        }
        let decoderAfter = Dictionary(uniqueKeysWithValues: net.decoder.parameters().flattened())
        let moved = decoderBefore.filter { key, value in abs(value - decoderAfter[key]!).max().item(Float.self) > 0 }
        XCTAssertGreaterThan(moved.count, decoderBefore.count / 2, "the decoder trains")
        XCTAssertEqual(net.tokens(batch).asArray(Int32.self), tokensBefore.asArray(Int32.self))
    }
}
