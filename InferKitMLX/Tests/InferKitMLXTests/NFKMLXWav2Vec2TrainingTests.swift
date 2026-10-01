//
//  NFKMLXWav2Vec2TrainingTests.swift
//  InferKitMLXTests
//
//  Wav2Vec2's CTC fine-tuning path. The objective and the recipe's steps are measured against
//  `run_reference.py wav2vec2_loss` on facebook/wav2vec2-base-960h (`IK_VAL_WAV2VEC2_BASE_960H`,
//  `IK_PARITY_WAV2VEC2_LOSS`); the rest runs on a tiny configuration.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
@testable import InferKitMLX

final class NFKMLXWav2Vec2TrainingTests: XCTestCase {

    override func tearDown() {
        Memory.clearCache()
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func record() throws -> (directory: URL, arrays: [String: MLXArray]) {
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_WAV2VEC2_BASE_960H"], let path = env["IK_PARITY_WAV2VEC2_LOSS"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_WAV2VEC2_BASE_960H and IK_PARITY_WAV2VEC2_LOSS (run_reference.py wav2vec2_loss)")
        }
        return (URL(fileURLWithPath: directory), try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays)
    }

    /// Evaluated copies of every parameter. `update(parameters:)` writes into the arrays a module holds,
    /// so a plain `parameters()` capture reads the post-update values.
    static func snapshot(_ net: Module) -> [String: MLXArray] {
        let copies = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 + 0) })
        eval(Array(copies.values))
        return copies
    }

    static var tiny: NFKMLXWav2Vec2Configuration {
        var c = NFKMLXWav2Vec2Configuration()
        c.hiddenSize = 32
        c.numHiddenLayers = 2
        c.numAttentionHeads = 4
        c.intermediateSize = 64
        c.convDimensions = [16, 16, 16]
        c.convKernels = [10, 3, 3]
        c.convStrides = [5, 2, 2]
        c.positionalConvKernel = 8
        c.positionalConvGroups = 4
        c.vocabularySize = 8
        c.maskTimeProbability = 0.2
        c.maskTimeLength = 3
        return c
    }

    /// The CTC loss and its gradient with respect to the logits agree with `torch.nn.functional.ctc_loss`
    /// on identical logits: the clip's own transcript, a label set whose doubled letters block the skip,
    /// both reductions, and an impossible alignment with and without `zero_infinity`.
    func testTheObjectiveMatchesTorchCTCLoss() throws {
        try requireMLXRuntime()
        let (_, rec) = try record()
        let logits = try XCTUnwrap(rec["logits"]).expandedDimensions(axis: 0)
        var lines = [String]()
        for name in ["real", "repeat"] {
            let labels = try XCTUnwrap(rec["labels_\(name)"]).asArray(Int64.self).map(Int.init)
            for (reduction, tag) in [(NFKMLXWav2Vec2Objective.Reduction.mean, "mean"), (.sum, "sum")] {
                let objective = NFKMLXWav2Vec2Objective(reduction: reduction)
                let (values, gradients) = valueAndGrad { (x: [MLXArray]) in [objective.loss(logits: x[0], labels: [labels])] }([logits])
                let (value, gradient) = (values[0], gradients[0])
                let reference = try XCTUnwrap(rec["loss_\(name)_\(tag)"]).item(Float.self)
                let referenceGradient = try XCTUnwrap(rec["grad_\(name)_\(tag)"])
                let relative = abs(value.item(Float.self) - reference) / abs(reference)
                let similarity = NFKMLXWav2Vec2Tests.cosine(gradient[0], referenceGradient)
                let maximum = abs(gradient[0] - referenceGradient).max().item(Float.self)
                lines.append("\(name) \(tag) loss \(value.item(Float.self)) vs \(reference) (rel \(relative)), "
                             + "gradient \(similarity) (max |d| \(maximum))")
                XCTAssertLessThan(relative, 1e-5, "\(name) \(tag) loss")
                XCTAssertGreaterThan(similarity, 0.999999, "\(name) \(tag) gradient")
            }
        }
        let long = try XCTUnwrap(rec["labels_long"]).asArray(Int64.self).map(Int.init)
        XCTAssertEqual(rec["loss_long_mean"]?.item(Float.self), .infinity)
        let impossible = NFKMLXWav2Vec2Objective(reduction: .sum).loss(logits: logits, labels: [long]).item(Float.self)
        XCTAssertGreaterThan(impossible, 1e29, "an impossible alignment reads as unbounded")
        let zeroed = NFKMLXWav2Vec2Objective(reduction: .mean, zeroInfinity: true)
        let (values, gradients) = valueAndGrad { (x: [MLXArray]) in [zeroed.loss(logits: x[0], labels: [long])] }([logits])
        let (value, gradient) = (values[0], gradients[0])
        XCTAssertEqual(value.item(Float.self), 0)
        XCTAssertEqual(abs(gradient).max().item(Float.self), 0)
        print("VALIDATION PARITY wav2vec2 objective: " + lines.joined(separator: "; "))
    }

    /// Three recipe steps (AdamW 1e-4, the linear decay, clipping at 1.0, the feature encoder frozen) on
    /// the release follow transformers' own steps under the reference's recorded SpecAugment masks: each
    /// step's loss, the first step's gradients, and every step's parameter movement.
    func testThreeRecipeStepsFollowTheReference() throws {
        try requireMLXRuntime()
        let (directory, rec) = try record()
        let net = try NFKMLXWav2Vec2.network(directoryURL: directory)
        net.update(parameters: ModuleParameters.unflattened(["masked_spec_embed": try XCTUnwrap(rec["masked_spec_embed"])]))
        let input = try XCTUnwrap(rec["input_values"]).reshaped([1, -1])
        let labels = try XCTUnwrap(rec["labels_real"]).asArray(Int64.self).map(Int.init)
        let masks = try (0 ..< 3).map { try XCTUnwrap(rec["mask\($0)"]).asType(.bool) }
        let objective = NFKMLXWav2Vec2Objective()

        func ours(_ name: String) -> String {
            String(name.dropFirst(name.hasPrefix("wav2vec2.") ? "wav2vec2.".count : 0))
                .replacingOccurrences(of: "parametrizations.weight.original0", with: "weight_g")
                .replacingOccurrences(of: "parametrizations.weight.original1", with: "weight_v")
        }
        func layout(_ reference: MLXArray) -> MLXArray { reference.ndim == 3 ? reference.transposed(0, 2, 1) : reference }
        let watched = rec.keys.filter { $0.hasPrefix("step0_delta.") }.map { String($0.dropFirst("step0_delta.".count)) }.sorted()
        let start = Self.snapshot(net)

        NFKMLXWav2Vec2.apply(.encoder, to: net)
        let gradients = valueAndGrad(model: net) { (net: NFKMLXWav2Vec2Net, arrays: [MLXArray]) in
            [objective(net, arrays[0], labels: labels, timeMask: arrays[1])]
        }(net, [input, masks[0]]).1.flattened()
        let gradientMap = Dictionary(uniqueKeysWithValues: gradients.map { ($0.0, $0.1) })
        var lines = [String]()
        for name in watched {
            guard let reference = rec["step_grad.\(name)"] else {
                XCTAssertNil(gradientMap[ours(name)], "\(name) is frozen and takes no gradient")
                continue
            }
            let similarity = NFKMLXWav2Vec2Tests.cosine(try XCTUnwrap(gradientMap[ours(name)], name), layout(reference))
            lines.append("grad \(ours(name)) \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "\(name) first-step gradient")
        }

        var drawn = 0
        var deltas = [[String: MLXArray]]()
        let waveform = try XCTUnwrap(rec["waveform"]).asArray(Float.self)
        let losses = try NFKMLXWav2Vec2.fineTune(
            net, examples: { _ in (samples: waveform, labels: labels) }, trainable: .encoder, objective: objective,
            timeMasks: { _ in
                defer { drawn += 1 }
                return masks[drawn]
            },
            optimizer: nil, learningRate: 1e-4, weightDecay: 0, steps: 3, clipGradientNorm: 1.0,
            learningRateSchedule: nil, checkpoint: nil,
            observer: { _ in
                let now = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
                deltas.append(Dictionary(uniqueKeysWithValues: watched.map { ($0, now[ours($0)]! - start[ours($0)]!) }))
                return true
            })
        // Adam's first steps move near-zero gradients by nearly the full rate, so float32 order differences
        // grow over the run; the bar is the reference's own float32 drift from its float64 replay, doubled.
        let referenceLosses = try XCTUnwrap(rec["step_losses"]).asType(.float32).asArray(Float.self)
        let exactLosses = try XCTUnwrap(rec["step_losses_f64"]).asType(.float32).asArray(Float.self)
        let floor = zip(referenceLosses, exactLosses).map { abs($0 - $1) / $1 }.max() ?? 0
        for (index, (loss, exact)) in zip(losses, exactLosses).enumerated() {
            let drift = abs(loss - exact) / exact
            lines.append("loss\(index) \(loss) (reference \(referenceLosses[index]), float64 \(exact), "
                         + "drift \(drift) against the reference's \(abs(referenceLosses[index] - exact) / exact))")
            XCTAssertLessThanOrEqual(drift, 2 * floor, "step \(index) loss")
        }
        XCTAssertEqual(deltas.count, 3)
        for (index, moved) in deltas.enumerated() {
            for name in watched {
                let reference = layout(try XCTUnwrap(rec["step\(index)_delta.\(name)"]))
                if abs(reference).max().item(Float.self) == 0 {
                    XCTAssertEqual(abs(moved[name]!).max().item(Float.self), 0, "\(name) stays frozen")
                    continue
                }
                let similarity = NFKMLXWav2Vec2Tests.cosine(moved[name]!, reference)
                if index == 2 { lines.append("moved \(ours(name)) \(similarity)") }
                XCTAssertGreaterThan(similarity, 0.999, "step \(index) \(name) movement")
            }
        }
        print("VALIDATION PARITY wav2vec2 steps: " + lines.joined(separator: ", "))
    }

    /// The position convolution's input gradient at the release geometry (768 channels, 128 taps,
    /// padding 64, 16 groups) on the default device against the same gradient on the CPU. MLX's GPU
    /// backward of a convolution wider than 16 taps is wrong here (cosine 0.88 for the fused form), and
    /// the sliced form is exact. The fused form's reading is printed.
    func testThePositionConvolutionGradientMatchesTheCPU() throws {
        try requireMLXRuntime()
        let x = MLXRandom.normal([1, 173, 768], key: MLXRandom.key(11))
        let kernel = MLXRandom.normal([768, 128, 48], key: MLXRandom.key(12)) * 0.02
        let upstream = MLXRandom.normal([1, 174, 768], key: MLXRandom.key(13))
        func gradient(on device: Device, _ forward: @escaping (MLXArray) -> MLXArray) -> MLXArray {
            Device.withDefaultDevice(device) {
                let g = valueAndGrad { (inputs: [MLXArray]) in [(forward(inputs[0]) * upstream).sum()] }([x]).1[0]
                eval(g)
                return g
            }
        }
        let sliced: (MLXArray) -> MLXArray = { NFKWav2Vec2WeightNormConv.groupedConvolution($0, kernel, padding: 64, groups: 16) }
        let fused: (MLXArray) -> MLXArray = { conv1d($0, kernel, padding: 64, groups: 16) }
        let reference = gradient(on: .cpu, fused)
        let slicedSimilarity = NFKMLXWav2Vec2Tests.cosine(gradient(on: .gpu, sliced), reference)
        let fusedSimilarity = NFKMLXWav2Vec2Tests.cosine(gradient(on: .gpu, fused), reference)
        print("VALIDATION PROBE 128-tap conv1d input gradient on the GPU against the CPU: sliced \(slicedSimilarity), "
              + "fused \(fusedSimilarity)")
        XCTAssertGreaterThan(slicedSimilarity, 0.99999999)
    }

    /// SpecAugment draws the reference's span count and span length, repeats from its seed, and masks
    /// nothing at probability zero.
    func testTimeMasksFollowTheReferenceSpanRule() throws {
        try requireMLXRuntime()
        var first = NFKMLXSplitMix64(seed: 7), second = NFKMLXSplitMix64(seed: 7)
        let a = try XCTUnwrap(NFKMLXSpecAugment.timeMask(frames: 173, probability: 0.05, length: 10, minimumMasks: 2,
                                                          using: &first))
        let b = try XCTUnwrap(NFKMLXSpecAugment.timeMask(frames: 173, probability: 0.05, length: 10, minimumMasks: 2,
                                                          using: &second))
        XCTAssertEqual(a.asArray(Bool.self), b.asArray(Bool.self))
        let masked = a.asType(.int32).sum().item(Int.self)
        XCTAssertTrue((10 ... 20).contains(masked), "two spans of ten, possibly overlapping: \(masked)")
        XCTAssertNil(NFKMLXSpecAugment.timeMask(frames: 173, probability: 0, length: 10, minimumMasks: 2, using: &first))
    }

    /// A tiny fine-tune lowers the loss, moves the encoder and head, and leaves the frozen feature encoder
    /// exactly where it was; a head-only run moves the head alone.
    func testAFineTuneLowersTheLossAndHoldsTheFrozenGroups() throws {
        try requireMLXRuntime()
        MLXRandom.seed(3)
        let samples = (0 ..< 4000).map { Float(sin(Double($0) * 0.05) * 0.3 + sin(Double($0) * 0.31) * 0.2) }
        for trainable in [NFKMLXWav2Vec2Trainable.encoder, .head] {
            let net = NFKMLXWav2Vec2Net(Self.tiny)
            let before = Self.snapshot(net)
            let losses = try NFKMLXWav2Vec2.fineTune(net, examples: { _ in (samples, [1, 2, 3, 3, 4]) },
                                                     trainable: trainable, timeMasking: false,
                                                     learningRate: 3e-3, steps: 12)
            XCTAssertLessThan(losses.last!, losses.first!, "\(trainable) loss falls")
            let after = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
            func moved(_ name: String) -> Bool { abs(after[name]! - before[name]!).max().item(Float.self) > 0 }
            XCTAssertFalse(moved("feature_extractor.conv_layers.0.conv.weight"), "the feature encoder is frozen")
            XCTAssertTrue(moved("lm_head.weight"))
            XCTAssertEqual(moved("encoder.layers.0.attention.q_proj.weight"), trainable == .encoder)
            XCTAssertEqual(moved("encoder.pos_conv_embed.conv.weight_v"), trainable == .encoder)
        }
    }

    /// A network retargeted to a consumer's vocabulary saves to a directory that the `@objc` factory and
    /// the network builder both reload with the same logits.
    func testTheFineTunedDirectoryRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        MLXRandom.seed(5)
        let release = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let saved = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {
            try? FileManager.default.removeItem(at: release)
            try? FileManager.default.removeItem(at: saved)
        }
        var headless = Self.tiny
        headless.vocabularySize = nil
        try NFKMLXWav2Vec2.save(NFKMLXWav2Vec2Net(headless), tokenizer: nil, toDirectoryURL: release)

        let tokenizer = try NFKMLXWav2Vec2Tokenizer(characters: ["A", "B", "C", "D", "E"])
        let net = try NFKMLXWav2Vec2.network(directoryURL: release, vocabulary: tokenizer.characters)
        XCTAssertEqual(net.configuration.vocabularySize, 8)
        let samples = (0 ..< 3200).map { Float(sin(Double($0) * 0.07)) }
        try NFKMLXWav2Vec2.fineTune(net, examples: { _ in (samples, tokenizer.labels(for: "AB CE")) },
                                    learningRate: 1e-3, steps: 2)
        try NFKMLXWav2Vec2.save(net, tokenizer: tokenizer, toDirectoryURL: saved)

        let input = NFKMLXWav2Vec2Processor.inputValues(samples, normalize: true)
        let expected = try XCTUnwrap(net.logits(input))
        let reloaded = try NFKMLXWav2Vec2.network(directoryURL: saved)
        XCTAssertEqual(abs(try XCTUnwrap(reloaded.logits(input)) - expected).max().item(Float.self), 0, accuracy: 1e-6)

        let backend = try NFKMLXWav2Vec2.backend(directoryURL: saved)
        XCTAssertTrue(backend.transcribes)
        let wav = NFKMLXWaveFile.data(samples: samples, sampleRate: 16000)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wav]))
        XCTAssertNotNil(result.output(forKey: NFKOutputText) as? String)
    }
}
