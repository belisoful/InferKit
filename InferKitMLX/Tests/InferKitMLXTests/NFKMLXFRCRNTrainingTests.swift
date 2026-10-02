//
//  NFKMLXFRCRNTrainingTests.swift
//  InferKitMLXTests
//
//  FRCRN's training path without released weights: the loader's level scaling, the differentiable
//  inverse STFT against the inference one, the objective's clamp and SI-SNR, and a fine-tune that
//  lowers the loss and reloads through both factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXFRCRNTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_003)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Two one-second clips of a tone under noise, `[2, 16000]` each.
    private func pair() -> (noisy: MLXArray, clean: MLXArray) {
        let clean = (0 ..< 2).flatMap { row in
            (0 ..< 16_000).map { index in 0.1 * sinf(2 * .pi * Float(180 + 70 * row) * Float(index) / 16_000) }
        }
        let cleanArray = MLXArray(clean, [2, 16_000])
        return (cleanArray + MLXRandom.normal([2, 16_000], key: MLXRandom.key(3)) * 0.02, cleanArray)
    }

    func testTheLevelHelperScalesTheLoudSamplesToMinus25Decibels() {
        let samples = (0 ..< 8000).map { index -> Float in index < 4000 ? 0.5 * sinf(Float(index) * 0.05) : 0.001 }
        let scaled = NFKMLXTrainingData.speechLevelNormalized(samples).map(Double.init)
        let power = scaled.map { $0 * $0 }
        let mean = power.reduce(0, +) / Double(power.count)
        let loud = power.filter { $0 > mean }
        let level = 10 * log10(loud.reduce(0, +) / Double(loud.count))
        // The reference's 1e-6 guard on the RMS moves the level by a hundred-thousandth of a decibel.
        XCTAssertEqual(level, -25, accuracy: 1e-3)
    }

    func testTheDifferentiableSynthesisMatchesTheInferenceOne() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXFRCRNConfiguration()
        let real = MLXRandom.normal([1, 321, 49], key: MLXRandom.key(5))
        let imaginary = MLXRandom.normal([1, 321, 49], key: MLXRandom.key(6))
        let ours = NFKFRCRNSynthesis(configuration).waveform(real: real, imaginary: imaginary)
        let inference = NFKMLXFRCRNBackend.stft(configuration).inverseComplex(real: real, imaginary: imaginary)
        let length = min(ours.dim(1), inference.dim(1))
        XCTAssertEqual(ours.dim(1), 16_000)
        // Within the first and last hop the overlap-added window falls toward zero, and dividing by it
        // magnifies the two transforms' rounding; between them two frames overlap and the envelope is 1.
        let hop = configuration.hopSize
        let interior = abs(ours[0..., hop ..< (length - hop)] - inference[0..., hop ..< (length - hop)]).max().item(Float.self)
        print("PROBE frcrn synthesis: interior worst \(interior) of peak \(abs(inference).max().item(Float.self))")
        XCTAssertLessThan(interior, 1e-5 * abs(inference).max().item(Float.self))
    }

    func testTheTargetMaskClampsAsTheReferenceWritesIt() throws {
        try requireMLXRuntime()
        // A clean clip far louder than the noisy one makes the target ratio exceed 2, which becomes 1.
        let (noisy, _) = pair()
        let clean = noisy * 10
        let configuration = NFKMLXFRCRNConfiguration()
        let frames = 49, bins = 321
        let ones = MLXArray.ones([2, bins, frames]), zeros = MLXArray.zeros([2, bins, frames])
        let terms = NFKMLXFRCRNObjective().loss(noisy: noisy, clean: clean, estimate: clean, maskReal: ones,
                                                maskImaginary: zeros, configuration: configuration)
        XCTAssertLessThan(terms.mask.item(Float.self), 1e-2, "a ratio of 10 is clamped to 1, which the mask matches")
        XCTAssertLessThan(terms.scaleInvariantSNR.item(Float.self), -50, "the estimate is the clean clip")
    }

    func testFineTuningLowersTheLossAndReloadsThroughBothFactories() throws {
        try requireMLXRuntime()
        let net = try NFKMLXFRCRN.network(weightsURL: nil)
        let (noisy, clean) = pair()
        let objective = NFKMLXFRCRNObjective()
        net.train(true)
        let before = objective(net, noisy, clean).item(Float.self)
        net.train(false)
        let losses = try NFKMLXFRCRN.fineTune(net, examples: { _ in (noisy, clean) }, steps: 4)
        XCTAssertEqual(losses.count, 4)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the batch it trains on")
        XCTAssertTrue(net.modules().allSatisfy { !$0.training }, "the run restores evaluation mode")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXFRCRN.network(weightsURL: url)
        let samples = noisy[0].asArray(Float.self)
        let a = NFKMLXFRCRNBackend.enhance(samples, net: net), b = NFKMLXFRCRNBackend.enhance(samples, net: reloaded)
        XCTAssertLessThan(abs(a - b).max().item(Float.self), 1e-5)
        XCTAssertNoThrow(try NFKMLXFRCRN.backend(weightsURL: url))
    }

    func testAClipOffTheFrameGridIsCutToIt() throws {
        try requireMLXRuntime()
        let net = try NFKMLXFRCRN.network(weightsURL: nil)
        let (noisy, clean) = pair()
        let losses = try NFKMLXFRCRN.fineTune(net, examples: { _ in (noisy[0..., 0 ..< 15_000], clean[0..., 0 ..< 15_000]) },
                                              optimizer: Adam(learningRate: 1e-4), steps: 1)
        XCTAssertTrue(losses.allSatisfy(\.isFinite), "15000 samples cut to 14720, 640 + 320 · 44")
    }

    func testTheReferenceOptimizerDecaysEveryWeightButTheBiases() throws {
        try requireMLXRuntime()
        let net = try NFKMLXFRCRN.network(weightsURL: nil)
        let optimizer = NFKMLXFRCRN.referenceOptimizer()
        let zero = net.trainableParameters().mapValues { MLXArray.zeros(like: $0) }
        let before = Dictionary(uniqueKeysWithValues: net.trainableParameters().flattened().map { ($0.0, $0.1 * 1) })
        optimizer.update(model: net, gradients: zero)
        eval(net)
        let after = Dictionary(uniqueKeysWithValues: net.trainableParameters().flattened())
        // With a zero gradient only the L2 term moves a parameter: a decayed weight moves, a bias does not.
        let weight = "unet.encoders.0.conv.conv_re.weight", bias = "unet.encoders.0.conv.conv_re.bias"
        XCTAssertGreaterThan(abs(after[weight]! - before[weight]!).max().item(Float.self), 0)
        XCTAssertEqual(abs(after[bias]! - before[bias]!).max().item(Float.self), 0)
    }
}
