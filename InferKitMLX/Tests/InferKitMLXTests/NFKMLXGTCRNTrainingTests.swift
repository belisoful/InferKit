//
//  NFKMLXGTCRNTrainingTests.swift
//  InferKitMLXTests
//
//  GTCRN's full fine-tune: the differentiable synthesis the SI-SNR term needs, the objective, the
//  recipe, and the round trip through the model's own factory. Parity of the objective against the
//  reference's `HybridLoss` lives in NFKMLXReferenceParityTests.
//

import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXGTCRNTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_924)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// A quarter second of a voiced tone and the same tone under noise, at 16 kHz.
    private func pair(seed: Int = 0) -> (noisy: [Float], clean: [Float]) {
        var state = UInt64(seed + 1) &* 0x9E37_79B9_7F4A_7C15
        let clean = (0 ..< 4000).map { index -> Float in
            let t = Float(index) / 16000
            return (1 ... 4).reduce(Float(0)) { $0 + 0.3 / Float($1) * sinf(2 * .pi * 150 * Float($1) * t) }
        }
        let noisy = clean.map { sample -> Float in
            state = state &* 6_364_136_223_846_793_005 &+ 1
            return sample + 0.1 * (Float(state >> 40) / Float(1 << 24) - 0.5)
        }
        return (noisy, clean)
    }

    func testTheSynthesisMatchesTheInferenceInverse() throws {
        try requireMLXRuntime()
        let spectrum = NFKMLXGTCRNFactory.spectrogram(for: pair().noisy)
        let differentiable = NFKGTCRNSynthesis.waveform(real: spectrum.real, imaginary: spectrum.imaginary)
        let stft = NFKMLXComplexSTFT(nFFT: 512, hop: 256, window: NFKGTCRNSynthesis.window)
        let reference = stft.inverseComplex(real: spectrum.real, imaginary: spectrum.imaginary)
        XCTAssertEqual(differentiable.shape, reference.shape)
        XCTAssertLessThan(abs(differentiable - reference).max().item(Float.self), 1e-5)
    }

    func testTheSynthesisResynthesizesTheSignal() throws {
        try requireMLXRuntime()
        let samples = pair().clean
        let spectrum = NFKMLXGTCRNFactory.spectrogram(for: samples)
        let waveform = NFKGTCRNSynthesis.waveform(real: spectrum.real, imaginary: spectrum.imaginary)
        let length = waveform.dim(1)
        let original = MLXArray(Array(samples[0 ..< length])).reshaped([1, length])
        XCTAssertLessThan(abs(waveform - original).max().item(Float.self), 1e-4)
    }

    func testAGradientReachesTheSpectrogramThroughTheSynthesis() throws {
        try requireMLXRuntime()
        let spectrum = NFKMLXGTCRNFactory.spectrogram(for: pair().noisy)
        let gradient = grad { (real: MLXArray) -> MLXArray in
            NFKGTCRNSynthesis.waveform(real: real, imaginary: spectrum.imaginary).square().sum()
        }(spectrum.real)
        XCTAssertGreaterThan(abs(gradient).max().item(Float.self), 0)
    }

    func testTheObjectiveScoresACloserPredictionLower() throws {
        try requireMLXRuntime()
        let (noisy, clean) = pair()
        let target = NFKMLXGTCRNFactory.spectrogram(for: clean)
        let far = NFKMLXGTCRNFactory.spectrogram(for: noisy)
        let near = NFKMLXGTCRNFactory.spectrogram(for: zip(noisy, clean).map { 0.2 * $0 + 0.8 * $1 })
        let objective = NFKMLXGTCRNObjective()
        XCTAssertLessThan(objective.loss(predicted: near, target: target).item(Float.self),
                          objective.loss(predicted: far, target: target).item(Float.self))
        let terms = objective.components(predicted: target, target: target)
        XCTAssertEqual(terms.real.item(Float.self), 0, accuracy: 1e-10)
        XCTAssertEqual(terms.magnitude.item(Float.self), 0, accuracy: 1e-10)
    }

    func testFineTuningDrivesTheLossDown() throws {
        try requireMLXRuntime()
        let net = try NFKMLXGTCRNFactory.network(weightsURL: nil)
        let example = pair()
        let history = try NFKMLXGTCRNFactory.fineTune(net, examples: { _ in example }, steps: 20)
        XCTAssertEqual(history.count, 20)
        XCTAssertTrue(history.allSatisfy { $0.isFinite })
        let early = history.prefix(3).reduce(0, +) / 3
        let late = history.suffix(3).reduce(0, +) / 3
        XCTAssertLessThan(late, early)
    }

    func testTheRunLeavesTheBatchNormalizationsInEvaluationMode() throws {
        try requireMLXRuntime()
        let net = try NFKMLXGTCRNFactory.network(weightsURL: nil)
        let example = pair()
        try NFKMLXGTCRNFactory.fineTune(net, examples: { _ in example }, steps: 2)
        XCTAssertFalse(net.training)
    }

    func testAFineTunedCheckpointLoadsThroughTheFactory() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gtcrn-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXGTCRNFactory.network(weightsURL: nil)
        let example = pair()
        try NFKMLXGTCRNFactory.fineTune(net, examples: { _ in example }, steps: 5)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXGTCRNFactory.network(weightsURL: url)
        let input = NFKMLXGTCRNFactory.spectrogram(for: example.noisy)
        let expected = net(real: input.real, imaginary: input.imaginary)
        let actual = reloaded(real: input.real, imaginary: input.imaginary)
        XCTAssertLessThan(abs(actual.real - expected.real).max().item(Float.self), 1e-5)
        XCTAssertLessThan(abs(actual.imaginary - expected.imaginary).max().item(Float.self), 1e-5)
        XCTAssertTrue(try NFKMLXGTCRNFactory.backend(weightsURL: url).isReady)
    }
}
