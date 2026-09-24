//
//  NFKMLXNUWave2TrainingTests.swift
//  InferKitMLXTests
//
//  NU-Wave 2's full fine-tune: the logSNR schedule and time draw, the training pair, a gradient through
//  the short-time Fourier convolutions, the recipe, and the round trip through the model's own factory.
//  Parity of the objective against `NuWave2.common_step` lives in NFKMLXReferenceParityTests.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXNUWave2TrainingTests: XCTestCase {

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

    /// 4,096 samples at 48 kHz: a low tone and a 15 kHz tone above a 16 kHz clip's band.
    private func wideband() -> [Float] {
        (0 ..< 4096).map { index in
            let t = Float(index) / 48000
            return 0.5 * sinf(2 * .pi * 440 * t) + 0.3 * sinf(2 * .pi * 15000 * t)
        }
    }

    func testTheScheduleSpansTheConfiguredLogSNRRange() throws {
        try requireMLXRuntime()
        let c = NFKMLXNUWave2Configuration()
        let values = NFKMLXNUWave2Objective().logSNR(time: MLXArray([Float(0), 1]), configuration: c)
            .asArray(Float.self)
        XCTAssertEqual(values[0], c.logSNRMaximum, accuracy: 1e-3)
        // At t = 1 the tangent's argument sits just below π/2, where float32 resolves it to about 1e-3.
        XCTAssertEqual(values[1], c.logSNRMinimum, accuracy: 5e-3)
    }

    func testTheTimeDrawIsStratifiedWithinTheUnitInterval() throws {
        try requireMLXRuntime()
        let times = NFKMLXNUWave2Objective.time(count: 4).asArray(Float.self)
        XCTAssertTrue(times.allSatisfy { $0 >= 0 && $0 < 1 })
        let sorted = times.sorted()
        for (a, b) in zip(sorted, sorted.dropFirst()) {
            XCTAssertEqual(b - a, 0.25, accuracy: 1e-5)
        }
    }

    func testTheTrainingPairBandLimitsTheNarrowbandCopy() throws {
        try requireMLXRuntime()
        let pair = NFKMLXNUWave2.trainingPair(wideband: wideband(), narrowbandRate: 16000)
        XCTAssertEqual(pair.audio.shape, pair.narrowband.shape)
        XCTAssertEqual(pair.audio.dim(1) % 256, 0)
        XCTAssertEqual(abs(pair.audio).max().item(Float.self), 1, accuracy: 1e-6)
        XCTAssertEqual(pair.band.sum().item(Int32.self), 171, "the bins below 8 kHz of 513")
        // The 15 kHz tone is gone from the narrow-band copy and the 440 Hz tone stays.
        let residual = pair.audio - pair.narrowband
        let interior = 512 ..< (pair.audio.dim(1) - 512)
        let removed = sqrt(residual[0, interior].square().mean()).item(Float.self)
        let kept = sqrt(pair.narrowband[0, interior].square().mean()).item(Float.self)
        XCTAssertGreaterThan(removed, 0.15)
        XCTAssertGreaterThan(kept, 0.3)
    }

    func testAGradientReachesTheSpectralBranches() throws {
        try requireMLXRuntime()
        let net = try NFKMLXNUWave2.network(weightsURL: nil)
        let pair = NFKMLXNUWave2.trainingPair(wideband: wideband(), narrowbandRate: 16000)
        let noise = MLXRandom.normal(pair.audio.shape)
        let objective = NFKMLXNUWave2Objective()
        let lossAndGradient = valueAndGrad(model: net) { net, _ in
            [objective.loss(net, audio: pair.audio, narrowband: pair.narrowband, band: pair.band,
                            time: MLXArray([Float(0.4)]), noise: noise)]
        }
        let (_, gradients) = lossAndGradient(net, [])
        let flat = gradients.flattened()
        let spectral = flat.filter { $0.0.contains("spectral") || $0.0.contains("fu.") || $0.0.contains("fourier") }
        XCTAssertFalse(spectral.isEmpty, "the gradient tree names the spectral branch: \(flat.map(\.0).prefix(8))")
        for (_, value) in flat {
            XCTAssertTrue(value.asArray(Float.self).allSatisfy(\.isFinite))
        }
        XCTAssertGreaterThan(spectral.map { abs($0.1).max().item(Float.self) }.max() ?? 0, 0)
    }

    func testFineTuningRunsAndLeavesTheNetworkFinite() throws {
        try requireMLXRuntime()
        let net = try NFKMLXNUWave2.network(weightsURL: nil)
        let pair = NFKMLXNUWave2.trainingPair(wideband: wideband(), narrowbandRate: 16000)
        let history = try NFKMLXNUWave2.fineTune(net, examples: { _ in pair }, steps: 3)
        XCTAssertEqual(history.count, 3)
        XCTAssertTrue(history.allSatisfy(\.isFinite))
    }

    // With the time and noise fixed the objective is deterministic, so a run over it must fall.
    func testAFixedDrawLowersTheObjective() throws {
        try requireMLXRuntime()
        let net = try NFKMLXNUWave2.network(weightsURL: nil)
        let pair = NFKMLXNUWave2.trainingPair(wideband: wideband(), narrowbandRate: 16000)
        let noise = MLXRandom.normal(pair.audio.shape)
        let objective = NFKMLXNUWave2Objective()
        let history = try NFKMLXTrainer.train(
            net, optimizer: Adam(learningRate: 1e-3, biasCorrection: true), steps: 8,
            sample: { _ in pair.audio },
            loss: { net, audio in
                objective.loss(net, audio: audio, narrowband: pair.narrowband, band: pair.band,
                               time: MLXArray([Float(0.4)]), noise: noise)
            })
        XCTAssertLessThan(history.last!, history.first!)
    }

    func testAFineTunedCheckpointLoadsThroughTheFactory() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("nuwave2-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXNUWave2.network(weightsURL: nil)
        let pair = NFKMLXNUWave2.trainingPair(wideband: wideband(), narrowbandRate: 16000)
        try NFKMLXNUWave2.fineTune(net, examples: { _ in pair }, steps: 2)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXNUWave2.network(weightsURL: url)
        let noise = MLXRandom.normal(pair.audio.shape)
        let level = MLXArray([Float(0.5)])
        let expected = net(noise, narrowband: pair.narrowband, band: pair.band, level: level)
        let actual = reloaded(noise, narrowband: pair.narrowband, band: pair.band, level: level)
        XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 1e-5)
        XCTAssertTrue(try NFKMLXNUWave2.backend(weightsURL: url).isReady)
    }
}
