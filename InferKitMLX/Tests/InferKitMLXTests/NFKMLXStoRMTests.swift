//
//  NFKMLXStoRMTests.swift
//  InferKitMLXTests
//
//  StoRM stochastic-regeneration speech enhancement / dereverberation (a few-step follow-on on SGMSE+).
//  The nets evaluate MLX arrays, so they run under `xcodebuild test`. The reference-parity test is gated
//  on the released EMA weights and the recorded oracle seams (IK_VAL_STORM + IK_PARITY_STORM, from
//  `run_reference.py storm`) and skips until both are present.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXStoRMTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private static func smallConfig(condition: NFKMLXStoRMCondition = .both) -> NFKMLXStoRMConfiguration {
        let base = NFKMLXSGMSEConfiguration(baseChannels: 8, channelMultipliers: [1, 2], residualBlocks: 2,
                                            attentionResolutions: [8], imageSize: 16)
        return NFKMLXStoRMConfiguration(base: base, condition: condition)
    }

    /// The two networks derive the right channel counts and modes from the configuration.
    func testTheDerivedConfigurationsMatchTheRoles() {
        let config = NFKMLXStoRMConfiguration(condition: .both)
        XCTAssertEqual(config.denoiserConfiguration.inputChannels, 2)
        XCTAssertFalse(config.denoiserConfiguration.conditional)
        XCTAssertFalse(config.denoiserConfiguration.scaleBySigma)
        XCTAssertEqual(config.scoreConfiguration.inputChannels, 6)     // both → x + y + y_denoised
        XCTAssertTrue(config.scoreConfiguration.conditional)
        XCTAssertEqual(NFKMLXStoRMConfiguration(condition: .noisy).scoreConfiguration.inputChannels, 4)
    }

    /// The predictor produces a finite denoised spectrogram, and the score net a finite score.
    func testTheNetworksProduceFiniteOutput() throws {
        try requireMLXRuntime()
        let net = NFKMLXStoRMNet(Self.smallConfig())
        let denoised = net.denoise(MLXRandom.normal([1, 16, 16, 2]))
        eval(denoised)
        XCTAssertEqual(denoised.shape, [1, 16, 16, 2])
        let score = net.score(MLXRandom.normal([1, 16, 16, 6]), sigmas: MLXArray([Float(0.5)]))
        eval(score)
        XCTAssertEqual(score.shape, [1, 16, 16, 2])
        XCTAssertTrue(score.reshaped([-1]).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    /// The keys mirror the reference `denoiser_net.*` / `score_net.*` layout.
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXStoRMNet(Self.smallConfig()).parameters().flattened().map(\.0))
        for expected in ["denoiser_net.all_modules.0.W", "denoiser_net.output_layer.weight",
                         "score_net.all_modules.0.W", "score_net.output_layer.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    /// The backend returns an enhanced clip (small full-geometry config, random weights — the pipeline,
    /// not the quality).
    func testTheBackendReturnsAnEnhancedClip() throws {
        try requireMLXRuntime()
        let base = NFKMLXSGMSEConfiguration(reverseSteps: 2, baseChannels: 8)
        let backend = try NFKMLXStoRM.backend(weightsURL: nil, seed: 0,
                                              config: NFKMLXStoRMConfiguration(base: base, condition: .both))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("storm-in-\(UUID()).wav")
        let tone = (0 ..< 4000).map { 0.4 * sinf(2 * .pi * 180 * Float($0) / 16000) }
        try NFKMLXWaveFile.write(samples: tone, sampleRate: 16000, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: 0.25, sampleRate: 16000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio))
    }

    /// PARITY (gated): the two StoRM networks against the recorded denoiser and score seams at a tiny
    /// random configuration (`run_reference.py storm`). The oracle saves both networks' weights into the
    /// record under `w::denoiser_net.*` / `w::score_net.*`, so no separate weights file is needed — this
    /// validates the discriminative-predictor and conditioned-score architecture (the NCSN++ backbone is
    /// already at released-weight parity via SGMSE+). Skips until the record is set.
    func testReferenceSeamsAtATinyConfiguration() throws {
        try requireMLXRuntime()
        guard let recordPath = ProcessInfo.processInfo.environment["IK_PARITY_STORM"] else {
            throw XCTSkip("set IK_PARITY_STORM (oracle record from run_reference.py storm) to run parity")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        func ints(_ k: String) -> [Int] { record[k]!.asArray(Int32.self).map(Int.init) }
        func flt(_ k: String) -> Float { record[k]!.asArray(Float.self)[0] }
        let attnLen = ints("cfg_attn_len")[0]
        let base = NFKMLXSGMSEConfiguration(
            fftSize: ints("cfg_n_fft")[0], hopSize: ints("cfg_hop")[0],
            specFactor: flt("cfg_spec_factor"), specAbsExponent: flt("cfg_spec_abs_exponent"),
            baseChannels: ints("cfg_nf")[0], channelMultipliers: ints("cfg_ch_mult"),
            residualBlocks: ints("cfg_num_res_blocks")[0],
            attentionResolutions: attnLen > 0 ? Array(ints("cfg_attn").prefix(attnLen)) : [],
            fourierScale: flt("cfg_fourier_scale"), imageSize: ints("cfg_image_size")[0],
            progressiveOutputSkip: ints("cfg_progressive")[0] == 1, windowPower: flt("cfg_window_power"))
        let condition = [NFKMLXStoRMCondition.noisy, .postDenoiser, .both][ints("cfg_condition")[0]]
        let config = NFKMLXStoRMConfiguration(base: base, condition: condition)
        let net = NFKMLXStoRMNet(config)

        // The oracle saved both networks' weights into the record under `w::…` (PyTorch layout); load them.
        let weights = record.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = String(key.dropFirst(3))
            return value.ndim == 4 ? (name, value.transposed(0, 2, 3, 1)) : (name, value)
        }
        try NFKMLXWeights.apply(weights, to: net)

        let inY = record["in_y"]!, inXt = record["in_xt"]!, denoiserOut = record["denoiser_out"]!
        let (bins, frames) = (inY.dim(0), inY.dim(1))
        let t = record["t"]!.asArray(Float.self)[0]

        func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Double {
            eval(mine)
            let a = mine.reshaped([-1]).asArray(Float.self), b = reference.reshaped([-1]).asArray(Float.self)
            let n = min(a.count, b.count)
            var d = 0.0, na = 0.0, nb = 0.0
            for i in 0 ..< n { d += Double(a[i]) * Double(b[i]); na += Double(a[i]) * Double(a[i]); nb += Double(b[i]) * Double(b[i]) }
            return d / (na.squareRoot() * nb.squareRoot() + 1e-20)
        }

        // Denoiser seam: y → y_denoised.
        let denoised = net.denoise(inY.reshaped([1, bins, frames, 2]))
        XCTAssertGreaterThan(cosine(denoised[0], denoiserOut), 0.999, "the denoiser seam diverges")

        // Score seam: [x_t, *conditioning] → raw score-net output.
        var parts = [inXt]
        switch condition {
        case .noisy: parts.append(inY)
        case .postDenoiser: parts.append(denoiserOut)
        case .both: parts.append(inY); parts.append(denoiserOut)
        }
        let packed = concatenated(parts, axis: -1).reshaped([1, bins, frames, config.scoreInputChannels])
        let score = net.score(packed, sigmas: MLXArray([t]))
        XCTAssertGreaterThan(cosine(score[0], record["score_out"]!), 0.999, "the score seam diverges")
    }
}
