//
//  NFKMLXSGMSETests.swift
//  InferKitMLXTests
//
//  SGMSE+ score-based speech dereverberation / enhancement. The OUVE scheduler is a value type and
//  its closed forms run under `swift test`; the NCSN++ net and the sampler evaluate MLX arrays, so
//  they run where MLX has a Metal library. The reference-parity test is gated on the released EMA
//  weights and the recorded oracle seam (`IK_VAL_SGMSE` + `IK_PARITY_SGMSE`, from `run_reference.py
//  sgmse`) and skips until both are present. Finalize on the M1 Max.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSGMSETests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    // MARK: OUVE scheduler (pure value type, no MLX)

    /// The forward marginal std grows with `t`, is positive at the prior, and the diffusion coefficient
    /// grows too — the shape the reverse SDE relies on.
    func testTheOUVEScheduleIsMonotonicAndPositive() {
        let scheduler = NFKMLXOUVEScheduler(NFKMLXSGMSEConfiguration())
        XCTAssertGreaterThan(scheduler.std(1), scheduler.std(0.03), "std increases toward the prior")
        XCTAssertGreaterThan(scheduler.std(1), 0, "the prior std is positive")
        XCTAssertGreaterThan(scheduler.diffusion(1), scheduler.diffusion(0.03), "diffusion increases with t")
    }

    /// The reverse schedule walks from `T = 1` down to `t_eps` in `reverseSteps` points.
    func testTheReverseScheduleSpansTheProcess() {
        let config = NFKMLXSGMSEConfiguration()
        let steps = NFKMLXOUVEScheduler(config).timesteps
        XCTAssertEqual(steps.count, config.reverseSteps)
        XCTAssertEqual(steps.first!, 1, accuracy: 1e-6)
        XCTAssertEqual(steps.last!, config.tEps, accuracy: 1e-6)
    }

    // MARK: Amplitude compression round-trip

    /// `spec_back(spec_fwd(X)) == X`, so the front-end compression is invertible.
    func testTheAmplitudeCompressionRoundTrips() throws {
        try requireMLXRuntime()
        let re = MLXArray([0.4, -1.2, 0.03, 2.5] as [Float]).reshaped([1, 2, 2])
        let im = MLXArray([-0.7, 0.9, 1.1, -0.2] as [Float]).reshaped([1, 2, 2])
        let (fRe, fIm) = NFKSGMSESpec.forward(re: re, im: im, factor: 0.15, exponent: 0.5)
        let (bRe, bIm) = NFKSGMSESpec.backward(re: fRe, im: fIm, factor: 0.15, exponent: 0.5)
        eval(bRe, bIm)
        let a = re.asArray(Float.self) + im.asArray(Float.self)
        let b = bRe.asArray(Float.self) + bIm.asArray(Float.self)
        for (x, y) in zip(a, b) { XCTAssertEqual(x, y, accuracy: 1e-4) }
    }

    // MARK: NCSN++ net (small config)

    private static func smallConfig() -> NFKMLXSGMSEConfiguration {
        NFKMLXSGMSEConfiguration(baseChannels: 8, channelMultipliers: [1, 2], residualBlocks: 2,
                                 attentionResolutions: [8], imageSize: 16)
    }

    /// The score net produces a finite `[B, F, T, 2]` score for a `[B, F, T, 4]` packed input.
    func testTheScoreNetProducesFiniteOutput() throws {
        try requireMLXRuntime()
        let net = NFKMLXNCSNppNet(Self.smallConfig())
        let packed = MLXRandom.normal([1, 16, 16, 4])
        let out = net(packed, sigmas: MLXArray([Float(0.5)]))
        eval(out)
        XCTAssertEqual(out.shape, [1, 16, 16, 2])
        XCTAssertTrue(out.reshaped([-1]).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    /// The module keys mirror the reference `all_modules.N.*` / `output_layer.*` layout.
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXNCSNppNet(Self.smallConfig())
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["all_modules.0.W",             // GaussianFourierProjection
                         "all_modules.1.weight",         // time-embedding Linear
                         "all_modules.3.weight",         // input conv
                         "output_layer.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // MARK: Backend

    /// The backend returns an enhanced clip. The front end fixes the freq axis at 256 bins, so the net
    /// must carry the full 7-resolution geometry; only the width (`baseChannels`) and the step count are
    /// reduced to keep the smoke test fast (random weights — the pipeline, not the quality).
    func testTheBackendReturnsAnEnhancedClip() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXSGMSE.backend(weightsURL: nil, seed: 0,
                                              config: NFKMLXSGMSEConfiguration(reverseSteps: 2, baseChannels: 8))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sgmse-in-\(UUID()).wav")
        let tone = (0 ..< 4000).map { 0.4 * sinf(2 * .pi * 180 * Float($0) / 16000) }
        try NFKMLXWaveFile.write(samples: tone, sampleRate: 16000, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: 0.25, sampleRate: 16000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio))
    }

    // MARK: Reference parity (gated)

    /// PARITY (gated): the released NCSN++ against the recorded net seam. Feeds the RECORDED `x_t` and `y`
    /// at the recorded `t`, so the STFT and sampler are out of the comparison, and checks the raw network
    /// output (the score is `-` this) plus the input-conv seam. Skips until the weights and record are set.
    func testReferenceNetSeamOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_SGMSE"],
              let recordPath = environment["IK_PARITY_SGMSE"] else {
            throw XCTSkip("set IK_VAL_SGMSE (EMA weights) and IK_PARITY_SGMSE (oracle record) to run parity")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        // Build the config from the recorded geometry: the released SGMSE+ variants differ (classic
        // 'ncsnpp' vs 'ncsnpp_48k'), so the oracle records nf / ch_mult / attn / progressive / window.
        func ints(_ key: String) -> [Int] { record[key]!.asArray(Int32.self).map(Int.init) }
        func float(_ key: String) -> Float { record[key]!.asArray(Float.self)[0] }
        let attnLen = ints("cfg_attn_len")[0]
        let config = NFKMLXSGMSEConfiguration(
            fftSize: ints("cfg_n_fft")[0], hopSize: ints("cfg_hop")[0],
            specFactor: float("cfg_spec_factor"), specAbsExponent: float("cfg_spec_abs_exponent"),
            baseChannels: ints("cfg_nf")[0], channelMultipliers: ints("cfg_ch_mult"),
            residualBlocks: ints("cfg_num_res_blocks")[0],
            attentionResolutions: attnLen > 0 ? Array(ints("cfg_attn").prefix(attnLen)) : [],
            fourierScale: float("cfg_fourier_scale"), imageSize: ints("cfg_image_size")[0],
            progressiveOutputSkip: ints("cfg_progressive")[0] == 1, windowPower: float("cfg_window_power"))
        let net = NFKMLXNCSNppNet(config)
        try NFKMLXSGMSE.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))

        let inXt = record["in_xt"]!                                        // [F, T, 2]
        let inY = record["in_y"]!                                          // [F, T, 2]
        let (bins, frames) = (inXt.dim(0), inXt.dim(1))
        let t = record["t"]!.asArray(Float.self)[0]
        // Packed [1, F, T, 4] = [xt.re, xt.im, y.re, y.im].
        let packed = concatenated([inXt, inY], axis: -1).reshaped([1, bins, frames, 4])

        func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
            eval(mine)
            let a = mine.reshaped([-1]).asArray(Float.self), b = reference.reshaped([-1]).asArray(Float.self)
            let n = min(a.count, b.count)
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
            return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
        }

        // Input-conv seam (all_modules.3), NHWC [1,F,T,nf] vs reference [F,T,nf].
        let convIn = (net.allModules[3] as! Conv2d)(packed)
        let convSimilarity = cosine(convIn[0], record["conv_in"]!)
        print("VALIDATION PARITY sgmse: input-conv seam cosine \(convSimilarity)")
        XCTAssertGreaterThan(convSimilarity, 0.999, "the input-conv seam diverges")

        // The full network output vs the recorded raw dnn output.
        let out = net(packed, sigmas: MLXArray([t]))
        let outputSimilarity = cosine(out[0], record["net_out"]!)
        print("VALIDATION PARITY sgmse: score-net output cosine \(outputSimilarity)")
        XCTAssertGreaterThan(outputSimilarity, 0.999, "the score-net output diverges")
    }
}
