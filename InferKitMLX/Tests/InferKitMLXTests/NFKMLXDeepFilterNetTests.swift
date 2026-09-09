//
//  NFKMLXDeepFilterNetTests.swift
//  InferKitMLXTests
//
//  DeepFilterNet3 (Rikorose/DeepFilterNet, dual MIT/Apache-2.0) real-time speech denoising. The
//  module forward evaluates MLX arrays, so it skips without a Metal library for MLX (see
//  Tools/mlx-metallib.sh). The reference-parity test is gated on the released weights and the
//  recorded oracle output (`IK_VAL_DEEPFILTERNET` + `IK_PARITY_DEEPFILTERNET`), from
//  `run_reference.py deepfilternet`.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXDeepFilterNetTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private static func tone(samples: Int, hz: Float = 220, sampleRate: Float = 48000) -> [Float] {
        (0 ..< samples).map { 0.5 * sinf(2 * .pi * hz * Float($0) / sampleRate) }
    }

    /// The module keys the loader targets, after the conv-layout remap and the GRU fold.
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXDeepFilterNetFactory.makeNet()
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["enc.erb_conv0.conv.weight",
                         "enc.erb_conv1.conv.weight",
                         "enc.erb_conv1.conv_pw.weight",
                         "enc.df_conv0.conv.weight",
                         "enc.df_fc_emb.weight",
                         "enc.emb_gru.gru.0.Wx",
                         "enc.emb_gru.linear_in.weight",
                         "enc.lsnr_fc.weight",
                         "erb_dec.conv3p.conv.weight",
                         "erb_dec.convt2.conv.weight",
                         "erb_dec.emb_gru.gru.1.Wx",
                         "df_dec.df_convp.conv.weight",
                         "df_dec.df_gru.gru.1.Wx",
                         "df_dec.df_skip.weight",
                         "df_dec.df_out.weight",
                         "erb_fb", "erb_inv_fb"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    /// With random weights the full DSP + net path produces finite enhanced audio near the input length.
    func testForwardProducesFiniteEnhancedAudio() throws {
        try requireMLXRuntime()
        let net = NFKMLXDeepFilterNetFactory.makeNet()
        let samples = Self.tone(samples: 24000)
        let enhanced = NFKMLXDeepFilterNetBackend.enhance(samples, net: net)
        XCTAssertEqual(enhanced.count, samples.count, accuracy: 480,
                       "the enhanced clip is the input length up to the transform's edge handling")
        XCTAssertTrue(enhanced.allSatisfy { $0.isFinite }, "enhanced audio is finite")
    }

    func testTheBackendReturnsAnEnhancedClip() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXDeepFilterNetFactory.backend(weightsURL: nil)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("dfn-in-\(UUID()).wav")
        try NFKMLXWaveFile.write(samples: Self.tone(samples: 24000), sampleRate: 48000, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: 0.5, sampleRate: 48000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio))
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        eval(mine)
        let a = mine.asArray(Float.self), b = reference.asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    /// Localizes the first divergence: feeds the net the RECORDED features and checks each stage against
    /// the oracle seams, transposing mine (NHWC `[1,T,F,C]`) into the reference's NCHW `[C,T,F]`.
    func testSeamsAgainstTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_DEEPFILTERNET"],
              let recordPath = environment["IK_PARITY_DEEPFILTERNET"] else {
            throw XCTSkip("set IK_VAL_DEEPFILTERNET and IK_PARITY_DEEPFILTERNET to run the seam comparison")
        }
        let net = NFKMLXDeepFilterNetFactory.makeNet()
        try NFKMLXDeepFilterNetFactory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        let (t, bins, nbDF) = (record["feat_erb"]!.dim(0), record["spec"]!.dim(1), 96)
        let spec = record["spec"]!.reshaped([1, t, bins, 2])
        let featERB = record["feat_erb"]!.reshaped([1, t, 32, 1])
        let featSpec = record["feat_spec"]!.reshaped([1, t, nbDF, 2])

        func report(_ name: String, _ value: Float) {
            XCTAssertGreaterThan(value, 0.9999, "seam \(name) diverges (cosine \(value))")
        }
        // NHWC [1,T,F,C] → NCHW [C,T,F].
        func nchw(_ x: MLXArray) -> MLXArray { x[0].transposed(2, 0, 1) }

        let (emb, e0, e1, e2, e3, c0, lsnr) = net.enc(featERB: net.padFeat(featERB), featSpec: net.padFeat(featSpec))
        report("e0", cosine(nchw(e0), record["e0"]!))
        report("e1", cosine(nchw(e1), record["e1"]!))
        report("e2", cosine(nchw(e2), record["e2"]!))
        report("e3", cosine(nchw(e3), record["e3"]!))
        report("c0", cosine(nchw(c0), record["c0"]!))

        // Sub-seams of the DF→emb path: c1 (df_conv1), cemb (df_fc_emb+ReLU), the combined pre-GRU embedding.
        let enc = net.enc
        let c1 = enc.dfConv1(c0)
        report("c1", cosine(nchw(c1), record["c1"]!))
        let (b, tt) = (e3.dim(0), e3.dim(1))
        let dfFlat = c1.reshaped([b, tt, c1.dim(2) * c1.dim(3)])
        let cemb = relu(enc.dfFCEmb(dfFlat))
        report("cemb", cosine(cemb[0], record["cemb"]!))
        let erbFlat = e3.reshaped([b, tt, e3.dim(2) * e3.dim(3)])
        report("emb_in", cosine((erbFlat + cemb)[0], record["emb_in"]!))

        report("emb", cosine(emb[0], record["emb"]!))
        report("lsnr", cosine(lsnr[0], record["lsnr"]!))

        report("erb_inv_fb", cosine(net.erbInvFB, record["erb_inv_fb"]!))
        let (specE, m, coefs, _) = net(spec: spec, featERB: featERB, featSpec: featSpec)
        report("m", cosine(m[0].squeezed(axis: 2), record["m"]!))
        report("df_coefs", cosine(coefs[0], record["df_coefs"]!))
        // Isolate the final assembly: the ERB-masked spectrum, then the deep-filter + high-bin composite.
        let maskBins = matmul(m.squeezed(axis: 3), net.erbInvFB).expandedDimensions(axis: 3)
        let specM = spec * maskBins
        report("spec_m", cosine(specM[0], record["spec_m"]!))
        let refSpecE = record["spec_e"]!
        report("spec_e_low", cosine(specE[0][0..., .stride(to: 96), 0...], refSpecE[0..., .stride(to: 96), 0...]))
        report("spec_e_high", cosine(specE[0][0..., 96..., 0...], refSpecE[0..., 96..., 0...]))
        report("spec_e", cosine(specE[0], refSpecE))
    }

    /// Localizes the DSP: MY analysis features against the recorded `spec`/`feat_erb`/`feat_spec`, and MY
    /// synthesis of the recorded `spec_e` against the reference's own (an internal re-run through libdf).
    func testDSPAgainstTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_DEEPFILTERNET"],
              let recordPath = environment["IK_PARITY_DEEPFILTERNET"] else {
            throw XCTSkip("set IK_VAL_DEEPFILTERNET and IK_PARITY_DEEPFILTERNET to run the DSP comparison")
        }
        let net = NFKMLXDeepFilterNetFactory.makeNet()
        try NFKMLXDeepFilterNetFactory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let dsp = NFKMLXDeepFilterNetDSP(config: net.config, erbFB: net.erbFB)
        let waveform = record["waveform"]!.asArray(Float.self)
        let (spec, featERB, featSpec) = dsp.features(waveform)
        func report(_ name: String, _ value: Float) {
            XCTAssertGreaterThan(value, 0.9999, "DSP \(name) diverges (cosine \(value))")
        }
        report("spec", cosine(spec[0], record["spec"]!))
        report("feat_erb", cosine(featERB[0].squeezed(axis: 2), record["feat_erb"]!))
        report("feat_spec", cosine(featSpec[0], record["feat_spec"]!))
    }

    /// PARITY: the released DeepFilterNet3 against the recorded oracle — the enhanced waveform end to end
    /// (analysis → net → synthesis), reproducing the DSP the reference runs through Rust `libdf`.
    func testReferenceParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_DEEPFILTERNET"],
              let recordPath = environment["IK_PARITY_DEEPFILTERNET"] else {
            throw XCTSkip("set IK_VAL_DEEPFILTERNET (weights) and IK_PARITY_DEEPFILTERNET (oracle record)")
        }
        let net = NFKMLXDeepFilterNetFactory.makeNet()
        try NFKMLXDeepFilterNetFactory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let waveform = record["waveform"]!.asArray(Float.self)
        let enhanced = NFKMLXDeepFilterNetBackend.enhance(waveform, net: net)
        let reference = record["output"]!.asArray(Float.self)
        let count = min(enhanced.count, reference.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< count {
            dot += enhanced[i] * reference[i]; na += enhanced[i] * enhanced[i]; nb += reference[i] * reference[i]
        }
        XCTAssertGreaterThan(dot / (sqrtf(na) * sqrtf(nb) + 1e-20), 0.999,
                             "the enhanced waveform matches the reference")
    }
}
