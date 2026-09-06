//
//  NFKMLXGTCRNTests.swift
//  InferKitMLXTests
//
//  GTCRN real-time speech enhancement (SCAFFOLD). The module layout and forward evaluate MLX arrays, so
//  they skip under `swift test` and run under `xcodebuild test`. The reference-parity test is gated on
//  the released weights and the recorded oracle output; it skips until those are present
//  (`IK_PARITY_GTCRN` + `IK_VAL_GTCRN`). Finalize it on the M1 Max against `run_reference.py gtcrn`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXGTCRNTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private static func tone(samples: Int, hz: Float = 220, sampleRate: Float = 16000) -> [Float] {
        (0 ..< samples).map { 0.5 * sinf(2 * .pi * hz * Float($0) / sampleRate) }
    }

    /// The module keys the loader targets (mirror the released names; the loader only folds GRUs and
    /// transposes convs).
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXGTCRNFactory.makeNet()
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["erb.erb_fc.weight",
                         "erb.ierb_fc.weight",
                         "encoder.en_convs.0.conv.weight",
                         "encoder.en_convs.2.point_conv1.weight",
                         "encoder.en_convs.2.tra.att_gru.Wx",
                         "dpgrnn1.intra_rnn.rnn1.forward.Wx",
                         "dpgrnn1.inter_rnn.rnn1.Wx",
                         "dpgrnn1.intra_ln.weight",
                         "decoder.de_convs.4.conv.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    /// With random weights the full path produces finite enhanced audio near the input length.
    func testForwardProducesFiniteEnhancedAudio() throws {
        try requireMLXRuntime()
        let config = NFKMLXGTCRNConfiguration()
        let net = NFKMLXGTCRNFactory.makeNet(config)
        let samples = Self.tone(samples: 8000)
        let enhanced = NFKMLXGTCRNBackend.enhance(samples, net: net, config: config)
        eval(enhanced)
        let out = enhanced[0].asArray(Float.self)
        XCTAssertEqual(out.count, samples.count, accuracy: config.hopSize,
                       "the enhanced clip is the input length up to the transform's edge handling")
        XCTAssertTrue(out.allSatisfy { $0.isFinite }, "enhanced audio is finite")
    }

    func testTheBackendReturnsAnEnhancedClip() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXGTCRNFactory.backend(weightsURL: nil)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gtcrn-in-\(UUID()).wav")
        try NFKMLXWaveFile.write(samples: Self.tone(samples: 8000), sampleRate: 16000, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: 0.5, sampleRate: 16000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio))
    }

    /// PARITY (SCAFFOLD, gated): the released GTCRN against the recorded oracle. Finalize on the M1 Max —
    /// set `IK_VAL_GTCRN` (converted weights) and `IK_PARITY_GTCRN` (record from `run_reference.py
    /// gtcrn`), then compare the encoder/dpgrnn/decoder seams and the waveform. Skips until both are set.
    func testReferenceParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let environment = ProcessInfo.processInfo.environment
        guard let weightsPath = environment["IK_VAL_GTCRN"],
              let recordPath = environment["IK_PARITY_GTCRN"] else {
            throw XCTSkip("set IK_VAL_GTCRN (weights) and IK_PARITY_GTCRN (oracle record) to run parity")
        }
        let config = NFKMLXGTCRNConfiguration()
        let net = NFKMLXGTCRNFactory.makeNet(config)
        try NFKMLXGTCRNFactory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let waveform = record["waveform"]!.asArray(Float.self)
        let enhanced = NFKMLXGTCRNBackend.enhance(waveform, net: net, config: config)
        eval(enhanced)
        let ours = enhanced[0].asArray(Float.self)
        let reference = record["output"]!.asArray(Float.self)
        let count = min(ours.count, reference.count)
        let a = Array(ours[0 ..< count]), b = Array(reference[0 ..< count])
        let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let na = sqrtf(a.reduce(0) { $0 + $1 * $1 }), nb = sqrtf(b.reduce(0) { $0 + $1 * $1 })
        XCTAssertGreaterThan(dot / (na * nb), 0.999, "the enhanced waveform matches the reference")
    }

    /// Localizes the first divergence: feeds the net the RECORDED input spectrogram (STFT out of the
    /// comparison) and checks each stage against the oracle seams, transposing mine (NHWC `[1,T,F,C]`)
    /// into the reference's NCHW `[1,C,T,F]` before the cosine.
    func testSeamsAgainstTheReference() throws {
        try requireMLXRuntime()
        let environment = ProcessInfo.processInfo.environment
        guard let weightsPath = environment["IK_VAL_GTCRN"],
              let recordPath = environment["IK_PARITY_GTCRN"] else {
            throw XCTSkip("set IK_VAL_GTCRN and IK_PARITY_GTCRN to run the seam comparison")
        }
        let net = NFKMLXGTCRNFactory.makeNet()
        try NFKMLXGTCRNFactory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
            eval(mine)
            let a = mine.asArray(Float.self), b = reference.asArray(Float.self)
            let n = min(a.count, b.count)
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
            return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
        }
        func report(_ name: String, _ value: Float) {
            XCTAssertGreaterThan(value, 0.999, "seam \(name) diverges (cosine \(value))")
        }

        // Recorded input spectrogram [F, T, 2] → real/imag [1, F, T].
        let inSpec = record["in_spec"]!                                    // [F, T, 2]
        let (bins, frames) = (inSpec.dim(0), inSpec.dim(1))
        let real = inSpec[0..., 0..., 0].reshaped([1, bins, frames])
        let imaginary = inSpec[0..., 0..., 1].reshaped([1, bins, frames])

        // Replicate the net's forward with per-stage capture (NHWC), comparing in NCHW.
        let realTF = real.transposed(0, 2, 1).expandedDimensions(axis: 3)   // [1, T, F, 1]
        let imagTF = imaginary.transposed(0, 2, 1).expandedDimensions(axis: 3)
        let magTF = sqrt(realTF * realTF + imagTF * imagTF)
        var h = concatenated([magTF, realTF, imagTF], axis: 3)             // [1, T, 257, 3]
        h = net.erb.bandMerge(h)
        h = NFKGTCRNSFE.apply(h)
        let (bottleneck, skips) = net.encoder(h)
        report("encoder", cosine(bottleneck[0].transposed(2, 0, 1), record["encoder"]![0]))
        // Each encoder skip.
        for (index, skip) in skips.enumerated() {
            report("en\(index)", cosine(skip[0].transposed(2, 0, 1), record["en\(index)"]![0]))
        }
        // The two dual-path RNN blocks.
        let g1 = net.dpgrnn1(bottleneck)
        report("dpgrnn1", cosine(g1[0].transposed(2, 0, 1), record["dpgrnn1"]![0]))
        let g2 = net.dpgrnn2(g1)
        report("dpgrnn2", cosine(g2[0].transposed(2, 0, 1), record["dpgrnn2"]![0]))
        // The decoder block by block, over the reversed skips.
        var hh = g2
        for (index, module) in net.decoder.deConvs.enumerated() {
            let input = hh + skips[skips.count - 1 - index]
            hh = (module as? NFKGTCRNConvBlock)?.callAsFunction(input) ?? (module as! NFKGTCRNGTConvBlock)(input)
            report("de\(index)", cosine(hh[0].transposed(2, 0, 1), record["de\(index)"]![0]))
        }
    }
}
