//
//  NFKMLXMPSENetTests.swift
//  InferKitMLXTests
//
//  MP-SENet speech enhancement (SCAFFOLD). The complex-STFT primitive, the module layout, and the
//  forward evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//  The reference-parity test is gated on the released weights and the recorded oracle output; it
//  skips until those are present (`IK_PARITY_MPSENET` + `IK_VAL_MPSENET`), the way every audio parity
//  test does. Finalize it on the M1 Max against `run_reference.py mpsenet`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXMPSENetTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private static func tone(samples: Int, hz: Float = 220, sampleRate: Float = 16000) -> [Float] {
        (0 ..< samples).map { 0.5 * sinf(2 * .pi * hz * Float($0) / sampleRate) }
    }

    // MARK: - Shared primitive

    /// The complex STFT is a shared primitive of the restoration vein, so it is validated on its own:
    /// transform then inverse of a tone recovers the interior (window-normalized overlap-add).
    func testComplexSTFTRoundTripsTheInterior() throws {
        try requireMLXRuntime()
        let stft = NFKMLXComplexSTFT(nFFT: 400, hop: 100, winLength: 400)
        let samples = Self.tone(samples: 4000)
        let signal = samples.withUnsafeBufferPointer { MLXArray($0, [1, samples.count]) }
        let (magnitude, phase) = stft.transform(signal)
        XCTAssertEqual(magnitude.dim(1), 201, "nFFT/2 + 1 bins")
        let reconstructed = stft.inverse(magnitude: magnitude, phase: phase)
        eval(reconstructed)
        let out = reconstructed[0].asArray(Float.self)
        // Compare the interior, away from the center-padding edges.
        let a = Array(samples[200 ..< 3800]), b = Array(out[200 ..< min(out.count, 3800)])
        let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let na = sqrtf(a.reduce(0) { $0 + $1 * $1 }), nb = sqrtf(b.reduce(0) { $0 + $1 * $1 })
        XCTAssertGreaterThan(dot / (na * nb), 0.999, "STFT round-trip should recover the interior")
    }

    // MARK: - Module layout

    /// The module keys the remap targets. A released checkpoint's names are translated onto these by
    /// `NFKMLXMPSENetFactory.remapReferenceKey` (finalized against `--list-keys`).
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXMPSENetFactory.makeNet()
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["dense_encoder.dense_conv_1_conv.weight",
                         "dense_encoder.dense_block.dense_block.0.conv.weight",
                         "TSTransformer.0.time_transformer.attention.in_proj_weight",
                         "TSTransformer.0.freq_transformer.ffn.gru.forward.Wx",
                         "TSTransformer.0.freq_transformer.ffn.linear.weight",
                         "mask_decoder.up.conv.weight",
                         "mask_decoder.lsigmoid.slope",
                         "phase_decoder.conv_r.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // MARK: - Forward

    /// With random weights the full path produces finite enhanced audio the length of the input. This
    /// exercises the STFT front end, the generator, and the iSTFT back end together (not parity).
    func testForwardProducesFiniteEnhancedAudioOfTheInputLength() throws {
        try requireMLXRuntime()
        let config = NFKMLXMPSENetConfiguration()
        let net = NFKMLXMPSENetFactory.makeNet(config)
        let samples = Self.tone(samples: 4000)
        let enhanced = NFKMLXMPSENetBackend.enhance(samples, net: net, config: config)
        eval(enhanced)
        let out = enhanced[0].asArray(Float.self)
        XCTAssertEqual(out.count, samples.count, accuracy: config.hopSize,
                       "the enhanced clip is the input length up to the transform's edge handling")
        XCTAssertTrue(out.allSatisfy { $0.isFinite }, "enhanced audio is finite")
    }

    func testTheBackendReturnsAnEnhancedClip() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXMPSENetFactory.backend(weightsURL: nil)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mpsenet-in-\(UUID()).wav")
        try NFKMLXWaveFile.write(samples: Self.tone(samples: 4000), sampleRate: 16000, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: 0.25, sampleRate: 16000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio))
    }

    // MARK: - Reference parity (SCAFFOLD, gated)

    /// PARITY: the released MP-SENet generator against the recorded oracle, seam by seam. Finalize on the
    /// M1 Max: point `IK_VAL_MPSENET` at the converted weights and `IK_PARITY_MPSENET` at the record from
    /// `run_reference.py mpsenet`, then compare noisy_mag/pha, the encoder, each TS block, denoised_mag/
    /// pha, and the waveform. Skips until both are set.
    func testReferenceParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_MPSENET"],
              let recordPath = environment["IK_PARITY_MPSENET"] else {
            throw XCTSkip("set IK_VAL_MPSENET (weights) and IK_PARITY_MPSENET (oracle record) to run parity")
        }
        let config = NFKMLXMPSENetConfiguration()
        let net = NFKMLXMPSENetFactory.makeNet(config)
        try NFKMLXMPSENetFactory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let waveform = record["waveform"]!.asArray(Float.self)
        let enhanced = NFKMLXMPSENetBackend.enhance(waveform, net: net, config: config)
        eval(enhanced)
        let ours = enhanced[0].asArray(Float.self)
        let reference = record["output"]!.asArray(Float.self)
        let count = min(ours.count, reference.count)
        let a = Array(ours[0 ..< count]), b = Array(reference[0 ..< count])
        let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let na = sqrtf(a.reduce(0) { $0 + $1 * $1 }), nb = sqrtf(b.reduce(0) { $0 + $1 * $1 })
        XCTAssertGreaterThan(dot / (na * nb), 0.999, "the enhanced waveform matches the reference")
    }

    /// Localizes the first divergence: feeds the net the RECORDED compressed magnitude and phase (so the
    /// STFT window is out of the comparison) and checks every stage against the oracle seams. Each seam
    /// is transposed into the reference's `[C, T, F]` (or `[F, T]`) layout before the cosine.
    func testSeamsAgainstTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_MPSENET"],
              let recordPath = environment["IK_PARITY_MPSENET"] else {
            throw XCTSkip("set IK_VAL_MPSENET and IK_PARITY_MPSENET to run the seam comparison")
        }
        let net = NFKMLXMPSENetFactory.makeNet()
        try NFKMLXMPSENetFactory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
            eval(mine)
            let a = mine.asArray(Float.self), b = reference.asArray(Float.self)
            let n = min(a.count, b.count)
            var dot: Float = 0, na: Float = 0, nb: Float = 0
            for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
            return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
        }
        func report(_ name: String, _ value: Float, threshold: Float = 0.999) {
            XCTAssertGreaterThan(value, threshold, "seam \(name) diverges (cosine \(value))")
        }
        // Phase is compared as the agreement mean(cos(Δ)); raw cosine over `atan2` angles is misleading
        // at the ±π wrap (a phase of +π and −π are identical but score as opposite).
        func phaseAgreement(_ mine: MLXArray, _ reference: MLXArray) -> Float {
            eval(mine)
            let a = mine.asArray(Float.self), b = reference.asArray(Float.self)
            let n = min(a.count, b.count)
            var sum: Float = 0
            for i in 0 ..< n { sum += cosf(a[i] - b[i]) }
            return sum / Float(n)
        }

        // Build the net input from the recorded compressed magnitude/phase, [1, bins, frames].
        let bins = record["noisy_mag"]!.dim(0), frames = record["noisy_mag"]!.dim(1)
        let magnitude = record["noisy_mag"]!.reshaped([1, bins, frames])
        let phase = record["noisy_pha"]!.reshaped([1, bins, frames])
        let magTF = magnitude.transposed(0, 2, 1).expandedDimensions(axis: 3)   // [1, T, F, 1]
        let phaTF = phase.transposed(0, 2, 1).expandedDimensions(axis: 3)
        var x = concatenated([magTF, phaTF], axis: 3)                          // [1, T, F, 2]

        // Encoder — reference seam is [C, T, F]; mine is [1, T, F, C] → [C, T, F].
        x = net.encoder(x)
        report("encoder", cosine(x[0].transposed(2, 0, 1), record["encoder"]!))
        // Each TS block.
        for (index, block) in net.blocks.enumerated() {
            x = block(x)
            report("ts\(index)", cosine(x[0].transposed(2, 0, 1), record["ts\(index)"]!))
        }
        // Decoders — reference seams are [F, T]; mine are [1, T, F, 1] → [F, T].
        let denoisedMag = (magTF * net.maskDecoder(x)).squeezed(axis: 3)[0].transposed(1, 0)
        report("denoised_mag", cosine(denoisedMag, record["denoised_mag"]!))
        let denoisedPha = net.phaseDecoder(x).squeezed(axis: 3)[0].transposed(1, 0)
        report("denoised_pha", phaseAgreement(denoisedPha, record["denoised_pha"]!))
    }
}
