//
//  NFKMLXMossFormer2Tests.swift
//  InferKitMLXTests
//
//  MossFormer2 SE 48K speech enhancement (SCAFFOLD). The module layout, the Kaldi-fbank front end,
//  and the forward evaluate MLX arrays, so they skip without a Metal library for MLX (see
//  Tools/mlx-metallib.sh). The reference-parity tests are gated on the released weights and the
//  recorded oracle output; they skip until `IK_VAL_MOSSFORMER2_SE` (weights) and
//  `IK_PARITY_MOSSFORMER2_SE` (the record from `run_reference.py mossformer2_se`, dither=0) are
//  set. Feeding the recorded 180-dim feature isolates the backbone from the Kaldi-fbank sub-port;
//  the fbank is validated separately against the record's `feature`. Finalize on the M1 Max.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXMossFormer2Tests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private static func tone(samples: Int, hz: Float = 220, sampleRate: Float = 48000) -> [Float] {
        (0 ..< samples).map { 0.5 * sinf(2 * .pi * hz * Float($0) / sampleRate) }
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        eval(mine)
        let a = mine.asArray(Float.self), b = reference.asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    /// Accumulates in double precision, which a clip of a million samples needs.
    private static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(a.count, b.count) {
            dot += Double(a[i]) * Double(b[i]); na += Double(a[i]) * Double(a[i]); nb += Double(b[i]) * Double(b[i])
        }
        return dot / ((na * nb).squareRoot() + 1e-300)
    }

    /// `‖a − b‖ / ‖b‖`, which a level error moves and a cosine does not.
    private static func relativeError(_ a: [Float], _ b: [Float]) -> Double {
        var difference = 0.0, norm = 0.0
        for i in 0 ..< min(a.count, b.count) {
            difference += pow(Double(a[i]) - Double(b[i]), 2); norm += Double(b[i]) * Double(b[i])
        }
        return (difference / (norm + 1e-300)).squareRoot()
    }

    // MARK: - Module layout

    /// The module keys the remap targets; a released checkpoint's `mossformer.*` names are translated
    /// onto these by `NFKMLXMossFormer2Factory.remapReferenceKey`.
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXMossFormer2Factory.makeNet()
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["conv1d_encoder.weight",
                         "pos_enc.scale",
                         "mdl.intra_mdl.mossformerM.layers.0.to_hidden.linear.weight",
                         "mdl.intra_mdl.mossformerM.layers.0.qk_offset_scale.gamma",
                         "mdl.intra_mdl.mossformerM.layers.0.to_hidden.norm.g",
                         "mdl.intra_mdl.mossformerM.fsmn.0.gated_fsmn.fsmn.conv1.weight",
                         "mdl.intra_mdl.norm.weight",
                         "output.weight",
                         "conv1_decoder.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // MARK: - Forward (random weights)

    /// The Kaldi-fbank front end produces the 180-dim feature; the MaskNet then a finite 961 mask.
    func testFrontEndAndMaskShapes() throws {
        try requireMLXRuntime()
        let config = NFKMLXMossFormer2Configuration()
        let feature = NFKMLXKaldiFbank.features(samples: Self.tone(samples: 24000), config: config)
        XCTAssertEqual(feature.dim(2), 180, "fbank + Δ + ΔΔ")
        let net = NFKMLXMossFormer2Factory.makeNet(config)
        let mask = net(feature)
        eval(mask)
        XCTAssertEqual(mask.dim(2), config.outChannelsFinal, "961-bin mask")
        XCTAssertTrue(mask.asArray(Float.self).allSatisfy { $0.isFinite && $0 >= 0 }, "mask is finite and non-negative")
    }

    func testTheBackendReturnsAnEnhancedClip() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXMossFormer2Factory.backend(weightsURL: nil)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("moss-in-\(UUID()).wav")
        try NFKMLXWaveFile.write(samples: Self.tone(samples: 24000), sampleRate: 48000, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: 0.5, sampleRate: 48000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio))
    }

    // MARK: - Long-clip decoding

    /// The configuration carries `config/inference/MossFormer2_SE_48K.yaml`'s decode values.
    func testDecodeGridFollowsTheReferenceConfiguration() {
        let config = NFKMLXMossFormer2Configuration()
        XCTAssertEqual(config.oneTimeDecodeSeconds, 20)
        XCTAssertEqual(config.decodeWindow, 192_000, "4 s at 48 kHz")
        XCTAssertEqual(config.decodeStride, 144_000, "0.75 of the window")
    }

    /// ClearerVoice's window plan through an identity window, on a grid of a 400-sample window, a
    /// 300-sample stride, and a 50-sample give-up length.
    func testWindowedDecodeStitchesLikeTheReference() {
        XCTAssertEqual(NFKMLXClearerVoiceDecoding.padding(count: 300, window: 400, stride: 300), 100)
        XCTAssertEqual(NFKMLXClearerVoiceDecoding.padding(count: 500, window: 400, stride: 300), 200)
        XCTAssertEqual(NFKMLXClearerVoiceDecoding.padding(count: 2550, window: 400, stride: 300), 450)

        let offGrid = (0 ..< 2550).map { Float($0 + 1) }
        var windows = 0
        let whole = NFKMLXClearerVoiceDecoding.stitched(offGrid, window: 400, stride: 300) { segment in
            windows += 1
            XCTAssertEqual(segment.count, 400)
            return segment
        }
        XCTAssertEqual(windows, 9, "the 3000-sample padded clip holds windows at 0, 300, …, 2400")
        XCTAssertEqual(whole, offGrid, "an off-grid clip is covered end to end")

        // (2500 − 400) is a multiple of the stride, so the decoder pads nothing and no window writes
        // the last give-up length.
        let onGrid = (0 ..< 2500).map { Float($0 + 1) }
        let trimmed = NFKMLXClearerVoiceDecoding.stitched(onGrid, window: 400, stride: 300) { $0 }
        XCTAssertEqual(Array(trimmed.prefix(2450)), Array(onGrid.prefix(2450)))
        XCTAssertEqual(Array(trimmed.suffix(50)), [Float](repeating: 0, count: 50))
    }

    // MARK: - Reference parity (SCAFFOLD, gated)

    /// The released MaskNet against the recorded oracle. Feeds the RECORDED 180-dim feature so the
    /// Kaldi-fbank front end is out of the comparison, and checks the encoder seam and the final mask.
    func testSeamsAgainstTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_MOSSFORMER2_SE"],
              let recordPath = environment["IK_PARITY_MOSSFORMER2_SE"] else {
            throw XCTSkip("set IK_VAL_MOSSFORMER2_SE (weights) and IK_PARITY_MOSSFORMER2_SE (oracle record) to run parity")
        }
        let net = NFKMLXMossFormer2Factory.makeNet()
        try NFKMLXMossFormer2Factory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        // Recorded feature [S, 180] → [1, S, 180].
        let feature = record["feature"]!.expandedDimensions(axis: 0)
        // Encoder seam: reference conv1d_encoder output is [512, S]; mine is [1, S, 512] → [512, S].
        let encoded = net.encoder(net.norm(feature))
        // Measured on the released weights (M1, float32): encoder 0.99999994.
        XCTAssertGreaterThan(cosine(encoded[0].transposed(1, 0), record["encoder"]!), 0.999, "encoder seam")

        // Block seams: reproduce the block-stack input (pos-added encoder output) and capture the first
        // and last FLASH-layer outputs, which the oracle hooked. Reference block outputs are [S, 512].
        let posAdded = encoded + net.posEnc.table(encoded.dim(1)).reshaped([1, encoded.dim(1), encoded.dim(2)])
        let layers = net.mdl.intra.block.layers
        let fsmn = net.mdl.intra.block.fsmn
        var h = posAdded
        for i in 0 ..< layers.count {
            let flashOut = layers[i](h)
            // Measured on the released weights: block 0 0.99999994, block last 1.0.
            if i == 0 { XCTAssertGreaterThan(cosine(flashOut[0], record["block0"]!), 0.999, "FLASH block 0") }
            if i == layers.count - 1 { XCTAssertGreaterThan(cosine(flashOut[0], record["block_last"]!), 0.999, "FLASH block last") }
            h = fsmn[i](flashOut)
        }

        // Final mask: the SE MaskNet returns [B, S, 961] (x[0].transpose(1,2) in the reference), so
        // mine [1, S, 961] → [S, 961] compares directly to the recorded [S, 961]. Measured 1.0.
        XCTAssertGreaterThan(cosine(net(feature)[0], record["mask"]!), 0.999, "961-bin mask")
    }

    /// The Kaldi-fbank front end against the reference's recorded 180-dim `feature` (fbank + Δ + ΔΔ),
    /// isolated from the backbone, on the clip scaled by 32768 as the reference's decoder scales it.
    func testKaldiFbankMatchesTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let recordPath = environment["IK_PARITY_MOSSFORMER2_SE"] else {
            throw XCTSkip("set IK_PARITY_MOSSFORMER2_SE (oracle record) to run the fbank comparison")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let waveform = record["waveform"]!.asArray(Float.self)
        let feature = NFKMLXKaldiFbank.features(samples: waveform.map { $0 * NFKMLXMossFormer2Backend.waveScale },
                                                config: NFKMLXMossFormer2Configuration())
        // Reference feature is [S, 180]; mine is [1, S, 180]. Compare the fbank band, Δ, and ΔΔ.
        let similarity = cosine(feature[0], record["feature"]!)
        print("VALIDATION PARITY mossformer2-se: fbank \(similarity), worst \(abs(feature[0] - record["feature"]!).max().item(Float.self))")
        XCTAssertGreaterThan(similarity, 0.999, "Kaldi fbank + deltas")
    }

    /// End to end on the released weights: the recorded waveform through the full enhance path
    /// (fbank → mask → masked iSTFT) against the reference's enhanced waveform.
    func testEnhancedWaveformMatchesTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_MOSSFORMER2_SE"],
              let recordPath = environment["IK_PARITY_MOSSFORMER2_SE"] else {
            throw XCTSkip("set IK_VAL_MOSSFORMER2_SE and IK_PARITY_MOSSFORMER2_SE to run the waveform comparison")
        }
        let config = NFKMLXMossFormer2Configuration()
        let net = NFKMLXMossFormer2Factory.makeNet(config)
        try NFKMLXMossFormer2Factory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let waveform = record["waveform"]!.asArray(Float.self)
        let enhanced = NFKMLXMossFormer2Backend.enhance(waveform, net: net, config: config)
        let reference = record["output"]!.asArray(Float.self)
        let n = min(enhanced.count, reference.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += enhanced[i] * reference[i]; na += enhanced[i] * enhanced[i]; nb += reference[i] * reference[i] }
        let similarity = dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
        print("VALIDATION PARITY mossformer2-se: enhanced waveform \(similarity)")
        XCTAssertGreaterThan(similarity, 0.999, "enhanced waveform matches the reference")
    }

    /// A 22.5 s clip, past the decoder's 20 s one-pass limit, through `enhance` against the reference's
    /// own `decode_one_audio_mossformer2_se_48k`. The reference's one-pass decode of the same clip is
    /// the control: the port must sit far closer to the windowed reference than to it.
    func testLongClipDecodesInWindowsLikeTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_MOSSFORMER2_SE"],
              let recordPath = environment["IK_PARITY_MOSSFORMER2_SE"] else {
            throw XCTSkip("set IK_VAL_MOSSFORMER2_SE and IK_PARITY_MOSSFORMER2_SE to run the long-clip comparison")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        guard let waveform = record["long_waveform"], let windowed = record["long_output"],
              let onePass = record["long_one_pass_output"] else {
            throw XCTSkip("the record predates the long-clip case; re-record run_reference.py mossformer2_se")
        }
        let config = NFKMLXMossFormer2Configuration()
        let net = NFKMLXMossFormer2Factory.makeNet(config)
        try NFKMLXMossFormer2Factory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let samples = waveform.asArray(Float.self)
        XCTAssertGreaterThan(Double(samples.count), Double(config.sampleRate) * config.oneTimeDecodeSeconds)

        let enhanced = NFKMLXMossFormer2Backend.enhance(samples, net: net, config: config)
        let reference = windowed.asArray(Float.self)
        XCTAssertEqual(enhanced.count, reference.count)
        let similarity = Self.cosine(enhanced, reference)
        let control = Self.cosine(enhanced, onePass.asArray(Float.self))
        print("VALIDATION PARITY mossformer2-se: long clip windowed \(similarity), against the one-pass reference \(control)")
        XCTAssertGreaterThan(similarity, 0.999, "the windowed decode matches the reference")
        XCTAssertLessThan(1 - similarity, (1 - control) / 10, "the port follows the windowed decode, not the one-pass one")
    }

    /// The backend's whole path against the reference's `inference.py`: the reader's `audio_norm`, the
    /// decoder, and the output scaled back by the factor `audio_norm` returned, on the one-second clip
    /// and the 22.5 s one. The reference decoder's output on the clip at its own level is the control.
    func testReaderLevelNormalizationMatchesTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_MOSSFORMER2_SE"],
              let recordPath = environment["IK_PARITY_MOSSFORMER2_SE"] else {
            throw XCTSkip("set IK_VAL_MOSSFORMER2_SE and IK_PARITY_MOSSFORMER2_SE to run the reader comparison")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        guard let decoderOutput = record["decoder_output"], let readerOutput = record["reader_output"],
              let longWaveform = record["long_waveform"], let longReaderOutput = record["long_reader_output"] else {
            throw XCTSkip("the record predates the reader case; re-record run_reference.py mossformer2_se")
        }
        // The hand-built short-clip pipeline and the reference's own decoder agree.
        XCTAssertLessThan(Self.relativeError(decoderOutput.asArray(Float.self), record["output"]!.asArray(Float.self)), 1e-5)

        let config = NFKMLXMossFormer2Configuration()
        let net = NFKMLXMossFormer2Factory.makeNet(config)
        try NFKMLXMossFormer2Factory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let cases = [("one-second", record["waveform"]!, readerOutput), ("22.5 s", longWaveform, longReaderOutput)]
        for (name, waveform, reference) in cases {
            let restored = NFKMLXMossFormer2Backend.restore(waveform.asArray(Float.self), sampleRate: 48000, net: net, config: config)
            let expected = reference.asArray(Float.self)
            XCTAssertEqual(restored.count, expected.count)
            let error = Self.relativeError(restored, expected)
            print("VALIDATION PARITY mossformer2-se: reader \(name) clip relative error \(error), cosine \(Self.cosine(restored, expected))")
            XCTAssertLessThan(error, 1e-3, "the \(name) clip matches the reference reader")
        }
        let control = Self.relativeError(decoderOutput.asArray(Float.self), readerOutput.asArray(Float.self))
        let port = Self.relativeError(NFKMLXMossFormer2Backend.restore(record["waveform"]!.asArray(Float.self), sampleRate: 48000,
                                                                       net: net, config: config),
                                      readerOutput.asArray(Float.self))
        print("VALIDATION PARITY mossformer2-se: reader control (decoder at the clip's own level) relative error \(control)")
        XCTAssertLessThan(port * 10, control, "the normalization changes the output beyond the port's error")
    }
}
