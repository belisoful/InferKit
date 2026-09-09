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
    /// isolated from the backbone. This is where a divergence surfaces if the kaldi reproduction is off.
    func testKaldiFbankMatchesTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let recordPath = environment["IK_PARITY_MOSSFORMER2_SE"] else {
            throw XCTSkip("set IK_PARITY_MOSSFORMER2_SE (oracle record) to run the fbank comparison")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let waveform = record["waveform"]!.asArray(Float.self)
        let feature = NFKMLXKaldiFbank.features(samples: waveform, config: NFKMLXMossFormer2Configuration())
        // Reference feature is [S, 180]; mine is [1, S, 180]. Compare the fbank band, Δ, and ΔΔ.
        // Measured on the released waveform: 1.0.
        XCTAssertGreaterThan(cosine(feature[0], record["feature"]!), 0.999, "Kaldi fbank + deltas")
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
        // Measured on the released weights (M1, float32): 0.9999998.
        XCTAssertGreaterThan(dot / (sqrtf(na) * sqrtf(nb) + 1e-20), 0.999, "enhanced waveform matches the reference")
    }
}
