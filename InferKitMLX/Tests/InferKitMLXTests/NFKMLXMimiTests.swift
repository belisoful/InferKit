//
//  NFKMLXMimiTests.swift
//  InferKitMLXTests
//
//  Mimi (kyutai/mimi, Kyutai, CC-BY-4.0), the transformer-in-codec neural audio codec. The module
//  evaluates MLX arrays, so these skip without a Metal library for MLX (see Tools/mlx-metallib.sh).
//  The parity test is gated on the released weights + the recorded oracle (`IK_VAL_MIMI` +
//  `IK_PARITY_MIMI`, from `run_reference.py mimi`), compared seam by seam.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXMimiTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        eval(mine)
        let a = mine.reshaped([-1]).asArray(Float.self), b = reference.reshaped([-1]).asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    /// The module keys the loader targets, including the folded codebook `embed` and the squeezed k1
    /// projection convolutions.
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXMimi.makeNet().parameters().flattened().map(\.0))
        for expected in ["encoder.layers.0.conv.weight", "encoder.layers.1.block.1.conv.weight",
                         "encoder_transformer.layers.0.self_attn.q_proj.weight",
                         "encoder_transformer.layers.0.self_attn_layer_scale.scale",
                         "encoder_transformer.layers.0.mlp.fc1.weight",
                         "downsample.conv.weight", "upsample.conv.weight",
                         "quantizer.semantic_residual_vector_quantizer.input_proj.weight",
                         "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed",
                         "quantizer.acoustic_residual_vector_quantizer.layers.30.codebook.embed",
                         "decoder.layers.14.conv.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    /// PARITY: the released Mimi against the recorded oracle, seam by seam (encoder, transformer,
    /// downsample, codes, quantizer decode, upsample, decoder transformer, waveform) and end to end.
    func testSeamParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let weightsPath = env["IK_VAL_MIMI"], let recordPath = env["IK_PARITY_MIMI"] else {
            throw XCTSkip("set IK_VAL_MIMI (model.safetensors) and IK_PARITY_MIMI (oracle record)")
        }
        let net = NFKMLXMimi.makeNet()
        try net.loadWeights(from: URL(fileURLWithPath: weightsPath))
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        func ref(_ key: String) -> MLXArray { rec[key]!.transposed(1, 0) }      // [C, T] → [T, C]
        func report(_ name: String, _ mine: MLXArray, _ reference: MLXArray) {
            XCTAssertGreaterThan(cosine(mine, reference), 0.999, "seam \(name) diverges")
        }

        let audio = rec["audio"]!                                               // [1, 24000]
        let x = audio.reshaped([1, audio.dim(1), 1])
        let emb = net.encoder(x)
        report("encoder", emb[0], ref("encoder_out"))
        let et = net.encoderTransformer(emb)
        report("transformer", et[0], ref("transformer_out"))
        let ds = net.downsample(et)
        report("downsample", ds[0], ref("downsample_out"))

        let codes = net.quantizer.encode(ds)                                    // [1, 32, T]
        eval(codes)
        XCTAssertEqual(codes[0].asType(.int32).asArray(Int32.self), rec["codes"]!.asArray(Int32.self),
                       "codes match the reference exactly")

        let dq = net.quantizer.decode(codes)
        report("quant_decode", dq[0], ref("quant_decode"))
        let us = net.upsample(dq)
        report("upsample", us[0], ref("upsample_out"))
        let dt = net.decoderTransformer(us)
        report("dtransformer", dt[0], ref("dtransformer_out"))
        let wav = net.decoder(dt)[0..., 0..., 0]
        XCTAssertGreaterThan(cosine(wav[0], rec["output"]!), 0.999, "waveform diverges")

        let full = net.decode(net.encode(audio))
        XCTAssertGreaterThan(cosine(full[0], rec["full_waveform"]!), 0.999, "end-to-end waveform diverges")
    }

    /// The reconstruction backend returns a 24 kHz clip; the registry builds it under the model name.
    func testTheBackendReconstructsAudio() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXMimi.backend(weightsURL: nil)
        let wave = NFKMLXWaveFile.data(samples: (0 ..< 4096).map { sinf(Float($0) * 0.05) * 0.4 }, sampleRate: 24000)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio), "the reconstructed clip is returned")
    }

    func testTheRegistryPathBuildsAReadyBackend() throws {
        try requireMLXRuntime()
        NFKMLXMimi.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXMimi.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXMimi.modelName, weightsURL: nil)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputAudio])
    }
}
