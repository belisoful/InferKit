//
//  NFKMLXChronosTests.swift
//  InferKitMLXTests
//
//  Chronos-Bolt (amazon/chronos-bolt-base, Amazon, Apache-2.0), a patched T5 encoder-decoder time-series
//  forecaster. The module evaluates MLX arrays, so these skip without a Metal library for MLX (see
//  Tools/mlx-metallib.sh). Parity is gated on the released weights + the recorded oracle (`IK_VAL_CHRONOS`
//  + `IK_PARITY_CHRONOS`, from `run_reference.py chronos`), compared seam by seam.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXChronosTests: XCTestCase {

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

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXChronos.makeNet().parameters().flattened().map(\.0))
        for expected in ["shared.weight", "input_patch_embedding.hidden_layer.weight",
                         "input_patch_embedding.residual_layer.weight", "output_patch_embedding.output_layer.weight",
                         "encoder.block.0.layer.0.SelfAttention.q.weight",
                         "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                         "encoder.block.0.layer.1.DenseReluDense.wi.weight",
                         "decoder.block.0.layer.1.EncDecAttention.q.weight",
                         "decoder.block.0.layer.2.DenseReluDense.wi.weight",
                         "encoder.final_layer_norm.weight", "decoder.final_layer_norm.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
        // The bias table lives only on the first block of each stack.
        XCTAssertFalse(names.contains("encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight"))
    }

    func testSeamParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let weightsPath = env["IK_VAL_CHRONOS"], let recordPath = env["IK_PARITY_CHRONOS"] else {
            throw XCTSkip("set IK_VAL_CHRONOS (model.safetensors) and IK_PARITY_CHRONOS (oracle record)")
        }
        let net = NFKMLXChronos.makeNet()
        try net.loadWeights(from: URL(fileURLWithPath: weightsPath))
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        func report(_ name: String, _ mine: MLXArray, _ reference: MLXArray) {
            XCTAssertGreaterThan(cosine(mine, reference), 0.999, "seam \(name) diverges")
        }

        let context = rec["context"]!.reshaped([1, rec["context"]!.dim(0)])       // [1, 512]

        // Instance normalization.
        let loc = context.mean(axis: -1, keepDims: true)
        let scale = sqrt(((context - loc) * (context - loc)).mean(axis: -1, keepDims: true))
        eval(loc, scale)
        XCTAssertLessThan(abs(loc.reshaped([-1]).asArray(Float.self)[0] - rec["loc"]!.asArray(Float.self)[0]), 1e-3, "loc")
        XCTAssertLessThan(abs(scale.reshaped([-1]).asArray(Float.self)[0] - rec["scale"]!.asArray(Float.self)[0]), 1e-3, "scale")
        let scaled = (context - loc) / scale

        // Patchify (512 is a multiple of 16, so no padding), input embedding, and the REG token.
        let nPatches = scaled.dim(1) / 16
        let patched = concatenated([scaled.reshaped([1, nPatches, 16]), MLXArray.ones([1, nPatches, 16])], axis: -1)
        var embeds = net.inputPatch(patched)
        embeds = concatenated([embeds, net.shared(MLXArray([Int32(1)])).reshaped([1, 1, net.config.dModel])], axis: 1)
        report("input_embeds", embeds[0], rec["input_embeds"]!)

        let pad = MLXArray.zeros([1, 1, 1, nPatches + 1])                          // all observed → no masking
        let encoded = net.encoder(embeds, paddingAdditive: pad)
        report("encoder_hidden", encoded[0], rec["encoder_hidden"]!)

        let decStart = net.shared(MLXArray([Int32(0)])).reshaped([1, 1, net.config.dModel])
        let decoded = net.decoder(decStart, encoder: encoded, encoderMask: pad)
        report("decoder_out", decoded[0], rec["decoder_out"]!)

        let qpScaled = net.outputPatch(decoded).reshaped([1, net.config.quantiles.count, net.config.predictionLength])
        report("quantile_preds_scaled", qpScaled[0], rec["quantile_preds_scaled"]!)

        // The whole network (un-scaled forecast).
        report("output", net(context)[0], rec["output"]!)
    }

    func testTheForecastObjectMatchesTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let weightsPath = env["IK_VAL_CHRONOS"], let recordPath = env["IK_PARITY_CHRONOS"] else {
            throw XCTSkip("set IK_VAL_CHRONOS and IK_PARITY_CHRONOS")
        }
        let chronos = try NFKMLXChronos.chronos(weightsURL: URL(fileURLWithPath: weightsPath))
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let context = rec["context"]!.asArray(Float.self)
        let rows = chronos.forecast(context: context, horizon: 64)
        XCTAssertEqual(rows.count, 9, "one row per quantile")
        XCTAssertEqual(rows[0].count, 64, "the horizon")
        // The reference forecast, quantile row by row.
        let ref = rec["output"]!.reshaped([9, 64])
        for q in 0 ..< 9 {
            let mine = rows[q].withUnsafeBufferPointer { MLXArray($0, [64]) }
            XCTAssertGreaterThan(cosine(mine, ref[q]), 0.999, "quantile \(q) diverges")
        }
    }

    func testTheForecastObjectRunsWithRandomWeights() throws {
        try requireMLXRuntime()
        let chronos = try NFKMLXChronos.chronos(weightsURL: nil)
        let rows = chronos.forecast(context: (0 ..< 100).map { sinf(Float($0) * 0.1) }, horizon: 32)
        XCTAssertEqual(rows.count, 9)
        XCTAssertEqual(rows[0].count, 32)
        XCTAssertEqual(chronos.quantileLevels.count, 9)
    }
}
