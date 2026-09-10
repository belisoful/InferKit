//
//  NFKMLXDDColorTests.swift
//  InferKitMLXTests
//
//  Weight-free structure tests for DDColor, plus the checkpoint coverage the loader promises.
//  Reference parity lives in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDDColorTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    func testTheParameterNamesMatchTheCheckpointLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXDDColor.makeNet(.tiny)
        let names = Set(net.parameters().flattened().map { $0.0 })
        for expected in ["encoder.arch.downsample_layers.0.0.weight",
                         "encoder.arch.downsample_layers.0.1.weight",
                         "encoder.arch.stages.0.0.dwconv.weight",
                         "encoder.arch.stages.0.0.pwconv1.weight",
                         "encoder.arch.stages.0.0.gamma",
                         "encoder.arch.norm3.weight",
                         "decoder.layers.0.shuf.conv.0.weight",
                         "decoder.layers.0.shuf.conv.1.weight",
                         "decoder.layers.0.bn.weight",
                         "decoder.layers.0.conv.0.weight",
                         "decoder.layers.0.conv.2.weight",
                         "decoder.last_shuf.conv.0.weight",
                         "decoder.color_decoder.query_feat.weight",
                         "decoder.color_decoder.level_embed.weight",
                         "decoder.color_decoder.input_proj.0.weight",
                         "decoder.color_decoder.transformer_cross_attention_layers.0.multihead_attn.in_proj_weight",
                         "decoder.color_decoder.transformer_self_attention_layers.0.self_attn.in_proj_weight",
                         "decoder.color_decoder.transformer_ffn_layers.0.linear1.weight",
                         "decoder.color_decoder.color_embed.layers.0.weight",
                         "refine_net.0.0.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // The ReLU slot in a `custom_conv_layer` Sequential still consumes its index, so the BatchNorm that
    // follows an activation lands at 2 rather than 1.
    func testTheActivationSlotShiftsTheBatchNormIndex() throws {
        try requireMLXRuntime()
        let net = NFKMLXDDColor.makeNet(.tiny)
        let names = Set(net.parameters().flattened().map { $0.0 })
        XCTAssertTrue(names.contains("decoder.layers.0.conv.2.weight"))      // conv, ReLU, BatchNorm
        XCTAssertFalse(names.contains("decoder.layers.0.conv.1.weight"))
        XCTAssertTrue(names.contains("decoder.layers.0.shuf.conv.1.weight")) // conv, BatchNorm
    }

    func testTheChromaHasTwoChannelsAtTheInputSize() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXDDColorConfiguration.tiny
        let net = NFKMLXDDColor.makeNet(configuration)
        let chroma = net(MLXArray.zeros([1, configuration.inputSize, configuration.inputSize, 3]))
        eval(chroma)
        XCTAssertEqual(chroma.shape, [1, configuration.inputSize, configuration.inputSize, 2])
    }

    // The position embedding pairs a sine with a cosine of the same frequency, and the row half
    // precedes the column half.
    func testThePositionEmbeddingIsTheReferenceLadder() {
        let embedding = NFKDDColorPositionEmbedding.sine(height: 4, width: 6, features: 8)
        XCTAssertEqual(embedding.shape, [1, 4, 6, 16])
        let values = embedding.reshaped([-1]).asArray(Float.self)
        // The lowest frequency divides by one, so the first row channel is sin of the scaled index.
        let scale = 2 * Float.pi
        let expected = sinf((1.0 / (4.0 + 1e-6)) * scale)
        XCTAssertEqual(values[0], expected, accuracy: 1e-5)
    }

    // Every released tensor is either loaded or named as deliberately dropped: the power-iteration
    // vectors the spectral fusion consumes, the encoder's classification head and its pooled norm, the
    // normalization buffers the port applies itself, and the BatchNorm step counters.
    func testEveryReleasedTensorIsLoadedOrNamedAsDropped() throws {
        try requireMLXRuntime()
        guard let weightsPath = NFKMLXValidationConfig.environment["IK_VAL_DDCOLOR"],
              FileManager.default.fileExists(atPath: weightsPath) else {
            throw XCTSkip("set IK_VAL_DDCOLOR")
        }
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: weightsPath))
        let net = NFKMLXDDColor.makeNet(.large)
        try NFKMLXDDColor.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let built = Set(net.parameters().flattened().map { $0.0 })

        var loaded = Set<String>()
        var dropped = 0
        var unaccounted = [String]()
        for (key, _) in checkpoint.arrays {
            guard let name = NFKMLXDDColor.remapReferenceKey(key) else { dropped += 1; continue }
            if built.contains(name) { loaded.insert(name) } else { unaccounted.append(key) }
        }
        print("[DDColor] coverage: loaded=\(loaded.count) built=\(built.count) "
              + "dropped=\(dropped) unaccounted=\(unaccounted.count)")
        XCTAssertEqual(unaccounted, [], "every released tensor must map onto a built parameter")
        XCTAssertEqual(loaded.count, built.count, "every built parameter must come from the checkpoint")
    }
}
