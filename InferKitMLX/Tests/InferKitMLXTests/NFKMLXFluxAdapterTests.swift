//
//  NFKMLXFluxAdapterTests.swift
//  InferKitMLXTests
//
//  FLUX.1 adapters held beside the transformer: a diffusers file reaches its layers by their own names, and a
//  kohya or Black Forest Labs file names the original model's fused layers, whose adapter splits across the
//  layers they hold. Measured layer by layer against the scaled low-rank product each part should add.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXFluxAdapterTests: XCTestCase {

    private var directories: [URL] = []

    override func tearDown() {
        for directory in directories {
            try? FileManager.default.removeItem(at: directory)
        }
        directories = []
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The tiny transformer: an inner width of 12 and a feed-forward of 48.
    private func tinyNet() -> NFKMLXFluxTransformerNet {
        let net = NFKMLXFluxTransformerNet(.tiny)
        eval(net)
        return net
    }

    private func file(_ arrays: [String: MLXArray], metadata: [String: String] = [:]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        directories.append(directory)
        let url = directory.appendingPathComponent("lora.safetensors")
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
        return url
    }

    private func layer(_ net: Module, _ path: String) -> Linear {
        Dictionary(net.leafModules().flattened(), uniquingKeysWith: { first, _ in first })[path] as! Linear
    }

    private func worst(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    private let x = MLXRandom.normal([1, 3, 12], key: MLXRandom.key(3))
    private let down = MLXRandom.normal([4, 12], key: MLXRandom.key(5)) * 0.2

    /// What a part's detour adds: `scale · x·downᵀ·upᵀ` over `up`'s rows `rows`.
    private func product(_ up: MLXArray, _ rows: Range<Int>, scale: Float) -> MLXArray {
        matmul(matmul(x, down.transposed(1, 0)), up[rows].transposed(1, 0)) * scale
    }

    func testAKohyaAdapterSplitsTheOriginalModelsFusedLayers() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let qkv = MLXRandom.normal([36, 4], key: MLXRandom.key(7)) * 0.2
        let linear1 = MLXRandom.normal([84, 4], key: MLXRandom.key(9)) * 0.2
        let modulation = MLXRandom.normal([24, 4], key: MLXRandom.key(11)) * 0.2
        let paths = ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_v",
                     "single_transformer_blocks.1.proj_mlp", "norm_out.linear"]
        let before = Dictionary(uniqueKeysWithValues: paths.map { ($0, layer(net, $0)(x)) })
        eval(Array(before.values))
        let url = try file([
            "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": down,
            "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": qkv,
            "lora_unet_double_blocks_0_img_attn_qkv.alpha": MLXArray(Float(8)),
            "lora_unet_single_blocks_1_linear1.lora_down.weight": down,
            "lora_unet_single_blocks_1_linear1.lora_up.weight": linear1,
            "lora_unet_single_blocks_1_linear1.alpha": MLXArray(Float(4)),
            "lora_unet_final_layer_adaLN_modulation_1.lora_down.weight": down,
            "lora_unet_final_layer_adaLN_modulation_1.lora_up.weight": modulation,
        ])
        XCTAssertEqual(try NFKMLXFluxAdapters.hold([(url, 1)], in: net), [1])
        let expected = [
            "transformer_blocks.0.attn.to_q": product(qkv, 0 ..< 12, scale: 2),
            "transformer_blocks.0.attn.to_k": product(qkv, 12 ..< 24, scale: 2),
            "transformer_blocks.0.attn.to_v": product(qkv, 24 ..< 36, scale: 2),
            "single_transformer_blocks.1.proj_mlp": product(linear1, 36 ..< 84, scale: 1),
            // The original model orders the final modulation shift, scale; diffusers orders it scale, shift.
            "norm_out.linear": product(concatenated([modulation[12...], modulation[..<12]], axis: 0), 0 ..< 24, scale: 1),
        ]
        for path in paths {
            XCTAssertLessThan(worst(layer(net, path)(x), before[path]! + expected[path]!), 1e-5, path)
        }
    }

    func testTheBlackForestLabsLayoutReadsAsKohyas() throws {
        try requireMLXRuntime()
        let kohyaNet = tinyNet()
        let bflNet = NFKMLXFluxTransformerNet(.tiny)
        bflNet.update(parameters: kohyaNet.parameters())
        let up = MLXRandom.normal([36, 4], key: MLXRandom.key(13)) * 0.2
        let kohya = try file(["lora_unet_double_blocks_1_txt_attn_qkv.lora_down.weight": down,
                              "lora_unet_double_blocks_1_txt_attn_qkv.lora_up.weight": up,
                              "lora_unet_double_blocks_1_txt_attn_qkv.alpha": MLXArray(Float(2))])
        let bfl = try file(["diffusion_model.double_blocks.1.txt_attn.qkv.lora_A.weight": down,
                            "diffusion_model.double_blocks.1.txt_attn.qkv.lora_B.weight": up,
                            "diffusion_model.double_blocks.1.txt_attn.qkv.alpha": MLXArray(Float(2))])
        try NFKMLXFluxAdapters.hold([(kohya, 0.7)], in: kohyaNet)
        try NFKMLXFluxAdapters.hold([(bfl, 0.7)], in: bflNet)
        for path in ["transformer_blocks.1.attn.add_q_proj", "transformer_blocks.1.attn.add_k_proj",
                     "transformer_blocks.1.attn.add_v_proj"] {
            XCTAssertTrue(layer(bflNet, path) is NFKMLXAdaptedLinear, path)
            XCTAssertLessThan(worst(layer(kohyaNet, path)(x), layer(bflNet, path)(x)), 1e-6, path)
        }
    }

    func testADiffusersAdapterTakesTheScalingItsMetadataStates() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let path = "transformer_blocks.1.attn.to_out.0"
        let before = layer(net, path)(x)
        eval(before)
        let up = MLXRandom.normal([12, 4], key: MLXRandom.key(15)) * 0.2
        let url = try file(["transformer.\(path).lora_A.weight": down, "transformer.\(path).lora_B.weight": up],
                           metadata: ["lora_adapter_metadata": #"{"transformer.r": 4, "transformer.lora_alpha": 12}"#])
        try NFKMLXFluxAdapters.hold([(url, 0.5)], in: net)
        XCTAssertLessThan(worst(layer(net, path)(x), before + product(up, 0 ..< 12, scale: 3 * 0.5)), 1e-5)
    }

    // [schnell] has no guidance embedding; a [dev] adapter's part for it is left out, as diffusers leaves it out.
    func testAGuidanceEmbeddingsAdapterIsLeftOutOfATransformerWithoutOne() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXFluxConfiguration.tiny
        configuration.guidanceEmbeds = false
        let net = NFKMLXFluxTransformerNet(configuration)
        eval(net)
        let up = MLXRandom.normal([12, 4], key: MLXRandom.key(17)) * 0.2
        let url = try file([
            "transformer.time_text_embed.guidance_embedder.linear_1.lora_A.weight": MLXRandom.normal([4, 256]),
            "transformer.time_text_embed.guidance_embedder.linear_1.lora_B.weight": MLXRandom.normal([12, 4]),
            "transformer.proj_out.lora_A.weight": down, "transformer.proj_out.lora_B.weight": MLXRandom.normal([8, 4]),
            "transformer.x_embedder.lora_A.weight": MLXRandom.normal([4, 8]), "transformer.x_embedder.lora_B.weight": up,
        ])
        try NFKMLXFluxAdapters.hold([(url, 1)], in: net)
        XCTAssertTrue(layer(net, "x_embedder") is NFKMLXAdaptedLinear)
        XCTAssertTrue(layer(net, "proj_out") is NFKMLXAdaptedLinear)
    }

    func testAnAdapterFLUXCannotHoldIsRefusedBeforeAnyLayerChanges() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let refused: [(String, [String: MLXArray])] = [
            ("a text encoder", ["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": down,
                                "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": down]),
            ("an XLabs file", ["double_blocks.0.processor.qkv_lora1.down.weight": down]),
            ("a block the transformer lacks", ["lora_unet_double_blocks_9_img_attn_proj.lora_down.weight": down,
                                               "lora_unet_double_blocks_9_img_attn_proj.lora_up.weight":
                                                   MLXArray.zeros([12, 4])]),
            ("fused outputs that do not add up", ["lora_unet_single_blocks_0_linear1.lora_down.weight": down,
                                                  "lora_unet_single_blocks_0_linear1.lora_up.weight":
                                                      MLXArray.zeros([80, 4])]),
            ("no layout", ["weird.weight": down]),
        ]
        for (reason, arrays) in refused {
            XCTAssertThrowsError(try NFKMLXFluxAdapters.hold([(try file(arrays), 1)], in: net), reason)
        }
        XCTAssertFalse(layer(net, "single_transformer_blocks.0.attn.to_q") is NFKMLXAdaptedLinear,
                       "a refusal leaves every layer as it was")
    }
}
