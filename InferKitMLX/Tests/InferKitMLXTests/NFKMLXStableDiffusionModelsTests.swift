//
//  NFKMLXStableDiffusionModelsTests.swift
//  InferKitMLXTests
//
//  The shared Stable Diffusion UNet and autoencoder at sizes a test can run, plus the pure key
//  remaps. The parity records prove the numbers against diffusers; these prove the geometry, the
//  remaps, and the configuration switches with no external files. The MLX forwards skip without a
//  Metal library for MLX (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXStableDiffusionModelsTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    // MARK: Remaps (pure)

    func testTheRemapTranslatesTheUNetKeys() {
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapUNetKey(
            "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj.weight"),
            "down_blocks.0.attentions.1.transformer_blocks.0.ff.proj.weight")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapUNetKey(
            "mid_block.attentions.0.transformer_blocks.0.ff.net.2.bias"),
            "mid_block.attentions.0.transformer_blocks.0.ff.out.bias")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapUNetKey(
            "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.weight"),
            "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.weight")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapUNetKey(
            "down_blocks.1.attentions.0.proj_in.weight"),
            "down_blocks.1.attentions.0.proj_in_conv.weight")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapUNetKey("conv_in.weight"), "conv_in.weight")
    }

    func testTheRemapTranslatesTheVAEKeys() {
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapVAEKey(
            "decoder.mid_block.attentions.0.to_out.0.weight"),
            "decoder.mid_block.attentions.0.to_out.weight")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapVAEKey(
            "encoder.down_blocks.1.resnets.0.conv_shortcut.weight"),
            "encoder.down_blocks.1.resnets.0.block.conv_shortcut.weight")
        // The SD 1.5 autoencoders predate the diffusers attention rename.
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapVAEKey(
            "encoder.mid_block.attentions.0.query.weight"),
            "encoder.mid_block.attentions.0.to_q.weight")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapVAEKey(
            "decoder.mid_block.attentions.0.proj_attn.bias"),
            "decoder.mid_block.attentions.0.to_out.bias")
        XCTAssertEqual(NFKMLXStableDiffusionModels.remapVAEKey("quant_conv.weight"), "quant_conv.weight")
    }

    func testTheReleasedConfigurationsCarryTheirLoadBearingSwitches() {
        XCTAssertEqual(NFKMLXSDUNetConfiguration.inpainting.inputChannels, 9,
                       "noisy latent + mask + masked-image latent")
        XCTAssertTrue(NFKMLXSDUNetConfiguration.marigold.usesLinearProjection)
        XCTAssertEqual(NFKMLXSDUNetConfiguration.upscaler.attends, [false, true, true, true],
                       "the upscaler's plain level is its FIRST")
        XCTAssertEqual(NFKMLXSDUNetConfiguration.upscaler.onlyCrossAttention, [true, true, true, false],
                       "three levels put the context's width on their first attention too")
        XCTAssertEqual(NFKMLXSDUNetConfiguration.upscaler.classEmbeddingCount, 1000)
        XCTAssertEqual(NFKMLXSDVAEConfiguration.upscaler.blockChannels.count, 3,
                       "one level shallower is where the upscaler's ×4 comes from")
    }

    // MARK: Geometry (needs MLX)

    /// The released structure at a size a test can run, over two narrow levels. Widths stay multiples
    /// of the normalization's group count.
    static func tinyUNet(inputChannels: Int = 4, onlyCross: Bool = false,
                         classes: Int? = nil) -> NFKMLXSDUNetConfiguration {
        var configuration = NFKMLXSDUNetConfiguration()
        configuration.inputChannels = inputChannels
        configuration.blockChannels = [32, 64]
        configuration.attends = [true, false]
        configuration.attentionHeads = [2, 2]
        configuration.onlyCrossAttention = [onlyCross, false]
        configuration.crossAttentionDimensions = 16
        configuration.classEmbeddingCount = classes
        return configuration
    }

    func testUNetPredictsAtTheLatentShapeThroughEveryConditioningPath() throws {
        try requireMLXRuntime()
        let latent = MLXRandom.uniform(low: -1, high: 1, [1, 8, 8, 4])
        let context = MLXRandom.uniform(low: -1, high: 1, [1, 7, 16])
        for (name, configuration) in [("plain", Self.tinyUNet()),
                                      ("only-cross", Self.tinyUNet(onlyCross: true)),
                                      ("class-embedded", Self.tinyUNet(classes: 10))] {
            let net = NFKMLXSDUNet(configuration: configuration)
            net.train(false)
            let out = net(latent, timestep: MLXArray([Int32(201)]), context: context,
                          classLabel: configuration.classEmbeddingCount.map { _ in MLXArray([Int32(3)]) })
            eval(out)
            XCTAssertEqual(out.shape, [1, 8, 8, 4], "\(name) UNet keeps the latent shape")
        }
    }

    func testAutoencoderRoundTripsAnImageAtTheConfiguredStride() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXSDVAEConfiguration()
        configuration.blockChannels = [32, 64, 64]                // three levels → stride 4
        let net = NFKMLXSDAutoencoder(configuration: configuration)
        net.train(false)
        let image = MLXRandom.uniform(low: -1, high: 1, [1, 32, 32, 3])
        let (mean, logVariance) = net.encode(image)
        eval(mean, logVariance)
        XCTAssertEqual(mean.shape, [1, 8, 8, 4], "two downsampling levels give a stride of 4")
        XCTAssertEqual(logVariance.shape, mean.shape)
        let decoded = net.decode(mean)
        eval(decoded)
        XCTAssertEqual(decoded.shape, image.shape)
    }

    // MARK: Round trip (needs MLX)

    func testACheckpointRoundTripReproducesTheUNetPrediction() throws {
        try requireMLXRuntime()
        let trained = NFKMLXSDUNet(configuration: Self.tinyUNet())
        // Write the diffusers layout: the remapped names inverted, 4-D tensors rotated back.
        let referenceLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened()
            .map { key, value -> (String, MLXArray) in
                var name = key
                name = name.replacingOccurrences(of: ".ff.proj.", with: ".ff.net.0.proj.")
                name = name.replacingOccurrences(of: ".ff.out.", with: ".ff.net.2.")
                name = name.replacingOccurrences(of: ".to_out.", with: ".to_out.0.")
                name = name.replacingOccurrences(of: ".proj_in_conv.", with: ".proj_in.")
                name = name.replacingOccurrences(of: ".proj_out_conv.", with: ".proj_out.")
                return (name, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
            })
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sd-unet-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: referenceLayout, url: url)

        let loaded = NFKMLXSDUNet(configuration: Self.tinyUNet())
        try NFKMLXStableDiffusionModels.loadUNetWeights(into: loaded, from: url)

        let latent = MLXRandom.uniform(low: -1, high: 1, [1, 8, 8, 4], key: MLXRandom.key(5))
        let context = MLXRandom.uniform(low: -1, high: 1, [1, 7, 16], key: MLXRandom.key(6))
        let expected = trained(latent, timestep: MLXArray([Int32(101)]), context: context)
        let actual = loaded(latent, timestep: MLXArray([Int32(101)]), context: context)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    // MARK: Converting load (needs MLX)

    // The float16 choice converts every floating weight of the UNet, the autoencoder, and the text tower as
    // it is read, to the bits a float32 load followed by a cast gives, which is what the float16 parity
    // tests hold to diffusers.
    func testAFloat16LoadConvertsEachNetworkAsAFloat32LoadAndACastDo() throws {
        try requireMLXRuntime()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("sd-cast-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }

        let unetURL = directory.appendingPathComponent("unet.safetensors")
        try save(arrays: Self.diffusersLayout(NFKMLXSDUNet(configuration: Self.tinyUNet())), url: unetURL)
        try assertConvertingLoad("unet", { NFKMLXSDUNet(configuration: Self.tinyUNet()) }) { net, dtype in
            try NFKMLXStableDiffusionModels.loadUNetWeights(into: net, from: unetURL, precision: .float32, castingTo: dtype)
        }

        var autoencoder = NFKMLXSDVAEConfiguration()
        autoencoder.blockChannels = [32, 64]
        let vaeURL = directory.appendingPathComponent("vae.safetensors")
        try save(arrays: Self.diffusersLayout(NFKMLXSDAutoencoder(configuration: autoencoder)), url: vaeURL)
        try assertConvertingLoad("vae", { NFKMLXSDAutoencoder(configuration: autoencoder) }) { net, dtype in
            try NFKMLXStableDiffusionModels.loadVAEWeights(into: net, from: vaeURL, precision: .float32, castingTo: dtype)
        }

        let textURL = directory.appendingPathComponent("text_encoder.safetensors")
        var text = Self.transformersLayout(NFKMLXSDTextEncoderNet(configuration: .tiny))
        text["text_model.embeddings.position_ids"] = MLXArray(Array(Int64(0) ..< 12)).reshaped([1, 12])
        try save(arrays: text, url: textURL)
        try assertConvertingLoad("text", { NFKMLXSDTextEncoderNet(configuration: .tiny) }) { net, dtype in
            try NFKMLXSDTextEncoder.loadWeights(into: net, from: textURL, precision: .float32, castingTo: dtype)
        }
    }

    /// Loads `make()` twice through `load`, at float32 and converting to float16, and asserts every
    /// parameter of the converted load is the float32 one cast to float16.
    private func assertConvertingLoad<Net: Module>(_ name: String, _ make: () -> Net,
                                                   load: (Net, DType?) throws -> Void) throws {
        let widened = make(), converted = make()
        try load(widened, nil)
        try load(converted, .float16)
        let expected = Dictionary(uniqueKeysWithValues: widened.parameters().flattened())
        let loaded = converted.parameters().flattened()
        XCTAssertEqual(loaded.count, expected.count, name)
        for (key, value) in loaded {
            let original = try XCTUnwrap(expected[key], "\(name) \(key)")
            XCTAssertEqual(original.dtype, .float32, "\(name) \(key)")
            XCTAssertEqual(value.dtype, .float16, "\(name) \(key)")
            XCTAssertTrue(arrayEqual(value, original.asType(.float16)).item(Bool.self), "\(name) \(key)")
        }
    }

    /// A UNet's or an autoencoder's parameters under diffusers' names, with 4-D tensors rotated back to
    /// `[out, in, kH, kW]`.
    static func diffusersLayout(_ net: Module) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { key, value -> (String, MLXArray) in
            var name = key
            name = name.replacingOccurrences(of: ".ff.proj.", with: ".ff.net.0.proj.")
            name = name.replacingOccurrences(of: ".ff.out.", with: ".ff.net.2.")
            name = name.replacingOccurrences(of: ".to_out.", with: ".to_out.0.")
            name = name.replacingOccurrences(of: ".proj_in_conv.", with: ".proj_in.")
            name = name.replacingOccurrences(of: ".proj_out_conv.", with: ".proj_out.")
            return (name, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
    }

    /// A text tower's parameters under transformers' `CLIPTextModel` names, the fused attention input
    /// projection split into its query, key, and value.
    static func transformersLayout(_ net: NFKMLXSDTextEncoderNet) -> [String: MLXArray] {
        let names = ["token_embedding.weight": "embeddings.token_embedding.weight",
                     "positional_embedding": "embeddings.position_embedding.weight",
                     "ln_final.weight": "final_layer_norm.weight", "ln_final.bias": "final_layer_norm.bias",
                     "ln_1": "layer_norm1", "ln_2": "layer_norm2", "mlp.c_fc": "mlp.fc1", "mlp.c_proj": "mlp.fc2",
                     "attn.out_proj": "self_attn.out_proj"]
        var arrays = [String: MLXArray]()
        for (key, value) in net.parameters().flattened() {
            if let name = names[key] {
                arrays["text_model." + name] = value
                continue
            }
            let parts = key.split(separator: ".").map(String.init)
            let layer = "text_model.encoder.layers.\(parts[2])."
            let part = parts.dropFirst(3).dropLast().joined(separator: ".")
            if part == "attn", let last = parts.last, last.hasPrefix("in_proj_") {
                let suffix = last == "in_proj_weight" ? "weight" : "bias"
                for (index, projection) in split(value, parts: 3, axis: 0).enumerated() {
                    arrays[layer + "self_attn.\(["q_proj", "k_proj", "v_proj"][index]).\(suffix)"] = projection
                }
            } else if let name = names[part], let last = parts.last {
                arrays[layer + name + "." + last] = value
            }
        }
        return arrays
    }
}
