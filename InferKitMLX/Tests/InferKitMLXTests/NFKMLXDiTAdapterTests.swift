//
//  NFKMLXDiTAdapterTests.swift
//  InferKitMLXTests
//
//  Diffusion transformer adapters held beside the transformer: the shared reader takes diffusers' names with or
//  without their prefix, the original model's fused names, and kohya's underscored names, and refuses what no held
//  detour computes before any layer changes. Each wired family holds a diffusers-named adapter that a strength of 0
//  leaves inert and a strength of 1 reads, and its catalog entry declares the adapters.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

/// `a.b_c` and `a_b.c`, which kohya writes alike, beside `d`.
private final class NFKDiTAdapterTestTree: Module {
    final class Pair: Module {
        @ModuleInfo(key: "b_c") var bc: Linear

        init(_ width: Int) {
            _bc.wrappedValue = Linear(width, width)
        }
    }

    final class Single: Module {
        @ModuleInfo(key: "c") var c: Linear

        init(_ width: Int) {
            _c.wrappedValue = Linear(width, width)
        }
    }

    @ModuleInfo(key: "a") var a: Pair
    @ModuleInfo(key: "a_b") var ab: Single
    @ModuleInfo(key: "d") var d: Linear

    init(_ width: Int) {
        _a.wrappedValue = Pair(width)
        _ab.wrappedValue = Single(width)
        _d.wrappedValue = Linear(width, width)
    }
}

final class NFKMLXDiTAdapterTests: XCTestCase {

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

    private func file(_ arrays: [String: MLXArray], metadata: [String: String] = [:]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        directories.append(directory)
        let url = directory.appendingPathComponent("lora.safetensors")
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
        return url
    }

    private func layers(_ net: Module) -> [String: Module] {
        Dictionary(net.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
    }

    private func linear(_ net: Module, _ path: String) -> Linear {
        layers(net)[path] as! Linear
    }

    private func worst(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// `[rank, inputs]` and `[outputs, rank]` halves drawn from `seed`.
    private func halves(_ seed: UInt64, inputs: Int, outputs: Int, rank: Int = 4) -> (down: MLXArray, up: MLXArray) {
        (MLXRandom.normal([rank, inputs], key: MLXRandom.key(seed)) * 0.3,
         MLXRandom.normal([outputs, rank], key: MLXRandom.key(seed + 100)) * 0.3)
    }

    /// What a detour adds for `x`: `scale · x·downᵀ·upᵀ`.
    private func product(_ x: MLXArray, _ down: MLXArray, _ up: MLXArray, scale: Float) -> MLXArray {
        matmul(matmul(x, down.transposed(1, 0)), up.transposed(1, 0)) * scale
    }

    private func flux2Net() -> NFKMLXFlux2TransformerNet {
        let net = NFKMLXFlux2TransformerNet(.tiny)
        eval(net)
        return net
    }

    private func copy(of net: NFKMLXFlux2TransformerNet) -> NFKMLXFlux2TransformerNet {
        let other = NFKMLXFlux2TransformerNet(.tiny)
        other.update(parameters: net.parameters())
        eval(other)
        return other
    }

    // MARK: The shared reader

    func testADiffusersFileReadsTheSameWithOrWithoutItsPrefix() throws {
        try requireMLXRuntime()
        let prefixedNet = flux2Net()
        let bareNet = copy(of: prefixedNet)
        let paths = ["single_transformer_blocks.0.attn.to_qkv_mlp_proj", "transformer_blocks.1.ff.linear_in"]
        var prefixed = [String: MLXArray](), bare = [String: MLXArray]()
        var expected = [String: (MLXArray, MLXArray)]()
        for (index, path) in paths.enumerated() {
            let weight = linear(prefixedNet, path).weight
            let (down, up) = halves(UInt64(20 + index), inputs: weight.dim(1), outputs: weight.dim(0))
            expected[path] = (down, up)
            prefixed["transformer.\(path).lora_A.weight"] = down
            prefixed["transformer.\(path).lora_B.weight"] = up
            prefixed["transformer.\(path).alpha"] = MLXArray(Float(8))
            bare["\(path).lora_A.weight"] = down
            bare["\(path).lora_B.weight"] = up
            bare["\(path).alpha"] = MLXArray(Float(8))
        }
        let x = MLXRandom.normal([1, 3, 16], key: MLXRandom.key(3))
        let before = Dictionary(uniqueKeysWithValues: paths.map { ($0, linear(prefixedNet, $0)(x)) })
        eval(Array(before.values))
        XCTAssertEqual(try NFKMLXDiTAdapters.hold([(try file(prefixed), 0.5)], in: prefixedNet), [0.5])
        try NFKMLXDiTAdapters.hold([(try file(bare), 0.5)], in: bareNet)
        for path in paths {
            XCTAssertTrue(linear(prefixedNet, path) is NFKMLXAdaptedLinear, path)
            XCTAssertTrue(linear(bareNet, path) is NFKMLXAdaptedLinear, path)
            let (down, up) = try XCTUnwrap(expected[path])
            XCTAssertLessThan(worst(linear(prefixedNet, path)(x), before[path]! + product(x, down, up, scale: 2 * 0.5)),
                              1e-5, path)
            XCTAssertEqual(worst(linear(prefixedNet, path)(x), linear(bareNet, path)(x)), 0, path)
        }
    }

    func testAPEFTDirectoryReadsItsScalingFromTheConfigurationBesideIt() throws {
        try requireMLXRuntime()
        let net = flux2Net()
        let path = "transformer_blocks.1.ff.linear_in"
        let weight = linear(net, path).weight
        let (down, up) = halves(31, inputs: weight.dim(1), outputs: weight.dim(0))
        let adapter = try file(["base_model.model.\(path).lora_A.weight": down, "base_model.model.\(path).lora_B.weight": up])
        try JSONSerialization.data(withJSONObject: ["peft_type": "LORA", "r": 4, "lora_alpha": 8])
            .write(to: adapter.deletingLastPathComponent().appendingPathComponent("adapter_config.json"))
        let x = MLXRandom.normal([1, 3, weight.dim(1)], key: MLXRandom.key(33))
        let before = linear(net, path)(x)
        eval(before)
        try NFKMLXDiTAdapters.hold([(adapter, 1)], in: net)
        XCTAssertLessThan(worst(linear(net, path)(x), before + product(x, down, up, scale: 2)), 1e-5,
                          "lora_alpha / r from adapter_config.json")
    }

    func testAnOriginalModelsFusedNameSplitsItsUpHalfAcrossTheLayersItIs() throws {
        try requireMLXRuntime()
        let net = flux2Net()
        let parts = ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_v"]
        let single = "single_transformer_blocks.1.attn.to_qkv_mlp_proj"
        let width = linear(net, parts[0]).weight.dim(0)
        let (down, up) = halves(30, inputs: 16, outputs: 3 * width)
        let (singleDown, singleUp) = halves(31, inputs: 16, outputs: linear(net, single).weight.dim(0))
        let x = MLXRandom.normal([1, 3, 16], key: MLXRandom.key(5))
        let before = Dictionary(uniqueKeysWithValues: (parts + [single]).map { ($0, linear(net, $0)(x)) })
        eval(Array(before.values))
        let url = try file([
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": down,
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": up,
            "diffusion_model.single_blocks.1.linear1.lora_A.weight": singleDown,
            "diffusion_model.single_blocks.1.linear1.lora_B.weight": singleUp,
        ])
        try NFKMLXDiTAdapters.hold([(url, 1)], in: net)
        for (index, path) in parts.enumerated() {
            let rows = up[(index * width) ..< ((index + 1) * width)]
            XCTAssertLessThan(worst(linear(net, path)(x), before[path]! + product(x, down, rows, scale: 1)), 1e-5, path)
        }
        XCTAssertLessThan(worst(linear(net, single)(x), before[single]! + product(x, singleDown, singleUp, scale: 1)),
                          1e-5, "FLUX.2's linear1 is the one fused projection diffusers names to_qkv_mlp_proj")
    }

    func testKohyaNamesResolveThroughTheOriginalTableAndThroughTheModulePaths() throws {
        try requireMLXRuntime()
        let net = flux2Net()
        let parts = ["transformer_blocks.0.attn.add_q_proj", "transformer_blocks.0.attn.add_k_proj",
                     "transformer_blocks.0.attn.add_v_proj"]
        let output = "transformer_blocks.1.attn.to_out.0"
        let modulation = "single_stream_modulation.linear"
        let width = linear(net, parts[0]).weight.dim(0)
        let (qkvDown, qkvUp) = halves(40, inputs: 16, outputs: 3 * width)
        let (outDown, outUp) = halves(41, inputs: 16, outputs: linear(net, output).weight.dim(0))
        let (modDown, modUp) = halves(42, inputs: 16, outputs: linear(net, modulation).weight.dim(0))
        let x = MLXRandom.normal([1, 3, 16], key: MLXRandom.key(7))
        let before = Dictionary(uniqueKeysWithValues: (parts + [output, modulation]).map { ($0, linear(net, $0)(x)) })
        eval(Array(before.values))
        let url = try file([
            "lora_unet_double_blocks_0_txt_attn_qkv.lora_down.weight": qkvDown,
            "lora_unet_double_blocks_0_txt_attn_qkv.lora_up.weight": qkvUp,
            "lora_unet_double_blocks_0_txt_attn_qkv.alpha": MLXArray(Float(8)),
            "lora_unet_transformer_blocks_1_attn_to_out_0.lora_down.weight": outDown,
            "lora_unet_transformer_blocks_1_attn_to_out_0.lora_up.weight": outUp,
            "lora_unet_transformer_blocks_1_attn_to_out_0.alpha": MLXArray(Float(2)),
            "lora_unet_single_stream_modulation_lin.lora_down.weight": modDown,
            "lora_unet_single_stream_modulation_lin.lora_up.weight": modUp,
        ])
        try NFKMLXDiTAdapters.hold([(url, 0.7)], in: net)
        for (index, path) in parts.enumerated() {
            let rows = qkvUp[(index * width) ..< ((index + 1) * width)]
            XCTAssertLessThan(worst(linear(net, path)(x), before[path]! + product(x, qkvDown, rows, scale: 2 * 0.7)),
                              1e-5, path)
        }
        XCTAssertLessThan(worst(linear(net, output)(x), before[output]! + product(x, outDown, outUp, scale: 0.5 * 0.7)),
                          1e-5, "a diffusers name with its dots read as underscores reaches its layer")
        XCTAssertLessThan(worst(linear(net, modulation)(x), before[modulation]! + product(x, modDown, modUp, scale: 0.7)),
                          1e-5, "a file without alpha is scaled by 1")
    }

    func testAnAmbiguousOrUnknownLayerAndALyCORISFileAreRefusedBeforeAnyLayerChanges() throws {
        try requireMLXRuntime()
        let tree = NFKDiTAdapterTestTree(8)
        eval(tree)
        let layout = NFKMLXDiTAdapterLayout(family: "the tree")
        let (down, up) = halves(50, inputs: 8, outputs: 8)
        let refused: [(String, [String: MLXArray])] = [
            ("a kohya name two layers share", ["lora_unet_d.lora_down.weight": down, "lora_unet_d.lora_up.weight": up,
                                                "lora_unet_a_b_c.lora_down.weight": down,
                                                "lora_unet_a_b_c.lora_up.weight": up]),
            ("a layer the tree lacks", ["lora_unet_d.lora_down.weight": down, "lora_unet_d.lora_up.weight": up,
                                        "lora_unet_e.lora_down.weight": down, "lora_unet_e.lora_up.weight": up]),
            ("a diffusers layer the tree lacks", ["transformer.d.lora_A.weight": down, "transformer.d.lora_B.weight": up,
                                                  "transformer.e.lora_A.weight": down, "transformer.e.lora_B.weight": up]),
            ("a LoKr", ["lora_unet_d.lokr_w1": down, "lora_unet_d.lokr_w2": up]),
            ("a LoHa", ["lora_unet_d.hada_w1_a": down, "lora_unet_d.hada_w1_b": up]),
            ("a text encoder", ["lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": down,
                                "lora_unet_d.lora_down.weight": down, "lora_unet_d.lora_up.weight": up]),
            ("no layout", ["weird.weight": down]),
        ]
        for (reason, arrays) in refused {
            XCTAssertThrowsError(try NFKMLXDiTAdapters.hold([(try file(arrays), 1)], in: tree, layout: layout), reason)
        }
        for path in ["a.b_c", "a_b.c", "d"] {
            XCTAssertFalse(layers(tree)[path] is NFKMLXAdaptedLinear, "a refusal leaves \(path) as it was")
        }
    }

    // MARK: The families

    /// Holds a diffusers-named adapter on `diffusers` (the port's `path`) at strength 0, and checks that `forward`
    /// is unchanged at 0 and changed at a run strength of 1.
    private func assertTheAdapterReadsItsStrength(_ net: Module, diffusers: String, path: String, family: String,
                                                  hold: ([(url: URL, strength: Float)]) throws -> Void,
                                                  forward: () -> MLXArray) throws {
        let plain = forward()
        eval(plain)
        let weight = try XCTUnwrap(layers(net)[path] as? Linear, "\(family): \(path)").weight
        let (down, up) = halves(60, inputs: weight.dim(1), outputs: weight.dim(0))
        try hold([(try file(["transformer.\(diffusers).lora_A.weight": down,
                             "transformer.\(diffusers).lora_B.weight": up * 3]), 0)])
        XCTAssertTrue(layers(net)[path] is NFKMLXAdaptedLinear, "\(family): \(path) holds the adapter")
        XCTAssertEqual(worst(forward(), plain), 0, "\(family): a strength of 0 leaves the output as it was")
        let strengthened = NFKMLXAdapterRun.$strengths.withValue([1]) { forward() }
        XCTAssertGreaterThan(worst(strengthened, plain), 1e-4, "\(family): a strength of 1 changes the output")
    }

    func testFlux2HoldsADiffusersAdapterAtItsRunStrength() throws {
        try requireMLXRuntime()
        let net = flux2Net()
        let hidden = MLXRandom.normal([1, 4, 8], key: MLXRandom.key(70))
        let encoder = MLXRandom.normal([1, 3, 24], key: MLXRandom.key(71))
        try assertTheAdapterReadsItsStrength(net, diffusers: "transformer_blocks.0.attn.to_q",
                                             path: "transformer_blocks.0.attn.to_q", family: "FLUX.2",
                                             hold: { _ = try NFKMLXDiTAdapters.hold($0, in: net) }) {
            net(hidden, encoderHidden: encoder, timestep: MLXArray([Float(0.5)]), guidance: MLXArray([Float(4)]),
                imageIds: NFKMLXFlux2TransformerNet.imageIds(height: 2, width: 2),
                textIds: NFKMLXFlux2TransformerNet.textIds(length: 3))
        }
    }

    func testSD3HoldsADiffusersAdapterAtItsRunStrength() throws {
        try requireMLXRuntime()
        let net = NFKMLXSD3TransformerNet(.tiny)
        eval(net)
        let latent = MLXRandom.normal([1, 4, 8, 8], key: MLXRandom.key(72))
        let encoder = MLXRandom.normal([1, 7, 24], key: MLXRandom.key(73))
        let pooled = MLXRandom.normal([1, 20], key: MLXRandom.key(74))
        try assertTheAdapterReadsItsStrength(net, diffusers: "transformer_blocks.0.attn.to_q",
                                             path: "transformer_blocks.0.attn.to_q", family: "SD3",
                                             hold: { _ = try NFKMLXDiTAdapters.hold($0, in: net) }) {
            net(latent, encoderHidden: encoder, pooled: pooled, timestep: MLXArray([Float(600)]))
        }
    }

    func testSANAHoldsADiffusersAdapterAtItsRunStrength() throws {
        try requireMLXRuntime()
        let net = NFKMLXSANATransformerNet(.tiny)
        eval(net)
        let latent = MLXRandom.normal([8, 4, 4], key: MLXRandom.key(75))
        let caption = MLXRandom.normal([5, 12], key: MLXRandom.key(76))
        try assertTheAdapterReadsItsStrength(net, diffusers: "transformer_blocks.0.attn1.to_q",
                                             path: "transformer_blocks.0.attn1.to_q", family: "SANA",
                                             hold: { _ = try NFKMLXDiTAdapters.hold($0, in: net) }) {
            net(latent, capFeats: caption, t: MLXArray([Float(500)]))
        }
    }

    // A one-by-one convolution holds a convolution detour of the PyTorch-stored halves; a depthwise one is refused.
    func testASANAConvolutionHoldsAConvolutionDetourAndADepthwiseOneIsRefused() throws {
        try requireMLXRuntime()
        let net = NFKMLXSANATransformerNet(.tiny)
        eval(net)
        let path = "transformer_blocks.0.ff.conv_inverted"
        let convolution = try XCTUnwrap(layers(net)[path] as? Conv2d)
        let (outputs, inputs) = (convolution.weight.dim(0), convolution.weight.dim(3))
        let (down, up) = halves(80, inputs: inputs, outputs: outputs)
        let x = MLXRandom.normal([1, 4, 4, inputs], key: MLXRandom.key(81))
        let before = convolution(x)
        eval(before)
        let depth = "transformer_blocks.0.ff.conv_depth"
        let depthWeight = try XCTUnwrap(layers(net)[depth] as? Conv2d).weight
        let depthwise = try file([
            "transformer.\(depth).lora_A.weight": MLXArray.zeros([4, depthWeight.dim(3), 3, 3]),
            "transformer.\(depth).lora_B.weight": MLXArray.zeros([depthWeight.dim(0), 4, 1, 1]),
        ])
        XCTAssertThrowsError(try NFKMLXDiTAdapters.hold([(depthwise, 1)], in: net), "a grouped convolution")
        let url = try file(["transformer.\(path).lora_A.weight": down.reshaped([4, inputs, 1, 1]),
                            "transformer.\(path).lora_B.weight": up.reshaped([outputs, 4, 1, 1]),
                            "transformer.\(path).alpha": MLXArray(Float(2))])
        try NFKMLXDiTAdapters.hold([(url, 1)], in: net)
        let adapted = try XCTUnwrap(layers(net)[path] as? NFKMLXAdaptedConv2d)
        XCTAssertLessThan(worst(adapted(x), before + product(x, down, up, scale: 0.5)), 1e-5)
    }

    func testZImageHoldsADiffusersAdapterAtItsRunStrength() throws {
        try requireMLXRuntime()
        let net = NFKMLXZImageTransformerNet(.tiny)
        eval(net)
        let latent = MLXRandom.normal([4, 1, 4, 4], key: MLXRandom.key(77))
        let caption = MLXRandom.normal([6, 24], key: MLXRandom.key(78))
        try assertTheAdapterReadsItsStrength(net, diffusers: "layers.0.attention.to_q", path: "layers.0.attention.to_q",
                                             family: "Z-Image", hold: { _ = try NFKMLXDiTAdapters.hold($0, in: net) }) {
            net(latent, capFeats: caption, t: MLXArray([Float(0.5)]))
        }
    }

    // The original model names each attention's output projection `out`, which diffusers reads as `to_out.0`.
    func testZImageReadsTheOriginalModelsOutputProjection() throws {
        try requireMLXRuntime()
        let net = NFKMLXZImageTransformerNet(.tiny)
        eval(net)
        let path = "noise_refiner.0.attention.to_out.0"
        let weight = linear(net, path).weight
        let (down, up) = halves(90, inputs: weight.dim(1), outputs: weight.dim(0))
        let x = MLXRandom.normal([2, weight.dim(1)], key: MLXRandom.key(91))
        let before = linear(net, path)(x)
        eval(before)
        try NFKMLXDiTAdapters.hold([(try file(["lora_unet_noise_refiner_0_attention_out.lora_down.weight": down,
                                               "lora_unet_noise_refiner_0_attention_out.lora_up.weight": up,
                                               "lora_unet_noise_refiner_0_attention_out.alpha": MLXArray(Float(8))]),
                                     1)], in: net)
        XCTAssertLessThan(worst(linear(net, path)(x), before + product(x, down, up, scale: 2)), 1e-5)
    }

    func testLTXVideoHoldsADiffusersAdapterAtItsRunStrength() throws {
        try requireMLXRuntime()
        let net = NFKMLXLTXTransformer.makeNet(.tiny)
        eval(net)
        let latent = MLXRandom.normal([1, 8, 16], key: MLXRandom.key(79))
        let caption = MLXRandom.normal([1, 6, 32], key: MLXRandom.key(80))
        try assertTheAdapterReadsItsStrength(net, diffusers: "transformer_blocks.0.attn1.to_q",
                                             path: "transformer_blocks.0.attn1.to_q", family: "LTX-Video",
                                             hold: { _ = try NFKMLXDiTAdapters.hold($0, in: net) }) {
            net(latent, text: caption, timestep: MLXArray([Float(500)]), grid: (2, 2, 2), ropeScale: (1, 1, 1))
        }
    }

    // diffusers' `global_proj.0` is the port's `global_proj.layers.0`; `preprocess_conv` is refused.
    func testStableAudioHoldsADiffusersAdapterThroughItsRenamedProjection() throws {
        try requireMLXRuntime()
        let net = NFKMLXStableAudioDiTNet(.tiny)
        eval(net)
        let latents = MLXRandom.normal([1, 16, 4], key: MLXRandom.key(82))
        let context = MLXRandom.normal([1, 5, 12], key: MLXRandom.key(83))
        let global = MLXRandom.normal([1, 1, 24], key: MLXRandom.key(84))
        let convolution = try file(["transformer.preprocess_conv.lora_A.weight": MLXArray.zeros([4, 4, 1]),
                                    "transformer.preprocess_conv.lora_B.weight": MLXArray.zeros([4, 4, 1])])
        XCTAssertThrowsError(try NFKMLXDiTAdapters.hold([(convolution, 1)], in: net),
                             "the port holds the convolution as a linear layer")
        try assertTheAdapterReadsItsStrength(net, diffusers: "global_proj.0", path: "global_proj.layers.0",
                                             family: "Stable Audio Open",
                                             hold: { _ = try NFKMLXDiTAdapters.hold($0, in: net) }) {
            net(latents, timestep: MLXArray([Float(0.5)]), context: context, global: global)
        }
    }

    // MARK: The catalog

    func testTheDiffusionTransformerEntriesTakeAdaptersTrainedOnTheirReleases() throws {
        NFKMLXReferenceModels.registerAll()
        let descriptors = Dictionary(uniqueKeysWithValues: NFKMLXModelDescriptors.modelDescriptors().map { ($0.identifier, $0) })
        let expected: [String: [String]] = [
            NFKMLXFlux2.modelName: ["black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-base-4B"],
            "flux.2-klein-9b": ["black-forest-labs/FLUX.2-klein-9B", "black-forest-labs/FLUX.2-klein-base-9B",
                                "black-forest-labs/FLUX.2-klein-9b-kv"],
            NFKMLXSD3Generator.modelName: [NFKModelAdapterBaseStableDiffusion3],
            NFKMLXCatalogImage.sd35LargeTurboName: [NFKModelAdapterBaseStableDiffusion3],
            NFKMLXSANAGenerator.modelName: [NFKMLXSANAGenerator.releaseRepo],
            NFKMLXZImageGenerator.modelName: [NFKMLXZImageGenerator.releaseRepo, "Tongyi-MAI/Z-Image"],
            NFKMLXCatalogImage.zImageBaseName: [NFKMLXZImageGenerator.releaseRepo, "Tongyi-MAI/Z-Image"],
            NFKMLXLTXVideoGenerator.modelName: [NFKMLXLTXVideoGenerator.releaseRepo],
            NFKMLXStableAudioGenerator.modelName: [NFKMLXStableAudioGenerator.releaseRepo],
        ]
        for (identifier, repositories) in expected {
            let descriptor = try XCTUnwrap(descriptors[identifier], identifier)
            XCTAssertEqual(descriptor.adapterBases, repositories, identifier)
            XCTAssertEqual(descriptor.option(withKey: NFKModelLoadOptionAdapters)?.stage, NFKModelOptionStageLoad, identifier)
            XCTAssertNotNil(descriptor.option(withKey: NFKParameterAdapterStrengths), identifier)
            XCTAssertEqual(descriptor.memoryModel?.adaptersOptionKey, NFKModelLoadOptionAdapters, identifier)
            XCTAssertTrue(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: identifier)
                .contains(NFKModelLoadOptionAdapters), identifier)
        }
        XCTAssertEqual(try XCTUnwrap(descriptors[NFKMLXSD3ControlNetGenerator.modelName]).adapterBases, [],
                       "a ControlNet generator holds no adapters")
    }
}
