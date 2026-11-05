//
//  NFKMLXSDAdapterTests.swift
//  InferKitMLXTests
//
//  Released Stable Diffusion LoRAs held beside a tiny model's weights: unmerged detours against the merged
//  weights they stand for, per-run strengths, stacking, the text towers' fused projection, the original
//  block names, and the load option. Weight-free: released adapters are measured in
//  NFKMLXReferenceParityTests.
//

import CoreGraphics
import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXSDAdapterTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func temporaryAdapter(_ arrays: [String: MLXArray], metadata: [String: String] = [:]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXSDAdapterTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("adapter.safetensors")
        try save(arrays: arrays, metadata: metadata, url: url)
        return url
    }

    /// A tiny UNet and a second one holding a copy of its weights.
    private func twinUNets() -> (NFKMLXSDUNet, NFKMLXSDUNet) {
        let first = NFKMLXSDUNet(configuration: NFKMLXSDTextToImageConfiguration.tiny.unet)
        first.train(false)
        eval(first)
        let second = NFKMLXSDUNet(configuration: NFKMLXSDTextToImageConfiguration.tiny.unet)
        second.train(false)
        second.update(parameters: first.parameters().mapValues { $0 + 0 })
        eval(second)
        return (first, second)
    }

    private func unetOutput(_ unet: NFKMLXSDUNet, strengths: [Float]? = nil) -> MLXArray {
        let latent = MLXRandom.normal([1, 8, 8, 4], key: MLXRandom.key(51))
        let context = MLXRandom.normal([1, 5, 32], key: MLXRandom.key(52))
        let output = NFKMLXAdapterRun.$strengths.withValue(strengths) {
            unet(latent, timestep: MLXArray([Int32(499)]), context: context)
        }
        eval(output)
        return output
    }

    /// A kohya adapter over a linear target, a 3×3 convolution, and the 1×1 projection, named the
    /// diffusers way, at `seed`.
    private func kohyaArrays(for unet: NFKMLXSDUNet, rank: Int = 2, alpha: Float = 1, seed: UInt64) -> [String: MLXArray] {
        let layers = Dictionary(uniqueKeysWithValues: unet.leafModules().flattened())
        let linear = (layers["down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k"] as! Linear).weight
        let conv = (layers["down_blocks.0.resnets.0.conv1"] as! Conv2d).weight
        let projection = (layers["down_blocks.0.attentions.0.proj_in_conv"] as! Conv2d).weight
        func key(_ offset: UInt64) -> MLXArray { MLXRandom.key(seed * 10 + offset) }
        return [
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight":
                MLXRandom.normal([rank, linear.dim(1)], key: key(1)),
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight":
                MLXRandom.normal([linear.dim(0), rank], key: key(2)),
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.alpha": MLXArray(alpha),
            "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight":
                MLXRandom.normal([rank, conv.dim(3), 3, 3], key: key(3)) * 0.1,
            "lora_unet_down_blocks_0_resnets_0_conv1.lora_up.weight":
                MLXRandom.normal([conv.dim(0), rank, 1, 1], key: key(4)) * 0.1,
            "lora_unet_down_blocks_0_resnets_0_conv1.alpha": MLXArray(alpha),
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight":
                MLXRandom.normal([rank, projection.dim(3), 1, 1], key: key(5)),
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight":
                MLXRandom.normal([projection.dim(0), rank, 1, 1], key: key(6)),
            "lora_unet_down_blocks_0_attentions_0_proj_in.alpha": MLXArray(alpha),
        ]
    }

    private func maximumDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    // MARK: Unmerged against merged

    // Held beside the weights, an adapter computes what its merge computes: the detours over a linear
    // layer, a 3×3 convolution, and a 1×1 projection, at the loaded strength and at a run's.
    func testAnUnmergedAdapterMatchesItsMerge() throws {
        try requireMLXRuntime()
        let (merged, held) = twinUNets()
        let base = unetOutput(held)
        let url = try temporaryAdapter(kohyaArrays(for: held, seed: 1))
        try NFKMLXSDLoRA.merge(adapterURL: url, into: merged, scale: 0.7)
        let report = try NFKMLXSDLoRA.attach([(url, 0.7)], unet: held, textEncoder: nil, secondaryTextEncoder: nil)
        XCTAssertEqual(report.adaptedLayers[.unet], 3)
        XCTAssertGreaterThan(report.byteCount, 0)

        let expected = unetOutput(merged)
        XCTAssertLessThan(maximumDifference(unetOutput(held), expected), 1e-4)
        XCTAssertGreaterThan(maximumDifference(expected, base), 1e-3, "the adapter changes the output")
        XCTAssertLessThan(maximumDifference(unetOutput(held, strengths: [0]), base), 1e-6,
                          "a run at strength 0 is the base model")
        XCTAssertLessThan(maximumDifference(unetOutput(held, strengths: [0.7]), expected), 1e-4)
    }

    // Two adapters stack in load order, each at its own strength, as two merges one after the other do.
    func testAdaptersStackEachAtItsOwnStrength() throws {
        try requireMLXRuntime()
        let (merged, held) = twinUNets()
        let first = try temporaryAdapter(kohyaArrays(for: held, seed: 2))
        let second = try temporaryAdapter(kohyaArrays(for: held, rank: 3, alpha: 6, seed: 3))
        try NFKMLXSDLoRA.merge(adapterURL: first, into: merged, scale: 0.5)
        try NFKMLXSDLoRA.merge(adapterURL: second, into: merged, scale: 1.2)
        try NFKMLXSDLoRA.attach([(first, 1), (second, 1)], unet: held, textEncoder: nil, secondaryTextEncoder: nil)
        XCTAssertLessThan(maximumDifference(unetOutput(held, strengths: [0.5, 1.2]), unetOutput(merged)), 1e-4)
    }

    // A PEFT adapter with diffusers' recorded configuration scales by lora_alpha / r; without the record
    // it scales by 1.
    func testPEFTMetadataSetsTheScaling() throws {
        try requireMLXRuntime()
        let (merged, held) = twinUNets()
        let layers = Dictionary(uniqueKeysWithValues: held.leafModules().flattened())
        let weight = (layers["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v"] as! Linear).weight
        let down = MLXRandom.normal([4, weight.dim(1)], key: MLXRandom.key(61))
        let up = MLXRandom.normal([weight.dim(0), 4], key: MLXRandom.key(62))
        let arrays = ["unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora_A.weight": down,
                      "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora_B.weight": up]
        let metadata = ["lora_adapter_metadata": #"{"unet.r": 4, "unet.lora_alpha": 2}"#]
        let url = try temporaryAdapter(arrays, metadata: metadata)
        try NFKMLXSDLoRA.merge(adapterURL: try temporaryAdapter(arrays), into: merged, scale: 0.5)
        try NFKMLXSDLoRA.attach([(url, 1)], unet: held, textEncoder: nil, secondaryTextEncoder: nil)
        XCTAssertLessThan(maximumDifference(unetOutput(held), unetOutput(merged)), 1e-4,
                          "lora_alpha 2 over r 4 is a merge without metadata at 0.5")
    }

    // MARK: The text towers

    // The query, key, and value adapters of a CLIP layer reach the parts of the fused input projection,
    // and the output projection and MLP adapters reach their layers: the tower encodes as one whose
    // weights took the same deltas.
    func testTextTowerAdaptersReachTheFusedProjection() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXSDTextToImageConfiguration.tiny.textEncoder
        let held = NFKMLXSDTextEncoderNet(configuration: configuration)
        held.train(false)
        eval(held)
        let merged = NFKMLXSDTextEncoderNet(configuration: configuration)
        merged.train(false)
        merged.update(parameters: held.parameters().mapValues { $0 + 0 })
        let width = configuration.width
        let rank = 2
        var arrays = [String: MLXArray]()
        var deltas = [String: MLXArray]()
        for (offset, name) in ["q_proj", "k_proj", "v_proj", "out_proj"].enumerated() {
            let down = MLXRandom.normal([rank, width], key: MLXRandom.key(70 + UInt64(offset)))
            let up = MLXRandom.normal([width, rank], key: MLXRandom.key(80 + UInt64(offset)))
            arrays["lora_te_text_model_encoder_layers_0_self_attn_\(name).lora_down.weight"] = down
            arrays["lora_te_text_model_encoder_layers_0_self_attn_\(name).lora_up.weight"] = up
            arrays["lora_te_text_model_encoder_layers_0_self_attn_\(name).alpha"] = MLXArray(Float(1))
            deltas[name] = matmul(up, down) * 0.5
        }
        let fc1 = held.transformer.resblocks[0].mlp.cFc.weight
        let fc1Down = MLXRandom.normal([rank, fc1.dim(1)], key: MLXRandom.key(90))
        let fc1Up = MLXRandom.normal([fc1.dim(0), rank], key: MLXRandom.key(91))
        arrays["text_encoder.text_model.encoder.layers.0.mlp.fc1.lora_A.weight"] = fc1Down
        arrays["text_encoder.text_model.encoder.layers.0.mlp.fc1.lora_B.weight"] = fc1Up

        let attention = merged.transformer.resblocks[0].attn
        let fused = attention.inProjWeight + concatenated([deltas["q_proj"]!, deltas["k_proj"]!, deltas["v_proj"]!], axis: 0)
        attention.update(parameters: ModuleParameters.unflattened(["in_proj_weight": fused]))
        let outProj = attention.outProj
        outProj.update(parameters: ModuleParameters.unflattened(["weight": outProj.weight + deltas["out_proj"]!]))
        let mlp = merged.transformer.resblocks[0].mlp.cFc
        mlp.update(parameters: ModuleParameters.unflattened(["weight": mlp.weight + matmul(fc1Up, fc1Down)]))

        let unet = NFKMLXSDUNet(configuration: NFKMLXSDTextToImageConfiguration.tiny.unet)
        let report = try NFKMLXSDLoRA.attach([(try temporaryAdapter(arrays), 1)], unet: unet, textEncoder: held,
                                             secondaryTextEncoder: nil)
        XCTAssertEqual(report.adaptedLayers[.textEncoder], 5)
        let tokens = [1, 5, 9, 3] + Array(repeating: 0, count: configuration.contextLength - 4)
        let expected = merged.encode(tokens).hidden
        let actual = held.encode(tokens).hidden
        eval(expected, actual)
        XCTAssertLessThan(maximumDifference(actual, expected), 1e-4)
    }

    // An adapter for a second text tower on a one-tower model, or a layer the model lacks, is refused
    // before any layer changes; so are the LoHa and DoRA layouts.
    func testAnAdapterForAnotherReleaseIsRefused() throws {
        try requireMLXRuntime()
        let unet = NFKMLXSDUNet(configuration: NFKMLXSDTextToImageConfiguration.tiny.unet)
        let tower = NFKMLXSDTextEncoderNet(configuration: NFKMLXSDTextToImageConfiguration.tiny.textEncoder)
        let secondTower = try temporaryAdapter([
            "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": MLXArray.zeros([2, 32]),
            "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": MLXArray.zeros([128, 2]),
        ])
        XCTAssertThrowsError(try NFKMLXSDLoRA.attach([(secondTower, 1)], unet: unet, textEncoder: tower,
                                                     secondaryTextEncoder: nil)) { error in
            XCTAssertTrue("\(error)".contains("second text tower"), "\(error)")
        }
        let missing = try temporaryAdapter([
            "lora_unet_input_blocks_7_1_transformer_blocks_0_attn1_to_q.lora_down.weight": MLXArray.zeros([2, 64]),
            "lora_unet_input_blocks_7_1_transformer_blocks_0_attn1_to_q.lora_up.weight": MLXArray.zeros([64, 2]),
        ])
        XCTAssertThrowsError(try NFKMLXSDLoRA.attach([(missing, 1)], unet: unet, textEncoder: tower,
                                                     secondaryTextEncoder: nil)) { error in
            XCTAssertTrue("\(error)".contains("input_blocks_7_1"), "\(error)")
        }
        XCTAssertFalse(unet.leafModules().flattened().contains { $0.1 is NFKMLXAdaptedLinear },
                       "a refused adapter changes no layer")
        let loha = try temporaryAdapter(["lora_unet_down_blocks_0_resnets_0_conv1.hada_w1_a": MLXArray.zeros([2, 2])])
        XCTAssertThrowsError(try NFKMLXSDLoRA.attach([(loha, 1)], unet: unet, textEncoder: tower,
                                                     secondaryTextEncoder: nil)) { error in
            XCTAssertTrue("\(error)".contains("LoHa"), "\(error)")
        }
    }

    // MARK: The original block names

    // kohya's SDXL files number the original model's input and output blocks; each translates to the
    // diffusers layer at the same place, for the residual blocks' layers, the transformers, and the
    // samplers.
    func testOriginalBlockNamesTranslateForEachRelease() {
        let sdxl = NFKMLXSDUNetConfiguration.sdxl
        let sd15 = NFKMLXSDTextToImageConfiguration.stableDiffusion15.unet
        let cases: [(String, NFKMLXSDUNetConfiguration, String)] = [
            ("down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q", sdxl,
             "input_blocks.4.1.transformer_blocks.0.attn1.to_q"),
            ("down_blocks.2.attentions.1.proj_out", sdxl, "input_blocks.8.1.proj_out"),
            ("down_blocks.0.resnets.0.time_emb_proj", sdxl, "input_blocks.1.0.emb_layers.1"),
            ("down_blocks.1.resnets.0.conv_shortcut", sdxl, "input_blocks.4.0.skip_connection"),
            ("down_blocks.1.downsamplers.0.conv", sdxl, "input_blocks.6.0.op"),
            ("mid_block.resnets.1.conv2", sdxl, "middle_block.2.out_layers.3"),
            ("mid_block.attentions.0.transformer_blocks.9.ff.net.2", sdxl, "middle_block.1.transformer_blocks.9.ff.net.2"),
            ("up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_out.0", sdxl,
             "output_blocks.2.1.transformer_blocks.0.attn2.to_out.0"),
            ("up_blocks.0.upsamplers.0.conv", sdxl, "output_blocks.2.2.conv"),
            ("up_blocks.2.resnets.2.conv1", sdxl, "output_blocks.8.0.in_layers.2"),
            ("add_embedding.linear_2", sdxl, "label_emb.0.2"),
            ("time_embedding.linear_1", sdxl, "time_embed.0"),
            ("conv_in", sdxl, "input_blocks.0.0"),
            ("conv_out", sdxl, "out.2"),
            ("up_blocks.0.upsamplers.0.conv", sd15, "output_blocks.2.1.conv"),
            ("up_blocks.1.upsamplers.0.conv", sd15, "output_blocks.5.2.conv"),
            ("down_blocks.2.attentions.1.proj_in", sd15, "input_blocks.8.1.proj_in"),
            ("down_blocks.3.resnets.1.conv2", sd15, "input_blocks.11.0.out_layers.3"),
            ("up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k", sd15,
             "output_blocks.11.1.transformer_blocks.0.attn1.to_k"),
        ]
        for (diffusers, configuration, original) in cases {
            XCTAssertEqual(NFKMLXSDLoRA.originalPath(diffusers, configuration: configuration), original, diffusers)
        }
    }

    // Every adaptable layer of a UNet has one original name, and no two layers share one.
    func testEveryUNetLayerHasItsOwnOriginalName() {
        let unet = NFKMLXSDUNet(configuration: NFKMLXSDTextToImageConfiguration.tiny.unet)
        var seen = [String: String]()
        for (path, module) in unet.leafModules().flattened() where module is Linear || module is Conv2d {
            let diffusers = NFKMLXSDLoRA.diffusersPath(path)
            guard let original = NFKMLXSDLoRA.originalPath(diffusers, configuration: unet.configuration) else {
                XCTFail("\(diffusers) has no original name")
                continue
            }
            XCTAssertNil(seen[original], "\(diffusers) and \(seen[original] ?? "") share \(original)")
            seen[original] = diffusers
        }
        XCTAssertGreaterThan(seen.count, 20)
    }

    // An adapter spelled with the original block names merges and attaches as the diffusers-named one does.
    func testOriginalBlockNamesReachTheSameLayers() throws {
        try requireMLXRuntime()
        let (diffusersNamed, originalNamed) = twinUNets()
        let arrays = kohyaArrays(for: diffusersNamed, seed: 4)
        var renamed = [String: MLXArray]()
        for (key, value) in arrays {
            renamed[key.replacingOccurrences(of: "down_blocks_0_attentions_0", with: "input_blocks_1_1")
                .replacingOccurrences(of: "down_blocks_0_resnets_0_conv1", with: "input_blocks_1_0_in_layers_2")] = value
        }
        try NFKMLXSDLoRA.attach([(try temporaryAdapter(arrays), 1)], unet: diffusersNamed, textEncoder: nil,
                                secondaryTextEncoder: nil)
        try NFKMLXSDLoRA.attach([(try temporaryAdapter(renamed), 1)], unet: originalNamed, textEncoder: nil,
                                secondaryTextEncoder: nil)
        XCTAssertEqual(maximumDifference(unetOutput(diffusersNamed), unetOutput(originalNamed)), 0)
    }

    // MARK: The backend

    // A backend holding an adapter lists the strengths key, and a run sets the strengths: 0 draws the base
    // model's picture, and a different strength a different one. A malformed value is refused.
    func testABackendTakesTheStrengthsPerRun() throws {
        try requireMLXRuntime()
        let model = NFKMLXSDTextToImageModel(configuration: .tiny.withSampler(.ddimTrailing, steps: 2))
        eval(model.pipeline, model.textEncoder)
        func picture(_ backend: NFKInferenceBackend, _ strengths: [Float]?) throws -> [UInt8] {
            var parameters: [String: Any] = [NFKParameterSeed: 3]
            if let strengths { parameters[NFKParameterAdapterStrengths] = strengths }
            let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "a tiny test"],
                                                                           parameters: parameters))
            let image = try XCTUnwrap(result.output(forKey: NFKOutputImage)) as! CGImage
            return NFKMLXImageBridge.rgbaBytes(from: image, colorSpace: CGColorSpaceCreateDeviceRGB()).bytes
        }
        let plain = model.makeBackend()
        XCTAssertFalse(plain.supportedParameterKeys.contains(NFKParameterAdapterStrengths))
        let base = try picture(plain, nil)

        var arrays = kohyaArrays(for: model.pipeline.unet, seed: 5)
        for (key, value) in arrays where key.hasSuffix("lora_up.weight") {
            arrays[key] = value * 20
        }
        try model.attachAdapters([(try temporaryAdapter(arrays), 1)])
        let adapted = model.makeBackend()
        XCTAssertTrue(adapted.supportedParameterKeys.contains(NFKParameterAdapterStrengths))
        XCTAssertEqual(try picture(adapted, [0]), base)
        XCTAssertNotEqual(try picture(adapted, nil), base)
        XCTAssertEqual(try picture(adapted, nil), try picture(adapted, [1]))
        XCTAssertThrowsError(try picture(adapted, [1, 1]), "more strengths than adapters")
    }

    // The load option lists the adapters, each strength 1 where the entry names none.
    func testTheLoadOptionReadsEachAdapter() throws {
        let options = NFKMLXLoadOptions([NFKModelLoadOptionAdapters: [
            [NFKModelAdapterPath: "/tmp/a.safetensors", NFKModelAdapterStrength: 0.8],
            [NFKModelAdapterPath: "/tmp/b.safetensors"],
        ]])
        let adapters = try options.adapters()
        XCTAssertEqual(adapters.map(\.url.path), ["/tmp/a.safetensors", "/tmp/b.safetensors"])
        XCTAssertEqual(adapters.map(\.strength), [0.8, 1])
        XCTAssertTrue(try NFKMLXLoadOptions().adapters().isEmpty)
        XCTAssertThrowsError(try NFKMLXLoadOptions([NFKModelLoadOptionAdapters: [["strength": 1]]]).adapters())
    }

    // Each Stable Diffusion entry takes adapters of its own base, and declares the option, the strengths
    // parameter, and the memory model's key.
    func testStableDiffusionEntriesDeclareAdapters() throws {
        NFKMLXReferenceModels.registerAll()
        let descriptors = Dictionary(uniqueKeysWithValues: NFKMLXModelDescriptors.modelDescriptors().map { ($0.identifier, $0) })
        let expected = ["stable-diffusion-v1-5": NFKModelAdapterBaseStableDiffusion1,
                        "stable-diffusion-2-1-base": NFKModelAdapterBaseStableDiffusion2,
                        "stable-diffusion-xl-base-1.0": NFKModelAdapterBaseStableDiffusionXL,
                        "sdxl-turbo": NFKModelAdapterBaseStableDiffusionXL,
                        "lcm-dreamshaper-v7": NFKModelAdapterBaseStableDiffusion1,
                        "lcm-lora-sdv1-5": NFKModelAdapterBaseStableDiffusion1,
                        "sdxl-lightning-4step-lora": NFKModelAdapterBaseStableDiffusionXL]
        for (identifier, base) in expected {
            let descriptor = try XCTUnwrap(descriptors[identifier], identifier)
            XCTAssertEqual(descriptor.adapterBases, [base], identifier)
            XCTAssertEqual(descriptor.option(withKey: NFKModelLoadOptionAdapters)?.stage, NFKModelOptionStageLoad, identifier)
            XCTAssertNotNil(descriptor.option(withKey: NFKParameterAdapterStrengths), identifier)
            XCTAssertEqual(descriptor.memoryModel?.adaptersOptionKey, NFKModelLoadOptionAdapters, identifier)
            XCTAssertTrue(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: identifier)
                .contains(NFKModelLoadOptionAdapters), identifier)
        }
        for descriptor in descriptors.values where descriptor.family != "stable-diffusion" {
            XCTAssertEqual(descriptor.adapterBases, [], "\(descriptor.identifier) takes no adapters yet")
        }
    }

    // The core's inspection reads an SDXL adapter spelled with the original block names against the
    // package's descriptors: the SDXL entries take it, and an SD 1.5 entry names them.
    func testInspectionFindsTheEntriesThatTakeAnAdapter() throws {
        let catalog = NFKModelCatalog()
        for descriptor in NFKMLXModelDescriptors.modelDescriptors() {
            catalog.addDescriptor(descriptor)
        }
        let url = try temporaryAdapter([
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight": MLXArray.zeros([8, 2048]),
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_up.weight": MLXArray.zeros([640, 8]),
            "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.alpha": MLXArray(Float(8)),
            "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": MLXArray.zeros([8, 1280]),
            "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": MLXArray.zeros([5120, 8]),
        ], metadata: ["modelspec.trigger_phrase": "watercolor, soft"])
        let report = try catalog.inspectAdapter(at: url, forModelIdentifier: "stable-diffusion-v1-5")
        XCTAssertEqual(report.layout, NFKModelAdapterLayoutKohya)
        XCTAssertEqual(report.base, NFKModelAdapterBaseStableDiffusionXL)
        XCTAssertEqual(report.rank, 8)
        XCTAssertEqual(report.triggerWords, ["watercolor", "soft"])
        XCTAssertEqual(report.fit, .fitsOtherModels)
        XCTAssertTrue(Set(report.fittingModelIdentifiers).isSuperset(of: ["stable-diffusion-xl-base-1.0", "sdxl-turbo"]),
                      "\(report.fittingModelIdentifiers)")
        XCTAssertEqual(try catalog.inspectAdapter(at: url, forModelIdentifier: "sdxl-turbo").fit, .fits)
    }
}
