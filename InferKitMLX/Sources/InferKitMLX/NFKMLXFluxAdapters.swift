//
//  NFKMLXFluxAdapters.swift
//  InferKitMLX
//
//  Released FLUX.1 LoRA adapters held beside the transformer's linear layers rather than merged into them, read
//  in the three layouts FLUX adapters are published in: diffusers', kohya's, and Black Forest Labs' own.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// Reads FLUX.1 adapters and holds them unmerged beside the transformer's layers.
///
/// @discussion The transformer is named as diffusers names it, so a diffusers file (`transformer.<layer>.lora_A.weight`
/// and `lora_B.weight`) reaches its layers directly, scaled by the `lora_adapter_metadata` it carries or by a layer's
/// `alpha`, else 1. A kohya file (`lora_unet_<layer>.lora_down.weight`, `lora_up.weight`, `alpha`) and a Black
/// Forest Labs file (`diffusion_model.<layer>.lora_A.weight`) name the original model's layers, which fuse some of
/// diffusers' layers:
/// - a double block's `img_attn.qkv` is `attn.to_q`, `to_k`, and `to_v`, and its `txt_attn.qkv` the `add_` three;
/// - a single block's `linear1` is `attn.to_q`, `to_k`, `to_v`, and `proj_mlp`;
/// - the final layer's `adaLN_modulation.1` is `norm_out.linear` with its shift and scale halves in the other order.
///
/// A fused layer's adapter shares its down half among the parts and splits its up half by each part's outputs. The
/// guidance embedding's adapter is left out of a transformer without one ([schnell]). A text encoder's adapter, an
/// XLabs file, or any other layer the transformer lacks is refused before any layer changes.
enum NFKMLXFluxAdapters {

    /// Holds each adapter beside `transformer`'s layers at its strength, in load order, and returns the strengths.
    @discardableResult
    static func hold(_ adapters: [(url: URL, strength: Float)], in transformer: NFKMLXFluxTransformerNet) throws -> [Float] {
        guard !adapters.isEmpty else { return [] }
        let layers = Dictionary(transformer.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        let original = originalLayers(doubleBlocks: transformer.transformerBlocks.count,
                                      singleBlocks: transformer.singleTransformerBlocks.count)
        var detours = [String: [NFKMLXLowRankDetour]]()
        for (index, adapter) in adapters.enumerated() {
            for (path, detour) in try read(adapter.url, adapter: index, strength: adapter.strength, layers: layers,
                                           original: original) {
                detours[path, default: []].append(detour)
            }
        }
        try NFKMLXAdapterLayers.attach(detours, to: transformer)
        return adapters.map(\.strength)
    }

    /// Each original-model layer name, dotted as Black Forest Labs writes it, and the diffusers layers it is.
    static func originalLayers(doubleBlocks: Int, singleBlocks: Int) -> [String: [String]] {
        var layers: [String: [String]] = [
            "img_in": ["x_embedder"], "txt_in": ["context_embedder"],
            "time_in.in_layer": ["time_text_embed.timestep_embedder.linear_1"],
            "time_in.out_layer": ["time_text_embed.timestep_embedder.linear_2"],
            "vector_in.in_layer": ["time_text_embed.text_embedder.linear_1"],
            "vector_in.out_layer": ["time_text_embed.text_embedder.linear_2"],
            "guidance_in.in_layer": ["time_text_embed.guidance_embedder.linear_1"],
            "guidance_in.out_layer": ["time_text_embed.guidance_embedder.linear_2"],
            "final_layer.linear": ["proj_out"], "final_layer.adaLN_modulation.1": ["norm_out.linear"],
        ]
        for block in 0 ..< doubleBlocks {
            let (from, to) = ("double_blocks.\(block).", "transformer_blocks.\(block).")
            layers[from + "img_attn.qkv"] = ["attn.to_q", "attn.to_k", "attn.to_v"].map { to + $0 }
            layers[from + "img_attn.proj"] = [to + "attn.to_out.0"]
            layers[from + "txt_attn.qkv"] = ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"].map { to + $0 }
            layers[from + "txt_attn.proj"] = [to + "attn.to_add_out"]
            layers[from + "img_mlp.0"] = [to + "ff.net.0.proj"]
            layers[from + "img_mlp.2"] = [to + "ff.net.2"]
            layers[from + "txt_mlp.0"] = [to + "ff_context.net.0.proj"]
            layers[from + "txt_mlp.2"] = [to + "ff_context.net.2"]
            layers[from + "img_mod.lin"] = [to + "norm1.linear"]
            layers[from + "txt_mod.lin"] = [to + "norm1_context.linear"]
        }
        for block in 0 ..< singleBlocks {
            let (from, to) = ("single_blocks.\(block).", "single_transformer_blocks.\(block).")
            layers[from + "linear1"] = ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"].map { to + $0 }
            layers[from + "linear2"] = [to + "proj_out"]
            layers[from + "modulation.lin"] = [to + "norm.linear"]
        }
        return layers
    }

    /// The detours one adapter file places, by layer path.
    static func read(_ url: URL, adapter: Int, strength: Float, layers: [String: Module],
                     original: [String: [String]]) throws -> [(String, NFKMLXLowRankDetour)] {
        let name = url.lastPathComponent
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        let keys = arrays.keys
        if keys.contains(where: { $0.hasPrefix("lora_te") || $0.hasPrefix("text_encoder") }) {
            throw NFKMLXError.loRANotApplicable("\(name) adapts a text encoder, which FLUX holds no adapter beside")
        }
        if keys.contains(where: { $0.contains(".processor.") }) {
            throw NFKMLXError.loRANotApplicable("\(name) is an XLabs adapter, a layout InferKit does not read")
        }
        if keys.contains(where: { $0.contains("lora_magnitude_vector") || $0.contains("dora_scale") }) {
            throw NFKMLXError.loRANotApplicable("\(name) is a DoRA, which a held adapter does not compute")
        }
        var halves = [(stored: String, down: MLXArray, up: MLXArray, targets: [String])]()
        if keys.contains(where: { $0.hasPrefix("transformer.") && $0.hasSuffix(".lora_A.weight") }) {
            for key in keys.sorted() where key.hasPrefix("transformer.") && key.hasSuffix(".lora_A.weight") {
                let stored = String(key.dropLast(".lora_A.weight".count))
                halves.append((stored, arrays[key]!, try pair(stored + ".lora_B.weight", in: arrays, name: name),
                               [String(stored.dropFirst("transformer.".count))]))
            }
        } else if keys.contains(where: { $0.hasPrefix("lora_unet_") }) {
            let kohya = Dictionary(uniqueKeysWithValues: original.map { ("lora_unet_" + $0.key.replacingOccurrences(of: ".", with: "_"), $0.value) })
            for key in keys.sorted() where key.hasPrefix("lora_unet_") && key.hasSuffix(".lora_down.weight") {
                let stored = String(key.dropLast(".lora_down.weight".count))
                guard let targets = kohya[stored] else {
                    throw NFKMLXError.loRANotApplicable("\(name) adapts \(stored), which names no layer of FLUX.1")
                }
                halves.append((stored, arrays[key]!, try pair(stored + ".lora_up.weight", in: arrays, name: name), targets))
            }
        } else if keys.contains(where: { $0.hasPrefix("diffusion_model.") }) {
            for key in keys.sorted() where key.hasPrefix("diffusion_model.")
                && (key.hasSuffix(".lora_A.weight") || key.hasSuffix(".lora_down.weight")) {
                let peft = key.hasSuffix(".lora_A.weight")
                let stored = String(key.dropLast((peft ? ".lora_A.weight" : ".lora_down.weight").count))
                guard let targets = original[String(stored.dropFirst("diffusion_model.".count))] else {
                    throw NFKMLXError.loRANotApplicable("\(name) adapts \(stored), which names no layer of FLUX.1")
                }
                halves.append((stored, arrays[key]!, try pair(stored + (peft ? ".lora_B.weight" : ".lora_up.weight"),
                                                              in: arrays, name: name), targets))
            }
        } else {
            throw NFKMLXError.loRANotApplicable(
                "\(name) holds no FLUX.1 LoRA tensors in the diffusers, kohya, or Black Forest Labs layout")
        }
        guard !halves.isEmpty else {
            throw NFKMLXError.loRANotApplicable("\(name) adapts no layer")
        }

        let peft = try NFKMLXWanLoRA.peftConfiguration(metadata, component: "transformer.")
        var placed = [(String, NFKMLXLowRankDetour)]()
        // An adapter trained on [dev] adapts its guidance embedding, which [schnell] does not have; that part is
        // left out, as diffusers leaves it out. Any other layer the transformer lacks is refused.
        let guidance = "time_text_embed.guidance_embedder."
        let embedsGuidance = layers.keys.contains { $0.hasPrefix(guidance) }
        for half in halves where embedsGuidance || !half.targets.allSatisfy({ $0.hasPrefix(guidance) }) {
            let rank = half.down.dim(0)
            let scale: Double
            if let alpha = arrays[half.stored + ".alpha"] {
                scale = Double(alpha.asType(.float32).item(Float.self)) / Double(rank)
            } else {
                scale = half.targets.count == 1 ? peft.preciseScaling(half.targets[0], rank: rank) ?? 1 : 1
            }
            var up = half.up
            if half.targets == ["norm_out.linear"] && !half.stored.hasPrefix("transformer.") {
                let width = up.dim(0) / 2
                up = concatenated([up[width...], up[..<width]], axis: 0)
            }
            var row = 0
            for target in half.targets {
                let outputs = try outputCount(of: target, in: layers, name: name)
                guard row + outputs <= up.dim(0) else {
                    throw NFKMLXError.loRANotApplicable(
                        "\(name) adapts \(half.stored) with \(up.dim(0)) outputs, fewer than the layers it fuses")
                }
                placed.append((target, try NFKMLXDecoderAdapters.detour(
                    down: half.down, up: up[row ..< row + outputs], scale: scale, adapter: adapter, strength: strength,
                    path: target, layers: layers, name: name)))
                row += outputs
            }
            guard row == up.dim(0) else {
                throw NFKMLXError.loRANotApplicable(
                    "\(name) adapts \(half.stored) with \(up.dim(0)) outputs, where the layers it names take \(row)")
            }
        }
        return placed
    }

    private static func pair(_ key: String, in arrays: [String: MLXArray], name: String) throws -> MLXArray {
        guard let array = arrays[key] else {
            throw NFKMLXError.loRANotApplicable("\(name) holds a down half without \(key)")
        }
        return array
    }

    private static func outputCount(of path: String, in layers: [String: Module], name: String) throws -> Int {
        switch layers[path] {
        case let quantized as QuantizedLinear: return quantized.shape.0
        case let linear as Linear: return linear.weight.dim(0)
        default: throw NFKMLXError.loRANotApplicable("\(name) adapts \(path), which is not a linear layer of FLUX.1")
        }
    }
}
