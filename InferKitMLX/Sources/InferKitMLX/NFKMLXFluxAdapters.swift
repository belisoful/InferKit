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

/// Reads FLUX.1 adapters and holds them unmerged beside the transformer's layers, through ``NFKMLXDiTAdapters``.
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
        try NFKMLXDiTAdapters.hold(adapters, in: transformer, layout: layout(
            doubleBlocks: transformer.transformerBlocks.count, singleBlocks: transformer.singleTransformerBlocks.count))
    }

    /// FLUX.1's layout over a transformer of `doubleBlocks` and `singleBlocks` blocks.
    static func layout(doubleBlocks: Int, singleBlocks: Int) -> NFKMLXDiTAdapterLayout {
        NFKMLXDiTAdapterLayout(family: "FLUX.1",
                               original: originalLayers(doubleBlocks: doubleBlocks, singleBlocks: singleBlocks),
                               leftOut: { $0.hasPrefix("time_text_embed.guidance_embedder.") },
                               swapsHalves: ["norm_out.linear"])
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
}
