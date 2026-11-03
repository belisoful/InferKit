//
//  NFKMLXLTX2LoRA.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// LoRAs for the LTX-2 transformer in Lightricks' own key format (`diffusion_model.<module>.lora_A.weight`),
/// such as LTX-2.5's spatial IC-LoRA, placed as adapters whose weight a caller sets per pass.
///
/// @discussion The keys are renamed as diffusers' `_convert_non_diffusers_ltx2_lora_to_diffusers` renames them
/// (`q_norm` → `norm_q`, `adaln_single` → `time_embed`, …). A module's detour is `B (A x) · alpha / rank ·
/// strength`; a release without an `alpha` takes `alpha = rank`, as peft does. Each adapted layer is an
/// ``NFKMLXLoRALinear`` over a float base or an ``NFKMLXQuantizedLoRALinear`` over a quantized one, so the base
/// weights stay as loaded and ``setStrength(_:in:)`` turns the adapter off at 0.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXLTX2LoRA {

    static let renames: [(String, String)] = [
        ("audio_patchify_proj", "audio_proj_in"), ("patchify_proj", "proj_in"),
        ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
        ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
        ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
        ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
        ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
        ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
        ("q_norm", "norm_q"), ("k_norm", "norm_k"),
        ("audio_prompt_adaln_single", "audio_prompt_adaln"), ("prompt_adaln_single", "prompt_adaln"),
    ]

    /// The transformer module a release key's prefix names, or nil for a key outside the transformer.
    public static func modulePath(_ key: String) -> String? {
        guard key.hasPrefix("diffusion_model.") else { return nil }
        var path = String(key.dropFirst("diffusion_model.".count))
        // diffusers renames in dictionary order; `patchify_proj` inside `audio_patchify_proj` is why the
        // longer name goes first here.
        for (old, new) in renames { path = path.replacingOccurrences(of: old, with: new) }
        if path.hasPrefix("adaln_single.") { path = "time_embed." + path.dropFirst("adaln_single.".count) }
        if path.hasPrefix("audio_adaln_single.") {
            path = "audio_time_embed." + path.dropFirst("audio_adaln_single.".count)
        }
        return path
    }

    /// Adapts every module `arrays` names and loads its `A` and `B`, at `strength`. Returns how many layers it
    /// adapted.
    @discardableResult
    public static func apply(_ arrays: [String: MLXArray], to transformer: NFKMLXLTX2TransformerNet,
                             strength: Float = 1) throws -> Int {
        var downs = [String: MLXArray](), ups = [String: MLXArray](), alphas = [String: Float]()
        for (key, value) in arrays {
            if key.hasSuffix(".lora_A.weight"), let path = modulePath(String(key.dropLast(".lora_A.weight".count))) {
                downs[path] = value
            } else if key.hasSuffix(".lora_B.weight"), let path = modulePath(String(key.dropLast(".lora_B.weight".count))) {
                ups[path] = value
            } else if key.hasSuffix(".alpha"), let path = modulePath(String(key.dropLast(".alpha".count))) {
                alphas[path] = value.asType(.float32).item(Float.self)
            }
        }
        guard Set(downs.keys) == Set(ups.keys), !downs.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration("the LoRA names no complete lora_A / lora_B pair")
        }
        var adapted = 0
        let byRank = Dictionary(grouping: downs.keys) { downs[$0]!.dim(0) }
        for (rank, paths) in byRank {
            let wanted = Set(paths)
            for alpha in Set(paths.map { alphas[$0] ?? Float(rank) }) {
                adapted += try NFKMLXLoRA.apply(to: transformer, rank: rank, alpha: alpha) { path, _ in
                    wanted.contains(path) && (alphas[path] ?? Float(rank)) == alpha
                }
            }
        }
        guard adapted == downs.count else {
            throw NFKMLXError.unsupportedConfiguration(
                "the LoRA names \(downs.count) layers; the transformer adapts \(adapted)")
        }
        var parameters = [(String, MLXArray)]()
        for (path, down) in downs {
            parameters.append(("\(path).lora_a", down.transposed().asType(.float32)))
            parameters.append(("\(path).lora_b", ups[path]!.transposed().asType(.float32)))
        }
        try transformer.update(parameters: ModuleParameters.unflattened(parameters), verify: .noUnusedKeys)
        setStrength(strength, in: transformer)
        return adapted
    }

    /// Sets every adapter's weight in `model`; 0 turns the LoRA off.
    public static func setStrength(_ strength: Float, in model: Module) {
        for case let adapter as NFKMLXLoRAAdapter in model.leafModules().flattened().map(\.1) {
            adapter.strength = strength
        }
    }

    /// The tensors of a LoRA file.
    public static func arrays(at url: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url)
    }
}
