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
/// The halves are held at the type the transformer computes in, and a float layer's sum is cast back to it, as
/// peft holds an adapter at its base layer's type and casts its sum to the base output's type. A streamed block
/// takes its adapters too: they stay resident while the stream reads the block's own weights in their turn.
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
    ///
    /// - Throws: `NFKMLXError.loRANotApplicable` when a module is not a linear layer of the transformer, is
    ///   already adapted, or its halves do not fit its weight.
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
        let type = transformer.computeType
        let layers = Dictionary(transformer.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        var replacements = [(String, Module)]()
        for (path, down) in downs.sorted(by: { $0.key < $1.key }) {
            let up = ups[path]!
            guard let layer = layers[path] as? Linear, !(layer is NFKMLXLoRAAdapter),
                  let (outputs, inputs) = shape(of: layer, at: path, in: transformer) else {
                throw NFKMLXError.loRANotApplicable("\(path) is not an unadapted linear layer of the transformer")
            }
            guard down.ndim == 2, up.ndim == 2, down.dim(0) == up.dim(1), down.dim(1) == inputs, up.dim(0) == outputs else {
                throw NFKMLXError.loRANotApplicable(
                    "\(path) takes halves \(down.shape) and \(up.shape), which do not fit its \(outputs) × \(inputs) weight")
            }
            let alpha = alphas[path] ?? Float(down.dim(0))
            let adapted: Linear
            if let quantized = layer as? QuantizedLinear {
                adapted = NFKMLXQuantizedLoRALinear(base: quantized, down: down.asType(type), up: up.asType(type),
                                                    alpha: alpha)
            } else {
                let float = NFKMLXLoRALinear(base: layer, down: down.asType(type), up: up.asType(type), alpha: alpha)
                float.resultType = type
                adapted = float
            }
            adapted.train(layer.training)
            replacements.append((path, adapted))
        }
        do {
            try NFKMLXModuleReplacement.place(replacements, in: transformer)
        } catch {
            throw NFKMLXError.loRANotApplicable("an adapted layer could not be put in place: \(error)")
        }
        transformer.freeze()
        transformer.unfreeze(keys: ["lora_a", "lora_b"])
        setStrength(strength, in: transformer)
        return replacements.count
    }

    /// The `(outputs, inputs)` of the linear layer at `path`: its weight's, or, for a streamed block's empty
    /// placeholder, the release's stored weight's.
    static func shape(of layer: Linear, at path: String,
                      in transformer: NFKMLXLTX2TransformerNet) -> (outputs: Int, inputs: Int)? {
        if let quantized = layer as? QuantizedLinear {
            return quantized.shape
        }
        if layer.weight.ndim == 2 {
            return layer.shape
        }
        guard let (index, name) = NFKMLXLTX2TransformerNet.block(ofKey: path + ".weight"),
              let stored = transformer.layerStream?.shape(ofLayer: index, tensor: name), stored.count == 2 else {
            return nil
        }
        return (stored[0], stored[1])
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
