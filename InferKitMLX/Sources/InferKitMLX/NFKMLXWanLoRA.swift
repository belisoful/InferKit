//
//  NFKMLXWanLoRA.swift
//  InferKitMLX
//
//  Released low-rank adapters for the Wan transformers: the files the Wan distillations and style
//  fine-tunes publish, merged into the weights of `NFKMLXWanTransformerNet` (Wan 2.1 T2V, Wan 2.2 TI2V)
//  or `NFKMLXWanAnimateNet` (Wan 2.2 Animate 2). A merged adapter costs nothing beyond the base model.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// Reads a released Wan LoRA and merges it into a Wan transformer, as diffusers' `WanLoraLoaderMixin`
/// loads it and `fuse_lora` folds it in.
///
/// @discussion Three layouts are read:
/// - **diffusers (PEFT):** keys `transformer.<module path>.lora_A.weight` / `.lora_B.weight`, an optional
///   `.lora_B.bias`, and the adapter's `transformer.r` and `transformer.lora_alpha` in the file's
///   `lora_adapter_metadata`. A file
///   without that metadata scales each layer by 1, the PEFT convention for `alpha == rank`.
/// - **original Wan:** keys `diffusion_model.<original name>` with `.lora_down.weight` / `.lora_up.weight`
///   and a scalar `.alpha`, or `.lora_A.weight` / `.lora_B.weight` without one, a `.diff_b` bias delta, and
///   a `.diff` full-weight delta on the head. The reference reads an `.alpha` on the blocks' layers alone and
///   refuses a file with one elsewhere, and so does the merge. The reference drops the normalizations' `.diff` keys, which
///   the releases store as zeros, and every other `.diff`, and drops a time-projection bias delta that comes
///   without its weights; the merge drops the same keys and counts them.
/// - **musubi-tuner:** keys `lora_unet_blocks_<n>_<layer>` with `.lora_down.weight`, `.lora_up.weight`, and
///   `.alpha`, over the blocks' attention and feed-forward layers.
/// - **DiffSynth-Studio:** the original names with no prefix and PEFT's adapter level,
///   `blocks.<n>.self_attn.q.lora_A.default.weight`, no alpha: DiffSynth fuses `scale · up · down`.
/// - **finetrainers:** the diffusers PEFT keys, its training configuration (`r`, `lora_alpha`) under the
///   metadata key `lora_config`. diffusers reads neither that key nor DiffSynth's spelling; the merge reads
///   both as the trainer that wrote them does.
///
/// Where a file states an alpha, the reference folds `alpha / rank` into the down and up weights at the
/// file's own precision before the merge, split into a power of two on the up weight; the merge forms the
/// same rounded weights. PEFT then holds the adapter at the base layer's type, so each delta is `up · down`
/// in float32, scaled by the adapter's own scaling and by `scale`, and added to the weight at the weight's
/// own type. A bias delta is added the same way.
///
/// Wan Animate's modules keep the original names, so its original-layout and musubi files are read onto
/// them directly. diffusers translates those layouts to the text-to-video transformer's names alone and
/// so does not load them into its own Animate transformer. A DoRA file is refused rather than read as a
/// plain LoRA.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXWanLoRA {

    /// What a merge did.
    public struct Report: Sendable {
        /// The layers whose weight or bias took a delta.
        public let mergedLayers: Int
        /// The keys the merge read and left alone, as the reference leaves them: the normalizations' `.diff`
        /// deltas, any other `.diff` but the head's, and a time-projection bias delta without its weights.
        public let droppedKeys: [String]
    }

    /// One layer's change before it is applied.
    struct Delta {
        /// The weight delta `[out, in]` in float32, already scaled.
        var weight: MLXArray?
        /// The bias delta in float32, already scaled.
        var bias: MLXArray?
    }

    /// Merges the adapter at `url` into a Wan text-to-video transformer at `scale`.
    @discardableResult
    public static func merge(adapterURL url: URL, into net: NFKMLXWanTransformerNet, scale: Float = 1) throws -> Report {
        let (deltas, dropped) = try deltas(adapterURL: url, naming: textToVideoPath(original:), scale: scale)
        return Report(mergedLayers: try apply(deltas, to: net), droppedKeys: dropped)
    }

    /// Merges the adapter at `url` into a Wan Animate transformer at `scale`.
    @discardableResult
    public static func merge(adapterURL url: URL, into net: NFKMLXWanAnimateNet, scale: Float = 1) throws -> Report {
        let (deltas, dropped) = try deltas(adapterURL: url, naming: animatePath(original:), scale: scale)
        return Report(mergedLayers: try apply(deltas, to: net), droppedKeys: dropped)
    }

    // MARK: Names

    /// The text-to-video transformer's module path for an original Wan layer name, or nil for a layer it
    /// does not have.
    static func textToVideoPath(original name: String) -> String? {
        let parts = name.split(separator: ".").map(String.init)
        if parts.count >= 3, parts[0] == "blocks", Int(parts[1]) != nil {
            let block = "blocks.\(parts[1])"
            let rest = parts[2...].joined(separator: ".")
            let attention = ["q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0",
                             "k_img": "add_k_proj", "v_img": "add_v_proj"]
            if parts.count == 4, parts[2] == "self_attn" || parts[2] == "cross_attn", let module = attention[parts[3]] {
                return "\(block).\(parts[2] == "self_attn" ? "attn1" : "attn2").\(module)"
            }
            return ["ffn.0": "\(block).ffn.net.0.proj", "ffn.2": "\(block).ffn.net.2"][rest]
        }
        return ["time_projection.1": "condition_embedder.time_proj", "head.head": "proj_out",
                "text_embedding.0": "condition_embedder.text_embedder.linear_1",
                "text_embedding.2": "condition_embedder.text_embedder.linear_2",
                "time_embedding.0": "condition_embedder.time_embedder.linear_1",
                "time_embedding.2": "condition_embedder.time_embedder.linear_2",
                "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
                "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2"][name]
    }

    /// The Animate transformer's module path for an original Wan layer name: its modules keep the original
    /// names but for the attention projections.
    static func animatePath(original name: String) -> String? {
        // The key map renames a projection only where a parameter name follows it.
        String(NFKMLXWanAnimate.moduleKey(forRelease: name + ".weight").dropLast(".weight".count))
    }

    /// A musubi-tuner layer spelling (`blocks_3_self_attn_q`, `blocks_3_ffn_0`) as the original name.
    static func musubiOriginal(_ spelling: String) -> String? {
        let parts = spelling.split(separator: "_").map(String.init)
        guard parts.count >= 3, parts[0] == "blocks", Int(parts[1]) != nil else { return nil }
        let rest = parts[2...].joined(separator: "_")
        for site in ["self_attn", "cross_attn"] where rest.hasPrefix(site + "_") {
            let projection = String(rest.dropFirst(site.count + 1))
            guard ["q", "k", "v", "o", "k_img", "v_img"].contains(projection) else { return nil }
            return "blocks.\(parts[1]).\(site).\(projection)"
        }
        return ["ffn_0": "blocks.\(parts[1]).ffn.0", "ffn_2": "blocks.\(parts[1]).ffn.2"][rest]
    }

    // MARK: Reading

    /// One layer's tensors as a file stores them.
    private struct Stored {
        var down: MLXArray?
        var up: MLXArray?
        var alpha: Float?
        var bias: MLXArray?
        var weight: MLXArray?
    }

    private enum Layout { case peft, original, musubi }

    /// The deltas an adapter file asks for, keyed by module path, and the keys the reference drops.
    static func deltas(adapterURL url: URL, naming: (String) -> String?, scale: Float) throws
        -> (deltas: [String: Delta], dropped: [String]) {
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        return try deltas(arrays: arrays, metadata: metadata, naming: naming, scale: scale)
    }

    static func deltas(arrays: [String: MLXArray], metadata: [String: String], naming: (String) -> String?,
                       scale: Float) throws -> (deltas: [String: Delta], dropped: [String]) {
        if arrays.keys.contains(where: { $0.contains("dora_scale") }) {
            throw NFKMLXError.loRANotApplicable("the adapter is a DoRA, which merges differently from a LoRA")
        }
        // DiffSynth-Studio saves the original names with no prefix, under PEFT's `.default` adapter level.
        let layout: Layout = arrays.keys.contains { $0.hasPrefix("diffusion_model.") || $0.contains(".default.") } ? .original
            : arrays.keys.contains { $0.hasPrefix("lora_unet_") } ? .musubi : .peft
        var layers = [String: Stored]()
        var dropped = [String]()
        let hasTimeProjectionWeight = arrays.keys.contains {
            $0.hasPrefix("diffusion_model.time_projection") && $0.hasSuffix(".weight")
        }
        for (key, value) in arrays {
            guard let (layer, part) = split(key, layout: layout) else {
                throw NFKMLXError.loRANotApplicable("the adapter's \(key) is not a LoRA tensor this reader knows")
            }
            if layout == .original {
                if part == "alpha" && !layer.hasPrefix("blocks.") {
                    throw NFKMLXError.loRANotApplicable(
                        "the adapter states an alpha for \(layer), which the reference reads on the blocks' layers alone")
                }
                let normDiff = (part == "diff" || part == "diff_b") && layer.contains("norm")
                let orphanTimeProjection = layer.hasPrefix("time_projection") && !hasTimeProjectionWeight
                if normDiff || orphanTimeProjection || (part == "diff" && layer != "head.head") {
                    dropped.append(key)
                    continue
                }
            }
            var stored = layers[layer] ?? Stored()
            switch part {
            case "down": stored.down = value
            case "up": stored.up = value
            case "alpha": stored.alpha = value.asType(.float32).item(Float.self)
            case "bias", "diff_b": stored.bias = value
            case "diff": stored.weight = value
            default: break
            }
            layers[layer] = stored
        }

        let configuration = try peftConfiguration(metadata)
        var deltas = [String: Delta]()
        for (layer, stored) in layers {
            let path: String?
            switch layout {
            case .peft: path = layer
            case .original: path = naming(layer)
            case .musubi: path = musubiOriginal(layer).flatMap(naming)
            }
            guard let path else {
                throw NFKMLXError.loRANotApplicable("the adapter's \(layer) names no layer of this transformer")
            }
            deltas[path] = try delta(stored, layer: layer, layout: layout,
                                     scaling: layout == .peft ? configuration.scaling(path, rank: stored.down?.dim(0))
                                                               : 1,
                                     scale: scale)
        }
        return (deltas, dropped.sorted())
    }

    /// A key's layer and its part: `down`, `up`, `alpha`, `bias` (a PEFT `lora_B.bias`), `diff_b`, or `diff`.
    private static func split(_ key: String, layout: Layout) -> (String, String)? {
        var name = key.replacingOccurrences(of: ".lora_A.default.", with: ".lora_A.")
            .replacingOccurrences(of: ".lora_B.default.", with: ".lora_B.")
        for prefix in ["diffusion_model.", "lora_unet_", "transformer."] where name.hasPrefix(prefix) {
            name = String(name.dropFirst(prefix.count))
            break
        }
        let suffixes: [(String, String)] = [
            (".lora_down.weight", "down"), (".lora_A.weight", "down"), (".lora_up.weight", "up"),
            (".lora_B.weight", "up"), (".lora_B.bias", "bias"), (".alpha", "alpha"), (".diff_b", "diff_b"),
            (".diff", "diff"),
        ]
        for (suffix, part) in suffixes where name.hasSuffix(suffix) {
            return (String(name.dropLast(suffix.count)), part)
        }
        return nil
    }

    /// One layer's delta. The original and musubi layouts fold `alpha / rank` into the down and up weights
    /// at their own precision, the factor split into a power of two on the up weight, as the reference's
    /// conversion does; the head's `.diff` is its whole weight delta.
    private static func delta(_ stored: Stored, layer: String, layout: Layout, scaling: Float,
                              scale: Float) throws -> Delta {
        var result = Delta(weight: nil, bias: nil)
        let strength = scaling * scale
        if let full = stored.weight {
            result.weight = full.asType(.float32) * strength
        }
        if stored.down != nil || stored.up != nil {
            guard var down = stored.down, var up = stored.up else {
                throw NFKMLXError.loRANotApplicable("the adapter's \(layer) lacks its down or up weight")
            }
            let rank = down.dim(0)
            guard down.ndim == 2, up.ndim == 2, up.dim(1) == rank else {
                throw NFKMLXError.loRANotApplicable(
                    "the adapter's \(layer) pairs a down weight \(down.shape) with an up weight \(up.shape)")
            }
            if layout == .musubi && stored.alpha == nil {
                throw NFKMLXError.loRANotApplicable("the adapter's \(layer) carries no alpha")
            }
            if let alpha = stored.alpha {
                let (downScale, upScale) = foldedScales(Double(alpha) / Double(rank))
                down = (down.asType(.float32) * Float(downScale)).asType(down.dtype)
                up = (up.asType(.float32) * Float(upScale)).asType(up.dtype)
            }
            let product = matmul(up.asType(.float32), down.asType(.float32)) * strength
            result.weight = result.weight.map { $0 + product } ?? product
        }
        if let bias = stored.bias {
            result.bias = bias.asType(.float32) * strength
        }
        return result
    }

    /// The reference's split of a LoRA scaling into a down factor and a power-of-two up factor.
    static func foldedScales(_ scale: Double) -> (down: Double, up: Double) {
        var down = scale, up = 1.0
        while down * 2 < up {
            down *= 2
            up /= 2
        }
        return (down, up)
    }

    // MARK: PEFT metadata

    /// A PEFT adapter's scaling, from the `lora_adapter_metadata` diffusers writes.
    private struct PEFTConfiguration {
        var rank: Int?
        var alpha: Float?
        var rankPattern: [String: Int] = [:]
        var alphaPattern: [String: Float] = [:]
        var rsLoRA = false

        /// The scaling of the layer at `path`: `alpha / rank` (`alpha / √rank` for rsLoRA), the patterns
        /// matched against the path's end as PEFT matches them; 1 where the file states no configuration.
        func scaling(_ path: String, rank stored: Int?) -> Float {
            guard let base = alpha ?? rank.map(Float.init) else { return 1 }
            let r = Self.patterned(rankPattern, path).map(Float.init) ?? Float(rank ?? stored ?? 1)
            let a = Self.patterned(alphaPattern, path) ?? base
            return rsLoRA ? a / r.squareRoot() : a / r
        }

        static func patterned<T>(_ pattern: [String: T], _ path: String) -> T? {
            pattern.first { key, _ in
                guard let expression = try? NSRegularExpression(pattern: "^(.*\\.)?(\(key))$") else { return false }
                return expression.firstMatch(in: path, range: NSRange(path.startIndex..., in: path)) != nil
            }?.value
        }
    }

    private static func peftConfiguration(_ metadata: [String: String]) throws -> PEFTConfiguration {
        let json: [String: Any]
        if let text = metadata["lora_adapter_metadata"] {
            guard let saved = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] else {
                throw NFKMLXError.loRANotApplicable("the adapter's lora_adapter_metadata is not an object")
            }
            // diffusers saves the configuration under the component's prefix and reads back those keys alone.
            json = Dictionary(uniqueKeysWithValues: saved.compactMap { key, value in
                key.hasPrefix("transformer.") ? (String(key.dropFirst("transformer.".count)), value) : nil
            })
        } else if let text = metadata["lora_config"] {
            // finetrainers writes the PEFT configuration it trained with, unprefixed, under its own key.
            guard let saved = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] else {
                throw NFKMLXError.loRANotApplicable("the adapter's lora_config is not an object")
            }
            json = saved
        } else {
            return PEFTConfiguration()
        }
        if (json["use_dora"] as? NSNumber)?.boolValue == true {
            throw NFKMLXError.loRANotApplicable("the adapter is a DoRA, which merges differently from a LoRA")
        }
        var configuration = PEFTConfiguration()
        configuration.rank = (json["r"] as? NSNumber)?.intValue
        configuration.alpha = (json["lora_alpha"] as? NSNumber)?.floatValue
        configuration.rankPattern = ((json["rank_pattern"] as? [String: NSNumber]) ?? [:]).mapValues(\.intValue)
        configuration.alphaPattern = ((json["alpha_pattern"] as? [String: NSNumber]) ?? [:]).mapValues(\.floatValue)
        configuration.rsLoRA = (json["use_rslora"] as? NSNumber)?.boolValue ?? false
        return configuration
    }

    // MARK: The merge

    /// Applies `deltas` to the linear layers of `root`, every target checked before any weight changes.
    /// `pathPrefix` is the part of each delta's path above `root`; deltas outside it are skipped.
    @discardableResult
    static func apply(_ deltas: [String: Delta], to root: Module, pathPrefix: String = "") throws -> Int {
        let layers = Dictionary(uniqueKeysWithValues: root.leafModules().flattened().map { ($0.0, $0.1) })
        var planned = [(Linear, Delta, String)]()
        for (path, delta) in deltas where path.hasPrefix(pathPrefix) {
            let local = String(path.dropFirst(pathPrefix.count))
            guard let module = layers[local] else {
                throw NFKMLXError.loRANotApplicable("the adapter's \(path) names no layer of this transformer")
            }
            guard let linear = module as? Linear, !(module is QuantizedLinear) else {
                throw NFKMLXError.loRANotApplicable("\(path) is not an unpacked linear layer, which a merge needs")
            }
            if let weight = delta.weight, weight.shape != linear.weight.shape {
                throw NFKMLXError.loRANotApplicable(
                    "the adapter's \(path) is \(weight.shape) where the layer is \(linear.weight.shape)")
            }
            if let bias = delta.bias, bias.shape != (linear.bias?.shape ?? []) {
                throw NFKMLXError.loRANotApplicable("the adapter's \(path) carries a bias the layer does not match")
            }
            planned.append((linear, delta, path))
        }
        for (linear, delta, _) in planned {
            var update = [(String, MLXArray)]()
            if let weight = delta.weight {
                update.append(("weight", linear.weight + weight.asType(linear.weight.dtype)))
            }
            if let bias = delta.bias, let base = linear.bias {
                update.append(("bias", base + bias.asType(base.dtype)))
            }
            linear.update(parameters: ModuleParameters.unflattened(update))
        }
        eval(planned.map(\.0))
        return planned.count
    }
}

/// The released adapters a generator merges into its transformer each time the transformer loads.
final class NFKWanAdapterList {
    private(set) var entries = [(url: URL, scale: Float)]()

    var isEmpty: Bool { entries.isEmpty }

    func add(_ url: URL, scale: Float) { entries.append((url, scale)) }

    func removeAll() { entries.removeAll() }

    /// Each adapter's deltas in the order they were added, read with `naming`. A layer two adapters change
    /// takes one delta after the other, as the reference fuses one adapter after another.
    func deltas(naming: (String) -> String?) throws -> [[String: NFKMLXWanLoRA.Delta]] {
        try entries.map { try NFKMLXWanLoRA.deltas(adapterURL: $0.url, naming: naming, scale: $0.scale).deltas }
    }
}
