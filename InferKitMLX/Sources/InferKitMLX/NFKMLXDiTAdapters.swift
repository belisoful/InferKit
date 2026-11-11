//
//  NFKMLXDiTAdapters.swift
//  InferKitMLX
//
//  Released LoRA adapters for the diffusion transformers, held beside the transformer's layers rather than
//  merged into them, read in the layouts the families' adapters are published in: diffusers' (with or without
//  its `transformer.` prefix), kohya's, and the original model's.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// How one diffusion transformer family's adapter files name its layers.
///
/// @discussion A layer is named three ways. diffusers names it as its own model class does, which the port
/// follows; ``renames`` covers the few module paths where the port departs from it. The original model, which
/// kohya's files also name with each dot read as an underscore, fuses some of diffusers' layers; ``original``
/// lists each one. A fused original layer's adapter shares its down half among the diffusers layers it lists and
/// splits its up half by each one's outputs, in order.
struct NFKMLXDiTAdapterLayout {
    /// The family's name in messages.
    var family: String
    /// Each original-model layer name, dotted as the original model writes it, and the diffusers layers it is.
    var original: [String: [String]] = [:]
    /// Prefix rewrites from a diffusers layer name to the port's module path, the first that matches applied.
    var renames: [(diffusers: String, port: String)] = []
    /// Whether an adapter on a diffusers layer is left out of a transformer that lacks the layer, as diffusers
    /// leaves out a FLUX [dev] adapter's guidance embedding on a release without one.
    var leftOut: (String) -> Bool = { _ in false }
    /// The diffusers layers whose up half swaps its two halves when an original-model file names them.
    var swapsHalves: Set<String> = []
    /// Why an adapter on a diffusers layer is refused, completing "adapts <layer>, …", or nil where it is held.
    var refusal: (String) -> String? = { _ in nil }

    /// The port's module path of a diffusers layer.
    func modulePath(_ diffusers: String) -> String {
        for rename in renames where diffusers == rename.diffusers || diffusers.hasPrefix(rename.diffusers + ".") {
            return rename.port + diffusers.dropFirst(rename.diffusers.count)
        }
        return diffusers
    }

    /// The diffusers name of a module path of the port.
    func diffusersName(_ path: String) -> String {
        for rename in renames where path == rename.port || path.hasPrefix(rename.port + ".") {
            return rename.diffusers + path.dropFirst(rename.port.count)
        }
        return path
    }
}

/// Reads diffusion transformer adapters and holds them unmerged beside the transformer's layers.
///
/// @discussion A file's layout is read from its names, the first that matches taken:
/// - diffusers: `transformer.<layer>.lora_A.weight` and `lora_B.weight`, scaled by the `lora_adapter_metadata` it
///   carries or by a layer's `alpha`, else 1;
/// - kohya: `lora_unet_<layer>.lora_down.weight`, `lora_up.weight`, and `alpha`, where `<layer>` is an
///   original-model name or a diffusers name with each dot read as an underscore;
/// - the original model's: `diffusion_model.<layer>` (or PEFT's `base_model.model.<layer>`), naming an
///   original-model layer or a diffusers one;
/// - diffusers' names without a prefix, as a ComfyUI copy writes them.
///
/// `lora_down` and `lora_up` read as `lora_A` and `lora_B` in the last two, and PEFT's `.default` adapter name is
/// dropped. A text encoder's adapter, an XLabs file, a DoRA, a LyCORIS LoKr or LoHa, a layer the transformer lacks
/// (other than one ``NFKMLXDiTAdapterLayout/leftOut`` names), and a kohya name that reads as two layers are refused
/// before any layer changes.
enum NFKMLXDiTAdapters {

    /// Holds each adapter beside `transformer`'s layers at its strength, in load order, and returns the strengths.
    @discardableResult
    static func hold(_ adapters: [(url: URL, strength: Float)], in transformer: Module,
                     layout: NFKMLXDiTAdapterLayout) throws -> [Float] {
        guard !adapters.isEmpty else { return [] }
        let layers = Dictionary(transformer.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        var detours = [String: [NFKMLXLowRankDetour]]()
        for (index, adapter) in adapters.enumerated() {
            for (path, detour) in try read(adapter.url, adapter: index, strength: adapter.strength, layers: layers,
                                           layout: layout) {
                detours[path, default: []].append(detour)
            }
        }
        try NFKMLXAdapterLayers.attach(detours, to: transformer)
        return adapters.map(\.strength)
    }

    /// The adapters an Objective-C factory names: each URL with the strength at its index.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` when the counts differ.
    static func adapters(_ urls: [URL], strengths: [NSNumber]) throws -> [(url: URL, strength: Float)] {
        guard urls.count == strengths.count else {
            throw NFKMLXError.unsupportedConfiguration("one strength per adapter: \(urls.count) adapters, "
                                                       + "\(strengths.count) strengths")
        }
        return zip(urls, strengths).map { (url: $0, strength: $1.floatValue) }
    }

    /// A backend's parameter keys, with `NFKParameterAdapterStrengths` where the model holds adapters.
    static func parameterKeys(_ keys: Set<String>, holding strengths: [Float]) -> Set<String> {
        strengths.isEmpty ? keys : keys.union([NFKParameterAdapterStrengths])
    }

    /// One stored adapter: its down and up halves and the diffusers layers they adapt.
    private struct Half {
        let stored: String
        let down: MLXArray
        let up: MLXArray
        let targets: [String]
        /// Whether the name is an original-model layer's.
        let fromOriginal: Bool
    }

    /// The detours one adapter file places, by module path.
    static func read(_ url: URL, adapter: Int, strength: Float, layers: [String: Module],
                     layout: NFKMLXDiTAdapterLayout) throws -> [(String, NFKMLXLowRankDetour)] {
        let name = url.lastPathComponent
        let (stored, metadata) = try MLX.loadArraysAndMetadata(url: url)
        try refuseUnheld(Array(stored.keys), name: name, family: layout.family)
        let arrays = Dictionary(stored.map { key, value in
            (key.replacingOccurrences(of: ".lora_A.default.", with: ".lora_A.")
                .replacingOccurrences(of: ".lora_B.default.", with: ".lora_B."), value)
        }, uniquingKeysWith: { first, _ in first })
        let halves = try self.halves(of: arrays, name: name, layers: layers, layout: layout)
        guard !halves.isEmpty else {
            throw NFKMLXError.loRANotApplicable("\(name) adapts no layer")
        }

        let peft = try configuration(beside: url, metadata: metadata)
        var placed = [(String, NFKMLXLowRankDetour)]()
        for half in halves {
            let paths = half.targets.map(layout.modulePath)
            if half.targets.allSatisfy(layout.leftOut) && paths.allSatisfy({ layers[$0] == nil }) {
                continue
            }
            for (target, path) in zip(half.targets, paths) {
                if let reason = layout.refusal(target) {
                    throw NFKMLXError.loRANotApplicable("\(name) adapts \(target), \(reason)")
                }
                guard layers[path] != nil else {
                    throw NFKMLXError.loRANotApplicable(
                        "\(name) adapts \(half.stored), whose \(target) is no layer of \(layout.family)")
                }
            }
            let rank = half.down.dim(0)
            let scale: Double
            if let alpha = arrays[half.stored + ".alpha"] {
                scale = Double(alpha.asType(.float32).item(Float.self)) / Double(rank)
            } else {
                scale = half.targets.count == 1 ? peft.preciseScaling(half.targets[0], rank: rank) ?? 1 : 1
            }
            var up = half.up
            if half.fromOriginal, half.targets.count == 1, layout.swapsHalves.contains(half.targets[0]) {
                let width = up.dim(0) / 2
                up = concatenated([up[width...], up[..<width]], axis: 0)
            }
            var row = 0
            for path in paths {
                let outputs = try outputCount(of: path, in: layers, name: name, family: layout.family)
                guard row + outputs <= up.dim(0) else {
                    throw NFKMLXError.loRANotApplicable(
                        "\(name) adapts \(half.stored) with \(up.dim(0)) outputs, fewer than the layers it fuses")
                }
                placed.append((path, try detour(down: half.down, up: up[row ..< row + outputs], scale: scale,
                                                adapter: adapter, strength: strength, path: path, layers: layers,
                                                name: name)))
                row += outputs
            }
            guard row == up.dim(0) else {
                throw NFKMLXError.loRANotApplicable(
                    "\(name) adapts \(half.stored) with \(up.dim(0)) outputs, where the layers it names take \(row)")
            }
        }
        return placed
    }

    /// The PEFT scaling an adapter states: diffusers' `lora_adapter_metadata` in the file, else the
    /// `adapter_config.json` PEFT's `save_pretrained` writes beside it, else none (every layer at 1).
    private static func configuration(beside url: URL, metadata: [String: String]) throws -> NFKMLXWanLoRA.PEFTConfiguration {
        let file = url.deletingLastPathComponent().appendingPathComponent("adapter_config.json")
        guard metadata["lora_adapter_metadata"] == nil, metadata["lora_config"] == nil,
              FileManager.default.fileExists(atPath: file.path) else {
            return try NFKMLXWanLoRA.peftConfiguration(metadata, component: "transformer.")
        }
        let json = try NFKMLXDecoderAdapters.configuration(beside: url)
        if let type = json["peft_type"] as? String, type.uppercased() != "LORA" {
            throw NFKMLXError.loRANotApplicable("\(url.lastPathComponent) is a \(type) adapter; a held adapter computes a LoRA's product")
        }
        return try NFKMLXWanLoRA.peftConfiguration(json: json)
    }

    /// Refuses a file whose adapters no held detour computes.
    private static func refuseUnheld(_ keys: [String], name: String, family: String) throws {
        if keys.contains(where: { $0.hasPrefix("lora_te") || $0.hasPrefix("text_encoder") }) {
            throw NFKMLXError.loRANotApplicable("\(name) adapts a text encoder, which \(family) holds no adapter beside")
        }
        if keys.contains(where: { $0.contains(".processor.") }) {
            throw NFKMLXError.loRANotApplicable("\(name) is an XLabs adapter, a layout InferKit does not read")
        }
        if keys.contains(where: { $0.contains("lora_magnitude_vector") || $0.contains("dora_scale") }) {
            throw NFKMLXError.loRANotApplicable("\(name) is a DoRA, which a held adapter does not compute")
        }
        if keys.contains(where: { $0.contains("lokr_") || $0.contains("hada_") }) {
            throw NFKMLXError.loRANotApplicable(
                "\(name) is a LyCORIS LoKr or LoHa, whose product a held adapter does not compute")
        }
    }

    /// The file's adapters in the first layout its names match.
    private static func halves(of arrays: [String: MLXArray], name: String, layers: [String: Module],
                               layout: NFKMLXDiTAdapterLayout) throws -> [Half] {
        let keys = arrays.keys
        if keys.contains(where: { $0.hasPrefix("transformer.") && $0.hasSuffix(".lora_A.weight") }) {
            return try keys.sorted().filter { $0.hasPrefix("transformer.") && $0.hasSuffix(".lora_A.weight") }.map { key in
                let stored = String(key.dropLast(".lora_A.weight".count))
                return Half(stored: stored, down: arrays[key]!, up: try pair(stored + ".lora_B.weight", in: arrays, name: name),
                            targets: [String(stored.dropFirst("transformer.".count))], fromOriginal: false)
            }
        }
        if keys.contains(where: { $0.hasPrefix("lora_unet_") }) {
            let kohya = Dictionary(layout.original.map { (underscored($0.key), $0.value) },
                                   uniquingKeysWith: { first, _ in first })
            let modules = underscoredModules(layers, layout: layout)
            return try keys.sorted().filter { $0.hasPrefix("lora_unet_") && $0.hasSuffix(".lora_down.weight") }.map { key in
                let stored = String(key.dropLast(".lora_down.weight".count))
                let layer = String(stored.dropFirst("lora_unet_".count))
                let targets: [String]
                if let fused = kohya[layer] {
                    targets = fused
                } else if let match = modules[layer] {
                    guard let match else {
                        throw NFKMLXError.loRANotApplicable(
                            "\(name) adapts \(stored), which names more than one layer of \(layout.family)")
                    }
                    targets = [match]
                } else {
                    throw NFKMLXError.loRANotApplicable("\(name) adapts \(stored), which names no layer of \(layout.family)")
                }
                return Half(stored: stored, down: arrays[key]!, up: try pair(stored + ".lora_up.weight", in: arrays, name: name),
                            targets: targets, fromOriginal: kohya[layer] != nil)
            }
        }
        if let prefix = ["diffusion_model.", "base_model.model."].first(where: { prefix in keys.contains { $0.hasPrefix(prefix) } }) {
            return try named(arrays, prefix: prefix, name: name, layout: layout)
        }
        if keys.contains(where: { $0.hasSuffix(".lora_A.weight") || $0.hasSuffix(".lora_down.weight") }) {
            return try named(arrays, prefix: "", name: name, layout: layout)
        }
        throw NFKMLXError.loRANotApplicable(
            "\(name) holds no \(layout.family) LoRA tensors in the diffusers, kohya, or original model's layout")
    }

    /// The adapters under `prefix`, each naming an original-model layer or a diffusers one.
    private static func named(_ arrays: [String: MLXArray], prefix: String, name: String,
                              layout: NFKMLXDiTAdapterLayout) throws -> [Half] {
        try arrays.keys.sorted().filter {
            $0.hasPrefix(prefix) && ($0.hasSuffix(".lora_A.weight") || $0.hasSuffix(".lora_down.weight"))
        }.map { key in
            let peftHalves = key.hasSuffix(".lora_A.weight")
            let stored = String(key.dropLast((peftHalves ? ".lora_A.weight" : ".lora_down.weight").count))
            let layer = String(stored.dropFirst(prefix.count))
            return Half(stored: stored, down: arrays[key]!,
                        up: try pair(stored + (peftHalves ? ".lora_B.weight" : ".lora_up.weight"), in: arrays, name: name),
                        targets: layout.original[layer] ?? [layer], fromOriginal: layout.original[layer] != nil)
        }
    }

    /// A dotted name as kohya writes it.
    static func underscored(_ name: String) -> String {
        name.replacingOccurrences(of: ".", with: "_")
    }

    /// The diffusers name of each linear and convolution layer of the transformer by its kohya form, or nil
    /// where two layers share that form.
    static func underscoredModules(_ layers: [String: Module], layout: NFKMLXDiTAdapterLayout) -> [String: String?] {
        var modules = [String: String?]()
        for (path, module) in layers where module is Linear || module is Conv2d {
            let diffusers = layout.diffusersName(path)
            let kohya = underscored(diffusers)
            if let existing = modules[kohya], existing != diffusers {
                modules[kohya] = .some(nil)
            } else {
                modules[kohya] = diffusers
            }
        }
        return modules
    }

    private static func pair(_ key: String, in arrays: [String: MLXArray], name: String) throws -> MLXArray {
        guard let array = arrays[key] else {
            throw NFKMLXError.loRANotApplicable("\(name) holds a down half without \(key)")
        }
        return array
    }

    private static func outputCount(of path: String, in layers: [String: Module], name: String,
                                    family: String) throws -> Int {
        switch layers[path] {
        case let quantized as QuantizedLinear: return quantized.shape.0
        case let linear as Linear: return linear.weight.dim(0)
        case let convolution as Conv2d: return convolution.weight.dim(0)
        default:
            throw NFKMLXError.loRANotApplicable(
                "\(name) adapts \(path), which is not a linear or convolution layer of \(family)")
        }
    }

    /// One detour beside the layer at `path`: a linear layer's, or a convolution's whose down half is a kernel
    /// `[rank, inputs, kH, kW]` as PyTorch stores it, run at the convolution's stride, padding, and dilation.
    private static func detour(down: MLXArray, up: MLXArray, scale: Double, adapter: Int, strength: Float,
                               path: String, layers: [String: Module], name: String) throws -> NFKMLXLowRankDetour {
        guard let convolution = layers[path] as? Conv2d else {
            return try NFKMLXDecoderAdapters.detour(down: down, up: up, scale: scale, adapter: adapter,
                                                    strength: strength, path: path, layers: layers, name: name)
        }
        guard convolution.groups == 1 else {
            throw NFKMLXError.loRANotApplicable(
                "\(name) adapts \(path), a grouped convolution, beside which a held adapter computes nothing")
        }
        let weight = convolution.weight
        let rank = down.dim(0)
        let kernel = down.ndim == 4 ? down.transposed(0, 2, 3, 1) : down.reshaped([rank, 1, 1, -1])
        let projection = up.reshaped([up.dim(0), -1])
        guard kernel.shape == [rank, weight.dim(1), weight.dim(2), weight.dim(3)],
              projection.shape == [weight.dim(0), rank] else {
            throw NFKMLXError.loRANotApplicable(
                "\(name) adapts \(path) with halves \(down.shape) and \(up.shape), which do not fit its "
                + "\(weight.dim(0)) × \(weight.dim(1)) × \(weight.dim(2)) × \(weight.dim(3)) kernel")
        }
        return NFKMLXLowRankDetour(down: kernel.asType(weight.dtype), up: projection.asType(weight.dtype),
                                   scale: scale, adapter: adapter, strength: strength)
    }
}

// MARK: - The families

extension NFKMLXDiTAdapters {

    /// Holds adapters beside FLUX.2's transformer. Black Forest Labs' and kohya's files name the original model,
    /// as diffusers' `_convert_non_diffusers_flux2_lora_to_diffusers` and `_convert_kohya_flux2_lora_to_diffusers`
    /// read them; a [dev] adapter's guidance-embedding part is left out of a release without one.
    @discardableResult
    static func hold(_ adapters: [(url: URL, strength: Float)], in transformer: NFKMLXFlux2TransformerNet) throws -> [Float] {
        try hold(adapters, in: transformer, layout: flux2Layout(doubleBlocks: transformer.transformerBlocks.count,
                                                                singleBlocks: transformer.singleTransformerBlocks.count))
    }

    /// FLUX.2's layout. The final layer's modulation keeps its halves in the original order, as diffusers reads it.
    static func flux2Layout(doubleBlocks: Int, singleBlocks: Int) -> NFKMLXDiTAdapterLayout {
        var original: [String: [String]] = [
            "img_in": ["x_embedder"], "txt_in": ["context_embedder"],
            "time_in.in_layer": ["time_guidance_embed.timestep_embedder.linear_1"],
            "time_in.out_layer": ["time_guidance_embed.timestep_embedder.linear_2"],
            "guidance_in.in_layer": ["time_guidance_embed.guidance_embedder.linear_1"],
            "guidance_in.out_layer": ["time_guidance_embed.guidance_embedder.linear_2"],
            "final_layer.linear": ["proj_out"], "final_layer.adaLN_modulation.1": ["norm_out.linear"],
            "single_stream_modulation.lin": ["single_stream_modulation.linear"],
            "double_stream_modulation_img.lin": ["double_stream_modulation_img.linear"],
            "double_stream_modulation_txt.lin": ["double_stream_modulation_txt.linear"],
        ]
        for block in 0 ..< doubleBlocks {
            let (from, to) = ("double_blocks.\(block).", "transformer_blocks.\(block).")
            original[from + "img_attn.qkv"] = ["attn.to_q", "attn.to_k", "attn.to_v"].map { to + $0 }
            original[from + "img_attn.proj"] = [to + "attn.to_out.0"]
            original[from + "txt_attn.qkv"] = ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"].map { to + $0 }
            original[from + "txt_attn.proj"] = [to + "attn.to_add_out"]
            original[from + "img_mlp.0"] = [to + "ff.linear_in"]
            original[from + "img_mlp.2"] = [to + "ff.linear_out"]
            original[from + "txt_mlp.0"] = [to + "ff_context.linear_in"]
            original[from + "txt_mlp.2"] = [to + "ff_context.linear_out"]
        }
        for block in 0 ..< singleBlocks {
            let (from, to) = ("single_blocks.\(block).", "single_transformer_blocks.\(block).attn.")
            original[from + "linear1"] = [to + "to_qkv_mlp_proj"]
            original[from + "linear2"] = [to + "to_out"]
        }
        return NFKMLXDiTAdapterLayout(family: "FLUX.2", original: original,
                                      leftOut: { $0.hasPrefix("time_guidance_embed.guidance_embedder.") })
    }

    /// Holds adapters beside Z-Image's transformer. An original-model file names each attention's output
    /// projection `out`, which diffusers' `_convert_non_diffusers_z_image_lora_to_diffusers` reads as `to_out.0`.
    @discardableResult
    static func hold(_ adapters: [(url: URL, strength: Float)], in transformer: NFKMLXZImageTransformerNet) throws -> [Float] {
        try hold(adapters, in: transformer, layout: zImageLayout(of: transformer))
    }

    /// Z-Image's layout over `transformer`'s blocks.
    static func zImageLayout(of transformer: Module) -> NFKMLXDiTAdapterLayout {
        let output = ".attention.to_out.0"
        var original = [String: [String]]()
        for (path, _) in transformer.leafModules().flattened() where path.hasSuffix(output) {
            original[String(path.dropLast(output.count)) + ".attention.out"] = [path]
        }
        return NFKMLXDiTAdapterLayout(family: "Z-Image", original: original)
    }

    /// Holds adapters beside SD3's transformer, named as diffusers names it. A transformer that reads blocks from
    /// the release on each pass is refused: a held detour wraps the layer the stream writes weights into.
    @discardableResult
    static func hold(_ adapters: [(url: URL, strength: Float)], in transformer: NFKMLXSD3TransformerNet) throws -> [Float] {
        guard !adapters.isEmpty else { return [] }
        guard transformer.layerStream == nil else {
            throw NFKMLXError.unsupportedConfiguration(sd3StreamRefusal)
        }
        return try hold(adapters, in: transformer, layout: NFKMLXDiTAdapterLayout(family: "Stable Diffusion 3"))
    }

    /// The refusal of adapters on an SD3 transformer whose blocks stream.
    static let sd3StreamRefusal = "\(NFKMLXSD3Generator.modelName) holds adapters beside layers held in memory; "
        + "load it with a residency that holds every layer"

    /// Holds adapters beside SANA's transformer, named as diffusers names it. A depthwise convolution
    /// (`ff.conv_depth`) is refused; the other convolutions hold a convolution detour.
    @discardableResult
    static func hold(_ adapters: [(url: URL, strength: Float)], in transformer: NFKMLXSANATransformerNet) throws -> [Float] {
        try hold(adapters, in: transformer, layout: NFKMLXDiTAdapterLayout(family: "SANA"))
    }

    /// Holds adapters beside LTX-Video's transformer, named as diffusers names it.
    @discardableResult
    static func hold(_ adapters: [(url: URL, strength: Float)], in transformer: NFKMLXLTXTransformerNet) throws -> [Float] {
        try hold(adapters, in: transformer, layout: NFKMLXDiTAdapterLayout(family: "LTX-Video"))
    }

    /// Holds adapters beside Stable Audio Open's transformer, named as diffusers names it.
    @discardableResult
    static func hold(_ adapters: [(url: URL, strength: Float)], in transformer: NFKMLXStableAudioDiTNet) throws -> [Float] {
        try hold(adapters, in: transformer, layout: stableAudioLayout)
    }

    /// Stable Audio Open's layout. The port holds each conditioning projection's `nn.Sequential` under
    /// `layers`, and holds the one-wide `preprocess_conv` and `postprocess_conv` as linear layers, beside which
    /// an adapter trained on the convolution is refused.
    static var stableAudioLayout: NFKMLXDiTAdapterLayout {
        NFKMLXDiTAdapterLayout(
            family: "Stable Audio Open",
            renames: ["timestep_proj", "global_proj", "cross_attention_proj"].map { (diffusers: $0, port: $0 + ".layers") },
            refusal: { layer in
                layer == "preprocess_conv" || layer == "postprocess_conv"
                    ? "a one-wide convolution the port holds as a linear layer, which takes no convolution's adapter" : nil
            })
    }
}
