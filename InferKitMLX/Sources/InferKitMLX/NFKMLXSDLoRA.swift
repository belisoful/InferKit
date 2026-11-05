//
//  NFKMLXSDLoRA.swift
//  InferKitMLX
//
//  Released low-rank adapters for Stable Diffusion: the files the LCM-LoRA and SDXL-Lightning
//  distillations publish, and the style and subject fine-tunes people train in the same layouts. An
//  adapter merges into the UNet's weights, so a run costs nothing beyond the base model, or it is held
//  beside the UNet's and the text towers' weights, so adapters stack and their strengths change per run.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// Reads a released Stable Diffusion LoRA, and merges it into a UNet or holds it beside a model's
/// weights.
///
/// @discussion Three key layouts are read:
/// - **kohya over diffusers names**, which the LCM-LoRA and SDXL-Lightning releases and most Stable
///   Diffusion 1.x fine-tunes use: `lora_unet_<module path with underscores>` with `.lora_down.weight`,
///   `.lora_up.weight`, and a scalar `.alpha`.
/// - **kohya over the original block names**, which most SDXL fine-tunes use:
///   `lora_unet_input_blocks_<n>_<m>_…`, `lora_unet_middle_block_<m>_…`, and `lora_unet_output_blocks_<n>_<m>_…`,
///   with the original names of the residual blocks' layers (`in_layers_2`, `out_layers_3`, `emb_layers_1`,
///   `skip_connection`) and the same parts. The block numbers translate through the UNet's own level and
///   layer counts.
/// - **PEFT**, which diffusers writes: `unet.<module path>` with `.lora_A.weight` and `.lora_B.weight`, or
///   the older `.lora.down.weight` and `.lora.up.weight`. The scaling is `lora_alpha / r` from the file's
///   `lora_adapter_metadata` where diffusers recorded it, and 1 otherwise.
///
/// The text towers' keys are `lora_te_` (or `lora_te1_`) and `lora_te2_` in kohya's layout and
/// `text_encoder.` and `text_encoder_2.` in PEFT's, over the `transformers` CLIP names. A convolution target
/// stores its down weight as a `[rank, in, kH, kW]` kernel and its up weight as `[out, rank, 1, 1]`. A key in
/// none of these layouts, such as LoHa's `hada_` or LoKr's `lokr_`, is refused, as is a target the model
/// has no layer for.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXSDLoRA {

    /// What a merge did: how many layers it changed and how many text-tower keys it left alone.
    public struct Report: Sendable {
        /// The linear and convolution layers whose weights took a delta.
        public let mergedLayers: Int
        /// The `lora_te…` and `text_encoder…` keys the merge did not reach.
        public let skippedTextEncoderKeys: Int
    }

    /// Merges the adapter at `url` into `unet` at `scale`.
    ///
    /// The delta for a linear target is `scale · scaling · up · down`, the scaling `alpha / rank` or PEFT's
    /// recorded one (``NFKMLXSDLoRA``); for a convolution the up
    /// weight contracts the down kernel's rank axis. The product forms in float32 and takes the base
    /// weight's type, as diffusers' `fuse_lora` does. A key that names no layer of this UNet is an error,
    /// so an adapter trained for another geometry fails before it changes a weight.
    ///
    /// - Parameters:
    ///   - url: the adapter's safetensors file.
    ///   - unet: the UNet whose weights take the delta.
    ///   - scale: the strength multiplier on every delta; 1 merges the adapter as released.
    ///   - ignoringTextEncoderKeys: merge the UNet half of an adapter that also adapts the text towers,
    ///     and count the keys left alone in the report. False, the default, refuses such a file.
    /// - Returns: the merge's layer count and the count of text-tower keys left alone.
    /// - Throws: `NFKMLXError.weightsMismatch` when a target is missing from the UNet, a shape disagrees,
    ///   a target lacks its pair of weights, or the file carries text-tower keys without the opt-in.
    @discardableResult
    public static func merge(adapterURL url: URL, into unet: NFKMLXSDUNet, scale: Float = 1,
                             ignoringTextEncoderKeys: Bool = false) throws -> Report {
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        return try merge(arrays: arrays, metadata: metadata, into: unet, scale: scale,
                         ignoringTextEncoderKeys: ignoringTextEncoderKeys)
    }

    static func merge(arrays: [String: MLXArray], metadata: [String: String] = [:], into unet: NFKMLXSDUNet,
                      scale: Float, ignoringTextEncoderKeys: Bool = false) throws -> Report {
        let parsed = try ParsedAdapter(arrays: arrays, metadata: metadata)
        let skipped = parsed.textEncoderKeyCount
        guard skipped == 0 || ignoringTextEncoderKeys else {
            throw NFKMLXError.weightsMismatch(
                "the adapter carries \(skipped) text-tower keys, which this merge does not reach; pass "
                + "ignoringTextEncoderKeys: true to merge its UNet half alone")
        }
        let layers = unetLayers(of: unet)
        let resolved = try parsed.targets[.unet, default: [:]].sorted(by: { $0.key < $1.key }).map { spelling, target in
            guard case .layer(_, let module)? = layers[spelling] else {
                throw NFKMLXError.weightsMismatch(missingLayerReason(spelling, component: .unet))
            }
            return (spelling, target, module)
        }
        for (spelling, target, module) in resolved {
            let scaling = parsed.scaling(of: spelling, component: .unet, target: target)
            try mergeTarget(target, into: module, named: spelling, strength: Float(Double(scale) * scaling))
        }
        return Report(mergedLayers: resolved.count, skippedTextEncoderKeys: skipped)
    }

    // MARK: Holding adapters unmerged

    /// What holding adapters did: the layers each component took detours on, and the bytes the detours
    /// hold.
    struct AttachReport {
        var adaptedLayers: [Component: Int] = [:]
        var byteCount = 0
    }

    /// Holds each adapter beside the weights it adapts, in order: adapter `i` reads its strength from
    /// ``NFKMLXAdapterRun/strengths`` at index `offset + i`, else runs at its own `strength`.
    ///
    /// @discussion A run adds `strength · scaling · up(down(x))` to each adapted layer's output, as PEFT's
    /// unmerged LoRA layer does, so the base weights stay as loaded. Each detour holds its arrays at the
    /// type of the weight it sits beside. Every adapter is read and checked against the model before any
    /// layer changes.
    ///
    /// - Throws: `NFKMLXError.loRANotApplicable` when a file is in no layout this reader knows, adapts a
    ///   tower the model does not have, names a layer the model lacks, or a shape disagrees.
    @discardableResult
    static func attach(_ adapters: [(url: URL, strength: Float)], startingAt offset: Int = 0, unet: NFKMLXSDUNet,
                       textEncoder: NFKMLXSDTextEncoderNet?,
                       secondaryTextEncoder: NFKMLXSDTextEncoderNet?) throws -> AttachReport {
        let models: [Component: Module] = [.unet: unet, .textEncoder: textEncoder, .textEncoder2: secondaryTextEncoder]
            .compactMapValues { $0 }
        let lookups: [Component: [String: Layer]] = [
            .unet: unetLayers(of: unet),
            .textEncoder: textEncoder.map(textEncoderLayers(of:)) ?? [:],
            .textEncoder2: secondaryTextEncoder.map(textEncoderLayers(of:)) ?? [:],
        ]
        var layerDetours = [Component: [String: [NFKMLXLowRankDetour]]]()
        var fusedDetours = [(NFKMLXFusedDetours, Int, NFKMLXLowRankDetour)]()
        var report = AttachReport()
        for (index, adapter) in adapters.enumerated() {
            let name = adapter.url.lastPathComponent
            let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: adapter.url)
            let parsed: ParsedAdapter
            do {
                parsed = try ParsedAdapter(arrays: arrays, metadata: metadata)
            } catch {
                throw NFKMLXError.loRANotApplicable("\(name): \(error)")
            }
            for (component, targets) in parsed.targets where !targets.isEmpty {
                guard models[component] != nil else {
                    throw NFKMLXError.loRANotApplicable(
                        "\(name) adapts \(component.title), which this model does not have: the adapter was "
                        + "trained for a different Stable Diffusion release")
                }
                for (spelling, target) in targets {
                    guard let layer = lookups[component]?[spelling] else {
                        throw NFKMLXError.loRANotApplicable("\(name): " + missingLayerReason(spelling, component: component))
                    }
                    let scale = parsed.scaling(of: spelling, component: component, target: target)
                    let detour = try makeDetour(target, beside: layer, named: "\(name)'s \(spelling)",
                                                scale: scale, adapter: offset + index, strength: adapter.strength)
                    report.byteCount += detour.byteCount
                    switch layer {
                    case .layer(let path, _):
                        layerDetours[component, default: [:]][path, default: []].append(detour)
                    case .fused(let attention, let part, _):
                        fusedDetours.append((attention.projectionDetours, part, detour))
                    }
                    report.adaptedLayers[component, default: 0] += 1
                }
            }
        }
        for (component, detours) in layerDetours {
            if let model = models[component] {
                try NFKMLXAdapterLayers.attach(detours, to: model)
            }
        }
        for (detours, part, detour) in fusedDetours {
            detours.parts[part].append(detour)
        }
        return report
    }

    // MARK: Parsing

    /// The parts of a model a released adapter reaches.
    enum Component: Hashable {
        case unet, textEncoder, textEncoder2

        var title: String {
            switch self {
            case .unet: return "the UNet"
            case .textEncoder: return "the first text tower"
            case .textEncoder2: return "the second text tower"
            }
        }

        /// The PEFT metadata prefix diffusers saves the component's configuration under.
        var peftPrefix: String {
            switch self {
            case .unet: return "unet."
            case .textEncoder: return "text_encoder."
            case .textEncoder2: return "text_encoder_2."
            }
        }
    }

    struct Target {
        var down: MLXArray?
        var up: MLXArray?
        var alpha: Float?
    }

    /// An adapter file's targets per component, each keyed by its spelling: a diffusers or `transformers`
    /// module path for PEFT, and `kohya:` with the underscored path for kohya.
    struct ParsedAdapter {
        var targets: [Component: [String: Target]] = [:]
        var textEncoderKeyCount = 0
        private var peft: [Component: NFKMLXWanLoRA.PEFTConfiguration] = [:]

        init(arrays: [String: MLXArray], metadata: [String: String]) throws {
            for (key, value) in arrays {
                guard let (component, spelling, part) = Self.split(key) else {
                    throw NFKMLXError.weightsMismatch(Self.unreadableReason(key))
                }
                if component != .unet {
                    textEncoderKeyCount += 1
                }
                var target = targets[component, default: [:]][spelling] ?? Target()
                switch part {
                case .down: target.down = value
                case .up: target.up = value
                case .alpha: target.alpha = value.asType(.float32).item(Float.self)
                }
                targets[component, default: [:]][spelling] = target
            }
            for component in targets.keys {
                peft[component] = try NFKMLXWanLoRA.peftConfiguration(metadata, component: component.peftPrefix)
            }
        }

        /// The scaling of one target: `alpha / rank` where the file states an alpha beside it, PEFT's
        /// configured scaling where diffusers recorded one, and 1 otherwise.
        func scaling(of spelling: String, component: Component, target: Target) -> Double {
            let rank = target.down?.dim(0)
            if let alpha = target.alpha, let rank {
                return Double(alpha) / Double(rank)
            }
            return peft[component]?.preciseScaling(spelling, rank: rank) ?? 1
        }

        private enum Part { case down, up, alpha }

        private static let kohyaPrefixes: [(String, Component)] = [
            ("lora_unet_", .unet), ("lora_te1_", .textEncoder), ("lora_te2_", .textEncoder2), ("lora_te_", .textEncoder),
        ]
        private static let peftPrefixes: [(String, Component)] = [
            ("unet.", .unet), ("text_encoder_2.", .textEncoder2), ("text_encoder.", .textEncoder),
        ]

        /// The key's component, its target's spelling, and which part it holds.
        private static func split(_ key: String) -> (Component, String, Part)? {
            for (prefix, component) in kohyaPrefixes where key.hasPrefix(prefix) {
                let body = String(key.dropFirst(prefix.count))
                for (suffix, part) in [(".lora_down.weight", Part.down), (".lora_up.weight", .up), (".alpha", .alpha)]
                where body.hasSuffix(suffix) {
                    return (component, kohyaSpelling(String(body.dropLast(suffix.count))), part)
                }
                return nil
            }
            for (prefix, component) in peftPrefixes where key.hasPrefix(prefix) {
                let body = String(key.dropFirst(prefix.count))
                for (suffix, part) in [(".lora_A.weight", Part.down), (".lora_B.weight", .up),
                                       (".lora.down.weight", .down), (".lora.up.weight", .up), (".alpha", .alpha)]
                where body.hasSuffix(suffix) {
                    return (component, String(body.dropLast(suffix.count)), part)
                }
                return nil
            }
            return nil
        }

        private static func unreadableReason(_ key: String) -> String {
            if key.contains("hada_") || key.contains("lokr_") {
                return "\(key) belongs to a LoHa or LoKr adapter, which this reader does not take; it reads LoRA "
                    + "adapters in the kohya and PEFT layouts"
            }
            if key.contains("dora_scale") || key.contains("lora_magnitude_vector") {
                return "\(key) belongs to a DoRA adapter, which this reader does not take"
            }
            return "\(key) is not a Stable Diffusion LoRA key; the reader takes the kohya "
                + "(lora_unet_ / lora_te_ …lora_down / lora_up / alpha) and PEFT (unet. / text_encoder. …lora_A / "
                + "lora_B) layouts"
        }
    }

    /// The kohya spelling joins a module path with underscores, which the module names also contain, so a
    /// path is never recovered by splitting: each layer of the model is spelled the same way and looked up.
    private static func kohyaSpelling(_ underscored: String) -> String { "kohya:" + underscored }

    private static func underscored(_ path: String) -> String {
        path.replacingOccurrences(of: ".", with: "_")
    }

    private static func missingLayerReason(_ spelling: String, component: Component) -> String {
        let name = spelling.hasPrefix("kohya:") ? String(spelling.dropFirst("kohya:".count)) : spelling
        return "the adapter names \(name), which \(component.title) has no layer for: the adapter was trained "
            + "for a different Stable Diffusion geometry"
    }

    // MARK: The model's layers

    /// Where one target lands: a whole layer, or one part of a fused projection of the given width.
    enum Layer {
        case layer(String, Module)
        case fused(NFKCLIPAttention, Int, Int)
    }

    /// Every linear and convolution of the UNet under each spelling an adapter may use: the diffusers path,
    /// its kohya spelling, and the kohya spelling of the original block name.
    static func unetLayers(of unet: NFKMLXSDUNet) -> [String: Layer] {
        var layers = [String: Layer]()
        for (path, module) in unet.leafModules().flattened() {
            guard module is Linear || module is Conv2d else { continue }
            let diffusers = diffusersPath(path)
            let layer = Layer.layer(path, module)
            layers[diffusers] = layer
            layers[kohyaSpelling(underscored(diffusers))] = layer
            if let original = originalPath(diffusers, configuration: unet.configuration) {
                layers[kohyaSpelling(underscored(original))] = layer
            }
        }
        return layers
    }

    /// Every adaptable layer of a CLIP text tower under its `transformers` path and that path's kohya
    /// spelling: the output projection and the two MLP layers, and the query, key, and value parts of the
    /// fused input projection.
    static func textEncoderLayers(of net: NFKMLXSDTextEncoderNet) -> [String: Layer] {
        var layers = [String: Layer]()
        func add(_ name: String, _ layer: Layer) {
            layers[name] = layer
            layers[kohyaSpelling(underscored(name))] = layer
        }
        let width = net.configuration.width
        for (index, block) in net.transformer.resblocks.enumerated() {
            let prefix = "text_model.encoder.layers.\(index)."
            let path = "transformer.resblocks.\(index)."
            add(prefix + "self_attn.out_proj", .layer(path + "attn.out_proj", block.attn.outProj))
            add(prefix + "mlp.fc1", .layer(path + "mlp.c_fc", block.mlp.cFc))
            add(prefix + "mlp.fc2", .layer(path + "mlp.c_proj", block.mlp.cProj))
            for (part, name) in ["q_proj", "k_proj", "v_proj"].enumerated() {
                add(prefix + "self_attn." + name, .fused(block.attn, part, width))
            }
        }
        return layers
    }

    /// The inverse of ``NFKMLXStableDiffusionModels/remapUNetKey(_:)``: this module's path spelled as
    /// the reference spells it.
    static func diffusersPath(_ path: String) -> String {
        var name = path
        name = name.replacingOccurrences(of: ".proj_in_conv", with: ".proj_in")
        name = name.replacingOccurrences(of: ".proj_out_conv", with: ".proj_out")
        name = name.replacingOccurrences(of: ".proj_in_linear", with: ".proj_in")
        name = name.replacingOccurrences(of: ".proj_out_linear", with: ".proj_out")
        name = name.replacingOccurrences(of: ".attn1.to_out", with: ".attn1.to_out.0")
        name = name.replacingOccurrences(of: ".attn2.to_out", with: ".attn2.to_out.0")
        name = name.replacingOccurrences(of: ".ff.proj", with: ".ff.net.0.proj")
        name = name.replacingOccurrences(of: ".ff.out", with: ".ff.net.2")
        return name
    }

    /// A diffusers UNet path spelled with the original model's block names, which number the input and
    /// output blocks through the levels in order, or nil for a layer the original names elsewhere.
    ///
    /// @discussion With `L` levels of `P` layers each: input block 0 is `conv_in`; level `l`'s layer `j` is
    /// input block `1 + l·(P+1) + j`, its residual block at `.0` and its transformer at `.1`; level `l`'s
    /// downsampler is input block `(l+1)·(P+1)`, `.0.op`. The middle block is the first residual block at
    /// `.0`, the transformer at `.1`, and the second residual block at `.2`. Up level `u`'s layer `j` (of
    /// `P+1`) is output block `u·(P+1) + j`, its upsampler after the transformer where the level attends.
    static func originalPath(_ path: String, configuration: NFKMLXSDUNetConfiguration) -> String? {
        let parts = path.split(separator: ".").map(String.init)
        let perLevel = configuration.layersPerBlock + 1
        func rest(_ from: Int) -> String { parts.dropFirst(from).joined(separator: ".") }
        func residual(_ from: Int) -> String? {
            guard parts.count == from + 1 else { return nil }
            switch parts[from] {
            case "conv1": return "in_layers.2"
            case "conv2": return "out_layers.3"
            case "time_emb_proj": return "emb_layers.1"
            case "conv_shortcut": return "skip_connection"
            default: return nil
            }
        }
        switch path {
        case "conv_in": return "input_blocks.0.0"
        case "conv_out": return "out.2"
        case "time_embedding.linear_1": return "time_embed.0"
        case "time_embedding.linear_2": return "time_embed.2"
        case "add_embedding.linear_1": return "label_emb.0.0"
        case "add_embedding.linear_2": return "label_emb.0.2"
        default: break
        }
        guard parts.count >= 4 else { return nil }
        switch (parts[0], Int(parts[1]), parts[2], Int(parts[3])) {
        case ("down_blocks", let level?, "resnets", let layer?):
            return residual(4).map { "input_blocks.\(1 + level * perLevel + layer).0.\($0)" }
        case ("down_blocks", let level?, "attentions", let layer?):
            return "input_blocks.\(1 + level * perLevel + layer).1.\(rest(4))"
        case ("down_blocks", let level?, "downsamplers", 0?) where rest(4) == "conv":
            return "input_blocks.\((level + 1) * perLevel).0.op"
        case ("mid_block", _, _, _):
            switch (parts[1], parts[2]) {
            case ("resnets", let index):
                guard let layer = Int(index), let inner = residual(3) else { return nil }
                return "middle_block.\(layer * 2).\(inner)"
            case ("attentions", "0"):
                return "middle_block.1.\(rest(3))"
            default:
                return nil
            }
        case ("up_blocks", let level?, "resnets", let layer?):
            return residual(4).map { "output_blocks.\(level * perLevel + layer).0.\($0)" }
        case ("up_blocks", let level?, "attentions", let layer?):
            return "output_blocks.\(level * perLevel + layer).1.\(rest(4))"
        case ("up_blocks", let level?, "upsamplers", 0?) where rest(4) == "conv":
            let attends = Array(configuration.attends.reversed())[level]
            return "output_blocks.\(level * perLevel + configuration.layersPerBlock).\(attends ? 2 : 1).conv"
        default:
            return nil
        }
    }

    // MARK: Detours

    /// A target's detour beside `layer`: the down weight as a `[rank, inputs]` matrix for a linear layer or
    /// a `[rank, kH, kW, inputs]` kernel for a convolution, the up weight as `[outputs, rank]`, both at the
    /// type of the layer's weight.
    private static func makeDetour(_ target: Target, beside layer: Layer, named name: String, scale: Double,
                                   adapter: Int, strength: Float) throws -> NFKMLXLowRankDetour {
        guard let down = target.down, let up = target.up else {
            throw NFKMLXError.loRANotApplicable("\(name) lacks its down or up weight")
        }
        let rank = down.dim(0)
        guard up.ndim >= 2, up.dim(1) == rank, up.size == up.dim(0) * rank else {
            throw NFKMLXError.loRANotApplicable(
                "\(name) has a down weight of rank \(rank) and an up weight of shape \(up.shape)")
        }
        let upMatrix = up.reshaped([up.dim(0), rank])
        let weight: MLXArray
        let expectedDown: [Int]
        let downShaped: MLXArray
        switch layer {
        case .layer(_, let convolution as Conv2d):
            weight = convolution.weight
            let (height, width, inputs) = (weight.dim(1), weight.dim(2), weight.dim(3))
            expectedDown = [rank, height, width, inputs]
            downShaped = down.ndim == 4 ? down.transposed(0, 2, 3, 1) : down.reshaped([rank, 1, 1, -1])
        case .layer(_, let linear as Linear):
            weight = linear.weight
            expectedDown = [rank, weight.dim(1)]
            downShaped = down.reshaped([rank, -1])
        case .fused(let attention, _, let width):
            weight = attention.inProjWeight
            expectedDown = [rank, width]
            downShaped = down.reshaped([rank, -1])
        case .layer(let path, _):
            throw NFKMLXError.loRANotApplicable("\(name) targets \(path), which is neither a linear nor a convolution layer")
        }
        let outputs: Int
        if case .fused(_, _, let width) = layer {
            outputs = width
        } else {
            outputs = weight.dim(0)
        }
        guard downShaped.shape == expectedDown, upMatrix.dim(0) == outputs else {
            throw NFKMLXError.loRANotApplicable(
                "\(name) is \(down.shape)·\(up.shape), which does not fit a layer of \(outputs) outputs over "
                + "\(expectedDown.dropFirst().map(String.init).joined(separator: "×")) inputs: the adapter was trained "
                + "for a different Stable Diffusion geometry")
        }
        let downStored = downShaped.asType(weight.dtype)
        let upStored = upMatrix.asType(weight.dtype)
        eval(downStored, upStored)
        return NFKMLXLowRankDetour(down: downStored, up: upStored, scale: scale, adapter: adapter, strength: strength)
    }

    // MARK: The merge

    private static func mergeTarget(_ target: Target, into module: Module, named name: String, strength: Float) throws {
        guard let down = target.down, let up = target.up else {
            throw NFKMLXError.weightsMismatch("the adapter's \(name) lacks its down or up weight")
        }
        let rank = down.dim(0)
        guard up.dim(1) == rank else {
            throw NFKMLXError.weightsMismatch(
                "the adapter's \(name) has a down weight of rank \(rank) and an up weight of rank \(up.dim(1))")
        }
        let down32 = down.asType(.float32)
        let up32 = up.asType(.float32)

        if let linear = module as? Linear {
            guard down32.ndim == 2 || down32.ndim == 4 else {
                throw NFKMLXError.weightsMismatch("the adapter's \(name) down weight is \(down32.ndim)-D")
            }
            // A 1×1 convolution kernel on a linear target is the same matrix with two unit axes.
            let downMatrix = down32.reshaped([rank, -1])
            let upMatrix = up32.reshaped([up32.dim(0), rank])
            let delta = matmul(upMatrix, downMatrix)
            try merge(delta: delta, into: linear, named: name, strength: strength)
            return
        }
        if let conv = module as? Conv2d {
            guard down32.ndim == 4, up32.ndim == 4 else {
                throw NFKMLXError.weightsMismatch("the adapter's \(name) is a convolution and needs 4-D weights")
            }
            // `[out, rank]` contracted with `[rank, in, kH, kW]` gives the reference's `[out, in, kH, kW]`,
            // which the module stores as `[out, kH, kW, in]`.
            let upMatrix = up32.reshaped([up32.dim(0), rank])
            let downMatrix = down32.reshaped([rank, -1])
            let delta = matmul(upMatrix, downMatrix)
                .reshaped([up32.dim(0), down32.dim(1), down32.dim(2), down32.dim(3)])
                .transposed(0, 2, 3, 1)
            try merge(delta: delta, into: conv, named: name, strength: strength)
            return
        }
        throw NFKMLXError.weightsMismatch("the adapter's \(name) is neither a linear nor a convolution here")
    }

    private static func merge(delta: MLXArray, into module: Module, named name: String, strength: Float) throws {
        guard let weight = module.parameters().flattened().first(where: { $0.0 == "weight" })?.1 else {
            throw NFKMLXError.weightsMismatch("\(name) has no weight to merge into")
        }
        guard weight.shape == delta.shape else {
            throw NFKMLXError.weightsMismatch(
                "the adapter's \(name) is \(delta.shape) where the UNet's weight is \(weight.shape)")
        }
        let merged = (weight.asType(.float32) + delta * strength).asType(weight.dtype)
        module.update(parameters: ModuleParameters.unflattened([("weight", merged)]))
    }
}
