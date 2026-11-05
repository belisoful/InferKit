//
//  NFKMLXDecoderAdapters.swift
//  InferKitMLX
//
//  Released LoRA adapters for the language decoders, held beside a decoder's linear layers rather than
//  merged into them: PEFT's `adapter_model.safetensors` and mlx-lm's `adapters.safetensors`, each read with
//  the `adapter_config.json` beside it.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// Reads decoder adapters and holds them unmerged beside a decoder's linear layers.
///
/// @discussion
/// - PEFT: `…<layer>.lora_A.weight` `[rank, inputs]` and `…<layer>.lora_B.weight` `[outputs, rank]`, scaled
///   as PEFT scales them from the `adapter_config.json` beside the file: `lora_alpha / r`, `lora_alpha / √r`
///   for rsLoRA, with `rank_pattern` and `alpha_pattern` matched against the layer's path.
/// - mlx-lm: `…<layer>.lora_a` `[inputs, rank]` and `…<layer>.lora_b` `[rank, outputs]`, scaled by
///   `lora_parameters.scale`, as mlx-lm's `LoRALinear` adds `scale · (x·a)·b`.
///
/// A layer is named by the decoder's own module path: PEFT's `base_model.model.` wrapper is dropped, and a
/// multimodal release's `model.language_model.` reads as `model.`.
enum NFKMLXDecoderAdapters {

    /// The two file layouts a decoder adapter comes in.
    enum Layout {
        case peft
        case mlxLM
    }

    /// Holds each adapter beside `model`'s layers at its strength, in load order, and returns the strengths.
    /// `modulePath` turns a layer's stored name, read as above, into `model`'s module path, or nil for a layer
    /// `model` does not have: Whisper's adapters name transformers' modules, which the port names as OpenAI does.
    ///
    /// - Throws: `NFKMLXError.loRANotApplicable` when a file is in neither layout, has no configuration beside
    ///   it, is a DoRA, carries whole layers (`modules_to_save`), or adapts a layer `model` lacks or one that is
    ///   not linear. Every adapter is read before any layer changes.
    @discardableResult
    static func hold(_ adapters: [(url: URL, strength: Float)], in model: Module,
                     modulePath: (String) -> String? = { $0 }) throws -> [Float] {
        guard !adapters.isEmpty else { return [] }
        let layers = Dictionary(model.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        var detours = [String: [NFKMLXLowRankDetour]]()
        for (index, adapter) in adapters.enumerated() {
            for (path, detour) in try read(adapter.url, adapter: index, strength: adapter.strength, layers: layers,
                                           modulePath: modulePath) {
                detours[path, default: []].append(detour)
            }
        }
        try NFKMLXAdapterLayers.attach(detours, to: model)
        return adapters.map(\.strength)
    }

    /// The detours one adapter file places, by layer path, their arrays at the type of the layer they adapt.
    static func read(_ url: URL, adapter: Int, strength: Float, layers: [String: Module],
                     modulePath: (String) -> String? = { $0 }) throws -> [(String, NFKMLXLowRankDetour)] {
        let name = url.lastPathComponent
        let arrays = try MLX.loadArrays(url: url)
        let configuration = try self.configuration(beside: url)
        if let type = configuration["peft_type"] as? String, type.uppercased() != "LORA" {
            throw NFKMLXError.loRANotApplicable("\(name) is a \(type) adapter; a held adapter computes a LoRA's product")
        }
        let layout = try self.layout(of: arrays, name: name)
        var placed = [(String, NFKMLXLowRankDetour)]()
        switch layout {
        case .peft:
            if let saved = configuration["modules_to_save"] as? [Any], !saved.isEmpty {
                throw NFKMLXError.loRANotApplicable("\(name) carries whole layers (modules_to_save), which a held adapter cannot")
            }
            let peft = try NFKMLXWanLoRA.peftConfiguration(json: configuration)
            for key in arrays.keys.sorted() where key.hasSuffix(".lora_A.weight") {
                let stored = String(key.dropLast(".lora_A.weight".count))
                guard let up = arrays[stored + ".lora_B.weight"], let down = arrays[key] else {
                    throw NFKMLXError.loRANotApplicable("\(name) holds \(key) without its lora_B half")
                }
                let scale = peft.preciseScaling(layerPath(stored), rank: down.dim(0)) ?? 1
                let path = try mapped(stored, modulePath, name: name)
                placed.append((path, try detour(down: down, up: up, scale: scale, adapter: adapter, strength: strength,
                                                path: path, layers: layers, name: name)))
            }
        case .mlxLM:
            if (configuration["fine_tune_type"] as? String) == "dora" {
                throw NFKMLXError.loRANotApplicable("\(name) is a DoRA, which a held adapter does not compute")
            }
            let parameters = configuration["lora_parameters"] as? [String: Any]
            guard let scale = (parameters?["scale"] as? NSNumber)?.doubleValue else {
                throw NFKMLXError.loRANotApplicable("\(name)'s adapter_config.json states no lora_parameters.scale")
            }
            for key in arrays.keys.sorted() where key.hasSuffix(".lora_a") {
                let stored = String(key.dropLast(".lora_a".count))
                guard let b = arrays[stored + ".lora_b"], let a = arrays[key] else {
                    throw NFKMLXError.loRANotApplicable("\(name) holds \(key) without its lora_b half")
                }
                let path = try mapped(stored, modulePath, name: name)
                placed.append((path, try detour(down: a.transposed(1, 0), up: b.transposed(1, 0), scale: scale,
                                                adapter: adapter, strength: strength, path: path, layers: layers,
                                                name: name)))
            }
        }
        guard !placed.isEmpty else {
            throw NFKMLXError.loRANotApplicable("\(name) adapts no layer")
        }
        return placed
    }

    /// The layout of an adapter's tensors. A DoRA's magnitudes, an AdaLoRA's singular values, and an embedding's
    /// adapter are refused.
    static func layout(of arrays: [String: MLXArray], name: String) throws -> Layout {
        let keys = arrays.keys
        if keys.contains(where: { $0.hasSuffix(".lora_E") }) {
            throw NFKMLXError.loRANotApplicable("\(name) is an AdaLoRA, whose singular values a held adapter does not compute")
        }
        if keys.contains(where: { $0.contains("lora_magnitude_vector") }) {
            throw NFKMLXError.loRANotApplicable("\(name) is a DoRA, which a held adapter does not compute")
        }
        if keys.contains(where: { $0.contains("lora_embedding_A") || $0.contains("lora_embedding_B") }) {
            throw NFKMLXError.loRANotApplicable("\(name) adapts an embedding, which a held adapter does not reach")
        }
        if keys.contains(where: { $0.hasSuffix(".lora_A.weight") }) {
            return .peft
        }
        if keys.contains(where: { $0.hasSuffix(".lora_a") }) {
            return .mlxLM
        }
        throw NFKMLXError.loRANotApplicable("\(name) holds no LoRA tensors in the PEFT or mlx-lm layout")
    }

    /// The `adapter_config.json` beside `url`, which both layouts write and which states the adapter's scaling.
    static func configuration(beside url: URL) throws -> [String: Any] {
        let file = url.deletingLastPathComponent().appendingPathComponent("adapter_config.json")
        guard let data = try? Data(contentsOf: file) else {
            throw NFKMLXError.loRANotApplicable(
                "\(url.lastPathComponent) has no adapter_config.json beside it, which states its scaling")
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.loRANotApplicable("the adapter_config.json beside \(url.lastPathComponent) is not an object")
        }
        return json
    }

    /// A stored layer name read as the decoder's module path.
    static func layerPath(_ stored: String) -> String {
        var path = stored
        if path.hasPrefix("base_model.model.") {
            path = String(path.dropFirst("base_model.model.".count))
        }
        if path.hasPrefix("model.language_model.") {
            path = "model." + path.dropFirst("model.language_model.".count)
        }
        return path
    }

    /// `stored` read as a module path of the model.
    private static func mapped(_ stored: String, _ modulePath: (String) -> String?, name: String) throws -> String {
        guard let path = modulePath(layerPath(stored)) else {
            throw NFKMLXError.loRANotApplicable("\(name) adapts \(stored), which names no layer of this model")
        }
        return path
    }

    /// One detour beside the linear layer at `path`, its halves `down` `[rank, inputs]` and `up`
    /// `[outputs, rank]` checked against the layer's shape and held at the layer's type.
    static func detour(down: MLXArray, up: MLXArray, scale: Double, adapter: Int, strength: Float,
                       path: String, layers: [String: Module], name: String) throws -> NFKMLXLowRankDetour {
        let shape: (outputs: Int, inputs: Int)
        let type: DType
        switch layers[path] {
        case let quantized as QuantizedLinear:
            shape = quantized.shape
            type = quantized.scales.dtype
        case let linear as Linear:
            shape = (linear.weight.dim(0), linear.weight.dim(1))
            type = linear.weight.dtype
        default:
            throw NFKMLXError.loRANotApplicable("\(name) adapts \(path), which is not a linear layer of this model")
        }
        guard down.ndim == 2, up.ndim == 2, down.dim(0) == up.dim(1), down.dim(1) == shape.inputs,
              up.dim(0) == shape.outputs else {
            throw NFKMLXError.loRANotApplicable(
                "\(name) adapts \(path) with halves \(down.shape) and \(up.shape), which do not fit its "
                + "\(shape.outputs) × \(shape.inputs) weight")
        }
        return NFKMLXLowRankDetour(down: down.asType(type), up: up.asType(type), scale: scale, adapter: adapter,
                                   strength: strength)
    }
}
