//
//  NFKMLXSDLoRA.swift
//  InferKitMLX
//
//  Released low-rank adapters for the Stable Diffusion UNet: the files the LCM-LoRA and SDXL-Lightning
//  distillations publish, and fine-tunes in the same layouts. An adapter merges into the UNet's
//  weights, so a run costs nothing beyond the base model and the result is an ordinary `NFKMLXSDUNet`.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// Reads a released Stable Diffusion LoRA and merges it into a UNet.
///
/// Two key layouts are read. The kohya layout over diffusers module names, which the LCM-LoRA and
/// SDXL-Lightning releases use, spells each target `lora_unet_<module path with underscores>` and
/// stores `.lora_down.weight`, `.lora_up.weight`, and a scalar `.alpha`. The PEFT layout, which
/// diffusers writes, spells the target `unet.<module path>` and stores `.lora_A.weight` and
/// `.lora_B.weight` with no alpha tensor; the strength is then 1, and an alpha recorded in the file's
/// metadata is not read. A kohya file written against the original model's block names
/// (`lora_unet_input_blocks_…`) is not translated and is refused. A convolution target stores its down
/// weight as a `[rank, in, kH, kW]` kernel and its up weight as a `[out, rank, 1, 1]` one. An adapter
/// that also carries text-tower keys (`lora_te…`, `text_encoder…`) is refused unless the caller asks
/// to merge the UNet half alone, because a silently half-merged adapter is a different model; the
/// measured releases carry none.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXSDLoRA {

    /// What a merge did: how many layers it changed and how many text-tower keys it left alone.
    public struct Report: Sendable {
        public let mergedLayers: Int
        public let skippedTextEncoderKeys: Int
    }

    /// Merges the adapter at `url` into `unet` at `scale`.
    ///
    /// The delta for a linear target is `scale · (alpha / rank) · up · down`; for a convolution the up
    /// weight contracts the down kernel's rank axis. The product forms in float32 and takes the base
    /// weight's type, as diffusers' `fuse_lora` does. A key that names no layer of this UNet is an error,
    /// so an adapter trained for another geometry fails before it changes a weight.
    ///
    /// - Parameter ignoringTextEncoderKeys: merge the UNet half of an adapter that also adapts the text
    ///   towers, and count the keys left alone in the report. False, the default, refuses such a file.
    /// - Throws: `NFKMLXError.weightsMismatch` when a target is missing from the UNet, a shape disagrees,
    ///   a target lacks its pair of weights, or the file carries text-tower keys without the opt-in.
    @discardableResult
    public static func merge(adapterURL url: URL, into unet: NFKMLXSDUNet, scale: Float = 1,
                             ignoringTextEncoderKeys: Bool = false) throws -> Report {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        return try merge(arrays: checkpoint.arrays, into: unet, scale: scale,
                         ignoringTextEncoderKeys: ignoringTextEncoderKeys)
    }

    static func merge(arrays: [String: MLXArray], into unet: NFKMLXSDUNet, scale: Float,
                      ignoringTextEncoderKeys: Bool = false) throws -> Report {
        let skipped = arrays.keys.filter(isTextEncoderKey).count
        guard skipped == 0 || ignoringTextEncoderKeys else {
            throw NFKMLXError.weightsMismatch(
                "the adapter carries \(skipped) text-tower keys, which this merge does not reach; pass "
                + "ignoringTextEncoderKeys: true to merge its UNet half alone")
        }
        let targets = try adapterTargets(arrays)
        let layers = adaptableLayers(of: unet)
        var merged = 0
        for (spelling, target) in targets.sorted(by: { $0.key < $1.key }) {
            guard let module = layers[spelling] else {
                throw NFKMLXError.weightsMismatch(
                    "the adapter names \(spelling), which this UNet has no layer for: the adapter was "
                    + "trained for a different Stable Diffusion geometry, or it spells the original "
                    + "model's block names (lora_unet_input_blocks_…), which this reader does not translate")
            }
            try apply(target, to: module, named: spelling, scale: scale)
            merged += 1
        }
        return Report(mergedLayers: merged, skippedTextEncoderKeys: skipped)
    }

    // MARK: Key layouts

    private struct Target {
        var down: MLXArray?
        var up: MLXArray?
        var alpha: Float?
    }

    private static func isTextEncoderKey(_ key: String) -> Bool {
        key.hasPrefix("lora_te") || key.hasPrefix("text_encoder")
    }

    /// Groups the adapter's arrays by target, keyed by the diffusers module path (`down_blocks.0.…`).
    private static func adapterTargets(_ arrays: [String: MLXArray]) throws -> [String: Target] {
        var targets = [String: Target]()
        for (key, value) in arrays where !isTextEncoderKey(key) {
            guard let (path, part) = split(key) else {
                throw NFKMLXError.weightsMismatch(
                    "\(key) is not a Stable Diffusion LoRA key; the reader takes the kohya "
                    + "(lora_unet_…lora_down / lora_up / alpha) and PEFT (unet.…lora_A / lora_B) layouts")
            }
            var target = targets[path] ?? Target()
            switch part {
            case .down: target.down = value
            case .up: target.up = value
            case .alpha: target.alpha = value.asType(.float32).item(Float.self)
            }
            targets[path] = target
        }
        return targets
    }

    private enum Part { case down, up, alpha }

    /// The target's diffusers path and which of its three parts a key holds.
    private static func split(_ key: String) -> (String, Part)? {
        if key.hasPrefix("lora_unet_") {
            let body = String(key.dropFirst("lora_unet_".count))
            for (suffix, part) in [(".lora_down.weight", Part.down), (".lora_up.weight", .up), (".alpha", .alpha)]
            where body.hasSuffix(suffix) {
                return (kohyaPath(String(body.dropLast(suffix.count))), part)
            }
            return nil
        }
        if key.hasPrefix("unet.") {
            let body = String(key.dropFirst("unet.".count))
            for (suffix, part) in [(".lora_A.weight", Part.down), (".lora_B.weight", .up),
                                   (".lora.down.weight", .down), (".lora.up.weight", .up), (".alpha", .alpha)]
            where body.hasSuffix(suffix) {
                return (String(body.dropLast(suffix.count)), part)
            }
            return nil
        }
        return nil
    }

    /// The kohya spelling joins a diffusers path with underscores, which the module names also contain.
    /// The path is recovered by matching the UNet's own layers, spelled the same way.
    private static func kohyaPath(_ spelling: String) -> String { "kohya:" + spelling }

    // MARK: The UNet's layers

    /// Every linear and convolution of the UNet, under the diffusers path and under the kohya spelling of
    /// that path.
    private static func adaptableLayers(of unet: NFKMLXSDUNet) -> [String: Module] {
        var layers = [String: Module]()
        for (path, module) in unet.leafModules().flattened() {
            guard module is Linear || module is Conv2d else { continue }
            let diffusers = diffusersPath(path)
            layers[diffusers] = module
            layers[kohyaPath(diffusers.replacingOccurrences(of: ".", with: "_"))] = module
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

    // MARK: The merge

    private static func apply(_ target: Target, to module: Module, named name: String, scale: Float) throws {
        guard let down = target.down, let up = target.up else {
            throw NFKMLXError.weightsMismatch("the adapter's \(name) lacks its down or up weight")
        }
        let rank = down.dim(0)
        guard up.dim(1) == rank else {
            throw NFKMLXError.weightsMismatch(
                "the adapter's \(name) has a down weight of rank \(rank) and an up weight of rank \(up.dim(1))")
        }
        // A PEFT file carries no alpha; its scaling is already in the weights.
        let strength = scale * (target.alpha.map { $0 / Float(rank) } ?? 1)
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
