//
//  NFKMLXWanAnimateRelease.swift
//  InferKitMLX
//
//  Reading a diffusers Wan 2.2 Animate 2 release: `transformer/` (WanAnimate2Transformer3DModel),
//  `text_encoder/` (umT5-XXL), `tokenizer/`, `image_encoder/` (CLIPVisionModel), `vae/` (the Wan 2.1
//  autoencoder), and `scheduler/`.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The two Wan 2.2 Animate 2 releases.
@objc(NFKMLXWanAnimateVariant)
public enum NFKMLXWanAnimateVariant: Int, Sendable {
    /// `Wan-AI/Wan2.2-Animate-2-14B-Diffusers`, sampled over 40 steps at a guidance scale of 3.
    case standard
    /// `Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers`, distilled to 10 Euler steps without guidance.
    case distilled

    /// The Hugging Face repository the release downloads from.
    var repository: String {
        switch self {
        case .standard: return "Wan-AI/Wan2.2-Animate-2-14B-Diffusers"
        case .distilled: return "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers"
        }
    }

    /// The reference pipeline's step count for the release.
    var defaultSteps: Int { self == .standard ? 40 : 10 }

    /// The reference pipeline's guidance scale for the release.
    var defaultGuidance: Float { self == .standard ? 3 : 1 }
}

/// How each stage of a diffusers Wan Animate release is read.
enum NFKMLXWanAnimateRelease {

    /// The transformer geometry a `transformer/config.json` states.
    static func transformerConfiguration(fromHuggingFace url: URL) throws -> NFKMLXWanAnimateConfiguration {
        let json = try NFKMLXWanRelease.json(url)
        if let kind = json["_class_name"] as? String, kind != "WanAnimate2Transformer3DModel" {
            throw NFKMLXError.unsupportedConfiguration("this reads a WanAnimate2Transformer3DModel, not \(kind)")
        }
        let base = NFKMLXWanAnimateConfiguration.base
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func bool(_ key: String, _ fallback: Bool) -> Bool { (json[key] as? NSNumber)?.boolValue ?? fallback }
        return NFKMLXWanAnimateConfiguration(
            inChannels: int("in_dim", base.inChannels), outChannels: int("out_dim", base.outChannels),
            dim: int("dim", base.dim), heads: int("num_heads", base.heads), layers: int("num_layers", base.layers),
            ffnDim: int("ffn_dim", base.ffnDim), textDim: int("text_dim", base.textDim),
            textLength: int("text_len", base.textLength), freqDim: int("freq_dim", base.freqDim),
            patchSize: json["patch_size"] as? [Int] ?? base.patchSize,
            eps: (json["eps"] as? NSNumber)?.floatValue ?? base.eps,
            crossAttentionNorm: bool("cross_attn_norm", base.crossAttentionNorm),
            useImageEmbedding: bool("use_img_emb", base.useImageEmbedding),
            referOffsetT: int("refer_offset_t", base.referOffsetT), referOffsetH: int("refer_offset_h", base.referOffsetH),
            referOffsetW: int("refer_offset_w", base.referOffsetW), referStride: int("refer_stride", base.referStride))
    }

    /// The block a transformer key belongs to and its name inside the block.
    static func block(ofKey key: String) -> (layer: Int, name: String)? {
        NFKMLXLayerStream.layer(ofDecoderName: key, layersKey: "blocks")
    }

    /// A transformer tensor in the module's layout: the patch convolution moves from `[out, in, t, h, w]`
    /// to `[out, t, h, w, in]`.
    static func adapted(key: String, value: MLXArray) -> (String, MLXArray) {
        (key, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value)
    }

    /// What the transformer holds at `precision`, each block counted as a repeated layer.
    static func transformerFootprint(directory: URL, precision: NFKMLXWeightPrecision) throws -> NFKMLXStageFootprint {
        try NFKMLXLayerStream.footprint(directory: directory, precision: precision, includes: { _ in true },
                                        layerOf: { block(ofKey: $0)?.layer })
    }

    /// What the transformer holds with every block's linear layers packed to `bits` at `groupSize`: each
    /// packed matrix as its words plus a scale and a bias per group, everything else at `precision`.
    static func quantizedTransformerFootprint(directory: URL, precision: NFKMLXWeightPrecision, bits: Int,
                                              groupSize: Int) throws -> NFKMLXStageFootprint {
        var layers = [Int: Int](), fixed = 0
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            for (key, entry) in try NFKMLXSafetensors.entries(inFile: url) {
                let element = precision == .float32 ? 4 : (NFKMLXSafetensors.dtype(entry.dtype)?.size ?? 2)
                let count = entry.shape.reduce(1, *)
                let packs = block(ofKey: key) != nil && key.hasSuffix(".weight") && entry.shape.count == 2
                    && entry.shape[1] % groupSize == 0
                let held = packs ? count * bits / 8 + 2 * (count / groupSize) * element : count * element
                if let layer = block(ofKey: key)?.layer {
                    layers[layer, default: 0] += held
                } else {
                    fixed += held
                }
            }
        }
        let layerBytes = (0 ..< (layers.keys.max().map { $0 + 1 } ?? 0)).map { layers[$0] ?? 0 }
        return NFKMLXStageFootprint(bytes: fixed + layerBytes.reduce(0, +), layerBytes: layerBytes)
    }

    /// Loads a `transformer/` directory into `net`.
    ///
    /// - Parameters:
    ///   - streamed: the blocks left in the release and read in their turn on every pass.
    ///   - bits: packs every block's linear layers to this many bits as each block loads, so the stored
    ///     form is never held whole; nil keeps them at `precision`. A packed transformer streams nothing.
    ///   - adapters: released LoRA deltas (``NFKMLXWanLoRA``) merged in order, each block's before it packs.
    ///     A streamed transformer takes none: its blocks are read from the release on every pass.
    static func loadTransformer(into net: NFKMLXWanAnimateNet, fromDirectory directory: URL,
                                precision: NFKMLXWeightPrecision, streamed: Set<Int> = [], bits: Int? = nil,
                                groupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize,
                                adapters: [[String: NFKMLXWanLoRA.Delta]] = []) throws {
        guard adapters.isEmpty || streamed.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration(
                "a streamed transformer reads \(streamed.count) blocks from the release on every pass and takes no "
                + "adapter; load it with a residency that holds the transformer")
        }
        guard let bits else {
            let vacated = try NFKMLXLayerStream.vacating(streamed, of: net.blocks, prefix: "blocks.",
                                                         directory: directory, precision: precision,
                                                         layerOf: block(ofKey:))
            let arrays = try NFKMLXReleaseWeights.materializedArrays(
                inDirectory: directory, precision: precision,
                remap: { key in block(ofKey: key).map { streamed.contains($0.layer) } == true ? nil : key },
                transform: adapted)
            try NFKMLXWeights.apply(arrays + (vacated?.placeholders ?? []), to: net, verifyShapes: true)
            for deltas in adapters {
                try NFKMLXWanLoRA.apply(deltas, to: net)
            }
            net.layerStream = vacated?.stream
            vacated?.stream.prime()
            return
        }
        guard streamed.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration("a packed transformer holds every block; it does not stream")
        }
        guard (2 ... 8).contains(bits), [32, 64, 128].contains(groupSize) else {
            throw NFKMLXError.unsupportedConfiguration(
                "a packed load takes 2 to 8 bits at a group size of 32, 64, or 128, not \(bits) at \(groupSize)")
        }
        let expected = Set(net.parameters().flattened().map(\.0))
        var outside = [(String, MLXArray)]()
        var unread = Set(expected.filter { block(ofKey: $0) != nil })
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            var blocks = [Int: [(String, MLXArray)]]()
            for (key, value) in try NFKMLXWeights.loadCheckpoint(url: url).arrays where expected.contains(key) {
                let typed = precision == .float32 && value.dtype.isFloatingPoint ? value.asType(.float32) : value
                if let (layer, name) = block(ofKey: key) {
                    blocks[layer, default: []].append((name, typed))
                    unread.remove(key)
                } else {
                    outside.append(adapted(key: key, value: typed))
                }
            }
            // A block whose tensors span two shards packs once its last shard is read.
            for (layer, arrays) in blocks.sorted(by: { $0.key < $1.key }) {
                let module = net.blocks[layer]
                module.update(parameters: ModuleParameters.unflattened(arrays))
                let complete = !unread.contains { block(ofKey: $0)?.layer == layer }
                if complete {
                    for deltas in adapters {
                        try NFKMLXWanLoRA.apply(deltas, to: module, pathPrefix: "blocks.\(layer).")
                    }
                    try NFKMLXQuantization.quantize(module: module, bits: bits, groupSize: groupSize)
                    eval(module)
                    NFKMLXGPU.clearCache()
                }
            }
        }
        let supplied = Set(outside.map(\.0))
        let missing = Array(unread) + expected.filter { block(ofKey: $0) == nil && !supplied.contains($0) }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch("the release lacks \(missing.count) parameters, such as \(missing.sorted().prefix(3))")
        }
        // The packed blocks name their parameters apart from the stored ones, so the rest applies on its
        // own after the coverage check above.
        try NFKMLXWeights.apply(outside, to: net, strict: false, verifyShapes: true)
        for deltas in adapters {
            try NFKMLXWanLoRA.apply(deltas.filter { !$0.key.hasPrefix("blocks.") }, to: net)
        }
        eval(net)
        NFKMLXGPU.clearCache()
    }
}
