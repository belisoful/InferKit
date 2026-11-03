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
        NFKMLXWanBlockedLoad.block(ofKey: key)
    }

    /// A transformer tensor in the module's layout.
    static func adapted(key: String, value: MLXArray) -> (String, MLXArray) {
        NFKMLXWanBlockedLoad.adapted(key: key, value: value)
    }

    /// What the transformer holds at `precision`, each block counted as a repeated layer.
    static func transformerFootprint(directory: URL, precision: NFKMLXWeightPrecision) throws -> NFKMLXStageFootprint {
        try NFKMLXWanBlockedLoad.footprint(directory: directory, precision: precision)
    }

    /// What the transformer holds with every block's linear layers packed to `bits` at `groupSize`.
    static func quantizedTransformerFootprint(directory: URL, precision: NFKMLXWeightPrecision, bits: Int,
                                              groupSize: Int) throws -> NFKMLXStageFootprint {
        try NFKMLXWanBlockedLoad.quantizedFootprint(directory: directory, precision: precision, bits: bits,
                                                    groupSize: groupSize)
    }

    /// Loads a `transformer/` directory into `net`, as ``NFKMLXWanBlockedLoad/load(into:blocks:fromDirectory:precision:streamed:bits:groupSize:adapters:)``
    /// reads it.
    static func loadTransformer(into net: NFKMLXWanAnimateNet, fromDirectory directory: URL,
                                precision: NFKMLXWeightPrecision, streamed: Set<Int> = [], bits: Int? = nil,
                                groupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize,
                                adapters: [[String: NFKMLXWanLoRA.Delta]] = []) throws {
        net.layerStream = try NFKMLXWanBlockedLoad.load(into: net, blocks: net.blocks, fromDirectory: directory,
                                                        precision: precision, streamed: streamed, bits: bits,
                                                        groupSize: groupSize, adapters: adapters)
    }
}
