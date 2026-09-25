//
//  NFKMLXSa2VASAM3.swift
//  InferKitMLX
//
//  Sa2VA's SAM 3 grounding (`ByteDance/Sa2VA-Qwen3-VL-4B-SAM3`): SAM 3's tracker driven by a `[SEG]`
//  embedding. For the conditioning frame a single image is, that tracker is SAM 2's: the SAM 3 ViT at
//  1008 pixels (a 72-token grid), the neck's tracker branch (`sam2_convs`, a second copy of the FPN
//  levels at scales 4, 2, and 1, the 0.5 level dropped), the tracker's `no_mem_embed` added to the
//  coarsest level, and SAM 2's prompt encoder and mask decoder with the `[SEG]` embedding appended to
//  the sparse prompt. The checkpoint keeps the original `sam3` package's names (`trunk.blocks.N` with a
//  fused `qkv`, `ln_pre`, `pos_embed` with a class-token row), which map onto the toolkit's SAM 3 ViT.
//

import Foundation
import MLX
import MLXNN

/// A grounding encoder a Sa2VA release drives from a `[SEG]` embedding.
protocol NFKSa2VAGrounding: Module {
    /// The square side the grounding image is resized to (SAM 2: 1024, SAM 3: 1008).
    var imageSize: Int { get }
    func segment(image: MLXArray, languageEmbedding: MLXArray) -> (highResolution: MLXArray, lowResolution: MLXArray)
    /// The image's feature levels, computed once for every object a training step segments.
    func imageLevels(_ image: MLXArray) -> [MLXArray]
    func segment(levels: [MLXArray], languageEmbedding: MLXArray) -> (highResolution: MLXArray, lowResolution: MLXArray)
    /// The mask decoder, the part of the grounding encoder a fine-tune trains.
    var trainedMaskDecoder: Module { get }
}

extension NFKSa2VAGroundingEncoder: NFKSa2VAGrounding {
    var imageSize: Int { configuration.imageSize }
    var trainedMaskDecoder: Module { tracker.maskDecoder }
}

/// SAM 3's tracker as Sa2VA grounds with it, for one conditioning frame.
final class NFKSa2VASAM3GroundingEncoder: Module, NFKSa2VAGrounding {
    @ModuleInfo(key: "backbone") var backbone: NFKMLXSAM3BackboneNet
    @ModuleInfo(key: "neck") var neck: NFKSAM3Neck
    @ModuleInfo(key: "sam_prompt_encoder") var promptEncoder: NFKSAMPromptEncoder
    @ModuleInfo(key: "sam_mask_decoder") var maskDecoder: NFKMLXSAM2Decoder
    @ParameterInfo(key: "no_mem_embed") var noMemoryEmbedding: MLXArray

    let configuration: NFKMLXSAM3Configuration
    var imageSize: Int { configuration.imageSize }

    init(_ configuration: NFKMLXSAM3Configuration = .base) {
        self.configuration = configuration
        _backbone.wrappedValue = NFKMLXSAM3BackboneNet(configuration)
        _neck.wrappedValue = NFKSAM3Neck(configuration)
        _promptEncoder.wrappedValue = NFKSAMPromptEncoder(NFKMLXSAMConfiguration.vitB)
        _maskDecoder.wrappedValue = NFKMLXSAM2Decoder()
        _noMemoryEmbedding.wrappedValue = MLXArray.zeros([1, 1, configuration.fpnHiddenSize])
        super.init()
    }

    /// `image` is the normalized `[1, 1008, 1008, 3]` grounding image. Returns the best-of-three mask
    /// logits at the image resolution and at the decoder's `[1, 288, 288]`.
    var trainedMaskDecoder: Module { maskDecoder }

    func segment(image: MLXArray, languageEmbedding: MLXArray) -> (highResolution: MLXArray, lowResolution: MLXArray) {
        segment(levels: imageLevels(image), languageEmbedding: languageEmbedding)
    }

    /// The neck's first three levels, 288, 144, and 72 across.
    func imageLevels(_ image: MLXArray) -> [MLXArray] { Array(neck(backbone(image)).prefix(3)) }

    func segment(levels: [MLXArray], languageEmbedding: MLXArray) -> (highResolution: MLXArray, lowResolution: MLXArray) {
        let grid = configuration.grid
        let hidden = configuration.fpnHiddenSize
        let conditioned = levels[2] + noMemoryEmbedding.reshaped([1, 1, 1, hidden])
        let emptyPoint = promptEncoder.sparse(points: [(x: Float(0), y: Float(0), label: -1)])
        let sparse = concatenated([emptyPoint, languageEmbedding], axis: 1)
        let decoded = maskDecoder(features: conditioned, positional: promptEncoder.positionEncoding.grid(grid, grid),
                                  sparse: sparse, dense: promptEncoder.dense(grid: grid),
                                  highResolution: [levels[0], levels[1]])
        let best = argMax(decoded.iou[0..., 1...], axis: -1).item(Int.self)
        let low = decoded.masks[0..., 1 + best]
        let high = NFKMLXResample.resizeBilinear(low.expandedDimensions(axis: 3), height: imageSize, width: imageSize)
        return (high.squeezed(axis: 3), low)
    }

    /// The module name of a checkpoint tensor under `grounding_encoder.sam2_model.`, or nil for one this
    /// path does not read (the detector's own `convs`, the memory path).
    static func moduleKey(_ key: String) -> String? {
        let backbonePrefix = "backbone.vision_backbone."
        if key.hasPrefix(backbonePrefix + "trunk.") {
            var name = String(key.dropFirst((backbonePrefix + "trunk.").count))
            let renames = [("patch_embed.proj.", "embeddings.patch_embeddings.projection."),
                           ("pos_embed", "embeddings.position_embeddings"), ("ln_pre.", "layer_norm."),
                           ("blocks.", "layers."), (".norm1.", ".layer_norm1."), (".norm2.", ".layer_norm2."),
                           (".attn.proj.", ".attention.o_proj.")]
            for (from, to) in renames { name = name.replacingOccurrences(of: from, with: to) }
            return "backbone." + name
        }
        if key.hasPrefix(backbonePrefix + "sam2_convs.") {
            var name = String(key.dropFirst((backbonePrefix + "sam2_convs.").count))
            let renames = [("dconv_2x2_0.", "scale_layers.0."), ("dconv_2x2_1.", "scale_layers.2."),
                           ("dconv_2x2.", "scale_layers.0."), ("conv_1x1.", "proj1."), ("conv_3x3.", "proj2.")]
            for (from, to) in renames { name = name.replacingOccurrences(of: from, with: to) }
            return "neck.fpn_layers." + name
        }
        if key == "no_mem_embed" { return key }
        guard key.hasPrefix("sam_prompt_encoder.") || key.hasPrefix("sam_mask_decoder.") else { return nil }
        // SAM 3's two-way transformer names its MLP `lin1`/`lin2`, where SAM 2's checkpoint has `layers.0`/`.1`.
        let named = key.replacingOccurrences(of: ".mlp.lin1.", with: ".mlp.layers.0.")
            .replacingOccurrences(of: ".mlp.lin2.", with: ".mlp.layers.1.")
        return NFKMLXSAM2.remapTrackerKey(named)
    }

    /// Loads the grounding subtree of a release's tensors (`grounding_encoder.sam2_model.*`).
    func load(_ arrays: [(String, MLXArray)], dtype: DType = .float32) throws {
        try NFKMLXWeights.apply(Self.mapped(arrays).map { ($0.0, $0.1.asType(dtype)) }, to: self)
    }

    /// The grounding subtree's tensors under module names and in MLX layouts: the fused `qkv` split into
    /// three projections, the class-token position row dropped, and the convolutions transposed.
    static func mapped(_ arrays: [(String, MLXArray)]) -> [(String, MLXArray)] {
        let prefix = "grounding_encoder.sam2_model."
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays where key.hasPrefix(prefix) {
            let inner = String(key.dropFirst(prefix.count))
            guard let name = Self.moduleKey(inner) else { continue }
            if let fused = name.range(of: ".attn.qkv.") {
                let layer = String(name[..<fused.lowerBound]) + ".attention."
                let suffix = String(name[fused.upperBound...])
                let parts = split(value, parts: 3, axis: 0)
                for (index, projection) in ["q_proj", "k_proj", "v_proj"].enumerated() {
                    mapped.append((layer + projection + "." + suffix, parts[index]))
                }
            } else if name == "backbone.embeddings.position_embeddings" {
                mapped.append((name, value[0..., 1...]))                              // the class-token row dropped
            } else if value.ndim == 4 {
                if name.contains("scale_layers") {
                    mapped.append((name, value.transposed(1, 2, 3, 0)))
                } else if name.hasPrefix("sam_") {
                    mapped.append((name, NFKMLXSa2VANet.groundingTransposed(name, value)))
                } else {
                    mapped.append((name, value.transposed(0, 2, 3, 1)))
                }
            } else {
                mapped.append((name, value))
            }
        }
        return mapped
    }
}
