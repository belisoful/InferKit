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
//  At half precision the grounding computes as the release does on CUDA (`NFKSAM3Autocast`).
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

    /// Whether the weights are half precision, where the grounding computes as `NFKSAM3Autocast` does.
    var autocasts: Bool { NFKReferenceRounding.isReduced(noMemoryEmbedding) }

    /// The neck's first three levels, 288, 144, and 72 across.
    func imageLevels(_ image: MLXArray) -> [MLXArray] {
        guard autocasts else { return Array(neck(backbone(image)).prefix(3)) }
        let hidden = NFKSAM3Autocast.backbone(backbone, image)
        return neck.fpnLayers.prefix(3).map { NFKSAM3Autocast.level($0, hidden) }
    }

    func segment(levels: [MLXArray], languageEmbedding: MLXArray) -> (highResolution: MLXArray, lowResolution: MLXArray) {
        let grid = configuration.grid
        let hidden = configuration.fpnHiddenSize
        let conditioned = levels[2] + noMemoryEmbedding.reshaped([1, 1, 1, hidden])
        let emptyPoint = promptEncoder.sparse(points: [(x: Float(0), y: Float(0), label: -1)])
        let positional = promptEncoder.positionEncoding.grid(grid, grid)
        let decoded: (masks: MLXArray, iou: MLXArray, objectScore: MLXArray)
        if autocasts {
            // The reference's sparse prompt starts as an empty float32 tensor, so the point tokens and the
            // `[SEG]` embedding concatenate onto it in float32.
            let sparse = concatenated([emptyPoint.asType(.float32), languageEmbedding.asType(.float32)], axis: 1)
            decoded = NFKSAM3Autocast.decode(maskDecoder, features: conditioned, positional: positional,
                                             sparse: sparse, dense: promptEncoder.dense(grid: grid),
                                             highResolution: [levels[0], levels[1]])
        } else {
            let sparse = concatenated([emptyPoint, languageEmbedding.asType(emptyPoint.dtype)], axis: 1)
            let full = maskDecoder(features: conditioned, positional: positional, sparse: sparse,
                                   dense: promptEncoder.dense(grid: grid), highResolution: [levels[0], levels[1]])
            decoded = (full.masks, full.iou, full.objectScore)
        }
        let best = argMax(decoded.iou[0..., 1...], axis: -1).item(Int.self)
        let low = Self.present(decoded.masks[0..., 1 + best], objectScore: decoded.objectScore).asType(.float32)
        let high = NFKMLXResample.resizeBilinear(low.expandedDimensions(axis: 3), height: imageSize, width: imageSize)
        return (high.squeezed(axis: 3), low)
    }

    /// `masks` where the decoder's object score is positive, else -1024 everywhere: the release's
    /// `_forward_sam_heads` replaces the masks of a frame it finds no object in before choosing one, so
    /// `predict_forward` returns an empty mask. Sa2VA's SAM 2 head comments that replacement out.
    static func present(_ masks: MLXArray, objectScore: MLXArray) -> MLXArray {
        which(objectScore.reshaped([-1] + Array(repeating: 1, count: masks.ndim - 1)) .> 0, masks,
              MLXArray(Float(-1024)).asType(masks.dtype))
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

/// SAM 3's grounding at half precision as Sa2VA's release computes it on CUDA: the release's tracker
/// enters `torch.autocast("cuda", bfloat16)` for the whole process, over bfloat16 weights.
///
/// Under autocast a projection or convolution casts its input to the weights' type and rounds once, and a
/// layer norm runs in float32 and returns float32. The ViT's residual stream therefore stays float32 from
/// `ln_pre` on, and so do the two-way transformer's tokens and, after its first layer, its image keys. Every
/// other operation keeps its inputs' type, so a sum takes the wider of its two terms. The ViT MLP's first
/// projection is cuBLASLt's fused bias and tanh GELU on the float32 accumulator, rounded once. Attention
/// follows torch's CPU flash kernel, the stand-in the reference is recorded on.
enum NFKSAM3Autocast {

    /// A projection on `x` cast to the weights' type.
    static func linear(_ layer: Linear, _ x: MLXArray) -> MLXArray { layer(x.asType(layer.weight.dtype)) }

    /// A convolution on `x` cast to the weights' type, accumulated in float32 with its bias and rounded once.
    static func conv(_ layer: Conv2d, _ x: MLXArray) -> MLXArray {
        let type = layer.weight.dtype
        var y = conv2d(x.asType(type).asType(.float32), layer.weight.asType(.float32), stride: .init(layer.stride),
                       padding: .init(layer.padding), dilation: .init(layer.dilation), groups: layer.groups)
        if let bias = layer.bias { y = y + bias.asType(.float32) }
        return y.asType(type)
    }

    /// A transposed convolution on `x` cast to the weights' type. torch rounds its bias apart, as MLX does.
    static func convTransposed(_ layer: ConvTransposed2d, _ x: MLXArray) -> MLXArray {
        layer(x.asType(layer.weight.dtype))
    }

    /// A layer norm in float32, returned in float32.
    static func layerNorm(_ norm: LayerNorm, _ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x.asType(.float32), weight: norm.weight?.asType(.float32), bias: norm.bias?.asType(.float32),
                          eps: norm.eps)
    }

    /// The detectron2 `LayerNorm2d` over channels-last `x`: the mean and the centering in `x`'s type,
    /// then the square (`pow`, float32 under autocast), the variance, and everything after it in float32.
    static func layerNorm2d(_ norm: LayerNorm, _ x: MLXArray) -> MLXArray {
        let centered = (x - NFKReferenceRounding.mean(x, axis: -1)).asType(.float32)
        let normalized = centered / sqrt((centered * centered).mean(axis: -1, keepDims: true) + norm.eps)
        guard let weight = norm.weight, let bias = norm.bias else { return normalized }
        return weight.asType(.float32) * normalized + bias.asType(.float32)
    }

    /// The exact GELU in `x`'s type, rounded once.
    static func gelu(_ x: MLXArray) -> MLXArray { NFKReferenceRounding.wide(x) { MLXNN.gelu($0) } }

    /// The ViT on the float32 image → its float32 feature map `[1, H/patch, W/patch, hidden]`.
    static func backbone(_ net: NFKMLXSAM3BackboneNet, _ image: MLXArray) -> MLXArray {
        let patches = conv(net.embeddings.patchEmbeddings.projection, image)
        let embedded = patches + net.embeddings.tiled(height: patches.dim(1), width: patches.dim(2))
        var hidden = layerNorm(net.layerNorm, embedded)
        for layer in net.layers {
            hidden = Self.layer(layer, hidden)
        }
        return hidden
    }

    /// One ViT layer on its float32 residual stream.
    static func layer(_ layer: NFKSAM3Layer, _ x: MLXArray) -> MLXArray {
        let normalized = layerNorm(layer.norm1, x)
        let attended: MLXArray
        if layer.windowSize > 0 {
            let (windows, padded) = NFKSAM3Windows.partition(normalized, size: layer.windowSize)
            attended = NFKSAM3Windows.join(attention(layer.attention, windows, rotary: layer.rotary),
                                           size: layer.windowSize, padded: padded, original: (x.dim(1), x.dim(2)))
        } else {
            attended = attention(layer.attention, normalized, rotary: layer.rotary)
        }
        let residual = x + attended
        return residual + mlp(layer.mlp, layerNorm(layer.norm2, residual))
    }

    /// The ViT's rotary self-attention. The rotation runs in float32 on the projected queries and keys and
    /// rounds back to their type.
    static func attention(_ attention: NFKSAM3Attention, _ x: MLXArray, rotary: NFKSAM3Rotary) -> MLXArray {
        let batch = x.dim(0), height = x.dim(1), width = x.dim(2)
        func split(_ projection: Linear) -> MLXArray {
            linear(projection, x).reshaped([batch, height * width, attention.heads, attention.headDim])
                .transposed(0, 2, 1, 3)
        }
        let queries = split(attention.qProj), keys = split(attention.kProj)
        let attended = NFKReferenceRounding.flashAttention(
            queries: rotary(queries.asType(.float32)).asType(queries.dtype),
            keys: rotary(keys.asType(.float32)).asType(keys.dtype), values: split(attention.vProj),
            scale: 1 / sqrt(Float(attention.headDim)), mask: nil)
        return linear(attention.oProj, attended.transposed(0, 2, 1, 3)
            .reshaped([batch, height, width, attention.heads * attention.headDim]))
    }

    /// The ViT MLP: ``widened(_:_:)``, then the second projection.
    static func mlp(_ mlp: NFKSAM3MLP, _ x: MLXArray) -> MLXArray { linear(mlp.fc2, widened(mlp, x)) }

    /// The ViT MLP's first projection, bias, and tanh GELU on the float32 accumulator, rounded once.
    static func widened(_ mlp: NFKSAM3MLP, _ x: MLXArray) -> MLXArray {
        let weight = mlp.fc1.weight
        var accumulated = matmul(x.asType(weight.dtype).asType(.float32), weight.asType(.float32).T)
        if let bias = mlp.fc1.bias { accumulated = accumulated + bias.asType(.float32) }
        return geluApproximate(accumulated).asType(weight.dtype)
    }

    /// One tracker neck level on the float32 feature map → the level in the weights' type.
    static func level(_ level: NFKSAM3FPNLayer, _ x: MLXArray) -> MLXArray {
        var out = x
        switch level.scaleFactor {
        case 4:
            out = gelu(convTransposed(level.scaleLayers[0] as! ConvTransposed2d, out))
            out = convTransposed(level.scaleLayers[2] as! ConvTransposed2d, out)
        case 2:
            out = convTransposed(level.scaleLayers[0] as! ConvTransposed2d, out)
        case 0.5:
            out = NFKMLXResample.maxPooled(out, kernel: 2, stride: 2)
        default:
            break
        }
        return conv(level.proj2, conv(level.proj1, out))
    }

    /// The two-way attention: projections in the weights' type, then the flash kernel.
    static func attention(_ attention: NFKSAMAttention, _ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let batch = q.dim(0)
        let inner = attention.qProj.weight.dim(0)
        let headDim = inner / attention.heads
        func split(_ x: MLXArray, _ projection: Linear) -> MLXArray {
            linear(projection, x).reshaped([batch, x.dim(1), attention.heads, headDim]).transposed(0, 2, 1, 3)
        }
        let attended = NFKReferenceRounding.flashAttention(
            queries: split(q, attention.qProj), keys: split(k, attention.kProj), values: split(v, attention.vProj),
            scale: 1 / sqrt(Float(headDim)), mask: nil)
        return linear(attention.outProj, attended.transposed(0, 2, 1, 3).reshaped([batch, q.dim(1), inner]))
    }

    /// One two-way block; the queries leave each norm in float32.
    static func twoWay(_ block: NFKSAMTwoWayBlock, _ queries: MLXArray, _ keys: MLXArray, queryPE: MLXArray,
                       keyPE: MLXArray) -> (MLXArray, MLXArray) {
        var out: MLXArray
        if block.skipFirstLayerPE {
            out = attention(block.selfAttn, queries, queries, queries)
        } else {
            let q = queries + queryPE
            out = queries + attention(block.selfAttn, q, q, queries)
        }
        out = layerNorm(block.norm1, out)
        var q = out + queryPE
        var k = keys + keyPE
        out = layerNorm(block.norm2, out + attention(block.crossTokenImage, q, k, keys))
        out = layerNorm(block.norm3, out + perceptron(block.mlp, out))
        q = out + queryPE
        k = keys + keyPE
        return (out, layerNorm(block.norm4, keys + attention(block.crossImageToken, k, q, out)))
    }

    /// A ReLU perceptron in the weights' type.
    static func perceptron(_ mlp: NFKSAMMLP, _ x: MLXArray) -> MLXArray {
        var y = x
        for (index, layer) in mlp.layers.enumerated() {
            y = linear(layer, y)
            if index < mlp.layers.count - 1 { y = relu(y) }
        }
        return y
    }

    /// The mask decoder as ``NFKMLXSAM2Decoder`` computes it, under autocast. `sparse` is float32,
    /// `features`, `positional`, `dense`, and the two finer levels in the weights' type.
    static func decode(_ decoder: NFKMLXSAM2Decoder, features: MLXArray, positional: MLXArray, sparse: MLXArray,
                       dense: MLXArray, highResolution: [MLXArray]) -> (masks: MLXArray, iou: MLXArray, objectScore: MLXArray) {
        let output = concatenated([decoder.objScoreToken, decoder.iouToken, decoder.maskTokens], axis: 0)
        var tokens = concatenated([output.reshaped([1, output.dim(0), output.dim(1)]).asType(sparse.dtype), sparse],
                                  axis: 1)
        let (height, width, channels) = (features.dim(1), features.dim(2), features.dim(3))
        var image = (features + dense).reshaped([1, height * width, channels])
        let queryPE = tokens
        for layer in decoder.layers {
            (tokens, image) = twoWay(layer, tokens, image, queryPE: queryPE, keyPE: positional)
        }
        let attended = attention(decoder.finalAttention, tokens + queryPE, image + positional, image)
        tokens = layerNorm(decoder.normFinalAttention, tokens + attended)
        let maskOut = tokens[0..., 2 ..< (2 + decoder.maskCount)]

        var upscaled = convTransposed(decoder.upscale1, image.reshaped([1, height, width, channels]))
            + conv(decoder.convS1, highResolution[1])
        upscaled = gelu(layerNorm2d(decoder.upscaleNorm, upscaled))
        upscaled = gelu(convTransposed(decoder.upscale2, upscaled) + conv(decoder.convS0, highResolution[0]))
        let hyperIn = concatenated((0 ..< decoder.maskCount).map { index in
            perceptron(decoder.hyper[index], maskOut[0..., index]).reshaped([1, 1, -1])
        }, axis: 1)
        let (uh, uw, uc) = (upscaled.dim(1), upscaled.dim(2), upscaled.dim(3))
        let masks = matmul(hyperIn, upscaled.reshaped([1, uh * uw, uc]).transposed(0, 2, 1))
            .reshaped([1, decoder.maskCount, uh, uw])
        let iou = perceptron(decoder.iouHead, tokens[0..., 1])
        return (masks, decoder.iouUsesSigmoid ? NFKReferenceRounding.sigmoid(iou) : iou,
                perceptron(decoder.objScoreHead, tokens[0..., 0]))
    }
}
