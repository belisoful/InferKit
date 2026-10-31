//
//  NFKMLXWanAnimateImageEncoder.swift
//  InferKitMLX
//
//  The Wan 2.2 Animate 2 image encoder: transformers' `CLIPVisionModel` (the visual tower of
//  open-clip's XLM-RoBERTa ViT-H/14), read at its second-to-last hidden state, which is how the
//  pipeline conditions every denoising pass on the reference image and the extract pass on the driving
//  video's first frame.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The image encoder's geometry, from a `CLIPVisionConfig`. Defaults are the released ViT-H/14.
public struct NFKMLXWanAnimateImageEncoderConfiguration: Sendable {
    public var hiddenSize: Int
    public var layers: Int
    public var heads: Int
    public var intermediateSize: Int
    public var imageSize: Int
    public var patchSize: Int
    public var layerNormEps: Float
    /// Whether the MLP's activation is CLIP's `quick_gelu` rather than the exact `gelu` open-clip's
    /// ViT-H uses.
    public var quickGELU: Bool

    public init(hiddenSize: Int = 1280, layers: Int = 32, heads: Int = 16, intermediateSize: Int = 5120,
                imageSize: Int = 224, patchSize: Int = 14, layerNormEps: Float = 1e-5, quickGELU: Bool = false) {
        self.hiddenSize = hiddenSize
        self.layers = layers
        self.heads = heads
        self.intermediateSize = intermediateSize
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.layerNormEps = layerNormEps
        self.quickGELU = quickGELU
    }

    /// The geometry an `image_encoder/config.json` states.
    static func configuration(fromHuggingFace url: URL) throws -> NFKMLXWanAnimateImageEncoderConfiguration {
        var json = try NFKMLXWanRelease.json(url)
        if let vision = json["vision_config"] as? [String: Any] {
            json = vision
        }
        let activation = json["hidden_act"] as? String ?? "gelu"
        guard activation == "gelu" || activation == "quick_gelu" else {
            throw NFKMLXError.unsupportedConfiguration("the image encoder's activation is \(activation)")
        }
        let base = NFKMLXWanAnimateImageEncoderConfiguration()
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        return NFKMLXWanAnimateImageEncoderConfiguration(
            hiddenSize: int("hidden_size", base.hiddenSize), layers: int("num_hidden_layers", base.layers),
            heads: int("num_attention_heads", base.heads), intermediateSize: int("intermediate_size", base.intermediateSize),
            imageSize: int("image_size", base.imageSize), patchSize: int("patch_size", base.patchSize),
            layerNormEps: (json["layer_norm_eps"] as? NSNumber)?.floatValue ?? base.layerNormEps,
            quickGELU: activation == "quick_gelu")
    }

    var grid: Int { imageSize / patchSize }
}

/// One encoder layer: pre-norm attention with biased projections, then a pre-norm MLP.
final class NFKWanAnimateImageEncoderLayer: Module {
    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var q: Linear
        @ModuleInfo(key: "k_proj") var k: Linear
        @ModuleInfo(key: "v_proj") var v: Linear
        @ModuleInfo(key: "out_proj") var out: Linear
        init(_ width: Int) {
            _q.wrappedValue = Linear(width, width)
            _k.wrappedValue = Linear(width, width)
            _v.wrappedValue = Linear(width, width)
            _out.wrappedValue = Linear(width, width)
        }
    }
    final class MLP: Module {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear
        let activation: NFKCLIPActivation
        init(_ width: Int, _ inner: Int, activation: NFKCLIPActivation) {
            self.activation = activation
            _fc1.wrappedValue = Linear(width, inner)
            _fc2.wrappedValue = Linear(inner, width)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(activation(fc1(x))) }
    }
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP
    let heads: Int

    init(_ c: NFKMLXWanAnimateImageEncoderConfiguration) {
        heads = c.heads
        _norm1.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _attention.wrappedValue = Attention(c.hiddenSize)
        _norm2.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _mlp.wrappedValue = MLP(c.hiddenSize, c.intermediateSize, activation: c.quickGELU ? .quickGELU : .gelu)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let n = norm1(x)
        let headDim = x.dim(2) / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([t.dim(0), t.dim(1), heads, headDim]).transposed(0, 2, 1, 3) }
        let attended = NFKReferenceRounding.attention(
            queries: split(attention.q(n)), keys: split(attention.k(n)), values: split(attention.v(n)),
            scale: 1 / sqrt(Float(headDim)), mask: nil)
        let h = x + attention.out(attended.transposed(0, 2, 1, 3).reshaped(x.shape))
        return h + mlp(norm2(h))
    }
}

/// The vision tower under `vision_model.`, as transformers' `CLIPVisionTransformer` names it.
final class NFKWanAnimateVisionTransformer: Module {
    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        init(_ c: NFKMLXWanAnimateImageEncoderConfiguration) {
            _patchEmbedding.wrappedValue = NFKConv2d(inputChannels: 3, outputChannels: c.hiddenSize,
                                                     kernelSize: IntOrPair(c.patchSize), stride: IntOrPair(c.patchSize),
                                                     bias: false)
            _classEmbedding.wrappedValue = MLXArray.zeros([c.hiddenSize])
            _positionEmbedding.wrappedValue = Embedding(embeddingCount: c.grid * c.grid + 1, dimensions: c.hiddenSize)
        }
    }
    final class Encoder: Module {
        @ModuleInfo(key: "layers") var layers: [NFKWanAnimateImageEncoderLayer]
        init(_ c: NFKMLXWanAnimateImageEncoderConfiguration) {
            _layers.wrappedValue = (0 ..< c.layers).map { _ in NFKWanAnimateImageEncoderLayer(c) }
        }
    }
    @ModuleInfo(key: "embeddings") var embeddings: Embeddings
    @ModuleInfo(key: "pre_layrnorm") var preNorm: LayerNorm
    @ModuleInfo(key: "encoder") var encoder: Encoder
    @ModuleInfo(key: "post_layernorm") var postNorm: LayerNorm

    init(_ c: NFKMLXWanAnimateImageEncoderConfiguration) {
        _embeddings.wrappedValue = Embeddings(c)
        _preNorm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _encoder.wrappedValue = Encoder(c)
        _postNorm.wrappedValue = NFKLayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
    }
}

/// The Wan Animate image encoder.
public final class NFKMLXWanAnimateImageEncoderNet: Module {
    @ModuleInfo(key: "vision_model") var visionModel: NFKWanAnimateVisionTransformer

    public let configuration: NFKMLXWanAnimateImageEncoderConfiguration

    /// CLIP's image normalization, which the pipeline applies after mapping −1…1 to 0…1.
    static let mean = MLXArray([Float(0.48145466), 0.4578275, 0.40821073])
    static let std = MLXArray([Float(0.26862954), 0.26130258, 0.27577711])

    public init(_ configuration: NFKMLXWanAnimateImageEncoderConfiguration) {
        self.configuration = configuration
        _visionModel.wrappedValue = NFKWanAnimateVisionTransformer(configuration)
    }

    /// The hidden state after `layerCount` encoder layers (transformers' `hidden_states[layerCount]`),
    /// `[B, 1 + patches, hidden]`, from normalized pixels `[B, imageSize, imageSize, 3]`.
    func hiddenState(_ pixels: MLXArray, afterLayers layerCount: Int) -> MLXArray {
        let tower = visionModel
        let patches = tower.embeddings.patchEmbedding(pixels)
        let batch = patches.dim(0), width = patches.dim(3)
        var x = patches.reshaped([batch, -1, width])
        let cls = broadcast(tower.embeddings.classEmbedding.reshaped([1, 1, width]), to: [batch, 1, width])
        x = concatenated([cls, x], axis: 1)
        x = x + tower.embeddings.positionEmbedding.weight[0 ..< x.dim(1)]
        x = tower.preNorm(x)
        for layer in tower.encoder.layers.prefix(layerCount) {
            x = layer(x)
        }
        return x
    }

    /// The features the transformer reads for an image `[H, W, 3]` in −1…1, `[1 + patches, hidden]`: the
    /// image resized to the encoder's side by torch's bicubic interpolation, mapped to 0…1, normalized,
    /// and read at the second-to-last hidden state.
    public func features(_ image: MLXArray) -> MLXArray {
        let side = configuration.imageSize
        let resized = NFKMLXPhi4MMImageProcessor.bicubicResized(image.asType(.float32), height: side, width: side)
        let normalized = ((resized * 0.5 + 0.5) - Self.mean) / Self.std
        let dtype = NFKReferenceRounding.parameterType(of: visionModel.embeddings.patchEmbedding)
        let state = hiddenState(normalized.asType(dtype).expandedDimensions(axis: 0),
                                afterLayers: configuration.layers - 1)[0]
        eval(state)
        return state
    }

    /// Loads an `image_encoder/` directory (one file or a shard index): the patch convolution moves from
    /// `[out, in, h, w]` to MLX's `[out, h, w, in]`, and every other tensor loads as stored. `precision`
    /// `.checkpoint` keeps the release's stored type.
    public static func loadWeights(into net: NFKMLXWanAnimateImageEncoderNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision,
                                                                 transform: { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        })
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
    }
}
