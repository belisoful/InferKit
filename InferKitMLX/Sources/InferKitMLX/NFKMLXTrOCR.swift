import Foundation
import MLX
import MLXNN

// MARK: - Configuration

/// The ViT image encoder geometry a TrOCR release declares in its `encoder` config block.
///
/// @discussion TrOCR pairs a ViT image encoder with a BART-style text decoder. The encoder patchifies
/// the image, prepends a class token, adds a learned position table, and runs pre-normalized
/// transformer blocks. The base and large releases use `google/vit` with no query/key/value bias; the
/// small releases use DeiT, which adds a distillation token after the class token and biases its
/// query, key, and value projections.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTrOCRVisionConfiguration: Sendable {
    public var imageSize: Int
    public var patchSize: Int
    public var hiddenSize: Int
    public var layers: Int
    public var heads: Int
    public var intermediateSize: Int
    public var layerNormEps: Float
    public var qkvBias: Bool
    /// A DeiT encoder's second prefix token (`embeddings.distillation_token`).
    public var distillationToken: Bool

    public init(imageSize: Int = 384, patchSize: Int = 16, hiddenSize: Int = 768, layers: Int = 12,
                heads: Int = 12, intermediateSize: Int = 3072, layerNormEps: Float = 1e-12,
                qkvBias: Bool = false, distillationToken: Bool = false) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.hiddenSize = hiddenSize
        self.layers = layers
        self.heads = heads
        self.intermediateSize = intermediateSize
        self.layerNormEps = layerNormEps
        self.qkvBias = qkvBias
        self.distillationToken = distillationToken
    }

    /// The base ViT encoder of `microsoft/trocr-base-handwritten`.
    public static let base = NFKMLXTrOCRVisionConfiguration()

    /// The large ViT encoder of `microsoft/trocr-large-*` (1024-wide, 24 layers, 16 heads).
    public static let large = NFKMLXTrOCRVisionConfiguration(hiddenSize: 1024, layers: 24, heads: 16,
                                                             intermediateSize: 4096)

    /// The small DeiT encoder of `microsoft/trocr-small-*` (384-wide, 12 layers, 6 heads).
    public static let small = NFKMLXTrOCRVisionConfiguration(hiddenSize: 384, layers: 12, heads: 6,
                                                             intermediateSize: 1536, qkvBias: true,
                                                             distillationToken: true)

    var patchesPerSide: Int { imageSize / patchSize }
    var patchCount: Int { patchesPerSide * patchesPerSide }
    var headDim: Int { hiddenSize / heads }
    var prefixTokens: Int { distillationToken ? 2 : 1 }

    /// Reads the ViT geometry from the `encoder` block of a vision-encoder-decoder `config.json`.
    public init(huggingFaceConfig json: [String: Any]) throws {
        let encoder = (json["encoder"] as? [String: Any]) ?? json
        func int(_ key: String, _ fallback: Int) -> Int { (encoder[key] as? NSNumber)?.intValue ?? fallback }
        self.init(imageSize: int("image_size", 384), patchSize: int("patch_size", 16),
                  hiddenSize: int("hidden_size", 768), layers: int("num_hidden_layers", 12),
                  heads: int("num_attention_heads", 12), intermediateSize: int("intermediate_size", 3072),
                  layerNormEps: Float((encoder["layer_norm_eps"] as? NSNumber)?.doubleValue ?? 1e-12),
                  qkvBias: (encoder["qkv_bias"] as? Bool) ?? false,
                  distillationToken: (encoder["model_type"] as? String) == "deit")
    }
}

// MARK: - Patch and position embeddings

/// The patch convolution (`embeddings.patch_embeddings.projection`), a stride-`patch` convolution
/// that flattens each non-overlapping window to one token.
final class NFKTrOCRPatchEmbeddings: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(_ c: NFKMLXTrOCRVisionConfiguration) {
        _projection.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.hiddenSize,
                                          kernelSize: IntOrPair(c.patchSize), stride: IntOrPair(c.patchSize))
        super.init()
    }

    /// - Parameter image: `[B, H, W, 3]`.
    /// - Returns: `[B, patches, hidden]`.
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        let patches = projection(image)                                                // [B, gh, gw, hidden]
        return patches.reshaped([patches.dim(0), patches.dim(1) * patches.dim(2), patches.dim(3)])
    }
}

/// Prepends the class token (and a DeiT encoder's distillation token after it) and adds the learned
/// position table (`embeddings`).
final class NFKTrOCRVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: NFKTrOCRPatchEmbeddings
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "distillation_token") var distillationToken: MLXArray?
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

    init(_ c: NFKMLXTrOCRVisionConfiguration) {
        _patchEmbeddings.wrappedValue = NFKTrOCRPatchEmbeddings(c)
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, c.hiddenSize])
        _distillationToken.wrappedValue = c.distillationToken ? MLXArray.zeros([1, 1, c.hiddenSize]) : nil
        _positionEmbeddings.wrappedValue = MLXArray.zeros([1, c.patchCount + c.prefixTokens, c.hiddenSize])
        super.init()
    }

    func callAsFunction(_ image: MLXArray) -> MLXArray {
        let patches = patchEmbeddings(image)                                           // [B, patches, hidden]
        let prefix = [clsToken, distillationToken].compactMap { $0 }.map {
            broadcast($0, to: [patches.dim(0), 1, patches.dim(2)])
        }
        return concatenated(prefix + [patches], axis: 1) + positionEmbeddings
    }
}

// MARK: - Transformer block

/// A `dense` linear, the shape both `intermediate` and `output` carry in a ViT block.
final class NFKTrOCRDense: Module {
    @ModuleInfo(key: "dense") var dense: Linear

    init(_ inFeatures: Int, _ outFeatures: Int) {
        _dense.wrappedValue = Linear(inFeatures, outFeatures)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { dense(x) }
}

/// The multi-head self-attention (`attention.attention`), scaled by the head dimension.
final class NFKTrOCRSelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    let heads: Int
    let headDim: Int

    init(_ c: NFKMLXTrOCRVisionConfiguration) {
        heads = c.heads
        headDim = c.headDim
        _query.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.qkvBias)
        _key.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.qkvBias)
        _value.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.qkvBias)
        super.init()
    }

    private func split(_ t: MLXArray) -> MLXArray {
        t.reshaped([t.dim(0), t.dim(1), heads, headDim]).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(query(x)), keys: split(key(x)), values: split(value(x)),
            scale: 1 / sqrt(Float(headDim)), mask: nil)
        return attended.transposed(0, 2, 1, 3).reshaped([x.dim(0), x.dim(1), heads * headDim])
    }
}

/// One pre-normalized ViT block: attention with a residual, then the MLP with a residual.
final class NFKTrOCRLayer: Module {
    @ModuleInfo(key: "attention") var attention: NFKTrOCRAttention
    @ModuleInfo(key: "intermediate") var intermediate: NFKTrOCRDense
    @ModuleInfo(key: "output") var output: NFKTrOCRDense
    @ModuleInfo(key: "layernorm_before") var layerNormBefore: LayerNorm
    @ModuleInfo(key: "layernorm_after") var layerNormAfter: LayerNorm

    init(_ c: NFKMLXTrOCRVisionConfiguration) {
        _attention.wrappedValue = NFKTrOCRAttention(c)
        _intermediate.wrappedValue = NFKTrOCRDense(c.hiddenSize, c.intermediateSize)
        _output.wrappedValue = NFKTrOCRDense(c.intermediateSize, c.hiddenSize)
        _layerNormBefore.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _layerNormAfter.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(layerNormBefore(x))
        return h + output(gelu(intermediate(layerNormAfter(h))))
    }
}

/// The attention sub-block: the multi-head attention (`attention`) then its output projection (`output`).
final class NFKTrOCRAttention: Module {
    @ModuleInfo(key: "attention") var attention: NFKTrOCRSelfAttention
    @ModuleInfo(key: "output") var output: NFKTrOCRDense

    init(_ c: NFKMLXTrOCRVisionConfiguration) {
        _attention.wrappedValue = NFKTrOCRSelfAttention(c)
        _output.wrappedValue = NFKTrOCRDense(c.hiddenSize, c.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { output(attention(x)) }
}

// MARK: - Encoder network

/// The TrOCR image encoder: patch and position embeddings, the transformer stack, and a final
/// layer norm. Its output is the memory the text decoder cross-attends.
///
/// @discussion Module keys mirror a transformers `ViTModel` with the vision-encoder-decoder
/// `encoder.` prefix removed: `embeddings.patch_embeddings.projection`, `embeddings.cls_token`,
/// `embeddings.position_embeddings`, `encoder.layer.N.attention.attention.{query,key,value}`,
/// `encoder.layer.N.attention.output.dense`, `encoder.layer.N.{intermediate,output}.dense`,
/// `encoder.layer.N.{layernorm_before,layernorm_after}`, and the final `layernorm`. The `pooler` a
/// `ViTModel` carries is unused and dropped on load.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXTrOCRVisionNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKTrOCRVisionEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKTrOCREncoder
    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm

    public let configuration: NFKMLXTrOCRVisionConfiguration

    public init(_ c: NFKMLXTrOCRVisionConfiguration) {
        configuration = c
        _embeddings.wrappedValue = NFKTrOCRVisionEmbeddings(c)
        _encoder.wrappedValue = NFKTrOCREncoder(c)
        _layerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        super.init()
    }

    /// - Parameter image: `[B, H, W, 3]`, already normalized.
    /// - Returns: the encoder output `[B, patches + 1, hidden]`.
    public func callAsFunction(_ image: MLXArray) -> MLXArray {
        var h = embeddings(image)
        for layer in encoder.layer { h = layer(h) }
        return layerNorm(h)
    }
}

/// The transformer stack (`encoder.layer`).
final class NFKTrOCREncoder: Module {
    @ModuleInfo(key: "layer") var layer: [NFKTrOCRLayer]

    init(_ c: NFKMLXTrOCRVisionConfiguration) {
        _layer.wrappedValue = (0 ..< c.layers).map { _ in NFKTrOCRLayer(c) }
        super.init()
    }
}

// MARK: - Weight remap

enum NFKMLXTrOCRVisionWeights {
    /// Maps a vision-encoder-decoder checkpoint key to this encoder's module key, or nil to skip the
    /// tensor. The encoder tensors carry an `encoder.` prefix; the unused `pooler` is dropped.
    static func visionKey(_ key: String) -> String? {
        guard key.hasPrefix("encoder.") else { return nil }
        let stripped = String(key.dropFirst("encoder.".count))
        if stripped.hasPrefix("pooler.") { return nil }
        return stripped
    }
}
