//
//  NFKMLXT5Encoder.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The T5 v1.1 text encoder (`T5EncoderModel`), the text conditioning for several diffusion models (LTX-Video
// here; PixArt / SD3 / Flux / Wan use the same family). A stack of pre-normalized transformer blocks with
// two differences from a standard transformer: the attention is UNSCALED and adds a bucketed
// relative-position bias (computed once, shared across layers), and the norm is T5LayerNorm — an RMS norm
// with a weight and no mean subtraction. The feed-forward is gated (`wo(gelu(wi_0(x)) · wi_1(x))`).

/// T5 encoder geometry. Defaults are T5-XXL v1.1 (the LTX text encoder).
public struct NFKMLXT5Configuration: Sendable {
    public var dModel: Int
    public var layers: Int
    public var heads: Int
    public var keyDim: Int
    public var ffDim: Int
    public var vocabularySize: Int
    public var relativeBuckets: Int
    public var relativeMaxDistance: Int
    public var layerNormEps: Float
    /// umT5 gives EVERY layer its own relative-position bias; plain T5 shares block 0's across the stack.
    public var perLayerBias: Bool

    public init(dModel: Int = 4096, layers: Int = 24, heads: Int = 64, keyDim: Int = 64, ffDim: Int = 10240,
                vocabularySize: Int = 32128, relativeBuckets: Int = 32, relativeMaxDistance: Int = 128,
                layerNormEps: Float = 1e-6, perLayerBias: Bool = false) {
        self.dModel = dModel
        self.layers = layers
        self.heads = heads
        self.keyDim = keyDim
        self.ffDim = ffDim
        self.vocabularySize = vocabularySize
        self.relativeBuckets = relativeBuckets
        self.relativeMaxDistance = relativeMaxDistance
        self.layerNormEps = layerNormEps
        self.perLayerBias = perLayerBias
    }

    public static let xxl = NFKMLXT5Configuration()

    /// umT5-XXL, Wan's text encoder: T5-XXL geometry with a per-layer relative-position bias.
    public static let umt5XXL = NFKMLXT5Configuration(vocabularySize: 256384, perLayerBias: true)

    public static let tiny = NFKMLXT5Configuration(dModel: 32, layers: 2, heads: 2, keyDim: 16, ffDim: 64,
                                                   vocabularySize: 128, relativeBuckets: 16, relativeMaxDistance: 32)

    public static let tinyUMT5 = NFKMLXT5Configuration(dModel: 32, layers: 3, heads: 2, keyDim: 16, ffDim: 64,
                                                       vocabularySize: 128, relativeBuckets: 16,
                                                       relativeMaxDistance: 32, perLayerBias: true)

    var inner: Int { heads * keyDim }
}

/// T5LayerNorm: an RMS norm with a learned scale and no mean subtraction.
final class NFKT5LayerNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(_ dim: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        weight * (x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps))
    }
}

/// T5 attention: unscaled scaled-dot-product with a relative-position bias added to the scores. Only the
/// first block carries the bias table; the encoder computes the bias once and passes it to every block.
final class NFKT5Attention: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding?

    let heads: Int
    let keyDim: Int
    let configuration: NFKMLXT5Configuration

    init(_ c: NFKMLXT5Configuration, hasBias: Bool) {
        heads = c.heads
        keyDim = c.keyDim
        configuration = c
        _q.wrappedValue = Linear(c.dModel, c.inner, bias: false)
        _k.wrappedValue = Linear(c.dModel, c.inner, bias: false)
        _v.wrappedValue = Linear(c.dModel, c.inner, bias: false)
        _o.wrappedValue = Linear(c.inner, c.dModel, bias: false)
        if hasBias {
            _relativeAttentionBias.wrappedValue = Embedding(embeddingCount: c.relativeBuckets, dimensions: c.heads)
        }
    }

    /// `[B, S, dModel]` + a `[1, heads, S, S]` bias → `[B, S, dModel]`.
    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped([batch, length, heads, keyDim]).transposed(0, 2, 1, 3)
        }
        // T5 attention is UNSCALED; the relative-position bias enters as the additive mask.
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(q(x)), keys: split(k(x)), values: split(v(x)), scale: 1, mask: bias)
        return o(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * keyDim]))
    }

    /// The bucketed relative-position bias `[1, heads, S, S]` for a sequence of length `S`.
    func computeBias(_ length: Int) -> MLXArray {
        var buckets = [Int32](repeating: 0, count: length * length)
        for i in 0 ..< length {
            for j in 0 ..< length {
                buckets[i * length + j] = Int32(Self.bucket(j - i, numBuckets: configuration.relativeBuckets,
                                                            maxDistance: configuration.relativeMaxDistance))
            }
        }
        let indices = buckets.withUnsafeBufferPointer { MLXArray($0, [length, length]) }
        let values = relativeAttentionBias!(indices)                      // [S, S, heads]
        return values.transposed(2, 0, 1).expandedDimensions(axis: 0)     // [1, heads, S, S]
    }

    /// The Mesh-TensorFlow relative-position bucketing (bidirectional, as the encoder is).
    static func bucket(_ relativePosition: Int, numBuckets: Int, maxDistance: Int) -> Int {
        var result = 0
        let half = numBuckets / 2
        if relativePosition > 0 { result += half }
        let distance = abs(relativePosition)
        let maxExact = half / 2
        if distance < maxExact {
            return result + distance
        }
        let large = maxExact + Int(log(Double(distance) / Double(maxExact))
            / log(Double(maxDistance) / Double(maxExact)) * Double(half - maxExact))
        return result + min(large, half - 1)
    }
}

/// The gated feed-forward: `wo(gelu(wi_0(x)) · wi_1(x))`.
final class NFKT5FeedForward: Module {
    @ModuleInfo(key: "wi_0") var wi0: Linear
    @ModuleInfo(key: "wi_1") var wi1: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ c: NFKMLXT5Configuration) {
        _wi0.wrappedValue = Linear(c.dModel, c.ffDim, bias: false)
        _wi1.wrappedValue = Linear(c.dModel, c.ffDim, bias: false)
        _wo.wrappedValue = Linear(c.ffDim, c.dModel, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { wo(geluApproximate(wi0(x)) * wi1(x)) }
}

/// The self-attention sublayer: a T5LayerNorm, the attention, and a residual.
final class NFKT5SelfAttentionLayer: Module {
    @ModuleInfo(key: "SelfAttention") var attention: NFKT5Attention
    @ModuleInfo(key: "layer_norm") var layerNorm: NFKT5LayerNorm

    init(_ c: NFKMLXT5Configuration, hasBias: Bool) {
        _attention.wrappedValue = NFKT5Attention(c, hasBias: hasBias)
        _layerNorm.wrappedValue = NFKT5LayerNorm(c.dModel, eps: c.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray { x + attention(layerNorm(x), bias: bias) }
}

/// The feed-forward sublayer: a T5LayerNorm, the gated FFN, and a residual.
final class NFKT5FeedForwardLayer: Module {
    @ModuleInfo(key: "DenseReluDense") var ff: NFKT5FeedForward
    @ModuleInfo(key: "layer_norm") var layerNorm: NFKT5LayerNorm

    init(_ c: NFKMLXT5Configuration) {
        _ff.wrappedValue = NFKT5FeedForward(c)
        _layerNorm.wrappedValue = NFKT5LayerNorm(c.dModel, eps: c.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x + ff(layerNorm(x)) }
}

/// One T5 block: the self-attention sublayer then the feed-forward sublayer.
final class NFKT5Block: Module {
    @ModuleInfo(key: "layer") var layer: [Module]

    init(_ c: NFKMLXT5Configuration, hasBias: Bool) {
        _layer.wrappedValue = [NFKT5SelfAttentionLayer(c, hasBias: hasBias), NFKT5FeedForwardLayer(c)]
    }

    var selfAttention: NFKT5SelfAttentionLayer { layer[0] as! NFKT5SelfAttentionLayer }

    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        (layer[1] as! NFKT5FeedForwardLayer)(selfAttention(x, bias: bias))
    }
}

/// The T5 encoder stack.
final class NFKT5Stack: Module {
    @ModuleInfo(key: "block") var block: [NFKT5Block]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: NFKT5LayerNorm

    init(_ c: NFKMLXT5Configuration) {
        _block.wrappedValue = (0 ..< c.layers).map { NFKT5Block(c, hasBias: c.perLayerBias || $0 == 0) }
        _finalLayerNorm.wrappedValue = NFKT5LayerNorm(c.dModel, eps: c.layerNormEps)
    }
}

/// The T5 encoder: a shared token embedding and the encoder stack.
final class NFKMLXT5EncoderNet: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "encoder") var encoder: NFKT5Stack

    let configuration: NFKMLXT5Configuration

    init(_ c: NFKMLXT5Configuration) {
        configuration = c
        _shared.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.dModel)
        _encoder.wrappedValue = NFKT5Stack(c)
    }

    /// Token ids `[B, S]` → the text embedding `[B, S, dModel]`.
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var hidden = shared(tokens)
        let length = tokens.shape[1]
        // umT5 gives every layer its own bias; plain T5 shares block 0's across the stack.
        let sharedBias = configuration.perLayerBias ? nil
            : encoder.block[0].selfAttention.attention.computeBias(length)
        for block in encoder.block {
            let bias = sharedBias ?? block.selfAttention.attention.computeBias(length)
            hidden = block(hidden, bias: bias)
        }
        return encoder.finalLayerNorm(hidden)
    }
}

/// Holds the network for capture across an isolation boundary.
private final class NFKT5Holder: @unchecked Sendable {
    let net: NFKMLXT5EncoderNet
    init(_ net: NFKMLXT5EncoderNet) { self.net = net }
}

/// The T5 text encoder object.
@objc(NFKMLXT5Encoder)
public final class NFKMLXT5Encoder: NSObject {

    private let holder: NFKT5Holder

    init(net: NFKMLXT5EncoderNet) { holder = NFKT5Holder(net); super.init() }

    /// Token ids → the L2-unnormalized text embedding `[B, S, dModel]`.
    public func encode(_ tokens: MLXArray) -> MLXArray {
        let embedding = holder.net(tokens)
        eval(embedding)
        return embedding
    }

    /// Builds the encoder from a downloaded release directory (sharded diffusers safetensors).
    public static func encoder(configuration: NFKMLXT5Configuration = .xxl, directory: URL) throws -> NFKMLXT5Encoder {
        let net = makeNet(configuration)
        try loadWeights(into: net, from: directory)
        return NFKMLXT5Encoder(net: net)
    }

    static func makeNet(_ configuration: NFKMLXT5Configuration = .xxl) -> NFKMLXT5EncoderNet {
        NFKMLXT5EncoderNet(configuration)
    }

    /// Loads a checkpoint. All weights are at most 2-D, so no transpose applies; the module keys mirror
    /// the reference's `T5EncoderModel`.
    static func loadWeights(into net: NFKMLXT5EncoderNet, from directory: URL) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory) { $0 }
        try NFKMLXWeights.apply(arrays, to: net)
    }
}
