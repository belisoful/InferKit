//
//  NFKMLXGemma3n.swift
//  InferKitMLX
//
//  The Gemma 3n text decoder (`gemma3n_text`), the language model inside the E2B and E4B releases.
//  A distinct architecture from every other Gemma here, sharing the family's name and little of its
//  block: four mechanisms none of the others carry, and each one changes the forward pass.
//
//  - **AltUp** (Alternating Updates). The residual stream is `altupInputCount` parallel copies. A layer
//    reads one of them, and a learned map predicts the others from it before the block runs and
//    corrects them from the block's output afterward. The copies are the hidden state, so every layer
//    works on `[copies, batch, length, hidden]`.
//  - **LAuReL** (Learned Augmented Residual Layer). A low-rank detour beside the attention residual,
//    normalized and added, with the sum halved by `sqrt(2)`.
//  - **Per-layer embeddings**. Beside the token embedding, a second much wider embedding gives every
//    layer its own slice, gated into that layer's output through the layer's own projection.
//  - **Activation sparsity**. The first layers zero everything in the feed-forward's gate below a
//    per-token Gaussian cutoff, so most of the width contributes nothing.
//
//  Two more differences are smaller and equally load-bearing: attention runs at scale **1.0** rather
//  than the usual `1/sqrt(headDim)`, and the values carry their own normalization with no weight.
//
//  The normalization is `x_norm · w`, the PLAIN scale — NOT Gemma 3's `x_norm · (1 + w)`. The two are
//  indistinguishable by shape and differ in every number.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

/// The geometry of a Gemma 3n text decoder.
public struct NFKMLXGemma3nConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    /// The feed-forward width PER LAYER. The config states a list, not a scalar, and the released
    /// sizes vary it down the stack.
    public var intermediateSizes: [Int]
    public var vocabularySize: Int
    /// Rotary base for the full-attention layers.
    public var ropeTheta: Float
    /// Rotary base for the sliding-window layers, which the config states separately.
    public var ropeLocalTheta: Float
    public var slidingWindow: Int
    public var layerTypes: [NFKMLXGemmaAttentionKind]
    public var rmsEpsilon: Float
    /// The logits are squashed through `tanh` at this scale; 0 disables it.
    public var finalLogitSoftcap: Float

    /// The width of each layer's own input embedding (`hidden_size_per_layer_input`).
    public var perLayerInputSize: Int
    /// The row count of the per-layer input embedding, which is smaller than the token vocabulary:
    /// the ids past it are the vision and audio vocabularies, which carry no per-layer embedding.
    public var perLayerVocabularySize: Int

    /// The parallel residual copies AltUp maintains.
    public var altupInputCount: Int
    /// Which copy a layer reads and writes.
    public var altupActiveIndex: Int
    /// Whether the corrected active copy is scaled by the learned `correct_output_scale`.
    public var altupCorrectScale: Bool

    /// The rank of the LAuReL detour.
    public var laurelRank: Int

    /// How many trailing layers reuse an earlier layer's keys and values instead of computing them.
    public var sharedKeyValueLayers: Int

    /// The fraction of each layer's feed-forward gate held at zero, per layer; 0 is no sparsity.
    public var activationSparsity: [Float]

    public init(hiddenSize: Int = 2048, layerCount: Int = 35, headCount: Int = 8,
                keyValueHeadCount: Int = 2, headDimensions: Int = 256,
                intermediateSizes: [Int]? = nil, vocabularySize: Int = 262_400,
                ropeTheta: Float = 1_000_000, ropeLocalTheta: Float = 10_000,
                slidingWindow: Int = 512, layerTypes: [NFKMLXGemmaAttentionKind]? = nil,
                rmsEpsilon: Float = 1e-6, finalLogitSoftcap: Float = 30,
                perLayerInputSize: Int = 256, perLayerVocabularySize: Int = 262_144,
                altupInputCount: Int = 4, altupActiveIndex: Int = 0, altupCorrectScale: Bool = true,
                laurelRank: Int = 64, sharedKeyValueLayers: Int = 15,
                activationSparsity: [Float]? = nil) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.intermediateSizes = intermediateSizes ?? Array(repeating: 16_384, count: layerCount)
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.ropeLocalTheta = ropeLocalTheta
        self.slidingWindow = slidingWindow
        // The released pattern is four sliding layers then a full one.
        self.layerTypes = layerTypes ?? (0 ..< layerCount).map { ($0 + 1) % 5 == 0 ? .full : .sliding }
        self.rmsEpsilon = rmsEpsilon
        self.finalLogitSoftcap = finalLogitSoftcap
        self.perLayerInputSize = perLayerInputSize
        self.perLayerVocabularySize = perLayerVocabularySize
        self.altupInputCount = altupInputCount
        self.altupActiveIndex = altupActiveIndex
        self.altupCorrectScale = altupCorrectScale
        self.laurelRank = laurelRank
        self.sharedKeyValueLayers = sharedKeyValueLayers
        self.activationSparsity = activationSparsity ?? Array(repeating: 0, count: layerCount)
    }

    /// The index of the first layer that reuses another layer's keys and values.
    var firstSharedKeyValueLayer: Int { layerCount - sharedKeyValueLayers }

    /// Whether `index` reuses another layer's keys and values.
    func sharesKeyValues(layer index: Int) -> Bool {
        sharedKeyValueLayers > 0 && index >= firstSharedKeyValueLayer
    }

    /// The layer whose keys and values `index` reuses: the LAST non-shared layer of the SAME kind.
    ///
    /// @discussion A sliding layer reuses a sliding layer's keys and a full layer reuses a full
    /// layer's, so the two kinds are tracked separately rather than the last non-shared layer being
    /// taken for both.
    func keyValueDonor(forLayer index: Int) -> Int? {
        guard sharesKeyValues(layer: index) else { return nil }
        let kind = layerTypes[index]
        for candidate in stride(from: firstSharedKeyValueLayer - 1, through: 0, by: -1)
        where layerTypes[candidate] == kind {
            return candidate
        }
        return nil
    }

    /// Whether `index` is a non-shared layer whose keys and values a later layer reuses.
    func donatesKeyValues(layer index: Int) -> Bool {
        guard index < firstSharedKeyValueLayer else { return false }
        let kind = layerTypes[index]
        for candidate in stride(from: firstSharedKeyValueLayer - 1, through: index + 1, by: -1)
        where layerTypes[candidate] == kind {
            return false
        }
        return true
    }

    /// The released `google/gemma-3n-E2B-it` text decoder.
    public static let e2b = NFKMLXGemma3nConfiguration()

    /// A small configuration exercising every mechanism: both attention kinds, a shared-key-value
    /// tail of each kind, and a sparse layer beside a dense one.
    public static let tiny = NFKMLXGemma3nConfiguration(
        hiddenSize: 64, layerCount: 6, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
        intermediateSizes: Array(repeating: 96, count: 6), vocabularySize: 140,
        slidingWindow: 4,
        layerTypes: [.sliding, .sliding, .full, .sliding, .sliding, .full],
        perLayerInputSize: 8, perLayerVocabularySize: 128,
        laurelRank: 8, sharedKeyValueLayers: 2,
        activationSparsity: [0.95, 0.95, 0, 0, 0, 0])
}

// MARK: - Normalization

/// Gemma 3n's RMS normalization: `x_norm · w`, or `x_norm` alone where the reference builds it
/// `with_scale: false` (the value normalization, which carries no parameter at all).
///
/// The reference normalizes in float32 and casts back, which a `.checkpoint`-precision load makes
/// visible.
final class NFKGemma3nNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray?
    let epsilon: Float

    init(dimensions: Int, eps: Float, scaled: Bool = true) {
        _weight.wrappedValue = scaled ? MLXArray.ones([dimensions]) : nil
        epsilon = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let wide = x.asType(.float32)
        // `mean(x²) + eps` then `^-0.5`, the reference's spelling of the reciprocal square root.
        var normalized = wide * rsqrt((wide * wide).mean(axis: -1, keepDims: true) + epsilon)
        if let weight {
            normalized = normalized * weight.asType(.float32)
        }
        return normalized.asType(x.dtype)
    }
}

// MARK: - LAuReL

/// The Learned Augmented Residual Layer: a low-rank detour added back to its own input, normalized.
final class NFKGemma3nLaurel: Module {
    @ModuleInfo(key: "linear_left") var left: Linear
    @ModuleInfo(key: "linear_right") var right: Linear
    @ModuleInfo(key: "post_laurel_norm") var norm: NFKGemma3nNorm

    init(_ c: NFKMLXGemma3nConfiguration) {
        _left.wrappedValue = Linear(c.hiddenSize, c.laurelRank, bias: false)
        _right.wrappedValue = Linear(c.laurelRank, c.hiddenSize, bias: false)
        _norm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x + norm(right(left(x))) }
}

// MARK: - Feed-forward

/// Gemma 3n's GeGLU feed-forward, with the family's activation sparsity on the gate.
final class NFKGemma3nFeedForward: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    /// The standard-normal quantile of the target sparsity, or nil where the layer is dense. It is a
    /// constant of the configuration, so it is computed once here rather than per token.
    let standardDeviations: Float?

    init(_ c: NFKMLXGemma3nConfiguration, layer: Int) {
        let width = c.intermediateSizes[layer]
        _gate.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        _down.wrappedValue = Linear(width, c.hiddenSize, bias: false)
        let sparsity = c.activationSparsity[layer]
        standardDeviations = sparsity > 0 ? Float(NFKGemma3nStatistics.standardNormalQuantile(Double(sparsity))) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gated = gate(x)
        if let standardDeviations {
            gated = sparsified(gated, standardDeviations: standardDeviations)
        }
        return down(geluApproximate(gated) * up(x))
    }

    /// Everything below `mean + k·sd` of the token's own gate held at zero, the reference's
    /// `_gaussian_topk`. The deviation is the POPULATION one (the reference passes `unbiased=False`).
    private func sparsified(_ x: MLXArray, standardDeviations k: Float) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let centered = x - mean
        let deviation = sqrt((centered * centered).mean(axis: -1, keepDims: true))
        return maximum(x - (mean + deviation * k), 0)
    }
}

/// The standard normal's inverse cumulative distribution, which the activation sparsity's cutoff is
/// expressed in. The reference reads it from `jax.scipy.stats.norm.ppf`; there is no Foundation
/// equivalent, so it is Acklam's rational approximation refined once by Halley's method against
/// `erfc`, which reaches the double's full precision.
enum NFKGemma3nStatistics {
    static func standardNormalQuantile(_ p: Double) -> Double {
        guard p > 0, p < 1 else { return p <= 0 ? -.infinity : .infinity }

        let a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        let b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                 6.680131188771972e+01, -1.328068155288572e+01]
        let c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                 -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        let d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                 3.754408661907416e+00]
        let low = 0.02425, high = 1 - low

        var x: Double
        if p < low {
            let q = (-2 * Foundation.log(p)).squareRoot()
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        } else if p <= high {
            let q = p - 0.5, r = q * q
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        } else {
            let q = (-2 * Foundation.log(1 - p)).squareRoot()
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        }

        // One Halley step against the true tail probability removes the approximation's ~1e-9 error.
        let error = 0.5 * Foundation.erfc(-x / 2.0.squareRoot()) - p
        let density = (2 * Double.pi).squareRoot() * Foundation.exp(x * x / 2)
        let u = error * density
        return x - u / (1 + x * u / 2)
    }
}

// MARK: - AltUp

/// Alternating Updates: the learned map between the residual stream's parallel copies.
///
/// `predict` builds every copy from the active one before the block runs; `correct` propagates what
/// the block did to the active copy back into the others. Both read a per-token router over the
/// active copy, so the mixing depends on the content rather than being fixed.
final class NFKGemma3nAltUp: Module {
    @ParameterInfo(key: "correct_output_scale") var correctOutputScale: MLXArray
    @ModuleInfo(key: "correction_coefs") var correctionCoefficients: Linear
    @ModuleInfo(key: "prediction_coefs") var predictionCoefficients: Linear
    @ModuleInfo(key: "modality_router") var router: Linear
    @ModuleInfo(key: "router_norm") var routerNorm: NFKGemma3nNorm

    let inputCount: Int
    let activeIndex: Int
    /// The router's input scale is the hidden size's RECIPROCAL, not its inverse square root.
    let routerInputScale: Float

    init(_ c: NFKMLXGemma3nConfiguration) {
        inputCount = c.altupInputCount
        activeIndex = c.altupActiveIndex
        routerInputScale = 1 / Float(c.hiddenSize)
        _correctOutputScale.wrappedValue = MLXArray.zeros([c.hiddenSize])
        _correctionCoefficients.wrappedValue = Linear(c.altupInputCount, c.altupInputCount, bias: false)
        _predictionCoefficients.wrappedValue = Linear(c.altupInputCount, c.altupInputCount * c.altupInputCount,
                                                      bias: false)
        _router.wrappedValue = Linear(c.hiddenSize, c.altupInputCount, bias: false)
        _routerNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    /// `[batch, length, inputCount]`, the per-token mixing weights. The `tanh` runs in float32.
    private func modalities(_ x: MLXArray) -> MLXArray {
        let routed = router(routerNorm(x) * routerInputScale)
        return tanh(routed.asType(.float32)).asType(x.dtype)
    }

    /// Every copy predicted from the active one. `hidden` is `[copies, batch, length, hidden]`.
    func predict(_ hidden: MLXArray) -> MLXArray {
        let m = modalities(hidden[activeIndex])
        let (batch, length) = (m.shape[0], m.shape[1])
        // The reference transposes each token's matrix so the multiply mixes the copies the right way.
        let coefficients = predictionCoefficients(m)
            .reshaped([batch, length, inputCount, inputCount])
            .transposed(0, 1, 3, 2)
        let mixed = matmul(hidden.transposed(1, 2, 3, 0), coefficients).transposed(3, 0, 1, 2)
        return (mixed + hidden).asType(hidden.dtype)
    }

    /// The predictions corrected by what the block produced from the active copy.
    func correct(predictions: MLXArray, activated: MLXArray) -> MLXArray {
        let m = modalities(activated)
        let innovation = (activated - predictions[activeIndex]).expandedDimensions(axis: 0)
        // A coefficient per copy, per token; the `+ 1` makes an untrained map the identity.
        let coefficients = (correctionCoefficients(m) + 1).transposed(2, 0, 1).expandedDimensions(axis: -1)
        return (innovation * coefficients + predictions).asType(activated.dtype)
    }

    /// The learned per-channel scale on the corrected active copy.
    func scaleCorrectedOutput(_ x: MLXArray) -> MLXArray { x * correctOutputScale }
}

// MARK: - Attention

/// Gemma 3n attention: grouped queries, per-head normalization of the queries, keys, AND values, and
/// a trailing run of layers that computes no keys or values at all.
///
/// The score scale is **1.0**. Every other decoder here divides by the square root of the head width;
/// this one does not, and the query normalization is what stands in for it.
final class NFKGemma3nAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: NFKGemma3nNorm
    @ModuleInfo(key: "k_proj") var keyProjection: Linear?
    @ModuleInfo(key: "v_proj") var valueProjection: Linear?
    @ModuleInfo(key: "k_norm") var keyNorm: NFKGemma3nNorm?
    @ModuleInfo(key: "v_norm") var valueNorm: NFKGemma3nNorm?

    let heads: Int
    let keyValueHeads: Int
    let headDimensions: Int
    let ropeBase: Float
    let sharesKeyValues: Bool

    init(_ c: NFKMLXGemma3nConfiguration, layer: Int) {
        heads = c.headCount
        keyValueHeads = c.keyValueHeadCount
        headDimensions = c.headDimensions
        let full = c.layerTypes[layer] == .full
        ropeBase = full ? c.ropeTheta : c.ropeLocalTheta
        sharesKeyValues = c.sharesKeyValues(layer: layer)

        _queryProjection.wrappedValue = Linear(c.hiddenSize, c.headCount * c.headDimensions, bias: false)
        _outputProjection.wrappedValue = Linear(c.headCount * c.headDimensions, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        if !sharesKeyValues {
            _keyProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions, bias: false)
            _valueProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions, bias: false)
            _keyNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
            // The value normalization carries NO weight, which is why the checkpoint has no tensor
            // for it and a strict load would refuse one.
            _valueNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon,
                                                     scaled: false)
        }
        super.init()
    }

    /// - Parameters:
    ///   - shared: the donor layer's keys and values, for a layer that computes none of its own.
    ///   - cache: appended to BEFORE the attention runs, so a cached step attends over the history
    ///     rather than over its own single position.
    /// - Returns: the attention output and the keys and values this layer attended over, which a
    ///   donor layer's caller stores for the layers that reuse them.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, offset: Int,
                        shared: (keys: MLXArray, values: MLXArray)?,
                        cache: NFKMLXGemma3nCache?, layer: Int)
        -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let (batch, length) = (x.shape[0], x.shape[1])

        var queries = queryNorm(queryProjection(x).reshaped([batch, length, heads, headDimensions]))
            .transposed(0, 2, 1, 3)
        queries = MLXFast.RoPE(queries, dimensions: headDimensions, traditional: false, base: ropeBase,
                               scale: 1, offset: offset)

        var keys: MLXArray, values: MLXArray
        if sharesKeyValues, let shared {
            // Already the donor's full-length keys, cache included; appending them again would
            // duplicate the history.
            (keys, values) = (shared.keys, shared.values)
        } else {
            guard let keyProjection, let valueProjection, let keyNorm, let valueNorm else {
                fatalError("a Gemma 3n layer that computes its own keys and values is missing their projections")
            }
            var k = keyNorm(keyProjection(x).reshaped([batch, length, keyValueHeads, headDimensions]))
                .transposed(0, 2, 1, 3)
            k = MLXFast.RoPE(k, dimensions: headDimensions, traditional: false, base: ropeBase,
                             scale: 1, offset: offset)
            keys = k
            values = valueNorm(valueProjection(x).reshaped([batch, length, keyValueHeads, headDimensions]))
                .transposed(0, 2, 1, 3)
            if let cache {
                (keys, values) = cache.update(layer: layer, keys: keys, values: values)
            }
        }

        // The masks are built float32; a `.checkpoint`-precision load makes this a bf16 module and the
        // fused attention refuses a mask that does not promote to its own type.
        let typedMask = mask.map { $0.asType(queries.dtype) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1, mask: typedMask)
        let output = outputProjection(
            attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDimensions]))
        return (output, keys, values)
    }
}

// MARK: - Decoder layer

/// One Gemma 3n block: AltUp's prediction, the attention and LAuReL detour halved together, the
/// feed-forward, AltUp's correction, and the per-layer input gated into the inactive copies.
final class NFKGemma3nBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKGemma3nAttention
    @ModuleInfo(key: "mlp") var feedForward: NFKGemma3nFeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKGemma3nNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: NFKGemma3nNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: NFKGemma3nNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: NFKGemma3nNorm
    @ModuleInfo(key: "altup") var altup: NFKGemma3nAltUp
    @ModuleInfo(key: "laurel") var laurel: NFKGemma3nLaurel
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: NFKGemma3nNorm

    let activeIndex: Int
    let correctScale: Bool

    init(_ c: NFKMLXGemma3nConfiguration, layer: Int) {
        activeIndex = c.altupActiveIndex
        correctScale = c.altupCorrectScale
        _attention.wrappedValue = NFKGemma3nAttention(c, layer: layer)
        _feedForward.wrappedValue = NFKGemma3nFeedForward(c, layer: layer)
        _inputNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postAttentionNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _preFeedForwardNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postFeedForwardNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _altup.wrappedValue = NFKGemma3nAltUp(c)
        _laurel.wrappedValue = NFKGemma3nLaurel(c)
        _perLayerInputGate.wrappedValue = Linear(c.hiddenSize, c.perLayerInputSize, bias: false)
        _perLayerProjection.wrappedValue = Linear(c.perLayerInputSize, c.hiddenSize, bias: false)
        _postPerLayerInputNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, perLayerInput: MLXArray, mask: MLXArray?, offset: Int,
                        shared: (keys: MLXArray, values: MLXArray)?,
                        cache: NFKMLXGemma3nCache?, layer: Int)
        -> (hidden: MLXArray, keys: MLXArray, values: MLXArray) {
        let predictions = altup.predict(hidden)
        let active = predictions[activeIndex]
        let normed = inputNorm(active)

        let laurelOutput = laurel(normed)
        let (attended, keys, values) = attention(normed, mask: mask, offset: offset, shared: shared,
                                                 cache: cache, layer: layer)

        let gated = active + postAttentionNorm(attended)
        // The two residual paths are averaged rather than summed.
        let withLaurel = (gated + laurelOutput) / Float(2).squareRoot()

        let projected = postFeedForwardNorm(feedForward(preFeedForwardNorm(withLaurel)))
        var corrected = altup.correct(predictions: predictions, activated: withLaurel + projected)

        var first = corrected[activeIndex]
        if correctScale {
            first = altup.scaleCorrectedOutput(first)
        }
        first = postPerLayerInputNorm(perLayerProjection(geluApproximate(perLayerInputGate(first)) * perLayerInput))

        // The per-layer input reaches the INACTIVE copies only. Built by concatenation rather than a
        // slice assignment, which MLX treats as an update on the array rather than a rebinding.
        let inactive = corrected[1...] + first.expandedDimensions(axis: 0)
        corrected = concatenated([corrected[0 ..< 1], inactive], axis: 0)
        return (corrected, keys, values)
    }
}

// MARK: - Masks

/// The attention masks a Gemma 3n pass needs, one per layer kind, over the WHOLE key range.
///
/// @discussion A sliding layer's keys are kept full-length here rather than trimmed to the window,
/// because the layers that share keys and values reuse a donor layer's FULL-length keys — a donor
/// trimmed to its own window would hand a shorter history to the layers reading it. The window is
/// therefore enforced by the mask alone, which is exact.
enum NFKMLXGemma3nMasks {
    static func make(length: Int, offset: Int, window: Int) -> (full: MLXArray?, sliding: MLXArray?) {
        let total = offset + length
        let rows = MLXArray(Int32(offset) ..< Int32(total)).reshaped([length, 1])
        let columns = MLXArray(Int32(0) ..< Int32(total)).reshaped([1, total])
        func additive(_ allowed: MLXArray) -> MLXArray {
            MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        }
        let causal = columns .<= rows
        let full = additive(causal)
        let sliding = additive(causal .&& ((rows - columns) .< Int32(window)))
        return (full, sliding)
    }
}

// MARK: - The decoder

/// The Gemma 3n text decoder.
public final class NFKMLXGemma3nNet: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: NFKGemma3nNorm
    @ModuleInfo(key: "altup_projections") var altupProjections: [Linear]
    @ModuleInfo(key: "altup_unembed_projections") var altupUnembedProjections: [Linear]
    @ModuleInfo(key: "layers") var layers: [NFKGemma3nBlock]
    @ModuleInfo(key: "norm") var norm: NFKGemma3nNorm

    public let configuration: NFKMLXGemma3nConfiguration
    private let embeddingScale: Float
    private let perLayerEmbeddingScale: Float
    private let perLayerProjectionScale: Float

    public init(_ c: NFKMLXGemma3nConfiguration) {
        configuration = c
        embeddingScale = sqrt(Float(c.hiddenSize))
        perLayerEmbeddingScale = sqrt(Float(c.perLayerInputSize))
        perLayerProjectionScale = 1 / sqrt(Float(c.hiddenSize))

        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _embedTokensPerLayer.wrappedValue = Embedding(embeddingCount: c.perLayerVocabularySize,
                                                      dimensions: c.layerCount * c.perLayerInputSize)
        _perLayerModelProjection.wrappedValue = Linear(c.hiddenSize, c.layerCount * c.perLayerInputSize,
                                                       bias: false)
        _perLayerProjectionNorm.wrappedValue = NFKGemma3nNorm(dimensions: c.perLayerInputSize, eps: c.rmsEpsilon)
        _altupProjections.wrappedValue = (1 ..< c.altupInputCount).map { _ in
            Linear(c.hiddenSize, c.hiddenSize, bias: false)
        }
        _altupUnembedProjections.wrappedValue = (1 ..< c.altupInputCount).map { _ in
            Linear(c.hiddenSize, c.hiddenSize, bias: false)
        }
        _layers.wrappedValue = (0 ..< c.layerCount).map { NFKGemma3nBlock(c, layer: $0) }
        _norm.wrappedValue = NFKGemma3nNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    /// The scaled token embedding, the main stream a caller splices image or audio soft tokens into.
    public func embed(_ tokens: MLXArray) -> MLXArray { embedTokens(tokens) * embeddingScale }

    /// Each layer's own input embedding for these tokens, `[batch, length, layers, perLayerInput]`.
    ///
    /// @discussion The per-layer vocabulary is smaller than the token vocabulary — the ids past it are
    /// the vision and audio tokens, which have no per-layer embedding — so those ids read row zero.
    public func perLayerEmbeddings(_ tokens: MLXArray) -> MLXArray {
        let c = configuration
        let clamped = MLX.where(tokens .< Int32(c.perLayerVocabularySize), tokens, MLXArray(Int32(0)))
        let embedded = embedTokensPerLayer(clamped) * perLayerEmbeddingScale
        return embedded.reshaped(tokens.shape + [c.layerCount, c.perLayerInputSize])
    }

    /// The per-layer inputs the stack reads: each layer's own embedding added to a slice of a
    /// projection of the main embedding, the sum halved.
    public func projectedPerLayerInputs(embeddings: MLXArray, perLayer: MLXArray?) -> MLXArray {
        let c = configuration
        var projection = perLayerModelProjection(embeddings) * perLayerProjectionScale
        projection = perLayerProjectionNorm(
            projection.reshaped(Array(embeddings.shape.dropLast()) + [c.layerCount, c.perLayerInputSize]))
        guard let perLayer else { return projection }
        return (projection + perLayer) * (1 / Float(2).squareRoot())
    }

    /// One hidden state expanded into AltUp's parallel copies, each projected then rescaled to the
    /// magnitude of the first — the reference's initialization, which keeps the copies comparable.
    private func expanded(_ hidden: MLXArray) -> MLXArray {
        var copies = [hidden]
        let target = sqrt((hidden * hidden).mean(axis: -1, keepDims: true))
        for projection in altupProjections {
            copies.append(matched(projection(hidden), to: target, like: hidden))
        }
        return stacked(copies, axis: 0)
    }

    private func matched(_ x: MLXArray, to target: MLXArray, like reference: MLXArray) -> MLXArray {
        let current = x.asType(reference.dtype)
        let magnitude = sqrt(maximum((current * current).mean(axis: -1, keepDims: true), MLXArray(Float(1e-5))))
        return current * target / magnitude
    }

    /// The post-norm hidden states over already-embedded inputs.
    ///
    /// - Parameters:
    ///   - perLayerInputs: `[batch, length, layers, perLayerInput]` from
    ///     ``projectedPerLayerInputs(embeddings:perLayer:)``.
    ///   - cache: the key-value cache, or nil to run the whole sequence in one pass.
    public func hiddenStates(fromEmbeddings embeddings: MLXArray, perLayerInputs: MLXArray,
                             cache: NFKMLXGemma3nCache? = nil) -> MLXArray {
        var trace = [MLXArray]()
        return forward(embeddings, perLayerInputs: perLayerInputs, cache: cache, trace: &trace)
    }

    /// The output projection on its own: post-norm hidden states → logits, tied to the token
    /// embedding and soft-capped where the configuration says so.
    public func logits(fromHidden hidden: MLXArray) -> MLXArray {
        var logits = embedTokens.asLinear(hidden)
        if configuration.finalLogitSoftcap > 0 {
            logits = tanh(logits / configuration.finalLogitSoftcap) * configuration.finalLogitSoftcap
        }
        return logits
    }

    /// The logits `[batch, length, vocabulary]` for token ids.
    public func callAsFunction(_ tokens: MLXArray, cache: NFKMLXGemma3nCache? = nil) -> MLXArray {
        let embeddings = embed(tokens)
        let perLayer = projectedPerLayerInputs(embeddings: embeddings, perLayer: perLayerEmbeddings(tokens))
        return logits(fromHidden: hiddenStates(fromEmbeddings: embeddings, perLayerInputs: perLayer, cache: cache))
    }

    /// The state entering the stack and the state each layer produces — the reference's
    /// `output_hidden_states` convention, so a divergence is located to a layer rather than guessed.
    ///
    /// @discussion Each entry is the ACTIVE copy of the residual stream, which is what the reference
    /// reports. The final norm is NOT applied to the last entry: Gemma 3n's reference does not tie
    /// its last hidden state to the last reported one, so the norm and the merge of AltUp's copies
    /// belong to `last_hidden_state` alone. Reading the Llama convention here reports a divergence at
    /// the final layer while every number is right.
    public func layerStates(_ tokens: MLXArray) -> [MLXArray] {
        var trace = [MLXArray]()
        let embeddings = embed(tokens)
        let perLayer = projectedPerLayerInputs(embeddings: embeddings, perLayer: perLayerEmbeddings(tokens))
        _ = forward(embeddings, perLayerInputs: perLayer, cache: nil, trace: &trace)
        return trace
    }

    private func forward(_ embeddings: MLXArray, perLayerInputs: MLXArray, cache: NFKMLXGemma3nCache?,
                         trace: inout [MLXArray]) -> MLXArray {
        let c = configuration
        let length = embeddings.shape[1]
        let offset = cache?.offset ?? 0
        let masks = NFKMLXGemma3nMasks.make(length: length, offset: offset, window: c.slidingWindow)

        var hidden = expanded(embeddings)
        trace.append(embeddings)

        // A donor layer's keys and values, read by the trailing layers that compute none.
        var donated = [Int: (keys: MLXArray, values: MLXArray)]()

        for (index, layer) in layers.enumerated() {
            let kind = c.layerTypes[index]
            let shared = c.keyValueDonor(forLayer: index).flatMap { donated[$0] }
            let (next, keys, values) = layer(hidden,
                                             perLayerInput: perLayerInputs[0..., 0..., index, 0...],
                                             mask: kind == .full ? masks.full : masks.sliding,
                                             offset: offset, shared: shared,
                                             cache: cache, layer: index)
            hidden = next
            if c.donatesKeyValues(layer: index) {
                // What the donor ACTUALLY attended over, its cached history included, because that is
                // what the layers sharing its keys read.
                donated[index] = (keys, values)
            }
            trace.append(hidden[c.altupActiveIndex])
        }

        // The copies are projected back together, rescaled to the first one's magnitude, and averaged.
        var copies = [hidden[0]]
        let target = sqrt((hidden[0] * hidden[0]).mean(axis: -1, keepDims: true))
        for (index, projection) in altupUnembedProjections.enumerated() {
            copies.append(matched(projection(hidden[index + 1]), to: target, like: hidden[0]))
        }
        let merged = norm(stacked(copies, axis: 0).mean(axis: 0))
        cache?.advance(by: length)
        return merged
    }
}

// MARK: - Cache

/// Gemma 3n's key-value cache.
///
/// @discussion Every layer's keys are kept in full rather than trimmed to a sliding window, and the
/// window is enforced by the mask. That is what key-value SHARING requires: a sliding donor layer
/// hands its keys to layers further down, and a donor trimmed to its own window would hand them a
/// shorter history than the reference does. Sharing already removes most of the cache — the trailing
/// `num_kv_shared_layers` store nothing at all — so the layers that remain can afford to be exact.
public final class NFKMLXGemma3nCache {
    private let store: NFKMLXKeyValueCache

    public init(layerCount: Int) {
        store = NFKMLXKeyValueCache(layerCount: layerCount)
    }

    /// Positions appended so far, which is the rotary offset of the next one.
    public var offset: Int { store.offset }

    func update(layer: Int, keys: MLXArray, values: MLXArray) -> (keys: MLXArray, values: MLXArray) {
        store.update(layer: layer, keys: keys, values: values)
    }

    func advance(by count: Int) { store.advance(by: count) }
}

// MARK: - Building from a release

/// Building a Gemma 3n decoder, reading a release's configuration, and loading its weights.
@objc(NFKMLXGemma3nLanguage)
public final class NFKMLXGemma3nLanguage: NSObject {

    static func makeNet(_ configuration: NFKMLXGemma3nConfiguration = .e2b) -> NFKMLXGemma3nNet {
        NFKMLXGemma3nNet(configuration)
    }

    /// Reads a released `config.json`, whose decoder sits under `text_config` in the tri-modal
    /// releases.
    ///
    /// @discussion Accepts `gemma3n_text` and the `gemma3n` wrapper. Gemma 3 and Gemma 4 share the
    /// family name and none of this block, so they are refused rather than loaded into this stack.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXGemma3nConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        return try configuration(fromJSON: json)
    }

    static func configuration(fromJSON json: [String: Any]) throws -> NFKMLXGemma3nConfiguration {
        let text = (json["text_config"] as? [String: Any]) ?? json
        let outer = (json["model_type"] as? String) ?? ""
        let kind = (text["model_type"] as? String) ?? outer
        guard kind == "gemma3n_text" || (kind == "gemma3n" && json["text_config"] == nil) else {
            throw NFKMLXError.unsupportedConfiguration("this reads a gemma3n text decoder, not \(kind)")
        }
        if !outer.isEmpty, outer != "gemma3n", outer != "gemma3n_text" {
            throw NFKMLXError.unsupportedConfiguration("this reads a gemma3n release, not \(outer)")
        }

        func integer(_ key: String, _ fallback: Int) -> Int { (text[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (text[key] as? NSNumber)?.floatValue ?? fallback }
        func flag(_ key: String, _ fallback: Bool) -> Bool { (text[key] as? NSNumber)?.boolValue ?? fallback }

        let layerCount = integer("num_hidden_layers", 35)

        // The feed-forward width is stated per layer, and the releases do vary it; a scalar is
        // accepted and repeated so a hand-written config still reads.
        let widths: [Int]
        if let list = text["intermediate_size"] as? [NSNumber] {
            widths = list.map(\.intValue)
        } else {
            widths = Array(repeating: integer("intermediate_size", 16_384), count: layerCount)
        }
        guard widths.count == layerCount else {
            throw NFKMLXError.unsupportedConfiguration(
                "intermediate_size names \(widths.count) layers of \(layerCount)")
        }

        let sparsity: [Float]
        if let list = text["activation_sparsity_pattern"] as? [NSNumber] {
            sparsity = list.map(\.floatValue)
        } else {
            sparsity = Array(repeating: 0, count: layerCount)
        }
        guard sparsity.count == layerCount else {
            throw NFKMLXError.unsupportedConfiguration(
                "activation_sparsity_pattern names \(sparsity.count) layers of \(layerCount)")
        }

        let layerTypes = (text["layer_types"] as? [String])?.compactMap { NFKMLXGemmaAttentionKind(rawValue: $0) }
        if let layerTypes, layerTypes.count != layerCount {
            throw NFKMLXError.unsupportedConfiguration("layer_types names \(layerTypes.count) layers of \(layerCount)")
        }

        // The rotary bases are stated either as a pair of scalars or, standardized by a newer
        // transformers, per layer kind under `rope_parameters`.
        var globalTheta = real("rope_theta", 1_000_000)
        var localTheta = real("rope_local_base_freq", 10_000)
        if let parameters = text["rope_parameters"] as? [String: Any] {
            if let full = parameters["full_attention"] as? [String: Any] {
                globalTheta = (full["rope_theta"] as? NSNumber)?.floatValue ?? globalTheta
                let type = ((full["rope_type"] ?? full["type"]) as? String) ?? "default"
                guard type == "default" else {
                    throw NFKMLXError.unsupportedConfiguration(
                        "Gemma 3n rope scaling of kind \(type) is not implemented")
                }
            }
            if let sliding = parameters["sliding_attention"] as? [String: Any] {
                localTheta = (sliding["rope_theta"] as? NSNumber)?.floatValue ?? localTheta
            }
        }
        if text["rope_scaling"] is [String: Any] {
            throw NFKMLXError.unsupportedConfiguration("Gemma 3n rope scaling is not implemented")
        }

        return NFKMLXGemma3nConfiguration(
            hiddenSize: integer("hidden_size", 2048),
            layerCount: layerCount,
            headCount: integer("num_attention_heads", 8),
            keyValueHeadCount: integer("num_key_value_heads", 2),
            headDimensions: integer("head_dim", 256),
            intermediateSizes: widths,
            vocabularySize: integer("vocab_size", 262_400),
            ropeTheta: globalTheta,
            ropeLocalTheta: localTheta,
            slidingWindow: integer("sliding_window", 512),
            layerTypes: layerTypes,
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            finalLogitSoftcap: real("final_logit_softcapping", 30),
            perLayerInputSize: integer("hidden_size_per_layer_input", 256),
            perLayerVocabularySize: integer("vocab_size_per_layer_input", 262_144),
            altupInputCount: integer("altup_num_inputs", 4),
            altupActiveIndex: integer("altup_active_idx", 0),
            altupCorrectScale: flag("altup_correct_scale", true),
            laurelRank: integer("laurel_rank", 64),
            sharedKeyValueLayers: integer("num_kv_shared_layers", 15),
            activationSparsity: sparsity)
    }

    /// The decoder's module key for a checkpoint key, or nil for a tensor that is not the decoder's.
    ///
    /// @discussion A text-only release stores the decoder under `model.`; a tri-modal one written by
    /// transformers 4.x under `language_model.model.` and one written by 5.x under
    /// `model.language_model.`. Three kinds of tensor are dropped: the tied `lm_head.weight`, which is
    /// the embedding written out again; the vision and audio towers with their embedders; and the key
    /// and value projections of the layers that SHARE them, which the release still carries and the
    /// reference itself ignores as unexpected.
    static func decoderName(of key: String, configuration: NFKMLXGemma3nConfiguration) -> String? {
        if key.hasSuffix("lm_head.weight") { return nil }
        for tower in ["vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"]
        where key.contains(".\(tower).") || key.hasPrefix("\(tower).") {
            return nil
        }
        guard let name = stripped(key, prefixes: ["model.language_model.", "language_model.model.", "model."])
        else { return nil }
        if let layer = sharedLayerIndex(of: name), configuration.sharesKeyValues(layer: layer) {
            return nil
        }
        return name
    }

    /// The layer index of a `layers.N.self_attn.{k_proj,v_proj,k_norm,v_norm}` key, or nil.
    private static func sharedLayerIndex(of name: String) -> Int? {
        let parts = name.split(separator: ".")
        guard parts.count >= 4, parts[0] == "layers", parts[2] == "self_attn",
              ["k_proj", "v_proj", "k_norm", "v_norm"].contains(String(parts[3])),
              let index = Int(parts[1]) else { return nil }
        return index
    }

    /// `key` with the first matching prefix removed, or nil when none matches.
    static func stripped(_ key: String, prefixes: [String]) -> String? {
        for prefix in prefixes where key.hasPrefix(prefix) {
            return String(key.dropFirst(prefix.count))
        }
        return nil
    }

    /// Loads the decoder from a released directory, single-file or sharded, taking only the language
    /// model's tensors; a strict apply then proves the decoder's own set is complete.
    static func loadWeights(into net: NFKMLXGemma3nNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        let configuration = net.configuration
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision,
                                                     remap: { decoderName(of: $0, configuration: configuration) })
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
