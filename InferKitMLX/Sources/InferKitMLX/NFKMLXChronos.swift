// Chronos-Bolt (`amazon/chronos-bolt-base`, Amazon, Apache-2.0), a time-series forecaster: a patched
// T5 encoder-decoder that reads a numeric context window and emits a quantile forecast over a horizon.
// Ported from the `chronos` package's `ChronosBoltModelForForecasting` (chronos_bolt.py), the reference
// the parity test measures against (`run_reference.py chronos`).
//
// The transformer is a plain T5 v1.0 (single `wi` feed-forward, ReLU, unscaled attention with a bucketed
// relative-position bias, RMS `T5LayerNorm`). The decoder runs a SINGLE token (a `decoder_start` id)
// that cross-attends to the encoder output, so its self-attention is over one position.
//
// Time-series has no place in InferKit's core input/output key vocabulary, so this ships as a Swift
// object (`NFKMLXChronos.forecast(context:horizon:)`), the way `NFKMLXModernBERTReranker` is an object
// rather than a backend.

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

/// The Chronos-Bolt configuration (`amazon/chronos-bolt-base` `config.json` + `chronos_config`).
public struct NFKMLXChronosConfiguration: Sendable {
    public var dModel = 768
    public var encoderLayers = 12
    public var decoderLayers = 12
    public var heads = 12
    public var keyDim = 64
    public var dFF = 3072
    public var relativeBuckets = 32
    public var relativeMaxDistance = 128
    public var layerNormEps: Float = 1e-6
    public var inputPatchSize = 16
    public var contextLength = 2048
    public var predictionLength = 64
    public var quantiles: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    public init() {}
    /// A tiny configuration for wiring and shape tests.
    public static let tiny = { var c = NFKMLXChronosConfiguration(); c.dModel = 32; c.encoderLayers = 2
        c.decoderLayers = 2; c.heads = 2; c.keyDim = 16; c.dFF = 48; c.predictionLength = 8; return c }()
}

// MARK: - T5 primitives (plain v1.0)

/// `T5LayerNorm`: an RMS norm with a learned scale and no mean subtraction.
final class NFKChronosLayerNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float) { _weight.wrappedValue = MLXArray.ones([dim]); self.eps = eps }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * rsqrt((x * x).mean(axis: -1, keepDims: true) + eps) * weight
    }
}

/// A T5 attention (self or cross): unscaled scaled-dot-product with an optional bucketed relative-position
/// bias (only the first block of a stack carries the bias table).
final class NFKChronosAttention: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding?
    let heads: Int, keyDim: Int, buckets: Int, maxDistance: Int, bidirectional: Bool

    init(_ c: NFKMLXChronosConfiguration, hasBias: Bool, bidirectional: Bool) {
        heads = c.heads; keyDim = c.keyDim; buckets = c.relativeBuckets
        maxDistance = c.relativeMaxDistance; self.bidirectional = bidirectional
        let inner = c.heads * c.keyDim
        _q.wrappedValue = Linear(c.dModel, inner, bias: false)
        _k.wrappedValue = Linear(c.dModel, inner, bias: false)
        _v.wrappedValue = Linear(c.dModel, inner, bias: false)
        _o.wrappedValue = Linear(inner, c.dModel, bias: false)
        if hasBias {
            _relativeAttentionBias.wrappedValue = Embedding(embeddingCount: c.relativeBuckets, dimensions: c.heads)
        }
    }

    /// `queries [B, Tq, C]`, `keys [B, Tk, C]` → `[B, Tq, C]`, with `mask` an additive `[·, ·, Tq, Tk]`.
    func callAsFunction(_ queries: MLXArray, keys: MLXArray, mask: MLXArray) -> MLXArray {
        let (b, tq, tk) = (queries.dim(0), queries.dim(1), keys.dim(1))
        func heads4(_ t: MLXArray, _ len: Int) -> MLXArray { t.reshaped([b, len, heads, keyDim]).transposed(0, 2, 1, 3) }
        let qh = heads4(q(queries), tq), kh = heads4(k(keys), tk), vh = heads4(v(keys), tk)
        let attended = MLXFast.scaledDotProductAttention(queries: qh, keys: kh, values: vh, scale: 1, mask: mask)
        return o(attended.transposed(0, 2, 1, 3).reshaped([b, tq, heads * keyDim]))
    }

    /// The bucketed relative-position bias `[1, heads, qLen, kLen]`.
    func computeBias(_ qLen: Int, _ kLen: Int) -> MLXArray {
        var idx = [Int32](repeating: 0, count: qLen * kLen)
        for i in 0 ..< qLen {
            for j in 0 ..< kLen {
                idx[i * kLen + j] = Int32(Self.bucket(j - i, buckets: buckets, maxDistance: maxDistance, bidirectional: bidirectional))
            }
        }
        let indices = idx.withUnsafeBufferPointer { MLXArray($0, [qLen, kLen]) }
        return relativeAttentionBias!(indices).transposed(2, 0, 1).expandedDimensions(axis: 0)  // [1, heads, qLen, kLen]
    }

    /// The Mesh-TensorFlow relative-position bucketing.
    static func bucket(_ relativePosition: Int, buckets: Int, maxDistance: Int, bidirectional: Bool) -> Int {
        var result = 0
        var n = buckets
        var rel = relativePosition
        if bidirectional {
            n /= 2
            if rel > 0 { result += n }
            rel = abs(rel)
        } else {
            rel = -min(rel, 0)
        }
        let maxExact = n / 2
        if rel < maxExact { return result + rel }
        let large = maxExact + Int(log(Double(rel) / Double(maxExact))
            / log(Double(maxDistance) / Double(maxExact)) * Double(n - maxExact))
        return result + min(large, n - 1)
    }
}

/// The plain T5 v1.0 feed-forward: `wo(relu(wi(x)))`, no biases.
final class NFKChronosFeedForward: Module {
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    init(_ c: NFKMLXChronosConfiguration) {
        _wi.wrappedValue = Linear(c.dModel, c.dFF, bias: false)
        _wo.wrappedValue = Linear(c.dFF, c.dModel, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { wo(relu(wi(x))) }
}

/// `layer.0` of an encoder block or the decoder's self-attention sublayer.
final class NFKChronosSelfAttnSublayer: Module {
    @ModuleInfo(key: "SelfAttention") var attention: NFKChronosAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: NFKChronosLayerNorm
    init(_ c: NFKMLXChronosConfiguration, hasBias: Bool, bidirectional: Bool) {
        _attention.wrappedValue = NFKChronosAttention(c, hasBias: hasBias, bidirectional: bidirectional)
        _layerNorm.wrappedValue = NFKChronosLayerNorm(c.dModel, eps: c.layerNormEps)
    }
    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        let n = layerNorm(x)
        return x + attention(n, keys: n, mask: bias)
    }
}

/// The decoder's cross-attention sublayer (`EncDecAttention`).
final class NFKChronosCrossAttnSublayer: Module {
    @ModuleInfo(key: "EncDecAttention") var attention: NFKChronosAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: NFKChronosLayerNorm
    init(_ c: NFKMLXChronosConfiguration) {
        _attention.wrappedValue = NFKChronosAttention(c, hasBias: false, bidirectional: false)
        _layerNorm.wrappedValue = NFKChronosLayerNorm(c.dModel, eps: c.layerNormEps)
    }
    func callAsFunction(_ x: MLXArray, encoder: MLXArray, mask: MLXArray) -> MLXArray {
        x + attention(layerNorm(x), keys: encoder, mask: mask)
    }
}

/// The feed-forward sublayer.
final class NFKChronosFFSublayer: Module {
    @ModuleInfo(key: "DenseReluDense") var ff: NFKChronosFeedForward
    @ModuleInfo(key: "layer_norm") var layerNorm: NFKChronosLayerNorm
    init(_ c: NFKMLXChronosConfiguration) {
        _ff.wrappedValue = NFKChronosFeedForward(c)
        _layerNorm.wrappedValue = NFKChronosLayerNorm(c.dModel, eps: c.layerNormEps)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + ff(layerNorm(x)) }
}

/// An encoder block: `layer = [self-attention, feed-forward]`.
final class NFKChronosEncoderBlock: Module {
    @ModuleInfo(key: "layer") var layer: [Module]
    init(_ c: NFKMLXChronosConfiguration, hasBias: Bool) {
        _layer.wrappedValue = [NFKChronosSelfAttnSublayer(c, hasBias: hasBias, bidirectional: true), NFKChronosFFSublayer(c)]
    }
    var selfAttn: NFKChronosSelfAttnSublayer { layer[0] as! NFKChronosSelfAttnSublayer }
    func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        var h = (layer[0] as! NFKChronosSelfAttnSublayer)(x, bias: bias)
        h = (layer[1] as! NFKChronosFFSublayer)(h)
        return h
    }
}

/// A decoder block: `layer = [self-attention, cross-attention, feed-forward]`.
final class NFKChronosDecoderBlock: Module {
    @ModuleInfo(key: "layer") var layer: [Module]
    init(_ c: NFKMLXChronosConfiguration, hasBias: Bool) {
        _layer.wrappedValue = [NFKChronosSelfAttnSublayer(c, hasBias: hasBias, bidirectional: false),
                               NFKChronosCrossAttnSublayer(c), NFKChronosFFSublayer(c)]
    }
    var selfAttn: NFKChronosSelfAttnSublayer { layer[0] as! NFKChronosSelfAttnSublayer }
    func callAsFunction(_ x: MLXArray, bias: MLXArray, encoder: MLXArray, encoderMask: MLXArray) -> MLXArray {
        var h = (layer[0] as! NFKChronosSelfAttnSublayer)(x, bias: bias)
        h = (layer[1] as! NFKChronosCrossAttnSublayer)(h, encoder: encoder, mask: encoderMask)
        h = (layer[2] as! NFKChronosFFSublayer)(h)
        return h
    }
}

/// The T5 encoder stack.
final class NFKChronosEncoder: Module {
    @ModuleInfo(key: "block") var block: [NFKChronosEncoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalNorm: NFKChronosLayerNorm
    init(_ c: NFKMLXChronosConfiguration) {
        _block.wrappedValue = (0 ..< c.encoderLayers).map { NFKChronosEncoderBlock(c, hasBias: $0 == 0) }
        _finalNorm.wrappedValue = NFKChronosLayerNorm(c.dModel, eps: c.layerNormEps)
    }
    /// `embeds [B, S, C]`, `paddingAdditive [B, 1, 1, S]` → `[B, S, C]`.
    func callAsFunction(_ embeds: MLXArray, paddingAdditive: MLXArray) -> MLXArray {
        let s = embeds.dim(1)
        let bias = block[0].selfAttn.attention.computeBias(s, s) + paddingAdditive   // [B, heads, S, S]
        var h = embeds
        for b in block { h = b(h, bias: bias) }
        return finalNorm(h)
    }
}

/// The T5 decoder stack, run over a single token that cross-attends to the encoder output.
final class NFKChronosDecoder: Module {
    @ModuleInfo(key: "block") var block: [NFKChronosDecoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalNorm: NFKChronosLayerNorm
    init(_ c: NFKMLXChronosConfiguration) {
        _block.wrappedValue = (0 ..< c.decoderLayers).map { NFKChronosDecoderBlock(c, hasBias: $0 == 0) }
        _finalNorm.wrappedValue = NFKChronosLayerNorm(c.dModel, eps: c.layerNormEps)
    }
    /// `decEmbeds [B, 1, C]`, `encoder [B, S, C]`, `encoderMask [B, 1, 1, S]` → `[B, 1, C]`.
    func callAsFunction(_ decEmbeds: MLXArray, encoder: MLXArray, encoderMask: MLXArray) -> MLXArray {
        let tq = decEmbeds.dim(1)
        let bias = block[0].selfAttn.attention.computeBias(tq, tq)    // [1, heads, 1, 1] for one token
        var h = decEmbeds
        for b in block { h = b(h, bias: bias, encoder: encoder, encoderMask: encoderMask) }
        return finalNorm(h)
    }
}

/// The patch-embedding residual block: `output(relu(hidden(x))) + residual(x)`, all `Linear` with bias.
final class NFKChronosResidualBlock: Module {
    @ModuleInfo(key: "hidden_layer") var hidden: Linear
    @ModuleInfo(key: "output_layer") var output: Linear
    @ModuleInfo(key: "residual_layer") var residual: Linear
    init(inDim: Int, hidden: Int, outDim: Int) {
        _hidden.wrappedValue = Linear(inDim, hidden, bias: true)
        _output.wrappedValue = Linear(hidden, outDim, bias: true)
        _residual.wrappedValue = Linear(inDim, outDim, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(relu(hidden(x))) + residual(x) }
}

// MARK: - The model

/// The Chronos-Bolt network: instance-norm + patchify + input embedding + T5 encoder-decoder + a
/// quantile head.
public final class NFKMLXChronosNet: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "input_patch_embedding") var inputPatch: NFKChronosResidualBlock
    @ModuleInfo(key: "output_patch_embedding") var outputPatch: NFKChronosResidualBlock
    @ModuleInfo(key: "encoder") var encoder: NFKChronosEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKChronosDecoder
    let config: NFKMLXChronosConfiguration

    public init(_ config: NFKMLXChronosConfiguration = .init()) {
        self.config = config
        _shared.wrappedValue = Embedding(embeddingCount: 2, dimensions: config.dModel)   // [decoder_start, reg]
        _inputPatch.wrappedValue = NFKChronosResidualBlock(inDim: config.inputPatchSize * 2, hidden: config.dFF, outDim: config.dModel)
        _outputPatch.wrappedValue = NFKChronosResidualBlock(inDim: config.dModel, hidden: config.dFF,
                                                            outDim: config.quantiles.count * config.predictionLength)
        _encoder.wrappedValue = NFKChronosEncoder(config)
        _decoder.wrappedValue = NFKChronosDecoder(config)
    }

    /// `context [B, L]` → quantile forecast `[B, quantiles, predictionLength]`.
    public func callAsFunction(_ context: MLXArray) -> MLXArray {
        let b = context.dim(0)
        var series = context
        if series.dim(1) > config.contextLength { series = series[0..., (series.dim(1) - config.contextLength)...] }

        // Instance normalization (standardize each series over time).
        let loc = series.mean(axis: -1, keepDims: true)
        var scale = sqrt(((series - loc) * (series - loc)).mean(axis: -1, keepDims: true))
        scale = MLX.where(scale .== MLXArray(Float(0)), MLXArray(Float(1e-5)), scale)
        var scaled = (series - loc) / scale

        // Patchify (front-pad to a multiple of the patch size; a padded position has mask 0).
        let patch = config.inputPatchSize
        let remainder = scaled.dim(1) % patch
        var mask = MLXArray.ones([b, scaled.dim(1)])
        if remainder != 0 {
            let pad = patch - remainder
            scaled = MLX.padded(scaled, widths: [IntOrPair(0), IntOrPair((pad, 0))], mode: .constant)
            mask = MLX.padded(mask, widths: [IntOrPair(0), IntOrPair((pad, 0))], mode: .constant)
        }
        let nPatches = scaled.dim(1) / patch
        let patchedContext = scaled.reshaped([b, nPatches, patch])
        let patchedMask = mask.reshaped([b, nPatches, patch])
        let observed = patchedMask .> MLXArray(Float(0))
        let maskedContext = MLX.where(observed, patchedContext, MLXArray(Float(0)))
        let patched = concatenated([maskedContext, patchedMask], axis: -1)     // [B, nPatches, 2·patch]
        var attention = (patchedMask.sum(axis: -1) .> MLXArray(Float(0))).asType(.float32)   // [B, nPatches]

        // Input embedding + the appended REG token.
        var embeds = inputPatch(patched)                                       // [B, nPatches, dModel]
        let reg = shared(MLXArray([Int32(1)])).reshaped([1, 1, config.dModel])
        embeds = concatenated([embeds, broadcast(reg, to: [b, 1, config.dModel])], axis: 1)
        attention = concatenated([attention, MLXArray.ones([b, 1])], axis: 1)   // [B, nPatches+1]

        let paddingAdditive = ((1 - attention) * -1e9).reshaped([b, 1, 1, attention.dim(1)])
        let encoded = encoder(embeds, paddingAdditive: paddingAdditive)

        // One-token decoder cross-attending to the encoder output.
        let decStart = broadcast(shared(MLXArray([Int32(0)])).reshaped([1, 1, config.dModel]), to: [b, 1, config.dModel])
        let decoded = decoder(decStart, encoder: encoded, encoderMask: paddingAdditive)   // [B, 1, dModel]

        // Quantile head, then un-scale.
        let flat = outputPatch(decoded).reshaped([b, config.quantiles.count, config.predictionLength])
        return flat * scale.reshaped([b, 1, 1]) + loc.reshaped([b, 1, 1])
    }
}

// MARK: - Weight loading

extension NFKMLXChronosNet {
    /// Loads the released `model.safetensors` (`amazon/chronos-bolt-base`). The module keys mirror the
    /// checkpoint's HF T5 names, so nothing is transposed; the tied `embed_tokens` aliases and the unused
    /// `lm_head` / `quantiles` buffer are dropped.
    public func loadWeights(from url: URL) throws {
        let raw = try NFKMLXWeights.loadCheckpoint(url: url).arrays
        let mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            if key.contains("embed_tokens") || key == "lm_head.weight" || key == "quantiles" { return nil }
            return (key, value)
        }
        try NFKMLXWeights.apply(mapped, to: self)
    }
}

// MARK: - The forecasting object

private final class NFKChronosHolder: @unchecked Sendable {
    let net: NFKMLXChronosNet
    init(_ net: NFKMLXChronosNet) { self.net = net }
}

/// Chronos-Bolt time-series forecasting. Not an `NFKInferenceBackend` (a numeric series has no core
/// input/output key); it is an object, like `NFKMLXModernBERTReranker`.
///
/// - Since: InferKit 0.4.0
@objc(NFKMLXChronos)
public final class NFKMLXChronos: NSObject {
    private let holder: NFKChronosHolder
    private let config: NFKMLXChronosConfiguration

    init(net: NFKMLXChronosNet) { holder = NFKChronosHolder(net); config = net.config }

    /// The quantile levels the forecast rows correspond to (0.1 … 0.9).
    @objc public var quantileLevels: [NSNumber] { config.quantiles.map { NSNumber(value: $0) } }

    /// Forecasts `horizon` steps beyond `context`, returning one row per quantile level (each `horizon`
    /// long). `horizon` is capped at the model's trained prediction length (64).
    public func forecast(context: [Float], horizon: Int) -> [[Float]] {
        let steps = min(max(horizon, 1), config.predictionLength)
        let x = context.withUnsafeBufferPointer { MLXArray($0, [1, context.count]) }
        let preds = holder.net(x)                                              // [1, Q, predictionLength]
        eval(preds)
        let q = config.quantiles.count
        let flat = preds.reshaped([q, config.predictionLength]).asArray(Float.self)
        return (0 ..< q).map { row in (0 ..< steps).map { flat[row * config.predictionLength + $0] } }
    }

    /// The median (0.5-quantile) point forecast, for an Objective-C caller.
    @objc(medianForecastForContext:horizon:)
    public func medianForecast(context: [NSNumber], horizon: Int) -> [NSNumber] {
        let rows = forecast(context: context.map { $0.floatValue }, horizon: horizon)
        let median = rows[rows.count / 2]
        return median.map { NSNumber(value: $0) }
    }

    static func makeNet(_ config: NFKMLXChronosConfiguration = .init()) -> NFKMLXChronosNet { NFKMLXChronosNet(config) }

    /// Builds a forecaster from optional local weights — a nil `weightsURL` builds random weights.
    ///
    /// - Since: InferKit 0.4.0
    @objc(chronosWithWeightsURL:error:)
    public static func chronos(weightsURL: URL?) throws -> NFKMLXChronos {
        let net = makeNet()
        if let weightsURL { try net.loadWeights(from: weightsURL) }
        return NFKMLXChronos(net: net)
    }

    /// Downloads the checkpoint from Hugging Face, then builds. Blocking on the network; run off the
    /// render thread.
    @objc(chronosWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func chronos(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXChronos {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try chronos(weightsURL: url)
    }
}
