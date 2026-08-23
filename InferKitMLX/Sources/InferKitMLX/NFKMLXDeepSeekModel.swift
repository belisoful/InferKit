//
//  NFKMLXDeepSeekModel.swift
//  InferKitMLX
//
//  The DeepSeek V4 decoder (`DeepseekV4ForCausalLM`): Multi-head Latent Attention over a
//  mixture-of-experts feed-forward. A third architecture family beside the dense stack
//  (`NFKMLXLanguageNet`) and the gated-recurrence hybrid (`NFKMLXHybridLanguageNet`).
//
//  BUILT, NOT MEASURED — and verified more weakly than the hybrid, for a reason worth stating.
//  The released checkpoint is QUANTIZED: attention weights are fp8 with 128×128 block scales, and the
//  experts are 4-bit packed two per int8 byte with their own scales. A float module's parameters
//  therefore do NOT correspond one-to-one with the checkpoint's tensors, so the structural check has to
//  derive what each float parameter would look like quantized. That derivation is itself an assumption,
//  which is why the test asserts it reproduces the observed shapes rather than trusting it.
//
//  Implemented: the attention (low-rank queries, a shared latent key-value, grouped low-rank output,
//  the learned attention sink, rotary applied to the trailing channels only) and the mixture of experts
//  (square-root-softplus scoring, hash routing on the first layers, bias-shifted top-k after them, the
//  clamped SwiGLU, the shared expert).
//
//  NOT implemented, and named so they are known rather than overlooked: the per-layer key-value
//  compressor and the learned sparse indexer that selects which compressed positions to attend to,
//  YaRN rope scaling, the DSpark speculative blocks, the hash-clustering head, and the multi-token
//  prediction layers. Attention here is dense over the sliding window, which is what the uncompressed
//  layers do; a compressed layer needs the indexer to be correct.
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// The geometry of a DeepSeek V4 decoder.
public struct NFKMLXDeepSeekConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var vocabularySize: Int
    public var rmsEpsilon: Float

    // Multi-head latent attention.
    public var headCount: Int
    public var headDimensions: Int
    /// The channels of each head the rotary embedding turns; they sit at the END of the head.
    public var ropeHeadDimensions: Int
    public var queryLoRARank: Int
    public var outputLoRARank: Int
    /// The output projection is applied per group of heads.
    public var outputGroups: Int
    public var slidingWindow: Int
    public var ropeTheta: Float
    /// The rotary base the COMPRESSED positions use, which the config names separately.
    public var compressRopeTheta: Float
    /// How many parallel copies the hyper-connected residual stream carries.
    public var hyperConnectionCopies: Int
    /// Sinkhorn iterations used to make the copy-mixing matrix near doubly stochastic.
    public var sinkhornIterations: Int
    /// The floor added inside that normalization.
    public var hyperConnectionEpsilon: Float

    /// The sparse indexer's own head count, width, and how many compressed positions it keeps.
    public var indexHeadCount: Int
    public var indexHeadDimensions: Int
    public var indexTopK: Int

    // Mixture of experts.
    public var routedExpertCount: Int
    public var sharedExpertCount: Int
    public var activatedExpertCount: Int
    public var expertIntermediateSize: Int
    public var routeScale: Float
    /// Both branches of the SwiGLU are clamped to this before multiplying.
    public var swigluLimit: Float
    /// The first layers route by a precomputed token-to-expert table rather than by score.
    public var hashLayerCount: Int

    /// Per-layer key-value compression ratio; 0 means a layer attends over the window alone.
    public var compressRatios: [Int]

    public init(hiddenSize: Int = 4096, layerCount: Int = 43, vocabularySize: Int = 129_280,
                rmsEpsilon: Float = 1e-6, headCount: Int = 64, headDimensions: Int = 512,
                ropeHeadDimensions: Int = 64, queryLoRARank: Int = 1024, outputLoRARank: Int = 1024,
                outputGroups: Int = 8, slidingWindow: Int = 128, ropeTheta: Float = 10_000,
                compressRopeTheta: Float = 160_000, hyperConnectionCopies: Int = 4,
                sinkhornIterations: Int = 20, hyperConnectionEpsilon: Float = 1e-6,
                indexHeadCount: Int = 64,
                indexHeadDimensions: Int = 128, indexTopK: Int = 512,
                routedExpertCount: Int = 256, sharedExpertCount: Int = 1,
                activatedExpertCount: Int = 6, expertIntermediateSize: Int = 2048,
                routeScale: Float = 1.5, swigluLimit: Float = 10, hashLayerCount: Int = 3,
                compressRatios: [Int]? = nil) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.vocabularySize = vocabularySize
        self.rmsEpsilon = rmsEpsilon
        self.headCount = headCount
        self.headDimensions = headDimensions
        self.ropeHeadDimensions = ropeHeadDimensions
        self.queryLoRARank = queryLoRARank
        self.outputLoRARank = outputLoRARank
        self.outputGroups = outputGroups
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.compressRopeTheta = compressRopeTheta
        self.hyperConnectionCopies = hyperConnectionCopies
        self.sinkhornIterations = sinkhornIterations
        self.hyperConnectionEpsilon = hyperConnectionEpsilon
        self.indexHeadCount = indexHeadCount
        self.indexHeadDimensions = indexHeadDimensions
        self.indexTopK = indexTopK
        self.routedExpertCount = routedExpertCount
        self.sharedExpertCount = sharedExpertCount
        self.activatedExpertCount = activatedExpertCount
        self.expertIntermediateSize = expertIntermediateSize
        self.routeScale = routeScale
        self.swigluLimit = swigluLimit
        self.hashLayerCount = hashLayerCount
        // The released pattern alternates a light and a heavy ratio after two uncompressed layers.
        self.compressRatios = compressRatios ?? (0 ..< layerCount).map { index in
            index < 2 ? 0 : (index % 2 == 0 ? 4 : 128)
        }
    }

    /// The released `deepseek-ai/DeepSeek-V4-Flash-0731` decoder.
    public static let v4Flash = NFKMLXDeepSeekConfiguration()

    /// The released `deepseek-ai/DeepSeek-V4-Pro-0813` decoder.
    public static let v4Pro = NFKMLXDeepSeekConfiguration(
        hiddenSize: 7168, layerCount: 61, headCount: 128, queryLoRARank: 1536,
        routedExpertCount: 384, expertIntermediateSize: 3072)

    /// Whether a layer routes by the precomputed table rather than by score.
    func routesByHash(_ layer: Int) -> Bool { layer < hashLayerCount }
}

/// DeepSeek's rotary: ADJACENT channel pairs rotated as complex numbers.
///
/// The reference builds it with `view_as_complex`, so a pair is `(2i, 2i+1)` — the interleaved
/// convention. The Qwen and Gemma decoders here use rotate-half, where a pair is `(i, i + width/2)`.
/// Nothing in a tensor's shape distinguishes them and every value differs, so the convention is
/// written out rather than selected by a flag. `inverse` applies the conjugate, which the attention
/// output needs: the values share their latent with the keys, so the rotation has to be undone.
final class NFKDeepSeekRotary {
    private let cosines: MLXArray
    private let sines: MLXArray
    let width: Int

    init(width: Int, theta: Float, maximumPositions: Int = 4096) {
        self.width = width
        let pairs = width / 2
        let frequencies = (0 ..< pairs).map { 1 / powf(theta, Float(2 * $0) / Float(width)) }
        let positions = MLXArray((0 ..< maximumPositions).map(Float.init)).reshaped([maximumPositions, 1])
        let angles = positions * MLXArray(frequencies).reshaped([1, pairs])
        cosines = cos(angles)
        sines = sin(angles)
    }

    /// `x` is `[batch, heads, length, width]`; only its leading `width` channels are rotated.
    func callAsFunction(_ x: MLXArray, offset: Int = 0, inverse: Bool = false,
                        stride: Int = 1) -> MLXArray {
        let length = x.shape[2]
        let pairs = width / 2
        // A compressed position advances by a whole window, so its angle steps by `stride`.
        let rows = MLXArray((0 ..< length).map { Int32(offset + $0 * stride) })
        let c = take(cosines, rows, axis: 0).reshaped([1, 1, length, pairs])
        let s = take(inverse ? -sines : sines, rows, axis: 0).reshaped([1, 1, length, pairs])

        let paired = x.reshaped(x.shape.dropLast() + [pairs, 2])
        let even = paired[.ellipsis, 0]
        let odd = paired[.ellipsis, 1]
        let rotated = stacked([even * c - odd * s, even * s + odd * c], axis: -1)
        return rotated.reshaped(x.shape)
    }
}

/// One expert: a SwiGLU whose branches are clamped before they multiply.
final class NFKDeepSeekExpert: Module {
    @ModuleInfo(key: "w1") var gate: Linear
    @ModuleInfo(key: "w2") var down: Linear
    @ModuleInfo(key: "w3") var up: Linear
    let limit: Float

    init(hiddenSize: Int, intermediateSize: Int, limit: Float) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self.limit = limit
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gated = gate(x)
        var lifted = up(x)
        if limit > 0 {
            // The reference clamps the gate from above only and the up branch from both sides.
            gated = clip(gated, max: limit)
            lifted = clip(lifted, min: -limit, max: limit)
        }
        return down(silu(gated) * lifted)
    }
}

/// Expert routing.
///
/// The first layers take their experts from a table indexed by the TOKEN ID, so routing there does not
/// depend on the hidden state at all. The remaining layers score with a square-root softplus, shift by
/// a learned bias for selection only, and renormalize the unshifted scores of whatever was selected.
final class NFKDeepSeekGate: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?
    @ParameterInfo(key: "tid2eid") var tokenToExpert: MLXArray?

    let configuration: NFKMLXDeepSeekConfiguration
    let routesByHash: Bool

    init(_ c: NFKMLXDeepSeekConfiguration, layer: Int) {
        configuration = c
        routesByHash = c.routesByHash(layer)
        _weight.wrappedValue = MLXArray.zeros([c.routedExpertCount, c.hiddenSize])
        _bias.wrappedValue = routesByHash ? nil : MLXArray.zeros([c.routedExpertCount])
        _tokenToExpert.wrappedValue = routesByHash
            ? MLXArray.zeros([c.vocabularySize, c.activatedExpertCount], type: Int32.self) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, tokens: MLXArray?) -> (weights: MLXArray, indices: MLXArray) {
        let scores = sqrt(softplus(matmul(x, weight.transposed())))
        var indices: MLXArray
        if routesByHash, let tokenToExpert, let tokens {
            indices = take(tokenToExpert, tokens.reshaped([-1]), axis: 0)
        } else {
            // The bias steers selection; the weights come from the unshifted scores.
            let shifted = bias.map { scores + $0 } ?? scores
            indices = argPartition(-shifted, kth: configuration.activatedExpertCount - 1, axis: -1)[
                0..., 0 ..< configuration.activatedExpertCount]
        }
        var chosen = takeAlong(scores, indices, axis: -1)
        chosen = chosen / chosen.sum(axis: -1, keepDims: true)
        return (chosen * configuration.routeScale, indices)
    }
}

/// Multi-head latent attention.
///
/// Queries pass through a low rank (`wq_a` → norm → `wq_b`); keys and values are ONE shared latent
/// vector per position (`wkv` → norm), which is what makes the cache small; the output is projected per
/// group of heads through a second low rank (`wo_a` → `wo_b`). A learned per-head sink sits alongside
/// the attended positions.
final class NFKDeepSeekAttention: Module {
    @ModuleInfo(key: "wq_a") var queryDown: Linear
    @ModuleInfo(key: "wq_b") var queryUp: Linear
    @ModuleInfo(key: "wkv") var latentKeyValue: Linear
    @ModuleInfo(key: "wo_a") var outputDown: Linear
    @ModuleInfo(key: "wo_b") var outputUp: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "kv_norm") var latentNorm: RMSNorm
    @ParameterInfo(key: "attn_sink") var sink: MLXArray
    @ModuleInfo(key: "compressor") var compressor: NFKDeepSeekCompressor?
    @ModuleInfo(key: "indexer") var indexer: NFKDeepSeekIndexer?

    let configuration: NFKMLXDeepSeekConfiguration
    let rope: NFKDeepSeekRotary

    init(_ c: NFKMLXDeepSeekConfiguration, compressRatio: Int = 0) {
        configuration = c
        rope = NFKDeepSeekRotary(width: c.ropeHeadDimensions, theta: c.ropeTheta)
        _queryDown.wrappedValue = Linear(c.hiddenSize, c.queryLoRARank, bias: false)
        _queryUp.wrappedValue = Linear(c.queryLoRARank, c.headCount * c.headDimensions, bias: false)
        _latentKeyValue.wrappedValue = Linear(c.hiddenSize, c.headDimensions, bias: false)
        _outputDown.wrappedValue = Linear(c.headCount * c.headDimensions / c.outputGroups,
                                          c.outputGroups * c.outputLoRARank, bias: false)
        _outputUp.wrappedValue = Linear(c.outputGroups * c.outputLoRARank, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: c.queryLoRARank, eps: c.rmsEpsilon)
        _latentNorm.wrappedValue = RMSNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        _sink.wrappedValue = MLXArray.zeros([c.headCount])
        // A layer compresses only when its ratio says so, and only the light ratio carries an indexer;
        // the heavy one selects its compressed positions by a fixed stride instead.
        _compressor.wrappedValue = compressRatio > 0
            ? NFKDeepSeekCompressor(c, ratio: compressRatio, headDimensions: c.headDimensions) : nil
        _indexer.wrappedValue = compressRatio == 4 ? NFKDeepSeekIndexer(c, ratio: compressRatio) : nil
        super.init()
    }

    /// Dense attention over the window. A compressed layer additionally attends to compressed
    /// positions chosen by the indexer, which is not implemented here.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.shape[0], x.shape[1])

        var queries = queryUp(queryNorm(queryDown(x)))
            .reshaped([batch, length, c.headCount, c.headDimensions])
        // The reference normalizes each head before the rotary rather than after the projection.
        queries = queries * rsqrt((queries * queries).mean(axis: -1, keepDims: true) + c.rmsEpsilon)

        // One latent vector per position serves every head as both key and value.
        var latent = latentNorm(latentKeyValue(x)).reshaped([batch, length, 1, c.headDimensions])

        let turned = c.ropeHeadDimensions
        queries = queries.transposed(0, 2, 1, 3)
        latent = latent.transposed(0, 2, 1, 3)
        // Rotary turns the TRAILING channels of each head, not the leading ones.
        queries = concatenated([queries[.ellipsis, 0 ..< (c.headDimensions - turned)],
                                rope(queries[.ellipsis, (c.headDimensions - turned)...], offset: 0)],
                               axis: -1)
        latent = concatenated([latent[.ellipsis, 0 ..< (c.headDimensions - turned)],
                               rope(latent[.ellipsis, (c.headDimensions - turned)...], offset: 0)],
                              axis: -1)

        // Attention with a learned per-head SINK: an extra logit that competes with the keys and
        // carries no value, so it drains probability mass without contributing. Written out rather
        // than delegated, because the fused call has nowhere to put the extra column.
        var scores = matmul(queries, latent.transposed(0, 1, 3, 2)) / sqrt(Float(c.headDimensions))
        if let mask { scores = scores + mask }
        let sinkColumn = sink.reshaped([1, c.headCount, 1, 1])
        let peak = maximum(scores.max(axis: -1, keepDims: true), sinkColumn)
        let weights = exp(scores - peak)
        let total = weights.sum(axis: -1, keepDims: true) + exp(sinkColumn - peak)
        var attended = matmul(weights / total, latent)

        // The values share the latent with the keys, so their rotary component has to be UNDONE on
        // the way out: the reference applies the conjugate rotation to the output's trailing channels.
        attended = concatenated([attended[.ellipsis, 0 ..< (c.headDimensions - turned)],
                                 rope(attended[.ellipsis, (c.headDimensions - turned)...],
                                      inverse: true)],
                                axis: -1)

        // The output projection runs per group of heads: each group has its own slice of `wo_a`.
        let grouped = attended.transposed(0, 2, 1, 3)
            .reshaped([batch, length, c.outputGroups, c.headCount * c.headDimensions / c.outputGroups])
        let perGroup = outputDown.weight.reshaped([c.outputGroups, c.outputLoRARank, -1])
        var reduced = [MLXArray]()
        for group in 0 ..< c.outputGroups {
            reduced.append(matmul(grouped[0..., 0..., group], perGroup[group].transposed()))
        }
        return outputUp(concatenated(reduced, axis: -1))
    }
}

/// The mixture of experts: routed experts plus one shared expert every token passes through.
final class NFKDeepSeekMoE: Module {
    @ModuleInfo(key: "gate") var gate: NFKDeepSeekGate
    @ModuleInfo(key: "experts") var experts: [NFKDeepSeekExpert]
    @ModuleInfo(key: "shared_experts") var shared: NFKDeepSeekExpert

    let configuration: NFKMLXDeepSeekConfiguration

    init(_ c: NFKMLXDeepSeekConfiguration, layer: Int) {
        configuration = c
        _gate.wrappedValue = NFKDeepSeekGate(c, layer: layer)
        _experts.wrappedValue = (0 ..< c.routedExpertCount).map { _ in
            NFKDeepSeekExpert(hiddenSize: c.hiddenSize, intermediateSize: c.expertIntermediateSize,
                              limit: c.swigluLimit)
        }
        _shared.wrappedValue = NFKDeepSeekExpert(hiddenSize: c.hiddenSize,
                                                 intermediateSize: c.expertIntermediateSize,
                                                 limit: c.swigluLimit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, tokens: MLXArray?) -> MLXArray {
        let c = configuration
        let shape = x.shape
        let flat = x.reshaped([-1, c.hiddenSize])
        let (weights, indices) = gate(flat, tokens: tokens)
        eval(indices)

        // Every token passes through the shared expert; the routed ones add to it.
        var result = shared(flat)
        let chosen = indices.asArray(Int32.self)
        let scale = weights.asArray(Float.self)
        let slots = c.activatedExpertCount

        for token in 0 ..< flat.shape[0] {
            for slot in 0 ..< slots {
                let expert = Int(chosen[token * slots + slot])
                guard expert >= 0 && expert < experts.count else { continue }
                let contribution = experts[expert](flat[token ..< (token + 1)])
                result[token ..< (token + 1)] = result[token ..< (token + 1)]
                    + contribution * scale[token * slots + slot]
            }
        }
        return result.reshaped(shape)
    }
}

/// One decoder layer.
final class NFKDeepSeekBlock: Module {
    @ModuleInfo(key: "attn") var attention: NFKDeepSeekAttention
    @ModuleInfo(key: "ffn") var feedForward: NFKDeepSeekMoE
    @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "hc_attn") var attentionConnection: NFKDeepSeekHyperConnection
    @ModuleInfo(key: "hc_ffn") var feedForwardConnection: NFKDeepSeekHyperConnection

    init(_ c: NFKMLXDeepSeekConfiguration, layer: Int) {
        _attention.wrappedValue = NFKDeepSeekAttention(c, compressRatio: c.compressRatios[layer])
        _feedForward.wrappedValue = NFKDeepSeekMoE(c, layer: layer)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _feedForwardNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _attentionConnection.wrappedValue = NFKDeepSeekHyperConnection(
            copies: c.hyperConnectionCopies, hiddenSize: c.hiddenSize,
            iterations: c.sinkhornIterations, epsilon: c.hyperConnectionEpsilon,
            normEpsilon: c.rmsEpsilon)
        _feedForwardConnection.wrappedValue = NFKDeepSeekHyperConnection(
            copies: c.hyperConnectionCopies, hiddenSize: c.hiddenSize,
            iterations: c.sinkhornIterations, epsilon: c.hyperConnectionEpsilon,
            normEpsilon: c.rmsEpsilon)
        super.init()
    }

    /// `x` is `[batch, length, copies, hidden]` — the hyper-connected residual stream.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, tokens: MLXArray?) -> MLXArray {
        let (readA, writeA, combineA) = attentionConnection.weights(x)
        let reducedA = attentionConnection.reduce(x, read: readA)
        let attended = attention(attentionNorm(reducedA), mask: mask)
        let afterAttention = attentionConnection.expand(attended, residual: x,
                                                        write: writeA, combine: combineA)

        let (readF, writeF, combineF) = feedForwardConnection.weights(afterAttention)
        let reducedF = feedForwardConnection.reduce(afterAttention, read: readF)
        let lifted = feedForward(feedForwardNorm(reducedF), tokens: tokens)
        return feedForwardConnection.expand(lifted, residual: afterAttention,
                                            write: writeF, combine: combineF)
    }
}

/// A DeepSeek V4 decoder. See the file comment for what is and is not implemented.
public final class NFKMLXDeepSeekNet: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKDeepSeekBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "head") var head: Linear
    @ModuleInfo(key: "hc_head") var headConnection: NFKDeepSeekHyperHead

    let configuration: NFKMLXDeepSeekConfiguration

    init(_ c: NFKMLXDeepSeekConfiguration) {
        configuration = c
        _embed.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layerCount).map { NFKDeepSeekBlock(c, layer: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _head.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
        _headConnection.wrappedValue = NFKDeepSeekHyperHead(
            copies: c.hyperConnectionCopies, hiddenSize: c.hiddenSize,
            epsilon: c.hyperConnectionEpsilon, normEpsilon: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        // The head's own connection collapses the copies back to one stream.
        head(finalState(hiddenStates(tokens).last!))
    }

    /// The collapsed, normed stream the head reads — what the reference reports as its last state.
    func finalState(_ hidden: MLXArray) -> MLXArray {
        norm(headConnection.reduce(hidden))
    }

    /// The residual stream entering the stack and after every layer, for the isolation harness.
    ///
    /// Each state is the full `[batch, length, copies, hidden]` stream, because that is what the
    /// reference's `output_hidden_states` records — the copies are the residual here, not a detail.
    func hiddenStates(_ tokens: MLXArray) -> [MLXArray] {
        // The residual stream is `copies` parallel copies of the embedding from the outset.
        let embedded = embed(tokens).expandedDimensions(axis: 2)
        var hidden = repeated(embedded, count: configuration.hyperConnectionCopies, axis: 2)
        var states = [hidden]

        let length = tokens.shape[1]
        let mask: MLXArray? = length > 1 ? NFKMLXLanguageNet.causalMask(length, offset: 0) : nil
        for layer in layers {
            hidden = layer(hidden, mask: mask, tokens: tokens)
            states.append(hidden)
        }
        return states
    }
}

/// Building the decoder, reading a release's configuration, and describing what its checkpoint holds.
@objc(NFKMLXDeepSeek)
public final class NFKMLXDeepSeek: NSObject {

    static func makeNet(_ configuration: NFKMLXDeepSeekConfiguration) -> NFKMLXDeepSeekNet {
        NFKMLXDeepSeekNet(configuration)
    }

    /// Reads a released `config.json`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXDeepSeekConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        guard (json["model_type"] as? String) == "deepseek_v4" else {
            throw NFKMLXError.unsupportedConfiguration("this reads the deepseek_v4 decoder")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (json[key] as? NSNumber)?.floatValue ?? fallback }

        return NFKMLXDeepSeekConfiguration(
            hiddenSize: integer("hidden_size", 4096),
            layerCount: integer("num_hidden_layers", 43),
            vocabularySize: integer("vocab_size", 129_280),
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            headCount: integer("num_attention_heads", 64),
            headDimensions: integer("head_dim", 512),
            ropeHeadDimensions: integer("qk_rope_head_dim", 64),
            queryLoRARank: integer("q_lora_rank", 1024),
            outputLoRARank: integer("o_lora_rank", 1024),
            outputGroups: integer("o_groups", 8),
            slidingWindow: integer("sliding_window", 128),
            ropeTheta: real("rope_theta", 10_000),
            compressRopeTheta: real("compress_rope_theta", 160_000),
            hyperConnectionCopies: integer("hc_mult", 4),
            sinkhornIterations: integer("hc_sinkhorn_iters", 20),
            hyperConnectionEpsilon: real("hc_eps", 1e-6),
            indexHeadCount: integer("index_n_heads", 64),
            indexHeadDimensions: integer("index_head_dim", 128),
            indexTopK: integer("index_topk", 512),
            routedExpertCount: integer("n_routed_experts", 256),
            sharedExpertCount: integer("n_shared_experts", 1),
            activatedExpertCount: integer("num_experts_per_tok", 6),
            expertIntermediateSize: integer("moe_intermediate_size", 2048),
            routeScale: real("routed_scaling_factor", 1.5),
            swigluLimit: real("swiglu_limit", 10),
            hashLayerCount: integer("num_hash_layers", 3),
            compressRatios: json["compress_ratios"] as? [Int])
    }

    /// The module key a release tensor name maps to.
    ///
    /// @discussion The module's keys ARE the release's names except where MLX's key rules force a
    /// nesting: the release flattens the hyper-connection parameters (`hc_attn_fn`, `hc_head_scale`)
    /// where the module holds them as children of a connection module, and the release calls the
    /// latent norm `attn.norm` where the module says `kv_norm` — `norm` alone would collide with the
    /// block pattern MLX uses for the final norm.
    static func moduleKey(forRelease name: String) -> String {
        var key = name
        for site in ["attn", "ffn", "head"] {
            key = key.replacingOccurrences(of: "hc_\(site)_fn", with: "hc_\(site).fn")
            key = key.replacingOccurrences(of: "hc_\(site)_base", with: "hc_\(site).base")
            key = key.replacingOccurrences(of: "hc_\(site)_scale", with: "hc_\(site).scale")
        }
        return key.replacingOccurrences(of: ".attn.norm.", with: ".attn.kv_norm.")
    }

    /// Decodes a released shard's quantized tensors into the float parameters a module holds.
    ///
    /// @discussion The release stores a weight as bytes plus a companion `.scale` tensor, so a
    /// checkpoint's arrays are not parameters until they are decoded. A weight whose last axis is
    /// half its declared width is 4-bit packed; anything else with a scale is fp8. A tensor with no
    /// scale — the norms, the biases, the routing tables — passes through as it is.
    ///
    /// Loading a whole release is out of reach on one machine, so this takes the arrays a caller has
    /// already read rather than a directory: it decodes a shard, or one tensor, at whatever scale the
    /// caller can hold.
    ///
    /// - Parameters:
    ///   - arrays: a shard's contents, weights and `.scale` companions together.
    ///   - shapes: the float shape each parameter takes, which is what separates 4-bit from fp8.
    ///     `expectedParameters(for:)` produces it.
    ///
    /// Introduced in InferKit 0.1.0.
    public static func dequantized(_ arrays: [String: MLXArray],
                                   shapes: [String: [Int]]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (name, value) in arrays where !name.hasSuffix(".scale") {
            guard let scale = arrays[name.replacingOccurrences(of: ".weight", with: ".scale")] else {
                result[name] = value
                continue
            }
            let packedFourBit = shapes[name].map { $0.count == 2 && value.shape.last == $0[1] / 2 }
                ?? false
            result[name] = packedFourBit
                ? NFKMLXDeepSeekQuantization.dequantizeFP4(packedBytes: value, scaleBytes: scale)
                : NFKMLXDeepSeekQuantization.dequantizeFP8(bytes: value, scaleBytes: scale)
        }
        return result
    }

    /// Every parameter the decoder declares, as `name → shape`, WITHOUT building it.
    ///
    /// @discussion A released configuration has 43 layers of 257 experts; instantiating that at float
    /// precision would need hundreds of gigabytes, so the structural check enumerates the architecture
    /// analytically instead. The names are the checkpoint's own.
    static func expectedParameters(for c: NFKMLXDeepSeekConfiguration) -> [String: [Int]] {
        var shapes = [String: [Int]]()
        shapes["embed.weight"] = [c.vocabularySize, c.hiddenSize]
        shapes["hc_head_fn"] = [c.hyperConnectionCopies, c.hyperConnectionCopies * c.hiddenSize]
        shapes["hc_head_base"] = [c.hyperConnectionCopies]
        shapes["hc_head_scale"] = [1]
        shapes["norm.weight"] = [c.hiddenSize]
        shapes["head.weight"] = [c.vocabularySize, c.hiddenSize]

        for layer in 0 ..< c.layerCount {
            let a = "layers.\(layer).attn."
            shapes[a + "wq_a.weight"] = [c.queryLoRARank, c.hiddenSize]
            shapes[a + "wq_b.weight"] = [c.headCount * c.headDimensions, c.queryLoRARank]
            shapes[a + "wkv.weight"] = [c.headDimensions, c.hiddenSize]
            shapes[a + "wo_a.weight"] = [c.outputGroups * c.outputLoRARank,
                                         c.headCount * c.headDimensions / c.outputGroups]
            shapes[a + "wo_b.weight"] = [c.hiddenSize, c.outputGroups * c.outputLoRARank]
            shapes[a + "q_norm.weight"] = [c.queryLoRARank]
            shapes[a + "kv_norm.weight"] = [c.headDimensions]
            shapes[a + "attn_sink"] = [c.headCount]

            // A compressed layer carries a compressor; the light ratio adds an indexer with its own.
            let ratio = layer < c.compressRatios.count ? c.compressRatios[layer] : 0
            if ratio > 0 {
                let width = (ratio == 4 ? 2 : 1) * c.headDimensions
                shapes[a + "compressor.wkv.weight"] = [width, c.hiddenSize]
                shapes[a + "compressor.wgate.weight"] = [width, c.hiddenSize]
                shapes[a + "compressor.norm.weight"] = [c.headDimensions]
                shapes[a + "compressor.ape"] = [ratio, width]
            }
            if ratio == 4 {
                let i = a + "indexer."
                shapes[i + "wq_b.weight"] = [c.indexHeadCount * c.indexHeadDimensions, c.queryLoRARank]
                shapes[i + "weights_proj.weight"] = [c.indexHeadCount, c.hiddenSize]
                let indexWidth = 2 * c.indexHeadDimensions
                shapes[i + "compressor.wkv.weight"] = [indexWidth, c.hiddenSize]
                shapes[i + "compressor.wgate.weight"] = [indexWidth, c.hiddenSize]
                shapes[i + "compressor.norm.weight"] = [c.indexHeadDimensions]
                shapes[i + "compressor.ape"] = [4, indexWidth]
            }

            shapes["layers.\(layer).attn_norm.weight"] = [c.hiddenSize]
            shapes["layers.\(layer).ffn_norm.weight"] = [c.hiddenSize]

            // Hyper-connections: two per block, plus the head's own at the top level.
            let mix = (2 + c.hyperConnectionCopies) * c.hyperConnectionCopies
            let mixWidth = c.hyperConnectionCopies * c.hiddenSize
            for part in ["attn", "ffn"] {
                shapes["layers.\(layer).hc_\(part)_fn"] = [mix, mixWidth]
                shapes["layers.\(layer).hc_\(part)_base"] = [mix]
                shapes["layers.\(layer).hc_\(part)_scale"] = [3]
            }

            let f = "layers.\(layer).ffn."
            shapes[f + "gate.weight"] = [c.routedExpertCount, c.hiddenSize]
            if c.routesByHash(layer) {
                shapes[f + "gate.tid2eid"] = [c.vocabularySize, c.activatedExpertCount]
            } else {
                shapes[f + "gate.bias"] = [c.routedExpertCount]
            }
            for expert in 0 ..< c.routedExpertCount {
                let e = f + "experts.\(expert)."
                shapes[e + "w1.weight"] = [c.expertIntermediateSize, c.hiddenSize]
                shapes[e + "w2.weight"] = [c.hiddenSize, c.expertIntermediateSize]
                shapes[e + "w3.weight"] = [c.expertIntermediateSize, c.hiddenSize]
            }
            shapes[f + "shared_experts.w1.weight"] = [c.expertIntermediateSize, c.hiddenSize]
            shapes[f + "shared_experts.w2.weight"] = [c.hiddenSize, c.expertIntermediateSize]
            shapes[f + "shared_experts.w3.weight"] = [c.expertIntermediateSize, c.hiddenSize]
        }
        return shapes
    }

    /// The shape a float parameter takes in the released checkpoint.
    ///
    /// @discussion The release is quantized, so a float weight is not stored as itself. Attention and
    /// shared-expert weights are fp8, one byte a value, keeping their shape. A routed expert is 4-bit
    /// packed two to a byte, so its last axis halves. This derivation is an assumption about the
    /// release's layout, and the structural test checks it against the observed shapes rather than
    /// trusting it.
    static func quantizedShape(of name: String, float: [Int]) -> [Int] {
        let packedFourBit = name.contains(".experts.") && !name.contains("shared_experts")
        guard packedFourBit, var shape = Optional(float), shape.count == 2 else { return float }
        shape[1] /= 2
        return shape
    }
}

/// The final collapse of the hyper-connection copies, before the shared norm and the head.
///
/// This is NOT the block connection at a smaller size: it predicts only the read weights — one
/// sigmoid gate per copy — with no write weights, no combination matrix, and no Sinkhorn, so its
/// `fn` is `[copies, copies * hidden]` where a block connection's is `[(2 + copies) * copies, ...]`.
/// The released `hc_head_fn` has exactly that shape, which is what caught the head being built as
/// the wrong class: the structural check compared DECLARED shapes against the release and passed,
/// while the module quietly built the bigger one.
final class NFKDeepSeekHyperHead: Module {
    @ParameterInfo(key: "fn") var projection: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    let epsilon: Float
    let normEpsilon: Float

    init(copies: Int, hiddenSize: Int, epsilon: Float, normEpsilon: Float) {
        self.epsilon = epsilon
        self.normEpsilon = normEpsilon
        _projection.wrappedValue = MLXArray.zeros([copies, copies * hiddenSize])
        _base.wrappedValue = MLXArray.zeros([copies])
        _scale.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    /// Collapses the copies into one stream, each weighted by its predicted gate.
    func reduce(_ x: MLXArray) -> MLXArray {
        let shape = x.shape                                  // [batch, length, copies, hidden]
        let flat = x.reshaped(shape[0], shape[1], shape[2] * shape[3])
        let inverse = rsqrt((flat * flat).mean(axis: -1, keepDims: true) + normEpsilon)
        let mixes = matmul(flat, projection.transposed()) * inverse
        let read = sigmoid(mixes * scale + base) + epsilon
        return (read.expandedDimensions(axis: -1) * x).sum(axis: 2)
    }
}

/// Compresses the key-value stream by pooling `compressRatio` consecutive positions into one.
///
/// Each position contributes a value (`wkv`) and a score (`wgate`); the scores are softmaxed across
/// the window and used to pool the values. `ape` is a learned per-slot bias, so a position's weight
/// depends on WHERE it sits in the window as well as on its content.
///
/// At ratio 4 the reference also compresses an OVERLAPPING window: the projections are twice as wide,
/// the second half pooling the window as given and the first half pooling it shifted back by one
/// window, so a boundary does not fall between two positions that belong together.
///
/// Prefill only. The reference's incremental decode keeps rolling state buffers, which a single
/// forward pass over a prompt never enters.
final class NFKDeepSeekCompressor: Module {
    @ModuleInfo(key: "wkv") var valueProjection: Linear
    @ModuleInfo(key: "wgate") var scoreProjection: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo(key: "ape") var positionBias: MLXArray

    let ratio: Int
    let headDimensions: Int
    let ropeHeadDimensions: Int
    let overlaps: Bool
    let rope: NFKDeepSeekRotary

    init(_ c: NFKMLXDeepSeekConfiguration, ratio: Int, headDimensions: Int) {
        self.ratio = ratio
        self.headDimensions = headDimensions
        ropeHeadDimensions = c.ropeHeadDimensions
        overlaps = ratio == 4
        let width = (overlaps ? 2 : 1) * headDimensions
        _valueProjection.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        _scoreProjection.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: headDimensions, eps: c.rmsEpsilon)
        _positionBias.wrappedValue = MLXArray.zeros([ratio, width])
        rope = NFKDeepSeekRotary(width: c.ropeHeadDimensions, theta: c.compressRopeTheta)
        super.init()
    }

    /// Pools `x` `[batch, length, hidden]` into `[batch, length / ratio, headDimensions]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray? {
        let (batch, length) = (x.shape[0], x.shape[1])
        let windows = length / ratio
        guard windows > 0 else { return nil }
        let kept = windows * ratio

        let width = (overlaps ? 2 : 1) * headDimensions
        var values = valueProjection(x)[0..., 0 ..< kept]
            .reshaped([batch, windows, ratio, width])
        var scores = scoreProjection(x)[0..., 0 ..< kept]
            .reshaped([batch, windows, ratio, width]) + positionBias

        if overlaps {
            // The second half of the width pools this window; the first half pools the PREVIOUS one,
            // so a window boundary is covered from both sides. The first window has no predecessor,
            // which the reference fills with zero values and −inf scores so they contribute nothing.
            values = shifted(values, taking: 0 ..< headDimensions, fill: 0)
            scores = shifted(scores, taking: 0 ..< headDimensions, fill: -Float.greatestFiniteMagnitude / 4)
        }

        // Softmax runs ACROSS the window, so the positions in a window compete with one another.
        let pooled = (values * softmax(scores, axis: 2)).sum(axis: 2)
        var compressed = norm(pooled)

        // The compressed positions carry a rotary too, at the window's own stride.
        let turned = ropeHeadDimensions
        let head = compressed[.ellipsis, 0 ..< (headDimensions - turned)]
        let tail = compressed[.ellipsis, (headDimensions - turned)...]
            .reshaped([batch, 1, windows, turned])
        compressed = concatenated([head, rope(tail, stride: ratio).reshaped([batch, windows, turned])],
                                  axis: -1)
        return compressed
    }

    /// Builds the overlapping arrangement: window `w` takes its own second half and window `w-1`'s
    /// first half, giving `2 · ratio` contributors of `headDimensions` each.
    private func shifted(_ tensor: MLXArray, taking range: Range<Int>, fill: Float) -> MLXArray {
        let (batch, windows) = (tensor.shape[0], tensor.shape[1])
        let own = tensor[.ellipsis, headDimensions...]
        let previous = tensor[.ellipsis, range]
        let padding = MLXArray.zeros([batch, 1, ratio, headDimensions]) + fill
        let lagged = concatenated([padding, previous[0..., 0 ..< (windows - 1)]], axis: 1)
        return concatenated([lagged, own], axis: 2)
    }
}

/// Chooses which compressed positions a query should attend to.
///
/// It runs its own compressor over the hidden states to build a scoring cache, projects the layer's
/// low-rank query into an index space, and scores every compressed position against it. A learned
/// per-head weight combines the heads, and the highest `indexTopK` positions are selected.
///
/// The reference additionally applies a **Hadamard rotation** to both the query and the compressed
/// keys before simulating fp4 quantization. That rotation is orthogonal and SHARED by both sides of
/// the dot product, so it cancels: it exists to spread information across channels for quantization,
/// not to change the score. Omitting it and the quantization simulation together yields the
/// unquantized scores, which is the same ranking the reference approximates.
final class NFKDeepSeekIndexer: Module {
    @ModuleInfo(key: "wq_b") var queryUp: Linear
    @ModuleInfo(key: "weights_proj") var headWeights: Linear
    @ModuleInfo(key: "compressor") var compressor: NFKDeepSeekCompressor

    let heads: Int
    let headDimensions: Int
    let ropeHeadDimensions: Int
    let topK: Int
    let ratio: Int
    let rope: NFKDeepSeekRotary

    init(_ c: NFKMLXDeepSeekConfiguration, ratio: Int) {
        heads = c.indexHeadCount
        headDimensions = c.indexHeadDimensions
        ropeHeadDimensions = c.ropeHeadDimensions
        topK = c.indexTopK
        self.ratio = ratio
        rope = NFKDeepSeekRotary(width: c.ropeHeadDimensions, theta: c.ropeTheta)
        _queryUp.wrappedValue = Linear(c.queryLoRARank, c.indexHeadCount * c.indexHeadDimensions,
                                       bias: false)
        _headWeights.wrappedValue = Linear(c.hiddenSize, c.indexHeadCount, bias: false)
        _compressor.wrappedValue = NFKDeepSeekCompressor(c, ratio: ratio,
                                                         headDimensions: c.indexHeadDimensions)
        super.init()
    }

    /// - Parameter lowRankQuery: the layer's `wq_a` output, which this shares rather than recomputing.
    /// - Returns: the chosen compressed positions per query, `[batch, length, k]`, or nil when the
    ///   sequence is too short to compress.
    func callAsFunction(_ x: MLXArray, lowRankQuery: MLXArray) -> MLXArray? {
        guard let compressed = compressor(x) else { return nil }
        let (batch, length) = (x.shape[0], x.shape[1])
        let windows = compressed.shape[1]

        var queries = queryUp(lowRankQuery).reshaped([batch, length, heads, headDimensions])
        let turned = ropeHeadDimensions
        let head = queries[.ellipsis, 0 ..< (headDimensions - turned)]
        let tail = queries[.ellipsis, (headDimensions - turned)...]
            .transposed(0, 2, 1, 3)
        queries = concatenated([head, rope(tail).transposed(0, 2, 1, 3)], axis: -1)

        // Every head scores every compressed position; a learned weight then combines the heads.
        let scores = einsum("bshd,btd->bsht", queries, compressed)
        let weights = headWeights(x) * (1 / sqrt(Float(headDimensions)) / sqrt(Float(heads)))
        var combined = (relu(scores) * weights.expandedDimensions(axis: -1)).sum(axis: 2)

        // A query may only see windows that are already complete at its own position.
        let positions = MLXArray((0 ..< length).map { Int32(($0 + 1) / ratio) }).reshaped([length, 1])
        let windowIndex = MLXArray((0 ..< windows).map(Int32.init)).reshaped([1, windows])
        let visible = windowIndex .< positions
        combined = MLX.where(visible, combined, MLXArray(-Float.greatestFiniteMagnitude / 4))
            .reshaped([batch, length, windows])

        let keep = min(topK, windows)
        let ranked = argPartition(-combined, kth: keep - 1, axis: -1)[0..., 0..., 0 ..< keep]

        // A position with nothing complete yet selects nothing; the reference marks those -1. The
        // gather is `takeAlong`, which picks per position — plain `take` would broadcast the whole
        // window axis against the ranking and produce a rank-5 result.
        let visiblePer = MLX.where(visible, MLXArray(Int32(1)), MLXArray(Int32(0)))
            .reshaped([1, length, windows])
        let allowed = takeAlong(broadcast(visiblePer, to: [batch, length, windows]), ranked, axis: -1)
        return MLX.where(allowed .> 0, ranked, MLXArray(Int32(-1)))
    }
}

/// Hyper-Connections: the residual stream is `hcMultiplier` parallel copies rather than one.
///
/// A block reduces the copies to a single stream before its attention or feed-forward (`reduce`) and
/// expands the result back afterwards (`expand`), with the mixing weights PREDICTED per position from
/// the copies themselves. `hc_*_fn` projects the flattened, RMS-normalized copies to
/// `(2 + hc) · hc` numbers, which split into a read weight per copy, a write weight per copy, and a
/// copy-to-copy combination matrix; the matrix is softmaxed and then Sinkhorn-normalized so it is
/// close to doubly stochastic, which keeps the copies from collapsing into one another.
///
/// This is what the top-level `hc_head_*` parameters collapse at the end of the stack. An earlier note
/// here called them a hash-clustering head; that was wrong, and it mattered: the residual stream's
/// rank is different from an ordinary decoder's.
final class NFKDeepSeekHyperConnection: Module {
    @ParameterInfo(key: "fn") var projection: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    let copies: Int
    let epsilon: Float
    let normEpsilon: Float
    let iterations: Int

    init(copies: Int, hiddenSize: Int, iterations: Int, epsilon: Float, normEpsilon: Float) {
        self.copies = copies
        self.iterations = iterations
        self.epsilon = epsilon
        self.normEpsilon = normEpsilon
        let mix = (2 + copies) * copies
        _projection.wrappedValue = MLXArray.zeros([mix, copies * hiddenSize])
        _base.wrappedValue = MLXArray.zeros([mix])
        _scale.wrappedValue = MLXArray.ones([3])
        super.init()
    }

    /// Splits the predicted mixes into read weights, write weights, and the combination matrix.
    func weights(_ x: MLXArray) -> (read: MLXArray, write: MLXArray, combine: MLXArray) {
        let shape = x.shape                                  // [batch, length, copies, hidden]
        let flat = x.reshaped(shape[0], shape[1], shape[2] * shape[3])
        let inverse = rsqrt((flat * flat).mean(axis: -1, keepDims: true) + normEpsilon)
        let mixes = matmul(flat, projection.transposed()) * inverse

        let read = sigmoid(mixes[0..., 0..., 0 ..< copies] * scale[0]
                           + base[0 ..< copies]) + epsilon
        let write = 2 * sigmoid(mixes[0..., 0..., copies ..< (2 * copies)] * scale[1]
                                + base[copies ..< (2 * copies)])

        var combine = mixes[0..., 0..., (2 * copies)...] * scale[2] + base[(2 * copies)...]
        combine = combine.reshaped(shape[0], shape[1], copies, copies)
        combine = softmax(combine, axis: -1) + epsilon
        combine = combine / (combine.sum(axis: -2, keepDims: true) + epsilon)
        // Sinkhorn: alternate row and column normalization so the matrix approaches doubly stochastic.
        for _ in 1 ..< max(iterations, 1) {
            combine = combine / (combine.sum(axis: -1, keepDims: true) + epsilon)
            combine = combine / (combine.sum(axis: -2, keepDims: true) + epsilon)
        }
        return (read, write, combine)
    }

    /// Collapses the copies into one stream, weighting each by its read weight.
    func reduce(_ x: MLXArray, read: MLXArray) -> MLXArray {
        (read.expandedDimensions(axis: -1) * x).sum(axis: 2)
    }

    /// Writes a single stream back across the copies, adding the combined previous copies.
    func expand(_ x: MLXArray, residual: MLXArray, write: MLXArray, combine: MLXArray) -> MLXArray {
        let written = write.expandedDimensions(axis: -1) * x.expandedDimensions(axis: -2)
        let mixed = (combine.expandedDimensions(axis: -1)
                     * residual.expandedDimensions(axis: -3)).sum(axis: 2)
        return written + mixed
    }
}
