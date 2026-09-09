//
//  NFKMLXGemma3.swift
//  InferKitMLX
//
//  The Gemma 3 text decoder (`gemma3_text`): the causal language model the 270M, 1B, 4B, 12B, and 27B
//  releases run, and the backbone EmbeddingGemma reads bidirectionally. One implementation serves both;
//  the encoder is this decoder with the causal mask removed, which is what the release's
//  `use_bidirectional_attention` flag means.
//
//  Gemma 3 differs from Gemma 4 in ways a shape cannot show: it normalizes with `x · (1 + w)` where
//  Gemma 4 scales by the weight directly, a full-attention head turns its whole width where Gemma 4
//  turns a fraction, and it carries no per-layer input embeddings. Five of every six layers attend
//  over a sliding window at a local rotary base; the sixth attends over everything at the global base,
//  which the larger sizes stretch by a linear rotary scaling factor. The multimodal releases pair this
//  decoder with a SigLIP vision tower (`NFKMLXGemma3Vision.swift`).
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

/// The geometry of a Gemma 3 text decoder.
public struct NFKMLXGemma3Configuration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    /// Rotary base for the full-attention layers.
    public var ropeTheta: Float
    /// Rotary base for the sliding-window layers, which the config states separately.
    public var ropeLocalTheta: Float
    /// The linear rotary scaling the FULL-attention layers apply (`rope_scaling.factor`); 1 is none.
    /// The 4B and larger releases stretch their global layers by 8 to reach a 128k context; the sliding
    /// layers are never scaled.
    public var ropeScalingFactor: Float
    public var slidingWindow: Int
    /// Which layers attend over a window and which over everything.
    public var layerTypes: [NFKMLXGemmaAttentionKind]
    /// The queries are scaled by `queryPreAttnScalar ** -0.5` before attention.
    public var queryPreAttnScalar: Float
    public var rmsEpsilon: Float
    /// The logits are squashed through `tanh` at this scale; 0 disables it (every Gemma 3 release).
    public var finalLogitSoftcap: Float
    /// The attention scores are squashed through `tanh` at this scale; 0 disables it (every Gemma 3
    /// release; Gemma 2 sets it).
    public var attentionLogitSoftcap: Float
    /// Whether every layer attends in both directions (EmbeddingGemma) rather than causally.
    public var isBidirectional: Bool

    public init(hiddenSize: Int = 640, layerCount: Int = 18, headCount: Int = 4,
                keyValueHeadCount: Int = 1, headDimensions: Int = 256, intermediateSize: Int = 2048,
                vocabularySize: Int = 262_144, ropeTheta: Float = 1_000_000, ropeLocalTheta: Float = 10_000,
                ropeScalingFactor: Float = 1, slidingWindow: Int = 512,
                layerTypes: [NFKMLXGemmaAttentionKind]? = nil, slidingWindowPattern: Int = 6,
                queryPreAttnScalar: Float = 256, rmsEpsilon: Float = 1e-6,
                finalLogitSoftcap: Float = 0, attentionLogitSoftcap: Float = 0,
                isBidirectional: Bool = false) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.ropeLocalTheta = ropeLocalTheta
        self.ropeScalingFactor = ropeScalingFactor
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes ?? Self.layerTypes(count: layerCount, pattern: slidingWindowPattern)
        self.queryPreAttnScalar = queryPreAttnScalar
        self.rmsEpsilon = rmsEpsilon
        self.finalLogitSoftcap = finalLogitSoftcap
        self.attentionLogitSoftcap = attentionLogitSoftcap
        self.isBidirectional = isBidirectional
    }

    /// The layer kinds a `sliding_window_pattern` derives: every `pattern`th layer (counting from 1) is
    /// full attention and the rest slide, the reference's `bool((i + 1) % pattern)` rule.
    public static func layerTypes(count: Int, pattern: Int) -> [NFKMLXGemmaAttentionKind] {
        (0 ..< count).map { (($0 + 1) % Swift.max(pattern, 1)) == 0 ? .full : .sliding }
    }

    /// The released `gemma-3-270m` text decoder.
    public static let gemma3_270M = NFKMLXGemma3Configuration()

    /// The released `gemma-3-1b` text decoder.
    public static let gemma3_1B = NFKMLXGemma3Configuration(
        hiddenSize: 1152, layerCount: 26, headCount: 4, keyValueHeadCount: 1, headDimensions: 256,
        intermediateSize: 6912)

    /// The released `gemma-3-4b` text decoder, whose global layers stretch their rotary by 8.
    public static let gemma3_4B = NFKMLXGemma3Configuration(
        hiddenSize: 2560, layerCount: 34, headCount: 8, keyValueHeadCount: 4, headDimensions: 256,
        intermediateSize: 10240, vocabularySize: 262_208, ropeScalingFactor: 8, slidingWindow: 1024)

    /// A small configuration that runs with random weights, for tests: a 4-position window over a
    /// sliding/sliding/full pattern, a rotary scaling on the full layer, and an attention soft-cap, so
    /// every mechanism the released sizes use is exercised at a size an oracle can run.
    public static let tiny = NFKMLXGemma3Configuration(
        hiddenSize: 64, layerCount: 3, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
        intermediateSize: 96, vocabularySize: 131, ropeScalingFactor: 8, slidingWindow: 4,
        slidingWindowPattern: 3, queryPreAttnScalar: 16, attentionLogitSoftcap: 50)

    /// Whether the layer at `index` attends over everything rather than a window.
    func isFullAttention(layer index: Int) -> Bool { layerTypes[index] == .full }
}

/// Gemma 3's normalization: `x · (1 + w)`, the weight initialized to zero. This is the difference from
/// Gemma 4's `x · w` that first broke a Gemma port here, so it is written out rather than shared.
final class NFKGemma3Norm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        epsilon = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = x * rsqrt((x * x).mean(axis: -1, keepDims: true) + epsilon)
        return normalized * (1 + weight)
    }
}

/// Gemma 3 attention: grouped queries, per-head query and key normalization before the rotary, and a
/// window that most layers use. The mask carries the causality (or its absence), so one attention
/// serves the decoder and the bidirectional encoder.
final class NFKGemma3Attention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: NFKGemma3Norm
    @ModuleInfo(key: "k_norm") var keyNorm: NFKGemma3Norm

    let heads: Int
    let keyValueHeads: Int
    let headDimensions: Int
    let scale: Float
    let softcap: Float
    let ropeBase: Float
    /// The rotary position scale, `1 / factor`: a linearly scaled rotary reads position `p` as `p / factor`.
    let ropeScale: Float

    init(_ c: NFKMLXGemma3Configuration, fullAttention: Bool) {
        heads = c.headCount
        keyValueHeads = c.keyValueHeadCount
        headDimensions = c.headDimensions
        scale = pow(c.queryPreAttnScalar, -0.5)
        softcap = c.attentionLogitSoftcap
        ropeBase = fullAttention ? c.ropeTheta : c.ropeLocalTheta
        ropeScale = fullAttention ? 1 / Swift.max(c.ropeScalingFactor, 1) : 1

        _queryProjection.wrappedValue = Linear(c.hiddenSize, c.headCount * c.headDimensions, bias: false)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions, bias: false)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions, bias: false)
        _outputProjection.wrappedValue = Linear(c.headCount * c.headDimensions, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = NFKGemma3Norm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        _keyNorm.wrappedValue = NFKGemma3Norm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        super.init()
    }

    /// - Parameters:
    ///   - mask: an additive mask `[length, keys]` over the keys this pass attends to, or nil for every key.
    ///   - cache: the layer's key-value cache; the rotary offset is the cache's position count.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: NFKMLXKeyValueCache?, layer: Int) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])

        var queries = queryProjection(x).reshaped([batch, length, heads, headDimensions])
        var keys = keyProjection(x).reshaped([batch, length, keyValueHeads, headDimensions])
        var values = valueProjection(x).reshaped([batch, length, keyValueHeads, headDimensions])

        // Gemma 3 normalizes each head BEFORE the rotary, over the head width.
        queries = queryNorm(queries)
        keys = keyNorm(keys)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = MLXFast.RoPE(queries, dimensions: headDimensions, traditional: false, base: ropeBase,
                               scale: ropeScale, offset: offset)
        keys = MLXFast.RoPE(keys, dimensions: headDimensions, traditional: false, base: ropeBase,
                            scale: ropeScale, offset: offset)

        if let cache {
            (keys, values) = cache.update(layer: layer, keys: keys, values: values)
        }

        // The masks are built float32, and a `.checkpoint`-precision load makes this a bf16 module; the
        // fused attention refuses a mask that does not promote to its own type, so the mask takes the
        // queries' dtype.
        let typedMask = mask.map { $0.asType(queries.dtype) }
        let attended: MLXArray
        if softcap > 0 {
            attended = softcappedAttention(queries: queries, keys: keys, values: values, mask: typedMask)
        } else {
            attended = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: typedMask)
        }
        return outputProjection(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDimensions]))
    }

    /// Attention with the softmax written out, for the `tanh` soft-cap the fused kernel has no slot for.
    private func softcappedAttention(queries: MLXArray, keys: MLXArray, values: MLXArray,
                                     mask: MLXArray?) -> MLXArray {
        let batch = queries.dim(0), keyCount = keys.dim(2)
        let groups = heads / keyValueHeads
        func spread(_ x: MLXArray) -> MLXArray {
            guard groups > 1 else { return x }
            return broadcast(x.expandedDimensions(axis: 2), to: [batch, keyValueHeads, groups, keyCount, headDimensions])
                .reshaped([batch, heads, keyCount, headDimensions])
        }
        var scores = matmul(queries, spread(keys).transposed(0, 1, 3, 2)) * scale
        scores = tanh(scores / softcap) * softcap
        if let mask { scores = scores + mask }
        return matmul(softmax(scores, axis: -1, precise: true), spread(values))
    }
}

/// Gemma 3's GeGLU feed-forward: a gate through the tanh-approximate GELU, an up projection, their
/// product projected back down.
final class NFKGemma3FeedForward: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: NFKMLXGemma3Configuration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geluApproximate(gate(x)) * up(x)) }
}

/// One Gemma 3 block: the sandwich normalization, a norm before AND after each of attention and the
/// feed-forward, each pair added back to the block's input.
final class NFKGemma3Block: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKGemma3Attention
    @ModuleInfo(key: "mlp") var feedForward: NFKGemma3FeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKGemma3Norm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: NFKGemma3Norm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: NFKGemma3Norm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: NFKGemma3Norm

    let fullAttention: Bool

    init(_ c: NFKMLXGemma3Configuration, fullAttention: Bool) {
        self.fullAttention = fullAttention
        _attention.wrappedValue = NFKGemma3Attention(c, fullAttention: fullAttention)
        _feedForward.wrappedValue = NFKGemma3FeedForward(c)
        _inputNorm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postAttentionNorm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _preFeedForwardNorm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postFeedForwardNorm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: NFKMLXKeyValueCache?, layer: Int) -> MLXArray {
        let attended = x + postAttentionNorm(attention(inputNorm(x), mask: mask, cache: cache, layer: layer))
        return attended + postFeedForwardNorm(feedForward(preFeedForwardNorm(attended)))
    }
}

/// The key-value caches a Gemma 3 generation keeps: one unbounded for the full-attention layers and one
/// bounded to the sliding window for the rest, the reference's hybrid cache.
///
/// @discussion A sliding layer never reads further back than its window, so retaining more than the
/// window costs memory for nothing; the full layers keep everything. Both advance together, so the
/// rotary offset is one number. A layer index addresses the cache of its own kind; the other cache's
/// slot for that index is never touched.
public final class NFKMLXGemma3Cache {
    let full: NFKMLXKeyValueCache
    let sliding: NFKMLXKeyValueCache
    /// The sliding cache's window, which is the configuration's.
    let window: Int

    public init(layerCount: Int, slidingWindow: Int) {
        window = Swift.max(slidingWindow, 2)
        full = NFKMLXKeyValueCache(layerCount: layerCount)
        sliding = NFKMLXKeyValueCache(layerCount: layerCount, window: window)
    }

    /// Positions appended so far, which is the rotary offset of the next one.
    public var offset: Int { full.offset }

    func cache(for kind: NFKMLXGemmaAttentionKind) -> NFKMLXKeyValueCache {
        kind == .full ? full : sliding
    }

    func advance(by count: Int) {
        full.advance(by: count)
        sliding.advance(by: count)
    }
}

/// The attention masks a Gemma 3 pass needs, one per layer kind.
///
/// @discussion A full layer admits every key at or before the query; a sliding layer additionally
/// refuses a key more than `window - 1` positions back. A multimodal prompt adds the reference's
/// blockwise rule on top: the tokens of ONE image attend to each other in both directions, so an
/// image token sees the rest of its own image whichever side it lies on. `blockIds` marks each
/// position of the whole sequence with its image's index, or -1 for text. A single-token step against
/// a cache needs no mask at all — the full cache holds only the past and the sliding cache is trimmed
/// to the window — so both come back nil then.
enum NFKMLXGemma3Masks {
    /// The masks for `length` new positions starting at `offset`, the sliding one built against the
    /// `min(offset, window - 1)` older positions its cache retains.
    static func make(length: Int, offset: Int, window: Int, blockIds: [Int]?,
                     bidirectional: Bool) -> (full: MLXArray?, sliding: MLXArray?) {
        let total = offset + length
        let hasBlocks = blockIds.map { $0.contains { $0 >= 0 } } ?? false
        if length == 1, !hasBlocks, !bidirectional { return (nil, nil) }

        let rows = MLXArray(Int32(offset) ..< Int32(total)).reshaped([length, 1])
        func admitted(keyStart: Int) -> MLXArray {
            let columns = MLXArray(Int32(keyStart) ..< Int32(total)).reshaped([1, total - keyStart])
            var allowed = bidirectional ? (columns .<= Int32(total)) : (columns .<= rows)
            if let blockIds, hasBlocks {
                let ids = MLXArray(blockIds.map(Int32.init))
                let rowBlocks = ids[offset ..< total].reshaped([length, 1])
                let columnBlocks = ids[keyStart ..< total].reshaped([1, total - keyStart])
                allowed = allowed .|| ((rowBlocks .== columnBlocks) .&& (rowBlocks .>= Int32(0)))
            }
            return allowed
        }
        func additive(_ allowed: MLXArray) -> MLXArray {
            MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        }
        let full = additive(admitted(keyStart: 0))

        let retained = Swift.min(offset, window - 1)
        let keyStart = offset - retained
        let columns = MLXArray(Int32(keyStart) ..< Int32(total)).reshaped([1, total - keyStart])
        let distance = rows - columns
        let inWindow = bidirectional ? (abs(distance) .< Int32(window)) : (distance .< Int32(window))
        let sliding = additive(admitted(keyStart: keyStart) .&& inWindow)
        return (full, sliding)
    }
}

/// The Gemma 3 text decoder: a scaled token embedding, the sandwich-normalized blocks alternating
/// sliding and full attention, a final normalization, and the tied output projection.
public final class NFKMLXGemma3Net: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKGemma3Block]
    @ModuleInfo(key: "norm") var norm: NFKGemma3Norm

    public let configuration: NFKMLXGemma3Configuration
    private let embeddingScale: Float

    public init(_ c: NFKMLXGemma3Configuration) {
        configuration = c
        embeddingScale = sqrt(Float(c.hiddenSize))
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layerCount).map { NFKGemma3Block(c, fullAttention: c.isFullAttention(layer: $0)) }
        _norm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    /// The scaled token embedding, the main stream a caller splices image soft tokens into before
    /// calling ``hiddenStates(fromEmbeddings:cache:blockIds:)``.
    public func embed(_ tokens: MLXArray) -> MLXArray {
        embedTokens(tokens) * embeddingScale
    }

    /// The post-norm hidden states `[batch, length, hidden]` over already-embedded inputs, through the
    /// cache when one is given. `blockIds` marks each position of the WHOLE sequence (cached positions
    /// included) with its image index or -1, for the bidirectional image attention; nil is plain causal.
    public func hiddenStates(fromEmbeddings embeddings: MLXArray, cache: NFKMLXGemma3Cache? = nil,
                             blockIds: [Int]? = nil) -> MLXArray {
        var trace = [MLXArray]()
        return forward(embeddings, cache: cache, blockIds: blockIds, trace: &trace)
    }

    /// The output projection on its own: post-norm hidden states → logits, tied to the embedding and
    /// soft-capped where the configuration says so.
    public func logits(fromHidden hidden: MLXArray) -> MLXArray {
        var logits = embedTokens.asLinear(hidden)
        if configuration.finalLogitSoftcap > 0 {
            logits = tanh(logits / configuration.finalLogitSoftcap) * configuration.finalLogitSoftcap
        }
        return logits
    }

    /// The logits `[batch, length, vocabulary]` for token ids, through the cache when one is given.
    public func callAsFunction(_ tokens: MLXArray, cache: NFKMLXGemma3Cache? = nil) -> MLXArray {
        logits(fromHidden: hiddenStates(fromEmbeddings: embed(tokens), cache: cache))
    }

    /// The state entering the stack and the state each layer produces, the final norm applied to the
    /// last — the reference's `output_hidden_states` convention, so a divergence is located to a layer.
    public func layerStates(_ tokens: MLXArray) -> [MLXArray] {
        var trace = [MLXArray]()
        _ = forward(embed(tokens), cache: nil, blockIds: nil, trace: &trace)
        return trace
    }

    private func forward(_ embeddings: MLXArray, cache: NFKMLXGemma3Cache?, blockIds: [Int]?,
                         trace: inout [MLXArray]) -> MLXArray {
        let c = configuration
        var hidden = embeddings
        let length = embeddings.shape[1]
        let masks = NFKMLXGemma3Masks.make(length: length, offset: cache?.offset ?? 0,
                                           window: c.slidingWindow, blockIds: blockIds,
                                           bidirectional: c.isBidirectional)
        trace.append(hidden)
        for (index, layer) in layers.enumerated() {
            let kind = c.layerTypes[index]
            hidden = layer(hidden, mask: kind == .full ? masks.full : masks.sliding,
                           cache: cache?.cache(for: kind), layer: index)
            trace.append(hidden)
        }
        cache?.advance(by: length)
        hidden = norm(hidden)
        trace[trace.count - 1] = hidden
        return hidden
    }
}

/// Building a Gemma 3 decoder, reading a release's configuration, and loading its weights.
@objc(NFKMLXGemma3Language)
public final class NFKMLXGemma3Language: NSObject {

    static func makeNet(_ configuration: NFKMLXGemma3Configuration = .gemma3_270M) -> NFKMLXGemma3Net {
        NFKMLXGemma3Net(configuration)
    }

    /// Reads a released `config.json`, whose decoder sits under `text_config` in a multimodal release.
    ///
    /// @discussion Accepts `gemma3_text` (a text-only release, and the multimodal releases' inner
    /// decoder) and `gemma3` (the multimodal wrapper). `gemma3n` and `gemma4` share a name and nothing
    /// else, so they are refused rather than loaded into this stack; `NFKMLXGemma3n` and
    /// `NFKMLXGemmaLanguage` are the readers that take them. The rotary scaling is read only
    /// in its `linear` kind, the one the family ships; another kind is refused because loading its
    /// weights under the wrong rotary runs and is wrong.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXGemma3Configuration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        return try configuration(fromJSON: json)
    }

    static func configuration(fromJSON json: [String: Any]) throws -> NFKMLXGemma3Configuration {
        let text = (json["text_config"] as? [String: Any]) ?? json
        let outer = (json["model_type"] as? String) ?? ""
        let kind = (text["model_type"] as? String) ?? outer
        guard kind == "gemma3_text" || (kind == "gemma3" && json["text_config"] == nil) else {
            throw NFKMLXError.unsupportedConfiguration(
                "this reads a gemma3 text decoder, not \(kind)" + Self.elsewhere(kind))
        }
        if !outer.isEmpty, outer != "gemma3", outer != "gemma3_text" {
            throw NFKMLXError.unsupportedConfiguration(
                "this reads a gemma3 release, not \(outer)" + Self.elsewhere(outer))
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (text[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (text[key] as? NSNumber)?.floatValue ?? fallback }

        // The scaling is stated either as `rope_scaling` (the released files) or, standardized by a
        // newer transformers, under `rope_parameters.full_attention`.
        var scaling: [String: Any]? = text["rope_scaling"] as? [String: Any]
        var globalTheta = real("rope_theta", 1_000_000)
        var localTheta = real("rope_local_base_freq", 10_000)
        if let parameters = text["rope_parameters"] as? [String: Any] {
            if let full = parameters["full_attention"] as? [String: Any] {
                globalTheta = (full["rope_theta"] as? NSNumber)?.floatValue ?? globalTheta
                if full["rope_type"] != nil || full["factor"] != nil { scaling = scaling ?? full }
            }
            if let sliding = parameters["sliding_attention"] as? [String: Any] {
                localTheta = (sliding["rope_theta"] as? NSNumber)?.floatValue ?? localTheta
            }
        }
        var factor: Float = 1
        if let scaling {
            let type = ((scaling["rope_type"] ?? scaling["type"]) as? String) ?? "default"
            guard type == "linear" || type == "default" else {
                throw NFKMLXError.unsupportedConfiguration(
                    "Gemma 3 rope scaling of kind \(type) is not implemented (linear is)")
            }
            if type == "linear" {
                factor = (scaling["factor"] as? NSNumber)?.floatValue ?? 1
            }
        }

        let layerCount = integer("num_hidden_layers", 18)
        let pattern = integer("sliding_window_pattern", integer("_sliding_window_pattern", 6))
        let layerTypes = (text["layer_types"] as? [String])?.compactMap { NFKMLXGemmaAttentionKind(rawValue: $0) }
        if let layerTypes, layerTypes.count != layerCount {
            throw NFKMLXError.unsupportedConfiguration("layer_types names \(layerTypes.count) layers of \(layerCount)")
        }
        let bidirectional = (text["use_bidirectional_attention"] as? NSNumber)?.boolValue ?? false
        // A bidirectional release states its window as the full span; the reference halves it plus one
        // into an exclusive bound (`(sliding_window // 2) + 1`), so `|q - k| < bound` covers the span.
        var window = integer("sliding_window", 512)
        if bidirectional {
            window = window / 2 + 1
        }

        return NFKMLXGemma3Configuration(
            hiddenSize: integer("hidden_size", 640),
            layerCount: layerCount,
            headCount: integer("num_attention_heads", 4),
            keyValueHeadCount: integer("num_key_value_heads", 1),
            headDimensions: integer("head_dim", 256),
            intermediateSize: integer("intermediate_size", 2048),
            vocabularySize: integer("vocab_size", 262_144),
            ropeTheta: globalTheta,
            ropeLocalTheta: localTheta,
            ropeScalingFactor: factor,
            slidingWindow: window,
            layerTypes: layerTypes,
            slidingWindowPattern: pattern,
            queryPreAttnScalar: real("query_pre_attn_scalar", 256),
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            finalLogitSoftcap: real("final_logit_softcapping", 0),
            attentionLogitSoftcap: real("attn_logit_softcapping", 0),
            isBidirectional: bidirectional)
    }

    /// The reader that does take `kind`, for a refusal that says where to go rather than only no.
    private static func elsewhere(_ kind: String) -> String {
        switch kind {
        case "gemma3n", "gemma3n_text": return " — NFKMLXGemma3n reads that one"
        case "gemma4", "gemma4_text", "gemma4_unified_text": return " — NFKMLXGemmaLanguage reads that one"
        default: return ""
        }
    }

    /// The decoder's module key for a checkpoint key, or nil for a tensor that is not the decoder's.
    ///
    /// @discussion A text-only release stores the decoder under `model.`; a multimodal one written by
    /// transformers 4.x under `language_model.model.` and one written by 5.x under
    /// `model.language_model.`. The tied `lm_head.weight` (under any of them, or at the top) is the
    /// embedding written out again and is dropped.
    static func decoderName(of key: String) -> String? {
        if key.hasSuffix("lm_head.weight") { return nil }
        if key.hasPrefix("model.vision_tower.") || key.hasPrefix("model.multi_modal_projector.") { return nil }
        return stripped(key, prefixes: ["model.language_model.", "language_model.model.", "model."])
    }

    /// `key` with the first matching prefix removed, or nil when none matches.
    static func stripped(_ key: String, prefixes: [String]) -> String? {
        for prefix in prefixes where key.hasPrefix(prefix) {
            return String(key.dropFirst(prefix.count))
        }
        return nil
    }

    /// Loads the decoder from a released directory, single-file or sharded, taking only the language
    /// model's tensors; a multimodal release's vision tower and projector are skipped, and a strict
    /// apply then proves the decoder's own set is complete.
    static func loadWeights(into net: NFKMLXGemma3Net, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision,
                                                     remap: decoderName(of:))
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
