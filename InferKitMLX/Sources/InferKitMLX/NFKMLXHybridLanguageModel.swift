//
//  NFKMLXHybridLanguageModel.swift
//  InferKitMLX
//
//  The hybrid decoder Qwen3.5, Qwen3.6, and Qwen3.8 are built from (`Qwen3_5ForConditionalGeneration`).
//  It is not the dense stack `NFKMLXLanguageNet` implements: three quarters of its layers replace
//  attention with a **gated delta-rule recurrence** — linear in sequence length, with a fixed-size
//  state instead of a growing key-value cache — and the remaining quarter is full attention whose
//  output is gated.
//
//  BUILT BUT NOT MEASURED. The smallest release in this family is 27B, which is about 54 GB at the
//  precision it ships in, so no forward pass against real weights has been run on this machine and
//  there is no parity record. What IS verified is structural: every parameter this module declares is
//  checked against the released checkpoint's own safetensors header, name by name and shape by shape,
//  and a small configuration runs end to end with random weights. Treat the numerics as unverified
//  until a machine that can hold the weights measures them.
//
//  Scope: the language model. A release in this family also carries a vision tower (`model.visual`)
//  and a multi-token-prediction head (`mtp`), which are separate features rather than parts of the
//  decoder.
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// Which kind of layer sits at a given depth.
public enum NFKMLXHybridLayerKind: String, Sendable {
    /// A gated delta-rule recurrence, linear in sequence length.
    case linearAttention = "linear_attention"
    /// Ordinary attention over the whole prefix, with a gated output.
    case fullAttention = "full_attention"
}

/// The geometry of a hybrid decoder.
public struct NFKMLXHybridConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    public var rmsEpsilon: Float

    // Full-attention layers.
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    public var ropeTheta: Float
    /// The fraction of each head's channels the rotary embedding turns; the rest pass through.
    public var partialRotaryFactor: Float
    /// Whether the query projection also emits a gate for the attention output.
    public var gatesAttentionOutput: Bool

    // Linear-attention layers.
    public var linearKeyHeadCount: Int
    public var linearKeyHeadDimensions: Int
    public var linearValueHeadCount: Int
    public var linearValueHeadDimensions: Int
    /// The depthwise convolution's kernel over the projected q, k, and v.
    public var linearConvolutionKernel: Int

    /// Whether the output projection reuses the embedding matrix. 4B ties; 27B does not.
    public var tiesWordEmbeddings: Bool

    /// One entry per layer. The releases place a full-attention layer every fourth.
    public var layerTypes: [NFKMLXHybridLayerKind]

    public init(hiddenSize: Int = 5120, layerCount: Int = 64, intermediateSize: Int = 17408,
                vocabularySize: Int = 248_320, rmsEpsilon: Float = 1e-6,
                headCount: Int = 24, keyValueHeadCount: Int = 4, headDimensions: Int = 256,
                ropeTheta: Float = 10_000_000, partialRotaryFactor: Float = 0.25,
                gatesAttentionOutput: Bool = true,
                linearKeyHeadCount: Int = 16, linearKeyHeadDimensions: Int = 128,
                linearValueHeadCount: Int = 48, linearValueHeadDimensions: Int = 128,
                linearConvolutionKernel: Int = 4, fullAttentionInterval: Int = 4,
                tiesWordEmbeddings: Bool = false,
                layerTypes: [NFKMLXHybridLayerKind]? = nil) {
        self.tiesWordEmbeddings = tiesWordEmbeddings
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.rmsEpsilon = rmsEpsilon
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.gatesAttentionOutput = gatesAttentionOutput
        self.linearKeyHeadCount = linearKeyHeadCount
        self.linearKeyHeadDimensions = linearKeyHeadDimensions
        self.linearValueHeadCount = linearValueHeadCount
        self.linearValueHeadDimensions = linearValueHeadDimensions
        self.linearConvolutionKernel = linearConvolutionKernel
        // The released pattern: every `fullAttentionInterval`-th layer attends fully, the rest recur.
        self.layerTypes = layerTypes ?? (0 ..< layerCount).map {
            ($0 + 1) % fullAttentionInterval == 0 ? .fullAttention : .linearAttention
        }
    }

    /// The width the linear branch's fused q/k/v projection emits.
    var linearQKVWidth: Int {
        2 * linearKeyHeadCount * linearKeyHeadDimensions
            + linearValueHeadCount * linearValueHeadDimensions
    }

    /// The width of the linear branch's value stream, which its gate and output projection match.
    var linearValueWidth: Int { linearValueHeadCount * linearValueHeadDimensions }

    /// How many of a head's channels the rotary embedding turns.
    var rotaryDimensions: Int { Int(Float(headDimensions) * partialRotaryFactor) }

    /// The released `Qwen/Qwen3.8-27B` language model.
    public static let qwen3_8_27B = NFKMLXHybridConfiguration()
}

/// The hybrid family's normalization: `x · (1 + w)`, with the weight stored as an offset from one.
///
/// Qwen3.5 / 3.6 / 3.8 use this; the DENSE Qwen3 stack does not, and neither does Gemma 4 — both of
/// those scale by the weight directly. The two conventions are indistinguishable by shape and differ
/// in every number, so each model's own source decides it. Note the gated norm inside the recurrence
/// (`linear_attn.norm`) is the PLAIN kind even here.
final class NFKHybridNorm: Module {
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

/// Full attention with a gated output.
///
/// The query projection emits twice the width a plain one would: the first half is the queries, the
/// second is a gate applied to the attention result. Only the leading `rotaryDimensions` channels of
/// each head are rotated, which is what `partial_rotary_factor` means.
final class NFKHybridAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: NFKHybridNorm
    @ModuleInfo(key: "k_norm") var keyNorm: NFKHybridNorm

    let configuration: NFKMLXHybridConfiguration
    let rope: RoPE

    init(_ c: NFKMLXHybridConfiguration) {
        configuration = c
        rope = RoPE(dimensions: max(c.rotaryDimensions, 1), traditional: false, base: c.ropeTheta)
        let queries = c.headCount * c.headDimensions
        _queryProjection.wrappedValue = Linear(c.hiddenSize,
                                               c.gatesAttentionOutput ? queries * 2 : queries,
                                               bias: false)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions,
                                             bias: false)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions,
                                               bias: false)
        _outputProjection.wrappedValue = Linear(queries, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = NFKHybridNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        _keyNorm.wrappedValue = NFKHybridNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.shape[0], x.shape[1])
        let width = c.headCount * c.headDimensions

        // The projection is viewed as [.., heads, 2 · headDim] and split on the LAST axis, so each
        // head's queries and gate are interleaved. Splitting the flat width into two halves takes the
        // wrong channels entirely.
        var queries: MLXArray
        var gate: MLXArray?
        if c.gatesAttentionOutput {
            let paired = queryProjection(x)
                .reshaped([batch, length, c.headCount, c.headDimensions * 2])
            queries = paired[.ellipsis, 0 ..< c.headDimensions]
            gate = paired[.ellipsis, c.headDimensions...].reshaped([batch, length, width])
        } else {
            queries = queryProjection(x).reshaped([batch, length, c.headCount, c.headDimensions])
            gate = nil
        }

        queries = queryNorm(queries)
        var keys = keyNorm(keyProjection(x)
            .reshaped([batch, length, c.keyValueHeadCount, c.headDimensions]))
        var values = valueProjection(x)
            .reshaped([batch, length, c.keyValueHeadCount, c.headDimensions])

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Partial rotary: only the leading channels turn, the rest are carried through untouched.
        let turned = c.rotaryDimensions
        if turned > 0 && turned < c.headDimensions {
            queries = concatenated([rope(queries[.ellipsis, 0 ..< turned], offset: 0),
                                    queries[.ellipsis, turned...]], axis: -1)
            keys = concatenated([rope(keys[.ellipsis, 0 ..< turned], offset: 0),
                                 keys[.ellipsis, turned...]], axis: -1)
        } else if turned > 0 {
            queries = rope(queries, offset: 0)
            keys = rope(keys, offset: 0)
        }

        // A `.checkpoint`-precision load makes this a bf16 module, and the fused attention refuses
        // a float32 mask that does not promote to its own type — invisible at float32.
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1 / sqrt(Float(c.headDimensions)), mask: mask.map { $0.asType(queries.dtype) })
        var result = attended.transposed(0, 2, 1, 3).reshaped([batch, length, width])
        if let gate {
            // The config field is named `output_gate_type: swish`, and the implementation applies a
            // plain sigmoid. The implementation is what the weights were trained against.
            result = result * sigmoid(gate)
        }
        return outputProjection(result)
    }
}

/// The gated delta-rule recurrence that replaces attention in three quarters of the layers.
///
/// One fused projection produces queries, keys, and values; a depthwise convolution mixes each over a
/// short window; a per-head decay and write strength drive a fixed-size state, which the queries then
/// read. The state is `[value heads, key dimensions, value dimensions]` and does not grow with the
/// sequence, which is what makes the layer linear.
final class NFKHybridLinearAttention: Module {
    @ModuleInfo(key: "in_proj_qkv") var qkvProjection: Linear
    @ModuleInfo(key: "in_proj_z") var gateProjection: Linear
    @ModuleInfo(key: "in_proj_a") var decayProjection: Linear
    @ModuleInfo(key: "in_proj_b") var writeProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "conv1d") var convolution: Conv1d

    @ParameterInfo(key: "A_log") var decayLog: MLXArray
    @ParameterInfo(key: "dt_bias") var stepBias: MLXArray

    let configuration: NFKMLXHybridConfiguration

    init(_ c: NFKMLXHybridConfiguration) {
        configuration = c
        _qkvProjection.wrappedValue = Linear(c.hiddenSize, c.linearQKVWidth, bias: false)
        _gateProjection.wrappedValue = Linear(c.hiddenSize, c.linearValueWidth, bias: false)
        _decayProjection.wrappedValue = Linear(c.hiddenSize, c.linearValueHeadCount, bias: false)
        _writeProjection.wrappedValue = Linear(c.hiddenSize, c.linearValueHeadCount, bias: false)
        _outputProjection.wrappedValue = Linear(c.linearValueWidth, c.hiddenSize, bias: false)
        // The gated normalization runs over one value head's channels.
        _norm.wrappedValue = RMSNorm(dimensions: c.linearValueHeadDimensions, eps: c.rmsEpsilon)
        // Depthwise over the fused projection: one filter per channel.
        _convolution.wrappedValue = Conv1d(inputChannels: c.linearQKVWidth,
                                           outputChannels: c.linearQKVWidth,
                                           kernelSize: c.linearConvolutionKernel,
                                           groups: c.linearQKVWidth, bias: false)
        _decayLog.wrappedValue = MLXArray.zeros([c.linearValueHeadCount])
        _stepBias.wrappedValue = MLXArray.zeros([c.linearValueHeadCount])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.shape[0], x.shape[1])

        // Causal short convolution: pad on the left so a position sees only itself and its history.
        let projected = qkvProjection(x)
        let padded = padded(projected, widths: [IntOrPair((0, 0)),
                                                IntOrPair((c.linearConvolutionKernel - 1, 0)),
                                                IntOrPair((0, 0))])
        let mixed = silu(convolution(padded))

        let keyWidth = c.linearKeyHeadCount * c.linearKeyHeadDimensions
        var queries = mixed[0..., 0..., 0 ..< keyWidth]
        var keys = mixed[0..., 0..., keyWidth ..< (2 * keyWidth)]
        let values = mixed[0..., 0..., (2 * keyWidth)...]

        queries = queries.reshaped([batch, length, c.linearKeyHeadCount, c.linearKeyHeadDimensions])
        keys = keys.reshaped([batch, length, c.linearKeyHeadCount, c.linearKeyHeadDimensions])
        let shaped = values.reshaped([batch, length, c.linearValueHeadCount, c.linearValueHeadDimensions])

        // The delta rule reads and writes a unit-norm key space.
        queries = queries / sqrt((queries * queries).sum(axis: -1, keepDims: true) + 1e-6)
        keys = keys / sqrt((keys * keys).sum(axis: -1, keepDims: true) + 1e-6)

        // Per-head decay and write strength. `A_log` is stored as a log so the rate stays positive;
        // the reference keeps `g` as a LOG rate and exponentiates it inside the recurrence.
        let decay = exp(-exp(decayLog) * softplus(decayProjection(x) + stepBias))
        let write = sigmoid(writeProjection(x))

        // Key heads are shared across a group of value heads, as grouped-query attention shares them.
        let group = c.linearValueHeadCount / c.linearKeyHeadCount
        var state = MLXArray.zeros([batch, c.linearValueHeadCount,
                                    c.linearKeyHeadDimensions, c.linearValueHeadDimensions])
        var outputs = [MLXArray]()
        outputs.reserveCapacity(length)

        // The reference scales the queries by the key width before the recurrence, once.
        let scale = 1 / sqrt(Float(c.linearKeyHeadDimensions))

        for step in 0 ..< length {
            let q = repeated(queries[0..., step], count: group, axis: 1) * scale
            let k = repeated(keys[0..., step], count: group, axis: 1)
            let v = shaped[0..., step]                                     // [batch, value heads, value dims]
            let g = decay[0..., step].reshaped([batch, c.linearValueHeadCount, 1, 1])
            let b = write[0..., step].reshaped([batch, c.linearValueHeadCount, 1])

            // Decay FIRST, then read what the decayed state holds for this key, then write the
            // correction toward v. Reading before decaying is a different recurrence entirely.
            state = state * g
            let held = (state * k.expandedDimensions(axis: -1)).sum(axis: 2)   // [batch, heads, value dims]
            let correction = (v - held) * b
            state = state + k.expandedDimensions(axis: -1) * correction.expandedDimensions(axis: 2)
            outputs.append((state * q.expandedDimensions(axis: -1)).sum(axis: 2))
        }

        let read = stacked(outputs, axis: 1)      // [batch, length, value heads, value dims]
        let gate = gateProjection(x).reshaped([batch, length, c.linearValueHeadCount,
                                               c.linearValueHeadDimensions])
        let gated = norm(read) * sigmoid(gate) * gate
        return outputProjection(gated.reshaped([batch, length, c.linearValueWidth]))
    }
}

/// One layer of the hybrid stack. Which branch it carries is fixed by its depth.
final class NFKHybridBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKHybridAttention?
    @ModuleInfo(key: "linear_attn") var linearAttention: NFKHybridLinearAttention?
    @ModuleInfo(key: "mlp") var feedForward: NFKLMFeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: NFKHybridNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: NFKHybridNorm

    let kind: NFKMLXHybridLayerKind

    init(_ c: NFKMLXHybridConfiguration, kind: NFKMLXHybridLayerKind) {
        self.kind = kind
        // Exactly one branch exists per layer, which is what the checkpoint carries: a linear layer
        // has no `self_attn.*` and a full layer has no `linear_attn.*`.
        _attention.wrappedValue = kind == .fullAttention ? NFKHybridAttention(c) : nil
        _linearAttention.wrappedValue = kind == .linearAttention ? NFKHybridLinearAttention(c) : nil
        _feedForward.wrappedValue = NFKLMFeedForward(NFKMLXLanguageConfiguration(
            hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize))
        _inputNorm.wrappedValue = NFKHybridNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _postNorm.wrappedValue = NFKHybridNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let normalized = inputNorm(x)
        let mixed: MLXArray
        if let attention {
            mixed = attention(normalized, mask: mask)
        } else if let linearAttention {
            mixed = linearAttention(normalized)
        } else {
            mixed = normalized
        }
        let residual = x + mixed
        return residual + feedForward(postNorm(residual))
    }
}

/// The hybrid decoder, under the `model.language_model.` prefix the releases use.
final class NFKHybridCore: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKHybridBlock]
    @ModuleInfo(key: "norm") var norm: NFKHybridNorm

    init(_ c: NFKMLXHybridConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layerCount).map { NFKHybridBlock(c, kind: c.layerTypes[$0]) }
        _norm.wrappedValue = NFKHybridNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }
}

/// A hybrid causal language model.
///
/// See the file comment: this is built and structurally verified against the released checkpoint, and
/// its numerics are unmeasured because the smallest release does not fit on the machine that built it.
public final class NFKMLXHybridLanguageNet: Module {
    @ModuleInfo(key: "model") var model: NFKHybridCore
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: NFKMLXHybridConfiguration

    init(_ c: NFKMLXHybridConfiguration) {
        configuration = c
        _model.wrappedValue = NFKHybridCore(c)
        // A tied release ships no `lm_head.weight`: the embedding matrix is the output projection.
        _lmHead.wrappedValue = c.tiesWordEmbeddings
            ? nil : Linear(c.hiddenSize, c.vocabularySize, bias: false)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var trace = [MLXArray]()
        return forward(tokens, trace: &trace)
    }

    /// The state entering the stack, then each layer's output, with the final normalization folded
    /// into the last entry — the convention the reference's `output_hidden_states` uses.
    func hiddenStates(_ tokens: MLXArray) -> [MLXArray] {
        var trace = [MLXArray]()
        _ = forward(tokens, trace: &trace)
        return trace
    }

    private func forward(_ tokens: MLXArray, trace: inout [MLXArray]) -> MLXArray {
        var hidden = model.embedTokens(tokens)
        trace.append(hidden)
        let length = tokens.shape[1]
        let mask: MLXArray? = length > 1 ? NFKMLXLanguageNet.causalMask(length, offset: 0) : nil
        for layer in model.layers {
            hidden = layer(hidden, mask: mask)
            trace.append(hidden)
        }
        hidden = model.norm(hidden)
        if !trace.isEmpty { trace[trace.count - 1] = hidden }
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }
}

/// Building the hybrid decoder and reading a release's configuration.
@objc(NFKMLXHybridLanguage)
public final class NFKMLXHybridLanguage: NSObject {

    static func makeNet(_ configuration: NFKMLXHybridConfiguration = .qwen3_8_27B)
        -> NFKMLXHybridLanguageNet {
        NFKMLXHybridLanguageNet(configuration)
    }

    /// Reads a `Qwen3_5ForConditionalGeneration` config, whose decoder lives under `text_config`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXHybridConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        // The decoder's own numbers are nested; the top level describes the multimodal wrapper.
        let text = (json["text_config"] as? [String: Any]) ?? json
        guard (text["model_type"] as? String)?.hasPrefix("qwen3_5") == true
                || (json["model_type"] as? String)?.hasPrefix("qwen3_5") == true else {
            throw NFKMLXError.unsupportedConfiguration(
                "this reads the qwen3_5 hybrid decoder; use NFKMLXLanguage for a dense one")
        }

        func integer(_ key: String, _ fallback: Int) -> Int { (text[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (text[key] as? NSNumber)?.floatValue ?? fallback }

        var theta: Float = real("rope_theta", 10_000_000)
        var rotaryFactor: Float = real("partial_rotary_factor", 0.25)
        if let rope = text["rope_parameters"] as? [String: Any] {
            theta = (rope["rope_theta"] as? NSNumber)?.floatValue ?? theta
            rotaryFactor = (rope["partial_rotary_factor"] as? NSNumber)?.floatValue ?? rotaryFactor
        }

        let kinds = (text["layer_types"] as? [String])?
            .compactMap { NFKMLXHybridLayerKind(rawValue: $0) }

        return NFKMLXHybridConfiguration(
            hiddenSize: integer("hidden_size", 5120),
            layerCount: integer("num_hidden_layers", 64),
            intermediateSize: integer("intermediate_size", 17408),
            vocabularySize: integer("vocab_size", 248_320),
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            headCount: integer("num_attention_heads", 24),
            keyValueHeadCount: integer("num_key_value_heads", 4),
            headDimensions: integer("head_dim", 256),
            ropeTheta: theta,
            partialRotaryFactor: rotaryFactor,
            gatesAttentionOutput: (text["attn_output_gate"] as? NSNumber)?.boolValue ?? true,
            linearKeyHeadCount: integer("linear_num_key_heads", 16),
            linearKeyHeadDimensions: integer("linear_key_head_dim", 128),
            linearValueHeadCount: integer("linear_num_value_heads", 48),
            linearValueHeadDimensions: integer("linear_value_head_dim", 128),
            linearConvolutionKernel: integer("linear_conv_kernel_dim", 4),
            fullAttentionInterval: integer("full_attention_interval", 4),
            tiesWordEmbeddings: (text["tie_word_embeddings"] as? NSNumber)?.boolValue
                ?? (json["tie_word_embeddings"] as? NSNumber)?.boolValue ?? false,
            layerTypes: kinds)
    }

    /// Loads a released hybrid decoder from its directory, following the shard index.
    ///
    /// The releases nest the decoder under `model.language_model.` beside a vision tower and a
    /// multi-token-prediction head; only the decoder's tensors are taken.
    static func loadWeights(into net: NFKMLXHybridLanguageNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        let tied = net.lmHead == nil
        let read = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) {
            guard let name = moduleKey(forReference: $0) else { return nil }
            return tied && name.hasPrefix("lm_head.") ? nil : name
        }
        // A depthwise convolution is stored [channels, 1, kernel] and wanted [channels, kernel, 1].
        let merged = read.map { name, value in
            (name, name.hasSuffix("conv1d.weight") && value.ndim == 3
                 ? value.transposed(0, 2, 1) : value)
        }
        try NFKMLXWeights.apply(merged, to: net)
    }

    /// The module key a checkpoint key maps to, or nil for a tensor this decoder does not carry.
    static func moduleKey(forReference key: String) -> String? {
        let prefix = "model.language_model."
        if key.hasPrefix(prefix) { return "model." + key.dropFirst(prefix.count) }
        if key == "lm_head.weight" { return key }
        return nil                      // the vision tower and the multi-token-prediction head
    }

    /// The checkpoint key a parameter of this module corresponds to.
    ///
    /// The releases nest the decoder under `model.language_model.`, beside a vision tower and a
    /// multi-token-prediction head that this module does not implement.
    static func referenceKey(for parameter: String) -> String {
        parameter.hasPrefix("model.")
            ? "model.language_model." + parameter.dropFirst("model.".count)
            : parameter
    }
}
