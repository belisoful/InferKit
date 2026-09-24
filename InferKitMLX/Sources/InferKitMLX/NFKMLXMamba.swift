//
//  NFKMLXMamba.swift
//  InferKitMLX
//
//  The Mamba-2 decoder (`Mamba2ForCausalLM`), the toolkit's first state-space model. It is not the
//  dense stack `NFKMLXLanguageNet` implements and not the gated delta-rule recurrence
//  `NFKMLXHybridLanguage` implements: every layer replaces attention with a **selective state-space
//  scan** (SSD), linear in sequence length, carrying a fixed-size `[heads, head_dim, state]` state
//  rather than a growing key-value cache.
//
//  One selective-scan implementation serves the whole SSM class: Codestral-Mamba (pure Mamba-2),
//  and the hybrid Mamba-attention decoders (Granite 4.0-H, Nemotron Nano 2) that add attention and a
//  mixture of experts on top of this mixer.
//
//  The scan here is the explicit sequential recurrence, mathematically equal to the chunked segment-sum
//  form HF's slow path runs. It is prefill-only, like the hybrid and DeepSeek decoders beside it; the
//  chunked scan and an incremental single-step cache are separate performance and generation features.
//
//  Reference: HF `transformers` `Mamba2Mixer.torch_forward`, the CPU slow path (no CUDA kernel needed).
//
//  The module tree mirrors the checkpoint's own nesting (`backbone.embeddings`, `backbone.layers.N`,
//  `backbone.norm_f`, `lm_head`) with REAL submodules, not dotted keys: MLX splits a parameter key on
//  `.`, so a dotted `@ModuleInfo` key flattens to the right name yet cannot receive an unflattened
//  weight, which loads clean and computes on random values.
//

import Foundation
import MLX
import MLXNN

/// The geometry of a Mamba-2 decoder.
public struct NFKMLXMamba2Configuration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var vocabularySize: Int
    public var rmsEpsilon: Float

    /// The SSM inner width, `expand · hiddenSize`. Equals `headCount · headDimensions`.
    public var intermediateSize: Int
    public var headCount: Int
    public var headDimensions: Int
    /// The state dimension per head (the SSM's `N`).
    public var stateSize: Int
    /// How many key/value groups the `B` and `C` projections carry; each spans `headCount / groupCount`
    /// heads.
    public var groupCount: Int
    /// The depthwise causal convolution's kernel over the concatenated `x`, `B`, and `C`.
    public var convolutionKernel: Int
    public var useConvolutionBias: Bool
    public var useProjectionBias: Bool

    /// Whether the output projection reuses the embedding matrix.
    public var tiesWordEmbeddings: Bool

    /// The lower and upper clamp on the discretization step after softplus.
    public var timeStepLowerLimit: Float
    public var timeStepUpperLimit: Float

    /// How many equal spans of the inner width the gated output norm normalizes within. `1` (standard
    /// Mamba-2, Granite) normalizes over the whole width; Nemotron-H sets this to its `n_groups`.
    public var gatedNormGroups: Int

    /// Whether the residual stream between blocks is held in float32 whatever the weights' precision
    /// (`residual_in_fp32`), each block's norm reading it rounded to the weights' type. Invisible in a
    /// float32 run; in bf16 it is where the reference keeps its stream.
    public var residualInFloat32: Bool

    public init(hiddenSize: Int = 4096, layerCount: Int = 64, vocabularySize: Int = 32768,
                rmsEpsilon: Float = 1e-5, intermediateSize: Int = 8192,
                headCount: Int = 128, headDimensions: Int = 64, stateSize: Int = 128,
                groupCount: Int = 8, convolutionKernel: Int = 4,
                useConvolutionBias: Bool = true, useProjectionBias: Bool = false,
                tiesWordEmbeddings: Bool = false,
                timeStepLowerLimit: Float = 0.0, timeStepUpperLimit: Float = .infinity,
                gatedNormGroups: Int = 1, residualInFloat32: Bool = true) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.vocabularySize = vocabularySize
        self.rmsEpsilon = rmsEpsilon
        self.intermediateSize = intermediateSize
        self.headCount = headCount
        self.headDimensions = headDimensions
        self.stateSize = stateSize
        self.groupCount = groupCount
        self.convolutionKernel = convolutionKernel
        self.useConvolutionBias = useConvolutionBias
        self.useProjectionBias = useProjectionBias
        self.tiesWordEmbeddings = tiesWordEmbeddings
        self.timeStepLowerLimit = timeStepLowerLimit
        self.timeStepUpperLimit = timeStepUpperLimit
        self.gatedNormGroups = gatedNormGroups
        self.residualInFloat32 = residualInFloat32
    }

    /// The width the depthwise convolution spans: the inner `x` plus the `B` and `C` streams.
    var convolutionWidth: Int { intermediateSize + 2 * groupCount * stateSize }

    /// The width the fused input projection emits: the gate, the convolution input, and one step per head.
    var inputProjectionWidth: Int { intermediateSize + convolutionWidth + headCount }

    /// The released `mistralai/Mamba-Codestral-7B-v0.1`.
    public static let codestral7B = NFKMLXMamba2Configuration()

    /// Reads a Mamba-2 geometry from a Hugging Face `config.json` dictionary. Rejects a config whose
    /// `model_type` is not `mamba2`.
    public static func configuration(fromHuggingFace json: [String: Any]) throws
        -> NFKMLXMamba2Configuration {
        let modelType = json["model_type"] as? String
        guard modelType == "mamba2" else {
            throw NFKMLXError.unsupportedConfiguration(
                "expected model_type mamba2, found \(modelType ?? "nil")")
        }
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? Int) ?? fallback }
        func float(_ key: String, _ fallback: Float) -> Float {
            if let d = json[key] as? Double { return Float(d) }
            if let i = json[key] as? Int { return Float(i) }
            return fallback
        }
        let hidden = int("hidden_size", 4096)
        let expand = int("expand", 2)
        let intermediate = (json["intermediate_size"] as? Int) ?? expand * hidden
        var lower: Float = 0.0
        var upper: Float = .infinity
        if let limit = json["time_step_limit"] as? [Any], limit.count == 2 {
            func asFloat(_ v: Any) -> Float? {
                if let d = v as? Double { return Float(d) }
                if let i = v as? Int { return Float(i) }
                return nil
            }
            lower = asFloat(limit[0]) ?? 0.0
            upper = asFloat(limit[1]) ?? .infinity
        }
        return NFKMLXMamba2Configuration(
            hiddenSize: hidden,
            layerCount: int("num_hidden_layers", 64),
            vocabularySize: int("vocab_size", 32768),
            rmsEpsilon: float("layer_norm_epsilon", 1e-5),
            intermediateSize: intermediate,
            headCount: int("num_heads", 128),
            headDimensions: int("head_dim", 64),
            stateSize: int("state_size", 128),
            groupCount: int("n_groups", 8),
            convolutionKernel: int("conv_kernel", 4),
            useConvolutionBias: (json["use_conv_bias"] as? Bool) ?? true,
            useProjectionBias: (json["use_bias"] as? Bool) ?? false,
            tiesWordEmbeddings: (json["tie_word_embeddings"] as? Bool) ?? false,
            timeStepLowerLimit: lower,
            timeStepUpperLimit: upper,
            residualInFloat32: (json["residual_in_fp32"] as? Bool) ?? true)
    }
}

/// Plain RMS normalization, `weight · rmsnorm(x)`. Mamba-2 scales by the weight directly, not by an
/// offset from one; the hybrid family's `(1 + w)` form does not apply here.
final class NFKMambaRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        epsilon = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let variance = mean(f * f, axis: -1, keepDims: true)
        let normalized = f * rsqrt(variance + epsilon)
        return weight * normalized.asType(x.dtype)
    }
}

/// The gated RMS normalization at the mixer's output: `weight · rmsnorm(y · silu(gate))`. The gate is
/// applied BEFORE the normalization, matching HF's `MambaRMSNormGated` (its `norm_before_gate` config
/// flag does not change this in the transformers implementation the port measures against).
///
/// @discussion The normalization is taken over `groups` equal spans of the last axis. The standard
/// Mamba-2 norm (Codestral) and Granite's leave `groups` at 1, normalizing over the whole width, which
/// matches HF's `MambaRMSNormGated`. Nemotron-H's mixer normalizes within `n_groups` groups (HF's
/// `Zamba2RMSNormGated` with `group_size = intermediate / n_groups`); the grouped path runs only when
/// `groups > 1`, so the ungrouped models compute byte-identically.
final class NFKMambaRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float
    let groups: Int

    init(dimensions: Int, eps: Float, groups: Int = 1) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        epsilon = eps
        self.groups = groups
        super.init()
    }

    /// - Parameters:
    ///   - x: the scan output, which the mixer hands over in float32 as the reference does.
    ///   - dtype: the type of the result. The reference forms `weight · normalized` in float32 and
    ///     rounds once, at the output projection's input.
    func callAsFunction(_ x: MLXArray, gate: MLXArray, dtype: DType) -> MLXArray {
        var f = x.asType(.float32)
        let g = gate.asType(.float32)
        f = f * (g * sigmoid(g))
        let normalized: MLXArray
        if groups > 1 {
            let shape = f.shape
            let grouped = f.reshaped(shape.dropLast() + [groups, shape[shape.count - 1] / groups])
            let variance = mean(grouped * grouped, axis: -1, keepDims: true)
            normalized = (grouped * rsqrt(variance + epsilon)).reshaped(shape)
        } else {
            let variance = mean(f * f, axis: -1, keepDims: true)
            normalized = f * rsqrt(variance + epsilon)
        }
        return (weight.asType(.float32) * normalized).asType(dtype)
    }
}

/// The mixer's depthwise causal convolution over `[B, L, C]`, a real submodule so its `weight` and
/// `bias` route from the checkpoint's `conv1d.weight` / `conv1d.bias`. Each output position sums the
/// kernel window ending at it, per channel, followed by SiLU. The released weight is `[C, 1, K]`, held
/// here `[C, K]` (the loader squeezes it).
final class NFKMambaDepthwiseConv1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray                     // [C, K]
    @ParameterInfo(key: "bias") var bias: MLXArray                         // [C]
    let kernel: Int

    init(channels: Int, kernel: Int) {
        self.kernel = kernel
        _weight.wrappedValue = MLXArray.zeros([channels, kernel])
        _bias.wrappedValue = MLXArray.zeros([channels])
        super.init()
    }

    /// In half precision the window sums in float32 and rounds once, then the SiLU rounds once, as
    /// torch's `Conv1d` and `silu` do.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let length = x.dim(1)
        let reduced = NFKReferenceRounding.isReduced(x)
        let input = reduced ? x.asType(.float32) : x
        // Left-pad the time axis so position t reads positions t-(kernel-1) ... t.
        let padded = padded(input, widths: [.init((0, 0)), .init((kernel - 1, 0)), .init((0, 0))])
        let taps = reduced ? weight.asType(.float32) : weight
        var accumulator = (reduced ? bias.asType(.float32) : bias).reshaped([1, 1, -1])
        for k in 0 ..< kernel {
            let slice = padded[0..., k ..< (k + length), 0...]
            accumulator = accumulator + slice * taps[0..., k].reshaped([1, 1, -1])
        }
        guard reduced else { return accumulator * sigmoid(accumulator) }
        return NFKReferenceRounding.silu(accumulator.asType(x.dtype))
    }
}

/// The Mamba-2 mixer: the fused input projection, the depthwise causal convolution over `x`/`B`/`C`,
/// the selective state-space scan, the gated normalization, and the output projection.
final class NFKMLXMamba2Mixer: Module {
    let config: NFKMLXMamba2Configuration

    @ModuleInfo(key: "in_proj") var inProjection: Linear
    @ModuleInfo(key: "conv1d") var conv1d: NFKMambaDepthwiseConv1d
    @ModuleInfo(key: "out_proj") var outProjection: Linear
    @ModuleInfo(key: "norm") var norm: NFKMambaRMSNormGated

    @ParameterInfo(key: "A_log") var aLog: MLXArray                        // [heads]
    @ParameterInfo(key: "D") var d: MLXArray                               // [heads]
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray                    // [heads]

    init(_ config: NFKMLXMamba2Configuration) {
        self.config = config
        _inProjection.wrappedValue = Linear(config.hiddenSize, config.inputProjectionWidth,
                                            bias: config.useProjectionBias)
        _conv1d.wrappedValue = NFKMambaDepthwiseConv1d(channels: config.convolutionWidth,
                                                       kernel: config.convolutionKernel)
        _outProjection.wrappedValue = Linear(config.intermediateSize, config.hiddenSize,
                                             bias: config.useProjectionBias)
        _norm.wrappedValue = NFKMambaRMSNormGated(dimensions: config.intermediateSize,
                                                  eps: config.rmsEpsilon,
                                                  groups: config.gatedNormGroups)
        _aLog.wrappedValue = MLXArray.zeros([config.headCount])
        _d.wrappedValue = MLXArray.zeros([config.headCount])
        _dtBias.wrappedValue = MLXArray.zeros([config.headCount])
        super.init()
    }

    /// The selective state-space scan. `x` `[B, L, heads, headDim]`, `b`/`c` `[B, L, heads, state]`,
    /// `dt` `[B, L, heads]`, `a` `[heads]` (already negative). Returns `[B, L, heads, headDim]`.
    private func selectiveScan(x: MLXArray, b: MLXArray, c: MLXArray, dt: MLXArray,
                               a: MLXArray) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1), heads = x.dim(2), headDim = x.dim(3)
        let state = b.dim(3)
        // dA `[B, L, heads]` = exp(dt · A); the decay applied at each step before reading the state.
        let dA = exp(dt * a.reshaped([1, 1, heads]))
        // dBx `[B, L, heads, headDim, state]` = dt · outer(x, B): the input written into the state.
        let dBx = dt.reshaped([batch, length, heads, 1, 1])
            * x.reshaped([batch, length, heads, headDim, 1])
            * b.reshaped([batch, length, heads, 1, state])
        var current = MLXArray.zeros([batch, heads, headDim, state], dtype: x.dtype)
        var outputs = [MLXArray]()
        outputs.reserveCapacity(length)
        for t in 0 ..< length {
            let decay = dA[0..., t, 0...].reshaped([batch, heads, 1, 1])
            current = decay * current + dBx[0..., t, 0..., 0..., 0...]
            // y_t `[B, heads, headDim]` = (state · C_t) summed over the state axis.
            let ct = c[0..., t, 0..., 0...].reshaped([batch, heads, 1, state])
            outputs.append((current * ct).sum(axis: -1))
        }
        return stacked(outputs, axis: 1)
    }

    /// `hidden` `[B, L, hiddenSize]` → `[B, L, hiddenSize]`.
    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let batch = hidden.dim(0), length = hidden.dim(1)
        let heads = config.headCount, headDim = config.headDimensions
        let groups = config.groupCount, state = config.stateSize
        let intermediate = config.intermediateSize

        let projected = inProjection(hidden)                              // [B, L, inputProjectionWidth]
        let gate = projected[0..., 0..., 0 ..< intermediate]
        let convInput = projected[0..., 0..., intermediate ..< (intermediate + config.convolutionWidth)]
        let dtRaw = projected[0..., 0..., (intermediate + config.convolutionWidth)...]

        let convolved = conv1d(convInput)                                 // [B, L, convDim]
        var offset = 0
        let xPart = convolved[0..., 0..., offset ..< (offset + intermediate)]
        offset += intermediate
        let bPart = convolved[0..., 0..., offset ..< (offset + groups * state)]
        offset += groups * state
        let cPart = convolved[0..., 0..., offset ..< (offset + groups * state)]

        // dt = softplus(dt + dt_bias), clamped to the configured step limit. The reference forms it in
        // the model's own type, so a half-precision step rounds before the float32 scan reads it.
        var dt: MLXArray
        if NFKReferenceRounding.isReduced(dtRaw) {
            dt = NFKReferenceRounding.softplus(dtRaw + dtBias.asType(dtRaw.dtype).reshaped([1, 1, heads]))
                .asType(.float32)
        } else {
            dt = softplus(dtRaw.asType(.float32) + dtBias.asType(.float32).reshaped([1, 1, heads]))
        }
        dt = clip(dt, min: config.timeStepLowerLimit, max: config.timeStepUpperLimit)
        let a = -exp(aLog.asType(.float32))                               // [heads]

        let xHeads = xPart.reshaped([batch, length, heads, headDim]).asType(.float32)
        // B and C carry one value per group; each group spans heads/groups heads.
        let headsPerGroup = heads / groups
        func expandGroups(_ t: MLXArray) -> MLXArray {
            let expanded = broadcast(t.reshaped([batch, length, groups, 1, state]),
                                     to: [batch, length, groups, headsPerGroup, state])
            return expanded.reshaped([batch, length, heads, state]).asType(.float32)
        }
        let bHeads = expandGroups(bPart)
        let cHeads = expandGroups(cPart)

        var y = selectiveScan(x: xHeads, b: bHeads, c: cHeads, dt: dt, a: a)
        // The per-head skip connection D · x, added before the reshape.
        y = y + d.asType(.float32).reshaped([1, 1, heads, 1]) * xHeads
        let normalized = norm(y.reshaped([batch, length, intermediate]), gate: gate, dtype: hidden.dtype)
        return outProjection(normalized)
    }
}

/// One Mamba-2 block: `residual + mixer(norm(residual))`.
final class NFKMLXMamba2Block: Module {
    @ModuleInfo(key: "norm") var norm: NFKMambaRMSNorm
    @ModuleInfo(key: "mixer") var mixer: NFKMLXMamba2Mixer
    let residualInFloat32: Bool

    init(_ config: NFKMLXMamba2Configuration) {
        _norm.wrappedValue = NFKMambaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
        _mixer.wrappedValue = NFKMLXMamba2Mixer(config)
        residualInFloat32 = config.residualInFloat32
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let mixed = mixer(norm(hidden.asType(norm.weight.dtype)))
        return (residualInFloat32 ? hidden.asType(.float32) : hidden) + mixed
    }
}

/// The Mamba-2 backbone: the token embedding, the block stack, and the final normalization. A real
/// submodule so the checkpoint's `backbone.*` keys route into it.
final class NFKMLXMamba2Backbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKMLXMamba2Block]
    @ModuleInfo(key: "norm_f") var finalNorm: NFKMambaRMSNorm

    init(_ config: NFKMLXMamba2Configuration) {
        _embeddings.wrappedValue = Embedding(embeddingCount: config.vocabularySize,
                                             dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.layerCount).map { _ in NFKMLXMamba2Block(config) }
        _finalNorm.wrappedValue = NFKMambaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsEpsilon)
        super.init()
    }
}

/// The Mamba-2 language model: the backbone and the language-model head.
public final class NFKMLXMamba2Net: Module {
    let config: NFKMLXMamba2Configuration

    @ModuleInfo(key: "backbone") var backbone: NFKMLXMamba2Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: NFKMLXMamba2Configuration) {
        self.config = config
        _backbone.wrappedValue = NFKMLXMamba2Backbone(config)
        _lmHead.wrappedValue = config.tiesWordEmbeddings
            ? nil : Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    /// The post-final-norm hidden states, `[B, L, hiddenSize]`.
    public func hiddenStates(_ tokens: MLXArray) -> MLXArray {
        backbone.finalNorm(blockStates(tokens).last!)
    }

    /// The state entering each block, for per-seam parity isolation: `[embedding, block0, … , blockN-1]`,
    /// each `[B, L, hiddenSize]`, before the final normalization. Matches the oracle's `hidden.{i}`
    /// record entries, whose first is the embedding output and whose `i+1` is block `i`'s output.
    public func blockStates(_ tokens: MLXArray) -> [MLXArray] {
        var hidden = backbone.embeddings(tokens)
        var states = [hidden]
        for layer in backbone.layers {
            hidden = layer(hidden)
            states.append(hidden)
        }
        return states
    }

    /// The next-token logits over the whole sequence, `[B, L, vocab]`.
    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let hidden = hiddenStates(tokens)
        // A float32 residual stream is rounded to the head's type first, as the reference does.
        if let lmHead {
            return lmHead(hidden.asType(lmHead.weight.dtype))
        }
        return backbone.embeddings.asLinear(hidden.asType(backbone.embeddings.weight.dtype))
    }
}

/// Builders and the released-weight loader for the Mamba-2 decoder. A class rather than an enum so the
/// backend factory can expose an `@objc` entry (an `@objc` member needs a class).
@objc(NFKMLXMamba)
public final class NFKMLXMamba: NSObject {
    /// Builds a Mamba-2 network from a geometry.
    public static func makeNet(_ config: NFKMLXMamba2Configuration) -> NFKMLXMamba2Net {
        NFKMLXMamba2Net(config)
    }

    /// Reads a Mamba-2 geometry from a release directory's `config.json`.
    ///
    /// The released config writes `time_step_limit` as `[0.0, Infinity]`. Python's `json` accepts the
    /// bare `Infinity` token; Foundation's `JSONSerialization` rejects it, so the non-finite literals
    /// are replaced first with `1e39`, a finite `Double` that rounds to an infinite `Float`.
    public static func configuration(fromDirectory directory: URL) throws
        -> NFKMLXMamba2Configuration {
        let url = directory.appendingPathComponent("config.json")
        var text = String(decoding: try Data(contentsOf: url), as: UTF8.self)
        for (token, replacement) in [("-Infinity", "-1e39"), ("Infinity", "1e39"), ("NaN", "0")] {
            text = text.replacingOccurrences(of: token, with: replacement)
        }
        let json = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] ?? [:]
        return try NFKMLXMamba2Configuration.configuration(fromHuggingFace: json)
    }

    /// Loads a released Mamba-2 decoder from its directory, following the shard index.
    ///
    /// The checkpoint keys mirror this module's own, so no renaming is needed. The one transform is the
    /// depthwise convolution weight, stored `[channels, 1, kernel]` and held here `[channels, kernel]`.
    /// The squeeze is guarded on the released 3-D shape, so a fine-tuned 2-D save round-trips unchanged.
    public static func loadWeights(into net: NFKMLXMamba2Net, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let tied = net.lmHead == nil
        let read = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) {
            tied && $0.hasPrefix("lm_head.") ? nil : $0
        }
        let merged = read.map { name, value in
            (name, name.hasSuffix("conv1d.weight") && value.ndim == 3
                 ? value.reshaped([value.dim(0), value.dim(2)]) : value)
        }
        try NFKMLXWeights.apply(merged, to: net)
    }
}
