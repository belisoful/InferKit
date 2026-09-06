// VoiceRestore (skirdey/voicerestore, MIT): a ~301M flow-matching (CFM) universal speech restorer,
// text-free (fixes noise + reverb + clipping + band-limiting together). This file is the flow-matching
// TRANSFORMER — the velocity net v_θ(x_t, t | cond=degraded_mel). The BigVGAN vocoder is
// `NFKMLXVoiceRestoreVocoder`; the CFM/midpoint sampler and the mel front end are in
// `NFKMLXVoiceRestoreBackend`.
//
// Grounded on the released source read 2026-09-06: `voice_restore.py` (the vendored E2-TTS `Transformer`),
// `model.py`, and the pinned deps `x-transformers==1.34.0` / `gateloop-transformer==0.2.5` /
// `rotary-embedding-torch==0.8.3`. The condition is ADDITIVE and frame-aligned:
// `x = proj_in(x_t) + cond_proj(degraded_mel)`. The sequence carries 32 learned REGISTER tokens prepended
// before the blocks (unpacked after), plus an absolute positional embedding added to the mel frames.

import Foundation
import InferKit
import MLX
import MLXNN

/// The VoiceRestore transformer configuration (`model.py` instantiation).
public struct NFKMLXVoiceRestoreConfiguration: Sendable {
    public var numChannels: Int          // mel bands, 100
    public var dim: Int                  // 768
    public var depth: Int                // 20
    public var heads: Int                // 16
    public var dimHead: Int              // 64
    public var ffMult: Int               // 4
    public var maxSeqLen: Int            // 1024 (abs_pos_emb rows)
    public var numRegisters: Int         // 32
    public var ropeTheta: Float
    public var softclamp: Float          // attention logit soft-clamp value, 50

    public init(numChannels: Int = 100, dim: Int = 768, depth: Int = 20, heads: Int = 16,
                dimHead: Int = 64, ffMult: Int = 4, maxSeqLen: Int = 2000, numRegisters: Int = 32,
                ropeTheta: Float = 10000, softclamp: Float = 50) {
        self.numChannels = numChannels
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dimHead = dimHead
        self.ffMult = ffMult
        self.maxSeqLen = maxSeqLen
        self.numRegisters = numRegisters
        self.ropeTheta = ropeTheta
        self.softclamp = softclamp
    }
}

// MARK: - Norms / modulation

/// `AdaptiveRMSNorm` (x-transformers 1.34.0): `F.normalize(x)·√dim·(1 + gamma)`, `gamma` a bias-free
/// linear of the time conditioning.
final class NFKVRAdaptiveRMSNorm: Module {
    @ModuleInfo(key: "to_gamma") var toGamma: Linear
    let scale: Float

    init(dim: Int, condDim: Int) {
        scale = powf(Float(dim), 0.5)
        _toGamma.wrappedValue = Linear(condDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cond: MLXArray) -> MLXArray {
        let normed = x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-12)   // F.normalize(x, dim=-1)
        let gamma = toGamma(cond).expandedDimensions(axis: 1)                   // [B, 1, dim]
        return normed * scale * (1 + gamma)
    }
}

/// `AdaLNZero` (voice_restore.py): a zero-init, `-2`-biased sigmoid gate of the time conditioning applied
/// to a sublayer output (`x * sigmoid(to_gamma(condition))`).
final class NFKVRAdaLNZero: Module {
    @ModuleInfo(key: "to_gamma") var toGamma: Linear

    init(dim: Int, condDim: Int) {
        _toGamma.wrappedValue = Linear(condDim, dim)
    }

    func callAsFunction(_ x: MLXArray, cond: MLXArray) -> MLXArray {
        let gate = sigmoid(toGamma(cond)).expandedDimensions(axis: 1)          // [B, 1, dim]
        return x * gate
    }
}

/// The gateloop's own RMSNorm (`gateloop_transformer.RMSNorm`): `F.normalize(x)·√dim·gamma`.
final class NFKVRGateLoopNorm: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    let scale: Float

    init(dim: Int) {
        scale = powf(Float(dim), 0.5)
        _gamma.wrappedValue = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-12) * scale * gamma
    }
}

// MARK: - Rotary (rotary-embedding-torch / x-transformers RotaryEmbedding, dim_head=64)

/// Rotary over the full head dimension, adjacent-pair (`rotate_half` on `(d r)` with `r=2`) convention,
/// θ = 10000, positions 0..S-1 along the sequence.
struct NFKVRRotary {
    let dim: Int
    let theta: Float
    init(dim: Int, theta: Float = 10000) { self.dim = dim; self.theta = theta }

    /// `t` `[B, H, S, dim]` → rotated by absolute position along S.
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let s = t.dim(2)
        let pairs = dim / 2
        var cosT = [Float](repeating: 0, count: s * pairs)
        var sinT = [Float](repeating: 0, count: s * pairs)
        for pos in 0 ..< s {
            for k in 0 ..< pairs {
                let angle = Float(pos) / powf(theta, Float(2 * k) / Float(dim))
                cosT[pos * pairs + k] = cosf(angle)
                sinT[pos * pairs + k] = sinf(angle)
            }
        }
        let cos = MLXArray(cosT, [1, 1, s, pairs, 1])
        let sin = MLXArray(sinT, [1, 1, s, pairs, 1])
        let paired = t.reshaped([t.dim(0), t.dim(1), s, pairs, 2])
        let a = paired[.ellipsis, 0 ..< 1]
        let b = paired[.ellipsis, 1 ..< 2]
        return concatenated([a * cos - b * sin, a * sin + b * cos], axis: -1)
            .reshaped([t.dim(0), t.dim(1), s, dim])
    }
}

// MARK: - Attention / FF / gateloop

/// x-transformers `Attention(dim, heads, dim_head, gate_value_heads=True, softclamp_logits=True)`:
/// separate q/k/v projections, rotary on q/k, soft-clamped logits, a per-head sigmoid value gate
/// (`to_v_head_gate`), then `to_out`.
final class NFKVRAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "to_v_head_gate") var toVHeadGate: Linear
    let heads: Int
    let dimHead: Int
    let softclamp: Float
    let rotary: NFKVRRotary

    init(dim: Int, heads: Int, dimHead: Int, softclamp: Float, rotary: NFKVRRotary) {
        self.heads = heads
        self.dimHead = dimHead
        self.softclamp = softclamp
        self.rotary = rotary
        let inner = heads * dimHead
        _toQ.wrappedValue = Linear(dim, inner, bias: false)
        _toK.wrappedValue = Linear(dim, inner, bias: false)
        _toV.wrappedValue = Linear(dim, inner, bias: false)
        _toOut.wrappedValue = Linear(inner, dim, bias: false)
        _toVHeadGate.wrappedValue = Linear(dim, heads)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        func heads4(_ t: MLXArray) -> MLXArray { t.reshaped([b, s, heads, dimHead]).transposed(0, 2, 1, 3) }
        let q = rotary(heads4(toQ(x)))
        let k = rotary(heads4(toK(x)))
        let v = heads4(toV(x))
        var sim = matmul(q, k.transposed(0, 1, 3, 2)) / sqrt(Float(dimHead))
        sim = tanh(sim / softclamp) * softclamp                                // softclamp_logits
        var out = matmul(softmax(sim, axis: -1), v)                            // [B, H, S, dh]
        let headGate = sigmoid(toVHeadGate(x)).transposed(0, 2, 1).expandedDimensions(axis: 3)  // [B, H, S, 1]
        out = out * headGate
        return toOut(out.transposed(0, 2, 1, 3).reshaped([b, s, heads * dimHead]))
    }
}

/// x-transformers `FeedForward(dim, mult, glu=True)`: a GEGLU (`value · gelu(gate)`) then a linear.
final class NFKVRFeedForward: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear                            // GLU.proj
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(dim: Int, mult: Int) {
        let inner = dim * mult
        _projIn.wrappedValue = Linear(dim, inner * 2)                          // GLU: value + gate
        _projOut.wrappedValue = Linear(inner, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(projIn(x), parts: 2, axis: -1)
        return projOut(parts[0] * gelu(parts[1]))
    }
}

/// `SimpleGateLoopLayer` (gateloop-transformer 0.2.5): an RMSNorm, a bias-free `Linear(dim, 3·dim)` into
/// q / kv / a, a sigmoid forget gate `a`, a per-channel first-order linear recurrence
/// `h_t = a_t·h_{t-1} + kv_t`, and the output `q_t·h_t`. The sequential scan is exact (MLX has no
/// associative scan). Run RESIDUALLY in the block (`x = gateloop(x) + x`).
final class NFKVRGateLoop: Module {
    @ModuleInfo(key: "norm") var norm: NFKVRGateLoopNorm
    @ModuleInfo(key: "to_qkva") var toQKVA: Linear

    init(dim: Int) {
        _norm.wrappedValue = NFKVRGateLoopNorm(dim: dim)
        _toQKVA.wrappedValue = Linear(dim, dim * 3, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let qkva = toQKVA(norm(x))                                            // [B, S, 3·dim]
        let parts = split(qkva, parts: 3, axis: -1)                           // q, kv, a
        let q = parts[0], kv = parts[1], a = sigmoid(parts[2])
        let s = x.dim(1)
        var h: MLXArray = kv[0..., 0, 0...]                                   // h_0 = a_0·0 + kv_0
        var outs = [q[0..., 0, 0...] * h]
        for t in 1 ..< s {
            h = a[0..., t, 0...] * h + kv[0..., t, 0...]
            outs.append(q[0..., t, 0...] * h)
        }
        return stacked(outs, axis: 1)                                        // [B, S, dim]
    }
}

/// One transformer block: a residual gateloop → adaptive-norm attention (adaLN-zero gated) → adaptive-norm
/// GEGLU feed-forward (adaLN-zero gated). `skipProj` is present only on the later-half blocks.
final class NFKVRBlock: Module {
    @ModuleInfo(key: "gateloop") var gateloop: NFKVRGateLoop
    @ModuleInfo(key: "skip_proj") var skipProj: Linear?
    @ModuleInfo(key: "attn_norm") var attnNorm: NFKVRAdaptiveRMSNorm
    @ModuleInfo(key: "attn") var attn: NFKVRAttention
    @ModuleInfo(key: "attn_adaln_zero") var attnGate: NFKVRAdaLNZero
    @ModuleInfo(key: "ff_norm") var ffNorm: NFKVRAdaptiveRMSNorm
    @ModuleInfo(key: "ff") var ff: NFKVRFeedForward
    @ModuleInfo(key: "ff_adaln_zero") var ffGate: NFKVRAdaLNZero

    init(_ config: NFKMLXVoiceRestoreConfiguration, hasSkip: Bool, rotary: NFKVRRotary) {
        _gateloop.wrappedValue = NFKVRGateLoop(dim: config.dim)
        _skipProj.wrappedValue = hasSkip ? Linear(config.dim * 2, config.dim, bias: false) : nil
        _attnNorm.wrappedValue = NFKVRAdaptiveRMSNorm(dim: config.dim, condDim: config.dim)
        _attn.wrappedValue = NFKVRAttention(dim: config.dim, heads: config.heads, dimHead: config.dimHead,
                                            softclamp: config.softclamp, rotary: rotary)
        _attnGate.wrappedValue = NFKVRAdaLNZero(dim: config.dim, condDim: config.dim)
        _ffNorm.wrappedValue = NFKVRAdaptiveRMSNorm(dim: config.dim, condDim: config.dim)
        _ff.wrappedValue = NFKVRFeedForward(dim: config.dim, mult: config.ffMult)
        _ffGate.wrappedValue = NFKVRAdaLNZero(dim: config.dim, condDim: config.dim)
    }

    func callAsFunction(_ x0: MLXArray, cond: MLXArray, skip: MLXArray?) -> MLXArray {
        var x = x0
        if let skipProj, let skip { x = skipProj(concatenated([x, skip], axis: -1)) }
        x = gateloop(x) + x
        x = x + attnGate(attn(attnNorm(x, cond: cond)), cond: cond)
        x = x + ffGate(ff(ffNorm(x, cond: cond)), cond: cond)
        return x
    }
}

// MARK: - The transformer + VoiceRestore net

/// The E2-TTS transformer: an absolute positional embedding on the mel frames, 32 prepended learned
/// register tokens, a time-conditioning MLP, `depth` blocks with U-net concat skips (first half pushed,
/// second half popped), and a final RMSNorm. The registers are unpacked before the output.
final class NFKVRTransformer: Module {
    @ModuleInfo(key: "abs_pos_emb") var absPosEmb: Embedding
    @ParameterInfo(key: "registers") var registers: MLXArray                  // [numRegisters, dim]
    @ModuleInfo(key: "time_cond_mlp") var timeLinear: Linear                  // Sequential(Rearrange, Linear, SiLU)
    @ModuleInfo(key: "layers") var layers: [NFKVRBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm
    let config: NFKMLXVoiceRestoreConfiguration
    let half: Int

    init(_ config: NFKMLXVoiceRestoreConfiguration) {
        self.config = config
        _absPosEmb.wrappedValue = Embedding(embeddingCount: config.maxSeqLen, dimensions: config.dim)
        _registers.wrappedValue = MLXArray.zeros([config.numRegisters, config.dim])
        _timeLinear.wrappedValue = Linear(1, config.dim)
        let rotary = NFKVRRotary(dim: config.dimHead, theta: config.ropeTheta)
        half = config.depth / 2
        _layers.wrappedValue = (0 ..< config.depth).map { NFKVRBlock(config, hasSkip: $0 >= config.depth / 2, rotary: rotary) }
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.dim)
    }

    /// `x [B, N, dim]` (the projected mel frames), scalar `times [B]` → `[B, N, dim]`.
    func callAsFunction(_ x0: MLXArray, times: MLXArray) -> MLXArray {
        let (b, n) = (x0.dim(0), x0.dim(1))
        let cond = silu(timeLinear(times.reshaped([b, 1])))                   // [B, dim]

        // Absolute positional embedding on the mel frames, then prepend the register tokens.
        let positions = MLXArray((0 ..< n).map { Int32($0) })
        var x = x0 + absPosEmb(positions)
        let regs = broadcast(registers.expandedDimensions(axis: 0), to: [b, config.numRegisters, config.dim])
        x = concatenated([regs, x], axis: 1)                                 // [B, R+N, dim]

        var skips = [MLXArray]()
        for (i, block) in layers.enumerated() {
            if i < half {
                skips.append(x)
                x = block(x, cond: cond, skip: nil)
            } else {
                x = block(x, cond: cond, skip: skips.removeLast())
            }
        }
        x = finalNorm(x)
        return x[0..., config.numRegisters..., 0...]                         // unpack (drop registers)
    }
}

/// The VoiceRestore velocity network. `x_t [B, N, 100]` (the flow state in mel space) and the degraded
/// mel condition `[B, N, 100]` → the velocity `[B, N, 100]`.
public final class NFKMLXVoiceRestore: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "transformer") var transformer: NFKVRTransformer
    @ModuleInfo(key: "to_pred") var toPred: Linear
    let config: NFKMLXVoiceRestoreConfiguration

    public init(_ config: NFKMLXVoiceRestoreConfiguration) {
        self.config = config
        _projIn.wrappedValue = Linear(config.numChannels, config.dim)
        _condProj.wrappedValue = Linear(config.numChannels, config.dim)
        _transformer.wrappedValue = NFKVRTransformer(config)
        _toPred.wrappedValue = Linear(config.dim, config.numChannels)
    }

    /// `cond` nil is the classifier-free-guidance unconditional branch (the additive condition term is
    /// dropped, leaving noise + time).
    public func callAsFunction(_ x: MLXArray, times: MLXArray, cond: MLXArray?) -> MLXArray {
        var h = projIn(x)
        if let cond { h = h + condProj(cond) }
        return toPred(transformer(h, times: times))
    }

    /// Classifier-free guidance: `pred + (pred − null)·cfgStrength` (skip when `cfgStrength < 1e-5`).
    public func guided(_ x: MLXArray, times: MLXArray, cond: MLXArray, cfgStrength: Float) -> MLXArray {
        let pred = self(x, times: times, cond: cond)
        guard cfgStrength >= 1e-5 else { return pred }
        let null = self(x, times: times, cond: nil)
        return pred + (pred - null) * cfgStrength
    }
}

// MARK: - Registration and weight loading

/// Registration and weight loading for the VoiceRestore transformer.
@objc(NFKMLXVoiceRestore_Factory)
public final class NFKMLXVoiceRestoreFactory: NSObject {
    @objc public static let modelName = "voicerestore"

    static func makeNet(_ config: NFKMLXVoiceRestoreConfiguration = .init()) -> NFKMLXVoiceRestore {
        NFKMLXVoiceRestore(config)
    }

    /// Loads a released VoiceRestore checkpoint (top key `model_state_dict`, no EMA). The block ModuleList
    /// indices 0..7 become the named submodules; the x-transformers `Sequential` wrappers (`time_cond_mlp`,
    /// GLU feed-forward, the gateloop `to_qkva`) drop their positional indices; the `final_norm` `g` and
    /// buffers are renamed / dropped. All weights are 2-D and pass through with no transpose.
    static func loadWeights(into net: NFKMLXVoiceRestore, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped: [(String, MLXArray)] = checkpoint.arrays.compactMap { key, value in
            guard let name = remapReferenceKey(key) else { return nil }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    private static let blockSlots = ["gateloop", "skip_proj", "attn_norm", "attn", "attn_adaln_zero",
                                     "ff_norm", "ff", "ff_adaln_zero"]

    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasSuffix(".inv_freq") || key.hasSuffix(".freqs") { return nil }
        var name = key
        // The transformer's block ModuleList: transformer.layers.{i}.{0..7}. -> named submodule.
        if let range = name.range(of: #"transformer\.layers\.(\d+)\.(\d)\."#, options: .regularExpression) {
            let slice = name[range]
            let comps = slice.split(separator: ".")                          // [transformer, layers, i, slot, ""]
            if let slot = Int(comps[3]), slot < blockSlots.count {
                name.replaceSubrange(range, with: "transformer.layers.\(comps[2]).\(blockSlots[slot]).")
            }
        }
        // Sequential wrappers drop their positional index.
        name = name.replacingOccurrences(of: "time_cond_mlp.1.", with: "time_cond_mlp.")   // Rearrange, Linear, SiLU
        name = name.replacingOccurrences(of: ".gateloop.to_qkva.0.", with: ".gateloop.to_qkva.")
        name = name.replacingOccurrences(of: ".ff.ff.0.proj.", with: ".ff.proj_in.")       // GLU
        name = name.replacingOccurrences(of: ".ff.ff.2.", with: ".ff.proj_out.")
        // x-transformers RMSNorm parameter is `g`; the final norm is MLXNN RMSNorm (`weight`).
        name = name.replacingOccurrences(of: "final_norm.g", with: "final_norm.weight")
        return name
    }

    /// Registers `voicerestore`. The registry passes a DIRECTORY holding the transformer checkpoint
    /// (`voicerestore.safetensors` or `pytorch_model.bin`) and `bigvgan_generator.pt`; a nil URL builds
    /// random-weight nets (the gallery path).
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in try backend(directoryURL: url) }
    }
}
