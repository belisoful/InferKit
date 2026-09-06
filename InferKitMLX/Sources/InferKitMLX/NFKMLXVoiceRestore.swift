// VoiceRestore (skirdey/voicerestore, MIT): a ~301M flow-matching (CFM) universal speech restorer,
// text-free. This file is the flow-matching TRANSFORMER (E2-TTS-derived, NOT F5). The BigVGAN vocoder
// and the CFM/midpoint sampler are separate increments; this is the velocity net v_θ(x_t, t | cond).
//
// Grounded on the released source read 2026-09-05: `voice_restore.py` (`VoiceRestore` + the vendored
// E2-TTS `Transformer`), `model.py`. The condition is ADDITIVE and frame-aligned:
// `x = proj_in(x_t) + cond_proj(degraded_mel)`. Deps pin `x-transformers==1.34.0`,
// `gateloop-transformer==0.2.5`, `rotary-embedding-torch==0.8.3`.
//
// SCAFFOLD STATUS: the transformer scaffold — proj_in/cond_proj/to_pred, rotary attention, RMSNorm,
// AdaptiveRMSNorm, AdaLNZero, the U-net concat skips, GEGLU feed-forward — is written from source.
// `FLAGGED:` marks the pieces needing verbatim confirmation at implement time: the `SimpleGateLoopLayer`
// recurrence (the one research-grade piece — left a marked stub, NOT fabricated), the `time_cond_mlp`
// shape, the AdaptiveRMSNorm `(1+γ)` convention, the AdaLNZero gate form, and the FF `glu` flag. Pinned
// against `run_reference.py voicerestore` (fp32 CPU) — oracle + gateloop + test are the next increment.

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
    public var ropeTheta: Float

    public init(numChannels: Int = 100, dim: Int = 768, depth: Int = 20, heads: Int = 16,
                dimHead: Int = 64, ffMult: Int = 4, ropeTheta: Float = 10000) {
        self.numChannels = numChannels
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dimHead = dimHead
        self.ffMult = ffMult
        self.ropeTheta = ropeTheta
    }
}

// MARK: - Norms / modulation

/// `AdaptiveRMSNorm` (x-transformers): RMS-normalize `x`, scale by `(1 + gamma)` where `gamma` is a
/// bias-free linear of the time conditioning. FLAGGED: confirm the `(1+γ)` vs `γ` convention at 1.34.0.
final class NFKVRAdaptiveRMSNorm: Module {
    @ModuleInfo(key: "to_gamma") var toGamma: Linear
    let scale: Float

    init(dim: Int, condDim: Int) {
        scale = powf(Float(dim), 0.5)
        _toGamma.wrappedValue = Linear(condDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cond: MLXArray) -> MLXArray {
        let normed = x * rsqrt((x * x).mean(axis: -1, keepDims: true) + 1e-5)
        let gamma = toGamma(cond).expandedDimensions(axis: 1)          // [B, 1, dim] over the sequence
        return normed * (1 + gamma)
    }
}

/// `AdaLNZero` (DiT-style, zero-init gate): scales a sublayer output by a learned gate of the time
/// conditioning. FLAGGED: confirm whether the gate is raw or `sigmoid`, and the zero-init offset.
final class NFKVRAdaLNZero: Module {
    @ModuleInfo(key: "to_gamma") var toGamma: Linear

    init(dim: Int, condDim: Int) {
        _toGamma.wrappedValue = Linear(condDim, dim)
    }

    func callAsFunction(_ x: MLXArray, cond: MLXArray) -> MLXArray {
        let gate = toGamma(cond).expandedDimensions(axis: 1)          // [B, 1, dim]
        return x * gate
    }
}

// MARK: - Rotary (rotary-embedding-torch, RotaryEmbedding(dim_head=64))

/// Rotary over the full head dimension, adjacent-pair (`view_as_complex`) convention, θ = 10000.
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

/// x-transformers `Attention(dim, heads, dim_head)`: separate q/k/v projections to `heads·dim_head`,
/// rotary on q/k, scaled dot-product, then `to_out`.
final class NFKVRAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    let heads: Int
    let dimHead: Int
    let rotary: NFKVRRotary

    init(dim: Int, heads: Int, dimHead: Int, rotary: NFKVRRotary) {
        self.heads = heads
        self.dimHead = dimHead
        self.rotary = rotary
        let inner = heads * dimHead
        _toQ.wrappedValue = Linear(dim, inner, bias: false)
        _toK.wrappedValue = Linear(dim, inner, bias: false)
        _toV.wrappedValue = Linear(dim, inner, bias: false)
        _toOut.wrappedValue = Linear(inner, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        func heads4(_ t: MLXArray) -> MLXArray { t.reshaped([b, s, heads, dimHead]).transposed(0, 2, 1, 3) }
        let q = rotary(heads4(toQ(x)))
        let k = rotary(heads4(toK(x)))
        let v = heads4(toV(x))
        let scores = matmul(q, k.transposed(0, 1, 3, 2)) / sqrt(Float(dimHead))
        let out = matmul(softmax(scores, axis: -1), v)                 // [B, H, S, dh]
        return toOut(out.transposed(0, 2, 1, 3).reshaped([b, s, heads * dimHead]))
    }
}

/// x-transformers `FeedForward(dim, mult)` with GEGLU. FLAGGED: confirm the `glu` flag at 1.34.0.
final class NFKVRFeedForward: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(dim: Int, mult: Int) {
        let inner = dim * mult
        _projIn.wrappedValue = Linear(dim, inner * 2)                  // GEGLU: value + gate
        _projOut.wrappedValue = Linear(inner, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(projIn(x), parts: 2, axis: -1)
        return projOut(parts[0] * gelu(parts[1]))
    }
}

/// `SimpleGateLoopLayer` (gateloop-transformer 0.2.5): a data-dependent gated linear recurrence run
/// before attention (E2-TTS's positional/mixing layer).
///
/// FLAGGED — NOT IMPLEMENTED: the 0.2.5 internals (gate parametrization real/complex, the `sigmoid` on
/// the forget gate, head/dim_inner, the associative-scan) were not byte-confirmable and are the one
/// research-grade piece. This is a marked identity pass-through so the transformer compiles and every
/// OTHER seam can be validated; the recurrence must be filled from the tagged `gateloop_transformer`
/// 0.2.5 source during the parity batch (a sequential cumulative recurrence over T is exact in MLX).
final class NFKVRGateLoop: Module {
    init(dim: Int) { super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }   // FLAGGED stub — see class doc.
}

/// One transformer block: gateloop mixing → adaptive-norm attention (adaLN-zero gated) → adaptive-norm
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
        _attn.wrappedValue = NFKVRAttention(dim: config.dim, heads: config.heads, dimHead: config.dimHead, rotary: rotary)
        _attnGate.wrappedValue = NFKVRAdaLNZero(dim: config.dim, condDim: config.dim)
        _ffNorm.wrappedValue = NFKVRAdaptiveRMSNorm(dim: config.dim, condDim: config.dim)
        _ff.wrappedValue = NFKVRFeedForward(dim: config.dim, mult: config.ffMult)
        _ffGate.wrappedValue = NFKVRAdaLNZero(dim: config.dim, condDim: config.dim)
    }

    func callAsFunction(_ x0: MLXArray, cond: MLXArray, skip: MLXArray?) -> MLXArray {
        var x = x0
        if let skipProj, let skip { x = skipProj(concatenated([x, skip], axis: -1)) }
        x = gateloop(x)
        x = x + attnGate(attn(attnNorm(x, cond: cond)), cond: cond)
        x = x + ffGate(ff(ffNorm(x, cond: cond)), cond: cond)
        return x
    }
}

// MARK: - The transformer + VoiceRestore net

/// The E2-TTS transformer: a time-conditioning MLP, `depth` blocks with U-net concat skips (first half
/// pushed, second half popped), and a final RMSNorm.
final class NFKVRTransformer: Module {
    @ModuleInfo(key: "time_cond_mlp") var timeMLP: Sequential
    @ModuleInfo(key: "layers") var layers: [NFKVRBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm
    let half: Int

    init(_ config: NFKMLXVoiceRestoreConfiguration) {
        // FLAGGED: time_cond_mlp is Rearrange→Linear→SiLU on the scalar timestep in this pinned version
        // (no RandomFourierEmbed). Confirm the exact input width (scalar vs sinusoidal embedding).
        _timeMLP.wrappedValue = Sequential(layers: [Linear(1, config.dim), SiLU()])
        let rotary = NFKVRRotary(dim: config.dimHead, theta: config.ropeTheta)
        half = config.depth / 2
        _layers.wrappedValue = (0 ..< config.depth).map { NFKVRBlock(config, hasSkip: $0 >= config.depth / 2, rotary: rotary) }
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.dim)
    }

    /// `x [B, S, dim]`, scalar `times [B]` → `[B, S, dim]`.
    func callAsFunction(_ x0: MLXArray, times: MLXArray) -> MLXArray {
        let cond = timeMLP(times.reshaped([times.dim(0), 1]))          // [B, dim]
        var x = x0
        var skips = [MLXArray]()
        for (i, block) in layers.enumerated() {
            if i < half {
                x = block(x, cond: cond, skip: nil)
                skips.append(x)
            } else {
                x = block(x, cond: cond, skip: skips.removeLast())
            }
        }
        return finalNorm(x)
    }
}

/// The VoiceRestore velocity network. `x_t [B, S, 100]` (the flow state in mel space) and the degraded
/// mel condition `[B, S, 100]` → the velocity `[B, S, 100]`.
public final class NFKMLXVoiceRestore: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "transformer") var transformer: NFKVRTransformer
    @ModuleInfo(key: "to_pred") var toPred: Linear

    public init(_ config: NFKMLXVoiceRestoreConfiguration) {
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
}

// MARK: - Registration and weight loading

/// Registration and weight loading for the VoiceRestore transformer.
@objc(NFKMLXVoiceRestore_Factory)
public final class NFKMLXVoiceRestoreFactory: NSObject {
    @objc public static let modelName = "voicerestore"

    static func makeNet(_ config: NFKMLXVoiceRestoreConfiguration = .init()) -> NFKMLXVoiceRestore {
        NFKMLXVoiceRestore(config)
    }

    /// Loads a released VoiceRestore checkpoint (top key `model_state_dict`, no EMA) into the net. Linear
    /// weights are 2-D and pass through. FLAGGED: the x-transformers `Attention`/`FeedForward` and
    /// `time_cond_mlp` submodule key names need reconciling against the checkpoint at 1.34.0 (this remap
    /// is the first approximation), and the gateloop keys are carried once its layer is implemented.
    static func loadWeights(into net: NFKMLXVoiceRestore, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped: [(String, MLXArray)] = checkpoint.arrays.compactMap { key, value in
            guard let name = remapReferenceKey(key) else { return nil }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    static func remapReferenceKey(_ key: String) -> String? {
        if key.contains(".rotary") || key.hasSuffix(".freqs") { return nil }
        var name = key
        // x-transformers Attention `to_out` is a Sequential(Linear, ...): to_out.0 → to_out.
        name = name.replacingOccurrences(of: "to_out.0.", with: "to_out.")
        // FeedForward Sequential: ff.0.proj (GEGLU proj_in) / ff.2 (proj_out) — FLAGGED indices.
        name = name.replacingOccurrences(of: "ff.ff.0.proj.", with: "ff.proj_in.")
        name = name.replacingOccurrences(of: "ff.ff.2.", with: "ff.proj_out.")
        return name
    }

    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { _ in throw NFKMLXError.unsupportedInput }
        // NOTE: VoiceRestore needs the BigVGAN vocoder + CFM sampler + the SimpleGateLoopLayer before a
        // runnable backend; the transformer registers for parity harness use, the rest is the next
        // increment.
    }
}
