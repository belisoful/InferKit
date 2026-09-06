// MossFormer2 SE 48K (modelscope/ClearerVoice-Studio, Apache-2.0): 48 kHz speech enhancement. A
// mask-predicting MossFormer2 backbone over a Kaldi-fbank + delta front end produces a real 961-bin
// magnitude mask, applied to the complex STFT (phase preserved) and inverted. This file is the shared
// MossFormer2 BACKBONE + the SE MaskNet; the SR port reuses the backbone (mel->mel) and adds a BigVGAN
// vocoder. The Kaldi-fbank front end and the mask/iSTFT back end are wired in the backend (a later
// increment); this file is the net.
//
// Grounded on the released source read 2026-09-05: `models/mossformer2_se/{mossformer2.py,
// mossformer2_block.py, fsmn.py, conv_module.py, layer_norm.py}`. Tensors flow in [B, S, C] (the
// reference runs the blocks batch-first over the sequence S; MLX's NLC convolutions match this without
// the reference's transposes).
//
// SCAFFOLD STATUS: structure and checkpoint key names mirror the released generator. Parity-sensitive
// choices are marked `PARITY:`; the two items the recon could not byte-confirm are marked `FLAGGED:`.
// Pinned against `run_reference.py mossformer2_se` (dither=0) on `last_best_checkpoint.pt` (the oracle
// and the seam test are the next increment).

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

/// The MossFormer2 SE 48K configuration (the released MaskNet geometry).
public struct NFKMLXMossFormer2Configuration: Sendable {
    public var sampleRate: Int
    public var fftLen: Int
    public var winLen: Int
    public var winInc: Int
    public var numMels: Int
    /// The 180-dim network input (fbank + Δ + ΔΔ over `numMels`).
    public var inChannels: Int
    public var dModel: Int
    public var numBlocks: Int
    public var groupSize: Int
    public var queryKeyDim: Int
    public var expansionFactor: Float
    public var fsmnLorder: Int
    public var fsmnHidden: Int
    /// The final mask width, `fftLen/2 + 1`.
    public var outChannelsFinal: Int
    /// The MaskNet's speaker count. The released SE net keeps the default 2 (its `conv1d_out` widens to
    /// `dModel·2`) and returns speaker 0, so this stays 2 for parity.
    public var numSpks: Int

    public var bins: Int { fftLen / 2 + 1 }

    public init(sampleRate: Int = 48000, fftLen: Int = 1920, winLen: Int = 1920, winInc: Int = 384,
                numMels: Int = 60, dModel: Int = 512, numBlocks: Int = 24, groupSize: Int = 256,
                queryKeyDim: Int = 128, expansionFactor: Float = 4, fsmnLorder: Int = 20,
                fsmnHidden: Int = 256) {
        self.sampleRate = sampleRate
        self.fftLen = fftLen
        self.winLen = winLen
        self.winInc = winInc
        self.numMels = numMels
        self.inChannels = numMels * 3
        self.dModel = dModel
        self.numBlocks = numBlocks
        self.groupSize = groupSize
        self.queryKeyDim = queryKeyDim
        self.expansionFactor = expansionFactor
        self.fsmnLorder = fsmnLorder
        self.fsmnHidden = fsmnHidden
        self.outChannelsFinal = fftLen / 2 + 1
        self.numSpks = 2
    }
}

// MARK: - Norms

/// `ScaleNorm` (lucidrains): `x / (‖x‖₂ over the last axis · dim^-0.5).clamp(min: eps) · g`, one scalar
/// gain. FLAGGED: the released `mossformer2_block.py` ScaleNorm was not byte-confirmed; this is the
/// standard lucidrains form the MossFormer family derives from.
final class NFKMossScaleNorm: Module, UnaryLayer {
    @ParameterInfo(key: "g") var g: MLXArray
    let scale: Float
    let eps: Float

    init(dim: Int, eps: Float = 1e-5) {
        scale = powf(Float(dim), -0.5)
        self.eps = eps
        _g.wrappedValue = MLXArray.ones([1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let norm = sqrt((x * x).sum(axis: -1, keepDims: true)) * scale
        return x / maximum(norm, eps) * g
    }
}

/// `CLayerNorm`: `nn.LayerNorm` over the channel axis. In [B, S, C] the channel axis is already last,
/// so this is a plain `LayerNorm(C)` (the reference transposes only because it holds [B, C, S]).
typealias NFKMossCLayerNorm = LayerNorm

// MARK: - Rotary (rotary_embedding_torch, adjacent-pair convention)

/// `RotaryEmbedding(dim = min(32, query_key_dim))` from rotary-embedding-torch: it rotates the FIRST 32
/// of the 128 query/key channels, adjacent-pair (`view_as_complex`) style — the same convention the LTX
/// and Z-Image ropes use — and leaves the remaining 96 untouched. θ = 10000, shared across all layers.
struct NFKMossRotary {
    let rotaryDim: Int
    let invFreq: [Float]

    init(rotaryDim: Int = 32, theta: Float = 10000) {
        self.rotaryDim = rotaryDim
        invFreq = stride(from: 0, to: rotaryDim, by: 2).map { 1 / powf(theta, Float($0) / Float(rotaryDim)) }
    }

    /// `t` `[B, S, 128]` → the same shape with the leading `rotaryDim` channels rotated by position.
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let s = t.dim(1)
        let pairs = rotaryDim / 2
        var cosT = [Float](repeating: 0, count: s * pairs)
        var sinT = [Float](repeating: 0, count: s * pairs)
        for pos in 0 ..< s {
            for k in 0 ..< pairs {
                let angle = Float(pos) * invFreq[k]
                cosT[pos * pairs + k] = cosf(angle)
                sinT[pos * pairs + k] = sinf(angle)
            }
        }
        let cos = MLXArray(cosT, [1, s, pairs, 1])
        let sin = MLXArray(sinT, [1, s, pairs, 1])
        let head = t[.ellipsis, 0 ..< rotaryDim].reshaped([t.dim(0), s, pairs, 2])
        let a = head[.ellipsis, 0 ..< 1]
        let b = head[.ellipsis, 1 ..< 2]
        let rotated = concatenated([a * cos - b * sin, a * sin + b * cos], axis: -1)
            .reshaped([t.dim(0), s, rotaryDim])
        return concatenated([rotated, t[.ellipsis, rotaryDim...]], axis: -1)
    }
}

// MARK: - ConvModule + FFConvM

/// `ConvModule`: `x + depthwise-Conv1d(k=17, pad=8)(x)`, no norm/GLU/activation. In [B, S, C] the
/// depthwise convolution runs over S directly (MLX NLC), so no transpose is needed.
final class NFKMossConvModule: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(dim: Int) {
        _conv.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 17,
                                    stride: 1, padding: 8, groups: dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x + conv(x) }
}

/// `FFConvM`: norm → Linear → SiLU → ConvModule (dropout is identity at inference). The norm is
/// `ScaleNorm` inside FLASH and `LayerNorm` inside the gated FSMN.
final class NFKMossFFConvM: Module {
    enum NormKind { case scale, layer }
    @ModuleInfo(key: "norm") var norm: UnaryLayer
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "conv_module") var convModule: NFKMossConvModule

    init(dimIn: Int, dimOut: Int, norm: NormKind) {
        switch norm {
        case .scale: _norm.wrappedValue = NFKMossScaleNorm(dim: dimIn)
        case .layer: _norm.wrappedValue = LayerNorm(dimensions: dimIn)
        }
        _linear.wrappedValue = Linear(dimIn, dimOut)
        _convModule.wrappedValue = NFKMossConvModule(dim: dimOut)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { convModule(silu(linear(norm(x)))) }
}

/// `OffsetScale(dim, heads=4)`: `x · gamma[h] + beta[h]` for each head, unbound into 4 tensors.
final class NFKMossOffsetScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray                    // [heads, dim]
    @ParameterInfo(key: "beta") var beta: MLXArray                      // [heads, dim]
    let heads: Int

    init(dim: Int, heads: Int = 4) {
        self.heads = heads
        _gamma.wrappedValue = MLXArray.ones([heads, dim])
        _beta.wrappedValue = MLXArray.zeros([heads, dim])
    }

    /// `x` `[B, S, dim]` → `heads` tensors each `[B, S, dim]`.
    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let scaled = x.expandedDimensions(axis: 2) * gamma.reshaped([1, 1, heads, gamma.dim(1)]) +
            beta.reshaped([1, 1, heads, beta.dim(1)])                   // [B, S, heads, dim]
        return (0 ..< heads).map { scaled[0..., 0..., $0, 0...] }
    }
}

// MARK: - FLASH attention

/// `FLASH_ShareA_FFConvM`: the MossFormer2 gated attention. Quadratic ReLU-squared local attention
/// within groups of `groupSize`, plus a linear global path, combined and gated by
/// `(att_u · v) · sigmoid(att_v · u)`. `to_hidden` produces `v, u` (each `dim·2` at expansion 4);
/// `to_qk` a shared 128-wide base scaled into four `OffsetScale` tensors; rotary rotates the leading 32.
final class NFKMossFLASH: Module {
    @ModuleInfo(key: "to_hidden") var toHidden: NFKMossFFConvM
    @ModuleInfo(key: "to_qk") var toQK: NFKMossFFConvM
    @ModuleInfo(key: "qk_offset_scale") var offsetScale: NFKMossOffsetScale
    @ModuleInfo(key: "to_out") var toOut: NFKMossFFConvM
    let groupSize: Int
    let rotary: NFKMossRotary

    init(dim: Int, groupSize: Int, queryKeyDim: Int, expansionFactor: Float, rotary: NFKMossRotary) {
        self.groupSize = groupSize
        self.rotary = rotary
        let hidden = Int(Float(dim) * expansionFactor)                  // dim·4
        _toHidden.wrappedValue = NFKMossFFConvM(dimIn: dim, dimOut: hidden, norm: .scale)
        _toQK.wrappedValue = NFKMossFFConvM(dimIn: dim, dimOut: queryKeyDim, norm: .scale)
        _offsetScale.wrappedValue = NFKMossOffsetScale(dim: queryKeyDim, heads: 4)
        _toOut.wrappedValue = NFKMossFFConvM(dimIn: dim * 2, dimOut: dim, norm: .scale)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, s, c) = (x.dim(0), x.dim(1), x.dim(2))
        // shift_tokens: shift the first half-channels forward by one frame (pad (1, -1) on the sequence).
        let half = c / 2
        let shift = x[0..., 0..., 0 ..< half]
        let shifted = concatenated([MLXArray.zeros([b, 1, half]), shift[0..., .stride(to: s - 1), 0...]], axis: 1)
        let normedX = concatenated([shifted, x[0..., 0..., half...]], axis: 2)

        let hidden = toHidden(normedX)
        let v = hidden[.ellipsis, .stride(to: hidden.dim(2) / 2)]
        let u = hidden[.ellipsis, .stride(from: hidden.dim(2) / 2)]
        let qkParts = offsetScale(toQK(normedX)).map { rotary($0) }
        let (quadQ, linQ, quadK, linK) = (qkParts[0], qkParts[1], qkParts[2], qkParts[3])
        let (attV, attU) = attention(quadQ: quadQ, linQ: linQ, quadK: quadK, linK: linK, v: v, u: u)
        let out = (attU * v) * sigmoid(attV * u)
        return x + toOut(out)
    }

    /// The grouped quadratic + global linear attention (non-causal). Returns the v-path and u-path
    /// outputs, each `[B, S, dim·2]`.
    private func attention(quadQ: MLXArray, linQ: MLXArray, quadK: MLXArray, linK: MLXArray,
                           v: MLXArray, u: MLXArray) -> (MLXArray, MLXArray) {
        let (b, s) = (v.dim(0), v.dim(1))
        let g = groupSize
        let padding = (g - s % g) % g
        let sPad = s + padding
        func pad(_ t: MLXArray) -> MLXArray {
            padding == 0 ? t : MLX.padded(t, widths: [IntOrPair(0), IntOrPair((0, padding)), IntOrPair(0)], mode: .constant)
        }
        let qqP = pad(quadQ), kkP = pad(quadK), vP = pad(v), uP = pad(u)
        let lqP = pad(linQ), lkP = pad(linK)
        let groups = sPad / g
        let d = quadQ.dim(2)
        let e = v.dim(2)

        // Quadratic local: within each group of `g`, attn = relu(q·kᵀ / g)² then attn·{v,u}.
        func group(_ t: MLXArray, width: Int) -> MLXArray { t.reshaped([b * groups, g, width]) }
        let qg = group(qqP, width: d), kg = group(kkP, width: d)
        let vg = group(vP, width: e), ug = group(uP, width: e)
        let sim = matmul(qg, kg.transposed(0, 2, 1)) / Float(g)
        let attn = pow(maximum(sim, 0), 2)
        let quadV = matmul(attn, vg).reshaped([b, sPad, e])
        let quadU = matmul(attn, ug).reshaped([b, sPad, e])

        // Linear global: lin_kv = (linKᵀ · v) / n summed over all frames, re-applied to lin_q. PARITY:
        // the reference divides by the ORIGINAL sequence length `n = x.shape[-2]`, not the padded length.
        let linKV = matmul(lkP.transposed(0, 2, 1), vP) / Float(s)     // [B, d, e]
        let linKU = matmul(lkP.transposed(0, 2, 1), uP) / Float(s)
        let linV = matmul(lqP, linKV)                                   // [B, sPad, e]
        let linU = matmul(lqP, linKU)

        let outV = (quadV + linV)[0..., .stride(to: s), 0...]
        let outU = (quadU + linU)[0..., .stride(to: s), 0...]
        _ = d
        return (outV, outU)
    }
}

// MARK: - Gated FSMN

/// `UniDeepFsmn`: Linear → ReLU → project(no bias) → a symmetric-padded depthwise `Conv2d([39,1])`
/// added back, then a residual over the input. Modeled with a `Conv2d` over an [B, S, 1, C] grid to
/// mirror the reference's NCHW `[B, C, S, 1]`.
final class NFKMossUniDeepFsmn: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "project") var project: Linear
    @ModuleInfo(key: "conv1") var conv: Conv2d
    let lorder: Int

    init(inputDim: Int, outputDim: Int, lorder: Int, hidden: Int) {
        self.lorder = lorder
        _linear.wrappedValue = Linear(inputDim, hidden)
        _project.wrappedValue = Linear(hidden, outputDim, bias: false)
        _conv.wrappedValue = Conv2d(inputChannels: outputDim, outputChannels: outputDim,
                                    kernelSize: IntOrPair((2 * lorder - 1, 1)), stride: 1,
                                    padding: IntOrPair((lorder - 1, 0)), groups: outputDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = project(relu(linear(x)))                               // [B, S, C]
        let grid = p.expandedDimensions(axis: 2)                       // [B, S, 1, C] ↔ ref [B, C, S, 1]
        return x + (grid + conv(grid)).squeezed(axis: 2)
    }
}

/// `Gated_FSMN`: `to_v(x) · fsmn(to_u(x)) + x`, the two projections `FFConvM` with `LayerNorm`.
final class NFKMossGatedFSMN: Module {
    @ModuleInfo(key: "to_u") var toU: NFKMossFFConvM
    @ModuleInfo(key: "to_v") var toV: NFKMossFFConvM
    @ModuleInfo(key: "fsmn") var fsmn: NFKMossUniDeepFsmn

    init(channels: Int, lorder: Int, hidden: Int) {
        _toU.wrappedValue = NFKMossFFConvM(dimIn: channels, dimOut: hidden, norm: .layer)
        _toV.wrappedValue = NFKMossFFConvM(dimIn: channels, dimOut: hidden, norm: .layer)
        _fsmn.wrappedValue = NFKMossUniDeepFsmn(inputDim: channels, outputDim: channels, lorder: lorder, hidden: hidden)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { toV(x) * fsmn(toU(x)) + x }
}

/// `Gated_FSMN_Block`: `conv1(PReLU) → CLayerNorm → Gated_FSMN → CLayerNorm → conv2`, over a residual.
/// The reference transposes to [B, C, S] for the 1×1 convs; in [B, S, C] they are `Conv1d` k1.
final class NFKMossGatedFSMNBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv1_prelu") var prelu: PReLU
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "gated_fsmn") var gatedFSMN: NFKMossGatedFSMN
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(dim: Int, inner: Int, lorder: Int) {
        _conv1.wrappedValue = Conv1d(inputChannels: dim, outputChannels: inner, kernelSize: 1)
        _prelu.wrappedValue = PReLU(count: inner)
        _norm1.wrappedValue = LayerNorm(dimensions: inner)
        _gatedFSMN.wrappedValue = NFKMossGatedFSMN(channels: inner, lorder: lorder, hidden: inner)
        _norm2.wrappedValue = LayerNorm(dimensions: inner)
        _conv2.wrappedValue = Conv1d(inputChannels: inner, outputChannels: dim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = norm1(prelu(conv1(x)))
        let g = norm2(gatedFSMN(h))
        return conv2(g) + x
    }
}

// MARK: - The block stack and MaskNet

/// `MossformerBlock_GFSMN`: `depth` interleaved FLASH-then-GatedFSMN layers over one shared rotary.
final class NFKMossBlockGFSMN: Module {
    @ModuleInfo(key: "layers") var layers: [NFKMossFLASH]
    @ModuleInfo(key: "fsmn") var fsmn: [NFKMossGatedFSMNBlock]

    init(_ config: NFKMLXMossFormer2Configuration) {
        let rotary = NFKMossRotary(rotaryDim: min(32, config.queryKeyDim))
        _layers.wrappedValue = (0 ..< config.numBlocks).map { _ in
            NFKMossFLASH(dim: config.dModel, groupSize: config.groupSize, queryKeyDim: config.queryKeyDim,
                         expansionFactor: config.expansionFactor, rotary: rotary)
        }
        _fsmn.wrappedValue = (0 ..< config.numBlocks).map { _ in
            NFKMossGatedFSMNBlock(dim: config.dModel, inner: config.fsmnHidden, lorder: config.fsmnLorder)
        }
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = x0
        for i in 0 ..< layers.count {
            x = layers[i](x)
            x = fsmn[i](x)
        }
        return x
    }
}

/// `MossFormerM`: the block stack then a final `LayerNorm(eps: 1e-6)`.
final class NFKMossFormerM: Module {
    @ModuleInfo(key: "mossformerM") var block: NFKMossBlockGFSMN
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ config: NFKMLXMossFormer2Configuration) {
        _block.wrappedValue = NFKMossBlockGFSMN(config)
        _norm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-6)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(block(x)) }
}

/// `Computation_Block`: the intra model with a GroupNorm and a `skip_around_intra` residual. Runs on
/// [B, S, C] (the reference permutes [B, C, S] → [B, S, C] for `intra_mdl`; here it is already so).
final class NFKMossComputationBlock: Module {
    @ModuleInfo(key: "intra_mdl") var intra: NFKMossFormerM
    @ModuleInfo(key: "intra_norm") var norm: GroupNorm

    init(_ config: NFKMLXMossFormer2Configuration) {
        _intra.wrappedValue = NFKMossFormerM(config)
        _norm.wrappedValue = GroupNorm(groupCount: 1, dimensions: config.dModel, eps: 1e-8, pytorchCompatible: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(intra(x)) + x }
}

/// `ScaledSinuEmbedding`: a fixed sinusoidal table times a learned scalar. Added to the encoded input.
final class NFKMossScaledSinuEmbedding: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray                    // [1]
    let dim: Int

    init(dim: Int) {
        self.dim = dim
        _scale.wrappedValue = MLXArray.ones([1])
    }

    /// `[S, dim]` positional embedding for a sequence of length `s`.
    func table(_ s: Int) -> MLXArray {
        let pairs = dim / 2
        var sinT = [Float](repeating: 0, count: s * pairs)
        var cosT = [Float](repeating: 0, count: s * pairs)
        for pos in 0 ..< s {
            for k in 0 ..< pairs {
                let inv = 1 / powf(10000, Float(2 * k) / Float(dim))
                sinT[pos * pairs + k] = sinf(Float(pos) * inv)
                cosT[pos * pairs + k] = cosf(Float(pos) * inv)
            }
        }
        let emb = concatenated([MLXArray(sinT, [s, pairs]), MLXArray(cosT, [s, pairs])], axis: 1)
        return emb * scale
    }
}

/// The MossFormer2 SE MaskNet (`MossFormer_MaskNet`, checkpoint prefix `mossformer.`). Input the 180-dim
/// feature `[B, S, 180]`; output the real 961-bin mask `[B, S, 961]`.
public final class NFKMLXMossFormer2SENet: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "conv1d_encoder") var encoder: Conv1d
    @ModuleInfo(key: "pos_enc") var posEnc: NFKMossScaledSinuEmbedding
    @ModuleInfo(key: "mdl") var mdl: NFKMossComputationBlock
    @ModuleInfo(key: "prelu") var prelu: PReLU
    @ModuleInfo(key: "conv1d_out") var convOut: Conv1d
    @ModuleInfo(key: "output") var output: Conv1d
    @ModuleInfo(key: "output_gate") var outputGate: Conv1d
    @ModuleInfo(key: "conv1_decoder") var decoder: Conv1d
    let dModel: Int

    public init(_ config: NFKMLXMossFormer2Configuration) {
        dModel = config.dModel
        _norm.wrappedValue = GroupNorm(groupCount: 1, dimensions: config.inChannels, eps: 1e-8, pytorchCompatible: true)
        _encoder.wrappedValue = Conv1d(inputChannels: config.inChannels, outputChannels: config.dModel, kernelSize: 1, bias: false)
        _posEnc.wrappedValue = NFKMossScaledSinuEmbedding(dim: config.dModel)
        _mdl.wrappedValue = NFKMossComputationBlock(config)
        _prelu.wrappedValue = PReLU(count: config.dModel)
        _convOut.wrappedValue = Conv1d(inputChannels: config.dModel, outputChannels: config.dModel * config.numSpks, kernelSize: 1)
        _output.wrappedValue = Conv1d(inputChannels: config.dModel, outputChannels: config.dModel, kernelSize: 1)
        _outputGate.wrappedValue = Conv1d(inputChannels: config.dModel, outputChannels: config.dModel, kernelSize: 1)
        _decoder.wrappedValue = Conv1d(inputChannels: config.dModel, outputChannels: config.outChannelsFinal, kernelSize: 1, bias: false)
    }

    /// `[B, S, 180]` → `[B, S, 961]` non-negative mask.
    public func callAsFunction(_ feature: MLXArray) -> MLXArray {
        var x = encoder(norm(feature))                                 // [B, S, 512]
        x = x + posEnc.table(x.dim(1)).reshaped([1, x.dim(1), x.dim(2)])
        x = mdl(x)
        x = prelu(x)
        // conv1d_out widens to dModel·numSpks; the SE net keeps speaker 0 (channels 0..<dModel), and
        // the 1×1 gated output / decoder are per-position, so slicing speaker 0 first is exact.
        let speaker0 = convOut(x)[0..., 0..., 0 ..< dModel]
        let gated = tanh(output(speaker0)) * sigmoid(outputGate(speaker0))
        return relu(decoder(gated))                                    // [B, S, 961]
    }
}

// MARK: - Masking STFT (hamming, center=False)

/// The SE masking STFT: hamming window (`periodic=false`), `center=false`, `return_complex=false`. Its
/// frame count matches the Kaldi fbank's snip-edges framing (same window and hop), so the mask aligns.
struct NFKMossSTFT {
    let nFFT: Int
    let hop: Int
    let window: MLXArray

    init(nFFT: Int, hop: Int) {
        self.nFFT = nFFT
        self.hop = hop
        let a = 2 * Float.pi / Float(nFFT - 1)
        window = MLXArray((0 ..< nFFT).map { 0.54 - 0.46 * cosf(a * Float($0)) })
    }

    /// `[L]` → real and imaginary parts, each `[bins, frames]` (no centering).
    func transform(_ signal: [Float]) -> (real: MLXArray, imaginary: MLXArray, frames: Int) {
        let frames = 1 + (signal.count - nFFT) / hop
        var gather = [Float](repeating: 0, count: frames * nFFT)
        for f in 0 ..< frames {
            for k in 0 ..< nFFT { gather[f * nFFT + k] = signal[f * hop + k] }
        }
        let framed = MLXArray(gather, [frames, nFFT]) * window.reshaped([1, nFFT])
        let spectrum = MLXFFT.rfft(framed, axis: 1)                    // [frames, bins]
        return (spectrum.realPart().transposed(1, 0), spectrum.imaginaryPart().transposed(1, 0), frames)
    }

    /// real and imaginary `[bins, frames]` → `[samples]`, window-squared overlap-add, `center=false`.
    func inverse(real: MLXArray, imaginary: MLXArray, length: Int) -> [Float] {
        let frames = real.dim(1)
        let complex = real.transposed(1, 0).asType(.complex64) +
            imaginary.transposed(1, 0).asType(.complex64) * MLXArray(real: 0, imaginary: 1)
        let time = MLXFFT.irfft(complex, n: nFFT, axis: 1)            // [frames, nFFT]
        let win = (window * window).asArray(Float.self)
        let framesValues = (time * window.reshaped([1, nFFT])).asArray(Float.self)
        let outLen = (frames - 1) * hop + nFFT
        var out = [Float](repeating: 0, count: outLen)
        var norm = [Float](repeating: 0, count: outLen)
        for f in 0 ..< frames {
            for k in 0 ..< nFFT {
                out[f * hop + k] += framesValues[f * nFFT + k]
                norm[f * hop + k] += win[k]
            }
        }
        for i in 0 ..< outLen where norm[i] > 1e-8 { out[i] /= norm[i] }
        return Array(out.prefix(length))
    }
}

// MARK: - Backend

private final class NFKMossHolder: @unchecked Sendable {
    let net: NFKMLXMossFormer2SENet
    let config: NFKMLXMossFormer2Configuration
    init(_ net: NFKMLXMossFormer2SENet, _ config: NFKMLXMossFormer2Configuration) { self.net = net; self.config = config }
}

/// MossFormer2 SE 48K speech enhancement as an InferKit backend. Reads `NFKInputAudio`; returns the
/// enhanced clip as a single `NFKAudioAsset` under `NFKOutputAudio`.
@objc(NFKMLXMossFormer2Backend)
public final class NFKMLXMossFormer2Backend: NSObject, NFKInferenceBackend {
    private let holder: NFKMossHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXMossFormer2SENet, config: NFKMLXMossFormer2Configuration, identifier: String,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKMossHolder(net, config)
        self.identifier = identifier
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let config = holder.config
        // PARITY: the fbank, mask, and STFT are defined at 48 kHz; resample a clip at another rate first.
        let input = sampleRate == config.sampleRate ? samples
            : NFKMLXAudioRate.matched(samples, from: sampleRate, to: config.sampleRate)
        let enhanced = Self.enhance(input, net: holder.net, config: config)
        let url = outputDirectory.appendingPathComponent("mossformer2-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: enhanced, sampleRate: config.sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(enhanced.count) / Double(config.sampleRate),
                                  sampleRate: Double(config.sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    /// The full path: Kaldi fbank → MaskNet → real mask applied to the hamming STFT (phase kept) →
    /// iSTFT. Exposed for the parity harness (which can feed a recorded feature instead).
    static func enhance(_ samples: [Float], net: NFKMLXMossFormer2SENet, config: NFKMLXMossFormer2Configuration) -> [Float] {
        let feature = NFKMLXKaldiFbank.features(samples: samples, config: config)   // [1, S, 180]
        let mask = net(feature)                                                     // [1, S, 961]
        let stft = NFKMossSTFT(nFFT: config.fftLen, hop: config.winInc)
        let (re, im, frames) = stft.transform(samples)                              // [961, T]
        let maskBinsFrames = mask[0].transposed(1, 0)[0..., .stride(to: frames)]    // [961, T]
        let maskedRe = re * maskBinsFrames
        let maskedIm = im * maskBinsFrames
        eval(maskedRe, maskedIm)
        return stft.inverse(real: maskedRe, imaginary: maskedIm, length: samples.count)
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do { job.finish(with: try self.runInference(for: request)) }
            catch { job.finish(withError: error as NSError) }
        }
        return job
    }

    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data { return NFKMLXWaveFile.read(data) }
        return nil
    }
}

// MARK: - Registration and weight loading

/// Registration and weight loading for MossFormer2 SE 48K.
@objc(NFKMLXMossFormer2_Factory)
public final class NFKMLXMossFormer2Factory: NSObject {
    @objc public static let modelName = "mossformer2-se"

    static func makeNet(_ config: NFKMLXMossFormer2Configuration = .init()) -> NFKMLXMossFormer2SENet {
        NFKMLXMossFormer2SENet(config)
    }

    /// Builds a backend from optional local weights (a converted safetensors, or the released `.pt`
    /// through the native torch reader). A nil `weightsURL` builds random weights (`isReady` true).
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let config = NFKMLXMossFormer2Configuration()
        let net = makeNet(config)
        if let weightsURL { try loadWeights(into: net, from: weightsURL) }
        return NFKMLXMossFormer2Backend(net: net, config: config, identifier: modelName)
    }

    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) }, completionHandler: completionHandler)
    }

    /// Registers `mossformer2-se` with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a released MossFormer2 SE checkpoint (`last_best_checkpoint.pt`, prefix `mossformer.`) into
    /// the MaskNet: unwrap the training container, drop the pos-enc / rotary buffers, remap the nested
    /// `nn.Sequential` indices, and transpose the convolution weights.
    static func loadWeights(into net: NFKMLXMossFormer2SENet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped: [(String, MLXArray)] = checkpoint.arrays.compactMap { key, value in
            guard let name = remapReferenceKey(key) else { return nil }
            let tensor: MLXArray
            if value.ndim == 4, checkpoint.needsConvTranspose {
                tensor = value.transposed(0, 2, 3, 1)                   // Conv2d [out,in,kH,kW] → [out,kH,kW,in]
            } else if value.ndim == 3, checkpoint.needsConvTranspose {
                tensor = value.transposed(0, 2, 1)                     // Conv1d [out,in,k] → [out,k,in]
            } else {
                tensor = value
            }
            return (name, tensor)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// The reference-name → module-key remap. Strips the `mossformer.` wrapper prefix, drops the fixed
    /// sinusoidal / rotary buffers, and translates the `nn.Sequential` indices of `FFConvM.mdl`
    /// (`0`=norm, `1`=linear, `3.sequential.1.conv`=conv), `ConvModule`, the gated-FSMN-block convs
    /// (`conv1.0`/`conv1.1`=conv/prelu), and the MaskNet output gates (`output.0`/`output_gate.0`).
    static func remapReferenceKey(_ key: String) -> String? {
        guard key.hasPrefix("mossformer.") else { return nil }
        var name = String(key.dropFirst("mossformer.".count))
        // Fixed buffers this port recomputes.
        if name.hasSuffix("pos_enc.inv_freq") || name.contains(".rotary") || name.hasSuffix(".freqs") { return nil }
        // FFConvM Sequential: mdl.0 → norm, mdl.1 → linear, mdl.3.sequential.1.conv → conv_module.conv.
        name = name.replacingOccurrences(of: ".mdl.3.sequential.1.conv.", with: ".conv_module.conv.")
        name = name.replacingOccurrences(of: ".mdl.0.", with: ".norm.")
        name = name.replacingOccurrences(of: ".mdl.1.", with: ".linear.")
        // Gated_FSMN_Block conv1 Sequential: conv1.0 → conv1, conv1.1 → conv1_prelu.
        name = name.replacingOccurrences(of: ".conv1.0.", with: ".conv1.")
        name = name.replacingOccurrences(of: ".conv1.1.", with: ".conv1_prelu.")
        // MaskNet gated output Sequentials: output.0 / output_gate.0 → the single conv (Tanh/Sigmoid
        // carry no parameters).
        name = name.replacingOccurrences(of: "output.0.", with: "output.")
        name = name.replacingOccurrences(of: "output_gate.0.", with: "output_gate.")
        return name
    }
}
