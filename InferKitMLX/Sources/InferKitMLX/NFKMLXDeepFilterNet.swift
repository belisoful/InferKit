// DeepFilterNet3 (Rikorose/DeepFilterNet, dual MIT/Apache-2.0): a ~2.3M-parameter real-time 48 kHz
// speech denoiser, the cheap counterpart to `NFKMLXDenoiser`. The NEURAL NET (`DfNet`) is a clean torch
// `nn.Module` — this file. The STFT / ERB / normalization DSP is Rust (`libdf`) and is reproduced
// separately in `NFKMLXDeepFilterNetDSP`, validated against libdf recordings; the deep-filter APPLY and
// the ERB-mask apply live inside the net, so they are here.
//
// Grounded on the released source and the pretrained `config.ini` read 2026-09-05 (`df/deepfilternet3.py`,
// `df/modules.py`, `df/multiframe.py`), and on the released `DeepFilterNet3` checkpoint's own state dict.
// Tensors flow in MLX NHWC `[B, T, F, C]` where the reference is NCHW `[B, C, T, F]`. Net boundary:
// `forward(spec, featERB, featSpec) -> (specE, mask, coefs, lsnr)`.
//
// The reference `Conv2dNormAct` derives its layout from the shapes, not a per-layer flag: `groups =
// gcd(in, out)`, a pointwise 1x1 follows only when `groups > 1` AND the kernel is not 1x1, and a causal
// time ConstantPad2d precedes the conv only when the time kernel exceeds 1. Every shape in the released
// state dict is reproduced by that rule (e.g. `erb_conv0` is a plain conv because `gcd(1, 64) = 1`, and
// `conv3p` is a depthwise 1x1 because `gcd(64, 64) = 64` even though the kernel is 1x1).

import Foundation
import InferKit
import MLX
import MLXNN

/// The DeepFilterNet3 configuration (the pretrained `config.ini`).
public struct NFKMLXDeepFilterNetConfiguration: Sendable {
    public var sampleRate: Int
    public var fftSize: Int
    public var hopSize: Int
    public var nbERB: Int
    public var nbDF: Int
    public var dfOrder: Int
    public var dfLookahead: Int
    public var convLookahead: Int
    public var convCh: Int
    public var embHiddenDim: Int
    public var embNumLayers: Int
    public var dfHiddenDim: Int
    public var dfNumLayers: Int
    public var encLinearGroups: Int
    public var linearGroups: Int
    public var dfLinearGroups: Int
    public var dfPathwayKernelT: Int

    public var bins: Int { fftSize / 2 + 1 }

    public init(sampleRate: Int = 48000, fftSize: Int = 960, hopSize: Int = 480, nbERB: Int = 32,
                nbDF: Int = 96, dfOrder: Int = 5, dfLookahead: Int = 2, convLookahead: Int = 2,
                convCh: Int = 64, embHiddenDim: Int = 256, embNumLayers: Int = 3,
                dfHiddenDim: Int = 256, dfNumLayers: Int = 2, encLinearGroups: Int = 32,
                linearGroups: Int = 16, dfLinearGroups: Int = 8, dfPathwayKernelT: Int = 5) {
        self.sampleRate = sampleRate
        self.fftSize = fftSize
        self.hopSize = hopSize
        self.nbERB = nbERB
        self.nbDF = nbDF
        self.dfOrder = dfOrder
        self.dfLookahead = dfLookahead
        self.convLookahead = convLookahead
        self.convCh = convCh
        self.embHiddenDim = embHiddenDim
        self.embNumLayers = embNumLayers
        self.dfHiddenDim = dfHiddenDim
        self.dfNumLayers = dfNumLayers
        self.encLinearGroups = encLinearGroups
        self.linearGroups = linearGroups
        self.dfLinearGroups = dfLinearGroups
        self.dfPathwayKernelT = dfPathwayKernelT
    }
}

private func nfkGCD(_ a: Int, _ b: Int) -> Int {
    var (x, y) = (abs(a), abs(b))
    while y != 0 { (x, y) = (y, x % y) }
    return x
}

// MARK: - Building blocks

/// `GroupedLinearEinsum`: weight `[groups, in/groups, out/groups]`, `einsum('btgi,gih->btgh')`, no bias.
final class NFKDFGroupedLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray                  // [g, in/g, out/g]
    let groups: Int

    init(inFeatures: Int, outFeatures: Int, groups: Int) {
        self.groups = groups
        _weight.wrappedValue = MLXArray.zeros([groups, inFeatures / groups, outFeatures / groups])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let inPerG = weight.dim(1), outPerG = weight.dim(2)
        // [B, T, in] → [g, B*T, in/g], matmul weight [g, in/g, out/g] → [g, B*T, out/g] → [B, T, out].
        let xg = x.reshaped([b * t, groups, inPerG]).transposed(1, 0, 2)
        let og = matmul(xg, weight).transposed(1, 0, 2)
        return og.reshaped([b, t, groups * outPerG])
    }
}

enum NFKDFActivation { case relu, sigmoid }

/// `Conv2dNormAct`: a (separable) convolution with causal time padding and SAME frequency padding, then
/// `BatchNorm2d` and an activation. The layout follows the reference exactly: `groups = gcd(in, out)`,
/// a pointwise 1x1 (`conv_pw`) follows only when `groups > 1` and the kernel is not 1x1, and a causal
/// front time pad precedes the conv only when the time kernel exceeds 1.
final class NFKDFConvNormAct: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "conv_pw") var pointwise: Conv2d?
    @ModuleInfo(key: "norm") var norm: BatchNorm
    let padTimeFront: Int
    let activation: NFKDFActivation

    init(inCh: Int, outCh: Int, kernel: (Int, Int), fstride: Int, activation: NFKDFActivation = .relu) {
        self.activation = activation
        let (kT, kF) = kernel
        padTimeFront = kT - 1
        let fpad = kF / 2                                              // dilation 1 throughout
        let groups = nfkGCD(inCh, outCh)
        let hasPointwise = groups > 1 && max(kT, kF) > 1
        _conv.wrappedValue = Conv2d(inputChannels: inCh, outputChannels: outCh,
                                    kernelSize: IntOrPair(kernel), stride: IntOrPair((1, fstride)),
                                    padding: IntOrPair((0, fpad)), groups: groups, bias: false)
        _pointwise.wrappedValue = hasPointwise
            ? Conv2d(inputChannels: outCh, outputChannels: outCh, kernelSize: 1, bias: false) : nil
        _norm.wrappedValue = BatchNorm(featureCount: outCh)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if padTimeFront > 0 {
            h = MLX.padded(h, widths: [IntOrPair(0), IntOrPair((padTimeFront, 0)), IntOrPair(0), IntOrPair(0)], mode: .constant)
        }
        h = conv(h)
        if let pointwise { h = pointwise(h) }
        h = norm(h)
        return activation == .relu ? relu(h) : sigmoid(h)
    }
}

/// A frequency-upsampling separable `ConvTranspose2dNormAct` (the ERB decoder's `convt2`/`convt1`): a
/// depthwise `ConvTranspose2d` over frequency (fstride 2, PyTorch `padding=(0,1)`, `output_padding=(0,1)`)
/// then a pointwise 1x1, `BatchNorm`, `ReLU`. The grouped transpose runs per group, because MLX's grouped
/// `ConvTranspose2d` does not match PyTorch's grouping (the shared GTCRN finding).
final class NFKDFConvTranspose: Module {
    @ModuleInfo(key: "conv") var conv: ConvTransposed2d
    @ModuleInfo(key: "conv_pw") var pointwise: Conv2d
    @ModuleInfo(key: "norm") var norm: BatchNorm

    init(inCh: Int, outCh: Int) {
        _conv.wrappedValue = ConvTransposed2d(inputChannels: inCh, outputChannels: outCh,
                                              kernelSize: IntOrPair((1, 3)), stride: IntOrPair((1, 2)),
                                              padding: IntOrPair((0, 1)), outputPadding: IntOrPair((0, 1)),
                                              groups: nfkGCD(inCh, outCh), bias: false)
        _pointwise.wrappedValue = Conv2d(inputChannels: outCh, outputChannels: outCh, kernelSize: 1, bias: false)
        _norm.wrappedValue = BatchNorm(featureCount: outCh)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = conv
        var merged: MLXArray
        if c.groups > 1 {
            let inPer = x.dim(3) / c.groups, outPer = c.weight.dim(0) / c.groups
            var parts = [MLXArray]()
            for group in 0 ..< c.groups {
                let xg = x[0..., 0..., 0..., group * inPer ..< (group + 1) * inPer]
                let wg = c.weight[group * outPer ..< (group + 1) * outPer]
                parts.append(convTransposed2d(xg, wg, stride: IntOrPair(c.stride),
                                              padding: IntOrPair(c.padding), dilation: IntOrPair(c.dilation),
                                              outputPadding: IntOrPair(c.outputPadding)))
            }
            merged = concatenated(parts, axis: 3)
        } else {
            merged = c(x)
        }
        return relu(norm(pointwise(merged)))
    }
}

/// `SqueezedGRU_S`: `GroupedLinear → ReLU → (stacked) GRU → GroupedLinear` (batch_first), no transpose
/// trap. The GRU is a stack of `numLayers` shared `NFKMLXGRUCell`s (loaded through the DFN fold).
final class NFKDFSqueezedGRU: Module {
    @ModuleInfo(key: "linear_in") var linearIn: NFKDFGroupedLinear
    @ModuleInfo(key: "gru") var gru: [NFKMLXGRUCell]
    @ModuleInfo(key: "linear_out") var linearOut: NFKDFGroupedLinear?      // `output_size=None` → Identity

    init(inputSize: Int, hidden: Int, output: Int?, groups: Int, numLayers: Int) {
        _linearIn.wrappedValue = NFKDFGroupedLinear(inFeatures: inputSize, outFeatures: hidden, groups: groups)
        _gru.wrappedValue = (0 ..< numLayers).map { _ in NFKMLXGRUCell(inputSize: hidden, hiddenSize: hidden) }
        _linearOut.wrappedValue = output.map { NFKDFGroupedLinear(inFeatures: hidden, outFeatures: $0, groups: groups) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = relu(linearIn(x))                                      // linear_in = Sequential(GroupedLinear, ReLU)
        for cell in gru { h = cell(h) }
        guard let linearOut else { return h }                          // output_size=None → Identity
        return relu(linearOut(h))                                      // linear_out = Sequential(GroupedLinear, ReLU)
    }
}

// MARK: - Encoder / decoders

/// The encoder: an ERB convolution pathway and a DF convolution pathway, summed (`enc_concat=False`),
/// then `emb_gru`. Returns the ERB skips e0..e3, the recurrent embedding, the DF skip c0 (the output of
/// `df_conv0`, before `df_conv1`), and the local-SNR estimate.
final class NFKDFEncoder: Module {
    @ModuleInfo(key: "erb_conv0") var erbConv0: NFKDFConvNormAct
    @ModuleInfo(key: "erb_conv1") var erbConv1: NFKDFConvNormAct
    @ModuleInfo(key: "erb_conv2") var erbConv2: NFKDFConvNormAct
    @ModuleInfo(key: "erb_conv3") var erbConv3: NFKDFConvNormAct
    @ModuleInfo(key: "df_conv0") var dfConv0: NFKDFConvNormAct
    @ModuleInfo(key: "df_conv1") var dfConv1: NFKDFConvNormAct
    @ModuleInfo(key: "df_fc_emb") var dfFCEmb: NFKDFGroupedLinear
    @ModuleInfo(key: "emb_gru") var embGRU: NFKDFSqueezedGRU
    @ModuleInfo(key: "lsnr_fc") var lsnrFC: Linear

    let config: NFKMLXDeepFilterNetConfiguration
    let lsnrScale: Float = 50                                          // lsnr_max - lsnr_min
    let lsnrOffset: Float = -15

    init(_ config: NFKMLXDeepFilterNetConfiguration) {
        self.config = config
        let c = config.convCh
        _erbConv0.wrappedValue = NFKDFConvNormAct(inCh: 1, outCh: c, kernel: (3, 3), fstride: 1)
        _erbConv1.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 3), fstride: 2)
        _erbConv2.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 3), fstride: 2)
        _erbConv3.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 3), fstride: 1)
        _dfConv0.wrappedValue = NFKDFConvNormAct(inCh: 2, outCh: c, kernel: (3, 3), fstride: 1)
        _dfConv1.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 3), fstride: 2)
        let embIn = c * config.nbERB / 4                               // 64·8 = 512
        _dfFCEmb.wrappedValue = NFKDFGroupedLinear(inFeatures: c * config.nbDF / 2, outFeatures: embIn, groups: config.encLinearGroups)
        _embGRU.wrappedValue = NFKDFSqueezedGRU(inputSize: embIn, hidden: config.embHiddenDim, output: embIn,
                                                groups: config.linearGroups, numLayers: 1)
        _lsnrFC.wrappedValue = Linear(embIn, 1)
    }

    /// Returns the recurrent embedding `[B, T, embIn]`, the ERB skips (e0..e3), the DF skip (c0), and lsnr.
    func callAsFunction(featERB: MLXArray, featSpec: MLXArray)
        -> (emb: MLXArray, e0: MLXArray, e1: MLXArray, e2: MLXArray, e3: MLXArray, c0: MLXArray, lsnr: MLXArray) {
        let e0 = erbConv0(featERB)
        let e1 = erbConv1(e0)
        let e2 = erbConv2(e1)
        let e3 = erbConv3(e2)                                           // [B, T, 8, C]
        let c0 = dfConv0(featSpec)                                      // [B, T, 96, C]  (the DF skip)
        let c1 = dfConv1(c0)                                            // [B, T, 48, C]
        let (b, t) = (e3.dim(0), e3.dim(1))
        let erbFlat = e3.reshaped([b, t, e3.dim(2) * e3.dim(3)])        // [B, T, 512]
        let dfFlat = c1.reshaped([b, t, c1.dim(2) * c1.dim(3)])         // [B, T, 3072]
        let cemb = relu(dfFCEmb(dfFlat))                               // df_fc_emb = Sequential(GroupedLinear, ReLU)
        let emb = embGRU(erbFlat + cemb)                              // combine = Add (enc_concat=False)
        let lsnr = sigmoid(lsnrFC(emb)) * lsnrScale + lsnrOffset
        return (emb, e0, e1, e2, e3, c0, lsnr)
    }
}

/// The ERB decoder: `emb_gru` then a U-net of depthwise-1x1 skip projections and (transposed) convolutions
/// to the 32-band ERB mask through a `Sigmoid`.
final class NFKDFErbDecoder: Module {
    @ModuleInfo(key: "emb_gru") var embGRU: NFKDFSqueezedGRU
    @ModuleInfo(key: "conv3p") var conv3p: NFKDFConvNormAct
    @ModuleInfo(key: "convt3") var convt3: NFKDFConvNormAct
    @ModuleInfo(key: "conv2p") var conv2p: NFKDFConvNormAct
    @ModuleInfo(key: "convt2") var convt2: NFKDFConvTranspose
    @ModuleInfo(key: "conv1p") var conv1p: NFKDFConvNormAct
    @ModuleInfo(key: "convt1") var convt1: NFKDFConvTranspose
    @ModuleInfo(key: "conv0p") var conv0p: NFKDFConvNormAct
    @ModuleInfo(key: "conv0_out") var conv0Out: NFKDFConvNormAct

    let config: NFKMLXDeepFilterNetConfiguration

    init(_ config: NFKMLXDeepFilterNetConfiguration) {
        self.config = config
        let c = config.convCh
        let embIn = c * config.nbERB / 4
        _embGRU.wrappedValue = NFKDFSqueezedGRU(inputSize: embIn, hidden: config.embHiddenDim, output: embIn,
                                                groups: config.linearGroups, numLayers: config.embNumLayers - 1)
        // The pathway convolutions are depthwise 1x1 (`groups = gcd(64, 64) = 64`, kernel 1x1 → no pointwise).
        _conv3p.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 1), fstride: 1)
        _convt3.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 3), fstride: 1)
        _conv2p.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 1), fstride: 1)
        _convt2.wrappedValue = NFKDFConvTranspose(inCh: c, outCh: c)
        _conv1p.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 1), fstride: 1)
        _convt1.wrappedValue = NFKDFConvTranspose(inCh: c, outCh: c)
        _conv0p.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 1), fstride: 1)
        _conv0Out.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: 1, kernel: (1, 3), fstride: 1, activation: .sigmoid)
    }

    /// `emb [B, T, embIn]` + the ERB skips → the ERB mask `[B, T, 32, 1]`.
    func callAsFunction(emb: MLXArray, e0: MLXArray, e1: MLXArray, e2: MLXArray, e3: MLXArray) -> MLXArray {
        let (b, t) = (e3.dim(0), e3.dim(1))
        let hidden = embGRU(emb).reshaped([b, t, e3.dim(2), e3.dim(3)])
        var x = convt3(conv3p(e3) + hidden)
        x = convt2(conv2p(e2) + x)
        x = convt1(conv1p(e1) + x)
        return conv0Out(conv0p(e0) + x)                               // conv0_out carries the Sigmoid
    }
}

/// The DF decoder: `df_gru` (+ a grouped-linear skip) then a grouped-linear head (Tanh) summed with a
/// local pathway convolution → the deep-filter coefficients `[B, dfOrder, T, nbDF, 2]`. (`df_fc_a` exists
/// in the checkpoint but the reference `forward` never applies it, so it is loaded and left unused.)
final class NFKDFDfDecoder: Module {
    @ModuleInfo(key: "df_convp") var dfConvP: NFKDFConvNormAct
    @ModuleInfo(key: "df_gru") var dfGRU: NFKDFSqueezedGRU
    @ModuleInfo(key: "df_skip") var dfSkip: NFKDFGroupedLinear
    @ModuleInfo(key: "df_out") var dfOut: NFKDFGroupedLinear
    @ModuleInfo(key: "df_fc_a") var dfFCA: Linear
    let config: NFKMLXDeepFilterNetConfiguration

    init(_ config: NFKMLXDeepFilterNetConfiguration) {
        self.config = config
        let c = config.convCh
        let embIn = c * config.nbERB / 4
        let dfOutCh = config.dfOrder * 2
        _dfConvP.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: dfOutCh, kernel: (config.dfPathwayKernelT, 1), fstride: 1)
        _dfGRU.wrappedValue = NFKDFSqueezedGRU(inputSize: embIn, hidden: config.dfHiddenDim, output: nil,
                                               groups: config.dfLinearGroups, numLayers: config.dfNumLayers)
        _dfSkip.wrappedValue = NFKDFGroupedLinear(inFeatures: embIn, outFeatures: config.dfHiddenDim, groups: config.linearGroups)
        _dfOut.wrappedValue = NFKDFGroupedLinear(inFeatures: config.dfHiddenDim, outFeatures: config.nbDF * dfOutCh, groups: config.linearGroups)
        _dfFCA.wrappedValue = Linear(config.dfHiddenDim, 1)
    }

    /// Returns the deep-filter coefficients `[B, dfOrder, T, nbDF, 2]`.
    func callAsFunction(emb: MLXArray, c0: MLXArray) -> MLXArray {
        let (b, t) = (emb.dim(0), emb.dim(1))
        let g = dfGRU(emb) + dfSkip(emb)
        let head = tanh(dfOut(g)).reshaped([b, t, config.nbDF, config.dfOrder * 2])
        let local = dfConvP(c0)                                        // [B, T, nbDF, dfOrder·2]
        let coefsFlat = head + local
        // [B, T, F, O*2] → [B, T, F, O, 2] → [B, O, T, F, 2]
        return coefsFlat.reshaped([b, t, config.nbDF, config.dfOrder, 2]).transposed(0, 3, 1, 2, 4)
    }
}

// MARK: - DfNet

/// The DeepFilterNet3 network. Inputs the pre-computed features from `NFKMLXDeepFilterNetDSP`: the raw
/// complex spectrogram `spec [B, T, F, 2]`, `featERB [B, T, 32, 1]`, `featSpec [B, T, 96, 2]`. Outputs the
/// enhanced spectrogram, the ERB mask, the deep-filter coefficients, and the local-SNR estimate.
public final class NFKMLXDeepFilterNet: Module {
    @ModuleInfo(key: "enc") var enc: NFKDFEncoder
    @ModuleInfo(key: "erb_dec") var erbDec: NFKDFErbDecoder
    @ModuleInfo(key: "df_dec") var dfDec: NFKDFDfDecoder
    @ParameterInfo(key: "erb_fb") var erbFB: MLXArray                  // [481, 32] the forward ERB bank (DSP)
    @ParameterInfo(key: "erb_inv_fb") var erbInvFB: MLXArray           // [32, 481] the inverse ERB bank (mask)
    let config: NFKMLXDeepFilterNetConfiguration

    public init(_ config: NFKMLXDeepFilterNetConfiguration) {
        self.config = config
        _enc.wrappedValue = NFKDFEncoder(config)
        _erbDec.wrappedValue = NFKDFErbDecoder(config)
        _dfDec.wrappedValue = NFKDFDfDecoder(config)
        _erbFB.wrappedValue = MLXArray.zeros([config.bins, config.nbERB])
        _erbInvFB.wrappedValue = MLXArray.zeros([config.nbERB, config.bins])
    }

    /// Shifts the features forward by `convLookahead` frames (the reference `pad_feat`: crop the front,
    /// zero-pad the tail), giving the causal convolutions their lookahead.
    func padFeat(_ x: MLXArray) -> MLXArray {
        let la = config.convLookahead
        guard la > 0 else { return x }
        let t = x.dim(1)
        let kept = x[0..., la ..< t, 0..., 0...]
        let tail = MLX.padded(kept, widths: [IntOrPair(0), IntOrPair((0, la)), IntOrPair(0), IntOrPair(0)], mode: .constant)
        return tail
    }

    /// The clean torch boundary. `spec [B, T, F, 2]`, `featERB [B, T, 32, 1]`, `featSpec [B, T, 96, 2]`.
    public func callAsFunction(spec: MLXArray, featERB: MLXArray, featSpec: MLXArray)
        -> (specE: MLXArray, mask: MLXArray, coefs: MLXArray, lsnr: MLXArray) {
        let (emb, e0, e1, e2, e3, c0, lsnr) = enc(featERB: padFeat(featERB), featSpec: padFeat(featSpec))
        let m = erbDec(emb: emb, e0: e0, e1: e1, e2: e2, e3: e3)        // [B, T, 32, 1]
        let coefs = dfDec(emb: emb, c0: c0)                            // [B, dfOrder, T, nbDF, 2]

        // ERB mask → bins (m · erb_inv_fb) and multiply the spectrum → spec_m.
        let maskBins = matmul(m.squeezed(axis: 3), erbInvFB).expandedDimensions(axis: 3)  // [B, T, F, 1]
        let specM = spec * maskBins

        // Deep filter the lowest nbDF bins (5-tap causal complex FIR); higher bins take spec_m.
        let low = applyDeepFilter(spec: spec, coefs: coefs)            // [B, T, nbDF, 2]
        let high = specM[0..., 0..., config.nbDF..., 0...]
        let specE = concatenated([low, high], axis: 2)
        return (specE, m, coefs, lsnr)
    }

    /// The deep filter (`MF.DF`): the lowest `nbDF` bins are padded in time by `(dfOrder-1-lookahead,
    /// lookahead)` and each output frame is the complex MAC of a `dfOrder`-frame window with its
    /// coefficients (`out[t,f] = Σ_n spec[t+n-past, f]·coef[n,t,f]`).
    private func applyDeepFilter(spec: MLXArray, coefs: MLXArray) -> MLXArray {
        let (b, t) = (spec.dim(0), spec.dim(1))
        let n = config.dfOrder
        let low = spec[0..., 0..., .stride(to: config.nbDF), 0...]      // [B, T, nbDF, 2]
        let past = n - 1 - config.dfLookahead
        let padded = MLX.padded(low, widths: [IntOrPair(0), IntOrPair((past, config.dfLookahead)), IntOrPair(0), IntOrPair(0)], mode: .constant)
        var reAcc = MLXArray.zeros([b, t, config.nbDF])
        var imAcc = MLXArray.zeros([b, t, config.nbDF])
        for tap in 0 ..< n {
            let window = padded[0..., .stride(from: tap, to: tap + t), 0..., 0...]   // [B, T, nbDF, 2]
            let cr = coefs[0..., tap, 0..., 0..., 0]                                  // [B, T, nbDF]
            let ci = coefs[0..., tap, 0..., 0..., 1]
            let sr = window[0..., 0..., 0..., 0]
            let si = window[0..., 0..., 0..., 1]
            reAcc = reAcc + sr * cr - si * ci
            imAcc = imAcc + sr * ci + si * cr
        }
        return stacked([reAcc, imAcc], axis: 3)                        // [B, T, nbDF, 2]
    }
}
