// DeepFilterNet3 (Rikorose/DeepFilterNet, dual MIT/Apache-2.0): real-time 48 kHz speech denoising. The
// NEURAL NET (`DfNet`) is a clean torch `nn.Module` — this file. The STFT / ERB / normalization DSP is
// Rust (`libdf`) and is reproduced separately in MLX validated against libdf recordings; the deep-filter
// APPLY and the ERB-mask apply live inside the net, so they are here.
//
// Grounded on the released source read 2026-09-05: `df/deepfilternet3.py`, `df/modules.py`,
// `df/multiframe.py`, `config.ini`. Tensors flow in MLX NHWC `[B, T, F, C]` where the reference is
// NCHW `[B, C, T, F]`. Net boundary: `forward(spec, feat_erb, feat_spec) -> (spec_e, m, lsnr, df_coefs)`.
//
// SCAFFOLD STATUS: module structure and checkpoint key names mirror `DfNet`. Parity-sensitive choices
// are marked `PARITY:`; the three details the recon flagged for verbatim confirmation are `FLAGGED:`.
// Pinned against `run_reference.py deepfilternet` at the net boundary (oracle + test are the next
// increment). Cross-check `mlx-community/DeepFilterNet-mlx` for the weight-key layout.

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
    public var dfHiddenDim: Int
    public var encLinearGroups: Int
    public var linearGroups: Int
    public var dfPathwayKernelT: Int

    public var bins: Int { fftSize / 2 + 1 }

    public init(sampleRate: Int = 48000, fftSize: Int = 960, hopSize: Int = 480, nbERB: Int = 32,
                nbDF: Int = 96, dfOrder: Int = 5, dfLookahead: Int = 2, convLookahead: Int = 2,
                convCh: Int = 64, embHiddenDim: Int = 256, dfHiddenDim: Int = 256,
                encLinearGroups: Int = 32, linearGroups: Int = 16, dfPathwayKernelT: Int = 5) {
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
        self.dfHiddenDim = dfHiddenDim
        self.encLinearGroups = encLinearGroups
        self.linearGroups = linearGroups
        self.dfPathwayKernelT = dfPathwayKernelT
    }
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
        // [B, T, in] → [g, B*T, in/g] → matmul weight [g, in/g, out/g] → [g, B*T, out/g] → [B, T, out].
        let xg = x.reshaped([b * t, groups, inPerG]).transposed(1, 0, 2)
        let og = matmul(xg, weight).transposed(1, 0, 2)
        return og.reshaped([b, t, groups * outPerG])
    }
}

/// `Conv2dNormAct`: a (separable) convolution with causal time padding and SAME frequency padding, then
/// `BatchNorm2d` and `ReLU`. `separable` inserts a pointwise 1×1 after the depthwise convolution.
/// FLAGGED: the exact per-layer `separable`/`bias` flags follow the DFN3 encoder/decoder config; confirm
/// against the checkpoint's key set.
final class NFKDFConvNormAct: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "conv_pw") var pointwise: Conv2d?
    @ModuleInfo(key: "norm") var norm: BatchNorm
    let padT: (Int, Int)
    let padF: Int

    init(inCh: Int, outCh: Int, kernel: (Int, Int), fstride: Int, dilation: Int,
         lookahead: Int, separable: Bool, bias: Bool) {
        padT = (kernel.0 - 1 - lookahead, lookahead)
        padF = kernel.1 / 2 + dilation - 1
        if separable {
            _conv.wrappedValue = Conv2d(inputChannels: inCh, outputChannels: inCh, kernelSize: IntOrPair(kernel),
                                        stride: IntOrPair((1, fstride)), padding: IntOrPair(0),
                                        dilation: IntOrPair((1, dilation)), groups: inCh, bias: bias)
            _pointwise.wrappedValue = Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 1, bias: false)
        } else {
            _conv.wrappedValue = Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: IntOrPair(kernel),
                                        stride: IntOrPair((1, fstride)), padding: IntOrPair(0),
                                        dilation: IntOrPair((1, dilation)), groups: 1, bias: bias)
            _pointwise.wrappedValue = nil
        }
        _norm.wrappedValue = BatchNorm(featureCount: outCh)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // NHWC: pad time (axis 1) causally, frequency (axis 2) symmetrically.
        var h = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(padT), IntOrPair((padF, padF)), IntOrPair(0)], mode: .constant)
        h = conv(h)
        if let pointwise { h = pointwise(h) }
        return relu(norm(h))
    }
}

/// `SqueezedGRU_S`: `GroupedLinear → ReLU → GRU → GroupedLinear` (batch_first), with an optional grouped
/// skip. The GRU is the shared `NFKMLXGRUCell` (loaded through `NFKMLXRecurrentFold`).
final class NFKDFSqueezedGRU: Module {
    @ModuleInfo(key: "linear_in") var linearIn: NFKDFGroupedLinear
    @ModuleInfo(key: "gru") var gru: NFKMLXGRUCell
    @ModuleInfo(key: "linear_out") var linearOut: NFKDFGroupedLinear

    init(inputSize: Int, hidden: Int, output: Int, groups: Int) {
        _linearIn.wrappedValue = NFKDFGroupedLinear(inFeatures: inputSize, outFeatures: hidden, groups: groups)
        _gru.wrappedValue = NFKMLXGRUCell(inputSize: hidden, hiddenSize: hidden)
        _linearOut.wrappedValue = NFKDFGroupedLinear(inFeatures: hidden, outFeatures: output, groups: groups)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearOut(gru(relu(linearIn(x))))
    }
}

// MARK: - Encoder / decoders

/// The encoder: an ERB convolution pathway and a DF convolution pathway, merged into `emb_gru`.
/// FLAGGED: the exact ERB/DF embedding combine (sum vs concat, `enc_concat=False`) needs verbatim
/// confirmation from `Encoder.forward`.
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

    init(_ config: NFKMLXDeepFilterNetConfiguration) {
        self.config = config
        let c = config.convCh, la = config.convLookahead
        _erbConv0.wrappedValue = NFKDFConvNormAct(inCh: 1, outCh: c, kernel: (3, 3), fstride: 1, dilation: 1, lookahead: la, separable: true, bias: false)
        _erbConv1.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 3), fstride: 2, dilation: 1, lookahead: la, separable: true, bias: false)
        _erbConv2.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 3), fstride: 2, dilation: 1, lookahead: la, separable: true, bias: false)
        _erbConv3.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 3), fstride: 1, dilation: 1, lookahead: la, separable: true, bias: false)
        _dfConv0.wrappedValue = NFKDFConvNormAct(inCh: 2, outCh: c, kernel: (3, 3), fstride: 1, dilation: 1, lookahead: la, separable: true, bias: false)
        _dfConv1.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 3), fstride: 2, dilation: 1, lookahead: la, separable: true, bias: false)
        let embIn = c * config.nbERB / 4                               // 64·8 = 512
        _dfFCEmb.wrappedValue = NFKDFGroupedLinear(inFeatures: c * config.nbDF / 2, outFeatures: embIn, groups: config.encLinearGroups)
        _embGRU.wrappedValue = NFKDFSqueezedGRU(inputSize: embIn, hidden: config.embHiddenDim, output: embIn, groups: config.linearGroups)
        _lsnrFC.wrappedValue = Linear(embIn, 1)
    }

    /// Returns the recurrent embedding `[B, T, embIn]`, the ERB skips (e0..e3), and the DF skip (c0).
    func callAsFunction(featERB: MLXArray, featSpec: MLXArray)
        -> (emb: MLXArray, e0: MLXArray, e1: MLXArray, e2: MLXArray, e3: MLXArray, c0: MLXArray) {
        let e0 = erbConv0(featERB)
        let e1 = erbConv1(e0)
        let e2 = erbConv2(e1)
        let e3 = erbConv3(e2)                                           // [B, T, 8, C]
        let c0 = dfConv1(dfConv0(featSpec))                            // [B, T, 48, C]
        let (b, t) = (e3.dim(0), e3.dim(1))
        let erbFlat = e3.reshaped([b, t, e3.dim(2) * e3.dim(3)])        // [B, T, 512]
        let dfFlat = c0.reshaped([b, t, c0.dim(2) * c0.dim(3)])        // [B, T, 3072]
        // FLAGGED: combine — the reference sums the two embeddings (enc_concat=False). Confirm.
        let embIn = erbFlat + dfFCEmb(dfFlat)
        let emb = embGRU(embIn)
        return (emb, e0, e1, e2, e3, c0)
    }
}

/// The ERB decoder: `emb_gru` then a U-net of skip projections and (transposed) convolutions to the
/// 32-band ERB mask through a `Sigmoid`.
final class NFKDFErbDecoder: Module {
    @ModuleInfo(key: "emb_gru") var embGRU: NFKDFSqueezedGRU
    @ModuleInfo(key: "conv3p") var conv3p: Conv2d
    @ModuleInfo(key: "convt3") var convt3: NFKDFConvNormAct
    @ModuleInfo(key: "conv2p") var conv2p: Conv2d
    @ModuleInfo(key: "convt2") var convt2: NFKDFConvTranspose
    @ModuleInfo(key: "conv1p") var conv1p: Conv2d
    @ModuleInfo(key: "convt1") var convt1: NFKDFConvTranspose
    @ModuleInfo(key: "conv0p") var conv0p: Conv2d
    @ModuleInfo(key: "conv0_out") var conv0Out: NFKDFConvNormAct

    init(_ config: NFKMLXDeepFilterNetConfiguration) {
        let c = config.convCh
        let embIn = c * config.nbERB / 4
        _embGRU.wrappedValue = NFKDFSqueezedGRU(inputSize: embIn, hidden: config.embHiddenDim, output: embIn, groups: config.linearGroups)
        _conv3p.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
        _convt3.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: c, kernel: (1, 3), fstride: 1, dilation: 1, lookahead: config.convLookahead, separable: true, bias: false)
        _conv2p.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
        _convt2.wrappedValue = NFKDFConvTranspose(inCh: c, outCh: c, lookahead: config.convLookahead)
        _conv1p.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
        _convt1.wrappedValue = NFKDFConvTranspose(inCh: c, outCh: c, lookahead: config.convLookahead)
        _conv0p.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
        _conv0Out.wrappedValue = NFKDFConvNormAct(inCh: c, outCh: 1, kernel: (1, 3), fstride: 1, dilation: 1, lookahead: config.convLookahead, separable: true, bias: false)
    }

    /// `emb [B, T, embIn]` + the ERB skips → ERB mask `[B, T, 32, 1]` (Sigmoid is applied by the caller
    /// via `conv0_out`'s activation; DFN3's `conv0_out` uses a Sigmoid final — PARITY).
    func callAsFunction(emb: MLXArray, e0: MLXArray, e1: MLXArray, e2: MLXArray, e3: MLXArray) -> MLXArray {
        let (b, t) = (e3.dim(0), e3.dim(1))
        let hidden = embGRU(emb).reshaped([b, t, e3.dim(2), e3.dim(3)])
        var x = convt3(hidden + conv3p(e3))
        x = convt2(x + conv2p(e2))
        x = convt1(x + conv1p(e1))
        x = conv0Out(x + conv0p(e0))
        return sigmoid(x)                                              // PARITY: conv0_out final Sigmoid
    }
}

/// A frequency-upsampling `ConvTranspose2d` with causal time padding, `BatchNorm2d`, `ReLU` (the ERB
/// decoder's `convt2`/`convt1`, fstride 2 on frequency).
final class NFKDFConvTranspose: Module {
    @ModuleInfo(key: "conv") var conv: ConvTransposed2d
    @ModuleInfo(key: "norm") var norm: BatchNorm
    let padT: (Int, Int)

    init(inCh: Int, outCh: Int, lookahead: Int) {
        padT = (1 - 1 - 0, 0)                                          // FLAGGED: exact convt time pad / output_padding
        _conv.wrappedValue = ConvTransposed2d(inputChannels: inCh, outputChannels: outCh,
                                              kernelSize: IntOrPair((1, 3)), stride: IntOrPair((1, 2)),
                                              padding: IntOrPair((0, 1)))
        _norm.wrappedValue = BatchNorm(featureCount: outCh)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { relu(norm(conv(x))) }
}

/// The DF decoder: `df_gru` (+ grouped skip) then a grouped-linear head (Tanh) and a local pathway
/// convolution, summed into the deep-filter coefficients; plus `df_fc_a` (Sigmoid) → the blend alpha.
/// FLAGGED: `df_convp` out-channels and the `df_skip` dims need verbatim confirmation.
final class NFKDFDfDecoder: Module {
    @ModuleInfo(key: "df_gru") var dfGRU: NFKDFSqueezedGRU
    @ModuleInfo(key: "df_skip") var dfSkip: NFKDFGroupedLinear
    @ModuleInfo(key: "df_convp") var dfConvP: NFKDFConvNormAct
    @ModuleInfo(key: "df_out") var dfOut: NFKDFGroupedLinear
    @ModuleInfo(key: "df_fc_a") var dfFCA: Linear
    let config: NFKMLXDeepFilterNetConfiguration

    init(_ config: NFKMLXDeepFilterNetConfiguration) {
        self.config = config
        let embIn = config.convCh * config.nbERB / 4
        _dfGRU.wrappedValue = NFKDFSqueezedGRU(inputSize: embIn, hidden: config.dfHiddenDim, output: config.dfHiddenDim, groups: config.linearGroups)
        _dfSkip.wrappedValue = NFKDFGroupedLinear(inFeatures: embIn, outFeatures: config.dfHiddenDim, groups: config.linearGroups)
        _dfConvP.wrappedValue = NFKDFConvNormAct(inCh: config.convCh, outCh: config.dfOrder * 2, kernel: (config.dfPathwayKernelT, 1), fstride: 1, dilation: 1, lookahead: 0, separable: true, bias: false)
        _dfOut.wrappedValue = NFKDFGroupedLinear(inFeatures: config.dfHiddenDim, outFeatures: config.nbDF * config.dfOrder * 2, groups: config.linearGroups)
        _dfFCA.wrappedValue = Linear(config.dfHiddenDim, 1)
    }

    /// Returns the deep-filter coefficients `[B, dfOrder, T, nbDF, 2]` and the alpha `[B, T, 1]`.
    func callAsFunction(emb: MLXArray, c0: MLXArray) -> (coefs: MLXArray, alpha: MLXArray) {
        let (b, t) = (emb.dim(0), emb.dim(1))
        let g = dfGRU(emb) + dfSkip(emb)
        let head = tanh(dfOut(g)).reshaped([b, t, config.nbDF, config.dfOrder * 2])
        let local = dfConvP(c0)                                        // [B, T, nbDF, dfOrder·2]
        let coefsFlat = head + local.reshaped([b, t, config.nbDF, config.dfOrder * 2])
        // → [B, dfOrder, T, nbDF, 2]
        let coefs = coefsFlat.reshaped([b, t, config.nbDF, config.dfOrder, 2]).transposed(0, 3, 1, 2, 4)
        let alpha = sigmoid(dfFCA(g))
        return (coefs, alpha)
    }
}

// MARK: - DfNet

/// The DeepFilterNet3 network. Inputs the pre-computed features (from the libdf DSP): the raw complex
/// spectrogram `spec [B, T, F, 2]`, `feat_erb [B, T, 32, 1]`, `feat_spec [B, T, 96, 2]`. Outputs the
/// enhanced spectrogram, the ERB mask, the LSNR, and the deep-filter coefficients.
public final class NFKMLXDeepFilterNet: Module {
    @ModuleInfo(key: "enc") var enc: NFKDFEncoder
    @ModuleInfo(key: "erb_dec") var erbDec: NFKDFErbDecoder
    @ModuleInfo(key: "df_dec") var dfDec: NFKDFDfDecoder
    @ParameterInfo(key: "erb_inv_fb") var erbInvFB: MLXArray            // [32, 481] checkpoint buffer
    let config: NFKMLXDeepFilterNetConfiguration

    public init(_ config: NFKMLXDeepFilterNetConfiguration) {
        self.config = config
        _enc.wrappedValue = NFKDFEncoder(config)
        _erbDec.wrappedValue = NFKDFErbDecoder(config)
        _dfDec.wrappedValue = NFKDFDfDecoder(config)
        _erbInvFB.wrappedValue = MLXArray.zeros([config.nbERB, config.bins])
    }

    /// The clean torch boundary. `spec [B, T, F, 2]`, `featERB [B, T, 32, 1]`, `featSpec [B, T, 96, 2]`.
    public func callAsFunction(spec: MLXArray, featERB: MLXArray, featSpec: MLXArray)
        -> (specE: MLXArray, mask: MLXArray, coefs: MLXArray) {
        let (emb, e0, e1, e2, e3, c0) = enc(featERB: featERB, featSpec: featSpec)
        let m = erbDec(emb: emb, e0: e0, e1: e1, e2: e2, e3: e3)        // [B, T, 32, 1]
        let (coefs, alpha) = dfDec(emb: emb, c0: c0)

        // ERB mask → bins (m · erb_inv_fb) and multiply the spectrum.
        let maskBins = matmul(m.squeezed(axis: 3), erbInvFB).expandedDimensions(axis: 3)  // [B, T, F, 1]
        let specM = spec * maskBins

        // Deep filter the lowest nbDF bins: 5-tap causal complex FIR over frames.
        let deep = applyDeepFilter(spec: spec, coefs: coefs)           // [B, T, nbDF, 2]
        let low = deep
        let high = specM[0..., 0..., config.nbDF..., 0...]
        // FLAGGED: the alpha blend between specM and the deep-filtered low band in DfNet.forward.
        _ = alpha
        let specE = concatenated([low, high], axis: 2)
        return (specE, m, coefs)
    }

    /// The deep filter: for each of the lowest `nbDF` bins, a complex FIR over a causal window of
    /// `dfOrder` frames (pad `(dfOrder-1-lookahead, lookahead)`). `takeAlong`-style gather + complex mul.
    private func applyDeepFilter(spec: MLXArray, coefs: MLXArray) -> MLXArray {
        let (b, t) = (spec.dim(0), spec.dim(1))
        let n = config.dfOrder
        let low = spec[0..., 0..., .stride(to: config.nbDF), 0...]      // [B, T, nbDF, 2]
        let padPast = n - 1 - config.dfLookahead
        let padded = MLX.padded(low, widths: [IntOrPair(0), IntOrPair((padPast, config.dfLookahead)), IntOrPair(0), IntOrPair(0)], mode: .constant)
        // Unfold time into windows of n: [B, T, nbDF, n, 2].
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

// MARK: - Registration and weight loading

/// Registration and weight loading for DeepFilterNet3.
@objc(NFKMLXDeepFilterNet_Factory)
public final class NFKMLXDeepFilterNetFactory: NSObject {
    @objc public static let modelName = "deepfilternet3"

    static func makeNet(_ config: NFKMLXDeepFilterNetConfiguration = .init()) -> NFKMLXDeepFilterNet {
        NFKMLXDeepFilterNet(config)
    }

    /// Loads a released DeepFilterNet3 checkpoint into `DfNet`: fold the GRUs, drop BatchNorm counters,
    /// remap the `Conv2dNormAct`/`SqueezedGRU_S` Sequential names, and transpose the convolutions.
    /// GroupedLinear weights are 3-D `[g, in/g, out/g]` and pass through unchanged.
    static func loadWeights(into net: NFKMLXDeepFilterNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let folded = NFKMLXRecurrentFold.fold(checkpoint.arrays)
        let mapped: [(String, MLXArray)] = folded.compactMap { key, value in
            guard let name = remapReferenceKey(key) else { return nil }
            let tensor: MLXArray
            if name.contains("df_dec.df_gru") || name.contains("emb_gru") || name.contains("df_fc_emb") ||
                name.contains("df_out") || name.contains("df_skip") || name.contains("linear_in") ||
                name.contains("linear_out") || name.hasSuffix("erb_inv_fb") {
                tensor = value                                          // GroupedLinear / buffer: no transpose
            } else if value.ndim == 4, checkpoint.needsConvTranspose {
                // ConvTranspose weights (convt) take the transposed-conv axis order; forward convs (0,2,3,1).
                tensor = name.contains(".convt") ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1)
            } else {
                tensor = value
            }
            return (name, tensor)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// FLAGGED: the `Conv2dNormAct`/`ConvTranspose2dNormAct` submodule layout (`.0`/`.1` for the conv and
    /// the pointwise/norm) and the `SqueezedGRU_S` `linear_in.0`/`linear_out.0` indices need confirming
    /// against the checkpoint key set; this remap is the first approximation.
    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasSuffix("num_batches_tracked") { return nil }
        var name = key
        // SqueezedGRU_S linear_in/out are Sequential(GroupedLinear, act): index 0 → the grouped linear.
        name = name.replacingOccurrences(of: "linear_in.0.", with: "linear_in.")
        name = name.replacingOccurrences(of: "linear_out.0.", with: "linear_out.")
        // Conv2dNormAct: `.conv.0` depthwise, `.conv.1` pointwise (separable), `.norm` BatchNorm.
        name = name.replacingOccurrences(of: ".conv.0.", with: ".conv.")
        name = name.replacingOccurrences(of: ".conv.1.", with: ".conv_pw.")
        return name
    }

    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { _ in throw NFKMLXError.unsupportedInput }
        // NOTE: DFN3 needs the libdf DSP feature front end (STFT/ERB/norm) before a runnable backend;
        // the net registers for parity harness use, the backend is the next increment.
    }
}
