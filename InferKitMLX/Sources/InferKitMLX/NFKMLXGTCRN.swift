// GTCRN (Xiaobin-Rong/gtcrn, MIT): an ultra-light (~48.2K parameter) grouped temporal convolutional
// recurrent network for real-time speech enhancement. The second speech-restoration port, and the tiny
// on-device denoiser the heavyweight restorers cannot be — it runs per frame in real time.
//
// Grounded against the released `gtcrn.py` (module names and forward flow read from source, not memory).
// Pipeline over a complex spectrogram: stack [magnitude, real, imag] → ERB band-merge (257 → 129) →
// subband feature extraction → a 5-block grouped-conv encoder → two dual-path grouped-RNN blocks → a
// 5-block grouped-conv decoder with skips → ERB band-split (129 → 257) → a complex ratio mask applied to
// the input spectrogram → iSTFT.
//
// SCAFFOLD STATUS: module tree and checkpoint keys load the released weights (`erb.`, `sfe.`,
// `encoder.en_convs.{i}`, `dpgrnn{1,2}.`, `decoder.de_convs.{i}`); the loader needs only the 4-D conv
// transpose and the shared GRU fold. Numeric choices are marked `PARITY:` — the reshape-heavy paths
// (DPGRNN dual-path permutes, the deconv temporal handling, the channel shuffle, SFE unfold order, the
// ERB projection axis, the STFT window) are pinned against `run_reference.py gtcrn`. Tensors flow in MLX
// NHWC `[B, T, F, C]` where the reference is NCHW `[B, C, T, F]`.
//
// NOTE: GTCRN's ERB is a LEARNED `nn.Linear` (the ERB filterbank only initializes it), so it loads as a
// Linear rather than through the `NFKMLXERB` primitive — that primitive's first real user is DeepFilterNet.

import Foundation
import InferKit
import MLX
import MLXNN

/// The released GTCRN configuration.
public struct NFKMLXGTCRNConfiguration: Sendable {
    public var sampleRate: Int
    public var fftSize: Int
    public var hopSize: Int
    public var nfreqs: Int          // fftSize/2 + 1 = 257
    public var erbLowBins: Int      // low bins kept unmerged (65)
    public var erbBands: Int        // ERB bands for the high frequencies (64)
    public var channels: Int        // encoder width (16)
    public var width: Int           // frequency width after the encoder (33)

    public init(sampleRate: Int = 16000, fftSize: Int = 512, hopSize: Int = 256,
                erbLowBins: Int = 65, erbBands: Int = 64, channels: Int = 16, width: Int = 33) {
        self.sampleRate = sampleRate
        self.fftSize = fftSize
        self.hopSize = hopSize
        self.nfreqs = fftSize / 2 + 1
        self.erbLowBins = erbLowBins
        self.erbBands = erbBands
        self.channels = channels
        self.width = width
    }
}

// MARK: - ERB band merge / split

/// The ERB frequency compression: the low `erbLowBins` bins pass through, and the high bins are
/// projected to `erbBands` by a learned linear (`erb_fc`), band-split back by its inverse (`ierb_fc`).
/// The projection runs over the frequency axis (axis 2 in NHWC), shared across batch, time, and channel.
final class NFKGTCRNERB: Module {
    @ModuleInfo(key: "erb_fc") var erbFC: Linear
    @ModuleInfo(key: "ierb_fc") var ierbFC: Linear
    let lowBins: Int

    init(_ config: NFKMLXGTCRNConfiguration) {
        lowBins = config.erbLowBins
        // The reference's erb_fc/ierb_fc are bias-FREE Linears holding the fixed ERB filterbank
        // (erb_filters and its transpose, requires_grad=False).
        let high = config.nfreqs - config.erbLowBins
        _erbFC.wrappedValue = Linear(high, config.erbBands, bias: false)
        _ierbFC.wrappedValue = Linear(config.erbBands, high, bias: false)
    }

    /// PARITY: the reference applies the linear on the last (frequency) axis of an NCHW tensor; here the
    /// frequency is axis 2, so the high band is moved to the last axis for the projection and moved back.
    private func project(_ x: MLXArray, through linear: Linear) -> MLXArray {
        linear(x.transposed(0, 1, 3, 2)).transposed(0, 1, 3, 2)
    }

    /// Band merge: `[B, T, 257, C]` → `[B, T, 129, C]`.
    func bandMerge(_ x: MLXArray) -> MLXArray {
        let low = x[0..., 0..., 0 ..< lowBins, 0...]
        let high = project(x[0..., 0..., lowBins..., 0...], through: erbFC)
        return concatenated([low, high], axis: 2)
    }

    /// Band split: `[B, T, 129, C]` → `[B, T, 257, C]`.
    func bandSplit(_ x: MLXArray) -> MLXArray {
        let low = x[0..., 0..., 0 ..< lowBins, 0...]
        let high = project(x[0..., 0..., lowBins..., 0...], through: ierbFC)
        return concatenated([low, high], axis: 2)
    }
}

// MARK: - Subband feature extraction

/// Subband feature extraction: a frequency unfold of kernel 3 (`nn.Unfold((1,3), padding=(0,1))`), which
/// stacks each frequency's `[f-1, f, f+1]` neighborhood into the channel axis. PARITY: the reference's
/// channel order is `c` outer, kernel position inner (`c*3 + k`).
enum NFKGTCRNSFE {
    static func apply(_ x: MLXArray) -> MLXArray {                          // [B, T, F, C] → [B, T, F, 3C]
        let (b, t, f, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((1, 1)), IntOrPair(0)], mode: .constant)
        let g0 = padded[0..., 0..., 0 ..< f, 0...]
        let g1 = padded[0..., 0..., 1 ..< f + 1, 0...]
        let g2 = padded[0..., 0..., 2 ..< f + 2, 0...]
        return stacked([g0, g1, g2], axis: 4).reshaped([b, t, f, c * 3])     // c outer, k inner
    }
}

// MARK: - Conv block

/// A convolution block: `Conv2d` (or `ConvTransposed2d` when `deconv`) + `BatchNorm` + `PReLU` (or a
/// tanh on the final block, which carries no parameters). Matches the reference `conv` / `bn` / `act`.
final class NFKGTCRNConvBlock: Module {
    @ModuleInfo(key: "conv") var conv: Module                              // Conv2d or ConvTransposed2d
    @ModuleInfo(key: "bn") var bn: BatchNorm
    @ModuleInfo(key: "act") var act: PReLU?                                // nil on the final (tanh) block
    let isLast: Bool

    init(inChannels: Int, outChannels: Int, kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int),
         groups: Int = 1, deconv: Bool = false, isLast: Bool = false) {
        self.isLast = isLast
        if deconv {
            _conv.wrappedValue = ConvTransposed2d(inputChannels: inChannels, outputChannels: outChannels,
                                                  kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                                  padding: IntOrPair(padding), groups: groups)
        } else {
            _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                        kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                        padding: IntOrPair(padding), groups: groups)
        }
        _bn.wrappedValue = BatchNorm(featureCount: outChannels)
        _act.wrappedValue = isLast ? nil : PReLU(count: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let convolved: MLXArray
        if let c = conv as? ConvTransposed2d, c.groups > 1 {
            // MLX's GROUPED transposed convolution does not match PyTorch's grouping, so run each group
            // as its own groups=1 transposed convolution (the groups=1 path is verified against the
            // reference). The weight is already [out, kH, kW, in/groups], sliced per output group.
            let inPer = x.dim(3) / c.groups, outPer = c.weight.dim(0) / c.groups
            var parts = [MLXArray]()
            for group in 0 ..< c.groups {
                let xg = x[0..., 0..., 0..., group * inPer ..< (group + 1) * inPer]
                let wg = c.weight[group * outPer ..< (group + 1) * outPer]
                parts.append(convTransposed2d(xg, wg, stride: IntOrPair(c.stride),
                                              padding: IntOrPair(c.padding), dilation: IntOrPair(c.dilation)))
            }
            var merged = concatenated(parts, axis: 3)
            if let bias = c.bias { merged = merged + bias }
            convolved = merged
        } else {
            convolved = (conv as? Conv2d)?.callAsFunction(x) ?? (conv as! ConvTransposed2d)(x)
        }
        let normed = bn(convolved)
        if let act { return act(normed) }
        return tanh(normed)
    }
}

// MARK: - Temporal recurrent attention

/// Temporal recurrent attention: a per-channel gate from the frame-power sequence through a GRU, a
/// linear, and a sigmoid, multiplying the input (`att_gru` / `att_fc`).
final class NFKGTCRNTRA: Module {
    @ModuleInfo(key: "att_gru") var attGRU: NFKMLXGRUCell
    @ModuleInfo(key: "att_fc") var attFC: Linear

    init(channels: Int) {
        _attGRU.wrappedValue = NFKMLXGRUCell(inputSize: channels, hiddenSize: channels * 2)
        _attFC.wrappedValue = Linear(channels * 2, channels)
    }

    /// `x` `[B, T, F, C]` → the gated `[B, T, F, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let power = (x * x).mean(axis: 2)                                   // [B, T, C], mean over frequency
        let gate = sigmoid(attFC(attGRU(power)))                           // [B, T, C]
        let (b, t, c) = (x.dim(0), x.dim(1), x.dim(3))
        return x * gate.reshaped([b, t, 1, c])
    }
}

// MARK: - Grouped temporal conv block

/// The grouped temporal convolution block: half the channels pass through, half go through subband
/// feature extraction, a pointwise contraction, a dilated depthwise convolution (causal in time), a
/// pointwise expansion, and temporal recurrent attention, then the two halves are channel-shuffled back.
final class NFKGTCRNGTConvBlock: Module {
    @ModuleInfo(key: "point_conv1") var pointConv1: Module
    @ModuleInfo(key: "point_bn1") var pointBN1: BatchNorm
    @ModuleInfo(key: "point_act") var pointAct: PReLU
    @ModuleInfo(key: "depth_conv") var depthConv: Module
    @ModuleInfo(key: "depth_bn") var depthBN: BatchNorm
    @ModuleInfo(key: "depth_act") var depthAct: PReLU
    @ModuleInfo(key: "point_conv2") var pointConv2: Module
    @ModuleInfo(key: "point_bn2") var pointBN2: BatchNorm
    @ModuleInfo(key: "tra") var tra: NFKGTCRNTRA
    let dilation: Int
    let deconv: Bool

    init(channels: Int, hidden: Int, dilation: Int, deconv: Bool = false) {
        self.dilation = dilation
        self.deconv = deconv
        let half = channels / 2
        func point(_ inC: Int, _ outC: Int) -> Module {
            deconv ? ConvTransposed2d(inputChannels: inC, outputChannels: outC, kernelSize: 1)
                   : Conv2d(inputChannels: inC, outputChannels: outC, kernelSize: 1)
        }
        _pointConv1.wrappedValue = point(half * 3, hidden)                 // sfe tripled the half-channels
        _pointBN1.wrappedValue = BatchNorm(featureCount: hidden)
        _pointAct.wrappedValue = PReLU(count: 1)
        // Depthwise (3,3), dilation (d,1), freq pad 1; the time axis is causally front-padded by
        // (kernel-1)·dilation in the forward. The transposed convolution then removes that padding
        // back via a time padding of 2·dilation (`padding=(2·dilation, 1)` in the reference), so the
        // frame count is preserved and no crop is needed.
        if deconv {
            _depthConv.wrappedValue = ConvTransposed2d(inputChannels: hidden, outputChannels: hidden,
                                                       kernelSize: IntOrPair((3, 3)), padding: IntOrPair((2 * dilation, 1)),
                                                       dilation: IntOrPair((dilation, 1)), groups: hidden)
        } else {
            _depthConv.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: hidden,
                                             kernelSize: IntOrPair((3, 3)), padding: IntOrPair((0, 1)),
                                             dilation: IntOrPair((dilation, 1)), groups: hidden)
        }
        _depthBN.wrappedValue = BatchNorm(featureCount: hidden)
        _depthAct.wrappedValue = PReLU(count: 1)
        _pointConv2.wrappedValue = point(hidden, half)
        _pointBN2.wrappedValue = BatchNorm(featureCount: half)
        _tra.wrappedValue = NFKGTCRNTRA(channels: half)
    }

    private func conv(_ module: Module, _ x: MLXArray) -> MLXArray {
        (module as? Conv2d)?.callAsFunction(x) ?? (module as! ConvTransposed2d)(x)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, f, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let half = channels / 2
        let x1 = x[0..., 0..., 0..., 0 ..< half]
        let x2 = x[0..., 0..., 0..., half...]
        var h = pointAct(pointBN1(conv(pointConv1, NFKGTCRNSFE.apply(x1))))
        // The reference front-pads the time axis by (kernel-1)·dilation for BOTH the regular and the
        // transposed convolution (`F.pad(h, [0,0,pad_size,0])`); the transposed conv's own time padding
        // of 2·dilation then removes the excess, so the frame count is preserved in both cases.
        let padded = MLX.padded(h, widths: [IntOrPair(0), IntOrPair((2 * dilation, 0)), IntOrPair(0), IntOrPair(0)], mode: .constant)
        h = depthAct(depthBN(conv(depthConv, padded)))
        h = pointBN2(conv(pointConv2, h))
        h = tra(h)
        // shuffle: interleave the processed and pass-through halves, channel = c*2 + g (c outer).
        return stacked([h, x2], axis: 4).reshaped([b, t, f, half * 2])
    }
}

// MARK: - Encoder

/// The encoder: two conv blocks that halve the frequency axis, then three grouped temporal conv blocks
/// at dilations 1, 2, 5. Returns the bottleneck and the per-block outputs for the decoder skips.
final class NFKGTCRNEncoder: Module {
    @ModuleInfo(key: "en_convs") var enConvs: [Module]

    init(_ config: NFKMLXGTCRNConfiguration) {
        let c = config.channels
        _enConvs.wrappedValue = [
            NFKGTCRNConvBlock(inChannels: 9, outChannels: c, kernel: (1, 5), stride: (1, 2), padding: (0, 2)),
            NFKGTCRNConvBlock(inChannels: c, outChannels: c, kernel: (1, 5), stride: (1, 2), padding: (0, 2), groups: 2),
            NFKGTCRNGTConvBlock(channels: c, hidden: c, dilation: 1),
            NFKGTCRNGTConvBlock(channels: c, hidden: c, dilation: 2),
            NFKGTCRNGTConvBlock(channels: c, hidden: c, dilation: 5),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, [MLXArray]) {
        var h = x
        var outputs = [MLXArray]()
        for module in enConvs {
            h = apply(module, h)
            outputs.append(h)
        }
        return (h, outputs)
    }

    private func apply(_ module: Module, _ x: MLXArray) -> MLXArray {
        if let block = module as? NFKGTCRNConvBlock { return block(x) }
        return (module as! NFKGTCRNGTConvBlock)(x)
    }
}

// MARK: - Dual-path grouped RNN

/// A LayerNorm over the last two axes `(width, channels)` with a learned `[width, channels]` weight and
/// bias (`nn.LayerNorm((width, channels))`). PARITY: epsilon 1e-8.
final class NFKGTCRNLayerNorm2D: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray                      // [width, channels]
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(width: Int, channels: Int) {
        _weight.wrappedValue = MLXArray.ones([width, channels])
        _bias.wrappedValue = MLXArray.zeros([width, channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {                        // [.., width, channels]
        let mean = x.mean(axes: [-2, -1], keepDims: true)
        let variance = ((x - mean) * (x - mean)).mean(axes: [-2, -1], keepDims: true)
        return (x - mean) / sqrt(variance + 1e-8) * weight + bias
    }
}

/// A grouped RNN: the feature axis is split into two groups, each run through its own GRU, and the
/// outputs concatenated (`rnn1` / `rnn2`). Bidirectional and unidirectional variants share this shape,
/// differing only in the cell type.
final class NFKGTCRNGroupedBiGRU: Module {
    @ModuleInfo(key: "rnn1") var rnn1: NFKMLXBiGRU
    @ModuleInfo(key: "rnn2") var rnn2: NFKMLXBiGRU

    init(inputSize: Int, hiddenSize: Int) {
        _rnn1.wrappedValue = NFKMLXBiGRU(inputSize: inputSize / 2, hiddenSize: hiddenSize / 2)
        _rnn2.wrappedValue = NFKMLXBiGRU(inputSize: inputSize / 2, hiddenSize: hiddenSize / 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([rnn1(x[.ellipsis, 0 ..< half]), rnn2(x[.ellipsis, half...])], axis: -1)
    }
}

final class NFKGTCRNGroupedGRU: Module {
    @ModuleInfo(key: "rnn1") var rnn1: NFKMLXGRUCell
    @ModuleInfo(key: "rnn2") var rnn2: NFKMLXGRUCell

    init(inputSize: Int, hiddenSize: Int) {
        _rnn1.wrappedValue = NFKMLXGRUCell(inputSize: inputSize / 2, hiddenSize: hiddenSize / 2)
        _rnn2.wrappedValue = NFKMLXGRUCell(inputSize: inputSize / 2, hiddenSize: hiddenSize / 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([rnn1(x[.ellipsis, 0 ..< half]), rnn2(x[.ellipsis, half...])], axis: -1)
    }
}

/// The dual-path grouped RNN: an intra-frame (bidirectional, over frequency) pass and an inter-frame
/// (unidirectional, over time) pass, each a grouped RNN with a linear, a 2-D LayerNorm, and a residual.
final class NFKGTCRNDPGRNN: Module {
    @ModuleInfo(key: "intra_rnn") var intraRNN: NFKGTCRNGroupedBiGRU
    @ModuleInfo(key: "intra_fc") var intraFC: Linear
    @ModuleInfo(key: "intra_ln") var intraLN: NFKGTCRNLayerNorm2D
    @ModuleInfo(key: "inter_rnn") var interRNN: NFKGTCRNGroupedGRU
    @ModuleInfo(key: "inter_fc") var interFC: Linear
    @ModuleInfo(key: "inter_ln") var interLN: NFKGTCRNLayerNorm2D

    init(_ config: NFKMLXGTCRNConfiguration) {
        let (c, w) = (config.channels, config.width)
        _intraRNN.wrappedValue = NFKGTCRNGroupedBiGRU(inputSize: c, hiddenSize: c / 2)
        _intraFC.wrappedValue = Linear(c, c)
        _intraLN.wrappedValue = NFKGTCRNLayerNorm2D(width: w, channels: c)
        _interRNN.wrappedValue = NFKGTCRNGroupedGRU(inputSize: c, hiddenSize: c)
        _interFC.wrappedValue = Linear(c, c)
        _interLN.wrappedValue = NFKGTCRNLayerNorm2D(width: w, channels: c)
    }

    /// `[B, T, F, C]` → `[B, T, F, C]`. PARITY: the intra pass runs over the frequency axis per frame,
    /// the inter pass over the time axis per frequency; the residual and LayerNorm order follow the
    /// reference.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, f, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        // Intra: sequence over frequency, batched over (B, T).
        var intra = intraRNN(x.reshaped([b * t, f, c]))
        intra = intraLN(intraFC(intra).reshaped([b, t, f, c]))
        let afterIntra = x + intra
        // Inter: sequence over time, batched over (B, F).
        let permuted = afterIntra.transposed(0, 2, 1, 3).reshaped([b * f, t, c])
        var inter = interRNN(permuted)
        inter = interFC(inter).reshaped([b, f, t, c])
        let interNorm = interLN(inter.transposed(0, 2, 1, 3))
        return afterIntra + interNorm
    }
}

// MARK: - Decoder

/// The decoder: three grouped temporal deconv blocks, then two conv-transpose blocks that restore the
/// frequency axis, each reading the encoder skip in reverse order.
final class NFKGTCRNDecoder: Module {
    @ModuleInfo(key: "de_convs") var deConvs: [Module]

    init(_ config: NFKMLXGTCRNConfiguration) {
        let c = config.channels
        _deConvs.wrappedValue = [
            NFKGTCRNGTConvBlock(channels: c, hidden: c, dilation: 5, deconv: true),
            NFKGTCRNGTConvBlock(channels: c, hidden: c, dilation: 2, deconv: true),
            NFKGTCRNGTConvBlock(channels: c, hidden: c, dilation: 1, deconv: true),
            NFKGTCRNConvBlock(inChannels: c, outChannels: c, kernel: (1, 5), stride: (1, 2), padding: (0, 2), groups: 2, deconv: true),
            NFKGTCRNConvBlock(inChannels: c, outChannels: 2, kernel: (1, 5), stride: (1, 2), padding: (0, 2), deconv: true, isLast: true),
        ]
    }

    /// `x` the bottleneck, `skips` the encoder outputs in forward order. Output `[B, T, 129, 2]`.
    func callAsFunction(_ x: MLXArray, skips: [MLXArray]) -> MLXArray {
        var h = x
        for (index, module) in deConvs.enumerated() {
            h = apply(module, h + skips[skips.count - 1 - index])
        }
        return h
    }

    private func apply(_ module: Module, _ x: MLXArray) -> MLXArray {
        if let block = module as? NFKGTCRNConvBlock { return block(x) }
        return (module as! NFKGTCRNGTConvBlock)(x)
    }
}

// MARK: - The network

/// The GTCRN generator: ERB merge, subband features, the encoder, two dual-path RNN blocks, the decoder,
/// the ERB split into a complex mask, and the complex product with the input spectrogram.
public final class NFKMLXGTCRN: Module {
    @ModuleInfo(key: "erb") var erb: NFKGTCRNERB
    @ModuleInfo(key: "encoder") var encoder: NFKGTCRNEncoder
    @ModuleInfo(key: "dpgrnn1") var dpgrnn1: NFKGTCRNDPGRNN
    @ModuleInfo(key: "dpgrnn2") var dpgrnn2: NFKGTCRNDPGRNN
    @ModuleInfo(key: "decoder") var decoder: NFKGTCRNDecoder

    public init(_ config: NFKMLXGTCRNConfiguration) {
        _erb.wrappedValue = NFKGTCRNERB(config)
        _encoder.wrappedValue = NFKGTCRNEncoder(config)
        _dpgrnn1.wrappedValue = NFKGTCRNDPGRNN(config)
        _dpgrnn2.wrappedValue = NFKGTCRNDPGRNN(config)
        _decoder.wrappedValue = NFKGTCRNDecoder(config)
    }

    /// Real and imaginary spectrograms, each `[1, bins, frames]` → enhanced real and imaginary, each
    /// `[1, bins, frames]`.
    public func callAsFunction(real: MLXArray, imaginary: MLXArray) -> (real: MLXArray, imaginary: MLXArray) {
        // [1, F, T] → [1, T, F, 1], stack [mag, real, imag] on the channel axis.
        let realTF = real.transposed(0, 2, 1).expandedDimensions(axis: 3)
        let imagTF = imaginary.transposed(0, 2, 1).expandedDimensions(axis: 3)
        let magTF = sqrt(realTF * realTF + imagTF * imagTF)
        var h = concatenated([magTF, realTF, imagTF], axis: 3)             // [1, T, 257, 3]
        h = erb.bandMerge(h)                                               // [1, T, 129, 3]
        h = NFKGTCRNSFE.apply(h)                                           // [1, T, 129, 9]
        let (bottleneck, skips) = encoder(h)                              // [1, T, 33, 16]
        var g = dpgrnn1(bottleneck)
        g = dpgrnn2(g)
        var mask = decoder(g, skips: skips)                               // [1, T, 129, 2]
        mask = erb.bandSplit(mask)                                        // [1, T, 257, 2]
        // Complex ratio mask applied to the input spectrogram.
        let maskReal = mask[0..., 0..., 0..., 0 ..< 1]
        let maskImag = mask[0..., 0..., 0..., 1...]
        let outReal = realTF * maskReal - imagTF * maskImag
        let outImag = imagTF * maskReal + realTF * maskImag
        func toBinsFrames(_ y: MLXArray) -> MLXArray { y.squeezed(axis: 3).transposed(0, 2, 1) }
        return (toBinsFrames(outReal), toBinsFrames(outImag))
    }
}

// MARK: - Backend

private final class NFKGTCRNHolder: @unchecked Sendable {
    let net: NFKMLXGTCRN
    let config: NFKMLXGTCRNConfiguration
    init(_ net: NFKMLXGTCRN, _ config: NFKMLXGTCRNConfiguration) { self.net = net; self.config = config }
}

/// Real-time speech denoising as an InferKit backend. Reads `NFKInputAudio`; returns the enhanced clip
/// as a single `NFKAudioAsset` under `NFKOutputAudio`.
@objc(NFKMLXGTCRNBackend)
public final class NFKMLXGTCRNBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKGTCRNHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXGTCRN, config: NFKMLXGTCRNConfiguration, identifier: String,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKGTCRNHolder(net, config)
        self.identifier = identifier
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, _) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let config = holder.config
        let enhanced = Self.enhance(samples, net: holder.net, config: config)
        eval(enhanced)
        let stream = enhanced.reshaped([enhanced.shape.last!]).asArray(Float.self)

        let url = outputDirectory.appendingPathComponent("gtcrn-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: stream, sampleRate: config.sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(stream.count) / Double(config.sampleRate),
                                  sampleRate: Double(config.sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    /// STFT → the network's complex mask → iSTFT. Exposed for the parity harness. GTCRN's analysis and
    /// synthesis window is a SQUARE-ROOT Hann (`infer.py`: `torch.hann_window(512).pow(0.5)`).
    static func enhance(_ samples: [Float], net: NFKMLXGTCRN, config: NFKMLXGTCRNConfiguration) -> MLXArray {
        let window = MLXArray(nfkPeriodicHann(config.fftSize).map { sqrtf($0) })
        let stft = NFKMLXComplexSTFT(nFFT: config.fftSize, hop: config.hopSize, window: window)
        let signal = samples.withUnsafeBufferPointer { MLXArray($0, [1, samples.count]) }
        let (magnitude, phase) = stft.transform(signal)
        let real = magnitude * cos(phase), imaginary = magnitude * sin(phase)
        let (enhancedReal, enhancedImag) = net(real: real, imaginary: imaginary)
        let enhancedMag = sqrt(enhancedReal * enhancedReal + enhancedImag * enhancedImag)
        let enhancedPhase = atan2(enhancedImag, enhancedReal)
        return stft.inverse(magnitude: enhancedMag, phase: enhancedPhase)
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

/// Registration and weight loading for GTCRN.
@objc(NFKMLXGTCRN_Factory)
public final class NFKMLXGTCRNFactory: NSObject {
    @objc public static let modelName = "gtcrn"

    static func makeNet(_ config: NFKMLXGTCRNConfiguration = .init()) -> NFKMLXGTCRN {
        let net = NFKMLXGTCRN(config)
        net.train(false)                                                  // BatchNorm reads its running statistics
        return net
    }

    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let config = NFKMLXGTCRNConfiguration()
        let net = makeNet(config)
        if let weightsURL { try loadWeights(into: net, from: weightsURL) }
        return NFKMLXGTCRNBackend(net: net, config: config, identifier: modelName)
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

    /// Registers `gtcrn` with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a released GTCRN checkpoint. The module keys already mirror the reference (`erb`, `sfe`,
    /// `encoder.en_convs`, `dpgrnn{1,2}`, `decoder.de_convs`, `point_conv*`, `tra`, `intra_rnn`, …), so
    /// the loader only folds the GRUs and transposes the 4-D convolution weights.
    static func loadWeights(into net: NFKMLXGTCRN, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let folded = NFKMLXRecurrentFold.fold(checkpoint.arrays)
        let mapped: [(String, MLXArray)] = folded.map { key, value in
            guard value.ndim == 4, checkpoint.needsConvTranspose else { return (key, value) }
            // The decoder's convolutions are ConvTranspose2d (use_deconv); PyTorch stores their weight
            // as [in, out/groups, kH, kW], which becomes MLX's [out, kH, kW, in/groups] with a
            // group-aware reshape. Regular (encoder) convolutions are [out, in/groups, kH, kW] → (0,2,3,1).
            guard key.hasPrefix("decoder.de_convs.") else { return (key, value.transposed(0, 2, 3, 1)) }
            let groups = key.contains(".depth_conv.") ? value.dim(0)        // depthwise: groups = channels
                : (key == "decoder.de_convs.3.conv.weight" ? 2 : 1)         // the one grouped ConvBlock
            return (key, deconvWeight(value, groups: groups))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// A PyTorch grouped `ConvTranspose2d` weight `[in, out/groups, kH, kW]` → MLX `[out, kH, kW, in/groups]`.
    /// Splits the group axis out of `in` and folds it back into `out`. For groups = 1 this equals
    /// `transposed(1,2,3,0)`; for the depthwise case (`in = out = groups`) it equals `transposed(0,2,3,1)`.
    static func deconvWeight(_ weight: MLXArray, groups: Int) -> MLXArray {
        let inC = weight.dim(0), outPerGroup = weight.dim(1), kH = weight.dim(2), kW = weight.dim(3)
        let inPerGroup = inC / groups, out = outPerGroup * groups
        return weight.reshaped([groups, inPerGroup, outPerGroup, kH, kW])
            .transposed(0, 2, 3, 4, 1)
            .reshaped([out, kH, kW, inPerGroup])
    }
}
