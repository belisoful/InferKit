// The UnivNet vocoder of Resemble Enhance: a GAN generator that turns the 160-channel acoustic features
// into a waveform through location-variable convolutions (each output segment convolves with a kernel a
// KernelPredictor generates from the conditioning). Grounded on the released
// `resemble_enhance/enhancer/univnet/{univnet,lvcnet,amp,alias_free_torch}.py`.
//
// Tensors flow NLC `[B, T, C]`. The noise input `z` is non-deterministic in the reference; parity feeds a
// recorded `z`. The location-variable convolution is a per-segment im2col matmul (MLX has no unfold).

import Foundation
import InferKit
import MLX
import MLXNN

// MARK: - Anti-aliased SnakeBeta activation (the UnivNet variant)

private func nfkReBesselI0(_ x: Double) -> Double {
    var sum = 1.0, term = 1.0
    let y = (x / 2) * (x / 2)
    var k = 1.0
    while true {
        term *= y / (k * k)
        sum += term
        if term < 1e-12 * sum { break }
        k += 1
    }
    return sum
}

/// `kaiser_sinc_filter1d(cutoff, half_width, kernel_size)`, an even-length windowed sinc low-pass
/// normalized to unit sum. Returned as `[kernelSize]` Float.
private func nfkReKaiserSinc(cutoff: Double, halfWidth: Double, kernelSize: Int) -> [Float] {
    let halfSize = kernelSize / 2
    let deltaF = 4 * halfWidth
    let a = 2.285 * Double(halfSize - 1) * .pi * deltaF + 7.95
    let beta: Double
    if a > 50 { beta = 0.1102 * (a - 8.7) }
    else if a >= 21 { beta = 0.5842 * pow(a - 21, 0.4) + 0.07886 * (a - 21) }
    else { beta = 0 }
    let i0Beta = nfkReBesselI0(beta)
    let denom = Double(kernelSize - 1)
    var window = [Double](repeating: 0, count: kernelSize)
    for n in 0 ..< kernelSize {
        let r = 2 * Double(n) / denom - 1
        window[n] = nfkReBesselI0(beta * (1 - r * r).squareRoot()) / i0Beta
    }
    var filter = [Double](repeating: 0, count: kernelSize)
    var total = 0.0
    for n in 0 ..< kernelSize {
        let time = Double(n - halfSize) + 0.5
        let arg = 2 * cutoff * time
        let sinc = arg == 0 ? 1.0 : sin(.pi * arg) / (.pi * arg)
        filter[n] = 2 * cutoff * window[n] * sinc
        total += filter[n]
    }
    return filter.map { Float($0 / total) }
}

/// Replicate-pads the time axis (axis 1) of `[B, T, C]` by repeating the edge frames.
private func nfkRePad(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    var parts = [MLXArray]()
    if left > 0 { parts.append(broadcast(x[0..., 0 ..< 1, 0...], to: [x.dim(0), left, x.dim(2)])) }
    parts.append(x)
    if right > 0 { parts.append(broadcast(x[0..., (x.dim(1) - 1) ..< x.dim(1), 0...], to: [x.dim(0), right, x.dim(2)])) }
    return parts.count == 1 ? x : concatenated(parts, axis: 1)
}

/// `SnakeBeta` (log-scale, UnivNet): `x + (1/β)·sin²(x·α)`, `α = clamp(exp(log_alpha), 1e-2, 50)`,
/// `β = clamp(exp(log_beta), 1e-2, 50)`.
final class NFKResembleSnakeBeta: Module {
    @ParameterInfo(key: "log_alpha") var logAlpha: MLXArray
    @ParameterInfo(key: "log_beta") var logBeta: MLXArray

    init(channels: Int) {
        _logAlpha.wrappedValue = MLXArray.zeros([channels])
        _logBeta.wrappedValue = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = clip(exp(logAlpha), min: 1e-2, max: 50)
        let b = clip(exp(logBeta), min: 1e-2, max: 50)
        let s = sin(x * a)
        return x + (1 / b) * (s * s)
    }
}

/// `UpActDown`: upsample ×2 (kaiser-sinc), the SnakeBeta activation, downsample ×2. Both resamplings are
/// fixed depthwise FIRs held off `parameters()`.
final class NFKResembleUpActDown: Module {
    @ModuleInfo(key: "act") var act: NFKResembleSnakeBeta
    let kernelSize = 12, ratio = 2
    let filter: [Float]
    let upPad: Int, upPadLeft: Int, upPadRight: Int
    let downPadLeft: Int, downPadRight: Int

    init(channels: Int) {
        _act.wrappedValue = NFKResembleSnakeBeta(channels: channels)
        filter = nfkReKaiserSinc(cutoff: 0.5 / 2, halfWidth: 0.6 / 2, kernelSize: kernelSize)
        upPad = kernelSize / ratio - 1
        upPadLeft = upPad * ratio + (kernelSize - ratio) / 2
        upPadRight = upPad * ratio + (kernelSize - ratio + 1) / 2
        downPadLeft = kernelSize / 2 - 1
        downPadRight = kernelSize / 2
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = x.dim(2)
        var h = nfkRePad(x, left: upPad, right: upPad)
        let up = broadcast(MLXArray(filter).reshaped([1, kernelSize, 1]), to: [c, kernelSize, 1])
        h = Float(ratio) * convTransposed1d(h, up, stride: ratio, padding: 0, groups: c)
        h = h[0..., upPadLeft ..< (h.dim(1) - upPadRight), 0...]
        h = act(h)
        h = nfkRePad(h, left: downPadLeft, right: downPadRight)
        let down = broadcast(MLXArray(filter).reshaped([1, kernelSize, 1]), to: [c, kernelSize, 1])
        return conv1d(h, down, stride: ratio, padding: 0, groups: c)
    }
}

// MARK: - AMP block

/// One AMP layer (`conv1` → anti-aliased SnakeBeta → `conv2`).
final class NFKResembleAMPLayer: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "act") var act: NFKResembleUpActDown
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(channels: Int, kernel: Int, dilation: Int) {
        _conv1.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel,
                                     padding: (kernel - 1) * dilation / 2, dilation: dilation)
        _act.wrappedValue = NFKResembleUpActDown(channels: channels)
        _conv2.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel,
                                     padding: (kernel - 1) / 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv2(act(conv1(x))) }
}

// MARK: - Kernel predictor + location-variable convolution

/// One residual conv pair inside the KernelPredictor (`conv1` → LeakyReLU → `conv2` → LeakyReLU), added
/// back to the input.
final class NFKResembleResidualConv: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(hidden: Int, kernel: Int) {
        _conv1.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: hidden, kernelSize: kernel, padding: (kernel - 1) / 2)
        _conv2.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: hidden, kernelSize: kernel, padding: (kernel - 1) / 2)
    }

    func callAsFunction(_ c: MLXArray) -> MLXArray {
        var h = leakyRelu(conv1(c), negativeSlope: 0.2)
        h = leakyRelu(conv2(h), negativeSlope: 0.2)
        return c + h
    }
}

/// `KernelPredictor`: generates the per-location convolution kernels and biases from the conditioning.
final class NFKResembleKernelPredictor: Module {
    @ModuleInfo(key: "input_conv") var inputConv: Conv1d
    @ModuleInfo(key: "residual_convs") var residualConvs: [NFKResembleResidualConv]
    @ModuleInfo(key: "kernel_conv") var kernelConv: Conv1d
    @ModuleInfo(key: "bias_conv") var biasConv: Conv1d
    let convInChannels: Int, convOutChannels: Int, convKernelSize: Int, convLayers: Int

    init(condChannels: Int, convInChannels: Int, convOutChannels: Int, convLayers: Int,
         convKernelSize: Int = 3, hidden: Int = 64, kpKernel: Int = 3) {
        self.convInChannels = convInChannels
        self.convOutChannels = convOutChannels
        self.convKernelSize = convKernelSize
        self.convLayers = convLayers
        _inputConv.wrappedValue = Conv1d(inputChannels: condChannels, outputChannels: hidden, kernelSize: 5, padding: 2)
        _residualConvs.wrappedValue = (0 ..< 3).map { _ in NFKResembleResidualConv(hidden: hidden, kernel: kpKernel) }
        let kernelChannels = convInChannels * convOutChannels * convKernelSize * convLayers
        let biasChannels = convOutChannels * convLayers
        _kernelConv.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: kernelChannels, kernelSize: kpKernel, padding: (kpKernel - 1) / 2)
        _biasConv.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: biasChannels, kernelSize: kpKernel, padding: (kpKernel - 1) / 2)
    }

    /// `c [1, L, condChannels]` → kernels `[1, L, convLayers, in, out, ksize]`, bias `[1, L, convLayers, out]`.
    func callAsFunction(_ c0: MLXArray) -> (MLXArray, MLXArray) {
        var c = leakyRelu(inputConv(c0), negativeSlope: 0.2)
        for rc in residualConvs { c = rc(c) }
        let L = c.dim(1)
        let k = kernelConv(c).reshaped([1, L, convLayers, convInChannels, convOutChannels, convKernelSize])
        let b = biasConv(c).reshaped([1, L, convLayers, convOutChannels])
        return (k, b)
    }
}

/// One LVCBlock: an upsampling transposed conv, an AMP block, then four location-variable convolution
/// stages gated (GAU) with the predicted kernels.
final class NFKResembleLVCBlock: Module {
    @ModuleInfo(key: "kernel_predictor") var kernelPredictor: NFKResembleKernelPredictor
    @ModuleInfo(key: "convt_pre") var convtPre: ConvTransposed1d
    @ModuleInfo(key: "amp_block") var ampBlock: [NFKResembleAMPLayer]
    @ModuleInfo(key: "conv_blocks") var convBlocks: [Conv1d]
    let inChannels: Int, condHopLength: Int, convKernelSize: Int
    let dilations: [Int]

    init(inChannels: Int, condChannels: Int, stride: Int, dilations: [Int] = [1, 3, 9, 27],
         convKernelSize: Int = 3, condHopLength: Int) {
        self.inChannels = inChannels
        self.condHopLength = condHopLength
        self.convKernelSize = convKernelSize
        self.dilations = dilations
        _kernelPredictor.wrappedValue = NFKResembleKernelPredictor(
            condChannels: condChannels, convInChannels: inChannels, convOutChannels: 2 * inChannels,
            convLayers: dilations.count, convKernelSize: convKernelSize)
        _convtPre.wrappedValue = ConvTransposed1d(inputChannels: inChannels, outputChannels: inChannels,
                                                  kernelSize: 2 * stride, stride: stride,
                                                  padding: stride / 2 + stride % 2, outputPadding: stride % 2)
        _ampBlock.wrappedValue = [1, 3, 5].map { NFKResembleAMPLayer(channels: inChannels, kernel: 3, dilation: $0) }
        _convBlocks.wrappedValue = dilations.map {
            Conv1d(inputChannels: inChannels, outputChannels: inChannels, kernelSize: convKernelSize,
                   padding: (convKernelSize - 1) * $0 / 2, dilation: $0)
        }
    }

    /// The location-variable convolution (dilation 1): a per-segment im2col matmul. `x [1, T, in]`,
    /// `kernel [1, L, in, out, ksize]`, `bias [1, L, out]`, `T == L·hop`.
    private func lvc(_ x: MLXArray, kernel: MLXArray, bias: MLXArray) -> MLXArray {
        let T = x.dim(1), inC = inChannels, outC = 2 * inChannels, ks = convKernelSize
        let L = kernel.dim(1), hop = condHopLength
        let padded = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((1, 1)), IntOrPair((0, 0))], mode: .constant)
        // im2col over time: windows[1, T, ks, in] = padded[1, t + k, in].
        var gather = [Int32]()
        for t in 0 ..< T { for k in 0 ..< ks { gather.append(Int32(t + k)) } }
        let windows = take(padded, MLXArray(gather), axis: 1).reshaped([1, T, ks, inC])
        // → [1, L, hop, in, ks] → flatten (in, ks).
        let winFlat = windows.reshaped([1, L, hop, ks, inC]).transposed(0, 1, 2, 4, 3).reshaped([1, L, hop, inC * ks])
        // kernel [1, L, in, out, ks] → [1, L, in, ks, out] → flatten (in, ks).
        let kerFlat = kernel.transposed(0, 1, 2, 4, 3).reshaped([1, L, inC * ks, outC])
        var o = matmul(winFlat, kerFlat)                     // [1, L, hop, out]
        o = o + bias.reshaped([1, L, 1, outC])
        return o.reshaped([1, T, outC])
    }

    func callAsFunction(_ x0: MLXArray, cond: MLXArray) -> MLXArray {
        var x = convtPre(leakyRelu(x0, negativeSlope: 0.2))  // Sequential(LeakyReLU, ConvTranspose)
        var amp = x                                          // AMPBlock: x + seq(x), residual over all 3 layers
        for layer in ampBlock { amp = layer(amp) }
        x = x + amp
        let (kernels, biases) = kernelPredictor(cond)
        for i in 0 ..< dilations.count {
            let out = leakyRelu(convBlocks[i](leakyRelu(x, negativeSlope: 0.2)), negativeSlope: 0.2)
            let kernel = kernels[0..., 0..., i, 0..., 0..., 0...].reshaped([1, kernels.dim(1), inChannels, 2 * inChannels, convKernelSize])
            let bias = biases[0..., 0..., i, 0...].reshaped([1, biases.dim(1), 2 * inChannels])
            let o = lvc(out, kernel: kernel, bias: bias)     // [1, T, 2·in]
            let gated = sigmoid(o[0..., 0..., 0 ..< inChannels]) * tanh(o[0..., 0..., inChannels...])
            x = x + gated
        }
        return x
    }
}

/// The UnivNet generator: `conv_pre` over the noise, four `LVCBlock`s conditioned on the features, then
/// `conv_post` (LeakyReLU → conv → Tanh).
public final class NFKMLXResembleUnivNet: Module {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [NFKResembleLVCBlock]
    @ModuleInfo(key: "conv_post") var convPost: Conv1d
    let strides = [7, 5, 4, 3]
    let dNoise = 128
    let scaleFactor: Int

    init(_ config: NFKMLXResembleConfiguration) {
        let nc = config.univnetNC
        let dInput = config.vocoderInputDim
        scaleFactor = config.hopSize
        _convPre.wrappedValue = Conv1d(inputChannels: dNoise, outputChannels: nc, kernelSize: 7)   // reflect pad done manually
        var hop = 1
        var blockList = [NFKResembleLVCBlock]()
        for stride in strides {
            hop *= stride
            blockList.append(NFKResembleLVCBlock(inChannels: nc, condChannels: dInput, stride: stride, condHopLength: hop))
        }
        _blocks.wrappedValue = blockList
        _convPost.wrappedValue = Conv1d(inputChannels: nc, outputChannels: 1, kernelSize: 7)
    }

    private func reflectPad(_ x: MLXArray, _ pad: Int) -> MLXArray {
        let l = x.dim(1)
        var idx = [Int32]()
        for i in stride(from: pad, through: 1, by: -1) { idx.append(Int32(i)) }
        idx.append(contentsOf: (0 ..< l).map { Int32($0) })
        for i in stride(from: l - 2, through: l - 1 - pad, by: -1) { idx.append(Int32(i)) }
        return take(x, MLXArray(idx), axis: 1)
    }

    /// `features [1, t, vocoderInputDim]` + noise `z [1, t + npad, dNoise]` → waveform `[1, samples, 1]`.
    /// The reference draws `z` internally; parity supplies it. `npad` defaults to 10 (the reference).
    public func callAsFunction(_ features: MLXArray, noise z: MLXArray, npad: Int = 10) -> MLXArray {
        let x = MLX.padded(features, widths: [IntOrPair((0, 0)), IntOrPair((0, npad)), IntOrPair((0, 0))], mode: .constant)
        var zc = convPre(reflectPad(z, 3))
        for block in blocks { zc = block(zc, cond: x) }
        zc = leakyRelu(zc, negativeSlope: 0.2)
        zc = convPost(reflectPad(zc, 3))
        zc = tanh(zc)
        let keep = zc.dim(1) - scaleFactor * npad
        return zc[0..., 0 ..< keep, 0...]
    }
}
