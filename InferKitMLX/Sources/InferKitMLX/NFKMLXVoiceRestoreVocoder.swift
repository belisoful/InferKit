// BigVGAN v2 (`nvidia/bigvgan_v2_24khz_100band_256x`, MIT), the vocoder VoiceRestore decodes its mel
// through. A HiFi-GAN-style generator with two BigVGAN additions: SnakeBeta periodic activations and an
// anti-aliased `Activation1d` (a fixed kaiser-sinc up/down FIR around each activation). Grounded on the
// released source read 2026-09-06 (`bigvgan.py`, `activations.py`, `alias_free_activation/torch/*`).
//
// Tensors flow NLC `[B, T, C]` (the layout MLX's Conv1d uses); the mel arrives `[B, T, 100]` and the
// waveform leaves `[B, T, 1]`.

import Foundation
import InferKit
import MLX
import MLXNN

/// The BigVGAN generator configuration (`bigvgan_v2_24khz_100band_256x.json`).
public struct NFKMLXBigVGANConfiguration: Sendable {
    public var numMels: Int
    public var upsampleRates: [Int]
    public var upsampleKernels: [Int]
    public var initialChannel: Int
    public var resblockKernels: [Int]
    public var resblockDilations: [[Int]]

    public init(numMels: Int = 100, upsampleRates: [Int] = [4, 4, 2, 2, 2, 2],
                upsampleKernels: [Int] = [8, 8, 4, 4, 4, 4], initialChannel: Int = 1536,
                resblockKernels: [Int] = [3, 7, 11],
                resblockDilations: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]) {
        self.numMels = numMels
        self.upsampleRates = upsampleRates
        self.upsampleKernels = upsampleKernels
        self.initialChannel = initialChannel
        self.resblockKernels = resblockKernels
        self.resblockDilations = resblockDilations
    }
}

// MARK: - Kaiser-sinc FIR (the anti-aliased activation's up/down filter)

private func nfkBesselI0(_ x: Double) -> Double {
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

/// `kaiser_sinc_filter1d(cutoff, half_width, kernel_size)` (junjun3518/alias-free-torch): a windowed
/// sinc low-pass, normalized to unit sum. Returned as `[kernelSize]` Float.
private func nfkKaiserSincFilter(cutoff: Double, halfWidth: Double, kernelSize: Int) -> [Float] {
    let halfSize = kernelSize / 2
    let deltaF = 4 * halfWidth
    let a = 2.285 * Double(halfSize - 1) * .pi * deltaF + 7.95
    let beta: Double
    if a > 50 { beta = 0.1102 * (a - 8.7) }
    else if a >= 21 { beta = 0.5842 * pow(a - 21, 0.4) + 0.07886 * (a - 21) }
    else { beta = 0 }
    let i0Beta = nfkBesselI0(beta)
    // torch.kaiser_window(kernelSize, periodic: false, beta): symmetric over kernelSize points.
    let denom = Double(kernelSize - 1)
    var window = [Double](repeating: 0, count: kernelSize)
    for n in 0 ..< kernelSize {
        let r = 2 * Double(n) / denom - 1
        window[n] = nfkBesselI0(beta * (1 - r * r).squareRoot()) / i0Beta
    }
    // even kernel: time = arange(-halfSize, halfSize) + 0.5
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

/// The anti-aliased `Activation1d`: upsample ×2 (kaiser-sinc), the activation, downsample ×2. Both
/// resamplings are fixed depthwise FIRs (the filter is a constant, held off `parameters()`).
final class NFKVRActivation1d: Module {
    @ModuleInfo(key: "act") var act: NFKVRSnakeBeta
    let ratio = 2
    let kernelSize = 12
    let upFilter: [Float]                                            // [k]
    let downFilter: [Float]
    // UpSample1d geometry.
    let upPad: Int, upPadLeft: Int, upPadRight: Int
    // DownSample1d geometry.
    let downPadLeft: Int, downPadRight: Int

    init(channels: Int) {
        _act.wrappedValue = NFKVRSnakeBeta(channels: channels)
        upFilter = nfkKaiserSincFilter(cutoff: 0.5 / 2, halfWidth: 0.6 / 2, kernelSize: kernelSize)
        downFilter = nfkKaiserSincFilter(cutoff: 0.5 / 2, halfWidth: 0.6 / 2, kernelSize: kernelSize)
        upPad = kernelSize / ratio - 1
        upPadLeft = upPad * ratio + (kernelSize - ratio) / 2
        upPadRight = upPad * ratio + (kernelSize - ratio + 1) / 2
        downPadLeft = kernelSize / 2 - 1                              // even kernel
        downPadRight = kernelSize / 2
    }

    /// `x [B, T, C]` → `[B, T, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = x.dim(2)
        // UpSample1d: replicate-pad by `upPad`, transposed conv (stride 2, groups C) × ratio, crop.
        var h = nfkReplicatePad(x, left: upPad, right: upPad)
        let upWeight = broadcast(MLXArray(upFilter).reshaped([1, kernelSize, 1]), to: [c, kernelSize, 1])
        h = Float(ratio) * convTransposed1d(h, upWeight, stride: ratio, padding: 0, groups: c)
        h = h[0..., upPadLeft ..< (h.dim(1) - upPadRight), 0...]
        h = act(h)
        // DownSample1d: replicate-pad, conv (stride ratio, groups C).
        h = nfkReplicatePad(h, left: downPadLeft, right: downPadRight)
        let downWeight = broadcast(MLXArray(downFilter).reshaped([1, kernelSize, 1]), to: [c, kernelSize, 1])
        return conv1d(h, downWeight, stride: ratio, padding: 0, groups: c)
    }
}

/// Replicate-pads the time axis (axis 1) of `[B, T, C]` by repeating the edge frames.
private func nfkReplicatePad(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    var parts = [MLXArray]()
    if left > 0 { parts.append(broadcast(x[0..., 0 ..< 1, 0...], to: [x.dim(0), left, x.dim(2)])) }
    parts.append(x)
    if right > 0 { parts.append(broadcast(x[0..., (x.dim(1) - 1) ..< x.dim(1), 0...], to: [x.dim(0), right, x.dim(2)])) }
    return parts.count == 1 ? x : concatenated(parts, axis: 1)
}

/// `SnakeBeta` (logscale): `x + (1/(exp(β)+1e-9))·sin²(exp(α)·x)`, per-channel `α`, `β`.
final class NFKVRSnakeBeta: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray                 // [C]
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.zeros([channels])
        _beta.wrappedValue = MLXArray.zeros([channels])
    }

    /// `x [B, T, C]` → `[B, T, C]` (channels last).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = exp(alpha), b = exp(beta)
        let s = sin(x * a)
        return x + (1 / (b + 1e-9)) * (s * s)
    }
}

/// `AMPBlock1`: three dilated conv pairs (`convs1` dilated, `convs2` dilation 1), each preceded by an
/// anti-aliased SnakeBeta, added back to the input.
final class NFKVRAMPBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "activations") var activations: [NFKVRActivation1d]

    init(channels: Int, kernel: Int, dilations: [Int]) {
        _convs1.wrappedValue = dilations.map {
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel,
                   padding: (kernel - 1) * $0 / 2, dilation: $0)
        }
        _convs2.wrappedValue = dilations.map { _ in
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel - 1) / 2)
        }
        _activations.wrappedValue = (0 ..< dilations.count * 2).map { _ in NFKVRActivation1d(channels: channels) }
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = x0
        for i in 0 ..< convs1.count {
            var xt = activations[2 * i](x)
            xt = convs1[i](xt)
            xt = activations[2 * i + 1](xt)
            xt = convs2[i](xt)
            x = xt + x
        }
        return x
    }
}

/// The BigVGAN generator: `conv_pre` → 6 upsample stages (a `ConvTranspose1d` then the summed AMP blocks)
/// → an anti-aliased SnakeBeta → `conv_post` → clamp to `[-1, 1]`.
public final class NFKMLXBigVGAN: Module {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "resblocks") var resblocks: [NFKVRAMPBlock]
    @ModuleInfo(key: "activation_post") var activationPost: NFKVRActivation1d
    @ModuleInfo(key: "conv_post") var convPost: Conv1d
    let config: NFKMLXBigVGANConfiguration
    let numKernels: Int

    public init(_ config: NFKMLXBigVGANConfiguration) {
        self.config = config
        numKernels = config.resblockKernels.count
        _convPre.wrappedValue = Conv1d(inputChannels: config.numMels, outputChannels: config.initialChannel, kernelSize: 7, padding: 3)
        var upList = [ConvTransposed1d]()
        var blockList = [NFKVRAMPBlock]()
        for (i, (u, k)) in zip(config.upsampleRates, config.upsampleKernels).enumerated() {
            let inCh = config.initialChannel / (1 << i)
            let outCh = config.initialChannel / (1 << (i + 1))
            upList.append(ConvTransposed1d(inputChannels: inCh, outputChannels: outCh, kernelSize: k,
                                           stride: u, padding: (k - u) / 2))
            for (rk, rd) in zip(config.resblockKernels, config.resblockDilations) {
                blockList.append(NFKVRAMPBlock(channels: outCh, kernel: rk, dilations: rd))
            }
        }
        _ups.wrappedValue = upList
        _resblocks.wrappedValue = blockList
        let finalChannels = config.initialChannel / (1 << config.upsampleRates.count)
        _activationPost.wrappedValue = NFKVRActivation1d(channels: finalChannels)
        _convPost.wrappedValue = Conv1d(inputChannels: finalChannels, outputChannels: 1, kernelSize: 7, padding: 3, bias: false)
    }

    /// `mel [B, T, numMels]` → waveform `[B, T·∏rates, 1]`.
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = convPre(mel)
        for i in 0 ..< config.upsampleRates.count {
            x = ups[i](x)
            var xs = resblocks[i * numKernels](x)
            for j in 1 ..< numKernels { xs = xs + resblocks[i * numKernels + j](x) }
            x = xs / Float(numKernels)
        }
        x = activationPost(x)
        x = convPost(x)
        return clip(x, min: -1, max: 1)
    }

    /// Loads the released `bigvgan_generator.pt` (`nvidia/bigvgan_v2_24khz_100band_256x`): fuse the
    /// weight-norm pairs, drop the fixed FIR buffers (this port recomputes them), collapse the single-
    /// element `ups.N.0` ModuleList, and transpose the convolution weights into MLX's NLC layout.
    public func loadWeights(from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let fused = NFKMLXMusic3.fusedWeightNorm(checkpoint.arrays)
        let mapped: [(String, MLXArray)] = fused.compactMap { key, value in
            if key.hasSuffix(".filter") { return nil }                 // the kaiser FIRs are recomputed
            let name = key.replacingOccurrences(of: #"^ups\.(\d+)\.0\."#, with: "ups.$1.", options: .regularExpression)
            let tensor: MLXArray
            if value.ndim == 3, checkpoint.needsConvTranspose {
                // ConvTranspose1d (ups): PyTorch [in, out, k] → MLX [out, k, in]. Conv1d: [out, in, k] → [out, k, in].
                tensor = name.hasPrefix("ups.") ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            } else {
                tensor = value
            }
            return (name, tensor)
        }
        try NFKMLXWeights.apply(mapped, to: self)
    }
}

/// Registration and weight loading for the BigVGAN vocoder.
@objc(NFKMLXBigVGAN_Factory)
public final class NFKMLXBigVGANFactory: NSObject {
    static func makeNet(_ config: NFKMLXBigVGANConfiguration = .init()) -> NFKMLXBigVGAN {
        NFKMLXBigVGAN(config)
    }
}
