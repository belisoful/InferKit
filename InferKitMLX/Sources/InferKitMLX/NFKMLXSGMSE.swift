// SGMSE+ (sp-uhh/sgmse, MIT): score-based generative speech enhancement and DEREVERBERATION — the
// generative anchor of the restoration vein and the first target with a dedicated dereverb checkpoint
// (WSJ0-REVERB 16 kHz, EARS-Reverb 48 kHz). A forward OUVE variance-exploding SDE walks a clean complex
// spectrogram toward the noisy/reverberant observation; inference runs the REVERSE SDE (a
// predictor-corrector sampler) from `x_T = y + noise` down to `x_0`, scored by an NCSN++ network, then
// inverts the STFT.
//
// SCAFFOLD STATUS (grounded against sp-uhh/sgmse): this file carries the configuration and the OUVE
// VE-SDE scheduler (the diffusion process, fully grounded from `sdes.py`). The remaining pieces are the
// NCSN++ score network (a score-SDE U-Net with FIR `[1,3,3,1]` up/down-sampling, self-attention, and a
// Gaussian-Fourier time embedding), the predictor-corrector sampler, the sqrt-Hann + amplitude-compressed
// STFT front end, and the backend. Inference reads the EMA weights, not the raw parameters.

import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

/// The SGMSE+ configuration. The SDE and STFT hyperparameters are read from the released checkpoint's
/// saved hparams; the defaults here are the 16 kHz dereverb settings.
public struct NFKMLXSGMSEConfiguration: Sendable {
    /// The Ornstein-Uhlenbeck stiffness (`theta`, 1.5).
    public var theta: Float
    /// The smallest and largest process noise scales.
    public var sigmaMin: Float
    public var sigmaMax: Float
    /// The number of reverse steps (`N`, 30) and the minimum process time (`t_eps`, 0.03).
    public var reverseSteps: Int
    public var tEps: Float
    /// The predictor-corrector step-size ratio (`snr`) for the annealed-Langevin corrector.
    public var correctorSNR: Float

    /// STFT: a sqrt-Hann window, center-padded. The amplitude is compressed by `X = |X|^exponent · factor`
    /// before the network reads it, and inverted after.
    public var fftSize: Int
    public var hopSize: Int
    public var specFactor: Float
    public var specAbsExponent: Float
    public var sampleRate: Int

    /// NCSN++ score-network geometry (`backbones/ncsnpp.py`). The base channel width, the per-resolution
    /// channel multipliers (its length is the number of resolutions), the residual blocks per resolution,
    /// the freq-axis resolutions that carry self-attention, the Gaussian-Fourier embedding scale, and the
    /// resolution the channel multipliers are indexed against.
    public var baseChannels: Int
    public var channelMultipliers: [Int]
    public var residualBlocks: Int
    public var attentionResolutions: [Int]
    public var fourierScale: Float
    public var imageSize: Int
    /// Whether the network uses the progressive input/output pyramid (`progressive='output_skip'`,
    /// `progressive_input='input_skip'` — the classic `ncsnpp` backbone). `false` is the `ncsnpp_48k`
    /// backbone (`progressive='none'`): no Combine on the down path, no per-level pyramid on the up path,
    /// a final GroupNorm + conv instead, and the output projection is applied BEFORE the `scale_by_sigma`
    /// division rather than after.
    public var progressiveOutputSkip: Bool
    /// The STFT window is `hann^windowPower`: `0.5` is the sqrt-Hann the classic checkpoints use, `1.0`
    /// the plain Hann the `ncsnpp_48k` checkpoints use.
    public var windowPower: Float
    /// The number of REAL input channels the network reads (`total_channels`): SGMSE+ packs `x` and the
    /// observation `y` into 4 (real/imag of each); StoRM's score net conditions on `[x, y, y_denoised]`
    /// for 6, and its discriminative predictor reads just `y` for 2. `output_layer` maps these to 2.
    public var inputChannels: Int
    /// Whether the network carries the Gaussian-Fourier time embedding (the score net) or omits it (the
    /// StoRM discriminative predictor, `discriminative=True` → `conditional=False`).
    public var conditional: Bool
    /// Whether the output is divided by the noise level before the projection (the score net;
    /// `False` for the discriminative predictor).
    public var scaleBySigma: Bool

    public init(theta: Float = 1.5, sigmaMin: Float = 0.05, sigmaMax: Float = 0.5, reverseSteps: Int = 30,
                tEps: Float = 0.03, correctorSNR: Float = 0.5, fftSize: Int = 510, hopSize: Int = 128,
                specFactor: Float = 0.15, specAbsExponent: Float = 0.5, sampleRate: Int = 16000,
                baseChannels: Int = 128, channelMultipliers: [Int] = [1, 1, 2, 2, 2, 2, 2],
                residualBlocks: Int = 2, attentionResolutions: [Int] = [16], fourierScale: Float = 16,
                imageSize: Int = 256, progressiveOutputSkip: Bool = true, windowPower: Float = 0.5,
                inputChannels: Int = 4, conditional: Bool = true, scaleBySigma: Bool = true) {
        self.theta = theta
        self.sigmaMin = sigmaMin
        self.sigmaMax = sigmaMax
        self.reverseSteps = reverseSteps
        self.tEps = tEps
        self.correctorSNR = correctorSNR
        self.fftSize = fftSize
        self.hopSize = hopSize
        self.specFactor = specFactor
        self.specAbsExponent = specAbsExponent
        self.sampleRate = sampleRate
        self.baseChannels = baseChannels
        self.channelMultipliers = channelMultipliers
        self.residualBlocks = residualBlocks
        self.attentionResolutions = attentionResolutions
        self.fourierScale = fourierScale
        self.imageSize = imageSize
        self.progressiveOutputSkip = progressiveOutputSkip
        self.windowPower = windowPower
        self.inputChannels = inputChannels
        self.conditional = conditional
        self.scaleBySigma = scaleBySigma
    }

    /// `log(sigma_max / sigma_min)`.
    public var logSigma: Float { logf(sigmaMax / sigmaMin) }
}

/// The OUVE variance-exploding SDE (`sgmse/sdes.py`), the diffusion process SGMSE+ scores. A value type
/// with no parameters, so it is verified against the reference with no weights.
///
/// The forward marginal of a clean spectrogram `x0` toward the observation `y` at time `t` is Gaussian
/// with `mean = e^{-θt}·x0 + (1 - e^{-θt})·y` and standard deviation `std(t)` below. The reverse sampler
/// starts from the prior `x_T = y + noise·std(1)` and walks down to `t_eps`.
public struct NFKMLXOUVEScheduler: Sendable {
    public let config: NFKMLXSGMSEConfiguration

    public init(_ config: NFKMLXSGMSEConfiguration) { self.config = config }

    /// The forward marginal mean `e^{-θt}·x0 + (1 - e^{-θt})·y`.
    public func mean(x0: MLXArray, y: MLXArray, t: Float) -> MLXArray {
        let decay = expf(-config.theta * t)
        return decay * x0 + (1 - decay) * y
    }

    /// The forward marginal standard deviation, the closed-form solution of the variance ODE:
    /// `sqrt( σmin² · e^{-2θt} · (e^{2(θ+logσ)t} - 1) · logσ / (θ + logσ) )`.
    public func std(_ t: Float) -> Float {
        let theta = config.theta, logSig = config.logSigma, sigmaMin = config.sigmaMin
        let numerator = sigmaMin * sigmaMin * expf(-2 * theta * t) * (expf(2 * (theta + logSig) * t) - 1) * logSig
        return sqrtf(numerator / (theta + logSig))
    }

    /// The diffusion coefficient `g(t) = σmin·(σmax/σmin)^t · sqrt(2·logσ)`.
    public func diffusion(_ t: Float) -> Float {
        config.sigmaMin * powf(config.sigmaMax / config.sigmaMin, t) * sqrtf(2 * config.logSigma)
    }

    /// The prior sample the reverse process starts from: `x_T = y + noise · std(1)`. The noise is a unit
    /// Gaussian the caller draws (a deterministic SplitMix64 stream, as the shipped schedulers do).
    public func prior(y: MLXArray, noise: MLXArray) -> MLXArray {
        y + noise * std(1)
    }

    /// The reverse-time schedule `linspace(1, t_eps, reverseSteps)` the predictor-corrector sampler
    /// visits, plus the fixed step `dt = (t_eps - 1) / (reverseSteps - 1)`.
    public var timesteps: [Float] {
        let n = config.reverseSteps
        guard n > 1 else { return [1] }
        let step = (config.tEps - 1) / Float(n - 1)
        return (0 ..< n).map { 1 + step * Float($0) }
    }
}

// MARK: - NCSN++ score network

// The FIR resampler (`ncsnpp_utils/up_or_down_sampling.py` + `op/upfirdn2d_native.py`), the one new op
// the port adds. The BigGAN resnet blocks and the progressive pyramid resample through the non-conv FIR
// path (`upsample_2d` / `downsample_2d`), so a fused conv-transpose FIR is not needed. `upfirdn2d` inserts
// zeros to upsample, pads, convolves the separable `[1,3,3,1]` kernel depthwise (one shared filter over
// every channel), then strides to downsample — modeled on the DAC/SNAC depthwise resamplers.
enum NFKSGMSEFIR {
    /// The separable resample filter `[1, 3, 3, 1]`.
    static let taps: [Float] = [1, 3, 3, 1]

    /// The normalized 2-D kernel `outer(taps, taps) / sum`, scaled by `gain`. `_setup_kernel` normalizes
    /// so a constant input keeps its magnitude; the caller supplies `gain = factor²` for upsampling and
    /// `1` for downsampling.
    static func kernel2D(gain: Float) -> MLXArray {
        let n = taps.count
        var flat = [Float](repeating: 0, count: n * n)
        var sum: Float = 0
        for i in 0 ..< n {
            for j in 0 ..< n {
                let v = taps[i] * taps[j]
                flat[i * n + j] = v
                sum += v
            }
        }
        for k in 0 ..< flat.count { flat[k] = flat[k] / sum * gain }
        return MLXArray(flat, [n, n])
    }

    /// `upfirdn2d` over a channels-last `[B, H, W, C]` tensor: insert `up - 1` zeros after each pixel on
    /// both spatial axes, pad by (`padLo`, `padHi`) on each axis, convolve the flipped kernel depthwise
    /// (valid), then keep every `down`-th element. The pad is symmetric across the two axes, as the
    /// reference is (`pad_x* == pad_y*`).
    static func upfirdn2d(_ input: MLXArray, kernel: MLXArray, up: Int, down: Int,
                          padLo: Int, padHi: Int) -> MLXArray {
        let shape = input.shape
        let (b, h, w, c) = (shape[0], shape[1], shape[2], shape[3])
        var out = input
        if up > 1 {
            // Insert `up - 1` zeros after each pixel: expand each spatial axis to a size-`up` block whose
            // first slot holds the pixel and the rest are zero-padded.
            out = out.reshaped([b, h, 1, w, 1, c])
            out = MLX.padded(out, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((0, up - 1)),
                                           IntOrPair(0), IntOrPair((0, up - 1)), IntOrPair(0)])
            out = out.reshaped([b, h * up, w * up, c])
        }
        out = MLX.padded(out, widths: [IntOrPair(0), IntOrPair((padLo, padHi)),
                                       IntOrPair((padLo, padHi)), IntOrPair(0)])
        // Depthwise valid convolution: every channel shares the one FIR filter. The kernel is symmetric,
        // so the reference's flip is a no-op and is skipped.
        let kh = kernel.shape[0], kw = kernel.shape[1]
        let weight = broadcast(kernel.reshaped([1, kh, kw, 1]), to: [c, kh, kw, 1])
        out = conv2d(out, weight, stride: IntOrPair(1), padding: IntOrPair(0), groups: c)
        if down > 1 {
            let hOut = out.shape[1], wOut = out.shape[2]
            let hIdx = MLXArray(Swift.stride(from: 0, to: hOut, by: down).map { Int32($0) })
            let wIdx = MLXArray(Swift.stride(from: 0, to: wOut, by: down).map { Int32($0) })
            out = out.take(hIdx, axis: 1).take(wIdx, axis: 2)
        }
        return out
    }

    /// `upsample_2d`: factor-2 FIR upsampling, `gain = factor² = 4`, pad `(2, 1)` (`p = 4 - 2 = 2`).
    static func upsample(_ x: MLXArray) -> MLXArray {
        upfirdn2d(x, kernel: kernel2D(gain: 4), up: 2, down: 1, padLo: 2, padHi: 1)
    }

    /// `downsample_2d`: factor-2 FIR downsampling, `gain = 1`, pad `(1, 1)`.
    static func downsample(_ x: MLXArray) -> MLXArray {
        upfirdn2d(x, kernel: kernel2D(gain: 1), up: 1, down: 2, padLo: 1, padHi: 1)
    }
}

/// The GroupNorm group count `min(channels / 4, 32)` the score network uses everywhere.
private func nfkNCSNppGroups(_ channels: Int) -> Int { min(channels / 4, 32) }

/// `GaussianFourierProjection`: a fixed random projection `W` (a non-trained buffer loaded from the
/// checkpoint), then `cat(sin(x·W·2π), cos(x·W·2π))`. The input is `log(sigma)`, so the embedding width
/// is `2 · baseChannels`.
final class NFKSGMSEGaussianFourier: Module {
    @ParameterInfo(key: "W") var w: MLXArray

    init(size: Int, scale: Float) {
        self._w.wrappedValue = MLXRandom.normal([size]) * scale
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let proj = x.reshaped([-1, 1]) * w.reshaped([1, -1]) * (2 * Float.pi)
        return concatenated([sin(proj), cos(proj)], axis: -1)
    }
}

/// `NIN`: a 1×1 channel mix stored as `W [in, out]` and `b [out]`, applied over the channel axis of a
/// channels-last tensor (a plain matmul, the reference's permute-contract-permute in NHWC).
final class NFKSGMSENIN: Module {
    @ParameterInfo(key: "W") var w: MLXArray
    @ParameterInfo(key: "b") var b: MLXArray

    init(inputChannels: Int, outputChannels: Int) {
        self._w.wrappedValue = MLXRandom.normal([inputChannels, outputChannels])
        self._b.wrappedValue = MLXArray.zeros([outputChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x.matmul(w) + b }
}

/// `AttnBlockpp`: channel-wise self-attention over the (freq, time) plane at attention resolutions.
final class NFKSGMSEAttnBlock: Module {
    @ModuleInfo(key: "GroupNorm_0") var groupNorm: NFKSDGroupNorm
    @ModuleInfo(key: "NIN_0") var nin0: NFKSGMSENIN
    @ModuleInfo(key: "NIN_1") var nin1: NFKSGMSENIN
    @ModuleInfo(key: "NIN_2") var nin2: NFKSGMSENIN
    @ModuleInfo(key: "NIN_3") var nin3: NFKSGMSENIN

    let channels: Int

    init(channels: Int) {
        self.channels = channels
        self._groupNorm.wrappedValue = NFKSDGroupNorm(groups: nfkNCSNppGroups(channels),
                                                      channels: channels, eps: 1e-6)
        self._nin0.wrappedValue = NFKSGMSENIN(inputChannels: channels, outputChannels: channels)
        self._nin1.wrappedValue = NFKSGMSENIN(inputChannels: channels, outputChannels: channels)
        self._nin2.wrappedValue = NFKSGMSENIN(inputChannels: channels, outputChannels: channels)
        self._nin3.wrappedValue = NFKSGMSENIN(inputChannels: channels, outputChannels: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape                                        // [batch, freq, time, channel]
        let (bsz, hh, ww, cc) = (shape[0], shape[1], shape[2], shape[3])
        let h = groupNorm(x)
        let q = nin0(h).reshaped([bsz, hh * ww, cc])
        let k = nin1(h).reshaped([bsz, hh * ww, cc])
        let v = nin2(h).reshaped([bsz, hh * ww, cc])
        var w = q.matmul(k.transposed(0, 2, 1)) * powf(Float(cc), -0.5)
        w = softmax(w, axis: -1)
        let attended = w.matmul(v).reshaped([bsz, hh, ww, cc])
        let out = nin3(attended)
        return (x + out) / sqrtf(2)
    }
}

/// `ResnetBlockBigGANpp`: a normalized, FIR-resampling residual block with the timestep embedding added
/// between its two convolutions and a `1/√2` skip rescale. `up` / `down` FIR-resample both the trunk and
/// the shortcut.
final class NFKSGMSEResnetBlock: Module {
    @ModuleInfo(key: "GroupNorm_0") var groupNorm0: NFKSDGroupNorm
    @ModuleInfo(key: "Conv_0") var conv0: Conv2d
    @ModuleInfo(key: "Dense_0") var dense0: Linear
    @ModuleInfo(key: "GroupNorm_1") var groupNorm1: NFKSDGroupNorm
    @ModuleInfo(key: "Conv_1") var conv1: Conv2d
    @ModuleInfo(key: "Conv_2") var conv2: Conv2d?

    let inChannels: Int
    let outChannels: Int
    let up: Bool
    let down: Bool

    init(inChannels: Int, outChannels: Int, timeChannels: Int, up: Bool = false, down: Bool = false) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.up = up
        self.down = down
        self._groupNorm0.wrappedValue = NFKSDGroupNorm(groups: nfkNCSNppGroups(inChannels),
                                                       channels: inChannels, eps: 1e-6)
        self._conv0.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                          kernelSize: 3, padding: 1)
        self._dense0.wrappedValue = Linear(timeChannels, outChannels)
        self._groupNorm1.wrappedValue = NFKSDGroupNorm(groups: nfkNCSNppGroups(outChannels),
                                                       channels: outChannels, eps: 1e-6)
        self._conv1.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels,
                                          kernelSize: 3, padding: 1)
        if inChannels != outChannels || up || down {
            self._conv2.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                              kernelSize: 1, padding: 0)
        } else {
            self._conv2.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, _ temb: MLXArray?) -> MLXArray {
        var h = silu(groupNorm0(x))
        var shortcut = x
        if up {
            h = NFKSGMSEFIR.upsample(h)
            shortcut = NFKSGMSEFIR.upsample(shortcut)
        } else if down {
            h = NFKSGMSEFIR.downsample(h)
            shortcut = NFKSGMSEFIR.downsample(shortcut)
        }
        h = conv0(h)
        // The discriminative predictor runs with no time embedding (`temb == nil`); the Dense weights
        // still load from the checkpoint, they are simply not applied.
        if let temb {
            let bias = dense0(silu(temb))
            h = h + bias.reshaped([bias.shape[0], 1, 1, outChannels])
        }
        h = silu(groupNorm1(h))
        h = conv1(h)
        if let conv2 { shortcut = conv2(shortcut) }
        return (shortcut + h) / sqrtf(2)
    }
}

/// `Combine` (input_skip, `sum`): the FIR-downsampled input pyramid mapped through a 1×1 conv and added
/// to the trunk.
final class NFKSGMSECombine: Module {
    @ModuleInfo(key: "Conv_0") var conv0: Conv2d

    init(inputChannels: Int, outputChannels: Int) {
        self._conv0.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: outputChannels,
                                          kernelSize: 1, padding: 0)
    }

    func callAsFunction(_ pyramid: MLXArray, _ h: MLXArray) -> MLXArray { conv0(pyramid) + h }
}

/// The NCSN++ score U-Net (`backbones/ncsnpp.py`), ported as the reference's flat `all_modules` list so
/// the checkpoint keys (`all_modules.N.*`) match with no remap, walked by an index counter that mirrors
/// the reference forward. `output_layer` is a separate 4→2 convolution. The progressive input/output
/// pyramids resample through the parameter-free FIR ops (`NFKSGMSEFIR`), which carry no weights.
///
/// Layout is channels-last `[batch, freq, time, channel]`, so the attention-resolution check is on the
/// freq axis (axis 1, the reference's `h.shape[-2]`).
public final class NFKMLXNCSNppNet: Module {
    @ModuleInfo(key: "all_modules") var allModules: [Module]
    @ModuleInfo(key: "output_layer") var outputLayer: Conv2d

    let config: NFKMLXSGMSEConfiguration
    let numResolutions: Int
    let allResolutions: [Int]

    public init(_ config: NFKMLXSGMSEConfiguration) {
        self.config = config
        let nf = config.baseChannels
        let chMult = config.channelMultipliers
        let numRes = config.residualBlocks
        let attn = config.attentionResolutions
        let tembDim = nf * 4
        self.numResolutions = chMult.count
        self.allResolutions = (0 ..< chMult.count).map { config.imageSize / (1 << $0) }
        let allRes = self.allResolutions

        let channels = config.inputChannels              // total_channels (spatial_channels = 1)
        self._outputLayer.wrappedValue = Conv2d(inputChannels: channels, outputChannels: 2,
                                                kernelSize: 1, padding: 0)

        func resnet(_ inCh: Int, _ outCh: Int, up: Bool = false, down: Bool = false) -> NFKSGMSEResnetBlock {
            NFKSGMSEResnetBlock(inChannels: inCh, outChannels: outCh, timeChannels: tembDim, up: up, down: down)
        }

        var modules: [Module] = []
        // Time embedding: the Gaussian-Fourier module is present whatever the mode (the discriminative
        // predictor still carries it, its output discarded); the two-layer MLP to 4·nf exists only when
        // the network is time-conditional (the score net).
        modules.append(NFKSGMSEGaussianFourier(size: nf, scale: config.fourierScale))
        if config.conditional {
            modules.append(Linear(2 * nf, tembDim))
            modules.append(Linear(tembDim, tembDim))
        }
        // Input convolution: total_channels → nf.
        modules.append(Conv2d(inputChannels: channels, outputChannels: nf, kernelSize: 3, padding: 1))

        var hsC = [nf]
        var inCh = nf
        for iLevel in 0 ..< numResolutions {
            for _ in 0 ..< numRes {
                let outCh = nf * chMult[iLevel]
                modules.append(resnet(inCh, outCh))
                inCh = outCh
                if attn.contains(allRes[iLevel]) { modules.append(NFKSGMSEAttnBlock(channels: inCh)) }
                hsC.append(inCh)
            }
            if iLevel != numResolutions - 1 {
                modules.append(resnet(inCh, inCh, down: true))
                // input_skip: a 1×1 combine of the downsampled input pyramid. Absent for progressive='none'.
                if config.progressiveOutputSkip {
                    modules.append(NFKSGMSECombine(inputChannels: channels, outputChannels: inCh))
                }
                hsC.append(inCh)
            }
        }

        inCh = hsC.last!
        modules.append(resnet(inCh, inCh))
        modules.append(NFKSGMSEAttnBlock(channels: inCh))          // the middle attention is unconditional
        modules.append(resnet(inCh, inCh))

        for iLevel in stride(from: numResolutions - 1, through: 0, by: -1) {
            for _ in 0 ..< (numRes + 1) {
                let outCh = nf * chMult[iLevel]
                modules.append(resnet(inCh + hsC.removeLast(), outCh))
                inCh = outCh
            }
            if attn.contains(allRes[iLevel]) { modules.append(NFKSGMSEAttnBlock(channels: inCh)) }
            // output_skip: a GroupNorm + conv3x3(inCh → channels) per level feeding the pyramid.
            if config.progressiveOutputSkip {
                modules.append(NFKSDGroupNorm(groups: nfkNCSNppGroups(inCh), channels: inCh, eps: 1e-6))
                modules.append(Conv2d(inputChannels: inCh, outputChannels: channels, kernelSize: 3, padding: 1))
            }
            if iLevel != 0 { modules.append(resnet(inCh, inCh, up: true)) }
        }

        // progressive='none': a final GroupNorm + conv3x3(inCh → channels) after the up path.
        if !config.progressiveOutputSkip {
            modules.append(NFKSDGroupNorm(groups: nfkNCSNppGroups(inCh), channels: inCh, eps: 1e-6))
            modules.append(Conv2d(inputChannels: inCh, outputChannels: channels, kernelSize: 3, padding: 1))
        }

        self._allModules.wrappedValue = modules
        super.init()
    }

    /// The score at noise levels `sigmas` for a packed 4-channel spectrogram `x` (`[batch, freq, time, 4]`
    /// = real/imag of the current estimate and of the observation). Returns the score's real/imag as
    /// `[batch, freq, time, 2]`.
    public func callAsFunction(_ x: MLXArray, sigmas: MLXArray) -> MLXArray {
        let modules = allModules
        var mIdx = 0
        let attn = config.attentionResolutions
        let progressive = config.progressiveOutputSkip

        // Time embedding. The Gaussian-Fourier module is always consumed; the two-layer MLP runs only
        // when the network is conditional (the discriminative predictor leaves `temb` nil).
        let fourier = modules[mIdx] as! NFKSGMSEGaussianFourier; mIdx += 1
        var temb: MLXArray? = nil
        if config.conditional {
            var e = fourier(log(sigmas))
            e = (modules[mIdx] as! Linear)(e); mIdx += 1
            e = (modules[mIdx] as! Linear)(silu(e)); mIdx += 1
            temb = e
        }

        // Down path.
        var inputPyramid = x
        var hs = [(modules[mIdx] as! Conv2d)(x)]; mIdx += 1
        for iLevel in 0 ..< numResolutions {
            for _ in 0 ..< config.residualBlocks {
                var h = (modules[mIdx] as! NFKSGMSEResnetBlock)(hs.last!, temb); mIdx += 1
                if attn.contains(h.shape[1]) {
                    h = (modules[mIdx] as! NFKSGMSEAttnBlock)(h); mIdx += 1
                }
                hs.append(h)
            }
            if iLevel != numResolutions - 1 {
                var h = (modules[mIdx] as! NFKSGMSEResnetBlock)(hs.last!, temb); mIdx += 1
                if progressive {
                    inputPyramid = NFKSGMSEFIR.downsample(inputPyramid)
                    h = (modules[mIdx] as! NFKSGMSECombine)(inputPyramid, h); mIdx += 1
                }
                hs.append(h)
            }
        }

        // Middle.
        var h = hs.last!
        h = (modules[mIdx] as! NFKSGMSEResnetBlock)(h, temb); mIdx += 1
        h = (modules[mIdx] as! NFKSGMSEAttnBlock)(h); mIdx += 1
        h = (modules[mIdx] as! NFKSGMSEResnetBlock)(h, temb); mIdx += 1

        // Up path (output_skip: the pyramid accumulates the score).
        var pyramid: MLXArray? = nil
        for iLevel in stride(from: numResolutions - 1, through: 0, by: -1) {
            for _ in 0 ..< (config.residualBlocks + 1) {
                let skip = hs.removeLast()
                h = (modules[mIdx] as! NFKSGMSEResnetBlock)(concatenated([h, skip], axis: -1), temb)
                mIdx += 1
            }
            if attn.contains(h.shape[1]) {
                h = (modules[mIdx] as! NFKSGMSEAttnBlock)(h); mIdx += 1
            }
            if progressive {
                let norm = modules[mIdx] as! NFKSDGroupNorm; mIdx += 1
                let toPyramid = modules[mIdx] as! Conv2d; mIdx += 1
                if iLevel == numResolutions - 1 {
                    pyramid = toPyramid(silu(norm(h)))
                } else {
                    pyramid = NFKSGMSEFIR.upsample(pyramid!) + toPyramid(silu(norm(h)))
                }
            }
            if iLevel != 0 {
                h = (modules[mIdx] as! NFKSGMSEResnetBlock)(h, temb); mIdx += 1
            }
        }

        let sigma = sigmas.reshaped([sigmas.shape[0], 1, 1, 1])
        let scale = config.scaleBySigma
        if progressive {
            // classic: the pyramid IS h; scaled by sigma (score net), then projected → 2.
            return outputLayer(scale ? pyramid! / sigma : pyramid!)
        }
        // progressive='none': a final GroupNorm + conv, projected → 2, THEN scaled by sigma.
        h = silu((modules[mIdx] as! NFKSDGroupNorm)(h)); mIdx += 1
        h = (modules[mIdx] as! Conv2d)(h); mIdx += 1
        let out = outputLayer(h)
        return scale ? out / sigma : out
    }
}

// MARK: - Amplitude compression + STFT front end

/// The amplitude compression the SGMSE+ front end applies to a complex spectrogram
/// (`data_module.spec_fwd` / `spec_back`, the `exponent` transform): `X → |X|^e · e^{i·angle} · factor`.
/// Both directions preserve the phase and rescale the magnitude, so they act on the real and imaginary
/// parts as one shared per-bin scale. The spectrogram is carried as a `(real, imaginary)` pair.
enum NFKSGMSESpec {
    private static let magnitudeFloor: Float = 1e-12

    /// `|X|^e · e^{i·angle} · factor`. As a per-element scale on `(re, im)`: `factor · |X|^{e-1}`.
    static func forward(re: MLXArray, im: MLXArray, factor: Float, exponent e: Float) -> (MLXArray, MLXArray) {
        let magnitude = maximum(sqrt(re * re + im * im), magnitudeFloor)
        let scale = factor * exp((e - 1) * log(magnitude))
        return (re * scale, im * scale)
    }

    /// The inverse: `X / factor`, then `|·|^{1/e} · e^{i·angle}`. As a per-element scale on `(re, im)`:
    /// `(|X| / factor)^{1/e - 1} / factor`.
    static func backward(re: MLXArray, im: MLXArray, factor: Float, exponent e: Float) -> (MLXArray, MLXArray) {
        let magnitude = maximum(sqrt(re * re + im * im), magnitudeFloor)
        let scale = exp((1 / e - 1) * log(magnitude / factor)) / factor
        return (re * scale, im * scale)
    }
}

extension NFKMLXSGMSEConfiguration {
    /// The front-end STFT: a sqrt-Hann window, center-padded, reproducing `torch.stft` at the SGMSE+
    /// settings (`window="sqrthann"`, `center=True`).
    var stft: NFKMLXComplexSTFT {
        let power = windowPower
        let window = MLXArray(nfkPeriodicHann(fftSize).map { powf($0, power) })
        return NFKMLXComplexSTFT(nFFT: fftSize, hop: hopSize, window: window)
    }
}

// MARK: - Predictor-corrector reverse-SDE sampler

/// A complex spectrogram carried as a `(real, imaginary)` pair of `[batch, freq, time]` arrays, the
/// state the reverse-SDE sampler walks.
struct NFKSGMSESpectrogram {
    var real: MLXArray
    var imaginary: MLXArray
}

/// The SGMSE+ predictor-corrector sampler (`sgmse/sampling`): a reverse-diffusion predictor over the
/// OUVE reverse SDE plus an annealed-Langevin corrector, run from the prior `x_T = y + noise·std(1)`
/// down to `t_eps`. The score is `-net(cat[x_t, y], t)` (the `ScoreModel` negates the network output).
/// The fresh Gaussian noise is the shipped deterministic SplitMix64 + Box–Muller stream (complex normal,
/// each component variance 1/2), so a run is repeatable without the MLX random state; a sampled clip is
/// not bitwise-comparable to the reference regardless, so this matches only the noise statistics.
struct NFKSGMSESampler {
    let net: NFKMLXNCSNppNet
    let scheduler: NFKMLXOUVEScheduler
    let seed: UInt64
    /// The SDE center the reverse process walks toward (the observation `y` for SGMSE+, the denoised
    /// estimate `y_denoised` for StoRM): the drift is `θ(observation − x)` and the prior is
    /// `observation + noise·std(1)`.
    let observation: NFKSGMSESpectrogram
    /// The extra channels the score network conditions on, concatenated after `x` (`[y]` for SGMSE+;
    /// `[y]`, `[y_denoised]`, or `[y, y_denoised]` for StoRM per its `condition`).
    let conditioning: [NFKSGMSESpectrogram]
    /// Whether to run the annealed-Langevin corrector before each predictor step (SGMSE+ does; StoRM's
    /// few-step default is `corrector='none'`).
    let useCorrector: Bool

    init(net: NFKMLXNCSNppNet, scheduler: NFKMLXOUVEScheduler, seed: UInt64,
         observation: NFKSGMSESpectrogram, conditioning: [NFKSGMSESpectrogram]? = nil,
         useCorrector: Bool = true) {
        self.net = net
        self.scheduler = scheduler
        self.seed = seed
        self.observation = observation
        self.conditioning = conditioning ?? [observation]
        self.useCorrector = useCorrector
    }

    var config: NFKMLXSGMSEConfiguration { scheduler.config }

    /// The score `-net(cat[x, *conditioning], t)`, split into a `(real, imaginary)` pair.
    private func score(_ x: NFKSGMSESpectrogram, t: Float) -> NFKSGMSESpectrogram {
        var parts = [x.real, x.imaginary]
        for c in conditioning { parts.append(c.real); parts.append(c.imaginary) }
        let out = net(stacked(parts, axis: -1), sigmas: MLXArray([t]))
        return NFKSGMSESpectrogram(real: -out[0..., 0..., 0..., 0], imaginary: -out[0..., 0..., 0..., 1])
    }

    /// Deterministic complex normal noise shaped like `template`, each component variance 1/2.
    private func noise(like template: MLXArray, tag: UInt64) -> NFKSGMSESpectrogram {
        let f = template.shape[1], t = template.shape[2]
        let re = NFKMLXDiffusionBackend.gaussianNoise(height: f, width: t, channels: 1, seed: seed &+ tag)
        let im = NFKMLXDiffusionBackend.gaussianNoise(height: f, width: t, channels: 1, seed: seed &+ tag &+ 0x1234_5678)
        let scale = sqrtf(0.5)
        return NFKSGMSESpectrogram(real: re.reshaped([1, f, t]) * scale, imaginary: im.reshaped([1, f, t]) * scale)
    }

    /// One annealed-Langevin corrector step (`AnnealedLangevinDynamics`, `n_steps = 1`).
    private func correct(_ x: NFKSGMSESpectrogram, t: Float, tag: UInt64)
        -> (NFKSGMSESpectrogram, NFKSGMSESpectrogram) {
        let std = scheduler.std(t)
        let grad = score(x, t: t)
        let n = noise(like: x.real, tag: tag)
        let stepSize = powf(config.correctorSNR * std, 2) * 2
        let meanRe = x.real + stepSize * grad.real
        let meanIm = x.imaginary + stepSize * grad.imaginary
        let sqrtStep = sqrtf(stepSize * 2)
        let mean = NFKSGMSESpectrogram(real: meanRe, imaginary: meanIm)
        let next = NFKSGMSESpectrogram(real: meanRe + n.real * sqrtStep, imaginary: meanIm + n.imaginary * sqrtStep)
        return (next, mean)
    }

    /// One reverse-diffusion predictor step (`ReverseDiffusionPredictor`). Returns the new state and its
    /// noise-free mean.
    private func predict(_ x: NFKSGMSESpectrogram, t: Float, stepSize: Float, tag: UInt64)
        -> (NFKSGMSESpectrogram, NFKSGMSESpectrogram) {
        // discretize: drift = θ(observation - x); diffusion = σ(t)·sqrt(2·logσ); f = drift·dt; G = diffusion·sqrt(dt).
        let theta = config.theta
        let bigG = scheduler.diffusion(t) * sqrtf(stepSize)
        let grad = score(x, t: t)
        let n = noise(like: x.real, tag: tag)
        // rev_f = f - G²·score; x_mean = x - rev_f = x - f + G²·score.
        func meanComponent(_ xc: MLXArray, _ oc: MLXArray, _ scoreC: MLXArray) -> MLXArray {
            let f = theta * (oc - xc) * stepSize
            return xc - (f - (bigG * bigG) * scoreC)
        }
        let meanRe = meanComponent(x.real, observation.real, grad.real)
        let meanIm = meanComponent(x.imaginary, observation.imaginary, grad.imaginary)
        let mean = NFKSGMSESpectrogram(real: meanRe, imaginary: meanIm)
        let next = NFKSGMSESpectrogram(real: meanRe + n.real * bigG, imaginary: meanIm + n.imaginary * bigG)
        return (next, mean)
    }

    /// Runs the full reverse loop from the prior and returns the denoised estimate `x_0` (the predictor's
    /// final mean, `denoise = True`).
    func sample() -> NFKSGMSESpectrogram {
        let prior = noise(like: observation.real, tag: 0)
        let std1 = scheduler.std(1)
        var xt = NFKSGMSESpectrogram(real: observation.real + prior.real * std1,
                                     imaginary: observation.imaginary + prior.imaginary * std1)
        var mean = xt

        let timesteps = scheduler.timesteps
        let n = timesteps.count
        for i in 0 ..< n {
            let t = timesteps[i]
            let stepSize = (i != n - 1) ? t - timesteps[i + 1] : timesteps[n - 1]
            let tag = UInt64(i) &* 0x100
            if useCorrector { (xt, mean) = correct(xt, t: t, tag: tag &+ 1) }
            (xt, mean) = predict(xt, t: t, stepSize: stepSize, tag: tag &+ 2)
            eval(xt.real, xt.imaginary)
        }
        return mean
    }
}

// MARK: - Backend + weight loading

private final class NFKSGMSEHolder: @unchecked Sendable {
    let net: NFKMLXNCSNppNet
    let config: NFKMLXSGMSEConfiguration
    init(_ net: NFKMLXNCSNppNet, _ config: NFKMLXSGMSEConfiguration) {
        self.net = net
        self.config = config
    }
}

/// SGMSE+ speech dereverberation / enhancement as an InferKit backend. Reads `NFKInputAudio` and returns
/// the enhanced clip as a single `NFKAudioAsset` under `NFKOutputAudio`. The reverse-SDE sampler is
/// multi-step and multi-second; run it off the render thread.
@objc(NFKMLXSGMSEBackend)
public final class NFKMLXSGMSEBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKSGMSEHolder
    private let identifier: String
    private let seed: UInt64
    private let outputDirectory: URL

    init(net: NFKMLXNCSNppNet, config: NFKMLXSGMSEConfiguration, identifier: String, seed: UInt64 = 0,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKSGMSEHolder(net, config)
        self.identifier = identifier
        self.seed = seed
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let input = samples.withUnsafeBufferPointer { MLXArray($0, [1, samples.count]) }
        let enhanced = NFKMLXSGMSE.enhance(input, net: holder.net, config: holder.config, seed: seed)
        eval(enhanced)
        let stream = enhanced.reshaped([enhanced.shape[enhanced.ndim - 1]]).asArray(Float.self)
        let length = stream.count

        let url = outputDirectory.appendingPathComponent("sgmse-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: stream, sampleRate: sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(length) / Double(sampleRate),
                                  sampleRate: Double(sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do {
                job.finish(with: try self.runInference(for: request))
            } catch {
                job.finish(withError: error as NSError)
            }
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

/// Registration, the enhance pipeline, and weight loading for SGMSE+.
@objc(NFKMLXSGMSE)
public final class NFKMLXSGMSE: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "sgmse"

    /// The complete enhance path (`model.enhance`): peak-normalize, STFT + amplitude-compress the input,
    /// pad the time axis to a multiple of 64, run the PC sampler, undo the compression, invert the STFT,
    /// crop to the input length, and restore the peak. `input` is `[1, samples]`.
    public static func enhance(_ input: MLXArray, net: NFKMLXNCSNppNet, config: NFKMLXSGMSEConfiguration,
                               seed: UInt64 = 0) -> MLXArray {
        let originalLength = input.shape[input.ndim - 1]
        let normFactor = maximum(input.abs().max(), MLXArray(Float(1e-8)))
        let normalized = input / normFactor

        let stft = config.stft
        let (re0, im0) = stft.transformComplex(normalized)                // [1, bins, frames]
        let (yRe0, yIm0) = NFKSGMSESpec.forward(re: re0, im: im0, factor: config.specFactor, exponent: config.specAbsExponent)

        let frames = yRe0.shape[2]
        let padTime = frames % 64 == 0 ? 0 : 64 - frames % 64
        let yRe = padTime > 0 ? MLX.padded(yRe0, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((0, padTime))]) : yRe0
        let yIm = padTime > 0 ? MLX.padded(yIm0, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((0, padTime))]) : yIm0

        let observation = NFKSGMSESpectrogram(real: yRe, imaginary: yIm)
        let sampler = NFKSGMSESampler(net: net, scheduler: NFKMLXOUVEScheduler(config), seed: seed,
                                      observation: observation)
        let result = sampler.sample()

        // Crop the padded frames back off, undo the compression, invert.
        let sRe = result.real[0..., 0..., 0 ..< frames]
        let sIm = result.imaginary[0..., 0..., 0 ..< frames]
        let (backRe, backIm) = NFKSGMSESpec.backward(re: sRe, im: sIm, factor: config.specFactor, exponent: config.specAbsExponent)
        let audio = stft.inverseComplex(real: backRe, imaginary: backIm)   // [1, samples]

        let produced = audio.shape[audio.ndim - 1]
        let cropped = produced >= originalLength ? audio[0..., 0 ..< originalLength] : audio
        return cropped * normFactor
    }

    /// Builds an SGMSE+ backend directly from optional local weights (the EMA safetensors). A nil
    /// `weightsURL` builds random weights (`isReady` is true). The reverse-SDE sampler is multi-second;
    /// run it off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(weightsURL: weightsURL, seed: 0)
    }

    /// Builds an SGMSE+ backend with an explicit sampler seed (for repeatable runs).
    public static func backend(weightsURL: URL?, seed: UInt64,
                               config: NFKMLXSGMSEConfiguration = NFKMLXSGMSEConfiguration()) throws -> any NFKInferenceBackend {
        let net = NFKMLXNCSNppNet(config)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXSGMSEBackend(net: net, config: config, identifier: modelName, seed: seed)
    }

    /// Downloads the EMA weights from Hugging Face, then builds.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers SGMSE+ (`sgmse`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads the EMA safetensors (the converter applies the checkpoint's EMA to the network and dumps the
    /// `dnn` state dict, whose keys are the `all_modules.N.*` / `output_layer.*` names the port mirrors).
    /// The only transform is the 4-D Conv2d weight transpose `[out, in, kH, kW]` → `[out, kH, kW, in]`;
    /// the `NIN` `W` (`[in, out]`) and every `Linear`/GroupNorm/Fourier weight are ≤ 2-D and pass through.
    static func loadWeights(into net: NFKMLXNCSNppNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let transpose = checkpoint.needsConvTranspose
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            (transpose && value.ndim == 4) ? (key, value.transposed(0, 2, 3, 1)) : (key, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
