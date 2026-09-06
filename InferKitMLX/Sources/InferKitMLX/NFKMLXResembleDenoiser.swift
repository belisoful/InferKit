// The stage-1 denoiser of Resemble Enhance: a complex STFT, a 2-D (frequency × time) UNet that predicts
// a magnitude mask and a phase residual, and the inverse STFT. Grounded on the released
// `resemble_enhance/denoiser/{denoiser,unet}.py`.
//
// The UNet runs NHWC `[B, F, T, C]` (H = frequency, W = time). The STFT reuses `NFKMLXComplexSTFT`.

import Foundation
import MLX
import MLXNN

// MARK: - 2-D UNet

/// A pre-activation residual block (GroupNorm → GELU → Conv2d, twice), added back to the input.
final class NFKResembleUNetResBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norms") var norms: [GroupNorm]
    @ModuleInfo(key: "convs") var convs: [Conv2d]

    init(dim: Int) {
        _norms.wrappedValue = (0 ..< 2).map { _ in GroupNorm(groupCount: dim / 16, dimensions: dim, pytorchCompatible: true) }
        _convs.wrappedValue = (0 ..< 2).map { _ in Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, padding: 1) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for i in 0 ..< 2 {
            h = norms[i](h)
            h = gelu(h)
            h = convs[i](h)
        }
        return x + h
    }
}

/// Nearest-neighbor resample of `[B, H, W, C]` on the spatial axes. `factor > 1` repeats (upsample),
/// `factor < 1` decimates (downsample), matching `nn.Upsample(scale_factor=...)`.
private func nfkResembleResample(_ x: MLXArray, factor: Double) -> MLXArray {
    if factor == 1 { return x }
    let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
    if factor > 1 {
        let r = Int(factor)
        var y = x.reshaped([b, h, 1, w, 1, c])
        y = broadcast(y, to: [b, h, r, w, r, c])
        return y.reshaped([b, h * r, w * r, c])
    } else {
        let step = Int(1 / factor)
        let hi = MLXArray(Swift.stride(from: 0, to: h, by: step).map { Int32($0) })
        let wi = MLXArray(Swift.stride(from: 0, to: w, by: step).map { Int32($0) })
        return take(take(x, hi, axis: 1), wi, axis: 2)
    }
}

/// A UNet stage: optional upsample, add the skip, a pre-conv, two residual blocks, optional downsample.
final class NFKResembleUNetBlock: Module {
    @ModuleInfo(key: "pre_conv") var preConv: Conv2d
    @ModuleInfo(key: "res_block1") var resBlock1: NFKResembleUNetResBlock
    @ModuleInfo(key: "res_block2") var resBlock2: NFKResembleUNetResBlock
    let scaleFactor: Double

    init(inputDim: Int, outputDim: Int? = nil, scaleFactor: Double = 1) {
        let out = outputDim ?? inputDim
        self.scaleFactor = scaleFactor
        _preConv.wrappedValue = Conv2d(inputChannels: inputDim, outputChannels: out, kernelSize: 3, padding: 1)
        _resBlock1.wrappedValue = NFKResembleUNetResBlock(dim: out)
        _resBlock2.wrappedValue = NFKResembleUNetResBlock(dim: out)
    }

    /// Returns `(downsampled output, skip)`.
    func callAsFunction(_ x0: MLXArray, skip: MLXArray? = nil) -> (MLXArray, MLXArray) {
        var x = x0
        if scaleFactor > 1 { x = nfkResembleResample(x, factor: scaleFactor) }
        if let skip { x = x + skip }
        x = preConv(x)
        x = resBlock1(x)
        x = resBlock2(x)
        let down = scaleFactor < 1 ? nfkResembleResample(x, factor: scaleFactor) : x
        return (down, x)
    }
}

/// The denoiser's 2-D UNet (`input_proj` → 4 encoder stages → 2 middle stages → 4 decoder stages → head).
final class NFKMLXResembleUNet: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Conv2d
    @ModuleInfo(key: "encoder_blocks") var encoderBlocks: [NFKResembleUNetBlock]
    @ModuleInfo(key: "middle_blocks") var middleBlocks: [NFKResembleUNetBlock]
    @ModuleInfo(key: "decoder_blocks") var decoderBlocks: [NFKResembleUNetBlock]
    @ModuleInfo(key: "head") var head: [Module]
    let numBlocks = 4

    init(inputDim: Int = 3, outputDim: Int = 3, hiddenDim: Int = 16, numMiddle: Int = 2) {
        let nb = 4
        _inputProj.wrappedValue = Conv2d(inputChannels: inputDim, outputChannels: hiddenDim, kernelSize: 3, padding: 1)
        _encoderBlocks.wrappedValue = (0 ..< nb).map {
            NFKResembleUNetBlock(inputDim: hiddenDim * (1 << $0), outputDim: hiddenDim * (1 << ($0 + 1)), scaleFactor: 0.5)
        }
        _middleBlocks.wrappedValue = (0 ..< numMiddle).map { _ in NFKResembleUNetBlock(inputDim: hiddenDim * (1 << nb)) }
        _decoderBlocks.wrappedValue = (0 ..< nb).reversed().map {
            NFKResembleUNetBlock(inputDim: hiddenDim * (1 << ($0 + 1)), outputDim: hiddenDim * (1 << $0), scaleFactor: 2)
        }
        _head.wrappedValue = [Conv2d(inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: 3, padding: 1),
                              NFKResembleGELU(),
                              Conv2d(inputChannels: hiddenDim, outputChannels: outputDim, kernelSize: 1)]
    }

    private var scaleFactor: Int { 1 << numBlocks }

    /// `x [1, F, T, 3]` → `[1, F, T, 3]`.
    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        let h0 = x0.dim(1), w0 = x0.dim(2)
        let hpad = (scaleFactor - h0 % scaleFactor) % scaleFactor
        let wpad = (scaleFactor - w0 % scaleFactor) % scaleFactor
        var x = MLX.padded(x0, widths: [IntOrPair((0, 0)), IntOrPair((0, hpad)), IntOrPair((0, wpad)), IntOrPair((0, 0))], mode: .constant)
        x = inputProj(x)
        var skips = [MLXArray]()
        for block in encoderBlocks {
            let (down, s) = block(x)
            x = down
            skips.append(s)
        }
        for block in middleBlocks { x = block(x).0 }
        for (block, s) in zip(decoderBlocks, skips.reversed()) { x = block(x, skip: s).0 }
        for layer in head { x = (layer as! UnaryLayer)(x) }
        return x[0..., 0 ..< h0, 0 ..< w0, 0...]
    }
}

// MARK: - Denoiser

/// The stage-1 denoiser: a mixed waveform → a cleaned waveform through the STFT-mask UNet.
public final class NFKMLXResembleDenoiser: Module {
    @ModuleInfo(key: "net") var net: NFKMLXResembleUNet
    let stft: NFKMLXComplexSTFT
    let eps: Float = 1e-7

    init(hopSize: Int = 420) {
        _net.wrappedValue = NFKMLXResembleUNet()
        let nFFT = hopSize * 4
        stft = NFKMLXComplexSTFT(nFFT: nFFT, hop: hopSize, winLength: nFFT)
    }

    private func normalize(_ x: MLXArray) -> MLXArray {
        x / (abs(x).max() + 1e-7)
    }

    /// `[samples]` → cleaned `[samples]`.
    public func callAsFunction(_ samplesArray: MLXArray) -> MLXArray {
        let length = samplesArray.dim(0)
        let x = normalize(samplesArray).reshaped([1, length])
        let (mag0, phase0) = stft.transform(x)               // [1, bins, frames+1]
        let frames = mag0.dim(2) - 1
        let mag = mag0[0..., 0..., 0 ..< frames]
        let cos = MLX.cos(phase0)[0..., 0..., 0 ..< frames]
        let sin = MLX.sin(phase0)[0..., 0..., 0 ..< frames]
        // net input: stack [mag, cos, sin] on channel → NHWC [1, bins, frames, 3].
        let stacked = concatenated([mag.expandedDimensions(axis: 3),
                                    cos.expandedDimensions(axis: 3),
                                    sin.expandedDimensions(axis: 3)], axis: 3)
        let out = net(stacked)                               // [1, bins, frames, 3]
        let magMask = sigmoid(out[0..., 0..., 0..., 0])      // [1, bins, frames]
        let real = tanh(out[0..., 0..., 0..., 1])
        let imag = tanh(out[0..., 0..., 0..., 2])
        let resMag = sqrt(real * real + imag * imag + eps)
        let cosRes = real / resMag
        let sinRes = imag / resMag
        let sepMag = maximum(mag * magMask, MLXArray(0))
        let sepCos = cos * cosRes - sin * sinRes
        let sepSin = sin * cosRes + cos * sinRes
        // istft: pad the dropped last frame (replicate), invert.
        let last = sepMag.dim(2) - 1
        let padMag = concatenated([sepMag, sepMag[0..., 0..., last ..< (last + 1)]], axis: 2)
        let padCos = concatenated([sepCos, sepCos[0..., 0..., last ..< (last + 1)]], axis: 2)
        let padSin = concatenated([sepSin, sepSin[0..., 0..., last ..< (last + 1)]], axis: 2)
        let outReal = padMag * padCos
        let outImag = padMag * padSin
        var wav = stft.inverseComplex(real: outReal, imaginary: outImag)[0]   // [samples']
        if wav.dim(0) < length {
            wav = MLX.padded(wav, widths: [IntOrPair((0, length - wav.dim(0)))], mode: .constant)
        } else if wav.dim(0) > length {
            wav = wav[0 ..< length]
        }
        return wav
    }
}
