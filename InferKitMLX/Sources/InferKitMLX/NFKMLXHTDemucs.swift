//
//  NFKMLXHTDemucs.swift
//  InferKitMLX
//
//  Hybrid Transformer Demucs (Demucs v4) — music source separation.
//
//  This is a different architecture from `NFKMLXDemucs`, not a configuration of it. A spectrogram
//  branch and a waveform branch run in parallel, a cross-transformer at the bottleneck lets each read
//  the other, and the two reconstructions are added. Tensors flow NHWC: the spectrogram branch is
//  `[batch, frequency, frame, channel]` where the reference is `[batch, channel, frequency, frame]`,
//  and the waveform branch is `[batch, sample, channel]`.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXFFT
import MLXNN
import MLXRandom

// MARK: Short-time Fourier transform

/// The transform pair the spectrogram branch is defined over: a Hann-windowed STFT normalized by
/// `sqrt(nFFT)`, and its inverse.
///
/// Both run in MLX over a batch of signals. Framing gathers each frame's samples, and the overlap-add
/// sums the frames' `nFFT / hop` hop-long pieces into the output block each one lands on, so a gradient
/// reaches the spectrogram branch through the inverse. The transforms run as radix-4 stages (``fft(_:)``).
enum NFKHTDemucsSpectrum {

    /// The discrete Fourier transform over the last axis of a complex array, as Cooley-Tukey stages of
    /// 4-point (and at most one 2-point) transforms joined by exactly rounded twiddle factors.
    ///
    /// @discussion MLX's GPU transform of 4,096 points carries 6.9e-7 relative error, six times the
    /// reference's pocketfft. In 4-point stages it carries 1.2e-7, pocketfft's own, and a trained network's
    /// gradients through the spectrogram branch depend on that difference. A length with another factor
    /// transforms that remainder in one MLX call.
    static func fft(_ x: MLXArray) -> MLXArray {
        let n = x.dim(-1)
        let radix = n % 4 == 0 ? 4 : (n % 2 == 0 ? 2 : n)
        guard radix < n else { return MLXFFT.fft(x, axis: -1) }
        let rest = n / radix
        let leading = Array(x.shape.dropLast())
        // n = n1 + radix · n2: transform over n2 first, turn by the twiddles, then over n1.
        let inner = fft(x.reshaped(leading + [rest, radix]).swappedAxes(-1, -2)).swappedAxes(-1, -2)
        var real = [Float](repeating: 0, count: rest * radix), imaginary = real
        for k2 in 0 ..< rest {
            for n1 in 0 ..< radix {
                let angle = -2 * Double.pi * Double(k2 * n1) / Double(n)
                real[k2 * radix + n1] = Float(cos(angle))
                imaginary[k2 * radix + n1] = Float(sin(angle))
            }
        }
        let twiddles = MLXArray(real, [rest, radix]) + MLXArray(real: 0, imaginary: 1) * MLXArray(imaginary, [rest, radix])
        let outer = MLXFFT.fft(inner * twiddles, axis: -1)       // X[k2 + rest · k1]
        return outer.swappedAxes(-1, -2).reshaped(leading + [n])
    }

    /// The transform of real signals over the last axis, its `n / 2 + 1` non-negative frequencies.
    static func rfft(_ x: MLXArray) -> MLXArray {
        let n = x.dim(-1)
        return fft(x.asType(.complex64))[.ellipsis, 0 ..< (n / 2 + 1)]
    }

    /// The inverse of ``rfft(_:)`` to `n` real samples. The negative frequencies are the conjugates of the
    /// positive ones, and the imaginary parts of the zero and Nyquist bins are ignored, as `irfft` does.
    static func irfft(_ half: MLXArray, n: Int) -> MLXArray {
        let mirrored = take(half, MLXArray(Array(Int32(1) ..< Int32(n - n / 2)).reversed()), axis: -1)
        let full = concatenated([half, conjugate(mirrored)], axis: -1)
        return fft(conjugate(full)).realPart() / Float(n)
    }

    static func hannWindow(_ size: Int) -> [Float] {
        // Periodic, as `torch.hann_window` is by default.
        (0 ..< size).map { 0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(size)) }
    }

    /// Reflects `left` samples at the start and `right` at the end of each signal `[signals, L]`, as
    /// `torch.nn.functional.pad` does in `reflect` mode: the edge sample itself is not repeated.
    static func reflectPadded(_ signals: MLXArray, left: Int, right: Int) -> MLXArray {
        let n = signals.dim(1)
        guard n > 1 else {
            return repeated(signals, count: left + n + right, axis: 1)
        }
        let indices = (-left ..< (n + right)).map { Int32(mirrored($0, n)) }
        return take(signals, MLXArray(indices), axis: 1)
    }

    private static func mirrored(_ index: Int, _ count: Int) -> Int {
        let period = 2 * (count - 1)
        var folded = ((index % period) + period) % period
        if folded >= count { folded = period - folded }
        return folded
    }

    /// `[signals, frames, nFFT/2 + 1]` complex, `torch.stft(normalized: true, center: true)` of
    /// `[signals, L]`.
    static func transform(_ signals: MLXArray, nFFT: Int, hop: Int) -> MLXArray {
        let count = signals.dim(0)
        let padded = reflectPadded(signals, left: nFFT / 2, right: nFFT / 2)
        let frames = 1 + signals.dim(1) / hop
        var gather = [Int32]()
        gather.reserveCapacity(frames * nFFT)
        for f in 0 ..< frames {
            for j in 0 ..< nFFT { gather.append(Int32(f * hop + j)) }
        }
        let window = MLXArray(hannWindow(nFFT))
        let framed = take(padded, MLXArray(gather), axis: 1).reshaped([count, frames, nFFT]) * window
        return rfft(framed) / MLXArray(sqrtf(Float(nFFT)))
    }

    /// The inverse of ``transform(_:nFFT:hop:)`` over `[signals, frames, nFFT/2 + 1]`, trimmed or
    /// zero-padded to `length` samples: `[signals, length]`.
    static func inverseTransform(_ spectrum: MLXArray, nFFT: Int, hop: Int, length: Int) -> MLXArray {
        let (count, frames) = (spectrum.dim(0), spectrum.dim(1))
        let pieces = nFFT / hop
        let window = hannWindow(nFFT)
        let time = irfft(spectrum * MLXArray(sqrtf(Float(nFFT))), n: nFFT) * MLXArray(window)

        let blocks = time.reshaped([count, frames, pieces, hop])
        var summed: MLXArray?
        for piece in 0 ..< pieces {
            let shifted = padded(blocks[0..., 0..., piece, 0...],
                                 widths: [.init((0, 0)), .init((piece, pieces - 1 - piece)), .init((0, 0))])
            summed = summed.map { $0 + shifted } ?? shifted
        }
        let total = (frames + pieces - 1) * hop
        var envelope = [Float](repeating: 0, count: total)
        for f in 0 ..< frames {
            for j in 0 ..< nFFT { envelope[f * hop + j] += window[j] * window[j] }
        }
        // The window overlap is near zero only at the outer edges, which the trim removes.
        let divisor = envelope.map { $0 > 1e-11 ? $0 : 1 }
        let signal = summed!.reshaped([count, total]) / MLXArray(divisor)

        let begin = nFFT / 2
        let end = min(begin + length, total)
        let kept = signal[0..., begin ..< end]
        guard end - begin < length else { return kept }
        return padded(kept, widths: [.init((0, 0)), .init((0, length - (end - begin)))])
    }
}

// MARK: Shared layers

/// `nn.GroupNorm(1, channels)` over a channels-last tensor: one mean and variance per sample across
/// every channel and every position together, then a per-channel scale and shift.
final class NFKHTGroupNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(_ channels: Int, eps: Float = 1e-5) {
        self._weight.wrappedValue = MLXArray.ones([channels])
        self._bias.wrappedValue = MLXArray.zeros([channels])
        self.eps = eps
    }

    let eps: Float

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let axes = Array(1 ..< x.ndim)
        let mean = x.mean(axes: axes, keepDims: true)
        let centered = x - mean
        let variance = (centered * centered).mean(axes: axes, keepDims: true)
        return centered * rsqrt(variance + eps) * weight + bias
    }
}

/// A learned per-channel scale on a residual branch, initialized near zero: 1e-3 in the dilated
/// branches (`dconv_init`) and 1e-4 in the transformer (`init_values`).
final class NFKHTLayerScale: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ channels: Int, initial: Float) { self._scale.wrappedValue = MLXArray.full([channels], values: MLXArray(initial)) }

    func callAsFunction(_ x: MLXArray) -> MLXArray { scale * x }
}

/// One entry of a ``NFKHTDConv`` residual branch: compress, normalize, GELU, expand to twice the
/// width, normalize, gate, and scale.
final class NFKHTDConvBlock: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv1d
    @ModuleInfo(key: "norm_in") var normIn: NFKHTGroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv1d
    @ModuleInfo(key: "norm_out") var normOut: NFKHTGroupNorm
    @ModuleInfo(key: "layer_scale") var layerScale: NFKHTLayerScale

    init(channels: Int, hidden: Int, dilation: Int, kernel: Int = 3) {
        self._convIn.wrappedValue = Conv1d(inputChannels: channels, outputChannels: hidden,
                                           kernelSize: kernel, padding: dilation * (kernel / 2),
                                           dilation: dilation)
        self._normIn.wrappedValue = NFKHTGroupNorm(hidden)
        self._convOut.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: 2 * channels,
                                            kernelSize: 1)
        self._normOut.wrappedValue = NFKHTGroupNorm(2 * channels)
        self._layerScale.wrappedValue = NFKHTLayerScale(channels, initial: 1e-3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let compressed = gelu(normIn(convIn(x)))
        return layerScale(glu(normOut(convOut(compressed)), axis: -1))
    }
}

/// The dilated residual branch every encoder and decoder layer carries. It convolves over time, so a
/// frequency-branch layer folds its frequency axis into the batch before calling it. Its hidden width is
/// the channels over `compress`, the releases' `dconv_comp` of 8.
final class NFKHTDConv: Module {
    @ModuleInfo(key: "layers") var layers: [NFKHTDConvBlock]

    init(channels: Int, compress: Int = 8, depth: Int = 2) {
        let hidden = channels / compress
        self._layers.wrappedValue = (0 ..< depth).map {
            NFKHTDConvBlock(channels: channels, hidden: hidden, dilation: 1 << $0)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in layers { out = out + layer(out) }
        return out
    }
}

// MARK: Encoder and decoder layers

/// An encoder layer of the spectrogram branch: a strided convolution over frequency, the dilated
/// residual branch over time, and a gated 1×1 rewrite.
final class NFKHTFrequencyEncoder: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "rewrite") var rewrite: Conv2d
    @ModuleInfo(key: "dconv") var dconv: NFKHTDConv

    init(inputChannels: Int, outputChannels: Int, kernel: Int = 8, stride: Int = 4) {
        self._conv.wrappedValue = Conv2d(inputChannels: inputChannels, outputChannels: outputChannels,
                                         kernelSize: [kernel, 1], stride: [stride, 1],
                                         padding: [kernel / 4, 0])
        self._rewrite.wrappedValue = Conv2d(inputChannels: outputChannels,
                                            outputChannels: 2 * outputChannels, kernelSize: 1)
        self._dconv.wrappedValue = NFKHTDConv(channels: outputChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = gelu(conv(x))                                   // [batch, frequency, frame, channel]
        let shape = y.shape
        y = dconv(y.reshaped([shape[0] * shape[1], shape[2], shape[3]])).reshaped(shape)
        return glu(rewrite(y), axis: -1)
    }
}

/// An encoder layer of the waveform branch. Same structure over one spatial axis.
final class NFKHTTimeEncoder: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "rewrite") var rewrite: Conv1d
    @ModuleInfo(key: "dconv") var dconv: NFKHTDConv

    let stride: Int

    init(inputChannels: Int, outputChannels: Int, kernel: Int = 8, stride: Int = 4) {
        self.stride = stride
        self._conv.wrappedValue = Conv1d(inputChannels: inputChannels, outputChannels: outputChannels,
                                         kernelSize: kernel, stride: stride, padding: kernel / 4)
        self._rewrite.wrappedValue = Conv1d(inputChannels: outputChannels,
                                            outputChannels: 2 * outputChannels, kernelSize: 1)
        self._dconv.wrappedValue = NFKHTDConv(channels: outputChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        let remainder = input.shape[1] % stride
        if remainder != 0 {
            input = padded(input, widths: [.init((0, 0)), .init((0, stride - remainder)), .init((0, 0))])
        }
        let y = dconv(gelu(conv(input)))
        return glu(rewrite(y), axis: -1)
    }
}

/// A decoder layer of the spectrogram branch. It returns the transposed-convolution output together
/// with the tensor that preceded it, which the reference calls `pre`.
final class NFKHTFrequencyDecoder: Module {
    @ModuleInfo(key: "conv_tr") var convTr: ConvTransposed2d
    @ModuleInfo(key: "rewrite") var rewrite: Conv2d
    @ModuleInfo(key: "dconv") var dconv: NFKHTDConv

    let trim: Int
    let last: Bool

    init(inputChannels: Int, outputChannels: Int, last: Bool, kernel: Int = 8, stride: Int = 4) {
        self.trim = kernel / 4
        self.last = last
        self._convTr.wrappedValue = ConvTransposed2d(inputChannels: inputChannels,
                                                     outputChannels: outputChannels,
                                                     kernelSize: [kernel, 1], stride: [stride, 1])
        self._rewrite.wrappedValue = Conv2d(inputChannels: inputChannels,
                                            outputChannels: 2 * inputChannels,
                                            kernelSize: 3, padding: 1)
        self._dconv.wrappedValue = NFKHTDConv(channels: inputChannels)
    }

    func callAsFunction(_ x: MLXArray, skip: MLXArray) -> (MLXArray, MLXArray) {
        var y = glu(rewrite(x + skip), axis: -1)
        let shape = y.shape
        y = dconv(y.reshaped([shape[0] * shape[1], shape[2], shape[3]])).reshaped(shape)
        var z = convTr(y)
        z = z[0..., trim ..< (z.shape[1] - trim), 0..., 0...]
        return (last ? z : gelu(z), y)
    }
}

/// A decoder layer of the waveform branch.
final class NFKHTTimeDecoder: Module {
    @ModuleInfo(key: "conv_tr") var convTr: ConvTransposed1d
    @ModuleInfo(key: "rewrite") var rewrite: Conv1d
    @ModuleInfo(key: "dconv") var dconv: NFKHTDConv

    let trim: Int
    let last: Bool

    init(inputChannels: Int, outputChannels: Int, last: Bool, kernel: Int = 8, stride: Int = 4) {
        self.trim = kernel / 4
        self.last = last
        self._convTr.wrappedValue = ConvTransposed1d(inputChannels: inputChannels,
                                                     outputChannels: outputChannels,
                                                     kernelSize: kernel, stride: stride)
        self._rewrite.wrappedValue = Conv1d(inputChannels: inputChannels,
                                            outputChannels: 2 * inputChannels,
                                            kernelSize: 3, padding: 1)
        self._dconv.wrappedValue = NFKHTDConv(channels: inputChannels)
    }

    func callAsFunction(_ x: MLXArray, skip: MLXArray, length: Int) -> MLXArray {
        let y = dconv(glu(rewrite(x + skip), axis: -1))
        var z = convTr(y)
        z = z[0..., trim ..< (trim + length), 0...]
        return last ? z : gelu(z)
    }
}

// MARK: Cross-transformer

/// The sinusoidal grids the bottleneck adds to its two token sequences. The waveform branch and the
/// spectrogram branch use different conventions, and each is reproduced as the reference writes it.
enum NFKHTDemucsPositions {

    /// `[length, dimensions]` — `create_sin_embedding`: the first half of the channels are cosines,
    /// the second half sines, over frequencies spaced by `maxPeriod^(index / (half - 1))`.
    static func sequence(length: Int, dimensions: Int, maxPeriod: Float = 10_000) -> MLXArray {
        let half = dimensions / 2
        var data = [Float](repeating: 0, count: length * dimensions)
        for position in 0 ..< length {
            for index in 0 ..< half {
                let phase = Float(position) / powf(maxPeriod, Float(index) / Float(half - 1))
                data[position * dimensions + index] = cosf(phase)
                data[position * dimensions + half + index] = sinf(phase)
            }
        }
        return data.withUnsafeBufferPointer { MLXArray($0, [length, dimensions]) }
    }

    /// `[height, width, dimensions]` — `create_2d_sin_embedding`: the low half of the channels encode
    /// the width (the frame) and the high half the height (the frequency), each alternating sine and
    /// cosine.
    static func grid(height: Int, width: Int, dimensions: Int, maxPeriod: Float = 10_000) -> MLXArray {
        let half = dimensions / 2
        let terms = half / 2
        var divisor = [Float](repeating: 0, count: terms)
        for i in 0 ..< terms { divisor[i] = expf(Float(2 * i) * -(logf(maxPeriod) / Float(half))) }

        var data = [Float](repeating: 0, count: height * width * dimensions)
        for h in 0 ..< height {
            for w in 0 ..< width {
                let base = (h * width + w) * dimensions
                for i in 0 ..< terms {
                    data[base + 2 * i] = sinf(Float(w) * divisor[i])
                    data[base + 2 * i + 1] = cosf(Float(w) * divisor[i])
                    data[base + half + 2 * i] = sinf(Float(h) * divisor[i])
                    data[base + half + 2 * i + 1] = cosf(Float(h) * divisor[i])
                }
            }
        }
        return data.withUnsafeBufferPointer { MLXArray($0, [height, width, dimensions]) }
    }
}

/// Multi-head attention in the reference's fused layout: one `in_proj_weight` holding the query, key,
/// and value projections stacked, and a separate output projection.
final class NFKHTAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let heads: Int
    let dimensions: Int
    /// The probability of dropping an attention weight while training, `nn.MultiheadAttention`'s `dropout`.
    let dropout: Float

    init(dimensions: Int, heads: Int, dropout: Float) {
        self.heads = heads
        self.dimensions = dimensions
        self.dropout = dropout
        // `nn.MultiheadAttention`'s initialization: Xavier-uniform over the stacked projection, zero biases.
        let bound = sqrt(6 / Float(4 * dimensions))
        self._inProjWeight.wrappedValue = MLXRandom.uniform(low: -bound, high: bound, [3 * dimensions, dimensions])
        self._inProjBias.wrappedValue = MLXArray.zeros([3 * dimensions])
        self._outProj.wrappedValue = Linear(dimensions, dimensions)
    }

    func callAsFunction(_ query: MLXArray, _ memory: MLXArray) -> MLXArray {
        let d = dimensions
        let q = project(query, offset: 0)
        let k = project(memory, offset: d)
        let v = project(memory, offset: 2 * d)
        let scale = 1 / sqrt(Float(d / heads))
        let out: MLXArray
        if training && dropout > 0 {
            // The fused kernel has no dropout, so training forms the weights to drop them.
            let weights = softmax(split(q).matmul(split(k).transposed(0, 1, 3, 2)) * scale, axis: -1)
            let keep = 1 - dropout
            let kept = MLXRandom.bernoulli(MLXArray(keep), weights.shape).asType(weights.dtype)
            out = (weights * kept / keep).matmul(split(v))
        } else {
            out = MLXFast.scaledDotProductAttention(queries: split(q), keys: split(k), values: split(v),
                                                    scale: scale, mask: nil)
        }
        let merged = out.transposed(0, 2, 1, 3).reshaped([query.shape[0], query.shape[1], d])
        return outProj(merged)
    }

    private func project(_ x: MLXArray, offset: Int) -> MLXArray {
        let d = dimensions
        return x.matmul(inProjWeight[offset ..< (offset + d), 0...].transposed(1, 0))
            + inProjBias[offset ..< (offset + d)]
    }

    private func split(_ x: MLXArray) -> MLXArray {
        x.reshaped([x.shape[0], x.shape[1], heads, dimensions / heads]).transposed(0, 2, 1, 3)
    }
}

/// A self-attention layer of the bottleneck: pre-norm attention and feed-forward, each scaled by its
/// own learned per-channel factor, then a group normalization over the whole sequence.
final class NFKHTSelfLayer: Module {
    @ModuleInfo(key: "attn") var attn: NFKHTAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm_out") var normOut: NFKHTGroupNorm
    @ModuleInfo(key: "gamma_1") var gamma1: NFKHTLayerScale
    @ModuleInfo(key: "gamma_2") var gamma2: NFKHTLayerScale
    @ModuleInfo(key: "dropout") var dropout: Dropout
    @ModuleInfo(key: "dropout1") var dropout1: Dropout
    @ModuleInfo(key: "dropout2") var dropout2: Dropout

    init(dimensions: Int, heads: Int, hidden: Int, dropout: Float) {
        self._attn.wrappedValue = NFKHTAttention(dimensions: dimensions, heads: heads, dropout: dropout)
        self._linear1.wrappedValue = Linear(dimensions, hidden)
        self._linear2.wrappedValue = Linear(hidden, dimensions)
        self._norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        self._norm2.wrappedValue = LayerNorm(dimensions: dimensions)
        self._normOut.wrappedValue = NFKHTGroupNorm(dimensions)
        self._gamma1.wrappedValue = NFKHTLayerScale(dimensions, initial: 1e-4)
        self._gamma2.wrappedValue = NFKHTLayerScale(dimensions, initial: 1e-4)
        self._dropout.wrappedValue = Dropout(p: dropout)
        self._dropout1.wrappedValue = Dropout(p: dropout)
        self._dropout2.wrappedValue = Dropout(p: dropout)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = norm1(x)
        var out = x + gamma1(dropout1(attn(normed, normed)))
        out = out + gamma2(dropout2(linear2(dropout(gelu(linear1(norm2(out)))))))
        return normOut(out)
    }
}

/// A cross-attention layer of the bottleneck: one branch's tokens attend to the other branch's.
final class NFKHTCrossLayer: Module {
    @ModuleInfo(key: "attn") var attn: NFKHTAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "norm_out") var normOut: NFKHTGroupNorm
    @ModuleInfo(key: "gamma_1") var gamma1: NFKHTLayerScale
    @ModuleInfo(key: "gamma_2") var gamma2: NFKHTLayerScale
    @ModuleInfo(key: "dropout") var dropout: Dropout
    @ModuleInfo(key: "dropout1") var dropout1: Dropout
    @ModuleInfo(key: "dropout2") var dropout2: Dropout

    init(dimensions: Int, heads: Int, hidden: Int, dropout: Float) {
        self._attn.wrappedValue = NFKHTAttention(dimensions: dimensions, heads: heads, dropout: dropout)
        self._linear1.wrappedValue = Linear(dimensions, hidden)
        self._linear2.wrappedValue = Linear(hidden, dimensions)
        self._norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        self._norm2.wrappedValue = LayerNorm(dimensions: dimensions)
        self._norm3.wrappedValue = LayerNorm(dimensions: dimensions)
        self._normOut.wrappedValue = NFKHTGroupNorm(dimensions)
        self._gamma1.wrappedValue = NFKHTLayerScale(dimensions, initial: 1e-4)
        self._gamma2.wrappedValue = NFKHTLayerScale(dimensions, initial: 1e-4)
        self._dropout.wrappedValue = Dropout(p: dropout)
        self._dropout1.wrappedValue = Dropout(p: dropout)
        self._dropout2.wrappedValue = Dropout(p: dropout)
    }

    func callAsFunction(_ query: MLXArray, _ memory: MLXArray) -> MLXArray {
        var out = query + gamma1(dropout1(attn(norm1(query), norm2(memory))))
        out = out + gamma2(dropout2(linear2(dropout(gelu(linear1(norm3(out)))))))
        return normOut(out)
    }
}

/// The bottleneck: five layers per branch, alternating self-attention and cross-attention, with the
/// two branches exchanging tokens at every cross layer.
final class NFKHTCrossTransformer: Module {
    @ModuleInfo(key: "norm_in") var normIn: LayerNorm
    @ModuleInfo(key: "norm_in_t") var normInT: LayerNorm
    @ModuleInfo(key: "self_layers") var selfLayers: [NFKHTSelfLayer]
    @ModuleInfo(key: "cross_layers") var crossLayers: [NFKHTCrossLayer]
    @ModuleInfo(key: "self_layers_t") var selfLayersT: [NFKHTSelfLayer]
    @ModuleInfo(key: "cross_layers_t") var crossLayersT: [NFKHTCrossLayer]

    let layerCount: Int

    init(dimensions: Int, heads: Int, hidden: Int, layers: Int, dropout: Float) {
        self.layerCount = layers
        self._normIn.wrappedValue = LayerNorm(dimensions: dimensions)
        self._normInT.wrappedValue = LayerNorm(dimensions: dimensions)
        let selfCount = (layers + 1) / 2, crossCount = layers / 2
        self._selfLayers.wrappedValue = (0 ..< selfCount).map { _ in
            NFKHTSelfLayer(dimensions: dimensions, heads: heads, hidden: hidden, dropout: dropout)
        }
        self._selfLayersT.wrappedValue = (0 ..< selfCount).map { _ in
            NFKHTSelfLayer(dimensions: dimensions, heads: heads, hidden: hidden, dropout: dropout)
        }
        self._crossLayers.wrappedValue = (0 ..< crossCount).map { _ in
            NFKHTCrossLayer(dimensions: dimensions, heads: heads, hidden: hidden, dropout: dropout)
        }
        self._crossLayersT.wrappedValue = (0 ..< crossCount).map { _ in
            NFKHTCrossLayer(dimensions: dimensions, heads: heads, hidden: hidden, dropout: dropout)
        }
    }

    /// - Parameters:
    ///   - x: `[batch, frequency, frame, channel]`, the spectrogram branch.
    ///   - xt: `[batch, sample, channel]`, the waveform branch.
    func callAsFunction(_ x: MLXArray, _ xt: MLXArray) -> (MLXArray, MLXArray) {
        let batch = x.shape[0], frequencies = x.shape[1], frames = x.shape[2], channels = x.shape[3]
        // The reference orders the spectrogram tokens frame-major, so a token's neighbors in the
        // sequence are the frequencies of one frame.
        let grid = NFKHTDemucsPositions.grid(height: frequencies, width: frames, dimensions: channels)
            .transposed(1, 0, 2).reshaped([1, frames * frequencies, channels])
        var tokens = x.transposed(0, 2, 1, 3).reshaped([batch, frames * frequencies, channels])
        tokens = normIn(tokens) + grid

        let sequence = NFKHTDemucsPositions.sequence(length: xt.shape[1], dimensions: channels)
            .reshaped([1, xt.shape[1], channels])
        var timeTokens = normInT(xt) + sequence

        for index in 0 ..< layerCount {
            if index % 2 == 0 {
                tokens = selfLayers[index / 2](tokens)
                timeTokens = selfLayersT[index / 2](timeTokens)
            } else {
                let previous = tokens
                tokens = crossLayers[index / 2](tokens, timeTokens)
                timeTokens = crossLayersT[index / 2](timeTokens, previous)
            }
        }
        let out = tokens.reshaped([batch, frames, frequencies, channels]).transposed(0, 2, 1, 3)
        return (out, timeTokens)
    }
}

// MARK: The network

/// The geometry of a Hybrid Transformer Demucs release.
public struct NFKMLXHTDemucsConfiguration: Sendable {
    public var sources: Int = 4
    public var audioChannels: Int = 2
    public var channels: Int = 48
    public var depth: Int = 4
    public var nFFT: Int = 4096
    public var bottomChannels: Int = 512
    public var transformerLayers: Int = 5
    public var transformerHeads: Int = 8
    public var transformerHidden: Int = 2048
    /// The probability the bottleneck's attention and feed-forward drop an activation while training,
    /// the reference's `t_dropout`; 0.02 in every release. Inference applies none. Introduced in InferKit 0.4.0.
    public var transformerDropout: Float = 0.02
    /// The reference's `freq_emb` weight and `ScaledEmbedding` factor. The checkpoint stores the
    /// embedding divided by `embeddingScale` and the forward multiplies it back, so the effective
    /// factor is their product (2.0 for the release), not `frequencyEmbeddingScale` alone.
    public var frequencyEmbeddingScale: Float = 0.2
    public var embeddingScale: Float = 10
    public var sampleRate: Int = 44_100
    /// The segment the release was trained on, in samples. A shorter clip is zero-padded up to it, as
    /// the reference does, because the bottleneck's positional grids and the branch normalizations are
    /// both length-dependent.
    public var trainingSamples: Int = 343_980

    public var hop: Int { nFFT / 4 }

    public init() {}

    public static let htdemucs = NFKMLXHTDemucsConfiguration()

    /// The released `htdemucs_6s`: six stems, guitar and piano after the four the base release
    /// separates. Its `bottom_channels` is 0, so it carries no channel samplers and runs the
    /// cross-transformer at the deepest encoder's own width (384) with a 1536-wide feed-forward.
    public static let htdemucs6s: NFKMLXHTDemucsConfiguration = {
        var configuration = NFKMLXHTDemucsConfiguration()
        configuration.sources = 6
        configuration.bottomChannels = 0
        configuration.transformerHidden = 1536
        return configuration
    }()

    /// The width the cross-transformer runs at: `bottomChannels`, or the deepest encoder's width when
    /// the release sets `bottom_channels` to 0 and builds no samplers.
    public var transformerWidth: Int {
        bottomChannels > 0 ? bottomChannels : channels << (depth - 1)
    }

    /// The stem names in the release's order; the four-stem releases stop at `vocals`.
    public static let stemNames = ["drums", "bass", "other", "vocals", "guitar", "piano"]
}

/// A bag of Hybrid Transformer Demucs models, the form the fine-tuned release (`htdemucs_ft`) takes:
/// four checkpoints of the base geometry, each fine-tuned for one stem, combined by per-source weights.
///
/// The reference's `BagOfModels` sums each model's estimate scaled by its weight for that source and
/// divides by the weights' total; `htdemucs_ft` uses one-hot weights, so each stem comes from the model
/// fine-tuned for it.
public final class NFKMLXHTDemucsBag {
    public let nets: [NFKMLXHTDemucsNet]
    /// `weights[model][source]`.
    public let weights: [[Float]]

    public var configuration: NFKMLXHTDemucsConfiguration { nets[0].configuration }

    /// The weights `htdemucs_ft` uses: model `i` contributes stem `i` alone.
    public static func oneHotWeights(models: Int, sources: Int) -> [[Float]] {
        (0 ..< models).map { model in (0 ..< sources).map { $0 == model ? 1 : 0 } }
    }

    public init(nets: [NFKMLXHTDemucsNet], weights: [[Float]]) throws {
        guard let first = nets.first else {
            throw NFKMLXError.unsupportedConfiguration("a bag holds at least one model")
        }
        guard weights.count == nets.count,
              weights.allSatisfy({ $0.count == first.configuration.sources }) else {
            throw NFKMLXError.unsupportedConfiguration(
                "a bag needs one weight per source for each of its \(nets.count) models")
        }
        self.nets = nets
        self.weights = weights
    }

    /// Separates `samples` (`[channels, length]`) into `[sources, channels, length]`, the weighted mean
    /// of the models' estimates.
    public func separate(_ samples: MLXArray, padsToTrainingSegment: Bool = true) -> MLXArray {
        let sources = configuration.sources
        var total: MLXArray?
        var totalWeight = [Float](repeating: 0, count: sources)
        for (net, weight) in zip(nets, weights) {
            let scaled = net.separate(samples, padsToTrainingSegment: padsToTrainingSegment)
                * MLXArray(weight).reshaped([sources, 1, 1])
            total = total.map { $0 + scaled } ?? scaled
            for (index, value) in weight.enumerated() { totalWeight[index] += value }
        }
        return total! / MLXArray(totalWeight).reshaped([sources, 1, 1])
    }
}

/// A separation together with the tensors at each stage boundary. Only the parity test reads these;
/// callers take ``NFKMLXHTDemucsNet/separate(_:padsToTrainingSegment:)``.
struct NFKHTDemucsTrace {
    let spectrogram: MLXArray                                   // [frequency, frame, channel · 2]
    let bottleneckIn: MLXArray                                  // [frequency, frame, channel]
    let bottleneckOut: MLXArray                                 // [frequency, frame, bottomChannel]
    let frequencyOut: MLXArray                                  // [frequency, frame, source · channel · 2]
    let timeOut: MLXArray                                       // [sample, source · channel]
    let waveform: MLXArray                                      // [source, channel, sample]
}

/// Hybrid Transformer Demucs. The spectrogram branch predicts a complex spectrogram, the waveform
/// branch predicts a waveform, and the two reconstructions are added.
public final class NFKMLXHTDemucsNet: Module {
    @ModuleInfo(key: "encoder") var encoder: [NFKHTFrequencyEncoder]
    @ModuleInfo(key: "decoder") var decoder: [NFKHTFrequencyDecoder]
    @ModuleInfo(key: "tencoder") var timeEncoder: [NFKHTTimeEncoder]
    @ModuleInfo(key: "tdecoder") var timeDecoder: [NFKHTTimeDecoder]
    @ModuleInfo(key: "freq_emb") var frequencyEmbedding: Embedding
    @ModuleInfo(key: "channel_upsampler") var channelUpsampler: Linear?
    @ModuleInfo(key: "channel_downsampler") var channelDownsampler: Linear?
    @ModuleInfo(key: "channel_upsampler_t") var timeUpsampler: Linear?
    @ModuleInfo(key: "channel_downsampler_t") var timeDownsampler: Linear?
    @ModuleInfo(key: "crosstransformer") var crossTransformer: NFKHTCrossTransformer

    public let configuration: NFKMLXHTDemucsConfiguration

    public init(configuration: NFKMLXHTDemucsConfiguration = .htdemucs) {
        self.configuration = configuration
        let c = configuration
        var frequencyIn = c.audioChannels * 2                   // real and imaginary as channels
        var timeIn = c.audioChannels
        var width = c.channels
        var encoders = [NFKHTFrequencyEncoder]()
        var timeEncoders = [NFKHTTimeEncoder]()
        var decoders = [NFKHTFrequencyDecoder]()
        var timeDecoders = [NFKHTTimeDecoder]()
        for index in 0 ..< c.depth {
            encoders.append(NFKHTFrequencyEncoder(inputChannels: frequencyIn, outputChannels: width))
            timeEncoders.append(NFKHTTimeEncoder(inputChannels: timeIn, outputChannels: width))
            let last = index == 0
            decoders.append(NFKHTFrequencyDecoder(
                inputChannels: width,
                outputChannels: last ? c.sources * c.audioChannels * 2 : frequencyIn, last: last))
            timeDecoders.append(NFKHTTimeDecoder(
                inputChannels: width,
                outputChannels: last ? c.sources * c.audioChannels : timeIn, last: last))
            frequencyIn = width
            timeIn = width
            width *= 2
        }
        // The reference builds the decoders front to back and inserts each at the head of the list, so
        // decoder 0 is the deepest.
        self._encoder.wrappedValue = encoders
        self._timeEncoder.wrappedValue = timeEncoders
        self._decoder.wrappedValue = decoders.reversed()
        self._timeDecoder.wrappedValue = timeDecoders.reversed()

        self._frequencyEmbedding.wrappedValue = Embedding(embeddingCount: c.nFFT / 8,
                                                          dimensions: c.channels)
        if c.bottomChannels > 0 {
            self._channelUpsampler.wrappedValue = Linear(frequencyIn, c.bottomChannels)
            self._channelDownsampler.wrappedValue = Linear(c.bottomChannels, frequencyIn)
            self._timeUpsampler.wrappedValue = Linear(timeIn, c.bottomChannels)
            self._timeDownsampler.wrappedValue = Linear(c.bottomChannels, timeIn)
        }
        self._crossTransformer.wrappedValue = NFKHTCrossTransformer(
            dimensions: c.transformerWidth, heads: c.transformerHeads,
            hidden: c.transformerHidden, layers: c.transformerLayers, dropout: c.transformerDropout)
        super.init()
        // The transformer drops activations in training mode, so a built network starts in evaluation.
        train(false)
    }

    /// Separates `samples` (`[channels, length]`) into `[sources, channels, length]`.
    ///
    /// - Parameters:
    ///   - samples: the stereo mix, `[channels, length]`.
    ///   - padsToTrainingSegment: when true (the default) a shorter clip is zero-padded to the
    ///     release's training segment and the result trimmed back, which is what the reference does
    ///     at inference. Pass false to run the clip at its own length.
    public func separate(_ samples: MLXArray, padsToTrainingSegment: Bool = true) -> MLXArray {
        trace(samples, padsToTrainingSegment: padsToTrainingSegment).waveform
    }

    /// Separates a batch `[N, channels, L]` into `[N, sources, channels, L]` at the clips' own length,
    /// each clip normalized by its own statistics: the reference's forward in training, where it pads
    /// nothing. Introduced in InferKit 0.4.0.
    public func estimate(_ mix: MLXArray) -> MLXArray {
        forward(mix).waveform
    }

    /// The separation together with the tensors bracketing each stage, so a numeric comparison against
    /// the reference names the stage that diverged rather than only the audio.
    func trace(_ samples: MLXArray, padsToTrainingSegment: Bool = true) -> NFKHTDemucsTrace {
        let c = configuration
        let requested = samples.shape[1]
        var mix = samples
        if padsToTrainingSegment && requested < c.trainingSamples {
            mix = padded(mix, widths: [.init((0, 0)), .init((0, c.trainingSamples - requested))])
        }
        let pass = forward(mix.expandedDimensions(axis: 0))
        let out = pass.waveform[0]
        return NFKHTDemucsTrace(
            spectrogram: pass.spectrogram[0], bottleneckIn: pass.bottleneckIn[0],
            bottleneckOut: pass.bottleneckOut[0], frequencyOut: pass.frequencyOut[0], timeOut: pass.timeOut[0],
            waveform: requested < out.shape[2] ? out[0..., 0..., 0 ..< requested] : out)
    }

    /// Both branches over `[N, channels, L]`, returning `[N, sources, channels, L]` and every seam batched.
    private func forward(_ mix: MLXArray)
        -> (waveform: MLXArray, spectrogram: MLXArray, bottleneckIn: MLXArray, bottleneckOut: MLXArray,
            frequencyOut: MLXArray, timeOut: MLXArray) {
        let c = configuration
        let batch = mix.shape[0], length = mix.shape[2]

        let spectrogramChannels = complexAsChannels(spectrogram(mix, length: length))
        var x = spectrogramChannels
        let mean = x.mean(axes: [1, 2, 3], keepDims: true)
        let standardDeviation = Self.standardDeviation(of: x, axes: [1, 2, 3])
        x = (x - mean) / (1e-5 + standardDeviation)

        var xt = mix.transposed(0, 2, 1)
        let timeMean = xt.mean(axes: [1, 2], keepDims: true)
        let timeDeviation = Self.standardDeviation(of: xt, axes: [1, 2])
        xt = (xt - timeMean) / (1e-5 + timeDeviation)

        var saved = [MLXArray](), savedTime = [MLXArray](), timeLengths = [Int]()
        for index in 0 ..< c.depth {
            timeLengths.append(xt.shape[1])
            xt = timeEncoder[index](xt)
            savedTime.append(xt)
            x = encoder[index](x)
            if index == 0 {
                let positions = MLXArray(Array(Int32(0) ..< Int32(x.shape[1])))
                let embedding = frequencyEmbedding(positions).reshaped([1, x.shape[1], 1, x.shape[3]])
                // `ScaledEmbedding` divides its weight by `scale` at initialization and multiplies it
                // back in the forward, so the stored weight is the embedding over that factor.
                x = x + (c.frequencyEmbeddingScale * c.embeddingScale) * embedding
            }
            saved.append(x)
        }

        let bottleneckIn = x
        let bottomFrequencies = x.shape[1], bottomFrames = x.shape[2]
        // The channel sampler is a 1×1 convolution over the flattened grid, which is frequency-major —
        // a different order from the transformer's own frame-major tokens.
        let width = c.transformerWidth
        var bottom = x
        var bottomTime = xt
        if let channelUpsampler, let timeUpsampler {
            bottom = channelUpsampler(x.reshaped([batch, bottomFrequencies * bottomFrames, x.shape[3]]))
                .reshaped([batch, bottomFrequencies, bottomFrames, width])
            bottomTime = timeUpsampler(xt)
        }
        (bottom, bottomTime) = crossTransformer(bottom, bottomTime)
        let bottleneckOut = bottom
        if let channelDownsampler, let timeDownsampler {
            x = channelDownsampler(bottom.reshaped([batch, bottomFrequencies * bottomFrames, width]))
                .reshaped([batch, bottomFrequencies, bottomFrames, saved[c.depth - 1].shape[3]])
            xt = timeDownsampler(bottomTime)
        } else {
            x = bottom
            xt = bottomTime
        }

        for index in 0 ..< c.depth {
            (x, _) = decoder[index](x, skip: saved.removeLast())
            xt = timeDecoder[index](xt, skip: savedTime.removeLast(), length: timeLengths.removeLast())
        }

        let frequencyOut = x, timeOut = xt
        x = x * standardDeviation + mean
        xt = xt * timeDeviation + timeMean

        let waveform = inverseSpectrogram(x, length: length)    // [N, sources, channels, length]
        let direct = xt.reshaped([batch, length, c.sources, c.audioChannels]).transposed(0, 2, 3, 1)
        return (waveform + direct, spectrogramChannels, bottleneckIn, bottleneckOut, frequencyOut, timeOut)
    }

    /// `torch.std` over `axes` of each example, which divides by `n - 1`; the branch normalizations
    /// depend on the difference.
    private static func standardDeviation(of x: MLXArray, axes: [Int]) -> MLXArray {
        let count = Float(axes.reduce(1) { $0 * x.shape[$1] })
        let centered = x - x.mean(axes: axes, keepDims: true)
        return sqrt((centered * centered).sum(axes: axes, keepDims: true) / (count - 1))
    }

    /// `[N, channels, frequency, frame]` complex, dropping the Nyquist bin and the frames the reference's
    /// extra padding introduces so that the frame count is exactly the sample count over the hop.
    private func spectrogram(_ mix: MLXArray, length: Int) -> MLXArray {
        let c = configuration
        let hop = c.hop
        let frameCount = (length + hop - 1) / hop
        let pad = hop / 2 * 3
        let signals = NFKHTDemucsSpectrum.reflectPadded(
            mix.reshaped([mix.shape[0] * mix.shape[1], length]), left: pad, right: pad + frameCount * hop - length)
        let spectrum = NFKHTDemucsSpectrum.transform(signals, nFFT: c.nFFT, hop: hop)
        return spectrum[0..., 2 ..< (2 + frameCount), 0 ..< (c.nFFT / 2)]
            .reshaped([mix.shape[0], mix.shape[1], frameCount, c.nFFT / 2]).transposed(0, 1, 3, 2)
    }

    /// `torch.view_as_real(...).permute(...)`: channel `c` of the input becomes channels `2c` and
    /// `2c + 1` of the output, `[N, frequency, frame, channels · 2]`.
    private func complexAsChannels(_ z: MLXArray) -> MLXArray {
        let real = z.realPart().expandedDimensions(axis: 4)
        let imaginary = z.imaginaryPart().expandedDimensions(axis: 4)
        let pairs = concatenated([real, imaginary], axis: 4)    // [N, channel, frequency, frame, 2]
        return pairs.transposed(0, 2, 3, 1, 4).reshaped([z.shape[0], z.shape[2], z.shape[3], z.shape[1] * 2])
    }

    /// Turns the spectrogram branch's `[N, frequency, frame, sources · channels · 2]` prediction back
    /// into `[N, sources, channels, length]` samples.
    private func inverseSpectrogram(_ x: MLXArray, length: Int) -> MLXArray {
        let c = configuration
        let batch = x.shape[0], frequencies = x.shape[1], frames = x.shape[2]
        let signals = c.sources * c.audioChannels
        let parts = x.reshaped([batch, frequencies, frames, signals, 2])
        var spectrum = parts.transposed(0, 3, 2, 1, 4)          // [N, signal, frame, frequency, 2]
        // The reference restores the Nyquist bin as zero and re-adds the two frames it dropped.
        spectrum = padded(spectrum, widths: [.init((0, 0)), .init((0, 0)), .init((2, 2)), .init((0, 1)),
                                             .init((0, 0))])
        let complexSpectrum = (spectrum[0..., 0..., 0..., 0..., 0] +
            MLXArray(real: 0, imaginary: 1) * spectrum[0..., 0..., 0..., 0..., 1])
            .reshaped([batch * signals, frames + 4, frequencies + 1])

        let hop = c.hop
        let pad = hop / 2 * 3
        let target = hop * ((length + hop - 1) / hop) + 2 * pad
        let waves = NFKHTDemucsSpectrum.inverseTransform(complexSpectrum, nFFT: c.nFFT, hop: hop, length: target)
        return waves[0..., pad ..< (pad + length)].reshaped([batch, c.sources, c.audioChannels, length])
    }
}

// MARK: Weights

public extension NFKMLXHTDemucs {

    /// Translates a reference `htdemucs` checkpoint key to this module's.
    static func remapReferenceKey(_ key: String) -> String {
        var name = key
        for index in 0 ..< 2 {
            let prefix = "dconv.layers.\(index)."
            name = name.replacingOccurrences(of: prefix + "0.", with: prefix + "conv_in.")
            name = name.replacingOccurrences(of: prefix + "1.", with: prefix + "norm_in.")
            name = name.replacingOccurrences(of: prefix + "3.", with: prefix + "conv_out.")
            name = name.replacingOccurrences(of: prefix + "4.", with: prefix + "norm_out.")
            name = name.replacingOccurrences(of: prefix + "6.", with: prefix + "layer_scale.")
        }
        // The bottleneck alternates two layer classes in one list; MLX holds each class in its own
        // array, so the reference's positions split into two index spaces.
        for branch in ["layers", "layers_t"] {
            let suffix = branch == "layers" ? "" : "_t"
            for (position, index) in [(0, 0), (2, 1), (4, 2)] {
                name = name.replacingOccurrences(of: "crosstransformer.\(branch).\(position).",
                                                 with: "crosstransformer.self_layers\(suffix).\(index).")
            }
            for (position, index) in [(1, 0), (3, 1)] {
                name = name.replacingOccurrences(of: "crosstransformer.\(branch).\(position).",
                                                 with: "crosstransformer.cross_layers\(suffix).\(index).")
            }
        }
        name = name.replacingOccurrences(of: ".self_attn.", with: ".attn.")
        name = name.replacingOccurrences(of: ".cross_attn.", with: ".attn.")
        name = name.replacingOccurrences(of: "freq_emb.embedding.weight", with: "freq_emb.weight")
        return name
    }

    /// Loads a reference checkpoint, or one `NFKMLXWeights` saved, into `net`.
    ///
    /// @discussion The releases store every tensor at float16. The reference's `load_state_dict` copies
    /// them into float32 parameters, and so does this loader: a half-precision parameter computes the same
    /// forward against float32 activations, but its gradient comes back at half precision, where a trained
    /// network's small gradients lose most of their digits.
    static func loadWeights(into net: NFKMLXHTDemucsNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { key, stored -> (String, MLXArray) in
            let name = remapReferenceKey(key)
            let value = stored.dtype == .float16 || stored.dtype == .bfloat16 ? stored.asType(.float32) : stored
            guard checkpoint.needsConvTranspose else { return (name, value) }
            if name.hasSuffix("conv_tr.weight") {
                // A transposed convolution stores `[in, out, kernel…]` where a forward one stores
                // `[out, in, kernel…]`, so the two need different rotations.
                return (name, value.ndim == 4 ? value.transposed(1, 2, 3, 0) : value.transposed(1, 2, 0))
            }
            // The channel samplers are 1×1 convolutions the module holds as linear layers, whose weight
            // is `[out, in]` with no kernel axis at all.
            if name.hasPrefix("channel_") && value.ndim == 3 { return (name, value.squeezed(axis: 2)) }
            if value.ndim == 4 { return (name, value.transposed(0, 2, 3, 1)) }
            if value.ndim == 3 { return (name, value.transposed(0, 2, 1)) }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

// MARK: Backend

/// Reads audio under `NFKInputAudio` and returns one `NFKAudioAsset` per stem, under its name.
@objc(NFKMLXHTDemucsBackend)
public final class NFKMLXHTDemucsBackend: NSObject, NFKInferenceBackend {

    private let configuration: NFKMLXHTDemucsConfiguration
    private let separate: (MLXArray) -> MLXArray
    private let identifier: String
    private let outputDirectory: URL
    private let stemNames: [String]

    init(net: NFKMLXHTDemucsNet, identifier: String = "htdemucs",
         stemNames: [String] = NFKMLXHTDemucsConfiguration.stemNames,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        configuration = net.configuration
        separate = { net.separate($0) }
        self.identifier = identifier
        self.stemNames = stemNames
        self.outputDirectory = outputDirectory
        super.init()
    }

    init(bag: NFKMLXHTDemucsBag, identifier: String = "htdemucs-ft",
         stemNames: [String] = NFKMLXHTDemucsConfiguration.stemNames,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        configuration = bag.configuration
        separate = { bag.separate($0) }
        self.identifier = identifier
        self.stemNames = stemNames
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { [] }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }

    /// The result keys the backend writes: one per stem, named as the backend names it. Introduced in
    /// InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { Set(stemNames.prefix(configuration.sources)) }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else {
            throw NFKMLXError.unsupportedInput
        }
        // The release is trained at 44.1 kHz; a clip at another rate puts every partial in the wrong
        // frequency bin, which separates cleanly into the wrong stems.
        let modelRate = configuration.sampleRate
        let matched = NFKMLXAudioRate.matched(samples, from: sampleRate, to: modelRate)
        let channels = configuration.audioChannels
        // `NFKMLXWaveFile` downmixes on read, so a mono clip drives both channels of a stereo model.
        let mix = (matched + matched).withUnsafeBufferPointer {
            MLXArray($0, [channels, matched.count])
        }
        let stems = separate(mix)
        eval(stems)

        var outputs: [String: Any] = [:]
        let length = stems.shape[2]
        for (index, name) in stemNames.enumerated() where index < configuration.sources {
            let stem = stems[index].transposed(1, 0).reshaped([-1]).asArray(Float.self)
            let url = outputDirectory.appendingPathComponent("htdemucs-\(name)-\(UUID().uuidString).wav")
            try NFKMLXWaveFile.write(samples: stem, sampleRate: modelRate, channels: channels, to: url)
            outputs[name] = NFKAudioAsset(fileURL: url, durationSeconds: Double(length) / Double(modelRate),
                                          sampleRate: Double(modelRate), channelCount: channels)
        }
        return NFKInferenceResult(outputs: outputs)
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

/// The single-model Hybrid Transformer Demucs release to build, for the Objective-C factory.
@objc(NFKMLXHTDemucsVariant)
public enum NFKMLXHTDemucsVariant: Int {
    /// `htdemucs`: drums, bass, other, vocals.
    case fourStem
    /// `htdemucs_6s`: the four stems plus guitar and piano.
    case sixStem
}

private final class NFKMLXHTDemucsBagHolder: @unchecked Sendable {
    let bag: NFKMLXHTDemucsBag
    init(_ bag: NFKMLXHTDemucsBag) { self.bag = bag }
}

/// Registration, factories, and weight loading for Hybrid Transformer Demucs.
@objc(NFKMLXHTDemucs)
public final class NFKMLXHTDemucs: NSObject {

    @objc public static let modelName = "htdemucs"
    @objc public static let sixStemModelName = "htdemucs-6s"
    @objc public static let fineTunedModelName = "htdemucs-ft"

    static func makeNet(_ configuration: NFKMLXHTDemucsConfiguration = .htdemucs) -> NFKMLXHTDemucsNet {
        let net = NFKMLXHTDemucsNet(configuration: configuration)
        net.train(false)
        return net
    }

    static func specs(for variant: NFKMLXHTDemucsVariant) -> (name: String, configuration: NFKMLXHTDemucsConfiguration) {
        switch variant {
        case .fourStem: return (modelName, .htdemucs)
        case .sixStem: return (sixStemModelName, .htdemucs6s)
        }
    }

    /// Builds a Hybrid Transformer Demucs backend from optional local weights. A nil `weightsURL`
    /// builds random weights (`isReady` is true). Run inference off the
    /// render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .fourStem, weightsURL: weightsURL)
    }

    /// Builds one of the released single-model geometries: the four-stem `htdemucs` or the six-stem
    /// `htdemucs_6s`, whose checkpoint fits only `.sixStem`.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXHTDemucsVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let spec = specs(for: variant)
        let net = makeNet(spec.configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXHTDemucsHolder(net)
        return NFKMLXHTDemucsBackend(net: holder.net, identifier: spec.name)
    }

    /// Builds the fine-tuned release (`htdemucs_ft`): four checkpoints of the four-stem geometry, in the
    /// release's order (drums, bass, other, vocals), each contributing the stem it was fine-tuned for.
    /// Four times the memory and the compute of one model, for the release's better separation.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithFineTunedWeightsURLs:error:)
    public static func backend(fineTunedWeightsURLs urls: [URL]) throws -> any NFKInferenceBackend {
        let bag = try fineTunedBag(weightsURLs: urls)
        let holder = NFKMLXHTDemucsBagHolder(bag)
        return NFKMLXHTDemucsBackend(bag: holder.bag, identifier: fineTunedModelName)
    }

    /// The fine-tuned bag itself, for a Swift caller who wants the stems as arrays.
    public static func fineTunedBag(weightsURLs urls: [URL]) throws -> NFKMLXHTDemucsBag {
        let sources = NFKMLXHTDemucsConfiguration.htdemucs.sources
        guard urls.count == sources else {
            throw NFKMLXError.unsupportedConfiguration(
                "htdemucs_ft is \(sources) checkpoints, one per stem; \(urls.count) were given")
        }
        let nets = try urls.map { url -> NFKMLXHTDemucsNet in
            let net = makeNet()
            try loadWeights(into: net, from: url)
            return net
        }
        return try NFKMLXHTDemucsBag(nets: nets, weights: NFKMLXHTDemucsBag.oneHotWeights(models: sources, sources: sources))
    }

    /// Downloads the checkpoint from Hugging Face, then builds. Blocking
    /// on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try backend(variant: .fourStem, repo: repo, weightsPath: weightsPath, revision: revision,
                        cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// The download factory at a chosen geometry.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXHTDemucsVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath,
                                                    revision: revision, cacheDirectoryURL: cacheDirectoryURL)
            return try backend(variant: variant, weightsURL: url)
        }
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        backend(variant: .fourStem, repo: repo, weightsPath: weightsPath, revision: revision,
                cacheDirectoryURL: cacheDirectoryURL, completionHandler: completionHandler)
    }

    /// The asynchronous download factory at a chosen geometry.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXHTDemucsVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers the single-model releases (`htdemucs`, `htdemucs-6s`) with `NFKMLXModelRegistry`. The
    /// fine-tuned bag takes four files, which the one-URL registry cannot name; build it through
    /// `backendWithFineTunedWeightsURLs:`.
    @objc public static func register() {
        for variant in [NFKMLXHTDemucsVariant.fourStem, .sixStem] {
            NFKMLXModelRegistry.register(name: specs(for: variant).name) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL)
            }
        }
    }
}

private final class NFKMLXHTDemucsHolder: @unchecked Sendable {
    let net: NFKMLXHTDemucsNet
    init(_ net: NFKMLXHTDemucsNet) { self.net = net }
}
