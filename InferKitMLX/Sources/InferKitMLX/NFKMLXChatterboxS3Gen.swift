//
//  NFKMLXChatterboxS3Gen.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXFFT
import MLXNN

// Chatterbox stage 4: S3Gen's token-to-mel half (CosyVoice 2 lineage). The voice prompt is read as a
// CAMPPlus x-vector over a Kaldi filterbank and as a 24 kHz log-mel; the prompt's speech codes and the
// target codes run through an UpsampleConformerEncoder (25 Hz codes to 50 Hz mel frames), and a causal
// conditional flow-matching decoder (a 1-D U-Net of causal residual blocks and transformer blocks)
// denoises the mel over ten Euler steps with classifier-free guidance, the prompt's mel held as the
// conditioning prefix. Every network here runs channels-last (`[batch, frames, channels]`); the
// reference is channels-first, so its convolution weights transpose at load.

// MARK: - Kaldi filterbank (CAMPPlus front end)

/// `torchaudio.compliance.kaldi.fbank` at its defaults with 80 bins and no dither, then the per-utterance
/// mean removed, which is how `CAMPPlus.inference` prepares 16 kHz audio.
enum NFKKaldiFbank {
    static let sampleRate = 16000
    static let frameLength = 400
    static let frameShift = 160
    static let paddedLength = 512
    static let bins = 80

    /// `[frames, bins]` log-mel energies, mean-removed over the frames.
    static func features(_ samples: [Float]) -> MLXArray {
        let frames = samples.count >= frameLength ? 1 + (samples.count - frameLength) / frameShift : 0
        // Povey window: a symmetric Hann raised to 0.85.
        let window = (0 ..< frameLength).map { powf(0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(frameLength - 1)), 0.85) }
        var framed = [Float](repeating: 0, count: frames * paddedLength)
        for frame in 0 ..< frames {
            let start = frame * frameShift
            var mean: Float = 0
            for index in 0 ..< frameLength { mean += samples[start + index] }
            mean /= Float(frameLength)
            var previous = samples[start] - mean                   // replicate padding for the pre-emphasis
            for index in 0 ..< frameLength {
                let current = samples[start + index] - mean
                framed[frame * paddedLength + index] = (current - 0.97 * previous) * window[index]
                previous = current
            }
        }
        let spectrum = rfft(framed.withUnsafeBufferPointer { MLXArray($0, [frames, paddedLength]) }, axis: 1)
        let power = spectrum.realPart() * spectrum.realPart() + spectrum.imaginaryPart() * spectrum.imaginaryPart()
        let energies = power.matmul(melBanks())
        let logMel = log(maximum(energies, MLXArray(Float.ulpOfOne)))
        return logMel - logMel.mean(axis: 0, keepDims: true)
    }

    /// Kaldi's HTK-scale triangular filterbank over the 257 FFT bins (the last bin carries no weight),
    /// 20 Hz to the Nyquist frequency.
    static func melBanks() -> MLXArray {
        func mel(_ hz: Float) -> Float { 1127 * logf(1 + hz / 700) }
        let fftBins = paddedLength / 2
        let binWidth = Float(sampleRate) / Float(paddedLength)
        let low = mel(20), high = mel(Float(sampleRate) / 2)
        let delta = (high - low) / Float(bins + 1)
        var banks = [Float](repeating: 0, count: (fftBins + 1) * bins)
        for bin in 0 ..< bins {
            let left = low + Float(bin) * delta, center = left + delta, right = center + delta
            for k in 0 ..< fftBins {
                let m = mel(binWidth * Float(k))
                let weight = max(0, min((m - left) / (center - left), (right - m) / (right - center)))
                banks[k * bins + bin] = weight
            }
        }
        return banks.withUnsafeBufferPointer { MLXArray($0, [fftBins + 1, bins]) }
    }
}

// MARK: - CAMPPlus x-vector

/// `batchnorm-relu` (or `batchnorm_`, the affine-free variant the embedding head uses).
final class NFKCAMPNonlinear: Module {
    @ModuleInfo(key: "batchnorm") var batchNorm: BatchNorm
    let relu: Bool
    init(channels: Int, affine: Bool = true, relu: Bool = true) {
        self.relu = relu
        _batchNorm.wrappedValue = BatchNorm(featureCount: channels, affine: affine)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = batchNorm(x)
        return relu ? MLXNN.relu(normed) : normed
    }
}

/// The FCM head's residual block over the (frequency, time) plane; a stride applies to frequency only.
final class NFKCAMPResBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm
    @ModuleInfo(key: "shortcut") var shortcut: [Module]

    init(inChannels: Int, channels: Int, stride: Int) {
        _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: channels, kernelSize: 3,
                                     stride: IntOrPair((stride, 1)), padding: 1, bias: false)
        _bn1.wrappedValue = BatchNorm(featureCount: channels)
        _conv2.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1, bias: false)
        _bn2.wrappedValue = BatchNorm(featureCount: channels)
        _shortcut.wrappedValue = stride != 1 || inChannels != channels
            ? [Conv2d(inputChannels: inChannels, outputChannels: channels, kernelSize: 1,
                      stride: IntOrPair((stride, 1)), bias: false), BatchNorm(featureCount: channels)]
            : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        let skip = shortcut.isEmpty ? x : (shortcut[1] as! BatchNorm)((shortcut[0] as! Conv2d)(x))
        return relu(out + skip)
    }
}

/// The 2-D front convolutional module: `[batch, frames, 80]` → `[batch, frames / 1, 32 · 10]` (frequency
/// halved three times, then folded into the channels as `channel · 10 + frequency`).
final class NFKCAMPFrontModule: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "layer1") var layer1: [NFKCAMPResBlock]
    @ModuleInfo(key: "layer2") var layer2: [NFKCAMPResBlock]
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm

    init(channels: Int = 32) {
        _conv1.wrappedValue = Conv2d(inputChannels: 1, outputChannels: channels, kernelSize: 3, padding: 1, bias: false)
        _bn1.wrappedValue = BatchNorm(featureCount: channels)
        _layer1.wrappedValue = [NFKCAMPResBlock(inChannels: channels, channels: channels, stride: 2),
                                NFKCAMPResBlock(inChannels: channels, channels: channels, stride: 1)]
        _layer2.wrappedValue = [NFKCAMPResBlock(inChannels: channels, channels: channels, stride: 2),
                                NFKCAMPResBlock(inChannels: channels, channels: channels, stride: 1)]
        _conv2.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3,
                                     stride: IntOrPair((2, 1)), padding: 1, bias: false)
        _bn2.wrappedValue = BatchNorm(featureCount: channels)
        super.init()
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        // The reference image is (batch, 1, frequency, time); channels-last that is [batch, frequency, time, 1].
        var x = features.transposed(0, 2, 1).expandedDimensions(axis: 3)
        x = relu(bn1(conv1(x)))
        for block in layer1 { x = block(x) }
        for block in layer2 { x = block(x) }
        x = relu(bn2(conv2(x)))                                            // [batch, frequency / 8, time, channels]
        let (batch, frequency, time, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.transposed(0, 2, 3, 1).reshaped([batch, time, channels * frequency])
    }
}

/// A 1-D convolution followed by `batchnorm-relu` (`TDNNLayer`).
final class NFKCAMPTDNNLayer: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d
    @ModuleInfo(key: "nonlinear") var nonlinear: NFKCAMPNonlinear
    init(inChannels: Int, outChannels: Int, kernel: Int, stride: Int, padding: Int) {
        _linear.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernel,
                                      stride: stride, padding: padding, bias: false)
        _nonlinear.wrappedValue = NFKCAMPNonlinear(channels: outChannels)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { nonlinear(linear(x)) }
}

/// Context-aware masking: a dilated local convolution gated by a sigmoid computed from the utterance
/// mean plus a 100-frame segment average.
final class NFKCAMPLayer: Module {
    @ModuleInfo(key: "linear_local") var local: Conv1d
    @ModuleInfo(key: "linear1") var linear1: Conv1d
    @ModuleInfo(key: "linear2") var linear2: Conv1d
    let segmentLength = 100

    init(bottleneck: Int, outChannels: Int, kernel: Int, dilation: Int) {
        _local.wrappedValue = Conv1d(inputChannels: bottleneck, outputChannels: outChannels, kernelSize: kernel,
                                     padding: (kernel - 1) / 2 * dilation, dilation: dilation, bias: false)
        _linear1.wrappedValue = Conv1d(inputChannels: bottleneck, outputChannels: bottleneck / 2, kernelSize: 1)
        _linear2.wrappedValue = Conv1d(inputChannels: bottleneck / 2, outputChannels: outChannels, kernelSize: 1)
        super.init()
    }

    /// `avg_pool1d(kernel 100, stride 100, ceil_mode)` then each segment's mean repeated over its frames:
    /// a trailing partial segment averages only the frames it holds.
    private func segmentPooled(_ x: MLXArray) -> MLXArray {
        let frames = x.dim(1)
        var pieces = [MLXArray]()
        var start = 0
        while start < frames {
            let end = min(start + segmentLength, frames)
            let mean = x[0..., start ..< end].mean(axis: 1, keepDims: true)
            pieces.append(broadcast(mean, to: [x.dim(0), end - start, x.dim(2)]))
            start = end
        }
        return concatenated(pieces, axis: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = local(x)
        let context = x.mean(axis: 1, keepDims: true) + segmentPooled(x)
        let gate = sigmoid(linear2(relu(linear1(context))))
        return y * gate
    }
}

/// One densely connected TDNN layer: bottleneck, then the context-aware masked convolution.
final class NFKCAMPDenseLayer: Module {
    @ModuleInfo(key: "nonlinear1") var nonlinear1: NFKCAMPNonlinear
    @ModuleInfo(key: "linear1") var linear1: Conv1d
    @ModuleInfo(key: "nonlinear2") var nonlinear2: NFKCAMPNonlinear
    @ModuleInfo(key: "cam_layer") var cam: NFKCAMPLayer

    init(inChannels: Int, outChannels: Int, bottleneck: Int, kernel: Int, dilation: Int) {
        _nonlinear1.wrappedValue = NFKCAMPNonlinear(channels: inChannels)
        _linear1.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: bottleneck, kernelSize: 1, bias: false)
        _nonlinear2.wrappedValue = NFKCAMPNonlinear(channels: bottleneck)
        _cam.wrappedValue = NFKCAMPLayer(bottleneck: bottleneck, outChannels: outChannels, kernel: kernel, dilation: dilation)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { cam(nonlinear2(linear1(nonlinear1(x)))) }
}

/// A dense block: every layer reads the concatenation of the block input and all earlier layers.
final class NFKCAMPDenseBlock: Module {
    @ModuleInfo(key: "layers") var layers: [NFKCAMPDenseLayer]
    init(layerCount: Int, inChannels: Int, growth: Int, bottleneck: Int, kernel: Int, dilation: Int) {
        _layers.wrappedValue = (0 ..< layerCount).map {
            NFKCAMPDenseLayer(inChannels: inChannels + $0 * growth, outChannels: growth, bottleneck: bottleneck,
                              kernel: kernel, dilation: dilation)
        }
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers { x = concatenated([x, layer(x)], axis: -1) }
        return x
    }
}

/// `batchnorm-relu` then a 1×1 convolution (`TransitLayer`).
final class NFKCAMPTransitLayer: Module {
    @ModuleInfo(key: "nonlinear") var nonlinear: NFKCAMPNonlinear
    @ModuleInfo(key: "linear") var linear: Conv1d
    init(inChannels: Int, outChannels: Int) {
        _nonlinear.wrappedValue = NFKCAMPNonlinear(channels: inChannels)
        _linear.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(nonlinear(x)) }
}

/// The embedding head: a 1×1 convolution then an affine-free batch norm (`DenseLayer`, `batchnorm_`).
final class NFKCAMPEmbeddingLayer: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d
    @ModuleInfo(key: "nonlinear") var nonlinear: NFKCAMPNonlinear
    init(inChannels: Int, outChannels: Int) {
        _linear.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
        _nonlinear.wrappedValue = NFKCAMPNonlinear(channels: outChannels, affine: false, relu: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { nonlinear(linear(x)) }
}

/// CAMPPlus (`speaker_encoder`): the FCM head, a TDNN, three dense blocks with transitions, statistics
/// pooling, and a 192-wide embedding.
public final class NFKCAMPPlusNet: Module {
    @ModuleInfo(key: "head") var head: NFKCAMPFrontModule
    @ModuleInfo(key: "tdnn") var tdnn: NFKCAMPTDNNLayer
    @ModuleInfo(key: "block1") var block1: NFKCAMPDenseBlock
    @ModuleInfo(key: "transit1") var transit1: NFKCAMPTransitLayer
    @ModuleInfo(key: "block2") var block2: NFKCAMPDenseBlock
    @ModuleInfo(key: "transit2") var transit2: NFKCAMPTransitLayer
    @ModuleInfo(key: "block3") var block3: NFKCAMPDenseBlock
    @ModuleInfo(key: "transit3") var transit3: NFKCAMPTransitLayer
    @ModuleInfo(key: "out_nonlinear") var outNonlinear: NFKCAMPNonlinear
    @ModuleInfo(key: "dense") var dense: NFKCAMPEmbeddingLayer

    public init(embeddingSize: Int = 192, growth: Int = 32, initialChannels: Int = 128) {
        _head.wrappedValue = NFKCAMPFrontModule()
        var channels = 32 * (80 / 8)
        _tdnn.wrappedValue = NFKCAMPTDNNLayer(inChannels: channels, outChannels: initialChannels, kernel: 5, stride: 2, padding: 2)
        channels = initialChannels
        let blocks = [(12, 3, 1), (24, 3, 2), (16, 3, 2)]
        var dense = [NFKCAMPDenseBlock](), transits = [NFKCAMPTransitLayer]()
        for (layers, kernel, dilation) in blocks {
            dense.append(NFKCAMPDenseBlock(layerCount: layers, inChannels: channels, growth: growth,
                                           bottleneck: 4 * growth, kernel: kernel, dilation: dilation))
            channels += layers * growth
            transits.append(NFKCAMPTransitLayer(inChannels: channels, outChannels: channels / 2))
            channels /= 2
        }
        _block1.wrappedValue = dense[0]; _transit1.wrappedValue = transits[0]
        _block2.wrappedValue = dense[1]; _transit2.wrappedValue = transits[1]
        _block3.wrappedValue = dense[2]; _transit3.wrappedValue = transits[2]
        _outNonlinear.wrappedValue = NFKCAMPNonlinear(channels: channels)
        _dense.wrappedValue = NFKCAMPEmbeddingLayer(inChannels: channels * 2, outChannels: embeddingSize)
        super.init()
    }

    /// The x-vector `[batch, 192]` of mean-removed Kaldi features `[batch, frames, 80]`.
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = head(features)
        x = tdnn(x)
        x = transit1(block1(x))
        x = transit2(block2(x))
        x = transit3(block3(x))
        x = outNonlinear(x)
        // Statistics pooling: the mean and the unbiased standard deviation over the frames.
        let mean = x.mean(axis: 1)
        let variance = (x - mean.expandedDimensions(axis: 1)).square().sum(axis: 1) / Float(max(x.dim(1) - 1, 1))
        let stats = concatenated([mean, sqrt(variance)], axis: -1).expandedDimensions(axis: 1)
        return dense(stats)[0..., 0, 0...]
    }

    /// The x-vector of a 16 kHz waveform.
    public func embedding(samples: [Float]) -> MLXArray {
        self(NFKKaldiFbank.features(samples).expandedDimensions(axis: 0))[0]
    }
}

// MARK: - 24 kHz prompt mel

/// Matcha-TTS's `mel_spectrogram` at CosyVoice's settings: 1920-point transform at hop 480 over 24 kHz
/// audio reflect-padded by 720 on both sides, magnitude `sqrt(power + 1e-9)`, an 80-band Slaney bank
/// to 8 kHz, and `log(max(x, 1e-5))`. Returns `[frames, 80]`.
public enum NFKChatterboxPromptMel {
    public static let sampleRate = 24000
    static let fftSize = 1920
    static let hop = 480

    public static func features(_ samples: [Float]) -> MLXArray {
        let pad = (fftSize - hop) / 2
        var padded = [Float]()
        padded.reserveCapacity(samples.count + 2 * pad)
        for index in stride(from: pad, through: 1, by: -1) { padded.append(samples[index]) }
        padded.append(contentsOf: samples)
        for index in 0 ..< pad { padded.append(samples[samples.count - 2 - index]) }
        let window = (0 ..< fftSize).map { 0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(fftSize)) }
        let frames = 1 + (padded.count - fftSize) / hop
        var framed = [Float](repeating: 0, count: frames * fftSize)
        for frame in 0 ..< frames {
            for index in 0 ..< fftSize { framed[frame * fftSize + index] = padded[frame * hop + index] * window[index] }
        }
        let spectrum = rfft(framed.withUnsafeBufferPointer { MLXArray($0, [frames, fftSize]) }, axis: 1)
        let magnitude = sqrt(spectrum.realPart() * spectrum.realPart() + spectrum.imaginaryPart() * spectrum.imaginaryPart() + 1e-9)
        let filters = NFKMLXMel.melFilters(sampleRate: sampleRate, bins: fftSize / 2 + 1, nMels: 80, fMaximum: 8000)
        return log(maximum(magnitude.matmul(filters), MLXArray(Float(1e-5))))
    }
}

// MARK: - Upsample conformer encoder

/// The espnet relative positional encoding: sinusoids for offsets `T-1 … -(T-1)`, `[1, 2T - 1, d]`, and
/// the `sqrt(d)` input scale it applies.
enum NFKS3RelativePositions {
    static func table(length: Int, dimensions: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: (2 * length - 1) * dimensions)
        for row in 0 ..< (2 * length - 1) {
            // Row 0 is offset T-1, row T-1 is offset 0, row 2T-2 is offset -(T-1).
            let position = Float(length - 1 - row)
            for pair in 0 ..< dimensions / 2 {
                let divisor = expf(Float(2 * pair) * -(logf(10000) / Float(dimensions)))
                values[row * dimensions + 2 * pair] = sinf(position * divisor)
                values[row * dimensions + 2 * pair + 1] = cosf(position * divisor)
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, 2 * length - 1, dimensions]) }
    }
}

/// `RelPositionMultiHeadedAttention`: content and position scores with the Transformer-XL shift.
final class NFKS3RelativeAttention: Module {
    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear
    @ParameterInfo(key: "pos_bias_u") var biasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var biasV: MLXArray
    let heads: Int

    init(width: Int, heads: Int) {
        self.heads = heads
        _linearQ.wrappedValue = Linear(width, width)
        _linearK.wrappedValue = Linear(width, width)
        _linearV.wrappedValue = Linear(width, width)
        _linearOut.wrappedValue = Linear(width, width)
        _linearPos.wrappedValue = Linear(width, width, bias: false)
        _biasU.wrappedValue = MLXArray.zeros([heads, width / heads])
        _biasV.wrappedValue = MLXArray.zeros([heads, width / heads])
        super.init()
    }

    /// `[b, h, T, 2T-1]` → `[b, h, T, T]`: the appendix-B shift that aligns each query's row of
    /// relative scores with the key positions.
    private func relativeShift(_ x: MLXArray) -> MLXArray {
        let (b, h, t1, t2) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let padded = concatenated([MLXArray.zeros([b, h, t1, 1]), x], axis: -1).reshaped([b, h, t2 + 1, t1])
        return padded[0..., 0..., 1...].reshaped([b, h, t1, t2])[0..., 0..., 0..., 0 ..< (t2 / 2 + 1)]
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = width / heads
        let q = linearQ(x).reshaped([batch, length, heads, headDim])                 // [b, T, h, d]
        let k = linearK(x).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        let v = linearV(x).reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        let p = linearPos(positions).reshaped([1, -1, heads, headDim]).transposed(0, 2, 1, 3)  // [1, h, 2T-1, d]
        let qU = (q + biasU).transposed(0, 2, 1, 3)
        let qV = (q + biasV).transposed(0, 2, 1, 3)
        let content = qU.matmul(k.transposed(0, 1, 3, 2))
        var position = qV.matmul(p.transposed(0, 1, 3, 2))
        if position.dim(3) != content.dim(3) { position = relativeShift(position) }
        let scores = (content + position) / sqrt(Float(headDim))
        let attended = softmax(scores, axis: -1).matmul(v)
        return linearOut(attended.transposed(0, 2, 1, 3).reshaped([batch, length, width]))
    }
}

/// `PositionwiseFeedForward` with the swish activation.
final class NFKS3FeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear
    init(width: Int, hidden: Int) {
        _w1.wrappedValue = Linear(width, hidden)
        _w2.wrappedValue = Linear(hidden, width)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x))) }
}

/// A pre-norm conformer layer with no macaron branch and no convolution module: attention then the
/// feed-forward, both residual, norms at epsilon 1e-12.
final class NFKS3ConformerLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKS3RelativeAttention
    @ModuleInfo(key: "feed_forward") var feedForward: NFKS3FeedForward
    @ModuleInfo(key: "norm_ff") var normFF: LayerNorm
    @ModuleInfo(key: "norm_mha") var normMHA: LayerNorm

    init(width: Int, heads: Int, hidden: Int) {
        _attention.wrappedValue = NFKS3RelativeAttention(width: width, heads: heads)
        _feedForward.wrappedValue = NFKS3FeedForward(width: width, hidden: hidden)
        _normFF.wrappedValue = LayerNorm(dimensions: width, eps: 1e-12)
        _normMHA.wrappedValue = LayerNorm(dimensions: width, eps: 1e-12)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        var x = x + attention(normMHA(x), positions: positions)
        x = x + feedForward(normFF(x))
        return x
    }
}

/// `LinearNoSubsampling`: a linear projection and a LayerNorm (the reference's `out` Sequential, whose
/// dropout slot carries nothing), then the positional scale.
final class NFKS3InputEmbed: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    init(width: Int) {
        _linear.wrappedValue = Linear(width, width)
        _norm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        super.init()
    }
    /// Returns the scaled input and its relative position table.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let out = norm(linear(x)) * sqrt(Float(x.dim(2)))
        return (out, NFKS3RelativePositions.table(length: x.dim(1), dimensions: x.dim(2)))
    }
}

/// `PreLookaheadLayer`: a convolution that reads three FUTURE frames, then a causal one, residual.
final class NFKS3PreLookahead: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    let lookahead: Int
    init(channels: Int, lookahead: Int = 3) {
        self.lookahead = lookahead
        _conv1.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: lookahead + 1)
        _conv2.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 3)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = leakyRelu(conv1(padded(x, widths: [.init((0, 0)), .init((0, lookahead)), .init((0, 0))])), negativeSlope: 0.01)
        out = conv2(padded(out, widths: [.init((0, 0)), .init((2, 0)), .init((0, 0))]))
        return out + x
    }
}

/// `Upsample1D`: nearest ×2 in time, four zeros on the left, a 5-wide convolution.
final class NFKS3Upsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let stride: Int
    init(channels: Int, stride: Int = 2) {
        self.stride = stride
        _conv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: stride * 2 + 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let repeated = repeated(x, count: stride, axis: 1)
        return conv(padded(repeated, widths: [.init((0, 0)), .init((stride * 2, 0)), .init((0, 0))]))
    }
}

/// `UpsampleConformerEncoder`: six conformer layers at the code rate, a ×2 upsample, four more at the
/// mel rate, and a final LayerNorm. `[batch, T, 512]` → `[batch, 2T, 512]`.
public final class NFKS3FlowEncoderNet: Module {
    @ModuleInfo(key: "embed") var embed: NFKS3InputEmbed
    @ModuleInfo(key: "pre_lookahead_layer") var preLookahead: NFKS3PreLookahead
    @ModuleInfo(key: "encoders") var encoders: [NFKS3ConformerLayer]
    @ModuleInfo(key: "up_layer") var upLayer: NFKS3Upsample
    @ModuleInfo(key: "up_embed") var upEmbed: NFKS3InputEmbed
    @ModuleInfo(key: "up_encoders") var upEncoders: [NFKS3ConformerLayer]
    @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm

    init(width: Int = 512, heads: Int = 8, hidden: Int = 2048, layers: Int = 6, upLayers: Int = 4) {
        _embed.wrappedValue = NFKS3InputEmbed(width: width)
        _preLookahead.wrappedValue = NFKS3PreLookahead(channels: width)
        _encoders.wrappedValue = (0 ..< layers).map { _ in NFKS3ConformerLayer(width: width, heads: heads, hidden: hidden) }
        _upLayer.wrappedValue = NFKS3Upsample(channels: width)
        _upEmbed.wrappedValue = NFKS3InputEmbed(width: width)
        _upEncoders.wrappedValue = (0 ..< upLayers).map { _ in NFKS3ConformerLayer(width: width, heads: heads, hidden: hidden) }
        _afterNorm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var (h, positions) = embed(x)
        h = preLookahead(h)
        for layer in encoders { h = layer(h, positions: positions) }
        h = upLayer(h)
        (h, positions) = upEmbed(h)
        for layer in upEncoders { h = layer(h, positions: positions) }
        return afterNorm(h)
    }
}

// MARK: - Conditional flow-matching estimator

/// A convolution whose kernel reads only the current and earlier frames (`CausalConv1d`).
final class NFKS3CausalConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let kernel: Int
    init(inChannels: Int, outChannels: Int, kernel: Int) {
        self.kernel = kernel
        _conv.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernel)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(padded(x, widths: [.init((0, 0)), .init((kernel - 1, 0)), .init((0, 0))]))
    }
}

/// `CausalBlock1D`: causal 3-wide convolution, LayerNorm over the channels, Mish.
final class NFKS3CausalBlock: Module {
    @ModuleInfo(key: "conv") var conv: NFKS3CausalConv
    @ModuleInfo(key: "norm") var norm: LayerNorm
    init(inChannels: Int, outChannels: Int) {
        _conv.wrappedValue = NFKS3CausalConv(inChannels: inChannels, outChannels: outChannels, kernel: 3)
        _norm.wrappedValue = LayerNorm(dimensions: outChannels)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { mish(norm(conv(x))) }
}

/// `CausalResnetBlock1D`: two causal blocks with the timestep embedding added between, plus a 1×1 skip.
final class NFKS3ResnetBlock: Module {
    @ModuleInfo(key: "block1") var block1: NFKS3CausalBlock
    @ModuleInfo(key: "block2") var block2: NFKS3CausalBlock
    @ModuleInfo(key: "mlp") var timeProjection: Linear
    @ModuleInfo(key: "res_conv") var skip: Conv1d
    init(inChannels: Int, outChannels: Int, timeWidth: Int) {
        _block1.wrappedValue = NFKS3CausalBlock(inChannels: inChannels, outChannels: outChannels)
        _block2.wrappedValue = NFKS3CausalBlock(inChannels: outChannels, outChannels: outChannels)
        _timeProjection.wrappedValue = Linear(timeWidth, outChannels)
        _skip.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, time: MLXArray) -> MLXArray {
        var h = block1(x)
        h = h + timeProjection(mish(time)).expandedDimensions(axis: 1)
        h = block2(h)
        return h + skip(x)
    }
}

/// diffusers' `Attention` as the estimator uses it: bias-free q/k/v to 8 × 64, a biased output projection.
final class NFKS3EstimatorAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    let heads: Int
    init(width: Int, heads: Int, headDim: Int) {
        self.heads = heads
        _toQ.wrappedValue = Linear(width, heads * headDim, bias: false)
        _toK.wrappedValue = Linear(width, heads * headDim, bias: false)
        _toV.wrappedValue = Linear(width, heads * headDim, bias: false)
        _toOut.wrappedValue = Linear(heads * headDim, width)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([batch, length, heads, -1]).transposed(0, 2, 1, 3) }
        let q = split(toQ(x)), k = split(toK(x)), v = split(toV(x))
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                         scale: 1 / sqrt(Float(q.dim(3))), mask: nil)
        return toOut(attended.transposed(0, 2, 1, 3).reshaped([batch, length, -1]))
    }
}

/// `BasicTransformerBlock` with plain LayerNorms and a GELU feed-forward (`ff.net.0.proj`, `ff.net.2`).
final class NFKS3TransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn1") var attention: NFKS3EstimatorAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "ff_proj") var ffProj: Linear
    @ModuleInfo(key: "ff_out") var ffOut: Linear
    init(width: Int, heads: Int, headDim: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: width)
        _attention.wrappedValue = NFKS3EstimatorAttention(width: width, heads: heads, headDim: headDim)
        _norm3.wrappedValue = LayerNorm(dimensions: width)
        _ffProj.wrappedValue = Linear(width, width * 4)
        _ffOut.wrappedValue = Linear(width * 4, width)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x + attention(norm1(x))
        x = x + ffOut(gelu(ffProj(norm3(x))))
        return x
    }
}

/// One U-Net stage: a residual block, four transformer blocks, and (down and up stages) a causal
/// convolution standing in for the resample, since the released configuration has one level.
final class NFKS3EstimatorStage: Module {
    @ModuleInfo(key: "resnet") var resnet: NFKS3ResnetBlock
    @ModuleInfo(key: "transformers") var transformers: [NFKS3TransformerBlock]
    @ModuleInfo(key: "resample") var resample: NFKS3CausalConv?
    init(inChannels: Int, outChannels: Int, timeWidth: Int, blocks: Int, heads: Int, headDim: Int, resample: Bool) {
        _resnet.wrappedValue = NFKS3ResnetBlock(inChannels: inChannels, outChannels: outChannels, timeWidth: timeWidth)
        _transformers.wrappedValue = (0 ..< blocks).map { _ in NFKS3TransformerBlock(width: outChannels, heads: heads, headDim: headDim) }
        _resample.wrappedValue = resample ? NFKS3CausalConv(inChannels: outChannels, outChannels: outChannels, kernel: 3) : nil
        super.init()
    }
    func callAsFunction(_ x: MLXArray, time: MLXArray) -> MLXArray {
        var h = resnet(x, time: time)
        for block in transformers { h = block(h) }
        return h
    }
}

/// `ConditionalDecoder`: the velocity estimator. Input `[batch, T, 80]` noisy mel; conditioning `mu`
/// `[batch, T, 80]`, speaker `[batch, 80]`, `cond` `[batch, T, 80]`; timestep `[batch]`.
public final class NFKS3EstimatorNet: Module {
    @ModuleInfo(key: "time_mlp_1") var timeLinear1: Linear
    @ModuleInfo(key: "time_mlp_2") var timeLinear2: Linear
    @ModuleInfo(key: "down_blocks") var downBlocks: [NFKS3EstimatorStage]
    @ModuleInfo(key: "mid_blocks") var midBlocks: [NFKS3EstimatorStage]
    @ModuleInfo(key: "up_blocks") var upBlocks: [NFKS3EstimatorStage]
    @ModuleInfo(key: "final_block") var finalBlock: NFKS3CausalBlock
    @ModuleInfo(key: "final_proj") var finalProjection: Conv1d
    let inChannels: Int

    init(inChannels: Int = 320, outChannels: Int = 80, channels: Int = 256, blocks: Int = 4, midBlocks: Int = 12,
         heads: Int = 8, headDim: Int = 64) {
        self.inChannels = inChannels
        let timeWidth = channels * 4
        _timeLinear1.wrappedValue = Linear(inChannels, timeWidth)
        _timeLinear2.wrappedValue = Linear(timeWidth, timeWidth)
        _downBlocks.wrappedValue = [NFKS3EstimatorStage(inChannels: inChannels, outChannels: channels, timeWidth: timeWidth,
                                                        blocks: blocks, heads: heads, headDim: headDim, resample: true)]
        _midBlocks.wrappedValue = (0 ..< midBlocks).map { _ in
            NFKS3EstimatorStage(inChannels: channels, outChannels: channels, timeWidth: timeWidth,
                                blocks: blocks, heads: heads, headDim: headDim, resample: false)
        }
        _upBlocks.wrappedValue = [NFKS3EstimatorStage(inChannels: channels * 2, outChannels: channels, timeWidth: timeWidth,
                                                      blocks: blocks, heads: heads, headDim: headDim, resample: true)]
        _finalBlock.wrappedValue = NFKS3CausalBlock(inChannels: channels, outChannels: channels)
        _finalProjection.wrappedValue = Conv1d(inputChannels: channels, outputChannels: outChannels, kernelSize: 1)
        super.init()
    }

    /// `SinusoidalPosEmb` at scale 1000 then the two-layer SiLU MLP: `[batch]` → `[batch, 4 · channels]`.
    func timeEmbedding(_ t: MLXArray) -> MLXArray {
        let half = inChannels / 2
        let frequencies = exp(MLXArray(0 ..< Int32(half)).asType(.float32) * -(logf(10000) / Float(half - 1)))
        let angles = 1000 * t.expandedDimensions(axis: 1) * frequencies.expandedDimensions(axis: 0)
        let embedding = concatenated([sin(angles), cos(angles)], axis: -1)
        return timeLinear2(silu(timeLinear1(embedding)))
    }

    public func callAsFunction(_ x: MLXArray, mu: MLXArray, speaker: MLXArray, condition: MLXArray, t: MLXArray) -> MLXArray {
        let time = timeEmbedding(t)
        let speakerFrames = broadcast(speaker.expandedDimensions(axis: 1), to: [x.dim(0), x.dim(1), speaker.dim(1)])
        var h = concatenated([x, mu, speakerFrames, condition], axis: -1)
        var skips = [MLXArray]()
        for stage in downBlocks {
            h = stage(h, time: time)
            skips.append(h)
            h = stage.resample!(h)
        }
        for stage in midBlocks { h = stage(h, time: time) }
        for stage in upBlocks {
            let skip = skips.removeLast()
            h = concatenated([h[0..., 0 ..< skip.dim(1)], skip], axis: -1)
            h = stage(h, time: time)
            h = stage.resample!(h)
        }
        return finalProjection(finalBlock(h))
    }
}

// MARK: - The flow

/// The flow-matching sampler settings (`CFM_PARAMS` and `inference`'s defaults).
public struct NFKS3FlowOptions: Sendable {
    public var steps: Int = 10
    public var guidance: Float = 0.7
    public init() {}
}

/// `CausalMaskedDiffWithXvec`: codes and a prompt in, a mel out.
public final class NFKS3FlowNet: Module {
    @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") var speakerProjection: Linear
    @ModuleInfo(key: "encoder") var encoder: NFKS3FlowEncoderNet
    @ModuleInfo(key: "encoder_proj") var encoderProjection: Linear
    @ModuleInfo(key: "estimator") var estimator: NFKS3EstimatorNet
    public let melChannels = 80

    init(vocabulary: Int = 6561, width: Int = 512, xVectorSize: Int = 192) {
        _inputEmbedding.wrappedValue = Embedding(embeddingCount: vocabulary, dimensions: width)
        _speakerProjection.wrappedValue = Linear(xVectorSize, melChannels)
        _encoder.wrappedValue = NFKS3FlowEncoderNet(width: width)
        _encoderProjection.wrappedValue = Linear(width, melChannels)
        _estimator.wrappedValue = NFKS3EstimatorNet()
        super.init()
    }

    /// The x-vector L2-normalized and projected to the mel width, `[1, 80]`.
    public func speakerEmbedding(xVector: MLXArray) -> MLXArray {
        let flat = xVector.reshaped([1, -1])
        return speakerProjection(flat / sqrt((flat * flat).sum(axis: -1, keepDims: true)))
    }

    /// The token embeddings `[1, T, 512]` for the prompt and target codes together.
    public func tokenEmbedding(_ tokens: [Int]) -> MLXArray {
        inputEmbedding(MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count]))
    }

    /// `mu`, the encoder output projected to the mel width, `[1, 2T, 80]`.
    public func conditioning(tokenEmbedding: MLXArray) -> MLXArray {
        encoderProjection(encoder(tokenEmbedding))
    }

    /// The cosine timestep schedule, `steps + 1` values from 0 to 1.
    public static func timeSpan(steps: Int) -> [Float] {
        (0 ... steps).map { 1 - cosf(Float($0) / Float(steps) * 0.5 * .pi) }
    }

    /// Runs the guided Euler solver from `noise` `[1, T, 80]`: the conditional and unconditional
    /// velocities are one batch of two, the unconditional row with zero `mu`, speaker, and condition.
    public func solve(noise: MLXArray, mu: MLXArray, speaker: MLXArray, condition: MLXArray,
                      options: NFKS3FlowOptions = NFKS3FlowOptions(),
                      onStep: ((Int, MLXArray) -> Void)? = nil) -> MLXArray {
        var x = noise
        let span = NFKS3FlowNet.timeSpan(steps: options.steps)
        let muPair = concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
        let speakerPair = concatenated([speaker, MLXArray.zeros(like: speaker)], axis: 0)
        let conditionPair = concatenated([condition, MLXArray.zeros(like: condition)], axis: 0)
        for step in 0 ..< options.steps {
            let t = MLXArray([span[step], span[step]])
            let velocity = estimator(concatenated([x, x], axis: 0), mu: muPair, speaker: speakerPair,
                                     condition: conditionPair, t: t)
            onStep?(step, velocity)
            let guided = (1 + options.guidance) * velocity[0 ..< 1] - options.guidance * velocity[1 ..< 2]
            x = x + (span[step + 1] - span[step]) * guided
            eval(x)
        }
        return x
    }

    /// The mel `[frames, 80]` for `tokens` given the prompt: the prompt's mel occupies the first frames
    /// of the conditioning and is dropped from the result. `noise` is `[2(prompt + tokens), 80]`, drawn
    /// fresh when nil.
    public func mel(tokens: [Int], promptTokens: [Int], promptMel: MLXArray, xVector: MLXArray,
                    noise: MLXArray? = nil, options: NFKS3FlowOptions = NFKS3FlowOptions()) -> MLXArray {
        let speaker = speakerEmbedding(xVector: xVector)
        let mu = conditioning(tokenEmbedding: tokenEmbedding(promptTokens + tokens))
        let total = mu.dim(1)
        let promptFrames = promptMel.dim(0)
        let condition = concatenated([promptMel, MLXArray.zeros([total - promptFrames, melChannels])], axis: 0)
            .expandedDimensions(axis: 0)
        let z = (noise ?? MLXRandom.normal([total, melChannels])).expandedDimensions(axis: 0)
        let output = solve(noise: z, mu: mu, speaker: speaker, condition: condition, options: options)
        return output[0, promptFrames...]
    }
}
