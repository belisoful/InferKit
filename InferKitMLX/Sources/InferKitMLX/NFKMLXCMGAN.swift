//
//  NFKMLXCMGAN.swift
//  InferKitMLX
//
//  CMGAN (ruizhecao96/CMGAN, MIT): a conformer-based metric GAN whose generator, `TSCNet`, denoises a
//  power-compressed complex spectrogram through a dense encoder, four two-stage (time, frequency)
//  conformer blocks, a magnitude-mask decoder, and a complex-residual decoder. Only the generator
//  runs at inference; the discriminator is a training device.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The released CMGAN geometry (VCTK-DEMAND, 16 kHz).
public struct NFKMLXCMGANConfiguration: Sendable {
    public var sampleRate = 16_000
    public var fftSize = 400
    public var hopSize = 100
    public var channels = 64
    public var blockCount = 4
    public var heads = 4
    public var feedForwardMultiplier = 4
    public var convolutionKernel = 31
    public var maximumRelativePosition = 512
    /// `mag ** 0.3` before the network and `** (1 / 0.3)` after it (`power_compress`).
    public var compressFactor: Float = 0.3
    public var bins: Int { fftSize / 2 + 1 }
    public var headDimension: Int { channels / heads }
    public init() {}
}

/// A parameter-free `nn.Sequential` entry (an activation, a rearrange, a dropout) that occupies its
/// index so the checkpoint's numeric keys land on the parameterized entries around it.
final class NFKCMGANMarker: Module {}

// MARK: - Dense blocks

/// The reference `DilatedDenseNet`: `depth` dilated convolutions (`conv{i}` / `norm{i}` / `prelu{i}`)
/// each reading every previous output concatenated with the block input. The layers are the MP-SENet
/// dense convolution (the same `(2, 3)` kernel, past-only time padding, and affine InstanceNorm).
final class NFKCMGANDenseNet: Module {
    @ModuleInfo(key: "layers") var layers: [NFKMPSEDenseConv]

    init(channels: Int, depth: Int = 4) {
        _layers.wrappedValue = (0 ..< depth).map {
            NFKMPSEDenseConv(inChannels: channels * ($0 + 1), denseChannel: channels, dilation: 1 << $0)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var skip = x
        var out = x
        for layer in layers {
            out = layer(skip)
            skip = concatenated([out, skip], axis: 3)
        }
        return out
    }
}

/// `conv_1` (1×1 to the width) → the dilated dense net → `conv_2` (a `(1, 3)` stride-`(1, 2)`
/// convolution halving the frequency axis). Both convolutions are the reference `nn.Sequential`
/// (Conv2d, affine InstanceNorm, PReLU), held as module arrays so the indices match.
final class NFKCMGANDenseEncoder: Module {
    @ModuleInfo(key: "conv_1") var conv1: [Module]
    @ModuleInfo(key: "dilated_dense") var dense: NFKCMGANDenseNet
    @ModuleInfo(key: "conv_2") var conv2: [Module]

    init(inputChannels: Int, channels: Int) {
        _conv1.wrappedValue = [Conv2d(inputChannels: inputChannels, outputChannels: channels, kernelSize: 1),
                               InstanceNorm(dimensions: channels, affine: true), PReLU(count: channels)]
        _dense.wrappedValue = NFKCMGANDenseNet(channels: channels)
        _conv2.wrappedValue = [Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: IntOrPair((1, 3)),
                                      stride: IntOrPair((1, 2)), padding: IntOrPair((0, 1))),
                               InstanceNorm(dimensions: channels, affine: true), PReLU(count: channels)]
    }

    private static func run(_ stack: [Module], _ x: MLXArray) -> MLXArray {
        let conv = stack[0] as! Conv2d, norm = stack[1] as! InstanceNorm, prelu = stack[2] as! PReLU
        return prelu(norm(conv(x)))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        Self.run(conv2, dense(Self.run(conv1, x)))
    }
}

// MARK: - Conformer

/// `Linear → Swish → Dropout → Linear → Dropout`, indices 0 … 4.
final class NFKCMGANFeedForward: Module {
    @ModuleInfo(key: "net") var net: [Module]

    init(dim: Int, multiplier: Int) {
        _net.wrappedValue = [Linear(dim, dim * multiplier), NFKCMGANMarker(), NFKCMGANMarker(),
                             Linear(dim * multiplier, dim), NFKCMGANMarker()]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let first = net[0] as! Linear, second = net[3] as! Linear
        return second(silu(first(x)))
    }
}

/// `PreNorm(dim, FeedForward)`: the norm, then the wrapped function.
final class NFKCMGANPreNormFeedForward: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "fn") var fn: NFKCMGANFeedForward

    init(dim: Int, multiplier: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: dim)
        _fn.wrappedValue = NFKCMGANFeedForward(dim: dim, multiplier: multiplier)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fn(norm(x)) }
}

/// `Scale(0.5, PreNorm(...))`: the half-weighted macaron feed-forward.
final class NFKCMGANScaledFeedForward: Module {
    @ModuleInfo(key: "fn") var fn: NFKCMGANPreNormFeedForward

    init(dim: Int, multiplier: Int) {
        _fn.wrappedValue = NFKCMGANPreNormFeedForward(dim: dim, multiplier: multiplier)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { 0.5 * fn(x) }
}

/// lucidrains' conformer attention: separate bias-free `to_q` / fused `to_kv`, a biased `to_out`, and
/// Shaw's relative position embedding, a learned `[2·max + 1, headDim]` table indexed by the clamped
/// query-key distance whose dot product with the query joins the content score.
final class NFKCMGANAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_kv") var toKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "rel_pos_emb") var relativePosition: Embedding
    let heads: Int
    let headDimension: Int
    let maximumPosition: Int

    init(dim: Int, heads: Int, headDimension: Int, maximumPosition: Int) {
        self.heads = heads
        self.headDimension = headDimension
        self.maximumPosition = maximumPosition
        let inner = heads * headDimension
        _toQ.wrappedValue = Linear(dim, inner, bias: false)
        _toKV.wrappedValue = Linear(dim, inner * 2, bias: false)
        _toOut.wrappedValue = Linear(inner, dim)
        _relativePosition.wrappedValue = Embedding(embeddingCount: 2 * maximumPosition + 1, dimensions: headDimension)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        let scale = Float(headDimension).squareRoot()
        let q = toQ(x).reshaped([b, n, heads, headDimension]).transposed(0, 2, 1, 3)
        let kv = toKV(x).reshaped([b, n, 2, heads, headDimension])
        let k = kv[0..., 0..., 0].transposed(0, 2, 1, 3)
        let v = kv[0..., 0..., 1].transposed(0, 2, 1, 3)
        var dots = matmul(q, k.transposed(0, 1, 3, 2)) / scale                 // [b, h, n, n]
        // Shaw's relative positions: distance clamped to ±max, offset by max, gathered to [n, n, d].
        let positions = MLXArray(0 ..< Int32(n))
        let distance = clip(positions.reshaped([n, 1]) - positions.reshaped([1, n]),
                            min: Int32(-maximumPosition), max: Int32(maximumPosition)) + Int32(maximumPosition)
        let relative = relativePosition(distance)                               // [n, n, d]
        // pos[b, h, i, j] = Σ_d q[b, h, i, d] · relative[i, j, d], as a matmul batched over i.
        let qByPosition = q.transposed(2, 0, 1, 3).reshaped([n, b * heads, headDimension])
        let positional = matmul(qByPosition, relative.transposed(0, 2, 1))     // [n, b·h, n]
        dots = dots + positional.reshaped([n, b, heads, n]).transposed(1, 2, 0, 3) / scale
        let out = matmul(softmax(dots, axis: -1), v)                            // [b, h, n, d]
        return toOut(out.transposed(0, 2, 1, 3).reshaped([b, n, heads * headDimension]))
    }
}

/// `PreNorm(dim, Attention)`.
final class NFKCMGANPreNormAttention: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "fn") var fn: NFKCMGANAttention

    init(dim: Int, heads: Int, headDimension: Int, maximumPosition: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: dim)
        _fn.wrappedValue = NFKCMGANAttention(dim: dim, heads: heads, headDimension: headDimension,
                                             maximumPosition: maximumPosition)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fn(norm(x)) }
}

/// `DepthWiseConv1d`: a depthwise convolution under a `conv` key, padded `(k/2, k/2)` (same).
final class NFKCMGANDepthwiseConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, kernel: Int) {
        _conv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel,
                                    padding: kernel / 2, groups: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// The conformer convolution module, the reference `nn.Sequential`: LayerNorm (0), a rearrange (1), a
/// 1×1 convolution to `2 · expansion · dim` (2), a GLU (3), the depthwise convolution (4), BatchNorm (5),
/// Swish (6), a 1×1 convolution back to `dim` (7), a rearrange (8), and dropout (9). MLX's channels-last
/// layout needs no rearrange, so those indices are markers.
final class NFKCMGANConvModule: Module {
    @ModuleInfo(key: "net") var net: [Module]

    init(dim: Int, kernel: Int, expansion: Int = 2) {
        let inner = dim * expansion
        _net.wrappedValue = [LayerNorm(dimensions: dim), NFKCMGANMarker(),
                             Conv1d(inputChannels: dim, outputChannels: inner * 2, kernelSize: 1), NFKCMGANMarker(),
                             NFKCMGANDepthwiseConv(channels: inner, kernel: kernel),
                             BatchNorm(featureCount: inner), NFKCMGANMarker(),
                             Conv1d(inputChannels: inner, outputChannels: dim, kernelSize: 1),
                             NFKCMGANMarker(), NFKCMGANMarker()]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let norm = net[0] as! LayerNorm, pointwiseIn = net[2] as! Conv1d, depthwise = net[4] as! NFKCMGANDepthwiseConv
        let batchNorm = net[5] as! BatchNorm, pointwiseOut = net[7] as! Conv1d
        let gated = pointwiseIn(norm(x))
        let half = gated.dim(2) / 2
        let glu = gated[0..., 0..., ..<half] * sigmoid(gated[0..., 0..., half...])
        return pointwiseOut(silu(batchNorm(depthwise(glu))))
    }
}

/// One conformer block: `ff1` (half) → attention → convolution → `ff2` (half) → `post_norm`, each
/// stage residual.
final class NFKCMGANConformerBlock: Module {
    @ModuleInfo(key: "ff1") var ff1: NFKCMGANScaledFeedForward
    @ModuleInfo(key: "attn") var attention: NFKCMGANPreNormAttention
    @ModuleInfo(key: "conv") var conv: NFKCMGANConvModule
    @ModuleInfo(key: "ff2") var ff2: NFKCMGANScaledFeedForward
    @ModuleInfo(key: "post_norm") var postNorm: LayerNorm

    init(_ c: NFKMLXCMGANConfiguration) {
        _ff1.wrappedValue = NFKCMGANScaledFeedForward(dim: c.channels, multiplier: c.feedForwardMultiplier)
        _attention.wrappedValue = NFKCMGANPreNormAttention(dim: c.channels, heads: c.heads, headDimension: c.headDimension,
                                                           maximumPosition: c.maximumRelativePosition)
        _conv.wrappedValue = NFKCMGANConvModule(dim: c.channels, kernel: c.convolutionKernel)
        _ff2.wrappedValue = NFKCMGANScaledFeedForward(dim: c.channels, multiplier: c.feedForwardMultiplier)
        _postNorm.wrappedValue = LayerNorm(dimensions: c.channels)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = ff1(input) + input
        x = attention(x) + x
        x = conv(x) + x
        x = ff2(x) + x
        return postNorm(x)
    }
}

/// The two-stage conformer block: a conformer over time (each frequency a sequence) and a conformer
/// over frequency (each frame a sequence), each with its own residual. Runs `[B, T, F, C]`.
final class NFKCMGANTSCB: Module {
    @ModuleInfo(key: "time_conformer") var time: NFKCMGANConformerBlock
    @ModuleInfo(key: "freq_conformer") var frequency: NFKCMGANConformerBlock

    init(_ c: NFKMLXCMGANConfiguration) {
        _time.wrappedValue = NFKCMGANConformerBlock(c)
        _frequency.wrappedValue = NFKCMGANConformerBlock(c)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, f, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let overTime = x.transposed(0, 2, 1, 3).reshaped([b * f, t, c])
        let timed = time(overTime) + overTime
        let overFrequency = timed.reshaped([b, f, t, c]).transposed(0, 2, 1, 3).reshaped([b * t, f, c])
        let done = frequency(overFrequency) + overFrequency
        return done.reshaped([b, t, f, c])
    }
}

// MARK: - Decoders

/// The magnitude-mask decoder: the dense net, the sub-pixel frequency upsample, `conv_1` `(1, 2)` to one
/// channel, an affine InstanceNorm + PReLU, a 1×1 `final_conv`, then `prelu_out`, a PReLU with one slope
/// per frequency bin (initialized at -0.25). The mask multiplies the compressed noisy magnitude.
final class NFKCMGANMaskDecoder: Module {
    @ModuleInfo(key: "dense_block") var dense: NFKCMGANDenseNet
    @ModuleInfo(key: "sub_pixel") var up: NFKMPSESubpixelUp
    @ModuleInfo(key: "conv_1") var conv1: Conv2d
    @ModuleInfo(key: "norm") var norm: InstanceNorm
    @ModuleInfo(key: "prelu") var prelu: PReLU
    @ModuleInfo(key: "final_conv") var finalConv: Conv2d
    @ModuleInfo(key: "prelu_out") var preluOut: PReLU

    init(_ c: NFKMLXCMGANConfiguration) {
        _dense.wrappedValue = NFKCMGANDenseNet(channels: c.channels)
        _up.wrappedValue = NFKMPSESubpixelUp(channels: c.channels)
        _conv1.wrappedValue = Conv2d(inputChannels: c.channels, outputChannels: 1, kernelSize: IntOrPair((1, 2)))
        _norm.wrappedValue = InstanceNorm(dimensions: 1, affine: true)
        _prelu.wrappedValue = PReLU(count: 1)
        _finalConv.wrappedValue = Conv2d(inputChannels: 1, outputChannels: 1, kernelSize: 1)
        _preluOut.wrappedValue = PReLU(count: c.bins, value: -0.25)
    }

    /// `[B, T, F/2, C]` → the mask `[B, T, F, 1]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = finalConv(prelu(norm(conv1(up(dense(x))))))                 // [B, T, F, 1]
        return preluOut(h.squeezed(axis: 3)).expandedDimensions(axis: 3)   // the per-bin slope runs over F
    }
}

/// The complex-residual decoder: the dense net, the sub-pixel upsample, an affine InstanceNorm + PReLU,
/// and a `(1, 2)` convolution to two channels (real, imaginary).
final class NFKCMGANComplexDecoder: Module {
    @ModuleInfo(key: "dense_block") var dense: NFKCMGANDenseNet
    @ModuleInfo(key: "sub_pixel") var up: NFKMPSESubpixelUp
    @ModuleInfo(key: "prelu") var prelu: PReLU
    @ModuleInfo(key: "norm") var norm: InstanceNorm
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ c: NFKMLXCMGANConfiguration) {
        _dense.wrappedValue = NFKCMGANDenseNet(channels: c.channels)
        _up.wrappedValue = NFKMPSESubpixelUp(channels: c.channels)
        _prelu.wrappedValue = PReLU(count: c.channels)
        _norm.wrappedValue = InstanceNorm(dimensions: c.channels, affine: true)
        _conv.wrappedValue = Conv2d(inputChannels: c.channels, outputChannels: 2, kernelSize: IntOrPair((1, 2)))
    }

    /// `[B, T, F/2, C]` → `[B, T, F, 2]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(prelu(norm(up(dense(x)))))
    }
}

// MARK: - Generator

/// The CMGAN generator (`TSCNet`) over a power-compressed complex spectrogram `[B, T, F, 2]` (real,
/// imaginary). The dense encoder reads `[magnitude, real, imaginary]`; the mask scales the noisy
/// magnitude under the noisy phase and the complex decoder adds a residual to both parts.
public final class NFKMLXCMGANNet: Module {
    @ModuleInfo(key: "dense_encoder") var encoder: NFKCMGANDenseEncoder
    @ModuleInfo(key: "tscb") var blocks: [NFKCMGANTSCB]
    @ModuleInfo(key: "mask_decoder") var maskDecoder: NFKCMGANMaskDecoder
    @ModuleInfo(key: "complex_decoder") var complexDecoder: NFKCMGANComplexDecoder
    let configuration: NFKMLXCMGANConfiguration

    init(_ c: NFKMLXCMGANConfiguration) {
        configuration = c
        _encoder.wrappedValue = NFKCMGANDenseEncoder(inputChannels: 3, channels: c.channels)
        _blocks.wrappedValue = (0 ..< c.blockCount).map { _ in NFKCMGANTSCB(c) }
        _maskDecoder.wrappedValue = NFKCMGANMaskDecoder(c)
        _complexDecoder.wrappedValue = NFKCMGANComplexDecoder(c)
        super.init()
    }

    /// The encoder's input: `[magnitude, real, imaginary]` on the channel axis.
    static func encoderInput(_ compressed: MLXArray) -> MLXArray {
        let magnitude = sqrt(compressed[0..., 0..., 0..., 0 ..< 1].square() + compressed[0..., 0..., 0..., 1 ..< 2].square())
        return concatenated([magnitude, compressed], axis: 3)
    }

    /// Compressed `[B, T, F, 2]` → enhanced compressed `[B, T, F, 2]`.
    public func callAsFunction(_ compressed: MLXArray) -> MLXArray {
        let (features, mask, residual) = stages(compressed)
        let magnitude = features[0..., 0..., 0..., 0 ..< 1]
        let phase = atan2(compressed[0..., 0..., 0..., 1 ..< 2], compressed[0..., 0..., 0..., 0 ..< 1])
        let masked = mask * magnitude
        let real = masked * cos(phase) + residual[0..., 0..., 0..., 0 ..< 1]
        let imaginary = masked * sin(phase) + residual[0..., 0..., 0..., 1 ..< 2]
        return concatenated([real, imaginary], axis: 3)
    }

    /// The seams the parity harness reads: the encoder input, the mask, and the complex residual, with
    /// every block output on the way.
    func stages(_ compressed: MLXArray) -> (features: MLXArray, mask: MLXArray, residual: MLXArray) {
        let features = Self.encoderInput(compressed)
        let deep = blockOutputs(features).last!
        return (features, maskDecoder(deep), complexDecoder(deep))
    }

    /// The dense encoder's output followed by every TSCB's output.
    func blockOutputs(_ features: MLXArray) -> [MLXArray] {
        var outputs = [encoder(features)]
        for block in blocks { outputs.append(block(outputs.last!)) }
        return outputs
    }
}

// MARK: - Backend

private final class NFKCMGANHolder: @unchecked Sendable {
    let net: NFKMLXCMGANNet
    init(_ net: NFKMLXCMGANNet) { self.net = net }
}

/// CMGAN as an InferKit backend: `NFKInputAudio` (any rate; resampled to 16 kHz) → the enhanced clip
/// under `NFKOutputAudio`.
@objc(NFKMLXCMGANBackend)
public final class NFKMLXCMGANBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKCMGANHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXCMGANNet, identifier: String, outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKCMGANHolder(net)
        self.identifier = identifier
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, rate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let configuration = holder.net.configuration
        let matched = NFKMLXAudioRate.matched(samples, from: rate, to: configuration.sampleRate)
        let enhanced = Self.enhance(matched, net: holder.net)
        eval(enhanced)
        let stream = enhanced.reshaped([enhanced.shape.last!]).asArray(Float.self)
        let url = outputDirectory.appendingPathComponent("cmgan-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: stream, sampleRate: configuration.sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(stream.count) / Double(configuration.sampleRate),
                                  sampleRate: Double(configuration.sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    /// The reference's 400-point periodic Hamming STFT at hop 100 (center, reflect).
    static func stft(_ configuration: NFKMLXCMGANConfiguration) -> NFKMLXComplexSTFT {
        let n = configuration.fftSize
        let hamming = (0 ..< n).map { 0.54 - 0.46 * cosf(2 * Float.pi * Float($0) / Float(n)) }
        return NFKMLXComplexSTFT(nFFT: n, hop: configuration.hopSize, window: MLXArray(hamming))
    }

    /// `evaluation.enhance_one_track`'s preparation: scale the clip to unit RMS and pad it to a multiple
    /// of the hop by repeating its first samples. Returns the scale, which the output divides by.
    static func prepared(_ samples: [Float], configuration: NFKMLXCMGANConfiguration) -> (padded: [Float], scale: Float) {
        let energy = samples.reduce(Float(0)) { $0 + $1 * $1 }
        let scale = (Float(samples.count) / energy).squareRoot()
        let hop = configuration.hopSize
        let paddedLength = (samples.count + hop - 1) / hop * hop
        let scaled = samples.map { $0 * scale }
        return (scaled + scaled[0 ..< paddedLength - samples.count], scale)
    }

    /// The compressed spectrum `[1, T, F, 2]` of the prepared clip.
    static func compressed(_ padded: [Float], configuration: NFKMLXCMGANConfiguration) -> MLXArray {
        let signal = padded.withUnsafeBufferPointer { MLXArray($0, [1, padded.count]) }
        let (magnitude, phase) = stft(configuration).transform(signal)          // [1, F, T]
        let compressed = pow(magnitude, Double(configuration.compressFactor))
        return stacked([compressed * cos(phase), compressed * sin(phase)], axis: 3).transposed(0, 2, 1, 3)
    }

    /// The reference path: prepare, compress, the generator, decompress, invert, undo the scale, and
    /// trim to the input length.
    static func enhance(_ samples: [Float], net: NFKMLXCMGANNet) -> MLXArray {
        let configuration = net.configuration
        let (padded, scale) = prepared(samples, configuration: configuration)
        let enhanced = net(compressed(padded, configuration: configuration)).transposed(0, 2, 1, 3)   // [1, F, T, 2]
        let real = enhanced[0..., 0..., 0..., 0], imaginary = enhanced[0..., 0..., 0..., 1]
        let magnitude = pow(sqrt(real.square() + imaginary.square()), Double(1 / configuration.compressFactor))
        let waveform = stft(configuration).inverse(magnitude: magnitude, phase: atan2(imaginary, real))
        return waveform[0..., 0 ..< samples.count] / scale
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

/// Registration and weight loading for CMGAN.
@objc(NFKMLXCMGAN)
public final class NFKMLXCMGAN: NSObject {
    @objc public static let modelName = "cmgan"

    static func makeNet(_ configuration: NFKMLXCMGANConfiguration = .init()) -> NFKMLXCMGANNet {
        let net = NFKMLXCMGANNet(configuration)
        net.train(false)                                                        // BatchNorm reads its running statistics
        return net
    }

    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL { try loadWeights(into: net, from: weightsURL) }
        return NFKMLXCMGANBackend(net: net, identifier: modelName)
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

    /// Registers `cmgan` with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads the released `ckpt` (a plain state dict the native torch reader opens). The `nn.Sequential`
    /// indices match the module arrays as they are; the dense nets' flat `conv{i}` / `norm{i}` / `prelu{i}`
    /// attributes and the four `TSCB_{i}` blocks map onto arrays; the 4-D and 3-D convolution weights
    /// transpose to channels-last; the BatchNorm counters are dropped.
    static func loadWeights(into net: NFKMLXCMGANNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped: [(String, MLXArray)] = checkpoint.arrays.compactMap { key, value in
            guard let name = remapReferenceKey(key) else { return nil }
            var tensor = value
            if checkpoint.needsConvTranspose {
                if value.ndim == 4 { tensor = value.transposed(0, 2, 3, 1) }
                if value.ndim == 3 { tensor = value.transposed(0, 2, 1) }
            }
            return (name, tensor)
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasSuffix("num_batches_tracked") { return nil }
        var name = key
        if name.hasPrefix("TSCB_"), let index = Int(name.dropFirst(5).prefix(1)) {
            name = "tscb.\(index - 1)." + name.dropFirst(7)
        }
        for stem in ["dilated_dense.", "dense_block."] {
            guard let range = name.range(of: stem) else { continue }
            let tail = name[range.upperBound...]                                // "conv3.weight"
            for part in ["conv", "norm", "prelu"] where tail.hasPrefix(part) {
                let afterPart = tail.dropFirst(part.count)
                guard let index = Int(afterPart.prefix(1)) else { continue }
                name = name[..<range.upperBound] + "layers.\(index - 1).\(part)" + afterPart.dropFirst(1)
            }
        }
        return name
    }
}
