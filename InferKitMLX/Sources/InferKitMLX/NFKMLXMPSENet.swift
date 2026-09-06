// MP-SENet (yxlu-0102/MP-SENet, MIT): single-forward speech enhancement in the time-frequency domain
// that denoises MAGNITUDE and PHASE in parallel. The first supervised restoration port, and the fast,
// deterministic counterweight to the SGMSE+ diffusion anchor. The complex STFT with phase is the shared
// `NFKMLXComplexSTFT` primitive.
//
// The released `model.py` core is a TS-TRANSFORMER, not the TS-conformer the paper's name suggests (the
// repo's `conformer.py` is unused by the release). Each block is a `TransformerBlock`
// (norm1 → fused-QKV self-attention → residual → norm2 → FFN → residual → norm3), and the FFN is a
// BIDIRECTIONAL GRU (hidden `2·dense`) followed by leaky-ReLU and a linear projection. Grounded against
// the released `models/model.py` and `models/transformer.py`.
//
// Pipeline: noisy waveform → STFT → magnitude (power-compressed by `compressFactor`) and phase →
// a DenseEncoder → `tsBlocks` time-then-frequency transformer blocks → a magnitude-mask decoder and a
// parallel phase decoder → the enhanced complex spectrogram → iSTFT.
//
// SCAFFOLD STATUS: structure and checkpoint key names load the released generator; the parity-sensitive
// numeric choices are marked `PARITY:` and are pinned against `run_reference.py mpsenet` on the released
// weights. Tensors flow in MLX's NHWC `[B, T, F, C]` where the reference is NCHW `[B, C, T, F]`.

import Foundation
import InferKit
import MLX
import MLXNN

/// The 16 kHz MP-SENet configuration (the released `MP-SENet` generator).
public struct NFKMLXMPSENetConfiguration: Sendable {
    public var sampleRate: Int
    public var fftSize: Int
    public var hopSize: Int
    public var winSize: Int
    public var denseChannel: Int
    public var tsBlocks: Int
    public var heads: Int
    public var denseDepth: Int
    /// Magnitude is raised to this power before the network reads it, and to its inverse after
    /// (`compress_factor`, 0.3 at 16 kHz). PARITY: load-bearing on the released weights.
    public var compressFactor: Float
    /// The learnable-sigmoid ceiling on the magnitude mask (`beta`, 2.0). PARITY.
    public var beta: Float

    public var bins: Int { fftSize / 2 + 1 }

    public init(sampleRate: Int = 16000, fftSize: Int = 400, hopSize: Int = 100, winSize: Int = 400,
                denseChannel: Int = 64, tsBlocks: Int = 4, heads: Int = 4, denseDepth: Int = 4,
                compressFactor: Float = 0.3, beta: Float = 2.0) {
        self.sampleRate = sampleRate
        self.fftSize = fftSize
        self.hopSize = hopSize
        self.winSize = winSize
        self.denseChannel = denseChannel
        self.tsBlocks = tsBlocks
        self.heads = heads
        self.denseDepth = denseDepth
        self.compressFactor = compressFactor
        self.beta = beta
    }
}

// MARK: - Dense block

/// One dilated dense convolution (`Conv2d` + affine `InstanceNorm` + per-channel `PReLU`), kernel
/// `(2, 3)` over `(time, frequency)`, dilation `(2^i, 1)`. The reference wraps a
/// `ConstantPad2d((1, 1, dilation, 0))` before the convolution (freq padded 1 each side, time padded
/// `dilation` on the PAST side only, so the even kernel keeps the frame count and stays causal), with
/// the convolution itself at padding 0. This folds that pad into the forward.
final class NFKMPSEDenseConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: InstanceNorm
    @ModuleInfo(key: "prelu") var prelu: PReLU
    let dilation: Int

    init(inChannels: Int, denseChannel: Int, dilation: Int) {
        self.dilation = dilation
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: denseChannel,
                                    kernelSize: IntOrPair((2, 3)), stride: 1,
                                    padding: IntOrPair((0, 1)),                 // freq symmetric 1; time pre-padded
                                    dilation: IntOrPair((dilation, 1)))
        _norm.wrappedValue = InstanceNorm(dimensions: denseChannel, affine: true)
        _prelu.wrappedValue = PReLU(count: denseChannel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((dilation, 0)), IntOrPair(0), IntOrPair(0)], mode: .constant)
        return prelu(norm(conv(padded)))
    }
}

/// The dense block: `depth` dilated convolutions, each reading the channel-wise concatenation of every
/// previous output with the block input (dilations `1, 2, 4, 8`). The reference nests this as
/// `dense_block.dense_block.{i}` (a module named `dense_block` holding a `ModuleList` also reached as
/// `dense_block`).
final class NFKMPSEDenseBlock: Module {
    @ModuleInfo(key: "dense_block") var layers: [NFKMPSEDenseConv]

    init(_ config: NFKMLXMPSENetConfiguration) {
        let c = config.denseChannel
        _layers.wrappedValue = (0 ..< config.denseDepth).map {
            NFKMPSEDenseConv(inChannels: c * ($0 + 1), denseChannel: c, dilation: 1 << $0)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var skip = x
        var out = x
        for layer in layers {
            out = layer(skip)
            skip = concatenated([out, skip], axis: 3)               // channel axis in NHWC
        }
        return out
    }
}

// MARK: - Dense encoder

/// `dense_conv_1` (1×1 to the dense width) → the dense block → `dense_conv_2` (a `(1,3)` stride-`(1,2)`
/// convolution that halves the frequency axis). Each convolution is `Conv2d` + affine `InstanceNorm` +
/// `PReLU`, the reference `nn.Sequential` (indices 0/1/2) translated onto the named submodules by the
/// loader.
final class NFKMPSEDenseEncoder: Module {
    @ModuleInfo(key: "dense_conv_1_conv") var conv1: Conv2d
    @ModuleInfo(key: "dense_conv_1_norm") var norm1: InstanceNorm
    @ModuleInfo(key: "dense_conv_1_prelu") var prelu1: PReLU
    @ModuleInfo(key: "dense_block") var block: NFKMPSEDenseBlock
    @ModuleInfo(key: "dense_conv_2_conv") var conv2: Conv2d
    @ModuleInfo(key: "dense_conv_2_norm") var norm2: InstanceNorm
    @ModuleInfo(key: "dense_conv_2_prelu") var prelu2: PReLU

    init(_ config: NFKMLXMPSENetConfiguration) {
        let c = config.denseChannel
        _conv1.wrappedValue = Conv2d(inputChannels: 2, outputChannels: c, kernelSize: 1)
        _norm1.wrappedValue = InstanceNorm(dimensions: c, affine: true)
        _prelu1.wrappedValue = PReLU(count: c)
        _block.wrappedValue = NFKMPSEDenseBlock(config)
        // PARITY: `(1,3)` stride `(1,2)` with a freq pad of 1 takes 201 bins → 101.
        _conv2.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: IntOrPair((1, 3)),
                                     stride: IntOrPair((1, 2)), padding: IntOrPair((0, 1)))
        _norm2.wrappedValue = InstanceNorm(dimensions: c, affine: true)
        _prelu2.wrappedValue = PReLU(count: c)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = prelu1(norm1(conv1(x)))
        return prelu2(norm2(conv2(block(h))))
    }
}

// MARK: - Transformer block

/// Fused-QKV self-attention with the `in_proj_weight` / `in_proj_bias` and `out_proj` that
/// `nn.MultiheadAttention` stores. No pre-norm here — the block's `norm1` is applied before this.
final class NFKMPSEAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray        // [3C, C]
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray            // [3C]
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int

    init(dim: Int, heads: Int) {
        self.heads = heads
        _inProjWeight.wrappedValue = MLXArray.zeros([3 * dim, dim])
        _inProjBias.wrappedValue = MLXArray.zeros([3 * dim])
        _outProj.wrappedValue = Linear(dim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, n, c) = (x.dim(0), x.dim(1), x.dim(2))
        let dk = c / heads
        let qkv = matmul(x, inProjWeight.transposed(1, 0)) + inProjBias          // [B, N, 3C]
        func head(_ slice: MLXArray) -> MLXArray { slice.reshaped([b, n, heads, dk]).transposed(0, 2, 1, 3) }
        let q = head(qkv[0..., 0..., 0 ..< c])
        let k = head(qkv[0..., 0..., c ..< 2 * c])
        let v = head(qkv[0..., 0..., 2 * c ..< 3 * c])
        let scores = matmul(q, k.transposed(0, 1, 3, 2)) / sqrt(Float(dk))
        let attended = matmul(softmax(scores, axis: -1), v)                      // [B, h, N, dk]
        return outProj(attended.transposed(0, 2, 1, 3).reshaped([b, n, c]))
    }
}

/// The FFN: a bidirectional GRU (hidden `2·dim`, the shared `NFKMLXBiGRU`), leaky-ReLU, then a linear
/// projection back to `dim`.
final class NFKMPSEFFN: Module {
    @ModuleInfo(key: "gru") var gru: NFKMLXBiGRU
    @ModuleInfo(key: "linear") var linear: Linear

    init(dim: Int) {
        _gru.wrappedValue = NFKMLXBiGRU(inputSize: dim, hiddenSize: dim * 2)
        _linear.wrappedValue = Linear(dim * 4, dim)                        // bidirectional 2·(2·dim) → dim
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = gru(x)                                                     // [B, L, 4·dim]
        // PARITY: F.leaky_relu default negative slope 0.01, written as a max to avoid an API dependency.
        let activated = maximum(h, 0.01 * h)
        return linear(activated)
    }
}

/// One transformer block: pre-norm self-attention with a residual, pre-norm FFN with a residual, then a
/// final norm (`norm1 → attention → +res → norm2 → ffn → +res → norm3`).
final class NFKMPSETransformer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: NFKMPSEAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: NFKMPSEFFN
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    init(_ config: NFKMLXMPSENetConfiguration) {
        let c = config.denseChannel
        _norm1.wrappedValue = LayerNorm(dimensions: c)
        _attention.wrappedValue = NFKMPSEAttention(dim: c, heads: config.heads)
        _norm2.wrappedValue = LayerNorm(dimensions: c)
        _ffn.wrappedValue = NFKMPSEFFN(dim: c)
        _norm3.wrappedValue = LayerNorm(dimensions: c)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // The reference's `nn.MultiheadAttention` and `nn.GRU` are created WITHOUT batch_first, so they
        // default to batch_first=false and run over the FIRST axis of the `[d0, d1, C]` input the TS
        // reshape hands them — not the sequence position the `time`/`freq` names suggest. The trained
        // weights depend on that, so transpose here to make this port's batch-first attention and GRU
        // operate over d0 as well. (The SAM2 memory-attention batch_first trap, again.)
        let xt = x.transposed(1, 0, 2)
        var h = xt + attention(norm1(xt))
        h = h + ffn(norm2(h))
        return norm3(h).transposed(1, 0, 2)
    }
}

/// A time-then-frequency transformer block: the time transformer runs over each frequency's frame
/// sequence, the frequency transformer over each frame's frequency sequence, each added as a residual.
/// Input and output are `[B, T, F, C]`.
final class NFKMPSETSTransformer: Module {
    @ModuleInfo(key: "time_transformer") var time: NFKMPSETransformer
    @ModuleInfo(key: "freq_transformer") var freq: NFKMPSETransformer

    init(_ config: NFKMLXMPSENetConfiguration) {
        _time.wrappedValue = NFKMPSETransformer(config)
        _freq.wrappedValue = NFKMPSETransformer(config)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, f, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        var h = x.transposed(0, 2, 1, 3).reshaped([b * f, t, c])           // [B*F, T, C]
        h = time(h) + h
        h = h.reshaped([b, f, t, c]).transposed(0, 2, 1, 3)                // [B, T, F, C]
        var g = h.reshaped([b * t, f, c])                                  // [B*T, F, C]
        g = freq(g) + g
        return g.reshaped([b, t, f, c])
    }
}

// MARK: - Subpixel upsample + learnable sigmoid

/// The reference `SPConvTranspose2d`: a constant freq pad, a `(1,3)` convolution to `channels·r`, then a
/// subpixel rearrangement that interleaves the extra channels along the frequency axis (`r = 2`). In
/// NHWC the channel axis splits as `(r outer, channels inner)` and merges with frequency as `(freq outer,
/// r inner)`, matching the reference view/permute. Its inner convolution is named `conv`.
final class NFKMPSESubpixelUp: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    let channels: Int
    let ratio: Int

    init(channels: Int, ratio: Int = 2) {
        self.channels = channels
        self.ratio = ratio
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels * ratio,
                                    kernelSize: IntOrPair((1, 3)))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((1, 1)), IntOrPair(0)], mode: .constant)
        let h = conv(padded)                                              // [B, T, F', C·r]
        let (b, t, fp) = (h.dim(0), h.dim(1), h.dim(2))
        return h.reshaped([b, t, fp, ratio, channels]).reshaped([b, t, fp * ratio, channels])
    }
}

/// `beta · sigmoid(slope · x)`, one learnable slope per frequency bin (`LearnableSigmoid_2d`).
final class NFKMPSELearnableSigmoid: Module {
    @ParameterInfo(key: "slope") var slope: MLXArray                       // [bins, 1]
    let beta: Float

    init(bins: Int, beta: Float) {
        self.beta = beta
        _slope.wrappedValue = MLXArray.ones([bins, 1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {                       // x [B, T, F, 1]
        let bins = slope.dim(0)
        return beta * sigmoid(slope.reshaped([1, 1, bins, 1]) * x)
    }
}

// MARK: - Decoders

/// The magnitude-mask decoder: the dense block, the subpixel upsample, an affine InstanceNorm + PReLU, a
/// `(1,2)` convolution to one channel, then the learnable sigmoid. The mask multiplies the noisy
/// magnitude. The reference `mask_conv` Sequential (SPConvTranspose2d / InstanceNorm / PReLU / Conv2d) is
/// translated onto these submodules by the loader.
final class NFKMPSEMaskDecoder: Module {
    @ModuleInfo(key: "dense_block") var block: NFKMPSEDenseBlock
    @ModuleInfo(key: "up") var up: NFKMPSESubpixelUp
    @ModuleInfo(key: "norm") var norm: InstanceNorm
    @ModuleInfo(key: "prelu") var prelu: PReLU
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "lsigmoid") var lsigmoid: NFKMPSELearnableSigmoid

    init(_ config: NFKMLXMPSENetConfiguration) {
        let c = config.denseChannel
        _block.wrappedValue = NFKMPSEDenseBlock(config)
        _up.wrappedValue = NFKMPSESubpixelUp(channels: c)
        _norm.wrappedValue = InstanceNorm(dimensions: c, affine: true)
        _prelu.wrappedValue = PReLU(count: c)
        _conv.wrappedValue = Conv2d(inputChannels: c, outputChannels: 1, kernelSize: IntOrPair((1, 2)))
        _lsigmoid.wrappedValue = NFKMPSELearnableSigmoid(bins: config.bins, beta: config.beta)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {                       // [B, T, F/2, C] → mask [B, T, F, 1]
        lsigmoid(conv(prelu(norm(up(block(x))))))
    }
}

/// The phase decoder: the dense block, the subpixel upsample, an affine InstanceNorm + PReLU, then two
/// parallel `(1,2)` convolutions whose `atan2` is the enhanced phase.
final class NFKMPSEPhaseDecoder: Module {
    @ModuleInfo(key: "dense_block") var block: NFKMPSEDenseBlock
    @ModuleInfo(key: "up") var up: NFKMPSESubpixelUp
    @ModuleInfo(key: "norm") var norm: InstanceNorm
    @ModuleInfo(key: "prelu") var prelu: PReLU
    @ModuleInfo(key: "conv_r") var convR: Conv2d
    @ModuleInfo(key: "conv_i") var convI: Conv2d

    init(_ config: NFKMLXMPSENetConfiguration) {
        let c = config.denseChannel
        _block.wrappedValue = NFKMPSEDenseBlock(config)
        _up.wrappedValue = NFKMPSESubpixelUp(channels: c)
        _norm.wrappedValue = InstanceNorm(dimensions: c, affine: true)
        _prelu.wrappedValue = PReLU(count: c)
        _convR.wrappedValue = Conv2d(inputChannels: c, outputChannels: 1, kernelSize: IntOrPair((1, 2)))
        _convI.wrappedValue = Conv2d(inputChannels: c, outputChannels: 1, kernelSize: IntOrPair((1, 2)))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {                       // [B, T, F/2, C] → phase [B, T, F, 1]
        let h = prelu(norm(up(block(x))))
        return atan2(convI(h), convR(h))
    }
}

// MARK: - The generator

/// The MP-SENet generator (`MPNet`): the dense encoder, the TS-transformer stack, and the parallel mask
/// and phase decoders. It reads and writes compressed magnitude and phase, each `[1, bins, frames]`.
public final class NFKMLXMPSENet: Module {
    @ModuleInfo(key: "dense_encoder") var encoder: NFKMPSEDenseEncoder
    @ModuleInfo(key: "TSTransformer") var blocks: [NFKMPSETSTransformer]
    @ModuleInfo(key: "mask_decoder") var maskDecoder: NFKMPSEMaskDecoder
    @ModuleInfo(key: "phase_decoder") var phaseDecoder: NFKMPSEPhaseDecoder

    public init(_ config: NFKMLXMPSENetConfiguration) {
        _encoder.wrappedValue = NFKMPSEDenseEncoder(config)
        _blocks.wrappedValue = (0 ..< config.tsBlocks).map { _ in NFKMPSETSTransformer(config) }
        _maskDecoder.wrappedValue = NFKMPSEMaskDecoder(config)
        _phaseDecoder.wrappedValue = NFKMPSEPhaseDecoder(config)
    }

    /// Compressed magnitude and phase, each `[1, bins, frames]` → enhanced compressed magnitude and
    /// phase, each `[1, bins, frames]`.
    public func callAsFunction(magnitude: MLXArray, phase: MLXArray) -> (magnitude: MLXArray, phase: MLXArray) {
        let magTF = magnitude.transposed(0, 2, 1).expandedDimensions(axis: 3)   // [1, T, F, 1]
        let phaTF = phase.transposed(0, 2, 1).expandedDimensions(axis: 3)
        var x = concatenated([magTF, phaTF], axis: 3)                          // [1, T, F, 2]
        x = encoder(x)                                                         // [1, T, F/2, C]
        for block in blocks { x = block(x) }
        let enhancedMag = magTF * maskDecoder(x)                               // [1, T, F, 1]
        let enhancedPha = phaseDecoder(x)                                      // [1, T, F, 1]
        func toBinsFrames(_ y: MLXArray) -> MLXArray { y.squeezed(axis: 3).transposed(0, 2, 1) }
        return (toBinsFrames(enhancedMag), toBinsFrames(enhancedPha))
    }
}

// MARK: - Backend

private final class NFKMPSEHolder: @unchecked Sendable {
    let net: NFKMLXMPSENet
    let config: NFKMLXMPSENetConfiguration
    init(_ net: NFKMLXMPSENet, _ config: NFKMLXMPSENetConfiguration) { self.net = net; self.config = config }
}

/// Speech enhancement (denoise plus dereverberation) as an InferKit backend. Reads `NFKInputAudio`;
/// returns the enhanced clip as a single `NFKAudioAsset` under `NFKOutputAudio`.
@objc(NFKMLXMPSENetBackend)
public final class NFKMLXMPSENetBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKMPSEHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXMPSENet, config: NFKMLXMPSENetConfiguration, identifier: String,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKMPSEHolder(net, config)
        self.identifier = identifier
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let config = holder.config
        // PARITY: a clip at another rate must be resampled to the model's rate first (the STFT and the
        // mask are defined at 16 kHz), via NFKMLXAudioRate.matched, as the VAD/tagger do.
        let enhanced = Self.enhance(samples, net: holder.net, config: config)
        eval(enhanced)
        let stream = enhanced.reshaped([enhanced.shape.last!]).asArray(Float.self)
        _ = sampleRate

        let url = outputDirectory.appendingPathComponent("mpsenet-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: stream, sampleRate: config.sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(stream.count) / Double(config.sampleRate),
                                  sampleRate: Double(config.sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    /// The full front-end + net + back-end path: STFT, magnitude compression, the generator, then
    /// decompression and iSTFT. Exposed for the parity harness.
    static func enhance(_ samples: [Float], net: NFKMLXMPSENet, config: NFKMLXMPSENetConfiguration) -> MLXArray {
        let stft = NFKMLXComplexSTFT(nFFT: config.fftSize, hop: config.hopSize, winLength: config.winSize)
        let signal = samples.withUnsafeBufferPointer { MLXArray($0, [1, samples.count]) }
        let (magnitude, phase) = stft.transform(signal)
        let compressed = pow(magnitude, Double(config.compressFactor))         // mag^0.3
        let (enhancedMag, enhancedPha) = net(magnitude: compressed, phase: phase)
        let decompressed = pow(enhancedMag, Double(1 / config.compressFactor))
        return stft.inverse(magnitude: decompressed, phase: enhancedPha)
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

/// Registration and weight loading for MP-SENet.
@objc(NFKMLXMPSENet_Factory)
public final class NFKMLXMPSENetFactory: NSObject {
    @objc public static let modelName = "mpsenet"

    static func makeNet(_ config: NFKMLXMPSENetConfiguration = .init()) -> NFKMLXMPSENet { NFKMLXMPSENet(config) }

    /// Builds a backend from optional local weights (a converted safetensors, or the released `.pth`
    /// through the native torch reader). A nil `weightsURL` builds random weights (`isReady` true).
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let config = NFKMLXMPSENetConfiguration()
        let net = makeNet(config)
        if let weightsURL { try loadWeights(into: net, from: weightsURL) }
        return NFKMLXMPSENetBackend(net: net, config: config, identifier: modelName)
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

    /// Registers `mpsenet` with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a released MP-SENet generator checkpoint (a bare state dict) into the net: fold the
    /// bidirectional GRUs, remap the reference names, and transpose the 4-D convolution weights.
    static func loadWeights(into net: NFKMLXMPSENet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let folded = NFKMLXRecurrentFold.fold(checkpoint.arrays)
        let mapped: [(String, MLXArray)] = folded.compactMap { key, value in
            guard let name = remapReferenceKey(key) else { return nil }
            // 4-D convolution weights transpose PyTorch [out, in, kH, kW] → MLX [out, kH, kW, in],
            // gated so a fine-tuned InferKit save (already MLX layout) is not double-transposed.
            let tensor = value.ndim == 4 && checkpoint.needsConvTranspose ? value.transposed(0, 2, 3, 1) : value
            return (name, tensor)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// The reference-name → module-key remap: the `nn.Sequential` indices of the dense convolutions and
    /// the decoder convolutions become the named submodules; everything else (the transformer norms,
    /// attention, the folded GRU keys, `phase_conv_r/i`, `lsigmoid.slope`) passes through unchanged.
    static func remapReferenceKey(_ key: String) -> String? {
        // Dense block entries first: dense_block.{i}.{1,2,3} → .{i}.conv/norm/prelu (index 0 is the
        // ConstantPad2d, which carries no parameters and is dropped — nil).
        guard var name = replaceDenseBlockIndices(key) else { return nil }
        // Dense encoder Sequentials: dense_conv_1.{0,1,2} / dense_conv_2.{0,1,2} → *_conv/_norm/_prelu.
        for stem in ["dense_conv_1", "dense_conv_2"] {
            for (index, part) in [("0", "conv"), ("1", "norm"), ("2", "prelu")] {
                name = name.replacingOccurrences(of: "\(stem).\(index).", with: "\(stem)_\(part).")
            }
        }
        // Mask decoder mask_conv Sequential → up.conv / norm / prelu / conv.
        name = name.replacingOccurrences(of: "mask_conv.0.conv.", with: "up.conv.")
        name = name.replacingOccurrences(of: "mask_conv.1.", with: "norm.")
        name = name.replacingOccurrences(of: "mask_conv.2.", with: "prelu.")
        name = name.replacingOccurrences(of: "mask_conv.3.", with: "conv.")
        // Phase decoder phase_conv Sequential → up.conv / norm / prelu (phase_conv_r/i pass through).
        name = name.replacingOccurrences(of: "phase_conv.0.conv.", with: "up.conv.")
        name = name.replacingOccurrences(of: "phase_conv.1.", with: "norm.")
        name = name.replacingOccurrences(of: "phase_conv.2.", with: "prelu.")
        name = name.replacingOccurrences(of: "phase_conv_r.", with: "conv_r.")
        name = name.replacingOccurrences(of: "phase_conv_i.", with: "conv_i.")
        // nn.MultiheadAttention nests as attention.attn? No — the reference attribute is `attention`,
        // and its fused params sit directly under it (in_proj_weight, in_proj_bias, out_proj).
        return name
    }

    /// Turns `...dense_block.dense_block.<i>.{1,2,3}...` into `...dense_block.dense_block.<i>.{conv,norm,
    /// prelu}...`, dropping the pad at index 0.
    private static func replaceDenseBlockIndices(_ key: String) -> String? {
        guard let range = key.range(of: "dense_block.dense_block.") else { return key }
        let head = String(key[key.startIndex ..< range.upperBound])
        var tail = String(key[range.upperBound...])                       // "<i>.<j>.<param>"
        let parts = tail.split(separator: ".", maxSplits: 2, omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 3, let index = Int(parts[0]) else { return key }
        let name: String
        switch parts[1] {
        case "0": return nil                                              // the ConstantPad2d carries no parameters
        case "1": name = "conv"
        case "2": name = "norm"
        case "3": name = "prelu"
        default: name = parts[1]
        }
        tail = "\(index).\(name).\(parts[2])"
        return head + tail
    }
}
