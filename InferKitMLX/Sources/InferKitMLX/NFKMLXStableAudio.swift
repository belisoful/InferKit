//
//  NFKMLXStableAudio.swift
//  InferKitMLX
//
//  Stable Audio Open 1.0's networks: the Oobleck waveform autoencoder, the latent diffusion transformer,
//  the seconds conditioners, and the cosine-schedule DPM-Solver++ SDE sampler that drives them. The
//  reference is diffusers' `StableAudioPipeline` (`AutoencoderOobleck`, `StableAudioDiTModel`,
//  `StableAudioProjectionModel`, `CosineDPMSolverMultistepScheduler`), which is what the released
//  `model_index.json` names.
//
//  The weights are under the Stability AI Community License: free below USD 1M annual revenue, a
//  separate license above it. See Docs/companions.md.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

// MARK: - Oobleck autoencoder

/// The Oobleck autoencoder's geometry. Defaults are the released `vae/config.json`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXOobleckConfiguration: Sendable, Equatable {
    public var encoderHiddenSize: Int
    public var downsamplingRatios: [Int]
    public var channelMultiples: [Int]
    public var decoderChannels: Int
    public var decoderInputChannels: Int
    public var audioChannels: Int
    public var samplingRate: Int

    public init(encoderHiddenSize: Int = 128, downsamplingRatios: [Int] = [2, 4, 4, 8, 8],
                channelMultiples: [Int] = [1, 2, 4, 8, 16], decoderChannels: Int = 128,
                decoderInputChannels: Int = 64, audioChannels: Int = 2, samplingRate: Int = 44100) {
        self.encoderHiddenSize = encoderHiddenSize
        self.downsamplingRatios = downsamplingRatios
        self.channelMultiples = channelMultiples
        self.decoderChannels = decoderChannels
        self.decoderInputChannels = decoderInputChannels
        self.audioChannels = audioChannels
        self.samplingRate = samplingRate
    }

    /// The released Stable Audio Open 1.0 autoencoder.
    public static let release = NFKMLXOobleckConfiguration()

    /// A small configuration with the release's structure, for tests that need no weights.
    static let tiny = NFKMLXOobleckConfiguration(encoderHiddenSize: 8, downsamplingRatios: [2, 4],
                                                 channelMultiples: [1, 2], decoderChannels: 8,
                                                 decoderInputChannels: 4, audioChannels: 2)

    /// Waveform samples per latent frame (2048 for the release).
    public var hopLength: Int { downsamplingRatios.reduce(1, *) }

    /// The latent channels: the encoder emits a mean and a scale of this width each.
    public var latentChannels: Int { decoderInputChannels }

    /// The geometry a diffusers `vae/config.json` states.
    static func configuration(fromHuggingFace url: URL) throws -> NFKMLXOobleckConfiguration {
        let json = try NFKMLXStableAudio.json(url)
        let base = NFKMLXOobleckConfiguration()
        return NFKMLXOobleckConfiguration(
            encoderHiddenSize: NFKMLXStableAudio.integer(json["encoder_hidden_size"]) ?? base.encoderHiddenSize,
            downsamplingRatios: NFKMLXStableAudio.integers(json["downsampling_ratios"]) ?? base.downsamplingRatios,
            channelMultiples: NFKMLXStableAudio.integers(json["channel_multiples"]) ?? base.channelMultiples,
            decoderChannels: NFKMLXStableAudio.integer(json["decoder_channels"]) ?? base.decoderChannels,
            decoderInputChannels: NFKMLXStableAudio.integer(json["decoder_input_channels"]) ?? base.decoderInputChannels,
            audioChannels: NFKMLXStableAudio.integer(json["audio_channels"]) ?? base.audioChannels,
            samplingRate: NFKMLXStableAudio.integer(json["sampling_rate"]) ?? base.samplingRate)
    }
}

/// The log-scale Snake activation with a separate amplitude: `x + sin²(exp(α)·x) / (exp(β) + 1e-9)`, one
/// learned `α` and `β` per channel. The checkpoint stores both as `[1, C, 1]` for the reference's NCL;
/// this module holds them `[1, 1, C]` for NLC, and the loader transposes a PyTorch-layout file.
final class NFKOobleckSnake: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.zeros([1, 1, channels])
        _beta.wrappedValue = MLXArray.zeros([1, 1, channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + (1 / (exp(beta) + 1e-9)) * sin(exp(alpha) * x).square()
    }
}

/// A dilated residual unit: snake → 7-wide dilated convolution → snake → 1×1 convolution, added back.
final class NFKOobleckResidualUnit: Module {
    let snake1: NFKOobleckSnake
    let conv1: Conv1d
    let snake2: NFKOobleckSnake
    let conv2: Conv1d

    init(dim: Int, dilation: Int) {
        snake1 = NFKOobleckSnake(channels: dim)
        conv1 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3 * dilation,
                       dilation: dilation)
        snake2 = NFKOobleckSnake(channels: dim)
        conv2 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + conv2(snake2(conv1(snake1(x))))
    }
}

/// An encoder stage: three residual units at dilations 1, 3, and 9, a snake, and a strided convolution
/// (kernel 2·stride) that divides the length by the stride and widens the channels.
final class NFKOobleckEncoderBlock: Module {
    @ModuleInfo(key: "res_unit1") var resUnit1: NFKOobleckResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: NFKOobleckResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: NFKOobleckResidualUnit
    let snake1: NFKOobleckSnake
    let conv1: Conv1d

    init(inputDim: Int, outputDim: Int, stride: Int) {
        _resUnit1.wrappedValue = NFKOobleckResidualUnit(dim: inputDim, dilation: 1)
        _resUnit2.wrappedValue = NFKOobleckResidualUnit(dim: inputDim, dilation: 3)
        _resUnit3.wrappedValue = NFKOobleckResidualUnit(dim: inputDim, dilation: 9)
        snake1 = NFKOobleckSnake(channels: inputDim)
        conv1 = Conv1d(inputChannels: inputDim, outputChannels: outputDim, kernelSize: 2 * stride, stride: stride,
                       padding: (stride + 1) / 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv1(snake1(resUnit3(resUnit2(resUnit1(x)))))
    }
}

/// A decoder stage: a snake, a transposed convolution (kernel 2·stride) that multiplies the length by the
/// stride and narrows the channels, and three residual units at dilations 1, 3, and 9.
final class NFKOobleckDecoderBlock: Module {
    let snake1: NFKOobleckSnake
    @ModuleInfo(key: "conv_t1") var convT1: NFKDemucsConvT1d
    @ModuleInfo(key: "res_unit1") var resUnit1: NFKOobleckResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: NFKOobleckResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: NFKOobleckResidualUnit

    init(inputDim: Int, outputDim: Int, stride: Int) {
        snake1 = NFKOobleckSnake(channels: inputDim)
        _convT1.wrappedValue = NFKDemucsConvT1d(inputDim, outputDim, kernel: 2 * stride, stride: stride,
                                                padding: (stride + 1) / 2)
        _resUnit1.wrappedValue = NFKOobleckResidualUnit(dim: outputDim, dilation: 1)
        _resUnit2.wrappedValue = NFKOobleckResidualUnit(dim: outputDim, dilation: 3)
        _resUnit3.wrappedValue = NFKOobleckResidualUnit(dim: outputDim, dilation: 9)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resUnit3(resUnit2(resUnit1(convT1(snake1(x)))))
    }
}

/// The encoder: waveform `[B, S, audioChannels]` → `[B, S / hop, 2 · latentChannels]`, the posterior's
/// mean and scale on channels.
final class NFKOobleckEncoder: Module {
    let conv1: Conv1d
    @ModuleInfo(key: "block") var block: [NFKOobleckEncoderBlock]
    let snake1: NFKOobleckSnake
    let conv2: Conv1d

    init(_ c: NFKMLXOobleckConfiguration) {
        let multiples = [1] + c.channelMultiples
        conv1 = Conv1d(inputChannels: c.audioChannels, outputChannels: c.encoderHiddenSize, kernelSize: 7, padding: 3)
        _block.wrappedValue = c.downsamplingRatios.enumerated().map { index, stride in
            NFKOobleckEncoderBlock(inputDim: c.encoderHiddenSize * multiples[index],
                                   outputDim: c.encoderHiddenSize * multiples[index + 1], stride: stride)
        }
        let width = c.encoderHiddenSize * multiples[multiples.count - 1]
        snake1 = NFKOobleckSnake(channels: width)
        conv2 = Conv1d(inputChannels: width, outputChannels: c.encoderHiddenSize, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2(snake1(block.reduce(conv1(x)) { $1($0) }))
    }
}

/// The decoder: latents `[B, T, latentChannels]` → waveform `[B, T · hop, audioChannels]`.
final class NFKOobleckDecoder: Module {
    let conv1: Conv1d
    @ModuleInfo(key: "block") var block: [NFKOobleckDecoderBlock]
    let snake1: NFKOobleckSnake
    let conv2: Conv1d

    init(_ c: NFKMLXOobleckConfiguration) {
        let multiples = [1] + c.channelMultiples
        let stages = c.downsamplingRatios.count
        conv1 = Conv1d(inputChannels: c.decoderInputChannels, outputChannels: c.decoderChannels * multiples[stages],
                       kernelSize: 7, padding: 3)
        _block.wrappedValue = c.downsamplingRatios.reversed().enumerated().map { index, stride in
            NFKOobleckDecoderBlock(inputDim: c.decoderChannels * multiples[stages - index],
                                   outputDim: c.decoderChannels * multiples[stages - index - 1], stride: stride)
        }
        snake1 = NFKOobleckSnake(channels: c.decoderChannels)
        conv2 = Conv1d(inputChannels: c.decoderChannels, outputChannels: c.audioChannels, kernelSize: 7, padding: 3,
                       bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2(snake1(block.reduce(conv1(x)) { $1($0) }))
    }
}

/// The Oobleck waveform autoencoder (`AutoencoderOobleck`): a Snake convolutional encoder to a diagonal
/// Gaussian over 64 latent channels at 1/2048 of the sample rate, and the mirrored decoder.
///
/// @discussion Layouts are NLC: a waveform is `[B, samples, channels]` and a latent `[B, frames, 64]`. The
/// posterior's standard deviation is `softplus(scale) + 1e-4`, as the reference's
/// `OobleckDiagonalGaussianDistribution` forms it. A built network starts in evaluation mode, because its
/// transposed convolutions compute differently while training.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXOobleckAutoencoderNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKOobleckEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKOobleckDecoder

    public let configuration: NFKMLXOobleckConfiguration

    public init(_ configuration: NFKMLXOobleckConfiguration = .release) {
        self.configuration = configuration
        _encoder.wrappedValue = NFKOobleckEncoder(configuration)
        _decoder.wrappedValue = NFKOobleckDecoder(configuration)
        super.init()
        train(false)
    }

    /// The posterior of `waveform` `[B, samples, audioChannels]`: its mean and standard deviation, each
    /// `[B, samples / hop, latentChannels]`.
    public func encode(_ waveform: MLXArray) -> (mean: MLXArray, standardDeviation: MLXArray) {
        let moments = encoder(waveform)
        let width = configuration.latentChannels
        let mean = moments[.ellipsis, 0 ..< width]
        let scale = moments[.ellipsis, width...]
        return (mean, softplus(scale) + 1e-4)
    }

    /// A latent drawn from the posterior of `waveform` with `noise`, a standard normal of the latent's shape:
    /// `mean + std · noise`.
    public func sample(_ waveform: MLXArray, noise: MLXArray) -> MLXArray {
        let posterior = encode(waveform)
        return posterior.mean + posterior.standardDeviation * noise
    }

    /// Latents `[B, frames, latentChannels]` → waveform `[B, frames · hop, audioChannels]`.
    public func decode(_ latents: MLXArray) -> MLXArray {
        decoder(latents)
    }
}

// MARK: - Diffusion transformer

/// The Stable Audio diffusion transformer's geometry. Defaults are the released `transformer/config.json`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXStableAudioDiTConfiguration: Sendable, Equatable {
    public var sampleSize: Int
    public var inChannels: Int
    public var layers: Int
    public var headDim: Int
    public var heads: Int
    public var keyValueHeads: Int
    public var outChannels: Int
    public var crossAttentionDim: Int
    public var timeProjectionDim: Int
    public var globalStatesInputDim: Int
    public var crossAttentionInputDim: Int

    public init(sampleSize: Int = 1024, inChannels: Int = 64, layers: Int = 24, headDim: Int = 64, heads: Int = 24,
                keyValueHeads: Int = 12, outChannels: Int = 64, crossAttentionDim: Int = 768,
                timeProjectionDim: Int = 256, globalStatesInputDim: Int = 1536, crossAttentionInputDim: Int = 768) {
        self.sampleSize = sampleSize
        self.inChannels = inChannels
        self.layers = layers
        self.headDim = headDim
        self.heads = heads
        self.keyValueHeads = keyValueHeads
        self.outChannels = outChannels
        self.crossAttentionDim = crossAttentionDim
        self.timeProjectionDim = timeProjectionDim
        self.globalStatesInputDim = globalStatesInputDim
        self.crossAttentionInputDim = crossAttentionInputDim
    }

    /// The released Stable Audio Open 1.0 transformer.
    public static let release = NFKMLXStableAudioDiTConfiguration()

    /// A small configuration with the release's structure, for tests that need no weights.
    static let tiny = NFKMLXStableAudioDiTConfiguration(sampleSize: 16, inChannels: 4, layers: 2, headDim: 8, heads: 4,
                                                        keyValueHeads: 2, outChannels: 4, crossAttentionDim: 16,
                                                        timeProjectionDim: 8, globalStatesInputDim: 24,
                                                        crossAttentionInputDim: 12)

    /// The model width, `heads · headDim`.
    public var innerDim: Int { heads * headDim }

    /// The rotary width, half of each head, as the reference pipeline builds its table.
    public var rotaryDim: Int { headDim / 2 }

    /// The geometry a diffusers `transformer/config.json` states.
    static func configuration(fromHuggingFace url: URL) throws -> NFKMLXStableAudioDiTConfiguration {
        let json = try NFKMLXStableAudio.json(url)
        let base = NFKMLXStableAudioDiTConfiguration()
        return NFKMLXStableAudioDiTConfiguration(
            sampleSize: NFKMLXStableAudio.integer(json["sample_size"]) ?? base.sampleSize,
            inChannels: NFKMLXStableAudio.integer(json["in_channels"]) ?? base.inChannels,
            layers: NFKMLXStableAudio.integer(json["num_layers"]) ?? base.layers,
            headDim: NFKMLXStableAudio.integer(json["attention_head_dim"]) ?? base.headDim,
            heads: NFKMLXStableAudio.integer(json["num_attention_heads"]) ?? base.heads,
            keyValueHeads: NFKMLXStableAudio.integer(json["num_key_value_attention_heads"]) ?? base.keyValueHeads,
            outChannels: NFKMLXStableAudio.integer(json["out_channels"]) ?? base.outChannels,
            crossAttentionDim: NFKMLXStableAudio.integer(json["cross_attention_dim"]) ?? base.crossAttentionDim,
            timeProjectionDim: NFKMLXStableAudio.integer(json["time_proj_dim"]) ?? base.timeProjectionDim,
            globalStatesInputDim: NFKMLXStableAudio.integer(json["global_states_input_dim"]) ?? base.globalStatesInputDim,
            crossAttentionInputDim: NFKMLXStableAudio.integer(json["cross_attention_input_dim"]) ?? base.crossAttentionInputDim)
    }
}

/// The Gaussian Fourier timestep features (`StableAudioGaussianFourierProjection`, `log=False`,
/// `flip_sin_to_cos=True`): `[cos(2π·t·W), sin(2π·t·W)]` over a fixed `W` drawn from a standard normal, as the
/// reference initializes it.
final class NFKStableAudioFourierProjection: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(embeddingSize: Int) {
        _weight.wrappedValue = MLXRandom.normal([embeddingSize])
    }

    /// `t` `[N]` → `[N, 2 · embeddingSize]`. The reference scales `t` by `2π` before the product with `W`.
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let projected = (t.reshaped([-1, 1]) * Float(2 * Double.pi)) * weight.reshaped([1, -1])
        return concatenated([cos(projected), sin(projected)], axis: -1)
    }
}

/// A two-layer MLP with a SiLU between, as the transformer's `nn.Sequential` projections hold it
/// (`0` and `2` the linears, `1` the activation).
final class NFKStableAudioProjectionMLP: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(_ inputDim: Int, _ outputDim: Int, bias: Bool) {
        _layers.wrappedValue = [Linear(inputDim, outputDim, bias: bias), NFKSD3Marker(),
                                Linear(outputDim, outputDim, bias: bias)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (layers[2] as! Linear)(silu((layers[0] as! Linear)(x)))
    }
}

/// The SwiGLU feed-forward (diffusers' `FeedForward` with `activation_fn="swiglu"`): `net.0.proj` projects
/// to twice the inner width, the first half is multiplied by the SiLU of the second, and `net.2` projects
/// back.
final class NFKStableAudioFeedForward: Module {
    @ModuleInfo(key: "net") var net: [Module]

    init(dim: Int, inner: Int) {
        _net.wrappedValue = [NFKSD3GELUProj(dim, inner * 2), NFKSD3Marker(), Linear(inner, dim)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = (net[0] as! NFKSD3GELUProj)(x)
        let halves = split(projected, parts: 2, axis: -1)
        return (net[2] as! Linear)(halves[0] * silu(halves[1]))
    }
}

/// The transformer's attention (`StableAudioAttnProcessor2_0`): bias-free projections, grouped key-value
/// heads repeated to the query heads, and a rotary embedding on the leading `rotaryDim` channels of each
/// query and key head in the self-attention. The cross-attention takes no rotary.
final class NFKStableAudioAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]

    let heads: Int
    let keyValueHeads: Int
    let headDim: Int

    init(queryDim: Int, keyValueDim: Int, heads: Int, keyValueHeads: Int, headDim: Int) {
        self.heads = heads
        self.keyValueHeads = keyValueHeads
        self.headDim = headDim
        _toQ.wrappedValue = Linear(queryDim, heads * headDim, bias: false)
        _toK.wrappedValue = Linear(keyValueDim, keyValueHeads * headDim, bias: false)
        _toV.wrappedValue = Linear(keyValueDim, keyValueHeads * headDim, bias: false)
        _toOut.wrappedValue = [Linear(heads * headDim, queryDim, bias: false), NFKSD3Marker()]
    }

    /// `hidden` `[B, S, queryDim]`, `context` `[B, L, keyValueDim]` (nil for self-attention) → `[B, S, queryDim]`.
    /// `rotary` is the `(cos, sin)` table `[S, rotaryDim]`, applied to the queries and, in self-attention, the keys.
    func callAsFunction(_ hidden: MLXArray, context: MLXArray?, rotary: (cos: MLXArray, sin: MLXArray)?) -> MLXArray {
        let b = hidden.dim(0)
        let source = context ?? hidden
        func heads(_ t: MLXArray, _ count: Int) -> MLXArray {
            t.reshaped([b, -1, count, headDim]).transposed(0, 2, 1, 3)
        }
        var q = heads(toQ(hidden), self.heads)
        var k = heads(toK(source), keyValueHeads)
        var v = heads(toV(source), keyValueHeads)
        if keyValueHeads != self.heads {
            let repeats = self.heads / keyValueHeads
            k = MLX.repeated(k, count: repeats, axis: 1)
            v = MLX.repeated(v, count: repeats, axis: 1)
        }
        if let rotary {
            q = NFKMLXStableAudio.rotated(q, rotary)
            if context == nil {
                k = NFKMLXStableAudio.rotated(k, rotary)
            }
        }
        let attended = NFKReferenceRounding.flashAttention(queries: q, keys: k, values: v,
                                                           scale: 1 / sqrt(Float(headDim)), mask: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([b, -1, self.heads * headDim])
        return (toOut[0] as! Linear)(merged)
    }
}

/// One transformer block (`StableAudioDiTBlock`): pre-norm self-attention with the rotary, pre-norm
/// cross-attention over the conditioning, and a pre-norm SwiGLU feed-forward, each added back.
final class NFKStableAudioDiTBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: NFKLayerNorm
    @ModuleInfo(key: "attn1") var attn1: NFKStableAudioAttention
    @ModuleInfo(key: "norm2") var norm2: NFKLayerNorm
    @ModuleInfo(key: "attn2") var attn2: NFKStableAudioAttention
    @ModuleInfo(key: "norm3") var norm3: NFKLayerNorm
    @ModuleInfo(key: "ff") var ff: NFKStableAudioFeedForward

    init(_ c: NFKMLXStableAudioDiTConfiguration) {
        let dim = c.innerDim
        _norm1.wrappedValue = NFKLayerNorm(dimensions: dim, eps: 1e-5)
        _attn1.wrappedValue = NFKStableAudioAttention(queryDim: dim, keyValueDim: dim, heads: c.heads,
                                                      keyValueHeads: c.heads, headDim: c.headDim)
        _norm2.wrappedValue = NFKLayerNorm(dimensions: dim, eps: 1e-5)
        _attn2.wrappedValue = NFKStableAudioAttention(queryDim: dim, keyValueDim: c.crossAttentionDim, heads: c.heads,
                                                      keyValueHeads: c.keyValueHeads, headDim: c.headDim)
        _norm3.wrappedValue = NFKLayerNorm(dimensions: dim, eps: 1e-5)
        _ff.wrappedValue = NFKStableAudioFeedForward(dim: dim, inner: dim * 4)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray, rotary: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        var hidden = attn1(norm1(x), context: nil, rotary: rotary) + x
        hidden = attn2(norm2(hidden), context: context, rotary: nil) + hidden
        return ff(norm3(hidden)) + hidden
    }
}

/// The Stable Audio diffusion transformer (`StableAudioDiTModel`).
///
/// @discussion The noisy latent `[B, frames, 64]` passes a residual 1×1 convolution (held as a `Linear`, which
/// LoRA adapts as the reference adapts its `Conv1d`) and a projection to the
/// model width. The global token (the two seconds conditioners' embeddings, projected, plus the Gaussian
/// Fourier timestep embedding) is prepended, so the blocks run over `frames + 1` positions with a
/// half-head rotary over all of them. The cross-attention reads the projected text-and-seconds sequence.
/// After the blocks the global token is dropped, and a projection and a residual 1×1 convolution return
/// the prediction `[B, frames, 64]`, a v-prediction under the EDM preconditioning
/// ``NFKMLXCosineDPMSolverScheduler`` applies.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXStableAudioDiTNet: Module {
    @ModuleInfo(key: "time_proj") var timeProj: NFKStableAudioFourierProjection
    @ModuleInfo(key: "timestep_proj") var timestepProj: NFKStableAudioProjectionMLP
    @ModuleInfo(key: "global_proj") var globalProj: NFKStableAudioProjectionMLP
    @ModuleInfo(key: "cross_attention_proj") var crossAttentionProj: NFKStableAudioProjectionMLP
    @ModuleInfo(key: "preprocess_conv") var preprocessConv: Linear
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [NFKStableAudioDiTBlock]
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "postprocess_conv") var postprocessConv: Linear

    public let configuration: NFKMLXStableAudioDiTConfiguration

    public init(_ c: NFKMLXStableAudioDiTConfiguration = .release) {
        configuration = c
        _timeProj.wrappedValue = NFKStableAudioFourierProjection(embeddingSize: c.timeProjectionDim / 2)
        _timestepProj.wrappedValue = NFKStableAudioProjectionMLP(c.timeProjectionDim, c.innerDim, bias: true)
        _globalProj.wrappedValue = NFKStableAudioProjectionMLP(c.globalStatesInputDim, c.innerDim, bias: false)
        _crossAttentionProj.wrappedValue = NFKStableAudioProjectionMLP(c.crossAttentionInputDim, c.crossAttentionDim,
                                                                       bias: false)
        _preprocessConv.wrappedValue = Linear(c.inChannels, c.inChannels, bias: false)
        _projIn.wrappedValue = Linear(c.inChannels, c.innerDim, bias: false)
        _transformerBlocks.wrappedValue = (0 ..< c.layers).map { _ in NFKStableAudioDiTBlock(c) }
        _projOut.wrappedValue = Linear(c.innerDim, c.outChannels, bias: false)
        _postprocessConv.wrappedValue = Linear(c.outChannels, c.outChannels, bias: false)
    }

    /// The prediction for `latents` `[B, frames, inChannels]` at `timestep` `[1]` or `[B]` (the preconditioned
    /// noise level, `atan(σ) · 2/π`), reading the text-and-seconds sequence `context`
    /// `[B, L, crossAttentionInputDim]` and the seconds embedding `global` `[B, 1, globalStatesInputDim]`.
    public func callAsFunction(_ latents: MLXArray, timestep: MLXArray, context: MLXArray,
                               global: MLXArray) -> MLXArray {
        let crossAttention = crossAttentionProj(context)
        let time = timestepProj(timeProj(timestep.asType(latents.dtype)))
        let globalToken = globalProj(global) + time.expandedDimensions(axis: 1)
        var hidden = projIn(preprocessConv(latents) + latents)
        hidden = concatenated([globalToken, hidden], axis: 1)
        let rotary = NFKMLXStableAudio.rotaryTable(length: hidden.dim(1), dimensions: configuration.rotaryDim)
        for block in transformerBlocks {
            hidden = block(hidden, context: crossAttention, rotary: rotary)
        }
        let output = projOut(hidden)[0..., 1...]
        return postprocessConv(output) + output
    }
}

// MARK: - Conditioning

/// The seconds conditioners' geometry, from the released `projection_model/config.json`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXStableAudioProjectionConfiguration: Sendable, Equatable {
    public var textEncoderDim: Int
    public var conditioningDim: Int
    public var minValue: Float
    public var maxValue: Float

    public init(textEncoderDim: Int = 768, conditioningDim: Int = 768, minValue: Float = 0, maxValue: Float = 512) {
        self.textEncoderDim = textEncoderDim
        self.conditioningDim = conditioningDim
        self.minValue = minValue
        self.maxValue = maxValue
    }

    /// The released Stable Audio Open 1.0 projection model.
    public static let release = NFKMLXStableAudioProjectionConfiguration()

    static let tiny = NFKMLXStableAudioProjectionConfiguration(textEncoderDim: 10, conditioningDim: 12, minValue: 0,
                                                               maxValue: 64)

    static func configuration(fromHuggingFace url: URL) throws -> NFKMLXStableAudioProjectionConfiguration {
        let json = try NFKMLXStableAudio.json(url)
        let base = NFKMLXStableAudioProjectionConfiguration()
        return NFKMLXStableAudioProjectionConfiguration(
            textEncoderDim: NFKMLXStableAudio.integer(json["text_encoder_dim"]) ?? base.textEncoderDim,
            conditioningDim: NFKMLXStableAudio.integer(json["conditioning_dim"]) ?? base.conditioningDim,
            minValue: (json["min_value"] as? NSNumber)?.floatValue ?? base.minValue,
            maxValue: (json["max_value"] as? NSNumber)?.floatValue ?? base.maxValue)
    }
}

/// The learned Fourier features of a normalized number (`StableAudioPositionalEmbedding`):
/// `[t, sin(2π·t·w), cos(2π·t·w)]`, `w` initialized from a standard normal as the reference initializes it.
final class NFKStableAudioPositionalEmbedding: Module {
    @ParameterInfo(key: "weights") var weights: MLXArray

    init(dim: Int) {
        _weights.wrappedValue = MLXRandom.normal([dim / 2])
    }

    func callAsFunction(_ times: MLXArray) -> MLXArray {
        let t = times.expandedDimensions(axis: -1)
        let frequencies = t * weights.reshaped([1, -1]) * 2 * Float.pi
        return concatenated([t, sin(frequencies), cos(frequencies)], axis: -1)
    }
}

/// A seconds conditioner (`StableAudioNumberConditioner`): the value is clamped to the conditioner's range,
/// normalized to `[0, 1]`, expanded into learned Fourier features, and projected to one conditioning token.
final class NFKStableAudioNumberConditioner: Module {
    @ModuleInfo(key: "time_positional_embedding") var timePositionalEmbedding: [Module]

    let minValue: Float
    let maxValue: Float

    init(_ c: NFKMLXStableAudioProjectionConfiguration, internalDim: Int = 256) {
        minValue = c.minValue
        maxValue = c.maxValue
        _timePositionalEmbedding.wrappedValue = [NFKStableAudioPositionalEmbedding(dim: internalDim),
                                                 Linear(internalDim + 1, c.conditioningDim)]
    }

    /// Seconds `[B]` → one token per value, `[B, 1, conditioningDim]`.
    func callAsFunction(_ seconds: MLXArray) -> MLXArray {
        let normalized = (clip(seconds, min: minValue, max: maxValue) - minValue) / (maxValue - minValue)
        let features = (timePositionalEmbedding[0] as! NFKStableAudioPositionalEmbedding)(normalized)
        return (timePositionalEmbedding[1] as! Linear)(features).expandedDimensions(axis: 1)
    }
}

/// The projection model (`StableAudioProjectionModel`): the text states' projection (the identity where
/// the text encoder's width is the conditioning width, as in the release) and the start and end seconds
/// conditioners.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXStableAudioProjectionNet: Module {
    @ModuleInfo(key: "text_projection") var textProjection: Linear?
    @ModuleInfo(key: "start_number_conditioner") var startNumberConditioner: NFKStableAudioNumberConditioner
    @ModuleInfo(key: "end_number_conditioner") var endNumberConditioner: NFKStableAudioNumberConditioner

    public let configuration: NFKMLXStableAudioProjectionConfiguration

    public init(_ c: NFKMLXStableAudioProjectionConfiguration = .release) {
        configuration = c
        if c.textEncoderDim != c.conditioningDim {
            _textProjection.wrappedValue = Linear(c.textEncoderDim, c.conditioningDim)
        }
        _startNumberConditioner.wrappedValue = NFKStableAudioNumberConditioner(c)
        _endNumberConditioner.wrappedValue = NFKStableAudioNumberConditioner(c)
    }

    /// The text states `[B, L, textEncoderDim]` in the conditioning width.
    public func projectedText(_ states: MLXArray) -> MLXArray {
        textProjection.map { $0(states) } ?? states
    }

    /// The start and end tokens for `start` and `end` seconds `[B]`, each `[B, 1, conditioningDim]`.
    public func seconds(start: MLXArray, end: MLXArray) -> (start: MLXArray, end: MLXArray) {
        (startNumberConditioner(start), endNumberConditioner(end))
    }
}

// MARK: - Sampler

/// The cosine-schedule DPM-Solver++ SDE sampler (`CosineDPMSolverMultistepScheduler`), as Stable Audio Open
/// samples.
///
/// @discussion The noise levels run exponentially (or on the Karras ramp) from `sigmaMax` to `sigmaMin`,
/// then to zero. The model reads `x · c_in` with `c_in = 1/√(σ² + σ_data²)` at the timestep `atan(σ)·2/π`,
/// and its v-prediction becomes the denoised estimate `c_skip·x + c_out·v`. A step is the second-order
/// multistep SDE update (midpoint or Heun), first order on the first step and on the last, which lands on
/// zero noise. Each step draws a standard normal of the latent's shape, the normalized increment of the
/// reference's Brownian tree over that step's interval.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXCosineDPMSolverScheduler {

    /// The sampler's configuration, the released `scheduler/scheduler_config.json`.
    public struct Configuration: Sendable, Equatable {
        public var sigmaMin: Float = 0.3
        public var sigmaMax: Float = 500
        public var sigmaData: Float = 1
        /// The exponential ramp (the release's) or the Karras ramp.
        public var karras = false
        public var solverOrder = 2
        public var rho: Float = 7
        /// The midpoint second-order update (the release's) or Heun's.
        public var heun = false
        public var lowerOrderFinal = true
        public var eulerAtFinal = false
        /// The last noise level is zero (the release's) or `sigmaMin`.
        public var finalSigmaIsZero = true

        public init() {}

        /// The configuration a diffusers `scheduler_config.json` states. Only the v-prediction the release
        /// trains is read.
        static func configuration(fromHuggingFace url: URL) throws -> Configuration {
            let json = try NFKMLXStableAudio.json(url)
            var c = Configuration()
            let float = { (key: String, fallback: Float) in (json[key] as? NSNumber)?.floatValue ?? fallback }
            c.sigmaMin = float("sigma_min", c.sigmaMin)
            c.sigmaMax = float("sigma_max", c.sigmaMax)
            c.sigmaData = float("sigma_data", c.sigmaData)
            c.rho = float("rho", c.rho)
            c.karras = (json["sigma_schedule"] as? String) == "karras"
            c.solverOrder = NFKMLXStableAudio.integer(json["solver_order"]) ?? c.solverOrder
            c.heun = (json["solver_type"] as? String) == "heun"
            c.lowerOrderFinal = json["lower_order_final"] as? Bool ?? c.lowerOrderFinal
            c.eulerAtFinal = json["euler_at_final"] as? Bool ?? c.eulerAtFinal
            c.finalSigmaIsZero = (json["final_sigmas_type"] as? String ?? "zero") == "zero"
            if let prediction = json["prediction_type"] as? String, prediction != "v_prediction" {
                throw NFKMLXError.unsupportedConfiguration("Stable Audio samples a v-prediction, not \(prediction)")
            }
            guard (1 ... 2).contains(c.solverOrder) else {
                throw NFKMLXError.unsupportedConfiguration("solver order \(c.solverOrder)")
            }
            return c
        }
    }

    public let configuration: Configuration
    /// The noise levels, `steps + 1` of them, the last the final level.
    public let sigmas: [Float]
    /// The timesteps the model reads, `atan(σ)·2/π` for each of the first `steps` levels.
    public let timesteps: [Float]

    private var modelOutputs: [MLXArray?]
    private var lowerOrderCount = 0
    /// The step the next ``step(modelOutput:sample:noise:)`` takes.
    public private(set) var stepIndex = 0

    public init(steps: Int, configuration: Configuration = Configuration()) {
        precondition(steps > 0, "a sampler takes at least one step")
        self.configuration = configuration
        let schedule = Self.noiseLevels(steps: steps, configuration: configuration)
        sigmas = schedule + [configuration.finalSigmaIsZero ? 0 : configuration.sigmaMin]
        timesteps = schedule.map { Foundation.atan($0) / Float.pi * 2 }
        modelOutputs = Array(repeating: nil, count: configuration.solverOrder)
    }

    /// The `steps` levels from `sigmaMax` down to `sigmaMin`, over torch's float32 `linspace`.
    static func noiseLevels(steps: Int, configuration c: Configuration) -> [Float] {
        let ramp = linspace(0, 1, steps)
        if c.karras {
            let minInverse = Foundation.pow(c.sigmaMin, 1 / c.rho), maxInverse = Foundation.pow(c.sigmaMax, 1 / c.rho)
            return ramp.map { Foundation.pow(maxInverse + $0 * (minInverse - maxInverse), c.rho) }
        }
        return linspace(Float(Foundation.log(Double(c.sigmaMin))), Float(Foundation.log(Double(c.sigmaMax))), steps)
            .map { Foundation.exp($0) }.reversed()
    }

    /// torch's float32 `linspace`: the first half counts up from `start` and the second half down from `end`.
    static func linspace(_ start: Float, _ end: Float, _ count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        let halfway = count / 2
        return (0 ..< count).map { index in
            index < halfway ? start + step * Float(index) : end - step * Float(count - index - 1)
        }
    }

    /// The noise level of the step about to run.
    public var sigma: Float { sigmas[stepIndex] }

    /// The model's input scale at the current step, `1/√(σ² + σ_data²)`.
    public var inputScale: Float {
        let s = sigma, d = configuration.sigmaData
        return 1 / Foundation.sqrt(s * s + d * d)
    }

    /// The model's timestep at the current step.
    public var timestep: Float { timesteps[stepIndex] }

    /// The denoised estimate from the v-prediction `modelOutput` at `sample` and the current level.
    func denoised(_ modelOutput: MLXArray, sample: MLXArray) -> MLXArray {
        let s = sigma, d = configuration.sigmaData
        let skip = d * d / (s * s + d * d)
        let out = -s * d / Foundation.sqrt(s * s + d * d)
        return skip * sample + out * modelOutput
    }

    /// Advances `sample` one step given the model's prediction at the current level and `noise`, a
    /// standard normal of the sample's shape.
    public func step(modelOutput: MLXArray, sample: MLXArray, noise: MLXArray) -> MLXArray {
        let count = timesteps.count
        let lowerOrderFinal = stepIndex == count - 1
            && (configuration.eulerAtFinal || (configuration.lowerOrderFinal && count < 15)
                || configuration.finalSigmaIsZero)
        let estimate = denoised(modelOutput, sample: sample)
        modelOutputs.removeFirst()
        modelOutputs.append(estimate)

        let next: MLXArray
        if configuration.solverOrder == 1 || lowerOrderCount < 1 || lowerOrderFinal {
            next = firstOrderUpdate(estimate, sample: sample, noise: noise)
        } else {
            next = secondOrderUpdate(estimate, previous: modelOutputs[modelOutputs.count - 2]!, sample: sample,
                                     noise: noise)
        }
        if lowerOrderCount < configuration.solverOrder {
            lowerOrderCount += 1
        }
        stepIndex += 1
        return next
    }

    /// `λ = −log σ`, the log signal-to-noise ratio with the signal scale 1.
    private static func lambda(_ sigma: Float) -> Float { -Foundation.log(sigma) }

    private func firstOrderUpdate(_ estimate: MLXArray, sample: MLXArray, noise: MLXArray) -> MLXArray {
        let sigmaT = sigmas[stepIndex + 1], sigmaS = sigmas[stepIndex]
        let h = Self.lambda(sigmaT) - Self.lambda(sigmaS)
        let decay = Foundation.exp(-h), decay2 = Foundation.exp(-2 * h)
        return (sigmaT / sigmaS * decay) * sample + (1 - decay2) * estimate
            + sigmaT * Foundation.sqrt(1 - decay2) * noise
    }

    private func secondOrderUpdate(_ estimate: MLXArray, previous: MLXArray, sample: MLXArray,
                                   noise: MLXArray) -> MLXArray {
        let sigmaT = sigmas[stepIndex + 1], sigmaS0 = sigmas[stepIndex], sigmaS1 = sigmas[stepIndex - 1]
        let lambdaS0 = Self.lambda(sigmaS0)
        let h = Self.lambda(sigmaT) - lambdaS0, h0 = lambdaS0 - Self.lambda(sigmaS1)
        let r0 = h0 / h
        let d1 = (1 / r0) * (estimate - previous)
        let decay = Foundation.exp(-h), decay2 = Foundation.exp(-2 * h)
        let secondOrder: MLXArray
        if configuration.heun {
            secondOrder = ((1 - decay2) / (-2 * h) + 1) * d1
        } else {
            secondOrder = 0.5 * (1 - decay2) * d1
        }
        return (sigmaT / sigmaS0 * decay) * sample + (1 - decay2) * estimate + secondOrder
            + sigmaT * Foundation.sqrt(1 - decay2) * noise
    }
}

// MARK: - Shared helpers and loaders

/// Stable Audio Open's shared arithmetic and weight loaders.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXStableAudio {

    /// A JSON integer, which a release may state as a float (`"sample_size": 1024.0`).
    static func integer(_ value: Any?) -> Int? {
        (value as? NSNumber)?.intValue
    }

    static func integers(_ value: Any?) -> [Int]? {
        (value as? [NSNumber])?.map(\.intValue)
    }

    static func json(_ url: URL) throws -> [String: Any] {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not an object")
        }
        return json
    }

    /// The rotary table the pipeline builds (`get_1d_rotary_pos_embed`, `use_real=True`,
    /// `repeat_interleave_real=False`): positions `0 ..< length` against `10000^(−2i/dimensions)`, each of
    /// `cos` and `sin` `[length, dimensions]` with the half-width table repeated.
    static func rotaryTable(length: Int, dimensions: Int) -> (cos: MLXArray, sin: MLXArray) {
        let exponents = MLXArray(stride(from: 0, to: dimensions, by: 2).map(Float.init)) / Float(dimensions)
        let frequencies = 1 / MLX.pow(MLXArray(Float(10000)), exponents)
        let positions = MLXArray((0 ..< length).map(Float.init)).reshaped([-1, 1])
        let angles = positions * frequencies.reshaped([1, -1])
        let c = cos(angles), s = sin(angles)
        return (concatenated([c, c], axis: -1), concatenated([s, s], axis: -1))
    }

    /// `x` `[B, H, S, D]` with its leading `rotaryDim` channels rotated half against half, in float32 and
    /// rounded back once, as diffusers' `apply_rotary_emb` with `use_real_unbind_dim=-2` rotates them.
    static func rotated(_ x: MLXArray, _ rotary: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let width = rotary.cos.dim(-1), half = width / 2
        let rotating = x[.ellipsis, 0 ..< width].asType(.float32)
        let real = rotating[.ellipsis, 0 ..< half], imaginary = rotating[.ellipsis, half...]
        let turned = concatenated([-imaginary, real], axis: -1)
        let result = (rotating * rotary.cos + turned * rotary.sin).asType(x.dtype)
        return concatenated([result, x[.ellipsis, width...]], axis: -1)
    }

    /// Collapses weight-normalized pairs, transposes the convolutions and the Snake parameters from
    /// PyTorch's layout, and nests each transposed convolution under its wrapper, for an
    /// `AutoencoderOobleck` checkpoint. A fine-tuned save is already in this layout and loads as stored.
    static func autoencoderArrays(_ arrays: [String: MLXArray], fromPyTorch: Bool = true) -> [(String, MLXArray)] {
        guard fromPyTorch else {
            return arrays.map { ($0, $1) }
        }
        return NFKMLXMusic3.fusedWeightNorm(arrays).map { key, value -> (String, MLXArray) in
            if key.hasSuffix(".alpha") || key.hasSuffix(".beta") {
                return (key, value.transposed(0, 2, 1))
            }
            if key.contains("conv_t1.") {
                let name = key.replacingOccurrences(of: "conv_t1.", with: "conv_t1.conv.")
                return (name, value.ndim == 3 ? value.transposed(1, 2, 0).expandedDimensions(axis: 2) : value)
            }
            return (key, value.ndim == 3 ? value.transposed(0, 2, 1) : value)
        }
    }

    /// Loads an `AutoencoderOobleck` checkpoint (the release's `vae/diffusion_pytorch_model.safetensors`, or
    /// a fine-tuned save) into `net`.
    public static func loadAutoencoderWeights(into net: NFKMLXOobleckAutoencoderNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        try NFKMLXWeights.apply(autoencoderArrays(checkpoint.arrays, fromPyTorch: checkpoint.needsConvTranspose), to: net,
                                verifyShapes: true)
    }

    /// The transformer's `nn.Sequential` projections hold their linears at `0` and `2`; this module holds
    /// them under `layers`, and its 1×1 convolutions as linears. A file this module saved is already in
    /// both forms: its projection keys name `layers`, and its convolution weights are `[out, in]`.
    static func transformerArrays(_ arrays: [(String, MLXArray)]) -> [(String, MLXArray)] {
        arrays.map { key, value -> (String, MLXArray) in
            var name = key
            for projection in ["timestep_proj", "global_proj", "cross_attention_proj"]
            where key.hasPrefix(projection + ".") && !key.hasPrefix(projection + ".layers.") {
                name = projection + ".layers." + key.dropFirst(projection.count + 1)
            }
            let convolution = key.hasPrefix("preprocess_conv.") || key.hasPrefix("postprocess_conv.")
            return (name, convolution && value.ndim == 3 ? value.squeezed(axis: 2) : value)
        }
    }

    /// Loads a `StableAudioDiTModel` checkpoint directory (the release's `transformer/`) into `net` at `dtype`.
    public static func loadTransformerWeights(into net: NFKMLXStableAudioDiTNet, fromDirectory directory: URL,
                                              dtype: DType = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, converting: dtype)
        try NFKMLXWeights.apply(transformerArrays(arrays), to: net, verifyShapes: true)
    }

    /// The projection model's conditioners hold their `nn.Sequential` under `time_positional_embedding`
    /// unchanged; only a file from another layout would need renaming.
    public static func loadProjectionWeights(into net: NFKMLXStableAudioProjectionNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        try NFKMLXWeights.apply(checkpoint.arrays.map { ($0, $1) }, to: net, verifyShapes: true)
    }
}
