//
//  NFKMLXMusic3.swift
//  InferKitMLX
//
//  MiniMax Music 3, stage 1 of the port: the Flow-VAE waveform decoder (a DAC-style Snake vocoder).
//  The full model is a hybrid — a Qwen3 autoregressive stage over RVQ codes conditions a flow-matching
//  DiT through fused hidden states, and this decoder turns the DiT's latents into stereo audio. The
//  vocoder is a pure function of its latent, so it reaches measured parity on its own before any other
//  stage exists. The reference is diffusers' own `MiniMaxMusic3Vocoder` (>= 0.40.0), which is what the
//  released checkpoint's `_class_name` names.
//
//  The weights are NOT permissively licensed: the MiniMax-Music3 Community License requires UI
//  attribution in commercial products and separate authorization above a revenue threshold. See
//  Docs/companions.md.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

/// Vocoder sizing. Defaults are the released `vocoder/config.json`.
struct NFKMLXMusic3VocoderConfiguration: Sendable {
    var latentChannels: Int = 128
    var decoderInputDim: Int = 1024
    var decoderHiddenDim: Int = 1536
    var upsamplingRatios: [Int] = [8, 8, 4, 2]
    var samplingRate: Int = 44100
    init() {}

    /// Samples per latent frame (512 for the release).
    var hop: Int { upsamplingRatios.reduce(1, *) }

    /// A small configuration with the same structure, for tests that need no weights.
    static var tiny: NFKMLXMusic3VocoderConfiguration {
        var c = NFKMLXMusic3VocoderConfiguration()
        c.latentChannels = 16
        c.decoderInputDim = 16
        c.decoderHiddenDim = 32
        c.upsamplingRatios = [4, 2]
        return c
    }
}

/// The Snake activation: `x + (1/(α+1e-9))·sin²(αx)`, one learned α per channel.
///
/// The checkpoint stores α as `[1, C, 1]` (the reference runs NCL); this module holds it `[1, 1, C]`
/// so it broadcasts over NLC tensors, and the loader transposes a PyTorch-layout file.
final class NFKMusic3Snake: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.ones([1, 1, channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + (1.0 / (alpha + 1e-9)) * MLX.sin(alpha * x).square()
    }
}

/// A dilated residual unit: snake → 7-wide dilated conv → snake → 1×1 conv, added back to the input.
final class NFKMusic3ResidualUnit: Module {
    let snake1: NFKMusic3Snake
    let conv1: Conv1d
    let snake2: NFKMusic3Snake
    let conv2: Conv1d

    init(dim: Int, dilation: Int) {
        snake1 = NFKMusic3Snake(channels: dim)
        conv1 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7,
                       padding: (7 - 1) * dilation / 2, dilation: dilation)
        snake2 = NFKMusic3Snake(channels: dim)
        conv2 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + conv2(snake2(conv1(snake1(x))))
    }
}

/// One upsampling stage: snake → transposed conv (kernel 2·stride, so the length multiplies exactly
/// by the stride) → three residual units at dilations 1/3/9, halving the width.
final class NFKMusic3VocoderBlock: Module {
    let snake1: NFKMusic3Snake
    @ModuleInfo(key: "conv_t1") var convT1: NFKDemucsConvT1d
    @ModuleInfo(key: "res_unit1") var resUnit1: NFKMusic3ResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: NFKMusic3ResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: NFKMusic3ResidualUnit

    init(inputDim: Int, outputDim: Int, stride: Int) {
        snake1 = NFKMusic3Snake(channels: inputDim)
        _convT1.wrappedValue = NFKDemucsConvT1d(inputDim, outputDim, kernel: 2 * stride,
                                                stride: stride,
                                                padding: (stride + 1) / 2)
        _resUnit1.wrappedValue = NFKMusic3ResidualUnit(dim: outputDim, dilation: 1)
        _resUnit2.wrappedValue = NFKMusic3ResidualUnit(dim: outputDim, dilation: 3)
        _resUnit3.wrappedValue = NFKMusic3ResidualUnit(dim: outputDim, dilation: 9)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        resUnit3(resUnit2(resUnit1(convT1(snake1(x)))))
    }
}

/// The Flow-VAE waveform decoder: latents `[B, T, latentChannels]` → stereo `[B, T·hop, 2]` in
/// `-1...1`. Stereo comes from folding the latent's two channel halves into the batch and decoding
/// each as a mono stream.
final class NFKMusic3VocoderNet: Module {
    @ModuleInfo(key: "dec_in_proj") var decInProj: Conv1d
    @ModuleInfo(key: "conv_in") var convIn: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [NFKMusic3VocoderBlock]
    @ModuleInfo(key: "snake_out") var snakeOut: NFKMusic3Snake
    @ModuleInfo(key: "conv_out") var convOut: Conv1d

    let configuration: NFKMLXMusic3VocoderConfiguration

    init(_ configuration: NFKMLXMusic3VocoderConfiguration) {
        self.configuration = configuration
        _decInProj.wrappedValue = Conv1d(inputChannels: configuration.latentChannels / 2,
                                         outputChannels: configuration.decoderInputDim, kernelSize: 1)
        _convIn.wrappedValue = Conv1d(inputChannels: configuration.decoderInputDim,
                                      outputChannels: configuration.decoderHiddenDim,
                                      kernelSize: 7, padding: 3)
        var stages = [NFKMusic3VocoderBlock]()
        for (index, stride) in configuration.upsamplingRatios.enumerated() {
            stages.append(NFKMusic3VocoderBlock(inputDim: configuration.decoderHiddenDim >> index,
                                                outputDim: configuration.decoderHiddenDim >> (index + 1),
                                                stride: stride))
        }
        _blocks.wrappedValue = stages
        let outputDim = configuration.decoderHiddenDim >> configuration.upsamplingRatios.count
        _snakeOut.wrappedValue = NFKMusic3Snake(channels: outputDim)
        _convOut.wrappedValue = Conv1d(inputChannels: outputDim, outputChannels: 1,
                                       kernelSize: 7, padding: 3)
    }

    /// `[B, T, latentChannels]` → `[B, T·hop, 2]`.
    func waveform(_ latents: MLXArray) -> MLXArray {
        let (batch, length) = (latents.shape[0], latents.shape[1])
        let half = configuration.latentChannels / 2
        // The reference reshapes `[B, C, T]` to `[2B, C/2, T]`: channels 0..<half become the first
        // stereo stream and the rest the second, ordered per batch row.
        let folded = latents.reshaped([batch, length, 2, half])
            .transposed(0, 2, 1, 3)
            .reshaped([batch * 2, length, half])
        var x = convIn(decInProj(folded))
        for block in blocks {
            x = block(x)
        }
        let wave = tanh(convOut(snakeOut(x)))                       // [2B, samples, 1]
        return wave.reshaped([batch, 2, wave.shape[1]]).transposed(0, 2, 1)
    }
}

// MARK: - RVQ depth decoder

/// Depth decoder sizing. Defaults are the released `rvq_depth_decoder/config.json`.
struct NFKMLXMusic3DepthConfiguration: Sendable {
    var hiddenSize: Int = 4096
    var layerCount: Int = 4
    var headCount: Int = 16
    var intermediateSize: Int = 6144
    var audioVocabSize: Int = 1024
    var codebookCount: Int = 8
    var maxPositions: Int = 16
    init() {}

    static var tiny: NFKMLXMusic3DepthConfiguration {
        var c = NFKMLXMusic3DepthConfiguration()
        c.hiddenSize = 64
        c.layerCount = 2
        c.headCount = 4
        c.intermediateSize = 128
        c.audioVocabSize = 32
        return c
    }
}

final class NFKMusic3DepthAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    let heads: Int
    let headDimensions: Int

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        headDimensions = dimensions / heads
        _toQ.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _toK.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _toV.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _toOut.wrappedValue = Linear(dimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (batch, length, _) = (x.shape[0], x.shape[1], x.shape[2])
        func split(_ projected: MLXArray) -> MLXArray {
            projected.reshaped([batch, length, heads, headDimensions]).transposed(0, 2, 1, 3)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(toQ(x)), keys: split(toK(x)), values: split(toV(x)),
            scale: 1 / sqrt(Float(headDimensions)), mask: mask.asType(x.dtype))
        return toOut(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDimensions]))
    }
}

/// One depth block: pre-normalized causal attention and a SwiGLU feed-forward, each added back.
final class NFKMusic3DepthBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "attn") var attention: NFKMusic3DepthAttention
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: NFKMLXMusic3DepthConfiguration) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _attention.wrappedValue = NFKMusic3DepthAttention(dimensions: c.hiddenSize, heads: c.headCount)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let attended = x + attention(inputNorm(x), mask: mask)
        let normed = postNorm(attended)
        return attended + down(silu(gate(normed)) * up(normed))
    }
}

/// The local language model that fills the seven residual codebooks within each frame. Its position
/// embedding is a learned table of 16 entries, not a rotary — the depth sequence never exceeds one
/// global hidden state plus the codes of one frame. It also owns the offset-packed embedding table
/// for the residual codebooks (`code + (index − 1) · audioVocabSize`), which the AR feedback reads.
final class NFKMusic3DepthDecoderNet: Module {
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: Embedding
    @ModuleInfo(key: "projection") var projection: Linear
    @ModuleInfo(key: "pos_embedding") var posEmbedding: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKMusic3DepthBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "audio_heads") var audioHeads: [Linear]

    let configuration: NFKMLXMusic3DepthConfiguration

    init(_ configuration: NFKMLXMusic3DepthConfiguration) {
        self.configuration = configuration
        _audioEmbeddings.wrappedValue = Embedding(
            embeddingCount: configuration.audioVocabSize * (configuration.codebookCount - 1),
            dimensions: configuration.hiddenSize)
        _projection.wrappedValue = Linear(configuration.hiddenSize, configuration.hiddenSize, bias: false)
        _posEmbedding.wrappedValue = Embedding(embeddingCount: configuration.maxPositions,
                                               dimensions: configuration.hiddenSize)
        _layers.wrappedValue = (0 ..< configuration.layerCount).map { _ in NFKMusic3DepthBlock(configuration) }
        _norm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: 1e-6)
        _audioHeads.wrappedValue = (0 ..< configuration.codebookCount - 1).map { _ in
            Linear(configuration.hiddenSize, configuration.audioVocabSize, bias: false)
        }
    }

    /// `[B, steps, hidden]` (already projected) → normalized hidden states `[B, steps, hidden]`;
    /// the last step feeds the next codebook head.
    func hiddenStates(_ inputsEmbeds: MLXArray) -> MLXArray {
        let length = inputsEmbeds.shape[1]
        var hidden = inputsEmbeds + posEmbedding(MLXArray(0 ..< length)).expandedDimensions(axis: 0)
        let mask = NFKMLXLanguageNet.causalMask(length, offset: 0)
        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return norm(hidden)
    }
}

// MARK: - Condition encoder

/// Condition encoder sizing. Defaults are the released `condition_encoder/config.json`.
struct NFKMLXMusic3ConditionConfiguration: Sendable {
    var hiddenDimensions: Int = 4096
    var conditionLayers: Int = 8
    var outputDimensions: Int = 2048
    var inputSamplingRate: Int = 24000
    var inputHopLength: Int = 960
    var outputSamplingRate: Int = 44100
    var outputHopLength: Int = 512
    init() {}

    /// Latent frames a frame count maps to (the reference truncates, so 13 frames → 44 latents).
    func latentLength(forFrames frames: Int) -> Int {
        max(1, Int(Double(frames) * Double(outputSamplingRate) / Double(inputSamplingRate)
            * Double(inputHopLength) / Double(outputHopLength)))
    }

    static var tiny: NFKMLXMusic3ConditionConfiguration {
        var c = NFKMLXMusic3ConditionConfiguration()
        c.hiddenDimensions = 4
        c.outputDimensions = 8
        return c
    }
}

/// Projects the AR stage's fused per-frame hidden states onto the latent timeline: a learned softmax
/// blend of the 8 per-codebook hidden states, a scalar gain, a 3-wide convolution, and a
/// nearest-neighbor resample from the 25 fps frame rate to the latent rate (≈ 3.445 latents a frame).
final class NFKMusic3ConditionEncoderNet: Module {
    @ParameterInfo(key: "layer_weight_logits") var layerWeightLogits: MLXArray
    @ParameterInfo(key: "layer_scale") var layerScale: MLXArray
    @ModuleInfo(key: "proj") var proj: Conv1d

    let configuration: NFKMLXMusic3ConditionConfiguration

    init(_ configuration: NFKMLXMusic3ConditionConfiguration) {
        self.configuration = configuration
        _layerWeightLogits.wrappedValue = MLXArray.zeros([configuration.conditionLayers])
        _layerScale.wrappedValue = MLXArray.ones([1])
        _proj.wrappedValue = Conv1d(inputChannels: configuration.hiddenDimensions,
                                    outputChannels: configuration.outputDimensions,
                                    kernelSize: 3, padding: 1)
    }

    /// `[B, frames, conditionLayers · hiddenDimensions]` → `[B, latentLength, outputDimensions]`.
    func condition(_ hidden: MLXArray) -> MLXArray {
        let (batch, frames) = (hidden.shape[0], hidden.shape[1])
        let stacked = hidden.reshaped([batch, frames, configuration.conditionLayers,
                                       configuration.hiddenDimensions])
        let weights = softmax(layerWeightLogits, axis: 0).reshaped([1, 1, configuration.conditionLayers, 1])
        let projected = proj(layerScale * (stacked * weights).sum(axis: 2))

        // PyTorch's nearest interpolation: source index = floor(target · scale), with the scale
        // taken at float precision, which is the arithmetic the trained resample rode on.
        let latentLength = configuration.latentLength(forFrames: frames)
        let scale = Float(frames) / Float(latentLength)
        let indices = (0 ..< latentLength).map { Int32(min(Int(scale * Float($0)), frames - 1)) }
        return projected.take(MLXArray(indices), axis: 1)
    }
}

// MARK: - Flow-matching DiT

/// DiT sizing. Defaults are the released `transformer/config.json`.
struct NFKMLXMusic3DiTConfiguration: Sendable {
    var latentChannels: Int = 128
    var conditionDimensions: Int = 2048
    var layerCount: Int = 36
    var headCount: Int = 32
    var headDimensions: Int = 64
    var feedForwardInnerDimensions: Int = 8192
    var rotaryDimensions: Int = 32
    var fourierDimensions: Int = 256
    init() {}

    var innerDimensions: Int { headCount * headDimensions }
    var concatChannels: Int { 2 * latentChannels + conditionDimensions }

    static var tiny: NFKMLXMusic3DiTConfiguration {
        var c = NFKMLXMusic3DiTConfiguration()
        c.latentChannels = 8
        c.conditionDimensions = 16
        c.layerCount = 2
        c.headCount = 4
        c.headDimensions = 8
        c.feedForwardInnerDimensions = 32
        c.rotaryDimensions = 4
        c.fourierDimensions = 8
        return c
    }
}

/// Random Fourier features over the flow-matching time in `0...1`; the projection is a trained
/// checkpoint weight, not a fixed table.
final class NFKMusic3FourierEmbedding: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int) {
        // The reference initializes with `torch.randn`; zeros would make every timestep encode
        // identically until a checkpoint loads. The draw is KEYED: an unkeyed one advances the
        // global RNG at module construction, which shifts the stream under every unseeded test
        // that runs later in the process.
        _weight.wrappedValue = MLXRandom.normal([dimensions / 2, 1], key: MLXRandom.key(0x4655))
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let angles = 2 * Float.pi * matmul(timestep.reshaped([-1, 1]), weight.transposed(1, 0))
        return concatenated([cos(angles), MLX.sin(angles)], axis: -1)
    }
}

/// The diffusers `TimestepEmbedding`: linear → silu → linear.
final class NFKMusic3TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputDimensions: Int, dimensions: Int) {
        _linear1.wrappedValue = Linear(inputDimensions, dimensions)
        _linear2.wrappedValue = Linear(dimensions, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

final class NFKMusic3DiTAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    let heads: Int
    let headDimensions: Int

    init(_ c: NFKMLXMusic3DiTConfiguration) {
        heads = c.headCount
        headDimensions = c.headDimensions
        _toQ.wrappedValue = Linear(c.innerDimensions, c.innerDimensions, bias: false)
        _toK.wrappedValue = Linear(c.innerDimensions, c.innerDimensions, bias: false)
        _toV.wrappedValue = Linear(c.innerDimensions, c.innerDimensions, bias: false)
        // The reference wraps the output projection with a Dropout in a ModuleList, so the
        // checkpoint key is `to_out.0`.
        _toOut.wrappedValue = [Linear(c.innerDimensions, c.innerDimensions, bias: false)]
    }

    /// Partial rotary: only the leading `rotaryDimensions` of each head turn, rotate-half convention.
    private func rotated(_ x: MLXArray, cosines: MLXArray, sines: MLXArray) -> MLXArray {
        let rotaryDimensions = cosines.shape.last!
        let leading = x[.ellipsis, ..<rotaryDimensions]
        let half = rotaryDimensions / 2
        let rotateHalf = concatenated([-leading[.ellipsis, half...], leading[.ellipsis, ..<half]], axis: -1)
        return concatenated([leading * cosines + rotateHalf * sines,
                             x[.ellipsis, rotaryDimensions...]], axis: -1)
    }

    func callAsFunction(_ x: MLXArray, cosines: MLXArray, sines: MLXArray) -> MLXArray {
        let (batch, length, _) = (x.shape[0], x.shape[1], x.shape[2])
        func split(_ projected: MLXArray) -> MLXArray {
            rotated(projected.reshaped([batch, length, heads, headDimensions]),
                    cosines: cosines, sines: sines).transposed(0, 2, 1, 3)
        }
        let values = toV(x).reshaped([batch, length, heads, headDimensions]).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(toQ(x)), keys: split(toK(x)), values: values,
            scale: 1 / sqrt(Float(headDimensions)), mask: nil)
        return toOut[0](attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDimensions]))
    }
}

/// One DiT block: pre-LayerNorm attention, then a gated feed-forward whose `ff_in` emits value and
/// gate halves (`value · silu(gate)`), each added back.
final class NFKMusic3DiTBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attention: NFKMusic3DiTAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ff_in") var feedForwardIn: Linear
    @ModuleInfo(key: "ff_out") var feedForwardOut: Linear

    let innerDimensions: Int

    init(_ c: NFKMLXMusic3DiTConfiguration) {
        innerDimensions = c.feedForwardInnerDimensions
        _norm1.wrappedValue = LayerNorm(dimensions: c.innerDimensions)
        _attention.wrappedValue = NFKMusic3DiTAttention(c)
        _norm2.wrappedValue = LayerNorm(dimensions: c.innerDimensions)
        _feedForwardIn.wrappedValue = Linear(c.innerDimensions, 2 * c.feedForwardInnerDimensions)
        _feedForwardOut.wrappedValue = Linear(c.feedForwardInnerDimensions, c.innerDimensions)
    }

    func callAsFunction(_ x: MLXArray, cosines: MLXArray, sines: MLXArray) -> MLXArray {
        let attended = x + attention(norm1(x), cosines: cosines, sines: sines)
        let projected = feedForwardIn(norm2(attended))
        let value = projected[.ellipsis, ..<innerDimensions]
        let gate = projected[.ellipsis, innerDimensions...]
        return attended + feedForwardOut(value * silu(gate))
    }
}

/// The flow-matching transformer: noisy latents + the frame-aligned condition → a velocity. The
/// timestep enters as random Fourier features prepended as token 0 of the sequence (stripped after
/// the blocks), and the input concatenates `[latent, zeros(latentChannels), condition]` on the
/// channel axis — the zeroed block is the reference's unfilled audio-prompt slot.
final class NFKMusic3DiTNet: Module {
    @ModuleInfo(key: "time_proj") var timeProj: NFKMusic3FourierEmbedding
    @ModuleInfo(key: "time_embed") var timeEmbed: NFKMusic3TimestepEmbedding
    @ModuleInfo(key: "preprocess_conv") var preprocessConv: Conv1d
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "transformer_blocks") var blocks: [NFKMusic3DiTBlock]
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "postprocess_conv") var postprocessConv: Conv1d

    let configuration: NFKMLXMusic3DiTConfiguration

    init(_ configuration: NFKMLXMusic3DiTConfiguration) {
        self.configuration = configuration
        _timeProj.wrappedValue = NFKMusic3FourierEmbedding(dimensions: configuration.fourierDimensions)
        _timeEmbed.wrappedValue = NFKMusic3TimestepEmbedding(
            inputDimensions: configuration.fourierDimensions, dimensions: configuration.innerDimensions)
        _preprocessConv.wrappedValue = Conv1d(inputChannels: configuration.concatChannels,
                                              outputChannels: configuration.concatChannels,
                                              kernelSize: 1, bias: false)
        _projIn.wrappedValue = Linear(configuration.concatChannels, configuration.innerDimensions, bias: false)
        _blocks.wrappedValue = (0 ..< configuration.layerCount).map { _ in NFKMusic3DiTBlock(configuration) }
        _projOut.wrappedValue = Linear(configuration.innerDimensions, configuration.latentChannels, bias: false)
        _postprocessConv.wrappedValue = Conv1d(inputChannels: configuration.latentChannels,
                                               outputChannels: configuration.latentChannels,
                                               kernelSize: 1, bias: false)
    }

    /// Rotary tables for `length` positions (the timestep token included, at position 0).
    private func rotary(_ length: Int) -> (MLXArray, MLXArray) {
        let dimensions = configuration.rotaryDimensions
        let inverseFrequencies = MLXArray(stride(from: 0, to: dimensions, by: 2).map {
            Float(1.0 / pow(10_000.0, Double($0) / Double(dimensions)))
        })
        let positions = MLXArray(0 ..< length).asType(.float32).reshaped([length, 1])
        let frequencies = positions * inverseFrequencies.reshaped([1, dimensions / 2])
        let doubled = concatenated([frequencies, frequencies], axis: -1).reshaped([1, length, 1, dimensions])
        return (cos(doubled), MLX.sin(doubled))
    }

    /// `latents` `[B, L, latentChannels]`, `timestep` `[B]` in `0...1` (0 is noise), `condition`
    /// `[B, L, conditionDimensions]` (zeros for the unconditional branch) → the velocity, same shape
    /// as `latents`.
    func velocity(latents: MLXArray, timestep: MLXArray, condition: MLXArray) -> MLXArray {
        let stacked = concatenated([latents, MLXArray.zeros(latents.shape), condition], axis: -1)
        var hidden = preprocessConv(stacked) + stacked

        let embeddedTime = timeEmbed(timeProj(timestep))
        hidden = projIn(hidden)
        hidden = concatenated([embeddedTime.expandedDimensions(axis: 1), hidden], axis: 1)

        let (cosines, sines) = rotary(hidden.shape[1])
        for block in blocks {
            hidden = block(hidden, cosines: cosines, sines: sines)
        }
        let out = projOut(hidden[0..., 1...])
        return postprocessConv(out) + out
    }
}

/// The flow-matching Euler schedule the release samples with: `FlowMatchEulerDiscreteScheduler`
/// at `num_train_timesteps` 1, `shift` 1, `invert_sigmas` true, driven with
/// `sigmas = linspace(1, 1/steps, steps)`. Inverting turns that into σ running 0 (noise) → 1 (data)
/// with a terminal 1 appended; the model consumes σ directly as its timestep.
enum NFKMusic3FlowSchedule {

    /// The σ sequence for `steps` inference steps, the terminal 1 included (`steps + 1` values).
    static func sigmas(steps: Int) -> [Float] {
        var values = (0 ..< steps).map { index -> Float in
            let t = steps == 1 ? 1.0 : 1.0 - Double(index) * (1.0 - 1.0 / Double(steps)) / Double(steps - 1)
            return 1 - Float(t)
        }
        values.append(1)
        return values
    }

    /// One Euler step: `x + (σ_next − σ) · v`.
    static func step(latent: MLXArray, velocity: MLXArray, sigma: Float, nextSigma: Float) -> MLXArray {
        latent + (nextSigma - sigma) * velocity
    }

    /// The window-overlap blend, applied before every solver step: the overlapping latents are
    /// driven toward the previous window's carry as σ → 1.
    static func blendedOverlap(noise: MLXArray, previous: MLXArray, sigma: Float) -> MLXArray {
        (1 - (1 - 1e-6) * sigma) * noise + sigma * previous
    }
}

// MARK: - Windowed flow matching

/// The chunked denoise loop: 200-frame windows at a 100-frame hop, each flow-matched with the
/// previous window's trailing latents as an overlap prompt. Continuity lives INSIDE the sampler —
/// the overlapping latents are re-blended toward the previous window's carry at every Euler step
/// and locked to it afterward — rather than in a crossfade of finished audio.
final class NFKMusic3FlowMatcher {
    let transformer: NFKMusic3DiTNet
    let conditionEncoder: NFKMusic3ConditionEncoderNet
    var guidanceScale: Float = 1.7

    init(transformer: NFKMusic3DiTNet, conditionEncoder: NFKMusic3ConditionEncoderNet) {
        self.transformer = transformer
        self.conditionEncoder = conditionEncoder
    }

    /// Frame index at which each window starts. One window when everything fits; otherwise the
    /// last window still reaches the final frame because the range stops before `frames − hop`.
    static func chunkStarts(frames: Int) -> [Int] {
        frames <= NFKMusic3Contract.chunkFrames
            ? [0]
            : Array(stride(from: 0, to: frames - NFKMusic3Contract.chunkHop,
                           by: NFKMusic3Contract.chunkHop))
    }

    /// Denoises every window and returns the per-window latents `[1, latents, channels]`, uncropped.
    ///
    /// - Parameter progress: called once per solver step with (step, total); returning false cancels
    ///   and returns nil.
    func latentChunks(frameHiddens: MLXArray, steps: Int = 30, seed: UInt64 = 0,
                      progress: ((Int, Int) -> Bool)? = nil) -> [MLXArray]? {
        let frames = frameHiddens.shape[1]
        let starts = Self.chunkStarts(frames: frames)
        let sigmas = NFKMusic3FlowSchedule.sigmas(steps: steps)
        let channels = transformer.configuration.latentChannels

        var chunks = [MLXArray]()
        var previousLatent: MLXArray?
        var previousCondition: MLXArray?
        var stepIndex = 0
        let totalSteps = starts.count * steps

        for (chunkIndex, start) in starts.enumerated() {
            let end = min(start + NFKMusic3Contract.chunkFrames, frames)
            var condition = conditionEncoder.condition(frameHiddens[0..., start ..< end])
            let length = condition.shape[1]

            var overlap = 0
            if let previousLatent, let previousCondition {
                overlap = min(previousLatent.shape[1], length)
                condition = concatenated([previousCondition[0..., ..<overlap],
                                          condition[0..., overlap...]], axis: 1)
            }

            var latents = NFKMLXDiffusionBackend.gaussianNoise(
                height: length, width: 1, channels: channels,
                seed: seed &+ UInt64(chunkIndex)).reshaped([1, length, channels])
            let noisePrompt = overlap > 0 ? latents[0..., ..<overlap] : nil

            for step in 0 ..< steps {
                let sigma = sigmas[step]
                if let noisePrompt, let previousLatent, overlap > 0 {
                    let blended = NFKMusic3FlowSchedule.blendedOverlap(
                        noise: noisePrompt, previous: previousLatent[0..., ..<overlap], sigma: sigma)
                    latents = concatenated([blended, latents[0..., overlap...]], axis: 1)
                }
                let timestep = MLXArray([sigma])
                let conditional = transformer.velocity(latents: latents, timestep: timestep,
                                                       condition: condition)
                var velocity = conditional
                if guidanceScale != 1 {
                    let unconditional = transformer.velocity(
                        latents: latents, timestep: timestep,
                        condition: MLXArray.zeros(condition.shape))
                    velocity = unconditional + (conditional - unconditional) * guidanceScale
                }
                latents = NFKMusic3FlowSchedule.step(latent: latents, velocity: velocity,
                                                     sigma: sigma, nextSigma: sigmas[step + 1])
                eval(latents)
                stepIndex += 1
                if let progress, !progress(stepIndex, totalSteps) { return nil }
            }

            if let previousLatent, overlap > 0 {
                latents = concatenated([previousLatent[0..., ..<overlap],
                                        latents[0..., overlap...]], axis: 1)
            }
            let overlapStart = max(0, length - 2 * NFKMusic3Contract.overlapLatents)
            let overlapEnd = max(overlapStart, length - NFKMusic3Contract.overlapLatents)
            previousLatent = latents[0..., overlapStart ..< overlapEnd]
            previousCondition = condition[0..., overlapStart ..< overlapEnd]
            chunks.append(latents)
        }
        return chunks
    }

    /// The sample span each window keeps when the decoded waveforms are stitched: every window
    /// after the first drops its leading 86 latents and every window before the last its trailing
    /// 258, so the kept spans tile the song.
    static func keptRange(chunkIndex: Int, chunkCount: Int, samples: Int,
                          hop: Int) -> Range<Int> {
        let left = chunkIndex == 0 ? 0 : NFKMusic3Contract.cropLeftLatents * hop
        let right = chunkIndex == chunkCount - 1 ? 0 : NFKMusic3Contract.cropRightLatents * hop
        return left ..< max(left, samples - right)
    }
}

// MARK: - Autoregressive stage

/// The release's token contract and sampling recipe. These are the Diffusers pipeline's constants;
/// the ids live inside the language model's own 200000-token vocabulary.
enum NFKMusic3Contract {
    static let audioEndToken = 151_670
    static let audioCFGToken = 151_654
    static let audioCodeOffset = 151_675
    static let semanticVocabulary = 16_384
    static let maxPromptTokens = 5_000
    static let maxAudioFrames = 9_000
    /// The language model's trained `max_position_embeddings`. The reference declares its frame and
    /// prompt limits and lets their sum exceed this; here the sum is enforced.
    static let positionBudget = 10_240
    static let framesPerSecond = 25.0
    static let guidanceScale: Float = 1.5
    static let guidanceTopK = 50
    static let samplingTopK = 50
    static let chunkFrames = 200
    static let chunkHop = 100
    static let overlapLatents = 172
    static let cropLeftLatents = 86
    static let cropRightLatents = 344 - 86
}

/// Top-k categorical sampling from a SplitMix64 stream, so a run is repeatable from its seed.
///
/// The threshold is the k-th LARGEST logit — taken by sorting on the CPU rather than through
/// `MLX.top`, whose result is unsorted (reading its last slot as the threshold silently turns
/// sampling into argmax, the trap `torch.topk`-shaped reference code falls into).
struct NFKMusic3Sampler {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    private mutating func nextUniform() -> Float {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return Float((z ^ (z >> 31)) >> 40) / Float(1 << 24)
    }

    /// Draws one index from the top-`k` of `logits`.
    mutating func sample(logits: [Float], topK: Int) -> Int {
        let guarded = logits.map { $0.isNaN ? -1e9 : min(max($0, -1e9), 1e9) }
        let threshold = guarded.sorted(by: >)[min(topK, guarded.count) - 1]
        let peak = guarded.max() ?? 0
        var probabilities = guarded.map { $0 < threshold ? 0 : exp($0 - peak) }
        let total = probabilities.reduce(0, +)
        guard total > 0 else { return guarded.firstIndex(of: peak) ?? 0 }
        for index in probabilities.indices { probabilities[index] /= total }

        var remaining = nextUniform()
        for (index, probability) in probabilities.enumerated() {
            remaining -= probability
            if remaining <= 0 { return index }
        }
        return probabilities.lastIndex(where: { $0 > 0 }) ?? 0
    }
}

/// The autoregressive stage: the global language model samples one semantic code per frame with
/// classifier-free guidance, the depth decoder fills the seven residual codebooks, and the
/// concatenated hidden states — not the codes — become the synthesis conditioning. The codes exist
/// to close the feedback loop: each frame's eight embeddings are summed into ONE language-model
/// position, which is why a frame costs one position of the 10240 budget.
final class NFKMusic3AutoregressiveStage {
    let languageModel: NFKMLXLanguageNet
    let depthDecoder: NFKMusic3DepthDecoderNet

    /// Additive vocabulary mask: 0 on the semantic band and the end token, -1e9 elsewhere.
    private let bandMask: MLXArray

    init(languageModel: NFKMLXLanguageNet, depthDecoder: NFKMusic3DepthDecoderNet) {
        self.languageModel = languageModel
        self.depthDecoder = depthDecoder
        var mask = [Float](repeating: -1e9, count: languageModel.configuration.vocabularySize)
        for id in NFKMusic3Contract.audioCodeOffset
            ..< NFKMusic3Contract.audioCodeOffset + NFKMusic3Contract.semanticVocabulary {
            mask[id] = 0
        }
        mask[NFKMusic3Contract.audioEndToken] = 0
        bandMask = MLXArray(mask)
    }

    /// The guided semantic distribution: band-masked CFG at scale 1.5, restricted to the
    /// conditional branch's top candidates (guidance may only re-rank tokens the conditional
    /// already liked, which is the guard against amplifying one it never wanted).
    func guidedSemanticLogits(lastHidden: MLXArray) -> MLXArray {
        let logits = languageModel.logits(fromHidden: lastHidden).asType(.float32) + bandMask
        let conditional = logits[0]
        let unconditional = logits[1]
        var guided = unconditional + (conditional - unconditional) * NFKMusic3Contract.guidanceScale

        let values = conditional.asArray(Float.self)
        let threshold = values.sorted(by: >)[NFKMusic3Contract.guidanceTopK - 1]
        guided = MLX.where(conditional .< MLXArray(threshold), MLXArray(Float(-1e9)), guided)
        return guided + bandMask
    }

    /// One frame's residual codes and the depth hidden states that condition synthesis.
    ///
    /// - Parameter forcedCodes: the frame's eight codes when replaying a recorded run (the parity
    ///   harness teacher-forces the reference's own choices, so the comparison measures the
    ///   networks rather than two different random streams).
    func depthCodes(lastHidden: MLXArray, semanticCode: Int, forcedCodes: [Int]?,
                    sampler: inout NFKMusic3Sampler) -> (codes: [Int], hidden: MLXArray) {
        let vocabulary = depthDecoder.configuration.audioVocabSize
        let codebooks = depthDecoder.configuration.codebookCount
        var sequence = [projected(languageModel.embed(
            MLXArray([Int32(semanticCode + NFKMusic3Contract.audioCodeOffset)]).reshaped([1, 1])))]
        sequence.insert(projected(lastHidden.expandedDimensions(axis: 1)), at: 0)

        var codes = [semanticCode]
        var hiddenParts = [MLXArray]()
        for index in 1 ..< codebooks {
            let hidden = depthDecoder.hiddenStates(concatenated(sequence, axis: 1))
            let last = hidden[0..., hidden.shape[1] - 1]
            hiddenParts.append(last[0 ..< 1])
            let logits = depthDecoder.audioHeads[index - 1](last).asType(.float32)
            let guided = logits[1] + (logits[0] - logits[1]) * NFKMusic3Contract.guidanceScale
            let code = forcedCodes?[index]
                ?? sampler.sample(logits: guided.asArray(Float.self),
                                  topK: NFKMusic3Contract.samplingTopK)
            codes.append(code)
            if index < codebooks - 1 {
                let embedded = depthDecoder.audioEmbeddings(
                    MLXArray([Int32(code + (index - 1) * vocabulary)]).reshaped([1, 1]))
                sequence.append(projected(broadcast(embedded, to: [2, 1, embedded.shape[2]])))
            }
        }
        return (codes, concatenated(hiddenParts, axis: -1))
    }

    /// Both CFG rows share the depth decoder's projection.
    private func projected(_ embedding: MLXArray) -> MLXArray {
        let batched = embedding.shape[0] == 2
            ? embedding : broadcast(embedding, to: [2, embedding.shape[1], embedding.shape[2]])
        return depthDecoder.projection(batched)
    }

    /// The frame's feedback embedding: all eight code embeddings summed into one position and
    /// scaled by `codebooks^-0.5`, batched over the CFG pair.
    func feedbackEmbedding(codes: [Int]) -> MLXArray {
        let vocabulary = depthDecoder.configuration.audioVocabSize
        var embedded = languageModel.embed(
            MLXArray([Int32(codes[0] + NFKMusic3Contract.audioCodeOffset)]).reshaped([1, 1]))
        for (index, code) in codes.dropFirst().enumerated() {
            embedded = embedded + depthDecoder.audioEmbeddings(
                MLXArray([Int32(code + index * vocabulary)]).reshaped([1, 1]))
        }
        let scaled = embedded * Float(pow(Double(codes.count), -0.5))
        return broadcast(scaled, to: [2, 1, scaled.shape[2]])
    }

    struct Generation {
        /// `[1, frames, codebooks · hidden]` — what the condition encoder consumes.
        let frameHiddens: MLXArray
        /// Every frame's eight codes, the warm-up frame included.
        let codes: [[Int]]
    }

    /// Runs the loop: prefill the prompt pair, then one semantic step and seven depth steps per
    /// frame. The first decode step only advances the state past `<|audio_start|>` and is not an
    /// emitted frame, which the reference's own loop structure fixes.
    ///
    /// - Parameter forcedFrames: recorded per-frame codes to replay instead of sampling.
    func generate(textIDs: MLXArray, maxFrames: Int, seed: UInt64 = 0,
                  forcedFrames: [[Int]]? = nil) throws -> Generation {
        guard textIDs.shape[1] <= NFKMusic3Contract.maxPromptTokens else {
            throw NFKMLXError.unsupportedConfiguration(
                "the assembled prompt has \(textIDs.shape[1]) tokens; the maximum is "
                + "\(NFKMusic3Contract.maxPromptTokens)")
        }
        let frameBudget = min(maxFrames, NFKMusic3Contract.maxAudioFrames)
        guard textIDs.shape[1] + frameBudget + 1 <= NFKMusic3Contract.positionBudget else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(textIDs.shape[1]) prompt tokens + \(frameBudget) frames exceed the language "
                + "model's \(NFKMusic3Contract.positionBudget)-position budget; shorten the prompt "
                + "or the duration")
        }

        var sampler = NFKMusic3Sampler(seed: seed)
        let cache = NFKMLXKeyValueCache(layerCount: languageModel.configuration.layerCount)
        var lastHidden = languageModel.hiddenStates(
            fromEmbeddings: languageModel.embed(textIDs), cache: cache)
        lastHidden = lastHidden[0..., lastHidden.shape[1] - 1]
        eval(lastHidden)

        var frameHiddens = [MLXArray]()
        var allCodes = [[Int]]()
        for frameIndex in 0 ... frameBudget {
            let sampled: Int
            if let forced = forcedFrames {
                guard frameIndex < forced.count else { break }
                sampled = forced[frameIndex][0] + NFKMusic3Contract.audioCodeOffset
            } else {
                let guided = guidedSemanticLogits(lastHidden: lastHidden)
                sampled = sampler.sample(logits: guided.asArray(Float.self),
                                         topK: NFKMusic3Contract.samplingTopK)
            }
            if sampled == NFKMusic3Contract.audioEndToken { break }

            let (codes, depthHidden) = depthCodes(
                lastHidden: lastHidden, semanticCode: sampled - NFKMusic3Contract.audioCodeOffset,
                forcedCodes: forcedFrames?[frameIndex], sampler: &sampler)
            allCodes.append(codes)
            if frameIndex > 0 {
                frameHiddens.append(concatenated([lastHidden[0 ..< 1], depthHidden], axis: -1))
                if frameHiddens.count >= frameBudget { break }
            }
            let feedback = feedbackEmbedding(codes: codes)
            lastHidden = languageModel.hiddenStates(fromEmbeddings: feedback, cache: cache)[0..., 0]
            eval(lastHidden)
        }
        guard !frameHiddens.isEmpty else {
            throw NFKMLXError.noOutput
        }
        return Generation(frameHiddens: stacked(frameHiddens, axis: 1), codes: allCodes)
    }
}

/// MiniMax Music 3, ported stage by stage: the vocoder, the RVQ depth decoder, the condition
/// encoder, the flow-matching DiT, and the autoregressive stage so far. The backend that chains
/// them lands behind these, each stage gated on its own parity record.
@objc(NFKMLXMusic3)
public final class NFKMLXMusic3: NSObject {

    static func makeVocoder(
        _ configuration: NFKMLXMusic3VocoderConfiguration = NFKMLXMusic3VocoderConfiguration()
    ) -> NFKMusic3VocoderNet {
        NFKMusic3VocoderNet(configuration)
    }

    static func makeDepthDecoder(
        _ configuration: NFKMLXMusic3DepthConfiguration = NFKMLXMusic3DepthConfiguration()
    ) -> NFKMusic3DepthDecoderNet {
        NFKMusic3DepthDecoderNet(configuration)
    }

    static func makeConditionEncoder(
        _ configuration: NFKMLXMusic3ConditionConfiguration = NFKMLXMusic3ConditionConfiguration()
    ) -> NFKMusic3ConditionEncoderNet {
        NFKMusic3ConditionEncoderNet(configuration)
    }

    static func makeDiT(
        _ configuration: NFKMLXMusic3DiTConfiguration = NFKMLXMusic3DiTConfiguration()
    ) -> NFKMusic3DiTNet {
        NFKMusic3DiTNet(configuration)
    }

    /// Loads the released `transformer/` DIRECTORY (sharded, float32) or a fine-tuned single-file
    /// save. The release's names are the module's own; only the two 1×1 convolutions transpose.
    static func loadDiTWeights(into net: NFKMusic3DiTNet, from url: URL,
                               precision: NFKMLXWeightPrecision = .float32) throws {
        var isDirectory: ObjCBool = false
        FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
        // A directory holding ONE file routes through the checkpoint reader, which is where a
        // quantized save's metadata lives; the released tree is sharded and takes the merge path.
        var singleFile: URL?
        if isDirectory.boolValue {
            let files = try NFKMLXReleaseWeights.files(inDirectory: url)
            singleFile = files.count == 1 ? files[0] : nil
        } else {
            singleFile = url
        }
        let pairs: [(String, MLXArray)]
        let needsConvTranspose: Bool
        if let singleFile {
            let checkpoint = try NFKMLXWeights.loadCheckpoint(url: singleFile)
            NFKMLXQuantization.matchStructure(of: checkpoint, on: net)
            let stored: NFKMLXWeightPrecision = checkpoint.quantization != nil ? .checkpoint : precision
            pairs = NFKMLXWeights.converted(checkpoint.arrays.map { ($0, $1) }, to: stored)
            needsConvTranspose = checkpoint.needsConvTranspose
        } else {
            pairs = try NFKMLXReleaseWeights.arrays(inDirectory: url, precision: precision)
            needsConvTranspose = true
        }
        let mapped = pairs.map { key, value -> (String, MLXArray) in
            needsConvTranspose && value.ndim == 3 ? (key, value.transposed(0, 2, 1)) : (key, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Loads the released `rvq_depth_decoder/diffusion_pytorch_model.safetensors` (or a fine-tuned
    /// save). The release's names are the module's own and every tensor is at most 2-D, so nothing
    /// transposes; the file ships bf16 and loads at the requested precision.
    static func loadDepthWeights(into net: NFKMusic3DepthDecoderNet, from url: URL,
                                 precision: NFKMLXWeightPrecision = .float32) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        NFKMLXQuantization.matchStructure(of: checkpoint, on: net)
        let stored: NFKMLXWeightPrecision = checkpoint.quantization != nil ? .checkpoint : precision
        let mapped = NFKMLXWeights.converted(checkpoint.arrays.map { ($0, $1) }, to: stored)
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Loads the released `condition_encoder/diffusion_pytorch_model.safetensors` (or a fine-tuned
    /// save). Only the projection is a convolution; the blend logits and the scale pass through.
    static func loadConditionWeights(into net: NFKMusic3ConditionEncoderNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            if checkpoint.needsConvTranspose, value.ndim == 3 {
                return (key, value.transposed(0, 2, 1))
            }
            return (key, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Loads the released `vocoder/diffusion_pytorch_model.safetensors` (or a fine-tuned save).
    ///
    /// The official file stores every convolution weight-NORMALIZED (`weight_g`/`weight_v`); the
    /// fusion `g·v/‖v‖` is what the reference's parametrization computes at inference, done here at
    /// load so no offline converter is needed — the release is already safetensors.
    static func loadVocoderWeights(into net: NFKMusic3VocoderNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        guard checkpoint.needsConvTranspose else {
            try NFKMLXWeights.apply(checkpoint.arrays.map { ($0, $1) }, to: net)
            return
        }
        let mapped = fusedWeightNorm(checkpoint.arrays).map { key, value -> (String, MLXArray) in
            let name = remapVocoderKey(key)
            if key.hasSuffix(".alpha") {                              // [1, C, 1] → [1, 1, C]
                return (name, value.transposed(0, 2, 1))
            }
            if key.contains("conv_t1"), value.ndim == 3 {             // ConvT [in, out, K] → [out, K, 1, in]
                return (name, value.transposed(1, 2, 0).expandedDimensions(axis: 2))
            }
            if value.ndim == 3 {                                      // Conv [out, in, K] → [out, K, in]
                return (name, value.transposed(0, 2, 1))
            }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// The transposed convolutions live inside `NFKDemucsConvT1d`'s `conv` wrapper.
    static func remapVocoderKey(_ key: String) -> String {
        key.replacingOccurrences(of: "conv_t1.weight", with: "conv_t1.conv.weight")
            .replacingOccurrences(of: "conv_t1.bias", with: "conv_t1.conv.bias")
    }

    /// Collapses `*.weight_g` + `*.weight_v` pairs into `*.weight` (`g·v/‖v‖`, the norm over every
    /// axis but the first — PyTorch's `weight_norm(dim: 0)`, which covers the forward and the
    /// transposed convolutions alike because `g` broadcasts along axis 0 in both layouts).
    static func fusedWeightNorm(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var fused = [String: MLXArray]()
        for (key, value) in arrays {
            if key.hasSuffix(".weight_g") { continue }
            guard key.hasSuffix(".weight_v") else {
                fused[key] = value
                continue
            }
            let base = String(key.dropLast(".weight_v".count))
            guard let gain = arrays[base + ".weight_g"] else {
                fused[key] = value
                continue
            }
            let norm = sqrt(value.square().sum(axes: Array(1 ..< value.ndim), keepDims: true))
            fused[base + ".weight"] = gain * value / maximum(norm, 1e-12)
        }
        return fused
    }
}
