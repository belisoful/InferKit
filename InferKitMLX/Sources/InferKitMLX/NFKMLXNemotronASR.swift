//
//  NFKMLXNemotronASR.swift
//  InferKitMLX
//
//  NVIDIA's cache-aware streaming FastConformer recognizer (Nemotron ASR Streaming), ported from the
//  transformers `nemotron_asr_streaming` model and NeMo's cache-aware conformer. Every operation that
//  reads the future in Parakeet's FastConformer is causal here: the subsampler pads time and frequency at
//  the front, the convolution module's depthwise convolution reads only past frames, and attention runs
//  under a chunked mask whose left context its key/value cache holds exactly. One encoder serves the
//  clip forward, under the chunked mask, and the stream, which carries the caches from chunk to chunk.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

// MARK: - Configuration

/// The released Nemotron ASR Streaming geometry, read from a release's `config.json`.
public struct NFKMLXNemotronASRConfiguration: Sendable {
    // Front end (the feature extractor of `processor_config.json`).
    public var sampleRate = 16000
    public var mels = 128
    public var windowSamples = 400
    public var hopSamples = 160
    public var fftSize = 512
    public var preemphasis: Float = 0.97
    // Encoder.
    public var dModel = 1024
    public var layers = 24
    public var heads = 8
    public var intermediate = 4096
    public var convKernel = 9
    public var subsamplingChannels = 256
    public var subsamplingFactor = 8
    public var subsamplingKernel = 3
    public var subsamplingStride = 2
    public var attentionBias = false
    public var convolutionBias = false
    public var scaleInput = false
    /// The attention's left context in encoder frames, `sliding_window − 1`.
    public var leftContext = 70
    /// The right contexts the release was trained with, in encoder frames.
    public var lookaheads = [13, 6, 1, 0]
    public var defaultLookahead = 13
    public var maxPositions = 5000
    // Decoder and joint.
    public var vocabulary = 1025
    public var blank = 1024
    public var decoderHidden = 640
    public var decoderLayers = 2
    public var maxSymbolsPerStep = 10

    public init() {}

    /// The released `nvidia/nemotron-speech-streaming-en-0.6b`.
    public static let english06B = NFKMLXNemotronASRConfiguration()

    /// A shrunk geometry for shape and stream tests.
    public static var tiny: NFKMLXNemotronASRConfiguration {
        var c = NFKMLXNemotronASRConfiguration()
        c.mels = 16
        c.dModel = 32
        c.layers = 2
        c.heads = 2
        c.intermediate = 64
        c.convKernel = 5
        c.subsamplingChannels = 8
        c.leftContext = 6
        c.lookaheads = [2, 1, 0]
        c.defaultLookahead = 2
        c.vocabulary = 13
        c.blank = 12
        c.decoderHidden = 16
        return c
    }

    /// The configuration `config.json` states.
    init(json: [String: Any]) throws {
        self.init()
        guard let encoder = json["encoder_config"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("config.json has no encoder_config")
        }
        func int(_ object: [String: Any], _ key: String, _ fallback: Int) -> Int { (object[key] as? Int) ?? fallback }
        dModel = int(encoder, "hidden_size", dModel)
        layers = int(encoder, "num_hidden_layers", layers)
        heads = int(encoder, "num_attention_heads", heads)
        intermediate = int(encoder, "intermediate_size", intermediate)
        convKernel = int(encoder, "conv_kernel_size", convKernel)
        mels = int(encoder, "num_mel_bins", mels)
        subsamplingChannels = int(encoder, "subsampling_conv_channels", subsamplingChannels)
        subsamplingFactor = int(encoder, "subsampling_factor", subsamplingFactor)
        subsamplingKernel = int(encoder, "subsampling_conv_kernel_size", subsamplingKernel)
        subsamplingStride = int(encoder, "subsampling_conv_stride", subsamplingStride)
        attentionBias = (encoder["attention_bias"] as? Bool) ?? attentionBias
        convolutionBias = (encoder["convolution_bias"] as? Bool) ?? convolutionBias
        scaleInput = (encoder["scale_input"] as? Bool) ?? scaleInput
        leftContext = int(encoder, "sliding_window", leftContext + 1) - 1
        lookaheads = (encoder["supported_num_lookahead_tokens"] as? [Int]) ?? lookaheads
        defaultLookahead = int(encoder, "default_num_lookahead_tokens", defaultLookahead)
        maxPositions = int(encoder, "max_position_embeddings", maxPositions)
        vocabulary = int(json, "vocab_size", vocabulary)
        blank = int(json, "blank_token_id", blank)
        decoderHidden = int(json, "decoder_hidden_size", decoderHidden)
        decoderLayers = int(json, "num_decoder_layers", decoderLayers)
        maxSymbolsPerStep = int(json, "max_symbols_per_step", maxSymbolsPerStep)
        guard leftContext >= 0, lookaheads.allSatisfy({ $0 >= 0 && leftContext % ($0 + 1) == 0 }) else {
            throw NFKMLXError.unsupportedConfiguration("a lookahead does not divide the left context, so a stream would not hold the clip's context")
        }
    }

    /// The mel frames of a stream's first chunk at `lookahead`: one frame, then eight per further encoder frame.
    func firstChunkFrames(lookahead: Int) -> Int { 1 + subsamplingFactor * lookahead }

    /// The mel frames of each later chunk at `lookahead`.
    func chunkFrames(lookahead: Int) -> Int { subsamplingFactor * (lookahead + 1) }

    /// The seconds one encoder frame spans.
    var encoderFrameSeconds: Double { Double(hopSamples * subsamplingFactor) / Double(sampleRate) }
}

// MARK: - Front end

/// The log-mel front end: preemphasis, a 512-point transform of zero-centered frames under a symmetric
/// 400-point Hann window, the power spectrum through a Slaney mel bank, and `log(x + 2⁻²⁴)`. Nothing is
/// normalized, so every frame depends only on the samples under its window, which is what lets a stream
/// compute the same frames a clip does.
final class NFKNemotronASRFrontEnd {
    let configuration: NFKMLXNemotronASRConfiguration
    let window: MLXArray            // [fftSize], the 400-point window centered
    let filterbank: MLXArray        // [bins, mels]

    init(_ configuration: NFKMLXNemotronASRConfiguration) {
        self.configuration = configuration
        let count = configuration.windowSamples
        let hann = (0 ..< count).map { 0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(count - 1)) }
        let pad = (configuration.fftSize - count) / 2
        let centered = [Float](repeating: 0, count: pad) + hann + [Float](repeating: 0, count: configuration.fftSize - count - pad)
        window = MLXArray(centered)
        filterbank = Self.slaneyFilterbank(sampleRate: configuration.sampleRate, fftSize: configuration.fftSize,
                                           mels: configuration.mels)
    }

    /// librosa's `filters.mel(norm="slaney")` from 0 Hz to Nyquist, computed in double precision and
    /// rounded once, as librosa's float64 bank cast to float32 is. `[bins, mels]`.
    static func slaneyFilterbank(sampleRate: Int, fftSize: Int, mels: Int) -> MLXArray {
        let bins = fftSize / 2 + 1
        let fSp = 200.0 / 3.0, minLogHz = 1000.0, minLogMel = minLogHz / fSp, logStep = log(6.4) / 27.0
        func hzToMel(_ f: Double) -> Double { f >= minLogHz ? minLogMel + log(f / minLogHz) / logStep : f / fSp }
        func melToHz(_ m: Double) -> Double { m >= minLogMel ? minLogHz * exp(logStep * (m - minLogMel)) : fSp * m }
        let nyquist = Double(sampleRate) / 2
        let maxMel = hzToMel(nyquist)
        let melPoints = (0 ..< mels + 2).map { melToHz(maxMel * Double($0) / Double(mels + 1)) }
        let fftFrequencies = (0 ..< bins).map { nyquist * Double($0) / Double(bins - 1) }
        var bank = [Float](repeating: 0, count: bins * mels)
        for m in 0 ..< mels {
            let enorm = 2.0 / (melPoints[m + 2] - melPoints[m])
            for k in 0 ..< bins {
                let lower = (fftFrequencies[k] - melPoints[m]) / (melPoints[m + 1] - melPoints[m])
                let upper = (melPoints[m + 2] - fftFrequencies[k]) / (melPoints[m + 2] - melPoints[m + 1])
                bank[k * mels + m] = Float(max(0, min(lower, upper)) * enorm)
            }
        }
        return MLXArray(bank).reshaped([bins, mels])
    }

    /// Preemphasized samples, the first kept as it is.
    static func preemphasized(_ samples: [Float], coefficient: Float, previous: Float?) -> [Float] {
        var out = samples
        for index in stride(from: samples.count - 1, through: 0, by: -1) {
            if index > 0 {
                out[index] = samples[index] - coefficient * samples[index - 1]
            } else if let previous {
                out[index] = samples[index] - coefficient * previous
            }
        }
        return out
    }

    /// Unwindowed frames `[frames, fftSize]` → log-mel features `[frames, mels]`.
    func logMel(frames: MLXArray) -> MLXArray {
        let spectrum = rfft(frames * window, axis: 1)
        let magnitude = sqrt(spectrum.realPart() * spectrum.realPart() + spectrum.imaginaryPart() * spectrum.imaginaryPart())
        let power = magnitude * magnitude
        return log(power.matmul(filterbank) + MLXArray(powf(2, -24)))
    }

    /// A clip's features `[1, frames, mels]`: the frames centered on every hop, zero-padded by half a
    /// transform at each end, cropped to the `⌊samples / hop⌋` frames the reference counts valid.
    func features(_ samples: [Float]) -> MLXArray {
        let size = configuration.fftSize, hop = configuration.hopSamples
        let signal = Self.preemphasized(samples, coefficient: configuration.preemphasis, previous: nil)
        let pad = size / 2
        let padded = [Float](repeating: 0, count: pad) + signal + [Float](repeating: 0, count: pad)
        let valid = samples.count / hop
        guard valid > 0 else {
            return MLXArray.zeros([1, 0, configuration.mels])
        }
        var rows = [Float](repeating: 0, count: valid * size)
        for frame in 0 ..< valid {
            for offset in 0 ..< size {
                rows[frame * size + offset] = padded[frame * hop + offset]
            }
        }
        let frames = rows.withUnsafeBufferPointer { MLXArray($0, [valid, size]) }
        return logMel(frames: frames).reshaped([1, valid, configuration.mels])
    }
}

// MARK: - Subsampling

/// A stride-2 causal convolution over the `(time, mel)` plane: two frames and two bins of zeros at the
/// front and one of each at the back. In a stream the time axis instead reads the one input frame the
/// chunk before ended with, two zero frames on the first chunk, and no frame at the back; with chunks of
/// the release's geometry that is the clip's padding.
final class NFKNemotronASRCausalConv2d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let stride: Int
    let groups: Int
    let kernel: Int

    init(inputChannels: Int, outputChannels: Int, kernel: Int, stride: Int, groups: Int = 1) {
        self.stride = stride
        self.groups = groups
        self.kernel = kernel
        _weight.wrappedValue = MLXArray.zeros([outputChannels, kernel, kernel, inputChannels / groups])
        _bias.wrappedValue = MLXArray.zeros([outputChannels])
    }

    /// `x` `[B, T, F, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let front = kernel - 1, back = stride - 1
        var h = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((front, back)), IntOrPair(0)], mode: .constant)
        if NFKMLXStreamCarry.current != nil {
            let (joined, previous) = NFKMLXStreamCarry.history(h, keep: kernel - stride, axis: 1, owner: self)
            h = previous > 0 ? joined
                : MLX.padded(joined, widths: [IntOrPair(0), IntOrPair((front, 0)), IntOrPair(0), IntOrPair(0)], mode: .constant)
        } else {
            h = MLX.padded(h, widths: [IntOrPair(0), IntOrPair((front, back)), IntOrPair(0), IntOrPair(0)], mode: .constant)
        }
        return conv2d(h, weight, stride: IntOrPair(stride), groups: groups) + bias
    }
}

/// One depthwise-separable stage: a depthwise causal stride-2 convolution and a pointwise one.
final class NFKNemotronASRSubsamplingLayer: Module {
    @ModuleInfo(key: "depthwise_conv") var depthwise: NFKNemotronASRCausalConv2d
    @ModuleInfo(key: "pointwise_conv") var pointwise: Conv2d

    init(_ config: NFKMLXNemotronASRConfiguration) {
        let c = config.subsamplingChannels
        _depthwise.wrappedValue = NFKNemotronASRCausalConv2d(inputChannels: c, outputChannels: c, kernel: config.subsamplingKernel,
                                                             stride: config.subsamplingStride, groups: c)
        _pointwise.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { pointwise(depthwise(x)) }
}

/// The 8× causal subsampler: a stride-2 stem, two depthwise-separable stride-2 stages, ReLU after each,
/// then `(channel, mel)` flattened per frame into a linear projection.
final class NFKNemotronASRSubsampling: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKNemotronASRCausalConv2d
    @ModuleInfo(key: "layers") var layers: [NFKNemotronASRSubsamplingLayer]
    @ModuleInfo(key: "linear") var linear: Linear

    init(_ config: NFKMLXNemotronASRConfiguration) {
        let stages = Int(log2(Double(config.subsamplingFactor)))
        _convIn.wrappedValue = NFKNemotronASRCausalConv2d(inputChannels: 1, outputChannels: config.subsamplingChannels,
                                                          kernel: config.subsamplingKernel, stride: config.subsamplingStride)
        _layers.wrappedValue = (1 ..< stages).map { _ in NFKNemotronASRSubsamplingLayer(config) }
        let pad = config.subsamplingKernel - 1 + config.subsamplingStride - 1
        let reducedMels = (0 ..< stages).reduce(config.mels) { m, _ in (m + pad - config.subsamplingKernel) / config.subsamplingStride + 1 }
        _linear.wrappedValue = Linear(config.subsamplingChannels * reducedMels, config.dModel)
    }

    /// `[B, T, mels]` → `[B, T', dModel]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = relu(convIn(x.expandedDimensions(axis: 3)))
        for layer in layers {
            h = relu(layer(h))
        }
        let (b, t, f, c) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        return linear(h.transposed(0, 1, 3, 2).reshaped([b, t, c * f]))
    }
}

// MARK: - Conformer layer

/// Relative-position attention with a key/value window: the keys and values of the current frames join
/// the last `leftContext` frames' in a stream, and a clip forward reads its chunked mask instead.
final class NFKNemotronASRAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    @ModuleInfo(key: "relative_k_proj") var relative: Linear
    @ParameterInfo(key: "bias_u") var biasU: MLXArray
    @ParameterInfo(key: "bias_v") var biasV: MLXArray
    let heads: Int
    let headDim: Int
    let leftContext: Int

    init(_ config: NFKMLXNemotronASRConfiguration) {
        heads = config.heads
        headDim = config.dModel / config.heads
        leftContext = config.leftContext
        let d = config.dModel
        _q.wrappedValue = Linear(d, d, bias: config.attentionBias)
        _k.wrappedValue = Linear(d, d, bias: config.attentionBias)
        _v.wrappedValue = Linear(d, d, bias: config.attentionBias)
        _o.wrappedValue = Linear(d, d, bias: config.attentionBias)
        _relative.wrappedValue = Linear(d, d, bias: false)
        _biasU.wrappedValue = MLXArray.zeros([heads, headDim])
        _biasV.wrappedValue = MLXArray.zeros([heads, headDim])
    }

    /// Transformer-XL's shift (appendix B): `[b, h, Q, P]` → each query's scores realigned to its offsets.
    private func relShift(_ x: MLXArray) -> MLXArray {
        let (b, h, t1, t2) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair(0), IntOrPair((1, 0))], mode: .constant)
        return padded.reshaped([b, h, t2 + 1, t1])[0..., 0..., 1..., 0...].reshaped([b, h, t1, t2])
    }

    /// `x` `[B, Q, D]`, `positions` the table of `2L − 1` offsets for the `L` keys, `mask` `[Q, L]` or nil.
    func callAsFunction(_ x: MLXArray, positions: (Int) -> MLXArray, mask: MLXArray?) -> MLXArray {
        let (b, queries) = (x.dim(0), x.dim(1))
        func split(_ y: MLXArray) -> MLXArray { y.reshaped([b, -1, heads, headDim]).transposed(0, 2, 1, 3) }
        let qh = q(x).reshaped([b, queries, heads, headDim])
        let keys = NFKMLXStreamCarry.history(split(k(x)), keep: leftContext, axis: 2, owner: k).joined
        let values = NFKMLXStreamCarry.history(split(v(x)), keep: leftContext, axis: 2, owner: v).joined
        let total = keys.dim(2)
        let p = relative(positions(total)).reshaped([1, -1, heads, headDim]).transposed(0, 2, 1, 3)
        let qU = (qh + biasU).transposed(0, 2, 1, 3)
        let qV = (qh + biasV).transposed(0, 2, 1, 3)
        let scaling = 1 / sqrt(Float(headDim))
        var bd = relShift(matmul(qV, p.transposed(0, 1, 3, 2)))[0..., 0..., 0..., 0 ..< total] * scaling
        if let mask {
            bd = MLX.where(mask, bd, MLXArray(-Float.infinity))
        }
        let scores = matmul(qU, keys.transposed(0, 1, 3, 2)) * scaling + bd
        let attended = matmul(softmax(scores, axis: -1, precise: true), values)
        return o(attended.transposed(0, 2, 1, 3).reshaped([b, queries, heads * headDim]))
    }
}

/// The convolution module: pointwise to twice the width, a GLU, a causal depthwise convolution over the
/// past `kernel − 1` frames, LayerNorm, Swish, pointwise back.
final class NFKNemotronASRConvModule: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwise1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwise2: Conv1d
    let kernel: Int

    init(_ config: NFKMLXNemotronASRConfiguration) {
        let d = config.dModel
        kernel = config.convKernel
        _pointwise1.wrappedValue = Conv1d(inputChannels: d, outputChannels: 2 * d, kernelSize: 1, bias: config.convolutionBias)
        _depthwise.wrappedValue = Conv1d(inputChannels: d, outputChannels: d, kernelSize: config.convKernel, groups: d,
                                         bias: config.convolutionBias)
        _norm.wrappedValue = LayerNorm(dimensions: d)
        _pointwise2.wrappedValue = Conv1d(inputChannels: d, outputChannels: d, kernelSize: 1, bias: config.convolutionBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated = pointwise1(x)
        let half = gated.dim(2) / 2
        let glu = gated[0..., 0..., 0 ..< half] * sigmoid(gated[0..., 0..., half...])
        let causal = NFKMLXStreamCarry.causallyPadded(glu, frames: kernel - 1, axis: 1, owner: self)
        return pointwise2(silu(norm(depthwise(causal))))
    }
}

/// Linear → Swish → Linear.
final class NFKNemotronASRFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ config: NFKMLXNemotronASRConfiguration) {
        _linear1.wrappedValue = Linear(config.dModel, config.intermediate, bias: config.attentionBias)
        _linear2.wrappedValue = Linear(config.intermediate, config.dModel, bias: config.attentionBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

/// One pre-norm conformer block: half a feed-forward, attention, the convolution module, half a
/// feed-forward, and an output norm.
final class NFKNemotronASRLayer: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFF1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var ff1: NFKNemotronASRFeedForward
    @ModuleInfo(key: "norm_self_att") var normAttention: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: NFKNemotronASRAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: NFKNemotronASRConvModule
    @ModuleInfo(key: "norm_feed_forward2") var normFF2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var ff2: NFKNemotronASRFeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(_ config: NFKMLXNemotronASRConfiguration) {
        let d = config.dModel
        _normFF1.wrappedValue = LayerNorm(dimensions: d)
        _ff1.wrappedValue = NFKNemotronASRFeedForward(config)
        _normAttention.wrappedValue = LayerNorm(dimensions: d)
        _attention.wrappedValue = NFKNemotronASRAttention(config)
        _normConv.wrappedValue = LayerNorm(dimensions: d)
        _conv.wrappedValue = NFKNemotronASRConvModule(config)
        _normFF2.wrappedValue = LayerNorm(dimensions: d)
        _ff2.wrappedValue = NFKNemotronASRFeedForward(config)
        _normOut.wrappedValue = LayerNorm(dimensions: d)
    }

    func callAsFunction(_ x: MLXArray, positions: (Int) -> MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + 0.5 * ff1(normFF1(x))
        h = h + attention(normAttention(h), positions: positions, mask: mask)
        h = h + conv(normConv(h))
        h = h + 0.5 * ff2(normFF2(h))
        return normOut(h)
    }
}

/// The cache-aware FastConformer encoder.
final class NFKNemotronASREncoder: Module {
    @ModuleInfo(key: "subsampling") var subsampling: NFKNemotronASRSubsampling
    @ModuleInfo(key: "layers") var layers: [NFKNemotronASRLayer]
    let configuration: NFKMLXNemotronASRConfiguration

    init(_ config: NFKMLXNemotronASRConfiguration) {
        configuration = config
        _subsampling.wrappedValue = NFKNemotronASRSubsampling(config)
        _layers.wrappedValue = (0 ..< config.layers).map { _ in NFKNemotronASRLayer(config) }
    }

    /// The interleaved sine and cosine table for offsets `L − 1 … −(L − 1)`, `[1, 2L − 1, D]`, at the
    /// reference's float32 inverse frequencies.
    func positionTable(keys length: Int) -> MLXArray {
        let d = configuration.dModel
        let inverse = stride(from: 0, to: d, by: 2).map { 1 / powf(10000, Float($0) / Float(d)) }
        let count = 2 * length - 1
        var table = [Float](repeating: 0, count: count * d)
        for row in 0 ..< count {
            let position = Float(length - 1 - row)
            for (index, frequency) in inverse.enumerated() {
                let angle = frequency * position
                table[row * d + 2 * index] = sinf(angle)
                table[row * d + 2 * index + 1] = cosf(angle)
            }
        }
        return MLXArray(table).reshaped([1, count, d])
    }

    /// The chunked-limited mask `[T, T]`: a query sees the keys of its own chunk of `lookahead + 1`
    /// frames and of the `leftContext / (lookahead + 1)` chunks before it.
    func chunkedMask(frames: Int, lookahead: Int) -> MLXArray {
        let chunk = lookahead + 1
        let leftChunks = configuration.leftContext / chunk
        let index = floorDivide(MLXArray(Int32(0) ..< Int32(frames)), MLXArray(Int32(chunk)))
        let difference = index.reshaped([frames, 1]) - index.reshaped([1, frames])
        return (difference .>= 0) .&& (difference .<= Int32(leftChunks))
    }

    /// Features `[1, T, mels]` → encoder frames `[1, T', D]`. A clip forward passes its lookahead for the
    /// chunked mask; a stream runs inside its carry, chunk by chunk, with `lookahead` nil.
    func callAsFunction(_ features: MLXArray, lookahead: Int?) -> MLXArray {
        var x = subsampling(features)
        if configuration.scaleInput {
            x = x * sqrtf(Float(configuration.dModel))
        }
        let mask = lookahead.map { chunkedMask(frames: x.dim(1), lookahead: $0) }
        var tables: [Int: MLXArray] = [:]
        let positions: (Int) -> MLXArray = { length in
            if let table = tables[length] {
                return table
            }
            let table = self.positionTable(keys: length)
            tables[length] = table
            return table
        }
        for layer in layers {
            x = layer(x, positions: positions, mask: mask)
        }
        return x
    }
}

// MARK: - Transducer

/// The prediction network: a token embedding into a stacked LSTM, then a projection to the joint width.
final class NFKNemotronASRDecoder: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "lstm") var lstm: NFKNemotronASRStackedLSTM
    @ModuleInfo(key: "decoder_projector") var projector: Linear

    init(_ config: NFKMLXNemotronASRConfiguration) {
        _embedding.wrappedValue = Embedding(embeddingCount: config.vocabulary, dimensions: config.decoderHidden)
        _lstm.wrappedValue = NFKNemotronASRStackedLSTM(inputSize: config.decoderHidden, hiddenSize: config.decoderHidden,
                                                       layers: config.decoderLayers)
        _projector.wrappedValue = Linear(config.decoderHidden, config.decoderHidden)
    }

    /// One step from `token` (the blank starts the sequence) → the projected output `[1, H]` and the state.
    func step(token: Int, state: [(MLXArray, MLXArray)]?) -> (MLXArray, [(MLXArray, MLXArray)]) {
        let input = embedding(MLXArray([Int32(token)]).reshaped([1, 1]))
        let (output, next) = lstm(input, state: state)
        return (projector(output).reshaped([1, -1]), next)
    }
}

/// A stacked LSTM in the PyTorch `nn.LSTM(num_layers:)` shape, each layer's state threaded.
final class NFKNemotronASRStackedLSTM: Module {
    @ModuleInfo(key: "layers") var layers: [LSTM]

    init(inputSize: Int, hiddenSize: Int, layers count: Int) {
        _layers.wrappedValue = (0 ..< count).map { LSTM(inputSize: $0 == 0 ? inputSize : hiddenSize, hiddenSize: hiddenSize) }
    }

    func callAsFunction(_ x: MLXArray, state: [(MLXArray, MLXArray)]?) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var h = x
        var next: [(MLXArray, MLXArray)] = []
        for (index, layer) in layers.enumerated() {
            let (hidden, cell) = layer(h, hidden: state?[index].0, cell: state?[index].1)
            h = hidden
            next.append((hidden[0..., -1, 0...], cell[0..., -1, 0...]))
        }
        return (h, next)
    }
}

/// The joint's output head over `ReLU(encoder projection + prediction)`.
final class NFKNemotronASRJoint: Module {
    @ModuleInfo(key: "head") var head: Linear

    init(_ config: NFKMLXNemotronASRConfiguration) {
        _head.wrappedValue = Linear(config.decoderHidden, config.vocabulary)
    }

    func callAsFunction(encoder f: MLXArray, prediction g: MLXArray) -> MLXArray { head(relu(f + g)) }
}

// MARK: - Model

/// The decoding state a transcription carries from frame to frame and, in a stream, from chunk to chunk:
/// the prediction network's LSTM state and output after the last emitted token.
public final class NFKMLXNemotronASRDecodingState {
    var lstm: [(MLXArray, MLXArray)]?
    var prediction: MLXArray?
    /// The encoder frames decoded so far.
    public internal(set) var frames = 0

    public init() {}
}

public final class NFKMLXNemotronASRNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKNemotronASREncoder
    @ModuleInfo(key: "encoder_projector") var encoderProjector: Linear
    @ModuleInfo(key: "decoder") var decoder: NFKNemotronASRDecoder
    @ModuleInfo(key: "joint") var joint: NFKNemotronASRJoint
    let frontEnd: NFKNemotronASRFrontEnd
    public let configuration: NFKMLXNemotronASRConfiguration

    public init(_ configuration: NFKMLXNemotronASRConfiguration) {
        self.configuration = configuration
        frontEnd = NFKNemotronASRFrontEnd(configuration)
        _encoder.wrappedValue = NFKNemotronASREncoder(configuration)
        _encoderProjector.wrappedValue = Linear(configuration.dModel, configuration.decoderHidden)
        _decoder.wrappedValue = NFKNemotronASRDecoder(configuration)
        _joint.wrappedValue = NFKNemotronASRJoint(configuration)
        super.init()
        train(false)
    }

    /// One recognized token and the encoder frame it was emitted at.
    public struct Token: Equatable {
        public let id: Int
        public let frame: Int
    }

    /// A clip's encoder frames `[1, T, D]` under the chunked mask of `lookahead`.
    public func encode(features: MLXArray, lookahead: Int) -> MLXArray {
        encoder(features, lookahead: lookahead)
    }

    /// Greedy RNN-T decoding of encoder frames `[1, T, D]` from `state`, which it advances: a blank moves
    /// to the next frame, a token stays on the frame and advances the prediction network, and after
    /// `maxSymbolsPerStep` tokens on one frame the frame advances.
    public func decode(encoded: MLXArray, state: NFKMLXNemotronASRDecodingState) -> [Token] {
        let projected = encoderProjector(encoded)[0]                     // [T, H]
        eval(projected)
        let blank = configuration.blank
        if state.prediction == nil {
            let (g, next) = decoder.step(token: blank, state: nil)
            state.prediction = g
            state.lstm = next
        }
        var tokens: [Token] = []
        for time in 0 ..< projected.dim(0) {
            let frame = projected[time ..< time + 1]
            var symbols = 0
            while symbols < configuration.maxSymbolsPerStep {
                let logits = joint(encoder: frame, prediction: state.prediction!)
                let k = argMax(logits, axis: -1).item(Int.self)
                if k == blank {
                    break
                }
                tokens.append(Token(id: k, frame: state.frames + time))
                let (g, next) = decoder.step(token: k, state: state.lstm)
                state.prediction = g
                state.lstm = next
                symbols += 1
            }
        }
        state.frames += projected.dim(0)
        return tokens
    }

    /// Waveform (16 kHz mono) → the recognized tokens, through the clip forward at `lookahead`.
    public func recognize(_ samples: [Float], lookahead: Int? = nil) -> [Token] {
        let features = frontEnd.features(samples)
        guard features.dim(1) > 0 else {
            return []
        }
        let encoded = encode(features: features, lookahead: lookahead ?? configuration.defaultLookahead)
        return decode(encoded: encoded, state: NFKMLXNemotronASRDecodingState())
    }
}

// MARK: - Tokenizer

/// The release's `tokenizer.json` piece table, read for decoding: a SentencePiece model, BPE (a piece →
/// id map) or Unigram (`[piece, score]` in id order), whose `▁` marks a word start and whose `<0xNN>`
/// pieces are byte fallbacks. Added special tokens (the blank among them) decode to nothing.
public struct NFKMLXNemotronASRTokenizer: Sendable {
    public let pieces: [String]
    public let specialIds: Set<Int>

    public init(tokenizerURL: URL) throws {
        let data = try Data(contentsOf: tokenizerURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any] else {
            throw NFKMLXError.malformedCheckpoint("tokenizer.json has no model")
        }
        var pieces: [String] = []
        if let vocab = model["vocab"] as? [String: Int] {
            pieces = [String](repeating: "", count: (vocab.values.max() ?? -1) + 1)
            for (piece, id) in vocab where id >= 0 {
                pieces[id] = piece
            }
        } else if let vocab = model["vocab"] as? [[Any]] {
            pieces = vocab.map { ($0.first as? String) ?? "" }
        } else {
            throw NFKMLXError.malformedCheckpoint("tokenizer.json's model has no vocab")
        }
        var specialIds = Set<Int>()
        for token in (root["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let id = token["id"] as? Int, let content = token["content"] as? String else {
                continue
            }
            if id >= pieces.count {
                pieces += [String](repeating: "", count: id + 1 - pieces.count)
            }
            pieces[id] = content
            if (token["special"] as? Bool) == true {
                specialIds.insert(id)
            }
        }
        self.pieces = pieces
        self.specialIds = specialIds
    }

    init(pieces: [String], specialIds: Set<Int>) {
        self.pieces = pieces
        self.specialIds = specialIds
    }

    /// The text of `ids`: the pieces joined, byte fallbacks decoded as UTF-8, `▁` as a space, and a
    /// leading space dropped when `atStart`, as a whole transcript drops it.
    public func text(for ids: [Int], atStart: Bool = true) -> String {
        var bytes: [UInt8] = []
        for id in ids where id >= 0 && id < pieces.count && !specialIds.contains(id) {
            let piece = pieces[id]
            if piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
               let byte = UInt8(piece.dropFirst(3).prefix(2), radix: 16) {
                bytes.append(byte)
            } else {
                bytes += Array(piece.replacingOccurrences(of: "\u{2581}", with: " ").utf8)
            }
        }
        let text = String(decoding: bytes, as: UTF8.self)
        return atStart && text.hasPrefix(" ") ? String(text.dropFirst()) : text
    }
}

// MARK: - Loading

@objc(NFKMLXNemotronASR)
public final class NFKMLXNemotronASR: NSObject {
    /// The registry name of the English release.
    @objc public static let modelName = "nemotron-asr-streaming"

    /// The `model_type` values the loader reads from a release's `config.json`.
    static let modelTypes = ["nemotron_asr_streaming"]

    /// A release key → the module's name, or nil to drop it. The LSTM's PyTorch layers fold separately.
    static func remapReferenceKey(_ key: String) -> String? {
        key.hasSuffix("num_batches_tracked") ? nil : key
    }

    /// Loads a release's `model.safetensors` (or a checkpoint `NFKMLXWeights.save` wrote) into `net`:
    /// the LSTM's `weight_ih_l<n>`/`weight_hh_l<n>` and summed biases fold into each layer's `Wx`/`Wh`/
    /// `bias`, and a release's convolutions transpose to MLX's channels-last layouts.
    public static func loadWeights(into net: NFKMLXNemotronASRNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        var arrays = checkpoint.arrays
        var folded: [String: MLXArray] = [:]
        let lstmBase = "decoder.lstm."
        for (key, value) in arrays where key.hasPrefix(lstmBase) && key.contains("_l") && !key.contains(".layers.") {
            let suffix = String(key.dropFirst(lstmBase.count))
            guard let marker = suffix.range(of: "_l", options: .backwards) else {
                continue
            }
            let target = "\(lstmBase)layers.\(suffix[marker.upperBound...])."
            switch String(suffix[..<marker.lowerBound]) {
            case "weight_ih": folded[target + "Wx"] = value
            case "weight_hh": folded[target + "Wh"] = value
            case "bias_ih", "bias_hh": folded[target + "bias"] = folded[target + "bias"].map { $0 + value } ?? value
            default: continue
            }
            arrays[key] = nil
        }
        var mapped: [(String, MLXArray)] = []
        for (key, value) in arrays {
            guard let name = remapReferenceKey(key) else {
                continue
            }
            var array = value
            if checkpoint.needsConvTranspose, name.hasSuffix(".weight") {
                if array.ndim == 4 {
                    array = array.transposed(0, 2, 3, 1)
                } else if array.ndim == 3 {
                    array = array.transposed(0, 2, 1)
                }
            }
            mapped.append((name, array))
        }
        mapped += folded.map { ($0.key, $0.value) }
        try NFKMLXWeights.apply(mapped, to: net)
        net.train(false)
    }

    /// The configuration of a release directory's `config.json`.
    static func configuration(directoryURL: URL) throws -> NFKMLXNemotronASRConfiguration {
        let data = try Data(contentsOf: directoryURL.appendingPathComponent("config.json"))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("config.json is not an object")
        }
        if let type = json["model_type"] as? String, !modelTypes.contains(type) {
            throw NFKMLXError.unsupportedConfiguration("config.json's model_type \(type) is not a Nemotron ASR Streaming release")
        }
        return try NFKMLXNemotronASRConfiguration(json: json)
    }

    /// Builds the recognizer from a release directory: `config.json`, `model.safetensors`, and
    /// `tokenizer.json`. Blocking on the load; run off the render thread. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let configuration = try configuration(directoryURL: directoryURL)
            let net = NFKMLXNemotronASRNet(configuration)
            try loadWeights(into: net, from: directoryURL.appendingPathComponent("model.safetensors"))
            let tokenizer = try NFKMLXNemotronASRTokenizer(tokenizerURL: directoryURL.appendingPathComponent("tokenizer.json"))
            return NFKMLXNemotronASRBackend(net: net, tokenizer: tokenizer, identifier: modelName)
        }
    }

    static let requiredFiles = ["config.json", "tokenizer.json"]
    static let optionalFiles = ["processor_config.json", "tokenizer_config.json", "generation_config.json"]
    static let weightFiles = ["model.safetensors"]

    /// Downloads a release into the hub cache and builds the recognizer from it. The public release is
    /// `nvidia/nemotron-speech-streaming-en-0.6b`. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let directory = try NFKMLXReleaseDownload.directory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                                                required: requiredFiles, optional: optionalFiles, weights: weightFiles)
            return try backend(directoryURL: directory)
        }
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers `nemotron-asr-streaming`; the registry's URL is a release directory.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(modelName) { try backend(directoryURL: $0) }
    }

    /// A random-weights recognizer at `configuration`, for shape and stream checks.
    public static func backend(configuration: NFKMLXNemotronASRConfiguration) -> any NFKInferenceBackend {
        let pieces = (0 ..< configuration.vocabulary).map { $0 == configuration.blank ? "<blank>" : "\u{2581}t\($0)" }
        return NFKMLXNemotronASRBackend(net: NFKMLXNemotronASRNet(configuration),
                                        tokenizer: NFKMLXNemotronASRTokenizer(pieces: pieces, specialIds: [configuration.blank]),
                                        identifier: modelName)
    }
}

// MARK: - Backend

private final class NFKNemotronASRHolder: @unchecked Sendable {
    let net: NFKMLXNemotronASRNet
    let tokenizer: NFKMLXNemotronASRTokenizer
    init(_ net: NFKMLXNemotronASRNet, _ tokenizer: NFKMLXNemotronASRTokenizer) {
        self.net = net
        self.tokenizer = tokenizer
    }
}

/// A streaming speech recognizer as an InferKit backend: 16 kHz audio under `NFKInputAudio` (resampled
/// when a clip is at another rate) → the transcript under `NFKOutputText` and one `NFKAudioSegment` per
/// token under `NFKOutputSegments`, timed at its 80 ms encoder frame. A request sets the lookahead under
/// ``NFKMLXGenerationParameterKey/lookaheadFrames``. Its live stream transcribes a source chunk by chunk.
@objc(NFKMLXNemotronASRBackend)
public final class NFKMLXNemotronASRBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKNemotronASRHolder
    private let identifier: String

    init(net: NFKMLXNemotronASRNet, tokenizer: NFKMLXNemotronASRTokenizer, identifier: String) {
        holder = NFKNemotronASRHolder(net, tokenizer)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { [NFKMLXGenerationParameterKey.lookaheadFrames] }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputText, NFKOutputSegments] }

    /// The lookahead a request or stream names, one the release was trained with.
    static func lookahead(_ parameters: [String: Any]?, configuration: NFKMLXNemotronASRConfiguration) throws -> Int {
        guard let value = parameters?[NFKMLXGenerationParameterKey.lookaheadFrames] else {
            return configuration.defaultLookahead
        }
        guard let number = value as? NSNumber, configuration.lookaheads.contains(number.intValue) else {
            throw NSError(domain: NFKInferenceErrorDomain, code: NFKInferenceError.error_InferenceUnsupported.rawValue, userInfo: [
                NSLocalizedDescriptionKey: "NFKMLXParameterLookaheadFrames is one of \(configuration.lookaheads.sorted())",
            ])
        }
        return number.intValue
    }

    /// The text and timed segments of `tokens`.
    static func result(_ tokens: [NFKMLXNemotronASRNet.Token], tokenizer: NFKMLXNemotronASRTokenizer,
                       configuration: NFKMLXNemotronASRConfiguration, atStart: Bool) -> NFKInferenceResult {
        let seconds = configuration.encoderFrameSeconds
        let segments = tokens.enumerated().map { index, token in
            NFKAudioSegment(startSeconds: Double(token.frame) * seconds, endSeconds: Double(token.frame + 1) * seconds,
                            label: tokenizer.text(for: [token.id], atStart: atStart && index == 0), confidence: 1)
        }
        return NFKInferenceResult(outputs: [NFKOutputText: tokenizer.text(for: tokens.map(\.id), atStart: atStart),
                                            NFKOutputSegments: segments])
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, rate) = Self.audio(from: request) else {
            throw NFKMLXError.unsupportedInput
        }
        let configuration = holder.net.configuration
        let lookahead = try Self.lookahead(request.parameters, configuration: configuration)
        let matched = NFKMLXAudioRate.matched(samples, from: rate, to: configuration.sampleRate)
        let tokens = holder.net.recognize(matched, lookahead: lookahead)
        return Self.result(tokens, tokenizer: holder.tokenizer, configuration: configuration, atStart: true)
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

    // MARK: Live streams

    /// Opens a live transcription over mono 16 kHz audio. Each unit's samples join those the units before
    /// left; the stream computes the log-mel frames they complete, and each chunk of frames the lookahead
    /// fixes (one frame then `8·lookahead`, then `8·(lookahead + 1)` at a time) runs through the encoder
    /// from the convolution caches and the attention's key/value window the chunks before left, then the
    /// transducer from its carried state. A unit's result holds the tokens those chunks emitted: their
    /// text under `NFKOutputText`, joined to the earlier units' text as they arrive, and one segment per
    /// token under `NFKOutputSegments`, timed from the stream's start. The tokens are final. The lookahead
    /// is the stream's parameter ``NFKMLXGenerationParameterKey/lookaheadFrames``. A unit at another rate
    /// or channel count fails with `kNFKError_InferenceUnsupported`. Introduced in InferKit 0.4.0.
    @objc(openLiveStreamWithParameters:error:)
    public func openLiveStream(parameters: [String: Any]?) throws -> any NFKLiveStream {
        let net = holder.net
        let tokenizer = holder.tokenizer
        let configuration = net.configuration
        let lookahead = try Self.lookahead(parameters, configuration: configuration)
        var state = NFKNemotronASRStreamState(configuration, lookahead: lookahead)
        return NFKBlockLiveStream(parameters: parameters, unitHandler: { unit, errorPointer in
            switch NFKMLXLiveAudio.monoSamples(unit, sampleRate: configuration.sampleRate) {
            case .failure(let error):
                errorPointer?.pointee = error
                return nil
            case .success(let samples):
                let atStart = !state.hasEmitted
                let tokens = Self.stream(samples, net: net, state: state)
                return Self.result(tokens, tokenizer: tokenizer, configuration: configuration, atStart: atStart)
            }
        }, resetHandler: { state = NFKNemotronASRStreamState(configuration, lookahead: lookahead) },
           closeHandler: { state = NFKNemotronASRStreamState(configuration, lookahead: lookahead) })
    }

    /// Transcribes the chunks `samples` completes after the stream's earlier samples and returns the tokens
    /// they emitted. The features, the encoder, and the decoding reproduce the clip forward chunk by chunk.
    static func stream(_ samples: [Float], net: NFKMLXNemotronASRNet, state: NFKNemotronASRStreamState) -> [NFKMLXNemotronASRNet.Token] {
        state.append(samples, frontEnd: net.frontEnd)
        var tokens: [NFKMLXNemotronASRNet.Token] = []
        while let chunk = state.takeChunk() {
            let encoded = state.carry.streaming { net.encoder(chunk, lookahead: nil) }
            state.carry.evaluate([encoded])
            tokens += net.decode(encoded: encoded, state: state.decoding)
        }
        if !tokens.isEmpty {
            state.hasEmitted = true
        }
        return tokens
    }

    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else {
            return nil
        }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data {
            return NFKMLXWaveFile.read(data)
        }
        return nil
    }
}

/// What a Nemotron ASR live stream carries between units: the preemphasized samples the next frames read,
/// the log-mel frames waiting for a whole chunk, the encoder's carry, and the transducer's state.
final class NFKNemotronASRStreamState {
    let configuration: NFKMLXNemotronASRConfiguration
    let lookahead: Int
    let carry = NFKMLXStreamCarry()
    let decoding = NFKMLXNemotronASRDecodingState()
    var hasEmitted = false
    /// Preemphasized samples from sample `firstSample` on.
    private var signal: [Float] = []
    private var firstSample = 0
    private var lastRaw: Float?
    private var framesComputed = 0
    private var pending: [MLXArray] = []
    private var pendingCount = 0
    private var chunksTaken = 0

    init(_ configuration: NFKMLXNemotronASRConfiguration, lookahead: Int) {
        self.configuration = configuration
        self.lookahead = lookahead
    }

    /// Adds samples and computes the log-mel frames they complete. Frame `f` is the clip's: the transform
    /// centered on sample `f·hop`, half a transform of zeros before the first sample, so it waits for
    /// sample `f·hop + fftSize/2 − 1`.
    func append(_ samples: [Float], frontEnd: NFKNemotronASRFrontEnd) {
        guard !samples.isEmpty else {
            return
        }
        signal += NFKNemotronASRFrontEnd.preemphasized(samples, coefficient: configuration.preemphasis, previous: lastRaw)
        lastRaw = samples.last
        let size = configuration.fftSize, hop = configuration.hopSamples, half = size / 2
        var rows: [Float] = []
        var count = 0
        while true {
            let center = (framesComputed + count) * hop
            let end = center + half
            guard end <= firstSample + signal.count else {
                break
            }
            for index in (center - half) ..< end {
                rows.append(index < 0 ? 0 : signal[index - firstSample])
            }
            count += 1
        }
        guard count > 0 else {
            return
        }
        framesComputed += count
        pending.append(frontEnd.logMel(frames: MLXArray(rows, [count, size])))
        pendingCount += count
        let keepFrom = max(0, framesComputed * hop - half)
        if keepFrom > firstSample {
            signal.removeFirst(keepFrom - firstSample)
            firstSample = keepFrom
        }
    }

    /// The next whole chunk of log-mel frames `[1, frames, mels]`, or nil while one is still filling.
    func takeChunk() -> MLXArray? {
        let need = chunksTaken == 0 ? configuration.firstChunkFrames(lookahead: lookahead)
            : configuration.chunkFrames(lookahead: lookahead)
        guard pendingCount >= need else {
            return nil
        }
        let all = concatenated(pending, axis: 0)
        let chunk = all[0 ..< need]
        let rest = all[need...]
        pending = rest.dim(0) > 0 ? [rest] : []
        pendingCount -= need
        chunksTaken += 1
        return chunk.reshaped([1, need, configuration.mels])
    }
}
