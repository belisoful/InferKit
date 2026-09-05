//
//  NFKMLXChatterbox.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXFFT
import MLXNN

// Chatterbox (Resemble AI, MIT): zero-shot voice-cloning text-to-speech, ported stage by stage from
// the `chatterbox-tts` reference. The reference voice is read three ways: the VoiceEncoder turns it
// into a 256-wide speaker embedding, the S3 speech tokenizer turns it into 25 Hz discrete speech codes
// (the same codes the T3 text-to-speech-token model generates), and S3Gen reads it as a mel prompt and
// an x-vector. This file holds the first two: the VoiceEncoder and the S3 tokenizer.

// MARK: - Voice encoder

/// The VoiceEncoder geometry (`VoiceEncConfig`): a 40-band unscaled power mel over 16 kHz audio, three
/// LSTM layers of 256, a 256-wide projection, and utterance partials of 160 frames.
public struct NFKMLXChatterboxVoiceConfiguration: Sendable {
    public var sampleRate: Int = 16000
    public var mels: Int = 40
    public var fftSize: Int = 400
    public var hopSamples: Int = 160
    public var maximumFrequency: Float = 8000
    public var hiddenSize: Int = 256
    public var embeddingSize: Int = 256
    public var layerCount: Int = 3
    /// Frames per partial utterance (`ve_partial_frames`).
    public var partialFrames: Int = 160
    /// The partial rate `embeds_from_wavs` defaults to (Resemble's 1.3), which sets the partial step.
    public var partialRate: Float = 1.3
    public var minimumCoverage: Float = 0.8
    /// The silence threshold `librosa.effects.trim` is called with, in decibels below the peak.
    public var trimTopDecibels: Float = 20
    public init() {}
    public static let released = NFKMLXChatterboxVoiceConfiguration()
    public static var tiny: NFKMLXChatterboxVoiceConfiguration {
        var configuration = NFKMLXChatterboxVoiceConfiguration()
        configuration.hiddenSize = 16
        configuration.embeddingSize = 8
        configuration.layerCount = 2
        return configuration
    }
}

/// The VoiceEncoder: a three-layer LSTM over mel partials, a projection, a ReLU, and an L2 norm.
public final class NFKMLXChatterboxVoiceEncoderNet: Module {
    @ModuleInfo(key: "lstm") var lstm: [LSTM]
    @ModuleInfo(key: "proj") var proj: Linear
    public let configuration: NFKMLXChatterboxVoiceConfiguration

    public init(_ configuration: NFKMLXChatterboxVoiceConfiguration = .released) {
        self.configuration = configuration
        var layers = [LSTM]()
        for index in 0 ..< configuration.layerCount {
            layers.append(LSTM(inputSize: index == 0 ? configuration.mels : configuration.hiddenSize,
                               hiddenSize: configuration.hiddenSize))
        }
        _lstm.wrappedValue = layers
        _proj.wrappedValue = Linear(configuration.hiddenSize, configuration.embeddingSize)
        super.init()
    }

    /// Embeds a batch of partials `[N, frames, mels]` to unit-length vectors `[N, embedding]`.
    public func embedPartials(_ mels: MLXArray) -> MLXArray {
        var hidden = mels
        var last = mels
        for layer in lstm {
            (hidden, _) = layer(hidden)
            last = hidden
        }
        let raw = relu(proj(last[0..., -1, 0...]))
        return raw / sqrt((raw * raw).sum(axis: -1, keepDims: true))
    }

    /// The utterance embedding of one unscaled mel `[frames, mels]`: the partial embeddings averaged
    /// and re-normalized (`VoiceEncoder.inference`).
    public func embed(mel: MLXArray) -> MLXArray {
        let partials = NFKMLXChatterboxVoiceFrontEnd.partials(of: mel, configuration: configuration)
        let embeddings = embedPartials(partials)
        let mean = embeddings.mean(axis: 0)
        return mean / sqrt((mean * mean).sum())
    }

    /// The speaker embedding of a 16 kHz waveform (`embeds_from_wavs`): trim, mel, partials, embed.
    public func embed(samples: [Float]) -> MLXArray {
        let trimmed = NFKMLXChatterboxVoiceFrontEnd.trimmed(samples, topDecibels: configuration.trimTopDecibels)
        return embed(mel: NFKMLXChatterboxVoiceFrontEnd.mel(trimmed, configuration: configuration))
    }
}

/// The VoiceEncoder's input side: librosa's silence trim, its unscaled power mel, and the partial
/// windows `stride_as_partials` cuts.
public enum NFKMLXChatterboxVoiceFrontEnd {
    /// `librosa.effects.trim(top_db:)`: frames whose RMS falls `topDecibels` below the loudest frame's
    /// are silence, and the signal is cut to the first and last non-silent frame (2048-sample frames at
    /// hop 512, zero-padded by a half frame at both ends).
    public static func trimmed(_ samples: [Float], topDecibels: Float,
                               frameLength: Int = 2048, hop: Int = 512) -> [Float] {
        guard !samples.isEmpty else { return samples }
        let half = frameLength / 2
        let padded = [Float](repeating: 0, count: half) + samples + [Float](repeating: 0, count: half)
        let frames = 1 + (padded.count - frameLength) / hop
        var rms = [Float](repeating: 0, count: frames)
        for frame in 0 ..< frames {
            var power: Float = 0
            for index in 0 ..< frameLength {
                let value = padded[frame * hop + index]
                power += value * value
            }
            rms[frame] = sqrtf(power / Float(frameLength))
        }
        let floor: Float = 1e-5
        let reference = 20 * log10f(max(floor, rms.max() ?? 0))
        let nonSilent = rms.indices.filter { 20 * log10f(max(floor, rms[$0])) - reference > -topDecibels }
        guard let first = nonSilent.first, let last = nonSilent.last else { return [] }
        let start = first * hop
        let end = min(samples.count, (last + 1) * hop)
        return Array(samples[start ..< end])
    }

    /// The unscaled power mel `[frames, mels]` (`melspectrogram`, `mel_type: amp`, power 2): a centered
    /// reflect-padded 400-point STFT at hop 160, every frame kept, through a Slaney filterbank capped
    /// at `maximumFrequency`. No log is taken.
    public static func mel(_ samples: [Float], configuration: NFKMLXChatterboxVoiceConfiguration) -> MLXArray {
        let power = NFKMLXMel.powerSpectrogram(samples, nFFT: configuration.fftSize,
                                              hop: configuration.hopSamples, dropsFinalFrame: false)
        let filters = NFKMLXMel.melFilters(sampleRate: configuration.sampleRate,
                                          bins: configuration.fftSize / 2 + 1, nMels: configuration.mels,
                                          fMaximum: configuration.maximumFrequency)
        return power.matmul(filters)
    }

    /// The partial step `get_frame_step` derives from the rate: `round((sampleRate / rate) / partialFrames)`.
    public static func partialStep(configuration: NFKMLXChatterboxVoiceConfiguration) -> Int {
        Int((Float(configuration.sampleRate) / configuration.partialRate / Float(configuration.partialFrames)).rounded())
    }

    /// `get_num_wins`: how many partials cover `frames`, and the padded length they span.
    static func partialCount(frames: Int, step: Int,
                             configuration: NFKMLXChatterboxVoiceConfiguration) -> (count: Int, target: Int) {
        let window = configuration.partialFrames
        let span = max(frames - window + step, 0)
        var count = span / step
        let remainder = span % step
        if count == 0 || Float(remainder + (window - step)) / Float(window) >= configuration.minimumCoverage {
            count += 1
        }
        return (count, window + step * (count - 1))
    }

    /// Cuts a mel `[frames, mels]` into overlapping partials `[N, partialFrames, mels]`, zero-padding
    /// the tail as the reference does.
    public static func partials(of mel: MLXArray, configuration: NFKMLXChatterboxVoiceConfiguration) -> MLXArray {
        let step = partialStep(configuration: configuration)
        let (count, target) = partialCount(frames: mel.dim(0), step: step, configuration: configuration)
        var padded = mel
        if target > mel.dim(0) {
            padded = concatenated([mel, MLXArray.zeros([target - mel.dim(0), mel.dim(1)])], axis: 0)
        }
        let windows = (0 ..< count).map { padded[($0 * step) ..< ($0 * step + configuration.partialFrames)] }
        return stacked(windows, axis: 0)
    }
}

// MARK: - S3 speech tokenizer

/// The S3TokenizerV2 geometry (`speech_tokenizer_v2_25hz`): Whisper's 128-band log-mel, two stride-2
/// convolutions (100 Hz frames to 25 Hz), six FSMN-attention blocks of width 1280 over 20 heads with a
/// 64-channel rotary, and an 8-channel base-3 finite scalar quantizer (6561 codes).
public struct NFKMLXS3TokenizerConfiguration: Sendable {
    public var sampleRate: Int = 16000
    public var mels: Int = 128
    public var stateSize: Int = 1280
    public var headCount: Int = 20
    public var layerCount: Int = 6
    public var fsmnKernel: Int = 31
    public var rotaryDimensions: Int = 64
    public var rotaryBase: Float = 10000
    public var codeChannels: Int = 8
    public var codeLevels: Int = 3
    public var normEpsilon: Float = 1e-5
    /// Speech tokens per second of audio.
    public var tokenRate: Int = 25
    public init() {}
    public static let released = NFKMLXS3TokenizerConfiguration()
    public static var tiny: NFKMLXS3TokenizerConfiguration {
        var configuration = NFKMLXS3TokenizerConfiguration()
        configuration.stateSize = 32
        configuration.headCount = 2
        configuration.layerCount = 2
        configuration.rotaryDimensions = 16
        return configuration
    }
    public var codebookSize: Int { Int(pow(Double(codeLevels), Double(codeChannels))) }
}

/// Whisper's multi-head attention with the FSMN memory: a depthwise convolution over the values,
/// added to the values and then to the attention output (`FSMNMultiHeadAttention`).
final class NFKS3FSMNAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear
    @ModuleInfo(key: "fsmn_block") var fsmn: Conv1d
    let heads: Int
    let rotaryDimensions: Int
    let rotaryBase: Float

    init(_ configuration: NFKMLXS3TokenizerConfiguration) {
        let width = configuration.stateSize
        heads = configuration.headCount
        rotaryDimensions = configuration.rotaryDimensions
        rotaryBase = configuration.rotaryBase
        _query.wrappedValue = Linear(width, width)
        _key.wrappedValue = Linear(width, width, bias: false)
        _value.wrappedValue = Linear(width, width)
        _out.wrappedValue = Linear(width, width)
        _fsmn.wrappedValue = Conv1d(inputChannels: width, outputChannels: width,
                                    kernelSize: configuration.fsmnKernel,
                                    padding: (configuration.fsmnKernel - 1) / 2, groups: width, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = width / heads
        let v = value(x)
        // The FSMN memory reads the values before the head split, zero-padded so the kernel is centered.
        let memory = fsmn(v) + v
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        }
        let q = MLXFast.RoPE(split(query(x)), dimensions: rotaryDimensions, traditional: false,
                             base: rotaryBase, scale: 1, offset: 0)
        let k = MLXFast.RoPE(split(key(x)), dimensions: rotaryDimensions, traditional: false,
                             base: rotaryBase, scale: 1, offset: 0)
        // The reference scales both q and k by headDim^-0.25, which is one scale of headDim^-0.5.
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: split(v),
                                                         scale: 1 / sqrt(Float(headDim)), mask: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, length, width])
        return out(merged) + memory
    }
}

/// One pre-norm residual block: FSMN attention, then a GELU feed-forward at four times the width.
final class NFKS3Block: Module {
    @ModuleInfo(key: "attn") var attention: NFKS3FSMNAttention
    @ModuleInfo(key: "attn_ln") var attentionNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: [Module]
    @ModuleInfo(key: "mlp_ln") var mlpNorm: LayerNorm

    init(_ configuration: NFKMLXS3TokenizerConfiguration) {
        let width = configuration.stateSize
        _attention.wrappedValue = NFKS3FSMNAttention(configuration)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: width, eps: configuration.normEpsilon)
        // The reference's `nn.Sequential(Linear, GELU, Linear)`: the activation holds index 1.
        _mlp.wrappedValue = [Linear(width, width * 4), NFKS3GELU(), Linear(width * 4, width)]
        _mlpNorm.wrappedValue = LayerNorm(dimensions: width, eps: configuration.normEpsilon)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x + attention(attentionNorm(x))
        let first = mlp[0] as! Linear, last = mlp[2] as! Linear
        x = x + last(gelu(first(mlpNorm(x))))
        return x
    }
}

/// A parameter-free marker holding the Sequential slot of the reference's `GELU`.
final class NFKS3GELU: Module {}

/// The audio encoder (`AudioEncoderV2`): two GELU convolutions then the block stack.
final class NFKS3Encoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [NFKS3Block]

    init(_ configuration: NFKMLXS3TokenizerConfiguration) {
        _conv1.wrappedValue = Conv1d(inputChannels: configuration.mels, outputChannels: configuration.stateSize,
                                     kernelSize: 3, stride: 2, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: configuration.stateSize, outputChannels: configuration.stateSize,
                                     kernelSize: 3, stride: 2, padding: 1)
        _blocks.wrappedValue = (0 ..< configuration.layerCount).map { _ in NFKS3Block(configuration) }
        super.init()
    }

    /// `[batch, frames, mels]` → `[batch, frames / 4, state]`.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = gelu(conv1(mel))
        x = gelu(conv2(x))
        for block in blocks {
            x = block(x)
        }
        return x
    }
}

/// The S3 speech tokenizer: 16 kHz audio → 25 Hz discrete speech codes.
public final class NFKMLXS3TokenizerNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKS3Encoder
    @ModuleInfo(key: "project_down") var projectDown: Linear
    public let configuration: NFKMLXS3TokenizerConfiguration

    public init(_ configuration: NFKMLXS3TokenizerConfiguration = .released) {
        self.configuration = configuration
        _encoder.wrappedValue = NFKS3Encoder(configuration)
        _projectDown.wrappedValue = Linear(configuration.stateSize, configuration.codeChannels)
        super.init()
    }

    /// Whisper's log-mel at 128 bands, `[frames, mels]` (the final STFT frame dropped, as Whisper does).
    public func logMel(_ samples: [Float]) -> MLXArray {
        NFKMLXMel.logMel(samples, sampleRate: configuration.sampleRate, nMels: configuration.mels)[0]
    }

    /// The encoder output `[frames / 4, state]` for one log-mel `[frames, mels]`.
    public func hidden(mel: MLXArray) -> MLXArray {
        encoder(mel.expandedDimensions(axis: 0))[0]
    }

    /// The finite-scalar-quantizer codes of encoder states `[T, state]`: each of the 8 channels is
    /// squashed to `{-1, 0, 1}` and the digits read as a base-3 number.
    public func codes(hidden: MLXArray) -> [Int] {
        let squashed = MLX.round(tanh(projectDown(hidden)) * 0.9990000128746033) + 1
        let powers = MLXArray((0 ..< configuration.codeChannels).map { Float(pow(Double(configuration.codeLevels), Double($0))) })
        let indices = (squashed * powers).sum(axis: -1)
        return indices.asArray(Float.self).map { Int($0.rounded()) }
    }

    /// 16 kHz samples → speech codes, `maximumCodes` truncating the mel to `4 × maximumCodes` frames first,
    /// as the reference's `max_len` does.
    public func tokenize(_ samples: [Float], maximumCodes: Int? = nil) -> [Int] {
        var mel = logMel(samples)
        if let maximumCodes, mel.dim(0) > maximumCodes * 4 {
            mel = mel[0 ..< (maximumCodes * 4)]
        }
        return codes(hidden: hidden(mel: mel))
    }
}

// MARK: - Loading

@objc(NFKMLXChatterbox)
public final class NFKMLXChatterbox: NSObject {
    override private init() {}

    /// Loads the released `ve.safetensors` into the voice encoder. The three PyTorch LSTM layers fold into
    /// MLX's `Wx`/`Wh`/`bias`; the training-only `similarity_weight`/`similarity_bias` are not parameters
    /// of the embedding and are dropped.
    public static func loadVoiceEncoderWeights(into net: NFKMLXChatterboxVoiceEncoderNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var mapped = [(String, MLXArray)]()
        var folded = [String: MLXArray]()
        for (key, value) in checkpoint.arrays {
            if key.hasPrefix("similarity_") { continue }
            if key.hasPrefix("lstm.") {
                let suffix = String(key.dropFirst("lstm.".count))
                guard let l = suffix.range(of: "_l") else { continue }
                let target = "lstm.\(suffix[l.upperBound...])."
                switch String(suffix[..<l.lowerBound]) {
                case "weight_ih": folded[target + "Wx"] = value
                case "weight_hh": folded[target + "Wh"] = value
                case "bias_ih", "bias_hh": folded[target + "bias"] = folded[target + "bias"].map { $0 + value } ?? value
                default: continue
                }
                continue
            }
            mapped.append((key, value))
        }
        mapped.append(contentsOf: folded.map { ($0.key, $0.value) })
        try NFKMLXWeights.apply(mapped, to: net)
        net.train(false)
    }

    /// Loads the S3 tokenizer from the released `s3gen.safetensors` (its `tokenizer.` subtree; the
    /// stored mel filterbank and window are recomputed, not loaded) or from a file holding the subtree
    /// alone. The 3-D convolutions transpose to MLX's `[out, k, in]`.
    public static func loadTokenizerWeights(into net: NFKMLXS3TokenizerNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var mapped = [(String, MLXArray)]()
        for (key, value) in checkpoint.arrays {
            var name = key
            if name.hasPrefix("tokenizer.") { name.removeFirst("tokenizer.".count) }
            else if key.contains(".") && !key.hasPrefix("encoder.") && !key.hasPrefix("quantizer.") { continue }
            if name == "_mel_filters" || name == "window" { continue }
            if name.hasPrefix("quantizer._codebook.project_down.") {
                name = "project_down." + name.dropFirst("quantizer._codebook.project_down.".count)
            }
            var array = value
            if checkpoint.needsConvTranspose, name.hasSuffix(".weight"), array.ndim == 3 {
                array = array.transposed(0, 2, 1)
            }
            mapped.append((name, array))
        }
        try NFKMLXWeights.apply(mapped, to: net)
        net.train(false)
    }
}
