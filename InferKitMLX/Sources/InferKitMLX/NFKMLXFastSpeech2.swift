//
//  NFKMLXFastSpeech2.swift
//  InferKitMLX
//
//  The FastSpeech2 conformer acoustic model (espnet's, through the transformers layout): phonemes →
//  mel spectrogram. This is the trained half the TTS chain was missing — `NFKMLXHiFiGANNet` turns the
//  mel into a waveform, and its released weights are already at parity.
//
//  Module keys ARE the released checkpoint's names (`encoder.conformer_layers.N.self_attn.linear_q`),
//  so the release loads with only the convolution transposes.
//
//  Introduced in InferKit 0.1.0.
//

import Foundation
import MLX
import MLXNN

public struct NFKMLXFastSpeech2Configuration: Sendable {
    public var vocabularySize: Int = 78
    public var hiddenSize: Int = 384
    public var headCount: Int = 2
    public var encoderLayers: Int = 4
    public var decoderLayers: Int = 4
    public var linearUnits: Int = 1536
    public var positionwiseKernel: Int = 3
    public var encoderConvKernel: Int = 7
    public var decoderConvKernel: Int = 31
    public var melBins: Int = 80
    public var durationChannels: Int = 256
    public var durationLayers: Int = 2
    public var durationKernel: Int = 3
    public var pitchChannels: Int = 256
    public var pitchLayers: Int = 5
    public var pitchKernel: Int = 5
    public var energyChannels: Int = 256
    public var energyLayers: Int = 2
    public var energyKernel: Int = 3
    public var varianceEmbedKernel: Int = 1
    public var postnetLayers: Int = 5
    public var postnetChannels: Int = 256
    public var postnetKernel: Int = 5
    public init() {}
}

/// Transformer-XL relative positions: the table runs +T−1 … −(T−1), and the input scales by √d.
final class NFKFS2RelPositionalEncoding {
    let dimensions: Int
    init(dimensions: Int) { self.dimensions = dimensions }

    func callAsFunction(_ x: MLXArray) -> (scaled: MLXArray, positions: MLXArray) {
        let length = x.shape[1]
        var table = [Float](repeating: 0, count: (2 * length - 1) * dimensions)
        for row in 0 ..< (2 * length - 1) {
            // Row 0 is position length−1 (the most positive), the middle row position 0.
            let position = Float(length - 1 - row)
            for pair in stride(from: 0, to: dimensions, by: 2) {
                let angle = position * expf(-logf(10_000) * Float(pair) / Float(dimensions))
                table[row * dimensions + pair] = sinf(angle)
                if pair + 1 < dimensions { table[row * dimensions + pair + 1] = cosf(angle) }
            }
        }
        let positions = table.withUnsafeBufferPointer {
            MLXArray($0, [1, 2 * length - 1, dimensions])
        }
        return (x * sqrt(Float(dimensions)), positions)
    }
}

/// Multi-head attention with the Transformer-XL relative-position term.
final class NFKFS2Attention: Module {
    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    let heads: Int
    let headDimensions: Int

    init(hiddenSize: Int, heads: Int) {
        self.heads = heads
        headDimensions = hiddenSize / heads
        _linearQ.wrappedValue = Linear(hiddenSize, hiddenSize)
        _linearK.wrappedValue = Linear(hiddenSize, hiddenSize)
        _linearV.wrappedValue = Linear(hiddenSize, hiddenSize)
        _linearOut.wrappedValue = Linear(hiddenSize, hiddenSize)
        _linearPos.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        _posBiasU.wrappedValue = MLXArray.zeros([heads, hiddenSize / heads])
        _posBiasV.wrappedValue = MLXArray.zeros([heads, hiddenSize / heads])
        super.init()
    }

    /// The paper's shifting trick: pad a zero column, fold, and the diagonal bands line up so entry
    /// `[t1, t2]` reads the score at relative distance `t1 - t2`.
    private func shifted(_ scores: MLXArray) -> MLXArray {
        let (batch, heads, time, width) = (scores.shape[0], scores.shape[1],
                                           scores.shape[2], scores.shape[3])
        let padded = concatenated([MLXArray.zeros([batch, heads, time, 1]), scores], axis: -1)
        let folded = padded.reshaped([batch, heads, width + 1, time])
        return folded[0..., 0..., 1...].reshaped([batch, heads, time, width])[.ellipsis,
                                                                              0 ..< (width / 2 + 1)]
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        let queries = linearQ(x).reshaped([batch, length, heads, headDimensions])
        let keys = linearK(x).reshaped([batch, length, heads, headDimensions])
        let values = linearV(x).reshaped([batch, length, heads, headDimensions])
        let encoded = linearPos(positions).reshaped([1, -1, heads, headDimensions])

        let withU = (queries + posBiasU).transposed(0, 2, 1, 3)
        let withV = (queries + posBiasV).transposed(0, 2, 1, 3)
        let content = matmul(withU, keys.transposed(0, 2, 3, 1))
        let position = shifted(matmul(withV, encoded.transposed(0, 2, 3, 1)))
        let scores = (content + position) / sqrt(Float(headDimensions))

        let attended = matmul(softmax(scores, axis: -1), values.transposed(0, 2, 1, 3))
        return linearOut(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDimensions]))
    }
}

/// The conformer convolution: pointwise ×2 → GLU → depthwise → BatchNorm → silu → pointwise.
final class NFKFS2ConvModule: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwise1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo(key: "norm") var norm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwise2: Conv1d

    init(channels: Int, kernel: Int) {
        _pointwise1.wrappedValue = Conv1d(inputChannels: channels, outputChannels: 2 * channels,
                                          kernelSize: 1)
        _depthwise.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels,
                                         kernelSize: kernel, padding: (kernel - 1) / 2,
                                         groups: channels)
        _norm.wrappedValue = BatchNorm(featureCount: channels)
        _pointwise2.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels,
                                          kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let doubled = pointwise1(x)
        let half = doubled.shape[2] / 2
        let gated = doubled[.ellipsis, 0 ..< half] * sigmoid(doubled[.ellipsis, half...])
        return pointwise2(silu(norm(depthwise(gated))))
    }
}

/// The position-wise feed-forward as two k-wide convolutions, FastSpeech's replacement for linears.
final class NFKFS2FeedForward: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(channels: Int, hidden: Int, kernel: Int) {
        _conv1.wrappedValue = Conv1d(inputChannels: channels, outputChannels: hidden,
                                     kernelSize: kernel, padding: (kernel - 1) / 2)
        _conv2.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: channels,
                                     kernelSize: kernel, padding: (kernel - 1) / 2)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv2(relu(conv1(x))) }
}

/// One conformer layer in the released POST-norm arrangement (`normalize_before: false`): each
/// sublayer's norm runs after its residual add, and a final norm closes the layer.
final class NFKFS2ConformerLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKFS2Attention
    @ModuleInfo(key: "feed_forward") var feedForward: NFKFS2FeedForward
    @ModuleInfo(key: "feed_forward_macaron") var macaron: NFKFS2FeedForward
    @ModuleInfo(key: "conv_module") var convolution: NFKFS2ConvModule
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: LayerNorm
    @ModuleInfo(key: "ff_layer_norm") var feedForwardNorm: LayerNorm
    @ModuleInfo(key: "ff_macaron_layer_norm") var macaronNorm: LayerNorm
    @ModuleInfo(key: "conv_layer_norm") var convolutionNorm: LayerNorm
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(_ c: NFKMLXFastSpeech2Configuration, convKernel: Int) {
        _attention.wrappedValue = NFKFS2Attention(hiddenSize: c.hiddenSize, heads: c.headCount)
        _feedForward.wrappedValue = NFKFS2FeedForward(channels: c.hiddenSize, hidden: c.linearUnits,
                                                      kernel: c.positionwiseKernel)
        _macaron.wrappedValue = NFKFS2FeedForward(channels: c.hiddenSize, hidden: c.linearUnits,
                                                  kernel: c.positionwiseKernel)
        _convolution.wrappedValue = NFKFS2ConvModule(channels: c.hiddenSize, kernel: convKernel)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize)
        _feedForwardNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize)
        _macaronNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize)
        _convolutionNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize)
        _finalNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        var hidden = macaronNorm(x + 0.5 * macaron(x))
        hidden = attentionNorm(hidden + attention(hidden, positions: positions))
        hidden = convolutionNorm(hidden + convolution(hidden))
        hidden = feedForwardNorm(hidden + 0.5 * feedForward(hidden))
        return finalNorm(hidden)
    }
}

/// A predictor layer: convolution → relu → LayerNorm over channels.
final class NFKFS2PredictorLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm

    init(input: Int, channels: Int, kernel: Int) {
        _conv.wrappedValue = Conv1d(inputChannels: input, outputChannels: channels,
                                    kernelSize: kernel, padding: (kernel - 1) / 2)
        _norm.wrappedValue = LayerNorm(dimensions: channels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(relu(conv(x))) }
}

/// Duration, pitch, and energy share this shape: predictor layers into a one-wide linear.
final class NFKFS2VariancePredictor: Module {
    @ModuleInfo(key: "conv_layers") var layers: [NFKFS2PredictorLayer]
    @ModuleInfo(key: "linear") var linear: Linear

    init(input: Int, channels: Int, kernel: Int, count: Int) {
        _layers.wrappedValue = (0 ..< count).map {
            NFKFS2PredictorLayer(input: $0 == 0 ? input : channels, channels: channels, kernel: kernel)
        }
        _linear.wrappedValue = Linear(channels, 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for layer in layers { hidden = layer(hidden) }
        return linear(hidden)                                  // [batch, length, 1]
    }
}

/// A predicted scalar per position, embedded by a one-wide convolution.
final class NFKFS2VarianceEmbedding: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, kernel: Int) {
        _conv.wrappedValue = Conv1d(inputChannels: 1, outputChannels: channels,
                                    kernelSize: kernel, padding: (kernel - 1) / 2)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// One postnet stage: bias-free convolution → BatchNorm → tanh (except the last stage).
final class NFKFS2PostnetLayer: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "batch_norm") var norm: BatchNorm
    let activates: Bool

    init(input: Int, output: Int, kernel: Int, activates: Bool) {
        _conv.wrappedValue = Conv1d(inputChannels: input, outputChannels: output,
                                    kernelSize: kernel, padding: (kernel - 1) / 2, bias: false)
        _norm.wrappedValue = BatchNorm(featureCount: output)
        self.activates = activates
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = norm(conv(x))
        return activates ? tanh(normalized) : normalized
    }
}

/// The mel head: a linear to the bins, refined by a convolutional postnet whose output adds back.
final class NFKFS2Postnet: Module {
    @ModuleInfo(key: "feat_out") var featOut: Linear
    @ModuleInfo(key: "layers") var layers: [NFKFS2PostnetLayer]

    init(_ c: NFKMLXFastSpeech2Configuration) {
        _featOut.wrappedValue = Linear(c.hiddenSize, c.melBins)
        _layers.wrappedValue = (0 ..< c.postnetLayers).map { index in
            NFKFS2PostnetLayer(input: index == 0 ? c.melBins : c.postnetChannels,
                               output: index == c.postnetLayers - 1 ? c.melBins : c.postnetChannels,
                               kernel: c.postnetKernel,
                               activates: index < c.postnetLayers - 1)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (before: MLXArray, after: MLXArray) {
        let before = featOut(x)
        var refined = before
        for layer in layers { refined = layer(refined) }
        return (before, before + refined)
    }
}

/// The phoneme encoder or the mel decoder: the same conformer stack, with an embedding only on the
/// encoder side.
final class NFKFS2Stack: Module {
    @ModuleInfo(key: "embed") var embed: Embedding?
    @ModuleInfo(key: "conformer_layers") var layers: [NFKFS2ConformerLayer]
    let positional: NFKFS2RelPositionalEncoding

    init(_ c: NFKMLXFastSpeech2Configuration, layerCount: Int, convKernel: Int, embeds: Bool) {
        _embed.wrappedValue = embeds
            ? Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize) : nil
        _layers.wrappedValue = (0 ..< layerCount).map { _ in
            NFKFS2ConformerLayer(c, convKernel: convKernel)
        }
        positional = NFKFS2RelPositionalEncoding(dimensions: c.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = embed.map { $0(x) } ?? x
        let (scaled, positions) = positional(hidden)
        hidden = scaled
        for layer in layers { hidden = layer(hidden, positions: positions) }
        return hidden
    }
}

/// The complete acoustic model, in the released layout.
public final class NFKMLXFastSpeech2Net: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKFS2Stack
    @ModuleInfo(key: "decoder") var decoder: NFKFS2Stack
    @ModuleInfo(key: "duration_predictor") var durationPredictor: NFKFS2VariancePredictor
    @ModuleInfo(key: "pitch_predictor") var pitchPredictor: NFKFS2VariancePredictor
    @ModuleInfo(key: "energy_predictor") var energyPredictor: NFKFS2VariancePredictor
    @ModuleInfo(key: "pitch_embed") var pitchEmbed: NFKFS2VarianceEmbedding
    @ModuleInfo(key: "energy_embed") var energyEmbed: NFKFS2VarianceEmbedding
    @ModuleInfo(key: "speech_decoder_postnet") var postnet: NFKFS2Postnet

    let configuration: NFKMLXFastSpeech2Configuration

    init(_ c: NFKMLXFastSpeech2Configuration) {
        configuration = c
        _encoder.wrappedValue = NFKFS2Stack(c, layerCount: c.encoderLayers,
                                            convKernel: c.encoderConvKernel, embeds: true)
        _decoder.wrappedValue = NFKFS2Stack(c, layerCount: c.decoderLayers,
                                            convKernel: c.decoderConvKernel, embeds: false)
        _durationPredictor.wrappedValue = NFKFS2VariancePredictor(
            input: c.hiddenSize, channels: c.durationChannels, kernel: c.durationKernel,
            count: c.durationLayers)
        _pitchPredictor.wrappedValue = NFKFS2VariancePredictor(
            input: c.hiddenSize, channels: c.pitchChannels, kernel: c.pitchKernel,
            count: c.pitchLayers)
        _energyPredictor.wrappedValue = NFKFS2VariancePredictor(
            input: c.hiddenSize, channels: c.energyChannels, kernel: c.energyKernel,
            count: c.energyLayers)
        _pitchEmbed.wrappedValue = NFKFS2VarianceEmbedding(channels: c.hiddenSize,
                                                           kernel: c.varianceEmbedKernel)
        _energyEmbed.wrappedValue = NFKFS2VarianceEmbedding(channels: c.hiddenSize,
                                                            kernel: c.varianceEmbedKernel)
        _postnet.wrappedValue = NFKFS2Postnet(c)
        super.init()
    }

    /// Phoneme ids → mel `[1, frames, melBins]`, with the intermediates a seam comparison reads.
    ///
    /// The variance order is the reference's: pitch and energy are predicted per PHONEME on the
    /// encoder output and embedded back into it BEFORE the durations stretch it to frames.
    func generate(_ tokens: MLXArray)
        -> (mel: MLXArray, encoded: MLXArray, durations: [Int], pitch: MLXArray, energy: MLXArray) {
        let encoded = encoder(tokens)
        let pitch = pitchPredictor(encoded)
        let energy = energyPredictor(encoded)
        // Inference reads the duration in the linear domain: `round(exp(x) - 1)`, floored at zero.
        let logDurations = durationPredictor(encoded).squeezed(axis: -1)
        let durations = logDurations.exp().asArray(Float.self).map {
            max(0, Int(($0 - 1).rounded()))
        }

        var hidden = encoded + pitchEmbed(pitch) + energyEmbed(energy)
        hidden = Self.lengthRegulated(hidden, durations: durations)
        let decoded = decoder(hidden)
        let (_, after) = postnet(decoded)
        return (after, encoded, durations, pitch, energy)
    }

    /// Repeats each position by its duration — FastSpeech's length regulator, with the reference's
    /// guard that an all-zero prediction still emits one frame.
    static func lengthRegulated(_ x: MLXArray, durations: [Int]) -> MLXArray {
        var counts = durations
        if counts.allSatisfy({ $0 == 0 }) { counts = counts.map { _ in 1 } }
        var indices = [Int32]()
        for (index, count) in counts.enumerated() {
            indices.append(contentsOf: [Int32](repeating: Int32(index), count: count))
        }
        return x.take(MLXArray(indices), axis: 1)
    }
}

/// Building the acoustic model and loading the released checkpoint.
public final class NFKMLXFastSpeech2: NSObject {

    static func makeNet(
        _ configuration: NFKMLXFastSpeech2Configuration = NFKMLXFastSpeech2Configuration()
    ) -> NFKMLXFastSpeech2Net {
        let net = NFKMLXFastSpeech2Net(configuration)
        net.train(false)                       // the batch norms run on their released statistics
        return net
    }

    /// Loads the released checkpoint. Module keys are the checkpoint's names, so only the
    /// convolution layouts translate; the batch-norm step counters have no counterpart.
    static func loadWeights(into net: NFKMLXFastSpeech2Net, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            if key.hasSuffix("num_batches_tracked") { return nil }
            // The raw transformers release prefixes every key with `model.`; the offline converter
            // strips it, so a converted file never carries it.
            let name = key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key
            if checkpoint.needsConvTranspose, value.ndim == 3 {
                return (name, value.transposed(0, 2, 1))
            }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

/// A complete trained voice: phoneme symbols → mel through the acoustic model, mel → waveform
/// through the vocoder. The two released checkpoints share their mel geometry (80 bins, 22050 Hz,
/// hop 256), which is what lets them chain without resampling.
public final class NFKMLXVoice {
    let acoustic: NFKMLXFastSpeech2Net
    let vocoder: NFKMLXHiFiGANNet
    let vocabulary: [String: Int]

    /// The LJSpeech release speaks at this rate.
    public static let sampleRate = 22_050

    init(acoustic: NFKMLXFastSpeech2Net, vocoder: NFKMLXHiFiGANNet,
         vocabulary: [String: Int]) {
        self.acoustic = acoustic
        self.vocoder = vocoder
        self.vocabulary = vocabulary
    }

    /// Builds the voice from the two released checkpoints and the release's own phoneme vocabulary,
    /// which is what "matching symbol tables" means: the ids the acoustic model was trained on.
    public static func voice(acousticURL: URL, vocoderURL: URL,
                             vocabularyURL: URL) throws -> NFKMLXVoice {
        let acoustic = NFKMLXFastSpeech2.makeNet()
        try NFKMLXFastSpeech2.loadWeights(into: acoustic, from: acousticURL)
        let vocoder = NFKMLXHiFiGANNet(NFKMLXHiFiGANConfiguration())
        try NFKMLXHiFiGAN.loadWeights(into: vocoder, from: vocoderURL)
        let data = try Data(contentsOf: vocabularyURL)
        guard let vocabulary = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(vocabularyURL.lastPathComponent) is not a symbol → id map")
        }
        return NFKMLXVoice(acoustic: acoustic, vocoder: vocoder, vocabulary: vocabulary)
    }

    /// The vocabulary's ids for a phoneme sequence; a symbol outside it becomes `<unk>`.
    public func identifiers(for phonemes: [String]) -> [Int] {
        let unknown = vocabulary["<unk>"] ?? 0
        return phonemes.map { vocabulary[$0] ?? unknown }
    }

    /// Phoneme symbols (the release's ARPAbet-with-stress set) → mono samples in `-1...1` at
    /// ``sampleRate``.
    public func speak(phonemes: [String]) -> [Float] {
        let ids = identifiers(for: phonemes)
        let (mel, _, _, _, _) = acoustic.generate(
            MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count]))
        let wave = vocoder.waveform(mel)
        return wave.reshaped([-1]).asArray(Float.self)
    }

    /// The voice as a text → audio backend: the caller's phonemizer turns the prompt into the
    /// release's symbols, and the chain does the rest.
    public func makeSpeechBackend(phonemize: @escaping @Sendable (String) -> [String])
        -> NFKMLXSpeechBackend {
        let holder = NFKMLXVoiceHolder(self)
        return NFKMLXSpeechBackend(
            identifier: "fastspeech2-voice",
            configuration: NFKMLXSpeechConfiguration(sampleRate: Self.sampleRate)) { text, _ in
            let samples = holder.voice.speak(phonemes: phonemize(text))
            return samples.withUnsafeBufferPointer { MLXArray($0, [samples.count]) }
        }
    }
}

private final class NFKMLXVoiceHolder: @unchecked Sendable {
    let voice: NFKMLXVoice
    init(_ voice: NFKMLXVoice) { self.voice = voice }
}
