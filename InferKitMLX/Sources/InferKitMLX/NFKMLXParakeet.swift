//
//  NFKMLXParakeet.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

// Parakeet-TDT (NVIDIA NeMo), a second on-device speech recognizer beside Whisper: a FastConformer
// encoder — a depthwise-striding 8× convolutional subsampler, then relative-position (Transformer-XL)
// conformer layers with a macaron feed-forward, a gated depthwise convolution module, and no biases —
// and a token-and-duration transducer: an LSTM prediction network and a joint that emits, per step, a
// token distribution AND a duration distribution over how many encoder frames to skip. Greedy TDT
// decoding therefore visits far fewer frames than a plain transducer. Ported from NeMo's own
// EncDecRNNTBPEModel; the released `parakeet-tdt-0.6b-v2` is CC-BY-4.0.

// MARK: - Configuration

public struct NFKMLXParakeetConfiguration: Sendable {
    // Front end (the NeMo AudioToMelSpectrogramPreprocessor).
    public var sampleRate: Int = 16000
    public var mels: Int = 128
    public var windowSamples: Int = 400
    public var hopSamples: Int = 160
    public var fftSize: Int = 512
    public var preemphasis: Float = 0.97
    // Encoder.
    public var dModel: Int = 1024
    public var layers: Int = 24
    public var heads: Int = 8
    public var feedForwardExpansion: Int = 4
    public var convKernel: Int = 9
    public var subsamplingChannels: Int = 256
    public var subsamplingFactor: Int = 8
    // Decoder / joint.
    public var vocabulary: Int = 1024
    public var predictionHidden: Int = 640
    public var predictionLayers: Int = 2
    public var jointHidden: Int = 640
    public var durations: [Int] = [0, 1, 2, 3, 4]
    public var maxSymbolsPerFrame: Int = 10

    public init() {}
    /// The released `nvidia/parakeet-tdt-0.6b-v2`.
    public static let tdt06B = NFKMLXParakeetConfiguration()
    /// A shrunk geometry for shape tests.
    public static var tiny: NFKMLXParakeetConfiguration {
        var c = NFKMLXParakeetConfiguration()
        c.mels = 16; c.dModel = 32; c.layers = 2; c.heads = 2; c.convKernel = 5; c.subsamplingChannels = 8
        c.vocabulary = 12; c.predictionHidden = 16; c.jointHidden = 16
        return c
    }
    var blank: Int { vocabulary }
}

// MARK: - Front end

/// The NeMo mel front end at Parakeet's settings: preemphasis, a zero-padded centered 512-point
/// transform under a symmetric Hann window, the power spectrum through the checkpoint's mel filterbank,
/// `log(x + 2^-24)`, then PER-FEATURE normalization (each mel band centered and scaled by its unbiased
/// standard deviation over time, plus 1e-5). Held outside the module graph so the constants stay out of
/// `parameters()`.
final class NFKParakeetFrontEnd {
    private let configuration: NFKMLXParakeetConfiguration
    private(set) var window: MLXArray          // [fftSize], centered
    private(set) var filterbank: MLXArray      // [mels, bins]

    init(_ configuration: NFKMLXParakeetConfiguration) {
        self.configuration = configuration
        let count = configuration.windowSamples
        let raw = (0 ..< count).map { 0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(count - 1)) }
        window = Self.centered(MLXArray(raw), in: configuration.fftSize)
        filterbank = NFKMLXMel.melFilters(sampleRate: configuration.sampleRate, bins: configuration.fftSize / 2 + 1,
                                          nMels: configuration.mels).transposed(1, 0)
    }

    func load(window raw: MLXArray) { window = Self.centered(raw, in: configuration.fftSize) }
    func load(filterbank raw: MLXArray) { filterbank = raw.ndim == 3 ? raw[0] : raw }

    private static func centered(_ raw: MLXArray, in size: Int) -> MLXArray {
        let total = size - raw.shape[0]
        guard total > 0 else { return raw }
        return MLX.padded(raw, widths: [IntOrPair((total / 2, total - total / 2))], mode: .constant)
    }

    /// `[1, frames, mels]`, normalized.
    func features(_ samples: [Float]) -> MLXArray {
        let size = configuration.fftSize, hop = configuration.hopSamples
        var signal = samples
        if configuration.preemphasis != 0 {
            for index in stride(from: signal.count - 1, to: 0, by: -1) {
                signal[index] -= configuration.preemphasis * signal[index - 1]
            }
        }
        // torch.stft(center: true, pad_mode: "constant"): zeros, not a reflection.
        let pad = size / 2
        let padded = [Float](repeating: 0, count: pad) + signal + [Float](repeating: 0, count: pad)
        let frames = 1 + (padded.count - size) / hop
        var frameData = [Float](repeating: 0, count: frames * size)
        for frame in 0 ..< frames {
            for offset in 0 ..< size { frameData[frame * size + offset] = padded[frame * hop + offset] }
        }
        let windowed = frameData.withUnsafeBufferPointer { MLXArray($0, [frames, size]) } * window
        let spectrum = rfft(windowed, axis: 1)
        let power = spectrum.realPart() * spectrum.realPart() + spectrum.imaginaryPart() * spectrum.imaginaryPart()
        var mel = log(power.matmul(filterbank.transposed(1, 0)) + MLXArray(powf(2, -24)))   // [frames, mels]
        // The reference's valid length is `floor((samples + pad − n_fft) / hop)` — WITHOUT the `+ 1` the
        // transform's own frame count carries — so the transform's last frame lies past the valid length:
        // it is zeroed, excluded from the normalization statistics, and masked through the encoder.
        // Cropping to the valid length is exactly that through the stride-2 subsampler.
        let valid = min(frames, (samples.count + 2 * pad - size) / hop)
        mel = mel[0 ..< valid]
        // per_feature: the unbiased standard deviation over time, guarded by 1e-5.
        let mean = mel.mean(axis: 0, keepDims: true)
        let centeredMel = mel - mean
        let variance = (centeredMel * centeredMel).sum(axis: 0, keepDims: true) / Float(max(valid - 1, 1))
        let normalized = centeredMel / (sqrt(variance) + 1e-5)
        return normalized.reshaped([1, valid, configuration.mels])
    }
}

// MARK: - Subsampling

/// The depthwise-striding subsampler: a full 3×3 stride-2 convolution, then two (depthwise 3×3 stride-2,
/// pointwise 1×1) pairs, ReLU after each stage, over the `(time, mel)` plane as an image with one
/// channel; the result flattens `(channel, mel)` per frame into a linear projection. The `nn.Sequential`
/// indices (ReLU at 1, 4, 7) are kept with marker modules so the checkpoint keys match.
final class NFKParakeetSubsampling: Module {
    @ModuleInfo(key: "conv") var conv: [Module]
    @ModuleInfo(key: "out") var out: Linear

    init(_ config: NFKMLXParakeetConfiguration) {
        let c = config.subsamplingChannels
        var stack: [Module] = [
            Conv2d(inputChannels: 1, outputChannels: c, kernelSize: 3, stride: 2, padding: 1), Module(),
        ]
        let stages = Int(log2(Double(config.subsamplingFactor)))
        for _ in 1 ..< stages {
            stack.append(Conv2d(inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1, groups: c))
            stack.append(Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1))
            stack.append(Module())
        }
        _conv.wrappedValue = stack
        let reducedMels = (0 ..< stages).reduce(config.mels) { m, _ in (m + 2 - 3) / 2 + 1 }
        _out.wrappedValue = Linear(c * reducedMels, config.dModel)
    }

    /// `[1, T, mels]` → `[1, T/8, dModel]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.expandedDimensions(axis: 3)                               // [1, T, F, 1] NHWC
        for module in conv {
            if let layer = module as? Conv2d { h = layer(h) } else { h = relu(h) }
        }
        let (b, t, f, c) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        return out(h.transposed(0, 1, 3, 2).reshaped([b, t, c * f]))        // (channel, mel) order
    }
}

// MARK: - Conformer layer

/// Relative-position multi-head attention (Transformer-XL): content and position scores with two learned
/// per-head biases, the position scores realigned by the appendix-B shift. No biases on the projections.
final class NFKParakeetAttention: Module {
    @ModuleInfo(key: "linear_q") var q: Linear
    @ModuleInfo(key: "linear_k") var k: Linear
    @ModuleInfo(key: "linear_v") var v: Linear
    @ModuleInfo(key: "linear_out") var outProj: Linear
    @ModuleInfo(key: "linear_pos") var pos: Linear
    @ParameterInfo(key: "pos_bias_u") var biasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var biasV: MLXArray
    let heads: Int
    let headDim: Int

    init(_ config: NFKMLXParakeetConfiguration) {
        heads = config.heads
        headDim = config.dModel / config.heads
        let d = config.dModel
        _q.wrappedValue = Linear(d, d, bias: false)
        _k.wrappedValue = Linear(d, d, bias: false)
        _v.wrappedValue = Linear(d, d, bias: false)
        _outProj.wrappedValue = Linear(d, d, bias: false)
        _pos.wrappedValue = Linear(d, d, bias: false)
        _biasU.wrappedValue = MLXArray.zeros([heads, headDim])
        _biasV.wrappedValue = MLXArray.zeros([heads, headDim])
    }

    /// `x` `[b, t1, t2]`-shaped scores over `2T−1` positions → the shifted `[b, h, T, 2T−1]`.
    private func relShift(_ x: MLXArray) -> MLXArray {
        let (b, h, t1, t2) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let paddedScores = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair(0), IntOrPair((1, 0))], mode: .constant)
        let viewed = paddedScores.reshaped([b, h, t2 + 1, t1])
        return viewed[0..., 0..., 1..., 0...].reshaped([b, h, t1, t2])
    }

    /// `x` `[B, T, D]`, `posEmb` `[1, 2T−1, D]`.
    func callAsFunction(_ x: MLXArray, posEmb: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        func split(_ y: MLXArray) -> MLXArray { y.reshaped([b, -1, heads, headDim]).transposed(0, 2, 1, 3) }
        let qh = q(x).reshaped([b, t, heads, headDim])                      // [b, T, h, dk]
        let kh = split(k(x)), vh = split(v(x))
        let p = pos(posEmb).reshaped([1, -1, heads, headDim]).transposed(0, 2, 1, 3)   // [1, h, 2T−1, dk]
        let qU = (qh + biasU).transposed(0, 2, 1, 3)
        let qV = (qh + biasV).transposed(0, 2, 1, 3)
        var bd = relShift(matmul(qV, p.transposed(0, 1, 3, 2)))              // [b, h, T, 2T−1]
        bd = bd[0..., 0..., 0..., 0 ..< t]
        let ac = matmul(qU, kh.transposed(0, 1, 3, 2))                      // [b, h, T, T]
        let scores = (ac + bd) / sqrt(Float(headDim))
        let attended = matmul(softmax(scores, axis: -1), vh)                // [b, h, T, dk]
        return outProj(attended.transposed(0, 2, 1, 3).reshaped([b, t, heads * headDim]))
    }
}

/// The conformer convolution module: pointwise 1×1 to twice the width, a GLU, a depthwise convolution,
/// BatchNorm, Swish, pointwise back. No biases.
final class NFKParakeetConvModule: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwise1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo(key: "batch_norm") var norm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwise2: Conv1d

    init(_ config: NFKMLXParakeetConfiguration) {
        let d = config.dModel
        _pointwise1.wrappedValue = Conv1d(inputChannels: d, outputChannels: 2 * d, kernelSize: 1, bias: false)
        _depthwise.wrappedValue = Conv1d(inputChannels: d, outputChannels: d, kernelSize: config.convKernel,
                                         padding: (config.convKernel - 1) / 2, groups: d, bias: false)
        _norm.wrappedValue = BatchNorm(featureCount: d)
        _pointwise2.wrappedValue = Conv1d(inputChannels: d, outputChannels: d, kernelSize: 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated = pointwise1(x)
        let half = gated.dim(2) / 2
        let glu = gated[0..., 0..., 0 ..< half] * sigmoid(gated[0..., 0..., half...])
        return pointwise2(silu(norm(depthwise(glu))))
    }
}

/// Linear → Swish → Linear, no biases.
final class NFKParakeetFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ config: NFKMLXParakeetConfiguration) {
        let d = config.dModel, ff = config.dModel * config.feedForwardExpansion
        _linear1.wrappedValue = Linear(d, ff, bias: false)
        _linear2.wrappedValue = Linear(ff, d, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

/// One pre-norm conformer block: half a feed-forward, attention, the convolution module, half a
/// feed-forward, then an output norm.
final class NFKParakeetLayer: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFF1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var ff1: NFKParakeetFeedForward
    @ModuleInfo(key: "norm_self_att") var normAttn: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: NFKParakeetAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: NFKParakeetConvModule
    @ModuleInfo(key: "norm_feed_forward2") var normFF2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var ff2: NFKParakeetFeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(_ config: NFKMLXParakeetConfiguration) {
        let d = config.dModel
        _normFF1.wrappedValue = LayerNorm(dimensions: d)
        _ff1.wrappedValue = NFKParakeetFeedForward(config)
        _normAttn.wrappedValue = LayerNorm(dimensions: d)
        _attention.wrappedValue = NFKParakeetAttention(config)
        _normConv.wrappedValue = LayerNorm(dimensions: d)
        _conv.wrappedValue = NFKParakeetConvModule(config)
        _normFF2.wrappedValue = LayerNorm(dimensions: d)
        _ff2.wrappedValue = NFKParakeetFeedForward(config)
        _normOut.wrappedValue = LayerNorm(dimensions: d)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray) -> MLXArray {
        var residual = x + 0.5 * ff1(normFF1(x))
        residual = residual + attention(normAttn(residual), posEmb: posEmb)
        residual = residual + conv(normConv(residual))
        residual = residual + 0.5 * ff2(normFF2(residual))
        return normOut(residual)
    }
}

/// The FastConformer encoder: subsampling, the relative positional table, the layer stack.
final class NFKParakeetEncoder: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: NFKParakeetSubsampling
    @ModuleInfo(key: "layers") var layers: [NFKParakeetLayer]
    let dModel: Int

    init(_ config: NFKMLXParakeetConfiguration) {
        dModel = config.dModel
        _preEncode.wrappedValue = NFKParakeetSubsampling(config)
        _layers.wrappedValue = (0 ..< config.layers).map { _ in NFKParakeetLayer(config) }
    }

    /// The sinusoidal table for relative offsets `T−1 … −(T−1)`, `[1, 2T−1, D]`.
    func positionEmbedding(length: Int) -> MLXArray {
        let count = 2 * length - 1
        let positions = (0 ..< count).map { Float(length - 1 - $0) }
        var table = [Float](repeating: 0, count: count * dModel)
        for (row, position) in positions.enumerated() {
            for i in stride(from: 0, to: dModel, by: 2) {
                let div = expf(Float(i) * -(logf(10000) / Float(dModel)))
                table[row * dModel + i] = sinf(position * div)
                table[row * dModel + i + 1] = cosf(position * div)
            }
        }
        return MLXArray(table).reshaped([1, count, dModel])
    }

    /// `[1, T, mels]` → `[1, T/8, D]`.
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = preEncode(features)
        let posEmb = positionEmbedding(length: x.dim(1))
        for layer in layers { x = layer(x, posEmb: posEmb) }
        return x
    }
}

// MARK: - Transducer

/// A stacked LSTM in the PyTorch `nn.LSTM(num_layers:)` shape, each layer's state threaded.
final class NFKParakeetStackedLSTM: Module {
    @ModuleInfo(key: "lstm") var lstm: [LSTM]

    init(inputSize: Int, hiddenSize: Int, layers: Int) {
        _lstm.wrappedValue = (0 ..< layers).map { LSTM(inputSize: $0 == 0 ? inputSize : hiddenSize, hiddenSize: hiddenSize) }
    }

    /// `x` `[B, 1, in]`, `state` per layer `(hidden [B, H], cell [B, H])` → output `[B, 1, H]` + new state.
    func callAsFunction(_ x: MLXArray, state: [(MLXArray, MLXArray)]?) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var h = x
        var next = [(MLXArray, MLXArray)]()
        for (index, layer) in lstm.enumerated() {
            let (allHidden, allCell) = layer(h, hidden: state?[index].0, cell: state?[index].1)
            h = allHidden
            next.append((allHidden[0..., -1, 0...], allCell[0..., -1, 0...]))
        }
        return (h, next)
    }
}

/// The prediction network: a token embedding (blank as padding) into the stacked LSTM. The blank / start
/// state is a zero vector rather than an embedding.
final class NFKParakeetPrediction: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "dec_rnn") var rnn: NFKParakeetStackedLSTM
    let hidden: Int

    init(_ config: NFKMLXParakeetConfiguration) {
        hidden = config.predictionHidden
        _embed.wrappedValue = Embedding(embeddingCount: config.vocabulary + 1, dimensions: hidden)
        _rnn.wrappedValue = NFKParakeetStackedLSTM(inputSize: hidden, hiddenSize: hidden, layers: config.predictionLayers)
    }

    /// The blank / start state feeds the LSTM a ZERO vector in place of an embedding (the reference's
    /// `blank_as_pad`); the LSTM still runs, and its output and state are what the joint and the next
    /// step read.
    func callAsFunction(token: Int?, state: [(MLXArray, MLXArray)]?) -> (MLXArray, [(MLXArray, MLXArray)]?) {
        let input = token.map { embed(MLXArray([Int32($0)]).reshaped([1, 1])) } ?? MLXArray.zeros([1, 1, hidden])
        let (g, next) = rnn(input, state: state)
        return (g, next)
    }
}

final class NFKParakeetDecoder: Module {
    @ModuleInfo(key: "prediction") var prediction: NFKParakeetPrediction
    init(_ config: NFKMLXParakeetConfiguration) { _prediction.wrappedValue = NFKParakeetPrediction(config) }
}

/// The joint: the encoder frame and the prediction output each projected to the joint width, summed,
/// ReLU, then a linear to `vocabulary + 1 (blank) + durations` logits. `joint_net` keeps the reference's
/// `nn.Sequential` indices (ReLU 0, Dropout 1, Linear 2).
final class NFKParakeetJoint: Module {
    @ModuleInfo(key: "pred") var pred: Linear
    @ModuleInfo(key: "enc") var enc: Linear
    @ModuleInfo(key: "joint_net") var jointNet: [Module]

    init(_ config: NFKMLXParakeetConfiguration) {
        _pred.wrappedValue = Linear(config.predictionHidden, config.jointHidden)
        _enc.wrappedValue = Linear(config.dModel, config.jointHidden)
        _jointNet.wrappedValue = [Module(), Module(),
                                  Linear(config.jointHidden, config.vocabulary + 1 + config.durations.count)]
    }

    /// `f` `[1, D]` one encoder frame, `g` `[1, 1, H]` → logits `[V + 1 + durations]`.
    func callAsFunction(frame f: MLXArray, prediction g: MLXArray) -> MLXArray {
        let summed = enc(f).reshaped([1, -1]) + pred(g).reshaped([1, -1])
        return (jointNet[2] as! Linear)(relu(summed))[0]
    }
}

// MARK: - Model

public final class NFKMLXParakeetNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKParakeetEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKParakeetDecoder
    @ModuleInfo(key: "joint") var joint: NFKParakeetJoint
    let frontEnd: NFKParakeetFrontEnd
    public let configuration: NFKMLXParakeetConfiguration

    public init(_ configuration: NFKMLXParakeetConfiguration) {
        self.configuration = configuration
        frontEnd = NFKParakeetFrontEnd(configuration)
        _encoder.wrappedValue = NFKParakeetEncoder(configuration)
        _decoder.wrappedValue = NFKParakeetDecoder(configuration)
        _joint.wrappedValue = NFKParakeetJoint(configuration)
    }

    /// One recognized token and the encoder frame it was emitted at.
    public struct Token { public let id: Int; public let frame: Int }

    /// `[1, frames, mels]` normalized features → encoder frames `[1, T, D]`.
    public func encode(features: MLXArray) -> MLXArray { encoder(features) }

    /// Greedy token-and-duration transducer decoding over the encoder frames `[1, T, D]`: at each frame the
    /// joint scores the next token and how many frames to skip; a non-blank token advances the prediction
    /// state; a zero duration keeps decoding the same frame (bounded by `maxSymbolsPerFrame`).
    public func decode(encoded: MLXArray) -> [Token] {
        let frames = encoded.dim(1)
        let blank = configuration.blank
        let durations = configuration.durations
        var tokens = [Token]()
        var state: [(MLXArray, MLXArray)]? = nil
        var lastToken: Int? = nil
        var time = 0
        while time < frames {
            let f = encoded[0, time, 0...]
            var symbols = 0
            var skip = 0
            var needLoop = true
            while needLoop && symbols < configuration.maxSymbolsPerFrame {
                let (g, nextState) = decoder.prediction(token: lastToken, state: state)
                let logits = joint(frame: f, prediction: g); eval(logits)
                let values = logits.asArray(Float.self)
                let tokenCount = blank + 1
                var k = 0
                for i in 1 ..< tokenCount where values[i] > values[k] { k = i }
                var d = 0
                for i in 1 ..< durations.count where values[tokenCount + i] > values[tokenCount + d] { d = i }
                skip = durations[d]
                if k != blank {
                    tokens.append(Token(id: k, frame: time))
                    state = nextState
                    lastToken = k
                }
                symbols += 1
                time += skip
                needLoop = skip == 0
            }
            if symbols == configuration.maxSymbolsPerFrame && skip == 0 { time += 1 }
        }
        return tokens
    }

    /// Waveform (16 kHz mono, `-1...1`) → the recognized tokens.
    public func recognize(_ samples: [Float]) -> [Token] {
        decode(encoded: encode(features: frontEnd.features(samples)))
    }
}

// MARK: - Tokenizer and loading

/// A SentencePiece piece table read from the release's `*_tokenizer.vocab` (one `piece<TAB>score` per
/// line, in id order). Recognition only DECODES, so the pieces alone suffice: they concatenate, the `▁`
/// word marker becomes a space, and a leading space is dropped — SentencePiece's own `decode`.
public struct NFKMLXParakeetVocabulary: Sendable {
    public let pieces: [String]

    public init(vocabURL: URL) throws {
        let text = try String(contentsOf: vocabURL, encoding: .utf8)
        pieces = text.split(separator: "\n", omittingEmptySubsequences: false).filter { !$0.isEmpty }
            .map { String($0.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)[0]) }
    }

    public func text(for ids: [Int]) -> String {
        let joined = ids.compactMap { $0 < pieces.count ? pieces[$0] : nil }.joined()
        let spaced = joined.replacingOccurrences(of: "\u{2581}", with: " ")
        return spaced.hasPrefix(" ") ? String(spaced.dropFirst()) : spaced
    }
}

public final class NFKMLXParakeet: NSObject {
    /// The registry name.
    @objc public static let modelName = "parakeet-tdt"

    /// Folds the prediction LSTM's PyTorch layers (`weight_ih_l<n>`/`hh`, two biases) into MLX's stacked
    /// `Wx`/`Wh`/`bias` under `lstm.<n>`; renames nothing else. Returns nil for a key to drop.
    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasPrefix("preprocessor.") || key.hasSuffix("num_batches_tracked") { return nil }
        return key
    }

    /// Loads the released `model_weights.ckpt` (or the `.nemo` archive holding it) into `net`, feeding the
    /// preprocessor's stored window and filterbank to the front end and transposing the convolutions to
    /// MLX's channels-last layouts (4-D `[out,in,kH,kW]` → `[out,kH,kW,in]`, 3-D `[out,in,k]` → `[out,k,in]`).
    public static func loadWeights(into net: NFKMLXParakeetNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var arrays = checkpoint.arrays
        if let window = arrays["preprocessor.featurizer.window"] { net.frontEnd.load(window: window) }
        if let filterbank = arrays["preprocessor.featurizer.fb"] { net.frontEnd.load(filterbank: filterbank) }
        // LSTM fold.
        var folded = [String: MLXArray]()
        let lstmBase = "decoder.prediction.dec_rnn.lstm."
        for (key, value) in arrays where key.hasPrefix(lstmBase) {
            let suffix = String(key.dropFirst(lstmBase.count))          // weight_ih_l0 …
            guard let l = suffix.range(of: "_l") else { continue }
            let layer = String(suffix[l.upperBound...])
            let kind = String(suffix[..<l.lowerBound])
            let target = "\(lstmBase)\(layer)."
            switch kind {
            case "weight_ih": folded[target + "Wx"] = value
            case "weight_hh": folded[target + "Wh"] = value
            case "bias_ih", "bias_hh":
                folded[target + "bias"] = folded[target + "bias"].map { $0 + value } ?? value
            default: continue
            }
            arrays[key] = nil
        }
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays {
            guard let name = remapReferenceKey(key) else { continue }
            var array = value
            if checkpoint.needsConvTranspose, name.hasSuffix(".weight") {
                if array.ndim == 4 { array = array.transposed(0, 2, 3, 1) }
                else if array.ndim == 3 { array = array.transposed(0, 2, 1) }
            }
            mapped.append((name, array))
        }
        mapped.append(contentsOf: folded.map { ($0.key, $0.value) })
        try NFKMLXWeights.apply(mapped, to: net)
        net.train(false)
    }
}

// MARK: - Backend

private final class NFKParakeetHolder: @unchecked Sendable {
    let net: NFKMLXParakeetNet
    let vocabulary: NFKMLXParakeetVocabulary?
    init(_ net: NFKMLXParakeetNet, vocabulary: NFKMLXParakeetVocabulary?) {
        self.net = net
        self.vocabulary = vocabulary
    }
}

/// A speech-to-text backend: audio under `NFKInputAudio` (an `NFKAudioAsset` WAV or raw WAV `NSData`,
/// resampled to 16 kHz) → the transcript under `NFKOutputText`, plus one `NFKAudioSegment` per token
/// under `NFKOutputSegments` — the TDT decoder emits every token at an encoder frame, and a frame is
/// eight 10 ms hops, so each token carries its 80 ms-resolution onset.
@objc(NFKMLXParakeetBackend)
public final class NFKMLXParakeetBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKParakeetHolder
    private let identifier: String

    init(net: NFKMLXParakeetNet, vocabulary: NFKMLXParakeetVocabulary?, identifier: String) {
        holder = NFKParakeetHolder(net, vocabulary: vocabulary)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result { return result }
        if let error = job.error { throw error }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = self.holder
        Task.detached(priority: .userInitiated) {
            do {
                guard let (samples, rate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
                let configuration = holder.net.configuration
                let matched = NFKMLXAudioRate.matched(samples, from: rate, to: configuration.sampleRate)
                let tokens = holder.net.recognize(matched)
                let ids = tokens.map(\.id)
                let text = holder.vocabulary?.text(for: ids) ?? ids.map(String.init).joined(separator: " ")
                let frameSeconds = Double(configuration.hopSamples * configuration.subsamplingFactor) / Double(configuration.sampleRate)
                let segments = tokens.map { token in
                    NFKAudioSegment(startSeconds: Double(token.frame) * frameSeconds,
                                    endSeconds: Double(token.frame + 1) * frameSeconds,
                                    label: holder.vocabulary?.text(for: [token.id]) ?? String(token.id), confidence: 1)
                }
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputText: text, NFKOutputSegments: segments]))
            } catch {
                job.finish(withError: error as NSError)
            }
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

extension NFKMLXParakeet {
    /// Builds the recognizer from an UNPACKED `.nemo` release directory (`tar -xf parakeet-tdt-0.6b-v2.nemo`):
    /// `model_weights.ckpt` loads through the native checkpoint reader and the `*_tokenizer.vocab` piece
    /// table decodes the ids. Blocking on the load; run off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        let net = NFKMLXParakeetNet(.tdt06B)
        try loadWeights(into: net, from: directoryURL.appendingPathComponent("model_weights.ckpt"))
        let vocabURL = try FileManager.default.contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: nil)
            .first { $0.lastPathComponent.hasSuffix("_tokenizer.vocab") }
        let vocabulary = try vocabURL.map { try NFKMLXParakeetVocabulary(vocabURL: $0) }
        return NFKMLXParakeetBackend(net: net, vocabulary: vocabulary, identifier: modelName)
    }

    /// A random-weights recognizer at the released geometry (or `configuration`), for shape checks.
    public static func backend(configuration: NFKMLXParakeetConfiguration = .tdt06B) -> any NFKInferenceBackend {
        let net = NFKMLXParakeetNet(configuration)
        net.train(false)
        return NFKMLXParakeetBackend(net: net, vocabulary: nil, identifier: modelName)
    }
}
