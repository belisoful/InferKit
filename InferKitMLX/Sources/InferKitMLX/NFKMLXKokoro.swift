//
//  NFKMLXKokoro.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXFFT
import MLXNN

// Kokoro-82M (hexgrad), a StyleTTS2 / iSTFTNet text-to-speech voice: phonemes + a style vector in, a
// 24 kHz waveform out. Ported from the released `KModel`. The pipeline is a PL-BERT (Albert) text
// encoder, a projection, a duration predictor (a DurationEncoder of LSTM + adaptive-LayerNorm blocks,
// then an LSTM and a duration head), an F0/energy predictor, a separate TextEncoder, the
// alignment-expansion by the predicted durations, and an iSTFTNet decoder with a harmonic sine source.
//
// The port takes phonemes DIRECTLY (the misaki front end is separate). A caller supplies the style
// vector from a Kokoro voicepack. The sine source's phase and additive noise are deterministic here, so
// the decoder is reproducible.

// MARK: - Configuration

public struct NFKMLXKokoroConfiguration: Sendable {
    public var hiddenDim: Int = 512
    public var styleDim: Int = 128
    public var nToken: Int = 178
    public var maxDur: Int = 50
    public var nLayer: Int = 3
    public var textEncoderKernelSize: Int = 5
    // PL-BERT (Albert).
    public var bertHidden: Int = 768
    public var bertHeads: Int = 12
    public var bertIntermediate: Int = 2048
    public var bertLayers: Int = 12
    public var bertEmbedding: Int = 128
    public var maxPositions: Int = 512
    // iSTFTNet.
    public var upsampleRates: [Int] = [10, 6]
    public var upsampleKernelSizes: [Int] = [20, 12]
    public var upsampleInitialChannel: Int = 512
    public var resblockKernelSizes: [Int] = [3, 7, 11]
    public var resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public var genISTFTNFFT: Int = 20
    public var genISTFTHopSize: Int = 5

    public init() {}
    public static let v1 = NFKMLXKokoroConfiguration()
}

// MARK: - Shared pieces

/// A weight-normalized 1-D convolution, `[N, L, Cin] → [N, L, Cout]`. The released weights are stored
/// `weight_g`/`weight_v` and fused at load (`NFKMLXMusic3.fusedWeightNorm`), so this is a plain `Conv1d`.
typealias NFKKokoroConv1d = Conv1d

/// A bidirectional LSTM, `[N, L, C] → [N, L, 2·hidden]` (forward and reversed concatenated), the
/// PyTorch `nn.LSTM(batch_first, bidirectional)` this model uses throughout.
final class NFKKokoroBiLSTM: Module {
    @ModuleInfo(key: "forward") var forward: LSTM
    @ModuleInfo(key: "reverse") var reverse: LSTM

    init(inputSize: Int, hiddenSize: Int) {
        _forward.wrappedValue = LSTM(inputSize: inputSize, hiddenSize: hiddenSize)
        _reverse.wrappedValue = LSTM(inputSize: inputSize, hiddenSize: hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let backwards = MLXArray((0 ..< x.shape[1]).reversed().map { Int32($0) })
        let f = forward(x).0
        let b = reverse(take(x, backwards, axis: 1)).0
        return concatenated([f, take(b, backwards, axis: 1)], axis: -1)
    }
}

/// AdaLayerNorm: layer-normalize (no affine) then scale and shift from the style vector — the predictor's
/// style conditioning. `x` `[N, L, C]`, `style` `[N, styleDim]`.
final class NFKKokoroAdaLayerNorm: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    let channels: Int

    init(styleDim: Int, channels: Int) {
        self.channels = channels
        _fc.wrappedValue = Linear(styleDim, channels * 2)
    }

    func callAsFunction(_ x: MLXArray, _ style: MLXArray) -> MLXArray {
        let h = fc(style)                                                  // [N, 2C]
        let gamma = h[0..., 0 ..< channels].expandedDimensions(axis: 1)    // [N, 1, C]
        let beta = h[0..., channels ..< (2 * channels)].expandedDimensions(axis: 1)
        let normalized = normalizeLastAxis(x)
        return (1 + gamma) * normalized + beta
    }
}

/// AdaIN1d: instance-normalize over time (no affine) then scale and shift from the style vector. Works in
/// `[N, L, C]` (channel is last), so the instance norm reduces over the length axis per channel.
final class NFKKokoroAdaIN1d: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    let channels: Int

    init(styleDim: Int, channels: Int) {
        self.channels = channels
        _fc.wrappedValue = Linear(styleDim, channels * 2)
    }

    func callAsFunction(_ x: MLXArray, _ style: MLXArray) -> MLXArray {
        let h = fc(style)
        let gamma = h[0..., 0 ..< channels].expandedDimensions(axis: 1)    // [N, 1, C]
        let beta = h[0..., channels ..< (2 * channels)].expandedDimensions(axis: 1)
        let mean = x.mean(axis: 1, keepDims: true)
        let variance = x.variance(axis: 1, keepDims: true)                 // population variance
        let normalized = (x - mean) * rsqrt(variance + 1e-5)
        return (1 + gamma) * normalized + beta
    }
}

/// Layer normalization over the last (channel) axis, no affine.
private func normalizeLastAxis(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let variance = x.variance(axis: -1, keepDims: true)
    return (x - mean) * rsqrt(variance + eps)
}

// MARK: - PL-BERT (Albert)

/// The Albert embeddings: word + position + token-type, then LayerNorm. Albert factorizes the embedding
/// (128-wide) from the hidden size (768).
final class NFKKokoroAlbertEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var word: Embedding
    @ModuleInfo(key: "position_embeddings") var position: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenType: Embedding
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ config: NFKMLXKokoroConfiguration) {
        _word.wrappedValue = Embedding(embeddingCount: config.nToken, dimensions: config.bertEmbedding)
        _position.wrappedValue = Embedding(embeddingCount: config.maxPositions, dimensions: config.bertEmbedding)
        _tokenType.wrappedValue = Embedding(embeddingCount: 2, dimensions: config.bertEmbedding)
        _norm.wrappedValue = LayerNorm(dimensions: config.bertEmbedding, eps: 1e-12)
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let length = ids.dim(1)
        let positions = MLXArray(Array(0 ..< length).map { Int32($0) }).reshaped([1, length])
        let types = MLXArray.zeros([1, length], dtype: .int32)
        return norm(word(ids) + position(positions) + tokenType(types))
    }
}

/// One shared Albert layer: self-attention with a post-LayerNorm, then a feed-forward with a
/// post-LayerNorm. All 12 layers of PL-BERT reuse this one layer's weights (Albert's parameter sharing).
final class NFKKokoroAlbertLayer: Module {
    @ModuleInfo(key: "attention") var attention: NFKKokoroAlbertAttention
    @ModuleInfo(key: "ffn") var ffn: Linear
    @ModuleInfo(key: "ffn_output") var ffnOutput: Linear
    @ModuleInfo(key: "full_layer_layer_norm") var fullNorm: LayerNorm

    init(_ config: NFKMLXKokoroConfiguration) {
        _attention.wrappedValue = NFKKokoroAlbertAttention(config)
        _ffn.wrappedValue = Linear(config.bertHidden, config.bertIntermediate)
        _ffnOutput.wrappedValue = Linear(config.bertIntermediate, config.bertHidden)
        _fullNorm.wrappedValue = LayerNorm(dimensions: config.bertHidden, eps: 1e-12)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attended = attention(x)
        let hidden = ffnOutput(geluApproximate(ffn(attended)))
        return fullNorm(attended + hidden)
    }
}

/// Albert self-attention: query/key/value → scaled dot-product → dense → residual LayerNorm.
final class NFKKokoroAlbertAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm
    let heads: Int
    let headDim: Int

    init(_ config: NFKMLXKokoroConfiguration) {
        heads = config.bertHeads
        headDim = config.bertHidden / config.bertHeads
        _query.wrappedValue = Linear(config.bertHidden, config.bertHidden)
        _key.wrappedValue = Linear(config.bertHidden, config.bertHidden)
        _value.wrappedValue = Linear(config.bertHidden, config.bertHidden)
        _dense.wrappedValue = Linear(config.bertHidden, config.bertHidden)
        _norm.wrappedValue = LayerNorm(dimensions: config.bertHidden, eps: 1e-12)
    }

    private func split(_ t: MLXArray) -> MLXArray {
        let (b, n) = (t.dim(0), t.dim(1))
        return t.reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(query(x)), keys: split(key(x)), values: split(value(x)),
            scale: 1.0 / sqrt(Float(headDim)), mask: .none)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([b, n, heads * headDim])
        return norm(x + dense(merged))
    }
}

/// PL-BERT: Albert embeddings, an embedding→hidden projection, then the shared layer applied `bertLayers`
/// times. Returns the last hidden state `[N, T, 768]`.
final class NFKKokoroAlbert: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKKokoroAlbertEmbeddings
    @ModuleInfo(key: "embedding_hidden_mapping_in") var mapping: Linear
    @ModuleInfo(key: "layer") var layer: NFKKokoroAlbertLayer
    let layerCount: Int

    init(_ config: NFKMLXKokoroConfiguration) {
        layerCount = config.bertLayers
        _embeddings.wrappedValue = NFKKokoroAlbertEmbeddings(config)
        _mapping.wrappedValue = Linear(config.bertEmbedding, config.bertHidden)
        _layer.wrappedValue = NFKKokoroAlbertLayer(config)
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var hidden = mapping(embeddings(ids))
        for _ in 0 ..< layerCount { hidden = layer(hidden) }
        return hidden
    }
}

// MARK: - Text encoder

/// A channel LayerNorm with affine gamma/beta (the reference's custom `LayerNorm`), over the last axis.
final class NFKKokoroChannelLN: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(_ channels: Int) {
        _gamma.wrappedValue = MLXArray.ones([channels])
        _beta.wrappedValue = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { normalizeLastAxis(x) * gamma + beta }
}

/// One TextEncoder CNN block: a weight-normalized convolution, a channel LayerNorm, and a leaky ReLU.
final class NFKKokoroTextConvBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "norm") var norm: NFKKokoroChannelLN

    init(_ channels: Int, kernel: Int) {
        _conv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: channels,
                                    kernelSize: kernel, padding: (kernel - 1) / 2)
        _norm.wrappedValue = NFKKokoroChannelLN(channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { leakyRelu(norm(conv(x)), negativeSlope: 0.2) }
}

/// The TextEncoder: phoneme embedding → CNN stack → bidirectional LSTM. Returns `[N, C, T]` (NCL) to
/// match the reference's alignment matmul. Works internally in `[N, T, C]`.
final class NFKKokoroTextEncoder: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "cnn") var cnn: [NFKKokoroTextConvBlock]
    @ModuleInfo(key: "lstm") var lstm: NFKKokoroBiLSTM

    init(_ config: NFKMLXKokoroConfiguration) {
        _embedding.wrappedValue = Embedding(embeddingCount: config.nToken, dimensions: config.hiddenDim)
        _cnn.wrappedValue = (0 ..< config.nLayer).map { _ in
            NFKKokoroTextConvBlock(config.hiddenDim, kernel: config.textEncoderKernelSize)
        }
        _lstm.wrappedValue = NFKKokoroBiLSTM(inputSize: config.hiddenDim, hiddenSize: config.hiddenDim / 2)
    }

    /// `ids` `[N, T]` → `[N, C, T]`.
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var x = embedding(ids)                                             // [N, T, C]
        for block in cnn { x = block(x) }
        x = lstm(x)                                                        // [N, T, C]
        return x.transposed(0, 2, 1)                                      // [N, C, T]
    }
}

// MARK: - AdainResBlk1d (predictor F0/N + decoder)

/// A residual block with AdaIN conditioning, a leaky-ReLU activation, and an optional ×2 upsample. Works
/// in `[N, C, T]` (NCL, the reference's layout) — the caller transposes at the boundary. Internally each
/// conv runs in `[N, T, C]`.
final class NFKKokoroAdainResBlk1d: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "norm1") var norm1: NFKKokoroAdaIN1d
    @ModuleInfo(key: "norm2") var norm2: NFKKokoroAdaIN1d
    @ModuleInfo(key: "conv1x1") var conv1x1: Conv1d?
    @ModuleInfo(key: "pool") var pool: ConvTransposed1d?                  // depthwise stride-2 upsample
    let upsample: Bool
    let learnedShortcut: Bool

    init(dimIn: Int, dimOut: Int, styleDim: Int, upsample: Bool) {
        self.upsample = upsample
        learnedShortcut = dimIn != dimOut
        _conv1.wrappedValue = Conv1d(inputChannels: dimIn, outputChannels: dimOut, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: dimOut, outputChannels: dimOut, kernelSize: 3, padding: 1)
        _norm1.wrappedValue = NFKKokoroAdaIN1d(styleDim: styleDim, channels: dimIn)
        _norm2.wrappedValue = NFKKokoroAdaIN1d(styleDim: styleDim, channels: dimOut)
        if learnedShortcut {
            _conv1x1.wrappedValue = Conv1d(inputChannels: dimIn, outputChannels: dimOut, kernelSize: 1, bias: false)
        }
        if upsample {
            _pool.wrappedValue = ConvTransposed1d(inputChannels: dimIn, outputChannels: dimIn, kernelSize: 3,
                                                  stride: 2, padding: 1, outputPadding: 1, groups: dimIn)
        }
    }

    /// Nearest ×2 upsample of `[N, T, C]` along T (the reference's `UpSample1d` for the shortcut).
    private func upsampleNearest(_ x: MLXArray) -> MLXArray {
        let (n, t, c) = (x.dim(0), x.dim(1), x.dim(2))
        return broadcast(x.reshaped([n, t, 1, c]), to: [n, t, 2, c]).reshaped([n, t * 2, c])
    }

    /// `x` `[N, C, T]`, `style` `[N, styleDim]` → `[N, Cout, T*(upsample ? 2 : 1)]`.
    func callAsFunction(_ x: MLXArray, _ style: MLXArray) -> MLXArray {
        // Work in NLC internally; the input and output are NCL to match the reference's seams.
        let input = x.transposed(0, 2, 1)
        let residual = self.residual(input, style)
        let shortcut = self.shortcut(input)
        return ((residual + shortcut) / sqrt(2.0)).transposed(0, 2, 1)
    }

    private func residual(_ input: MLXArray, _ style: MLXArray) -> MLXArray {
        var x = norm1(input, style)
        x = leakyRelu(x, negativeSlope: 0.2)
        if let pool { x = pool(x) }
        x = conv1(x)
        x = norm2(x, style)
        x = leakyRelu(x, negativeSlope: 0.2)
        return conv2(x)
    }

    private func shortcut(_ input: MLXArray) -> MLXArray {
        var x = upsample ? upsampleNearest(input) : input
        if let conv1x1 { x = conv1x1(x) }
        return x
    }
}

// MARK: - Prosody predictor

/// The DurationEncoder: `nlayers` of (bidirectional LSTM + AdaLayerNorm), re-concatenating the style
/// vector after each AdaLayerNorm. Input `d_en` `[N, C, T]` (NCL) + `style` `[N, styleDim]`, output
/// `[N, T, C + styleDim]`.
final class NFKKokoroDurationEncoder: Module {
    @ModuleInfo(key: "lstms") var lstms: [NFKKokoroBiLSTM]
    @ModuleInfo(key: "norms") var norms: [NFKKokoroAdaLayerNorm]
    let styleDim: Int

    init(_ config: NFKMLXKokoroConfiguration) {
        styleDim = config.styleDim
        let inputSize = config.hiddenDim + config.styleDim
        _lstms.wrappedValue = (0 ..< config.nLayer).map { _ in
            NFKKokoroBiLSTM(inputSize: inputSize, hiddenSize: config.hiddenDim / 2)
        }
        _norms.wrappedValue = (0 ..< config.nLayer).map { _ in
            NFKKokoroAdaLayerNorm(styleDim: config.styleDim, channels: config.hiddenDim)
        }
    }

    func callAsFunction(_ dEn: MLXArray, _ style: MLXArray) -> MLXArray {
        let t = dEn.dim(2)
        let styleChannel = broadcast(style.expandedDimensions(axis: 2), to: [dEn.dim(0), styleDim, t])  // [N, sty, T]
        var x = concatenated([dEn, styleChannel], axis: 1)                // [N, C+sty, T]
        for index in 0 ..< lstms.count {
            let lstmOut = lstms[index](x.transposed(0, 2, 1))             // [N, T, hidden]
            let normed = norms[index](lstmOut, style)                     // AdaLN over channels [N, T, hidden]
            x = concatenated([normed.transposed(0, 2, 1), styleChannel], axis: 1)   // [N, hidden+sty, T]
        }
        return x.transposed(0, 2, 1)                                      // [N, T, C+sty]
    }
}

/// The prosody predictor: the DurationEncoder + an LSTM and a duration head (predicting per-phoneme
/// durations), and `F0Ntrain` predicting the per-frame F0 and energy from the alignment-expanded encoding.
final class NFKKokoroProsodyPredictor: Module {
    @ModuleInfo(key: "text_encoder") var textEncoder: NFKKokoroDurationEncoder
    @ModuleInfo(key: "lstm") var lstm: NFKKokoroBiLSTM
    @ModuleInfo(key: "duration_proj") var durationProj: Linear
    @ModuleInfo(key: "shared") var shared: NFKKokoroBiLSTM
    @ModuleInfo(key: "F0") var f0Blocks: [NFKKokoroAdainResBlk1d]
    @ModuleInfo(key: "N") var nBlocks: [NFKKokoroAdainResBlk1d]
    @ModuleInfo(key: "F0_proj") var f0Proj: Conv1d
    @ModuleInfo(key: "N_proj") var nProj: Conv1d

    init(_ config: NFKMLXKokoroConfiguration) {
        let hidden = config.hiddenDim, style = config.styleDim
        _textEncoder.wrappedValue = NFKKokoroDurationEncoder(config)
        _lstm.wrappedValue = NFKKokoroBiLSTM(inputSize: hidden + style, hiddenSize: hidden / 2)
        _durationProj.wrappedValue = Linear(hidden, config.maxDur)
        _shared.wrappedValue = NFKKokoroBiLSTM(inputSize: hidden + style, hiddenSize: hidden / 2)
        _f0Blocks.wrappedValue = [
            NFKKokoroAdainResBlk1d(dimIn: hidden, dimOut: hidden, styleDim: style, upsample: false),
            NFKKokoroAdainResBlk1d(dimIn: hidden, dimOut: hidden / 2, styleDim: style, upsample: true),
            NFKKokoroAdainResBlk1d(dimIn: hidden / 2, dimOut: hidden / 2, styleDim: style, upsample: false),
        ]
        _nBlocks.wrappedValue = [
            NFKKokoroAdainResBlk1d(dimIn: hidden, dimOut: hidden, styleDim: style, upsample: false),
            NFKKokoroAdainResBlk1d(dimIn: hidden, dimOut: hidden / 2, styleDim: style, upsample: true),
            NFKKokoroAdainResBlk1d(dimIn: hidden / 2, dimOut: hidden / 2, styleDim: style, upsample: false),
        ]
        _f0Proj.wrappedValue = Conv1d(inputChannels: hidden / 2, outputChannels: 1, kernelSize: 1)
        _nProj.wrappedValue = Conv1d(inputChannels: hidden / 2, outputChannels: 1, kernelSize: 1)
    }

    /// `d_en` `[N, C, T]`, `style` `[N, styleDim]` → the DurationEncoder output `d` `[N, T, C+sty]` and the
    /// per-phoneme durations `[N, T]` (float, pre-round).
    func durations(_ dEn: MLXArray, _ style: MLXArray) -> (d: MLXArray, duration: MLXArray) {
        let d = textEncoder(dEn, style)                                   // [N, T, 640]
        let x = lstm(d)                                                   // [N, T, 512]
        let duration = sigmoid(durationProj(x)).sum(axis: -1)            // [N, T]
        return (d, duration)
    }

    /// `en` `[N, C, frames]` (the alignment-expanded encoding), `style` `[N, styleDim]` → F0 and energy,
    /// each `[N, frames·2]` (the F0/N blocks upsample once).
    func f0n(_ en: MLXArray, _ style: MLXArray) -> (f0: MLXArray, n: MLXArray) {
        let x = shared(en.transposed(0, 2, 1)).transposed(0, 2, 1)       // [N, 512, frames]
        var f0 = x
        for block in f0Blocks { f0 = block(f0, style) }
        f0 = f0Proj(f0.transposed(0, 2, 1)).transposed(0, 2, 1)          // [N, 1, frames·2]
        var n = x
        for block in nBlocks { n = block(n, style) }
        n = nProj(n.transposed(0, 2, 1)).transposed(0, 2, 1)
        return (f0[0..., 0, 0...], n[0..., 0, 0...])                      // [N, frames·2]
    }
}

/// Builds the alignment matrix `[T, frames]` from integer durations: each phoneme `i` maps to a run of
/// `dur[i]` consecutive frames (a one-hot expansion).
func kokoroAlignment(durations: [Int]) -> MLXArray {
    let frames = durations.reduce(0, +)
    var rows = [Int32]()
    var columns = [Int32]()
    var frame = 0
    for (phoneme, count) in durations.enumerated() {
        for _ in 0 ..< count {
            rows.append(Int32(phoneme))
            columns.append(Int32(frame))
            frame += 1
        }
    }
    let matrix = MLXArray.zeros([durations.count, frames])
    matrix[MLXArray(rows), MLXArray(columns)] = MLXArray.ones([frames])
    return matrix
}

// MARK: - Top-level model (text path; the decoder is added below)

public final class NFKMLXKokoroNet: Module {
    @ModuleInfo(key: "bert") var bert: NFKKokoroAlbert
    @ModuleInfo(key: "bert_encoder") var bertEncoder: Linear
    @ModuleInfo(key: "predictor") var predictor: NFKKokoroProsodyPredictor
    @ModuleInfo(key: "text_encoder") var textEncoder: NFKKokoroTextEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKKokoroDecoder
    public let config: NFKMLXKokoroConfiguration

    public init(_ config: NFKMLXKokoroConfiguration) {
        self.config = config
        _bert.wrappedValue = NFKKokoroAlbert(config)
        _bertEncoder.wrappedValue = Linear(config.bertHidden, config.hiddenDim)
        _predictor.wrappedValue = NFKKokoroProsodyPredictor(config)
        _textEncoder.wrappedValue = NFKKokoroTextEncoder(config)
        _decoder.wrappedValue = NFKKokoroDecoder(config)
    }

    /// Synthesizes the waveform for `ids` `[1, T]` (with BOS/EOS) and the voice `refS` `[1, 256]`. When
    /// `durationsOverride` is nil the predictor's own rounded durations set the alignment.
    public func synthesize(ids: MLXArray, refS: MLXArray, durationsOverride: [Int]? = nil) -> MLXArray {
        let predictorStyle = refS[0..., 128 ..< 256]
        let dEn = bertEncoder(bert(ids)).transposed(0, 2, 1)
        let (d, duration) = predictor.durations(dEn, predictorStyle)
        let durations: [Int]
        if let durationsOverride {
            durations = durationsOverride
        } else {
            eval(duration)
            durations = duration[0].asArray(Float.self).map { max(1, Int($0.rounded())) }
        }
        let alignment = kokoroAlignment(durations: durations).expandedDimensions(axis: 0)
        let en = matmul(d.transposed(0, 2, 1), alignment)
        let (f0, n) = predictor.f0n(en, predictorStyle)
        let asr = matmul(textEncoder(ids), alignment)
        return decoder(asr, f0Curve: f0, n: n, style: refS[0..., 0 ..< 128])[0]
    }

    /// The intermediate tensors of one synthesis, for parity localization.
    public struct Seams {
        public let bert: MLXArray          // [1, T, 768]
        public let dEn: MLXArray           // [1, 512, T]
        public let d: MLXArray             // [1, T, 640]
        public let duration: MLXArray      // [1, T]
        public let f0: MLXArray            // [1, frames·2]
        public let n: MLXArray             // [1, frames·2]
        public let tEn: MLXArray           // [1, 512, T]
        public let asr: MLXArray           // [1, 512, frames]
    }

    /// Runs the text path. `ids` `[1, T]` (with BOS/EOS), `refS` `[1, 256]` the voice style. `durationsOverride`
    /// supplies the integer alignment (the reference's), so F0/N/asr are independent of duration rounding.
    public func textPath(ids: MLXArray, refS: MLXArray, durationsOverride: [Int]) -> Seams {
        let bertOut = bert(ids)                                            // [1, T, 768]
        let dEn = bertEncoder(bertOut).transposed(0, 2, 1)               // [1, 512, T]
        let predictorStyle = refS[0..., 128 ..< 256]                     // [1, 128]
        let (d, duration) = predictor.durations(dEn, predictorStyle)     // d [1,T,640], duration [1,T]
        let alignment = kokoroAlignment(durations: durationsOverride).expandedDimensions(axis: 0)  // [1, T, frames]
        let en = matmul(d.transposed(0, 2, 1), alignment)                // [1, 640, frames]
        let (f0, n) = predictor.f0n(en, predictorStyle)                  // [1, frames·2]
        let tEn = textEncoder(ids)                                       // [1, 512, T]
        let asr = matmul(tEn, alignment)                                 // [1, 512, frames]
        return Seams(bert: bertOut, dEn: dEn, d: d, duration: duration, f0: f0, n: n, tEn: tEn, asr: asr)
    }
}

// MARK: - Weight loading

public final class NFKMLXKokoro: NSObject {
    /// Fuses a `weight_g`/`weight_v` pair into `weight` (`g · v / ‖v‖`, the norm over every axis but the
    /// first — PyTorch's `weight_norm` default dim 0).
    static func fuseWeightNorm(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in arrays {
            if key.hasSuffix(".weight_g") { continue }
            guard key.hasSuffix(".weight_v") else { result[key] = value; continue }
            let base = String(key.dropLast(".weight_v".count))
            guard let gain = arrays[base + ".weight_g"] else { result[key] = value; continue }
            let axes = Array(1 ..< value.ndim)
            let norm = sqrt((value * value).sum(axes: axes, keepDims: true))
            result[base + ".weight"] = gain * value / norm
        }
        return result
    }

    /// Folds a PyTorch LSTM's separate input/hidden weights and biases into MLX's `Wx`/`Wh`/`bias`, under a
    /// `forward`/`reverse` submodule. The gate order (i, f, g, o) matches, so the matrices transfer as is.
    private static func foldLSTM(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var result = arrays
        let suffixes = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
                        "weight_ih_l0_reverse", "weight_hh_l0_reverse", "bias_ih_l0_reverse", "bias_hh_l0_reverse"]
        var bases = Set<String>()
        for key in arrays.keys where key.hasSuffix(".weight_ih_l0") {
            bases.insert(String(key.dropLast(".weight_ih_l0".count)))
        }
        for base in bases {
            for suffix in suffixes { result["\(base).\(suffix)"] = nil }
            for (direction, tag) in [("forward", ""), ("reverse", "_reverse")] {
                if let wx = arrays["\(base).weight_ih_l0\(tag)"] { result["\(base).\(direction).Wx"] = wx }
                if let wh = arrays["\(base).weight_hh_l0\(tag)"] { result["\(base).\(direction).Wh"] = wh }
                let biasIH = arrays["\(base).bias_ih_l0\(tag)"]
                let biasHH = arrays["\(base).bias_hh_l0\(tag)"]
                if let biasIH, let biasHH { result["\(base).\(direction).bias"] = biasIH + biasHH }
            }
        }
        return result
    }

    /// Renames a checkpoint key onto the module's parameter path. Returns nil for a key to drop (the
    /// pooler, or the decoder, which the text-path net does not carry yet).
    static func remapKey(_ key: String) -> String? {
        // The released checkpoint stores each top-level module's state_dict under a `module.`
        // DataParallel prefix (`bert.module.embeddings...`); strip it as KModel does.
        var name = key.replacingOccurrences(of: ".module.", with: ".")
        if name.hasPrefix("bert.pooler.") { return nil }
        name = name.replacingOccurrences(of: "bert.encoder.embedding_hidden_mapping_in.", with: "bert.embedding_hidden_mapping_in.")
        name = name.replacingOccurrences(of: "bert.encoder.albert_layer_groups.0.albert_layers.0.", with: "bert.layer.")
        // Duration encoder: interleaved lstms.{even}=LSTM, lstms.{odd}=AdaLayerNorm.
        if let range = name.range(of: #"predictor\.text_encoder\.lstms\.(\d+)"#, options: .regularExpression) {
            let matched = String(name[range])
            let index = Int(matched.split(separator: ".").last!)!
            let replacement = index % 2 == 0
                ? "predictor.text_encoder.lstms.\(index / 2)"
                : "predictor.text_encoder.norms.\(index / 2)"
            name = name.replacingCharacters(in: range, with: replacement)
        }
        name = name.replacingOccurrences(of: "predictor.duration_proj.linear_layer.", with: "predictor.duration_proj.")
        // TextEncoder CNN Sequential indices: 0 = conv, 1 = channel LayerNorm.
        if let range = name.range(of: #"text_encoder\.cnn\.(\d+)\.0\."#, options: .regularExpression) {
            let block = name[range].split(separator: ".")[2]
            name = name.replacingCharacters(in: range, with: "text_encoder.cnn.\(block).conv.")
        }
        if let range = name.range(of: #"text_encoder\.cnn\.(\d+)\.1\."#, options: .regularExpression) {
            let block = name[range].split(separator: ".")[2]
            name = name.replacingCharacters(in: range, with: "text_encoder.cnn.\(block).norm.")
        }
        return name
    }

    /// Stacks a generator resblock's `alpha1`/`alpha2` `ParameterList` entries (`…alpha1.0/1/2`, each
    /// `[1, C, 1]`) into one `[3, 1, 1, C]` parameter matching the module, transposing to channels-last.
    private static func stackAlphas(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var result = arrays
        var bases = Set<String>()
        for key in arrays.keys {
            if let range = key.range(of: #"\.alpha[12]\.\d+$"#, options: .regularExpression) {
                bases.insert(String(key[..<range.lowerBound]) + "." + key[key.index(range.lowerBound, offsetBy: 1)...].prefix(6))
            }
        }
        for base in bases {
            var entries = [(Int, MLXArray)]()
            for (key, value) in arrays where key.hasPrefix(base + ".") {
                if let index = Int(key.dropFirst(base.count + 1)) { entries.append((index, value)) }
            }
            guard !entries.isEmpty else { continue }
            entries.sort { $0.0 < $1.0 }
            let stacked = MLX.stacked(entries.map { $0.1.transposed(0, 2, 1) }, axis: 0)   // [n, 1, 1, C]
            for (key, _) in arrays where key.hasPrefix(base + ".") { result[key] = nil }
            result[base] = stacked
        }
        return result
    }

    /// Loads the full model weights (text path + iSTFTNet decoder) into `net`.
    public static func loadWeights(into net: NFKMLXKokoroNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let fused = stackAlphas(foldLSTM(fuseWeightNorm(checkpoint.arrays)))
        var mapped = [(String, MLXArray)]()
        for (key, value) in fused {
            guard let name = remapKey(key) else { continue }
            let transposed: MLXArray
            if name.hasSuffix(".weight"), value.ndim == 3 {
                // The generator's upsampling transposed convolutions store [in, out, k]; MLX wants
                // [out, k, in]. Every other 1-D convolution (forward and the depthwise pool) is
                // [out, in, k] → [out, k, in].
                transposed = name.contains(".generator.ups.") ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            } else {
                transposed = value
            }
            mapped.append((name, transposed))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Loads only the text-path weights (the decoder keys land as ignored extras) — for the text-path seam
    /// test on a net that still carries the decoder.
    public static func loadTextWeights(into net: NFKMLXKokoroNet, from url: URL) throws {
        try loadWeights(into: net, from: url)
    }
}

// MARK: - iSTFTNet decoder

/// The Snake activation used by the generator's residual blocks: `x + (1/a)·sin(a·x)²`, `a` per channel.
private func kokoroSnake(_ x: MLXArray, _ alpha: MLXArray) -> MLXArray {
    x + (1.0 / alpha) * sin(alpha * x).square()
}

/// PyTorch `F.interpolate(mode: "linear", align_corners: false)` over `[N, C, L]`, resampling the last
/// axis to `outSize`.
private func kokoroLinearResample(_ x: MLXArray, outSize: Int) -> MLXArray {
    let l = x.dim(2)
    if l == outSize { return x }
    let scale = Float(l) / Float(outSize)
    var lo = [Int32](), hi = [Int32](), frac = [Float]()
    for o in 0 ..< outSize {
        let src = min(max((Float(o) + 0.5) * scale - 0.5, 0), Float(l - 1))
        let low = Int(src.rounded(.down))
        lo.append(Int32(low)); hi.append(Int32(min(low + 1, l - 1))); frac.append(src - Float(low))
    }
    let low = take(x, MLXArray(lo), axis: 2)
    let high = take(x, MLXArray(hi), axis: 2)
    let weight = MLXArray(frac).reshaped([1, 1, outSize])
    return low * (1 - weight) + high * weight
}

/// A periodic Hann window of `length` (`get_window("hann", length, fftbins=true)`).
private func kokoroHann(_ length: Int) -> MLXArray {
    var values = [Float](repeating: 0, count: length)
    for i in 0 ..< length { values[i] = 0.5 - 0.5 * cos(2 * Float.pi * Float(i) / Float(length)) }
    return MLXArray(values)
}

/// The short-time Fourier transform pair the generator's source uses (`torch.stft`/`torch.istft`,
/// `center: true`, reflect padding, a Hann window).
struct NFKKokoroSTFT {
    let nFFT: Int
    let hop: Int
    let window: MLXArray

    init(nFFT: Int, hop: Int) {
        self.nFFT = nFFT
        self.hop = hop
        window = kokoroHann(nFFT)
    }

    /// Reflect-pads `[1, L]` by `pad` on each side (PyTorch `pad_mode="reflect"`: the edge sample is not
    /// repeated).
    private func reflectPad(_ signal: MLXArray, pad: Int) -> MLXArray {
        let l = signal.dim(1)
        var indices = [Int32]()
        for i in stride(from: pad, through: 1, by: -1) { indices.append(Int32(i)) }
        indices.append(contentsOf: (0 ..< l).map { Int32($0) })
        for i in stride(from: l - 2, through: l - 1 - pad, by: -1) { indices.append(Int32(i)) }
        return take(signal, MLXArray(indices), axis: 1)
    }

    /// `[1, L]` → magnitude and phase, each `[1, nFFT/2 + 1, frames]`.
    func transform(_ signal: MLXArray) -> (magnitude: MLXArray, phase: MLXArray) {
        let pad = nFFT / 2
        let padded = reflectPad(signal, pad: pad)                          // [1, L + nFFT]
        let length = padded.dim(1)
        let frames = 1 + (length - nFFT) / hop
        var gather = [Int32]()
        for f in 0 ..< frames {
            for k in 0 ..< nFFT { gather.append(Int32(f * hop + k)) }
        }
        let framed = take(padded, MLXArray(gather), axis: 1).reshaped([1, frames, nFFT]) * window.reshaped([1, 1, nFFT])
        let spectrum = MLXFFT.rfft(framed, axis: 2)                        // [1, frames, bins] complex
        let magnitude = spectrum.abs().transposed(0, 2, 1)                // [1, bins, frames]
        let phase = atan2(spectrum.imaginaryPart(), spectrum.realPart()).transposed(0, 2, 1)
        return (magnitude, phase)
    }

    /// magnitude and phase `[1, bins, frames]` → `[1, samples]`, overlap-add with window normalization,
    /// center padding removed.
    func inverse(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        let frames = magnitude.dim(2)
        let real = (magnitude * cos(phase)).transposed(0, 2, 1)           // [1, frames, bins]
        let imaginary = (magnitude * sin(phase)).transposed(0, 2, 1)
        let complex = real.asType(.complex64) + imaginary.asType(.complex64) * MLXArray(real: 0, imaginary: 1)
        let time = MLXFFT.irfft(complex, n: nFFT, axis: 2)                // [1, frames, nFFT] real
        let windowed = time * window.reshaped([1, 1, nFFT])
        // Overlap-add over Swift buffers (MLX has no scatter-add), plus the window-squared normalization.
        let outLength = (frames - 1) * hop + nFFT
        let framesValues = windowed[0].asArray(Float.self)                // frames * nFFT
        let windowValues = (window * window).asArray(Float.self)          // nFFT
        var output = [Float](repeating: 0, count: outLength)
        var normalization = [Float](repeating: 0, count: outLength)
        for f in 0 ..< frames {
            let start = f * hop
            for k in 0 ..< nFFT {
                output[start + k] += framesValues[f * nFFT + k]
                normalization[start + k] += windowValues[k]
            }
        }
        let pad = nFFT / 2
        var result = [Float](repeating: 0, count: outLength - 2 * pad)
        for i in 0 ..< result.count {
            let norm = normalization[i + pad]
            result[i] = norm > 1e-11 ? output[i + pad] / norm : 0
        }
        return MLXArray(result).reshaped([1, result.count])
    }
}

/// The harmonic sine source (`SourceModuleHnNSF`): a deterministic harmonic excitation from F0, merged by
/// a learned linear + tanh. The phase and additive noise are omitted (deterministic), matching the
/// oracle's deterministic run.
final class NFKKokoroSourceModule: Module {
    @ModuleInfo(key: "l_linear") var linear: Linear
    let harmonics: Int
    let samplingRate: Float
    let upsampleScale: Int
    let sineAmp: Float = 0.1

    init(harmonics: Int, samplingRate: Float, upsampleScale: Int) {
        self.harmonics = harmonics
        self.samplingRate = samplingRate
        self.upsampleScale = upsampleScale
        _linear.wrappedValue = Linear(harmonics + 1, 1)
    }

    /// `f0` `[1, L, 1]` (already upsampled) → the merged source `[1, L, 1]`.
    func callAsFunction(_ f0: MLXArray) -> MLXArray {
        let l = f0.dim(1)
        let multipliers = MLXArray((1 ... (harmonics + 1)).map { Float($0) }).reshaped([1, 1, harmonics + 1])
        let fn = f0 * multipliers                                          // [1, L, dim]
        var rad = fn / samplingRate
        rad = rad - floor(rad)                                             // mod 1
        // Downsample the phase increments, cumulative-sum, then upsample the phase (the reference trick).
        let down = kokoroLinearResample(rad.transposed(0, 2, 1), outSize: l / upsampleScale)   // [1, dim, L/scale]
        let phaseLow = cumsum(down.transposed(0, 2, 1), axis: 1) * (2 * Float.pi)              // [1, L/scale, dim]
        let phase = kokoroLinearResample(phaseLow.transposed(0, 2, 1) * Float(upsampleScale), outSize: l)
        let sines = sin(phase.transposed(0, 2, 1)) * sineAmp              // [1, L, dim]
        let uv = (f0 .> 10).asType(f0.dtype)                             // voiced where f0 > 10 (the reference's threshold)
        let sineWaves = sines * uv                                        // deterministic (no noise)
        return tanh(linear(sineWaves))                                    // [1, L, 1]
    }
}

/// A generator residual block (`AdaINResBlock1`): three dilated conv pairs, each with AdaIN + Snake.
final class NFKKokoroGenResBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "adain1") var adain1: [NFKKokoroAdaIN1d]
    @ModuleInfo(key: "adain2") var adain2: [NFKKokoroAdaIN1d]
    @ParameterInfo(key: "alpha1") var alpha1: MLXArray                    // [3, 1, 1, C]
    @ParameterInfo(key: "alpha2") var alpha2: MLXArray

    init(channels: Int, kernel: Int, dilations: [Int], styleDim: Int) {
        _convs1.wrappedValue = dilations.map {
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel,
                   padding: ($0 * (kernel - 1)) / 2, dilation: $0)
        }
        _convs2.wrappedValue = dilations.map { _ in
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel - 1) / 2)
        }
        _adain1.wrappedValue = dilations.map { _ in NFKKokoroAdaIN1d(styleDim: styleDim, channels: channels) }
        _adain2.wrappedValue = dilations.map { _ in NFKKokoroAdaIN1d(styleDim: styleDim, channels: channels) }
        _alpha1.wrappedValue = MLXArray.ones([dilations.count, 1, 1, channels])
        _alpha2.wrappedValue = MLXArray.ones([dilations.count, 1, 1, channels])
    }

    /// `x` `[N, C, T]`, `style` `[N, styleDim]` → `[N, C, T]`.
    func callAsFunction(_ x: MLXArray, _ style: MLXArray) -> MLXArray {
        var result = x
        for j in 0 ..< convs1.count {
            var xt = adain1[j](result.transposed(0, 2, 1), style)         // NLC
            xt = kokoroSnake(xt, alpha1[j])
            xt = convs1[j](xt)
            xt = adain2[j](xt, style)
            xt = kokoroSnake(xt, alpha2[j])
            xt = convs2[j](xt).transposed(0, 2, 1)                        // back to NCL
            result = result + xt
        }
        return result
    }
}

/// The iSTFTNet generator: a harmonic sine source, upsampling transposed convolutions with residual
/// blocks, noise-branch convolutions, and an inverse STFT head.
final class NFKKokoroGenerator: Module {
    @ModuleInfo(key: "m_source") var source: NFKKokoroSourceModule
    @ModuleInfo(key: "noise_convs") var noiseConvs: [Conv1d]
    @ModuleInfo(key: "noise_res") var noiseRes: [NFKKokoroGenResBlock]
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "resblocks") var resblocks: [NFKKokoroGenResBlock]
    @ModuleInfo(key: "conv_post") var convPost: Conv1d
    let numKernels: Int
    let numUpsamples: Int
    let stft: NFKKokoroSTFT
    let upsampleScale: Int

    init(_ config: NFKMLXKokoroConfiguration) {
        let style = config.styleDim
        numKernels = config.resblockKernelSizes.count
        numUpsamples = config.upsampleRates.count
        upsampleScale = config.upsampleRates.reduce(1, *) * config.genISTFTHopSize
        stft = NFKKokoroSTFT(nFFT: config.genISTFTNFFT, hop: config.genISTFTHopSize)
        _source.wrappedValue = NFKKokoroSourceModule(harmonics: 8, samplingRate: 24000, upsampleScale: upsampleScale)

        var upConvs = [ConvTransposed1d]()
        var noise = [Conv1d]()
        var noiseResblocks = [NFKKokoroGenResBlock]()
        var blocks = [NFKKokoroGenResBlock]()
        for i in 0 ..< numUpsamples {
            let inChannels = config.upsampleInitialChannel / (1 << i)
            let outChannels = config.upsampleInitialChannel / (1 << (i + 1))
            upConvs.append(ConvTransposed1d(inputChannels: inChannels, outputChannels: outChannels,
                                            kernelSize: config.upsampleKernelSizes[i], stride: config.upsampleRates[i],
                                            padding: (config.upsampleKernelSizes[i] - config.upsampleRates[i]) / 2))
            for j in 0 ..< numKernels {
                blocks.append(NFKKokoroGenResBlock(channels: outChannels, kernel: config.resblockKernelSizes[j],
                                                   dilations: config.resblockDilationSizes[j], styleDim: style))
            }
            if i + 1 < numUpsamples {
                let strideF0 = config.upsampleRates[(i + 1)...].reduce(1, *)
                noise.append(Conv1d(inputChannels: config.genISTFTNFFT + 2, outputChannels: outChannels,
                                    kernelSize: strideF0 * 2, stride: strideF0, padding: (strideF0 + 1) / 2))
                noiseResblocks.append(NFKKokoroGenResBlock(channels: outChannels, kernel: 7, dilations: [1, 3, 5], styleDim: style))
            } else {
                noise.append(Conv1d(inputChannels: config.genISTFTNFFT + 2, outputChannels: outChannels, kernelSize: 1))
                noiseResblocks.append(NFKKokoroGenResBlock(channels: outChannels, kernel: 11, dilations: [1, 3, 5], styleDim: style))
            }
        }
        _ups.wrappedValue = upConvs
        _noiseConvs.wrappedValue = noise
        _noiseRes.wrappedValue = noiseResblocks
        _resblocks.wrappedValue = blocks
        _convPost.wrappedValue = Conv1d(inputChannels: config.upsampleInitialChannel / (1 << numUpsamples),
                                        outputChannels: config.genISTFTNFFT + 2, kernelSize: 7, padding: 3)
    }

    /// `x` `[1, C, T]` (NCL), `style` `[1, styleDim]`, `f0` `[1, frames·2]` → the waveform `[1, samples]`.
    func callAsFunction(_ input: MLXArray, _ style: MLXArray, f0: MLXArray) -> MLXArray {
        callWithConvPost(input, style, f0: f0).audio
    }

    /// The harmonic sine excitation `[1, L, 1]` for `f0` `[1, frames·2]` — a parity seam.
    public func sineSource(f0: MLXArray) -> MLXArray {
        let frames = f0.dim(1)
        let upsampled = broadcast(f0.reshaped([1, frames, 1]), to: [1, frames, upsampleScale]).reshaped([1, frames * upsampleScale, 1])
        return source(upsampled)
    }

    /// The same forward, additionally returning the conv_post output (the pre-iSTFT seam).
    func callWithConvPost(_ input: MLXArray, _ style: MLXArray, f0: MLXArray) -> (audio: MLXArray, convPost: MLXArray) {
        // Harmonic source: nearest-upsample F0, generate the excitation, and STFT it into the noise band.
        let frames = f0.dim(1)
        let upsampled = broadcast(f0.reshaped([1, frames, 1]), to: [1, frames, upsampleScale]).reshaped([1, frames * upsampleScale, 1])
        let harSource = source(upsampled)                                 // [1, L, 1]
        let (harSpec, harPhase) = stft.transform(harSource[0..., 0..., 0])  // [1, bins, T_stft]
        let har = concatenated([harSpec, harPhase], axis: 1)              // [1, 22, T_stft]

        var x = input.transposed(0, 2, 1)                                 // NLC
        for i in 0 ..< numUpsamples {
            x = leakyRelu(x, negativeSlope: 0.1)
            var xSource = noiseConvs[i](har.transposed(0, 2, 1))          // NLC
            xSource = noiseRes[i](xSource.transposed(0, 2, 1), style).transposed(0, 2, 1)   // NCL in/out
            x = ups[i](x)                                                 // NLC upsample
            if i == numUpsamples - 1 {
                // reflection_pad (1, 0) along time.
                x = concatenated([x[0..., 1 ..< 2, 0...], x], axis: 1)
            }
            x = x + xSource
            var summed = resblocks[i * numKernels](x.transposed(0, 2, 1), style)
            for j in 1 ..< numKernels {
                summed = summed + resblocks[i * numKernels + j](x.transposed(0, 2, 1), style)
            }
            x = (summed / Float(numKernels)).transposed(0, 2, 1)          // NLC
        }
        x = leakyRelu(x, negativeSlope: 0.01)                            // the reference's bare leaky_relu default
        let post = convPost(x).transposed(0, 2, 1)                       // [1, 22, T_stft]
        let bins = stft.nFFT / 2 + 1
        let spec = exp(post[0..., 0 ..< bins, 0...])
        let phase = sin(post[0..., bins ..< (stft.nFFT + 2), 0...])
        return (stft.inverse(magnitude: spec, phase: phase), post)
    }
}

/// The Decoder: composes the alignment-expanded asr with the F0/energy curves, runs the AdaIN residual
/// stack, and the iSTFTNet generator.
final class NFKKokoroDecoder: Module {
    @ModuleInfo(key: "encode") var encode: NFKKokoroAdainResBlk1d
    @ModuleInfo(key: "decode") var decode: [NFKKokoroAdainResBlk1d]
    @ModuleInfo(key: "F0_conv") var f0Conv: Conv1d
    @ModuleInfo(key: "N_conv") var nConv: Conv1d
    @ModuleInfo(key: "asr_res") var asrRes: [Conv1d]
    @ModuleInfo(key: "generator") var generator: NFKKokoroGenerator

    init(_ config: NFKMLXKokoroConfiguration) {
        let hidden = config.hiddenDim, style = config.styleDim
        _encode.wrappedValue = NFKKokoroAdainResBlk1d(dimIn: hidden + 2, dimOut: 1024, styleDim: style, upsample: false)
        _decode.wrappedValue = [
            NFKKokoroAdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: style, upsample: false),
            NFKKokoroAdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: style, upsample: false),
            NFKKokoroAdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: style, upsample: false),
            NFKKokoroAdainResBlk1d(dimIn: 1024 + 2 + 64, dimOut: 512, styleDim: style, upsample: true),
        ]
        _f0Conv.wrappedValue = Conv1d(inputChannels: 1, outputChannels: 1, kernelSize: 3, stride: 2, padding: 1)
        _nConv.wrappedValue = Conv1d(inputChannels: 1, outputChannels: 1, kernelSize: 3, stride: 2, padding: 1)
        _asrRes.wrappedValue = [Conv1d(inputChannels: hidden, outputChannels: 64, kernelSize: 1)]
        _generator.wrappedValue = NFKKokoroGenerator(config)
    }

    /// `asr` `[1, 512, frames]`, `f0Curve`/`n` `[1, frames·2]`, `style` `[1, 128]` → `[1, samples]`.
    func callAsFunction(_ asr: MLXArray, f0Curve: MLXArray, n: MLXArray, style: MLXArray) -> MLXArray {
        let f0 = f0Conv(f0Curve.reshaped([1, f0Curve.dim(1), 1])).transposed(0, 2, 1)   // [1, 1, frames]
        let energy = nConv(n.reshaped([1, n.dim(1), 1])).transposed(0, 2, 1)            // [1, 1, frames]
        var x = encode(concatenated([asr, f0, energy], axis: 1), style)                // [1, 1024, frames]
        let asrRegulated = asrRes[0](asr.transposed(0, 2, 1)).transposed(0, 2, 1)       // [1, 64, frames]
        for block in decode {
            x = concatenated([x, asrRegulated, f0, energy], axis: 1)                    // [1, 1090, frames]
            x = block(x, style)
        }
        return generator(x, style, f0: f0Curve)
    }

    /// The seams for parity localization: the encode output, the generator input, the conv_post output.
    public struct Seams { public let encode: MLXArray; public let genIn: MLXArray; public let convPost: MLXArray }

    public func callWithSeams(_ asr: MLXArray, f0Curve: MLXArray, n: MLXArray, style: MLXArray) -> (audio: MLXArray, seams: Seams) {
        let f0 = f0Conv(f0Curve.reshaped([1, f0Curve.dim(1), 1])).transposed(0, 2, 1)
        let energy = nConv(n.reshaped([1, n.dim(1), 1])).transposed(0, 2, 1)
        let encoded = encode(concatenated([asr, f0, energy], axis: 1), style)
        var x = encoded
        let asrRegulated = asrRes[0](asr.transposed(0, 2, 1)).transposed(0, 2, 1)
        for block in decode {
            x = concatenated([x, asrRegulated, f0, energy], axis: 1)
            x = block(x, style)
        }
        let (audio, convPost) = generator.callWithConvPost(x, style, f0: f0Curve)
        return (audio, Seams(encode: encoded, genIn: x, convPost: convPost))
    }
}

// MARK: - Vocabulary, voice, and speech backend

extension NFKMLXKokoro {
    /// Reads the phoneme→id vocabulary from a release's `config.json`.
    public static func loadVocab(directoryURL: URL) throws -> [String: Int] {
        let data = try Data(contentsOf: directoryURL.appendingPathComponent("config.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let vocab = json?["vocab"] as? [String: Int] else {
            throw NFKMLXError.unsupportedConfiguration("config.json has no vocab")
        }
        return vocab
    }

    /// Loads a Kokoro voicepack (`[styles, 1, 256]`). A released `.pt` voicepack is a bare tensor the
    /// native checkpoint reader does not interpret, so it converts to safetensors offline (a single
    /// `voice` tensor); this reads that. A sibling `<name>.safetensors` is used when present.
    public static func loadVoice(url: URL) throws -> MLXArray {
        let safetensorsURL = url.pathExtension == "safetensors"
            ? url : url.deletingPathExtension().appendingPathExtension("safetensors")
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: safetensorsURL)
        guard let voice = checkpoint.arrays["voice"] ?? checkpoint.arrays.values.first else {
            throw NFKMLXError.unsupportedConfiguration("expected a single voicepack tensor")
        }
        return voice
    }
}

extension NFKMLXKokoroNet {
    /// Synthesizes a waveform from a phoneme string and a voicepack. The phonemes are mapped to ids by
    /// `vocab` (per Unicode scalar, as the reference iterates), wrapped in the boundary token 0, and the
    /// style is the voicepack row for the phoneme count.
    public func synthesize(phonemes: String, voice: MLXArray, vocab: [String: Int]) -> MLXArray {
        let tokens = phonemes.unicodeScalars.compactMap { vocab[String($0)] }
        let ids = [0] + tokens + [0]
        let idArray = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let index = min(max(phonemes.unicodeScalars.count - 1, 0), voice.dim(0) - 1)
        let refS = voice[index]                                           // [1, 256]
        return synthesize(ids: idArray, refS: refS)
    }
}

/// Holds the model, voice, and vocabulary for capture in the speech backend's `@Sendable` closure.
private final class NFKKokoroHolder: @unchecked Sendable {
    let net: NFKMLXKokoroNet
    let voice: MLXArray
    let vocab: [String: Int]
    init(_ net: NFKMLXKokoroNet, voice: MLXArray, vocab: [String: Int]) {
        self.net = net
        self.voice = voice
        self.vocab = vocab
    }
}

extension NFKMLXKokoro {
    /// Builds a text-to-speech backend from a Kokoro release directory and a voice name. The backend
    /// reads a **phoneme** string under `NFKInputPrompt` (Kokoro's front end, misaki, is separate) and
    /// returns a 24 kHz waveform. `voiceName` names a file under `voices/` (e.g. `af_heart`).
    public static func speechBackend(directoryURL: URL, voiceName: String) throws -> NFKMLXSpeechBackend {
        let net = NFKMLXKokoroNet(.v1)
        try loadWeights(into: net, from: directoryURL.appendingPathComponent("kokoro-v1_0.pth"))
        let vocab = try loadVocab(directoryURL: directoryURL)
        let voice = try loadVoice(url: directoryURL.appendingPathComponent("voices/\(voiceName).pt"))
        let holder = NFKKokoroHolder(net, voice: voice, vocab: vocab)
        return NFKMLXSpeechBackend(identifier: "kokoro",
                                   configuration: NFKMLXSpeechConfiguration(sampleRate: 24000)) { phonemes, _ in
            holder.net.synthesize(phonemes: phonemes, voice: holder.voice, vocab: holder.vocab)
        }
    }

    /// The Objective-C entry: a phoneme string under `NFKInputPrompt` → a 24 kHz WAV under
    /// `NFKOutputAudio`. `voiceName` names a `voices/<name>.pt` in the release directory.
    @objc(kokoroBackendWithDirectoryURL:voiceName:error:)
    public static func backend(directoryURL: URL, voiceName: String) throws -> NFKMLXSpeechBackend {
        try speechBackend(directoryURL: directoryURL, voiceName: voiceName)
    }
}
