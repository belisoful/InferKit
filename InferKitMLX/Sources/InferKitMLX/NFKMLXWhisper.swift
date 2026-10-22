//
//  NFKMLXWhisper.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

// Whisper is an encoder-decoder transformer for speech-to-text. `NFKMLXWhisperBackend` runs it as an
// audio → text InferKit backend: audio under `NFKInputAudio` (an `NFKAudioAsset` WAV, or NSData) →
// a log-mel spectrogram → the audio encoder → a greedy autoregressive text decoder → tokens under
// `NFKOutputText`. Tensors flow NHWC-style (channels last); the mel uses MLXFFT's `rfft`.
//
// The module structure and names follow OpenAI Whisper (`encoder`/`decoder`, `blocks.N`,
// `attn`/`cross_attn`, `mlp.0`/`mlp.2`), so a converted checkpoint loads with the conv-weight
// transpose. The mel filterbank is HTK-triangular (the reference uses librosa/Slaney) and audio is
// assumed already 16 kHz — both sweep items. Detokenization uses a supplied `NFKTokenizer`, or returns
// raw token ids.

/// Whisper dimensions. Defaults are the `tiny` model.
public struct NFKMLXWhisperConfiguration: Sendable {
    public var nMels: Int = 80
    public var nAudioState: Int = 384
    public var nAudioHead: Int = 6
    public var nAudioLayer: Int = 4
    public var nAudioCtx: Int = 1500
    public var nVocab: Int = 51865
    public var nTextState: Int = 384
    public var nTextHead: Int = 6
    public var nTextLayer: Int = 4
    public var nTextCtx: Int = 448
    /// The decode prompt, the stop token, and the suppression boundary. Defaults are the real Whisper
    /// multilingual ids: the prompt is `<|startoftranscript|>` `<|en|>` `<|transcribe|>`
    /// `<|notimestamps|>`; generation runs until `<|endoftext|>` (`endToken`); `suppressFrom` masks the
    /// special and timestamp ids (everything at or above it) so greedy decoding stays in the text vocabulary.
    public var promptTokens: [Int] = [50258, 50259, 50359, 50363]
    public var endToken: Int = 50257
    public var suppressFrom: Int = 50258
    public var maxTokens: Int = 64

    /// Ids masked at every step, on top of the `suppressFrom` range.
    ///
    /// The reference's default `suppress_tokens` is `"-1"`, which expands to its non-speech set — the
    /// punctuation and music symbols a transcript should never contain. Build it from the model's own
    /// tokenizer with ``NFKMLXWhisperSuppression/nonSpeechTokens(using:)`` rather than hard-coding ids,
    /// because the set is whatever those symbols encode to in that vocabulary.
    public var suppressTokens: [Int] = []

    /// Whether the first sampled position may be a space or an immediate end of text.
    ///
    /// The reference's `SuppressBlank` masks `" "` and `<|endoftext|>` at that one position only, so a
    /// segment cannot open with whitespace or decode to nothing.
    public var suppressesBlankStart: Bool = true

    /// The id of `<|0.00|>`, where the timestamp range opens.
    ///
    /// Every id from here to the end of the vocabulary is a time rather than a word, one per 20
    /// milliseconds of the 30-second window. It sits one past `<|notimestamps|>`, which is the last
    /// token of the default prompt — large-v3 carries an extra language token, so both shift together.
    public var timestampBegin: Int = 50364

    /// Seconds a timestamp id advances, which is the window divided by the encoder's frame count.
    public var timestampPrecision: Float = 0.02

    /// How far into the window the FIRST timestamp may fall, in timestamp ids.
    ///
    /// The reference's `max_initial_timestamp` is one second, which at this precision is 50 ids. `nil`
    /// leaves the opening timestamp unbounded.
    public var maxInitialTimestampIndex: Int? = 50

    public init() {}

    /// The released `tiny` model, and this type's defaults.
    public static let tiny = NFKMLXWhisperConfiguration()

    /// The released `base` model.
    public static let base = NFKMLXWhisperConfiguration(state: 512, heads: 8, layers: 6)

    /// The released `small` model.
    public static let small = NFKMLXWhisperConfiguration(state: 768, heads: 12, layers: 12)

    /// The released `medium` model.
    public static let medium = NFKMLXWhisperConfiguration(state: 1024, heads: 16, layers: 24)

    /// The released `large-v1` and `large-v2` models.
    ///
    /// Both are the 1280-wide, 32-layer geometry `large-v3` also has, with the 80-band front end and
    /// the 51865-token vocabulary every size before v3 shares. The two releases differ only in their
    /// training, so one configuration fits either checkpoint.
    public static let large = NFKMLXWhisperConfiguration(state: 1280, heads: 20, layers: 32)

    /// The released `large-v3` model.
    ///
    /// It differs from every earlier size in more than depth: its front end produces **128** mel bands
    /// rather than 80, and its vocabulary is one token larger, which shifts `<|transcribe|>` and
    /// `<|notimestamps|>` up by one. Reusing the smaller sizes' prompt would ask it for a different
    /// task; both values come from the model's own tokenizer.
    public static let largeV3: NFKMLXWhisperConfiguration = {
        var configuration = NFKMLXWhisperConfiguration(state: 1280, heads: 20, layers: 32)
        configuration.nMels = 128
        configuration.nVocab = 51866
        configuration.promptTokens = [50258, 50259, 50360, 50364]
        configuration.timestampBegin = 50365
        return configuration
    }()

    /// The released `large-v3-turbo` model.
    ///
    /// The `large-v3` encoder over a decoder pruned to **four** layers, which is where the speed comes
    /// from: decoding runs the decoder once per token and the encoder once per window. It is the one
    /// released size whose encoder and decoder differ in depth, so it is built from `largeV3` rather
    /// than from the shared-depth initializer.
    public static let largeV3Turbo: NFKMLXWhisperConfiguration = {
        var configuration = NFKMLXWhisperConfiguration.largeV3
        configuration.nTextLayer = 4
        return configuration
    }()

    /// A size where the encoder and decoder share their width, head count, and depth, which every
    /// released Whisper but `large-v3-turbo` does.
    init(state: Int, heads: Int, layers: Int) {
        self.init()
        nAudioState = state
        nAudioHead = heads
        nAudioLayer = layers
        nTextState = state
        nTextHead = heads
        nTextLayer = layers
    }
}

/// The Whisper size to build, for the Objective-C factory.
///
/// The cases are every released multilingual size: `tiny`, `base`, `small`, `medium`, `large`
/// (`large-v1` and `large-v2`, one geometry), `largeV3`, and `largeV3Turbo`. The English-only
/// `.en` releases share each size's geometry and differ in the tokenizer; they load under the same case.
@objc(NFKMLXWhisperVariant)
public enum NFKMLXWhisperVariant: Int {
    case tiny
    case small
    case medium
    case largeV3
    case base
    case large
    case largeV3Turbo
}

/// The reference decoder's non-speech suppression set.
///
/// `whisper.decode()` masks a curated list of symbol tokens at every step. The list is defined by the
/// symbols themselves, not by fixed ids, so it is computed against the tokenizer the model ships with:
/// an English-only and a multilingual vocabulary encode them to different ids.
public enum NFKMLXWhisperSuppression {

    /// The reference `Tokenizer.non_speech_tokens` symbol list.
    static let symbols: [String] = {
        let single = Array("\"#()*+/:;<=>@[\\]^_`{|}~「」『』").map(String.init)
        let multi = ["<<", ">>", "<<<", ">>>", "--", "---", "-(", "-[", "('", "(\"", "((", "))",
                     "(((", ")))", "[[", "]]", "{{", "}}", "♪♪", "♪♪♪"]
        return single + multi
    }()

    /// Musical symbols, which the reference suppresses even where they encode to several tokens.
    static let miscellaneous: [String] = Array("♩♪♫♬♭♮♯").map(String.init)

    /// Computes the non-speech ids for `tokenizer`, following the reference rule: a symbol contributes
    /// its first token when it encodes to exactly one token, and a musical symbol contributes its first
    /// token however many it encodes to. `" -"` and `" '"` are always included.
    public static func nonSpeechTokens(using tokenizer: NFKTokenizer) -> [Int] {
        var result = Set<Int>()
        func first(_ text: String) -> [Int] { tokenizer.encode(text).map(\.intValue) }

        for opener in [" -", " '"] {
            if let id = first(opener).first { result.insert(id) }
        }
        for symbol in symbols {
            for candidate in [symbol, " " + symbol] where first(candidate).count == 1 {
                if let id = first(candidate).first { result.insert(id) }
            }
        }
        for symbol in miscellaneous {
            for candidate in [symbol, " " + symbol] {
                if let id = first(candidate).first { result.insert(id) }
            }
        }
        return result.sorted()
    }
}

// MARK: Log-mel spectrogram

enum NFKMLXMel {
    /// `[frames, nFFT/2 + 1]` — a Hann-windowed short-time power spectrum, centered on each frame by
    /// reflecting `nFFT/2` samples at both ends (`torch.stft(center: true)`). `dropsFinalFrame` follows
    /// Whisper's `stft[..., :-1]`; the reference front ends that keep every frame pass false.
    static func powerSpectrogram(_ samples: [Float], nFFT: Int, hop: Int, dropsFinalFrame: Bool) -> MLXArray {
        let window = (0 ..< nFFT).map { 0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(nFFT)) }
        let pad = nFFT / 2
        var padded: [Float]
        if samples.count >= pad + 1 {
            padded = (1 ... pad).reversed().map { samples[$0] } + samples
                   + (0 ..< pad).map { samples[samples.count - 2 - $0] }
        } else {
            padded = [Float](repeating: 0, count: pad) + samples + [Float](repeating: 0, count: pad)
        }
        if padded.count < nFFT { padded += [Float](repeating: 0, count: nFFT - padded.count) }
        let available = 1 + (padded.count - nFFT) / hop
        let frames = max(1, dropsFinalFrame ? available - 1 : available)

        var frameData = [Float](repeating: 0, count: frames * nFFT)
        for f in 0 ..< frames {
            for j in 0 ..< nFFT {
                frameData[f * nFFT + j] = padded[f * hop + j] * window[j]
            }
        }
        let frameArray = frameData.withUnsafeBufferPointer { MLXArray($0, [frames, nFFT]) }
        let spectrum = rfft(frameArray, axis: 1)
        return spectrum.realPart() * spectrum.realPart() + spectrum.imaginaryPart() * spectrum.imaginaryPart()
    }

    static func logMel(_ samples: [Float], sampleRate: Int, nMels: Int) -> MLXArray {
        let nFFT = 400, hop = 160
        let bins = nFFT / 2 + 1
        let power = powerSpectrogram(samples, nFFT: nFFT, hop: hop, dropsFinalFrame: true)
        let frames = power.shape[0]

        let filters = melFilters(sampleRate: sampleRate, bins: bins, nMels: nMels)   // [bins, nMels]
        let mel = power.matmul(filters)                         // [frames, nMels]
        var logMel = log(maximum(mel, MLXArray(1e-10))) / logf(10)
        logMel = maximum(logMel, logMel.max() - 8)
        logMel = (logMel + 4) / 4
        return logMel.reshaped([1, frames, nMels])
    }

    /// `[bins, nMels]`, the filterbank shared by every mel front end here. A model trained on a band
    /// narrower than the full spectrum (PANNs takes 50 Hz to 14 kHz) sets `fMinimum`/`fMaximum`; the
    /// defaults span the whole range, as Whisper and MarbleNet do.
    static func melFilters(sampleRate: Int, bins: Int, nMels: Int,
                           fMinimum: Float = 0, fMaximum: Float? = nil) -> MLXArray {
        // Slaney-scale mel filterbank with Slaney area normalization — librosa `mel(htk=False,
        // norm='slaney')`, which is what OpenAI Whisper's precomputed filters use, and what NeMo builds
        // for MarbleNet. The HTK log-mel scale is a different curve and shifts the features enough to
        // corrupt transcription.
        let nyquist = Float(sampleRate) / 2
        let fSp: Float = 200.0 / 3.0                          // linear step below 1 kHz
        let minLogHz: Float = 1000
        let minLogMel = minLogHz / fSp
        let logStep = logf(6.4) / 27.0
        func hzToMel(_ f: Float) -> Float { f < minLogHz ? f / fSp : minLogMel + logf(f / minLogHz) / logStep }
        func melToHz(_ m: Float) -> Float { m < minLogMel ? m * fSp : minLogHz * expf(logStep * (m - minLogMel)) }

        let melMin = hzToMel(fMinimum), melMax = hzToMel(fMaximum ?? nyquist)
        let hzPoints = (0 ... nMels + 1).map { melToHz(melMin + (melMax - melMin) * Float($0) / Float(nMels + 1)) }
        let fftFreqs = (0 ..< bins).map { Float($0) * nyquist / Float(bins - 1) }

        var filter = [Float](repeating: 0, count: bins * nMels)
        for m in 0 ..< nMels {
            let lower = hzPoints[m], center = hzPoints[m + 1], upper = hzPoints[m + 2]
            let enorm: Float = 2.0 / (upper - lower)            // Slaney normalization by filter width
            for k in 0 ..< bins {
                let f = fftFreqs[k]
                let weight = max(0, min((f - lower) / (center - lower), (upper - f) / (upper - center)))
                filter[k * nMels + m] = weight * enorm
            }
        }
        return filter.withUnsafeBufferPointer { MLXArray($0, [bins, nMels]) }
    }
}

// MARK: Transformer blocks

/// Whisper multi-head attention (self or cross), with an optional additive mask.
final class NFKWhisperAttention: Module {
    // These carry `@ModuleInfo` so a projection can be substituted, which is what low-rank adaptation
    // does. MLX writes a replacement child only through the wrapper; a plain property cannot receive
    // one. The wrapper keys match the property names, so checkpoint names are unchanged.
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear
    private let heads: Int

    init(state: Int, heads: Int) {
        self.heads = heads
        _query.wrappedValue = Linear(state, state)
        _key.wrappedValue = Linear(state, state, bias: false)
        _value.wrappedValue = Linear(state, state)
        _out.wrappedValue = Linear(state, state)
    }

    func callAsFunction(_ x: MLXArray, source: MLXArray?, mask: MLXArray?) -> MLXArray {
        let context = source ?? x
        let (batch, tokens, state) = (x.shape[0], x.shape[1], x.shape[2])
        let contextTokens = context.shape[1]
        let headDim = state / heads
        let scale = powf(Float(headDim), -0.25)

        if NFKReferenceRounding.isReduced(x) {
            return out(reducedAttention(x, context: context, mask: mask))
        }
        let q = (query(x) * scale).reshaped([batch, tokens, heads, headDim]).transposed(0, 2, 1, 3)
            .reshaped([batch * heads, tokens, headDim])
        let k = (key(context) * scale).reshaped([batch, contextTokens, heads, headDim]).transposed(0, 2, 1, 3)
            .reshaped([batch * heads, contextTokens, headDim])
        let v = value(context).reshaped([batch, contextTokens, heads, headDim]).transposed(0, 2, 1, 3)
            .reshaped([batch * heads, contextTokens, headDim])

        var scores = q.matmul(k.transposed(0, 2, 1))
        if let mask { scores = scores + mask }
        let attended = softmax(scores, axis: -1).matmul(v)
            .reshaped([batch, heads, tokens, headDim]).transposed(0, 2, 1, 3).reshaped([batch, tokens, state])
        return out(attended)
    }

    /// Attention for `x`'s rows over what `cache` holds: the keys and values of the rows before them, which
    /// `x`'s own are appended to, or for cross-attention `source`'s, projected once. The arithmetic per
    /// row is the uncached call's.
    func callAsFunction(_ x: MLXArray, source: MLXArray?, mask: MLXArray?, cache: NFKWhisperAttentionCache) -> MLXArray {
        let (batch, tokens, state) = (x.shape[0], x.shape[1], x.shape[2])
        let headDim = state / heads
        if let source {
            if cache.keys == nil {
                cache.keys = key(source)
                cache.values = value(source)
            }
        } else {
            let keys = key(x)
            let values = value(x)
            cache.keys = cache.keys.map { concatenated([$0, keys], axis: 1) } ?? keys
            cache.values = cache.values.map { concatenated([$0, values], axis: 1) } ?? values
        }
        let keys = cache.keys!
        let values = cache.values!
        let contextTokens = keys.shape[1]
        func split(_ y: MLXArray, _ length: Int) -> MLXArray {
            y.reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3)
        }
        if NFKReferenceRounding.isReduced(x) {
            let q = split(NFKReferenceRounding.scaled(query(x), by: 1 / Float(headDim).squareRoot()), tokens)
            var scores = q.matmul(split(keys, contextTokens).transposed(0, 1, 3, 2))
            if let mask { scores = scores + mask.asType(scores.dtype) }
            return out(NFKReferenceRounding.halfSoftmax(scores, fast: softmax(scores, axis: -1, precise: true))
                .matmul(split(values, contextTokens))
                .transposed(0, 2, 1, 3).reshaped([batch, tokens, state]))
        }
        let scale = powf(Float(headDim), -0.25)
        let q = split(query(x) * scale, tokens).reshaped([batch * heads, tokens, headDim])
        let k = split(keys * scale, contextTokens).reshaped([batch * heads, contextTokens, headDim])
        let v = split(values, contextTokens).reshaped([batch * heads, contextTokens, headDim])
        var scores = q.matmul(k.transposed(0, 2, 1))
        if let mask { scores = scores + mask }
        let attended = softmax(scores, axis: -1).matmul(v)
            .reshaped([batch, heads, tokens, headDim]).transposed(0, 2, 1, 3).reshaped([batch, tokens, state])
        return out(attended)
    }

    /// Half-precision attention rounded as transformers' Whisper attention rounds it: the query scaled
    /// by `headDim^-0.5` in float32 and rounded once, the scores rounded, the softmax formed in float32.
    private func reducedAttention(_ x: MLXArray, context: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, tokens, state) = (x.shape[0], x.shape[1], x.shape[2])
        let headDim = state / heads
        func split(_ y: MLXArray) -> MLXArray {
            y.reshaped([batch, -1, heads, headDim]).transposed(0, 2, 1, 3)
        }
        let q = split(NFKReferenceRounding.scaled(query(x), by: 1 / Float(headDim).squareRoot()))
        var scores = q.matmul(split(key(context)).transposed(0, 1, 3, 2))
        if let mask { scores = scores + mask.asType(scores.dtype) }
        return NFKReferenceRounding.halfSoftmax(scores, fast: softmax(scores, axis: -1, precise: true))
            .matmul(split(value(context)))
            .transposed(0, 2, 1, 3).reshaped([batch, tokens, state])
    }
}

/// One attention's keys and values, as its projections produced them, for a decode that runs one step
/// at a time.
final class NFKWhisperAttentionCache {
    var keys: MLXArray?
    var values: MLXArray?
}

/// A decoder block's self- and cross-attention caches.
final class NFKWhisperBlockCache {
    let attention = NFKWhisperAttentionCache()
    let crossAttention = NFKWhisperAttentionCache()
}

/// A Whisper residual block: self-attention, optional cross-attention, and an MLP.
final class NFKWhisperBlock: Module {
    let attn: NFKWhisperAttention
    @ModuleInfo(key: "attn_ln") var attnLN: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: NFKWhisperAttention?
    @ModuleInfo(key: "cross_attn_ln") var crossAttnLN: LayerNorm?
    @ModuleInfo(key: "mlp") var mlp: [Module]
    @ModuleInfo(key: "mlp_ln") var mlpLN: LayerNorm

    init(state: Int, heads: Int, cross: Bool) {
        attn = NFKWhisperAttention(state: state, heads: heads)
        _attnLN.wrappedValue = NFKLayerNorm(dimensions: state)
        if cross {
            _crossAttn.wrappedValue = NFKWhisperAttention(state: state, heads: heads)
            _crossAttnLN.wrappedValue = NFKLayerNorm(dimensions: state)
        }
        _mlp.wrappedValue = [Linear(state, state * 4), GELU(), Linear(state * 4, state)]
        _mlpLN.wrappedValue = NFKLayerNorm(dimensions: state)
    }

    func callAsFunction(_ x: MLXArray, audio: MLXArray?, mask: MLXArray?) -> MLXArray {
        var h = x + attn(attnLN(x), source: nil, mask: mask)
        if let crossAttn, let crossAttnLN, let audio {
            h = h + crossAttn(crossAttnLN(h), source: audio, mask: nil)
        }
        return h + feedForward(h)
    }

    /// The block over new rows, with each attention's earlier keys and values in `cache`.
    func callAsFunction(_ x: MLXArray, audio: MLXArray?, mask: MLXArray?, cache: NFKWhisperBlockCache) -> MLXArray {
        var h = x + attn(attnLN(x), source: nil, mask: mask, cache: cache.attention)
        if let crossAttn, let crossAttnLN, let audio {
            h = h + crossAttn(crossAttnLN(h), source: audio, mask: nil, cache: cache.crossAttention)
        }
        return h + feedForward(h)
    }

    private func feedForward(_ h: MLXArray) -> MLXArray {
        let activation = mlp[1] as! GELU
        return (mlp[2] as! Linear)(NFKReferenceRounding.wide((mlp[0] as! Linear)(mlpLN(h))) { activation($0) })
    }
}

/// The audio encoder: two convolutions, sinusoidal positions, and self-attention blocks.
final class NFKWhisperEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [NFKWhisperBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    init(_ c: NFKMLXWhisperConfiguration) {
        _conv1.wrappedValue = NFKConv1d(inputChannels: c.nMels, outputChannels: c.nAudioState, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = NFKConv1d(inputChannels: c.nAudioState, outputChannels: c.nAudioState, kernelSize: 3, stride: 2, padding: 1)
        _blocks.wrappedValue = (0 ..< c.nAudioLayer).map { _ in NFKWhisperBlock(state: c.nAudioState, heads: c.nAudioHead, cross: false) }
        _lnPost.wrappedValue = NFKLayerNorm(dimensions: c.nAudioState)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = NFKReferenceRounding.gelu(conv1(mel))
        x = NFKReferenceRounding.gelu(conv2(x))
        // The float32 table would promote a half-precision encoder to float32; the reference keeps its
        // table in float32 and rounds the sum once.
        x = (x + NFKMLXWhisperNet.sinusoids(length: x.shape[1], channels: x.shape[2])).asType(x.dtype)
        for block in blocks { x = block(x, audio: nil, mask: nil) }
        return lnPost(x)
    }
}

/// The text decoder: token + positional embeddings, self/cross-attention blocks, and tied logits.
final class NFKWhisperDecoder: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [NFKWhisperBlock]
    @ModuleInfo(key: "ln") var ln: LayerNorm

    init(_ c: NFKMLXWhisperConfiguration) {
        _tokenEmbedding.wrappedValue = Embedding(embeddingCount: c.nVocab, dimensions: c.nTextState)
        _positionalEmbedding.wrappedValue = MLXArray.zeros([c.nTextCtx, c.nTextState])
        _blocks.wrappedValue = (0 ..< c.nTextLayer).map { _ in NFKWhisperBlock(state: c.nTextState, heads: c.nTextHead, cross: true) }
        _ln.wrappedValue = NFKLayerNorm(dimensions: c.nTextState)
    }

    func callAsFunction(_ tokens: MLXArray, audio: MLXArray) -> MLXArray {
        let length = tokens.shape[1]
        var x = tokenEmbedding(tokens) + positionalEmbedding[0 ..< length]
        let mask = NFKMLXWhisperNet.causalMask(length)
        for block in blocks { x = block(x, audio: audio, mask: mask) }
        x = ln(x)
        return x.matmul(tokenEmbedding.weight.transposed(1, 0))
    }

    /// The logits of `tokens`, which follow the `caches`' rows: each block's caches gain the new rows'
    /// keys and values, and the first call fills the cross-attention's from `audio`.
    func callAsFunction(_ tokens: MLXArray, audio: MLXArray, caches: [NFKWhisperBlockCache]) -> MLXArray {
        let length = tokens.shape[1]
        let offset = caches.first?.attention.keys?.shape[1] ?? 0
        var x = tokenEmbedding(tokens) + positionalEmbedding[offset ..< (offset + length)]
        let mask = length > 1 ? NFKMLXWhisperNet.causalMask(length, after: offset) : nil
        for (block, cache) in zip(blocks, caches) { x = block(x, audio: audio, mask: mask, cache: cache) }
        x = ln(x)
        return x.matmul(tokenEmbedding.weight.transposed(1, 0))
    }
}

/// The Whisper model: audio encoder + text decoder, with greedy transcription.
/// The Whisper encoder-decoder transformer.
///
/// Adapting it to a consumer's own domain works on this type directly: build one with
/// ``NFKMLXWhisper/network(weightsURL:configuration:)`` and train it with
/// ``NFKMLXWhisper/fineTune(_:examples:rank:alpha:objective:optimizer:steps:clipGradientNorm:accumulationSteps:checkpoint:observer:)``.
public final class NFKMLXWhisperNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKWhisperEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKWhisperDecoder

    let configuration: NFKMLXWhisperConfiguration

    init(_ configuration: NFKMLXWhisperConfiguration) {
        self.configuration = configuration
        _encoder.wrappedValue = NFKWhisperEncoder(configuration)
        _decoder.wrappedValue = NFKWhisperDecoder(configuration)
    }

    static func sinusoids(length: Int, channels: Int) -> MLXArray {
        let half = channels / 2
        var values = [Float](repeating: 0, count: length * channels)
        let logTimescale = logf(10000) / Float(max(half - 1, 1))
        for t in 0 ..< length {
            for i in 0 ..< half {
                let scaled = Float(t) * expf(-logTimescale * Float(i))
                values[t * channels + i] = sinf(scaled)
                values[t * channels + half + i] = cosf(scaled)
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [length, channels]) }
    }

    /// The mask of `length` new rows that follow `offset` earlier ones: each sees every earlier row and the
    /// new rows up to itself.
    static func causalMask(_ length: Int, after offset: Int) -> MLXArray {
        let width = offset + length
        var values = [Float](repeating: 0, count: length * width)
        for i in 0 ..< length {
            for j in (offset + i + 1) ..< width {
                values[i * width + j] = -1e9
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [length, width]) }
    }

    static func causalMask(_ length: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: length * length)
        for i in 0 ..< length {
            for j in (i + 1) ..< length {
                values[i * length + j] = -1e9
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [length, length]) }
    }

    /// The float32 log-mel in the encoder's type, so a half-precision load computes at half precision as
    /// the reference does with its input features cast to the model's type.
    func typed(_ mel: MLXArray) -> MLXArray {
        mel.asType(NFKReferenceRounding.parameterType(of: encoder))
    }

    /// Greedy transcription: seeds the decode prompt, applies the decoder's suppression rules, and
    /// returns the generated token ids (prompt excluded).
    ///
    /// Two masks, as in the reference. The standing one covers the special and timestamp range plus
    /// `suppressTokens` — the non-speech symbols. The opening one additionally blocks a space and an
    /// immediate `<|endoftext|>`, and applies only to the first sampled position, which is the
    /// reference's `SuppressBlank`.
    func transcribe(_ mel: MLXArray) -> [Int] {
        let audio = encoder(typed(mel))
        var tokens = configuration.promptTokens
        let promptCount = tokens.count
        let standing = suppressionMask()
        let opening = openingSuppressionMask(standing)
        for step in 0 ..< configuration.maxTokens {
            let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
            let logits = decoder(tokenArray, audio: audio)[0, tokens.count - 1]
                       + (step == 0 ? opening : standing)
            let next = logits.argMax().item(Int.self)
            if next == configuration.endToken { break }
            tokens.append(next)
        }
        return Array(tokens.dropFirst(promptCount))
    }

    /// A span of the clip the decoder marked, with the tokens it transcribed inside it.
    public struct TimedSegment: Sendable {
        /// Seconds from the start of the window.
        public var start: Float
        /// Seconds from the start of the window. Equals `start` for a segment left open at the end.
        public var end: Float
        /// The transcribed ids between the two timestamps, specials excluded.
        public var tokens: [Int]
    }

    /// Greedy transcription that keeps the segment times, which the plain path masks away.
    ///
    /// @discussion The times only exist when `<|notimestamps|>` is left OUT of the prompt and the
    /// timestamp range stays unmasked, so this is a different decode rather than a different reading
    /// of the same one. `ApplyTimestampRules` then orders the result: a timestamp is followed by text
    /// and text by a timestamp, so they come in pairs; a timestamp never precedes an earlier one; the
    /// opening position must be a timestamp, and no later than `maxInitialTimestampIndex`; and where
    /// the timestamps together hold more probability than any single word, a timestamp is taken even
    /// though no single one leads.
    ///
    /// The tokens are returned beside the segments, so a caller who only wants a transcript reads
    /// them without pairing anything up.
    ///
    /// Introduced in InferKit 0.1.0.
    public func transcribeWithTimestamps(_ mel: MLXArray) -> (segments: [TimedSegment], tokens: [Int]) {
        let audio = encoder(typed(mel))
        let prompt = configuration.promptTokens.filter { $0 != noTimestampsToken }
        var tokens = prompt
        let standing = suppressionMask(through: configuration.timestampBegin)
        let opening = openingSuppressionMask(standing)
        for _ in 0 ..< configuration.maxTokens {
            let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
            var logits = decoder(tokenArray, audio: audio)[0, tokens.count - 1]
                       + (tokens.count == prompt.count ? opening : standing)
            logits = logits + timestampRuleMask(sampled: Array(tokens.dropFirst(prompt.count)))
            logits = takingATimestamp(logits) ? logits + textMask() : logits
            let next = logits.argMax().item(Int.self)
            if next == configuration.endToken { break }
            tokens.append(next)
        }
        let sampled = Array(tokens.dropFirst(prompt.count))
        return (segments(from: sampled), sampled)
    }

    /// `<|notimestamps|>`, which sits immediately below the timestamp range.
    var noTimestampsToken: Int { configuration.timestampBegin - 1 }

    /// `<|nospeech|>`, two below the timestamp range in every multilingual vocabulary.
    var noSpeechToken: Int { configuration.timestampBegin - 2 }

    /// `<|startofprev|>`, which opens the previous window's text in a conditioned prompt.
    var startOfPreviousToken: Int { configuration.timestampBegin - 3 }

    /// `<|transcribe|>` and `<|translate|>`, the task a decode prompt names after its language.
    var transcribeToken: Int { configuration.timestampBegin - 5 }
    var translateToken: Int { configuration.timestampBegin - 6 }

    /// The language tokens, from the one after `<|startoftranscript|>` to the one before `<|translate|>`.
    var languageTokens: Range<Int> { (configuration.promptTokens[0] + 1) ..< translateToken }

    /// The decode prompt that opens a window: `<|startoftranscript|>`, the language, the task, and
    /// `<|notimestamps|>` for a decode without timestamps.
    func promptSequence(language: Int, translates: Bool, timestamps: Bool) -> [Int] {
        [configuration.promptTokens[0], language, translates ? translateToken : transcribeToken]
            + (timestamps ? [] : [noTimestampsToken])
    }

    /// The language token the model gives a window, as the reference's `detect_language` reads it: the
    /// logits after `<|startoftranscript|>` alone, over the language tokens.
    ///
    /// Introduced in InferKit 0.4.0.
    public func detectLanguage(_ mel: MLXArray) -> Int {
        let audio = encoder(typed(mel))
        let start = MLXArray([Int32(configuration.promptTokens[0])]).reshaped([1, 1])
        let logits = decoder(start, audio: audio)[0, 0].asType(.float32)
        return languageTokens.lowerBound + logits[languageTokens].argMax().item(Int.self)
    }

    // MARK: Long-form transcription

    /// One window's greedy decode as the reference's `DecodingTask` runs it at temperature 0: the ids it
    /// sampled (the closing `<|endoftext|>` excluded), the summed log-probability of every id it took
    /// (the closing one included), and the probability the opening position gave `<|nospeech|>`.
    ///
    /// Introduced in InferKit 0.4.0.
    public struct WindowDecode: Sendable {
        public var tokens: [Int]
        public var sumLogProbability: Double
        public var noSpeechProbability: Double
        /// The reference's `avg_logprob`: the sum over one more than the sampled ids.
        public var averageLogProbability: Double { sumLogProbability / Double(tokens.count + 1) }
    }

    /// Decodes one 30-second window, conditioned on `previous`: the reference prepends `<|startofprev|>`
    /// and the last `nTextCtx / 2 - 1` previous ids to the decode prompt. The rules are the plain or the
    /// timestamped decode's, applied from the first sampled position.
    ///
    /// Introduced in InferKit 0.4.0.
    public func decodeWindow(_ mel: MLXArray, previous: [Int], timestamps: Bool) -> WindowDecode {
        decodeWindow(mel, previous: previous, sequence: timestamps ? configuration.promptTokens.filter { $0 != noTimestampsToken }
                                                                    : configuration.promptTokens,
                     timestamps: timestamps)
    }

    /// Decodes one window as ``decodeWindow(_:previous:timestamps:)`` does, opened by `sequence`, the prompt
    /// ``promptSequence(language:translates:timestamps:)`` makes.
    ///
    /// Introduced in InferKit 0.4.0.
    public func decodeWindow(_ mel: MLXArray, previous: [Int], sequence: [Int], timestamps: Bool) -> WindowDecode {
        let audio = encoder(typed(mel))
        let context = Array(previous.suffix(configuration.nTextCtx / 2 - 1))
        var tokens = (context.isEmpty ? [] : [startOfPreviousToken] + context) + sequence
        let sampleBegin = tokens.count
        let startIndex = context.isEmpty ? 0 : context.count + 1
        let standing = referenceSuppressionMask()
        let opening = openingSuppressionMask(standing)
        var sum = 0.0
        var noSpeech = 0.0
        let caches = decoder.blocks.map { _ in NFKWhisperBlockCache() }
        var fresh = tokens
        // The reference's sample_len, half the text context, whatever the configuration's single-window limit.
        for _ in 0 ..< configuration.nTextCtx / 2 {
            let tokenArray = MLXArray(fresh.map { Int32($0) }).reshaped([1, fresh.count])
            let all = decoder(tokenArray, audio: audio, caches: caches)[0].asType(.float32)
            if tokens.count == sampleBegin {
                let opened = softmax(all[startIndex], axis: -1)
                noSpeech = Double(opened[noSpeechToken].item(Float.self))
            }
            var logits = all[fresh.count - 1] + (tokens.count == sampleBegin ? opening : standing)
            if timestamps {
                logits = logits + timestampRuleMask(sampled: Array(tokens.dropFirst(sampleBegin)))
                logits = takingATimestamp(logits) ? logits + textMask() : logits
            }
            let next = logits.argMax().item(Int.self)
            sum += Double((logits - logSumExp(logits))[next].item(Float.self))
            if next == configuration.endToken { break }
            tokens.append(next)
            fresh = [next]
            if tokens.count > configuration.nTextCtx { break }
        }
        return WindowDecode(tokens: Array(tokens.dropFirst(sampleBegin)), sumLogProbability: sum, noSpeechProbability: noSpeech)
    }

    /// A span of a recording, in seconds from its start, with the ids decoded inside it.
    ///
    /// Introduced in InferKit 0.4.0.
    public struct RecordingSegment: Sendable {
        public var start: Double
        public var end: Double
        public var tokens: [Int]
    }

    /// The mel frames a window covers: 30 seconds at the 10 ms hop.
    static let windowFrames = 3000

    /// The segments one window's ids make, in seconds from the recording's start (`offset` is the window's
    /// start), and the mel frames `seek` moves by: the reference's reading of a window that covered
    /// `frames` frames. Each timestamp is two frames.
    func windowSegments(_ tokens: [Int], offset: Double, frames: Int) -> (segments: [RecordingSegment], advance: Int) {
        let begin = configuration.timestampBegin
        // The reference's time_precision: two mel frames of 160 samples at 16 kHz, a double.
        let precision = Double(2 * 160) / 16_000
        let isTimestamp = tokens.map { $0 >= begin }
        let singleTimestampEnding = isTimestamp.count >= 2 && !isTimestamp[isTimestamp.count - 2] && isTimestamp[isTimestamp.count - 1]
        var consecutive = [Int]()
        for index in 1 ..< max(tokens.count, 1) where isTimestamp[index - 1] && isTimestamp[index] {
            consecutive.append(index)
        }
        guard !consecutive.isEmpty else {
            var duration = Double(frames) * 0.01
            if let lastTimestamp = tokens.last(where: { $0 >= begin }), lastTimestamp != begin {
                duration = Double(lastTimestamp - begin) * precision
            }
            return ([RecordingSegment(start: offset, end: offset + duration, tokens: tokens)], frames)
        }
        let slices = singleTimestampEnding ? consecutive + [tokens.count] : consecutive
        var segments = [RecordingSegment]()
        var last = 0
        for slice in slices {
            let piece = Array(tokens[last ..< slice])
            segments.append(RecordingSegment(start: offset + Double(piece[0] - begin) * precision,
                                             end: offset + Double(piece[piece.count - 1] - begin) * precision, tokens: piece))
            last = slice
        }
        return (segments, singleTimestampEnding ? frames : (tokens[last - 1] - begin) * 2)
    }

    /// A whole recording, transcribed as the reference's `transcribe` does at temperature 0 with each
    /// window conditioned on the text before it. `language` is the language token, or nil to detect it
    /// with ``detectLanguage(_:)``; `translates` asks for English text with `<|translate|>`. `progress`
    /// receives the share of the recording read, the segments, and the ids after each window, and stops
    /// the transcription by answering false.
    ///
    /// @discussion The log-mel is computed once over the recording followed by 30 seconds of silence, so
    /// its range is the whole file's. Each pass takes the 30 seconds from `seek`, zero-padded in the mel,
    /// and decodes it:
    /// - A window the model calls silent (`<|nospeech|>` above 0.6, and an average log-probability not
    ///   above -1) is skipped whole.
    /// - With timestamps, each pair of adjacent timestamps closes a segment. A window that ends on a
    ///   single timestamp moves `seek` past the window; otherwise `seek` moves to the last closed
    ///   timestamp, so the rest is decoded again in the next window. A window with no closed pair is one
    ///   segment up to its last timestamp, or its length.
    /// - Without timestamps, each window is one segment of its length.
    /// A segment that has no length or no text keeps its place with no ids. `decodeText` turns ids into
    /// a segment's text for that test.
    ///
    /// Introduced in InferKit 0.4.0.
    public func transcribeRecording(_ samples: [Float], timestamps: Bool, language: Int? = nil, translates: Bool = false,
                                    decodeText: ([Int]) -> String,
                                    progress: ((_ fraction: Double, _ segments: [RecordingSegment], _ tokens: [Int]) -> Bool)? = nil)
        -> (segments: [RecordingSegment], tokens: [Int], language: Int) {
        let rate = 16_000
        let mel = NFKMLXMel.logMel(samples + [Float](repeating: 0, count: 30 * rate), sampleRate: rate, nMels: configuration.nMels)
        let contentFrames = mel.shape[1] - Self.windowFrames
        // The reference detects on the recording's first 30 seconds of mel, its padding included.
        let spoken = language ?? detectLanguage(mel[0..., 0 ..< Self.windowFrames, 0...])
        let sequence = promptSequence(language: spoken, translates: translates, timestamps: timestamps)
        var seek = 0
        var allTokens = [Int]()
        var segments = [RecordingSegment]()
        while seek < contentFrames {
            let offset = Double(seek * 160) / 16_000
            let size = min(Self.windowFrames, contentFrames - seek)
            var window = mel[0..., seek ..< (seek + size), 0...]
            if size < Self.windowFrames {
                window = padded(window, widths: [.init((0, 0)), .init((0, Self.windowFrames - size)), .init((0, 0))])
            }
            let decoded = decodeWindow(window, previous: allTokens, sequence: sequence, timestamps: timestamps)
            if decoded.noSpeechProbability > 0.6 && !(decoded.averageLogProbability > -1.0) {
                seek += size
                if let progress, !progress(Double(min(seek, contentFrames)) / Double(contentFrames), segments, allTokens) { break }
                continue
            }
            var (current, advance) = windowSegments(decoded.tokens, offset: offset, frames: size)
            // The timestamp rules give every segment a length, so the reference always moves on. A window
            // that would not is taken whole.
            seek += advance > 0 ? advance : size
            for index in current.indices {
                let text = decodeText(current[index].tokens.filter { $0 < configuration.endToken })
                if current[index].start == current[index].end || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    current[index].tokens = []
                }
            }
            segments += current
            allTokens += current.flatMap(\.tokens)
            if let progress, !progress(Double(min(seek, contentFrames)) / Double(contentFrames), segments, allTokens) { break }
        }
        return (segments, allTokens, spoken)
    }

    /// Pairs the sampled ids into spans. A trailing timestamp with no closing one stays open.
    private func segments(from sampled: [Int]) -> [TimedSegment] {
        var result = [TimedSegment]()
        var start: Float?
        var tokens = [Int]()
        for id in sampled {
            guard id >= configuration.timestampBegin else {
                if start != nil { tokens.append(id) }
                continue
            }
            let seconds = Float(id - configuration.timestampBegin) * configuration.timestampPrecision
            if let opened = start {
                result.append(TimedSegment(start: opened, end: seconds, tokens: tokens))
                start = nil
                tokens = []
            } else {
                start = seconds
            }
        }
        if let opened = start {
            result.append(TimedSegment(start: opened, end: opened, tokens: tokens))
        }
        return result
    }

    // The pairing and ordering rules, as an additive mask over the whole vocabulary.
    private func timestampRuleMask(sampled: [Int]) -> MLXArray {
        var mask = [Float](repeating: 0, count: configuration.nVocab)
        let begin = configuration.timestampBegin
        func block(_ range: Range<Int>) {
            for id in range.clamped(to: 0 ..< configuration.nVocab) { mask[id] = -1e9 }
        }
        if noTimestampsToken >= 0 { mask[noTimestampsToken] = -1e9 }

        let lastWasTimestamp = sampled.last.map { $0 >= begin } ?? false
        let penultimateWasTimestamp = sampled.count < 2 || sampled[sampled.count - 2] >= begin
        if lastWasTimestamp {
            // A timestamp after a timestamp closes a pair, so the next token has to be a word; a
            // timestamp after a word opens one, so the next has to be a timestamp or the end.
            block(penultimateWasTimestamp ? begin ..< configuration.nVocab
                                          : 0 ..< configuration.endToken)
        }
        if let latest = sampled.last(where: { $0 >= begin }) {
            // A segment cannot run backward, and cannot be empty either — which is what the `+ 1`
            // rules out, on every step except the one that closes a pair.
            block(begin ..< (lastWasTimestamp && !penultimateWasTimestamp ? latest : latest + 1))
        }
        if sampled.isEmpty {
            block(0 ..< begin)
            if let limit = configuration.maxInitialTimestampIndex {
                block((begin + limit + 1) ..< configuration.nVocab)
            }
        }
        return mask.withUnsafeBufferPointer { MLXArray($0, [configuration.nVocab]) }
    }

    // The reference takes a timestamp when the timestamps together outweigh the best single word, so
    // a span boundary is not lost to a vocabulary that spreads its mass over many spellings.
    private func takingATimestamp(_ logits: MLXArray) -> Bool {
        let logProbabilities = logits - logSumExp(logits)
        let begin = configuration.timestampBegin
        let timestamps = logSumExp(logProbabilities[begin...]).item(Float.self)
        let bestWord = logProbabilities[..<begin].max().item(Float.self)
        return timestamps > bestWord
    }

    private func textMask() -> MLXArray {
        var mask = [Float](repeating: 0, count: configuration.nVocab)
        for id in 0 ..< min(configuration.timestampBegin, configuration.nVocab) { mask[id] = -1e9 }
        return mask.withUnsafeBufferPointer { MLXArray($0, [configuration.nVocab]) }
    }

    /// The id a leading space encodes to, which `SuppressBlank` masks at the opening position. Whisper's
    /// vocabulary is byte-level, so a space is the single token `"Ġ"`.
    static let spaceToken = 220

    // An additive logit mask: 0 for an allowed token, -1e9 for a suppressed one. Ids at or above
    // `suppressFrom` are the specials and timestamps; `suppressTokens` carries the non-speech set.
    // Empty suppression (suppressFrom >= nVocab, no listed ids) leaves the logits unchanged.
    private func suppressionMask() -> MLXArray {
        suppressionMask(through: configuration.nVocab)
    }

    // `upperBound` is where the masked range stops: the whole vocabulary for the plain decode, and the
    // start of the timestamps for the timestamped one, which needs them left open.
    private func suppressionMask(through upperBound: Int) -> MLXArray {
        var mask = [Float](repeating: 0, count: configuration.nVocab)
        if configuration.suppressFrom < upperBound {
            for i in configuration.suppressFrom ..< min(upperBound, configuration.nVocab) { mask[i] = -1e9 }
        }
        for id in configuration.suppressTokens where id >= 0 && id < configuration.nVocab {
            mask[id] = -1e9
        }
        return mask.withUnsafeBufferPointer { MLXArray($0, [configuration.nVocab]) }
    }

    /// The reference's `SuppressTokens` for a `transcribe` decode: the non-speech symbols, `<|transcribe|>`,
    /// `<|translate|>`, `<|startoftranscript|>`, `<|startofprev|>`, `<|startoflm|>`, and `<|nospeech|>`. The
    /// language tokens and the timestamps stay open; the timestamp rules order the timestamps.
    func referenceSuppressionMask() -> MLXArray {
        var mask = [Float](repeating: 0, count: configuration.nVocab)
        let specials = [transcribeToken, translateToken, configuration.promptTokens[0], startOfPreviousToken,
                        configuration.timestampBegin - 4, noSpeechToken]
        for id in configuration.suppressTokens + specials where id >= 0 && id < configuration.nVocab {
            mask[id] = -1e9
        }
        return mask.withUnsafeBufferPointer { MLXArray($0, [configuration.nVocab]) }
    }

    // `SuppressBlank`: at the first sampled position only, a space and an immediate end of text are
    // masked too, so a segment cannot open with whitespace or decode to nothing.
    private func openingSuppressionMask(_ standing: MLXArray) -> MLXArray {
        guard configuration.suppressesBlankStart else { return standing }
        var mask = standing.asArray(Float.self)
        for id in [NFKMLXWhisperNet.spaceToken, configuration.endToken]
        where id >= 0 && id < configuration.nVocab {
            mask[id] = -1e9
        }
        return mask.withUnsafeBufferPointer { MLXArray($0, [configuration.nVocab]) }
    }
}

/// Whisper speech-to-text as an InferKit backend.
@objc(NFKMLXWhisperBackend)
public final class NFKMLXWhisperBackend: NSObject, NFKInferenceBackend {

    let net: NFKMLXWhisperNet
    let tokenizer: NFKTokenizer?
    private let identifier: String
    /// Each language code's token: the release's own `added_tokens.json` where it was read, else the
    /// reference's language order over the model's language tokens.
    let languages: [String: Int]

    init(net: NFKMLXWhisperNet, tokenizer: NFKTokenizer?, identifier: String = "whisper", languages: [String: Int]? = nil) {
        self.net = net
        self.tokenizer = tokenizer
        self.identifier = identifier
        self.languages = languages ?? NFKMLXWhisper.languageTable(for: net)
        super.init()
    }

    /// Whether a result carries the transcript's segment times beside its text, for a request that does
    /// not set ``NFKMLXGenerationParameterKey/timestamps``.
    ///
    /// @discussion Off by default, which is the decode every other size and every earlier release of
    /// this backend performs. A catalog build turns it on. Turning it on changes the decode rather than the reading of one: the
    /// prompt drops `<|notimestamps|>` and the timestamp range stays unmasked, so the model is asked a
    /// different question and may answer it with different words.
    ///
    /// A timestamped result adds `NSArray<NFKAudioSegment *>` under `NFKOutputSegments`, each span
    /// labelled with the text inside it when a tokenizer is attached.
    ///
    /// Introduced in InferKit 0.1.0.
    @objc public var emitsTimestamps: Bool = false

    /// Whether a run transcribes the whole recording, window after window as the reference's `transcribe`
    /// does. Off, a run transcribes the first 30 seconds.
    ///
    /// @discussion Off by default, which keeps the single-window decode `whisper.decode` performs. A
    /// catalog build turns it on. Each window decodes up to `nTextCtx / 2` tokens, the reference's `sample_len`; the non-speech symbols are masked only on a backend built with a tokenizer. The recording is resampled to 16 kHz, segments carry seconds from the
    /// start of the file, and each window is conditioned on the text before it. Introduced in InferKit 0.4.0.
    @objc public var transcribesWholeRecording: Bool = false

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        transcribesWholeRecording ? [NFKMLXGenerationParameterKey.timestamps, NFKParameterSourceLanguage, NFKMLXGenerationParameterKey.speechTask]
                                  : [NFKMLXGenerationParameterKey.timestamps]
    }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }

    /// The result keys the backend writes: the transcript, and the timed spans when ``emitsTimestamps`` is
    /// set. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> {
        transcribesWholeRecording ? [NFKOutputText, NFKOutputSegments, NFKOutputLanguage] : [NFKOutputText, NFKOutputSegments]
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try run(request, observer: nil)
    }

    /// A job over ``runInference(for:)``'s decode. A long-form run reports the share of the recording read,
    /// with the transcript and segments so far as its partial result, after each window, and ends at the
    /// next window once the job is cancelled.
    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let backend = NFKMLXWhisperBackendRun(backend: self)
        Task.detached(priority: .userInitiated) {
            do {
                let result = try backend.backend.run(request) { fraction, partial in
                    job.reportProgress(fraction, partialResult: partial)
                    return job.status != .cancelled
                }
                job.finish(with: result)
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    private func run(_ request: NFKInferenceRequest,
                     observer: ((_ fraction: Double, _ partial: NFKInferenceResult) -> Bool)?) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else {
            throw NFKMLXError.unsupportedInput
        }
        let timestamps = (request.parameter(forKey: NFKMLXGenerationParameterKey.timestamps) as? NSNumber)?.boolValue ?? emitsTimestamps
        if transcribesWholeRecording {
            var language: Int?
            if let code = request.parameter(forKey: NFKParameterSourceLanguage) as? String {
                guard let token = languages[code.lowercased()] else {
                    throw NFKMLXError.unsupportedConfiguration("\(identifier) has no language \"\(code)\"")
                }
                language = token
            }
            let task = request.parameter(forKey: NFKMLXGenerationParameterKey.speechTask) as? String ?? NFKMLXWhisper.transcribeTask
            guard [NFKMLXWhisper.transcribeTask, NFKMLXWhisper.translateTask].contains(task) else {
                throw NFKMLXError.unsupportedConfiguration("\(identifier) has no task \"\(task)\"; it transcribes or translates")
            }
            return transcribeRecording(NFKMLXAudioRate.matched(samples, from: sampleRate, to: 16_000), timestamps: timestamps,
                                       language: language, translates: task == NFKMLXWhisper.translateTask, observer: observer)
        }
        // Whisper is trained only on 30-second inputs; pad with silence (or trim) to that length so a
        // short clip is not out of the model's distribution — the dominant cause of word errors.
        let targetSamples = 30 * sampleRate
        var prepared = samples
        if prepared.count < targetSamples {
            prepared += [Float](repeating: 0, count: targetSamples - prepared.count)
        } else if prepared.count > targetSamples {
            prepared = Array(prepared.prefix(targetSamples))
        }
        let mel = NFKMLXMel.logMel(prepared, sampleRate: sampleRate, nMels: net.configuration.nMels)
        guard timestamps else {
            return NFKInferenceResult(outputs: [NFKOutputText: transcript(net.transcribe(mel))])
        }
        let (spans, tokens) = net.transcribeWithTimestamps(mel)
        // The timestamp ids are markers, not words, so the transcript is the text between them.
        let words = tokens.filter { $0 < net.configuration.timestampBegin }
        let segments = spans.map {
            NFKAudioSegment(startSeconds: Double($0.start), endSeconds: Double($0.end),
                            label: transcript($0.tokens), confidence: 1)
        }
        return NFKInferenceResult(outputs: [NFKOutputText: transcript(words),
                                            NFKOutputSegments: segments])
    }

    /// The whole recording's transcript and, with timestamps, its segments in file time. A segment the
    /// reference clears (no length or no text) is left out.
    private func transcribeRecording(_ samples: [Float], timestamps: Bool, language: Int?, translates: Bool,
                                     observer: ((_ fraction: Double, _ partial: NFKInferenceResult) -> Bool)?) -> NFKInferenceResult {
        let (segments, tokens, spoken) = net.transcribeRecording(
            samples, timestamps: timestamps, language: language, translates: translates, decodeText: { transcript($0) },
            progress: observer.map { observer in
                { fraction, segments, tokens in
                    observer(fraction, self.result(segments: segments, tokens: tokens, language: nil, timestamps: timestamps))
                }
            })
        return result(segments: segments, tokens: tokens, language: spoken, timestamps: timestamps)
    }

    /// The transcript, its segments with timestamps (a segment with no ids left out), and the language.
    private func result(segments: [NFKMLXWhisperNet.RecordingSegment], tokens: [Int], language: Int?,
                        timestamps: Bool) -> NFKInferenceResult {
        let begin = net.configuration.timestampBegin
        var outputs: [String: Any] = [NFKOutputText: transcript(tokens.filter { $0 < begin })]
        if let language {
            outputs[NFKOutputLanguage] = languages.first { $0.value == language }?.key
        }
        if timestamps {
            outputs[NFKOutputSegments] = segments.filter { !$0.tokens.isEmpty }.map {
                NFKAudioSegment(startSeconds: $0.start, endSeconds: $0.end,
                                label: transcript($0.tokens.filter { $0 < begin }), confidence: 1)
            }
        }
        return NFKInferenceResult(outputs: outputs)
    }

    private func transcript(_ tokens: [Int]) -> String {
        guard let tokenizer else { return tokens.map(String.init).joined(separator: " ") }
        return tokenizer.decode(tokens.map { NSNumber(value: $0) })
    }

    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data {
            return NFKMLXWaveFile.read(data)
        }
        return nil
    }
}

/// Carries the backend into the detached task that runs a submitted job. The backend's net and tokenizer
/// are read-only once built.
private struct NFKMLXWhisperBackendRun: @unchecked Sendable {
    let backend: NFKMLXWhisperBackend
}

/// Registration and weight loading for Whisper.
@objc(NFKMLXWhisper)
public final class NFKMLXWhisper: NSObject {

    @objc public static let modelName = "whisper-tiny"

    static func makeNet(_ configuration: NFKMLXWhisperConfiguration = NFKMLXWhisperConfiguration()) -> NFKMLXWhisperNet {
        NFKMLXWhisperNet(configuration)
    }

    /// Builds a Whisper speech-to-text backend directly from optional local weights — no registry
    /// required. A nil `weightsURL` builds random weights (`isReady` is true). Reads audio under
    /// `NFKInputAudio`, returns text under `NFKOutputText`. Run inference
    /// off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(weightsURL: weightsURL, tokenizer: nil)
    }

    /// Builds the backend with the model's tokenizer, which decodes the ids into text and supplies the
    /// reference decoder's non-speech suppression set.
    ///
    /// @discussion Without a tokenizer the result carries token ids and suppression is the special and
    /// timestamp range alone. With one, the transcript is text and the decoder additionally masks the
    /// symbols `whisper.decode()` masks, which is what makes the two agree token for token.
    @objc(backendWithWeightsURL:tokenizer:error:)
    public static func backend(weightsURL: URL?, tokenizer: NFKTokenizer?) throws -> any NFKInferenceBackend {
        try backend(weightsURL: weightsURL, tokenizer: tokenizer, timestamps: false)
    }

    /// Builds the backend asking for the transcript's segment times as well as its words.
    ///
    /// @discussion A timestamped result adds `NSArray<NFKAudioSegment *>` under `NFKOutputSegments`,
    /// each span labelled with the text inside it. It is a different decode rather than a different
    /// reading of one — the prompt drops `<|notimestamps|>` and the timestamp range stays unmasked —
    /// so the model may choose different words than the plain path does, which is why it is asked for
    /// rather than always produced.
    ///
    /// Introduced in InferKit 0.1.0.
    @objc(backendWithWeightsURL:tokenizer:timestamps:error:)
    public static func backend(weightsURL: URL?, tokenizer: NFKTokenizer?,
                               timestamps: Bool) throws -> any NFKInferenceBackend {
        try backend(variant: .tiny, weightsURL: weightsURL, tokenizer: tokenizer, timestamps: timestamps)
    }

    /// The geometry each released size is built as. A checkpoint fits only the size it was trained as;
    /// large-v3 also carries an extra language token and 128 mel bands, which its configuration sets.
    static func configuration(for variant: NFKMLXWhisperVariant) -> NFKMLXWhisperConfiguration {
        switch variant {
        case .tiny: return .tiny
        case .base: return .base
        case .small: return .small
        case .medium: return .medium
        case .large: return .large
        case .largeV3: return .largeV3
        case .largeV3Turbo: return .largeV3Turbo
        }
    }

    /// Builds Whisper at one of the released sizes — the enum's whole point. Every size is at parity in
    /// the tests, and this is the only factory that reaches small, medium, and large-v3. The weights load
    /// as the file stores them (`NFKMLXWeightPrecisionCheckpoint`).
    @objc(backendWithVariant:weightsURL:tokenizer:timestamps:error:)
    public static func backend(variant: NFKMLXWhisperVariant, weightsURL: URL?,
                               tokenizer: NFKTokenizer?, timestamps: Bool) throws -> any NFKInferenceBackend {
        try backend(variant: variant, weightsURL: weightsURL, tokenizer: tokenizer, timestamps: timestamps,
                    precision: .checkpoint)
    }

    /// The registry name of a size, which a backend built at that size reports as its identifier.
    /// large-v1 and large-v2 share the `large` geometry, so a `large` backend reports `whisper-large`
    /// unless the registry builds it under its release's own name.
    static func name(for variant: NFKMLXWhisperVariant) -> String {
        switch variant {
        case .tiny: return modelName
        case .base: return "whisper-base"
        case .small: return "whisper-small"
        case .medium: return "whisper-medium"
        case .large: return "whisper-large"
        case .largeV3: return "whisper-large-v3"
        case .largeV3Turbo: return "whisper-large-v3-turbo"
        }
    }

    /// Builds a size at a chosen precision. `NFKMLXWeightPrecisionCheckpoint` keeps the file's type, so
    /// OpenAI's float16 `.pt` releases compute at float16, half the memory of float32;
    /// `NFKMLXWeightPrecisionFloat32` widens a half-precision file and computes in float32. The backend's
    /// identifier is the size's registry name, such as `whisper-base`.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithVariant:weightsURL:tokenizer:timestamps:precision:error:)
    public static func backend(variant: NFKMLXWhisperVariant, weightsURL: URL?, tokenizer: NFKTokenizer?,
                               timestamps: Bool, precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        try backend(variant: variant, weightsURL: weightsURL, tokenizer: tokenizer, timestamps: timestamps,
                    precision: precision, identifier: name(for: variant))
    }

    /// Builds a size whose backend reports `identifier`: the name the registry builds it under.
    static func backend(variant: NFKMLXWhisperVariant, weightsURL: URL?, tokenizer: NFKTokenizer?,
                        timestamps: Bool, precision: NFKMLXWeightPrecision,
                        identifier: String) throws -> any NFKInferenceBackend {
        var configuration = self.configuration(for: variant)
        if let tokenizer {
            configuration.suppressTokens = NFKMLXWhisperSuppression.nonSpeechTokens(using: tokenizer)
        }
        let net = NFKMLXWhisperNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL, precision: precision)
        }
        let holder = NFKMLXWhisperHolder(net)
        let backend = NFKMLXWhisperBackend(net: holder.net, tokenizer: tokenizer, identifier: identifier)
        backend.emitsTimestamps = timestamps
        return backend
    }

    /// The tokens a catalog build decodes at most: the reference's `sample_len`, half the text context.
    static func catalogConfiguration(for variant: NFKMLXWhisperVariant) -> NFKMLXWhisperConfiguration {
        var configuration = self.configuration(for: variant)
        configuration.maxTokens = configuration.nTextCtx / 2
        return configuration
    }

    /// Builds a size as the catalog does, from what its location holds, with timestamps on by default and
    /// the reference's decode length.
    ///
    /// - A directory reads the tokenizer (`vocab.json` and `merges.txt`) and the weights inside it: a
    ///   `.pt` or `model.safetensors`, else `weightsDownload`, the release's checkpoint URL, where the
    ///   model cache that holds the directory keeps it (the cache `repository`'s snapshot lives in), else
    ///   the default cache. The catalog's source is the release's repository with the checkpoint as a URL
    ///   file, so the build finds both.
    /// - A file is the weights, with the tokenizer read beside it when both files are there.
    /// - nil builds random weights without a tokenizer.
    static func catalogBackend(variant: NFKMLXWhisperVariant, location: URL?, weightsDownload: URL?, repository: String? = nil,
                               precision: NFKMLXWeightPrecision, identifier: String) throws -> any NFKInferenceBackend {
        try catalogBackend(configuration: catalogConfiguration(for: variant), location: location,
                           weightsDownload: weightsDownload, repository: repository, precision: precision, identifier: identifier)
    }

    /// Builds a catalog backend of `configuration`, as ``catalogBackend(variant:location:weightsDownload:precision:identifier:)``
    /// builds a released size.
    static func catalogBackend(configuration base: NFKMLXWhisperConfiguration, location: URL?, weightsDownload: URL?,
                               repository: String? = nil, precision: NFKMLXWeightPrecision,
                               identifier: String) throws -> NFKMLXWhisperBackend {
        var directory: URL?
        var weights: URL?
        if let location {
            var isDirectory: ObjCBool = false
            FileManager.default.fileExists(atPath: location.path, isDirectory: &isDirectory)
            directory = isDirectory.boolValue ? location : location.deletingLastPathComponent()
            weights = isDirectory.boolValue ? try catalogWeights(in: location, download: weightsDownload, repository: repository) : location
        }
        let tokenizer = directory.flatMap(tokenizer(in:))
        var configuration = base
        if let tokenizer {
            configuration.suppressTokens = NFKMLXWhisperSuppression.nonSpeechTokens(using: tokenizer)
        }
        let net = NFKMLXWhisperNet(configuration)
        if let weights {
            try loadWeights(into: net, from: weights, precision: precision)
        }
        let holder = NFKMLXWhisperHolder(net)
        let backend = NFKMLXWhisperBackend(net: holder.net, tokenizer: tokenizer, identifier: identifier,
                                           languages: directory.flatMap { languageTable(in: $0, for: holder.net) })
        backend.emitsTimestamps = true
        backend.transcribesWholeRecording = true
        return backend
    }

    /// The tokenizer files a release directory holds beside its weights. `added_tokens.json` names the
    /// language tokens.
    static let tokenizerFiles = ["vocab.json", "merges.txt", "added_tokens.json"]

    /// The task values ``NFKMLXGenerationParameterKey/speechTask`` takes.
    static let transcribeTask = "transcribe"
    static let translateTask = "translate"

    /// The reference's language codes in token order. A size with fewer language tokens takes the first
    /// that many; large-v3 adds `yue`.
    static let languageCodes = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
    ]

    /// The language codes a size knows, in token order.
    static func languageCodes(for variant: NFKMLXWhisperVariant) -> [String] {
        let configuration = self.configuration(for: variant)
        return Array(languageCodes.prefix(configuration.timestampBegin - 6 - (configuration.promptTokens[0] + 1)))
    }

    /// Each language code's token, in the reference's order over the net's language tokens.
    static func languageTable(for net: NFKMLXWhisperNet) -> [String: Int] {
        Dictionary(uniqueKeysWithValues: zip(languageCodes, net.languageTokens))
    }

    /// Each language code's token as a release's `added_tokens.json` names it (`"<|en|>": 50259`), over the
    /// net's language tokens, or nil when the file is absent.
    static func languageTable(in directory: URL, for net: NFKMLXWhisperNet) -> [String: Int]? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("added_tokens.json")),
              let added = (try? JSONSerialization.jsonObject(with: data)) as? [String: NSNumber] else { return nil }
        let tokens = net.languageTokens
        var table = [String: Int]()
        for (token, id) in added where tokens.contains(id.intValue) && token.hasPrefix("<|") && token.hasSuffix("|>") {
            table[String(token.dropFirst(2).dropLast(2))] = id.intValue
        }
        return table.isEmpty ? nil : table
    }

    /// The byte-level BPE tokenizer a release's `vocab.json` and `merges.txt` in directory spell, or nil when
    /// either is absent.
    static func tokenizer(in directory: URL) -> NFKTokenizer? {
        let files = tokenizerFiles.prefix(2).map { directory.appendingPathComponent($0) }
        guard files.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) else { return nil }
        return try? NFKTokenizer(forManifest: ["tokenizer": ["type": "bpe-bytelevel"]], directory: directory)
    }

    /// The checkpoint a release directory holds, else the URL download the cache holding the directory
    /// keeps, else the one the default cache keeps.
    private static func catalogWeights(in directory: URL, download: URL?, repository: String?) throws -> URL {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        if let name = names.sorted().first(where: { $0.hasSuffix(".pt") || $0 == "model.safetensors" }) {
            return directory.appendingPathComponent(name)
        }
        guard let download else {
            throw NFKMLXError.weightsMismatch("no Whisper checkpoint in \(directory.path)")
        }
        // The cache keeps a repository's snapshot at <cache>/<org>/<name>/<revision>.
        var roots = [URL]()
        if let repository {
            var root = directory.deletingLastPathComponent()
            for _ in repository.split(separator: "/") { root = root.deletingLastPathComponent() }
            roots.append(root)
        }
        roots.append(NFKHFHub.defaultCacheDirectoryURL())
        let source = NFKModelSource(kind: NFKModelSourceKindURL, repository: nil, revision: nil, url: download,
                                    layout: NFKModelSourceLayoutFile,
                                    files: [NFKModelFile(path: download.lastPathComponent, role: NFKModelFileRoleWeights)],
                                    requiresConversion: false, extensions: nil)
        for root in roots {
            let status = NFKHFHub(cacheDirectoryURL: root).downloadStatus(source: source)
            if status.presence == .complete, let local = status.localURL {
                return local
            }
        }
        throw NFKMLXError.weightsMismatch("the Whisper checkpoint \(download.lastPathComponent) is not downloaded")
    }

    /// Builds a size from local weights, ids-only (no tokenizer, no timestamps).
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXWhisperVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: variant, weightsURL: weightsURL, tokenizer: nil, timestamps: false)
    }

    /// Builds a size from local weights at a chosen precision, ids-only.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithVariant:weightsURL:precision:error:)
    public static func backend(variant: NFKMLXWhisperVariant, weightsURL: URL?,
                               precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        try backend(variant: variant, weightsURL: weightsURL, tokenizer: nil, timestamps: false, precision: precision)
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
            return try backend(weightsURL: url)
        }
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// The download factory at a chosen size, so small/medium/large-v3 are reachable over the network.
    /// Ids-only; wrap the result with the model's tokenizer through the local variant factory for text.
    /// Blocking; run off the render thread.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXWhisperVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
            return try backend(variant: variant, weightsURL: url)
        }
    }

    /// The download factory at a chosen size and precision. Blocking; run off the render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:precision:error:)
    public static func backend(variant: NFKMLXWhisperVariant, repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
            return try backend(variant: variant, weightsURL: url, precision: precision)
        }
    }

    /// The asynchronous form of the download factory at a chosen size and precision.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:precision:completionHandler:)
    public static func backend(variant: NFKMLXWhisperVariant, repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, precision: NFKMLXWeightPrecision,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0, precision: precision) },
                               completionHandler: completionHandler)
    }

    /// The asynchronous form of the variant download factory.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXWhisperVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers Whisper (`whisper-tiny`) with `NFKMLXModelRegistry`, building as the catalog does
    /// (`catalogBackend`): the tokenizer and checkpoint its location holds, long-form, with timestamps.
    @objc public static func register() {
        let release = NFKMLXCatalogAudio.whisperRelease(named: modelName)
        let download = release.flatMap { URL(string: $0.url) }
        NFKMLXModelRegistry.register(name: modelName) { location in
            try catalogBackend(variant: .tiny, location: location, weightsDownload: download, repository: release?.repository,
                               precision: .checkpoint, identifier: modelName)
        }
    }

    /// Loads a safetensors checkpoint, transposing convolution weights: 4-D `[out,in,kH,kW]` →
    /// `[out,kH,kW,in]`, 3-D Conv1d `[out,in,k]` → `[out,k,in]`.
    static func loadWeights(into net: NFKMLXWhisperNet, from url: URL, remap: (String) -> String = { $0 },
                            precision: NFKMLXWeightPrecision = .checkpoint) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        let raw = checkpoint.arrays
        // A transformers export names the same tensors its own way; translate rather than reject.
        let huggingFace = raw.keys.contains { $0.hasPrefix("model.encoder.") || $0.hasPrefix("model.decoder.") }
        let mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            guard let name = huggingFace ? huggingFaceKey(key).map(remap) : remap(key) else { return nil }
            if checkpoint.needsConvTranspose, value.ndim == 4 { return (name, value.transposed(0, 2, 3, 1)) }
            if checkpoint.needsConvTranspose, value.ndim == 3 { return (name, value.transposed(0, 2, 1)) }
            return (name, value)
        }
        try NFKMLXWeights.apply(NFKMLXWeights.converted(mapped, to: precision), to: net)
    }

    /// The OpenAI-layout name a transformers Whisper tensor maps to, or nil for one with no
    /// counterpart (`proj_out` is the tied embedding stored again).
    ///
    /// @discussion transformers renames every module (`self_attn.q_proj` for `attn.query`, `fc1` for
    /// `mlp.0`, `embed_positions` for `positional_embedding`) without changing a single tensor, so a
    /// checkpoint exported from it is the same model under different keys. The mapping is asserted by
    /// renaming a real checkpoint into this form and getting the identical transcription back.
    static func huggingFaceKey(_ key: String) -> String? {
        if key == "proj_out.weight" { return nil }             // the tied embedding, stored twice
        var name = key
        if name.hasPrefix("model.") { name.removeFirst("model.".count) }
        name = name.replacingOccurrences(of: ".layers.", with: ".blocks.")
        name = name.replacingOccurrences(of: ".encoder_attn_layer_norm.", with: ".cross_attn_ln.")
        name = name.replacingOccurrences(of: ".self_attn_layer_norm.", with: ".attn_ln.")
        name = name.replacingOccurrences(of: ".final_layer_norm.", with: ".mlp_ln.")
        name = name.replacingOccurrences(of: ".encoder_attn.", with: ".cross_attn.")
        name = name.replacingOccurrences(of: ".self_attn.", with: ".attn.")
        name = name.replacingOccurrences(of: ".q_proj.", with: ".query.")
        name = name.replacingOccurrences(of: ".k_proj.", with: ".key.")
        name = name.replacingOccurrences(of: ".v_proj.", with: ".value.")
        name = name.replacingOccurrences(of: ".out_proj.", with: ".out.")
        name = name.replacingOccurrences(of: ".fc1.", with: ".mlp.0.")
        name = name.replacingOccurrences(of: ".fc2.", with: ".mlp.2.")
        name = name.replacingOccurrences(of: "decoder.embed_tokens.", with: "decoder.token_embedding.")
        if name == "encoder.embed_positions.weight" { return "encoder.positional_embedding" }
        if name == "decoder.embed_positions.weight" { return "decoder.positional_embedding" }
        name = name.replacingOccurrences(of: "encoder.layer_norm.", with: "encoder.ln_post.")
        name = name.replacingOccurrences(of: "decoder.layer_norm.", with: "decoder.ln.")
        return name
    }
}

private final class NFKMLXWhisperHolder: @unchecked Sendable {
    let net: NFKMLXWhisperNet
    init(_ net: NFKMLXWhisperNet) { self.net = net }
}
