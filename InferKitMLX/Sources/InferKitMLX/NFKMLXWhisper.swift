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

    /// The released `small` model.
    public static let small = NFKMLXWhisperConfiguration(state: 768, heads: 12, layers: 12)

    /// The released `medium` model.
    public static let medium = NFKMLXWhisperConfiguration(state: 1024, heads: 16, layers: 24)

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

    /// A size where the encoder and decoder share their width, head count, and depth, which every
    /// released Whisper does.
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
@objc(NFKMLXWhisperVariant)
public enum NFKMLXWhisperVariant: Int {
    case tiny
    case small
    case medium
    case largeV3
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
        _attnLN.wrappedValue = LayerNorm(dimensions: state)
        if cross {
            _crossAttn.wrappedValue = NFKWhisperAttention(state: state, heads: heads)
            _crossAttnLN.wrappedValue = LayerNorm(dimensions: state)
        }
        _mlp.wrappedValue = [Linear(state, state * 4), GELU(), Linear(state * 4, state)]
        _mlpLN.wrappedValue = LayerNorm(dimensions: state)
    }

    func callAsFunction(_ x: MLXArray, audio: MLXArray?, mask: MLXArray?) -> MLXArray {
        var h = x + attn(attnLN(x), source: nil, mask: mask)
        if let crossAttn, let crossAttnLN, let audio {
            h = h + crossAttn(crossAttnLN(h), source: audio, mask: nil)
        }
        let feed = (mlp[2] as! Linear)((mlp[1] as! GELU)((mlp[0] as! Linear)(mlpLN(h))))
        return h + feed
    }
}

/// The audio encoder: two convolutions, sinusoidal positions, and self-attention blocks.
final class NFKWhisperEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [NFKWhisperBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm

    init(_ c: NFKMLXWhisperConfiguration) {
        _conv1.wrappedValue = Conv1d(inputChannels: c.nMels, outputChannels: c.nAudioState, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: c.nAudioState, outputChannels: c.nAudioState, kernelSize: 3, stride: 2, padding: 1)
        _blocks.wrappedValue = (0 ..< c.nAudioLayer).map { _ in NFKWhisperBlock(state: c.nAudioState, heads: c.nAudioHead, cross: false) }
        _lnPost.wrappedValue = LayerNorm(dimensions: c.nAudioState)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = gelu(conv1(mel))
        x = gelu(conv2(x))
        x = x + NFKMLXWhisperNet.sinusoids(length: x.shape[1], channels: x.shape[2])
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
        _ln.wrappedValue = LayerNorm(dimensions: c.nTextState)
    }

    func callAsFunction(_ tokens: MLXArray, audio: MLXArray) -> MLXArray {
        let length = tokens.shape[1]
        var x = tokenEmbedding(tokens) + positionalEmbedding[0 ..< length]
        let mask = NFKMLXWhisperNet.causalMask(length)
        for block in blocks { x = block(x, audio: audio, mask: mask) }
        x = ln(x)
        return x.matmul(tokenEmbedding.weight.transposed(1, 0))
    }
}

/// The Whisper model: audio encoder + text decoder, with greedy transcription.
/// The Whisper encoder-decoder transformer.
///
/// Adapting it to a consumer's own domain works on this type directly: build one with
/// ``NFKMLXWhisper/network(weightsURL:configuration:)`` and train it with
/// ``NFKMLXWhisper/fineTune(_:examples:rank:alpha:objective:optimizer:steps:clipGradientNorm:checkpoint:observer:)``.
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

    static func causalMask(_ length: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: length * length)
        for i in 0 ..< length {
            for j in (i + 1) ..< length {
                values[i * length + j] = -1e9
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [length, length]) }
    }

    /// Greedy transcription: seeds the decode prompt, applies the decoder's suppression rules, and
    /// returns the generated token ids (prompt excluded).
    ///
    /// Two masks, as in the reference. The standing one covers the special and timestamp range plus
    /// `suppressTokens` — the non-speech symbols. The opening one additionally blocks a space and an
    /// immediate `<|endoftext|>`, and applies only to the first sampled position, which is the
    /// reference's `SuppressBlank`.
    func transcribe(_ mel: MLXArray) -> [Int] {
        let audio = encoder(mel)
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
        let audio = encoder(mel)
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

    private let net: NFKMLXWhisperNet
    private let tokenizer: NFKTokenizer?
    private let identifier: String

    init(net: NFKMLXWhisperNet, tokenizer: NFKTokenizer?, identifier: String = "whisper") {
        self.net = net
        self.tokenizer = tokenizer
        self.identifier = identifier
        super.init()
    }

    /// Whether a result carries the transcript's segment times beside its text.
    ///
    /// @discussion Off by default, which is the decode every other size and every earlier release of
    /// this backend performs. Turning it on changes the decode rather than the reading of one: the
    /// prompt drops `<|notimestamps|>` and the timestamp range stays unmasked, so the model is asked a
    /// different question and may answer it with different words.
    ///
    /// A timestamped result adds `NSArray<NFKAudioSegment *>` under `NFKOutputSegments`, each span
    /// labelled with the text inside it when a tokenizer is attached.
    ///
    /// Introduced in InferKit 0.1.0.
    @objc public var emitsTimestamps: Bool = false

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else {
            throw NFKMLXError.unsupportedInput
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
        guard emitsTimestamps else {
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
        var configuration = NFKMLXWhisperConfiguration()
        if let tokenizer {
            configuration.suppressTokens = NFKMLXWhisperSuppression.nonSpeechTokens(using: tokenizer)
        }
        let net = NFKMLXWhisperNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXWhisperHolder(net)
        let backend = NFKMLXWhisperBackend(net: holder.net, tokenizer: tokenizer, identifier: modelName)
        backend.emitsTimestamps = timestamps
        return backend
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
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

    /// Registers Whisper (`whisper-tiny`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a safetensors checkpoint, transposing convolution weights: 4-D `[out,in,kH,kW]` →
    /// `[out,kH,kW,in]`, 3-D Conv1d `[out,in,k]` → `[out,k,in]`.
    static func loadWeights(into net: NFKMLXWhisperNet, from url: URL, remap: (String) -> String = { $0 }) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        // A transformers export names the same tensors its own way; translate rather than reject.
        let huggingFace = raw.keys.contains { $0.hasPrefix("model.encoder.") || $0.hasPrefix("model.decoder.") }
        let mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            guard let name = huggingFace ? huggingFaceKey(key).map(remap) : remap(key) else { return nil }
            if checkpoint.needsConvTranspose, value.ndim == 4 { return (name, value.transposed(0, 2, 3, 1)) }
            if checkpoint.needsConvTranspose, value.ndim == 3 { return (name, value.transposed(0, 2, 1)) }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
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
