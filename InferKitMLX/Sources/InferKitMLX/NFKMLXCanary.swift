//
//  NFKMLXCanary.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

// Canary-1B-v2 (NVIDIA NeMo `EncDecMultiTaskModel`, CC-BY-4.0): a multitask speech model that
// transcribes and translates. Its acoustic front is a FastConformer encoder — the same layer as
// Parakeet's, here with biases (`attention_bias`, `convolution_bias`) — reused verbatim through
// `NFKParakeetEncoder`. Its head is an attention encoder-decoder: a Transformer decoder with
// self-attention, cross-attention into the encoder frames, and a ReLU feed-forward, generating the
// transcription token by token from a task prompt of control tokens (source language, target language,
// punctuation, timestamps, diarization). Ported from NeMo's own model on the released `.nemo`.

// MARK: - Configuration

public struct NFKMLXCanaryConfiguration: Sendable {
    // Front end + FastConformer encoder.
    public var sampleRate: Int = 16000
    public var mels: Int = 128
    public var dModel: Int = 1024
    public var encoderLayers: Int = 32
    public var encoderHeads: Int = 8
    public var feedForwardExpansion: Int = 4
    public var convKernel: Int = 9
    public var subsamplingChannels: Int = 256
    public var subsamplingFactor: Int = 8
    // Transformer decoder.
    public var decoderLayers: Int = 8
    public var decoderHeads: Int = 8
    public var decoderHeadDim: Int = 128
    public var decoderIntermediate: Int = 4096
    public var vocabulary: Int = 16384
    public var maxDecoderPositions: Int = 1024
    // Special tokens (from the release tokenizer).
    public var bos: Int = 4                 // <|startoftranscript|>
    public var eos: Int = 3                 // <|endoftext|>
    public var pad: Int = 2                 // <pad>
    public var startOfContext: Int = 7      // <|startofcontext|>
    public var maxDecodeTokens: Int = 512

    public init() {}

    /// The released `nvidia/canary-1b-v2`.
    public static let v2 = NFKMLXCanaryConfiguration()

    /// A shrunk geometry for shape tests.
    public static var tiny: NFKMLXCanaryConfiguration {
        var c = NFKMLXCanaryConfiguration()
        c.mels = 16; c.dModel = 32; c.encoderLayers = 2; c.encoderHeads = 2; c.convKernel = 5
        c.subsamplingChannels = 8; c.decoderLayers = 2; c.decoderHeads = 2; c.decoderHeadDim = 16
        c.decoderIntermediate = 64; c.vocabulary = 32; c.maxDecoderPositions = 128; c.maxDecodeTokens = 8
        return c
    }

    /// The FastConformer encoder shares Parakeet's implementation, with biases enabled.
    var encoderConfiguration: NFKMLXParakeetConfiguration {
        var c = NFKMLXParakeetConfiguration()
        c.sampleRate = sampleRate; c.mels = mels; c.dModel = dModel; c.layers = encoderLayers
        c.heads = encoderHeads; c.feedForwardExpansion = feedForwardExpansion; c.convKernel = convKernel
        c.subsamplingChannels = subsamplingChannels; c.subsamplingFactor = subsamplingFactor
        c.useBias = true
        return c
    }
}

// MARK: - Decoder

/// Multi-head attention for the Transformer decoder, biased throughout. Self-attention is causal and
/// reads the decoder states; cross-attention reads the decoder states as queries and the encoder frames
/// as keys and values. The decoder recomputes over the whole growing sequence each greedy step, so the
/// causal mask is applied additively rather than through a cache.
final class NFKCanaryAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    let heads: Int
    let headDim: Int

    init(_ config: NFKMLXCanaryConfiguration) {
        heads = config.decoderHeads
        headDim = config.decoderHeadDim
        let d = config.dModel, inner = heads * headDim
        _q.wrappedValue = Linear(d, inner, bias: true)
        _k.wrappedValue = Linear(d, inner, bias: true)
        _v.wrappedValue = Linear(d, inner, bias: true)
        _o.wrappedValue = Linear(inner, d, bias: true)
    }

    /// `query` `[1, Tq, D]`, `keyValue` `[1, Tk, D]`; `causal` adds the upper-triangular −inf mask.
    func callAsFunction(query: MLXArray, keyValue: MLXArray, causal: Bool) -> MLXArray {
        let (b, tq) = (query.dim(0), query.dim(1))
        let tk = keyValue.dim(1)
        func split(_ x: MLXArray, _ t: Int) -> MLXArray {
            x.reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        }
        let qh = split(q(query), tq)
        let kh = split(k(keyValue), tk)
        let vh = split(v(keyValue), tk)
        var scores = matmul(qh, kh.transposed(0, 1, 3, 2)) / sqrt(Float(headDim))   // [b, h, Tq, Tk]
        if causal {
            let mask = MLXArray(0 ..< Int32(tq)).reshaped([tq, 1]) .< MLXArray(0 ..< Int32(tk)).reshaped([1, tk])
            scores = MLX.where(mask, MLXArray(-Float.greatestFiniteMagnitude), scores)
        }
        let attended = matmul(softmax(scores, axis: -1), vh)                        // [b, h, Tq, dk]
        return o(attended.transposed(0, 2, 1, 3).reshaped([b, tq, heads * headDim]))
    }
}

/// One decoder block: pre-norm self-attention, pre-norm cross-attention into the encoder, pre-norm
/// ReLU feed-forward, each through a residual.
final class NFKCanaryDecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var selfNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: NFKCanaryAttention
    @ModuleInfo(key: "post_attention_layernorm") var crossNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var crossAttn: NFKCanaryAttention
    @ModuleInfo(key: "final_layernorm") var mlpNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: NFKMLXCanaryConfiguration) {
        let d = config.dModel
        _selfNorm.wrappedValue = LayerNorm(dimensions: d)
        _selfAttn.wrappedValue = NFKCanaryAttention(config)
        _crossNorm.wrappedValue = LayerNorm(dimensions: d)
        _crossAttn.wrappedValue = NFKCanaryAttention(config)
        _mlpNorm.wrappedValue = LayerNorm(dimensions: d)
        _fc1.wrappedValue = Linear(d, config.decoderIntermediate, bias: true)
        _fc2.wrappedValue = Linear(config.decoderIntermediate, d, bias: true)
    }

    func callAsFunction(_ x: MLXArray, encoder: MLXArray) -> MLXArray {
        var h = x + selfAttn(query: selfNorm(x), keyValue: selfNorm(x), causal: true)
        h = h + crossAttn(query: crossNorm(h), keyValue: encoder, causal: false)
        h = h + fc2(relu(fc1(mlpNorm(h))))
        return h
    }
}

/// Holds the fixed positional table off the module graph, so it stays out of `parameters()` (the
/// reference computes it rather than storing it, and a stored `MLXArray` property would look like a
/// weight the checkpoint must cover).
final class NFKCanaryPositions: @unchecked Sendable {
    let table: MLXArray
    init(length: Int, channels: Int) { table = NFKCanaryDecoder.sinusoids(length: length, channels: channels) }
}

/// The Transformer decoder: a token embedding, fixed sinusoidal positions, an embedding LayerNorm, the
/// block stack, a final norm, and a tied output projection.
final class NFKCanaryDecoder: Module {
    @ModuleInfo(key: "embed_tokens") var embed: Embedding
    @ModuleInfo(key: "embedding_layernorm") var embedNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [NFKCanaryDecoderLayer]
    @ModuleInfo(key: "norm") var norm: LayerNorm
    let dModel: Int
    let maxPositions: Int
    /// `<|startoftranscript|>` etc. carry no position offset relative to the reference; the fixed table
    /// is built once at the released maximum and sliced per call.
    private let positions: NFKCanaryPositions

    init(_ config: NFKMLXCanaryConfiguration) {
        dModel = config.dModel
        maxPositions = config.maxDecoderPositions
        _embed.wrappedValue = Embedding(embeddingCount: config.vocabulary, dimensions: config.dModel)
        _embedNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        _layers.wrappedValue = (0 ..< config.decoderLayers).map { _ in NFKCanaryDecoderLayer(config) }
        _norm.wrappedValue = LayerNorm(dimensions: config.dModel)
        positions = NFKCanaryPositions(length: config.maxDecoderPositions, channels: config.dModel)
    }

    /// The NeMo `FixedPositionalEncoding`: `sin`/`cos` interleaved, `pos / 10000^(2i/d)`, the whole
    /// table scaled by `1/sqrt(d)` (which the reference adds to the unscaled token embedding).
    static func sinusoids(length: Int, channels: Int) -> MLXArray {
        let scale = 1 / sqrtf(Float(channels))
        var table = [Float](repeating: 0, count: length * channels)
        for pos in 0 ..< length {
            for i in stride(from: 0, to: channels, by: 2) {
                let div = expf(Float(i) * -(logf(10000) / Float(channels)))
                table[pos * channels + i] = sinf(Float(pos) * div) * scale
                if i + 1 < channels { table[pos * channels + i + 1] = cosf(Float(pos) * div) * scale }
            }
        }
        return MLXArray(table).reshaped([length, channels])
    }

    /// `tokens` `[1, T]` decoder ids, `encoder` `[1, Te, D]` → hidden states `[1, T, D]`.
    func hidden(tokens: MLXArray, encoder: MLXArray) -> MLXArray {
        let t = tokens.dim(1)
        var h = embed(tokens) + positions.table[0 ..< t].reshaped([1, t, dModel])
        h = embedNorm(h)
        for layer in layers { h = layer(h, encoder: encoder) }
        return norm(h)
    }
}

// MARK: - Tokenizer

/// The release's `tokenizer.json`, read for the two things recognition needs: building the task prompt
/// from named control tokens, and decoding the generated ids to text. The model is a Metaspace BPE
/// (SentencePiece-style: the `▁` word marker, `byte_fallback`), so decoding concatenates the pieces,
/// turns `▁` into a space, and drops a leading space — and control tokens (`<|...|>`, `<pad>`, `<unk>`)
/// are dropped from the text.
public struct NFKMLXCanaryTokenizer: Sendable {
    /// The piece for each id, in id order.
    public let pieces: [String]
    /// The ids the tokenizer marks special (control tokens), dropped from decoded text.
    public let specialIds: Set<Int>
    private let byContent: [String: Int]

    public init(tokenizerURL: URL) throws {
        let data = try Data(contentsOf: tokenizerURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int] else {
            throw NFKMLXError.malformedCheckpoint("canary tokenizer.json is not a BPE model with a vocab")
        }
        var pieces = [String](repeating: "", count: vocab.count)
        var byContent = [String: Int]()
        for (piece, id) in vocab {
            if id >= 0 && id < pieces.count { pieces[id] = piece }
            byContent[piece] = id
        }
        var specialIds = Set<Int>()
        if let added = root["added_tokens"] as? [[String: Any]] {
            for token in added {
                guard let id = token["id"] as? Int, let content = token["content"] as? String else { continue }
                byContent[content] = id
                if id >= 0 && id < pieces.count { pieces[id] = content }
                if (token["special"] as? Bool) == true { specialIds.insert(id) }
            }
        }
        self.pieces = pieces
        self.specialIds = specialIds
        self.byContent = byContent
    }

    private init(pieces: [String], specialIds: Set<Int>, byContent: [String: Int]) {
        self.pieces = pieces
        self.specialIds = specialIds
        self.byContent = byContent
    }

    /// An empty tokenizer for the random-weights shape-test backend, which never decodes.
    public static let empty = NFKMLXCanaryTokenizer(pieces: [], specialIds: [], byContent: [:])

    /// The id of a named control token (`<|startoftranscript|>`, `<|en|>`, …), or nil.
    public func id(of content: String) -> Int? { byContent[content] }

    /// The transcription prompt: `<|startofcontext|><|startoftranscript|><|emo:undefined|>` then the
    /// source and target language, punctuation, and the off switches for inverse text normalization,
    /// timestamps, and diarization — the release's own chat template for an ASR turn.
    public func transcriptionPrompt(source: String, target: String, punctuation: Bool) -> [Int] {
        let names = ["<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
                     "<|\(source)|>", "<|\(target)|>", punctuation ? "<|pnc|>" : "<|nopnc|>",
                     "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"]
        return names.compactMap { byContent[$0] }
    }

    /// Concatenate the pieces of the non-special ids, turn the `▁` marker into a space, drop a leading one.
    public func text(for ids: [Int]) -> String {
        let joined = ids.filter { !specialIds.contains($0) }
            .compactMap { $0 >= 0 && $0 < pieces.count ? pieces[$0] : nil }.joined()
        let spaced = joined.replacingOccurrences(of: "\u{2581}", with: " ")
        return spaced.hasPrefix(" ") ? String(spaced.dropFirst()) : spaced
    }
}

// MARK: - Model

public final class NFKMLXCanaryNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKParakeetEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKCanaryDecoder
    @ModuleInfo(key: "proj_out") var projOut: Linear
    let frontEnd: NFKParakeetFrontEnd
    public let configuration: NFKMLXCanaryConfiguration

    public init(_ configuration: NFKMLXCanaryConfiguration) {
        self.configuration = configuration
        frontEnd = NFKParakeetFrontEnd(configuration.encoderConfiguration)
        _encoder.wrappedValue = NFKParakeetEncoder(configuration.encoderConfiguration)
        _decoder.wrappedValue = NFKCanaryDecoder(configuration)
        _projOut.wrappedValue = Linear(configuration.dModel, configuration.vocabulary, bias: true)
    }

    /// `[1, frames, mels]` normalized features → encoder frames `[1, T, D]`.
    public func encode(features: MLXArray) -> MLXArray { encoder(features) }

    /// Logits `[1, T, V]` over the decoder ids given the encoder frames.
    public func logits(tokens: MLXArray, encoder encoded: MLXArray) -> MLXArray {
        projOut(decoder.hidden(tokens: tokens, encoder: encoded))
    }

    /// Greedy autoregressive decoding: the prompt seeds the decoder, then each step appends the argmax of
    /// the last position's logits until `<|endoftext|>` or the position bound. Recomputes over the whole
    /// sequence each step (the clip is short), which reproduces the reference's teacher-forced logits.
    public func decode(encoded: MLXArray, prompt: [Int]) -> [Int] {
        var ids = prompt.map { Int32($0) }
        var generated = [Int]()
        let bound = min(configuration.maxDecodeTokens, configuration.maxDecoderPositions - prompt.count)
        for _ in 0 ..< bound {
            let tokens = MLXArray(ids).reshaped([1, ids.count])
            let step = logits(tokens: tokens, encoder: encoded)[0, ids.count - 1, 0...]
            eval(step)
            let values = step.asArray(Float.self)
            var best = 0
            for i in 1 ..< values.count where values[i] > values[best] { best = i }
            if best == configuration.eos { break }
            generated.append(best)
            ids.append(Int32(best))
        }
        return generated
    }

    /// Waveform (16 kHz mono, `-1...1`) → the generated token ids for the given prompt.
    public func recognize(_ samples: [Float], prompt: [Int]) -> [Int] {
        decode(encoded: encode(features: frontEnd.features(samples)), prompt: prompt)
    }
}

// MARK: - Loading

public final class NFKMLXCanary: NSObject {
    /// The registry name.
    @objc public static let modelName = "canary-1b-v2"

    /// Renames NeMo's `EncDecMultiTaskModel` keys onto the module. The FastConformer encoder keys match
    /// `NFKParakeetEncoder` already; the Transformer decoder is `transf_decoder._decoder.layers.N` with
    /// three sub-layers (self-attention `first_sub_layer`, cross-attention `second_sub_layer`,
    /// feed-forward `third_sub_layer`), the embedding under `_embedding`, and the output head under
    /// `log_softmax.mlp.layer0`. The fixed position table is dropped — the decoder computes the sinusoids.
    /// Returns nil for a key to drop.
    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasSuffix("num_batches_tracked") || key.hasPrefix("preprocessor.") { return nil }
        if key == "transf_decoder._embedding.position_embedding.pos_enc" { return nil }
        if key.hasPrefix("encoder.") { return key }
        var name = key
        name = name.replacingOccurrences(of: "transf_decoder._embedding.token_embedding.", with: "decoder.embed_tokens.")
        name = name.replacingOccurrences(of: "transf_decoder._embedding.layer_norm.", with: "decoder.embedding_layernorm.")
        name = name.replacingOccurrences(of: "transf_decoder._decoder.final_layer_norm.", with: "decoder.norm.")
        name = name.replacingOccurrences(of: "transf_decoder._decoder.layers.", with: "decoder.layers.")
        name = name.replacingOccurrences(of: ".layer_norm_1.", with: ".input_layernorm.")
        name = name.replacingOccurrences(of: ".layer_norm_2.", with: ".post_attention_layernorm.")
        name = name.replacingOccurrences(of: ".layer_norm_3.", with: ".final_layernorm.")
        name = name.replacingOccurrences(of: ".first_sub_layer.", with: ".self_attn.")
        name = name.replacingOccurrences(of: ".second_sub_layer.", with: ".encoder_attn.")
        name = name.replacingOccurrences(of: ".third_sub_layer.dense_in.", with: ".fc1.")
        name = name.replacingOccurrences(of: ".third_sub_layer.dense_out.", with: ".fc2.")
        name = name.replacingOccurrences(of: ".query_net.", with: ".q_proj.")
        name = name.replacingOccurrences(of: ".key_net.", with: ".k_proj.")
        name = name.replacingOccurrences(of: ".value_net.", with: ".v_proj.")
        name = name.replacingOccurrences(of: ".out_projection.", with: ".o_proj.")
        name = name.replacingOccurrences(of: "log_softmax.mlp.layer0.", with: "proj_out.")
        return name
    }

    /// Loads the released `model_weights.ckpt` into `net`, feeding the preprocessor's stored window and
    /// filterbank to the front end and transposing the convolutions to MLX's channels-last layouts.
    public static func loadWeights(into net: NFKMLXCanaryNet, from url: URL) throws {
        try loadWeights(into: net, checkpoint: try NFKMLXWeights.loadCheckpoint(url: url))
    }

    static func loadWeights(into net: NFKMLXCanaryNet, checkpoint: NFKMLXWeights.Checkpoint) throws {
        let arrays = checkpoint.arrays
        if let window = arrays["preprocessor.featurizer.window"] { net.frontEnd.load(window: window) }
        if let filterbank = arrays["preprocessor.featurizer.fb"] { net.frontEnd.load(filterbank: filterbank) }
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
        try NFKMLXWeights.apply(mapped, to: net)
        net.train(false)
    }
}
