//
//  NFKMLXEmbeddingGemma.swift
//  InferKitMLX
//
//  EmbeddingGemma-300M: a second text embedder, and a second embedding architecture. Where
//  Qwen3-Embedding is the causal decoder read one layer earlier, EmbeddingGemma is a BIDIRECTIONAL
//  encoder — the Gemma 3 text backbone with the causal mask removed — mean-pooled over every token
//  and run through a Dense bottleneck.
//
//  The backbone is Gemma 3 (`gemma3_text`), NOT the causal Gemma 4 (`gemma4_text`) `NFKMLXGemmaLanguage`
//  implements. The two are different: Gemma 3 normalizes with `x · (1 + w)` where Gemma 4 uses `x · w`,
//  Gemma 3 turns a full-attention head's whole width where Gemma 4 turns a fraction, and Gemma 3 carries
//  no per-layer input embeddings. The blocks are the Gemma 3 decoder's (`NFKMLXGemma3.swift`); this
//  encoder runs them bidirectionally, with no key-value cache and no logit head.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

/// The geometry of the Gemma 3 text encoder EmbeddingGemma is built on.
///
/// @discussion The encoder is the Gemma 3 decoder (``NFKMLXGemma3Configuration``) read bidirectionally,
/// so this is a thin view over that configuration. `slidingWindow` is stated as the release states it,
/// the full span; the reference turns it into the exclusive bound `span / 2 + 1` on `|q - k|`, and
/// so does ``geometry``. EmbeddingGemma's parity query is far shorter than either number, which is
/// why the earlier reading of the span as the bound went unmeasured.
public struct NFKMLXGemma3EncoderConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    /// Rotary base for the full-attention layers.
    public var ropeTheta: Float
    /// Rotary base for the sliding-window layers, which the config states separately.
    public var ropeLocalTheta: Float
    /// The bidirectional span a sliding layer sees, as the release's `sliding_window` states it.
    public var slidingWindow: Int
    /// How the layers alternate. Every `slidingWindowPattern`th layer is full attention; the rest are
    /// sliding-window. EmbeddingGemma is bidirectional, so a sliding layer sees a symmetric window and a
    /// full layer sees everything.
    public var slidingWindowPattern: Int
    /// The queries are scaled by `queryPreAttnScalar ** -0.5` before attention.
    public var queryPreAttnScalar: Float
    public var rmsEpsilon: Float

    public init(hiddenSize: Int = 768, layerCount: Int = 24, headCount: Int = 3,
                keyValueHeadCount: Int = 1, headDimensions: Int = 256, intermediateSize: Int = 1152,
                vocabularySize: Int = 262_144, ropeTheta: Float = 1_000_000, ropeLocalTheta: Float = 10_000,
                slidingWindow: Int = 512, slidingWindowPattern: Int = 6, queryPreAttnScalar: Float = 256,
                rmsEpsilon: Float = 1e-6) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.ropeLocalTheta = ropeLocalTheta
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.queryPreAttnScalar = queryPreAttnScalar
        self.rmsEpsilon = rmsEpsilon
    }

    /// The released `embeddinggemma-300m` backbone geometry.
    public static let embeddingGemma300M = NFKMLXGemma3EncoderConfiguration()

    /// A small configuration that runs with random weights, for tests. Keeps the 6-layer pattern so a
    /// full-attention layer is exercised.
    public static let tiny = NFKMLXGemma3EncoderConfiguration(
        hiddenSize: 64, layerCount: 6, headCount: 2, keyValueHeadCount: 1, headDimensions: 32,
        intermediateSize: 128, vocabularySize: 512, ropeTheta: 1_000_000, ropeLocalTheta: 10_000,
        slidingWindow: 4, slidingWindowPattern: 6, queryPreAttnScalar: 32)

    /// Whether the layer at `index` is full attention (every `slidingWindowPattern`th, counting from 1).
    func isFullAttention(layer index: Int) -> Bool { (index + 1) % slidingWindowPattern == 0 }

    /// The decoder configuration this encoder is built from: the same geometry, bidirectional, with
    /// the window turned into the reference's exclusive bound.
    public var geometry: NFKMLXGemma3Configuration {
        NFKMLXGemma3Configuration(
            hiddenSize: hiddenSize, layerCount: layerCount, headCount: headCount,
            keyValueHeadCount: keyValueHeadCount, headDimensions: headDimensions,
            intermediateSize: intermediateSize, vocabularySize: vocabularySize, ropeTheta: ropeTheta,
            ropeLocalTheta: ropeLocalTheta, slidingWindow: slidingWindow / 2 + 1,
            slidingWindowPattern: slidingWindowPattern, queryPreAttnScalar: queryPreAttnScalar,
            rmsEpsilon: rmsEpsilon, isBidirectional: true)
    }
}

/// The Gemma 3 text encoder: a scaled token embedding, the sandwich-normalized blocks, and a final
/// normalization. Bidirectional, so a forward reads the whole sequence at once with no cache. The
/// blocks are the decoder's (`NFKGemma3Block`); only the masks differ.
public final class NFKMLXGemma3EncoderNet: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKGemma3Block]
    @ModuleInfo(key: "norm") var norm: NFKGemma3Norm

    let configuration: NFKMLXGemma3EncoderConfiguration
    private let geometry: NFKMLXGemma3Configuration
    private let embeddingScale: Float

    init(_ c: NFKMLXGemma3EncoderConfiguration) {
        configuration = c
        let geometry = c.geometry
        self.geometry = geometry
        embeddingScale = sqrt(Float(c.hiddenSize))
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layerCount).map {
            NFKGemma3Block(geometry, fullAttention: c.isFullAttention(layer: $0))
        }
        _norm.wrappedValue = NFKGemma3Norm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    /// The token hidden states `[1, length, hidden]`, post-final-norm.
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        layerStates(tokens).last!
    }

    /// The state entering the stack and the state each layer produces, the final norm applied to the
    /// last — the reference's `output_hidden_states` convention, so a divergence is located to a layer.
    func layerStates(_ tokens: MLXArray) -> [MLXArray] {
        var hidden = embedTokens(tokens) * embeddingScale
        var states = [hidden]
        let masks = NFKMLXGemma3Masks.make(length: tokens.shape[1], offset: 0, window: geometry.slidingWindow,
                                           blockIds: nil, bidirectional: true)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: configuration.isFullAttention(layer: index) ? nil : masks.sliding,
                           cache: nil, layer: index)
            states.append(hidden)
        }
        states[states.count - 1] = norm(hidden)
        return states
    }
}

/// The pooling, projection, and markers EmbeddingGemma applies over the encoder's hidden states.
public struct NFKMLXEmbeddingGemmaConfiguration: Sendable {
    /// A token prepended to the encoded ids (the beginning-of-sequence marker), or nil to prepend none.
    public var prependedToken: Int?
    /// A token appended after the encoded ids (the end-of-sequence marker), or nil to append none.
    public var appendedToken: Int?
    /// The Matryoshka width to truncate the embedding to before the final normalization, or nil for the
    /// full width. EmbeddingGemma is trained so 512, 256, and 128 are usable truncations of its 768.
    public var dimensions: Int?

    public init(prependedToken: Int? = 2, appendedToken: Int? = 1, dimensions: Int? = nil) {
        self.prependedToken = prependedToken
        self.appendedToken = appendedToken
        self.dimensions = dimensions
    }
}

/// EmbeddingGemma as a text embedder: the bidirectional encoder, mean pooling over every token, the
/// two Dense projections the sentence-transformers head carries, Matryoshka truncation, and L2
/// normalization.
final class NFKMLXEmbeddingGemmaEmbedder: NFKTextEmbedding {
    let net: NFKMLXGemma3EncoderNet
    /// The first Dense projection `[3072, 768]`, applied as `w · x` (no bias, Identity activation).
    let dense2: MLXArray
    /// The second Dense projection `[768, 3072]`.
    let dense3: MLXArray
    let configuration: NFKMLXEmbeddingGemmaConfiguration

    init(net: NFKMLXGemma3EncoderNet, dense2: MLXArray, dense3: MLXArray,
         configuration: NFKMLXEmbeddingGemmaConfiguration) {
        self.net = net
        self.dense2 = dense2
        self.dense3 = dense3
        self.configuration = configuration
    }

    var embeddingDimensions: Int { configuration.dimensions ?? dense3.dim(0) }

    func embed(tokens: [Int]) -> MLXArray {
        var ids = tokens
        if let prepended = configuration.prependedToken { ids.insert(prepended, at: 0) }
        if let appended = configuration.appendedToken { ids.append(appended) }
        let input = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let hidden = net(input)                          // [1, length, hidden]

        let pooled = hidden[0].mean(axis: 0)             // mean pooling over every token
        var projected = dense3.matmul(dense2.matmul(pooled))   // 768 -> 3072 -> 768, no bias

        if let dimensions = configuration.dimensions, dimensions < projected.dim(0) {
            projected = projected[0 ..< dimensions]
        }
        return projected / sqrt((projected * projected).sum())
    }
}

/// EmbeddingGemma as an InferKit backend, and its Objective-C factories.
///
/// `NFKMLXEmbeddingGemma` is the released `google/embeddinggemma-300m` (through the ungated
/// `unsloth/embeddinggemma-300m` mirror): the bidirectional Gemma 3 encoder, mean-pooled, projected
/// through a Dense bottleneck, and L2-normalized. A query carries a task prompt and a document a
/// different one; ``query(_:)`` and ``document(_:)`` build the two forms the model is trained on.
@objc(NFKMLXEmbeddingGemma)
public final class NFKMLXEmbeddingGemma: NSObject {

    /// A name for the backend the factories produce.
    @objc public static let modelName = "embeddinggemma-300m"

    /// Formats a retrieval query the way EmbeddingGemma is trained to read it.
    @objc public static func query(_ text: String) -> String { "task: search result | query: \(text)" }

    /// Formats a document the way EmbeddingGemma is trained to read it, with no title.
    @objc public static func document(_ text: String) -> String { "title: none | text: \(text)" }

    /// Loads the backbone and the two Dense projections from a release directory.
    static func loadWeights(into net: NFKMLXGemma3EncoderNet, fromDirectory directory: URL)
        throws -> (dense2: MLXArray, dense3: MLXArray) {
        try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: .float32)
        // The backbone's keys are the checkpoint's (no `model.` prefix), so nothing is remapped. The
        // Dense projections live in their own subdirectories, which `files(inDirectory:)` does not read.
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: .float32)
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
        return (try dense(directory, "2_Dense"), try dense(directory, "3_Dense"))
    }

    private static func dense(_ directory: URL, _ subdirectory: String) throws -> MLXArray {
        let url = directory.appendingPathComponent(subdirectory).appendingPathComponent("model.safetensors")
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        guard let weight = checkpoint.arrays["linear.weight"] else {
            throw NFKMLXError.malformedCheckpoint("\(subdirectory)/model.safetensors has no linear.weight")
        }
        return weight.asType(.float32)
    }

    /// Builds an embedder from optional local weights and a tokenizer, for a caller not loading a whole
    /// release directory. A nil `weightsURL` builds random weights (the pipeline runs, the embeddings
    /// are meaningless), and nil Dense weights build random projections.
    public static func backend(weightsURL: URL?, dense2URL: URL?, dense3URL: URL?,
                               tokenizer: NFKTokenizer?,
                               configuration: NFKMLXGemma3EncoderConfiguration = .embeddingGemma300M,
                               embedding: NFKMLXEmbeddingGemmaConfiguration = NFKMLXEmbeddingGemmaConfiguration())
        throws -> any NFKInferenceBackend {
        let net = NFKMLXGemma3EncoderNet(configuration)
        if let weightsURL {
            let mapped = try NFKMLXWeights.loadCheckpoint(url: weightsURL).arrays.map { ($0, $1) }
            try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
        }
        let dense2 = try loadDense(dense2URL, rows: 4 * configuration.hiddenSize, columns: configuration.hiddenSize)
        let dense3 = try loadDense(dense3URL, rows: configuration.hiddenSize, columns: 4 * configuration.hiddenSize)
        let embedder = NFKMLXEmbeddingGemmaEmbedder(net: net, dense2: dense2, dense3: dense3,
                                                    configuration: embedding)
        return NFKMLXTextEmbeddingBackend(embedder: embedder, tokenizer: tokenizer, identifier: modelName)
    }

    private static func loadDense(_ url: URL?, rows: Int, columns: Int) throws -> MLXArray {
        guard let url else { return MLXRandom.normal([rows, columns]) * 0.02 }
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        guard let weight = checkpoint.arrays["linear.weight"] else {
            throw NFKMLXError.malformedCheckpoint("\(url.lastPathComponent) has no linear.weight")
        }
        return weight.asType(.float32)
    }

    /// Builds an EmbeddingGemma backend from a downloaded release directory holding the backbone,
    /// `config.json`, the `2_Dense`/`3_Dense` projections, and the tokenizer files.
    ///
    /// A nil `dimensions` keeps the full 768-wide embedding; a smaller value truncates each embedding to
    /// that Matryoshka width before normalizing. Run inference off the render thread.
    public static func backend(directoryURL: URL, dimensions: Int? = nil)
        throws -> any NFKInferenceBackend {
        let net = NFKMLXGemma3EncoderNet(.embeddingGemma300M)
        let (dense2, dense3) = try loadWeights(into: net, fromDirectory: directoryURL)
        let embedding = NFKMLXEmbeddingGemmaConfiguration(dimensions: dimensions)
        let embedder = NFKMLXEmbeddingGemmaEmbedder(net: net, dense2: dense2, dense3: dense3,
                                                    configuration: embedding)
        let tokenizer = NFKMLXGemmaTokenizer(directoryURL: directoryURL)
        return NFKMLXTextEmbeddingBackend(embedder: embedder,
                                          tokenize: tokenizer.map { t in { t.encode($0) } },
                                          identifier: modelName)
    }

    /// The Objective-C entry: builds from a release directory. Run inference off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, dimensions: nil)
    }

    /// The Objective-C entry that truncates each embedding to a Matryoshka width (0 keeps the full width).
    @objc(backendWithDirectoryURL:outputDimensions:error:)
    public static func backend(directoryURL: URL, outputDimensions dimensions: Int)
        throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, dimensions: dimensions > 0 ? dimensions : nil)
    }

}

/// Gemma's tokenizer, read directly from a release's `tokenizer.json`.
///
/// @discussion Gemma's fast tokenizer is byte-fallback BPE with a metaspace normalizer: a space becomes
/// `▁`, the whole normalized string is one pre-token, and merges combine characters by rank. It is
/// neither the byte-level BPE the GPT-2/Qwen path uses (which maps bytes into a printable alphabet) nor
/// the unigram Viterbi `NFKUnigramTokenizer` runs (Gemma's `tokenizer.model` scores are merge ranks, so
/// a max-score path picks the wrong pieces), so it is its own reader. No offline conversion is needed;
/// the release's `tokenizer.json` is read as it ships. The file's `added_tokens` (`<bos>`,
/// `<start_of_turn>`, `<start_of_image>`, `<image_soft_token>`, …) are matched as literals before the
/// merge, as the reference matches them, so a rendered chat template or an image placeholder run
/// encodes to its ids rather than being spelled out in pieces.
final class NFKMLXGemmaTokenizer {
    private let vocabulary: [String: Int]
    /// The id-to-piece reverse table, for decoding; the added tokens are in it too.
    private let pieces: [Int: String]
    /// A merge `"left\u{0}right"` mapped to its rank; a lower rank is a higher merge priority.
    private let ranks: [String: Int]
    private let unknownId: Int
    /// The added tokens, matched as literals in the text before the merge.
    private let addedTokens: [String: Int]
    /// The ids of the added tokens flagged `special`, which a decode for display leaves out.
    let specialIds: Set<Int>

    /// The metaspace SentencePiece renders a space as.
    private static let metaspace = "\u{2581}"

    /// The id of a special token literal (`<bos>`, `<start_of_turn>`, `<image_soft_token>`), or nil
    /// when neither the vocabulary nor the added tokens carry it.
    func id(forToken content: String) -> Int? { addedTokens[content] ?? vocabulary[content] }

    /// The text a token-id sequence decodes to: each id's piece, with the metaspace turned back into a
    /// space and byte-fallback pieces (`<0xHH>`) reassembled into their bytes. `skipSpecial` leaves the
    /// special markers (`<eos>`, `<start_of_turn>`, …) out, which is what a displayed reply wants.
    func decode(_ ids: [Int], skipSpecial: Bool = false) -> String {
        var bytes = [UInt8]()
        for id in ids {
            if skipSpecial, specialIds.contains(id) { continue }
            guard let piece = pieces[id] else { continue }
            if piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
               let byte = UInt8(piece.dropFirst(3).dropLast(), radix: 16) {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: piece.replacingOccurrences(of: Self.metaspace, with: " ").utf8)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    convenience init?(directoryURL: URL) {
        self.init(tokenizerJSON: directoryURL.appendingPathComponent("tokenizer.json"))
    }

    init?(tokenizerJSON url: URL) {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any],
              let vocabulary = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any] else { return nil }
        self.vocabulary = vocabulary
        var pieces = [Int: String](minimumCapacity: vocabulary.count)
        for (piece, id) in vocabulary { pieces[id] = piece }
        var added = [String: Int]()
        var specials = Set<Int>()
        for entry in (json["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let content = entry["content"] as? String, let id = (entry["id"] as? NSNumber)?.intValue else { continue }
            added[content] = id
            pieces[id] = content
            if (entry["special"] as? NSNumber)?.boolValue ?? false { specials.insert(id) }
        }
        self.pieces = pieces
        addedTokens = added
        specialIds = specials
        var ranks = [String: Int](minimumCapacity: merges.count)
        for (index, entry) in merges.enumerated() {
            // A merge is `["left", "right"]` in a recent tokenizer.json and `"left right"` in an older one.
            if let pair = entry as? [String], pair.count == 2 {
                ranks[pair[0] + "\u{0}" + pair[1]] = index
            } else if let text = entry as? String, let space = text.firstIndex(of: " ") {
                ranks[String(text[..<space]) + "\u{0}" + String(text[text.index(after: space)...])] = index
            }
        }
        self.ranks = ranks
        unknownId = (model["unk_token"] as? String).flatMap { vocabulary[$0] } ?? 3
    }

    /// The token ids for `text`, with no markers added (the embedder wraps them in BOS and EOS, the
    /// generation backend prepends BOS). An added token written literally in the text encodes to its id.
    func encode(_ text: String) -> [Int] {
        var ids = [Int]()
        for segment in segments(of: text) {
            switch segment {
            case .special(let id): ids.append(id)
            case .text(let plain): ids += encodePlain(plain)
            }
        }
        return ids
    }

    private enum Segment {
        case special(Int)
        case text(String)
    }

    /// Splits the text at every added token written literally in it. Every added token is spelled
    /// `<…>`, so a candidate runs from a `<` to the next `>`, which keeps the scan linear.
    private func segments(of text: String) -> [Segment] {
        guard !addedTokens.isEmpty, text.contains("<") else { return [.text(text)] }
        var result = [Segment]()
        var plain = ""
        var index = text.startIndex
        while index < text.endIndex {
            if text[index] == "<", let close = text[index...].firstIndex(of: ">"),
               let id = addedTokens[String(text[index ... close])] {
                if !plain.isEmpty { result.append(.text(plain)); plain = "" }
                result.append(.special(id))
                index = text.index(after: close)
            } else {
                plain.append(text[index])
                index = text.index(after: index)
            }
        }
        if !plain.isEmpty { result.append(.text(plain)) }
        return result
    }

    private func encodePlain(_ text: String) -> [Int] {
        let normalized = text.replacingOccurrences(of: " ", with: Self.metaspace)
        // The space split the pre-tokenizer would do is a no-op after normalization, so the whole
        // string is one pre-token. Each character stands alone, or falls back to its UTF-8 bytes.
        var symbols = [String]()
        for scalar in normalized.unicodeScalars {
            let piece = String(scalar)
            if vocabulary[piece] != nil {
                symbols.append(piece)
            } else {
                for byte in Array(piece.utf8) { symbols.append(String(format: "<0x%02X>", byte)) }
            }
        }
        merge(&symbols)
        return symbols.map { vocabulary[$0] ?? unknownId }
    }

    /// The BPE merge loop: repeatedly merge the adjacent pair of highest priority (lowest rank) until
    /// none remains. A merged symbol is the concatenation of its parts, which the vocabulary carries.
    private func merge(_ symbols: inout [String]) {
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0 ..< (symbols.count - 1) {
                if let rank = ranks[symbols[index] + "\u{0}" + symbols[index + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            if bestIndex < 0 { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }
    }
}
