//
//  NFKMLXTextEmbedder.swift
//  InferKitMLX
//
//  Text embeddings through MLX. The package embedded images (CLIP) and had no way to embed text, so
//  there was no semantic search, retrieval, clustering, or reranking over a consumer's own corpus.
//
//  A modern text embedder is the decoder-only language model with its output projection removed: the
//  post-final-norm hidden states are pooled to one vector and L2-normalized, so a dot product is a
//  cosine similarity. `NFKMLXLanguageNet` already exposes those states through `hiddenStates(fromEmbeddings:)`,
//  so Qwen3-Embedding is the Qwen3 dense decoder this package already runs, read a layer earlier.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// How a sequence of token hidden states collapses to one embedding vector.
public enum NFKMLXTextEmbeddingPooling: Sendable {
    /// The hidden state of the final token. Qwen3-Embedding is trained this way, with an end token
    /// appended (see ``NFKMLXTextEmbedderConfiguration/appendedToken``) so the pooled position has
    /// attended to the whole sequence.
    case lastToken
    /// The mean of every token's hidden state. EmbeddingGemma and the encoder families pool this way.
    case mean
}

/// The pooling and post-processing a text embedder applies over a language model's hidden states.
///
/// @discussion The geometry of the model itself is `NFKMLXLanguageConfiguration`; this carries only
/// what turns its hidden states into an embedding. A released embedder's card names each choice: the
/// pooling mode, whether an end token is appended before encoding, whether the result is normalized,
/// and the Matryoshka dimension a leading slice is a usable embedding at.
public struct NFKMLXTextEmbedderConfiguration: Sendable {
    /// How the token hidden states collapse to one vector.
    public var pooling: NFKMLXTextEmbeddingPooling
    /// A token id appended to every input before encoding, or nil to append none. Qwen3-Embedding
    /// appends `<|endoftext|>`, whose hidden state is what last-token pooling reads.
    public var appendedToken: Int?
    /// Whether the pooled vector is L2-normalized, so a dot product between two embeddings is their
    /// cosine similarity.
    public var normalizes: Bool
    /// The Matryoshka dimension to truncate the embedding to before normalizing, or nil for the full
    /// width. Qwen3-Embedding is trained so a leading slice of the embedding is itself usable, which
    /// is what lets a caller trade accuracy for storage without a second model.
    public var dimensions: Int?

    public init(pooling: NFKMLXTextEmbeddingPooling, appendedToken: Int? = nil,
                normalizes: Bool = true, dimensions: Int? = nil) {
        self.pooling = pooling
        self.appendedToken = appendedToken
        self.normalizes = normalizes
        self.dimensions = dimensions
    }

    /// The `Qwen/Qwen3-Embedding-0.6B` recipe: last-token pooling with `<|endoftext|>` appended, and
    /// L2 normalization. `appendedToken` defaults to `<|endoftext|>`'s id; a release directory reads
    /// its own from the tokenizer.
    public static func qwen3Embedding(appendedToken: Int = 151_643,
                                      dimensions: Int? = nil) -> NFKMLXTextEmbedderConfiguration {
        NFKMLXTextEmbedderConfiguration(pooling: .lastToken, appendedToken: appendedToken,
                                        normalizes: true, dimensions: dimensions)
    }
}

/// A model that turns a token sequence into one embedding vector. Backs ``NFKMLXTextEmbeddingBackend``,
/// so a family with its own pooling and projection (EmbeddingGemma's mean pooling and Dense bottleneck)
/// serves the same backend as the last-token pooled decoder here.
protocol NFKTextEmbedding: AnyObject {
    /// The embedding of one token sequence.
    func embed(tokens: [Int]) -> MLXArray
    /// How wide the embedding is, after any Matryoshka truncation.
    var embeddingDimensions: Int { get }
}

/// A text embedder: a decoder-only language model pooled and normalized into one embedding per input.
///
/// The model is `NFKMLXLanguageNet` read through its `hiddenStates(fromEmbeddings:)` seam, so nothing
/// about the transformer is re-implemented here.
final class NFKMLXTextEmbedder: NFKTextEmbedding {

    var embeddingDimensions: Int { configuration.dimensions ?? net.configuration.hiddenSize }

    let net: NFKMLXLanguageNet
    let configuration: NFKMLXTextEmbedderConfiguration

    init(net: NFKMLXLanguageNet, configuration: NFKMLXTextEmbedderConfiguration) {
        self.net = net
        self.configuration = configuration
    }

    /// The embedding of one token sequence, appended token added, pooled, truncated, and normalized as
    /// the configuration asks.
    func embed(tokens: [Int]) -> MLXArray {
        var ids = tokens
        if let appended = configuration.appendedToken { ids.append(appended) }
        let input = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let hidden = net.hiddenStates(fromEmbeddings: net.embed(input))     // [1, length, hidden]

        var pooled: MLXArray
        switch configuration.pooling {
        case .lastToken:
            // Batch of one and no padding, so the final position is the appended token's.
            pooled = hidden[0, ids.count - 1]
        case .mean:
            pooled = hidden[0].mean(axis: 0)
        }

        if let dimensions = configuration.dimensions, dimensions < pooled.dim(0) {
            pooled = pooled[0 ..< dimensions]
        }
        if configuration.normalizes {
            pooled = pooled / sqrt((pooled * pooled).sum())
        }
        return pooled
    }
}

/// Holds the embedder for capture in the backend's `@Sendable` closures. The tokenizer is a plain
/// closure rather than a type so a family with its own tokenizer (EmbeddingGemma's byte-fallback BPE)
/// plugs in beside the core `NFKTokenizer` the decoder embedder uses; the holder's `@unchecked
/// Sendable` is what lets a non-`Sendable` tokenizer be captured across the generation task.
private final class NFKTextEmbedderHolder: @unchecked Sendable {
    let embedder: any NFKTextEmbedding
    let tokenize: ((String) -> [Int])?
    init(_ embedder: any NFKTextEmbedding, _ tokenize: ((String) -> [Int])?) {
        self.embedder = embedder
        self.tokenize = tokenize
    }
}

/// A text-embedding backend. Text under `NFKInputPrompt` (or a message sequence under
/// `NFKInputMessages`, joined) encodes to an embedding under `NFKOutputEmbedding`.
///
/// @discussion The embedding is the pooled, normalized hidden state of the language model the backend
/// wraps. A caller that has token ids already, or that runs its own tokenizer, uses ``embedding(forTokens:)``;
/// the request path needs the tokenizer the backend was built with. Run inference off the render thread.
@objc(NFKMLXTextEmbeddingBackend)
public final class NFKMLXTextEmbeddingBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKTextEmbedderHolder
    private let identifier: String

    /// Builds a backend over an embedder and a tokenizer.
    ///
    /// The `tokenize` closure turns the request's text into token ids; it is nil when the caller drives
    /// ``embedding(forTokens:)`` with ids it produced itself. A core `NFKTokenizer` adapts with
    /// `{ tokenizer.encode($0).map(\.intValue) }`.
    init(embedder: any NFKTextEmbedding, tokenize: ((String) -> [Int])?, identifier: String) {
        holder = NFKTextEmbedderHolder(embedder, tokenize)
        self.identifier = identifier
        super.init()
    }

    /// Builds a backend whose text path runs a core `NFKTokenizer`.
    convenience init(embedder: any NFKTextEmbedding, tokenizer: NFKTokenizer?, identifier: String) {
        self.init(embedder: embedder,
                  tokenize: tokenizer.map { t in { t.encode($0).map(\.intValue) } },
                  identifier: identifier)
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    /// How wide an embedding this backend produces, after any Matryoshka truncation.
    @objc public var embeddingDimensions: Int { holder.embedder.embeddingDimensions }

    /// The embedding of a token sequence the caller tokenized, as floats. The appended end token,
    /// pooling, truncation, and normalization the backend was configured with all apply.
    @objc(embeddingForTokens:)
    public func embedding(forTokens tokens: [NSNumber]) -> [NSNumber] {
        Self.numbers(holder.embedder.embed(tokens: tokens.map(\.intValue)))
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result {
            return result
        }
        if let error = job.error {
            throw error
        }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = self.holder
        Task.detached(priority: .userInitiated) {
            do {
                guard let text = Self.text(from: request) else {
                    throw NFKMLXError.unsupportedInput
                }
                guard let tokenize = holder.tokenize else {
                    throw NFKMLXError.unsupportedConfiguration(
                        "text embedding needs a tokenizer; build the backend with one")
                }
                let embedding = Self.numbers(holder.embedder.embed(tokens: tokenize(text)))
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputEmbedding: embedding]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    /// The text to embed: the prompt, or a message sequence joined by newlines. Shares the language
    /// backend's reader so the two agree on how a request carries text.
    private static func text(from request: NFKInferenceRequest) -> String? {
        NFKMLXLanguageBackend.prompt(from: request)
    }

    private static func numbers(_ embedding: MLXArray) -> [NSNumber] {
        eval(embedding)
        return embedding.asArray(Float.self).map { NSNumber(value: $0) }
    }
}

/// Qwen3-Embedding as an InferKit backend, and its Objective-C factories.
///
/// `NFKMLXQwen3Embedding` is the released `Qwen/Qwen3-Embedding-0.6B`: the Qwen3-0.6B dense decoder with
/// last-token pooling over an appended `<|endoftext|>` and L2 normalization. The backbone loads through
/// the same release reader `NFKMLXLanguage` uses, so nothing about the transformer or its checkpoint is
/// duplicated here.
///
/// A query is embedded with a one-sentence instruction prepended and a document without one; the two are
/// asymmetric on purpose, and ``instruct(task:query:)`` builds the query form the model was trained on.
@objc(NFKMLXQwen3Embedding)
public final class NFKMLXQwen3Embedding: NSObject {

    /// A name for the backend the factories produce.
    @objc public static let modelName = "qwen3-embedding-0.6b"

    /// Formats a retrieval query the way Qwen3-Embedding is trained to read it: a one-sentence task
    /// description, then the query. A document is embedded as-is, with no instruction.
    @objc public static func instruct(task: String, query: String) -> String {
        "Instruct: \(task)\nQuery:\(query)"
    }

    /// Builds an embedder from optional single-file weights and a tokenizer, for a caller who is not
    /// loading a whole release directory.
    ///
    /// A nil `weightsURL` builds random weights: the pipeline runs and the embeddings are meaningless,
    /// which is what a smoke test or an example uses to exercise the path without a download. The
    /// `configuration` is the backbone's geometry and `embedding` is the pooling recipe; the defaults
    /// are Qwen3-Embedding-0.6B's. Without a tokenizer the request path cannot encode text, so a caller
    /// tokenizes elsewhere and uses ``NFKMLXTextEmbeddingBackend/embedding(forTokens:)``.
    public static func backend(weightsURL: URL?, tokenizer: NFKTokenizer?,
                               configuration: NFKMLXLanguageConfiguration = .qwen3_0_6B,
                               embedding: NFKMLXTextEmbedderConfiguration = .qwen3Embedding())
        throws -> any NFKInferenceBackend {
        let net = NFKMLXLanguage.makeNet(configuration)
        if let weightsURL {
            try NFKMLXLanguage.loadWeights(into: net, from: weightsURL)
        }
        let embedder = NFKMLXTextEmbedder(net: net, configuration: embedding)
        return NFKMLXTextEmbeddingBackend(embedder: embedder, tokenizer: tokenizer, identifier: modelName)
    }

    /// Loads a Qwen3-Embedding release's weights into the backbone.
    ///
    /// @discussion The released checkpoint is the BASE model (`AutoModel`/`Qwen3Model`), so its keys
    /// carry no `model.` prefix (`embed_tokens.weight`, `layers.N.…`, `norm.weight`) and no `lm_head`.
    /// The decoder module keeps the causal-LM layout, so the loader prepends `model.` and drops any
    /// output projection the base checkpoint does not have. The weights are bf16; a float32 load upcasts
    /// them exactly.
    static func loadWeights(into net: NFKMLXLanguageNet, fromDirectory directory: URL) throws {
        try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: .float32)
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: .float32) { key in
            if key.hasPrefix("lm_head.") { return nil }
            return key.hasPrefix("model.") ? key : "model.\(key)"
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    /// Builds a Qwen3-Embedding backend from a downloaded release directory holding the weights,
    /// `config.json`, and the tokenizer files.
    ///
    /// A nil `dimensions` keeps the full 1024-wide embedding; a smaller value truncates each embedding
    /// to that Matryoshka width before normalizing. Run inference off the render thread.
    public static func backend(directoryURL: URL, dimensions: Int? = nil)
        throws -> any NFKInferenceBackend {
        let configuration = try NFKMLXLanguage.configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        let net = NFKMLXLanguage.makeNet(configuration)
        try loadWeights(into: net, fromDirectory: directoryURL)
        let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL)
        // Qwen3-Embedding's tokenizer appends `<|endoftext|>` (not the chat `eos_token`), whose hidden
        // state is what last-token pooling reads; the id is in the release's own special tokens.
        let (specials, _) = NFKMLXLanguage.specialTokens(inDirectory: directoryURL)
        let appended = specials["<|endoftext|>"] ?? 151_643
        let embeddingConfiguration = NFKMLXTextEmbedderConfiguration.qwen3Embedding(
            appendedToken: appended, dimensions: dimensions)
        let embedder = NFKMLXTextEmbedder(net: net, configuration: embeddingConfiguration)
        return NFKMLXTextEmbeddingBackend(embedder: embedder, tokenizer: tokenizer,
                                          identifier: modelName)
    }

    /// The Objective-C entry: builds from a release directory, reading its `config.json`, tokenizer,
    /// and weights. Run inference off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, dimensions: nil)
    }

    /// The Objective-C entry that truncates each embedding to a Matryoshka width. A `dimensions` of 0
    /// keeps the full width.
    @objc(backendWithDirectoryURL:outputDimensions:error:)
    public static func backend(directoryURL: URL, outputDimensions dimensions: Int)
        throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, dimensions: dimensions > 0 ? dimensions : nil)
    }
}
