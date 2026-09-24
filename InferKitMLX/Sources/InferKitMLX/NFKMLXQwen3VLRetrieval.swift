//
//  NFKMLXQwen3VLRetrieval.swift
//  InferKitMLX
//
//  Qwen3-VL as a retrieval pair: an embedder and a reranker over text, images, and the two together.
//  The package embedded text with a text-only decoder and reranked with a text-only cross-encoder, so
//  a corpus of images could be searched only through CLIP's shared space, which carries no instruction
//  and no document text.
//
//  Both releases are the Qwen3-VL backbone this package already runs, read through its hidden-state
//  seam: the embedder pools the last position and normalizes, and the reranker reads the same position
//  through the output projection and takes one logit difference.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN

/// Building the prompt both Qwen3-VL retrieval models are trained to read.
///
/// The releases carry a chat template, and the reference scripts drive it with a system turn holding
/// the instruction and a user turn holding the content. The template writes one `<|image_pad|>` per
/// image and the processor expands it to one token per merged patch, which is what this builds
/// directly.
enum NFKQwen3VLRetrievalPrompt {

    /// The chat turn markers the template emits.
    static let systemOpen = "<|im_start|>system\n"
    static let userOpen = "<|im_start|>user\n"
    static let turnEnd = "<|im_end|>\n"
    static let generationPrompt = "<|im_start|>assistant\n"

    /// The vision block for `count` merged image tokens, empty when there is no image.
    static func vision(tokens count: Int) -> String {
        count > 0 ? "<|vision_start|>" + String(repeating: "<|image_pad|>", count: count) + "<|vision_end|>" : ""
    }

    /// The instruction with terminating punctuation, which is what the reference appends when the
    /// caller's instruction ends in anything else.
    static func punctuated(_ instruction: String) -> String {
        let trimmed = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let last = trimmed.unicodeScalars.last else { return trimmed }
        switch last.properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
             .initialPunctuation, .finalPunctuation, .otherPunctuation:
            return trimmed
        default:
            return trimmed + "."
        }
    }

    /// The whole prompt: a system turn, a user turn, and the generation prompt whose final token is the
    /// position both models read.
    static func prompt(system: String, user: String) -> String {
        systemOpen + system + turnEnd + userOpen + user + turnEnd + generationPrompt
    }

    /// The user turn's content, the reference's "NULL" when the caller passes neither text nor image.
    static func content(text: String?, imageTokens: Int) -> String {
        let body = text ?? ""
        if body.isEmpty && imageTokens == 0 { return "NULL" }
        return vision(tokens: imageTokens) + body
    }
}

/// Qwen3-VL-Embedding: one normalized vector for a text, an image, or an instruction-conditioned pair.
///
/// `NFKMLXQwen3VLEmbedder` is the released `Qwen/Qwen3-VL-Embedding-2B`, the Qwen3-VL backbone
/// ``NFKMLXQwen3VL`` runs under a retrieval fine-tune. The instruction goes in the system turn, the
/// content in the user turn, and the embedding is the hidden state of the generation prompt's last
/// token, L2-normalized, so a dot product between two embeddings is their cosine similarity.
///
/// @discussion Text and images land in one space, so a text query retrieves an image and an image
/// query retrieves a document. The instruction is part of the input rather than a wrapper around it:
/// the same content embeds differently under a different instruction, which is how one model serves
/// several retrieval tasks. Run embedding off the render thread.
@objc(NFKMLXQwen3VLEmbedder)
public final class NFKMLXQwen3VLEmbedder: NSObject, NFKTextEmbedding {

    /// A name for the model the factories produce.
    @objc public static let modelName = "qwen3-vl-embedding-2b"

    /// The instruction the release embeds with when the caller names none.
    @objc public static let defaultInstruction = "Represent the user's input."

    static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]
    static let optionalFiles = ["vocab.json", "merges.txt", "added_tokens.json"]
    static let weightFiles = ["model.safetensors", "model.safetensors.index.json"]

    private let visionNet: NFKMLXQwen3VLVisionNet
    private let decoder: NFKMLXLanguageNet
    private let tokenizer: NFKTokenizer
    private let processor: NFKMLXQwen3VLImageProcessor
    private let appendedToken: Int

    /// A fine-tuned adapter over the released embedding space, or nil to embed as released.
    ///
    /// Setting one makes every later embedding the adapted one. ``NFKMLXQwen3VLEmbedder/loadAdapter(from:)``
    /// installs one a fine-tune wrote.
    public var adapter: NFKMLXQwen3VLEmbeddingAdapter?

    /// How wide an embedding this model produces.
    @objc public var embeddingDimensions: Int { decoder.configuration.hiddenSize }

    init(visionNet: NFKMLXQwen3VLVisionNet, decoder: NFKMLXLanguageNet, tokenizer: NFKTokenizer,
         processor: NFKMLXQwen3VLImageProcessor, appendedToken: Int) {
        self.visionNet = visionNet
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.appendedToken = appendedToken
        super.init()
    }

    /// Loads the whole model — the vision tower, the decoder, and the release tokenizer — from a
    /// downloaded release directory.
    @objc(embedderWithDirectoryURL:error:)
    public static func embedder(directoryURL: URL) throws -> NFKMLXQwen3VLEmbedder {
        let vision = try NFKMLXQwen3VL.visionNet(directoryURL: directoryURL)
        let decoder = try NFKMLXQwen3VL.decoder(directoryURL: directoryURL)
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.weightsMismatch("the release carries no tokenizer files")
        }
        let (specials, _) = NFKMLXLanguage.specialTokens(inDirectory: directoryURL)
        return NFKMLXQwen3VLEmbedder(
            visionNet: vision, decoder: decoder, tokenizer: tokenizer,
            processor: NFKMLXQwen3VLImageProcessor.processor(inDirectory: directoryURL),
            appendedToken: specials["<|endoftext|>"] ?? 151_643)
    }

    /// The prompt the model reads for a text-and-image input, with `imageTokens` merged image
    /// positions. A caller that tokenizes elsewhere uses this to build the same input.
    public static func prompt(text: String?, imageTokens: Int, instruction: String?) -> String {
        NFKQwen3VLRetrievalPrompt.prompt(
            system: NFKQwen3VLRetrievalPrompt.punctuated(instruction ?? defaultInstruction),
            user: NFKQwen3VLRetrievalPrompt.content(text: text, imageTokens: imageTokens))
    }

    /// The token ids of an input under `instruction`, with `imageTokens` merged image positions.
    ///
    /// @discussion The release's `tokenizer.json` carries a template post-processor that appends
    /// `<|endoftext|>` to every encoding, and the appended token is the position the embedding is
    /// pooled at. The core tokenizer implements no post-processor, so the token is appended here. The
    /// reranker release carries no such post-processor, which is why it pools a different position.
    public func promptTokens(text: String?, imageTokens: Int = 0, instruction: String? = nil) -> [Int] {
        var ids = tokenizer.encode(Self.prompt(text: text, imageTokens: imageTokens,
                                               instruction: instruction)).map(\.intValue)
        ids.append(appendedToken)
        return ids
    }

    /// The embedding of a token sequence the caller built, pooled at the last position and normalized.
    ///
    /// The sequence carries no image, so this is the text path; ``embeddingVector(text:image:instruction:)``
    /// is the one that encodes an image.
    func embed(tokens: [Int]) -> MLXArray {
        adapted(NFKMLXQwen3VL.hiddenStates(decoder: decoder, inputIds: tokens, visionFeatures: nil,
                                           deepstack: [], gridT: 1, gridH: 1, gridW: 1))
    }

    /// The pooled embedding, through the installed adapter when there is one.
    func adapted(_ hidden: MLXArray) -> MLXArray {
        let pooled = Self.pooled(hidden)
        guard let adapter else { return pooled }
        return adapter(pooled.reshaped([1, pooled.dim(0)])).reshaped([-1])
    }

    /// The embedding of a text, an image, or both, under an instruction.
    ///
    /// A nil `instruction` embeds under ``defaultInstruction``. Passing neither text nor image embeds
    /// the reference's "NULL" placeholder rather than failing.
    public func embeddingVector(text: String?, image: CGImage?, instruction: String? = nil) -> MLXArray {
        guard let image else {
            return embed(tokens: promptTokens(text: text, instruction: instruction))
        }
        let (pixelValues, grid) = processor.process(image)
        let (features, deepstack) = visionNet(pixelValues, grid: grid)
        let merged = grid.t * (grid.h / processor.mergeSize) * (grid.w / processor.mergeSize)
        let ids = promptTokens(text: text, imageTokens: merged, instruction: instruction)
        let hidden = NFKMLXQwen3VL.hiddenStates(decoder: decoder, inputIds: ids, visionFeatures: features,
                                                deepstack: deepstack, gridT: grid.t, gridH: grid.h,
                                                gridW: grid.w)
        return adapted(hidden)
    }

    /// The last position's hidden state, L2-normalized.
    static func pooled(_ hidden: MLXArray) -> MLXArray {
        let last = hidden[0, hidden.dim(1) - 1]
        return last / sqrt((last * last).sum())
    }

    /// The embedding of a text under ``defaultInstruction``, as floats.
    @objc(embeddingForText:)
    public func embedding(forText text: String) -> [NSNumber] {
        Self.numbers(embeddingVector(text: text, image: nil, instruction: nil))
    }

    /// The embedding of a text under `instruction`, as floats. An empty instruction embeds under
    /// ``defaultInstruction``.
    @objc(embeddingForText:instruction:)
    public func embedding(forText text: String, instruction: String) -> [NSNumber] {
        Self.numbers(embeddingVector(text: text, image: nil,
                                     instruction: instruction.isEmpty ? nil : instruction))
    }

    /// The embedding of an image, optionally with text beside it, as floats. An empty text or
    /// instruction is the absent one.
    @objc(embeddingForImage:text:instruction:)
    public func embedding(forImage image: CGImage, text: String, instruction: String) -> [NSNumber] {
        Self.numbers(embeddingVector(text: text.isEmpty ? nil : text, image: image,
                                     instruction: instruction.isEmpty ? nil : instruction))
    }

    /// Builds a text-embedding backend over the model, so a caller reaches it through
    /// `NFKInferenceBackend` the way the other embedders are reached. The request's text is wrapped in
    /// the model's prompt before it is encoded.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        let embedder = try embedder(directoryURL: directoryURL)
        return NFKMLXTextEmbeddingBackend(embedder: embedder,
                                          tokenize: { embedder.promptTokens(text: $0) },
                                          identifier: modelName)
    }

    /// Downloads a Qwen3-VL-Embedding release into the hub cache and loads the whole model.
    ///
    /// @discussion The download fetches `config.json`, the tokenizer files, `preprocessor_config.json`
    /// (whose pixel bounds set how many tokens an image occupies), and the weights, following a shard
    /// index. A file already in the cache is not fetched again. The call blocks on the network; run it
    /// off the render thread. The public releases are `Qwen/Qwen3-VL-Embedding-2B` and
    /// `Qwen/Qwen3-VL-Embedding-8B`.
    @objc(embedderWithRepo:revision:cacheDirectoryURL:error:)
    public static func embedder(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXQwen3VLEmbedder {
        try embedder(directoryURL: try downloadedDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL))
    }

    /// The asynchronous form of ``embedder(repo:revision:cacheDirectoryURL:)``. The handler runs on a
    /// background queue.
    @objc(embedderWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func embedder(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                completionHandler: @escaping (NFKMLXQwen3VLEmbedder?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try embedder(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// Downloads a Qwen3-VL-Embedding release, as ``embedder(repo:revision:cacheDirectoryURL:)`` does,
    /// and builds the text-embedding backend over it.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try downloadedDirectory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL))
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) { try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL) }
    }

    private static func downloadedDirectory(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> URL {
        try NFKMLXReleaseDownload.directory(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                            required: requiredFiles, optional: optionalFiles, weights: weightFiles)
    }

    private static func numbers(_ embedding: MLXArray) -> [NSNumber] {
        eval(embedding)
        return embedding.asArray(Float.self).map { NSNumber(value: $0) }
    }
}

/// Qwen3-VL-Reranker: the relevance of a document to a query, either of which carries an image.
///
/// `NFKMLXQwen3VLReranker` is the released `Qwen/Qwen3-VL-Reranker-2B`. The query and the document go
/// into one prompt under an instruction, and the score is the model's preference for answering "yes"
/// over "no" at the generation prompt's last position, through a sigmoid. The reference builds that
/// preference as a one-output linear layer holding the difference of the output projection's two rows,
/// which is the difference of the two logits because the projection carries no bias.
///
/// @discussion A reranker reads the query and the document together, so it resolves pairs an embedding
/// similarity scores alike. It costs one forward pass per document, which is why a retrieval stage
/// narrows the candidates first. Run scoring off the render thread.
@objc(NFKMLXQwen3VLReranker)
public final class NFKMLXQwen3VLReranker: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "qwen3-vl-reranker-2b"

    /// The instruction the release judges with when the caller names none.
    @objc public static let defaultInstruction =
        "Given a search query, retrieve relevant candidates that answer the query."

    static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]
    static let optionalFiles = ["vocab.json", "merges.txt", "added_tokens.json", "1_LogitScore/config.json"]
    static let weightFiles = ["model.safetensors", "model.safetensors.index.json"]

    /// The system turn the release is trained with, which names the two answers it scores.
    static let judgement = "Judge whether the Document meets the requirements based on the Query and "
        + "the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."

    private let visionNet: NFKMLXQwen3VLVisionNet
    private let decoder: NFKMLXLanguageNet
    private let tokenizer: NFKTokenizer
    private let processor: NFKMLXQwen3VLImageProcessor
    private let yesToken: Int
    private let noToken: Int

    /// A fine-tuned scoring head, or nil to score with the release's own direction.
    ///
    /// Setting one makes every later score the retargeted one.
    /// ``NFKMLXQwen3VLReranker/loadHead(from:)`` installs one a fine-tune wrote.
    public var head: NFKMLXQwen3VLRerankerHead?

    init(visionNet: NFKMLXQwen3VLVisionNet, decoder: NFKMLXLanguageNet, tokenizer: NFKTokenizer,
         processor: NFKMLXQwen3VLImageProcessor, yesToken: Int, noToken: Int) {
        self.visionNet = visionNet
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.yesToken = yesToken
        self.noToken = noToken
        super.init()
    }

    /// Loads the whole model from a downloaded release directory, reading the two scored token ids from
    /// the release's `1_LogitScore/config.json`.
    @objc(rerankerWithDirectoryURL:error:)
    public static func reranker(directoryURL: URL) throws -> NFKMLXQwen3VLReranker {
        let vision = try NFKMLXQwen3VL.visionNet(directoryURL: directoryURL)
        let decoder = try NFKMLXQwen3VL.decoder(directoryURL: directoryURL)
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.weightsMismatch("the release carries no tokenizer files")
        }
        let (yes, no) = scoredTokens(inDirectory: directoryURL)
        return NFKMLXQwen3VLReranker(
            visionNet: vision, decoder: decoder, tokenizer: tokenizer,
            processor: NFKMLXQwen3VLImageProcessor.processor(inDirectory: directoryURL),
            yesToken: yes, noToken: no)
    }

    /// Downloads a Qwen3-VL-Reranker release into the hub cache and loads the whole model.
    ///
    /// @discussion The download fetches `config.json`, the tokenizer files, `preprocessor_config.json`,
    /// the `1_LogitScore/config.json` that names the two scored tokens, and the weights, following a
    /// shard index. A file already in the cache is not fetched again. The call blocks on the network;
    /// run it off the render thread. The public releases are `Qwen/Qwen3-VL-Reranker-2B` and
    /// `Qwen/Qwen3-VL-Reranker-8B`.
    @objc(rerankerWithRepo:revision:cacheDirectoryURL:error:)
    public static func reranker(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXQwen3VLReranker {
        try reranker(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``reranker(repo:revision:cacheDirectoryURL:)``. The handler runs on a
    /// background queue.
    @objc(rerankerWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func reranker(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                completionHandler: @escaping (NFKMLXQwen3VLReranker?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try reranker(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// The token ids the release scores, from its `1_LogitScore` module, defaulting to the ids of
    /// "yes" and "no" in the Qwen vocabulary.
    static func scoredTokens(inDirectory directory: URL) -> (yes: Int, no: Int) {
        let url = directory.appendingPathComponent("1_LogitScore/config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return (9693, 2152)
        }
        return (json["true_token_id"] as? Int ?? 9693, json["false_token_id"] as? Int ?? 2152)
    }

    /// The prompt for one query-document pair, with `queryImageTokens` and `documentImageTokens` merged
    /// image positions on either side.
    public static func prompt(query: String?, queryImageTokens: Int, document: String?,
                              documentImageTokens: Int, instruction: String?) -> String {
        let user = "<Instruct>: " + (instruction ?? defaultInstruction)
            + "<Query>:" + NFKQwen3VLRetrievalPrompt.content(text: query, imageTokens: queryImageTokens)
            + "\n<Document>:"
            + NFKQwen3VLRetrievalPrompt.content(text: document, imageTokens: documentImageTokens)
        return NFKQwen3VLRetrievalPrompt.prompt(system: judgement, user: user)
    }

    /// The relevance of a document to a query, between 0 and 1. Either side takes an image beside its
    /// text; a nil instruction judges under ``defaultInstruction``.
    public func score(query: String?, queryImage: CGImage? = nil, document: String?,
                      documentImage: CGImage? = nil, instruction: String? = nil) -> Double {
        var pixels = [MLXArray]()
        var grids = [(t: Int, h: Int, w: Int)]()
        for image in [queryImage, documentImage].compactMap({ $0 }) {
            let (values, grid) = processor.process(image)
            pixels.append(values)
            grids.append(grid)
        }
        let merged = grids.map { $0.t * ($0.h / processor.mergeSize) * ($0.w / processor.mergeSize) }
        let text = Self.prompt(query: query, queryImageTokens: queryImage == nil ? 0 : merged[0],
                               document: document,
                               documentImageTokens: documentImage == nil ? 0 : merged[merged.count - 1],
                               instruction: instruction)
        let ids = tokenizer.encode(text).map(\.intValue)

        var features: MLXArray?
        var deepstack = [MLXArray]()
        var grid = (t: 1, h: 1, w: 1)
        if !pixels.isEmpty {
            // One tower pass over the images the pair carries, concatenated in prompt order.
            var outputs = [MLXArray]()
            var stacks = [[MLXArray]]()
            for (values, each) in zip(pixels, grids) {
                let (output, stack) = visionNet(values, grid: each)
                outputs.append(output.reshaped([-1, decoder.configuration.hiddenSize]))
                stacks.append(stack)
            }
            features = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
            deepstack = stacks.count == 1 ? stacks[0] : (0 ..< stacks[0].count).map { index in
                concatenated(stacks.map { $0[index].reshaped([-1, $0[index].dim(-1)]) }, axis: 0)
            }
            grid = grids[0]
        }
        let hidden = NFKMLXQwen3VL.hiddenStates(decoder: decoder, inputIds: ids, visionFeatures: features,
                                                deepstack: deepstack, gridT: grid.t, gridH: grid.h,
                                                gridW: grid.w)
        let last = hidden[0..., (hidden.dim(1) - 1)...]
        if let head {
            let logit = head(last.reshaped([1, last.dim(-1)]))
            eval(logit)
            return 1 / (1 + exp(-Double(logit.item(Float.self))))
        }
        return Self.score(logits: decoder.logits(fromHidden: last),
                          yesToken: yesToken, noToken: noToken)
    }

    /// The release's own scoring direction, the output projection's "yes" row minus its "no" row.
    /// A retargeted head starts here, so an untrained one reproduces the released score.
    public func scoringDirection() -> MLXArray {
        let ids = MLXArray([Int32(yesToken), Int32(noToken)])
        let rows = decoder.lmHead.map { $0.weight.take(ids, axis: 0) } ?? decoder.embed(ids)
        return rows[0] - rows[1]
    }

    /// The sigmoid of the difference between the two scored logits, which is the reference's linear
    /// layer over the difference of the output projection's rows.
    static func score(logits: MLXArray, yesToken: Int, noToken: Int) -> Double {
        let flat = logits.reshaped([-1])
        let difference = flat[yesToken] - flat[noToken]
        eval(difference)
        return 1 / (1 + exp(-Double(difference.item(Float.self))))
    }

    /// The relevance of `document` to `query` under ``defaultInstruction``, between 0 and 1.
    @objc(scoreForQuery:document:)
    public func score(query: String, document: String) -> Double {
        score(query: query, document: document, instruction: nil)
    }

    /// The relevance of `document` to `query` under `instruction`. An empty instruction judges under
    /// ``defaultInstruction``.
    @objc(scoreForQuery:document:instruction:)
    public func score(query: String, document: String, instruction: String) -> Double {
        score(query: query, document: document, instruction: instruction.isEmpty ? nil : instruction)
    }

    /// The relevance of each document to `query`, in the documents' order. One forward pass per
    /// document.
    @objc(scoresForQuery:documents:)
    public func scores(query: String, documents: [String]) -> [NSNumber] {
        documents.map { NSNumber(value: score(query: query, document: $0, instruction: nil)) }
    }

    /// The documents' indices ordered from most relevant to least, which is the reranking.
    @objc(rankedIndicesForQuery:documents:)
    public func rankedIndices(query: String, documents: [String]) -> [NSNumber] {
        let scored = scores(query: query, documents: documents).map(\.doubleValue)
        return scored.indices.sorted { scored[$0] > scored[$1] }.map { NSNumber(value: $0) }
    }

    /// The relevance of an image document to a text query, between 0 and 1. An empty text or
    /// instruction is the absent one.
    @objc(scoreForQuery:documentImage:documentText:instruction:)
    public func score(query: String, documentImage: CGImage, documentText: String,
                      instruction: String) -> Double {
        score(query: query, queryImage: nil, document: documentText.isEmpty ? nil : documentText,
              documentImage: documentImage, instruction: instruction.isEmpty ? nil : instruction)
    }
}
