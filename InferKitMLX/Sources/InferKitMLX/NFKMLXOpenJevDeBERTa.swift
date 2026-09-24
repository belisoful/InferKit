//
//  NFKMLXOpenJevDeBERTa.swift
//  InferKitMLX
//
//  open-jev-deberta-v3-large (`com-kotobalabs/open-jev-deberta-v3-large`, Apache-2.0), a community
//  reproduction of Jev's typed decisions on the DeBERTa-v3-large encoder. One pass reads the state and
//  every question together:
//
//      [CLS] [STATE] state ( [Q] instructions ( [OPT] option )* )* [SEP]
//
//  and each option is scored by a small head over the mean of its own text tokens, the mean of its
//  question's text tokens, and their product. A softmax within each question's options, at the
//  release's fitted temperature, is that question's distribution. The three markers are tokens the
//  release added to DeBERTa's vocabulary; the pooled spans exclude them.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

// MARK: - Configuration

/// The geometry, limits, special tokens, and calibration of an open-jev-deberta release.
public struct NFKMLXOpenJevDeBERTaConfiguration: Sendable {
    public var encoder: NFKMLXDeBERTaV2Configuration
    /// The temperature the logits are divided by before the softmax, fitted on a validation split.
    public var temperature: Float
    /// The most state tokens a sequence keeps.
    public var maxStateTokens: Int
    /// The longest sequence; the state is cut further so every question fits.
    public var maxLength: Int
    public var clsToken: Int
    public var sepToken: Int
    public var padToken: Int
    public var stateMarker: Int
    public var questionMarker: Int
    public var optionMarker: Int
    public var modelName: String

    public init(encoder: NFKMLXDeBERTaV2Configuration, temperature: Float = 1.05, maxStateTokens: Int = 256,
                maxLength: Int = 512, clsToken: Int = 1, sepToken: Int = 2, padToken: Int = 0,
                stateMarker: Int = 128_001, questionMarker: Int = 128_002, optionMarker: Int = 128_003,
                modelName: String = "open-jev-deberta-v3-large") {
        self.encoder = encoder
        self.temperature = temperature
        self.maxStateTokens = maxStateTokens
        self.maxLength = maxLength
        self.clsToken = clsToken
        self.sepToken = sepToken
        self.padToken = padToken
        self.stateMarker = stateMarker
        self.questionMarker = questionMarker
        self.optionMarker = optionMarker
        self.modelName = modelName
    }

    /// The published release.
    public static let v3Large = NFKMLXOpenJevDeBERTaConfiguration(encoder: .v3Large)

    /// A small configuration for tests.
    public static let tiny = NFKMLXOpenJevDeBERTaConfiguration(
        encoder: .tiny, temperature: 1, maxStateTokens: 24, maxLength: 64,
        stateMarker: 5, questionMarker: 6, optionMarker: 7, modelName: "open-jev-deberta-tiny")

    /// Reads a release directory: `config.json` for the encoder, `open_jev_config.json` for the
    /// limits and the temperature, and `added_tokens.json` / `tokenizer_config.json` for the ids.
    public init(directoryURL: URL) throws {
        func json(_ name: String) throws -> [String: Any] {
            let url = directoryURL.appendingPathComponent(name)
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
                throw NFKMLXError.unsupportedConfiguration("\(name) is not a JSON object")
            }
            return object
        }
        let release = try json("open_jev_config.json")
        guard (release["pool"] as? String ?? "span") == "span" else {
            throw NFKMLXError.unsupportedConfiguration("this port reads the span-pooled release; pool is \(release["pool"]!)")
        }
        let added = (try? json("added_tokens.json")) as? [String: Int] ?? [:]
        var special = [String: Int]()
        let decoder = (try? json("tokenizer_config.json"))?["added_tokens_decoder"] as? [String: [String: Any]] ?? [:]
        for (id, entry) in decoder {
            if let content = entry["content"] as? String, let value = Int(id) { special[content] = value }
        }
        func token(_ literal: String, _ fallback: Int) -> Int { added[literal] ?? special[literal] ?? fallback }
        self.init(encoder: try NFKMLXDeBERTaV2Configuration(configURL: directoryURL.appendingPathComponent("config.json")),
                  temperature: (release["temperature"] as? NSNumber)?.floatValue ?? 1,
                  maxStateTokens: (release["max_state_tokens"] as? NSNumber)?.intValue ?? 256,
                  maxLength: (release["max_len"] as? NSNumber)?.intValue ?? 512,
                  clsToken: token("[CLS]", 1), sepToken: token("[SEP]", 2), padToken: token("[PAD]", 0),
                  stateMarker: token("[STATE]", 128_001), questionMarker: token("[Q]", 128_002),
                  optionMarker: token("[OPT]", 128_003))
    }
}

// MARK: - The prompt

/// The sequence one state and its questions build, with the token spans each question and option
/// text occupies. The markers sit just before each span and are not part of it.
public struct NFKMLXOpenJevDeBERTaPrompt: Sendable {
    public let tokens: [Int]
    /// Each question's instruction text.
    public let questionSpans: [Range<Int>]
    /// Each question's option texts, in option order.
    public let optionSpans: [[Range<Int>]]

    /// Builds the sequence the way the release's collator does. The state is cut to
    /// `maxStateTokens`, and further when the questions need the room; questions that do not fit in
    /// `maxLength` on their own are refused.
    public init(state: Any, questions: [NFKDecisionQuestion], tokenizer: NFKMLXDecisionTokenizer,
                configuration c: NFKMLXOpenJevDeBERTaConfiguration) throws {
        let encoded = questions.map { question in
            (tokenizer.encode(question.instructions), question.renderedOptions.map(tokenizer.encode))
        }
        let questionTokens = encoded.reduce(0) { total, item in
            total + 1 + item.0.count + item.1.reduce(0) { $0 + 1 + $1.count }
        }
        let stateBudget = c.maxLength - 3 - questionTokens
        guard stateBudget >= 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "the questions need \(questionTokens + 3) tokens, more than the \(c.maxLength) the model reads")
        }
        let stateTokens = tokenizer.encode(NFKMLXLayaPrompt.serialize(state: state))
        var ids = [c.clsToken, c.stateMarker] + stateTokens.prefix(Swift.min(c.maxStateTokens, stateBudget))
        var questionSpans = [Range<Int>]()
        var optionSpans = [[Range<Int>]]()
        for (instructions, options) in encoded {
            questionSpans.append((ids.count + 1) ..< (ids.count + 1 + instructions.count))
            ids += [c.questionMarker] + instructions
            var spans = [Range<Int>]()
            for option in options {
                spans.append((ids.count + 1) ..< (ids.count + 1 + option.count))
                ids += [c.optionMarker] + option
            }
            optionSpans.append(spans)
        }
        ids.append(c.sepToken)
        tokens = ids
        self.questionSpans = questionSpans
        self.optionSpans = optionSpans
    }
}

// MARK: - The network

/// The DeBERTa encoder under `backbone.` and the scoring head under `head.`.
public final class NFKMLXOpenJevDeBERTaNet: Module {
    @ModuleInfo(key: "backbone") public var backbone: NFKMLXDeBERTaV2Net
    /// `Linear(3·hidden, hidden)`, GELU, `Linear(hidden, 1)`, keyed by position as the release's
    /// `nn.Sequential` saves them.
    @ModuleInfo(key: "head") public var head: [Module]

    public let configuration: NFKMLXOpenJevDeBERTaConfiguration

    public init(_ c: NFKMLXOpenJevDeBERTaConfiguration) {
        configuration = c
        let hidden = c.encoder.hiddenSize
        _backbone.wrappedValue = NFKMLXDeBERTaV2Net(c.encoder)
        _head.wrappedValue = [Linear(3 * hidden, hidden), GELU(), Linear(hidden, 1)]
        super.init()
    }

    /// The raw logits `[batch, questions, options]`, `-inf` where a question has fewer options.
    ///
    /// - Parameters:
    ///   - tokens: `[batch, length]`.
    ///   - attentionMask: `[batch, length]`, nil when nothing is padded.
    ///   - pooling: `[batch, questions · options + questions, length]`: each row averages one span,
    ///     the options first (question-major) and then each question's instruction text.
    ///   - optionMask: `[batch, questions, options]`, true where an option exists.
    public func logits(tokens: MLXArray, attentionMask: MLXArray?, pooling: MLXArray, optionMask: MLXArray) -> MLXArray {
        let (batch, questions, options) = (optionMask.shape[0], optionMask.shape[1], optionMask.shape[2])
        let hidden = backbone(tokens, attentionMask: attentionMask)
        let pooled = matmul(pooling, hidden)
        let width = hidden.shape[2]
        let optionMeans = pooled[0..., 0 ..< (questions * options)].reshaped([batch, questions, options, width])
        let questionMeans = broadcast(pooled[0..., (questions * options)...].expandedDimensions(axis: 2),
                                      to: [batch, questions, options, width])
        var x = concatenated([questionMeans, optionMeans, questionMeans * optionMeans], axis: -1)
        x = (head[0] as! Linear)(x)
        x = gelu(x)
        x = (head[2] as! Linear)(x).squeezed(axis: -1)
        return which(optionMask, x, MLXArray(-Float.infinity))
    }

    /// The inputs one prompt makes: its tokens `[1, length]`, the pooling rows, and the option mask.
    static func inputs(_ prompts: [NFKMLXOpenJevDeBERTaPrompt], padToken: Int)
        -> (tokens: MLXArray, attentionMask: MLXArray?, pooling: MLXArray, optionMask: MLXArray) {
        let length = prompts.map(\.tokens.count).max() ?? 0
        let questions = prompts.map(\.questionSpans.count).max() ?? 0
        let options = prompts.flatMap { $0.optionSpans.map(\.count) }.max() ?? 0
        let slots = questions * options + questions
        var tokens = [Int32](repeating: Int32(padToken), count: prompts.count * length)
        var valid = [Float](repeating: 0, count: prompts.count * length)
        var pooling = [Float](repeating: 0, count: prompts.count * slots * length)
        var mask = [Bool](repeating: false, count: prompts.count * questions * options)
        for (b, prompt) in prompts.enumerated() {
            for (i, token) in prompt.tokens.enumerated() {
                tokens[b * length + i] = Int32(token)
                valid[b * length + i] = 1
            }
            func average(_ span: Range<Int>, into slot: Int) {
                // An empty span averages to zero, as the reference's count floor of one leaves it.
                let weight = 1 / Float(Swift.max(span.count, 1))
                for position in span { pooling[(b * slots + slot) * length + position] = weight }
            }
            for (q, spans) in prompt.optionSpans.enumerated() {
                for (o, span) in spans.enumerated() {
                    average(span, into: q * options + o)
                    mask[(b * questions + q) * options + o] = true
                }
            }
            for (q, span) in prompt.questionSpans.enumerated() {
                average(span, into: questions * options + q)
            }
        }
        let padded = prompts.contains { $0.tokens.count < length }
        return (MLXArray(tokens, [prompts.count, length]),
                padded ? MLXArray(valid, [prompts.count, length]) : nil,
                MLXArray(pooling, [prompts.count, slots, length]),
                MLXArray(mask, [prompts.count, questions, options]))
    }
}

// MARK: - The model

/// open-jev-deberta-v3-large: typed decisions from one DeBERTa pass over the state and every question.
///
/// @discussion It takes the core's `NFKDecisionQuestion`s and returns `NFKDecisionAnswer`s, the same
/// objects `NFKTypeSafeBackend` and `NFKMLXLaya` use. Every question is read in the same sequence, so
/// answers depend on the questions' order; the ordered API keeps the caller's order, and the
/// dictionary API reads the questions in sorted identifier order. A choice takes 2 to 255 options and a
/// score 2 to 10 levels. Run it off the render thread.
@objc(NFKMLXOpenJevDeBERTa)
public final class NFKMLXOpenJevDeBERTa: NSObject, NFKMLXDecisionModel {

    /// The Hugging Face repository the release is published in.
    @objc public static let repository = "com-kotobalabs/open-jev-deberta-v3-large"

    /// The repository commit the reference-parity measurements were taken at.
    @objc public static let measuredRevision = "19bf9a64815add579fbf6c907bef584d9277a8e4"

    /// The files the model needs from the repository.
    @objc public static let releaseFiles = ["config.json", "open_jev_config.json", "added_tokens.json",
                                            "tokenizer_config.json", "spm.model", "head.safetensors",
                                            "model.safetensors"]

    public let net: NFKMLXOpenJevDeBERTaNet
    public let tokenizer: NFKMLXDecisionTokenizer?
    public var configuration: NFKMLXOpenJevDeBERTaConfiguration { net.configuration }
    public var decisionModelName: String { configuration.modelName }

    public init(net: NFKMLXOpenJevDeBERTaNet, tokenizer: NFKMLXDecisionTokenizer?) {
        self.net = net
        self.tokenizer = tokenizer
        super.init()
    }

    // MARK: Deciding

    /// The sequence the questions build over the state.
    public func prompt(state: Any, questions: [NFKDecisionQuestion]) throws -> NFKMLXOpenJevDeBERTaPrompt {
        guard let tokenizer else { throw NFKMLXError.unsupportedConfiguration("the model has no tokenizer") }
        for question in questions {
            try Self.validate(question)
        }
        return try NFKMLXOpenJevDeBERTaPrompt(state: state, questions: questions, tokenizer: tokenizer,
                                              configuration: configuration)
    }

    /// Each question's calibrated distribution over its options, in the questions' order.
    public func distributions(state: Any, questions: [NFKDecisionQuestion]) throws -> [[Float]] {
        let prompt = try prompt(state: state, questions: questions)
        let inputs = NFKMLXOpenJevDeBERTaNet.inputs([prompt], padToken: configuration.padToken)
        let logits = net.logits(tokens: inputs.tokens, attentionMask: inputs.attentionMask,
                                pooling: inputs.pooling, optionMask: inputs.optionMask)
        let probabilities = softmax(logits[0] / configuration.temperature, axis: -1, precise: true)
        eval(probabilities)
        let rows = probabilities.asArray(Float.self)
        let width = probabilities.shape[1]
        return questions.enumerated().map { index, question in
            Array(rows[(index * width) ..< (index * width + question.renderedOptions.count)])
        }
    }

    /// Answers the questions in order.
    public func decide(state: Any, questions: [NFKDecisionQuestion]) throws -> [NFKDecisionAnswer] {
        let rows = try distributions(state: state, questions: questions)
        return zip(questions, rows).map { Self.answer(question: $0, probabilities: $1) }
    }

    /// Answers the questions in order.
    @objc(answersForState:questionList:error:)
    public func answers(state: Any, questionList: [NFKDecisionQuestion]) throws -> [NFKDecisionAnswer] {
        try decide(state: state, questions: questionList)
    }

    /// Answers every question, keyed as the questions were, reading them in sorted identifier order.
    @objc(answersForState:questions:error:)
    public func answers(state: Any, questions: [String: NFKDecisionQuestion]) throws -> [String: NFKDecisionAnswer] {
        let ordered = questions.keys.sorted().map { (identifier: $0, question: questions[$0]!) }
        return try decide(state: state, identifiedQuestions: ordered).answers
    }

    public func decide(state: Any, identifiedQuestions: [(identifier: String, question: NFKDecisionQuestion)])
        throws -> (answers: [String: NFKDecisionAnswer], inputTokens: Int) {
        let questions = identifiedQuestions.map(\.question)
        let answered = try decide(state: state, questions: questions)
        let tokens = try prompt(state: state, questions: questions).tokens.count
        var answers = [String: NFKDecisionAnswer]()
        for (entry, answer) in zip(identifiedQuestions, answered) { answers[entry.identifier] = answer }
        return (answers, tokens)
    }

    static func validate(_ question: NFKDecisionQuestion) throws {
        switch question.type {
        case .choice where !(2 ... 255).contains(question.options.count):
            throw NFKMLXError.unsupportedConfiguration("a choice takes 2 to 255 options")
        case .score where !(2 ... 10).contains(question.options.count):
            throw NFKMLXError.unsupportedConfiguration("a score takes 2 to 10 levels")
        default:
            break
        }
    }

    /// The answer the release's `readout` gives, in the wire shape: the confidence is the largest
    /// probability, a score is the expected level index, and a noul is the probability of `yes`.
    static func answer(question: NFKDecisionQuestion, probabilities: [Float]) -> NFKDecisionAnswer {
        let keys = question.answerKeys
        let best = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
        let confidence = Double(probabilities.max() ?? 0)
        var byKey = [String: Double]()
        for (key, probability) in zip(keys, probabilities) { byKey[key] = Double(probability) }
        switch question.type {
        case .choice:
            return .decided(["type": "choice", "choice": keys[best], "probabilities": byKey, "confidence": confidence])
        case .score:
            let expected = probabilities.enumerated().reduce(0.0) { $0 + Double($1.offset) * Double($1.element) }
            let legend = Dictionary(uniqueKeysWithValues: zip(keys, question.options))
            return .decided(["type": "score", "score": expected, "legend": legend, "probabilities": byKey,
                             "confidence": confidence])
        default:
            return .decided(["type": "noul", "noul": Double(probabilities.count > 1 ? probabilities[1] : 0)])
        }
    }

    // MARK: Building

    /// The network, for training or inspection. Nil weights is the random initialization; a file
    /// saved by `NFKMLXWeights.save` loads under its own keys.
    public static func network(weightsURL: URL?, configuration: NFKMLXOpenJevDeBERTaConfiguration = .v3Large)
        throws -> NFKMLXOpenJevDeBERTaNet {
        let net = NFKMLXOpenJevDeBERTaNet(configuration)
        if let weightsURL {
            let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
            try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value.asType(.float32)) }, to: net,
                                    verifyShapes: true)
        }
        return net
    }

    /// Loads a release directory: the encoder from `model.safetensors`, the head from `head.safetensors`.
    static func loadRelease(into net: NFKMLXOpenJevDeBERTaNet, directory: URL) throws {
        try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: .float32)
        let encoder = try NFKMLXWeights.loadCheckpoint(url: directory.appendingPathComponent("model.safetensors"))
        let head = try NFKMLXWeights.loadCheckpoint(url: directory.appendingPathComponent("head.safetensors"))
        let arrays = encoder.arrays.map { ("backbone." + $0.key, $0.value.asType(.float32)) }
            + head.arrays.map { ("head." + $0.key, $0.value.asType(.float32)) }
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
    }

    /// The release's SentencePiece tokenizer, from `spm.model`, summing scores in double precision as
    /// the release's `tokenizer.json` does, which is the tokenizer its code loads.
    public static func tokenizer(inDirectory directory: URL) throws -> NFKMLXDecisionTokenizer {
        NFKMLXDecisionTokenizer(sentencePiece: try NFKMLXSentencePieceSegmenter(
            contentsOf: directory.appendingPathComponent("spm.model"), accumulatesInDoublePrecision: true))
    }

    /// Builds the model from a release directory.
    @objc(openJevWithDirectoryURL:error:)
    public static func openJev(directoryURL: URL) throws -> NFKMLXOpenJevDeBERTa {
        try openJev(directoryURL: directoryURL, weightsURL: nil)
    }

    /// Builds the model from a release directory with its weights replaced by a fine-tuned file.
    @objc(openJevWithDirectoryURL:weightsURL:error:)
    public static func openJev(directoryURL: URL, weightsURL: URL?) throws -> NFKMLXOpenJevDeBERTa {
        let configuration = try NFKMLXOpenJevDeBERTaConfiguration(directoryURL: directoryURL)
        let net: NFKMLXOpenJevDeBERTaNet
        if let weightsURL {
            net = try network(weightsURL: weightsURL, configuration: configuration)
        } else {
            net = NFKMLXOpenJevDeBERTaNet(configuration)
            try loadRelease(into: net, directory: directoryURL)
        }
        return NFKMLXOpenJevDeBERTa(net: net, tokenizer: try tokenizer(inDirectory: directoryURL))
    }

    /// The model as an inference backend.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> NFKMLXDecisionBackend {
        NFKMLXDecisionBackend(model: try openJev(directoryURL: directoryURL))
    }

    /// This model as an inference backend.
    @objc public func makeBackend() -> NFKMLXDecisionBackend { NFKMLXDecisionBackend(model: self) }

    // MARK: Downloading

    /// Downloads the release into the hub cache and returns its folder. A cached file is not fetched
    /// again. Blocks on the network; run it off the render thread.
    @objc(downloadRevision:cacheDirectoryURL:error:)
    public static func download(revision: String?, cacheDirectoryURL: URL?) throws -> URL {
        try download(revision: revision,
                     hub: NFKHFHub(cacheDirectoryURL: cacheDirectoryURL ?? NFKHFHub.defaultCacheDirectoryURL()))
    }

    static func download(revision: String?, hub: NFKHFHub) throws -> URL {
        var folder: URL?
        for path in releaseFiles {
            folder = try hub.downloadRepo(repository, revision: revision, path: path, sha256: nil)
                .deletingLastPathComponent()
        }
        guard let folder else { throw NFKMLXError.weightsMismatch("the release names no files") }
        return folder
    }

    /// Downloads the release (or reads it from the cache) and builds the model.
    @objc(openJevWithRevision:cacheDirectoryURL:error:)
    public static func openJev(revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXOpenJevDeBERTa {
        try openJev(directoryURL: download(revision: revision, cacheDirectoryURL: cacheDirectoryURL))
    }

    /// The asynchronous form of ``openJev(revision:cacheDirectoryURL:)``; the handler runs on a
    /// background queue.
    @objc(openJevWithRevision:cacheDirectoryURL:completionHandler:)
    public static func openJev(revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXOpenJevDeBERTa?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try openJev(revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// Downloads the release (or reads it from the cache) and builds the backend.
    @objc(backendWithRevision:cacheDirectoryURL:error:)
    public static func backend(revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXDecisionBackend {
        try openJev(revision: revision, cacheDirectoryURL: cacheDirectoryURL).makeBackend()
    }

    /// The asynchronous form of ``backend(revision:cacheDirectoryURL:)``.
    @objc(backendWithRevision:cacheDirectoryURL:completionHandler:)
    public static func backend(revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXDecisionBackend?, Error?) -> Void) {
        openJev(revision: revision, cacheDirectoryURL: cacheDirectoryURL) { model, error in
            completionHandler(model?.makeBackend(), error)
        }
    }
}
