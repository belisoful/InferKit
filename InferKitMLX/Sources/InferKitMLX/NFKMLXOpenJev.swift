//
//  NFKMLXOpenJev.swift
//  InferKitMLX
//
//  Open-Jev (`ZefanCai/Open-Jev-2B`, `-9B`, and `-27B-v1.1`; weights Apache-2.0, loader MIT): typed
//  decisions from a Qwen3.5-architecture text model (Qwen3.5-2B, Qwen3.5-9B, Qwen3.8-27B). Every candidate answer is its own prompt, a chat-templated question that
//  asks whether the proposed answer is correct, and a float32 linear head reads the text model's last
//  hidden state to one score per candidate. A softmax over a question's candidate scores, at the saved
//  temperature, is its distribution; a noul is one prompt whose score is set against a fixed zero.
//
//  A release is a rank-8 LoRA adapter over six projections of the text model (the full-attention
//  `q_proj`/`k_proj`/`v_proj`/`o_proj` and the recurrence's `in_proj_qkv`/`out_proj`), the head, and the
//  calibration. It needs the exact base revision it was trained on, which is downloaded
//  separately. The decoder is `NFKMLXHybridLanguageNet`, at reference parity on its own.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// Which Open-Jev release a model loads.
@objc(NFKMLXOpenJevVariant)
public enum NFKMLXOpenJevVariant: Int, Sendable, CaseIterable {
    /// `ZefanCai/Open-Jev-2B`, over Qwen3.5-2B.
    case twoB = 0
    /// `ZefanCai/Open-Jev-9B`, over Qwen3.5-9B.
    case nineB = 1
    /// `ZefanCai/Open-Jev-27B-v1.1`, over Qwen3.8-27B.
    case twentySevenB = 2

    /// The repository holding the adapter, the head, and the calibration.
    public var repository: String {
        switch self {
        case .twoB: return "ZefanCai/Open-Jev-2B"
        case .nineB: return "ZefanCai/Open-Jev-9B"
        case .twentySevenB: return "ZefanCai/Open-Jev-27B-v1.1"
        }
    }

    /// The repository commit this package's measurements were taken at.
    public var measuredRevision: String {
        switch self {
        case .twoB: return "0c7aa498b1627be8da4acf34c863ff0ee0a92785"
        case .nineB: return "47e966881e489511c0c7f5633a9e1960a676a551"
        case .twentySevenB: return "28cf73067d5b337860bbef3c85b8b82ba8730956"
        }
    }

    /// The base model the adapter was trained on.
    public var baseRepository: String {
        switch self {
        case .twoB: return "Qwen/Qwen3.5-2B"
        case .nineB: return "Qwen/Qwen3.5-9B"
        case .twentySevenB: return "Qwen/Qwen3.8-27B"
        }
    }

    /// The base commit the release names in `model.json`; the adapter is only valid over it.
    public var baseRevision: String {
        switch self {
        case .twoB: return "15852e8c16360a2fea060d615a32b45270f8a8fc"
        case .nineB: return "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        case .twentySevenB: return "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        }
    }
}

// MARK: - Configuration

/// The decoder geometry, adapter shape, limits, and calibration of an Open-Jev release.
public struct NFKMLXOpenJevConfiguration: Sendable {
    /// The Qwen3.5 text model. The output projection is never built: the head reads the hidden state.
    public var decoder: NFKMLXHybridConfiguration
    public var temperature: Float
    /// The longest candidate prompt; a longer one is refused rather than truncated.
    public var maxLength: Int
    public var loraRank: Int
    public var loraAlpha: Float
    /// The projection names the adapter covers.
    public var targetModules: [String]
    public var baseModel: String
    public var baseRevision: String
    public var modelName: String

    public init(decoder: NFKMLXHybridConfiguration, temperature: Float, maxLength: Int = 4096, loraRank: Int = 8,
                loraAlpha: Float = 16,
                targetModules: [String] = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"],
                baseModel: String = "Qwen/Qwen3.5-2B", baseRevision: String = "", modelName: String = "open-jev") {
        var geometry = decoder
        geometry.tiesWordEmbeddings = true
        self.decoder = geometry
        self.temperature = temperature
        self.maxLength = maxLength
        self.loraRank = loraRank
        self.loraAlpha = loraAlpha
        self.targetModules = targetModules
        self.baseModel = baseModel
        self.baseRevision = baseRevision
        self.modelName = modelName
    }

    /// A small configuration for tests: four layers, one of them full attention.
    public static let tiny = NFKMLXOpenJevConfiguration(
        decoder: NFKMLXHybridConfiguration(hiddenSize: 32, layerCount: 4, intermediateSize: 64, vocabularySize: 512,
                                           headCount: 2, keyValueHeadCount: 1, headDimensions: 16,
                                           linearKeyHeadCount: 2, linearKeyHeadDimensions: 8,
                                           linearValueHeadCount: 2, linearValueHeadDimensions: 8),
        temperature: 1.5, maxLength: 256, modelName: "open-jev-tiny")

    /// Reads a release's `checkpoint` directory (`model.json`, `temperature.json`,
    /// `adapter/adapter_config.json`) and the base model's `config.json`.
    public init(checkpointDirectoryURL checkpoint: URL, baseDirectoryURL base: URL) throws {
        func json(_ url: URL) throws -> [String: Any] {
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
                throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
            }
            return object
        }
        let model = try json(checkpoint.appendingPathComponent("model.json"))
        let calibration = try json(checkpoint.appendingPathComponent("temperature.json"))
        let adapter = try json(checkpoint.appendingPathComponent("adapter/adapter_config.json"))
        let baseModel = model["model_id"] as? String ?? ""
        let size = baseModel.split(separator: "/").last?.split(separator: "-").dropFirst().joined(separator: "-") ?? ""
        let rank = (adapter["r"] as? NSNumber)?.intValue ?? (model["lora_rank"] as? NSNumber)?.intValue ?? 8
        self.init(decoder: try NFKMLXHybridLanguage.configuration(fromHuggingFace: base.appendingPathComponent("config.json")),
                  temperature: (calibration["temperature"] as? NSNumber)?.floatValue ?? 1,
                  maxLength: (model["max_length"] as? NSNumber)?.intValue ?? 4096,
                  loraRank: rank,
                  loraAlpha: (adapter["lora_alpha"] as? NSNumber)?.floatValue ?? Float(2 * rank),
                  targetModules: adapter["target_modules"] as? [String]
                      ?? ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"],
                  baseModel: baseModel, baseRevision: model["revision"] as? String ?? "",
                  modelName: size.isEmpty ? "open-jev" : "open-jev-\(size.lowercased())")
    }
}

// MARK: - The prompts

/// The text of each candidate a question asks about, the way the loader's `candidate_prompts` writes
/// it, and the chat turn it is wrapped in.
public enum NFKMLXOpenJevPrompt {

    /// One prompt per candidate: a choice's options (a described one as `name: description`), a
    /// score's levels, or a noul's single yes-or-no prompt with its meanings under the question.
    ///
    /// A noul takes both meanings or neither, as the loader's schema does.
    public static func candidates(state: Any, question: NFKDecisionQuestion) throws -> [String] {
        var asked = question.instructions
        if question.type == .noul {
            let meanings = (question.descriptions["true"], question.descriptions["false"])
            switch meanings {
            case let (yes?, no?):
                asked += "\nYes means: \(yes)\nNo means: \(no)"
            case (nil, nil):
                break
            default:
                throw NFKMLXError.unsupportedConfiguration("an Open-Jev noul takes both meanings or neither")
            }
        }
        let prefix = "Context:\n\(NFKMLXLayaPrompt.serialize(state: state))\n\nQuestion: \(asked)\n"
        if question.type == .noul {
            return [prefix + "Is the answer to this question yes? Answer Yes or No."]
        }
        return question.renderedOptions.map {
            prefix + "Proposed answer: \($0)\nIs this proposed answer correct? Answer Yes or No."
        }
    }

    /// The Qwen3.5 chat template's rendering of one user turn with the generation prompt and thinking
    /// switched off, which is how the loader applies it. The template trims the turn's content.
    public static func chat(_ content: String) -> String {
        "<|im_start|>user\n\(content.trimmingCharacters(in: .whitespacesAndNewlines))<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }
}

// MARK: - The network

/// The Qwen3.5 text model under `decoder.` and the scalar head under `head.`.
public final class NFKMLXOpenJevNet: Module {
    @ModuleInfo(key: "decoder") public var decoder: NFKMLXHybridLanguageNet
    @ModuleInfo(key: "head") public var head: Linear

    public let configuration: NFKMLXOpenJevConfiguration

    public init(_ c: NFKMLXOpenJevConfiguration) {
        configuration = c
        _decoder.wrappedValue = NFKMLXHybridLanguage.makeNet(c.decoder)
        _head.wrappedValue = Linear(c.decoder.hiddenSize, 1)
        super.init()
    }

    /// One candidate's raw score: the head over the normalized hidden state of its last token, in
    /// float32 whatever precision the decoder runs at.
    public func score(tokens: [Int]) -> MLXArray {
        let hidden = decoder.normalizedHidden(MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count]))
        return head(hidden[0, tokens.count - 1].asType(.float32).reshaped([1, -1])).reshaped([])
    }

    /// A question's raw logits from its candidates' scores; a noul's are `[0, s]`.
    public func logits(candidates: [[Int]], type: NFKDecisionType) -> MLXArray {
        let scores = candidates.map { score(tokens: $0) }
        return type == .noul ? stacked([MLXArray(Float(0)), scores[0]]) : stacked(scores)
    }

    /// Adapts the target projections with LoRA at the release's rank and scale, and freezes the rest.
    /// The factories call it before loading a release's adapter; a network built directly calls it
    /// to start a fresh adapter, whose zero `B` leaves every score as the base gives it.
    public func adapt() throws {
        let targets = Set(configuration.targetModules)
        _ = try NFKMLXLoRA.apply(to: decoder, rank: configuration.loraRank, alpha: configuration.loraAlpha) { path, _ in
            path.hasPrefix("model.layers.") && targets.contains(String(path.split(separator: ".").last ?? ""))
        }
    }
}

extension NFKMLXHybridLanguageNet {
    /// The final-normalized hidden states `[batch, length, hidden]`, without the output projection:
    /// the text model's `last_hidden_state`.
    func normalizedHidden(_ tokens: MLXArray) -> MLXArray {
        var hidden = model.embedTokens(tokens)
        let length = tokens.shape[1]
        let mask: MLXArray? = length > 1 ? NFKMLXLanguageNet.causalMask(length, offset: 0) : nil
        for layer in model.layers {
            hidden = layer(hidden, mask: mask)
        }
        return model.norm(hidden)
    }
}

// MARK: - The model

/// The files an Open-Jev release and its base are downloaded to.
@objc(NFKMLXOpenJevRelease)
public final class NFKMLXOpenJevRelease: NSObject {
    /// The release's `package/checkpoint` directory.
    @objc public let checkpointDirectoryURL: URL
    /// The base model at the revision the release names.
    @objc public let baseDirectoryURL: URL

    @objc public init(checkpointDirectoryURL: URL, baseDirectoryURL: URL) {
        self.checkpointDirectoryURL = checkpointDirectoryURL
        self.baseDirectoryURL = baseDirectoryURL
        super.init()
    }
}

/// Open-Jev: typed decisions by scoring each candidate answer with a Qwen3.5 text model.
///
/// @discussion It takes the core's `NFKDecisionQuestion`s and returns `NFKDecisionAnswer`s, the same
/// objects `NFKTypeSafeBackend` and `NFKMLXLaya` use. Each candidate is its own pass, so a question costs
/// one prompt per option (one for a noul), and questions are independent of each other and of their
/// order. A choice takes 1 to 255 options, a score 2 to 10 levels, and a prompt longer than the
/// release's 4,096 tokens is refused. Run it off the render thread.
@objc(NFKMLXOpenJev)
public final class NFKMLXOpenJev: NSObject, NFKMLXDecisionModel {

    public let net: NFKMLXOpenJevNet
    public let tokenizer: NFKMLXDecisionTokenizer?
    public var configuration: NFKMLXOpenJevConfiguration { net.configuration }
    public var decisionModelName: String { configuration.modelName }

    public init(net: NFKMLXOpenJevNet, tokenizer: NFKMLXDecisionTokenizer?) {
        self.net = net
        self.tokenizer = tokenizer
        super.init()
    }

    // MARK: Deciding

    /// Each candidate's chat-templated token ids.
    public func candidateTokens(state: Any, question: NFKDecisionQuestion) throws -> [[Int]] {
        guard let tokenizer else { throw NFKMLXError.unsupportedConfiguration("the model has no tokenizer") }
        try Self.validate(question)
        return try NFKMLXOpenJevPrompt.candidates(state: state, question: question).map { text in
            let tokens = tokenizer.encode(NFKMLXOpenJevPrompt.chat(text))
            guard tokens.count <= configuration.maxLength else {
                throw NFKMLXError.unsupportedConfiguration(
                    "a candidate prompt of \(tokens.count) tokens exceeds the \(configuration.maxLength) the release reads")
            }
            return tokens
        }
    }

    /// The raw logits of a question's candidates, before the temperature.
    public func logits(state: Any, question: NFKDecisionQuestion) throws -> [Float] {
        let values = net.logits(candidates: try candidateTokens(state: state, question: question), type: question.type)
        eval(values)
        return values.asArray(Float.self)
    }

    /// The calibrated distribution over a question's candidates (a noul's is `[no, yes]`).
    public func distribution(state: Any, question: NFKDecisionQuestion) throws -> [Double] {
        Self.softmax(try logits(state: state, question: question), temperature: Double(configuration.temperature))
    }

    /// Answers one question.
    @objc(answerForState:question:error:)
    public func answer(state: Any, question: NFKDecisionQuestion) throws -> NFKDecisionAnswer {
        Self.answer(question: question, probabilities: try distribution(state: state, question: question))
    }

    /// Answers every question, keyed as the questions were.
    @objc(answersForState:questions:error:)
    public func answers(state: Any, questions: [String: NFKDecisionQuestion]) throws -> [String: NFKDecisionAnswer] {
        try decide(state: state, identifiedQuestions: questions.map { (identifier: $0.key, question: $0.value) }).answers
    }

    public func decide(state: Any, identifiedQuestions: [(identifier: String, question: NFKDecisionQuestion)])
        throws -> (answers: [String: NFKDecisionAnswer], inputTokens: Int) {
        var answers = [String: NFKDecisionAnswer]()
        var tokens = 0
        for (identifier, question) in identifiedQuestions {
            let candidates = try candidateTokens(state: state, question: question)
            tokens += candidates.reduce(0) { $0 + $1.count }
            let values = net.logits(candidates: candidates, type: question.type)
            eval(values)
            let probabilities = Self.softmax(values.asArray(Float.self), temperature: Double(configuration.temperature))
            answers[identifier] = Self.answer(question: question, probabilities: probabilities)
        }
        return (answers, tokens)
    }

    static func validate(_ question: NFKDecisionQuestion) throws {
        switch question.type {
        case .choice where !(1 ... 255).contains(question.options.count):
            throw NFKMLXError.unsupportedConfiguration("a choice takes 1 to 255 candidates")
        case .score where !(2 ... 10).contains(question.options.count):
            throw NFKMLXError.unsupportedConfiguration("a score takes 2 to 10 levels")
        default:
            break
        }
    }

    /// The loader's stable softmax, in double precision.
    static func softmax(_ logits: [Float], temperature: Double) -> [Double] {
        let peak = Double(logits.max() ?? 0)
        let weights = logits.map { Foundation.exp((Double($0) - peak) / temperature) }
        let total = weights.reduce(0, +)
        return weights.map { $0 / total }
    }

    /// The loader's `format_response`, in the wire shape. A choice's confidence is how far its largest
    /// probability sits above uniform, scaled to [0, 1]; a score's is one minus the expected distance
    /// from the mode over what a uniform distribution would give.
    static func answer(question: NFKDecisionQuestion, probabilities: [Double]) -> NFKDecisionAnswer {
        let keys = question.answerKeys
        var byKey = [String: Double]()
        for (key, probability) in zip(keys, probabilities) { byKey[key] = probability }
        let count = probabilities.count
        let mode = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
        switch question.type {
        case .choice:
            let uniform = 1 / Double(count)
            let confidence = count == 1 ? 1 : (probabilities[mode] - uniform) / (1 - uniform)
            return .decided(["type": "choice", "choice": keys[mode], "probabilities": byKey, "confidence": confidence])
        case .score:
            let distance = probabilities.enumerated().reduce(0.0) { $0 + $1.element * Double(abs($1.offset - mode)) }
            let center = Double(count - 1) / 2
            let uniformDistance = (0 ..< count).reduce(0.0) { $0 + abs(Double($1) - center) } / Double(count)
            let expected = probabilities.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
            let legend = Dictionary(uniqueKeysWithValues: zip(keys, question.options))
            return .decided(["type": "score", "score": expected, "probabilities": byKey, "legend": legend,
                             "confidence": Swift.max(0, 1 - distance / uniformDistance)])
        default:
            return .decided(["type": "noul", "noul": count > 1 ? probabilities[1] : 0])
        }
    }

    // MARK: Building

    /// Builds the network over a base directory: the decoder's weights at `precision`, the adapter
    /// from `adapter/adapter_model.safetensors`, and the head from `head.safetensors` (a fine-tune's)
    /// or `head.pt` (the release's), always at float32.
    public static func network(checkpointDirectoryURL checkpoint: URL, baseDirectoryURL base: URL,
                               precision: NFKMLXWeightPrecision = .checkpoint) throws -> NFKMLXOpenJevNet {
        let configuration = try NFKMLXOpenJevConfiguration(checkpointDirectoryURL: checkpoint, baseDirectoryURL: base)
        try NFKMLXReleaseWeights.verifyFits(inDirectory: base, precision: precision)
        let net = NFKMLXOpenJevNet(configuration)
        try NFKMLXHybridLanguage.loadWeights(into: net.decoder, fromDirectory: base, precision: precision)
        try net.adapt()
        try loadAdapter(into: net, from: checkpoint.appendingPathComponent("adapter/adapter_model.safetensors"))
        try loadHead(into: net, checkpointDirectory: checkpoint)
        return net
    }

    /// The module key a PEFT adapter key maps to, with whether the tensor is stored transposed.
    static func adapterKey(forReference key: String) -> String? {
        let prefix = "base_model.model."
        guard key.hasPrefix(prefix) else { return nil }
        let path = key.dropFirst(prefix.count)
        if path.hasSuffix(".lora_A.weight") {
            return "decoder.model." + path.dropLast(".lora_A.weight".count) + ".lora_a"
        }
        if path.hasSuffix(".lora_B.weight") {
            return "decoder.model." + path.dropLast(".lora_B.weight".count) + ".lora_b"
        }
        return nil
    }

    /// Loads a PEFT adapter. PEFT stores `A` as `[rank, inputs]` and `B` as `[outputs, rank]`; the
    /// detour here multiplies by their transposes. The adapter takes the decoder's element type.
    static func loadAdapter(into net: NFKMLXOpenJevNet, from url: URL) throws {
        let dtype = net.decoder.model.embedTokens.weight.dtype
        let arrays = try NFKMLXWeights.loadCheckpoint(url: url).arrays
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays {
            guard let name = adapterKey(forReference: key) else {
                throw NFKMLXError.weightsMismatch("the adapter carries \(key), which no adapted projection takes")
            }
            mapped.append((name, value.transposed().asType(dtype)))
        }
        try applyPart(mapped, to: net) { $0.hasSuffix(".lora_a") || $0.hasSuffix(".lora_b") }
    }

    /// Loads the parameters `part` selects and no others: every one must be supplied, at its shape.
    static func applyPart(_ mapped: [(String, MLXArray)], to net: Module, part: (String) -> Bool) throws {
        let expected = Dictionary(uniqueKeysWithValues: net.parameters().flattened().filter { part($0.0) })
        let supplied = Dictionary(mapped, uniquingKeysWith: { first, _ in first })
        let missing = expected.keys.filter { supplied[$0] == nil }.sorted()
        let foreign = supplied.keys.filter { expected[$0] == nil }.sorted()
        let misshapen = supplied.compactMap { key, value in
            expected[key].flatMap { $0.shape == value.shape ? nil : "\(key) \(value.shape) for \($0.shape)" }
        }.sorted()
        guard missing.isEmpty, foreign.isEmpty, misshapen.isEmpty else {
            throw NFKMLXError.weightsMismatch("missing \(missing.prefix(3)), unexpected \(foreign.prefix(3)), "
                                              + "mismatched \(misshapen.prefix(3))")
        }
        net.update(parameters: ModuleParameters.unflattened(mapped))
        eval(net)
    }

    static func loadHead(into net: NFKMLXOpenJevNet, checkpointDirectory: URL) throws {
        let tuned = checkpointDirectory.appendingPathComponent("head.safetensors")
        let url = FileManager.default.fileExists(atPath: tuned.path) ? tuned : checkpointDirectory.appendingPathComponent("head.pt")
        let arrays = try NFKMLXWeights.loadCheckpoint(url: url).arrays
        try applyPart(arrays.map { ("head." + $0.key, $0.value.asType(.float32)) }, to: net) { $0.hasPrefix("head.") }
    }

    /// Builds the model from a release's `checkpoint` directory and its base, at the base's own
    /// precision (bfloat16, as the loader runs it).
    @objc(openJevWithCheckpointDirectoryURL:baseDirectoryURL:error:)
    public static func openJev(checkpointDirectoryURL: URL, baseDirectoryURL: URL) throws -> NFKMLXOpenJev {
        try openJev(checkpointDirectoryURL: checkpointDirectoryURL, baseDirectoryURL: baseDirectoryURL, precision: .checkpoint)
    }

    /// Builds the model at a chosen precision: `.float32` is what the parity measurements use and what
    /// a fine-tune needs, at twice the memory.
    @objc(openJevWithCheckpointDirectoryURL:baseDirectoryURL:precision:error:)
    public static func openJev(checkpointDirectoryURL: URL, baseDirectoryURL: URL,
                               precision: NFKMLXWeightPrecision) throws -> NFKMLXOpenJev {
        guard let tokenizer = tokenizer(inDirectory: baseDirectoryURL) else {
            throw NFKMLXError.unsupportedConfiguration(
                "the base \(baseDirectoryURL.lastPathComponent) has no readable tokenizer, so no candidate could be scored")
        }
        let net = try network(checkpointDirectoryURL: checkpointDirectoryURL, baseDirectoryURL: baseDirectoryURL,
                              precision: precision)
        return NFKMLXOpenJev(net: net, tokenizer: tokenizer)
    }

    /// The base model's Qwen3.5 tokenizer, with its chat markers as special tokens.
    public static func tokenizer(inDirectory baseDirectoryURL: URL) -> NFKMLXDecisionTokenizer? {
        NFKMLXLanguage.releaseTokenizer(inDirectory: baseDirectoryURL).map(NFKMLXDecisionTokenizer.init(tokenizer:))
    }

    /// The model as an inference backend.
    @objc(backendWithCheckpointDirectoryURL:baseDirectoryURL:error:)
    public static func backend(checkpointDirectoryURL: URL, baseDirectoryURL: URL) throws -> NFKMLXDecisionBackend {
        NFKMLXDecisionBackend(model: try openJev(checkpointDirectoryURL: checkpointDirectoryURL,
                                                 baseDirectoryURL: baseDirectoryURL))
    }

    /// This model as an inference backend.
    @objc public func makeBackend() -> NFKMLXDecisionBackend { NFKMLXDecisionBackend(model: self) }

    // MARK: Downloading

    /// The release files under `package/checkpoint/`.
    @objc public static let checkpointFiles = ["model.json", "temperature.json", "head.pt",
                                               "adapter/adapter_config.json", "adapter/adapter_model.safetensors"]

    /// The base model's files beside its weight shards, which the shard index names.
    @objc public static let baseFiles = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                                         "merges.txt", "model.safetensors.index.json"]

    /// Downloads a release at `revision` (nil for its measured revision) and the base at the revision
    /// the release names, into the hub cache. A cached file is not fetched again. The base is several
    /// gigabytes. Blocks on the network; run it off the render thread.
    @objc(downloadVariant:revision:cacheDirectoryURL:error:)
    public static func download(variant: NFKMLXOpenJevVariant, revision: String?,
                                cacheDirectoryURL: URL?) throws -> NFKMLXOpenJevRelease {
        try download(variant: variant, revision: revision,
                     hub: NFKHFHub(cacheDirectoryURL: cacheDirectoryURL ?? NFKHFHub.defaultCacheDirectoryURL()))
    }

    static func download(variant: NFKMLXOpenJevVariant, revision: String?, hub: NFKHFHub) throws -> NFKMLXOpenJevRelease {
        let release = revision ?? variant.measuredRevision
        var checkpoint: URL?
        for path in checkpointFiles {
            let local = try hub.downloadRepo(variant.repository, revision: release, path: "package/checkpoint/\(path)", sha256: nil)
            if path == "model.json" { checkpoint = local.deletingLastPathComponent() }
        }
        guard let checkpoint else { throw NFKMLXError.weightsMismatch("the release names no model.json") }
        let model = try JSONSerialization.jsonObject(with: Data(contentsOf: checkpoint.appendingPathComponent("model.json"))) as? [String: Any]
        let baseRevision = model?["revision"] as? String ?? variant.baseRevision
        var base: URL?
        for path in baseFiles {
            base = try hub.downloadRepo(variant.baseRepository, revision: baseRevision, path: path, sha256: nil)
                .deletingLastPathComponent()
        }
        guard let base else { throw NFKMLXError.weightsMismatch("the base names no files") }
        let index = try JSONSerialization.jsonObject(with: Data(contentsOf: base.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any]
        let shards = Set(((index?["weight_map"] as? [String: String]) ?? [:]).values).sorted()
        for shard in shards {
            _ = try hub.downloadRepo(variant.baseRepository, revision: baseRevision, path: shard, sha256: nil)
        }
        return NFKMLXOpenJevRelease(checkpointDirectoryURL: checkpoint, baseDirectoryURL: base)
    }

    /// Downloads a release and its base (or reads them from the cache) and builds the model at the
    /// base's own precision.
    @objc(openJevWithVariant:revision:cacheDirectoryURL:error:)
    public static func openJev(variant: NFKMLXOpenJevVariant, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXOpenJev {
        let release = try download(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try openJev(checkpointDirectoryURL: release.checkpointDirectoryURL, baseDirectoryURL: release.baseDirectoryURL)
    }

    /// The asynchronous form of ``openJev(variant:revision:cacheDirectoryURL:)``; the handler runs on
    /// a background queue.
    @objc(openJevWithVariant:revision:cacheDirectoryURL:completionHandler:)
    public static func openJev(variant: NFKMLXOpenJevVariant, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXOpenJev?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try openJev(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// Downloads a release and its base (or reads them from the cache) and builds the backend.
    @objc(backendWithVariant:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXOpenJevVariant, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXDecisionBackend {
        try openJev(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL).makeBackend()
    }

    /// The asynchronous form of ``backend(variant:revision:cacheDirectoryURL:)``.
    @objc(backendWithVariant:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXOpenJevVariant, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXDecisionBackend?, Error?) -> Void) {
        openJev(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL) { model, error in
            completionHandler(model?.makeBackend(), error)
        }
    }
}
