//
//  NFKMLXTranslateGemma.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

// TranslateGemma (Google, Gemma terms): Gemma 3 fine-tuned for translation, driven by a chat template
// whose one user item names the source and target languages. The network is the shipped Gemma 3
// (`NFKMLXGemma3Model`); this file adds the template rendered in Swift (the release's Jinja carries a
// 580-entry language table and multi-line string concatenation), the language resolution, the
// translator face, and the supervised fine-tuning recipe.

/// A loaded TranslateGemma release.
///
/// @discussion The prompt follows the release's `chat_template.jinja` exactly: `<bos><start_of_turn>user`,
/// the "You are a professional … translator" instruction naming both languages by name and code, the
/// text, `<end_of_turn>`, and the opened model turn. The language table is read from that file, so a
/// tag the release names (`de`, `de-DE`, `zh-Hant`) resolves as written and any other BCP-47 tag falls
/// back to its primary subtag. Decoding is greedy and stops at `<end_of_turn>` or `<eos>`; the beam
/// knobs of the seq2seq translators do not apply.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXTranslateGemmaTranslator: NFKMLXTranslator {
    public let model: NFKMLXGemma3Model
    /// The release's language table: code (`de-DE`) → name (`German`).
    public let languages: [String: String]
    public let identifier: String
    public var defaultDecoding: NFKMLXSeq2SeqDecoding

    init(model: NFKMLXGemma3Model, languages: [String: String], identifier: String) {
        self.model = model
        self.languages = languages
        self.identifier = identifier
        defaultDecoding = NFKMLXSeq2SeqDecoding(beams: 1, maxTokens: 256, startToken: model.tokens.beginOfSequence,
                                                endToken: model.tokens.endOfTurn)
    }

    public var fixedSourceLanguage: String? { nil }
    public var fixedTargetLanguage: String? { nil }

    /// The table's spelling of a BCP-47 tag: the tag as written (primary lowercase, script title case,
    /// region upper case), then primary with script, then the primary alone; nil when none is named.
    public func code(for language: String) -> String? {
        let parts = language.replacingOccurrences(of: "_", with: "-").split(separator: "-").map(String.init)
        guard let first = parts.first else { return nil }
        let primary = first.lowercased()
        let rest = parts.dropFirst().map { part -> String in
            if part.count == 4 { return part.prefix(1).uppercased() + part.dropFirst().lowercased() }
            return part.uppercased()
        }
        var candidates = [([primary] + rest).joined(separator: "-")]
        if let script = rest.first(where: { $0.count == 4 }) { candidates.append("\(primary)-\(script)") }
        candidates.append(primary)
        return candidates.first { languages[$0] != nil }
    }

    public func supports(language: String) -> Bool { code(for: language) != nil }

    /// The user turn and the opened model turn for `text`, without the leading `<bos>`.
    public func prompt(text: String, sourceCode: String, targetCode: String) -> String {
        let source = languages[sourceCode] ?? sourceCode
        let target = languages[targetCode] ?? targetCode
        return "<start_of_turn>user\nYou are a professional \(source) (\(sourceCode)) to \(target) (\(targetCode)) translator. "
            + "Your goal is to accurately convey the meaning and nuances of the original \(source) text while adhering to "
            + "\(target) grammar, vocabulary, and cultural sensitivities.\n"
            + "Produce only the \(target) translation, without any additional explanations or commentary. "
            + "Please translate the following \(source) text into \(target):\n\n\n"
            + text.trimmingCharacters(in: .whitespacesAndNewlines)
            + "<end_of_turn>\n<start_of_turn>model\n"
    }

    /// The ids the model reads: `<bos>` and the rendered prompt.
    public func promptTokens(text: String, sourceCode: String, targetCode: String) -> [Int] {
        model.promptTokens(prompt(text: text, sourceCode: sourceCode, targetCode: targetCode))
    }

    /// Greedy continuation ids for a prompt, stop token excluded.
    public func generate(promptTokens ids: [Int], maxTokens: Int) throws -> [Int] {
        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = maxTokens
        return try model.generate(tokens: ids, options: options)
    }

    public func translate(_ text: String, from source: String?, to target: String,
                          decoding: NFKMLXSeq2SeqDecoding) throws -> String {
        guard let source, let sourceCode = code(for: source) else {
            throw NFKMLXTranslationBackend.error(.error_InferenceMissingInput,
                                                 "\(identifier) needs the source language, which was neither given nor detected")
        }
        guard let targetCode = code(for: target) else {
            throw NFKMLXTranslationBackend.error(.error_InferenceUnsupported, "\(identifier) does not translate into \(target)")
        }
        let ids = promptTokens(text: text, sourceCode: sourceCode, targetCode: targetCode)
        let produced = try generate(promptTokens: ids, maxTokens: decoding.maxTokens)
        return model.decode(produced).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The `"code": "Name"` pairs of the template's language table.
    static func languageTable(inDirectory directory: URL) -> [String: String] {
        guard let text = try? String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8),
              let end = text.range(of: "-%}") else { return [:] }
        let table = text[..<end.lowerBound]
        var result = [String: String]()
        let pattern = try! NSRegularExpression(pattern: #""([^"]+)":\s*"([^"]+)""#)
        let whole = String(table)
        for match in pattern.matches(in: whole, range: NSRange(whole.startIndex..., in: whole)) {
            guard let codeRange = Range(match.range(at: 1), in: whole),
                  let nameRange = Range(match.range(at: 2), in: whole) else { continue }
            result[String(whole[codeRange])] = String(whole[nameRange])
        }
        return result
    }
}

/// The supervised fine-tuning objective: cross-entropy over the model turn's tokens alone, the prompt
/// positions masked, as the reference's `labels=-100` over the prompt scores.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTranslateGemmaObjective: Sendable {
    public var labelSmoothing: Float

    public init(labelSmoothing: Float = 0) {
        self.labelSmoothing = labelSmoothing
    }

    /// Scores `net` on one example: the prompt ids `[P]` and the model turn's ids `[T]` (translation
    /// then `<end_of_turn>`).
    public func callAsFunction(_ net: NFKMLXGemma3Net, _ prompt: MLXArray, _ target: MLXArray) -> MLXArray {
        let full = concatenated([prompt, target])
        let logits = net(full.reshaped([1, full.shape[0]]))
        return loss(logits: logits, promptLength: prompt.shape[0], target: target)
    }

    /// Scores full-sequence logits `[1, P + T, vocabulary]` on the `T` target positions.
    public func loss(logits: MLXArray, promptLength: Int, target: MLXArray) -> MLXArray {
        let length = target.shape[0]
        guard length > 0, promptLength > 0 else { return MLXArray(Float(0)) }
        let vocabulary = logits.shape[2]
        // The reference scores float32 logits whatever the weights' precision; a bfloat16 log-softmax
        // over a 262k vocabulary is what a half-precision load would otherwise feed the loss.
        let predictions = logits[0, (promptLength - 1) ..< (promptLength - 1 + length), 0...]
            .reshaped([length, vocabulary]).asType(.float32)
        return crossEntropy(logits: predictions, targets: target, labelSmoothing: labelSmoothing, reduction: .mean)
    }
}

/// Registration, download, and construction of TranslateGemma backends.
///
/// @discussion A release directory is a Gemma 3 release (`config.json`, sharded `model-*.safetensors`
/// with their index, `tokenizer.json`) plus the `chat_template.jinja` that names the languages. The 4B
/// is 8.6 GB of bfloat16; `.checkpoint` keeps it at that size and `.float32` doubles it, which is
/// what the parity is measured at.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXTranslateGemma)
public final class NFKMLXTranslateGemma: NSObject {
    /// The registered name.
    @objc public static let modelName = "translategemma"
    static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    static let optionalFiles = ["generation_config.json", "special_tokens_map.json", "added_tokens.json"]
    static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// Loads a release directory as a translator.
    public static func translator(directoryURL directory: URL,
                                  precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXTranslateGemmaTranslator {
        try translator(model: try NFKMLXGemma3.model(directoryURL: directory, precision: precision), directoryURL: directory)
    }

    /// Wraps a loaded Gemma 3 model with the release's language table.
    public static func translator(model: NFKMLXGemma3Model, directoryURL directory: URL) throws -> NFKMLXTranslateGemmaTranslator {
        let languages = NFKMLXTranslateGemmaTranslator.languageTable(inDirectory: directory)
        guard !languages.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) has no chat_template.jinja naming its languages")
        }
        return NFKMLXTranslateGemmaTranslator(model: model, languages: languages, identifier: modelName)
    }

    /// Wraps an adapted decoder with the release's tokenizer, vision tower, and language table.
    public static func translator(decoder: NFKMLXGemma3Net, directoryURL directory: URL,
                                  precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXTranslateGemmaTranslator {
        let parts = try NFKMLXGemma3.load(directory: directory, precision: precision, decoder: false)
        let model = NFKMLXGemma3Model(decoder: decoder, vision: parts.vision, projector: parts.projector,
                                      tokenizer: parts.tokenizer, tokens: parts.tokens,
                                      chatTemplate: NFKMLXGemma3.chatTemplate(inDirectory: directory))
        return try translator(model: model, directoryURL: directory)
    }

    /// Builds the backend from a release directory at float32.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: .float32)
    }

    /// Builds the backend from a release directory at the given precision.
    @objc(backendWithDirectoryURL:precision:error:)
    public static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        NFKMLXTranslationBackend(translator: try translator(directoryURL: directoryURL, precision: precision))
    }

    /// Downloads a release (`google/translategemma-4b-it`; the repo is gated, so the hub needs an
    /// access token) and builds the backend. Blocking on the network.
    @objc(backendWithRepo:revision:cacheDirectoryURL:precision:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles), precision: precision)
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:precision:)``.
    @objc(backendWithRepo:revision:cacheDirectoryURL:precision:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?, precision: NFKMLXWeightPrecision,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, precision: precision)
        }
    }

    /// Registers `translategemma` with `NFKMLXModelRegistry`; the registry's URL is the release
    /// directory, loaded at the checkpoint's own precision.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in
            guard let url else { throw NFKMLXError.unsupportedConfiguration("translategemma builds from a release directory, not without weights") }
            return try backend(directoryURL: url, precision: .checkpoint)
        }
    }

    /// The projections LoRA targets: query and value of every decoder attention.
    static func isDecoderAttentionProjection(_ path: String) -> Bool {
        path.hasPrefix("layers.") && (path.hasSuffix(".q_proj") || path.hasSuffix(".v_proj"))
    }

    /// Adapts a TranslateGemma decoder's attention with LoRA and trains it on translation examples.
    ///
    /// - Parameters:
    ///   - net: the decoder, `NFKMLXTranslateGemma.translator(directoryURL:).model.decoder` for a
    ///     release, or a fresh ``NFKMLXGemma3Net`` for a test.
    ///   - examples: supplies one example per step: the prompt ids from
    ///     ``NFKMLXTranslateGemmaTranslator/promptTokens(text:sourceCode:targetCode:)`` and the model
    ///     turn's ids (the translation followed by `<end_of_turn>`).
    ///   - rank: the LoRA width. Nil trains every parameter.
    ///   - alpha: the adapter's strength, applied as `alpha / rank`.
    ///   - objective: the token loss.
    ///   - optimizer: the update rule. Nil uses the optimizer transformers' `Trainer` defaults to,
    ///     `torch.optim.AdamW` (bias-corrected) with no weight decay; the reference publishes no training
    ///     script beyond its model's `labels=` loss, and the learning rate is this package's choice.
    ///   - steps: how many examples to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Adapt a float32 load. Call `NFKMLXLoRA.merge(into:)` before saving; the saved decoder reloads
    /// into a ``NFKMLXGemma3Net`` of the same configuration through `NFKMLXWeights.apply`, and
    /// ``translator(decoder:directoryURL:precision:)`` wraps it with the release's tokenizer.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXGemma3Net,
        examples: (Int) -> (prompt: MLXArray, target: MLXArray),
        rank: Int? = 8,
        alpha: Float = 16,
        objective: NFKMLXTranslateGemmaObjective = NFKMLXTranslateGemmaObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        if let rank {
            let adapted = try NFKMLXLoRA.apply(to: net, rank: rank, alpha: alpha) { path, _ in
                isDecoderAttentionProjection(path)
            }
            guard adapted > 0 else {
                throw NFKMLXError.trainingDataMismatch("no decoder attention projections were found to adapt, so nothing would train")
            }
        }
        return try NFKMLXTrainer.train(
            net, optimizer: optimizer ?? NFKMLXReferenceOptimizers.adamW(learningRate: 1e-4, weightDecay: 0), steps: steps,
            batch: { let example = examples($0); return (example.prompt, example.target) },
            loss: objective.callAsFunction,
            clipGradientNorm: clipGradientNorm, checkpoint: checkpoint, observer: observer)
    }
}
