//
//  NFKMLXSeedX.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

// Seed-X (ByteDance Seed, OpenMDW): a Mistral 7B decoder trained to translate among 28 languages,
// prompted with a plain instruction that names both languages and ends in the target's tag. The
// network is the shipped language decoder (`NFKMLXLanguageNet`); this file adds the prompt, the
// language table, the translator face, and the fine-tuning recipe over the language family's.

/// A loaded Seed-X release.
///
/// @discussion The prompt follows the release's model card exactly: `<s>`, then `Translate the
/// following <Source> sentence into <Target>:`, a line break, the text, a space, and the target's tag
/// (`<de>`), which the release was trained to expect. The release has no chat template. The languages
/// are the 28 the card lists, named as it names them. A BCP-47 tag resolves by its primary subtag
/// (`zh-Hant` → `zh`, `pt-BR` → `pt`); `nb` and `no` are separate tags in the release. Decoding stops
/// at `</s>` and is greedy by default, as the release's generation config leaves it; `decoding.beams`
/// above 1 runs a beam search (the card's example searches four), scored as transformers' `generate`
/// scores one. The release opens its continuation with `<s>`, which the translation's text leaves out,
/// as the card's decode does.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXSeedXTranslator: NFKMLXTranslator {
    public let net: NFKMLXLanguageNet
    public let tokenizer: NFKTokenizer
    public let identifier: String
    public var defaultDecoding: NFKMLXSeq2SeqDecoding

    /// The release's languages: tag (`de`) → the name its prompt uses (`German`).
    public static let languages: [String: String] = [
        "ar": "Arabic", "cs": "Czech", "da": "Danish", "de": "German", "en": "English", "es": "Spanish",
        "fi": "Finnish", "fr": "French", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian",
        "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nb": "Norwegian Bokmal",
        "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
        "ru": "Russian", "sv": "Swedish", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
        "vi": "Vietnamese", "zh": "Chinese",
    ]

    init(net: NFKMLXLanguageNet, tokenizer: NFKTokenizer, identifier: String) {
        self.net = net
        self.tokenizer = tokenizer
        self.identifier = identifier
        defaultDecoding = NFKMLXSeq2SeqDecoding(beams: 1, maxTokens: 512, startToken: tokenizer.bosTokenId,
                                                endToken: tokenizer.eosTokenId)
    }

    public var fixedSourceLanguage: String? { nil }
    public var fixedTargetLanguage: String? { nil }
    public var layerStreams: [NFKMLXLayerStream] { [net.layerStream].compactMap { $0 } }

    /// The release's tag for a BCP-47 tag: its primary subtag, lowercased, where the release lists it.
    public func code(for language: String) -> String? {
        let primary = NFKMLXTranslationBackend.primary(language)
        return Self.languages[primary] != nil ? primary : nil
    }

    public func supports(language: String) -> Bool { code(for: language) != nil }

    /// The prompt text for two release tags, without the leading `<s>`.
    public func prompt(text: String, sourceCode: String, targetCode: String) -> String {
        let source = Self.languages[sourceCode] ?? sourceCode
        let target = Self.languages[targetCode] ?? targetCode
        return "Translate the following \(source) sentence into \(target):\n\(text) <\(targetCode)>"
    }

    /// The prompt's ids: `<s>` and the prompt text.
    public func promptTokens(text: String, sourceCode: String, targetCode: String) -> [Int] {
        [tokenizer.bosTokenId] + tokenizer.encode(prompt(text: text, sourceCode: sourceCode, targetCode: targetCode)).map(\.intValue)
    }

    /// The greedy continuation of `ids`, without the `</s>` that ends it. Throws a failed read of a
    /// streamed layer, which the next call reads again.
    public func generate(promptTokens ids: [Int], maxTokens: Int) throws -> [Int] {
        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = maxTokens
        options.stopTokens = [tokenizer.eosTokenId]
        let produced = net.generate(prompt: ids, options: options)
        do {
            try net.verifyStream()
        } catch {
            net.layerStream?.clearFailure()
            throw error
        }
        return produced
    }

    /// The best continuation of `ids` a `decoding.beams`-wide beam search finds, without the `</s>` that
    /// ends it. Throws a failed read of a streamed layer, which the next call reads again. Introduced in
    /// InferKit 0.4.0.
    public func beamSearch(promptTokens ids: [Int], decoding: NFKMLXSeq2SeqDecoding) throws -> [Int] {
        do {
            return try net.beamSearch(prompt: ids, decoding: decoding)
        } catch {
            net.layerStream?.clearFailure()
            throw error
        }
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
        let produced = decoding.beams > 1 ? try beamSearch(promptTokens: ids, decoding: decoding)
            : try generate(promptTokens: ids, maxTokens: decoding.maxTokens)
        return translation(of: produced)
    }

    /// The translation `ids` spell, without the special tokens the release can open a continuation with
    /// (`<s>`), as the model card decodes it (`skip_special_tokens`).
    public func translation(of ids: [Int]) -> String {
        let text = (tokenizer as? NFKMLXMetaspaceTokenizer)?.displayText(ids)
            ?? tokenizer.decode(ids.filter { $0 != tokenizer.bosTokenId && $0 != tokenizer.eosTokenId }
                .map { NSNumber(value: $0) })
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The ids a fine-tune scores for `translation`: its pieces and the `</s>` that ends it.
    public func targetTokens(translation: String) -> [Int] {
        tokenizer.encode(translation).map(\.intValue) + [tokenizer.eosTokenId]
    }
}

/// The two Seed-X translation releases. Introduced in InferKit 0.4.0.
@objc(NFKMLXSeedXVariant)
public enum NFKMLXSeedXVariant: Int, Sendable {
    /// `ByteDance-Seed/Seed-X-PPO-7B`, reinforcement-trained; the release the card recommends.
    case ppo
    /// `ByteDance-Seed/Seed-X-Instruct-7B`, instruction-tuned.
    case instruct

    /// The variant's Hugging Face repository.
    public var repository: String {
        switch self {
        case .ppo: return "ByteDance-Seed/Seed-X-PPO-7B"
        case .instruct: return "ByteDance-Seed/Seed-X-Instruct-7B"
        }
    }
}

/// Registration, download, and construction of Seed-X translation backends.
///
/// @discussion A release directory holds `config.json`, `tokenizer.json`, and `model.safetensors`, 15 GB
/// of bfloat16. At `.checkpoint` precision, the registry's choice, it loads at the release's bfloat16;
/// at `.float32` it is 30 GB, which a residency streams where it does not fit
/// (``NFKMLXResidency/automatic`` holds what fits and reads the other layers in their turn). The weights
/// are under the OpenMDW license.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXSeedX)
public final class NFKMLXSeedX: NSObject {
    /// The registered name.
    @objc public static let modelName = "seed-x"
    static let requiredFiles = ["config.json", "tokenizer.json"]
    static let optionalFiles = ["generation_config.json"]
    static let weightFiles = ["model.safetensors"]

    /// Loads a release directory as a translator, held as `residency` says. A streamed decoder is read
    /// from the release on every pass.
    public static func translator(directoryURL directory: URL, precision: NFKMLXWeightPrecision = .float32,
                                  residency: NFKMLXResidency = .automatic) throws -> NFKMLXSeedXTranslator {
        let (net, tokenizer) = try NFKMLXLanguage.loadedReleaseWithTokenizer(at: directory, precision: precision,
                                                                            residency: residency, streams: true)
        return NFKMLXSeedXTranslator(net: net, tokenizer: tokenizer, identifier: modelName)
    }

    /// ``translator(directoryURL:precision:residency:)`` as a catalog's load options name it: their
    /// precision (the release's bfloat16 when they name none) and residency, and a runtime weight
    /// quantization of the decoder, which a streamed decoder refuses.
    static func translator(directoryURL directory: URL, loadOptions: NFKMLXLoadOptions) throws -> NFKMLXSeedXTranslator {
        let translator = try translator(directoryURL: directory, precision: loadOptions.precision ?? .checkpoint,
                                        residency: loadOptions.residency)
        if loadOptions.quantizationBits != nil {
            try translator.net.requireHeldLayers(for: "quantize at load")
        }
        try loadOptions.applyWeightQuantization(to: translator.net)
        return translator
    }

    /// Wraps a decoder, such as one a fine-tune adapted, with the tokenizer of the release it came from.
    public static func translator(decoder: NFKMLXLanguageNet, directoryURL directory: URL) throws -> NFKMLXSeedXTranslator {
        guard let tokenizer = NFKMLXLanguage.tokenizer(directoryURL: directory) else {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) has no readable tokenizer")
        }
        return NFKMLXSeedXTranslator(net: decoder, tokenizer: tokenizer, identifier: modelName)
    }

    /// Builds the backend from a release directory at its bfloat16.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: .checkpoint, residency: .automatic)
    }

    /// Builds the backend from a release directory at `precision`.
    @objc(backendWithDirectoryURL:precision:error:)
    public static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: precision, residency: .automatic)
    }

    /// Builds the backend from a release directory at `precision`, held as `residency` says.
    @objc(backendWithDirectoryURL:precision:residency:error:)
    public static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        NFKMLXTranslationBackend(translator: try translator(directoryURL: directoryURL, precision: precision,
                                                            residency: residency))
    }

    /// Downloads a release and builds the backend at `precision`. Blocking on the network.
    @objc(backendWithVariant:revision:cacheDirectoryURL:precision:error:)
    public static func backend(variant: NFKMLXSeedXVariant, revision: String?, cacheDirectoryURL: URL?,
                               precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: variant.repository, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles), precision: precision)
    }

    /// The asynchronous form of ``backend(variant:revision:cacheDirectoryURL:precision:)``.
    @objc(backendWithVariant:revision:cacheDirectoryURL:precision:completionHandler:)
    public static func backend(variant: NFKMLXSeedXVariant, revision: String?, cacheDirectoryURL: URL?,
                               precision: NFKMLXWeightPrecision,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL, precision: precision)
        }
    }

    /// Registers `seed-x` with `NFKMLXModelRegistry`; the registry's URL is the release directory, loaded
    /// at its bfloat16.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in
            guard let url else { throw NFKMLXError.unsupportedConfiguration("seed-x builds from a release directory, not without weights") }
            return try backend(directoryURL: url, precision: .checkpoint)
        }
    }

    /// Adapts a Seed-X decoder's attention query and value projections with LoRA on translation pairs,
    /// through the language family's recipe: each example is the prompt's ids from
    /// ``NFKMLXSeedXTranslator/promptTokens(text:sourceCode:targetCode:)`` and the target's from
    /// ``NFKMLXSeedXTranslator/targetTokens(translation:)``, and the loss scores the target alone, as the
    /// release's reward-model demonstration lays a pair out. Adapt a `.float32` load that holds every
    /// layer; a streamed decoder throws. Returns each step's loss.
    ///
    /// - Parameters:
    ///   - net: the decoder, from ``translator(directoryURL:precision:residency:)`` at `.float32`.
    ///   - examples: the prompt and target ids for step `step`.
    ///   - rank: the adapters' rank.
    ///   - alpha: the adapters' scale numerator.
    ///   - optimizer: the update rule; nil uses the family's AdamW at 1e-4.
    ///   - steps: how many updates to take.
    ///   - clipGradientNorm: bounds the global gradient norm before each update.
    ///   - accumulationSteps: how many examples each update averages; `steps` counts updates. 1, the
    ///     default, updates after every example.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXLanguageNet,
        examples: (Int) -> (prompt: [Int], target: [Int]),
        rank: Int? = 8,
        alpha: Float = 16,
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        accumulationSteps: Int = 1,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXLanguage.fineTune(
            net, examples: { step in
                let (prompt, target) = examples(step)
                return (MLXArray((prompt + target).map { Int32($0) }), prompt.count)
            }, rank: rank, alpha: alpha, optimizer: optimizer, steps: steps, clipGradientNorm: clipGradientNorm,
            accumulationSteps: accumulationSteps, checkpoint: checkpoint, observer: observer)
    }
}
