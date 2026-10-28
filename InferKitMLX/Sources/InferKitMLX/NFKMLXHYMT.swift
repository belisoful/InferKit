//
//  NFKMLXHYMT.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

// HY-MT (Tencent): dense Hunyuan decoders trained to translate among 38 languages, prompted with a
// plain instruction inside the release's chat template. The network is the shipped language decoder
// (`NFKMLXLanguageNet`), which reads Hunyuan's dense configuration here; this file adds the prompt,
// the language table, the translator face, and the fine-tuning recipe over the language family's.

/// How the language decoder reads Hunyuan's dense configuration (`hunyuan_v1_dense`).
///
/// @discussion The block is Llama's with one addition: each attention head's queries and keys are
/// RMS-normalized after the rotary embedding (`use_qk_norm`, under the checkpoint names
/// `query_layernorm` and `key_layernorm`). A `dynamic` `rope_scaling` that states `alpha` raises the
/// rotary base to `rope_theta · alpha^(d / (d − 2))` for head width `d`, as transformers'
/// `HunYuanDenseV1RotaryEmbedding` does. Transformers rescales that rotary again only once a sequence
/// passes `max_position_embeddings` (262,144 in the HY-MT releases), which is the window the reader
/// states. Cross-layer attention (`use_cla`), a norm other than RMS, and a model without rotary
/// embeddings are refused.
enum NFKMLXHunyuanDense {
    static let modelType = "hunyuan_v1_dense"

    static func read(_ json: [String: Any], into configuration: inout NFKMLXLanguageConfiguration) throws {
        func flag(_ key: String) -> Bool? { (json[key] as? NSNumber)?.boolValue }
        if flag("use_cla") == true {
            throw NFKMLXError.unsupportedConfiguration(
                "the config shares keys and values across layers (use_cla), which this network does not implement")
        }
        if let norm = json["norm_type"] as? String, norm != "rms" {
            throw NFKMLXError.unsupportedConfiguration("the config normalizes with \(norm), where this network reads RMS norms")
        }
        if flag("use_rotary_pos_emb") == false {
            throw NFKMLXError.unsupportedConfiguration("the config turns the rotary embedding off, which this network always applies")
        }
        configuration.normalizesRotatedQueryAndKey = flag("use_qk_norm") ?? false
        configuration.ropeTheta = try rotaryBase(json, headDimensions: configuration.headDimensions)
    }

    /// The rotary base a Hunyuan configuration states, rounded to float32 once, as transformers raises
    /// a Python float to the exponents.
    static func rotaryBase(_ json: [String: Any], headDimensions: Int) throws -> Float {
        let theta = (json["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000
        guard let scaling = json["rope_scaling"] as? [String: Any] else { return Float(theta) }
        let name = ((scaling["rope_type"] ?? scaling["type"]) as? String)?.lowercased() ?? "default"
        if name == "default" { return Float(theta) }
        guard name == "dynamic", let alpha = (scaling["alpha"] as? NSNumber)?.doubleValue, alpha > 0, headDimensions > 2 else {
            throw NFKMLXError.unsupportedConfiguration(
                "rope_scaling type '\(name)' without an alpha is not Hunyuan's scaled base, which is all this reader takes")
        }
        let width = Double(headDimensions)
        return Float(theta * pow(alpha, width / (width - 2)))
    }
}

/// The two HY-MT generations, which share their geometry and tokenizers and differ in the instruction
/// their model cards prompt with. Introduced in InferKit 0.4.0.
@objc(NFKMLXHYMTRelease)
public enum NFKMLXHYMTRelease: Int, Sendable {
    /// HY-MT1.5 (December 2025), under the Tencent HY Community License.
    case hyMT15
    /// Hy-MT2 (May 2026), under Apache-2.0.
    case hyMT2

    /// The registered name of the generation's translator.
    public var modelName: String {
        switch self {
        case .hyMT15: return "hy-mt1.5"
        case .hyMT2: return "hy-mt2"
        }
    }
}

/// The four HY-MT translation releases. Introduced in InferKit 0.4.0.
@objc(NFKMLXHYMTVariant)
public enum NFKMLXHYMTVariant: Int, Sendable, CaseIterable {
    /// `tencent/HY-MT1.5-1.8B`.
    case hyMT15_1_8B
    /// `tencent/HY-MT1.5-7B`.
    case hyMT15_7B
    /// `tencent/Hy-MT2-1.8B`.
    case hyMT2_1_8B
    /// `tencent/Hy-MT2-7B`.
    case hyMT2_7B

    /// The variant's Hugging Face repository.
    public var repository: String {
        switch self {
        case .hyMT15_1_8B: return "tencent/HY-MT1.5-1.8B"
        case .hyMT15_7B: return "tencent/HY-MT1.5-7B"
        case .hyMT2_1_8B: return "tencent/Hy-MT2-1.8B"
        case .hyMT2_7B: return "tencent/Hy-MT2-7B"
        }
    }

    /// The generation the variant belongs to, which decides its prompt.
    public var release: NFKMLXHYMTRelease {
        switch self {
        case .hyMT15_1_8B, .hyMT15_7B: return .hyMT15
        case .hyMT2_1_8B, .hyMT2_7B: return .hyMT2
        }
    }
}

/// A loaded HY-MT release.
///
/// @discussion The prompt is the model card's default instruction for the generation, naming only the
/// target language: the Chinese instruction with the language's Chinese name when either language is
/// Chinese (`zh`, `zh-Hant`, `yue`), and the English one with its English name otherwise. The text
/// follows the instruction after one line break in HY-MT1.5 and two in Hy-MT2, as each card's template
/// lays it out. The instruction goes inside the release's chat template, whose form follows the
/// tokenizer: the 1.8B releases open `<｜hy_begin▁of▁sentence｜><｜hy_User｜>` and the 7B releases
/// `<|startoftext|>`, and each ends the user turn as ``opensAssistantTurn`` says. The languages are the
/// 38 the cards list; a BCP-47 tag resolves by its primary subtag, Chinese by its script (`zh-TW` →
/// `zh-Hant`), and `fil` reads as `tl`. Decoding stops at the end tokens the release's
/// `generation_config.json` names and is greedy by default, where the release's generation config
/// samples (temperature 0.7, top-k 20, top-p 0.8, repetition penalty 1.05); `decoding.beams` above 1
/// runs a beam search scored as transformers' `generate` scores one. The translation's text leaves out
/// every special token, as the cards decode (`skip_special_tokens`).
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXHYMTTranslator: NFKMLXTranslator {
    public let net: NFKMLXLanguageNet
    public let tokenizer: NFKTokenizer
    public let release: NFKMLXHYMTRelease
    public let identifier: String
    public var defaultDecoding: NFKMLXSeq2SeqDecoding
    /// Whether the 1.8B template's user turn ends by opening the assistant turn (`<｜hy_Assistant｜>`,
    /// transformers' `add_generation_prompt=True`, the default) rather than with
    /// `<｜hy_place▁holder▁no▁8｜>`. The 7B template ends the user turn with `<|extra_0|>` either way.
    public var opensAssistantTurn: Bool
    /// The template's markers: the ids that open the prompt, and the ids that close the user turn with
    /// the assistant turn opened and without it.
    let opening: [Int]
    let assistantOpening: [Int]
    let closing: [Int]
    /// Every id the release marks special, which the translation's text leaves out.
    let specialTokens: Set<Int>

    /// The releases' languages: tag (`de`) → the English and Chinese names their prompts use.
    public static let languages: [String: (english: String, chinese: String)] = [
        "zh": ("Chinese", "中文"), "en": ("English", "英语"), "fr": ("French", "法语"),
        "pt": ("Portuguese", "葡萄牙语"), "es": ("Spanish", "西班牙语"), "ja": ("Japanese", "日语"),
        "tr": ("Turkish", "土耳其语"), "ru": ("Russian", "俄语"), "ar": ("Arabic", "阿拉伯语"),
        "ko": ("Korean", "韩语"), "th": ("Thai", "泰语"), "it": ("Italian", "意大利语"),
        "de": ("German", "德语"), "vi": ("Vietnamese", "越南语"), "ms": ("Malay", "马来语"),
        "id": ("Indonesian", "印尼语"), "tl": ("Filipino", "菲律宾语"), "hi": ("Hindi", "印地语"),
        "zh-Hant": ("Traditional Chinese", "繁体中文"), "pl": ("Polish", "波兰语"), "cs": ("Czech", "捷克语"),
        "nl": ("Dutch", "荷兰语"), "km": ("Khmer", "高棉语"), "my": ("Burmese", "缅甸语"),
        "fa": ("Persian", "波斯语"), "gu": ("Gujarati", "古吉拉特语"), "ur": ("Urdu", "乌尔都语"),
        "te": ("Telugu", "泰卢固语"), "mr": ("Marathi", "马拉地语"), "he": ("Hebrew", "希伯来语"),
        "bn": ("Bengali", "孟加拉语"), "ta": ("Tamil", "泰米尔语"), "uk": ("Ukrainian", "乌克兰语"),
        "bo": ("Tibetan", "藏语"), "kk": ("Kazakh", "哈萨克语"), "mn": ("Mongolian", "蒙古语"),
        "ug": ("Uyghur", "维吾尔语"), "yue": ("Cantonese", "粤语"),
    ]

    /// The tags whose pairs take the Chinese instruction.
    static let chineseCodes: Set<String> = ["zh", "zh-Hant", "yue"]

    init(net: NFKMLXLanguageNet, tokenizer: NFKTokenizer, release: NFKMLXHYMTRelease, directory: URL) throws {
        let specials = NFKMLXLanguage.specialTokenIDs(inDirectory: directory)
        func marker(_ literal: String) throws -> Int {
            guard let id = specials[literal] else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(directory.lastPathComponent)'s tokenizer has no \(literal), which its chat template opens with")
            }
            return id
        }
        if specials["<｜hy_User｜>"] != nil {
            opening = [try marker("<｜hy_begin▁of▁sentence｜>"), try marker("<｜hy_User｜>")]
            assistantOpening = [try marker("<｜hy_Assistant｜>")]
            closing = [try marker("<｜hy_place▁holder▁no▁8｜>")]
        } else {
            opening = [try marker("<|startoftext|>")]
            assistantOpening = [try marker("<|extra_0|>")]
            closing = assistantOpening
        }
        self.net = net
        self.tokenizer = tokenizer
        self.release = release
        identifier = release.modelName
        specialTokens = Set(specials.values)
        opensAssistantTurn = true
        let ends = Self.endTokens(inDirectory: directory) ?? [tokenizer.eosTokenId]
        defaultDecoding = NFKMLXSeq2SeqDecoding(beams: 1, maxTokens: 512, startToken: opening[0], endToken: ends[0],
                                                additionalEndTokens: Array(ends.dropFirst()))
    }

    /// The end tokens a release's `generation_config.json` names (one id or a list), or nil where it
    /// names none.
    static func endTokens(inDirectory directory: URL) -> [Int]? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("generation_config.json")),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        if let single = json["eos_token_id"] as? NSNumber { return [single.intValue] }
        let list = ((json["eos_token_id"] as? [Any]) ?? []).compactMap { ($0 as? NSNumber)?.intValue }
        return list.isEmpty ? nil : list
    }

    public var fixedSourceLanguage: String? { nil }
    public var fixedTargetLanguage: String? { nil }
    public var layerStreams: [NFKMLXLayerStream] { [net.layerStream].compactMap { $0 } }

    /// The releases' tag for a BCP-47 tag: Chinese by its script, `fil` as `tl`, any other by its
    /// primary subtag, where the releases list it.
    public func code(for language: String) -> String? {
        let primary = NFKMLXTranslationBackend.primary(language)
        if primary == "zh" {
            return NFKMLXTranslationBackend.impliedScript(language) == "Hant" ? "zh-Hant" : "zh"
        }
        let code = primary == "fil" ? "tl" : primary
        return Self.languages[code] != nil ? code : nil
    }

    public func supports(language: String) -> Bool { code(for: language) != nil }

    /// The instruction and text for a target tag, the Chinese instruction when either side is Chinese.
    /// `sourceCode` nil (an undetected source) takes the English instruction unless the target is Chinese.
    public func prompt(text: String, sourceCode: String?, targetCode: String) -> String {
        let names = Self.languages[targetCode] ?? (targetCode, targetCode)
        let chinese = Self.chineseCodes.contains(targetCode) || sourceCode.map(Self.chineseCodes.contains) == true
        let separator = release == .hyMT15 ? "\n" : "\n\n"
        if chinese {
            return "将以下文本翻译为\(names.chinese)，注意只需要输出翻译后的结果，不要额外解释：\(separator)\(text)"
        }
        switch release {
        case .hyMT15:
            return "Translate the following segment into \(names.english), without additional explanation.\(separator)\(text)"
        case .hyMT2:
            return "Translate the following text into \(names.english). Note that you should only output the translated "
                + "result without any additional explanation:\(separator)\(text)"
        }
    }

    /// The prompt's ids: the template's opening, the instruction and text, and the user turn's end.
    public func promptTokens(text: String, sourceCode: String?, targetCode: String) -> [Int] {
        opening + tokenizer.encode(prompt(text: text, sourceCode: sourceCode, targetCode: targetCode)).map(\.intValue)
            + (opensAssistantTurn ? assistantOpening : closing)
    }

    /// The greedy continuation of `ids`, without the end token that ends it. Throws a failed read of a
    /// streamed layer, which the next call reads again.
    public func generate(promptTokens ids: [Int], maxTokens: Int) throws -> [Int] {
        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = maxTokens
        options.stopTokens = Set([defaultDecoding.endToken] + defaultDecoding.additionalEndTokens)
        let produced = net.generate(prompt: ids, options: options)
        do {
            try net.verifyStream()
        } catch {
            net.layerStream?.clearFailure()
            throw error
        }
        return produced
    }

    /// The best continuation of `ids` a `decoding.beams`-wide beam search finds, without the end token
    /// that ends it. Throws a failed read of a streamed layer, which the next call reads again.
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
        guard let targetCode = code(for: target) else {
            throw NFKMLXTranslationBackend.error(.error_InferenceUnsupported, "\(identifier) does not translate into \(target)")
        }
        let ids = promptTokens(text: text, sourceCode: source.flatMap(code(for:)), targetCode: targetCode)
        let produced = decoding.beams > 1 ? try beamSearch(promptTokens: ids, decoding: decoding)
            : try generate(promptTokens: ids, maxTokens: decoding.maxTokens)
        return translation(of: produced)
    }

    /// The translation `ids` spell, without the release's special tokens.
    public func translation(of ids: [Int]) -> String {
        tokenizer.decode(ids.filter { !specialTokens.contains($0) }.map { NSNumber(value: $0) })
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The ids a fine-tune scores for `translation`: its pieces and the end token that closes the
    /// assistant turn.
    public func targetTokens(translation: String) -> [Int] {
        tokenizer.encode(translation).map(\.intValue) + [defaultDecoding.endToken]
    }
}

/// Registration, download, and construction of HY-MT translation backends.
///
/// @discussion A release directory holds `config.json`, `tokenizer.json`, `tokenizer_config.json`, and
/// the weights: the 1.8B releases are one 4.1 GB `model.safetensors` of bfloat16, and the 7B releases
/// 16.1 GB in four shards with their index. At `.checkpoint` precision, the registry's choice, a release
/// loads at its bfloat16; at `.float32` it doubles, which a residency streams where it does not fit
/// (``NFKMLXResidency/automatic`` holds what fits and reads the other layers in their turn). HY-MT1.5 is
/// under the Tencent HY Community License, which does not cover the European Union, the United Kingdom,
/// or South Korea; Hy-MT2 is under Apache-2.0.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXHYMT)
public final class NFKMLXHYMT: NSObject {
    static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    static let optionalFiles = ["generation_config.json"]
    static let weightFiles = ["model.safetensors.index.json", "model.safetensors"]

    /// Loads a release directory as `release`'s translator, held as `residency` says. A streamed decoder
    /// is read from the release on every pass.
    public static func translator(directoryURL directory: URL, release: NFKMLXHYMTRelease,
                                  precision: NFKMLXWeightPrecision = .float32,
                                  residency: NFKMLXResidency = .automatic) throws -> NFKMLXHYMTTranslator {
        let (net, tokenizer) = try NFKMLXLanguage.loadedReleaseWithTokenizer(at: directory, precision: precision,
                                                                            residency: residency, streams: true)
        return try NFKMLXHYMTTranslator(net: net, tokenizer: tokenizer, release: release, directory: directory)
    }

    /// ``translator(directoryURL:release:precision:residency:)`` as a catalog's load options name it:
    /// their precision (the release's bfloat16 when they name none) and residency, and a runtime weight
    /// quantization of the decoder, which a streamed decoder refuses.
    static func translator(directoryURL directory: URL, release: NFKMLXHYMTRelease,
                           loadOptions: NFKMLXLoadOptions) throws -> NFKMLXHYMTTranslator {
        let translator = try translator(directoryURL: directory, release: release,
                                        precision: loadOptions.precision ?? .checkpoint, residency: loadOptions.residency)
        if loadOptions.quantizationBits != nil {
            try translator.net.requireHeldLayers(for: "quantize at load")
        }
        try loadOptions.applyWeightQuantization(to: translator.net)
        return translator
    }

    /// Wraps a decoder, such as one a fine-tune adapted, with the tokenizer and template of the release
    /// it came from.
    public static func translator(decoder: NFKMLXLanguageNet, directoryURL directory: URL,
                                  release: NFKMLXHYMTRelease) throws -> NFKMLXHYMTTranslator {
        guard let tokenizer = NFKMLXLanguage.tokenizer(directoryURL: directory) else {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) has no readable tokenizer")
        }
        return try NFKMLXHYMTTranslator(net: decoder, tokenizer: tokenizer, release: release, directory: directory)
    }

    /// Builds `release`'s backend from a release directory at its bfloat16.
    @objc(backendWithRelease:directoryURL:error:)
    public static func backend(release: NFKMLXHYMTRelease, directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(release: release, directoryURL: directoryURL, precision: .checkpoint, residency: .automatic)
    }

    /// Builds `release`'s backend from a release directory at `precision`.
    @objc(backendWithRelease:directoryURL:precision:error:)
    public static func backend(release: NFKMLXHYMTRelease, directoryURL: URL,
                               precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        try backend(release: release, directoryURL: directoryURL, precision: precision, residency: .automatic)
    }

    /// Builds `release`'s backend from a release directory at `precision`, held as `residency` says.
    @objc(backendWithRelease:directoryURL:precision:residency:error:)
    public static func backend(release: NFKMLXHYMTRelease, directoryURL: URL, precision: NFKMLXWeightPrecision,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        NFKMLXTranslationBackend(translator: try translator(directoryURL: directoryURL, release: release,
                                                            precision: precision, residency: residency))
    }

    /// Downloads a release and builds the backend at `precision`. Blocking on the network.
    @objc(backendWithVariant:revision:cacheDirectoryURL:precision:error:)
    public static func backend(variant: NFKMLXHYMTVariant, revision: String?, cacheDirectoryURL: URL?,
                               precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        try backend(release: variant.release, directoryURL: try NFKMLXReleaseDownload.directory(
            repo: variant.repository, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles), precision: precision)
    }

    /// The asynchronous form of ``backend(variant:revision:cacheDirectoryURL:precision:)``.
    @objc(backendWithVariant:revision:cacheDirectoryURL:precision:completionHandler:)
    public static func backend(variant: NFKMLXHYMTVariant, revision: String?, cacheDirectoryURL: URL?,
                               precision: NFKMLXWeightPrecision,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL, precision: precision)
        }
    }

    /// Registers `hy-mt1.5` and `hy-mt2` with `NFKMLXModelRegistry`; the registry's URL is the release
    /// directory, loaded at its bfloat16.
    @objc public static func register() {
        for release in [NFKMLXHYMTRelease.hyMT15, .hyMT2] {
            NFKMLXModelRegistry.register(name: release.modelName) { url in
                guard let url else {
                    throw NFKMLXError.unsupportedConfiguration("\(release.modelName) builds from a release directory, not without weights")
                }
                return try backend(release: release, directoryURL: url, precision: .checkpoint)
            }
        }
    }

    /// Adapts an HY-MT decoder's attention query and value projections with LoRA on translation pairs,
    /// through the language family's recipe: each example is the prompt's ids from
    /// ``NFKMLXHYMTTranslator/promptTokens(text:sourceCode:targetCode:)`` and the target's from
    /// ``NFKMLXHYMTTranslator/targetTokens(translation:)``, and the loss scores the target alone, as the
    /// release's chat-template fine-tuning lays an exchange out. Adapt a `.float32` load that holds every
    /// layer; a streamed decoder throws. Returns each step's loss.
    ///
    /// - Parameters:
    ///   - net: the decoder, from ``translator(directoryURL:release:precision:residency:)`` at `.float32`.
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
