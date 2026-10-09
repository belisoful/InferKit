//
//  NFKMLXNLLB.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX

// NLLB-200, Meta's translator over 202 languages (CC-BY-NC-4.0 weights): the M2M-100 network over a
// 256k SentencePiece BPE vocabulary, its ids one past the model file's because fairseq's four specials
// lead, with a `xxx_Xxxx` language code (ISO 639-3 and script) leading the source and forced as the
// decoder's first token.

/// The released NLLB-200 sizes.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXNLLBVariant)
public enum NFKMLXNLLBVariant: Int {
    /// `facebook/nllb-200-distilled-600M`: 12 + 12 layers at 1024.
    case distilled600M
    /// `facebook/nllb-200-1.3B`: 24 + 24 layers at 1024.
    case m1_3B
    /// `facebook/nllb-200-distilled-1.3B`: the 1.3B geometry, distilled.
    case distilled1_3B
    /// `facebook/nllb-200-3.3B`: 24 + 24 layers at 2048.
    case m3_3B

    var repo: String {
        switch self {
        case .distilled600M: return "facebook/nllb-200-distilled-600M"
        case .m1_3B: return "facebook/nllb-200-1.3B"
        case .distilled1_3B: return "facebook/nllb-200-distilled-1.3B"
        case .m3_3B: return "facebook/nllb-200-3.3B"
        }
    }

    var name: String {
        switch self {
        case .distilled600M: return "nllb-200"
        case .m1_3B: return "nllb-200-1.3b"
        case .distilled1_3B: return "nllb-200-distilled-1.3b"
        case .m3_3B: return "nllb-200-3.3b"
        }
    }
}

/// A loaded NLLB-200 release.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXNLLBTranslator: NFKMLXTranslator {
    public let net: NFKMLXSeq2SeqNet
    public let tokenizer: NFKMLXSentencePieceTokenizer
    /// The language-code marker ids by NLLB code (`eng_Latn`).
    public let languageIds: [String: Int]
    public let identifier: String
    public var defaultDecoding: NFKMLXSeq2SeqDecoding

    /// The 202 language codes, in the order that numbers their markers after the vocabulary
    /// (`FAIRSEQ_LANGUAGE_CODES` of transformers' `NllbTokenizer`; a release's `special_tokens_map.json`
    /// lists the same).
    public static let languages = [
        "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn", "ajp_Arab", "aka_Latn", "amh_Ethi",
        "apc_Arab", "arb_Arab", "ars_Arab", "ary_Arab", "arz_Arab", "asm_Beng", "ast_Latn", "awa_Deva", "ayr_Latn",
        "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl", "bem_Latn", "ben_Beng", "bho_Deva",
        "bjn_Arab", "bjn_Latn", "bod_Tibt", "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn",
        "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt",
        "ell_Grek", "eng_Latn", "epo_Latn", "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "pes_Arab", "fij_Latn",
        "fin_Latn", "fon_Latn", "fra_Latn", "fur_Latn", "fuv_Latn", "gla_Latn", "gle_Latn", "glg_Latn", "grn_Latn",
        "guj_Gujr", "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn",
        "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn", "jpn_Jpan", "kab_Latn", "kac_Latn",
        "kam_Latn", "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor", "knc_Arab", "knc_Latn", "kaz_Cyrl", "kbp_Latn",
        "kea_Latn", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn", "kon_Latn", "kor_Hang", "kmr_Latn",
        "lao_Laoo", "lvs_Latn", "lij_Latn", "lim_Latn", "lin_Latn", "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn",
        "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn", "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Latn",
        "mkd_Cyrl", "plt_Latn", "mlt_Latn", "mni_Beng", "khk_Cyrl", "mos_Latn", "mri_Latn", "zsm_Latn", "mya_Mymr",
        "nld_Latn", "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn", "nya_Latn", "oci_Latn", "gaz_Latn",
        "ory_Orya", "pag_Latn", "pan_Guru", "pap_Latn", "pol_Latn", "por_Latn", "prs_Arab", "pbt_Arab", "quy_Latn",
        "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Beng", "scn_Latn", "shn_Mymr", "sin_Sinh",
        "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn", "als_Latn",
        "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn", "szl_Latn", "tam_Taml", "tat_Cyrl",
        "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi", "taq_Latn", "taq_Tfng", "tpi_Latn", "tsn_Latn",
        "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng", "uig_Arab", "ukr_Cyrl", "umb_Latn",
        "urd_Arab", "uzn_Latn", "vec_Latn", "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn",
        "yue_Hant", "zho_Hans", "zho_Hant", "zul_Latn",
    ]

    /// The individual language NLLB names where BCP-47 uses a macrolanguage or a different ISO 639-3
    /// code: Standard Arabic, Iranian Persian, Southern Pashto, North Azerbaijani, Tosk Albanian, Coastal
    /// Swahili, Halh Mongolian, Standard Latvian, Northern Uzbek, Plateau Malagasy, Nepali, West Central
    /// Oromo, Northern Kurdish, Standard Malay, Norwegian Bokmål, Nigerian Fulfulde, Central Aymara,
    /// Ayacucho Quechua, Eastern Yiddish, Dari.
    public static let individualLanguage: [String: String] = [
        "ar": "arb", "fa": "pes", "ps": "pbt", "az": "azj", "sq": "als", "sw": "swh", "mn": "khk", "lv": "lvs",
        "uz": "uzn", "mg": "plt", "ne": "npi", "om": "gaz", "ku": "kmr", "ms": "zsm", "no": "nob", "ff": "fuv",
        "ay": "ayr", "qu": "quy", "yi": "ydd", "prs": "prs",
    ]

    init(net: NFKMLXSeq2SeqNet, tokenizer: NFKMLXSentencePieceTokenizer, languageIds: [String: Int],
         identifier: String, beams: Int, maxTokens: Int) {
        self.net = net
        self.tokenizer = tokenizer
        self.languageIds = languageIds
        self.identifier = identifier
        let c = net.configuration
        defaultDecoding = NFKMLXSeq2SeqDecoding(beams: beams, maxTokens: maxTokens, earlyStopping: true,
                                                startToken: c.decoderStartTokenId, endToken: c.eosTokenId)
    }

    public var fixedSourceLanguage: String? { nil }
    public var fixedTargetLanguage: String? { nil }

    /// The NLLB code for a BCP-47 tag: the ISO 639-3 code (NLLB's individual language where the tag is
    /// a macrolanguage) with the tag's script; without a script subtag, the script the tag implies or
    /// the one script the release has for that language. Nil when the release has no such language,
    /// lacks the script the tag names, or names the language in two scripts and the tag picks neither.
    public func code(for language: String) -> String? {
        let primary = NFKMLXTranslationBackend.primary(language)
        let named = NFKMLXTranslationBackend.script(language)
        let script = named ?? NFKMLXTranslationBackend.impliedScript(language)
        var bases = [String]()
        if let individual = Self.individualLanguage[primary] { bases.append(individual) }
        if let three = NFKMLXLanguageCodes.iso639_3[primary] { bases.append(three) }
        bases.append(primary)
        for base in bases {
            if let script, languageIds["\(base)_\(script)"] != nil { return "\(base)_\(script)" }
            guard named == nil else { continue }
            let matches = Self.languages.filter { $0.hasPrefix(base + "_") && languageIds[$0] != nil }
            if matches.count == 1 { return matches[0] }
        }
        return nil
    }

    public func supports(language: String) -> Bool { code(for: language) != nil }

    /// The source ids: the language marker, the pieces, and the end token.
    public func sourceIds(for text: String, source: String) -> [Int] {
        var ids = [languageIds[code(for: source) ?? source] ?? net.configuration.eosTokenId]
        ids += tokenizer.encode(text, dummyPrefix: nil)
        ids.append(net.configuration.eosTokenId)
        return ids
    }

    /// The text generated ids spell: markers and the four specials left out, as the reference's
    /// `skip_special_tokens` decode leaves them, and the result trimmed as it trims.
    public func text(of ids: [Int]) -> String {
        let markers = Set(languageIds.values)
        let plain = ids.filter { $0 >= 4 && !markers.contains($0) }
        return tokenizer.decode(ids: plain).trimmingCharacters(in: .whitespaces)
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
        var decoding = decoding
        decoding.forcedFirstToken = languageIds[targetCode]
        let generated = NFKMLXSeq2SeqDecoder.generate(net, source: sourceIds(for: text, source: sourceCode), decoding: decoding)
        return self.text(of: generated)
    }
}

/// Registration, download, and construction of NLLB-200 translation backends.
///
/// @discussion A release directory holds `config.json`, `sentencepiece.bpe.model`,
/// `special_tokens_map.json`, and `pytorch_model.bin` (the 3.3B in shards under
/// `pytorch_model.bin.index.json`), the layout of every `facebook/nllb-200-*` repo. The release numbers
/// its vocabulary one past the model file: `<s>`, `<pad>`, `</s>`, `<unk>` take 0 to 3, every other
/// piece its model id plus one, the 202 language codes follow the vocabulary in the order the release
/// lists them, and `<mask>` closes. The decoder starts from `</s>` with the target code forced first.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXNLLB)
public final class NFKMLXNLLB: NSObject {
    /// The registered name of the distilled 600M.
    @objc public static let modelName = "nllb-200"
    static let requiredFiles = ["config.json", "sentencepiece.bpe.model", "special_tokens_map.json"]
    static let optionalFiles = ["generation_config.json", "tokenizer_config.json"]
    static let weightFiles = ["model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json"]

    /// Loads a release directory as a translator.
    public static func translator(directoryURL directory: URL, variant: NFKMLXNLLBVariant = .distilled600M) throws -> NFKMLXNLLBTranslator {
        try translator(net: try network(directoryURL: directory), directoryURL: directory, variant: variant)
    }

    /// Wraps a network with the release's tokenizer.
    public static func translator(net: NFKMLXSeq2SeqNet, directoryURL directory: URL,
                                  variant: NFKMLXNLLBVariant = .distilled600M) throws -> NFKMLXNLLBTranslator {
        let segmenter = try NFKMLXSentencePieceSegmenter(contentsOf: directory.appendingPathComponent("sentencepiece.bpe.model"))
        let (entries, languageIds) = releaseTable(segmenter: segmenter, languages: languageCodes(inDirectory: directory))
        let tokenizer = NFKMLXSentencePieceTokenizer(segmenter: segmenter, vocabularyEntries: entries, unknownToken: "<unk>",
                                                     eosTokenId: net.configuration.eosTokenId, bosTokenId: 0)
        let generation = (try? JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("generation_config.json")))) as? [String: Any]
        let config = (try? JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("config.json")))) as? [String: Any]
        // The releases name no beam count, and transformers' generation default is 1: the reference
        // decodes greedily. NFKMLXTranslationParameterKey.beamCount turns on a beam search per request.
        let beams = (generation?["num_beams"] as? NSNumber)?.intValue ?? (config?["num_beams"] as? NSNumber)?.intValue ?? 1
        let maxTokens = (generation?["max_length"] as? NSNumber)?.intValue ?? (config?["max_length"] as? NSNumber)?.intValue ?? 200
        return NFKMLXNLLBTranslator(net: net, tokenizer: tokenizer, languageIds: languageIds, identifier: variant.name,
                                    beams: beams, maxTokens: maxTokens)
    }

    /// The release's piece-to-id table and marker ids over a model file: the four fairseq specials,
    /// every model piece from its third at its id plus one, the language codes, then `<mask>`.
    static func releaseTable(segmenter: NFKMLXSentencePieceSegmenter,
                             languages: [String]) -> (entries: [(piece: String, id: Int)], languageIds: [String: Int]) {
        var entries: [(piece: String, id: Int)] = [("<s>", 0), ("<pad>", 1), ("</s>", 2), ("<unk>", 3)]
        let count = segmenter.pieceCount
        for id in 3 ..< count {
            if let piece = segmenter.piece(at: id) { entries.append((piece, id + 1)) }
        }
        var languageIds = [String: Int]()
        for (index, code) in languages.enumerated() {
            languageIds[code] = count + 1 + index
            entries.append((code, count + 1 + index))
        }
        entries.append(("<mask>", count + 1 + languages.count))
        return (entries, languageIds)
    }

    /// The language codes a release lists (`special_tokens_map.json`'s `additional_special_tokens`), or
    /// the tokenizer's own list when the file names none.
    static func languageCodes(inDirectory directory: URL) -> [String] {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("special_tokens_map.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let codes = json["additional_special_tokens"] as? [String], !codes.isEmpty else {
            return NFKMLXNLLBTranslator.languages
        }
        return codes
    }

    /// Builds the network alone, ready to adapt. A release directory supplies its own geometry from its
    /// `config.json` and its weights. Without a directory the network takes `configuration` (the tiny
    /// test geometry when nil) and random weights. A `configuration` passed with a directory overrides
    /// the release's `config.json`.
    public static func network(directoryURL directory: URL?,
                               configuration: NFKMLXSeq2SeqConfiguration? = nil) throws -> NFKMLXSeq2SeqNet {
        guard let directory else {
            return NFKMLXSeq2SeqNet(configuration ?? .tinyM2M100)
        }
        let net = NFKMLXSeq2SeqNet(try configuration ?? NFKMLXSeq2SeqConfiguration(
            huggingFaceConfigURL: directory.appendingPathComponent("config.json")))
        try net.loadWeights(fromDirectory: directory)
        return net
    }

    /// Builds the distilled 600M backend from a release directory.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(variant: .distilled600M, directoryURL: directoryURL)
    }

    /// Builds a variant's backend from a release directory.
    @objc(backendWithVariant:directoryURL:error:)
    public static func backend(variant: NFKMLXNLLBVariant, directoryURL: URL) throws -> any NFKInferenceBackend {
        NFKMLXTranslationBackend(translator: try translator(directoryURL: directoryURL, variant: variant))
    }

    /// Downloads a variant's release and builds the backend. Blocking on the network.
    @objc(backendWithVariant:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXNLLBVariant, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: variant, directoryURL: try NFKMLXReleaseDownload.directory(
            repo: variant.repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``backend(variant:revision:cacheDirectoryURL:)``.
    @objc(backendWithVariant:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXNLLBVariant, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) { try backend(variant: variant, revision: revision, cacheDirectoryURL: cacheDirectoryURL) }
    }

    /// Registers `nllb-200` (the distilled 600M) with `NFKMLXModelRegistry`; the registry's URL is the
    /// release directory.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in
            guard let url else { throw NFKMLXError.unsupportedConfiguration("nllb-200 builds from a release directory, not without weights") }
            return try backend(variant: .distilled600M, directoryURL: url)
        }
    }
}
