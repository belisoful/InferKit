//
//  NFKMLXRepositoryMatchingTranslation.swift
//  InferKitMLX
//
//  The translators' judgments of another repository: Marian, M2M-100 (and SMaLL-100), and NLLB-200, each
//  read by the shared encoder-decoder reader and its family's tokenizer files, and HY-MT, read by the
//  language decoder's readers. M2M-100 and NLLB-200 share the `m2m_100` model type, so each judge takes
//  only its own tokenizer.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXFamilyJudge {

    /// Marian (`marian`), read by ``NFKMLXSeq2SeqConfiguration``, its tokenizer from `source.spm`,
    /// `target.spm`, and the one `vocab.json` both sides share.
    static let marian = NFKMLXFamilyJudge(
        reads: { $0.family == "marian" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            try verifySeq2SeqRepository(repository, template: template,
                                        files: NFKMLXMarian.requiredFiles.filter { $0 != "config.json" })
            let name = repository.repository
            let tokenizer = try repository.jsonObjectOfFile(atPath: "tokenizer_config.json") as? [String: Any]
            if (tokenizer?["separate_vocabs"] as? NSNumber)?.boolValue == true || repository.file(atPath: "target_vocab.json") != nil {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                    "\(name) keeps a separate target vocabulary, where the loader reads one vocab.json for both sides")
            }
            return Reading()
        })

    /// M2M-100 and SMaLL-100 (`m2m_100` with a `vocab.json` and the M2M-100 tokenizer), read by
    /// ``NFKMLXSeq2SeqConfiguration``. A repository shipping SMaLL-100's tokenizer code takes the SMaLL-100
    /// entry, whose loader puts the target language on the source, and any other takes M2M-100's.
    static let m2m100 = NFKMLXFamilyJudge(
        reads: { $0.family == "m2m100" && !$0.modelTypes.isEmpty },
        template: { repository, template in
            let small100 = repository.file(atPath: "tokenization_small100.py") != nil
            return NFKMLXModelDescriptors.entries.first { $0.identifier == (small100 ? "small100" : "m2m100") } ?? template
        },
        verify: { repository, template in
            try verifySeq2SeqRepository(repository, template: template,
                                        files: NFKMLXM2M100.requiredFiles.filter { $0 != "config.json" })
            try verifyTokenizerClass(of: repository, among: ["M2M100Tokenizer"], required: false)
            return Reading()
        })

    /// NLLB-200 (`m2m_100` with the NLLB tokenizer), read by ``NFKMLXSeq2SeqConfiguration``, its tokenizer
    /// from `sentencepiece.bpe.model` and the language codes `special_tokens_map.json` lists.
    static let nllb = NFKMLXFamilyJudge(
        reads: { $0.family == "nllb" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            try verifySeq2SeqRepository(repository, template: template,
                                        files: NFKMLXNLLB.requiredFiles.filter { $0 != "config.json" })
            try verifyTokenizerClass(of: repository, among: ["NllbTokenizer", "NllbTokenizerFast"], required: true)
            return Reading()
        })

    /// HY-MT (`hunyuan_v1_dense`), read by the language decoder's configuration reader and tokenizer, with
    /// one of the two chat templates' markers among `tokenizer.json`'s added tokens. A repository reads
    /// through Hy-MT2's entry, whose instruction is the current generation's; the two generations share
    /// their tokenizers and templates.
    static let hymt = NFKMLXFamilyJudge(
        reads: { $0.family == "hy-mt" && !$0.modelTypes.isEmpty },
        template: { _, template in
            NFKMLXModelDescriptors.entries.first { $0.identifier == NFKMLXHYMTRelease.hyMT2.modelName } ?? template
        },
        verify: { repository, template in
            let configuration = repository.configuration
            let name = repository.repository
            _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXLanguage.configuration(fromJSON: configuration)
            }
            try NFKMLXRepositoryJudgment.verifyUnreadFields(of: configuration, repository: name)
            let quantization = try NFKMLXRepositoryJudgment.verifyQuantization(of: configuration, repository: name)
            try NFKMLXRepositoryJudgment.verifyTokenizer(of: repository)
            let tokenizer = try repository.jsonObjectOfFile(atPath: "tokenizer.json") as? [String: Any]
            let added = Set(((tokenizer?["added_tokens"] as? [[String: Any]]) ?? []).compactMap { $0["content"] as? String })
            guard hymtTemplateMarkers.contains(where: added.isSuperset(of:)) else {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                    "the tokenizer of \(name) holds neither HY-MT chat template's markers")
            }
            return Reading(quantization: quantization)
        })

    /// The markers each HY-MT chat template opens and closes a user turn with: the 1.8B releases', then the
    /// 7B releases'.
    static let hymtTemplateMarkers: [Set<String>] = [
        ["<｜hy_begin▁of▁sentence｜>", "<｜hy_User｜>", "<｜hy_Assistant｜>", "<｜hy_place▁holder▁no▁8｜>"],
        ["<|startoftext|>", "<|extra_0|>"],
    ]

    /// The checks every encoder-decoder translator shares: float weights, the files its tokenizer reads,
    /// the shared reader, and the fields that reader does not read.
    private static func verifySeq2SeqRepository(_ repository: NFKModelRepository, template: NFKMLXCatalogEntry,
                                                files: [String]) throws {
        let configuration = repository.configuration
        let name = repository.repository
        try NFKMLXRepositoryJudgment.verifyFloatWeights(of: configuration, repository: name)
        if let missing = files.first(where: { repository.file(atPath: $0) == nil }) {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer, "\(name) has no \(missing), which the loader reads")
        }
        _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
            try NFKMLXSeq2SeqConfiguration(huggingFaceConfig: configuration)
        }
        try verifySeq2SeqUnreadFields(of: configuration, repository: name)
    }

    /// Refuses the encoder-decoder fields ``NFKMLXSeq2SeqConfiguration`` does not read, stated at values the
    /// output depends on: a decoder head count other than the encoder's (both stacks take the encoder's),
    /// and embeddings the encoder and decoder do not share.
    static func verifySeq2SeqUnreadFields(of configuration: [String: Any], repository: String) throws {
        if let decoder = configuration["decoder_attention_heads"] as? NSNumber,
           let encoder = configuration["encoder_attention_heads"] as? NSNumber, decoder != encoder {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                "\(repository) runs \(decoder) decoder heads beside \(encoder) encoder heads, where the loader builds one count")
        }
        let shared = (configuration["share_encoder_decoder_embeddings"] as? NSNumber)?.boolValue ?? true
        let decoderVocabulary = configuration["decoder_vocab_size"] as? NSNumber
        if !shared || (decoderVocabulary != nil && decoderVocabulary != configuration["vocab_size"] as? NSNumber) {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                "\(repository) gives its decoder its own embeddings, where the loader shares the encoder's")
        }
    }

    /// Refuses a `tokenizer_config.json` tokenizer class outside `classes`, and, where `required`, one
    /// stating none.
    static func verifyTokenizerClass(of repository: NFKModelRepository, among classes: Set<String>,
                                     required: Bool) throws {
        let name = repository.repository
        var stated: String?
        if repository.file(atPath: "tokenizer_config.json") != nil {
            stated = (try repository.jsonObjectOfFile(atPath: "tokenizer_config.json") as? [String: Any])?["tokenizer_class"] as? String
        }
        if let stated, !classes.contains(stated) {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                "\(name) tokenizes with \(stated), where the loader reads " + classes.sorted().joined(separator: " or "))
        }
        if stated == nil && required {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                "\(name) names no tokenizer class, and the loader reads only " + classes.sorted().joined(separator: " or "))
        }
    }
}
