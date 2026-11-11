//
//  NFKMLXRepositoryMatchingAudio.swift
//  InferKitMLX
//
//  The speech encoders' judgments of another repository: Wav2Vec2 and HuBERT, and W2V-BERT, each read by
//  the family's own configuration and vocabulary readers.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXFamilyJudge {

    /// Wav2Vec2 and HuBERT (`wav2vec2`, `hubert`), read by ``NFKMLXWav2Vec2Configuration``, a CTC release's
    /// vocabulary by ``NFKMLXWav2Vec2Tokenizer``. The loader refuses a missing or an extra parameter, so the
    /// judgment refuses the heads it does not build, and a CTC vocabulary its head cannot index.
    static let wav2Vec2 = NFKMLXFamilyJudge(
        reads: { ["wav2vec2", "hubert"].contains($0.family) && !$0.modelTypes.isEmpty },
        template: { repository, template in
            NFKMLXFamilyJudge.speechTemplate(of: repository, among: NFKMLXModelDescriptors.entries
                .filter { $0.family == template.family && !$0.modelTypes.isEmpty }) ?? template
        },
        verify: { repository, template in
            try verifySpeechRepository(repository, suffixes: ["ForCTC", "Model", "ForPreTraining"])
            let directory = try NFKMLXRepositoryJudgment.snapshot(
                of: repository, holding: ["config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json"])
            let configuration = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXWav2Vec2Configuration(configurationURL: directory.appendingPathComponent("config.json"))
            }
            try verifyCTCVocabulary(of: repository, in: directory, headSize: configuration.vocabularySize)
            return Reading()
        },
        weightFiles: NFKMLXRepositoryJudgment.singleWeightFile("model.safetensors"))

    /// W2V-BERT 2.0 (`wav2vec2-bert`), read by ``NFKMLXWav2Vec2BertConfiguration``, a fine-tuned CTC head's
    /// vocabulary by ``NFKMLXWav2Vec2Tokenizer``.
    static let w2vBert = NFKMLXFamilyJudge(
        reads: { $0.family == "w2v-bert" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            try verifySpeechRepository(repository, suffixes: ["ForCTC", "Model"])
            let directory = try NFKMLXRepositoryJudgment.snapshot(
                of: repository, holding: ["config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json"])
            let configuration = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXWav2Vec2BertConfiguration(configurationURL: directory.appendingPathComponent("config.json"))
            }
            try verifyCTCVocabulary(of: repository, in: directory,
                                    headSize: configuration.vocabularySize.flatMap { $0 > 0 ? $0 : nil })
            return Reading()
        },
        weightFiles: NFKMLXRepositoryJudgment.singleWeightFile("model.safetensors"),
        optionalFiles: ["vocab.json": NFKModelFileRoleTokenizer, "tokenizer_config.json": NFKModelFileRoleTokenizer])

    /// Nemotron ASR Streaming (`nemotron_asr_streaming`), read by ``NFKMLXNemotronASRConfiguration`` and
    /// ``NFKMLXNemotronASRTokenizer``. The loader builds a SiLU conformer, a ReLU joint, a 3 × 3 stride-2
    /// causal subsampler, and a plain transducer, so the judgment refuses another activation, kernel, or
    /// stride, grouped key/value heads, a token-and-duration head, and a tokenizer model the reader does
    /// not decode.
    static let nemotronASR = NFKMLXFamilyJudge(
        reads: { $0.family == "nemotron-asr" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            try verifySpeechRepository(repository, suffixes: ["ForRNNT"])
            let directory = try NFKMLXRepositoryJudgment.snapshot(of: repository, holding: ["config.json", "tokenizer.json"])
            let configuration = repository.configuration
            let encoder = configuration["encoder_config"] as? [String: Any] ?? [:]
            func refuse(_ reason: String) -> NSError {
                NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration, "\(repository.repository): \(reason)")
            }
            if let act = encoder["hidden_act"] as? String, act != "silu" {
                throw refuse("the encoder's activation is \(act); the loader builds SiLU")
            }
            if let act = configuration["hidden_act"] as? String, act != "relu" {
                throw refuse("the joint's activation is \(act); the loader builds ReLU")
            }
            if (encoder["subsampling_conv_kernel_size"] as? Int ?? 3) != 3 || (encoder["subsampling_conv_stride"] as? Int ?? 2) != 2 {
                throw refuse("the subsampler is not 3 × 3 at stride 2, which fixes the stream's chunk geometry")
            }
            if let kv = encoder["num_key_value_heads"] as? Int, kv != (encoder["num_attention_heads"] as? Int ?? kv) {
                throw refuse("the attention groups its key/value heads, which the loader does not")
            }
            if let durations = configuration["durations"] as? [Any], !durations.isEmpty {
                throw refuse("the head predicts durations; the loader decodes a plain transducer")
            }
            let read = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXNemotronASRConfiguration(json: configuration)
            }
            let tokenizerData = try Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))
            let model = ((try? JSONSerialization.jsonObject(with: tokenizerData)) as? [String: Any])?["model"] as? [String: Any]
            guard let type = model?["type"] as? String, ["BPE", "Unigram"].contains(type) else {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                    "\(repository.repository)'s tokenizer.json is not a BPE or Unigram model the reader decodes")
            }
            let tokenizer = try NFKMLXNemotronASRTokenizer(tokenizerURL: directory.appendingPathComponent("tokenizer.json"))
            guard read.blank == read.vocabulary - 1, tokenizer.pieces.count >= read.blank else {
                throw refuse("the blank is not the head's last logit, or the tokenizer names fewer pieces than the head emits")
            }
            return Reading()
        },
        weightFiles: NFKMLXRepositoryJudgment.singleWeightFile("model.safetensors"))

    /// The family's entry with the head the repository carries: a CTC release for a `…ForCTC` architecture,
    /// else an encoder release. nil where the family has no such entry.
    static func speechTemplate(of repository: NFKModelRepository,
                               among entries: [NFKMLXCatalogEntry]) -> NFKMLXCatalogEntry? {
        let transcribes = (repository.configuration["architectures"] as? [String] ?? []).contains { $0.hasSuffix("ForCTC") }
        return entries.first { $0.tasks.contains(NFKModelTaskTranscription) == transcribes }
    }

    /// The checks every speech encoder shares: float weights, and a head the loader builds.
    private static func verifySpeechRepository(_ repository: NFKModelRepository, suffixes: [String]) throws {
        try NFKMLXRepositoryJudgment.verifyFloatWeights(of: repository.configuration, repository: repository.repository)
        try NFKMLXRepositoryJudgment.verifyArchitecture(of: repository.configuration, repository: repository.repository,
                                                        suffixes: suffixes)
    }

    /// Refuses a CTC head whose vocabulary the reader does not read as the release defines it: a missing
    /// or unreadable `vocab.json`, a tokenizer class other than the character-level CTC tokenizer, and an id
    /// past the head's width.
    private static func verifyCTCVocabulary(of repository: NFKModelRepository, in directory: URL,
                                            headSize: Int?) throws {
        guard let headSize else { return }
        let name = repository.repository
        guard repository.file(atPath: "vocab.json") != nil else {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                "\(name) has a CTC head and no vocab.json, which the transcript is decoded with")
        }
        let tokenizerClass = (try? repository.jsonObjectOfFile(atPath: "tokenizer_config.json") as? [String: Any])?
            .flatMap { $0["tokenizer_class"] as? String }
        if let tokenizerClass, tokenizerClass != "Wav2Vec2CTCTokenizer" {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                "\(name) decodes with \(tokenizerClass), where the reader maps each CTC id to its vocab.json token")
        }
        let tokenizer: NFKMLXWav2Vec2Tokenizer
        do {
            tokenizer = try NFKMLXWav2Vec2Tokenizer(directoryURL: directory)
        } catch {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                "the vocab.json of \(name) is not one the reader takes: " + error.localizedDescription)
        }
        if tokenizer.tokens.count > headSize {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                "the vocab.json of \(name) holds ids past the \(headSize) its CTC head scores")
        }
    }
}
