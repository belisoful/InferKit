//
//  NFKMLXRepositoryMatchingVision.swift
//  InferKitMLX
//
//  The vision-language decoders' judgments of another repository: Qwen3-VL, its decoder read by the dense
//  reader and its tower by the tower's own reader, with the token ids and rotary sections the integration
//  fixes held to the released values.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXFamilyJudge {

    /// Qwen3-VL (`qwen3_vl`, and `qwen3_vl_moe` for the mixture), its `text_config` read by
    /// ``NFKMLXLanguage/configuration(fromJSON:)`` with the dense decoder's refusals, its tower by
    /// `NFKMLXQwen3VLVisionConfiguration.configuration(fromJSON:)`, and its tokenizer by the release reader.
    /// The integration fixes the image token, the merge size, the interleaved M-RoPE sections, and three
    /// DeepStack layers, so a configuration stating other values is refused.
    static let qwen3VL = NFKMLXFamilyJudge(
        reads: { $0.family == "qwen3-vl" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            let configuration = repository.configuration
            let name = repository.repository
            try NFKMLXRepositoryJudgment.verifyFloatWeights(of: configuration, repository: name)
            guard var text = configuration["text_config"] as? [String: Any] else {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration, "\(name) states no text_config")
            }
            // transformers 4.56 writes the wrapper's name into `text_config.architectures`; the loader drops it.
            text["architectures"] = nil
            let decoder = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXLanguage.configuration(fromJSON: text)
            }
            try NFKMLXRepositoryJudgment.verifyUnreadFields(of: text, repository: name)
            let tower = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXQwen3VLVisionConfiguration.configuration(fromJSON: configuration)
            }
            try NFKMLXRepositoryJudgment.verifyFixedFields(of: configuration, repository: name,
                                                           ["image_token_id": NFKMLXQwen3VL.imageTokenId])
            // transformers 5.x states the sections under `rope_parameters`, earlier releases under `rope_scaling`.
            for key in ["rope_scaling", "rope_parameters"] {
                try NFKMLXRepositoryJudgment.verifyFixedFields(of: text[key] as? [String: Any] ?? [:], repository: name,
                    ["mrope_section": NFKMLXQwen3VL.mropeSection, "mrope_interleaved": true])
            }
            if tower.spatialMergeSize != NFKMLXQwen3VL.spatialMergeSize
                || tower.deepstackLayers.count != NFKMLXQwen3VL.deepstackLayerCount {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                    "\(name) merges \(tower.spatialMergeSize)×\(tower.spatialMergeSize) patches into \(tower.deepstackLayers.count) "
                    + "DeepStack layers, where the integration merges 2×2 into 3")
            }
            try NFKMLXRepositoryJudgment.verifyTokenizer(of: repository)
            return Reading(fusesExperts: decoder.clampedSwiGLU != nil)
        })

    /// TrOCR (`vision-encoder-decoder` with a ViT or DeiT encoder and a `trocr` decoder), read by
    /// ``NFKMLXTrOCRVisionConfiguration`` and ``NFKMLXSeq2SeqConfiguration``, its tokenizer from RoBERTa's `vocab.json` and `merges.txt` or
    /// XLM-R's `sentencepiece.bpe.model`. The model type is the generic vision-encoder-decoder's, so the
    /// judgment requires the two halves the loader builds.
    static let trocr = NFKMLXFamilyJudge(
        reads: { $0.family == "trocr" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            let configuration = repository.configuration
            let name = repository.repository
            try NFKMLXRepositoryJudgment.verifyFloatWeights(of: configuration, repository: name)
            let encoder = configuration["encoder"] as? [String: Any] ?? [:]
            let decoder = configuration["decoder"] as? [String: Any] ?? [:]
            let encoderType = encoder["model_type"] as? String ?? "none"
            let decoderType = decoder["model_type"] as? String ?? "none"
            guard ["vit", "deit"].contains(encoderType), decoderType == "trocr" else {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedArchitecture,
                    "\(name) pairs a \(encoderType) encoder with a \(decoderType) decoder, where the loader builds "
                    + "a ViT or DeiT encoder and a TrOCR decoder")
            }
            try NFKMLXRepositoryJudgment.verifyFixedFields(of: encoder, repository: name, ["hidden_act": "gelu"])
            let byteLevel = repository.file(atPath: "vocab.json") != nil && repository.file(atPath: "merges.txt") != nil
            guard byteLevel || repository.file(atPath: "sentencepiece.bpe.model") != nil else {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                    "\(name) has neither vocab.json and merges.txt nor sentencepiece.bpe.model, the files the loader reads")
            }
            try verifyTokenizerClass(of: repository, among: ["RobertaTokenizer", "RobertaTokenizerFast",
                                                             "XLMRobertaTokenizer", "XLMRobertaTokenizerFast"],
                                     required: false)
            _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                (try NFKMLXTrOCRVisionConfiguration(huggingFaceConfig: configuration),
                 try NFKMLXSeq2SeqConfiguration(huggingFaceConfig: decoder))
            }
            return Reading()
        })

    /// SmolVLM2 (`smolvlm`), its `text_config` read by ``NFKMLXLanguage/configuration(fromJSON:)`` with the
    /// dense decoder's refusals, its SigLIP tower by the tower reader, and its tokenizer by the release
    /// reader (`NFKMLXLanguage.tokenizerScheme(tokenizerJSON:tokenizerConfiguration:)`), which the loader
    /// reads it through. The tower runs the tanh GELU, so a stated tower activation other than it is refused.
    static let smolVLM = NFKMLXFamilyJudge(
        reads: { $0.family == "smolvlm" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            let configuration = repository.configuration
            let name = repository.repository
            try NFKMLXRepositoryJudgment.verifyFloatWeights(of: configuration, repository: name)
            guard let text = configuration["text_config"] as? [String: Any] else {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration, "\(name) states no text_config")
            }
            _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXLanguage.configuration(fromJSON: text)
            }
            try NFKMLXRepositoryJudgment.verifyUnreadFields(of: text, repository: name)
            let tower = configuration["vision_config"] as? [String: Any] ?? [:]
            try NFKMLXRepositoryJudgment.verifyFixedFields(of: tower, repository: name,
                ["model_type": "smolvlm_vision", "hidden_act": "gelu_pytorch_tanh"])
            guard repository.file(atPath: "tokenizer.json") != nil,
                  let tokenizer = try repository.jsonObjectOfFile(atPath: "tokenizer.json") as? [String: Any] else {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                    "\(name) has no readable tokenizer.json, which the loader reads")
            }
            let tokenizerConfiguration = repository.file(atPath: "tokenizer_config.json") == nil ? nil
                : try repository.jsonObjectOfFile(atPath: "tokenizer_config.json") as? [String: Any]
            do {
                _ = try NFKMLXLanguage.tokenizerScheme(tokenizerJSON: tokenizer, tokenizerConfiguration: tokenizerConfiguration)
            } catch {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                    "the tokenizer of \(name) is not one the loader reproduces: " + error.localizedDescription)
            }
            return Reading()
        })
}
