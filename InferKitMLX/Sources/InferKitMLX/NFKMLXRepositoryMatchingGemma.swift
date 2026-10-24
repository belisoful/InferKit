//
//  NFKMLXRepositoryMatchingGemma.swift
//  InferKitMLX
//
//  The Gemma decoders' judgments of another repository: Gemma 3, Gemma 3n, and Gemma 4, each read by its
//  generation's configuration reader, and the tokenizer every generation ships, held to the form
//  `NFKMLXGemmaTokenizer` reproduces.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXFamilyJudge {

    /// Gemma 3 (`gemma3_text`, and `gemma3` with a SigLIP vision tower), read by
    /// `NFKMLXGemma3Language.configuration(fromJSON:)`.
    static let gemma3 = NFKMLXFamilyJudge(
        reads: { $0.family == "gemma3" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            let configuration = repository.configuration
            try verifyGemmaRepository(repository)
            _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXGemma3Language.configuration(fromJSON: configuration)
            }
            try verifyTower(configuration["vision_config"], of: repository, modelType: "siglip_vision_model")
            return Reading()
        })

    /// Gemma 3n (`gemma3n`), read by `NFKMLXGemma3nLanguage.configuration(fromJSON:)` and its audio
    /// encoder's reader. The vision tower is built at the released MobileNet-V5 geometry, so another
    /// tower is refused.
    static let gemma3n = NFKMLXFamilyJudge(
        reads: { $0.family == "gemma3n" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            let configuration = repository.configuration
            try verifyGemmaRepository(repository)
            _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXGemma3nLanguage.configuration(fromJSON: configuration)
            }
            if configuration["audio_config"] != nil {
                _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                    try NFKMLXGemma3nAudio.configuration(fromJSON: configuration)
                }
            }
            try verifyTower(configuration["vision_config"], of: repository, modelType: "gemma3n_vision")
            if let vision = configuration["vision_config"] as? [String: Any],
               let architecture = vision["architecture"] as? String, architecture != "mobilenetv5_300m_enc" {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                    "\(repository.repository) carries a \(architecture) vision tower, where the loader builds MobileNet-V5")
            }
            return Reading()
        })

    /// Gemma 4 (`gemma4`), read by `NFKMLXGemmaLanguage.configuration(fromHuggingFace:)`. The reader takes
    /// each attention kind's rotary base and the full attention's partial factor and builds the released
    /// kinds, so another kind is refused. The loader reads the decoder under `model.language_model.`.
    static let gemma4 = NFKMLXFamilyJudge(
        reads: { $0.family == "gemma4" && !$0.modelTypes.isEmpty },
        template: { repository, template in
            let text = (repository.configuration["text_config"] as? [String: Any]) ?? repository.configuration
            let mixture = (text["enable_moe_block"] as? NSNumber)?.boolValue ?? false
            return NFKMLXModelDescriptors.entries.first {
                $0.family == template.family && !$0.modelTypes.isEmpty
                    && $0.residencies.contains(NFKModelResidencyPaged) == mixture
            } ?? template
        },
        verify: { repository, template in
            try verifyGemmaRepository(repository)
            let text = (repository.configuration["text_config"] as? [String: Any]) ?? repository.configuration
            let name = repository.repository
            if let scaling = text["rope_scaling"], !(scaling is NSNull) {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                    "\(name) scales its rotary (rope_scaling), which the loader does not apply")
            }
            let rope = text["rope_parameters"] as? [String: Any] ?? [:]
            for (kind, built) in [("full_attention", "proportional"), ("sliding_attention", "default")] {
                if let stated = (rope[kind] as? [String: Any])?["rope_type"] as? String, stated != built {
                    throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                        "\(name) runs its \(kind) rotary as \(stated), where the loader builds \(built)")
                }
            }
            let directory = try NFKMLXRepositoryJudgment.snapshot(of: repository, holding: ["config.json"])
            _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXGemmaLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
            }
            return Reading()
        },
        verifyTensors: { tensors, repository in
            guard tensors.contains(where: { $0.name.hasPrefix("model.language_model.") }) else {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedWeightFormat,
                    "\(repository.repository) stores its decoder outside model.language_model., where the loader reads it")
            }
        })

    /// The checks every Gemma generation shares: float weights, the released activation without attention
    /// biases, a head tied to the embedding, and the tokenizer.
    private static func verifyGemmaRepository(_ repository: NFKModelRepository) throws {
        let configuration = repository.configuration
        let name = repository.repository
        try NFKMLXRepositoryJudgment.verifyFloatWeights(of: configuration, repository: name)
        let text = (configuration["text_config"] as? [String: Any]) ?? configuration
        try NFKMLXRepositoryJudgment.verifyFixedFields(of: text, repository: name,
            ["hidden_activation": "gelu_pytorch_tanh", "hidden_act": "gelu_pytorch_tanh", "attention_bias": false])
        // The loaders drop `lm_head.weight` and run the embedding as the head, as every Gemma release ties them.
        for scope in [configuration, text] where (scope["tie_word_embeddings"] as? NSNumber)?.boolValue == false {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                "\(name) unties its output head from the embedding, which the loader runs as the head")
        }
        try verifyGemmaTokenizer(of: repository)
    }

    /// Refuses a tower whose `model_type` is not the one the loader builds.
    private static func verifyTower(_ tower: Any?, of repository: NFKModelRepository, modelType: String) throws {
        guard let tower = tower as? [String: Any], let stated = tower["model_type"] as? String, stated != modelType else {
            return
        }
        throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
            "\(repository.repository) carries a \(stated) tower, where the loader builds \(modelType)")
    }

    /// The tokenizer form `NFKMLXGemmaTokenizer` reproduces, as every Gemma generation's `tokenizer.json`
    /// writes it: a space normalized to `▁`, the whole text one pre-token, and byte-fallback BPE decoded
    /// back through the same steps.
    static let gemmaTokenizerForm: [String: Any] = [
        "normalizer": ["type": "Replace", "pattern": ["String": " "], "content": "\u{2581}"],
        "pre_tokenizer": ["type": "Split", "pattern": ["String": " "], "behavior": "MergedWithPrevious", "invert": false],
        "decoder": ["type": "Sequence", "decoders": [
            ["type": "Replace", "pattern": ["String": "\u{2581}"], "content": " "], ["type": "ByteFallback"], ["type": "Fuse"]]],
    ]

    /// Refuses a tokenizer outside the form `NFKMLXGemmaTokenizer` reproduces. The reader takes any BPE
    /// vocabulary and merges, so a Gemma 2 or a byte-level tokenizer would encode to valid ids of the
    /// wrong pieces.
    static func verifyGemmaTokenizer(of repository: NFKModelRepository) throws {
        let name = repository.repository
        guard repository.file(atPath: "tokenizer.json") != nil else {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer, "\(name) has no tokenizer.json, which the loader reads")
        }
        guard let tokenizer = try repository.jsonObjectOfFile(atPath: "tokenizer.json") as? [String: Any],
              let model = tokenizer["model"] as? [String: Any] else {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer, "the tokenizer.json of \(name) is not a tokenizer definition")
        }
        let present = { (value: Any?) in value != nil && !(value is NSNull) }
        let affixes = [model["continuing_subword_prefix"], model["end_of_word_suffix"]].contains { ($0 as? String).map { !$0.isEmpty } ?? false }
        guard model["type"] as? String == "BPE", (model["byte_fallback"] as? NSNumber)?.boolValue == true,
              (model["ignore_merges"] as? NSNumber)?.boolValue != true, !present(model["dropout"]), !affixes,
              model["merges"] is [Any], model["vocab"] is [String: Any] else {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                "\(name) tokenizes with a model other than the byte-fallback BPE Gemma's tokenizer reproduces")
        }
        for (key, form) in gemmaTokenizerForm.sorted(by: { $0.key < $1.key }) {
            guard let stated = tokenizer[key], (stated as AnyObject).isEqual(form) else {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedTokenizer,
                    "the \(key) of \(name)'s tokenizer is not the form Gemma's tokenizer reproduces")
            }
        }
    }
}
