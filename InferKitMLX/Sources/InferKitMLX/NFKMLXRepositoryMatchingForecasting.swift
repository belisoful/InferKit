//
//  NFKMLXRepositoryMatchingForecasting.swift
//  InferKitMLX
//
//  The forecasters' judgments of another repository: TimesFM 2.5, in its official and transformers
//  layouts, read by its own configuration reader.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXFamilyJudge {

    /// TimesFM 2.5 (`timesfm` in the official layout, `timesfm2_5` in transformers'), read by
    /// ``NFKMLXTimesFMConfiguration``. TimesFM 1.0 and 2.0 share the `timesfm` model type and are another
    /// network, so the judgment requires the 2.5 quantile head and refuses their frequency embedding. The
    /// network fixes its activation, biases, key-value heads, and rotary base, so a configuration stating
    /// another value for one is refused.
    static let timesFM = NFKMLXFamilyJudge(
        reads: { $0.family == "timesfm" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            let configuration = repository.configuration
            let name = repository.repository
            try NFKMLXRepositoryJudgment.verifyFloatWeights(of: configuration, repository: name)
            if configuration["freq_size"] != nil || configuration["use_positional_embedding"] != nil {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedArchitecture,
                    "\(name) is TimesFM 1.0 or 2.0 (a frequency embedding), a network the TimesFM 2.5 loader does not build")
            }
            guard configuration["quantile_horizon_length"] != nil || configuration["output_quantile_len"] != nil else {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedArchitecture,
                    "\(name) states no continuous quantile head, which TimesFM 2.5 has and the loader builds")
            }
            let rotary = (configuration["rope_parameters"] as? [String: Any])?["rope_theta"] ?? configuration["rope_theta"]
            var fixed: [String: Any] = ["activation": "swish", "attention_bias": false, "use_bias": false]
            if let heads = configuration["num_attention_heads"] {
                fixed["num_key_value_heads"] = heads
            }
            try NFKMLXRepositoryJudgment.verifyFixedFields(of: configuration, repository: name, fixed)
            if let rotary, !((rotary as AnyObject).isEqual(10_000)) {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                    "\(name) states a rotary base of \(rotary), where the loader builds 10000")
            }
            let directory = try NFKMLXRepositoryJudgment.snapshot(of: repository, holding: ["config.json"])
            _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXTimesFMConfiguration(configurationURL: directory.appendingPathComponent("config.json"))
            }
            return Reading()
        },
        weightFiles: NFKMLXRepositoryJudgment.singleWeightFile("model.safetensors"))
}
