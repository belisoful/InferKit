//
//  NFKMLXRepositoryMatchingHybrid.swift
//  InferKitMLX
//
//  The hybrid Mamba-2 decoders' judgments of another repository: Granite 4.0-H and Nemotron-H, each read
//  by its own configuration reader and the release tokenizer reader, with the layer kinds and activations
//  the networks fix held to the released values.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXFamilyJudge {

    /// Granite 4.0-H (`granitemoehybrid` without positional embeddings), read by
    /// ``NFKMLXGraniteHybridConfiguration``. Granite 4.0's attention-only releases share the model type
    /// and rotate their positions, which the network does not, so a stated position embedding other than
    /// `nope` is refused. The reader reads an unknown layer kind as Mamba, so one is refused. A repository
    /// with routed experts takes the mixture release.
    static let graniteH = NFKMLXFamilyJudge(
        reads: { $0.family == "granite-4.0-h" && !$0.modelTypes.isEmpty },
        template: { repository, template in
            let mixture = ((repository.configuration["num_local_experts"] as? NSNumber)?.intValue ?? 0) > 0
            return NFKMLXModelDescriptors.entries.first {
                $0.family == template.family && !$0.modelTypes.isEmpty && ($0.residencies.count > 1) == mixture
            } ?? template
        },
        verify: { repository, template in
            let configuration = repository.configuration
            let name = repository.repository
            try NFKMLXRepositoryJudgment.verifyFloatWeights(of: configuration, repository: name)
            if (configuration["position_embedding_type"] as? String ?? "nope") != "nope" {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                    "\(name) rotates its attention positions, where the Granite 4.0-H network attends without them")
            }
            try NFKMLXRepositoryJudgment.verifyFixedFields(of: configuration, repository: name,
                                                           ["hidden_act": "silu", "attention_bias": false])
            let kinds = Set(configuration["layer_types"] as? [String] ?? [])
            if let unbuilt = kinds.subtracting(["mamba", "attention"]).sorted().first {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedArchitecture,
                    "\(name) has \(unbuilt) layers, which the Granite 4.0-H network does not build")
            }
            _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXGraniteHybridConfiguration.configuration(fromHuggingFace: configuration)
            }
            try NFKMLXRepositoryJudgment.verifyTokenizer(of: repository)
            return Reading()
        })

    /// Nemotron-H (`nemotron_h`), read by ``NFKMLXNemotronHConfiguration``. The network builds Mamba-2,
    /// attention, and squared-ReLU MLP layers, and the reader reads an unknown kind as an MLP, so a layer of
    /// another kind (the routed experts of a mixture release) is refused.
    static let nemotronH = NFKMLXFamilyJudge(
        reads: { $0.family == "nemotron-h" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            let configuration = repository.configuration
            let name = repository.repository
            try NFKMLXRepositoryJudgment.verifyFloatWeights(of: configuration, repository: name)
            let pattern = configuration["hybrid_override_pattern"] as? String ?? ""
            let kinds = configuration["layers_block_type"] as? [String] ?? []
            let unbuiltLetter = pattern.contains { !"M*-".contains($0) }
            let unbuiltKind = !Set(kinds).isSubset(of: ["linear_attention", "full_attention", "mlp"])
            if unbuiltLetter || unbuiltKind {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedArchitecture,
                    "\(name) has layers other than Mamba-2, attention, and MLP (routed experts), which the Nemotron-H "
                    + "network does not build")
            }
            try NFKMLXRepositoryJudgment.verifyFixedFields(of: configuration, repository: name,
                                                           ["mlp_hidden_act": "relu2", "attention_bias": false])
            _ = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXNemotronHConfiguration.configuration(fromHuggingFace: configuration)
            }
            try NFKMLXRepositoryJudgment.verifyTokenizer(of: repository)
            return Reading()
        })
}
