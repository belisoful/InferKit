//
//  NFKMLXRepositoryMatchingDecision.swift
//  InferKitMLX
//
//  The typed-decision checkpoints' judgment of another repository: a pplx-decider checkpoint, a
//  `Qwen3_5Model` backbone with a readout under the joint inline-readout protocol, read by its own
//  configuration readers and the release tokenizer reader.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXFamilyJudge {

    /// pplx-decider (`qwen3_5`, architecture `Qwen3_5Model`), read by ``NFKMLXPplxDeciderConfiguration``.
    /// A Qwen3.5 text release shares the model type and carries an output projection instead of a
    /// readout, so a repository whose architecture is not the bare `Qwen3_5Model`, or that lacks the
    /// protocol files and the readout, is refused. The reader drops a layer kind it does not know, so one
    /// is refused, and the readout must hold one row per answer code at the decoder's width.
    static let pplxDecider = NFKMLXFamilyJudge(
        reads: { $0.family == "pplx-decider" && !$0.modelTypes.isEmpty },
        verify: { repository, template in
            let configuration = repository.configuration
            let name = repository.repository
            try NFKMLXRepositoryJudgment.verifyFloatWeights(of: configuration, repository: name)
            try NFKMLXRepositoryJudgment.verifyArchitecture(of: configuration, repository: name, suffixes: ["Qwen3_5Model"])
            let protocolFiles = ["decision_config.json", "joint_config.json", "readout.safetensors"]
            if let absent = protocolFiles.first(where: { repository.file(atPath: $0) == nil }) {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedArchitecture,
                    "\(name) has no \(absent), so it is not a decision checkpoint the readout reads")
            }
            let text = configuration["text_config"] as? [String: Any] ?? configuration
            try NFKMLXRepositoryJudgment.verifyFixedFields(of: text, repository: name,
                                                           ["hidden_act": "silu", "attention_bias": false])
            let kinds = text["layer_types"] as? [String] ?? []
            if let unbuilt = Set(kinds).subtracting(["linear_attention", "full_attention"]).sorted().first {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedArchitecture,
                    "\(name) has \(unbuilt) layers, which the Qwen3.5 decoder does not build")
            }
            let directory = try NFKMLXRepositoryJudgment.snapshot(
                of: repository, holding: ["config.json", "decision_config.json", "joint_config.json"])
            let read = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXPplxDeciderConfiguration(directoryURL: directory)
            }
            if !kinds.isEmpty && kinds.count != read.decoder.layerCount {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                    "\(name) lists \(kinds.count) layer kinds for \(read.decoder.layerCount) layers")
            }
            let readout = try repository.safetensorsHeaderOfFile(atPath: "readout.safetensors")
            let shape = (readout["weight"] as? [String: Any])?["shape"] as? [Int]
            let expected = [read.codes.count, read.decoder.hiddenSize]
            if shape != expected || readout.keys.filter({ $0 != "__metadata__" }).count != 1 {
                throw NFKMLXRepositoryJudgment.refusal(.unsupportedConfiguration,
                    "the readout of \(name) holds \(readout.keys.sorted()), where the loader reads weight \(expected)")
            }
            try NFKMLXRepositoryJudgment.verifyTokenizer(of: repository)
            return Reading()
        })
}
