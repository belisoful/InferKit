//
//  NFKMLXRepositoryMatchingDecisionTests.swift
//  InferKitMLXTests
//
//  The typed-decision checkpoints' judgment of another repository, from files placed in a hub cache: a
//  pplx-decider checkpoint read, and a Qwen3.5 text release, a checkpoint without its readout or outside
//  the protocol, a readout of the wrong shape, and the fields and layer kinds the decoder does not build
//  refused. Nothing here reaches MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXRepositoryMatchingDecisionTests: XCTestCase {

    private var fixture: NFKMLXRepositoryFixture!

    override func setUp() {
        super.setUp()
        fixture = NFKMLXRepositoryFixture()
    }

    override func tearDown() {
        fixture.remove()
        super.tearDown()
    }

    private static let template = NFKMLXPplxDeciderVariant.flash.registryName

    private static var configuration: [String: Any] {
        ["model_type": "qwen3_5", "architectures": ["Qwen3_5Model"], "tie_word_embeddings": true,
         "text_config": ["model_type": "qwen3_5_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                         "vocab_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                         "linear_num_key_heads": 2, "linear_key_head_dim": 16, "linear_num_value_heads": 2,
                         "linear_value_head_dim": 16, "hidden_act": "silu", "attention_bias": false,
                         "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                         "rope_parameters": ["rope_theta": 10_000_000, "partial_rotary_factor": 0.25]] as [String: Any]]
    }

    private static var decision: [String: Any] {
        ["attention_mode": "noncausal_full_attention", "pooling": "last", "temperature": 1.5, "codes": ["A", "B", "C", "D"],
         "token_ids": [32, 33, 34, 35], "base_model": "Qwen/Qwen3.5-4B", "revision": "0"]
    }

    private func repository(_ configuration: [String: Any] = configuration, decision: [String: Any] = decision,
                            joint: [String: Any] = NFKMLXPplxDeciderConfiguration.protocolDeclaration,
                            readout: [Int]? = [4, 64]) throws -> NFKModelRepository {
        var files: [String: Data] = [
            "config.json": NFKMLXRepositoryFixture.json(configuration),
            "decision_config.json": NFKMLXRepositoryFixture.json(decision),
            "joint_config.json": NFKMLXRepositoryFixture.json(joint),
            "tokenizer.json": NFKMLXRepositoryFixture.json(NFKMLXRepositoryFixture.byteLevelTokenizer()),
            "tokenizer_config.json": NFKMLXRepositoryFixture.json(["tokenizer_class": "Qwen2Tokenizer"]),
            "model.safetensors": NFKMLXRepositoryFixture.safetensors([("language_model.embed_tokens.weight", [256, 64]),
                                                                      ("language_model.norm.weight", [64])]),
        ]
        if let readout {
            files["readout.safetensors"] = NFKMLXRepositoryFixture.safetensors([("weight", readout)])
        }
        return try fixture.repository("org/decider-tiny", files: files, configuration: configuration, license: nil)
    }

    func testADeciderRepositoryIsRead() throws {
        let descriptor = try NFKMLXModelDescriptors.descriptor(for: try repository(),
                                                               matching: try NFKMLXRepositoryFixture.template(Self.template))
        XCTAssertEqual(descriptor.registryName, Self.template)
    }

    func testATextReleaseOrACheckpointWithoutItsReadoutIsRefused() throws {
        var text = Self.configuration
        text["architectures"] = ["Qwen3_5ForConditionalGeneration"]
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedArchitecture, try repository(text), template: Self.template)
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedArchitecture, try repository(readout: nil), template: Self.template)
        var kinds = Self.configuration
        var decoder = kinds["text_config"] as! [String: Any]
        decoder["layer_types"] = ["linear_attention", "linear_attention", "sliding_attention", "full_attention"]
        kinds["text_config"] = decoder
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedArchitecture, try repository(kinds), template: Self.template)
    }

    func testAConfigurationTheLoaderReadsWronglyIsRefused() throws {
        var joint = NFKMLXPplxDeciderConfiguration.protocolDeclaration
        joint["version"] = 2
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(joint: joint), template: Self.template)
        var causal = Self.decision
        causal["attention_mode"] = "causal"
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(decision: causal),
                                                  template: Self.template)
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(readout: [4, 32]),
                                                  template: Self.template)
        var activation = Self.configuration
        var decoder = activation["text_config"] as! [String: Any]
        decoder["hidden_act"] = "gelu"
        activation["text_config"] = decoder
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(activation), template: Self.template)
    }
}
