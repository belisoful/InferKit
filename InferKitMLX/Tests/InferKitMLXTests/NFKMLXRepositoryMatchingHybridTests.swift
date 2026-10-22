//
//  NFKMLXRepositoryMatchingHybridTests.swift
//  InferKitMLXTests
//
//  The hybrid Mamba-2 decoders' judgments of another repository, from files placed in a hub cache:
//  Granite 4.0-H dense and with routed experts, Nemotron-H, and the layer kinds, positions, and activations
//  their networks do not build refused. Nothing here reaches MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXRepositoryMatchingHybridTests: XCTestCase {

    private var fixture: NFKMLXRepositoryFixture!

    override func setUp() {
        super.setUp()
        fixture = NFKMLXRepositoryFixture()
    }

    override func tearDown() {
        fixture.remove()
        super.tearDown()
    }

    private static var granite: [String: Any] {
        ["model_type": "granitemoehybrid", "architectures": ["GraniteMoeHybridForCausalLM"], "hidden_size": 64,
         "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 256,
         "layer_types": ["mamba", "attention"], "position_embedding_type": "nope", "hidden_act": "silu",
         "attention_bias": false, "num_local_experts": 0, "shared_intermediate_size": 128, "rope_theta": 10000]
    }

    private static var nemotron: [String: Any] {
        ["model_type": "nemotron_h", "architectures": ["NemotronHForCausalLM"], "hidden_size": 64,
         "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 256, "hybrid_override_pattern": "M*-",
         "mlp_hidden_act": "relu2", "attention_bias": false, "mlp_bias": false, "intermediate_size": 128]
    }

    private func repository(_ configuration: [String: Any]) throws -> NFKModelRepository {
        let weights = NFKMLXRepositoryFixture.safetensors([("model.layers.0.mixer.in_proj.weight", [128, 64]),
                                                           ("model.layers.1.mixer.in_proj.weight", [128, 64])])
        return try fixture.repository("org/hybrid-tiny", files: [
            "config.json": NFKMLXRepositoryFixture.json(configuration),
            "tokenizer.json": NFKMLXRepositoryFixture.json(NFKMLXRepositoryFixture.byteLevelTokenizer()),
            "tokenizer_config.json": NFKMLXRepositoryFixture.json(["tokenizer_class": "GPT2Tokenizer"]),
            "model.safetensors": weights], configuration: configuration, license: nil)
    }

    private func descriptor(_ configuration: [String: Any], template identifier: String) throws -> NFKModelDescriptor {
        try NFKMLXModelDescriptors.descriptor(for: try repository(configuration),
                                              matching: try NFKMLXRepositoryFixture.template(identifier))
    }

    func testAGraniteRepositoryIsReadAndAMixtureTakesTheMixtureRelease() throws {
        XCTAssertEqual(try descriptor(Self.granite, template: NFKMLXGraniteHybrid.graniteModelName).registryName,
                       NFKMLXGraniteHybrid.graniteModelName)
        var mixture = Self.granite
        mixture["num_local_experts"] = 4
        mixture["num_experts_per_tok"] = 2
        mixture["intermediate_size"] = 32
        XCTAssertEqual(try descriptor(mixture, template: NFKMLXGraniteHybrid.graniteModelName).registryName,
                       NFKMLXGraniteHybrid.graniteTinyModelName)
    }

    func testGraniteFieldsTheNetworkDoesNotBuildAreRefused() throws {
        var rotary = Self.granite
        rotary["position_embedding_type"] = "rope"
        var activation = Self.granite
        activation["hidden_act"] = "gelu"
        for configuration in [rotary, activation] {
            try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(configuration),
                                                      template: NFKMLXGraniteHybrid.graniteModelName)
        }
        var kinds = Self.granite
        kinds["layer_types"] = ["mamba", "sliding_attention"]
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedArchitecture, try repository(kinds),
                                                  template: NFKMLXGraniteHybrid.graniteModelName)
    }

    func testANemotronRepositoryIsRead() throws {
        XCTAssertEqual(try descriptor(Self.nemotron, template: NFKMLXNemotronH.nemotronModelName).family, "nemotron-h")
    }

    func testNemotronLayersAndActivationsTheNetworkDoesNotBuildAreRefused() throws {
        var experts = Self.nemotron
        experts["hybrid_override_pattern"] = "M*E"
        var named = Self.nemotron
        named["hybrid_override_pattern"] = nil
        named["layers_block_type"] = ["mamba", "attention", "mlp"]
        for configuration in [experts, named] {
            try NFKMLXRepositoryFixture.assertRefusal(.unsupportedArchitecture, try repository(configuration),
                                                      template: NFKMLXNemotronH.nemotronModelName)
        }
        var activation = Self.nemotron
        activation["mlp_hidden_act"] = "silu"
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(activation),
                                                  template: NFKMLXNemotronH.nemotronModelName)
    }
}
