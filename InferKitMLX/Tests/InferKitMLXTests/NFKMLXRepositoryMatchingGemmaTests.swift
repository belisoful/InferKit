//
//  NFKMLXRepositoryMatchingGemmaTests.swift
//  InferKitMLXTests
//
//  The Gemma decoders' judgments of another repository, from files placed in a hub cache: each
//  generation read on its template, a Gemma 4 mixture on the mixture release, and the tokenizer, the
//  towers, the rotary kinds, and the weight layout the loaders do not read refused. Nothing here reaches MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXRepositoryMatchingGemmaTests: XCTestCase {

    private var fixture: NFKMLXRepositoryFixture!

    override func setUp() {
        super.setUp()
        fixture = NFKMLXRepositoryFixture()
    }

    override func tearDown() {
        fixture.remove()
        super.tearDown()
    }

    // MARK: Fixtures

    /// A two-layer Gemma text decoder's fields, shared by the generations.
    private static var text: [String: Any] {
        ["hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
         "intermediate_size": 128, "vocab_size": 256, "rms_norm_eps": 1e-6, "hidden_activation": "gelu_pytorch_tanh",
         "attention_bias": false, "sliding_window": 512, "max_position_embeddings": 32768]
    }

    private static var gemma3Text: [String: Any] {
        var configuration = text
        configuration["model_type"] = "gemma3_text"
        configuration["architectures"] = ["Gemma3ForCausalLM"]
        configuration["layer_types"] = ["sliding_attention", "full_attention"]
        return configuration
    }

    private static var gemma4: [String: Any] {
        var decoder = text
        decoder["model_type"] = "gemma4_text"
        decoder["layer_types"] = ["sliding_attention", "full_attention"]
        decoder["rope_parameters"] = ["full_attention": ["partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0,
                                                         "rope_type": "proportional"],
                                      "sliding_attention": ["rope_theta": 10000.0, "rope_type": "default"]]
        return ["model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"], "text_config": decoder]
    }

    /// Gemma's tokenizer form with a two-piece vocabulary.
    private static var tokenizer: [String: Any] {
        var tokenizer = NFKMLXFamilyJudge.gemmaTokenizerForm
        tokenizer["model"] = ["type": "BPE", "byte_fallback": true, "ignore_merges": false, "unk_token": "<unk>",
                              "vocab": ["<unk>": 0, "\u{2581}a": 1], "merges": [String]()]
        return tokenizer
    }

    private func repository(_ configuration: [String: Any], tokenizer: [String: Any] = tokenizer,
                            prefix: String = "model.") throws -> NFKModelRepository {
        let weights = NFKMLXRepositoryFixture.safetensors([(prefix + "layers.0.mlp.gate_proj.weight", [128, 64]),
                                                           (prefix + "layers.1.mlp.gate_proj.weight", [128, 64])])
        return try fixture.repository("org/gemma-tiny",
                                      files: ["config.json": NFKMLXRepositoryFixture.json(configuration),
                                              "tokenizer.json": NFKMLXRepositoryFixture.json(tokenizer),
                                              "tokenizer_config.json": NFKMLXRepositoryFixture.json([:] as [String: Any]),
                                              "model.safetensors": weights],
                                      configuration: configuration, license: nil)
    }

    private func descriptor(_ repository: NFKModelRepository, template identifier: String) throws -> NFKModelDescriptor {
        try NFKMLXModelDescriptors.descriptor(for: repository, matching: try NFKMLXRepositoryFixture.template(identifier))
    }

    // MARK: Read

    func testAGemma3TextRepositoryIsRead() throws {
        let descriptor = try descriptor(try repository(Self.gemma3Text), template: "gemma-3-270m-it")
        XCTAssertEqual(descriptor.family, "gemma3")
        XCTAssertEqual(descriptor.registryName, "gemma-3-270m-it")
    }

    func testAGemma4RepositoryIsReadAndAMixtureTakesTheMixtureRelease() throws {
        let dense = try descriptor(try repository(Self.gemma4, prefix: "model.language_model."), template: "gemma-4-e2b-it")
        XCTAssertEqual(dense.registryName, "gemma-4-e2b-it")
        var mixture = Self.gemma4
        var decoder = try XCTUnwrap(mixture["text_config"] as? [String: Any])
        decoder["enable_moe_block"] = true
        decoder["num_experts"] = 4
        decoder["top_k_experts"] = 2
        decoder["moe_intermediate_size"] = 32
        mixture["text_config"] = decoder
        let routed = try descriptor(try repository(mixture, prefix: "model.language_model."), template: "gemma-4-e2b-it")
        XCTAssertEqual(routed.registryName, "gemma-4-26b-a4b-it")
    }

    func testAMultimodalGemma3AndAGemma3nRepositoryAreRead() throws {
        let multimodal: [String: Any] = ["model_type": "gemma3", "architectures": ["Gemma3ForConditionalGeneration"],
                                         "text_config": Self.gemma3Text.merging(["architectures": NSNull()]) { $1 },
                                         "vision_config": ["model_type": "siglip_vision_model", "hidden_size": 64]]
        XCTAssertEqual(try descriptor(try repository(multimodal, prefix: "model.language_model."),
                                      template: "gemma-3-4b-it").registryName, "gemma-3-4b-it")
        XCTAssertEqual(try descriptor(try repository(Self.gemma3n, prefix: "model.language_model."),
                                      template: "gemma-3n-e2b-it").family, "gemma3n")
    }

    /// A Gemma 3n release: its text decoder and the released MobileNet-V5 tower.
    private static var gemma3n: [String: Any] {
        var configuration = text
        configuration["model_type"] = "gemma3n_text"
        configuration["intermediate_size"] = [128, 128]
        configuration["layer_types"] = ["sliding_attention", "full_attention"]
        return ["model_type": "gemma3n", "architectures": ["Gemma3nForConditionalGeneration"], "text_config": configuration,
                "vision_config": ["model_type": "gemma3n_vision", "architecture": "mobilenetv5_300m_enc", "hidden_size": 2048]]
    }

    // MARK: Refusals

    func testAnUntiedHeadIsRefused() throws {
        var top = Self.gemma3Text
        top["tie_word_embeddings"] = false
        var nested = Self.gemma4
        var decoder = try XCTUnwrap(nested["text_config"] as? [String: Any])
        decoder["tie_word_embeddings"] = false
        nested["text_config"] = decoder
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(top), template: "gemma-3-270m-it")
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration,
                                                  try repository(nested, prefix: "model.language_model."),
                                                  template: "gemma-4-e2b-it")
    }

    func testAModelTypeTheAskedEntryDoesNotListIsRefused() throws {
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedArchitecture, try repository(Self.gemma3Text),
                                                  template: "gemma-3-4b-it")
    }

    func testATokenizerOutsideGemmasFormIsRefused() throws {
        var prepended = Self.tokenizer
        prepended["normalizer"] = ["type": "Sequence", "normalizers": [
            ["type": "Prepend", "prepend": "\u{2581}"], ["type": "Replace", "pattern": ["String": " "], "content": "\u{2581}"]]]
        var byteLevel = Self.tokenizer
        byteLevel["model"] = ["type": "BPE", "vocab": ["a": 0], "merges": [String]()]
        for tokenizer in [prepended, byteLevel] {
            try NFKMLXRepositoryFixture.assertRefusal(.unsupportedTokenizer, try repository(Self.gemma3Text, tokenizer: tokenizer),
                                                      template: "gemma-3-270m-it")
        }
    }

    func testAnActivationOrBiasTheDecoderDoesNotBuildIsRefused() throws {
        for (key, value) in [("hidden_activation", "silu"), ("attention_bias", true)] as [(String, Any)] {
            var configuration = Self.gemma3Text
            configuration[key] = value
            try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(configuration),
                                                      template: "gemma-3-270m-it")
        }
    }

    func testARotaryScalingTheGemma3ReaderRefusesIsRefused() throws {
        var configuration = Self.gemma3Text
        configuration["rope_scaling"] = ["rope_type": "yarn", "factor": 4.0]
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(configuration),
                                                  template: "gemma-3-270m-it")
    }

    func testAGemma4RotaryKindTheLoaderDoesNotBuildIsRefused() throws {
        var configuration = Self.gemma4
        var decoder = try XCTUnwrap(configuration["text_config"] as? [String: Any])
        decoder["rope_parameters"] = ["full_attention": ["rope_theta": 1_000_000.0, "rope_type": "yarn", "factor": 4.0],
                                      "sliding_attention": ["rope_theta": 10000.0, "rope_type": "default"]]
        configuration["text_config"] = decoder
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration,
                                                  try repository(configuration, prefix: "model.language_model."),
                                                  template: "gemma-4-e2b-it")
    }

    func testAGemma4DecoderOutsideTheLanguageModelPrefixIsRefused() throws {
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedWeightFormat, try repository(Self.gemma4),
                                                  template: "gemma-4-e2b-it")
    }

    func testAGemma3nVisionTowerOtherThanMobileNetIsRefused() throws {
        var release = Self.gemma3n
        release["vision_config"] = ["model_type": "gemma3n_vision", "architecture": "resnet50", "hidden_size": 2048]
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration,
                                                  try repository(release, prefix: "model.language_model."),
                                                  template: "gemma-3n-e2b-it")
    }
}
