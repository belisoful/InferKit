//
//  NFKMLXRepositoryMatchingVisionTests.swift
//  InferKitMLXTests
//
//  The vision-language and text-recognition judgments of another repository, from files placed in a hub
//  cache: Qwen3-VL with the values its integration fixes, SmolVLM2 with its tower and tokenizer, and TrOCR
//  among the vision-encoder-decoder repositories that share its model type. Nothing here reaches MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXRepositoryMatchingVisionTests: XCTestCase {

    private var fixture: NFKMLXRepositoryFixture!

    override func setUp() {
        super.setUp()
        fixture = NFKMLXRepositoryFixture()
    }

    override func tearDown() {
        fixture.remove()
        super.tearDown()
    }

    // MARK: Qwen3-VL

    /// A two-layer Qwen3-VL: its decoder, its tower, and the token ids the release states.
    private static var qwen3VL: [String: Any] {
        let text: [String: Any] = [
            "model_type": "qwen3_vl_text", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 32, "intermediate_size": 128, "vocab_size": 256, "rope_theta": 5_000_000,
            "rms_norm_eps": 1e-6, "hidden_act": "silu", "tie_word_embeddings": true, "max_position_embeddings": 262_144,
            "rope_scaling": ["mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"]]
        let vision: [String: Any] = [
            "model_type": "qwen3_vl", "hidden_size": 64, "depth": 2, "num_heads": 2, "intermediate_size": 128,
            "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 64,
            "num_position_embeddings": 2304, "deepstack_visual_indexes": [0, 1, 1]]
        return ["model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"],
                "image_token_id": 151_655, "text_config": text, "vision_config": vision]
    }

    private func qwen3VLRepository(_ configuration: [String: Any] = qwen3VL) throws -> NFKModelRepository {
        let weights = NFKMLXRepositoryFixture.safetensors([("model.language_model.layers.0.mlp.gate_proj.weight", [128, 64]),
                                                           ("model.language_model.layers.1.mlp.gate_proj.weight", [128, 64])])
        return try fixture.repository("org/qwen3-vl-tiny", files: [
            "config.json": NFKMLXRepositoryFixture.json(configuration),
            "tokenizer.json": NFKMLXRepositoryFixture.json(NFKMLXRepositoryFixture.byteLevelTokenizer()),
            "tokenizer_config.json": NFKMLXRepositoryFixture.json(["tokenizer_class": "Qwen2Tokenizer"]),
            "model.safetensors": weights], configuration: configuration, license: nil)
    }

    func testAQwen3VLRepositoryIsRead() throws {
        let descriptor = try NFKMLXModelDescriptors.descriptor(for: try qwen3VLRepository(),
                                                               matching: try NFKMLXRepositoryFixture.template("qwen3-vl-2b"))
        XCTAssertEqual(descriptor.family, "qwen3-vl")
        XCTAssertEqual(descriptor.registryName, "qwen3-vl-2b")
    }

    func testValuesTheIntegrationFixesAreRefusedAtOtherValues() throws {
        var token = Self.qwen3VL
        token["image_token_id"] = 151_000
        var sections = Self.qwen3VL
        var text = try XCTUnwrap(sections["text_config"] as? [String: Any])
        text["rope_scaling"] = ["mrope_interleaved": true, "mrope_section": [16, 24, 24], "rope_type": "default"]
        sections["text_config"] = text
        var merge = Self.qwen3VL
        var vision = try XCTUnwrap(merge["vision_config"] as? [String: Any])
        vision["spatial_merge_size"] = 3
        merge["vision_config"] = vision
        var activation = Self.qwen3VL
        text = try XCTUnwrap(activation["text_config"] as? [String: Any])
        text["hidden_act"] = "gelu"
        activation["text_config"] = text
        var parameters = Self.qwen3VL
        text = try XCTUnwrap(parameters["text_config"] as? [String: Any])
        text["rope_scaling"] = nil
        text["rope_parameters"] = ["mrope_interleaved": true, "mrope_section": [16, 24, 24], "rope_type": "default",
                                   "rope_theta": 5_000_000]
        parameters["text_config"] = text
        for configuration in [token, sections, merge, activation, parameters] {
            try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try qwen3VLRepository(configuration),
                                                      template: "qwen3-vl-2b")
        }
    }

    func testADenseQwen3VLOnTheMixtureEntryIsRefused() throws {
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedArchitecture, try qwen3VLRepository(),
                                                  template: "qwen3-vl-30b-a3b")
    }

    // MARK: SmolVLM2

    /// A two-layer SmolVLM2: a Llama decoder and a SigLIP tower, with SmolLM's digit-splitting tokenizer.
    private static var smolVLM: [String: Any] {
        let text: [String: Any] = [
            "model_type": "llama", "architectures": ["VLlama3ForCausalLM"], "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 128, "vocab_size": 256,
            "rope_theta": 100_000, "rms_norm_eps": 1e-5, "max_position_embeddings": 8192]
        let vision: [String: Any] = ["model_type": "smolvlm_vision", "hidden_size": 64, "num_hidden_layers": 2,
                                     "num_attention_heads": 2, "intermediate_size": 128, "image_size": 512, "patch_size": 16]
        return ["model_type": "smolvlm", "architectures": ["SmolVLMForConditionalGeneration"], "image_token_id": 49_190,
                "scale_factor": 4, "text_config": text, "vision_config": vision]
    }

    /// SmolLM's tokenizer form: individual digits, then a GPT-2 byte-level split.
    private static var smolLMTokenizer: [String: Any] {
        ["model": ["type": "BPE", "vocab": [String: Int](), "merges": [String]()],
         "pre_tokenizer": ["type": "Sequence", "pretokenizers": [
            ["type": "Digits", "individual_digits": true],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true]]],
         "decoder": ["type": "ByteLevel"]]
    }

    private func smolVLMRepository(_ configuration: [String: Any] = smolVLM,
                                   tokenizer: [String: Any] = smolLMTokenizer) throws -> NFKModelRepository {
        let weights = NFKMLXRepositoryFixture.safetensors([("model.text_model.layers.0.mlp.gate_proj.weight", [128, 64]),
                                                           ("model.text_model.layers.1.mlp.gate_proj.weight", [128, 64])])
        return try fixture.repository("org/smolvlm-tiny", files: [
            "config.json": NFKMLXRepositoryFixture.json(configuration),
            "tokenizer.json": NFKMLXRepositoryFixture.json(tokenizer),
            "model.safetensors": weights], configuration: configuration, license: nil)
    }

    func testASmolVLMRepositoryIsRead() throws {
        let descriptor = try NFKMLXModelDescriptors.descriptor(for: try smolVLMRepository(),
                                                               matching: try NFKMLXRepositoryFixture.template("smolvlm2-500m"))
        XCTAssertEqual(descriptor.family, "smolvlm")
    }

    func testSmolVLMFieldsTheLoaderDoesNotReadAreRefused() throws {
        var tower = Self.smolVLM
        var vision = try XCTUnwrap(tower["vision_config"] as? [String: Any])
        vision["hidden_act"] = "relu"
        tower["vision_config"] = vision
        var decoder = Self.smolVLM
        var text = try XCTUnwrap(decoder["text_config"] as? [String: Any])
        text["hidden_act"] = "gelu"
        decoder["text_config"] = text
        for configuration in [tower, decoder] {
            try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try smolVLMRepository(configuration),
                                                      template: "smolvlm2-500m")
        }
        var tokenizer = Self.smolLMTokenizer
        tokenizer["pre_tokenizer"] = ["type": "Sequence", "pretokenizers": [
            ["type": "Digits", "individual_digits": false],
            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true]]]
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedTokenizer, try smolVLMRepository(tokenizer: tokenizer),
                                                  template: "smolvlm2-500m")
    }

    // MARK: TrOCR

    private static func trOCR(encoder: String = "vit", decoder: String = "trocr") -> [String: Any] {
        ["model_type": "vision-encoder-decoder", "architectures": ["VisionEncoderDecoderModel"],
         "encoder": ["model_type": encoder, "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
                     "intermediate_size": 128, "image_size": 384, "patch_size": 16, "hidden_act": "gelu"],
         "decoder": ["model_type": decoder, "d_model": 64, "decoder_layers": 2, "decoder_attention_heads": 2,
                     "decoder_ffn_dim": 128, "vocab_size": 256, "pad_token_id": 1, "eos_token_id": 2,
                     "activation_function": "gelu", "max_position_embeddings": 512]]
    }

    private func trOCRRepository(_ configuration: [String: Any] = trOCR(),
                                 tokenizer: [String: Data]? = nil, tokenizerClass: String = "RobertaTokenizer") throws -> NFKModelRepository {
        var files: [String: Data] = [
            "config.json": NFKMLXRepositoryFixture.json(configuration),
            "tokenizer_config.json": NFKMLXRepositoryFixture.json(["tokenizer_class": tokenizerClass]),
            "model.safetensors": NFKMLXRepositoryFixture.safetensors([("decoder.model.decoder.layers.0.fc1.weight", [128, 64])],
                                                                     dtype: "F32", elementBytes: 4)]
        files.merge(tokenizer ?? ["vocab.json": NFKMLXRepositoryFixture.json(["a": 0]),
                                  "merges.txt": Data("#version: 0.2\n".utf8)]) { $1 }
        return try fixture.repository("org/trocr-tiny", files: files, configuration: configuration, license: nil)
    }

    func testATrOCRRepositoryIsRead() throws {
        let descriptor = try NFKMLXModelDescriptors.descriptor(
            for: try trOCRRepository(), matching: try NFKMLXRepositoryFixture.template("trocr-base-printed"))
        XCTAssertEqual(descriptor.family, "trocr")
    }

    func testAnotherVisionEncoderDecoderIsRefused() throws {
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedArchitecture,
                                                  try trOCRRepository(Self.trOCR(encoder: "donut-swin", decoder: "mbart")),
                                                  template: "trocr-base-printed")
    }

    func testATrOCRTokenizerTheLoaderDoesNotReadIsRefused() throws {
        try NFKMLXRepositoryFixture.assertRefusal(
            .unsupportedTokenizer,
            try trOCRRepository(tokenizer: ["tokenizer.json": NFKMLXRepositoryFixture.json(NFKMLXRepositoryFixture.byteLevelTokenizer())]),
            template: "trocr-base-printed")
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedTokenizer, try trOCRRepository(tokenizerClass: "BertTokenizer"),
                                                  template: "trocr-base-printed")
    }
}
