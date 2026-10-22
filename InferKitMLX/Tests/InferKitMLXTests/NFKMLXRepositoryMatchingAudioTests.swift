//
//  NFKMLXRepositoryMatchingAudioTests.swift
//  InferKitMLXTests
//
//  The speech encoders' judgments of another repository, from files placed in a hub cache: the template
//  a CTC or an encoder repository takes within its family, a fine-tuned head's vocabulary carried into
//  the source, and each refusal and its code. Nothing here reaches MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXRepositoryMatchingAudioTests: XCTestCase {

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

    /// A Wav2Vec2 configuration with `architecture`'s head, a CTC head scoring four characters.
    private static func configuration(_ architecture: String = "Wav2Vec2ForCTC",
                                      modelType: String = "wav2vec2") -> [String: Any] {
        ["model_type": modelType, "architectures": [architecture], "hidden_size": 8, "num_hidden_layers": 2,
         "num_attention_heads": 2, "intermediate_size": 16, "vocab_size": 4, "hidden_act": "gelu",
         "feat_extract_activation": "gelu", "feat_extract_norm": "group", "do_stable_layer_norm": false]
    }

    private static let vocabulary = ["<pad>": 0, "<s>": 1, "</s>": 2, "|": 3]

    /// Two encoder layers' projections in float32.
    private static var weights: Data {
        NFKMLXRepositoryFixture.safetensors([("wav2vec2.encoder.layers.0.attention.q_proj.weight", [8, 8]),
                                             ("wav2vec2.encoder.layers.1.attention.q_proj.weight", [8, 8]),
                                             ("lm_head.weight", [4, 8])], dtype: "F32", elementBytes: 4)
    }

    private func repository(_ configuration: [String: Any] = configuration(),
                            vocabulary: [String: Int]? = vocabulary, tokenizerClass: String? = nil,
                            weights: [String: Data]? = nil, name: String = "org/wav2vec2-tiny") throws -> NFKModelRepository {
        var files: [String: Data] = ["config.json": NFKMLXRepositoryFixture.json(configuration),
                                     "preprocessor_config.json": NFKMLXRepositoryFixture.json(["do_normalize": true])]
        files["vocab.json"] = vocabulary.map(NFKMLXRepositoryFixture.json)
        files["tokenizer_config.json"] = tokenizerClass.map { NFKMLXRepositoryFixture.json(["tokenizer_class": $0]) }
        files.merge(weights ?? ["model.safetensors": Self.weights]) { $1 }
        return try fixture.repository(name, files: files, configuration: configuration,
                                      license: NFKModelLicense(identifier: "apache-2.0", name: "apache-2.0", url: nil))
    }

    private func descriptor(_ repository: NFKModelRepository, template identifier: String) throws -> NFKModelDescriptor {
        try NFKMLXModelDescriptors.descriptor(for: repository, matching: try NFKMLXRepositoryFixture.template(identifier))
    }

    // MARK: Templates

    func testACTCRepositoryTakesTheFamilysTranscribingTemplate() throws {
        let descriptor = try descriptor(try repository(tokenizerClass: "Wav2Vec2CTCTokenizer"), template: "wav2vec2-base")
        XCTAssertEqual(descriptor.registryName, "wav2vec2-base-960h")
        XCTAssertTrue(descriptor.tasks.contains(NFKModelTaskTranscription))
        XCTAssertEqual(descriptor.family, "wav2vec2")
        let paths = Set(descriptor.defaultVariant?.source?.files.map(\.path) ?? [])
        XCTAssertTrue(paths.isSuperset(of: ["config.json", "vocab.json", "model.safetensors"]), "\(paths)")
        XCTAssertEqual(descriptor.defaultVariant?.parameterCount, 160)
    }

    func testAnEncoderRepositoryTakesTheFamilysEncoderTemplate() throws {
        let descriptor = try descriptor(try repository(Self.configuration("Wav2Vec2Model"), vocabulary: nil,
                                                       weights: Self.bothFormats),
                                        template: "wav2vec2-base-960h")
        XCTAssertEqual(descriptor.registryName, "wav2vec2-base")
        XCTAssertFalse(descriptor.tasks.contains(NFKModelTaskTranscription))
        assertSourceWeights(descriptor, ["model.safetensors"])
    }

    func testAHuBERTRepositoryStaysInItsFamily() throws {
        let descriptor = try descriptor(try repository(Self.configuration("HubertForCTC", modelType: "hubert"),
                                                       weights: Self.bothFormats, name: "org/hubert-tiny"),
                                        template: "hubert-base-ls960")
        XCTAssertEqual(descriptor.family, "hubert")
        XCTAssertTrue(descriptor.tasks.contains(NFKModelTaskTranscription))
        assertSourceWeights(descriptor, ["model.safetensors"])
    }

    /// The safetensors weights beside a `pytorch_model.bin`, as many releases ship both.
    private static var bothFormats: [String: Data] {
        ["model.safetensors": weights, "pytorch_model.bin": Data(count: 16)]
    }

    /// Asserts the variant downloads exactly `paths` as its weights: the files the judgment measured.
    private func assertSourceWeights(_ descriptor: NFKModelDescriptor, _ paths: [String],
                                     file: StaticString = #filePath, line: UInt = #line) {
        let weights = (descriptor.defaultVariant?.source?.files ?? []).filter { $0.role == NFKModelFileRoleWeights }
        XCTAssertEqual(weights.map(\.path), paths, file: file, line: line)
        XCTAssertTrue(weights.allSatisfy { !$0.isOptional }, file: file, line: line)
    }

    func testAFineTunedW2VBertCarriesItsVocabulary() throws {
        var configuration = Self.configuration("Wav2Vec2BertForCTC", modelType: "wav2vec2-bert")
        configuration["hidden_act"] = "swish"
        configuration["feat_extract_activation"] = nil
        let descriptor = try descriptor(try repository(configuration, tokenizerClass: "Wav2Vec2CTCTokenizer",
                                                       name: "org/w2v-bert-ctc"),
                                        template: NFKMLXWav2Vec2Bert.modelName)
        let paths = Set(descriptor.defaultVariant?.source?.files.map(\.path) ?? [])
        XCTAssertTrue(paths.isSuperset(of: ["vocab.json", "tokenizer_config.json"]), "\(paths)")
    }

    // MARK: Refusals

    func testAHeadTheLoaderDoesNotBuildIsRefused() throws {
        try NFKMLXRepositoryFixture.assertRefusal(
            .unsupportedArchitecture, try repository(Self.configuration("Wav2Vec2ForSequenceClassification")),
            template: "wav2vec2-base-960h")
    }

    func testAnActivationTheReaderRefusesIsRefused() throws {
        var configuration = Self.configuration()
        configuration["hidden_act"] = "relu"
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(configuration),
                                                  template: "wav2vec2-base-960h")
    }

    func testACTCHeadWithoutItsVocabularyIsRefused() throws {
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedTokenizer, try repository(vocabulary: nil),
                                                  template: "wav2vec2-base-960h")
    }

    func testAVocabularyPastTheHeadIsRefused() throws {
        var vocabulary = Self.vocabulary
        vocabulary["A"] = 4
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedTokenizer, try repository(vocabulary: vocabulary),
                                                  template: "wav2vec2-base-960h")
    }

    func testAPhonemeTokenizerIsRefused() throws {
        try NFKMLXRepositoryFixture.assertRefusal(
            .unsupportedTokenizer, try repository(tokenizerClass: "Wav2Vec2PhonemeCTCTokenizer"),
            template: "wav2vec2-base-960h")
    }

    func testShardedWeightsAreRefused() throws {
        let shard = Self.weights
        let index = NFKMLXRepositoryFixture.json(["weight_map": ["lm_head.weight": "model-00001-of-00001.safetensors"]])
        try NFKMLXRepositoryFixture.assertRefusal(
            .unsupportedWeightFormat,
            try repository(weights: ["model.safetensors.index.json": index, "model-00001-of-00001.safetensors": shard]),
            template: "wav2vec2-base-960h")
    }

    func testAQuantizedSpeechRepositoryIsRefused() throws {
        var configuration = Self.configuration()
        configuration["quantization_config"] = ["quant_method": "bitsandbytes", "load_in_8bit": true]
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedQuantization, try repository(configuration),
                                                  template: "wav2vec2-base-960h")
    }
}
