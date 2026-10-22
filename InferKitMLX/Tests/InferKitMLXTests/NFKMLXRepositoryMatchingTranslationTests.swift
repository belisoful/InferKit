//
//  NFKMLXRepositoryMatchingTranslationTests.swift
//  InferKitMLXTests
//
//  The translators' judgments of another repository, from files placed in a hub cache: Marian, M2M-100
//  and SMaLL-100, and NLLB-200, the last two telling their shared `m2m_100` model type apart by tokenizer.
//  Nothing here reaches MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXRepositoryMatchingTranslationTests: XCTestCase {

    private var fixture: NFKMLXRepositoryFixture!

    override func setUp() {
        super.setUp()
        fixture = NFKMLXRepositoryFixture()
    }

    override func tearDown() {
        fixture.remove()
        super.tearDown()
    }

    /// A two-layer encoder-decoder of `modelType` with eight heads a side.
    private static func configuration(_ modelType: String) -> [String: Any] {
        ["model_type": modelType, "d_model": 64, "encoder_layers": 2, "decoder_layers": 2, "encoder_attention_heads": 8,
         "decoder_attention_heads": 8, "encoder_ffn_dim": 128, "decoder_ffn_dim": 128, "vocab_size": 256,
         "pad_token_id": 1, "eos_token_id": 2, "activation_function": modelType == "marian" ? "swish" : "relu",
         "max_position_embeddings": 512, "scale_embedding": true]
    }

    private static let weights = NFKMLXRepositoryFixture.safetensors([("model.encoder.layers.0.fc1.weight", [128, 64]),
                                                                      ("model.decoder.layers.0.fc1.weight", [128, 64])],
                                                                     dtype: "F32", elementBytes: 4)

    private func repository(_ configuration: [String: Any], files: [String: Data]) throws -> NFKModelRepository {
        var all = files
        all["config.json"] = NFKMLXRepositoryFixture.json(configuration)
        all["model.safetensors"] = Self.weights
        return try fixture.repository("org/translator-tiny", files: all, configuration: configuration, license: nil)
    }

    private static let blob = Data("piece".utf8)

    private static var marianFiles: [String: Data] {
        ["source.spm": blob, "target.spm": blob, "vocab.json": NFKMLXRepositoryFixture.json(["a": 0]),
         "tokenizer_config.json": NFKMLXRepositoryFixture.json(["source_lang": "en", "target_lang": "fr"])]
    }

    private static func m2m100Files(tokenizerClass: String?) -> [String: Data] {
        var files: [String: Data] = ["sentencepiece.bpe.model": blob, "vocab.json": NFKMLXRepositoryFixture.json(["a": 0])]
        files["tokenizer_config.json"] = tokenizerClass.map { NFKMLXRepositoryFixture.json(["tokenizer_class": $0]) }
        return files
    }

    private static var nllbFiles: [String: Data] {
        ["sentencepiece.bpe.model": blob,
         "special_tokens_map.json": NFKMLXRepositoryFixture.json(["additional_special_tokens": ["eng_Latn", "fra_Latn"]]),
         "tokenizer_config.json": NFKMLXRepositoryFixture.json(["tokenizer_class": "NllbTokenizer"])]
    }

    private func descriptor(_ repository: NFKModelRepository, template identifier: String) throws -> NFKModelDescriptor {
        try NFKMLXModelDescriptors.descriptor(for: repository, matching: try NFKMLXRepositoryFixture.template(identifier))
    }

    // MARK: Read

    func testAMarianRepositoryIsRead() throws {
        let descriptor = try descriptor(try repository(Self.configuration("marian"), files: Self.marianFiles),
                                        template: NFKMLXMarian.modelName)
        XCTAssertEqual(descriptor.family, "marian")
    }

    func testAnM2M100RepositoryIsReadAndSMaLL100TakesItsEntry() throws {
        let m2m100 = try descriptor(try repository(Self.configuration("m2m_100"), files: Self.m2m100Files(tokenizerClass: nil)),
                                    template: "m2m100")
        XCTAssertEqual(m2m100.registryName, "m2m100")
        var small100 = Self.m2m100Files(tokenizerClass: "M2M100Tokenizer")
        small100["tokenization_small100.py"] = Data("# tokenizer".utf8)
        let retemplated = try descriptor(try repository(Self.configuration("m2m_100"), files: small100), template: "m2m100")
        XCTAssertEqual(retemplated.registryName, "small100")
        let plain = try descriptor(try repository(Self.configuration("m2m_100"), files: Self.m2m100Files(tokenizerClass: nil)),
                                   template: "small100")
        XCTAssertEqual(plain.registryName, "m2m100", "an M2M-100 repository takes M2M-100's entry from SMaLL-100's")
    }

    func testAShardedTranslatorDownloadsItsIndexAndShards() throws {
        let shard = Self.weights
        let index = NFKMLXRepositoryFixture.json(["weight_map": ["model.encoder.layers.0.fc1.weight": "model-00001-of-00002.safetensors",
                                                                 "model.decoder.layers.0.fc1.weight": "model-00002-of-00002.safetensors"]])
        var files = Self.marianFiles
        files["model.safetensors.index.json"] = index
        files["model-00001-of-00002.safetensors"] = shard
        files["model-00002-of-00002.safetensors"] = shard
        files["pytorch_model.bin"] = Data(count: 16)
        let configuration = Self.configuration("marian")
        var all = files
        all["config.json"] = NFKMLXRepositoryFixture.json(configuration)
        let read = try fixture.repository("org/marian-sharded", files: all, configuration: configuration, license: nil)
        let descriptor = try descriptor(read, template: NFKMLXMarian.modelName)
        let weights = (descriptor.defaultVariant?.source?.files ?? []).filter { $0.role == NFKModelFileRoleWeights }.map(\.path)
        XCTAssertEqual(weights, ["model.safetensors.index.json", "model-00001-of-00002.safetensors",
                                 "model-00002-of-00002.safetensors"])
    }

    func testAnNLLBRepositoryIsReadByNLLBAndNotByM2M100() throws {
        let nllbTemplate = try XCTUnwrap(NFKMLXModelDescriptors.entries.first { $0.family == "nllb" }?.identifier)
        let read = try repository(Self.configuration("m2m_100"), files: Self.nllbFiles)
        XCTAssertEqual(try descriptor(read, template: nllbTemplate).family, "nllb")
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedTokenizer, read, template: "m2m100")
        let m2m100 = try repository(Self.configuration("m2m_100"), files: Self.m2m100Files(tokenizerClass: "M2M100Tokenizer"))
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedTokenizer, m2m100, template: nllbTemplate)
    }

    // MARK: Refusals

    func testSeparateVocabulariesAreRefused() throws {
        var files = Self.marianFiles
        files["tokenizer_config.json"] = NFKMLXRepositoryFixture.json(["separate_vocabs": true])
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedTokenizer,
                                                  try repository(Self.configuration("marian"), files: files),
                                                  template: NFKMLXMarian.modelName)
        var unshared = Self.configuration("marian")
        unshared["share_encoder_decoder_embeddings"] = false
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration, try repository(unshared, files: Self.marianFiles),
                                                  template: NFKMLXMarian.modelName)
    }

    func testADecoderHeadCountOtherThanTheEncodersIsRefused() throws {
        var configuration = Self.configuration("marian")
        configuration["decoder_attention_heads"] = 4
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedConfiguration,
                                                  try repository(configuration, files: Self.marianFiles),
                                                  template: NFKMLXMarian.modelName)
    }

    func testATranslatorWithoutItsTokenizerFilesIsRefused() throws {
        var files = Self.marianFiles
        files["target.spm"] = nil
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedTokenizer,
                                                  try repository(Self.configuration("marian"), files: files),
                                                  template: NFKMLXMarian.modelName)
        var unnamed = Self.nllbFiles
        unnamed["tokenizer_config.json"] = nil
        let nllbTemplate = try XCTUnwrap(NFKMLXModelDescriptors.entries.first { $0.family == "nllb" }?.identifier)
        try NFKMLXRepositoryFixture.assertRefusal(.unsupportedTokenizer,
                                                  try repository(Self.configuration("m2m_100"), files: unnamed),
                                                  template: nllbTemplate)
    }
}
