//
//  NFKMLXTranslationTests.swift
//  InferKitMLXTests
//
//  The SentencePiece reader, the BART-family and T5 encoder-decoders, the greedy and beam decoders, and
//  the translation backend. The parity tests load a release directory (IK_VAL_MARIAN / IK_VAL_M2M100 /
//  IK_VAL_MADLAD) and compare seam by seam against the record `run_reference.py marian|m2m100|madlad`
//  writes (IK_PARITY_MARIAN / IK_PARITY_M2M100 / IK_PARITY_MADLAD).
//

import XCTest
import InferKit
import MLX
import MLXFast
import MLXNN
@testable import InferKitMLX

final class NFKMLXTranslationTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        eval(mine)
        let a = mine.reshaped([-1]).asArray(Float.self), b = reference.reshaped([-1]).asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    private static let sentences = [
        "Hello world! How are you?",
        "The quick brown fox jumps over the lazy dog.",
        "  spaced   out  text  ",
        "na\u{ef}ve caf\u{e9} \u{2014} r\u{e9}sum\u{e9} 2024",
        "\u{fb01}ne \u{bd}",
    ]
    private static let target = "Der schnelle braune Fuchs springt \u{fc}ber den faulen Hund."

    override func tearDown() {
        // The parity tests load one release after another; clearing reaches MLX's runtime, which
        // needs a Metal library it can find.
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    // MARK: SentencePiece (no MLX)

    /// A protobuf `ModelProto` assembled by hand: two normal pieces, the unknown marker, and a
    /// trainer spec naming the algorithm.
    private func syntheticModel(kind: UInt8, pieces: [(String, Float)], addsDummyPrefix: Bool = true,
                                removesExtraWhitespace: Bool = true) throws -> NFKMLXSentencePieceModel {
        func varint(_ value: Int) -> [UInt8] {
            var v = value, out = [UInt8]()
            repeat {
                var byte = UInt8(v & 0x7f)
                v >>= 7
                if v != 0 { byte |= 0x80 }
                out.append(byte)
            } while v != 0
            return out
        }
        func lengthDelimited(field: Int, _ bytes: [UInt8]) -> [UInt8] { varint(field << 3 | 2) + varint(bytes.count) + bytes }
        var body = [UInt8]()
        var all: [(String, Float, Int)] = [("<unk>", 0, 2)]
        all += pieces.map { ($0.0, $0.1, 1) }
        for (text, score, type) in all {
            var piece = lengthDelimited(field: 1, Array(text.utf8))
            piece += varint(2 << 3 | 5) + withUnsafeBytes(of: score.bitPattern.littleEndian) { Array($0) }
            piece += varint(3 << 3 | 0) + varint(type)
            body += lengthDelimited(field: 1, piece)
        }
        body += lengthDelimited(field: 2, varint(3 << 3 | 0) + varint(Int(kind)))
        var spec = lengthDelimited(field: 1, Array("nmt_nfkc".utf8))
        spec += varint(3 << 3 | 0) + varint(addsDummyPrefix ? 1 : 0)
        spec += varint(4 << 3 | 0) + varint(removesExtraWhitespace ? 1 : 0)
        body += lengthDelimited(field: 3, spec)
        return try NFKMLXSentencePieceModel(data: Data(body))
    }

    func testTheReaderWalksASyntheticUnigramModel() throws {
        let model = try syntheticModel(kind: 1, pieces: [("\u{2581}he", -1), ("llo", -2), ("\u{2581}hello", -2.5), ("l", -5), ("o", -5), ("\u{2581}", -3)])
        XCTAssertEqual(model.kind, .unigram)
        XCTAssertEqual(model.pieces.count, 7)
        XCTAssertEqual(model.unknownId, 0)
        XCTAssertTrue(model.appliesNFKC)
        let segmenter = NFKMLXSentencePieceSegmenter(model: model)
        // ▁hello (-2.5) beats ▁he + llo (-3).
        XCTAssertEqual(segmenter.pieces(for: "hello"), ["\u{2581}hello"])
        XCTAssertEqual(segmenter.decode(segmenter.encode("hello")), "hello")
    }

    func testAnUncoveredCharacterBecomesTheUnknownPiece() throws {
        let model = try syntheticModel(kind: 1, pieces: [("\u{2581}a", -1), ("b", -1), ("\u{2581}", -2)])
        let segmenter = NFKMLXSentencePieceSegmenter(model: model)
        XCTAssertEqual(segmenter.encode("a\u{e9}b"), [1, 0, 2], "é has no piece")
        XCTAssertEqual(segmenter.encode("a\u{e9}\u{e9}b"), [1, 0, 2], "consecutive unknowns fuse")
    }

    func testTheBPESegmenterMergesByScore() throws {
        // Scores rank the merges: "ab" first, then "abc".
        let model = try syntheticModel(kind: 2, pieces: [("a", -10), ("b", -10), ("c", -10), ("ab", -1), ("abc", -2), ("bc", -3), ("\u{2581}", -4)])
        let segmenter = NFKMLXSentencePieceSegmenter(model: model)
        XCTAssertEqual(model.kind, .bpe)
        XCTAssertEqual(segmenter.pieces(for: "abc"), ["\u{2581}", "abc"])
        XCTAssertEqual(segmenter.pieces(for: "bc"), ["\u{2581}", "bc"])
    }

    func testNormalizationFollowsNMTNFKC() throws {
        XCTAssertEqual(NFKMLXSentencePieceSegmenter.nmtNFKC("\u{fb01}ne \u{bd}"), "fine 1\u{2044}2")
        XCTAssertEqual(NFKMLXSentencePieceSegmenter.nmtNFKC("a\u{a0}b\u{200b}c"), "a bc")
        // A proto without a character map falls back to NFKC and still applies the spec's whitespace rules.
        let segmenter = NFKMLXSentencePieceSegmenter(model: try syntheticModel(kind: 1, pieces: [("a", -1)]))
        XCTAssertFalse(segmenter.normalizer.hasCharacterMap)
        XCTAssertEqual(segmenter.normalize("  spaced   out  text  "), "\u{2581}spaced\u{2581}out\u{2581}text")
        XCTAssertEqual(segmenter.normalize("   "), "")
        XCTAssertEqual(segmenter.normalize("x", dummyPrefix: false), "x")
        // A model that keeps whitespace (InternLM2's spec) keeps every space of a run.
        let keeping = NFKMLXSentencePieceSegmenter(model: try syntheticModel(kind: 2, pieces: [("a", -1)], addsDummyPrefix: false,
                                                                            removesExtraWhitespace: false))
        XCTAssertEqual(keeping.normalize("  leading  x "), "\u{2581}\u{2581}leading\u{2581}\u{2581}x\u{2581}")
        XCTAssertEqual(keeping.normalize("  x", dummyPrefix: true), "\u{2581}\u{2581}\u{2581}x")
    }

    /// OPUS-MT's `vocab.json` is the union of the source and target vocabularies, so a character the
    /// source model does not know can still have a release id; MarianTokenizer finds it by the
    /// unknown run's text, and decodes such a piece as itself.
    func testAnUnknownRunMapsThroughTheReleaseTableByItsText() throws {
        let model = try syntheticModel(kind: 1, pieces: [("\u{2581}na", -1), ("ve", -1), ("\u{2581}", -2)])
        let segmenter = NFKMLXSentencePieceSegmenter(model: model)
        let segments = segmenter.segments(of: "na\u{ef}ve")
        XCTAssertEqual(segments.map(\.id), [1, 0, 2])
        XCTAssertEqual(segments.map(\.surface), ["\u{2581}na", "\u{ef}", "ve"])
        XCTAssertEqual(segmenter.segments(of: "na\u{ef}\u{ef}ve").map(\.surface), ["\u{2581}na", "\u{ef}\u{ef}", "ve"], "a fused run keeps its whole text")
        let tokenizer = NFKMLXSentencePieceTokenizer(
            segmenter: segmenter, vocabularyEntries: [("\u{2581}na", 10), ("ve", 11), ("\u{ef}", 12), ("<unk>", 1)], eosTokenId: 0)
        XCTAssertEqual(tokenizer.encode("na\u{ef}ve", dummyPrefix: nil), [10, 12, 11])
        XCTAssertEqual(tokenizer.encode("na\u{ef}\u{ef}ve", dummyPrefix: nil), [10, 1, 11], "a run the table does not name is unknown")
        XCTAssertEqual(tokenizer.decode(ids: [10, 12, 11]), "na\u{ef}ve")
    }

    /// SentencePiece matches pieces byte for byte, and a vocabulary carries both a composed and a
    /// decomposed spelling, or both orders of two combining marks, as distinct pieces; a Swift `String`
    /// key would merge them.
    func testCanonicallyEquivalentPiecesStayDistinct() throws {
        let composed = "\u{e9}", decomposed = "e\u{301}"
        let fathaShadda = "\u{64e}\u{651}", shaddaFatha = "\u{651}\u{64e}"
        let model = try syntheticModel(kind: 1, pieces: [(composed, -1), (decomposed, -2), (fathaShadda, -1), (shaddaFatha, -2), ("\u{2581}", -3)])
        let segmenter = NFKMLXSentencePieceSegmenter(model: model)
        XCTAssertEqual(segmenter.id(of: composed), 1)
        XCTAssertEqual(segmenter.id(of: decomposed), 2)
        XCTAssertEqual(segmenter.id(of: fathaShadda), 3)
        XCTAssertEqual(segmenter.id(of: shaddaFatha), 4)
        XCTAssertEqual(segmenter.encodeNormalized(shaddaFatha), [4], "the text's own mark order picks its own piece")
        XCTAssertEqual(segmenter.encodeNormalized(fathaShadda), [3])
        let tokenizer = NFKMLXSentencePieceTokenizer(segmenter: segmenter,
                                                     vocabularyEntries: [(composed, 10), (decomposed, 11), ("<unk>", 1)], eosTokenId: 0)
        XCTAssertEqual(tokenizer.id(ofPiece: composed), 10)
        XCTAssertEqual(tokenizer.id(ofPiece: decomposed), 11)
        XCTAssertEqual(tokenizer.piece(ofId: 11), decomposed)
    }

    /// The character map of a released model, against values SentencePiece 0.2.2 gives for the same
    /// strings: zero-width joiner to a space, combining marks left in their order, compatibility
    /// characters folded, decomposed sequences composed.
    func testTheNormalizerRunsTheModelsCharacterMap() throws {
        guard let directory = NFKMLXValidationConfig.environment["IK_VAL_M2M100"],
              FileManager.default.fileExists(atPath: directory) else {
            throw XCTSkip("set IK_VAL_M2M100 to an M2M-100 release directory")
        }
        let segmenter = try NFKMLXSentencePieceSegmenter(contentsOf: URL(fileURLWithPath: directory).appendingPathComponent("sentencepiece.bpe.model"))
        XCTAssertTrue(segmenter.normalizer.hasCharacterMap)
        XCTAssertEqual(segmenter.normalize("a\u{200d}b"), "\u{2581}a\u{2581}b")
        XCTAssertEqual(segmenter.normalize("\u{633}\u{651}\u{64e}"), "\u{2581}\u{633}\u{651}\u{64e}")
        XCTAssertEqual(segmenter.normalize("\u{fb01}ne \u{bd}"), "\u{2581}fine\u{2581}1\u{2044}2")
        XCTAssertEqual(segmenter.normalize("e\u{301}"), "\u{2581}\u{e9}")
        XCTAssertEqual(segmenter.normalize("Vie\u{302}\u{323}t"), "\u{2581}Vi\u{1ec7}t")
        XCTAssertEqual(segmenter.normalize("\u{1112}\u{1161}\u{11ab}"), "\u{2581}\u{d55c}")
        XCTAssertEqual(segmenter.normalize("  spaced   out  text  "), "\u{2581}spaced\u{2581}out\u{2581}text")
        XCTAssertEqual(segmenter.normalize("\u{ff21}\u{ff22}\u{3000}\u{ff76}\u{ff9e}"), "\u{2581}AB\u{2581}\u{30ac}")
        XCTAssertEqual(segmenter.normalize("x\t\ny"), "\u{2581}x\u{2581}y")
        XCTAssertEqual(segmenter.normalize("   "), "")
    }

    // MARK: Configuration and backend contract (no MLX)

    func testTheConfigurationReadsAMarianReleaseConfig() throws {
        let config: [String: Any] = ["model_type": "marian", "vocab_size": 58101, "d_model": 512, "encoder_layers": 6,
                                     "decoder_layers": 6, "encoder_attention_heads": 8, "encoder_ffn_dim": 2048,
                                     "decoder_ffn_dim": 2048, "activation_function": "swish", "scale_embedding": true,
                                     "pad_token_id": 58100, "eos_token_id": 0, "decoder_start_token_id": 58100,
                                     "max_position_embeddings": 512]
        let c = try NFKMLXSeq2SeqConfiguration(huggingFaceConfig: config)
        XCTAssertEqual(c.positions, .marianSinusoidal)
        XCTAssertFalse(c.normalizeBefore)
        XCTAssertFalse(c.finalLayerNorm)
        XCTAssertTrue(c.finalLogitsBias)
        XCTAssertTrue(c.scaleEmbedding)
        XCTAssertEqual(c.activation, .swish)
        XCTAssertEqual(c.decoderStartTokenId, 58100)
    }

    func testTheConfigurationReadsAnM2M100ReleaseConfig() throws {
        let config: [String: Any] = ["model_type": "m2m_100", "vocab_size": 128112, "d_model": 1024, "encoder_layers": 12,
                                     "decoder_layers": 12, "encoder_attention_heads": 16, "encoder_ffn_dim": 4096,
                                     "decoder_ffn_dim": 4096, "activation_function": "relu", "scale_embedding": true,
                                     "pad_token_id": 1, "eos_token_id": 2, "decoder_start_token_id": 2]
        let c = try NFKMLXSeq2SeqConfiguration(huggingFaceConfig: config)
        XCTAssertEqual(c.positions, .fairseqSinusoidal)
        XCTAssertTrue(c.normalizeBefore)
        XCTAssertTrue(c.finalLayerNorm)
        XCTAssertFalse(c.finalLogitsBias)
        XCTAssertEqual(c.activation, .relu)
    }

    func testTheConfigurationReadsAnNLLBReleaseConfig() throws {
        let config: [String: Any] = ["model_type": "m2m_100", "vocab_size": 256206, "d_model": 1024, "encoder_layers": 12,
                                     "decoder_layers": 12, "encoder_attention_heads": 16, "encoder_ffn_dim": 4096,
                                     "decoder_ffn_dim": 4096, "activation_function": "relu", "scale_embedding": true,
                                     "pad_token_id": 1, "eos_token_id": 2, "decoder_start_token_id": 2, "bos_token_id": 0]
        let c = try NFKMLXSeq2SeqConfiguration(huggingFaceConfig: config)
        XCTAssertEqual(c.vocabularySize, 256206)
        XCTAssertEqual(c.positions, .fairseqSinusoidal)
        XCTAssertTrue(c.normalizeBefore)
        XCTAssertEqual(c.decoderStartTokenId, 2)
    }

    /// The release numbers its vocabulary as fairseq did: four specials, every model piece one past
    /// its id, the language codes after the vocabulary in the listed order, then `<mask>`.
    func testTheNLLBReleaseTableFollowsFairseq() throws {
        // A release's model file holds <unk>, <s>, </s> first; the synthetic one puts <unk> first on its own.
        let model = try syntheticModel(kind: 1, pieces: [("<s>", 0), ("</s>", 0), ("\u{2581}he", -1), ("llo", -2), ("\u{2581}", -3)])
        let segmenter = NFKMLXSentencePieceSegmenter(model: model)
        let (entries, languageIds) = NFKMLXNLLB.releaseTable(segmenter: segmenter, languages: ["eng_Latn", "deu_Latn"])
        XCTAssertEqual(entries.prefix(4).map(\.id), [0, 1, 2, 3])
        XCTAssertEqual(entries.prefix(4).map(\.piece), ["<s>", "<pad>", "</s>", "<unk>"])
        XCTAssertEqual(entries.first { $0.piece == "\u{2581}" }?.id, segmenter.id(of: "\u{2581}")! + 1)
        XCTAssertEqual(languageIds, ["eng_Latn": segmenter.pieceCount + 1, "deu_Latn": segmenter.pieceCount + 2])
        XCTAssertEqual(entries.last?.piece, "<mask>")
        XCTAssertEqual(entries.last?.id, segmenter.pieceCount + 3)
        let tokenizer = NFKMLXSentencePieceTokenizer(segmenter: segmenter, vocabularyEntries: entries, eosTokenId: 2, bosTokenId: 0)
        let translator = NFKMLXNLLBTranslator(net: NFKMLXSeq2SeqNet(.tinyM2M100), tokenizer: tokenizer, languageIds: languageIds,
                                              identifier: "nllb-200", beams: 5, maxTokens: 8)
        XCTAssertEqual(translator.sourceIds(for: "hello", source: "en"),
                       [languageIds["eng_Latn"]!, segmenter.id(of: "\u{2581}he")! + 1, segmenter.id(of: "llo")! + 1, 2])
        XCTAssertEqual(translator.text(of: [languageIds["deu_Latn"]!, segmenter.id(of: "\u{2581}he")! + 1, segmenter.id(of: "llo")! + 1, 2]), "hello",
                       "the marker, the end token, and the leading space are left out")
        XCTAssertEqual(translator.text(of: [3, segmenter.id(of: "\u{2581}he")! + 1]), "he", "the unknown token is left out, as the reference's decode leaves it")
    }

    func testNLLBResolvesLanguageTagsToItsCodes() throws {
        let model = try syntheticModel(kind: 1, pieces: [("<s>", 0), ("</s>", 0), ("a", -1)])
        let segmenter = NFKMLXSentencePieceSegmenter(model: model)
        let (entries, languageIds) = NFKMLXNLLB.releaseTable(segmenter: segmenter, languages: NFKMLXNLLBTranslator.languages)
        let translator = NFKMLXNLLBTranslator(
            net: NFKMLXSeq2SeqNet(.tinyM2M100), tokenizer: NFKMLXSentencePieceTokenizer(segmenter: segmenter, vocabularyEntries: entries, eosTokenId: 2),
            languageIds: languageIds, identifier: "nllb-200", beams: 5, maxTokens: 8)
        XCTAssertEqual(languageIds.count, 202)
        XCTAssertEqual(translator.code(for: "en"), "eng_Latn")
        XCTAssertEqual(translator.code(for: "en-GB"), "eng_Latn")
        XCTAssertEqual(translator.code(for: "zh"), "zho_Hans")
        XCTAssertEqual(translator.code(for: "zh-TW"), "zho_Hant")
        XCTAssertEqual(translator.code(for: "zh-Hant"), "zho_Hant")
        XCTAssertEqual(translator.code(for: "yue"), "yue_Hant")
        XCTAssertEqual(translator.code(for: "ar"), "arb_Arab", "Standard Arabic for the macrolanguage")
        XCTAssertEqual(translator.code(for: "fa"), "pes_Arab")
        XCTAssertEqual(translator.code(for: "no"), "nob_Latn")
        XCTAssertEqual(translator.code(for: "sr"), "srp_Cyrl")
        XCTAssertEqual(translator.code(for: "sr-Latn"), nil, "the release writes Serbian in Cyrillic only")
        XCTAssertEqual(translator.code(for: "ja"), "jpn_Jpan")
        XCTAssertEqual(translator.code(for: "ko"), "kor_Hang")
        XCTAssertEqual(translator.code(for: "pt-BR"), "por_Latn")
        XCTAssertEqual(translator.code(for: "ace"), nil, "two scripts and no tag to choose one")
        XCTAssertEqual(translator.code(for: "ace-Latn"), "ace_Latn")
        XCTAssertEqual(translator.code(for: "eng_Latn"), "eng_Latn", "an NLLB code passes through")
        XCTAssertNil(translator.code(for: "xx"))
        XCTAssertTrue(translator.supports(language: "hi"))
    }

    func testASeq2SeqReleaseMayShardItsPyTorchWeights() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("seq2seq-shards-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let index = ["weight_map": ["model.shared.weight": "pytorch_model-00001-of-00002.bin", "model.encoder.layers.0.fc1.weight": "pytorch_model-00002-of-00002.bin",
                                    "model.encoder.layers.0.fc2.weight": "pytorch_model-00001-of-00002.bin"]]
        try JSONSerialization.data(withJSONObject: index).write(to: directory.appendingPathComponent("pytorch_model.bin.index.json"))
        XCTAssertEqual(try NFKMLXSeq2SeqNet.weightFiles(in: directory).map(\.lastPathComponent),
                       ["pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"])
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("seq2seq-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        XCTAssertThrowsError(try NFKMLXSeq2SeqNet.weightFiles(in: empty))
    }

    func testLanguageTagsCanonicalize() {
        XCTAssertEqual(NFKMLXTranslationBackend.canonical("PT_br"), "pt-br")
        XCTAssertEqual(NFKMLXTranslationBackend.primary("zh-Hant-TW"), "zh")
        XCTAssertEqual(NFKMLXTranslationBackend.script("zh-Hant-TW"), "Hant")
        XCTAssertNil(NFKMLXTranslationBackend.script("pt-BR"))
    }

    func testParagraphsSplitAndTheirSeparatorsSurvive() {
        let segments = NFKMLXTranslationBackend.segments(of: "One.\n\n  Two.  \nThree.", splitsSentences: false)
        var texts = [String](), separators = [String]()
        for segment in segments {
            switch segment {
            case .text(let t): texts.append(t)
            case .separator(let s): separators.append(s)
            }
        }
        XCTAssertEqual(texts, ["One.", "Two.", "Three."])
        XCTAssertEqual(separators, ["\n", "\n", "  ", "  ", "\n"])
    }

    func testDetectionNamesEnglish() {
        XCTAssertEqual(NFKMLXTranslationBackend.detectLanguage(of: "The quick brown fox jumps over the lazy dog."), "en")
    }

    func testADecoderOnlyConfigurationReadsAnOutsideMemory() throws {
        try requireMLXRuntime()
        // TrOCR: no encoder layers, a 768-wide memory under a 1024-wide decoder, and an untied head.
        let config: [String: Any] = ["model_type": "trocr", "vocab_size": 64, "d_model": 16, "decoder_layers": 2,
                                     "decoder_attention_heads": 2, "decoder_ffn_dim": 32, "activation_function": "gelu",
                                     "cross_attention_hidden_size": 12, "tie_word_embeddings": false,
                                     "pad_token_id": 1, "eos_token_id": 2, "decoder_start_token_id": 2,
                                     "max_position_embeddings": 32, "scale_embedding": false]
        let c = try NFKMLXSeq2SeqConfiguration(huggingFaceConfig: config)
        XCTAssertEqual(c.encoderLayers, 0)
        XCTAssertEqual(c.crossAttentionWidth, 12)
        XCTAssertTrue(c.untiedOutputProjection)
        XCTAssertTrue(c.layerNormEmbedding)
        XCTAssertEqual(c.positions, .learned)
        let net = NFKMLXSeq2SeqNet(c)
        XCTAssertNil(net.encoder)
        XCTAssertNotNil(net.lmHead)
        let names = Set(net.parameters().flattened().map(\.0))
        XCTAssertFalse(names.contains { $0.hasPrefix("encoder.") })
        XCTAssertTrue(names.contains("decoder.embed_positions.weight"))
        XCTAssertTrue(names.contains("lm_head.weight"))
        let memory = MLXRandom.normal([1, 5, 12])
        let logits = net.decode(MLXArray([Int32(2), 7, 8]).reshaped([1, 3]), memory: memory, cache: net.makeCache())
        XCTAssertEqual(logits.shape, [1, 3, 64])
        // A VisionEncoderDecoder checkpoint names the decoder under `decoder.model.decoder` and carries
        // no `shared`; the first embed_tokens supplies it and output_projection becomes lm_head.
        XCTAssertEqual(NFKMLXSeq2SeqNet.moduleKey(for: "decoder.model.decoder.layers.0.self_attn.q_proj.weight", configuration: c, hasShared: false),
                       "decoder.layers.0.self_attn.q_proj.weight")
        XCTAssertEqual(NFKMLXSeq2SeqNet.moduleKey(for: "decoder.model.decoder.embed_tokens.weight", configuration: c, hasShared: false), "shared.weight")
        XCTAssertEqual(NFKMLXSeq2SeqNet.moduleKey(for: "decoder.output_projection.weight", configuration: c, hasShared: false), "lm_head.weight")
        XCTAssertNil(NFKMLXSeq2SeqNet.moduleKey(for: "model.encoder.embed_tokens.weight", configuration: .tinyMarian, hasShared: true))
    }

    // MARK: Tiny networks

    func testTheCachedDecodeMatchesTheTeacherForcedLogits() throws {
        try requireMLXRuntime()
        for configuration in [NFKMLXSeq2SeqConfiguration.tinyMarian, .tinyM2M100] {
            let net = NFKMLXSeq2SeqNet(configuration)
            let source = MLXArray([Int32(5), 6, 7, 8]).reshaped([1, 4])
            let target = MLXArray([Int32(configuration.decoderStartTokenId), 9, 10, 11]).reshaped([1, 4])
            let whole = net(source: source, target: target)
            let memory = net.encode(source)
            let cache = net.makeCache()
            var stepped = [MLXArray]()
            for t in 0 ..< 4 {
                stepped.append(net.decode(target[0..., t ..< t + 1], memory: memory, cache: cache))
            }
            let joined = concatenated(stepped, axis: 1)
            XCTAssertEqual(cache.length, 4)
            XCTAssertGreaterThan(cosine(joined, whole), 0.9999)
            XCTAssertLessThan(abs(joined - whole).max().item(Float.self), 1e-4)
        }
    }

    func testTheT5CachedDecodeMatchesTheTeacherForcedLogits() throws {
        try requireMLXRuntime()
        let net = NFKMLXT5Seq2SeqNet(.tiny)
        let source = MLXArray([Int32(5), 6, 7, 8, 9]).reshaped([1, 5])
        let target = MLXArray([Int32(0), 9, 10, 11]).reshaped([1, 4])
        let whole = net(source: source, target: target)
        let memory = net.encode(source)
        let cache = net.makeCache()
        var stepped = [MLXArray]()
        for t in 0 ..< 4 {
            stepped.append(net.decode(target[0..., t ..< t + 1], memory: memory, cache: cache))
        }
        let joined = concatenated(stepped, axis: 1)
        XCTAssertLessThan(abs(joined - whole).max().item(Float.self), 1e-4)
        XCTAssertNotNil(net.lmHead, "MADLAD's head is untied")
    }

    func testTheCausalBucketsMirrorTheReference() {
        // bidirectional=False, 32 buckets, max distance 128: exact below 16, log-spaced after.
        XCTAssertEqual(NFKT5CachedAttention.causalBucket(0, numBuckets: 32, maxDistance: 128), 0)
        XCTAssertEqual(NFKT5CachedAttention.causalBucket(-15, numBuckets: 32, maxDistance: 128), 15)
        XCTAssertEqual(NFKT5CachedAttention.causalBucket(-16, numBuckets: 32, maxDistance: 128), 16)
        XCTAssertEqual(NFKT5CachedAttention.causalBucket(-127, numBuckets: 32, maxDistance: 128), 31)
        XCTAssertEqual(NFKT5CachedAttention.causalBucket(-1000, numBuckets: 32, maxDistance: 128), 31)
        XCTAssertEqual(NFKT5CachedAttention.causalBucket(5, numBuckets: 32, maxDistance: 128), 0, "the future has no bucket of its own")
    }

    func testBeamSearchHonorsTheForcedFirstTokenAndEnds() throws {
        try requireMLXRuntime()
        let net = NFKMLXSeq2SeqNet(.tinyM2M100)
        var decoding = NFKMLXSeq2SeqDecoding(beams: 3, maxTokens: 6, earlyStopping: true, startToken: 2, endToken: 2,
                                             forcedFirstToken: 40)
        let beam = NFKMLXSeq2SeqDecoder.generate(net, source: [5, 6, 7, 2], decoding: decoding)
        XCTAssertEqual(beam.first, 40)
        XCTAssertLessThanOrEqual(beam.count, 6)
        XCTAssertFalse(beam.dropFirst().contains(2), "the end token is not part of the output")
        decoding.beams = 1
        let greedy = NFKMLXSeq2SeqDecoder.generate(net, source: [5, 6, 7, 2], decoding: decoding)
        XCTAssertEqual(greedy.first, 40)
        decoding.suppressedTokens = [40]
        decoding.forcedFirstToken = nil
        let suppressed = NFKMLXSeq2SeqDecoder.generate(net, source: [5, 6, 7, 2], decoding: decoding)
        XCTAssertFalse(suppressed.contains(40))
    }

    func testNoRepeatNgramBansTheCompletionsTransformersBans() {
        // transformers' NoRepeatNGramLogitsProcessor: the last n-1 tokens, matched anywhere earlier.
        XCTAssertEqual(NFKMLXSeq2SeqDecoder.repeatedNgramCompletions([2, 0, 5, 7, 0, 5], size: 3), [7])
        XCTAssertEqual(NFKMLXSeq2SeqDecoder.repeatedNgramCompletions([2, 0, 0], size: 3), [])
        XCTAssertEqual(NFKMLXSeq2SeqDecoder.repeatedNgramCompletions([2, 0, 0, 0], size: 3), [0])
        XCTAssertEqual(NFKMLXSeq2SeqDecoder.repeatedNgramCompletions([4, 4], size: 1), [4, 4])
        XCTAssertEqual(NFKMLXSeq2SeqDecoder.repeatedNgramCompletions([2], size: 3), [], "shorter than the n-gram")
        XCTAssertEqual(NFKMLXSeq2SeqDecoder.repeatedNgramCompletions([2, 0, 5, 2, 0], size: 0), [])
    }

    func testBeamSearchForcesTheLastTokenAndNeverRepeatsAnNgram() throws {
        try requireMLXRuntime()
        let net = NFKMLXSeq2SeqNet(.tinyM2M100)
        var decoding = NFKMLXSeq2SeqDecoding(beams: 3, maxTokens: 12, earlyStopping: true, startToken: 2, endToken: 2,
                                             forcedLastToken: 2, noRepeatNgramSize: 2)
        for beams in [3, 1] {
            decoding.beams = beams
            let ids = [2] + NFKMLXSeq2SeqDecoder.generate(net, source: [5, 6, 7, 2], decoding: decoding)
            let bigrams = zip(ids, ids.dropFirst()).map { [$0, $1] }
            XCTAssertEqual(Set(bigrams).count, bigrams.count, "\(beams) beams repeat a bigram: \(ids)")
            XCTAssertLessThan(ids.count - 1, decoding.maxTokens, "the forced end token closes the sequence")
        }
    }

    func testTheSinusoidTablesMatchTheirFormulas() throws {
        try requireMLXRuntime()
        let marian = NFKMLXSeq2SeqNet.marianSinusoids(length: 4, channels: 8)
        let m = marian.asArray(Float.self)
        XCTAssertEqual(m[0], 0, accuracy: 1e-6)
        XCTAssertEqual(m[4], 1, accuracy: 1e-6)
        XCTAssertEqual(m[8 + 1], sinf(1 / powf(10000, 2.0 / 8)), accuracy: 1e-6)
        let fairseq = NFKMLXSeq2SeqNet.fairseqSinusoids(length: 4, channels: 8, paddingIndex: 1)
        let f = fairseq.asArray(Float.self)
        XCTAssertEqual(f[8 ..< 16].map { abs($0) }.max()!, 0, "the padding row is zero")
        XCTAssertEqual(f[16 + 1], sinf(2 * expf(-1 * logf(10000) / 3)), accuracy: 1e-6)
    }

    // MARK: Released weights

    private func release(_ weightsKey: String, _ recordKey: String) throws -> (URL, [String: MLXArray]) {
        let env = NFKMLXValidationConfig.environment
        guard let directory = env[weightsKey], let record = env[recordKey],
              FileManager.default.fileExists(atPath: directory), FileManager.default.fileExists(atPath: record) else {
            throw XCTSkip("set \(weightsKey) (release directory) and \(recordKey) (oracle record), both present on disk")
        }
        let arrays = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: record)).arrays
        return (URL(fileURLWithPath: directory), arrays)
    }

    private func ints(_ array: MLXArray) -> [Int] { array.asArray(Int32.self).map(Int.init) }

    /// A reference output sequence without its start token and from its end token on.
    private func generated(_ ids: [Int], endToken: Int) -> [Int] {
        let body = Array(ids.dropFirst())
        guard let end = body.firstIndex(of: endToken) else { return body }
        return Array(body[..<end])
    }

    private func checkSeams<Model: NFKMLXSeq2SeqDecodable>(_ model: Model, encode: (MLXArray) -> MLXArray,
                                                          decode: (MLXArray, MLXArray) -> MLXArray,
                                                          record: [String: MLXArray], decoding: NFKMLXSeq2SeqDecoding,
                                                          sourceIds: [Int], name: String, tolerance: Float = 0.999) {
        XCTAssertEqual(sourceIds, ints(record["source_ids"]!), "\(name) source ids")
        let source = MLXArray(sourceIds.map { Int32($0) }).reshaped([1, sourceIds.count])
        let memory = encode(source)
        let encoderCosine = cosine(memory[0], record["encoder_hidden"]!)
        XCTAssertGreaterThan(encoderCosine, tolerance, "\(name) encoder")
        let decoderInput = record["decoder_input"]!
        let logits = decode(decoderInput.reshaped([1, decoderInput.dim(0)]), memory)
        let logitCosine = cosine(logits[0], record["output"]!)
        XCTAssertGreaterThan(logitCosine, tolerance, "\(name) teacher-forced logits")
        let argmax = logits[0].argMax(axis: -1).asArray(Int32.self)
        let referenceArgmax = record["output"]!.argMax(axis: -1).asArray(Int32.self)
        let agreeing = zip(argmax, referenceArgmax).filter { $0 == $1 }.count
        print("\(name): encoder cosine \(encoderCosine); logit cosine \(logitCosine); argmax \(agreeing)/\(argmax.count)")
        var greedyDecoding = decoding
        greedyDecoding.beams = 1
        let greedy = NFKMLXSeq2SeqDecoder.generate(model, source: sourceIds, decoding: greedyDecoding)
        XCTAssertEqual(greedy, generated(ints(record["greedy"]!), endToken: decoding.endToken), "\(name) greedy tokens")
        var beamDecoding = decoding
        beamDecoding.beams = ints(record["beams"]!)[0]
        let beam = NFKMLXSeq2SeqDecoder.generate(model, source: sourceIds, decoding: beamDecoding)
        XCTAssertEqual(beam, generated(ints(record["beam"]!), endToken: decoding.endToken), "\(name) beam tokens")
    }

    func testMarianMatchesTheReference() throws {
        try requireMLXRuntime()
        let (directory, record) = try release("IK_VAL_MARIAN", "IK_PARITY_MARIAN")
        let translator = try NFKMLXMarian.translator(directoryURL: directory)
        XCTAssertEqual(translator.sourceLanguage, "en")
        XCTAssertEqual(translator.targetLanguage, "de")
        for (index, sentence) in Self.sentences.enumerated() {
            XCTAssertEqual(translator.sourceIds(for: sentence, target: nil), ints(record["tokens_\(index)"]!), "tokens of \(sentence)")
        }
        var decoding = translator.defaultDecoding
        decoding.maxTokens = 64
        checkSeams(translator.net, encode: { translator.net.encode($0) }, decode: { translator.net.decode($0, memory: $1, cache: translator.net.makeCache()) },
                   record: record, decoding: decoding, sourceIds: translator.sourceIds(for: Self.sentences[1], target: nil), name: "marian")
        let text = try translator.translate(Self.sentences[1], from: "en", to: "de", decoding: decoding)
        print("marian:", text)
        XCTAssertFalse(text.isEmpty)
        let targetIds = translator.targetTokenizer.encode(Self.target, dummyPrefix: nil) + [0]
        XCTAssertEqual(targetIds, ints(record["target_ids"]!), "target tokenization")
        let loss = NFKMLXTranslationObjective()(translator.net, record["source_ids"]!, record["target_ids"]!)
        XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: 1e-2, "training loss")
        print("loss mine \(loss.item(Float.self)) reference \(record["loss"]!.asArray(Float.self)[0])")
    }

    func testM2M100MatchesTheReference() throws {
        try checkM2M100(weightsKey: "IK_VAL_M2M100", recordKey: "IK_PARITY_M2M100", variant: .m418M, name: "m2m100")
    }

    func testM2M100_1_2BMatchesTheReference() throws {
        try checkM2M100(weightsKey: "IK_VAL_M2M100_1_2B", recordKey: "IK_PARITY_M2M100_1_2B", variant: .m1_2B, name: "m2m100-1.2b")
    }

    func testNLLBMatchesTheReference() throws {
        try checkNLLB(weightsKey: "IK_VAL_NLLB", recordKey: "IK_PARITY_NLLB", variant: .distilled600M, name: "nllb-200")
    }

    func testNLLBDistilled1_3BMatchesTheReference() throws {
        try checkNLLB(weightsKey: "IK_VAL_NLLB_DISTILLED_1_3B", recordKey: "IK_PARITY_NLLB_DISTILLED_1_3B", variant: .distilled1_3B, name: "nllb-200-distilled-1.3b")
    }

    func testNLLB1_3BMatchesTheReference() throws {
        try checkNLLB(weightsKey: "IK_VAL_NLLB_1_3B", recordKey: "IK_PARITY_NLLB_1_3B", variant: .m1_3B, name: "nllb-200-1.3b")
    }

    func testNLLB3_3BMatchesTheReference() throws {
        try checkNLLB(weightsKey: "IK_VAL_NLLB_3_3B", recordKey: "IK_PARITY_NLLB_3_3B", variant: .m3_3B, name: "nllb-200-3.3b")
    }

    private func checkNLLB(weightsKey: String, recordKey: String, variant: NFKMLXNLLBVariant, name: String) throws {
        try requireMLXRuntime()
        let (directory, record) = try release(weightsKey, recordKey)
        let translator = try NFKMLXNLLB.translator(directoryURL: directory, variant: variant)
        XCTAssertEqual(translator.languageIds["eng_Latn"], 256047)
        for (index, sentence) in Self.sentences.enumerated() {
            XCTAssertEqual(translator.sourceIds(for: sentence, source: "en"), ints(record["tokens_\(index)"]!), "tokens of \(sentence)")
        }
        var decoding = translator.defaultDecoding
        decoding.maxTokens = 64
        decoding.forcedFirstToken = translator.languageIds["deu_Latn"]
        checkSeams(translator.net, encode: { translator.net.encode($0) }, decode: { translator.net.decode($0, memory: $1, cache: translator.net.makeCache()) },
                   record: record, decoding: decoding, sourceIds: translator.sourceIds(for: Self.sentences[1], source: "en"), name: name)
        var greedy = decoding
        greedy.beams = 1
        let greedyText = try translator.translate(Self.sentences[1], from: "en", to: "de", decoding: greedy)
        XCTAssertEqual(greedyText, utf8(record["greedy_text"]!), "\(name) greedy text")
        var beam = decoding
        beam.beams = ints(record["beams"]!)[0]
        let beamText = try translator.translate(Self.sentences[1], from: "en", to: "de", decoding: beam)
        XCTAssertEqual(beamText, utf8(record["beam_text"]!), "\(name) beam text")
        print("\(name):", greedyText)
        XCTAssertEqual([translator.languageIds["deu_Latn"]!] + translator.tokenizer.encode(Self.target, dummyPrefix: nil) + [2],
                       ints(record["target_ids"]!), "target tokenization")
        let loss = NFKMLXTranslationObjective()(translator.net, record["source_ids"]!, record["target_ids"]!)
        XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: 1e-2, "training loss")
        print("loss mine \(loss.item(Float.self)) reference \(record["loss"]!.asArray(Float.self)[0])")
    }

    /// SMaLL-100 carries the target marker on the source and starts its decoder plain.
    func testSMaLL100MatchesTheReference() throws {
        try checkM2M100(weightsKey: "IK_VAL_SMALL100", recordKey: "IK_PARITY_SMALL100", variant: .small100, name: "small100")
    }

    private func checkM2M100(weightsKey: String, recordKey: String, variant: NFKMLXM2M100Variant, name: String) throws {
        try requireMLXRuntime()
        let (directory, record) = try release(weightsKey, recordKey)
        let translator = try NFKMLXM2M100.translator(directoryURL: directory, variant: variant)
        XCTAssertEqual(translator.languageIds["en"], 128022)
        for (index, sentence) in Self.sentences.enumerated() {
            XCTAssertEqual(translator.sourceIds(for: sentence, source: "en", target: "de"), ints(record["tokens_\(index)"]!), "tokens of \(sentence)")
        }
        var decoding = translator.defaultDecoding
        decoding.maxTokens = 64
        if variant != .small100 {
            decoding.forcedFirstToken = translator.languageIds["de"]
        }
        checkSeams(translator.net, encode: { translator.net.encode($0) }, decode: { translator.net.decode($0, memory: $1, cache: translator.net.makeCache()) },
                   record: record, decoding: decoding, sourceIds: translator.sourceIds(for: Self.sentences[1], source: "en", target: "de"), name: name)
        let text = try translator.translate(Self.sentences[1], from: "en", to: "de", decoding: decoding)
        print("\(name):", text)
        XCTAssertFalse(text.isEmpty)
        let marker: [Int] = variant == .small100 ? [] : [translator.languageIds["de"]!]
        XCTAssertEqual(marker + translator.tokenizer.encode(Self.target, dummyPrefix: nil) + [2],
                       ints(record["target_ids"]!), "target tokenization")
        let loss = NFKMLXTranslationObjective()(translator.net, record["source_ids"]!, record["target_ids"]!)
        XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: 1e-2, "training loss")
        print("loss mine \(loss.item(Float.self)) reference \(record["loss"]!.asArray(Float.self)[0])")
    }

    func testMADLADMatchesTheReference() throws {
        try checkMADLAD(weightsKey: "IK_VAL_MADLAD", recordKey: "IK_PARITY_MADLAD", half: false, name: "madlad")
    }

    /// The 7B is 33 GB of float32, so the reference records at bfloat16 (`MADLAD_DTYPE=bfloat16`) and
    /// the port loads `half`: two bfloat16 stacks, held to the tolerances the TranslateGemma 12B set.
    func testMADLAD7BMatchesTheReference() throws {
        try checkMADLAD(weightsKey: "IK_VAL_MADLAD_7B", recordKey: "IK_PARITY_MADLAD_7B", half: true, name: "madlad-7b")
    }

    private func checkMADLAD(weightsKey: String, recordKey: String, half: Bool, name: String) throws {
        try requireMLXRuntime()
        let (directory, record) = try release(weightsKey, recordKey)
        let translator = try NFKMLXMADLAD.translator(directoryURL: directory, half: half)
        for (index, sentence) in Self.sentences.enumerated() {
            XCTAssertEqual(translator.sourceIds(for: sentence, target: "de"), ints(record["tokens_\(index)"]!), "tokens of \(sentence)")
        }
        var decoding = translator.defaultDecoding
        decoding.maxTokens = 64
        checkSeams(translator.net, encode: { translator.net.encode($0) }, decode: { translator.net.decode($0, memory: $1, cache: translator.net.makeCache()) },
                   record: record, decoding: decoding, sourceIds: translator.sourceIds(for: Self.sentences[1], target: "de")!, name: name,
                   tolerance: half ? 0.995 : 0.999)
        let text = try translator.translate(Self.sentences[1], from: nil, to: "de", decoding: decoding)
        print("\(name):", text)
        XCTAssertFalse(text.isEmpty)
        XCTAssertEqual(translator.segmenter.encode(Self.target) + [2], ints(record["target_ids"]!), "target tokenization")
        let loss = NFKMLXTranslationObjective()(translator.net, record["source_ids"]!, record["target_ids"]!)
        XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: half ? 0.1 : 1e-2, "training loss")
        print("loss mine \(loss.item(Float.self)) reference \(record["loss"]!.asArray(Float.self)[0])")
    }

    // MARK: TranslateGemma

    private static let translateGemmaPrompt = "<start_of_turn>user\nYou are a professional English (en) to German (de) translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to German grammar, vocabulary, and cultural sensitivities.\nProduce only the German translation, without any additional explanations or commentary. Please translate the following English text into German:\n\n\nThe quick brown fox jumps over the lazy dog.<end_of_turn>\n<start_of_turn>model\n"

    private func tinyGemmaTokenizer() throws -> NFKMLXGemmaTokenizer {
        var vocabulary: [String: Int] = ["<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3]
        for (index, piece) in ["\u{2581}", "H", "i", "\n", "u", "s", "e", "r", "m", "o", "d", "l", "\n\n"].enumerated() {
            vocabulary[piece] = 10 + index
        }
        let added: [[String: Any]] = [
            ["id": 2, "content": "<bos>", "special": true],
            ["id": 105, "content": "<start_of_turn>", "special": true],
            ["id": 106, "content": "<end_of_turn>", "special": true],
        ]
        let json: [String: Any] = ["model": ["vocab": vocabulary, "merges": [["\n", "\n"]], "unk_token": "<unk>"],
                                   "added_tokens": added]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("translategemma-tok-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url))
    }

    private func tinyTranslateGemma() throws -> NFKMLXTranslateGemmaTranslator {
        try requireMLXRuntime()
        let model = NFKMLXGemma3Model(decoder: NFKMLXGemma3Net(.tiny), vision: nil, projector: nil,
                                      tokenizer: try tinyGemmaTokenizer(), tokens: NFKMLXGemma3Tokens(), chatTemplate: nil)
        return NFKMLXTranslateGemmaTranslator(model: model, languages: ["en": "English", "de": "German", "de-DE": "German",
                                                                         "zh-Hant": "Chinese", "pt-BR": "Portuguese"],
                                              identifier: "translategemma")
    }

    func testTranslateGemmaRendersTheReleaseTemplate() throws {
        let translator = try tinyTranslateGemma()
        XCTAssertEqual(translator.prompt(text: "  The quick brown fox jumps over the lazy dog. ", sourceCode: "en", targetCode: "de"),
                       Self.translateGemmaPrompt)
    }

    func testTranslateGemmaResolvesLanguageTags() throws {
        let translator = try tinyTranslateGemma()
        XCTAssertEqual(translator.code(for: "de_de"), "de-DE")
        XCTAssertEqual(translator.code(for: "de-AT"), "de", "an unnamed region falls back to the language")
        XCTAssertEqual(translator.code(for: "ZH-hant"), "zh-Hant")
        XCTAssertEqual(translator.code(for: "pt-br"), "pt-BR")
        XCTAssertNil(translator.code(for: "xx"))
        XCTAssertTrue(translator.supports(language: "en-GB"))
    }

    func testTranslateGemmaReadsTheLanguageTableFromTheTemplate() throws {
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        let template = "{%- set languages = {\n    \"de\": \"German\",\n    \"de-DE\": \"German\",\n    \"en\": \"English\",\n}\n-%}\n{{ bos_token }}\n{%- if messages[0][\"role\"] != \"user\" -%}{{ raise_exception(\"x\") }}{%- endif -%}"
        try template.write(to: scratch.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
        let table = NFKMLXTranslateGemmaTranslator.languageTable(inDirectory: scratch)
        XCTAssertEqual(table, ["de": "German", "de-DE": "German", "en": "English"])
    }

    func testTranslateGemmaObjectiveScoresTheModelTurnOnly() throws {
        try requireMLXRuntime()
        // Confident logits at the target positions score near zero regardless of the prompt positions.
        let vocabulary = 8
        var values = [Float](repeating: 0, count: 5 * vocabulary)
        let target: [Int32] = [3, 5]
        values[2 * vocabulary + 3] = 30
        values[3 * vocabulary + 5] = 30
        let loss = NFKMLXTranslateGemmaObjective().loss(logits: MLXArray(values, [1, 5, vocabulary]), promptLength: 3,
                                                        target: MLXArray(target))
        XCTAssertLessThan(loss.item(Float.self), 1e-4)
    }

    func testTranslateGemmaFineTuningAdaptsTheAttentionAndReloads() throws {
        try requireMLXRuntime()
        let net = NFKMLXGemma3Net(.tiny)
        let prompt = MLXArray([Int32(2), 105, 7, 8, 9, 106, 105])
        let target = MLXArray([Int32(11), 12, 106])
        let before = NFKMLXTranslateGemmaObjective()(net, prompt, target).item(Float.self)
        let losses = try NFKMLXTranslateGemma.fineTune(net, examples: { _ in (prompt, target) }, rank: 2, steps: 10)
        XCTAssertLessThan(losses.last!, before)
        let adapted = net.leafModules().flattened().filter { $0.1 is NFKMLXLoRALinear }.map(\.0)
        XCTAssertFalse(adapted.isEmpty)
        XCTAssertTrue(adapted.allSatisfy { $0.hasPrefix("layers.") && ($0.hasSuffix(".q_proj") || $0.hasSuffix(".v_proj")) })
        try NFKMLXLoRA.merge(into: net)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        try NFKMLXWeights.save(net, to: url)
        let reloaded = NFKMLXGemma3Net(.tiny)
        try NFKMLXWeights.apply(Array(try NFKMLXWeights.loadCheckpoint(url: url).arrays), to: reloaded)
        let full = concatenated([prompt, target]).reshaped([1, 10])
        XCTAssertLessThan(abs(net(full) - reloaded(full)).max().item(Float.self), 1e-5)
    }

    func testTranslateGemmaMatchesTheReference() throws {
        try checkTranslateGemma(weightsKey: "IK_VAL_TRANSLATEGEMMA", recordKey: "IK_PARITY_TRANSLATEGEMMA",
                                precision: .float32, name: "translategemma")
    }

    /// The 12B is 24 GB of bfloat16: the reference runs at that precision (`TRANSLATEGEMMA_DTYPE=bfloat16`)
    /// and the port loads at `.checkpoint`, so the comparison is bfloat16 against bfloat16. The release is
    /// past the working set of a 32 GB machine, so the decoder streams the layers it cannot hold.
    func testTranslateGemma12BMatchesTheReference() throws {
        try checkTranslateGemma(weightsKey: "IK_VAL_TRANSLATEGEMMA_12B", recordKey: "IK_PARITY_TRANSLATEGEMMA_12B",
                                precision: .checkpoint, residency: .streamed, name: "translategemma-12b")
    }

    /// The streamed 12B with the 4B held beside it as its draft: the greedy continuation is the record's,
    /// in fewer passes of the 12B than tokens.
    func testTranslateGemma12BDraftedByThe4BMatchesTheReference() throws {
        try checkTranslateGemmaDrafted(weightsKey: "IK_VAL_TRANSLATEGEMMA_12B", recordKey: "IK_PARITY_TRANSLATEGEMMA_12B",
                                       name: "translategemma-12b")
    }

    /// The streamed 27B with the 4B as its draft, which holds no 27B layer beside the 4B on a 32 GB machine.
    func testTranslateGemma27BDraftedByThe4BMatchesTheReference() throws {
        try checkTranslateGemmaDrafted(weightsKey: "IK_VAL_TRANSLATEGEMMA_27B", recordKey: "IK_PARITY_TRANSLATEGEMMA_27B",
                                       name: "translategemma-27b")
    }

    /// MLX's own memory beside the machine's working set, for a streamed run's record.
    private static var memoryReading: String {
        String(format: "; MLX peak %.1f GB, active %.1f GB, cache %.1f GB, working set %.1f GB",
               Double(NFKMLXGPU.peakMemory) / 1e9, Double(NFKMLXGPU.activeMemory) / 1e9,
               Double(NFKMLXGPU.cacheMemory) / 1e9, Double(NFKMLXGPU.recommendedWorkingSetSize) / 1e9)
    }

    private func checkTranslateGemmaDrafted(weightsKey: String, recordKey: String, name: String) throws {
        try requireMLXRuntime()
        let (directory, record) = try release(weightsKey, recordKey)
        guard let draftPath = NFKMLXValidationConfig.environment["IK_VAL_TRANSLATEGEMMA"],
              FileManager.default.fileExists(atPath: draftPath) else {
            throw XCTSkip("set IK_VAL_TRANSLATEGEMMA to the translategemma-4b-it release directory, the draft")
        }
        let translator = try NFKMLXTranslateGemma.translator(directoryURL: directory, draftDirectoryURL: URL(fileURLWithPath: draftPath),
                                                             precision: .checkpoint, residency: .streamed)
        let stream = try XCTUnwrap(translator.model.decoder.layerStream)
        let ids = translator.promptTokens(text: Self.sentences[1], sourceCode: "en", targetCode: "de")
        XCTAssertEqual(ids, ints(record["tokens"]!), "the rendered template's ids")
        let started = Date()
        let produced = try translator.generate(promptTokens: ids, maxTokens: 48)
        let seconds = Date().timeIntervalSince(started)
        XCTAssertEqual(produced, ints(record["continuation"]!).filter { $0 != 1 && $0 != 106 }, "greedy continuation")
        let report = translator.model.lastSpeculativeReport
        print("\(name) drafted by the 4B: \(produced.count) tokens in \(report.rounds + 1) passes, "
              + "\(report.accepted) of \(report.proposed) proposals kept, \(stream.layers.count) of "
              + "\(translator.model.decoder.configuration.layerCount) layers streamed, " + String(format: "%.1f s", seconds)
              + Self.memoryReading)
        XCTAssertLessThan(report.rounds + 1, produced.count, "fewer passes of the streamed model than tokens")
    }

    /// The 27B is 55 GB of bfloat16, so neither side holds it whole: the reference reads each decoder layer
    /// as its forward reaches it (`run_reference.py translategemma_layerwise`), and the port streams the
    /// layers it cannot hold.
    func testTranslateGemma27BMatchesTheReference() throws {
        try checkTranslateGemma(weightsKey: "IK_VAL_TRANSLATEGEMMA_27B", recordKey: "IK_PARITY_TRANSLATEGEMMA_27B",
                                precision: .checkpoint, residency: .streamed, name: "translategemma-27b")
    }

    private func checkTranslateGemma(weightsKey: String, recordKey: String, precision: NFKMLXWeightPrecision,
                                     residency: NFKMLXResidency = .automatic, name: String) throws {
        try requireMLXRuntime()
        let (directory, record) = try release(weightsKey, recordKey)
        let translator = try NFKMLXTranslateGemma.translator(directoryURL: directory, precision: precision,
                                                             residency: residency)
        let stream = translator.model.decoder.layerStream
        if residency == .streamed {
            XCTAssertNotNil(stream, "a release past the working set streams")
        }
        XCTAssertGreaterThan(translator.languages.count, 500)
        let ids = translator.promptTokens(text: Self.sentences[1], sourceCode: "en", targetCode: "de")
        XCTAssertEqual(ids, ints(record["tokens"]!), "the rendered template's ids")
        let logits = translator.model.logits(tokens: ids, softTokens: nil)
        let tail = logits[0, (ids.count - 16)..., 0...]
        let logitCosine = cosine(tail, record["output"]!)
        // A float32 load reproduces the reference to float error; a .checkpoint load compares bfloat16
        // against a bfloat16 reference, where the two accumulation orders drift apart over the stack.
        XCTAssertGreaterThan(logitCosine, precision == .float32 ? 0.999 : 0.995)
        if record["hidden_last.0"] != nil {
            let states = translator.model.decoder.layerStates(MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count]))
            var worst: (layer: Int, cosine: Float) = (-1, 1)
            for (index, state) in states.enumerated() {
                guard let reference = record["hidden_last.\(index)"] else { continue }
                let layerCosine = cosine(state[0, ids.count - 1], reference)
                if layerCosine < worst.cosine { worst = (index, layerCosine) }
            }
            print("\(name): worst layer state cosine \(worst.cosine) at layer \(worst.layer) of \(states.count - 1)")
            XCTAssertGreaterThan(worst.cosine, precision == .float32 ? 0.999 : 0.99, "a layer diverges beyond precision drift")
        }
        let argmax = tail.argMax(axis: -1).asArray(Int32.self)
        let agreeing = zip(argmax, record["output"]!.argMax(axis: -1).asArray(Int32.self)).filter { $0 == $1 }.count
        let produced = try translator.generate(promptTokens: ids, maxTokens: 48)
        let reference = ints(record["continuation"]!).filter { $0 != 1 && $0 != 106 }
        XCTAssertEqual(produced, reference, "greedy continuation")
        // A streamed decoder reads every layer it does not hold on each step, so the translation is
        // decoded from the continuation above rather than generated a second time.
        let text = stream == nil
            ? try translator.translate(Self.sentences[1], from: "en", to: "de", decoding: translator.defaultDecoding)
            : translator.model.decode(produced).trimmingCharacters(in: .whitespacesAndNewlines)
        print("\(name):", text, "| logit cosine \(logitCosine); argmax \(agreeing)/16")
        XCTAssertFalse(text.isEmpty)
        let loss = NFKMLXTranslateGemmaObjective()(translator.model.decoder, record["tokens"]!, record["target_ids"]!)
        // Two bfloat16 stacks differ in accumulation order, and a 262k-way cross-entropy over a dozen
        // positions amplifies the logit drift; the 12B measures 0.43 against 0.48.
        XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0],
                       accuracy: precision == .float32 ? 1e-2 : 0.1, "fine-tuning loss")
        print("loss mine \(loss.item(Float.self)) reference \(record["loss"]!.asArray(Float.self)[0])")
        if let stream {
            let (bytes, seconds) = stream.readStatistics
            print("\(name): \(stream.layers.count) of \(translator.model.decoder.configuration.layerCount) layers streamed, "
                  + String(format: "%.1f GB read at %.2f GB/s", Double(bytes) / 1e9, Double(bytes) / 1e9 / max(seconds, 1e-9))
                  + Self.memoryReading)
        }
    }

    // MARK: Language probes

    /// One probe set per source language, the table `run_reference.py` records (`TRANSLATION_PROBES`);
    /// index 1 is the sentence the seams, the generations, and the loss use.
    private static let probes: [String: [String]] = {
        var table: [String: [String]] = [
            "en": sentences,
            "ja": [
                "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}\u{3001}\u{4e16}\u{754c}\u{ff01}\u{304a}\u{5143}\u{6c17}\u{3067}\u{3059}\u{304b}\u{ff1f}",  // こんにちは、世界！お元気ですか？
                "\u{7d20}\u{65e9}\u{3044}\u{8336}\u{8272}\u{306e}\u{72d0}\u{304c}\u{6020}\u{3051}\u{8005}\u{306e}\u{72ac}\u{3092}\u{98db}\u{3073}\u{8d8a}\u{3048}\u{308b}\u{3002}",  // 素早い茶色の狐が怠け者の犬を飛び越える。
                "\u{ff21}\u{ff22}\u{ff23}\u{ff11}\u{ff12}\u{ff13}\u{3000}\u{5168}\u{89d2}\u{3068}\u{534a}\u{89d2}\u{ff76}\u{ff9e}\u{ff77}\u{ff9e}",  // ＡＢＣ１２３<ideographic space>全角と半角ｶﾞｷﾞ
                "\u{6771}\u{4eac}\u{ff08}\u{3068}\u{3046}\u{304d}\u{3087}\u{3046}\u{ff09}\u{306f}\u{65e5}\u{672c}\u{306e}\u{9996}\u{90fd}\u{3067}\u{3059}\u{3002}\u{3231}\u{30c6}\u{30b9}\u{30c8}",  // 東京（とうきょう）は日本の首都です。㈱テスト
                "\u{4eca}\u{65e5}\u{306f}\u{6674}\u{308c}\u{3067}\u{3059}\u{1f600} \u{2460}\u{2461}\u{2462}",  // 今日は晴れです😀 ①②③
            ],
            "zh": [
                "\u{4f60}\u{597d}\u{ff0c}\u{4e16}\u{754c}\u{ff01}\u{4f60}\u{597d}\u{5417}\u{ff1f}",  // 你好，世界！你好吗？
                "\u{654f}\u{6377}\u{7684}\u{68d5}\u{8272}\u{72d0}\u{72f8}\u{8df3}\u{8fc7}\u{4e86}\u{61d2}\u{72d7}\u{3002}",  // 敏捷的棕色狐狸跳过了懒狗。
                "\u{4eca}\u{5929}\u{5929}\u{6c14}\u{5f88}\u{597d}\u{ff0c}\u{6211}\u{4eec}\u{53bb}\u{516c}\u{56ed}\u{5427}\u{ff01}",  // 今天天气很好，我们去公园吧！
                "\u{9577}\u{6c5f}\u{662f}\u{4e2d}\u{570b}\u{6700}\u{9577}\u{7684}\u{6cb3}\u{6d41}\u{3002}",  // 長江是中國最長的河流。
                "\u{4ef7}\u{683c}\u{662f}\u{ff11}\u{ff12}\u{ff13}\u{5143}\u{1f600} \u{3299}",  // 价格是１２３元😀 ㊙
            ],
            "ar": [
                "\u{645}\u{631}\u{62d}\u{628}\u{627} \u{628}\u{627}\u{644}\u{639}\u{627}\u{644}\u{645}! \u{643}\u{64a}\u{641} \u{62d}\u{627}\u{644}\u{643}\u{61f}",  // مرحبا بالعالم! كيف حالك؟
                "\u{627}\u{644}\u{62b}\u{639}\u{644}\u{628} \u{627}\u{644}\u{628}\u{646}\u{64a} \u{627}\u{644}\u{633}\u{631}\u{64a}\u{639} \u{64a}\u{642}\u{641}\u{632} \u{641}\u{648}\u{642} \u{627}\u{644}\u{643}\u{644}\u{628} \u{627}\u{644}\u{643}\u{633}\u{648}\u{644}.",  // الثعلب البني السريع يقفز فوق الكلب الكسول.
                "\u{671}\u{644}\u{633}\u{64e}\u{651}\u{644}\u{64e}\u{627}\u{645}\u{64f} \u{639}\u{64e}\u{644}\u{64e}\u{64a}\u{652}\u{643}\u{64f}\u{645}\u{652} \u{648}\u{64e}\u{631}\u{64e}\u{62d}\u{652}\u{645}\u{64e}\u{629}\u{64f} \u{671}\u{644}\u{644}\u{64e}\u{651}\u{670}\u{647}\u{650}",  // ٱلسَّلَامُ عَلَيْكُمْ وَرَحْمَةُ ٱللَّٰهِ
                "\u{fefb} \u{628}\u{623}\u{633}\u{60c} \u{634}\u{643}\u{631}\u{627}\u{64b} \u{fedf}\u{fee0}",  // <U+FEFB> بأس، شكراً <U+FEDF><U+FEE0>
                "\u{627}\u{644}\u{642}\u{627}\u{647}\u{631}\u{629} \u{647}\u{64a} \u{639}\u{627}\u{635}\u{645}\u{629} \u{645}\u{635}\u{631} \u{1f600} \u{661}\u{662}\u{663}",  // القاهرة هي عاصمة مصر 😀 ١٢٣
            ],
            "vi": [
                "Xin ch\u{e0}o th\u{1ebf} gi\u{1edb}i! B\u{1ea1}n c\u{f3} kh\u{1ecf}e kh\u{f4}ng?",  // Xin chào thế giới! Bạn có khỏe không?
                "Con c\u{e1}o n\u{e2}u nhanh nh\u{1eb9}n nh\u{1ea3}y qua con ch\u{f3} l\u{1b0}\u{1edd}i.",  // Con cáo nâu nhanh nhẹn nhảy qua con chó lười.
                "",
                "T\u{f4}i y\u{ea}u Vi\u{1ec7}t Nam v\u{e0} ti\u{1ebf}ng Vi\u{1ec7}t.",  // Tôi yêu Việt Nam và tiếng Việt.
                "Gi\u{e1} l\u{e0} 123.456 \u{111}\u{1ed3}ng \u{1f600}",  // Giá là 123.456 đồng 😀
            ],
            "th": [
                "\u{e2a}\u{e27}\u{e31}\u{e2a}\u{e14}\u{e35}\u{e04}\u{e23}\u{e31}\u{e1a} \u{e1c}\u{e21}\u{e0a}\u{e37}\u{e48}\u{e2d}\u{e2a}\u{e21}\u{e0a}\u{e32}\u{e22}",  // สวัสดีครับ ผมชื่อสมชาย
                "\u{e2a}\u{e38}\u{e19}\u{e31}\u{e02}\u{e08}\u{e34}\u{e49}\u{e07}\u{e08}\u{e2d}\u{e01}\u{e2a}\u{e35}\u{e19}\u{e49}\u{e33}\u{e15}\u{e32}\u{e25}\u{e01}\u{e23}\u{e30}\u{e42}\u{e14}\u{e14}\u{e02}\u{e49}\u{e32}\u{e21}\u{e2a}\u{e38}\u{e19}\u{e31}\u{e02}\u{e02}\u{e35}\u{e49}\u{e40}\u{e01}\u{e35}\u{e22}\u{e08}",  // สุนัขจิ้งจอกสีน้ำตาลกระโดดข้ามสุนัขขี้เกียจ
                "\u{e1b}\u{e23}\u{e30}\u{e40}\u{e17}\u{e28}\u{e44}\u{e17}\u{e22}\u{e21}\u{e35}\u{e1b}\u{e23}\u{e30}\u{e0a}\u{e32}\u{e01}\u{e23}\u{e1b}\u{e23}\u{e30}\u{e21}\u{e32}\u{e13}\u{e40}\u{e08}\u{e47}\u{e14}\u{e2a}\u{e34}\u{e1a}\u{e25}\u{e49}\u{e32}\u{e19}\u{e04}\u{e19}",  // ประเทศไทยมีประชากรประมาณเจ็ดสิบล้านคน
                "\u{e27}\u{e31}\u{e19}\u{e19}\u{e35}\u{e49}\u{e2d}\u{e32}\u{e01}\u{e32}\u{e28}\u{e14}\u{e35}\u{e21}\u{e32}\u{e01} \u{e40}\u{e23}\u{e32}\u{e44}\u{e1b}\u{e2a}\u{e27}\u{e19}\u{e2a}\u{e32}\u{e18}\u{e32}\u{e23}\u{e13}\u{e30}\u{e01}\u{e31}\u{e19}\u{e40}\u{e16}\u{e2d}\u{e30}",  // วันนี้อากาศดีมาก เราไปสวนสาธารณะกันเถอะ
                "\u{e23}\u{e32}\u{e04}\u{e32} \u{e51}\u{e52}\u{e53} \u{e1a}\u{e32}\u{e17} \u{1f600}",  // ราคา ๑๒๓ บาท 😀
            ],
            "hi": [
                "\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947} \u{926}\u{941}\u{928}\u{93f}\u{92f}\u{93e}! \u{906}\u{92a} \u{915}\u{948}\u{938}\u{947} \u{939}\u{948}\u{902}?",  // नमस्ते दुनिया! आप कैसे हैं?
                "\u{924}\u{947}\u{91c}\u{93c} \u{92d}\u{942}\u{930}\u{940} \u{932}\u{94b}\u{92e}\u{921}\u{93c}\u{940} \u{906}\u{932}\u{938}\u{940} \u{915}\u{941}\u{924}\u{94d}\u{924}\u{947} \u{915}\u{947} \u{90a}\u{92a}\u{930} \u{938}\u{947} \u{915}\u{942}\u{926} \u{91c}\u{93e}\u{924}\u{940} \u{939}\u{948}\u{964}",  // तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूद जाती है।
                "\u{92d}\u{93e}\u{930}\u{924} \u{90f}\u{915} \u{935}\u{93f}\u{936}\u{93e}\u{932} \u{926}\u{947}\u{936} \u{939}\u{948}\u{964}",  // भारत एक विशाल देश है।
                "\u{92e}\u{948}\u{902} \u{939}\u{93f}\u{928}\u{94d}\u{926}\u{940} \u{938}\u{940}\u{916} \u{930}\u{939}\u{93e} \u{939}\u{942}\u{901}\u{964} \u{915}\u{94d}\u{200d}\u{937}",  // मैं हिन्दी सीख रहा हूँ। क्<ZWJ>ष
                "\u{915}\u{940}\u{92e}\u{924} \u{967}\u{968}\u{969} \u{930}\u{941}\u{92a}\u{92f}\u{947} \u{939}\u{948} \u{1f600}",  // कीमत १२३ रुपये है 😀
            ],
            "ko": [
                "\u{c548}\u{b155}\u{d558}\u{c138}\u{c694}, \u{c138}\u{acc4}! \u{c798} \u{c9c0}\u{b0b4}\u{c138}\u{c694}?",  // 안녕하세요, 세계! 잘 지내세요?
                "\u{be60}\u{b978} \u{ac08}\u{c0c9} \u{c5ec}\u{c6b0}\u{ac00} \u{ac8c}\u{c73c}\u{b978} \u{ac1c}\u{b97c} \u{b6f0}\u{c5b4}\u{b118}\u{c2b5}\u{b2c8}\u{b2e4}.",  // 빠른 갈색 여우가 게으른 개를 뛰어넘습니다.
                "\u{d55c}\u{ad6d}\u{c758} \u{c218}\u{b3c4}\u{b294} \u{c11c}\u{c6b8}\u{c785}\u{b2c8}\u{b2e4}.",  // 한국의 수도는 서울입니다.
                "",
                "\u{ac00}\u{aca9}\u{c740} 123\u{c6d0}\u{c785}\u{b2c8}\u{b2e4} \u{1f600} \u{3131}\u{3134}\u{3137}",  // 가격은 123원입니다 😀 ㄱㄴㄷ
            ],
        ]
        table["vi"]![2] = table["vi"]![1].decomposedStringWithCanonicalMapping
        table["ko"]![3] = table["ko"]![2].decomposedStringWithCanonicalMapping
        return table
    }()

    private static let targets: [String: String] = [
        "de": target,
        "en": "The quick brown fox jumps over the lazy dog.",  // The quick brown fox jumps over the lazy dog.
        "zh": "\u{654f}\u{6377}\u{7684}\u{68d5}\u{8272}\u{72d0}\u{72f8}\u{8df3}\u{8fc7}\u{4e86}\u{61d2}\u{72d7}\u{3002}",  // 敏捷的棕色狐狸跳过了懒狗。
        "zh-Hant": "\u{654f}\u{6377}\u{7684}\u{68d5}\u{8272}\u{72d0}\u{72f8}\u{8df3}\u{904e}\u{4e86}\u{61f6}\u{72d7}\u{3002}",  // 敏捷的棕色狐狸跳過了懶狗。
        "ja": "\u{7d20}\u{65e9}\u{3044}\u{8336}\u{8272}\u{306e}\u{72d0}\u{304c}\u{6020}\u{3051}\u{8005}\u{306e}\u{72ac}\u{3092}\u{98db}\u{3073}\u{8d8a}\u{3048}\u{308b}\u{3002}",  // 素早い茶色の狐が怠け者の犬を飛び越える。
        "ar": "\u{627}\u{644}\u{62b}\u{639}\u{644}\u{628} \u{627}\u{644}\u{628}\u{646}\u{64a} \u{627}\u{644}\u{633}\u{631}\u{64a}\u{639} \u{64a}\u{642}\u{641}\u{632} \u{641}\u{648}\u{642} \u{627}\u{644}\u{643}\u{644}\u{628} \u{627}\u{644}\u{643}\u{633}\u{648}\u{644}.",  // الثعلب البني السريع يقفز فوق الكلب الكسول.
        "vi": "Con c\u{e1}o n\u{e2}u nhanh nh\u{1eb9}n nh\u{1ea3}y qua con ch\u{f3} l\u{1b0}\u{1edd}i.",  // Con cáo nâu nhanh nhẹn nhảy qua con chó lười.
        "th": "\u{e2a}\u{e38}\u{e19}\u{e31}\u{e02}\u{e08}\u{e34}\u{e49}\u{e07}\u{e08}\u{e2d}\u{e01}\u{e2a}\u{e35}\u{e19}\u{e49}\u{e33}\u{e15}\u{e32}\u{e25}\u{e01}\u{e23}\u{e30}\u{e42}\u{e14}\u{e14}\u{e02}\u{e49}\u{e32}\u{e21}\u{e2a}\u{e38}\u{e19}\u{e31}\u{e02}\u{e02}\u{e35}\u{e49}\u{e40}\u{e01}\u{e35}\u{e22}\u{e08}",  // สุนัขจิ้งจอกสีน้ำตาลกระโดดข้ามสุนัขขี้เกียจ
        "hi": "\u{924}\u{947}\u{91c}\u{93c} \u{92d}\u{942}\u{930}\u{940} \u{932}\u{94b}\u{92e}\u{921}\u{93c}\u{940} \u{906}\u{932}\u{938}\u{940} \u{915}\u{941}\u{924}\u{94d}\u{924}\u{947} \u{915}\u{947} \u{90a}\u{92a}\u{930} \u{938}\u{947} \u{915}\u{942}\u{926} \u{91c}\u{93e}\u{924}\u{940} \u{939}\u{948}\u{964}",  // तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूद जाती है।
        "ko": "\u{be60}\u{b978} \u{ac08}\u{c0c9} \u{c5ec}\u{c6b0}\u{ac00} \u{ac8c}\u{c73c}\u{b978} \u{ac1c}\u{b97c} \u{b6f0}\u{c5b4}\u{b118}\u{c2b5}\u{b2c8}\u{b2e4}.",  // 빠른 갈색 여우가 게으른 개를 뛰어넘습니다.
    ]

    private struct ProbePair {
        let weightsKey: String
        let recordKey: String
        let source: String
        let target: String
    }

    private static let marianProbePairs = [
        ProbePair(weightsKey: "IK_VAL_MARIAN_JA_EN", recordKey: "IK_PARITY_MARIAN_JA_EN", source: "ja", target: "en"),
        ProbePair(weightsKey: "IK_VAL_MARIAN_ZH_EN", recordKey: "IK_PARITY_MARIAN_ZH_EN", source: "zh", target: "en"),
        ProbePair(weightsKey: "IK_VAL_MARIAN_EN_ZH", recordKey: "IK_PARITY_MARIAN_EN_ZH", source: "en", target: "zh"),
        ProbePair(weightsKey: "IK_VAL_MARIAN_EN_ZH", recordKey: "IK_PARITY_MARIAN_EN_ZH_HANT", source: "en", target: "zh-Hant"),
        ProbePair(weightsKey: "IK_VAL_MARIAN_AR_EN", recordKey: "IK_PARITY_MARIAN_AR_EN", source: "ar", target: "en"),
        ProbePair(weightsKey: "IK_VAL_MARIAN_VI_EN", recordKey: "IK_PARITY_MARIAN_VI_EN", source: "vi", target: "en"),
        ProbePair(weightsKey: "IK_VAL_MARIAN_TH_EN", recordKey: "IK_PARITY_MARIAN_TH_EN", source: "th", target: "en"),
        ProbePair(weightsKey: "IK_VAL_MARIAN_HI_EN", recordKey: "IK_PARITY_MARIAN_HI_EN", source: "hi", target: "en"),
        ProbePair(weightsKey: "IK_VAL_MARIAN_KO_EN", recordKey: "IK_PARITY_MARIAN_KO_EN", source: "ko", target: "en"),
    ]

    /// The pairs a multilingual release is probed on: every probe language into English, and English
    /// into both Chinese scripts and Japanese.
    private static func multilingualProbePairs(_ family: String, weightsKey: String, hant: Bool = true) -> [ProbePair] {
        var pairs = ["ja", "zh", "ar", "vi", "th", "hi", "ko"].map {
            ProbePair(weightsKey: weightsKey, recordKey: "IK_PARITY_\(family)_\($0.uppercased())_EN", source: $0, target: "en")
        }
        pairs.append(ProbePair(weightsKey: weightsKey, recordKey: "IK_PARITY_\(family)_EN_ZH", source: "en", target: "zh"))
        if hant {
            pairs.append(ProbePair(weightsKey: weightsKey, recordKey: "IK_PARITY_\(family)_EN_ZH_HANT", source: "en", target: "zh-Hant"))
        }
        pairs.append(ProbePair(weightsKey: weightsKey, recordKey: "IK_PARITY_\(family)_EN_JA", source: "en", target: "ja"))
        return pairs
    }

    private func utf8(_ array: MLXArray) -> String {
        String(decoding: array.asArray(Int32.self).map { UInt8(truncatingIfNeeded: $0) }, as: UTF8.self)
    }

    /// The oracle record for `key`, or nil when the key or its file is absent.
    private func record(_ key: String) throws -> [String: MLXArray]? {
        guard let path = NFKMLXValidationConfig.environment[key], FileManager.default.fileExists(atPath: path) else { return nil }
        return try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
    }

    /// Gemma's vocabulary keeps a nukta letter in both its precomposed and decomposed spellings.
    func testTheGemmaTokenizerKeepsCanonicallyEquivalentPiecesApart() throws {
        let precomposed = "\u{95c}", decomposed = "\u{921}\u{93c}"
        // Written as text: a Swift dictionary literal with both spellings traps on duplicate keys, the
        // same equivalence the tokenizer must not apply.
        let json = """
        {"model": {"type": "BPE", "unk_token": "<unk>", "merges": [["\u{921}", "\u{93c}"]],
                   "vocab": {"<unk>": 0, "\u{2581}": 1, "\u{921}": 2, "\u{93c}": 3, "\(precomposed)": 4, "\(decomposed)": 5}},
         "added_tokens": []}
        """
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gemma-tokenizer-\(UUID().uuidString).json")
        try Data(json.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let tokenizer = try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url))
        XCTAssertEqual(tokenizer.encode(precomposed), [4])
        XCTAssertEqual(tokenizer.encode(decomposed), [5], "the merge of the decomposed pair is its own piece")
        XCTAssertEqual(tokenizer.decode([5]), decomposed)
    }

    func testMarianResolvesChineseMarkersOfAGroupRelease() {
        XCTAssertEqual(Array(NFKMLXMarianTranslator.markerCandidates(for: "zh").prefix(4)), ["zho_Hans", "zho", "cmn_Hans", "cmn"])
        XCTAssertEqual(NFKMLXMarianTranslator.markerCandidates(for: "zh-TW").first, "zho_Hant")
        XCTAssertTrue(NFKMLXMarianTranslator.markerCandidates(for: "zh-TW").contains("cmn_Hant"))
        XCTAssertTrue(NFKMLXMarianTranslator.markerCandidates(for: "zh-Hant-HK").contains("cmn_Hant"))
        XCTAssertEqual(NFKMLXMarianTranslator.markerCandidates(for: "yue"), ["yue"])
        XCTAssertEqual(NFKMLXMarianTranslator.markerCandidates(for: "de-AT"), ["deu", "de"])
    }

    func testTheBackendImpliesTheChineseScriptFromTheRegion() {
        XCTAssertEqual(NFKMLXTranslationBackend.impliedScript("zh"), "Hans")
        XCTAssertEqual(NFKMLXTranslationBackend.impliedScript("zh-CN"), "Hans")
        XCTAssertEqual(NFKMLXTranslationBackend.impliedScript("zh-TW"), "Hant")
        XCTAssertEqual(NFKMLXTranslationBackend.impliedScript("zh-HK"), "Hant")
        XCTAssertEqual(NFKMLXTranslationBackend.impliedScript("zh-Hans-HK"), "Hans", "a script subtag wins over the region")
        XCTAssertEqual(NFKMLXTranslationBackend.impliedScript("sr-Latn"), "Latn")
        XCTAssertNil(NFKMLXTranslationBackend.impliedScript("ja"))
    }

    func testMarianProbePairsMatchTheReference() throws {
        try requireMLXRuntime()
        var ran = 0
        for pair in Self.marianProbePairs {
            guard let loaded = try? release(pair.weightsKey, pair.recordKey) else { continue }
            let (directory, record) = loaded
            ran += 1
            let translator = try NFKMLXMarian.translator(directoryURL: directory)
            let sentences = Self.probes[pair.source]!
            for (index, sentence) in sentences.enumerated() {
                XCTAssertEqual(translator.sourceIds(for: sentence, target: pair.target), ints(record["tokens_\(index)"]!),
                               "\(pair.recordKey) tokens of \(sentence)")
            }
            var decoding = translator.defaultDecoding
            decoding.maxTokens = 64
            checkSeams(translator.net, encode: { translator.net.encode($0) },
                       decode: { translator.net.decode($0, memory: $1, cache: translator.net.makeCache()) },
                       record: record, decoding: decoding, sourceIds: translator.sourceIds(for: sentences[1], target: pair.target),
                       name: pair.recordKey)
            var greedy = decoding
            greedy.beams = 1
            let greedyText = try translator.translate(sentences[1], from: pair.source, to: pair.target, decoding: greedy)
            XCTAssertEqual(greedyText, utf8(record["greedy_text"]!), "\(pair.recordKey) greedy text")
            let beamText = try translator.translate(sentences[1], from: pair.source, to: pair.target, decoding: decoding)
            XCTAssertEqual(beamText, utf8(record["beam_text"]!), "\(pair.recordKey) beam text")
            print("\(pair.recordKey):", greedyText)
            let targetIds = translator.targetTokenizer.encode(Self.targets[pair.target]!, dummyPrefix: nil) + [translator.net.configuration.eosTokenId]
            XCTAssertEqual(targetIds, ints(record["target_ids"]!), "\(pair.recordKey) target tokenization")
            let loss = NFKMLXTranslationObjective()(translator.net, record["source_ids"]!, record["target_ids"]!)
            XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: 1e-2, "\(pair.recordKey) training loss")
        }
        if ran == 0 { throw XCTSkip("no IK_VAL_MARIAN_<pair> release with its IK_PARITY_MARIAN_<pair> record is present") }
    }

    func testNLLBProbePairsMatchTheReference() throws {
        try requireMLXRuntime()
        let pairs = Self.multilingualProbePairs("NLLB", weightsKey: "IK_VAL_NLLB")
        let records = try pairs.compactMap { pair in try record(pair.recordKey).map { (pair, $0) } }
        guard let directory = NFKMLXValidationConfig.environment["IK_VAL_NLLB"], !records.isEmpty else {
            throw XCTSkip("set IK_VAL_NLLB and at least one IK_PARITY_NLLB_<pair> record")
        }
        let translator = try NFKMLXNLLB.translator(directoryURL: URL(fileURLWithPath: directory))
        for (pair, record) in records {
            let sentences = Self.probes[pair.source]!
            let targetCode = translator.code(for: pair.target)!
            for (index, sentence) in sentences.enumerated() {
                XCTAssertEqual(translator.sourceIds(for: sentence, source: pair.source), ints(record["tokens_\(index)"]!),
                               "\(pair.recordKey) tokens of \(sentence)")
            }
            var decoding = translator.defaultDecoding
            decoding.maxTokens = 64
            decoding.forcedFirstToken = translator.languageIds[targetCode]
            checkSeams(translator.net, encode: { translator.net.encode($0) },
                       decode: { translator.net.decode($0, memory: $1, cache: translator.net.makeCache()) },
                       record: record, decoding: decoding, sourceIds: translator.sourceIds(for: sentences[1], source: pair.source), name: pair.recordKey)
            var greedy = decoding
            greedy.beams = 1
            let greedyText = try translator.translate(sentences[1], from: pair.source, to: pair.target, decoding: greedy)
            XCTAssertEqual(greedyText, utf8(record["greedy_text"]!), "\(pair.recordKey) greedy text")
            var beam = decoding
            beam.beams = ints(record["beams"]!)[0]
            let beamText = try translator.translate(sentences[1], from: pair.source, to: pair.target, decoding: beam)
            XCTAssertEqual(beamText, utf8(record["beam_text"]!), "\(pair.recordKey) beam text")
            print("\(pair.recordKey):", greedyText)
            XCTAssertEqual([translator.languageIds[targetCode]!] + translator.tokenizer.encode(Self.targets[pair.target]!, dummyPrefix: nil) + [2],
                           ints(record["target_ids"]!), "\(pair.recordKey) target tokenization")
            let loss = NFKMLXTranslationObjective()(translator.net, record["source_ids"]!, record["target_ids"]!)
            XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: 1e-2, "\(pair.recordKey) training loss")
        }
    }

    func testM2M100ProbePairsMatchTheReference() throws {
        try requireMLXRuntime()
        let pairs = Self.multilingualProbePairs("M2M100", weightsKey: "IK_VAL_M2M100", hant: false)
        let records = try pairs.compactMap { pair in try record(pair.recordKey).map { (pair, $0) } }
        guard let directory = NFKMLXValidationConfig.environment["IK_VAL_M2M100"], !records.isEmpty else {
            throw XCTSkip("set IK_VAL_M2M100 and at least one IK_PARITY_M2M100_<pair> record")
        }
        let translator = try NFKMLXM2M100.translator(directoryURL: URL(fileURLWithPath: directory))
        for (pair, record) in records {
            let sentences = Self.probes[pair.source]!
            let sourceCode = translator.code(for: pair.source)!
            let targetCode = translator.code(for: pair.target)!
            for (index, sentence) in sentences.enumerated() {
                XCTAssertEqual(translator.sourceIds(for: sentence, source: sourceCode, target: targetCode), ints(record["tokens_\(index)"]!),
                               "\(pair.recordKey) tokens of \(sentence)")
            }
            var decoding = translator.defaultDecoding
            decoding.maxTokens = 64
            decoding.forcedFirstToken = translator.languageIds[targetCode]
            checkSeams(translator.net, encode: { translator.net.encode($0) },
                       decode: { translator.net.decode($0, memory: $1, cache: translator.net.makeCache()) },
                       record: record, decoding: decoding,
                       sourceIds: translator.sourceIds(for: sentences[1], source: sourceCode, target: targetCode), name: pair.recordKey)
            var greedy = decoding
            greedy.beams = 1
            let greedyText = try translator.translate(sentences[1], from: pair.source, to: pair.target, decoding: greedy)
            XCTAssertEqual(greedyText, utf8(record["greedy_text"]!), "\(pair.recordKey) greedy text")
            let beamText = try translator.translate(sentences[1], from: pair.source, to: pair.target, decoding: decoding)
            XCTAssertEqual(beamText, utf8(record["beam_text"]!), "\(pair.recordKey) beam text")
            print("\(pair.recordKey):", greedyText)
            XCTAssertEqual([translator.languageIds[targetCode]!] + translator.tokenizer.encode(Self.targets[pair.target]!, dummyPrefix: nil) + [2],
                           ints(record["target_ids"]!), "\(pair.recordKey) target tokenization")
            let loss = NFKMLXTranslationObjective()(translator.net, record["source_ids"]!, record["target_ids"]!)
            XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: 1e-2, "\(pair.recordKey) training loss")
        }
    }

    func testMADLADProbePairsMatchTheReference() throws {
        try requireMLXRuntime()
        let pairs = Self.multilingualProbePairs("MADLAD", weightsKey: "IK_VAL_MADLAD")
        let records = try pairs.compactMap { pair in try record(pair.recordKey).map { (pair, $0) } }
        guard let directory = NFKMLXValidationConfig.environment["IK_VAL_MADLAD"], !records.isEmpty else {
            throw XCTSkip("set IK_VAL_MADLAD and at least one IK_PARITY_MADLAD_<pair> record")
        }
        let translator = try NFKMLXMADLAD.translator(directoryURL: URL(fileURLWithPath: directory))
        for (pair, record) in records {
            let sentences = Self.probes[pair.source]!
            for (index, sentence) in sentences.enumerated() {
                XCTAssertEqual(translator.sourceIds(for: sentence, target: pair.target), ints(record["tokens_\(index)"]!),
                               "\(pair.recordKey) tokens of \(sentence)")
            }
            var decoding = translator.defaultDecoding
            decoding.maxTokens = 64
            checkSeams(translator.net, encode: { translator.net.encode($0) },
                       decode: { translator.net.decode($0, memory: $1, cache: translator.net.makeCache()) },
                       record: record, decoding: decoding,
                       sourceIds: translator.sourceIds(for: sentences[1], target: pair.target)!, name: pair.recordKey)
            var greedy = decoding
            greedy.beams = 1
            let greedyText = try translator.translate(sentences[1], from: pair.source, to: pair.target, decoding: greedy)
            XCTAssertEqual(greedyText, utf8(record["greedy_text"]!), "\(pair.recordKey) greedy text")
            var beam = decoding
            beam.beams = ints(record["beams"]!)[0]
            let beamText = try translator.translate(sentences[1], from: pair.source, to: pair.target, decoding: beam)
            XCTAssertEqual(beamText, utf8(record["beam_text"]!), "\(pair.recordKey) beam text")
            print("\(pair.recordKey):", greedyText)
            XCTAssertEqual(translator.segmenter.encode(Self.targets[pair.target]!) + [2], ints(record["target_ids"]!),
                           "\(pair.recordKey) target tokenization")
            let loss = NFKMLXTranslationObjective()(translator.net, record["source_ids"]!, record["target_ids"]!)
            XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: 1e-2, "\(pair.recordKey) training loss")
        }
    }

    func testTranslateGemmaProbePairsMatchTheReference() throws {
        try requireMLXRuntime()
        let pairs = Self.multilingualProbePairs("TRANSLATEGEMMA", weightsKey: "IK_VAL_TRANSLATEGEMMA")
        let records = try pairs.compactMap { pair in try record(pair.recordKey).map { (pair, $0) } }
        guard let directory = NFKMLXValidationConfig.environment["IK_VAL_TRANSLATEGEMMA"], !records.isEmpty else {
            throw XCTSkip("set IK_VAL_TRANSLATEGEMMA and at least one IK_PARITY_TRANSLATEGEMMA_<pair> record")
        }
        let translator = try NFKMLXTranslateGemma.translator(directoryURL: URL(fileURLWithPath: directory), precision: .float32)
        for (pair, record) in records {
            let sentence = Self.probes[pair.source]![1]
            let ids = translator.promptTokens(text: sentence, sourceCode: translator.code(for: pair.source)!, targetCode: translator.code(for: pair.target)!)
            XCTAssertEqual(ids, ints(record["tokens"]!), "\(pair.recordKey) template ids")
            let logits = translator.model.logits(tokens: ids, softTokens: nil)
            let tail = logits[0, (ids.count - 16)..., 0...]
            let logitCosine = cosine(tail, record["output"]!)
            XCTAssertGreaterThan(logitCosine, 0.999, "\(pair.recordKey) logits")
            let produced = try translator.generate(promptTokens: ids, maxTokens: 48)
            let reference = ints(record["continuation"]!).filter { $0 != 1 && $0 != 106 }
            XCTAssertEqual(produced, reference, "\(pair.recordKey) greedy continuation")
            var decoding = translator.defaultDecoding
            decoding.maxTokens = 48
            let text = try translator.translate(sentence, from: pair.source, to: pair.target, decoding: decoding)
            XCTAssertEqual(text, utf8(record["greedy_text"]!).trimmingCharacters(in: .whitespacesAndNewlines), "\(pair.recordKey) greedy text")
            print("\(pair.recordKey):", text, "| logit cosine \(logitCosine)")
            let loss = NFKMLXTranslateGemmaObjective()(translator.model.decoder, record["tokens"]!, record["target_ids"]!)
            XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: 1e-2, "\(pair.recordKey) fine-tuning loss")
        }
    }

    // MARK: Customization

    func testTheObjectiveShiftsTheTargetBehindTheStartToken() throws {
        let input = NFKMLXTranslationObjective.decoderInput(for: MLXArray([Int32(7), 8, 9, 0]), startToken: 63)
        XCTAssertEqual(input.asArray(Int32.self), [63, 7, 8, 9])
        XCTAssertEqual(NFKMLXTranslationObjective.decoderInput(for: MLXArray([Int32(0)]), startToken: 63).asArray(Int32.self), [63])
    }

    func testTheObjectiveScoresAlignedPredictionsAsCertain() throws {
        try requireMLXRuntime()
        let target = MLXArray([Int32(3), 5, 0])
        var logits = [Float](repeating: -20, count: 3 * 8)
        logits[0 * 8 + 3] = 20; logits[1 * 8 + 5] = 20; logits[2 * 8 + 0] = 20
        let loss = NFKMLXTranslationObjective().loss(logits: MLXArray(logits, [1, 3, 8]), target: target)
        XCTAssertLessThan(loss.item(Float.self), 1e-4)
    }

    func testMarianFineTuningAdaptsTheDecoderAndRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        try checkFineTuningRoundTrip(
            network: { try NFKMLXMarian.network(directoryURL: $0, configuration: .tinyMarian) },
            fineTune: { net, source, target in try NFKMLXMarian.fineTune(net, examples: { _ in (source, target) }, rank: 2, steps: 12) },
            source: [5, 6, 7, 0], target: [9, 10, 11, 0])
    }

    func testM2M100FineTuningAdaptsTheDecoderAndRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        try checkFineTuningRoundTrip(
            network: { try NFKMLXM2M100.network(directoryURL: $0, configuration: .tinyM2M100) },
            fineTune: { net, source, target in try NFKMLXM2M100.fineTune(net, examples: { _ in (source, target) }, rank: 2, steps: 12) },
            source: [60, 5, 6, 7, 2], target: [61, 9, 10, 11, 2])
    }

    func testNLLBFineTuningAdaptsTheDecoderAndRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        try checkFineTuningRoundTrip(
            network: { try NFKMLXNLLB.network(directoryURL: $0, configuration: .tinyM2M100) },
            fineTune: { net, source, target in try NFKMLXNLLB.fineTune(net, examples: { _ in (source, target) }, rank: 2, steps: 12) },
            source: [60, 5, 6, 7, 2], target: [61, 9, 10, 11, 2])
    }

    /// Fine-tunes a tiny network on one pair, merges the adapters, saves, and reloads the checkpoint
    /// through the model's own `network(directoryURL:)`.
    private func checkFineTuningRoundTrip(network: (URL?) throws -> NFKMLXSeq2SeqNet,
                                          fineTune: (NFKMLXSeq2SeqNet, MLXArray, MLXArray) throws -> [Float],
                                          source sourceIds: [Int32], target targetIds: [Int32],
                                          file: StaticString = #filePath, line: UInt = #line) throws {
        let net = try network(nil)
        let encoderBefore = net.encoder!.layers[0].fc1.weight
        eval(encoderBefore)
        let source = MLXArray(sourceIds)
        let target = MLXArray(targetIds)
        let objective = NFKMLXTranslationObjective()
        let before = objective(net, source, target).item(Float.self)
        let losses = try fineTune(net, source, target)
        XCTAssertEqual(losses.count, 12, file: file, line: line)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the pair it trains on", file: file, line: line)
        XCTAssertEqual(abs(net.encoder!.layers[0].fc1.weight - encoderBefore).max().item(Float.self), 0, "the encoder stays frozen", file: file, line: line)
        let adapted = net.leafModules().flattened().filter { $0.1 is NFKMLXLoRALinear }.map(\.0)
        XCTAssertFalse(adapted.isEmpty, file: file, line: line)
        XCTAssertTrue(adapted.allSatisfy { $0.hasPrefix("decoder.layers.") && ($0.hasSuffix(".q_proj") || $0.hasSuffix(".v_proj")) }, file: file, line: line)
        let merged = try NFKMLXLoRA.merge(into: net)
        XCTAssertEqual(merged, adapted.count, file: file, line: line)
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        try NFKMLXWeights.save(net, to: scratch.appendingPathComponent("model.safetensors"))
        let reloaded = try network(scratch)
        let shape = [1, sourceIds.count]
        let a = net(source: source.reshaped(shape), target: target.reshaped([1, targetIds.count]))
        let b = reloaded(source: source.reshaped(shape), target: target.reshaped([1, targetIds.count]))
        XCTAssertLessThan(abs(a - b).max().item(Float.self), 1e-5, "the merged checkpoint reloads through the factory", file: file, line: line)
    }

    func testTheNetworkFactoryTakesTheReleasesOwnGeometry() throws {
        try requireMLXRuntime()
        // A geometry unlike every test preset, so adopting a preset instead of the release shows.
        let config: [String: Any] = [
            "model_type": "marian", "vocab_size": 40, "d_model": 24, "encoder_layers": 1, "decoder_layers": 3,
            "encoder_attention_heads": 2, "encoder_ffn_dim": 48, "decoder_ffn_dim": 48,
            "max_position_embeddings": 32, "activation_function": "swish", "scale_embedding": true,
            "pad_token_id": 39, "eos_token_id": 0, "decoder_start_token_id": 39,
        ]
        let release = NFKMLXSeq2SeqNet(try NFKMLXSeq2SeqConfiguration(huggingFaceConfig: config))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try NFKMLXWeights.save(release, to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))

        let loaded = try NFKMLXMarian.network(directoryURL: directory)
        XCTAssertEqual(loaded.configuration.dModel, 24)
        XCTAssertEqual(loaded.configuration.encoderLayers, 1)
        XCTAssertEqual(loaded.configuration.decoderLayers, 3)
        XCTAssertEqual(loaded.configuration.vocabularySize, 40)
        let source = MLXArray([Int32(5), 6, 7, 0]).reshaped([1, 4])
        let target = MLXArray([Int32(39), 10, 11, 0]).reshaped([1, 4])
        let difference = abs(release(source: source, target: target) - loaded(source: source, target: target)).max()
        XCTAssertLessThan(difference.item(Float.self), 1e-6, "the release's weights load into the release's geometry")
    }

    // MARK: Dropout

    func testDropoutRatesReadFromEachFamilysConfig() throws {
        let m2m = NFKMLXSeq2SeqDropout(huggingFaceConfig: [
            "dropout": 0.1, "attention_dropout": 0.1, "activation_dropout": 0.0,
            "encoder_layerdrop": 0.05, "decoder_layerdrop": 0.05])
        XCTAssertEqual(m2m, NFKMLXSeq2SeqDropout(dropout: 0.1, attentionDropout: 0.1, encoderLayerDrop: 0.05,
                                                 decoderLayerDrop: 0.05))
        XCTAssertEqual(NFKMLXSeq2SeqDropout(huggingFaceConfig: ["dropout_rate": 0.1]),
                       NFKMLXSeq2SeqDropout(dropout: 0.1, attentionDropout: 0.1, activationDropout: 0.1),
                       "T5's one rate fills the three positions it applies at")
        XCTAssertEqual(NFKMLXSeq2SeqDropout(huggingFaceConfig: [:]), .none)

        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        for (key, nested) in [("text_config", 0.1), ("decoder", 0.2)] {
            let json: [String: Any] = ["dropout": 0.9, key: ["dropout": nested, "attention_dropout": nested]]
            try JSONSerialization.data(withJSONObject: json).write(to: scratch.appendingPathComponent("config.json"))
            let rates = try NFKMLXSeq2SeqDropout(releaseDirectoryURL: scratch)
            XCTAssertEqual(rates.dropout, Float(nested), accuracy: 1e-7, "\(key) holds the language model's rates")
            XCTAssertEqual(rates.attentionDropout, Float(nested), accuracy: 1e-7)
        }
    }

    private static let heavyDropout = NFKMLXSeq2SeqDropout(dropout: 0.3, attentionDropout: 0.3, activationDropout: 0.3,
                                                           encoderLayerDrop: 0.3, decoderLayerDrop: 0.3)

    func testANetworkAppliesNoDropoutInEvaluation() throws {
        try requireMLXRuntime()
        let source = MLXArray([Int32(5), 6, 7, 8]).reshaped([1, 4])
        let target = MLXArray([Int32(1), 9, 10, 11]).reshaped([1, 4])
        let net = NFKMLXSeq2SeqNet(.tinyM2M100)
        XCTAssertFalse(net.training, "built in evaluation mode")
        let plain = net(source: source, target: target)
        net.dropout = Self.heavyDropout
        XCTAssertEqual(abs(net(source: source, target: target) - plain).max().item(Float.self), 0)

        let t5 = NFKMLXT5Seq2SeqNet(.tiny)
        XCTAssertFalse(t5.training)
        let t5Plain = t5(source: source, target: target)
        t5.dropout = Self.heavyDropout
        XCTAssertEqual(abs(t5(source: source, target: target) - t5Plain).max().item(Float.self), 0)
    }

    func testDropoutRandomizesOnlyTheTrainingForward() throws {
        try requireMLXRuntime()
        let source = MLXArray([Int32(5), 6, 7, 8]).reshaped([1, 4])
        let target = MLXArray([Int32(1), 9, 10, 11]).reshaped([1, 4])
        let nets: [(String, Module, (MLXArray, MLXArray) -> MLXArray, (NFKMLXSeq2SeqDropout) -> Void)] = {
            let marian = NFKMLXSeq2SeqNet(.tinyMarian), t5 = NFKMLXT5Seq2SeqNet(.tiny)
            return [("Marian", marian, { marian(source: $0, target: $1) }, { marian.dropout = $0 }),
                    ("T5", t5, { t5(source: $0, target: $1) }, { t5.dropout = $0 })]
        }()
        for (name, net, forward, setDropout) in nets {
            let evaluated = forward(source, target)
            net.train(true)
            XCTAssertEqual(abs(forward(source, target) - evaluated).max().item(Float.self), 0,
                           "\(name): no rate, no change in training")
            setDropout(NFKMLXSeq2SeqDropout(dropout: 0.3, attentionDropout: 0.3, activationDropout: 0.3))
            let first = forward(source, target), second = forward(source, target)
            XCTAssertGreaterThan(abs(first - second).max().item(Float.self), 0, "\(name): each training forward draws")
            XCTAssertTrue(isFinite(first).all().item(Bool.self))
            net.train(false)
            XCTAssertEqual(abs(forward(source, target) - evaluated).max().item(Float.self), 0, "\(name): evaluation again")
        }
    }

    func testAFullLayerDropSkipsEveryLayer() throws {
        try requireMLXRuntime()
        let source = MLXArray([Int32(5), 6, 7, 8]).reshaped([1, 4])
        let target = MLXArray([Int32(2), 9, 10, 11]).reshaped([1, 4])
        let net = NFKMLXSeq2SeqNet(.tinyM2M100)
        net.train(true)
        net.dropout = NFKMLXSeq2SeqDropout(encoderLayerDrop: 1, decoderLayerDrop: 1)
        let skipped = net(source: source, target: target)
        net.encoder!.layers[0].fc1.update(parameters: ModuleParameters.unflattened([("weight", net.encoder!.layers[0].fc1.weight * 5)]))
        net.decoder.layers[1].fc2.update(parameters: ModuleParameters.unflattened([("weight", net.decoder.layers[1].fc2.weight * 5)]))
        XCTAssertEqual(abs(net(source: source, target: target) - skipped).max().item(Float.self), 0,
                       "no layer's weights reach the output")
    }

    func testTheExplicitAttentionMatchesTheFusedKernel() throws {
        try requireMLXRuntime()
        let queries = MLXRandom.normal([1, 2, 3, 4]), keys = MLXRandom.normal([1, 2, 5, 4])
        let values = MLXRandom.normal([1, 2, 5, 4]), mask = MLXRandom.normal([1, 2, 3, 5])
        let fused = MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: 0.5, mask: mask)
        let explicit = NFKDropout.explicitAttention(queries: queries, keys: keys, values: values, scale: 0.5, mask: mask)
        XCTAssertLessThan(abs(fused - explicit).max().item(Float.self), 1e-5)
    }

    func testMarianFineTunesWithTheReleaseDropoutAndReturnsToEvaluation() throws {
        try requireMLXRuntime()
        MLXRandom.seed(11)
        let net = try NFKMLXMarian.network(directoryURL: nil, configuration: .tinyMarian)
        net.dropout = NFKMLXSeq2SeqDropout(dropout: 0.1)
        let source = MLXArray([Int32(5), 6, 7, 0])
        let target = MLXArray([Int32(9), 10, 11, 0])
        let before = NFKMLXTranslationObjective()(net, source, target).item(Float.self)
        let losses = try NFKMLXMarian.fineTune(net, examples: { _ in (source, target) }, rank: 2, steps: 12)
        XCTAssertTrue(losses.allSatisfy(\.isFinite))
        XCTAssertLessThan(NFKMLXTranslationObjective()(net, source, target).item(Float.self), before)
        XCTAssertFalse(net.training, "the run restores evaluation, so inference does not drop")
        let a = net(source: source.reshaped([1, 4]), target: target.reshaped([1, 4]))
        let b = net(source: source.reshaped([1, 4]), target: target.reshaped([1, 4]))
        XCTAssertEqual(abs(a - b).max().item(Float.self), 0)
    }

    func testMADLADFineTuningAdaptsTheDecoderAndReloads() throws {
        try requireMLXRuntime()
        let net = try NFKMLXMADLAD.network(directoryURL: nil, configuration: .tiny)
        let source = MLXArray([Int32(5), 6, 7, 2])
        let target = MLXArray([Int32(9), 10, 11, 2])
        let before = NFKMLXTranslationObjective()(net, source, target).item(Float.self)
        let losses = try NFKMLXMADLAD.fineTune(net, examples: { _ in (source, target) }, rank: 2, steps: 12)
        XCTAssertLessThan(losses.last!, before)
        let adapted = net.leafModules().flattened().filter { $0.1 is NFKMLXLoRALinear }.map(\.0)
        XCTAssertTrue(adapted.allSatisfy { $0.hasPrefix("decoder.block.") && ($0.hasSuffix(".q") || $0.hasSuffix(".v")) })
        XCTAssertFalse(adapted.isEmpty)
        try NFKMLXLoRA.merge(into: net)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        try NFKMLXWeights.save(net, to: url)
        let reloaded = NFKMLXT5Seq2SeqNet(.tiny)
        try reloaded.loadWeights(from: url)
        let a = net(source: source.reshaped([1, 4]), target: target.reshaped([1, 4]))
        let b = reloaded(source: source.reshaped([1, 4]), target: target.reshaped([1, 4]))
        XCTAssertLessThan(abs(a - b).max().item(Float.self), 1e-5)
    }
}
