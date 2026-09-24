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

    // MARK: SentencePiece (no MLX)

    /// A protobuf `ModelProto` assembled by hand: two normal pieces, the unknown marker, and a
    /// trainer spec naming the algorithm.
    private func syntheticModel(kind: UInt8, pieces: [(String, Float)]) throws -> NFKMLXSentencePieceModel {
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
        body += lengthDelimited(field: 3, lengthDelimited(field: 1, Array("nmt_nfkc".utf8)))
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

    func testNormalizationFollowsNMTNFKC() {
        XCTAssertEqual(NFKMLXSentencePieceSegmenter.nmtNFKC("\u{fb01}ne \u{bd}"), "fine 1\u{2044}2")
        XCTAssertEqual(NFKMLXSentencePieceSegmenter.collapseWhitespace("  spaced   out  text  "), "spaced out text")
        XCTAssertEqual(NFKMLXSentencePieceSegmenter.nmtNFKC("a\u{a0}b\u{200b}c"), "a bc")
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
                                                          sourceIds: [Int], name: String) {
        XCTAssertEqual(sourceIds, ints(record["source_ids"]!), "\(name) source ids")
        let source = MLXArray(sourceIds.map { Int32($0) }).reshaped([1, sourceIds.count])
        let memory = encode(source)
        let encoderCosine = cosine(memory[0], record["encoder_hidden"]!)
        XCTAssertGreaterThan(encoderCosine, 0.999, "\(name) encoder")
        let decoderInput = record["decoder_input"]!
        let logits = decode(decoderInput.reshaped([1, decoderInput.dim(0)]), memory)
        let logitCosine = cosine(logits[0], record["output"]!)
        XCTAssertGreaterThan(logitCosine, 0.999, "\(name) teacher-forced logits")
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
        try requireMLXRuntime()
        let (directory, record) = try release("IK_VAL_M2M100", "IK_PARITY_M2M100")
        let translator = try NFKMLXM2M100.translator(directoryURL: directory)
        XCTAssertEqual(translator.languageIds["en"], 128022)
        for (index, sentence) in Self.sentences.enumerated() {
            XCTAssertEqual(translator.sourceIds(for: sentence, source: "en", target: "de"), ints(record["tokens_\(index)"]!), "tokens of \(sentence)")
        }
        var decoding = translator.defaultDecoding
        decoding.maxTokens = 64
        decoding.forcedFirstToken = translator.languageIds["de"]
        checkSeams(translator.net, encode: { translator.net.encode($0) }, decode: { translator.net.decode($0, memory: $1, cache: translator.net.makeCache()) },
                   record: record, decoding: decoding, sourceIds: translator.sourceIds(for: Self.sentences[1], source: "en", target: "de"), name: "m2m100")
        let text = try translator.translate(Self.sentences[1], from: "en", to: "de", decoding: decoding)
        print("m2m100:", text)
        XCTAssertFalse(text.isEmpty)
        XCTAssertEqual([translator.languageIds["de"]!] + translator.tokenizer.encode(Self.target, dummyPrefix: nil) + [2],
                       ints(record["target_ids"]!), "target tokenization")
        let loss = NFKMLXTranslationObjective()(translator.net, record["source_ids"]!, record["target_ids"]!)
        XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: 1e-2, "training loss")
        print("loss mine \(loss.item(Float.self)) reference \(record["loss"]!.asArray(Float.self)[0])")
    }

    func testMADLADMatchesTheReference() throws {
        try requireMLXRuntime()
        let (directory, record) = try release("IK_VAL_MADLAD", "IK_PARITY_MADLAD")
        let translator = try NFKMLXMADLAD.translator(directoryURL: directory)
        for (index, sentence) in Self.sentences.enumerated() {
            XCTAssertEqual(translator.sourceIds(for: sentence, target: "de"), ints(record["tokens_\(index)"]!), "tokens of \(sentence)")
        }
        var decoding = translator.defaultDecoding
        decoding.maxTokens = 64
        checkSeams(translator.net, encode: { translator.net.encode($0) }, decode: { translator.net.decode($0, memory: $1, cache: translator.net.makeCache()) },
                   record: record, decoding: decoding, sourceIds: translator.sourceIds(for: Self.sentences[1], target: "de")!, name: "madlad")
        let text = try translator.translate(Self.sentences[1], from: nil, to: "de", decoding: decoding)
        print("madlad:", text)
        XCTAssertFalse(text.isEmpty)
        XCTAssertEqual(translator.segmenter.encode(Self.target) + [2], ints(record["target_ids"]!), "target tokenization")
        let loss = NFKMLXTranslationObjective()(translator.net, record["source_ids"]!, record["target_ids"]!)
        XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0], accuracy: 1e-2, "training loss")
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
    /// and the port loads at `.checkpoint`, so the comparison is bfloat16 against bfloat16.
    func testTranslateGemma12BMatchesTheReference() throws {
        try checkTranslateGemma(weightsKey: "IK_VAL_TRANSLATEGEMMA_12B", recordKey: "IK_PARITY_TRANSLATEGEMMA_12B",
                                precision: .checkpoint, name: "translategemma-12b")
    }

    private func checkTranslateGemma(weightsKey: String, recordKey: String, precision: NFKMLXWeightPrecision, name: String) throws {
        try requireMLXRuntime()
        let (directory, record) = try release(weightsKey, recordKey)
        let translator = try NFKMLXTranslateGemma.translator(directoryURL: directory, precision: precision)
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
        let text = try translator.translate(Self.sentences[1], from: "en", to: "de", decoding: translator.defaultDecoding)
        print("\(name):", text, "| logit cosine \(logitCosine); argmax \(agreeing)/16")
        XCTAssertFalse(text.isEmpty)
        let loss = NFKMLXTranslateGemmaObjective()(translator.model.decoder, record["tokens"]!, record["target_ids"]!)
        // Two bfloat16 stacks differ in accumulation order, and a 262k-way cross-entropy over a dozen
        // positions amplifies the logit drift; the 12B measures 0.43 against 0.48.
        XCTAssertEqual(loss.item(Float.self), record["loss"]!.asArray(Float.self)[0],
                       accuracy: precision == .float32 ? 1e-2 : 0.1, "fine-tuning loss")
        print("loss mine \(loss.item(Float.self)) reference \(record["loss"]!.asArray(Float.self)[0])")
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
        let net = try NFKMLXMarian.network(directoryURL: nil, configuration: .tinyMarian)
        let encoderBefore = net.encoder!.layers[0].fc1.weight
        eval(encoderBefore)
        let source = MLXArray([Int32(5), 6, 7, 0])
        let target = MLXArray([Int32(9), 10, 11, 0])
        let objective = NFKMLXTranslationObjective()
        let before = objective(net, source, target).item(Float.self)
        let losses = try NFKMLXMarian.fineTune(net, examples: { _ in (source, target) }, rank: 2, steps: 12)
        XCTAssertEqual(losses.count, 12)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the pair it trains on")
        XCTAssertEqual(abs(net.encoder!.layers[0].fc1.weight - encoderBefore).max().item(Float.self), 0, "the encoder stays frozen")
        let adapted = net.leafModules().flattened().filter { $0.1 is NFKMLXLoRALinear }.map(\.0)
        XCTAssertFalse(adapted.isEmpty)
        XCTAssertTrue(adapted.allSatisfy { $0.hasPrefix("decoder.layers.") && ($0.hasSuffix(".q_proj") || $0.hasSuffix(".v_proj")) })
        let merged = try NFKMLXLoRA.merge(into: net)
        XCTAssertEqual(merged, adapted.count)
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        try NFKMLXWeights.save(net, to: scratch.appendingPathComponent("model.safetensors"))
        let reloaded = try NFKMLXMarian.network(directoryURL: scratch, configuration: .tinyMarian)
        let a = net(source: source.reshaped([1, 4]), target: target.reshaped([1, 4]))
        let b = reloaded(source: source.reshaped([1, 4]), target: target.reshaped([1, 4]))
        XCTAssertLessThan(abs(a - b).max().item(Float.self), 1e-5, "the merged checkpoint reloads through the factory")
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
