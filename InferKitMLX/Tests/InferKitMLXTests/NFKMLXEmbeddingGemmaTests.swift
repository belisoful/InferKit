//
//  NFKMLXEmbeddingGemmaTests.swift
//  InferKitMLXTests
//
//  Weight-free structural tests for the EmbeddingGemma encoder, its Dense head, and the Gemma BPE
//  tokenizer. Numeric parity against the reference lives in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXEmbeddingGemmaTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    private func tinyEmbedder(dimensions: Int? = nil,
                              prepended: Int? = nil, appended: Int? = nil) -> NFKMLXEmbeddingGemmaEmbedder {
        NFKMLXRandom.seed(5)
        let net = NFKMLXGemma3EncoderNet(.tiny)
        let width = NFKMLXGemma3EncoderConfiguration.tiny.hiddenSize
        // Random projections, so the Dense head carries the pooled vector rather than mapping every
        // input to the same constant.
        let dense2 = MLXRandom.normal([4 * width, width]) * 0.05
        let dense3 = MLXRandom.normal([width, 4 * width]) * 0.05
        return NFKMLXEmbeddingGemmaEmbedder(
            net: net, dense2: dense2, dense3: dense3,
            configuration: NFKMLXEmbeddingGemmaConfiguration(prependedToken: prepended,
                                                             appendedToken: appended, dimensions: dimensions))
    }

    private func length(_ vector: MLXArray) -> Float {
        eval(vector)
        return sqrt(vector.asArray(Float.self).map { $0 * $0 }.reduce(0, +))
    }

    func testTheEmbeddingIsUnitLengthAndDeterministic() throws {
        try requireMLXRuntime()
        let embedder = tinyEmbedder()
        let first = embedder.embed(tokens: [3, 17, 42, 5])
        let second = embedder.embed(tokens: [3, 17, 42, 5])
        XCTAssertEqual(first.dim(0), 64, "the full embedding width")
        XCTAssertEqual(length(first), 1, accuracy: 1e-4, "a normalized embedding is unit length")
        XCTAssertEqual(first.asArray(Float.self), second.asArray(Float.self))
    }

    func testMatryoshkaTruncationShortensAndRenormalizes() throws {
        try requireMLXRuntime()
        let truncated = tinyEmbedder(dimensions: 16).embed(tokens: [3, 17, 42, 5])
        XCTAssertEqual(truncated.dim(0), 16, "truncated to the requested width")
        XCTAssertEqual(length(truncated), 1, accuracy: 1e-4, "the truncated embedding is renormalized")
        XCTAssertEqual(tinyEmbedder().embeddingDimensions, 64)
        XCTAssertEqual(tinyEmbedder(dimensions: 16).embeddingDimensions, 16)
    }

    func testTheMarkersChangeTheEmbedding() throws {
        try requireMLXRuntime()
        let plain = tinyEmbedder().embed(tokens: [3, 17, 42])
        let wrapped = tinyEmbedder(prepended: 2, appended: 1).embed(tokens: [3, 17, 42])
        eval(plain, wrapped)
        XCTAssertNotEqual(plain.asArray(Float.self), wrapped.asArray(Float.self),
                          "the BOS and EOS markers add tokens the mean pooling reads")
    }

    func testTheSlidingWindowEngagesForALongInput() throws {
        try requireMLXRuntime()
        // The tiny config's window is 4; a sequence longer than that exercises the mask path rather
        // than the all-visible shortcut.
        let embedding = tinyEmbedder().embed(tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9])
        XCTAssertEqual(length(embedding), 1, accuracy: 1e-4, "the windowed forward still produces a vector")
    }

    func testTheFullAttentionLayerIsPlacedEverySixth() {
        let c = NFKMLXGemma3EncoderConfiguration.embeddingGemma300M
        XCTAssertFalse(c.isFullAttention(layer: 0))
        XCTAssertTrue(c.isFullAttention(layer: 5), "the sixth layer is full attention")
        XCTAssertTrue(c.isFullAttention(layer: 11))
        XCTAssertEqual((0 ..< 24).filter { c.isFullAttention(layer: $0) }.count, 4)
    }

    func testTheBackendFactoryBuildsWithRandomWeights() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(1)
        let backend = try NFKMLXEmbeddingGemma.backend(
            weightsURL: nil, dense2URL: nil, dense3URL: nil, tokenizer: nil,
            configuration: .tiny) as? NFKMLXTextEmbeddingBackend
        let embedder = try XCTUnwrap(backend)
        XCTAssertEqual(embedder.backendIdentifier, "embeddinggemma-300m")
        let embedding = embedder.embedding(forTokens: [3, 17, 42].map { NSNumber(value: $0) })
        XCTAssertEqual(embedding.count, embedder.embeddingDimensions)
    }

    func testThePromptHelpersMatchTheTrainedFormat() {
        XCTAssertEqual(NFKMLXEmbeddingGemma.query("hello"), "task: search result | query: hello")
        XCTAssertEqual(NFKMLXEmbeddingGemma.document("hello"), "title: none | text: hello")
    }

    // The Gemma BPE tokenizer's merge loop, on a hand-built tokenizer.json: a metaspace-normalized
    // string merges by rank, and an out-of-vocabulary character falls back to its UTF-8 bytes.
    func testTheGemmaBPETokenizerMergesAndFallsBack() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let vocab: [String: Int] = ["a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, "\u{2581}": 5,
                                    "\u{2581}a": 6, "<0xE2>": 7, "<0x9C>": 8, "<0x93>": 9]
        let model: [String: Any] = ["vocab": vocab,
                                    "merges": [["a", "b"], ["ab", "c"], ["\u{2581}", "a"]],
                                    "unk_token": "a"]
        let json: [String: Any] = ["model": model]
        try JSONSerialization.data(withJSONObject: json)
            .write(to: directory.appendingPathComponent("tokenizer.json"))

        let tokenizer = try XCTUnwrap(NFKMLXGemmaTokenizer(directoryURL: directory))
        // "abc" merges a+b -> ab (rank 0), then ab+c -> abc (rank 1): one token.
        XCTAssertEqual(tokenizer.encode("abc"), [4])
        // A leading space becomes the metaspace and merges with the following letter.
        XCTAssertEqual(tokenizer.encode(" a"), [6])
        // "✓" (U+2713) is not in the vocabulary, so it falls back to its three UTF-8 bytes.
        XCTAssertEqual(tokenizer.encode("\u{2713}"), [7, 8, 9])
    }
}
