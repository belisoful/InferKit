//
//  NFKMLXTextEmbedderTests.swift
//  InferKitMLXTests
//
//  Weight-free structural tests for the text embedder: pooling, Matryoshka truncation, normalization,
//  and the public backend surface. Numeric parity against Qwen3-Embedding's own reference lives in
//  NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXTextEmbedderTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    private func tinyNet() -> NFKMLXLanguageNet {
        NFKMLXRandom.seed(3)
        return NFKMLXLanguage.makeNet(.tiny)
    }

    private func length(_ vector: MLXArray) -> Float {
        eval(vector)
        return sqrt(vector.asArray(Float.self).map { $0 * $0 }.reduce(0, +))
    }

    func testTheEmbeddingIsUnitLengthAndDeterministic() throws {
        try requireMLXRuntime()
        let embedder = NFKMLXTextEmbedder(net: tinyNet(), configuration:
            NFKMLXTextEmbedderConfiguration(pooling: .lastToken, appendedToken: 1, normalizes: true))
        let first = embedder.embed(tokens: [3, 17, 42, 5])
        let second = embedder.embed(tokens: [3, 17, 42, 5])
        XCTAssertEqual(length(first), 1, accuracy: 1e-4, "a normalized embedding is unit length")
        XCTAssertEqual(first.asArray(Float.self), second.asArray(Float.self),
                       "the same input embeds identically")
    }

    func testMatryoshkaTruncationShortensAndRenormalizes() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let full = NFKMLXTextEmbedder(net: net, configuration:
            NFKMLXTextEmbedderConfiguration(pooling: .lastToken, appendedToken: nil, normalizes: true))
            .embed(tokens: [3, 17, 42, 5])
        let truncated = NFKMLXTextEmbedder(net: net, configuration:
            NFKMLXTextEmbedderConfiguration(pooling: .lastToken, appendedToken: nil, normalizes: true,
                                            dimensions: 16))
            .embed(tokens: [3, 17, 42, 5])
        eval(full, truncated)
        XCTAssertEqual(truncated.dim(0), 16, "the embedding is truncated to the requested width")
        XCTAssertEqual(length(truncated), 1, accuracy: 1e-4, "the truncated embedding is renormalized")
        // The truncated embedding is the full one's leading slice, renormalized (to float precision:
        // the norm is reduced over a different graph, so the last bits differ).
        let leading = full[0 ..< 16]
        let renormalized = leading / sqrt((leading * leading).sum())
        eval(renormalized)
        for (ours, expected) in zip(truncated.asArray(Float.self), renormalized.asArray(Float.self)) {
            XCTAssertEqual(ours, expected, accuracy: 1e-5, "truncation takes a leading slice")
        }
    }

    func testLastTokenAndMeanPoolingDiffer() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let last = NFKMLXTextEmbedder(net: net, configuration:
            NFKMLXTextEmbedderConfiguration(pooling: .lastToken, appendedToken: nil, normalizes: false))
            .embed(tokens: [3, 17, 42, 5, 9])
        let mean = NFKMLXTextEmbedder(net: net, configuration:
            NFKMLXTextEmbedderConfiguration(pooling: .mean, appendedToken: nil, normalizes: false))
            .embed(tokens: [3, 17, 42, 5, 9])
        eval(last, mean)
        XCTAssertNotEqual(last.asArray(Float.self), mean.asArray(Float.self),
                          "the two poolings read different positions")
    }

    func testTheAppendedTokenChangesTheLastTokenEmbedding() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let withEnd = NFKMLXTextEmbedder(net: net, configuration:
            NFKMLXTextEmbedderConfiguration(pooling: .lastToken, appendedToken: 1, normalizes: true))
            .embed(tokens: [3, 17, 42])
        let withoutEnd = NFKMLXTextEmbedder(net: net, configuration:
            NFKMLXTextEmbedderConfiguration(pooling: .lastToken, appendedToken: nil, normalizes: true))
            .embed(tokens: [3, 17, 42])
        eval(withEnd, withoutEnd)
        XCTAssertNotEqual(withEnd.asArray(Float.self), withoutEnd.asArray(Float.self),
                          "appending the end token moves the pooled position")
    }

    func testTheBackendEmbeddingForTokensMatchesTheEmbedder() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let configuration = NFKMLXTextEmbedderConfiguration(pooling: .lastToken, appendedToken: 1,
                                                            normalizes: true)
        let backend = NFKMLXTextEmbeddingBackend(embedder: NFKMLXTextEmbedder(net: net,
                                                 configuration: configuration),
                                                 tokenizer: nil, identifier: "tiny-embedder")
        XCTAssertEqual(backend.embeddingDimensions, NFKMLXLanguageConfiguration.tiny.hiddenSize)
        let viaBackend = backend.embedding(forTokens: [3, 17, 42].map { NSNumber(value: $0) })
            .map { $0.floatValue }
        let direct = NFKMLXTextEmbedder(net: net, configuration: configuration).embed(tokens: [3, 17, 42])
        eval(direct)
        XCTAssertEqual(viaBackend, direct.asArray(Float.self), "the backend runs the embedder's path")
    }

    func testTheInstructionFormatMatchesTheReferenceTemplate() {
        XCTAssertEqual(NFKMLXQwen3Embedding.instruct(task: "Retrieve relevant passages",
                                                     query: "capital of France"),
                       "Instruct: Retrieve relevant passages\nQuery:capital of France")
    }
}
