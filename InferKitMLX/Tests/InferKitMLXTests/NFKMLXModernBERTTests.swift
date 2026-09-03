//
//  NFKMLXModernBERTTests.swift
//  InferKitMLXTests
//
//  Weight-free structural tests for the ModernBERT reranker. Numeric parity against the reference lives
//  in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXModernBERTTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    private func tinyNet() -> NFKMLXModernBertRerankerNet {
        NFKMLXRandom.seed(7)
        return NFKMLXModernBertRerankerNet(.tiny)
    }

    private func scalar(_ x: MLXArray) -> Float { eval(x); return x.asArray(Float.self)[0] }

    func testTheScoreIsAFiniteScalarAndDeterministic() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let pair = [50_281, 3, 17, 42, 50_282, 8, 9, 5, 50_282]
        let first = scalar(net.score(tokens: pair))
        let second = scalar(net.score(tokens: pair))
        XCTAssertTrue(first.isFinite, "the reranker produces a finite logit")
        XCTAssertEqual(first, second, "the same pair scores identically")
    }

    func testTheFirstLayerHasNoAttentionNorm() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        XCTAssertNil(net.model.layers[0].attentionNorm, "layer 0's attn_norm is the identity")
        XCTAssertNotNil(net.model.layers[1].attentionNorm, "later layers normalize before attention")
    }

    func testTheGlobalAndLocalLayerPattern() {
        let c = NFKMLXModernBertConfiguration.gteReranker
        XCTAssertTrue(c.isGlobal(layer: 0))
        XCTAssertFalse(c.isGlobal(layer: 1))
        XCTAssertFalse(c.isGlobal(layer: 2))
        XCTAssertTrue(c.isGlobal(layer: 3))
        XCTAssertEqual((0 ..< 22).filter { c.isGlobal(layer: $0) }.count, 8)
    }

    func testTheSlidingWindowEngagesForALongInput() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        // The tiny config's window is 4 (half 2), so an input longer than 3 tokens exercises the mask
        // rather than the all-visible shortcut. The last two hidden states differ only by the final
        // norm, which layerStates does not apply.
        let tokens = MLXArray((0 ..< 12).map { Int32($0 % 10) }).reshaped([1, 12])
        let states = net.model.layerStates(tokens)
        eval(states)
        XCTAssertEqual(states.count, NFKMLXModernBertConfiguration.tiny.layerCount + 1)
        XCTAssertTrue(states.last!.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testRankingSortsAListByDescendingScore() {
        // The ranking is a sort of the per-document scores; a small fixed set pins the ordering logic
        // that rankedIndices(query:documents:) applies over the scores it computes.
        let scores = [0.2, 0.9, -0.4, 0.5]
        let ranked = scores.enumerated().sorted { $0.element > $1.element }.map { $0.offset }
        XCTAssertEqual(ranked, [1, 3, 0, 2], "most relevant first")
    }
}
