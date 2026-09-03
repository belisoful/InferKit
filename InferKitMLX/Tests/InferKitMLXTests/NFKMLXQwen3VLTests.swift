//
//  NFKMLXQwen3VLTests.swift
//  InferKitMLXTests
//
//  Weight-free structural tests for the Qwen3-VL vision tower. Numeric parity against the reference
//  lives in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXQwen3VLTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    private static let tiny = NFKMLXQwen3VLVisionConfiguration(
        hiddenSize: 32, depth: 4, headCount: 2, intermediateSize: 64, patchSize: 2, temporalPatchSize: 2,
        spatialMergeSize: 2, outHiddenSize: 16, positionGridSide: 4, deepstackLayers: [1, 2])

    func testTheVisionTowerMergesAndProducesDeepstackFeatures() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(6)
        let net = NFKMLXQwen3VLVisionNet(Self.tiny)
        // A 4×4 patch grid: 16 patches of the flattened patch width (3·2·2·2 = 24).
        let pixels = MLXRandom.normal([16, 24])
        let (output, deepstack) = net(pixels, grid: (t: 1, h: 4, w: 4))
        eval(output)
        XCTAssertEqual(output.shape, [4, 16], "2×2 blocks fold to one token at the decoder width")
        XCTAssertEqual(deepstack.count, 2, "one deepstack feature per configured layer")
        for feature in deepstack {
            eval(feature)
            XCTAssertEqual(feature.shape, [4, 16])
        }
    }

    func testThePositionEmbeddingAndRotaryShapes() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(6)
        let net = NFKMLXQwen3VLVisionNet(Self.tiny)
        let positions = net.interpolatedPositionEmbedding(grid: (t: 1, h: 4, w: 4))
        eval(positions)
        XCTAssertEqual(positions.shape, [16, 32], "one position embedding per patch")

        let (cos, sin) = net.rotaryEmbedding(grid: (t: 1, h: 4, w: 4))
        eval(cos, sin)
        XCTAssertEqual(cos.shape, [1, 16, 16], "the rotary table covers the full head dimension")
        XCTAssertEqual(sin.shape, [1, 16, 16])
    }

    func testThePatchEmbeddingProjectsToTheHiddenWidth() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(6)
        let net = NFKMLXQwen3VLVisionNet(Self.tiny)
        let embedded = net.patchEmbed(MLXRandom.normal([16, 24]))
        eval(embedded)
        XCTAssertEqual(embedded.shape, [16, 32])
    }
}
