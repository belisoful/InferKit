//
//  NFKMLXSmolVLMTests.swift
//  InferKitMLXTests
//
//  Weight-free structural tests for the SmolVLM2 vision-language model: the SigLIP encoder, the
//  pixel-shuffle connector, the image processor's tiling, and the prompt expansion. Numeric parity
//  against the reference lives in NFKMLXReferenceParityTests.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSmolVLMTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    func testTheVisionEncoderProducesPatchFeatures() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(4)
        let net = NFKMLXSigLIPNet(.tiny)
        // Two tiles of a 64×64 image with 16-patch grid → 16 patches per tile.
        let pixels = MLXRandom.normal([2, 64, 64, 3])
        let features = net(pixels)
        XCTAssertEqual(features.shape, [2, 16, 32], "one feature per patch")
    }

    func testTheConnectorReducesTokensByScaleSquared() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(4)
        // A 8×8 = 64-patch grid at scale 4 folds to 64 / 16 = 4 tokens of 16× the channels projected.
        let connector = NFKMLXSmolVLMConnector(visionHidden: 32, decoderHidden: 48, scaleFactor: 4)
        let features = connector(MLXRandom.normal([2, 64, 32]))
        XCTAssertEqual(features.shape, [2, 4, 48], "scale² fewer tokens at the decoder width")
    }

    func testTheImageProcessorTilesToTheGrid() throws {
        try requireMLXRuntime()
        // A square image scales its longest edge to 2048 and splits 4×4, plus a global tile: 17 tiles.
        let square = NFKMLXSmolVLMTests.solidImage(width: 300, height: 300)
        let (pixels, rows, cols) = NFKMLXSmolVLMImageProcessor.process(square)
        XCTAssertEqual([rows, cols], [4, 4])
        XCTAssertEqual(pixels.shape, [17, 3, 512, 512], "16 sub-tiles and a global thumbnail")

        // A wide image scales to 2048×512 and splits 1×4, plus a global tile: 5 tiles.
        let wide = NFKMLXSmolVLMTests.solidImage(width: 1024, height: 256)
        let (widePixels, wideRows, wideCols) = NFKMLXSmolVLMImageProcessor.process(wide)
        XCTAssertEqual([wideRows, wideCols], [1, 4])
        XCTAssertEqual(widePixels.shape, [5, 3, 512, 512])

        // The pixels are normalized to -1 … 1.
        eval(pixels)
        let values = pixels.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(values.min() ?? 0, -1.0001)
        XCTAssertLessThanOrEqual(values.max() ?? 0, 1.0001)
    }

    func testThePromptExpandsToTheRightImageTokenCount() {
        // A 2×2 tiling: 4 sub-tiles + 1 global, each 64 image tokens → 320 `<image>` markers.
        let prompt = NFKMLXSmolVLM.prompt(rows: 2, cols: 2, question: "What is this?")
        let count = prompt.components(separatedBy: "<image>").count - 1
        XCTAssertEqual(count, 5 * NFKMLXSmolVLM.tokensPerTile)
        XCTAssertTrue(prompt.hasPrefix("User:"))
        XCTAssertTrue(prompt.contains("<row_2_col_2>"))
        XCTAssertTrue(prompt.contains("<global-img>"))
        XCTAssertTrue(prompt.hasSuffix("<end_of_utterance>\nAssistant:"))
    }

    func testFusionSplicesVisionTokensAtImagePositions() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(4)
        let vision = NFKMLXSigLIPNet(.tiny)
        let connector = NFKMLXSmolVLMConnector(visionHidden: 32, decoderHidden: 64, scaleFactor: 4)
        var decoderConfiguration = NFKMLXLanguageConfiguration.tiny
        decoderConfiguration.hiddenSize = 64
        let decoder = NFKMLXLanguage.makeNet(decoderConfiguration)
        let net = NFKMLXSmolVLMNet(vision: vision, connector: connector, decoder: decoder, imageTokenId: 7)

        // Two image tokens (id 7) among text tokens; two vision features to splice in.
        let features = MLXRandom.normal([1, 2, 64])
        let embeddings = net.fusedEmbeddings(inputIds: [3, 7, 7, 5], imageFeatures: features)
        eval(embeddings)
        XCTAssertEqual(embeddings.shape, [1, 4, 64])
        // The image positions carry the (flattened) features, not the token embedding.
        let spliced = embeddings[0, 1].asArray(Float.self)
        let expected = features.reshaped([-1, 64])[0].asArray(Float.self)
        XCTAssertEqual(spliced, expected, "an image position reads its vision feature")
    }

    private static func solidImage(width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 128, count: width * height * 4)
        let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return context.makeImage()!
    }
}
