//
//  NFKMLXSigLIP2Tests.swift
//  InferKitMLXTests
//
//  SigLIP 2. Embeddings evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild`.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSigLIP2Tests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func net() -> NFKMLXSigLIP2Net { NFKMLXSigLIP2.makeNet(.tiny) }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(net().parameters().flattened().map(\.0))
        for expected in ["vision.embeddings.patch_embedding.weight", "vision.encoder.layers.0.self_attn.q_proj.weight",
                         "vision.post_layernorm.weight", "vision.head.probe", "vision.head.attention.in_proj_weight",
                         "vision.head.mlp.fc1.weight", "text.embeddings.token_embedding.weight",
                         "text.encoder.layers.0.mlp.fc1.weight", "text.final_layer_norm.weight", "text.head.weight",
                         "logit_scale", "logit_bias"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testImageEmbeddingIsUnitLength() throws {
        try requireMLXRuntime()
        let c = NFKMLXSigLIP2Configuration.tiny
        let pixels = MLXRandom.uniform(low: -1, high: 1, [1, c.vision.imageSize, c.vision.imageSize, 3])
        let embedding = net().imageEmbedding(pixels)
        XCTAssertEqual(embedding.shape, [1, c.vision.hiddenSize], "one pooled embedding per image")
        let norm = sqrt(embedding.square().sum().item(Float.self))
        XCTAssertEqual(norm, 1, accuracy: 1e-4, "the image embedding is L2-normalized")
    }

    func testTextEmbeddingShapeAndUnitLength() throws {
        try requireMLXRuntime()
        let c = NFKMLXSigLIP2Configuration.tiny
        let tokens = MLXArray((0 ..< 2 * c.text.maxPositions).map { Int32($0 % c.text.vocabularySize) },
                              [2, c.text.maxPositions])
        let embedding = net().textEmbedding(tokens)
        XCTAssertEqual(embedding.shape, [2, c.text.hiddenSize], "one embedding per caption")
        eval(embedding)
        let first = embedding[0]
        XCTAssertEqual(sqrt(first.square().sum().item(Float.self)), 1, accuracy: 1e-4)
    }

    func testLogitsAreOnePerTextAgainstTheImage() throws {
        try requireMLXRuntime()
        let c = NFKMLXSigLIP2Configuration.tiny
        let pixels = MLXRandom.uniform(low: -1, high: 1, [1, c.vision.imageSize, c.vision.imageSize, 3])
        let tokens = MLXArray((0 ..< 3 * c.text.maxPositions).map { Int32($0 % c.text.vocabularySize) },
                              [3, c.text.maxPositions])
        let logits = net().logits(image: pixels, text: tokens)
        XCTAssertEqual(logits.reshaped([-1]).shape, [3], "one sigmoid logit per caption")
    }

    func testTheBackendReturnsAnEmbedding() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXSigLIP2.backend(weightsURL: nil)
        let image = Self.solidImage(side: 224)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
        XCTAssertNotNil(result.output(forKey: NFKOutputEmbedding))
    }

    static func solidImage(side: Int) -> CGImage {
        var bytes = [UInt8](repeating: 128, count: side * side * 4)
        for i in stride(from: 3, to: bytes.count, by: 4) { bytes[i] = 255 }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: side * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }
}
