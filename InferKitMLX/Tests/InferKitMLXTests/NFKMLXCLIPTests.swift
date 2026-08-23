//
//  NFKMLXCLIPTests.swift
//  InferKitMLXTests
//
//  Parameter names match the reference OpenAI CLIP without a GPU. The forwards and the weight
//  round-trip evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXCLIPTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXCLIPNet {
        NFKMLXCLIPNet(.tiny)
    }

    // MARK: Parameter names

    func testParameterNamesMatchTheReferenceCheckpoint() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["visual.conv1.weight", "visual.class_embedding", "visual.positional_embedding",
                         "visual.transformer.resblocks.0.attn.in_proj_weight",
                         "visual.transformer.resblocks.0.attn.out_proj.weight",
                         "visual.transformer.resblocks.0.mlp.c_fc.weight", "visual.proj",
                         "token_embedding.weight", "positional_embedding",
                         "transformer.resblocks.0.ln_1.weight", "ln_final.weight",
                         "text_projection", "logit_scale"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // MARK: Forwards (needs MLX)

    func testImageEmbeddingIsUnitLengthAndDeterministic() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let image = Self.image(height: 20, width: 24)
        let a = net.encodeImage(image)
        let b = net.encodeImage(image)
        eval(a, b)
        XCTAssertEqual(a.shape, [NFKMLXCLIPConfiguration.tiny.embedDimensions])
        XCTAssertEqual(Self.norm(a), 1, accuracy: 1e-4, "L2-normalized")
        XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self), "deterministic")
    }

    func testTextEmbeddingIsUnitLengthInTheSharedSpace() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let embedding = net.encodeText([1, 8, 5, 63, 2])                    // 63 is the highest id → the EOT position
        eval(embedding)
        XCTAssertEqual(embedding.shape, [NFKMLXCLIPConfiguration.tiny.embedDimensions],
                       "text and image embeddings share a dimension")
        XCTAssertEqual(Self.norm(embedding), 1, accuracy: 1e-4, "L2-normalized")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()

        // Save in PyTorch layout: only the patch-embedding convolution is 4-D and transposes.
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("clip-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = tinyNet()
        try NFKMLXCLIP.loadWeights(into: loaded, from: url)

        let image = Self.image(height: 16, width: 16)
        let expected = trained.encodeImage(image)
        let actual = loaded.encodeImage(image)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendEmbedsACGImage() throws {
        try requireMLXRuntime()
        NFKMLXCLIP.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXCLIP.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXCLIP.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(8, 8)]))
        let embedding = try XCTUnwrap(result.embedding, "embedding output present")
        XCTAssertEqual(embedding.count, NFKMLXCLIPConfiguration.base.embedDimensions, "ViT-B/32 embedding width")
    }

    // MARK: Helpers

    static func norm(_ x: MLXArray) -> Float {
        sqrtf(x.asArray(Float.self).reduce(0) { $0 + $1 * $1 })
    }

    static func image(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 37) % 256) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [height, width, 3]) }
    }

    static func solid(_ width: Int, _ height: Int) -> CGImage {
        let pixels = [UInt8](repeating: 128, count: width * height * 4)
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }
}
