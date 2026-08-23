//
//  NFKMLXSegFormerTests.swift
//  InferKitMLXTests
//
//  The MiT transformer + MLP-head segmenter. The forward and the weight round-trip evaluate MLX
//  arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSegFormerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXSegFormerNet {
        NFKMLXSegFormerNet(.tiny)
    }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["stage1.patch_embed.proj.weight", "stage1.blocks.0.attn.q.weight",
                         "stage1.blocks.0.attn.sr.weight", "stage1.blocks.0.ffn.dwconv.weight",
                         "stage4.norm.weight", "linear_c.0.weight", "linear_c.3.weight",
                         "linear_fuse.weight", "classifier.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testLogitsAreAtQuarterResolutionWithOneChannelPerClass() throws {
        try requireMLXRuntime()
        let logits = tinyNet().logits(Self.image(height: 32, width: 32).reshaped([1, 32, 32, 3]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 8, 8, NFKMLXSegFormerConfiguration.tiny.classCount],
                       "stage-1 is H/4, one channel per class")
    }

    func testSegmentationIsALabelMapAtInputSize() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let map = net.segment(Self.image(height: 32, width: 32))
        eval(map)
        XCTAssertEqual(map.shape, [32, 32, 1], "label map at input resolution")
        let values = map.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0, "class index scaled to 0...1")
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("segformer-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = tinyNet()
        try NFKMLXSegFormer.loadWeights(into: loaded, from: url)

        let image = Self.image(height: 32, width: 32).reshaped([1, 32, 32, 3])
        let expected = trained.logits(image)
        let actual = loaded.logits(image)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendSegmentsACGImage() throws {
        try requireMLXRuntime()
        NFKMLXSegFormer.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXSegFormer.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXSegFormer.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64, 64)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 64, "label map keeps the input size")
    }

    // MARK: Helpers

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

    static func cgImage(_ value: Any?) throws -> CGImage {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.noOutput
        }
        return (value as! CGImage)
    }
}
