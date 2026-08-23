//
//  NFKMLXSwinIRTests.swift
//  InferKitMLXTests
//
//  The Swin Transformer SR network. Window partition/reverse and the relative-position index are pure
//  Swift; the forward and weight round-trip evaluate MLX arrays and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSwinIRTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXSwinIRNet {
        NFKMLXSwinIRNet(.tiny)
    }

    // MARK: Upsampler geometry

    // The reference `Upsample` builds a power-of-two scale from repeated ×2 stages and a scale of
    // three from ONE ×3 stage. A factor-3 shuffle is not a pair of ×2 shuffles, so the stage count and
    // the per-stage channel expansion both change with the scale.
    func testTheUpsamplerMatchesTheScaleItIsBuiltFor() {
        let cases: [(scale: Int, stages: Int, factor: Int)] = [
            (2, 1, 2), (4, 2, 2), (8, 3, 2), (3, 1, 3),
        ]
        for (scale, stages, factor) in cases {
            var configuration = NFKMLXSwinIRConfiguration.tiny
            configuration.scale = scale
            XCTAssertTrue(configuration.isSupportedScale, "scale \(scale)")
            XCTAssertEqual(configuration.upsampleStages, stages, "scale \(scale) stage count")
            XCTAssertEqual(configuration.shuffleFactor, factor, "scale \(scale) shuffle factor")
        }
    }

    func testAnUnsupportedScaleIsRejectedRatherThanBuiltWrong() {
        var configuration = NFKMLXSwinIRConfiguration.tiny
        configuration.scale = 5
        XCTAssertFalse(configuration.isSupportedScale)
        XCTAssertThrowsError(try NFKMLXSwinIR.makeNet(configuration)) { error in
            guard case NFKMLXError.unsupportedConfiguration = error else {
                return XCTFail("expected unsupportedConfiguration, got \(error)")
            }
        }
    }

    func testAScaleOfThreeUpscalesByThree() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXSwinIRConfiguration.tiny
        configuration.scale = 3
        let net = try NFKMLXSwinIR.makeNet(configuration)

        let side = configuration.windowSize * 2
        let output = net.upscale(MLXArray.zeros([side, side, 3]) + 0.5)
        eval(output)
        XCTAssertEqual(output.shape, [side * 3, side * 3, 3])
    }

    // A ×3 upsampler packs `9·C` channels into one stage; a ×4 packs `4·C` into each of two. Loading
    // one into the other would mis-shape, so the parameter tree has to say which it is.
    func testTheScaleThreeUpsamplerHasOneStageOfNineChannels() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXSwinIRConfiguration.tiny
        configuration.scale = 3
        let names = try NFKMLXSwinIR.makeNet(configuration).parameters().flattened().map(\.0)
        XCTAssertTrue(names.contains("upsample.0.weight"))
        XCTAssertFalse(names.contains("upsample.1.weight"), "a ×3 scale is a single stage")

        let weight = try XCTUnwrap(try NFKMLXSwinIR.makeNet(configuration).parameters().flattened()
            .first { $0.0 == "upsample.0.weight" }?.1)
        XCTAssertEqual(weight.shape[0], configuration.embedDimensions * 9, "one stage packs 9·C channels")
    }

    // MARK: Pure-Swift window helpers (no GPU)

    func testRelativePositionIndexCoversTheBiasTable() {
        let ws = 4
        let index = NFKSwinOps.relativePositionIndex(windowSize: ws)
        XCTAssertEqual(index.count, ws * ws * ws * ws)
        XCTAssertGreaterThanOrEqual(index.min() ?? -1, 0)
        XCTAssertLessThan(Int(index.max() ?? 0), (2 * ws - 1) * (2 * ws - 1), "index stays inside the bias table")
    }

    // MARK: Parameter names

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["conv_first.weight", "layers.0.blocks.0.attn.qkv.weight",
                         "layers.0.blocks.1.attn.relative_position_bias_table", "layers.0.conv.weight",
                         "conv_after_body.weight", "conv_before_upsample.weight",
                         "upsample.0.weight", "conv_last.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // MARK: Forward and weight loading (needs MLX)

    func testUpscalingByTheScaleFactorStaysInRangeAndIsDeterministic() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let input = Self.image(side: 8)                          // a multiple of the window size (4)
        let a = net.upscale(input)
        let b = net.upscale(input)
        eval(a, b)
        XCTAssertEqual(a.shape, [16, 16, 3], "×2 upscale")
        let values = a.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0, "clipped to 0...1")
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
        XCTAssertEqual(values, b.asArray(Float.self), "deterministic")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("swinir-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = tinyNet()
        try NFKMLXSwinIR.loadWeights(into: loaded, from: url)

        let input = Self.image(side: 8)
        let expected = trained.upscale(input)
        let actual = loaded.upscale(input)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendUpscalesACGImage() throws {
        try requireMLXRuntime()
        NFKMLXSwinIR.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXSwinIR.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXSwinIR.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(8, 8)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 32, "lightweight config is ×4 (8 → 32)")
    }

    // MARK: Helpers

    static func image(side: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: side * side * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 37) % 256) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [side, side, 3]) }
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
