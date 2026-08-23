//
//  NFKMLXLatentDiffusionModelsTests.swift
//  InferKitMLXTests
//
//  Marigold depth and the SD ×4 latent upscaler on the diffusion backend, at a tiny configuration.
//  These evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXLatentDiffusionModelsTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    /// The released structure at a size a test can run: the same blocks, attention, and level count,
    /// over two narrow levels. Channel widths stay multiples of the normalization's 32 groups.
    static func tinyUNet(_ base: NFKMLXSDUNetConfiguration) -> NFKMLXSDUNetConfiguration {
        var configuration = base
        configuration.blockChannels = [32, 64]
        configuration.attends = Array(base.attends.prefix(2))
        configuration.attentionHeads = [2, 2]
        configuration.onlyCrossAttention = Array(base.onlyCrossAttention.prefix(2))
        return configuration
    }

    /// Narrows the autoencoder but keeps its LEVEL COUNT, because that count is the spatial ratio the
    /// model is defined by — dropping one would silently turn the ×4 upscaler into a ×2.
    static func tinyVAE(_ base: NFKMLXSDVAEConfiguration) -> NFKMLXSDVAEConfiguration {
        var configuration = base
        configuration.blockChannels = base.blockChannels.indices.map { $0 == 0 ? 32 : 64 }
        return configuration
    }

    func testMarigoldProducesADepthImageAtInputSize() throws {
        try requireMLXRuntime()
        let backend = Self.marigold().makeBackend()
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(16, 16, 100, 140, 60)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 16, "depth returns at the input size")
        XCTAssertEqual(output.height, 16)
    }

    func testMarigoldCheckpointRoundTripReproducesTheOutput() throws {
        try requireMLXRuntime()
        let trained = Self.marigold()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.pipeline.parameters().flattened()
            .map { key, value in (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value) })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("marigold-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = Self.marigold()
        try loaded.pipeline.loadWeights(from: url)

        let request = NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(16, 16, 100, 140, 60)])
        let a = Self.pixels(of: try Self.cgImage(try trained.makeBackend().runInference(for: request).output(forKey: NFKOutputImage)))
        let b = Self.pixels(of: try Self.cgImage(try loaded.makeBackend().runInference(for: request).output(forKey: NFKOutputImage)))
        XCTAssertEqual(a, b, "loaded weights reproduce the depth output")
    }

    func testUpscalerProducesAFourTimesImage() throws {
        try requireMLXRuntime()
        let backend = Self.upscaler().makeBackend()
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(16, 16, 200, 100, 50)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 64, "×4 upscale of the 16-px input")
        XCTAssertEqual(output.height, 64)
    }

    static func marigold() -> NFKMLXMarigoldModel {
        NFKMLXMarigoldModel(unet: tinyUNet(.marigold), vae: tinyVAE(.stableDiffusion), steps: 2)
    }

    static func upscaler() -> NFKMLXSDUpscalerModel {
        NFKMLXSDUpscalerModel(unet: tinyUNet(.upscaler), vae: tinyVAE(.upscaler), steps: 2,
                              noiseLevel: 20)
    }

    static func solid(_ width: Int, _ height: Int, _ red: UInt8, _ green: UInt8, _ blue: UInt8) -> CGImage {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for pixel in 0 ..< (width * height) {
            pixels[pixel * 4] = red; pixels[pixel * 4 + 1] = green
            pixels[pixel * 4 + 2] = blue; pixels[pixel * 4 + 3] = 255
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    static func cgImage(_ value: Any?) throws -> CGImage {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else { throw NFKMLXError.noOutput }
        return (value as! CGImage)
    }

    static func pixels(of image: CGImage) -> [UInt8] { [UInt8](image.dataProvider!.data! as Data) }
}
