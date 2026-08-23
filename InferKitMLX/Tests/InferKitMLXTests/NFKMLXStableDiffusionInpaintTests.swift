//
//  NFKMLXStableDiffusionInpaintTests.swift
//  InferKitMLXTests
//
//  A latent-diffusion inpaint pipeline (VAE + UNet) on the diffusion backend. A tiny configuration
//  keeps the DDIM loop cheap. These evaluate MLX arrays, so they skip under `swift test` and run
//  under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXStableDiffusionInpaintTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    /// The released structure at a size a test can run: the same nine-channel input, blocks, and
    /// attention over two narrow levels. Widths stay multiples of the normalization's 32 groups.
    private func tinyConfiguration() -> NFKMLXSDInpaintConfiguration {
        var configuration = NFKMLXSDInpaintConfiguration.stableDiffusion15
        configuration.unet.blockChannels = [32, 64]
        configuration.unet.attends = [true, false]
        configuration.unet.attentionHeads = [2, 2]
        configuration.unet.onlyCrossAttention = [false, false]
        configuration.vae.blockChannels = [32, 64]
        configuration.steps = 2
        return configuration
    }

    func testTheInpaintPipelineRunsThroughTheDiffusionBackend() throws {
        try requireMLXRuntime()
        let backend = NFKMLXStableDiffusionInpaint.makeModel(tinyConfiguration()).makeBackend()
        let request = NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(16, 16, 180, 90, 40),
                                                   NFKInputMask: Self.leftHalfMask(16, 16)])
        let result = try backend.runInference(for: request)
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 16, "VAE encode ÷4 then decode ×4 returns to input size")
        XCTAssertEqual(output.height, 16)
    }

    func testCheckpointRoundTripReproducesTheDenoiseAndDecode() throws {
        try requireMLXRuntime()
        let trained = NFKMLXStableDiffusionInpaint.makeModel(tinyConfiguration())
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.pipeline.parameters().flattened()
            .map { key, value in (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value) })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sdinpaint-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXStableDiffusionInpaint.makeModel(tinyConfiguration())
        try loaded.pipeline.loadWeights(from: url)

        // VAE decode round-trip.
        let latent = Self.tensor([4, 4, 4], seed: 3)
        let decodedExpected = trained.decode(latent)
        let decodedActual = loaded.decode(latent)
        eval(decodedExpected, decodedActual)
        XCTAssertEqual(decodedExpected.asArray(Float.self), decodedActual.asArray(Float.self), "VAE decode")

        // UNet denoise round-trip.
        let context = NFKDiffusionContext(
            conditioning: ["masked": Self.tensor([4, 4, 4], seed: 5), "maskLatent": Self.tensor([4, 4, 1], seed: 6)],
            width: 4, height: 4)
        let step = NFKDDIMScheduler().steps(2)[0]
        let denoiseExpected = trained.denoise(latent, step, context)
        let denoiseActual = loaded.denoise(latent, step, context)
        eval(denoiseExpected, denoiseActual)
        XCTAssertEqual(denoiseExpected.asArray(Float.self), denoiseActual.asArray(Float.self), "UNet denoise")
    }

    static func tensor(_ shape: [Int], seed: Int) -> MLXArray {
        let count = shape.reduce(1, *)
        var values = [Float](repeating: 0, count: count)
        for i in 0 ..< count {
            values[i] = Float(((i + seed) * 47) % 256) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, shape) }
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

    static func leftHalfMask(_ width: Int, _ height: Int) -> CGImage {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for row in 0 ..< height {
            for col in 0 ..< width {
                let value: UInt8 = col < width / 2 ? 255 : 0
                let base = (row * width + col) * 4
                pixels[base] = value; pixels[base + 1] = value; pixels[base + 2] = value; pixels[base + 3] = 255
            }
        }
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
