//
//  NFKMLXDiffusionBackendTests.swift
//  InferKitMLXTests
//
//  The scheduler math and the contract need no GPU. The sampling round-trips evaluate MLX arrays,
//  which needs the MLX Metal library the Xcode build system bundles but `swift test` does not, so
//  those skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDiffusionBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    // MARK: Scheduler (no MLX)

    func testDDIMScheduleRisesInSignalAndEndsClean() {
        let schedule = NFKDDIMScheduler().steps(12)
        XCTAssertEqual(schedule.count, 12)
        XCTAssertLessThan(schedule.first!.alphaBar, schedule.last!.alphaBar, "noise falls, so alphaBar rises")
        for i in 1 ..< schedule.count {
            XCTAssertGreaterThan(schedule[i].alphaBar, schedule[i - 1].alphaBar)
        }
        // The released Stable Diffusion schedulers set `set_alpha_to_one: false`, so the last step
        // denoises against the ratio at training step 0 rather than a perfectly clean 1.
        XCTAssertEqual(schedule.last!.alphaBarPrev, 0.99915, accuracy: 1e-4,
                       "the final step lands on the reference's own final signal ratio")
        XCTAssertEqual(NFKDDIMScheduler(setsAlphaToOne: true).steps(12).last!.alphaBarPrev, 1,
                       "asking for a clean landing gives one")
    }

    // The reference walks a fixed stride and adds `steps_offset`; it does not divide the training range
    // evenly, and the difference moves every step of the loop.
    func testDDIMScheduleVisitsTheReferenceTrainingSteps() {
        XCTAssertEqual(NFKDDIMScheduler().steps(20).map(\.train),
                       [951, 901, 851, 801, 751, 701, 651, 601, 551, 501,
                        451, 401, 351, 301, 251, 201, 151, 101, 51, 1])
        XCTAssertEqual(NFKDDIMScheduler(stepsOffset: 0).steps(4).map(\.train), [750, 500, 250, 0])
    }

    func testDDIMPrevSignalChainsToTheNextStep() {
        let schedule = NFKDDIMScheduler().steps(8)
        for i in 0 ..< schedule.count - 1 {
            XCTAssertEqual(schedule[i].alphaBarPrev, schedule[i + 1].alphaBar, accuracy: 1e-6)
        }
    }

    func testStartIndexSkipsEarlyStepsForLowerStrength() {
        XCTAssertEqual(NFKMLXDiffusionBackend.startIndex(forStrength: 1, count: 10), 0)
        XCTAssertEqual(NFKMLXDiffusionBackend.startIndex(forStrength: 0.5, count: 10), 5)
        XCTAssertEqual(NFKMLXDiffusionBackend.startIndex(forStrength: 0, count: 10), 9)
    }

    // MARK: Contract (no MLX)

    func testBackendReportsTheSuppliedIdentityAndReadiness() {
        let backend = NFKMLXDiffusionBackend(identifier: "upscaler", isReady: false,
                                             encode: { _, _, _ in NFKDiffusionContext(width: 1, height: 1) },
                                             denoise: { latent, _, _, _ in latent })
        XCTAssertEqual(backend.backendIdentifier, "upscaler")
        XCTAssertFalse(backend.isReady)
    }

    func testAnInferenceWhoseEncodeNeedsAnImageFailsWithoutOne() {
        let backend = NFKMLXDiffusionBackend(
            encode: { _, image, _ in
                guard image != nil else { throw NFKMLXError.unsupportedInput }
                return NFKDiffusionContext(width: 1, height: 1)
            },
            denoise: { latent, _, _, _ in latent })
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "no image"])))
    }

    // MARK: Sampling (needs MLX)

    func testNoiseIsDeterministicForASeedAndVariesAcrossSeeds() throws {
        try requireMLXRuntime()
        let a = NFKMLXDiffusionBackend.gaussianNoise(height: 4, width: 4, channels: 3, seed: 7)
        let b = NFKMLXDiffusionBackend.gaussianNoise(height: 4, width: 4, channels: 3, seed: 7)
        let c = NFKMLXDiffusionBackend.gaussianNoise(height: 4, width: 4, channels: 3, seed: 8)
        eval(a, b, c)
        XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self))
        XCTAssertNotEqual(a.asArray(Float.self), c.asArray(Float.self))
    }

    func testTheOracleLoopConvergesToItsTarget() throws {
        try requireMLXRuntime()
        let backend = NFKMLXDiffusionBackend(
            configuration: NFKDiffusionConfiguration(steps: 6),
            // The oracle drives the loop to an exact target, which only lands exactly when the final
            // step denoises fully — the released models leave a residual there by design.
            scheduler: NFKDDIMScheduler(predictionType: .epsilon, setsAlphaToOne: true),
            encode: { _, image, _ in
                let target = image! * 0 + 0.25                  // constant 0.25 everywhere
                return NFKDiffusionContext(conditioning: ["target": target],
                                           width: image!.shape[1], height: image!.shape[0])
            },
            denoise: NFKMLXReferenceModels.oracleDenoise,
            decode: { clip($0, min: 0, max: 1) })
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(2, 2, 10, 20, 30)]))
        let pixels = Self.pixels(of: try Self.cgImage(result.output(forKey: NFKOutputImage)))
        XCTAssertEqual(Int(pixels[0]), 64, accuracy: 2, "0.25 -> ~64")
        XCTAssertEqual(Int(pixels[1]), 64, accuracy: 2)
        XCTAssertEqual(Int(pixels[2]), 64, accuracy: 2)
    }

    func testUpscalerDoublesTheSizeAndPreservesColor() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerDiffusionUpscaler()
        let backend = try NFKMLXModelRegistry.backend(named: "diffusion-upscaler", weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(2, 2, 200, 100, 50)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 4)
        XCTAssertEqual(output.height, 4)
        let pixels = Self.pixels(of: output)
        XCTAssertEqual(Array(pixels[0 ..< 3]), [200, 100, 50], "upscaled block keeps the source color")
    }

    func testDepthOutputsGrayscaleFromLuminance() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerDiffusionDepth()
        let backend = try NFKMLXModelRegistry.backend(named: "diffusion-depth", weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(2, 2, 255, 0, 0)]))
        let pixels = Self.pixels(of: try Self.cgImage(result.output(forKey: NFKOutputImage)))
        XCTAssertEqual(pixels[0], pixels[1], "gray: R == G")
        XCTAssertEqual(pixels[1], pixels[2], "gray: G == B")
        XCTAssertEqual(Int(pixels[0]), 76, accuracy: 2, "0.299 luminance of pure red -> ~76")
    }

    func testInpaintKeepsTheUnmaskedRegionAndFillsTheMasked() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerDiffusionInpainter()
        let backend = try NFKMLXModelRegistry.backend(named: "diffusion-inpaint", weightsURL: nil)
        let plate = Self.solid(2, 1, 255, 0, 0)                  // two red pixels
        let mask = Self.pixelsImage(2, 1, [[255, 255, 255, 255], [0, 0, 0, 255]])   // regenerate left, keep right
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate, NFKInputMask: mask]))
        let pixels = Self.pixels(of: try Self.cgImage(result.output(forKey: NFKOutputImage)))
        XCTAssertEqual(Int(pixels[0]), 128, accuracy: 2, "masked pixel filled to 0.5 gray")
        XCTAssertEqual(Array(pixels[4 ..< 7]), [255, 0, 0], "kept pixel equals the source")
    }

    // MARK: Helpers

    static func solid(_ width: Int, _ height: Int, _ red: UInt8, _ green: UInt8, _ blue: UInt8) -> CGImage {
        pixelsImage(width, height, Array(repeating: [red, green, blue, 255], count: width * height))
    }

    static func pixelsImage(_ width: Int, _ height: Int, _ rgba: [[UInt8]]) -> CGImage {
        let flat = rgba.flatMap { $0 }
        let provider = CGDataProvider(data: Data(flat) as CFData)!
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

    static func pixels(of image: CGImage) -> [UInt8] {
        [UInt8](image.dataProvider!.data! as Data)
    }
}
