//
//  NFKMLXDiffusionExtrasTests.swift
//  InferKitMLXTests
//
//  The LCM scheduler and the ControlNet reference pipeline. These evaluate MLX arrays, so they skip
//  under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDiffusionExtrasTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    // MARK: LCM scheduler

    func testLCMScheduleIsHighestNoiseFirst() throws {
        try requireMLXRuntime()
        let steps = NFKLCMScheduler().steps(6)
        XCTAssertEqual(steps.count, 6)
        XCTAssertEqual(steps.first?.index, 0)
        XCTAssertLessThan(steps.first!.alphaBar, steps.last!.alphaBar, "noise decreases along the schedule")
    }

    func testLCMStepDiffersFromDDIM() throws {
        try requireMLXRuntime()
        let ddim = NFKDDIMScheduler(predictionType: .epsilon)
        let lcm = NFKLCMScheduler(predictionType: .epsilon)
        let timestep = ddim.steps(4)[1]                          // a mid-noise (non-final) step
        let latent = Self.pattern([1, 4, 4, 4], scale: 0.6)
        let prediction = Self.pattern([1, 4, 4, 4], scale: 0.2)

        let ddimNext = ddim.step(prediction: prediction, timestep: timestep, latent: latent)
        let lcmNext = lcm.step(prediction: prediction, timestep: timestep, latent: latent)
        eval(ddimNext, lcmNext)
        XCTAssertNotEqual(ddimNext.asArray(Float.self), lcmNext.asArray(Float.self),
                          "LCM re-noises with fresh noise and applies the consistency boundary")
    }

    func testLCMSamplingIsRepeatableForAFixedSeed() throws {
        try requireMLXRuntime()
        let timestep = NFKDDIMScheduler().steps(4)[1]
        let latent = Self.pattern([1, 4, 4, 4], scale: 0.5)
        let prediction = Self.pattern([1, 4, 4, 4], scale: 0.3)
        let a = NFKLCMScheduler(seed: 7).step(prediction: prediction, timestep: timestep, latent: latent)
        let b = NFKLCMScheduler(seed: 7).step(prediction: prediction, timestep: timestep, latent: latent)
        eval(a, b)
        XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self), "same seed → same noise stream")
    }

    func testLCMLoopDrivesTheLatentTowardTheCleanTarget() throws {
        try requireMLXRuntime()
        let scheduler = NFKLCMScheduler(predictionType: .epsilon)
        let target = MLXArray.zeros([1, 8, 8, 4]) + 0.3
        var latent = Self.pattern([1, 8, 8, 4], scale: 1.0)      // start far from the target
        let initialDistance = mean(abs(latent - target))

        // An oracle epsilon that recovers `target` as the clean latent each step (stands in for a UNet).
        for timestep in scheduler.steps(8) {
            let sqrtBar = sqrtf(timestep.alphaBar)
            let sqrtComplement = sqrtf(1 - timestep.alphaBar)
            let epsilon = (latent - target * sqrtBar) / sqrtComplement
            latent = scheduler.step(prediction: epsilon, timestep: timestep, latent: latent)
        }
        let finalDistance = mean(abs(latent - target))
        eval(initialDistance, finalDistance)
        XCTAssertLessThan(finalDistance.item(Float.self), initialDistance.item(Float.self),
                          "few-step LCM sampling converges toward the clean estimate")
    }

    // MARK: ControlNet reference

    func testControlNetReferenceRunsWithAControlMap() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerControlNet()
        XCTAssertTrue(NFKMLXModelRegistry.registeredModelNames.contains("diffusion-controlnet"))
        let backend = try NFKMLXModelRegistry.backend(named: "diffusion-controlnet", weightsURL: nil)

        // The control map (an edge/depth/pose image) drives generation via conditioning["control"].
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputControl: Self.solid(24, 40)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 24)
        XCTAssertEqual(output.height, 40, "the output takes the control map's size")
    }

    func testControlNetReferenceRequiresAControlMap() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerControlNet()
        let backend = try NFKMLXModelRegistry.backend(named: "diffusion-controlnet", weightsURL: nil)
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(16, 16)])),
                             "without a control map there is nothing to condition on")
    }

    // MARK: Helpers

    static func pattern(_ shape: [Int], scale: Float) -> MLXArray {
        let count = shape.reduce(1, *)
        var values = [Float](repeating: 0, count: count)
        for i in 0 ..< count {
            values[i] = sinf(Float(i) * 0.7) * scale
        }
        return values.withUnsafeBufferPointer { MLXArray($0, shape) }
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
