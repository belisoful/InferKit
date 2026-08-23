//
//  NFKMLXVideoSRTests.swift
//  InferKitMLXTests
//
//  The BasicVSR video super-resolution network. The forward, the bidirectional propagation, the flow
//  warp, and the weight round-trip evaluate MLX arrays, so they skip under `swift test` and run under
//  `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXVideoSRTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXVideoSRNet {
        NFKMLXVideoSRNet(.tiny)
    }

    func testParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["spynet.basic_module.0.basic_module.0.conv.weight", "spynet.mean", "spynet.std",
                         "backward_resblocks.conv_in.weight", "backward_resblocks.blocks.0.conv1.weight",
                         "forward_resblocks.blocks.0.conv2.bias", "fusion.weight",
                         "upsample1.upsample_conv.weight", "upsample2.upsample_conv.weight",
                         "conv_hr.weight", "conv_last.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheReferenceKeyRemapStripsTheGeneratorAndTheSequential() {
        XCTAssertEqual(NFKMLXVideoSR.remapReferenceKey("generator.backward_resblocks.main.0.weight"),
                       "backward_resblocks.conv_in.weight")
        XCTAssertEqual(NFKMLXVideoSR.remapReferenceKey("generator.forward_resblocks.main.2.14.conv1.bias"),
                       "forward_resblocks.blocks.14.conv1.bias")
        XCTAssertEqual(NFKMLXVideoSR.remapReferenceKey("generator.spynet.basic_module.3.basic_module.2.conv.weight"),
                       "spynet.basic_module.3.basic_module.2.conv.weight")
        XCTAssertEqual(NFKMLXVideoSR.remapReferenceKey("generator.upsample1.upsample_conv.bias"),
                       "upsample1.upsample_conv.bias")
    }

    // MARK: Flow warp

    func testAZeroFlowWarpIsTheIdentity() throws {
        try requireMLXRuntime()
        let image = Self.frame(height: 6, width: 8, seed: 1).reshaped([1, 6, 8, 3])
        let warped = NFKMLXVideoSRNet.flowWarp(image, flow: MLXArray.zeros([1, 6, 8, 2]), zeroOutside: true)
        eval(warped)
        XCTAssertEqual(warped.asArray(Float.self), image.asArray(Float.self))
    }

    func testAUnitFlowShiftsAndThePaddingModesDifferAtTheEdge() throws {
        try requireMLXRuntime()
        let image = Self.frame(height: 4, width: 6, seed: 2).reshaped([1, 4, 6, 3])
        // dx = 1: output[y, x] = input[y, x + 1]; the last column samples outside the map.
        var flowValues = [Float](repeating: 0, count: 4 * 6 * 2)
        for i in stride(from: 0, to: flowValues.count, by: 2) {
            flowValues[i] = 1
        }
        let flow = flowValues.withUnsafeBufferPointer { MLXArray($0, [1, 4, 6, 2]) }

        let zeros = NFKMLXVideoSRNet.flowWarp(image, flow: flow, zeroOutside: true)
        let border = NFKMLXVideoSRNet.flowWarp(image, flow: flow, zeroOutside: false)
        eval(zeros, border)
        let source = image.asArray(Float.self)
        let shifted = zeros.asArray(Float.self)
        // Interior: an exact one-pixel shift.
        XCTAssertEqual(shifted[0], source[3], accuracy: 1e-6)
        // Last column: zeros padding contributes nothing, border padding repeats the edge.
        let lastZeros = zeros[0, 0, 5].asArray(Float.self)
        let lastBorder = border[0, 0, 5].asArray(Float.self)
        XCTAssertEqual(lastZeros, [0, 0, 0])
        XCTAssertEqual(lastBorder, image[0, 0, 5].asArray(Float.self))
    }

    // MARK: Forward and propagation (needs MLX)

    func testUpscalingQuadruplesTheFrame() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let sr = net.upscale(Self.frame(height: 8, width: 12, seed: 1))
        eval(sr)
        XCTAssertEqual(sr.shape, [32, 48, 3], "×4 upscale")
    }

    func testThePropagationIsBidirectional() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let first = Self.frame(height: 8, width: 8, seed: 1)
        let laterA = Self.frame(height: 8, width: 8, seed: 2)
        let laterB = Self.frame(height: 8, width: 8, seed: 3)

        let withA = net.upscaleSequence([first, laterA])[0]
        let withB = net.upscaleSequence([first, laterB])[0]
        eval(withA, withB)
        XCTAssertNotEqual(withA.asArray(Float.self), withB.asArray(Float.self),
                          "changing a LATER frame changes an earlier frame's result — the backward branch is active")
    }

    func testUpscaleSequenceReturnsOneFramePerInput() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let frames = (0 ..< 3).map { Self.frame(height: 8, width: 8, seed: $0) }
        let outputs = net.upscaleSequence(frames)
        XCTAssertEqual(outputs.count, 3)
        outputs.forEach { eval($0); XCTAssertEqual($0.shape, [32, 32, 3]) }
    }

    func testAFrameThatIsNotAMultipleOfTheStrideUpscalesAtFourTimesItsOwnSize() throws {
        try requireMLXRuntime()
        // SPyNet upsizes internally to a multiple of its pyramid depth and rescales the flow back, so
        // an arbitrary frame size has to survive that round trip and the propagation warps after it.
        let net = tinyNet()
        for (height, width) in [(33, 33), (30, 45)] {
            let upscaled = net.upscale(Self.frame(height: height, width: width, seed: 1))
            eval(upscaled)
            XCTAssertEqual(upscaled.shape, [height * 4, width * 4, 3])
        }
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let onDisk = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("videosr-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: onDisk, url: url)

        let loaded = tinyNet()
        try NFKMLXVideoSR.loadWeights(into: loaded, from: url)

        let frame = Self.frame(height: 8, width: 8, seed: 3)
        let expected = trained.upscale(frame)
        let actual = loaded.upscale(frame)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendUpscalesACGImage() throws {
        try requireMLXRuntime()
        NFKMLXVideoSR.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXVideoSR.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXVideoSR.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(8, 8)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 32, "the reference network is ×4 (8 → 32)")
    }

    // MARK: Helpers

    static func frame(height: Int, width: Int, seed: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 37 + seed * 91) % 256) / 255.0
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
