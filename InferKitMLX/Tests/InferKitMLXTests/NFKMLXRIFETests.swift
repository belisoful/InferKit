//
//  NFKMLXRIFETests.swift
//  InferKitMLXTests
//
//  IFNet + a bilinear backward warp. These evaluate MLX arrays, so they skip under `swift test` and
//  run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXRIFETests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    func testWarpWithZeroFlowIsIdentity() throws {
        try requireMLXRuntime()
        let img = Self.gradient(height: 4, width: 4)
        let flow = MLXArray.zeros([1, 4, 4, 2])
        let warped = NFKMLXRIFENet.warp(img, flow: flow)
        eval(warped)
        XCTAssertEqual(warped.asArray(Float.self), img.asArray(Float.self), "zero flow returns the image")
    }

    func testWarpShiftsSamplingByTheFlow() throws {
        try requireMLXRuntime()
        // A horizontal ramp: value increases with x. A +1 x-flow samples one column to the right.
        let img = Self.gradient(height: 1, width: 4)
        var flowValues = [Float](repeating: 0, count: 1 * 1 * 4 * 2)
        for x in 0 ..< 4 { flowValues[(x) * 2] = 1 }              // flow_x = +1 everywhere
        let flow = flowValues.withUnsafeBufferPointer { MLXArray($0, [1, 1, 4, 2]) }
        let warped = NFKMLXRIFENet.warp(img, flow: flow)
        eval(warped)
        let out = warped.asArray(Float.self)
        let src = img.asArray(Float.self)
        XCTAssertEqual(out[0], src[3], accuracy: 1e-6, "pixel 0 sampled column 1's value")   // R channel
    }

    func testInterpolateProducesAMiddleFrameAtInputSize() throws {
        try requireMLXRuntime()
        let net = NFKMLXRIFE.makeNet()
        let middle = net.interpolate(Self.gradient(height: 20, width: 24), Self.gradient(height: 20, width: 24))
        eval(middle)
        XCTAssertEqual(middle.shape, [1, 20, 24, 3], "batched in, padded to 32, cropped back to the input size")
        let values = middle.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0)
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
    }

    func testACheckpointRoundTripReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = NFKMLXRIFE.makeNet()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            // The flow and mask heads are transposed convolutions, stored `[in, out, kH, kW]`
            // upstream rather than the forward convolutions' `[out, in, kH, kW]`.
            (key, value.ndim == 4
                ? (key.contains(".up1.") || key.contains(".up2.")
                   ? value.transposed(3, 0, 1, 2) : value.transposed(0, 3, 1, 2))
                : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("rife-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXRIFE.makeNet()
        try NFKMLXRIFE.loadWeights(into: loaded, from: url)

        let (a, b) = (Self.gradient(height: 16, width: 16), Self.gradient(height: 16, width: 16))
        let expected = trained.interpolate(a, b)
        let actual = loaded.interpolate(a, b)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    func testTheRegisteredBackendInterpolatesTwoFrames() throws {
        try requireMLXRuntime()
        NFKMLXRIFE.register()
        let backend = try NFKMLXModelRegistry.backend(named: "rife", weightsURL: nil)
        let request = NFKInferenceRequest(inputs: ["frame0": Self.solid(16, 16, 40, 80, 120),
                                                   "frame1": Self.solid(16, 16, 200, 160, 120)])
        let output = try Self.cgImage(try backend.runInference(for: request).output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 16)
        XCTAssertEqual(output.height, 16)
    }

    // MARK: v4

    func testV4ParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXRIFEv4.makeNet().parameters().flattened().map(\.0))
        for expected in ["encode.cnn0.weight", "encode.cnn3.weight",
                         "block0.conv0.0.conv.weight", "block0.convblock.0.conv.weight",
                         "block0.convblock.0.beta", "block0.lastconv.weight", "block3.lastconv.bias"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
        // v4 activates with a parameter-free leaky ReLU where HDv3 used a PReLU.
        XCTAssertFalse(names.contains { $0.contains("conv0.0.prelu") }, "no PReLU weights in v4")
    }

    func testTheV4RemapNamesTheSequentialEntries() {
        XCTAssertEqual(NFKMLXRIFEv4.remapReferenceKey("block0.conv0.0.0.weight"), "block0.conv0.0.conv.weight")
        XCTAssertEqual(NFKMLXRIFEv4.remapReferenceKey("block2.lastconv.0.bias"), "block2.lastconv.bias")
        XCTAssertEqual(NFKMLXRIFEv4.remapReferenceKey("block0.convblock.3.beta"), "block0.convblock.3.beta")
        XCTAssertEqual(NFKMLXRIFEv4.remapReferenceKey("encode.cnn0.weight"), "encode.cnn0.weight")
    }

    func testV4InterpolatesAtAnArbitraryTimestep() throws {
        try requireMLXRuntime()
        // The timestep is what v4 adds over the earlier generations: it can land anywhere between the
        // frames, not only halfway.
        let net = NFKMLXRIFEv4.makeNet()
        let a = Self.gradient(height: 64, width: 64)
        let b = Self.gradient(height: 64, width: 64)
        let quarter = net.interpolate(a, b, timestep: 0.25)
        let middle = net.interpolate(a, b, timestep: 0.5)
        eval(quarter, middle)
        XCTAssertEqual(quarter.shape, [1, 64, 64, 3])
        XCTAssertNotEqual(quarter.asArray(Float.self), middle.asArray(Float.self),
                          "a different timestep produces a different frame")
    }

    static func gradient(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let base = (y * width + x) * 3
                values[base] = Float(x) / Float(max(width - 1, 1))
                values[base + 1] = Float(y) / Float(max(height - 1, 1))
                values[base + 2] = 0.5
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, height, width, 3]) }
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
}
