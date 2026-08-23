//
//  NFKMLXBiSeNetTests.swift
//  InferKitMLXTests
//
//  The two-path BiSeNet segmenter. The forward and the weight round-trip evaluate MLX arrays, so they
//  skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXBiSeNetTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXBiSeNetNet {
        let net = NFKMLXBiSeNetNet(.tiny)
        net.train(false)
        return net
    }

    func testParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXBiSeNet.makeNet(.tiny).parameters().flattened().map(\.0))
        for expected in ["sp.conv1.conv.weight", "sp.conv_out.bn.weight",
                         "cp.resnet.conv1.weight", "cp.resnet.layer1.0.conv1.weight",
                         "cp.resnet.layer2.0.downsample_conv.weight",
                         "cp.arm16.conv_atten.weight", "cp.arm32.bn_atten.weight",
                         "cp.conv_head32.conv.weight", "cp.conv_avg.conv.weight",
                         "ffm.convblk.conv.weight", "ffm.conv.weight", "ffm.bn.weight",
                         "conv_out.conv.conv.weight", "conv_out.conv_out.bias",
                         "conv_out16.conv_out.weight", "conv_out32.conv.conv.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheReferenceKeyRemapNamesTheProjectionShortcut() {
        XCTAssertEqual(NFKMLXBiSeNet.remapReferenceKey("cp.resnet.layer2.0.downsample.0.weight"),
                       "cp.resnet.layer2.0.downsample_conv.weight")
        XCTAssertEqual(NFKMLXBiSeNet.remapReferenceKey("cp.resnet.layer3.0.downsample.1.running_mean"),
                       "cp.resnet.layer3.0.downsample_bn.running_mean")
        XCTAssertEqual(NFKMLXBiSeNet.remapReferenceKey("ffm.convblk.conv.weight"), "ffm.convblk.conv.weight")
    }

    func testLogitsComeBackAtInputResolutionWithOneChannelPerClass() throws {
        try requireMLXRuntime()
        // The reference's head upsamples ×8, so the logits leave at the resolution they came in at.
        let net = NFKMLXBiSeNet.makeNet(.tiny)
        let logits = net.logits(Self.image(height: 64, width: 64).reshaped([1, 64, 64, 3]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 64, 64, NFKMLXBiSeNetConfiguration.tiny.classCount],
                       "input resolution, one channel per class")
    }

    func testSegmentationIsALabelMapAtInputSize() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let map = net.segment(Self.image(height: 64, width: 48))
        eval(map)
        XCTAssertEqual(map.shape, [64, 48, 1], "label map at input resolution")
        let values = map.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0)
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let onDisk = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("bisenet-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: onDisk, url: url)

        let loaded = NFKMLXBiSeNetNet(.tiny)
        loaded.train(false)
        try NFKMLXBiSeNet.loadWeights(into: loaded, from: url)

        let image = MLXArray.zeros([1, 64, 64, 3]).asType(.float32) + 0.35
        let expected = trained.logits(image)
        let actual = loaded.logits(image)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendSegmentsACGImage() throws {
        try requireMLXRuntime()
        NFKMLXBiSeNet.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXBiSeNet.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXBiSeNet.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64, 64)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 64, "label map keeps the input size")
    }

    // MARK: BiSeNetV2

    func testV2ParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXBiSeNetV2.makeNet(classCount: 4).parameters().flattened().map(\.0))
        for expected in ["detail_s1.0.conv.weight", "detail_s3.2.bn.weight",
                         "stem.conv.conv.weight", "stem.left1.conv.weight", "stem.fuse.bn.weight",
                         "s3.0.dw1_conv.weight", "s3.0.shortcut_dw_conv.weight", "s3.1.conv2_conv.weight",
                         "s5_5.conv_gap.conv.weight", "bga.left1_dw.weight", "bga.right2_pw.weight",
                         "head_conv.conv.weight", "head_out.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheV2RemapNamesEverySequential() {
        XCTAssertEqual(NFKMLXBiSeNetV2.remapReferenceKey("detail.S1.0.conv.weight"), "detail_s1.0.conv.weight")
        XCTAssertEqual(NFKMLXBiSeNetV2.remapReferenceKey("segment.S1S2.left.1.conv.weight"), "stem.left2.conv.weight")
        XCTAssertEqual(NFKMLXBiSeNetV2.remapReferenceKey("segment.S3.0.dwconv1.0.weight"), "s3.0.dw1_conv.weight")
        XCTAssertEqual(NFKMLXBiSeNetV2.remapReferenceKey("segment.S3.0.shortcut.2.weight"), "s3.0.shortcut_conv.weight")
        XCTAssertEqual(NFKMLXBiSeNetV2.remapReferenceKey("segment.S5_5.conv_gap.conv.weight"), "s5_5.conv_gap.conv.weight")
        XCTAssertEqual(NFKMLXBiSeNetV2.remapReferenceKey("bga.left1.2.weight"), "bga.left1_pw.weight")
        XCTAssertEqual(NFKMLXBiSeNetV2.remapReferenceKey("head.conv_out.0.bias"), "head_out.bias")
    }

    func testTheV2PixelShuffleMatchesTheReferenceChannelOrder() throws {
        try requireMLXRuntime()
        // PyTorch maps channel `c·r² + i·r + j` to output pixel `(c, h·r + i, w·r + j)`; a factor of
        // four is not two ×2 shuffles, so the ordering is asserted directly.
        let values = (0 ..< 16).map { Float($0) }
        let x = values.withUnsafeBufferPointer { MLXArray($0, [1, 1, 1, 16]) }
        let shuffled = NFKMLXPixelShuffle.apply(x, factor: 4)
        eval(shuffled)
        XCTAssertEqual(shuffled.shape, [1, 4, 4, 1])
        XCTAssertEqual(shuffled.reshaped([-1]).asArray(Float.self), values,
                       "a single channel group unrolls in row-major order")
    }

    func testV2SegmentationIsALabelMapAtInputSize() throws {
        try requireMLXRuntime()
        let net = NFKMLXBiSeNetV2.makeNet(classCount: 4)
        net.train(false)
        let map = net.segment(Self.image(height: 64, width: 48))
        eval(map)
        XCTAssertEqual(map.shape, [64, 48, 1], "label map at input resolution")
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
