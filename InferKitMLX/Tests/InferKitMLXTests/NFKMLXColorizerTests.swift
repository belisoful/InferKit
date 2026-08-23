//
//  NFKMLXColorizerTests.swift
//  InferKitMLXTests
//
//  The eccv16 colorizer and its Lab color math. Everything evaluates MLX arrays, so the tests skip
//  under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXColorizerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXColorizerNet {
        let net = NFKMLXColorizerNet(.tiny)
        net.train(false)                                        // the production factory path: BN uses running stats
        return net
    }

    // MARK: Lab color math

    func testKnownColorsMapToTheirLabCoordinates() throws {
        try requireMLXRuntime()
        // White, mid gray, black: neutral axis (a = b = 0), reference L* values; black exercises the
        // linear segments of both piecewise curves, white and gray the power segments.
        let rgb = [Float](arrayLiteral: 1, 1, 1, 0.5, 0.5, 0.5, 0, 0, 0)
            .withUnsafeBufferPointer { MLXArray($0, [3, 1, 3]) }
        let lab = NFKLabColor.toLab(rgb)
        eval(lab)
        let values = lab.asArray(Float.self)
        XCTAssertEqual(values[0], 100.0, accuracy: 0.1, "white L*")
        XCTAssertEqual(values[1], 0.0, accuracy: 0.1, "white a*")
        XCTAssertEqual(values[2], 0.0, accuracy: 0.1, "white b*")
        XCTAssertEqual(values[3], 53.39, accuracy: 0.1, "mid-gray L* (sRGB 0.5 → linear 0.214)")
        XCTAssertEqual(values[4], 0.0, accuracy: 0.1)
        XCTAssertEqual(values[6], 0.0, accuracy: 0.1, "black L*")
    }

    func testLabRoundTripIsTheIdentityAcrossTheGamut() throws {
        try requireMLXRuntime()
        // A spread of colors, including saturated primaries and near-black values below the sRGB
        // gamma knee.
        var values = [Float]()
        for r in [0.0, 0.003, 0.25, 0.5, 1.0] {
            for g in [0.0, 0.4, 0.9] {
                for b in [0.02, 0.6, 1.0] {
                    values += [Float(r), Float(g), Float(b)]
                }
            }
        }
        let rgb = values.withUnsafeBufferPointer { MLXArray($0, [values.count / 3, 1, 3]) }
        let back = NFKLabColor.toRGB(NFKLabColor.toLab(rgb))
        eval(back)
        let original = rgb.asArray(Float.self)
        let recovered = back.asArray(Float.self)
        for i in 0 ..< original.count {
            XCTAssertEqual(recovered[i], original[i], accuracy: 0.01, "channel \(i)")
        }
    }

    // MARK: Parameter names

    func testParameterNamesMatchTheConverterOutput() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["conv1_1.weight", "conv1_2.weight", "norm1.weight",
                         "conv3_3.weight", "norm4.bias", "conv5_1.weight", "conv7_3.weight",
                         "deconv8_1.weight", "conv8_2.weight", "conv8_313.weight", "out_ab.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
        XCTAssertFalse(names.contains("out_ab.bias"), "the annealed-mean readout has no bias")
    }

    // MARK: Forward

    func testColorizationKeepsSizeAndRangeAndIsDeterministic() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let input = Self.grayImage(height: 12, width: 20)
        let a = net.colorize(input)
        let b = net.colorize(input)
        eval(a, b)
        XCTAssertEqual(a.shape, [12, 20, 3], "output matches input size")
        let values = a.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0)
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
        XCTAssertEqual(values, b.asArray(Float.self), "deterministic")
    }

    func testTheChromaBinsSumToOneAfterSoftmax() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let resolution = NFKMLXColorizerConfiguration.tiny.resolution
        let l = MLXArray.zeros([1, resolution, resolution, 1])
        let distribution = softmax(net.logits(l), axis: -1)
        eval(distribution)
        let sums = sum(distribution, axis: -1)
        eval(sums)
        for value in sums.asArray(Float.self) {
            XCTAssertEqual(value, 1.0, accuracy: 1e-4, "a probability over the ab bins")
        }
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()

        // Save in the on-disk layout [out, in, kH, kW]. The converter has already swapped the
        // ConvTranspose axes into this same layout, so a single transpose covers every 4-D weight.
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("colorizer-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = tinyNet()
        try NFKMLXColorizer.loadWeights(into: loaded, from: url)

        let input = Self.grayImage(height: 8, width: 8)
        let expected = trained.colorize(input)
        let actual = loaded.colorize(input)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendColorizesACGImage() throws {
        try requireMLXRuntime()
        NFKMLXColorizer.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXColorizer.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXColorizer.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(16, 16)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 16, "colorization keeps the input size")
    }

    // MARK: siggraph17

    func testSiggraphParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXSiggraphColorizer.makeNet().parameters().flattened().map(\.0))
        for expected in ["model1.convs.0.weight", "model1.norm.weight", "model3.convs.2.weight",
                         "model8up.weight", "model3short8.weight", "model8.convs.1.weight",
                         "model9.norm.weight", "model10.convs.0.weight", "model_out.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheSiggraphRemapCountsConvolutionsPerBlock() {
        // Encoder blocks open with a convolution; decoder blocks open with a ReLU, so the same slot
        // number means different layers depending on the block.
        XCTAssertEqual(NFKMLXSiggraphColorizer.remapReferenceKey("model1.0.weight"), "model1.convs.0.weight")
        XCTAssertEqual(NFKMLXSiggraphColorizer.remapReferenceKey("model1.4.weight"), "model1.norm.weight")
        XCTAssertEqual(NFKMLXSiggraphColorizer.remapReferenceKey("model3.4.weight"), "model3.convs.2.weight")
        XCTAssertEqual(NFKMLXSiggraphColorizer.remapReferenceKey("model8.1.weight"), "model8.convs.0.weight")
        XCTAssertEqual(NFKMLXSiggraphColorizer.remapReferenceKey("model8.5.weight"), "model8.norm.weight")
        XCTAssertEqual(NFKMLXSiggraphColorizer.remapReferenceKey("model10.1.bias"), "model10.convs.0.bias")
        XCTAssertEqual(NFKMLXSiggraphColorizer.remapReferenceKey("model8up.0.weight"), "model8up.weight")
        XCTAssertEqual(NFKMLXSiggraphColorizer.remapReferenceKey("model_out.0.bias"), "model_out.bias")
    }

    func testSiggraphColorizationKeepsTheInputSize() throws {
        try requireMLXRuntime()
        let net = NFKMLXSiggraphColorizer.makeNet()
        net.train(false)
        // The network subsamples three times, so a side that is a multiple of eight round-trips.
        let colorized = net.colorize(Self.grayImage(height: 64, width: 64))
        eval(colorized)
        XCTAssertEqual(colorized.shape, [64, 64, 3], "colorized at the size it came in at")
    }

    // MARK: Helpers

    static func grayImage(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for pixel in 0 ..< (height * width) {
            let gray = Float((pixel * 23) % 256) / 255.0        // R = G = B: a grayscale photo
            values[pixel * 3] = gray
            values[pixel * 3 + 1] = gray
            values[pixel * 3 + 2] = gray
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
