//
//  NFKMLXMODNetTests.swift
//  InferKitMLXTests
//
//  The three-branch portrait matting network. The forward and the weight round-trip evaluate MLX
//  arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXMODNetTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXMODNetNet {
        NFKMLXMODNetNet(.tiny)
    }

    func testParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["backbone.stem.conv.weight", "backbone.blocks.0.dw.conv.weight",
                         "backbone.blocks.1.expand.conv.weight", "backbone.blocks.0.project.weight",
                         "backbone.last.bn.weight",
                         "se_block.reduce.weight", "se_block.expand.weight",
                         "conv_lr16x.conv.weight", "conv_lr16x.norm.bnorm.weight", "conv_lr.conv.weight",
                         "tohr_enc2x.conv.weight", "conv_hr4x.0.conv.weight", "conv_hr2x.3.conv.weight",
                         "conv_hr.1.conv.weight", "conv_lr4x.conv.weight", "conv_f.1.conv.bias"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheReferenceKeyRemapCoversTheBackboneAndTheBranches() {
        let c = NFKMLXMODNetConfiguration.base
        // The stem and the final 1×1 expansion bracket the inverted-residual list.
        XCTAssertEqual(NFKMLXMODNet.remapReferenceKey("module.lr_branch.backbone.model.features.0.0.weight", configuration: c),
                       "backbone.stem.conv.weight")
        XCTAssertEqual(NFKMLXMODNet.remapReferenceKey("module.lr_branch.backbone.model.features.18.1.bias", configuration: c),
                       "backbone.last.bn.bias")
        // features.1 is the expansion-1 block: its depthwise pair sits at 0/1 and projection at 3/4.
        XCTAssertEqual(NFKMLXMODNet.remapReferenceKey("module.lr_branch.backbone.model.features.1.conv.0.weight", configuration: c),
                       "backbone.blocks.0.dw.conv.weight")
        XCTAssertEqual(NFKMLXMODNet.remapReferenceKey("module.lr_branch.backbone.model.features.1.conv.3.weight", configuration: c),
                       "backbone.blocks.0.project.weight")
        // features.2 expands, so the same slots mean different layers.
        XCTAssertEqual(NFKMLXMODNet.remapReferenceKey("module.lr_branch.backbone.model.features.2.conv.0.weight", configuration: c),
                       "backbone.blocks.1.expand.conv.weight")
        XCTAssertEqual(NFKMLXMODNet.remapReferenceKey("module.lr_branch.backbone.model.features.2.conv.6.weight", configuration: c),
                       "backbone.blocks.1.project.weight")
        // Branch prefixes drop, and each Conv2dIBNormRelu's `layers` Sequential becomes semantic.
        XCTAssertEqual(NFKMLXMODNet.remapReferenceKey("module.hr_branch.conv_hr4x.0.layers.0.weight", configuration: c),
                       "conv_hr4x.0.conv.weight")
        XCTAssertEqual(NFKMLXMODNet.remapReferenceKey("module.f_branch.conv_f2x.layers.1.bnorm.weight", configuration: c),
                       "conv_f2x.norm.bnorm.weight")
        XCTAssertEqual(NFKMLXMODNet.remapReferenceKey("module.lr_branch.se_block.fc.0.weight", configuration: c),
                       "se_block.reduce.weight")
    }

    func testMattingProducesForegroundPlusAlphaInRange() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let matte = net.matte(Self.image(height: 32, width: 32))
        eval(matte)
        XCTAssertEqual(matte.shape, [32, 32, 4], "straight foreground plus one alpha channel")
        let alpha = matte[0..., 0..., 3].asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(alpha.min()), 0, "alpha is a sigmoid probability")
        XCTAssertLessThanOrEqual(try XCTUnwrap(alpha.max()), 1)
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("modnet-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = tinyNet()
        try NFKMLXMODNet.loadWeights(into: loaded, from: url)

        let image = Self.image(height: 16, width: 16)
        let expected = trained.matte(image)
        let actual = loaded.matte(image)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendMattesACGImage() throws {
        try requireMLXRuntime()
        NFKMLXMODNet.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXMODNet.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXMODNet.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(32, 32)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage), "foreground emitted")
        XCTAssertNotNil(result.output(forKey: NFKOutputMask), "matte emitted on its own")
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
}
