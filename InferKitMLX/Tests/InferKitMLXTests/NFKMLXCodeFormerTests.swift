//
//  NFKMLXCodeFormerTests.swift
//  InferKitMLXTests
//
//  The VQGAN + Transformer face restorer. The forward, the code prediction, and the weight round-trip
//  evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXCodeFormerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXCodeFormerNet {
        NFKMLXCodeFormerNet(.tiny)
    }

    // MARK: Parameter names

    func testParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["encoder.blocks.0.weight", "encoder.blocks.1.conv1.weight",
                         "encoder.blocks.1.norm1.weight", "quantize.embedding.weight",
                         "feat_emb.weight", "position_emb",
                         "ft_layers.0.in_proj_weight", "ft_layers.0.out_proj.weight",
                         "ft_layers.0.linear1.weight", "idx_pred_layer.norm.weight",
                         "idx_pred_layer.proj.weight", "generator.blocks.0.weight",
                         "fuse_blocks.0.encode_enc.conv1.weight", "fuse_blocks.0.scale1.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheReferenceKeyRemapCoversTheDictionariesAndSequentials() {
        let connect = NFKMLXCodeFormerConfiguration.base.connectResolutions
        XCTAssertEqual(NFKMLXCodeFormer.remapReferenceKey("fuse_convs_dict.32.encode_enc.conv1.weight",
                                                          connectResolutions: connect),
                       "fuse_blocks.0.encode_enc.conv1.weight")
        XCTAssertEqual(NFKMLXCodeFormer.remapReferenceKey("fuse_convs_dict.256.scale.0.weight",
                                                          connectResolutions: connect),
                       "fuse_blocks.3.scale1.weight")
        XCTAssertEqual(NFKMLXCodeFormer.remapReferenceKey("fuse_convs_dict.64.shift.2.bias",
                                                          connectResolutions: connect),
                       "fuse_blocks.1.shift2.bias")
        XCTAssertEqual(NFKMLXCodeFormer.remapReferenceKey("idx_pred_layer.0.weight",
                                                          connectResolutions: connect),
                       "idx_pred_layer.norm.weight")
        XCTAssertEqual(NFKMLXCodeFormer.remapReferenceKey("idx_pred_layer.1.weight",
                                                          connectResolutions: connect),
                       "idx_pred_layer.proj.weight")
        XCTAssertEqual(NFKMLXCodeFormer.remapReferenceKey("ft_layers.3.self_attn.in_proj_weight",
                                                          connectResolutions: connect),
                       "ft_layers.3.in_proj_weight")
        XCTAssertEqual(NFKMLXCodeFormer.remapReferenceKey("encoder.blocks.17.q.weight",
                                                          connectResolutions: connect),
                       "encoder.blocks.17.q.weight")
    }

    func testTheBaseGeometryMatchesTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        // Spot geometry the released codeformer.pth fixes: 25 coder blocks each, captures at the
        // reference's fuse_encoder_block indices and fuses at fuse_generator_block's.
        let net = NFKMLXCodeFormerNet(.base)
        XCTAssertEqual(net.encoder.blocks.count, 25)
        XCTAssertEqual(net.generator.blocks.count, 25)
        XCTAssertEqual(net.captureIndices, [14, 11, 8, 5], "encoder captures at 32/64/128/256")
        XCTAssertEqual(net.fuseIndices, [9, 12, 15, 18], "generator fuses at 32/64/128/256")
    }

    // MARK: Forward and code prediction (needs MLX)

    func testRestorationReturnsAModelResolutionImageInRange() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let restored = net.restore(Self.image(height: 24, width: 20))
        eval(restored)
        XCTAssertEqual(restored.shape, [NFKMLXCodeFormerConfiguration.tiny.resolution,
                                        NFKMLXCodeFormerConfiguration.tiny.resolution, 3])
        let values = restored.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0, "generator maps to 0...1")
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
    }

    func testPredictedIndicesStayWithinTheCodebook() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let resolution = NFKMLXCodeFormerConfiguration.tiny.resolution
        var latent = Self.image(height: resolution, width: resolution).reshaped([1, resolution, resolution, 3]) * 2 - 1
        for block in net.encoder.blocks {
            latent = block(latent)
        }
        let logits = net.codeLogits(latent)
        let indices = logits.argMax(axis: -1)
        eval(indices)
        XCTAssertEqual(logits.shape, [NFKMLXCodeFormerConfiguration.tiny.latentTokens,
                                      NFKMLXCodeFormerConfiguration.tiny.codebookSize])
        let ids = indices.asArray(Int32.self)
        XCTAssertGreaterThanOrEqual(ids.min() ?? 0, 0)
        XCTAssertLessThan(ids.max() ?? 0, Int32(NFKMLXCodeFormerConfiguration.tiny.codebookSize))
    }

    func testTheFidelityWeightChangesTheRestoration() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let image = Self.image(height: 32, width: 32)
        let generative = net.restore(image, fidelity: 0)
        let faithful = net.restore(image, fidelity: 1)
        eval(generative, faithful)
        XCTAssertNotEqual(generative.asArray(Float.self), faithful.asArray(Float.self),
                          "the controllable feature transformation is active")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("codeformer-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = tinyNet()
        try NFKMLXCodeFormer.loadWeights(into: loaded, from: url)

        let image = Self.image(height: 16, width: 16)
        let expected = trained.restore(image)
        let actual = loaded.restore(image)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheFidelityFactoryBuildsABackendAtThatFidelity() throws {
        try requireMLXRuntime()
        // The fidelity is the Objective-C-facing knob, so it has to survive the factory rather than
        // only the Swift configuration. Out-of-range values clamp instead of scaling the modulation.
        let request = NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64, 64)])
        let generative = try NFKMLXCodeFormer.backend(fidelity: 0, weightsURL: nil)
            .runInference(for: request)
        let faithful = try NFKMLXCodeFormer.backend(fidelity: 2, weightsURL: nil)
            .runInference(for: request)
        let a = try Self.cgImage(generative.output(forKey: NFKOutputImage))
        let b = try Self.cgImage(faithful.output(forKey: NFKOutputImage))
        XCTAssertEqual(a.width, NFKMLXCodeFormerConfiguration.base.resolution)
        XCTAssertEqual(b.width, NFKMLXCodeFormerConfiguration.base.resolution)
    }

    func testTheRegisteredBackendRestoresACGImage() throws {
        try requireMLXRuntime()
        NFKMLXCodeFormer.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXCodeFormer.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXCodeFormer.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64, 64)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, NFKMLXCodeFormerConfiguration.base.resolution, "restored at the model resolution")
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
