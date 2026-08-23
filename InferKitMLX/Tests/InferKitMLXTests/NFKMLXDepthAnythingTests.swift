//
//  NFKMLXDepthAnythingTests.swift
//  InferKitMLXTests
//
//  DINOv2 + DPT is large; these use a tiny configuration so the forward and weight round-trip run
//  quickly. They evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDepthAnythingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    // A small ViT + DPT: 4 blocks, 2 heads, 42×42 input (3×3 patches of 14), so the forward is cheap.
    private func tinyConfiguration() -> NFKMLXDepthConfiguration {
        var configuration = NFKMLXDepthConfiguration()
        configuration.inputSize = 42
        configuration.embedDimensions = 16
        configuration.depth = 4
        configuration.heads = 2
        configuration.hooks = [0, 1, 2, 3]
        configuration.features = 8
        configuration.outChannels = [4, 8, 16, 32]
        return configuration
    }

    func testTheForwardProducesANormalizedGrayscaleDepthMapAtInputSize() throws {
        try requireMLXRuntime()
        // Deterministic random weights: an unseeded forward occasionally produces a near-constant map,
        // whose normalization (span clamped to 1e-6) collapses to all-zero, failing the max == 1 check.
        MLX.seed(0)
        let net = NFKMLXDepthAnything.makeNet(tinyConfiguration())
        let output = net.depth(Self.image(height: 20, width: 28))
        eval(output)
        XCTAssertEqual(output.shape, [20, 28, 3], "depth returns to the input size, as a gray image")

        let values = output.asArray(Float.self)
        XCTAssertEqual(try XCTUnwrap(values.min()), 0, accuracy: 1e-4, "normalized to 0")
        XCTAssertEqual(try XCTUnwrap(values.max()), 1, accuracy: 1e-4, "normalized to 1")
        for pixel in stride(from: 0, to: values.count, by: 3) {
            XCTAssertEqual(values[pixel], values[pixel + 1], accuracy: 1e-6, "gray: R == G")
            XCTAssertEqual(values[pixel + 1], values[pixel + 2], accuracy: 1e-6, "gray: G == B")
        }
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = NFKMLXDepthAnything.makeNet(tinyConfiguration())
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("dav2-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXDepthAnything.makeNet(tinyConfiguration())
        try NFKMLXDepthAnything.loadWeights(into: loaded, from: url)

        let input = Self.image(height: 16, width: 16)
        let expected = trained.depth(input)
        let actual = loaded.depth(input)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    func testTheReferenceKeyLayoutIsWhatTheConverterExpects() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXDepthAnything.makeNet(tinyConfiguration()).parameters().flattened().map(\.0))
        for expected in ["pretrained.patch_embed.proj.weight", "pretrained.cls_token", "pretrained.pos_embed",
                         "pretrained.blocks.0.attn.qkv.weight", "pretrained.blocks.0.ls1.gamma",
                         "pretrained.blocks.0.mlp.fc1.weight", "pretrained.norm.weight",
                         "depth_head.projects.0.weight", "depth_head.resize_layers.0.weight",
                         "depth_head.scratch.layer1_rn.weight", "depth_head.scratch.refinenet1.resConfUnit1.conv1.weight",
                         "depth_head.scratch.output_conv1.weight", "depth_head.scratch.output_conv2.0.weight",
                         "depth_head.scratch.output_conv2.2.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testThePresetsCarryTheExpectedDimensions() {
        XCTAssertEqual(NFKMLXDepthConfiguration.small.embedDimensions, 384)
        XCTAssertEqual(NFKMLXDepthConfiguration.base.embedDimensions, 768)
        XCTAssertEqual(NFKMLXDepthConfiguration.base.outChannels, [96, 192, 384, 768])
        XCTAssertEqual(NFKMLXDepthConfiguration.large.embedDimensions, 1024)
        XCTAssertEqual(NFKMLXDepthConfiguration.large.depth, 24)
        XCTAssertEqual(NFKMLXDepthConfiguration.large.hooks, [4, 11, 17, 23])
    }

    func testTheBasePresetProducesAGrayscaleDepthMap() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXDepthConfiguration.base
        configuration.inputSize = 42                            // 3×3 patches, so the ViT-B forward is cheap
        let output = NFKMLXDepthAnything.makeNet(configuration).depth(Self.image(height: 12, width: 16))
        eval(output)
        XCTAssertEqual(output.shape, [12, 16, 3])
        let values = output.asArray(Float.self)
        XCTAssertEqual(values[0], values[1], accuracy: 1e-6, "gray")
    }

    static func image(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 53) % 256) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [height, width, 3]) }
    }
}
