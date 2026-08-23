//
//  NFKMLXResNetTests.swift
//  InferKitMLXTests
//
//  The shared residual backbone. The forward evaluates MLX arrays, so those tests skip under
//  `swift test` and run under `xcodebuild test`.
//

import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXResNetTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    func testAnUndilatedBackboneReachesStrideThirtyTwo() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXResNetConfiguration.tiny
        configuration.replaceStrideWithDilation = [false, false, false]
        let features = NFKMLXResNetBackbone(configuration)(MLXArray.zeros([1, 64, 64, 3]))
        eval(features)
        XCTAssertEqual(features.shape, [1, 2, 2, configuration.outputChannels])
    }

    // Replacing the last two strides with dilations is what holds DeepLab's features at stride 8; a
    // backbone that strided instead would still run, just at a quarter of the detail.
    func testDilatingTheLastTwoStagesHoldsStrideEight() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXResNetConfiguration.tiny
        let features = NFKMLXResNetBackbone(configuration)(MLXArray.zeros([1, 64, 64, 3]))
        eval(features)
        XCTAssertEqual(features.shape, [1, 8, 8, configuration.outputChannels])
    }

    func testTheProjectionShortcutRemapsFromTheReferenceSequential() {
        XCTAssertEqual(NFKMLXResNetBackbone.remapReferenceKey("layer1.0.downsample.0.weight"),
                       "layer1.0.downsample_conv.weight")
        XCTAssertEqual(NFKMLXResNetBackbone.remapReferenceKey("layer2.0.downsample.1.running_mean"),
                       "layer2.0.downsample_bn.running_mean")
        XCTAssertEqual(NFKMLXResNetBackbone.remapReferenceKey("layer3.2.conv2.weight"),
                       "layer3.2.conv2.weight", "keys outside a shortcut pass through")
    }
}
