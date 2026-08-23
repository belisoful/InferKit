//
//  NFKMLXRVMTests.swift
//  InferKitMLXTests
//
//  The recurrent matting network. The forward, the recurrent-state carry, and the weight round-trip
//  evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXRVMTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXRVMNet {
        NFKMLXRVMNet(.tiny)
    }

    // MARK: Parameter names

    func testParameterNamesFollowTheReferenceLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["backbone.stem.conv.weight", "backbone.stem.bn.running_mean",
                         "backbone.blocks.0.dw.conv.weight", "backbone.blocks.1.expand.conv.weight",
                         "backbone.blocks.2.se.fc1.weight", "backbone.blocks.0.project.bn.weight",
                         "backbone.last.conv.weight",
                         "aspp.aspp1.conv.weight", "aspp.aspp1.bn.weight", "aspp.aspp2.conv.weight",
                         "decoder.decode4.gru.ih.conv.weight", "decoder.decode4.gru.hh.conv.bias",
                         "decoder.decode3.conv.conv.weight", "decoder.decode3.conv.bn.running_var",
                         "decoder.decode0.conv.conv1.weight", "decoder.decode0.conv.conv2.weight",
                         "project_mat.conv.weight", "project_seg.conv.weight",
                         "refiner.box_filter.weight", "refiner.conv.conv3.bias"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheReferenceKeyRemapCoversEveryBackboneForm() {
        let blocks = NFKMLXRVMConfiguration.large.blocks
        // Stem and final expansion.
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("backbone.features.0.0.weight", blocks: blocks),
                       "backbone.stem.conv.weight")
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("backbone.features.16.1.running_mean", blocks: blocks),
                       "backbone.last.bn.running_mean")
        // features.1 has no expansion: position 0 is the depthwise stage.
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("backbone.features.1.block.0.0.weight", blocks: blocks),
                       "backbone.blocks.0.dw.conv.weight")
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("backbone.features.1.block.1.1.bias", blocks: blocks),
                       "backbone.blocks.0.project.bn.bias")
        // features.4 expands and squeezes: expand, dw, se, project.
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("backbone.features.4.block.0.0.weight", blocks: blocks),
                       "backbone.blocks.3.expand.conv.weight")
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("backbone.features.4.block.2.fc1.weight", blocks: blocks),
                       "backbone.blocks.3.se.fc1.weight")
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("backbone.features.4.block.3.1.weight", blocks: blocks),
                       "backbone.blocks.3.project.bn.weight")
        // features.7 expands without squeeze: position 2 is the projection.
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("backbone.features.7.block.2.0.weight", blocks: blocks),
                       "backbone.blocks.6.project.conv.weight")
        // The context module, decoder Sequentials, GRUs, and refiner head are positional too.
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("aspp.aspp1.0.weight", blocks: blocks),
                       "aspp.aspp1.conv.weight")
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("aspp.aspp2.1.weight", blocks: blocks),
                       "aspp.aspp2.conv.weight")
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("decoder.decode4.gru.ih.0.weight", blocks: blocks),
                       "decoder.decode4.gru.ih.conv.weight")
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("decoder.decode1.conv.1.running_mean", blocks: blocks),
                       "decoder.decode1.conv.bn.running_mean")
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("decoder.decode0.conv.3.weight", blocks: blocks),
                       "decoder.decode0.conv.conv2.weight")
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("refiner.conv.6.bias", blocks: blocks),
                       "refiner.conv.conv3.bias")
        // Names already in module form pass through untouched.
        XCTAssertEqual(NFKMLXRVM.remapReferenceKey("project_mat.conv.weight", blocks: blocks),
                       "project_mat.conv.weight")
    }

    // MARK: Forward and recurrence (needs MLX)

    func testMattingASingleImageProducesForegroundPlusAlphaInRange() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let matte = net.matte(Self.frame(height: 32, width: 32, seed: 1))
        eval(matte)
        XCTAssertEqual(matte.shape, [32, 32, 4], "straight foreground plus one alpha channel")
        let alpha = matte[0..., 0..., 3].asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(alpha.min()), 0, "alpha is clamped to 0...1")
        XCTAssertLessThanOrEqual(try XCTUnwrap(alpha.max()), 1)
    }

    func testTheRecurrentStateChangesTheNextFramesResult() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let frame1 = Self.frame(height: 32, width: 32, seed: 1).reshaped([1, 32, 32, 3])
        let frame2 = Self.frame(height: 32, width: 32, seed: 2).reshaped([1, 32, 32, 3])

        let (_, _, state1) = net.forward(frame1, state: NFKMLXRVMNet.initialState)
        let (_, alphaFresh, _) = net.forward(frame2, state: NFKMLXRVMNet.initialState)
        let (_, alphaCarried, _) = net.forward(frame2, state: state1)
        eval(alphaFresh, alphaCarried)
        XCTAssertNotEqual(alphaFresh.asArray(Float.self), alphaCarried.asArray(Float.self),
                          "carrying the previous frame's state changes the matte — recurrence is active")
    }

    func testADownsampledPassRunsTheGuidedFilterAtTheFullFrameSize() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        net.train(false)
        let frame = Self.frame(height: 64, width: 64, seed: 4).reshaped([1, 64, 64, 3])
        let (foreground, alpha, _) = net.forward(frame, state: NFKMLXRVMNet.initialState, downsampleRatio: 0.5)
        eval(foreground, alpha)
        XCTAssertEqual(foreground.shape, [1, 64, 64, 3], "the refiner lifts the result back to the frame size")
        XCTAssertEqual(alpha.shape, [1, 64, 64, 1])
        let values = alpha.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0)
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
    }

    func testAFrameThatIsNotAMultipleOfTheStrideMattesAtItsOwnSize() throws {
        try requireMLXRuntime()
        // The backend hands the plate over at its native size, so an ordinary photo reaches the
        // network with odd dimensions: the decoder's source pooling hits a one-sample remainder and
        // every upsampled feature needs cropping back to its skip.
        let net = tinyNet()
        net.train(false)
        for (height, width) in [(33, 33), (30, 45), (17, 19)] {
            let matte = net.matte(Self.frame(height: height, width: width, seed: 1))
            eval(matte)
            XCTAssertEqual(matte.shape, [height, width, 4], "\(height)×\(width) survives the decoder crops")
        }
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("rvm-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = tinyNet()
        try NFKMLXRVM.loadWeights(into: loaded, from: url, blocks: NFKMLXRVMConfiguration.tiny.blocks)

        let frame = Self.frame(height: 16, width: 16, seed: 3)
        let expected = trained.matte(frame)
        let actual = loaded.matte(frame)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendMattesACGImage() throws {
        try requireMLXRuntime()
        NFKMLXRVM.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXRVM.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXRVM.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(32, 32)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage), "foreground emitted")
        XCTAssertNotNil(result.output(forKey: NFKOutputMask), "matte emitted on its own")
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

    // MARK: Fine-tuning

    /// The recipe question the port shipped with: do gradients flow through the squeeze-excitation
    /// gates and the hardswish activations, or does a fine-tune silently leave those blocks frozen?
    /// The tiny configuration covers every block form (expand, SE, hardswish, dilation), so a few
    /// steps against a fixed target answer it: the loss must fall, and the SE parameters must MOVE —
    /// a decreasing loss alone could ride on the decoder while the backbone stays untouched.
    func testAFineTuneMovesTheSqueezeExciteAndHardswishBlocks() throws {
        try requireMLXRuntime()
        let net = NFKMLXRVMNet(.tiny)
        let before = Dictionary(uniqueKeysWithValues: net.parameters().flattened()
            .filter { $0.0.contains(".se.") }
            .map { ($0.0, $0.1.asArray(Float.self)) })
        XCTAssertFalse(before.isEmpty, "the tiny configuration carries squeeze-excitation blocks")

        let frame = MLXArray.ones([1, 32, 32, 3]) * 0.5
        let target = MLXArray.ones([1, 32, 32, 1])
        let losses = try NFKMLXTrainer.train(
            net, optimizer: SGD(learningRate: 0.05), steps: 6,
            batch: { _ in (frame, target) },
            loss: { model, input, expected in
                let (_, alpha, _) = model.forward(input, state: NFKMLXRVMNet.initialState)
                return ((alpha - expected) * (alpha - expected)).mean()
            })

        XCTAssertLessThan(losses.last!, losses.first!, "the loss falls over the run")
        let after = Dictionary(uniqueKeysWithValues: net.parameters().flattened()
            .filter { $0.0.contains(".se.") }
            .map { ($0.0, $0.1.asArray(Float.self)) })
        let moved = before.filter { key, values in
            zip(values, after[key] ?? values).contains { abs($0 - $1) > 1e-9 }
        }
        XCTAssertFalse(moved.isEmpty,
                       "gradients reach the squeeze-excitation gates through the hardswish blocks")
    }
}
