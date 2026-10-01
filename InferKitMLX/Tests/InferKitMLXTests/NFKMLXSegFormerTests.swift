//
//  NFKMLXSegFormerTests.swift
//  InferKitMLXTests
//
//  The MiT transformer + MLP-head segmenter. The forward and the weight round-trip evaluate MLX
//  arrays, so they skip without a Metal library for MLX (see Tools/mlx-metallib.sh).
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSegFormerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyNet() -> NFKMLXSegFormerNet {
        NFKMLXSegFormerNet(.tiny)
    }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["stage1.patch_embed.proj.weight", "stage1.blocks.0.attn.q.weight",
                         "stage1.blocks.0.attn.sr.weight", "stage1.blocks.0.ffn.dwconv.weight",
                         "stage4.norm.weight", "linear_c.0.weight", "linear_c.3.weight",
                         "linear_fuse.weight", "classifier.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testLogitsAreAtQuarterResolutionWithOneChannelPerClass() throws {
        try requireMLXRuntime()
        let logits = tinyNet().logits(Self.image(height: 32, width: 32).reshaped([1, 32, 32, 3]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 8, 8, NFKMLXSegFormerConfiguration.tiny.classCount],
                       "stage-1 is H/4, one channel per class")
    }

    /// The backend segments through the network as built, so a decode-head BatchNorm left in training
    /// mode would normalize each image by its own statistics and overwrite the released ones.
    func testANetworkIsBuiltInEvaluationModeAndKeepsItsStatistics() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        XCTAssertFalse(net.batchNorm.training)
        func statistics() -> [Float] {
            net.parameters().flattened().first { $0.0 == "batch_norm.running_mean" }!.1.asArray(Float.self)
        }
        let before = statistics()
        eval(net.segment(Self.image(height: 32, width: 32)))
        XCTAssertEqual(statistics(), before, "inference reads the running statistics without folding into them")
    }

    func testTheDropPathRisesAcrossEveryStagesBlocks() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXSegFormerConfiguration.tiny
        configuration.depths = [2, 1, 1, 1]
        let net = NFKMLXSegFormerNet(configuration)
        let shares = [net.stage1, net.stage2, net.stage3, net.stage4].flatMap { $0.blocks.map(\.depth) }
        XCTAssertEqual(shares, [0, 0.25, 0.5, 0.75, 1], "torch.linspace(0, rate, sum(depths))")
        XCTAssertEqual(net.dropout, .none)
    }

    func testTheDropoutRunsOnlyInTraining() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(20_260_930)
        let net = tinyNet()
        let image = Self.image(height: 32, width: 32).reshaped([1, 32, 32, 3])
        let plain = net.logits(image)
        net.dropout = .reference
        XCTAssertEqual(abs(net.logits(image) - plain).max().item(Float.self), 0, "evaluation never drops")
        net.train(true)
        net.dropout = .none
        let undropped = net.logits(image)
        XCTAssertEqual(abs(net.logits(image) - undropped).max().item(Float.self), 0)
        net.dropout = .reference
        XCTAssertGreaterThan(abs(net.logits(image) - undropped).max().item(Float.self), 0)
    }

    func testTheClassifierDropoutZeroesWholeChannels() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(20_260_930)
        let net = tinyNet()
        net.dropout = NFKMLXSegFormerDropout(classifier: 0.5)
        net.train(true)
        let dropped = net.channelDropout(MLXArray.ones([2, 3, 3, 16]))
        let perChannel = dropped.transposed(0, 3, 1, 2).reshaped([32, 9])
        XCTAssertEqual(perChannel.min(axis: 1).asArray(Float.self), perChannel.max(axis: 1).asArray(Float.self),
                       "Dropout2d drops a channel at every position")
        XCTAssertEqual(Set(perChannel[0..., 0].asArray(Float.self)), [0, 2])
    }

    func testSegmentationIsALabelMapAtInputSize() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let map = net.segment(Self.image(height: 32, width: 32))
        eval(map)
        XCTAssertEqual(map.shape, [32, 32, 1], "label map at input resolution")
        let values = map.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), 0, "class index scaled to 0...1")
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("segformer-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = tinyNet()
        try NFKMLXSegFormer.loadWeights(into: loaded, from: url)

        let image = Self.image(height: 32, width: 32).reshaped([1, 32, 32, 3])
        let expected = trained.logits(image)
        let actual = loaded.logits(image)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendSegmentsACGImage() throws {
        try requireMLXRuntime()
        NFKMLXSegFormer.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXSegFormer.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXSegFormer.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64, 64)]))
        let output = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 64, "label map keeps the input size")
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
