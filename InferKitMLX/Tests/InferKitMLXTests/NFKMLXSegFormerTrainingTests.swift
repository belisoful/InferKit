//
//  NFKMLXSegFormerTrainingTests.swift
//  InferKitMLXTests
//
//  Fine-tuning SegFormer onto a consumer's own class set. Two things here are load-bearing beyond the
//  loop itself: the encoder actually stays frozen for a head-only run, and retargeting the class count
//  drops the checkpoint's classifier instead of letting MLX adopt its shape.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXSegFormerTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func setUp() {
        super.setUp()
        NFKMLXRandom.seed(20_260_814)
    }

    /// A tiny configuration: the full model is far too slow to train in a test.
    private func smallNet(classCount: Int = 4) -> NFKMLXSegFormerNet {
        var configuration = NFKMLXSegFormerConfiguration.tiny
        configuration.classCount = classCount
        return NFKMLXSegFormerNet(configuration)
    }

    /// One annotated example: a textured image and a two-region label map.
    private func example(size: Int = 32, classCount: Int = 4) -> (image: MLXArray, labels: MLXArray) {
        var pixels = [Float](repeating: 0, count: size * size * 3)
        for i in 0 ..< pixels.count {
            pixels[i] = Float((i * 29) % 200) / 255.0
        }
        let image = pixels.withUnsafeBufferPointer { MLXArray($0, [size, size, 3]) }

        var indices = [Int32](repeating: 0, count: size * size)
        for row in 0 ..< size {
            for column in 0 ..< size {
                indices[row * size + column] = Int32((row < size / 2 ? 1 : 2) % classCount)
            }
        }
        let labels = indices.withUnsafeBufferPointer { MLXArray($0, [size, size]) }
        return (image, labels)
    }

    // MARK: - The objective

    func testTheObjectiveScoresAnAnnotatedExample() throws {
        try requireMLXRuntime()
        let net = smallNet()
        let sample = example()
        let loss = NFKMLXSegFormerObjective()(net, sample.image, sample.labels)
        XCTAssertTrue(loss.item(Float.self).isFinite, "cross-entropy over the upsampled logits")
        XCTAssertGreaterThan(loss.item(Float.self), 0)
    }

    func testTrainingDrivesTheLossDown() throws {
        try requireMLXRuntime()
        let net = smallNet()
        let sample = example()

        let history = try NFKMLXSegFormer.fineTune(net, examples: { _ in sample },
                                                   optimizer: AdamW(learningRate: 1e-3), steps: 40)

        XCTAssertEqual(history.count, 40)
        XCTAssertLessThan(history.last!, history.first!,
                          "the head learned the example: \(history.first!) -> \(history.last!)")
    }

    // MARK: - Freezing

    func testAHeadOnlyRunLeavesTheEncoderUntouched() throws {
        try requireMLXRuntime()
        let net = smallNet()
        let sample = example()
        let encoderBefore = net.stage1.parameters().flattened().map { $0.1.asArray(Float.self) }
        let headBefore = net.classifier.weight.asArray(Float.self)

        try NFKMLXSegFormer.fineTune(net, examples: { _ in sample }, trainable: .decodeHead,
                                     optimizer: AdamW(learningRate: 1e-3), steps: 10)

        let encoderAfter = net.stage1.parameters().flattened().map { $0.1.asArray(Float.self) }
        XCTAssertEqual(encoderAfter.count, encoderBefore.count)
        for (after, before) in zip(encoderAfter, encoderBefore) {
            XCTAssertEqual(after, before, "the frozen encoder did not move")
        }
        XCTAssertNotEqual(net.classifier.weight.asArray(Float.self), headBefore,
                          "the decode head did")
    }

    func testTrainingEverythingMovesTheEncoder() throws {
        try requireMLXRuntime()
        let net = smallNet()
        let sample = example()
        let before = net.stage1.patchEmbed.proj.weight.asArray(Float.self)

        try NFKMLXSegFormer.fineTune(net, examples: { _ in sample }, trainable: .everything,
                                     optimizer: AdamW(learningRate: 1e-3), steps: 10)

        XCTAssertNotEqual(net.stage1.patchEmbed.proj.weight.asArray(Float.self), before)
    }

    // MARK: - Retargeting the class set

    func testRetargetingTheClassCountDropsTheCheckpointClassifier() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("segformer-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        // A checkpoint trained on eight classes, retargeted to the consumer's three.
        let source = smallNet(classCount: 8)
        try NFKMLXWeights.save(source, to: url)

        var configuration = NFKMLXSegFormerConfiguration.tiny
        configuration.classCount = 3
        let retargeted = try Self.network(url: url, configuration: configuration)

        // MLX's update(parameters:) adopts a checkpoint's shapes rather than validating them, so
        // keeping the classifier would silently restore the eight-class head.
        XCTAssertEqual(retargeted.classifier.weight.shape.first, 3,
                       "the classifier stayed at the consumer's class count")
        XCTAssertEqual(retargeted.stage1.patchEmbed.proj.weight.asArray(Float.self),
                       source.stage1.patchEmbed.proj.weight.asArray(Float.self),
                       "while everything the consumer is not retargeting still loaded")
    }

    func testAMatchingClassCountLoadsTheClassifier() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("segformer-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let source = smallNet(classCount: 4)
        try NFKMLXWeights.save(source, to: url)

        var configuration = NFKMLXSegFormerConfiguration.tiny
        configuration.classCount = 4
        let loaded = try Self.network(url: url, configuration: configuration)
        XCTAssertEqual(loaded.classifier.weight.asArray(Float.self),
                       source.classifier.weight.asArray(Float.self),
                       "the same class set keeps the trained head")
    }

    // MARK: - The whole customization path

    func testAFineTunedCheckpointRoundTrips() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("segformer-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = smallNet()
        let sample = example()
        try NFKMLXSegFormer.fineTune(net, examples: { _ in sample },
                                     optimizer: AdamW(learningRate: 1e-3), steps: 8)
        try NFKMLXWeights.save(net, to: url)

        var configuration = NFKMLXSegFormerConfiguration.tiny
        configuration.classCount = 4
        let reloaded = try Self.network(url: url, configuration: configuration)
        let expected = net.segment(sample.image)
        let actual = reloaded.segment(sample.image)
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self),
                       "the fine-tuned checkpoint reproduces the segmentation exactly")
    }

    /// The public `network(weightsURL:classCount:)` builds the full-size model, which is far too slow
    /// for a test, so these exercise the same load path at the compact geometry.
    private static func network(url: URL, configuration: NFKMLXSegFormerConfiguration) throws -> NFKMLXSegFormerNet {
        let net = NFKMLXSegFormerNet(configuration)
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { key, value in
            (key, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        let outputs = mapped.first { $0.0 == "classifier.weight" }?.1.shape.first
        guard let outputs, outputs != configuration.classCount else {
            try NFKMLXWeights.apply(mapped, to: net)
            return net
        }
        try NFKMLXWeights.apply(mapped.filter { !$0.0.hasPrefix("classifier.") }, to: net, strict: false)
        return net
    }
}
