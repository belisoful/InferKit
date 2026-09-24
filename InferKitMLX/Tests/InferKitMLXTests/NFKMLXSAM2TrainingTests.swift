//
//  NFKMLXSAM2TrainingTests.swift
//  InferKitMLXTests
//
//  Fine-tuning SAM 2 onto a consumer's own subject. Two things here are load-bearing beyond the loop
//  itself: the image encoder and the whole memory path actually stay frozen for a head-only run, and
//  a fine-tuned checkpoint reloads through the model's own loader without the transpose a released
//  file needs.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXSAM2TrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_921)
    }

    /// A tiny configuration: the released tiny is 39M parameters over a 1024-pixel frame, which no
    /// test can train. The widths that matter to the recipe — the decoder's 256 and the memory's 64 —
    /// are the released ones, because they are what the frozen and trainable groups are drawn across.
    private func smallNet() -> NFKMLXSAM2TrackerNet {
        let encoder = NFKMLXSAM2Configuration(
            embedDimensions: 12, heads: 1, stages: [1, 1, 1, 1], windowSpec: [2, 2, 2, 2],
            globalAttentionBlocks: [3], backgroundWindow: 2, neckChannels: 256)
        return NFKMLXSAM2TrackerNet(
            NFKMLXSAM2TrackerConfiguration(encoder: encoder, imageSize: 64,
                                           occlusionSpatialEmbedding: true,
                                           temporalPositionEncodingForObjectPointers: true))
    }

    /// One annotated frame: a textured plate, a click inside the subject, and the subject's mask.
    private func example(size: Int = 64)
        -> (image: MLXArray, points: [(x: Float, y: Float, label: Int)], target: MLXArray) {
        var pixels = [Float](repeating: 0, count: size * size * 3)
        for index in 0 ..< pixels.count {
            pixels[index] = Float((index * 31) % 200) / 255.0
        }
        var mask = [Float](repeating: 0, count: size * size)
        for row in (size / 4) ..< (3 * size / 4) {
            for column in (size / 4) ..< (3 * size / 4) {
                mask[row * size + column] = 1
                let base = (row * size + column) * 3
                pixels[base] = 0.9
                pixels[base + 1] = 0.9
            }
        }
        let image = pixels.withUnsafeBufferPointer { MLXArray($0, [1, size, size, 3]) }
        let target = mask.withUnsafeBufferPointer { MLXArray($0, [1, size, size]) }
        return (image, [(Float(size / 2), Float(size / 2), 1)], target)
    }

    // MARK: - The objective

    func testTheObjectiveScoresAnAnnotatedFrame() throws {
        try requireMLXRuntime()
        let net = smallNet()
        let sample = example()
        let loss = NFKMLXSAM2Objective()(net, sample.image, points: sample.points, sample.target)
        eval(loss)
        let value = loss.item(Float.self)
        XCTAssertTrue(value.isFinite, "the objective scores a frame to a finite loss")
        XCTAssertGreaterThan(value, 0, "every term of the reference's objective is non-negative")
    }

    /// A target with nothing in it trains the object score alone: the reference multiplies the mask
    /// terms by whether the target holds an object, so an empty frame contributes only the class term.
    func testAnEmptyTargetScoresOnlyTheObjectTerm() throws {
        try requireMLXRuntime()
        let objective = NFKMLXSAM2Objective()
        let masks = MLXRandom.normal([1, 3, 8, 8])
        let empty = MLXArray.zeros([1, 8, 8])
        let parts = objective.components(masks: masks, target: empty,
                                         intersectionOverUnion: MLXRandom.uniform(low: 0, high: 1, [1, 3]),
                                         objectScore: MLXArray([Float(0.5)]).reshaped([1, 1]))
        eval(parts.mask, parts.dice, parts.intersectionOverUnion, parts.object)
        XCTAssertEqual(parts.mask.item(Float.self), 0, accuracy: 1e-7)
        XCTAssertEqual(parts.dice.item(Float.self), 0, accuracy: 1e-7)
        XCTAssertEqual(parts.intersectionOverUnion.item(Float.self), 0, accuracy: 1e-7)
        XCTAssertGreaterThan(parts.object.item(Float.self), 0,
                             "an absent object still trains the object score")
    }

    // MARK: - The run

    func testAFineTuneLowersTheLoss() throws {
        try requireMLXRuntime()
        let net = smallNet()
        let sample = example()
        let losses = try NFKMLXSAM2.fineTune(net, examples: { _ in sample },
                                            optimizer: AdamW(learningRate: 1e-3), steps: 12)
        XCTAssertEqual(losses.count, 12)
        XCTAssertLessThan(losses.suffix(3).reduce(0, +) / 3, losses.prefix(3).reduce(0, +) / 3,
                          "the loss falls over the run")
    }

    func testAHeadOnlyRunLeavesTheEncoderAndMemoryPathUntouched() throws {
        try requireMLXRuntime()
        let net = smallNet()
        let sample = example()

        func snapshot(_ module: Module) -> [Float] {
            module.parameters().flattened().flatMap { $0.1.reshaped([-1]).asArray(Float.self) }
        }
        let encoderBefore = snapshot(net.imageEncoder)
        let memoryBefore = snapshot(net.memoryEncoder)
        let attentionBefore = snapshot(net.memoryAttention)
        let decoderBefore = snapshot(net.maskDecoder)
        let pointerBefore = snapshot(net.objectPointerProjection)

        try NFKMLXSAM2.fineTune(net, examples: { _ in sample }, trainable: .maskDecoder,
                                optimizer: AdamW(learningRate: 1e-3), steps: 6)

        XCTAssertEqual(snapshot(net.imageEncoder), encoderBefore, "the image encoder stays frozen")
        XCTAssertEqual(snapshot(net.memoryEncoder), memoryBefore, "the memory encoder stays frozen")
        XCTAssertEqual(snapshot(net.memoryAttention), attentionBefore,
                       "the memory attention stays frozen")
        XCTAssertEqual(snapshot(net.objectPointerProjection), pointerBefore,
                       "the tracker's own parameters belong to the memory path and stay frozen")
        XCTAssertNotEqual(snapshot(net.maskDecoder), decoderBefore, "the mask decoder trains")
    }

    // MARK: - The whole customization path

    func testAFineTunedCheckpointRoundTrips() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sam2-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = smallNet()
        let sample = example()
        try NFKMLXSAM2.fineTune(net, examples: { _ in sample },
                                optimizer: AdamW(learningRate: 1e-3), steps: 4)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = smallNet()
        try NFKMLXSAM2.loadWeights(into: reloaded, from: url)
        let expected = net.segment(image: sample.image, points: sample.points)
        let actual = reloaded.segment(image: sample.image, points: sample.points)
        eval(expected.masks, actual.masks)
        XCTAssertEqual(actual.masks.reshaped([-1]).asArray(Float.self),
                       expected.masks.reshaped([-1]).asArray(Float.self),
                       "the fine-tuned checkpoint reproduces the segmentation exactly")
    }
}
