//
//  NFKMLXSiggraphColorizerTrainingTests.swift
//  InferKitMLXTests
//
//  The SIGGRAPH-17 colorizer's fine-tune without released weights: the reference's CIELAB, the grayscale
//  filter, the hint patches, the objective, and a fine-tune that lowers the loss and reloads through both
//  factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

/// Draws that replay a recorded sequence.
struct NFKSiggraphReplayedDraws: NFKSiggraphHintDraws {
    var units: [Double]
    var choices: [Int]
    var normals: [Double]

    mutating func unit() -> Double { units.removeFirst() }
    mutating func choice(_ sizes: [Int]) -> Int { choices.removeFirst() }
    mutating func normal(mean: Double, deviation: Double) -> Double { normals.removeFirst() }
}

final class NFKMLXSiggraphColorizerTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_007)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Two colorful 32-pixel images and a grayscale one.
    private func images() -> MLXArray {
        let colored = MLXRandom.uniform(low: 0, high: 1, [2, 32, 32, 3], key: MLXRandom.key(1))
        let gray = MLXRandom.uniform(low: 0, high: 1, [1, 32, 32, 1], key: MLXRandom.key(2))
        return concatenated([colored, concatenated([gray, gray, gray], axis: -1)], axis: 0)
    }

    func testTheReferenceLabMatchesTheCIEWhiteAndGray() throws {
        try requireMLXRuntime()
        let white = NFKMLXSiggraphColorizer.referenceLab(MLXArray.ones([1, 1, 1, 3]))
        XCTAssertEqual(white[0, 0, 0, 0].item(Float.self), 0.5, accuracy: 1e-4, "L 100 normalizes to 0.5")
        XCTAssertEqual(abs(white[.ellipsis, 1 ..< 3]).max().item(Float.self), 0, accuracy: 1e-4, "white has no chroma")
        let gray = NFKMLXSiggraphColorizer.referenceLab(MLXArray([Float(0.5), 0.5, 0.5], [1, 1, 1, 3]))
        XCTAssertEqual(gray[0, 0, 0, 0].item(Float.self) * 100 + 50, 53.39, accuracy: 0.01)
    }

    func testGrayscaleImagesAreDroppedAndHintsFillWholePatches() throws {
        try requireMLXRuntime()
        // Units below 1 − p add a patch; the 0.99 closes each image's loop.
        var draws = NFKSiggraphReplayedDraws(units: [0.1, 0.99, 0.99], choices: [3], normals: [5.2, 7.8])
        let example = try XCTUnwrap(NFKMLXSiggraphColorizer.trainingExample(images(), hintProbability: 0.125, draws: &draws))
        XCTAssertEqual(example.input.shape, [2, 32, 32, 4], "the grayscale image is dropped")
        XCTAssertTrue(draws.units.isEmpty && draws.choices.isEmpty && draws.normals.isEmpty, "every draw is consumed")
        let mask = example.input[.ellipsis, 3]
        XCTAssertEqual(mask[0].max().item(Float.self), 0.5, "a hinted pixel's mask is 1 − 0.5")
        XCTAssertEqual((mask[0] .> 0).sum().item(Int.self), 9, "one 3-pixel patch")
        XCTAssertEqual(mask[1].max().item(Float.self), -0.5, "the second image receives no hint")
        // The patch sits at (5, 7) and holds the mean of the true ab there.
        let hint = example.input[0, 5 ..< 8, 7 ..< 10, 1 ..< 3]
        let truth = example.target[0, 5 ..< 8, 7 ..< 10, 0...].mean(axes: [0, 1])
        XCTAssertLessThan(abs(hint - truth).max().item(Float.self), 1e-6)
        XCTAssertNil(NFKMLXSiggraphColorizer.trainingExample(images()[2 ..< 3]), "an all-grayscale batch has no example")
    }

    func testTheObjectiveIsTenTimesTheSummedL1() throws {
        try requireMLXRuntime()
        let target = MLXArray.zeros([1, 2, 2, 2])
        let prediction = MLXArray.ones([1, 2, 2, 2]) * 0.1
        XCTAssertEqual(NFKMLXSiggraphColorizerObjective().loss(prediction: prediction, target: target).item(Float.self), 2, accuracy: 1e-6)
    }

    func testFineTuningLowersTheLossAndReloadsThroughBothFactories() throws {
        try requireMLXRuntime()
        let net = try NFKMLXSiggraphColorizer.network(weightsURL: nil)
        let batch = images()[0 ..< 2]
        let example = try XCTUnwrap(NFKMLXSiggraphColorizer.trainingExample(batch, seed: 0))
        let objective = NFKMLXSiggraphColorizerObjective()
        net.train(true)
        let before = objective(net, example.input, example.target).item(Float.self)
        net.train(false)
        let losses = try NFKMLXSiggraphColorizer.fineTune(net, examples: { _ in batch }, optimizer: Adam(learningRate: 1e-3),
                                                          steps: 4, hintProbability: 1)
        XCTAssertEqual(losses.count, 4)
        XCTAssertLessThan(losses.last!, before)
        XCTAssertTrue(net.modules().allSatisfy { !$0.training }, "the run restores evaluation mode")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXSiggraphColorizer.network(weightsURL: url)
        XCTAssertLessThan(abs(net(example.input) - reloaded(example.input)).max().item(Float.self), 1e-5)
        XCTAssertNoThrow(try NFKMLXSiggraphColorizer.backend(weightsURL: url))
    }
}
