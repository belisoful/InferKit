//
//  NFKMLXColorizerTrainingTests.swift
//  InferKitMLXTests
//
//  The ECCV-16 colorizer's fine-tune without released weights: the soft encoding, the prior factor, the gray
//  mask, the frozen readout and normalizations, and a fine-tune that lowers the loss and reloads through both
//  factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXColorizerTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_008)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Bin centers on a 10-unit grid, the layout of `pts_in_hull`.
    private func grid() -> MLXArray {
        let values = (-5 ..< 5).flatMap { a in (-5 ..< 5).flatMap { b in [Float(a * 10), Float(b * 10)] } }
        return MLXArray(values, [100, 2])
    }

    func testTheSoftEncodingSpreadsOverTheTenNearestBins() throws {
        try requireMLXRuntime()
        let encoded = NFKMLXColorizerObjective().encoding(MLXArray([Float(3), -4], [1, 2]), binCenters: grid())
        XCTAssertEqual((encoded .> 0).sum().item(Int.self), 10)
        XCTAssertEqual(encoded.sum().item(Float.self), 1, accuracy: 1e-6)
        XCTAssertEqual(argMax(encoded, axis: -1).item(Int.self), 5 * 10 + 5, "the nearest bin, (0, 0), weighs most")
    }

    func testThePriorFactorHasUnitExpectationUnderThePrior() throws {
        try requireMLXRuntime()
        let objective = NFKMLXColorizerObjective()
        let expectation = (objective.priorFactor * MLXArray(objective.priorProbabilities)).sum().item(Float.self)
        XCTAssertEqual(expectation, 1, accuracy: 1e-4)
        XCTAssertEqual(objective.priorProbabilities.count, 313)
    }

    func testAGrayImageCarriesNoGradientWeight() throws {
        try requireMLXRuntime()
        let net = try NFKMLXColorizer.network(weightsURL: nil)
        let gray = MLXArray.ones([1, 32, 32, 3]) * 0.4
        let example = NFKMLXColorizer.trainingExample(gray)
        XCTAssertEqual(example.ab.shape, [1, 8, 8, 2])
        let terms = NFKMLXColorizerObjective().loss(logits: net.logits(example.lightness), ab: example.ab,
                                                    binCenters: NFKMLXColorizerObjective.binCenters)
        XCTAssertGreaterThan(terms.crossEntropy.item(Float.self), 0)
        XCTAssertEqual(terms.rebalanced.item(Float.self), 0)
        XCTAssertEqual(NFKMLXColorizerObjective.binCenters.shape, [313, 2])
        let readout = net.outAb.weight.reshaped([2, -1]).transposed(1, 0) * 110
        XCTAssertLessThan(abs(readout - NFKMLXColorizerObjective.binCenters).max().item(Float.self), 1e-4,
                          "a new network's readout starts at the bin centers")
    }

    func testFineTuningLowersTheLossHoldsTheFrozenLayersAndReloads() throws {
        try requireMLXRuntime()
        let net = try NFKMLXColorizer.network(weightsURL: nil)
        let images = MLXRandom.uniform(low: 0, high: 1, [2, 32, 32, 3], key: MLXRandom.key(3))
        let held = Dictionary(uniqueKeysWithValues: net.parameters().flattened()
            .filter { $0.0.hasPrefix("norm") || $0.0.hasPrefix("out_ab") }.map { ($0.0, $0.1 * 1) })
        let objective = NFKMLXColorizerObjective()
        let example = NFKMLXColorizer.trainingExample(images)
        net.train(true)
        let before = objective(net, example.lightness, example.ab).item(Float.self)
        net.train(false)
        let losses = try NFKMLXColorizer.fineTune(net, examples: { _ in images }, optimizer: Adam(learningRate: 1e-4), steps: 4)
        XCTAssertEqual(losses.count, 4)
        XCTAssertLessThan(losses.last!, before)
        let after = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        for (key, value) in held where !key.contains("running") {
            XCTAssertEqual(abs(after[key]! - value).max().item(Float.self), 0, "\(key) holds")
        }

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXColorizer.network(weightsURL: url)
        XCTAssertLessThan(abs(net.logits(example.lightness) - reloaded.logits(example.lightness)).max().item(Float.self), 1e-5)
        XCTAssertNoThrow(try NFKMLXColorizer.backend(weightsURL: url))
    }

    func testTheFrozenNormalizationsNormalizeWithEachBatchAsCaffesDo() throws {
        try requireMLXRuntime()
        let net = try NFKMLXColorizer.network(weightsURL: nil)
        let images = MLXRandom.uniform(low: 0, high: 1, [2, 32, 32, 3], key: MLXRandom.key(5))
        let objective = NFKMLXColorizerObjective()
        let example = NFKMLXColorizer.trainingExample(images)
        net.train(true)
        let batchStatistics = objective(net, example.lightness, example.ab).item(Float.self)
        net.train(false)
        let runningStatistics = objective(net, example.lightness, example.ab).item(Float.self)
        func runningMean() -> [Float] {
            Dictionary(uniqueKeysWithValues: net.norm4.parameters().flattened())["running_mean"]!.asArray(Float.self)
        }
        let released = runningMean()
        var normalizationsTrained = [Bool]()
        let losses = try NFKMLXColorizer.fineTune(net, examples: { _ in images }, steps: 1, observer: { _ in
            normalizationsTrained.append(net.norm4.training)
            return true
        })
        print("eccv16 frozen normalizations: first loss \(losses[0]), batch statistics \(batchStatistics), "
              + "running statistics \(runningStatistics)")
        XCTAssertEqual(normalizationsTrained, [true])
        XCTAssertEqual(losses[0], batchStatistics, accuracy: abs(batchStatistics) * 1e-5)
        XCTAssertNotEqual(runningMean(), released,
                          "the run folds each batch into the running statistics")
        XCTAssertFalse(net.norm4.training, "the run returns the network to the mode it found it in")
    }

    func testTheReferenceScheduleStepsAt215And430ThousandOf500() {
        let schedule = NFKMLXColorizer.referenceSchedule(steps: 500)
        XCTAssertEqual(schedule.multiplier(214), 1)
        XCTAssertEqual(schedule.multiplier(215), 0.316, accuracy: 1e-6)
        XCTAssertEqual(schedule.multiplier(430), 0.316 * 0.316, accuracy: 1e-6)
    }
}
