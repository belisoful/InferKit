//
//  NFKMLXMossFormer2TrainingTests.swift
//  InferKitMLXTests
//
//  MossFormer2 SE's training path without released weights: the phase-sensitive target and its clamp,
//  the dropout's mode, the dithered features, and a fine-tune that lowers the loss and reloads through
//  both factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXMossFormer2TrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_004)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Two quarter-second clips of a tone under noise at 48 kHz, `[2, 12000]` each.
    private func pair() -> (noisy: MLXArray, clean: MLXArray) {
        let clean = (0 ..< 2).flatMap { row in
            (0 ..< 12_000).map { index in 0.1 * sinf(2 * .pi * Float(200 + 90 * row) * Float(index) / 48_000) }
        }
        let cleanArray = MLXArray(clean, [2, 12_000])
        return (cleanArray + MLXRandom.normal([2, 12_000], key: MLXRandom.key(3)) * 0.02, cleanArray)
    }

    /// A configuration small enough to run many forwards quickly.
    private func small() -> NFKMLXMossFormer2Configuration {
        NFKMLXMossFormer2Configuration(dModel: 32, numBlocks: 2, groupSize: 16, queryKeyDim: 16, fsmnHidden: 16)
    }

    func testTheTargetIsTheUnitMaskWhereTheCleanClipIsTheNoisyOne() throws {
        try requireMLXRuntime()
        let (noisy, _) = pair()
        let objective = NFKMLXMossFormer2Objective()
        let frames = NFKMLXKaldiFbank.frameCount(12_000, config: objective.configuration)
        let ones = MLXArray.ones([2, frames, 961]), zeros = MLXArray.zeros([2, frames, 961])
        XCTAssertLessThan(objective.loss(noisy: noisy, clean: noisy, mask: ones).item(Float.self), 1e-6)
        XCTAssertGreaterThan(objective.loss(noisy: noisy, clean: noisy, mask: zeros).item(Float.self), 1)
    }

    func testTheTargetClampsToZeroAndOne() throws {
        try requireMLXRuntime()
        let (noisy, _) = pair()
        let objective = NFKMLXMossFormer2Objective()
        let frames = NFKMLXKaldiFbank.frameCount(12_000, config: objective.configuration)
        let ones = MLXArray.ones([2, frames, 961]), zeros = MLXArray.zeros([2, frames, 961])
        XCTAssertLessThan(objective.loss(noisy: noisy, clean: noisy * 3, mask: ones).item(Float.self), 1e-6,
                          "a ratio of 9 is clamped to 1")
        XCTAssertLessThan(objective.loss(noisy: noisy, clean: -noisy, mask: zeros).item(Float.self), 1e-6,
                          "opposite phase is clamped to 0")
    }

    func testDropoutActsOnlyInTrainingAtEveryReferenceSite() throws {
        try requireMLXRuntime()
        let configuration = small()
        let net = NFKMLXMossFormer2Factory.makeNet(configuration)
        let dropouts = net.modules().compactMap { $0 as? Dropout }
        XCTAssertEqual(dropouts.count, configuration.numBlocks * 6, "five FFConvM outputs and the attention per block")
        let feature = NFKMLXMossFormer2Factory.features(pair().noisy, configuration: configuration)
        XCTAssertEqual(abs(net(feature) - net(feature)).max().item(Float.self), 0, "a built network evaluates")
        net.train(true)
        XCTAssertGreaterThan(abs(net(feature) - net(feature)).max().item(Float.self), 0, "training drops")
    }

    func testDitherAddsNoiseDrawnFromTheKey() throws {
        try requireMLXRuntime()
        let (noisy, _) = pair()
        let plain = NFKMLXMossFormer2Factory.features(noisy)
        let first = NFKMLXMossFormer2Factory.features(noisy, dither: 1, key: MLXRandom.key(1))
        let again = NFKMLXMossFormer2Factory.features(noisy, dither: 1, key: MLXRandom.key(1))
        let other = NFKMLXMossFormer2Factory.features(noisy, dither: 1, key: MLXRandom.key(2))
        XCTAssertEqual(abs(first - again).max().item(Float.self), 0)
        XCTAssertGreaterThan(abs(first - other).max().item(Float.self), 0)
        XCTAssertGreaterThan(abs(first - plain).max().item(Float.self), 0)
        XCTAssertEqual(plain.shape, [2, NFKMLXKaldiFbank.frameCount(12_000, config: NFKMLXMossFormer2Configuration()), 180])
    }

    func testFineTuningLowersTheLossAndReloadsThroughBothFactories() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXMossFormer2Configuration()
        configuration.dropout = 0
        let net = try NFKMLXMossFormer2Factory.network(weightsURL: nil, configuration: configuration)
        let (noisy, clean) = pair()
        let objective = NFKMLXMossFormer2Objective()
        let feature = NFKMLXMossFormer2Factory.features(noisy)
        let before = objective(net, feature, noisy, clean).item(Float.self)
        let losses = try NFKMLXMossFormer2Factory.fineTune(net, examples: { _ in (noisy, clean) }, steps: 4)
        XCTAssertEqual(losses.count, 4)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the batch it trains on")
        XCTAssertTrue(net.modules().allSatisfy { !$0.training }, "the run restores evaluation mode")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXMossFormer2Factory.network(weightsURL: url)
        XCTAssertLessThan(abs(net(feature) - reloaded(feature)).max().item(Float.self), 1e-5)
        XCTAssertNoThrow(try NFKMLXMossFormer2Factory.backend(weightsURL: url))
    }

    func testTheReferenceOptimizerIsAdamWithoutDecay() throws {
        try requireMLXRuntime()
        let net = NFKMLXMossFormer2Factory.makeNet(small())
        let optimizer = NFKMLXMossFormer2Factory.referenceOptimizer()
        let before = Dictionary(uniqueKeysWithValues: net.trainableParameters().flattened().map { ($0.0, $0.1 * 1) })
        optimizer.update(model: net, gradients: net.trainableParameters().mapValues { MLXArray.zeros(like: $0) })
        eval(net)
        let after = Dictionary(uniqueKeysWithValues: net.trainableParameters().flattened())
        // With a zero gradient and no decay, nothing moves.
        for (name, value) in before {
            XCTAssertEqual(abs(after[name]! - value).max().item(Float.self), 0, name)
        }
    }
}
