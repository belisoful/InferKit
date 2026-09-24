//
//  NFKMLXVADTrainingTests.swift
//  InferKitMLXTests
//
//  MarbleNet's full fine-tune: the objective, the SpecAugment masks, the frame labels, the schedule's
//  shape, the training-only dropout, the recipe, and the round trip through the model's own loader.
//  Parity of the objective and the schedule against the release lives in NFKMLXReferenceParityTests.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXVADTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_924)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// A second at 16 kHz: a voiced stack for the first half, quiet noise after.
    private func clip() -> [Float] {
        (0 ..< 16000).map { index in
            let t = Float(index) / 16000
            guard t < 0.5 else { return 0.02 * sinf(Float(index) * 12.9898) }
            return (1 ... 5).reduce(Float(0)) { $0 + 0.3 / Float($1) * sinf(2 * .pi * 150 * Float($1) * t) }
        }
    }

    private func example() -> (samples: [Float], labels: [Int32]) {
        let samples = clip()
        let frames = NFKMLXVAD.frameCount(samples: samples.count)
        return (samples, NFKMLXVAD.frameLabels(speech: [(0, 0.5)], frameCount: frames))
    }

    func testTheFrameCountMatchesTheNetwork() throws {
        try requireMLXRuntime()
        let net = try NFKMLXVAD.network(weightsURL: nil)
        let samples = clip()
        XCTAssertEqual(net.logits(net.frontEnd.logMel(samples)).dim(1), NFKMLXVAD.frameCount(samples: samples.count))
    }

    func testFramesAreLabeledByTheirMidpoints() {
        XCTAssertEqual(NFKMLXVAD.frameLabels(speech: [(0.02, 0.07)], frameCount: 5), [0, 1, 1, 0, 0])
    }

    func testTheObjectiveIsTheMaskedMeanCrossEntropy() throws {
        try requireMLXRuntime()
        let logits = MLXArray([Float(2), 0, 0, 2, 1, 1]).reshaped([1, 3, 2])
        let labels = MLXArray([Int32(0), 0, 1]).reshaped([1, 3])
        let objective = NFKMLXVADObjective()
        let each = [log(1 + exp(Float(-2))), log(1 + exp(Float(2))), log(Float(2))]
        XCTAssertEqual(objective.loss(logits: logits, labels: labels).item(Float.self),
                       each.reduce(0, +) / 3, accuracy: 1e-6)
        let mask = MLXArray([Int32(1), 0, 1]).reshaped([1, 3])
        XCTAssertEqual(objective.loss(logits: logits, labels: labels, mask: mask).item(Float.self),
                       (each[0] + each[2]) / 2, accuracy: 1e-6)
    }

    func testSpecAugmentZeroesWholeBandsAndSpans() throws {
        try requireMLXRuntime()
        let mel = MLXArray.ones([1, 200, 80])
        let masked = NFKMLXVADSpecAugment.reference(mel)
        let zeros = (masked .== 0)
        let fullColumns = zeros.all(axis: 1).sum().item(Int.self)       // masked frequency bands
        let fullRows = zeros.all(axis: 2).sum().item(Int.self)          // masked time spans
        XCTAssertLessThanOrEqual(fullColumns, 5 * 9)
        XCTAssertLessThanOrEqual(fullRows, 5 * 9)
        // Every zero sits in a masked band or a masked span.
        let covered = logicalOr(zeros.all(axis: 1, keepDims: true), zeros.all(axis: 2, keepDims: true))
        XCTAssertEqual(logicalAnd(zeros, logicalNot(covered)).sum().item(Int.self), 0)
    }

    func testTheScheduleWarmsHoldsAndDecaysToItsFloor() {
        let schedule = NFKMLXLearningRateSchedule.nemoPolynomialHoldDecay(steps: 40, warmupRatio: 0.05, holdRatio: 0.15,
                                                                          power: 2, minimumScale: 1e-6)
        XCTAssertEqual(schedule.multiplier(0), 1.0 / 3, accuracy: 1e-6)
        XCTAssertEqual(schedule.multiplier(2), 1)
        XCTAssertEqual(schedule.multiplier(7), 1)
        XCTAssertEqual(schedule.multiplier(8), 1)
        XCTAssertEqual(schedule.multiplier(24), Float(0.999999 * 0.25 + 1e-6), accuracy: 1e-6)
        XCTAssertEqual(schedule.multiplier(44), 1e-6, accuracy: 1e-9)
    }

    func testTheDropoutActsOnlyInTraining() throws {
        try requireMLXRuntime()
        let net = try NFKMLXVAD.network(weightsURL: nil)
        let mel = net.frontEnd.logMel(clip())
        let first = net.logits(mel), second = net.logits(mel)
        XCTAssertEqual(first.asArray(Float.self), second.asArray(Float.self))
        net.train(true)
        let noisy = net.logits(mel)
        net.train(false)
        XCTAssertGreaterThan(abs(noisy - first).max().item(Float.self), 0)
    }

    func testFineTuningRunsAndRestoresEvaluationMode() throws {
        try requireMLXRuntime()
        let net = try NFKMLXVAD.network(weightsURL: nil)
        let item = example()
        let history = try NFKMLXVAD.fineTune(net, examples: { _ in item }, steps: 4)
        XCTAssertEqual(history.count, 4)
        XCTAssertTrue(history.allSatisfy(\.isFinite))
        XCTAssertFalse(net.training)
    }

    func testAFineTunedCheckpointLoadsThroughTheLoader() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vad-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXVAD.network(weightsURL: nil)
        let item = example()
        try NFKMLXVAD.fineTune(net, examples: { _ in item }, steps: 2)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXVAD.network(weightsURL: url)
        let mel = net.frontEnd.logMel(item.samples)
        XCTAssertLessThan(abs(reloaded.logits(mel) - net.logits(mel)).max().item(Float.self), 1e-5)
        XCTAssertTrue(try NFKMLXVAD.backend(weightsURL: url).isReady)
    }
}
