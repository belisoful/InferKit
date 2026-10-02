//
//  NFKMLXRIFEv4TrainingTests.swift
//  InferKitMLXTests
//
//  RIFE v4's fine-tune without released weights: the batched warp, the training pass against inference, the
//  objective's terms, the schedule and scales, the target encoder, and a fine-tune that lowers the loss and
//  reloads through both factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXRIFEv4TrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_009)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Two smooth frames and the one between them, `[N, 64, 64, 3]` in [0, 1].
    private func triplet(count: Int = 2) -> (MLXArray, MLXArray, MLXArray) {
        func frame(_ shift: Float) -> MLXArray {
            let y = MLXArray(0 ..< 64).asType(.float32).reshaped([1, 64, 1, 1]) / 64
            let x = MLXArray(0 ..< 64).asType(.float32).reshaped([1, 1, 64, 1]) / 64 + shift
            let channel = MLXArray([Float(0), 1.3, 2.6]).reshaped([1, 1, 1, 3])
            let base = 0.5 + 0.4 * sin(6 * x + channel) * cos(5 * y + channel)
            return broadcast(base, to: [count, 64, 64, 3])
        }
        return (frame(0), frame(0.06), frame(0.03))
    }

    func testTheBatchedWarpMatchesEachFrameWarpedAlone() throws {
        try requireMLXRuntime()
        let images = MLXRandom.uniform(low: 0, high: 1, [2, 12, 10, 3], key: MLXRandom.key(1))
        let flow = MLXRandom.normal([2, 12, 10, 2], key: MLXRandom.key(2)) * 3
        let batched = NFKMLXRIFENet.warp(images, flow: flow)
        for index in 0 ..< 2 {
            let alone = NFKMLXRIFENet.warp(images[index ..< index + 1], flow: flow[index ..< index + 1])
            XCTAssertEqual(abs(batched[index ..< index + 1] - alone).max().item(Float.self), 0)
        }
    }

    func testTheTrainingPassAtTheInferenceScalesFinishesOnTheInterpolatedFrame() throws {
        try requireMLXRuntime()
        let net = try NFKMLXRIFEv4.network(weightsURL: nil)
        let (frame0, frame1, _) = triplet()
        let pass = net.trainingPass(frame0, frame1, timestep: MLXArray([Float(0.5), 0.5]), scales: [8, 4, 2, 1])
        XCTAssertEqual(pass.flows.count, 4)
        XCTAssertEqual(pass.merged.count, 4)
        XCTAssertEqual(pass.teacherFlow.shape, [2, 64, 64, 4])
        for index in 0 ..< 2 {
            let interpolated = net.interpolate(frame0[index ..< index + 1], frame1[index ..< index + 1], timestep: 0.5)
            XCTAssertLessThan(abs(pass.merged[3][index ..< index + 1] - interpolated).max().item(Float.self), 1e-6)
        }
    }

    func testTheObjectiveSumsItsWeightedTerms() throws {
        try requireMLXRuntime()
        let net = try NFKMLXRIFEv4.network(weightsURL: nil)
        let objective = try NFKMLXRIFEv4Objective(vggWeightsURL: nil, net: net)
        let (frame0, frame1, middle) = triplet()
        let pass = net.trainingPass(frame0, frame1, timestep: MLXArray([Float(0.5), 0.5]), scales: [4, 2, 1, 1])
        let terms = objective.components(pass, target: middle)
        let expected = terms.perceptual + 0.1 * terms.teacher + terms.consistency + 0.05 * terms.l1
        XCTAssertEqual(objective.loss(pass, target: middle).item(Float.self), expected.item(Float.self), accuracy: 1e-6)
        let l1 = pass.merged.reduce(MLXArray(Float(0))) { ($0 + abs($1 - middle).mean()) * 0.8 }
        XCTAssertEqual(terms.l1.item(Float.self), l1.item(Float.self), accuracy: 1e-7)
        XCTAssertEqual(objective.ssim(middle, middle).item(Float.self), 1, accuracy: 1e-5)
        XCTAssertEqual(objective.vgg(middle, middle).item(Float.self), 0)
        XCTAssertEqual(objective.encoderDistance(middle, middle).item(Float.self), 0)
        XCTAssertFalse(terms.consistency.item(Float.self).isNaN)
    }

    func testTheEncoderFeaturesAreTheActivationsTheReferencesInPlaceReLULeaves() throws {
        try requireMLXRuntime()
        let net = try NFKMLXRIFEv4.network(weightsURL: nil)
        let (frame0, _, _) = triplet()
        let features = net.encode.features(frame0)
        XCTAssertEqual(features.count, 4)
        let first = leakyRelu(net.encode.cnn0(frame0), negativeSlope: 0.2)
        XCTAssertEqual(abs(features[0] - first).max().item(Float.self), 0)
        XCTAssertEqual(abs(features[3] - net.encode(frame0)).max().item(Float.self), 0)
    }

    func testTheTargetEncoderMovesOnePercentTowardTheNetwork() throws {
        try requireMLXRuntime()
        let net = try NFKMLXRIFEv4.network(weightsURL: nil)
        let objective = try NFKMLXRIFEv4Objective(vggWeightsURL: nil, net: net)
        // `update(parameters:)` writes into the arrays it replaces, so the starting values are copies.
        let start = Dictionary(uniqueKeysWithValues: objective.targetEncoder.parameters().flattened().map { ($0.0, $0.1 * 1) })
        let network = Dictionary(uniqueKeysWithValues: net.encode.parameters().flattened())
        for (key, value) in start {
            XCTAssertEqual(abs(value - network[key]!).max().item(Float.self), 0, "\(key) starts as the network's")
        }
        net.encode.update(parameters: ModuleParameters.unflattened(network.map { ($0.key, $0.value + 1) }))
        objective.updateTarget(from: net)
        for (key, value) in objective.targetEncoder.parameters().flattened() {
            XCTAssertLessThan(abs(value - (start[key]! + 0.01)).max().item(Float.self), 1e-6, key)
        }
    }

    func testTheReferenceScheduleWarmsUpThenFollowsACosine() {
        let schedule = NFKMLXRIFEv4.referenceSchedule(steps: 6000)
        XCTAssertEqual(schedule.multiplier(0), 1e-3, accuracy: 1e-9)
        XCTAssertEqual(schedule.multiplier(1000), 0.5 + 1e-3, accuracy: 1e-6)
        XCTAssertEqual(schedule.multiplier(2000), 1 + 1e-3, accuracy: 1e-6)
        XCTAssertEqual(schedule.multiplier(4000), 0.5 + 1e-3, accuracy: 1e-6)
        XCTAssertEqual(schedule.multiplier(6000), 1e-3, accuracy: 1e-6)
    }

    func testTheScalesFollowTheDraw() {
        XCTAssertEqual(NFKMLXRIFEv4.referenceScales(draw: 0.1), [4, 2, 1, 1])
        XCTAssertEqual(NFKMLXRIFEv4.referenceScales(draw: 0.3), [2, 1, 1, 1])
        XCTAssertEqual(NFKMLXRIFEv4.referenceScales(draw: 0.6), [8, 4, 2, 1])
    }

    func testFineTuningLowersTheLossMovesTheTargetAndReloads() throws {
        try requireMLXRuntime()
        let net = try NFKMLXRIFEv4.network(weightsURL: nil)
        let objective = try NFKMLXRIFEv4Objective(vggWeightsURL: nil, net: net)
        let (frame0, frame1, middle) = triplet(count: 1)
        let timestep = MLXArray([Float(0.5)])
        let target = Dictionary(uniqueKeysWithValues: objective.targetEncoder.parameters().flattened().map { ($0.0, $0.1 * 1) })
        let before = objective(net, frame0, frame1, middle, timestep: timestep).item(Float.self)
        let losses = try NFKMLXRIFEv4.fineTune(net, examples: { _ in (frame0, frame1, middle, timestep) }, objective: objective,
                                               optimizer: Adam(learningRate: 1e-4), steps: 4)
        XCTAssertEqual(losses.count, 4)
        XCTAssertTrue(losses.allSatisfy { $0.isFinite })
        XCTAssertLessThan(objective(net, frame0, frame1, middle, timestep: timestep).item(Float.self), before)
        let moved = objective.targetEncoder.parameters().flattened().map { abs($0.1 - target[$0.0]!).max().item(Float.self) }
        XCTAssertGreaterThan(moved.max()!, 0, "the target encoder follows the network")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXRIFEv4.network(weightsURL: url)
        XCTAssertLessThan(abs(net.interpolate(frame0, frame1) - reloaded.interpolate(frame0, frame1)).max().item(Float.self), 1e-6)
        XCTAssertNoThrow(try NFKMLXRIFEv4.backend(weightsURL: url))
    }
}
