//
//  NFKMLXRetinaFaceTrainingTests.swift
//  InferKitMLXTests
//
//  RetinaFace's training path without released weights: the prior matching's rules, the batched forward
//  against the inference one, the reference schedule, and a fine-tune that lowers the loss and reloads
//  through every factory.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXRetinaFaceTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_005)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func face(_ x1: Float, _ y1: Float, _ x2: Float, _ y2: Float, landmarks: Bool = true) -> NFKMLXRetinaFaceAnnotation {
        let (w, h) = (x2 - x1, y2 - y1)
        let points: [Float] = [x1 + 0.3 * w, y1 + 0.35 * h, x1 + 0.7 * w, y1 + 0.35 * h, x1 + 0.5 * w, y1 + 0.55 * h,
                               x1 + 0.35 * w, y1 + 0.75 * h, x1 + 0.65 * w, y1 + 0.75 * h]
        return NFKMLXRetinaFaceAnnotation(x1: x1, y1: y1, x2: x2, y2: y2, landmarks: landmarks ? points : nil)
    }

    func testMatchingFollowsTheReferenceRules() throws {
        try requireMLXRuntime()
        let priors = NFKMLXRetinaFace.anchors(height: 160, width: 160, configuration: NFKMLXRetinaFaceConfiguration())
            .asArray(Float.self)
        let objective = NFKMLXRetinaFaceObjective()
        let faces = [face(0.30, 0.30, 0.75, 0.85), face(0.05, 0.60, 0.25, 0.95, landmarks: false),
                     face(0.90, 0.02, 0.92, 0.04)]
        let matched = objective.match(faces, priors: priors, variance: [0.1, 0.2])
        XCTAssertTrue(matched.labels.contains(1), "a face with landmarks labels its priors 1")
        XCTAssertTrue(matched.labels.contains(-1), "a face without landmarks labels its priors −1")
        XCTAssertTrue(Set(matched.labels).isSubset(of: [-1, 0, 1]))
        // The tiny face overlaps no prior by 0.2 and is never labeled; alone, it leaves only background.
        XCTAssertTrue(objective.match([faces[2]], priors: priors, variance: [0.1, 0.2]).labels.allSatisfy { $0 == 0 })
    }

    func testTheBatchedForwardMatchesTheInferenceOne() throws {
        try requireMLXRuntime()
        let net = try NFKMLXRetinaFace.network(weightsURL: nil)
        let images = MLXRandom.uniform(low: 0, high: 1, [2, 96, 96, 3], key: MLXRandom.key(4))
        let batched = net.logits(NFKMLXRetinaFace.trainingInput(images))
        let single = net(NFKMLXRetinaFace.prepared(images[1]))
        XCTAssertLessThan(abs(softmax(batched.logits[1], axis: -1) - single.scores).max().item(Float.self), 1e-5)
        XCTAssertLessThan(abs(batched.boxes[1] - single.boxes).max().item(Float.self), 1e-4)
        XCTAssertLessThan(abs(batched.landmarks[1] - single.landmarks).max().item(Float.self), 1e-4)
    }

    func testTheReferenceScheduleCutsTheRateAtEpochs190And220Of250() {
        let schedule = NFKMLXRetinaFace.referenceSchedule(steps: 250)
        XCTAssertEqual(schedule.multiplier(189), 1)
        XCTAssertEqual(schedule.multiplier(190), 0.1, accuracy: 1e-7)
        XCTAssertEqual(schedule.multiplier(220), 0.01, accuracy: 1e-8)
        XCTAssertEqual(NFKMLXRetinaFace.referenceSchedule(steps: 1).multiplier(0), 1, "a short run starts at the base rate")
    }

    func testFineTuningLowersTheLossAndReloadsThroughEveryFactory() throws {
        try requireMLXRuntime()
        let net = try NFKMLXRetinaFace.network(weightsURL: nil)
        let images = MLXRandom.uniform(low: 0, high: 1, [2, 96, 96, 3], key: MLXRandom.key(5))
        let faces = [[face(0.2, 0.2, 0.6, 0.7)], [face(0.4, 0.3, 0.9, 0.8), face(0.05, 0.05, 0.3, 0.35, landmarks: false)]]
        let objective = NFKMLXRetinaFaceObjective()
        net.train(true)
        let before = objective(net, NFKMLXRetinaFace.trainingInput(images), faces).item(Float.self)
        net.train(false)
        let losses = try NFKMLXRetinaFace.fineTune(net, examples: { _ in (images, faces) },
                                                   optimizer: Adam(learningRate: 1e-3), steps: 4)
        XCTAssertEqual(losses.count, 4)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the batch it trains on")
        XCTAssertTrue(net.modules().allSatisfy { !$0.training }, "the run restores evaluation mode")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXRetinaFace.network(weightsURL: url)
        let input = NFKMLXRetinaFace.trainingInput(images)
        XCTAssertLessThan(abs(net.logits(input).logits - reloaded.logits(input).logits).max().item(Float.self), 1e-5)
        XCTAssertNoThrow(try NFKMLXRetinaFace.detector(weightsURL: url))
        XCTAssertNoThrow(try NFKMLXRetinaFace.backend(weightsURL: url))
    }

    func testTheReferenceOptimizerDecaysEveryParameter() throws {
        try requireMLXRuntime()
        let net = try NFKMLXRetinaFace.network(weightsURL: nil)
        let optimizer = NFKMLXRetinaFace.referenceOptimizer()
        let before = Dictionary(uniqueKeysWithValues: net.trainableParameters().flattened().map { ($0.0, $0.1 * 1) })
        optimizer.update(model: net, gradients: net.trainableParameters().mapValues { MLXArray.zeros(like: $0) })
        eval(net)
        let after = Dictionary(uniqueKeysWithValues: net.trainableParameters().flattened())
        // With a zero gradient only the decay moves a parameter, and every nonzero one moves.
        for (name, value) in before where abs(value).max().item(Float.self) > 0 {
            XCTAssertGreaterThan(abs(after[name]! - value).max().item(Float.self), 0, name)
        }
    }
}
