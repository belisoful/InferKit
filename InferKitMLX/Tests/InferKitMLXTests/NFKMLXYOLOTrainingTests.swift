//
//  NFKMLXYOLOTrainingTests.swift
//  InferKitMLXTests
//
//  YOLOv8's fine-tune: the raw head outputs, the retarget to a consumer's class count, the reference
//  optimizer and schedule, the weight average, the recipe, and the round trip through the backend.
//  The objective's parity lives in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXYOLOTrainingTests: XCTestCase {

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

    /// Two 64-pixel images with a box each.
    private func batch() -> (images: MLXArray, targets: [[NFKMLXYOLOBox]]) {
        (MLXRandom.uniform(low: 0, high: 1, [2, 64, 64, 3]),
         [[NFKMLXYOLOBox(classIndex: 0, x1: 10, y1: 12, x2: 46, y2: 50)],
          [NFKMLXYOLOBox(classIndex: 2, x1: 8, y1: 30, x2: 60, y2: 62)]])
    }

    func testTheHeadOutputsCoverEveryAnchorOfEveryScale() throws {
        try requireMLXRuntime()
        let net = NFKMLXYOLONet(.tiny)
        let outputs = net.headOutputs(MLXRandom.uniform(low: 0, high: 1, [2, 64, 64, 3]))
        XCTAssertEqual(outputs.strides, [8, 16, 32])
        XCTAssertEqual(outputs.featureSizes.map(\.height), [8, 4, 2])
        XCTAssertEqual(outputs.distribution.shape, [2, 84, 4 * NFKMLXYOLOConfiguration.tiny.regMax])
        XCTAssertEqual(outputs.logits.shape, [2, 84, NFKMLXYOLOConfiguration.tiny.classCount])
    }

    func testTheHeadBiasesStartAtUltralyticsPriors() throws {
        try requireMLXRuntime()
        let net = NFKMLXYOLONet(.tiny)
        net.initializeHeadBiases()
        let classes = Float(NFKMLXYOLOConfiguration.tiny.classCount)
        XCTAssertEqual(net.detect.cv2[0].out.bias!.asArray(Float.self), [Float](repeating: 2, count: 4 * NFKMLXYOLOConfiguration.tiny.regMax))
        XCTAssertEqual(net.detect.cv3[2].out.bias!.asArray(Float.self)[0], logf(5 / classes / powf(640 / 32, 2)), accuracy: 1e-6)
    }

    func testARetargetTransfersEverythingButTheClassBranches() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("yolo-coco-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let released = try NFKMLXYOLO.network(classCount: 80, weightsURL: nil)
        try NFKMLXWeights.save(released, to: url)

        let retargeted = try NFKMLXYOLO.network(classCount: 3, weightsURL: url)
        XCTAssertEqual(retargeted.conv0.conv.weight.asArray(Float.self), released.conv0.conv.weight.asArray(Float.self))
        XCTAssertEqual(retargeted.detect.cv2[1].out.weight.asArray(Float.self), released.detect.cv2[1].out.weight.asArray(Float.self))
        XCTAssertEqual(retargeted.detect.cv3[0].out.weight.shape.first, 3)
        XCTAssertEqual(retargeted.detect.cv3[0].out.bias!.asArray(Float.self)[0], logf(5 / 3 / powf(80, 2)), accuracy: 1e-6)
    }

    func testTheReferenceOptimizerFollowsAutoForAShortRun() throws {
        try requireMLXRuntime()
        let optimizer = NFKMLXYOLO.referenceOptimizer(classCount: 3, batchSize: 2)
        let rates = try XCTUnwrap(NFKMLXLearningRateSchedule.scheduledGroups(of: optimizer)).map(\.1)
        XCTAssertEqual(rates, [0.001429, 0.001429])
    }

    func testTheScheduleWarmsUpThenFallsLinearlyByEpoch() {
        let schedule = NFKMLXLearningRateSchedule.ultralytics(steps: 20, stepsPerEpoch: 5)
        XCTAssertEqual(schedule.multiplier(0), 0)
        XCTAssertEqual(schedule.multiplier(5), Float(5.0 / 15 * (0.75 * 0.99 + 0.01)), accuracy: 1e-6)
        XCTAssertEqual(schedule.multiplier(15), Float(0.25 * 0.99 + 0.01), accuracy: 1e-6)
        XCTAssertEqual(schedule.multiplier(19), Float(0.25 * 0.99 + 0.01), accuracy: 1e-6)
    }

    func testAHeadOnlyRunLeavesTheBackboneAndStatisticsUntouched() throws {
        try requireMLXRuntime()
        let net = NFKMLXYOLONet(.tiny)
        let item = batch()
        eval(net)
        let backbone = net.conv0.conv.weight.asArray(Float.self)
        func statistics() -> [Float] {
            Dictionary(uniqueKeysWithValues: net.parameters().flattened())["conv0.bn.running_mean"]!.asArray(Float.self)
        }
        let before = statistics()
        let head = net.detect.cv3[0].out.weight.asArray(Float.self)
        let history = try NFKMLXYOLO.fineTune(net, examples: { _ in item }, trainable: .head,
                                              optimizer: AdamW(learningRate: 1e-2), steps: 3, stepsPerEpoch: 3,
                                              learningRateSchedule: .constant, averagesWeights: false)
        XCTAssertTrue(history.allSatisfy(\.isFinite))
        XCTAssertEqual(net.conv0.conv.weight.asArray(Float.self), backbone)
        XCTAssertEqual(statistics(), before)
        XCTAssertNotEqual(net.detect.cv3[0].out.weight.asArray(Float.self), head)
        XCTAssertFalse(net.training)
    }

    func testAFullRunLowersTheLoss() throws {
        try requireMLXRuntime()
        let net = NFKMLXYOLONet(.tiny)
        net.initializeHeadBiases()
        let item = batch()
        let history = try NFKMLXYOLO.fineTune(net, examples: { _ in item }, optimizer: AdamW(learningRate: 3e-3),
                                              steps: 12, stepsPerEpoch: 12, learningRateSchedule: .constant,
                                              averagesWeights: false)
        XCTAssertLessThan(history.suffix(3).reduce(0, +), history.prefix(3).reduce(0, +))
    }

    func testTheWeightAverageFollowsModelEMA() throws {
        try requireMLXRuntime()
        let linear = Linear(3, 2)
        linear.update(parameters: ModuleParameters.unflattened(["weight": MLXArray.ones([2, 3]), "bias": MLXArray.zeros([2])]))
        var average = NFKYOLOWeightAverage(linear)
        for step in 0 ..< 3 {
            linear.update(parameters: ModuleParameters.unflattened(["weight": MLXArray.ones([2, 3]) * Float(step + 2),
                                                                    "bias": MLXArray.ones([2]) * Float(-(step + 1))]))
            average.update(from: linear)
        }
        average.apply(to: linear)
        var expected: Double = 1
        for step in 0 ..< 3 {
            let decay = 0.9999 * (1 - exp(-Double(step + 1) / 2000))
            expected = expected * decay + Double(step + 2) * (1 - decay)
        }
        XCTAssertEqual(linear.weight.asArray(Float.self)[0], Float(expected), accuracy: 1e-5)
    }

    func testAFineTunedCheckpointLoadsAtItsOwnClassCount() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("yolo-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXYOLO.network(classCount: 3, weightsURL: nil)
        let item = batch()
        try NFKMLXYOLO.fineTune(net, examples: { _ in item }, steps: 2, stepsPerEpoch: 2)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXYOLO.network(classCount: 3, weightsURL: url)
        let images = item.images
        XCTAssertLessThan(abs(reloaded.headOutputs(images).logits - net.headOutputs(images).logits).max().item(Float.self), 1e-4)
        XCTAssertEqual(try NFKMLXYOLO.classCount(in: url), 3)
        XCTAssertTrue(try NFKMLXYOLO.backend(variant: .nano, weightsURL: url, labels: ["a", "b", "c"]).isReady)
    }
}
