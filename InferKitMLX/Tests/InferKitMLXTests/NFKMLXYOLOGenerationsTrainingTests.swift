//
//  NFKMLXYOLOGenerationsTrainingTests.swift
//  InferKitMLXTests
//
//  The later YOLO generations' fine-tune: both branches of an end-to-end head, the one-to-one branch's
//  detached features, the bias priors, the retarget, the recipe under each objective, and the round trip
//  through the release backend. The objectives' parity lives in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXYOLOGenerationsTrainingTests: XCTestCase {

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

    private func batch() -> (images: MLXArray, targets: [[NFKMLXYOLOBox]]) {
        (MLXRandom.uniform(low: 0, high: 1, [2, 64, 64, 3]),
         [[NFKMLXYOLOBox(classIndex: 0, x1: 10, y1: 12, x2: 46, y2: 50)],
          [NFKMLXYOLOBox(classIndex: 2, x1: 8, y1: 30, x2: 60, y2: 62)]])
    }

    func testAnEndToEndHeadYieldsBothBranches() throws {
        try requireMLXRuntime()
        let net = try NFKMLXYOLOGenerations.network(release: .v26Nano, classCount: 3, weightsURL: nil)
        let outputs = net.headOutputs(batch().images)
        XCTAssertEqual(outputs.oneToMany.distribution.shape, [2, 84, 4])
        XCTAssertEqual(outputs.oneToMany.logits.shape, [2, 84, 3])
        XCTAssertEqual(try XCTUnwrap(outputs.oneToOne).logits.shape, [2, 84, 3])
        let standard = try NFKMLXYOLOGenerations.network(release: .v11Nano, classCount: 3, weightsURL: nil)
        XCTAssertNil(standard.headOutputs(batch().images).oneToOne)
    }

    // The attention blocks reshaped with a batch of one, which inference never exceeds and training does.
    func testABatchMatchesItsImagesOneAtATime() throws {
        try requireMLXRuntime()
        for release in [NFKMLXYOLORelease.v26Nano, .v12Nano] {
            let net = try NFKMLXYOLOGenerations.network(release: release, classCount: 3, weightsURL: nil)
            let images = batch().images
            let together = net.headOutputs(images).oneToMany.logits
            let first = net.headOutputs(images[0 ..< 1]).oneToMany.logits
            let second = net.headOutputs(images[1 ..< 2]).oneToMany.logits
            XCTAssertLessThan(abs(together - concatenated([first, second], axis: 0)).max().item(Float.self), 1e-4, "\(release)")
        }
    }

    func testTheOneToOneBranchDoesNotTrainTheBackbone() throws {
        try requireMLXRuntime()
        let net = try NFKMLXYOLOGenerations.network(release: .v26Nano, classCount: 3, weightsURL: nil)
        let item = batch()
        let objective = NFKMLXYOLOObjective()
        let lossAndGradient = valueAndGrad(model: net) { net, _ in
            let outputs = net.headOutputs(item.images)
            let one = outputs.oneToOne!
            return [objective.loss(boxDistribution: one.distribution, classLogits: one.logits,
                                   featureSizes: outputs.featureSizes, strides: outputs.strides, targets: item.targets)]
        }
        let (_, gradients) = lossAndGradient(net, [])
        let flat = gradients.flattened()
        let backbone = flat.filter { $0.0.hasPrefix("model.0.") }
        let oneToOneHead = flat.filter { $0.0.contains("one2one_cv3") }
        XCTAssertFalse(backbone.isEmpty)
        XCTAssertEqual(backbone.map { abs($0.1).max().item(Float.self) }.max(), 0)
        XCTAssertGreaterThan(oneToOneHead.map { abs($0.1).max().item(Float.self) }.max() ?? 0, 0)
    }

    func testBothBranchesStartAtTheBiasPriors() throws {
        try requireMLXRuntime()
        let net = try NFKMLXYOLOGenerations.network(release: .v26Nano, classCount: 3, weightsURL: nil)
        let prior = logf(5 / 3 / powf(640 / 16, 2))
        let one = try XCTUnwrap(net.head.oneToOneClass?[1] as? NFKYOLOClassBranch)
        let many = try XCTUnwrap(net.head.cv3[1] as? NFKYOLOClassBranch)
        XCTAssertEqual(one.out.bias!.asArray(Float.self)[0], prior, accuracy: 1e-6)
        XCTAssertEqual(many.out.bias!.asArray(Float.self)[2], prior, accuracy: 1e-6)
        XCTAssertEqual(net.head.oneToOneBox?[0].out.bias!.asArray(Float.self), [Float](repeating: 2, count: 4))
    }

    func testARetargetTransfersEverythingButTheClassBranches() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("yolo26-coco-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let released = try NFKMLXYOLOGenerations.network(release: .v26Nano, classCount: 80, weightsURL: nil)
        try NFKMLXWeights.save(released, to: url)
        let retargeted = try NFKMLXYOLOGenerations.network(release: .v26Nano, classCount: 3, weightsURL: url)
        let before = Dictionary(uniqueKeysWithValues: released.parameters().flattened())
        let after = Dictionary(uniqueKeysWithValues: retargeted.parameters().flattened())
        XCTAssertEqual(after["model.0.conv.weight"]?.asArray(Float.self), before["model.0.conv.weight"]?.asArray(Float.self))
        XCTAssertEqual(try NFKMLXYOLOGenerations.classCount(in: url, release: .v26Nano), 80)
    }

    func testAStandardGenerationTrains() throws {
        try requireMLXRuntime()
        let net = try NFKMLXYOLOGenerations.network(release: .v11Nano, classCount: 3, weightsURL: nil)
        let item = batch()
        let history = try NFKMLXYOLOGenerations.fineTune(net, examples: { _ in item }, optimizer: AdamW(learningRate: 1e-3),
                                                         steps: 3, stepsPerEpoch: 3, learningRateSchedule: .constant,
                                                         averagesWeights: false)
        XCTAssertTrue(history.allSatisfy(\.isFinite))
        XCTAssertFalse(net.training)
    }

    func testAnEndToEndGenerationTrainsAndReloadsThroughTheBackend() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("yolo26-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let net = try NFKMLXYOLOGenerations.network(release: .v26Nano, classCount: 3, weightsURL: nil)
        let item = batch()
        let history = try NFKMLXYOLOGenerations.fineTune(net, examples: { _ in item }, steps: 2, stepsPerEpoch: 1)
        XCTAssertTrue(history.allSatisfy(\.isFinite))
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXYOLOGenerations.network(release: .v26Nano, classCount: 3, weightsURL: url)
        let images = item.images
        XCTAssertLessThan(abs(reloaded.headOutputs(images).oneToMany.logits - net.headOutputs(images).oneToMany.logits)
            .max().item(Float.self), 1e-4)
        XCTAssertEqual(try NFKMLXYOLOGenerations.classCount(in: url, release: .v26Nano), 3)
        XCTAssertTrue(try NFKMLXYOLOGenerations.backend(release: .v26Nano, weightsURL: url, labels: ["a", "b", "c"]).isReady)
    }
}
