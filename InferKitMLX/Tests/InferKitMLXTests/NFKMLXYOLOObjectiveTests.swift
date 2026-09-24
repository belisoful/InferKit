//
//  NFKMLXYOLOObjectiveTests.swift
//  InferKitMLXTests
//
//  The YOLO objective's parts on constructed inputs: the anchor grid, CIoU, the task-aligned assigner's
//  edge cases, the empty-image case, and a gradient through every term. Parity against ultralytics'
//  `v8DetectionLoss` lives in NFKMLXReferenceParityTests.
//

import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXYOLOObjectiveTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_924)
    }

    private let sizes = [(height: 8, width: 8), (height: 4, width: 4), (height: 2, width: 2)]
    private let strides = [8, 16, 32]

    /// Head outputs whose boxes sit near two bins per side, so each box overlaps its neighborhood.
    private func outputs(batch: Int = 1, classes: Int = 3) -> (distribution: MLXArray, logits: MLXArray) {
        let bins = MLXArray((0 ..< 16).map { -0.5 * powf(Float($0) - 2.2, 2) })
        let distribution = broadcast(bins.reshaped([1, 1, 1, 16]), to: [batch, 84, 4, 16])
            + 0.3 * MLXRandom.normal([batch, 84, 4, 16])
        return (distribution.reshaped([batch, 84, 64]), MLXRandom.normal([batch, 84, classes]))
    }

    func testTheAnchorGridIsScaleByScaleRowMajor() throws {
        try requireMLXRuntime()
        let (points, anchorStrides) = NFKMLXYOLOObjective.anchors(featureSizes: sizes, strides: strides)
        XCTAssertEqual(points.shape, [84, 2])
        XCTAssertEqual(Array(points.asArray(Float.self).prefix(4)), [0.5, 0.5, 1.5, 0.5])
        XCTAssertEqual(anchorStrides.asArray(Float.self)[64], 16)
        XCTAssertEqual(Array(points.asArray(Float.self)[(80 * 2) ..< (80 * 2 + 2)]), [0.5, 0.5])
    }

    func testCompleteIoUIsOneForIdenticalBoxesAndPenalizesDistance() throws {
        try requireMLXRuntime()
        let box = MLXArray([Float(1), 1, 5, 7]).reshaped([1, 4])
        XCTAssertEqual(NFKMLXYOLOObjective.completeIoU(box, box).item(Float.self), 1, accuracy: 1e-5)
        XCTAssertEqual(NFKYOLOTaskAlignedAssigner.completeIoU([1, 1, 5, 7], [1, 1, 5, 7]), 1, accuracy: 1e-5)
        let apart = NFKYOLOTaskAlignedAssigner.completeIoU([0, 0, 4, 4], [10, 10, 14, 14])
        XCTAssertLessThan(apart, 0, "no overlap and distant centers score below zero")
    }

    func testABoxSmallerThanTheFirstStrideStillGetsAnchors() throws {
        try requireMLXRuntime()
        let (distribution, logits) = outputs()
        let tiny = NFKMLXYOLOBox(classIndex: 2, x1: 40, y1: 6, x2: 45, y2: 14)
        let terms = NFKMLXYOLOObjective().components(boxDistribution: distribution, classLogits: logits,
                                                     featureSizes: sizes, strides: strides, targets: [[tiny]])
        XCTAssertGreaterThan(terms.box.item(Float.self), 0, "the grown box has candidates, so the box term is live")
    }

    func testAnAnchorClaimedTwiceGoesToTheBoxItOverlapsMost() {
        let assigner = NFKYOLOTaskAlignedAssigner(topK: 10, secondTopK: nil, alpha: 0.5, beta: 6, strides: [8], classes: 2)
        // One anchor at (12, 12); two boxes contain it, and its prediction matches the second.
        let result = assigner.assign(scores: [0.5, 0.5], boxes: [10, 10, 20, 20], anchors: [12, 12],
                                     batch: 1, anchorCount: 1,
                                     targets: [[NFKMLXYOLOBox(classIndex: 0, x1: 0, y1: 0, x2: 16, y2: 16),
                                                NFKMLXYOLOBox(classIndex: 1, x1: 9, y1: 9, x2: 21, y2: 21)]])
        XCTAssertEqual(result.foreground, [0])
        XCTAssertEqual(result.foregroundBoxes, [9, 9, 21, 21])
        XCTAssertGreaterThan(result.scores[1], 0)
        XCTAssertEqual(result.scores[0], 0)
    }

    func testAnImageWithNoBoxesScoresOnlyTheClasses() throws {
        try requireMLXRuntime()
        let (distribution, logits) = outputs()
        let terms = NFKMLXYOLOObjective().components(boxDistribution: distribution, classLogits: logits,
                                                     featureSizes: sizes, strides: strides, targets: [[]])
        XCTAssertEqual(terms.box.item(Float.self), 0)
        XCTAssertEqual(terms.dfl.item(Float.self), 0)
        let expected = 0.5 * (maximum(logits, 0) + log1p(exp(-abs(logits)))).sum().item(Float.self)
        XCTAssertEqual(terms.classification.item(Float.self), expected, accuracy: expected * 1e-5)
    }

    func testTheGradientReachesTheBoxesAndTheClasses() throws {
        try requireMLXRuntime()
        let (distribution, logits) = outputs(batch: 2)
        let targets = [[NFKMLXYOLOBox(classIndex: 0, x1: 10, y1: 12, x2: 46, y2: 50)],
                       [NFKMLXYOLOBox(classIndex: 1, x1: 8, y1: 30, x2: 60, y2: 62)]]
        let objective = NFKMLXYOLOObjective()
        let gradients = grad { (inputs: [MLXArray]) -> MLXArray in
            objective.loss(boxDistribution: inputs[0], classLogits: inputs[1], featureSizes: self.sizes,
                           strides: self.strides, targets: targets)
        }([distribution, logits])
        for gradient in gradients {
            XCTAssertTrue(gradient.asArray(Float.self).allSatisfy(\.isFinite))
            XCTAssertGreaterThan(abs(gradient).max().item(Float.self), 0)
        }
    }
}
