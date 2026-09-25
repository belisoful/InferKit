//
//  NFKMLXRTDetrObjectiveTests.swift
//  InferKitMLXTests
//
//  The RT-DETR objective's parts on constructed inputs: the matcher, an exact prediction, the
//  denoising matching's group layout, the empty-image case, and a gradient through every term. Parity
//  against transformers' `RTDetrLoss` lives in NFKMLXReferenceParityTests.
//

import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXRTDetrObjectiveTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_924)
    }

    private let truth = NFKMLXRTDetrTarget(classes: [2, 0],
                                           boxes: MLXArray([0.3, 0.3, 0.2, 0.2, 0.7, 0.6, 0.3, 0.4] as [Float], [2, 4]))

    /// Six queries of three classes, with the two truths planted at queries 4 and 1 and confident there.
    private func planted() -> NFKMLXRTDetrPredictions {
        var boxes: [Float] = []
        for _ in 0 ..< 6 {
            boxes += [0.5, 0.5, 0.1, 0.1]
        }
        boxes.replaceSubrange(16 ..< 20, with: [0.3, 0.3, 0.2, 0.2])
        boxes.replaceSubrange(4 ..< 8, with: [0.7, 0.6, 0.3, 0.4])
        var logits = [Float](repeating: -8, count: 18)
        logits[4 * 3 + 2] = 8
        logits[1 * 3 + 0] = 8
        return NFKMLXRTDetrPredictions(logits: MLXArray(logits, [1, 6, 3]), boxes: MLXArray(boxes, [1, 6, 4]))
    }

    func testTheMatcherFindsThePlantedQueries() throws {
        try requireMLXRuntime()
        let predictions = planted()
        let matches = NFKMLXRTDetrObjective().match(logits: predictions.logits[0], boxes: predictions.boxes[0],
                                                     target: truth)
        XCTAssertEqual(matches.map(\.query), [1, 4])
        XCTAssertEqual(matches.map(\.target), [1, 0])
    }

    func testAnExactPredictionLeavesOnlyTheClassTerm() throws {
        try requireMLXRuntime()
        let losses = NFKMLXRTDetrObjective().losses(final: planted(), targets: [truth])
        XCTAssertEqual(try XCTUnwrap(losses["loss_bbox"]).item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(losses["loss_giou"]).item(Float.self), 0, accuracy: 1e-5)
        XCTAssertLessThan(try XCTUnwrap(losses["loss_vfl"]).item(Float.self), 1e-2)
    }

    func testDenoisingQueriesMatchTheTruthTheyWereNoisedFrom() throws {
        try requireMLXRuntime()
        // Two groups of 2 · 2 queries: the positives are 0, 1 and 4, 5; each group's copy of the truth
        // sits exactly on its positives, so the matching by position scores zero box error.
        var boxes = [Float](repeating: 0.5, count: 8 * 4)
        for group in 0 ..< 2 {
            boxes.replaceSubrange((group * 4) * 4 ..< (group * 4 + 2) * 4,
                                  with: [0.3, 0.3, 0.2, 0.2, 0.7, 0.6, 0.3, 0.4])
        }
        let denoising = NFKMLXRTDetrPredictions(logits: MLXRandom.normal([1, 8, 3]), boxes: MLXArray(boxes, [1, 8, 4]))
        let losses = NFKMLXRTDetrObjective().losses(final: planted(), denoising: [denoising],
                                                     denoisingPositives: [[0, 1, 4, 5]], denoisingGroups: 2,
                                                     targets: [truth])
        XCTAssertEqual(try XCTUnwrap(losses["loss_bbox_dn_0"]).item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(losses["loss_giou_dn_0"]).item(Float.self), 0, accuracy: 1e-5)
    }

    func testAnImageWithoutBoxesScoresOnlyItsClasses() throws {
        try requireMLXRuntime()
        let empty = NFKMLXRTDetrTarget(classes: [], boxes: MLXArray.zeros([0, 4]))
        let losses = NFKMLXRTDetrObjective().losses(final: planted(), targets: [empty])
        XCTAssertEqual(try XCTUnwrap(losses["loss_bbox"]).item(Float.self), 0)
        XCTAssertEqual(try XCTUnwrap(losses["loss_giou"]).item(Float.self), 0)
        XCTAssertGreaterThan(try XCTUnwrap(losses["loss_vfl"]).item(Float.self), 0)
    }

    func testTheGradientReachesLogitsAndBoxes() throws {
        try requireMLXRuntime()
        let logits = MLXRandom.normal([1, 6, 3])
        let boxes = concatenated([0.3 + 0.4 * MLXRandom.uniform(0 ..< 1, [1, 6, 2]),
                                  0.1 + 0.2 * MLXRandom.uniform(0 ..< 1, [1, 6, 2])], axis: -1)
        let objective = NFKMLXRTDetrObjective()
        let gradients = grad { (inputs: [MLXArray]) -> MLXArray in
            objective.loss(final: NFKMLXRTDetrPredictions(logits: inputs[0], boxes: inputs[1]), targets: [self.truth])
        }([logits, boxes])
        for gradient in gradients {
            let values = gradient.asArray(Float.self)
            XCTAssertTrue(values.allSatisfy(\.isFinite))
            XCTAssertGreaterThan(values.map(abs).max() ?? 0, 0)
        }
    }
}
