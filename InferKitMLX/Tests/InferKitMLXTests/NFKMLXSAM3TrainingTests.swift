//
//  NFKMLXSAM3TrainingTests.swift
//  InferKitMLXTests
//
//  Fine-tuning SAM 3's detector onto a consumer's own notion of an instance. Three things here are
//  load-bearing beyond the loop: the assignment is a selection and carries no gradient, an image the
//  prompt names nothing in still trains the presence head, and a fine-tuned detector reloads through
//  its own loader.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXSAM3TrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_922)
    }

    /// A tiny detector: the released one is 25M parameters over a 72x72 level, which no test can
    /// train. The query count stays small so the assignment is checkable by hand.
    private func smallDetector() -> NFKMLXSAM3DetectorNet {
        NFKMLXSAM3DetectorNet(NFKMLXSAM3DetectorConfiguration(
            hiddenSize: 32, intermediateSize: 48, heads: 2, encoderLayers: 1, decoderLayers: 2,
            queryCount: 8, upsamplingStages: 3))
    }

    /// What the frozen encoders would have produced: three levels, their position encodings, and a
    /// projected prompt.
    private func encoded(width: Int = 32)
        -> (levels: [MLXArray], positions: [MLXArray], prompt: MLXArray, valid: MLXArray) {
        let sides = [16, 8, 4]
        let levels = sides.map { MLXRandom.normal([1, $0, $0, width]) * 0.2 }
        let positions = sides.map {
            NFKMLXSAM2PositionEmbedding.sine(height: $0, width: $0, features: width)
        }
        let prompt = MLXRandom.normal([1, 5, width]) * 0.2
        let valid = MLXArray([Float(1), 1, 1, 0, 0]).reshaped([1, 5])
        eval(levels, positions, prompt, valid)
        return (levels, positions, prompt, valid)
    }

    /// Two boxes in the centre convention, well inside the frame.
    private func targets() -> MLXArray {
        MLXArray([Float(0.3), 0.3, 0.2, 0.2, 0.7, 0.6, 0.25, 0.3]).reshaped([2, 4])
    }

    // MARK: - The matcher

    func testTheAssignmentIsOneToOneAndWithinRange() throws {
        try requireMLXRuntime()
        let detector = smallDetector()
        let inputs = encoded()
        let detected = detector(levels: inputs.levels, positions: inputs.positions,
                                prompt: inputs.prompt, promptValid: inputs.valid)
        let centres = NFKSAM3BoxMetrics.cornersToCenter(detected.boxes)
        let objective = NFKMLXSAM3Objective()
        let matched = objective.match(logits: detected.logits, boxes: centres, targets: targets())
        XCTAssertEqual(matched.count, 2, "every target is assigned a query")
        XCTAssertEqual(Set(matched).count, 2, "no query answers two targets")
        for query in matched {
            XCTAssertTrue((0 ..< 8).contains(query), "the assigned query is in range")
        }
    }

    /// The Hungarian assignment minimizes the total cost, which is the only property the objective
    /// depends on; a greedy pick does not have it.
    func testTheAssignmentMinimizesTheTotalCost() throws {
        try requireMLXRuntime()
        // A cost matrix whose greedy choice is not its optimum: row 0 prefers column 0, but taking
        // it forces row 1 onto a far worse column.
        let cost: [Float] = [1, 2,
                             2, 90,
                             50, 60]
        let assignment = NFKMLXHungarian.assign(cost: cost, rows: 3, columns: 2)
        XCTAssertEqual(assignment, [1, 0],
                       "the optimum pairs target 0 with query 1 and target 1 with query 0")
    }

    // MARK: - The objective

    func testAnImageWithNoTargetTrainsOnlyThePresenceHead() throws {
        try requireMLXRuntime()
        let objective = NFKMLXSAM3Objective()
        let parts = objective.components(logits: MLXRandom.normal([1, 8]),
                                         boxes: MLXRandom.uniform(low: 0.2, high: 0.6, [1, 8, 4]),
                                         presence: MLXArray([Float(1.5)]).reshaped([1, 1]),
                                         targets: MLXArray.zeros([0, 4]), matched: [])
        eval(parts.classification, parts.presence, parts.box, parts.generalizedIoU)
        XCTAssertEqual(parts.classification.item(Float.self), 0, accuracy: 1e-7)
        XCTAssertEqual(parts.box.item(Float.self), 0, accuracy: 1e-7)
        XCTAssertEqual(parts.generalizedIoU.item(Float.self), 0, accuracy: 1e-7)
        XCTAssertGreaterThan(parts.presence.item(Float.self), 0,
                             "an image the prompt names nothing in still teaches the model to say no")
    }

    // MARK: - The run

    func testAFineTuneLowersTheLoss() throws {
        try requireMLXRuntime()
        let detector = smallDetector()
        let inputs = encoded()
        let boxes = targets()
        let losses = try NFKMLXSAM3.fineTune(
            detector,
            examples: { _ in (inputs.levels, inputs.positions, inputs.prompt, inputs.valid, boxes) },
            optimizer: AdamW(learningRate: 1e-3), steps: 12)
        XCTAssertEqual(losses.count, 12)
        XCTAssertLessThan(losses.suffix(3).reduce(0, +) / 3, losses.prefix(3).reduce(0, +) / 3,
                          "the loss falls over the run")
    }

    func testTheCheaperLevelLeavesTheDetrEncoderUntouched() throws {
        try requireMLXRuntime()
        let detector = smallDetector()
        let inputs = encoded()
        let boxes = targets()

        func snapshot(_ module: Module) -> [Float] {
            module.parameters().flattened().flatMap { $0.1.reshaped([-1]).asArray(Float.self) }
        }
        let encoderBefore = snapshot(detector.encoder)
        let decoderBefore = snapshot(detector.decoder)

        try NFKMLXSAM3.fineTune(
            detector,
            examples: { _ in (inputs.levels, inputs.positions, inputs.prompt, inputs.valid, boxes) },
            trainable: .decoder, optimizer: AdamW(learningRate: 1e-3), steps: 6)

        XCTAssertEqual(snapshot(detector.encoder), encoderBefore, "the DETR encoder stays frozen")
        XCTAssertNotEqual(snapshot(detector.decoder), decoderBefore, "the decoder trains")
    }

    // MARK: - The whole customization path

    func testAFineTunedDetectorRoundTrips() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sam3-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let detector = smallDetector()
        let inputs = encoded()
        let boxes = targets()
        try NFKMLXSAM3.fineTune(
            detector,
            examples: { _ in (inputs.levels, inputs.positions, inputs.prompt, inputs.valid, boxes) },
            optimizer: AdamW(learningRate: 1e-3), steps: 4)
        try NFKMLXWeights.save(detector, to: url)

        // The saved file carries module names, so it loads through the same path a release does —
        // with the prefix a release keeps its detector under.
        let reloaded = smallDetector()
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value) }, to: reloaded)

        let expected = detector(levels: inputs.levels, positions: inputs.positions,
                                prompt: inputs.prompt, promptValid: inputs.valid)
        let actual = reloaded(levels: inputs.levels, positions: inputs.positions,
                              prompt: inputs.prompt, promptValid: inputs.valid)
        eval(expected.boxes, actual.boxes)
        XCTAssertEqual(actual.boxes.reshaped([-1]).asArray(Float.self),
                       expected.boxes.reshaped([-1]).asArray(Float.self),
                       "the fine-tuned detector reproduces its boxes exactly")
    }
}
