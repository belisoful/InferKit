//
//  NFKMLXAllInOneTrainingTests.swift
//  InferKitMLXTests
//
//  All-In-One's full fine-tune: the annotation-to-frame targets, the training-only dropouts, the
//  recipe's freezing, and the round trip through the model's own loader. Parity of the targets, the
//  objective, and timm's RAdam against the authors' sources lives in NFKMLXReferenceParityTests.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXAllInOneTrainingTests: XCTestCase {

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

    private func example(frames: Int = 200) throws -> (spectrograms: MLXArray, targets: NFKMLXAllInOneTargets) {
        let spectrograms = MLXRandom.uniform(low: 0, high: 2, [1, 4, frames, 81])
        let targets = try NFKMLXAllInOneTargets(beatTimes: stride(from: 0.0, to: 2.0, by: 0.5).map { $0 },
                                                downbeatTimes: [0, 1], sectionBoundaries: [0, 1.2],
                                                sectionLabels: ["start", "verse", "chorus"], frameCount: frames)
        return (spectrograms, targets)
    }

    func testWideningHalvesTheNeighborsAndQuartersTheSecondRing() {
        let events: [Float] = [0, 0, 0, 1, 0, 0, 0]
        XCTAssertEqual(NFKMLXAllInOneTargets.widened(events, neighbors: 1), [0, 0, 0.5, 1, 0.5, 0, 0])
        XCTAssertEqual(NFKMLXAllInOneTargets.widened(events, neighbors: 2), [0, 0.25, 0.5, 1, 0.5, 0.25, 0])
        XCTAssertEqual(NFKMLXAllInOneTargets.widened([1, 0, 0], neighbors: 1), [1, 0.5, 0])
    }

    func testTimesLandOnLibrosasFrames() {
        let c = NFKMLXAllInOneConfiguration.harmonix
        XCTAssertEqual(NFKMLXAllInOneTargets.frame(0.505, configuration: c), 50)
        XCTAssertEqual(NFKMLXAllInOneTargets.frame(1.4999, configuration: c), 149)
        XCTAssertEqual(NFKMLXAllInOneTargets.frame(0, configuration: c), 0)
    }

    func testEachFrameTakesTheLabelOfTheLastBoundaryAtOrBeforeIt() throws {
        try requireMLXRuntime()
        let targets = try NFKMLXAllInOneTargets(beatTimes: [], downbeatTimes: [], sectionBoundaries: [0.02, 0.05],
                                                sectionLabels: ["start", "verse", "end"], frameCount: 7)
        let labels = NFKMLXAllInOneConfiguration.harmonix.labels
        let expected = ["start", "start", "verse", "verse", "verse", "end", "end"].map { Int32(labels.firstIndex(of: $0)!) }
        XCTAssertEqual(targets.function.asArray(Int32.self), expected)
        XCTAssertEqual(targets.section.asArray(Float.self), [0.25, 0.5, 1, 0.5, 0.5, 1, 0.5])
    }

    func testMismatchedOrUnknownLabelsAreRefused() {
        XCTAssertThrowsError(try NFKMLXAllInOneTargets(beatTimes: [], downbeatTimes: [], sectionBoundaries: [1],
                                                       sectionLabels: ["start"], frameCount: 10))
        XCTAssertThrowsError(try NFKMLXAllInOneTargets(beatTimes: [], downbeatTimes: [], sectionBoundaries: [1],
                                                       sectionLabels: ["start", "refrain"], frameCount: 10))
    }

    func testTheDropoutsActOnlyInTraining() throws {
        try requireMLXRuntime()
        let net = try NFKMLXAllInOne.network(weightsURL: nil)
        let input = try example(frames: 64).spectrograms
        let first = net(input).beat, second = net(input).beat
        XCTAssertEqual(first.asArray(Float.self), second.asArray(Float.self), "evaluation is deterministic")
        net.train(true)
        let noisy = net(input).beat
        net.train(false)
        XCTAssertGreaterThan(abs(noisy - first).max().item(Float.self), 0, "training draws dropout")
    }

    func testFineTuningTrainsTheNetworkAndLeavesTheFrontEndFixed() throws {
        try requireMLXRuntime()
        let net = try NFKMLXAllInOne.network(weightsURL: nil)
        let item = try example()
        eval(net.frontEnd.filterbank, net.beatHead.classifier.weight)
        let filterbank = net.frontEnd.filterbank.asArray(Float.self)
        let head = net.beatHead.classifier.weight.asArray(Float.self)
        let history = try NFKMLXAllInOne.fineTune(net, examples: { _ in item }, steps: 3)
        XCTAssertEqual(history.count, 3)
        XCTAssertTrue(history.allSatisfy(\.isFinite))
        XCTAssertEqual(net.frontEnd.filterbank.asArray(Float.self), filterbank)
        XCTAssertNotEqual(net.beatHead.classifier.weight.asArray(Float.self), head)
        XCTAssertFalse(net.training, "the run leaves the dropouts off")
    }

    func testAFineTunedCheckpointLoadsThroughTheLoader() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("allin1-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXAllInOne.network(weightsURL: nil)
        let item = try example()
        try NFKMLXAllInOne.fineTune(net, examples: { _ in item }, steps: 2)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXAllInOne.network(weightsURL: url)
        let expected = net(item.spectrograms), actual = reloaded(item.spectrograms)
        XCTAssertLessThan(abs(actual.beat - expected.beat).max().item(Float.self), 1e-5)
        XCTAssertLessThan(abs(actual.function - expected.function).max().item(Float.self), 1e-5)
        XCTAssertTrue(try NFKMLXAllInOne.backend(weightsURL: url).isReady)
    }
}
