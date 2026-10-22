//
//  NFKMLXZeroDCEPlusTrainingTests.swift
//  InferKitMLXTests
//
//  Fine-tuning Zero-DCE++. The four losses are Zero-DCE's and are checked there; these tests cover what
//  the successor changes (the weights, the shared curve map, the training scale factor) and run the
//  whole customization path: train on unlabeled photos, save, and load the result through the public
//  factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXZeroDCEPlusTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_021)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    func testTheObjectiveDefaultsAreTheTrainingScripts() {
        let objective = NFKMLXZeroDCEPlusObjective()
        XCTAssertEqual(objective.smoothnessWeight, 1600)
        XCTAssertEqual(objective.spatialWeight, 1)
        XCTAssertEqual(objective.colorWeight, 5)
        XCTAssertEqual(objective.exposureWeight, 10)
        XCTAssertEqual(objective.wellExposedLevel, 0.6)
        XCTAssertEqual(objective.scaleFactor, 1)
    }

    func testTheObjectiveSumsItsWeightedTermsOverTheSharedCurveMap() throws {
        try requireMLXRuntime()
        let net = NFKMLXZeroDCEPlusNet(filters: 8)
        let photos = NFKMLXZeroDCETrainingTests.darkPhotos(count: 2, size: 24)
        let objective = NFKMLXZeroDCEPlusObjective()

        let (enhanced, curve) = net.enhancedWithCurve(photos, scaleFactor: 1)
        XCTAssertEqual(curve.shape, [2, 24, 24, 3], "one three-channel map, applied at every iteration")
        let expected = 1600 * NFKMLXZeroDCELoss.illuminationSmoothness(curve)
            + NFKMLXZeroDCELoss.spatialConsistency(enhanced, original: photos)
            + 5 * NFKMLXZeroDCELoss.colorConstancy(enhanced)
            + 10 * NFKMLXZeroDCELoss.exposure(enhanced, wellExposedLevel: 0.6)
        XCTAssertEqual(objective(net, photos).item(Float.self), expected.item(Float.self), accuracy: 1e-5)
    }

    func testTheReleasedScaleFactorLiftsTheCurveMapToTheBatch() throws {
        try requireMLXRuntime()
        let net = NFKMLXZeroDCEPlusNet(filters: 8)
        let photos = NFKMLXZeroDCETrainingTests.darkPhotos(count: 1, size: 48)

        let (enhanced, curve) = net.enhancedWithCurve(photos, scaleFactor: 12)
        XCTAssertEqual(curve.shape, [1, 48, 48, 3])
        let inference = net(photos)
        eval(enhanced, inference)
        XCTAssertEqual(enhanced.asArray(Float.self), inference.asArray(Float.self),
                       "the network's own factor is 12, so its inference forward is the same pass")
    }

    func testAStepMovesEveryWeightAtEitherScaleFactor() throws {
        try requireMLXRuntime()
        let photos = NFKMLXZeroDCETrainingTests.darkPhotos(count: 2, size: 48)
        for scale in [1, 12] {
            let net = NFKMLXZeroDCEPlusNet(filters: 8)
            let before = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 * 1) })
            eval(Array(before.values))

            try NFKMLXZeroDCEPlus.fineTune(net, photos: { _ in photos },
                                           objective: NFKMLXZeroDCEPlusObjective(scaleFactor: scale), steps: 1)

            for (name, after) in net.parameters().flattened() {
                let moved = abs(after - before[name]!).max().item(Float.self)
                XCTAssertGreaterThan(moved, 0, "\(name) trains at scale \(scale)")
            }
        }
    }

    func testFineTuningLowersTheLossAndBrightens() throws {
        try requireMLXRuntime()
        let net = NFKMLXZeroDCEPlusNet(filters: 8)
        let photos = NFKMLXZeroDCETrainingTests.darkPhotos(count: 2, size: 32)
        let before = net(photos).mean().item(Float.self)

        let history = try NFKMLXZeroDCEPlus.fineTune(net, photos: { _ in photos },
                                                     optimizer: Adam(learningRate: 1e-3), steps: 80)

        XCTAssertEqual(history.count, 80)
        XCTAssertLessThan(history.last!, history.first!, "the objective fell: \(history.first!) -> \(history.last!)")
        XCTAssertGreaterThan(net(photos).mean().item(Float.self), before, "training on dark photos alone brightened the model")
    }

    func testAFineTunedCheckpointReloadsThroughThePublicFactories() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("zerodce-plus-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXZeroDCEPlus.network(weightsURL: nil)
        let photos = NFKMLXZeroDCETrainingTests.darkPhotos(count: 1, size: 48)
        try NFKMLXZeroDCEPlus.fineTune(net, photos: { _ in photos }, optimizer: Adam(learningRate: 1e-3), steps: 10)
        try NFKMLXWeights.save(net, to: url)

        // The saved file is in the module's layout, so the loader must skip the transpose it applies to the
        // released checkpoint. Reproducing the forward exactly is what proves it did.
        let reloaded = try NFKMLXZeroDCEPlus.network(weightsURL: url)
        let expected = net(photos), actual = reloaded(photos)
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self), "the fine-tuned checkpoint round-trips exactly")

        let plate = NFKMLXZeroDCETrainingTests.darkImage(48)
        let backend = try NFKMLXZeroDCEPlus.backend(weightsURL: url)
        let output = try NFKMLXZeroDCETrainingTests.cgImage(
            try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: plate])).output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 48, "enhancement keeps the input size")
    }
}
