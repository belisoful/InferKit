//
//  NFKMLXDeepLabTrainingTests.swift
//  InferKitMLXTests
//
//  Retargeting DeepLabV3: the criterion, the batch requirement, the training-only dropout, the frozen
//  backbone, the auxiliary head and its keys, the retargeted classifiers, the recipe, and the round trip.
//  Parity against torchvision's `references/segmentation` lives in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXDeepLabTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_003)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func tiny(classCount: Int = 4) -> NFKMLXDeepLabConfiguration {
        var configuration = NFKMLXDeepLabConfiguration.tiny
        configuration.classCount = classCount
        return configuration
    }

    /// Two textured 32×32 images with a two-region label map, a few pixels unscored.
    private func batch(classCount: Int = 4) -> (images: MLXArray, labels: MLXArray) {
        let size = 32
        let pixels = (0 ..< 2 * size * size * 3).map { Float(($0 * 29) % 200) / 255 }
        let labels = (0 ..< 2 * size * size).map { index -> Int32 in
            index % 97 == 0 ? 255 : Int32(((index % (size * size)) / size < size / 2 ? 1 : 2) % classCount)
        }
        return (MLXArray(pixels, [2, size, size, 3]), MLXArray(labels, [2, size, size]))
    }

    func testTheCriterionIgnoresLabel255AndHalvesTheAuxiliaryTerm() throws {
        try requireMLXRuntime()
        // Two classes. The main logits are level, so every scored pixel costs ln 2; the auxiliary
        // logits favor class 0 by 2, so a 0 costs ln(1 + e⁻²) and a 1 costs ln(1 + e²).
        let labels = MLXArray([Int32(0), 1, 255, 0, 1, 1, 255, 0], [2, 2, 2])
        let main = MLXArray.zeros([2, 2, 2, 2])
        let auxiliary = MLXArray(Array(repeating: [Float(1), -1], count: 8).flatMap { $0 }, [2, 2, 2, 2])
        let loss = NFKMLXDeepLabObjective().loss(logits: main, auxiliaryLogits: auxiliary, labels: labels)
        let auxiliaryTerm = (3 * log(1 + exp(-2.0)) + 3 * log(1 + exp(2.0))) / 6
        XCTAssertEqual(Double(loss.item(Float.self)), log(2) + 0.5 * auxiliaryTerm, accuracy: 1e-5)
    }

    func testAPooledNormalizationFoldsTheUnbiasedVarianceAsPyTorchDoes() throws {
        try requireMLXRuntime()
        // Two pooled values per channel, 1 and 3: the batch normalizes with their population variance,
        // 1, and folds their unbiased variance, 2, into the running variance at momentum 0.1.
        let norm = NFKTorchBatchNorm(featureCount: 1)
        norm.train(true)
        let output = norm(MLXArray([Float(1), 3], [2, 1, 1, 1]))
        let statistics = Dictionary(uniqueKeysWithValues: norm.parameters().flattened())
        XCTAssertEqual(try XCTUnwrap(statistics["running_var"]).item(Float.self), 0.9 + 0.1 * 2, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(statistics["running_mean"]).item(Float.self), 0.1 * 2, accuracy: 1e-6)
        XCTAssertEqual(output.reshaped([-1]).asArray(Float.self)[0], -1 / (1 + 1e-5).squareRoot(), accuracy: 1e-5)
        norm.train(false)
        let evaluated = norm(MLXArray([Float(1)], [1, 1, 1, 1])).item(Float.self)
        XCTAssertEqual(evaluated, (1 - 0.2) / (1.1 + 1e-5).squareRoot(), accuracy: 1e-5, "evaluation reads the running statistics")
    }

    func testAFineTuneNeedsABatchOfTwo() throws {
        try requireMLXRuntime()
        let net = try NFKMLXDeepLab.network(weightsURL: nil, configuration: tiny())
        let item = batch()
        XCTAssertThrowsError(try NFKMLXDeepLab.fineTune(net, examples: { _ in (item.images[0 ..< 1], item.labels[0 ..< 1]) },
                                                        steps: 1))
    }

    func testTheDropoutsActOnlyInTraining() throws {
        try requireMLXRuntime()
        let net = try NFKMLXDeepLab.network(weightsURL: nil, configuration: tiny())
        let images = NFKMLXDeepLabNet.normalized(batch().images)
        net.train(true)
        let first = net.trainingLogits(images), second = net.trainingLogits(images)
        net.train(false)
        XCTAssertGreaterThan(abs(first.main - second.main).max().item(Float.self), 0, "the ASPP dropout")
        XCTAssertGreaterThan(abs(first.auxiliary - second.auxiliary).max().item(Float.self), 0, "the auxiliary dropout")
        XCTAssertEqual(net.logits(images).asArray(Float.self), net.logits(images).asArray(Float.self))
    }

    func testAHeadRunLeavesTheBackboneAndTheAuxiliaryHeadUntouched() throws {
        try requireMLXRuntime()
        let net = try NFKMLXDeepLab.network(weightsURL: nil, configuration: tiny())
        let item = batch()
        let frozen = net.parameters().flattened().filter { $0.0.hasPrefix("backbone.") || $0.0.hasPrefix("auxiliary.") }
            .map { ($0.0, $0.1.asArray(Float.self)) }
        let classifier = net.classifier.weight.asArray(Float.self)

        let history = try NFKMLXDeepLab.fineTune(net, examples: { _ in item }, steps: 3)

        XCTAssertEqual(history.count, 3)
        let after = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        for (name, before) in frozen {
            XCTAssertEqual(after[name]?.asArray(Float.self), before, "\(name) stayed frozen, statistics included")
        }
        XCTAssertNotEqual(net.classifier.weight.asArray(Float.self), classifier, "the head trained")
        XCTAssertFalse(net.training)
    }

    func testTrainingEverythingMovesTheBackboneAndTheAuxiliaryHead() throws {
        try requireMLXRuntime()
        let net = try NFKMLXDeepLab.network(weightsURL: nil, configuration: tiny())
        let item = batch()
        let stem = net.backbone.conv1.weight.asArray(Float.self)
        let auxiliary = net.auxiliary.classifier.weight.asArray(Float.self)
        try NFKMLXDeepLab.fineTune(net, examples: { _ in item }, trainable: .everything, steps: 2)
        XCTAssertNotEqual(net.backbone.conv1.weight.asArray(Float.self), stem)
        XCTAssertNotEqual(net.auxiliary.classifier.weight.asArray(Float.self), auxiliary)
    }

    func testTrainingDrivesTheLossDown() throws {
        try requireMLXRuntime()
        let net = try NFKMLXDeepLab.network(weightsURL: nil, configuration: tiny())
        let item = batch()
        let history = try NFKMLXDeepLab.fineTune(net, examples: { _ in item }, optimizer: SGD(learningRate: 0.05, momentum: 0.9),
                                                 steps: 30)
        XCTAssertLessThan(history.last!, history.first!, "\(history.first!) -> \(history.last!)")
    }

    func testTheAuxiliaryHeadReadsTheReferenceKeys() {
        XCTAssertEqual(NFKMLXDeepLab.remapReferenceKey("aux_classifier.0.weight", poolBranch: 4), "auxiliary.conv.conv.weight")
        XCTAssertEqual(NFKMLXDeepLab.remapReferenceKey("aux_classifier.1.running_var", poolBranch: 4),
                       "auxiliary.conv.norm.running_var")
        XCTAssertEqual(NFKMLXDeepLab.remapReferenceKey("aux_classifier.4.bias", poolBranch: 4), "auxiliary.classifier.bias")
    }

    func testRetargetingTheClassCountDropsBothClassifiers() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("deeplab-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let source = try NFKMLXDeepLab.network(weightsURL: nil, configuration: tiny(classCount: 6))
        try NFKMLXWeights.save(source, to: url)

        let retargeted = try NFKMLXDeepLab.network(weightsURL: url, configuration: tiny(classCount: 3))
        XCTAssertEqual(retargeted.classifier.weight.dim(0), 3)
        XCTAssertEqual(retargeted.auxiliary.classifier.weight.dim(0), 3)
        XCTAssertEqual(retargeted.head.conv.weight.asArray(Float.self), source.head.conv.weight.asArray(Float.self),
                       "while everything else loaded")

        let kept = try NFKMLXDeepLab.network(weightsURL: url, configuration: tiny(classCount: 6))
        XCTAssertEqual(kept.classifier.weight.asArray(Float.self), source.classifier.weight.asArray(Float.self))
    }

    func testARetargetedCheckpointLabelsAtItsOwnClassCountThroughTheFactory() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("deeplab-factory-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let net = try NFKMLXDeepLab.network(weightsURL: nil, configuration: NFKMLXDeepLabConfiguration(classCount: 3))
        // Every pixel's top class is the last of three, which a three-class label map encodes as white.
        net.update(parameters: ModuleParameters.unflattened([("classifier.bias", MLXArray([Float(0), 0, 50]))]))
        try NFKMLXWeights.save(net, to: url)

        let backend = try NFKMLXDeepLab.backend(weightsURL: url)
        let image = NFKMLXLabelMapTesting.solid(64, 64)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
        XCTAssertEqual(try NFKMLXLabelMapTesting.levels(result.output(forKey: NFKOutputImage)), [255])
    }

    func testAFineTunedCheckpointRoundTrips() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("deeplab-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXDeepLab.network(weightsURL: nil, configuration: tiny())
        let item = batch()
        try NFKMLXDeepLab.fineTune(net, examples: { _ in item }, trainable: .everything, steps: 2)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXDeepLab.network(weightsURL: url, configuration: tiny())
        let image = item.images[0]
        XCTAssertEqual(reloaded.segment(image).asArray(Float.self), net.segment(image).asArray(Float.self))
    }
}
