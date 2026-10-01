//
//  NFKMLXAudioTaggerTrainingTests.swift
//  InferKitMLXTests
//
//  Retargeting the PANNs tagger: the objective, SpecAugment, the training-only dropout, the frozen base,
//  the retargeted classifier, AMSGrad's running maximum, the recipe, and the round trip with the
//  filterbank. Parity against PANNs' own recipe lives in NFKMLXReferenceParityTests.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXAudioTaggerTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_002)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func tiny(classCount: Int = 3) -> NFKMLXAudioTaggerConfiguration {
        var configuration = NFKMLXAudioTaggerConfiguration.tiny
        configuration.classCount = classCount
        return configuration
    }

    /// Half a second at the tiny configuration's 16 kHz: a tone, then noise.
    private func example(classCount: Int = 3) -> (samples: [Float], sampleRate: Int, targets: [Float]) {
        let samples = (0 ..< 8000).map { index -> Float in
            index < 4000 ? 0.3 * sinf(2 * .pi * 440 * Float(index) / 16000) : 0.05 * sinf(Float(index) * 12.9898)
        }
        return (samples, 16000, (0 ..< classCount).map { $0 == 1 ? 1 : 0 })
    }

    func testTheObjectiveIsTheClipBinaryCrossEntropy() throws {
        try requireMLXRuntime()
        let loss = NFKMLXAudioTaggerObjective().loss(logits: MLXArray([Float(2), -1, 0, 1], [2, 2]),
                                                     targets: MLXArray([Float(1), 0, 0, 1], [2, 2]))
        let sigmoid = { (x: Double) in 1 / (1 + exp(-x)) }
        let expected = -(log(sigmoid(2)) + log(1 - sigmoid(-1)) + log(1 - sigmoid(0)) + log(sigmoid(1))) / 4
        XCTAssertEqual(Double(loss.item(Float.self)), expected, accuracy: 1e-6)
    }

    func testSpecAugmentZeroesWholeStripesOnBothAxes() throws {
        try requireMLXRuntime()
        let augmentation = NFKMLXAudioTaggerSpecAugment(timeDropWidth: 8, timeStripes: 2, frequencyDropWidth: 4,
                                                        frequencyStripes: 2)
        let masked = augmentation(MLXArray.ones([1, 40, 16, 1])).reshaped([40, 16]).asArray(Float.self)
        var zeroed = 0
        for frame in 0 ..< 40 {
            for band in 0 ..< 16 where masked[frame * 16 + band] == 0 {
                zeroed += 1
                let wholeFrame = (0 ..< 16).allSatisfy { masked[frame * 16 + $0] == 0 }
                let wholeBand = (0 ..< 40).allSatisfy { masked[$0 * 16 + band] == 0 }
                XCTAssertTrue(wholeFrame || wholeBand, "a zero lies on a masked stripe")
            }
        }
        XCTAssertGreaterThan(zeroed, 0)
    }

    func testDropoutAndSpecAugmentActOnlyInTraining() throws {
        try requireMLXRuntime()
        let net = try NFKMLXAudioTagger.network(weightsURL: nil, configuration: tiny())
        let mel = net.frontEnd.logMel(example().samples)
        let first = net.logits(mel, augmentation: .reference), second = net.logits(mel)
        XCTAssertEqual(first.asArray(Float.self), second.asArray(Float.self))
        net.train(true)
        let noisy = net.logits(mel)
        net.train(false)
        XCTAssertGreaterThan(abs(noisy - first).max().item(Float.self), 0)
    }

    func testAClassifierRunLeavesTheBaseUntouched() throws {
        try requireMLXRuntime()
        let net = try NFKMLXAudioTagger.network(weightsURL: nil, configuration: tiny())
        let item = example()
        let base = net.parameters().flattened().filter { !$0.0.hasPrefix("fc_audioset.") }
            .map { ($0.0, $0.1.asArray(Float.self)) }
        let classifier = net.classifier.weight.asArray(Float.self)

        let history = try NFKMLXAudioTagger.fineTune(net, examples: { _ in item }, steps: 3)

        XCTAssertEqual(history.count, 3)
        let after = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        for (name, before) in base {
            XCTAssertEqual(after[name]?.asArray(Float.self), before, "\(name) stayed frozen, statistics included")
        }
        XCTAssertNotEqual(net.classifier.weight.asArray(Float.self), classifier, "the classifier trained")
        XCTAssertFalse(net.training)
    }

    func testTrainingDrivesTheLossDown() throws {
        try requireMLXRuntime()
        let net = try NFKMLXAudioTagger.network(weightsURL: nil, configuration: tiny())
        let item = example()
        let history = try NFKMLXAudioTagger.fineTune(net, examples: { _ in item }, augmentation: nil, steps: 30)
        XCTAssertLessThan(history.last!, history.first!, "\(history.first!) -> \(history.last!)")
    }

    func testRetargetingTheClassCountDropsTheCheckpointClassifier() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("audio-tagger-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let source = try NFKMLXAudioTagger.network(weightsURL: nil, configuration: tiny(classCount: 6))
        try NFKMLXAudioTagger.save(source, to: url)

        let retargeted = try NFKMLXAudioTagger.network(weightsURL: url, configuration: tiny(classCount: 2))
        XCTAssertEqual(retargeted.classifier.weight.dim(0), 2, "the classifier stayed at the consumer's classes")
        XCTAssertEqual(retargeted.fc1.weight.asArray(Float.self), source.fc1.weight.asArray(Float.self),
                       "while the base loaded")

        let kept = try NFKMLXAudioTagger.network(weightsURL: url, configuration: tiny(classCount: 6))
        XCTAssertEqual(kept.classifier.weight.asArray(Float.self), source.classifier.weight.asArray(Float.self),
                       "the same class set keeps the trained classifier")
    }

    func testAMSGradDividesByTheLargestSecondMomentSoFar() throws {
        try requireMLXRuntime()
        // A large gradient then a small one: the second step divides by the first step's second moment,
        // the larger, where Adam would divide by the decayed one.
        let layer = Linear(1, 1, bias: false)
        layer.update(parameters: ModuleParameters.unflattened([("weight", MLXArray.zeros([1, 1]))]))
        let optimizer = NFKMLXAMSGrad(learningRate: 0.1)
        for gradient in [Float(1), 0.01] {
            optimizer.update(model: layer, gradients: ModuleParameters.unflattened([("weight", MLXArray([gradient], [1, 1]))]))
        }
        let (b1, b2, rate) = (0.9, 0.999, 0.1)
        let m2 = 0.1 + (1 - b1) * (0.01 - 0.1)
        let largest = (1 - b2) * 1.0
        let second = rate / (1 - b1 * b1) * m2 / (largest.squareRoot() / (1 - b2 * b2).squareRoot() + 1e-8)
        XCTAssertEqual(Double(layer.weight.item(Float.self)), -rate - second, accuracy: 1e-6)
    }

    func testAFineTunedCheckpointRoundTripsWithItsFilterbank() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("audio-tagger-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXAudioTagger.network(weightsURL: nil, configuration: tiny())
        let item = example()
        try NFKMLXAudioTagger.fineTune(net, examples: { _ in item }, steps: 2)
        try NFKMLXAudioTagger.save(net, to: url)

        XCTAssertNotNil(try NFKMLXWeights.loadCheckpoint(url: url).arrays["logmel_extractor.melW"],
                        "the file carries the filterbank its loader reads back")
        let reloaded = try NFKMLXAudioTagger.network(weightsURL: url, configuration: tiny())
        let mel = net.frontEnd.logMel(item.samples)
        XCTAssertEqual(reloaded.frontEnd.logMel(item.samples).asArray(Float.self), mel.asArray(Float.self))
        XCTAssertEqual(reloaded.logits(mel).asArray(Float.self), net.logits(mel).asArray(Float.self))
    }
}
