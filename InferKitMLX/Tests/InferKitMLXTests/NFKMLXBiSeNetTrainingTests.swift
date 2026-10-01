//
//  NFKMLXBiSeNetTrainingTests.swift
//  InferKitMLXTests
//
//  Retargeting BiSeNet V1: the OHEM loss's two branches, the schedule, the batch requirement, the
//  reference optimizer's groups, the frozen ResNet-18, the retargeted classifiers, the crop, the recipe,
//  and the round trip. Parity against CoinCheung's training code lives in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXBiSeNetTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_004)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func tiny(classCount: Int = 4) -> NFKMLXBiSeNetConfiguration {
        NFKMLXBiSeNetConfiguration(baseChannels: 8, classCount: classCount)
    }

    /// Two textured images with a two-region label map, a few pixels unscored.
    private func batch(size: Int = 64, classCount: Int = 4) -> (images: MLXArray, labels: MLXArray) {
        let pixels = (0 ..< 2 * size * size * 3).map { Float(($0 * 29) % 200) / 255 }
        let labels = (0 ..< 2 * size * size).map { index -> Int32 in
            index % 97 == 0 ? 255 : Int32(((index % (size * size)) / size < size / 2 ? 1 : 2) % classCount)
        }
        return (MLXArray(pixels, [2, size, size, 3]), MLXArray(labels, [2, size, size]))
    }

    /// Two-class logits for pixels labeled 0, each pixel's margin `a` costing `ln(1 + e⁻ᵃ)`.
    private func margins(_ values: [Float]) -> MLXArray {
        MLXArray(values.flatMap { [$0, 0] }, [1, 1, values.count, 2])
    }

    func testTheOHEMLossAveragesTheHardPixelsWhenThereAreEnough() throws {
        try requireMLXRuntime()
        // 32 scored pixels need two hard ones. Three are hard (margins −1, 0, 0.2, all above −log 0.7).
        let values = [Float](repeating: 5, count: 29) + [-1, 0, 0.2]
        let loss = NFKMLXBiSeNetObjective().ohem(margins(values), MLXArray.zeros([1, 1, 32], type: Int32.self))
        let cost = { (a: Double) in log(1 + exp(-a)) }
        XCTAssertEqual(Double(loss.item(Float.self)), (cost(-1) + cost(0) + cost(0.2)) / 3, accuracy: 1e-5)
    }

    func testTheOHEMLossTakesTheLargestWhenTooFewAreHard() throws {
        try requireMLXRuntime()
        // 33 scored pixels need two hard ones and only one is; the largest easy one joins it. The
        // ignored pixel's huge cost counts neither toward the pixels nor the mean.
        let values = [Float](repeating: 5, count: 31) + [-1, 2, -20]
        var labels = [Int32](repeating: 0, count: 34)
        labels[33] = 255
        let loss = NFKMLXBiSeNetObjective().ohem(margins(values), MLXArray(labels, [1, 1, 34]))
        let cost = { (a: Double) in log(1 + exp(-a)) }
        XCTAssertEqual(Double(loss.item(Float.self)), (cost(-1) + cost(2)) / 2, accuracy: 1e-5)
    }

    func testTheScheduleWarmsExponentiallyThenDecays() {
        let schedule = NFKMLXLearningRateSchedule.exponentialWarmupPoly(steps: 1500, power: 0.9, warmupSteps: 1000,
                                                                        warmupRatio: 0.1)
        XCTAssertEqual(schedule.multiplier(0), 0.1, accuracy: 1e-7)
        XCTAssertEqual(schedule.multiplier(500), Float(0.1.squareRoot()), accuracy: 1e-7)
        XCTAssertEqual(schedule.multiplier(1000), 1, accuracy: 1e-7)
        XCTAssertEqual(schedule.multiplier(1250), Float(pow(0.5, 0.9)), accuracy: 1e-7)
        XCTAssertEqual(schedule.multiplier(1500), 0)
    }

    func testAFineTuneNeedsABatchOfTwo() throws {
        try requireMLXRuntime()
        let net = try NFKMLXBiSeNet.network(weightsURL: nil, configuration: tiny())
        let item = batch()
        XCTAssertThrowsError(try NFKMLXBiSeNet.fineTune(net, examples: { _ in (item.images[0 ..< 1], item.labels[0 ..< 1]) },
                                                        steps: 1))
    }

    func testTheReferenceOptimizerBoostsTheFusionAndHeadsAndDecaysOnlyConvolutionWeights() throws {
        try requireMLXRuntime()
        let net = try NFKMLXBiSeNet.network(weightsURL: nil, configuration: tiny())
        let item = batch()
        net.train(true)
        let images = NFKMLXBiSeNetNet.normalized(item.images)
        let gradients = valueAndGrad(model: net) { net, _ in
            let logits = net.trainingLogits(images)
            return [NFKMLXBiSeNetObjective().loss(logits: logits.main, auxiliaryLogits: logits.auxiliary,
                                                  labels: item.labels)]
        }(net, []).1
        let before = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 * 1) })
        eval(Array(before.values))

        NFKMLXBiSeNet.referenceOptimizer(for: net).update(model: net, gradients: gradients)
        net.train(false)

        let after = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        let gradient = Dictionary(uniqueKeysWithValues: gradients.flattened())
        // From zero momentum, SGD's first step is `rate · (g + decay · θ)`.
        for (name, rate, decay) in [("ffm.convblk.conv.weight", Float(0.1), Float(5e-4)), ("ffm.bn.weight", 0.1, 0),
                                    ("cp.arm16.conv.conv.weight", 0.01, 5e-4), ("sp.conv1.bn.bias", 0.01, 0),
                                    ("conv_out.conv_out.bias", 0.1, 0)] {
            let start = try XCTUnwrap(before[name], name)
            let expected = start - rate * (try XCTUnwrap(gradient[name], name) + decay * start)
            XCTAssertLessThan(abs(try XCTUnwrap(after[name], name) - expected).max().item(Float.self), 1e-6, name)
        }
    }

    func testAnAllButBackboneRunLeavesTheResNetUntouched() throws {
        try requireMLXRuntime()
        let net = try NFKMLXBiSeNet.network(weightsURL: nil, configuration: tiny())
        let item = batch()
        let resnet = net.cp.resnet.parameters().flattened().map { $0.1.asArray(Float.self) }
        let spatial = net.sp.conv1.conv.weight.asArray(Float.self)

        let history = try NFKMLXBiSeNet.fineTune(net, examples: { _ in item }, steps: 3)

        XCTAssertEqual(history.count, 3)
        for (after, before) in zip(net.cp.resnet.parameters().flattened().map { $0.1.asArray(Float.self) }, resnet) {
            XCTAssertEqual(after, before, "the frozen ResNet-18 did not move, statistics included")
        }
        XCTAssertNotEqual(net.sp.conv1.conv.weight.asArray(Float.self), spatial, "the spatial path trained")
        XCTAssertFalse(net.training)
    }

    func testABatchIsCroppedToMultiplesOf32() throws {
        try requireMLXRuntime()
        let net = try NFKMLXBiSeNet.network(weightsURL: nil, configuration: tiny())
        let item = batch(size: 70)
        let history = try NFKMLXBiSeNet.fineTune(net, examples: { _ in item }, steps: 1)
        XCTAssertTrue(history.allSatisfy(\.isFinite))
    }

    func testTrainingDrivesTheLossDown() throws {
        try requireMLXRuntime()
        let net = try NFKMLXBiSeNet.network(weightsURL: nil, configuration: tiny())
        let item = batch()
        let history = try NFKMLXBiSeNet.fineTune(net, examples: { _ in item }, optimizer: SGD(learningRate: 0.05, momentum: 0.9),
                                                 steps: 30)
        XCTAssertLessThan(history.last!, history.first!, "\(history.first!) -> \(history.last!)")
    }

    func testRetargetingTheClassCountDropsTheThreeClassifiers() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("bisenet-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let source = try NFKMLXBiSeNet.network(weightsURL: nil, configuration: tiny(classCount: 6))
        try NFKMLXWeights.save(source, to: url)

        let retargeted = try NFKMLXBiSeNet.network(weightsURL: url, configuration: tiny(classCount: 3))
        for head in [retargeted.convOut, retargeted.convOut16, retargeted.convOut32] {
            XCTAssertEqual(head.convOut.weight.dim(0), 3)
        }
        XCTAssertEqual(retargeted.ffm.conv.weight.asArray(Float.self), source.ffm.conv.weight.asArray(Float.self),
                       "while everything else loaded")
    }

    func testARetargetedCheckpointLabelsAtItsOwnClassCountThroughTheFactory() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("bisenet-factory-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let net = try NFKMLXBiSeNet.network(weightsURL: nil, configuration: NFKMLXBiSeNetConfiguration(classCount: 3))
        // Every pixel's top class is the last of three, which a three-class label map encodes as white.
        net.update(parameters: ModuleParameters.unflattened([("conv_out.conv_out.bias", MLXArray([Float(0), 0, 50]))]))
        try NFKMLXWeights.save(net, to: url)

        let backend = try NFKMLXBiSeNet.backend(weightsURL: url)
        let image = NFKMLXLabelMapTesting.solid(64, 64)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
        XCTAssertEqual(try NFKMLXLabelMapTesting.levels(result.output(forKey: NFKOutputImage)), [255])
    }

    func testAFineTunedCheckpointRoundTrips() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("bisenet-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXBiSeNet.network(weightsURL: nil, configuration: tiny())
        let item = batch()
        try NFKMLXBiSeNet.fineTune(net, examples: { _ in item }, trainable: .everything, steps: 2)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXBiSeNet.network(weightsURL: url, configuration: tiny())
        let image = item.images[0]
        XCTAssertEqual(reloaded.segment(image).asArray(Float.self), net.segment(image).asArray(Float.self))
    }
}
