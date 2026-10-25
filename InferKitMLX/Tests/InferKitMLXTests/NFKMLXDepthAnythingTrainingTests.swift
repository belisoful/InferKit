//
//  NFKMLXDepthAnythingTrainingTests.swift
//  InferKitMLXTests
//
//  Depth Anything V2's metric fine-tune without released weights: the SiLog loss and its mask, the metric
//  head's range, the reference schedule, the encoder-only load, the output convolution's row-staged weight
//  gradient, and a fine-tune that lowers the loss and reloads.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXDepthAnythingTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_006)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// A small metric configuration: 112 pixels, 8 patches of 14.
    private func small() -> NFKMLXDepthConfiguration {
        var configuration = NFKMLXDepthConfiguration.small
        configuration.inputSize = 112
        configuration.maxDepth = 20
        return configuration
    }

    func testTheSiLogLossIsScaleInvariantAndCountsOnlyTheValidRange() throws {
        try requireMLXRuntime()
        let objective = NFKMLXDepthSiLogObjective()
        let depth = MLXRandom.uniform(low: 1, high: 10, [2, 8, 8], key: MLXRandom.key(1))
        let ones = MLXArray.ones([2, 8, 8])
        // λ = 0.5 leaves half the squared log scale: a prediction off by a constant factor scores √(0.5)·|log k|.
        let scaled = objective.loss(prediction: depth * 2, depth: depth, valid: ones).item(Float.self)
        XCTAssertEqual(scaled, sqrtf(0.5) * logf(2), accuracy: 1e-5)
        // Pixels outside the mask or the range do not count, whatever the prediction there: pixel 0's depth is
        // beyond the range and pixel 1 is masked out.
        let index = MLXArray(Int32(0) ..< Int32(128)).reshaped([2, 8, 8])
        let target = MLX.where(index .== 0, MLXArray(Float(50)), depth)
        let mask = MLX.where(index .== 1, MLXArray(Float(0)), MLXArray(Float(1)))
        let base = objective.loss(prediction: depth, depth: target, valid: mask).item(Float.self)
        let wild = MLX.where(index .== 0, MLXArray(Float(0.001)), MLX.where(index .== 1, MLXArray(Float(1e4)), depth))
        XCTAssertEqual(objective.loss(prediction: wild, depth: target, valid: mask).item(Float.self), base, accuracy: 1e-6)
    }

    func testTheMetricHeadMapsIntoItsRangeAndTheRelativeOneIsNonNegative() throws {
        try requireMLXRuntime()
        let images = MLXRandom.uniform(low: 0, high: 1, [2, 112, 112, 3], key: MLXRandom.key(2))
        let metric = NFKMLXDepthAnything.makeNet(small())(NFKMLXDepthAnything.trainingInput(images))
        XCTAssertEqual(metric.shape, [2, 112, 112])
        XCTAssertGreaterThan(metric.min().item(Float.self), 0)
        XCTAssertLessThan(metric.max().item(Float.self), 20)
        var relative = small()
        relative.maxDepth = nil
        XCTAssertGreaterThanOrEqual(NFKMLXDepthAnything.makeNet(relative)(NFKMLXDepthAnything.trainingInput(images)).min().item(Float.self), 0)
    }

    func testTheReferenceScheduleStepsTheRateAfterEachUpdate() {
        let schedule = NFKMLXDepthAnything.referenceSchedule(steps: 100)
        XCTAssertEqual(schedule.multiplier(0), 1)
        XCTAssertEqual(schedule.multiplier(1), 1, "the rate set after update 0 is (1 − 0/100)^0.9")
        XCTAssertEqual(schedule.multiplier(51), powf(0.5, 0.9), accuracy: 1e-6)
    }

    func testTheEncoderOnlyLoadKeepsTheHeadAtItsInitialization() throws {
        try requireMLXRuntime()
        let source = NFKMLXDepthAnything.makeNet(small())
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(source, to: url)
        let loaded = try NFKMLXDepthAnything.network(weightsURL: url, configuration: small(), encoderOnly: true)
        let sourceParameters = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        for (key, value) in loaded.parameters().flattened() {
            let difference = abs(value - sourceParameters[key]!).max().item(Float.self)
            if key.hasPrefix("pretrained.") {
                XCTAssertEqual(difference, 0, key)
            } else if key.hasSuffix("weight") {
                XCTAssertGreaterThan(difference, 0, "\(key) keeps its own initialization")
            }
        }
    }

    func testFineTuningLowersTheLossAndReloads() throws {
        try requireMLXRuntime()
        let net = try NFKMLXDepthAnything.network(weightsURL: nil, configuration: small())
        let images = MLXRandom.uniform(low: 0, high: 1, [2, 112, 112, 3], key: MLXRandom.key(3))
        let depth = MLXRandom.uniform(low: 2, high: 12, [2, 112, 112], key: MLXRandom.key(4))
        let valid = MLXArray.ones([2, 112, 112])
        let objective = NFKMLXDepthSiLogObjective()
        let before = objective(net, NFKMLXDepthAnything.trainingInput(images), depth, valid).item(Float.self)
        let losses = try NFKMLXDepthAnything.fineTune(net, examples: { _ in (images, depth, valid) },
                                                     optimizer: Adam(learningRate: 1e-4), steps: 4, mirrors: true)
        XCTAssertEqual(losses.count, 4)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the batch it trains on")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXDepthAnything.network(weightsURL: url, configuration: small())
        let input = NFKMLXDepthAnything.trainingInput(images)
        XCTAssertLessThan(abs(net(input) - reloaded(input)).max().item(Float.self), 1e-4)
    }

    // The head's output convolution at the metric fine-tune's size: two 518-pixel maps of 32 rectified
    // channels. Its value is MLX's `conv2d` bit for bit, and its weight gradient lands nearer float64 than the
    // one-product gradient does.
    func testTheOutputConvolutionSumsItsWeightGradientByRow() throws {
        try requireMLXRuntime()
        let convolution = Conv2d(inputChannels: 32, outputChannels: 1, kernelSize: 1)
        let x = relu(MLXRandom.normal([2, 518, 518, 32], key: MLXRandom.key(3)) * 0.5 + 0.2)
        let cotangent = MLXRandom.normal([2, 518, 518, 1], key: MLXRandom.key(4)) * 0.1 + 0.05
        let value = NFKMLXStagedReduction.pointwiseConvolution(convolution, x)
        XCTAssertTrue(arrayEqual(value, convolution(x)).item(Bool.self))

        func weightGradient(_ forward: @escaping (Conv2d, MLXArray) -> MLXArray) -> [Float] {
            let gradients = valueAndGrad(model: convolution) { convolution, arrays in
                [(forward(convolution, arrays[0]) * cotangent).sum()]
            }(convolution, [x]).1.flattened()
            return Dictionary(uniqueKeysWithValues: gradients)["weight"]!.asArray(Float.self)
        }
        let staged = weightGradient { NFKMLXStagedReduction.pointwiseConvolution($0, $1) }
        let product = weightGradient { $0($1) }
        let inputs = x.reshaped([-1, 32]).asArray(Float.self), cotangents = cotangent.asArray(Float.self)
        var exact = [Double](repeating: 0, count: 32)
        for position in cotangents.indices {
            let g = Double(cotangents[position])
            for channel in 0 ..< 32 {
                exact[channel] += g * Double(inputs[position * 32 + channel])
            }
        }
        func distance(_ values: [Float]) -> Double {
            let difference = zip(values, exact).map { (Double($0) - $1) * (Double($0) - $1) }.reduce(0, +)
            return (difference / exact.map { $0 * $0 }.reduce(0, +)).squareRoot()
        }
        print("PARITY depth-pointwise-gradient: from float64 by row \(distance(staged)), one product \(distance(product))")
        XCTAssertLessThan(distance(staged), 2e-7)
        XCTAssertLessThan(distance(staged), distance(product) / 4)
    }

    func testTheReferenceOptimizerRunsTheHeadAtTenTimesTheEncoderRate() throws {
        try requireMLXRuntime()
        let net = NFKMLXDepthAnything.makeNet(small())
        let optimizer = NFKMLXDepthAnything.referenceOptimizer(for: net)
        let before = Dictionary(uniqueKeysWithValues: net.trainableParameters().flattened().map { ($0.0, $0.1 * 1) })
        let gradients = net.trainableParameters().mapValues { MLXArray.ones(like: $0) }
        optimizer.update(model: net, gradients: gradients)
        eval(net)
        let after = Dictionary(uniqueKeysWithValues: net.trainableParameters().flattened())
        // Adam's first step moves a parameter by its rate (the decay aside), so the rates read off the steps.
        let encoder = abs(after["pretrained.blocks.0.attn.qkv.bias"]! - before["pretrained.blocks.0.attn.qkv.bias"]!).max().item(Float.self)
        let head = abs(after["depth_head.scratch.output_conv1.bias"]! - before["depth_head.scratch.output_conv1.bias"]!).max().item(Float.self)
        XCTAssertEqual(head / encoder, 10, accuracy: 0.1)
    }
}
