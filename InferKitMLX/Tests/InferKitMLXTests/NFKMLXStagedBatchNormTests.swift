//
//  NFKMLXStagedBatchNormTests.swift
//  InferKitMLXTests
//
//  `NFKTorchBatchNorm`'s staged statistics against float64 statistics computed here: a nearly constant
//  channel's running variance (unbiased, as PyTorch folds it), normalized output, and input gradient on
//  both devices, against MLX's own reduction on the CPU, which loses them, and identity with
//  `BatchNorm` in evaluation mode.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXStagedBatchNormTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// `[2, 160, 200, 2]`: channel 0 is 0.3 varying by 1e-4, channel 1 varies by 1.
    private func batch() -> MLXArray {
        let noise = MLXRandom.normal([2, 160, 200, 2], key: MLXRandom.key(7))
        return noise * MLXArray([Float(1e-4), 1]) + MLXArray([Float(0.3), 0])
    }

    /// Per-channel float64 mean and population variance of a channel-last batch.
    private func statistics(_ values: [Float], channels: Int) -> (mean: [Double], variance: [Double]) {
        let count = Double(values.count / channels)
        var mean = [Double](repeating: 0, count: channels)
        for (index, value) in values.enumerated() {
            mean[index % channels] += Double(value) / count
        }
        var variance = [Double](repeating: 0, count: channels)
        for (index, value) in values.enumerated() {
            let deviation = Double(value) - mean[index % channels]
            variance[index % channels] += deviation * deviation / count
        }
        return (mean, variance)
    }

    private func relative(_ ours: [Float], _ reference: [Double], channel: Int, channels: Int) -> Double {
        var error = 0.0, size = 0.0
        for index in stride(from: channel, to: ours.count, by: channels) {
            error += (Double(ours[index]) - reference[index]) * (Double(ours[index]) - reference[index])
            size += reference[index] * reference[index]
        }
        return (error / size).squareRoot()
    }

    func testANearlyConstantChannelKeepsItsStatisticsOnBothDevices() throws {
        try requireMLXRuntime()
        let x = batch()
        let values = x.asArray(Float.self)
        let reference = statistics(values, channels: 2)
        let scale = reference.variance.map { 1 / ($0 + 1e-5).squareRoot() }
        let normalized = values.enumerated().map { (Double($0.element) - reference.mean[$0.offset % 2]) * scale[$0.offset % 2] }
        // The loss Σ r·y; its input gradient is (r − mean r − ŷ · mean(r·ŷ)) / σ per channel.
        let weights = MLXRandom.normal(x.shape, key: MLXRandom.key(8))
        let r = weights.asArray(Float.self)
        var meanR = [Double](repeating: 0, count: 2), meanRY = [Double](repeating: 0, count: 2)
        for index in values.indices {
            meanR[index % 2] += Double(r[index]) / Double(values.count / 2)
            meanRY[index % 2] += Double(r[index]) * normalized[index] / Double(values.count / 2)
        }
        let gradient = values.indices.map { (Double(r[$0]) - meanR[$0 % 2] - normalized[$0] * meanRY[$0 % 2]) * scale[$0 % 2] }

        func measure(_ norm: BatchNorm, on device: Device, unbiased: Bool) -> (variance: Double, output: Double, gradient: Double) {
            let count = Double(values.count / 2)
            let folded = reference.variance[0] * (unbiased ? count / (count - 1) : 1)
            norm.train(true)
            return Device.withDefaultDevice(device) {
                let output = norm(x)
                let running = Dictionary(uniqueKeysWithValues: norm.parameters().flattened())["running_var"]!
                let ours = grad { (input: MLXArray) in (norm(input) * weights).sum() }(x)
                eval(output, running, ours)
                let variance = Double(running.asArray(Float.self)[0])
                return (abs(variance - folded) / folded,
                        relative(output.asArray(Float.self), normalized, channel: 0, channels: 2),
                        relative(ours.asArray(Float.self), gradient, channel: 0, channels: 2))
            }
        }
        let plain = measure(BatchNorm(featureCount: 2, momentum: 1), on: .cpu, unbiased: false)
        for device in [Device.cpu, Device.gpu] {
            let staged = measure(NFKTorchBatchNorm(featureCount: 2, momentum: 1), on: device, unbiased: true)
            let mlx = measure(BatchNorm(featureCount: 2, momentum: 1), on: device, unbiased: false)
            print("STAGED-BN \(device) from float64: staged variance \(staged.variance), output \(staged.output), gradient \(staged.gradient); "
                  + "BatchNorm variance \(mlx.variance), output \(mlx.output), gradient \(mlx.gradient)")
            XCTAssertLessThan(staged.variance, 1e-5, "\(device)")
            XCTAssertLessThan(staged.output, 1e-6, "\(device)")
            XCTAssertLessThan(staged.gradient, 1e-6, "\(device)")
        }
        // MLX's own reduction on the same batch, so the bounds above are known to discriminate.
        XCTAssertGreaterThan(plain.variance, 1e-3)
        XCTAssertGreaterThan(plain.output, 1e-2)
    }

    func testEvaluationComputesWhatBatchNormDoes() throws {
        try requireMLXRuntime()
        let x = batch()
        let staged = NFKTorchBatchNorm(featureCount: 2)
        let plain = BatchNorm(featureCount: 2)
        let parameters = ModuleParameters.unflattened(["weight": MLXArray([Float(1.5), 0.5]), "bias": MLXArray([Float(0.1), -0.2]),
                                                       "running_mean": MLXArray([Float(0.3), 0.1]), "running_var": MLXArray([Float(1e-8), 0.9])])
        staged.update(parameters: parameters)
        plain.update(parameters: parameters)
        staged.train(false)
        plain.train(false)
        for device in [Device.cpu, Device.gpu] {
            XCTAssertEqual(Device.withDefaultDevice(device) { abs(staged(x) - plain(x)).max().item(Float.self) }, 0, "\(device)")
        }
    }
}
