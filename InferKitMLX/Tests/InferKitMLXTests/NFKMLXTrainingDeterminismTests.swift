//
//  NFKMLXTrainingDeterminismTests.swift
//  InferKitMLXTests
//
//  What a seed does and does not fix in a training run. Seeding makes the weights reproducible;
//  it does not make MLX's gradients reproducible, so a test that reads a short run's loss is
//  reading noise unless the run is pinned to the CPU and long enough to clear it. The measurements
//  behind this are in Docs/agent-reference/mlx-runtime-gotchas.md.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXTrainingDeterminismTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The seed fixes the weights exactly: the same seed builds the same net, and its first forward
    /// scores the same, whichever device runs it and however often.
    func testASeedFixesTheWeightsAndTheFirstLoss() throws {
        try requireMLXRuntime()
        func weightSumAndFirstLoss() -> (sum: Float, loss: Float) {
            NFKMLXRandom.seed(20_260_904)
            let net = NFKMLXRVMNet(.tiny)
            let sum = net.parameters().flattened().sorted { $0.0 < $1.0 }
                .flatMap { $0.1.asArray(Float.self) }
                .reduce(Float(0)) { $0 + $1 }
            net.train(true)
            let frame = MLXArray.ones([1, 32, 32, 3]) * 0.5
            let target = MLXArray.ones([1, 32, 32, 1])
            let (_, alpha, _) = net.forward(frame, state: NFKMLXRVMNet.initialState)
            let loss = ((alpha - target) * (alpha - target)).mean()
            eval(loss)
            return (sum, loss.item(Float.self))
        }
        var readings: [(sum: Float, loss: Float)] = []
        NFKMLXDevice.perform(on: .cpu) {
            readings = (0 ..< 3).map { _ in weightSumAndFirstLoss() }
        }
        XCTAssertEqual(Set(readings.map(\.sum)).count, 1, "the same seed builds the same weights")
        XCTAssertEqual(Set(readings.map(\.loss)).count, 1, "so the first forward scores the same")
    }

    /// The gradients are another matter: the same seeded run trains to a different place each time.
    /// A test that reads a loss has to leave room for that, which is what pinning the fine-tune in
    /// ``NFKMLXRVMTests`` to the CPU buys. On the GPU the spread swallows six steps of progress.
    func testASeededRunStillFallsEveryTimeOnTheCPU() throws {
        try requireMLXRuntime()
        var trials: [[Float]] = []
        NFKMLXDevice.perform(on: .cpu) {
            trials = (0 ..< 3).compactMap { _ in
                NFKMLXRandom.seed(20_260_904)
                let net = NFKMLXRVMNet(.tiny)
                let frame = MLXArray.ones([1, 32, 32, 3]) * 0.5
                let target = MLXArray.ones([1, 32, 32, 1])
                return try? NFKMLXTrainer.train(
                    net, optimizer: SGD(learningRate: 0.05), steps: 6,
                    batch: { _ in (frame, target) },
                    loss: { model, input, expected in
                        let (_, alpha, _) = model.forward(input, state: NFKMLXRVMNet.initialState)
                        return ((alpha - expected) * (alpha - expected)).mean()
                    },
                    clipGradientNorm: 1)
            }
        }
        XCTAssertEqual(trials.count, 3)
        for losses in trials {
            XCTAssertLessThan(losses.last!, losses.first!,
                              "a CPU run of these six steps falls every time: \(losses.first!) -> \(losses.last!)")
        }
    }
}
