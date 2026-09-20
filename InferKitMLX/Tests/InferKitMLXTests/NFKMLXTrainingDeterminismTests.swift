//
//  NFKMLXTrainingDeterminismTests.swift
//  InferKitMLXTests
//
//  What a seed does and does not fix in a training run. Seeding makes the weights reproducible. It
//  does not make a forward pass reproducible, and it does not make MLX's gradients reproducible, so
//  a test that reads a short run's loss is reading noise. The measurements behind this are in
//  Docs/agent-reference/mlx-runtime-gotchas.md.
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

    /// The seed fixes the weights exactly. The same seed builds the same net, bit for bit, which is
    /// the whole of what seeding guarantees.
    ///
    /// Run on the GPU. Training-mode work on the CPU kills its process about one time in ten, in
    /// MLX's own convolution, which `Docs/mlx-runtime-hazards.md` records.
    ///
    /// The first loss is a separate matter. Measured over 180 builds from one seed, the parameter
    /// sum was identical every time while the loss of the first forward took four distinct values on
    /// the CPU, spanning 0.51985323 to 0.52548116. A `BatchNorm` over a batch of one divides by the
    /// batch's own standard deviation, which turns an accumulation difference of about 1e-07 into a
    /// difference of about 5e-03 in the loss. The tolerance here is three times that measured
    /// spread.
    func testASeedFixesTheWeightsAndTheFirstLossToWithinTheMeasuredSpread() throws {
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
        NFKMLXDevice.perform(on: .gpu) {
            readings = (0 ..< 3).map { _ in weightSumAndFirstLoss() }
        }
        XCTAssertEqual(Set(readings.map(\.sum)).count, 1, "the same seed builds the same weights")
        for reading in readings {
            XCTAssertEqual(reading.loss, readings[0].loss, accuracy: 0.02,
                           "the first forward scores the same to within its measured spread")
        }
    }

    /// The gradients are another matter: the same seeded run trains to a different place each time.
    /// A test that reads a loss has to leave room for that.
    ///
    /// Twelve steps rather than six, because six do not clear the spread. Measured over 60 runs, a
    /// six-step GPU run ended above where it started 8 times with the buffer cache left alone, and
    /// no times under the trainer's default policy. Twelve steps ended between 0.344 and 0.656 of
    /// the first loss over 20 runs under that default, so the threshold below clears the worst run
    /// measured. The run is on the GPU because training-mode work on the CPU kills its process
    /// about one time in ten, and no consumer fine-tune runs there by default.
    func testASeededRunTrainsDown() throws {
        try requireMLXRuntime()
        var losses: [Float] = []
        NFKMLXDevice.perform(on: .gpu) {
            NFKMLXRandom.seed(20_260_904)
            let net = NFKMLXRVMNet(.tiny)
            let frame = MLXArray.ones([1, 32, 32, 3]) * 0.5
            let target = MLXArray.ones([1, 32, 32, 1])
            losses = (try? NFKMLXTrainer.train(
                net, optimizer: SGD(learningRate: 0.05), steps: 12,
                batch: { _ in (frame, target) },
                loss: { model, input, expected in
                    let (_, alpha, _) = model.forward(input, state: NFKMLXRVMNet.initialState)
                    return ((alpha - expected) * (alpha - expected)).mean()
                },
                clipGradientNorm: 1)) ?? []
        }
        XCTAssertEqual(losses.count, 12)
        let first = try XCTUnwrap(losses.first)
        let last = try XCTUnwrap(losses.last)
        XCTAssertLessThan(last, first * 0.85,
                          "twelve seeded steps train down: \(first) -> \(last)")
    }
}
