//
//  NFKMLXBufferCacheGradientTests.swift
//  InferKitMLXTests
//
//  MLX's Metal buffer cache hands a recycled buffer to a backward pass that does not fully
//  initialize it, so every gradient after the first in a process can be wrong by orders of
//  magnitude. Reclaiming the cache first is the workaround. The measurements and the minimal
//  reproduction are in Docs/agent-reference/mlx-runtime-gotchas.md.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXBufferCacheGradientTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The smallest graph found that shows it: the tiny backbone's stem and first four blocks, whose
    /// fourth block is the first to disagree. The loss is `(out * out).mean()`, and its gradient at
    /// the seeded weights has a norm of about 1.4e-07.
    private static func stageFourGradientNorm() -> Double {
        NFKMLXRandom.seed(4242)
        let backbone = NFKRVMBackbone(NFKMLXRVMConfiguration.tiny)
        backbone.train(false)
        let count = 32 * 32 * 3
        let x = MLXArray((0 ..< count).map { Float(($0 * 37) % 251) / 251.0 - 0.5 })
            .reshaped([1, 32, 32, 3])
        let lossAndGradient = valueAndGrad(model: backbone) { (model: NFKRVMBackbone, arrays: [MLXArray]) in
            let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 1, 3])
            let deviation = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 1, 3])
            var out = model.stem((arrays[0] - mean) / deviation)
            for (index, block) in model.blocks.enumerated() where index < 4 {
                out = block(out)
            }
            return [(out * out).mean()]
        }
        let (_, gradient) = lossAndGradient(backbone, [x])
        eval(gradient)
        var norm = 0.0
        for (_, value) in gradient.flattened() {
            for v in value.asArray(Float.self) { norm += Double(v) * Double(v) }
        }
        return norm.squareRoot()
    }

    /// The CPU is the accurate device: central finite differences along its own gradient direction
    /// converge on the norm it claims. That is what makes it the reference here.
    func testTheCPUGradientIsTheAccurateOne() throws {
        try requireMLXRuntime()
        var norm = 0.0
        NFKMLXDevice.perform(on: .cpu) { norm = Self.stageFourGradientNorm() }
        XCTAssertEqual(norm, 1.42553e-07, accuracy: 1e-10, "the reference norm at these weights")
    }

    /// The trainer reclaims the cache itself, so a fine-tune on the GPU converges rather than
    /// wandering. Before the guard the same six steps ended higher than they started about two
    /// times in five.
    func testAGPUFineTuneConvergesThroughTheTrainer() throws {
        try requireMLXRuntime()
        var trials: [[Float]] = []
        NFKMLXDevice.perform(on: .gpu) {
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
                              "a GPU run falls: \(losses.first!) -> \(losses.last!)")
        }
    }

    /// Reclaiming the buffer cache before each backward keeps the GPU on that same answer. Without
    /// it, the second and later gradients in a process come back wrong by a factor of about a
    /// million; with it, every one of them is right. The assertion holds whether or not MLX still
    /// recycles the buffer, so it survives an upstream fix.
    func testReclaimingTheCacheKeepsTheGPUGradientCorrect() throws {
        try requireMLXRuntime()
        var reference = 0.0
        NFKMLXDevice.perform(on: .cpu) { reference = Self.stageFourGradientNorm() }
        var norms: [Double] = []
        NFKMLXDevice.perform(on: .gpu) {
            norms = (0 ..< 4).map { _ in
                NFKMLXGPU.clearCache()
                return Self.stageFourGradientNorm()
            }
        }
        for (index, norm) in norms.enumerated() {
            XCTAssertEqual(norm, reference, accuracy: reference * 0.01,
                           "GPU backward \(index) with the cache reclaimed: \(norm) against \(reference)")
        }
    }
}
