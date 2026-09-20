//
//  NFKMLXGradientDeterminismTests.swift
//  InferKitMLXTests
//
//  What holds and what does not when a backward pass runs twice. Every layer on its own is exact on
//  both devices and the forward is exact whole; the composed backward is not, on the GPU. The
//  measurements behind the difference are in Docs/agent-reference/mlx-runtime-gotchas.md.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXGradientDeterminismTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// A varied input, so a normalization has something to normalize.
    private static func variedFrame(channels: Int = 8, size: Int = 32) -> MLXArray {
        let count = size * size * channels
        return MLXArray((0 ..< count).map { Float(($0 * 37) % 251) / 251.0 })
            .reshaped([1, size, size, channels])
    }

    /// The norm of one layer's gradient for a fixed input, repeated on one device.
    private static func gradientNorms<M: Module & UnaryLayer>(of build: () -> M, repeats: Int) -> [Double] {
        (0 ..< repeats).map { _ in
            NFKMLXRandom.seed(4242)
            let layer = build()
            let x = variedFrame()
            let lossAndGradient = valueAndGrad(model: layer) { (model: M, arrays: [MLXArray]) in
                let y = model(arrays[0])
                return [(y * y).mean()]
            }
            let (_, gradients) = lossAndGradient(layer, [x])
            eval(gradients)
            var norm = 0.0
            for (_, value) in gradients.flattened() {
                for v in value.asArray(Float.self) { norm += Double(v) * Double(v) }
            }
            return norm.squareRoot()
        }
    }

    /// One layer at a time, a backward pass is reproducible on each device and the two devices
    /// agree. This is what rules out any single kernel when a composed graph misbehaves.
    func testOneLayersBackwardIsReproducibleAndAgreesAcrossDevices() throws {
        try requireMLXRuntime()
        func check(_ name: String, _ build: @escaping () -> some Module & UnaryLayer) {
            var gpu: [Double] = []
            var cpu: [Double] = []
            NFKMLXDevice.perform(on: .gpu) { gpu = Self.gradientNorms(of: build, repeats: 3) }
            NFKMLXDevice.perform(on: .cpu) { cpu = Self.gradientNorms(of: build, repeats: 3) }
            XCTAssertEqual(Set(gpu).count, 1, "\(name) repeats on the GPU: \(gpu)")
            XCTAssertEqual(Set(cpu).count, 1, "\(name) repeats on the CPU: \(cpu)")
            XCTAssertEqual(gpu[0], cpu[0], accuracy: Swift.max(cpu[0] * 1e-4, 1e-9),
                           "\(name) agrees across devices")
        }
        check("conv") { Conv2d(inputChannels: 8, outputChannels: 8, kernelSize: 3, padding: 1, bias: false) }
        check("depthwise conv") {
            Conv2d(inputChannels: 8, outputChannels: 8, kernelSize: 3, padding: 1, groups: 8, bias: false)
        }
        check("grouped conv") {
            Conv2d(inputChannels: 8, outputChannels: 8, kernelSize: 3, padding: 1, groups: 4, bias: false)
        }
        check("batch norm in training mode") {
            let norm = BatchNorm(featureCount: 8)
            norm.train(true)
            return norm
        }
        check("linear") { Linear(8, 8) }
    }

    /// The forward of a whole composed net is exact: the same loss, and the same pixels inside the
    /// output's clamp, on either device and however often. Whatever moves in the backward, it does
    /// not come from the forward reading different numbers.
    func testTheComposedForwardIsExactOnBothDevices() throws {
        try requireMLXRuntime()
        func reading() -> (loss: Float, active: Int) {
            NFKMLXRandom.seed(20_260_904)
            let net = NFKMLXRVMNet(.tiny)
            net.train(false)
            let frame = MLXArray.ones([1, 32, 32, 3]) * 0.5
            let target = MLXArray.ones([1, 32, 32, 1])
            let (_, alpha, _) = net.forward(frame, state: NFKMLXRVMNet.initialState)
            let loss = ((alpha - target) * (alpha - target)).mean()
            eval(loss, alpha)
            return (loss.item(Float.self), alpha.asArray(Float.self).filter { $0 > 0 && $0 < 1 }.count)
        }
        var gpu: [(loss: Float, active: Int)] = []
        var cpu: [(loss: Float, active: Int)] = []
        NFKMLXDevice.perform(on: .gpu) { gpu = (0 ..< 3).map { _ in reading() } }
        NFKMLXDevice.perform(on: .cpu) { cpu = (0 ..< 3).map { _ in reading() } }
        XCTAssertEqual(Set(gpu.map(\.loss)).count, 1, "the GPU forward repeats: \(gpu.map(\.loss))")
        XCTAssertEqual(Set(cpu.map(\.loss)).count, 1, "the CPU forward repeats: \(cpu.map(\.loss))")
        XCTAssertEqual(gpu[0].loss, cpu[0].loss, accuracy: 1e-5, "the two devices agree")
        XCTAssertEqual(Set(gpu.map(\.active) + cpu.map(\.active)).count, 1,
                       "the same pixels sit inside the clamp: \(gpu.map(\.active)) \(cpu.map(\.active))")
    }
}
