//
//  NFKMLXUpstreamWatchTests.swift
//  InferKitMLXTests
//
//  Watches the two MLX defects the package works around, so the workarounds are removed when the
//  runtime stops needing them rather than carried forever. The backward defect has a known fix in
//  mlx core 0.32.0, and the package waits on an mlx-swift release that vendors it. These tests report and do not fail. A
//  defect that is still present is the expected state, and a test that failed on it would make the
//  suite red for something no change here can fix. Read the printed lines, or grep a test log for
//  "UPSTREAM WATCH". The defects are recorded in Docs/agent-reference/mlx-runtime-gotchas.md.
//

import XCTest
import Foundation
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXUpstreamWatchTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The gradient of the smallest graph that shows the buffer-cache fault, taken with whatever
    /// cache policy the caller has set.
    private static func gradientNorm() -> Double {
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

    /// Reports whether a GPU backward is still wrong when the buffer cache is left alone.
    ///
    /// A reading that shows the fault is conclusive. A clean reading is not, so this test never
    /// reports one as a fix.
    ///
    /// The fault needs the buffer cache to hold this graph's own buffers. In a fresh process it is
    /// deterministic, at 0 of 20 readings matching the CPU on three runs out of three. In a process
    /// that has already run a training test it is not, at 12, 16, and 20 of 20 on three runs,
    /// because the trainer runs with the cache limit at zero and leaves the cache empty behind it.
    /// Five discarded readings ahead of the judged ones do not repopulate it reliably. Why an empty
    /// cache yields the correct answer here is the same open question as the fault itself.
    ///
    /// The authoritative check is this test alone in a fresh process:
    ///
    /// ```
    /// swift test --filter NFKMLXUpstreamWatchTests
    /// ```
    ///
    /// Run that way, `not observed` means the defect is gone and
    /// ``NFKMLXTrainingCachePolicy/disabledOnGPU`` can stop being the default. Run inside a suite,
    /// `not observed` means only that this process did not reproduce it.
    ///
    /// The defect is in mlx core, which fixes it in 0.32.0. Measured in Python against `mlx`
    /// directly, GPU readings matching the CPU go from 0 of 25 on core 0.31.1 to 25 of 25 on 0.32.0.
    /// mlx-swift 0.31.6 vendors core 0.31.1. This watch is the check to run on the release that
    /// brings core 0.32.0 or later into the package.
    func testWhetherTheBufferCacheStillCorruptsABackward() throws {
        try requireMLXRuntime()
        var reference = 0.0
        NFKMLXDevice.perform(on: .cpu) { reference = Self.gradientNorm() }

        var norms: [Double] = []
        NFKMLXDevice.perform(on: .gpu) {
            let limitBefore = NFKMLXGPU.cacheLimit
            NFKMLXGPU.synchronize()
            NFKMLXGPU.setCacheLimit(1 << 29)
            defer {
                NFKMLXGPU.synchronize()
                NFKMLXGPU.setCacheLimit(limitBefore)
            }
            // The fault needs the cache to hold this graph's own buffers. A test that ran before
            // this one may have left the cache empty, which reads as correct and is not. These
            // readings populate it and are discarded.
            for _ in 0 ..< 5 { _ = Self.gradientNorm() }
            norms = (0 ..< 20).map { _ in Self.gradientNorm() }
        }

        let matching = norms.filter { abs($0 - reference) <= reference * 0.01 }.count
        // A clean reading inside a suite means this process did not reproduce the fault, which is
        // not the same as the fault being gone. Only a fresh process can say that, so the wording
        // never claims a fix on its own.
        let verdict = matching == norms.count
            ? "not observed in this process; authoritative only when run alone in a fresh process"
            : "still present"
        print("UPSTREAM WATCH buffer-cache backward: \(verdict) "
              + "(\(matching)/\(norms.count) warmed backwards match the CPU's \(reference))")
        XCTAssertEqual(norms.count, 20, "the watch took its readings")
    }

    /// Reports whether MLX's CPU convolution still takes the path that crashes.
    ///
    /// `conv_2D_cpu` uses an explicit-GEMM convolution only when dilation is 1 and the group count
    /// is 1, and otherwise calls `slow_conv_2D`, which faults on unmapped memory under
    /// training-mode work. This watch does not run that path, because a crash would truncate the
    /// suite rather than report anything. It times a depthwise convolution against a dense one of
    /// the same shape on the CPU: the slow path is several times the cost of the GEMM path, so the
    /// ratio falling to about 1 is the sign that MLX routes them the same way now.
    func testWhetherTheCPUConvolutionStillHasASlowPath() throws {
        try requireMLXRuntime()
        var dense = 0.0
        var depthwise = 0.0
        NFKMLXDevice.perform(on: .cpu) {
            let x = MLXArray.ones([1, 64, 64, 32]) * 0.5
            let dw = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 3, padding: 1,
                            groups: 32, bias: false)
            let dn = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 3, padding: 1,
                            bias: false)
            eval(dw(x), dn(x))
            func seconds(_ body: () -> MLXArray) -> Double {
                let start = Date()
                for _ in 0 ..< 20 { eval(body()) }
                return Date().timeIntervalSince(start)
            }
            dense = seconds { dn(x) }
            depthwise = seconds { dw(x) }
        }
        let ratio = depthwise / Swift.max(dense, 1e-9)
        let verdict = ratio < 1.5 ? "APPEARS FIXED" : "still present"
        print(String(format: "UPSTREAM WATCH cpu grouped convolution: %@ "
                     + "(depthwise %.3f s against dense %.3f s, ratio %.2f)",
                     verdict, depthwise, dense, ratio))
        XCTAssertGreaterThan(dense, 0, "the watch timed both convolutions")
    }
}
