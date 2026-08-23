//
//  NFKMLXRuntimeTests.swift
//  InferKitMLXTests
//
//  The Objective-C-facing MLX runtime wrappers: reproducible seeding, GPU memory management, and the
//  compute-device selection. These
//  evaluate MLX, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXRuntimeTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    func testSeedMakesWeightInitializationReproducible() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(7)
        let first = Linear(8, 8).weight
        NFKMLXRandom.seed(7)
        let same = Linear(8, 8).weight
        NFKMLXRandom.seed(8)
        let different = Linear(8, 8).weight
        eval(first, same, different)
        XCTAssertEqual(first.asArray(Float.self), same.asArray(Float.self), "same seed → identical weights")
        XCTAssertNotEqual(first.asArray(Float.self), different.asArray(Float.self), "different seed → different weights")
    }

    func testTheScopedDeviceSelectionAppliesAndUnwinds() throws {
        try requireMLXRuntime()
        let before = NFKMLXDevice.currentType
        var inside: NFKMLXDeviceType?
        NFKMLXDevice.perform(on: .cpu) { inside = NFKMLXDevice.currentType }
        XCTAssertEqual(inside, .cpu)
        XCTAssertEqual(NFKMLXDevice.currentType, before, "the previous device is restored")
    }

    func testAModelEvaluatesOnTheSelectedDevice() throws {
        try requireMLXRuntime()
        let net = Linear(8, 8)
        var result: [Float] = []
        NFKMLXDevice.perform(on: .cpu) {
            let output = net(MLXArray.zeros([1, 8]) + 0.5)
            eval(output)
            result = output.asArray(Float.self)
        }
        XCTAssertEqual(result.count, 8, "the CPU backend evaluates a real layer")
    }

    // mlx-swift scopes the device to a task-local value, so a queue the caller dispatches to takes the
    // global default. The documented boundary on `perform(on:block:)` depends on this staying true.
    func testTheSelectionDoesNotCrossAnAsynchronousDispatch() throws {
        try requireMLXRuntime()
        let outer = NFKMLXDevice.currentType
        let observed = expectation(description: "the queue reports its device")
        var onQueue: NFKMLXDeviceType?
        NFKMLXDevice.perform(on: .cpu) {
            DispatchQueue.global().async { onQueue = NFKMLXDevice.currentType; observed.fulfill() }
            wait(for: [observed], timeout: 10)
        }
        XCTAssertEqual(onQueue, outer, "a dispatched block keeps the global default, not the scoped one")
    }

    func testGPUMemoryKnobsReadAndWrite() throws {
        try requireMLXRuntime()
        let previous = NFKMLXGPU.cacheLimit
        defer { NFKMLXGPU.setCacheLimit(previous) }

        NFKMLXGPU.setCacheLimit(64 * 1024 * 1024)
        XCTAssertEqual(NFKMLXGPU.cacheLimit, 64 * 1024 * 1024, "the cache limit round-trips")
        NFKMLXGPU.clearCache()
        NFKMLXGPU.resetPeakMemory()
        XCTAssertGreaterThanOrEqual(NFKMLXGPU.activeMemory, 0)
        XCTAssertGreaterThanOrEqual(NFKMLXGPU.cacheMemory, 0)
        XCTAssertGreaterThanOrEqual(NFKMLXGPU.peakMemory, 0)
        XCTAssertGreaterThanOrEqual(NFKMLXGPU.memoryLimit, 0)
    }
}
