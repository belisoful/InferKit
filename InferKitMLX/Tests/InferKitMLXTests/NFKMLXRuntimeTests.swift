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

    // MARK: What the machine has

    func testTheMachineReportsItsMemoryAndArchitecture() throws {
        try requireMLXRuntime()
        XCTAssertGreaterThan(NFKMLXGPU.physicalMemory, 0, "the host reports its RAM")
        XCTAssertGreaterThan(NFKMLXGPU.recommendedWorkingSetSize, 0, "Metal recommends a working set")
        XCTAssertLessThanOrEqual(NFKMLXGPU.recommendedWorkingSetSize, NFKMLXGPU.physicalMemory,
                                 "the recommendation cannot exceed the machine")
        XCTAssertFalse(NFKMLXGPU.deviceArchitecture.isEmpty)
    }

    // Cache is reclaimable and active memory is not, which is the distinction a caller deciding
    // whether to unload a model depends on.
    func testReclaimableMemoryTracksTheCacheAndIsFreedByClearing() throws {
        try requireMLXRuntime()
        let previous = NFKMLXGPU.cacheLimit
        defer { NFKMLXGPU.setCacheLimit(previous) }
        NFKMLXGPU.setCacheLimit(64 * 1024 * 1024)

        // Allocate and drop, which leaves buffers in the cache rather than returning them.
        for _ in 0 ..< 4 {
            let scratch = MLXArray.zeros([256, 256])
            eval(scratch)
        }
        XCTAssertEqual(NFKMLXGPU.reclaimableMemory, NFKMLXGPU.cacheMemory)
        NFKMLXGPU.clearCache()
        XCTAssertEqual(NFKMLXGPU.reclaimableMemory, 0, "clearing returns the cache to the system")
    }

    func testMemoryPressureIsAFractionOfTheRecommendedWorkingSet() throws {
        try requireMLXRuntime()
        let pressure = NFKMLXGPU.memoryPressure
        XCTAssertGreaterThanOrEqual(pressure, 0)
        XCTAssertLessThan(pressure, 1.5, "a plausible share of the budget")
    }

    func testStandingLimitsApplyBothCapsAndCanBeRestored() throws {
        try requireMLXRuntime()
        let previousCache = NFKMLXGPU.cacheLimit
        let previousMemory = NFKMLXGPU.memoryLimit
        defer {
            NFKMLXGPU.setCacheLimit(previousCache)
            NFKMLXGPU.setMemoryLimit(previousMemory)
        }

        NFKMLXGPU.applyStandingLimits()
        XCTAssertEqual(NFKMLXGPU.cacheLimit, NFKMLXGPU.defaultCacheCap)
        XCTAssertGreaterThan(NFKMLXGPU.memoryLimit, 0)
        XCTAssertLessThanOrEqual(NFKMLXGPU.memoryLimit, NFKMLXGPU.recommendedWorkingSetSize)
    }

    // Passing zero leaves the memory limit alone, so a caller can cap the cache without also
    // committing to a memory ceiling.
    func testStandingLimitsLeaveTheMemoryLimitAloneWhenTheFractionIsZero() throws {
        try requireMLXRuntime()
        let previousCache = NFKMLXGPU.cacheLimit
        let previousMemory = NFKMLXGPU.memoryLimit
        defer {
            NFKMLXGPU.setCacheLimit(previousCache)
            NFKMLXGPU.setMemoryLimit(previousMemory)
        }

        NFKMLXGPU.applyStandingLimits(cacheBytes: 32 * 1024 * 1024,
                                      fractionOfRecommendedWorkingSet: 0)
        XCTAssertEqual(NFKMLXGPU.cacheLimit, 32 * 1024 * 1024)
        XCTAssertEqual(NFKMLXGPU.memoryLimit, previousMemory)
    }
}
