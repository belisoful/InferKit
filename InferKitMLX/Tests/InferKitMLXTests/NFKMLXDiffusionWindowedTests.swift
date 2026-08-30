//
//  NFKMLXDiffusionWindowedTests.swift
//  InferKitMLXTests
//
//  Windowed continuation for long-form diffusion output: the window arithmetic needs no GPU; the
//  sampling checks evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDiffusionWindowedTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    // MARK: Window arithmetic (no MLX)

    func testWindowStartsTileTheOutputWithFullWindows() {
        XCTAssertEqual(NFKMLXDiffusionBackend.windowStarts(totalWidth: 8, windowWidth: 8, hop: 4), [0],
                       "one window covers an output no longer than the window")
        let starts = NFKMLXDiffusionBackend.windowStarts(totalWidth: 16, windowWidth: 8, hop: 4)
        XCTAssertEqual(starts, [0, 4, 8], "windows stride by the hop; the last ends exactly at the total")
        // Every window is full-width and the pieces (leading-to-next, last full) cover the output once.
        var covered = 0
        for (i, start) in starts.enumerated() {
            XCTAssertLessThanOrEqual(start + 8, 16, "no window runs past the end")
            covered += i < starts.count - 1 ? starts[i + 1] - start : 8
        }
        XCTAssertEqual(covered, 16)
        // A pulled-back final window (uneven hop) still ends exactly at the total.
        XCTAssertEqual(NFKMLXDiffusionBackend.windowStarts(totalWidth: 20, windowWidth: 8, hop: 6).last, 12)
    }

    // MARK: Sampling (needs MLX)

    private let denoise: (MLXArray, NFKDiffusionTimestep, Int) -> MLXArray = { latent, _, _ in
        latent * MLXArray(Float(0.3))                  // deterministic, content-dependent
    }

    // One window reduces to the ordinary sampler loop over the same noise — the windowing adds
    // nothing when there is nothing to continue.
    func testWindowedContinuationMatchesThePlainLoopForOneWindow() throws {
        try requireMLXRuntime()
        let scheduler = NFKDDIMScheduler()
        let steps = 6, seed: UInt64 = 7, width = 8, height = 4, channels = 4

        let windowed = try XCTUnwrap(NFKMLXDiffusionBackend.windowedContinuation(
            totalWidth: width, height: height, channels: channels,
            windowWidth: width, hop: width, scheduler: scheduler, steps: steps, seed: seed,
            denoiseWindow: denoise))

        let schedule = scheduler.steps(steps)
        let noise = NFKMLXDiffusionBackend.gaussianNoise(height: height, width: width,
                                                         channels: channels, seed: seed)
        var latent = scheduler.initialLatent(noise: noise, first: schedule[0])
        for i in 0 ..< schedule.count {
            latent = scheduler.step(prediction: denoise(latent, schedule[i], 0),
                                    timestep: schedule[i], latent: latent)
        }
        eval(windowed, latent)
        XCTAssertEqual((windowed - latent).abs().max().item(Float.self), 0, accuracy: 1e-5,
                       "the single-window path is the plain loop")
    }

    func testWindowedContinuationAssemblesTheRequestedLengthDeterministically() throws {
        try requireMLXRuntime()
        let scheduler = NFKDDIMScheduler()
        func run() -> MLXArray? {
            NFKMLXDiffusionBackend.windowedContinuation(
                totalWidth: 16, height: 4, channels: 4, windowWidth: 8, hop: 4,
                scheduler: scheduler, steps: 5, seed: 11, denoiseWindow: denoise)
        }
        let a = try XCTUnwrap(run())
        let b = try XCTUnwrap(run())
        eval(a, b)
        XCTAssertEqual(a.shape, [4, 16, 4], "the assembled latent is the requested length")
        XCTAssertEqual((a - b).abs().max().item(Float.self), 0, "the same seed reproduces the output")
    }

    // The overlap hold is what makes it a continuation rather than independent tiles: turning it off
    // (hop == windowWidth, no overlap) must change the result.
    func testTheOverlapHoldChangesTheResult() throws {
        try requireMLXRuntime()
        let scheduler = NFKDDIMScheduler()
        let continued = try XCTUnwrap(NFKMLXDiffusionBackend.windowedContinuation(
            totalWidth: 16, height: 4, channels: 4, windowWidth: 8, hop: 4,
            scheduler: scheduler, steps: 5, seed: 3, denoiseWindow: denoise))
        let independent = try XCTUnwrap(NFKMLXDiffusionBackend.windowedContinuation(
            totalWidth: 16, height: 4, channels: 4, windowWidth: 8, hop: 8,
            scheduler: scheduler, steps: 5, seed: 3, denoiseWindow: denoise))
        eval(continued, independent)
        XCTAssertEqual(continued.shape, independent.shape)
        XCTAssertGreaterThan((continued - independent).abs().max().item(Float.self), 0,
                             "holding the overlap changes the output")
    }

    func testWindowedContinuationCancels() throws {
        try requireMLXRuntime()
        var steps = 0
        let cancelled = NFKMLXDiffusionBackend.windowedContinuation(
            totalWidth: 16, height: 4, channels: 4, windowWidth: 8, hop: 4,
            scheduler: NFKDDIMScheduler(), steps: 5, seed: 1, denoiseWindow: denoise,
            progress: { _, _ in steps += 1; return steps < 3 })
        XCTAssertNil(cancelled, "returning false from progress cancels and returns nil")
    }
}
