//
//  NFKMLXLoadFootprintTests.swift
//  InferKitMLXTests
//
//  The release readers' memory, held to a budget. Each test writes a synthetic release of a few
//  gigabytes, loads it through one reader, and reads the kernel's own peak footprint for the load: the
//  interval maximum `proc_pid_rusage` reports, reset just before the load, so earlier tests in the
//  process do not count. The transient is that peak less the footprint before the load and the bytes
//  the result holds. Measured on synthetic 2 GiB releases (2026-10-09), a read-first conversion holds
//  one 256 MiB group above its result and a stored-type read holds nothing; a lazy widening, its reads
//  left to one evaluation, holds 1216 MiB. The budgets sit at twice the read-first measurement.
//
//  A measurement starts only once the footprint holds no memory MLX has released: the GPU and CPU
//  streams drained, MLX's cache at 0 bytes, and the footprint less MLX's own bytes back at the level
//  the class reads with nothing released. A start taken earlier counted up to 1.8 GiB that the load
//  then took over, and the guard failed (`mlx-runtime-gotchas.md`).
//

import Darwin
import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXLoadFootprintTests: XCTestCase {

    private static let mebibyte = 1 << 20

    /// The readers evaluate in groups of this many stored bytes (`NFKMLXReleaseWeights`).
    private static let groupBytes = 256 << 20

    /// One synthetic tensor: `[4096, 4096]` bfloat16 is 32 MiB, eight to a group.
    private static let tensorShape = [4096, 4096]

    /// How far the footprint less MLX's own bytes may sit above its quiet level when a measurement starts.
    private static let releasedTolerance = 64 * mebibyte

    /// The lowest settled footprint less MLX's active and cached bytes the class has read: the process's
    /// own memory with no released MLX buffer still counted.
    nonisolated(unsafe) private static var quietUnownedFootprint = Int.max

    private var directory: URL!

    override func setUpWithError() throws {
        try requireMLXRuntime()
        try XCTSkipIf(Self.resetFootprintInterval == nil,
                      "proc_reset_footprint_interval is not available, so a load's own peak cannot be read")
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXLoadFootprintTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try warmUp()
        _ = try Self.settledFootprint(requiringQuiet: false)
    }

    override func tearDownWithError() throws {
        if let directory {
            try? FileManager.default.removeItem(at: directory)
        }
        NFKMLXGPU.clearCache()
    }

    // MARK: The kernel's footprint ledger

    // The interval maximum resets to the current footprint and then tracks the kernel's maximum exactly.
    // Its reset has no public header; a missing symbol skips the class.
    private static let resetFootprintInterval: (@convention(c) (pid_t) -> Int32)? = {
        guard let symbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "proc_reset_footprint_interval") else {
            return nil
        }
        return unsafeBitCast(symbol, to: (@convention(c) (pid_t) -> Int32).self)
    }()

    private static func footprint() -> (now: Int, intervalPeak: Int) {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard status == 0 else { return (0, 0) }
        return (Int(info.ri_phys_footprint), Int(info.ri_interval_max_phys_footprint))
    }

    /// MLX's live and cached bytes.
    private static var mlxBytes: Int { NFKMLXGPU.activeMemory + NFKMLXGPU.cacheMemory }

    /// The footprint once memory MLX released has left it.
    ///
    /// @discussion Each reading drains the GPU and CPU streams, so command-buffer completions have
    /// returned their buffers to MLX's cache, then empties the cache. The footprint is read every 50 ms
    /// until MLX's cache reads 0, the footprint holds within 4 MiB for 200 ms, and, with
    /// `requiringQuiet`, the footprint less MLX's own bytes is within ``releasedTolerance`` of
    /// ``quietUnownedFootprint``. The settled reading lowers the quiet level when it is lower. Past 30 s
    /// it throws, naming the cache, the footprint, and the memory MLX no longer owns.
    private static func settledFootprint(requiringQuiet: Bool) throws -> Int {
        let deadline = Date().addingTimeInterval(30)
        var last = -1
        var steady = 0
        while true {
            Stream.gpu.synchronize()
            Stream.cpu.synchronize()
            NFKMLXGPU.clearCache()
            let now = footprint().now
            let cache = NFKMLXGPU.cacheMemory
            let unowned = now - mlxBytes
            steady = last >= 0 && abs(last - now) < 4 * mebibyte ? steady + 1 : 0
            last = now
            let quiet = !requiringQuiet || unowned - releasedTolerance <= quietUnownedFootprint
            if cache == 0, steady >= 4, quiet {
                quietUnownedFootprint = min(quietUnownedFootprint, unowned)
                return now
            }
            if Date() > deadline {
                throw NSError(domain: "NFKMLXLoadFootprintTests", code: 1, userInfo: [NSLocalizedDescriptionKey:
                    "the footprint did not settle in 30 s: MLX's cache holds \(cache) bytes, the footprint is "
                    + "\(now) bytes, MLX holds \(NFKMLXGPU.activeMemory) bytes active, and the footprint less "
                    + "MLX's own bytes is \(unowned), against a quiet level of \(quietUnownedFootprint) "
                    + "and a tolerance of \(releasedTolerance)"])
            }
            usleep(50_000)
        }
    }

    /// Runs the operations the class measures on small arrays, so the device, its queues, and the kernels
    /// exist before the quiet level is read.
    private func warmUp() throws {
        let url = directory.appendingPathComponent("warm-up.safetensors")
        let small = MLXRandom.uniform(low: -1, high: 1, [64, 64])
        try save(arrays: ["a": small, "b": small.asType(.bfloat16)], url: url)
        let read = try NFKMLXWeights.loadCheckpoint(url: url).arrays
        eval(read.values.flatMap { [$0.asType(.bfloat16), $0.asType(.float32)] })
        try FileManager.default.removeItem(at: url)
    }

    /// The bytes a load held above its result at its peak, with the result evaluated and held.
    private func transient(_ label: String, _ load: () throws -> [MLXArray]) throws -> Int {
        let before = try Self.settledFootprint(requiringQuiet: true)
        let unownedBefore = before - Self.mlxBytes
        _ = Self.resetFootprintInterval?(getpid())
        let result = try load()
        eval(result)
        let peak = Self.footprint().intervalPeak
        let after = Self.footprint().now
        let held = result.reduce(0) { $0 + $1.nbytes }
        let transient = peak - before - held
        print("FOOTPRINT \(label): start \(before / Self.mebibyte) MiB, peak \((peak - before) / Self.mebibyte) MiB "
              + "and end \((after - before) / Self.mebibyte) MiB above the start, result \(held / Self.mebibyte) MiB, "
              + "transient \(transient / Self.mebibyte) MiB, start beside MLX's own bytes \(unownedBefore / Self.mebibyte) MiB")
        XCTAssertGreaterThanOrEqual(peak - before + 64 * Self.mebibyte, held,
                                    "\(label): the measured peak must cover the result, or the start had not settled")
        return transient
    }

    // MARK: Synthetic releases

    /// Writes `shards` files of `tensorsPerShard` tensors each at `dtype`, with a shard index naming them.
    private func writeShardedRelease(shards: Int, tensorsPerShard: Int, dtype: DType) throws {
        var weightMap = [String: String]()
        for shard in 0 ..< shards {
            let file = "model-\(shard + 1)-of-\(shards).safetensors"
            var arrays = [String: MLXArray]()
            for index in 0 ..< tensorsPerShard {
                let name = "layers.\(shard * tensorsPerShard + index).weight"
                arrays[name] = MLXRandom.uniform(low: -1, high: 1, Self.tensorShape).asType(dtype)
                weightMap[name] = file
            }
            try save(arrays: arrays, url: directory.appendingPathComponent(file))
        }
        let index = try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
        try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        _ = try Self.settledFootprint(requiringQuiet: true)
    }

    // MARK: Budgets

    // A 2 GiB bfloat16 release widened to float32. Read-first, one group's reads are the transient; left
    // to one evaluation, the reads pile up beside the result. The lazy control proves the measurement
    // tells the two apart.
    func testAWideningReadHoldsOneGroupAboveItsResult() throws {
        try writeShardedRelease(shards: 2, tensorsPerShard: 32, dtype: .bfloat16)
        let budget = 2 * Self.groupBytes

        let readFirst = try transient("materializedArrays bf16 → f32") {
            try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: .float32).map(\.1)
        }
        XCTAssertLessThanOrEqual(readFirst, budget, "a read-first widening holds one group above its result")

        let lazy = try transient("lazy arrays bf16 → f32 (control)") {
            try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: .float32).map(\.1)
        }
        XCTAssertGreaterThan(lazy, budget,
                             "the lazy control must exceed the budget, or this measurement cannot see a regression")
    }

    // A 2 GiB float32 release narrowed to bfloat16: one group of reads beside the narrowed result.
    func testANarrowingReadHoldsOneGroupAboveItsResult() throws {
        try writeShardedRelease(shards: 2, tensorsPerShard: 16, dtype: .float32)
        let narrowed = try transient("arrays converting f32 → bf16") {
            try NFKMLXReleaseWeights.arrays(inDirectory: directory, converting: .bfloat16).map(\.1)
        }
        XCTAssertLessThanOrEqual(narrowed, 2 * Self.groupBytes, "a narrowing read holds one group above its result")
    }

    // A 2 GiB bfloat16 release kept at its stored type: each read lands in the buffer the result keeps.
    func testAStoredTypeReadHoldsLittleAboveItsResult() throws {
        try writeShardedRelease(shards: 2, tensorsPerShard: 32, dtype: .bfloat16)
        let stored = try transient("materializedArrays at the checkpoint's type") {
            try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: .checkpoint).map(\.1)
        }
        XCTAssertLessThanOrEqual(stored, Self.groupBytes, "a stored-type read holds little above its result")
    }

    // A 2 GiB single-file checkpoint read first: the arrays the checkpoint keeps are the result.
    func testAReadFirstCheckpointHoldsLittleAboveItsResult() throws {
        var arrays = [String: MLXArray]()
        for index in 0 ..< 64 {
            arrays["layers.\(index).weight"] = MLXRandom.uniform(low: -1, high: 1, Self.tensorShape).asType(.bfloat16)
        }
        let url = directory.appendingPathComponent("model.safetensors")
        try save(arrays: arrays, url: url)
        arrays.removeAll()
        _ = try Self.settledFootprint(requiringQuiet: true)
        let single = try transient("materializedCheckpoint") {
            Array(try NFKMLXWeights.materializedCheckpoint(url: url).arrays.values)
        }
        XCTAssertLessThanOrEqual(single, Self.groupBytes, "a read-first checkpoint holds little above its result")
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }
}
