//
//  NFKMLXRuntime.swift
//  InferKitMLX
//
//  Objective-C access to MLX's global runtime knobs. mlx-swift exposes these as free functions and a
//  `GPU` enum, neither of which bridges to Objective-C (free functions and Swift enums are not `@objc`).
//  These thin `NSObject` wrappers make the ones an Objective-C consumer needs — reproducible seeding and
//  GPU memory management — callable from Objective-C. A Swift caller can keep using `MLX.seed` / `MLX.GPU`
//  directly.
//

import Foundation
import MLX

/// Objective-C access to MLX's global random seed. Seeding makes weight initialization and any sampling
/// reproducible across runs, so two runs with the same seed produce identical results.
@objc(NFKMLXRandom)
public final class NFKMLXRandom: NSObject {

    /// Seeds MLX's global random state. Objective-C: `[NFKMLXRandom seed:42]`.
    @objc public static func seed(_ value: UInt64) {
        MLX.seed(value)
    }
}

/// Objective-C access to MLX GPU memory management: the cache and soft-memory limits, cache clearing, and
/// live memory statistics. A production app running large models uses these to cap memory and avoid
/// out-of-memory, and to observe usage. All values are in bytes.
@objc(NFKMLXGPU)
public final class NFKMLXGPU: NSObject {

    /// Bytes currently allocated to live arrays.
    @objc public static var activeMemory: Int { MLX.Memory.activeMemory }

    /// Bytes held in the reusable buffer cache (not yet returned to the system).
    @objc public static var cacheMemory: Int { MLX.Memory.cacheMemory }

    /// The high-water mark of active memory since the last ``resetPeakMemory()``.
    @objc public static var peakMemory: Int { MLX.Memory.peakMemory }

    /// The current buffer-cache size limit.
    @objc public static var cacheLimit: Int { MLX.Memory.cacheLimit }

    /// The current soft memory limit.
    @objc public static var memoryLimit: Int { MLX.Memory.memoryLimit }

    /// Sets the buffer-cache size limit in bytes (0 disables caching).
    @objc public static func setCacheLimit(_ bytes: Int) {
        MLX.Memory.cacheLimit = bytes
    }

    /// Sets the soft memory limit in bytes; allocations beyond it try to free cache first.
    @objc public static func setMemoryLimit(_ bytes: Int) {
        MLX.Memory.memoryLimit = bytes
    }

    /// Returns the buffer cache to the system.
    @objc public static func clearCache() {
        MLX.Memory.clearCache()
    }

    /// Resets ``peakMemory`` to the current active memory.
    @objc public static func resetPeakMemory() {
        // The one `MLX.GPU` member this file still uses: the memory getters and limits all moved to
        // `MLX.Memory`, but it has no reset, and this call carries no deprecation attribute.
        MLX.GPU.resetPeakMemory()
    }

    // MARK: What the machine has

    /// The host's physical memory in bytes.
    ///
    /// @discussion Apple Silicon is unified, so this is the pool the weights, the key-value cache and
    /// the rest of the app all draw from. It is the honest ceiling for "will this model fit", where
    /// ``memoryLimit`` is only what MLX has been told to respect.
    @objc public static var physicalMemory: Int { MLX.GPU.deviceInfo().memorySize }

    /// Metal's recommended working-set size in bytes, or 0 when there is no Metal device.
    ///
    /// @discussion This is the amount Metal expects a process to keep resident without the system
    /// starting to evict, so it is the budget a model should be sized against rather than
    /// ``physicalMemory``. It is well below the physical total on every configuration.
    @objc public static var recommendedWorkingSetSize: Int {
        MLX.GPU.maxRecommendedWorkingSetBytes() ?? 0
    }

    /// Bytes that could be returned to the system right now without freeing a live array — the buffer
    /// cache, which ``clearCache()`` reclaims.
    ///
    /// @discussion A caller deciding whether to unload a model asks this first: cache is reclaimable,
    /// active memory is not, and the two are worth distinguishing before evicting anything.
    @objc public static var reclaimableMemory: Int { MLX.Memory.cacheMemory }

    /// How much of ``recommendedWorkingSetSize`` is already spoken for, in `0...1`, or a negative
    /// value when Metal reports no recommendation.
    ///
    /// @discussion Crossing a watermark is the signal to unload a model or shrink a prefill chunk.
    /// The numerator is active memory alone, since the cache is reclaimable.
    @objc public static var memoryPressure: Double {
        let budget = recommendedWorkingSetSize
        guard budget > 0 else { return -1 }
        return Double(activeMemory) / Double(budget)
    }

    /// The GPU architecture name Metal reports, for a log line or a bug report.
    @objc public static var deviceArchitecture: String { MLX.GPU.deviceInfo().architecture }

    // MARK: Standing limits

    /// The default standing cache cap: 256 MB.
    ///
    /// @discussion MLX's buffer cache grows to hold whatever the largest recent allocation needed, and
    /// it is not returned between models. A process that loads several models in sequence accumulates
    /// that, and the accumulation starves the next large forward — which surfaces as a Metal
    /// command-buffer timeout, a process kill rather than an error. A standing cap is the version of
    /// ``clearCache()`` that does not have to be remembered at every boundary.
    @objc public static let defaultCacheCap = 256 * 1024 * 1024

    /// Applies a standing cache cap of ``defaultCacheCap`` and a soft memory limit at 85% of Metal's
    /// recommended working set.
    ///
    /// @discussion An app that loads more than one model calls this once at startup. The cache cap
    /// bounds what MLX holds between allocations; the memory limit means allocations past it free
    /// cache before growing.
    ///
    /// Introduced in InferKit 0.1.0.
    ///
    /// Objective-C: `[NFKMLXGPU applyStandingLimits]`, or
    /// `applyStandingLimitsWithCacheBytes:fractionOfRecommendedWorkingSet:` to choose them. The two
    /// are separate methods rather than one with defaults, because a Swift default argument does not
    /// bridge — an Objective-C caller would otherwise have to spell out both every time.
    @objc public static func applyStandingLimits() {
        applyStandingLimits(cacheBytes: defaultCacheCap, fractionOfRecommendedWorkingSet: 0.85)
    }

    /// Applies a standing cache cap and a soft memory limit, both chosen by the caller.
    ///
    /// - Parameters:
    ///   - cacheBytes: the buffer-cache cap.
    ///   - fractionOfRecommendedWorkingSet: the share of Metal's recommendation to allow as the soft
    ///     limit, in `0...1`. Pass 0 to leave the memory limit alone.
    ///
    /// Introduced in InferKit 0.1.0.
    @objc(applyStandingLimitsWithCacheBytes:fractionOfRecommendedWorkingSet:)
    public static func applyStandingLimits(cacheBytes: Int,
                                           fractionOfRecommendedWorkingSet: Double) {
        setCacheLimit(max(cacheBytes, 0))
        let budget = recommendedWorkingSetSize
        let fraction = min(max(fractionOfRecommendedWorkingSet, 0), 1)
        if budget > 0 && fraction > 0 {
            setMemoryLimit(Int(Double(budget) * fraction))
        }
    }

    // MARK: Wired memory

    /// Runs `body` with a raised wired-memory limit, which keeps `bytes` of the working set resident
    /// rather than pageable for the duration.
    ///
    /// @discussion Wiring the weights of a model that is about to run repeatedly stops the system
    /// paging them back out between calls. The limit is scoped: it is released when `body` returns,
    /// so a caller cannot leak a raised limit past the work that needed it.
    ///
    /// This is deliberately **not** exposed to Objective-C, and there is deliberately no
    /// `setWiredLimit:`. mlx-swift 0.31.6 admits a wired limit only through an async, scoped ticket —
    /// its synchronous `withWiredLimit` is deprecated and a documented no-op — so a persistent setter
    /// could only be a knob that silently did nothing.
    ///
    /// Introduced in InferKit 0.1.0.
    public static func withWiredLimit<R>(_ bytes: Int,
                                         _ body: () async throws -> R) async rethrows -> R {
        let ticket = WiredMemoryTicket(size: max(bytes, 0), policy: wiredPolicy,
                                       manager: .shared, kind: .active)
        return try await ticket.withWiredLimit(body)
    }

    // One policy identity for the whole package, so concurrent tickets sum against each other rather
    // than each admitting itself in isolation. A fresh `WiredSumPolicy()` per call would mint a new
    // identity every time and defeat the manager's accounting.
    private static let wiredPolicy = WiredSumPolicy(
        id: UUID(uuidString: "3F2A6C41-9E5B-4D18-A7C0-5B2E1F84D963")!)
}
