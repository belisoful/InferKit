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
}
