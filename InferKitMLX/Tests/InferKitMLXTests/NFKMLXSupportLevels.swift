//
//  NFKMLXSupportLevels.swift
//  InferKitMLX tests
//

import XCTest

/// The released-weight tests that a support-level change retired, keyed by release.
///
/// `Docs/removal-candidates.md` defines the levels. A release listed here keeps its code and its
/// tiny-geometry and unit tests; only its released-weight tests stop running. Level 2 keeps one
/// released size, so the table names the sizes it retires. Level 3 keeps no released-weight test, so
/// the table names every release of the model. Setting `IK_TEST_RETIRED=1` runs the retired tests anyway.
enum NFKMLXSupportLevels {
    static let retired: [String: Int] = [
        "colorizer-eccv16": 3,
        "zero-dce": 3,
        "ltx-video-0.9.0": 3,
        "sam-vit-b": 3,
        "sam-vit-l": 3,
        "sam-vit-h": 3,
        "hunyuan-mt-7b": 3,
        "whisper-large-v2": 3,
        "yolov8s": 2,
        "yolov8m": 2,
        "yolov8l": 2,
        "yolov8x": 2,
        "depth-anything-v2-base": 2,
        "depth-anything-v2-large": 2,
        "gemma-2-9b": 2,
        "gemma-2-27b": 2,
        "m2m100-1.2b": 2,
        "gemma3n-e4b": 2,
        "rtdetr-r18vd": 2,
        "rtdetr-r34vd": 2,
        "rtdetr-r101vd": 2,
    ]

    /// Whether the released-weight tests of `release` are retired in this run.
    static func isRetired(_ release: String) -> Bool {
        guard NFKMLXValidationConfig.environment["IK_TEST_RETIRED"] != "1" else {
            return false
        }
        return retired[release] != nil
    }

    /// Skips the calling test when every release it checks is retired.
    static func skipIfRetired(_ releases: String...) throws {
        guard releases.allSatisfy(isRetired) else {
            return
        }
        let levels = releases.map { "\($0) (level \(retired[$0] ?? 0))" }.joined(separator: ", ")
        throw XCTSkip("retired by support level: \(levels); set IK_TEST_RETIRED=1 to run it")
    }
}
