//
//  NFKMLXReleaseWeightsTests.swift
//  InferKitMLXTests
//
//  The fit-before-load predicate: a release whose weights exceed the memory budget is refused before
//  any tensor is materialized, so a load that would kill the process becomes an error naming the
//  shortfall. Pure file-size arithmetic, so this runs under `swift test`. The readers' conversions
//  are held to the CPU's bits, which needs MLX.
//

import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXReleaseWeightsTests: XCTestCase {

    private func makeRelease(bytes: Int) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(count: bytes).write(to: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    func testWeightBytesSumsTheShardFiles() throws {
        let directory = try makeRelease(bytes: 4096)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertEqual(try NFKMLXReleaseWeights.weightBytes(inDirectory: directory), 4096)
    }

    func testAReleaseLargerThanTheBudgetIsRefused() throws {
        let directory = try makeRelease(bytes: 1_000_000)          // 1 MB stored
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertThrowsError(try NFKMLXReleaseWeights.verifyFits(
            inDirectory: directory, precision: .checkpoint, budget: 512_000)) { error in
            let detail = (error as? NFKMLXError)?.errorDescription ?? ""
            XCTAssertTrue(detail.contains("working set"), "expected the shortfall message: \(detail)")
        }
        XCTAssertNoThrow(try NFKMLXReleaseWeights.verifyFits(
            inDirectory: directory, precision: .checkpoint, budget: 2_000_000))
    }

    func testAFloat32LoadDoublesAHalfPrecisionReleaseInTheEstimate() throws {
        let directory = try makeRelease(bytes: 1_000_000)          // 1 MB stored → 2 MB at float32
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertThrowsError(try NFKMLXReleaseWeights.verifyFits(
            inDirectory: directory, precision: .float32, budget: 1_500_000))
        XCTAssertNoThrow(try NFKMLXReleaseWeights.verifyFits(
            inDirectory: directory, precision: .checkpoint, budget: 1_500_000))
    }

    func testAnUnknownBudgetDoesNotGateTheLoad() throws {
        let directory = try makeRelease(bytes: 1_000_000)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertNoThrow(try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, budget: 0))
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func words(_ array: MLXArray) -> [UInt32] {
        array.view(dtype: array.dtype.size == 4 ? .uint32 : .uint16, stream: .cpu)
            .asType(.uint32, stream: .cpu).asArray(UInt32.self)
    }

    /// `array`'s words with every NaN replaced by one marker: the GPU keeps a NaN a NaN but may change
    /// its sign and payload.
    private func comparable(_ array: MLXArray) -> [UInt32] {
        let (exponent, mantissa): (UInt32, UInt32) = switch array.dtype {
        case .float32: (0x7F80_0000, 0x007F_FFFF)
        case .bfloat16: (0x7F80, 0x007F)
        default: (0x7C00, 0x03FF)
        }
        return words(array).map { $0 & exponent == exponent && $0 & mantissa != 0 ? .max : $0 }
    }

    // Subnormals of each sign, the smallest normals, float32 values that round into float16's subnormal
    // range or past its largest finite value, infinities, a NaN, and negative zero, in each float type.
    func testTheReadersConvertEdgeValuesToTheCPUsBits() throws {
        try requireMLXRuntime()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let single: [UInt32] = [0x0000_0001, 0x0040_0000, 0x8000_0001, 0x0080_0000, 0x3380_0000, 0x387F_C000,
                                0x3F80_0001, 0x7F7F_FFFF, 0x7F80_0000, 0x7FC0_0000, 0x8000_0000]
        let brain: [UInt16] = [0x0001, 0x007F, 0x8001, 0x0080, 0x3F81, 0x7F7F, 0x7F80, 0x7FC0, 0x8000]
        let half: [UInt16] = [0x0001, 0x03FF, 0x8001, 0x0400, 0x3C01, 0x7BFF, 0x7C00, 0x7E00, 0x8000]
        let stored = ["single": MLXArray(single).view(dtype: .float32),
                      "brain": MLXArray(brain).view(dtype: .bfloat16),
                      "half": MLXArray(half).view(dtype: .float16)]
        try save(arrays: stored, url: directory.appendingPathComponent("model.safetensors"))
        func expected(_ key: String, _ dtype: DType) throws -> [UInt32] {
            comparable(try XCTUnwrap(stored[key]).asType(dtype, stream: .cpu))
        }

        let widened = Dictionary(uniqueKeysWithValues: try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory))
        for key in ["brain", "half"] {
            XCTAssertEqual(comparable(try XCTUnwrap(widened[key])), try expected(key, .float32), "\(key) to float32")
        }
        XCTAssertNotEqual(words(try XCTUnwrap(widened["brain"]))[0], 0, "a bfloat16 subnormal survives widening")
        for dtype in [DType.bfloat16, .float16] {
            let narrowed = Dictionary(uniqueKeysWithValues: try NFKMLXReleaseWeights.arrays(inDirectory: directory,
                                                                                            converting: dtype))
            for key in ["single", "brain", "half"] where stored[key]?.dtype != dtype {
                XCTAssertEqual(comparable(try XCTUnwrap(narrowed[key])), try expected(key, dtype), "\(key) to \(dtype)")
            }
        }
    }
}
