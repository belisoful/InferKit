//
//  NFKMLXReleaseWeightsTests.swift
//  InferKitMLXTests
//
//  The fit-before-load predicate: a release whose weights exceed the memory budget is refused before
//  any tensor is materialized, so a load that would kill the process becomes an error naming the
//  shortfall. Pure file-size arithmetic, so this runs under `swift test`.
//

import XCTest
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
}
