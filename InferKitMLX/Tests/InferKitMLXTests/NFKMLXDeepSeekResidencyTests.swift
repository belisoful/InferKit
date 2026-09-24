//
//  NFKMLXDeepSeekResidencyTests.swift
//  InferKitMLXTests
//
//  How a residency comes to a DeepSeek paging policy, on the released V4.1 Flash geometry, with
//  budgets derived from its own byte counts so each branch of the plan is reached. Arithmetic only.
//

import XCTest
@testable import InferKitMLX

final class NFKMLXDeepSeekResidencyTests: XCTestCase {

    private let flash = NFKMLXDeepSeekConfiguration.v41Flash
    private let reserve = NFKMLXResidencyBudget.reserve
    private let gib = 1 << 30

    private func paging(_ residency: NFKMLXResidency, _ budget: Int) throws -> NFKMLXDeepSeekPaging {
        try NFKMLXDeepSeek.paging(for: flash, residency: residency, budget: budget, includesDraftStack: false)
    }

    func testAutomaticPagesOnlyWhatDoesNotFitAndHoldsStoredWhereThatFits() throws {
        let whole = NFKMLXDeepSeek.decoderBytes(for: flash, paging: .none)
        let stored = NFKMLXDeepSeek.decoderBytes(for: flash, paging: .mapped)
        let mapped = NFKMLXDeepSeek.decoderBytes(for: flash, paging: .fullyMapped)

        XCTAssertEqual(try paging(.automatic, whole + reserve), .none, "a machine that holds the decoded weights pages nothing")

        // The cache is half of what the fully mapped decoder leaves; the experts held stored fit beside it here.
        let roomy = 2 * stored - mapped + reserve + gib
        XCTAssertLessThan(roomy, whole + reserve)
        let held = try paging(.automatic, roomy)
        XCTAssertTrue(held.routedExperts && held.ngramTables && held.mapsNgramTables && !held.mapsRoutedExperts,
                      "the experts are held stored, decoding without reading the release")
        XCTAssertEqual(held.expertCacheBytes, (roomy - reserve - mapped) / 2)

        let tight = try paging(.automatic, mapped + reserve + gib)
        XCTAssertEqual(tight.mapsRoutedExperts, true, "short of that, every paged group stays in the release")
        XCTAssertEqual(tight.expertCacheBytes, gib / 2)

        XCTAssertThrowsError(try paging(.automatic, mapped + reserve - gib), "fully mapped and still too large is refused")
        XCTAssertEqual(try paging(.automatic, 0), .none, "a machine that reports no budget is never paged")
    }

    func testEachExplicitResidencyDecidesAsItSays() throws {
        let whole = NFKMLXDeepSeek.decoderBytes(for: flash, paging: .none)
        XCTAssertThrowsError(try paging(.resident, 64 * gib), "resident refuses what is known not to fit")
        XCTAssertEqual(try paging(.resident, whole + reserve), .none)
        XCTAssertEqual(try paging(.staged, 64 * gib), .none, "one stage has nothing to take turns with")
        let paged = try paging(.paged, whole + reserve)
        XCTAssertTrue(paged.mapsRoutedExperts && paged.mapsNgramTables, "paged leaves every paged group in the release")
    }
}
