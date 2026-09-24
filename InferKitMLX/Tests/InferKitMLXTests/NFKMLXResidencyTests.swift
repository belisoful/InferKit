//
//  NFKMLXResidencyTests.swift
//  InferKitMLXTests
//

import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXResidencyTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private let gib = 1 << 30

    // Two questions, two answers on a machine that reports no budget: holding everything resident is
    // not risked there, and a single stage is not refused there.
    func testTheBudgetAnswersHoldingAndLoadingDifferently() {
        let budget = Int(21.25 * Double(gib))
        XCTAssertTrue(NFKMLXResidencyBudget.holds(17 * gib, budget: budget))
        XCTAssertFalse(NFKMLXResidencyBudget.holds(18 * gib, budget: budget),
                       "the 4 GiB reserve is part of the arithmetic")
        XCTAssertFalse(NFKMLXResidencyBudget.holds(1 * gib, budget: 0),
                       "a machine that reports no budget is not held resident")
        XCTAssertTrue(NFKMLXResidencyBudget.admits(1 * gib, budget: 0),
                      "and a stage is not refused there either")
        XCTAssertFalse(NFKMLXResidencyBudget.admits(18 * gib, budget: budget))
    }

    func testEachResidencyDecidesAsItSays() throws {
        let budget = Int(21.25 * Double(gib))
        XCTAssertTrue(try NFKMLXResidencyBudget.holdsResident(9 * gib, residency: .automatic, budget: budget))
        XCTAssertFalse(try NFKMLXResidencyBudget.holdsResident(28 * gib, residency: .automatic, budget: budget))
        XCTAssertFalse(try NFKMLXResidencyBudget.holdsResident(1 * gib, residency: .staged, budget: budget),
                       "staged never holds, even where everything would fit")
        XCTAssertTrue(try NFKMLXResidencyBudget.holdsResident(9 * gib, residency: .resident, budget: budget))
        XCTAssertThrowsError(try NFKMLXResidencyBudget.holdsResident(28 * gib, residency: .resident,
                                                                     budget: budget),
                             "asked to hold what is known not to fit, it says so rather than loading")
        XCTAssertTrue(try NFKMLXResidencyBudget.holdsResident(28 * gib, residency: .resident, budget: 0),
                      "an explicit resident request on a machine that reports no budget is honored")
    }

    // A resident model loads each stage once and holds it; a staged one loads each on every use and
    // holds nothing; switching a model from resident to staged releases what it held.
    func testStagesLoadOnUseAndAreHeldOnlyWhenResident() throws {
        try requireMLXRuntime()
        var loads = 0
        let stage = NFKMLXStage<MLXArray> {
            loads += 1
            return MLXArray([Float(1), 2, 3])
        }
        let model = NFKMLXStagedModel(resident: true)
        for _ in 0 ..< 3 {
            XCTAssertEqual(try model.with(stage) { $0 * 2 }.asArray(Float.self), [2, 4, 6])
        }
        XCTAssertEqual(loads, 1, "a resident model loads a stage once")
        XCTAssertTrue(stage.isHeld)

        var released = 0
        model.setResident(false, releasing: [{ released += 1; stage.release() }])
        XCTAssertEqual(released, 1, "leaving residency releases the stages")
        XCTAssertFalse(stage.isHeld)
        loads = 0
        for _ in 0 ..< 3 {
            XCTAssertEqual(try model.with(stage) { [$0 + 1] }[0].asArray(Float.self), [2, 3, 4])
        }
        XCTAssertEqual(loads, 3, "a staged model loads a stage on every use")
        XCTAssertFalse(stage.isHeld, "and holds it after none of them")
        model.setResident(false, releasing: [{ released += 1 }])
        XCTAssertEqual(released, 1, "staying staged releases nothing more")
    }
}
