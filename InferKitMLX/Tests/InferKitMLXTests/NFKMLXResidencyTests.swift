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

    // A stage's wider precision is taken where the placement affords it: beside the others when held
    // resident, on its own when staged, and never on a machine that reports no budget.
    func testAWiderPrecisionIsTakenWhereThePlacementAffordsIt() throws {
        let budget = Int(21.25 * Double(gib))
        func plan(_ encoder: Int, _ pipeline: Int, _ budget: Int,
                  _ residency: NFKMLXResidency) throws -> NFKMLXResidencyPlan {
            try NFKMLXResidencyBudget.plan([NFKMLXStageFootprint(bytes: encoder * gib, widenedBytes: 2 * encoder * gib),
                                            NFKMLXStageFootprint(bytes: pipeline * gib)],
                                           residency: residency, budget: budget)
        }
        let small = try plan(4, 5, budget, .automatic)
        XCTAssertTrue(small.holdsStagesResident)
        XCTAssertTrue(small.widens(0), "8 + 5 widened still fits beside the reserve")
        let tight = try plan(7, 7, budget, .automatic)
        XCTAssertEqual(tight, .resident, "held resident as stored; widened, 21 would not fit")
        let staged = try plan(8, 12, budget, .automatic)
        XCTAssertFalse(staged.holdsStagesResident)
        XCTAssertTrue(staged.widens(0), "staged, the encoder fits alone at 16")
        XCTAssertFalse(staged.widens(1), "a stage with one precision never widens")
        XCTAssertEqual(try plan(8, 12, 0, .automatic), .staged, "no budget, no wider precision")
        XCTAssertEqual(try plan(8, 12, 0, .resident), .resident)
    }

    // A staged placement refuses a stage that cannot load on its own, at the precision the plan chose.
    func testEachStageOfAStagedPlacementMustLoad() throws {
        let budget = Int(21.25 * Double(gib))
        let stages = [NFKMLXStageFootprint(bytes: 8 * gib, widenedBytes: 16 * gib), NFKMLXStageFootprint(bytes: 20 * gib)]
        let placement = try NFKMLXResidencyBudget.plan(stages, residency: .automatic, budget: budget)
        XCTAssertThrowsError(try NFKMLXResidencyBudget.verifyEachStageLoads(stages, plan: placement, budget: budget,
                                                                           names: ["encoder", "transformer"]))
        let loadable = [stages[0], NFKMLXStageFootprint(bytes: 12 * gib)]
        XCTAssertNoThrow(try NFKMLXResidencyBudget.verifyEachStageLoads(
            loadable, plan: try NFKMLXResidencyBudget.plan(loadable, residency: .staged, budget: budget),
            budget: budget, names: []))
        XCTAssertNoThrow(try NFKMLXResidencyBudget.verifyEachStageLoads(stages, plan: placement, budget: 0, names: []),
                         "a machine that reports no budget refuses no stage")
    }
}
