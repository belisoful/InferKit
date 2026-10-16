//
//  NFKMLXPlannerDifferentialTests.swift
//  InferKitMLXTests
//
//  The residency planner moved to the core (NFKModelResidencyPlan), and NFKMLXResidencyBudget calls
//  through to it. These tests hold the Swift planner as it was before the move and compare it with the
//  core planner, and with the call-through, over a grid of footprints, residencies, and budgets: the
//  plans are equal, and a refusal is the same sentence.
//

import XCTest
import InferKit
@testable import InferKitMLX

/// The Swift planner as it stood before it moved to the core, kept as the reference the differential
/// tests compare against.
private enum NFKMLXLegacyPlanner {

    struct Plan: Equatable {
        var holdsStagesResident: Bool
        var pagesExperts: Bool
        var expertCacheBytes: Int
        var widenedStages: Set<Int> = []
        var streamedStages = [Int: Int]()

        static let resident = Plan(holdsStagesResident: true, pagesExperts: false, expertCacheBytes: 0)
        static let staged = Plan(holdsStagesResident: false, pagesExperts: false, expertCacheBytes: 0)

        func widens(_ index: Int) -> Bool { widenedStages.contains(index) }

        func streams(_ index: Int) -> Bool { streamedStages[index] != nil }

        func heldLayers(_ index: Int, of stage: NFKMLXStageFootprint) -> Range<Int> {
            guard let room = streamedStages[index] else { return 0 ..< stage.layerBytes.count }
            var total = 0
            var count = 0
            for bytes in stage.layerBytes {
                guard total + bytes <= room else { break }
                total += bytes
                count += 1
            }
            return 0 ..< count
        }
    }

    struct Refusal: Error, Equatable {
        let sentence: String
    }

    static let reserve = 4 << 30
    static let workingSetFraction = 0.85
    static let defaultExpertCacheBytes = 4 << 30

    static func leastWorkingSet(holding bytes: Int64) -> Int64 {
        Int64((Double(bytes + Int64(reserve)) / workingSetFraction).rounded(.up))
    }

    static func holds(_ bytes: Int, budget: Int) -> Bool {
        budget > 0 && bytes + reserve <= budget
    }

    static func admits(_ bytes: Int, budget: Int) -> Bool {
        budget <= 0 || bytes + reserve <= budget
    }

    static func plan(_ stages: [NFKMLXStageFootprint], residency: NFKMLXResidency, budget: Int) throws -> Plan {
        let total = stages.reduce(0) { $0 + $1.bytes }
        let largest = stages.max { $0.bytes < $1.bytes } ?? NFKMLXStageFootprint(bytes: 0)
        let largestUnpaged = stages.map(\.unpagedBytes).max() ?? 0
        let pageable = stages.contains { $0.pageableBytes > 0 }
        func paged() throws -> Plan {
            guard admits(largestUnpaged, budget: budget) else {
                throw Refusal(sentence:
                    "the largest stage needs \(gib(largestUnpaged)) with its routed experts paged, plus "
                    + "a \(gib(reserve)) reserve, against a \(gib(budget)) budget")
            }
            let largestPageable = stages.map(\.pageableBytes).max() ?? 0
            let headroom = budget > 0 ? (budget - reserve - largestUnpaged) / 2 : defaultExpertCacheBytes
            return Plan(holdsStagesResident: false, pagesExperts: true,
                        expertCacheBytes: max(0, min(largestPageable, headroom)))
        }
        let streamable = stages.indices.filter { !stages[$0].layerBytes.isEmpty }
        let oversized = streamable.filter { budget <= 0 || !admits(stages[$0].bytes, budget: budget) }
        func streamed() throws -> Plan {
            var plan = Plan.staged
            for index in oversized {
                let minimum = stages[index].streamedMinimumBytes
                guard admits(minimum, budget: budget) else {
                    throw Refusal(sentence:
                        "stage \(index) needs \(gib(minimum)) streamed, two layers in flight, plus a "
                        + "\(gib(reserve)) reserve, against a \(gib(budget)) budget")
                }
                plan.streamedStages[index] = budget > 0 ? budget - reserve - minimum : 0
            }
            return plan
        }
        let placement: Plan
        switch residency {
        case .resident:
            guard admits(total, budget: budget) else {
                throw Refusal(sentence:
                    "the stages need \(gib(total)) together plus a \(gib(reserve)) reserve, against a "
                    + "\(gib(budget)) budget; hold them staged"
                    + (pageable ? " or paged" : "") + (streamable.isEmpty ? "" : " or streamed"))
            }
            placement = .resident
        case .staged:
            placement = .staged
        case .paged:
            placement = pageable ? try paged() : .staged
        case .streamed:
            placement = try streamed()
        case .automatic:
            if holds(total, budget: budget) {
                placement = .resident
            } else if pageable, budget > 0, !holds(largest.bytes, budget: budget) {
                placement = try paged()
            } else if budget > 0, !oversized.isEmpty {
                placement = try streamed()
            } else {
                placement = .staged
            }
        }
        var widened = placement
        widened.widenedStages = widenedStages(stages, resident: placement.holdsStagesResident, budget: budget)
        return widened
    }

    private static func widenedStages(_ stages: [NFKMLXStageFootprint], resident: Bool, budget: Int) -> Set<Int> {
        var total = stages.reduce(0) { $0 + $1.bytes }
        var widened = Set<Int>()
        for (index, stage) in stages.enumerated() {
            guard let wide = stage.widenedBytes else { continue }
            let fits = resident ? holds(total - stage.bytes + wide, budget: budget) : holds(wide, budget: budget)
            if fits {
                widened.insert(index)
                total += wide - stage.bytes
            }
        }
        return widened
    }

    static func verifyEachStageLoads(_ stages: [NFKMLXStageFootprint], plan: Plan,
                                     budget: Int, names: [String]) throws {
        guard !plan.holdsStagesResident else { return }
        for (index, stage) in stages.enumerated() where !plan.streams(index) {
            let bytes = plan.widens(index) ? stage.widenedBytes ?? stage.bytes
                : (plan.pagesExperts ? stage.unpagedBytes : stage.bytes)
            guard admits(bytes, budget: budget) else {
                let name = index < names.count ? names[index] : "stage \(index)"
                throw Refusal(sentence:
                    "\(name) alone needs \(gib(bytes)) plus a \(gib(reserve)) reserve, against a "
                    + "\(gib(budget)) budget")
            }
        }
    }

    static func gib(_ bytes: Int) -> String {
        String(format: "%.1f GiB", Double(bytes) / 1_073_741_824)
    }
}

final class NFKMLXPlannerDifferentialTests: XCTestCase {

    private static let gib = 1 << 30
    private static let mib = 1 << 20

    private static let residencies: [NFKMLXResidency] = [.automatic, .resident, .staged, .paged, .streamed]

    /// Budgets from no budget to a large Mac, with uneven ones between the round figures, so the plans cross
    /// every threshold: 0 and below report none.
    private static let budgets: [Int] = {
        var budgets = [-1, 0, 1, 4 * gib, 4 * gib + 1, 6 * gib, 8 * gib, 12 * gib, 13 * gib + 517 * mib,
                       16 * gib, 21 * gib + 256 * mib, 24 * gib, 27 * gib + 211 * mib, 32 * gib, 48 * gib,
                       54 * gib + 409 * mib, 64 * gib, 96 * gib, 108 * gib + 819 * mib, 128 * gib, 192 * gib,
                       217 * gib + 614 * mib, 512 * gib]
        for working in [8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192] {
            budgets.append(Int(Double(working * gib) * 0.85))
        }
        return budgets
    }()

    /// Footprints that exercise every branch: one dense stage small and large, mixtures whose experts page,
    /// dense stages that stream, staged pipelines with a stage that widens, mixed pipelines, and no stages.
    private static let footprints: [[NFKMLXStageFootprint]] = {
        let layers40 = Array(repeating: 450 * mib, count: 40)
        let unevenLayers = (0 ..< 24).map { (300 + 37 * $0) * mib }
        return [
            [],
            [NFKMLXStageFootprint(bytes: 0)],
            [NFKMLXStageFootprint(bytes: 2 * gib)],
            [NFKMLXStageFootprint(bytes: 9 * gib)],
            [NFKMLXStageFootprint(bytes: 40 * gib)],
            [NFKMLXStageFootprint(bytes: 30 * gib, pageableBytes: 24 * gib)],
            [NFKMLXStageFootprint(bytes: 60 * gib, pageableBytes: 58 * gib)],
            [NFKMLXStageFootprint(bytes: 120 * gib, pageableBytes: 20 * gib)],
            [NFKMLXStageFootprint(bytes: 13 * gib, layerBytes: Array(repeating: gib, count: 10))],
            [NFKMLXStageFootprint(bytes: 20 * gib, layerBytes: layers40)],
            [NFKMLXStageFootprint(bytes: 9 * gib, layerBytes: unevenLayers)],
            [NFKMLXStageFootprint(bytes: 64 * gib, layerBytes: Array(repeating: 2 * gib, count: 30))],
            [NFKMLXStageFootprint(bytes: 8 * gib, widenedBytes: 16 * gib), NFKMLXStageFootprint(bytes: 20 * gib)],
            [NFKMLXStageFootprint(bytes: 9 * gib, widenedBytes: 18 * gib), NFKMLXStageFootprint(bytes: 24 * gib),
             NFKMLXStageFootprint(bytes: 300 * mib, widenedBytes: 600 * mib)],
            [NFKMLXStageFootprint(bytes: 5 * gib, widenedBytes: 10 * gib),
             NFKMLXStageFootprint(bytes: 3 * gib, widenedBytes: 6 * gib),
             NFKMLXStageFootprint(bytes: 1 * gib)],
            [NFKMLXStageFootprint(bytes: 15 * gib, layerBytes: layers40),
             NFKMLXStageFootprint(bytes: 30 * gib, pageableBytes: 26 * gib)],
            [NFKMLXStageFootprint(bytes: 2 * gib, widenedBytes: 4 * gib),
             NFKMLXStageFootprint(bytes: 26 * gib, layerBytes: Array(repeating: 600 * mib, count: 36)),
             NFKMLXStageFootprint(bytes: 1 * gib, pageableBytes: 512 * mib)],
            [NFKMLXStageFootprint(bytes: 7 * gib, pageableBytes: 6 * gib, widenedBytes: 14 * gib,
                                  layerBytes: Array(repeating: 200 * mib, count: 30))],
        ]
    }()

    private static let names = ["text encoder", "transformer"]

    private func coreFootprints(_ stages: [NFKMLXStageFootprint]) -> [NFKModelStageFootprint] {
        stages.map {
            NFKModelStageFootprint(byteCount: $0.bytes, pageableByteCount: $0.pageableBytes,
                                   widenedByteCount: $0.widenedBytes.map(NSNumber.init(value:)),
                                   layerByteCounts: $0.layerBytes.map(NSNumber.init(value:)))
        }
    }

    private func residencyName(_ residency: NFKMLXResidency) -> String {
        switch residency {
        case .automatic: return NFKModelResidencyAutomatic
        case .resident: return NFKModelResidencyResident
        case .staged: return NFKModelResidencyStaged
        case .paged: return NFKModelResidencyPaged
        case .streamed: return NFKModelResidencyStreamed
        }
    }

    /// The legacy plan, or its refusal's sentence.
    private func legacy(_ stages: [NFKMLXStageFootprint], _ residency: NFKMLXResidency,
                        _ budget: Int) -> Result<NFKMLXLegacyPlanner.Plan, NFKMLXLegacyPlanner.Refusal> {
        do {
            return .success(try NFKMLXLegacyPlanner.plan(stages, residency: residency, budget: budget))
        } catch let refusal as NFKMLXLegacyPlanner.Refusal {
            return .failure(refusal)
        } catch {
            return .failure(NFKMLXLegacyPlanner.Refusal(sentence: "unexpected \(error)"))
        }
    }

    private func refusalSentence(of error: Error) -> String {
        if case NFKMLXError.unsupportedConfiguration(let sentence) = error {
            return sentence
        }
        return (error as NSError).localizedDescription
    }

    private func assertPlan(_ core: NFKModelResidencyPlan, equals plan: NFKMLXLegacyPlanner.Plan,
                            stages: [NFKMLXStageFootprint], _ context: String) {
        XCTAssertEqual(core.holdsStagesResident, plan.holdsStagesResident, context)
        XCTAssertEqual(core.pagesExperts, plan.pagesExperts, context)
        XCTAssertEqual(core.expertCacheByteCount, plan.expertCacheBytes, context)
        XCTAssertEqual(Set(core.widenedStageIndexes.map { $0 }), plan.widenedStages, context)
        var streamed = [Int: Int]()
        for (index, room) in core.streamedStageByteCounts {
            streamed[index.intValue] = room.intValue
        }
        XCTAssertEqual(streamed, plan.streamedStages, context)
        let coreStages = coreFootprints(stages)
        for index in stages.indices {
            let range = core.heldLayerRangeOfStage(at: UInt(index), footprint: coreStages[index])
            XCTAssertEqual(range.location ..< range.location + range.length,
                           plan.heldLayers(index, of: stages[index]), "\(context), stage \(index)")
        }
    }

    // The core planner gives the legacy planner's plan, or its refusal's sentence, for every footprint,
    // residency, and budget in the grid.
    func testTheCorePlannerMatchesTheLegacyPlannerOverTheGrid() {
        var compared = 0
        for stages in Self.footprints {
            for residency in Self.residencies {
                for budget in Self.budgets {
                    let context = "\(stages.map(\.bytes)) \(residencyName(residency)) at \(budget)"
                    var core: NFKModelResidencyPlan?
                    var error: NSError?
                    do {
                        core = try NFKModelResidencyPlan(forStages: coreFootprints(stages),
                                                         residency: residencyName(residency), budget: budget)
                    } catch let refusal as NSError {
                        error = refusal
                    }
                    switch legacy(stages, residency, budget) {
                    case .success(let plan):
                        guard let core else {
                            XCTFail("\(context): the core refused \(error?.localizedDescription ?? "")")
                            continue
                        }
                        assertPlan(core, equals: plan, stages: stages, context)
                    case .failure(let refusal):
                        XCTAssertNil(core, context)
                        XCTAssertEqual(error?.localizedDescription, refusal.sentence, context)
                        XCTAssertEqual(error?.domain, NFKModelCatalogErrorDomain, context)
                        XCTAssertEqual(error?.code, NFKModelCatalogError.Code.doesNotFit.rawValue, context)
                    }
                    compared += 1
                }
            }
        }
        XCTAssertEqual(compared, Self.footprints.count * Self.residencies.count * Self.budgets.count)
    }

    // The Swift API, which now calls the core, gives the legacy planner's plan, or throws its sentence.
    func testTheCallThroughMatchesTheLegacyPlannerOverTheGrid() {
        for stages in Self.footprints {
            for residency in Self.residencies {
                for budget in Self.budgets {
                    let context = "\(stages.map(\.bytes)) \(residencyName(residency)) at \(budget)"
                    switch legacy(stages, residency, budget) {
                    case .success(let expected):
                        do {
                            let plan = try NFKMLXResidencyBudget.plan(stages, residency: residency, budget: budget)
                            XCTAssertEqual(plan.holdsStagesResident, expected.holdsStagesResident, context)
                            XCTAssertEqual(plan.pagesExperts, expected.pagesExperts, context)
                            XCTAssertEqual(plan.expertCacheBytes, expected.expertCacheBytes, context)
                            XCTAssertEqual(plan.widenedStages, expected.widenedStages, context)
                            XCTAssertEqual(plan.streamedStages, expected.streamedStages, context)
                            for index in stages.indices {
                                XCTAssertEqual(plan.widens(index), expected.widens(index), context)
                                XCTAssertEqual(plan.streams(index), expected.streams(index), context)
                                XCTAssertEqual(plan.heldLayers(index, of: stages[index]),
                                               expected.heldLayers(index, of: stages[index]), context)
                            }
                            assertVerification(stages, plan: plan, legacy: expected, budget: budget, context)
                        } catch {
                            XCTFail("\(context): the call-through refused \(refusalSentence(of: error))")
                        }
                    case .failure(let refusal):
                        XCTAssertThrowsError(try NFKMLXResidencyBudget.plan(stages, residency: residency,
                                                                            budget: budget), context) { error in
                            XCTAssertEqual(refusalSentence(of: error), refusal.sentence, context)
                        }
                    }
                }
            }
        }
    }

    /// Checks that each stage of a staged plan loads alone, as both planners check it, at every budget of the
    /// grid, with stage names and without.
    private func assertVerification(_ stages: [NFKMLXStageFootprint], plan: NFKMLXResidencyPlan,
                                    legacy: NFKMLXLegacyPlanner.Plan, budget: Int, _ context: String) {
        for names in [Self.names, []] {
            var legacySentence: String?
            do {
                try NFKMLXLegacyPlanner.verifyEachStageLoads(stages, plan: legacy, budget: budget, names: names)
            } catch let refusal as NFKMLXLegacyPlanner.Refusal {
                legacySentence = refusal.sentence
            } catch {
                legacySentence = "unexpected \(error)"
            }
            var sentence: String?
            do {
                try NFKMLXResidencyBudget.verifyEachStageLoads(stages, plan: plan, budget: budget, names: names)
            } catch {
                sentence = refusalSentence(of: error)
            }
            XCTAssertEqual(sentence, legacySentence, "\(context), verifying with \(names.count) names")
        }
    }

    // The budget arithmetic the planner rests on is the legacy arithmetic.
    func testTheBudgetArithmeticMatchesTheLegacyArithmetic() {
        XCTAssertEqual(NFKMLXResidencyBudget.reserve, NFKMLXLegacyPlanner.reserve)
        XCTAssertEqual(NFKMLXResidencyBudget.workingSetFraction, NFKMLXLegacyPlanner.workingSetFraction)
        XCTAssertEqual(NFKMLXResidencyBudget.defaultExpertCacheBytes, NFKMLXLegacyPlanner.defaultExpertCacheBytes)
        let byteCounts = [0, 1, Self.mib, Self.gib, 3 * Self.gib + 1, 17 * Self.gib, 18 * Self.gib,
                          27 * Self.gib + 123_457, 512 * Self.gib]
        for bytes in byteCounts {
            XCTAssertEqual(NFKMLXResidencyBudget.leastWorkingSet(holding: Int64(bytes)),
                           NFKMLXLegacyPlanner.leastWorkingSet(holding: Int64(bytes)), "\(bytes)")
            XCTAssertEqual(NFKMLXResidencyBudget.gib(bytes), NFKMLXLegacyPlanner.gib(bytes), "\(bytes)")
            for budget in Self.budgets {
                XCTAssertEqual(NFKMLXResidencyBudget.holds(bytes, budget: budget),
                               NFKMLXLegacyPlanner.holds(bytes, budget: budget), "\(bytes) at \(budget)")
                XCTAssertEqual(NFKMLXResidencyBudget.admits(bytes, budget: budget),
                               NFKMLXLegacyPlanner.admits(bytes, budget: budget), "\(bytes) at \(budget)")
            }
        }
    }
}
