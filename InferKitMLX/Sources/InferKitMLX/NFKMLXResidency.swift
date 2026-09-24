//
//  NFKMLXResidency.swift
//  InferKitMLX
//
//  How a model holds its weights against the machine's working set. Two mechanisms answer it, and
//  one plan chooses between them.
//
//  - Staging changes WHEN whole stages are loaded. A model built from stages that run one after
//    another (FLUX.1 and FLUX.2: text encoders, then a transformer; MiniMax Music 3: a language
//    model, then a flow-matching transformer, then a vocoder) never needs two stages at once, so a
//    release too large to hold whole still runs when its stages take turns.
//  - Paging changes WHERE part of one stage lives. A mixture's routed experts stay where the release
//    stores them (`NFKMLXExpertStore`) and are materialized as the router reaches them, so a stage
//    too large to hold whole still runs when only its routed experts are left behind.
//
//  Neither changes what any stage computes. Staging is tried first because it costs a load per run;
//  paging costs a read per step.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

/// How a staged model holds its stages.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXResidency) public enum NFKMLXResidency: Int, Sendable {
    /// Holds every stage loaded between runs where they fit the machine's working set together, and
    /// stages them where they do not.
    case automatic
    /// Holds every stage loaded between runs; a run fails where they do not fit together.
    case resident
    /// Loads each stage for its turn and releases it after, on every run. Slower, since each run reads
    /// every stage from disk, and the way a release fits that does not fit whole.
    case staged
    /// Stages as ``staged`` does, and leaves each stage's routed experts in the release, materializing
    /// the ones a step routes to through a bounded cache. The smallest footprint, and the slowest:
    /// every step reads the experts it routes to that the cache does not hold. A model without routed
    /// experts holds its stages as ``staged`` does.
    case paged
}

/// What one stage weighs: every byte it holds resident, how many of those are routed experts a paged
/// load leaves in the release, and what it would hold at a wider precision it prefers.
struct NFKMLXStageFootprint: Equatable {
    let bytes: Int
    let pageableBytes: Int
    /// The stage at a wider precision it takes where the plan can afford it, such as a text encoder
    /// stored at bfloat16 and run at float32; nil for a stage with one precision.
    let widenedBytes: Int?

    init(bytes: Int, pageableBytes: Int = 0, widenedBytes: Int? = nil) {
        self.bytes = bytes
        self.pageableBytes = pageableBytes
        self.widenedBytes = widenedBytes
    }

    /// What the stage holds with its routed experts paged.
    var unpagedBytes: Int { bytes - pageableBytes }
}

/// How a model holds its weights: whether its stages stay loaded between runs, and whether its routed
/// experts are paged.
struct NFKMLXResidencyPlan: Equatable {
    /// Whether the stages stay loaded between runs.
    let holdsStagesResident: Bool
    /// Whether the routed experts stay in the release, materialized as they are routed to.
    let pagesExperts: Bool
    /// The bytes a paged load's cache may hold in materialized experts; zero where nothing is paged.
    let expertCacheBytes: Int
    /// The positions of the stages that load at their wider precision.
    var widenedStages: Set<Int> = []

    static let resident = NFKMLXResidencyPlan(holdsStagesResident: true, pagesExperts: false,
                                              expertCacheBytes: 0)
    static let staged = NFKMLXResidencyPlan(holdsStagesResident: false, pagesExperts: false,
                                            expertCacheBytes: 0)

    /// Whether the stage at `index` loads at its wider precision.
    func widens(_ index: Int) -> Bool { widenedStages.contains(index) }
}

/// The working set a staged model plans against, and the rule it plans by.
///
/// @discussion The budget is 0.85 of the working set Metal recommends, not live free memory: a resident
/// model's own weights would count against free memory on the next check and evict themselves. Every
/// placement keeps a reserve for activations. A budget of 0 is a machine that reports none, and the two
/// questions asked of it get different answers. Whether to HOLD stages resident, or take the larger of
/// two precisions, is answered no, because staging always runs and holding everything may not; that is
/// Music 3's rule, "an unreadable budget stages rather than gambling". Whether a single stage may LOAD at
/// all is answered yes, as `NFKMLXReleaseWeights.verifyFits` answers it, because refusing would leave
/// such a machine unable to run anything.
enum NFKMLXResidencyBudget {
    /// The reserve a placement keeps beside its weights: activations, and Music 3's key-value cache for
    /// its guidance pair at the full position budget (about 3 GB at bfloat16).
    static let reserve = 4 << 30

    /// 0.85 of the working set Metal recommends for this machine, or 0 where it reports none.
    static func current() -> Int {
        let recommended = NFKHardwareProfile.current.recommendedWorkingSetSize
        return recommended > 0 ? Int(Double(recommended) * 0.85) : 0
    }

    /// Whether `bytes` of weights are known to fit `budget` beside the reserve: the test for holding
    /// stages resident or choosing a larger precision. False on a machine that reports no budget.
    static func holds(_ bytes: Int, budget: Int) -> Bool {
        budget > 0 && bytes + reserve <= budget
    }

    /// Whether `bytes` of weights may load at all: the test for refusing a stage outright. True on a
    /// machine that reports no budget.
    static func admits(_ bytes: Int, budget: Int) -> Bool {
        budget <= 0 || bytes + reserve <= budget
    }

    /// Whether stages totalling `bytes` stay resident under `residency`: `.automatic` where they are
    /// known to fit, `.staged` and `.paged` never, and `.resident` always, throwing where they are
    /// known NOT to fit rather than loading until the process is killed.
    static func holdsResident(_ bytes: Int, residency: NFKMLXResidency, budget: Int) throws -> Bool {
        try plan([NFKMLXStageFootprint(bytes: bytes)], residency: residency, budget: budget)
            .holdsStagesResident
    }

    /// What a paged load's expert cache holds where the machine reports no budget.
    static let defaultExpertCacheBytes = 4 << 30

    /// How a model of `stages` is held under `residency`.
    ///
    /// @discussion Each residency decides as it says:
    /// - `.resident` → every stage held, no paging; throws where the stages are known not to fit.
    /// - `.staged` → stages take turns, no paging.
    /// - `.paged` → stages take turns and routed experts are paged; throws where a stage is known not
    ///   to fit even with its experts paged.
    /// - `.automatic` → the first of those three that is known to fit. Every stage held where the
    ///   total fits; paged where the largest stage is known not to fit whole and has experts to page;
    ///   staged otherwise. A machine that reports no budget is staged and never paged, because paging
    ///   is chosen only against a known shortfall.
    ///
    /// A paged load's cache takes half of what the largest stage leaves of the budget once its
    /// unpaged weights and the reserve are counted, and never more than that stage's experts. The
    /// other half is left to the operating system's page cache, which holds the release's recently
    /// read experts. The split is a policy choice that no measurement here has tuned.
    ///
    /// A stage with a wider precision takes it where that is known to fit, and residency is decided
    /// first, at each stage's own precision:
    /// - held resident → stages widen in order while the total, widened, still fits.
    /// - staged or paged → a stage widens where it fits alone, widened.
    /// - a machine that reports no budget → no stage widens.
    static func plan(_ stages: [NFKMLXStageFootprint], residency: NFKMLXResidency,
                     budget: Int) throws -> NFKMLXResidencyPlan {
        let total = stages.reduce(0) { $0 + $1.bytes }
        let largest = stages.max { $0.bytes < $1.bytes } ?? NFKMLXStageFootprint(bytes: 0)
        let largestUnpaged = stages.map(\.unpagedBytes).max() ?? 0
        let pageable = stages.contains { $0.pageableBytes > 0 }
        func paged() throws -> NFKMLXResidencyPlan {
            guard admits(largestUnpaged, budget: budget) else {
                throw NFKMLXError.unsupportedConfiguration(
                    "the largest stage needs \(gib(largestUnpaged)) with its routed experts paged, plus "
                    + "a \(gib(reserve)) reserve, against a \(gib(budget)) working set")
            }
            let largestPageable = stages.map(\.pageableBytes).max() ?? 0
            let headroom = budget > 0 ? (budget - reserve - largestUnpaged) / 2 : defaultExpertCacheBytes
            return NFKMLXResidencyPlan(holdsStagesResident: false, pagesExperts: true,
                                       expertCacheBytes: max(0, min(largestPageable, headroom)))
        }
        let placement: NFKMLXResidencyPlan
        switch residency {
        case .resident:
            guard admits(total, budget: budget) else {
                throw NFKMLXError.unsupportedConfiguration(
                    "the stages need \(gib(total)) together plus a \(gib(reserve)) reserve, against a "
                    + "\(gib(budget)) working set; hold them staged"
                    + (pageable ? " or paged" : ""))
            }
            placement = .resident
        case .staged:
            placement = .staged
        case .paged:
            placement = pageable ? try paged() : .staged
        case .automatic:
            if holds(total, budget: budget) {
                placement = .resident
            } else if pageable, budget > 0, !holds(largest.bytes, budget: budget) {
                placement = try paged()
            } else {
                placement = .staged
            }
        }
        var widened = placement
        widened.widenedStages = widenedStages(stages, resident: placement.holdsStagesResident, budget: budget)
        return widened
    }

    /// The stages that take their wider precision under a placement.
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

    /// Throws where a staged placement has a stage known not to load on its own, at the precision
    /// `plan` chose for it: the refusal a single release's `verifyFits` gives, made before any stage
    /// loads. A resident placement was checked whole when it was planned.
    static func verifyEachStageLoads(_ stages: [NFKMLXStageFootprint], plan: NFKMLXResidencyPlan,
                                     budget: Int, names: [String]) throws {
        guard !plan.holdsStagesResident else { return }
        for (index, stage) in stages.enumerated() {
            let bytes = plan.widens(index) ? stage.widenedBytes ?? stage.bytes
                : (plan.pagesExperts ? stage.unpagedBytes : stage.bytes)
            guard admits(bytes, budget: budget) else {
                let name = index < names.count ? names[index] : "stage \(index)"
                throw NFKMLXError.unsupportedConfiguration(
                    "\(name) alone needs \(gib(bytes)) plus a \(gib(reserve)) reserve, against a "
                    + "\(gib(budget)) working set")
            }
        }
    }

    static func gib(_ bytes: Int) -> String {
        String(format: "%.1f GiB", Double(bytes) / 1_073_741_824)
    }
}

/// One stage of a staged model: loaded on use, held where the model is resident.
final class NFKMLXStage<Value> {
    private var held: Value?
    private let load: () throws -> Value

    init(_ load: @escaping () throws -> Value) {
        self.load = load
    }

    var isHeld: Bool { held != nil }

    func release() {
        held = nil
    }

    /// The loaded value, held for later runs where `keep` says so.
    fileprivate func acquire(keep: Bool) throws -> Value {
        if let held {
            return held
        }
        let value = try load()
        if keep {
            held = value
        }
        return value
    }
}

/// The stages of one staged model and the lock that serializes their use.
///
/// @discussion Runs serialize on the lock: a stage is gigabytes of GPU-resident weights, and two
/// concurrent runs would each load their own. A staged use loads the stage, runs the body, drops the
/// stage, and clears MLX's buffer cache before the next stage loads. What the body returns has to be
/// EVALUATED before it returns: an unevaluated result is a graph over the stage's weights and keeps them
/// alive, which is why the `MLXArray` forms, ``with(_:_:)``, evaluate for the caller.
final class NFKMLXStagedModel: @unchecked Sendable {
    private let lock = NSRecursiveLock()
    private var isResident: Bool

    init(resident: Bool) {
        isResident = resident
    }

    /// Whether the stages are held between runs.
    var resident: Bool { exclusively { isResident } }

    /// Holds the stages for the whole of `body`, so a second caller cannot load a stage beside one this
    /// caller holds.
    func exclusively<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    /// Changes whether the stages are held, releasing `stages` when they no longer are. A model that can
    /// only plan once its weights are present (Music 3 builds before its release is downloaded) plans at
    /// the start of each run.
    func setResident(_ resident: Bool, releasing stages: [() -> Void]) {
        exclusively {
            if isResident && !resident {
                stages.forEach { $0() }
                NFKMLXGPU.clearCache()
            }
            isResident = resident
        }
    }

    /// Runs `body` with `stage` loaded. A staged model releases the stage and clears MLX's buffer cache
    /// after `body`; the caller has evaluated what `body` returns (the `with` forms do it for arrays).
    func use<Value, T>(_ stage: NFKMLXStage<Value>, _ body: (Value) throws -> T) throws -> T {
        try exclusively {
            let keep = isResident
            let result = try { try body(stage.acquire(keep: keep)) }()
            if !keep {
                NFKMLXGPU.clearCache()
            }
            return result
        }
    }

    /// ``use(_:_:)`` for a stage whose result is an array, evaluated before the stage is released.
    func with<Value>(_ stage: NFKMLXStage<Value>, _ body: (Value) throws -> MLXArray) throws -> MLXArray {
        try use(stage) { (value: Value) -> MLXArray in
            let result = try body(value)
            eval(result)
            return result
        }
    }

    /// ``use(_:_:)`` for a stage whose result is several arrays, evaluated before the stage is released.
    func with<Value>(_ stage: NFKMLXStage<Value>, _ body: (Value) throws -> [MLXArray]) throws -> [MLXArray] {
        try use(stage) { (value: Value) -> [MLXArray] in
            let result = try body(value)
            eval(result)
            return result
        }
    }
}
