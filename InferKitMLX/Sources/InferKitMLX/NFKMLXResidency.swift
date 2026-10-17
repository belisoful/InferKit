//
//  NFKMLXResidency.swift
//  InferKitMLX
//
//  How a model holds its weights against the machine's working set. Three mechanisms answer it, and
//  one plan chooses between them.
//
//  - Staging changes WHEN whole stages are loaded. A model built from stages that run one after
//    another (FLUX.1 and FLUX.2: text encoders, then a transformer; MiniMax Music 3: a language
//    model, then a flow-matching transformer, then a vocoder) never needs two stages at once, so a
//    release too large to hold whole still runs when its stages take turns.
//  - Paging changes WHERE part of one stage lives. A mixture's routed experts stay where the release
//    stores them (`NFKMLXExpertStore`) and are materialized as the router reaches them, so a stage
//    too large to hold whole still runs when only its routed experts are left behind.
//  - Streaming changes WHEN one stage's layers are loaded. A dense stage too large to hold whole has
//    no experts to leave behind, so it holds the repeated layers that fit and reads each of the others
//    from the release in its turn (`NFKMLXLayerStream`), on every pass.
//
//  None changes what any stage computes. Staging is tried first because it costs a load per run;
//  paging costs a read per step, and streaming a read of every layer it does not hold per pass.
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
    /// Stages as ``staged`` does, and leaves a stage's repeated layers in the release where the stage
    /// does not fit whole: it holds the layers that fit and reads each of the others in its turn, on
    /// every pass. The way a dense model larger than the working set runs, and slower than paging,
    /// since every pass reads every layer it does not hold. A model without streamable layers holds
    /// its stages as ``staged`` does.
    case streamed
}

extension NFKMLXResidency {

    /// The residency an `NFKModelResidency*` load-option value names, or nil for any other value.
    /// Introduced in InferKit 0.4.0.
    public init?(optionValue: String) {
        switch optionValue {
        case NFKModelResidencyAutomatic: self = .automatic
        case NFKModelResidencyResident: self = .resident
        case NFKModelResidencyStaged: self = .staged
        case NFKModelResidencyPaged: self = .paged
        case NFKModelResidencyStreamed: self = .streamed
        default: return nil
        }
    }

    /// The `NFKModelResidency*` load-option value that names the residency. Introduced in InferKit 0.4.0.
    public var optionValue: String {
        switch self {
        case .automatic: return NFKModelResidencyAutomatic
        case .resident: return NFKModelResidencyResident
        case .staged: return NFKModelResidencyStaged
        case .paged: return NFKModelResidencyPaged
        case .streamed: return NFKModelResidencyStreamed
        }
    }
}

/// What one stage weighs: every byte it holds resident, how many of those are routed experts a paged
/// load leaves in the release, what it would hold at a wider precision it prefers, and the repeated
/// layers a streamed load may leave in the release.
struct NFKMLXStageFootprint: Equatable {
    let bytes: Int
    let pageableBytes: Int
    /// The stage at a wider precision it takes where the plan can afford it, such as a text encoder
    /// stored at bfloat16 and run at float32; nil for a stage with one precision.
    let widenedBytes: Int?
    /// Each repeated layer's bytes, in the order a streamed load holds them, counted inside ``bytes``;
    /// empty for a stage that cannot stream. For one stack that is the order a pass runs them; an
    /// encoder-decoder lists the decoder's first, since it runs on every token.
    let layerBytes: [Int]

    init(bytes: Int, pageableBytes: Int = 0, widenedBytes: Int? = nil, layerBytes: [Int] = []) {
        self.bytes = bytes
        self.pageableBytes = pageableBytes
        self.widenedBytes = widenedBytes
        self.layerBytes = layerBytes
    }

    /// The footprint as the core planner reads it.
    var core: NFKModelStageFootprint {
        NFKModelStageFootprint(byteCount: bytes, pageableByteCount: pageableBytes,
                               widenedByteCount: widenedBytes.map(NSNumber.init(value:)),
                               layerByteCounts: layerBytes.map(NSNumber.init(value:)))
    }

    /// What the stage holds with its routed experts paged.
    var unpagedBytes: Int { bytes - pageableBytes }

    /// What the stage holds outside its repeated layers.
    var unstreamableBytes: Int { bytes - layerBytes.reduce(0, +) }

    /// What a streamed load of the stage needs at the least: everything outside the repeated layers,
    /// and the two largest layers, the one computing and the one being read.
    var streamedMinimumBytes: Int {
        unstreamableBytes + layerBytes.sorted(by: >).prefix(2).reduce(0, +)
    }
}

/// How a model holds its weights: whether its stages stay loaded between runs, and whether its routed
/// experts are paged. It reads the core planner's `NFKModelResidencyPlan`.
struct NFKMLXResidencyPlan: Equatable {
    /// The core planner's plan.
    let core: NFKModelResidencyPlan

    /// Whether the stages stay loaded between runs.
    var holdsStagesResident: Bool { core.holdsStagesResident }
    /// Whether the routed experts stay in the release, materialized as they are routed to.
    var pagesExperts: Bool { core.pagesExperts }
    /// The bytes a paged load's cache may hold in materialized experts; zero where nothing is paged.
    var expertCacheBytes: Int { core.expertCacheByteCount }
    /// The positions of the stages that load at their wider precision.
    var widenedStages: Set<Int> { Set(core.widenedStageIndexes.map { $0 }) }
    /// The stages that stream their repeated layers, each with the bytes it may hold in layers beyond
    /// the two a pass has in flight.
    var streamedStages: [Int: Int] {
        Dictionary(uniqueKeysWithValues: core.streamedStageByteCounts.map { ($0.key.intValue, $0.value.intValue) })
    }

    static let resident = emptyPlan(.resident)
    static let staged = emptyPlan(.staged)

    /// The plan of no stages, which every residency places without a refusal.
    private static func emptyPlan(_ residency: NFKMLXResidency) -> NFKMLXResidencyPlan {
        guard let core = try? NFKModelResidencyPlan(forStages: [], residency: residency.optionValue, budget: 0) else {
            preconditionFailure("the planner refused a model of no stages")
        }
        return NFKMLXResidencyPlan(core: core)
    }

    init(core: NFKModelResidencyPlan) {
        self.core = core
    }

    static func == (lhs: NFKMLXResidencyPlan, rhs: NFKMLXResidencyPlan) -> Bool {
        lhs.core.isEqual(rhs.core)
    }

    /// Whether the stage at `index` loads at its wider precision.
    func widens(_ index: Int) -> Bool { core.widensStage(at: UInt(index)) }

    /// Whether the stage at `index` streams its repeated layers.
    func streams(_ index: Int) -> Bool { core.streamsStage(at: UInt(index)) }

    /// The layers of the stage at `index` a load holds: every one where the stage does not stream, and
    /// otherwise the first ones, in order, while their bytes fit what the plan gave the stage.
    func heldLayers(_ index: Int, of stage: NFKMLXStageFootprint) -> Range<Int> {
        let range = core.heldLayerRangeOfStage(at: UInt(index), footprint: stage.core)
        return range.location ..< range.location + range.length
    }
}

/// The working set a staged model plans against, and the rule it plans by.
///
/// @discussion The planner is the core's `NFKModelResidencyPlan` and its budget `NFKModelMemoryBudget`;
/// these are the Swift spellings the package's loaders use, and each refusal arrives as
/// `NFKMLXError.unsupportedConfiguration` carrying the core planner's sentence. The budget is 0.85 of the
/// working set Metal recommends, not live free memory: a resident model's own weights would count
/// against free memory on the next check and evict themselves. Every placement keeps a reserve for
/// activations: 4 GiB from a budget of 8 GiB, and half the budget below it. A budget of 0 is a machine
/// that reports none, and the two questions asked of it get different answers. Whether to HOLD stages
/// resident, or take the larger of two precisions, is answered no, because staging always runs and
/// holding everything may not; that is Music 3's rule, "an unreadable budget stages rather than
/// gambling". Whether a single stage may LOAD at all is answered yes, as `NFKMLXReleaseWeights.verifyFits`
/// answers it, because refusing would leave such a machine unable to run anything.
enum NFKMLXResidencyBudget {
    /// The most a placement keeps beside its weights, the reserve of every budget of 8 GiB or more:
    /// activations, and Music 3's key-value cache for its guidance pair at the full position budget
    /// (about 3 GB at bfloat16).
    static let maximumReserve = NFKModelMemoryBudget.maximumReserveByteCount

    /// The reserve a placement keeps beside its weights against `budget`: the core's
    /// `reserveByteCountForBudgetByteCount:`, half the budget below 8 GiB and ``maximumReserve`` from it.
    static func reserve(forBudget budget: Int) -> Int {
        NFKModelMemoryBudget.reserveByteCount(forBudgetByteCount: budget)
    }

    /// The share of Metal's recommended working set a placement plans against.
    static let workingSetFraction = NFKModelMemoryBudget.workingSetFraction

    /// The planned share of the working set Metal recommends for this machine, or 0 where it reports none.
    static func current() -> Int {
        NFKModelMemoryBudget.current().byteCount
    }

    /// The least budget that holds `bytes` of weights beside its own reserve: the core's
    /// `leastBudgetByteCountHoldingByteCount:`.
    static func leastBudget(holding bytes: Int) -> Int {
        NFKModelMemoryBudget.leastBudgetByteCountHoldingByteCount(bytes)
    }

    /// The least recommended working set whose budget holds `bytes` of weights beside that budget's
    /// reserve: the core's `leastWorkingSetHoldingByteCount:`. The catalog's `minimumMemoryBytes`.
    static func leastWorkingSet(holding bytes: Int64) -> Int64 {
        Int64(NFKModelMemoryBudget.leastWorkingSetHoldingByteCount(Int(bytes)))
    }

    /// Whether `bytes` of weights are known to fit `budget` beside its reserve: the test for holding
    /// stages resident or choosing a larger precision. False on a machine that reports no budget.
    static func holds(_ bytes: Int, budget: Int) -> Bool {
        budget > 0 && bytes + reserve(forBudget: budget) <= budget
    }

    /// Whether `bytes` of weights may load at all: the test for refusing a stage outright. True on a
    /// machine that reports no budget.
    static func admits(_ bytes: Int, budget: Int) -> Bool {
        budget <= 0 || bytes + reserve(forBudget: budget) <= budget
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

    /// How a model of `stages` is held under `residency`: `NFKModelResidencyPlan`'s
    /// `planForStages:residency:budget:error:`, whose discussion states each residency's rule, the paged
    /// cache's share, the streamed layers, and which stages widen.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` with the planner's sentence where the residency is
    ///   known not to fit.
    static func plan(_ stages: [NFKMLXStageFootprint], residency: NFKMLXResidency,
                     budget: Int) throws -> NFKMLXResidencyPlan {
        do {
            return NFKMLXResidencyPlan(core: try NFKModelResidencyPlan(
                forStages: stages.map(\.core), residency: residency.optionValue, budget: budget))
        } catch {
            throw refusal(error)
        }
    }

    /// Throws where a staged placement has a stage known not to load on its own, at the precision
    /// `plan` chose for it: the refusal a single release's `verifyFits` gives, made before any stage
    /// loads. A resident placement was checked whole when it was planned, and a streamed stage was
    /// checked streamed.
    static func verifyEachStageLoads(_ stages: [NFKMLXStageFootprint], plan: NFKMLXResidencyPlan,
                                     budget: Int, names: [String]) throws {
        do {
            try NFKModelResidencyPlan.verifyEachStageLoads(stages.map(\.core), plan: plan.core, budget: budget,
                                                           names: names)
        } catch {
            throw refusal(error)
        }
    }

    /// A byte count as the planner's sentences write it: "27.2 GiB".
    static func gib(_ bytes: Int) -> String {
        NFKModelMemoryBudget.formattedByteCount(bytes)
    }

    /// The core planner's refusal as the error the package's loaders throw.
    private static func refusal(_ error: Error) -> NFKMLXError {
        .unsupportedConfiguration((error as NSError).localizedDescription)
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
