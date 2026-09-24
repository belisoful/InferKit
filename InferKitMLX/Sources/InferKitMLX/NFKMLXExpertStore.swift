//
//  NFKMLXExpertStore.swift
//  InferKitMLX
//
//  Holding a mixture's routed experts outside the module, where a release stores them, and
//  materializing only the experts a step routes to. Every mixture here reads its experts through
//  one of two seams: a stacked `[experts, out, in]` tensor dispatched through a gathered matrix
//  multiply, or one module per expert. The store serves both, so paging is a property of the load
//  and not of the model family.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX

/// One tensor of one expert, where a store reads it from.
///
/// @discussion A source is either held in memory or left in the release. Materializing one returns
/// the array the layer computes with, which may be a decode of what is stored.
protocol NFKMLXExpertSource {
    /// The bytes the source keeps in memory.
    var heldBytes: Int { get }
    /// The bytes the source leaves in a mapped release file.
    var mappedBytes: Int { get }
    /// The array the layer computes with. The result may be lazy; the store evaluates it.
    func materialize() -> MLXArray
}

/// An expert tensor read as stored, then passed through `transform`.
///
/// @discussion The transform carries what the release's layout needs to become the module's: a
/// transposition, a conversion to the load precision, a view of packed bytes as words. It runs on
/// one expert's slice, so the arithmetic is the resident loader's applied to less of the tensor.
struct NFKMLXExpertTensor: NFKMLXExpertSource {
    enum Storage {
        case held(MLXArray)
        case mapped(NFKMLXMappedFile, offset: Int)
    }

    let storage: Storage
    let dtype: DType
    let shape: [Int]
    var transform: (MLXArray) -> MLXArray = { $0 }

    var byteCount: Int { shape.reduce(dtype.size, *) }

    var heldBytes: Int {
        if case .held(let array) = storage { return array.nbytes }
        return 0
    }

    var mappedBytes: Int {
        if case .mapped = storage { return byteCount }
        return 0
    }

    func materialize() -> MLXArray {
        switch storage {
        case .held(let array):
            return transform(array)
        case .mapped(let file, let offset):
            let bytes = file.bytes(at: offset, shape: [byteCount])
            let typed = dtype == .uint8 ? bytes : bytes.view(dtype: dtype)
            return transform(typed.reshaped(shape))
        }
    }

    /// Expert `expert` of a tensor stored stacked `[experts, …]`, left in the release.
    ///
    /// @discussion A stacked tensor is row-major, so one expert's slice is one contiguous byte range
    /// and needs no gather.
    static func mapped(_ entry: NFKMLXSafetensorsEntry, expert: Int, in file: NFKMLXMappedFile,
                       transform: @escaping (MLXArray) -> MLXArray = { $0 }) -> NFKMLXExpertTensor? {
        guard let dtype = NFKMLXSafetensors.dtype(entry.dtype), entry.shape.count >= 2,
              (0 ..< entry.shape[0]).contains(expert) else { return nil }
        let shape = Array(entry.shape.dropFirst())
        let stride = shape.reduce(dtype.size, *)
        let offset = entry.start + expert * stride
        guard file.contains(offset: offset, count: stride) else { return nil }
        return NFKMLXExpertTensor(storage: .mapped(file, offset: offset), dtype: dtype, shape: shape,
                                  transform: transform)
    }

    /// A tensor the release stores whole for one expert, left in the release.
    static func mapped(_ entry: NFKMLXSafetensorsEntry, in file: NFKMLXMappedFile,
                       transform: @escaping (MLXArray) -> MLXArray = { $0 }) -> NFKMLXExpertTensor? {
        guard let dtype = NFKMLXSafetensors.dtype(entry.dtype),
              entry.byteCount == entry.shape.reduce(dtype.size, *),
              file.contains(offset: entry.start, count: entry.byteCount) else { return nil }
        return NFKMLXExpertTensor(storage: .mapped(file, offset: entry.start), dtype: dtype,
                                  shape: entry.shape, transform: transform)
    }
}

/// A mixture's routed experts, held outside the module and materialized as the router reaches them.
///
/// @discussion A token reaches a few of a mixture's experts, so on any one step most of what a
/// resident load holds is not read. The store keeps each expert where the load left it, in memory
/// or in the mapped release, and materializes an expert when a layer asks for it. A bounded cache
/// keeps the most recently used materialized experts, because routing is skewed and a hot expert
/// would otherwise be read again on every step that reaches it.
///
/// Experts are addressed by a group, which names one expert-holding layer (the module path of a
/// stacked projection, or a DeepSeek layer), and an index inside it. Each expert carries one or
/// more named parts: a weight, its quantization scales and biases, or a module's three matrices.
///
/// Paging changes where the weights live and nothing about what the layer computes. Each routed
/// token meets the same matrix it would meet resident, so a paged decoder produces the same values
/// as a resident one.
///
/// The store is Swift-only because its contents are `MLXArray`s. An Objective-C caller chooses
/// paging through ``NFKMLXResidency/paged`` on a factory that takes a residency.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXExpertStore: @unchecked Sendable {

    private struct Address: Hashable {
        let group: String
        let expert: Int
    }

    private let lock = NSLock()
    private var sources = [Address: [String: any NFKMLXExpertSource]]()
    private var cache = [Address: [String: MLXArray]]()
    private var lastUse = [Address: Int]()
    private var costs = [Address: Int]()
    private var clock = 0
    private var cachedBytesHeld = 0
    private var materialized = 0
    private var hits = 0
    private var held = 0
    private var mapped = 0
    private var budget: Int

    /// The bytes the cache may hold in materialized experts.
    ///
    /// @discussion Lowering it evicts down to the new bound immediately. Zero materializes every
    /// routed expert on every step that reaches it, which is the least memory and the most work.
    public var cacheByteBudget: Int {
        get { locked { budget } }
        set { locked { budget = newValue; evictDownToBudget() } }
    }

    /// How many experts have been materialized from their sources.
    public var materializeCount: Int { locked { materialized } }

    /// How many requests for an expert the cache answered without materializing it.
    public var cacheHitCount: Int { locked { hits } }

    /// The bytes the sources keep in memory, which is what the store adds to the working set
    /// before any expert is materialized.
    public var heldBytes: Int { locked { held } }

    /// The bytes the sources leave in the release.
    public var mappedBytes: Int { locked { mapped } }

    /// What the materialized experts currently cached occupy.
    public var cachedBytes: Int { locked { cachedBytesHeld } }

    /// How many experts the store holds, across every group.
    public var expertCount: Int { locked { sources.count } }

    /// A store whose cache holds up to `cacheByteBudget` bytes of materialized experts.
    public init(cacheByteBudget: Int) {
        budget = cacheByteBudget
    }

    /// Takes one part of one expert.
    func store(_ source: any NFKMLXExpertSource, group: String, expert: Int, part: String) {
        locked {
            let address = Address(group: group, expert: expert)
            if let replaced = sources[address]?[part] {
                held -= replaced.heldBytes
                mapped -= replaced.mappedBytes
            }
            sources[address, default: [:]][part] = source
            held += source.heldBytes
            mapped += source.mappedBytes
        }
    }

    /// Every part `parts` names, for experts `0 ..< experts` of `group`, that the store lacks.
    func missing(group: String, experts: Int, parts: [String]) -> [(expert: Int, part: String)] {
        locked {
            (0 ..< experts).flatMap { expert in
                parts.filter { sources[Address(group: group, expert: expert)]?[$0] == nil }
                    .map { (expert, $0) }
            }
        }
    }

    /// Expert `index` of `group`, every part materialized, or nil where the store does not hold it.
    func expert(group: String, index: Int) -> [String: MLXArray]? {
        locked { acquire(Address(group: group, expert: index)) }
    }

    /// The experts `experts` names, each part stacked `[experts.count, …]` in that order.
    ///
    /// @discussion The stack is what a gathered matrix multiply reads in place of the whole
    /// `[experts, …]` tensor, with the routing indices renumbered into it. It is built from the
    /// materialized experts, so it holds what was routed to and nothing else.
    func bank(group: String, experts: [Int]) -> [String: MLXArray] {
        locked {
            let members = experts.compactMap { acquire(Address(group: group, expert: $0)) }
            guard members.count == experts.count, let parts = members.first?.keys else { return [:] }
            var bank = [String: MLXArray]()
            for part in parts {
                bank[part] = stacked(members.compactMap { $0[part] }, axis: 0)
            }
            return bank
        }
    }

    /// Drops every materialized expert, keeping the sources.
    public func clearCache() {
        locked {
            cache.removeAll()
            lastUse.removeAll()
            costs.removeAll()
            cachedBytesHeld = 0
        }
    }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    /// The cached expert, or a fresh materialization cached where the budget has room. Runs on the
    /// lock.
    private func acquire(_ address: Address) -> [String: MLXArray]? {
        clock += 1
        if let cached = cache[address] {
            hits += 1
            lastUse[address] = clock
            return cached
        }
        guard let parts = sources[address] else { return nil }
        let arrays = parts.mapValues { $0.materialize() }
        // A materialization is a lazy graph over the source bytes, and an unevaluated one pins them
        // along with every intermediate of a decode. Evaluating here is what lets them go.
        eval(Array(arrays.values))
        materialized += 1
        let cost = arrays.values.reduce(0) { $0 + $1.nbytes }
        guard cost <= budget else { return arrays }
        cache[address] = arrays
        lastUse[address] = clock
        costs[address] = cost
        cachedBytesHeld += cost
        evictDownToBudget()
        return arrays
    }

    /// Evicts least-recently-used experts until the cache is within its budget. Runs on the lock.
    private func evictDownToBudget() {
        while cachedBytesHeld > budget, !cache.isEmpty {
            guard let oldest = cache.keys.min(by: { (lastUse[$0] ?? 0) < (lastUse[$1] ?? 0) }) else { break }
            cachedBytesHeld -= costs.removeValue(forKey: oldest) ?? 0
            cache.removeValue(forKey: oldest)
            lastUse.removeValue(forKey: oldest)
        }
        if cache.isEmpty {
            cachedBytesHeld = 0
        }
    }
}

/// What a paged expert layer reads: the store, and the group its experts are filed under.
struct NFKMLXExpertPager {
    let store: NFKMLXExpertStore
    let group: String

    /// The experts `indices` routes to, each part stacked, and `indices` renumbered into that stack.
    ///
    /// @discussion Reading the indices is a synchronization point: which experts to materialize is
    /// known only once the router has run.
    func bank(for indices: MLXArray) -> (parts: [String: MLXArray], indices: MLXArray) {
        let flat = indices.asType(.int32).asArray(Int32.self)
        let routed = Array(Set(flat)).sorted()
        var position = [Int32: UInt32]()
        for (slot, expert) in routed.enumerated() {
            position[expert] = UInt32(slot)
        }
        let renumbered = MLXArray(flat.map { position[$0] ?? 0 }).reshaped(indices.shape)
        return (store.bank(group: group, experts: routed.map(Int.init)), renumbered)
    }

    /// `x @ W[e]ᵀ` for each routed expert `e`, where the group's `weight` part is `[out, in]` per
    /// expert: the gathered matrix multiply a resident `[experts, out, in]` stack runs, over the
    /// routed experts alone.
    ///
    /// @discussion The result is evaluated before it returns. An unevaluated result is a graph over
    /// the bank, and a forward pass that stays lazy across layers would otherwise hold every layer's
    /// bank at once, which is the resident model's footprint again.
    ///
    /// - Parameter inputMajor: whether the group's weights are stored `[in, out]` and multiplied as
    ///   stored, which is the memory layout a resident load of such a release multiplies.
    func gatherMM(_ x: MLXArray, experts: MLXArray, inputMajor: Bool = false) -> MLXArray {
        let (bank, local) = bank(for: experts)
        let weight = bank["weight"]!
        let result = MLX.gatherMM(x, inputMajor ? weight : weight.swappedAxes(-1, -2), rhsIndices: local)
        eval(result)
        return result
    }

    /// ``gatherMM(_:experts:)`` for a group stored quantized: `weight` packed, with `scales` and,
    /// for an affine packing, `biases`.
    func gatherQuantizedMM(_ x: MLXArray, experts: MLXArray,
                           quantization: NFKMLXWeights.Quantization) -> MLXArray {
        let (bank, local) = bank(for: experts)
        let result = MLX.gatherQuantizedMM(x, bank["weight"]!, scales: bank["scales"]!,
                                           biases: bank["biases"], rhsIndices: local, transpose: true,
                                           groupSize: quantization.groupSize, bits: quantization.bits,
                                           mode: quantization.mode)
        eval(result)
        return result
    }
}

extension NFKMLXSafetensors {
    /// The MLX element type a safetensors dtype name reads as, or nil for one MLX has no type for.
    /// The fp8 forms read as bytes, which is what their decoders take.
    static func dtype(_ name: String) -> DType? {
        switch name {
        case "F32": return .float32
        case "F16": return .float16
        case "BF16": return .bfloat16
        case "I64": return .int64
        case "I32": return .int32
        case "I16": return .int16
        case "I8": return .int8
        case "U64": return .uint64
        case "U32": return .uint32
        case "U16": return .uint16
        case "U8", "F8_E4M3", "F8_E5M2", "F8_E8M0": return .uint8
        case "BOOL": return .bool
        default: return nil
        }
    }
}
