//
//  NFKMLXExpertPaging.swift
//  InferKitMLX
//
//  The loader half of paging: finding a release's routed-expert tensors from its shard headers,
//  weighing them, and filing them in an `NFKMLXExpertStore` as byte ranges of the mapped release.
//  A family supplies one thing, the classifier that says which of its tensors are experts and how
//  a slice of one becomes what its module computes with.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX

/// A release tensor a paged load leaves in the release, and the expert layer it feeds.
struct NFKMLXExpertSlice {
    /// The module path of the expert-holding layer, which is the store's group.
    let group: String
    /// Which of the layer's arrays the tensor is: `weight`, `scales`, or `biases`.
    let part: String
    /// The expert the tensor holds, or nil where it stacks every expert along its first axis.
    let expert: Int?
    /// What one expert's slice needs to become the module's layout. It runs after the conversion to
    /// the load precision.
    var transform: (MLXArray) -> MLXArray = { $0 }
    /// Whether the slice is stored `[in, out]`, the transpose of the module's `[out, in]`, and is
    /// multiplied as stored. A resident load transposes such a stack and multiplies by the transpose
    /// of that, which reads the stored layout again; a paged load that rebuilt `[out, in]` would run
    /// the multiply over a different memory layout and accumulate in a different order.
    var inputMajor = false
}

/// A release's tensors split into the experts a paged load leaves behind and everything it loads.
///
/// @discussion The inventory reads shard headers and nothing else, so weighing a release for a plan
/// costs kilobytes whatever the release's size.
struct NFKMLXExpertInventory {
    struct Tensor {
        let file: URL
        let entry: NFKMLXSafetensorsEntry
        let slices: [NFKMLXExpertSlice]
    }

    /// The release tensors that feed expert layers.
    let experts: [String: Tensor]
    /// What the release holds resident at the load precision.
    let totalBytes: Int
    /// What its expert tensors hold resident at the load precision.
    let pageableBytes: Int

    /// The release's weight at the load precision, and how much of it a paged load leaves behind.
    var footprint: NFKMLXStageFootprint {
        NFKMLXStageFootprint(bytes: totalBytes, pageableBytes: pageableBytes)
    }

    /// Whether `key`, a release tensor name, stays in the release on a paged load.
    func isExpert(_ key: String) -> Bool { experts[key] != nil }

    /// Reads every shard header under `directory` and classifies each tensor.
    ///
    /// - Parameter classify: the slices a release tensor feeds, or an empty array for a tensor the
    ///   load keeps. One tensor may feed several layers, as a fused gate-and-up projection does.
    init(inDirectory directory: URL, precision: NFKMLXWeightPrecision,
         classify: (String, NFKMLXSafetensorsEntry) -> [NFKMLXExpertSlice]) throws {
        var experts = [String: Tensor]()
        var total = 0
        var pageable = 0
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            for (key, entry) in try NFKMLXSafetensors.entries(inFile: url) {
                let resident = Self.residentBytes(entry, precision: precision)
                total += resident
                let slices = classify(key, entry)
                guard !slices.isEmpty else { continue }
                experts[key] = Tensor(file: url, entry: entry, slices: slices)
                pageable += resident
            }
        }
        self.experts = experts
        totalBytes = total
        pageableBytes = pageable
    }

    /// The inventory of a release a load is about to plan from.
    ///
    /// @discussion A release whose headers cannot be read is first held to the working set by its file
    /// sizes, so a release larger than the machine is refused for that, the refusal a resident load
    /// has always given, before the unreadable header is reported.
    static func planning(inDirectory directory: URL, precision: NFKMLXWeightPrecision,
                         classify: (String, NFKMLXSafetensorsEntry) -> [NFKMLXExpertSlice]) throws
        -> NFKMLXExpertInventory {
        do {
            return try NFKMLXExpertInventory(inDirectory: directory, precision: precision, classify: classify)
        } catch {
            try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: precision)
            throw error
        }
    }

    /// What a tensor occupies loaded at `precision`: a half-precision float doubles at float32, and
    /// every other type loads as stored.
    static func residentBytes(_ entry: NFKMLXSafetensorsEntry, precision: NFKMLXWeightPrecision) -> Int {
        let widens = precision == .float32 && (entry.dtype == "F16" || entry.dtype == "BF16")
        return widens ? entry.byteCount * 2 : entry.byteCount
    }

    /// Files every expert slice in `store` as a byte range of its mapped shard.
    ///
    /// @discussion Each shard is mapped once and shared by every slice read from it; the mapping
    /// lives as long as the store does. A half-precision float converts to float32 on
    /// materialization where `precision` says so, which is the conversion the resident loader
    /// applies to the whole tensor.
    func register(into store: NFKMLXExpertStore, precision: NFKMLXWeightPrecision) throws {
        var mappings = [URL: NFKMLXMappedFile]()
        for (key, tensor) in experts {
            let file: NFKMLXMappedFile
            if let mapped = mappings[tensor.file] {
                file = mapped
            } else {
                file = try NFKMLXMappedFile(url: tensor.file)
                mappings[tensor.file] = file
            }
            let widens = precision == .float32 && (tensor.entry.dtype == "F16" || tensor.entry.dtype == "BF16")
            for slice in tensor.slices {
                let transform: (MLXArray) -> MLXArray = widens
                    ? { slice.transform($0.asType(.float32)) } : slice.transform
                if let expert = slice.expert {
                    guard let source = NFKMLXExpertTensor.mapped(tensor.entry, in: file, transform: transform) else {
                        throw Self.unreadable(key, tensor.entry)
                    }
                    store.store(source, group: slice.group, expert: expert, part: slice.part)
                    continue
                }
                for expert in 0 ..< (tensor.entry.shape.first ?? 0) {
                    guard let source = NFKMLXExpertTensor.mapped(tensor.entry, expert: expert, in: file,
                                                                 transform: transform) else {
                        throw Self.unreadable(key, tensor.entry)
                    }
                    store.store(source, group: slice.group, expert: expert, part: slice.part)
                }
            }
        }
    }

    /// Throws naming the first expert part `store` lacks, for layers whose groups, expert counts and
    /// parts `layers` lists.
    ///
    /// @discussion A paged layer holds no parameters, so the coverage check that catches a missing
    /// tensor on a resident load cannot see one missing here. This is that check.
    static func verifyComplete(_ store: NFKMLXExpertStore,
                               layers: [(group: String, experts: Int, parts: [String])]) throws {
        let missing = layers.flatMap { layer in
            store.missing(group: layer.group, experts: layer.experts, parts: layer.parts)
                .map { "\(layer.group).\($0.expert).\($0.part)" }
        }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch(
                "\(missing.count) expert tensors are absent from the paged release, starting with "
                + "\(missing.sorted()[0])")
        }
    }

    /// Loads a release held as `residency` plans it.
    ///
    /// @discussion The plan weighs the release from its shard headers. Where it pages, `install` puts
    /// paged layers reading from a new store in place of the resident ones and returns what each must
    /// find there, the experts are filed as byte ranges of the mapped release, and `load` loads every
    /// other tensor, skipping each key its argument names. Where it does not page, `load` loads
    /// everything and nothing else runs.
    ///
    /// - Returns: the store the paged layers read, or nil where nothing is paged.
    static func load(directory: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                     classify: (String, NFKMLXSafetensorsEntry) -> [NFKMLXExpertSlice],
                     install: (NFKMLXExpertStore) throws -> [(group: String, experts: Int, parts: [String])],
                     load: ((String) -> Bool) throws -> Void) throws -> NFKMLXExpertStore? {
        let inventory = try planning(inDirectory: directory, precision: precision, classify: classify)
        let plan = try NFKMLXResidencyBudget.plan([inventory.footprint], residency: residency,
                                                  budget: NFKMLXResidencyBudget.current())
        guard plan.pagesExperts else {
            try load { _ in false }
            return nil
        }
        let store = NFKMLXExpertStore(cacheByteBudget: plan.expertCacheBytes)
        let layers = try install(store)
        try inventory.register(into: store, precision: precision)
        try load(inventory.isExpert)
        try verifyComplete(store, layers: layers)
        return store
    }

    /// The classifier for a release that stores each expert projection stacked `[experts, out, in]`,
    /// which is the module's own layout: a tensor whose module key ends in one of `projections` pages
    /// whole, filed under that key.
    ///
    /// - Parameter moduleKey: the module path a release key loads into, or nil for a tensor the
    ///   module does not take.
    static func stacked(projections: [String], moduleKey: @escaping (String) -> String?)
        -> (String, NFKMLXSafetensorsEntry) -> [NFKMLXExpertSlice] {
        { key, entry in
            guard entry.shape.count == 3, let name = moduleKey(key),
                  projections.contains(where: { name.hasSuffix(".experts." + $0) }) else { return [] }
            return [NFKMLXExpertSlice(group: name, part: "weight", expert: nil)]
        }
    }

    private static func unreadable(_ key: String, _ entry: NFKMLXSafetensorsEntry) -> Error {
        NFKMLXError.unsupportedConfiguration(
            "\(key) is stored \(entry.dtype) \(entry.shape), which a paged load cannot slice by expert")
    }
}
