//
//  NFKMLXDeepSeekPaging.swift
//  InferKitMLX
//
//  Holding the parts of a DeepSeek release that a step barely touches in the form the release
//  stores them, and decoding only what a step reads: the routed experts an expert at a time, and
//  the n-gram tables a row at a time.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// What a load holds in the form the release stores it rather than decoded.
///
/// @discussion A released V4.1 Flash decodes to 1423.0 GiB of bf16 parameters (2843.2 GiB in
/// float32), and two groups are almost all of it: the routed experts and the two n-gram tables. Both
/// are barely touched by any one step — a token reaches `activatedExpertCount` of 384 experts, and
/// `engramHashColumns` rows of 384 million — so both are worth holding stored and decoding on
/// demand. Nothing else is: the attention weights and the shared expert are read by every token,
/// and holding those stored would decode them as often as a resident load reads them.
///
/// Each mechanism costs time. Paging the experts decodes an expert per routed expert per chunk;
/// paging the tables gathers and decodes a handful of rows per position. Neither changes what the
/// decoder computes: a paged decoder produces the same values as a resident one, not close ones.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXDeepSeekPaging: Sendable, Equatable {

    /// Holds the routed experts 4-bit and decodes one as the router reaches it.
    public var routedExperts: Bool

    /// Holds the n-gram tables fp8 and decodes a row as it is looked up, which is what the
    /// release's per-row scales are for.
    public var ngramTables: Bool

    /// Whether the n-gram tables are MAPPED from the release rather than read into memory.
    ///
    /// @discussion Holding a table stored takes it from four bytes a channel to one; mapping takes
    /// it to nothing at all, because the pages a lookup touches are the only ones resident and the
    /// operating system reclaims the rest. A lookup reads a handful of rows of `engramHeadDimensions`
    /// bytes, which is the access pattern mapping suits best, and the row-gather already decodes per
    /// lookup so nothing is given up. It costs a read from the release for what is not in the page
    /// cache, so the release has to stay where it was loaded from.
    ///
    public var mapsNgramTables: Bool

    /// Whether the routed experts are MAPPED from the release rather than read into memory.
    ///
    /// @discussion The mechanism is the n-gram tables', and the trade is not. A table row is
    /// `engramHeadDimensions` bytes and a lookup reads a handful; an expert is three matrices of
    /// megabytes and a token reaches `activatedExpertCount` of them in every layer, so a step with
    /// a cold cache reads far more from the release than a table lookup ever does. Whether that is
    /// usable depends on the machine's storage and on how skewed the routing turns out to be,
    /// neither of which this package has measured. A caller that maps the experts should RAISE
    /// `expertCacheBytes`, because the cache is now standing in front of a read rather than a
    /// decode.
    public var mapsRoutedExperts: Bool

    /// What a paged load keeps in decoded experts. Zero decodes every routed expert on every chunk
    /// that reaches it, which is the least memory and the most work.
    public var expertCacheBytes: Int

    public init(routedExperts: Bool = false, ngramTables: Bool = false,
                mapsNgramTables: Bool = false, mapsRoutedExperts: Bool = false,
                expertCacheBytes: Int = NFKMLXDeepSeekPaging.defaultExpertCacheBytes) {
        self.routedExperts = routedExperts
        self.ngramTables = ngramTables
        self.mapsNgramTables = mapsNgramTables
        self.mapsRoutedExperts = mapsRoutedExperts
        self.expertCacheBytes = expertCacheBytes
    }

    /// Everything resident, which is what a machine large enough for the release wants.
    public static let none = NFKMLXDeepSeekPaging()

    /// Every group this loader can hold stored, read into memory.
    public static let all = NFKMLXDeepSeekPaging(routedExperts: true, ngramTables: true)

    /// Every group held stored, with the n-gram tables mapped rather than read into memory.
    public static let mapped = NFKMLXDeepSeekPaging(routedExperts: true, ngramTables: true,
                                                    mapsNgramTables: true)

    /// Every group mapped: the smallest a load gets here, and the most it reads per step.
    public static let fullyMapped = NFKMLXDeepSeekPaging(routedExperts: true, ngramTables: true,
                                                         mapsNgramTables: true,
                                                         mapsRoutedExperts: true)

    /// Whether anything at all is held stored.
    public var isEmpty: Bool { !routedExperts && !ngramTables }

    /// What a paged load keeps in decoded experts where a caller names no figure.
    ///
    /// @discussion Routing is skewed, so a cache turns a hot expert into one decode rather than one
    /// per chunk that reaches it. The figure is a policy choice: no machine here holds the release,
    /// so the hit rate it buys is unmeasured, and a caller with a machine that runs the model sets
    /// its own.
    public static let defaultExpertCacheBytes = 4 << 30
}

/// How a DeepSeek release is loaded, where Objective-C reaches every choice.
///
/// @discussion Each property is a construction-time choice, because each changes what the decoder
/// holds: the paging preset decides which groups stay in the release, speculation loads the draft
/// stack's parameters, and the two numeric modes decide what the modules hold and compute in.
/// Every default reproduces a load computing in bf16, the release's own dtype, held as
/// ``NFKMLXResidency/automatic`` decides.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXDeepSeekLoadOptions)
public final class NFKMLXDeepSeekLoadOptions: NSObject {
    /// How the release is held where ``paging`` names no preset: ``NFKMLXResidency/automatic``
    /// loads resident where the decoded weights fit and pages otherwise. Introduced in InferKit 0.4.0.
    @objc public var residency: NFKMLXResidency = .automatic
    /// Which groups stay in the release, chosen explicitly. Any preset but `none` overrides
    /// ``residency``.
    @objc public var paging: NFKMLXDeepSeekPagingMode = .none
    /// What a load with an explicit ``paging`` preset keeps in decoded experts; a residency plan sizes
    /// its own cache.
    @objc public var expertCacheBytes: Int = NFKMLXDeepSeekPaging.defaultExpertCacheBytes
    /// Loads the release's draft stack, which a request then turns on with
    /// `NFKMLXGenerationParameterKey.draftTokens`.
    @objc public var speculates = false
    /// Rounds activations where the release's own inference code rounds them.
    @objc public var quantizesActivations = false
    /// Computes in float32 in place of the bf16 the release declares, at twice the bytes a step
    /// reads.
    @objc public var computesInFloat32 = false

    @objc public override init() { super.init() }
}

/// The paging presets, as Objective-C reaches them.
///
/// @discussion ``NFKMLXDeepSeekPaging`` is a Swift struct, which does not bridge, so each of its
/// presets crosses as a case here. A Swift caller uses the struct and can combine the groups freely.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXDeepSeekPagingMode)
public enum NFKMLXDeepSeekPagingMode: Int, Sendable {
    /// Every parameter decoded to float.
    case none
    /// The routed experts and the n-gram tables held in memory as the release stores them.
    case all
    /// As `all`, with the n-gram tables left in the release rather than read in.
    case mapped
    /// Every paged group left in the release: the least memory and the most read per step.
    case fullyMapped

    /// The Swift policy this preset names.
    public var policy: NFKMLXDeepSeekPaging {
        switch self {
        case .none: return .none
        case .all: return .all
        case .mapped: return .mapped
        case .fullyMapped: return .fullyMapped
        }
    }
}

/// The n-gram table in the form the release stores it, decoding a row as it is looked up.
///
/// @discussion This is the one group the reference itself never holds decoded. Its scales are one
/// `e8m0` per row per 32 channels, which exists so that a row can be decoded on its own; a square
/// blocking would have forced 32 rows to be decoded together. A lookup gathers the stored bytes for
/// the rows it wants and decodes those, so the table costs a byte a channel instead of four and the
/// decode is proportional to what is read rather than to what is held.
final class NFKDeepSeekStoredTable {

    /// A table's bytes mapped out of the release, as the byte ranges its two tensors occupy.
    struct Mapping {
        let file: NFKMLXMappedFile
        /// Where row 0 of the values starts, and how wide a row is.
        let values: Int
        let valueStride: Int
        /// The same for the scales, absent where the table is stored as floats.
        let scales: Int?
        let scaleStride: Int
    }

    /// `[rows, dimensions]` held in memory, fp8 bytes or the floats themselves.
    private let stored: MLXArray?
    /// `[rows, dimensions / blockSize]`, `e8m0`, held in memory.
    private let scale: MLXArray?
    /// Or the same two tensors, left in the release and copied out a row at a time.
    private let mapping: Mapping?
    let rowCount: Int
    let dimensions: Int
    let blockSize: Int

    init(stored: MLXArray, scale: MLXArray?, dimensions: Int, blockSize: Int) {
        self.stored = stored
        self.scale = scale
        mapping = nil
        rowCount = stored.dim(0)
        self.dimensions = dimensions
        self.blockSize = blockSize
    }

    init(mapping: Mapping, rowCount: Int, dimensions: Int, blockSize: Int) {
        stored = nil
        scale = nil
        self.mapping = mapping
        self.rowCount = rowCount
        self.dimensions = dimensions
        self.blockSize = blockSize
    }

    /// What this table holds in memory. A mapped table holds nothing: the pages a lookup touches
    /// are resident and the operating system reclaims them.
    var storedBytes: Int { (stored?.nbytes ?? 0) + (scale?.nbytes ?? 0) }

    /// The rows `indices` names, decoded. `indices` has any shape; the result gains a trailing
    /// `dimensions` axis, which is what `Embedding` does with the same argument.
    func callAsFunction(_ indices: MLXArray) -> MLXArray {
        let flat = indices.reshaped([-1])
        let gathered = mapping.map { gather(flat, through: $0) }
            ?? (values: take(stored!, flat, axis: 0), scales: scale.map { take($0, flat, axis: 0) })
        guard let scales = gathered.scales else {
            return gathered.values.reshaped(indices.shape + [dimensions])
        }
        let decoded = NFKMLXDeepSeekQuantization.dequantizeFP8(bytes: gathered.values,
                                                              scaleBytes: scales,
                                                              blockSize: blockSize)
        return decoded.reshaped(indices.shape + [dimensions])
    }

    /// Copies the rows a lookup names out of the mapping.
    ///
    /// @discussion The copy is what keeps the mapping out of MLX. A gather run as an MLX operation
    /// would take the whole table as its source, and a source of 98 GB would be made resident to
    /// run it; copying first means MLX only ever sees the few kilobytes that were read. A row is
    /// `dimensions` bytes and a lookup reads a handful of them.
    private func gather(_ indices: MLXArray, through mapping: Mapping)
        -> (values: MLXArray, scales: MLXArray?) {
        let rows = indices.asArray(Int32.self).map { Int($0) }
        var values = [UInt8](repeating: 0, count: rows.count * mapping.valueStride)
        var scales = mapping.scales == nil
            ? [] : [UInt8](repeating: 0, count: rows.count * mapping.scaleStride)
        values.withUnsafeMutableBytes { valueBytes in
            for (slot, row) in rows.enumerated() {
                // A hash may name a row outside the table; the reference clamps rather than reads
                // past the end, and a mapping must not be asked to.
                let clamped = Swift.min(Swift.max(row, 0), Swift.max(rowCount - 1, 0))
                mapping.file.copy(to: valueBytes.baseAddress! + slot * mapping.valueStride,
                                  offset: mapping.values + clamped * mapping.valueStride,
                                  count: mapping.valueStride)
            }
        }
        if let scaleBase = mapping.scales {
            scales.withUnsafeMutableBytes { scaleBytes in
                for (slot, row) in rows.enumerated() {
                    let clamped = Swift.min(Swift.max(row, 0), Swift.max(rowCount - 1, 0))
                    mapping.file.copy(to: scaleBytes.baseAddress! + slot * mapping.scaleStride,
                                      offset: scaleBase + clamped * mapping.scaleStride,
                                      count: mapping.scaleStride)
                }
            }
        }
        let valueArray = MLXArray(values).reshaped([rows.count, mapping.valueStride])
        guard mapping.scales != nil else { return (valueArray, nil) }
        return (valueArray, MLXArray(scales).reshaped([rows.count, mapping.scaleStride]))
    }
}

/// One matrix of a routed expert, as the release stores it.
struct NFKDeepSeekStoredMatrix {
    /// A matrix left in the release, as the byte ranges its two tensors occupy.
    struct Mapping {
        let file: NFKMLXMappedFile
        let values: Int
        let valueShape: [Int]
        let scales: Int?
        let scaleShape: [Int]
    }

    /// The stored tensor: fp8 bytes, 4-bit pairs packed two to a byte, or the floats themselves
    /// where the checkpoint carries no companion scale. Absent where the matrix is mapped.
    let stored: MLXArray?
    /// The `e8m0` block scales, absent where the matrix is stored as floats or is mapped.
    let scale: MLXArray?
    /// Or the same two tensors, left in the release and copied out when the expert is decoded.
    let mapping: Mapping?
    /// The float shape the module holds it at, which is what separates 4-bit from fp8.
    let shape: [Int]

    init(stored: MLXArray, scale: MLXArray?, shape: [Int]) {
        self.stored = stored
        self.scale = scale
        mapping = nil
        self.shape = shape
    }

    init(mapping: Mapping, shape: [Int]) {
        stored = nil
        scale = nil
        self.mapping = mapping
        self.shape = shape
    }

    /// What holding this matrix in stored form costs. A mapped matrix holds nothing.
    var storedBytes: Int { (stored?.nbytes ?? 0) + (scale?.nbytes ?? 0) }

    /// A 4-bit weight is packed two values to a byte, so its stored last axis is half its float one.
    /// That is a property of the pair rather than of the release: a matrix stored fp8 keeps its
    /// full width.
    func decoded(fp8BlockSize: Int) -> MLXArray {
        // Copied out of the mapping first, so what follows is the same arithmetic on the same bytes
        // whether they were held in memory or left in the release.
        let values: MLXArray
        let scales: MLXArray?
        if let mapping {
            values = mapping.file.bytes(at: mapping.values, shape: mapping.valueShape)
            scales = mapping.scales.map { mapping.file.bytes(at: $0, shape: mapping.scaleShape) }
        } else {
            values = stored!
            scales = scale
        }
        guard let scales else { return values }
        let packedFourBit = shape.count == 2 && values.dim(values.ndim - 1) == shape[1] / 2
        return packedFourBit
            ? NFKMLXDeepSeekQuantization.dequantizeFP4(packedBytes: values, scaleBytes: scales)
            : NFKMLXDeepSeekQuantization.dequantizeFP8(bytes: values, scaleBytes: scales,
                                                       blockSize: fp8BlockSize)
    }
}

/// One routed-expert matrix as the store reads it: the release's stored form, decoded to the
/// decoder's compute type on materialization.
struct NFKDeepSeekExpertMatrix: NFKMLXExpertSource {
    let matrix: NFKDeepSeekStoredMatrix
    let fp8BlockSize: Int
    let computeType: DType

    var heldBytes: Int { matrix.storedBytes }

    var mappedBytes: Int {
        guard let mapping = matrix.mapping else { return 0 }
        let values = mapping.valueShape.reduce(1, *)
        return values + (mapping.scales == nil ? 0 : mapping.scaleShape.reduce(1, *))
    }

    func materialize() -> MLXArray {
        matrix.decoded(fp8BlockSize: fp8BlockSize).asType(computeType)
    }
}

/// The routed experts of a DeepSeek release, held as the release stores them and decoded on demand.
///
/// @discussion A released V4.1 Flash carries 40 layers of 384 routed experts. As floats they are
/// 2.17 TB, which no machine holds; in the 4-bit form the release ships they are about an eighth of
/// that. A token reaches `activatedExpertCount` of the 384, so the great majority of what a resident
/// load would hold is never read on any one step.
///
/// The experts live in an ``NFKMLXExpertStore``, the store every paged mixture here reads, filed one
/// group per layer with the matrices `w1`, `w2` and `w3` as its parts. This type adds what is DeepSeek's
/// own: decoding the release's fp8 and 4-bit forms, and building the clamped SwiGLU expert module a
/// routed token runs through. A bounded cache keeps the most recently decoded experts, because routing
/// is skewed and a hot expert would otherwise decode once per chunk that reaches it.
///
/// Decoding is not free: an expert is three matrices, and a 40-layer step that activates six experts
/// a layer decodes 240 of them. This trades time for a model that fits, which is the only trade
/// available on a machine smaller than the decoded weights.
///
/// The store is Swift-only because it is built from the release's stored tensors, which are
/// `MLXArray`s. An Objective-C caller reaches a paged model through
/// `deepSeekPagedBackendWithDirectoryURL:expertCacheBytes:error:`, which carries the one policy
/// choice a caller makes.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXDeepSeekExpertStore {

    /// The store the decoded experts are cached in and the stored ones filed in.
    public let experts: NFKMLXExpertStore

    private let fp8BlockSize: Int
    private let swigluLimit: Float
    private let servedBlock: Int?
    private let computeType: DType

    /// The bytes the cache may hold in decoded experts.
    ///
    /// @discussion Lowering it evicts down to the new bound immediately. Zero decodes every routed
    /// expert on every chunk that reaches it, which is the least memory and the most work.
    public var cacheByteBudget: Int {
        get { experts.cacheByteBudget }
        set { experts.cacheByteBudget = newValue }
    }

    /// How many experts have been decoded from their stored bytes.
    public var decodeCount: Int { experts.materializeCount }

    /// How many routing requests the cache answered without a decode.
    public var cacheHitCount: Int { experts.cacheHitCount }

    /// What the stored weights occupy, which is what a paged load adds to the machine's working set.
    public var storedBytes: Int { experts.heldBytes }

    /// What the decoded experts currently held occupy.
    public var cachedBytes: Int { experts.cachedBytes }

    /// How many routed experts the store holds, across every layer.
    public var expertCount: Int { experts.expertCount }

    init(configuration: NFKMLXDeepSeekConfiguration, cacheByteBudget: Int) {
        experts = NFKMLXExpertStore(cacheByteBudget: cacheByteBudget)
        fp8BlockSize = configuration.fp8BlockSize
        swigluLimit = configuration.swigluLimit
        servedBlock = configuration.quantizesActivations ? configuration.fp8BlockSize : nil
        computeType = configuration.computeType
    }

    private static func group(_ layer: Int) -> String { "layers.\(layer)" }

    /// Takes one stored matrix of one routed expert.
    ///
    /// - Parameter matrix: `w1`, `w2` or `w3`, which is how the release names an expert's three
    ///   matrices and the part the decode looks them up by.
    func store(_ value: NFKDeepSeekStoredMatrix, layer: Int, expert: Int, matrix: String) {
        experts.store(NFKDeepSeekExpertMatrix(matrix: value, fp8BlockSize: fp8BlockSize, computeType: computeType),
                      group: Self.group(layer), expert: expert, part: matrix)
    }

    /// Whether every expert the configuration declares arrived with all three of its matrices.
    ///
    /// @discussion A paged load drops the expert keys from the module, so the coverage check that
    /// catches a missing parameter on a resident load cannot see them. This is that check.
    func verifyComplete(_ c: NFKMLXDeepSeekConfiguration) throws {
        let missing = (0 ..< c.layerCount).flatMap { layer in
            experts.missing(group: Self.group(layer), experts: c.routedExpertCount, parts: ["w1", "w2", "w3"])
                .map { "layers.\(layer).ffn.experts.\($0.expert).\($0.part).weight" }
        }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch(
                "\(missing.count) routed-expert matrices are absent from the paged store, starting "
                + "with \(missing.sorted()[0])")
        }
    }

    /// The expert at `index` of `layer`, decoded from its stored bytes or taken from the cache.
    func expert(layer: Int, index: Int) -> NFKDeepSeekExpert? {
        guard let matrices = experts.expert(group: Self.group(layer), index: index),
              let gate = matrices["w1"], let down = matrices["w2"], let up = matrices["w3"] else { return nil }
        return NFKDeepSeekExpert(gate: gate, down: down, up: up, limit: swigluLimit, servedBlock: servedBlock)
    }

    /// Drops every decoded expert, keeping the stored bytes.
    public func clearCache() {
        experts.clearCache()
    }
}

/// What a paged mixture of experts reads: the store, and which layer's experts it wants from it.
final class NFKDeepSeekExpertPager {
    let store: NFKMLXDeepSeekExpertStore
    let layer: Int

    init(store: NFKMLXDeepSeekExpertStore, layer: Int) {
        self.store = store
        self.layer = layer
    }
}
