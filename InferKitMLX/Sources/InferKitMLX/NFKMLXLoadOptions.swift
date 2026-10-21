//
//  NFKMLXLoadOptions.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The load options a registered factory builds with: the resolved values of a descriptor's load
/// options, read as the MLX types the package's factories take.
///
/// @discussion The values are keyed by the core's `NFKModelLoadOption*` keys and hold the values the
/// descriptor resolved, defaults filled: residency an `NFKModelResidency*` string, precision an
/// `NFKModelPrecision*` string, the quantization, cache, and draft bits numbers, and the draft model a
/// descriptor identifier. Each reading is nil, or the package's default, when its key is absent, so a
/// factory keeps its own default for an option its descriptor does not declare.
///
/// - `residency` → ``NFKMLXResidency``; absent, ``NFKMLXResidency/automatic``.
/// - `precision` → ``NFKMLXWeightPrecision``: `float32` is `.float32`; `checkpoint`, `bfloat16`, and
///   `float16` are `.checkpoint`, the release's stored half precision.
/// - `quantizationBits` → a runtime weight quantization at `quantizationGroupSize` (64 when absent); 0 is
///   none.
/// - `contextLength`, `cacheBits`, `cacheGroupSize` → ``generationOptions(base:)``; `cacheBits` 0 is an
///   unquantized cache.
/// - `draftModel` → ``draftURL``, the draft's files, which the build locates; `draftBits` 0 keeps the draft
///   at the model's precision.
///
/// A backend built with the options reports ``values`` as its `loadOptionValues`. A factory reports the
/// residency a load resolved to through ``reportResidency(_:)``, and the backend reports it in `modelInfo`
/// under `NFKModelInfoResidency` (``resolvedResidency``).
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXLoadOptions: @unchecked Sendable {

    /// The group size a runtime weight quantization uses when the options name none. Introduced in
    /// InferKit 0.4.0.
    public static let defaultQuantizationGroupSize = 64

    /// The group size a quantized key-value cache uses when the options name none. Introduced in
    /// InferKit 0.4.0.
    public static let defaultCacheGroupSize = 64

    /// The resolved load options, keyed by `NFKModelLoadOption*` keys. Introduced in InferKit 0.4.0.
    public let values: [String: Any]

    /// The local copy of the draft model's files, or nil when the options name no draft. Introduced in
    /// InferKit 0.4.0.
    public let draftURL: URL?

    private let report = NFKMLXLoadReport()

    /// Options over resolved values, with the draft model's files at `draftURL`. Introduced in
    /// InferKit 0.4.0.
    public init(_ values: [String: Any] = [:], draftURL: URL? = nil) {
        self.values = values
        self.draftURL = draftURL
    }

    // MARK: Readings

    /// The residency the options ask for; ``NFKMLXResidency/automatic`` when they name none or one the
    /// package does not know. Introduced in InferKit 0.4.0.
    public var residency: NFKMLXResidency {
        string(NFKModelLoadOptionResidency).flatMap(NFKMLXResidency.init(optionValue:)) ?? .automatic
    }

    /// The `NFKModelPrecision*` value the options name, or nil. Introduced in InferKit 0.4.0.
    public var precisionName: String? {
        string(NFKModelLoadOptionPrecision)
    }

    /// The precision the weights load at, or nil when the options name none, so the factory keeps its own
    /// default. Introduced in InferKit 0.4.0.
    public var precision: NFKMLXWeightPrecision? {
        precisionName.flatMap(Self.weightPrecision(named:))
    }

    /// Whether the options ask for a half precision: true for `checkpoint`, `bfloat16`, and `float16`,
    /// false for `float32`, nil when they name none. A factory that takes a `half:` flag reads this.
    /// Introduced in InferKit 0.4.0.
    public var halfPrecision: Bool? {
        precision.map { $0 == .checkpoint }
    }

    /// The type a factory converts a float32 release to where it offers `bfloat16` beside the published
    /// precision: nil for no precision or `checkpoint`, `.bfloat16` for `bfloat16` where `offersBFloat16`.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` naming `model` for any other precision, or for
    ///   `bfloat16` where the model has no bfloat16 parity check.
    func bfloat16Cast(of model: String, offersBFloat16: Bool = true) throws -> DType? {
        try cast(of: model, offering: offersBFloat16 ? [NFKModelPrecisionBFloat16: .bfloat16] : [:])
    }

    /// The type a factory converts a release to for a precision it offers: nil for no precision,
    /// `checkpoint`, or a precision in `reading`, which the factory reads through ``precision``, and the
    /// offered type for a precision `offered` names.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` naming `model` for any other precision, which the
    ///   model has no parity check for.
    func cast(of model: String, offering offered: [String: DType], reading: Set<String> = []) throws -> DType? {
        guard let name = precisionName, name != NFKModelPrecisionCheckpoint, !reading.contains(name) else {
            return nil
        }
        if let dtype = offered[name] {
            return dtype
        }
        let loads = reading.union(offered.keys).sorted().map { " or at \($0)" }.joined()
        throw NFKMLXError.unsupportedConfiguration(
            "\(model) loads as published\(loads); precision \(name) has no parity check for it")
    }

    /// The bits a runtime weight quantization packs each weight into, or nil for none (absent or 0).
    /// Introduced in InferKit 0.4.0.
    public var quantizationBits: Int? {
        positiveInteger(NFKModelLoadOptionQuantizationBits)
    }

    /// The group size of a runtime weight quantization: the options' value, else
    /// ``defaultQuantizationGroupSize``. Introduced in InferKit 0.4.0.
    public var quantizationGroupSize: Int {
        positiveInteger(NFKModelLoadOptionQuantizationGroupSize) ?? Self.defaultQuantizationGroupSize
    }

    /// The default context length a request may override, or nil. Introduced in InferKit 0.4.0.
    public var contextLength: Int? {
        positiveInteger(NFKModelLoadOptionContextLength)
    }

    /// The bits of a quantized key-value cache, or nil for an unquantized one (absent or 0). Introduced in
    /// InferKit 0.4.0.
    public var cacheBits: Int? {
        positiveInteger(NFKModelLoadOptionCacheBits)
    }

    /// The group size of a quantized key-value cache: the options' value, else ``defaultCacheGroupSize``.
    /// Introduced in InferKit 0.4.0.
    public var cacheGroupSize: Int {
        positiveInteger(NFKModelLoadOptionCacheGroupSize) ?? Self.defaultCacheGroupSize
    }

    /// The identifier of the descriptor that drafts tokens, or nil when the options name none (absent,
    /// empty, or `none`). Introduced in InferKit 0.4.0.
    public var draftModel: String? {
        guard let name = string(NFKModelLoadOptionDraftModel), !name.isEmpty, name != "none" else {
            return nil
        }
        return name
    }

    /// The bits the draft's weights are quantized to, or nil to keep it at the model's precision (absent or
    /// 0). Introduced in InferKit 0.4.0.
    public var draftBits: Int? {
        positiveInteger(NFKModelLoadOptionDraftBits)
    }

    /// The draft's bits as ``draftBits`` reads them, or `defaultBits` when the options leave the key out: a
    /// factory whose draft quantizes by default keeps that default. Introduced in InferKit 0.4.0.
    public func draftBits(orDefault defaultBits: Int?) -> Int? {
        values[NFKModelLoadOptionDraftBits] == nil ? defaultBits : draftBits
    }

    /// `base` with the options' load-stage defaults: ``contextLength`` as its context window, and
    /// ``cacheBits`` at ``cacheGroupSize`` as its cache quantization. A request's own parameters still
    /// override them. Introduced in InferKit 0.4.0.
    public func generationOptions(base: NFKMLXGenerationOptions = NFKMLXGenerationOptions()) -> NFKMLXGenerationOptions {
        var options = base
        if let contextLength {
            options.contextWindow = contextLength
        }
        if let cacheBits {
            options.cacheQuantization = .init(bits: cacheBits, groupSize: cacheGroupSize)
        } else if values[NFKModelLoadOptionCacheBits] != nil {
            options.cacheQuantization = nil
        }
        return options
    }

    // MARK: Applying

    /// Quantizes `module`'s eligible `Linear` layers and stacked experts to ``quantizationBits`` at
    /// ``quantizationGroupSize``, then evaluates it and returns the replaced weights to the system. Does
    /// nothing when the options ask for no quantization. Embeddings stay as loaded, since a tied model's
    /// embedding is also its output projection. Introduced in InferKit 0.4.0.
    public func applyWeightQuantization(to module: Module) throws {
        guard let bits = quantizationBits else {
            return
        }
        try NFKMLXQuantization.quantize(module: module, bits: bits, groupSize: quantizationGroupSize)
        // The quantized arrays are lazy over the weights they replace; evaluated here, those weights are
        // freed and cleared from MLX's buffer cache instead of sitting beside the quantized ones.
        eval(module)
        NFKMLXGPU.clearCache()
    }

    // MARK: Reporting

    /// Records the residency the load resolved to, such as `.paged` for an automatic load that paged its
    /// experts. Introduced in InferKit 0.4.0.
    public func reportResidency(_ residency: NFKMLXResidency) {
        report.set(residency)
    }

    /// The residency the load resolved to, as the factory reported it or the built backend shows it, or
    /// nil when neither says. A backend built with these options reports it in `modelInfo` under
    /// `NFKModelInfoResidency`; its `loadOptionValues` keep ``values``, the residency as requested.
    /// Introduced in InferKit 0.4.0.
    public var resolvedResidency: NFKMLXResidency? {
        report.residency
    }

    /// Reads the residency a built backend shows, over what its factory reported: paged experts
    /// (`pagesExperts`), streamed layers (`streamedLayerCount`), or held stages (`holdsStagesResident`). A
    /// backend that shows none keeps the factory's report.
    func noteResidency(of backend: any NFKInferenceBackend) {
        guard let object = backend as? NSObject else {
            return
        }
        if object.responds(to: NSSelectorFromString("pagesExperts")),
           let paged = object.value(forKey: "pagesExperts") as? Bool {
            report.set(paged ? .paged : .resident)
        }
        if object.responds(to: NSSelectorFromString("streamedLayerCount")),
           let layers = object.value(forKey: "streamedLayerCount") as? Int, layers > 0 {
            report.set(.streamed)
        }
        if object.responds(to: NSSelectorFromString("holdsStagesResident")),
           let resident = object.value(forKey: "holdsStagesResident") as? Bool {
            report.set(resident ? .resident : .staged)
        }
    }

    // MARK: Values

    /// The precision an `NFKModelPrecision*` value names, or nil for one the package does not load.
    /// Introduced in InferKit 0.4.0.
    public static func weightPrecision(named name: String) -> NFKMLXWeightPrecision? {
        switch name {
        case NFKModelPrecisionFloat32:
            return .float32
        case NFKModelPrecisionCheckpoint, NFKModelPrecisionBFloat16, NFKModelPrecisionFloat16:
            return .checkpoint
        default:
            return nil
        }
    }

    /// The local copy of a draft model's files: a file URL or an absolute path as given, or the files of the
    /// descriptor a catalog has under the identifier, found by the catalog's locator. Nil when `value` names
    /// no draft (absent, empty, or `none`).
    ///
    /// - Throws: `NFKModelCatalogErrorUnknownModel` when the catalog has no such descriptor, and
    ///   `NFKModelCatalogErrorInvalidOptions` when its files are not present.
    static func draftLocation(_ value: Any?, in catalog: NFKModelCatalog) throws -> URL? {
        if let url = value as? URL {
            return url
        }
        guard let name = value as? String, !name.isEmpty, name != "none" else {
            return nil
        }
        if name.hasPrefix("/") {
            return URL(fileURLWithPath: name)
        }
        guard let descriptor = catalog.descriptor(withIdentifier: name) else {
            throw NSError(domain: NFKModelCatalogErrorDomain, code: NFKModelCatalogError.Code.unknownModel.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "The catalog has no draft model \"\(name)\"."])
        }
        var error: NSError?
        if let location = catalog.locator(descriptor, nil, &error) {
            return location
        }
        throw error ?? NSError(domain: NFKModelCatalogErrorDomain,
                               code: NFKModelCatalogError.Code.invalidOptions.rawValue,
                               userInfo: [NSLocalizedDescriptionKey:
                                          "The draft model \"\(name)\" has no local copy of its files."])
    }

    private func string(_ key: String) -> String? {
        values[key] as? String
    }

    private func positiveInteger(_ key: String) -> Int? {
        let value: Int?
        switch values[key] {
        case let number as NSNumber:
            value = number.intValue
        case let text as String:
            value = Int(text)
        default:
            value = nil
        }
        guard let value, value > 0 else {
            return nil
        }
        return value
    }
}

/// The residency one build reported, shared by every copy of the options it was built with.
private final class NFKMLXLoadReport: @unchecked Sendable {
    private let lock = NSLock()
    private var reported: NFKMLXResidency?

    var residency: NFKMLXResidency? { lock.withLock { reported } }

    func set(_ residency: NFKMLXResidency) {
        lock.withLock { reported = residency }
    }
}
