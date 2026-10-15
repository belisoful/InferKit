//
//  NFKMLXCatalogBuilding.swift
//  InferKitMLX
//
//  Building a catalog descriptor's backend: the MLX side of `NFKModelCatalog`'s build entry.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXModelDescriptors {

    /// Builds the backend of an MLX descriptor from a catalog build request.
    ///
    /// @discussion `NFKModelCatalog`'s `backendForModelIdentifier:options:error:` calls this for the
    /// package's descriptors, and for any descriptor of the MLX engine through the builder
    /// ``registerBackendBuilder()`` registers. The build:
    /// 1. names the model by the variant's registry name, else the descriptor's, else its identifier, and
    ///    registers every model (``NFKMLXReferenceModels/registerAll()``) once when the name is not
    ///    registered;
    /// 2. locates a draft model under `NFKModelLoadOptionDraftModel` through the request's catalog;
    /// 3. builds through the registry with the request's load options as ``NFKMLXLoadOptions``, from the
    ///    request's location, and sets the property options on the backend the factory returns, except
    ///    those the factory consumes, which it read as arguments.
    ///
    /// A backend built from a location unloads and builds again with the same options, setting the
    /// property options again. Its `loadOptionValues` reports the load options as requested, and its
    /// `modelInfo` the residency the load resolved to under `NFKModelInfoResidency`. A property option that names an MLX enumeration, such as
    /// `NFKMLXBackend.precision`, takes the enumeration's case for its `NFKModelPrecision*` or
    /// `NFKModelResidency*` value.
    ///
    /// - Throws: `NFKModelCatalogErrorNoBuilder` when the package registers no model under the name,
    ///   `NFKModelCatalogErrorInvalidOptions` when the backend has no property a property option names,
    ///   and what the factory throws.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendForBuildRequest:error:)
    public static func backend(for request: NFKModelBuildRequest) throws -> any NFKInferenceBackend {
        let name = request.variant?.registryName ?? request.descriptor.registryName ?? request.descriptor.identifier
        if !NFKMLXModelRegistry.isModelRegistered(name) {
            _ = registeredEveryModel
        }
        guard NFKMLXModelRegistry.isModelRegistered(name) else {
            throw NSError(domain: NFKModelCatalogErrorDomain, code: NFKModelCatalogError.Code.noBuilder.rawValue,
                          userInfo: [NSLocalizedDescriptionKey:
                                     "InferKitMLX registers no model named \"\(name)\" to build \(request.descriptor.identifier)."])
        }
        let draftURL = try NFKMLXLoadOptions.draftLocation(request.loadOptions[NFKModelLoadOptionDraftModel],
                                                            in: request.catalog)
        let options = NFKMLXLoadOptions(request.loadOptions, draftURL: draftURL)
        let consumed = NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: name)
        return try NFKMLXModelRegistry.backend(named: name, weightsURL: request.location, loadOptions: options) {
            try applyPropertyOptions(of: request, to: $0, skipping: consumed)
        }
    }

    /// Registers the builder of the MLX engine's descriptors with the model catalog, so a descriptor of the
    /// engine that another provider or a manifest adds builds as the package's own do.
    /// ``NFKMLXReferenceModels/registerAll()`` calls it. Introduced in InferKit 0.4.0.
    @objc public static func registerBackendBuilder() {
        NFKModelCatalog.registerBackendBuilder(NFKMLXEngineBuilder(), forEngine: NFKModelEngineMLX)
    }

    /// Every model registered, once per process.
    private static let registeredEveryModel: Void = NFKMLXReferenceModels.registerAll()

    /// Sets the request's property options on `backend`, except those under `consumed` keys, which the
    /// factory read as arguments. A value that names an MLX enumeration's case is converted where the
    /// property holds a number.
    static func applyPropertyOptions(of request: NFKModelBuildRequest, to backend: any NFKInferenceBackend,
                                     skipping consumed: Set<String> = []) throws {
        let applied = request.propertyOptions.filter { !consumed.contains($0.key) }
        guard !applied.isEmpty else {
            return
        }
        let converted = convertedPropertyOptions(applied, of: request.descriptor, for: backend)
        guard converted != nil || applied.count != request.propertyOptions.count else {
            try request.applyPropertyOptions(toBackend: backend)
            return
        }
        var loadOptions = request.loadOptions.filter { !consumed.contains($0.key) }
        loadOptions.merge(converted ?? [:]) { _, new in new }
        let request = NFKModelBuildRequest(descriptor: request.descriptor, variant: request.variant,
                                           loadOptions: loadOptions, requestDefaults: request.requestDefaults,
                                           location: request.location, memoryEstimate: request.memoryEstimate,
                                           catalog: request.catalog)
        try request.applyPropertyOptions(toBackend: backend)
    }

    /// The property options whose string value names an MLX enumeration's case and whose property holds a
    /// number, as those numbers; nil when none needs converting.
    private static func convertedPropertyOptions(_ propertyOptions: [String: Any], of descriptor: NFKModelDescriptor,
                                                 for backend: any NFKInferenceBackend) -> [String: Any]? {
        guard let object = backend as? NSObject else {
            return nil
        }
        var converted = [String: Any]()
        for (key, value) in propertyOptions {
            guard let name = value as? String,
                  let keyPath = descriptor.option(withKey: key)?.propertyKeyPath,
                  !keyPath.contains("."), object.responds(to: NSSelectorFromString(keyPath)),
                  object.value(forKey: keyPath) is NSNumber,
                  let number = enumerationValue(named: name) else {
                continue
            }
            converted[key] = number
        }
        return converted.isEmpty ? nil : converted
    }

    /// The raw value of the MLX enumeration case an option value names: a precision or a residency.
    private static func enumerationValue(named name: String) -> NSNumber? {
        if let precision = NFKMLXLoadOptions.weightPrecision(named: name) {
            return NSNumber(value: precision.rawValue)
        }
        if let residency = NFKMLXResidency(optionValue: name) {
            return NSNumber(value: residency.rawValue)
        }
        return nil
    }
}

/// The model catalog's builder for the MLX engine, which builds as ``NFKMLXModelDescriptors`` does.
final class NFKMLXEngineBuilder: NSObject, NFKModelBackendBuilding {

    func backend(for request: NFKModelBuildRequest) throws -> any NFKInferenceBackend {
        try NFKMLXModelDescriptors.backend(for: request)
    }
}
