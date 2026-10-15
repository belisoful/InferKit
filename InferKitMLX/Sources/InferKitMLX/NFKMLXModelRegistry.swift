//
//  NFKMLXModelRegistry.swift
//  InferKitMLX
//

import Foundation
import InferKit

/// A registry of MLX model backends by name, so an Objective-C consumer (an FCPX plugin, an app) can
/// build and run a local MLX model without writing Swift.
///
/// An MLX model's forward pass is Swift over `MLXArray`, which Objective-C cannot express — so the
/// bring-your-own-closure backends (`NFKMLXModuleBackend`, `NFKMLXMattingBackend`,
/// `NFKMLXTensorBackend`) are constructed from Swift. This registry bridges the gap: a model author
/// registers a factory once from Swift (capturing the forward), and the consumer builds the backend
/// by name and drives it through the Objective-C `NFKInferenceBackend` protocol.
///
/// ```objc
/// // Objective-C, after the Swift side has registered "corridor-key":
/// NSError *error = nil;
/// id<NFKInferenceBackend> keyer = [NFKMLXModelRegistry backendNamed:@"corridor-key" weightsURL:url error:&error];
/// NFKInferenceResult *result = [keyer runInferenceForRequest:request error:&error];
/// ```
@objc(NFKMLXModelRegistry)
public final class NFKMLXModelRegistry: NSObject {

    /// Builds a backend, loading from `weightsURL` when the model needs it. Runs on the caller's thread.
    public typealias Factory = @Sendable (_ weightsURL: URL?) throws -> any NFKInferenceBackend

    /// Builds a backend with load options, loading from `weightsURL` when the model needs it. Runs on the
    /// caller's thread. Introduced in InferKit 0.4.0.
    public typealias OptionsFactory = @Sendable (_ weightsURL: URL?, _ options: NFKMLXLoadOptions) throws
        -> any NFKInferenceBackend

    /// One registered model: its factory and the load-option keys the factory reads.
    private struct Registration {
        let consumes: Set<String>
        let factory: OptionsFactory
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var registrations: [String: Registration] = [:]

    /// Registers a model factory under `name` (call once, e.g. at launch). Swift-only: the factory
    /// captures the model's `MLXArray` forward. Registering a name again replaces it. The factory reads no
    /// load options.
    public static func register(name: String, factory: @escaping Factory) {
        register(name: name, consumes: []) { weightsURL, _ in try factory(weightsURL) }
    }

    /// Registers a model factory that reads load options under `name`, replacing any registration of the
    /// name. `consumes` lists the `NFKModelLoadOption*` keys the factory reads; a build reports the others
    /// it was given unread. Introduced in InferKit 0.4.0.
    public static func register(name: String, consumes: Set<String>, factory: @escaping OptionsFactory) {
        lock.withLock {
            registrations[name] = Registration(consumes: consumes, factory: factory)
        }
    }

    /// Removes a registered model.
    @objc public static func unregisterModel(named name: String) {
        lock.withLock {
            registrations[name] = nil
        }
    }

    /// The registered model names, sorted.
    @objc public static var registeredModelNames: [String] {
        lock.withLock { registrations.keys.sorted() }
    }

    /// Whether a model is registered under `name`.
    @objc public static func isModelRegistered(_ name: String) -> Bool {
        lock.withLock { registrations[name] != nil }
    }

    /// The `NFKModelLoadOption*` keys the factory registered under `name` reads, empty for a name that is
    /// not registered or a factory that reads none. Introduced in InferKit 0.4.0.
    @objc(consumedOptionKeysForModelNamed:)
    public static func consumedOptionKeys(forModelNamed name: String) -> Set<String> {
        lock.withLock { registrations[name]?.consumes ?? [] }
    }

    /// Builds the backend registered under `name`, passing `weightsURL` to its factory. Returns nil
    /// with an error when no model is registered under the name or the factory throws.
    ///
    /// @discussion A backend built from a weights URL (a file or a release directory) can unload its model
    /// and load it again: the module, matting, speech, language, and generation backends keep the factory
    /// call to load from, and any other backend comes back wrapped in an ``NFKMLXReloadableBackend``, whose
    /// `wrappedBackend` reaches the model's class while it is loaded. Its
    /// `residentByteCount` and `peakByteCount` measure this build. A backend built without a URL holds
    /// random weights or none, which another build would not reproduce, so it keeps its model and its
    /// `unload` does nothing.
    @objc(backendNamed:weightsURL:error:)
    public static func backend(named name: String, weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(named: name, weightsURL: weightsURL, loadOptions: nil)
    }

    /// Builds the backend registered under `name` with load options, as ``backend(named:weightsURL:)``
    /// builds it.
    ///
    /// @discussion `options` holds resolved load options keyed by `NFKModelLoadOption*` keys, such as
    /// `@{NFKModelLoadOptionResidency: NFKModelResidencyPaged, NFKModelLoadOptionQuantizationBits: @4}`
    /// (see ``NFKMLXLoadOptions``). A draft model under `NFKModelLoadOptionDraftModel` is a file URL or a
    /// path to the draft's release, or a descriptor identifier whose files the shared catalog's locator
    /// finds. A backend that unloads builds again with the same options. Its `loadOptionValues` reports them
    /// as given, and its `modelInfo` the residency the load resolved to under `NFKModelInfoResidency`.
    /// `NFKModelCatalog`'s build entry validates options
    /// against a descriptor first; this entry passes them to the factory as given. Introduced in
    /// InferKit 0.4.0.
    @objc(backendNamed:weightsURL:options:error:)
    public static func backend(named name: String, weightsURL: URL?,
                               options: [String: Any]?) throws -> any NFKInferenceBackend {
        guard let options else {
            return try backend(named: name, weightsURL: weightsURL, loadOptions: nil)
        }
        let draftURL = try NFKMLXLoadOptions.draftLocation(options[NFKModelLoadOptionDraftModel],
                                                            in: NFKModelCatalog.shared)
        return try backend(named: name, weightsURL: weightsURL,
                           loadOptions: NFKMLXLoadOptions(options, draftURL: draftURL))
    }

    /// Builds the backend registered under `name` with `loadOptions`, running `configure` on each backend
    /// the factory builds, the first and every one after an unload. A build without load options passes
    /// the factory empty ones, and its backend reports none.
    static func backend(named name: String, weightsURL: URL?, loadOptions: NFKMLXLoadOptions?,
                        configure: @escaping (any NFKInferenceBackend) throws -> Void = { _ in })
        throws -> any NFKInferenceBackend {
        guard let factory = lock.withLock({ registrations[name]?.factory }) else {
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceNotReady.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "no MLX model is registered under '\(name)'"])
        }
        let build: () throws -> any NFKInferenceBackend = {
            let backend = try factory(weightsURL, loadOptions ?? NFKMLXLoadOptions())
            try configure(backend)
            loadOptions?.noteResidency(of: backend)
            return backend
        }
        guard weightsURL != nil else {
            let backend = try build()
            NFKMLXReload.reportLoadOptions(of: backend, loadOptions)
            return backend
        }
        return try NFKMLXReload.reloadable(loadOptions: loadOptions, build)
    }
}
