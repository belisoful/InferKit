//
//  NFKMLXReloadableBackend.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

/// A backend that can unload any MLX backend's model and build it again on demand.
///
/// @discussion It holds a closure that builds the backend, and the backend that closure last built.
/// ``unload()`` releases that backend, which frees its weights and caches, and the next run, or
/// ``prepare()``, builds it again through the closure. The closure must build the same model each time:
/// from a weights file or a release directory, not from random weights.
///
/// `NFKMLXModelRegistry`, `NFKMLXHub`, and the asynchronous download factories build reloadable backends.
/// The module, matting, speech, language, and generation backends unload in place and keep their class.
/// Every other class arrives wrapped in this one, and ``wrappedBackend`` reaches the model's own class
/// while it is loaded. A backend from a direct synchronous factory call reloads only when a caller wraps
/// it:
///
/// ```objc
/// id<NFKInferenceBackend> upscaler = [[NFKMLXReloadableBackend alloc] initWithProvider:^id<NFKInferenceBackend>(NSError **error) {
///     return [NFKMLXRealESRGAN backendWithVariant:NFKMLXRealESRGANVariantX4 weightsURL:url error:error];
/// } error:&error];
/// ```
///
/// The identifier and the declared keys are those of the first backend built, and they hold while the
/// model is unloaded. An Objective-C message the wrapper does not implement goes to the backend it holds,
/// which is built first when the model is unloaded. A Swift cast of the wrapper to the model's class
/// fails; cast ``wrappedBackend`` instead.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXReloadableBackend)
public final class NFKMLXReloadableBackend: NSObject, NFKInferenceBackend {

    /// Builds the backend; called once at creation and again after each unload. Introduced in InferKit 0.4.0.
    public typealias Build = () throws -> any NFKInferenceBackend

    private let lifecycle: NFKMLXLifecycle<any NFKInferenceBackend>
    private let identifier: String
    private let backendClass: AnyClass
    private let inputKeys: Set<String>
    private let parameterKeys: Set<String>
    private let outputKeys: Set<String>
    private let infoLock = NSLock()
    private var lastModelInfo: [String: Any] = [:]

    init(backend: any NFKInferenceBackend, measurement: NFKMLXLoadMeasurement?, build: @escaping Build,
         loadOptions: NFKMLXLoadOptions? = nil) {
        lifecycle = NFKMLXLifecycle(backend)
        identifier = backend.backendIdentifier
        backendClass = object_getClass(backend) ?? NSObject.self
        inputKeys = backend.supportedInputKeys ?? []
        parameterKeys = backend.supportedParameterKeys ?? []
        outputKeys = backend.supportedOutputKeys ?? []
        super.init()
        lifecycle.enableReload(build, measurement: measurement)
        if let loadOptions {
            lifecycle.setLoadOptions(loadOptions)
        }
    }

    /// Gives the wrapper the load options its backend was built with.
    func setLoadOptions(_ options: NFKMLXLoadOptions) {
        lifecycle.setLoadOptions(options)
    }

    /// Builds the backend through `build` and keeps `build` to build it again after an unload.
    ///
    /// - Throws: what the first build throws.
    ///
    /// Introduced in InferKit 0.4.0.
    public convenience init(build: @escaping Build) throws {
        let (backend, measurement) = try NFKMLXLoadMeasurement.measuring(build)
        self.init(backend: backend, measurement: measurement, build: build)
    }

    /// The Objective-C form of ``init(build:)``: `provider` returns the backend, or nil with an error.
    /// Objective-C: `-initWithProvider:error:`. Introduced in InferKit 0.4.0.
    @objc(initWithProvider:error:)
    public convenience init(provider: @escaping NFKServedBackendProvider) throws {
        try self.init(build: {
            var error: NSError?
            if let backend = provider(&error) {
                return backend
            }
            throw error ?? NFKMLXError.notReady
        })
    }

    /// The backend the wrapper holds, which a Swift caller casts to the model's own class, or nil while the
    /// model is unloaded. Reading it does not load the model; a run or ``prepare()`` does. A caller that
    /// keeps the returned backend keeps its memory after an unload. Introduced in InferKit 0.4.0.
    @objc public var wrappedBackend: (any NFKInferenceBackend)? {
        lifecycle.ifLoaded { $0 }
    }

    // MARK: NFKInferenceBackend

    /// The held backend's answer, or true while the model is unloaded, since a run builds it.
    @objc public var isReady: Bool { lifecycle.ifLoaded { $0.isReady } ?? true }

    @objc public var backendIdentifier: String { identifier }

    /// The first backend's request parameters. Present only when that backend declares them.
    /// Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { parameterKeys }

    /// The first backend's request inputs. Present only when that backend declares them.
    /// Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { inputKeys }

    /// The first backend's result keys. Present only when that backend declares them.
    /// Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { outputKeys }

    /// The held backend's `modelInfo`, the last one read while the model is unloaded, with
    /// `NFKModelInfoResidentBytes` and `NFKModelInfoPeakBytes` from the last measured build, and
    /// `NFKModelInfoResidency` from a build with load options. Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        if let current = lifecycle.ifLoaded({ $0.modelInfo ?? [:] }) {
            infoLock.withLock { lastModelInfo = current }
        }
        var info = infoLock.withLock { lastModelInfo }
        info.merge(lifecycle.loadInfo) { _, measured in measured }
        return info
    }

    /// The held backend's `backendStatus`, or empty while the model is unloaded. Present only when the
    /// held backend reports one. Introduced in InferKit 0.4.0.
    @objc public var backendStatus: [String: Any] {
        lifecycle.ifLoaded { $0.backendStatus ?? [:] } ?? [:]
    }

    /// The load options the backend was built with, keyed by `NFKModelLoadOption*` keys, as requested with
    /// defaults filled; empty for a backend built without options. Introduced in InferKit 0.4.0.
    @objc public var loadOptionValues: [String: Any] { lifecycle.loadOptionValues }

    /// Releases the held backend, which frees its weights and caches; the next run builds it again. With
    /// runs in flight, the backend is released when the last of them ends. Introduced in InferKit 0.4.0.
    @objc public func unload() {
        lifecycle.unload()
    }

    /// Whether the wrapper holds a built backend. Introduced in InferKit 0.4.0.
    @objc public var isLoaded: Bool { lifecycle.isLoaded }

    /// The rise in MLX's active memory over the last build, or 0 while the model is unloaded. Introduced
    /// in InferKit 0.4.0.
    @objc public var residentByteCount: Int { lifecycle.residentByteCount }

    /// The peak of MLX's active memory during the last build, above its level before it. Introduced in
    /// InferKit 0.4.0.
    @objc public var peakByteCount: Int { lifecycle.peakByteCount }

    /// Builds the backend when the model is unloaded, then prepares it. Introduced in InferKit 0.4.0.
    @objc(prepareWithError:)
    public func prepare() throws {
        try lifecycle.prepare()
        try lifecycle.withState { backend in
            var error: NSError?
            guard NFKInferencePrepare(backend, &error) else {
                throw error ?? NFKMLXError.notReady
            }
        }
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try lifecycle.withState { try $0.runInference(for: request) }
    }

    /// Runs the request on a background task: a held backend that submits its own jobs reports progress and
    /// partial results through the returned job, and one that does not runs synchronously there. The run
    /// counts as in flight until the held backend's job ends. Introduced in InferKit 0.4.0.
    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let lifecycle = self.lifecycle
        Task.detached(priority: .userInitiated) {
            if job.status == .cancelled {
                return
            }
            do {
                try lifecycle.begin()
            } catch {
                job.finish(withError: error as NSError)
                return
            }
            guard let backend = try? lifecycle.loadedState() else {
                lifecycle.end()
                job.finish(withError: NFKMLXError.notReady as NSError)
                return
            }
            guard let inner = backend.submitInferenceJob?(for: request) else {
                do {
                    job.finish(with: try backend.runInference(for: request))
                } catch {
                    job.finish(withError: error as NSError)
                }
                lifecycle.end()
                return
            }
            inner.progressHandler = { inner in
                job.reportProgress(inner.progress, partialResult: inner.partialResult)
            }
            job.cancellationHandler = { inner.cancel() }
            inner.completionHandler = { inner in
                if let result = inner.result {
                    job.finish(with: result)
                } else if inner.status != .cancelled {
                    job.finish(withError: inner.error ?? NFKMLXError.noOutput as NSError)
                }
                lifecycle.end()
            }
        }
        return job
    }

    // MARK: Objective-C forwarding

    private static let heldOnlySelectors: Set<Selector> = [
        NSSelectorFromString("supportedParameterKeys"), NSSelectorFromString("supportedInputKeys"),
        NSSelectorFromString("supportedOutputKeys"), NSSelectorFromString("backendStatus"),
    ]

    /// Answers for the held backend's class on the optional keys and status, and for either object
    /// otherwise.
    public override func responds(to aSelector: Selector!) -> Bool {
        guard let aSelector else {
            return false
        }
        if Self.heldOnlySelectors.contains(aSelector) {
            return class_respondsToSelector(backendClass, aSelector)
        }
        return super.responds(to: aSelector) || class_respondsToSelector(backendClass, aSelector)
    }

    /// The held backend, built first when the model is unloaded, for a message only it implements.
    public override func forwardingTarget(for aSelector: Selector!) -> Any? {
        guard let aSelector, class_respondsToSelector(backendClass, aSelector) else {
            return nil
        }
        return try? lifecycle.withState { $0 }
    }
}
