//
//  NFKMLXReloadableBackend.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
internal import InferKitMLXSupport
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
/// which is built first when the model is unloaded. A forwarded message counts as a run in flight until
/// it returns, so an unload during it frees the model when it returns. An object the message returns that
/// holds part of the model keeps that part after an unload. A Swift cast of the wrapper to the model's
/// class fails; cast ``wrappedBackend`` instead.
///
/// A build after an unload reads local files only. When the files the model was built from are gone,
/// the run fails with ``NFKMLXReloadError/modelFilesRemoved`` and makes no network request.
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
        _ = Self.forwardingInstalled
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

    /// The held backend's count of the request's input tokens, as a run in flight. The backend is built
    /// first when the model is unloaded. The wrapper responds to `tokenCountForRequest:error:` only when
    /// the held backend's class counts tokens. Introduced in InferKit 0.4.0.
    @objc(tokenCountForRequest:error:)
    public func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        try lifecycle.withState { backend in
            guard backend.responds(to: Self.tokenCountSelector) else {
                throw NFKMLXError.unsupportedConfiguration("\(identifier) does not count tokens")
            }
            return try unsafeBitCast(backend, to: NFKMLXTokenCounting.self).tokenCount(for: request)
        }
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

    private static let tokenCountSelector = NSSelectorFromString("tokenCountForRequest:error:")

    private static let heldOnlySelectors: Set<Selector> = [
        NSSelectorFromString("supportedParameterKeys"), NSSelectorFromString("supportedInputKeys"),
        NSSelectorFromString("supportedOutputKeys"), NSSelectorFromString("backendStatus"), tokenCountSelector,
    ]

    /// Answers for the held backend's class on the optional keys, status, and token count, and for either
    /// object otherwise.
    public override func responds(to aSelector: Selector!) -> Bool {
        guard let aSelector else {
            return false
        }
        if Self.heldOnlySelectors.contains(aSelector) {
            return class_respondsToSelector(backendClass, aSelector)
        }
        return super.responds(to: aSelector) || class_respondsToSelector(backendClass, aSelector)
    }

    /// Gives the class the two methods of the runtime's full forwarding path. Swift cannot declare
    /// `methodSignatureForSelector:` or `forwardInvocation:`, so they are added as block implementations.
    /// `forwardInvocation:` is the support target's, which raises the exception ``forward(_:)`` returns
    /// once every Swift frame of the forward has returned.
    private static let forwardingInstalled: Void = {
        typealias Signature = @convention(block) (NFKMLXReloadableBackend, Selector) -> AnyObject?
        let signature: Signature = { wrapper, selector in wrapper.forwardedSignature(for: selector) }
        install(imp_implementationWithBlock(signature), for: NFKMLXInvocation.signatureSelector)
        install(NFKMLXForwardInvocationImplementation { receiver, invocation in
            (receiver as AnyObject as? NFKMLXReloadableBackend)?.forward(invocation as AnyObject)
        }, for: NFKMLXInvocation.forwardSelector)
    }()

    private static func install(_ implementation: IMP, for selector: Selector) {
        let types = class_getInstanceMethod(NSObject.self, selector).flatMap(method_getTypeEncoding)
        class_replaceMethod(NFKMLXReloadableBackend.self, selector, implementation, types)
    }

    /// The signature of a message the wrapper or `NSObject` implements, from `NSObject`, and otherwise of the
    /// held class's method; nil when neither implements it.
    private func forwardedSignature(for selector: Selector) -> AnyObject? {
        if class_getInstanceMethod(NFKMLXReloadableBackend.self, selector) != nil {
            return NFKMLXInvocation.objectSignature(of: self, for: selector)
        }
        guard let method = class_getInstanceMethod(backendClass, selector),
              let types = method_getTypeEncoding(method) else {
            return nil
        }
        return NFKMLXInvocation.signature(types: types)
    }

    /// Sends a message only the held class implements to the held backend, built first when the model is
    /// unloaded, as a run in flight from before the message to its return, and returns the exception the
    /// sender receives, or nil. An exception the message raises is caught in Objective-C and returned
    /// after the run ends, so no Swift frame holds the backend while it unwinds. A backend that cannot be
    /// built gives `NSInvalidArgumentException` with the build error's reason.
    private func forward(_ invocation: AnyObject) -> NSException? {
        let selector = NFKMLXInvocation.selector(of: invocation)
        switch invokeAsRun(invocation, selector: selector) {
        case .answered:
            return nil
        case .raised(let exception):
            return exception
        case .refused(let reason):
            return NSException(name: .invalidArgumentException,
                               reason: "\(identifier) cannot answer \(NSStringFromSelector(selector)): \(reason)",
                               userInfo: nil)
        }
    }

    private enum ForwardOutcome {
        case answered
        case raised(NSException)
        case refused(String)
    }

    /// Invokes `invocation` on the held backend inside a run, which ends before this returns.
    private func invokeAsRun(_ invocation: AnyObject, selector: Selector) -> ForwardOutcome {
        do {
            try lifecycle.begin()
        } catch {
            return .refused(error.localizedDescription)
        }
        defer { lifecycle.end() }
        let backend: any NFKInferenceBackend
        do {
            backend = try lifecycle.loadedState()
        } catch {
            return .refused(error.localizedDescription)
        }
        guard backend.responds(to: selector) else {
            return .refused("the built backend does not implement it")
        }
        if let exception = NFKMLXInvokeCatching(invocation, backend) {
            return .raised(exception)
        }
        return .answered
    }
}

/// The `NSInvocation` and `NSMethodSignature` calls the wrapper's forwarding makes. Both classes are
/// unavailable in Swift, so each method is called through its implementation.
private enum NFKMLXInvocation {
    static let signatureSelector = NSSelectorFromString("methodSignatureForSelector:")
    static let forwardSelector = NSSelectorFromString("forwardInvocation:")

    /// The invocation's selector.
    static func selector(of invocation: AnyObject) -> Selector {
        typealias Getter = @convention(c) (AnyObject, Selector) -> Selector
        let selector = NSSelectorFromString("selector")
        return unsafeBitCast(implementation(of: selector, in: invocation), to: Getter.self)(invocation, selector)
    }

    /// `+[NSMethodSignature signatureWithObjCTypes:]`.
    static func signature(types: UnsafePointer<CChar>) -> AnyObject? {
        typealias Make = @convention(c) (AnyClass, Selector, UnsafePointer<CChar>) -> AnyObject?
        guard let signatureClass = NSClassFromString("NSMethodSignature") else {
            return nil
        }
        let selector = NSSelectorFromString("signatureWithObjCTypes:")
        guard let method = class_getClassMethod(signatureClass, selector) else {
            return nil
        }
        return unsafeBitCast(method_getImplementation(method), to: Make.self)(signatureClass, selector, types)
    }

    /// `NSObject`'s own `methodSignatureForSelector:` for `object`.
    static func objectSignature(of object: AnyObject, for selector: Selector) -> AnyObject? {
        typealias Lookup = @convention(c) (AnyObject, Selector, Selector) -> AnyObject?
        guard let method = class_getInstanceMethod(NSObject.self, signatureSelector) else {
            return nil
        }
        return unsafeBitCast(method_getImplementation(method), to: Lookup.self)(object, signatureSelector, selector)
    }

    /// Raises `NSInvalidArgumentException` with `reason`.
    static func raise(_ reason: String) -> Never {
        NSException(name: .invalidArgumentException, reason: reason, userInfo: nil).raise()
        fatalError(reason)
    }

    private static func implementation(of selector: Selector, in object: AnyObject) -> IMP {
        guard let method = class_getInstanceMethod(object_getClass(object), selector) else {
            raise("\(type(of: object)) does not implement \(NSStringFromSelector(selector))")
        }
        return method_getImplementation(method)
    }
}

/// The optional token count of `NFKInferenceBackend`, sent as a message after a `responds(to:)` check.
/// The Swift 6.4 compiler crashes generating the call through the protocol's optional requirement.
@objc private protocol NFKMLXTokenCounting {
    @objc(tokenCountForRequest:error:)
    func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber
}
