//
//  NFKMLXModuleBackend.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX

/// An InferKit backend for a bring-your-own MLX image model.
///
/// The bundled `NFKMLXBackend` runs the Stable Diffusion releases this package implements.
/// A custom architecture (a model you built with `MLXNN`, e.g. a super-resolution or style network
/// that this package does not ship) plugs in here: supply a `forward` closure that maps an input
/// `MLXArray` to an output `MLXArray`, and this backend handles the InferKit contract and the
/// image bridge around it.
///
/// The image bridge is InferKitMLX's Stable Diffusion `Image` helper: a `CGImage` under
/// `NFKInputImage` becomes an `[H, W, 3]` `MLXArray` with values in `0...1`, the closure runs, and
/// the returned array becomes a `CGImage` under `NFKOutputImage`. A closure that needs a different
/// layout or normalization does that reshaping itself. Because a Swift closure over `MLXArray` is
/// not representable in Objective-C, this backend is constructed from Swift; an Objective-C consumer
/// still drives it through the `NFKInferenceBackend` protocol.
@objc(NFKMLXModuleBackend)
public final class NFKMLXModuleBackend: NSObject, NFKInferenceBackend {

    public typealias Forward = @Sendable (MLXArray) -> MLXArray

    /// A forward that also reads the request, for a model conditioned on more than the input image
    /// (a style image under `NFKInputControl`, a per-request strength).
    public typealias RequestForward = @Sendable (_ image: MLXArray, _ request: NFKInferenceRequest) -> MLXArray

    private let lifecycle: NFKMLXLifecycle<RequestForward>
    private let identifier: String
    private let ready: Bool
    private let forwardInputKeys: Set<String>
    private let forwardParameterKeys: Set<String>

    /// - Parameters:
    ///   - identifier: The value reported by `backendIdentifier`.
    ///   - isReady: Whether the model's weights are already loaded. A closure that lazily loads on
    ///     first call passes `true`; one that needs an explicit load step passes `false` and the
    ///     consumer loads before submitting work.
    ///   - forward: Maps an input image tensor to an output image tensor.
    public init(identifier: String = "mlx-module",
                isReady: Bool = true,
                forward: @escaping Forward) {
        self.identifier = identifier
        self.ready = isReady
        self.forwardInputKeys = []
        self.forwardParameterKeys = []
        self.lifecycle = NFKMLXLifecycle({ image, _ in forward(image) })
        super.init()
    }

    /// The request-aware form: `forward` receives the bridged input image and the request it came from.
    ///
    /// - Parameters:
    ///   - identifier: The value reported by `backendIdentifier`.
    ///   - isReady: Whether the model's weights are already loaded.
    ///   - forwardInputKeys: The request inputs `requestForward` reads beyond `NFKInputImage`, such
    ///     as a style image under `NFKInputControl`. They join `supportedInputKeys`.
    ///   - forwardParameterKeys: The request parameters `requestForward` reads, such as
    ///     `NFKParameterStrength`. They join `supportedParameterKeys`.
    ///   - requestForward: Maps an input image tensor and its request to an output image tensor.
    public init(identifier: String = "mlx-module",
                isReady: Bool = true,
                forwardInputKeys: Set<String> = [],
                forwardParameterKeys: Set<String> = [],
                requestForward: @escaping RequestForward) {
        self.identifier = identifier
        self.ready = isReady
        self.forwardInputKeys = forwardInputKeys
        self.forwardParameterKeys = forwardParameterKeys
        self.lifecycle = NFKMLXLifecycle(requestForward)
        super.init()
    }

    // MARK: NFKInferenceBackend

    @objc public var isReady: Bool { ready }

    @objc public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads: none of its own, plus whatever the request-aware
    /// forward was built to read. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { forwardParameterKeys }

    /// The request inputs the backend reads: the image, plus whatever the request-aware forward was
    /// built to read. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { forwardInputKeys.union([NFKInputImage]) }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputImage] }

    /// `NFKModelInfoResidentBytes` and `NFKModelInfoPeakBytes` from the last measured load, and
    /// `NFKModelInfoResidency` from a build with load options; empty before either. Introduced in
    /// InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] { lifecycle.loadInfo }

    // MARK: Unloading

    /// Frees the forward and the model it holds, then MLX's cache; the next run or ``prepare()`` loads the
    /// model again. It does nothing for a backend built from a closure alone, which has no source to load
    /// from: only one that `NFKMLXModelRegistry` or a download factory built from weights can unload. With
    /// runs in flight, the model is freed when the last of them ends. Introduced in InferKit 0.4.0.
    @objc public func unload() {
        lifecycle.unload()
    }

    /// Whether the backend holds its model. Introduced in InferKit 0.4.0.
    @objc public var isLoaded: Bool { lifecycle.isLoaded }

    /// The rise in MLX's active memory over the model's last load, or 0 while it is unloaded or before a
    /// measured load. Introduced in InferKit 0.4.0.
    @objc public var residentByteCount: Int { lifecycle.residentByteCount }

    /// The peak of MLX's active memory during the model's last measured load, above its level before it.
    /// Introduced in InferKit 0.4.0.
    @objc public var peakByteCount: Int { lifecycle.peakByteCount }

    /// The load options the backend was built with, keyed by `NFKModelLoadOption*` keys, as requested with
    /// defaults filled; empty for a backend built without options. Introduced in InferKit 0.4.0.
    @objc public var loadOptionValues: [String: Any] { lifecycle.loadOptionValues }

    /// Loads the model when it is unloaded. Introduced in InferKit 0.4.0.
    @objc(prepareWithError:)
    public func prepare() throws {
        try lifecycle.prepare()
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result {
            return result
        }
        if let error = job.error {
            throw error
        }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let lifecycle = self.lifecycle
        Task.detached(priority: .userInitiated) {
            do {
                let cgImage = try lifecycle.withState { try NFKMLXModuleBackend.run(request, forward: $0) }
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputImage: cgImage]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    private static func run(_ request: NFKInferenceRequest, forward: RequestForward) throws -> CGImage {
        guard let value = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedInput
        }
        // Through the shared bridge, which accepts a CGImage or an MTLTexture as the input image.
        let input = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
        let output = forward(input, request)
        eval(output)
        return try NFKMLXImageBridge.cgImage(from: output, options: NFKMLXImageOptions())
    }
}

extension NFKMLXModuleBackend: NFKMLXReloading {

    var canReload: Bool { lifecycle.canReload }

    func setLoadOptions(_ options: NFKMLXLoadOptions) {
        lifecycle.setLoadOptions(options)
    }

    func enableReload(_ build: @escaping () throws -> any NFKInferenceBackend, measurement: NFKMLXLoadMeasurement?) {
        lifecycle.enableReload(building: build, as: NFKMLXModuleBackend.self, measurement: measurement) {
            try $0.lifecycle.loadedState()
        }
    }
}
