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

    private let forward: Forward
    private let identifier: String
    private let ready: Bool

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
        self.forward = forward
        super.init()
    }

    // MARK: NFKInferenceBackend

    @objc public var isReady: Bool { ready }

    @objc public var backendIdentifier: String { identifier }

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
        let forward = self.forward
        Task.detached {
            do {
                let cgImage = try NFKMLXModuleBackend.run(request, forward: forward)
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputImage: cgImage]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    private static func run(_ request: NFKInferenceRequest, forward: Forward) throws -> CGImage {
        guard let value = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedInput
        }
        // Through the shared bridge, which accepts a CGImage or an MTLTexture as the input image.
        let input = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
        let output = forward(input)
        eval(output)
        return try NFKMLXImageBridge.cgImage(from: output, options: NFKMLXImageOptions())
    }
}
