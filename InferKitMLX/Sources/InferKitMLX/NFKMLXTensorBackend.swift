//
//  NFKMLXTensorBackend.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import Metal
import InferKit
import MLX

/// Binds an InferKit request/result key to a tensor in the forward's dictionary.
public struct NFKMLXTensorPort: Sendable {
    /// The InferKit key: a request input key (for an input) or a result output key (for an output).
    public var key: String
    /// The name of the tensor in the forward's dictionary.
    public var tensorName: String
    /// For an input, the channel count to read (1, 3, or 4). Ignored for an output.
    public var channels: Int

    public init(key: String, tensorName: String, channels: Int = 3) {
        self.key = key
        self.tensorName = tensorName
        self.channels = channels
    }
}

/// How a tensor backend maps request inputs and result outputs.
public struct NFKMLXTensorConfiguration: @unchecked Sendable {
    public var inputs: [NFKMLXTensorPort]
    public var outputs: [NFKMLXTensorPort]
    public var imageOptions: NFKMLXImageOptions
    public var outputsTexture: Bool
    public var device: MTLDevice?

    public init(inputs: [NFKMLXTensorPort],
                outputs: [NFKMLXTensorPort],
                imageOptions: NFKMLXImageOptions = NFKMLXImageOptions(),
                outputsTexture: Bool = false,
                device: MTLDevice? = nil) {
        self.inputs = inputs
        self.outputs = outputs
        self.imageOptions = imageOptions
        self.outputsTexture = outputsTexture
        self.device = device
    }
}

/// A general bring-your-own MLX backend over named image tensors: several inputs in, several outputs
/// out. Where `NFKMLXModuleBackend` is one image in, one out, and `NFKMLXMattingBackend` is a plate
/// plus a hint to a matte, this covers the rest — a compositing model that reads a foreground plate
/// and a background, a model that returns both an image and a mask — by naming each port.
///
/// Each configured input image (a `CGImage` or an `MTLTexture` under its request key) becomes a
/// tensor in the forward's dictionary; each configured output tensor becomes an image under its
/// result key. An input absent from the request is simply omitted from the dictionary.
@objc(NFKMLXTensorBackend)
public final class NFKMLXTensorBackend: NSObject, NFKInferenceBackend {

    public typealias Forward = @Sendable ([String: MLXArray]) -> [String: MLXArray]

    /// A forward that also reads the request, for a model conditioned on a per-request value (an
    /// interpolation timestep). Introduced in InferKit 0.4.0.
    public typealias RequestForward = @Sendable (_ tensors: [String: MLXArray], _ request: NFKInferenceRequest) -> [String: MLXArray]

    private let forward: RequestForward
    private let identifier: String
    private let ready: Bool
    private let configuration: NFKMLXTensorConfiguration
    private let forwardParameterKeys: Set<String>

    public init(identifier: String = "mlx-tensor",
                isReady: Bool = true,
                configuration: NFKMLXTensorConfiguration,
                forward: @escaping Forward) {
        self.identifier = identifier
        self.ready = isReady
        self.configuration = configuration
        self.forwardParameterKeys = []
        self.forward = { tensors, _ in forward(tensors) }
        super.init()
    }

    /// The request-aware form: `requestForward` receives the bridged input tensors and the request
    /// they came from. Introduced in InferKit 0.4.0.
    ///
    /// - Parameters:
    ///   - identifier: The value reported by `backendIdentifier`.
    ///   - isReady: Whether the model's weights are already loaded.
    ///   - configuration: The input and output ports.
    ///   - forwardParameterKeys: The request parameters `requestForward` reads. They form
    ///     `supportedParameterKeys`.
    ///   - requestForward: Maps the named input tensors and their request to named output tensors.
    public init(identifier: String = "mlx-tensor",
                isReady: Bool = true,
                configuration: NFKMLXTensorConfiguration,
                forwardParameterKeys: Set<String>,
                requestForward: @escaping RequestForward) {
        self.identifier = identifier
        self.ready = isReady
        self.configuration = configuration
        self.forwardParameterKeys = forwardParameterKeys
        self.forward = requestForward
        super.init()
    }

    // MARK: NFKInferenceBackend

    @objc public var isReady: Bool { ready }

    @objc public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads: those the request-aware forward was built to read.
    /// Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { forwardParameterKeys }

    /// The request inputs the backend reads: one per configured input port. Introduced in
    /// InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { Set(configuration.inputs.map(\.key)) }

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
        let configuration = self.configuration
        Task.detached(priority: .userInitiated) {
            do {
                let outputs = try NFKMLXTensorBackend.run(request, configuration: configuration, forward: forward)
                job.finish(with: NFKInferenceResult(outputs: outputs))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    private static func run(_ request: NFKInferenceRequest,
                            configuration: NFKMLXTensorConfiguration,
                            forward: RequestForward) throws -> [String: Any] {
        let colorSpace = configuration.imageOptions.colorSpace
        var named: [String: MLXArray] = [:]
        for port in configuration.inputs {
            guard let value = request.input(forKey: port.key) else {
                continue
            }
            named[port.tensorName] = try NFKMLXImageBridge.tensor(from: value, channels: port.channels, colorSpace: colorSpace)
        }
        guard !named.isEmpty else {
            throw NFKMLXError.unsupportedInput
        }

        let produced = forward(named, request)

        var outputs: [String: Any] = [:]
        for port in configuration.outputs {
            guard let array = produced[port.tensorName] else {
                continue
            }
            eval(array)
            if configuration.outputsTexture {
                guard let device = configuration.device ?? MTLCreateSystemDefaultDevice() else {
                    throw NFKMLXImageBridge.BridgeError.noMetalDevice
                }
                outputs[port.key] = try NFKMLXImageBridge.texture(from: array, device: device, options: configuration.imageOptions)
            } else {
                outputs[port.key] = try NFKMLXImageBridge.cgImage(from: array, options: configuration.imageOptions)
            }
        }
        return outputs
    }
}
