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

    private let forward: Forward
    private let identifier: String
    private let ready: Bool
    private let configuration: NFKMLXTensorConfiguration

    public init(identifier: String = "mlx-tensor",
                isReady: Bool = true,
                configuration: NFKMLXTensorConfiguration,
                forward: @escaping Forward) {
        self.identifier = identifier
        self.ready = isReady
        self.configuration = configuration
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
        let configuration = self.configuration
        Task.detached {
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
                            forward: Forward) throws -> [String: Any] {
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

        let produced = forward(named)

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
