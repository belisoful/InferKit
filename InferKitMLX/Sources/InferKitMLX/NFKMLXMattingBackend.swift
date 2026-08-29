//
//  NFKMLXMattingBackend.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import Metal
import InferKit
import MLX

/// How the matting backend reads inputs, tiles the work, and writes outputs.
public struct NFKMattingConfiguration: @unchecked Sendable {
    /// The channel count the forward expects for the plate, 3 (RGB) or 4 (RGBA).
    public var plateChannels: Int
    /// Color space and premultiplication for the written image.
    public var imageOptions: NFKMLXImageOptions
    /// Process the plate in `tileSize` × `tileSize` tiles (0 runs the whole image at once). A large
    /// plate is bounded in memory by running the forward tile by tile and stitching the results.
    public var tileSize: Int
    /// Also emit the alpha matte on its own, as a gray image under `NFKOutputMask`.
    public var emitsMatte: Bool
    /// Return `MTLTexture`s instead of `CGImage`s (for a Metal render pipeline).
    public var outputsTexture: Bool
    /// The device for texture output. Defaults to the system default device.
    public var device: MTLDevice?

    public init(plateChannels: Int = 3,
                imageOptions: NFKMLXImageOptions = NFKMLXImageOptions(),
                tileSize: Int = 0,
                emitsMatte: Bool = false,
                outputsTexture: Bool = false,
                device: MTLDevice? = nil) {
        precondition(plateChannels == 3 || plateChannels == 4, "plateChannels must be 3 or 4")
        self.plateChannels = plateChannels
        self.imageOptions = imageOptions
        self.tileSize = tileSize
        self.emitsMatte = emitsMatte
        self.outputsTexture = outputsTexture
        self.device = device
    }
}

/// An InferKit backend for a bring-your-own MLX image-matting model (a green/blue-screen keyer, a
/// portrait matter, a background remover).
///
/// Where `NFKMLXModuleBackend` is single RGB image in, single RGB image out, a matting model needs an
/// alpha channel out and often a hint in. This backend carries both, through `NFKMLXImageBridge`
/// (which preserves alpha and accepts either a `CGImage` or an `MTLTexture`):
///
/// - Input: the plate under `NFKInputImage` and an optional hint (a trimap or coarse alpha) under
///   `NFKInputMask`. Each is a `CGImage` or an `MTLTexture`.
/// - Forward: a closure mapping the plate tensor `[H, W, C]` and the optional hint tensor `[H, W, 1]`
///   (both `0...1`) to an output tensor `[H, W, 4]` — straight foreground RGB and the alpha matte.
/// - Output: the composited RGBA image under `NFKOutputImage`, and (when `emitsMatte`) the matte on
///   its own under `NFKOutputMask`. `CGImage` by default, `MTLTexture` when the configuration asks.
///
/// See `NFKMattingConfiguration` for tiling, premultiplication, color space, and texture output.
@objc(NFKMLXMattingBackend)
public final class NFKMLXMattingBackend: NSObject, NFKInferenceBackend {

    /// Maps the plate tensor `[H, W, C]` and an optional hint tensor `[H, W, 1]` (both `0...1`) to an
    /// output tensor `[H, W, 4]` (straight foreground RGB + alpha matte, `0...1`).
    public typealias Forward = @Sendable (_ plate: MLXArray, _ hint: MLXArray?) -> MLXArray

    /// The same mapping, with the originating request available — for a model whose behavior depends on a
    /// parameter the plate cannot carry, such as SAM's click point.
    public typealias RequestForward = @Sendable (_ plate: MLXArray, _ hint: MLXArray?,
                                                 _ request: NFKInferenceRequest) -> MLXArray

    private let forward: RequestForward
    private let identifier: String
    private let ready: Bool
    private let configuration: NFKMattingConfiguration

    public init(identifier: String = "mlx-matting",
                isReady: Bool = true,
                configuration: NFKMattingConfiguration = NFKMattingConfiguration(),
                forward: @escaping Forward) {
        self.identifier = identifier
        self.ready = isReady
        self.configuration = configuration
        self.forward = { plate, hint, _ in forward(plate, hint) }
        super.init()
    }

    public init(identifier: String = "mlx-matting",
                isReady: Bool = true,
                configuration: NFKMattingConfiguration = NFKMattingConfiguration(),
                requestForward: @escaping RequestForward) {
        self.identifier = identifier
        self.ready = isReady
        self.configuration = configuration
        self.forward = requestForward
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
        Task.detached(priority: .userInitiated) {
            do {
                let outputs = try NFKMLXMattingBackend.run(request, configuration: configuration, forward: forward)
                job.finish(with: NFKInferenceResult(outputs: outputs))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    private static func run(_ request: NFKInferenceRequest,
                            configuration: NFKMattingConfiguration,
                            forward: RequestForward) throws -> [String: Any] {
        guard let plateValue = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedInput
        }
        let colorSpace = configuration.imageOptions.colorSpace
        let (plateBytes, width, height) = try NFKMLXImageBridge.rgbaBytes(from: plateValue, colorSpace: colorSpace)
        let hint = try request.input(forKey: NFKInputMask).map {
            try NFKMLXImageBridge.rgbaBytes(from: $0, colorSpace: colorSpace)
        }

        let tile = configuration.tileSize > 0 ? configuration.tileSize : max(width, height, 1)
        var result = [UInt8](repeating: 0, count: width * height * 4)
        var y = 0
        while y < height {
            let tileHeight = min(tile, height - y)
            var x = 0
            while x < width {
                let tileWidth = min(tile, width - x)
                let platePatch = subImage(plateBytes, width: width, x: x, y: y, tileWidth: tileWidth, tileHeight: tileHeight)
                let plateTensor = NFKMLXImageBridge.tensor(rgba: platePatch, width: tileWidth, height: tileHeight,
                                                           channels: configuration.plateChannels)
                let hintTensor = hint.map { hintBytes in
                    let patch = subImage(hintBytes.bytes, width: width, x: x, y: y, tileWidth: tileWidth, tileHeight: tileHeight)
                    return NFKMLXImageBridge.tensor(rgba: patch, width: tileWidth, height: tileHeight, channels: 1)
                }
                let output = forward(plateTensor, hintTensor, request)
                eval(output)
                let (patchBytes, patchWidth, patchHeight) = try NFKMLXImageBridge.rgbaBytes(from: output)
                guard patchWidth == tileWidth, patchHeight == tileHeight else {
                    throw NFKMLXError.noOutput
                }
                place(patchBytes, tileWidth: tileWidth, tileHeight: tileHeight, into: &result, width: width, x: x, y: y)
                x += tileWidth
            }
            y += tileHeight
        }

        var outputs: [String: Any] = [:]
        outputs[NFKOutputImage] = try makeOutput(rgba: result, width: width, height: height, configuration: configuration)
        if configuration.emitsMatte {
            outputs[NFKOutputMask] = try makeOutput(rgba: matte(from: result), width: width, height: height, configuration: configuration)
        }
        return outputs
    }

    private static func makeOutput(rgba: [UInt8], width: Int, height: Int,
                                   configuration: NFKMattingConfiguration) throws -> Any {
        if configuration.outputsTexture {
            guard let device = configuration.device ?? MTLCreateSystemDefaultDevice() else {
                throw NFKMLXImageBridge.BridgeError.noMetalDevice
            }
            return try NFKMLXImageBridge.texture(rgba: rgba, width: width, height: height,
                                                 device: device, options: configuration.imageOptions)
        }
        return try NFKMLXImageBridge.cgImage(rgba: rgba, width: width, height: height, options: configuration.imageOptions)
    }

    // MARK: Byte-level tiling and matte (no MLX)

    /// A tile of RGBA8 bytes copied out of a larger RGBA8 buffer.
    static func subImage(_ source: [UInt8], width: Int, x: Int, y: Int, tileWidth: Int, tileHeight: Int) -> [UInt8] {
        var tile = [UInt8](repeating: 0, count: tileWidth * tileHeight * 4)
        for row in 0 ..< tileHeight {
            let sourceStart = ((y + row) * width + x) * 4
            let tileStart = row * tileWidth * 4
            for index in 0 ..< (tileWidth * 4) {
                tile[tileStart + index] = source[sourceStart + index]
            }
        }
        return tile
    }

    /// Copies a tile of RGBA8 bytes into a larger RGBA8 buffer at (x, y).
    static func place(_ tile: [UInt8], tileWidth: Int, tileHeight: Int, into destination: inout [UInt8], width: Int, x: Int, y: Int) {
        for row in 0 ..< tileHeight {
            let destinationStart = ((y + row) * width + x) * 4
            let tileStart = row * tileWidth * 4
            for index in 0 ..< (tileWidth * 4) {
                destination[destinationStart + index] = tile[tileStart + index]
            }
        }
    }

    /// A gray RGBA image whose value is the alpha of the input, so the matte can be shown on its own.
    static func matte(from rgba: [UInt8]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: rgba.count)
        var index = 0
        while index < rgba.count {
            let alpha = rgba[index + 3]
            out[index] = alpha; out[index + 1] = alpha; out[index + 2] = alpha; out[index + 3] = 255
            index += 4
        }
        return out
    }
}
