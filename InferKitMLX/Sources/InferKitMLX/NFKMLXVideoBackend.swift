//
//  NFKMLXVideoBackend.swift
//  InferKitMLX
//
//  The first backend that PRODUCES video. `NFKModalityVideo` and the `NFKInputVideo` / `NFKOutputVideo`
//  keys have been in the core's vocabulary with nothing emitting a clip; this reads one, runs a
//  sequence transform over its frames, and writes the result back out as an `NFKVideoAsset`.
//
//  A clip is a sequence, not a batch of independent images: frame interpolation reads pairs and
//  returns more frames than it took, and video super-resolution propagates state both forward and
//  backward through time. The transform is therefore `[MLXArray] -> [MLXArray]` rather than per-frame,
//  and a model that IS per-frame simply maps over it.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// How a video backend reads and writes the clips it works on.
public struct NFKMLXVideoConfiguration: Sendable {

    /// The frame rate the result is written at, or nil to keep the source's.
    ///
    /// Frame interpolation changes it: a transform that returns two frames for every one it took has
    /// doubled the rate, and writing at the source's rate would produce a slow-motion clip instead of
    /// a smoother one.
    public var outputFramesPerSecond: Double?

    /// A rate multiplier applied to the source's, used when the transform's output length is known to
    /// scale with the input's. Ignored when ``outputFramesPerSecond`` is set.
    public var frameRateMultiplier: Double = 1

    /// The most frames decoded from a source clip.
    public var frameLimit: Int = 900

    /// Where the result is written. A nil directory writes to the system temporary directory.
    public var outputDirectoryURL: URL?

    public init() {}
}

/// Holds the transform for capture in the backend's `@Sendable` body.
private final class NFKMLXVideoTransformBox: @unchecked Sendable {
    let transform: ([MLXArray]) -> [MLXArray]
    init(_ transform: @escaping ([MLXArray]) -> [MLXArray]) { self.transform = transform }
}

/// A bring-your-own MLX video model as an InferKit backend.
///
/// Reads an `NFKVideoAsset` under `NFKInputVideo`, decodes it, hands the frames to the transform as
/// `[H, W, 3]` tensors in `0...1`, encodes what comes back, and returns a new `NFKVideoAsset` under
/// `NFKOutputVideo`. The toolkit owns the decode, the bridge, and the encode; the caller owns the
/// model, mirroring `NFKMLXModuleBackend`.
///
/// Inference over a clip is long. The contract's `submitInferenceJobForRequest:` reports progress per
/// frame and takes cancellation.
public final class NFKMLXVideoBackend: NSObject, NFKInferenceBackend {

    private let identifier: String
    private let ready: Bool
    private let configuration: NFKMLXVideoConfiguration
    private let box: NFKMLXVideoTransformBox

    /// Builds a video backend around a sequence transform.
    ///
    /// - Parameter transform: takes every decoded frame as `[H, W, 3]` in `0...1` and returns the
    ///   frames to write, which need not be the same count.
    public init(identifier: String, isReady: Bool = true,
                configuration: NFKMLXVideoConfiguration = NFKMLXVideoConfiguration(),
                transform: @escaping ([MLXArray]) -> [MLXArray]) {
        self.identifier = identifier
        self.ready = isReady
        self.configuration = configuration
        self.box = NFKMLXVideoTransformBox(transform)
        super.init()
    }

    public var isReady: Bool { ready }
    public var backendIdentifier: String { identifier }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let asset = request.input(forKey: NFKInputVideo) as? NFKVideoAsset,
              let sourceURL = asset.fileURL else {
            throw NFKMLXError.unsupportedInput
        }
        let clip = try NFKMLXVideoFile.read(sourceURL, frameLimit: configuration.frameLimit)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let frames = clip.frames.map { frame -> MLXArray in
            let (bytes, width, height) = NFKMLXImageBridge.rgbaBytes(from: frame, colorSpace: colorSpace)
            return NFKMLXImageBridge.tensor(rgba: bytes, width: width, height: height, channels: 3)
        }

        let produced = box.transform(frames)
        guard !produced.isEmpty else { throw NFKMLXError.noOutput }
        eval(produced)

        var options = NFKMLXImageOptions()
        options.colorSpace = colorSpace
        let written = try produced.map { try NFKMLXImageBridge.cgImage(from: $0, options: options) }

        let rate = configuration.outputFramesPerSecond
            ?? clip.framesPerSecond * configuration.frameRateMultiplier
        let outputURL = (configuration.outputDirectoryURL ?? FileManager.default.temporaryDirectory)
            .appendingPathComponent("\(identifier)-\(UUID().uuidString).mov")
        try NFKMLXVideoFile.write(written, framesPerSecond: rate, to: outputURL)

        let first = written[0]
        let result = NFKVideoAsset(fileURL: outputURL,
                                   durationSeconds: Double(written.count) / rate,
                                   framesPerSecond: rate,
                                   dimensions: CGSize(width: first.width, height: first.height))
        return NFKInferenceResult(outputs: [NFKOutputVideo: result])
    }
}
