//
//  NFKMLXSAM2VideoBackend.swift
//  InferKitMLX
//
//  SAM 2 video tracking behind the InferKit contract: a clip and a prompt on its first frame become the
//  object's mask on every frame, carried by the tracker's memory.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// The request-parameter keys the SAM 2 video backend reads beyond `NFKSAMPointKey`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXSAM2ParameterKey)
public final class NFKMLXSAM2ParameterKey: NSObject {
    /// A box around the object on the first frame, as an `NSArray` of four numbers `[x0, y0, x1, y1]`
    /// in pixels of the source frame. With `NFKSAMPointKey` the click joins the box; without it the box
    /// is the whole prompt. Introduced in InferKit 0.4.0.
    @objc public static let box = "NFKMLXParameterSAM2Box"
}

/// Holds a tracker for the video backend.
private final class NFKMLXSAM2VideoHolder: @unchecked Sendable {
    let net: NFKMLXSAM2TrackerNet
    init(_ net: NFKMLXSAM2TrackerNet) { self.net = net }
}

extension NFKMLXSAM2 {

    /// The registry name and identifier of the video backend. Introduced in InferKit 0.4.0.
    @objc public static let videoModelName = "sam2-video"

    /// The request inputs the video backend reads.
    static let videoInputKeys: Set<String> = [NFKInputVideo]

    /// The request parameters the video backend reads.
    static let videoParameterKeys: Set<String> = [NFKSAMPointKey, NFKMLXSAM2ParameterKey.box]

    /// The video backend over `net`.
    static func videoBackend(net: NFKMLXSAM2TrackerNet) -> any NFKInferenceBackend {
        let holder = NFKMLXSAM2VideoHolder(net)
        return NFKMLXGenerationBackend(identifier: videoModelName, inputKeys: videoInputKeys,
                                       parameterKeys: videoParameterKeys) { request, job in
            guard let asset = request.input(forKey: NFKInputVideo) as? NFKVideoAsset,
                  let sourceURL = asset.fileURL else {
                throw NFKMLXError.unsupportedInput
            }
            let clip = try NFKMLXVideoFile.read(sourceURL)
            guard let first = clip.frames.first else { throw NFKMLXError.unsupportedInput }
            let prompt = videoPrompt(from: request, width: first.width, height: first.height,
                                     size: holder.net.configuration.imageSize)
            job.reportProgress(0)
            let session = NFKMLXSAM2TrackerSession(frameCount: clip.frames.count)
            var masks = [CGImage]()
            var scores = [NSNumber]()
            for (index, frame) in clip.frames.enumerated() {
                if job.status == .cancelled { return nil }
                let (mask, score) = try track(frame, index: index, prompt: index == 0 ? prompt : nil,
                                              net: holder.net, session: session)
                masks.append(mask)
                scores.append(NSNumber(value: score))
                job.reportProgress(Double(index + 1) / Double(clip.frames.count))
            }
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("\(videoModelName)-\(UUID().uuidString).mov")
            try NFKMLXVideoFile.write(masks, framesPerSecond: clip.framesPerSecond, to: url)
            let video = NFKVideoAsset(fileURL: url,
                                      durationSeconds: Double(masks.count) / clip.framesPerSecond,
                                      framesPerSecond: clip.framesPerSecond,
                                      dimensions: CGSize(width: first.width, height: first.height))
            return NFKInferenceResult(outputs: [NFKOutputVideo: video,
                                                NFKOutputStructured: ["objectScores": scores]])
        }
    }

    /// Tracks one frame into `session` and returns its mask at the frame's size, gray on black, with the
    /// object score. The frame's memory is evaluated here so the next frame's graph does not hold it.
    private static func track(_ frame: CGImage, index: Int, prompt: [(x: Float, y: Float, label: Int)]?,
                              net: NFKMLXSAM2TrackerNet,
                              session: NFKMLXSAM2TrackerSession) throws -> (CGImage, Float) {
        let (height, width) = (frame.height, frame.width)
        let size = net.configuration.imageSize
        let plate = try NFKMLXImageBridge.tensor(from: frame, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
        let resized = NFKMLXResample.resizeBilinear(plate.reshaped([1, height, width, 3]), height: size, width: size)
        // SAM 2 normalizes with ImageNet statistics on the 0...1 scale.
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let deviation = MLXArray([Float(0.229), 0.224, 0.225])
        let prediction = net.track(image: (resized - mean) / deviation, frameIndex: index, points: prompt,
                                   session: session)
        let alpha = NFKMLXResample.resizeBilinear(sigmoid(prediction.maskLogits).expandedDimensions(axis: 3),
                                                  height: height, width: width).reshaped([height, width, 1])
        let mask = concatenated([alpha, alpha, alpha], axis: 2)
        var held = [mask, prediction.objectScore]
        if let tracked = session.frames[index] {
            held += [tracked.memory, tracked.memoryPosition, tracked.objectPointer]
        }
        eval(held)
        return (try NFKMLXGenerationBackend.cgImage(mask), prediction.objectScore.reshaped([-1])[0].item(Float.self))
    }

    /// The first frame's prompt in pixels of the model's input: the box's corners (labels 2 and 3) where
    /// the request gives a box, then the click (label 1) where it gives one or gives no box.
    static func videoPrompt(from request: NFKInferenceRequest, width: Int, height: Int,
                            size: Int) -> [(x: Float, y: Float, label: Int)] {
        let scale = Float(size)
        var prompt = [(x: Float, y: Float, label: Int)]()
        if let box = request.parameter(forKey: NFKMLXSAM2ParameterKey.box) as? [NSNumber], box.count >= 4,
           width > 0, height > 0 {
            func clamped(_ value: NSNumber, _ extent: Int) -> Float {
                min(max(value.floatValue / Float(extent), 0), 1) * scale
            }
            prompt.append((clamped(box[0], width), clamped(box[1], height), 2))
            prompt.append((clamped(box[2], width), clamped(box[3], height), 3))
        }
        if prompt.isEmpty || request.parameter(forKey: NFKSAMPointKey) != nil {
            let point = NFKMLXSAM2.point(from: request, width: width, height: height)
            prompt.append((point.x * scale, point.y * scale, 1))
        }
        return prompt
    }

    /// Builds the tracker over whole clips at a chosen size and release from optional local weights. A
    /// nil `weightsURL` builds random weights.
    ///
    /// @discussion The backend reads an `NFKVideoAsset` under `NFKInputVideo` (required) and a prompt on
    /// its first frame: a click under `NFKSAMPointKey` (`[x, y]` in pixels), a box under
    /// ``NFKMLXSAM2ParameterKey/box``, or both; with neither it clicks the frame's center. The tracker
    /// segments the first frame from the prompt and every later frame from its memory of the frames
    /// before. The result carries:
    /// - `NFKOutputVideo`: an `NFKVideoAsset` of the masks (a QuickTime movie in the temporary
    ///   directory) at the source's size and frame rate, the mask probability as gray, white on the
    ///   object.
    /// - `NFKOutputStructured`: `objectScores`, an `NSArray` with one number per frame, above zero where
    ///   the model reads the object as present.
    ///
    /// Run inference off the render thread: a clip is one forward pass per frame. Introduced in InferKit
    /// 0.4.0.
    @objc(videoBackendWithVariant:release:weightsURL:error:)
    public static func videoBackend(variant: NFKMLXSAM2Variant, release: NFKMLXSAM2Release,
                                    weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeTracker(variant: variant, release: release)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return videoBackend(net: net)
    }

    /// ``videoBackend(variant:release:weightsURL:)`` at the tiny SAM 2.1 size, the image backend's
    /// default. Introduced in InferKit 0.4.0.
    @objc(videoBackendWithWeightsURL:error:)
    public static func videoBackend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try videoBackend(variant: .tiny, release: .sam21, weightsURL: weightsURL)
    }

    /// Downloads the checkpoint from Hugging Face, then builds the video backend at a chosen size and
    /// release. A file already in the cache is not fetched again. The call blocks on the network, so run
    /// it off the render thread. Introduced in InferKit 0.4.0.
    @objc(videoBackendWithVariant:release:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func videoBackend(variant: NFKMLXSAM2Variant, release: NFKMLXSAM2Release, repo: String,
                                    weightsPath: String, revision: String?,
                                    cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try videoBackend(variant: variant, release: release, weightsURL: url)
    }

    /// The download factory at the tiny SAM 2.1 size. Introduced in InferKit 0.4.0.
    @objc(videoBackendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func videoBackend(repo: String, weightsPath: String, revision: String?,
                                    cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try videoBackend(variant: .tiny, release: .sam21, repo: repo, weightsPath: weightsPath,
                         revision: revision, cacheDirectoryURL: cacheDirectoryURL)
    }

    /// The asynchronous form of
    /// ``videoBackend(variant:release:repo:weightsPath:revision:cacheDirectoryURL:)``. The handler runs on
    /// the download's background queue. Introduced in InferKit 0.4.0.
    @objc(videoBackendWithVariant:release:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func videoBackend(variant: NFKMLXSAM2Variant, release: NFKMLXSAM2Release, repo: String,
                                    weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                                    completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try videoBackend(variant: variant, release: release, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// The asynchronous form of ``videoBackend(repo:weightsPath:revision:cacheDirectoryURL:)``.
    /// Introduced in InferKit 0.4.0.
    @objc(videoBackendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func videoBackend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                                    completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        videoBackend(variant: .tiny, release: .sam21, repo: repo, weightsPath: weightsPath, revision: revision,
                     cacheDirectoryURL: cacheDirectoryURL, completionHandler: completionHandler)
    }

    /// Registers the video backend (`sam2-video`) with `NFKMLXModelRegistry`, delegating to
    /// ``videoBackend(weightsURL:)``. Introduced in InferKit 0.4.0.
    @objc public static func registerVideo() {
        NFKMLXModelRegistry.register(name: videoModelName) { weightsURL in try videoBackend(weightsURL: weightsURL) }
    }
}
