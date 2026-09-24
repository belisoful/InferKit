import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// The weight loader, the V-JEPA 2 video processor, mean-pooled feature extraction, an @objc directory
// factory, and an NFKInferenceBackend that embeds an image or a video clip.

extension NFKMLXVJEPA2Net {
    /// Loads the released `model.safetensors`, transposing the 5-D tubelet convolution weight from
    /// PyTorch's `[out, in, kT, kH, kW]` to MLX's channels-last `[out, kT, kH, kW, in]` and dropping the
    /// pretraining predictor. The module keys mirror an encoder release's `encoder.*`; a classification
    /// release nests the encoder under `vjepa2.` beside its `pooler.*` and `classifier.*`.
    public func loadWeights(fromDirectory directory: URL) throws {
        try loadWeights(fromDirectory: directory, leavingFresh: [])
    }

    /// Loads every parameter but those under `fresh` (module-name prefixes such as `classifier.`), which
    /// keep their initialization; the checkpoint's tensors under them are skipped. Every other parameter
    /// must be supplied.
    func loadWeights(fromDirectory directory: URL, leavingFresh fresh: [String]) throws {
        let url = directory.appendingPathComponent("model.safetensors")
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            let key = key.hasPrefix("vjepa2.") ? String(key.dropFirst("vjepa2.".count)) : key
            if key.hasPrefix("predictor.") { return nil }
            if key.hasSuffix("num_batches_tracked") { return nil }
            if fresh.contains(where: { key.hasPrefix($0) }) { return nil }
            if checkpoint.needsConvTranspose && value.ndim == 5 {
                return (key, value.transposed(0, 2, 3, 4, 1))
            }
            return (key, value)
        }
        guard !fresh.isEmpty else {
            try NFKMLXWeights.apply(mapped, to: self)
            return
        }
        let supplied = Set(mapped.map(\.0))
        let missing = parameters().flattened().map(\.0).filter { name in
            !supplied.contains(name) && !fresh.contains { name.hasPrefix($0) }
        }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch("V-JEPA 2 checkpoint lacks \(missing.count) parameters, "
                                              + "starting with \(missing.prefix(3).joined(separator: ", "))")
        }
        let owned = Set(parameters().flattened().map(\.0))
        try NFKMLXWeights.apply(mapped.filter { owned.contains($0.0) }, to: self, strict: false)
    }
}

/// The V-JEPA 2 video processor: resize the shortest edge to the release's `shortest_edge` (292 for a
/// 256 crop, 438 for 384), center-crop to `cropSize`, rescale to `0…1`, and apply the ImageNet
/// normalization. A clip is assembled as `[1, frames, cropSize, cropSize, 3]` in the NDHWC layout the
/// encoder's 3D convolution consumes.
public enum NFKMLXVJEPA2Processor {
    static let imageNetMean: [Float] = [0.485, 0.456, 0.406]
    static let imageNetStd: [Float] = [0.229, 0.224, 0.225]

    /// Resizes a single `[H, W, 3]` frame (already in `0…1`) to a `cropSize × cropSize` center crop and
    /// normalizes it. The shortest edge is scaled to `shortestEdge`, preserving aspect, then the center
    /// `cropSize` square is taken.
    static func normalizedFrame(_ frame: MLXArray, cropSize: Int, shortestEdge: Int) -> MLXArray {
        let (height, width) = (frame.dim(0), frame.dim(1))
        let scale = Double(shortestEdge) / Double(min(height, width))
        let newHeight = max(cropSize, Int((Double(height) * scale).rounded()))
        let newWidth = max(cropSize, Int((Double(width) * scale).rounded()))
        let batched = frame.reshaped([1, height, width, 3])
        let resized = NFKMLXResample.resizeBilinear(batched, height: newHeight, width: newWidth)[0]
        let top = (newHeight - cropSize) / 2
        let left = (newWidth - cropSize) / 2
        let cropped = resized[top ..< (top + cropSize), left ..< (left + cropSize), 0...]
        let mean = MLXArray(imageNetMean).reshaped([1, 1, 3])
        let std = MLXArray(imageNetStd).reshaped([1, 1, 3])
        return (cropped - mean) / std
    }

    /// Assembles a normalized clip `[1, frames, cropSize, cropSize, 3]` from raw `[H, W, 3]` frames in
    /// `0…1`. Frames are sampled uniformly to `frameCount`; a clip shorter than the tubelet is padded by
    /// repeating its last frame, matching the reference's frame duplication for still images.
    public static func clip(from frames: [MLXArray], frameCount: Int, cropSize: Int, tubeletSize: Int,
                            shortestEdge: Int? = nil) -> MLXArray {
        precondition(!frames.isEmpty, "a clip needs at least one frame")
        let indices = sampleIndices(available: frames.count, want: frameCount, tubeletSize: tubeletSize)
        let edge = shortestEdge ?? cropSize * 256 / 224
        let normalized = indices.map { normalizedFrame(frames[$0], cropSize: cropSize, shortestEdge: edge) }
        let stacked = MLX.stacked(normalized, axis: 0)                       // [T, cropSize, cropSize, 3]
        return stacked.reshaped([1, normalized.count, cropSize, cropSize, 3])
    }

    /// Uniformly spaced frame indices. A single frame (a still image) is repeated to the tubelet size so
    /// the 3D convolution has a full tubelet; otherwise `want` indices span the available frames.
    static func sampleIndices(available: Int, want: Int, tubeletSize: Int) -> [Int] {
        if available == 1 {
            return Array(repeating: 0, count: max(tubeletSize, 1))
        }
        let count = min(want, max(available - (available % tubeletSize), tubeletSize))
        if count >= available {
            var indices = Array(0 ..< available)
            while indices.count % tubeletSize != 0 { indices.append(available - 1) }
            return indices
        }
        return (0 ..< count).map { Int((Double($0) * Double(available - 1) / Double(count - 1)).rounded()) }
    }
}

extension NFKMLXVJEPA2Net {
    /// A classification release's classes for `clip`, most confident first: the softmax of the class
    /// logits, labeled from the release's `id2label`. Empty for an encoder release.
    public func classifications(_ clip: MLXArray) -> [NFKClassification] {
        classifications(features: self(clip))
    }

    /// The ranked classes from already-computed encoder features `[1, tokens, hidden]`.
    func classifications(features: MLXArray) -> [NFKClassification] {
        guard let pooled = pooled(features), let classifier else { return [] }
        let probabilities = softmax(classifier(pooled)[0], axis: -1)
        eval(probabilities)
        return probabilities.asArray(Float.self).enumerated()
            .sorted { $0.element > $1.element }
            .map { NFKClassification(label: configuration.labels[$0.offset], classIndex: $0.offset,
                                     confidence: Double($0.element)) }
    }

    /// The clip-level embedding: the encoder's per-token features mean-pooled over the token axis, the
    /// pooling a retrieval or classification consumer applies to `get_vision_features`.
    public func embedding(_ clip: MLXArray) -> [Float] {
        embedding(features: self(clip))
    }

    /// The mean-pooled embedding of already-computed encoder features `[1, tokens, hidden]`.
    func embedding(features: MLXArray) -> [Float] {
        let pooled = features.mean(axis: 1)[0]                               // [hidden]
        eval(pooled)
        return pooled.asArray(Float.self)
    }
}

// MARK: - Factory

/// V-JEPA 2 (`facebook/vjepa2-*`, MIT), a self-supervised video ViT ported into `MLXNN` at reference
/// parity: the ViT-L, ViT-H, and ViT-g encoders, and the video-classification releases, whose attentive
/// pooler and classifier rank Something-Something v2 or Diving48 classes. Produces a mean-pooled feature
/// embedding from a video or image, and ranked classes from a classification release.
@objc(NFKMLXVJEPA2)
public final class NFKMLXVJEPA2: NSObject {
    @objc public static let modelName = "vjepa2-vitl-fpc64-256"
    static let requiredFiles = ["config.json"]
    static let optionalFiles = ["video_preprocessor_config.json"]
    static let weightFiles = ["model.safetensors"]

    /// Builds a backend from a released V-JEPA 2 directory (`model.safetensors` + `config.json`, and
    /// `video_preprocessor_config.json` when the release ships one). Run inference off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> NFKMLXVJEPA2Backend {
        let net = try NFKMLXVJEPA2Net(configurationURL: directoryURL.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: directoryURL)
        return NFKMLXVJEPA2Backend(net: net, configuration: net.configuration)
    }

    /// The asynchronous form of the directory factory. Blocking work runs at user-initiated quality of
    /// service off the calling thread.
    @objc(backendWithDirectoryURL:completionHandler:)
    public static func backend(directoryURL: URL,
                               completionHandler: @escaping (NFKMLXVJEPA2Backend?, Error?) -> Void) {
        Task.detached(priority: .userInitiated) {
            do { completionHandler(try backend(directoryURL: directoryURL), nil) }
            catch { completionHandler(nil, error) }
        }
    }

    /// Downloads a release into the hub cache and builds the backend.
    ///
    /// @discussion The download fetches `config.json`, `video_preprocessor_config.json`, and
    /// `model.safetensors`; the repo's `original/model.pth` is not fetched. A file the cache already
    /// holds is not fetched again. The call blocks on the network; call it off the render thread. The
    /// public releases are the encoders `facebook/vjepa2-vitl-fpc64-256`, `-vith-fpc64-256`,
    /// `-vitg-fpc64-256`, and `-vitg-fpc64-384`, and the classifiers `-vitl-fpc16-256-ssv2`,
    /// `-vitl-fpc32-256-diving48`, `-vitg-fpc64-384-ssv2`, and `-vitg-fpc32-384-diving48`; none is gated.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXVJEPA2Backend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``. The download and the
    /// build run at user-initiated quality of service off the calling thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXVJEPA2Backend?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }
}

// MARK: - Backend

/// The V-JEPA 2 inference backend: a video (`NFKInputVideo`) or an image (`NFKInputImage`) in, a
/// mean-pooled feature embedding (`NFKOutputEmbedding`, `[NSNumber]`) out, and from a classification
/// release the ranked classes (`NFKOutputClassifications`, `NSArray<NFKClassification *>`) too.
public final class NFKMLXVJEPA2Backend: NSObject, NFKInferenceBackend {
    private let holder: Holder
    private let frameCount: Int
    private let cropSize: Int
    private let tubeletSize: Int
    private let shortestEdge: Int

    final class Holder: @unchecked Sendable {
        let net: NFKMLXVJEPA2Net
        init(_ net: NFKMLXVJEPA2Net) { self.net = net }
    }

    /// The number of frames sampled from a video clip. Defaults to the model's trained clip length.
    @objc public var frameLimit: Int = 900

    init(net: NFKMLXVJEPA2Net, configuration: NFKMLXVJEPA2Configuration) {
        self.holder = Holder(net)
        self.frameCount = configuration.framesPerClip
        self.cropSize = configuration.cropSize
        self.tubeletSize = configuration.tubeletSize
        self.shortestEdge = configuration.shortestEdge
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { NFKMLXVJEPA2.modelName }
    @objc public var supportedParameterKeys: Set<String> { [] }
    @objc public var supportedInputKeys: Set<String> { [NFKInputVideo, NFKInputImage] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result { return result }
        if let error = job.error { throw error }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = self.holder
        let frameCount = self.frameCount
        let cropSize = self.cropSize
        let tubeletSize = self.tubeletSize
        let shortestEdge = self.shortestEdge
        let frameLimit = self.frameLimit
        Task.detached(priority: .userInitiated) {
            do {
                let frames = try Self.frames(from: request, frameLimit: frameLimit)
                let clip = NFKMLXVJEPA2Processor.clip(from: frames, frameCount: frameCount, cropSize: cropSize,
                                                      tubeletSize: tubeletSize, shortestEdge: shortestEdge)
                let features = holder.net(clip)
                var outputs: [String: Any] = [NFKOutputEmbedding: holder.net.embedding(features: features).map { NSNumber(value: $0) }]
                if holder.net.classifies {
                    outputs[NFKOutputClassifications] = holder.net.classifications(features: features)
                }
                job.finish(with: NFKInferenceResult(outputs: outputs))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    /// Decodes the request's video into `[H, W, 3]` frames in `0…1`, or bridges a single image into a
    /// one-frame clip.
    private static func frames(from request: NFKInferenceRequest, frameLimit: Int) throws -> [MLXArray] {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        if let asset = request.input(forKey: NFKInputVideo) as? NFKVideoAsset, let url = asset.fileURL {
            let clip = try NFKMLXVideoFile.read(url, frameLimit: frameLimit)
            return clip.frames.map { frame in
                let (bytes, width, height) = NFKMLXImageBridge.rgbaBytes(from: frame, colorSpace: colorSpace)
                return NFKMLXImageBridge.tensor(rgba: bytes, width: width, height: height, channels: 3)
            }
        }
        if let value = request.input(forKey: NFKInputImage) {
            return [try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: colorSpace)]
        }
        throw NFKMLXError.unsupportedInput
    }
}
