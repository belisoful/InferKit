import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// The weight loader, the DETR image processor, the DETR post-processing, an @objc directory factory,
// and an NFKInferenceBackend that reads one table image and emits the detected structure.

extension NFKMLXTableTransformerNet {
    /// Loads the released `model.safetensors`, transposing 4-D convolution weights from PyTorch's
    /// `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]` and dropping the batch counters.
    /// The module keys mirror a timm-backbone checkpoint (`model.*`, `class_labels_classifier.*`,
    /// `bbox_predictor.*`); the v1.1 releases' transformers ResNet names map onto them.
    public func loadWeights(fromDirectory directory: URL) throws {
        try loadWeights(fromDirectory: directory, skippingClassifier: false)
    }

    /// Loads every parameter, or every one but the class head's, which then keeps its fresh
    /// initialization for a new class set.
    func loadWeights(fromDirectory directory: URL, skippingClassifier: Bool) throws {
        let url = directory.appendingPathComponent("model.safetensors")
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            if key.hasSuffix("num_batches_tracked") { return nil }
            if skippingClassifier && key.hasPrefix("class_labels_classifier.") { return nil }
            return (Self.timmBackboneKey(key),
                    checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        guard skippingClassifier else {
            try NFKMLXWeights.apply(mapped, to: self)
            return
        }
        let supplied = Set(mapped.map(\.0))
        let missing = parameters().flattened().map(\.0).filter {
            !supplied.contains($0) && !$0.hasPrefix("class_labels_classifier.")
        }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch("Table Transformer checkpoint lacks \(missing.count) parameters, "
                                              + "starting with \(missing.prefix(3).joined(separator: ", "))")
        }
        try NFKMLXWeights.apply(mapped, to: self, strict: false)
    }

    /// The timm ResNet name of a transformers `ResNetBackbone` parameter (`use_timm_backbone` false):
    /// the stem's `embedder.embedder` is `conv1` / `bn1`, stage `s` layer `b` is `layer{s+1}.{b}`, its
    /// `layer.0` / `layer.1` are `conv1` + `bn1` / `conv2` + `bn2`, and `shortcut` is `downsample`. A timm
    /// key passes through unchanged.
    static func timmBackboneKey(_ key: String) -> String {
        let prefix = "model.backbone.conv_encoder.model."
        guard key.hasPrefix(prefix) else { return key }
        var parts = key.dropFirst(prefix.count).split(separator: ".").map(String.init)
        if parts.starts(with: ["embedder", "embedder"]), parts.count >= 3 {
            let module = parts[2] == "convolution" ? "conv1" : "bn1"
            return prefix + ([module] + parts.dropFirst(3)).joined(separator: ".")
        }
        guard parts.starts(with: ["encoder", "stages"]), parts.count >= 7, parts[3] == "layers",
              let stage = Int(parts[2]) else { return key }
        let block = parts[4]
        var renamed = ["layer\(stage + 1)", block]
        switch (parts[5], parts[6]) {
        case ("layer", let index) where parts.count >= 8:
            let number = (Int(index) ?? 0) + 1
            renamed.append(parts[7] == "convolution" ? "conv\(number)" : "bn\(number)")
            parts = Array(parts.dropFirst(8))
        case ("shortcut", let module):
            renamed += ["downsample", module == "convolution" ? "0" : "1"]
            parts = Array(parts.dropFirst(7))
        default:
            return key
        }
        return prefix + (renamed + parts).joined(separator: ".")
    }
}

/// How a release sizes its input, from its `preprocessor_config.json`.
public enum NFKMLXTableTransformerSizing: Sendable, Equatable {
    /// The shortest edge becomes the first value unless the longest would pass the second (`size` and
    /// `max_size`, or `shortest_edge` and `longest_edge`).
    case shortestEdge(Int, longestEdge: Int)
    /// The longest edge becomes the value (`longest_edge` alone, the v1.1 releases).
    case longestEdge(Int)

    /// The structure-recognition release's: 800, capped at 1000.
    public static let structureRecognition = NFKMLXTableTransformerSizing.shortestEdge(800, longestEdge: 1000)

    /// Reads `preprocessor_config.json`; a release without one sizes as ``structureRecognition``.
    public static func release(at directory: URL) -> NFKMLXTableTransformerSizing {
        let url = directory.appendingPathComponent("preprocessor_config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .structureRecognition
        }
        if let size = json["size"] as? [String: Any] {
            let shortest = (size["shortest_edge"] as? NSNumber)?.intValue
            let longest = (size["longest_edge"] as? NSNumber)?.intValue
            switch (shortest, longest) {
            case let (shortest?, longest?): return .shortestEdge(shortest, longestEdge: longest)
            case let (nil, longest?): return .longestEdge(longest)
            default: return .structureRecognition
            }
        }
        guard let size = (json["size"] as? NSNumber)?.intValue else { return .structureRecognition }
        return .shortestEdge(size, longestEdge: (json["max_size"] as? NSNumber)?.intValue ?? 1333)
    }
}

/// The DETR image processor and post-processing.
public enum NFKMLXTableTransformerProcessor {
    static let imageNetMean: [Float] = [0.485, 0.456, 0.406]
    static let imageNetStd: [Float] = [0.229, 0.224, 0.225]

    /// The reference's aspect-preserving target size. Under `.shortestEdge` the shortest edge becomes
    /// the size and, if that would push the longest past the cap, both scale down so the longest is the
    /// cap (DETR's `get_size_with_aspect_ratio`). Under `.longestEdge` both edges scale by the one factor
    /// that brings the longer to the value, truncated (DETR's `get_image_size_for_max_height_width`).
    static func targetSize(height: Int, width: Int,
                           sizing: NFKMLXTableTransformerSizing = .structureRecognition) -> (height: Int, width: Int) {
        let shortestEdge: Int, longestEdge: Int
        switch sizing {
        case .longestEdge(let edge):
            let scale = min(Double(edge) / Double(height), Double(edge) / Double(width))
            return (Int(Double(height) * scale), Int(Double(width) * scale))
        case .shortestEdge(let shortest, let longest):
            (shortestEdge, longestEdge) = (shortest, longest)
        }
        var size = Double(shortestEdge)
        let minOriginal = Double(min(height, width))
        let maxOriginal = Double(max(height, width))
        var rawSize: Double?
        if maxOriginal / minOriginal * size > Double(longestEdge) {
            rawSize = Double(longestEdge) * minOriginal / maxOriginal
            size = (rawSize!).rounded()
        }
        if (height <= width && Double(height) == size) || (width <= height && Double(width) == size) {
            return (height, width)
        }
        if width < height {
            let scaled = rawSize ?? size
            return (Int((scaled * Double(height) / Double(width)).rounded()), Int(size))
        }
        let scaled = rawSize ?? size
        return (Int(size), Int((scaled * Double(width) / Double(height)).rounded()))
    }

    /// Bridges an image, resizes it the reference's way, and applies the ImageNet normalization.
    /// Returns `[1, H, W, 3]` NHWC.
    public static func pixelValues(_ value: Any,
                                   sizing: NFKMLXTableTransformerSizing = .structureRecognition) throws -> MLXArray {
        let native = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
        let (height, width) = (native.dim(0), native.dim(1))
        let target = targetSize(height: height, width: width, sizing: sizing)
        let batched = native.reshaped([1, height, width, 3])
        let resized = NFKMLXResample.resizeBilinear(batched, height: target.height, width: target.width)
        let mean = MLXArray(imageNetMean).reshaped([1, 1, 1, 3])
        let std = MLXArray(imageNetStd).reshaped([1, 1, 1, 3])
        return (resized - mean) / std
    }
}

extension NFKMLXTableTransformerNet {
    /// Detects table-structure objects in a bridged, normalized image `[1, H, W, 3]`: forward, softmax
    /// the class logits, drop the trailing no-object class, threshold each query, and convert the
    /// center-format boxes to normalized corners. DETR is a one-to-one detector, so there is no non-max
    /// suppression. The resize preserves aspect, so a normalized box maps to the original frame unchanged.
    public func detect(_ pixels: MLXArray, threshold: Float) -> [NFKDetection] {
        let detection = self(pixels)
        let probabilities = softmax(detection.logits, axis: -1)            // [Q, num_labels + 1]
        eval(probabilities, detection.boxes)
        let probabilityValues = probabilities.asArray(Float.self)
        let boxValues = detection.boxes.asArray(Float.self)
        let queries = detection.logits.dim(0)
        let classes = config.numLabels

        var results = [NFKDetection]()
        for query in 0 ..< queries {
            var bestClass = 0
            var bestScore: Float = 0
            for k in 0 ..< classes {
                let score = probabilityValues[query * (classes + 1) + k]
                if score > bestScore {
                    bestScore = score
                    bestClass = k
                }
            }
            if bestScore < threshold { continue }
            let cx = boxValues[query * 4], cy = boxValues[query * 4 + 1]
            let w = boxValues[query * 4 + 2], h = boxValues[query * 4 + 3]
            let minX = min(max(cx - w / 2, 0), 1), maxX = min(max(cx + w / 2, 0), 1)
            let minY = min(max(cy - h / 2, 0), 1), maxY = min(max(cy + h / 2, 0), 1)
            let rect = CGRect(x: CGFloat(minX), y: CGFloat(minY),
                              width: CGFloat(maxX - minX), height: CGFloat(maxY - minY))
            let label = bestClass < config.labels.count ? config.labels[bestClass] : nil
            results.append(NFKDetection(label: label, classIndex: bestClass,
                                        confidence: Double(bestScore), boundingBox: rect))
        }
        return results
    }
}

// MARK: - Factory

/// Table Transformer (`microsoft/table-transformer-*`, MIT): table detection and table-structure
/// recognition, a vanilla DETR with a ResNet-18 backbone, ported into `MLXNN` at reference parity on all
/// five releases.
@objc(NFKMLXTableTransformer)
public final class NFKMLXTableTransformer: NSObject {
    @objc public static let modelName = "table-transformer-structure-recognition"
    static let requiredFiles = ["config.json"]
    static let optionalFiles = ["preprocessor_config.json"]
    static let weightFiles = ["model.safetensors"]

    /// The detection confidence threshold. The released model is confident on a clean table crop.
    @objc public static var confidenceThreshold: Float = 0.6

    /// Builds a backend from a released Table Transformer directory (`model.safetensors` + `config.json`,
    /// and `preprocessor_config.json` when the release ships one). The directory's `config.json` supplies
    /// the geometry and the class labels, and its preprocessor configuration the input size. Run
    /// inference off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> NFKMLXTableTransformerBackend {
        let net = try NFKMLXTableTransformerNet(
            configurationURL: directoryURL.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: directoryURL)
        return NFKMLXTableTransformerBackend(net: net, sizing: .release(at: directoryURL))
    }

    /// The asynchronous form of the directory factory. Blocking work runs at user-initiated quality of
    /// service off the calling thread.
    @objc(backendWithDirectoryURL:completionHandler:)
    public static func backend(directoryURL: URL,
                               completionHandler: @escaping (NFKMLXTableTransformerBackend?, Error?) -> Void) {
        Task.detached(priority: .userInitiated) {
            do { completionHandler(try backend(directoryURL: directoryURL), nil) }
            catch { completionHandler(nil, error) }
        }
    }

    /// Downloads a release into the hub cache and builds the backend.
    ///
    /// @discussion The download fetches `config.json`, `preprocessor_config.json`, and
    /// `model.safetensors`. A file the cache already holds is not fetched again. The call blocks on the
    /// network; call it off the render thread. The public releases are
    /// `microsoft/table-transformer-detection`, `microsoft/table-transformer-structure-recognition`, and
    /// the structure-recognition v1.1 releases (`-v1.1-all`, `-v1.1-fin`, `-v1.1-pub`); none is gated.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> NFKMLXTableTransformerBackend {
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
                               completionHandler: @escaping (NFKMLXTableTransformerBackend?, Error?) -> Void) {
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

/// The Table Transformer inference backend: one image in, the detected table structure
/// (`NFKOutputDetections`) out.
public final class NFKMLXTableTransformerBackend: NSObject, NFKInferenceBackend {
    private let holder: Holder
    /// How the release sizes its input.
    public let sizing: NFKMLXTableTransformerSizing

    final class Holder: @unchecked Sendable {
        let net: NFKMLXTableTransformerNet
        init(_ net: NFKMLXTableTransformerNet) { self.net = net }
    }

    init(net: NFKMLXTableTransformerNet, sizing: NFKMLXTableTransformerSizing) {
        self.holder = Holder(net)
        self.sizing = sizing
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { NFKMLXTableTransformer.modelName }
    @objc public var supportedParameterKeys: Set<String> { [] }
    @objc public var supportedInputKeys: Set<String> { [NFKInputImage] }

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
        let sizing = self.sizing
        Task.detached(priority: .userInitiated) {
            do {
                guard let value = request.input(forKey: NFKInputImage) else {
                    throw NFKMLXError.unsupportedInput
                }
                let pixels = try NFKMLXTableTransformerProcessor.pixelValues(value, sizing: sizing)
                let detections = holder.net.detect(pixels, threshold: NFKMLXTableTransformer.confidenceThreshold)
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputDetections: detections]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}
