//
//  NFKMLXRFDetrSegmentation.swift
//  InferKit
//
//  RF-DETR instance segmentation (`RfDetrForInstanceSegmentation`): the detector this package already
//  carries, with a mask head over the projected spatial features. The head is the only new network;
//  the backbone, the encoder, and the decoder are the detector's, at a geometry the configuration
//  already expresses (patch 12, one window, four decoder layers, a hundred queries).
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

/// One block of the mask head: a depthwise convolution, a channels-last normalization, a pointwise
/// projection and the activation, added back to the input.
///
/// The activation is the EXACT error-function gelu, which is what the config's `gelu` names; the
/// tanh approximation is a different function here (`gelu_pytorch_tanh`) and would cost real digits.
///
/// The reference keeps the channels-last form because it is faster in PyTorch; the arithmetic is the
/// same either way, and MLX is channels-last already, so no permutation is needed here.
final class NFKRFDetrSegmentationBlock: Module {
    @ModuleInfo(key: "dwconv") var depthwise: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pointwise: Linear

    init(_ width: Int) {
        _depthwise.wrappedValue = Conv2d(inputChannels: width, outputChannels: width,
                                         kernelSize: 3, padding: 1, groups: width)
        _norm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
        _pointwise.wrappedValue = Linear(width, width)
        super.init()
    }

    /// - Parameter x: `[1, H, W, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + gelu(pointwise(norm(depthwise(x))))
    }
}

/// The query side of the head: a normalized feed-forward added back to the queries.
final class NFKRFDetrSegmentationMLPBlock: Module {
    @ModuleInfo(key: "norm_in") var norm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(_ width: Int, intermediateSize: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: width)
        _layers.wrappedValue = [Linear(width, intermediateSize), Linear(intermediateSize, width)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + layers[1](gelu(layers[0](norm(x))))
    }
}

/// The mask head: the spatial features are resampled to the mask resolution and then walked through
/// one block per decoder layer, and each layer's queries are projected and multiplied against that
/// block's output to give that layer's masks.
public final class NFKMLXRFDetrSegmentationHead: Module {
    @ModuleInfo(key: "blocks") var blocks: [NFKRFDetrSegmentationBlock]
    @ModuleInfo(key: "spatial_features_proj") var spatialProjection: Conv2d
    @ModuleInfo(key: "query_features_block") var queryBlock: NFKRFDetrSegmentationMLPBlock
    @ModuleInfo(key: "query_features_proj") var queryProjection: Linear
    @ParameterInfo(key: "bias") var bias: MLXArray

    let downsampleRatio: Int

    public init(width: Int, blockCount: Int, intermediateSize: Int, downsampleRatio: Int) {
        _blocks.wrappedValue = (0 ..< blockCount).map { _ in NFKRFDetrSegmentationBlock(width) }
        _spatialProjection.wrappedValue = Conv2d(inputChannels: width, outputChannels: width, kernelSize: 1)
        _queryBlock.wrappedValue = NFKRFDetrSegmentationMLPBlock(width, intermediateSize: intermediateSize)
        _queryProjection.wrappedValue = Linear(width, width)
        _bias.wrappedValue = MLXArray.zeros([1])
        self.downsampleRatio = downsampleRatio
        super.init()
    }

    /// The masks one layer's queries give over already-projected spatial features.
    ///
    /// - Parameters:
    ///   - queries: `[1, Q, C]`.
    ///   - spatial: `[1, H, W, C]`.
    /// - Returns: `[Q, H, W]`.
    func masks(queries: MLXArray, spatial: MLXArray) -> MLXArray {
        let (height, width) = (spatial.dim(1), spatial.dim(2))
        let projected = queryProjection(queryBlock(queries))                    // [1, Q, C]
        let flattened = spatial.reshaped([1, height * width, spatial.dim(3)])   // [1, HW, C]
        let logits = matmul(projected, flattened.transposed(0, 2, 1))           // [1, Q, HW]
        return logits.reshaped([-1, height, width]) + bias
    }

    /// One set of masks per decoder layer, the last being the model's own.
    ///
    /// - Parameters:
    ///   - spatial: the projected feature map, `[1, H, W, C]`.
    ///   - queries: each decoder layer's normalized output, `[1, Q, C]`.
    ///   - imageSize: the input's height and width, which the mask resolution divides.
    public func callAsFunction(spatial: MLXArray, queries: [MLXArray],
                               imageSize: (height: Int, width: Int)) -> [MLXArray] {
        var features = NFKMLXResample.resizeBilinear(
            spatial, height: imageSize.height / downsampleRatio,
            width: imageSize.width / downsampleRatio)
        var output = [MLXArray]()
        for (block, layerQueries) in zip(blocks, queries) {
            features = block(features)
            output.append(masks(queries: layerQueries, spatial: spatialProjection(features)))
        }
        return output
    }
}

/// `RfDetrForInstanceSegmentation`: the detector's own stack, with the mask head reading the
/// projector's output and every decoder layer's queries.
///
/// The key paths match the detector's, so the released checkpoint's names reach this module through
/// the same remap, and the head's own names (`segmentation_head.…`) need none.
public final class NFKMLXRFDetrSegmentationNet: Module {
    @ModuleInfo(key: "model") var model: NFKRFDetrModel
    @ModuleInfo(key: "class_embed") var classEmbed: Linear
    @ModuleInfo(key: "bbox_embed") var bboxEmbed: NFKRFDetrMLP
    @ModuleInfo(key: "segmentation_head") var segmentationHead: NFKMLXRFDetrSegmentationHead

    public let config: NFKMLXRFDetrConfiguration

    public init(_ config: NFKMLXRFDetrConfiguration) {
        self.config = config
        _model.wrappedValue = NFKRFDetrModel(config)
        _classEmbed.wrappedValue = Linear(config.dModel, config.numLabels)
        _bboxEmbed.wrappedValue = NFKRFDetrMLP(inputDim: config.dModel, hiddenDim: config.dModel,
                                               outputDim: 4, numLayers: 3)
        _segmentationHead.wrappedValue = NFKMLXRFDetrSegmentationHead(
            width: config.dModel, blockCount: config.decoderLayers,
            intermediateSize: config.segmentationIntermediateSize,
            downsampleRatio: config.maskDownsampleRatio)
        super.init()
    }

    /// The staged segmentation, for parity localization and post-processing.
    public struct Segmentation {
        /// `[Q, numLabels]`.
        public let logits: MLXArray
        /// `[Q, 4]`, centre-width-height.
        public let boxes: MLXArray
        /// One set per decoder layer, each `[Q, H / ratio, W / ratio]`. The reference's prediction is
        /// the last.
        public let masksPerLayer: [MLXArray]
        /// The reference's own masks, `[Q, H / ratio, W / ratio]`.
        public var masks: MLXArray { masksPerLayer[masksPerLayer.count - 1] }
    }

    public func callAsFunction(_ pixels: MLXArray) -> Segmentation {
        let selection = model.select(pixels)
        let states = model.decodeStates(selection)
        let last = states[config.decoderLayers - 1]
        let logits = classEmbed(last[0])
        let delta = bboxEmbed(last[0])
        let boxes = nfkRFDetrRefineBboxes(reference: selection.initReference, deltas: delta)
        // The reference's `backbone_features` is the projector's output, the tensor whose flatten
        // becomes the encoder's source, which is what `projected` holds.
        let masks = segmentationHead(spatial: selection.projected, queries: states,
                                     imageSize: (pixels.dim(1), pixels.dim(2)))
        return Segmentation(logits: logits, boxes: boxes, masksPerLayer: masks)
    }
}

extension NFKMLXRFDetrSegmentationNet {
    /// Instances in a bridged image `[H, W, 3]` (`0...1`): the detector's own post-processing, with
    /// each surviving query's mask alongside it.
    ///
    /// The preprocessing is the detector's — square resize, ImageNet normalization, sigmoid, a
    /// per-query threshold, no non-max suppression — so the boxes and labels are the detector's.
    ///
    /// - Returns: the detections, and their masks `[K, h, w]` as probabilities at the mask resolution.
    public func segment(_ image: MLXArray, labels: [String]?) -> (detections: [NFKDetection], masks: MLXArray) {
        let batched = image.ndim == 3
            ? image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]) : image
        let resized = NFKMLXResample.resizeBilinear(batched, height: config.inputResolution,
                                                    width: config.inputResolution)
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 1, 3])
        let std = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 1, 3])
        let out = self((resized - mean) / std)

        let scores = sigmoid(out.logits); eval(scores)
        let boxes = out.boxes; eval(boxes)
        let scoreValues = scores.asArray(Float.self)
        let boxValues = boxes.asArray(Float.self)
        let classes = config.numLabels

        var results = [NFKDetection]()
        var kept = [Int]()
        for query in 0 ..< out.logits.dim(0) {
            var bestClass = 0
            var bestScore: Float = 0
            for k in 0 ..< classes {
                let score = scoreValues[query * classes + k]
                if score > bestScore { bestScore = score; bestClass = k }
            }
            if bestScore < config.confidenceThreshold { continue }
            let cx = boxValues[query * 4], cy = boxValues[query * 4 + 1]
            let w = boxValues[query * 4 + 2], h = boxValues[query * 4 + 3]
            let minX = min(max(cx - w / 2, 0), 1), maxX = min(max(cx + w / 2, 0), 1)
            let minY = min(max(cy - h / 2, 0), 1), maxY = min(max(cy + h / 2, 0), 1)
            results.append(NFKDetection(
                label: labels.flatMap { bestClass < $0.count ? $0[bestClass] : nil },
                classIndex: bestClass, confidence: Double(bestScore),
                boundingBox: CGRect(x: CGFloat(minX), y: CGFloat(minY),
                                    width: CGFloat(maxX - minX), height: CGFloat(maxY - minY))))
            kept.append(query)
        }
        let masks = kept.isEmpty
            ? MLXArray.zeros([0, out.masks.dim(1), out.masks.dim(2)])
            : sigmoid(out.masks[MLXArray(kept.map { Int32($0) })])
        eval(masks)
        return (results, masks)
    }
}

/// The released RF-DETR segmentation size to build, for the Objective-C factory. A checkpoint fits
/// only its own.
@objc(NFKMLXRFDetrSegmentationVariant)
public enum NFKMLXRFDetrSegmentationVariant: Int {
    case nano
    case small
    case preview
    case medium
    case large
    case extraLarge
    case extraExtraLarge
}

/// RF-DETR instance segmentation as an InferKit backend: an image under `NFKInputImage` → the
/// instances under `NFKOutputDetections` and their combined mask under `NFKOutputMask`.
///
/// The per-instance masks stay on `NFKMLXRFDetrSegmentationNet.segment(_:labels:)`, because the mask
/// key carries one image; the backend's mask is the per-pixel maximum over the instances, which is the
/// foreground the model found.
@objc(NFKMLXRFDetrSegmentationBackend)
public final class NFKMLXRFDetrSegmentationBackend: NSObject, NFKInferenceBackend {
    private final class Holder: @unchecked Sendable {
        let net: NFKMLXRFDetrSegmentationNet
        let labels: [String]?
        init(_ net: NFKMLXRFDetrSegmentationNet, labels: [String]?) {
            self.net = net
            self.labels = labels
        }
    }

    private let holder: Holder
    private let name: String

    init(net: NFKMLXRFDetrSegmentationNet, identifier: String, labels: [String]?) {
        holder = Holder(net, labels: labels)
        name = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { name }

    /// The request inputs the backend reads.
    @objc public var supportedInputKeys: Set<String> { [NFKInputImage] }

    /// The request parameters the backend reads.
    @objc public var supportedParameterKeys: Set<String> { [] }

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
        Task.detached(priority: .userInitiated) {
            do {
                guard let value = request.input(forKey: NFKInputImage) else {
                    throw NFKMLXError.unsupportedInput
                }
                let image = try NFKMLXImageBridge.tensor(from: value, channels: 3,
                                                         colorSpace: CGColorSpaceCreateDeviceRGB())
                let (detections, masks) = holder.net.segment(image, labels: holder.labels)
                var outputs: [String: Any] = [NFKOutputDetections: detections]
                if masks.dim(0) > 0 {
                    let combined = masks.max(axis: 0).expandedDimensions(axis: -1)
                    eval(combined)
                    outputs[NFKOutputMask] = try NFKMLXImageBridge.cgImage(
                        from: combined, options: NFKMLXImageOptions())
                }
                job.finish(with: NFKInferenceResult(outputs: outputs))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

@objc(NFKMLXRFDetrSegmentation)
public final class NFKMLXRFDetrSegmentation: NSObject {
    /// The registry name the model builds under.
    @objc public static let modelName = "rf-detr-seg"

    static func specs(for variant: NFKMLXRFDetrSegmentationVariant)
        -> (name: String, configuration: NFKMLXRFDetrConfiguration) {
        switch variant {
        case .nano: return (modelName, .segNano)
        case .small: return ("rf-detr-seg-small", .segSmall)
        case .preview: return ("rf-detr-seg-preview", .segPreview)
        case .medium: return ("rf-detr-seg-medium", .segMedium)
        case .large: return ("rf-detr-seg-large", .segLarge)
        case .extraLarge: return ("rf-detr-seg-xlarge", .segXLarge)
        case .extraExtraLarge: return ("rf-detr-seg-xxlarge", .segXXLarge)
        }
    }

    /// The segmentation backend at the nano geometry. Nil weights build a randomly initialized net.
    @objc(backendWithWeightsURL:labels:error:)
    public static func backend(weightsURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        try backend(variant: .nano, weightsURL: weightsURL, labels: labels)
    }

    /// The segmentation backend at `variant`'s geometry, from a released `model.safetensors`.
    @objc(backendWithVariant:weightsURL:labels:error:)
    public static func backend(variant: NFKMLXRFDetrSegmentationVariant, weightsURL: URL?,
                               labels: [String]?) throws -> any NFKInferenceBackend {
        let spec = specs(for: variant)
        let net = NFKMLXRFDetrSegmentationNet(spec.configuration)
        if let weightsURL {
            try NFKMLXRFDetr.loadWeights(into: net, from: weightsURL)
        }
        net.train(false)
        return NFKMLXRFDetrSegmentationBackend(net: net, identifier: spec.name, labels: labels)
    }

    /// Downloads the weights and builds the nano backend. Blocking; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:labels:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        try backend(variant: .nano, repo: repo, weightsPath: weightsPath, revision: revision,
                    cacheDirectoryURL: cacheDirectoryURL, labels: labels)
    }

    /// Downloads the weights and builds `variant`'s backend. Blocking; run off the render thread.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:labels:error:)
    public static func backend(variant: NFKMLXRFDetrSegmentationVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               labels: [String]?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url, labels: labels)
    }

    /// The asynchronous peer of the download factory, at the nano geometry.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:labels:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, labels: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        backend(variant: .nano, repo: repo, weightsPath: weightsPath, revision: revision,
                cacheDirectoryURL: cacheDirectoryURL, labels: labels, completionHandler: completionHandler)
    }

    /// Registers every released size under its own name, so the registry reaches them by name.
    @objc public static func register() {
        for variant in [NFKMLXRFDetrSegmentationVariant.nano, .small, .preview, .medium, .large,
                        .extraLarge, .extraExtraLarge] {
            NFKMLXModelRegistry.register(name: specs(for: variant).name) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL, labels: nil)
            }
        }
    }

    /// The asynchronous peer of the download factory, honoring the variant.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:labels:completionHandler:)
    public static func backend(variant: NFKMLXRFDetrSegmentationVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?, labels: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        Task.detached(priority: .userInitiated) {
            do {
                completionHandler(try backend(variant: variant, repo: repo, weightsPath: weightsPath,
                                              revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                              labels: labels), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }
}
