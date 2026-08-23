//
//  NFKMLXPose.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// Top-down human pose estimation locates a person's joints. Following SimpleBaseline (Xiao et al.), a
// residual backbone downsamples a cropped person to a small feature map, and a stack of transposed
// convolutions expands it back to a set of joint heatmaps — one channel per joint. Each joint's
// location is the argmax of its heatmap, and its confidence is the peak value. The result is an
// NSArray<NFKKeypoint *> under NFKOutputPose.
//
// The backbone is `NFKMLXResNetBackbone` in the reference ResNet-50 form, so a released SimpleBaseline
// checkpoint maps onto it. Three transposed convolutions take its stride-32 features back to stride 4.
// The caller crops and centers the person before estimation. Tensors flow in NHWC.

/// Pose dimensions. Defaults are the released 256×192, 17-joint (COCO) model; `tiny` keeps tests fast.
public struct NFKMLXPoseConfiguration: Sendable {
    public var backbone: NFKMLXResNetConfiguration
    /// The trained input geometry. A person crop is taller than it is wide, so the reference is 256×192.
    public var inputHeight: Int
    public var inputWidth: Int
    public var deconvChannels: Int
    public var deconvCount: Int
    public var keypointCount: Int

    public init(backbone: NFKMLXResNetConfiguration = NFKMLXResNetConfiguration(),
                inputHeight: Int = 256, inputWidth: Int = 192, deconvChannels: Int = 256,
                deconvCount: Int = 3, keypointCount: Int = 17) {
        self.backbone = backbone
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        self.deconvChannels = deconvChannels
        self.deconvCount = deconvCount
        self.keypointCount = keypointCount
    }

    public static let simpleBaseline = NFKMLXPoseConfiguration()

    public static let tiny = NFKMLXPoseConfiguration(
        backbone: NFKMLXResNetConfiguration(blocks: [1, 1, 1, 1], width: 8),
        inputHeight: 64, inputWidth: 32, deconvChannels: 16, keypointCount: 4)
}

/// The SimpleBaseline pose network: a residual backbone and a transposed-convolution heatmap head.
final class NFKMLXPoseNet: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKMLXResNetBackbone
    @ModuleInfo(key: "deconv") var deconv: [ConvTransposed2d]
    @ModuleInfo(key: "deconv_bn") var deconvBN: [BatchNorm]
    @ModuleInfo(key: "final") var finalConv: Conv2d

    let configuration: NFKMLXPoseConfiguration

    init(_ c: NFKMLXPoseConfiguration) {
        configuration = c
        _backbone.wrappedValue = NFKMLXResNetBackbone(c.backbone)

        // Each transposed convolution doubles the resolution, so three take the stride-32 feature back
        // to stride 4 (one channel per joint at the end).
        var inChannels = c.backbone.outputChannels
        var deconvs: [ConvTransposed2d] = []
        var deconvBNs: [BatchNorm] = []
        for _ in 0 ..< c.deconvCount {
            deconvs.append(ConvTransposed2d(inputChannels: inChannels, outputChannels: c.deconvChannels,
                                            kernelSize: 4, stride: 2, padding: 1, bias: false))
            deconvBNs.append(BatchNorm(featureCount: c.deconvChannels))
            inChannels = c.deconvChannels
        }
        _deconv.wrappedValue = deconvs
        _deconvBN.wrappedValue = deconvBNs
        _finalConv.wrappedValue = Conv2d(inputChannels: c.deconvChannels, outputChannels: c.keypointCount, kernelSize: 1)
    }

    /// Produces joint heatmaps `[1, H/4, W/4, keypointCount]` from a normalized image `[1, H, W, 3]`.
    func heatmaps(_ image: MLXArray) -> MLXArray {
        var x = backbone(image)                                // stride 32
        for (transpose, norm) in zip(deconv, deconvBN) {
            x = relu(norm(transpose(x)))                       // ×2 each, back to stride 4
        }
        return finalConv(x)
    }

    /// The ImageNet statistics the backbone is trained on.
    static func normalized(_ image: MLXArray) -> MLXArray {
        (image - MLXArray([Float(0.485), 0.456, 0.406])) / MLXArray([Float(0.229), 0.224, 0.225])
    }

    /// Estimates the pose in a bridged image `[H, W, 3]` (`0...1`): the image resizes to the model
    /// geometry, heatmaps are decoded to normalized joint positions, and `jointNames` names them.
    func estimate(_ image: MLXArray, jointNames: [String]?) -> [NFKKeypoint] {
        let batched = Self.normalized(image).reshaped([1, image.shape[0], image.shape[1], image.shape[2]])
        let resized = NFKMLXResample.resizeBilinear(batched, height: configuration.inputHeight,
                                                    width: configuration.inputWidth)
        let maps = heatmaps(resized)
        eval(maps)

        let (h, w, k) = (maps.shape[1], maps.shape[2], maps.shape[3])
        let values = maps.asArray(Float.self)
        var keypoints: [NFKKeypoint] = []
        for joint in 0 ..< k {
            var bestValue = -Float.greatestFiniteMagnitude
            var bestRow = 0
            var bestCol = 0
            for row in 0 ..< h {
                for col in 0 ..< w {
                    let value = values[(row * w + col) * k + joint]
                    if value > bestValue {
                        bestValue = value
                        bestRow = row
                        bestCol = col
                    }
                }
            }
            // A heatmap cell is four input pixels wide, so the reference refines the peak by a quarter
            // cell toward its larger neighbor. It leaves a peak on the border alone, having no pair to
            // compare.
            var (shiftX, shiftY) = (CGFloat(0), CGFloat(0))
            func value(_ row: Int, _ col: Int) -> Float { values[(row * w + col) * k + joint] }
            func quarter(_ difference: Float) -> CGFloat { difference == 0 ? 0 : (difference > 0 ? 0.25 : -0.25) }
            if bestCol > 1, bestCol < w - 1, bestRow > 1, bestRow < h - 1 {
                shiftX = quarter(value(bestRow, bestCol + 1) - value(bestRow, bestCol - 1))
                shiftY = quarter(value(bestRow + 1, bestCol) - value(bestRow - 1, bestCol))
            }
            let position = CGPoint(x: (CGFloat(bestCol) + shiftX) / CGFloat(w),
                                   y: (CGFloat(bestRow) + shiftY) / CGFloat(h))
            let name = jointNames.flatMap { joint < $0.count ? $0[joint] : nil }
            keypoints.append(NFKKeypoint(name: name, index: joint, position: position,
                                         confidence: Double(min(max(bestValue, 0), 1))))
        }
        return keypoints
    }
}

/// Holds the network and joint names for capture in the backend's `@Sendable` closure.
private final class NFKPoseHolder: @unchecked Sendable {
    let net: NFKMLXPoseNet
    let jointNames: [String]?
    init(_ net: NFKMLXPoseNet, jointNames: [String]?) {
        self.net = net
        self.jointNames = jointNames
    }
}

/// A pose-estimation backend: an image under `NFKInputImage` produces keypoints under `NFKOutputPose`.
@objc(NFKMLXPoseBackend)
public final class NFKMLXPoseBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKPoseHolder
    private let identifier: String

    init(net: NFKMLXPoseNet, identifier: String, jointNames: [String]?) {
        holder = NFKPoseHolder(net, jointNames: jointNames)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
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
        let holder = self.holder
        Task.detached {
            do {
                guard let value = request.input(forKey: NFKInputImage) else {
                    throw NFKMLXError.unsupportedInput
                }
                let image = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
                let pose = holder.net.estimate(image, jointNames: holder.jointNames)
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputPose: pose]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

/// SimpleBaseline pose estimation as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXPoseNet` is the real heatmap-based pose model. Random weights run (proving the pipeline); a
/// trained checkpoint locates joints accurately. Load a **safetensors** checkpoint; the loader
/// transposes 4-D convolution weights `[out, in, kH, kW]` to MLX's `[out, kH, kW, in]` — the converter
/// swaps the ConvTranspose axes into that same on-disk layout first.
@objc(NFKMLXPose)
public final class NFKMLXPose: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "pose-simplebaseline"

    /// Builds a pose backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). `jointNames` names the joints when
    /// available. Run inference off the render thread.
    @objc(backendWithWeightsURL:jointNames:error:)
    public static func backend(weightsURL: URL?, jointNames: [String]?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXPoseBackend(net: net, identifier: modelName, jointNames: jointNames)
    }

    static func makeNet(_ configuration: NFKMLXPoseConfiguration = .simpleBaseline) -> NFKMLXPoseNet {
        let net = NFKMLXPoseNet(configuration)
        net.train(false)                                       // BatchNorm running statistics
        return net
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:jointNames:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, jointNames: [String]?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url, jointNames: jointNames)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:jointNames:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, jointNames: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0, jointNames: jointNames) },
                               completionHandler: completionHandler)
    }

    /// Registers pose estimation (`pose-simplebaseline`) with `NFKMLXModelRegistry`, delegating to
    /// `backend(weightsURL:jointNames:)`. The registered backend has no joint names; a caller that wants
    /// names builds through the factory.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL, jointNames: nil)
        }
    }

    /// The reference builds its head from an `nn.Sequential` of (transposed convolution, normalization,
    /// activation) triples, so its keys are positional: `deconv_layers.0`, `.3`, `.6` are the
    /// convolutions and `.1`, `.4`, `.7` their normalizations. The released model also prefixes the
    /// backbone and the head, and carries the input statistics as buffers the module applies itself.
    static func remapReferenceKey(_ key: String) -> String {
        if key.hasPrefix("data_preprocessor.") {
            return key
        }
        var key = key
        if key.hasPrefix("backbone.") {
            return "backbone." + NFKMLXResNetBackbone.remapReferenceKey(String(key.dropFirst("backbone.".count)))
        }
        if key.hasPrefix("head.") {
            key = String(key.dropFirst("head.".count))
        }
        guard key.hasPrefix("deconv_layers.") else {
            return key.hasPrefix("final_layer.") ? "final." + key.dropFirst("final_layer.".count) : key
        }
        let parts = key.split(separator: ".").map(String.init)
        guard let slot = Int(parts[1]) else { return key }
        let tail = parts[2...].joined(separator: ".")
        return slot % 3 == 0 ? "deconv.\(slot / 3).\(tail)" : "deconv_bn.\(slot / 3).\(tail)"
    }

    /// Loads a safetensors checkpoint into `net`, transposing 4-D convolution weights from the on-disk
    /// `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXPoseNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remapReferenceKey(key), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
