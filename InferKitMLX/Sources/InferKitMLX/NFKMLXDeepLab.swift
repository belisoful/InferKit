//
//  NFKMLXDeepLab.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// DeepLabV3 assigns a class to every pixel with a convolutional backbone and an Atrous Spatial Pyramid
// Pooling (ASPP) head. ASPP looks at several receptive fields in parallel — a 1×1 convolution, three
// dilated 3×3 convolutions at increasing rates, and a global image-pooling branch — then fuses them, so
// context at multiple scales informs each pixel's label. The argmax over classes is a label map,
// emitted as a grayscale image whose pixel value encodes the class index (matching `NFKMLXSegFormer`).
//
// The backbone is `NFKMLXResNetBackbone` with its last two stages dilated, so the features reaching the
// head are at stride 8 rather than 32 — the substitution DeepLab is built around. Tensors flow in NHWC.

/// DeepLab dimensions. Defaults are the released model; `tiny` keeps tests fast.
public struct NFKMLXDeepLabConfiguration: Sendable {
    public var backbone: NFKMLXResNetConfiguration
    public var asppChannels: Int
    public var dilations: [Int]
    public var classCount: Int

    public init(backbone: NFKMLXResNetConfiguration = .deepLab, asppChannels: Int = 256,
                dilations: [Int] = [12, 24, 36], classCount: Int = 21) {
        self.backbone = backbone
        self.asppChannels = asppChannels
        self.dilations = dilations
        self.classCount = classCount
    }

    public static let base = NFKMLXDeepLabConfiguration()

    public static let tiny = NFKMLXDeepLabConfiguration(backbone: .tiny, asppChannels: 8,
                                                        dilations: [1, 2, 3], classCount: 4)
}

/// One ASPP branch: a convolution at the branch's dilation, normalized and activated.
final class NFKDeepLabBranch: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: BatchNorm

    init(_ inChannels: Int, _ outChannels: Int, dilation: Int?) {
        _conv.wrappedValue = dilation.map {
            Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3,
                   padding: IntOrPair($0), dilation: IntOrPair($0), bias: false)
        } ?? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
        _norm.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { relu(norm(conv(x))) }
}

/// The ASPP head: a 1×1 branch, dilated 3×3 branches, a global-pooling branch, and a fusion projection.
final class NFKDeepLabASPP: Module {
    @ModuleInfo(key: "convs") var convs: [NFKDeepLabBranch]
    @ModuleInfo(key: "pool") var pool: NFKDeepLabBranch
    @ModuleInfo(key: "project") var project: NFKDeepLabBranch

    init(inChannels: Int, outChannels: Int, dilations: [Int]) {
        var branches = [NFKDeepLabBranch(inChannels, outChannels, dilation: nil)]
        branches += dilations.map { NFKDeepLabBranch(inChannels, outChannels, dilation: $0) }
        _convs.wrappedValue = branches
        _pool.wrappedValue = NFKDeepLabBranch(inChannels, outChannels, dilation: nil)
        // The branches plus the pooled one concatenate before the projection.
        _project.wrappedValue = NFKDeepLabBranch(outChannels * (branches.count + 1), outChannels, dilation: nil)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var branches = convs.map { $0(x) }
        let pooled = pool(mean(x, axes: [1, 2], keepDims: true))    // global context
        branches.append(NFKMLXResample.resizeBilinear(pooled, height: x.shape[1], width: x.shape[2]))
        return project(concatenated(branches, axis: 3))
    }
}

/// The DeepLabV3 network: a dilated residual backbone, an ASPP head, and a per-pixel classifier.
final class NFKMLXDeepLabNet: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKMLXResNetBackbone
    @ModuleInfo(key: "aspp") var aspp: NFKDeepLabASPP
    @ModuleInfo(key: "head") var head: NFKDeepLabBranch
    @ModuleInfo(key: "classifier") var classifier: Conv2d

    let configuration: NFKMLXDeepLabConfiguration

    init(_ c: NFKMLXDeepLabConfiguration) {
        configuration = c
        _backbone.wrappedValue = NFKMLXResNetBackbone(c.backbone)
        _aspp.wrappedValue = NFKDeepLabASPP(inChannels: c.backbone.outputChannels,
                                            outChannels: c.asppChannels, dilations: c.dilations)
        // The reference follows ASPP with one 3×3 convolution before the 1×1 classifier.
        _head.wrappedValue = NFKDeepLabBranch(c.asppChannels, c.asppChannels, dilation: 1)
        _classifier.wrappedValue = Conv2d(inputChannels: c.asppChannels, outputChannels: c.classCount, kernelSize: 1)
    }

    /// Produces class logits `[1, h, w, classCount]` at the backbone's output stride (8).
    func logits(_ image: MLXArray) -> MLXArray {
        classifier(head(aspp(backbone(image))))
    }

    /// Segments a bridged image `[H, W, 3]` (`0...1`), returning a grayscale label map `[H, W, 1]` whose
    /// value is the class index scaled to `0...1`.
    func segment(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let scores = logits(Self.normalized(image).reshaped([1, height, width, 3]))
        // The reference upsamples the logits, then takes the label; taking the label first would
        // quantize before the interpolation and blur class boundaries into nonexistent classes.
        let full = NFKMLXResample.resizeBilinear(scores, height: height, width: width)
        let normalized = full.argMax(axis: -1).asType(.float32) / Float(max(configuration.classCount - 1, 1))
        return normalized.reshaped([height, width, 1])
    }

    /// The ImageNet statistics the backbone is trained on.
    static func normalized(_ image: MLXArray) -> MLXArray {
        (image - MLXArray([Float(0.485), 0.456, 0.406])) / MLXArray([Float(0.229), 0.224, 0.225])
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKDeepLabHolder: @unchecked Sendable {
    let net: NFKMLXDeepLabNet
    init(_ net: NFKMLXDeepLabNet) { self.net = net }
}

/// DeepLabV3 semantic segmentation as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXDeepLabNet` is the real ASPP-based segmenter. Random weights run (proving the pipeline); a
/// trained checkpoint segments accurately. The output is a grayscale label map under `NFKOutputImage`;
/// recover the class index as `round(gray·(classCount−1))`. Load a **safetensors** checkpoint; the
/// loader transposes 4-D convolution weights `[out, in, kH, kW]` to MLX's `[out, kH, kW, in]`.
@objc(NFKMLXDeepLab)
public final class NFKMLXDeepLab: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "deeplabv3"

    static func makeNet(_ configuration: NFKMLXDeepLabConfiguration = .base) -> NFKMLXDeepLabNet {
        let net = NFKMLXDeepLabNet(configuration)
        net.train(false)                                       // BatchNorm running statistics
        return net
    }

    /// Builds a segmentation backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKDeepLabHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in holder.net.segment(image) }
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers DeepLabV3 (`deeplabv3`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// The reference builds its head from `nn.Sequential`s, so its keys are positional: `classifier.0`
    /// is ASPP (whose branches nest a convolution at `.0` and a normalization at `.1`, the pooling
    /// branch one slot later because the pool itself occupies `.0`), `classifier.1`/`.2` are the 3×3
    /// convolution and its normalization, and `classifier.4` is the classifier. Map those onto the
    /// module's names. The auxiliary classifier the checkpoint carries is a training aid and is ignored.
    static func remapReferenceKey(_ key: String, poolBranch: Int) -> String {
        if key.hasPrefix("backbone.") {
            return "backbone." + NFKMLXResNetBackbone.remapReferenceKey(String(key.dropFirst("backbone.".count)))
        }
        guard key.hasPrefix("classifier.") else { return key }
        let parts = key.split(separator: ".").map(String.init)
        let tail = { (dropped: Int) in parts[dropped...].joined(separator: ".") }
        switch parts[1] {
        case "0" where parts.count >= 5 && parts[2] == "convs":
            // The pooling branch opens with the pool itself, so its convolution and normalization sit
            // one slot later than every other branch's.
            let branch = Int(parts[3]) ?? 0
            let convSlot = branch == poolBranch ? "1" : "0"
            let name = parts[4] == convSlot ? "conv" : "norm"
            return branch == poolBranch ? "aspp.pool.\(name).\(tail(5))"
                                        : "aspp.convs.\(branch).\(name).\(tail(5))"
        case "0" where parts.count >= 4 && parts[2] == "project":
            return "aspp.project.\(parts[3] == "0" ? "conv" : "norm").\(tail(4))"
        case "1": return "head.conv.\(tail(2))"
        case "2": return "head.norm.\(tail(2))"
        case "4": return "classifier.\(tail(2))"
        default: return key
        }
    }

    /// Loads a safetensors checkpoint into `net`, transposing 4-D convolution weights from PyTorch's
    /// `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXDeepLabNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let poolBranch = net.configuration.dilations.count + 1
        let mapped = raw.map { key, value in
            (remapReferenceKey(key, poolBranch: poolBranch),
             checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
