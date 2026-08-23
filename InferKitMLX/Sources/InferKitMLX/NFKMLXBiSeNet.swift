//
//  NFKMLXBiSeNet.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// BiSeNet segments in real time by splitting the work into two paths. A shallow Spatial Path keeps
// high-resolution detail with a few strided convolutions; a deep Context Path downsamples quickly and
// gathers scene context, refining each stage with an Attention Refinement Module (a global-pooled gate)
// and a global-context branch. A Feature Fusion Module combines the two, and a classifier produces the
// label map, emitted as a grayscale image (matching `NFKMLXSegFormer` / `NFKMLXDeepLab`).
//
// The backbone is a compact strided-convolution network standing in for the reference ResNet/Xception;
// matching the exact keys is a validation-sweep item. The two-path structure, attention refinement, and
// fusion are the real pipeline. Tensors flow in NHWC.

/// BiSeNet dimensions. Defaults size a compact model; `tiny` keeps tests fast.
public struct NFKMLXBiSeNetConfiguration: Sendable {
    public var baseChannels: Int
    public var classCount: Int

    /// The context path's ResNet-18 stem width; the released model is 64.
    public init(baseChannels: Int = 64, classCount: Int = 19) {
        self.baseChannels = baseChannels
        self.classCount = classCount
    }

    public static let base = NFKMLXBiSeNetConfiguration()

    public static let tiny = NFKMLXBiSeNetConfiguration(baseChannels: 8, classCount: 4)
}

/// A convolution → batch norm → ReLU block with an optional stride.
/// Convolution + batch normalization + ReLU, the reference's `ConvBNReLU`.
final class NFKBiSeNetConvBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm

    init(inChannels: Int, outChannels: Int, kernel: Int = 3, stride: Int = 1, padding: Int? = nil) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                    padding: IntOrPair(padding ?? kernel / 2), bias: false)
        _bn.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { relu(bn(conv(x))) }
}

/// A ResNet-18 basic block: two 3×3 convolutions with an identity that projects when the shape
/// changes. The reference keeps the projection in a `Sequential`, so its keys are `downsample.0/1`.
final class NFKBiSeNetBasicBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm
    @ModuleInfo(key: "downsample_conv") var downsampleConv: Conv2d?
    @ModuleInfo(key: "downsample_bn") var downsampleBN: BatchNorm?

    init(inChannels: Int, outChannels: Int, stride: Int) {
        _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3,
                                     stride: IntOrPair(stride), padding: 1, bias: false)
        _bn1.wrappedValue = BatchNorm(featureCount: outChannels)
        _conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3,
                                     padding: 1, bias: false)
        _bn2.wrappedValue = BatchNorm(featureCount: outChannels)
        if stride != 1 || inChannels != outChannels {
            _downsampleConv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                                  kernelSize: 1, stride: IntOrPair(stride), bias: false)
            _downsampleBN.wrappedValue = BatchNorm(featureCount: outChannels)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        var identity = x
        if let downsampleConv, let downsampleBN {
            identity = downsampleBN(downsampleConv(x))
        }
        return relu(out + identity)
    }
}

/// The ResNet-18 the context path runs on, emitting the features at strides 8, 16, and 32.
final class NFKBiSeNetResNet18: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "layer1") var layer1: [NFKBiSeNetBasicBlock]
    @ModuleInfo(key: "layer2") var layer2: [NFKBiSeNetBasicBlock]
    @ModuleInfo(key: "layer3") var layer3: [NFKBiSeNetBasicBlock]
    @ModuleInfo(key: "layer4") var layer4: [NFKBiSeNetBasicBlock]

    init(width: Int = 64) {
        _conv1.wrappedValue = Conv2d(inputChannels: 3, outputChannels: width, kernelSize: 7, stride: 2,
                                     padding: 3, bias: false)
        _bn1.wrappedValue = BatchNorm(featureCount: width)
        func stage(_ inChannels: Int, _ outChannels: Int, stride: Int) -> [NFKBiSeNetBasicBlock] {
            [NFKBiSeNetBasicBlock(inChannels: inChannels, outChannels: outChannels, stride: stride),
             NFKBiSeNetBasicBlock(inChannels: outChannels, outChannels: outChannels, stride: 1)]
        }
        _layer1.wrappedValue = stage(width, width, stride: 1)
        _layer2.wrappedValue = stage(width, width * 2, stride: 2)
        _layer3.wrappedValue = stage(width * 2, width * 4, stride: 2)
        _layer4.wrappedValue = stage(width * 4, width * 8, stride: 2)
    }

    func callAsFunction(_ x: MLXArray) -> (feat8: MLXArray, feat16: MLXArray, feat32: MLXArray) {
        var out = NFKMLXResample.maxPooled(relu(bn1(conv1(x))), kernel: 3, stride: 2, padding: 1)
        for block in layer1 { out = block(out) }
        for block in layer2 { out = block(out) }
        let feat8 = out
        for block in layer3 { out = block(out) }
        let feat16 = out
        for block in layer4 { out = block(out) }
        return (feat8, feat16, out)
    }
}

/// The attention refinement module: a convolution whose channels are gated by a globally pooled,
/// normalized attention vector.
final class NFKBiSeNetARM: Module {
    @ModuleInfo(key: "conv") var conv: NFKBiSeNetConvBlock
    @ModuleInfo(key: "conv_atten") var convAtten: Conv2d
    @ModuleInfo(key: "bn_atten") var bnAtten: BatchNorm

    init(inChannels: Int, outChannels: Int) {
        _conv.wrappedValue = NFKBiSeNetConvBlock(inChannels: inChannels, outChannels: outChannels, kernel: 3)
        _convAtten.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels,
                                         kernelSize: 1, bias: false)
        _bnAtten.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let feature = conv(x)
        let attention = sigmoid(bnAtten(convAtten(mean(feature, axes: [1, 2], keepDims: true))))
        return feature * attention
    }
}

/// The deep context path: ResNet-18, attention refinement at strides 16 and 32, and a global-context
/// branch. The reference upsamples here with plain `nn.Upsample(scale_factor: 2)`, whose default mode
/// is **nearest** — the output head is the bilinear one.
final class NFKBiSeNetContextPath: Module {
    @ModuleInfo(key: "resnet") var resnet: NFKBiSeNetResNet18
    @ModuleInfo(key: "arm16") var arm16: NFKBiSeNetARM
    @ModuleInfo(key: "arm32") var arm32: NFKBiSeNetARM
    @ModuleInfo(key: "conv_head32") var convHead32: NFKBiSeNetConvBlock
    @ModuleInfo(key: "conv_head16") var convHead16: NFKBiSeNetConvBlock
    @ModuleInfo(key: "conv_avg") var convAvg: NFKBiSeNetConvBlock

    init(width: Int = 64, contextChannels: Int = 128) {
        _resnet.wrappedValue = NFKBiSeNetResNet18(width: width)
        _arm16.wrappedValue = NFKBiSeNetARM(inChannels: width * 4, outChannels: contextChannels)
        _arm32.wrappedValue = NFKBiSeNetARM(inChannels: width * 8, outChannels: contextChannels)
        _convHead32.wrappedValue = NFKBiSeNetConvBlock(inChannels: contextChannels, outChannels: contextChannels, kernel: 3)
        _convHead16.wrappedValue = NFKBiSeNetConvBlock(inChannels: contextChannels, outChannels: contextChannels, kernel: 3)
        _convAvg.wrappedValue = NFKBiSeNetConvBlock(inChannels: width * 8, outChannels: contextChannels,
                                                    kernel: 1, padding: 0)
    }

    func callAsFunction(_ x: MLXArray) -> (feat8: MLXArray, feat16: MLXArray) {
        let (_, feat16, feat32) = resnet(x)
        let average = convAvg(mean(feat32, axes: [1, 2], keepDims: true))
        let sum32 = arm32(feat32) + average
        let up32 = convHead32(NFKMLXResample.upsampleNearest(sum32, scale: 2))
        let sum16 = arm16(feat16) + up32
        let up16 = convHead16(NFKMLXResample.upsampleNearest(sum16, scale: 2))
        return (up16, up32)
    }
}

/// The shallow spatial path: three strided convolutions and a 1×1 projection, preserving detail the
/// context path throws away.
final class NFKBiSeNetSpatialPath: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKBiSeNetConvBlock
    @ModuleInfo(key: "conv2") var conv2: NFKBiSeNetConvBlock
    @ModuleInfo(key: "conv3") var conv3: NFKBiSeNetConvBlock
    @ModuleInfo(key: "conv_out") var convOut: NFKBiSeNetConvBlock

    init(width: Int = 64, outChannels: Int = 128) {
        _conv1.wrappedValue = NFKBiSeNetConvBlock(inChannels: 3, outChannels: width, kernel: 7, stride: 2, padding: 3)
        _conv2.wrappedValue = NFKBiSeNetConvBlock(inChannels: width, outChannels: width, kernel: 3, stride: 2)
        _conv3.wrappedValue = NFKBiSeNetConvBlock(inChannels: width, outChannels: width, kernel: 3, stride: 2)
        _convOut.wrappedValue = NFKBiSeNetConvBlock(inChannels: width, outChannels: outChannels, kernel: 1, padding: 0)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { convOut(conv3(conv2(conv1(x)))) }
}

/// The feature fusion module: concatenate both paths, then add a channel-gated copy of the result.
final class NFKBiSeNetFFM: Module {
    @ModuleInfo(key: "convblk") var convblk: NFKBiSeNetConvBlock
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm

    init(inChannels: Int, outChannels: Int) {
        _convblk.wrappedValue = NFKBiSeNetConvBlock(inChannels: inChannels, outChannels: outChannels,
                                                    kernel: 1, padding: 0)
        _conv.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels,
                                    kernelSize: 1, bias: false)
        _bn.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ spatial: MLXArray, _ context: MLXArray) -> MLXArray {
        let feature = convblk(concatenated([spatial, context], axis: 3))
        let attention = sigmoid(bn(conv(mean(feature, axes: [1, 2], keepDims: true))))
        return feature + feature * attention
    }
}

/// A segmentation head: a 3×3 convolution then a 1×1 to the class count.
final class NFKBiSeNetOutput: Module {
    @ModuleInfo(key: "conv") var conv: NFKBiSeNetConvBlock
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(inChannels: Int, midChannels: Int, classCount: Int) {
        _conv.wrappedValue = NFKBiSeNetConvBlock(inChannels: inChannels, outChannels: midChannels, kernel: 3)
        _convOut.wrappedValue = Conv2d(inputChannels: midChannels, outputChannels: classCount, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { convOut(conv(x)) }
}

/// The BiSeNet network: a detail-preserving spatial path, a deep context path, and the fusion module
/// that combines them into class logits.
final class NFKMLXBiSeNetNet: Module {
    @ModuleInfo(key: "cp") var cp: NFKBiSeNetContextPath
    @ModuleInfo(key: "sp") var sp: NFKBiSeNetSpatialPath
    @ModuleInfo(key: "ffm") var ffm: NFKBiSeNetFFM
    @ModuleInfo(key: "conv_out") var convOut: NFKBiSeNetOutput
    @ModuleInfo(key: "conv_out16") var convOut16: NFKBiSeNetOutput
    @ModuleInfo(key: "conv_out32") var convOut32: NFKBiSeNetOutput

    let configuration: NFKMLXBiSeNetConfiguration

    init(_ configuration: NFKMLXBiSeNetConfiguration = .base) {
        self.configuration = configuration
        let width = configuration.baseChannels
        let context = width * 2                                 // 128 at the released width
        _cp.wrappedValue = NFKBiSeNetContextPath(width: width, contextChannels: context)
        _sp.wrappedValue = NFKBiSeNetSpatialPath(width: width, outChannels: context)
        _ffm.wrappedValue = NFKBiSeNetFFM(inChannels: context * 2, outChannels: context * 2)
        _convOut.wrappedValue = NFKBiSeNetOutput(inChannels: context * 2, midChannels: context * 2,
                                                 classCount: configuration.classCount)
        // The auxiliary heads supervise the context path during training. Inference never reads them,
        // but the released checkpoint carries them, so they are built to keep the load complete.
        _convOut16.wrappedValue = NFKBiSeNetOutput(inChannels: context, midChannels: width,
                                                   classCount: configuration.classCount)
        _convOut32.wrappedValue = NFKBiSeNetOutput(inChannels: context, midChannels: width,
                                                   classCount: configuration.classCount)
    }

    /// Class logits `[1, H, W, classCount]` at the input resolution, for a batched image
    /// `[1, H, W, 3]`. The reference upsamples its head's output ×8 bilinearly.
    func logits(_ image: MLXArray) -> MLXArray {
        let (feat8, _) = cp(image)
        let fused = ffm(sp(image), feat8)
        let out = convOut(fused)
        return NFKMLXResample.resizeBilinear(out, height: image.shape[1], width: image.shape[2])
    }

    /// The label map as a grayscale image `[H, W, 1]`, the same convention `NFKMLXSegFormer` and
    /// `NFKMLXDeepLab` use: recover the class index as `round(gray · (classCount − 1))`.
    func segment(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let batched = image.reshaped([1, height, width, image.shape[2]])

        // The context path adds a ×2 upsample of the stride-32 feature to the stride-16 one, which
        // only lines up when both sides are multiples of 32 — the reference has the same constraint
        // and simply requires such inputs. An arbitrary frame is resized for the network here.
        let alignedHeight = max(32, (height + 31) / 32 * 32)
        let alignedWidth = max(32, (width + 31) / 32 * 32)
        var input = batched
        if alignedHeight != height || alignedWidth != width {
            input = NFKMLXResample.resizeBilinear(batched, height: alignedHeight, width: alignedWidth)
        }

        // Resize the logits, not the labels: interpolating class indices would invent classes that
        // neither neighbour predicted.
        var predicted = logits(NFKMLXBiSeNetNet.normalized(input))
        if alignedHeight != height || alignedWidth != width {
            predicted = NFKMLXResample.resizeBilinear(predicted, height: height, width: width)
        }
        let labels = predicted.argMax(axis: -1)
        let scale = Float(max(configuration.classCount - 1, 1))
        return (labels.asType(.float32) / scale).reshaped([height, width, 1])
    }

    /// ImageNet normalization, which the reference applies before the network.
    static func normalized(_ image: MLXArray) -> MLXArray {
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 1, 3])
        let deviation = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 1, 3])
        return (image - mean) / deviation
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKBiSeNetHolder: @unchecked Sendable {
    let net: NFKMLXBiSeNetNet
    init(_ net: NFKMLXBiSeNetNet) { self.net = net }
}

/// BiSeNet real-time semantic segmentation as an InferKit backend, and its registration for the
/// Objective-C path.
///
/// `NFKMLXBiSeNetNet` is the real two-path segmenter. Random weights run (proving the pipeline); a
/// trained checkpoint segments accurately. The output is a grayscale label map under `NFKOutputImage`;
/// recover the class index as `round(gray·(classCount−1))`. Load a **safetensors** checkpoint; the
/// loader transposes 4-D convolution weights `[out, in, kH, kW]` to MLX's `[out, kH, kW, in]`.
@objc(NFKMLXBiSeNet)
public final class NFKMLXBiSeNet: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "bisenet"

    static func makeNet(_ configuration: NFKMLXBiSeNetConfiguration = .base) -> NFKMLXBiSeNetNet {
        NFKMLXBiSeNetNet(configuration)
    }

    /// Builds a segmentation backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXBiSeNetNet(.base)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        net.train(false)                                       // BatchNorm running statistics
        let holder = NFKBiSeNetHolder(net)
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

    /// Registers BiSeNet (`bisenet`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Loads a safetensors checkpoint into `net`, transposing 4-D convolution weights from PyTorch's
    /// `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`.
    /// Maps the reference's names onto the module's: only the ResNet block's projection shortcut is
    /// positional (`downsample.0` the convolution, `downsample.1` its normalization).
    static func remapReferenceKey(_ key: String) -> String {
        guard let range = key.range(of: "downsample.") else { return key }
        let slot = key[range.upperBound...].prefix(1)
        let name = slot == "0" ? "downsample_conv." : "downsample_bn."
        return key[..<range.lowerBound] + name + key[range.upperBound...].dropFirst(2)
    }

    static func loadWeights(into net: NFKMLXBiSeNetNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remapReferenceKey(key), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

// MARK: - BiSeNetV2
//
// V2 replaces V1's ResNet context path with a purpose-built pair: a wide, shallow **detail** branch
// and a narrow, deep **semantic** branch of Gather-and-Expansion layers, fused by a bilateral
// aggregation layer. Its released checkpoint predates the current reference head: it emits
// `classes × upFactor²` channels and **pixel-shuffles** to full resolution, where the repository now
// emits `classes` and interpolates. Everything before the heads is unchanged.

/// A general pixel shuffle: `[N, H, W, C·r²]` → `[N, H·r, W·r, C]`, matching PyTorch's channel order
/// (`c·r² + i·r + j`). `NFKMLXSwinIRNet.pixelShuffle2` is the ×2 special case; a factor of eight is
/// not three ×2 shuffles, because the channel interleaving differs.
/// PyTorch's `PixelShuffle`: a `[N, H, W, C·r²]` map becomes `[N, H·r, W·r, C]`, reading the packed
/// channels in the reference's `c·r² + i·r + j` order. A factor-`r` shuffle is NOT a chain of smaller
/// ones — the channel interleaving differs — so the factor is applied in one step.
enum NFKMLXPixelShuffle {
    static func apply(_ x: MLXArray, factor: Int) -> MLXArray {
        let (n, h, w) = (x.shape[0], x.shape[1], x.shape[2])
        let channels = x.shape[3] / (factor * factor)
        return x.reshaped([n, h, w, channels, factor, factor])
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped([n, h * factor, w * factor, channels])
    }
}

/// The Gather-and-Expansion layer. At stride one it is a residual block; at stride two it downsamples
/// through two depthwise stages and carries a depthwise-then-pointwise shortcut.
final class NFKBiSeNetGELayer: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKBiSeNetConvBlock
    @ModuleInfo(key: "dw1_conv") var dw1Conv: Conv2d
    @ModuleInfo(key: "dw1_bn") var dw1BN: BatchNorm
    @ModuleInfo(key: "dw2_conv") var dw2Conv: Conv2d?
    @ModuleInfo(key: "dw2_bn") var dw2BN: BatchNorm?
    @ModuleInfo(key: "conv2_conv") var conv2Conv: Conv2d
    @ModuleInfo(key: "conv2_bn") var conv2BN: BatchNorm
    @ModuleInfo(key: "shortcut_dw_conv") var shortcutDWConv: Conv2d?
    @ModuleInfo(key: "shortcut_dw_bn") var shortcutDWBN: BatchNorm?
    @ModuleInfo(key: "shortcut_conv") var shortcutConv: Conv2d?
    @ModuleInfo(key: "shortcut_bn") var shortcutBN: BatchNorm?
    let stride: Int

    init(inChannels: Int, outChannels: Int, stride: Int, expansion: Int = 6) {
        let hidden = inChannels * expansion
        self.stride = stride
        _conv1.wrappedValue = NFKBiSeNetConvBlock(inChannels: inChannels, outChannels: inChannels, kernel: 3)
        _dw1Conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: hidden, kernelSize: 3,
                                       stride: IntOrPair(stride), padding: 1, groups: inChannels, bias: false)
        _dw1BN.wrappedValue = BatchNorm(featureCount: hidden)
        if stride == 2 {
            // The stride-two form expands once to downsample and again at full width.
            _dw2Conv.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: hidden, kernelSize: 3,
                                           padding: 1, groups: hidden, bias: false)
            _dw2BN.wrappedValue = BatchNorm(featureCount: hidden)
            _shortcutDWConv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: inChannels,
                                                  kernelSize: 3, stride: 2, padding: 1,
                                                  groups: inChannels, bias: false)
            _shortcutDWBN.wrappedValue = BatchNorm(featureCount: inChannels)
            _shortcutConv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                                kernelSize: 1, bias: false)
            _shortcutBN.wrappedValue = BatchNorm(featureCount: outChannels)
        }
        _conv2Conv.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: outChannels,
                                         kernelSize: 1, bias: false)
        _conv2BN.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var feature = dw1BN(dw1Conv(conv1(x)))
        // The stride-one form activates after its single depthwise stage; the stride-two form only
        // after the second.
        if let dw2Conv, let dw2BN {
            feature = relu(dw2BN(dw2Conv(feature)))
        } else {
            feature = relu(feature)
        }
        feature = conv2BN(conv2Conv(feature))

        var identity = x
        if let shortcutDWConv, let shortcutDWBN, let shortcutConv, let shortcutBN {
            identity = shortcutBN(shortcutConv(shortcutDWBN(shortcutDWConv(x))))
        }
        return relu(feature + identity)
    }
}

/// The stem: one strided convolution, then a narrow branch and a max pool over the same feature,
/// concatenated and fused.
final class NFKBiSeNetStem: Module {
    @ModuleInfo(key: "conv") var conv: NFKBiSeNetConvBlock
    @ModuleInfo(key: "left1") var left1: NFKBiSeNetConvBlock
    @ModuleInfo(key: "left2") var left2: NFKBiSeNetConvBlock
    @ModuleInfo(key: "fuse") var fuse: NFKBiSeNetConvBlock

    override init() {
        _conv.wrappedValue = NFKBiSeNetConvBlock(inChannels: 3, outChannels: 16, kernel: 3, stride: 2)
        _left1.wrappedValue = NFKBiSeNetConvBlock(inChannels: 16, outChannels: 8, kernel: 1, padding: 0)
        _left2.wrappedValue = NFKBiSeNetConvBlock(inChannels: 8, outChannels: 16, kernel: 3, stride: 2)
        _fuse.wrappedValue = NFKBiSeNetConvBlock(inChannels: 32, outChannels: 16, kernel: 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let feature = conv(x)
        let left = left2(left1(feature))
        let right = NFKMLXResample.maxPooled(feature, kernel: 3, stride: 2, padding: 1)
        return fuse(concatenated([left, right], axis: 3))
    }
}

/// The context-embedding block: a globally pooled, normalized branch added back to the feature.
final class NFKBiSeNetCEBlock: Module {
    @ModuleInfo(key: "bn") var bn: BatchNorm
    @ModuleInfo(key: "conv_gap") var convGap: NFKBiSeNetConvBlock
    @ModuleInfo(key: "conv_last") var convLast: NFKBiSeNetConvBlock

    override init() {
        _bn.wrappedValue = BatchNorm(featureCount: 128)
        _convGap.wrappedValue = NFKBiSeNetConvBlock(inChannels: 128, outChannels: 128, kernel: 1, padding: 0)
        _convLast.wrappedValue = NFKBiSeNetConvBlock(inChannels: 128, outChannels: 128, kernel: 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convLast(convGap(bn(mean(x, axes: [1, 2], keepDims: true))) + x)
    }
}

/// Bilateral guided aggregation: each branch gates the other, at its own scale, before they merge.
final class NFKBiSeNetBGA: Module {
    @ModuleInfo(key: "left1_dw") var left1DW: Conv2d
    @ModuleInfo(key: "left1_bn") var left1BN: BatchNorm
    @ModuleInfo(key: "left1_pw") var left1PW: Conv2d
    @ModuleInfo(key: "left2_conv") var left2Conv: Conv2d
    @ModuleInfo(key: "left2_bn") var left2BN: BatchNorm
    @ModuleInfo(key: "right1_conv") var right1Conv: Conv2d
    @ModuleInfo(key: "right1_bn") var right1BN: BatchNorm
    @ModuleInfo(key: "right2_dw") var right2DW: Conv2d
    @ModuleInfo(key: "right2_bn") var right2BN: BatchNorm
    @ModuleInfo(key: "right2_pw") var right2PW: Conv2d
    @ModuleInfo(key: "conv_conv") var convConv: Conv2d
    @ModuleInfo(key: "conv_bn") var convBN: BatchNorm

    override init() {
        _left1DW.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 128, kernelSize: 3,
                                       padding: 1, groups: 128, bias: false)
        _left1BN.wrappedValue = BatchNorm(featureCount: 128)
        _left1PW.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 128, kernelSize: 1, bias: false)
        _left2Conv.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 128, kernelSize: 3,
                                         stride: 2, padding: 1, bias: false)
        _left2BN.wrappedValue = BatchNorm(featureCount: 128)
        _right1Conv.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 128, kernelSize: 3,
                                          padding: 1, bias: false)
        _right1BN.wrappedValue = BatchNorm(featureCount: 128)
        _right2DW.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 128, kernelSize: 3,
                                        padding: 1, groups: 128, bias: false)
        _right2BN.wrappedValue = BatchNorm(featureCount: 128)
        _right2PW.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 128, kernelSize: 1, bias: false)
        _convConv.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 128, kernelSize: 3,
                                        padding: 1, bias: false)
        _convBN.wrappedValue = BatchNorm(featureCount: 128)
    }

    func callAsFunction(detail: MLXArray, semantic: MLXArray) -> MLXArray {
        let left1 = left1PW(left1BN(left1DW(detail)))
        let left2 = NFKMLXResample.averagePooled(left2BN(left2Conv(detail)), kernel: 3, stride: 2, padding: 1)
        let right1 = right1BN(right1Conv(semantic))
        let right2 = right2PW(right2BN(right2DW(semantic)))
        // The reference upsamples both gates with a plain `nn.Upsample`, whose default is nearest.
        let left = left1 * sigmoid(NFKMLXResample.upsampleNearest(right1, scale: 4))
        let right = NFKMLXResample.upsampleNearest(left2 * sigmoid(right2), scale: 4)
        return relu(convBN(convConv(left + right)))
    }
}

/// The BiSeNetV2 network. Only the main head is built: the four auxiliary heads supervise training
/// and inference never reads them, so their checkpoint keys are simply left unused.
final class NFKMLXBiSeNetV2Net: Module {
    @ModuleInfo(key: "detail_s1") var detailS1: [NFKBiSeNetConvBlock]
    @ModuleInfo(key: "detail_s2") var detailS2: [NFKBiSeNetConvBlock]
    @ModuleInfo(key: "detail_s3") var detailS3: [NFKBiSeNetConvBlock]
    @ModuleInfo(key: "stem") var stem: NFKBiSeNetStem
    @ModuleInfo(key: "s3") var s3: [NFKBiSeNetGELayer]
    @ModuleInfo(key: "s4") var s4: [NFKBiSeNetGELayer]
    @ModuleInfo(key: "s5_4") var s54: [NFKBiSeNetGELayer]
    @ModuleInfo(key: "s5_5") var s55: NFKBiSeNetCEBlock
    @ModuleInfo(key: "bga") var bga: NFKBiSeNetBGA
    @ModuleInfo(key: "head_conv") var headConv: NFKBiSeNetConvBlock
    @ModuleInfo(key: "head_out") var headOut: Conv2d

    let classCount: Int
    let upFactor = 8

    init(classCount: Int = 19) {
        self.classCount = classCount
        _detailS1.wrappedValue = [NFKBiSeNetConvBlock(inChannels: 3, outChannels: 64, kernel: 3, stride: 2),
                                  NFKBiSeNetConvBlock(inChannels: 64, outChannels: 64, kernel: 3)]
        _detailS2.wrappedValue = [NFKBiSeNetConvBlock(inChannels: 64, outChannels: 64, kernel: 3, stride: 2),
                                  NFKBiSeNetConvBlock(inChannels: 64, outChannels: 64, kernel: 3),
                                  NFKBiSeNetConvBlock(inChannels: 64, outChannels: 64, kernel: 3)]
        _detailS3.wrappedValue = [NFKBiSeNetConvBlock(inChannels: 64, outChannels: 128, kernel: 3, stride: 2),
                                  NFKBiSeNetConvBlock(inChannels: 128, outChannels: 128, kernel: 3),
                                  NFKBiSeNetConvBlock(inChannels: 128, outChannels: 128, kernel: 3)]
        _stem.wrappedValue = NFKBiSeNetStem()
        _s3.wrappedValue = [NFKBiSeNetGELayer(inChannels: 16, outChannels: 32, stride: 2),
                            NFKBiSeNetGELayer(inChannels: 32, outChannels: 32, stride: 1)]
        _s4.wrappedValue = [NFKBiSeNetGELayer(inChannels: 32, outChannels: 64, stride: 2),
                            NFKBiSeNetGELayer(inChannels: 64, outChannels: 64, stride: 1)]
        _s54.wrappedValue = [NFKBiSeNetGELayer(inChannels: 64, outChannels: 128, stride: 2),
                             NFKBiSeNetGELayer(inChannels: 128, outChannels: 128, stride: 1),
                             NFKBiSeNetGELayer(inChannels: 128, outChannels: 128, stride: 1),
                             NFKBiSeNetGELayer(inChannels: 128, outChannels: 128, stride: 1)]
        _s55.wrappedValue = NFKBiSeNetCEBlock()
        _bga.wrappedValue = NFKBiSeNetBGA()
        _headConv.wrappedValue = NFKBiSeNetConvBlock(inChannels: 128, outChannels: 1024, kernel: 3)
        // The released head emits one channel per class per output pixel of the shuffle.
        _headOut.wrappedValue = Conv2d(inputChannels: 1024, outputChannels: classCount * upFactor * upFactor,
                                       kernelSize: 1)
    }

    /// The label map as a grayscale image `[H, W, 1]`, the convention the other segmenters share.
    /// The bilateral aggregation adds a ×4 upsample of the semantic branch to the detail branch, so
    /// the sides have to be multiples of 32 — the frame is resized for the network and the logits
    /// resized back, never the labels.
    func segment(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let batched = image.reshaped([1, height, width, image.shape[2]])
        let alignedHeight = max(32, (height + 31) / 32 * 32)
        let alignedWidth = max(32, (width + 31) / 32 * 32)
        var input = batched
        if alignedHeight != height || alignedWidth != width {
            input = NFKMLXResample.resizeBilinear(batched, height: alignedHeight, width: alignedWidth)
        }
        var predicted = logits(NFKMLXBiSeNetNet.normalized(input))
        if alignedHeight != height || alignedWidth != width {
            predicted = NFKMLXResample.resizeBilinear(predicted, height: height, width: width)
        }
        let labels = predicted.argMax(axis: -1)
        return (labels.asType(.float32) / Float(max(classCount - 1, 1))).reshaped([height, width, 1])
    }

    /// Class logits `[1, H, W, classCount]` for a normalized image `[1, H, W, 3]`.
    func logits(_ image: MLXArray) -> MLXArray {
        var detail = image
        for block in detailS1 { detail = block(detail) }
        for block in detailS2 { detail = block(detail) }
        for block in detailS3 { detail = block(detail) }

        var semantic = stem(image)
        for layer in s3 { semantic = layer(semantic) }
        for layer in s4 { semantic = layer(semantic) }
        for layer in s54 { semantic = layer(semantic) }
        semantic = s55(semantic)

        let fused = bga(detail: detail, semantic: semantic)
        return NFKMLXPixelShuffle.apply(headOut(headConv(fused)), factor: upFactor)
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKBiSeNetV2Holder: @unchecked Sendable {
    let net: NFKMLXBiSeNetV2Net
    init(_ net: NFKMLXBiSeNetV2Net) { self.net = net }
}

/// BiSeNetV2 as an InferKit backend, and its registration for the Objective-C path.
///
/// Emits a grayscale class-label map under `NFKOutputImage`, the same convention as
/// `NFKMLXBiSeNet`, `NFKMLXSegFormer`, and `NFKMLXDeepLab`.
@objc(NFKMLXBiSeNetV2)
public final class NFKMLXBiSeNetV2: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "bisenet-v2"

    static func makeNet(classCount: Int = 19) -> NFKMLXBiSeNetV2Net {
        NFKMLXBiSeNetV2Net(classCount: classCount)
    }

    /// Builds a segmentation backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true).
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXBiSeNetV2Net()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        net.train(false)
        let holder = NFKBiSeNetV2Holder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in
            holder.net.segment(image)
        }
    }

    /// Downloads the checkpoint, then builds the backend. Blocking on the network; run off the render
    /// thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers BiSeNetV2 (`bisenet-v2`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Maps the reference's names onto the module's. Nearly everything is a numbered `Sequential`
    /// whose entries this module names, and the four `aux*` heads have no counterpart — they
    /// supervise training only, so their keys pass through unmapped and are ignored as extras.
    static func remapReferenceKey(_ key: String) -> String {
        var key = key
        for (branch, stage) in [("detail.S1.", "detail_s1."), ("detail.S2.", "detail_s2."),
                                ("detail.S3.", "detail_s3.")] where key.hasPrefix(branch) {
            key = stage + key.dropFirst(branch.count)
        }
        key = key.replacingOccurrences(of: "segment.S1S2.", with: "stem.")
        key = key.replacingOccurrences(of: "segment.S3.", with: "s3.")
        key = key.replacingOccurrences(of: "segment.S4.", with: "s4.")
        key = key.replacingOccurrences(of: "segment.S5_4.", with: "s5_4.")
        key = key.replacingOccurrences(of: "segment.S5_5.", with: "s5_5.")
        // The stem's narrow branch is a two-entry Sequential.
        key = key.replacingOccurrences(of: "stem.left.0.", with: "stem.left1.")
        key = key.replacingOccurrences(of: "stem.left.1.", with: "stem.left2.")
        // Each Gather-and-Expansion layer packs its depthwise stages, projection, and shortcut into
        // numbered Sequentials.
        for (reference, name) in [("dwconv1.0.", "dw1_conv."), ("dwconv1.1.", "dw1_bn."),
                                  ("dwconv2.0.", "dw2_conv."), ("dwconv2.1.", "dw2_bn."),
                                  ("dwconv.0.", "dw1_conv."), ("dwconv.1.", "dw1_bn."),
                                  ("conv2.0.", "conv2_conv."), ("conv2.1.", "conv2_bn."),
                                  ("shortcut.0.", "shortcut_dw_conv."), ("shortcut.1.", "shortcut_dw_bn."),
                                  ("shortcut.2.", "shortcut_conv."), ("shortcut.3.", "shortcut_bn.")] {
            key = key.replacingOccurrences(of: "." + reference, with: "." + name)
        }
        // Bilateral aggregation: two of its four paths are depthwise-then-pointwise pairs.
        for (reference, name) in [("bga.left1.0.", "bga.left1_dw."), ("bga.left1.1.", "bga.left1_bn."),
                                  ("bga.left1.2.", "bga.left1_pw."), ("bga.left2.0.", "bga.left2_conv."),
                                  ("bga.left2.1.", "bga.left2_bn."), ("bga.right1.0.", "bga.right1_conv."),
                                  ("bga.right1.1.", "bga.right1_bn."), ("bga.right2.0.", "bga.right2_dw."),
                                  ("bga.right2.1.", "bga.right2_bn."), ("bga.right2.2.", "bga.right2_pw."),
                                  ("bga.conv.0.", "bga.conv_conv."), ("bga.conv.1.", "bga.conv_bn.")] {
            key = key.replacingOccurrences(of: reference, with: name)
        }
        key = key.replacingOccurrences(of: "head.conv.", with: "head_conv.")
        key = key.replacingOccurrences(of: "head.conv_out.0.", with: "head_out.")
        return key
    }

    /// Loads a safetensors checkpoint. The auxiliary heads are dropped: they exist only to supervise
    /// training, and building them would cost the largest tensors in the file for no inference use.
    static func loadWeights(into net: NFKMLXBiSeNetV2Net, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            guard !key.hasPrefix("aux") else { return nil }
            let name = remapReferenceKey(key)
            return (name, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
