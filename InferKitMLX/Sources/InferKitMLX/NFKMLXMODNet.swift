//
//  NFKMLXMODNet.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// MODNet mattes a portrait without a trimap by splitting the problem across three branches that share
// one MobileNetV2 encoder: a low-resolution branch that decides what the subject is, a high-resolution
// branch that recovers the boundary detail the low-resolution one cannot, and a fusion branch that
// combines them into the alpha matte. This is the reference network (ZHKKKe/MODNet), including its
// distinctive `IBNorm` — the first half of a layer's channels are batch-normalized and the rest
// instance-normalized without affine terms, then concatenated. Tensors flow in NHWC.

/// MODNet sizing. The defaults are the released photographic model; `tiny` keeps tests fast.
public struct NFKMLXMODNetConfiguration: Sendable {
    /// The width the high-resolution and fusion branches work at.
    public var hrChannels: Int
    /// Encoder widths taken from the backbone at strides 2, 4, 8, 16, and 32.
    public var encoderChannels: [Int]
    /// MobileNetV2's inverted-residual settings: expansion, output channels, repeats, first stride.
    public var residualSettings: [[Int]]
    public var stemChannels: Int
    public var lastChannels: Int

    public init(hrChannels: Int = 32, encoderChannels: [Int] = [16, 24, 32, 96, 1280],
                residualSettings: [[Int]] = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
                                             [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]],
                stemChannels: Int = 32, lastChannels: Int = 1280) {
        self.hrChannels = hrChannels
        self.encoderChannels = encoderChannels
        self.residualSettings = residualSettings
        self.stemChannels = stemChannels
        self.lastChannels = lastChannels
    }

    public static let base = NFKMLXMODNetConfiguration()

    /// A narrow stack that keeps every structural feature (expansion-1 block, strides, five captures).
    public static let tiny = NFKMLXMODNetConfiguration(
        hrChannels: 8, encoderChannels: [8, 8, 8, 8, 16],
        residualSettings: [[1, 8, 1, 1], [6, 8, 1, 2], [6, 8, 1, 2], [6, 8, 1, 2], [6, 8, 1, 1],
                           [6, 8, 1, 2], [6, 8, 1, 1]],
        stemChannels: 8, lastChannels: 16)

    /// The backbone stage indices whose outputs the branches consume, at strides 2, 4, 8, 16, 32.
    var captureIndices: [Int] {
        var indices = [Int]()
        var stage = 0
        for setting in residualSettings {
            stage += setting[2]
            indices.append(stage)
        }
        // enc2x / enc4x / enc8x / enc16x are the ends of settings 0, 1, 2, and 4; enc32x is the final
        // 1×1 expansion, which sits one past the last residual stage.
        return [indices[0], indices[1], indices[2], indices[4], indices[6] + 1]
    }
}

/// A convolution followed by batch normalization and ReLU6 — MobileNetV2's stem and final expansion.
final class NFKMODNetConvBN: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm

    init(inChannels: Int, outChannels: Int, kernel: Int = 3, stride: Int = 1, groups: Int = 1) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                    padding: IntOrPair(kernel / 2), groups: groups, bias: false)
        _bn.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { clip(bn(conv(x)), min: 0, max: 6) }
}

/// MobileNetV2's inverted residual: an optional 1×1 expansion, a depthwise convolution, and a linear
/// 1×1 projection. The reference stores the whole thing in one `conv` Sequential whose indices depend
/// on whether the expansion is present, which `remapReferenceKey` translates.
final class NFKMODNetInvertedResidual: Module {
    @ModuleInfo(key: "expand") var expand: NFKMODNetConvBN?
    @ModuleInfo(key: "dw") var depthwise: NFKMODNetConvBN
    @ModuleInfo(key: "project") var projectConv: Conv2d
    @ModuleInfo(key: "project_bn") var projectBN: BatchNorm
    let usesResidual: Bool

    init(inChannels: Int, outChannels: Int, stride: Int, expansion: Int) {
        let hidden = inChannels * expansion
        if expansion != 1 {
            _expand.wrappedValue = NFKMODNetConvBN(inChannels: inChannels, outChannels: hidden, kernel: 1)
        }
        _depthwise.wrappedValue = NFKMODNetConvBN(inChannels: hidden, outChannels: hidden, kernel: 3,
                                                  stride: stride, groups: hidden)
        _projectConv.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: outChannels,
                                           kernelSize: 1, bias: false)
        _projectBN.wrappedValue = BatchNorm(featureCount: outChannels)
        usesResidual = stride == 1 && inChannels == outChannels
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = expand.map { $0(x) } ?? x
        out = depthwise(out)
        out = projectBN(projectConv(out))
        return usesResidual ? x + out : out
    }
}

/// The MobileNetV2 encoder, emitting the five features the branches consume.
final class NFKMODNetBackbone: Module {
    @ModuleInfo(key: "stem") var stem: NFKMODNetConvBN
    @ModuleInfo(key: "blocks") var blocks: [NFKMODNetInvertedResidual]
    @ModuleInfo(key: "last") var last: NFKMODNetConvBN
    let captures: [Int]

    init(_ c: NFKMLXMODNetConfiguration) {
        _stem.wrappedValue = NFKMODNetConvBN(inChannels: 3, outChannels: c.stemChannels, kernel: 3, stride: 2)
        var blocks = [NFKMODNetInvertedResidual]()
        var inChannels = c.stemChannels
        for setting in c.residualSettings {
            let (expansion, outChannels, repeats, stride) = (setting[0], setting[1], setting[2], setting[3])
            for index in 0 ..< repeats {
                blocks.append(NFKMODNetInvertedResidual(inChannels: inChannels, outChannels: outChannels,
                                                        stride: index == 0 ? stride : 1, expansion: expansion))
                inChannels = outChannels
            }
        }
        _blocks.wrappedValue = blocks
        _last.wrappedValue = NFKMODNetConvBN(inChannels: inChannels, outChannels: c.lastChannels, kernel: 1)
        captures = c.captureIndices
    }

    /// Returns the features at strides 2, 4, 8, 16, and 32. The stem counts as stage zero, so a
    /// capture index of `n` is the output after `n` inverted-residual blocks.
    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var out = stem(x)
        var captured = [MLXArray]()
        var stage = 0
        func recordIfCaptured() {
            if let slot = captures.firstIndex(of: stage), captured.count == slot {
                captured.append(out)
            }
        }
        for block in blocks {
            out = block(out)
            stage += 1
            recordIfCaptured()
        }
        out = last(out)
        stage += 1
        recordIfCaptured()
        return captured
    }
}

/// The reference `IBNorm`: the first half of the channels are batch-normalized, the rest
/// instance-normalized **without affine terms**, and the two halves concatenated. A checkpoint
/// therefore carries parameters for only half the width.
final class NFKMODNetIBNorm: Module {
    @ModuleInfo(key: "bnorm") var bnorm: BatchNorm
    let batchChannels: Int

    init(channels: Int) {
        batchChannels = channels / 2
        _bnorm.wrappedValue = BatchNorm(featureCount: batchChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = bnorm(x[0..., 0..., 0..., 0 ..< batchChannels])
        let rest = x[0..., 0..., 0..., batchChannels...]
        let mean = MLX.mean(rest, axes: [1, 2], keepDims: true)
        let variance = MLX.mean((rest - mean) * (rest - mean), axes: [1, 2], keepDims: true)
        return concatenated([normalized, (rest - mean) / sqrt(variance + 1e-5)], axis: 3)
    }
}

/// Convolution → `IBNorm` → ReLU, with either of the last two optional. The reference wraps them in a
/// `layers` Sequential, so its keys are numbered and the remap translates them.
final class NFKMODNetConvIBNormRelu: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: NFKMODNetIBNorm?
    let usesReLU: Bool

    init(inChannels: Int, outChannels: Int, kernel: Int, stride: Int = 1, padding: Int = 0,
         withIBNorm: Bool = true, withReLU: Bool = true) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                    padding: IntOrPair(padding))
        if withIBNorm {
            _norm.wrappedValue = NFKMODNetIBNorm(channels: outChannels)
        }
        usesReLU = withReLU
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x)
        if let norm {
            out = norm(out)
        }
        return usesReLU ? relu(out) : out
    }
}

/// Squeeze-and-excitation over the encoder's deepest feature: pool, two bias-free projections, gate.
final class NFKMODNetSEBlock: Module {
    @ModuleInfo(key: "reduce") var reduce: Linear
    @ModuleInfo(key: "expand") var expand: Linear

    init(channels: Int, reduction: Int) {
        _reduce.wrappedValue = Linear(channels, channels / reduction, bias: false)
        _expand.wrappedValue = Linear(channels / reduction, channels, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pooled = MLX.mean(x, axes: [1, 2])                  // [N, C]
        let gate = sigmoid(expand(relu(reduce(pooled))))
        return x * gate.reshaped([x.shape[0], 1, 1, x.shape[3]])
    }
}

/// The MODNet network: MobileNetV2 encoder, the low- and high-resolution branches, and the fusion
/// branch that produces the matte.
final class NFKMLXMODNetNet: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKMODNetBackbone
    @ModuleInfo(key: "se_block") var seBlock: NFKMODNetSEBlock
    @ModuleInfo(key: "conv_lr16x") var convLR16x: NFKMODNetConvIBNormRelu
    @ModuleInfo(key: "conv_lr8x") var convLR8x: NFKMODNetConvIBNormRelu
    @ModuleInfo(key: "conv_lr") var convLR: NFKMODNetConvIBNormRelu
    @ModuleInfo(key: "tohr_enc2x") var toHR2x: NFKMODNetConvIBNormRelu
    @ModuleInfo(key: "conv_enc2x") var convEnc2x: NFKMODNetConvIBNormRelu
    @ModuleInfo(key: "tohr_enc4x") var toHR4x: NFKMODNetConvIBNormRelu
    @ModuleInfo(key: "conv_enc4x") var convEnc4x: NFKMODNetConvIBNormRelu
    @ModuleInfo(key: "conv_hr4x") var convHR4x: [NFKMODNetConvIBNormRelu]
    @ModuleInfo(key: "conv_hr2x") var convHR2x: [NFKMODNetConvIBNormRelu]
    @ModuleInfo(key: "conv_hr") var convHR: [NFKMODNetConvIBNormRelu]
    @ModuleInfo(key: "conv_lr4x") var convLR4x: NFKMODNetConvIBNormRelu
    @ModuleInfo(key: "conv_f2x") var convF2x: NFKMODNetConvIBNormRelu
    @ModuleInfo(key: "conv_f") var convF: [NFKMODNetConvIBNormRelu]

    let configuration: NFKMLXMODNetConfiguration

    init(_ c: NFKMLXMODNetConfiguration = .base) {
        configuration = c
        let hr = c.hrChannels
        let enc = c.encoderChannels
        _backbone.wrappedValue = NFKMODNetBackbone(c)

        _seBlock.wrappedValue = NFKMODNetSEBlock(channels: enc[4], reduction: 4)
        _convLR16x.wrappedValue = NFKMODNetConvIBNormRelu(inChannels: enc[4], outChannels: enc[3], kernel: 5, padding: 2)
        _convLR8x.wrappedValue = NFKMODNetConvIBNormRelu(inChannels: enc[3], outChannels: enc[2], kernel: 5, padding: 2)
        _convLR.wrappedValue = NFKMODNetConvIBNormRelu(inChannels: enc[2], outChannels: 1, kernel: 3, stride: 2,
                                                       padding: 1, withIBNorm: false, withReLU: false)

        _toHR2x.wrappedValue = NFKMODNetConvIBNormRelu(inChannels: enc[0], outChannels: hr, kernel: 1)
        _convEnc2x.wrappedValue = NFKMODNetConvIBNormRelu(inChannels: hr + 3, outChannels: hr, kernel: 3, stride: 2, padding: 1)
        _toHR4x.wrappedValue = NFKMODNetConvIBNormRelu(inChannels: enc[1], outChannels: hr, kernel: 1)
        _convEnc4x.wrappedValue = NFKMODNetConvIBNormRelu(inChannels: 2 * hr, outChannels: 2 * hr, kernel: 3, padding: 1)
        _convHR4x.wrappedValue = [
            NFKMODNetConvIBNormRelu(inChannels: 3 * hr + 3, outChannels: 2 * hr, kernel: 3, padding: 1),
            NFKMODNetConvIBNormRelu(inChannels: 2 * hr, outChannels: 2 * hr, kernel: 3, padding: 1),
            NFKMODNetConvIBNormRelu(inChannels: 2 * hr, outChannels: hr, kernel: 3, padding: 1),
        ]
        _convHR2x.wrappedValue = [
            NFKMODNetConvIBNormRelu(inChannels: 2 * hr, outChannels: 2 * hr, kernel: 3, padding: 1),
            NFKMODNetConvIBNormRelu(inChannels: 2 * hr, outChannels: hr, kernel: 3, padding: 1),
            NFKMODNetConvIBNormRelu(inChannels: hr, outChannels: hr, kernel: 3, padding: 1),
            NFKMODNetConvIBNormRelu(inChannels: hr, outChannels: hr, kernel: 3, padding: 1),
        ]
        _convHR.wrappedValue = [
            NFKMODNetConvIBNormRelu(inChannels: hr + 3, outChannels: hr, kernel: 3, padding: 1),
            NFKMODNetConvIBNormRelu(inChannels: hr, outChannels: 1, kernel: 1, withIBNorm: false, withReLU: false),
        ]

        _convLR4x.wrappedValue = NFKMODNetConvIBNormRelu(inChannels: enc[2], outChannels: hr, kernel: 5, padding: 2)
        _convF2x.wrappedValue = NFKMODNetConvIBNormRelu(inChannels: 2 * hr, outChannels: hr, kernel: 3, padding: 1)
        _convF.wrappedValue = [
            NFKMODNetConvIBNormRelu(inChannels: hr + 3, outChannels: hr / 2, kernel: 3, padding: 1),
            NFKMODNetConvIBNormRelu(inChannels: hr / 2, outChannels: 1, kernel: 1, withIBNorm: false, withReLU: false),
        ]
    }

    private static func scaled(_ x: MLXArray, by factor: Double) -> MLXArray {
        NFKMLXResample.resizeBilinear(x, height: Int(Double(x.shape[1]) * factor),
                                      width: Int(Double(x.shape[2]) * factor))
    }

    /// The alpha matte `[1, H, W, 1]` in `0...1` for a batched image `[1, H, W, 3]` in `0...1`. The
    /// reference normalizes its input to `-1...1` before the encoder.
    func matte(batched image: MLXArray) -> MLXArray {
        let img = image * 2 - 1
        let features = backbone(img)
        let (enc2x, enc4x, enc32x) = (features[0], features[1], features[4])

        // Low-resolution branch: what the subject is.
        let gated = seBlock(enc32x)
        let lr16x = convLR16x(Self.scaled(gated, by: 2))
        let lr8x = convLR8x(Self.scaled(lr16x, by: 2))

        // High-resolution branch: the boundary detail the low-resolution branch cannot carry.
        let img2x = Self.scaled(img, by: 0.5)
        let img4x = Self.scaled(img, by: 0.25)
        let hrEnc2x = toHR2x(enc2x)
        var hr4x = convEnc2x(concatenated([img2x, hrEnc2x], axis: 3))
        hr4x = convEnc4x(concatenated([hr4x, toHR4x(enc4x)], axis: 3))
        hr4x = concatenated([hr4x, Self.scaled(lr8x, by: 2), img4x], axis: 3)
        for layer in convHR4x {
            hr4x = layer(hr4x)
        }
        var hr2x = concatenated([Self.scaled(hr4x, by: 2), hrEnc2x], axis: 3)
        for layer in convHR2x {
            hr2x = layer(hr2x)
        }

        // Fusion branch: the matte.
        let lr4x = convLR4x(Self.scaled(lr8x, by: 2))
        var fused = convF2x(concatenated([Self.scaled(lr4x, by: 2), hr2x], axis: 3))
        fused = concatenated([Self.scaled(fused, by: 2), img], axis: 3)
        for layer in convF {
            fused = layer(fused)
        }
        return sigmoid(fused)
    }

    /// Mattes a bridged image `[H, W, 3]` (`0...1`), returning straight foreground plus alpha as
    /// `[H, W, 4]`. The network's strides need sides that are multiples of 32, so an arbitrary frame
    /// is resized for the network and the matte resized back.
    func matte(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let batched = image.reshaped([1, height, width, image.shape[2]])
        let alignedHeight = max(32, (height + 31) / 32 * 32)
        let alignedWidth = max(32, (width + 31) / 32 * 32)
        var input = batched
        if alignedHeight != height || alignedWidth != width {
            input = NFKMLXResample.resizeBilinear(batched, height: alignedHeight, width: alignedWidth)
        }
        var alpha = matte(batched: input)
        if alignedHeight != height || alignedWidth != width {
            alpha = NFKMLXResample.resizeBilinear(alpha, height: height, width: width)
        }
        return concatenated([batched * alpha, alpha], axis: 3).reshaped([height, width, 4])
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKMODNetHolder: @unchecked Sendable {
    let net: NFKMLXMODNetNet
    init(_ net: NFKMLXMODNetNet) { self.net = net }
}

/// MODNet trimap-free portrait matting as an InferKit backend, and its registration for the
/// Objective-C path.
///
/// `NFKMLXMODNetNet` is the reference three-branch network. Random weights run (proving the
/// pipeline); the released photographic checkpoint, converted to **safetensors**, mattes accurately.
@objc(NFKMLXMODNet)
public final class NFKMLXMODNet: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "modnet"

    static func makeNet(_ configuration: NFKMLXMODNetConfiguration = .base) -> NFKMLXMODNetNet {
        NFKMLXMODNetNet(configuration)
    }

    /// Builds a matting backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXMODNetNet(.base)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        net.train(false)
        let holder = NFKMODNetHolder(net)
        var configuration = NFKMattingConfiguration()
        configuration.emitsMatte = true
        return NFKMLXMattingBackend(identifier: modelName, configuration: configuration) { plate, _ in
            holder.net.matte(plate)
        }
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

    /// Registers MODNet (`modnet`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Maps the released checkpoint's names onto the module's. The checkpoint carries the training
    /// wrapper's `module.` prefix, keeps each branch as its own submodule, and stores every composite
    /// as a numbered `Sequential`: the backbone's `features.N`, each inverted residual's `conv.M`
    /// (whose positions shift when the expansion is absent), and every `Conv2dIBNormRelu`'s `layers`.
    static func remapReferenceKey(_ key: String, configuration: NFKMLXMODNetConfiguration) -> String {
        var key = key
        if key.hasPrefix("module.") {
            key = String(key.dropFirst("module.".count))
        }
        for branch in ["lr_branch.", "hr_branch.", "f_branch."] where key.hasPrefix(branch) {
            key = String(key.dropFirst(branch.count))
        }
        // The high-resolution branch holds the backbone too; only the low-resolution branch's copy is
        // the one this module keeps, and both carry identical weights.
        if key.hasPrefix("backbone.model.features.") {
            let rest = key.dropFirst("backbone.model.features.".count)
            guard let dot = rest.firstIndex(of: "."), let stage = Int(rest[..<dot]) else { return key }
            let tail = String(rest[rest.index(after: dot)...])
            let blockCount = configuration.residualSettings.reduce(0) { $0 + $1[2] }
            if stage == 0 {
                return "backbone.stem." + (tail.hasPrefix("0.") ? "conv." + tail.dropFirst(2)
                                                                : "bn." + tail.dropFirst(2))
            }
            if stage == blockCount + 1 {
                return "backbone.last." + (tail.hasPrefix("0.") ? "conv." + tail.dropFirst(2)
                                                                : "bn." + tail.dropFirst(2))
            }
            guard tail.hasPrefix("conv.") else { return key }
            let inner = tail.dropFirst("conv.".count)
            guard let innerDot = inner.firstIndex(of: "."), let slot = Int(inner[..<innerDot]) else { return key }
            let suffix = String(inner[inner.index(after: innerDot)...])
            // An expansion-1 block has no 1×1 expansion, so its depthwise pair sits at 0/1 rather
            // than 3/4 and the projection at 3/4 rather than 6/7.
            let expanded = expansion(ofBlock: stage - 1, in: configuration) != 1
            let names: [Int: String] = expanded
                ? [0: "expand.conv", 1: "expand.bn", 3: "dw.conv", 4: "dw.bn", 6: "project", 7: "project_bn"]
                : [0: "dw.conv", 1: "dw.bn", 3: "project", 4: "project_bn"]
            guard let name = names[slot] else { return key }
            return "backbone.blocks.\(stage - 1).\(name)." + suffix
        }
        // `Conv2dIBNormRelu` wraps convolution and normalization in `layers`; the SE block's two
        // bias-free projections sit in `fc`.
        key = key.replacingOccurrences(of: ".layers.0.", with: ".conv.")
        key = key.replacingOccurrences(of: ".layers.1.", with: ".norm.")
        key = key.replacingOccurrences(of: "se_block.fc.0.", with: "se_block.reduce.")
        key = key.replacingOccurrences(of: "se_block.fc.2.", with: "se_block.expand.")
        return key
    }

    /// The expansion factor configured for a given inverted-residual block index.
    private static func expansion(ofBlock index: Int, in configuration: NFKMLXMODNetConfiguration) -> Int {
        var remaining = index
        for setting in configuration.residualSettings {
            if remaining < setting[2] {
                return setting[0]
            }
            remaining -= setting[2]
        }
        return 6
    }

    /// Loads a safetensors checkpoint into `net`, remapping the reference's names and transposing 4-D
    /// convolution weights from PyTorch's `[out, in, kH, kW]` to MLX's `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXMODNetNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        var mapped = [String: MLXArray]()
        for (key, value) in raw {
            let name = remapReferenceKey(key, configuration: net.configuration)
            // The checkpoint stores the backbone twice — once per branch that holds a reference to
            // it — and both copies are identical, so the first one wins.
            if mapped[name] == nil {
                mapped[name] = checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
            }
        }
        try NFKMLXWeights.apply(mapped.map { ($0.key, $0.value) }, to: net)
    }
}
