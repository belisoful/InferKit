//
//  NFKMLXResNet.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXNN

// The residual backbone the segmentation and pose models are built on (He et al., "Deep Residual
// Learning for Image Recognition"). This is the bottleneck form used from ResNet-50 up: a 7×7 stem, then
// four stages whose blocks widen ×4 on output. It follows the reference layout, including the
// substitution DeepLab relies on — a stage keeps its resolution and dilates its 3×3 convolution instead
// of striding — so a released checkpoint's keys map onto it directly. Tensors flow in NHWC.

/// Residual backbone sizing.
public struct NFKMLXResNetConfiguration: Sendable {
    /// Bottleneck blocks per stage; ResNet-50 is `[3, 4, 6, 3]`.
    public var blocks: [Int]
    /// The stem's channel count, and the first stage's bottleneck width. Each stage doubles it.
    public var width: Int
    /// Replaces the stride of stages two, three, and four with a dilation. DeepLab dilates the last two,
    /// holding the output stride at 8.
    public var replaceStrideWithDilation: [Bool]

    public init(blocks: [Int] = [3, 4, 6, 3], width: Int = 64,
                replaceStrideWithDilation: [Bool] = [false, false, false]) {
        self.blocks = blocks
        self.width = width
        self.replaceStrideWithDilation = replaceStrideWithDilation
    }

    /// ResNet-50 with the last two stages dilated — the DeepLabV3 backbone.
    public static let deepLab = NFKMLXResNetConfiguration(replaceStrideWithDilation: [false, true, true])

    /// A two-stage, narrow stack for tests.
    public static let tiny = NFKMLXResNetConfiguration(blocks: [1, 1, 1, 1], width: 8,
                                                       replaceStrideWithDilation: [false, true, true])

    /// The channels the last stage emits.
    var outputChannels: Int { width * 8 * NFKResNetBottleneck.expansion }
}

/// A bottleneck block: a 1×1 narrowing, a 3×3 at the block's stride or dilation, a 1×1 widening ×4, and
/// an identity that projects when the shape changes.
final class NFKResNetBottleneck: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm
    @ModuleInfo(key: "conv3") var conv3: Conv2d
    @ModuleInfo(key: "bn3") var bn3: BatchNorm
    @ModuleInfo(key: "downsample_conv") var downsampleConv: Conv2d?
    @ModuleInfo(key: "downsample_bn") var downsampleBN: BatchNorm?

    static let expansion = 4

    init(inChannels: Int, width: Int, stride: Int, dilation: Int, downsample: Bool) {
        let outChannels = width * Self.expansion
        _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: width, kernelSize: 1, bias: false)
        _bn1.wrappedValue = BatchNorm(featureCount: width)
        _conv2.wrappedValue = Conv2d(inputChannels: width, outputChannels: width, kernelSize: 3,
                                     stride: IntOrPair(stride), padding: IntOrPair(dilation),
                                     dilation: IntOrPair(dilation), bias: false)
        _bn2.wrappedValue = BatchNorm(featureCount: width)
        _conv3.wrappedValue = Conv2d(inputChannels: width, outputChannels: outChannels, kernelSize: 1, bias: false)
        _bn3.wrappedValue = BatchNorm(featureCount: outChannels)
        if downsample {
            _downsampleConv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                                  kernelSize: 1, stride: IntOrPair(stride), bias: false)
            _downsampleBN.wrappedValue = BatchNorm(featureCount: outChannels)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(bn1(conv1(x)))
        out = relu(bn2(conv2(out)))
        out = bn3(conv3(out))
        var identity = x
        if let downsampleConv, let downsampleBN {
            identity = downsampleBN(downsampleConv(x))
        }
        return relu(out + identity)
    }
}

/// The residual backbone: stem, four bottleneck stages. `[1, H, W, 3]` → `[1, H/s, W/s, outputChannels]`,
/// where the stride `s` is 32 undilated and 8 with the last two stages dilated.
final class NFKMLXResNetBackbone: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "layer1") var layer1: [NFKResNetBottleneck]
    @ModuleInfo(key: "layer2") var layer2: [NFKResNetBottleneck]
    @ModuleInfo(key: "layer3") var layer3: [NFKResNetBottleneck]
    @ModuleInfo(key: "layer4") var layer4: [NFKResNetBottleneck]

    init(_ configuration: NFKMLXResNetConfiguration) {
        _conv1.wrappedValue = Conv2d(inputChannels: 3, outputChannels: configuration.width, kernelSize: 7,
                                     stride: 2, padding: 3, bias: false)
        _bn1.wrappedValue = BatchNorm(featureCount: configuration.width)

        var inChannels = configuration.width
        var dilation = 1
        func stage(_ index: Int, width: Int, stride: Int) -> [NFKResNetBottleneck] {
            // The reference gives the stage's FIRST block the dilation the previous stage ended at, and
            // only the rest the new one.
            let previousDilation = dilation
            var stride = stride
            if index > 0, configuration.replaceStrideWithDilation[index - 1] {
                dilation *= stride
                stride = 1
            }
            let outChannels = width * NFKResNetBottleneck.expansion
            var blocks = [NFKResNetBottleneck(inChannels: inChannels, width: width, stride: stride,
                                              dilation: previousDilation,
                                              downsample: stride != 1 || inChannels != outChannels)]
            inChannels = outChannels
            for _ in 1 ..< configuration.blocks[index] {
                blocks.append(NFKResNetBottleneck(inChannels: inChannels, width: width, stride: 1,
                                                  dilation: dilation, downsample: false))
            }
            return blocks
        }

        _layer1.wrappedValue = stage(0, width: configuration.width, stride: 1)
        _layer2.wrappedValue = stage(1, width: configuration.width * 2, stride: 2)
        _layer3.wrappedValue = stage(2, width: configuration.width * 4, stride: 2)
        _layer4.wrappedValue = stage(3, width: configuration.width * 8, stride: 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = NFKMLXResample.maxPooled(relu(bn1(conv1(x))), kernel: 3, stride: 2, padding: 1)
        for stage in [layer1, layer2, layer3, layer4] {
            for block in stage {
                out = block(out)
            }
        }
        return out
    }

    /// The reference nests a block's projection shortcut in a two-entry `Sequential`, so its keys are
    /// `downsample.0` (the convolution) and `downsample.1` (the normalization).
    static func remapReferenceKey(_ key: String) -> String {
        guard let range = key.range(of: "downsample.") else { return key }
        let slot = key[range.upperBound...].prefix(1)
        let name = slot == "0" ? "downsample_conv." : "downsample_bn."
        return key[..<range.lowerBound] + name + key[range.upperBound...].dropFirst(2)
    }
}
