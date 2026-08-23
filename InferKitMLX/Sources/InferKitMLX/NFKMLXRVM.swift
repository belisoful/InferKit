//
//  NFKMLXRVM.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// Robust Video Matting separates a subject from its background without a green screen or a trimap. Its
// defining feature is a recurrent decoder: a ConvGRU at each decoder scale carries hidden state from
// one frame to the next, so temporal information sharpens the alpha matte and stabilizes it over a
// clip. A caller threads the four recurrent states across frames through `forward`; the first frame
// passes nil states.
//
// The network is the reference `MattingNetwork` (PeterL1n/RobustVideoMatting, mobilenetv3 variant):
// a torchvision MobileNetV3-Large encoder (inverted residuals with squeeze-and-excitation, hardswish,
// BatchNorm at epsilon 1e-3, the last stage dilated), an LR-ASPP context module, a recurrent decoder
// whose ConvGRU runs on half of each stage's channels, and a deep-guided-filter refiner for running
// the network at a reduced resolution while matting the full frame. The foreground head predicts a
// residual added to the source, and the alpha is a clamp, not a sigmoid. Tensors flow in NHWC.

/// One inverted-residual block's shape: torchvision's `InvertedResidualConfig`.
struct NFKRVMBlockSpec: Sendable {
    var input: Int
    var kernel: Int
    var expanded: Int
    var output: Int
    var usesSE: Bool
    var usesHardswish: Bool
    var stride: Int
    var dilation: Int
}

/// Robust Video Matting sizing. `large` is the released `rvm_mobilenetv3` geometry; `tiny` keeps the
/// same structure at test scale.
public struct NFKMLXRVMConfiguration: Sendable {
    var stemChannels: Int
    var lastChannels: Int
    var asppChannels: Int
    var blocks: [NFKRVMBlockSpec]
    /// Block indices whose outputs become the skip features `f1`, `f2`, `f3` (at 1/2, 1/4, 1/8).
    var captures: [Int]
    /// Decoder stage widths, coarse to fine: `[decode3, decode2, decode1, decode0]`.
    var decoderChannels: [Int]
    var refinerHiddenChannels: Int

    /// The released MobileNetV3-Large geometry (RVM passes torchvision the dilated large config).
    public static let large = NFKMLXRVMConfiguration(
        stemChannels: 16, lastChannels: 960, asppChannels: 128,
        blocks: [
            NFKRVMBlockSpec(input: 16, kernel: 3, expanded: 16, output: 16, usesSE: false, usesHardswish: false, stride: 1, dilation: 1),
            NFKRVMBlockSpec(input: 16, kernel: 3, expanded: 64, output: 24, usesSE: false, usesHardswish: false, stride: 2, dilation: 1),
            NFKRVMBlockSpec(input: 24, kernel: 3, expanded: 72, output: 24, usesSE: false, usesHardswish: false, stride: 1, dilation: 1),
            NFKRVMBlockSpec(input: 24, kernel: 5, expanded: 72, output: 40, usesSE: true, usesHardswish: false, stride: 2, dilation: 1),
            NFKRVMBlockSpec(input: 40, kernel: 5, expanded: 120, output: 40, usesSE: true, usesHardswish: false, stride: 1, dilation: 1),
            NFKRVMBlockSpec(input: 40, kernel: 5, expanded: 120, output: 40, usesSE: true, usesHardswish: false, stride: 1, dilation: 1),
            NFKRVMBlockSpec(input: 40, kernel: 3, expanded: 240, output: 80, usesSE: false, usesHardswish: true, stride: 2, dilation: 1),
            NFKRVMBlockSpec(input: 80, kernel: 3, expanded: 200, output: 80, usesSE: false, usesHardswish: true, stride: 1, dilation: 1),
            NFKRVMBlockSpec(input: 80, kernel: 3, expanded: 184, output: 80, usesSE: false, usesHardswish: true, stride: 1, dilation: 1),
            NFKRVMBlockSpec(input: 80, kernel: 3, expanded: 184, output: 80, usesSE: false, usesHardswish: true, stride: 1, dilation: 1),
            NFKRVMBlockSpec(input: 80, kernel: 3, expanded: 480, output: 112, usesSE: true, usesHardswish: true, stride: 1, dilation: 1),
            NFKRVMBlockSpec(input: 112, kernel: 3, expanded: 672, output: 112, usesSE: true, usesHardswish: true, stride: 1, dilation: 1),
            NFKRVMBlockSpec(input: 112, kernel: 5, expanded: 672, output: 160, usesSE: true, usesHardswish: true, stride: 2, dilation: 2),
            NFKRVMBlockSpec(input: 160, kernel: 5, expanded: 960, output: 160, usesSE: true, usesHardswish: true, stride: 1, dilation: 2),
            NFKRVMBlockSpec(input: 160, kernel: 5, expanded: 960, output: 160, usesSE: true, usesHardswish: true, stride: 1, dilation: 2),
        ],
        captures: [0, 2, 5],
        decoderChannels: [80, 40, 32, 16],
        refinerHiddenChannels: 16)

    /// The same structure at test scale, covering every block form (expand, SE, hardswish, dilation).
    public static let tiny = NFKMLXRVMConfiguration(
        stemChannels: 8, lastChannels: 32, asppChannels: 16,
        blocks: [
            NFKRVMBlockSpec(input: 8, kernel: 3, expanded: 8, output: 8, usesSE: false, usesHardswish: false, stride: 1, dilation: 1),
            NFKRVMBlockSpec(input: 8, kernel: 3, expanded: 16, output: 12, usesSE: false, usesHardswish: false, stride: 2, dilation: 1),
            NFKRVMBlockSpec(input: 12, kernel: 5, expanded: 24, output: 16, usesSE: true, usesHardswish: false, stride: 2, dilation: 1),
            NFKRVMBlockSpec(input: 16, kernel: 3, expanded: 32, output: 16, usesSE: true, usesHardswish: true, stride: 2, dilation: 1),
            NFKRVMBlockSpec(input: 16, kernel: 5, expanded: 32, output: 16, usesSE: true, usesHardswish: true, stride: 1, dilation: 2),
        ],
        captures: [0, 1, 2],
        decoderChannels: [8, 6, 4, 4],
        refinerHiddenChannels: 4)

    var featureChannels: [Int] { captures.map { blocks[$0].output } }
}

private func nfkHardsigmoid(_ x: MLXArray) -> MLXArray {
    clip((x + 3) / 6, min: 0, max: 1)
}

/// torchvision's channel rounding for the squeeze width: to the nearest multiple of eight, never
/// dropping below 90 percent of the request.
private func nfkDivisibleChannels(_ value: Int, _ divisor: Int = 8) -> Int {
    var rounded = max(divisor, (value + divisor / 2) / divisor * divisor)
    if Double(rounded) < 0.9 * Double(value) {
        rounded += divisor
    }
    return rounded
}

/// torchvision's `Conv2dNormActivation` without the parameter-free activation entry. The reference
/// stores these positionally (convolution at Sequential index 0, BatchNorm at 1); the keys here are
/// semantic because MLX's `update(parameters:)` parses a numeric key as an array index, and
/// `remapReferenceKey` translates the positions.
final class NFKRVMConvBN: Module {
    enum Activation { case none, relu, hardswish }

    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm
    let activation: Activation

    /// torchvision's MobileNetV3 normalizes at epsilon 1e-3, so that is the default here; the
    /// framework's 1e-5 rescales every block (the same defect NeMo's 1e-3 exposed in the VAD). The
    /// decoder's plain `nn.BatchNorm2d` stages pass the framework default back in.
    init(inChannels: Int, outChannels: Int, kernel: Int = 1, stride: Int = 1, dilation: Int = 1,
         groups: Int = 1, activation: Activation, bnEps: Float = 1e-3) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                    padding: IntOrPair((kernel - 1) / 2 * dilation),
                                    dilation: IntOrPair(dilation), groups: groups, bias: false)
        _bn.wrappedValue = BatchNorm(featureCount: outChannels, eps: bnEps)
        self.activation = activation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = bn(conv(x))
        switch activation {
        case .none: return normalized
        case .relu: return relu(normalized)
        case .hardswish: return hardSwish(normalized)
        }
    }
}

/// torchvision's `SqueezeExcitation`: global pool → `fc1` → ReLU → `fc2` → hardsigmoid gate.
final class NFKRVMSqueezeExcite: Module {
    @ModuleInfo(key: "fc1") var fc1: Conv2d
    @ModuleInfo(key: "fc2") var fc2: Conv2d

    init(channels: Int, squeeze: Int) {
        _fc1.wrappedValue = Conv2d(inputChannels: channels, outputChannels: squeeze, kernelSize: 1)
        _fc2.wrappedValue = Conv2d(inputChannels: squeeze, outputChannels: channels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pooled = mean(x, axes: [1, 2], keepDims: true)
        return x * nfkHardsigmoid(fc2(relu(fc1(pooled))))
    }
}

/// A MobileNetV3 inverted residual: optional 1×1 expansion, depthwise convolution at the block's
/// stride or dilation, optional squeeze-and-excitation, and a linear 1×1 projection. The residual
/// connects when the configured stride is one and the channel count is unchanged.
final class NFKRVMInvertedResidual: Module {
    @ModuleInfo(key: "expand") var expand: NFKRVMConvBN?
    @ModuleInfo(key: "dw") var depthwise: NFKRVMConvBN
    @ModuleInfo(key: "se") var se: NFKRVMSqueezeExcite?
    @ModuleInfo(key: "project") var project: NFKRVMConvBN
    let usesResidual: Bool

    init(_ spec: NFKRVMBlockSpec) {
        let activation: NFKRVMConvBN.Activation = spec.usesHardswish ? .hardswish : .relu
        if spec.expanded != spec.input {
            _expand.wrappedValue = NFKRVMConvBN(inChannels: spec.input, outChannels: spec.expanded,
                                                activation: activation)
        }
        // A dilated block keeps its resolution: the dilation replaces the stride, as torchvision does.
        _depthwise.wrappedValue = NFKRVMConvBN(inChannels: spec.expanded, outChannels: spec.expanded,
                                               kernel: spec.kernel,
                                               stride: spec.dilation > 1 ? 1 : spec.stride,
                                               dilation: spec.dilation, groups: spec.expanded,
                                               activation: activation)
        if spec.usesSE {
            _se.wrappedValue = NFKRVMSqueezeExcite(channels: spec.expanded,
                                                   squeeze: nfkDivisibleChannels(spec.expanded / 4))
        }
        _project.wrappedValue = NFKRVMConvBN(inChannels: spec.expanded, outChannels: spec.output,
                                             activation: .none)
        usesResidual = spec.stride == 1 && spec.input == spec.output
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = expand.map { $0(x) } ?? x
        out = depthwise(out)
        if let se {
            out = se(out)
        }
        out = project(out)
        return usesResidual ? x + out : out
    }
}

/// The MobileNetV3-Large encoder: stem, inverted residuals, and the final 1×1 expansion, emitting the
/// four skip features. ImageNet normalization happens here, as the reference encoder's forward does.
final class NFKRVMBackbone: Module {
    @ModuleInfo(key: "stem") var stem: NFKRVMConvBN
    @ModuleInfo(key: "blocks") var blocks: [NFKRVMInvertedResidual]
    @ModuleInfo(key: "last") var last: NFKRVMConvBN
    let captures: [Int]

    init(_ configuration: NFKMLXRVMConfiguration) {
        _stem.wrappedValue = NFKRVMConvBN(inChannels: 3, outChannels: configuration.stemChannels,
                                          kernel: 3, stride: 2, activation: .hardswish)
        _blocks.wrappedValue = configuration.blocks.map { NFKRVMInvertedResidual($0) }
        _last.wrappedValue = NFKRVMConvBN(inChannels: configuration.blocks.last!.output,
                                          outChannels: configuration.lastChannels,
                                          activation: .hardswish)
        captures = configuration.captures
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 1, 3])
        let deviation = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 1, 3])
        var out = stem((x - mean) / deviation)
        var skips = [MLXArray]()
        for (index, block) in blocks.enumerated() {
            out = block(out)
            if skips.count < captures.count, index == captures[skips.count] {
                skips.append(out)
            }
        }
        return (skips[0], skips[1], skips[2], last(out))
    }
}

/// One convolution stored at a Sequential index, for reference blocks whose other entries carry no
/// parameters.
final class NFKRVMSeqConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(inChannels: Int, outChannels: Int, kernel: Int = 3, padding: Int = 1, bias: Bool = true) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel), padding: IntOrPair(padding), bias: bias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// The reference LR-ASPP: a 1×1 + BatchNorm + ReLU branch gated by a globally pooled 1×1 sigmoid
/// branch (whose convolution sits at Sequential index 1, after the parameter-free pooling).
final class NFKRVMReferenceLRASPP: Module {
    final class PooledGate: Module {
        @ModuleInfo(key: "conv") var conv: Conv2d
        init(inChannels: Int, outChannels: Int) {
            _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                        kernelSize: 1, bias: false)
        }
    }

    @ModuleInfo(key: "aspp1") var aspp1: NFKRVMConvBN
    @ModuleInfo(key: "aspp2") var aspp2: PooledGate

    init(inChannels: Int, outChannels: Int) {
        _aspp1.wrappedValue = NFKRVMConvBN(inChannels: inChannels, outChannels: outChannels,
                                           activation: .relu, bnEps: 1e-5)
        _aspp2.wrappedValue = PooledGate(inChannels: inChannels, outChannels: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        aspp1(x) * sigmoid(aspp2.conv(mean(x, axes: [1, 2], keepDims: true)))
    }
}

/// The reference ConvGRU: one convolution produces the reset and update gates (reset first), a second
/// the candidate. The new hidden state is also the output.
final class NFKRVMReferenceGRU: Module {
    @ModuleInfo(key: "ih") var ih: NFKRVMSeqConv
    @ModuleInfo(key: "hh") var hh: NFKRVMSeqConv
    let channels: Int

    init(channels: Int) {
        _ih.wrappedValue = NFKRVMSeqConv(inChannels: channels * 2, outChannels: channels * 2)
        _hh.wrappedValue = NFKRVMSeqConv(inChannels: channels * 2, outChannels: channels)
        self.channels = channels
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> MLXArray {
        let hidden = state ?? MLXArray.zeros(x.shape)
        let gates = sigmoid(ih(concatenated([x, hidden], axis: 3)))
        let reset = gates[0..., 0..., 0..., 0 ..< channels]
        let update = gates[0..., 0..., 0..., channels ..< channels * 2]
        let candidate = tanh(hh(concatenated([x, reset * hidden], axis: 3)))
        return (1 - update) * hidden + update * candidate
    }
}

/// The decoder's bottleneck stage: the GRU runs on the second half of the channels, the first half
/// passes through.
final class NFKRVMBottleneckBlock: Module {
    @ModuleInfo(key: "gru") var gru: NFKRVMReferenceGRU
    let channels: Int

    init(channels: Int) {
        _gru.wrappedValue = NFKRVMReferenceGRU(channels: channels / 2)
        self.channels = channels
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let kept = x[0..., 0..., 0..., 0 ..< channels / 2]
        let recurrent = gru(x[0..., 0..., 0..., channels / 2 ..< channels], state: state)
        return (concatenated([kept, recurrent], axis: 3), recurrent)
    }
}

/// A decoder upsampling stage: ×2 bilinear upsample cropped to the skip size, fuse with the skip
/// feature and the pooled source, then the half-channel GRU.
final class NFKRVMUpsamplingBlock: Module {
    @ModuleInfo(key: "conv") var conv: NFKRVMConvBN
    @ModuleInfo(key: "gru") var gru: NFKRVMReferenceGRU
    let outChannels: Int

    init(inChannels: Int, skipChannels: Int, sourceChannels: Int, outChannels: Int) {
        _conv.wrappedValue = NFKRVMConvBN(inChannels: inChannels + skipChannels + sourceChannels,
                                          outChannels: outChannels, kernel: 3, activation: .relu,
                                          bnEps: 1e-5)
        _gru.wrappedValue = NFKRVMReferenceGRU(channels: outChannels / 2)
        self.outChannels = outChannels
    }

    func callAsFunction(_ x: MLXArray, skip: MLXArray, source: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let up = NFKMLXResample.resizeBilinear(x, height: x.shape[1] * 2, width: x.shape[2] * 2)
        let cropped = up[0..., 0 ..< skip.shape[1], 0 ..< skip.shape[2], 0...]
        let fused = conv(concatenated([cropped, skip, source], axis: 3))
        let kept = fused[0..., 0..., 0..., 0 ..< outChannels / 2]
        let recurrent = gru(fused[0..., 0..., 0..., outChannels / 2 ..< outChannels], state: state)
        return (concatenated([kept, recurrent], axis: 3), recurrent)
    }
}

/// The decoder's output stage: ×2 upsample to the source resolution, fuse with the source, and two
/// convolution + BatchNorm + ReLU rounds (Sequential indices 0/1 and 3/4).
final class NFKRVMOutputBlock: Module {
    final class Stack: Module {
        @ModuleInfo(key: "conv1") var conv1: Conv2d
        @ModuleInfo(key: "bn1") var bn1: BatchNorm
        @ModuleInfo(key: "conv2") var conv2: Conv2d
        @ModuleInfo(key: "bn2") var bn2: BatchNorm

        init(inChannels: Int, outChannels: Int) {
            _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                         kernelSize: 3, padding: 1, bias: false)
            _bn1.wrappedValue = BatchNorm(featureCount: outChannels)
            _conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels,
                                         kernelSize: 3, padding: 1, bias: false)
            _bn2.wrappedValue = BatchNorm(featureCount: outChannels)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            relu(bn2(conv2(relu(bn1(conv1(x))))))
        }
    }

    @ModuleInfo(key: "conv") var conv: Stack

    init(inChannels: Int, sourceChannels: Int, outChannels: Int) {
        _conv.wrappedValue = Stack(inChannels: inChannels + sourceChannels, outChannels: outChannels)
    }

    func callAsFunction(_ x: MLXArray, source: MLXArray) -> MLXArray {
        let up = NFKMLXResample.resizeBilinear(x, height: x.shape[1] * 2, width: x.shape[2] * 2)
        let cropped = up[0..., 0 ..< source.shape[1], 0 ..< source.shape[2], 0...]
        return conv(concatenated([cropped, source], axis: 3))
    }
}

/// The recurrent decoder: bottleneck at 1/16, three upsampling stages fusing skips with the pooled
/// source, and the output stage back at source resolution.
final class NFKRVMDecoder: Module {
    @ModuleInfo(key: "decode4") var decode4: NFKRVMBottleneckBlock
    @ModuleInfo(key: "decode3") var decode3: NFKRVMUpsamplingBlock
    @ModuleInfo(key: "decode2") var decode2: NFKRVMUpsamplingBlock
    @ModuleInfo(key: "decode1") var decode1: NFKRVMUpsamplingBlock
    @ModuleInfo(key: "decode0") var decode0: NFKRVMOutputBlock

    init(featureChannels: [Int], asppChannels: Int, decoderChannels: [Int]) {
        _decode4.wrappedValue = NFKRVMBottleneckBlock(channels: asppChannels)
        _decode3.wrappedValue = NFKRVMUpsamplingBlock(inChannels: asppChannels, skipChannels: featureChannels[2],
                                                      sourceChannels: 3, outChannels: decoderChannels[0])
        _decode2.wrappedValue = NFKRVMUpsamplingBlock(inChannels: decoderChannels[0], skipChannels: featureChannels[1],
                                                      sourceChannels: 3, outChannels: decoderChannels[1])
        _decode1.wrappedValue = NFKRVMUpsamplingBlock(inChannels: decoderChannels[1], skipChannels: featureChannels[0],
                                                      sourceChannels: 3, outChannels: decoderChannels[2])
        _decode0.wrappedValue = NFKRVMOutputBlock(inChannels: decoderChannels[2], sourceChannels: 3,
                                                  outChannels: decoderChannels[3])
    }

    /// The reference's `AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)`: replicating an odd
    /// edge before an exact 2×2 mean equals averaging only the real samples in the partial window.
    static func pooledHalf(_ x: MLXArray) -> MLXArray {
        var padded = x
        if x.shape[1] % 2 != 0 || x.shape[2] % 2 != 0 {
            padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((0, x.shape[1] % 2)),
                                            IntOrPair((0, x.shape[2] % 2)), IntOrPair(0)], mode: .edge)
        }
        return AvgPool2d(kernelSize: 2, stride: 2)(padded)
    }

    func callAsFunction(_ source: MLXArray, _ f1: MLXArray, _ f2: MLXArray, _ f3: MLXArray, _ f4: MLXArray,
                        state: [MLXArray?]) -> (MLXArray, [MLXArray]) {
        let s1 = Self.pooledHalf(source)
        let s2 = Self.pooledHalf(s1)
        let s3 = Self.pooledHalf(s2)
        let (x4, r4) = decode4(f4, state: state[3])
        let (x3, r3) = decode3(x4, skip: f3, source: s3, state: state[2])
        let (x2, r2) = decode2(x3, skip: f2, source: s2, state: state[1])
        let (x1, r1) = decode1(x2, skip: f1, source: s1, state: state[0])
        let hidden = decode0(x1, source: source)
        return (hidden, [r1, r2, r3, r4])
    }
}

/// The 1×1 output projection the reference wraps in a `Projection` module.
final class NFKRVMProjection: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(inChannels: Int, outChannels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// The deep guided filter: box-filter statistics of the low-resolution result against the
/// low-resolution source, a learned per-pixel gain from those statistics plus the decoder's hidden
/// features, and a bilinear lift of gain and offset to the full frame.
final class NFKRVMRefiner: Module {
    final class Head: Module {
        @ModuleInfo(key: "conv1") var conv1: Conv2d
        @ModuleInfo(key: "bn1") var bn1: BatchNorm
        @ModuleInfo(key: "conv2") var conv2: Conv2d
        @ModuleInfo(key: "bn2") var bn2: BatchNorm
        @ModuleInfo(key: "conv3") var conv3: Conv2d

        init(hiddenChannels: Int) {
            _conv1.wrappedValue = Conv2d(inputChannels: 8 + hiddenChannels, outputChannels: hiddenChannels,
                                         kernelSize: 1, bias: false)
            _bn1.wrappedValue = BatchNorm(featureCount: hiddenChannels)
            _conv2.wrappedValue = Conv2d(inputChannels: hiddenChannels, outputChannels: hiddenChannels,
                                         kernelSize: 1, bias: false)
            _bn2.wrappedValue = BatchNorm(featureCount: hiddenChannels)
            _conv3.wrappedValue = Conv2d(inputChannels: hiddenChannels, outputChannels: 4, kernelSize: 1)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            conv3(relu(bn2(conv2(relu(bn1(conv1(x)))))))
        }
    }

    @ModuleInfo(key: "box_filter") var boxFilter: Conv2d
    @ModuleInfo(key: "conv") var conv: Head

    init(hiddenChannels: Int) {
        _boxFilter.wrappedValue = Conv2d(inputChannels: 4, outputChannels: 4, kernelSize: 3,
                                         padding: 1, groups: 4, bias: false)
        _conv.wrappedValue = Head(hiddenChannels: hiddenChannels)
    }

    func callAsFunction(fineSource: MLXArray, baseSource: MLXArray, baseForeground: MLXArray,
                        baseAlpha: MLXArray, baseHidden: MLXArray) -> (MLXArray, MLXArray) {
        let fineX = concatenated([fineSource, mean(fineSource, axes: [3], keepDims: true)], axis: 3)
        let baseX = concatenated([baseSource, mean(baseSource, axes: [3], keepDims: true)], axis: 3)
        let baseY = concatenated([baseForeground, baseAlpha], axis: 3)

        let meanX = boxFilter(baseX)
        let meanY = boxFilter(baseY)
        let covarianceXY = boxFilter(baseX * baseY) - meanX * meanY
        let varianceX = boxFilter(baseX * baseX) - meanX * meanX

        let gain = conv(concatenated([covarianceXY, varianceX, baseHidden], axis: 3))
        let offset = meanY - gain * meanX

        let (height, width) = (fineSource.shape[1], fineSource.shape[2])
        let liftedGain = NFKMLXResample.resizeBilinear(gain, height: height, width: width)
        let liftedOffset = NFKMLXResample.resizeBilinear(offset, height: height, width: width)
        let out = liftedGain * fineX + liftedOffset
        return (out[0..., 0..., 0..., 0 ..< 3], out[0..., 0..., 0..., 3 ..< 4])
    }
}

/// The Robust Video Matting network: encoder, LR-ASPP, recurrent decoder, output projections, and the
/// deep-guided-filter refiner.
final class NFKMLXRVMNet: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKRVMBackbone
    @ModuleInfo(key: "aspp") var aspp: NFKRVMReferenceLRASPP
    @ModuleInfo(key: "decoder") var decoder: NFKRVMDecoder
    @ModuleInfo(key: "project_mat") var projectMat: NFKRVMProjection
    @ModuleInfo(key: "project_seg") var projectSeg: NFKRVMProjection
    @ModuleInfo(key: "refiner") var refiner: NFKRVMRefiner

    init(_ configuration: NFKMLXRVMConfiguration = .large) {
        _backbone.wrappedValue = NFKRVMBackbone(configuration)
        _aspp.wrappedValue = NFKRVMReferenceLRASPP(inChannels: configuration.lastChannels,
                                                   outChannels: configuration.asppChannels)
        _decoder.wrappedValue = NFKRVMDecoder(featureChannels: configuration.featureChannels,
                                              asppChannels: configuration.asppChannels,
                                              decoderChannels: configuration.decoderChannels)
        _projectMat.wrappedValue = NFKRVMProjection(inChannels: configuration.decoderChannels[3], outChannels: 4)
        _projectSeg.wrappedValue = NFKRVMProjection(inChannels: configuration.decoderChannels[3], outChannels: 1)
        _refiner.wrappedValue = NFKRVMRefiner(hiddenChannels: configuration.refinerHiddenChannels)
    }

    /// The recurrent state carried between frames: one hidden tensor per decoder GRU, ordered
    /// `[r1, r2, r3, r4]` as the reference returns them (fine to coarse).
    typealias State = [MLXArray?]

    static var initialState: State { [nil, nil, nil, nil] }

    /// Mattes one frame `[1, H, W, 3]` (`0...1`), threading and returning the recurrent state. Returns
    /// the straight foreground `[1, H, W, 3]`, the alpha `[1, H, W, 1]`, and the next state.
    ///
    /// A `downsampleRatio` below one runs the network on a bilinearly reduced frame and lifts the
    /// result back through the deep guided filter — the reference's high-resolution recipe.
    func forward(_ frame: MLXArray, state: State, downsampleRatio: Float = 1) -> (foreground: MLXArray, alpha: MLXArray, state: State) {
        let source = frame
        let base: MLXArray
        if downsampleRatio != 1 {
            base = NFKMLXResample.resizeBilinear(source,
                                                 height: Int(Float(source.shape[1]) * downsampleRatio),
                                                 width: Int(Float(source.shape[2]) * downsampleRatio))
        } else {
            base = source
        }

        let (f1, f2, f3, encoded) = backbone(base)
        let f4 = aspp(encoded)
        let (hidden, nextState) = decoder(base, f1, f2, f3, f4, state: state)

        let projected = projectMat(hidden)
        var foregroundResidual = projected[0..., 0..., 0..., 0 ..< 3]
        var alpha = projected[0..., 0..., 0..., 3 ..< 4]
        if downsampleRatio != 1 {
            (foregroundResidual, alpha) = refiner(fineSource: source, baseSource: base,
                                                  baseForeground: foregroundResidual, baseAlpha: alpha,
                                                  baseHidden: hidden)
        }
        let foreground = clip(foregroundResidual + source, min: 0, max: 1)
        return (foreground, clip(alpha, min: 0, max: 1), nextState.map { Optional($0) })
    }

    /// Mattes a single bridged image `[H, W, 3]` with no temporal context, returning straight
    /// foreground plus alpha as `[H, W, 4]`.
    func matte(_ image: MLXArray) -> MLXArray {
        let batched = image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]])
        let (foreground, alpha, _) = forward(batched, state: Self.initialState)
        return concatenated([foreground, alpha], axis: 3).reshaped([image.shape[0], image.shape[1], 4])
    }
}

// MARK: - Shared separable building blocks
//
// These predate the reference-faithful RVM network above and remain because MODNet's placeholder
// encoder is built from `NFKRVMSeparableBlock` and `NFKRVMLRASPP`. They go when MODNet gets its real
// MobileNetV2 backbone and e-ASPP.

/// A depthwise-separable convolution block with an optional stride: a depthwise 3×3 followed by a
/// pointwise 1×1, each with ReLU.
final class NFKRVMSeparableBlock: Module {
    @ModuleInfo(key: "depthwise") var depthwise: Conv2d
    @ModuleInfo(key: "pointwise") var pointwise: Conv2d

    init(inChannels: Int, outChannels: Int, stride: Int) {
        _depthwise.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: inChannels, kernelSize: 3,
                                         stride: IntOrPair(stride), padding: 1, groups: inChannels)
        _pointwise.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(pointwise(relu(depthwise(x))))
    }
}

/// An LR-ASPP context module in compact form: a 1×1 projection modulated by a global-context gate.
final class NFKRVMLRASPP: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "scale") var scale: Conv2d

    init(inChannels: Int, outChannels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
        _scale.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pooled = mean(x, axes: [1, 2], keepDims: true)          // global average pool over H, W
        return relu(conv(x)) * sigmoid(scale(pooled))
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKRVMHolder: @unchecked Sendable {
    let net: NFKMLXRVMNet
    init(_ net: NFKMLXRVMNet) { self.net = net }
}

/// Robust Video Matting as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXRVMNet` is the reference recurrent-matting network. Random weights run (proving the
/// pipeline); the released `rvm_mobilenetv3` checkpoint, converted to **safetensors**, makes the matte
/// accurate. The matting backend mattes a single image (no temporal state); a video consumer threads
/// the recurrent state frame by frame through `NFKMLXRVMNet.forward`.
@objc(NFKMLXRVM)
public final class NFKMLXRVM: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "robust-video-matting"

    /// Builds a matting backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXRVMNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        net.train(false)
        let holder = NFKRVMHolder(net)
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

    /// Registers Robust Video Matting (`robust-video-matting`) with `NFKMLXModelRegistry`, delegating
    /// to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Maps the reference's positional Sequential names onto the module's semantic ones (MLX's
    /// `update(parameters:)` parses a numeric key as an array index, so positions cannot be kept).
    /// In the backbone, what `block.M` holds depends on the block's shape: the expansion exists only
    /// when the expanded width differs from the input, and squeeze-and-excitation only where
    /// configured. Every positional rule is scoped by its prefix — a global digit rewrite would eat
    /// unrelated names (the SAM remap's lesson).
    static func remapReferenceKey(_ key: String, blocks: [NFKRVMBlockSpec]) -> String {
        /// Rewrites the position after `prefix` through `names` (nil entries are parameter-free
        /// Sequential slots that never appear in a checkpoint).
        func positional(_ key: String, after prefix: String, names: [String?]) -> String? {
            guard key.hasPrefix(prefix) else { return nil }
            let sub = key.dropFirst(prefix.count)
            guard let dot = sub.firstIndex(of: "."), let position = Int(sub[..<dot]),
                  position < names.count, let name = names[position] else { return nil }
            return prefix + name + String(sub[dot...])
        }

        let convBN: [String?] = ["conv", "bn"]
        if key.hasPrefix("backbone.features.") {
            let rest = key.dropFirst("backbone.features.".count)
            guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]) else { return key }
            let remainder = String(rest[rest.index(after: dot)...])
            if index == 0 {
                return "backbone.stem." + (positional(remainder, after: "", names: convBN) ?? remainder)
            }
            if index == blocks.count + 1 {
                return "backbone.last." + (positional(remainder, after: "", names: convBN) ?? remainder)
            }
            guard remainder.hasPrefix("block.") else { return key }
            let sub = remainder.dropFirst("block.".count)
            guard let subDot = sub.firstIndex(of: "."), let position = Int(sub[..<subDot]) else { return key }
            let spec = blocks[index - 1]
            var names = spec.expanded != spec.input ? ["expand"] : []
            names.append("dw")
            if spec.usesSE {
                names.append("se")
            }
            names.append("project")
            let tail = String(sub[sub.index(after: subDot)...])
            // The squeeze module's `fc1`/`fc2` names already match; the ConvBN stages are positional.
            let mapped = names[position] == "se" ? tail : (positional(tail, after: "", names: convBN) ?? tail)
            return "backbone.blocks.\(index - 1).\(names[position])." + mapped
        }

        if let mapped = positional(key, after: "aspp.aspp1.", names: convBN) { return mapped }
        if let mapped = positional(key, after: "aspp.aspp2.", names: [nil, "conv"]) { return mapped }
        for stage in ["decode3", "decode2", "decode1"] {
            if let mapped = positional(key, after: "decoder.\(stage).conv.", names: convBN) { return mapped }
        }
        if let mapped = positional(key, after: "decoder.decode0.conv.",
                                   names: ["conv1", "bn1", nil, "conv2", "bn2"]) { return mapped }
        for stage in ["decode4", "decode3", "decode2", "decode1"] {
            for gate in ["ih", "hh"] {
                if let mapped = positional(key, after: "decoder.\(stage).gru.\(gate).", names: ["conv"]) { return mapped }
            }
        }
        if let mapped = positional(key, after: "refiner.conv.",
                                   names: ["conv1", "bn1", nil, "conv2", "bn2", nil, "conv3"]) { return mapped }
        return key
    }

    /// Loads a safetensors checkpoint into `net`, remapping the reference's positional backbone names
    /// and transposing 4-D convolution weights from PyTorch's `[out, in, kH, kW]` to MLX's
    /// channels-last `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXRVMNet, from url: URL,
                            blocks: [NFKRVMBlockSpec] = NFKMLXRVMConfiguration.large.blocks) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remapReferenceKey(key, blocks: blocks), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
