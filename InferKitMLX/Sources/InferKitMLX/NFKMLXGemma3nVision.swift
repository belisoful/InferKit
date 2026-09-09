//
//  NFKMLXGemma3nVision.swift
//  InferKitMLX
//
//  The Gemma 3n vision tower: MobileNetV5-300M, a convolutional encoder rather than the SigLIP
//  transformer every other vision model here carries. It reads a 768×768 frame and produces a 16×16
//  grid of 2048-wide features, which become the 256 soft tokens the decoder splices into its prompt.
//
//  The release reaches it through `timm` rather than through transformers, so the reference is timm's
//  own `MobileNetV5Encoder`. Four stages of mobile blocks — edge residuals, universal inverted
//  residuals, and multi-QUERY attention over the feature map — feed a multi-scale fusion adapter that
//  joins the last two stages.
//
//  Five things are load-bearing and none of them shows in a shape.
//
//  - **Padding is TensorFlow's `SAME`, which is ASYMMETRIC at stride two.** A 3×3 stride-2 convolution
//    pads (0, 1) and a 5×5 pads (1, 2). Symmetric padding gives the same output size and shifts every
//    pixel.
//  - **There is no BatchNorm anywhere.** Every normalization is an RMS norm over the channel axis with
//    a weight and no bias, no running statistics, and no train/eval difference. The checkpoint's `bn`
//    names are legacy.
//  - **The activation is the tanh-approximate GELU**, not the exact error-function form.
//  - **The fusion concatenates the coarse stage AFTER the fine one** and upsamples it by nearest
//    neighbor; the order is invisible in the `[3840, 1920, 1, 1]` weight it feeds.
//  - **The frame is scaled to `0...1` and NOT normalized further**, though the release's preprocessor
//    file states a mean and a standard deviation. It sets `do_normalize` false.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// MARK: - Pieces

/// The tower's normalization: an RMS norm over the channel axis with a weight and no bias.
///
/// @discussion The checkpoint calls these `bn`, which is a legacy name; there are no running
/// statistics anywhere in the tower. The reference does NOT widen to float32, so this does not either.
final class NFKGemma3nVisionNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float
    let activated: Bool

    init(channels: Int, eps: Float = 1e-6, activated: Bool) {
        _weight.wrappedValue = MLXArray.ones([channels])
        epsilon = eps
        self.activated = activated
        super.init()
    }

    /// `x` is `[batch, height, width, channels]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = x * rsqrt((x * x).mean(axis: -1, keepDims: true) + epsilon) * weight
        return activated ? geluApproximate(normalized) : normalized
    }
}

/// A convolution under TensorFlow's `SAME` padding, which is asymmetric wherever the stride divides
/// the input unevenly.
final class NFKGemma3nVisionConv: Module {
    @ModuleInfo(key: "conv") var convolution: Conv2d
    let kernel: Int
    let stride: Int

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int, stride: Int, groups: Int = 1,
         bias: Bool = false) {
        self.kernel = kernel
        self.stride = stride
        _convolution.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                           kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                           padding: IntOrPair(0), groups: groups, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convolution(NFKGemma3nVisionPadding.same(x, kernel: kernel, stride: stride))
    }
}

/// A convolution and its normalization, which the checkpoint keys as `conv` and `bn`.
final class NFKGemma3nVisionConvNorm: Module {
    @ModuleInfo(key: "conv") var convolution: Conv2d
    @ModuleInfo(key: "bn") var norm: NFKGemma3nVisionNorm
    let kernel: Int
    let stride: Int

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int, stride: Int, groups: Int = 1,
         bias: Bool = false, activated: Bool) {
        self.kernel = kernel
        self.stride = stride
        _convolution.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                           kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                           padding: IntOrPair(0), groups: groups, bias: bias)
        _norm.wrappedValue = NFKGemma3nVisionNorm(channels: outChannels, activated: activated)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(convolution(NFKGemma3nVisionPadding.same(x, kernel: kernel, stride: stride)))
    }
}

/// TensorFlow's `SAME` padding, computed from the input size rather than assumed.
///
/// @discussion At stride two the total padding is odd, and the reference puts the smaller half FIRST:
/// a 3×3 stride-2 convolution over an even input pads (0, 1), a 5×5 pads (1, 2). A symmetric pad of
/// one or two produces the same output size and a shifted picture, so nothing about the shapes would
/// reveal the mistake.
enum NFKGemma3nVisionPadding {
    static func widths(input: Int, kernel: Int, stride: Int) -> (before: Int, after: Int) {
        let output = (input + stride - 1) / stride
        let total = Swift.max((output - 1) * stride + kernel - input, 0)
        return (total / 2, total - total / 2)
    }

    static func same(_ x: MLXArray, kernel: Int, stride: Int) -> MLXArray {
        guard kernel > 1 else { return x }
        let vertical = widths(input: x.shape[1], kernel: kernel, stride: stride)
        let horizontal = widths(input: x.shape[2], kernel: kernel, stride: stride)
        guard vertical != (0, 0) || horizontal != (0, 0) else { return x }
        return padded(x, widths: [IntOrPair(0), IntOrPair((vertical.before, vertical.after)),
                                  IntOrPair((horizontal.before, horizontal.after)), IntOrPair(0)])
    }
}

/// The learned per-channel scale each residual block applies before it rejoins its input.
final class NFKGemma3nVisionLayerScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(channels: Int) {
        _gamma.wrappedValue = MLXArray.ones([channels]) * 1e-5
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

// MARK: - Blocks

/// The common face of the tower's blocks, so one stage can hold a mix of them.
class NFKGemma3nVisionBlock: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

/// An edge residual: a full convolution that expands, then a pointwise projection back.
///
/// It carries no layer scale, which is what separates it from the inverted residuals around it.
final class NFKGemma3nVisionEdgeResidual: NFKGemma3nVisionBlock {
    @ModuleInfo(key: "conv_exp") var expand: Conv2d
    @ModuleInfo(key: "bn1") var expandNorm: NFKGemma3nVisionNorm
    @ModuleInfo(key: "conv_pwl") var project: Conv2d
    @ModuleInfo(key: "bn2") var projectNorm: NFKGemma3nVisionNorm

    let kernel: Int
    let stride: Int
    let skip: Bool

    init(inChannels: Int, midChannels: Int, outChannels: Int, kernel: Int, stride: Int, skip: Bool) {
        self.kernel = kernel
        self.stride = stride
        self.skip = skip
        _expand.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: midChannels,
                                      kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                      padding: IntOrPair(0), bias: false)
        _expandNorm.wrappedValue = NFKGemma3nVisionNorm(channels: midChannels, activated: true)
        _project.wrappedValue = Conv2d(inputChannels: midChannels, outputChannels: outChannels,
                                       kernelSize: IntOrPair(1), bias: false)
        _projectNorm.wrappedValue = NFKGemma3nVisionNorm(channels: outChannels, activated: false)
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = NFKGemma3nVisionPadding.same(x, kernel: kernel, stride: stride)
        let projected = projectNorm(project(expandNorm(expand(padded))))
        return skip ? projected + x : projected
    }
}

/// A universal inverted residual: an optional depthwise convolution, a pointwise expansion, an
/// optional second depthwise convolution, and a pointwise projection.
///
/// @discussion The order is the trap. The FIRST depthwise convolution runs BEFORE the expansion and
/// carries NO activation; the second runs after it and does. Where the block downsamples, the stride
/// sits on the second depthwise convolution, not the first.
final class NFKGemma3nVisionInvertedResidual: NFKGemma3nVisionBlock {
    @ModuleInfo(key: "dw_start") var depthwiseStart: NFKGemma3nVisionConvNorm?
    @ModuleInfo(key: "pw_exp") var expand: NFKGemma3nVisionConvNorm
    @ModuleInfo(key: "dw_mid") var depthwiseMiddle: NFKGemma3nVisionConvNorm?
    @ModuleInfo(key: "pw_proj") var project: NFKGemma3nVisionConvNorm
    @ModuleInfo(key: "layer_scale") var layerScale: NFKGemma3nVisionLayerScale?

    let skip: Bool

    init(inChannels: Int, midChannels: Int, outChannels: Int,
         startKernel: Int?, middleKernel: Int?, stride: Int, skip: Bool, scaled: Bool = true) {
        self.skip = skip
        if let startKernel {
            _depthwiseStart.wrappedValue = NFKGemma3nVisionConvNorm(
                inChannels, inChannels, kernel: startKernel,
                stride: middleKernel == nil ? stride : 1, groups: inChannels, activated: false)
        }
        _expand.wrappedValue = NFKGemma3nVisionConvNorm(inChannels, midChannels, kernel: 1, stride: 1,
                                                        activated: true)
        if let middleKernel {
            _depthwiseMiddle.wrappedValue = NFKGemma3nVisionConvNorm(
                midChannels, midChannels, kernel: middleKernel, stride: stride, groups: midChannels,
                activated: true)
        }
        _project.wrappedValue = NFKGemma3nVisionConvNorm(midChannels, outChannels, kernel: 1, stride: 1,
                                                         activated: false)
        if scaled {
            _layerScale.wrappedValue = NFKGemma3nVisionLayerScale(channels: outChannels)
        }
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        if let depthwiseStart { hidden = depthwiseStart(hidden) }
        hidden = expand(hidden)
        if let depthwiseMiddle { hidden = depthwiseMiddle(hidden) }
        hidden = project(hidden)
        // The scale runs whether or not there is a residual to rejoin.
        if let layerScale { hidden = layerScale(hidden) }
        return skip ? hidden + x : hidden
    }
}

/// Multi-QUERY attention over a feature map: many query heads share ONE key head and ONE value head.
///
/// @discussion There is no positional embedding of any kind. The depthwise convolutions elsewhere in
/// the tower are what carry position. Where the keys and values are strided, they are downsampled by
/// a depthwise convolution and normalized BEFORE their pointwise projection.
final class NFKGemma3nVisionAttention: NFKGemma3nVisionBlock {
    @ModuleInfo(key: "norm") var norm: NFKGemma3nVisionNorm
    @ModuleInfo(key: "attn") var attention: NFKGemma3nVisionMultiQuery
    @ModuleInfo(key: "layer_scale") var layerScale: NFKGemma3nVisionLayerScale

    init(dimensions: Int, heads: Int, keyDimensions: Int, keyValueStride: Int) {
        _norm.wrappedValue = NFKGemma3nVisionNorm(channels: dimensions, activated: false)
        _attention.wrappedValue = NFKGemma3nVisionMultiQuery(
            dimensions: dimensions, heads: heads, keyDimensions: keyDimensions,
            keyValueStride: keyValueStride)
        _layerScale.wrappedValue = NFKGemma3nVisionLayerScale(channels: dimensions)
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + layerScale(attention(norm(x)))
    }
}

/// One key and value head shared across every query head.
final class NFKGemma3nVisionMultiQuery: Module {
    @ModuleInfo(key: "query") var query: NFKGemma3nVisionProjection
    @ModuleInfo(key: "key") var key: NFKGemma3nVisionProjection
    @ModuleInfo(key: "value") var value: NFKGemma3nVisionProjection
    @ModuleInfo(key: "output") var output: NFKGemma3nVisionProjection

    let heads: Int
    let keyDimensions: Int
    let scale: Float

    init(dimensions: Int, heads: Int, keyDimensions: Int, keyValueStride: Int) {
        self.heads = heads
        self.keyDimensions = keyDimensions
        scale = Float(pow(Double(keyDimensions), -0.5))
        _query.wrappedValue = NFKGemma3nVisionProjection(dimensions, heads * keyDimensions, stride: 1)
        _key.wrappedValue = NFKGemma3nVisionProjection(dimensions, keyDimensions, stride: keyValueStride)
        _value.wrappedValue = NFKGemma3nVisionProjection(dimensions, keyDimensions, stride: keyValueStride)
        _output.wrappedValue = NFKGemma3nVisionProjection(heads * keyDimensions, dimensions, stride: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, height, width) = (x.shape[0], x.shape[1], x.shape[2])
        let queryCount = height * width

        // The channel axis is head-major: channel = head · keyDimensions + d.
        let queries = query(x).reshaped([batch, queryCount, heads, keyDimensions]).transposed(0, 2, 1, 3)
        let keyMap = key(x)
        let keys = keyMap.reshaped([batch, 1, keyMap.shape[1] * keyMap.shape[2], keyDimensions])
        let valueMap = value(x)
        let values = valueMap.reshaped([batch, 1, valueMap.shape[1] * valueMap.shape[2], keyDimensions])

        // One key head serves every query head, so the fused call broadcasts it.
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: nil)
        return output(attended.transposed(0, 2, 1, 3).reshaped([batch, height, width, heads * keyDimensions]))
    }
}

/// One of the attention's four projections: an optional strided depthwise downsample and its
/// normalization, then a pointwise projection.
final class NFKGemma3nVisionProjection: Module {
    // The checkpoint keys this as a bare convolution, so it is one here rather than the conv-and-norm
    // pair the rest of the tower uses; its normalization is a separate sibling.
    @ModuleInfo(key: "down_conv") var downsample: Conv2d?
    @ModuleInfo(key: "norm") var norm: NFKGemma3nVisionNorm?
    @ModuleInfo(key: "proj") var projection: Conv2d

    let downsampleStride: Int

    init(_ inChannels: Int, _ outChannels: Int, stride: Int) {
        downsampleStride = stride
        if stride > 1 {
            _downsample.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: inChannels,
                                              kernelSize: IntOrPair(3), stride: IntOrPair(stride),
                                              padding: IntOrPair(0), groups: inChannels, bias: false)
            _norm.wrappedValue = NFKGemma3nVisionNorm(channels: inChannels, activated: false)
        }
        _projection.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                          kernelSize: IntOrPair(1), bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        if let downsample {
            hidden = downsample(NFKGemma3nVisionPadding.same(hidden, kernel: 3, stride: downsampleStride))
        }
        if let norm { hidden = norm(hidden) }
        return projection(hidden)
    }
}

/// The multi-scale fusion adapter: the last two stages joined, mixed, pooled to the token grid, and
/// normalized.
final class NFKGemma3nVisionFusion: Module {
    @ModuleInfo(key: "ffn") var feedForward: NFKGemma3nVisionInvertedResidual
    @ModuleInfo(key: "norm") var norm: NFKGemma3nVisionNorm

    let outputResolution: Int

    init(inChannels: Int, outChannels: Int, expansion: Int, resolution: Int) {
        outputResolution = resolution
        // The fusion's own feed-forward carries NO layer scale, where every other inverted residual
        // in the tower does.
        _feedForward.wrappedValue = NFKGemma3nVisionInvertedResidual(
            inChannels: inChannels, midChannels: expansion, outChannels: outChannels,
            startKernel: nil, middleKernel: nil, stride: 1, skip: false, scaled: false)
        _norm.wrappedValue = NFKGemma3nVisionNorm(channels: outChannels, activated: false)
        super.init()
    }

    /// `inputs` are the stages in order, FINE first. The coarse ones are lifted to the fine one's
    /// size by nearest neighbor and concatenated behind it.
    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        let target = (inputs[0].shape[1], inputs[0].shape[2])
        let resized = inputs.map { NFKGemma3nVisionResample.nearest($0, height: target.0, width: target.1) }
        let mixed = feedForward(concatenated(resized, axis: -1))
        return norm(NFKGemma3nVisionResample.averagePooled(mixed, to: outputResolution))
    }
}

/// Nearest-neighbor resizing and the exact average pooling the fusion ends with.
enum NFKGemma3nVisionResample {
    static func nearest(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        guard x.shape[1] != height || x.shape[2] != width else { return x }
        let rows = (0 ..< height).map { Int32($0 * x.shape[1] / height) }
        let columns = (0 ..< width).map { Int32($0 * x.shape[2] / width) }
        return x.take(MLXArray(rows), axis: 1).take(MLXArray(columns), axis: 2)
    }

    /// A whole-number average pool, which is what the reference takes when the side divides evenly.
    static func averagePooled(_ x: MLXArray, to resolution: Int) -> MLXArray {
        let (batch, height, width, channels) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        guard height % resolution == 0, width % resolution == 0 else {
            return NFKGemma3nVisionResample.nearest(x, height: resolution, width: resolution)
        }
        let (rows, columns) = (height / resolution, width / resolution)
        return x.reshaped([batch, resolution, rows, resolution, columns, channels])
            .mean(axes: [2, 4])
    }
}

// MARK: - The tower

/// The Gemma 3n vision tower.
///
/// @discussion The geometry is the released `mobilenetv5_300m_enc` and is written out rather than
/// derived from an architecture string: the release ships one size, and a table that can be read is
/// worth more here than a parser for a notation nothing else in this package uses.
public final class NFKMLXGemma3nVisionNet: Module {
    @ModuleInfo(key: "conv_stem") var stem: NFKGemma3nVisionConvNorm
    @ModuleInfo(key: "blocks") var blocks: [[NFKGemma3nVisionBlock]]
    @ModuleInfo(key: "msfa") var fusion: NFKGemma3nVisionFusion

    /// The side of the token grid the tower produces, so `resolution²` soft tokens.
    public let tokenGrid: Int
    /// The width of each soft token.
    public let outputChannels: Int

    public init(tokenGrid: Int = 16, outputChannels: Int = 2048) {
        self.tokenGrid = tokenGrid
        self.outputChannels = outputChannels
        _stem.wrappedValue = NFKGemma3nVisionConvNorm(3, 64, kernel: 3, stride: 2, bias: true,
                                                      activated: true)

        func inverted(_ inChannels: Int, _ mid: Int, _ out: Int, start: Int? = nil, middle: Int? = nil,
                      stride: Int = 1, skip: Bool) -> NFKGemma3nVisionBlock {
            NFKGemma3nVisionInvertedResidual(inChannels: inChannels, midChannels: mid,
                                             outChannels: out, startKernel: start,
                                             middleKernel: middle, stride: stride, skip: skip)
        }

        var stages = [[NFKGemma3nVisionBlock]]()

        // Stage 0: edge residuals, which carry no layer scale.
        var stage: [NFKGemma3nVisionBlock] = [
            NFKGemma3nVisionEdgeResidual(inChannels: 64, midChannels: 256, outChannels: 128,
                                         kernel: 3, stride: 2, skip: false)
        ]
        for _ in 0 ..< 2 {
            stage.append(NFKGemma3nVisionEdgeResidual(inChannels: 128, midChannels: 512,
                                                      outChannels: 128, kernel: 3, stride: 1, skip: true))
        }
        stages.append(stage)

        // Stage 1: inverted residuals whose leading depthwise kernel alternates 5 and 3.
        stage = [inverted(128, 768, 256, start: 3, middle: 5, stride: 2, skip: false)]
        for index in 0 ..< 4 {
            stage.append(inverted(256, 1024, 256, start: index % 2 == 0 ? 5 : 3, skip: true))
        }
        stages.append(stage)

        // Stage 2: a downsample, seven depthwise blocks, one pointwise block at expansion one, then
        // attention alternating with a pointwise feed-forward.
        stage = [inverted(256, 1536, 640, start: 5, middle: 5, stride: 2, skip: false)]
        for _ in 0 ..< 7 { stage.append(inverted(640, 2560, 640, start: 5, skip: true)) }
        stage.append(inverted(640, 640, 640, skip: true))
        for _ in 0 ..< 14 {
            stage.append(NFKGemma3nVisionAttention(dimensions: 640, heads: 12, keyDimensions: 64,
                                                   keyValueStride: 2))
            stage.append(inverted(640, 1280, 640, skip: true))
        }
        stages.append(stage)

        // Stage 3: the same shape at twice the width, with the keys and values no longer strided.
        stage = [inverted(640, 3840, 1280, start: 5, middle: 5, stride: 2, skip: false)]
        for _ in 0 ..< 19 {
            stage.append(NFKGemma3nVisionAttention(dimensions: 1280, heads: 16, keyDimensions: 96,
                                                   keyValueStride: 1))
            stage.append(inverted(1280, 2560, 1280, skip: true))
        }
        stages.append(stage)

        _blocks.wrappedValue = stages
        _fusion.wrappedValue = NFKGemma3nVisionFusion(inChannels: 640 + 1280, outChannels: outputChannels,
                                                      expansion: 3840, resolution: tokenGrid)
        super.init()
    }

    /// The token grid for a frame `[batch, height, width, 3]` in `0...1`, as
    /// `[batch, tokenGrid, tokenGrid, outputChannels]`.
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        var hidden = stem(pixels)
        var captured = [MLXArray]()
        for (index, stage) in blocks.enumerated() {
            for block in stage {
                hidden = block(hidden)
            }
            // The fusion reads the last two stages, the finer one first.
            if index >= blocks.count - 2 { captured.append(hidden) }
        }
        return fusion(captured)
    }

    /// The soft tokens a decoder reads: the grid flattened row-major to `[batch, tokens, channels]`.
    public func softTokens(_ pixels: MLXArray) -> MLXArray {
        let grid = self(pixels)
        return grid.reshaped([grid.shape[0], grid.shape[1] * grid.shape[2], grid.shape[3]])
    }
}

// MARK: - Building from a release

/// Building a Gemma 3n vision tower and loading its weights.
@objc(NFKMLXGemma3nVision)
public final class NFKMLXGemma3nVision: NSObject {

    static func makeNet() -> NFKMLXGemma3nVisionNet { NFKMLXGemma3nVisionNet() }

    /// The tower's module key for a checkpoint key, or nil for a tensor that is not the tower's.
    static func towerName(of key: String) -> String? {
        stripped(key, prefixes: ["model.vision_tower.timm_model.", "vision_tower.timm_model.",
                                 "timm_model."])
    }

    static func stripped(_ key: String, prefixes: [String]) -> String? {
        for prefix in prefixes where key.hasPrefix(prefix) {
            return String(key.dropFirst(prefix.count))
        }
        return nil
    }

    /// A checkpoint tensor in MLX's layout. Every convolution here is 4-D and moves its input-channel
    /// axis to the end; a depthwise convolution's `[channels, 1, k, k]` takes the same treatment.
    static func converted(_ name: String, _ value: MLXArray) -> MLXArray {
        value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
    }

    /// Loads the tower from a released directory, taking only the vision tower's tensors.
    static func loadWeights(into net: NFKMLXGemma3nVisionNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision,
                                                     remap: towerName(of:))
        try NFKMLXWeights.apply(mapped.map { ($0.0, converted($0.0, $0.1)) }, to: net)
    }
}
