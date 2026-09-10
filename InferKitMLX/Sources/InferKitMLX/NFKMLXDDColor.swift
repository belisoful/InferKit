//
//  NFKMLXDDColor.swift
//  InferKitMLX
//
//  DDColor (piddnad/DDColor, Apache-2.0), the modern colorizer beside the 2016 and 2017 ports: a
//  ConvNeXt encoder, a spectral-normalized U-Net decoder, and a Mask2Former-style color decoder whose
//  learned queries become per-pixel color attention maps. Reference parity is measured against the
//  authors' own `DDColor`. Tensors flow NHWC.
//

import Foundation
import CoreGraphics
import MLX
import MLXNN
import MLXFast
import InferKit

// MARK: - Configuration

/// The DDColor geometry. Defaults are the released ConvNeXt-L model.
public struct NFKMLXDDColorConfiguration: Sendable {
    /// The ConvNeXt stage depths and widths.
    public var depths: [Int] = [3, 3, 27, 3]
    public var dims: [Int] = [192, 384, 768, 1536]
    /// The decoder's widest feature width (`nf`).
    public var decoderWidth: Int = 512
    /// The color decoder's learned queries, which become the color attention maps.
    public var queryCount: Int = 100
    public var scaleCount: Int = 3
    public var decoderLayerCount: Int = 9
    public var hiddenDimensions: Int = 256
    public var colorEmbedDimensions: Int = 256
    public var feedForwardDimensions: Int = 2048
    public var headCount: Int = 8
    /// The square the encoder runs at.
    public var inputSize: Int = 512
    /// The channels the refinement emits: the two chroma channels of Lab.
    public var outputChannels: Int = 2
    public var layerNormEps: Float = 1e-6

    public init() {}

    /// The released `ddcolor_modelscope` / `ddcolor_paper` / `ddcolor_artistic` geometry (ConvNeXt-L).
    public static var large: NFKMLXDDColorConfiguration { NFKMLXDDColorConfiguration() }

    /// A small configuration for weight-free tests.
    public static var tiny: NFKMLXDDColorConfiguration {
        var configuration = NFKMLXDDColorConfiguration()
        configuration.depths = [1, 1, 1, 1]
        configuration.dims = [8, 16, 32, 64]
        configuration.decoderWidth = 32
        configuration.queryCount = 6
        configuration.decoderLayerCount = 3
        configuration.hiddenDimensions = 16
        configuration.colorEmbedDimensions = 16
        configuration.feedForwardDimensions = 32
        configuration.headCount = 2
        configuration.inputSize = 64
        return configuration
    }

    /// The three widths the color decoder projects from, which are the U-Net's own stage outputs.
    var decoderInputChannels: [Int] { [decoderWidth, decoderWidth, decoderWidth / 2] }
    /// The width the last pixel shuffle emits, which the color queries dot against.
    var pixelEmbedDimensions: Int { decoderWidth / 2 }
}

// MARK: - ConvNeXt encoder

/// A ConvNeXt block: a 7×7 depthwise convolution, a channel LayerNorm, a 4× pointwise expansion
/// through a GELU, and a learned per-channel scale, added back to the input.
final class NFKDDColorConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var depthwise: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pointwise1: Linear
    @ModuleInfo(key: "pwconv2") var pointwise2: Linear
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dimensions: Int, eps: Float) {
        _depthwise.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimensions,
                                         kernelSize: 7, padding: 3, groups: dimensions)
        _norm.wrappedValue = LayerNorm(dimensions: dimensions, eps: eps)
        _pointwise1.wrappedValue = Linear(dimensions, dimensions * 4, bias: true)
        _pointwise2.wrappedValue = Linear(dimensions * 4, dimensions, bias: true)
        _gamma.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // The reference permutes to channels-last for the norm and the pointwise pair, which is where
        // NHWC already is.
        x + gamma * pointwise2(gelu(pointwise1(norm(depthwise(x)))))
    }
}

/// The ConvNeXt trunk (`encoder.arch`).
///
/// The reference's `forward_features` calls each `norm{i}` **for its hook's side effect only** — the
/// assignment back to `x` is commented out — so the hooked feature is the normalized stage output
/// while the trunk carries the un-normalized one forward. Normalizing the trunk instead is a quiet
/// change to every later stage.
final class NFKDDColorConvNeXt: Module {
    @ModuleInfo(key: "downsample_layers") var downsampleLayers: [[Module]]
    @ModuleInfo(key: "stages") var stages: [[NFKDDColorConvNeXtBlock]]
    @ModuleInfo(key: "norm0") var norm0: LayerNorm
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    init(_ c: NFKMLXDDColorConfiguration) {
        // The stem is a stride-4 4×4 convolution then a norm; each later stage is a norm then a
        // stride-2 2×2 convolution, so the two orders differ and the indices carry it.
        var downsamples: [[Module]] = [[
            Conv2d(inputChannels: 3, outputChannels: c.dims[0], kernelSize: 4, stride: 4),
            LayerNorm(dimensions: c.dims[0], eps: c.layerNormEps),
        ]]
        for index in 0 ..< 3 {
            downsamples.append([
                LayerNorm(dimensions: c.dims[index], eps: c.layerNormEps),
                Conv2d(inputChannels: c.dims[index], outputChannels: c.dims[index + 1],
                       kernelSize: 2, stride: 2),
            ])
        }
        _downsampleLayers.wrappedValue = downsamples
        _stages.wrappedValue = (0 ..< 4).map { index in
            (0 ..< c.depths[index]).map { _ in
                NFKDDColorConvNeXtBlock(dimensions: c.dims[index], eps: c.layerNormEps)
            }
        }
        _norm0.wrappedValue = LayerNorm(dimensions: c.dims[0], eps: c.layerNormEps)
        _norm1.wrappedValue = LayerNorm(dimensions: c.dims[1], eps: c.layerNormEps)
        _norm2.wrappedValue = LayerNorm(dimensions: c.dims[2], eps: c.layerNormEps)
        _norm3.wrappedValue = LayerNorm(dimensions: c.dims[3], eps: c.layerNormEps)
    }

    /// - Returns: the four hooked features, fine → coarse, each normalized by its own `norm{i}`.
    func hookedFeatures(_ image: MLXArray) -> [MLXArray] {
        let norms = [norm0, norm1, norm2, norm3]
        var x = image
        var hooks = [MLXArray]()
        for index in 0 ..< 4 {
            let downsample = downsampleLayers[index]
            if index == 0 {
                x = (downsample[1] as! LayerNorm)((downsample[0] as! Conv2d)(x))
            } else {
                x = (downsample[1] as! Conv2d)((downsample[0] as! LayerNorm)(x))
            }
            for block in stages[index] { x = block(x) }
            hooks.append(norms[index](x))
        }
        return hooks
    }
}

// MARK: - U-Net decoder

/// A convolution the reference spectral-normalizes, optionally followed by a ReLU and a BatchNorm.
///
/// `custom_conv_layer` builds an `nn.Sequential`, so the entries carry indices: the convolution is 0,
/// a ReLU is 1 when the layer activates, and a BatchNorm follows it. The stack is held as a `[Module]`
/// array rather than named properties, because MLX parses a numeric `@ModuleInfo` key as an array
/// index and aborts the process when the checkpoint's list meets a child module (see
/// `mlx-runtime-gotchas.md`). A `[Module]` array is what those keys are for.
enum NFKDDColorConvStack {
    /// The parameter-free ReLU slot, which still consumes its Sequential index.
    final class Activation: Module {}

    static func make(inputChannels: Int, outputChannels: Int, kernelSize: Int,
                     activates: Bool, batchNorm: Bool) -> [Module] {
        // `custom_conv_layer` gives the convolution a bias only when no BatchNorm follows it.
        var stack: [Module] = [Conv2d(inputChannels: inputChannels, outputChannels: outputChannels,
                                      kernelSize: IntOrPair(kernelSize),
                                      padding: IntOrPair((kernelSize - 1) / 2), bias: !batchNorm)]
        if activates { stack.append(Activation()) }
        if batchNorm { stack.append(BatchNorm(featureCount: outputChannels)) }
        return stack
    }

    static func forward(_ stack: [Module], _ x: MLXArray) -> MLXArray {
        var out = x
        for entry in stack {
            switch entry {
            case let conv as Conv2d: out = conv(out)
            case let norm as BatchNorm: out = norm(out)
            default: out = relu(out)
            }
        }
        return out
    }
}

/// `CustomPixelShuffle_ICNR`: a 1×1 convolution to `scale²` times the width, a ReLU, a pixel shuffle,
/// and a blur that averages each output cell with its up-left neighbours.
final class NFKDDColorPixelShuffle: Module {
    @ModuleInfo(key: "conv") var conv: [Module]
    private let scale: Int
    private let blurs: Bool

    init(inputChannels: Int, outputChannels: Int, scale: Int, blurs: Bool, batchNorm: Bool) {
        self.scale = scale
        self.blurs = blurs
        _conv.wrappedValue = NFKDDColorConvStack.make(inputChannels: inputChannels,
                                                      outputChannels: outputChannels * scale * scale,
                                                      kernelSize: 1, activates: false, batchNorm: batchNorm)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shuffled = NFKMLXPixelShuffle.apply(relu(NFKDDColorConvStack.forward(conv, x)), factor: scale)
        guard blurs else { return shuffled }
        // ReplicationPad2d((1, 0, 1, 0)) then AvgPool2d(2, stride: 1): the checkerboard remedy from
        // "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts".
        let padded = MLX.padded(shuffled,
                                widths: [IntOrPair(0), IntOrPair((1, 0)), IntOrPair((1, 0)), IntOrPair(0)],
                                mode: .edge)
        let (height, width) = (shuffled.shape[1], shuffled.shape[2])
        var total = padded[0..., 0 ..< height, 0 ..< width, 0...]
        total = total + padded[0..., 1 ..< (height + 1), 0 ..< width, 0...]
        total = total + padded[0..., 0 ..< height, 1 ..< (width + 1), 0...]
        total = total + padded[0..., 1 ..< (height + 1), 1 ..< (width + 1), 0...]
        return total / 4
    }
}

/// `UnetBlockWide`: shuffle the deeper path up, concatenate the hooked skip through its own BatchNorm,
/// and fuse.
final class NFKDDColorUnetBlock: Module {
    @ModuleInfo(key: "shuf") var shuffle: NFKDDColorPixelShuffle
    @ModuleInfo(key: "bn") var norm: BatchNorm
    @ModuleInfo(key: "conv") var conv: [Module]

    init(inputChannels: Int, skipChannels: Int, outputChannels: Int, blurs: Bool) {
        _shuffle.wrappedValue = NFKDDColorPixelShuffle(inputChannels: inputChannels,
                                                       outputChannels: outputChannels,
                                                       scale: 2, blurs: blurs, batchNorm: true)
        _norm.wrappedValue = BatchNorm(featureCount: skipChannels)
        _conv.wrappedValue = NFKDDColorConvStack.make(inputChannels: outputChannels + skipChannels,
                                                      outputChannels: outputChannels,
                                                      kernelSize: 3, activates: true, batchNorm: true)
    }

    func callAsFunction(_ x: MLXArray, skip: MLXArray) -> MLXArray {
        NFKDDColorConvStack.forward(conv, relu(concatenated([shuffle(x), norm(skip)], axis: -1)))
    }
}

// MARK: - Color decoder

/// `nn.MultiheadAttention` in the reference's fused layout: one `in_proj` for the query, key, and
/// value, then `out_proj`. Sequences here are `[tokens, batch, width]`, which is what the reference's
/// default `batch_first=False` means.
final class NFKDDColorAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear
    private let heads: Int

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        _inProjWeight.wrappedValue = MLXArray.zeros([dimensions * 3, dimensions])
        _inProjBias.wrappedValue = MLXArray.zeros([dimensions * 3])
        _outProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
    }

    /// - Parameters:
    ///   - query: `[queries, 1, width]`.
    ///   - key: `[keys, 1, width]`.
    ///   - value: `[keys, 1, width]`.
    func callAsFunction(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
        let width = query.shape[2]
        let headDimensions = width / heads
        let weights = inProjWeight.split(parts: 3, axis: 0)
        let biases = inProjBias.split(parts: 3, axis: 0)
        func project(_ x: MLXArray, _ index: Int) -> MLXArray {
            let flat = x.reshaped([x.shape[0], width])
            return matmul(flat, weights[index].transposed(1, 0)) + biases[index]
        }
        // [tokens, width] → [1, heads, tokens, headDimensions]
        func split(_ x: MLXArray) -> MLXArray {
            x.reshaped([1, x.shape[0], heads, headDimensions]).transposed(0, 2, 1, 3)
        }
        let q = split(project(query, 0))
        let k = split(project(key, 1))
        let v = split(project(value, 2))
        let scale = 1.0 / sqrtf(Float(headDimensions))
        let context = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let merged = context.transposed(0, 2, 1, 3).reshaped([query.shape[0], width])
        return outProj(merged).reshaped([query.shape[0], 1, width])
    }
}

/// A post-norm self-attention layer: the query position is added to the query and the key, never to
/// the value.
final class NFKDDColorSelfAttentionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKDDColorAttention
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(dimensions: Int, heads: Int) {
        _attention.wrappedValue = NFKDDColorAttention(dimensions: dimensions, heads: heads)
        _norm.wrappedValue = LayerNorm(dimensions: dimensions)
    }

    func callAsFunction(_ x: MLXArray, queryPosition: MLXArray) -> MLXArray {
        let q = x + queryPosition
        return norm(x + attention(query: q, key: q, value: x))
    }
}

/// A post-norm cross-attention layer: the query position joins the query, the memory position joins
/// the key, and the value is the bare memory.
final class NFKDDColorCrossAttentionLayer: Module {
    @ModuleInfo(key: "multihead_attn") var attention: NFKDDColorAttention
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(dimensions: Int, heads: Int) {
        _attention.wrappedValue = NFKDDColorAttention(dimensions: dimensions, heads: heads)
        _norm.wrappedValue = LayerNorm(dimensions: dimensions)
    }

    func callAsFunction(_ x: MLXArray, memory: MLXArray, position: MLXArray,
                        queryPosition: MLXArray) -> MLXArray {
        let attended = attention(query: x + queryPosition, key: memory + position, value: memory)
        return norm(x + attended)
    }
}

/// A post-norm feed-forward layer.
final class NFKDDColorFFNLayer: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(dimensions: Int, hidden: Int) {
        _linear1.wrappedValue = Linear(dimensions, hidden, bias: true)
        _linear2.wrappedValue = Linear(hidden, dimensions, bias: true)
        _norm.wrappedValue = LayerNorm(dimensions: dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(x + linear2(relu(linear1(x)))) }
}

/// The three-layer projection that turns a decoded query into a color embedding.
final class NFKDDColorMLP: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(inputDimensions: Int, hidden: Int, outputDimensions: Int, count: Int) {
        let widths = [inputDimensions] + Array(repeating: hidden, count: count - 1)
        let outputs = Array(repeating: hidden, count: count - 1) + [outputDimensions]
        _layers.wrappedValue = (0 ..< count).map { Linear(widths[$0], outputs[$0], bias: true) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for (index, layer) in layers.enumerated() {
            out = index < layers.count - 1 ? relu(layer(out)) : layer(out)
        }
        return out
    }
}

/// `MultiScaleColorDecoder`: learned color queries cross-attend to the U-Net's three stage outputs in
/// turn, and the decoded queries dot against the pixel embedding to make one attention map each.
final class NFKDDColorColorDecoder: Module {
    @ModuleInfo(key: "transformer_self_attention_layers") var selfAttention: [NFKDDColorSelfAttentionLayer]
    @ModuleInfo(key: "transformer_cross_attention_layers") var crossAttention: [NFKDDColorCrossAttentionLayer]
    @ModuleInfo(key: "transformer_ffn_layers") var feedForward: [NFKDDColorFFNLayer]
    @ModuleInfo(key: "decoder_norm") var decoderNorm: LayerNorm
    @ModuleInfo(key: "query_feat") var queryFeatures: Embedding
    @ModuleInfo(key: "query_embed") var queryPositions: Embedding
    @ModuleInfo(key: "level_embed") var levelEmbedding: Embedding
    @ModuleInfo(key: "input_proj") var inputProjections: [Conv2d]
    @ModuleInfo(key: "color_embed") var colorEmbed: NFKDDColorMLP

    private let configuration: NFKMLXDDColorConfiguration

    init(_ c: NFKMLXDDColorConfiguration) {
        configuration = c
        _selfAttention.wrappedValue = (0 ..< c.decoderLayerCount).map { _ in
            NFKDDColorSelfAttentionLayer(dimensions: c.hiddenDimensions, heads: c.headCount)
        }
        _crossAttention.wrappedValue = (0 ..< c.decoderLayerCount).map { _ in
            NFKDDColorCrossAttentionLayer(dimensions: c.hiddenDimensions, heads: c.headCount)
        }
        _feedForward.wrappedValue = (0 ..< c.decoderLayerCount).map { _ in
            NFKDDColorFFNLayer(dimensions: c.hiddenDimensions, hidden: c.feedForwardDimensions)
        }
        _decoderNorm.wrappedValue = LayerNorm(dimensions: c.hiddenDimensions)
        _queryFeatures.wrappedValue = Embedding(embeddingCount: c.queryCount, dimensions: c.hiddenDimensions)
        _queryPositions.wrappedValue = Embedding(embeddingCount: c.queryCount, dimensions: c.hiddenDimensions)
        _levelEmbedding.wrappedValue = Embedding(embeddingCount: c.scaleCount, dimensions: c.hiddenDimensions)
        _inputProjections.wrappedValue = c.decoderInputChannels.prefix(c.scaleCount).map {
            Conv2d(inputChannels: $0, outputChannels: c.hiddenDimensions, kernelSize: 1)
        }
        _colorEmbed.wrappedValue = NFKDDColorMLP(inputDimensions: c.hiddenDimensions,
                                                 hidden: c.hiddenDimensions,
                                                 outputDimensions: c.colorEmbedDimensions, count: 3)
    }

    /// - Parameters:
    ///   - scales: the U-Net's three stage outputs, coarse → fine, each `[1, h, w, c]`.
    ///   - pixels: the pixel embedding the queries dot against, `[1, H, W, embed]`.
    /// - Returns: one attention map per query, `[1, H, W, queries]`.
    func callAsFunction(scales: [MLXArray], pixels: MLXArray) -> MLXArray {
        var memory = [MLXArray]()
        var positions = [MLXArray]()
        for (index, scale) in scales.enumerated() {
            let (h, w) = (scale.shape[1], scale.shape[2])
            let sine = NFKDDColorPositionEmbedding.sine(height: h, width: w,
                                                        features: configuration.hiddenDimensions / 2)
            positions.append(sine.reshaped([h * w, 1, configuration.hiddenDimensions]))
            let projected = inputProjections[index](scale)
                .reshaped([h * w, 1, configuration.hiddenDimensions])
            let level = levelEmbedding.weight[index].reshaped([1, 1, configuration.hiddenDimensions])
            memory.append(projected + level)
        }

        let queryPosition = queryPositions.weight.reshaped([configuration.queryCount, 1,
                                                            configuration.hiddenDimensions])
        var output = queryFeatures.weight.reshaped([configuration.queryCount, 1,
                                                    configuration.hiddenDimensions])
        for index in 0 ..< configuration.decoderLayerCount {
            let level = index % configuration.scaleCount
            // Cross-attention runs first, which is the Mask2Former order.
            output = crossAttention[index](output, memory: memory[level], position: positions[level],
                                           queryPosition: queryPosition)
            output = selfAttention[index](output, queryPosition: queryPosition)
            output = feedForward[index](output)
        }

        let decoded = colorEmbed(decoderNorm(output).reshaped([configuration.queryCount,
                                                               configuration.hiddenDimensions]))
        // einsum("bqc,bchw->bqhw"): each query's embedding dots the pixel embedding at every position.
        let (height, width) = (pixels.shape[1], pixels.shape[2])
        let flat = pixels.reshaped([height * width, configuration.pixelEmbedDimensions])
        return matmul(flat, decoded.transposed(1, 0))
            .reshaped([1, height, width, configuration.queryCount])
    }
}

/// `PositionEmbeddingSine` with `normalize: true`: the cumulative row and column index over the grid,
/// scaled to `2π`, through the usual sine and cosine ladder.
enum NFKDDColorPositionEmbedding {
    static func sine(height: Int, width: Int, features: Int, temperature: Float = 10000) -> MLXArray {
        let scale = 2 * Float.pi
        let epsilon: Float = 1e-6
        // The reference counts from one (`cumsum` over an all-true mask) and divides by the last row
        // or column, so the grid runs from `1/n · 2π` to `2π`.
        let yEmbed = (1 ... height).map { (Float($0) / (Float(height) + epsilon)) * scale }
        let xEmbed = (1 ... width).map { (Float($0) / (Float(width) + epsilon)) * scale }
        let dims = (0 ..< features).map { powf(temperature, 2 * Float($0 / 2) / Float(features)) }

        var values = [Float](repeating: 0, count: height * width * features * 2)
        for row in 0 ..< height {
            for column in 0 ..< width {
                let base = (row * width + column) * features * 2
                for index in 0 ..< features {
                    let y = yEmbed[row] / dims[index]
                    let x = xEmbed[column] / dims[index]
                    // The reference stacks sin over the even channels and cos over the odd ones, then
                    // flattens, so a pair `(2i, 2i+1)` is `(sin, cos)` of the same frequency.
                    let pair = index / 2
                    if index % 2 == 0 {
                        values[base + pair * 2] = sinf(y)
                        values[base + features + pair * 2] = sinf(x)
                    } else {
                        values[base + pair * 2 + 1] = cosf(y)
                        values[base + features + pair * 2 + 1] = cosf(x)
                    }
                }
            }
        }
        return MLXArray(values, [1, height, width, features * 2])
    }
}

// MARK: - Model

/// The whole model: the ConvNeXt encoder, the U-Net decoder with its color decoder, and the refinement
/// that turns the color attention maps and the input back into two chroma channels.
final class NFKMLXDDColorNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKDDColorEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKDDColorDecoder
    @ModuleInfo(key: "refine_net") var refine: [[Module]]

    let configuration: NFKMLXDDColorConfiguration

    init(_ c: NFKMLXDDColorConfiguration) {
        configuration = c
        _encoder.wrappedValue = NFKDDColorEncoder(c)
        _decoder.wrappedValue = NFKDDColorDecoder(c)
        // A Sequential of one, so the convolution carries index 0.
        _refine.wrappedValue = [NFKDDColorConvStack.make(inputChannels: c.queryCount + 3,
                                                         outputChannels: c.outputChannels,
                                                         kernelSize: 1, activates: false, batchNorm: false)]
    }

    /// The ImageNet statistics the reference normalizes a three-channel input with, inside its forward.
    static func normalized(_ image: MLXArray) -> MLXArray {
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225])
        return (image - mean) / standardDeviation
    }

    /// - Parameter image: a three-channel gray image `[1, inputSize, inputSize, 3]` in `0...1`.
    /// - Returns: the two chroma channels, `[1, inputSize, inputSize, 2]`.
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        let prepared = NFKMLXDDColorNet.normalized(image)
        let maps = decoder(encoder.arch.hookedFeatures(prepared))
        return NFKDDColorConvStack.forward(refine[0], concatenated([maps, prepared], axis: -1))
    }

    /// The color attention maps, before the refinement reads them.
    func colorMaps(_ prepared: MLXArray) -> MLXArray {
        decoder(encoder.arch.hookedFeatures(prepared))
    }
}

/// The encoder wrapper the checkpoint names (`encoder.arch`).
final class NFKDDColorEncoder: Module {
    @ModuleInfo(key: "arch") var arch: NFKDDColorConvNeXt
    init(_ c: NFKMLXDDColorConfiguration) { _arch.wrappedValue = NFKDDColorConvNeXt(c) }
}

/// The U-Net decoder and the color decoder it feeds.
final class NFKDDColorDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKDDColorUnetBlock]
    @ModuleInfo(key: "last_shuf") var lastShuffle: NFKDDColorPixelShuffle
    @ModuleInfo(key: "color_decoder") var colorDecoder: NFKDDColorColorDecoder

    init(_ c: NFKMLXDDColorConfiguration) {
        // The three blocks walk the hooks coarse → fine; the last halves the width.
        let widths = [c.decoderWidth, c.decoderWidth, c.decoderWidth / 2]
        var blocks = [NFKDDColorUnetBlock]()
        var inputChannels = c.dims[3]
        for index in 0 ..< 3 {
            blocks.append(NFKDDColorUnetBlock(inputChannels: inputChannels,
                                              skipChannels: c.dims[2 - index],
                                              outputChannels: widths[index], blurs: true))
            inputChannels = widths[index]
        }
        _layers.wrappedValue = blocks
        _lastShuffle.wrappedValue = NFKDDColorPixelShuffle(inputChannels: c.pixelEmbedDimensions,
                                                           outputChannels: c.pixelEmbedDimensions,
                                                           scale: 4, blurs: true, batchNorm: false)
        _colorDecoder.wrappedValue = NFKDDColorColorDecoder(c)
    }

    /// - Parameter hooks: the encoder's four hooked features, fine → coarse.
    func callAsFunction(_ hooks: [MLXArray]) -> MLXArray {
        let out0 = layers[0](hooks[3], skip: hooks[2])
        let out1 = layers[1](out0, skip: hooks[1])
        let out2 = layers[2](out1, skip: hooks[0])
        let pixels = lastShuffle(out2)
        return colorDecoder(scales: [out0, out1, out2], pixels: pixels)
    }
}

// MARK: - Public surface

/// The released DDColor weights.
///
/// - Since: InferKit 0.4.0
@objc(NFKMLXDDColorVariant)
public enum NFKMLXDDColorVariant: Int {
    /// `ddcolor_modelscope`, the release the authors' demo runs.
    case modelscope
    /// `ddcolor_paper`, the weights the paper reports.
    case paper
    /// `ddcolor_artistic`, tuned for saturation rather than fidelity.
    case artistic
}

/// DDColor: automatic colorization through learned color queries, at reference parity against the
/// authors' own `DDColor`.
///
/// - Since: InferKit 0.4.0
@objc(NFKMLXDDColor)
public final class NFKMLXDDColor: NSObject {
    /// The registry name the modelscope release builds under.
    @objc public static let modelName = "ddcolor"

    static func specs(for variant: NFKMLXDDColorVariant) -> (name: String, configuration: NFKMLXDDColorConfiguration) {
        switch variant {
        case .modelscope: return (modelName, .large)
        case .paper: return ("ddcolor-paper", .large)
        case .artistic: return ("ddcolor-artistic", .large)
        }
    }

    static func makeNet(_ configuration: NFKMLXDDColorConfiguration = .large) -> NFKMLXDDColorNet {
        let net = NFKMLXDDColorNet(configuration)
        net.train(false)                                                // BatchNorm running statistics
        return net
    }

    /// Builds a colorization backend from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .modelscope, weightsURL: weightsURL)
    }

    /// Builds one of the released weights.
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXDDColorVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let spec = specs(for: variant)
        let net = makeNet(spec.configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
            net.train(false)
        }
        let holder = NFKDDColorHolder(net)
        return NFKMLXModuleBackend(identifier: spec.name, isReady: true) { image in
            holder.net.colorize(image)
        }
    }

    /// Downloads the checkpoint from Hugging Face, then builds. Blocking on the network; run off the
    /// render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .modelscope, repo: repo, weightsPath: weightsPath, revision: revision,
                    cacheDirectoryURL: cacheDirectoryURL)
    }

    /// The download factory at a chosen release.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXDDColorVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        backend(variant: .modelscope, repo: repo, weightsPath: weightsPath, revision: revision,
                cacheDirectoryURL: cacheDirectoryURL, completionHandler: completionHandler)
    }

    /// The asynchronous download factory at a chosen release.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXDDColorVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers the released weights with `NFKMLXModelRegistry`.
    @objc public static func register() {
        for variant in [NFKMLXDDColorVariant.modelscope, .paper, .artistic] {
            NFKMLXModelRegistry.register(name: specs(for: variant).name) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL)
            }
        }
    }

    /// Loads a released checkpoint, fusing the spectral normalization the reference's convolutions
    /// carry.
    ///
    /// `nn.utils.spectral_norm` stores `weight_orig` beside the power-iteration vectors `weight_u` and
    /// `weight_v`, and in eval mode divides the weight by `uᵀ W v` using the vectors as stored rather
    /// than iterating again. Fusing that here is what the weight-norm fusions elsewhere in the package
    /// do for `g·v/‖v‖`.
    static func loadWeights(into net: NFKMLXDDColorNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let arrays = checkpoint.arrays
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays {
            guard let name = remapReferenceKey(key) else { continue }
            var array = value
            if key.hasSuffix(".weight_orig") {
                let stem = String(key.dropLast(".weight_orig".count))
                guard let u = arrays[stem + ".weight_u"], let v = arrays[stem + ".weight_v"] else {
                    throw NFKMLXError.malformedCheckpoint(
                        "\(stem) carries a spectral-normalized weight with no power-iteration vectors")
                }
                let rows = value.shape[0]
                let flat = value.reshaped([rows, value.size / rows])
                let sigma = matmul(u.reshaped([1, rows]), matmul(flat, v.reshaped([v.size, 1])))
                array = value / sigma.reshaped([])
            }
            if checkpoint.needsConvTranspose && array.ndim == 4 {
                array = array.transposed(0, 2, 3, 1)
            }
            mapped.append((name, array))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Maps a checkpoint key onto the built module, or nil to drop one the colorization path does not
    /// use: the power-iteration vectors the fusion consumes, the encoder's classification head and its
    /// pooled norm, the input normalization the port applies itself, and the BatchNorm step counters.
    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasSuffix(".weight_u") || key.hasSuffix(".weight_v") { return nil }
        if key.hasSuffix("num_batches_tracked") { return nil }
        if key == "mean" || key == "std" { return nil }
        if key.hasPrefix("encoder.arch.head_cls.") { return nil }
        if key.hasPrefix("encoder.arch.norm.") { return nil }   // the pooled classification norm
        if key.hasSuffix(".weight_orig") { return String(key.dropLast("_orig".count)) }
        return key
    }
}

private final class NFKDDColorHolder: @unchecked Sendable {
    let net: NFKMLXDDColorNet
    init(_ net: NFKMLXDDColorNet) { self.net = net }
}

extension NFKMLXDDColorNet {
    /// Colorizes a bridged `[H, W, 3]` image in `0...1`.
    ///
    /// The lightness comes from the caller's own image at full resolution and only the chroma is
    /// predicted, so luminance is preserved exactly — the convention the two older colorizers here
    /// also follow.
    func colorize(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let lightness = NFKLabColor.toLab(image)[0..., 0..., 0 ..< 1]           // [H, W, 1], 0...100

        // The reference feeds the gray image back as three channels, which is the lightness alone
        // taken through Lab with no chroma.
        let gray = NFKLabColor.toRGB(concatenated([lightness, MLXArray.zeros([height, width, 2])], axis: 2))
        let resized = NFKMLXResample.resizeBilinear(gray.reshaped([1, height, width, 3]),
                                                    height: configuration.inputSize,
                                                    width: configuration.inputSize)
        let chroma = self(resized)                                              // [1, size, size, 2]
        let full = NFKMLXResample.resizeBilinear(chroma, height: height, width: width)
        return NFKLabColor.toRGB(concatenated([lightness, full.reshaped([height, width, 2])], axis: 2))
    }
}
