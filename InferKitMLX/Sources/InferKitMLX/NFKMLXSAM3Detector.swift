//
//  NFKMLXSAM3Detector.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// SAM 3's detector: the half of the model that turns an encoded image and an encoded phrase into
// boxes, scores, and masks for every instance the phrase names. The two encoders are in
// `NFKMLXSAM3.swift`; this is what reads them.
//
// Four stages run in order. The DETR ENCODER fuses one vision level with the prompt over six layers
// of self-attention and cross-attention. The DETR DECODER runs 200 learned queries and a presence
// token over that, refining a learned box per query at every layer, with a relative-position bias
// built from the current boxes so a query attends around where it currently points. The SCORING head
// dots each query against the mean-pooled prompt. The MASK DECODER lifts the encoder's output back
// up the feature pyramid and dots the queries against it, one mask per query.
//
// Every attention here is dense: SAM 3 has no deformable sampling anywhere, which is what lets the
// whole detector be ordinary matrix arithmetic. Two details about the reference are easy to miss and
// both change numbers. Its `hidden_act` is RELU, not the gelu its encoders use. And its layer
// normalizations are constructed WITHOUT an epsilon, so they take PyTorch's 1e-5 default while the
// configuration's `layer_norm_eps` of 1e-6 goes unread.

/// SAM 3 detector geometry. Defaults are the released `facebook/sam3` detector.
public struct NFKMLXSAM3DetectorConfiguration: Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var heads: Int
    public var encoderLayers: Int
    public var decoderLayers: Int
    public var queryCount: Int
    /// Stages the pixel decoder builds. The released model ships three and runs two, because the
    /// count of stages it climbs is one less than the feature levels it is given.
    public var upsamplingStages: Int
    public var layerNormEpsilon: Float
    /// Bounds the reference applies to keep two heads' logits in range.
    public var presenceClamp: Float
    public var scoreClamp: Float

    public init(hiddenSize: Int = 256, intermediateSize: Int = 2048, heads: Int = 8,
                encoderLayers: Int = 6, decoderLayers: Int = 6, queryCount: Int = 200,
                upsamplingStages: Int = 3, layerNormEpsilon: Float = 1e-5,
                presenceClamp: Float = 10, scoreClamp: Float = 12) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.heads = heads
        self.encoderLayers = encoderLayers
        self.decoderLayers = decoderLayers
        self.queryCount = queryCount
        self.upsamplingStages = upsamplingStages
        self.layerNormEpsilon = layerNormEpsilon
        self.presenceClamp = presenceClamp
        self.scoreClamp = scoreClamp
    }

    public static let base = NFKMLXSAM3DetectorConfiguration()

    var headDim: Int { hiddenSize / heads }
}

/// Dense attention over separate query, key, and value streams, with an optional additive mask.
final class NFKSAM3DenseAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let heads: Int
    let headDim: Int

    init(_ configuration: NFKMLXSAM3DetectorConfiguration) {
        self.heads = configuration.heads
        self.headDim = configuration.headDim
        let width = configuration.hiddenSize
        _qProj.wrappedValue = Linear(width, width)
        _kProj.wrappedValue = Linear(width, width)
        _vProj.wrappedValue = Linear(width, width)
        _oProj.wrappedValue = Linear(width, width)
    }

    /// `query` `[1, Q, C]`, `key` and `value` `[1, K, C]`, `mask` `[1, heads, Q, K]` or nil.
    func callAsFunction(query: MLXArray, key: MLXArray, value: MLXArray,
                        mask: MLXArray? = nil) -> MLXArray {
        let batch = query.dim(0), queries = query.dim(1), keys = key.dim(1)
        func split(_ x: MLXArray, _ projection: Linear, _ count: Int) -> MLXArray {
            projection(x).reshaped([batch, count, heads, headDim]).transposed(0, 2, 1, 3)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(query, qProj, queries), keys: split(key, kProj, keys),
            values: split(value, vProj, keys), scale: 1 / sqrt(Float(headDim)),
            mask: mask.map { .array($0) } ?? .none)
        return oProj(attended.transposed(0, 2, 1, 3).reshaped([batch, queries, heads * headDim]))
    }
}

/// The detector's feed-forward. Its activation is the RELU the released configuration names, where
/// the two encoders use a gelu.
final class NFKSAM3DetrMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ configuration: NFKMLXSAM3DetectorConfiguration) {
        _fc1.wrappedValue = Linear(configuration.hiddenSize, configuration.intermediateSize)
        _fc2.wrappedValue = Linear(configuration.intermediateSize, configuration.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(relu(fc1(x))) }
}

/// The two- or three-layer MLP the decoder's heads are built from, under the reference's own
/// `layer1` / `layer2` / `layer3` names.
final class NFKSAM3DecoderMLP: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear
    @ModuleInfo(key: "layer3") var layer3: Linear?

    init(input: Int, hidden: Int, output: Int, layers: Int) {
        if layers == 3 {
            _layer1.wrappedValue = Linear(input, hidden)
            _layer2.wrappedValue = Linear(hidden, hidden)
            _layer3.wrappedValue = Linear(hidden, output)
        } else {
            _layer1.wrappedValue = Linear(input, hidden)
            _layer2.wrappedValue = Linear(hidden, output)
            _layer3.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let first = relu(layer1(x))
        guard let layer3 else { return layer2(first) }
        return layer3(relu(layer2(first)))
    }
}

/// One DETR encoder layer: self-attention over the vision tokens with their position encoding, then
/// cross-attention into the prompt, then the feed-forward.
final class NFKSAM3DetrEncoderLayer: Module {
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: NFKSAM3DenseAttention
    @ModuleInfo(key: "cross_attn") var crossAttention: NFKSAM3DenseAttention
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKSAM3DetrMLP
    @ModuleInfo(key: "layer_norm3") var norm3: LayerNorm

    init(_ configuration: NFKMLXSAM3DetectorConfiguration) {
        let width = configuration.hiddenSize, epsilon = configuration.layerNormEpsilon
        _norm1.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
        _selfAttention.wrappedValue = NFKSAM3DenseAttention(configuration)
        _crossAttention.wrappedValue = NFKSAM3DenseAttention(configuration)
        _norm2.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
        _mlp.wrappedValue = NFKSAM3DetrMLP(configuration)
        _norm3.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
    }

    func callAsFunction(_ vision: MLXArray, prompt: MLXArray, position: MLXArray,
                        promptMask: MLXArray?) -> MLXArray {
        var hidden = norm1(vision)
        let withPosition = hidden + position
        // The position encoding reaches the query and the key but not the value.
        var out = vision + selfAttention(query: withPosition, key: withPosition, value: hidden)
        hidden = norm2(out)
        out = out + crossAttention(query: hidden, key: prompt, value: prompt, mask: promptMask)
        return out + mlp(norm3(out))
    }
}

/// One DETR decoder layer: self-attention among the queries, cross-attention into the prompt, then
/// cross-attention into the vision tokens under the box relative-position bias.
final class NFKSAM3DetrDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: NFKSAM3DenseAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfNorm: LayerNorm
    @ModuleInfo(key: "text_cross_attn") var textAttention: NFKSAM3DenseAttention
    @ModuleInfo(key: "text_cross_attn_layer_norm") var textNorm: LayerNorm
    @ModuleInfo(key: "vision_cross_attn") var visionAttention: NFKSAM3DenseAttention
    @ModuleInfo(key: "vision_cross_attn_layer_norm") var visionNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKSAM3DetrMLP
    @ModuleInfo(key: "mlp_layer_norm") var mlpNorm: LayerNorm

    init(_ configuration: NFKMLXSAM3DetectorConfiguration) {
        let width = configuration.hiddenSize, epsilon = configuration.layerNormEpsilon
        _selfAttention.wrappedValue = NFKSAM3DenseAttention(configuration)
        _selfNorm.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
        _textAttention.wrappedValue = NFKSAM3DenseAttention(configuration)
        _textNorm.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
        _visionAttention.wrappedValue = NFKSAM3DenseAttention(configuration)
        _visionNorm.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
        _mlp.wrappedValue = NFKSAM3DetrMLP(configuration)
        _mlpNorm.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
    }

    /// The normalizations run AFTER each residual here, where the encoder's run before it.
    func callAsFunction(_ queries: MLXArray, queryPosition: MLXArray, prompt: MLXArray,
                        vision: MLXArray, visionPosition: MLXArray, promptMask: MLXArray?,
                        visionBias: MLXArray?) -> MLXArray {
        var hidden = queries
        var withPosition = hidden + queryPosition
        hidden = selfNorm(hidden + selfAttention(query: withPosition, key: withPosition, value: hidden))
        withPosition = hidden + queryPosition
        hidden = textNorm(hidden + textAttention(query: withPosition, key: prompt, value: prompt,
                                                 mask: promptMask))
        withPosition = hidden + queryPosition
        hidden = visionNorm(hidden + visionAttention(query: withPosition, key: vision + visionPosition,
                                                     value: vision, mask: visionBias))
        return mlpNorm(hidden + mlp(hidden))
    }
}

/// The DETR encoder: six layers over one flattened vision level.
final class NFKSAM3DetrEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKSAM3DetrEncoderLayer]

    init(_ configuration: NFKMLXSAM3DetectorConfiguration) {
        _layers.wrappedValue = (0 ..< configuration.encoderLayers).map { _ in
            NFKSAM3DetrEncoderLayer(configuration)
        }
    }

    func callAsFunction(_ vision: MLXArray, prompt: MLXArray, position: MLXArray,
                        promptMask: MLXArray?) -> MLXArray {
        var hidden = vision
        for layer in layers {
            hidden = layer(hidden, prompt: prompt, position: position, promptMask: promptMask)
        }
        return hidden
    }
}

/// What one pass of the DETR decoder leaves behind.
public struct NFKMLXSAM3DecoderOutput {
    /// The last layer's queries, `[1, queries, C]`.
    public var queries: MLXArray
    /// The boxes the last layer refined, `[1, queries, 4]` as `(cx, cy, w, h)` in `0...1`.
    public var boxes: MLXArray
    /// The presence token's logit, clamped as the reference clamps it.
    public var presence: MLXArray
}

/// The DETR decoder: learned queries and a presence token over the encoder's output, refining one
/// box per query at every layer.
final class NFKSAM3DetrDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKSAM3DetrDecoderLayer]
    @ModuleInfo(key: "output_layer_norm") var outputNorm: LayerNorm
    @ModuleInfo(key: "box_head") var boxHead: NFKSAM3DecoderMLP
    @ModuleInfo(key: "query_embed") var queryEmbed: Embedding
    @ModuleInfo(key: "reference_points") var referencePoints: Embedding
    @ModuleInfo(key: "presence_token") var presenceToken: Embedding
    @ModuleInfo(key: "presence_head") var presenceHead: NFKSAM3DecoderMLP
    @ModuleInfo(key: "presence_layer_norm") var presenceNorm: LayerNorm
    @ModuleInfo(key: "ref_point_head") var referenceHead: NFKSAM3DecoderMLP
    @ModuleInfo(key: "box_rpb_embed_x") var biasX: NFKSAM3DecoderMLP
    @ModuleInfo(key: "box_rpb_embed_y") var biasY: NFKSAM3DecoderMLP

    let configuration: NFKMLXSAM3DetectorConfiguration

    init(_ configuration: NFKMLXSAM3DetectorConfiguration) {
        self.configuration = configuration
        let width = configuration.hiddenSize, epsilon = configuration.layerNormEpsilon
        _layers.wrappedValue = (0 ..< configuration.decoderLayers).map { _ in
            NFKSAM3DetrDecoderLayer(configuration)
        }
        _outputNorm.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
        _boxHead.wrappedValue = NFKSAM3DecoderMLP(input: width, hidden: width, output: 4, layers: 3)
        _queryEmbed.wrappedValue = Embedding(embeddingCount: configuration.queryCount, dimensions: width)
        _referencePoints.wrappedValue = Embedding(embeddingCount: configuration.queryCount, dimensions: 4)
        _presenceToken.wrappedValue = Embedding(embeddingCount: 1, dimensions: width)
        _presenceHead.wrappedValue = NFKSAM3DecoderMLP(input: width, hidden: width, output: 1, layers: 3)
        _presenceNorm.wrappedValue = LayerNorm(dimensions: width, eps: epsilon)
        _referenceHead.wrappedValue = NFKSAM3DecoderMLP(input: 2 * width, hidden: width, output: width,
                                                        layers: 2)
        _biasX.wrappedValue = NFKSAM3DecoderMLP(input: 2, hidden: width, output: configuration.heads,
                                                layers: 2)
        _biasY.wrappedValue = NFKSAM3DecoderMLP(input: 2, hidden: width, output: configuration.heads,
                                                layers: 2)
    }

    /// The relative-position bias a set of boxes puts on the vision tokens: the log-scaled distance
    /// from each grid line to the box's two edges, embedded per head and summed across the axes.
    func bias(boxes: MLXArray, height: Int, width: Int) -> MLXArray {
        let queries = boxes.dim(1)
        let corners = NFKSAM3Boxes.centerToCorners(boxes)                  // [1, Q, 4]
        let rows = MLXArray((0 ..< height).map { Float($0) / Float(height) }).reshaped([1, 1, height, 1])
        let columns = MLXArray((0 ..< width).map { Float($0) / Float(width) }).reshaped([1, 1, width, 1])
        // The y edges are corners 1 and 3, the x edges 0 and 2.
        let yEdges = concatenated([corners[0..., 0..., 1 ..< 2], corners[0..., 0..., 3 ..< 4]], axis: -1)
        let xEdges = concatenated([corners[0..., 0..., 0 ..< 1], corners[0..., 0..., 2 ..< 3]], axis: -1)
        let deltaY = NFKSAM3Boxes.logScaled(rows - yEdges.reshaped([1, queries, 1, 2]))
        let deltaX = NFKSAM3Boxes.logScaled(columns - xEdges.reshaped([1, queries, 1, 2]))
        let embeddedY = biasY(deltaY)                                      // [1, Q, H, heads]
        let embeddedX = biasX(deltaX)                                      // [1, Q, W, heads]
        let combined = embeddedY.expandedDimensions(axis: 3) + embeddedX.expandedDimensions(axis: 2)
        return combined.reshaped([1, queries, height * width, configuration.heads])
            .transposed(0, 3, 1, 2)                                        // [1, heads, Q, H·W]
    }

    /// `vision` `[1, N, C]` over an `height × width` grid.
    func callAsFunction(vision: MLXArray, prompt: MLXArray, visionPosition: MLXArray,
                        promptMask: MLXArray?, height: Int,
                        width: Int) -> NFKMLXSAM3DecoderOutput {
        let hiddenSize = configuration.hiddenSize
        var hidden = concatenated([presenceToken.weight.reshaped([1, 1, hiddenSize]),
                                   queryEmbed.weight.reshaped([1, -1, hiddenSize])], axis: 1)
        var boxes = sigmoid(referencePoints.weight).reshaped([1, -1, 4])
        var queries = hidden[0..., 1...]
        var presence = MLXArray.zeros([1, 1])

        for layer in layers {
            let queryPosition = referenceHead(NFKSAM3Boxes.sineEncoded(boxes, features: hiddenSize / 2))
            // The presence token carries no position and attends to every vision token equally, so
            // its row of the bias is zero and its query position is zero.
            let padded = concatenated([MLXArray.zeros([1, 1, hiddenSize]), queryPosition], axis: 1)
            let visionBias = concatenated(
                [MLXArray.zeros([1, configuration.heads, 1, height * width]),
                 bias(boxes: boxes, height: height, width: width)], axis: 2)

            hidden = layer(hidden, queryPosition: padded, prompt: prompt, vision: vision,
                           visionPosition: visionPosition, promptMask: promptMask,
                           visionBias: visionBias)

            queries = outputNorm(hidden[0..., 1...])
            boxes = sigmoid(boxHead(queries) + NFKSAM3Boxes.inverseSigmoid(boxes))
            presence = clip(presenceHead(presenceNorm(hidden[0..., ..<1])).reshaped([1, 1]),
                            min: -configuration.presenceClamp, max: configuration.presenceClamp)
        }
        return NFKMLXSAM3DecoderOutput(queries: queries, boxes: boxes, presence: presence)
    }
}

/// Box arithmetic the decoder needs: the two coordinate conventions, the inverse of the sigmoid the
/// boxes live under, the log scaling the bias applies, and the sine encoding of a box.
enum NFKSAM3Boxes {
    /// `(cx, cy, w, h)` → `(x1, y1, x2, y2)`.
    static func centerToCorners(_ boxes: MLXArray) -> MLXArray {
        let centerX = boxes[.ellipsis, 0 ..< 1], centerY = boxes[.ellipsis, 1 ..< 2]
        let width = boxes[.ellipsis, 2 ..< 3], height = boxes[.ellipsis, 3 ..< 4]
        return concatenated([centerX - 0.5 * width, centerY - 0.5 * height,
                             centerX + 0.5 * width, centerY + 0.5 * height], axis: -1)
    }

    /// The reference's `inverse_sigmoid`, whose clamps keep a box at either extreme finite.
    static func inverseSigmoid(_ x: MLXArray, epsilon: Float = 1e-3) -> MLXArray {
        let bounded = clip(x, min: 0, max: 1)
        return log(clip(bounded, min: epsilon) / clip(1 - bounded, min: epsilon))
    }

    /// `sign(8x)·log2(|8x| + 1) / log2(8)`, which is what compresses a distance before it is embedded.
    static func logScaled(_ x: MLXArray) -> MLXArray {
        let scaled = x * 8
        return sign(scaled) * log2(abs(scaled) + 1) / log2f(8)
    }

    /// The reference's `encode_boxes`: each of `(y, x, w, h)` through a sine ladder, in that order.
    static func sineEncoded(_ boxes: MLXArray, features: Int, temperature: Float = 10000) -> MLXArray {
        let scale = 2 * Float.pi
        let dimensions = (0 ..< features).map { powf(temperature, 2 * Float($0 / 2) / Float(features)) }
        let queries = boxes.dim(1)
        let values = boxes.reshaped([queries, 4]).asArray(Float.self)
        var encoded = [Float](repeating: 0, count: queries * 4 * features)
        // The concatenation is y, x, w, h, where a box is stored (cx, cy, w, h).
        let order = [1, 0, 2, 3]
        for query in 0 ..< queries {
            for (slot, axis) in order.enumerated() {
                let coordinate = values[query * 4 + axis] * scale
                let base = query * 4 * features + slot * features
                for pair in 0 ..< features / 2 {
                    let angle = coordinate / dimensions[2 * pair]
                    encoded[base + 2 * pair] = sinf(angle)
                    encoded[base + 2 * pair + 1] = cosf(angle)
                }
            }
        }
        return MLXArray(encoded, [1, queries, 4 * features])
    }
}

/// The scoring head: each query dotted against the mean-pooled prompt.
final class NFKSAM3Scoring: Module {
    @ModuleInfo(key: "text_mlp") var textMLP: NFKSAM3DecoderMLP
    @ModuleInfo(key: "text_mlp_out_norm") var textNorm: LayerNorm
    @ModuleInfo(key: "text_proj") var textProjection: Linear
    @ModuleInfo(key: "query_proj") var queryProjection: Linear

    let clamp: Float
    let scale: Float

    init(_ configuration: NFKMLXSAM3DetectorConfiguration) {
        let width = configuration.hiddenSize
        _textMLP.wrappedValue = NFKSAM3DecoderMLP(input: width, hidden: configuration.intermediateSize,
                                                  output: width, layers: 2)
        _textNorm.wrappedValue = LayerNorm(dimensions: width, eps: configuration.layerNormEpsilon)
        _textProjection.wrappedValue = Linear(width, width)
        _queryProjection.wrappedValue = Linear(width, width)
        self.clamp = configuration.scoreClamp
        self.scale = 1 / sqrt(Float(width))
    }

    /// `queries` `[1, Q, C]`, `prompt` `[1, L, C]`, `valid` `[1, L]` (1 for a real token) → `[1, Q]`.
    func callAsFunction(queries: MLXArray, prompt: MLXArray, valid: MLXArray?) -> MLXArray {
        let refined = textNorm(textMLP(prompt) + prompt)
        var pooled: MLXArray
        if let valid {
            let weights = valid.reshaped([1, -1, 1])
            pooled = (refined * weights).sum(axis: 1) / MLX.maximum(weights.sum(axis: 1), MLXArray(Float(1)))
        } else {
            pooled = refined.mean(axis: 1)
        }
        let projected = textProjection(pooled).reshaped([1, -1, 1])        // [1, C, 1]
        let scores = matmul(queryProjection(queries), projected) * scale
        return clip(scores.reshaped([1, -1]), min: -clamp, max: clamp)
    }
}

/// The pixel decoder: the encoder's output lifted back up the feature pyramid.
final class NFKSAM3PixelDecoder: Module {
    @ModuleInfo(key: "conv_layers") var convolutions: [Conv2d]
    @ModuleInfo(key: "norms") var norms: [GroupNorm]

    init(_ configuration: NFKMLXSAM3DetectorConfiguration) {
        let width = configuration.hiddenSize
        _convolutions.wrappedValue = (0 ..< configuration.upsamplingStages).map { _ in
            Conv2d(inputChannels: width, outputChannels: width, kernelSize: 3, padding: 1)
        }
        _norms.wrappedValue = (0 ..< configuration.upsamplingStages).map { _ in
            GroupNorm(groupCount: 8, dimensions: width, pytorchCompatible: true)
        }
    }

    /// `levels` runs finest to coarsest; the walk starts at the coarsest and climbs.
    func callAsFunction(_ levels: [MLXArray]) -> MLXArray {
        var hidden = levels[levels.count - 1]
        for (stage, level) in levels.dropLast().reversed().enumerated() {
            hidden = NFKMLXResample.upsampleNearest(hidden, scale: level.dim(1) / hidden.dim(1)) + level
            hidden = relu(norms[stage](convolutions[stage](hidden)))
        }
        return hidden
    }
}

/// The mask embedder: three linear layers with a relu between them.
final class NFKSAM3MaskEmbedder: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(_ configuration: NFKMLXSAM3DetectorConfiguration) {
        let width = configuration.hiddenSize
        _layers.wrappedValue = (0 ..< 3).map { _ in Linear(width, width) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers[2](relu(layers[1](relu(layers[0](x)))))
    }
}

/// The mask decoder: the encoder's output cross-attends the prompt, replaces the finest level of the
/// pyramid, and is dotted against the queries, one mask per query.
final class NFKSAM3MaskDecoder: Module {
    @ModuleInfo(key: "pixel_decoder") var pixelDecoder: NFKSAM3PixelDecoder
    @ModuleInfo(key: "mask_embedder") var maskEmbedder: NFKSAM3MaskEmbedder
    @ModuleInfo(key: "instance_projection") var instanceProjection: Conv2d
    @ModuleInfo(key: "semantic_projection") var semanticProjection: Conv2d
    @ModuleInfo(key: "prompt_cross_attn") var promptAttention: NFKSAM3DenseAttention
    @ModuleInfo(key: "prompt_cross_attn_norm") var promptNorm: LayerNorm

    init(_ configuration: NFKMLXSAM3DetectorConfiguration) {
        let width = configuration.hiddenSize
        _pixelDecoder.wrappedValue = NFKSAM3PixelDecoder(configuration)
        _maskEmbedder.wrappedValue = NFKSAM3MaskEmbedder(configuration)
        _instanceProjection.wrappedValue = Conv2d(inputChannels: width, outputChannels: width,
                                                  kernelSize: 1)
        _semanticProjection.wrappedValue = Conv2d(inputChannels: width, outputChannels: 1, kernelSize: 1)
        _promptAttention.wrappedValue = NFKSAM3DenseAttention(configuration)
        _promptNorm.wrappedValue = LayerNorm(dimensions: width, eps: configuration.layerNormEpsilon)
    }

    /// `levels` runs finest to coarsest and `encoded` `[1, N, C]` covers the coarsest of them.
    func callAsFunction(queries: MLXArray, levels: [MLXArray], encoded: MLXArray,
                        prompt: MLXArray?, promptMask: MLXArray?)
        -> (masks: MLXArray, semantic: MLXArray) {
        var vision = encoded
        if let prompt {
            let normalized = promptNorm(vision)
            vision = vision + promptAttention(query: normalized, key: prompt, value: prompt,
                                              mask: promptMask)
        }
        var pyramid = levels
        let coarsest = levels[levels.count - 1]
        pyramid[levels.count - 1] = vision.reshaped([1, coarsest.dim(1), coarsest.dim(2), -1])

        let pixels = pixelDecoder(pyramid)
        let instances = instanceProjection(pixels)                         // [1, H, W, C]
        let embedded = maskEmbedder(queries)                               // [1, Q, C]
        let (height, width) = (instances.dim(1), instances.dim(2))
        let flat = instances.reshaped([1, height * width, -1]).transposed(0, 2, 1)
        let masks = matmul(embedded, flat).reshaped([1, -1, height, width])
        return (masks, semanticProjection(pixels))
    }
}

/// What one detection pass returns.
public struct NFKMLXSAM3Detection {
    /// Mask logits at the pyramid's finest level, `[1, queries, H, W]`.
    public var masks: MLXArray
    /// Boxes as `(x1, y1, x2, y2)` in `0...1`, `[1, queries, 4]`.
    public var boxes: MLXArray
    /// One logit per query, `[1, queries]`.
    public var logits: MLXArray
    /// A single logit saying whether the prompt names anything in the image at all.
    public var presence: MLXArray
    /// A prompt-conditioned foreground map, `[1, H, W, 1]`.
    public var semantic: MLXArray
}

/// SAM 3's detector.
public final class NFKMLXSAM3DetectorNet: Module {
    @ModuleInfo(key: "detr_encoder") var encoder: NFKSAM3DetrEncoder
    @ModuleInfo(key: "detr_decoder") var decoder: NFKSAM3DetrDecoder
    @ModuleInfo(key: "dot_product_scoring") var scoring: NFKSAM3Scoring
    @ModuleInfo(key: "mask_decoder") var maskDecoder: NFKSAM3MaskDecoder

    public let configuration: NFKMLXSAM3DetectorConfiguration

    public init(_ configuration: NFKMLXSAM3DetectorConfiguration = .base) {
        self.configuration = configuration
        _encoder.wrappedValue = NFKSAM3DetrEncoder(configuration)
        _decoder.wrappedValue = NFKSAM3DetrDecoder(configuration)
        _scoring.wrappedValue = NFKSAM3Scoring(configuration)
        _maskDecoder.wrappedValue = NFKSAM3MaskDecoder(configuration)
    }

    /// Detects every instance the prompt names.
    ///
    /// - Parameters:
    ///   - levels: the vision encoder's FPN levels WITHOUT its coarsest, finest first. The detector
    ///     reads the coarsest of what remains and lifts its result back to the finest.
    ///   - positions: the sine position encoding of each level, in the same order.
    ///   - prompt: the projected prompt `[1, L, C]`.
    ///   - promptValid: 1 for a real prompt token and 0 for padding, `[1, L]`.
    public func callAsFunction(levels: [MLXArray], positions: [MLXArray], prompt: MLXArray,
                               promptValid: MLXArray?) -> NFKMLXSAM3Detection {
        let coarsest = levels[levels.count - 1]
        let (height, width) = (coarsest.dim(1), coarsest.dim(2))
        let hiddenSize = configuration.hiddenSize
        let vision = coarsest.reshaped([1, height * width, hiddenSize])
        let position = positions[positions.count - 1].reshaped([1, height * width, hiddenSize])
        let mask = promptValid.map { additiveMask($0) }

        let encoded = encoder(vision, prompt: prompt, position: position, promptMask: mask)
        let decoded = decoder(vision: encoded, prompt: prompt, visionPosition: position,
                              promptMask: mask, height: height, width: width)
        let logits = scoring(queries: decoded.queries, prompt: prompt, valid: promptValid)
        let masks = maskDecoder(queries: decoded.queries, levels: levels, encoded: encoded,
                                prompt: prompt, promptMask: mask)
        return NFKMLXSAM3Detection(masks: masks.masks,
                                   boxes: NFKSAM3Boxes.centerToCorners(decoded.boxes),
                                   logits: logits, presence: decoded.presence,
                                   semantic: masks.semantic)
    }

    /// A key-padding mask as the attention takes it: zero where the prompt is real, and the floor
    /// where it is padding.
    func additiveMask(_ valid: MLXArray) -> MLXArray {
        let length = valid.dim(valid.ndim - 1)
        return ((1 - valid.reshaped([1, 1, 1, length])) * -Float.greatestFiniteMagnitude)
    }
}

extension NFKMLXSAM3 {

    /// The detector at a chosen geometry.
    public static func makeDetectorNet(
        _ configuration: NFKMLXSAM3DetectorConfiguration = .base) -> NFKMLXSAM3DetectorNet {
        NFKMLXSAM3DetectorNet(configuration)
    }

    /// Reads the detector's geometry from a released `config.json`.
    ///
    /// @discussion The epsilon is NOT read from `layer_norm_eps`. The reference builds every
    /// normalization in the detector without one, so they take PyTorch's 1e-5 default and the
    /// configuration's 1e-6 goes unread; reading it would put this port a thousandfold off the
    /// weights it loads.
    public static func detectorConfiguration(fromHuggingFace url: URL) throws -> NFKMLXSAM3DetectorConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let detector = json["detector_config"] as? [String: Any],
              let encoder = detector["detr_encoder_config"] as? [String: Any],
              let decoder = detector["detr_decoder_config"] as? [String: Any],
              let masks = detector["mask_decoder_config"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a SAM 3 configuration")
        }
        func integer(_ source: [String: Any], _ key: String, _ fallback: Int) -> Int {
            (source[key] as? NSNumber)?.intValue ?? fallback
        }
        return NFKMLXSAM3DetectorConfiguration(
            hiddenSize: integer(encoder, "hidden_size", 256),
            intermediateSize: integer(encoder, "intermediate_size", 2048),
            heads: integer(encoder, "num_attention_heads", 8),
            encoderLayers: integer(encoder, "num_layers", 6),
            decoderLayers: integer(decoder, "num_layers", 6),
            queryCount: integer(decoder, "num_queries", 200),
            upsamplingStages: integer(masks, "num_upsampling_stages", 3))
    }

    /// Loads the detector out of a released checkpoint, ignoring everything else it holds.
    public static func loadDetectorWeights(into net: NFKMLXSAM3DetectorNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let prefix = "detector_model."
        let wanted = ["detr_encoder.", "detr_decoder.", "dot_product_scoring.", "mask_decoder."]
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix(prefix) else { return nil }
            let name = String(key.dropFirst(prefix.count))
            guard wanted.contains(where: { name.hasPrefix($0) }) else { return nil }
            guard checkpoint.needsConvTranspose, value.ndim == 4 else { return (name, value) }
            return (name, value.transposed(0, 2, 3, 1))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

/// SAM 3's image path: the vision encoder, the prompt side, and the detector, with the wiring
/// between them.
///
/// A worded prompt goes in and every instance it names comes out. The three networks load from one
/// released checkpoint, which is read once. Box prompts are NOT supported: the geometry encoder that
/// takes them is the one detector stage still unported, and a release's 94 tensors for it go unread.
public final class NFKMLXSAM3ImageModel {
    public let vision: NFKMLXSAM3VisionNet
    public let text: NFKMLXSAM3TextNet
    public let detector: NFKMLXSAM3DetectorNet

    public init(vision: NFKMLXSAM3VisionNet, text: NFKMLXSAM3TextNet,
                detector: NFKMLXSAM3DetectorNet) {
        self.vision = vision
        self.text = text
        self.detector = detector
    }

    /// Detects every instance the prompt names.
    ///
    /// - Parameters:
    ///   - image: the plate `[1, size, size, 3]`, normalized, at the vision configuration's size.
    ///   - tokens: the prompt's ids `[1, L]`, padded to the trained context.
    ///   - valid: 1 for a real prompt token and 0 for padding, `[1, L]`.
    public func detect(image: MLXArray, tokens: MLXArray, valid: MLXArray? = nil) -> NFKMLXSAM3Detection {
        // The detector reads every level but the coarsest, which the neck emits for the tracker.
        let levels = Array(vision(image).dropLast())
        let width = vision.configuration.fpnHiddenSize
        let positions = levels.map {
            NFKMLXSAM2PositionEmbedding.sine(height: $0.dim(1), width: $0.dim(2), features: width)
        }
        let padding = valid.map { (1 - $0) * -Float.greatestFiniteMagnitude }
        let prompt = text.prompt(tokens, padding: padding)
        return detector(levels: levels, positions: positions, prompt: prompt, promptValid: valid)
    }
}

extension NFKMLXSAM3 {

    /// The image path at a released configuration read from its `config.json`.
    public static func makeImageModel(fromHuggingFace url: URL) throws -> NFKMLXSAM3ImageModel {
        NFKMLXSAM3ImageModel(vision: makeVisionNet(try configuration(fromHuggingFace: url)),
                             text: makeTextNet(try textConfiguration(fromHuggingFace: url)),
                             detector: makeDetectorNet(try detectorConfiguration(fromHuggingFace: url)))
    }

    /// The image path at the released geometry.
    public static func makeImageModel() -> NFKMLXSAM3ImageModel {
        NFKMLXSAM3ImageModel(vision: makeVisionNet(.base), text: makeTextNet(.base),
                             detector: makeDetectorNet(.base))
    }

    /// Loads all three networks from one released checkpoint, reading the 3.44 GB file once.
    public static func loadWeights(into model: NFKMLXSAM3ImageModel, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let prefix = "detector_model."
        var visionArrays = [(String, MLXArray)]()
        var textArrays = [(String, MLXArray)]()
        var detectorArrays = [(String, MLXArray)]()
        let detectorStages = ["detr_encoder.", "detr_decoder.", "dot_product_scoring.", "mask_decoder."]

        for (key, value) in checkpoint.arrays {
            guard key.hasPrefix(prefix) else { continue }
            let name = String(key.dropFirst(prefix.count))
            let transposed = checkpoint.needsConvTranspose && value.ndim == 4
            if name.hasPrefix("vision_encoder.") {
                let inner = String(name.dropFirst("vision_encoder.".count))
                guard transposed else { visionArrays.append((inner, value)); continue }
                // The neck's scale layers are transposed convolutions; everything else is forward.
                visionArrays.append((inner, inner.contains("scale_layers")
                                     ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1)))
            } else if name.hasPrefix("text_encoder.") || name.hasPrefix("text_projection.") {
                textArrays.append((name, value))
            } else if detectorStages.contains(where: { name.hasPrefix($0) }) {
                detectorArrays.append((name, transposed ? value.transposed(0, 2, 3, 1) : value))
            }
        }
        try NFKMLXWeights.apply(visionArrays, to: model.vision)
        try NFKMLXWeights.apply(textArrays, to: model.text)
        try NFKMLXWeights.apply(detectorArrays, to: model.detector)
    }
}
