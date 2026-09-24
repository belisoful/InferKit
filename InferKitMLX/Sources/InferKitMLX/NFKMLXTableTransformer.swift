import Foundation
import MLX
import MLXFast
import MLXNN

/// The geometry of a Table Transformer release, read from its `config.json`.
///
/// Table Transformer (`TableTransformerForObjectDetection`, microsoft/table-transformer-*) is a
/// vanilla DETR: a ResNet-18 backbone with frozen batch norm, a normalized 2D sine position
/// embedding, a 1x1 projection to `dModel`, a pre-norm transformer encoder and decoder, and the
/// class / box heads over the decoder queries. The one architectural difference from post-norm DETR
/// is the pre-norm placement (a layer norm precedes each sub-block, and a final layer norm follows
/// each stack).
public struct NFKMLXTableTransformerConfiguration: Sendable {
    public var dModel: Int = 256
    public var encoderLayers: Int = 6
    public var decoderLayers: Int = 6
    public var encoderAttentionHeads: Int = 8
    public var decoderAttentionHeads: Int = 8
    public var encoderFFNDim: Int = 2048
    public var decoderFFNDim: Int = 2048
    public var numQueries: Int = 125
    public var numLabels: Int = 6
    public var labels: [String] = []

    public init() {}

    /// The released `microsoft/table-transformer-structure-recognition` geometry.
    public static var structureRecognition: NFKMLXTableTransformerConfiguration {
        var config = NFKMLXTableTransformerConfiguration()
        config.numLabels = 6
        config.labels = ["table", "table column", "table row", "table column header",
                         "table projected row header", "table spanning cell"]
        return config
    }

    /// Reads a release's `config.json`.
    public init(configurationURL url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("config.json is not a JSON object")
        }
        self.init()
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? Int) ?? fallback }
        dModel = int("d_model", dModel)
        encoderLayers = int("encoder_layers", encoderLayers)
        decoderLayers = int("decoder_layers", decoderLayers)
        encoderAttentionHeads = int("encoder_attention_heads", encoderAttentionHeads)
        decoderAttentionHeads = int("decoder_attention_heads", decoderAttentionHeads)
        encoderFFNDim = int("encoder_ffn_dim", encoderFFNDim)
        decoderFFNDim = int("decoder_ffn_dim", decoderFFNDim)
        numQueries = int("num_queries", numQueries)
        if let id2label = json["id2label"] as? [String: String] {
            numLabels = id2label.count
            labels = (0 ..< numLabels).map { id2label[String($0)] ?? "class \($0)" }
        }
    }
}

// MARK: - Backbone (ResNet-18 with frozen batch norm)

/// Batch norm with fixed statistics and affine parameters, folded into a per-channel scale and bias.
/// The backbone is frozen at inference, so the running statistics are constant. Channels last (NHWC),
/// so the `[C]` parameters broadcast against the last axis. Epsilon is the reference's 1e-5.
final class NFKTableTransformerFrozenBatchNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    init(_ channels: Int) {
        _weight.wrappedValue = MLXArray.ones([channels])
        _bias.wrappedValue = MLXArray.zeros([channels])
        _runningMean.wrappedValue = MLXArray.zeros([channels])
        _runningVar.wrappedValue = MLXArray.ones([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scale = weight * rsqrt(runningVar + 1e-5)
        return x * scale + (bias - runningMean * scale)
    }
}

/// A torchvision-style ResNet basic block: two 3x3 convolutions with frozen batch norm, and an
/// optional 1x1 downsample on the residual when the stage changes resolution or width.
final class NFKTableTransformerBasicBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: NFKTableTransformerFrozenBatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: NFKTableTransformerFrozenBatchNorm
    @ModuleInfo(key: "downsample") var downsample: [Module]?

    init(inChannels: Int, outChannels: Int, stride: Int, downsample: Bool) {
        _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                     kernelSize: 3, stride: IntOrPair(stride), padding: 1, bias: false)
        _bn1.wrappedValue = NFKTableTransformerFrozenBatchNorm(outChannels)
        _conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels,
                                     kernelSize: 3, stride: 1, padding: 1, bias: false)
        _bn2.wrappedValue = NFKTableTransformerFrozenBatchNorm(outChannels)
        if downsample {
            _downsample.wrappedValue = [
                Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                       kernelSize: 1, stride: IntOrPair(stride), bias: false),
                NFKTableTransformerFrozenBatchNorm(outChannels)]
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var identity = x
        if let downsample {
            identity = (downsample[1] as! NFKTableTransformerFrozenBatchNorm)((downsample[0] as! Conv2d)(x))
        }
        var out = relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        return relu(out + identity)
    }
}

/// The ResNet-18 feature extractor. Only the final stage (512 channels, stride 32) feeds the
/// transformer, matching the reference, which reads the last of the backbone's feature maps.
final class NFKTableTransformerResNet: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: NFKTableTransformerFrozenBatchNorm
    @ModuleInfo(key: "layer1") var layer1: [NFKTableTransformerBasicBlock]
    @ModuleInfo(key: "layer2") var layer2: [NFKTableTransformerBasicBlock]
    @ModuleInfo(key: "layer3") var layer3: [NFKTableTransformerBasicBlock]
    @ModuleInfo(key: "layer4") var layer4: [NFKTableTransformerBasicBlock]

    override init() {
        _conv1.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 64,
                                     kernelSize: 7, stride: 2, padding: 3, bias: false)
        _bn1.wrappedValue = NFKTableTransformerFrozenBatchNorm(64)
        _layer1.wrappedValue = NFKTableTransformerResNet.stage(in: 64, out: 64, stride: 1)
        _layer2.wrappedValue = NFKTableTransformerResNet.stage(in: 64, out: 128, stride: 2)
        _layer3.wrappedValue = NFKTableTransformerResNet.stage(in: 128, out: 256, stride: 2)
        _layer4.wrappedValue = NFKTableTransformerResNet.stage(in: 256, out: 512, stride: 2)
    }

    private static func stage(in inChannels: Int, out outChannels: Int, stride: Int) -> [NFKTableTransformerBasicBlock] {
        [NFKTableTransformerBasicBlock(inChannels: inChannels, outChannels: outChannels,
                                       stride: stride, downsample: stride != 1 || inChannels != outChannels),
         NFKTableTransformerBasicBlock(inChannels: outChannels, outChannels: outChannels,
                                       stride: 1, downsample: false)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = relu(bn1(conv1(x)))
        h = NFKMLXResample.maxPooled(h, kernel: 3, stride: 2, padding: 1)
        for block in layer1 { h = block(h) }
        for block in layer2 { h = block(h) }
        for block in layer3 { h = block(h) }
        for block in layer4 { h = block(h) }
        return h
    }
}

/// The two wrapper levels the reference's `TableTransformerConvEncoder` adds around the ResNet, so the
/// module keys mirror the checkpoint's `backbone.conv_encoder.model.*`.
final class NFKTableTransformerConvEncoder: Module {
    @ModuleInfo(key: "model") var model: NFKTableTransformerResNet

    override init() { _model.wrappedValue = NFKTableTransformerResNet() }

    func callAsFunction(_ x: MLXArray) -> MLXArray { model(x) }
}

final class NFKTableTransformerBackbone: Module {
    @ModuleInfo(key: "conv_encoder") var convEncoder: NFKTableTransformerConvEncoder

    override init() { _convEncoder.wrappedValue = NFKTableTransformerConvEncoder() }

    func callAsFunction(_ x: MLXArray) -> MLXArray { convEncoder(x) }
}

// MARK: - DETR transformer

/// DETR multi-head attention. The position embeddings are added to the queries and keys before their
/// projections; the values project the original hidden states without position. `queryPos` and
/// `keyValuePos` are the two independent position tensors: for self-attention they are the same, and
/// for the decoder's cross-attention `queryPos` is the object queries and `keyValuePos` the encoder's
/// spatial sine embedding.
final class NFKTableTransformerAttention: Module {
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int
    let headDim: Int

    init(_ embedDim: Int, heads: Int) {
        self.heads = heads
        self.headDim = embedDim / heads
        _kProj.wrappedValue = Linear(embedDim, embedDim)
        _vProj.wrappedValue = Linear(embedDim, embedDim)
        _qProj.wrappedValue = Linear(embedDim, embedDim)
        _outProj.wrappedValue = Linear(embedDim, embedDim)
    }

    private func split(_ t: MLXArray) -> MLXArray {
        t.reshaped([t.dim(0), t.dim(1), heads, headDim]).transposed(0, 2, 1, 3)
    }

    func callAsFunction(query: MLXArray, queryPos: MLXArray?, keyValue: MLXArray, keyValuePos: MLXArray?) -> MLXArray {
        let (b, n) = (query.dim(0), query.dim(1))
        let qIn = queryPos == nil ? query : query + queryPos!
        let kIn = keyValuePos == nil ? keyValue : keyValue + keyValuePos!
        let q = split(qProj(qIn))
        let k = split(kProj(kIn))
        let v = split(vProj(keyValue))
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDim)), mask: .none)
        return outProj(attn.transposed(0, 2, 1, 3).reshaped([b, n, heads * headDim]))
    }
}

/// A pre-norm DETR encoder layer: self-attention then a ReLU feed-forward, each over a normalized
/// input with a residual around the sub-block.
final class NFKTableTransformerEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKTableTransformerAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(_ config: NFKMLXTableTransformerConfiguration) {
        _selfAttn.wrappedValue = NFKTableTransformerAttention(config.dModel, heads: config.encoderAttentionHeads)
        _selfAttnNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        _fc1.wrappedValue = Linear(config.dModel, config.encoderFFNDim)
        _fc2.wrappedValue = Linear(config.encoderFFNDim, config.dModel)
        _finalNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ x: MLXArray, position: MLXArray) -> MLXArray {
        let normed = selfAttnNorm(x)
        var h = x + selfAttn(query: normed, queryPos: position, keyValue: normed, keyValuePos: position)
        h = h + fc2(relu(fc1(finalNorm(h))))
        return h
    }
}

/// The encoder stack, with the final layer norm the pre-norm design requires.
final class NFKTableTransformerEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKTableTransformerEncoderLayer]
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm

    init(_ config: NFKMLXTableTransformerConfiguration) {
        _layers.wrappedValue = (0 ..< config.encoderLayers).map { _ in NFKTableTransformerEncoderLayer(config) }
        _layernorm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ x: MLXArray, position: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h, position: position) }
        return layernorm(h)
    }
}

/// A pre-norm DETR decoder layer: self-attention over the queries, cross-attention into the encoder
/// memory, then a ReLU feed-forward.
final class NFKTableTransformerDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKTableTransformerAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttn: NFKTableTransformerAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(_ config: NFKMLXTableTransformerConfiguration) {
        _selfAttn.wrappedValue = NFKTableTransformerAttention(config.dModel, heads: config.decoderAttentionHeads)
        _selfAttnNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        _encoderAttn.wrappedValue = NFKTableTransformerAttention(config.dModel, heads: config.decoderAttentionHeads)
        _encoderAttnNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
        _fc1.wrappedValue = Linear(config.dModel, config.decoderFFNDim)
        _fc2.wrappedValue = Linear(config.decoderFFNDim, config.dModel)
        _finalNorm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ x: MLXArray, memory: MLXArray, spatialPosition: MLXArray, queryPosition: MLXArray) -> MLXArray {
        let selfNormed = selfAttnNorm(x)
        var h = x + selfAttn(query: selfNormed, queryPos: queryPosition,
                             keyValue: selfNormed, keyValuePos: queryPosition)
        let crossNormed = encoderAttnNorm(h)
        h = h + encoderAttn(query: crossNormed, queryPos: queryPosition,
                            keyValue: memory, keyValuePos: spatialPosition)
        h = h + fc2(relu(fc1(finalNorm(h))))
        return h
    }
}

/// The decoder stack, with the final layer norm the pre-norm design requires.
final class NFKTableTransformerDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKTableTransformerDecoderLayer]
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm

    init(_ config: NFKMLXTableTransformerConfiguration) {
        _layers.wrappedValue = (0 ..< config.decoderLayers).map { _ in NFKTableTransformerDecoderLayer(config) }
        _layernorm.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ x: MLXArray, memory: MLXArray, spatialPosition: MLXArray, queryPosition: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, memory: memory, spatialPosition: spatialPosition, queryPosition: queryPosition)
        }
        return layernorm(h)
    }
}

/// The box head: a three-layer perceptron with ReLU on all but the last layer.
final class NFKTableTransformerMLP: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(inputDim: Int, hiddenDim: Int, outputDim: Int, numLayers: Int) {
        let widths = [inputDim] + Array(repeating: hiddenDim, count: numLayers - 1) + [outputDim]
        _layers.wrappedValue = (0 ..< numLayers).map { Linear(widths[$0], widths[$0 + 1]) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for (index, layer) in layers.enumerated() {
            h = index < layers.count - 1 ? relu(layer(h)) : layer(h)
        }
        return h
    }
}

/// The base model: backbone, input projection, encoder, decoder, and the query position table.
final class NFKTableTransformerModel: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKTableTransformerBackbone
    @ModuleInfo(key: "input_projection") var inputProjection: Conv2d
    @ModuleInfo(key: "encoder") var encoder: NFKTableTransformerEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKTableTransformerDecoder
    @ModuleInfo(key: "query_position_embeddings") var queryPositionEmbeddings: Embedding

    init(_ config: NFKMLXTableTransformerConfiguration) {
        _backbone.wrappedValue = NFKTableTransformerBackbone()
        _inputProjection.wrappedValue = Conv2d(inputChannels: 512, outputChannels: config.dModel, kernelSize: 1)
        _encoder.wrappedValue = NFKTableTransformerEncoder(config)
        _decoder.wrappedValue = NFKTableTransformerDecoder(config)
        _queryPositionEmbeddings.wrappedValue = Embedding(embeddingCount: config.numQueries, dimensions: config.dModel)
    }
}

// MARK: - The detection network

/// Table Transformer (`TableTransformerForObjectDetection`) in `MLXNN`, at reference parity against
/// transformers' own model on the released weights.
public final class NFKMLXTableTransformerNet: Module {
    @ModuleInfo(key: "model") var model: NFKTableTransformerModel
    @ModuleInfo(key: "class_labels_classifier") var classifier: Linear
    @ModuleInfo(key: "bbox_predictor") var bboxPredictor: NFKTableTransformerMLP
    public let config: NFKMLXTableTransformerConfiguration

    public init(_ config: NFKMLXTableTransformerConfiguration) {
        self.config = config
        _model.wrappedValue = NFKTableTransformerModel(config)
        _classifier.wrappedValue = Linear(config.dModel, config.numLabels + 1)
        _bboxPredictor.wrappedValue = NFKTableTransformerMLP(
            inputDim: config.dModel, hiddenDim: config.dModel, outputDim: 4, numLayers: 3)
    }

    /// Reads a release's `config.json` for the geometry.
    public convenience init(configurationURL url: URL) throws {
        try self.init(NFKMLXTableTransformerConfiguration(configurationURL: url))
    }

    /// The staged outputs of one forward, for parity localization and the public detection.
    public struct Detection {
        public let backboneFeatures: MLXArray     // [1, H/32, W/32, 512] NHWC
        public let encoderLast: MLXArray          // [1, S, dModel]
        public let decoderLast: MLXArray          // [1, num_queries, dModel]
        public let logits: MLXArray               // [num_queries, num_labels + 1]
        public let boxes: MLXArray                // [num_queries, 4] cxcywh in 0...1
    }

    /// The normalized 2D sine position embedding over an `H x W` feature map, flattened to `[1, HW,
    /// dModel]` in the same row-major order the feature map flattens. For a fully valid image the mask
    /// is all ones, so the row and column indices are their cumulative sums. Each axis contributes
    /// `dModel / 2` channels; the sine and cosine of a shared frequency interleave, and the row
    /// channels precede the column channels.
    func sinePositionEmbedding(height: Int, width: Int) -> MLXArray {
        let half = config.dModel / 2                                       // 128
        let pairs = half / 2                                               // 64 shared frequencies
        let scale: Float = 2 * Float.pi
        let dimIndex = MLXArray(0 ..< pairs).asType(.float32)
        let frequency = exp(2 * dimIndex / Float(half) * Float(log(10000.0)))     // [pairs]

        func axis(_ length: Int) -> MLXArray {
            let index = MLXArray(0 ..< length).asType(.float32) + 1                              // [length]
            let embed = index / (Float(length) + 1e-6) * scale                                   // [length]
            let value = embed.reshaped([length, 1]) / frequency.reshaped([1, pairs])                // [length, pairs]
            let interleaved = MLX.stacked([sin(value), cos(value)], axis: 2)                         // [length, pairs, 2]
            return interleaved.reshaped([length, half])                                             // [length, 128]
        }

        let posY = axis(height)                                            // [H, 128]
        let posX = axis(width)                                             // [W, 128]
        let yGrid = broadcast(posY.reshaped([height, 1, half]), to: [height, width, half])
        let xGrid = broadcast(posX.reshaped([1, width, half]), to: [height, width, half])
        return concatenated([yGrid, xGrid], axis: 2).reshaped([1, height * width, config.dModel])
    }

    /// `pixels` `[1, H, W, 3]` NHWC (already normalized) → the full staged detection.
    public func callAsFunction(_ pixels: MLXArray) -> Detection {
        let features = model.backbone(pixels)                              // [1, fh, fw, 512]
        let projected = model.inputProjection(features)                    // [1, fh, fw, dModel]
        let (fh, fw) = (projected.dim(1), projected.dim(2))
        let source = projected.reshaped([1, fh * fw, config.dModel])       // [1, S, dModel]
        let spatialPosition = sinePositionEmbedding(height: fh, width: fw) // [1, S, dModel]

        let memory = model.encoder(source, position: spatialPosition)      // [1, S, dModel]

        let queryPosition = model.queryPositionEmbeddings.weight.expandedDimensions(axis: 0)   // [1, Q, dModel]
        let queries = MLXArray.zeros([1, config.numQueries, config.dModel])
        let decoded = model.decoder(queries, memory: memory,
                                    spatialPosition: spatialPosition, queryPosition: queryPosition)  // [1, Q, dModel]

        let logits = classifier(decoded)[0]                                // [Q, num_labels + 1]
        let boxes = sigmoid(bboxPredictor(decoded))[0]                     // [Q, 4]
        return Detection(backboneFeatures: features, encoderLast: memory,
                         decoderLast: decoded, logits: logits, boxes: boxes)
    }
}
