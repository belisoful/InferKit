//
//  NFKMLXFlorence2Model.swift
//  InferKitMLX
//

// The full Florence-2 model: the DaViT vision tower and projector (NFKMLXFlorence2.swift) feeding a BART
// encoder-decoder (the shared NFKMLXSeq2SeqTransformer). The projected image tokens concatenate BEFORE
// the text-prompt embeddings (image first, the reference's `_merge_input_ids_with_image_features`), the
// encoder runs over the joined sequence, and the decoder cross-attends and emits the vocabulary (task
// and location tokens included). The BART weights load through the shared seq2seq loader after the
// `language_model.model.` prefix is stripped.

import Foundation
import InferKit
import MLX
import MLXNN

public final class NFKMLXFlorence2Net: Module {
    @ModuleInfo(key: "vision_tower") var vision: NFKMLXFlorence2VisionNet
    @ModuleInfo(key: "projector") var projector: NFKMLXFlorence2Projector
    @ModuleInfo(key: "language_model") var language: NFKMLXSeq2SeqNet

    let visionConfig: NFKMLXFlorence2VisionConfiguration
    let textConfig: NFKMLXSeq2SeqConfiguration
    public let imageTokenId: Int

    /// Florence-2's BART-large text model.
    public static let bartLarge = NFKMLXSeq2SeqConfiguration(
        vocabularySize: 51289, dModel: 1024, encoderLayers: 12, decoderLayers: 12, heads: 16,
        encoderFFDim: 4096, decoderFFDim: 4096, maxPositions: 4096,
        activation: .gelu, positions: .learned, normalizeBefore: false, finalLayerNorm: false,
        layerNormEmbedding: true, scaleEmbedding: false, finalLogitsBias: true,
        layerNormEps: 1e-5, padTokenId: 1, eosTokenId: 2, decoderStartTokenId: 2)

    /// Florence-2-base's BART-base text model.
    public static let bartBase = NFKMLXSeq2SeqConfiguration(
        vocabularySize: 51289, dModel: 768, encoderLayers: 6, decoderLayers: 6, heads: 12,
        encoderFFDim: 3072, decoderFFDim: 3072, maxPositions: 1024,
        activation: .gelu, positions: .learned, normalizeBefore: false, finalLayerNorm: false,
        layerNormEmbedding: true, scaleEmbedding: false, finalLogitsBias: true,
        layerNormEps: 1e-5, padTokenId: 1, eosTokenId: 2, decoderStartTokenId: 2)

    /// The vision and text geometry a release's `config.json` states. The DaViT widths, heads, groups,
    /// depths, patches, and window come from `vision_config`, the projection from its `projection_dim`,
    /// and the BART sizes from `text_config`; what a config leaves out keeps Florence-2-large's value.
    public static func configuration(fromConfigURL url: URL) throws
        -> (vision: NFKMLXFlorence2VisionConfiguration, text: NFKMLXSeq2SeqConfiguration) {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        let v = json["vision_config"] as? [String: Any] ?? [:]
        let t = json["text_config"] as? [String: Any] ?? [:]
        func ints(_ d: [String: Any], _ key: String, _ fallback: [Int]) -> [Int] { (d[key] as? [NSNumber])?.map(\.intValue) ?? fallback }
        func int(_ d: [String: Any], _ key: String, _ fallback: Int) -> Int { (d[key] as? NSNumber)?.intValue ?? fallback }
        let large = NFKMLXFlorence2VisionConfiguration.large
        let positions = (v["image_pos_embed"] as? [String: Any]).map { int($0, "max_pos_embeddings", large.maxPositionEmbeddings) }
        let vision = NFKMLXFlorence2VisionConfiguration(
            depths: ints(v, "depths", large.depths), patchSize: ints(v, "patch_size", large.patchSize),
            patchStride: ints(v, "patch_stride", large.patchStride), patchPadding: ints(v, "patch_padding", large.patchPadding),
            patchPreNorm: (v["patch_prenorm"] as? [Bool]) ?? large.patchPreNorm,
            embedDim: ints(v, "dim_embed", large.embedDim), numHeads: ints(v, "num_heads", large.numHeads),
            numGroups: ints(v, "num_groups", large.numGroups), windowSize: int(v, "window_size", large.windowSize),
            projectionDim: int(v, "projection_dim", int(json, "projection_dim", large.projectionDim)),
            maxPositionEmbeddings: positions ?? large.maxPositionEmbeddings)
        var text = bartLarge
        text.vocabularySize = int(t, "vocab_size", text.vocabularySize)
        text.dModel = int(t, "d_model", text.dModel)
        text.encoderLayers = int(t, "encoder_layers", text.encoderLayers)
        text.decoderLayers = int(t, "decoder_layers", text.decoderLayers)
        text.heads = int(t, "encoder_attention_heads", text.heads)
        text.encoderFFDim = int(t, "encoder_ffn_dim", text.encoderFFDim)
        text.decoderFFDim = int(t, "decoder_ffn_dim", text.decoderFFDim)
        text.maxPositions = int(t, "max_position_embeddings", text.maxPositions)
        return (vision, text)
    }

    public init(vision visionConfig: NFKMLXFlorence2VisionConfiguration = .large,
                text textConfig: NFKMLXSeq2SeqConfiguration = NFKMLXFlorence2Net.bartLarge,
                imageTokenId: Int = 51289) {
        self.visionConfig = visionConfig
        self.textConfig = textConfig
        self.imageTokenId = imageTokenId
        _vision.wrappedValue = NFKMLXFlorence2VisionNet(visionConfig)
        _projector.wrappedValue = NFKMLXFlorence2Projector(embedDim: visionConfig.embedDim.last!,
                                                           projectionDim: visionConfig.projectionDim,
                                                           maxPositionEmbeddings: visionConfig.maxPositionEmbeddings)
        _language.wrappedValue = NFKMLXSeq2SeqNet(textConfig)
        super.init()
    }

    /// `pixels`: `[B, H, W, 3]` (NHWC, normalized). Returns the projected image tokens `[B, 1+H*W, dModel]`.
    public func imageFeatures(_ pixels: MLXArray) -> MLXArray { projector(vision(pixels)) }

    /// Concatenates the image tokens with the text embeddings (image first), scaled by the BART embedding
    /// scale, ready for `NFKMLXSeq2SeqNet.encode(embeddings:)`.
    public func fusedEmbeddings(imageFeatures: MLXArray, inputIds: MLXArray) -> MLXArray {
        let scale: Float = textConfig.scaleEmbedding ? sqrtf(Float(textConfig.dModel)) : 1
        let text = language.shared(inputIds).asType(.float32) * scale
        return concatenated([imageFeatures, text], axis: 1)
    }

    /// The encoder output over the joined image+prompt sequence `[B, 1+H*W+T, dModel]`.
    public func encode(pixels: MLXArray, inputIds: MLXArray) -> MLXArray {
        language.encode(embeddings: fusedEmbeddings(imageFeatures: imageFeatures(pixels), inputIds: inputIds))
    }

    /// Loads a released Florence-2 checkpoint into all three parts in one pass. Vision and projector go
    /// through the DaViT remap; the BART subtree drops its `language_model.model.` prefix and routes
    /// through the shared seq2seq key map (which drops the tied embedding/head copies).
    public func loadWeights(from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        // A file `NFKMLXWeights.save` wrote (a fine-tune) holds this module's own names and layout.
        guard checkpoint.needsConvTranspose else {
            try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value) }, to: self)
            return
        }
        var arrays: [(String, MLXArray)] = []
        for (key, value) in checkpoint.arrays {
            if let mapped = NFKMLXFlorence2Weights.visionKey(key) {
                let transposed = value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
                arrays.append(("vision_tower." + mapped, transposed.asType(.float32)))
            } else if let mapped = NFKMLXFlorence2Weights.projectorKey(key) {
                arrays.append(("projector." + mapped, value.asType(.float32)))
            } else if key.hasPrefix("language_model.") {
                let stripped = String(key.dropFirst("language_model.".count))
                if let mapped = NFKMLXSeq2SeqNet.moduleKey(for: stripped,
                                                           learnedPositions: textConfig.positions == .learned) {
                    arrays.append(("language_model." + mapped, value.asType(.float32)))
                }
            }
        }
        try NFKMLXWeights.apply(arrays, to: self)
    }
}
