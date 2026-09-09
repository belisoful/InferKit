//
//  NFKMLXGemma3nFusion.swift
//  InferKitMLX
//
//  Joining Gemma 3n's three modalities: the embedder that lifts a tower's output into the decoder's
//  space, and the model that splices the result into a prompt.
//
//  Gemma 3n splices in two stages, which is unlike the other multimodal models here. A placeholder
//  token is first embedded HARD, through a small per-modality table indexed by an offset into the
//  token vocabulary; the soft tokens the tower produced then overwrite those positions. A caller who
//  only replaced the soft positions would still be right, and a caller who only did the hard pass
//  would get a model that runs and ignores the picture.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// Lifts a tower's output, or a per-modality token id, into the decoder's embedding space.
public final class NFKGemma3nMultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "hard_embedding_norm") var hardNorm: NFKGemma3nNorm
    @ModuleInfo(key: "soft_embedding_norm") var softNorm: NFKGemma3nNorm
    @ModuleInfo(key: "embedding_projection") var projection: Linear
    @ModuleInfo(key: "embedding_post_projection_norm") var postNorm: NFKGemma3nNorm

    /// The first token id this modality owns.
    public let vocabularyOffset: Int
    /// How many ids it owns.
    public let vocabularySize: Int

    public init(hiddenSize: Int, textHiddenSize: Int, vocabularySize: Int, vocabularyOffset: Int,
                eps: Float = 1e-6) {
        self.vocabularyOffset = vocabularyOffset
        self.vocabularySize = vocabularySize
        _embedding.wrappedValue = Embedding(embeddingCount: vocabularySize, dimensions: hiddenSize)
        _hardNorm.wrappedValue = NFKGemma3nNorm(dimensions: hiddenSize, eps: eps)
        _softNorm.wrappedValue = NFKGemma3nNorm(dimensions: hiddenSize, eps: eps)
        _projection.wrappedValue = Linear(hiddenSize, textHiddenSize, bias: false)
        // The closing normalization carries no weight, so the checkpoint has no tensor for it.
        _postNorm.wrappedValue = NFKGemma3nNorm(dimensions: textHiddenSize, eps: eps, scaled: false)
        super.init()
    }

    /// A tower's output, projected into the decoder's space.
    public func callAsFunction(soft tokens: MLXArray) -> MLXArray {
        postNorm(projection(softNorm(tokens)))
    }

    /// A per-modality token id, projected into the decoder's space. The ids are absolute, so the
    /// modality's offset comes off before the table is read.
    public func callAsFunction(hard ids: MLXArray) -> MLXArray {
        postNorm(projection(hardNorm(embedding(ids - Int32(vocabularyOffset)))))
    }
}

/// Which token ids stand in for what, and how many soft tokens a picture or a clip becomes.
public struct NFKMLXGemma3nTokens: Sendable {
    public var imageToken: Int
    public var audioToken: Int
    public var visionOffset: Int
    public var visionVocabularySize: Int
    public var audioOffset: Int
    public var audioVocabularySize: Int
    public var visionSoftTokens: Int
    public var audioSoftTokens: Int

    public init(imageToken: Int = 262_145, audioToken: Int = 262_273,
                visionOffset: Int = 262_144, visionVocabularySize: Int = 128,
                audioOffset: Int = 262_272, audioVocabularySize: Int = 128,
                visionSoftTokens: Int = 256, audioSoftTokens: Int = 188) {
        self.imageToken = imageToken
        self.audioToken = audioToken
        self.visionOffset = visionOffset
        self.visionVocabularySize = visionVocabularySize
        self.audioOffset = audioOffset
        self.audioVocabularySize = audioVocabularySize
        self.visionSoftTokens = visionSoftTokens
        self.audioSoftTokens = audioSoftTokens
    }
}

/// The tri-modal Gemma 3n: the decoder, the towers a release carries, and the embedders between them.
public final class NFKMLXGemma3nModel: Module {
    @ModuleInfo(key: "language_model") var decoder: NFKMLXGemma3nNet
    @ModuleInfo(key: "vision_tower") var vision: NFKMLXGemma3nVisionNet?
    @ModuleInfo(key: "audio_tower") var audio: NFKMLXGemma3nAudioNet?
    @ModuleInfo(key: "embed_vision") var visionEmbedder: NFKGemma3nMultimodalEmbedder?
    @ModuleInfo(key: "embed_audio") var audioEmbedder: NFKGemma3nMultimodalEmbedder?

    public let tokens: NFKMLXGemma3nTokens

    public init(decoder: NFKMLXGemma3nNet, vision: NFKMLXGemma3nVisionNet?, audio: NFKMLXGemma3nAudioNet?,
                visionEmbedder: NFKGemma3nMultimodalEmbedder?, audioEmbedder: NFKGemma3nMultimodalEmbedder?,
                tokens: NFKMLXGemma3nTokens = NFKMLXGemma3nTokens()) {
        self.tokens = tokens
        _decoder.wrappedValue = decoder
        _vision.wrappedValue = vision
        _audio.wrappedValue = audio
        _visionEmbedder.wrappedValue = visionEmbedder
        _audioEmbedder.wrappedValue = audioEmbedder
        super.init()
    }

    /// A frame's soft tokens in the decoder's space: `[batch, visionSoftTokens, hidden]`.
    ///
    /// @discussion The grid is scaled by the square root of its own width before the embedder reads
    /// it, which the normalization then mostly removes — mostly, because the epsilon does not scale.
    public func imageFeatures(_ pixels: MLXArray) throws -> MLXArray {
        guard let vision, let visionEmbedder else {
            throw NFKMLXError.unsupportedConfiguration("this Gemma 3n carries no vision tower")
        }
        let grid = vision.softTokens(pixels)
        return visionEmbedder(soft: grid * sqrt(Float(grid.shape[2])))
    }

    /// A clip's soft tokens in the decoder's space, padded to `audioSoftTokens` with the modality's
    /// last id, which is what the reference pads short clips with.
    public func audioFeatures(_ mel: MLXArray, valid: MLXArray?) throws -> MLXArray {
        guard let audio, let audioEmbedder else {
            throw NFKMLXError.unsupportedConfiguration("this Gemma 3n carries no audio tower")
        }
        let (encoded, mask) = audio(mel, valid: valid)
        var features = audioEmbedder(soft: encoded)

        let padding = audioEmbedder(hard: MLXArray([Int32(audioEmbedder.vocabularyOffset
                                                          + audioEmbedder.vocabularySize - 1)]))
            .reshaped([1, 1, features.shape[2]])
        // A frame the encoder produced but the clip did not fill takes the padding embedding.
        if let mask {
            features = MLX.where(mask.expandedDimensions(axis: -1), features, padding)
        }
        let shortfall = tokens.audioSoftTokens - features.shape[1]
        if shortfall > 0 {
            features = concatenated(
                [features, broadcast(padding, to: [features.shape[0], shortfall, features.shape[2]])],
                axis: 1)
        }
        return features
    }

    /// The decoder's input embeddings for a prompt, with each modality's placeholders filled.
    ///
    /// @discussion The placeholders are embedded twice: once through the modality's own table, which
    /// is what a release does for every id in its range, and again by the tower's soft tokens, which
    /// overwrite them. Only the second pass carries the picture.
    public func fusedEmbeddings(tokenIds: MLXArray, image: MLXArray? = nil,
                                audioMel: MLXArray? = nil, audioValid: MLXArray? = nil) throws
        -> (embeddings: MLXArray, perLayerInputs: MLXArray) {
        var embeddings = decoder.embed(tokenIds)
        let perLayer = decoder.perLayerEmbeddings(tokenIds)

        // The bound is the NEXT modality's first id, or everything above this one. `Int.max` would
        // trap converting to the `Int32` the comparison runs in.
        if let visionEmbedder {
            embeddings = replaced(embeddings, tokenIds: tokenIds, embedder: visionEmbedder,
                                  upperBound: Int32(audioEmbedder?.vocabularyOffset ?? Int(Int32.max)))
        }
        if let audioEmbedder {
            embeddings = replaced(embeddings, tokenIds: tokenIds, embedder: audioEmbedder,
                                  upperBound: Int32.max)
        }
        if let image {
            embeddings = spliced(embeddings, at: tokenIds .== Int32(tokens.imageToken),
                                 features: try imageFeatures(image))
        }
        if let audioMel {
            embeddings = spliced(embeddings, at: tokenIds .== Int32(tokens.audioToken),
                                 features: try audioFeatures(audioMel, valid: audioValid))
        }
        return (embeddings, decoder.projectedPerLayerInputs(embeddings: embeddings, perLayer: perLayer))
    }

    /// The logits for a prompt carrying at most one picture and one clip.
    public func callAsFunction(_ tokenIds: MLXArray, image: MLXArray? = nil, audioMel: MLXArray? = nil,
                               audioValid: MLXArray? = nil, cache: NFKMLXGemma3nCache? = nil) throws
        -> MLXArray {
        let (embeddings, perLayer) = try fusedEmbeddings(tokenIds: tokenIds, image: image,
                                                         audioMel: audioMel, audioValid: audioValid)
        return decoder.logits(fromHidden: decoder.hiddenStates(fromEmbeddings: embeddings,
                                                               perLayerInputs: perLayer, cache: cache))
    }

    /// The ids a modality owns, embedded through its own table in place of the text embedding.
    private func replaced(_ embeddings: MLXArray, tokenIds: MLXArray,
                          embedder: NFKGemma3nMultimodalEmbedder, upperBound: Int32) -> MLXArray {
        let owned = (tokenIds .>= Int32(embedder.vocabularyOffset)) .&& (tokenIds .< upperBound)
        // An id outside the range would index the table out of bounds, so the others are read as the
        // modality's last id and then discarded.
        let dummy = Int32(embedder.vocabularyOffset + embedder.vocabularySize - 1)
        let safe = MLX.where(owned, tokenIds, MLXArray(dummy))
        return MLX.where(owned.expandedDimensions(axis: -1), embedder(hard: safe), embeddings)
    }

    /// `features` written into the positions `mask` marks, in order.
    ///
    /// @discussion MLX has no masked scatter, so the position's rank among the marked ones is its
    /// index into the features — a running count, which is exact whether the marked positions are
    /// contiguous or not.
    private func spliced(_ embeddings: MLXArray, at mask: MLXArray, features: MLXArray) -> MLXArray {
        let (batch, perBatch) = (features.shape[0], features.shape[1])
        let flat = features.reshaped([-1, features.shape[features.ndim - 1]])
        // Each row's rank counts within its own batch entry, so the entry's own block of features is
        // where it reads.
        let starts = MLXArray((0 ..< batch).map { Int32($0 * perBatch) }).reshaped([batch, 1])
        let rank = cumsum(mask.asType(.int32), axis: 1) - 1 + starts
        let bounded = clip(rank, min: MLXArray(Int32(0)), max: MLXArray(Int32(flat.shape[0] - 1)))
        let gathered = flat.take(bounded.reshaped([-1]), axis: 0).reshaped(embeddings.shape)
        return MLX.where(mask.expandedDimensions(axis: -1), gathered, embeddings)
    }
}

// MARK: - Building a release

/// Building the tri-modal Gemma 3n from a released directory.
@objc(NFKMLXGemma3n)
public final class NFKMLXGemma3n: NSObject {
    /// The networks: the decoder and whichever towers the release carries.
    public let model: NFKMLXGemma3nModel
    /// The release's own tokenizer, which is Gemma's byte-fallback SentencePiece BPE.
    let tokenizer: NFKMLXGemmaTokenizer

    init(model: NFKMLXGemma3nModel, tokenizer: NFKMLXGemmaTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
        super.init()
    }

    /// The token vocabulary a release states, read from its `config.json`.
    public static func tokens(fromJSON json: [String: Any]) -> NFKMLXGemma3nTokens {
        func integer(_ container: [String: Any], _ key: String, _ fallback: Int) -> Int {
            (container[key] as? NSNumber)?.intValue ?? fallback
        }
        let vision = (json["vision_config"] as? [String: Any]) ?? [:]
        let audio = (json["audio_config"] as? [String: Any]) ?? [:]
        return NFKMLXGemma3nTokens(
            imageToken: integer(json, "image_token_id", 262_145),
            audioToken: integer(json, "audio_token_id", 262_273),
            visionOffset: integer(vision, "vocab_offset", 262_144),
            visionVocabularySize: integer(vision, "vocab_size", 128),
            audioOffset: integer(audio, "vocab_offset", 262_272),
            audioVocabularySize: integer(audio, "vocab_size", 128),
            visionSoftTokens: integer(json, "vision_soft_tokens_per_image", 256),
            audioSoftTokens: integer(json, "audio_soft_tokens_per_image", 188))
    }

    /// The tri-modal model's module key for a checkpoint key, or nil for a tensor it does not hold.
    static func name(of key: String, decoder: NFKMLXGemma3nConfiguration) -> String? {
        if key.hasSuffix("lm_head.weight") { return nil }
        if let tower = NFKMLXGemma3nVision.towerName(of: key) { return "vision_tower.\(tower)" }
        if let tower = NFKMLXGemma3nAudio.encoderName(of: key) { return "audio_tower.\(tower)" }
        for embedder in ["embed_vision", "embed_audio"] {
            if let name = NFKMLXGemma3nLanguage.stripped(key, prefixes: ["model.\(embedder).", "\(embedder)."]) {
                return "\(embedder).\(name)"
            }
        }
        guard let name = NFKMLXGemma3nLanguage.decoderName(of: key, configuration: decoder) else { return nil }
        return "language_model.\(name)"
    }

    /// A checkpoint tensor in MLX's layout. Only the towers hold convolutions, and each converts its
    /// own the way its loader does.
    static func converted(_ name: String, _ value: MLXArray) -> MLXArray {
        if name.hasPrefix("vision_tower.") {
            return NFKMLXGemma3nVision.converted(name, value)
        }
        if name.hasPrefix("audio_tower.") {
            return NFKMLXGemma3nAudio.converted(String(name.dropFirst("audio_tower.".count)), value)
        }
        return value
    }

    /// Reads a released directory and builds the whole model: the decoder, whichever towers the
    /// release carries, and the embedders between them.
    public static func model(directoryURL directory: URL,
                             precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXGemma3nModel {
        let configURL = directory.appendingPathComponent("config.json")
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL))
                as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("config.json is not a JSON object")
        }
        let textConfiguration = try NFKMLXGemma3nLanguage.configuration(fromJSON: json)
        let tokenVocabulary = tokens(fromJSON: json)

        let decoder = NFKMLXGemma3nNet(textConfiguration)
        var vision: NFKMLXGemma3nVisionNet?
        var audio: NFKMLXGemma3nAudioNet?
        var visionEmbedder: NFKGemma3nMultimodalEmbedder?
        var audioEmbedder: NFKGemma3nMultimodalEmbedder?

        if let visionConfiguration = json["vision_config"] as? [String: Any] {
            let width = (visionConfiguration["hidden_size"] as? NSNumber)?.intValue ?? 2048
            let grid = Int(Double(tokenVocabulary.visionSoftTokens).squareRoot().rounded())
            vision = NFKMLXGemma3nVisionNet(tokenGrid: grid, outputChannels: width)
            visionEmbedder = NFKGemma3nMultimodalEmbedder(
                hiddenSize: width, textHiddenSize: textConfiguration.hiddenSize,
                vocabularySize: tokenVocabulary.visionVocabularySize,
                vocabularyOffset: tokenVocabulary.visionOffset,
                eps: (visionConfiguration["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6)
        }
        if json["audio_config"] != nil {
            let audioConfiguration = try NFKMLXGemma3nAudio.configuration(fromJSON: json)
            audio = NFKMLXGemma3nAudioNet(audioConfiguration)
            audioEmbedder = NFKGemma3nMultimodalEmbedder(
                hiddenSize: audioConfiguration.hiddenSize, textHiddenSize: textConfiguration.hiddenSize,
                vocabularySize: tokenVocabulary.audioVocabularySize,
                vocabularyOffset: tokenVocabulary.audioOffset, eps: audioConfiguration.rmsEpsilon)
        }

        let model = NFKMLXGemma3nModel(decoder: decoder, vision: vision, audio: audio,
                                       visionEmbedder: visionEmbedder, audioEmbedder: audioEmbedder,
                                       tokens: tokenVocabulary)
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision,
                                                     remap: { name(of: $0, decoder: textConfiguration) })
        try NFKMLXWeights.apply(mapped.map { ($0.0, converted($0.0, $0.1)) }, to: model)
        return model
    }
}
