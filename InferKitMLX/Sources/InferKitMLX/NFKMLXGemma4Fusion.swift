//
//  NFKMLXGemma4Fusion.swift
//  InferKitMLX
//
//  The multimodal fusion that joins the Gemma 4 vision and audio towers to the text decoder: a tower's
//  soft tokens are projected into the language model's space by a `Gemma4MultimodalEmbedder`, and the
//  projected tokens replace the text embeddings at the placeholder positions the tokenizer emitted for
//  the image or audio.
//
//  The embedder and the splice are the novel pieces and are verified here. Running the whole chain end
//  to end (image/audio to an answer) additionally needs the released tri-modal weights — the vision and
//  audio towers, the embedders, and the E-series decoder together — which the text-only E2B release does
//  not carry; the components are each at reference parity.
//

import Foundation
import CoreGraphics
import MLX
import MLXNN

/// Projects a tower's soft tokens into the language model's embedding space: a scale-free RMS norm,
/// then a bias-free linear projection to the text hidden size.
public final class NFKMLXGemma4MultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var projection: Linear
    let epsilon: Float

    public init(multimodalHidden: Int, textHidden: Int, eps: Float = 1e-6) {
        _projection.wrappedValue = Linear(multimodalHidden, textHidden, bias: false)
        epsilon = eps
        super.init()
    }

    public func callAsFunction(_ softTokens: MLXArray) -> MLXArray {
        let normalized = softTokens * rsqrt((softTokens * softTokens).mean(axis: -1, keepDims: true) + epsilon)
        return projection(normalized)
    }
}

/// The splice that fuses projected soft tokens into a text embedding sequence.
public enum NFKMLXGemma4Fusion {
    /// Replaces the embeddings at the placeholder positions with the projected soft tokens, in order.
    /// `textEmbeddings` is `[1, sequence, hidden]`, `softTokens` is `[count, hidden]`, and
    /// `isPlaceholder` marks the sequence positions that carry a soft token (their count must equal
    /// `softTokens`'s). This is the same `where`-over-a-gathered-index splice the SmolVLM fusion uses.
    public static func fuse(textEmbeddings: MLXArray, softTokens: MLXArray,
                            isPlaceholder: [Bool]) -> MLXArray {
        let sequence = isPlaceholder.count
        let hidden = textEmbeddings.shape[textEmbeddings.ndim - 1]
        let flatSoft = softTokens.reshaped([-1, hidden])

        var featureIndex = [Int32](repeating: 0, count: sequence)
        var placeholder = [Float](repeating: 0, count: sequence)
        var counter: Int32 = 0
        for position in 0 ..< sequence where isPlaceholder[position] {
            featureIndex[position] = counter
            counter += 1
            placeholder[position] = 1
        }
        let gathered = flatSoft.take(MLXArray(featureIndex), axis: 0)          // [sequence, hidden]
        let mask = MLXArray(placeholder).reshaped([sequence, 1]) .> 0
        let text = textEmbeddings.reshaped([sequence, hidden])
        return MLX.where(mask, gathered, text).reshaped([1, sequence, hidden])
    }
}

/// The full `Gemma4ForConditionalGeneration` chain: an image and/or audio and a token sequence with
/// placeholder tokens in, a generated continuation out. It processes the image and audio into the
/// towers, projects their soft tokens into the decoder's space, splices them at the placeholder
/// positions, and runs the E-series decoder prefill-only over the fused embeddings.
///
/// The decoder is prefill-only (Gemma carries no key-value cache), so each step re-runs the growing
/// sequence with the same soft tokens spliced. The whole numeric path is verified only once the
/// released tri-modal weights are available; every component it chains is at reference parity.
public final class NFKMLXGemma4ConditionalGeneration {
    private let decoder: NFKMLXGemmaNet
    private let visionTower: NFKMLXGemma4VisionNet?
    private let visionEmbedder: NFKMLXGemma4MultimodalEmbedder?
    private let audioTower: NFKMLXGemma4AudioNet?
    private let audioEmbedder: NFKMLXGemma4MultimodalEmbedder?
    private let imageProcessor: NFKMLXGemma4ImageProcessor
    private let featureExtractor: NFKMLXGemma4AudioFeatureExtractor
    private let imageTokenId: Int
    private let audioTokenId: Int
    private let padTokenId: Int
    private let stopTokens: Set<Int>

    public init(decoder: NFKMLXGemmaNet,
                visionTower: NFKMLXGemma4VisionNet? = nil,
                visionEmbedder: NFKMLXGemma4MultimodalEmbedder? = nil,
                audioTower: NFKMLXGemma4AudioNet? = nil,
                audioEmbedder: NFKMLXGemma4MultimodalEmbedder? = nil,
                imageProcessor: NFKMLXGemma4ImageProcessor = NFKMLXGemma4ImageProcessor(),
                featureExtractor: NFKMLXGemma4AudioFeatureExtractor = NFKMLXGemma4AudioFeatureExtractor(),
                imageTokenId: Int, audioTokenId: Int, padTokenId: Int, stopTokens: Set<Int> = []) {
        self.decoder = decoder
        self.visionTower = visionTower
        self.visionEmbedder = visionEmbedder
        self.audioTower = audioTower
        self.audioEmbedder = audioEmbedder
        self.imageProcessor = imageProcessor
        self.featureExtractor = featureExtractor
        self.imageTokenId = imageTokenId
        self.audioTokenId = audioTokenId
        self.padTokenId = padTokenId
        self.stopTokens = stopTokens
    }

    /// The main embeddings for a token sequence, with the image and audio soft tokens spliced in at
    /// their placeholder positions. A caller with pre-computed soft tokens can splice them directly;
    /// this is the seam the generation loop reuses each step.
    public func fusedEmbeddings(tokens: [Int], visionSoftTokens: MLXArray?,
                                audioSoftTokens: MLXArray?) -> (embeddings: MLXArray, tokens: MLXArray) {
        // A placeholder is embedded as the pad token; the soft token replaces it afterward.
        let padded = tokens.map { ($0 == imageTokenId || $0 == audioTokenId) ? padTokenId : $0 }
        let ids = MLXArray(padded.map(Int32.init)).reshaped([1, padded.count])
        var embeddings = decoder.embed(ids)
        if let visionSoftTokens {
            embeddings = NFKMLXGemma4Fusion.fuse(textEmbeddings: embeddings, softTokens: visionSoftTokens,
                                                 isPlaceholder: tokens.map { $0 == imageTokenId })
        }
        if let audioSoftTokens {
            embeddings = NFKMLXGemma4Fusion.fuse(textEmbeddings: embeddings, softTokens: audioSoftTokens,
                                                 isPlaceholder: tokens.map { $0 == audioTokenId })
        }
        return (embeddings, ids)
    }

    /// The projected vision soft tokens for an image, or nil when no vision tower is configured.
    public func visionSoftTokens(for image: CGImage) -> MLXArray? {
        guard let visionTower, let visionEmbedder else { return nil }
        let (pixels, positions) = imageProcessor.process(image)
        let soft = visionTower.softTokens(pixels.reshaped([1, pixels.shape[0], pixels.shape[1]]),
                                          positionIds: positions.reshaped([1, positions.shape[0], 2]))
        return visionEmbedder(soft).reshaped([-1, soft.shape[soft.ndim - 1]]).reshaped([-1, decoderHidden])
    }

    /// The projected audio soft tokens for a waveform, or nil when no audio tower is configured.
    public func audioSoftTokens(for waveform: [Float]) -> MLXArray? {
        guard let audioTower, let audioEmbedder else { return nil }
        let features = featureExtractor.features(waveform)
        let encoded = audioTower(features.reshaped([1, features.shape[0], features.shape[1]]))
        return audioEmbedder(encoded).reshaped([-1, decoderHidden])
    }

    private var decoderHidden: Int { decoder.configuration.hiddenSize }

    /// Greedy prefill-only generation from an image and/or audio and a placeholder-carrying prompt.
    public func generate(promptTokens: [Int], image: CGImage? = nil, waveform: [Float]? = nil,
                         maxTokens: Int = 64) -> [Int] {
        let visionSoft = image.flatMap(visionSoftTokens)
        let audioSoft = waveform.flatMap(audioSoftTokens)
        var tokens = promptTokens
        var produced = [Int]()
        for _ in 0 ..< max(maxTokens, 0) {
            let (embeddings, ids) = fusedEmbeddings(tokens: tokens, visionSoftTokens: visionSoft,
                                                    audioSoftTokens: audioSoft)
            let logits = decoder.logits(fromEmbeddings: embeddings, tokens: ids)
            let next = logits[0, tokens.count - 1].argMax(axis: -1).item(Int.self)
            if stopTokens.contains(next) { break }
            produced.append(next)
            tokens.append(next)
        }
        return produced
    }
}
