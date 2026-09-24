//
//  NFKMLXSigLIP2Probe.swift
//  InferKitMLX
//
//  A consumer's own image classifier over a frozen SigLIP 2 embedding.
//
//  The customization is the shared linear probe, `NFKMLXEmbeddingProbe`, at SigLIP 2's width. The towers
//  stay frozen: images are encoded once through the attention-pooled vision tower, and the probe trains
//  over the cached vectors. A full fine-tune of the towers is the sigmoid contrastive objective over
//  large batches of image-text pairs, which is not a device workload.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

extension NFKMLXSigLIP2 {

    /// Builds a SigLIP 2 model object at one of the released sizes from optional local weights.
    ///
    /// - Since: InferKit 0.5.0
    @objc(modelWithVariant:weightsURL:error:)
    public static func model(variant: NFKMLXSigLIP2Variant, weightsURL: URL?) throws -> NFKMLXSigLIP2 {
        try model(configuration: specs(for: variant).configuration, weightsURL: weightsURL)
    }

    /// The width of the image embedding, which is the width a probe over this model reads.
    ///
    /// - Since: InferKit 0.5.0
    @objc public var embeddingDimensions: Int { holder.net.configuration.vision.hiddenSize }

    /// Encodes images into cached L2-normalized embeddings `[N, embeddingDimensions]`.
    ///
    /// Run this once. The towers are frozen for a probe, so an embedding never changes, and re-encoding
    /// per step would be the whole cost of the run. Encoding is multi-second over many images; call it
    /// off the render thread.
    ///
    /// - Since: InferKit 0.5.0
    public func imageEmbeddings(for images: [CGImage]) throws -> MLXArray {
        guard !images.isEmpty else {
            throw NFKMLXError.trainingDataMismatch("a probe needs at least one image to encode")
        }
        let imageSize = holder.net.configuration.vision.imageSize
        let encoded = try images.map { image -> MLXArray in
            let embedding = holder.net.imageEmbedding(try Self.pixelValues(from: image, imageSize: imageSize))
            eval(embedding)
            return embedding
        }
        return concatenated(encoded, axis: 0)
    }

    /// Wraps a trained probe as an InferKit backend: an image under `NFKInputImage` becomes ranked
    /// `NFKClassification`s under `NFKOutputClassifications`.
    ///
    /// - Since: InferKit 0.5.0
    public func probeBackend(probe: NFKMLXEmbeddingProbe, labels: [String]? = nil) throws -> any NFKInferenceBackend {
        guard probe.embedDimensions == embeddingDimensions else {
            throw NFKMLXError.trainingDataMismatch(
                "the probe reads \(probe.embedDimensions)-wide embeddings and this SigLIP 2 produces \(embeddingDimensions)")
        }
        let holder = self.holder
        return NFKMLXEmbeddingProbeBackend(probe: probe, identifier: "siglip2-probe", labels: labels) { value in
            guard CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
                throw NFKMLXError.unsupportedInput
            }
            let pixels = try NFKMLXSigLIP2.pixelValues(from: value as! CGImage,
                                                       imageSize: holder.net.configuration.vision.imageSize)
            return holder.net.imageEmbedding(pixels)
        }
    }

    /// Wraps a saved probe as an InferKit backend, reading the probe from the file
    /// `NFKMLXWeights.save` wrote.
    ///
    /// This is the Objective-C reach into a probe: a consumer trains through
    /// `NFKMLXEmbeddingProbe.train` in Swift, saves, and an app loads the result here.
    ///
    /// - Since: InferKit 0.5.0
    @objc(probeBackendWithProbeURL:labels:error:)
    public func probeBackend(probeURL: URL, labels: [String]?) throws -> any NFKInferenceBackend {
        try probeBackend(probe: NFKMLXEmbeddingProbe(weightsURL: probeURL), labels: labels)
    }
}
