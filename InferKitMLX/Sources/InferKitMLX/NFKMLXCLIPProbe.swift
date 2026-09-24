//
//  NFKMLXCLIPProbe.swift
//  InferKitMLX
//
//  A custom image classifier trained on a device from a handful of examples.
//
//  CLIP's embedding already separates most visual concepts; what a consumer lacks is the mapping from
//  that space to *their* categories. A linear probe learns exactly that mapping and nothing else, which
//  is why it is the cheapest useful customization in this package: both towers stay frozen, so the
//  embeddings can be computed ONCE per image and the training loop then runs over cached vectors. A
//  step costs one 512-wide matrix multiply rather than a transformer forward, and a run finishes in
//  seconds on a few dozen photos.
//
//  A contrastive fine-tune of CLIP itself needs large batches for its negatives and is not a device
//  workload. The probe is the shared `NFKMLXEmbeddingProbe`, so what it saves is a companion file and
//  CLIP's weights stay as released.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/// A linear classifier over a CLIP embedding: the shared ``NFKMLXEmbeddingProbe``.
public typealias NFKMLXCLIPProbe = NFKMLXEmbeddingProbe

/// The backend a trained CLIP probe answers through: the shared ``NFKMLXEmbeddingProbeBackend``.
public typealias NFKMLXCLIPProbeBackend = NFKMLXEmbeddingProbeBackend

extension NFKMLXCLIP {

    /// Builds the CLIP network itself, for encoding a consumer's images before training a probe.
    public static func network(weightsURL: URL?,
                               configuration: NFKMLXCLIPConfiguration = NFKMLXCLIPConfiguration()) throws -> NFKMLXCLIPNet {
        let net = NFKMLXCLIPNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Encodes images into cached embeddings `[N, embedDimensions]`.
    ///
    /// Run this once. The towers are frozen for a probe, so an embedding never changes and re-encoding
    /// per step would be the whole cost of the run.
    ///
    /// Encoding is multi-second over many images; call it off the render thread.
    public static func embeddings(for images: [CGImage], using net: NFKMLXCLIPNet,
                                  colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()) throws -> MLXArray {
        guard !images.isEmpty else {
            throw NFKMLXError.trainingDataMismatch("a probe needs at least one image to encode")
        }
        let encoded = try images.map { image -> MLXArray in
            let tensor = try NFKMLXTrainingData.tensor(image, colorSpace: colorSpace)
            return net.encodeImage(tensor)
        }
        return stacked(encoded, axis: 0)
    }

    /// Trains a probe on cached embeddings, returning the loss from each step.
    ///
    /// This is ``NFKMLXEmbeddingProbe/train(_:embeddings:labels:sampler:optimizer:steps:clipGradientNorm:checkpoint:observer:)``.
    /// CLIP's own linear probe is an L-BFGS logistic regression, so the optimizer is this package's choice.
    @discardableResult
    public static func trainProbe(
        _ probe: NFKMLXCLIPProbe,
        embeddings: MLXArray,
        labels: MLXArray,
        sampler: NFKMLXBatchSampler? = nil,
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXEmbeddingProbe.train(probe, embeddings: embeddings, labels: labels, sampler: sampler,
                                       optimizer: optimizer, steps: steps, clipGradientNorm: clipGradientNorm,
                                       checkpoint: checkpoint, observer: observer)
    }

    /// Wraps a trained probe as an InferKit backend: an image under `NFKInputImage` becomes ranked
    /// `NFKClassification`s under `NFKOutputClassifications`.
    public static func probeBackend(net: NFKMLXCLIPNet, probe: NFKMLXCLIPProbe,
                                    labels: [String]? = nil) -> any NFKInferenceBackend {
        let encoder = NFKCLIPEncoderHolder(net)
        return NFKMLXEmbeddingProbeBackend(probe: probe, identifier: "clip-probe", labels: labels) { value in
            let tensor = try NFKMLXImageBridge.tensor(from: value, channels: 3,
                                                      colorSpace: CGColorSpaceCreateDeviceRGB())
            return encoder.net.encodeImage(tensor)
        }
    }
}

/// Holds the CLIP network for capture in the probe backend's embedder.
private final class NFKCLIPEncoderHolder: @unchecked Sendable {
    let net: NFKMLXCLIPNet
    init(_ net: NFKMLXCLIPNet) { self.net = net }
}
