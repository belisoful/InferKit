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
//  Contrast with a contrastive fine-tune of CLIP itself, which needs large batches for negatives and is
//  not a device workload. The probe is a separate small model, so what it saves is a companion file
//  rather than modified CLIP weights.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/// A linear classifier over a CLIP embedding.
public final class NFKMLXCLIPProbe: Module {

    @ModuleInfo(key: "classifier") var classifier: Linear

    /// How many categories it predicts.
    public let classCount: Int

    /// - Parameters:
    ///   - embedDimensions: the CLIP embedding width, 512 for ViT-B/32.
    ///   - classCount: the consumer's own categories.
    public init(embedDimensions: Int = 512, classCount: Int) {
        self.classCount = classCount
        _classifier.wrappedValue = Linear(embedDimensions, classCount)
    }

    /// Scores cached embeddings `[N, embedDimensions]`, returning logits `[N, classCount]`.
    public func callAsFunction(_ embeddings: MLXArray) -> MLXArray {
        classifier(embeddings)
    }
}

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
    /// - Parameters:
    ///   - probe: the classifier to train.
    ///   - embeddings: cached vectors `[N, embedDimensions]` from ``embeddings(for:using:colorSpace:)``.
    ///   - labels: one class index per embedding, `[N]`.
    ///   - sampler: draws which examples each step sees. Nil trains on the whole set every step, which
    ///     is what a few dozen examples want.
    ///   - optimizer: the update rule. Nil uses `AdamW`.
    ///   - steps: how many updates to run.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - checkpoint: writes the probe periodically.
    ///   - observer: receives each step and can end the run early.
    ///
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
        guard embeddings.shape[0] == labels.shape[0] else {
            throw NFKMLXError.trainingDataMismatch(
                "\(embeddings.shape[0]) embeddings and \(labels.shape[0]) labels were supplied; "
                + "a probe needs one class index per image")
        }
        return try NFKMLXTrainer.train(
            probe, optimizer: optimizer ?? AdamW(learningRate: 1e-3), steps: steps,
            batch: { step in
                guard let sampler else {
                    return (embeddings, labels)
                }
                let indices = MLXArray(sampler.indices(forStep: step).map { Int32($0) })
                return (embeddings[indices], labels[indices])
            },
            loss: { probe, batch, targets in
                crossEntropy(logits: probe(batch), targets: targets, reduction: .mean)
            },
            clipGradientNorm: clipGradientNorm, checkpoint: checkpoint, observer: observer)
    }

    /// Wraps a trained probe as an InferKit backend: an image under `NFKInputImage` becomes ranked
    /// `NFKClassification`s under `NFKOutputClassifications`.
    public static func probeBackend(net: NFKMLXCLIPNet, probe: NFKMLXCLIPProbe,
                                    labels: [String]? = nil) -> any NFKInferenceBackend {
        NFKMLXCLIPProbeBackend(net: net, probe: probe, identifier: "clip-probe", labels: labels)
    }
}

/// Holds the networks and labels for capture in the backend's `@Sendable` closure.
private final class NFKCLIPProbeHolder: @unchecked Sendable {
    let net: NFKMLXCLIPNet
    let probe: NFKMLXCLIPProbe
    let labels: [String]?
    init(net: NFKMLXCLIPNet, probe: NFKMLXCLIPProbe, labels: [String]?) {
        self.net = net
        self.probe = probe
        self.labels = labels
    }
}

/// A consumer's own image classifier, built on a frozen CLIP embedding. Reads `NFKInputImage`; returns
/// ranked classes under `NFKOutputClassifications`.
@objc(NFKMLXCLIPProbeBackend)
public final class NFKMLXCLIPProbeBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKCLIPProbeHolder
    private let identifier: String

    init(net: NFKMLXCLIPNet, probe: NFKMLXCLIPProbe, identifier: String, labels: [String]?) {
        holder = NFKCLIPProbeHolder(net: net, probe: probe, labels: labels)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedInput
        }
        let tensor = try NFKMLXImageBridge.tensor(from: value, channels: 3,
                                                  colorSpace: CGColorSpaceCreateDeviceRGB())
        let embedding = holder.net.encodeImage(tensor)
        let logits = holder.probe(embedding.reshaped([1, embedding.shape[0]]))
        return NFKInferenceResult(outputs: [NFKOutputClassifications: Self.ranked(logits, labels: holder.labels)])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached {
            do {
                job.finish(with: try self.runInference(for: request))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    /// Softmax over the logits, most confident first, so the confidences read as probabilities over
    /// the consumer's categories.
    private static func ranked(_ logits: MLXArray, labels: [String]?) -> [NFKClassification] {
        let probabilities = softmax(logits, axis: -1).reshaped([logits.shape[1]]).asArray(Float.self)
        return probabilities.enumerated()
            .sorted { $0.element > $1.element }
            .map { index, score in
                NFKClassification(label: labels.flatMap { index < $0.count ? $0[index] : nil },
                                  classIndex: index, confidence: Double(score))
            }
    }
}
