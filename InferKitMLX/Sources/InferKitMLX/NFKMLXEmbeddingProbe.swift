//
//  NFKMLXEmbeddingProbe.swift
//  InferKitMLX
//
//  A consumer's own classifier over any frozen embedding: a linear probe.
//
//  An embedding model already separates most concepts; what a consumer lacks is the mapping from that
//  space to their own categories. The probe learns that mapping and nothing else. The embedder stays
//  frozen, so each example is encoded once and the training loop runs over cached vectors: a step is
//  one matrix multiply, and a run over a few dozen examples finishes in seconds. The probe is a separate
//  small model, so what it saves is a companion file, and the embedder's weights stay as released.
//
//  The embedder is whatever produces the vectors. `NFKMLXCLIP` and `NFKMLXSigLIP2` encode images for
//  it and wrap a trained probe as a backend.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/*!
 @abstract A linear classifier over a frozen embedding.
 @discussion Scores cached embeddings `[N, embedDimensions]` into logits `[N, classCount]`. Train it with
 ``train(_:embeddings:labels:sampler:optimizer:steps:clipGradientNorm:checkpoint:observer:)``, save it
 with `NFKMLXWeights.save`, and reload it with ``init(weightsURL:)``. Introduced in InferKit 0.5.0.
 */
public final class NFKMLXEmbeddingProbe: Module {

    @ModuleInfo(key: "classifier") var classifier: Linear

    /// The embedding width it reads.
    public let embedDimensions: Int

    /// How many categories it predicts.
    public let classCount: Int

    /// - Parameters:
    ///   - embedDimensions: the embedder's output width, 512 for CLIP ViT-B/32 and 768 for SigLIP 2 base.
    ///   - classCount: the consumer's own categories.
    public init(embedDimensions: Int, classCount: Int) {
        self.embedDimensions = embedDimensions
        self.classCount = classCount
        _classifier.wrappedValue = Linear(embedDimensions, classCount)
    }

    /// Reloads a saved probe, reading its width and class count from the file.
    public convenience init(weightsURL: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
        guard let weight = checkpoint.arrays["classifier.weight"], weight.ndim == 2 else {
            throw NFKMLXError.trainingDataMismatch("\(weightsURL.lastPathComponent) holds no probe classifier")
        }
        self.init(embedDimensions: weight.shape[1], classCount: weight.shape[0])
        try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value) }, to: self, verifyShapes: true)
    }

    /// Scores cached embeddings `[N, embedDimensions]`, returning logits `[N, classCount]`.
    public func callAsFunction(_ embeddings: MLXArray) -> MLXArray {
        classifier(embeddings)
    }

    /// Trains a probe on cached embeddings, returning the loss from each step.
    ///
    /// - Parameters:
    ///   - probe: the classifier to train.
    ///   - embeddings: cached vectors `[N, embedDimensions]` from the embedder.
    ///   - labels: one class index per embedding, `[N]`.
    ///   - sampler: draws which examples each step sees. Nil trains on the whole set every step, which
    ///     is what a few dozen examples want.
    ///   - optimizer: the update rule. Nil uses AdamW at 1e-3 with weight decay 0.01, bias-corrected as
    ///     `torch.optim.AdamW` is. The references' own linear probes are closed-form or L-BFGS logistic
    ///     regressions, so the optimizer is this package's choice.
    ///   - steps: how many updates to run.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - checkpoint: writes the probe periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    @discardableResult
    public static func train(
        _ probe: NFKMLXEmbeddingProbe,
        embeddings: MLXArray,
        labels: MLXArray,
        sampler: NFKMLXBatchSampler? = nil,
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        guard embeddings.ndim == 2, embeddings.shape[1] == probe.embedDimensions else {
            throw NFKMLXError.trainingDataMismatch(
                "embeddings of shape \(embeddings.shape) were supplied to a probe \(probe.embedDimensions) wide")
        }
        guard embeddings.shape[0] == labels.shape[0] else {
            throw NFKMLXError.trainingDataMismatch(
                "\(embeddings.shape[0]) embeddings and \(labels.shape[0]) labels were supplied; "
                + "a probe needs one class index per example")
        }
        return try NFKMLXFineTune.run(
            probe,
            freezing: {},
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: 1e-3, weightDecay: 0.01) },
            referenceSchedule: { .constant },
            steps: steps,
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
}

/// Holds the probe, labels, and embedder for capture in the backend's `@Sendable` closure.
private final class NFKEmbeddingProbeHolder: @unchecked Sendable {
    let probe: NFKMLXEmbeddingProbe
    let labels: [String]?
    let embed: (Any) throws -> MLXArray
    init(probe: NFKMLXEmbeddingProbe, labels: [String]?, embed: @escaping (Any) throws -> MLXArray) {
        self.probe = probe
        self.labels = labels
        self.embed = embed
    }
}

/*!
 @abstract A consumer's own image classifier over a frozen embedding.
 @discussion Reads `NFKInputImage` and returns every class under `NFKOutputClassifications`, most
 confident first, with softmax confidences. `NFKMLXCLIP` and `NFKMLXSigLIP2` build it around their
 image encoders. Introduced in InferKit 0.5.0.
 */
@objc(NFKMLXEmbeddingProbeBackend)
public final class NFKMLXEmbeddingProbeBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKEmbeddingProbeHolder
    private let identifier: String

    /// - Parameter embed: turns an `NFKInputImage` value into one embedding of `probe.embedDimensions`
    ///   elements.
    init(probe: NFKMLXEmbeddingProbe, identifier: String, labels: [String]?,
         embed: @escaping (Any) throws -> MLXArray) {
        holder = NFKEmbeddingProbeHolder(probe: probe, labels: labels, embed: embed)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads.
    @objc public var supportedParameterKeys: Set<String> { [] }

    /// The request inputs the backend reads.
    @objc public var supportedInputKeys: Set<String> { [NFKInputImage] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedInput
        }
        let embedding = try holder.embed(value)
        let logits = holder.probe(embedding.reshaped([1, holder.probe.embedDimensions]))
        return NFKInferenceResult(outputs: [NFKOutputClassifications: Self.ranked(logits, labels: holder.labels)])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
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
