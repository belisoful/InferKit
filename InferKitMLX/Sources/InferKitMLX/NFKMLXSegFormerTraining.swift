//
//  NFKMLXSegFormerTraining.swift
//  InferKitMLX
//
//  Fine-tuning SegFormer, the head-only customization recipe.
//
//  Segmentation is where retargeting a model matters most: a consumer rarely wants ADE20K's 150 classes
//  and almost always wants their own few. That is a decode-head problem, not an encoder problem — the
//  MiT encoder already produces good features, and freezing it is what makes the run fit on a device.
//  The all-MLP head is small enough to train from a handful of annotated frames.
//
//  The loss follows the reference: logits are produced at stage-1 resolution and bilinearly upsampled to
//  the label resolution before cross-entropy, rather than downsampling the labels.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a SegFormer fine-tune updates.
public enum NFKMLXSegFormerTrainable: Sendable {

    /// The all-MLP decode head only, with the MiT encoder frozen.
    ///
    /// This is the default and the one that fits on a device: a frozen parameter produces no gradient
    /// and carries no optimizer state, so the memory a run needs falls to the head's share. It is also
    /// the safer fit, because a small annotated set cannot damage the pretrained features.
    case decodeHead

    /// Every parameter, including the encoder. Needs more data and more memory.
    case everything
}

/// The supervised objective a SegFormer fine-tune minimizes.
public struct NFKMLXSegFormerObjective: Sendable {

    /// Smooths the target distribution, which steadies a run over few annotated examples.
    public var labelSmoothing: Float

    public init(labelSmoothing: Float = 0) {
        self.labelSmoothing = labelSmoothing
    }

    /// Scores `net` on one annotated example: an image `[H, W, 3]` in `0...1` and class indices
    /// `[H, W]`.
    public func callAsFunction(_ net: NFKMLXSegFormerNet, _ image: MLXArray, _ labels: MLXArray) -> MLXArray {
        loss(logits: net.logits(net.normalized(image)), labels: labels)
    }

    /// Scores class logits directly, without a network.
    ///
    /// The forward pass and the loss are separable so the loss can be compared against the reference
    /// on identical logits: `NFKMLXReferenceParityTests` scores the tensors the reference scored, and
    /// a mismatch can then only come from the arithmetic here.
    ///
    /// - Parameters:
    ///   - logits: class scores `[1, h, w, classCount]` at the decode head's own resolution.
    ///   - labels: class indices `[H, W]` at full resolution.
    ///
    public func loss(logits: MLXArray, labels: MLXArray) -> MLXArray {
        let (height, width) = (labels.shape[0], labels.shape[1])
        // The reference upsamples the logits to the label resolution rather than downsampling the
        // labels, which would throw away the thin structures segmentation is judged on.
        let full = NFKMLXResample.resizeBilinear(logits, height: height, width: width)
        let classCount = full.shape[3]
        return crossEntropy(logits: full.reshaped([height * width, classCount]),
                            targets: labels.reshaped([height * width]),
                            labelSmoothing: labelSmoothing, reduction: .mean)
    }
}

extension NFKMLXSegFormer {

    /// Builds the segmentation network itself, ready to fine-tune, rather than a backend wrapping it.
    ///
    /// - Parameters:
    ///   - weightsURL: a converted or fine-tuned checkpoint. Nil leaves the network at its random
    ///     initialization.
    ///   - classCount: the consumer's own class set. When it differs from the checkpoint's, the
    ///     classifier is left randomly initialized and everything else loads, which is exactly what
    ///     retargeting a segmentation model means.
    ///
    /// Dropping the classifier is not optional in that case: MLX's `update(parameters:)` adopts a
    /// checkpoint's shapes wholesale rather than validating them, so a 150-class classifier would
    /// silently replace the consumer's smaller one and the model would emit the old class set.
    public static func network(weightsURL: URL?, classCount: Int = 150) throws -> NFKMLXSegFormerNet {
        var configuration = NFKMLXSegFormerConfiguration()
        configuration.classCount = classCount
        let net = NFKMLXSegFormerNet(configuration)
        guard let weightsURL else {
            return net
        }

        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
        let mapped = checkpoint.arrays.map { key, value in
            (remapReferenceKey(key),
             checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        let classifierOutputs = mapped.first { $0.0 == "classifier.weight" }?.1.shape.first
        guard let classifierOutputs, classifierOutputs != classCount else {
            try NFKMLXWeights.apply(mapped, to: net)
            return net
        }
        try NFKMLXWeights.apply(mapped.filter { !$0.0.hasPrefix("classifier.") }, to: net, strict: false)
        return net
    }

    /// Fine-tunes a segmentation network on a consumer's own annotated frames, returning the loss from
    /// each step.
    ///
    /// The whole customization path is three calls: ``network(weightsURL:classCount:)`` to build, this
    /// to train, and `NFKMLXWeights.save` to write a checkpoint that ``backend(weightsURL:)`` loads
    /// like any other.
    ///
    /// - Parameters:
    ///   - net: the network to train, from ``network(weightsURL:classCount:)``.
    ///   - examples: supplies one annotated example per step: an image `[H, W, 3]` in `0...1` and
    ///     class indices `[H, W]`, both from `NFKMLXTrainingData`.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: the supervised loss.
    ///   - optimizer: the update rule. Nil uses `AdamW`, the reference's choice for the head.
    ///   - steps: how many examples to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXSegFormerNet,
        examples: (Int) -> (image: MLXArray, labels: MLXArray),
        trainable: NFKMLXSegFormerTrainable = .decodeHead,
        objective: NFKMLXSegFormerObjective = NFKMLXSegFormerObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        apply(trainable, to: net)
        return try NFKMLXTrainer.train(net, optimizer: optimizer ?? AdamW(learningRate: 6e-5),
                                       steps: steps,
                                       batch: { let example = examples($0); return (example.image, example.labels) },
                                       loss: objective.callAsFunction,
                                       clipGradientNorm: clipGradientNorm, checkpoint: checkpoint,
                                       observer: observer)
    }

    /// Freezes the encoder for a head-only run, or unfreezes everything.
    private static func apply(_ trainable: NFKMLXSegFormerTrainable, to net: NFKMLXSegFormerNet) {
        switch trainable {
        case .everything:
            net.unfreeze()
        case .decodeHead:
            net.unfreeze()
            for stage in [net.stage1, net.stage2, net.stage3, net.stage4] {
                stage.freeze()
            }
        }
    }
}
