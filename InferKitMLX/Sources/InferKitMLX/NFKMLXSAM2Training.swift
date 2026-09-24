//
//  NFKMLXSAM2Training.swift
//  InferKitMLX
//
//  Fine-tuning SAM 2, the head-only customization recipe.
//
//  SAM 2 segments what a click points at, and a consumer who wants it to point at THEIR subject — a
//  product on a shelf, a cell in a slide, a part on a line — is retargeting the head, not the
//  encoder. The Hiera trunk already produces good features and freezing it is what makes the run fit
//  on a device: a frozen parameter produces no gradient and carries no optimizer state, so a
//  head-only run over the tiny release trains about 5M parameters instead of 39M.
//
//  The objective is the reference's `MultiStepMultiMasksAndIous`, at the weights its own fine-tuning
//  configuration sets (mask 20, dice 1, IoU 1, class 1, IoU by L1, every IoU supervised). Three of
//  its details are the kind a paper does not state and a reimplementation gets wrong: the focal and
//  dice terms are backpropagated only through the multimask slot with the LOWEST combined loss, the
//  IoU head is supervised against the actual IoU the prediction achieves rather than a label, and
//  every mask term is multiplied by whether the target holds an object at all, so an empty frame
//  trains the object score and nothing else.
//
//  A video fine-tune, which backpropagates through the memory bank across a clip, is not this. What
//  ships here is the conditioning frame's path, which is what a device holds.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a SAM 2 fine-tune updates.
public enum NFKMLXSAM2Trainable: Sendable {

    /// The prompt encoder and mask decoder, with the image encoder and the whole memory path frozen.
    ///
    /// This is the default and the one that fits on a device. It is also the safer fit: a small set
    /// of annotated frames cannot damage pretrained features it never reaches.
    case maskDecoder

    /// Every parameter. Needs more data and more memory, and on a clip it needs the video objective
    /// this recipe does not ship.
    case everything
}

/// The supervised objective a SAM 2 fine-tune minimizes: the reference's
/// `MultiStepMultiMasksAndIous`, over one prediction step.
public struct NFKMLXSAM2Objective: Sendable {

    /// The reference's own fine-tuning weights.
    public var maskWeight: Float
    public var diceWeight: Float
    public var intersectionOverUnionWeight: Float
    public var objectWeight: Float
    /// The focal parameters for the mask term.
    public var focalAlpha: Float
    public var focalGamma: Float
    /// The focal parameters for the object-score term. A negative alpha turns the weighting off,
    /// and a gamma of zero makes the term a plain cross-entropy, which is what the reference sets.
    public var objectFocalAlpha: Float
    public var objectFocalGamma: Float
    /// Whether the IoU head is supervised on every multimask slot or only the best one.
    public var supervisesEveryIntersectionOverUnion: Bool
    /// Whether the IoU head takes an L1 loss rather than the squared one.
    public var intersectionOverUnionUsesL1: Bool

    public init(maskWeight: Float = 20, diceWeight: Float = 1,
                intersectionOverUnionWeight: Float = 1, objectWeight: Float = 1,
                focalAlpha: Float = 0.25, focalGamma: Float = 2,
                objectFocalAlpha: Float = -1, objectFocalGamma: Float = 0,
                supervisesEveryIntersectionOverUnion: Bool = true,
                intersectionOverUnionUsesL1: Bool = true) {
        self.maskWeight = maskWeight
        self.diceWeight = diceWeight
        self.intersectionOverUnionWeight = intersectionOverUnionWeight
        self.objectWeight = objectWeight
        self.focalAlpha = focalAlpha
        self.focalGamma = focalGamma
        self.objectFocalAlpha = objectFocalAlpha
        self.objectFocalGamma = objectFocalGamma
        self.supervisesEveryIntersectionOverUnion = supervisesEveryIntersectionOverUnion
        self.intersectionOverUnionUsesL1 = intersectionOverUnionUsesL1
    }

    /// Scores `net` on one annotated frame: an image `[1, size, size, 3]`, normalized, and a binary
    /// target mask `[1, size, size]`.
    public func callAsFunction(_ net: NFKMLXSAM2TrackerNet, _ image: MLXArray,
                               points: [(x: Float, y: Float, label: Int)],
                               _ target: MLXArray) -> MLXArray {
        let predicted = net.segment(image: image, points: points)
        return loss(masks: predicted.masks, target: target,
                    intersectionOverUnion: predicted.intersectionOverUnion,
                    objectScore: predicted.objectScore)
    }

    /// Scores predictions directly, without a network.
    ///
    /// The forward pass and the loss are separable so the loss can be compared against the reference
    /// on identical tensors: `NFKMLXReferenceParityTests` scores the tensors the reference scored,
    /// and a mismatch can then only come from the arithmetic here.
    ///
    /// - Parameters:
    ///   - masks: mask logits `[1, slots, H, W]`, every multimask slot.
    ///   - target: the binary target `[1, H, W]`.
    ///   - intersectionOverUnion: the decoder's quality estimate `[1, slots]`.
    ///   - objectScore: the object-presence logit `[1, 1]`.
    ///   - objectCount: the objects the batch holds, which every term is divided by.
    public func loss(masks: MLXArray, target: MLXArray, intersectionOverUnion: MLXArray,
                     objectScore: MLXArray, objectCount: Float = 1) -> MLXArray {
        let parts = components(masks: masks, target: target,
                               intersectionOverUnion: intersectionOverUnion,
                               objectScore: objectScore, objectCount: objectCount)
        return parts.mask * maskWeight + parts.dice * diceWeight
            + parts.intersectionOverUnion * intersectionOverUnionWeight + parts.object * objectWeight
    }

    /// The four terms before they are weighted, each a scalar, in the reference's own order.
    ///
    /// The objective reduces to their weighted sum; they are separate so a parity run can say which
    /// one disagrees rather than only that the total does.
    public func components(masks: MLXArray, target: MLXArray, intersectionOverUnion: MLXArray,
                           objectScore: MLXArray, objectCount: Float = 1)
        -> (mask: MLXArray, dice: MLXArray, intersectionOverUnion: MLXArray, object: MLXArray) {
        let targets = broadcast(target.expandedDimensions(axis: 1), to: masks.shape)
        let focal = NFKMLXSAM2Losses.focal(masks, targets, alpha: focalAlpha, gamma: focalGamma,
                                           objectCount: objectCount, perSlot: true)
        let dice = NFKMLXSAM2Losses.dice(masks, targets, objectCount: objectCount, perSlot: true)
        let quality = NFKMLXSAM2Losses.quality(masks, targets, predicted: intersectionOverUnion,
                                               objectCount: objectCount, usesL1: intersectionOverUnionUsesL1)

        // The target holds an object when any of its pixels does. A frame without one trains the
        // object score and nothing else, which is how the reference teaches absence.
        let present = (target.reshaped([target.dim(0), -1]) .> 0).any(axis: -1)
            .asType(masks.dtype).reshaped([-1, 1])
        let object = NFKMLXSAM2Losses.focal(objectScore, present, alpha: objectFocalAlpha,
                                            gamma: objectFocalGamma, objectCount: objectCount,
                                            perSlot: false)

        // Backpropagate the mask terms only through the slot whose combined loss is lowest; the
        // decoder is trained to offer alternatives, not to make all three right.
        var maskTerm = focal, diceTerm = dice, qualityTerm = quality
        if masks.dim(1) > 1 {
            let combined = focal * maskWeight + dice * diceWeight
            // The choice of slot is a selection, not a differentiable function of the losses that
            // made it. MLX asks for a gradient through a gather's indices unless one says otherwise,
            // where the reference's `argmin` simply returns an integer tensor that carries none.
            let best = stopGradient(argMin(combined, axis: -1).reshaped([-1, 1]))
            maskTerm = takeAlong(focal, best, axis: -1)
            diceTerm = takeAlong(dice, best, axis: -1)
            qualityTerm = supervisesEveryIntersectionOverUnion
                ? quality.mean(axis: -1, keepDims: true)
                : takeAlong(quality, best, axis: -1)
        }
        return ((maskTerm * present).sum(), (diceTerm * present).sum(),
                (qualityTerm * present).sum(), object)
    }
}

/// The three terms the reference's SAM 2 objective is built from, each keeping its own shape so the
/// caller can select a slot before reducing.
enum NFKMLXSAM2Losses {

    /// Sigmoid focal loss. `perSlot` keeps one value per multimask slot, averaged over the pixels;
    /// otherwise the pixels average and the batch sums, which is what the object-score term wants.
    static func focal(_ logits: MLXArray, _ targets: MLXArray, alpha: Float, gamma: Float,
                      objectCount: Float, perSlot: Bool) -> MLXArray {
        let probability = sigmoid(logits)
        // Binary cross-entropy from the logits, in the stable form the reference's own
        // `binary_cross_entropy_with_logits` uses.
        let entropy = MLX.maximum(logits, 0) - logits * targets + log1p(exp(-MLX.abs(logits)))
        let matched = probability * targets + (1 - probability) * (1 - targets)
        var loss = entropy * pow(1 - matched, gamma)
        if alpha >= 0 {
            loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
        }
        guard perSlot else {
            return loss.reshaped([loss.dim(0), -1]).mean(axis: -1).sum() / objectCount
        }
        return loss.reshaped([loss.dim(0), loss.dim(1), -1]).mean(axis: -1) / objectCount
    }

    /// Dice loss over the sigmoid of the logits, one value per slot.
    static func dice(_ logits: MLXArray, _ targets: MLXArray, objectCount: Float,
                     perSlot: Bool) -> MLXArray {
        let probability = sigmoid(logits).reshaped([logits.dim(0), logits.dim(1), -1])
        let flat = targets.reshaped([targets.dim(0), targets.dim(1), -1])
        let numerator = 2 * (probability * flat).sum(axis: -1)
        let denominator = probability.sum(axis: -1) + flat.sum(axis: -1)
        let loss = 1 - (numerator + 1) / (denominator + 1)
        return perSlot ? loss / objectCount : loss.sum() / objectCount
    }

    /// The IoU head's loss against the IoU its own mask actually achieves, thresholded at zero.
    static func quality(_ logits: MLXArray, _ targets: MLXArray, predicted: MLXArray,
                        objectCount: Float, usesL1: Bool) -> MLXArray {
        // Both sides are 0 or 1, so the product is their intersection and the larger their union.
        let mask = (logits.reshaped([logits.dim(0), logits.dim(1), -1]) .> 0).asType(.float32)
        let truth = (targets.reshaped([targets.dim(0), targets.dim(1), -1]) .> 0).asType(.float32)
        let intersection = (mask * truth).sum(axis: -1)
        let union = MLX.maximum(mask, truth).sum(axis: -1)
        let actual = intersection / clip(union, min: 1)
        let difference = predicted - actual
        return (usesL1 ? MLX.abs(difference) : difference.square()) / objectCount
    }
}

/// Carries the current step's click from the batch source to the loss, which the trainer's fixed
/// two-tensor step cannot.
private final class NFKMLXSAM2Prompt: @unchecked Sendable {
    var points: [(x: Float, y: Float, label: Int)] = []
}

extension NFKMLXSAM2 {

    /// Builds the tracker itself, ready to fine-tune, rather than a backend wrapping it.
    ///
    /// - Parameters:
    ///   - weightsURL: a released or fine-tuned checkpoint. Nil leaves the network at its random
    ///     initialization.
    ///   - variant: the image encoder's size, which a released checkpoint fixes.
    ///   - release: 2.0 or 2.1, which decides two parameters a checkpoint either carries or does not.
    ///
    /// Nothing is dropped on load the way a retargeted classifier is: SAM 2 predicts class-agnostic
    /// masks, so a consumer's own subject needs no new head, only a trained one.
    public static func network(weightsURL: URL?, variant: NFKMLXSAM2Variant = .tiny,
                               release: NFKMLXSAM2Release = .sam21) throws -> NFKMLXSAM2TrackerNet {
        let net = makeTracker(variant: variant, release: release)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Fine-tunes the tracker on a consumer's own annotated frames, returning the loss from each step.
    ///
    /// The whole customization path is three calls: ``network(weightsURL:variant:release:)`` to
    /// build, this to train, and `NFKMLXWeights.save` to write a checkpoint that
    /// ``backend(variant:release:weightsURL:)`` loads like any other.
    ///
    /// - Parameters:
    ///   - net: the network to train, from ``network(weightsURL:variant:release:)``.
    ///   - examples: supplies one annotated frame per step: a normalized image
    ///     `[1, size, size, 3]`, the clicks that prompt it in pixels of that image, and a binary
    ///     target `[1, size, size]`.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: the supervised loss.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.AdamW` (bias-corrected):
    ///     5e-6, and 3e-6 for the image encoder with the trunk's blocks decayed by 0.9 a layer from the
    ///     top, and weight decay 0.1 on every parameter but biases and `nn.LayerNorm` weights.
    ///   - steps: how many frames to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update: 0.1, the reference's.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's fvcore
    ///     cosine from the full rate to a tenth of it over the run, when the reference optimizer runs.
    ///     With a caller's optimizer, nil holds that optimizer's rate constant.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXSAM2TrackerNet,
        examples: (Int) -> (image: MLXArray, points: [(x: Float, y: Float, label: Int)], target: MLXArray),
        trainable: NFKMLXSAM2Trainable = .maskDecoder,
        objective: NFKMLXSAM2Objective = NFKMLXSAM2Objective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 0.1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let prompt = NFKMLXSAM2Prompt()
        return try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: { referenceOptimizer(for: net) },
            referenceSchedule: { .cosine(steps: steps, endScale: 0.1) },
            steps: steps,
            batch: { step in
                let example = examples(step)
                prompt.points = example.points
                return (example.image, example.target)
            },
            loss: { net, image, target in objective(net, image, points: prompt.points, target) },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// The reference's AdamW: `sam2.1_hiera_b+_MOSE_finetune.yaml` decays every parameter by 0.1 but
    /// the ones named `*bias*` and those of a `torch.nn.LayerNorm`. The mask embedder's two norms and the
    /// decoder's upscaling norm are `LayerNorm2d` there, a different class, so their weights decay.
    /// The image encoder trains at 3e-6, scaled per trunk layer by `layer_decay_param_modifier`.
    private static func referenceOptimizer(for net: NFKMLXSAM2TrackerNet) -> Optimizer {
        let layerNorms = NFKMLXReferenceOptimizers.layerNormPrefixes(
            in: net, excluding: ["mask_embed.norm1", "mask_embed.norm2", "upscale_norm"])
        let exempt = NFKMLXReferenceOptimizers.biasOrLayerNorm(layerNorms)
        let blocks = net.imageEncoder.trunk.blocks.count
        return NFKMLXReferenceOptimizers.adamW(learningRate: 5e-6, over: net) { key in
            let weightDecay: Float = exempt(key) ? 0 : 0.1
            guard key.hasPrefix("image_encoder.") else { return (1, weightDecay) }
            let vision = 3e-6 / 5e-6 * trunkLayerDecay(key, blocks: blocks)
            return (Float(vision), weightDecay)
        }
    }

    /// Hiera's `get_layer_id` under the configuration's decay of 0.9: block `i` of `blocks` is scaled
    /// by `0.9^(blocks − i)`, the patch embedding by `0.9^(blocks + 1)`, and the position embeddings
    /// (an override) and everything outside the trunk by 1.
    static func trunkLayerDecay(_ key: String, blocks: Int) -> Double {
        let trunk = "image_encoder.trunk."
        guard key.hasPrefix(trunk), !key.contains("pos_embed") else { return 1 }
        let name = key.dropFirst(trunk.count)
        if name.hasPrefix("patch_embed") { return pow(0.9, Double(blocks + 1)) }
        if name.hasPrefix("blocks."), let index = Int(name.split(separator: ".")[1]) {
            return pow(0.9, Double(blocks - index))
        }
        return 1
    }

    /// Freezes everything but the prompt encoder and mask decoder, or unfreezes everything.
    ///
    /// The tracker's own parameters — the no-memory embedding, the temporal encodings, the object
    /// pointer projection — belong to the memory path and stay frozen with it.
    private static func apply(_ trainable: NFKMLXSAM2Trainable, to net: NFKMLXSAM2TrackerNet) {
        switch trainable {
        case .everything:
            net.unfreeze()
        case .maskDecoder:
            net.freeze()
            net.promptEncoder.unfreeze()
            net.maskDecoder.unfreeze()
        }
    }
}
