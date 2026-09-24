//
//  NFKMLXSAM3Training.swift
//  InferKitMLX
//
//  Fine-tuning SAM 3's detector, the head-only customization recipe.
//
//  SAM 3 already names what a consumer asks for in words, and what a consumer usually wants changed is
//  what it counts as an instance in THEIR images: a defect on a casting, a species in a tank, a part
//  on a board. That is a detector problem. The ViT and the text tower stay frozen, which is what makes
//  the run fit: the detector is about 25M parameters against the release's 850M, and because the two
//  encoders are separate modules their output is computed ONCE per image and reused, so a run costs
//  one forward through 850M frozen parameters per image and nothing after.
//
//  The objective is the reference's own, at the settings its `odinw_text_only_train.yaml` sets for
//  exactly this case: a `BinaryHungarianMatcherV2` (class 2, box 5, GIoU 2, focal alpha 0.25,
//  gamma 2), then `Boxes` (L1 5, GIoU 2) and `IABCEMdetr` (classification 20, presence 20, positive
//  weight 5). Its segmentation term is OFF there (`enable_segmentation: False`), so this ships what
//  the reference trains and no more.
//
//  Two details of `IABCEMdetr` are the kind only its code states. A matched query is not regressed
//  toward 1: its target is `p^alpha · IoU^(1-alpha)` clamped at 0.01, so a query is asked to be as
//  confident as its box is good. And an image whose prompt names nothing present contributes NO
//  classification loss at all, only the presence term, which is what teaches the model to say no.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a SAM 3 detector fine-tune updates.
public enum NFKMLXSAM3Trainable: Sendable {

    /// The whole detector: the DETR encoder, the decoder, the scoring head, and the mask decoder.
    case detector

    /// The decoder, the scoring head, and the mask decoder, with the DETR encoder frozen too. The
    /// cheapest level, for a consumer whose images look like the pretraining set and whose quarrel is
    /// only with what counts as an instance.
    case decoder
}

/// The supervised objective a SAM 3 detector fine-tune minimizes: the reference's `Boxes` and
/// `IABCEMdetr` behind its `BinaryHungarianMatcherV2`.
public struct NFKMLXSAM3Objective: Sendable {

    /// The matching costs.
    public var matchClassCost: Float
    public var matchBoxCost: Float
    public var matchGeneralizedIoUCost: Float
    public var matchAlpha: Float
    public var matchGamma: Float

    /// The loss weights.
    public var boxWeight: Float
    public var generalizedIoUWeight: Float
    public var classificationWeight: Float
    public var presenceWeight: Float
    /// Scales a matched query's own term, where an unmatched one is scaled by its probability.
    public var positiveWeight: Float
    /// The exponent that blends a query's confidence with its box quality into the target it is
    /// regressed toward, and the exponent that discounts an easy negative.
    public var alpha: Float
    public var gamma: Float
    public var presenceAlpha: Float
    public var presenceGamma: Float

    public init(matchClassCost: Float = 2, matchBoxCost: Float = 5,
                matchGeneralizedIoUCost: Float = 2, matchAlpha: Float = 0.25, matchGamma: Float = 2,
                boxWeight: Float = 5, generalizedIoUWeight: Float = 2,
                classificationWeight: Float = 20, presenceWeight: Float = 20,
                positiveWeight: Float = 5, alpha: Float = 0.25, gamma: Float = 2,
                presenceAlpha: Float = 0.5, presenceGamma: Float = 0) {
        self.matchClassCost = matchClassCost
        self.matchBoxCost = matchBoxCost
        self.matchGeneralizedIoUCost = matchGeneralizedIoUCost
        self.matchAlpha = matchAlpha
        self.matchGamma = matchGamma
        self.boxWeight = boxWeight
        self.generalizedIoUWeight = generalizedIoUWeight
        self.classificationWeight = classificationWeight
        self.presenceWeight = presenceWeight
        self.positiveWeight = positiveWeight
        self.alpha = alpha
        self.gamma = gamma
        self.presenceAlpha = presenceAlpha
        self.presenceGamma = presenceGamma
    }

    /// The query each target box is assigned to, by the reference's matching cost.
    ///
    /// Matching is a selection, not part of the gradient: the reference computes it under
    /// `no_grad`, and so does this — the cost matrix is evaluated, the assignment runs on the host,
    /// and only the chosen indices re-enter the differentiable arithmetic.
    ///
    /// - Parameters:
    ///   - logits: one logit per query `[1, Q]`.
    ///   - boxes: predicted boxes `[1, Q, 4]` as `(cx, cy, w, h)` in `0...1`.
    ///   - targets: ground-truth boxes `[T, 4]` in the same convention.
    /// - Returns: for each target, the query it matched, in target order.
    public func match(logits: MLXArray, boxes: MLXArray, targets: MLXArray) -> [Int] {
        let queries = logits.dim(1), count = targets.dim(0)
        guard count > 0 else { return [] }
        let predicted = boxes.reshaped([queries, 4])
        let score = logits.reshaped([queries])
        let probability = sigmoid(score)

        // The L1 cost between every query's box and every target's.
        let boxCost = MLX.abs(predicted.expandedDimensions(axis: 1)
                              - targets.expandedDimensions(axis: 0)).sum(axis: -1)
        let generalized = NFKSAM3BoxMetrics.generalizedIoU(NFKSAM3Boxes.centerToCorners(predicted),
                                                           NFKSAM3Boxes.centerToCorners(targets))
        // The focal classification cost, computed through log-sigmoid for the stability the
        // reference's non-`stable` branch relies on.
        let positive = -matchAlpha * pow(1 - probability, matchGamma) * logSigmoid(score)
        let negative = (1 - matchAlpha) * pow(probability, matchGamma) * logSigmoid(-score)
        let classCost = (positive + negative).reshaped([queries, 1])

        let cost = boxCost * matchBoxCost + classCost * matchClassCost
            - generalized * matchGeneralizedIoUCost
        eval(cost)
        let flat = cost.reshaped([-1]).asArray(Float.self)
        return NFKMLXHungarian.assign(cost: flat, rows: queries, columns: count)
    }

    /// The four terms before they are weighted, each a scalar, in the reference's own order.
    ///
    /// - Parameters:
    ///   - logits: one logit per query `[1, Q]`.
    ///   - boxes: predicted boxes `[1, Q, 4]` as `(cx, cy, w, h)`.
    ///   - presence: the presence logit `[1, 1]`.
    ///   - targets: ground-truth boxes `[T, 4]`. An empty set is an image the prompt names nothing in.
    ///   - matched: the query each target was assigned, from ``match(logits:boxes:targets:)``.
    public func components(logits: MLXArray, boxes: MLXArray, presence: MLXArray,
                           targets: MLXArray, matched: [Int])
        -> (classification: MLXArray, presence: MLXArray, box: MLXArray, generalizedIoU: MLXArray) {
        let queries = logits.dim(1)
        let score = logits.reshaped([1, queries])
        let probability = sigmoid(score)
        let present: Float = matched.isEmpty ? 0 : 1

        // The presence term is the only one an empty image contributes to.
        let target = MLXArray([present]).reshaped([1, 1])
        let presenceTerm = NFKSAM3Focal.loss(presence.reshaped([1, 1]), target,
                                             alpha: presenceAlpha, gamma: presenceGamma) / 1

        guard !matched.isEmpty else {
            let zero = MLXArray(Float(0))
            return (zero, presenceTerm, zero, zero)
        }

        let indices = MLXArray(matched.map { Int32($0) })
        var assignment = [Float](repeating: 0, count: queries)
        for query in matched { assignment[query] = 1 }
        let assigned = MLXArray(assignment).reshaped([1, queries])

        let predicted = boxes.reshaped([queries, 4])
        let chosen = predicted[indices]                                    // [T, 4]
        let count = Float(targets.dim(0))

        // The box terms, over the matched pairs only.
        let boxTerm = MLX.abs(chosen - targets).sum() / count
        let overlap = NFKSAM3BoxMetrics.diagonalGeneralizedIoU(
            NFKSAM3Boxes.centerToCorners(chosen), NFKSAM3Boxes.centerToCorners(targets))
        let generalizedTerm = (1 - overlap).sum() / count

        // A matched query is regressed toward its own confidence blended with its box's quality,
        // not toward 1. The blend is a constant as far as the gradient is concerned.
        let quality = NFKSAM3BoxMetrics.diagonalIoU(NFKSAM3Boxes.centerToCorners(chosen),
                                                    NFKSAM3Boxes.centerToCorners(targets))
        let confidence = sigmoid(score.reshaped([queries])[indices])
        let blended = stopGradient(clip(pow(confidence, alpha) * pow(quality, 1 - alpha), min: 0.01))
        var softTarget = [Float](repeating: 0, count: queries)
        let blendedValues = blended.asArray(Float.self)
        for (slot, query) in matched.enumerated() { softTarget[query] = blendedValues[slot] }
        let soft = MLXArray(softTarget).reshaped([1, queries])

        let positives = NFKSAM3Focal.crossEntropy(score, soft) * assigned * positiveWeight
        let negatives = NFKSAM3Focal.crossEntropy(score, assigned) * (1 - assigned)
            * pow(probability, gamma)
        return ((positives + negatives).mean(), presenceTerm, boxTerm, generalizedTerm)
    }

    /// The weighted total the recipe minimizes.
    public func loss(logits: MLXArray, boxes: MLXArray, presence: MLXArray, targets: MLXArray,
                     matched: [Int]) -> MLXArray {
        let parts = components(logits: logits, boxes: boxes, presence: presence, targets: targets,
                               matched: matched)
        return parts.classification * classificationWeight + parts.presence * presenceWeight
            + parts.box * boxWeight + parts.generalizedIoU * generalizedIoUWeight
    }

    /// Scores a detector on one annotated image, matching first.
    public func callAsFunction(_ detector: NFKMLXSAM3DetectorNet, levels: [MLXArray],
                               positions: [MLXArray], prompt: MLXArray, promptValid: MLXArray?,
                               targets: MLXArray) -> MLXArray {
        let detected = detector(levels: levels, positions: positions, prompt: prompt,
                                promptValid: promptValid)
        // The matcher reads the boxes in the centre convention the objective regresses.
        let centres = NFKSAM3BoxMetrics.cornersToCenter(detected.boxes)
        let matched = match(logits: detected.logits, boxes: centres, targets: targets)
        return loss(logits: detected.logits, boxes: centres, presence: detected.presence,
                    targets: targets, matched: matched)
    }
}

/// Binary cross-entropy and the focal loss built on it, in the stable form the reference's
/// `binary_cross_entropy_with_logits` uses.
enum NFKSAM3Focal {
    static func crossEntropy(_ logits: MLXArray, _ targets: MLXArray) -> MLXArray {
        MLX.maximum(logits, 0) - logits * targets + log1p(exp(-MLX.abs(logits)))
    }

    /// The reference's `sigmoid_focal_loss` at its reducing default: the pixels average and the
    /// batch sums.
    static func loss(_ logits: MLXArray, _ targets: MLXArray, alpha: Float, gamma: Float) -> MLXArray {
        let probability = sigmoid(logits)
        var loss = crossEntropy(logits, targets)
        if gamma != 0 {
            let matched = probability * targets + (1 - probability) * (1 - targets)
            loss = loss * pow(1 - matched, gamma)
        }
        if alpha >= 0 {
            loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
        }
        return loss.mean(axis: 1).sum()
    }
}

/// Box overlap, in the two forms the objective needs: every pair, and the matched pairs alone.
enum NFKSAM3BoxMetrics {

    /// `(x1, y1, x2, y2)` → `(cx, cy, w, h)`.
    static func cornersToCenter(_ boxes: MLXArray) -> MLXArray {
        let x1 = boxes[.ellipsis, 0 ..< 1], y1 = boxes[.ellipsis, 1 ..< 2]
        let x2 = boxes[.ellipsis, 2 ..< 3], y2 = boxes[.ellipsis, 3 ..< 4]
        return concatenated([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis: -1)
    }

    private static func area(_ boxes: MLXArray) -> MLXArray {
        (boxes[.ellipsis, 2] - boxes[.ellipsis, 0]) * (boxes[.ellipsis, 3] - boxes[.ellipsis, 1])
    }

    /// The generalized IoU of every box in `a` against every box in `b`, `[A, B]`.
    static func generalizedIoU(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let left = MLX.maximum(a[0..., .newAxis, 0 ..< 2], b[.newAxis, 0..., 0 ..< 2])
        let right = MLX.minimum(a[0..., .newAxis, 2 ..< 4], b[.newAxis, 0..., 2 ..< 4])
        let side = clip(right - left, min: 0)
        let intersection = side[.ellipsis, 0] * side[.ellipsis, 1]
        let union = area(a).reshaped([-1, 1]) + area(b).reshaped([1, -1]) - intersection
        let overlap = intersection / union

        let outerLeft = MLX.minimum(a[0..., .newAxis, 0 ..< 2], b[.newAxis, 0..., 0 ..< 2])
        let outerRight = MLX.maximum(a[0..., .newAxis, 2 ..< 4], b[.newAxis, 0..., 2 ..< 4])
        let outerSide = clip(outerRight - outerLeft, min: 0)
        let enclosing = outerSide[.ellipsis, 0] * outerSide[.ellipsis, 1]
        return overlap - (enclosing - union) / enclosing
    }

    /// The IoU of `a[i]` against `b[i]`, `[N]`.
    static func diagonalIoU(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let left = MLX.maximum(a[0..., 0 ..< 2], b[0..., 0 ..< 2])
        let right = MLX.minimum(a[0..., 2 ..< 4], b[0..., 2 ..< 4])
        let side = clip(right - left, min: 0)
        let intersection = side[0..., 0] * side[0..., 1]
        return intersection / (area(a) + area(b) - intersection)
    }

    /// The generalized IoU of `a[i]` against `b[i]`, `[N]`.
    static func diagonalGeneralizedIoU(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let left = MLX.maximum(a[0..., 0 ..< 2], b[0..., 0 ..< 2])
        let right = MLX.minimum(a[0..., 2 ..< 4], b[0..., 2 ..< 4])
        let side = clip(right - left, min: 0)
        let intersection = side[0..., 0] * side[0..., 1]
        let union = area(a) + area(b) - intersection
        let outerLeft = MLX.minimum(a[0..., 0 ..< 2], b[0..., 0 ..< 2])
        let outerRight = MLX.maximum(a[0..., 2 ..< 4], b[0..., 2 ..< 4])
        let outerSide = clip(outerRight - outerLeft, min: 0)
        let enclosing = outerSide[0..., 0] * outerSide[0..., 1]
        return intersection / union - (enclosing - union) / enclosing
    }
}

/// The rectangular assignment problem, by the shortest-augmenting-path Hungarian method.
///
/// A DETR objective is undefined without it: the loss has to know which query answers which target
/// before it can score either, and the answer is the assignment of minimum total cost. `scipy`'s
/// `linear_sum_assignment` is what the reference calls; this is the same algorithm, on the host,
/// because the result is a set of indices rather than anything differentiable.
enum NFKMLXHungarian {

    /// - Parameters:
    ///   - cost: a `rows × columns` matrix in row-major order.
    /// - Returns: for each column, the row it is assigned, in column order. Columns are the targets,
    ///   which never outnumber the queries in this objective.
    static func assign(cost: [Float], rows: Int, columns: Int) -> [Int] {
        guard rows > 0, columns > 0, columns <= rows else { return [] }
        // The algorithm below assigns every ROW, so it runs over the transpose: targets become its
        // rows, which are the side that must all be matched.
        let n = columns, m = rows
        func at(_ row: Int, _ column: Int) -> Float { cost[(column - 1) * columns + (row - 1)] }

        var u = [Float](repeating: 0, count: n + 1)
        var v = [Float](repeating: 0, count: m + 1)
        var assignment = [Int](repeating: 0, count: m + 1)
        var path = [Int](repeating: 0, count: m + 1)

        for row in 1 ... n {
            assignment[0] = row
            var column = 0
            var minimum = [Float](repeating: .greatestFiniteMagnitude, count: m + 1)
            var used = [Bool](repeating: false, count: m + 1)
            repeat {
                used[column] = true
                let currentRow = assignment[column]
                var delta = Float.greatestFiniteMagnitude
                var next = 0
                for candidate in 1 ... m where !used[candidate] {
                    let value = at(currentRow, candidate) - u[currentRow] - v[candidate]
                    if value < minimum[candidate] {
                        minimum[candidate] = value
                        path[candidate] = column
                    }
                    if minimum[candidate] < delta {
                        delta = minimum[candidate]
                        next = candidate
                    }
                }
                for candidate in 0 ... m {
                    if used[candidate] {
                        u[assignment[candidate]] += delta
                        v[candidate] -= delta
                    } else {
                        minimum[candidate] -= delta
                    }
                }
                column = next
            } while assignment[column] != 0
            repeat {
                let previous = path[column]
                assignment[column] = assignment[previous]
                column = previous
            } while column != 0
        }

        var chosen = [Int](repeating: 0, count: n)
        for candidate in 1 ... m where assignment[candidate] != 0 {
            chosen[assignment[candidate] - 1] = candidate - 1
        }
        return chosen
    }
}

/// Carries the current step's frozen encoder output from the batch source to the loss.
private final class NFKMLXSAM3Batch: @unchecked Sendable {
    var levels: [MLXArray] = []
    var positions: [MLXArray] = []
    var prompt = MLXArray(Float(0))
    var promptValid: MLXArray?
}

extension NFKMLXSAM3 {

    /// Builds the detector itself, ready to fine-tune, rather than the whole image model.
    ///
    /// - Parameters:
    ///   - weightsURL: a released or fine-tuned checkpoint. Nil leaves the detector at its random
    ///     initialization.
    ///   - configuration: the detector's geometry, from ``detectorConfiguration(fromHuggingFace:)``.
    public static func network(weightsURL: URL?,
                               configuration: NFKMLXSAM3DetectorConfiguration = .base) throws -> NFKMLXSAM3DetectorNet {
        let net = NFKMLXSAM3DetectorNet(configuration)
        if let weightsURL {
            try loadDetectorWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// What the frozen encoders produce for one image and prompt, which a fine-tune reuses at every
    /// step rather than recomputing.
    ///
    /// This is the whole reason a head retarget is cheap here: the 850M parameters a run does not
    /// train are also 850M it does not run more than once per image.
    public static func encode(image: MLXArray, tokens: MLXArray, valid: MLXArray?,
                              using model: NFKMLXSAM3ImageModel)
        -> (levels: [MLXArray], positions: [MLXArray], prompt: MLXArray) {
        let levels = Array(model.vision(image).dropLast())
        let width = model.vision.configuration.fpnHiddenSize
        let positions = levels.map {
            NFKMLXSAM2PositionEmbedding.sine(height: $0.dim(1), width: $0.dim(2), features: width)
        }
        let padding = valid.map { (1 - $0) * -Float.greatestFiniteMagnitude }
        let prompt = model.text.prompt(tokens, padding: padding)
        eval(levels, positions, prompt)
        return (levels, positions, prompt)
    }

    /// The sine position encodings a set of FPN levels takes, which depend only on their sizes.
    ///
    /// ``encode(image:tokens:valid:using:)`` returns these alongside the levels. This is the same
    /// thing for a consumer who cached the levels from an earlier run and no longer holds the
    /// encoders in memory.
    public static func positionEncodings(for levels: [MLXArray], width: Int = 256) -> [MLXArray] {
        levels.map {
            NFKMLXSAM2PositionEmbedding.sine(height: $0.dim(1), width: $0.dim(2), features: width)
        }
    }

    /// Fine-tunes the detector on a consumer's own annotated images, returning the loss from each step.
    ///
    /// - Parameters:
    ///   - detector: the detector to train, from ``network(weightsURL:configuration:)``.
    ///   - examples: supplies one annotated image per step: what ``encode(image:tokens:valid:using:)``
    ///     returned for it, the prompt's validity mask, and the ground-truth boxes `[T, 4]` as
    ///     `(cx, cy, w, h)` in `0...1`. An empty box set is an image the prompt names nothing in,
    ///     which trains the presence head.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `detector`.
    ///   - objective: the supervised loss.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.AdamW` (bias-corrected)
    ///     at 8e-5, the rate its `odinw_text_only_train.yaml` gives the transformer, with weight decay
    ///     0.1 on every parameter but biases and `nn.LayerNorm` weights.
    ///   - steps: how many images to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update: 0.1, the reference's.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's inverse
    ///     square root (timescale 20) with a 20-step linear warm-up and a 20-step cool-down at the end
    ///     of the run, when the reference optimizer runs. With a caller's optimizer, nil holds that
    ///     optimizer's rate constant.
    ///   - checkpoint: writes the detector periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ detector: NFKMLXSAM3DetectorNet,
        examples: (Int) -> (levels: [MLXArray], positions: [MLXArray], prompt: MLXArray,
                            promptValid: MLXArray?, boxes: MLXArray),
        trainable: NFKMLXSAM3Trainable = .detector,
        objective: NFKMLXSAM3Objective = NFKMLXSAM3Objective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 0.1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let carried = NFKMLXSAM3Batch()
        return try NFKMLXFineTune.run(
            detector,
            freezing: { apply(trainable, to: detector) },
            optimizer: optimizer,
            reference: {
                NFKMLXReferenceOptimizers.adamW(
                    learningRate: 8e-5, weightDecay: 0.1,
                    exempting: NFKMLXReferenceOptimizers.biasOrLayerNorm(
                        NFKMLXReferenceOptimizers.layerNormPrefixes(in: detector)))
            },
            referenceSchedule: { .inverseSquareRoot(steps: steps, timescale: 20, warmupSteps: 20, cooldownSteps: 20) },
            steps: steps,
            sample: { step in
                let example = examples(step)
                carried.levels = example.levels
                carried.positions = example.positions
                carried.prompt = example.prompt
                carried.promptValid = example.promptValid
                return example.boxes
            },
            loss: { detector, boxes in
                objective(detector, levels: carried.levels, positions: carried.positions,
                          prompt: carried.prompt, promptValid: carried.promptValid, targets: boxes)
            },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// Freezes the DETR encoder for the cheaper level, or leaves the whole detector trainable.
    private static func apply(_ trainable: NFKMLXSAM3Trainable, to detector: NFKMLXSAM3DetectorNet) {
        switch trainable {
        case .detector:
            detector.unfreeze()
        case .decoder:
            detector.unfreeze()
            detector.encoder.freeze()
        }
    }
}
