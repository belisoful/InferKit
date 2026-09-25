//
//  NFKMLXRTDetrTraining.swift
//  InferKitMLX
//
//  The objective an RT-DETR fine-tune minimizes: transformers' `RTDetrLoss` (4.57), the loss the
//  release's model class computes from `labels=`.
//
//  Each set of predictions is matched one-to-one to the ground truth by the Hungarian assignment over a
//  focal class cost, an L1 box cost, and a GIoU cost (2, 5, 2). A matched query's class target is its
//  box's IoU with the truth (varifocal), every other query's is zero, and the matched boxes add an L1 and
//  a `1 − GIoU` term. The sum runs over the final layer, every earlier decoder layer and the encoder's
//  top-k proposals (each matched afresh), and the contrastive-denoising queries, which are matched to the
//  truth they were noised from rather than by cost.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/*!
 @abstract A set of RT-DETR predictions the objective scores: class logits `[batch, queries, classes]` and
 boxes `[batch, queries, 4]` as normalized centers and sizes.
 @discussion Introduced in InferKit 0.5.0.
 */
public struct NFKMLXRTDetrPredictions {
    public var logits: MLXArray
    public var boxes: MLXArray

    public init(logits: MLXArray, boxes: MLXArray) {
        self.logits = logits
        self.boxes = boxes
    }
}

/*!
 @abstract One image's ground truth for an RT-DETR fine-tune: class indices and normalized center boxes.
 @discussion Introduced in InferKit 0.5.0.
 */
public struct NFKMLXRTDetrTarget {
    public var classes: [Int]
    /// `[count, 4]` normalized `(cx, cy, w, h)`.
    public var boxes: MLXArray

    public init(classes: [Int], boxes: MLXArray) {
        self.classes = classes
        self.boxes = boxes
    }
}

/*!
 @abstract The objective an RT-DETR fine-tune minimizes: transformers' `RTDetrLoss`.
 @discussion `loss(final:auxiliary:denoising:denoisingPositives:denoisingGroups:targets:)` sums the
 weighted varifocal, L1, and GIoU terms over every prediction set. Measured against transformers by
 `run_reference.py rtdetr_loss`. Introduced in InferKit 0.5.0.
 */
public struct NFKMLXRTDetrObjective: Sendable {
    public var matcherClassCost: Float = 2
    public var matcherBoxCost: Float = 5
    public var matcherGIoUCost: Float = 2
    public var matcherAlpha: Float = 0.25
    public var matcherGamma: Float = 2
    public var varifocalWeight: Float = 1
    public var boxWeight: Float = 5
    public var generalizedIoUWeight: Float = 2
    public var focalAlpha: Float = 0.75
    public var focalGamma: Float = 2

    public init() {}

    /// The matched `(query, target)` pairs of one image, by minimum total cost.
    func match(logits: MLXArray, boxes: MLXArray, target: NFKMLXRTDetrTarget) -> [(query: Int, target: Int)] {
        let count = target.classes.count
        guard count > 0 else {
            return []
        }
        let (logits, boxes) = (stopGradient(logits), stopGradient(boxes))
        let queries = logits.dim(0)
        let probabilities = sigmoid(take(logits, MLXArray(target.classes.map { Int32($0) }), axis: 1))   // [Q, T]
        let negative = (1 - matcherAlpha) * pow(probabilities, matcherGamma) * -log(1 - probabilities + 1e-8)
        let positive = matcherAlpha * pow(1 - probabilities, matcherGamma) * -log(probabilities + 1e-8)
        let classCost = positive - negative
        let boxCost = abs(boxes.expandedDimensions(axis: 1) - target.boxes.expandedDimensions(axis: 0)).sum(axis: -1)
        let giouCost = -NFKSAM3BoxMetrics.generalizedIoU(NFKSAM3Boxes.centerToCorners(boxes),
                                                         NFKSAM3Boxes.centerToCorners(target.boxes))
        let cost = matcherBoxCost * boxCost + matcherClassCost * classCost + matcherGIoUCost * giouCost
        eval(cost)
        let assigned = NFKMLXHungarian.assign(cost: cost.reshaped([-1]).asArray(Float.self), rows: queries, columns: count)
        return assigned.enumerated().map { (query: $0.element, target: $0.offset) }.sorted { $0.query < $1.query }
    }

    /// The three weighted terms for one prediction set under a matching.
    func terms(_ predictions: NFKMLXRTDetrPredictions, matches: [[(query: Int, target: Int)]],
               targets: [NFKMLXRTDetrTarget], boxCount: Float) -> (varifocal: MLXArray, box: MLXArray, generalizedIoU: MLXArray) {
        let (batch, queries, classes) = (predictions.logits.dim(0), predictions.logits.dim(1), predictions.logits.dim(2))
        var flatQueries = [Int32](), targetBoxes = [MLXArray](), targetClasses = [Int]()
        for image in 0 ..< batch {
            for pair in matches[image] {
                flatQueries.append(Int32(image * queries + pair.query))
                targetBoxes.append(targets[image].boxes[pair.target])
                targetClasses.append(targets[image].classes[pair.target])
            }
        }
        let logits = predictions.logits
        guard !flatQueries.isEmpty else {
            let zero = MLXArray(Float(0))
            let score = sigmoid(stopGradient(logits))
            let weight = focalAlpha * pow(score, focalGamma)
            let bce = weight * (maximum(logits, 0) + log1p(exp(-abs(logits))))
            return (varifocalWeight * bce.mean(axis: 1).sum() * Float(queries) / boxCount, zero, zero)
        }
        let index = MLXArray(flatQueries)
        let matchedBoxes = take(predictions.boxes.reshaped([batch * queries, 4]), index, axis: 0)
        let truthBoxes = stacked(targetBoxes, axis: 0)
        let predictedCorners = NFKSAM3Boxes.centerToCorners(matchedBoxes)
        let truthCorners = NFKSAM3Boxes.centerToCorners(truthBoxes)

        // Varifocal: the matched queries' class targets are their boxes' IoU with the truth.
        let iou = stopGradient(NFKSAM3BoxMetrics.diagonalIoU(predictedCorners, truthCorners))
        let overlaps = iou.asArray(Float.self)
        var scores = [Float](repeating: 0, count: batch * queries * classes)
        var onehot = scores
        for (position, (query, cls)) in zip(flatQueries, targetClasses).enumerated() {
            scores[Int(query) * classes + cls] = overlaps[position]
            onehot[Int(query) * classes + cls] = 1
        }
        let targetScore = MLXArray(scores).reshaped([batch, queries, classes])
        let isTarget = MLXArray(onehot).reshaped([batch, queries, classes])
        let predictedScore = sigmoid(stopGradient(logits))
        let weight = focalAlpha * pow(predictedScore, focalGamma) * (1 - isTarget) + targetScore
        let bce = weight * (maximum(logits, 0) - logits * targetScore + log1p(exp(-abs(logits))))
        let varifocal = bce.mean(axis: 1).sum() * Float(queries) / boxCount

        let box = abs(matchedBoxes - truthBoxes).sum() / boxCount
        let giou = (1 - NFKSAM3BoxMetrics.diagonalGeneralizedIoU(predictedCorners, truthCorners)).sum() / boxCount
        return (varifocalWeight * varifocal, boxWeight * box, generalizedIoUWeight * giou)
    }

    /// Every weighted term, keyed as transformers keys them: `loss_vfl`, `loss_bbox`, and `loss_giou` for
    /// the final set, with `_aux_<i>` for the auxiliary sets and `_dn_<i>` for the denoising sets.
    ///
    /// - Parameters:
    ///   - auxiliary: the earlier decoder layers' sets, then the encoder's top-k proposals.
    ///   - denoisingPositives: for each image, the denoising query indices that carry its targets in group
    ///     order (`dn_positive_idx`), `targets.count · groups` of them.
    public func losses(final: NFKMLXRTDetrPredictions, auxiliary: [NFKMLXRTDetrPredictions] = [],
                       denoising: [NFKMLXRTDetrPredictions] = [], denoisingPositives: [[Int]] = [],
                       denoisingGroups: Int = 0, targets: [NFKMLXRTDetrTarget]) -> [String: MLXArray] {
        let boxCount = Float(max(targets.map(\.classes.count).reduce(0, +), 1))
        var result = [String: MLXArray]()
        func record(_ t: (varifocal: MLXArray, box: MLXArray, generalizedIoU: MLXArray), suffix: String) {
            result["loss_vfl" + suffix] = t.varifocal
            result["loss_bbox" + suffix] = t.box
            result["loss_giou" + suffix] = t.generalizedIoU
        }
        for (index, predictions) in ([final] + auxiliary).enumerated() {
            let matches = (0 ..< targets.count).map {
                match(logits: predictions.logits[$0], boxes: predictions.boxes[$0], target: targets[$0])
            }
            record(terms(predictions, matches: matches, targets: targets, boxCount: boxCount),
                   suffix: index == 0 ? "" : "_aux_\(index - 1)")
        }
        let denoisingMatches = targets.enumerated().map { image, target in
            (0 ..< (denoisingPositives.indices.contains(image) ? denoisingPositives[image].count : 0)).map { position in
                (query: denoisingPositives[image][position], target: position % max(target.classes.count, 1))
            }
        }
        for (index, predictions) in denoising.enumerated() {
            record(terms(predictions, matches: denoisingMatches, targets: targets,
                         boxCount: boxCount * Float(denoisingGroups)),
                   suffix: "_dn_\(index)")
        }
        return result
    }

    /// The total of `losses(…)`, the value a fine-tune step differentiates.
    public func loss(final: NFKMLXRTDetrPredictions, auxiliary: [NFKMLXRTDetrPredictions] = [],
                     denoising: [NFKMLXRTDetrPredictions] = [], denoisingPositives: [[Int]] = [],
                     denoisingGroups: Int = 0, targets: [NFKMLXRTDetrTarget]) -> MLXArray {
        losses(final: final, auxiliary: auxiliary, denoising: denoising, denoisingPositives: denoisingPositives,
               denoisingGroups: denoisingGroups, targets: targets).values.reduce(MLXArray(Float(0)), +)
    }
}

/*!
 @abstract The contrastive-denoising queries of one training batch: noised copies of the ground truth the
 decoder must restore, placed ahead of the matching queries.
 @discussion Each group holds every image's truth twice, padded to the batch's largest count: a positive
 copy whose corners move by up to half the box's size, and a negative copy moved by between one and two
 halves. A quarter of the copies (at the reference's ratio 0.5) take a random class. The attention mask
 keeps the matching queries from reading any denoising query and each group from reading another.
 Introduced in InferKit 0.5.0.
 */
public struct NFKMLXRTDetrDenoisingGroup {
    /// `[batch, count]` class indices, the padding class (`classCount`) where an image has fewer boxes.
    public let classes: MLXArray
    /// `[batch, count, 4]` noised boxes, inverse-sigmoided.
    public let boxesUnactivated: MLXArray
    /// `[count + queries, count + queries]` additive self-attention mask.
    public let mask: MLXArray
    /// For each image, the positive queries in group order, which the objective matches to its boxes.
    public let positives: [[Int]]
    public let groups: Int
    public let count: Int

    /// The random draws the noise is built from, in the reference's order; tests inject them.
    struct Draws {
        var labelChance: [Float]      // [batch, count], uniform in 0..<1
        var newLabels: [Int]          // [batch, count], uniform classes
        var signs: [Float]            // [batch, count, 4], ±1
        var magnitudes: [Float]       // [batch, count, 4], uniform in 0..<1
    }

    /// transformers' `get_contrastive_denoising_training_group`, the original's construction. Nil when
    /// no image has a box.
    public static func make(targets: [NFKMLXRTDetrTarget], classCount: Int, queryCount: Int,
                            denoisingQueries: Int = 100, labelNoiseRatio: Float = 0.5,
                            boxNoiseScale: Float = 1) -> NFKMLXRTDetrDenoisingGroup? {
        make(targets: targets, classCount: classCount, queryCount: queryCount, denoisingQueries: denoisingQueries,
             labelNoiseRatio: labelNoiseRatio, boxNoiseScale: boxNoiseScale) { batch, count in
            let shape = [batch, count]
            let draws = [MLXRandom.uniform(0 ..< 1, shape), MLXRandom.randInt(0 ..< classCount, shape).asType(.float32),
                         MLXRandom.randInt(0 ..< 2, shape + [4]).asType(.float32) * 2 - 1,
                         MLXRandom.uniform(0 ..< 1, shape + [4])]
            eval(draws)
            return Draws(labelChance: draws[0].asArray(Float.self), newLabels: draws[1].asArray(Float.self).map { Int($0) },
                         signs: draws[2].asArray(Float.self), magnitudes: draws[3].asArray(Float.self))
        }
    }

    static func make(targets: [NFKMLXRTDetrTarget], classCount: Int, queryCount: Int, denoisingQueries: Int,
                     labelNoiseRatio: Float, boxNoiseScale: Float,
                     draw: (_ batch: Int, _ count: Int) -> Draws) -> NFKMLXRTDetrDenoisingGroup? {
        let counts = targets.map(\.classes.count)
        let largest = counts.max() ?? 0
        guard denoisingQueries > 0, largest > 0 else {
            return nil
        }
        let groups = max(denoisingQueries / largest, 1)
        let (batch, count) = (targets.count, largest * 2 * groups)
        let truths = targets.map { $0.classes.isEmpty ? [] : $0.boxes.asArray(Float.self) }
        let draws = draw(batch, count)

        var classes = [Int32](), boxes = [Float](), positives = [[Int]](repeating: [], count: batch)
        for image in 0 ..< batch {
            for slot in 0 ..< count {
                let (box, negative) = (slot % largest, (slot / largest) % 2 == 1)
                let flat = image * count + slot
                guard box < counts[image] else {
                    classes.append(Int32(classCount))
                    boxes += [Float](repeating: boxNoiseScale > 0 ? NFKRTDetrDecoder.inverseSigmoidScalar(0) : 0, count: 4)
                    continue
                }
                if !negative {
                    positives[image].append(slot)
                }
                let relabel = labelNoiseRatio > 0 && draws.labelChance[flat] < labelNoiseRatio * 0.5
                classes.append(Int32(relabel ? draws.newLabels[flat] : targets[image].classes[box]))

                let (cx, cy, w, h) = (truths[image][box * 4], truths[image][box * 4 + 1],
                                      truths[image][box * 4 + 2], truths[image][box * 4 + 3])
                var corners = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                if boxNoiseScale > 0 {
                    let half = [w / 2, h / 2, w / 2, h / 2]
                    for side in 0 ..< 4 {
                        let magnitude = draws.magnitudes[flat * 4 + side] + (negative ? 1 : 0)
                        corners[side] = min(max(corners[side] + draws.signs[flat * 4 + side] * magnitude
                                                * half[side] * boxNoiseScale, 0), 1)
                    }
                }
                let center = [(corners[0] + corners[2]) / 2, (corners[1] + corners[3]) / 2,
                              corners[2] - corners[0], corners[3] - corners[1]]
                boxes += boxNoiseScale > 0 ? center.map { NFKRTDetrDecoder.inverseSigmoidScalar($0) } : center
            }
        }

        let total = count + queryCount
        var mask = [Float](repeating: 0, count: total * total)
        for row in count ..< total {
            for column in 0 ..< count {
                mask[row * total + column] = -.infinity
            }
        }
        for group in 0 ..< groups {
            let (start, end) = (largest * 2 * group, largest * 2 * (group + 1))
            for row in start ..< end {
                for column in 0 ..< count where column < start || column >= end {
                    mask[row * total + column] = -.infinity
                }
            }
        }
        return NFKMLXRTDetrDenoisingGroup(classes: MLXArray(classes, [batch, count]),
                                          boxesUnactivated: MLXArray(boxes, [batch, count, 4]),
                                          mask: MLXArray(mask, [total, total]), positives: positives,
                                          groups: groups, count: count)
    }
}

/*!
 @abstract Every prediction set one training forward scores: the final layer, the auxiliary sets (each
 earlier decoder layer, then the encoder's top-k proposals), and each layer's denoising set.
 @discussion Introduced in InferKit 0.5.0.
 */
public struct NFKMLXRTDetrTrainingOutputs {
    public let final: NFKMLXRTDetrPredictions
    public let auxiliary: [NFKMLXRTDetrPredictions]
    public let denoising: [NFKMLXRTDetrPredictions]
}

extension NFKMLXRTDetrNet {

    /// The training forward over a batch `[batch, H, W, 3]`: the top-k query selection per image, the
    /// denoising queries ahead of the matching ones, and the decoder's training path.
    ///
    /// The selected queries' features and every starting box are detached, as the reference detaches
    /// them; the encoder's own top-k proposals keep their gradient. A `denoising` group is read only when
    /// the network was built with ``NFKMLXRTDetrConfiguration/denoisingQueries`` above zero.
    ///
    /// - Since: InferKit 0.5.0
    public func trainingOutputs(_ pixels: MLXArray, denoising: NFKMLXRTDetrDenoisingGroup?) -> NFKMLXRTDetrTrainingOutputs {
        let encoded = encode(pixels)
        let (batch, queries) = (pixels.dim(0), config.numQueries)
        let order = stopGradient(argSort(-encoded.encClass.max(axis: -1), axis: 1)[0..., 0 ..< queries])   // [B, Q]
        func selected(_ x: MLXArray) -> MLXArray {
            takeAlong(x, broadcast(order.expandedDimensions(axis: -1), to: [batch, queries, x.dim(2)]), axis: 1)
        }
        var referenceUnact = selected(encoded.encCoord)
        let proposals = NFKMLXRTDetrPredictions(logits: selected(encoded.encClass), boxes: sigmoid(referenceUnact))
        var target = stopGradient(selected(encoded.outputMemory))
        var mask: MLXArray?
        var denoisingCount = 0
        if let denoising, let embedding = denoisingClassEmbed {
            let embedded = embedding(denoising.classes)
            let padding = (denoising.classes .== Int32(config.numLabels)).expandedDimensions(axis: -1)
            target = concatenated([MLX.where(padding, stopGradient(embedded), embedded), target], axis: 1)
            referenceUnact = concatenated([denoising.boxesUnactivated, referenceUnact], axis: 1)
            mask = denoising.mask
            denoisingCount = denoising.count
        }
        let (logits, boxes) = decoder(target, value: encoded.sourceFlatten,
                                      referencePointsUnact: stopGradient(referenceUnact), shapes: encoded.shapes,
                                      mask: mask, training: true)
        let matching = zip(logits, boxes).map {
            NFKMLXRTDetrPredictions(logits: $0[0..., denoisingCount...], boxes: $1[0..., denoisingCount...])
        }
        let denoisingSets = denoisingCount == 0 ? [] : zip(logits, boxes).map {
            NFKMLXRTDetrPredictions(logits: $0[0..., ..<denoisingCount], boxes: $1[0..., ..<denoisingCount])
        }
        return NFKMLXRTDetrTrainingOutputs(final: matching[matching.count - 1],
                                           auxiliary: Array(matching.dropLast()) + [proposals],
                                           denoising: denoisingSets)
    }
}

extension NFKMLXRTDetrObjective {

    /// The loss of one training forward, with its denoising group.
    public func loss(_ outputs: NFKMLXRTDetrTrainingOutputs, denoising: NFKMLXRTDetrDenoisingGroup?,
                     targets: [NFKMLXRTDetrTarget]) -> MLXArray {
        loss(final: outputs.final, auxiliary: outputs.auxiliary,
             denoising: denoising == nil ? [] : outputs.denoising,
             denoisingPositives: denoising?.positives ?? [], denoisingGroups: denoising?.groups ?? 0,
             targets: targets)
    }
}

/*!
 @abstract What an RT-DETR fine-tune updates.
 @discussion Introduced in InferKit 0.5.0.
 */
public enum NFKMLXRTDetrTrainable: Sendable {

    /// Every parameter the reference trains: all but the backbone's stem and its batch normalizations,
    /// which the reference freezes (`freeze_at: 0`, `freeze_norm: True`).
    case everything

    /// The decoder and the query selection alone, with the backbone and the hybrid encoder frozen.
    case decoder
}

extension NFKMLXRTDetrNet {

    /// The original implementation's `_reset_parameters`: the class heads' biases at a prior of 0.01, the
    /// box heads' last layers at zero, and Xavier-uniform weights on the query-selection projection and
    /// the query-position head.
    func initializeHeads() {
        let prior = -Foundation.log((1 - 0.01) / 0.01)
        let classHeads = [encScoreHead] + decoder.classEmbed
        let boxHeads = [encBboxHead] + decoder.bboxEmbed
        var updates = [(String, MLXArray)]()
        for (index, head) in classHeads.enumerated() {
            let name = index == 0 ? "enc_score_head" : "decoder.class_embed.\(index - 1)"
            updates.append(("\(name).bias", MLXArray.ones(head.bias!.shape) * Float(prior)))
        }
        for (index, head) in boxHeads.enumerated() {
            let name = index == 0 ? "enc_bbox_head" : "decoder.bbox_embed.\(index - 1)"
            let last = head.layers.count - 1
            updates.append(("\(name).layers.\(last).weight", MLXArray.zeros(head.layers[last].weight.shape)))
            updates.append(("\(name).layers.\(last).bias", MLXArray.zeros(head.layers[last].bias!.shape)))
        }
        func xavier(_ layer: Linear) -> MLXArray {
            let (fanOut, fanIn) = (layer.weight.dim(0), layer.weight.dim(1))
            let bound = Float((6 / Double(fanIn + fanOut)).squareRoot())
            return MLXRandom.uniform(-bound ..< bound, layer.weight.shape)
        }
        updates.append(("enc_output.0.weight", xavier(encOutput[0] as! Linear)))
        for index in 0 ..< 2 {
            updates.append(("decoder.query_pos_head.layers.\(index).weight", xavier(decoder.queryPosHead.layers[index])))
        }
        update(parameters: ModuleParameters.unflattened(updates))
        eval(self)
    }
}

extension NFKMLXRTDetr {

    /// Builds an RT-DETR network at a released size for `classCount` classes, ready to fine-tune.
    ///
    /// The heads start at the original implementation's initialization, and `weightsURL` (a release or
    /// a file `NFKMLXWeights.save` wrote) transfers every tensor whose shape matches, as the original's
    /// `load_tuning_state` does: at a class count other than the checkpoint's, the class heads and the
    /// denoising class embedding keep their fresh initialization and everything else loads. The network
    /// carries the reference's 100 contrastive-denoising queries.
    ///
    /// - Since: InferKit 0.5.0
    public static func network(variant: NFKMLXRTDetrVariant = .r50vd, classCount: Int = 80,
                               weightsURL: URL?) throws -> NFKMLXRTDetrNet {
        var configuration = specs(for: variant).configuration
        configuration.numLabels = classCount
        configuration.denoisingQueries = 100
        let net = NFKMLXRTDetrNet(configuration)
        net.initializeHeads()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL, matchingShapesOnly: true)
        }
        return net
    }

    /// A release's training setup, from its configuration file in the original repository
    /// (`configs/rtdetr/` for RT-DETR, `configs/rtdetrv2/` for RT-DETRv2).
    struct ReferenceRecipe {
        /// 1 or 2, which decides the original's module names.
        let version: Int
        /// `freeze_at: 0`: the backbone's stem does not train.
        let freezesStem: Bool
        /// `freeze_norm: True`: the backbone's batch normalizations are `FrozenBatchNorm2d`.
        let freezesBackboneNormalizations: Bool
        /// The `optimizer.params` groups in order, each a pattern over the original's parameter names,
        /// a rate as a multiple of the base 1e-4, and a weight decay where the group sets one.
        let groups: [(pattern: NSRegularExpression, rateScale: Float, weightDecay: Float?)]
        /// `lr_warmup_scheduler`: the updates over which the rate rises linearly, or zero.
        let warmupSteps: Int
    }

    static func referenceRecipe(for variant: NFKMLXRTDetrVariant) -> ReferenceRecipe {
        func groups(_ entries: [(String, Float, Float?)]) -> [(pattern: NSRegularExpression, rateScale: Float, weightDecay: Float?)] {
            entries.map { (try! NSRegularExpression(pattern: $0.0), $0.1, $0.2) }
        }
        switch variant {
        case .r18vd:
            return ReferenceRecipe(version: 1, freezesStem: false, freezesBackboneNormalizations: false, groups: groups([
                ("^(?=.*backbone)(?=.*norm).*$", 0.1, 0),
                ("^(?=.*backbone)(?!.*norm).*$", 0.1, nil),
                ("^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", 1, 0),
            ]), warmupSteps: 0)
        case .r34vd:
            return ReferenceRecipe(version: 1, freezesStem: false, freezesBackboneNormalizations: false, groups: groups([
                ("^(?=.*backbone)(?=.*norm|bn).*$", 0.1, 0),
                ("^(?=.*backbone)(?!.*norm|bn).*$", 0.1, nil),
                ("^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$", 1, 0),
            ]), warmupSteps: 0)
        case .r50vd:
            return ReferenceRecipe(version: 1, freezesStem: true, freezesBackboneNormalizations: true, groups: groups([
                ("backbone", 0.1, nil),
                ("^(?=.*encoder(?=.*bias|.*norm.*weight)).*$", 1, 0),
                ("^(?=.*decoder(?=.*bias|.*norm.*weight)).*$", 1, 0),
            ]), warmupSteps: 0)
        case .r101vd:
            // The file's `optimizer.params` replaces the included list whole, so no group is undecayed.
            return ReferenceRecipe(version: 1, freezesStem: true, freezesBackboneNormalizations: true, groups: groups([
                ("backbone", 0.01, nil),
            ]), warmupSteps: 0)
        case .v2R18VD:
            return ReferenceRecipe(version: 2, freezesStem: false, freezesBackboneNormalizations: false, groups: groups([
                ("^(?=.*(?:norm|bn)).*$", 1, 0),
            ]), warmupSteps: 2000)
        case .v2R34VD:
            return ReferenceRecipe(version: 2, freezesStem: false, freezesBackboneNormalizations: false, groups: groups([
                ("^(?=.*backbone)(?!.*norm|bn).*$", 0.5, nil),
                ("^(?=.*backbone)(?=.*norm|bn).*$", 0.5, 0),
                ("^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$", 1, 0),
            ]), warmupSteps: 2000)
        case .v2R50VD:
            return ReferenceRecipe(version: 2, freezesStem: true, freezesBackboneNormalizations: true, groups: groups([
                ("^(?=.*backbone)(?!.*norm).*$", 0.1, nil),
                ("^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", 1, 0),
            ]), warmupSteps: 2000)
        case .v2R101VD:
            return ReferenceRecipe(version: 2, freezesStem: true, freezesBackboneNormalizations: true, groups: groups([
                ("^(?=.*backbone)(?!.*norm|bn).*$", 0.01, nil),
                ("^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", 1, 0),
            ]), warmupSteps: 2000)
        }
    }

    /// The original implementation's name for a parameter, as far as its optimizer patterns read it:
    /// which of `backbone`, `encoder`, and `decoder` it sits under, and whether its name says `norm`,
    /// `bn`, or `bias`.
    ///
    /// transformers moves the query selection and the denoising embedding to the model's root, where
    /// the original keeps them under `decoder.`, and names every input projection by index. The
    /// original names the decoder's input projection `conv` and `norm`; RT-DETRv2 names the encoder's
    /// the same way and its query-selection projection `proj` and `norm`, where RT-DETR indexes both.
    static func originalParameterName(_ key: String, version: Int) -> String {
        let parts = key.split(separator: ".").map(String.init)
        func projection(_ index: String) -> String {
            index == "0" ? "conv" : "norm"
        }
        switch parts[0] {
        case "encoder_input_proj":
            let layer = version == 2 ? projection(parts[2]) : parts[2]
            return (["encoder", "input_proj", parts[1], layer] + parts[3...]).joined(separator: ".")
        case "decoder_input_proj":
            return (["decoder", "input_proj", parts[1], projection(parts[2])] + parts[3...]).joined(separator: ".")
        case "enc_output":
            let layer = version == 2 ? (parts[1] == "0" ? "proj" : "norm") : parts[1]
            return (["decoder", "enc_output", layer] + parts[2...]).joined(separator: ".")
        case "enc_score_head", "enc_bbox_head", "denoising_class_embed":
            return "decoder." + key
        default:
            return key
        }
    }

    /// A parameter's rate (as a multiple of 1e-4) and weight decay under a release's recipe: the first
    /// group whose pattern its original name matches, as `get_optim_params` assigns it, or the base
    /// rate and decay 1e-4.
    static func referenceGroup(for key: String, recipe: ReferenceRecipe) -> (rateScale: Float, weightDecay: Float) {
        let name = originalParameterName(key, version: recipe.version)
        let range = NSRange(name.startIndex..., in: name)
        guard let group = recipe.groups.first(where: { $0.pattern.firstMatch(in: name, range: range) != nil }) else {
            return (1, 1e-4)
        }
        return (group.rateScale, group.weightDecay ?? 1e-4)
    }

    /// The original's AdamW at 1e-4 with betas 0.9 and 0.999, over the release's groups.
    static func referenceOptimizer(for net: NFKMLXRTDetrNet, recipe: ReferenceRecipe) -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: 1e-4, over: net) { referenceGroup(for: $0, recipe: recipe) }
    }

    /// Freezes what the release's recipe freezes (its stem, its backbone's batch normalizations), every
    /// other batch normalization's running statistics, and for ``NFKMLXRTDetrTrainable/decoder`` the
    /// backbone and the hybrid encoder whole.
    static func applyFreezing(_ trainable: NFKMLXRTDetrTrainable, recipe: ReferenceRecipe, to net: NFKMLXRTDetrNet) {
        net.unfreeze()
        if trainable == .decoder {
            net.backbone.freeze()
            net.encoder.freeze()
            net.encoderInputProj.joined().forEach { $0.freeze() }
        }
        if recipe.freezesStem {
            net.backbone.model.embedder.freeze()
        }
        for (_, module) in net.leafModules().flattened() {
            guard let norm = module as? BatchNorm else {
                continue
            }
            // A parent's unfreeze clears the statistics' own freeze.
            norm.freeze(recursive: false, keys: ["running_mean", "running_var"])
        }
        if recipe.freezesBackboneNormalizations {
            for (_, module) in net.backbone.leafModules().flattened() where module is BatchNorm {
                module.freeze()
            }
        }
    }

    /// Fine-tunes an RT-DETR network on a consumer's own labeled images, returning the loss from each
    /// step.
    ///
    /// The whole path is three calls: ``network(variant:classCount:weightsURL:)`` to build, this to
    /// train, and `NFKMLXWeights.save` to write a checkpoint that `backendWithVariant:weightsURL:labels:error:`
    /// loads at its own class count.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(variant:classCount:weightsURL:)``.
    ///   - variant: the release `net` was built from, whose configuration file sets the reference's
    ///     freezing, optimizer groups, and warm-up.
    ///   - examples: supplies one batch per step: images `[batch, H, W, 3]` in `0...1`, the release's
    ///     640 square in the reference, and each image's classes and normalized center boxes.
    ///     Augmentation is the caller's; the reference trains with photometric distortion, zoom-out,
    ///     IoU crop, and flips.
    ///   - trainable: every parameter the release's recipe trains, or the decoder alone.
    ///   - objective: the reference's criterion.
    ///   - optimizer: the update rule. Nil uses the release's AdamW groups.
    ///   - steps: how many batches to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference clips at 0.1.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the release's schedule when the
    ///     reference optimizer runs: RT-DETRv2's linear warm-up over 2,000 updates, and otherwise a constant
    ///     rate (each release's step decay falls at epoch 1000, after its training ends).
    ///   - averagesWeights: keeps the reference's exponential moving average of every weight and running
    ///     statistic (decay `0.9999 · (1 − e^(−updates / 2000))`) and leaves it in `net` at the end, which
    ///     is what the reference evaluates and saves.
    ///   - checkpoint: writes the network periodically, before the average is applied.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is minutes; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXRTDetrNet,
        variant: NFKMLXRTDetrVariant,
        examples: @escaping (Int) -> (images: MLXArray, targets: [NFKMLXRTDetrTarget]),
        trainable: NFKMLXRTDetrTrainable = .everything,
        objective: NFKMLXRTDetrObjective = NFKMLXRTDetrObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 0.1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        averagesWeights: Bool = true,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        var average = averagesWeights ? NFKMLXModelWeightAverage(net) : nil
        let configuration = net.config
        let recipe = referenceRecipe(for: variant)
        let history = try NFKMLXFineTune.run(
            net,
            freezing: { applyFreezing(trainable, recipe: recipe, to: net) },
            optimizer: optimizer,
            reference: { referenceOptimizer(for: net, recipe: recipe) },
            referenceSchedule: { recipe.warmupSteps > 0 ? .linearWarmup(steps: recipe.warmupSteps) : .constant },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                var rows = [Float]()
                for (image, target) in example.targets.enumerated() where !target.classes.isEmpty {
                    let boxes = target.boxes.asArray(Float.self)
                    for (index, cls) in target.classes.enumerated() {
                        rows += [Float(image), Float(cls)] + boxes[(index * 4) ..< (index * 4 + 4)]
                    }
                }
                return [example.images, MLXArray(rows, [rows.count / 6, 6])]
            },
            loss: { net, arrays in
                let rows = arrays[1].asArray(Float.self)
                var classes = [[Int]](repeating: [], count: arrays[0].dim(0))
                var boxes = [[Float]](repeating: [], count: arrays[0].dim(0))
                for row in stride(from: 0, to: rows.count, by: 6) {
                    classes[Int(rows[row])].append(Int(rows[row + 1]))
                    boxes[Int(rows[row])] += rows[(row + 2) ..< (row + 6)]
                }
                let targets = zip(classes, boxes).map {
                    NFKMLXRTDetrTarget(classes: $0, boxes: MLXArray($1, [$0.count, 4]))
                }
                let denoising = NFKMLXRTDetrDenoisingGroup.make(
                    targets: targets, classCount: configuration.numLabels, queryCount: configuration.numQueries,
                    denoisingQueries: configuration.denoisingQueries)
                return objective.loss(net.trainingOutputs(arrays[0], denoising: denoising),
                                      denoising: denoising, targets: targets)
            },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint,
            observer: { step in
                average?.update(from: net)
                return observer?(step) ?? true
            })
        average?.apply(to: net)
        return history
    }
}
