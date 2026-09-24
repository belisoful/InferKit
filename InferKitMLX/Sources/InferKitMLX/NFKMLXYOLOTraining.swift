//
//  NFKMLXYOLOTraining.swift
//  InferKitMLX
//
//  The objective a YOLO detector trains against: ultralytics' `v8DetectionLoss` (ultralytics 8.4.120,
//  AGPL-3.0), which every generation from v8 through YOLO26 shares.
//
//  Each anchor predicts a distribution over `regMax` bins for each side of its box and a logit per
//  class. `TaskAlignedAssigner` decides which anchors answer for which ground-truth box from the
//  detached predictions: an anchor is a candidate when its center falls inside the box (a box smaller
//  than the first stride grows to the second), each box keeps its ten candidates with the highest
//  alignment `score^0.5 · CIoU^6`, an anchor claimed by several boxes goes to the one it overlaps most,
//  and the class target is scaled by the anchor's alignment relative to its box's best. The loss is
//  binary cross-entropy on the class logits, `1 − CIoU` and the distribution focal loss on the assigned
//  anchors' boxes, weighted by those targets, gained 0.5, 7.5, and 1.5, and scaled by the batch size.
//
//  The assignment carries no gradient in the reference (`torch.no_grad`), so it runs on the host here
//  from evaluated arrays; the decode, CIoU, DFL, and cross-entropy stay in MLX so the gradient reaches
//  the head.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/*!
 @abstract A ground-truth box for a YOLO fine-tune: a class index and corners in input pixels.
 @discussion Introduced in InferKit 0.5.0.
 */
public struct NFKMLXYOLOBox: Sendable, Equatable {
    public var classIndex: Int
    public var x1: Float
    public var y1: Float
    public var x2: Float
    public var y2: Float

    public init(classIndex: Int, x1: Float, y1: Float, x2: Float, y2: Float) {
        self.classIndex = classIndex
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    }
}

/*!
 @abstract The objective a YOLO fine-tune minimizes: ultralytics' `v8DetectionLoss`.
 @discussion Reads the head's raw outputs over every anchor of every scale, the box distributions
 `[batch, anchors, 4 · regMax]` and the class logits `[batch, anchors, classes]`, with the anchors in
 the head's order (scale by scale, row-major within a scale). Measured against ultralytics by
 `run_reference.py yolo_loss`. Introduced in InferKit 0.5.0.
 */
public struct NFKMLXYOLOObjective: Sendable {
    public var boxGain: Float = 7.5
    public var classGain: Float = 0.5
    public var dflGain: Float = 1.5
    /// The candidates each box keeps, `tal_topk`.
    public var topK = 10
    /// A second, stricter cut after the overlaps are resolved; the end-to-end heads' one-to-one branch
    /// keeps one. Nil keeps `topK`.
    public var secondTopK: Int?
    public var alpha: Float = 0.5
    public var beta: Float = 6

    public init() {}

    /// The anchor centers in grid units and each anchor's stride, `[anchors, 2]` and `[anchors, 1]`:
    /// `make_anchors` with the half-cell offset.
    public static func anchors(featureSizes: [(height: Int, width: Int)], strides: [Int]) -> (points: MLXArray, strides: MLXArray) {
        var points = [Float](), perAnchor = [Float]()
        for (size, stride) in zip(featureSizes, strides) {
            for y in 0 ..< size.height {
                for x in 0 ..< size.width {
                    points += [Float(x) + 0.5, Float(y) + 0.5]
                    perAnchor.append(Float(stride))
                }
            }
        }
        return (MLXArray(points).reshaped([-1, 2]), MLXArray(perAnchor).reshaped([-1, 1]))
    }

    /// The three weighted terms, each a scalar before the batch scaling: box (`1 − CIoU`), class (BCE),
    /// and DFL, which is an L1 on the normalized side distances when the head predicts them directly
    /// (`regMax` 1, YOLO26).
    public func components(boxDistribution: MLXArray, classLogits: MLXArray,
                           featureSizes: [(height: Int, width: Int)], strides: [Int],
                           targets: [[NFKMLXYOLOBox]]) -> (box: MLXArray, classification: MLXArray, dfl: MLXArray) {
        let (batch, anchorCount, classes) = (classLogits.dim(0), classLogits.dim(1), classLogits.dim(2))
        let regMax = boxDistribution.dim(2) / 4
        let (points, anchorStrides) = Self.anchors(featureSizes: featureSizes, strides: strides)

        // Decode: the expected bin of each side's distribution, then corners around the anchor.
        let bins = MLXArray((0 ..< regMax).map { Float($0) })
        let distances = regMax > 1
            ? (softmax(boxDistribution.reshaped([batch, anchorCount, 4, regMax]), axis: -1) * bins).sum(axis: -1)
            : boxDistribution
        let predicted = concatenated([points - distances[0..., 0..., 0 ..< 2], points + distances[0..., 0..., 2 ..< 4]], axis: -1)

        // The assignment, from detached values.
        let scores = sigmoid(classLogits)
        let pixelBoxes = predicted * anchorStrides
        eval(scores, pixelBoxes, points, anchorStrides)
        let assignment = NFKYOLOTaskAlignedAssigner(topK: topK, secondTopK: secondTopK, alpha: alpha, beta: beta,
                                                   strides: strides, classes: classes)
            .assign(scores: scores.asArray(Float.self), boxes: pixelBoxes.asArray(Float.self),
                    anchors: (points * anchorStrides).asArray(Float.self), batch: batch, anchorCount: anchorCount,
                    targets: targets)

        let targetScores = MLXArray(assignment.scores).reshaped([batch, anchorCount, classes])
        let scoreSum = max(assignment.scores.reduce(0, +), 1)
        let bce = maximum(classLogits, 0) - classLogits * targetScores + log1p(exp(-abs(classLogits)))
        let classification = bce.sum() / scoreSum

        guard !assignment.foreground.isEmpty else {
            let zero = MLXArray(Float(0))
            return (zero, classGain * classification, zero)
        }
        let foreground = MLXArray(assignment.foreground.map { Int32($0) })
        let weight = MLXArray(assignment.foregroundWeights).reshaped([-1, 1])
        let flatStrides = broadcast(anchorStrides.reshaped([1, anchorCount, 1]), to: [batch, anchorCount, 1])
            .reshaped([batch * anchorCount, 1])
        let flatPoints = broadcast(points.reshaped([1, anchorCount, 2]), to: [batch, anchorCount, 2])
            .reshaped([batch * anchorCount, 2])
        let chosenStride = take(flatStrides, foreground, axis: 0)
        let targetBoxes = MLXArray(assignment.foregroundBoxes).reshaped([-1, 4]) / chosenStride
        let predictedBoxes = take(predicted.reshaped([batch * anchorCount, 4]), foreground, axis: 0)
        let iou = Self.completeIoU(predictedBoxes, targetBoxes)
        let box = ((1 - iou) * weight).sum() / scoreSum

        var dfl = MLXArray(Float(0))
        if regMax == 1 {
            // Without a distribution the third term is an L1 on the side distances, in pixels normalized
            // by the input's width and height.
            let anchorsChosen = take(flatPoints, foreground, axis: 0)
            let target = concatenated([anchorsChosen - targetBoxes[0..., 0 ..< 2], targetBoxes[0..., 2 ..< 4] - anchorsChosen], axis: -1)
            let predictedSides = take(boxDistribution.reshaped([batch * anchorCount, 4]), foreground, axis: 0)
            let size = (height: Float(featureSizes[0].height * strides[0]), width: Float(featureSizes[0].width * strides[0]))
            let normalizer = MLXArray([size.width, size.height, size.width, size.height]).reshaped([1, 4])
            let error = abs((predictedSides - target) * chosenStride / normalizer).mean(axis: -1, keepDims: true)
            dfl = (error * weight).sum() / scoreSum
        } else {
            let anchorsChosen = take(flatPoints, foreground, axis: 0)
            let sides = concatenated([anchorsChosen - targetBoxes[0..., 0 ..< 2], targetBoxes[0..., 2 ..< 4] - anchorsChosen], axis: -1)
            let target = clip(clip(sides, min: 0, max: Float(regMax - 1) - 0.01), min: 0, max: Float(regMax) - 1 - 0.01)
            let left = floor(target)
            let rightWeight = target - left
            let leftWeight = 1 - rightWeight
            let logits = take(boxDistribution.reshaped([batch * anchorCount, 4, regMax]), foreground, axis: 0)
            let logProbabilities = logits - logSumExp(logits, axis: -1, keepDims: true)
            let leftIndex = left.asType(.int32).expandedDimensions(axis: -1)
            let pickedLeft = takeAlong(logProbabilities, leftIndex, axis: -1).squeezed(axis: -1)
            let pickedRight = takeAlong(logProbabilities, leftIndex + 1, axis: -1).squeezed(axis: -1)
            let perAnchor = -(pickedLeft * leftWeight + pickedRight * rightWeight).mean(axis: -1, keepDims: true)
            dfl = (perAnchor * weight).sum() / scoreSum
        }
        return (boxGain * box, classGain * classification, dflGain * dfl)
    }

    /// The loss the reference backpropagates: the three terms summed and scaled by the batch size.
    public func loss(boxDistribution: MLXArray, classLogits: MLXArray,
                     featureSizes: [(height: Int, width: Int)], strides: [Int], targets: [[NFKMLXYOLOBox]]) -> MLXArray {
        let terms = components(boxDistribution: boxDistribution, classLogits: classLogits,
                               featureSizes: featureSizes, strides: strides, targets: targets)
        return (terms.box + terms.classification + terms.dfl) * Float(classLogits.dim(0))
    }

    /// `bbox_iou(..., xywh=False, CIoU=True)` over corner boxes `[n, 4]`, with the aspect term's
    /// weight held out of the gradient as the reference holds it.
    static func completeIoU(_ first: MLXArray, _ second: MLXArray) -> MLXArray {
        let eps: Float = 1e-7
        let (ax1, ay1, ax2, ay2) = (first[0..., 0], first[0..., 1], first[0..., 2], first[0..., 3])
        let (bx1, by1, bx2, by2) = (second[0..., 0], second[0..., 1], second[0..., 2], second[0..., 3])
        let (aw, ah) = (ax2 - ax1, ay2 - ay1 + eps)
        let (bw, bh) = (bx2 - bx1, by2 - by1 + eps)
        let intersection = clip(minimum(ax2, bx2) - maximum(ax1, bx1), min: 0)
            * clip(minimum(ay2, by2) - maximum(ay1, by1), min: 0)
        let union = aw * ah + bw * bh - intersection + eps
        let iou = intersection / union
        let enclosingWidth = maximum(ax2, bx2) - minimum(ax1, bx1)
        let enclosingHeight = maximum(ay2, by2) - minimum(ay1, by1)
        let diagonal = enclosingWidth.square() + enclosingHeight.square() + eps
        let centerDistance = ((bx1 + bx2 - ax1 - ax2).square() + (by1 + by2 - ay1 - ay2).square()) / 4
        let aspect = Float(4 / (Double.pi * Double.pi)) * (atan(bw / bh) - atan(aw / ah)).square()
        let aspectWeight = stopGradient(aspect / (aspect - iou + (1 + eps)))
        return (iou - (centerDistance / diagonal + aspect * aspectWeight)).reshaped([-1, 1])
    }
}

/*!
 @abstract The objective an end-to-end YOLO (v10, YOLO26) fine-tune minimizes: ultralytics' `E2ELoss`.
 @discussion The one-to-many branch trains under `v8DetectionLoss` with ten candidates per box, and the
 one-to-one branch, which serves inference without suppression, with seven candidates cut to the single
 best. The one-to-many weight starts at 0.8 and falls linearly over the run to 0.1 by epoch, and the
 one-to-one branch takes the rest. Measured against ultralytics by `run_reference.py yolo_e2e_loss`.
 Introduced in InferKit 0.5.0.
 */
public struct NFKMLXYOLOEndToEndObjective: Sendable {
    public var oneToMany: NFKMLXYOLOObjective
    public var oneToOne: NFKMLXYOLOObjective
    public var initialOneToManyWeight: Float = 0.8
    public var finalOneToManyWeight: Float = 0.1

    public init() {
        oneToMany = NFKMLXYOLOObjective()
        oneToOne = NFKMLXYOLOObjective()
        oneToOne.topK = 7
        oneToOne.secondTopK = 1
    }

    /// `E2ELoss.decay`: the one-to-many weight during `epoch` of a run of `epochs`.
    public func oneToManyWeight(epoch: Int, epochs: Int) -> Float {
        max(1 - Float(epoch) / Float(max(epochs - 1, 1)), 0) * (initialOneToManyWeight - finalOneToManyWeight)
            + finalOneToManyWeight
    }

    /// The weighted sum of the two branches' batch-scaled losses.
    public func loss(oneToMany many: (distribution: MLXArray, logits: MLXArray),
                     oneToOne one: (distribution: MLXArray, logits: MLXArray),
                     featureSizes: [(height: Int, width: Int)], strides: [Int], targets: [[NFKMLXYOLOBox]],
                     epoch: Int = 0, epochs: Int = 1) -> MLXArray {
        let weight = oneToManyWeight(epoch: epoch, epochs: epochs)
        let manyLoss = oneToMany.loss(boxDistribution: many.distribution, classLogits: many.logits,
                                      featureSizes: featureSizes, strides: strides, targets: targets)
        let oneLoss = oneToOne.loss(boxDistribution: one.distribution, classLogits: one.logits,
                                    featureSizes: featureSizes, strides: strides, targets: targets)
        return manyLoss * weight + oneLoss * max(1 - weight, 0)
    }
}

/// ultralytics' `TaskAlignedAssigner`, over host arrays. Ties in a top-k keep the lower anchor index.
struct NFKYOLOTaskAlignedAssigner {
    let topK: Int
    let secondTopK: Int?
    let alpha: Float
    let beta: Float
    let strides: [Int]
    let classes: Int

    struct Result {
        /// Class targets `[batch · anchors · classes]`, already scaled by the normalized alignment.
        var scores: [Float]
        /// Flat `batch · anchors` indices of the assigned anchors.
        var foreground: [Int]
        /// Each assigned anchor's box in pixels, `[foreground · 4]`.
        var foregroundBoxes: [Float]
        /// Each assigned anchor's total class target, the weight of its box terms.
        var foregroundWeights: [Float]
    }

    func assign(scores: [Float], boxes: [Float], anchors: [Float], batch: Int, anchorCount: Int,
                targets: [[NFKMLXYOLOBox]]) -> Result {
        var result = Result(scores: [Float](repeating: 0, count: batch * anchorCount * classes),
                            foreground: [], foregroundBoxes: [], foregroundWeights: [])
        let boxCount = targets.map(\.count).max() ?? 0
        guard boxCount > 0 else {
            return result
        }
        for image in 0 ..< batch {
            let gts = targets[image]
            // Padded boxes are all zeros and never candidates; they still take part in the argmaxes.
            let padded = gts.map { [$0.x1, $0.y1, $0.x2, $0.y2] } + [[Float]](repeating: [0, 0, 0, 0], count: boxCount - gts.count)
            let valid = (0 ..< boxCount).map { $0 < gts.count && padded[$0].reduce(0, +) > 0 }
            var overlaps = [[Float]](repeating: [Float](repeating: 0, count: anchorCount), count: boxCount)
            var align = overlaps
            var inside = [[Bool]](repeating: [Bool](repeating: false, count: anchorCount), count: boxCount)
            for n in 0 ..< boxCount where valid[n] {
                let region = expanded(padded[n])
                for a in 0 ..< anchorCount {
                    let x = anchors[2 * a], y = anchors[2 * a + 1]
                    let eps: Float = 1e-9
                    guard x - region[0] > eps, y - region[1] > eps, region[2] - x > eps, region[3] - y > eps else {
                        continue
                    }
                    inside[n][a] = true
                    let offset = image * anchorCount + a
                    let predicted = Array(boxes[(offset * 4) ..< (offset * 4 + 4)])
                    let overlap = max(Self.completeIoU(padded[n], predicted), 0)
                    let score = scores[offset * classes + gts[n].classIndex]
                    overlaps[n][a] = overlap
                    align[n][a] = pow(score, alpha) * pow(overlap, beta)
                }
            }
            // The top-k by alignment over every anchor, then only the candidates inside the box.
            var positive = [[Float]](repeating: [Float](repeating: 0, count: anchorCount), count: boxCount)
            for n in 0 ..< boxCount where valid[n] {
                for a in Self.topIndices(align[n], count: topK) where inside[n][a] {
                    positive[n][a] = 1
                }
            }
            // An anchor claimed by several boxes goes to the box it overlaps most.
            for a in 0 ..< anchorCount where (0 ..< boxCount).map({ positive[$0][a] }).reduce(0, +) > 1 {
                let best = Self.firstArgmax((0 ..< boxCount).map { overlaps[$0][a] })
                for n in 0 ..< boxCount {
                    positive[n][a] = n == best ? 1 : 0
                }
            }
            if let secondTopK, secondTopK != topK {
                for n in 0 ..< boxCount {
                    let kept = Set(Self.topIndices((0 ..< anchorCount).map { align[n][$0] * positive[n][$0] }, count: secondTopK))
                    for a in 0 ..< anchorCount where !kept.contains(a) {
                        positive[n][a] = 0
                    }
                }
            }
            // The class target, scaled by each anchor's alignment relative to its box's best.
            let bestAlign = (0 ..< boxCount).map { n in (0 ..< anchorCount).map { align[n][$0] * positive[n][$0] }.max() ?? 0 }
            let bestOverlap = (0 ..< boxCount).map { n in (0 ..< anchorCount).map { overlaps[n][$0] * positive[n][$0] }.max() ?? 0 }
            for a in 0 ..< anchorCount {
                let column = (0 ..< boxCount).map { positive[$0][a] }
                guard column.reduce(0, +) > 0 else {
                    continue
                }
                let owner = Self.firstArgmax(column)
                let scale = (0 ..< boxCount).map { n in align[n][a] * positive[n][a] * bestOverlap[n] / (bestAlign[n] + 1e-9) }.max() ?? 0
                let flat = image * anchorCount + a
                result.scores[flat * classes + gts[owner].classIndex] = scale
                result.foreground.append(flat)
                result.foregroundBoxes += padded[owner]
                result.foregroundWeights.append(scale)
            }
        }
        return result
    }

    /// A box narrower or shorter than the first stride grows to the second along that side, about its center.
    private func expanded(_ box: [Float]) -> [Float] {
        let small = Float(strides[0]), grown = Float(strides.count > 1 ? strides[1] : strides[0])
        let (cx, cy) = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        var (w, h) = (box[2] - box[0], box[3] - box[1])
        if w < small {
            w = grown
        }
        if h < small {
            h = grown
        }
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    }

    static func topIndices(_ values: [Float], count: Int) -> [Int] {
        Array(values.indices.sorted { values[$0] != values[$1] ? values[$0] > values[$1] : $0 < $1 }.prefix(count))
    }

    static func firstArgmax(_ values: [Float]) -> Int {
        var best = 0
        for index in values.indices where values[index] > values[best] {
            best = index
        }
        return best
    }

    /// The reference's CIoU over two corner boxes, as floats.
    static func completeIoU(_ a: [Float], _ b: [Float]) -> Float {
        let eps: Float = 1e-7
        let (aw, ah) = (a[2] - a[0], a[3] - a[1] + eps)
        let (bw, bh) = (b[2] - b[0], b[3] - b[1] + eps)
        let intersection = max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)
        let union = aw * ah + bw * bh - intersection + eps
        let iou = intersection / union
        let cw = max(a[2], b[2]) - min(a[0], b[0]), ch = max(a[3], b[3]) - min(a[1], b[1])
        let diagonal = cw * cw + ch * ch + eps
        let dx = b[0] + b[2] - a[0] - a[2], dy = b[1] + b[3] - a[1] - a[3]
        let centerDistance = (dx * dx + dy * dy) / 4
        let v = Float(4 / (Double.pi * Double.pi)) * powf(atanf(bw / bh) - atanf(aw / ah), 2)
        let weight = v / (v - iou + (1 + eps))
        return iou - (centerDistance / diagonal + v * weight)
    }
}

/// Which parameters a YOLO fine-tune updates.
public enum NFKMLXYOLOTrainable: Sendable {

    /// Every parameter, ultralytics' default (`freeze=None`). The distribution-focal expectation is
    /// fixed either way, as the reference always freezes `.dfl`.
    case everything

    /// The detection head alone, with the backbone and neck frozen: the reference's `freeze=22`.
    case head
}

extension NFKMLXLearningRateSchedule {

    /// ultralytics' default schedule over a run of `steps` updates in epochs of `stepsPerEpoch`: the
    /// linear `lf(e) = max(1 − e / epochs, 0) · (1 − finalScale) + finalScale` stepped once per epoch, and
    /// over the first `round(min(warmupEpochs, epochs − 1) · stepsPerEpoch)` updates a warm-up that
    /// interpolates from 0 to it. The warm-up is the reference's for AdamW, whose bias rate starts at 0.
    ///
    /// Introduced in InferKit 0.5.0.
    public static func ultralytics(steps: Int, stepsPerEpoch: Int, warmupEpochs: Double = 3,
                                   finalScale: Double = 0.01) -> NFKMLXLearningRateSchedule {
        let perEpoch = max(stepsPerEpoch, 1)
        let epochs = (steps + perEpoch - 1) / perEpoch
        let warmup = Int((min(warmupEpochs, Double(max(epochs - 1, 0))) * Double(perEpoch)).rounded(.toNearestOrEven))
        return NFKMLXLearningRateSchedule { step in
            let epoch = step / perEpoch
            let scale = max(1 - Double(epoch) / Double(max(epochs, 1)), 0) * (1 - finalScale) + finalScale
            guard step < warmup else {
                return Float(scale)
            }
            return Float(Double(step) / Double(warmup) * scale)
        }
    }
}

extension NFKMLXYOLO {

    /// Builds the YOLOv8 network at a released size for `classCount` classes, ready to fine-tune.
    ///
    /// The head's biases start at ultralytics' `bias_init` values, and `weightsURL` (a release or a file
    /// `NFKMLXWeights.save` wrote) transfers every tensor whose shape matches, as ultralytics'
    /// `intersect_dicts` does: at a class count other than the checkpoint's, the class branches keep
    /// their fresh initialization and everything else loads.
    ///
    /// - Since: InferKit 0.5.0
    public static func network(variant: NFKMLXYOLOVariant = .nano, classCount: Int = 80,
                               weightsURL: URL?) throws -> NFKMLXYOLONet {
        var configuration = geometry(variant)
        configuration.classCount = classCount
        let net = NFKMLXYOLONet(configuration)
        net.initializeHeadBiases()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL, matchingShapesOnly: true)
        }
        return net
    }

    /// ultralytics' `optimizer=auto` for a run under 10,000 updates: AdamW at
    /// `round(0.002 · 5 / (4 + classes), 6)`, betas 0.9 and 0.999, with the weight decay
    /// `0.0005 · batch · accumulate / 64` on the convolution weights and none on biases or
    /// normalization weights.
    static func referenceOptimizer(classCount: Int, batchSize: Int) -> Optimizer {
        let rate = (0.002 * 5 / Double(4 + classCount) * 1e6).rounded(.toNearestOrEven) / 1e6
        let accumulate = max(Int((64 / Double(max(batchSize, 1))).rounded(.toNearestOrEven)), 1)
        let decay = 0.0005 * Double(batchSize * accumulate) / 64
        return NFKMLXReferenceOptimizers.adamW(learningRate: Float(rate), weightDecay: Float(decay)) { key in
            key.contains("bias") || key.hasSuffix(".bn.weight")
        }
    }

    /// Fine-tunes a YOLOv8 network on a consumer's own labeled images, returning the loss from each step.
    ///
    /// The whole path is three calls: ``network(variant:classCount:weightsURL:)`` to build, this to
    /// train, and `NFKMLXWeights.save` to write a checkpoint that `backendWithVariant:weightsURL:labels:error:`
    /// loads at its own class count.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(variant:classCount:weightsURL:)``.
    ///   - examples: supplies one batch per step: images `[batch, H, W, 3]` in `0...1` with sides a
    ///     multiple of 32, and each image's boxes in its own pixels. Augmentation is the caller's; the
    ///     reference trains on mosaics with color and geometric jitter.
    ///   - trainable: every parameter (the reference's default) or the head alone.
    ///   - objective: ultralytics' `v8DetectionLoss`.
    ///   - optimizer: the update rule. Nil uses ultralytics' `optimizer=auto` choice for a short run,
    ///     AdamW with its decay groups, sized from the first batch.
    ///   - steps: how many batches to train on.
    ///   - stepsPerEpoch: the batches in one pass over the consumer's data, which the reference's
    ///     schedule counts in.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference clips at 10.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses ``NFKMLXLearningRateSchedule/ultralytics(steps:stepsPerEpoch:warmupEpochs:finalScale:)``
    ///     when the reference optimizer runs.
    ///   - averagesWeights: keeps ultralytics' exponential moving average of every weight and running
    ///     statistic (`ModelEMA`, decay `0.9999 · (1 − e^(−updates / 2000))`) and leaves it in `net` at the
    ///     end, which is what the reference saves.
    ///   - checkpoint: writes the network periodically, before the average is applied.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is minutes; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXYOLONet,
        examples: (Int) -> (images: MLXArray, targets: [[NFKMLXYOLOBox]]),
        trainable: NFKMLXYOLOTrainable = .everything,
        objective: NFKMLXYOLOObjective = NFKMLXYOLOObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        stepsPerEpoch: Int,
        clipGradientNorm: Float? = 10,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        averagesWeights: Bool = true,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        var average = averagesWeights ? NFKYOLOWeightAverage(net) : nil
        let history = try NFKMLXFineTune.run(
            net,
            freezing: {
                switch trainable {
                case .everything:
                    net.unfreeze()
                case .head:
                    net.freeze()
                    net.detect.unfreeze()
                }
                net.detect.dfl.freeze()
                // A parent's unfreeze clears the batch normalizations' own freeze of their statistics.
                for (_, module) in net.leafModules().flattened() {
                    if let norm = module as? BatchNorm {
                        norm.freeze(recursive: false, keys: ["running_mean", "running_var"])
                    }
                }
            },
            optimizer: optimizer,
            reference: {
                referenceOptimizer(classCount: net.configuration.classCount, batchSize: examples(0).images.dim(0))
            },
            referenceSchedule: { .ultralytics(steps: steps, stepsPerEpoch: stepsPerEpoch) },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                let rows = example.targets.enumerated().flatMap { image, boxes in
                    boxes.flatMap { [Float(image), Float($0.classIndex), $0.x1, $0.y1, $0.x2, $0.y2] }
                }
                return [example.images, MLXArray(rows).reshaped([-1, 6])]
            },
            loss: { net, arrays in
                let outputs = net.headOutputs(arrays[0])
                let rows = arrays[1].asArray(Float.self)
                var targets = [[NFKMLXYOLOBox]](repeating: [], count: arrays[0].dim(0))
                for row in stride(from: 0, to: rows.count, by: 6) {
                    targets[Int(rows[row])].append(NFKMLXYOLOBox(classIndex: Int(rows[row + 1]), x1: rows[row + 2],
                                                                 y1: rows[row + 3], x2: rows[row + 4], y2: rows[row + 5]))
                }
                return objective.loss(boxDistribution: outputs.distribution, classLogits: outputs.logits,
                                      featureSizes: outputs.featureSizes, strides: outputs.strides, targets: targets)
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

/// ultralytics' `ModelEMA`: a moving average of every floating weight and running statistic, decayed by
/// `0.9999 · (1 − e^(−updates / 2000))` so the early updates count for more.
struct NFKYOLOWeightAverage {
    private var shadow: [String: MLXArray]
    private var updates = 0

    init(_ net: Module) {
        shadow = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 * 1) })
    }

    mutating func update(from net: Module) {
        updates += 1
        let decay = Float(0.9999 * (1 - exp(-Double(updates) / 2000)))
        for (key, value) in net.parameters().flattened() {
            if let previous = shadow[key] {
                shadow[key] = previous * decay + value * (1 - decay)
            }
        }
        eval(Array(shadow.values))
    }

    func apply(to net: Module) {
        net.update(parameters: ModuleParameters.unflattened(shadow.map { ($0.key, $0.value) }))
    }
}
