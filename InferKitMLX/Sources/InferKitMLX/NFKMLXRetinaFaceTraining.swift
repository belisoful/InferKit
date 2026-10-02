//
//  NFKMLXRetinaFaceTraining.swift
//  InferKitMLX
//
//  RetinaFace's training path, ported from biubug6/Pytorch_Retinaface at b984b4b: the prior matching of
//  `utils/box_utils.py`, `layers/modules/multibox_loss.py`, and `train.py`'s SGD and step schedule.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// A face annotated for a RetinaFace fine-tune: its box corners and, when annotated, its five
/// landmarks, every coordinate a fraction of the image's width or height.
///
/// @discussion A face without landmarks trains the box and the class and not the landmark head, as the
/// reference's label of −1 marks it. Introduced in InferKit 0.4.0.
public struct NFKMLXRetinaFaceAnnotation: Sendable, Equatable {
    public var x1: Float
    public var y1: Float
    public var x2: Float
    public var y2: Float
    /// Five `(x, y)` pairs in the order the detector reports them: the two eyes, the nose, and the two
    /// mouth corners. Nil when the face carries none.
    public var landmarks: [Float]?

    public init(x1: Float, y1: Float, x2: Float, y2: Float, landmarks: [Float]? = nil) {
        precondition(landmarks == nil || landmarks!.count == 10, "five landmarks are ten coordinates")
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.landmarks = landmarks
    }
}

/// The objective a RetinaFace fine-tune minimizes: `MultiBoxLoss` with `train.py`'s settings.
///
/// @discussion Each prior takes the face it overlaps most; a prior under `overlapThreshold` is
/// background, and every face keeps its own best prior unless that prior overlaps it under 0.2, in
/// which case the face is ignored. The loss is smooth L1 on the encoded boxes of the matched priors,
/// cross-entropy on those priors and on `negativeRatio` times as many background priors of the highest
/// loss, each divided by the matched count, and smooth L1 on the landmarks of the matched faces that
/// carry them, divided by their count. The box term is weighted by `locationWeight`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXRetinaFaceObjective: Sendable {
    public var overlapThreshold: Float = 0.35
    public var negativeRatio: Int = 7
    public var locationWeight: Float = 2

    public init() {}

    /// Scores the network on prepared images `[N, H, W, 3]` against each image's faces.
    public func callAsFunction(_ net: NFKMLXRetinaFaceNet, _ images: MLXArray,
                               _ faces: [[NFKMLXRetinaFaceAnnotation]]) -> MLXArray {
        let outputs = net.logits(images)
        let priors = NFKMLXRetinaFace.anchors(height: images.dim(1), width: images.dim(2),
                                              configuration: net.configuration)
        let terms = loss(boxes: outputs.boxes, logits: outputs.logits, landmarks: outputs.landmarks,
                         priors: priors, faces: faces, variance: net.configuration.variance)
        return locationWeight * terms.location + terms.confidence + terms.landmark
    }

    /// The three terms for the network's training outputs over centre-form `priors` `[P, 4]`.
    public func loss(boxes: MLXArray, logits: MLXArray, landmarks: MLXArray, priors: MLXArray,
                     faces: [[NFKMLXRetinaFaceAnnotation]], variance: [Float])
        -> (location: MLXArray, confidence: MLXArray, landmark: MLXArray) {
        loss(boxes: boxes, logits: logits, landmarks: landmarks, priors: priors, faces: faces, variance: variance,
             selected: nil)
    }

    /// The three terms with the classification term over `selected` `[N, P]` in place of the mined
    /// priors when one is given.
    func loss(boxes: MLXArray, logits: MLXArray, landmarks: MLXArray, priors: MLXArray,
              faces: [[NFKMLXRetinaFaceAnnotation]], variance: [Float], selected: MLXArray?)
        -> (location: MLXArray, confidence: MLXArray, landmark: MLXArray) {
        let count = priors.dim(0)
        let priorValues = priors.asArray(Float.self)
        var locationTarget = [Float](), landmarkTarget = [Float](), labels = [Int32]()
        for image in faces {
            let matched = match(image, priors: priorValues, variance: variance)
            locationTarget += matched.location
            landmarkTarget += matched.landmarks
            labels += matched.labels
        }
        let batch = faces.count
        let label = MLXArray(labels, [batch, count])
        let location = MLXArray(locationTarget, [batch, count, 4])
        let landmark = MLXArray(landmarkTarget, [batch, count, 10])

        let withLandmarks = (label .> 0).asType(.float32)
        let landmarkCount = max(withLandmarks.sum().item(Float.self), 1)
        let landmarkLoss = (Self.smoothL1(landmarks - landmark) * withLandmarks.expandedDimensions(axis: -1)).sum()
            / landmarkCount

        let matched = (label .!= 0)
        let positive = matched.asType(.float32)
        let positiveCount = max(positive.sum().item(Float.self), 1)
        let locationLoss = (Self.smoothL1(boxes - location) * positive.expandedDimensions(axis: -1)).sum()
            / positiveCount

        let perPrior = Self.classificationLoss(logits, matched: matched)
        let confidenceLoss = (perPrior * (selected ?? selection(perPrior, matched: matched))).sum() / positiveCount
        return (locationLoss, confidenceLoss, landmarkLoss)
    }

    /// Each prior's cross-entropy against its target, face for a matched prior and background otherwise.
    static func classificationLoss(_ logits: MLXArray, matched: MLXArray) -> MLXArray {
        let target = matched.asType(.int32).expandedDimensions(axis: -1)
        return logSumExp(logits, axis: -1) - takeAlong(logits, target, axis: -1).squeezed(axis: -1)
    }

    /// The priors the classification term scores: the matched ones and, by hard negative mining, the
    /// `negativeRatio` times as many background priors of the highest loss in each image.
    func selection(_ perPrior: MLXArray, matched: MLXArray) -> MLXArray {
        let mining = MLX.where(matched, MLXArray(Float(0)), stopGradient(perPrior))
        let rank = argSort(argSort(-mining, axis: 1), axis: 1)
        let negatives = minimum(matched.asType(.float32).sum(axis: 1, keepDims: true) * Float(negativeRatio),
                                Float(perPrior.dim(1) - 1))
        return logicalOr(matched, rank.asType(.float32) .< negatives).asType(.float32)
    }

    /// `F.smooth_l1_loss` at β = 1, elementwise.
    static func smoothL1(_ difference: MLXArray) -> MLXArray {
        let magnitude = abs(difference)
        return MLX.where(magnitude .< 1, 0.5 * square(difference), magnitude - 0.5)
    }

    /// `utils.box_utils.match` for one image: per prior, the encoded box and landmarks of its face and
    /// the face's label (1 with landmarks, −1 without, 0 for background).
    func match(_ faces: [NFKMLXRetinaFaceAnnotation], priors: [Float], variance: [Float])
        -> (location: [Float], landmarks: [Float], labels: [Int32]) {
        let count = priors.count / 4
        let background = ([Float](repeating: 0, count: count * 4), [Float](repeating: 0, count: count * 10),
                          [Int32](repeating: 0, count: count))
        guard !faces.isEmpty else { return background }
        func overlap(_ face: NFKMLXRetinaFaceAnnotation, _ prior: Int) -> Float {
            let (cx, cy, w, h) = (priors[prior * 4], priors[prior * 4 + 1], priors[prior * 4 + 2], priors[prior * 4 + 3])
            let (px1, py1, px2, py2) = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            let width = max(min(face.x2, px2) - max(face.x1, px1), 0)
            let height = max(min(face.y2, py2) - max(face.y1, py1), 0)
            let intersection = width * height
            let union = (face.x2 - face.x1) * (face.y2 - face.y1) + (px2 - px1) * (py2 - py1) - intersection
            return intersection / union
        }
        var bestPrior = [Int](repeating: 0, count: faces.count)
        var bestPriorOverlap = [Float](repeating: -1, count: faces.count)
        var bestFace = [Int](repeating: 0, count: count)
        var bestFaceOverlap = [Float](repeating: -1, count: count)
        for prior in 0 ..< count {
            for (index, face) in faces.enumerated() {
                let value = overlap(face, prior)
                if value > bestPriorOverlap[index] {
                    bestPriorOverlap[index] = value
                    bestPrior[index] = prior
                }
                if value > bestFaceOverlap[prior] {
                    bestFaceOverlap[prior] = value
                    bestFace[prior] = index
                }
            }
        }
        // A face whose best prior overlaps it under 0.2 is too hard to match; with none left the image
        // is background throughout.
        let valid = faces.indices.filter { bestPriorOverlap[$0] >= 0.2 }
        guard !valid.isEmpty else { return background }
        for index in valid {
            bestFaceOverlap[bestPrior[index]] = 2
        }
        // Every face, valid or not, claims its best prior, as the reference's loop writes it.
        for index in faces.indices {
            bestFace[bestPrior[index]] = index
        }
        var location = [Float](), landmarks = [Float](), labels = [Int32]()
        location.reserveCapacity(count * 4)
        landmarks.reserveCapacity(count * 10)
        for prior in 0 ..< count {
            let face = faces[bestFace[prior]]
            let (cx, cy, w, h) = (priors[prior * 4], priors[prior * 4 + 1], priors[prior * 4 + 2], priors[prior * 4 + 3])
            location += [((face.x1 + face.x2) / 2 - cx) / (variance[0] * w),
                         ((face.y1 + face.y2) / 2 - cy) / (variance[0] * h),
                         log((face.x2 - face.x1) / w) / variance[1],
                         log((face.y2 - face.y1) / h) / variance[1]]
            let points = face.landmarks ?? [Float](repeating: -1, count: 10)
            for point in 0 ..< 5 {
                landmarks += [(points[point * 2] - cx) / (variance[0] * w),
                              (points[point * 2 + 1] - cy) / (variance[0] * h)]
            }
            let label: Int32 = face.landmarks == nil ? -1 : 1
            labels.append(bestFaceOverlap[prior] < overlapThreshold ? 0 : label)
        }
        return (location, landmarks, labels)
    }
}

public extension NFKMLXRetinaFace {

    /// `cfg_mnet`'s batch, 32 images. Introduced in InferKit 0.4.0.
    static let referenceBatchSize = 32

    /// `cfg_mnet`'s training size, 640 pixels square. Introduced in InferKit 0.4.0.
    static let referenceImageSize = 640

    /// Builds RetinaFace for training or for reloading a trained checkpoint. With a `weightsURL` the
    /// released checkpoint or a file `NFKMLXWeights` saved loads; without one the network is randomly
    /// initialized. Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?) throws -> NFKMLXRetinaFaceNet {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// The network's input for RGB images `[N, H, W, 3]` in [0, 1]: BGR in 0…255 less the reference's
    /// channel mean, as `preproc` leaves it. Introduced in InferKit 0.4.0.
    static func trainingInput(_ images: MLXArray) -> MLXArray {
        let rgb = images * 255
        let bgr = concatenated([rgb[.ellipsis, 2 ..< 3], rgb[.ellipsis, 1 ..< 2], rgb[.ellipsis, 0 ..< 1]], axis: -1)
        return bgr - MLXArray(channelMean)
    }

    /// The optimizer `train.py` builds: `torch.optim.SGD` at 1e-3, momentum 0.9, and a weight decay of
    /// 5e-4 on every parameter. Introduced in InferKit 0.4.0.
    static func referenceOptimizer() -> Optimizer {
        NFKMLXSGD(learningRate: 1e-3, momentum: 0.9, weightDecay: 5e-4)
    }

    /// `train.py`'s schedule over a run of `steps`: the rate falls tenfold at epochs 190 and 220 of 250,
    /// placed at those fractions of the run and never at its first step. Introduced in InferKit 0.4.0.
    static func referenceSchedule(steps: Int) -> NFKMLXLearningRateSchedule {
        .multiStep(milestones: [max(steps * 190 / 250, 1), max(steps * 220 / 250, 1)], gamma: 0.1)
    }

    /// Trains RetinaFace on images and the faces in them.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:)``.
    ///   - examples: supplies one batch per step: RGB images `[N, H, W, 3]` in [0, 1] and each image's
    ///     faces. The reference trains on 640-pixel squares (``referenceImageSize``) after a random
    ///     crop, color distortion, and a mirror, which are the caller's to apply.
    ///   - objective: the loss.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer()``.
    ///   - steps: how many updates to train for.
    ///   - clipGradientNorm: bounds the global gradient norm. The reference clips nothing.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is ``referenceSchedule(steps:)`` with the
    ///     reference optimizer and a constant rate with a caller's.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every parameter trains, the batch normalizations on each batch's statistics. Save with
    /// `NFKMLXWeights.save`; ``network(weightsURL:)``, `detector(weightsURL:)`, and `backend(weightsURL:)`
    /// read the file back. A run is minutes; call it off the render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXRetinaFaceNet,
        examples: (Int) -> (images: MLXArray, faces: [[NFKMLXRetinaFaceAnnotation]]),
        objective: NFKMLXRetinaFaceObjective = NFKMLXRetinaFaceObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { referenceOptimizer() },
            referenceSchedule: { referenceSchedule(steps: steps) },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                let rows = example.faces.enumerated().flatMap { image, faces in
                    faces.flatMap { face -> [Float] in
                        [Float(image), face.landmarks == nil ? 0 : 1, face.x1, face.y1, face.x2, face.y2]
                            + (face.landmarks ?? [Float](repeating: 0, count: 10))
                    }
                }
                return [trainingInput(example.images), MLXArray(rows).reshaped([-1, 16])]
            },
            loss: { net, arrays in
                let rows = arrays[1].asArray(Float.self)
                var faces = [[NFKMLXRetinaFaceAnnotation]](repeating: [], count: arrays[0].dim(0))
                for row in stride(from: 0, to: rows.count, by: 16) {
                    faces[Int(rows[row])].append(NFKMLXRetinaFaceAnnotation(
                        x1: rows[row + 2], y1: rows[row + 3], x2: rows[row + 4], y2: rows[row + 5],
                        landmarks: rows[row + 1] > 0 ? Array(rows[(row + 6) ..< (row + 16)]) : nil))
                }
                return objective(net, arrays[0], faces)
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
