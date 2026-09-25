//
//  NFKMLXYOLOGenerationsTraining.swift
//  InferKitMLX
//
//  Fine-tuning the later YOLO generations (v9, v10, 11, 12, YOLO26) on a consumer's own labeled images,
//  with ultralytics' own recipe. The generations that suppress duplicates themselves (v10, YOLO26) train
//  both of their head's branches under `E2ELoss`, the one-to-one branch on features the backbone's
//  gradient does not see; the others train under `v8DetectionLoss` as YOLOv8 does. The optimizer,
//  schedule, and weight average are YOLOv8's (`NFKMLXYOLOTraining.swift`).
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

extension NFKMLXYOLOGenerationNet {

    /// The head's raw outputs for images `[batch, H, W, 3]` in `0...1` whose sides are multiples of 32,
    /// flattened over every anchor as the objective reads them. The one-to-one branch, present on the
    /// end-to-end generations, reads the features with their gradient stopped, as the reference detaches
    /// them in training.
    func headOutputs(_ images: MLXArray) -> (oneToMany: (distribution: MLXArray, logits: MLXArray),
                                             oneToOne: (distribution: MLXArray, logits: MLXArray)?,
                                             featureSizes: [(height: Int, width: Int)], strides: [Int]) {
        let scales = features(images)
        let batch = images.dim(0)
        let sizes = scales.map { (height: $0.dim(1), width: $0.dim(2)) }
        let strides = scales.map { images.dim(1) / $0.dim(1) }
        func flattened(_ maps: (boxes: [MLXArray], scores: [MLXArray])) -> (distribution: MLXArray, logits: MLXArray) {
            (concatenated(maps.boxes.map { $0.reshaped([batch, -1, 4 * head.regMax]) }, axis: 1),
             concatenated(maps.scores.map { $0.reshaped([batch, -1, head.classCount]) }, axis: 1))
        }
        let many = flattened(head.maps(scales, oneToOne: false))
        let one = head.endToEnd ? flattened(head.maps(scales.map { stopGradient($0) }, oneToOne: true)) : nil
        return (many, one, sizes, strides)
    }

    /// ultralytics' `Detect.bias_init` over every branch: box outputs at 2, class outputs at
    /// `log(5 / classes / (640 / stride)²)`.
    func initializeHeadBiases() {
        let classes = Float(head.classCount)
        let boxBranches = [head.cv2] + (head.oneToOneBox.map { [$0] } ?? [])
        let classBranches = [head.cv3] + (head.oneToOneClass.map { [$0] } ?? [])
        for branches in boxBranches {
            for branch in branches {
                branch.out.update(parameters: ModuleParameters.unflattened(["bias": MLXArray.ones([4 * head.regMax]) * 2]))
            }
        }
        for branches in classBranches {
            for (index, branch) in branches.enumerated() {
                let prior = logf(5 / classes / powf(640 / Float(strides[index]), 2))
                let output: Conv2d? = (branch as? NFKYOLOClassBranch)?.out ?? (branch as? NFKYOLODetectBranch)?.out
                output?.update(parameters: ModuleParameters.unflattened(["bias": MLXArray.ones([head.classCount]) * prior]))
            }
        }
    }
}

extension NFKMLXYOLOGenerations {

    /// Builds a release's network for `classCount` classes, ready to fine-tune. The head's biases start
    /// at `bias_init`'s values, and `weightsURL` transfers every tensor whose shape matches
    /// (`intersect_dicts`), so a class count other than the checkpoint's keeps fresh class branches.
    ///
    /// - Since: InferKit 0.5.0
    public static func network(release: NFKMLXYOLORelease, classCount: Int = 80,
                               weightsURL: URL?) throws -> NFKMLXYOLOGenerationNet {
        let net = makeNet(release, classCount: classCount)
        net.initializeHeadBiases()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL, matchingShapesOnly: true)
        }
        return net
    }

    /// Fine-tunes a release on a consumer's own labeled images, returning the loss from each step.
    ///
    /// The parameters and defaults are ``NFKMLXYOLO/fineTune(_:examples:trainable:objective:optimizer:steps:stepsPerEpoch:clipGradientNorm:learningRateSchedule:averagesWeights:checkpoint:observer:)``'s.
    /// An end-to-end release (v10, YOLO26) trains both branches under `endToEndObjective`, its
    /// one-to-many weight stepping by epoch; the others train under `objective`. The result saves with
    /// `NFKMLXWeights.save` and loads through `backendWithRelease:weightsURL:labels:error:` at its own
    /// class count.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXYOLOGenerationNet,
        examples: (Int) -> (images: MLXArray, targets: [[NFKMLXYOLOBox]]),
        trainable: NFKMLXYOLOTrainable = .everything,
        objective: NFKMLXYOLOObjective = NFKMLXYOLOObjective(),
        endToEndObjective: NFKMLXYOLOEndToEndObjective = NFKMLXYOLOEndToEndObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        stepsPerEpoch: Int,
        clipGradientNorm: Float? = 10,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        averagesWeights: Bool = true,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        var average = averagesWeights ? NFKMLXModelWeightAverage(net) : nil
        let perEpoch = max(stepsPerEpoch, 1)
        let epochs = (steps + perEpoch - 1) / perEpoch
        let history = try NFKMLXFineTune.run(
            net,
            freezing: {
                switch trainable {
                case .everything:
                    net.unfreeze()
                case .head:
                    net.freeze()
                    net.head.unfreeze()
                }
                net.head.dfl?.freeze()
                for (_, module) in net.leafModules().flattened() {
                    if let norm = module as? BatchNorm {
                        norm.freeze(recursive: false, keys: ["running_mean", "running_var"])
                    }
                }
            },
            optimizer: optimizer,
            reference: {
                NFKMLXYOLO.referenceOptimizer(classCount: net.classCount, batchSize: examples(0).images.dim(0))
            },
            referenceSchedule: { .ultralytics(steps: steps, stepsPerEpoch: stepsPerEpoch) },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                let rows = example.targets.enumerated().flatMap { image, boxes in
                    boxes.flatMap { [Float(image), Float($0.classIndex), $0.x1, $0.y1, $0.x2, $0.y2] }
                }
                return [example.images, MLXArray(rows).reshaped([-1, 6]), MLXArray(Int32(step / perEpoch))]
            },
            loss: { net, arrays in
                let outputs = net.headOutputs(arrays[0])
                let rows = arrays[1].asArray(Float.self)
                var targets = [[NFKMLXYOLOBox]](repeating: [], count: arrays[0].dim(0))
                for row in stride(from: 0, to: rows.count, by: 6) {
                    targets[Int(rows[row])].append(NFKMLXYOLOBox(classIndex: Int(rows[row + 1]), x1: rows[row + 2],
                                                                 y1: rows[row + 3], x2: rows[row + 4], y2: rows[row + 5]))
                }
                guard let one = outputs.oneToOne else {
                    return objective.loss(boxDistribution: outputs.oneToMany.distribution,
                                          classLogits: outputs.oneToMany.logits, featureSizes: outputs.featureSizes,
                                          strides: outputs.strides, targets: targets)
                }
                return endToEndObjective.loss(oneToMany: outputs.oneToMany, oneToOne: one,
                                              featureSizes: outputs.featureSizes, strides: outputs.strides,
                                              targets: targets, epoch: arrays[2].item(Int.self), epochs: epochs)
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
