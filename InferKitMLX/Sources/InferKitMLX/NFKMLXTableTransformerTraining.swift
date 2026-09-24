//
//  NFKMLXTableTransformerTraining.swift
//  InferKitMLX
//
//  Fine-tuning Table Transformer on a consumer's own tables, the way microsoft/table-transformer
//  trains it: its vendored DETR `SetCriterion` behind a `HungarianMatcher`, scored on the last decoder
//  layer (the released configurations set `aux_loss` false), with AdamW over two rate groups.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a Table Transformer fine-tune updates. Every frozen batch norm stays frozen, as in
/// the reference.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXTableTransformerTrainable: Sendable {
    /// The class and box heads.
    case heads
    /// The heads, the encoder and decoder, the input projection, and the query embeddings.
    case transformer
    /// What the reference trains: the transformer and heads, and the backbone's last three stages
    /// (DETR freezes the stem and the first stage).
    case everything
}

/// DETR's set objective as Table Transformer trains with it: each target is assigned its query by
/// the Hungarian method, then a class cross-entropy over every query (the no-object class down-weighted),
/// an L1 box term, and a generalized-IoU term over the matched pairs.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTableTransformerObjective: Sendable {
    /// `ce_loss_coef`, `bbox_loss_coef`, `giou_loss_coef`.
    public var classWeight: Float
    public var boxWeight: Float
    public var generalizedIoUWeight: Float
    /// `eos_coef`: the cross-entropy weight of the no-object class.
    public var noObjectWeight: Float
    /// `set_cost_class`, `set_cost_bbox`, `set_cost_giou`: the matching cost's terms.
    public var matchClassCost: Float
    public var matchBoxCost: Float
    public var matchGeneralizedIoUCost: Float

    /// The released configurations' values.
    public init(classWeight: Float = 1, boxWeight: Float = 5, generalizedIoUWeight: Float = 2,
                noObjectWeight: Float = 0.4, matchClassCost: Float = 1, matchBoxCost: Float = 5,
                matchGeneralizedIoUCost: Float = 2) {
        self.classWeight = classWeight
        self.boxWeight = boxWeight
        self.generalizedIoUWeight = generalizedIoUWeight
        self.noObjectWeight = noObjectWeight
        self.matchClassCost = matchClassCost
        self.matchBoxCost = matchBoxCost
        self.matchGeneralizedIoUCost = matchGeneralizedIoUCost
    }

    /// Scores `net` on one image: pixels `[1, H, W, 3]` from the release's processor and targets
    /// `[N, 5]`, each row `[class, cx, cy, w, h]` with the box normalized to `0...1`.
    public func callAsFunction(_ net: NFKMLXTableTransformerNet, _ pixels: MLXArray, _ targets: MLXArray) -> MLXArray {
        let detection = net(pixels)
        return loss(logits: detection.logits, boxes: detection.boxes, targets: targets)
    }

    /// The target of each query, from `DETR`'s `HungarianMatcher`: the cost is the negative class
    /// probability, the L1 distance, and the negative generalized IoU, weighted.
    ///
    /// - Returns: for each target, the query it matched, in target order.
    public func match(logits: MLXArray, boxes: MLXArray, targets: MLXArray) -> [Int] {
        let queries = logits.dim(0), count = targets.dim(0)
        guard count > 0 else { return [] }
        let classes = targets[0..., 0].asType(.int32)
        let wanted = targets[0..., 1...]
        let probability = softmax(logits, axis: -1).take(classes, axis: 1)                        // [Q, N]
        let boxCost = MLX.abs(boxes.expandedDimensions(axis: 1) - wanted.expandedDimensions(axis: 0)).sum(axis: -1)
        let generalized = NFKSAM3BoxMetrics.generalizedIoU(NFKSAM3Boxes.centerToCorners(boxes),
                                                           NFKSAM3Boxes.centerToCorners(wanted))
        let cost = boxCost * matchBoxCost - probability * matchClassCost - generalized * matchGeneralizedIoUCost
        eval(cost)
        return NFKMLXHungarian.assign(cost: cost.reshaped([-1]).asArray(Float.self), rows: queries, columns: count)
    }

    /// The three terms before they are weighted: `loss_ce`, `loss_bbox`, `loss_giou`.
    ///
    /// - Parameters:
    ///   - logits: class logits `[Q, labels + 1]`, the last the no-object class.
    ///   - boxes: predicted boxes `[Q, 4]` as `(cx, cy, w, h)` in `0...1`.
    ///   - targets: `[N, 5]` rows of `[class, cx, cy, w, h]`.
    public func terms(logits: MLXArray, boxes: MLXArray, targets: MLXArray)
        -> (classification: MLXArray, box: MLXArray, generalizedIoU: MLXArray) {
        let queries = logits.dim(0), noObject = logits.dim(1) - 1
        let matched = match(logits: logits, boxes: boxes, targets: targets)
        var assigned = [Int32](repeating: Int32(noObject), count: queries)
        let classes = targets.dim(0) > 0 ? targets[0..., 0].asType(.int32).asArray(Int32.self) : []
        for (target, query) in matched.enumerated() {
            assigned[query] = classes[target]
        }
        // `F.cross_entropy` with class weights: the weighted mean, divided by the weights' sum.
        let weights = MLXArray(assigned.map { $0 == Int32(noObject) ? noObjectWeight : 1 })
        let negativeLogLikelihood = -takeAlong(logSoftmax(logits, axis: -1), MLXArray(assigned).reshaped([queries, 1]),
                                               axis: 1).reshaped([queries])
        let classification = (weights * negativeLogLikelihood).sum() / weights.sum()

        // DETR clamps the box count to at least one.
        let count = Float(max(matched.count, 1))
        guard !matched.isEmpty else { return (classification, MLXArray(Float(0)), MLXArray(Float(0))) }
        let predicted = boxes.take(MLXArray(matched.map(Int32.init)), axis: 0)
        let wanted = targets[0..., 1...]
        let box = MLX.abs(predicted - wanted).sum() / count
        let overlap = NFKSAM3BoxMetrics.generalizedIoU(NFKSAM3Boxes.centerToCorners(predicted),
                                                       NFKSAM3Boxes.centerToCorners(wanted))
        let pairs = MLXArray(0 ..< matched.count)
        let diagonal = takeAlong(overlap, pairs.reshaped([-1, 1]), axis: 1).reshaped([-1])
        return (classification, box, (1 - diagonal).sum() / count)
    }

    /// The weighted sum of the three terms.
    public func loss(logits: MLXArray, boxes: MLXArray, targets: MLXArray) -> MLXArray {
        let parts = terms(logits: logits, boxes: boxes, targets: targets)
        return parts.classification * classWeight + parts.box * boxWeight + parts.generalizedIoU * generalizedIoUWeight
    }
}

extension NFKMLXTableTransformer {

    /// Builds the network itself, ready to fine-tune, from a release directory or a directory
    /// ``save(_:toDirectoryURL:release:)`` wrote.
    ///
    /// - Parameter labels: the consumer's own classes. Nil keeps the release's. A different class
    ///   count leaves the classifier at its fresh initialization and loads everything else; MLX adopts
    ///   a checkpoint's shapes rather than validating them, so the release's classifier would otherwise
    ///   replace the new one.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, labels: [String]? = nil) throws -> NFKMLXTableTransformerNet {
        var configuration = try NFKMLXTableTransformerConfiguration(
            configurationURL: directoryURL.appendingPathComponent("config.json"))
        guard let labels, labels != configuration.labels else {
            let net = NFKMLXTableTransformerNet(configuration)
            try net.loadWeights(fromDirectory: directoryURL)
            return net
        }
        let keepsClassifier = labels.count == configuration.numLabels
        configuration.labels = labels
        configuration.numLabels = labels.count
        let net = NFKMLXTableTransformerNet(configuration)
        try net.loadWeights(fromDirectory: directoryURL, skippingClassifier: !keepsClassifier)
        return net
    }

    /// Fine-tunes a Table Transformer on a consumer's own tables, returning the loss from each step.
    ///
    /// - Parameters:
    ///   - net: the network to train.
    ///   - examples: supplies one image per step: its pixels (from the release's processor) and its
    ///     targets `[N, 5]`, rows of `[class, cx, cy, w, h]` normalized to `0...1`.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: the set loss.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.AdamW` with weight decay
    ///     1e-4 on every parameter, at 5e-5, and 1e-5 for the backbone.
    ///   - steps: how many images to train on.
    ///   - clipGradientNorm: bounds the global gradient norm; the reference clips at 0.1.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's `StepLR`,
    ///     0.9 per epoch, with `stepsPerEpoch` updates to an epoch, when the reference optimizer runs.
    ///     With a caller's optimizer, nil holds that optimizer's rate constant.
    ///   - stepsPerEpoch: updates per epoch for the reference schedule: the consumer's set size.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXTableTransformerNet,
        examples: (Int) -> (pixels: MLXArray, targets: MLXArray),
        trainable: NFKMLXTableTransformerTrainable = .everything,
        objective: NFKMLXTableTransformerObjective = NFKMLXTableTransformerObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 0.1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        stepsPerEpoch: Int = 1_000,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: {
                NFKMLXReferenceOptimizers.adamW(learningRate: 5e-5, over: net) { key in
                    (key.hasPrefix("model.backbone.") ? 0.2 : 1, 1e-4)
                }
            },
            referenceSchedule: {
                NFKMLXLearningRateSchedule { pow(Float(0.9), Float($0 / max(stepsPerEpoch, 1))) }
            },
            steps: steps,
            batch: { let example = examples($0); return (example.pixels, example.targets) },
            loss: objective.callAsFunction,
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// Writes `net` as a release directory: `model.safetensors` in the module's own layout, the
    /// release's `config.json` with the network's labels as `id2label`, and its
    /// `preprocessor_config.json`. ``backend(directoryURL:)`` and ``network(directoryURL:labels:)`` load it.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXTableTransformerNet, toDirectoryURL directory: URL, release: URL) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        try NFKMLXWeights.save(net, to: directory.appendingPathComponent("model.safetensors"))
        let data = try Data(contentsOf: release.appendingPathComponent("config.json"))
        var config = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let labels = net.config.labels
        config["id2label"] = Dictionary(uniqueKeysWithValues: labels.enumerated().map { (String($0.offset), $0.element) })
        config["label2id"] = Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($0.element, $0.offset) })
        config["num_labels"] = labels.count
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("config.json"))
        let processor = release.appendingPathComponent("preprocessor_config.json")
        let copy = directory.appendingPathComponent("preprocessor_config.json")
        if manager.fileExists(atPath: processor.path), processor.standardizedFileURL != copy.standardizedFileURL {
            if manager.fileExists(atPath: copy.path) {
                try manager.removeItem(at: copy)
            }
            try manager.copyItem(at: processor, to: copy)
        }
    }

    /// Freezes everything, then unfreezes what `trainable` names; the frozen batch norms stay frozen.
    private static func apply(_ trainable: NFKMLXTableTransformerTrainable, to net: NFKMLXTableTransformerNet) {
        net.freeze()
        net.classifier.unfreeze()
        net.bboxPredictor.unfreeze()
        guard trainable != .heads else { return }
        net.model.inputProjection.unfreeze()
        net.model.encoder.unfreeze()
        net.model.decoder.unfreeze()
        net.model.queryPositionEmbeddings.unfreeze()
        guard trainable == .everything else { return }
        let resnet = net.model.backbone.convEncoder.model
        for block in resnet.layer2 + resnet.layer3 + resnet.layer4 {
            block.unfreeze()
            for (_, module) in block.leafModules().flattened() where module is NFKTableTransformerFrozenBatchNorm {
                module.freeze()
            }
        }
    }
}
