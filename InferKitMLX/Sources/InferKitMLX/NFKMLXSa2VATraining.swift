//
//  NFKMLXSa2VATraining.swift
//  InferKitMLX
//
//  Fine-tuning Sa2VA on a consumer's own referring-segmentation examples, the way its authors fine-tune
//  it (bytedance/Sa2VA, `projects/sa2va/configs/sa2va_finetune.py` and `Sa2VAModel.forward`): LoRA on
//  every linear layer of the language model with its embeddings and head trained whole, the projector,
//  the `[SEG]` bridge, and SAM 2's mask decoder trained, and the vision tower and the rest of SAM 2
//  frozen. The objective is the language model's shifted cross-entropy plus a point-sampled sigmoid
//  cross-entropy and dice on the mask each `[SEG]` drives.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// One Sa2VA training example.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXSa2VAExample {
    /// The InternVL tiles `[tiles, 3, 448, 448]` from ``NFKMLXSa2VAProcessor/tilePixels(_:side:)``.
    public var pixelValues: MLXArray
    /// The prompt and the answer, `[T]`: the instruction in the release's template with its image
    /// context tokens, then the answer carrying one `[SEG]` per object and the template's end.
    public var inputIds: MLXArray
    /// `[T]`: the answer's ids where the loss applies, -100 over the prompt.
    public var labels: MLXArray
    /// The normalized SAM input `[1, side, side, 3]` from ``NFKMLXSa2VAProcessor/groundingPixels(_:side:)``.
    public var groundingImage: MLXArray
    /// One binary mask per `[SEG]`, `[objects, height, width]`, at any resolution.
    public var masks: MLXArray
    /// A Qwen-VL release's patch grid `[3]` (`t`, `h`, `w`) from its image processor; nil for the
    /// InternVL and LLaVA releases.
    public var grid: MLXArray?

    public init(pixelValues: MLXArray, inputIds: MLXArray, labels: MLXArray, groundingImage: MLXArray, masks: MLXArray,
                grid: MLXArray? = nil) {
        self.pixelValues = pixelValues
        self.inputIds = inputIds
        self.labels = labels
        self.groundingImage = groundingImage
        self.masks = masks
        self.grid = grid
    }

    /// The example as the trainer's per-step arrays, and back.
    var arrays: [MLXArray] { [pixelValues, inputIds, labels, groundingImage, masks] + (grid.map { [$0] } ?? []) }

    init(arrays: [MLXArray]) {
        self.init(pixelValues: arrays[0], inputIds: arrays[1], labels: arrays[2], groundingImage: arrays[3],
                  masks: arrays[4], grid: arrays.count > 5 ? arrays[5] : nil)
    }
}

/// A Sa2VA network the fine-tuning recipe drives: the InternVL, Qwen-VL, and LLaVA releases.
protocol NFKSa2VAAdaptable: Module {
    var adaptedLanguage: NFKMLXLanguageNet { get }
    var adaptedBridge: NFKSa2VATextHiddenFCS { get }
    var adaptedGrounding: NFKSa2VAGrounding { get }
    var adaptedSegmentationToken: Int { get }
    /// The projector, which the reference trains for InternVL; Qwen-VL's merger and LLaVA's projector
    /// stay frozen with their vision towers.
    var trainedProjector: Module? { get }
    /// The decoder's hidden states `[1, T, hidden]` over the example's image and ids: `final` after the
    /// final norm, which the head projects, and `segmentation`, which the `[SEG]` bridge reads.
    func trainingHidden(_ example: NFKMLXSa2VAExample, ids: [Int]) -> (final: MLXArray, segmentation: MLXArray)
}

extension NFKMLXSa2VANet: NFKSa2VAAdaptable {
    var adaptedLanguage: NFKMLXLanguageNet { language }
    var adaptedBridge: NFKSa2VATextHiddenFCS { textHiddenFCS }
    var adaptedGrounding: NFKSa2VAGrounding { grounding }
    var adaptedSegmentationToken: Int { configuration.segmentationTokenId }
    var trainedProjector: Module? { projector }
    func trainingHidden(_ example: NFKMLXSa2VAExample, ids: [Int]) -> (final: MLXArray, segmentation: MLXArray) {
        let hidden = language.hiddenStates(fromEmbeddings: fusedEmbeddings(inputIds: ids, imageFeatures: imageFeatures(pixelValues: example.pixelValues)))
        return (hidden, hidden)
    }
}

extension NFKMLXSa2VAQwenNet: NFKSa2VAAdaptable {
    var adaptedLanguage: NFKMLXLanguageNet { decoder }
    var adaptedBridge: NFKSa2VATextHiddenFCS { textHiddenFCS }
    var adaptedGrounding: NFKSa2VAGrounding { grounding }
    var adaptedSegmentationToken: Int { segmentationTokenId }
    var trainedProjector: Module? { nil }
    func trainingHidden(_ example: NFKMLXSa2VAExample, ids: [Int]) -> (final: MLXArray, segmentation: MLXArray) {
        let values = (example.grid ?? MLXArray([Int32(1), 1, 1])).asArray(Int32.self).map(Int.init)
        let grid = (t: values[0], h: values[1], w: values[2])
        let features = imageFeatures(pixelValues: example.pixelValues, grid: grid)
        let unnormalized = NFKMLXQwen3VL.hiddenStates(
            decoder: decoder, inputIds: ids, visionFeatures: features.output, deepstack: features.deepstack,
            gridT: grid.t, gridH: grid.h, gridW: grid.w, applyFinalNorm: false, layout: layout)
        let final = decoder.model.norm(unnormalized)
        return (final, segmentationReadsBeforeFinalNorm ? unnormalized : final)
    }
}

extension NFKMLXSa2VALLaVANet: NFKSa2VAAdaptable {
    var adaptedLanguage: NFKMLXLanguageNet { language }
    var adaptedBridge: NFKSa2VATextHiddenFCS { textHiddenFCS }
    var adaptedGrounding: NFKSa2VAGrounding { grounding }
    var adaptedSegmentationToken: Int { segmentationTokenId }
    var trainedProjector: Module? { nil }
    func trainingHidden(_ example: NFKMLXSa2VAExample, ids: [Int]) -> (final: MLXArray, segmentation: MLXArray) {
        let hidden = language.hiddenStates(fromEmbeddings: fusedEmbeddings(inputIds: ids, imageFeatures: imageFeatures(example.pixelValues)))
        return (hidden, hidden)
    }
}

/// The objective `Sa2VAModel.forward` computes: the language model's loss, plus `maskWeight` times the
/// sigmoid cross-entropy and `diceWeight` times the dice of each predicted mask, both on points drawn
/// with more weight where the prediction is least certain.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXSa2VAObjective: Sendable {
    /// `loss_mask.loss_weight`: 2.
    public var maskWeight: Float
    /// `loss_dice.loss_weight`: 0.5.
    public var diceWeight: Float
    /// `check_obj_number`'s `fix_number`: every example is scored on this many objects, the ones it has
    /// repeated in order when there are fewer, a random subset when there are more.
    public var objectCount: Int
    /// `num_points`, `oversample_ratio`, `importance_sample_ratio`.
    public var pointCount: Int
    public var oversampleRatio: Float
    public var importanceSampleRatio: Float

    /// The released fine-tuning configuration's values.
    public init(maskWeight: Float = 2, diceWeight: Float = 0.5, objectCount: Int = 5, pointCount: Int = 12_544,
                oversampleRatio: Float = 3, importanceSampleRatio: Float = 0.75) {
        self.maskWeight = maskWeight
        self.diceWeight = diceWeight
        self.objectCount = objectCount
        self.pointCount = pointCount
        self.oversampleRatio = oversampleRatio
        self.importanceSampleRatio = importanceSampleRatio
    }

    /// `InternVLMLLM._compute_loss`: logits `[1, T, vocabulary]` scored against the labels shifted one
    /// position, the mean over the labels that are not -100.
    public func languageLoss(logits: MLXArray, labels: MLXArray) -> MLXArray {
        let length = labels.dim(0)
        let predictions = logits[0, 0 ..< (length - 1), 0...].asType(.float32)
        let targets = labels[1...]
        let kept = (targets .!= Int32(-100)).asType(.float32)
        let safeTargets = MLX.where(targets .== Int32(-100), MLXArray(Int32(0)), targets)
        let negativeLogLikelihood = crossEntropy(logits: predictions, targets: safeTargets, reduction: .none)
        return (negativeLogLikelihood * kept).sum() / kept.sum()
    }

    /// mmdet's `point_sample`, `grid_sample` with bilinear weights, zero padding, and `align_corners`
    /// false: maps `[N, H, W]` read at points `[N, P, 2]` given as `(x, y)` in `0...1` → `[N, P]`.
    public static func pointSample(_ maps: MLXArray, _ points: MLXArray) -> MLXArray {
        let (count, height, width) = (maps.dim(0), maps.dim(1), maps.dim(2))
        let x = points[0..., 0..., 0] * Float(width) - 0.5
        let y = points[0..., 0..., 1] * Float(height) - 0.5
        let x0 = floor(x), y0 = floor(y)
        let flat = maps.reshaped([count * height * width])
        let base = (MLXArray(0 ..< count).asType(.float32) * Float(height * width)).reshaped([count, 1])
        func corner(_ cx: MLXArray, _ cy: MLXArray) -> MLXArray {
            let inside = (cx .>= 0) .&& (cx .<= Float(width - 1)) .&& (cy .>= 0) .&& (cy .<= Float(height - 1))
            let index = base + clip(cy, min: 0, max: Float(height - 1)) * Float(width) + clip(cx, min: 0, max: Float(width - 1))
            return MLX.where(inside, flat.take(index.asType(.int32)), MLXArray(Float(0)))
        }
        let wx = x - x0, wy = y - y0
        return corner(x0, y0) * (1 - wx) * (1 - wy) + corner(x0 + 1, y0) * wx * (1 - wy)
            + corner(x0, y0 + 1) * (1 - wx) * wy + corner(x0 + 1, y0 + 1) * wx * wy
    }

    /// mmdet's `get_uncertain_point_coords_with_randomness`: `pointCount × oversampleRatio` uniform
    /// candidates per mask, the `importanceSampleRatio` share of them where the logit is nearest zero,
    /// and the rest uniform. The draws are this process's; the reference's come from its own generator.
    public func uncertainPoints(masks: MLXArray) -> MLXArray {
        let count = masks.dim(0)
        let candidates = MLXRandom.uniform(low: 0, high: 1, [count, Int(Float(pointCount) * oversampleRatio), 2])
        let random = MLXRandom.uniform(low: 0, high: 1, [count, pointCount - Int(importanceSampleRatio * Float(pointCount)), 2])
        return uncertainPoints(masks: masks, candidates: candidates, random: random)
    }

    /// The selection from given draws: the candidates with the smallest `|logit|`, then the uniform ones.
    func uncertainPoints(masks: MLXArray, candidates: MLXArray, random: MLXArray) -> MLXArray {
        let kept = Int(importanceSampleRatio * Float(pointCount))
        let certainty = MLX.abs(Self.pointSample(stopGradient(masks), candidates))                  // [N, S]
        let order = argSort(certainty, axis: 1)[0..., 0 ..< kept]                                   // [N, kept]
        let chosen = takeAlong(candidates, broadcast(order.expandedDimensions(axis: 2), to: [order.dim(0), kept, 2]),
                               axis: 1)
        return concatenated([chosen, random], axis: 1)
    }

    /// The two mask terms before they are weighted: mmdet's `binary_cross_entropy` (soft targets, the
    /// sum over `N × P + 1e-4`) and its naive `dice_loss` (eps 1, the sum over `N + 1e-4`), each divided
    /// with mmdet's float32 epsilon added.
    ///
    /// - Parameters:
    ///   - masks: mask logits `[N, 256, 256]`.
    ///   - targets: binary masks `[N, 256, 256]`, from ``resizedTargets(_:side:)``.
    ///   - points: `[N, P, 2]` from ``uncertainPoints(masks:)``.
    public func maskTerms(masks: MLXArray, targets: MLXArray, points: MLXArray) -> (mask: MLXArray, dice: MLXArray) {
        let predicted = Self.pointSample(masks, points)                                              // [N, P]
        let wanted = stopGradient(Self.pointSample(targets.asType(.float32), points))
        let count = Float(masks.dim(0)), samples = Float(points.dim(1))
        let epsilon = Float.ulpOfOne
        let crossEntropy = MLX.maximum(predicted, 0) - predicted * wanted + log1p(exp(-MLX.abs(predicted)))
        let mask = crossEntropy.sum() / (count * samples + 1e-4 + epsilon)
        let probability = sigmoid(predicted)
        let overlap = (probability * wanted).sum(axis: 1)
        let dice = 1 - (2 * overlap + 1) / (probability.sum(axis: 1) + wanted.sum(axis: 1) + 1)
        return (mask, dice.sum() / (count + 1e-4 + epsilon))
    }

    /// Masks `[N, H, W]` resized to `side` by nearest neighbour, as `F.interpolate(mode="nearest")`.
    public static func resizedTargets(_ masks: MLXArray, side: Int) -> MLXArray {
        let (height, width) = (masks.dim(1), masks.dim(2))
        let rows = MLXArray((0 ..< side).map { Int32(min(Int(Double($0) * Double(height) / Double(side)), height - 1)) })
        let columns = MLXArray((0 ..< side).map { Int32(min(Int(Double($0) * Double(width) / Double(side)), width - 1)) })
        return masks.take(rows, axis: 1).take(columns, axis: 2).asType(.float32)
    }

    /// `check_obj_number`: the objects an example is scored on, as indices into its own.
    func objectOrder(_ count: Int) -> [Int] {
        guard count > 0 else { return [] }
        if count >= objectCount {
            return count == objectCount ? Array(0 ..< count) : Array(Array(0 ..< count).shuffled().prefix(objectCount))
        }
        return (0 ..< objectCount).map { $0 % count }
    }

    /// Scores an InternVL release on one example, the mask points drawn fresh.
    public func callAsFunction(_ net: NFKMLXSa2VANet, _ example: NFKMLXSa2VAExample) -> MLXArray { total(net, example) }

    /// Scores a Qwen-VL release on one example (its `grid` set), the mask points drawn fresh.
    public func callAsFunction(_ net: NFKMLXSa2VAQwenNet, _ example: NFKMLXSa2VAExample) -> MLXArray { total(net, example) }

    /// Scores the LLaVA release on one example, the mask points drawn fresh.
    public func callAsFunction(_ net: NFKMLXSa2VALLaVANet, _ example: NFKMLXSa2VAExample) -> MLXArray { total(net, example) }

    func total<Net: NFKSa2VAAdaptable>(_ net: Net, _ example: NFKMLXSa2VAExample) -> MLXArray {
        let parts = terms(net, example, points: nil)
        return parts.language + parts.mask * maskWeight + parts.dice * diceWeight
    }

    /// The three terms before the mask terms are weighted, and the low-resolution mask logits scored.
    /// `points` fixes the mask points; nil draws them.
    func terms<Net: NFKSa2VAAdaptable>(_ net: Net, _ example: NFKMLXSa2VAExample, points: MLXArray?)
        -> (language: MLXArray, mask: MLXArray, dice: MLXArray, masks: MLXArray?) {
        let ids = example.inputIds.asArray(Int32.self).map(Int.init)
        let hidden = net.trainingHidden(example, ids: ids)
        let language = languageLoss(logits: net.adaptedLanguage.logits(fromHidden: hidden.final), labels: example.labels)

        let segments = ids.indices.filter { ids[$0] == net.adaptedSegmentationToken }
        let order = objectOrder(min(segments.count, example.masks.dim(0)))
        guard !order.isEmpty else {
            return (language, MLXArray(Float(0)), MLXArray(Float(0)), nil)
        }
        let embeddings = net.adaptedBridge(hidden.segmentation[0].take(MLXArray(order.map { Int32(segments[$0]) }), axis: 0))
        let grounding = net.adaptedGrounding
        let levels = grounding.imageLevels(example.groundingImage)
        let masks = MLX.concatenated(order.indices.map { index in
            grounding.segment(levels: levels, languageEmbedding: embeddings[index].reshaped([1, 1, -1])).lowResolution
        }, axis: 0)                                                                                  // [objects, grid, grid]
        let targets = Self.resizedTargets(example.masks.take(MLXArray(order.map(Int32.init)), axis: 0), side: masks.dim(1))
        let (mask, dice) = maskTerms(masks: masks, targets: targets, points: points ?? uncertainPoints(masks: masks))
        return (language, mask, dice, masks)
    }
}

extension NFKMLXLearningRateSchedule {
    /// mmengine's `LinearLR` warm-up then `CosineAnnealingLR` to zero, as a one-epoch run of `steps`
    /// iterations converts them: `warmupRatio × steps` warm-up iterations, of which the last reaches the
    /// full rate (`LinearParamScheduler` counts `end − begin − 1` steps), starting at `startFactor`.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func mmengineWarmupCosine(steps: Int, warmupRatio: Float = 0.05,
                                            startFactor: Float = 1e-5) -> NFKMLXLearningRateSchedule {
        let warmup = Int(Double(warmupRatio) * Double(steps))
        return NFKMLXLearningRateSchedule { step in
            if step < warmup {
                guard warmup > 1 else { return startFactor }
                return startFactor + (1 - startFactor) * Float(step) / Float(warmup - 1)
            }
            let span = Float(max(steps - warmup, 1))
            return 0.5 * (1 + cos(Float.pi * Float(step - warmup) / span))
        }
    }
}

extension NFKMLXSa2VA {

    /// Builds the network itself at float32, ready to fine-tune, from a release directory or a directory
    /// ``save(_:toDirectoryURL:release:)`` wrote.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL) throws -> NFKMLXSa2VANet {
        let net = NFKMLXSa2VANet(try NFKMLXSa2VANet.configuration(fromDirectory: directoryURL))
        try net.loadWeights(fromDirectory: directoryURL, dtype: .float32)
        return net
    }

    /// Fine-tunes a Sa2VA network on a consumer's own examples, returning the loss from each step.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(directoryURL:)``.
    ///   - examples: supplies one example per step.
    ///   - rank: the LoRA width on every linear layer of the language model but its head (128 in the
    ///     reference); the embeddings and the head train whole, as the reference's `modules_to_save`.
    ///   - alpha: the adapter's strength, applied as `alpha / rank` (256 in the reference).
    ///   - objective: the language and mask loss.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.AdamW` at 4e-5, betas 0.9 and
    ///     0.999, weight decay 0.05 on every trained parameter.
    ///   - steps: how many examples to train on.
    ///   - clipGradientNorm: bounds the global gradient norm; the reference clips at 1.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's
    ///     ``NFKMLXLearningRateSchedule/mmengineWarmupCosine(steps:warmupRatio:startFactor:)`` when the
    ///     reference optimizer runs; with a caller's optimizer, nil holds that optimizer's rate constant.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Call `NFKMLXLoRA.merge(into:)` before ``save(_:toDirectoryURL:release:)``.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXSa2VANet,
        examples: (Int) -> NFKMLXSa2VAExample,
        rank: Int = 128,
        alpha: Float = 256,
        objective: NFKMLXSa2VAObjective = NFKMLXSa2VAObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try run(net, examples: examples, rank: rank, alpha: alpha, objective: objective, optimizer: optimizer, steps: steps,
                clipGradientNorm: clipGradientNorm, learningRateSchedule: learningRateSchedule, checkpoint: checkpoint,
                observer: observer)
    }

    /// Fine-tunes a Qwen-VL release (Qwen3-VL or Qwen2.5-VL, SAM 2 or SAM 3 grounding) the same way,
    /// its merger frozen with the vision tower as the reference's `Qwen2_5_VL` wrapper freezes it. Each
    /// example carries its `grid`.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXSa2VAQwenNet,
        examples: (Int) -> NFKMLXSa2VAExample,
        rank: Int = 128,
        alpha: Float = 256,
        objective: NFKMLXSa2VAObjective = NFKMLXSa2VAObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try run(net, examples: examples, rank: rank, alpha: alpha, objective: objective, optimizer: optimizer, steps: steps,
                clipGradientNorm: clipGradientNorm, learningRateSchedule: learningRateSchedule, checkpoint: checkpoint,
                observer: observer)
    }

    /// Fine-tunes the LLaVA release the same way, its projector frozen as the reference's `LLaVAModel`
    /// wrapper freezes it.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXSa2VALLaVANet,
        examples: (Int) -> NFKMLXSa2VAExample,
        rank: Int = 128,
        alpha: Float = 256,
        objective: NFKMLXSa2VAObjective = NFKMLXSa2VAObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try run(net, examples: examples, rank: rank, alpha: alpha, objective: objective, optimizer: optimizer, steps: steps,
                clipGradientNorm: clipGradientNorm, learningRateSchedule: learningRateSchedule, checkpoint: checkpoint,
                observer: observer)
    }

    static func run<Net: NFKSa2VAAdaptable>(
        _ net: Net, examples: (Int) -> NFKMLXSa2VAExample, rank: Int, alpha: Float, objective: NFKMLXSa2VAObjective,
        optimizer: Optimizer?, steps: Int, clipGradientNorm: Float?, learningRateSchedule: NFKMLXLearningRateSchedule?,
        checkpoint: NFKMLXTrainingCheckpoint?, observer: NFKMLXTrainer.Observer?
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: { try prepare(net, rank: rank, alpha: alpha) },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: 4e-5, weightDecay: 0.05) },
            referenceSchedule: { .mmengineWarmupCosine(steps: steps) },
            steps: steps,
            arrays: { examples($0).arrays },
            loss: { model, arrays in objective.total(model, NFKMLXSa2VAExample(arrays: arrays)) },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// The reference's trained set: LoRA on every linear layer of the language model but its head, the
    /// embeddings and head whole, the `[SEG]` bridge, the grounding encoder's mask decoder, and, for
    /// InternVL, the projector.
    static func prepare<Net: NFKSa2VAAdaptable>(_ net: Net, rank: Int, alpha: Float) throws {
        net.freeze()
        let language = net.adaptedLanguage
        let adapted = try NFKMLXLoRA.apply(to: language, rank: rank, alpha: alpha) { path, _ in
            !path.hasSuffix("lm_head")
        }
        guard adapted > 0 else {
            throw NFKMLXError.trainingDataMismatch("no language-model projections were found to adapt, so nothing would train")
        }
        language.model.embedTokens.unfreeze()
        language.lmHead?.unfreeze()
        net.trainedProjector?.unfreeze()
        net.adaptedBridge.unfreeze()
        net.adaptedGrounding.trainedMaskDecoder.unfreeze()
    }

    /// Writes `net` as a release directory: `model.safetensors` in the module's own layout at float32, and
    /// every other file of `release` (configuration, tokenizer, template) copied. ``backend(directoryURL:)``
    /// and ``network(directoryURL:)`` load it.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXSa2VANet, toDirectoryURL directory: URL, release: URL) throws {
        try write(net, toDirectoryURL: directory, release: release)
    }

    /// Writes a fine-tuned Qwen-VL release; ``NFKMLXSa2VAQwenNet/load(directoryURL:)`` and
    /// ``backend(directoryURL:)`` load it.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXSa2VAQwenNet, toDirectoryURL directory: URL, release: URL) throws {
        try write(net, toDirectoryURL: directory, release: release)
    }

    /// Writes the fine-tuned LLaVA release; ``NFKMLXSa2VALLaVANet/load(directoryURL:)`` and
    /// ``backend(directoryURL:)`` load it.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXSa2VALLaVANet, toDirectoryURL directory: URL, release: URL) throws {
        try write(net, toDirectoryURL: directory, release: release)
    }

    static func write(_ net: Module, toDirectoryURL directory: URL, release: URL) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        try NFKMLXWeights.save(net, to: directory.appendingPathComponent("model.safetensors"))
        let weights: Set<String> = ["safetensors", "bin", "pth", "pt"]
        for name in try manager.contentsOfDirectory(atPath: release.path) {
            let source = release.appendingPathComponent(name)
            var isDirectory: ObjCBool = false
            guard manager.fileExists(atPath: source.path, isDirectory: &isDirectory), !isDirectory.boolValue,
                  !weights.contains(source.pathExtension), name != "model.safetensors.index.json" else { continue }
            let destination = directory.appendingPathComponent(name)
            if manager.fileExists(atPath: destination.path) {
                try manager.removeItem(at: destination)
            }
            try manager.copyItem(at: source, to: destination)
        }
    }
}
