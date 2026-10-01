//
//  NFKMLXWav2Vec2Training.swift
//  InferKitMLX
//
//  Fine-tuning Wav2Vec2 and HuBERT for speech recognition the way transformers' own recipe does
//  (`run_speech_recognition_ctc.py`, shipped beside `facebook/wav2vec2-xls-r-2b`): a CTC head over the
//  encoder, the convolutional feature encoder frozen, SpecAugment time masks written with
//  `masked_spec_embed`, and `torch.nn.functional.ctc_loss` over the head's float32 log-softmax.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Which parameters a Wav2Vec2 / HuBERT fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXWav2Vec2Trainable: Sendable {
    /// Everything but the convolutional feature encoder: the reference's `freeze_feature_encoder`
    /// default.
    case encoder
    /// Every parameter, the feature encoder included.
    case all
    /// The CTC head alone, over frozen features.
    case head
}

/// The objective a Wav2Vec2 / HuBERT fine-tune minimizes: transformers' `Wav2Vec2ForCTC` loss,
/// `ctc_loss` over the head's float32 log-softmax with the pad token as the blank.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWav2Vec2Objective: Sendable {
    /// How per-utterance losses combine (`ctc_loss_reduction`).
    public enum Reduction: Sendable {
        /// Each utterance's loss divided by its label count, then averaged over the batch: the
        /// training script's default.
        case mean
        /// The utterances' losses summed: the releases' own `config.json` setting.
        case sum
    }

    public var reduction: Reduction
    /// Replaces an infinite loss (labels no alignment can produce in the frames available) and its
    /// gradient with zero (`ctc_zero_infinity`).
    public var zeroInfinity: Bool
    /// The CTC blank (`pad_token_id`).
    public var blank: Int

    public init(reduction: Reduction = .mean, zeroInfinity: Bool = false, blank: Int = 0) {
        self.reduction = reduction
        self.zeroInfinity = zeroInfinity
        self.blank = blank
    }

    /// A log-likelihood below this is an alignment that cannot exist. The recursion runs in finite
    /// arithmetic so its gradient stays finite where `-inf - (-inf)` would be NaN.
    static let impossible: Float = -1e30

    /// Scores `net` on a normalized waveform `[1, samples]` against one utterance's label ids, with
    /// SpecAugment's masked frames replaced when `timeMask` (`[1, frames]`, boolean) is given.
    public func callAsFunction(_ net: NFKMLXWav2Vec2Net, _ waveform: MLXArray, labels: [Int],
                               timeMask: MLXArray? = nil) -> MLXArray {
        guard let head = net.head else {
            preconditionFailure("a CTC fine-tune needs a head; build the network with a vocabulary")
        }
        let projected = net.featureProjection(net.extractFeatures(waveform))
        let logits = head(net.encode(projected, timeMask: timeMask).output)
        return loss(logits: logits, labels: [labels])
    }

    /// The reduced loss from logits `[B, frames, vocabulary]` and each utterance's label ids. An utterance
    /// reads its first `frames[b]` frames, or all of them when `frames` is nil (the reference's
    /// `input_lengths`).
    public func loss(logits: MLXArray, labels: [[Int]], frames: [Int]? = nil) -> MLXArray {
        let logProbabilities = logSoftmax(logits.asType(.float32), axis: -1)
        var losses = [MLXArray]()
        for (index, utterance) in labels.enumerated() {
            let count = frames?[index] ?? logProbabilities.dim(1)
            var nll = -Self.logLikelihood(logProbabilities[index, ..<count], labels: utterance, blank: blank)
            if zeroInfinity {
                nll = MLX.where(nll .>= -Self.impossible / 2, MLXArray(Float(0)), nll)
            }
            if reduction == .mean {
                nll = nll / Float(max(utterance.count, 1))
            }
            losses.append(nll)
        }
        let stacked = MLX.stacked(losses)
        return reduction == .mean ? stacked.mean() : stacked.sum()
    }

    /// The CTC forward recursion over the label sequence extended with blanks, in log space:
    /// `α_t(s) = logsumexp(α_{t−1}(s), α_{t−1}(s−1), α_{t−1}(s−2) when l′_s ≠ blank and l′_s ≠ l′_{s−2})
    /// + log p_t(l′_s)`, ending in either of the last two states.
    static func logLikelihood(_ logProbabilities: MLXArray, labels: [Int], blank: Int) -> MLXArray {
        let frames = logProbabilities.dim(0)
        var extended = [blank]
        for label in labels { extended += [label, blank] }
        let states = extended.count
        let emissions = take(logProbabilities, MLXArray(extended.map(Int32.init)), axis: 1)       // [frames, S]
        let skips = MLXArray((0 ..< states).map { s in
            s >= 2 && extended[s] != blank && extended[s] != extended[s - 2]
        })
        let floor = MLXArray(impossible)
        let initial = MLXArray((0 ..< states).map { $0 < 2 ? Float(0) : impossible })
        var alpha = MLX.where(initial .== 0, emissions[0], floor)
        let pad1 = MLXArray([impossible])
        let pad2 = MLXArray([impossible, impossible])
        for t in 1 ..< max(frames, 1) {
            let stay = alpha
            let advance = concatenated([pad1, alpha[..<(states - 1)]])
            let skip = MLX.where(skips, concatenated([pad2, alpha[..<(states - 2)]]), floor)
            alpha = logSumExp(stacked([stay, advance, skip]), axis: 0) + emissions[t]
        }
        if states == 1 { return alpha[0] }
        return logAddExp(alpha[states - 1], alpha[states - 2])
    }
}

/// SplitMix64, the seeded generator SpecAugment's draws use so a run repeats from its seed.
struct NFKMLXSplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// SpecAugment's time masking as transformers' `_compute_mask_indices` draws it for one utterance: a
/// span count of `int(prob · frames / length + ε)` (ε uniform in `[0, 1)`), at least `minimumMasks`,
/// start frames chosen without replacement, each span `length` frames and clipped at the end.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXSpecAugment {
    /// A boolean `[1, frames]` time mask, or nil when the probability masks nothing.
    public static func timeMask<G: RandomNumberGenerator>(frames: Int, probability: Float, length: Int,
                                                         minimumMasks: Int, using generator: inout G) -> MLXArray? {
        guard probability > 0, length >= 1, length <= frames else { return nil }
        let epsilon = Float.random(in: 0 ..< 1, using: &generator)
        var spans = max(Int(probability * Float(frames) / Float(length) + epsilon), minimumMasks)
        if spans * length > frames { spans = frames / length }
        spans = min(spans, max(frames - (length - 1), 0))
        guard spans > 0 else { return nil }
        var starts = Array(0 ..< (frames - (length - 1)))
        starts.shuffle(using: &generator)
        var mask = [Bool](repeating: false, count: frames)
        for start in starts.prefix(spans) {
            for offset in 0 ..< length { mask[min(start + offset, frames - 1)] = true }
        }
        return MLXArray(mask).reshaped([1, frames])
    }
}

extension NFKMLXWav2Vec2 {

    /// Builds the network itself, ready to fine-tune, from a released directory.
    ///
    /// - Parameters:
    ///   - directoryURL: a Wav2Vec2 or HuBERT release, or a directory ``save(_:tokenizer:toDirectoryURL:)``
    ///     wrote.
    ///   - vocabulary: the consumer's own characters in id order, the blank first. Nil keeps the release's
    ///     head, and a pretraining release without one gets none. With a vocabulary, the head is freshly
    ///     initialized at that size unless the release's head already has it, and everything else loads.
    ///
    /// @discussion Dropping a differently sized head is required: MLX's `update(parameters:)` adopts a
    /// checkpoint's shapes rather than validating them, so the release's head would replace the
    /// consumer's.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, vocabulary: [String]? = nil) throws -> NFKMLXWav2Vec2Net {
        let release = try NFKMLXWav2Vec2Configuration(configurationURL: directoryURL.appendingPathComponent("config.json"))
        let weights = try NFKMLXWav2Vec2Net.weightsURL(inDirectory: directoryURL)
        guard let vocabulary, vocabulary.count != release.vocabularySize else {
            let net = NFKMLXWav2Vec2Net(release)
            try net.loadWeights(url: weights, leavingFresh: [])
            return net
        }
        guard vocabulary.count >= 2 else {
            throw NFKMLXError.unsupportedConfiguration("a CTC vocabulary needs the blank and at least one character")
        }
        var configuration = release
        configuration.vocabularySize = vocabulary.count
        let net = NFKMLXWav2Vec2Net(configuration)
        try net.loadWeights(url: weights, leavingFresh: ["lm_head."])
        return net
    }

    /// Fine-tunes a CTC head and encoder on a consumer's own transcribed speech, returning the loss from
    /// each step.
    ///
    /// The whole customization path is three calls: ``network(directoryURL:vocabulary:)`` to build, this
    /// to train, and ``save(_:tokenizer:toDirectoryURL:)`` to write a directory that
    /// ``backend(directoryURL:)`` loads like any release.
    ///
    /// - Parameters:
    ///   - net: the network to train; it must carry a CTC head.
    ///   - examples: supplies one utterance per step: its 16 kHz samples and its label ids
    ///     (``NFKMLXWav2Vec2Tokenizer/labels(for:)``).
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: the CTC loss.
    ///   - timeMasking: whether SpecAugment masks feature frames, at the release's `mask_time_prob`,
    ///     `mask_time_length`, and `mask_time_min_masks`, as the reference applies it in training.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.AdamW` (bias-corrected,
    ///     betas 0.9 and 0.999, epsilon 1e-8) at `learningRate` with `weightDecay`.
    ///   - learningRate: the training script's peak rate (the `TrainingArguments` default).
    ///   - weightDecay: the reference optimizer's weight decay.
    ///   - steps: how many utterances to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update, 1.0 as the reference's
    ///     `max_grad_norm`.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's linear decay
    ///     to zero over the run with no warm-up (`get_linear_schedule_with_warmup`), when the reference
    ///     optimizer runs. With a caller's optimizer, nil holds that optimizer's rate constant.
    ///   - seed: seeds SpecAugment's draws.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXWav2Vec2Net,
        examples: (Int) -> (samples: [Float], labels: [Int]),
        trainable: NFKMLXWav2Vec2Trainable = .encoder,
        objective: NFKMLXWav2Vec2Objective = NFKMLXWav2Vec2Objective(),
        timeMasking: Bool = true,
        optimizer: Optimizer? = nil,
        learningRate: Float = 5e-5,
        weightDecay: Float = 0,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        seed: UInt64 = 0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        let c = net.configuration
        var generator = NFKMLXSplitMix64(seed: seed)
        return try fineTune(
            net, examples: examples, trainable: trainable, objective: objective,
            timeMasks: { frames in
                timeMasking ? NFKMLXSpecAugment.timeMask(frames: frames, probability: c.maskTimeProbability,
                                                         length: c.maskTimeLength, minimumMasks: c.maskTimeMinimumMasks,
                                                         using: &generator) : nil
            },
            optimizer: optimizer, learningRate: learningRate, weightDecay: weightDecay, steps: steps,
            clipGradientNorm: clipGradientNorm, learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// The recipe with its time masks supplied per step by `timeMasks(frames)`, which is how a
    /// measurement applies the reference's own draws.
    static func fineTune(
        _ net: NFKMLXWav2Vec2Net,
        examples: (Int) -> (samples: [Float], labels: [Int]),
        trainable: NFKMLXWav2Vec2Trainable,
        objective: NFKMLXWav2Vec2Objective,
        timeMasks: (Int) -> MLXArray?,
        optimizer: Optimizer?,
        learningRate: Float,
        weightDecay: Float,
        steps: Int,
        clipGradientNorm: Float?,
        learningRateSchedule: NFKMLXLearningRateSchedule?,
        checkpoint: NFKMLXTrainingCheckpoint?,
        observer: NFKMLXTrainer.Observer?
    ) throws -> [Float] {
        guard net.transcribes else {
            throw NFKMLXError.unsupportedConfiguration(
                "this Wav2Vec2 network has no CTC head; build it with network(directoryURL:vocabulary:)")
        }
        let c = net.configuration
        var utterances = [[Int]]()
        return try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: learningRate, weightDecay: weightDecay) },
            referenceSchedule: { .poly(steps: steps) },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                utterances = [example.labels]
                let input = NFKMLXWav2Vec2Processor.inputValues(example.samples, normalize: c.normalizesInput)
                let mask = timeMasks(c.frameCount(samples: example.samples.count))
                return mask.map { [input, $0] } ?? [input]
            },
            loss: { net, arrays in
                objective(net, arrays[0], labels: utterances[0], timeMask: arrays.count > 1 ? arrays[1] : nil)
            },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// Writes `net` as a release directory: `model.safetensors` in the module's own layout, the
    /// `config.json` and `preprocessor_config.json` its configuration reads back from, and, when
    /// `tokenizer` is given, `vocab.json`. ``backend(directoryURL:)`` and
    /// ``network(directoryURL:vocabulary:)`` load it.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXWav2Vec2Net, tokenizer: NFKMLXWav2Vec2Tokenizer?,
                            toDirectoryURL directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try NFKMLXWeights.save(net, to: directory.appendingPathComponent("model.safetensors"))
        let c = net.configuration
        var config: [String: Any] = [
            "model_type": c.modelType,
            "hidden_size": c.hiddenSize, "num_hidden_layers": c.numHiddenLayers,
            "num_attention_heads": c.numAttentionHeads, "intermediate_size": c.intermediateSize,
            "conv_dim": c.convDimensions, "conv_kernel": c.convKernels, "conv_stride": c.convStrides,
            "conv_bias": c.convBias, "feat_extract_norm": c.featureNorm.rawValue,
            "do_stable_layer_norm": c.stableLayerNorm, "feat_proj_layer_norm": c.featureProjectionLayerNorm,
            "num_conv_pos_embeddings": c.positionalConvKernel,
            "num_conv_pos_embedding_groups": c.positionalConvGroups,
            "layer_norm_eps": Double(c.layerNormEps), "pad_token_id": c.padTokenID,
            "mask_time_prob": Double(c.maskTimeProbability), "mask_time_length": c.maskTimeLength,
            "mask_time_min_masks": c.maskTimeMinimumMasks,
            "hidden_act": "gelu", "feat_extract_activation": "gelu",
        ]
        if let vocabularySize = c.vocabularySize {
            config["vocab_size"] = vocabularySize
            config["architectures"] = [c.modelType == "hubert" ? "HubertForCTC" : "Wav2Vec2ForCTC"]
        } else {
            config["architectures"] = [c.modelType == "hubert" ? "HubertModel" : "Wav2Vec2Model"]
        }
        let processor: [String: Any] = ["do_normalize": c.normalizesInput, "sampling_rate": NFKMLXWav2Vec2Processor.sampleRate,
                                        "feature_size": 1, "padding_value": 0.0]
        var files: [(String, Any)] = [("config.json", config), ("preprocessor_config.json", processor)]
        if let tokenizer {
            files.append(("vocab.json", tokenizer.ids))
            files.append(("tokenizer_config.json", ["pad_token": tokenizer.padToken, "unk_token": tokenizer.unknownToken,
                                                    "word_delimiter_token": tokenizer.wordDelimiter]))
        }
        for (name, object) in files {
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
                .write(to: directory.appendingPathComponent(name))
        }
    }

    /// Freezes everything but the parameters `trainable` names.
    static func apply(_ trainable: NFKMLXWav2Vec2Trainable, to net: NFKMLXWav2Vec2Net) {
        switch trainable {
        case .all:
            net.unfreeze()
        case .encoder:
            net.unfreeze()
            net.featureEncoder.freeze()
        case .head:
            net.freeze()
            net.head?.unfreeze()
        }
    }
}

extension NFKMLXWav2Vec2Tokenizer {
    /// A tokenizer over a consumer's own characters in id order: the blank (`<pad>`) first, then the word
    /// delimiter `|`, `<unk>`, and the characters. Its `characters` are what
    /// ``NFKMLXWav2Vec2/network(directoryURL:vocabulary:)`` sizes a new head to.
    ///
    /// Introduced in InferKit 0.4.0.
    public convenience init(characters: [String]) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var ordered = ["<pad>", "|", "<unk>"]
        for character in characters where !ordered.contains(character) { ordered.append(character) }
        let map = Dictionary(uniqueKeysWithValues: ordered.enumerated().map { ($0.element, $0.offset) })
        try JSONSerialization.data(withJSONObject: map).write(to: directory.appendingPathComponent("vocab.json"))
        try self.init(directoryURL: directory)
    }

    /// The tokens in id order.
    public var characters: [String] { tokens }
}
