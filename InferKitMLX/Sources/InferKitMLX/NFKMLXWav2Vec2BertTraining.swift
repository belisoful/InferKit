//
//  NFKMLXWav2Vec2BertTraining.swift
//  InferKitMLX
//
//  Fine-tuning W2V-BERT 2.0 for speech recognition the way Hugging Face's recipe does ("Fine-Tune W2V2-Bert
//  for low-resource ASR"): `Wav2Vec2BertForCTC` with the output adapter added (`add_adapter=True`) and a CTC
//  head over the consumer's characters, every dropout, layer drop, and time mask off, CTC reduced by mean,
//  every parameter trained by the Trainer's AdamW at 5e-5 with a 500-step linear warm-up and a linear decay.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Which parameters a W2V-BERT fine-tune updates.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXWav2Vec2BertTrainable: Sendable {
    /// Every parameter: the reference's recipe.
    case all
    /// The adapter and the CTC head, over the frozen encoder.
    case adapterAndHead
    /// The CTC head alone.
    case head
}

extension NFKMLXWav2Vec2BertNet {
    /// Loads a directory's `model.safetensors`, leaving parameters under `fresh` (module prefixes such as
    /// `adapter.` and `lm_head.`) at their initialization. Every other parameter must be supplied.
    func loadWeights(fromDirectory directory: URL, leavingFresh fresh: [String]) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: directory.appendingPathComponent("model.safetensors"))
        var mapped = [String: MLXArray]()
        for (rawKey, value) in checkpoint.arrays {
            let key = rawKey.hasPrefix("wav2vec2_bert.") ? String(rawKey.dropFirst("wav2vec2_bert.".count)) : rawKey
            if key.hasPrefix("quantizer.") || key.hasPrefix("project_") || fresh.contains(where: { key.hasPrefix($0) }) {
                continue
            }
            mapped[key] = value.ndim == 3 && checkpoint.needsConvTranspose ? value.transposed(0, 2, 1) : value
        }
        let owned = Set(parameters().flattened().map(\.0))
        let missing = owned.filter { name in mapped[name] == nil && !fresh.contains { name.hasPrefix($0) } }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch("the W2V-BERT checkpoint lacks \(missing.count) parameters, starting with "
                                              + missing.sorted().prefix(3).joined(separator: ", "))
        }
        try NFKMLXWeights.apply(mapped.filter { owned.contains($0.key) }.map { ($0.key, $0.value) }, to: self, strict: false)
    }

    /// transformers' `Wav2Vec2BertPreTrainedModel._init_weights` over the adapter and the head: linear
    /// weights normal with deviation 0.02 and zero biases, layer norms at one and zero, convolution weights
    /// Kaiming-normal and biases uniform within `±√(groups / (in · kernel))`.
    func initializeAsReference(prefixes: [String]) {
        var parameters = [(String, MLXArray)]()
        for (path, module) in leafModules().flattened() where prefixes.contains(where: { path.hasPrefix($0) }) {
            if let conv = module as? Conv1d {
                let (outputs, kernel, inputs) = (conv.weight.dim(0), conv.weight.dim(1), conv.weight.dim(2))
                parameters.append((path + ".weight", MLXRandom.normal([outputs, kernel, inputs]) * (2 / Float(inputs * kernel)).squareRoot()))
                if conv.bias != nil {
                    let bound = (1 / Float(inputs * kernel)).squareRoot()
                    parameters.append((path + ".bias", MLXRandom.uniform(low: -bound, high: bound, [outputs])))
                }
            } else if let linear = module as? Linear {
                parameters.append((path + ".weight", MLXRandom.normal(linear.weight.shape) * 0.02))
                if linear.bias != nil { parameters.append((path + ".bias", MLXArray.zeros([linear.weight.dim(0)]))) }
            } else if let norm = module as? LayerNorm, let weight = norm.weight {
                parameters.append((path + ".weight", MLXArray.ones(weight.shape)))
                if norm.bias != nil { parameters.append((path + ".bias", MLXArray.zeros(weight.shape))) }
            }
        }
        update(parameters: ModuleParameters.unflattened(parameters))
    }
}

extension NFKMLXWav2Vec2Bert {

    /// Builds the network itself, ready to fine-tune.
    ///
    /// - Parameters:
    ///   - directoryURL: the release, or a directory ``save(_:tokenizer:toDirectoryURL:)`` wrote.
    ///   - vocabulary: the consumer's characters in id order, the blank first. Nil loads the directory as it
    ///     is. With a vocabulary, a directory without the adapter and a head of that size gets both fresh,
    ///     initialized as the reference initializes them, and everything else loads.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, vocabulary: [String]? = nil) throws -> NFKMLXWav2Vec2BertNet {
        let release = try NFKMLXWav2Vec2BertConfiguration(configurationURL: directoryURL.appendingPathComponent("config.json"))
        guard let vocabulary, !(release.addsAdapter && release.vocabularySize == vocabulary.count) else {
            let net = NFKMLXWav2Vec2BertNet(release)
            try net.loadWeights(fromDirectory: directoryURL, leavingFresh: [])
            return net
        }
        guard vocabulary.count >= 2 else {
            throw NFKMLXError.unsupportedConfiguration("a CTC vocabulary needs the blank and at least one character")
        }
        var configuration = release
        configuration.addsAdapter = true
        configuration.vocabularySize = vocabulary.count
        configuration.maskTimeProbability = 0
        let net = NFKMLXWav2Vec2BertNet(configuration)
        let fresh = release.addsAdapter ? ["lm_head."] : ["adapter.", "lm_head."]
        net.initializeAsReference(prefixes: fresh)
        try net.loadWeights(fromDirectory: directoryURL, leavingFresh: fresh)
        return net
    }

    /// Fine-tunes the encoder, adapter, and CTC head on a consumer's own transcribed speech, returning the
    /// loss from each step.
    ///
    /// - Parameters:
    ///   - net: the network to train; it must carry a CTC head (``network(directoryURL:vocabulary:)``).
    ///   - examples: one utterance per step: its 16 kHz samples and label ids
    ///     (``NFKMLXWav2Vec2Tokenizer/labels(for:)``).
    ///   - trainable: every parameter (the reference), the adapter and head, or the head alone.
    ///   - objective: the CTC loss, reduced by mean as the recipe sets it.
    ///   - timeMasking: whether SpecAugment masks frames at the network's `mask_time_prob`; the recipe sets
    ///     it to zero, so it is off.
    ///   - optimizer: the update rule. Nil uses the `Trainer`'s `torch.optim.AdamW` (bias-corrected, betas
    ///     0.9 and 0.999, epsilon 1e-8, no weight decay) at `learningRate`.
    ///   - learningRate: the recipe's peak rate.
    ///   - warmupSteps: the recipe's linear warm-up. A run shorter than it trains below the peak throughout.
    ///   - steps: how many utterances to train on.
    ///   - clipGradientNorm: the `Trainer`'s `max_grad_norm`.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates. 1, the
    ///     default, updates after every batch. The reference updates on 16 utterances over 2
    ///     accumulated steps.
    ///   - learningRateSchedule: nil uses the `Trainer`'s linear warm-up and decay when the reference
    ///     optimizer runs, and a constant rate with a caller's optimizer.
    ///   - seed: seeds SpecAugment's draws.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXWav2Vec2BertNet,
        examples: (Int) -> (samples: [Float], labels: [Int]),
        trainable: NFKMLXWav2Vec2BertTrainable = .all,
        objective: NFKMLXWav2Vec2Objective = NFKMLXWav2Vec2Objective(),
        timeMasking: Bool = false,
        optimizer: Optimizer? = nil,
        learningRate: Float = 5e-5,
        warmupSteps: Int = 500,
        steps: Int,
        clipGradientNorm: Float? = 1.0,
        accumulationSteps: Int = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        seed: UInt64 = 0,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        guard net.head != nil else {
            throw NFKMLXError.unsupportedConfiguration(
                "this W2V-BERT network has no CTC head; build it with network(directoryURL:vocabulary:)")
        }
        let c = net.configuration
        var generator = NFKMLXSplitMix64(seed: seed)
        var utterance = [Int]()
        return try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: learningRate, weightDecay: 0) },
            referenceSchedule: { .linearWithWarmup(steps: steps, warmupSteps: warmupSteps) },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                utterance = example.labels
                let (features, mask) = NFKMLXWav2Vec2BertProcessor.inputFeatures(example.samples)
                let timeMask = timeMasking
                    ? NFKMLXSpecAugment.timeMask(frames: features.dim(1), probability: c.maskTimeProbability,
                                                 length: c.maskTimeLength, minimumMasks: c.maskTimeMinimumMasks,
                                                 using: &generator)
                    : nil
                return [features, mask] + (timeMask.map { [$0] } ?? [])
            },
            loss: { net, arrays in
                let (hidden, outputMask) = net.encode(arrays[0], mask: arrays[1], timeMask: arrays.count > 2 ? arrays[2] : nil)
                let frames = outputMask.map { $0.asType(.int32).sum(axis: -1).asArray(Int32.self).map(Int.init) }
                return objective.loss(logits: net.head!(hidden), labels: [utterance], frames: frames)
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// Freezes everything but the parameters `trainable` names.
    static func apply(_ trainable: NFKMLXWav2Vec2BertTrainable, to net: NFKMLXWav2Vec2BertNet) {
        switch trainable {
        case .all:
            net.unfreeze()
        case .adapterAndHead:
            net.freeze()
            net.adapter?.unfreeze()
            net.head?.unfreeze()
        case .head:
            net.freeze()
            net.head?.unfreeze()
        }
    }

    /// Writes `net` as a release directory: `model.safetensors`, the `config.json` and
    /// `preprocessor_config.json` it reads back from, and, when `tokenizer` is given, `vocab.json`.
    /// ``backend(directoryURL:)`` and ``network(directoryURL:vocabulary:)`` load it.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXWav2Vec2BertNet, tokenizer: NFKMLXWav2Vec2Tokenizer?,
                            toDirectoryURL directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try NFKMLXWeights.save(net, to: directory.appendingPathComponent("model.safetensors"))
        let c = net.configuration
        var config: [String: Any] = [
            "model_type": "wav2vec2-bert", "hidden_size": c.hiddenSize, "num_hidden_layers": c.numHiddenLayers,
            "num_attention_heads": c.numAttentionHeads, "intermediate_size": c.intermediateSize,
            "feature_projection_input_dim": c.featureDimensions, "conv_depthwise_kernel_size": c.depthwiseKernel,
            "left_max_position_embeddings": c.leftPositions, "right_max_position_embeddings": c.rightPositions,
            "layer_norm_eps": Double(c.layerNormEps), "position_embeddings_type": "relative_key", "hidden_act": "swish",
            "add_adapter": c.addsAdapter, "num_adapter_layers": c.adapterLayers, "adapter_kernel_size": c.adapterKernel,
            "adapter_stride": c.adapterStride, "output_hidden_size": c.outputHiddenSize, "adapter_act": "relu",
            "pad_token_id": c.padTokenID, "mask_time_prob": Double(c.maskTimeProbability),
            "mask_time_length": c.maskTimeLength, "mask_time_min_masks": c.maskTimeMinimumMasks,
        ]
        if let vocabularySize = c.vocabularySize {
            config["vocab_size"] = vocabularySize
            config["architectures"] = ["Wav2Vec2BertForCTC"]
        } else {
            config["architectures"] = ["Wav2Vec2BertModel"]
        }
        let processor: [String: Any] = ["feature_extractor_type": "SeamlessM4TFeatureExtractor", "feature_size": 80,
                                        "num_mel_bins": 80, "padding_value": 1, "sampling_rate": 16000, "stride": 2,
                                        "return_attention_mask": true]
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
}
