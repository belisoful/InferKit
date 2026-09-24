//
//  NFKMLXOpenJevTraining.swift
//  InferKitMLX
//
//  Customizing Open-Jev the way the release was trained: the LoRA adapter and the head move, the base
//  weights stay as released, and the objective is the soft-target cross-entropy over a question's
//  candidates plus 0.1 times their Brier score. A tuned model saves in the release's own layout, so the
//  same factory reloads it.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/// One labeled question about a state, with the target distribution over its candidates.
public struct NFKMLXOpenJevExample {
    public var state: Any
    public var question: NFKDecisionQuestion
    /// The target over the candidates in order, summing to one; a noul's is `[1 - y, y]`.
    public var target: [Float]

    public init(state: Any, question: NFKDecisionQuestion, target: [Float]) {
        self.state = state
        self.question = question
        self.target = target
    }

    /// A one-hot example from the index of the right candidate.
    public init(state: Any, question: NFKDecisionQuestion, label: Int) {
        let count = question.type == .noul ? 2 : question.options.count
        self.init(state: state, question: question, target: (0 ..< count).map { $0 == label ? 1 : 0 })
    }

    /// A noul and whether its statement holds.
    public init(state: Any, question: NFKDecisionQuestion, holds: Bool) {
        self.init(state: state, question: question, target: holds ? [0, 1] : [1, 0])
    }
}

/// The loader's training loss for one record: `-Σ t·log softmax(z) + brierWeight·Σ (softmax(z) - t)²`
/// over the raw candidate logits `z`, before the temperature.
public struct NFKMLXOpenJevObjective: Sendable {
    /// The Brier term's weight; both releases train at 0.1.
    public var brierWeight: Float

    public init(brierWeight: Float = 0.1) {
        self.brierWeight = brierWeight
    }

    public func loss(logits: MLXArray, target: MLXArray) -> MLXArray {
        let logProbabilities = logits - logSumExp(logits, axis: -1, keepDims: true)
        let difference = exp(logProbabilities) - target
        return -(target * logProbabilities).sum() + brierWeight * (difference * difference).sum()
    }
}

extension NFKMLXOpenJev {

    /// Freezes everything except the adapter and the head, which is what the release trains.
    public static func freeze(_ net: NFKMLXOpenJevNet) {
        net.freeze()
        net.decoder.unfreeze(keys: ["lora_a", "lora_b"])
        net.head.unfreeze()
    }

    /// Fine-tunes the adapter and the head, `batchSize` records a step in order, cycling.
    ///
    /// - Parameters:
    ///   - examples: labeled questions; each target has one entry per candidate.
    ///   - steps: how many optimizer steps to take.
    ///   - learningRate: the adapter's AdamW rate; the 2B and 9B releases train at 5e-5, the 27B at 2e-5.
    ///   - headLearningRate: the head's AdamW rate; the 2B and 9B releases train at 1e-4, the 27B at 5e-5.
    ///   - objective: the loader's loss.
    ///   - batchSize: records averaged into one step, the loader's gradient accumulation (4).
    ///   - learningRateSchedule: multiplies both rates at each step. Nil is the loader's constant rate.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Both optimizers are AdamW with weight decay 0.01, betas (0.9, 0.999), and bias correction, as
    /// the loader's are, and the gradient norm is clipped to 1. The base must be loaded at `.float32`:
    /// a half-precision adapter cannot hold the small steps a fine-tune takes. The temperature is not
    /// refitted. Save the result with ``save(to:)``. A run is multi-second; call it off the main thread.
    @discardableResult
    public func fineTune(examples: [NFKMLXOpenJevExample], steps: Int, learningRate: Float = 5e-5,
                         headLearningRate: Float = 1e-4, objective: NFKMLXOpenJevObjective = NFKMLXOpenJevObjective(),
                         batchSize: Int = 4, learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
                         observer: NFKMLXTrainer.Observer? = nil) throws -> [Float] {
        guard net.decoder.model.embedTokens.weight.dtype == .float32 else {
            throw NFKMLXError.trainingDataMismatch("load the base at .float32 to fine-tune it")
        }
        guard !examples.isEmpty, batchSize > 0 else { throw NFKMLXError.trainingDataMismatch("no examples") }
        let encoded = try examples.map { example -> (candidates: [[Int]], type: NFKDecisionType, target: MLXArray) in
            let candidates = try candidateTokens(state: example.state, question: example.question)
            let count = example.question.type == .noul ? 2 : candidates.count
            guard example.target.count == count else {
                throw NFKMLXError.trainingDataMismatch(
                    "a target of \(example.target.count) entries does not match the question's \(count) candidates")
            }
            return (candidates, example.question.type, MLXArray(example.target))
        }
        Self.freeze(net)
        let optimizer = NFKMLXReferenceOptimizers.adamW(learningRate: learningRate, over: net) { key in
            (rateScale: key.hasPrefix("head.") ? headLearningRate / learningRate : 1, weightDecay: 0.01)
        }
        var current = 0
        return try NFKMLXTrainer.train(
            net, optimizer: optimizer, steps: steps,
            sample: { step in current = step; return MLXArray(Int32(step)) },
            loss: { model, _ in
                let records = (0 ..< batchSize).map { encoded[(current * batchSize + $0) % encoded.count] }
                let losses = records.map { record in
                    objective.loss(logits: model.logits(candidates: record.candidates, type: record.type), target: record.target)
                }
                return stacked(losses).mean()
            },
            clipGradientNorm: 1, learningRateSchedule: learningRateSchedule ?? .constant, observer: observer)
    }

    /// Writes the model as a release `checkpoint` directory: the adapter in PEFT's layout
    /// (`adapter/adapter_model.safetensors`, `adapter/adapter_config.json`), the head as
    /// `head.safetensors`, and `model.json` and `temperature.json`. The same base reloads it through
    /// ``openJev(checkpointDirectoryURL:baseDirectoryURL:)``.
    @objc(saveToDirectoryURL:error:)
    public func save(to directory: URL) throws {
        let adapterDirectory = directory.appendingPathComponent("adapter")
        try FileManager.default.createDirectory(at: adapterDirectory, withIntermediateDirectories: true)
        var adapter = [String: MLXArray]()
        for (path, module) in net.decoder.leafModules().flattened() {
            guard let layer = module as? NFKMLXLoRALinear, path.hasPrefix("model.") else { continue }
            let reference = "base_model.model." + path.dropFirst("model.".count)
            adapter[reference + ".lora_A.weight"] = layer.loraA.transposed().asType(.float32)
            adapter[reference + ".lora_B.weight"] = layer.loraB.transposed().asType(.float32)
        }
        try MLX.save(arrays: adapter, metadata: ["format": "pt"],
                     url: adapterDirectory.appendingPathComponent("adapter_model.safetensors"))
        try MLX.save(arrays: ["weight": net.head.weight, "bias": net.head.bias!],
                     url: directory.appendingPathComponent("head.safetensors"))
        let c = configuration
        let files: [(String, [String: Any])] = [
            ("adapter/adapter_config.json", ["peft_type": "LORA", "r": c.loraRank, "lora_alpha": c.loraAlpha,
                                             "lora_dropout": 0.0, "bias": "none", "target_modules": c.targetModules,
                                             "base_model_name_or_path": ""]),
            ("model.json", ["model_id": c.baseModel, "revision": c.baseRevision, "max_length": c.maxLength,
                            "lora_rank": c.loraRank, "method": "independent_candidate_lora_nll_brier"]),
            ("temperature.json", ["temperature": Double(c.temperature)]),
        ]
        for (name, object) in files {
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
                .write(to: directory.appendingPathComponent(name))
        }
    }
}
