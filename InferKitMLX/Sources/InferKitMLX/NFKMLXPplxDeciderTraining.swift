//
//  NFKMLXPplxDeciderTraining.swift
//  InferKitMLX
//
//  Customizing pplx-decider: a LoRA adapter over the text model's attention and recurrence projections
//  and the readout move, the backbone stays as released. The release publishes no training code; its
//  model card says the Flash release learned the 27B decision models' full probability distributions.
//  The objective here is that distillation's: the cross-entropy of each question's logits over its
//  options against a target distribution, which differs from the KL divergence to the target by the
//  target's own entropy. A tuned model saves an adapter directory that reloads over the same release.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/// One labeled request: a state, its questions in the order they are asked, and a target distribution
/// over each question's options. Introduced in InferKit 0.4.0.
public struct NFKMLXPplxDeciderExample {
    public var state: Any
    public var questions: [NFKDecisionQuestion]
    /// Each question's target over its options in order, summing to one; a noul's is `[1 - y, y]`.
    public var targets: [[Float]]

    public init(state: Any, questions: [NFKDecisionQuestion], targets: [[Float]]) {
        self.state = state
        self.questions = questions
        self.targets = targets
    }

    /// An example with one-hot targets from the index of each question's right option (1 for a noul
    /// that holds).
    public init(state: Any, questions: [NFKDecisionQuestion], labels: [Int]) {
        let counts = questions.map { NFKMLXPplxDeciderPrompt.optionTexts($0).count }
        self.init(state: state, questions: questions,
                  targets: zip(counts, labels).map { count, label in (0 ..< count).map { $0 == label ? 1 : 0 } })
    }
}

/// The distillation objective over one request: the mean over its questions of
/// `-Σ t·log softmax(z)` on the raw logits `z` over the question's options, before the temperature.
/// Introduced in InferKit 0.4.0.
public struct NFKMLXPplxDeciderObjective: Sendable {

    public init() {}

    public func loss(logits: [MLXArray], targets: [MLXArray]) -> MLXArray {
        let terms = zip(logits, targets).map { z, t in
            -(t * (z - logSumExp(z, axis: -1, keepDims: true))).sum()
        }
        return stacked(terms).mean()
    }
}

/// The LoRA adapter a pplx-decider network carries. Introduced in InferKit 0.4.0.
public struct NFKMLXPplxDeciderAdapter: Sendable {
    public var rank: Int
    public var alpha: Float
    /// The projection names the adapter covers.
    public var targetModules: [String]

    /// The attention projections and the recurrence's `in_proj_qkv` and `out_proj`: the set the adapter
    /// released for this decoder (Open-Jev on Qwen3.5) covers, since PEFT names no default for `qwen3_5`.
    public static let defaultTargetModules = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]

    public init(rank: Int = 8, alpha: Float = 16, targetModules: [String] = defaultTargetModules) {
        self.rank = rank
        self.alpha = alpha
        self.targetModules = targetModules
    }
}

extension NFKMLXPplxDeciderNet {

    /// Adapts the target projections of the text model with LoRA and freezes the rest. A fresh
    /// adapter's zero `B` leaves every readout as the release gives it.
    public func adapt(_ adapter: NFKMLXPplxDeciderAdapter = NFKMLXPplxDeciderAdapter()) throws {
        let targets = Set(adapter.targetModules)
        try NFKMLXLoRA.apply(to: decoder, rank: adapter.rank, alpha: adapter.alpha) { path, _ in
            path.hasPrefix("model.layers.") && targets.contains(String(path.split(separator: ".").last ?? ""))
        }
    }

    /// The rank, scale, and targets of the adapter the network carries, or nil without one.
    public var adapter: NFKMLXPplxDeciderAdapter? {
        let adapted = decoder.leafModules().flattened().compactMap { path, module in
            (module as? NFKMLXLoRALinear).map { (String(path.split(separator: ".").last ?? ""), $0) }
        }
        guard let first = adapted.first?.1 else { return nil }
        return NFKMLXPplxDeciderAdapter(rank: first.loraA.dim(1), alpha: first.scale * Float(first.loraA.dim(1)),
                                        targetModules: Array(Set(adapted.map(\.0))).sorted())
    }
}

extension NFKMLXPplxDecider {

    /// Freezes everything except the adapter and the readout.
    public static func freeze(_ net: NFKMLXPplxDeciderNet) {
        net.freeze()
        net.decoder.unfreeze(keys: ["lora_a", "lora_b"])
        net.readout.unfreeze()
    }

    /// Fine-tunes the adapter and the readout, `batchSize` requests a step in order, cycling.
    ///
    /// - Parameters:
    ///   - examples: labeled requests; each target has one entry per option of its question.
    ///   - steps: how many optimizer steps to take.
    ///   - adapter: the adapter to start where the network carries none.
    ///   - learningRate: the adapter's AdamW rate.
    ///   - readoutLearningRate: the readout's AdamW rate.
    ///   - batchSize: requests averaged into one step.
    ///   - learningRateSchedule: multiplies both rates at each step. Nil keeps them constant.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Both optimizers are PyTorch's AdamW (weight decay 0.01, betas 0.9 and 0.999, bias correction), and
    /// the gradient norm is clipped to 1. The release names no recipe, so these defaults are Open-Jev's on
    /// the same decoder. The model must be loaded at `.float32`: a half-precision adapter cannot hold the
    /// small steps a fine-tune takes. The temperature is not refitted. Save the result with
    /// ``saveAdapter(to:)``. A run is multi-second; call it off the main thread.
    @discardableResult
    public func fineTune(examples: [NFKMLXPplxDeciderExample], steps: Int,
                         adapter: NFKMLXPplxDeciderAdapter = NFKMLXPplxDeciderAdapter(), learningRate: Float = 5e-5,
                         readoutLearningRate: Float = 1e-4, batchSize: Int = 4,
                         learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
                         observer: NFKMLXTrainer.Observer? = nil) throws -> [Float] {
        guard net.decoder.model.norm.weight.dtype == .float32, !(net.decoder.model.embedTokens is QuantizedEmbedding) else {
            throw NFKMLXError.trainingDataMismatch("load the model at .float32, unpacked, to fine-tune it")
        }
        guard !examples.isEmpty, batchSize > 0 else { throw NFKMLXError.trainingDataMismatch("no examples") }
        let encoded = try examples.map { example -> (sequence: NFKMLXPplxDeciderSequence, targets: [MLXArray]) in
            let sequence = try encode(state: example.state, questions: example.questions)
            guard example.targets.map(\.count) == sequence.optionCounts else {
                throw NFKMLXError.trainingDataMismatch(
                    "targets of \(example.targets.map(\.count)) entries do not match the questions' \(sequence.optionCounts) options")
            }
            return (sequence, example.targets.map { MLXArray($0) })
        }
        if net.adapter == nil {
            try net.adapt(adapter)
        }
        let net = self.net
        let objective = NFKMLXPplxDeciderObjective()
        var current = 0
        return try NFKMLXFineTune.run(
            net,
            freezing: { Self.freeze(net) },
            optimizer: nil,
            reference: {
                NFKMLXReferenceOptimizers.adamW(learningRate: learningRate, over: net) { key in
                    (rateScale: key.hasPrefix("readout.") ? readoutLearningRate / learningRate : 1, weightDecay: 0.01)
                }
            },
            referenceSchedule: { .constant },
            steps: steps,
            sample: { step in current = step; return MLXArray(Int32(step)) },
            loss: { model, _ in
                let records = (0 ..< batchSize).map { encoded[(current * batchSize + $0) % encoded.count] }
                return stacked(records.map { objective.loss(logits: model.logits($0.sequence), targets: $0.targets) }).mean()
            },
            clipGradientNorm: 1, learningRateSchedule: learningRateSchedule, observer: observer)
    }

    /// The module key a PEFT adapter key maps to. The keys are PEFT's over `Qwen3_5Model`
    /// (`base_model.model.language_model.layers…`).
    static func adapterKey(forReference key: String) -> String? {
        let prefix = "base_model.model.language_model."
        guard key.hasPrefix(prefix) else { return nil }
        let path = key.dropFirst(prefix.count)
        for (suffix, name) in [(".lora_A.weight", ".lora_a"), (".lora_B.weight", ".lora_b")] where path.hasSuffix(suffix) {
            return "decoder.model." + path.dropLast(suffix.count) + name
        }
        return nil
    }

    /// Writes the adapter and the readout: `adapter_model.safetensors` and `adapter_config.json` in
    /// PEFT's layout, `readout.safetensors`, and `decision_config.json` with the temperature. The release
    /// directory it was tuned over reloads it through ``decider(directoryURL:adapterDirectoryURL:precision:)``.
    @objc(saveAdapterToDirectoryURL:error:)
    public func saveAdapter(to directory: URL) throws {
        guard let adapter = net.adapter else {
            throw NFKMLXError.unsupportedConfiguration("the network carries no adapter to save")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var tensors = [String: MLXArray]()
        for (path, module) in net.decoder.leafModules().flattened() {
            guard let layer = module as? NFKMLXLoRALinear, path.hasPrefix("model.") else { continue }
            let reference = "base_model.model.language_model." + path.dropFirst("model.".count)
            tensors[reference + ".lora_A.weight"] = layer.loraA.transposed().asType(.float32)
            tensors[reference + ".lora_B.weight"] = layer.loraB.transposed().asType(.float32)
        }
        try MLX.save(arrays: tensors, metadata: ["format": "pt"], url: directory.appendingPathComponent("adapter_model.safetensors"))
        try MLX.save(arrays: ["weight": net.readout.weight.asType(.float32)], url: directory.appendingPathComponent("readout.safetensors"))
        let c = configuration
        let files: [(String, [String: Any])] = [
            ("adapter_config.json", ["peft_type": "LORA", "r": adapter.rank, "lora_alpha": adapter.alpha, "lora_dropout": 0.0,
                                     "bias": "none", "target_modules": adapter.targetModules,
                                     "base_model_name_or_path": c.baseModel]),
            ("decision_config.json", ["temperature": temperature, "codes": c.codes, "token_ids": c.codeTokenIds,
                                      "attention_mode": "noncausal_full_attention", "pooling": "last",
                                      "base_model": c.baseModel, "revision": c.baseRevision]),
        ]
        for (name, object) in files {
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
                .write(to: directory.appendingPathComponent(name))
        }
    }

    /// Builds the model from a release directory with a saved adapter and readout over it.
    @objc(deciderWithDirectoryURL:adapterDirectoryURL:precision:error:)
    public static func decider(directoryURL: URL, adapterDirectoryURL: URL,
                               precision: NFKMLXWeightPrecision) throws -> NFKMLXPplxDecider {
        let model = try decider(directoryURL: directoryURL, precision: precision)
        if let temperature = try loadAdapter(into: model.net, adapterDirectoryURL: adapterDirectoryURL) {
            model.temperature = temperature
        }
        return model
    }

    /// Installs a saved adapter and readout on `net` at the decoder's type, and returns the temperature
    /// the adapter directory saves.
    @discardableResult
    static func loadAdapter(into net: NFKMLXPplxDeciderNet, adapterDirectoryURL: URL) throws -> Double? {
        let configurationURL = adapterDirectoryURL.appendingPathComponent("adapter_config.json")
        let peft = try JSONSerialization.jsonObject(with: Data(contentsOf: configurationURL)) as? [String: Any] ?? [:]
        let rank = (peft["r"] as? NSNumber)?.intValue ?? 8
        try net.adapt(NFKMLXPplxDeciderAdapter(
            rank: rank, alpha: (peft["lora_alpha"] as? NSNumber)?.floatValue ?? Float(2 * rank),
            targetModules: peft["target_modules"] as? [String] ?? NFKMLXPplxDeciderAdapter.defaultTargetModules))
        let dtype = net.decoder.model.norm.weight.dtype
        var mapped = [(String, MLXArray)]()
        for (key, value) in try NFKMLXWeights.materializedCheckpoint(
            url: adapterDirectoryURL.appendingPathComponent("adapter_model.safetensors")).arrays {
            guard let name = adapterKey(forReference: key) else {
                throw NFKMLXError.weightsMismatch("the adapter carries \(key), which no adapted projection takes")
            }
            mapped.append((name, value.transposed().asType(dtype)))
        }
        try NFKMLXOpenJev.applyPart(mapped, to: net) { $0.hasSuffix(".lora_a") || $0.hasSuffix(".lora_b") }
        try loadReadout(into: net, from: adapterDirectoryURL.appendingPathComponent("readout.safetensors"), dtype: dtype)
        let decision = try? JSONSerialization.jsonObject(
            with: Data(contentsOf: adapterDirectoryURL.appendingPathComponent("decision_config.json"))) as? [String: Any]
        guard let temperature = (decision?["temperature"] as? NSNumber)?.doubleValue, temperature.isFinite, temperature > 0 else {
            return nil
        }
        return temperature
    }
}
