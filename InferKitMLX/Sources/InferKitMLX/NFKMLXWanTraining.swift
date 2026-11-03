//
//  NFKMLXWanTraining.swift
//  InferKitMLX
//
//  Fine-tuning the Wan text-to-video transformer with LoRA as three community trainers do: finetrainers
//  (huggingface/finetrainers at 684c0db), DiffSynth-Studio (modelscope/DiffSynth-Studio at 1e8ae1d), and
//  musubi-tuner (kohya-ss/musubi-tuner at f8a1b03). Neither Wan-Video nor diffusers publishes a training
//  recipe for Wan, so each trainer is a reference of its own. The three share one objective — the
//  velocity target `noise − clean` at a noise level `σ`, the input `(1 − σ)·clean + σ·noise`, and a mean
//  squared error over every element in float32 — and differ in how they draw `σ`, the timestep the
//  transformer reads, the loss weight, the clean latent, the text features, the adapted layers, and the
//  optimizer. ``NFKMLXWanTrainingPreset`` names each trainer's settings.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// torch's float32 `linspace` on the CPU: the first half counts up from `start` and the second half down
/// from `end` by the same float32 step, each entry one fused multiply-add.
enum NFKTorchLinspace {
    static func float32(from start: Float, to end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        let halfway = count / 2
        return (0 ..< count).map { index in
            index < halfway ? start.addingProduct(step, Float(index)) : end.addingProduct(-step, Float(count - index - 1))
        }
    }
}

/// How a trainer draws the `u` on `[0, 1)` its table index or shift starts from.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXWanSigmaDensity: Sendable, Equatable {
    /// `u` uniform on `[0, 1)`.
    case uniform
    /// `u = sigmoid(n)` for `n` normal with this mean and standard deviation.
    case logitNormal(mean: Float, std: Float)
    /// SD3's mode density: `u = 1 − v − scale·(cos²(πv/2) − 1 + v)` for `v` uniform.
    case mode(scale: Float)

    /// `u` for the raw draw `raw`: the normal draw for ``logitNormal(mean:std:)`` (already scaled and
    /// shifted), the uniform draw otherwise.
    func u(_ raw: Float) -> Float {
        switch self {
        case .uniform: return raw
        case .logitNormal: return 1 / (1 + exp(-raw))
        case .mode(let scale):
            let c = cos(Float.pi * raw / 2)
            return 1 - raw - scale * (c * c - 1 + raw)
        }
    }

    func raw(count: Int, key: MLXArray) -> MLXArray {
        switch self {
        case .logitNormal(let mean, let std): return MLXRandom.normal([count], key: key) * std + mean
        case .uniform, .mode: return MLXRandom.uniform(low: 0, high: 1, [count], key: key)
        }
    }
}

/// One training draw: the noise level, the timestep the transformer reads, and the loss weight.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanTrainingDraw: Sendable, Equatable {
    public var sigma: Float
    public var timestep: Float
    public var weight: Float
}

/// How a trainer turns one random draw into ``NFKMLXWanTrainingDraw``.
///
/// @discussion Each case documents what its raw draw is, so a caller can replay a trainer's own draws.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXWanSigmaSampler: Sendable, Equatable {
    /// finetrainers: `u` from `density`, the index `⌊1000u⌋` into the unshifted table `σᵢ = (1000 − i)/1000`
    /// (diffusers' `FlowMatchEulerDiscreteScheduler`, float32), and the timestep `⌊σ·1000⌋` in float32,
    /// so `σ = 0.999` reads 998. The raw draw is the density's.
    case finetrainers(NFKMLXWanSigmaDensity)
    /// DiffSynth-Studio: an index uniform on `[⌊1000·lower⌋, ⌊1000·upper⌋)` into the Wan table
    /// `σ = shift·s/(1 + (shift − 1)s)` over `s = linspace(1, 0, 1001)[:-1]` (torch, float32), the timestep
    /// `1000σ` rounded to the transformer's type, and `σ` and the loss weight read back at the table entry
    /// nearest that timestep. The raw draw is the index. Wan 2.2's experts train on their side of a boundary
    /// (T2V: `upper` 0.417 for the high-noise expert, `lower` 0.417 for the low-noise one).
    case diffSynth(shift: Float, lower: Float, upper: Float)
    /// musubi-tuner's `sigma` sampling: `u` from `density`, the index `⌊u·(maximum − minimum) + minimum⌋`
    /// into the unshifted table `linspace(1, 0, 1001)[:-1]·1000` (float32), `σ` the table's entry at that
    /// index. `minimum` and `maximum` count table positions, so a range of 900–1000 reaches timesteps 100–1.
    /// The raw draw is the density's.
    case musubiTable(NFKMLXWanSigmaDensity, minimum: Float, maximum: Float)
    /// musubi-tuner's `shift` sampling: `t = sigmoid(scale·n)` for `n` standard normal, shifted to
    /// `shift·t/(1 + (shift − 1)t)`, stretched onto `[minimum, maximum]` (fractions), `σ = t`, and the
    /// timestep `1000t + 1`. The raw draw is `n`.
    case musubiShift(shift: Float, sigmoidScale: Float, minimum: Float, maximum: Float)

    /// The DiffSynth-Studio Wan table and its loss weights.
    static func diffSynthTable(shift: Float) -> (sigmas: [Float], timesteps: [Float], weights: [Float]) {
        let s = Array(NFKTorchLinspace.float32(from: 1, to: 0, count: 1001).dropLast())
        let sigmas = s.map { shift * $0 / (1 + (shift - 1) * $0) }
        let timesteps = sigmas.map { $0 * 1000 }
        // `exp(−2·((x − 500)/1000)²)`, shifted to a minimum of 0 and scaled to a mean of 1, with torch's float32
        // exponential and sum.
        let centered = timesteps.map { ($0 - 500) / 1000 }
        let y = NFKReferenceRounding.torchExp(MLXArray(centered.map { -2 * ($0 * $0) })).asArray(Float.self)
        let floor = y.min() ?? 0
        let shifted = y.map { $0 - floor }
        let total = NFKReferenceRounding.torchLastAxisSum(MLXArray(shifted)).item(Float.self)
        return (sigmas, timesteps, shifted.map { $0 * (1000 / total) })
    }

    /// The musubi-tuner and finetrainers unshifted table, float32.
    static let musubiTimesteps = Array(NFKTorchLinspace.float32(from: 1, to: 0, count: 1001).dropLast()).map { $0 * 1000 }
    static let musubiSigmas = NFKTorchLinspace.float32(from: 1, to: 0, count: 1001)

    /// The draw for `raw`, with the timestep rounded to `dtype` where the trainer rounds it.
    public func draw(_ raw: Float, weighting: NFKMLXWanLossWeighting = .none, dtype: DType = .float32) -> NFKMLXWanTrainingDraw {
        switch self {
        case .finetrainers(let density):
            let index = min(max(Int(density.u(raw) * 1000), 0), 999)
            let sigma = Float(Double(1000 - index) / 1000)
            let timestep = Float(Int(sigma * 1000))
            return NFKMLXWanTrainingDraw(sigma: sigma, timestep: timestep, weight: weighting.weight(sigma))
        case .diffSynth(let shift, _, _):
            let table = Self.diffSynthTable(shift: shift)
            let index = min(max(Int(raw), 0), 999)
            let timestep = MLXArray(table.timesteps[index]).asType(dtype).asType(.float32).item(Float.self)
            let nearest = table.timesteps.indices.min { abs(table.timesteps[$0] - timestep) < abs(table.timesteps[$1] - timestep) }!
            return NFKMLXWanTrainingDraw(sigma: table.sigmas[nearest], timestep: timestep,
                                         weight: weighting == .diffSynth ? table.weights[nearest] : weighting.weight(table.sigmas[nearest]))
        case .musubiTable(let density, let minimum, let maximum):
            let index = min(max(Int(density.u(raw) * (maximum - minimum) + minimum), 0), 999)
            let timestep = Self.musubiTimesteps[index]
            let sigma = Self.musubiSigmas[index]
            return NFKMLXWanTrainingDraw(sigma: sigma, timestep: timestep, weight: weighting.weight(sigma))
        case .musubiShift(let shift, let sigmoidScale, let minimum, let maximum):
            var t = 1 / (1 + exp(-raw * sigmoidScale))
            t = t * shift / (1 + (shift - 1) * t)
            t = t * (maximum - minimum) + minimum
            return NFKMLXWanTrainingDraw(sigma: t, timestep: t * 1000 + 1, weight: weighting.weight(t))
        }
    }

    /// `count` raw draws from `key`, each what ``draw(_:weighting:dtype:)`` reads.
    public func raw(count: Int, key: MLXArray) -> [Float] {
        let values: MLXArray
        switch self {
        case .finetrainers(let density), .musubiTable(let density, _, _):
            values = density.raw(count: count, key: key)
        case .diffSynth(_, let lower, let upper):
            let low = Int(lower * 1000), high = Int(upper * 1000)
            values = MLXRandom.randInt(low: low, high: max(high, low + 1), [count], key: key).asType(.float32)
        case .musubiShift:
            values = MLXRandom.normal([count], key: key)
        }
        return values.asArray(Float.self)
    }
}

/// A trainer's loss weight for a noise level.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXWanLossWeighting: Sendable, Equatable {
    case none
    /// `σ⁻²`.
    case sigmaSqrt
    /// `2 / (π(1 − 2σ + 2σ²))`.
    case cosmap
    /// DiffSynth-Studio's Gaussian over the timestep, read from its table (``NFKMLXWanSigmaSampler/diffSynth(shift:lower:upper:)``).
    case diffSynth

    func weight(_ sigma: Float) -> Float {
        switch self {
        case .none, .diffSynth: return 1
        case .sigmaSqrt: return 1 / (sigma * sigma)
        case .cosmap: return 2 / (Float.pi * (1 - 2 * sigma + 2 * sigma * sigma))
        }
    }
}

/// One training example: a clip's latents and its prompt's text features.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanTrainingExample {
    /// The normalized latent mean `[16, frames, height, width]`.
    public var latent: MLXArray
    /// The normalized latent log-variance, the same shape, for a trainer that samples the posterior.
    public var logVariance: MLXArray?
    /// The text features `[tokens, textDim]`.
    public var text: MLXArray

    public init(latent: MLXArray, logVariance: MLXArray? = nil, text: MLXArray) {
        self.latent = latent
        self.logVariance = logVariance
        self.text = text
    }
}

/// The objective a trainer trains Wan with.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanObjective: Sendable {
    public var sampler: NFKMLXWanSigmaSampler
    public var weighting: NFKMLXWanLossWeighting
    /// Whether the clean latent is a sample of the encoder's posterior, `μ + exp(½·clamp(logσ², −30, 20))·ε`
    /// (finetrainers), rather than its mean.
    public var samplesPosterior: Bool

    public init(sampler: NFKMLXWanSigmaSampler, weighting: NFKMLXWanLossWeighting, samplesPosterior: Bool) {
        self.sampler = sampler
        self.weighting = weighting
        self.samplesPosterior = samplesPosterior
    }

    /// The clean latent of `example` for the posterior draw `epsilon`.
    public func clean(_ example: NFKMLXWanTrainingExample, epsilon: MLXArray?) -> MLXArray {
        guard samplesPosterior, let logVariance = example.logVariance, let epsilon else { return example.latent }
        return example.latent + exp(0.5 * clip(logVariance, min: -30, max: 20)) * epsilon
    }

    /// The loss of one example: `weight · mean((prediction − (noise − clean))²)` in float32.
    public static func loss(prediction: MLXArray, clean: MLXArray, noise: MLXArray, weight: Float) -> MLXArray {
        let target = noise.asType(.float32) - clean.asType(.float32)
        return (prediction.asType(.float32) - target).square().mean() * weight
    }

    /// The loss of `net` on one example at `draw`, with `noise` and the posterior draw `epsilon`.
    public func score(_ net: NFKMLXWanTransformerNet, example: NFKMLXWanTrainingExample, draw: NFKMLXWanTrainingDraw,
                      noise: MLXArray, epsilon: MLXArray?) -> MLXArray {
        let dtype = NFKReferenceRounding.parameterType(of: net)
        let x0 = clean(example, epsilon: epsilon)
        let noisy = (1 - draw.sigma) * x0 + draw.sigma * noise
        let prediction = net(noisy.asType(dtype), text: example.text.asType(dtype), t: MLXArray([draw.timestep]))
        return Self.loss(prediction: prediction, clean: x0, noise: noise, weight: draw.weight)
    }
}

/// The layers a trainer adapts by default, over the text-to-video transformer's names.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXWanLoRATargets: Sendable, Equatable {
    /// The query, key, value, and output projections of both attentions in every block (finetrainers'
    /// Wan examples: `blocks.*(to_q|to_k|to_v|to_out.0)`).
    case attention
    /// Those and both feed-forward projections (DiffSynth-Studio's `q,k,v,o,ffn.0,ffn.2`; musubi-tuner's every
    /// linear layer of a block, which for text-to-video is the same set).
    case attentionAndFeedForward

    func contains(_ path: String) -> Bool {
        guard path.hasPrefix("blocks.") else { return false }
        let attention = ["to_q", "to_k", "to_v"].contains { path.hasSuffix(".\($0)") } || path.hasSuffix(".to_out.0")
        switch self {
        case .attention: return attention
        case .attentionAndFeedForward:
            return attention || path.hasSuffix(".ffn.net.0.proj") || path.hasSuffix(".ffn.net.2")
        }
    }
}

/// A trainer's settings for a Wan LoRA fine-tune: the objective, the adapted layers, and the optimizer.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanTrainingPreset: Sendable {
    public var objective: NFKMLXWanObjective
    public var targets: NFKMLXWanLoRATargets
    public var rank: Int
    public var alpha: Float
    /// Whether the text features keep the encoder's outputs at the padded positions (finetrainers encodes
    /// the padded prompt with no mask); otherwise they are zeros past the prompt, as the pipeline reads them.
    public var keepsPaddedText: Bool
    public var learningRate: Float
    public var betas: (Float, Float)
    public var weightDecay: Float
    public var warmupSteps: Int
    /// Whether the warm-up's first update runs at a rate of 0 (`step / warmup`) rather than `1 / warmup`
    /// (`(step + 1) / warmup`).
    public var warmupStartsAtZero: Bool
    public var clipGradientNorm: Float?
    /// The file layout ``NFKMLXWanVideoGenerator/saveAdapters(of:to:preset:)`` writes.
    public var format: NFKMLXWanLoRAFormat

    /// finetrainers' Wan 2.1 text-to-video example (`examples/training/sft/wan/crush_smol_lora`): `σ` from a
    /// logit-normal (0, 1) over the unshifted table, no loss weight, the posterior sampled, rank and alpha 32
    /// on the attention projections, AdamW at 5e-5 with betas (0.9, 0.99) and weight decay 1e-4, a 1000-step
    /// warm-up from 0, and the gradient norm clipped to 1.
    public static let finetrainers = NFKMLXWanTrainingPreset(
        objective: .init(sampler: .finetrainers(.logitNormal(mean: 0, std: 1)), weighting: .none, samplesPosterior: true),
        targets: .attention, rank: 32, alpha: 32, keepsPaddedText: true, learningRate: 5e-5, betas: (0.9, 0.99),
        weightDecay: 1e-4, warmupSteps: 1000, warmupStartsAtZero: true, clipGradientNorm: 1, format: .diffusers)

    /// DiffSynth-Studio's Wan LoRA scripts: an index uniform over the whole shift-5 table, its Gaussian loss
    /// weight, the posterior mean, rank and alpha 32 on the attention and feed-forward projections, AdamW at
    /// 1e-4 with weight decay 0.01, a 5-step warm-up, and no clipping.
    public static let diffSynth = NFKMLXWanTrainingPreset(
        objective: .init(sampler: .diffSynth(shift: 5, lower: 0, upper: 1), weighting: .diffSynth, samplesPosterior: false),
        targets: .attentionAndFeedForward, rank: 32, alpha: 32, keepsPaddedText: false, learningRate: 1e-4,
        betas: (0.9, 0.999), weightDecay: 0.01, warmupSteps: 5, warmupStartsAtZero: false, clipGradientNorm: nil,
        format: .diffSynth)

    /// musubi-tuner's documented Wan recipe (`docs/wan.md`): `shift` sampling at 3.0, no loss weight, the
    /// posterior mean, rank 32 at alpha 1 on every linear layer of the blocks, AdamW at 2e-4 (the recipe
    /// names `adamw8bit`; this runs float32 AdamW with torch's defaults), a constant rate, and the gradient
    /// norm clipped to 1.
    public static let musubi = NFKMLXWanTrainingPreset(
        objective: .init(sampler: .musubiShift(shift: 3, sigmoidScale: 1, minimum: 0, maximum: 1), weighting: .none,
                         samplesPosterior: false),
        targets: .attentionAndFeedForward, rank: 32, alpha: 1, keepsPaddedText: false, learningRate: 2e-4,
        betas: (0.9, 0.999), weightDecay: 0.01, warmupSteps: 0, warmupStartsAtZero: false, clipGradientNorm: 1,
        format: .musubi)

    public init(objective: NFKMLXWanObjective, targets: NFKMLXWanLoRATargets, rank: Int, alpha: Float,
                keepsPaddedText: Bool, learningRate: Float, betas: (Float, Float), weightDecay: Float, warmupSteps: Int,
                warmupStartsAtZero: Bool, clipGradientNorm: Float?, format: NFKMLXWanLoRAFormat) {
        self.objective = objective
        self.targets = targets
        self.rank = rank
        self.alpha = alpha
        self.keepsPaddedText = keepsPaddedText
        self.learningRate = learningRate
        self.betas = betas
        self.weightDecay = weightDecay
        self.warmupSteps = warmupSteps
        self.warmupStartsAtZero = warmupStartsAtZero
        self.clipGradientNorm = clipGradientNorm
        self.format = format
    }

    /// The preset's learning-rate multiplier for a zero-based update.
    var schedule: NFKMLXLearningRateSchedule {
        let warmup = warmupSteps, fromZero = warmupStartsAtZero
        guard warmup > 0 else { return .constant }
        return NFKMLXLearningRateSchedule { step in
            min(1, Float(fromZero ? step : step + 1) / Float(warmup))
        }
    }
}

/// The layout a trainer saves its adapters in. Each loads back through ``NFKMLXWanLoRA``.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXWanLoRAFormat: Sendable, Equatable {
    /// diffusers' PEFT keys (`transformer.<path>.lora_A.weight`), the rank and alpha under `lora_config`,
    /// as finetrainers saves them.
    case diffusers
    /// The original Wan names under PEFT's adapter level, `blocks.<n>.self_attn.q.lora_A.default.weight`,
    /// the scaling folded into the up weight, as DiffSynth-Studio saves them.
    case diffSynth
    /// musubi-tuner's `lora_unet_<name>.lora_down.weight` / `.lora_up.weight` with a scalar `.alpha`.
    case musubi
}

extension NFKMLXWanVideoGenerator {

    /// Builds the transformer of a text-to-video release directory, ready to fine-tune, at its stored type.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL) throws -> NFKMLXWanTransformerNet {
        let directory = directoryURL.appendingPathComponent("transformer")
        let net = NFKMLXWanTransformerNet(try NFKMLXWanRelease.transformerConfiguration(
            fromHuggingFace: directory.appendingPathComponent("config.json")))
        try NFKMLXWanRelease.loadTransformer(into: net, fromDirectory: directory, precision: .checkpoint)
        return net
    }

    /// Fine-tunes the text-to-video transformer with LoRA as `preset`'s trainer does, returning each step's
    /// loss.
    ///
    /// The customization path is four calls: ``trainingExample(video:prompt:preset:)`` for each clip,
    /// ``network(directoryURL:)`` to build, this to train, and ``saveAdapters(of:to:preset:)`` to write the
    /// adapters in the trainer's own layout, which a generator applies with ``addAdapter(at:scale:)``.
    ///
    /// - Parameters:
    ///   - net: the transformer to train; the adapters persist on it.
    ///   - examples: one batch per step; the batch's loss is the mean of its examples' losses.
    ///   - preset: the trainer whose objective, targets, and optimizer the run follows.
    ///   - optimizer: nil uses torch's AdamW at the preset's rate, betas, and weight decay.
    ///   - steps: how many updates to run.
    ///   - seed: seeds the draws, the noise, and the posterior samples.
    ///   - accumulationSteps: how many batches each update averages.
    ///   - learningRateSchedule: nil uses the preset's warm-up with the reference optimizer.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The trainers run bfloat16 weights under autocast; this runs in the type the weights load at. A step is
    /// multi-second; call it off the render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXWanTransformerNet,
        examples: (Int) -> [NFKMLXWanTrainingExample],
        preset: NFKMLXWanTrainingPreset = .finetrainers,
        optimizer: Optimizer? = nil,
        steps: Int,
        seed: UInt64 = 0,
        accumulationSteps: Int = 1,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        var key = MLXRandom.key(seed)
        var batch = [NFKMLXWanTrainingExample]()
        var draws = [NFKMLXWanTrainingDraw]()
        let objective = preset.objective
        let dtype = NFKReferenceRounding.parameterType(of: net)
        return try NFKMLXFineTune.run(
            net,
            freezing: {
                try NFKMLXLoRA.apply(to: net, rank: preset.rank, alpha: preset.alpha) { path, _ in
                    preset.targets.contains(path)
                }
            },
            optimizer: optimizer,
            reference: {
                NFKMLXReferenceOptimizers.adamW(learningRate: preset.learningRate, betas: preset.betas,
                                                weightDecay: preset.weightDecay)
            },
            referenceSchedule: { preset.schedule },
            steps: steps,
            arrays: { step in
                batch = examples(step)
                let keys = MLXRandom.split(key: key, into: 4)
                key = keys[0]
                draws = objective.sampler.raw(count: batch.count, key: keys[1]).map {
                    objective.sampler.draw($0, weighting: objective.weighting, dtype: dtype)
                }
                let noise = batch.enumerated().map { index, example in
                    MLXRandom.normal(example.latent.shape, key: MLXRandom.split(key: keys[2], into: batch.count)[index])
                }
                let epsilon = batch.enumerated().map { index, example in
                    MLXRandom.normal(example.latent.shape, key: MLXRandom.split(key: keys[3], into: batch.count)[index])
                }
                return noise + epsilon
            },
            loss: { net, arrays in
                let count = batch.count
                let losses = (0 ..< count).map { index in
                    objective.score(net, example: batch[index], draw: draws[index], noise: arrays[index],
                                    epsilon: arrays[count + index])
                }
                return MLX.stacked(losses).mean()
            },
            clipGradientNorm: preset.clipGradientNorm, accumulationSteps: accumulationSteps,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }

    /// Writes the adapters `net` carries in `preset`'s trainer's layout.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func saveAdapters(of net: NFKMLXWanTransformerNet, to url: URL,
                                    preset: NFKMLXWanTrainingPreset = .finetrainers) throws {
        var arrays = [String: MLXArray]()
        var metadata = [String: String]()
        var rank = 0
        for (path, layer) in net.leafModules().flattened() {
            guard let adapter = layer as? NFKMLXLoRALinear else { continue }
            // Module `[in, rank]` and `[rank, out]` are PEFT's `lora_A [rank, in]` and `lora_B [out, rank]`.
            let down = adapter.loraA.transposed(1, 0), up = adapter.loraB.transposed(1, 0)
            rank = down.dim(0)
            switch preset.format {
            case .diffusers:
                arrays["transformer.\(path).lora_A.weight"] = down
                arrays["transformer.\(path).lora_B.weight"] = up
            case .diffSynth:
                let name = try originalName(path)
                arrays["\(name).lora_A.default.weight"] = down
                arrays["\(name).lora_B.default.weight"] = up * adapter.scale
            case .musubi:
                let name = "lora_unet_" + (try originalName(path)).replacingOccurrences(of: ".", with: "_")
                arrays["\(name).lora_down.weight"] = down
                arrays["\(name).lora_up.weight"] = up
                arrays["\(name).alpha"] = MLXArray(adapter.scale * Float(down.dim(0)))
            }
        }
        if preset.format == .diffusers {
            let configuration: [String: Any] = ["r": rank, "lora_alpha": preset.alpha, "init_lora_weights": true]
            metadata["lora_config"] = String(decoding: try JSONSerialization.data(withJSONObject: configuration,
                                                                                    options: [.sortedKeys]), as: UTF8.self)
            metadata["format"] = "pt"
        }
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
    }

    /// The original Wan name of a text-to-video module path, ``NFKMLXWanLoRA/textToVideoPath(original:)``
    /// inverted for the blocks' layers.
    static func originalName(_ path: String) throws -> String {
        let parts = path.split(separator: ".").map(String.init)
        guard parts.count >= 4, parts[0] == "blocks" else {
            throw NFKMLXError.unsupportedConfiguration("\(path) is not a block layer a trainer's layout names")
        }
        let block = "blocks.\(parts[1])"
        let rest = parts[2...].joined(separator: ".")
        let names = ["attn1.to_q": "self_attn.q", "attn1.to_k": "self_attn.k", "attn1.to_v": "self_attn.v",
                     "attn1.to_out.0": "self_attn.o", "attn2.to_q": "cross_attn.q", "attn2.to_k": "cross_attn.k",
                     "attn2.to_v": "cross_attn.v", "attn2.to_out.0": "cross_attn.o", "ffn.net.0.proj": "ffn.0",
                     "ffn.net.2": "ffn.2"]
        guard let name = names[rest] else {
            throw NFKMLXError.unsupportedConfiguration("\(path) has no original Wan name")
        }
        return "\(block).\(name)"
    }
}
