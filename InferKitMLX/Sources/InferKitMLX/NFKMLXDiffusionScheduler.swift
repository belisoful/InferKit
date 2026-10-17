//
//  NFKMLXDiffusionScheduler.swift
//  InferKitMLX
//

import Foundation
import MLX

/// What a diffusion model predicts at each step, which fixes how the scheduler recovers the clean
/// latent and the noise from that prediction.
///
/// - `epsilon`: the added noise (SD 1.x/2.x, SDXL).
/// - `vPrediction`: the velocity (SD 2.x-v, some fine-tunes).
/// - `sample`: the clean latent directly.
public enum NFKDiffusionPredictionType: Sendable {
    case epsilon
    case vPrediction
    case sample
}

/// One entry in an inference schedule: where it sits in the loop, its point on the training noise
/// schedule, and the cumulative signal ratio (`alphaBar`) there and at the next lower-noise step.
public struct NFKDiffusionTimestep: Sendable {
    /// Position in the inference schedule, `0` at the highest-noise step.
    public let index: Int
    /// Index into the training schedule (`0 ..< trainingSteps`).
    public let train: Int
    /// Cumulative product of `1 - beta` up to this step (the signal ratio, in `0...1`).
    public let alphaBar: Float
    /// `alphaBar` at the next lower-noise step; `1` at the final step, where the latent is clean.
    public let alphaBarPrev: Float

    public init(index: Int, train: Int, alphaBar: Float, alphaBarPrev: Float) {
        self.index = index
        self.train = train
        self.alphaBar = alphaBar
        self.alphaBarPrev = alphaBarPrev
    }
}

/// The sampler a diffusion backend iterates. It owns the noise schedule and the per-step update; the
/// backend owns the loop, the model forward, and the image bridge. A model whose training needs a
/// different sampler (flow matching for FLUX/SD3, an ancestral Euler variant) adopts this protocol.
public protocol NFKDiffusionScheduler: Sendable {
    /// The inference schedule for `count` steps, ordered highest-noise first.
    func steps(_ count: Int) -> [NFKDiffusionTimestep]

    /// The starting latent for text-to-image, given unit-variance `noise` and the first step.
    func initialLatent(noise: MLXArray, first: NFKDiffusionTimestep) -> MLXArray

    /// Advances one step: the model's `prediction` on `latent` at `timestep` becomes the next latent.
    func step(prediction: MLXArray, timestep: NFKDiffusionTimestep, latent: MLXArray) -> MLXArray

    /// Noises a clean latent to `timestep`. Used to start image-to-image and inpainting, and to keep
    /// the known region consistent across an inpaint loop.
    func addNoise(clean: MLXArray, noise: MLXArray, timestep: NFKDiffusionTimestep) -> MLXArray

    /// This sampler drawing its own randomness from `seed`, for a sampler that draws any (the latent
    /// consistency sampler re-noises every non-final step). The backend calls it with the request's
    /// seed before a run. A deterministic sampler returns itself, which is the default.
    /// Introduced in InferKit 0.4.0.
    func seeded(_ seed: UInt64) -> any NFKDiffusionScheduler
}

public extension NFKDiffusionScheduler {
    func seeded(_ seed: UInt64) -> any NFKDiffusionScheduler { self }
}

/// How an inference schedule walks the training range.
///
/// - `leading`: `0, stride, 2·stride, …`, shifted by `stepsOffset`. The Stable Diffusion 1.x and 2.x
///   releases sample this way.
/// - `trailing`: `trainingSteps, trainingSteps − stride, …`, each less one, and no offset. The
///   distilled few-step releases need it — at one step, leading visits training step 1, where almost
///   no noise remains and the model has nothing to remove.
public enum NFKDiffusionTimestepSpacing: Sendable {
    case leading
    case trailing
}

/// A deterministic DDIM sampler (η = 0) over a scaled-linear beta schedule, the default for the
/// Stable Diffusion family. It supports the three common prediction types; a real integration picks
/// the type its weights were trained with.
public struct NFKDDIMScheduler: NFKDiffusionScheduler {

    public let trainingSteps: Int
    public let predictionType: NFKDiffusionPredictionType
    private let alphaBars: [Float]
    private let stepsOffset: Int
    private let finalAlphaBar: Float
    private let spacing: NFKDiffusionTimestepSpacing

    /// - Parameters:
    ///   - trainingSteps: The training schedule length (SD uses 1000).
    ///   - betaStart: The scaled-linear beta schedule's start endpoint (SD default).
    ///   - betaEnd: The scaled-linear beta schedule's end endpoint (SD default).
    ///   - predictionType: What the model predicts each step.
    ///   - stepsOffset: Added to every training step the schedule visits. The released Stable
    ///     Diffusion scheduler configurations set this to 1, which shifts the whole schedule.
    ///   - setsAlphaToOne: When true the final step denoises against a signal ratio of 1; when false
    ///     (the released Stable Diffusion setting) it uses the ratio at training step 0.
    ///   - spacing: How the schedule walks the training range. `trailing` ignores `stepsOffset`, as
    ///     the reference does.
    public init(trainingSteps: Int = 1000,
                betaStart: Float = 0.00085,
                betaEnd: Float = 0.012,
                predictionType: NFKDiffusionPredictionType = .epsilon,
                stepsOffset: Int = 1,
                setsAlphaToOne: Bool = false,
                spacing: NFKDiffusionTimestepSpacing = .leading) {
        self.trainingSteps = max(trainingSteps, 1)
        self.predictionType = predictionType
        self.stepsOffset = stepsOffset
        self.spacing = spacing

        var bars = [Float]()
        bars.reserveCapacity(self.trainingSteps)
        var cumulative: Float = 1
        let low = sqrtf(betaStart)
        let high = sqrtf(betaEnd)
        for t in 0 ..< self.trainingSteps {
            let fraction = self.trainingSteps > 1 ? Float(t) / Float(self.trainingSteps - 1) : 0
            let beta = powf(low + (high - low) * fraction, 2)
            cumulative *= (1 - beta)
            bars.append(cumulative)
        }
        self.alphaBars = bars
        self.finalAlphaBar = setsAlphaToOne ? 1 : bars[0]
    }

    // A guard against dividing by zero on a pathological schedule. The released Stable Diffusion
    // schedules never come near it, so it does not move a real run — the previous 1e-4 floor did.
    private func bar(_ train: Int) -> Float {
        max(alphaBars[min(max(train, 0), trainingSteps - 1)], 1e-12)
    }

    /// The cumulative signal ratio at a training step, for a sampler that shares this beta schedule.
    func alphaBar(at train: Int) -> Float { bar(train) }

    /// The training step whose signal ratio is `value`, as ``alphaBar(at:)`` returned it; the ratios fall
    /// strictly with the step, so the match is unique. A value from elsewhere maps to the nearest step.
    func train(withAlphaBar value: Float) -> Int {
        if let exact = alphaBars.firstIndex(of: value) { return exact }
        var nearest = 0
        for (train, bar) in alphaBars.enumerated() where abs(bar - value) < abs(alphaBars[nearest] - value) {
            nearest = train
        }
        return nearest
    }

    public func steps(_ count: Int) -> [NFKDiffusionTimestep] {
        let n = max(count, 1)
        // The reference walks a fixed stride through the training schedule rather than dividing the
        // range evenly: at 20 leading steps of 1000 that is 951…1, not 999…49.
        let stride = max(trainingSteps / n, 1)
        let trains: [Int]
        switch spacing {
        case .leading:
            trains = (0 ..< n).map { i in max(0, (n - 1 - i) * stride + stepsOffset) }
        case .trailing:
            // Counted down from the end of the training range, each step less one, and no offset.
            let exact = Float(trainingSteps) / Float(n)
            trains = (0 ..< n).map { i in max(0, Int((Float(trainingSteps) - Float(i) * exact).rounded()) - 1) }
        }
        return (0 ..< n).map { i in
            // The previous step is one stride down the TRAINING schedule; below zero the reference
            // substitutes its final signal ratio.
            let previous = trains[i] - stride
            return NFKDiffusionTimestep(index: i,
                                        train: trains[i],
                                        alphaBar: bar(trains[i]),
                                        alphaBarPrev: previous >= 0 ? bar(previous) : finalAlphaBar)
        }
    }

    public func initialLatent(noise: MLXArray, first: NFKDiffusionTimestep) -> MLXArray {
        // At the highest-noise step alphaBar ≈ 0, so x_T is the unit-variance noise itself.
        noise
    }

    public func addNoise(clean: MLXArray, noise: MLXArray, timestep: NFKDiffusionTimestep) -> MLXArray {
        clean * sqrtf(timestep.alphaBar) + noise * sqrtf(1 - timestep.alphaBar)
    }

    public func step(prediction: MLXArray, timestep: NFKDiffusionTimestep, latent: MLXArray) -> MLXArray {
        let sqrtBar = sqrtf(timestep.alphaBar)
        let sqrtComplement = sqrtf(1 - timestep.alphaBar)
        let cleanLatent: MLXArray
        let noise: MLXArray
        switch predictionType {
        case .epsilon:
            noise = prediction
            cleanLatent = (latent - noise * sqrtComplement) / sqrtBar
        case .sample:
            cleanLatent = prediction
            noise = (latent - cleanLatent * sqrtBar) / sqrtComplement
        case .vPrediction:
            cleanLatent = latent * sqrtBar - prediction * sqrtComplement
            noise = prediction * sqrtBar + latent * sqrtComplement
        }
        let sqrtBarPrev = sqrtf(timestep.alphaBarPrev)
        let sqrtComplementPrev = sqrtf(1 - timestep.alphaBarPrev)
        return cleanLatent * sqrtBarPrev + noise * sqrtComplementPrev
    }
}

/// A Latent Consistency Model sampler over the same scaled-linear beta schedule, for few-step
/// generation (typically 4–8 steps). Each step recovers the clean latent, applies the LCM consistency
/// boundary condition — `denoised = c_out·x₀ + c_skip·latent`, with `c_skip` / `c_out` a function of
/// the training step — then, unless it is the final step, re-noises to the next level with **fresh**
/// noise (the multistep-LCM behavior). The last step (`alphaBarPrev == 1`) returns the clean estimate.
///
/// The fresh noise comes from the same deterministic SplitMix64 + Box–Muller stream the diffusion
/// backend uses, keyed by the run's seed and the step, so a run is repeatable without the MLX random
/// state and a request's `NFKParameterSeed` changes every draw; ``noise`` substitutes a caller's own,
/// which is how a run reproduces another implementation's draw. An LCM-distilled UNet or an LCM-LoRA
/// merged into a base UNet runs with this scheduler.
///
/// The schedule is the reference's: the training range is divided into `originalInferenceSteps`
/// evenly spaced steps (`19, 39, …, 999` at the default 50), and an inference schedule of `n` steps
/// takes every `50 / n`-th of them from the top, so 4 steps visit `999, 759, 499, 259`. The signal
/// ratio after each step is the next step's. The final step returns the clean estimate without
/// re-noising, as the reference does whatever its `set_alpha_to_one` says, so the schedule marks it
/// with a signal ratio of 1.
public struct NFKLCMScheduler: NFKDiffusionScheduler {

    public let trainingSteps: Int
    public let predictionType: NFKDiffusionPredictionType
    /// How many evenly spaced training steps the model was distilled over (the reference's
    /// `original_inference_steps`, 50 for every released LCM). Introduced in InferKit 0.4.0.
    public let originalInferenceSteps: Int
    private let sigmaData: Float
    private let timestepScaling: Float
    private var seed: UInt64
    private let ddim: NFKDDIMScheduler

    /// The noise a non-final step re-noises with, in place of the seeded stream: given the step, it
    /// returns `[height, width, channels]` or `[1, height, width, channels]` unit-variance noise, or nil
    /// to use the stream. A parity test hands over the reference's own draws through it.
    /// Introduced in InferKit 0.4.0.
    public var noise: (@Sendable (NFKDiffusionTimestep) -> MLXArray?)?

    /// - Parameters:
    ///   - trainingSteps: The training schedule length (SD uses 1000).
    ///   - betaStart: The scaled-linear beta schedule's start endpoint (SD default).
    ///   - betaEnd: The scaled-linear beta schedule's end endpoint (SD default).
    ///   - predictionType: What the distilled model predicts each step.
    ///   - originalInferenceSteps: The distillation's step count, which spaces the schedule.
    ///   - sigmaData: The LCM boundary-condition data scale (diffusers default).
    ///   - timestepScaling: The LCM boundary-condition timestep scale (diffusers default).
    ///   - seed: Seeds the per-step noise stream, so the sampler is repeatable. The backend replaces it
    ///     with the request's seed through ``seeded(_:)``.
    public init(trainingSteps: Int = 1000,
                betaStart: Float = 0.00085,
                betaEnd: Float = 0.012,
                predictionType: NFKDiffusionPredictionType = .epsilon,
                originalInferenceSteps: Int = 50,
                sigmaData: Float = 0.5,
                timestepScaling: Float = 10,
                seed: UInt64 = 0) {
        self.trainingSteps = max(trainingSteps, 1)
        self.predictionType = predictionType
        self.originalInferenceSteps = min(max(originalInferenceSteps, 1), self.trainingSteps)
        self.sigmaData = sigmaData
        self.timestepScaling = timestepScaling
        self.seed = seed
        // DDIM carries the beta schedule; the step spacing is this sampler's own.
        ddim = NFKDDIMScheduler(trainingSteps: trainingSteps, betaStart: betaStart, betaEnd: betaEnd,
                                predictionType: predictionType)
    }

    public func seeded(_ seed: UInt64) -> any NFKDiffusionScheduler {
        var copy = self
        copy.seed = seed
        return copy
    }

    public func steps(_ count: Int) -> [NFKDiffusionTimestep] {
        // The reference takes `count` of its `originalInferenceSteps` distillation steps, highest noise
        // first, at indices floor(i · original / count).
        let n = min(max(count, 1), originalInferenceSteps)
        let stride = trainingSteps / originalInferenceSteps
        let origin = (1 ... originalInferenceSteps).map { $0 * stride - 1 }.reversed().map { $0 }
        let trains = (0 ..< n).map { i in
            origin[Int((Float(i) * Float(originalInferenceSteps) / Float(n)).rounded(.down))]
        }
        return (0 ..< n).map { i in
            NFKDiffusionTimestep(index: i, train: trains[i], alphaBar: ddim.alphaBar(at: trains[i]),
                                 alphaBarPrev: i + 1 < n ? ddim.alphaBar(at: trains[i + 1]) : 1)
        }
    }

    public func initialLatent(noise: MLXArray, first: NFKDiffusionTimestep) -> MLXArray { noise }

    public func addNoise(clean: MLXArray, noise: MLXArray, timestep: NFKDiffusionTimestep) -> MLXArray {
        ddim.addNoise(clean: clean, noise: noise, timestep: timestep)
    }

    /// The LCM boundary scalings at a training step: `c_skip` weights the noisy latent, `c_out` the
    /// predicted clean latent.
    private func boundary(_ train: Int) -> (skip: Float, out: Float) {
        let scaled = Float(train) * timestepScaling
        let denominator = scaled * scaled + sigmaData * sigmaData
        return (sigmaData * sigmaData / denominator, scaled / sqrtf(denominator))
    }

    public func step(prediction: MLXArray, timestep: NFKDiffusionTimestep, latent: MLXArray) -> MLXArray {
        let sqrtBar = sqrtf(timestep.alphaBar)
        let sqrtComplement = sqrtf(1 - timestep.alphaBar)
        let cleanLatent: MLXArray
        switch predictionType {
        case .epsilon:    cleanLatent = (latent - prediction * sqrtComplement) / sqrtBar
        case .sample:     cleanLatent = prediction
        case .vPrediction: cleanLatent = latent * sqrtBar - prediction * sqrtComplement
        }

        let (skip, out) = boundary(timestep.train)
        let denoised = cleanLatent * out + latent * skip        // the consistency function's clean estimate
        if timestep.alphaBarPrev >= 1 {
            return denoised                                     // final step: return the clean latent
        }
        // Re-noise to the next level with the caller's draw, or with fresh step-keyed deterministic noise.
        // The backend hands the sampler an unbatched `[height, width, channels]` latent; a caller may
        // batch one, so the draw takes whatever shape arrived.
        let spatial = Array(latent.shape.suffix(3))
        let freshNoise = (noise?(timestep)
            ?? NFKMLXDiffusionBackend.gaussianNoise(height: spatial[0], width: spatial[1], channels: spatial[2],
                                                    seed: seed &+ UInt64(timestep.train) &+ 1))
            .reshaped(latent.shape).asType(denoised.dtype)
        return denoised * sqrtf(timestep.alphaBarPrev) + freshNoise * sqrtf(1 - timestep.alphaBarPrev)
    }
}

/// The trajectory consistency sampler (diffusers' `TCDScheduler`), for a UNet distilled by
/// trajectory consistency distillation: the Hyper-SD unified adapters, which run at one to eight
/// steps from one file.
///
/// The schedule is the latent consistency sampler's: `count` of the `originalInferenceSteps`
/// distillation steps, highest noise first. Each step forms the clean estimate from the prediction,
/// then re-noises it to the training step `floor((1 − eta) · next)`, where `next` is the following
/// step's training step and 0 after the last; at `eta` 1 that is training step 0, whose signal ratio
/// is just under 1. A non-final step then moves from that level to the next step's with fresh
/// noise, `sqrt(alphaBarNext / alphaBarS) · estimate + sqrt(1 − alphaBarNext / alphaBarS) · noise`;
/// the final step returns the estimate. At `eta` 0 the sampler is deterministic.
///
/// The fresh noise comes from the same seeded stream the latent consistency sampler draws from, keyed
/// by the run's seed and the step; ``noise`` substitutes a caller's own draws, which is how a parity
/// run reproduces the reference's.
///
/// Introduced in InferKit 0.4.0.
public struct NFKTCDScheduler: NFKDiffusionScheduler {

    public let trainingSteps: Int
    public let predictionType: NFKDiffusionPredictionType
    /// How many evenly spaced training steps the model was distilled over (the reference's
    /// `original_inference_steps`, 50 for the released adapters).
    public let originalInferenceSteps: Int
    /// The stochastic parameter (the reference's `eta`, the paper's gamma), `0...1`. 1 re-noises every
    /// non-final step from the clean estimate, the setting the Hyper-SD unified adapters publish; lower
    /// values keep more of each step's own noise.
    public let eta: Float
    private var seed: UInt64
    private let ddim: NFKDDIMScheduler

    /// The noise a non-final step re-noises with, in place of the seeded stream. See
    /// ``NFKLCMScheduler/noise``.
    public var noise: (@Sendable (NFKDiffusionTimestep) -> MLXArray?)?

    /// - Parameters:
    ///   - trainingSteps: The training schedule length (SD uses 1000).
    ///   - betaStart: The scaled-linear beta schedule's start endpoint (SD default).
    ///   - betaEnd: The scaled-linear beta schedule's end endpoint (SD default).
    ///   - predictionType: What the distilled model predicts each step.
    ///   - originalInferenceSteps: The distillation's step count, which spaces the schedule.
    ///   - eta: The stochastic parameter, `0...1`.
    ///   - seed: Seeds the per-step noise stream. The backend replaces it with the request's seed
    ///     through ``seeded(_:)``.
    public init(trainingSteps: Int = 1000,
                betaStart: Float = 0.00085,
                betaEnd: Float = 0.012,
                predictionType: NFKDiffusionPredictionType = .epsilon,
                originalInferenceSteps: Int = 50,
                eta: Float = 1,
                seed: UInt64 = 0) {
        self.trainingSteps = max(trainingSteps, 1)
        self.predictionType = predictionType
        self.originalInferenceSteps = min(max(originalInferenceSteps, 1), self.trainingSteps)
        self.eta = min(max(eta, 0), 1)
        self.seed = seed
        ddim = NFKDDIMScheduler(trainingSteps: trainingSteps, betaStart: betaStart, betaEnd: betaEnd,
                                predictionType: predictionType)
    }

    public func seeded(_ seed: UInt64) -> any NFKDiffusionScheduler {
        var copy = self
        copy.seed = seed
        return copy
    }

    /// The training steps an inference schedule of `count` steps visits, highest noise first.
    func trainingSchedule(_ count: Int) -> [Int] {
        let n = min(max(count, 1), originalInferenceSteps)
        let stride = trainingSteps / originalInferenceSteps
        let origin = (1 ... originalInferenceSteps).map { $0 * stride - 1 }.reversed().map { $0 }
        return (0 ..< n).map { i in
            origin[Int((Float(i) * Float(originalInferenceSteps) / Float(n)).rounded(.down))]
        }
    }

    /// The reference's `prev_timestep`: the next step's training step, and 0 after the last, whose
    /// signal ratio is the schedule's first (not 1).
    public func steps(_ count: Int) -> [NFKDiffusionTimestep] {
        let trains = trainingSchedule(count)
        return trains.indices.map { i in
            let next = i + 1 < trains.count ? trains[i + 1] : 0
            return NFKDiffusionTimestep(index: i, train: trains[i], alphaBar: ddim.alphaBar(at: trains[i]),
                                        alphaBarPrev: ddim.alphaBar(at: next))
        }
    }

    public func initialLatent(noise: MLXArray, first: NFKDiffusionTimestep) -> MLXArray { noise }

    public func addNoise(clean: MLXArray, noise: MLXArray, timestep: NFKDiffusionTimestep) -> MLXArray {
        ddim.addNoise(clean: clean, noise: noise, timestep: timestep)
    }

    /// The training step a step re-noises its clean estimate to, `floor((1 − eta) · next)`. The next
    /// step's training step is recovered from the signal ratio the schedule recorded for it.
    func transitionTrain(for timestep: NFKDiffusionTimestep) -> Int {
        let next = ddim.train(withAlphaBar: timestep.alphaBarPrev)
        return Int((Double(1 - eta) * Double(next)).rounded(.down))
    }

    public func step(prediction: MLXArray, timestep: NFKDiffusionTimestep, latent: MLXArray) -> MLXArray {
        let sqrtBar = sqrtf(timestep.alphaBar)
        let sqrtComplement = sqrtf(1 - timestep.alphaBar)
        let cleanLatent: MLXArray
        let epsilon: MLXArray
        switch predictionType {
        case .epsilon:
            cleanLatent = (latent - prediction * sqrtComplement) / sqrtBar
            epsilon = prediction
        case .sample:
            cleanLatent = prediction
            epsilon = (latent - prediction * sqrtBar) / sqrtComplement
        case .vPrediction:
            cleanLatent = latent * sqrtBar - prediction * sqrtComplement
            epsilon = prediction * sqrtBar + latent * sqrtComplement
        }
        let alphaBarS = ddim.alphaBar(at: transitionTrain(for: timestep))
        let estimate = cleanLatent * sqrtf(alphaBarS) + epsilon * sqrtf(1 - alphaBarS)
        // No distillation step sits at training step 0, so a next step of 0 marks the final step.
        if eta == 0 || ddim.train(withAlphaBar: timestep.alphaBarPrev) == 0 {
            return estimate
        }
        let ratio = timestep.alphaBarPrev / alphaBarS
        let spatial = Array(latent.shape.suffix(3))
        let freshNoise = (noise?(timestep)
            ?? NFKMLXDiffusionBackend.gaussianNoise(height: spatial[0], width: spatial[1], channels: spatial[2],
                                                    seed: seed &+ UInt64(timestep.train) &+ 1))
            .reshaped(latent.shape).asType(estimate.dtype)
        return estimate * sqrtf(ratio) + freshNoise * sqrtf(1 - ratio)
    }
}
