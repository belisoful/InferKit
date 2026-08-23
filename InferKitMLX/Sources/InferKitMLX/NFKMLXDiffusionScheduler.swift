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
/// backend uses (keyed by the step), so a run is repeatable without the MLX random state. A real
/// integration loads an LCM-distilled UNet and keeps this scheduler.
public struct NFKLCMScheduler: NFKDiffusionScheduler {

    public let trainingSteps: Int
    public let predictionType: NFKDiffusionPredictionType
    private let sigmaData: Float
    private let timestepScaling: Float
    private let seed: UInt64
    private let ddim: NFKDDIMScheduler

    /// - Parameters:
    ///   - trainingSteps: The training schedule length (SD uses 1000).
    ///   - betaStart: The scaled-linear beta schedule's start endpoint (SD default).
    ///   - betaEnd: The scaled-linear beta schedule's end endpoint (SD default).
    ///   - predictionType: What the distilled model predicts each step.
    ///   - sigmaData: The LCM boundary-condition data scale (diffusers default).
    ///   - timestepScaling: The LCM boundary-condition timestep scale (diffusers default).
    ///   - seed: Seeds the per-step noise stream, so the sampler is repeatable.
    public init(trainingSteps: Int = 1000,
                betaStart: Float = 0.00085,
                betaEnd: Float = 0.012,
                predictionType: NFKDiffusionPredictionType = .epsilon,
                sigmaData: Float = 0.5,
                timestepScaling: Float = 10,
                seed: UInt64 = 0) {
        self.trainingSteps = max(trainingSteps, 1)
        self.predictionType = predictionType
        self.sigmaData = sigmaData
        self.timestepScaling = timestepScaling
        self.seed = seed
        // Reuse DDIM for the noise schedule and the step spacing.
        ddim = NFKDDIMScheduler(trainingSteps: trainingSteps, betaStart: betaStart, betaEnd: betaEnd, predictionType: predictionType)
    }

    public func steps(_ count: Int) -> [NFKDiffusionTimestep] { ddim.steps(count) }

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
        // Re-noise to the next level with fresh, step-keyed deterministic noise.
        let (n, h, w, c) = (latent.shape[0], latent.shape[1], latent.shape[2], latent.shape[3])
        let freshNoise = NFKMLXDiffusionBackend.gaussianNoise(height: h, width: w, channels: c,
                                                              seed: seed &+ UInt64(timestep.train) &+ 1)
            .reshaped([n, h, w, c])
        return denoised * sqrtf(timestep.alphaBarPrev) + freshNoise * sqrtf(1 - timestep.alphaBarPrev)
    }
}
