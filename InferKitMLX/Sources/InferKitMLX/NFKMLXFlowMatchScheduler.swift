//
//  NFKMLXFlowMatchScheduler.swift
//  InferKitMLX
//

import Foundation
import MLX

// The rectified-flow sampler diffusers ships as `FlowMatchEulerDiscreteScheduler`, the sampler LTX-Video,
// Flux, SD3, and Wan use. The schedule is a sigma ramp from 1 to 0 with resolution-dependent DYNAMIC
// SHIFTING (a per-sequence-length `mu` warps the ramp) and a terminal stretch so the last non-zero sigma
// lands on `shiftTerminal`. A step is one Euler update `x + (σ_next − σ)·v`, the model predicting the
// velocity. This is a value type with no parameters.

/// Rectified-flow schedule configuration. Defaults are the released LTX-Video scheduler.
public struct NFKMLXFlowMatchConfiguration: Sendable {
    public var trainTimesteps: Int
    public var baseShift: Float
    public var maxShift: Float
    public var baseSequenceLength: Int
    public var maxSequenceLength: Int
    public var shiftTerminal: Float?
    public var useDynamicShifting: Bool
    /// Where the linear sigma ramp ends before the shift is applied.
    ///
    /// @discussion The releases disagree. The scheduler's own default ends the ramp at its
    /// `sigma_min` (`1 / trainTimesteps`, shifted once more under a static shift), which is what the
    /// LTX-Video and Stable Diffusion 3 pipelines use; the FLUX
    /// and Qwen-Image pipelines pass an explicit ramp ending at `1 / steps` instead
    /// (`np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)`), so every sigma after the
    /// first differs from the default schedule at the same step count, the last one most.
    public var rampEndsAtStepFraction: Bool
    /// Whether the linear sigma ramp ends at 0.
    ///
    /// @discussion The Z-Image pipeline sets the scheduler's `sigma_min` to 0 before building the
    /// schedule, so the last of its steps sits at sigma 0 and moves nothing: a release sampled at nine
    /// steps evaluates its transformer eight times. Takes precedence over ``rampEndsAtStepFraction``.
    public var rampEndsAtZero: Bool = false
    /// Whether `mu` comes from FLUX.2's empirical fit rather than the linear interpolation in
    /// sequence length every earlier release uses.
    ///
    /// @discussion FLUX.2 replaces `calculate_shift` with `compute_empirical_mu`, which depends on the
    /// STEP COUNT as well as the sequence length: two lines in sequence length are fitted at 10 and
    /// 200 steps, and the shift interpolates linearly between them in the number of steps. Above a
    /// sequence length of 4300 the 200-step line is used alone. `baseShift` and the sequence-length
    /// bounds are unused when this is set.
    public var usesEmpiricalShift: Bool

    public init(trainTimesteps: Int = 1000, baseShift: Float = 0.95, maxShift: Float = 2.05,
                baseSequenceLength: Int = 1024, maxSequenceLength: Int = 4096, shiftTerminal: Float? = 0.1,
                useDynamicShifting: Bool = true, rampEndsAtStepFraction: Bool = false,
                usesEmpiricalShift: Bool = false) {
        self.rampEndsAtStepFraction = rampEndsAtStepFraction
        self.usesEmpiricalShift = usesEmpiricalShift
        self.trainTimesteps = trainTimesteps
        self.baseShift = baseShift
        self.maxShift = maxShift
        self.baseSequenceLength = baseSequenceLength
        self.maxSequenceLength = maxSequenceLength
        self.shiftTerminal = shiftTerminal
        self.useDynamicShifting = useDynamicShifting
    }

    public static let ltxVideo = NFKMLXFlowMatchConfiguration()

    /// LTX-Video's schedule as its pipeline runs it: the scheduler configuration of ``ltxVideo`` fed the
    /// pipeline's own `linspace(1, 1 / steps, steps)` ramp, which diffusers' `LTXPipeline` passes.
    public static let ltxVideoPipeline = NFKMLXFlowMatchConfiguration(rampEndsAtStepFraction: true)

    /// FLUX.2's schedule: the empirical shift, the pipeline's own `1 / steps` ramp, no terminal
    /// stretch. The released `scheduler_config.json` still carries `base_shift` 0.5 and `max_shift`
    /// 1.15, which the empirical fit replaces rather than reads.
    public static let flux2 = NFKMLXFlowMatchConfiguration(
        baseShift: 0.5, maxShift: 1.15, baseSequenceLength: 256, maxSequenceLength: 4096,
        shiftTerminal: nil, useDynamicShifting: true, rampEndsAtStepFraction: true,
        usesEmpiricalShift: true)

    /// Z-Image's schedule, the released `Tongyi-MAI/Z-Image` scheduler config: a static shift of 6.0
    /// over a ramp ending at sigma 0.
    public static let zImage = NFKMLXFlowMatchConfiguration.staticShiftToZero(6.0)

    /// Z-Image-Turbo's schedule, its released scheduler config: a static shift of 3.0 over a ramp
    /// ending at sigma 0.
    public static let zImageTurbo = NFKMLXFlowMatchConfiguration.staticShiftToZero(3.0)

    static func staticShiftToZero(_ shift: Float) -> NFKMLXFlowMatchConfiguration {
        var configuration = NFKMLXFlowMatchConfiguration(baseShift: shift, shiftTerminal: nil,
                                                         useDynamicShifting: false)
        configuration.rampEndsAtZero = true
        return configuration
    }

    /// A static-shift flow schedule at SANA's `flow_shift` (3.0). SANA's released sampler is a
    /// `DPMSolverMultistepScheduler` (flow prediction); this is the rectified-flow stand-in the pipeline
    /// glue runs, the way the LTX pipeline substitutes DDIM for SDXL-Turbo's named sampler.
    public static let sana = NFKMLXFlowMatchConfiguration(
        baseShift: 3.0, shiftTerminal: nil, useDynamicShifting: false)

    /// A static-shift flow schedule at Wan's `flow_shift` (5.0). Wan's released sampler is a
    /// `UniPCMultistepScheduler`; this is the rectified-flow stand-in the pipeline glue runs.
    public static let wan = NFKMLXFlowMatchConfiguration(
        baseShift: 5.0, shiftTerminal: nil, useDynamicShifting: false)

    /// Stable Diffusion 3 / 3.5's schedule: a static shift of 3.0 (the released
    /// `FlowMatchEulerDiscreteScheduler` config, `use_dynamic_shifting=False`) over the scheduler's
    /// own default ramp; the SD3 pipelines pass no ramp of their own.
    public static let sd3 = NFKMLXFlowMatchConfiguration(
        baseShift: 3.0, shiftTerminal: nil, useDynamicShifting: false)

    /// FLUX.1 [dev]'s schedule: resolution-dependent dynamic shifting (base 0.5, max 1.15 over the
    /// 256…4096 sequence range), no terminal stretch, and the `1 / steps` ramp its pipeline passes.
    public static let flux = NFKMLXFlowMatchConfiguration(
        baseShift: 0.5, maxShift: 1.15, baseSequenceLength: 256, maxSequenceLength: 4096,
        shiftTerminal: nil, useDynamicShifting: true, rampEndsAtStepFraction: true)

    /// FLUX.1 [schnell]'s schedule: a static shift of 1.0 (the four-step distillation's own config)
    /// over the `1 / steps` ramp its pipeline passes, so four steps run at sigmas 1, 0.75, 0.5, 0.25.
    public static let fluxSchnell = NFKMLXFlowMatchConfiguration(
        baseShift: 1.0, shiftTerminal: nil, useDynamicShifting: false, rampEndsAtStepFraction: true)

    /// Qwen-Image 2.1's schedule: dynamic shifting over the 256…8192 sequence range (base 0.5, max
    /// 0.9), a terminal stretch onto 0.02, and the ramp its pipeline passes rather than the
    /// scheduler's own default.
    public static let qwenImage21 = NFKMLXFlowMatchConfiguration(
        baseShift: 0.5, maxShift: 0.9, baseSequenceLength: 256, maxSequenceLength: 8192,
        shiftTerminal: 0.02, useDynamicShifting: true, rampEndsAtStepFraction: true)
}

/// A rectified-flow sampler.
public struct NFKMLXFlowMatchScheduler {
    public let configuration: NFKMLXFlowMatchConfiguration
    /// The sigma schedule, `steps + 1` values ending in 0.
    public private(set) var sigmas: [Float] = []
    /// The timestep the model is conditioned on at each step (`sigma · trainTimesteps`).
    public private(set) var timesteps: [Float] = []

    public init(_ configuration: NFKMLXFlowMatchConfiguration = .ltxVideo) {
        self.configuration = configuration
    }

    /// The resolution-dependent shift `mu` for a latent sequence length.
    public func shift(forSequenceLength length: Int) -> Float {
        let c = configuration
        let m = (c.maxShift - c.baseShift) / Float(c.maxSequenceLength - c.baseSequenceLength)
        return Float(length) * m + (c.baseShift - m * Float(c.baseSequenceLength))
    }

    /// The shift `mu`, taking the step count for the releases whose fit depends on it.
    ///
    /// @discussion The arithmetic runs in double precision because the reference's is a `float`
    /// division of fitted constants; at single precision the interpolation's intercept loses digits
    /// the sigmas then carry.
    public func shift(forSequenceLength length: Int, steps: Int) -> Float {
        guard configuration.usesEmpiricalShift else { return shift(forSequenceLength: length) }
        let (a1, b1) = (8.73809524e-05, 1.89833333)
        let (a2, b2) = (0.00016927, 0.45666666)
        let sequence = Double(length)
        let atTwoHundred = a2 * sequence + b2
        if sequence > 4300 { return Float(atTwoHundred) }
        let atTen = a1 * sequence + b1
        let slope = (atTwoHundred - atTen) / 190
        return Float(slope * Double(steps) + (atTwoHundred - 200 * slope))
    }

    /// Builds the schedule for `steps` inference steps at a latent sequence length (which sets the shift).
    public mutating func setTimesteps(_ steps: Int, sequenceLength: Int) {
        let train = Float(configuration.trainTimesteps)
        // A sigma ramp from 1 down to its end: the step fraction for the pipelines that pass their
        // own ramp, and the scheduler's `sigma_min` otherwise. Diffusers builds `sigma_min` from
        // `1/train` at construction and, under a static shift, shifts it there as well, so the ramp's
        // end is the shifted value and the shift below is applied to it a second time.
        let end: Float
        if configuration.rampEndsAtZero {
            end = 0
        } else if configuration.rampEndsAtStepFraction {
            end = 1 / Float(steps)
        } else if configuration.useDynamicShifting {
            end = 1 / train
        } else {
            end = configuration.baseShift / train / (1 + (configuration.baseShift - 1) / train)
        }
        var s = (0 ..< steps).map { index -> Float in
            1 - (1 - end) * Float(index) / Float(max(steps - 1, 1))
        }
        if configuration.useDynamicShifting {
            let expMu = exp(shift(forSequenceLength: sequenceLength, steps: steps))
            s = s.map { expMu / (expMu + (1 / $0 - 1)) }                 // exponential time shift
        } else {
            s = s.map { configuration.baseShift * $0 / (1 + (configuration.baseShift - 1) * $0) }
        }
        if let terminal = configuration.shiftTerminal {
            let scale = (1 - s[s.count - 1]) / (1 - terminal)            // stretch so the last sigma is `terminal`
            s = s.map { 1 - (1 - $0) / scale }
        }
        timesteps = s.map { $0 * train }
        sigmas = s + [0]
    }

    /// One Euler step: `sample + (σ_next − σ)·velocity`.
    public func step(velocity: MLXArray, sample: MLXArray, index: Int) -> MLXArray {
        NFKReferenceRounding.eulerStep(sample, velocity: velocity, dt: sigmas[index + 1] - sigmas[index])
    }

    /// Adds noise to a clean latent at a given sigma, the flow interpolation `(1 − σ)·x + σ·noise`.
    public func addNoise(_ sample: MLXArray, noise: MLXArray, sigma: Float) -> MLXArray {
        (1 - sigma) * sample + sigma * noise
    }
}
