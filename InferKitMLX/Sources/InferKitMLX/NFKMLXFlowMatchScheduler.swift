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

    public init(trainTimesteps: Int = 1000, baseShift: Float = 0.95, maxShift: Float = 2.05,
                baseSequenceLength: Int = 1024, maxSequenceLength: Int = 4096, shiftTerminal: Float? = 0.1,
                useDynamicShifting: Bool = true) {
        self.trainTimesteps = trainTimesteps
        self.baseShift = baseShift
        self.maxShift = maxShift
        self.baseSequenceLength = baseSequenceLength
        self.maxSequenceLength = maxSequenceLength
        self.shiftTerminal = shiftTerminal
        self.useDynamicShifting = useDynamicShifting
    }

    public static let ltxVideo = NFKMLXFlowMatchConfiguration()

    /// Z-Image's schedule: a smaller resolution shift, no terminal stretch (its `sigma_min` is 0).
    public static let zImage = NFKMLXFlowMatchConfiguration(
        baseShift: 0.5, maxShift: 1.15, baseSequenceLength: 256, maxSequenceLength: 4096,
        shiftTerminal: nil, useDynamicShifting: true)

    /// A static-shift flow schedule at SANA's `flow_shift` (3.0). SANA's released sampler is a
    /// `DPMSolverMultistepScheduler` (flow prediction); this is the rectified-flow stand-in the pipeline
    /// glue runs, the way the LTX pipeline substitutes DDIM for SDXL-Turbo's named sampler.
    public static let sana = NFKMLXFlowMatchConfiguration(
        baseShift: 3.0, shiftTerminal: nil, useDynamicShifting: false)

    /// A static-shift flow schedule at Wan's `flow_shift` (5.0). Wan's released sampler is a
    /// `UniPCMultistepScheduler`; this is the rectified-flow stand-in the pipeline glue runs.
    public static let wan = NFKMLXFlowMatchConfiguration(
        baseShift: 5.0, shiftTerminal: nil, useDynamicShifting: false)
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

    /// Builds the schedule for `steps` inference steps at a latent sequence length (which sets the shift).
    public mutating func setTimesteps(_ steps: Int, sequenceLength: Int) {
        let train = Float(configuration.trainTimesteps)
        // A sigma ramp from 1 to 1/train.
        var s = (0 ..< steps).map { index -> Float in
            1 - (1 - 1 / train) * Float(index) / Float(max(steps - 1, 1))
        }
        if configuration.useDynamicShifting {
            let expMu = exp(shift(forSequenceLength: sequenceLength))
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
        sample + (sigmas[index + 1] - sigmas[index]) * velocity
    }

    /// Adds noise to a clean latent at a given sigma, the flow interpolation `(1 − σ)·x + σ·noise`.
    public func addNoise(_ sample: MLXArray, noise: MLXArray, sigma: Float) -> MLXArray {
        (1 - sigma) * sample + sigma * noise
    }
}
