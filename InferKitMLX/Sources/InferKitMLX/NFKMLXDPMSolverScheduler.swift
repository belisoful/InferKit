//
//  NFKMLXDPMSolverScheduler.swift
//  InferKitMLX
//

import Foundation
import MLX

// The DPM-Solver++ multistep sampler (`DPMSolverMultistepScheduler`), SANA's released sampler in its
// flow-prediction configuration (`algorithm_type: dpmsolver++`, `solver_order: 2`, `use_flow_sigmas`,
// `solver_type: midpoint`, `final_sigmas_type: zero`). A value type with no weights: the schedule is a
// static-shift flow sigma ramp, and each step converts the flow velocity to a data prediction `x0` and
// takes a first- or second-order multistep DPM-Solver++ update. The coefficients are `Float` scalars
// (so the `log 0` / `exp(-inf)` at the terminal zero sigma resolve to a clean `x0`), applied to the
// `MLXArray` sample.

/// DPM-Solver++ geometry.
public struct NFKMLXDPMSolverConfiguration: Sendable {
    public var flowShift: Float
    public var solverOrder: Int
    public var trainTimesteps: Int

    public init(flowShift: Float = 3.0, solverOrder: Int = 2, trainTimesteps: Int = 1000) {
        self.flowShift = flowShift
        self.solverOrder = solverOrder
        self.trainTimesteps = trainTimesteps
    }

    /// SANA's released configuration (flow_shift 3.0, second-order).
    public static let sana = NFKMLXDPMSolverConfiguration(flowShift: 3.0)
}

public struct NFKMLXDPMSolverScheduler {
    public let configuration: NFKMLXDPMSolverConfiguration
    /// The sigma schedule, `steps + 1` values ending in 0.
    public private(set) var sigmas: [Float] = []
    /// The timestep the model is conditioned on at each step (`sigma · trainTimesteps`).
    public private(set) var timesteps: [Float] = []

    private var stepIndex = 0
    private var modelOutputs: [MLXArray] = []                              // the data-prediction (x0) history
    private var lowerOrderNums = 0

    public init(_ configuration: NFKMLXDPMSolverConfiguration = .sana) {
        self.configuration = configuration
    }

    /// Builds the flow sigma schedule for `steps` inference steps.
    public mutating func setTimesteps(_ steps: Int, sequenceLength: Int = 0) {
        let train = Float(configuration.trainTimesteps)
        let shift = configuration.flowShift
        // alphas = linspace(1, 1/train, steps + 1); sigmas = shift·(1−α)/(1 + (shift−1)(1−α)).
        var s = (0 ... steps).map { index -> Float in
            let alpha = 1 - (1 - 1 / train) * Float(index) / Float(steps)
            let raw = 1 - alpha
            return shift * raw / (1 + (shift - 1) * raw)
        }
        s.reverse()
        s.removeLast()                                                     // drop the terminal (flipped) zero
        // diffusers stores the flow timesteps as an integer truncation of `sigma · trainTimesteps`.
        timesteps = s.map { ($0 * train).rounded(.towardZero) }
        sigmas = s + [0]
        stepIndex = 0
        modelOutputs = []
        lowerOrderNums = 0
    }

    /// One sampler step. `velocity` is the flow prediction; `index` is accepted for signature parity but
    /// the scheduler tracks its own step counter (it carries multistep history).
    public mutating func step(velocity: MLXArray, sample: MLXArray, index: Int) -> MLXArray {
        let i = stepIndex
        let lowerOrderFinal = i == timesteps.count - 1                     // final_sigmas_type == zero
        let x0 = sample - sigmas[i] * velocity                            // flow → data prediction
        modelOutputs.append(x0)
        if modelOutputs.count > configuration.solverOrder { modelOutputs.removeFirst() }

        let previous: MLXArray
        if configuration.solverOrder == 1 || lowerOrderNums < 1 || lowerOrderFinal {
            previous = firstOrder(x0, sample: sample)
        } else {
            previous = secondOrder(sample: sample)
        }
        if lowerOrderNums < configuration.solverOrder { lowerOrderNums += 1 }
        stepIndex += 1
        return previous
    }

    private func lambda(_ sigma: Float) -> Float { log(1 - sigma) - log(sigma) }

    private func firstOrder(_ m0: MLXArray, sample: MLXArray) -> MLXArray {
        let sigmaT = sigmas[stepIndex + 1], sigmaS = sigmas[stepIndex]
        let alphaT = 1 - sigmaT
        let h = lambda(sigmaT) - lambda(sigmaS)
        return (sigmaT / sigmaS) * sample - (alphaT * (exp(-h) - 1)) * m0
    }

    private func secondOrder(sample: MLXArray) -> MLXArray {
        let sigmaT = sigmas[stepIndex + 1], sigmaS0 = sigmas[stepIndex], sigmaS1 = sigmas[stepIndex - 1]
        let alphaT = 1 - sigmaT
        let h = lambda(sigmaT) - lambda(sigmaS0)
        let h0 = lambda(sigmaS0) - lambda(sigmaS1)
        let r0 = h0 / h
        let m0 = modelOutputs[modelOutputs.count - 1]
        let m1 = modelOutputs[modelOutputs.count - 2]
        let d0 = m0
        let d1 = (1 / r0) * (m0 - m1)
        let coefficient = alphaT * (exp(-h) - 1)
        return (sigmaT / sigmaS0) * sample - coefficient * d0 - 0.5 * coefficient * d1
    }
}
