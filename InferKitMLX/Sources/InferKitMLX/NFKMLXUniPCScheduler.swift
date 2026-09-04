//
//  NFKMLXUniPCScheduler.swift
//  InferKitMLX
//

import Foundation
import MLX

// The UniPC multistep sampler (`UniPCMultistepScheduler`), Wan's released sampler in its flow-prediction
// configuration (`solver_order: 2`, `solver_type: bh2`, `predict_x0`, `use_flow_sigmas`, flow_shift 5.0).
// A value type with no weights: a predictor-corrector over a static-shift flow sigma ramp. Each step
// converts the flow velocity to a data prediction `x0`, corrects the previous sample (from step 1 on),
// then predicts the next. The `B(h)` coefficients are `Float` scalars; the order-2 corrector's 2×2 linear
// system is solved in closed form.

/// UniPC geometry.
public struct NFKMLXUniPCConfiguration: Sendable {
    public var flowShift: Float
    public var solverOrder: Int
    public var trainTimesteps: Int

    public init(flowShift: Float = 5.0, solverOrder: Int = 2, trainTimesteps: Int = 1000) {
        self.flowShift = flowShift
        self.solverOrder = solverOrder
        self.trainTimesteps = trainTimesteps
    }

    /// Wan's released configuration (flow_shift 5.0, second-order).
    public static let wan = NFKMLXUniPCConfiguration(flowShift: 5.0)
}

public struct NFKMLXUniPCScheduler {
    public let configuration: NFKMLXUniPCConfiguration
    public private(set) var sigmas: [Float] = []
    public private(set) var timesteps: [Float] = []

    private var stepIndex = 0
    private var modelOutputs: [MLXArray?] = []                             // the data-prediction (x0) history
    private var lastSample: MLXArray?
    private var lowerOrderNums = 0
    private var currentOrder = 0

    public init(_ configuration: NFKMLXUniPCConfiguration = .wan) {
        self.configuration = configuration
    }

    public mutating func setTimesteps(_ steps: Int, sequenceLength: Int = 0) {
        let train = Float(configuration.trainTimesteps)
        let shift = configuration.flowShift
        var s = (0 ... steps).map { index -> Float in
            let alpha = 1 - (1 - 1 / train) * Float(index) / Float(steps)
            let raw = 1 - alpha
            return shift * raw / (1 + (shift - 1) * raw)
        }
        s.reverse()
        s.removeLast()
        timesteps = s.map { ($0 * train).rounded(.towardZero) }
        sigmas = s + [0]
        stepIndex = 0
        modelOutputs = Array(repeating: nil, count: configuration.solverOrder)
        lastSample = nil
        lowerOrderNums = 0
        currentOrder = 0
    }

    private func lambda(_ sigma: Float) -> Float { log(1 - sigma) - log(sigma) }

    public mutating func step(velocity: MLXArray, sample: MLXArray, index: Int) -> MLXArray {
        let i = stepIndex
        let x0 = sample - sigmas[i] * velocity                            // flow → data prediction

        var corrected = sample
        if i > 0, lastSample != nil {
            corrected = corrector(thisX0: x0, order: currentOrder)
        }

        for j in 0 ..< (configuration.solverOrder - 1) { modelOutputs[j] = modelOutputs[j + 1] }
        modelOutputs[modelOutputs.count - 1] = x0

        let target = min(configuration.solverOrder, timesteps.count - i)   // lower_order_final
        currentOrder = min(target, lowerOrderNums + 1)
        lastSample = corrected

        let previous = predictor(currentX0: x0, sample: corrected, order: currentOrder)
        if lowerOrderNums < configuration.solverOrder { lowerOrderNums += 1 }
        stepIndex += 1
        return previous
    }

    /// The B(h) scalar coefficients shared by the predictor and corrector.
    private func coefficients(sigmaT: Float, sigmaS0: Float) -> (c1: Float, phi1: Float, bh: Float, hh: Float) {
        let h = lambda(sigmaT) - lambda(sigmaS0)
        let hh = -h                                                        // predict_x0
        let phi1 = expm1(hh)                                              // e^{hh} − 1
        return (sigmaT / sigmaS0, phi1, expm1(hh), hh)
    }

    private func predictor(currentX0 m0: MLXArray, sample: MLXArray, order: Int) -> MLXArray {
        let sigmaT = sigmas[stepIndex + 1], sigmaS0 = sigmas[stepIndex]
        let alphaT = 1 - sigmaT
        let (c1, phi1, bh, _) = coefficients(sigmaT: sigmaT, sigmaS0: sigmaS0)
        var result = c1 * sample - (alphaT * phi1) * m0
        if order >= 2, let m1 = modelOutputs[modelOutputs.count - 2] {
            let h = lambda(sigmaT) - lambda(sigmaS0)
            let rk = (lambda(sigmas[stepIndex - 1]) - lambda(sigmaS0)) / h
            let d1 = (1 / rk) * (m1 - m0)                                 // (mi − m0) / rk
            result = result - (alphaT * bh) * (0.5 * d1)                  // rhos_p = [0.5] at order 2
        }
        return result
    }

    private func corrector(thisX0 modelT: MLXArray, order: Int) -> MLXArray {
        let sigmaT = sigmas[stepIndex], sigmaS0 = sigmas[stepIndex - 1]
        let alphaT = 1 - sigmaT
        let (c1, phi1, bh, hh) = coefficients(sigmaT: sigmaT, sigmaS0: sigmaS0)
        guard let m0 = modelOutputs[modelOutputs.count - 1], let x = lastSample else { return modelT }
        let d1t = modelT - m0
        var result = c1 * x - (alphaT * phi1) * m0

        if order >= 2, let m1 = modelOutputs[modelOutputs.count - 2] {
            let h = lambda(sigmaT) - lambda(sigmaS0)
            let rk = (lambda(sigmas[stepIndex - 2]) - lambda(sigmaS0)) / h
            let d1 = (1 / rk) * (m1 - m0)
            // Solve [[1, 1], [rk, 1]] · rhos = b for the two-term corrector.
            let phiK0 = phi1 / hh - 1
            let b0 = phiK0 / bh
            let phiK1 = phiK0 / hh - 0.5
            let b1 = phiK1 * 2 / bh
            let rho0 = (b0 - b1) / (1 - rk)
            let rho1 = b0 - rho0
            result = result - (alphaT * bh) * (rho0 * d1 + rho1 * d1t)
        } else {
            result = result - (alphaT * bh) * (0.5 * d1t)                 // rhos_c = [0.5] at order 1
        }
        return result
    }
}
