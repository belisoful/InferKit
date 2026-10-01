//
//  NFKMLXResumableOptimizers.swift
//  InferKitMLX
//
//  Optimizers whose per-parameter state a training checkpoint writes and a resumed run restores.
//
//  mlx-swift's `OptimizerBase` keeps its state in internal storage keyed by the parameter tree, with
//  no way to read it by parameter name or to set it, so a run resumed from a weights-only checkpoint
//  restarts Adam's moments at zero and takes its first steps several times too large. These types
//  compute exactly what mlx-swift's `Adam`, `AdamW`, and `SGD` compute, operation for operation, and
//  keep their state by parameter path where a checkpoint can reach it.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// An optimizer whose state a training checkpoint can write and a resumed run can restore.
///
/// Introduced in InferKit 0.4.0.
public protocol NFKMLXResumableOptimizer: Optimizer {

    /// The optimizer's state, keyed by parameter path and slot.
    func stateArrays() -> [String: MLXArray]

    /// Replaces the optimizer's state with arrays ``stateArrays()`` produced.
    func restore(stateArrays: [String: MLXArray]) throws
}

/// Reads and writes the state of an optimizer, including each group of a `MultiOptimizer`.
enum NFKMLXOptimizerState {

    /// The state of `optimizer`, each `MultiOptimizer` group under its index.
    static func arrays(of optimizer: Optimizer) throws -> [String: MLXArray] {
        if let resumable = optimizer as? NFKMLXResumableOptimizer {
            return resumable.stateArrays()
        }
        if let groups = optimizer as? MultiOptimizer {
            var arrays = [String: MLXArray]()
            for (index, group) in groups.optimizers.enumerated() {
                for (key, value) in try Self.arrays(of: group) {
                    arrays["\(index)/\(key)"] = value
                }
            }
            return arrays
        }
        throw NFKMLXError.unsupportedConfiguration(
            "\(type(of: optimizer)) keeps its state where a checkpoint cannot read it; use a "
            + "recipe's reference optimizer or an NFKMLXResumableOptimizer such as NFKMLXAdam")
    }

    /// Restores `arrays` into `optimizer`, each `MultiOptimizer` group from its index.
    static func restore(_ arrays: [String: MLXArray], into optimizer: Optimizer) throws {
        if let resumable = optimizer as? NFKMLXResumableOptimizer {
            try resumable.restore(stateArrays: arrays)
            return
        }
        if let groups = optimizer as? MultiOptimizer {
            for (index, group) in groups.optimizers.enumerated() {
                let prefix = "\(index)/"
                let own = arrays.filter { $0.key.hasPrefix(prefix) }
                try restore(Dictionary(uniqueKeysWithValues: own.map { (String($0.key.dropFirst(prefix.count)), $0.value) }),
                            into: group)
            }
            return
        }
        _ = try Self.arrays(of: optimizer)
    }

    /// Writes `optimizer`'s state and the number of completed updates to `url`, through a scratch
    /// file so a process killed mid-write leaves the previous file whole.
    static func save(_ optimizer: Optimizer, completedSteps: Int, to url: URL) throws {
        let arrays = try Self.arrays(of: optimizer)
        eval(Array(arrays.values))
        let scratch = url.deletingLastPathComponent().appendingPathComponent(".\(UUID().uuidString).safetensors")
        try MLX.save(arrays: arrays, metadata: [completedStepsKey: "\(completedSteps)"], url: scratch)
        _ = try FileManager.default.replaceItemAt(url, withItemAt: scratch)
    }

    /// Reads a state file ``save(_:completedSteps:to:)`` wrote, returning its arrays and step count.
    static func load(from url: URL) throws -> (arrays: [String: MLXArray], completedSteps: Int) {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        guard let steps = metadata[completedStepsKey].flatMap(Int.init) else {
            throw NFKMLXError.checkpointNotReadable("\(url.path) is not an optimizer-state file")
        }
        return (arrays, steps)
    }

    static let completedStepsKey = "inferkit.completed_steps"
}

/// Per-parameter state keyed by path, shared by the resumable optimizers.
private func restoredSlots(_ arrays: [String: MLXArray], slots: [String]) throws -> [String: [MLXArray]] {
    var grouped = [String: [String: MLXArray]]()
    for (key, value) in arrays {
        guard let dot = key.lastIndex(of: "."), slots.contains(String(key[key.index(after: dot)...])) else {
            throw NFKMLXError.checkpointNotReadable("optimizer state key \(key) names no slot of \(slots)")
        }
        grouped[String(key[..<dot]), default: [:]][String(key[key.index(after: dot)...])] = value
    }
    return try grouped.mapValues { values in
        try slots.map { slot in
            guard let value = values[slot] else {
                throw NFKMLXError.checkpointNotReadable("optimizer state is missing the \(slot) slot")
            }
            return value
        }
    }
}

/// Adam, AdamW, and Adam with L2 regularization, computing what mlx-swift's `Adam` and `AdamW`
/// compute, with state a checkpoint can carry.
///
/// - `weightDecay` shrinks the parameter by `learningRate · weightDecay` before the step, as
///   `torch.optim.AdamW` and mlx-swift's `AdamW` do.
/// - `l2` joins the gradient as `l2 · θ` before the moments, as `torch.optim.Adam`'s `weight_decay` does.
/// - `biasCorrection` applies PyTorch's correction of both moments; mlx-swift leaves it off by default.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXAdam: NFKMLXResumableOptimizer, NFKMLXRateScheduled {
    public var learningRate: Float
    public let betas: (Float, Float)
    public let eps: Float
    public let weightDecay: Float
    public let l2: Float
    public let biasCorrection: Bool

    private var moments = [String: (m: MLXArray, v: MLXArray, step: MLXArray)]()

    public init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8,
                weightDecay: Float = 0, l2: Float = 0, biasCorrection: Bool = true) {
        self.learningRate = learningRate
        self.betas = betas
        self.eps = eps
        self.weightDecay = weightDecay
        self.l2 = l2
        self.biasCorrection = biasCorrection
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let (b1, b2) = betas
        var updated = [(String, MLXArray)]()
        for (key, gradient) in gradients.flattened() {
            guard var parameter = parameters[key] else { continue }
            var gradient = gradient
            if l2 != 0 {
                gradient = gradient + l2 * parameter
            }
            if weightDecay != 0 {
                parameter = parameter * (1 - learningRate * weightDecay)
            }
            let previous = moments[key]
                ?? (MLXArray.zeros(like: parameter), MLXArray.zeros(like: parameter), MLXArray(0))
            let step = previous.step + 1
            let m = b1 * previous.m + (1 - b1) * gradient
            let v = b2 * previous.v + (1 - b2) * square(gradient)
            let change: MLXArray
            if biasCorrection {
                let c1 = learningRate / (1 - pow(b1, step))
                let c2 = rsqrt(1 - pow(b2, step))
                change = (c1 * m) / (sqrt(v) * c2 + eps)
            } else {
                change = learningRate * m / (sqrt(v) + eps)
            }
            moments[key] = (m, v, step)
            updated.append((key, parameter - change))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
    }

    public func innerState() -> [MLXArray] {
        moments.values.flatMap { [$0.m, $0.v, $0.step] }
    }

    public func stateArrays() -> [String: MLXArray] {
        var arrays = [String: MLXArray]()
        for (key, state) in moments {
            arrays["\(key).m"] = state.m
            arrays["\(key).v"] = state.v
            arrays["\(key).step"] = state.step
        }
        return arrays
    }

    public func restore(stateArrays: [String: MLXArray]) throws {
        moments = try restoredSlots(stateArrays, slots: ["m", "v", "step"]).mapValues { ($0[0], $0[1], $0[2]) }
    }
}

/// Stochastic gradient descent with momentum, computing what mlx-swift's `SGD` computes, with state a
/// checkpoint can carry. `weightDecay` joins the gradient (L2), as in `torch.optim.SGD`.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXSGD: NFKMLXResumableOptimizer, NFKMLXRateScheduled {
    public var learningRate: Float
    public let momentum: Float
    public let weightDecay: Float
    public let dampening: Float
    public let nesterov: Bool

    private var velocities = [String: MLXArray]()

    public init(learningRate: Float, momentum: Float = 0, weightDecay: Float = 0, dampening: Float = 0,
                nesterov: Bool = false) {
        self.learningRate = learningRate
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.dampening = dampening
        self.nesterov = nesterov
    }

    public func update(model: Module, gradients: ModuleParameters) {
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var updated = [(String, MLXArray)]()
        for (key, gradient) in gradients.flattened() {
            guard let parameter = parameters[key] else { continue }
            var gradient = gradient
            if weightDecay != 0 {
                gradient = gradient + weightDecay * parameter
            }
            guard momentum > 0 else {
                updated.append((key, parameter - learningRate * gradient))
                continue
            }
            var v = (velocities[key] ?? MLXArray.zeros(like: parameter)) * momentum
            v = dampening > 0 ? v + (1 - dampening) * gradient : v + gradient
            velocities[key] = v
            updated.append((key, parameter - learningRate * (nesterov ? gradient + momentum * v : v)))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
    }

    public func innerState() -> [MLXArray] {
        Array(velocities.values)
    }

    public func stateArrays() -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: velocities.map { ("\($0.key).velocity", $0.value) })
    }

    public func restore(stateArrays: [String: MLXArray]) throws {
        velocities = try restoredSlots(stateArrays, slots: ["velocity"]).mapValues { $0[0] }
    }
}
