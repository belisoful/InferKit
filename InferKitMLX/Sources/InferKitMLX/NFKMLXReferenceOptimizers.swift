//
//  NFKMLXReferenceOptimizers.swift
//  InferKitMLX
//
//  The optimizers the fine-tuning recipes default to, built the way PyTorch builds them.
//
//  mlx-swift's `Adam` and `AdamW` leave the bias correction of the moment estimates out unless asked
//  for it, and every PyTorch Adam applies it. Without it each update is `(1 − β1ᵗ) / √(1 − β2ᵗ)` times
//  the reference's: with betas 0.9 and 0.999 that is 3.2 at step one, about 6.5 near step ten, and
//  still 1.3 at step one thousand, so a short on-device run takes steps several times larger than the
//  recipe it ports (with β1 0.5, sixteen times at step one). The training configurations the recipes
//  port also name parameter groups: SAM 2, SAM 3, and SegFormer give biases and normalization weights
//  no weight decay, SegFormer trains its decode head at ten times the base rate, and SAM 2 trains its
//  image encoder at a lower rate that falls further per trunk layer.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

enum NFKMLXReferenceOptimizers {

    /// `torch.optim.AdamW`: bias-corrected, with the weight decay applied to the parameter rather than
    /// the gradient.
    static func adamW(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8,
                      weightDecay: Float) -> AdamW {
        AdamW(learningRate: learningRate, betas: betas, eps: eps, weightDecay: weightDecay, biasCorrection: true)
    }

    /// `torch.optim.AdamW` over two parameter groups: `weightDecay` on every parameter the predicate
    /// does not exempt, none on the ones it does.
    static func adamW(learningRate: Float, weightDecay: Float,
                      exempting exempt: @escaping (String) -> Bool) -> Optimizer {
        MultiOptimizer(optimizers: [adamW(learningRate: learningRate, weightDecay: 0),
                                    adamW(learningRate: learningRate, weightDecay: weightDecay)],
                       filters: [{ key, _ in exempt(key) }])
    }

    /// `torch.optim.AdamW` over the parameter groups a configuration names: `group` gives each of
    /// `module`'s trainable parameters its rate (as a multiple of `learningRate`) and its weight decay,
    /// and the parameters that agree on both share one optimizer. Build it after freezing.
    static func adamW(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), over module: Module,
                      group: (String) -> (rateScale: Float, weightDecay: Float)) -> Optimizer {
        var settings = [(rateScale: Float, weightDecay: Float)]()
        var members = [Set<String>]()
        for (key, _) in module.trainableParameters().flattened() {
            let setting = group(key)
            if let index = settings.firstIndex(where: { $0 == setting }) {
                members[index].insert(key)
            } else {
                settings.append(setting)
                members.append([key])
            }
        }
        let optimizers = settings.map {
            adamW(learningRate: learningRate * $0.rateScale, betas: betas, weightDecay: $0.weightDecay)
        }
        guard optimizers.count > 1 else {
            return optimizers.first ?? adamW(learningRate: learningRate, betas: betas, weightDecay: 0)
        }
        let filters = members.dropLast().map { keys in { (key: String, _: MLXArray) in keys.contains(key) } }
        return MultiOptimizer(optimizers: optimizers, filters: Array(filters))
    }

    /// timm's `RAdam` as `create_optimizer_v2` builds it: two groups, with no weight decay on a
    /// parameter of one dimension or fewer or one whose name ends in `.bias` (timm's
    /// `param_groups_weight_decay`), and `weightDecay` on the rest.
    static func rAdam(learningRate: Float, weightDecay: Float) -> Optimizer {
        MultiOptimizer(optimizers: [NFKMLXRAdam(learningRate: learningRate, weightDecay: 0),
                                    NFKMLXRAdam(learningRate: learningRate, weightDecay: weightDecay)],
                       filters: [{ key, parameter in parameter.ndim <= 1 || key.hasSuffix(".bias") }])
    }

    /// The paths of the `LayerNorm`s under `module`, each with a trailing `.`, for a configuration that
    /// exempts `torch.nn.LayerNorm` by class. `excluding` names the ones that are another class in the
    /// reference (a channel-wise `LayerNorm2d`, built here as a `LayerNorm`), matched as path suffixes.
    static func layerNormPrefixes(in module: Module, excluding: [String] = []) -> [String] {
        module.leafModules().flattened().compactMap { path, leaf in
            guard leaf is LayerNorm, !excluding.contains(where: { path.hasSuffix($0) }) else { return nil }
            return path + "."
        }
    }

    /// The SAM configurations' no-decay group: a parameter whose name contains `bias`, and every
    /// parameter of a `torch.nn.LayerNorm`.
    static func biasOrLayerNorm(_ layerNorms: [String]) -> (String) -> Bool {
        { key in key.contains("bias") || layerNorms.contains(where: { key.hasPrefix($0) }) }
    }
}

/// `torch.optim.Adam` with its `weight_decay`: the decay joins the gradient before the moment estimates
/// (L2 regularization), where AdamW shrinks the parameter directly. Bias-corrected, as torch's is.
final class NFKMLXL2Adam: Adam {
    let weightDecay: Float

    init(learningRate: Float, weightDecay: Float) {
        self.weightDecay = weightDecay
        super.init(learningRate: learningRate, biasCorrection: true)
    }

    override func applySingle(gradient: MLXArray, parameter: MLXArray, state: AdamState) -> (MLXArray, AdamState) {
        super.applySingle(gradient: gradient + weightDecay * parameter, parameter: parameter, state: state)
    }
}

/// timm's `RAdam` (`timm/optim/radam.py`, unchanged through timm 0.9): Adam with the variance of the
/// adaptive rate rectified. While the length of the approximated simple moving average is under 5 the
/// step is the bias-corrected momentum alone, unnormalized; from there on it is the bias-corrected
/// Adam step scaled by the rectification term. The weight decay shrinks the parameter directly by
/// `weightDecay · learningRate`, decoupled from the gradient, on every step.
///
/// It adopts `Optimizer` directly because mlx-swift's `OptimizerBase` and `AdamState` expose no
/// initializer outside their module. The per-step scalars are computed in double precision, as the
/// reference computes them in Python.
final class NFKMLXRAdam: Optimizer, NFKMLXRateScheduled {
    var learningRate: Float
    let betas: (Double, Double)
    let eps: Float
    let weightDecay: Float

    /// Each parameter's moments and step, keyed by its flattened path.
    private var moments = [String: (m: MLXArray, v: MLXArray, step: Int)]()

    init(learningRate: Float, betas: (Double, Double) = (0.9, 0.999), eps: Float = 1e-8, weightDecay: Float = 0) {
        self.learningRate = learningRate
        self.betas = betas
        self.eps = eps
        self.weightDecay = weightDecay
    }

    func update(model: Module, gradients: ModuleParameters) {
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let (b1, b2) = betas
        var updated = [(String, MLXArray)]()
        for (key, gradient) in gradients.flattened() {
            guard let parameter = parameters[key] else { continue }
            let previous = moments[key] ?? (MLXArray.zeros(like: parameter), MLXArray.zeros(like: parameter), 0)
            let step = previous.step + 1
            let m = Float(b1) * previous.m + Float(1 - b1) * gradient
            let v = Float(b2) * previous.v + Float(1 - b2) * square(gradient)
            moments[key] = (m, v, step)

            let b2t: Double = Foundation.pow(b2, Double(step))
            let momentumCorrection: Double = 1 - Foundation.pow(b1, Double(step))
            let smaMaximum: Double = 2 / (1 - b2) - 1
            let sma: Double = smaMaximum - 2 * Double(step) * b2t / (1 - b2t)
            let rate = Double(learningRate)
            let stepSize: Double
            if sma >= 5 {
                let variance: Double = (1 - b2t) * (sma - 4) / (smaMaximum - 4)
                let rectification: Double = (variance * (sma - 2) / sma * smaMaximum / (smaMaximum - 2)).squareRoot()
                stepSize = rate * rectification / momentumCorrection
            } else {
                stepSize = rate / momentumCorrection
            }
            var next = parameter
            if weightDecay != 0 {
                next = next - Float(Double(weightDecay) * rate) * next
            }
            let direction = sma >= 5 ? m / (sqrt(v) + eps) : m
            updated.append((key, next - Float(stepSize) * direction))
        }
        model.update(parameters: ModuleParameters.unflattened(updated))
    }

    func innerState() -> [MLXArray] {
        moments.values.flatMap { [$0.m, $0.v] }
    }
}
