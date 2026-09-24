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
