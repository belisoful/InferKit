//
//  NFKMLXLoRA.swift
//  InferKitMLX
//
//  Low-rank adaptation, for the models where head-only is too weak and full fine-tuning is too heavy.
//
//  A transformer stack has no small head to train: SegFormer's decode head works because the encoder
//  already produces the right features, but adapting CLIP or Whisper to a consumer's own domain means
//  reaching into the attention blocks. Doing that fully needs an optimizer state proportional to the
//  whole model, which a device does not have. LoRA replaces each targeted `Linear` with the same layer
//  plus a rank-r detour, and trains only the detour: the parameter count falls by roughly the ratio of
//  the rank to the layer width, and the optimizer state with it.
//
//  The adapters merge back into the base weights before saving, so a customized model is still one
//  ordinary safetensors that the model's existing factory loads. There is no adapter format to carry
//  around and no second loading path.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// A `Linear` with a trainable low-rank detour: `y = Wx + b + scale·(x·A)·B`.
///
/// `B` starts at zero, so an adapted model produces exactly what it produced before training. That
/// matters for a fine-tune, where the starting point is a checkpoint worth preserving.
public final class NFKMLXLoRALinear: Linear {

    @ParameterInfo(key: "lora_a") public var loraA: MLXArray
    @ParameterInfo(key: "lora_b") public var loraB: MLXArray

    /// `alpha / rank`, the conventional LoRA scaling.
    public let scale: Float

    /// Wraps `base`, keeping its weights and adding the detour.
    public init(base: Linear, rank: Int, alpha: Float) {
        let (outputs, inputs) = base.shape
        let deviation = sqrt(1.0 / Float(inputs))
        self._loraA.wrappedValue = MLXRandom.uniform(-deviation ..< deviation, [inputs, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputs])
        self.scale = alpha / Float(rank)
        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(x) + matmul(matmul(x, loraA), loraB) * scale
    }

    /// The equivalent plain `Linear`, with the detour folded into the weights.
    ///
    /// `Linear` computes `x·Wᵀ`, so the detour `x·A·B` corresponds to a weight delta of `(A·B)ᵀ`.
    public func merged() -> Linear {
        Linear(weight: weight + matmul(loraA, loraB).transposed() * scale, bias: bias)
    }
}

/// Adds and removes LoRA adapters across a model.
public enum NFKMLXLoRA {

    /// Replaces every `Linear` the predicate selects with an adapted one, then freezes everything
    /// except the adapters.
    ///
    /// - Parameters:
    ///   - model: the model to adapt, in place.
    ///   - rank: the detour's width. 4–16 covers most fine-tunes; cost grows linearly with it.
    ///   - alpha: the adapter's strength, applied as `alpha / rank`.
    ///   - predicate: receives each candidate's key path and layer. The default adapts every `Linear`,
    ///     which is rarely what a consumer wants: targeting the attention projections is the usual
    ///     choice and is far cheaper.
    /// - Returns: how many layers were adapted, so a caller can tell a predicate that matched nothing
    ///   from one that matched everything.
    ///
    /// Layers already adapted are skipped, so applying twice does not stack detours.
    ///
    /// - Throws: `NFKMLXError.loRANotApplicable` when a selected layer cannot be replaced because
    ///   its declaring model stores it in a plain property. MLX can only substitute a child module
    ///   through a `@ModuleInfo` wrapper, and the non-throwing path aborts the process rather than
    ///   reporting it. Also when a selected layer is quantized — see the note below.
    ///
    /// **A quantized layer cannot be adapted here.** `QuantizedLinear` is a subclass of `Linear`, so
    /// it satisfies the predicate's type without announcing itself, and its `weight` is packed
    /// integers rather than the values the layer computes with. Wrapping one would build a detour
    /// around a weight that is not a weight. Load the model at float precision to fine-tune it.
    @discardableResult
    public static func apply(to model: Module, rank: Int = 8, alpha: Float = 16,
                             where predicate: (String, Linear) -> Bool = { _, _ in true }) throws -> Int {
        var quantized: [String] = []
        let replacements = model.leafModules().flattened().compactMap { path, layer -> (String, Module)? in
            guard let linear = layer as? Linear, !(linear is NFKMLXLoRALinear),
                  predicate(path, linear) else {
                return nil
            }
            if linear is QuantizedLinear {
                quantized.append(path)
                return nil
            }
            return (path, NFKMLXLoRALinear(base: linear, rank: rank, alpha: alpha))
        }
        guard quantized.isEmpty else {
            throw NFKMLXError.loRANotApplicable(
                "\(quantized.count) selected layer(s) are quantized (\(quantized.prefix(3).joined(separator: ", "))"
                + "\(quantized.count > 3 ? ", …" : "")). A quantized layer stores packed integers "
                + "where a Linear stores its weights, so an adapter built around one adapts nothing. "
                + "Load the model at float precision, or exclude these layers in the predicate.")
        }
        do {
            try replace(replacements, in: model)
        } catch {
            throw NFKMLXError.loRANotApplicable(
                "a selected layer cannot be adapted: \(error). A model exposes a layer for replacement "
                + "by declaring it with @ModuleInfo; a plain property cannot receive one.")
        }

        // Freezing everything and reopening only the adapters is what makes the run cheap: a frozen
        // parameter produces no gradient and carries no optimizer state.
        model.freeze()
        model.unfreeze(keys: ["lora_a", "lora_b"])
        return replacements.count
    }

    /// Folds every adapter back into its base weights, leaving plain `Linear` layers.
    ///
    /// Call this before saving. The result is an ordinary checkpoint: the model's existing
    /// `backendWith…weightsURL:` factory loads it, with no adapter format and no second loading path.
    ///
    /// - Returns: how many adapters were merged.
    ///
    /// - Throws: `NFKMLXError.loRANotApplicable` if a replacement cannot be written back, which
    ///   cannot happen for adapters this type applied, or if the model has been quantized since the
    ///   adapters were applied — see the note below.
    ///
    /// **Never merge into a quantized base.** A merged weight is `W + (A·B)ᵀ·scale`, and a rank-r
    /// detour's contribution to any one weight is small by construction. Requantizing that sum snaps
    /// every contribution below half a quantization step back to the value it started from, so the
    /// training is discarded silently: the file is the right size, loads without complaint, and holds
    /// the original model. Merge into full-precision weights and quantize afterward, in that order.
    @discardableResult
    public static func merge(into model: Module) throws -> Int {
        var quantized: [String] = []
        let replacements = model.leafModules().flattened().compactMap { path, layer -> (String, Module)? in
            if layer is QuantizedLinear {
                quantized.append(path)
                return nil
            }
            guard let adapted = layer as? NFKMLXLoRALinear else {
                return nil
            }
            return (path, adapted.merged())
        }
        guard quantized.isEmpty else {
            throw NFKMLXError.loRANotApplicable(
                "the model holds \(quantized.count) quantized layer(s); merging a low-rank delta into "
                + "a quantized weight and requantizing rounds the delta away, discarding the training "
                + "without reporting anything. Merge at float precision, then quantize.")
        }
        do {
            try replace(replacements, in: model)
        } catch {
            throw NFKMLXError.loRANotApplicable("an adapter could not be merged back: \(error)")
        }
        model.unfreeze()
        return replacements.count
    }

    /// Writes each `(path, module)` replacement into `model`. The fast path is one net-wide update. It
    /// fails when the adapters cover a SUBSET of a module array — a hybrid decoder that adapts only its
    /// attention layers, whose gapped indices `unflattened` renders as a dictionary that
    /// `update(modules:)` cannot route into an array. The fallback then reaches each replacement's
    /// owning module and updates a single-module leaf key, which never crosses an array. It runs only
    /// after the fast path throws, so a model whose adapters form a full array is updated exactly as
    /// before.
    private static func replace(_ replacements: [(String, Module)], in model: Module) throws {
        do {
            try model.update(modules: ModuleChildren.unflattened(replacements), verify: .none)
        } catch {
            for (path, replacement) in replacements {
                let components = path.split(separator: ".").map(String.init)
                guard let leaf = components.last,
                      let owner = owningModule(Array(components.dropLast()), of: model) else {
                    throw error
                }
                try owner.update(modules: ModuleChildren.unflattened([(leaf, replacement)]), verify: .none)
            }
        }
    }

    /// The module at `components` under `root`, descending `@ModuleInfo` children, module arrays (an
    /// integer index component), and module dictionaries (a key component).
    private static func owningModule(_ components: [String], of root: Module) -> Module? {
        var current = root
        var index = 0
        while index < components.count {
            guard let item = current.items()[components[index]] else { return nil }
            switch item {
            case .value(.module(let child)):
                current = child
                index += 1
            case .array(let elements):
                guard index + 1 < components.count, let position = Int(components[index + 1]),
                      position >= 0, position < elements.count,
                      case .value(.module(let child)) = elements[position] else { return nil }
                current = child
                index += 2
            case .dictionary(let entries):
                guard index + 1 < components.count,
                      case .value(.module(let child))? = entries[components[index + 1]] else { return nil }
                current = child
                index += 2
            default:
                return nil
            }
        }
        return current
    }

    /// How many parameters a run would train, so a caller can size it before starting.
    public static func trainableParameterCount(of model: Module) -> Int {
        model.trainableParameters().flattened().reduce(0) { $0 + $1.1.size }
    }
}
