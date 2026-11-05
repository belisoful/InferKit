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
//  A quantized base takes an adapter too (QLoRA): the packed weights stay frozen, the gradient reaches
//  the detour through the quantized product, and the adapters are saved on their own, since merging
//  into a quantized weight rounds the training away.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// A layer carrying a trainable low-rank detour `scale·(x·A)·B` beside its base product: an
/// ``NFKMLXLoRALinear`` over a float base or an ``NFKMLXQuantizedLoRALinear`` over a quantized one.
///
/// Introduced in InferKit 0.4.0.
public protocol NFKMLXLoRAAdapter: Linear {
    /// `A` `[inputs, rank]`, saved as `lora_a`.
    var loraA: MLXArray { get }
    /// `B` `[rank, outputs]`, saved as `lora_b`.
    var loraB: MLXArray { get }
    /// `alpha / rank`.
    var scale: Float { get }
    /// The adapter's weight on ``scale``; 0 turns the detour off.
    var strength: Float { get set }
}

/// A `Linear` with a trainable low-rank detour: `y = Wx + b + scale·(x·A)·B`.
///
/// `B` starts at zero, so an adapted model produces exactly what it produced before training. That
/// matters for a fine-tune, where the starting point is a checkpoint worth preserving.
public final class NFKMLXLoRALinear: Linear, NFKMLXLoRAAdapter {

    @ParameterInfo(key: "lora_a") public var loraA: MLXArray
    @ParameterInfo(key: "lora_b") public var loraB: MLXArray

    /// `alpha / rank`, the conventional LoRA scaling.
    public let scale: Float

    /// The adapter's weight, multiplying ``scale`` as peft's `set_adapters(adapter_weights:)` does; 0 turns
    /// the detour off without computing it. 1 by default.
    ///
    /// Introduced in InferKit 0.4.0.
    public var strength: Float = 1

    /// The dropout on the detour's input while the layer trains, as peft's `lora_dropout` applies it
    /// (`lora_B(lora_A(dropout(x)))`); the base path never drops. 0 by default.
    ///
    /// Introduced in InferKit 0.4.0.
    public var dropout: Float

    /// The type the layer's output is cast to, as PEFT casts a float32 adapter's sum back to the type of
    /// the base projection's output (`torch_result_dtype`). Nil keeps the type the sum promotes to.
    ///
    /// Introduced in InferKit 0.4.0.
    public var resultType: DType?

    /// Wraps `base`, keeping its weights and adding the detour.
    public init(base: Linear, rank: Int, alpha: Float, dropout: Float = 0) {
        self.dropout = dropout
        let (outputs, inputs) = base.shape
        let deviation = sqrt(1.0 / Float(inputs))
        self._loraA.wrappedValue = MLXRandom.uniform(-deviation ..< deviation, [inputs, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputs])
        self.scale = alpha / Float(rank)
        super.init(weight: base.weight, bias: base.bias)
    }

    /// Wraps `base` with a released adapter's halves as the file stores them, `down` `[rank, inputs]` and
    /// `up` `[outputs, rank]`. `base` may be a streamed block's empty placeholder, which the stream fills
    /// in its turn.
    init(base: Linear, down: MLXArray, up: MLXArray, alpha: Float) {
        self.dropout = 0
        self._loraA.wrappedValue = down.transposed()
        self._loraB.wrappedValue = up.transposed()
        self.scale = alpha / Float(down.dim(0))
        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard strength != 0 else { return super.callAsFunction(x) }
        let sum = super.callAsFunction(x)
            + matmul(matmul(NFKDropout.apply(x, rate: dropout, active: training), loraA), loraB) * (scale * strength)
        return resultType.map { sum.asType($0) } ?? sum
    }

    /// The equivalent plain `Linear`, with the detour folded into the weights.
    ///
    /// `Linear` computes `x·Wᵀ`, so the detour `x·A·B` corresponds to a weight delta of `(A·B)ᵀ`.
    public func merged() -> Linear {
        Linear(weight: weight + matmul(loraA, loraB).transposed() * scale, bias: bias)
    }
}

/// A `QuantizedLinear` with a trainable low-rank detour (QLoRA): `y = Q(W)x + b + scale·(x·A)·B`. The
/// packed weight, its scales, and its biases stay frozen; the detour trains in the input's type.
///
/// @discussion The adapter cannot be merged: folding a rank-r delta into a quantized weight and
/// requantizing rounds it away. ``NFKMLXLoRA/adapterArrays(of:)`` saves the adapters on their own, and
/// ``NFKMLXLoRA/loadAdapters(_:into:)`` places them over the same quantized base.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXQuantizedLoRALinear: QuantizedLinear, NFKMLXLoRAAdapter {

    @ParameterInfo(key: "lora_a") public var loraA: MLXArray
    @ParameterInfo(key: "lora_b") public var loraB: MLXArray

    /// `alpha / rank`, the conventional LoRA scaling.
    public let scale: Float

    /// The adapter's weight, multiplying ``scale``; 0 turns the detour off. 1 by default.
    ///
    /// Introduced in InferKit 0.4.0.
    public var strength: Float = 1

    /// The dropout on the detour's input while the layer trains, as peft's `lora_dropout` applies it.
    public var dropout: Float

    /// Wraps `base`, keeping its packed weights and adding the detour.
    public init(base: QuantizedLinear, rank: Int, alpha: Float, dropout: Float = 0) {
        self.dropout = dropout
        let (outputs, inputs) = base.shape
        let deviation = sqrt(1.0 / Float(inputs))
        self._loraA.wrappedValue = MLXRandom.uniform(-deviation ..< deviation, [inputs, rank])
        self._loraB.wrappedValue = MLXArray.zeros([rank, outputs])
        self.scale = alpha / Float(rank)
        super.init(weight: base.weight, bias: base.bias, scales: base.scales, biases: base.biases,
                   groupSize: base.groupSize, bits: base.bits, mode: base.mode, globalScale: base.globalScale)
    }

    /// Wraps `base` with a released adapter's halves as the file stores them, `down` `[rank, inputs]` and
    /// `up` `[outputs, rank]`.
    init(base: QuantizedLinear, down: MLXArray, up: MLXArray, alpha: Float) {
        self.dropout = 0
        self._loraA.wrappedValue = down.transposed()
        self._loraB.wrappedValue = up.transposed()
        self.scale = alpha / Float(down.dim(0))
        super.init(weight: base.weight, bias: base.bias, scales: base.scales, biases: base.biases,
                   groupSize: base.groupSize, bits: base.bits, mode: base.mode, globalScale: base.globalScale)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard strength != 0 else { return super.callAsFunction(x) }
        let detour = matmul(matmul(NFKDropout.apply(x, rate: dropout, active: training),
                                   loraA.asType(x.dtype)), loraB.asType(x.dtype))
        return super.callAsFunction(x) + detour * (scale * strength)
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
    ///   - dropout: the dropout on each detour's input while the model trains, peft's `lora_dropout`.
    ///     0, the default, drops nothing.
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
    ///   reporting it.
    ///
    /// A selected `QuantizedLinear` becomes an ``NFKMLXQuantizedLoRALinear`` (QLoRA): its packed
    /// weights stay frozen and the detour trains over them. Such a model cannot be merged; save its
    /// adapters with ``adapterArrays(of:)``.
    @discardableResult
    public static func apply(to model: Module, rank: Int = 8, alpha: Float = 16, dropout: Float = 0,
                             where predicate: (String, Linear) -> Bool = { _, _ in true }) throws -> Int {
        let replacements = model.leafModules().flattened().compactMap { path, layer -> (String, Module)? in
            guard let linear = layer as? Linear, !(linear is NFKMLXLoRAAdapter),
                  predicate(path, linear) else {
                return nil
            }
            let adapted: Linear = (linear as? QuantizedLinear).map {
                NFKMLXQuantizedLoRALinear(base: $0, rank: rank, alpha: alpha, dropout: dropout)
            } ?? NFKMLXLoRALinear(base: linear, rank: rank, alpha: alpha, dropout: dropout)
            // A new module starts in training mode; the adapter takes the mode of the layer it wraps.
            adapted.train(linear.training)
            return (path, adapted)
        }
        do {
            try NFKMLXModuleReplacement.place(replacements, in: model)
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
            try NFKMLXModuleReplacement.place(replacements, in: model)
        } catch {
            throw NFKMLXError.loRANotApplicable("an adapter could not be merged back: \(error)")
        }
        model.unfreeze()
        NFKMLXTrainer.freezeHeldState(of: model)
        return replacements.count
    }

    /// Every adapter's `A` and `B`, keyed `<layer path>.lora_a` / `.lora_b`: the whole of a run over a
    /// quantized base, which cannot be merged. `MLX.save(arrays:url:)` writes them.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func adapterArrays(of model: Module) -> [String: MLXArray] {
        var arrays: [String: MLXArray] = [:]
        for (path, layer) in model.leafModules().flattened() {
            guard let adapter = layer as? NFKMLXLoRAAdapter else { continue }
            arrays["\(path).lora_a"] = adapter.loraA
            arrays["\(path).lora_b"] = adapter.loraB
        }
        return arrays
    }

    /// Places saved adapters over a model that ``apply(to:rank:alpha:dropout:where:)`` adapted the same
    /// way, so a run's adapters reload over a freshly loaded base.
    ///
    /// - Returns: how many adapters were placed.
    /// - Throws: `NFKMLXError.loRANotApplicable` when an array names no adapter of the model, an
    ///   adapter has no saved array, or a shape differs.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func loadAdapters(_ arrays: [String: MLXArray], into model: Module) throws -> Int {
        let adapters = Dictionary(uniqueKeysWithValues: model.leafModules().flattened().compactMap { path, layer in
            (layer as? NFKMLXLoRAAdapter).map { (path, $0) }
        })
        let named = Set(arrays.keys.map { key in String(key.split(separator: ".").dropLast().joined(separator: ".")) })
        if let stray = named.first(where: { adapters[$0] == nil }) {
            throw NFKMLXError.loRANotApplicable("the saved adapter \(stray) names no adapted layer of the model")
        }
        for (path, adapter) in adapters {
            guard let a = arrays["\(path).lora_a"], let b = arrays["\(path).lora_b"] else {
                throw NFKMLXError.loRANotApplicable("no saved adapter for \(path)")
            }
            guard a.shape == adapter.loraA.shape, b.shape == adapter.loraB.shape else {
                throw NFKMLXError.loRANotApplicable(
                    "the saved adapter \(path) is \(a.shape)·\(b.shape); the layer's is "
                    + "\(adapter.loraA.shape)·\(adapter.loraB.shape)")
            }
            _ = adapter.update(parameters: ModuleParameters.unflattened(["lora_a": a, "lora_b": b]))
        }
        return adapters.count
    }

    /// How many parameters a run would train, so a caller can size it before starting. Running
    /// statistics and quantized arrays that an `unfreeze()` thawed are not counted, because the run
    /// freezes them again.
    public static func trainableParameterCount(of model: Module) -> Int {
        let held = Set(model.leafModules().flattened().flatMap { path, module in
            NFKMLXTrainer.heldKeys(of: module).map { path.isEmpty ? $0 : "\(path).\($0)" }
        })
        return model.trainableParameters().flattened()
            .filter { !held.contains($0.0) }
            .reduce(0) { $0 + $1.1.size }
    }
}
