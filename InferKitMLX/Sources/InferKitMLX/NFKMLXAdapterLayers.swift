//
//  NFKMLXAdapterLayers.swift
//  InferKitMLX
//
//  Released low-rank adapters held beside the weights they adapt rather than merged into them. Each
//  adapted layer keeps its own weights and adds one rank-r detour per adapter, so several adapters
//  stack, and each adapter's strength changes per run without touching a weight.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// The strengths a run gives the adapters a model holds unmerged.
///
/// @discussion A run sets ``strengths`` for its own task, one strength per adapter in load order; a
/// detour reads its adapter's entry and falls back to the strength it was loaded at where the run sets
/// none. The value is task-local, so two runs of one model in flight at once each read their own.
enum NFKMLXAdapterRun {
    @TaskLocal static var strengths: [Float]?
}

/// One adapter's low-rank detour beside a layer: `strength · scale · up(down(x))`, as PEFT's LoRA
/// layer adds `lora_B(lora_A(x)) · scaling` to the base layer's output.
///
/// @discussion `down` is `[rank, inputs]` for a linear layer and the kernel `[rank, kH, kW, inputs]` for
/// a convolution, applied with the base convolution's stride, padding, and dilation. `up` is
/// `[outputs, rank]`, a 1×1 projection in either case.
final class NFKMLXLowRankDetour {
    let down: MLXArray
    let up: MLXArray
    /// `alpha / rank`, or 1 for a file that states no alpha.
    let scale: Double
    /// The adapter's position in load order, which indexes ``NFKMLXAdapterRun/strengths``.
    let adapter: Int
    /// The strength the adapter was loaded at.
    let strength: Float

    init(down: MLXArray, up: MLXArray, scale: Double, adapter: Int, strength: Float) {
        self.down = down
        self.up = up
        self.scale = scale
        self.adapter = adapter
        self.strength = strength
    }

    /// The strength this run gives the adapter.
    var runStrength: Float {
        guard let strengths = NFKMLXAdapterRun.strengths, strengths.indices.contains(adapter) else {
            return strength
        }
        return strengths[adapter]
    }

    /// The factor on the rank-r path for a run strength: PEFT's `scaling`, the product formed in double
    /// precision as Python forms it.
    private func factor(_ strength: Float) -> Float {
        Float(Double(strength) * scale)
    }

    /// The rank-r path of a linear layer for `x`, scaled, or nil where the run gives the adapter no
    /// strength.
    func linear(_ x: MLXArray) -> MLXArray? {
        let strength = runStrength
        guard strength != 0 else { return nil }
        let reduced = matmul(x, down.asType(x.dtype).transposed(1, 0))
        return matmul(reduced, up.asType(x.dtype).transposed(1, 0)) * factor(strength)
    }

    /// The rank-r path of `convolution` for `x`, scaled, or nil where the run gives the adapter no
    /// strength.
    func convolution(_ x: MLXArray, of convolution: Conv2d) -> MLXArray? {
        let strength = runStrength
        guard strength != 0 else { return nil }
        let reduced = conv2d(x, down.asType(x.dtype), stride: .init(convolution.stride),
                             padding: .init(convolution.padding), dilation: .init(convolution.dilation))
        return matmul(reduced, up.asType(x.dtype).transposed(1, 0)) * factor(strength)
    }

    /// The bytes the detour holds.
    var byteCount: Int { down.nbytes + up.nbytes }
}

/// The detours beside one layer in load order, and the layer they were placed beside. A reference type
/// the module reflection passes over, so neither the arrays nor the layer are parameters twice.
final class NFKMLXDetourList {
    var detours: [NFKMLXLowRankDetour] = []
    /// The replaced layer, which computes the base product with its own rounding (``NFKLinear``,
    /// ``NFKConv2d``) on the same weights the adapted layer reports.
    let base: Module

    init(base: Module) {
        self.base = base
    }
}

/// The detours beside each part of a fused projection, such as the query, key, and value parts of
/// CLIP's `in_proj_weight`, which a released adapter adapts as three separate layers.
final class NFKMLXFusedDetours {
    var parts: [[NFKMLXLowRankDetour]]

    init(parts: Int) {
        self.parts = Array(repeating: [], count: parts)
    }

    var isEmpty: Bool { parts.allSatisfy(\.isEmpty) }

    /// `outputs`, one per part of the fused projection of `x`, each with its detours added.
    func adding(to outputs: [MLXArray], input x: MLXArray) -> [MLXArray] {
        zip(outputs, parts).map { output, detours in
            detours.reduce(output) { output, detour in detour.linear(x).map { output + $0 } ?? output }
        }
    }
}

/// A `Linear` with the detours of the adapters held beside it.
final class NFKMLXAdaptedLinear: Linear {
    let detourList: NFKMLXDetourList

    init(base: Linear) {
        detourList = NFKMLXDetourList(base: base)
        super.init(weight: base.weight, bias: base.bias)
        train(base.training)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = (detourList.base as? Linear)?.callAsFunction(x) ?? super.callAsFunction(x)
        return detourList.detours.reduce(output) { output, detour in
            detour.linear(x).map { output + $0 } ?? output
        }
    }
}

/// A `Conv2d` with the detours of the adapters held beside it.
final class NFKMLXAdaptedConv2d: Conv2d {
    let detourList: NFKMLXDetourList

    init(base: Conv2d) {
        detourList = NFKMLXDetourList(base: base)
        let (outputs, height, width, inputs) = (base.weight.dim(0), base.weight.dim(1), base.weight.dim(2),
                                                base.weight.dim(3))
        super.init(inputChannels: inputs * base.groups, outputChannels: outputs, kernelSize: .init((height, width)),
                   stride: .init(base.stride), padding: .init(base.padding), dilation: .init(base.dilation),
                   groups: base.groups, bias: base.bias != nil)
        // The initializer drew placeholder weights, which take the base layer's before anything evaluates them.
        var parameters = [("weight", base.weight)]
        if let bias = base.bias {
            parameters.append(("bias", bias))
        }
        update(parameters: ModuleParameters.unflattened(parameters))
        train(base.training)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = (detourList.base as? Conv2d)?.callAsFunction(x) ?? super.callAsFunction(x)
        return detourList.detours.reduce(output) { output, detour in
            detour.convolution(x, of: self).map { output + $0 } ?? output
        }
    }
}

/// Places detours beside the layers of a model.
enum NFKMLXAdapterLayers {

    /// Adds each layer's detours beside it, replacing a plain `Linear` or `Conv2d` with its adapted
    /// form the first time an adapter reaches it.
    ///
    /// - Parameter detours: the detours by layer path, as `leafModules().flattened()` names the layer.
    /// - Throws: `NFKMLXError.loRANotApplicable` when a path names no linear or convolution layer of the
    ///   model, or the layer cannot be replaced.
    static func attach(_ detours: [String: [NFKMLXLowRankDetour]], to model: Module) throws {
        guard !detours.isEmpty else { return }
        let layers = Dictionary(model.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        var replacements = [(String, Module)]()
        for (path, added) in detours.sorted(by: { $0.key < $1.key }) {
            switch layers[path] {
            case let adapted as NFKMLXAdaptedLinear:
                adapted.detourList.detours += added
            case let adapted as NFKMLXAdaptedConv2d:
                adapted.detourList.detours += added
            case let linear as Linear where !(linear is QuantizedLinear) && !(linear is NFKMLXLoRAAdapter):
                let adapted = NFKMLXAdaptedLinear(base: linear)
                adapted.detourList.detours = added
                replacements.append((path, adapted))
            case let convolution as Conv2d:
                let adapted = NFKMLXAdaptedConv2d(base: convolution)
                adapted.detourList.detours = added
                replacements.append((path, adapted))
            default:
                throw NFKMLXError.loRANotApplicable("\(path) is not a linear or convolution layer an adapter can reach")
            }
        }
        do {
            try NFKMLXModuleReplacement.place(replacements, in: model)
        } catch {
            throw NFKMLXError.loRANotApplicable("an adapted layer could not be put in place: \(error)")
        }
    }

    /// The bytes every detour of `model` holds.
    static func detourByteCount(of model: Module) -> Int {
        model.leafModules().flattened().reduce(0) { total, entry in
            switch entry.1 {
            case let adapted as NFKMLXAdaptedLinear:
                return total + adapted.detourList.detours.reduce(0) { $0 + $1.byteCount }
            case let adapted as NFKMLXAdaptedConv2d:
                return total + adapted.detourList.detours.reduce(0) { $0 + $1.byteCount }
            default:
                return total
            }
        }
    }
}
