//
//  NFKMLXReadQuantization.swift
//  InferKitMLX
//
//  A weight quantization applied while a release is read: each eligible weight is packed as its group of
//  tensors is read and converted, so a load quantized to fit holds the packed weights and one group,
//  never the release at full width.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// The layers a load packs as it reads, and the packing.
///
/// @discussion ``prepare(_:bits:groupSize:)`` replaces the module's eligible `Linear` layers with
/// `QuantizedLinear` before any weight is read. The replaced layers' initial values stay unevaluated, so
/// the replacement holds no memory. ``expand(_:_:moduleKey:)`` then packs each eligible weight with the
/// call `QuantizedLinear` makes on a loaded `Linear`, at the element type the load converted it to, so the
/// packed words, scales, and biases equal those of a load followed by ``NFKMLXQuantization/quantize(module:bits:groupSize:includeEmbeddings:)``.
/// Embeddings and a mixture's stacked experts are left to that later quantization.
struct NFKMLXReadQuantization {
    let bits: Int
    let groupSize: Int
    /// The module paths of the layers packed as they are read.
    let packedLayers: Set<String>
    /// The weight elements those layers hold.
    let elementCount: Int

    /// Gives `module` the quantized structure of its eligible `Linear` layers and returns the packing.
    static func prepare(_ module: Module, bits: Int, groupSize: Int) throws -> NFKMLXReadQuantization {
        var replacements = [(String, Module)]()
        var elements = 0
        for (path, layer) in module.leafModules().flattened() {
            guard let linear = layer as? Linear, !(linear is QuantizedLinear),
                  linear.weight.shape[1] % groupSize == 0 else {
                continue
            }
            replacements.append((path, QuantizedLinear(linear, groupSize: groupSize, bits: bits, mode: .affine)))
            elements += linear.weight.shape[0] * linear.weight.shape[1]
        }
        try NFKMLXModuleReplacement.place(replacements, in: module)
        return NFKMLXReadQuantization(bits: bits, groupSize: groupSize, packedLayers: Set(replacements.map(\.0)),
                                      elementCount: elements)
    }

    /// The tensors `value` becomes: its packed words, scales, and biases where `name` is the weight of a
    /// packed layer under `moduleKey`, else `value` itself.
    func expand(_ name: String, _ value: MLXArray, moduleKey: (String) -> String = { $0 }) -> [(String, MLXArray)] {
        let suffix = ".weight"
        let key = moduleKey(name)
        guard key.hasSuffix(suffix), name.hasSuffix(suffix), packedLayers.contains(String(key.dropLast(suffix.count))) else {
            return [(name, value)]
        }
        let base = String(name.dropLast(suffix.count))
        let (words, scales, biases) = quantized(value, groupSize: groupSize, bits: bits, mode: .affine)
        return [(name, words), (base + ".scales", scales)] + (biases.map { [(base + ".biases", $0)] } ?? [])
    }

    /// The bytes the packed layers' weights take at `precision` unpacked: 16-bit as a release stores them,
    /// or float32.
    func unpackedBytes(at precision: NFKMLXWeightPrecision) -> Int {
        elementCount * (precision == .float32 ? 4 : 2)
    }

    /// The bytes the packed layers' weights take packed: `bits` per element, and a scale and a bias per
    /// group at the element type the load converts to.
    func packedBytes(at precision: NFKMLXWeightPrecision) -> Int {
        let groups = elementCount / groupSize
        return elementCount * bits / 8 + groups * 2 * (precision == .float32 ? 4 : 2)
    }
}
