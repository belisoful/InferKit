//
//  NFKMLXGemma3nMatFormer.swift
//  InferKitMLX
//
//  MatFormer: Gemma 3n's smaller size is NESTED inside its larger one, so a release can be sliced to a
//  size between them without retraining. This is a checkpoint operation — it rewrites weights and
//  never runs a forward pass — and it is the one part of the family that has no reference
//  implementation anywhere: `transformers` does not implement it.
//
//  **The mechanism was derived from the two released checkpoints rather than read from a
//  specification, and every rule below is byte-verified against them.** Slicing E4B at E2B's geometry
//  reproduces the released E2B EXACTLY, tensor for tensor, which is what
//  `testGemma3nMatFormerExtractsTheReleasedE2B` asserts. The released E2B is the byte oracle here, the
//  same standing the offline converters have for the native checkpoint reader.
//
//  Four rules, two of which a naive reading gets wrong:
//
//  - **Layers are SELECTED, not truncated.** E2B keeps E4B's layers 0–19 and 25–34 and drops 20–24 —
//    the first five of E4B's shared-key-value region, which are the layers that compute no keys or
//    values and so are the cheapest to lose.
//  - **The feed-forward is a PREFIX slice**: the first `intermediateSize` rows of `gate_proj` and
//    `up_proj`, and the matching columns of `down_proj`.
//  - **The per-layer embedding is sliced by COLUMN BLOCK**, one 256-wide block per kept layer, NOT by
//    taking the leading columns. A prefix loads cleanly and gives every layer the wrong slice.
//  - **The per-layer projection is sliced by ROW BLOCK**, the same way and with the same trap.
//
//  Everything else — attention, the normalizations, AltUp, LAuReL, the token embedding — is carried
//  across untouched, and is byte-identical between the two releases.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

/// Which of a larger Gemma 3n's layers to keep, and how wide each kept layer's feed-forward is.
public struct NFKMLXGemma3nSlice: Sendable {
    /// The base model's layer indices to keep, in increasing order.
    public var keptLayers: [Int]
    /// The feed-forward width of each kept layer, in the same order.
    public var intermediateSizes: [Int]

    public init(keptLayers: [Int], intermediateSizes: [Int]) {
        self.keptLayers = keptLayers
        self.intermediateSizes = intermediateSizes
    }

    /// One width for every kept layer.
    public init(keptLayers: [Int], intermediateSize: Int) {
        self.init(keptLayers: keptLayers,
                  intermediateSizes: Array(repeating: intermediateSize, count: keptLayers.count))
    }

    /// The released E2B, as a slice of the released E4B.
    ///
    /// @discussion Derived from the two checkpoints and verified byte for byte, not taken from a
    /// specification: the kept layers are E4B's 0–19 and 25–34, and the feed-forward is the leading
    /// half of E4B's 16384.
    public static let e2bFromE4B = NFKMLXGemma3nSlice(
        keptLayers: Array(0 ..< 20) + Array(25 ..< 35), intermediateSize: 8192)

    /// A slice that keeps every layer and narrows the feed-forward, which is the finer of MatFormer's
    /// two knobs and the one a size between the releases is usually reached with.
    public static func width(_ intermediateSize: Int, layers: Int) -> NFKMLXGemma3nSlice {
        NFKMLXGemma3nSlice(keptLayers: Array(0 ..< layers), intermediateSize: intermediateSize)
    }
}

extension NFKMLXGemma3n {

    /// The configuration a slice of `base` produces.
    ///
    /// @discussion `num_kv_shared_layers` is RECOMPUTED rather than carried over: it is the number of
    /// kept layers that lay in the base's shared region, which for E2B out of E4B is 10 of E4B's 15.
    /// The per-layer lists — the attention kinds, the activation sparsity — are subset in the same
    /// order.
    public static func configuration(slicing base: NFKMLXGemma3nConfiguration,
                                     by slice: NFKMLXGemma3nSlice) throws -> NFKMLXGemma3nConfiguration {
        try validate(slice, against: base)
        var sliced = base
        sliced.layerCount = slice.keptLayers.count
        sliced.intermediateSizes = slice.intermediateSizes
        sliced.layerTypes = slice.keptLayers.map { base.layerTypes[$0] }
        sliced.activationSparsity = slice.keptLayers.map { base.activationSparsity[$0] }
        sliced.sharedKeyValueLayers = slice.keptLayers.filter { $0 >= base.firstSharedKeyValueLayer }.count
        return sliced
    }

    /// Refuses a slice that would change what the model computes in a way the weights cannot express.
    ///
    /// @discussion The load-bearing constraint is that a dropped layer must come from the base's
    /// SHARED region. A layer outside it may be some later layer's key-value donor, and dropping it
    /// silently re-points that layer at a different donor — a model that loads and is wrong.
    static func validate(_ slice: NFKMLXGemma3nSlice, against base: NFKMLXGemma3nConfiguration) throws {
        guard slice.keptLayers.count == slice.intermediateSizes.count else {
            throw NFKMLXError.unsupportedConfiguration(
                "the slice names \(slice.keptLayers.count) layers and \(slice.intermediateSizes.count) widths")
        }
        guard !slice.keptLayers.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration("a slice keeps no layers")
        }
        guard zip(slice.keptLayers, slice.keptLayers.dropFirst()).allSatisfy({ $0 < $1 }) else {
            throw NFKMLXError.unsupportedConfiguration("a slice's kept layers must increase")
        }
        guard let last = slice.keptLayers.last, last < base.layerCount, slice.keptLayers[0] >= 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "a slice names a layer outside the base's \(base.layerCount)")
        }
        for width in slice.intermediateSizes where width <= 0 {
            throw NFKMLXError.unsupportedConfiguration("a slice's feed-forward width must be positive")
        }
        for (position, layer) in slice.keptLayers.enumerated()
        where slice.intermediateSizes[position] > base.intermediateSizes[layer] {
            throw NFKMLXError.unsupportedConfiguration(
                "layer \(layer) is \(base.intermediateSizes[layer]) wide, so it cannot be sliced to "
                    + "\(slice.intermediateSizes[position])")
        }
        let dropped = Set(0 ..< base.layerCount).subtracting(slice.keptLayers)
        for layer in dropped.sorted() where layer < base.firstSharedKeyValueLayer {
            throw NFKMLXError.unsupportedConfiguration(
                "layer \(layer) computes its own keys and values, so dropping it would re-point the "
                    + "layers that share them; a slice may drop only from layer "
                    + "\(base.firstSharedKeyValueLayer) up")
        }
    }

    /// One decoder tensor sliced, or nil where the slice drops it.
    ///
    /// - Parameter name: the key with the release's prefixes already removed.
    static func sliced(_ name: String, _ value: MLXArray, by slice: NFKMLXGemma3nSlice,
                       base: NFKMLXGemma3nConfiguration) -> (String, MLXArray)? {
        let perLayer = base.perLayerInputSize

        // The per-layer tables are ONE tensor holding a block per layer, so a dropped layer takes its
        // block out of the middle. Concatenating the kept blocks is what the release does; the leading
        // blocks are a different, wrong model.
        if name == "embed_tokens_per_layer.weight" {
            let blocks = slice.keptLayers.map { value[0..., ($0 * perLayer) ..< (($0 + 1) * perLayer)] }
            return (name, concatenated(blocks, axis: 1))
        }
        if name == "per_layer_model_projection.weight" {
            let blocks = slice.keptLayers.map { value[($0 * perLayer) ..< (($0 + 1) * perLayer), 0...] }
            return (name, concatenated(blocks, axis: 0))
        }

        guard let (layer, rest) = layerIndex(of: name) else { return (name, value) }
        guard let position = slice.keptLayers.firstIndex(of: layer) else { return nil }
        let renamed = "layers.\(position)." + rest
        let width = slice.intermediateSizes[position]

        // The feed-forward narrows from the front: rows of the two projections into it, columns of the
        // one out of it.
        if rest == "mlp.gate_proj.weight" || rest == "mlp.up_proj.weight" {
            return (renamed, value[0 ..< width, 0...])
        }
        if rest == "mlp.down_proj.weight" {
            return (renamed, value[0..., 0 ..< width])
        }
        return (renamed, value)
    }

    /// The layer index a `layers.N.…` key names, and the rest of the key.
    private static func layerIndex(of name: String) -> (Int, String)? {
        let parts = name.split(separator: ".", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count == 3, parts[0] == "layers", let index = Int(parts[1]) else { return nil }
        return (index, String(parts[2]))
    }

    /// The decoder tensors of a release, sliced to a smaller size.
    static func slicedDecoderWeights(inDirectory directory: URL, slice: NFKMLXGemma3nSlice,
                                     base: NFKMLXGemma3nConfiguration,
                                     precision: NFKMLXWeightPrecision) throws -> [(String, MLXArray)] {
        try validate(slice, against: base)
        let full = try NFKMLXReleaseWeights.arrays(
            inDirectory: directory, precision: precision,
            remap: { NFKMLXGemma3nLanguage.decoderName(of: $0, configuration: base) })
        return full.compactMap { Self.sliced($0.0, $0.1, by: slice, base: base) }
    }

    /// A Gemma 3n decoder built by slicing a LARGER release, without a checkpoint of its own.
    ///
    /// @discussion This is the capability MatFormer exists for: one download serves every size between
    /// the two releases. `NFKMLXGemma3nSlice.e2bFromE4B` reproduces the released E2B exactly; any other
    /// slice is a size no checkpoint was ever published for, so **it carries no reference and its
    /// quality is not measured here** — it is a documented slice of a model that is at parity, which is
    /// a weaker claim than parity and is meant as one.
    public static func decoder(slicing directory: URL, by slice: NFKMLXGemma3nSlice,
                               precision: NFKMLXWeightPrecision = .float32) throws
        -> (net: NFKMLXGemma3nNet, configuration: NFKMLXGemma3nConfiguration) {
        let base = try NFKMLXGemma3nLanguage.configuration(
            fromHuggingFace: directory.appendingPathComponent("config.json"))
        let configuration = try configuration(slicing: base, by: slice)
        let net = NFKMLXGemma3nNet(configuration)
        let weights = try slicedDecoderWeights(inDirectory: directory, slice: slice, base: base,
                                               precision: precision)
        try NFKMLXWeights.apply(weights, to: net)
        return (net, configuration)
    }
}
