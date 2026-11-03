//
//  NFKMLXWanBlockedLoad.swift
//  InferKitMLX
//
//  Loading a 14B Wan transformer block by block: the blocks held at the release's precision, left in the
//  release and streamed in their turn, or packed to a few bits as each block is read so the stored form
//  is never held whole. The Animate transformer and the text- and image-to-video transformers share it.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

enum NFKMLXWanBlockedLoad {

    /// The block a transformer key belongs to and its name inside the block.
    static func block(ofKey key: String) -> (layer: Int, name: String)? {
        NFKMLXLayerStream.layer(ofDecoderName: key, layersKey: "blocks")
    }

    /// A transformer tensor in the module's layout: the patch convolution moves from `[out, in, t, h, w]`
    /// to `[out, t, h, w, in]`.
    static func adapted(key: String, value: MLXArray) -> (String, MLXArray) {
        (key, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value)
    }

    /// What the transformer holds at `precision`, each block counted as a repeated layer.
    static func footprint(directory: URL, precision: NFKMLXWeightPrecision) throws -> NFKMLXStageFootprint {
        try NFKMLXLayerStream.footprint(directory: directory, precision: precision, includes: { _ in true },
                                        layerOf: { block(ofKey: $0)?.layer })
    }

    /// What the transformer holds with every block's linear layers packed to `bits` at `groupSize`: each
    /// packed matrix as its words plus a scale and a bias per group, everything else at `precision`.
    static func quantizedFootprint(directory: URL, precision: NFKMLXWeightPrecision, bits: Int,
                                   groupSize: Int) throws -> NFKMLXStageFootprint {
        var layers = [Int: Int](), fixed = 0
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            for (key, entry) in try NFKMLXSafetensors.entries(inFile: url) {
                let element = precision == .float32 ? 4 : (NFKMLXSafetensors.dtype(entry.dtype)?.size ?? 2)
                let count = entry.shape.reduce(1, *)
                let packs = block(ofKey: key) != nil && key.hasSuffix(".weight") && entry.shape.count == 2
                    && entry.shape[1] % groupSize == 0
                let held = packs ? count * bits / 8 + 2 * (count / groupSize) * element : count * element
                if let layer = block(ofKey: key)?.layer {
                    layers[layer, default: 0] += held
                } else {
                    fixed += held
                }
            }
        }
        let layerBytes = (0 ..< (layers.keys.max().map { $0 + 1 } ?? 0)).map { layers[$0] ?? 0 }
        return NFKMLXStageFootprint(bytes: fixed + layerBytes.reduce(0, +), layerBytes: layerBytes)
    }

    /// Loads a `transformer/` directory into `net`, whose `blocks` are its repeated layers, and returns the
    /// stream of the blocks left in the release (nil where every block is held).
    ///
    /// - Parameters:
    ///   - streamed: the blocks left in the release and read in their turn on every pass.
    ///   - bits: packs every block's linear layers to this many bits as each block loads, so the stored
    ///     form is never held whole; nil keeps them at `precision`. A packed transformer streams nothing.
    ///   - adapters: released LoRA deltas (``NFKMLXWanLoRA``) merged in order, each block's before it packs.
    ///     A streamed transformer takes none: its blocks are read from the release on every pass.
    static func load(into net: Module, blocks: [Module], fromDirectory directory: URL,
                     precision: NFKMLXWeightPrecision, streamed: Set<Int> = [], bits: Int? = nil,
                     groupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize,
                     adapters: [[String: NFKMLXWanLoRA.Delta]] = []) throws -> NFKMLXLayerStream? {
        guard adapters.isEmpty || streamed.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration(
                "a streamed transformer reads \(streamed.count) blocks from the release on every pass and takes no "
                + "adapter; load it with a residency that holds the transformer")
        }
        guard let bits else {
            let vacated = try NFKMLXLayerStream.vacating(streamed, of: blocks, prefix: "blocks.",
                                                         directory: directory, precision: precision,
                                                         layerOf: block(ofKey:))
            let arrays = try NFKMLXReleaseWeights.materializedArrays(
                inDirectory: directory, precision: precision,
                remap: { key in block(ofKey: key).map { streamed.contains($0.layer) } == true ? nil : key },
                transform: adapted)
            try NFKMLXWeights.apply(arrays + (vacated?.placeholders ?? []), to: net, verifyShapes: true)
            for deltas in adapters {
                try NFKMLXWanLoRA.apply(deltas, to: net)
            }
            vacated?.stream.prime()
            return vacated?.stream
        }
        guard streamed.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration("a packed transformer holds every block; it does not stream")
        }
        guard (2 ... 8).contains(bits), [32, 64, 128].contains(groupSize) else {
            throw NFKMLXError.unsupportedConfiguration(
                "a packed load takes 2 to 8 bits at a group size of 32, 64, or 128, not \(bits) at \(groupSize)")
        }
        let expected = Set(net.parameters().flattened().map(\.0))
        var outside = [(String, MLXArray)]()
        var unread = Set(expected.filter { block(ofKey: $0) != nil })
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            var read = [Int: [(String, MLXArray)]]()
            for (key, value) in try NFKMLXWeights.loadCheckpoint(url: url).arrays where expected.contains(key) {
                let typed = precision == .float32 && value.dtype.isFloatingPoint ? value.asType(.float32) : value
                if let (layer, name) = block(ofKey: key) {
                    read[layer, default: []].append((name, typed))
                    unread.remove(key)
                } else {
                    outside.append(adapted(key: key, value: typed))
                }
            }
            // A block whose tensors span two shards packs once its last shard is read.
            for (layer, arrays) in read.sorted(by: { $0.key < $1.key }) {
                let module = blocks[layer]
                module.update(parameters: ModuleParameters.unflattened(arrays))
                let complete = !unread.contains { block(ofKey: $0)?.layer == layer }
                if complete {
                    for deltas in adapters {
                        try NFKMLXWanLoRA.apply(deltas, to: module, pathPrefix: "blocks.\(layer).")
                    }
                    try NFKMLXQuantization.quantize(module: module, bits: bits, groupSize: groupSize)
                    eval(module)
                    NFKMLXGPU.clearCache()
                }
            }
        }
        let supplied = Set(outside.map(\.0))
        let missing = Array(unread) + expected.filter { block(ofKey: $0) == nil && !supplied.contains($0) }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch("the release lacks \(missing.count) parameters, such as \(missing.sorted().prefix(3))")
        }
        // The packed blocks name their parameters apart from the stored ones, so the rest applies on its
        // own after the coverage check above.
        try NFKMLXWeights.apply(outside, to: net, strict: false, verifyShapes: true)
        for deltas in adapters {
            try NFKMLXWanLoRA.apply(deltas.filter { !$0.key.hasPrefix("blocks.") }, to: net)
        }
        eval(net)
        NFKMLXGPU.clearCache()
        return nil
    }
}
