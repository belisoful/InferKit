//
//  NFKMLXTrellisSparse.swift
//  InferKitMLX
//
//  TRELLIS's sparse half: one batch of voxels as features `[N, C]` over integer coordinates, the
//  submanifold convolution, the structured-latent flow, and the swin-attention decoders to the mesh
//  features and the Gaussians.
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// The occupied voxels of one sparse tensor, in a fixed order, with the neighbor tables and window
/// partitions the layers read cached per grid.
final class NFKTrellisVoxelGrid {
    /// Row-major `(x, y, z)` triples.
    let coordinates: [Int32]
    let resolution: Int
    private var neighborTable: MLXArray?
    private var partitions = [Int: NFKTrellisWindowPartition]()

    init(coordinates: [Int32], resolution: Int) {
        precondition(coordinates.count % 3 == 0)
        self.coordinates = coordinates
        self.resolution = resolution
    }

    convenience init(_ coordinates: MLXArray, resolution: Int) {
        self.init(coordinates: coordinates.asType(.int32).reshaped([-1]).asArray(Int32.self), resolution: resolution)
    }

    var count: Int { coordinates.count / 3 }

    var positions: MLXArray { MLXArray(coordinates).reshaped([count, 3]) }

    static func key(_ x: Int32, _ y: Int32, _ z: Int32) -> Int64 {
        (Int64(x) << 42) | (Int64(y) << 21) | Int64(z)
    }

    /// For each voxel and each of the 27 offsets (x-major, −1...1 per axis), the index of the voxel at the
    /// voxel's coordinate plus the offset, or `count` where none is occupied. `[N, 27]`.
    var neighbors: MLXArray {
        if let neighborTable {
            return neighborTable
        }
        var index = [Int64: Int32](minimumCapacity: count)
        for voxel in 0 ..< count {
            index[Self.key(coordinates[3 * voxel], coordinates[3 * voxel + 1], coordinates[3 * voxel + 2])] = Int32(voxel)
        }
        let missing = Int32(count)
        var table = [Int32](repeating: missing, count: count * 27)
        for voxel in 0 ..< count {
            let (x, y, z) = (coordinates[3 * voxel], coordinates[3 * voxel + 1], coordinates[3 * voxel + 2])
            var slot = voxel * 27
            for dx in Int32(-1) ... 1 {
                for dy in Int32(-1) ... 1 {
                    for dz in Int32(-1) ... 1 {
                        if let neighbor = index[Self.key(x + dx, y + dy, z + dz)] {
                            table[slot] = neighbor
                        }
                        slot += 1
                    }
                }
            }
        }
        let built = MLXArray(table).reshaped([count, 27])
        neighborTable = built
        return built
    }

    /// The grid at half the resolution (each coordinate halved, duplicates merged, in x-major order) and
    /// each voxel's index in it.
    func halved() -> (grid: NFKTrellisVoxelGrid, parent: [Int32]) {
        var keys = [Int64](repeating: 0, count: count)
        for voxel in 0 ..< count {
            keys[voxel] = Self.key(coordinates[3 * voxel] / 2, coordinates[3 * voxel + 1] / 2, coordinates[3 * voxel + 2] / 2)
        }
        let unique = Array(Set(keys)).sorted()
        var slot = [Int64: Int32](minimumCapacity: unique.count)
        var halved = [Int32]()
        halved.reserveCapacity(unique.count * 3)
        for (index, key) in unique.enumerated() {
            slot[key] = Int32(index)
            halved += [Int32(key >> 42), Int32((key >> 21) & 0x1FFFFF), Int32(key & 0x1FFFFF)]
        }
        return (NFKTrellisVoxelGrid(coordinates: halved, resolution: resolution / 2), keys.map { slot[$0]! })
    }

    /// Each voxel split into its eight children at twice the resolution, voxel by voxel, children in
    /// x-major order.
    func subdivided() -> NFKTrellisVoxelGrid {
        var children = [Int32]()
        children.reserveCapacity(coordinates.count * 8)
        for voxel in 0 ..< count {
            let (x, y, z) = (coordinates[3 * voxel] * 2, coordinates[3 * voxel + 1] * 2, coordinates[3 * voxel + 2] * 2)
            for dx in Int32(0) ... 1 {
                for dy in Int32(0) ... 1 {
                    for dz in Int32(0) ... 1 {
                        children += [x + dx, y + dy, z + dz]
                    }
                }
            }
        }
        return NFKTrellisVoxelGrid(coordinates: children, resolution: resolution * 2)
    }

    /// The voxels grouped by the `size`³ window they fall in after shifting every coordinate by `shift`.
    func partition(size: Int, shift: Int) -> NFKTrellisWindowPartition {
        let cacheKey = size * 1000 + shift
        if let cached = partitions[cacheKey] {
            return cached
        }
        let built = NFKTrellisWindowPartition(grid: self, size: size, shift: shift)
        partitions[cacheKey] = built
        return built
    }
}

/// Windows as a padded batch: `gather` `[W, L]` reads each window's voxels (padding reads index N),
/// `mask` `[W, 1, 1, L]` is −∞ on padding keys, and `scatter` `[N]` is each voxel's row in the flattened
/// `[W · L]` batch.
struct NFKTrellisWindowPartition {
    let gather: MLXArray
    let mask: MLXArray
    let scatter: MLXArray
    let windows: Int
    let length: Int

    init(grid: NFKTrellisVoxelGrid, size: Int, shift: Int) {
        let count = grid.count
        var maxima = [Int32](repeating: 0, count: 3)
        for voxel in 0 ..< count {
            for axis in 0 ..< 3 {
                maxima[axis] = max(maxima[axis], grid.coordinates[3 * voxel + axis] + Int32(shift))
            }
        }
        let windowCounts = maxima.map { (Int($0) + 1 + size - 1) / size }
        var members = [Int: [Int32]]()
        for voxel in 0 ..< count {
            let w = (0 ..< 3).map { (Int(grid.coordinates[3 * voxel + $0]) + shift) / size }
            members[(w[0] * windowCounts[1] + w[1]) * windowCounts[2] + w[2], default: []].append(Int32(voxel))
        }
        let ordered = members.keys.sorted().map { members[$0]! }
        windows = ordered.count
        length = ordered.map(\.count).max() ?? 0
        var gather = [Int32](repeating: Int32(count), count: windows * length)
        var mask = [Float](repeating: -Float.infinity, count: windows * length)
        var scatter = [Int32](repeating: 0, count: count)
        for (window, voxels) in ordered.enumerated() {
            for (slot, voxel) in voxels.enumerated() {
                gather[window * length + slot] = voxel
                mask[window * length + slot] = 0
                scatter[Int(voxel)] = Int32(window * length + slot)
            }
        }
        self.gather = MLXArray(gather).reshaped([windows, length])
        self.mask = MLXArray(mask).reshaped([windows, 1, 1, length])
        self.scatter = MLXArray(scatter)
    }
}

// MARK: - Layers

/// Submanifold convolution: each output voxel sums the weights times the features of the occupied voxels
/// at its own coordinate plus each kernel offset; outputs sit on the input's voxels. The weight is
/// `[out, k, k, k, in]` with k = 1 or 3, kernel offsets x-major.
final class NFKTrellisSparseConvolution: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let kernel: Int

    init(inputChannels: Int, outputChannels: Int, kernel: Int) {
        self.kernel = kernel
        let scale = 1 / Float(inputChannels * kernel * kernel * kernel).squareRoot()
        _weight.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [outputChannels, kernel, kernel, kernel, inputChannels])
        _bias.wrappedValue = MLXArray.zeros([outputChannels])
    }

    /// The gathered taps go through one matrix product while they fit in this many elements, else one
    /// product per offset.
    static let fusedLimit = 1 << 27

    func callAsFunction(_ x: MLXArray, grid: NFKTrellisVoxelGrid, fusedLimit: Int = NFKTrellisSparseConvolution.fusedLimit) -> MLXArray {
        let outputs = weight.dim(0), inputs = weight.dim(4)
        if kernel == 1 {
            return matmul(x, weight.reshaped([outputs, inputs]).transposed()) + bias
        }
        let table = grid.neighbors
        let padded = concatenated([x, MLXArray.zeros([1, inputs], dtype: x.dtype)], axis: 0)
        let taps = weight.reshaped([outputs, 27, inputs])
        if grid.count * 27 * inputs <= fusedLimit {
            let gathered = take(padded, table.reshaped([-1]), axis: 0).reshaped([grid.count, 27 * inputs])
            return matmul(gathered, taps.reshaped([outputs, 27 * inputs]).transposed()) + bias
        }
        var out = broadcast(bias, to: [grid.count, outputs])
        for offset in 0 ..< 27 {
            let gathered = take(padded, table[0..., offset], axis: 0)
            out = out + matmul(gathered, taps[0..., offset, 0...].transposed())
            eval(out)
        }
        return out
    }
}

/// Holds a sparse convolution under `conv`, as the checkpoint names it.
final class NFKTrellisSparseConvolutionSlot: Module {
    @ModuleInfo(key: "conv") var conv: NFKTrellisSparseConvolution
    init(inputChannels: Int, outputChannels: Int, kernel: Int) {
        _conv.wrappedValue = NFKTrellisSparseConvolution(inputChannels: inputChannels, outputChannels: outputChannels, kernel: kernel)
    }
    func callAsFunction(_ x: MLXArray, grid: NFKTrellisVoxelGrid) -> MLXArray { conv(x, grid: grid) }
}

/// The flow's residual block on voxels: an optional halving or restoring of the grid first, then
/// LayerNorm, SiLU, convolution, a timestep-modulated LayerNorm, SiLU, convolution, plus a projection of
/// the input.
final class NFKTrellisSparseResidualBlock: Module {
    enum Resampling { case none, halve, restore }

    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "conv1") var conv1: NFKTrellisSparseConvolutionSlot
    @ModuleInfo(key: "conv2") var conv2: NFKTrellisSparseConvolutionSlot
    @ModuleInfo(key: "emb_layers") var embedding: [Module]
    @ModuleInfo(key: "skip_connection") var skip: Linear?
    let resampling: Resampling

    init(channels: Int, outputChannels: Int, embeddingWidth: Int, resampling: Resampling) {
        self.resampling = resampling
        _norm1.wrappedValue = LayerNorm(dimensions: channels, eps: 1e-6)
        _conv1.wrappedValue = NFKTrellisSparseConvolutionSlot(inputChannels: channels, outputChannels: outputChannels, kernel: 3)
        _conv2.wrappedValue = NFKTrellisSparseConvolutionSlot(inputChannels: outputChannels, outputChannels: outputChannels, kernel: 3)
        _embedding.wrappedValue = [NFKTrellisSlot(), Linear(embeddingWidth, outputChannels * 2)]
        _skip.wrappedValue = channels == outputChannels ? nil : Linear(channels, outputChannels)
    }

    func callAsFunction(_ x: MLXArray, grid: NFKTrellisVoxelGrid, timestep: MLXArray) -> MLXArray {
        let modulation = split((embedding[1] as! Linear)(silu(timestep)), parts: 2, axis: 1)
        var h = conv1(silu(norm1(x)), grid: grid)
        h = NFKTrellisAttention.normalized(h, eps: 1e-6) * (1 + modulation[0]) + modulation[1]
        h = conv2(silu(h), grid: grid)
        return h + (skip.map { $0(x) } ?? x)
    }
}

/// The structured-latent flow's grid changes: halving averages each parent's children the way torch's
/// `scatter_reduce(..., "mean")` does over a zero-filled tensor (the zero counts as one more element), and
/// restoring copies each parent's features back to its children.
enum NFKTrellisResampling {
    static func halve(_ x: MLXArray, parent: [Int32], parents: Int) -> MLXArray {
        var counts = [Float](repeating: 1, count: parents)
        for index in parent {
            counts[Int(index)] += 1
        }
        let sums = MLXArray.zeros([parents, x.dim(1)], dtype: x.dtype).at[MLXArray(parent)].add(x)
        return sums / expandedDimensions(MLXArray(counts), axis: 1)
    }

    static func restore(_ x: MLXArray, parent: [Int32]) -> MLXArray {
        take(x, MLXArray(parent), axis: 0)
    }
}

// MARK: - Structured-latent flow

/// The flow transformer over the structured latent: residual blocks on the voxels that halve the grid,
/// absolute position embeddings, modulated cross blocks conditioned on the image tokens, and residual
/// blocks that restore the grid with skip connections. Its output is the velocity. Build one with
/// `NFKMLXTrellis.latentFlow(weightsURL:configuration:)`. Introduced in InferKit 0.4.0.
public final class NFKMLXTrellisLatentFlow: Module {
    @ModuleInfo(key: "t_embedder") var timestepEmbedder: NFKTrellisTimestepEmbedder
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "input_blocks") var inputBlocks: [NFKTrellisSparseResidualBlock]
    @ModuleInfo(key: "blocks") var blocks: [NFKTrellisModulatedCrossBlock]
    @ModuleInfo(key: "out_blocks") var outputBlocks: [NFKTrellisSparseResidualBlock]
    @ModuleInfo(key: "out_layer") var outputLayer: Linear

    public let configuration: NFKMLXTrellisConfiguration

    init(_ configuration: NFKMLXTrellisConfiguration) {
        self.configuration = configuration
        let width = configuration.flowWidth, io = configuration.latentFlowIOWidth
        _timestepEmbedder.wrappedValue = NFKTrellisTimestepEmbedder(width: width, frequencyWidth: configuration.frequencyWidth)
        _inputLayer.wrappedValue = Linear(configuration.latentChannels, io)
        let same = configuration.latentFlowIOBlocks - 1
        _inputBlocks.wrappedValue = (0 ..< same).map { _ in
            NFKTrellisSparseResidualBlock(channels: io, outputChannels: io, embeddingWidth: width, resampling: .none)
        } + [NFKTrellisSparseResidualBlock(channels: io, outputChannels: width, embeddingWidth: width, resampling: .halve)]
        _blocks.wrappedValue = (0 ..< configuration.flowBlocks).map { _ in
            NFKTrellisModulatedCrossBlock(width: width, contextWidth: configuration.conditionWidth, heads: configuration.flowHeads,
                                          hidden: configuration.flowHidden, crossHeadNorm: configuration.latentCrossAttentionHeadNorm)
        }
        _outputBlocks.wrappedValue = [
            NFKTrellisSparseResidualBlock(channels: width * 2, outputChannels: io, embeddingWidth: width, resampling: .restore),
        ] + (0 ..< same).map { _ in
            NFKTrellisSparseResidualBlock(channels: io * 2, outputChannels: io, embeddingWidth: width, resampling: .none)
        }
        _outputLayer.wrappedValue = Linear(io, configuration.latentChannels)
    }

    /// The velocity `[N, latentChannels]` at latent features `x` `[N, latentChannels]` on `coordinates`
    /// `[N, 3]` of the latent grid, timestep `t` `[1]` on the 0...1000 scale, and condition tokens
    /// `[1, L, conditionWidth]`. One structure at a time.
    public func callAsFunction(_ x: MLXArray, coordinates: MLXArray, t: MLXArray, condition: MLXArray) -> MLXArray {
        velocity(x, grid: NFKTrellisVoxelGrid(coordinates, resolution: configuration.occupancyResolution), t: t, condition: condition)
    }

    func velocity(_ x: MLXArray, grid: NFKTrellisVoxelGrid, t: MLXArray, condition: MLXArray) -> MLXArray {
        let (coarse, parent) = grid.halved()
        let timestep = timestepEmbedder(t)
        var h = inputLayer(x)
        var skips = [MLXArray]()
        for block in inputBlocks.dropLast() {
            h = block(h, grid: grid, timestep: timestep)
            skips.append(h)
        }
        h = inputBlocks[inputBlocks.count - 1](NFKTrellisResampling.halve(h, parent: parent, parents: coarse.count), grid: coarse,
                                               timestep: timestep)
        let coarseSkip = h
        h = h + NFKTrellisPositions.absolute(coarse.positions, width: configuration.flowWidth)
        var tokens = expandedDimensions(h, axis: 0)
        for block in blocks {
            tokens = block(tokens, timestep: timestep, context: condition)
        }
        h = NFKTrellisResampling.restore(concatenated([tokens[0], coarseSkip], axis: 1), parent: parent)
        h = outputBlocks[0](h, grid: grid, timestep: timestep)
        for (block, skip) in zip(outputBlocks.dropFirst(), skips.reversed()) {
            h = block(concatenated([h, skip], axis: 1), grid: grid, timestep: timestep)
        }
        return outputLayer(NFKTrellisAttention.normalized(h, eps: 1e-5))
    }
}

// MARK: - Decoders

/// Self-attention within `size`³ windows of the voxel grid, the windows shifted by `shift` voxels.
final class NFKTrellisWindowedAttention: Module {
    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    let heads: Int
    let size: Int
    let shift: Int

    init(width: Int, heads: Int, size: Int, shift: Int) {
        self.heads = heads
        self.size = size
        self.shift = shift
        _toQKV.wrappedValue = Linear(width, width * 3)
        _toOut.wrappedValue = Linear(width, width)
    }

    func callAsFunction(_ x: MLXArray, grid: NFKTrellisVoxelGrid) -> MLXArray {
        let width = x.dim(1), headWidth = width / heads
        let partition = grid.partition(size: size, shift: shift)
        let qkv = concatenated([toQKV(x), MLXArray.zeros([1, width * 3], dtype: x.dtype)], axis: 0)
        let windows = take(qkv, partition.gather.reshaped([-1]), axis: 0)
            .reshaped([partition.windows, partition.length, 3, heads, headWidth])
        func part(_ index: Int) -> MLXArray { windows[0..., 0..., index].transposed(0, 2, 1, 3) }
        let attended = MLXFast.scaledDotProductAttention(queries: part(0), keys: part(1), values: part(2),
                                                         scale: 1 / Float(headWidth).squareRoot(), mask: partition.mask)
        let rows = attended.transposed(0, 2, 1, 3).reshaped([partition.windows * partition.length, width])
        return toOut(take(rows, partition.scatter, axis: 0))
    }
}

/// LayerNorm (no affine), windowed self-attention, LayerNorm, feed-forward, each with a residual.
final class NFKTrellisSparseTransformerBlock: Module {
    @ModuleInfo(key: "attn") var attention: NFKTrellisWindowedAttention
    @ModuleInfo(key: "mlp") var feedForward: NFKTrellisFeedForward

    init(width: Int, heads: Int, hidden: Int, window: Int, shift: Int) {
        _attention.wrappedValue = NFKTrellisWindowedAttention(width: width, heads: heads, size: window, shift: shift)
        _feedForward.wrappedValue = NFKTrellisFeedForward(width: width, hidden: hidden)
    }

    func callAsFunction(_ x: MLXArray, grid: NFKTrellisVoxelGrid) -> MLXArray {
        let x = x + attention(NFKTrellisAttention.normalized(x, eps: 1e-6), grid: grid)
        return x + feedForward(NFKTrellisAttention.normalized(x, eps: 1e-6))
    }
}

/// GroupNorm and SiLU, every voxel split into eight, convolution, GroupNorm, SiLU, convolution, plus a
/// 1³ convolution of the split input.
final class NFKTrellisSubdivideBlock: Module {
    @ModuleInfo(key: "act_layers") var activation: [Module]
    @ModuleInfo(key: "out_layers") var outputLayers: [Module]
    @ModuleInfo(key: "skip_connection") var skip: NFKTrellisSparseConvolutionSlot

    init(channels: Int, outputChannels: Int) {
        _activation.wrappedValue = [GroupNorm(groupCount: 32, dimensions: channels, pytorchCompatible: true), NFKTrellisSlot()]
        _outputLayers.wrappedValue = [
            NFKTrellisSparseConvolutionSlot(inputChannels: channels, outputChannels: outputChannels, kernel: 3),
            GroupNorm(groupCount: 32, dimensions: outputChannels, pytorchCompatible: true), NFKTrellisSlot(),
            NFKTrellisSparseConvolutionSlot(inputChannels: outputChannels, outputChannels: outputChannels, kernel: 3),
        ]
        _skip.wrappedValue = NFKTrellisSparseConvolutionSlot(inputChannels: channels, outputChannels: outputChannels, kernel: 1)
    }

    static func groupNorm(_ norm: Module, _ x: MLXArray) -> MLXArray {
        (norm as! GroupNorm)(expandedDimensions(x, axis: 0))[0]
    }

    func callAsFunction(_ x: MLXArray, grid: NFKTrellisVoxelGrid, children: NFKTrellisVoxelGrid) -> MLXArray {
        let split = { (h: MLXArray) in repeated(h, count: 8, axis: 0) }
        var h = split(silu(Self.groupNorm(activation[0], x)))
        h = (outputLayers[0] as! NFKTrellisSparseConvolutionSlot)(h, grid: children)
        h = silu(Self.groupNorm(outputLayers[1], h))
        h = (outputLayers[3] as! NFKTrellisSparseConvolutionSlot)(h, grid: children)
        return h + skip(split(x), grid: children)
    }
}

/// The decoders' swin blocks: windows shift by half a window on every other block.
enum NFKTrellisDecoderTrunk {
    static func blocks(_ configuration: NFKMLXTrellisConfiguration) -> [NFKTrellisSparseTransformerBlock] {
        (0 ..< configuration.decoderBlocks).map { index in
            NFKTrellisSparseTransformerBlock(width: configuration.decoderWidth, heads: configuration.decoderHeads,
                                             hidden: configuration.decoderWidth * 4, window: configuration.decoderWindow,
                                             shift: index % 2 == 0 ? 0 : configuration.decoderWindow / 2)
        }
    }
}

/// The mesh decoder: the trunk, two subdivisions to four times the latent resolution, and the projection
/// to the FlexiCubes features (SDF, deformation, weights, and colors per voxel). Build one with
/// `NFKMLXTrellis.meshDecoder(weightsURL:configuration:)`. Introduced in InferKit 0.4.0.
public final class NFKMLXTrellisMeshDecoder: Module {
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "blocks") var blocks: [NFKTrellisSparseTransformerBlock]
    @ModuleInfo(key: "upsample") var upsample: [NFKTrellisSubdivideBlock]
    @ModuleInfo(key: "out_layer") var outputLayer: Linear

    public let configuration: NFKMLXTrellisConfiguration

    init(_ configuration: NFKMLXTrellisConfiguration) {
        self.configuration = configuration
        let width = configuration.decoderWidth
        _inputLayer.wrappedValue = Linear(configuration.latentChannels, width)
        _blocks.wrappedValue = NFKTrellisDecoderTrunk.blocks(configuration)
        _upsample.wrappedValue = [NFKTrellisSubdivideBlock(channels: width, outputChannels: width / 4),
                                  NFKTrellisSubdivideBlock(channels: width / 4, outputChannels: width / 8)]
        _outputLayer.wrappedValue = Linear(width / 8, configuration.meshFeatureChannels)
    }

    /// The mesh features `[64N, meshFeatureChannels]` and their voxels at four times the resolution for
    /// structured-latent features `[N, latentChannels]` on `grid`.
    func features(_ latent: MLXArray, grid: NFKTrellisVoxelGrid) -> (features: MLXArray, grid: NFKTrellisVoxelGrid) {
        var h = inputLayer(latent) + NFKTrellisPositions.absolute(grid.positions, width: configuration.decoderWidth)
        for block in blocks {
            h = block(h, grid: grid)
        }
        var current = grid
        for block in upsample {
            let children = current.subdivided()
            h = block(h, grid: current, children: children)
            eval(h)
            current = children
        }
        return (outputLayer(h), current)
    }
}

/// The Gaussians of one structure: `count` = 32 per voxel, each field as the reference stores it and as
/// its activation reads it.
struct NFKTrellisGaussians {
    /// Stored fields: positions in 0...1 `[M, 3]`, DC colors `[M, 1, 3]`, scales, rotations, opacities.
    let storedPositions: MLXArray
    let colors: MLXArray
    let storedScales: MLXArray
    let storedRotations: MLXArray
    let storedOpacities: MLXArray
    let minimumScale: Float
    let scaleBias: Float
    let opacityBias: Float

    /// Centers in −0.5...0.5.
    var positions: MLXArray { storedPositions - 0.5 }
    /// softplus(stored + bias), widened by the minimum kernel size in quadrature.
    var scales: MLXArray {
        let s = softplus(storedScales + scaleBias)
        return sqrt(s * s + minimumScale * minimumScale)
    }
    /// Unit quaternions (w, x, y, z) of stored + (1, 0, 0, 0).
    var rotations: MLXArray {
        let q = storedRotations + MLXArray([Float(1), 0, 0, 0])
        return q / maximum(sqrt(sum(q * q, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
    }
    var opacities: MLXArray { sigmoid(storedOpacities + opacityBias) }
}

/// The Gaussian decoder: the trunk, a normalization, and a projection to 32 Gaussians per voxel, each
/// offset from its voxel center within 1.5 voxels. Build one with
/// `NFKMLXTrellis.gaussianDecoder(weightsURL:configuration:)`. Introduced in InferKit 0.4.0.
public final class NFKMLXTrellisGaussianDecoder: Module {
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "blocks") var blocks: [NFKTrellisSparseTransformerBlock]
    @ModuleInfo(key: "out_layer") var outputLayer: Linear
    @ParameterInfo(key: "offset_perturbation") var perturbation: MLXArray

    public let configuration: NFKMLXTrellisConfiguration

    init(_ configuration: NFKMLXTrellisConfiguration) {
        self.configuration = configuration
        let count = configuration.gaussiansPerVoxel
        _inputLayer.wrappedValue = Linear(configuration.latentChannels, configuration.decoderWidth)
        _blocks.wrappedValue = NFKTrellisDecoderTrunk.blocks(configuration)
        _outputLayer.wrappedValue = Linear(configuration.decoderWidth, count * 14)
        _perturbation.wrappedValue = MLXArray.zeros([count, 3])
    }

    /// The output layer's features `[N, 14 · gaussiansPerVoxel]` for structured-latent features on `grid`.
    func features(_ latent: MLXArray, grid: NFKTrellisVoxelGrid) -> MLXArray {
        var h = inputLayer(latent) + NFKTrellisPositions.absolute(grid.positions, width: configuration.decoderWidth)
        for block in blocks {
            h = block(h, grid: grid)
        }
        return outputLayer(NFKTrellisAttention.normalized(h, eps: 1e-5))
    }

    /// The Gaussians the features of `grid` describe. Channels are laid out as positions (3), colors (3),
    /// scales (3), rotations (4), and opacities (1), each for all Gaussians of a voxel in turn; rotations
    /// read at a tenth of their value.
    func gaussians(_ features: MLXArray, grid: NFKTrellisVoxelGrid) -> NFKTrellisGaussians {
        let count = configuration.gaussiansPerVoxel, voxels = features.dim(0)
        let resolution = Float(configuration.occupancyResolution)
        func field(_ start: Int, _ width: Int) -> MLXArray {
            features[0..., (start * count) ..< ((start + width) * count)].reshaped([voxels, count, width])
        }
        let centers = (grid.positions.asType(.float32) + 0.5) / resolution
        let offsets = tanh(field(0, 3) + perturbation) / resolution * 0.5 * configuration.gaussianVoxelSpan
        let positions = (expandedDimensions(centers, axis: 1) + offsets).reshaped([-1, 3])
        let scaleBias = configuration.gaussianScaleBias
        return NFKTrellisGaussians(
            storedPositions: positions,
            colors: field(3, 3).reshaped([-1, 1, 3]),
            storedScales: field(6, 3).reshaped([-1, 3]),
            storedRotations: field(9, 4).reshaped([-1, 4]) * 0.1,
            storedOpacities: field(13, 1).reshaped([-1, 1]),
            minimumScale: configuration.gaussianMinimumScale,
            scaleBias: scaleBias + log(-expm1(-scaleBias)),
            opacityBias: log(configuration.gaussianOpacityBias / (1 - configuration.gaussianOpacityBias)))
    }
}

extension NFKMLXTrellis {
    /// The structured-latent flow with `slat_flow_img_dit_L_64l8p2_fp16.safetensors` at `weightsURL`
    /// loaded, at float32; nil builds random weights.
    public static func latentFlow(weightsURL: URL?, configuration: NFKMLXTrellisConfiguration = .released) throws
        -> NFKMLXTrellisLatentFlow {
        let flow = NFKMLXTrellisLatentFlow(configuration)
        if let weightsURL {
            try load(weightsURL, into: flow, transposingConvolutions: false) { _ in true }
        }
        return flow
    }

    /// The mesh decoder with `slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors` at `weightsURL` loaded, at
    /// float32; nil builds random weights.
    public static func meshDecoder(weightsURL: URL?, configuration: NFKMLXTrellisConfiguration = .released) throws
        -> NFKMLXTrellisMeshDecoder {
        let decoder = NFKMLXTrellisMeshDecoder(configuration)
        if let weightsURL {
            try load(weightsURL, into: decoder, transposingConvolutions: false) { _ in true }
        }
        return decoder
    }

    /// The Gaussian decoder with `slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors` at `weightsURL` loaded, at
    /// float32; nil builds random weights.
    public static func gaussianDecoder(weightsURL: URL?, configuration: NFKMLXTrellisConfiguration = .released) throws
        -> NFKMLXTrellisGaussianDecoder {
        let decoder = NFKMLXTrellisGaussianDecoder(configuration)
        if let weightsURL {
            try load(weightsURL, into: decoder, transposingConvolutions: false) { _ in true }
        }
        return decoder
    }
}
