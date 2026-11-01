//
//  NFKMLXLTX2DiffusionDecoder.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

// The LTX-2.5 diffusion video decoder (`LTX2VideoDiffusionDecoderModel`, Lightricks). It reads the
// latents the convolutional autoencoder writes and is itself a small diffusion model. Four deterministic
// stages of neighborhood-attention blocks, each followed by a linear channel expansion and a
// depth-to-space shuffle, turn the latent into a context volume; a fifth stage denoises patchified pixels
// against that context through a shared adaptive-norm modulation of the timestep. The release predicts
// x0 in one step, so stage 5 runs once on pure noise and its prediction is the clip.
//
// Every attention is a 3-D neighborhood attention: a query attends to the `kernel` window centered on it
// and shifted inward at the grid's border so that it always holds exactly `kernel` positions (NATTEN's
// `na3d`). The query and key take a per-head RMS norm, the query the `1 / sqrt(headDim)` factor, and both
// an absolute 3-D rotary over the tensor's own indices that rotates adjacent channel pairs.
//
// Tensors flow channels-last, `[batch, frames, height, width, channels]`. The latent's last frame is
// replicated `2 ⌊kT / 2⌋` times before stage 1 so that the inward-shifted window at the trailing border
// sees replicated frames, and the frames those produce are cropped before stage 5.

/// The geometry of an LTX-2 diffusion decoder, read from a release's `diffusion_decoder/config.json`.
public struct NFKMLXLTX2DiffusionDecoderConfiguration: Sendable, Equatable {
    public var outChannels: Int
    public var latentChannels: Int
    public var patchSize: Int
    public var headDim: Int
    /// The widths of the four deterministic stages and then of stage 5.
    public var stageChannels: [Int]
    /// The blocks of the four deterministic stages and then of stage 5.
    public var stageDepths: [Int]
    /// Each deterministic stage's (time, height, width) window.
    public var stageKernels: [[Int]]
    /// Each deterministic stage's upsampling (time, height, width) stride.
    public var upsampleStrides: [[Int]]
    /// The factor each upsampler divides the width by.
    public var upsampleReductions: [Int]
    /// Stage 5's (time, height, width) window.
    public var diffusionKernel: [Int]
    public var timeEmbeddingDim: Int
    /// The factor the stage-5 timestep is scaled by before the sinusoid.
    public var timestepScale: Float
    /// Whether stage 5 predicts x0 (`x0`) rather than a velocity (`v`).
    public var predictsSample: Bool
    /// The stage-5 steps a decode takes by default, 1 in the release.
    public var defaultSteps: Int
    public var spatialCompression: Int
    public var temporalCompression: Int

    public init(outChannels: Int = 3, latentChannels: Int = 128, patchSize: Int = 4, headDim: Int = 64,
                stageChannels: [Int] = [2048, 1024, 512, 512, 256], stageDepths: [Int] = [4, 6, 4, 2, 8],
                stageKernels: [[Int]] = [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5]],
                upsampleStrides: [[Int]] = [[1, 2, 2], [2, 1, 1], [2, 2, 2], [2, 2, 2]],
                upsampleReductions: [Int] = [2, 2, 1, 2], diffusionKernel: [Int] = [11, 11, 11],
                timeEmbeddingDim: Int = 384, timestepScale: Float = 1000, predictsSample: Bool = true,
                defaultSteps: Int = 1, spatialCompression: Int = 32, temporalCompression: Int = 8) {
        self.outChannels = outChannels
        self.latentChannels = latentChannels
        self.patchSize = patchSize
        self.headDim = headDim
        self.stageChannels = stageChannels
        self.stageDepths = stageDepths
        self.stageKernels = stageKernels
        self.upsampleStrides = upsampleStrides
        self.upsampleReductions = upsampleReductions
        self.diffusionKernel = diffusionKernel
        self.timeEmbeddingDim = timeEmbeddingDim
        self.timestepScale = timestepScale
        self.predictsSample = predictsSample
        self.defaultSteps = defaultSteps
        self.spatialCompression = spatialCompression
        self.temporalCompression = temporalCompression
    }

    /// The LTX-2.5 release's decoder (`Lightricks/LTX-2.5-Diffusers`, `diffusion_decoder/config.json`).
    public static let ltx25 = NFKMLXLTX2DiffusionDecoderConfiguration()

    /// The release's stage arrangement at a sixteenth of its width, with windows small enough for tiny
    /// grids and unequal per axis, for weight-free tests. Its 16-wide heads split the rotary 4 / 6 / 6.
    public static let tiny = NFKMLXLTX2DiffusionDecoderConfiguration(
        latentChannels: 8, headDim: 16, stageChannels: [128, 64, 32, 32, 32], stageDepths: [1, 2, 1, 1, 2],
        stageKernels: [[3, 3, 3], [3, 5, 5], [3, 3, 3], [3, 5, 5]], upsampleReductions: [2, 2, 1, 1],
        diffusionKernel: [5, 7, 7], timeEmbeddingDim: 16)

    /// The trailing latent frames replicated before stage 1, `2 ⌊kT / 2⌋` of the first stage's window.
    var trailingPadFrames: Int { (stageKernels[0][0] / 2) * 2 }
}

/// One axis of a neighborhood window over a grid of `length` positions.
///
/// @discussion Queries go in blocks of `block` consecutive positions; a block's keys are the `box`
/// positions from ``boxStart(_:)``, which hold every key any of its queries reads. The last block starts
/// at `length − block`, so it may repeat positions of the one before; ``skip(_:)`` counts them.
struct NFKLTX2NeighborhoodAxis {
    let length: Int
    let kernel: Int
    let block: Int
    let box: Int
    let starts: [Int]

    init(length: Int, kernel: Int, block: Int = 8) {
        self.length = length
        self.kernel = min(kernel, length)
        self.block = min(block, length)
        box = min(length, self.block + self.kernel - 1)
        starts = Array(stride(from: 0, to: length - self.block, by: self.block)) + [length - self.block]
    }

    /// The first position of the window query `q` reads: centered, shifted inward at the borders.
    func windowStart(_ q: Int) -> Int { min(max(q - kernel / 2, 0), length - kernel) }

    /// The first key of the box the block at `start` reads.
    func boxStart(_ start: Int) -> Int { min(max(start - kernel / 2, 0), length - box) }

    /// `[block, box]`: whether each query of the block at `start` reads each key of its box.
    func mask(_ start: Int) -> MLXArray {
        let low = boxStart(start)
        let first = MLXArray((0 ..< block).map { Int32(windowStart(start + $0) - low) }).reshaped([block, 1])
        let keys = MLXArray((0 ..< box).map(Int32.init)).reshaped([1, box])
        return (keys .>= first) .&& (keys .< first + Int32(kernel))
    }

    /// The leading positions of block `index` the block before it already covers.
    func skip(_ index: Int) -> Int { index == 0 ? 0 : starts[index - 1] + block - starts[index] }
}

/// The scores one neighborhood call may hold, bounding how many width blocks run together.
private let ltx2NeighborhoodScoreBytes = 256 << 20

/// 3-D neighborhood attention over `[B, T, H, W, heads, D]` queries, keys, and values (the query already
/// scaled) → `[B, T, H, W, heads · D]`.
///
/// @discussion Time and height blocks run in turn and the width blocks of one row together: each width
/// block gathers its key box, and one masked attention covers them all.
func ltx2NeighborhoodAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, kernel: [Int]) -> MLXArray {
    let (batch, heads, depth) = (q.dim(0), q.dim(4), q.dim(5))
    let time = NFKLTX2NeighborhoodAxis(length: q.dim(1), kernel: kernel[0])
    let rows = NFKLTX2NeighborhoodAxis(length: q.dim(2), kernel: kernel[1])
    let cols = NFKLTX2NeighborhoodAxis(length: q.dim(3), kernel: kernel[2])
    let queries = time.block * rows.block * cols.block
    let keys = time.box * rows.box * cols.box
    let perBlock = batch * heads * queries * keys * 4
    let groupSize = max(1, ltx2NeighborhoodScoreBytes / max(perBlock, 1))
    let groups = stride(from: 0, to: cols.starts.count, by: groupSize).map {
        Array($0 ..< min($0 + groupSize, cols.starts.count))
    }
    let colMasks = cols.starts.map { cols.mask($0) }
    let kept = MLXArray(cols.starts.indices.flatMap { index in
        (cols.skip(index) ..< cols.block).map { Int32(index * cols.block + $0) }
    })
    let evaluatesRows = q.dim(1) * q.dim(2) * q.dim(3) > 1 << 16

    var frames = [MLXArray]()
    for (timeIndex, timeStart) in time.starts.enumerated() {
        let timeLow = time.boxStart(timeStart)
        let timeMask = time.mask(timeStart)
        var bands = [MLXArray]()
        for (rowIndex, rowStart) in rows.starts.enumerated() {
            let rowLow = rows.boxStart(rowStart)
            let rowMask = rows.mask(rowStart)
            let qBand = q[0..., timeStart ..< timeStart + time.block, rowStart ..< rowStart + rows.block]
            let kBand = k[0..., timeLow ..< timeLow + time.box, rowLow ..< rowLow + rows.box]
            let vBand = v[0..., timeLow ..< timeLow + time.box, rowLow ..< rowLow + rows.box]
            var pieces = [MLXArray]()
            for group in groups {
                let count = group.count
                let queryColumns = MLXArray(group.flatMap { index in
                    (0 ..< cols.block).map { Int32(cols.starts[index] + $0) }
                })
                let keyColumns = MLXArray(group.flatMap { index in
                    (0 ..< cols.box).map { Int32(cols.boxStart(cols.starts[index]) + $0) }
                })
                func gathered(_ x: MLXArray, _ columns: MLXArray, _ t: Int, _ h: Int, _ w: Int) -> MLXArray {
                    x.take(columns, axis: 3).reshaped([batch, t, h, count, w, heads, depth])
                        .transposed(0, 3, 5, 1, 2, 4, 6).reshaped([batch * count, heads, t * h * w, depth])
                }
                let qg = gathered(qBand, queryColumns, time.block, rows.block, cols.block)
                let kg = gathered(kBand, keyColumns, time.box, rows.box, cols.box)
                let vg = gathered(vBand, keyColumns, time.box, rows.box, cols.box)
                let colMask = stacked(group.map { colMasks[$0] })
                let mask = timeMask.reshaped([1, time.block, 1, 1, time.box, 1, 1])
                    .&& rowMask.reshaped([1, 1, rows.block, 1, 1, rows.box, 1])
                    .&& colMask.reshaped([count, 1, 1, cols.block, 1, 1, cols.box])
                let laid = broadcast(mask.reshaped([1, count, 1, queries, keys]),
                                     to: [batch, count, 1, queries, keys]).reshaped([batch * count, 1, queries, keys])
                let out = MLXFast.scaledDotProductAttention(queries: qg, keys: kg, values: vg, scale: 1, mask: laid)
                pieces.append(out.reshaped([batch, count, heads, time.block, rows.block, cols.block, depth])
                    .transposed(0, 3, 4, 1, 5, 2, 6)
                    .reshaped([batch, time.block, rows.block, count * cols.block, heads * depth]))
            }
            let band = concatenated(pieces, axis: 3).take(kept, axis: 3)
            bands.append(band[0..., 0..., rows.skip(rowIndex)...])
        }
        let frame = concatenated(bands, axis: 2)[0..., time.skip(timeIndex)...]
        if evaluatesRows { eval(frame) }
        frames.append(frame)
    }
    return concatenated(frames, axis: 1)
}

/// The decoder's absolute 3-D rotary: the head splits into (time, height, width) chunks, a quarter to
/// time and the rest halved, each rotating adjacent channel pairs by its axis's 0-based index at
/// `base ^ (−2i / chunk)`.
struct NFKLTX2DecoderRotary {
    let split: (Int, Int, Int)
    let base: Double

    init(headDim: Int, base: Double = 10000) {
        var time = (headDim / 4) / 2 * 2
        var space = (headDim - time) / 2
        if space % 2 != 0 {
            time -= 2
            space = (headDim - time) / 2
        }
        split = (time, space, space)
        self.base = base
    }

    /// `[length, dim / 2]` angles, the frequencies formed in double precision as the reference does.
    func angles(length: Int, dim: Int) -> MLXArray {
        let frequencies = stride(from: 0, to: dim, by: 2).map { Float(1 / pow(base, Double($0) / Double(dim))) }
        let positions = MLXArray((0 ..< length).map(Float.init)).reshaped([length, 1])
        return positions * MLXArray(frequencies).reshaped([1, -1])
    }

    func rotate(_ x: MLXArray, angles: MLXArray, axis: Int) -> MLXArray {
        var shape = [1, 1, 1, 1, 1, angles.dim(1)]
        shape[axis] = angles.dim(0)
        let c = cos(angles).reshaped(shape), s = sin(angles).reshaped(shape)
        let pairs = x.asType(.float32).reshaped(Array(x.shape.dropLast()) + [x.dim(-1) / 2, 2])
        let even = pairs[.ellipsis, 0], odd = pairs[.ellipsis, 1]
        return stacked([even * c - odd * s, even * s + odd * c], axis: -1).reshaped(x.shape).asType(x.dtype)
    }

    /// `[B, T, H, W, heads, headDim]`, rotated.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (t, h, _) = split
        let timeAngles = angles(length: x.dim(1), dim: split.0)
        let rowAngles = angles(length: x.dim(2), dim: split.1)
        let colAngles = angles(length: x.dim(3), dim: split.2)
        return concatenated([
            rotate(x[.ellipsis, 0 ..< t], angles: timeAngles, axis: 1),
            rotate(x[.ellipsis, t ..< t + h], angles: rowAngles, axis: 2),
            rotate(x[.ellipsis, (t + h)...], angles: colAngles, axis: 3),
        ], axis: -1)
    }
}

/// The decoder's neighborhood attention: biased projections, a per-head RMS norm on the query and key,
/// the query scaled, both rotated, and the output projection.
final class NFKLTX2NeighborhoodAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    let heads: Int
    let headDim: Int
    let kernel: [Int]
    let rotary: NFKLTX2DecoderRotary

    init(dim: Int, kernel: [Int], headDim: Int) {
        heads = dim / headDim
        self.headDim = headDim
        self.kernel = kernel
        rotary = NFKLTX2DecoderRotary(headDim: headDim)
        _toQ.wrappedValue = Linear(dim, dim)
        _toK.wrappedValue = Linear(dim, dim)
        _toV.wrappedValue = Linear(dim, dim)
        _toOut.wrappedValue = [Linear(dim, dim)]
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
    }

    /// `[B, T, H, W, C]` → `[B, T, H, W, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = Array(x.shape.dropLast()) + [heads, headDim]
        let q = normQ(toQ(x).reshaped(shape)) * Float(pow(Double(headDim), -0.5))
        let k = normK(toK(x).reshaped(shape))
        let v = toV(x).reshaped(shape)
        return toOut[0](ltx2NeighborhoodAttention(rotary(q), rotary(k), v, kernel: kernel))
    }
}

/// The gated MLP `w_down(silu(w_gate(x)) · w_up(x))`, evaluated over at most 16384 tokens at a time so the
/// three hidden-width tensors stay bounded; the MLP is pointwise across tokens.
final class NFKLTX2SwiGLU: Module {
    @ModuleInfo(key: "w_up") var up: Linear
    @ModuleInfo(key: "w_gate") var gate: Linear
    @ModuleInfo(key: "w_down") var down: Linear

    static let tile = 16384

    init(dim: Int, ratio: Double = 4) {
        let hidden = (Int(Double(dim) * ratio) + 15) / 16 * 16
        _up.wrappedValue = Linear(dim, hidden, bias: false)
        _gate.wrappedValue = Linear(dim, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let tokens = x.size / x.dim(-1)
        func mlp(_ t: MLXArray) -> MLXArray { down(NFKReferenceRounding.silu(gate(t)) * up(t)) }
        guard tokens > Self.tile else { return mlp(x) }
        let flat = x.reshaped([tokens, x.dim(-1)])
        let tiles = stride(from: 0, to: tokens, by: Self.tile).map { start -> MLXArray in
            let tile = mlp(flat[start ..< min(start + Self.tile, tokens)])
            eval(tile)
            return tile
        }
        return concatenated(tiles, axis: 0).reshaped(x.shape)
    }
}

/// A deterministic-stage block: pre-norm neighborhood attention and gated MLP, each added back.
final class NFKLTX2DecoderBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn") var attn: NFKLTX2NeighborhoodAttention
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: NFKLTX2SwiGLU

    init(dim: Int, kernel: [Int], headDim: Int) {
        _norm1.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-6)
        _attn.wrappedValue = NFKLTX2NeighborhoodAttention(dim: dim, kernel: kernel, headDim: headDim)
        _norm2.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-6)
        _mlp.wrappedValue = NFKLTX2SwiGLU(dim: dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attn(norm1(x))
        return h + mlp(norm2(h))
    }
}

/// A stage-5 block: the latent context projected in, then the deterministic block's two halves with each
/// norm's output scaled and shifted by the shared modulation plus the block's own table. The residuals
/// are ungated: the release folds its static gates into the following weights.
final class NFKLTX2DecoderDiffusionBlock: Module {
    @ModuleInfo(key: "context_proj") var contextProj: Linear
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "attn") var attn: NFKLTX2NeighborhoodAttention
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: NFKLTX2SwiGLU

    init(dim: Int, kernel: [Int], contextChannels: Int, headDim: Int, parameters: Int) {
        _contextProj.wrappedValue = Linear(contextChannels, dim)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([parameters, dim])
        _norm1.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-6)
        _attn.wrappedValue = NFKLTX2NeighborhoodAttention(dim: dim, kernel: kernel, headDim: headDim)
        _norm2.wrappedValue = RMSNorm(dimensions: dim, eps: 1e-6)
        _mlp.wrappedValue = NFKLTX2SwiGLU(dim: dim)
    }

    /// `modulation` `[B, parameters, C]`, the shared projection's chunks in the reference's order (scale,
    /// shift, gate for attention, the same for the MLP, a context gate).
    func callAsFunction(_ x: MLXArray, context: MLXArray, modulation: MLXArray) -> MLXArray {
        let table = modulation + scaleShiftTable
        func chunk(_ index: Int) -> MLXArray { table[0..., index].reshaped([table.dim(0), 1, 1, 1, -1]) }
        var h = x + contextProj(context)
        h = h + attn(norm1(h) * (1 + chunk(0)) + chunk(1))
        return h + mlp(norm2(h) * (1 + chunk(3)) + chunk(4))
    }
}

/// A linear channel expansion and a channels-last depth-to-space shuffle by `stride`. A temporal stride of
/// two leaves a duplicate leading frame, which is dropped unless a temporal tile away from the clip's
/// start asks to keep it.
final class NFKLTX2DecoderUpsampler: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    let stride: [Int]
    let outChannels: Int

    init(inChannels: Int, stride: [Int], reduction: Int) {
        self.stride = stride
        let expanded = stride.reduce(1, *) * inChannels / reduction
        outChannels = expanded / stride.reduce(1, *)
        _proj.wrappedValue = Linear(inChannels, expanded)
    }

    func callAsFunction(_ x: MLXArray, dropsLeadingFrame: Bool = true) -> MLXArray {
        let (b, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let shuffled = proj(x).reshaped([b, t, h, w, outChannels, stride[0], stride[1], stride[2]])
            .transposed(0, 1, 5, 2, 6, 3, 7, 4)
            .reshaped([b, t * stride[0], h * stride[1], w * stride[2], outChannels])
        return stride[0] == 2 && dropsLeadingFrame ? shuffled[0..., 1...] : shuffled
    }
}

/// The shared adaptive-norm head: SiLU and one linear from the embedded timestep to seven modulation
/// vectors.
final class NFKLTX2DecoderAdaptiveNorm: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    let parameters: Int

    init(dim: Int, timeEmbeddingDim: Int, parameters: Int = 7) {
        self.parameters = parameters
        _proj.wrappedValue = Linear(timeEmbeddingDim, parameters * dim)
    }

    /// `[B, timeEmbeddingDim]` → `[B, parameters, dim]`.
    func callAsFunction(_ embedded: MLXArray) -> MLXArray {
        proj(NFKReferenceRounding.silu(embedded)).reshaped([embedded.dim(0), parameters, -1])
    }
}

/// The inner decoder, at the release's `decoder.` prefix.
public final class NFKMLXLTX2DiffusionDecoderNet: Module {
    @ModuleInfo(key: "conv_in") var convIn: Linear
    @ModuleInfo(key: "det_stages") var stages: [[NFKLTX2DecoderBlock]]
    @ModuleInfo(key: "upsamples") var upsamples: [NFKLTX2DecoderUpsampler]
    @ModuleInfo(key: "t_embedder") var timeEmbedder: NFKLTX2TimestepEmbedder
    @ModuleInfo(key: "conv_in_x_t") var convInSample: Linear
    @ModuleInfo(key: "shared_adaln") var sharedAdaptiveNorm: NFKLTX2DecoderAdaptiveNorm
    @ModuleInfo(key: "diff_blocks") var diffusionBlocks: [NFKLTX2DecoderDiffusionBlock]
    @ModuleInfo(key: "norm_out") var normOut: RMSNorm
    @ModuleInfo(key: "conv_out") var convOut: Linear

    public let configuration: NFKMLXLTX2DiffusionDecoderConfiguration

    public init(_ c: NFKMLXLTX2DiffusionDecoderConfiguration) {
        configuration = c
        _convIn.wrappedValue = Linear(c.latentChannels, c.stageChannels[0])
        _stages.wrappedValue = c.upsampleStrides.indices.map { stage in
            (0 ..< c.stageDepths[stage]).map { _ in
                NFKLTX2DecoderBlock(dim: c.stageChannels[stage], kernel: c.stageKernels[stage], headDim: c.headDim)
            }
        }
        _upsamples.wrappedValue = c.upsampleStrides.indices.map {
            NFKLTX2DecoderUpsampler(inChannels: c.stageChannels[$0], stride: c.upsampleStrides[$0],
                                    reduction: c.upsampleReductions[$0])
        }
        _timeEmbedder.wrappedValue = NFKLTX2TimestepEmbedder(c.timeEmbeddingDim)
        let width = c.stageChannels[c.stageChannels.count - 1]
        let pixels = c.outChannels * c.patchSize * c.patchSize
        _convInSample.wrappedValue = Linear(pixels, width)
        _sharedAdaptiveNorm.wrappedValue = NFKLTX2DecoderAdaptiveNorm(dim: width, timeEmbeddingDim: c.timeEmbeddingDim)
        _diffusionBlocks.wrappedValue = (0 ..< c.stageDepths[c.stageDepths.count - 1]).map { _ in
            NFKLTX2DecoderDiffusionBlock(dim: width, kernel: c.diffusionKernel, contextChannels: width,
                                         headDim: c.headDim, parameters: 7)
        }
        _normOut.wrappedValue = RMSNorm(dimensions: width, eps: 1e-6)
        _convOut.wrappedValue = Linear(width, pixels)
    }

    /// Every deterministic stage but the last: a latent `[B, F, H, W, latentChannels]` → features, the
    /// replicated trailing frames still in them.
    public func earlyStages(_ latent: MLXArray) -> MLXArray {
        let pad = configuration.trailingPadFrames
        var h = latent
        if pad > 0 {
            let last = latent[0..., (latent.dim(1) - 1)...]
            h = concatenated([latent, broadcast(last, to: [last.dim(0), pad] + Array(last.shape.dropFirst(2)))],
                             axis: 1)
        }
        h = convIn(h)
        for stage in 0 ..< stages.count - 1 {
            h = stages[stage].reduce(h) { $1($0) }
            h = upsamples[stage](h)
        }
        return h
    }

    /// The last deterministic stage: ``earlyStages(_:)`` features → the stage-5 context. A temporal tile
    /// away from the clip's start keeps the shuffle's leading frame, and only the tile at its end crops
    /// the replicated frames.
    public func lastStage(_ features: MLXArray, dropsLeadingFrame: Bool = true,
                          cropsTrailingFrames: Bool = true) -> MLXArray {
        var h = stages[stages.count - 1].reduce(features) { $1($0) }
        h = upsamples[upsamples.count - 1](h, dropsLeadingFrame: dropsLeadingFrame)
        let crop = configuration.trailingPadFrames * configuration.temporalCompression
        if cropsTrailingFrames, crop > 0 {
            h = h[0..., 0 ..< (h.dim(1) - crop)]
        }
        return h
    }

    /// The decode's context for a latent, the seam the reference's `forward_stage_4` returns.
    public func context(_ latent: MLXArray) -> MLXArray { lastStage(earlyStages(latent)) }

    /// One stage-5 pass: the prediction for the noisy pixels `sample` `[B, F, 4 h, 4 w, 3]` at `timestep`
    /// (one per batch row, in `0…1`) against `context` `[B, F, h, w, C]`.
    public func prediction(context: MLXArray, sample: MLXArray, timestep: MLXArray) -> MLXArray {
        let embedded = timeEmbedder(timestep * configuration.timestepScale)
        let modulation = sharedAdaptiveNorm(embedded)
        var h = convInSample(ltx2Patchify(sample, patch: configuration.patchSize))
        for block in diffusionBlocks {
            h = block(h, context: context, modulation: modulation)
        }
        return ltx2Unpatchify(convOut(normOut(h)), patch: configuration.patchSize)
    }

    /// Denoises `sample` over `steps` stage-5 steps from `t = 1`. A one-step x0 decoder returns its single
    /// prediction; otherwise each step is a reverse Euler update in float32.
    public func denoise(context: MLXArray, sample initial: MLXArray, steps: Int) -> MLXArray {
        let batch = context.dim(0)
        let times = (0 ..< steps).map { steps == 1 ? Float(1) : 1 - Float($0) * (1 - 1 / Float(steps)) / Float(steps - 1) }
        if steps == 1, configuration.predictsSample {
            return prediction(context: context, sample: initial, timestep: MLXArray([Float](repeating: 1, count: batch)))
        }
        var sample = initial
        for index in 0 ..< steps {
            let now = times[index]
            let next = index + 1 < steps ? times[index + 1] : 0
            var output = prediction(context: context, sample: sample,
                                    timestep: MLXArray([Float](repeating: now, count: batch))).asType(.float32)
            let wide = sample.asType(.float32)
            if configuration.predictsSample {
                output = (wide - output) / now
            }
            sample = (wide - (now - next) * output).asType(sample.dtype)
        }
        return sample
    }
}

/// Space-to-depth on height and width, channels-last: `[B, F, H, W, C]` → `[B, F, H / p, W / p, C p²]`,
/// the channel packed `(channel, width offset, height offset)` as the reference packs it.
func ltx2Patchify(_ x: MLXArray, patch p: Int) -> MLXArray {
    let (b, f, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
    return x.reshaped([b, f, h / p, p, w / p, p, c]).transposed(0, 1, 2, 4, 6, 5, 3)
        .reshaped([b, f, h / p, w / p, c * p * p])
}

/// The inverse of ``ltx2Patchify(_:patch:)``.
func ltx2Unpatchify(_ x: MLXArray, patch p: Int) -> MLXArray {
    let (b, f, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let c = x.dim(4) / (p * p)
    return x.reshaped([b, f, h, w, c, p, p]).transposed(0, 1, 2, 6, 3, 5, 4).reshaped([b, f, h * p, w * p, c])
}

/// The tiles a tiled decode cuts: sizes and strides in decoded pixels and frames, the reference's
/// defaults.
public struct NFKMLXLTX2DecodeTiling: Sendable, Equatable {
    public var height: Int
    public var width: Int
    public var frames: Int
    public var heightStride: Int
    public var widthStride: Int
    public var frameStride: Int

    public init(height: Int = 768, width: Int = 768, frames: Int = 80, heightStride: Int = 704,
                widthStride: Int = 704, frameStride: Int = 56) {
        self.height = height
        self.width = width
        self.frames = frames
        self.heightStride = heightStride
        self.widthStride = widthStride
        self.frameStride = frameStride
    }
}

/// The LTX-2.5 diffusion decoder with its latent statistics.
public final class NFKMLXLTX2DiffusionDecoder {
    public let decoder: NFKMLXLTX2DiffusionDecoderNet
    /// The latents' per-channel mean and standard deviation the release carries.
    public let statistics: (mean: MLXArray, std: MLXArray)

    public init(decoder: NFKMLXLTX2DiffusionDecoderNet, statistics: (mean: MLXArray, std: MLXArray)) {
        self.decoder = decoder
        self.statistics = statistics
    }

    public var configuration: NFKMLXLTX2DiffusionDecoderConfiguration { decoder.configuration }

    /// Decodes a denormalized latent `[B, F, H, W, C]` → the clip `[B, 8 (F − 1) + 1, 32 H, 32 W, 3]` in
    /// about `−1…1`. `noise` draws the pixel noise each denoise starts from, given its shape; `tiling`,
    /// where set and the latent exceeds one tile, runs the last deterministic stage and stage 5 per tile.
    public func decode(_ latent: MLXArray, steps: Int? = nil, tiling: NFKMLXLTX2DecodeTiling? = nil,
                       noise: ([Int]) -> MLXArray) -> MLXArray {
        let c = configuration
        let steps = steps ?? c.defaultSteps
        let latent = latent.asType(decoder.convIn.weight.dtype)
        if let tiling, latent.dim(1) > tiling.frames / c.temporalCompression
            || latent.dim(2) > tiling.height / c.spatialCompression
            || latent.dim(3) > tiling.width / c.spatialCompression {
            return tiledDecode(latent, steps: steps, tiling: tiling, noise: noise)
        }
        let context = decoder.context(latent)
        eval(context)
        let shape = [latent.dim(0), context.dim(1), context.dim(2) * c.patchSize, context.dim(3) * c.patchSize,
                     c.outChannels]
        return decoder.denoise(context: context, sample: noise(shape).asType(context.dtype), steps: steps)
    }

    /// The reference's tile intervals: overlapping `[start, end)` spans spaced `stride` apart, a trailing
    /// remnant shorter than `minimum` merged into the tile before it.
    static func intervals(length: Int, tile: Int, stride: Int, minimum: Int) -> [(Int, Int)] {
        guard length > tile else { return [(0, length)] }
        var starts = Array(Swift.stride(from: 0, to: length, by: stride))
        while starts.count > 1, length - starts[starts.count - 1] < minimum {
            starts.removeLast()
        }
        return starts.dropLast().map { ($0, min($0 + tile, length)) } + [(starts[starts.count - 1], length)]
    }

    /// `b` with its first `extent` positions on `axis` blended linearly from `a`'s last ones.
    static func blend(_ a: MLXArray, _ b: MLXArray, axis: Int, extent requested: Int) -> MLXArray {
        let extent = min(a.dim(axis), b.dim(axis), requested)
        guard extent > 0 else { return b }
        var shape = [Int](repeating: 1, count: b.ndim)
        shape[axis] = extent
        let weight = MLXArray((0 ..< extent).map { Float($0) / Float(extent) }).reshaped(shape).asType(b.dtype)
        func span(_ x: MLXArray, _ range: Range<Int>) -> MLXArray { x.take(MLXArray(range.map(Int32.init)), axis: axis) }
        let tail = span(a, (a.dim(axis) - extent) ..< a.dim(axis))
        let head = span(b, 0 ..< extent)
        let rest = span(b, extent ..< b.dim(axis))
        return concatenated([tail * (1 - weight) + head * weight, rest], axis: axis)
    }

    /// The reference's tiled decode: the early stages over the whole latent, the last deterministic stage
    /// and stage 5 per tile on the grid entering the last stage, the tiles blended over their overlaps. A
    /// one-step x0 decode draws each tile's noise; a multi-step one slices one canvas.
    func tiledDecode(_ latent: MLXArray, steps: Int, tiling: NFKMLXLTX2DecodeTiling,
                     noise: ([Int]) -> MLXArray) -> MLXArray {
        let c = configuration
        let p = c.patchSize
        let stride = c.upsampleStrides[c.upsampleStrides.count - 1]
        let scale = (stride[0], stride[1] * p, stride[2] * p)
        let tile = (tiling.frames / scale.0, tiling.height / scale.1, tiling.width / scale.2)
        let step = (tiling.frameStride / scale.0, tiling.heightStride / scale.1, tiling.widthStride / scale.2)
        let lastKernel = c.stageKernels[c.stageKernels.count - 1]
        let minimum = (0 ..< 3).map { max(lastKernel[$0], (c.diffusionKernel[$0] + stride[$0] - 1) / stride[$0]) }

        let features = decoder.earlyStages(latent)
        eval(features)
        let ghost = c.trailingPadFrames * c.upsampleStrides.dropLast().reduce(1) { $0 * $1[0] }
        let frames = features.dim(1) - ghost
        let (height, width) = (features.dim(2), features.dim(3))
        let timeTiles = Self.intervals(length: frames, tile: tile.0, stride: step.0, minimum: minimum[0])
        let rowTiles = Self.intervals(length: height, tile: tile.1, stride: step.1, minimum: minimum[1])
        let colTiles = Self.intervals(length: width, tile: tile.2, stride: step.2, minimum: minimum[2])
        let blendFrames = (tile.0 - step.0) * scale.0
        let blendRows = (tile.1 - step.1) * scale.1
        let blendCols = (tile.2 - step.2) * scale.2
        let singleStep = steps == 1 && c.predictsSample
        let duplicates = scale.0 == 2 ? 1 : 0
        let canvas: MLXArray? = singleStep ? nil : noise([
            latent.dim(0), frames * scale.0 - duplicates, height * scale.1, width * scale.2, c.outChannels,
        ]).asType(features.dtype)

        var groups = [MLXArray]()
        for (t0, t1) in timeTiles {
            let origin = t0 == 0, trailing = t1 == frames
            let featureEnd = trailing ? features.dim(1) : t1
            var rows = [[MLXArray]]()
            for (h0, h1) in rowTiles {
                var row = [MLXArray]()
                for (w0, w1) in colTiles {
                    let context = decoder.lastStage(features[0..., t0 ..< featureEnd, h0 ..< h1, w0 ..< w1],
                                                    dropsLeadingFrame: origin, cropsTrailingFrames: trailing)
                    eval(context)
                    let shape = [latent.dim(0), context.dim(1), context.dim(2) * p, context.dim(3) * p, c.outChannels]
                    let sample: MLXArray
                    if let canvas {
                        let pixelStart = t0 * scale.0 - (origin ? 0 : duplicates)
                        sample = canvas[0..., pixelStart ..< pixelStart + shape[1],
                                        h0 * scale.1 ..< h0 * scale.1 + shape[2],
                                        w0 * scale.2 ..< w0 * scale.2 + shape[3]]
                    } else {
                        sample = noise(shape).asType(context.dtype)
                    }
                    let decoded = decoder.denoise(context: context, sample: sample, steps: steps)
                    eval(decoded)
                    row.append(decoded)
                }
                rows.append(row)
            }
            // The reference blends in place, so a tile's blended form is what the tiles after it blend from.
            var bands = [MLXArray]()
            for i in rows.indices {
                var pieces = [MLXArray]()
                for j in rows[i].indices {
                    var tile = rows[i][j]
                    if i > 0 { tile = Self.blend(rows[i - 1][j], tile, axis: 2, extent: blendRows) }
                    if j > 0 { tile = Self.blend(rows[i][j - 1], tile, axis: 3, extent: blendCols) }
                    rows[i][j] = tile
                    let keepRows = i < rows.count - 1 ? step.1 * scale.1 : tile.dim(2)
                    let keepCols = j < rows[i].count - 1 ? step.2 * scale.2 : tile.dim(3)
                    pieces.append(tile[0..., 0..., 0 ..< keepRows, 0 ..< keepCols])
                }
                bands.append(concatenated(pieces, axis: 3))
            }
            groups.append(concatenated(bands, axis: 2))
        }
        var result = [MLXArray]()
        for k in groups.indices {
            var group = groups[k]
            if k > 0 { group = Self.blend(groups[k - 1], group, axis: 1, extent: blendFrames) }
            groups[k] = group
            if k < groups.count - 1 {
                group = group[0..., 0 ..< (step.0 * scale.0 - (k == 0 ? duplicates : 0))]
            }
            result.append(group)
        }
        return concatenated(result, axis: 1)
    }
}

/// Reading, building, and loading the LTX-2.5 diffusion decoder.
public enum NFKMLXLTX2DiffusionDecoders {

    /// The geometry a `diffusion_decoder/config.json` describes.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXLTX2DiffusionDecoderConfiguration {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "LTX2VideoDiffusionDecoderModel" {
            throw NFKMLXError.unsupportedConfiguration("this reads an LTX-2 diffusion video decoder, not \(kind)")
        }
        let base = NFKMLXLTX2DiffusionDecoderConfiguration.ltx25
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func integers(_ key: String, _ fallback: [Int]) -> [Int] {
            (json[key] as? [NSNumber])?.map(\.intValue) ?? fallback
        }
        func triples(_ key: String, _ fallback: [[Int]]) -> [[Int]] {
            (json[key] as? [[NSNumber]])?.map { $0.map(\.intValue) } ?? fallback
        }
        let output = json["decoder_model_output_type"] as? String ?? "x0"
        guard output == "x0" || output == "v" else {
            throw NFKMLXError.unsupportedConfiguration("a diffusion decoder predicts x0 or v, not \(output)")
        }
        let configuration = NFKMLXLTX2DiffusionDecoderConfiguration(
            outChannels: integer("out_channels", base.outChannels),
            latentChannels: integer("latent_channels", base.latentChannels),
            patchSize: integer("patch_size", base.patchSize),
            headDim: integer("decoder_head_dim", base.headDim),
            stageChannels: integers("decoder_stage_channels", base.stageChannels),
            stageDepths: integers("decoder_stage_depths", base.stageDepths),
            stageKernels: triples("decoder_stage_kernels", base.stageKernels),
            upsampleStrides: triples("decoder_upsample_strides", base.upsampleStrides),
            upsampleReductions: integers("decoder_upsample_channel_reductions", base.upsampleReductions),
            diffusionKernel: integers("decoder_stage5_kernel", base.diffusionKernel),
            timeEmbeddingDim: integer("decoder_t_emb_dim", base.timeEmbeddingDim),
            timestepScale: (json["decoder_timestep_scale_multiplier"] as? NSNumber)?.floatValue ?? base.timestepScale,
            predictsSample: output == "x0",
            defaultSteps: integer("decoder_num_inference_steps", base.defaultSteps),
            spatialCompression: integer("spatial_compression_ratio", base.spatialCompression),
            temporalCompression: integer("temporal_compression_ratio", base.temporalCompression))
        if (json["scaling_factor"] as? NSNumber).map({ $0.doubleValue != 1 }) == true {
            throw NFKMLXError.unsupportedConfiguration("this decoder reads latents at a scaling factor of 1")
        }
        let stages = configuration.upsampleStrides.count
        guard configuration.stageChannels.count == stages + 1, configuration.stageDepths.count == stages + 1,
              configuration.stageKernels.count == stages, configuration.upsampleReductions.count == stages,
              configuration.headDim % 8 == 0 else {
            throw NFKMLXError.unsupportedConfiguration("the diffusion decoder's stage lists disagree in length")
        }
        for stage in 0 ..< stages
        where configuration.stageChannels[stage + 1] != configuration.stageChannels[stage] / configuration.upsampleReductions[stage] {
            throw NFKMLXError.unsupportedConfiguration(
                "stage \(stage + 1) of the diffusion decoder is not stage \(stage)'s width over its reduction")
        }
        return configuration
    }

    /// The release tensors a text-to-video decode never reads: `decoder.type_emb`, the tag Lightricks'
    /// keyframe-aware decode adds to keyframe latents before `conv_in`, which diffusers' module does not
    /// declare.
    static func isUnread(_ key: String) -> Bool { key == "decoder.type_emb" }

    /// Loads a release's `diffusion_decoder/` directory at `precision` and returns the decoder with the
    /// latents' statistics.
    public static func load(from directory: URL, precision: NFKMLXWeightPrecision = .float32) throws
        -> NFKMLXLTX2DiffusionDecoder {
        let net = NFKMLXLTX2DiffusionDecoderNet(
            try configuration(fromHuggingFace: directory.appendingPathComponent("config.json")))
        let arrays = try NFKMLXReleaseWeights.materializedArrays(
            inDirectory: directory, precision: precision,
            remap: { ($0.hasPrefix("decoder.") && !isUnread($0)) || $0.hasPrefix("latents_") ? $0 : nil })
        return try apply(arrays, to: net)
    }

    /// Applies `decoder.`-prefixed tensors and the latent statistics.
    static func apply(_ arrays: [(String, MLXArray)], to net: NFKMLXLTX2DiffusionDecoderNet) throws
        -> NFKMLXLTX2DiffusionDecoder {
        var statistics = [String: MLXArray]()
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays {
            if key == "latents_mean" || key == "latents_std" {
                statistics[key] = value.asType(.float32)
            } else if key.hasPrefix("decoder."), !isUnread(key) {
                mapped.append((String(key.dropFirst("decoder.".count)), value))
            }
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
        let channels = net.configuration.latentChannels
        return NFKMLXLTX2DiffusionDecoder(
            decoder: net, statistics: (statistics["latents_mean"] ?? MLXArray.zeros([channels]),
                                       statistics["latents_std"] ?? MLXArray.ones([channels])))
    }
}
