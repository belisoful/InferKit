//
//  NFKMLXFlorence2.swift
//  InferKitMLX
//

// Florence-2 (Microsoft, MIT) is a unified vision model: one image plus a task prompt in, and a text
// sequence out that carries captions, detected boxes, or segmentation polygons as location tokens.
// The vision encoder is DaViT, whose every block pairs a windowed SPATIAL attention with a grouped
// CHANNEL attention, a dual-attention form the toolkit did not have. A multi-modal projector turns the
// vision grid into a token sequence, which scatters into a BART encoder-decoder at the image-token
// positions; the decoder cross-attends and the head emits the vocabulary, location tokens included.
//
// This file ports the DaViT vision tower and the projector at reference parity. The BART language model
// is the shared NFKMLXSeq2SeqTransformer (model_type "florence2_language"); the fusion is wired in
// NFKMLXFlorence2+Generate.swift. Tensors flow NHWC; convolution weights transpose from PyTorch
// [out, in, kH, kW] to MLX [out, kH, kW, in] on load. Reference: transformers modeling_florence2.py.

import Foundation
import InferKit
import MLX
import MLXNN

// MARK: - Configuration

/// Geometry of the DaViT vision tower. The default is Florence-2-large.
public struct NFKMLXFlorence2VisionConfiguration: Sendable {
    public var inChannels: Int
    public var depths: [Int]
    public var patchSize: [Int]
    public var patchStride: [Int]
    public var patchPadding: [Int]
    public var patchPreNorm: [Bool]
    public var embedDim: [Int]
    public var numHeads: [Int]
    public var numGroups: [Int]
    public var windowSize: Int
    public var mlpRatio: Double
    public var qkvBias: Bool
    public var projectionDim: Int
    public var maxPositionEmbeddings: Int

    public init(
        inChannels: Int = 3,
        depths: [Int] = [1, 1, 9, 1],
        patchSize: [Int] = [7, 3, 3, 3],
        patchStride: [Int] = [4, 2, 2, 2],
        patchPadding: [Int] = [3, 1, 1, 1],
        patchPreNorm: [Bool] = [false, true, true, true],
        embedDim: [Int] = [256, 512, 1024, 2048],
        numHeads: [Int] = [8, 16, 32, 64],
        numGroups: [Int] = [8, 16, 32, 64],
        windowSize: Int = 12,
        mlpRatio: Double = 4.0,
        qkvBias: Bool = true,
        projectionDim: Int = 1024,
        maxPositionEmbeddings: Int = 50
    ) {
        self.inChannels = inChannels
        self.depths = depths
        self.patchSize = patchSize
        self.patchStride = patchStride
        self.patchPadding = patchPadding
        self.patchPreNorm = patchPreNorm
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.numGroups = numGroups
        self.windowSize = windowSize
        self.mlpRatio = mlpRatio
        self.qkvBias = qkvBias
        self.projectionDim = projectionDim
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }

    /// Florence-2-large.
    public static let large = NFKMLXFlorence2VisionConfiguration()
    /// Florence-2-base: half the widths, and a 768-wide projection to match BART-base.
    public static let base = NFKMLXFlorence2VisionConfiguration(embedDim: [128, 256, 512, 1024],
                                                                numHeads: [4, 8, 16, 32],
                                                                numGroups: [4, 8, 16, 32],
                                                                projectionDim: 768)
}

// MARK: - Patch embedding

/// One stage's patch embedding: a strided convolution with a LayerNorm applied before it (stages 1-3)
/// or after it (stage 0). Input and output are NHWC.
final class NFKFlorence2ConvEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    let preNorm: Bool

    init(inChannels: Int, embedDim: Int, kernelSize: Int, stride: Int, padding: Int, preNorm: Bool) {
        self.preNorm = preNorm
        _proj.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: embedDim,
                                    kernelSize: IntOrPair(kernelSize), stride: IntOrPair(stride),
                                    padding: IntOrPair(padding))
        _norm.wrappedValue = LayerNorm(dimensions: preNorm ? inChannels : embedDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        if preNorm {
            out = norm(out)
        }
        out = proj(out)
        if !preNorm {
            out = norm(out)
        }
        return out
    }
}

// MARK: - Feed-forward

/// The block feed-forward: two linears with an exact-GELU between.
final class NFKFlorence2MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dim: Int, hidden: Int) {
        _fc1.wrappedValue = Linear(dim, hidden)
        _fc2.wrappedValue = Linear(hidden, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

// MARK: - Windowed spatial attention

/// Local self-attention over non-overlapping `windowSize`×`windowSize` windows. The input is padded to
/// a window multiple, partitioned, attended per window, merged, and cropped back. Scale is head_dim^-0.5.
final class NFKFlorence2WindowAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let heads: Int
    let windowSize: Int

    init(dim: Int, heads: Int, windowSize: Int, qkvBias: Bool) {
        self.heads = heads
        self.windowSize = windowSize
        _qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        _proj.wrappedValue = Linear(dim, dim)
        super.init()
    }

    /// `x`: `[B, H, W, C]`. Returns `[B, H*W, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let ws = windowSize
        let padB = (ws - h % ws) % ws
        let padR = (ws - w % ws) % ws
        var padded = x
        if padB > 0 || padR > 0 {
            padded = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, padB)), IntOrPair((0, padR)), IntOrPair((0, 0))])
        }
        let ph = h + padB
        let pw = w + padR

        // Partition into windows: [B, ph/ws, ws, pw/ws, ws, C] -> [numWin, ws*ws, C]
        var windows = padded.reshaped([b, ph / ws, ws, pw / ws, ws, c])
        windows = windows.transposed(0, 1, 3, 2, 4, 5).reshaped([-1, ws * ws, c])

        let numWin = windows.shape[0]
        let n = ws * ws
        let headDim = c / heads
        let scale = 1.0 / sqrtf(Float(headDim))
        let triple = qkv(windows).reshaped([numWin, n, 3, heads, headDim]).transposed(2, 0, 3, 1, 4)
        let q = triple[0], k = triple[1], v = triple[2]     // [numWin, heads, n, headDim]
        var attn = q.matmul(k.transposed(0, 1, 3, 2)) * scale
        attn = softmax(attn, axis: -1)
        var out = attn.matmul(v).transposed(0, 2, 1, 3).reshaped([numWin, n, c])
        out = proj(out)

        // Merge windows back and crop to [B, H*W, C].
        out = out.reshaped([b, ph / ws, pw / ws, ws, ws, c])
        out = out.transposed(0, 1, 3, 2, 4, 5).reshaped([b, ph, pw, c])
        out = out[0..., 0 ..< h, 0 ..< w, 0...]
        return out.reshaped([b, h * w, c])
    }
}

// MARK: - Grouped channel attention

/// Attention across the CHANNEL axis within groups: the tokens act as the feature dimension and each
/// group of channels attends channel-to-channel. Scale is num_tokens^-0.5. Input and output are `[B, N, C]`.
final class NFKFlorence2ChannelAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let groups: Int

    init(dim: Int, groups: Int, qkvBias: Bool) {
        self.groups = groups
        _qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        _proj.wrappedValue = Linear(dim, dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, n, c) = (x.shape[0], x.shape[1], x.shape[2])
        let d = c / groups
        let triple = qkv(x).reshaped([b, n, 3, groups, d]).transposed(2, 0, 3, 4, 1)
        let q = triple[0], k = triple[1], v = triple[2]     // [B, groups, d, N]
        let scale = 1.0 / sqrtf(Float(n))
        var attn = q.matmul(k.transposed(0, 1, 3, 2)) * scale   // [B, groups, d, d]
        attn = softmax(attn, axis: -1)
        var out = attn.matmul(v)                                // [B, groups, d, N]
        out = out.transposed(0, 3, 1, 2).reshaped([b, n, c])    // [B, N, groups, d] -> [B, N, C]
        return proj(out)
    }
}

// MARK: - Blocks

/// Depthwise 3×3 convolution (groups = channels) with a residual, run in NHWC.
final class NFKFlorence2DepthwiseConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(dim: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3,
                                    padding: 1, groups: dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) + x }
}

/// Spatial mixing block: depthwise conv + windowed attention, then depthwise conv + MLP, each pre-normed.
final class NFKFlorence2SpatialBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKFlorence2DepthwiseConv
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NFKFlorence2WindowAttention
    @ModuleInfo(key: "conv2") var conv2: NFKFlorence2DepthwiseConv
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: NFKFlorence2MLP

    init(dim: Int, heads: Int, windowSize: Int, mlpRatio: Double, qkvBias: Bool) {
        _conv1.wrappedValue = NFKFlorence2DepthwiseConv(dim: dim)
        _norm1.wrappedValue = LayerNorm(dimensions: dim)
        _attn.wrappedValue = NFKFlorence2WindowAttention(dim: dim, heads: heads, windowSize: windowSize, qkvBias: qkvBias)
        _conv2.wrappedValue = NFKFlorence2DepthwiseConv(dim: dim)
        _norm2.wrappedValue = LayerNorm(dimensions: dim)
        _ffn.wrappedValue = NFKFlorence2MLP(dim: dim, hidden: Int(Double(dim) * mlpRatio))
        super.init()
    }

    /// `x`: `[B, H, W, C]`. Returns `[B, H, W, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        var out = conv1(x)                                  // [B,H,W,C]
        var tokens = out.reshaped([b, h * w, c])
        tokens = tokens + attn(norm1(tokens).reshaped([b, h, w, c]))
        out = tokens.reshaped([b, h, w, c])
        out = conv2(out)
        tokens = out.reshaped([b, h * w, c])
        tokens = tokens + ffn(norm2(tokens))
        return tokens.reshaped([b, h, w, c])
    }
}

/// Channel mixing block: depthwise conv + grouped channel attention, then depthwise conv + MLP.
final class NFKFlorence2ChannelBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKFlorence2DepthwiseConv
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NFKFlorence2ChannelAttention
    @ModuleInfo(key: "conv2") var conv2: NFKFlorence2DepthwiseConv
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: NFKFlorence2MLP

    init(dim: Int, groups: Int, mlpRatio: Double, qkvBias: Bool) {
        _conv1.wrappedValue = NFKFlorence2DepthwiseConv(dim: dim)
        _norm1.wrappedValue = LayerNorm(dimensions: dim)
        _attn.wrappedValue = NFKFlorence2ChannelAttention(dim: dim, groups: groups, qkvBias: qkvBias)
        _conv2.wrappedValue = NFKFlorence2DepthwiseConv(dim: dim)
        _norm2.wrappedValue = LayerNorm(dimensions: dim)
        _ffn.wrappedValue = NFKFlorence2MLP(dim: dim, hidden: Int(Double(dim) * mlpRatio))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        var out = conv1(x)
        var tokens = out.reshaped([b, h * w, c])
        tokens = tokens + attn(norm1(tokens))
        out = tokens.reshaped([b, h, w, c])
        out = conv2(out)
        tokens = out.reshaped([b, h * w, c])
        tokens = tokens + ffn(norm2(tokens))
        return tokens.reshaped([b, h, w, c])
    }
}

/// One DaViT block: a spatial block followed by a channel block.
final class NFKFlorence2VisionBlock: Module {
    @ModuleInfo(key: "spatial") var spatial: NFKFlorence2SpatialBlock
    @ModuleInfo(key: "channel") var channel: NFKFlorence2ChannelBlock

    init(dim: Int, heads: Int, groups: Int, windowSize: Int, mlpRatio: Double, qkvBias: Bool) {
        _spatial.wrappedValue = NFKFlorence2SpatialBlock(dim: dim, heads: heads, windowSize: windowSize,
                                                         mlpRatio: mlpRatio, qkvBias: qkvBias)
        _channel.wrappedValue = NFKFlorence2ChannelBlock(dim: dim, groups: groups, mlpRatio: mlpRatio, qkvBias: qkvBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { channel(spatial(x)) }
}

// MARK: - Vision backbone

/// The DaViT vision tower: four stages, each a patch embedding followed by its blocks. `convs` and
/// `blocks` are sibling arrays so the module keys are `convs.N` and `blocks.N.M`, matching the
/// checkpoint after the `vision_tower.` prefix is stripped.
public final class NFKMLXFlorence2VisionNet: Module {
    @ModuleInfo(key: "convs") var convs: [NFKFlorence2ConvEmbed]
    @ModuleInfo(key: "blocks") var blocks: [[NFKFlorence2VisionBlock]]
    let config: NFKMLXFlorence2VisionConfiguration

    public init(_ config: NFKMLXFlorence2VisionConfiguration) {
        self.config = config
        var convList: [NFKFlorence2ConvEmbed] = []
        var blockStages: [[NFKFlorence2VisionBlock]] = []
        for stage in 0 ..< config.embedDim.count {
            let inC = stage == 0 ? config.inChannels : config.embedDim[stage - 1]
            convList.append(NFKFlorence2ConvEmbed(inChannels: inC, embedDim: config.embedDim[stage],
                                                  kernelSize: config.patchSize[stage],
                                                  stride: config.patchStride[stage],
                                                  padding: config.patchPadding[stage],
                                                  preNorm: config.patchPreNorm[stage]))
            var stageBlocks: [NFKFlorence2VisionBlock] = []
            for _ in 0 ..< config.depths[stage] {
                stageBlocks.append(NFKFlorence2VisionBlock(dim: config.embedDim[stage],
                                                           heads: config.numHeads[stage],
                                                           groups: config.numGroups[stage],
                                                           windowSize: config.windowSize,
                                                           mlpRatio: config.mlpRatio,
                                                           qkvBias: config.qkvBias))
            }
            blockStages.append(stageBlocks)
        }
        _convs.wrappedValue = convList
        _blocks.wrappedValue = blockStages
        super.init()
    }

    /// `pixelValues`: `[B, H, W, 3]` (NHWC, ImageNet-normalized). Returns `[B, H', W', C]`.
    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var out = pixelValues
        for stage in 0 ..< convs.count {
            out = convs[stage](out)
            for block in blocks[stage] {
                out = block(out)
            }
        }
        return out
    }
}

// MARK: - Multi-modal projector

/// Turns the vision grid `[B, H, W, C]` into a token sequence `[B, 1+H*W, projectionDim]`: it adds a
/// learned 2-D position embedding and a cosine temporal embedding, prepends the mean-pooled token, then
/// projects and normalizes. `image_projection` is a bare `[C, projectionDim]` parameter (applied as
/// `x @ W`), matching the checkpoint.
public final class NFKMLXFlorence2Projector: Module {
    @ModuleInfo(key: "row_embeddings") var rowEmbeddings: Embedding
    @ModuleInfo(key: "column_embeddings") var columnEmbeddings: Embedding
    @ParameterInfo(key: "image_projection") var imageProjection: MLXArray
    @ModuleInfo(key: "image_proj_norm") var imageProjNorm: LayerNorm
    let embedDim: Int

    public init(embedDim: Int, projectionDim: Int, maxPositionEmbeddings: Int) {
        self.embedDim = embedDim
        _rowEmbeddings.wrappedValue = Embedding(embeddingCount: maxPositionEmbeddings, dimensions: embedDim / 2)
        _columnEmbeddings.wrappedValue = Embedding(embeddingCount: maxPositionEmbeddings, dimensions: embedDim - embedDim / 2)
        _imageProjection.wrappedValue = MLXArray.zeros([embedDim, projectionDim])
        _imageProjNorm.wrappedValue = LayerNorm(dimensions: projectionDim)
        super.init()
    }

    /// `features`: `[B, H, W, C]`. Returns `[B, 1+H*W, projectionDim]`.
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        let (b, h, w, c) = (features.shape[0], features.shape[1], features.shape[2], features.shape[3])
        // Learned 2-D position: pos[h,w] = concat(column_emb[w], row_emb[h]).
        let colEmb = columnEmbeddings(MLXArray(Array(Int32(0) ..< Int32(w))))   // [W, C/2]
        let rowEmb = rowEmbeddings(MLXArray(Array(Int32(0) ..< Int32(h))))      // [H, C/2]
        let colGrid = broadcast(colEmb.reshaped([1, w, c - c / 2]), to: [h, w, c - c / 2])
        let rowGrid = broadcast(rowEmb.reshaped([h, 1, c / 2]), to: [h, w, c / 2])
        let pos = concatenated([colGrid, rowGrid], axis: -1).reshaped([1, h, w, c])
        var tokens = (features + pos).reshaped([b, h * w, c])
        // Cosine temporal embedding at position 0 is [0,1,0,1,...] (sin(0)=0 interleaved with cos(0)=1).
        tokens = tokens + NFKMLXFlorence2Projector.temporalRow0(c)
        // Prepend the mean-pooled token, then project and normalize.
        let pooled = tokens.mean(axis: 1, keepDims: true)                       // [B, 1, C]
        let seq = concatenated([pooled, tokens], axis: 1)                       // [B, 1+HW, C]
        let projected = seq.matmul(imageProjection)                            // [B, 1+HW, projectionDim]
        return imageProjNorm(projected)
    }

    /// The temporal embedding row for a single frame: alternating 0 and 1 across the channel axis.
    static func temporalRow0(_ c: Int) -> MLXArray {
        var v = [Float](repeating: 0, count: c)
        for i in stride(from: 1, to: c, by: 2) { v[i] = 1 }
        return MLXArray(v).reshaped([1, 1, c])
    }
}

// MARK: - Weight loading

enum NFKMLXFlorence2Weights {
    /// Maps a raw Florence-2 checkpoint key (original davit naming) to this port's vision-tower key,
    /// or nil to skip. The checkpoint wraps each block's convolution and attention in `.fn`/PreNorm
    /// containers, which this flattens: `spatial_block.window_attn.norm` -> `spatial.norm1`,
    /// `spatial_block.window_attn.fn.qkv` -> `spatial.attn.qkv`, `spatial_block.conv1.fn.dw` ->
    /// `spatial.conv1.conv`, `ffn.fn.net.fc1` -> `ffn.fc1`, and the same for `channel_block`.
    static func visionKey(_ key: String) -> String? {
        guard key.hasPrefix("vision_tower.") else { return nil }
        var k = String(key.dropFirst("vision_tower.".count))
        if k.hasPrefix("convs.") { return k }               // convs.N.proj / convs.N.norm already match
        guard k.hasPrefix("blocks.") else { return nil }
        k = k.replacingOccurrences(of: "spatial_block.", with: "spatial.")
        k = k.replacingOccurrences(of: "channel_block.", with: "channel.")
        k = k.replacingOccurrences(of: "window_attn.norm.", with: "norm1.")
        k = k.replacingOccurrences(of: "channel_attn.norm.", with: "norm1.")
        k = k.replacingOccurrences(of: "window_attn.fn.", with: "attn.")
        k = k.replacingOccurrences(of: "channel_attn.fn.", with: "attn.")
        k = k.replacingOccurrences(of: "ffn.norm.", with: "norm2.")
        k = k.replacingOccurrences(of: "ffn.fn.net.", with: "ffn.")
        k = k.replacingOccurrences(of: "conv1.fn.dw.", with: "conv1.conv.")
        k = k.replacingOccurrences(of: "conv2.fn.dw.", with: "conv2.conv.")
        return k
    }

    /// Maps a raw projector key to this port's projector key, or nil to skip.
    static func projectorKey(_ key: String) -> String? {
        switch true {
        case key == "image_projection": return "image_projection"
        case key.hasPrefix("image_proj_norm."): return key
        case key == "image_pos_embed.row_embeddings.weight": return "row_embeddings.weight"
        case key == "image_pos_embed.column_embeddings.weight": return "column_embeddings.weight"
        default: return nil
        }
    }

    /// Loads a released Florence-2 checkpoint into the vision tower and the projector in one pass,
    /// transposing 4-D convolution weights from PyTorch `[out, in, kH, kW]` to MLX `[out, kH, kW, in]`.
    /// The BART language model loads separately through the shared seq2seq loader.
    static func load(vision: NFKMLXFlorence2VisionNet, projector: NFKMLXFlorence2Projector, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var visionArrays: [(String, MLXArray)] = []
        var projectorArrays: [(String, MLXArray)] = []
        for (key, value) in checkpoint.arrays {
            if let mapped = visionKey(key) {
                let transposed = (checkpoint.needsConvTranspose && value.ndim == 4) ? value.transposed(0, 2, 3, 1) : value
                visionArrays.append((mapped, transposed.asType(.float32)))
            } else if let mapped = projectorKey(key) {
                projectorArrays.append((mapped, value.asType(.float32)))
            }
        }
        try NFKMLXWeights.apply(visionArrays, to: vision)
        try NFKMLXWeights.apply(projectorArrays, to: projector)
    }
}
