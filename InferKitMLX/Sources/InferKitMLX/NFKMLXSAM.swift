//
//  NFKMLXSAM.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Segment Anything: promptable segmentation. `NFKMLXSAMBackend` runs it as image → mask: a plate under
// `NFKInputImage` and a click point (the `NFKSAMPointKey` parameter, `[x, y]` in pixels; defaults to the
// image center) → a mask under `NFKOutputMask`. The pipeline is a ViT image encoder, a prompt encoder
// (point → sparse tokens), and a two-way-transformer mask decoder with a hypernetwork mask head.
// Tensors flow NHWC.
//
// Faithful to the reference pipeline end to end — verified at reference parity against the official
// `segment-anything` predictor (encoder seam cosine 0.999999; selected-mask binary agreement 99.7% on
// `sam_vit_b`) by `NFKMLXReferenceParityTests`. SAM 2's Hiera encoder / video memory are future variants.

/// SAM dimensions. Defaults are a compact configuration; real SAM uses image 1024, embed 256.
public struct NFKMLXSAMConfiguration: Sendable {
    public var imageSize: Int = 256
    public var patchSize: Int = 16
    public var encoderEmbed: Int = 256
    public var encoderDepth: Int = 4
    public var encoderHeads: Int = 4
    public var embedDim: Int = 256                 // transformer / mask-embedding width
    public var decoderDepth: Int = 2
    public var decoderHeads: Int = 8
    public var numMaskTokens: Int = 4              // 3 multimask + 1
    /// The window size for windowed attention (blocks outside `globalAttnIndexes`).
    public var windowSize: Int = 14
    /// The encoder blocks that use global attention (all others are windowed).
    public var globalAttnIndexes: [Int] = [2, 5, 8, 11]
    public init() {}
    var grid: Int { imageSize / patchSize }

    /// The released `sam_vit_b` geometry.
    public static var vitB: NFKMLXSAMConfiguration {
        var configuration = NFKMLXSAMConfiguration()
        configuration.imageSize = 1024
        configuration.encoderEmbed = 768
        configuration.encoderDepth = 12
        configuration.encoderHeads = 12
        return configuration
    }
}

/// Random Fourier positional encoding of 2-D coordinates in `0...1`.
final class NFKSAMPositionEncoding: Module {
    @ModuleInfo(key: "gaussian") var gaussian: MLXArray     // [2, dim/2]

    init(dim: Int) {
        _gaussian.wrappedValue = MLXArray.zeros([2, dim / 2])
    }

    /// `coords` `[..., 2]` in `0...1` → `[..., dim]`.
    func callAsFunction(_ coords: MLXArray) -> MLXArray {
        let scaled = (2 * coords - 1).matmul(gaussian) * (2 * Float.pi)
        return concatenated([sin(scaled), cos(scaled)], axis: -1)
    }

    /// The positional encoding of the `h × w` grid, flattened to `[1, h*w, dim]`.
    func grid(_ h: Int, _ w: Int) -> MLXArray {
        var coords = [Float](repeating: 0, count: h * w * 2)
        for y in 0 ..< h {
            for x in 0 ..< w {
                coords[(y * w + x) * 2] = (Float(x) + 0.5) / Float(w)
                coords[(y * w + x) * 2 + 1] = (Float(y) + 0.5) / Float(h)
            }
        }
        let array = coords.withUnsafeBufferPointer { MLXArray($0, [h * w, 2]) }
        return callAsFunction(array).reshaped([1, h * w, gaussian.shape[1] * 2])
    }
}

/// SAM's windowed multi-head attention with decomposed relative-position embeddings.
final class NFKSAMImageAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "rel_pos_h") var relPosH: MLXArray
    @ModuleInfo(key: "rel_pos_w") var relPosW: MLXArray
    private let heads: Int

    init(dim: Int, heads: Int, inputSize: Int) {
        self.heads = heads
        _qkv.wrappedValue = Linear(dim, dim * 3)
        _proj.wrappedValue = Linear(dim, dim)
        _relPosH.wrappedValue = MLXArray.zeros([2 * inputSize - 1, dim / heads])
        _relPosW.wrappedValue = MLXArray.zeros([2 * inputSize - 1, dim / heads])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let headDim = c / heads
        let scale = 1.0 / sqrtf(Float(headDim))

        let fused = qkv(x).reshaped([batch, h * w, 3, heads, headDim]).transposed(2, 0, 3, 1, 4)
            .reshaped([3, batch * heads, h * w, headDim])
        let q = fused[0], k = fused[1], v = fused[2]

        var attn = (q * scale).matmul(k.transposed(0, 2, 1))    // [batch*heads, h*w, h*w]
        attn = Self.addRelativePosition(attn, q: q, relPosH: relPosH, relPosW: relPosW, height: h, width: w)
        attn = softmax(attn, axis: -1)
        let out = attn.matmul(v).reshaped([batch, heads, h, w, headDim]).transposed(0, 2, 3, 1, 4).reshaped([batch, h, w, c])
        return proj(out)
    }

    /// Gathers `relPos` `[2·size-1, dim]` into `[size, size, dim]` by relative coordinate `i - j + size - 1`.
    static func relative(_ size: Int, _ relPos: MLXArray) -> MLXArray {
        let dim = relPos.shape[1]
        var index = [Int32](repeating: 0, count: size * size)
        for i in 0 ..< size {
            for j in 0 ..< size {
                index[i * size + j] = Int32(i - j + size - 1)
            }
        }
        return relPos.take(MLXArray(index), axis: 0).reshaped([size, size, dim])
    }

    /// Adds decomposed relative-position bias to the attention logits.
    static func addRelativePosition(_ attn: MLXArray, q: MLXArray, relPosH: MLXArray, relPosW: MLXArray,
                                    height: Int, width: Int) -> MLXArray {
        let batch = q.shape[0], dim = q.shape[2]
        let rh = relative(height, relPosH)                      // [h, h, dim]
        let rw = relative(width, relPosW)                       // [w, w, dim]
        let rq = q.reshaped([batch, height, width, dim])

        // relH[b,i,j,p] = Σc rq[b,i,j,c] · rh[i,p,c]; relW[b,i,j,p] = Σc rq[b,i,j,c] · rw[j,p,c].
        let relH = rq.transposed(1, 0, 2, 3)                    // [h, batch, w, dim]
            .matmul(rh.transposed(0, 2, 1).reshaped([height, 1, dim, height]))   // [h, batch, w, h]
            .transposed(1, 0, 2, 3)                             // [batch, h, w, h]
        let relW = rq.transposed(2, 0, 1, 3)                    // [w, batch, h, dim]
            .matmul(rw.transposed(0, 2, 1).reshaped([width, 1, dim, width]))     // [w, batch, h, w]
            .transposed(1, 2, 0, 3)                             // [batch, h, w, w]

        let biased = attn.reshaped([batch, height, width, height, width])
            + relH[0..., 0..., 0..., 0..., .newAxis]
            + relW[0..., 0..., 0..., .newAxis, 0...]
        return biased.reshaped([batch, height * width, height * width])
    }
}

/// A SAM ViT block: norm → (windowed) attention → residual, norm → MLP → residual.
final class NFKSAMViTBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NFKSAMImageAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear

    private let windowSize: Int

    init(dim: Int, heads: Int, windowSize: Int, gridSize: Int) {
        self.windowSize = windowSize
        _norm1.wrappedValue = LayerNorm(dimensions: dim)
        _attn.wrappedValue = NFKSAMImageAttention(dim: dim, heads: heads, inputSize: windowSize > 0 ? windowSize : gridSize)
        _norm2.wrappedValue = LayerNorm(dimensions: dim)
        _lin1.wrappedValue = Linear(dim, dim * 4)
        _lin2.wrappedValue = Linear(dim * 4, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (h, w) = (x.shape[1], x.shape[2])
        var attended = norm1(x)
        var padded = (h, w)
        if windowSize > 0 {
            (attended, padded) = Self.partition(attended, windowSize)
        }
        attended = attn(attended)
        if windowSize > 0 {
            attended = Self.unpartition(attended, windowSize, padded: padded, original: (h, w))
        }
        let residual = x + attended
        return residual + lin2(gelu(lin1(norm2(residual))))
    }

    /// Partitions `[B, H, W, C]` into `[B·nH·nW, ws, ws, C]`, zero-padding to a multiple of `ws`.
    static func partition(_ x: MLXArray, _ ws: Int) -> (MLXArray, (Int, Int)) {
        let (b, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let padH = (ws - h % ws) % ws, padW = (ws - w % ws) % ws
        let xp = padH > 0 || padW > 0
            ? MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, padH)), IntOrPair((0, padW)), IntOrPair((0, 0))], mode: .constant)
            : x
        let (hp, wp) = (h + padH, w + padW)
        let windows = xp.reshaped([b, hp / ws, ws, wp / ws, ws, c]).transposed(0, 1, 3, 2, 4, 5)
            .reshaped([b * (hp / ws) * (wp / ws), ws, ws, c])
        return (windows, (hp, wp))
    }

    static func unpartition(_ windows: MLXArray, _ ws: Int, padded: (Int, Int), original: (Int, Int)) -> MLXArray {
        let (hp, wp) = padded
        let c = windows.shape[3]
        let batch = windows.shape[0] / ((hp / ws) * (wp / ws))
        let x = windows.reshaped([batch, hp / ws, wp / ws, ws, ws, c]).transposed(0, 1, 3, 2, 4, 5).reshaped([batch, hp, wp, c])
        return x[0..., 0 ..< original.0, 0 ..< original.1, 0...]
    }
}

/// The ViT image encoder with windowed/global attention and SAM's convolutional neck.
final class NFKSAMImageEncoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: Conv2d
    @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [NFKSAMViTBlock]
    @ModuleInfo(key: "neck_conv1") var neckConv1: Conv2d
    @ModuleInfo(key: "neck_ln1") var neckLN1: LayerNorm
    @ModuleInfo(key: "neck_conv2") var neckConv2: Conv2d
    @ModuleInfo(key: "neck_ln2") var neckLN2: LayerNorm

    init(_ c: NFKMLXSAMConfiguration) {
        _patchEmbed.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.encoderEmbed,
                                          kernelSize: IntOrPair(c.patchSize), stride: IntOrPair(c.patchSize))
        _posEmbed.wrappedValue = MLXArray.zeros([1, c.grid, c.grid, c.encoderEmbed])
        let global = Set(c.globalAttnIndexes)
        _blocks.wrappedValue = (0 ..< c.encoderDepth).map { index in
            NFKSAMViTBlock(dim: c.encoderEmbed, heads: c.encoderHeads,
                           windowSize: global.contains(index) ? 0 : c.windowSize, gridSize: c.grid)
        }
        // The reference's neck convolutions carry no bias (`nn.Conv2d(..., bias=False)`).
        _neckConv1.wrappedValue = Conv2d(inputChannels: c.encoderEmbed, outputChannels: c.embedDim,
                                         kernelSize: 1, bias: false)
        _neckLN1.wrappedValue = LayerNorm(dimensions: c.embedDim)
        _neckConv2.wrappedValue = Conv2d(inputChannels: c.embedDim, outputChannels: c.embedDim,
                                         kernelSize: 3, padding: 1, bias: false)
        _neckLN2.wrappedValue = LayerNorm(dimensions: c.embedDim)
    }

    /// `[1, imageSize, imageSize, 3]` → `[1, grid, grid, embedDim]`.
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        var x = patchEmbed(image) + posEmbed                    // [1, grid, grid, encoderEmbed]
        for block in blocks { x = block(x) }
        return neckLN2(neckConv2(neckLN1(neckConv1(x))))
    }
}

/// The prompt encoder: a click point → sparse tokens; a default dense (no-mask) embedding.
final class NFKSAMPromptEncoder: Module {
    @ModuleInfo(key: "position_encoding") var positionEncoding: NFKSAMPositionEncoding
    @ModuleInfo(key: "point_embeddings") var pointEmbeddings: [MLXArray]   // [negative, positive]
    @ModuleInfo(key: "not_a_point_embed") var notAPointEmbed: MLXArray
    @ModuleInfo(key: "no_mask_embed") var noMaskEmbed: MLXArray

    init(_ c: NFKMLXSAMConfiguration) {
        _positionEncoding.wrappedValue = NFKSAMPositionEncoding(dim: c.embedDim)
        _pointEmbeddings.wrappedValue = [MLXArray.zeros([1, c.embedDim]), MLXArray.zeros([1, c.embedDim])]
        _notAPointEmbed.wrappedValue = MLXArray.zeros([1, c.embedDim])
        _noMaskEmbed.wrappedValue = MLXArray.zeros([1, c.embedDim])
    }

    /// A point at normalized `(x, y)` with `positive` label → the sparse tokens `[1, 2, embedDim]`.
    /// The reference pads every point-only prompt with a second token — `not_a_point_embed`, its
    /// positional encoding zeroed — and the decoder is trained on that pair, so a lone click token
    /// shifts the attention it sees.
    func sparse(pointX: Float, pointY: Float, positive: Bool) -> MLXArray {
        let coords = [pointX, pointY].withUnsafeBufferPointer { MLXArray($0, [1, 1, 2]) }
        let encoded = positionEncoding(coords)                  // [1, 1, embedDim]
        let click = encoded + pointEmbeddings[positive ? 1 : 0]
        let padding = notAPointEmbed.reshaped([1, 1, notAPointEmbed.shape[1]])
        return concatenated([click, padding], axis: 1)
    }

    /// The default dense embedding `[1, grid, grid, embedDim]` for "no mask input".
    func dense(grid: Int) -> MLXArray {
        broadcast(noMaskEmbed.reshaped([1, 1, 1, noMaskEmbed.shape[1]]),
                  to: [1, grid, grid, noMaskEmbed.shape[1]])
    }
}

/// Downsampled multi-head attention used in the mask decoder's two-way transformer.
final class NFKSAMAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    private let heads: Int

    init(dim: Int, heads: Int, downsample: Int = 1) {
        self.heads = heads
        let inner = dim / downsample
        _qProj.wrappedValue = Linear(dim, inner)
        _kProj.wrappedValue = Linear(dim, inner)
        _vProj.wrappedValue = Linear(dim, inner)
        _outProj.wrappedValue = Linear(inner, dim)
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let batch = q.shape[0]
        let inner = qProj.weight.shape[0]
        let headDim = inner / heads
        func split(_ x: MLXArray, _ proj: Linear) -> MLXArray {
            proj(x).reshaped([batch, x.shape[1], heads, headDim]).transposed(0, 2, 1, 3)
        }
        let qh = split(q, qProj), kh = split(k, kProj), vh = split(v, vProj)
        let scores = softmax(qh.matmul(kh.transposed(0, 1, 3, 2)) / sqrtf(Float(headDim)), axis: -1)
        let context = scores.matmul(vh).transposed(0, 2, 1, 3).reshaped([batch, q.shape[1], inner])
        return outProj(context)
    }
}

/// One two-way block: token self-attention, token↔image cross-attention, and an MLP.
final class NFKSAMTwoWayBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKSAMAttention
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "cross_token_image") var crossTokenImage: NFKSAMAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKSAMMLP
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "norm4") var norm4: LayerNorm
    @ModuleInfo(key: "cross_image_token") var crossImageToken: NFKSAMAttention

    private let skipFirstLayerPE: Bool

    init(dim: Int, heads: Int, skipFirstLayerPE: Bool = false) {
        self.skipFirstLayerPE = skipFirstLayerPE
        _selfAttn.wrappedValue = NFKSAMAttention(dim: dim, heads: heads)
        _norm1.wrappedValue = LayerNorm(dimensions: dim)
        _crossTokenImage.wrappedValue = NFKSAMAttention(dim: dim, heads: heads, downsample: 2)
        _norm2.wrappedValue = LayerNorm(dimensions: dim)
        // The reference block's MLP widens to 2048 (`mlp_dim`), eight times the embedding.
        _mlp.wrappedValue = NFKSAMMLP(dim: dim, hidden: dim * 8, out: dim, layers: 2)
        _norm3.wrappedValue = LayerNorm(dimensions: dim)
        _norm4.wrappedValue = LayerNorm(dimensions: dim)
        _crossImageToken.wrappedValue = NFKSAMAttention(dim: dim, heads: heads, downsample: 2)
    }

    func callAsFunction(_ queries: MLXArray, _ keys: MLXArray, queryPE: MLXArray, keyPE: MLXArray) -> (MLXArray, MLXArray) {
        // The reference's first layer attends without positional terms and REPLACES the queries with the
        // attention output (`skip_first_layer_pe`) — no residual. Every later layer adds the PE and a
        // residual. Getting this wrong drifts every downstream mask just enough to flip near-tied IoU votes.
        var out: MLXArray
        if skipFirstLayerPE {
            out = selfAttn(queries, queries, queries)
        } else {
            let q = queries + queryPE
            out = queries + selfAttn(q, q, queries)
        }
        out = norm1(out)
        var q = out + queryPE
        var k = keys + keyPE
        out = norm2(out + crossTokenImage(q, k, keys))
        out = norm3(out + mlp(out))
        q = out + queryPE
        k = keys + keyPE
        let updatedKeys = norm4(keys + crossImageToken(k, q, out))
        return (out, updatedKeys)
    }
}

/// A small multi-layer perceptron (ReLU between layers).
final class NFKSAMMLP: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(dim: Int, hidden: Int, out: Int, layers count: Int) {
        var linears = [Linear]()
        var input = dim
        for i in 0 ..< count {
            let output = i == count - 1 ? out : hidden
            linears.append(Linear(input, output))
            input = output
        }
        _layers.wrappedValue = linears
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for (index, layer) in layers.enumerated() {
            y = layer(y)
            if index < layers.count - 1 { y = relu(y) }
        }
        return y
    }
}

/// The mask decoder: a two-way transformer, mask/IoU tokens, upscaling, and a hypernetwork mask head.
final class NFKSAMMaskDecoder: Module {
    @ModuleInfo(key: "iou_token") var iouToken: MLXArray
    @ModuleInfo(key: "mask_tokens") var maskTokens: MLXArray
    @ModuleInfo(key: "layers") var layers: [NFKSAMTwoWayBlock]
    @ModuleInfo(key: "final_attn") var finalAttn: NFKSAMAttention
    @ModuleInfo(key: "norm_final") var normFinal: LayerNorm
    @ModuleInfo(key: "up1") var up1: ConvTransposed2d
    @ModuleInfo(key: "up_ln") var upLN: LayerNorm
    @ModuleInfo(key: "up2") var up2: ConvTransposed2d
    @ModuleInfo(key: "hyper") var hyper: [NFKSAMMLP]
    @ModuleInfo(key: "iou_head") var iouHead: NFKSAMMLP

    private let numMaskTokens: Int

    init(_ c: NFKMLXSAMConfiguration) {
        numMaskTokens = c.numMaskTokens
        _iouToken.wrappedValue = MLXArray.zeros([1, c.embedDim])
        _maskTokens.wrappedValue = MLXArray.zeros([c.numMaskTokens, c.embedDim])
        _layers.wrappedValue = (0 ..< c.decoderDepth).map { index in
            NFKSAMTwoWayBlock(dim: c.embedDim, heads: c.decoderHeads, skipFirstLayerPE: index == 0)
        }
        _finalAttn.wrappedValue = NFKSAMAttention(dim: c.embedDim, heads: c.decoderHeads, downsample: 2)
        _normFinal.wrappedValue = LayerNorm(dimensions: c.embedDim)
        _up1.wrappedValue = ConvTransposed2d(inputChannels: c.embedDim, outputChannels: c.embedDim / 4, kernelSize: 2, stride: 2)
        _upLN.wrappedValue = LayerNorm(dimensions: c.embedDim / 4)
        _up2.wrappedValue = ConvTransposed2d(inputChannels: c.embedDim / 4, outputChannels: c.embedDim / 8, kernelSize: 2, stride: 2)
        _hyper.wrappedValue = (0 ..< c.numMaskTokens).map { _ in NFKSAMMLP(dim: c.embedDim, hidden: c.embedDim, out: c.embedDim / 8, layers: 3) }
        _iouHead.wrappedValue = NFKSAMMLP(dim: c.embedDim, hidden: c.embedDim, out: c.numMaskTokens, layers: 3)
    }

    /// Returns masks `[1, 4·grid, 4·grid, numMaskTokens]` and IoU scores `[1, numMaskTokens]`.
    func callAsFunction(_ image: MLXArray, imagePE: MLXArray, sparse: MLXArray) -> (masks: MLXArray, iou: MLXArray) {
        let (grid, dim) = (image.shape[1], image.shape[3])
        let tokens = concatenated([iouToken.reshaped([1, 1, dim]), maskTokens.reshaped([1, numMaskTokens, dim]), sparse], axis: 1)

        var queries = tokens
        var keys = image.reshaped([1, grid * grid, dim])
        for layer in layers {
            (queries, keys) = layer(queries, keys, queryPE: tokens, keyPE: imagePE)
        }
        let q = queries + tokens
        let k = keys + imagePE
        queries = normFinal(queries + finalAttn(q, k, keys))

        let iouOut = iouHead(queries[0..., 0])
        let maskTokenOut = queries[0..., 1 ..< (1 + numMaskTokens)]

        var upscaled = gelu(upLN(up1(keys.reshaped([1, grid, grid, dim]))))
        upscaled = gelu(up2(upscaled))                          // [1, 4·grid, 4·grid, dim/8]
        let (uh, uw, uc) = (upscaled.shape[1], upscaled.shape[2], upscaled.shape[3])

        var perMask = [MLXArray]()
        for i in 0 ..< numMaskTokens {
            perMask.append(hyper[i](maskTokenOut[0..., i]).reshaped([1, 1, uc]))
        }
        let weights = concatenated(perMask, axis: 1)            // [1, numMaskTokens, dim/8]
        let flatUpscaled = upscaled.reshaped([1, uh * uw, uc])
        let masks = weights.matmul(flatUpscaled.transposed(0, 2, 1))   // [1, numMaskTokens, uh*uw]
        return (masks.reshaped([1, numMaskTokens, uh, uw]).transposed(0, 2, 3, 1), iouOut)
    }
}

/// The full SAM model.
final class NFKMLXSAMNet: Module {
    @ModuleInfo(key: "image_encoder") var imageEncoder: NFKSAMImageEncoder
    @ModuleInfo(key: "prompt_encoder") var promptEncoder: NFKSAMPromptEncoder
    @ModuleInfo(key: "mask_decoder") var maskDecoder: NFKSAMMaskDecoder

    let configuration: NFKMLXSAMConfiguration

    init(_ configuration: NFKMLXSAMConfiguration) {
        self.configuration = configuration
        _imageEncoder.wrappedValue = NFKSAMImageEncoder(configuration)
        _promptEncoder.wrappedValue = NFKSAMPromptEncoder(configuration)
        _maskDecoder.wrappedValue = NFKSAMMaskDecoder(configuration)
    }

    /// Segments `image` `[1, imageSize, imageSize, 3]` around the normalized point `(x, y)`; returns a
    /// single-channel mask `[1, 4·grid, 4·grid, 1]` in `0...1` (the highest-IoU of the multimask outputs).
    func segment(_ image: MLXArray, pointX: Float, pointY: Float) -> MLXArray {
        let embedding = imageEncoder(image)
        let grid = embedding.shape[1]
        let imagePE = promptEncoder.positionEncoding.grid(grid, grid)
        let sparse = promptEncoder.sparse(pointX: pointX, pointY: pointY, positive: true)
        let dense = promptEncoder.dense(grid: grid)
        let (masks, iou) = maskDecoder(embedding + dense, imagePE: imagePE, sparse: sparse)

        // The reference's multimask output uses mask tokens 1...3 and picks the best by predicted IoU;
        // token 0 is the single-mask alternative and stays out of the comparison.
        let best = 1 + iou[0, 1...].argMax().item(Int.self)
        let mask = masks[0..., 0..., 0..., best ..< (best + 1)]
        return sigmoid(mask)
    }
}

/// The point-prompt parameter key: `[x, y]` in pixels.
public let NFKSAMPointKey = "samPoint"

/// The SAM geometry to build. `compact` is the small default that runs with random weights; `vitB`
/// matches the released `sam_vit_b` checkpoint.
@objc(NFKMLXSAMVariant)
public enum NFKMLXSAMVariant: Int {
    case compact
    case vitB
}

/// Segment Anything as an InferKit backend, and its registration.
@objc(NFKMLXSAM)

public final class NFKMLXSAM: NSObject {

    @objc public static let modelName = "sam"

    static func makeNet(_ configuration: NFKMLXSAMConfiguration = NFKMLXSAMConfiguration()) -> NFKMLXSAMNet {
        NFKMLXSAMNet(configuration)
    }

    static func configuration(for variant: NFKMLXSAMVariant) -> NFKMLXSAMConfiguration {
        switch variant {
        case .compact: return NFKMLXSAMConfiguration()
        case .vitB: return .vitB
        }
    }

    /// Builds a SAM backend at a chosen geometry. The released `sam_vit_b` checkpoint only fits `.vitB`;
    /// `.compact` is the small default that runs with random weights.
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXSAMVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXSAMNet(configuration(for: variant))
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXSAMHolder(net)
        var configuration = NFKMattingConfiguration()
        configuration.emitsMatte = true
        return NFKMLXMattingBackend(identifier: modelName, configuration: configuration) { plate, _, request in
            let point = NFKMLXSAMHolder.point(from: request, width: plate.shape[1], height: plate.shape[0])
            return holder.segment(plate, point: point)
        }
    }

    /// Builds a SAM segmentation backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true). The request supplies the plate
    /// under `NFKInputImage` and an optional point under `NFKSAMPointKey`.
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .compact, weightsURL: weightsURL)
    }

    /// Downloads the checkpoint from Hugging Face, then builds at a chosen geometry — no registry
    /// required. The released `sam_vit_b` checkpoint needs `.vitB`.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXSAMVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url)
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .compact, repo: repo, weightsPath: weightsPath, revision: revision,
                    cacheDirectoryURL: cacheDirectoryURL)
    }

    /// The asynchronous form of the variant download factory: downloads on a background queue, then
    /// builds and delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXSAMVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        backend(variant: .compact, repo: repo, weightsPath: weightsPath, revision: revision,
                cacheDirectoryURL: cacheDirectoryURL, completionHandler: completionHandler)
    }

    /// Registers SAM (`sam`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    static func loadWeights(into net: NFKMLXSAMNet, from url: URL,
                            remap: (String) -> String = remapReferenceKey) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value -> (String, MLXArray) in
            let name = remap(key)
            guard checkpoint.needsConvTranspose, value.ndim == 4 else { return (name, value) }
            // The ViT's positional embedding is already channels-last `[1, grid, grid, embed]`, so the
            // convolution transpose would scramble it.
            if name.hasSuffix("pos_embed") { return (name, value) }
            // The mask decoder upscales with transposed convolutions, which PyTorch stores as
            // `[in, out, kH, kW]` — a different axis order from a forward convolution's `[out, in, kH, kW]`.
            if name.hasSuffix("up1.weight") || name.hasSuffix("up2.weight") {
                return (name, value.transposed(1, 2, 3, 0))
            }
            return (name, value.transposed(0, 2, 3, 1))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Maps a released `segment_anything` checkpoint's names onto this module's. The reference nests the
    /// encoder MLP under `mlp`, keeps the neck and output upscaling as `nn.Sequential` (so their layers are
    /// positional), routes the mask decoder's attention through a `transformer` submodule, and stores the
    /// prompt embeddings as `nn.Embedding` (hence the trailing `.weight`). The layouts agree; only the
    /// names differ.
    static func remapReferenceKey(_ key: String) -> String {
        var name = key

        // Image encoder.
        name = name.replacingOccurrences(of: "image_encoder.patch_embed.proj.", with: "image_encoder.patch_embed.")
        if name.hasPrefix("image_encoder.") {
            // The encoder inlines its MLP; the mask decoder keeps a nested one, so this must not apply there.
            name = name.replacingOccurrences(of: ".mlp.lin", with: ".lin")
        }
        for (index, replacement) in [("0", "neck_conv1"), ("1", "neck_ln1"), ("2", "neck_conv2"), ("3", "neck_ln2")] {
            name = name.replacingOccurrences(of: "image_encoder.neck.\(index).",
                                             with: "image_encoder.\(replacement).")
        }

        // Mask decoder: the two-way transformer is inlined here, so drop its level and rename the
        // cross-attention directions.
        name = name.replacingOccurrences(of: "mask_decoder.transformer.final_attn_token_to_image.",
                                         with: "mask_decoder.final_attn.")
        name = name.replacingOccurrences(of: "mask_decoder.transformer.norm_final_attn.",
                                         with: "mask_decoder.norm_final.")
        name = name.replacingOccurrences(of: "mask_decoder.transformer.layers.", with: "mask_decoder.layers.")
        name = name.replacingOccurrences(of: ".cross_attn_token_to_image.", with: ".cross_token_image.")
        name = name.replacingOccurrences(of: ".cross_attn_image_to_token.", with: ".cross_image_token.")
        name = name.replacingOccurrences(of: ".mlp.lin1.", with: ".mlp.layers.0.")
        name = name.replacingOccurrences(of: ".mlp.lin2.", with: ".mlp.layers.1.")
        for (index, replacement) in [("0", "up1"), ("1", "up_ln"), ("3", "up2")] {
            name = name.replacingOccurrences(of: "mask_decoder.output_upscaling.\(index).",
                                             with: "mask_decoder.\(replacement).")
        }
        name = name.replacingOccurrences(of: "mask_decoder.output_hypernetworks_mlps.", with: "mask_decoder.hyper.")
        name = name.replacingOccurrences(of: "mask_decoder.iou_prediction_head.", with: "mask_decoder.iou_head.")
        name = name.replacingOccurrences(of: "mask_decoder.iou_token.weight", with: "mask_decoder.iou_token")
        name = name.replacingOccurrences(of: "mask_decoder.mask_tokens.weight", with: "mask_decoder.mask_tokens")

        // Prompt encoder: embeddings rather than raw parameters in the reference.
        name = name.replacingOccurrences(of: "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
                                         with: "prompt_encoder.position_encoding.gaussian")
        name = name.replacingOccurrences(of: "prompt_encoder.no_mask_embed.weight",
                                         with: "prompt_encoder.no_mask_embed")
        name = name.replacingOccurrences(of: "prompt_encoder.not_a_point_embed.weight",
                                         with: "prompt_encoder.not_a_point_embed")
        if name.hasPrefix("prompt_encoder.point_embeddings."), name.hasSuffix(".weight") {
            name = String(name.dropLast(".weight".count))
        }
        return name
    }
}

private final class NFKMLXSAMHolder: @unchecked Sendable {
    let net: NFKMLXSAMNet
    init(_ net: NFKMLXSAMNet) { self.net = net }

    /// Runs the plate through SAM at the model's input size and returns `[H, W, 4]`: the plate carried
    /// as straight foreground with the resized mask as alpha (the matting backend's contract).
    /// The click point in `0...1`, read from the request's `NFKSAMPointKey` parameter (`[x, y]` in
    /// pixels for a `width` × `height` plate). Defaults to the image center when absent.
    static func point(from request: NFKInferenceRequest, width: Int, height: Int) -> (x: Float, y: Float) {
        guard let value = request.parameter(forKey: NFKSAMPointKey) as? [NSNumber], value.count >= 2,
              width > 0, height > 0 else {
            return (0.5, 0.5)
        }
        return (min(max(value[0].floatValue / Float(width), 0), 1),
                min(max(value[1].floatValue / Float(height), 0), 1))
    }

    func segment(_ plate: MLXArray, point: (x: Float, y: Float)) -> MLXArray {
        let (h, w) = (plate.shape[0], plate.shape[1])
        let size = net.configuration.imageSize
        let resized = NFKMLXResample.resizeBilinear(plate.reshaped([1, h, w, 3]), height: size, width: size)
        // SAM's `preprocess` normalizes with ImageNet statistics on the 0...255 scale; the bridge hands
        // over 0...1, so rescale first. Feeding the raw plate leaves the encoder far outside the
        // distribution it was trained on and washes the mask out.
        let mean = MLXArray([Float(123.675), 116.28, 103.53])
        let standardDeviation = MLXArray([Float(58.395), 57.12, 57.375])
        let normalized = (resized * 255 - mean) / standardDeviation
        let mask = net.segment(normalized, pointX: point.x, pointY: point.y)   // [1, m, m, 1]
        let alpha = NFKMLXResample.resizeBilinear(mask, height: h, width: w).reshaped([h, w, 1])
        return concatenated([plate, alpha], axis: 2)
    }
}
