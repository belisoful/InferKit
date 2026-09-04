//
//  NFKMLXZImageTransformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// The Z-Image S3-DiT (`ZImageTransformer2DModel`, Alibaba Tongyi): the denoising transformer of a 6B
// text-to-image model, and the third DiT family here beside the SD UNet and the LTX transformer. It is
// SINGLE-STREAM — the image latent tokens and the caption tokens are concatenated into one sequence and
// every layer's self-attention runs over the join, rather than a separate cross-attention branch. Three
// stages: a `noise_refiner` (modulated blocks over the image tokens alone), a `context_refiner`
// (un-modulated blocks over the caption tokens alone), then the unified `layers` over the concatenation.
//
// Base text-to-image feeds only the image latent (`x`) and the Qwen3 caption features (`cap_feats`); the
// SigLIP visual-semantic tokens are the edit variant's input, not this path. For parity the caption
// features are supplied directly, so the DiT is validated in isolation (no Qwen3), as the LTX DiT is.

/// Z-Image DiT geometry. Defaults are the released 6B model.
public struct NFKMLXZImageConfiguration: Sendable {
    public var inChannels: Int
    public var dim: Int
    public var layers: Int
    public var refinerLayers: Int
    public var heads: Int
    public var normEps: Float
    public var captionFeatureDim: Int
    public var ropeTheta: Float
    public var timestepScale: Float
    public var patchSize: Int
    public var framePatchSize: Int
    public var axesDims: [Int]

    public init(inChannels: Int = 16, dim: Int = 3840, layers: Int = 30, refinerLayers: Int = 2,
                heads: Int = 30, normEps: Float = 1e-5, captionFeatureDim: Int = 2560,
                ropeTheta: Float = 256.0, timestepScale: Float = 1000.0, patchSize: Int = 2,
                framePatchSize: Int = 1, axesDims: [Int] = [32, 48, 48]) {
        self.inChannels = inChannels
        self.dim = dim
        self.layers = layers
        self.refinerLayers = refinerLayers
        self.heads = heads
        self.normEps = normEps
        self.captionFeatureDim = captionFeatureDim
        self.ropeTheta = ropeTheta
        self.timestepScale = timestepScale
        self.patchSize = patchSize
        self.framePatchSize = framePatchSize
        self.axesDims = axesDims
    }

    public static let base = NFKMLXZImageConfiguration()

    public static let tiny = NFKMLXZImageConfiguration(
        inChannels: 4, dim: 32, layers: 2, refinerLayers: 1, heads: 2, captionFeatureDim: 24,
        axesDims: [4, 6, 6])

    var headDim: Int { dim / heads }
    var feedForwardDim: Int { Int(Double(dim) / 3.0 * 8.0) }
    var patchKey: String { "\(patchSize)-\(framePatchSize)" }
    /// The `f_patch·patch·patch·in_channels` a patch token carries.
    var patchInputDim: Int { framePatchSize * patchSize * patchSize * inChannels }
}

/// Rotary position built from a 3-axis coordinate grid, the reference's `view_as_complex` convention:
/// each axis contributes `axisDim/2` complex angles, and the query pairs adjacent channels.
struct NFKZImageRope {
    let perAxisFreqs: [MLXArray]                                            // axis a: [axisDim/2]
    let theta: Float

    init(axesDims: [Int], theta: Float) {
        self.theta = theta
        self.perAxisFreqs = axesDims.map { d in
            let k = MLXArray(stride(from: 0, to: d, by: 2).map { Float($0) })
            return pow(MLXArray(theta), -(k / Float(d)))                    // theta^(-(2k)/d)
        }
    }

    /// Position ids `[N, 3]` (float) → per-position `(cos, sin)` tables `[N, sum(axisDim/2)]`.
    func table(positions: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        var angles: [MLXArray] = []
        for axis in 0 ..< perAxisFreqs.count {
            let coord = positions[0..., axis].reshaped([-1, 1])             // [N, 1]
            angles.append(coord * perAxisFreqs[axis].reshaped([1, -1]))     // [N, axisDim/2]
        }
        let a = concatenated(angles, axis: -1)                             // [N, 64]
        return (cos(a), sin(a))
    }
}

/// Applies the complex rotary to `x` `[N, heads, headDim]` given `(cos, sin)` `[N, 64]`.
func zImageApplyRope(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
    let n = x.dim(0), heads = x.dim(1), headDim = x.dim(2)
    let pairs = x.reshaped([n, heads, headDim / 2, 2])
    let xr = pairs[0..., 0..., 0..., 0]                                     // [N, heads, 64]
    let xi = pairs[0..., 0..., 0..., 1]
    let cc = c.reshaped([n, 1, headDim / 2])
    let ss = s.reshaped([n, 1, headDim / 2])
    let outR = xr * cc - xi * ss
    let outI = xr * ss + xi * cc
    return stacked([outR, outI], axis: -1).reshaped([n, heads, headDim])
}

/// The single-stream attention: fused q/k/v, per-head RMS query/key norm, complex rotary, softmax SDPA.
final class NFKZImageAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]                         // [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    let heads: Int
    let headDim: Int

    init(_ config: NFKMLXZImageConfiguration) {
        self.heads = config.heads
        self.headDim = config.headDim
        _toQ.wrappedValue = Linear(config.dim, config.dim, bias: false)
        _toK.wrappedValue = Linear(config.dim, config.dim, bias: false)
        _toV.wrappedValue = Linear(config.dim, config.dim, bias: false)
        _toOut.wrappedValue = [Linear(config.dim, config.dim, bias: false)]
        _normQ.wrappedValue = RMSNorm(dimensions: config.headDim, eps: 1e-5)
        _normK.wrappedValue = RMSNorm(dimensions: config.headDim, eps: 1e-5)
    }

    /// `x` `[N, dim]` (batch 1) → `[N, dim]`.
    func callAsFunction(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
        let n = x.dim(0)
        var q = toQ(x).reshaped([n, heads, headDim])
        var k = toK(x).reshaped([n, heads, headDim])
        let v = toV(x).reshaped([n, heads, headDim])
        q = normQ(q)
        k = normK(k)
        q = zImageApplyRope(q, cos: c, sin: s)
        k = zImageApplyRope(k, cos: c, sin: s)
        // [1, heads, N, headDim] for SDPA.
        let qh = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let kh = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let vh = v.transposed(1, 0, 2).expandedDimensions(axis: 0)
        let scale = 1.0 / sqrt(Float(headDim))
        let out = MLXFast.scaledDotProductAttention(queries: qh, keys: kh, values: vh, scale: scale, mask: .none)
        let merged = out[0].transposed(1, 0, 2).reshaped([n, heads * headDim])
        return (toOut[0] as! Linear)(merged)
    }
}

/// SwiGLU feed-forward (`w2(silu(w1 x) · w3 x)`).
final class NFKZImageFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        _w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        _w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

/// A transformer block: sandwich RMS norms around the attention and the feed-forward, with an optional
/// adaptive-norm modulation (four parameters: scale/gate for each of attention and feed-forward).
final class NFKZImageBlock: Module {
    @ModuleInfo(key: "attention") var attention: NFKZImageAttention
    @ModuleInfo(key: "feed_forward") var feedForward: NFKZImageFeedForward
    @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
    @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: [Module]?    // [Linear] when modulated

    let modulation: Bool

    init(_ config: NFKMLXZImageConfiguration, modulation: Bool) {
        self.modulation = modulation
        _attention.wrappedValue = NFKZImageAttention(config)
        _feedForward.wrappedValue = NFKZImageFeedForward(dim: config.dim, hiddenDim: config.feedForwardDim)
        _attentionNorm1.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _ffnNorm1.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _attentionNorm2.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _ffnNorm2.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        if modulation {
            _adaLNModulation.wrappedValue = [Linear(min(config.dim, 256), 4 * config.dim)]
        } else {
            _adaLNModulation.wrappedValue = nil
        }
    }

    /// `x` `[N, dim]`, `adaln` `[256]` → `[N, dim]`.
    func callAsFunction(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray, adaln: MLXArray?) -> MLXArray {
        if modulation, let adaln, let mod = adaLNModulation {
            let params = (mod[0] as! Linear)(adaln)                        // [4·dim]
            let dim = x.dim(1)
            let scaleMSA = 1.0 + params[0 ..< dim].reshaped([1, dim])
            let gateMSA = tanh(params[dim ..< 2 * dim].reshaped([1, dim]))
            let scaleMLP = 1.0 + params[2 * dim ..< 3 * dim].reshaped([1, dim])
            let gateMLP = tanh(params[3 * dim ..< 4 * dim].reshaped([1, dim]))
            let attnOut = attention(attentionNorm1(x) * scaleMSA, cos: c, sin: s)
            var h = x + gateMSA * attentionNorm2(attnOut)
            h = h + gateMLP * ffnNorm2(feedForward(ffnNorm1(h) * scaleMLP))
            return h
        }
        let attnOut = attention(attentionNorm1(x), cos: c, sin: s)
        var h = x + attentionNorm2(attnOut)
        h = h + ffnNorm2(feedForward(ffnNorm1(h)))
        return h
    }
}

/// The final layer: an affine-free LayerNorm scaled by an adaptive term, then the projection to patches.
final class NFKZImageFinalLayer: Module {
    @ModuleInfo(key: "norm_final") var normFinal: LayerNorm
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: [Module]     // [SiLU-marker, Linear]

    init(dim: Int, outChannels: Int) {
        _normFinal.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _linear.wrappedValue = Linear(dim, outChannels)
        _adaLNModulation.wrappedValue = [Module(), Linear(min(dim, 256), dim)]
    }

    func callAsFunction(_ x: MLXArray, adaln: MLXArray) -> MLXArray {
        let scale = 1.0 + (adaLNModulation[1] as! Linear)(silu(adaln))     // [dim]
        let h = normFinal(x) * scale.reshaped([1, -1])
        return linear(h)
    }
}

/// A `nn.ModuleDict` with a single `"<patch>-<fpatch>"` entry.
final class NFKZImagePatchDict<T: Module>: Module {
    @ModuleInfo(key: "2-1") var entry: T
    init(_ entry: T) { _entry.wrappedValue = entry }
}

/// The timestep embedder: sinusoidal features through `Linear → SiLU → Linear`.
final class NFKZImageTimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") var mlp: [Module]                              // [Linear, SiLU-marker, Linear]
    let frequencySize: Int

    init(outSize: Int, midSize: Int, frequencySize: Int = 256) {
        self.frequencySize = frequencySize
        _mlp.wrappedValue = [Linear(frequencySize, midSize), Module(), Linear(midSize, outSize)]
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let half = frequencySize / 2
        let freqs = exp(-log(10000.0) * MLXArray(0 ..< half).asType(.float32) / Float(half))
        let args = t.reshaped([-1, 1]).asType(.float32) * freqs.reshaped([1, -1])
        let emb = concatenated([cos(args), sin(args)], axis: -1)           // [B, 256]
        return (mlp[2] as! Linear)(silu((mlp[0] as! Linear)(emb)))
    }
}

/// The Z-Image transformer.
public final class NFKMLXZImageTransformerNet: Module {
    @ModuleInfo(key: "all_x_embedder") var allXEmbedder: NFKZImagePatchDict<Linear>
    @ModuleInfo(key: "all_final_layer") var allFinalLayer: NFKZImagePatchDict<NFKZImageFinalLayer>
    @ModuleInfo(key: "noise_refiner") var noiseRefiner: [NFKZImageBlock]
    @ModuleInfo(key: "context_refiner") var contextRefiner: [NFKZImageBlock]
    @ModuleInfo(key: "t_embedder") var tEmbedder: NFKZImageTimestepEmbedder
    @ModuleInfo(key: "cap_embedder") var capEmbedder: [Module]             // [RMSNorm, Linear]
    @ModuleInfo(key: "layers") var layers: [NFKZImageBlock]
    @ParameterInfo(key: "x_pad_token") var xPadToken: MLXArray
    @ParameterInfo(key: "cap_pad_token") var capPadToken: MLXArray

    let config: NFKMLXZImageConfiguration
    let rope: NFKZImageRope

    public init(_ config: NFKMLXZImageConfiguration) {
        self.config = config
        self.rope = NFKZImageRope(axesDims: config.axesDims, theta: config.ropeTheta)
        _allXEmbedder.wrappedValue = NFKZImagePatchDict(Linear(config.patchInputDim, config.dim))
        _allFinalLayer.wrappedValue = NFKZImagePatchDict(
            NFKZImageFinalLayer(dim: config.dim, outChannels: config.patchInputDim))
        _noiseRefiner.wrappedValue = (0 ..< config.refinerLayers).map { _ in NFKZImageBlock(config, modulation: true) }
        _contextRefiner.wrappedValue = (0 ..< config.refinerLayers).map { _ in NFKZImageBlock(config, modulation: false) }
        _tEmbedder.wrappedValue = NFKZImageTimestepEmbedder(outSize: min(config.dim, 256), midSize: 1024)
        _capEmbedder.wrappedValue = [RMSNorm(dimensions: config.captionFeatureDim, eps: config.normEps),
                                     Linear(config.captionFeatureDim, config.dim)]
        _layers.wrappedValue = (0 ..< config.layers).map { _ in NFKZImageBlock(config, modulation: true) }
        _xPadToken.wrappedValue = MLXArray.zeros([1, config.dim])
        _capPadToken.wrappedValue = MLXArray.zeros([1, config.dim])
    }

    /// Patchify one image `[C, F, H, W]` into tokens `[Ntok, patchInputDim]` in the reference's
    /// `(f h w) (pf ph pw c)` order.
    private func patchify(_ image: MLXArray) -> MLXArray {
        let c = image.dim(0), f = image.dim(1), h = image.dim(2), w = image.dim(3)
        let pf = config.framePatchSize, ph = config.patchSize, pw = config.patchSize
        let ft = f / pf, ht = h / ph, wt = w / pw
        let v = image.reshaped([c, ft, pf, ht, ph, wt, pw])
        // c f pf h ph w pw -> (f h w) (pf ph pw c)
        return v.transposed(1, 3, 5, 2, 4, 6, 0).reshaped([ft * ht * wt, pf * ph * pw * c])
    }

    private func unpatchify(_ tokens: MLXArray, f: Int, h: Int, w: Int) -> MLXArray {
        let pf = config.framePatchSize, ph = config.patchSize, pw = config.patchSize
        let ft = f / pf, ht = h / ph, wt = w / pw
        let oc = config.inChannels
        let v = tokens.reshaped([ft, ht, wt, pf, ph, pw, oc])
        // f h w pf ph pw c -> c (f pf) (h ph) (w pw)
        return v.transposed(6, 0, 3, 1, 4, 2, 5).reshaped([oc, f, h, w])
    }

    private static func padCount(_ length: Int, multiple: Int = 32) -> Int {
        (multiple - length % multiple) % multiple
    }

    /// Velocity prediction. `x` `[C, F, H, W]`, `capFeats` `[Lc, captionFeatureDim]`, `t` scalar in `0...1`.
    public func callAsFunction(_ x: MLXArray, capFeats: MLXArray, t: MLXArray) -> MLXArray {
        let f = x.dim(1), h = x.dim(2), w = x.dim(3)
        let pf = config.framePatchSize, ph = config.patchSize, pw = config.patchSize
        let ft = f / pf, ht = h / ph, wt = w / pw

        let adaln = tEmbedder(t * config.timestepScale)[0]                 // [256]

        // Caption tokens: pad length to a multiple of 32, positions continue sequentially from 1.
        let capOri = capFeats.dim(0)
        let capPad = Self.padCount(capOri)
        let capLen = capOri + capPad
        var capFeat = capFeats
        if capPad > 0 {
            capFeat = concatenated([capFeats, tiled(capFeats[(capOri - 1)...], repetitions: [capPad, 1])], axis: 0)
        }
        var cap = (capEmbedder[1] as! Linear)((capEmbedder[0] as! RMSNorm)(capFeat))
        if capPad > 0 {
            let mask = concatenated([MLXArray.zeros([capOri]), MLXArray.ones([capPad])]).asType(.bool)
            cap = MLX.where(mask.reshaped([capLen, 1]), capPadToken, cap)
        }
        let capPositions = MLXArray((0 ..< capLen).map { Float(1 + $0) }).reshaped([capLen, 1])
        let capPos = concatenated([capPositions, MLXArray.zeros([capLen, 2])], axis: 1)
        let (capCos, capSin) = rope.table(positions: capPos)

        // Image tokens: patchify, embed, pad to a multiple of 32 with the learned pad token at (0,0,0).
        let imageOri = ft * ht * wt
        let imagePad = Self.padCount(imageOri)
        let imageLen = imageOri + imagePad
        var tokens = patchify(x)                                          // [imageOri, patchInputDim]
        if imagePad > 0 {
            tokens = concatenated([tokens, tiled(tokens[(imageOri - 1)...], repetitions: [imagePad, 1])], axis: 0)
        }
        var img = allXEmbedder.entry(tokens)
        if imagePad > 0 {
            let mask = concatenated([MLXArray.zeros([imageOri]), MLXArray.ones([imagePad])]).asType(.bool)
            img = MLX.where(mask.reshaped([imageLen, 1]), xPadToken, img)
        }
        // Image position ids: the (f, h, w) grid starting after the caption, then (0,0,0) for padding.
        var pf_ = [Float](), ph_ = [Float](), pw_ = [Float]()
        let base = Float(capLen + 1)
        for fi in 0 ..< ft {
            for hi in 0 ..< ht {
                for wi in 0 ..< wt {
                    pf_.append(base + Float(fi)); ph_.append(Float(hi)); pw_.append(Float(wi))
                }
            }
        }
        for _ in 0 ..< imagePad { pf_.append(0); ph_.append(0); pw_.append(0) }
        let imgPos = concatenated([MLXArray(pf_).reshaped([imageLen, 1]),
                                   MLXArray(ph_).reshaped([imageLen, 1]),
                                   MLXArray(pw_).reshaped([imageLen, 1])], axis: 1)
        let (imgCos, imgSin) = rope.table(positions: imgPos)

        // Refine each stream, then run the unified sequence.
        for block in noiseRefiner { img = block(img, cos: imgCos, sin: imgSin, adaln: adaln) }
        for block in contextRefiner { cap = block(cap, cos: capCos, sin: capSin, adaln: nil) }

        var unified = concatenated([img, cap], axis: 0)
        let uCos = concatenated([imgCos, capCos], axis: 0)
        let uSin = concatenated([imgSin, capSin], axis: 0)
        for block in layers { unified = block(unified, cos: uCos, sin: uSin, adaln: adaln) }

        let out = allFinalLayer.entry(unified, adaln: adaln)              // [imageLen+capLen, patchInputDim]
        let imageTokens = out[0 ..< imageOri]                            // drop caption and padding
        return unpatchify(imageTokens, f: f, h: h, w: w)
    }
}
