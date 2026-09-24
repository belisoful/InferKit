import Foundation
import MLX
import MLXFast
import MLXNN

/*!
 @abstract Geometry of a V-JEPA 2 vision encoder (`facebook/vjepa2-*`).
 @discussion The released encoder is a ViT-L: a 3D-convolution tubelet patch embedding, 24 pre-norm
 transformer blocks with 3D rotary attention, and a final layer norm. No class token and no learned
 position table; position enters only through the rotary embedding. The predictor and attentive-pooler
 heads in the checkpoint belong to pretraining and classification and are not part of feature extraction.
 Introduced in InferKit 0.4.0.
 */
public struct NFKMLXVJEPA2Configuration: Sendable {
    public var hiddenSize: Int = 1024
    public var numHiddenLayers: Int = 24
    public var numAttentionHeads: Int = 16
    public var mlpRatio: Double = 4.0
    public var patchSize: Int = 16
    public var tubeletSize: Int = 2
    public var framesPerClip: Int = 64
    public var cropSize: Int = 256
    public var inChannels: Int = 3
    public var layerNormEps: Float = 1e-6
    public var qkvBias: Bool = true
    /// The self-attention layers of a classification release's attentive pooler (`num_pooler_layers`);
    /// 0 for an encoder release, which has no pooler.
    public var poolerLayers: Int = 0
    /// A classification release's class names in index order (`id2label`); empty for an encoder release.
    public var labels: [String] = []
    /// The frame resize's shortest edge (`video_preprocessor_config.json`'s `size.shortest_edge`, else
    /// `crop_size · 256 / 224`, the reference processor's default).
    public var shortestEdge: Int = 292

    /// The `facebook/vjepa2-vitl-fpc64-256` geometry.
    public static let vitLarge = NFKMLXVJEPA2Configuration()

    public init() {}

    /// Reads a Hugging Face `config.json` for a `vjepa2` model.
    public init(configurationURL: URL) throws {
        let data = try Data(contentsOf: configurationURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func dbl(_ key: String, _ fallback: Double) -> Double { (json[key] as? NSNumber)?.doubleValue ?? fallback }
        hiddenSize = int("hidden_size", hiddenSize)
        numHiddenLayers = int("num_hidden_layers", numHiddenLayers)
        numAttentionHeads = int("num_attention_heads", numAttentionHeads)
        mlpRatio = dbl("mlp_ratio", mlpRatio)
        patchSize = int("patch_size", patchSize)
        tubeletSize = int("tubelet_size", tubeletSize)
        framesPerClip = int("frames_per_clip", framesPerClip)
        cropSize = int("crop_size", int("image_size", cropSize))
        inChannels = int("in_chans", inChannels)
        layerNormEps = Float(dbl("layer_norm_eps", Double(layerNormEps)))
        qkvBias = (json["qkv_bias"] as? NSNumber)?.boolValue ?? qkvBias
        let architectures = json["architectures"] as? [String] ?? []
        if architectures.contains("VJEPA2ForVideoClassification"), let names = json["id2label"] as? [String: String] {
            poolerLayers = int("num_pooler_layers", 3)
            labels = names.compactMap { key, name in Int(key).map { ($0, name) } }.sorted { $0.0 < $1.0 }.map(\.1)
        }
        shortestEdge = cropSize * 256 / 224
        let processorURL = configurationURL.deletingLastPathComponent().appendingPathComponent("video_preprocessor_config.json")
        if let data = try? Data(contentsOf: processorURL),
           let processor = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let edge = ((processor["size"] as? [String: Any])?["shortest_edge"] as? NSNumber)?.intValue {
            shortestEdge = edge
        }
    }

    /// Tokens along the spatial grid on one axis (`crop_size / patch_size`).
    var gridSize: Int { cropSize / patchSize }
    var headDim: Int { hiddenSize / numAttentionHeads }
    /// The per-axis rotary width `2 * floor(floor(headDim/3)/2)`, applied to each of the three axes.
    var rotaryAxisDim: Int { 2 * ((headDim / 3) / 2) }
}

/// The tubelet patch embedding: a non-overlapping 3D convolution (`tubelet × patch × patch`) over a
/// `[B, T, H, W, C]` clip, flattened to `[B, N, hidden]` in temporal-major, then row-major, order.
final class NFKVJEPA2PatchEmbeddings: Module {
    @ModuleInfo(key: "proj") var proj: Conv3d

    init(_ c: NFKMLXVJEPA2Configuration) {
        _proj.wrappedValue = Conv3d(
            inputChannels: c.inChannels, outputChannels: c.hiddenSize,
            kernelSize: IntOrTriple((c.tubeletSize, c.patchSize, c.patchSize)),
            stride: IntOrTriple((c.tubeletSize, c.patchSize, c.patchSize)))
    }

    func callAsFunction(_ clip: MLXArray) -> MLXArray {
        let patched = proj(clip)                                     // [B, T', H', W', hidden]
        let s = patched.shape
        return patched.reshaped([s[0], s[1] * s[2] * s[3], s[4]])    // [B, N, hidden]
    }
}

/// Wraps the patch embedding so its checkpoint keys read `embeddings.patch_embeddings.proj.*`.
final class NFKVJEPA2Embeddings: Module {
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: NFKVJEPA2PatchEmbeddings

    init(_ c: NFKMLXVJEPA2Configuration) {
        _patchEmbeddings.wrappedValue = NFKVJEPA2PatchEmbeddings(c)
    }

    func callAsFunction(_ clip: MLXArray) -> MLXArray { patchEmbeddings(clip) }
}

/// Applies one axis of V-JEPA 2's rotary embedding to `[B, heads, N, axisDim]`.
///
/// The reference (`rotate_queries_or_keys`) pairs adjacent channels `(0,1),(2,3),…` for the rotation
/// but tiles the cosine/sine tables as two concatenated halves `[c₀…c₉, c₀…c₉]`, so channel `k` is
/// scaled by frequency `k mod (axisDim/2)`. This reproduces that arithmetic exactly rather than
/// assuming a standard interleaved rotation.
private func vjepa2RotateAxis(_ x: MLXArray, positions: MLXArray, tables: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
    let shape = x.shape
    let half = shape[3] / 2
    // Pair adjacent channels: y = [-x_odd, x_even] interleaved.
    let paired = x.reshaped([shape[0], shape[1], shape[2], half, 2])
    let even = paired[0..., 0..., 0..., 0..., 0]
    let odd = paired[0..., 0..., 0..., 0..., 1]
    let rotated = MLX.stacked([-odd, even], axis: -1).reshaped(shape)
    return x * tables.cos + rotated * tables.sin
}

/// 3D rotary self-attention. Queries and keys are rotated along the temporal, height, and width axes
/// (each `rotaryAxisDim` channels; any remainder is left unrotated) before scaled dot-product attention.
final class NFKVJEPA2Attention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    let heads: Int
    let headDim: Int
    let axisDim: Int
    let scale: Float

    init(_ c: NFKMLXVJEPA2Configuration) {
        heads = c.numAttentionHeads
        headDim = c.headDim
        axisDim = c.rotaryAxisDim
        scale = 1.0 / Float(headDim).squareRoot()
        _query.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.qkvBias)
        _key.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.qkvBias)
        _value.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.qkvBias)
        _proj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize)
    }

    private func split(_ t: MLXArray) -> MLXArray {
        t.reshaped([t.dim(0), t.dim(1), heads, headDim]).transposed(0, 2, 1, 3)
    }

    /// Rotates `qk` [B, heads, N, headDim] over the three position axes, concatenating any unrotated tail.
    private func applyRotary(_ qk: MLXArray, _ rot: NFKVJEPA2RotaryTables) -> MLXArray {
        guard axisDim > 0 else { return qk }
        var pieces: [MLXArray] = []
        for axis in 0..<3 {
            let lo = axis * axisDim
            let slice = qk[0..., 0..., 0..., lo ..< (lo + axisDim)]
            pieces.append(vjepa2RotateAxis(slice, positions: rot.positions[axis], tables: rot.tables[axis]))
        }
        let used = 3 * axisDim
        if used < headDim {
            pieces.append(qk[0..., 0..., 0..., used ..< headDim])
        }
        return concatenated(pieces, axis: -1)
    }

    func callAsFunction(_ x: MLXArray, rotary: NFKVJEPA2RotaryTables) -> MLXArray {
        var q = split(query(x))
        var k = split(key(x))
        let v = split(value(x))
        q = applyRotary(q, rotary)
        k = applyRotary(k, rotary)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([x.dim(0), x.dim(1), heads * headDim])
        return proj(merged)
    }
}

/// The block feed-forward: `fc2(gelu(fc1(x)))`. The attentive pooler's layers use the same shape.
final class NFKVJEPA2MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ c: NFKMLXVJEPA2Configuration) {
        let hidden = Int(Double(c.hiddenSize) * c.mlpRatio)
        _fc1.wrappedValue = Linear(c.hiddenSize, hidden)
        _fc2.wrappedValue = Linear(hidden, c.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// One pre-norm transformer block: `x + attn(norm1(x))` then `x + mlp(norm2(x))`.
final class NFKVJEPA2Layer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: NFKVJEPA2Attention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKVJEPA2MLP

    init(_ c: NFKMLXVJEPA2Configuration) {
        _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _attention.wrappedValue = NFKVJEPA2Attention(c)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _mlp.wrappedValue = NFKVJEPA2MLP(c)
    }

    func callAsFunction(_ x: MLXArray, rotary: NFKVJEPA2RotaryTables) -> MLXArray {
        var h = x + attention(norm1(x), rotary: rotary)
        h = h + mlp(norm2(h))
        return h
    }
}

/// The precomputed rotary cosine/sine tables and position vectors for one token count. The three axes
/// are temporal, height, and width; each carries `[N]` positions and `[1, 1, N, axisDim]` cos/sin tables.
final class NFKVJEPA2RotaryTables {
    let positions: [MLXArray]
    let tables: [(cos: MLXArray, sin: MLXArray)]

    init(tokenCount N: Int, gridSize: Int, axisDim: Int) {
        let tokensPerFrame = gridSize * gridSize
        var frame = [Float](), height = [Float](), width = [Float]()
        frame.reserveCapacity(N); height.reserveCapacity(N); width.reserveCapacity(N)
        for i in 0..<N {
            let f = i / tokensPerFrame
            let rem = i - tokensPerFrame * f
            let h = rem / gridSize
            frame.append(Float(f)); height.append(Float(h)); width.append(Float(rem - gridSize * h))
        }
        let pos = [MLXArray(frame), MLXArray(height), MLXArray(width)]
        positions = pos

        // omega[j] = 10000^(-j / (axisDim/2)); freq = pos ⊗ omega; tables tile the two halves.
        var built: [(cos: MLXArray, sin: MLXArray)] = []
        if axisDim > 0 {
            let half = axisDim / 2
            let omega = MLX.exp(-(MLXArray(0..<half).asType(.float32) / Float(half)) * Float(log(10000.0)))
            for p in pos {
                let freq = p.reshaped([N, 1]) * omega.reshaped([1, half])   // [N, half]
                let cosFull = concatenated([MLX.cos(freq), MLX.cos(freq)], axis: 1).reshaped([1, 1, N, axisDim])
                let sinFull = concatenated([MLX.sin(freq), MLX.sin(freq)], axis: 1).reshaped([1, 1, N, axisDim])
                built.append((cosFull, sinFull))
            }
        }
        tables = built
    }
}

/// The V-JEPA 2 vision encoder. `callAsFunction` maps a normalized clip `[B, T, H, W, C]` to the
/// per-token feature sequence `[B, N, hidden]` that the reference `get_vision_features` returns.
final class NFKVJEPA2Encoder: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKVJEPA2Embeddings
    @ModuleInfo(key: "layer") var layer: [NFKVJEPA2Layer]
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm

    let gridSize: Int
    let axisDim: Int

    init(_ c: NFKMLXVJEPA2Configuration) {
        gridSize = c.gridSize
        axisDim = c.rotaryAxisDim
        _embeddings.wrappedValue = NFKVJEPA2Embeddings(c)
        _layer.wrappedValue = (0 ..< c.numHiddenLayers).map { _ in NFKVJEPA2Layer(c) }
        _layernorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
    }

    /// The hidden state after each block (index `i` = input to block `i`, last = final normed output),
    /// used by the seam test. Element 0 is the patch embedding.
    func hiddenStates(_ clip: MLXArray) -> [MLXArray] {
        var h = embeddings(clip)
        let rotary = NFKVJEPA2RotaryTables(tokenCount: h.dim(1), gridSize: gridSize, axisDim: axisDim)
        var out = [h]
        for block in layer {
            h = block(h, rotary: rotary)
            out.append(h)
        }
        out.append(layernorm(h))
        return out
    }

    func callAsFunction(_ clip: MLXArray) -> MLXArray {
        var h = embeddings(clip)
        let rotary = NFKVJEPA2RotaryTables(tokenCount: h.dim(1), gridSize: gridSize, axisDim: axisDim)
        for block in layer {
            h = block(h, rotary: rotary)
        }
        return layernorm(h)
    }
}

// MARK: - Attentive pooler (classification releases)

/// Scaled dot-product attention over `heads` heads of `[B, T, hidden]` projections.
private func vjepa2Attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, heads: Int) -> MLXArray {
    let headDim = q.dim(2) / heads
    func split(_ t: MLXArray) -> MLXArray { t.reshaped([t.dim(0), t.dim(1), heads, headDim]).transposed(0, 2, 1, 3) }
    let attended = MLXFast.scaledDotProductAttention(queries: split(q), keys: split(k), values: split(v),
                                                     scale: 1 / sqrt(Float(headDim)), mask: nil)
    return attended.transposed(0, 2, 1, 3).reshaped([q.dim(0), q.dim(1), q.dim(2)])
}

/// A pooler self-attention layer: `x + out(attn(norm1(x)))`, then `x + mlp(norm2(x))`.
final class NFKVJEPA2PoolerSelfAttentionLayer: Module {
    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var q: Linear
        @ModuleInfo(key: "k_proj") var k: Linear
        @ModuleInfo(key: "v_proj") var v: Linear
        @ModuleInfo(key: "out_proj") var out: Linear
        init(_ width: Int) {
            _q.wrappedValue = Linear(width, width)
            _k.wrappedValue = Linear(width, width)
            _v.wrappedValue = Linear(width, width)
            _out.wrappedValue = Linear(width, width)
        }
    }
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKVJEPA2MLP
    let heads: Int

    init(_ c: NFKMLXVJEPA2Configuration) {
        heads = c.numAttentionHeads
        _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _attention.wrappedValue = Attention(c.hiddenSize)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _mlp.wrappedValue = NFKVJEPA2MLP(c)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let n = norm1(x)
        let h = x + attention.out(vjepa2Attention(attention.q(n), attention.k(n), attention.v(n), heads: heads))
        return h + mlp(norm2(h))
    }
}

/// The pooler's cross-attention layer: the learned query attends the NORMALIZED tokens (the query itself
/// is not normalized, and there is no output projection), then `x + mlp(norm2(x))`.
final class NFKVJEPA2PoolerCrossAttentionLayer: Module {
    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var q: Linear
        @ModuleInfo(key: "k_proj") var k: Linear
        @ModuleInfo(key: "v_proj") var v: Linear
        init(_ width: Int) {
            _q.wrappedValue = Linear(width, width)
            _k.wrappedValue = Linear(width, width)
            _v.wrappedValue = Linear(width, width)
        }
    }
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "cross_attn") var attention: Attention
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKVJEPA2MLP
    let heads: Int

    init(_ c: NFKMLXVJEPA2Configuration) {
        heads = c.numAttentionHeads
        _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _attention.wrappedValue = Attention(c.hiddenSize)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _mlp.wrappedValue = NFKVJEPA2MLP(c)
    }

    func callAsFunction(queries: MLXArray, tokens: MLXArray) -> MLXArray {
        let n = norm1(tokens)
        let h = queries + vjepa2Attention(attention.q(queries), attention.k(n), attention.v(n), heads: heads)
        return h + mlp(norm2(h))
    }
}

/// The classification releases' attentive pooler (`pooler`): self-attention layers over the tokens,
/// then one learned query token cross-attends them.
final class NFKVJEPA2AttentivePooler: Module {
    @ParameterInfo(key: "query_tokens") var queryTokens: MLXArray
    @ModuleInfo(key: "cross_attention_layer") var crossAttention: NFKVJEPA2PoolerCrossAttentionLayer
    @ModuleInfo(key: "self_attention_layers") var selfAttention: [NFKVJEPA2PoolerSelfAttentionLayer]

    init(_ c: NFKMLXVJEPA2Configuration) {
        _queryTokens.wrappedValue = MLXArray.zeros([1, 1, c.hiddenSize])
        _crossAttention.wrappedValue = NFKVJEPA2PoolerCrossAttentionLayer(c)
        _selfAttention.wrappedValue = (0 ..< c.poolerLayers).map { _ in NFKVJEPA2PoolerSelfAttentionLayer(c) }
    }

    /// `[B, tokens, hidden]` → `[B, hidden]`.
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var h = tokens
        for layer in selfAttention { h = layer(h) }
        let queries = broadcast(queryTokens, to: [h.dim(0), 1, h.dim(2)])
        return crossAttention(queries: queries, tokens: h)[0..., 0, 0...]
    }
}

/*!
 @abstract The V-JEPA 2 vision encoder as an `MLXNN` module.
 @discussion Wraps the 24-layer encoder under the checkpoint's `encoder.*` key namespace. The forward
 pass takes a normalized clip `[B, frames, height, width, 3]` and returns the token feature sequence
 `[B, tokens, hidden]`, matching the reference `VJEPA2Model.get_vision_features`.
 Introduced in InferKit 0.4.0.
 */
public final class NFKMLXVJEPA2Net: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKVJEPA2Encoder
    @ModuleInfo(key: "pooler") var pooler: NFKVJEPA2AttentivePooler?
    @ModuleInfo(key: "classifier") var classifier: Linear?

    public let configuration: NFKMLXVJEPA2Configuration

    public init(_ configuration: NFKMLXVJEPA2Configuration) {
        self.configuration = configuration
        _encoder.wrappedValue = NFKVJEPA2Encoder(configuration)
        let classifies = !configuration.labels.isEmpty
        _pooler.wrappedValue = classifies ? NFKVJEPA2AttentivePooler(configuration) : nil
        _classifier.wrappedValue = classifies ? Linear(configuration.hiddenSize, configuration.labels.count) : nil
        super.init()
    }

    /// Whether the release carries the attentive pooler and classifier (a video-classification release).
    public var classifies: Bool { classifier != nil }

    /// A classification release's class logits `[B, labels]`: the encoder's tokens through the attentive
    /// pooler and the linear classifier. Nil for an encoder release.
    public func classLogits(_ clip: MLXArray) -> MLXArray? {
        guard let pooler, let classifier else { return nil }
        return classifier(pooler(self(clip)))
    }

    /// The pooler's output `[B, hidden]` from the encoder's features, for parity measurement.
    func pooled(_ features: MLXArray) -> MLXArray? { pooler?(features) }

    public convenience init(configurationURL: URL) throws {
        try self.init(NFKMLXVJEPA2Configuration(configurationURL: configurationURL))
    }

    /// The per-token vision features `[B, tokens, hidden]`.
    public func callAsFunction(_ clip: MLXArray) -> MLXArray { encoder(clip) }

    /// Per-seam hidden states for parity measurement: `[patch embedding, block₀ … block₂₃, final norm]`.
    public func hiddenStates(_ clip: MLXArray) -> [MLXArray] { encoder.hiddenStates(clip) }
}
