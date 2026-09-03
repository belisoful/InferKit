//
//  NFKMLXQwen3VL.swift
//  InferKitMLX
//
//  Qwen3-VL's vision tower: the second vision-language model's image encoder, and a second vision
//  architecture beside SmolVLM's SigLIP. Where SigLIP is a plain ViT with a learned position embedding
//  and a pixel-shuffle connector, Qwen3-VL's encoder is a 2D-rotary ViT whose patches are laid out in
//  2×2 merge blocks, a bilinearly interpolated position embedding, a merger that folds each block to the
//  decoder width, and a "deepstack" of three feature maps taken from intermediate layers. The decoder is
//  the Qwen3 dense stack `NFKMLXLanguageNet` already runs.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

/// The geometry of the Qwen3-VL vision encoder.
public struct NFKMLXQwen3VLVisionConfiguration: Sendable {
    public var hiddenSize: Int
    public var depth: Int
    public var headCount: Int
    public var intermediateSize: Int
    public var patchSize: Int
    public var temporalPatchSize: Int
    public var spatialMergeSize: Int
    public var outHiddenSize: Int
    public var positionGridSide: Int
    public var deepstackLayers: [Int]
    public var layerNormEpsilon: Float

    public init(hiddenSize: Int = 1024, depth: Int = 24, headCount: Int = 16, intermediateSize: Int = 4096,
                patchSize: Int = 16, temporalPatchSize: Int = 2, spatialMergeSize: Int = 2,
                outHiddenSize: Int = 2048, positionGridSide: Int = 48,
                deepstackLayers: [Int] = [5, 11, 17], layerNormEpsilon: Float = 1e-6) {
        self.hiddenSize = hiddenSize
        self.depth = depth
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.spatialMergeSize = spatialMergeSize
        self.outHiddenSize = outHiddenSize
        self.positionGridSide = positionGridSide
        self.deepstackLayers = deepstackLayers
        self.layerNormEpsilon = layerNormEpsilon
    }

    public static let qwen3VL2B = NFKMLXQwen3VLVisionConfiguration()

    var headDimensions: Int { hiddenSize / headCount }
    var patchInputSize: Int { 3 * temporalPatchSize * patchSize * patchSize }
    var mergedSize: Int { hiddenSize * spatialMergeSize * spatialMergeSize }
}

/// The patch embedding. The reference convolves each `[temporal, patch, patch]` patch with a full-size
/// kernel, which for a kernel equal to the patch is one linear projection over the flattened patch.
final class NFKQwen3VLPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ c: NFKMLXQwen3VLVisionConfiguration) {
        _proj.wrappedValue = Linear(c.patchInputSize, c.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray { proj(pixelValues) }
}

/// Qwen3-VL vision attention: one fused QKV projection with a bias, 2D rotary on the queries and keys,
/// and full attention over an image's patches.
final class NFKQwen3VLVisionAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    let heads: Int
    let headDimensions: Int
    let scale: Float

    init(_ c: NFKMLXQwen3VLVisionConfiguration) {
        heads = c.headCount
        headDimensions = c.headDimensions
        scale = 1 / sqrt(Float(c.headDimensions))
        _qkv.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: true)
        _proj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let length = x.dim(0)
        let projected = qkv(x).reshaped([length, 3, heads, headDimensions])
        var queries = projected[0..., 0].transposed(1, 0, 2)         // [heads, length, headDim]
        var keys = projected[0..., 1].transposed(1, 0, 2)
        let values = projected[0..., 2].transposed(1, 0, 2)

        queries = applyRotary(queries, cos: cos, sin: sin)
        keys = applyRotary(keys, cos: cos, sin: sin)

        let attention = MLXFast.scaledDotProductAttention(
            queries: queries.expandedDimensions(axis: 0), keys: keys.expandedDimensions(axis: 0),
            values: values.expandedDimensions(axis: 0), scale: scale, mask: nil)[0]
        return proj(attention.transposed(1, 0, 2).reshaped([length, heads * headDimensions]))
    }

    /// The 2D rotary embedding: `x·cos + rotateHalf(x)·sin`, `cos`/`sin` shared across heads.
    private func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = x.dim(2) / 2
        let rotated = concatenated([-x[0..., 0..., half...], x[0..., 0..., 0 ..< half]], axis: -1)
        return x * cos + rotated * sin
    }
}

/// Qwen3-VL vision feed-forward: a projection up, the tanh-approximate GELU, a projection back.
final class NFKQwen3VLVisionMLP: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    init(_ c: NFKMLXQwen3VLVisionConfiguration) {
        _fc1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
}

/// One Qwen3-VL vision block: pre-normalized attention and feed-forward, each added back.
final class NFKQwen3VLVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attention: NFKQwen3VLVisionAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKQwen3VLVisionMLP

    init(_ c: NFKMLXQwen3VLVisionConfiguration) {
        _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        _attention.wrappedValue = NFKQwen3VLVisionAttention(c)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        _mlp.wrappedValue = NFKQwen3VLVisionMLP(c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let attended = x + attention(norm1(x), cos: cos, sin: sin)
        return attended + mlp(norm2(attended))
    }
}

/// The patch merger: a normalization, then a 2×2 block folded to `spatialMergeSize²` times the channels
/// and projected to the decoder width. The main merger normalizes before folding; a deepstack merger
/// normalizes after (the `postShuffle` flag).
final class NFKQwen3VLMerger: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    let mergedSize: Int
    let postShuffle: Bool

    init(_ c: NFKMLXQwen3VLVisionConfiguration, postShuffle: Bool) {
        mergedSize = c.mergedSize
        self.postShuffle = postShuffle
        _norm.wrappedValue = LayerNorm(dimensions: postShuffle ? c.mergedSize : c.hiddenSize,
                                       eps: c.layerNormEpsilon)
        _fc1.wrappedValue = Linear(c.mergedSize, c.mergedSize, bias: true)
        _fc2.wrappedValue = Linear(c.mergedSize, c.outHiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let folded = postShuffle ? norm(x.reshaped([-1, mergedSize])) : norm(x).reshaped([-1, mergedSize])
        return fc2(gelu(fc1(folded)))
    }
}

/// The Qwen3-VL vision encoder. It returns the merged image features and the three deepstack feature
/// maps the decoder injects at its first layers.
public final class NFKMLXQwen3VLVisionNet: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: NFKQwen3VLPatchEmbed
    @ModuleInfo(key: "pos_embed") var positionEmbedding: Embedding
    @ModuleInfo(key: "blocks") var blocks: [NFKQwen3VLVisionBlock]
    @ModuleInfo(key: "merger") var merger: NFKQwen3VLMerger
    @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [NFKQwen3VLMerger]

    let configuration: NFKMLXQwen3VLVisionConfiguration

    init(_ c: NFKMLXQwen3VLVisionConfiguration) {
        configuration = c
        _patchEmbed.wrappedValue = NFKQwen3VLPatchEmbed(c)
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: c.positionGridSide * c.positionGridSide,
                                                    dimensions: c.hiddenSize)
        _blocks.wrappedValue = (0 ..< c.depth).map { _ in NFKQwen3VLVisionBlock(c) }
        _merger.wrappedValue = NFKQwen3VLMerger(c, postShuffle: false)
        _deepstackMergers.wrappedValue = c.deepstackLayers.map { _ in NFKQwen3VLMerger(c, postShuffle: true) }
        super.init()
    }

    /// `pixelValues` is `[patches, patchInputSize]` in the processor's 2×2-merge-block order, `grid` a
    /// `[temporal, height, width]` patch grid. Returns the merged features `[patches / 4, outHidden]` and
    /// the three deepstack feature maps.
    public func callAsFunction(_ pixelValues: MLXArray, grid: (t: Int, h: Int, w: Int))
        -> (output: MLXArray, deepstack: [MLXArray]) {
        var hidden = patchEmbed(pixelValues) + interpolatedPositionEmbedding(grid: grid)
        let (cos, sin) = rotaryEmbedding(grid: grid)

        var deepstack = [MLXArray]()
        for (index, block) in blocks.enumerated() {
            hidden = block(hidden, cos: cos, sin: sin)
            if let stack = configuration.deepstackLayers.firstIndex(of: index) {
                deepstack.append(deepstackMergers[stack](hidden))
            }
        }
        return (merger(hidden), deepstack)
    }

    /// The learned 48×48 position embedding bilinearly interpolated to the image's grid, then reordered
    /// into 2×2 merge-block order to line up with the patches.
    func interpolatedPositionEmbedding(grid: (t: Int, h: Int, w: Int)) -> MLXArray {
        let side = configuration.positionGridSide
        let (h, w) = (grid.h, grid.w)
        func samples(_ count: Int) -> (floor: [Int], ceil: [Int], fraction: [Float]) {
            var floors = [Int](), ceils = [Int](), fractions = [Float]()
            for index in 0 ..< count {
                let coordinate = count == 1 ? 0 : Float(index) * Float(side - 1) / Float(count - 1)
                let floor = Int(coordinate)
                floors.append(floor)
                ceils.append(Swift.min(floor + 1, side - 1))
                fractions.append(coordinate - Float(floor))
            }
            return (floors, ceils, fractions)
        }
        let rows = samples(h), columns = samples(w)

        var indices = [[Int32]](repeating: [], count: 4)
        var weights = [[Float]](repeating: [], count: 4)
        let merge = configuration.spatialMergeSize
        for blockRow in 0 ..< (h / merge) {
            for blockColumn in 0 ..< (w / merge) {
                for intraRow in 0 ..< merge {
                    for intraColumn in 0 ..< merge {
                        let i = blockRow * merge + intraRow, j = blockColumn * merge + intraColumn
                        let corners = [(rows.floor[i], columns.floor[j], (1 - rows.fraction[i]) * (1 - columns.fraction[j])),
                                       (rows.floor[i], columns.ceil[j], (1 - rows.fraction[i]) * columns.fraction[j]),
                                       (rows.ceil[i], columns.floor[j], rows.fraction[i] * (1 - columns.fraction[j])),
                                       (rows.ceil[i], columns.ceil[j], rows.fraction[i] * columns.fraction[j])]
                        for (corner, value) in corners.enumerated() {
                            indices[corner].append(Int32(value.0 * side + value.1))
                            weights[corner].append(value.2)
                        }
                    }
                }
            }
        }
        let count = h * w
        var result = MLXArray.zeros([count, configuration.hiddenSize])
        for corner in 0 ..< 4 {
            let gathered = positionEmbedding(MLXArray(indices[corner]))
            result = result + gathered * MLXArray(weights[corner]).reshaped([count, 1])
        }
        return grid.t == 1 ? result : concatenated(Array(repeating: result, count: grid.t), axis: 0)
    }

    /// The 2D rotary cosine and sine tables for the patch grid, in merge-block order.
    func rotaryEmbedding(grid: (t: Int, h: Int, w: Int)) -> (cos: MLXArray, sin: MLXArray) {
        let rotaryDimensions = configuration.headDimensions / 2                 // 32
        let frequencies = (0 ..< rotaryDimensions / 2).map {
            1 / pow(Float(10_000), Float(2 * $0) / Float(rotaryDimensions))
        }
        let merge = configuration.spatialMergeSize
        var table = [Float]()
        for _ in 0 ..< grid.t {
            for blockRow in 0 ..< (grid.h / merge) {
                for blockColumn in 0 ..< (grid.w / merge) {
                    for intraRow in 0 ..< merge {
                        for intraColumn in 0 ..< merge {
                            let row = Float(blockRow * merge + intraRow)
                            let column = Float(blockColumn * merge + intraColumn)
                            table.append(contentsOf: frequencies.map { row * $0 })
                            table.append(contentsOf: frequencies.map { column * $0 })
                        }
                    }
                }
            }
        }
        let count = grid.t * grid.h * grid.w
        let rotary = MLXArray(table).reshaped([count, rotaryDimensions])
        let doubled = concatenated([rotary, rotary], axis: -1)                  // [count, headDim]
        return (cos(doubled).expandedDimensions(axis: 0), sin(doubled).expandedDimensions(axis: 0))
    }
}

/// Building and loading Qwen3-VL.
///
/// `NFKMLXQwen3VL` is the released `Qwen/Qwen3-VL-2B-Instruct`. The vision tower is ``NFKMLXQwen3VLVisionNet``;
/// the decoder is the Qwen3 dense stack `NFKMLXLanguage` already runs, loaded from the checkpoint's
/// `model.language_model.` subtree.
@objc(NFKMLXQwen3VL)
public final class NFKMLXQwen3VL: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "qwen3-vl-2b"

    /// Builds the vision tower from a downloaded release directory.
    public static func visionNet(directoryURL: URL) throws -> NFKMLXQwen3VLVisionNet {
        let net = NFKMLXQwen3VLVisionNet(.qwen3VL2B)
        try loadVisionWeights(into: net, directoryURL: directoryURL)
        return net
    }

    /// Loads the `model.visual.` subtree. The patch-embedding convolution weight is stored 5-D
    /// (`[out, channels, temporal, patch, patch]`) and flattens to a linear weight `[out, patchInput]`.
    static func loadVisionWeights(into net: NFKMLXQwen3VLVisionNet, directoryURL: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(
            url: directoryURL.appendingPathComponent("model.safetensors"))
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("model.visual.") else { return nil }
            let stripped = String(key.dropFirst("model.visual.".count))
            var array = value
            if stripped == "patch_embed.proj.weight", value.ndim == 5 {
                array = value.reshaped([value.dim(0), -1])
            }
            let keeps = array.dtype != .float16 && array.dtype != .bfloat16
            return (stripped, keeps ? array : array.asType(.float32))
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }
}
