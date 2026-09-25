//
//  NFKMLXQwen25VL.swift
//  InferKitMLX
//
//  Qwen2.5-VL's vision tower. Beside Qwen3-VL's, which the toolkit also runs, it has a bias-free patch
//  embedding and no learned position table (the 2D rotary is all the position it carries), RMSNorm
//  blocks with a SiLU-gated feed-forward, WINDOWED attention (tokens attend within 112-pixel windows)
//  except at four full-attention blocks, and a single patch merger without a deepstack. The windows are
//  a permutation: the tokens are reordered so each window is contiguous, attended window by window, and
//  restored to raster order after the merger. The decoder is the dense Qwen2 stack under M-RoPE with
//  Qwen2.5-VL's contiguous section layout.
//

import Foundation
import MLX
import MLXNN

/// The Qwen2.5-VL vision geometry, from a release's `vision_config`.
public struct NFKMLXQwen25VLVisionConfiguration: Sendable {
    public var depth = 32
    public var hiddenSize = 1280
    public var heads = 16
    public var intermediateSize = 3420
    public var outHiddenSize = 2048
    public var patchSize = 14
    public var temporalPatchSize = 2
    public var spatialMergeSize = 2
    public var windowSize = 112
    public var fullAttentionBlocks = [7, 15, 23, 31]

    public init(json: [String: Any]) {
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        depth = int("depth", depth)
        hiddenSize = int("hidden_size", hiddenSize)
        heads = int("num_heads", heads)
        intermediateSize = int("intermediate_size", intermediateSize)
        outHiddenSize = int("out_hidden_size", outHiddenSize)
        patchSize = int("patch_size", patchSize)
        temporalPatchSize = int("temporal_patch_size", temporalPatchSize)
        spatialMergeSize = int("spatial_merge_size", spatialMergeSize)
        windowSize = int("window_size", windowSize)
        fullAttentionBlocks = (json["fullatt_block_indexes"] as? [NSNumber])?.map(\.intValue) ?? fullAttentionBlocks
    }

    var headDim: Int { hiddenSize / heads }
    var patchInputSize: Int { 3 * temporalPatchSize * patchSize * patchSize }
}

/// One vision block: `x + proj(attn(rms1(x)))`, then `x + down(silu(gate(rms2(x))) · up(rms2(x)))`.
final class NFKQwen25VLVisionBlock: Module {
    final class Attention: Module {
        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear
        init(_ width: Int) {
            _qkv.wrappedValue = Linear(width, 3 * width)
            _proj.wrappedValue = Linear(width, width)
        }
    }
    final class MLP: Module {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        init(_ width: Int, _ inner: Int) {
            _gate.wrappedValue = Linear(width, inner)
            _up.wrappedValue = Linear(width, inner)
            _down.wrappedValue = Linear(inner, width)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
    }
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "attn") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP
    let heads: Int

    init(_ c: NFKMLXQwen25VLVisionConfiguration) {
        heads = c.heads
        _norm1.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _norm2.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _attention.wrappedValue = Attention(c.hiddenSize)
        _mlp.wrappedValue = MLP(c.hiddenSize, c.intermediateSize)
    }

    /// `x` is `[tokens, hidden]`; `mask` is an additive `[tokens, tokens]` block mask or nil for full
    /// attention.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        let tokens = x.dim(0), width = x.dim(1), headDim = width / heads
        let fused = attention.qkv(norm1(x)).reshaped([tokens, 3, heads, headDim])
        func rotated(_ t: MLXArray) -> MLXArray {
            let half = headDim / 2
            let rotatedHalf = concatenated([-t[0..., 0..., half...], t[0..., 0..., ..<half]], axis: -1)
            return t * cos + rotatedHalf * sin
        }
        let q = rotated(fused[0..., 0]).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let k = rotated(fused[0..., 1]).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let v = fused[0..., 2].transposed(1, 0, 2).expandedDimensions(axis: 0)
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                         scale: 1 / sqrt(Float(headDim)),
                                                         mask: mask.map { .array($0) } ?? .none)
        let h = x + attention.proj(attended[0].transposed(1, 0, 2).reshaped([tokens, width]))
        return h + mlp(norm2(h))
    }
}

/// The patch merger: `mlp(rms(x))` over each 2×2 block of tokens.
final class NFKQwen25VLMerger: Module {
    @ModuleInfo(key: "ln_q") var norm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: [Linear]

    init(_ c: NFKMLXQwen25VLVisionConfiguration) {
        let merged = c.hiddenSize * c.spatialMergeSize * c.spatialMergeSize
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: 1e-6)
        _mlp.wrappedValue = [Linear(merged, merged), Linear(merged, c.outHiddenSize)]
    }

    func callAsFunction(_ x: MLXArray, mergeUnit: Int) -> MLXArray {
        let grouped = norm(x).reshaped([-1, x.dim(1) * mergeUnit])
        return mlp[1](gelu(mlp[0](grouped)))
    }
}

/// The Qwen2.5-VL vision tower: processed patches `[patches, 3·2·14²]` over a `(t, h, w)` grid in, the
/// merged tokens `[patches / 4, outHidden]` out, in raster order.
public final class NFKMLXQwen25VLVisionNet: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: NFKQwen25VLPatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [NFKQwen25VLVisionBlock]
    @ModuleInfo(key: "merger") var merger: NFKQwen25VLMerger

    public let configuration: NFKMLXQwen25VLVisionConfiguration

    public init(_ c: NFKMLXQwen25VLVisionConfiguration) {
        configuration = c
        _patchEmbed.wrappedValue = NFKQwen25VLPatchEmbed(c)
        _blocks.wrappedValue = (0 ..< c.depth).map { _ in NFKQwen25VLVisionBlock(c) }
        _merger.wrappedValue = NFKQwen25VLMerger(c)
    }

    public func callAsFunction(_ pixelValues: MLXArray, grid: (t: Int, h: Int, w: Int)) -> MLXArray {
        let c = configuration
        let unit = c.spatialMergeSize * c.spatialMergeSize
        let (windowOrder, windowLengths) = Self.windows(grid: grid, configuration: c)
        let order = MLXArray(windowOrder.map(Int32.init))

        // Tokens in 2×2 merge-block groups, the groups permuted into window order.
        var hidden = patchEmbed(pixelValues)
        let patches = hidden.dim(0)
        hidden = hidden.reshaped([patches / unit, unit, -1]).take(order, axis: 0).reshaped([patches, -1])
        let angles = Self.rotaryAngles(grid: grid, configuration: c)
            .reshaped([patches / unit, unit, -1]).take(order, axis: 0).reshaped([patches, -1])
        let doubled = concatenated([angles, angles], axis: -1)                    // [patches, headDim]
        let cosTable = cos(doubled).expandedDimensions(axis: 1), sinTable = sin(doubled).expandedDimensions(axis: 1)

        let windowMask = Self.blockMask(lengths: windowLengths.map { $0 * unit }, total: patches)
        let imageMask = Self.blockMask(lengths: Array(repeating: grid.h * grid.w, count: grid.t), total: patches)
        for (index, block) in blocks.enumerated() {
            let mask = c.fullAttentionBlocks.contains(index) ? imageMask : windowMask
            hidden = block(hidden, cos: cosTable, sin: sinTable, mask: mask)
        }
        let merged = merger(hidden, mergeUnit: unit)
        let restore = MLXArray(Self.inverse(windowOrder).map(Int32.init))
        return merged.take(restore, axis: 0)
    }

    /// The rotary angles `[patches, headDim / 2]`: each patch's height and width positions (in merge-block
    /// order) times the vision rotary's `headDim / 2` frequencies, height first.
    static func rotaryAngles(grid: (t: Int, h: Int, w: Int), configuration c: NFKMLXQwen25VLVisionConfiguration) -> MLXArray {
        let dim = c.headDim / 2
        let inverse = stride(from: 0, to: dim, by: 2).map { 1 / powf(10_000, Float($0) / Float(dim)) }
        let m = c.spatialMergeSize
        var values = [Float]()
        for _ in 0 ..< grid.t {
            for blockRow in 0 ..< grid.h / m {
                for blockColumn in 0 ..< grid.w / m {
                    for r in 0 ..< m {
                        for s in 0 ..< m {
                            let (h, w) = (blockRow * m + r, blockColumn * m + s)
                            values += inverse.map { Float(h) * $0 } + inverse.map { Float(w) * $0 }
                        }
                    }
                }
            }
        }
        return MLXArray(values).reshaped([-1, dim])
    }

    /// The reference's `get_window_index`: the merge-block groups visited window by window (each window
    /// `windowSize / merge / patch` groups on a side, raster order inside), and each window's group count.
    static func windows(grid: (t: Int, h: Int, w: Int), configuration c: NFKMLXQwen25VLVisionConfiguration)
        -> (order: [Int], lengths: [Int]) {
        let side = c.windowSize / c.spatialMergeSize / c.patchSize
        let (llmH, llmW) = (grid.h / c.spatialMergeSize, grid.w / c.spatialMergeSize)
        let (windowsH, windowsW) = ((llmH + side - llmH % side) / side, (llmW + side - llmW % side) / side)
        var order = [Int](), lengths = [Int]()
        for t in 0 ..< grid.t {
            for wh in 0 ..< windowsH {
                for ww in 0 ..< windowsW {
                    var count = 0
                    for r in 0 ..< side {
                        for s in 0 ..< side {
                            let (h, w) = (wh * side + r, ww * side + s)
                            guard h < llmH, w < llmW else { continue }
                            order.append(t * llmH * llmW + h * llmW + w)
                            count += 1
                        }
                    }
                    if count > 0 { lengths.append(count) }
                }
            }
        }
        return (order, lengths)
    }

    /// An additive mask that lets each contiguous run of `lengths` attend only within itself.
    static func blockMask(lengths: [Int], total: Int) -> MLXArray {
        var segment = [Int32](repeating: 0, count: total)
        var start = 0
        for (index, length) in lengths.enumerated() {
            for position in start ..< Swift.min(start + length, total) { segment[position] = Int32(index) }
            start += length
        }
        let ids = MLXArray(segment)
        let same = ids.reshaped([total, 1]) .== ids.reshaped([1, total])
        return MLX.where(same, MLXArray(Float(0)), MLXArray(-Float.infinity))
    }

    static func inverse(_ order: [Int]) -> [Int] {
        var result = [Int](repeating: 0, count: order.count)
        for (position, value) in order.enumerated() { result[value] = position }
        return result
    }
}

/// The bias-free tubelet embedding: a `[hidden, 3·2·14²]` projection of each flattened patch.
final class NFKQwen25VLPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ c: NFKMLXQwen25VLVisionConfiguration) {
        _proj.wrappedValue = Linear(c.patchInputSize, c.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

extension NFKMLXQwen25VLVisionNet {
    /// Loads `<outerPrefix>model.visual.*` (Sa2VA's `model.`), the 5-D patch convolution flattened to its
    /// linear form.
    static func load(directoryURL: URL, configuration: NFKMLXQwen25VLVisionConfiguration,
                     outerPrefix: String, dtype: DType = .float32) throws -> NFKMLXQwen25VLVisionNet {
        let net = NFKMLXQwen25VLVisionNet(configuration)
        let prefix = outerPrefix + "model.visual."
        let mapped = try NFKMLXReleaseWeights.arrays(inDirectory: directoryURL, converting: dtype) { key in
            key.hasPrefix(prefix) ? String(key.dropFirst(prefix.count)) : nil
        }.map { key, value -> (String, MLXArray) in
            let named = key.replacingOccurrences(of: "merger.mlp.2.", with: "merger.mlp.1.")
            if named == "patch_embed.proj.weight", value.ndim == 5 {
                return (named, value.reshaped([value.dim(0), -1]).asType(dtype))
            }
            return (named, value.asType(dtype))
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
        return net
    }
}
