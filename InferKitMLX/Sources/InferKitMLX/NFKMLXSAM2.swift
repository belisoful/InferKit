//
//  NFKMLXSAM2.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// SAM 2 replaces SAM's plain ViT with **Hiera**, a hierarchical encoder: the feature map halves in
// resolution and doubles in width at each of four stages, attention runs inside local windows except
// at a few designated global blocks, and the stage transitions pool the queries so a block can change
// resolution and width at once. An FPN neck then projects the captured stages to one width and fuses
// them top-down. This file ports that image encoder; the prompt encoder, mask decoder, and the video
// memory path are separate concerns. Tensors flow in NHWC.

/// PyTorch-compatible bicubic resampling, which the position embedding needs: SAM 2 interpolates a
/// small learned grid up to the patch grid with `mode="bicubic"`, and neither bilinear nor nearest
/// reproduces it. Uses the same `a = -0.75` Keys kernel and half-pixel centers PyTorch does.
enum NFKMLXBicubic {
    private static func weights(_ t: Float) -> [Float] {
        let a: Float = -0.75
        let (x1, x2) = (t, 1 - t)
        // The two inner taps fall within one sample, the two outer ones between one and two.
        func near(_ x: Float) -> Float { ((a + 2) * x - (a + 3)) * x * x + 1 }
        func far(_ x: Float) -> Float { (((x - 5) * x + 8) * x - 4) * a }
        return [far(x1 + 1), near(x1), near(x2), far(x2 + 1)]
    }

    /// Resamples `[N, H, W, C]` to `height` × `width`.
    static func resize(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        func axis(_ outSize: Int, _ inSize: Int) -> (indices: [MLXArray], taps: [MLXArray]) {
            let scale = Float(inSize) / Float(outSize)
            var columns = [[Int32]](repeating: [], count: 4)
            var taps = [[Float]](repeating: [], count: 4)
            for out in 0 ..< outSize {
                let source = (Float(out) + 0.5) * scale - 0.5
                let base = Int(floor(source))
                let weight = weights(source - Float(base))
                for tap in 0 ..< 4 {
                    // PyTorch clamps the sample position at the border rather than zero-padding.
                    columns[tap].append(Int32(min(max(base - 1 + tap, 0), inSize - 1)))
                    taps[tap].append(weight[tap])
                }
            }
            return (columns.map { MLXArray($0) }, taps.map { MLXArray($0) })
        }

        let (rows, rowTaps) = axis(height, h)
        var vertical = MLXArray.zeros([n, height, w, c])
        for tap in 0 ..< 4 {
            vertical = vertical + x[0..., rows[tap]] * rowTaps[tap].reshaped([1, height, 1, 1])
        }
        let (cols, colTaps) = axis(width, w)
        var out = MLXArray.zeros([n, height, width, c])
        for tap in 0 ..< 4 {
            out = out + vertical[0..., 0..., cols[tap]] * colTaps[tap].reshaped([1, 1, width, 1])
        }
        return out
    }
}

/// The two-layer MLP each Hiera block ends with.
final class NFKSAM2MLP: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(dimensions: Int, hidden: Int) {
        _layers.wrappedValue = [Linear(dimensions, hidden), Linear(hidden, dimensions)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { layers[1](gelu(layers[0](x))) }
}

/// Multi-scale attention: one fused projection for queries, keys, and values, with the queries
/// optionally max-pooled so the block can downsample at a stage boundary.
final class NFKSAM2Attention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let heads: Int
    let poolsQueries: Bool

    init(dimensions: Int, outDimensions: Int, heads: Int, poolsQueries: Bool) {
        _qkv.wrappedValue = Linear(dimensions, outDimensions * 3)
        _proj.wrappedValue = Linear(outDimensions, outDimensions)
        self.heads = heads
        self.poolsQueries = poolsQueries
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, h, w) = (x.shape[0], x.shape[1], x.shape[2])
        let projected = qkv(x).reshaped([batch, h * w, 3, heads, -1])
        var queries = projected[0..., 0..., 0]
        let keys = projected[0..., 0..., 1]
        let values = projected[0..., 0..., 2]

        var (qh, qw) = (h, w)
        if poolsQueries {
            // The queries pool in their spatial layout, which is what changes the block's output size.
            let spatial = queries.reshaped([batch, h, w, -1])
            let pooled = NFKMLXResample.maxPooled(spatial, kernel: 2, stride: 2)
            (qh, qw) = (pooled.shape[1], pooled.shape[2])
            queries = pooled.reshaped([batch, qh * qw, heads, -1])
        }

        let headDimensions = queries.shape[3]
        let scale = pow(Float(headDimensions), -0.5)
        let attention = softmax(matmul(queries.transposed(0, 2, 1, 3),
                                       keys.transposed(0, 2, 3, 1)) * scale, axis: -1)
        let attended = matmul(attention, values.transposed(0, 2, 1, 3))
        return proj(attended.transposed(0, 2, 1, 3).reshaped([batch, qh, qw, -1]))
    }
}

/// One Hiera block: pre-norm windowed attention, then a pre-norm MLP, with the shortcut projected and
/// pooled when the block changes width or resolution.
final class NFKSAM2Block: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NFKSAM2Attention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKSAM2MLP
    @ModuleInfo(key: "proj") var proj: Linear?
    let windowSize: Int
    let poolsQueries: Bool

    init(dimensions: Int, outDimensions: Int, heads: Int, windowSize: Int, poolsQueries: Bool) {
        _norm1.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-6)
        _attn.wrappedValue = NFKSAM2Attention(dimensions: dimensions, outDimensions: outDimensions,
                                              heads: heads, poolsQueries: poolsQueries)
        _norm2.wrappedValue = LayerNorm(dimensions: outDimensions, eps: 1e-6)
        _mlp.wrappedValue = NFKSAM2MLP(dimensions: outDimensions, hidden: outDimensions * 4)
        if dimensions != outDimensions {
            _proj.wrappedValue = Linear(dimensions, outDimensions)
        }
        self.windowSize = windowSize
        self.poolsQueries = poolsQueries
    }

    /// Splits `[N, H, W, C]` into `[N·windows, size, size, C]`, padding the bottom and right edges.
    static func partition(_ x: MLXArray, size: Int) -> (windows: MLXArray, padded: (Int, Int)) {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let padH = (size - h % size) % size
        let padW = (size - w % size) % size
        var padded = x
        if padH > 0 || padW > 0 {
            padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((0, padH)), IntOrPair((0, padW)),
                                            IntOrPair(0)], mode: .constant, value: MLXArray(Float(0)))
        }
        let (ph, pw) = (h + padH, w + padW)
        let windows = padded.reshaped([n, ph / size, size, pw / size, size, c])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([-1, size, size, c])
        return (windows, (ph, pw))
    }

    /// The inverse, cropping back to the pre-padding size.
    static func unpartition(_ windows: MLXArray, size: Int, padded: (Int, Int),
                            original: (Int, Int)) -> MLXArray {
        let (ph, pw) = padded
        let c = windows.shape[3]
        let n = windows.shape[0] / ((ph / size) * (pw / size))
        var x = windows.reshaped([n, ph / size, pw / size, size, size, c])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([n, ph, pw, c])
        if ph != original.0 || pw != original.1 {
            x = x[0..., 0 ..< original.0, 0 ..< original.1, 0...]
        }
        return x
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = norm1(x)
        var shortcut = x
        if let proj {
            // The shortcut takes the same projection and pooling the attention applies, so the
            // residual still lines up after a stage change.
            var projected = proj(normalized)
            if poolsQueries {
                projected = NFKMLXResample.maxPooled(projected, kernel: 2, stride: 2)
            }
            shortcut = projected
        }

        var attended: MLXArray
        if windowSize > 0 {
            let (height, width) = (normalized.shape[1], normalized.shape[2])
            let (windows, padded) = Self.partition(normalized, size: windowSize)
            let out = attn(windows)
            // After query pooling the windows are half the size, and the target extent is the
            // shortcut's — the reference recomputes the padded extent from it for exactly this reason.
            let size = poolsQueries ? windowSize / 2 : windowSize
            let target = poolsQueries ? (shortcut.shape[1], shortcut.shape[2]) : (height, width)
            let extent = poolsQueries
                ? (target.0 + (size - target.0 % size) % size, target.1 + (size - target.1 % size) % size)
                : padded
            attended = Self.unpartition(out, size: size, padded: extent, original: target)
        } else {
            attended = attn(normalized)
        }

        let residual = shortcut + attended
        return residual + mlp(norm2(residual))
    }
}

/// SAM 2 image-encoder sizing. `tiny` is the released `sam2_hiera_tiny` geometry.
public struct NFKMLXSAM2Configuration: Sendable {
    public var embedDimensions: Int
    public var heads: Int
    /// Blocks per stage; the width doubles and the resolution halves between stages.
    public var stages: [Int]
    /// Attention window per stage. A block listed in `globalAttentionBlocks` ignores it.
    public var windowSpec: [Int]
    public var globalAttentionBlocks: [Int]
    /// The learned position grid's size, before it is resampled to the patch grid.
    public var backgroundWindow: Int
    public var neckChannels: Int

    public init(embedDimensions: Int = 96, heads: Int = 1, stages: [Int] = [1, 2, 7, 2],
                windowSpec: [Int] = [8, 4, 14, 7], globalAttentionBlocks: [Int] = [5, 7, 9],
                backgroundWindow: Int = 7, neckChannels: Int = 256) {
        self.embedDimensions = embedDimensions
        self.heads = heads
        self.stages = stages
        self.windowSpec = windowSpec
        self.globalAttentionBlocks = globalAttentionBlocks
        self.backgroundWindow = backgroundWindow
        self.neckChannels = neckChannels
    }

    /// The released `sam2_hiera_tiny` geometry, which the defaults spell out.
    /// The released `sam2_hiera_tiny` geometry.
    public static let tiny = NFKMLXSAM2Configuration()

    /// The released `sam2_hiera_base_plus` geometry.
    ///
    /// Its config sets only the width and head count, so the stage depths, window sizes, and global
    /// attention blocks are the Hiera constructor's own defaults — which is why they are written out
    /// here rather than left implicit.
    public static let basePlus = NFKMLXSAM2Configuration(
        embedDimensions: 112, heads: 2, stages: [2, 3, 16, 3],
        windowSpec: [8, 4, 14, 7], globalAttentionBlocks: [12, 16, 20], backgroundWindow: 14)

    /// The released `sam2_hiera_large` geometry.
    ///
    /// Every axis differs from the defaults: 48 blocks weighted heavily toward the third stage, a
    /// coarser window at that stage, global attention much later, and a 7×7 background grid.
    public static let large = NFKMLXSAM2Configuration(
        embedDimensions: 144, heads: 2, stages: [2, 6, 36, 4],
        windowSpec: [8, 4, 16, 8], globalAttentionBlocks: [23, 33, 43], backgroundWindow: 7)

    /// The block index ending each stage.
    var stageEnds: [Int] {
        var ends = [Int]()
        var total = 0
        for count in stages {
            total += count
            ends.append(total - 1)
        }
        return ends
    }
}

/// The Hiera trunk: a patch embedding, a resampled position grid, and the block stack, returning the
/// feature map at the end of each stage.
final class NFKMLXSAM2Trunk: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: Conv2d
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ParameterInfo(key: "pos_embed_window") var posEmbedWindow: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [NFKSAM2Block]

    let configuration: NFKMLXSAM2Configuration

    init(_ c: NFKMLXSAM2Configuration) {
        configuration = c
        _patchEmbed.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.embedDimensions,
                                          kernelSize: 7, stride: 4, padding: 3)
        _posEmbed.wrappedValue = MLXArray.zeros([1, c.backgroundWindow, c.backgroundWindow, c.embedDimensions])
        _posEmbedWindow.wrappedValue = MLXArray.zeros([1, c.windowSpec[0], c.windowSpec[0], c.embedDimensions])

        var built = [NFKSAM2Block]()
        var dimensions = c.embedDimensions
        var heads = c.heads
        var stage = 1
        let ends = c.stageEnds
        // A stage's first block is the one that pools; it sits just past the previous stage's end.
        let poolBlocks = Set(ends.dropLast().map { $0 + 1 })
        for index in 0 ..< c.stages.reduce(0, +) {
            var outDimensions = dimensions
            // The window comes from the stage the block is *entering from*: the reference reads
            // `window_spec[cur_stage - 1]` before advancing the stage, so a transition block keeps the
            // previous stage's window. Reversing the two makes the pooled windows the wrong size and
            // the reassembly no longer tiles the output.
            let window = c.globalAttentionBlocks.contains(index) ? 0 : c.windowSpec[stage - 1]
            if index > 0, ends.contains(index - 1) {
                outDimensions = dimensions * 2
                heads *= 2
                stage += 1
            }
            built.append(NFKSAM2Block(dimensions: dimensions, outDimensions: outDimensions,
                                      heads: heads, windowSize: window,
                                      poolsQueries: poolBlocks.contains(index)))
            dimensions = outDimensions
        }
        _blocks.wrappedValue = built
    }

    /// The learned grid resampled to the patch grid, plus the window grid tiled over it.
    func positionEmbedding(height: Int, width: Int) -> MLXArray {
        let resampled = NFKMLXBicubic.resize(posEmbed, height: height, width: width)
        let window = posEmbedWindow
        return resampled + tiled(window, repetitions: [1, height / window.shape[1],
                                                       width / window.shape[2], 1])
    }

    /// The feature map at the end of every stage, each `[1, H, W, C]`.
    func callAsFunction(_ image: MLXArray) -> [MLXArray] {
        var x = patchEmbed(image)
        x = x + positionEmbedding(height: x.shape[1], width: x.shape[2])
        var outputs = [MLXArray]()
        let ends = configuration.stageEnds
        for (index, block) in blocks.enumerated() {
            x = block(x)
            if ends.contains(index) {
                outputs.append(x)
            }
        }
        return outputs
    }
}

/// The FPN neck: a 1×1 projection per captured stage, fused top-down on the deeper levels only.
final class NFKMLXSAM2Neck: Module {
    @ModuleInfo(key: "convs") var convs: [Conv2d]
    /// Levels that receive the coarser level's features; the finer ones stay lateral-only.
    let topDownLevels: Set<Int>

    init(channels: [Int], outChannels: Int, topDownLevels: Set<Int> = [2, 3]) {
        _convs.wrappedValue = channels.map {
            Conv2d(inputChannels: $0, outputChannels: outChannels, kernelSize: 1)
        }
        self.topDownLevels = topDownLevels
    }

    /// `features` runs fine to coarse; the result is in the same order.
    func callAsFunction(_ features: [MLXArray]) -> [MLXArray] {
        var out = [MLXArray?](repeating: nil, count: convs.count)
        var previous: MLXArray?
        let last = convs.count - 1
        for level in stride(from: last, through: 0, by: -1) {
            // The convolutions are stored coarse-to-fine while the features arrive fine-to-coarse.
            let lateral = convs[last - level](features[level])
            if topDownLevels.contains(level), let prior = previous {
                previous = lateral + NFKMLXResample.upsampleNearest(prior, scale: 2)
            } else {
                previous = lateral
            }
            out[level] = previous
        }
        return out.map { $0! }
    }
}

/// The SAM 2 image encoder: Hiera plus the FPN neck.
final class NFKMLXSAM2EncoderNet: Module {
    @ModuleInfo(key: "trunk") var trunk: NFKMLXSAM2Trunk
    @ModuleInfo(key: "neck") var neck: NFKMLXSAM2Neck
    /// How many of the coarsest levels the encoder discards, as the reference's `scalp` does.
    let scalp: Int

    init(_ c: NFKMLXSAM2Configuration = .tiny, scalp: Int = 1) {
        _trunk.wrappedValue = NFKMLXSAM2Trunk(c)
        // The neck is configured coarse-to-fine, matching how the reference lists its channels.
        let widths = (0 ..< c.stages.count).map { c.embedDimensions * (1 << $0) }
        _neck.wrappedValue = NFKMLXSAM2Neck(channels: widths.reversed(), outChannels: c.neckChannels)
        self.scalp = scalp
    }

    /// The FPN levels an image `[1, H, W, 3]` produces, finest first, with the coarsest `scalp`
    /// levels dropped. The last of them is what SAM 2 calls the vision features.
    func features(_ image: MLXArray) -> [MLXArray] {
        let levels = neck(trunk(image))
        return scalp > 0 ? Array(levels.dropLast(scalp)) : levels
    }

    /// The feature map the mask decoder reads.
    func visionFeatures(_ image: MLXArray) -> MLXArray {
        features(image).last!
    }
}

/// The SAM 2 networks — the Hiera image encoder, the prompt encoder and mask decoder, and the video
/// memory path — and their weight loading. One released checkpoint carries them all.
///
/// `NFKMLXSAM` remains the model wired to a backend; these are the SAM 2 ports at reference parity,
/// for a consumer assembling the video tracking loop.
@objc(NFKMLXSAM2)
public final class NFKMLXSAM2: NSObject {

    static func makeEncoder(_ configuration: NFKMLXSAM2Configuration = .tiny) -> NFKMLXSAM2EncoderNet {
        NFKMLXSAM2EncoderNet(configuration)
    }

    static func makeDecoder() -> NFKMLXSAM2Decoder { NFKMLXSAM2Decoder() }

    static func makeMemoryAttention() -> NFKMLXSAM2MemoryAttentionNet { NFKMLXSAM2MemoryAttentionNet() }

    static func makeMemoryEncoder() -> NFKMLXSAM2MemoryEncoderNet { NFKMLXSAM2MemoryEncoderNet() }

    /// Maps the checkpoint's memory-path names onto the module's. The mask downsampler is a numbered
    /// `Sequential` whose activations carry no parameters, and the fuser wraps its blocks in `layers`.
    static func remapMemoryKey(_ key: String) -> String? {
        if key.hasPrefix("memory_attention.") {
            return String(key.dropFirst("memory_attention.".count))
        }
        guard key.hasPrefix("memory_encoder.") else { return nil }
        var name = String(key.dropFirst("memory_encoder.".count))
        // The downsampler's Sequential runs convolution, normalization, activation per stage, so its
        // parameterized entries are 0/1, 3/4, 6/7, 9/10, and the final projection at 12.
        for (stage, slot) in [(0, 0), (1, 3), (2, 6), (3, 9)] {
            name = name.replacingOccurrences(of: "mask_downsampler.encoder.\(slot).",
                                             with: "mask_convs.\(stage).")
            name = name.replacingOccurrences(of: "mask_downsampler.encoder.\(slot + 1).",
                                             with: "mask_norms.\(stage).")
        }
        name = name.replacingOccurrences(of: "mask_downsampler.encoder.12.", with: "mask_out.")
        name = name.replacingOccurrences(of: "fuser.layers.", with: "fuser.")
        return name
    }

    /// Loads the memory attention and memory encoder from a converted checkpoint.
    static func loadMemoryWeights(into attention: NFKMLXSAM2MemoryAttentionNet,
                                  encoder: NFKMLXSAM2MemoryEncoderNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        var attentionArrays = [(String, MLXArray)]()
        var encoderArrays = [(String, MLXArray)]()
        for (key, value) in raw {
            guard let name = remapMemoryKey(key) else { continue }
            let mapped = checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
            if key.hasPrefix("memory_attention.") {
                attentionArrays.append((name, mapped))
            } else {
                encoderArrays.append((name, mapped))
            }
        }
        try NFKMLXWeights.apply(attentionArrays, to: attention)
        try NFKMLXWeights.apply(encoderArrays, to: encoder)
    }

    static func makePromptEncoder() -> NFKSAMPromptEncoder {
        NFKSAMPromptEncoder(NFKMLXSAMConfiguration.vitB)
    }

    /// Maps the checkpoint's decoder and prompt-encoder names onto the module's. The reference keeps
    /// its two-way transformer under a `transformer` submodule and its upscaling in a numbered
    /// `Sequential`; the cross-attention names are shortened the same way `NFKMLXSAM` shortens them.
    static func remapDecoderKey(_ key: String) -> String? {
        if key.hasPrefix("sam_prompt_encoder.") {
            var name = String(key.dropFirst("sam_prompt_encoder.".count))
            name = name.replacingOccurrences(of: "pe_layer.positional_encoding_gaussian_matrix",
                                             with: "position_encoding.gaussian")
            // The embeddings are `nn.Embedding`s upstream and plain parameters here.
            name = name.replacingOccurrences(of: ".weight", with: "")
            return name.hasPrefix("mask_downscaling") ? nil : name
        }
        guard key.hasPrefix("sam_mask_decoder.") else { return nil }
        var name = String(key.dropFirst("sam_mask_decoder.".count))
        name = name.replacingOccurrences(of: "transformer.layers.", with: "transformer_layers.")
        name = name.replacingOccurrences(of: "transformer.final_attn_token_to_image.",
                                         with: "final_attn_token_to_image.")
        name = name.replacingOccurrences(of: "transformer.norm_final_attn.", with: "norm_final_attn.")
        name = name.replacingOccurrences(of: ".cross_attn_token_to_image.", with: ".cross_token_image.")
        name = name.replacingOccurrences(of: ".cross_attn_image_to_token.", with: ".cross_image_token.")
        // The upscaling Sequential: 0 and 2 are the transposed convolutions, 1 the normalization.
        name = name.replacingOccurrences(of: "output_upscaling.0.", with: "upscale1.")
        name = name.replacingOccurrences(of: "output_upscaling.1.", with: "upscale_norm.")
        name = name.replacingOccurrences(of: "output_upscaling.3.", with: "upscale2.")
        name = name.replacingOccurrences(of: "output_hypernetworks_mlps.", with: "hyper.")
        name = name.replacingOccurrences(of: "iou_prediction_head.", with: "iou_head.")
        name = name.replacingOccurrences(of: "pred_obj_score_head.", with: "obj_score_head.")
        name = name.replacingOccurrences(of: "iou_token.weight", with: "iou_token")
        name = name.replacingOccurrences(of: "mask_tokens.weight", with: "mask_tokens")
        name = name.replacingOccurrences(of: "obj_score_token.weight", with: "obj_score_token")
        return name
    }

    /// Loads the decoder and prompt encoder from a converted checkpoint.
    static func loadDecoderWeights(into decoder: NFKMLXSAM2Decoder, prompt: NFKSAMPromptEncoder,
                                   from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        var decoderArrays = [(String, MLXArray)]()
        var promptArrays = [(String, MLXArray)]()
        for (key, value) in raw {
            guard let name = remapDecoderKey(key) else { continue }
            // A transposed convolution is stored `[in, out, kH, kW]`; the forward ones `[out, in, …]`.
            let mapped: MLXArray
            if checkpoint.needsConvTranspose, value.ndim == 4 {
                mapped = name.hasPrefix("upscale") ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1)
            } else {
                mapped = value
            }
            if key.hasPrefix("sam_prompt_encoder.") {
                promptArrays.append((name, mapped))
            } else {
                decoderArrays.append((name, mapped))
            }
        }
        try NFKMLXWeights.apply(decoderArrays, to: decoder)
        try NFKMLXWeights.apply(promptArrays, to: prompt)
    }

    /// Maps the released checkpoint's names onto the module's. The encoder's weights sit under an
    /// `image_encoder.` prefix; inside, the patch embedding and the neck's projections are each
    /// wrapped in a one-entry `Sequential`, and the MLP's layers are already an array.
    static func remapReferenceKey(_ key: String) -> String? {
        guard key.hasPrefix("image_encoder.") else { return nil }
        var key = String(key.dropFirst("image_encoder.".count))
        key = key.replacingOccurrences(of: "trunk.patch_embed.proj.", with: "trunk.patch_embed.")
        key = key.replacingOccurrences(of: "neck.convs.", with: "neck.convs.")
        key = key.replacingOccurrences(of: ".conv.weight", with: ".weight")
        key = key.replacingOccurrences(of: ".conv.bias", with: ".bias")
        return key
    }

    /// Loads a converted checkpoint into the encoder. Everything outside `image_encoder` — the prompt
    /// encoder, mask decoder, and the whole video memory path — is skipped rather than treated as an
    /// error, since this module is the encoder alone.
    static func loadWeights(into net: NFKMLXSAM2EncoderNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            guard let name = remapReferenceKey(key) else { return nil }
            if checkpoint.needsConvTranspose, value.ndim == 4 {
                // The position grids are stored `[1, C, H, W]` like a convolution weight, and the
                // same axis move puts them in this layout.
                return (name, value.transposed(0, 2, 3, 1))
            }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

// MARK: - Prompt encoder and mask decoder
//
// SAM 2's decoder is SAM's two-way transformer with three additions: an **object-score token** that
// leads the token sequence and its own prediction head, and **high-resolution features** taken from
// the FPN's finer levels and added during upscaling, which is what sharpens its mask boundaries.
// The attention, two-way block, MLP, and prompt encoder are shared with `NFKMLXSAM`.

/// The SAM 2 mask decoder.
final class NFKMLXSAM2Decoder: Module {
    @ParameterInfo(key: "obj_score_token") var objScoreToken: MLXArray
    @ParameterInfo(key: "iou_token") var iouToken: MLXArray
    @ParameterInfo(key: "mask_tokens") var maskTokens: MLXArray
    @ModuleInfo(key: "transformer_layers") var layers: [NFKSAMTwoWayBlock]
    @ModuleInfo(key: "final_attn_token_to_image") var finalAttention: NFKSAMAttention
    @ModuleInfo(key: "norm_final_attn") var normFinalAttention: LayerNorm
    @ModuleInfo(key: "upscale1") var upscale1: ConvTransposed2d
    @ModuleInfo(key: "upscale_norm") var upscaleNorm: LayerNorm
    @ModuleInfo(key: "upscale2") var upscale2: ConvTransposed2d
    @ModuleInfo(key: "conv_s0") var convS0: Conv2d
    @ModuleInfo(key: "conv_s1") var convS1: Conv2d
    @ModuleInfo(key: "hyper") var hyper: [NFKSAMMLP]
    @ModuleInfo(key: "iou_head") var iouHead: NFKSAMMLP
    @ModuleInfo(key: "obj_score_head") var objScoreHead: NFKSAMMLP

    let maskCount: Int

    init(dimensions: Int = 256, heads: Int = 8, maskCount: Int = 4, depth: Int = 2) {
        self.maskCount = maskCount
        _objScoreToken.wrappedValue = MLXArray.zeros([1, dimensions])
        _iouToken.wrappedValue = MLXArray.zeros([1, dimensions])
        _maskTokens.wrappedValue = MLXArray.zeros([maskCount, dimensions])
        _layers.wrappedValue = (0 ..< depth).map {
            NFKSAMTwoWayBlock(dim: dimensions, heads: heads, skipFirstLayerPE: $0 == 0)
        }
        _finalAttention.wrappedValue = NFKSAMAttention(dim: dimensions, heads: heads, downsample: 2)
        _normFinalAttention.wrappedValue = LayerNorm(dimensions: dimensions)
        _upscale1.wrappedValue = ConvTransposed2d(inputChannels: dimensions, outputChannels: dimensions / 4,
                                                  kernelSize: 2, stride: 2)
        _upscaleNorm.wrappedValue = LayerNorm(dimensions: dimensions / 4)
        _upscale2.wrappedValue = ConvTransposed2d(inputChannels: dimensions / 4, outputChannels: dimensions / 8,
                                                  kernelSize: 2, stride: 2)
        _convS0.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimensions / 8, kernelSize: 1)
        _convS1.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimensions / 4, kernelSize: 1)
        _hyper.wrappedValue = (0 ..< maskCount).map { _ in
            NFKSAMMLP(dim: dimensions, hidden: dimensions, out: dimensions / 8, layers: 3)
        }
        _iouHead.wrappedValue = NFKSAMMLP(dim: dimensions, hidden: dimensions, out: maskCount, layers: 3)
        _objScoreHead.wrappedValue = NFKSAMMLP(dim: dimensions, hidden: dimensions, out: 1, layers: 3)
    }

    /// Predicts the mask logits `[1, maskCount, 4H, 4W]`, the IoU scores `[1, maskCount]`, and the
    /// object score `[1, 1]`.
    ///
    /// - Parameters:
    ///   - features: the encoder's vision features `[1, H, W, C]`.
    ///   - positional: the image positional encoding `[1, H·W, C]`.
    ///   - sparse: the prompt's sparse tokens `[1, N, C]`.
    ///   - dense: the dense prompt embedding `[1, H, W, C]`.
    ///   - highResolution: the FPN's two finer levels, finest first.
    func callAsFunction(features: MLXArray, positional: MLXArray, sparse: MLXArray, dense: MLXArray,
                        highResolution: [MLXArray]) -> (masks: MLXArray, iou: MLXArray, objectScore: MLXArray) {
        // The object-score token leads, then the IoU token, then the mask tokens, then the prompt.
        let output = concatenated([objScoreToken, iouToken, maskTokens], axis: 0)
        var tokens = concatenated([output.reshaped([1, output.shape[0], output.shape[1]]), sparse], axis: 1)

        let (height, width) = (features.shape[1], features.shape[2])
        let channels = features.shape[3]
        var image = (features + dense).reshaped([1, height * width, channels])
        let imagePE = positional

        // The query positional term is the ORIGINAL token embedding at every layer and again at the
        // final attention — not the running queries, which change each layer. Feeding the running
        // ones drifts the masks without breaking anything visibly.
        let queryPE = tokens
        for layer in layers {
            (tokens, image) = layer(tokens, image, queryPE: queryPE, keyPE: imagePE)
        }
        let attended = finalAttention(tokens + queryPE, image + imagePE, image)
        tokens = normFinalAttention(tokens + attended)

        let iouOut = tokens[0..., 1]
        let maskOut = tokens[0..., 2 ..< (2 + maskCount)]

        // The high-resolution features join during upscaling, which is what SAM 2 adds over SAM. Their
        // 1×1 projections live here even though the reference applies them in its base model before
        // calling the decoder, so this takes the FPN levels as they come off the neck.
        var upscaled = upscale1(image.reshaped([1, height, width, channels])) + convS1(highResolution[1])
        upscaled = gelu(upscaleNorm(upscaled))
        upscaled = gelu(upscale2(upscaled) + convS0(highResolution[0]))

        let hyperIn = concatenated((0 ..< maskCount).map { index in
            hyper[index](maskOut[0..., index]).reshaped([1, 1, -1])
        }, axis: 1)
        let (uh, uw, uc) = (upscaled.shape[1], upscaled.shape[2], upscaled.shape[3])
        let flat = upscaled.reshaped([1, uh * uw, uc]).transposed(0, 2, 1)
        let masks = matmul(hyperIn, flat).reshaped([1, maskCount, uh, uw])
        return (masks, iouHead(iouOut), objScoreHead(tokens[0..., 0]))
    }
}

// MARK: - Video memory path
//
// What makes SAM 2 a tracker rather than a per-frame segmenter: the **memory encoder** turns a frame's
// features and its predicted mask into a compact memory, and **memory attention** conditions the next
// frame's features on those memories before the decoder ever sees them. Its attention carries axial
// **rotary** position embeddings — the spatial positions rotate query and key pairs rather than being
// added — which is the one piece with no counterpart anywhere else in this toolkit.

/// Axial rotary position embeddings: each pair of channels is rotated by an angle that depends on the
/// token's x or y coordinate, half the channels tracking each axis.
enum NFKMLXAxialRotary {
    /// Cosines and sines for an `endX × endY` grid, each `[endX·endY, dim/2]`.
    static func frequencies(dimensions: Int, endX: Int, endY: Int, theta: Float = 10000) -> (cos: MLXArray, sin: MLXArray) {
        let pairs = dimensions / 4                              // half the pairs per axis
        var cosines = [Float](repeating: 0, count: endX * endY * pairs * 2)
        var sines = [Float](repeating: 0, count: endX * endY * pairs * 2)
        for token in 0 ..< endX * endY {
            let x = Float(token % endX), y = Float(token / endX)
            for pair in 0 ..< pairs {
                let frequency = 1 / powf(theta, Float(pair * 4) / Float(dimensions))
                // The x frequencies occupy the first half of the pairs, the y ones the second.
                let base = token * pairs * 2
                cosines[base + pair] = cosf(x * frequency)
                sines[base + pair] = sinf(x * frequency)
                cosines[base + pairs + pair] = cosf(y * frequency)
                sines[base + pairs + pair] = sinf(y * frequency)
            }
        }
        let shape = [endX * endY, pairs * 2]
        return (cosines.withUnsafeBufferPointer { MLXArray($0, shape) },
                sines.withUnsafeBufferPointer { MLXArray($0, shape) })
    }

    /// Rotates `[batch, heads, tokens, dim]` in adjacent channel pairs.
    static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let shape = x.shape
        let pairs = shape[3] / 2
        let split = x.reshaped([shape[0], shape[1], shape[2], pairs, 2])
        let even = split[0..., 0..., 0..., 0..., 0]
        let odd = split[0..., 0..., 0..., 0..., 1]
        let c = cos.reshaped([1, 1, cos.shape[0], pairs])
        let s = sin.reshaped([1, 1, sin.shape[0], pairs])
        let rotatedEven = even * c - odd * s
        let rotatedOdd = even * s + odd * c
        return stacked([rotatedEven, rotatedOdd], axis: -1).reshaped(shape)
    }
}

/// Attention with optional rotary position embeddings on the queries and keys. The keys may be longer
/// than the queries — memory attention concatenates several frames — in which case the frequencies
/// repeat over them, and a trailing run of object-pointer tokens is excluded from the rotation.
final class NFKSAM2MemoryAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let heads: Int
    let usesRotary: Bool

    init(dimensions: Int, keyDimensions: Int, heads: Int, downsample: Int = 1, usesRotary: Bool = true) {
        let inner = dimensions / downsample
        _qProj.wrappedValue = Linear(dimensions, inner)
        _kProj.wrappedValue = Linear(keyDimensions, inner)
        _vProj.wrappedValue = Linear(keyDimensions, inner)
        _outProj.wrappedValue = Linear(inner, dimensions)
        self.heads = heads
        self.usesRotary = usesRotary
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, excludeFromRotary: Int = 0) -> MLXArray {
        let batch = q.shape[0]
        let inner = qProj.weight.shape[0]
        let headDimensions = inner / heads
        func split(_ x: MLXArray, _ projection: Linear) -> MLXArray {
            projection(x).reshaped([batch, x.shape[1], heads, headDimensions]).transposed(0, 2, 1, 3)
        }
        var queries = split(q, qProj)
        var keys = split(k, kProj)
        let values = split(v, vProj)

        if usesRotary {
            let side = Int(Double(queries.shape[2]).squareRoot().rounded())
            let (cos, sin) = NFKMLXAxialRotary.frequencies(dimensions: headDimensions, endX: side, endY: side)
            queries = NFKMLXAxialRotary.apply(queries, cos: cos, sin: sin)

            // Object-pointer tokens sit at the end of the keys and carry no spatial position, so the
            // rotation stops before them; the frequencies tile over whatever whole frames remain.
            let rotatable = keys.shape[2] - excludeFromRotary
            if rotatable > 0 {
                let repeats = max(rotatable / cos.shape[0], 1)
                let tiledCos = tiled(cos, repetitions: [repeats, 1])
                let tiledSin = tiled(sin, repetitions: [repeats, 1])
                let head = NFKMLXAxialRotary.apply(keys[0..., 0..., 0 ..< rotatable, 0...],
                                                   cos: tiledCos, sin: tiledSin)
                keys = rotatable == keys.shape[2]
                    ? head
                    : concatenated([head, keys[0..., 0..., rotatable..., 0...]], axis: 2)
            }
        }

        let scores = softmax(queries.matmul(keys.transposed(0, 1, 3, 2)) / sqrtf(Float(headDimensions)), axis: -1)
        let context = scores.matmul(values).transposed(0, 2, 1, 3).reshaped([batch, q.shape[1], inner])
        return outProj(context)
    }
}

/// One memory-attention layer: rotary self-attention over the frame, rotary cross-attention into the
/// memory, then an MLP — each pre-normalized and residual.
final class NFKSAM2MemoryLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKSAM2MemoryAttention
    @ModuleInfo(key: "cross_attn_image") var crossAttn: NFKSAM2MemoryAttention
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(dimensions: Int = 256, memoryDimensions: Int = 64, heads: Int = 1, hidden: Int = 2048) {
        _selfAttn.wrappedValue = NFKSAM2MemoryAttention(dimensions: dimensions, keyDimensions: dimensions,
                                                        heads: heads)
        // The cross attention reads a narrower memory and halves its internal width.
        _crossAttn.wrappedValue = NFKSAM2MemoryAttention(dimensions: dimensions,
                                                         keyDimensions: memoryDimensions,
                                                         heads: heads)
        _norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        _norm2.wrappedValue = LayerNorm(dimensions: dimensions)
        _norm3.wrappedValue = LayerNorm(dimensions: dimensions)
        _linear1.wrappedValue = Linear(dimensions, hidden)
        _linear2.wrappedValue = Linear(hidden, dimensions)
    }

    func callAsFunction(_ target: MLXArray, memory: MLXArray, queryPosition: MLXArray,
                        memoryPosition: MLXArray, objectPointerTokens: Int) -> MLXArray {
        var out = target
        let normalized = norm1(out)
        out = out + selfAttn(normalized, normalized, normalized)

        let queries = norm2(out)
        out = out + crossAttn(queries, memory + memoryPosition, memory,
                              excludeFromRotary: objectPointerTokens)
        return out + linear2(relu(linear1(norm3(out))))
    }
}

/// The memory-attention stack, conditioning a frame's features on the memories of earlier frames.
final class NFKMLXSAM2MemoryAttentionNet: Module {
    @ModuleInfo(key: "layers") var layers: [NFKSAM2MemoryLayer]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(dimensions: Int = 256, depth: Int = 4) {
        _layers.wrappedValue = (0 ..< depth).map { _ in NFKSAM2MemoryLayer(dimensions: dimensions) }
        _norm.wrappedValue = LayerNorm(dimensions: dimensions)
    }

    /// `current` and `memory` are `[1, tokens, C]` — batch-first, unlike the reference: its
    /// `MemoryAttention` takes SEQUENCE-first inputs and transposes them itself (`batch_first=True`
    /// describes its layers, not its inputs), so a comparison against it must transpose or every
    /// token attends only to itself. The reference also scales the current frame's own positional
    /// encoding by 0.1 before the stack, which is easy to miss and changes every layer.
    func callAsFunction(current: MLXArray, memory: MLXArray, currentPosition: MLXArray,
                        memoryPosition: MLXArray, objectPointerTokens: Int = 0) -> MLXArray {
        var out = current + 0.1 * currentPosition
        for layer in layers {
            out = layer(out, memory: memory, queryPosition: currentPosition,
                        memoryPosition: memoryPosition, objectPointerTokens: objectPointerTokens)
        }
        return norm(out)
    }
}

/// A ConvNeXt block: a depthwise 7×7, a channel-wise normalization, a widening and narrowing pair of
/// pointwise projections, and a learned per-channel scale on the residual.
final class NFKSAM2ConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var dwconv: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dimensions: Int) {
        _dwconv.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimensions,
                                      kernelSize: 7, padding: 3, groups: dimensions)
        _norm.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-6)
        _pwconv1.wrappedValue = Linear(dimensions, dimensions * 4)
        _pwconv2.wrappedValue = Linear(dimensions * 4, dimensions)
        _gamma.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = pwconv2(gelu(pwconv1(norm(dwconv(x)))))
        return x + gamma * out
    }
}

/// The memory encoder: downsample the predicted mask, add it to the projected frame features, fuse
/// with ConvNeXt blocks, and project to the memory's narrower width.
final class NFKMLXSAM2MemoryEncoderNet: Module {
    @ModuleInfo(key: "mask_convs") var maskConvs: [Conv2d]
    @ModuleInfo(key: "mask_norms") var maskNorms: [LayerNorm]
    @ModuleInfo(key: "mask_out") var maskOut: Conv2d
    @ModuleInfo(key: "pix_feat_proj") var pixelProjection: Conv2d
    @ModuleInfo(key: "fuser") var fuser: [NFKSAM2ConvNeXtBlock]
    @ModuleInfo(key: "out_proj") var outProjection: Conv2d

    init(dimensions: Int = 256, outDimensions: Int = 64, fuserDepth: Int = 2) {
        // The mask arrives at the full frame resolution and halves four times — a total stride of 16,
        // which is what lands it on the feature grid — its channels growing four-fold each time.
        let widths = [1, 4, 16, 64, 256]
        _maskConvs.wrappedValue = (0 ..< 4).map {
            Conv2d(inputChannels: widths[$0], outputChannels: widths[$0 + 1], kernelSize: 3,
                   stride: 2, padding: 1)
        }
        _maskNorms.wrappedValue = (1 ... 4).map { LayerNorm(dimensions: widths[$0], eps: 1e-6) }
        _maskOut.wrappedValue = Conv2d(inputChannels: 256, outputChannels: dimensions, kernelSize: 1)
        _pixelProjection.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimensions, kernelSize: 1)
        _fuser.wrappedValue = (0 ..< fuserDepth).map { _ in NFKSAM2ConvNeXtBlock(dimensions: dimensions) }
        _outProjection.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: outDimensions, kernelSize: 1)
    }

    /// `features` `[1, H, W, C]` and the mask at the full frame resolution `[1, 16H, 16W, 1]` → the memory
    /// `[1, H, W, outDimensions]`.
    func callAsFunction(features: MLXArray, maskLogits: MLXArray) -> MLXArray {
        var mask = sigmoid(maskLogits)
        for (conv, norm) in zip(maskConvs, maskNorms) {
            mask = gelu(norm(conv(mask)))
        }
        mask = maskOut(mask)

        var out = pixelProjection(features) + mask
        for block in fuser {
            out = block(out)
        }
        return outProjection(out)
    }
}
