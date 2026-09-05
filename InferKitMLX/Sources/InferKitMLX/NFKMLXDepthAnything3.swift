//
//  NFKMLXDepthAnything3.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Depth Anything 3 (monocular depth) — a DINOv2 ViT backbone and a DualDPT head. It runs as a single
// forward through `NFKMLXModuleBackend`: an RGB image in, a grayscale depth map out.
//
// The backbone is a variant of the Depth Anything V2 DINOv2, not a reuse: from block `altStart` (4 for
// the small release) it adds 2D rotary embeddings, per-head query/key normalization, a learned camera
// token injected into the class-token slot, and alternating local/global attention. A single image is
// one view, so the "global" (cross-view) blocks see uniform rotary positions, whose rotation cancels
// in the query·key product — they reduce to query/key-normalized self-attention. Each hooked layer's
// feature is the concatenation of the preceding local block's output and the current global block's
// output (`cat_token`), so the head reads `2 × embedDimensions`.
//
// The head is the V2 DPT lineage plus a token pre-LayerNorm, a UV positional embedding added to the
// feature maps, and an `exp`-depth / `exp+1`-confidence output convention (V2 emitted relative
// disparity). The reference DualDPT carries a second "ray" branch and a camera head for pose; neither
// is built here (monocular depth needs only the main branch), and their checkpoint tensors are named
// as deliberately unimplemented in `NFKMLXDepthAnything3Tests`.
//
// Reference parity is measured against the authors' `depth_anything_3` package. Tensors flow NHWC.

/// The DA3 ViT + DualDPT dimensions. Defaults are DA3-SMALL (DINOv2 ViT-Small).
public struct NFKMLXDepth3Configuration: Sendable {
    public var patchSize: Int = 14
    public var inputSize: Int = 518                                // 37 × 14
    public var embedDimensions: Int = 384
    public var depth: Int = 12
    public var heads: Int = 6
    public var mlpRatio: Int = 4
    /// The four encoder blocks whose outputs feed the head (DA3-SMALL: 5, 7, 9, 11).
    public var hooks: [Int] = [5, 7, 9, 11]
    /// The block from which rotary, query/key norm, camera token, and local/global alternation begin.
    public var altStart: Int = 4
    public var ropeStart: Int = 4
    public var qkNormStart: Int = 4
    public var ropeFrequency: Float = 100
    /// The DualDPT fusion width (DA3-SMALL: 64).
    public var features: Int = 64
    /// The per-hook reassemble widths (fine → coarse).
    public var outChannels: [Int] = [48, 96, 192, 384]

    public init() {}

    /// DA3-SMALL (the default): 384-d, 12 blocks, 6 heads.
    public static var small: NFKMLXDepth3Configuration { NFKMLXDepth3Configuration() }

    var tokenGrid: Int { inputSize / patchSize }
    /// The head reads concatenated local+global features (`cat_token`).
    var headInputDimensions: Int { embedDimensions * 2 }
}

// MARK: - 2D rotary

/// The DINOv2 2D rotary embedding (`RotaryPositionEmbedding2D`). Splits each head's channels in half:
/// the first half rotates by the token's row (y), the second by its column (x). Applied to the query
/// and key of shape `[batch, heads, tokens, headDim]`.
enum NFKDA3Rope2D {
    /// Builds the (cos, sin) tables for one spatial axis: `[maxPosition, halfHalf]`, where `halfHalf`
    /// is `headDim / 4` doubled back to `headDim / 2` as the reference concatenates `(angles, angles)`.
    private static func components(axisDim: Int, maxPosition: Int, frequency: Float) -> (MLXArray, MLXArray) {
        let count = axisDim / 2
        let exponents = (0 ..< count).map { Float($0 * 2) / Float(axisDim) }
        let invFreq = exponents.map { 1.0 / powf(frequency, $0) }
        var cosRows = [Float](), sinRows = [Float]()
        cosRows.reserveCapacity(maxPosition * axisDim)
        sinRows.reserveCapacity(maxPosition * axisDim)
        for position in 0 ..< maxPosition {
            var angles = invFreq.map { Float(position) * $0 }          // [count]
            angles += angles                                           // cat(angles, angles) -> [axisDim]
            cosRows += angles.map { cosf($0) }
            sinRows += angles.map { sinf($0) }
        }
        let cos = MLXArray(cosRows, [maxPosition, axisDim])
        let sin = MLXArray(sinRows, [maxPosition, axisDim])
        return (cos, sin)
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let dim = x.shape[x.ndim - 1]
        let x1 = x[.ellipsis, 0 ..< dim / 2]
        let x2 = x[.ellipsis, dim / 2 ..< dim]
        return concatenated([-x2, x1], axis: -1)
    }

    /// Applies the embedding to `tokens` `[batch, heads, N, headDim]` given `positions` `[N, 2]` (y, x).
    static func apply(_ tokens: MLXArray, positionsY: [Int], positionsX: [Int], frequency: Float) -> MLXArray {
        let headDim = tokens.shape[tokens.ndim - 1]
        let axisDim = headDim / 2
        let maxPosition = (max(positionsY.max() ?? 0, positionsX.max() ?? 0)) + 1
        let (cos, sin) = components(axisDim: axisDim, maxPosition: maxPosition, frequency: frequency)

        let vertical = tokens[.ellipsis, 0 ..< axisDim]
        let horizontal = tokens[.ellipsis, axisDim ..< headDim]

        func rope1D(_ x: MLXArray, _ positions: [Int]) -> MLXArray {
            let idx = MLXArray(positions.map { Int32($0) })
            let c = cos[idx].reshaped([1, 1, positions.count, axisDim])   // broadcast over batch, heads
            let s = sin[idx].reshaped([1, 1, positions.count, axisDim])
            return (x * c) + (rotateHalf(x) * s)
        }

        let v = rope1D(vertical, positionsY)
        let h = rope1D(horizontal, positionsX)
        return concatenated([v, h], axis: -1)
    }
}

// MARK: - Backbone blocks

/// Multi-head self-attention with optional query/key norm and 2D rotary (`attn.qkv`, `attn.proj`,
/// and `attn.q_norm` / `attn.k_norm` where present).
final class NFKDA3Attention: Module {
    let qkv: Linear
    let proj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm?
    let heads: Int
    let frequency: Float

    init(dimensions: Int, heads: Int, qkNorm: Bool, frequency: Float) {
        self.heads = heads
        self.frequency = frequency
        qkv = Linear(dimensions, dimensions * 3, bias: true)
        proj = Linear(dimensions, dimensions, bias: true)
        if qkNorm {
            _qNorm.wrappedValue = LayerNorm(dimensions: dimensions / heads)
            _kNorm.wrappedValue = LayerNorm(dimensions: dimensions / heads)
        }
    }

    /// - Parameter rope: (positionsY, positionsX) per token, or nil to skip rotary.
    func callAsFunction(_ x: MLXArray, rope: ([Int], [Int])?) -> MLXArray {
        let (batch, tokens, dimensions) = (x.shape[0], x.shape[1], x.shape[2])
        let headDim = dimensions / heads
        let scale = 1.0 / sqrtf(Float(headDim))

        let fused = qkv(x).reshaped([batch, tokens, 3, heads, headDim]).transposed(2, 0, 3, 1, 4)
        let parts = fused.split(parts: 3, axis: 0)
        var q = parts[0].reshaped([batch, heads, tokens, headDim])
        var k = parts[1].reshaped([batch, heads, tokens, headDim])
        let v = parts[2].reshaped([batch, heads, tokens, headDim])

        if let qNorm, let kNorm {
            q = qNorm(q)
            k = kNorm(k)
        }
        if let (py, px) = rope {
            q = NFKDA3Rope2D.apply(q, positionsY: py, positionsX: px, frequency: frequency)
            k = NFKDA3Rope2D.apply(k, positionsY: py, positionsX: px, frequency: frequency)
        }

        let scores = softmax((q * scale).matmul(k.transposed(0, 1, 3, 2)), axis: -1)
        let context = scores.matmul(v)
            .transposed(0, 2, 1, 3)
            .reshaped([batch, tokens, dimensions])
        return proj(context)
    }
}

/// The per-channel learned scale DINOv2 applies after attention and the MLP (`ls1.gamma`, `ls2.gamma`).
final class NFKDA3LayerScale: Module {
    let gamma: MLXArray
    init(dimensions: Int) { gamma = MLXArray.ones([dimensions]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

/// The block MLP (`mlp.fc1`, `mlp.fc2`).
final class NFKDA3MLP: Module {
    let fc1: Linear
    let fc2: Linear
    init(dimensions: Int, hidden: Int) {
        fc1 = Linear(dimensions, hidden, bias: true)
        fc2 = Linear(hidden, dimensions, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// One transformer block: norm → attention → scale (residual), norm → MLP → scale (residual). The
/// LayerNorm epsilon is 1e-6 (the DA3 default), not MLXNN's 1e-5.
final class NFKDA3Block: Module {
    let norm1: LayerNorm
    let attn: NFKDA3Attention
    let ls1: NFKDA3LayerScale
    let norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKDA3MLP
    let ls2: NFKDA3LayerScale

    init(dimensions: Int, heads: Int, mlpRatio: Int, qkNorm: Bool, frequency: Float) {
        norm1 = LayerNorm(dimensions: dimensions, eps: 1e-6)
        attn = NFKDA3Attention(dimensions: dimensions, heads: heads, qkNorm: qkNorm, frequency: frequency)
        ls1 = NFKDA3LayerScale(dimensions: dimensions)
        norm2 = LayerNorm(dimensions: dimensions, eps: 1e-6)
        _mlp.wrappedValue = NFKDA3MLP(dimensions: dimensions, hidden: dimensions * mlpRatio)
        ls2 = NFKDA3LayerScale(dimensions: dimensions)
    }

    func callAsFunction(_ x: MLXArray, rope: ([Int], [Int])?) -> MLXArray {
        var out = x + ls1(attn(norm1(x), rope: rope))
        out = out + ls2(mlp(norm2(out)))
        return out
    }
}

/// The DINOv2 patch-embedding convolution (`patch_embed.proj`).
final class NFKDA3PatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    init(patchSize: Int, dimensions: Int) {
        _proj.wrappedValue = Conv2d(inputChannels: 3, outputChannels: dimensions,
                                    kernelSize: IntOrPair(patchSize), stride: IntOrPair(patchSize))
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

/// The DINOv2 ViT backbone (`pretrained.*`). Returns the token features at the configured hook blocks,
/// each `[1, grid*grid, 2*dimensions]` (concatenated local+global, class token dropped).
final class NFKDA3Encoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: NFKDA3PatchEmbed
    @ModuleInfo(key: "cls_token") var clsToken: MLXArray
    @ModuleInfo(key: "camera_token") var cameraToken: MLXArray
    @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray
    let blocks: [NFKDA3Block]
    let norm: LayerNorm

    private let configuration: NFKMLXDepth3Configuration

    init(_ configuration: NFKMLXDepth3Configuration) {
        self.configuration = configuration
        let dimensions = configuration.embedDimensions
        let grid = configuration.tokenGrid
        _patchEmbed.wrappedValue = NFKDA3PatchEmbed(patchSize: configuration.patchSize, dimensions: dimensions)
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, dimensions])
        _cameraToken.wrappedValue = MLXArray.zeros([1, 2, dimensions])
        _posEmbed.wrappedValue = MLXArray.zeros([1, grid * grid + 1, dimensions])
        blocks = (0 ..< configuration.depth).map { index in
            NFKDA3Block(dimensions: dimensions, heads: configuration.heads, mlpRatio: configuration.mlpRatio,
                        qkNorm: index >= configuration.qkNormStart, frequency: configuration.ropeFrequency)
        }
        norm = LayerNorm(dimensions: dimensions)
    }

    /// The rotary positions for the whole token sequence (class token at 0). `uniform` collapses every
    /// patch to the same position, which is the cross-view ("global") case for a single image.
    private func positions(uniform: Bool) -> ([Int], [Int]) {
        let grid = configuration.tokenGrid
        var y = [0], x = [0]                                            // class/camera token at (0, 0)
        for row in 0 ..< grid {
            for col in 0 ..< grid {
                y.append(uniform ? 1 : row + 1)
                x.append(uniform ? 1 : col + 1)
            }
        }
        return (y, x)
    }

    /// - Parameter image: `[1, inputSize, inputSize, 3]`.
    /// - Returns: the hooked features, each `[1, grid*grid, 2*dimensions]`.
    func hookedFeatures(_ image: MLXArray) -> [MLXArray] {
        let dimensions = configuration.embedDimensions
        let grid = configuration.tokenGrid
        let patches = patchEmbed(image)                                 // [1, grid, grid, dimensions]
        var tokens = patches.reshaped([1, grid * grid, dimensions])
        tokens = concatenated([clsToken, tokens], axis: 1) + posEmbed   // [1, N+1, dimensions]

        let localPositions = positions(uniform: false)
        let globalPositions = positions(uniform: true)
        let hookSet = Set(configuration.hooks)

        var localOutput = tokens
        var outputs = [MLXArray]()
        for (index, block) in blocks.enumerated() {
            // The camera token replaces the class-token slot at `altStart`, before that block runs.
            if index == configuration.altStart {
                let camera = cameraToken[0..., 0 ..< 1, 0...]           // ref view token [1, 1, dim]
                tokens = concatenated([camera, tokens[0..., 1..., 0...]], axis: 1)
            }
            let isGlobal = index >= configuration.altStart && index % 2 == 1
            let rope: ([Int], [Int])?
            if index < configuration.ropeStart {
                rope = nil
            } else {
                rope = isGlobal ? globalPositions : localPositions
            }
            tokens = block(tokens, rope: rope)
            if !isGlobal {
                localOutput = tokens
            }
            if hookSet.contains(index) {
                // cat_token: the preceding local block's output beside the current global block's,
                // the final LayerNorm applied only to the global half (the reference's `norm=True`
                // over the second half). The class token is then dropped.
                let combined = concatenated([localOutput, norm(tokens)], axis: -1)
                outputs.append(combined[0..., 1..., 0...])
            }
        }
        return outputs
    }
}

// MARK: - Head (DualDPT depth branch)

/// A DPT residual convolution unit (`resConfUnitN`): relu → conv → relu → conv, added to the input.
final class NFKDA3ResidualUnit: Module {
    let conv1: Conv2d
    let conv2: Conv2d
    init(features: Int) {
        conv1 = Conv2d(inputChannels: features, outputChannels: features, kernelSize: 3, padding: 1)
        conv2 = Conv2d(inputChannels: features, outputChannels: features, kernelSize: 3, padding: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + conv2(relu(conv1(relu(x))))
    }
}

/// A DPT feature-fusion block (`refinenetN`): optionally add a same-resolution skip through
/// `resConfUnit1`, refine through `resConfUnit2`, resize (bilinear, align-corners), and shrink 1×1.
final class NFKDA3Fusion: Module {
    @ModuleInfo(key: "resConfUnit1") var unit1: NFKDA3ResidualUnit?
    @ModuleInfo(key: "resConfUnit2") var unit2: NFKDA3ResidualUnit
    @ModuleInfo(key: "out_conv") var outConv: Conv2d

    init(features: Int, hasResidual: Bool) {
        if hasResidual {
            _unit1.wrappedValue = NFKDA3ResidualUnit(features: features)
        }
        _unit2.wrappedValue = NFKDA3ResidualUnit(features: features)
        _outConv.wrappedValue = Conv2d(inputChannels: features, outputChannels: features, kernelSize: 1)
    }

    /// - Parameter size: the target (height, width), or nil to upsample by 2×.
    func callAsFunction(_ x: MLXArray, skip: MLXArray?, size: (Int, Int)?) -> MLXArray {
        var out = x
        if let skip, let unit1 {
            out = out + unit1(skip)
        }
        out = unit2(out)
        let target = size ?? (out.shape[1] * 2, out.shape[2] * 2)
        out = NFKDA3Resample.bilinearAlignCorners(out, height: target.0, width: target.1)
        return outConv(out)
    }
}

/// The DualDPT head, depth branch only. The `_aux` fusion chain, the aux heads, and the camera
/// decoder/encoder of the reference are not built (monocular depth needs the main branch alone).
final class NFKDA3Head: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "projects") var projects: [Conv2d]
    @ModuleInfo(key: "resize_layers") var resizeLayers: [Module]
    @ModuleInfo(key: "scratch") var scratch: NFKDA3Scratch

    private let configuration: NFKMLXDepth3Configuration

    init(_ configuration: NFKMLXDepth3Configuration) {
        self.configuration = configuration
        let dimensions = configuration.headInputDimensions
        let out = configuration.outChannels
        _norm.wrappedValue = LayerNorm(dimensions: dimensions)
        _projects.wrappedValue = out.map { Conv2d(inputChannels: dimensions, outputChannels: $0, kernelSize: 1) }
        _resizeLayers.wrappedValue = [
            ConvTransposed2d(inputChannels: out[0], outputChannels: out[0], kernelSize: 4, stride: 4),
            ConvTransposed2d(inputChannels: out[1], outputChannels: out[1], kernelSize: 2, stride: 2),
            Identity(),
            Conv2d(inputChannels: out[3], outputChannels: out[3], kernelSize: 3, stride: 2, padding: 1),
        ]
        _scratch.wrappedValue = NFKDA3Scratch(features: configuration.features, outChannels: out)
    }

    /// - Parameter features: four hooked token features, each `[1, grid*grid, headInputDimensions]`.
    /// - Returns: `(depth, confidence)`, each `[1, H, W]` at the input resolution.
    func callAsFunction(_ features: [MLXArray]) -> (MLXArray, MLXArray) {
        let grid = configuration.tokenGrid
        let width = configuration.inputSize
        let height = configuration.inputSize
        var pyramid = [MLXArray]()
        for (index, tokens) in features.enumerated() {
            let dimensions = tokens.shape[2]
            var x = norm(tokens)
            x = x.reshaped([1, grid, grid, dimensions])                 // NHWC
            x = projects[index](x)
            x = NFKDA3Head.addUVPositionEmbedding(x, width: width, height: height)
            x = (resizeLayers[index] as? UnaryLayer)?.callAsFunction(x) ?? x
            pyramid.append(x)
        }

        var fused = scratch.fuse(pyramid)                               // includes output_conv1
        fused = NFKDA3Resample.bilinearAlignCorners(fused, height: height, width: width)
        fused = NFKDA3Head.addUVPositionEmbedding(fused, width: width, height: height)
        let logits = scratch.outputConv2Forward(fused)                  // [1, H, W, 2]

        let depth = exp(logits[0..., 0..., 0..., 0])
        let confidence = exp(logits[0..., 0..., 0..., 1]) + 1
        return (depth, confidence)
    }

    /// The head's intermediate seams (NHWC), for reference-parity localization.
    func seams(_ features: [MLXArray]) -> [String: MLXArray] {
        let grid = configuration.tokenGrid
        let width = configuration.inputSize, height = configuration.inputSize
        var out = [String: MLXArray]()
        var pyramid = [MLXArray]()
        for (index, tokens) in features.enumerated() {
            var x = norm(tokens).reshaped([1, grid, grid, tokens.shape[2]])
            x = projects[index](x)
            x = NFKDA3Head.addUVPositionEmbedding(x, width: width, height: height)
            x = (resizeLayers[index] as? UnaryLayer)?.callAsFunction(x) ?? x
            out["stage\(index)"] = x
            pyramid.append(x)
        }
        let fused = scratch.fuse(pyramid)
        out["fused"] = fused
        var head = NFKDA3Resample.bilinearAlignCorners(fused, height: height, width: width)
        head = NFKDA3Head.addUVPositionEmbedding(head, width: width, height: height)
        out["logits"] = scratch.outputConv2Forward(head)
        return out
    }

    /// The UV positional embedding the head adds to a feature map (`_add_pos_embed`, ratio 0.1). A
    /// normalized UV grid over the current spatial size is turned into a sin/cos embedding of the
    /// feature width and added.
    static func addUVPositionEmbedding(_ x: MLXArray, width: Int, height: Int, ratio: Float = 0.1) -> MLXArray {
        let (ph, pw, channels) = (x.shape[1], x.shape[2], x.shape[3])
        let aspect = Float(width) / Float(height)
        let embed = NFKDA3Head.uvEmbedding(gridWidth: pw, gridHeight: ph, aspect: aspect, channels: channels)
        return x + (embed.reshaped([1, ph, pw, channels]) * ratio)
    }

    /// `create_uv_grid` + `position_grid_to_embed` (omega_0 = 100), returning `[ph, pw, channels]`.
    private static func uvEmbedding(gridWidth pw: Int, gridHeight ph: Int, aspect: Float, channels: Int) -> MLXArray {
        let diag = sqrtf(aspect * aspect + 1)
        let spanX = aspect / diag
        let spanY = 1 / diag
        func linspace(_ lo: Float, _ hi: Float, _ n: Int) -> [Float] {
            guard n > 1 else { return [lo] }
            return (0 ..< n).map { lo + (hi - lo) * Float($0) / Float(n - 1) }
        }
        let xs = linspace(-spanX * Float(pw - 1) / Float(pw), spanX * Float(pw - 1) / Float(pw), pw)
        let ys = linspace(-spanY * Float(ph - 1) / Float(ph), spanY * Float(ph - 1) / Float(ph), ph)

        // create_uv_grid uses meshgrid(indexing="xy") over (x_coords[pw], y_coords[ph]) and stacks
        // (u, v); position_grid_to_embed then reads it as [H=ph, W=pw, 2] row-major. The u channel is
        // the x coordinate (varies along width), the v channel the y coordinate (varies along height).
        let half = channels / 2
        func sincos(_ value: Float) -> [Float] {
            // make_sincos_pos_embed(half, value, omega_0=100): [sin(value*omega)…, cos(value*omega)…]
            let d = half / 2
            var out = [Float]()
            out.reserveCapacity(half)
            for j in 0 ..< d { out.append(sinf(value / powf(100, Float(j) / Float(d)))) }
            for j in 0 ..< d { out.append(cosf(value / powf(100, Float(j) / Float(d)))) }
            return out
        }
        var rows = [Float]()
        rows.reserveCapacity(ph * pw * channels)
        for row in 0 ..< ph {
            for col in 0 ..< pw {
                rows += sincos(xs[col])                                 // emb_x from the u/x coordinate
                rows += sincos(ys[row])                                 // emb_y from the v/y coordinate
            }
        }
        return MLXArray(rows, [ph, pw, channels])
    }
}

/// The DPT `scratch`: per-level input convolutions, four fusion blocks, and the output convolutions.
final class NFKDA3Scratch: Module {
    @ModuleInfo(key: "layer1_rn") var layer1: Conv2d
    @ModuleInfo(key: "layer2_rn") var layer2: Conv2d
    @ModuleInfo(key: "layer3_rn") var layer3: Conv2d
    @ModuleInfo(key: "layer4_rn") var layer4: Conv2d
    @ModuleInfo(key: "refinenet1") var refine1: NFKDA3Fusion
    @ModuleInfo(key: "refinenet2") var refine2: NFKDA3Fusion
    @ModuleInfo(key: "refinenet3") var refine3: NFKDA3Fusion
    @ModuleInfo(key: "refinenet4") var refine4: NFKDA3Fusion
    @ModuleInfo(key: "output_conv1") var outputConv1: Conv2d
    @ModuleInfo(key: "output_conv2") var outputConv2: [Module]

    init(features: Int, outChannels: [Int]) {
        _layer1.wrappedValue = Conv2d(inputChannels: outChannels[0], outputChannels: features, kernelSize: 3, padding: 1, bias: false)
        _layer2.wrappedValue = Conv2d(inputChannels: outChannels[1], outputChannels: features, kernelSize: 3, padding: 1, bias: false)
        _layer3.wrappedValue = Conv2d(inputChannels: outChannels[2], outputChannels: features, kernelSize: 3, padding: 1, bias: false)
        _layer4.wrappedValue = Conv2d(inputChannels: outChannels[3], outputChannels: features, kernelSize: 3, padding: 1, bias: false)
        _refine1.wrappedValue = NFKDA3Fusion(features: features, hasResidual: true)
        _refine2.wrappedValue = NFKDA3Fusion(features: features, hasResidual: true)
        _refine3.wrappedValue = NFKDA3Fusion(features: features, hasResidual: true)
        _refine4.wrappedValue = NFKDA3Fusion(features: features, hasResidual: false)
        _outputConv1.wrappedValue = Conv2d(inputChannels: features, outputChannels: features / 2, kernelSize: 3, padding: 1)
        // A Sequential(conv, relu, conv) in the reference, so the convolutions carry indices 0 and 2.
        _outputConv2.wrappedValue = [
            Conv2d(inputChannels: features / 2, outputChannels: 32, kernelSize: 3, padding: 1),
            ReLU(),
            Conv2d(inputChannels: 32, outputChannels: 2, kernelSize: 1),
        ]
    }

    private func size(_ x: MLXArray) -> (Int, Int) { (x.shape[1], x.shape[2]) }

    /// Fuses the reassembled pyramid coarse → fine and applies `output_conv1`, as the reference `_fuse`
    /// does. `pyramid` is fine → coarse (`layer1` … `layer4`).
    func fuse(_ pyramid: [MLXArray]) -> MLXArray {
        let l1 = layer1(pyramid[0])
        let l2 = layer2(pyramid[1])
        let l3 = layer3(pyramid[2])
        let l4 = layer4(pyramid[3])

        var path = refine4(l4, skip: nil, size: size(l3))
        path = refine3(path, skip: l3, size: size(l2))
        path = refine2(path, skip: l2, size: size(l1))
        path = refine1(path, skip: l1, size: nil)                      // upsample ×2
        return outputConv1(path)
    }

    func outputConv2Forward(_ x: MLXArray) -> MLXArray {
        let conv0 = outputConv2[0] as! Conv2d
        let conv2 = outputConv2[2] as! Conv2d
        return conv2(relu(conv0(x)))
    }
}

// MARK: - Resample

enum NFKDA3Resample {
    /// Resamples `[N, H, W, C]` to `height` × `width` by bilinear interpolation with
    /// `align_corners=true` (the DA3 head and fusion default), matching `custom_interpolate`.
    static func bilinearAlignCorners(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        if h == height && w == width { return x }
        func axis(_ outSize: Int, _ inSize: Int) -> (MLXArray, MLXArray, [Float]) {
            var lo = [Int32](), hi = [Int32](), frac = [Float]()
            let scale = outSize > 1 ? Float(inSize - 1) / Float(outSize - 1) : 0
            for o in 0 ..< outSize {
                let src = Float(o) * scale
                let low = min(Int(src.rounded(.down)), inSize - 1)
                lo.append(Int32(low))
                hi.append(Int32(min(low + 1, inSize - 1)))
                frac.append(src - Float(low))
            }
            return (MLXArray(lo), MLXArray(hi), frac)
        }
        let (y0, y1, yf) = axis(height, h)
        let (x0, x1, xf) = axis(width, w)
        let rows0 = x[0..., y0], rows1 = x[0..., y1]
        let c00 = rows0[0..., 0..., x0], c01 = rows0[0..., 0..., x1]
        let c10 = rows1[0..., 0..., x0], c11 = rows1[0..., 0..., x1]
        let wy = MLXArray(yf).reshaped([1, height, 1, 1])
        let wx = MLXArray(xf).reshaped([1, 1, width, 1])
        let top = c00 * (1 - wx) + c01 * wx
        let bottom = c10 * (1 - wx) + c11 * wx
        return (top * (1 - wy) + bottom * wy).reshaped([n, height, width, c])
    }
}

// MARK: - Model + backend

@objc(NFKMLXDepthAnything3)
public final class NFKMLXDepthAnything3: NSObject {

    @objc public static let modelName = "depth-anything-3-small"

    static func makeNet(_ configuration: NFKMLXDepth3Configuration = .small) -> NFKMLXDepthAnything3Net {
        NFKMLXDepthAnything3Net(configuration)
    }

    /// Builds a Depth Anything 3 backend from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true). Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXDepthAnything3Net(.small)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXDepth3Holder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in holder.net.depth(image) }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required. Blocking on the
    /// network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a checkpoint, keeping only the monocular depth path: the backbone (`model.backbone.` →
    /// `pretrained.`) and the DualDPT main branch (`model.head.`), transposing 4-D convolution weights
    /// to MLX's channels-last layout. The aux/ray fusion chain, the aux heads, and the camera
    /// decoder/encoder are dropped (a nil remap skips them).
    static func loadWeights(into net: NFKMLXDepthAnything3Net, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        // The transposed-convolution resize layers (`resize_layers.0` and `.1`) store their weight as
        // PyTorch `[C_in, C_out, kH, kW]`, which MLX's ConvTransposed2d reads as `[C_out, kH, kW, C_in]`
        // — a different axis order from a regular convolution's `[out, kH, kW, in]`. Both are square
        // (in == out), so the regular-convolution transpose still loads but scrambles the kernel.
        let convTranspose: Set<String> = ["head.resize_layers.0.weight", "head.resize_layers.1.weight"]
        var mapped = [(String, MLXArray)]()
        for (key, value) in checkpoint.arrays {
            guard let remapped = remap(key) else { continue }
            let array: MLXArray
            if checkpoint.needsConvTranspose && value.ndim == 4 {
                array = convTranspose.contains(remapped) ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1)
            } else {
                array = value
            }
            mapped.append((remapped, array))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Maps a checkpoint key onto the built module, or nil to drop a tensor the depth path does not use.
    static func remap(_ key: String) -> String? {
        if key.hasPrefix("model.backbone.pretrained.") {
            return "backbone." + key.dropFirst("model.backbone.pretrained.".count)
        }
        if key.hasPrefix("model.head.") {
            let tail = String(key.dropFirst("model.head.".count))
            // Drop the aux fusion chain and aux heads (monocular depth uses the main branch only).
            if tail.contains("_aux") { return nil }
            return "head." + tail
        }
        return nil                                                     // model.cam_enc / model.cam_dec
    }
}

/// The full model: backbone (`backbone`) + DualDPT depth head (`head`).
final class NFKMLXDepthAnything3Net: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKDA3Encoder
    @ModuleInfo(key: "head") var head: NFKDA3Head

    let configuration: NFKMLXDepth3Configuration

    init(_ configuration: NFKMLXDepth3Configuration) {
        self.configuration = configuration
        _backbone.wrappedValue = NFKDA3Encoder(configuration)
        _head.wrappedValue = NFKDA3Head(configuration)
    }

    /// The raw network on a prepared `[1, inputSize, inputSize, 3]` image: the four hooked features.
    func features(_ image: MLXArray) -> [MLXArray] { backbone.hookedFeatures(image) }

    /// Maps a bridged image `[H, W, 3]` (`0...1`) to a grayscale depth image `[H, W, 3]` (`0...1`),
    /// near = bright. The encoder runs at the fixed input size; the map resizes back and normalizes.
    func depth(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let resized = NFKDA3Resample.bilinearAlignCorners(
            image.reshaped([1, height, width, image.shape[2]]),
            height: configuration.inputSize, width: configuration.inputSize)
        // The pipeline normalizes with ImageNet statistics before the backbone (`_normalize_image`).
        // The resize here is bilinear where the reference uses a PIL resize, a documented consumer
        // approximation; the network is at parity on the reference's own pixel values.
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225])
        let prepared = (resized - mean) / standardDeviation
        let (depthMap, _) = head(backbone.hookedFeatures(prepared))    // [1, h, w]
        let full = NFKDA3Resample.bilinearAlignCorners(
            depthMap.reshaped([1, configuration.inputSize, configuration.inputSize, 1]),
            height: height, width: width)

        let minimum = full.min()
        let span = maximum(full.max() - minimum, MLXArray(1e-6))
        let normalized = ((full - minimum) / span).reshaped([height, width, 1])
        return concatenated([normalized, normalized, normalized], axis: 2)
    }
}

private final class NFKMLXDepth3Holder: @unchecked Sendable {
    let net: NFKMLXDepthAnything3Net
    init(_ net: NFKMLXDepthAnything3Net) { self.net = net }
}
