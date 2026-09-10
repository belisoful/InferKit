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
    /// The camera encoder's trunk depth and head count. Every release carries the reference's defaults
    /// and overrides only the width, which is the backbone's.
    public var cameraTrunkDepth: Int = 4
    public var cameraHeads: Int = 16

    public init() {}

    /// DA3-SMALL (the default): 384-d, 12 blocks, 6 heads.
    public static var small: NFKMLXDepth3Configuration { NFKMLXDepth3Configuration() }

    /// DA3-BASE (`vitb`): 768-d, 12 blocks, 12 heads, the same hooks and starts as small, a 128-wide
    /// DualDPT reading `[96, 192, 384, 768]`.
    public static var base: NFKMLXDepth3Configuration {
        var configuration = NFKMLXDepth3Configuration()
        configuration.embedDimensions = 768
        configuration.heads = 12
        configuration.features = 128
        configuration.outChannels = [96, 192, 384, 768]
        return configuration
    }

    /// DA3-LARGE (`vitl`): 1024-d, 24 blocks, 16 heads. Its rotary, query/key norm, camera token, and
    /// local/global alternation begin at block 8 rather than 4, and its hooks sit at 11/15/19/23; the
    /// DualDPT is 256 wide over `[256, 512, 1024, 1024]`.
    public static var large: NFKMLXDepth3Configuration {
        var configuration = NFKMLXDepth3Configuration()
        configuration.embedDimensions = 1024
        configuration.depth = 24
        configuration.heads = 16
        configuration.hooks = [11, 15, 19, 23]
        configuration.altStart = 8
        configuration.ropeStart = 8
        configuration.qkNormStart = 8
        configuration.features = 256
        configuration.outChannels = [256, 512, 1024, 1024]
        return configuration
    }

    var tokenGrid: Int { inputSize / patchSize }
    /// The head reads concatenated local+global features (`cat_token`).
    var headInputDimensions: Int { embedDimensions * 2 }
    /// The camera decoder reads a camera token, which carries the same concatenated width.
    var cameraDecoderDimensions: Int { embedDimensions * 2 }
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

/// The block MLP (`mlp.fc1`, `mlp.fc2`). The camera encoder's `pose_branch` is the same shape with a
/// narrower input than output, so the widths are given separately there.
final class NFKDA3MLP: Module {
    let fc1: Linear
    let fc2: Linear
    convenience init(dimensions: Int, hidden: Int) {
        self.init(inputDimensions: dimensions, hidden: hidden, outputDimensions: dimensions)
    }
    init(inputDimensions: Int, hidden: Int, outputDimensions: Int) {
        fc1 = Linear(inputDimensions, hidden, bias: true)
        fc2 = Linear(hidden, outputDimensions, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// One transformer block: norm → attention → scale (residual), norm → MLP → scale (residual).
///
/// The reference carries two of these. The backbone's (`dinov2/layers/block.py`) defaults its
/// LayerNorm to `ln_eps` 1e-6; the camera encoder's (`utils/block.py`) takes a plain `nn.LayerNorm`,
/// so torch's 1e-5. The epsilon is a parameter here because both are built.
final class NFKDA3Block: Module {
    let norm1: LayerNorm
    let attn: NFKDA3Attention
    let ls1: NFKDA3LayerScale
    let norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKDA3MLP
    let ls2: NFKDA3LayerScale

    init(dimensions: Int, heads: Int, mlpRatio: Int, qkNorm: Bool, frequency: Float,
         layerNormEps: Float = 1e-6) {
        norm1 = LayerNorm(dimensions: dimensions, eps: layerNormEps)
        attn = NFKDA3Attention(dimensions: dimensions, heads: heads, qkNorm: qkNorm, frequency: frequency)
        ls1 = NFKDA3LayerScale(dimensions: dimensions)
        norm2 = LayerNorm(dimensions: dimensions, eps: layerNormEps)
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
    func hookedFeatures(_ image: MLXArray) -> [MLXArray] { hooked(image).features }

    /// The hooked features beside the camera token each hook carries.
    ///
    /// The reference records `(out_x[:, :, 0], out_x)` per hook and normalizes only what feeds the
    /// head, so the camera token is the concatenated local and global halves **before** the final
    /// LayerNorm. The camera decoder reads the last hook's token.
    ///
    /// - Parameter image: `[1, inputSize, inputSize, 3]`.
    /// - Returns: the features, each `[1, grid*grid, 2*dimensions]`, and the camera tokens, each
    ///   `[1, 2*dimensions]`.
    func hooked(_ image: MLXArray, conditioning: MLXArray? = nil) -> (features: [MLXArray], cameraTokens: [MLXArray]) {
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
        var cameras = [MLXArray]()
        for (index, block) in blocks.enumerated() {
            // The camera token replaces the class-token slot at `altStart`, before that block runs.
            // A caller that knows the camera supplies the encoder's pose token in its place, which is
            // the reference's `cam_token` keyword.
            if index == configuration.altStart {
                let camera = conditioning ?? cameraToken[0..., 0 ..< 1, 0...]   // [1, 1, dim]
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
                // The camera token is that same slot 0 with no norm on either half.
                let raw = concatenated([localOutput[0..., 0, 0...], tokens[0..., 0, 0...]], axis: -1)
                cameras.append(raw)
            }
        }
        return (outputs, cameras)
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

/// What the DualDPT head predicts: the main branch's depth and its confidence, and the aux branch's
/// ray map and its confidence.
///
/// The depth is at the input resolution. The ray map stays at the finest fusion resolution, which is
/// the reference's own convention (only the main path is interpolated to the image size).
struct NFKDA3Prediction {
    /// Exponentiated depth, `[1, H, W]`.
    var depth: MLXArray
    /// The main branch's confidence, `[1, H, W]`.
    var confidence: MLXArray
    /// The ray map, `[1, h, w, 6]`.
    var ray: MLXArray
    /// The ray branch's confidence, `[1, h, w]`.
    var rayConfidence: MLXArray
}

/// The DualDPT head: a shared reassembly and two independent fusion chains, the main branch
/// predicting depth and the aux branch predicting rays.
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
    func callAsFunction(_ features: [MLXArray]) -> NFKDA3Prediction {
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

        let (main, aux) = scratch.fuse(pyramid)                         // includes output_conv1
        var fused = NFKDA3Resample.bilinearAlignCorners(main, height: height, width: width)
        fused = NFKDA3Head.addUVPositionEmbedding(fused, width: width, height: height)
        let logits = scratch.outputConv2Forward(fused)                  // [1, H, W, 2]

        // The aux branch reads the finest level only, and is not interpolated to the image size.
        let finest = NFKDA3Head.addUVPositionEmbedding(aux[aux.count - 1], width: width, height: height)
        let rayLogits = scratch.outputConv2AuxForward(finest, level: NFKDA3Scratch.auxLevels - 1)

        return NFKDA3Prediction(
            depth: exp(logits[0..., 0..., 0..., 0]),
            confidence: exp(logits[0..., 0..., 0..., 1]) + 1,
            ray: rayLogits[0..., 0..., 0..., 0 ..< 6],
            rayConfidence: exp(rayLogits[0..., 0..., 0..., 6]) + 1)
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
        let (fused, aux) = scratch.fuse(pyramid)
        out["fused"] = fused
        var head = NFKDA3Resample.bilinearAlignCorners(fused, height: height, width: width)
        head = NFKDA3Head.addUVPositionEmbedding(head, width: width, height: height)
        out["logits"] = scratch.outputConv2Forward(head)
        for (index, level) in aux.enumerated() {
            out["aux\(index)"] = level
        }
        let finest = NFKDA3Head.addUVPositionEmbedding(aux[aux.count - 1], width: width, height: height)
        out["aux_pos"] = finest
        let stack = scratch.outputConv2Aux[NFKDA3Scratch.auxLevels - 1]
        let stepped = (stack[0] as! Conv2d)(finest)
        out["aux_conv0"] = stepped
        out["aux_norm"] = (stack[2] as! LayerNorm)(stepped)
        out["ray_logits"] = scratch.outputConv2AuxForward(finest, level: NFKDA3Scratch.auxLevels - 1)
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
    @ModuleInfo(key: "refinenet1_aux") var refine1Aux: NFKDA3Fusion
    @ModuleInfo(key: "refinenet2_aux") var refine2Aux: NFKDA3Fusion
    @ModuleInfo(key: "refinenet3_aux") var refine3Aux: NFKDA3Fusion
    @ModuleInfo(key: "refinenet4_aux") var refine4Aux: NFKDA3Fusion
    @ModuleInfo(key: "output_conv1_aux") var outputConv1Aux: [[Module]]
    @ModuleInfo(key: "output_conv2_aux") var outputConv2Aux: [[Module]]

    /// The number of aux pyramid levels the reference keeps (`aux_pyramid_levels`, 4 in every release).
    static let auxLevels = 4

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
        // The aux (ray) branch runs its own fusion chain over the same reassembled pyramid.
        _refine1Aux.wrappedValue = NFKDA3Fusion(features: features, hasResidual: true)
        _refine2Aux.wrappedValue = NFKDA3Fusion(features: features, hasResidual: true)
        _refine3Aux.wrappedValue = NFKDA3Fusion(features: features, hasResidual: true)
        _refine4Aux.wrappedValue = NFKDA3Fusion(features: features, hasResidual: false)
        // `_make_aux_out1_block` at `aux_out1_conv_num` 5: five 3×3 convolutions alternating between
        // the feature width and half of it, with no activation between them.
        _outputConv1Aux.wrappedValue = (0 ..< NFKDA3Scratch.auxLevels).map { _ in
            (0 ..< 5).map { index -> Module in
                let narrows = index % 2 == 0
                return Conv2d(inputChannels: narrows ? features : features / 2,
                              outputChannels: narrows ? features / 2 : features,
                              kernelSize: 3, padding: 1)
            }
        }
        // Sequential(conv, permute, layerNorm, permute, relu, conv): the two permutes bracket a
        // channel-last LayerNorm, which is what NHWC already is, so indices 1, 3, and 4 are markers.
        _outputConv2Aux.wrappedValue = (0 ..< NFKDA3Scratch.auxLevels).map { _ in
            [
                Conv2d(inputChannels: features / 2, outputChannels: 32, kernelSize: 3, padding: 1),
                Module(),
                LayerNorm(dimensions: 32),
                Module(),
                Module(),
                Conv2d(inputChannels: 32, outputChannels: 7, kernelSize: 1),
            ]
        }
    }

    private func size(_ x: MLXArray) -> (Int, Int) { (x.shape[1], x.shape[2]) }

    /// Fuses the reassembled pyramid coarse → fine, as the reference `_fuse` does, and applies
    /// `output_conv1` to the main path and `output_conv1_aux[i]` to each aux level.
    ///
    /// - Parameter pyramid: the reassembled stages, fine → coarse (`layer1` … `layer4`).
    /// - Returns: the main path and the aux pyramid, coarsest level first.
    func fuse(_ pyramid: [MLXArray]) -> (main: MLXArray, aux: [MLXArray]) {
        let l1 = layer1(pyramid[0])
        let l2 = layer2(pyramid[1])
        let l3 = layer3(pyramid[2])
        let l4 = layer4(pyramid[3])

        var path = refine4(l4, skip: nil, size: size(l3))
        var auxPath = refine4Aux(l4, skip: nil, size: size(l3))
        var aux = [auxPath]

        path = refine3(path, skip: l3, size: size(l2))
        auxPath = refine3Aux(auxPath, skip: l3, size: size(l2))
        aux.append(auxPath)

        path = refine2(path, skip: l2, size: size(l1))
        auxPath = refine2Aux(auxPath, skip: l2, size: size(l1))
        aux.append(auxPath)

        path = refine1(path, skip: l1, size: nil)                      // upsample ×2
        auxPath = refine1Aux(auxPath, skip: l1, size: nil)
        aux.append(auxPath)

        let necked = aux.enumerated().map { index, level in
            outputConv1Aux[index].reduce(level) { ($1 as! Conv2d)($0) }
        }
        return (outputConv1(path), necked)
    }

    func outputConv2Forward(_ x: MLXArray) -> MLXArray {
        let conv0 = outputConv2[0] as! Conv2d
        let conv2 = outputConv2[2] as! Conv2d
        return conv2(relu(conv0(x)))
    }

    /// The aux head at one pyramid level: conv, channel LayerNorm, relu, 1×1 to seven channels.
    func outputConv2AuxForward(_ x: MLXArray, level: Int) -> MLXArray {
        let stack = outputConv2Aux[level]
        let conv0 = stack[0] as! Conv2d
        let norm = stack[2] as! LayerNorm
        let conv5 = stack[5] as! Conv2d
        return conv5(relu(norm(conv0(x))))
    }
}

// MARK: - Camera head

/// The camera pose the decoder predicts, in the reference's nine-number encoding.
struct NFKDA3Pose {
    /// The translation, `[1, 3]`.
    var translation: MLXArray
    /// The rotation as a scalar-last (xyzw) quaternion, `[1, 4]`.
    var quaternion: MLXArray
    /// The vertical and horizontal fields of view in radians, `[1, 2]`.
    var fieldOfView: MLXArray

    /// The concatenated `[t, qvec, fov]` encoding the reference passes between its stages, `[1, 9]`.
    var encoding: MLXArray { concatenated([translation, quaternion, fieldOfView], axis: -1) }
}

/// The camera decoder (`cam_dec`): two hidden layers over a hook's camera token, then three heads for
/// the translation, the rotation quaternion, and the field of view.
final class NFKDA3CameraDecoder: Module {
    // A Sequential(linear, relu, linear, relu), so the linears carry indices 0 and 2.
    @ModuleInfo(key: "backbone") var backbone: [Module]
    @ModuleInfo(key: "fc_t") var translation: Linear
    @ModuleInfo(key: "fc_qvec") var quaternion: Linear
    // A Sequential(linear, relu): the field of view is non-negative.
    @ModuleInfo(key: "fc_fov") var fieldOfView: [Module]

    init(dimensions: Int) {
        _backbone.wrappedValue = [
            Linear(dimensions, dimensions, bias: true),
            Module(),
            Linear(dimensions, dimensions, bias: true),
            Module(),
        ]
        _translation.wrappedValue = Linear(dimensions, 3, bias: true)
        _quaternion.wrappedValue = Linear(dimensions, 4, bias: true)
        _fieldOfView.wrappedValue = [Linear(dimensions, 2, bias: true), Module()]
    }

    /// - Parameter token: a camera token, `[1, 2*dimensions]`.
    func callAsFunction(_ token: MLXArray) -> NFKDA3Pose {
        let first = backbone[0] as! Linear
        let second = backbone[2] as! Linear
        let hidden = relu(second(relu(first(token))))
        let fov = fieldOfView[0] as! Linear
        return NFKDA3Pose(translation: translation(hidden),
                          quaternion: quaternion(hidden),
                          fieldOfView: relu(fov(hidden)))
    }
}

/// The camera encoder (`cam_enc`): a known pose becomes the token the backbone reads in place of its
/// learned one, so a caller with calibrated cameras conditions the model rather than letting it guess.
final class NFKDA3CameraEncoder: Module {
    @ModuleInfo(key: "pose_branch") var poseBranch: NFKDA3MLP
    @ModuleInfo(key: "token_norm") var tokenNorm: LayerNorm
    @ModuleInfo(key: "trunk") var trunk: [NFKDA3Block]
    @ModuleInfo(key: "trunk_norm") var trunkNorm: LayerNorm

    init(dimensions: Int, heads: Int, depth: Int, mlpRatio: Int) {
        // The reference's `Mlp(in_features=9, hidden_features=dim // 2, out_features=dim)`.
        _poseBranch.wrappedValue = NFKDA3MLP(inputDimensions: NFKDA3CameraEncoder.encodingWidth,
                                             hidden: dimensions / 2, outputDimensions: dimensions)
        _tokenNorm.wrappedValue = LayerNorm(dimensions: dimensions)
        _trunk.wrappedValue = (0 ..< depth).map { _ in
            // No rotary and no query/key norm on this trunk: the reference passes neither, and its
            // block takes a plain `nn.LayerNorm` where the backbone's takes `ln_eps` 1e-6.
            NFKDA3Block(dimensions: dimensions, heads: heads, mlpRatio: mlpRatio, qkNorm: false,
                        frequency: 0, layerNormEps: 1e-5)
        }
        _trunkNorm.wrappedValue = LayerNorm(dimensions: dimensions)
    }

    /// The width of the pose encoding the trunk reads: translation, quaternion, and two fields of view.
    static let encodingWidth = 9

    /// - Parameter encoding: the pose encoding, `[1, views, 9]`.
    /// - Returns: the conditioning tokens, `[1, views, dimensions]`.
    func callAsFunction(_ encoding: MLXArray) -> MLXArray {
        var tokens = tokenNorm(poseBranch(encoding))
        for block in trunk {
            tokens = block(tokens, rope: nil)
        }
        return trunkNorm(tokens)
    }

    /// The reference's `extri_intri_to_pose_encoding`: a camera-to-world rotation and translation with
    /// a pinhole intrinsic matrix become the nine-number encoding the trunk reads.
    ///
    /// - Parameters:
    ///   - rotation: the camera-to-world rotation, `[3, 3]`.
    ///   - translation: the camera-to-world translation, `[3]`.
    ///   - focalLengths: `(fx, fy)` in pixels.
    ///   - imageSize: `(height, width)` in pixels.
    static func encoding(rotation: MLXArray, translation: MLXArray,
                         focalLengths: (Float, Float), imageSize: (Int, Int)) -> MLXArray {
        let quaternion = NFKDA3Rotation.quaternion(fromMatrix: rotation)
        let fovHeight = 2 * atan((Float(imageSize.0) / 2) / focalLengths.1)
        let fovWidth = 2 * atan((Float(imageSize.1) / 2) / focalLengths.0)
        let fov = MLXArray([fovHeight, fovWidth])
        return concatenated([translation.reshaped([3]), quaternion, fov], axis: 0).reshaped([1, 1, encodingWidth])
    }
}

/// The quaternion conventions the camera head uses. The reference stores a rotation scalar-last
/// (`xyzw`), which is the opposite of the more common scalar-first order.
enum NFKDA3Rotation {
    /// `quat_to_mat`: a scalar-last quaternion becomes a rotation matrix `[3, 3]`.
    static func matrix(fromQuaternion q: MLXArray) -> MLXArray {
        let values = q.reshaped([4]).asArray(Float.self)
        let (i, j, k, r) = (values[0], values[1], values[2], values[3])
        let twoS = 2.0 / (i * i + j * j + k * k + r * r)
        let rows: [Float] = [
            1 - twoS * (j * j + k * k), twoS * (i * j - k * r), twoS * (i * k + j * r),
            twoS * (i * j + k * r), 1 - twoS * (i * i + k * k), twoS * (j * k - i * r),
            twoS * (i * k - j * r), twoS * (j * k + i * r), 1 - twoS * (i * i + j * j),
        ]
        return MLXArray(rows, [3, 3])
    }

    /// The inverse: a rotation matrix `[3, 3]` becomes a scalar-last quaternion `[4]`. The branch on
    /// the largest diagonal term is what keeps the square root away from zero.
    static func quaternion(fromMatrix m: MLXArray) -> MLXArray {
        let a = m.reshaped([9]).asArray(Float.self)
        func at(_ row: Int, _ column: Int) -> Float { a[row * 3 + column] }
        let trace = at(0, 0) + at(1, 1) + at(2, 2)
        var x: Float, y: Float, z: Float, w: Float
        if trace > 0 {
            let s = sqrtf(trace + 1) * 2
            w = 0.25 * s
            x = (at(2, 1) - at(1, 2)) / s
            y = (at(0, 2) - at(2, 0)) / s
            z = (at(1, 0) - at(0, 1)) / s
        } else if at(0, 0) > at(1, 1) && at(0, 0) > at(2, 2) {
            let s = sqrtf(1 + at(0, 0) - at(1, 1) - at(2, 2)) * 2
            w = (at(2, 1) - at(1, 2)) / s
            x = 0.25 * s
            y = (at(0, 1) + at(1, 0)) / s
            z = (at(0, 2) + at(2, 0)) / s
        } else if at(1, 1) > at(2, 2) {
            let s = sqrtf(1 + at(1, 1) - at(0, 0) - at(2, 2)) * 2
            w = (at(0, 2) - at(2, 0)) / s
            x = (at(0, 1) + at(1, 0)) / s
            y = 0.25 * s
            z = (at(1, 2) + at(2, 1)) / s
        } else {
            let s = sqrtf(1 + at(2, 2) - at(0, 0) - at(1, 1)) * 2
            w = (at(1, 0) - at(0, 1)) / s
            x = (at(0, 2) + at(2, 0)) / s
            y = (at(1, 2) + at(2, 1)) / s
            z = 0.25 * s
        }
        return MLXArray([x, y, z, w])
    }
}

/// A camera the model predicts: where it is and what it sees.
@objc(NFKMLXDepth3Camera)
public final class NFKMLXDepth3Camera: NSObject {
    /// The camera-to-world translation.
    @objc public let translation: [NSNumber]
    /// The camera-to-world rotation, row-major `[3, 3]` flattened to nine numbers.
    @objc public let rotation: [NSNumber]
    /// The horizontal focal length in pixels, at the size the estimate was made for.
    @objc public let focalLengthX: Double
    /// The vertical focal length in pixels.
    @objc public let focalLengthY: Double
    /// The horizontal field of view in radians.
    @objc public let fieldOfViewX: Double
    /// The vertical field of view in radians.
    @objc public let fieldOfViewY: Double

    init(translation: [Float], rotation: [Float], focalLengthX: Float, focalLengthY: Float,
         fieldOfViewX: Float, fieldOfViewY: Float) {
        self.translation = translation.map { NSNumber(value: $0) }
        self.rotation = rotation.map { NSNumber(value: $0) }
        self.focalLengthX = Double(focalLengthX)
        self.focalLengthY = Double(focalLengthY)
        self.fieldOfViewX = Double(fieldOfViewX)
        self.fieldOfViewY = Double(fieldOfViewY)
        super.init()
    }

    /// The reference's `pose_encoding_to_extri_intri`: the nine-number encoding becomes a rotation, a
    /// translation, and the focal lengths a pinhole intrinsic matrix carries at `imageSize`.
    static func camera(from pose: NFKDA3Pose, imageSize: (height: Int, width: Int)) -> NFKMLXDepth3Camera {
        let rotation = NFKDA3Rotation.matrix(fromQuaternion: pose.quaternion).reshaped([9]).asArray(Float.self)
        let translation = pose.translation.reshaped([3]).asArray(Float.self)
        let fov = pose.fieldOfView.reshaped([2]).asArray(Float.self)
        // The reference clamps the tangent away from zero so a degenerate field of view cannot divide.
        let tangentHeight = Swift.max(tanf(fov[0] / 2), 1e-6)
        let tangentWidth = Swift.max(tanf(fov[1] / 2), 1e-6)
        return NFKMLXDepth3Camera(translation: translation, rotation: rotation,
                                  focalLengthX: (Float(imageSize.width) / 2) / tangentWidth,
                                  focalLengthY: (Float(imageSize.height) / 2) / tangentHeight,
                                  fieldOfViewX: fov[1], fieldOfViewY: fov[0])
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

/// The Depth Anything 3 size to build, for the Objective-C factory. A released checkpoint fits only
/// its own size: `DA3-SMALL` → `.small`, `DA3-BASE` → `.base`, `DA3-LARGE` → `.large`.
@objc(NFKMLXDepth3Variant)
public enum NFKMLXDepth3Variant: Int {
    case small
    case base
    case large
}

@objc(NFKMLXDepthAnything3)
public final class NFKMLXDepthAnything3: NSObject {

    @objc public static let modelName = "depth-anything-3-small"
    @objc public static let baseModelName = "depth-anything-3-base"
    @objc public static let largeModelName = "depth-anything-3-large"

    static func makeNet(_ configuration: NFKMLXDepth3Configuration = .small) -> NFKMLXDepthAnything3Net {
        NFKMLXDepthAnything3Net(configuration)
    }

    static func specs(for variant: NFKMLXDepth3Variant) -> (name: String, configuration: NFKMLXDepth3Configuration) {
        switch variant {
        case .small: return (modelName, .small)
        case .base: return (baseModelName, .base)
        case .large: return (largeModelName, .large)
        }
    }

    /// Builds a Depth Anything 3 backend from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true). Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .small, weightsURL: weightsURL)
    }

    /// Builds one of the released sizes from optional local weights.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXDepth3Variant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let spec = specs(for: variant)
        let net = NFKMLXDepthAnything3Net(spec.configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXDepth3Holder(net)
        return NFKMLXModuleBackend(identifier: spec.name, isReady: true) { image in holder.net.depth(image) }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required. Blocking on the
    /// network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .small, repo: repo, weightsPath: weightsPath, revision: revision,
                    cacheDirectoryURL: cacheDirectoryURL)
    }

    /// The download factory at a chosen size.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXDepth3Variant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        backend(variant: .small, repo: repo, weightsPath: weightsPath, revision: revision,
                cacheDirectoryURL: cacheDirectoryURL, completionHandler: completionHandler)
    }

    /// The asynchronous download factory at a chosen size.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXDepth3Variant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers the three sizes (`depth-anything-3-small`, `-base`, `-large`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        for variant in [NFKMLXDepth3Variant.small, .base, .large] {
            NFKMLXModelRegistry.register(name: specs(for: variant).name) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL)
            }
        }
    }

    /// Loads a checkpoint whole: the backbone (`model.backbone.pretrained.` → `backbone.`), both
    /// DualDPT branches (`model.head.`), and the camera decoder and encoder, transposing 4-D
    /// convolution weights to MLX's channels-last layout. Every released tensor is consumed.
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
        mapped.append(contentsOf: sharedAuxNormals(in: mapped))
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// The aux head's channel LayerNorm, copied from level 0 onto the levels a release omits.
    ///
    /// The reference builds `ln_seq` once and splices that one list into all four `output_conv2_aux`
    /// Sequentials, so a single `nn.LayerNorm` **instance** is shared across the levels. PyTorch
    /// deduplicates a shared module in a state dict, saving it once under the first name that reaches
    /// it, so a released checkpoint carries `output_conv2_aux.0.2` alone and loading it sets all four.
    /// MLX has four distinct modules, so level 0's parameters are copied onto the rest here.
    ///
    /// Reading the absence as "unlearned, left at the `nn.LayerNorm` init" instead is wrong and quiet:
    /// the ray head is level 3, and the identity affine scored its logits at 0.9991 against the
    /// reference's 0.9999999999.
    static func sharedAuxNormals(in mapped: [(String, MLXArray)]) -> [(String, MLXArray)] {
        let source = Dictionary(mapped.map { ($0.0, $0.1) }, uniquingKeysWith: { first, _ in first })
        var supplied = [(String, MLXArray)]()
        for parameter in ["weight", "bias"] {
            let shared = "head.scratch.output_conv2_aux.0.2.\(parameter)"
            guard let value = source[shared] else { continue }
            for level in 1 ..< NFKDA3Scratch.auxLevels {
                supplied.append(("head.scratch.output_conv2_aux.\(level).2.\(parameter)", value))
            }
        }
        return supplied
    }

    /// Maps a checkpoint key onto the built module. Every key in a released checkpoint maps, so a nil
    /// here means the file carries something this port does not build.
    static func remap(_ key: String) -> String? {
        if key.hasPrefix("model.backbone.pretrained.") {
            return "backbone." + key.dropFirst("model.backbone.pretrained.".count)
        }
        if key.hasPrefix("model.head.") {
            return "head." + key.dropFirst("model.head.".count)
        }
        if key.hasPrefix("model.cam_dec.") {
            return "cam_dec." + key.dropFirst("model.cam_dec.".count)
        }
        if key.hasPrefix("model.cam_enc.") {
            return "cam_enc." + key.dropFirst("model.cam_enc.".count)
        }
        return nil
    }
}

/// The full model: backbone (`backbone`), DualDPT head (`head`), and the camera decoder and encoder
/// (`cam_dec`, `cam_enc`).
final class NFKMLXDepthAnything3Net: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKDA3Encoder
    @ModuleInfo(key: "head") var head: NFKDA3Head
    @ModuleInfo(key: "cam_dec") var cameraDecoder: NFKDA3CameraDecoder
    @ModuleInfo(key: "cam_enc") var cameraEncoder: NFKDA3CameraEncoder

    let configuration: NFKMLXDepth3Configuration

    init(_ configuration: NFKMLXDepth3Configuration) {
        self.configuration = configuration
        _backbone.wrappedValue = NFKDA3Encoder(configuration)
        _head.wrappedValue = NFKDA3Head(configuration)
        _cameraDecoder.wrappedValue = NFKDA3CameraDecoder(dimensions: configuration.cameraDecoderDimensions)
        _cameraEncoder.wrappedValue = NFKDA3CameraEncoder(dimensions: configuration.embedDimensions,
                                                          heads: configuration.cameraHeads,
                                                          depth: configuration.cameraTrunkDepth,
                                                          mlpRatio: configuration.mlpRatio)
    }

    /// The raw network on a prepared `[1, inputSize, inputSize, 3]` image: the four hooked features.
    func features(_ image: MLXArray) -> [MLXArray] { backbone.hookedFeatures(image) }

    /// Everything the model predicts from one prepared `[1, inputSize, inputSize, 3]` image: the depth
    /// and ray maps with their confidences, and the camera the decoder reads off the last hook's token.
    ///
    /// - Parameter conditioning: a pose token from `cameraEncoder`, or nil to let the backbone use its
    ///   own learned camera token.
    func predict(_ image: MLXArray, conditioning: MLXArray? = nil)
        -> (prediction: NFKDA3Prediction, pose: NFKDA3Pose) {
        let (features, cameras) = backbone.hooked(image, conditioning: conditioning)
        return (head(features), cameraDecoder(cameras[cameras.count - 1]))
    }

    /// The pipeline's ImageNet normalization (`_normalize_image`).
    static func normalized(_ image: MLXArray) -> MLXArray {
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225])
        return (image - mean) / standardDeviation
    }

    /// Resizes a bridged image `[H, W, 3]` (`0...1`) to the encoder's input size and normalizes it.
    ///
    /// The resize is bilinear where the reference uses a PIL resize, a documented consumer
    /// approximation; the network is at parity on the reference's own pixel values.
    func prepared(_ image: MLXArray) -> MLXArray {
        let resized = NFKDA3Resample.bilinearAlignCorners(
            image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]),
            height: configuration.inputSize, width: configuration.inputSize)
        return NFKMLXDepthAnything3Net.normalized(resized)
    }

    /// Maps a bridged image `[H, W, 3]` (`0...1`) to a grayscale depth image `[H, W, 3]` (`0...1`),
    /// near = bright. The encoder runs at the fixed input size; the map resizes back and normalizes.
    func depth(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let prepared = prepared(image)
        let depthMap = head(backbone.hookedFeatures(prepared)).depth   // [1, h, w]
        let full = NFKDA3Resample.bilinearAlignCorners(
            depthMap.reshaped([1, configuration.inputSize, configuration.inputSize, 1]),
            height: height, width: width)

        let minimum = full.min()
        let span = maximum(full.max() - minimum, MLXArray(1e-6))
        let normalized = ((full - minimum) / span).reshaped([height, width, 1])
        return concatenated([normalized, normalized, normalized], axis: 2)
    }
}

/// Depth Anything 3 as an object, for the predictions a single-image backend cannot carry: the camera
/// the model reads off its own camera token, and the ray map beside the depth.
///
/// The depth path is `NFKMLXDepthAnything3.backend(variant:weightsURL:)`, which returns a grayscale
/// image through the ordinary inference contract. This class is the way to the rest.
///
/// - Since: InferKit 0.4.0
@objc(NFKMLXDepth3Estimator)
public final class NFKMLXDepth3Estimator: NSObject {
    private let net: NFKMLXDepthAnything3Net

    init(net: NFKMLXDepthAnything3Net) {
        self.net = net
        super.init()
    }

    /// Builds an estimator from optional local weights. A nil `weightsURL` builds random weights, as
    /// the backend factories do.
    @objc(estimatorWithVariant:weightsURL:error:)
    public static func estimator(variant: NFKMLXDepth3Variant, weightsURL: URL?) throws -> NFKMLXDepth3Estimator {
        let net = NFKMLXDepthAnything3Net(NFKMLXDepthAnything3.specs(for: variant).configuration)
        if let weightsURL {
            try NFKMLXDepthAnything3.loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXDepth3Estimator(net: net)
    }

    /// Downloads the checkpoint from Hugging Face, then builds. Blocking on the network; run off the
    /// render thread.
    @objc(estimatorWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func estimator(variant: NFKMLXDepth3Variant, repo: String, weightsPath: String,
                                 revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXDepth3Estimator {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try estimator(variant: variant, weightsURL: url)
    }

    /// The camera the model reads off the image: where it is and what it sees.
    ///
    /// The focal lengths are in pixels at the image's own size, so a caller can build an intrinsic
    /// matrix directly.
    @objc(cameraForImage:error:)
    public func camera(for image: CGImage) throws -> NFKMLXDepth3Camera {
        let tensor = try NFKMLXImageBridge.tensor(from: image, channels: 3,
                                              colorSpace: CGColorSpaceCreateDeviceRGB())
        let pose = net.predict(net.prepared(tensor)).pose
        return NFKMLXDepth3Camera.camera(from: pose, imageSize: (image.height, image.width))
    }

    /// The ray map: six Plücker channels per position at the head's own resolution, beside its
    /// confidence.
    ///
    /// Swift-only, because the maps are `MLXArray`s. An Objective-C caller reads the camera above,
    /// which is what the ray branch is there to support.
    public func rays(for image: CGImage) throws -> (ray: MLXArray, confidence: MLXArray) {
        let tensor = try NFKMLXImageBridge.tensor(from: image, channels: 3,
                                              colorSpace: CGColorSpaceCreateDeviceRGB())
        let prediction = net.predict(net.prepared(tensor)).prediction
        return (prediction.ray, prediction.rayConfidence)
    }

    /// Conditions the backbone on a known camera rather than letting it predict one, which is what the
    /// camera encoder is for.
    ///
    /// - Parameters:
    ///   - rotation: the camera-to-world rotation, row-major nine numbers.
    ///   - translation: the camera-to-world translation, three numbers.
    ///   - focalLengthX: the horizontal focal length in pixels.
    ///   - focalLengthY: the vertical focal length in pixels.
    @objc(cameraForImage:knownRotation:translation:focalLengthX:focalLengthY:error:)
    public func camera(for image: CGImage, knownRotation rotation: [NSNumber], translation: [NSNumber],
                       focalLengthX: Double, focalLengthY: Double) throws -> NFKMLXDepth3Camera {
        let encoding = NFKDA3CameraEncoder.encoding(
            rotation: MLXArray(rotation.map { $0.floatValue }, [3, 3]),
            translation: MLXArray(translation.map { $0.floatValue }),
            focalLengths: (Float(focalLengthX), Float(focalLengthY)),
            imageSize: (image.height, image.width))
        let tensor = try NFKMLXImageBridge.tensor(from: image, channels: 3,
                                              colorSpace: CGColorSpaceCreateDeviceRGB())
        let conditioning = net.cameraEncoder(encoding)
        let pose = net.predict(net.prepared(tensor), conditioning: conditioning).pose
        return NFKMLXDepth3Camera.camera(from: pose, imageSize: (image.height, image.width))
    }
}

private final class NFKMLXDepth3Holder: @unchecked Sendable {
    let net: NFKMLXDepthAnything3Net
    init(_ net: NFKMLXDepthAnything3Net) { self.net = net }
}
