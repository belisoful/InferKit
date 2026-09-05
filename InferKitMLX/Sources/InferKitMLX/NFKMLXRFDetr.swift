//
//  NFKMLXRFDetr.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXFast
import MLXNN

// RF-DETR (Roboflow, `RfDetrForObjectDetection`, transformers 5.16), a two-stage Group-DETR object
// detector, ported from transformers' own implementation. It is a distinct architecture from RT-DETR:
// a WINDOWED DINOv2 backbone, a C2f + RepVGG scale projector over the selected DINOv2 stages, two-stage
// query selection, MIXED queries (a learned reference_point_embed refined by the top-k proposals plus a
// learned query_feat), and an LW-DETR deformable decoder over a SINGLE feature level.
//
// STATUS. COMPLETE and at REFERENCE PARITY on BOTH the tiny random config and the released
// Roboflow/rf-detr-base weights, validated seam by seam in `NFKMLXReferenceParityTests`
// (`testRFDetrMatchesTheReference` and `testRFDetrMatchesTheReferenceOnReleasedWeights` — the latter
// loads the RELEASED file through the on-device `loadWeights`): every seam (bb.N, projector, first-stage
// class head, decoder last hidden, final logits/pred_boxes) matches over the reference's own top-k
// selection, and the end-to-end run is within the top-k tie tolerance. Two seam bugs were caught: the
// re-partition shape (SEAM 1) and the antialiased-bicubic coefficient (SEAM 2). `loadWeights` converts
// the original Roboflow naming on device (`remapReferenceKey` + the fused-qkv split), so the released
// file loads directly; registered in `registerAll` with the @objc download factories, and `detect()`
// applies the image processor's ImageNet normalization.
//
// SEAMS (the classes of bug the DA3/RT-DETR ports hit; validated at tiny unless noted):
//   1. WINDOW PARTITION. The reference reshapes the patch grid as [nWin, W_pw, nWin, H_pw] — width and
//      height SWAPPED — its own comment calls this a preserved original-implementation bug; mirrored
//      verbatim, validated correct at bb.N. The global-attention block re-partitions using the
//      UNPARTITIONED shape (the reference reassigns `hidden_states` before reading `.shape`) — that was
//      the one seam bug the tiny parity caught.
//   2. POSITION-EMBEDDING INTERPOLATION. The released backbone interpolates its 518-trained pos_embed to
//      the processor resolution (560, i.e. 37->40 patches) with `bicubic, align_corners=false,
//      antialias=true`. Antialias is NOT a no-op when upsampling: it uses the PIL cubic coefficient
//      a=-0.5 (torch's non-antialias bicubic uses -0.75) with per-output weight normalization —
//      `antialiasResampleMatrix`. This was the one seam bug the released-weights parity caught.
//   3. DIRECT BOX SPACE. RF-DETR refines boxes in DIRECT normalized [0,1] coordinates via
//      `nfkRFDetrRefineBboxes` (cxcy = delta_xy * ref_wh + ref_xy; wh = exp(delta_wh) * ref_wh), NOT the
//      sigmoid / inverse-sigmoid space RT-DETR uses. Validated correct.
//   4. CHANNELS-FIRST LAYERNORM. The projector's `RfDetrLayerNorm(channels_first)` normalizes over the
//      channel axis; in MLX's NHWC that is the last axis, so a plain `LayerNorm(channels)` matches with
//      no permute — the conv norms use eps 1e-5 while the projector's final norm uses 1e-6. Validated.
//   5. TOP-K FLOAT TIE. `torch.topk` and MLX `argSort` can break a sub-ulp score tie differently; at the
//      tiny config the end-to-end selection matched exactly. For the released weights, verify the decoder
//      over the recorded `topk_ind` and assert the end-to-end boxes at a tie-reflecting tolerance.
//
// The deformable cross-attention is multi-scale deformable ATTENTION (a `grid_sample` bilinear gather
// via `takeAlong`), portable to MLX, and structurally shared with `NFKMLXRTDetr`.

// MARK: - Configuration

public struct NFKMLXRFDetrConfiguration: Sendable {
    // DINOv2 windowed backbone.
    public var backboneHiddenSize: Int
    public var backboneLayers: Int
    public var backboneHeads: Int
    public var mlpRatio: Int
    public var patchSize: Int
    public var backboneImageSize: Int          // the pos_embed training resolution
    public var numWindows: Int
    public var useSwiGLUFFN: Bool
    public var layerScaleValue: Float
    public var backboneLayerNormEps: Float
    public var useMaskToken: Bool
    /// The selected backbone stages, 0-based layer indices (a stage `stageN` is the output of layer N-1).
    public var outIndices: [Int]
    public var applyLayerNorm: Bool
    // Projector.
    public var hiddenExpansion: Float
    public var c2fNumBlocks: Int
    /// The projector activation: `silu` in the released model.
    public var projectorUsesSilu: Bool
    // Decoder.
    public var dModel: Int
    public var decoderLayers: Int
    public var decoderSelfAttentionHeads: Int
    public var decoderCrossAttentionHeads: Int
    public var decoderNPoints: Int
    public var decoderFFNDim: Int
    public var numFeatureLevels: Int
    public var numQueries: Int
    public var groupDetr: Int
    public var numLabels: Int
    public var layerNormEps: Float
    /// The square input the image processor resizes to.
    public var inputResolution: Int
    /// The minimum per-query class probability a detection is emitted at.
    public var confidenceThreshold: Float = 0.3

    /// The window-block layer indices, derived the reference's way: `range(outIndices.last+1)` minus the
    /// out indices. A layer NOT in this set does global (unpartitioned) attention.
    public var windowBlockIndices: Set<Int> {
        guard let last = outIndices.max() else { return [] }
        var indices = Set(0 ... last)
        for out in outIndices { indices.remove(out) }
        return indices
    }

    public init(backboneHiddenSize: Int, backboneLayers: Int, backboneHeads: Int, mlpRatio: Int,
                patchSize: Int, backboneImageSize: Int, numWindows: Int, useSwiGLUFFN: Bool,
                layerScaleValue: Float, backboneLayerNormEps: Float, useMaskToken: Bool,
                outIndices: [Int], applyLayerNorm: Bool, hiddenExpansion: Float, c2fNumBlocks: Int,
                projectorUsesSilu: Bool, dModel: Int, decoderLayers: Int, decoderSelfAttentionHeads: Int,
                decoderCrossAttentionHeads: Int, decoderNPoints: Int, decoderFFNDim: Int,
                numFeatureLevels: Int, numQueries: Int, groupDetr: Int, numLabels: Int,
                layerNormEps: Float, inputResolution: Int) {
        self.backboneHiddenSize = backboneHiddenSize
        self.backboneLayers = backboneLayers
        self.backboneHeads = backboneHeads
        self.mlpRatio = mlpRatio
        self.patchSize = patchSize
        self.backboneImageSize = backboneImageSize
        self.numWindows = numWindows
        self.useSwiGLUFFN = useSwiGLUFFN
        self.layerScaleValue = layerScaleValue
        self.backboneLayerNormEps = backboneLayerNormEps
        self.useMaskToken = useMaskToken
        self.outIndices = outIndices
        self.applyLayerNorm = applyLayerNorm
        self.hiddenExpansion = hiddenExpansion
        self.c2fNumBlocks = c2fNumBlocks
        self.projectorUsesSilu = projectorUsesSilu
        self.dModel = dModel
        self.decoderLayers = decoderLayers
        self.decoderSelfAttentionHeads = decoderSelfAttentionHeads
        self.decoderCrossAttentionHeads = decoderCrossAttentionHeads
        self.decoderNPoints = decoderNPoints
        self.decoderFFNDim = decoderFFNDim
        self.numFeatureLevels = numFeatureLevels
        self.numQueries = numQueries
        self.groupDetr = groupDetr
        self.numLabels = numLabels
        self.layerNormEps = layerNormEps
        self.inputResolution = inputResolution
    }

    /// The parity configuration: the `run_reference.py rf_detr` tiny random model, carrying every
    /// structural form (a global-attention block among windowed ones, two selected stages, a C2f
    /// projector, group-DETR, and a two-layer decoder over one 4x4 feature level).
    public static let tiny = NFKMLXRFDetrConfiguration(
        backboneHiddenSize: 32, backboneLayers: 4, backboneHeads: 2, mlpRatio: 4, patchSize: 14,
        backboneImageSize: 56, numWindows: 2, useSwiGLUFFN: false, layerScaleValue: 1.0,
        backboneLayerNormEps: 1e-6, useMaskToken: true, outIndices: [2, 4], applyLayerNorm: true,
        hiddenExpansion: 0.5, c2fNumBlocks: 2, projectorUsesSilu: true, dModel: 32, decoderLayers: 2,
        decoderSelfAttentionHeads: 2, decoderCrossAttentionHeads: 4, decoderNPoints: 4, decoderFFNDim: 48,
        numFeatureLevels: 1, numQueries: 10, groupDetr: 2, numLabels: 4, layerNormEps: 1e-5,
        inputResolution: 56)

    /// The released `Roboflow/rf-detr-base` geometry (DINOv2-small windowed backbone, 91 classes, run at
    /// 560). Values verified against the release `config.json` / `preprocessor_config.json` and the
    /// checkpoint shapes: `decoder_n_points` is 2 (NOT 4), `num_labels` 91 (class_embed [91, 256]).
    public static let base = NFKMLXRFDetrConfiguration(
        backboneHiddenSize: 384, backboneLayers: 12, backboneHeads: 6, mlpRatio: 4, patchSize: 14,
        backboneImageSize: 518, numWindows: 4, useSwiGLUFFN: false, layerScaleValue: 1.0,
        backboneLayerNormEps: 1e-6, useMaskToken: true, outIndices: [2, 5, 8, 11], applyLayerNorm: true,
        hiddenExpansion: 0.5, c2fNumBlocks: 3, projectorUsesSilu: true, dModel: 256, decoderLayers: 3,
        decoderSelfAttentionHeads: 8, decoderCrossAttentionHeads: 16, decoderNPoints: 2, decoderFFNDim: 2048,
        numFeatureLevels: 1, numQueries: 300, groupDetr: 13, numLabels: 91, layerNormEps: 1e-5,
        inputResolution: 560)
}

// MARK: - DINOv2 windowed backbone

/// The DINOv2 patch-embedding convolution (`embeddings.patch_embeddings.projection`, WITH bias).
final class NFKRFDetrPatchEmbeddings: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(_ config: NFKMLXRFDetrConfiguration) {
        _projection.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: config.backboneHiddenSize,
            kernelSize: IntOrPair(config.patchSize), stride: IntOrPair(config.patchSize), bias: true)
    }

    /// `pixels` `[B, H, W, 3]` NHWC -> tokens `[B, Hp*Wp, C]` (row-major: h outer, w inner).
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let embedded = projection(pixels)                                // [B, Hp, Wp, C]
        let (b, hp, wp, c) = (embedded.dim(0), embedded.dim(1), embedded.dim(2), embedded.dim(3))
        return embedded.reshaped([b, hp * wp, c])
    }
}

/// DINOv2 embeddings: the CLS token, position embeddings, the patch conv, and (in RF-DETR) the window
/// partition. `mask_token` is loaded for coverage and unused at inference.
final class NFKRFDetrDinov2Embeddings: Module {
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "mask_token") var maskToken: MLXArray
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: NFKRFDetrPatchEmbeddings
    let config: NFKMLXRFDetrConfiguration

    init(_ config: NFKMLXRFDetrConfiguration) {
        self.config = config
        let numPatches = (config.backboneImageSize / config.patchSize) * (config.backboneImageSize / config.patchSize)
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, config.backboneHiddenSize])
        _maskToken.wrappedValue = MLXArray.zeros([1, config.backboneHiddenSize])
        _positionEmbeddings.wrappedValue = MLXArray.zeros([1, numPatches + 1, config.backboneHiddenSize])
        _patchEmbeddings.wrappedValue = NFKRFDetrPatchEmbeddings(config)
    }

    /// `pixels` `[B, H, W, 3]` -> embeddings, window-partitioned when `numWindows > 1`
    /// `[B * numWindows^2, 1 + patchesPerWindow, C]`.
    func callAsFunction(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        let batch = pixels.dim(0)
        var tokens = patchEmbeddings(pixels)                             // [B, Hp*Wp, C]
        let cls = broadcast(clsToken, to: [batch, 1, config.backboneHiddenSize])
        tokens = concatenated([cls, tokens], axis: 1)                    // [B, 1+Hp*Wp, C]
        tokens = tokens + interpolatePositionEmbedding(height: height, width: width)
        if config.numWindows > 1 {
            tokens = windowPartition(tokens, height: height, width: width)
        }
        return tokens
    }

    /// The DINOv2 position embedding, interpolated to the current patch grid when it differs from the
    /// trained one. For the tiny config the counts match and this returns the table directly.
    ///
    /// KNOWN-RISK SEAM: the released path is `bicubic, align_corners=false, antialias=true`. Antialias
    /// is a lowpass over the source before sampling, which a plain bicubic resize does NOT do. Measured
    /// at `rf_detr_real` — implement the antialiased resample here when that seam is worked.
    func interpolatePositionEmbedding(height: Int, width: Int) -> MLXArray {
        let numPatches = (height / config.patchSize) * (width / config.patchSize)
        let numPositions = positionEmbeddings.dim(1) - 1
        if numPatches == numPositions && height == width {
            return positionEmbeddings
        }
        // The reference interpolates bicubic, align_corners=false, ANTIALIAS=true. Antialias is NOT a
        // no-op even when upsampling: it uses the PIL cubic coefficient a=-0.5 (torch's non-antialias
        // bicubic, e.g. NFKMLXBicubic, uses a=-0.75) with per-output weight normalization. Measured:
        // a=-0.5, support 2, invscale 1 reproduces torch's antialias=true 37->40 resample to 2e-6.
        let classPos = positionEmbeddings[0..., 0 ..< 1, 0...]
        let side = Int(Double(numPositions).squareRoot().rounded())
        let dim = positionEmbeddings.dim(2)
        let newHeight = height / config.patchSize
        let newWidth = width / config.patchSize
        let grid = positionEmbeddings[0..., 1..., 0...].reshaped([side, side, dim])
        let wH = NFKRFDetrDinov2Embeddings.antialiasResampleMatrix(from: side, to: newHeight)   // [newH, side]
        let wW = NFKRFDetrDinov2Embeddings.antialiasResampleMatrix(from: side, to: newWidth)     // [newW, side]
        let vertical = matmul(wH, grid.reshaped([side, side * dim])).reshaped([newHeight, side, dim])
        let horizontal = matmul(vertical.transposed(0, 2, 1), wW.transposed(1, 0))               // [newH, dim, newW]
        let resized = horizontal.transposed(0, 2, 1).reshaped([1, newHeight * newWidth, dim])
        return concatenated([classPos, resized], axis: 1)
    }

    /// The 1-D antialiased-bicubic resample operator `[outN, inN]` for torch's
    /// `interpolate(mode: bicubic, align_corners: false, antialias: true)`. It uses the PIL cubic
    /// coefficient a = -0.5 (NOT torch's non-antialias -0.75), half-pixel centers, a support of 2 (widened
    /// by 1/scale when downsampling), and per-output weight normalization over the border-clamped taps.
    static func antialiasResampleMatrix(from inN: Int, to outN: Int) -> MLXArray {
        let a: Float = -0.5
        func filter(_ value: Float) -> Float {
            let x = abs(value)
            if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
            if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
            return 0
        }
        let scale = Float(inN) / Float(outN)
        let support: Float = scale > 1 ? 2 * scale : 2
        let invscale: Float = scale > 1 ? 1 / scale : 1
        var rows = [Float](repeating: 0, count: outN * inN)
        for i in 0 ..< outN {
            let center = scale * (Float(i) + 0.5)
            let start = max(Int(center - support + 0.5), 0)
            let end = min(Int(center + support + 0.5), inN)
            var weights = [Float](); var total: Float = 0
            for j in start ..< end {
                let w = filter((Float(j) - center + 0.5) * invscale)
                weights.append(w); total += w
            }
            for (offset, j) in (start ..< end).enumerated() { rows[i * inN + j] = weights[offset] / total }
        }
        return MLXArray(rows).reshaped([outN, inN])
    }

    /// Splits the patch grid into `numWindows^2` local windows, each with a replicated CLS. Mirrors the
    /// reference's reshape EXACTLY, including its width/height-swapped grouping (its own comment marks
    /// this as an original-implementation bug it preserves). KNOWN-RISK SEAM 1.
    func windowPartition(_ tokens: MLXArray, height: Int, width: Int) -> MLXArray {
        let batch = tokens.dim(0)
        let channels = tokens.dim(2)
        let windows = config.numWindows
        let nH = height / config.patchSize
        let nW = width / config.patchSize
        let wPerWindow = nW / windows
        let hPerWindow = nH / windows

        let cls = tokens[0..., 0 ..< 1, 0...]                            // [B, 1, C]
        var pixel = tokens[0..., 1..., 0...].reshaped([batch, nH, nW, channels])
        // Reference: view(B, windows, wPerWindow, windows, hPerWindow, C).transpose(2, 3)
        pixel = pixel.reshaped([batch, windows, wPerWindow, windows, hPerWindow, channels])
        pixel = pixel.transposed(0, 1, 3, 2, 4, 5)
        pixel = pixel.reshaped([batch * windows * windows, hPerWindow * wPerWindow, channels])
        let clsRepeated = broadcast(cls.reshaped([batch, 1, 1, channels]),
                                    to: [batch, windows * windows, 1, channels])
            .reshaped([batch * windows * windows, 1, channels])
        return concatenated([clsRepeated, pixel], axis: 1)
    }
}

/// DINOv2 self-attention (`attention.attention`, separate query/key/value) + output (`attention.output.dense`).
final class NFKRFDetrDinov2Attention: Module {
    @ModuleInfo(key: "attention") var attention: NFKRFDetrDinov2SelfAttention
    @ModuleInfo(key: "output") var output: NFKRFDetrDinov2SelfOutput

    init(_ config: NFKMLXRFDetrConfiguration) {
        _attention.wrappedValue = NFKRFDetrDinov2SelfAttention(config)
        _output.wrappedValue = NFKRFDetrDinov2SelfOutput(config)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { output(attention(x)) }
}

final class NFKRFDetrDinov2SelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    let heads: Int
    let headDim: Int

    init(_ config: NFKMLXRFDetrConfiguration) {
        heads = config.backboneHeads
        headDim = config.backboneHiddenSize / config.backboneHeads
        _query.wrappedValue = Linear(config.backboneHiddenSize, config.backboneHiddenSize)
        _key.wrappedValue = Linear(config.backboneHiddenSize, config.backboneHiddenSize)
        _value.wrappedValue = Linear(config.backboneHiddenSize, config.backboneHiddenSize)
    }

    private func split(_ t: MLXArray) -> MLXArray {
        let (b, n) = (t.dim(0), t.dim(1))
        return t.reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        let q = split(query(x)), k = split(key(x)), v = split(value(x))
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDim)), mask: .none)
        return attn.transposed(0, 2, 1, 3).reshaped([b, n, heads * headDim])
    }
}

/// The DINOv2 attention output projection (`attention.output.dense`). The residual add lives in the layer.
final class NFKRFDetrDinov2SelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(_ config: NFKMLXRFDetrConfiguration) {
        _dense.wrappedValue = Linear(config.backboneHiddenSize, config.backboneHiddenSize)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { dense(x) }
}

/// DINOv2 layer scale: `hidden * lambda1`.
final class NFKRFDetrLayerScale: Module {
    @ParameterInfo(key: "lambda1") var lambda1: MLXArray
    init(_ config: NFKMLXRFDetrConfiguration) {
        _lambda1.wrappedValue = MLXArray.ones([config.backboneHiddenSize]) * config.layerScaleValue
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * lambda1 }
}

/// DINOv2 MLP (`mlp.fc1` / `mlp.fc2`, GELU). The SwiGLU variant (`weights_in` / `weights_out`) is not
/// used by the released model; add it here when a SwiGLU release is targeted.
final class NFKRFDetrDinov2MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(_ config: NFKMLXRFDetrConfiguration) {
        let hidden = config.backboneHiddenSize * config.mlpRatio
        _fc1.wrappedValue = Linear(config.backboneHiddenSize, hidden)
        _fc2.wrappedValue = Linear(hidden, config.backboneHiddenSize)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// One DINOv2 block. A `global_attention` layer unpartitions the window-batched sequences into one
/// sequence per image before attending and re-partitions after; a window layer attends within windows.
final class NFKRFDetrDinov2Layer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: NFKRFDetrDinov2Attention
    @ModuleInfo(key: "layer_scale1") var layerScale1: NFKRFDetrLayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKRFDetrDinov2MLP
    @ModuleInfo(key: "layer_scale2") var layerScale2: NFKRFDetrLayerScale
    let globalAttention: Bool
    let numWindows: Int

    init(_ config: NFKMLXRFDetrConfiguration, layerIndex: Int) {
        globalAttention = !config.windowBlockIndices.contains(layerIndex)
        numWindows = config.numWindows
        _norm1.wrappedValue = LayerNorm(dimensions: config.backboneHiddenSize, eps: config.backboneLayerNormEps)
        _attention.wrappedValue = NFKRFDetrDinov2Attention(config)
        _layerScale1.wrappedValue = NFKRFDetrLayerScale(config)
        _norm2.wrappedValue = LayerNorm(dimensions: config.backboneHiddenSize, eps: config.backboneLayerNormEps)
        _mlp.wrappedValue = NFKRFDetrDinov2MLP(config)
        _layerScale2.wrappedValue = NFKRFDetrLayerScale(config)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let residual = hidden
        var attended: MLXArray
        if globalAttention {
            // The reference reads `hidden_states.shape` AFTER reassigning it to the unpartitioned
            // tensor, so the re-partition uses the merged [B/windows^2, windows^2*seq, C] shape.
            let unpartitioned = windowUnpartitionBeforeAttention(hidden)
            let attnOut = attention(norm1(unpartitioned))
            attended = windowPartitionAfterAttention(attnOut, originalShape: unpartitioned.shape)
        } else {
            attended = attention(norm1(hidden))
        }
        attended = layerScale1(attended)
        var x = attended + residual
        let residual2 = x
        x = layerScale2(mlp(norm2(x)))
        return x + residual2
    }

    /// Merges the window-batched sequences back into one sequence per image for global attention.
    private func windowUnpartitionBeforeAttention(_ hidden: MLXArray) -> MLXArray {
        let (b, seq, c) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let windowsSquared = numWindows * numWindows
        return hidden.reshaped([b / windowsSquared, windowsSquared * seq, c])
    }

    /// Reshapes the global-attention output back into window-batched form.
    private func windowPartitionAfterAttention(_ output: MLXArray, originalShape: [Int]) -> MLXArray {
        let (b, seq, c) = (originalShape[0], originalShape[1], originalShape[2])
        let windowsSquared = numWindows * numWindows
        return output.reshaped([b * windowsSquared, seq / windowsSquared, c])
    }
}

/// The DINOv2 encoder + the selected-stage feature-map extraction (`layernorm`, drop CLS, unpartition,
/// reshape to `[B, Hp, Wp, C]`).
final class NFKRFDetrDinov2Backbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKRFDetrDinov2Embeddings
    @ModuleInfo(key: "encoder") var encoder: NFKRFDetrDinov2Encoder
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm
    let config: NFKMLXRFDetrConfiguration

    init(_ config: NFKMLXRFDetrConfiguration) {
        self.config = config
        _embeddings.wrappedValue = NFKRFDetrDinov2Embeddings(config)
        _encoder.wrappedValue = NFKRFDetrDinov2Encoder(config)
        _layernorm.wrappedValue = LayerNorm(dimensions: config.backboneHiddenSize, eps: config.backboneLayerNormEps)
    }

    /// `pixels` `[B, H, W, 3]` -> the selected feature maps, each `[B, Hp, Wp, C]` NHWC.
    func callAsFunction(_ pixels: MLXArray, height: Int, width: Int) -> [MLXArray] {
        let embedded = embeddings(pixels, height: height, width: width)
        // hidden_states = [embedding_output, after L0, after L1, ...]; a stageN feature is index N.
        let hiddenStates = encoder(embedded)
        var maps = [MLXArray]()
        for index in config.outIndices {
            var hidden = hiddenStates[index]
            if config.applyLayerNorm { hidden = layernorm(hidden) }
            hidden = hidden[0..., 1..., 0...]                            // drop CLS
            if config.numWindows > 1 {
                hidden = windowUnpartition(hidden, height: height, width: width)
            }
            let nH = height / config.patchSize
            let nW = width / config.patchSize
            maps.append(hidden.reshaped([pixels.dim(0), nH, nW, config.backboneHiddenSize]))
        }
        return maps
    }

    /// Reassembles windowed patch tokens into the image-level grid order before reshaping to a map.
    /// Mirrors the reference's transpose(2, 3) exactly (the width/height-swapped grouping). SEAM 1.
    private func windowUnpartition(_ hidden: MLXArray, height: Int, width: Int) -> MLXArray {
        let windows = config.numWindows
        let nH = height / config.patchSize
        let nW = width / config.patchSize
        let (batchWindows, seq, c) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let windowsSquared = windows * windows
        let hPerWindow = nH / windows
        let wPerWindow = nW / windows
        var x = hidden.reshaped([batchWindows / windowsSquared, windowsSquared * seq, c])
        x = x.reshaped([batchWindows / windowsSquared, windows, windows, hPerWindow, wPerWindow, c])
        x = x.transposed(0, 1, 3, 2, 4, 5)
        return x.reshaped([batchWindows / windowsSquared, nH * nW, c])
    }
}

final class NFKRFDetrDinov2Encoder: Module {
    @ModuleInfo(key: "layer") var layer: [NFKRFDetrDinov2Layer]

    init(_ config: NFKMLXRFDetrConfiguration) {
        _layer.wrappedValue = (0 ..< config.backboneLayers).map { NFKRFDetrDinov2Layer(config, layerIndex: $0) }
    }

    /// Returns every hidden state: the embedding output, then the output after each layer.
    func callAsFunction(_ embedded: MLXArray) -> [MLXArray] {
        var hidden = embedded
        var states = [embedded]
        for block in layer {
            hidden = block(hidden)
            states.append(hidden)
        }
        return states
    }
}

// MARK: - Scale projector (C2f + RepVGG)

/// A projector convolution: `conv` (bias-free) + `norm` (channels-first LayerNorm over the channel axis,
/// which in NHWC is the last axis) + optional SiLU.
final class NFKRFDetrConvNorm: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    let activate: Bool

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int, activate: Bool, eps: Float) {
        self.activate = activate
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel), stride: 1,
                                    padding: IntOrPair(kernel / 2), bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: outChannels, eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = norm(conv(x))
        return activate ? silu(y) : y
    }
}

/// A RepVGG block: two 3x3 conv-norms in sequence (`conv1` then `conv2`).
final class NFKRFDetrRepVgg: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKRFDetrConvNorm
    @ModuleInfo(key: "conv2") var conv2: NFKRFDetrConvNorm

    init(_ channels: Int, eps: Float, activate: Bool) {
        _conv1.wrappedValue = NFKRFDetrConvNorm(channels, channels, kernel: 3, activate: activate, eps: eps)
        _conv2.wrappedValue = NFKRFDetrConvNorm(channels, channels, kernel: 3, activate: activate, eps: eps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv2(conv1(x)) }
}

/// The C2f layer: a 1x1 `conv1` split into two halves, RepVGG bottlenecks chained off the second half,
/// every intermediate concatenated, then a 1x1 `conv2` to `d_model`.
final class NFKRFDetrC2FLayer: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKRFDetrConvNorm
    @ModuleInfo(key: "conv2") var conv2: NFKRFDetrConvNorm
    @ModuleInfo(key: "bottlenecks") var bottlenecks: [NFKRFDetrRepVgg]
    let hiddenChannels: Int

    init(_ config: NFKMLXRFDetrConfiguration, inChannels: Int) {
        let activate = config.projectorUsesSilu
        let eps = config.layerNormEps
        let hidden = Int(Float(config.dModel) * config.hiddenExpansion)
        hiddenChannels = hidden
        _conv1.wrappedValue = NFKRFDetrConvNorm(inChannels, 2 * hidden, kernel: 1, activate: activate, eps: eps)
        let conv2In = (2 + config.c2fNumBlocks) * hidden
        _conv2.wrappedValue = NFKRFDetrConvNorm(conv2In, config.dModel, kernel: 1, activate: activate, eps: eps)
        _bottlenecks.wrappedValue = (0 ..< config.c2fNumBlocks).map { _ in
            NFKRFDetrRepVgg(hidden, eps: eps, activate: activate)
        }
    }

    /// `x` `[B, H, W, inChannels]` NHWC. Splits along the channel (last) axis.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let split1 = conv1(x)                                            // [B, H, W, 2*hidden]
        var all = [split1[0..., 0..., 0..., 0 ..< hiddenChannels],
                   split1[0..., 0..., 0..., hiddenChannels ..< (2 * hiddenChannels)]]
        var hidden = all[all.count - 1]
        for bottleneck in bottlenecks {
            hidden = bottleneck(hidden)
            all.append(hidden)
        }
        return conv2(concatenated(all, axis: 3))
    }
}

/// The scale projector: concat the selected DINOv2 feature maps (channel axis) -> C2f -> channels-first
/// LayerNorm (eps 1e-6, distinct from the conv norms' 1e-5).
final class NFKRFDetrScaleProjector: Module {
    @ModuleInfo(key: "projector_layer") var projectorLayer: NFKRFDetrC2FLayer
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: NFKMLXRFDetrConfiguration) {
        let inChannels = config.backboneHiddenSize * config.outIndices.count
        _projectorLayer.wrappedValue = NFKRFDetrC2FLayer(config, inChannels: inChannels)
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-6)
    }

    /// `maps` each `[B, H, W, C]` NHWC -> `[B, H, W, d_model]`.
    func callAsFunction(_ maps: [MLXArray]) -> MLXArray {
        layerNorm(projectorLayer(concatenated(maps, axis: 3)))
    }
}

/// The backbone + projector (`RfDetrConvEncoder`).
final class NFKRFDetrConvEncoder: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKRFDetrDinov2Backbone
    @ModuleInfo(key: "projector") var projector: NFKRFDetrScaleProjector

    init(_ config: NFKMLXRFDetrConfiguration) {
        _backbone.wrappedValue = NFKRFDetrDinov2Backbone(config)
        _projector.wrappedValue = NFKRFDetrScaleProjector(config)
    }

    func callAsFunction(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        projector(backbone(pixels, height: height, width: width))
    }
}

// MARK: - Decoder pieces (deformable attention, MLP head, LW-DETR self-attention)

/// A simple MLP (relu between hidden layers), used by the query-position head and the box/score heads
/// (`RfDetrMLPPredictionHead`). Shares the shape of RT-DETR's `NFKRTDetrMLP`.
final class NFKRFDetrMLP: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(inputDim: Int, hiddenDim: Int, outputDim: Int, numLayers: Int) {
        var dims = [inputDim]
        dims.append(contentsOf: Array(repeating: hiddenDim, count: numLayers - 1))
        dims.append(outputDim)
        _layers.wrappedValue = (0 ..< numLayers).map { Linear(dims[$0], dims[$0 + 1]) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for (index, layer) in layers.enumerated() {
            hidden = index < layers.count - 1 ? relu(layer(hidden)) : layer(hidden)
        }
        return hidden
    }
}

/// LW-DETR self-attention (`self_attn`): the query position is added to q and k; the value reads the
/// hidden state WITHOUT position. Separate `q_proj`/`k_proj`/`v_proj`/`o_proj`. (The group-DETR split is
/// a training-only technique and is skipped at inference.)
final class NFKRFDetrSelfAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let heads: Int
    let headDim: Int

    init(_ config: NFKMLXRFDetrConfiguration) {
        heads = config.decoderSelfAttentionHeads
        headDim = config.dModel / config.decoderSelfAttentionHeads
        let projected = heads * headDim
        _qProj.wrappedValue = Linear(config.dModel, projected)
        _kProj.wrappedValue = Linear(config.dModel, projected)
        _vProj.wrappedValue = Linear(config.dModel, projected)
        _oProj.wrappedValue = Linear(projected, config.dModel)
    }

    private func split(_ t: MLXArray) -> MLXArray {
        let (b, n) = (t.dim(0), t.dim(1))
        return t.reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ hidden: MLXArray, position: MLXArray) -> MLXArray {
        let (b, n) = (hidden.dim(0), hidden.dim(1))
        let withPos = hidden + position
        let q = split(qProj(withPos)), k = split(kProj(withPos)), v = split(vProj(hidden))
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDim)), mask: .none)
        return oProj(attn.transposed(0, 2, 1, 3).reshaped([b, n, heads * headDim]))
    }
}

/// Multi-scale deformable cross-attention (`cross_attn`): bilinear sampling of the flattened encoder
/// features at learned per-head/level/point offsets. Structurally shared with `NFKRTDetrDeformableAttention`;
/// RF-DETR uses a single feature level and 4-coordinate reference points.
final class NFKRFDetrDeformableAttention: Module {
    @ModuleInfo(key: "sampling_offsets") var samplingOffsets: Linear
    @ModuleInfo(key: "attention_weights") var attentionWeights: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear
    let heads: Int
    let levels: Int
    let points: Int
    let dModel: Int

    init(_ config: NFKMLXRFDetrConfiguration) {
        heads = config.decoderCrossAttentionHeads
        levels = config.numFeatureLevels
        points = config.decoderNPoints
        dModel = config.dModel
        _samplingOffsets.wrappedValue = Linear(dModel, heads * levels * points * 2)
        _attentionWeights.wrappedValue = Linear(dModel, heads * levels * points)
        _valueProj.wrappedValue = Linear(dModel, dModel)
        _outputProj.wrappedValue = Linear(dModel, dModel)
    }

    /// Bilinear gather of `value` `[heads, H, W, hd]` at normalized `coords` `[heads, n, 2]` in `0...1`,
    /// with `align_corners=false` and zero padding (the reference's `grid_sample`).
    private func sample(_ value: MLXArray, coords: MLXArray, height: Int, width: Int) -> MLXArray {
        let (headCount, n, hd) = (value.dim(0), coords.dim(1), value.dim(3))
        let flat = value.reshaped([headCount, height * width, hd])
        let px: MLXArray = coords[0..., 0..., 0] * Float(width) - Float(0.5)
        let py: MLXArray = coords[0..., 0..., 1] * Float(height) - Float(0.5)
        let x0 = floor(px), y0 = floor(py)
        let x1 = x0 + Float(1), y1 = y0 + Float(1)
        let wx = px - x0, wy = py - y0
        let corners: [(MLXArray, MLXArray, MLXArray)] = [
            (x0, y0, (Float(1) - wx) * (Float(1) - wy)),
            (x1, y0, wx * (Float(1) - wy)),
            (x0, y1, (Float(1) - wx) * wy),
            (x1, y1, wx * wy),
        ]
        var output = MLXArray.zeros([headCount, n, hd])
        let maxX = Float(width - 1), maxY = Float(height - 1)
        for (xc, yc, weight) in corners {
            let insideX = logicalAnd(xc .>= Float(0), xc .<= maxX)
            let insideY = logicalAnd(yc .>= Float(0), yc .<= maxY)
            let valid = logicalAnd(insideX, insideY).asType(value.dtype)
            let xClamped = clip(xc, min: Float(0), max: maxX).asType(.int32)
            let yClamped = clip(yc, min: Float(0), max: maxY).asType(.int32)
            let flatIndex = yClamped * Int32(width) + xClamped
            let indices = broadcast(flatIndex.reshaped([headCount, n, 1]), to: [headCount, n, hd])
            let gathered = takeAlong(flat, indices, axis: 1)
            output = output + gathered * (weight * valid).reshaped([headCount, n, 1])
        }
        return output
    }

    /// `hidden` `[1, Q, dModel]` already carrying its position, `value` the flattened encoder features
    /// `[1, S, dModel]`, `referencePoints` `[1, Q, levels, 4]`, `shapes` per-level `(h, w)`.
    func callAsFunction(_ hidden: MLXArray, value valueInput: MLXArray, referencePoints: MLXArray,
                        shapes: [(Int, Int)]) -> MLXArray {
        let q = hidden.dim(1)
        let hd = dModel / heads
        let value = valueProj(valueInput).reshaped([valueInput.dim(1), heads, hd])
        let offsets = samplingOffsets(hidden).reshaped([q, heads, levels, points, 2])
        var weights = attentionWeights(hidden).reshaped([q, heads, levels * points])
        weights = softmax(weights, axis: -1).reshaped([q, heads, levels, points])

        // 4-coordinate reference points: location = ref_xy + offset / n_points * ref_wh * 0.5.
        let ref = referencePoints.reshaped([q, 1, levels, 1, 4])
        let refXY = ref[0..., 0..., 0..., 0..., 0 ..< 2]
        let refWH = ref[0..., 0..., 0..., 0..., 2 ..< 4]
        let locations = refXY + (offsets / Float(points)) * refWH * Float(0.5)

        var levelStart = 0
        var perLevel = [MLXArray]()
        for (levelIndex, (h, w)) in shapes.enumerated() {
            let count = h * w
            let levelValue = value[levelStart ..< (levelStart + count)]
            levelStart += count
            let reshaped = levelValue.transposed(1, 0, 2).reshaped([heads, h, w, hd])
            let levelLoc = locations[0..., 0..., levelIndex, 0..., 0...]
            let coords = levelLoc.transposed(1, 0, 2, 3).reshaped([heads, q * points, 2])
            let sampled = sample(reshaped, coords: coords, height: h, width: w)
            perLevel.append(sampled.reshaped([heads, q, points, hd]))
        }
        let stackedLevels = stacked(perLevel, axis: 2).reshaped([heads, q, levels * points, hd])
        let weightHead = weights.transposed(1, 0, 2, 3).reshaped([heads, q, levels * points, 1])
        let combined = (stackedLevels * weightHead).sum(axis: 2)
        return outputProj(combined.transposed(1, 0, 2).reshaped([1, q, heads * hd]))
    }
}

/// One decoder layer: self-attention + norm, deformable cross-attention + norm, MLP (with an internal
/// residual, `RfDetrMLP`) + norm.
final class NFKRFDetrDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKRFDetrSelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: NFKRFDetrDeformableAttention
    @ModuleInfo(key: "cross_attn_layer_norm") var crossAttnNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKRFDetrDecoderMLP
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: NFKMLXRFDetrConfiguration) {
        _selfAttn.wrappedValue = NFKRFDetrSelfAttention(config)
        _selfAttnNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        _crossAttn.wrappedValue = NFKRFDetrDeformableAttention(config)
        _crossAttnNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        _mlp.wrappedValue = NFKRFDetrDecoderMLP(config)
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
    }

    func callAsFunction(_ hidden: MLXArray, position: MLXArray, value: MLXArray,
                        referencePoints: MLXArray, shapes: [(Int, Int)]) -> MLXArray {
        var x = hidden + selfAttn(hidden, position: position)
        x = selfAttnNorm(x)
        let cross = crossAttn(x + position, value: value, referencePoints: referencePoints, shapes: shapes)
        x = crossAttnNorm(x + cross)
        return layerNorm(mlp(x))
    }
}

/// The decoder feed-forward (`mlp`, `RfDetrMLP`): `fc1` (relu) -> `fc2`, with an internal residual add.
final class NFKRFDetrDecoderMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: NFKMLXRFDetrConfiguration) {
        _fc1.wrappedValue = Linear(config.dModel, config.decoderFFNDim)
        _fc2.wrappedValue = Linear(config.decoderFFNDim, config.dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x + fc2(relu(fc1(x))) }
}

/// The LW-DETR decoder: a stack of layers, a final `layernorm` applied to each layer's output, and the
/// `ref_point_head` that maps the sinusoidal-embedded reference points to the per-layer query position.
/// The reference points are CONSTANT across layers (no iterative refinement) — one box refinement runs
/// afterward in the detection head.
final class NFKRFDetrDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKRFDetrDecoderLayer]
    @ModuleInfo(key: "layernorm") var layernorm: LayerNorm
    @ModuleInfo(key: "ref_point_head") var refPointHead: NFKRFDetrMLP
    let config: NFKMLXRFDetrConfiguration

    init(_ config: NFKMLXRFDetrConfiguration) {
        self.config = config
        _layers.wrappedValue = (0 ..< config.decoderLayers).map { _ in NFKRFDetrDecoderLayer(config) }
        _layernorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        _refPointHead.wrappedValue = NFKRFDetrMLP(inputDim: 2 * config.dModel, hiddenDim: config.dModel,
                                                  outputDim: config.dModel, numLayers: 2)
    }

    /// `target` `[1, Q, dModel]`, `referencePoints` `[1, Q, 4]` (direct box space), `value` the flattened
    /// encoder features. Returns the decoder's last hidden state `[1, Q, dModel]`.
    func callAsFunction(_ target: MLXArray, value: MLXArray, referencePoints: MLXArray,
                        shapes: [(Int, Int)]) -> MLXArray {
        // get_reference: valid_ratios are 1 (no padding), so reference_points_inputs = ref broadcast to
        // the feature levels; the query position is the sinusoidal embedding of the first level's coords.
        let q = referencePoints.dim(1)
        let referenceInputs = broadcast(referencePoints.expandedDimensions(axis: 2), to: [1, q, shapes.count, 4])
        let querySine = NFKRFDetrDecoder.sinusoidalPositionEmbedding(
            referenceInputs[0..., 0..., 0, 0...], numPosFeats: config.dModel / 2)
        let queryPos = refPointHead(querySine)

        var hidden = target
        var lastHidden = target
        for layer in layers {
            hidden = layer(hidden, position: queryPos, value: value,
                           referencePoints: referenceInputs, shapes: shapes)
            lastHidden = layernorm(hidden)
        }
        return lastHidden
    }

    /// The DETR sinusoidal embedding of normalized coordinates (`encode_sinusoidal_position_embedding`):
    /// per coordinate, interleaved sin/cos; the first two coordinates (x, y) are swapped to the
    /// `[pos_y, pos_x, ...]` convention. `pos` `[..., nCoords]` -> `[..., nCoords * numPosFeats]`.
    ///
    /// KNOWN-RISK SEAM: the sin/cos interleave (`stack(e[0::2].sin(), e[1::2].cos()).flatten`) and the
    /// x/y swap must match exactly; validate against `init_ref` -> `dec_last`.
    static func sinusoidalPositionEmbedding(_ pos: MLXArray, numPosFeats: Int, temperature: Float = 10000) -> MLXArray {
        let scale = 2 * Float.pi
        var dimT = [Float](repeating: 0, count: numPosFeats)
        for i in 0 ..< numPosFeats {
            dimT[i] = pow(temperature, Float(2 * (i / 2)) / Float(numPosFeats))
        }
        let dimTArray = MLXArray(dimT)                                   // [numPosFeats]
        let nCoords = pos.dim(pos.ndim - 1)
        var perCoord = [MLXArray]()
        for coord in 0 ..< nCoords {
            let value = pos[.ellipsis, coord]                           // [...] drops the coord axis
            let scaled = value.expandedDimensions(axis: value.ndim) * scale / dimTArray   // [..., numPosFeats]
            // Interleave sin of the even channels with cos of the odd channels: reshape to consecutive
            // pairs [..., numPosFeats/2, 2], take [...,0] (evens) and [...,1] (odds), stack, flatten.
            let leading = Array(scaled.shape.dropLast())
            let paired = scaled.reshaped(leading + [numPosFeats / 2, 2])
            let evens = paired[.ellipsis, 0]                            // [..., numPosFeats/2]
            let odds = paired[.ellipsis, 1]
            let interleaved = stacked([sin(evens), cos(odds)], axis: evens.ndim)   // [..., numPosFeats/2, 2]
            perCoord.append(interleaved.reshaped(leading + [numPosFeats]))
        }
        if perCoord.count >= 2 { perCoord.swapAt(0, 1) }
        return concatenated(perCoord, axis: perCoord[0].ndim - 1)
    }
}

/// RF-DETR's direct-space box refinement: cxcy = delta_xy * ref_wh + ref_xy; wh = exp(delta_wh) * ref_wh.
/// NOT the sigmoid / inverse-sigmoid space RT-DETR uses. KNOWN-RISK SEAM 3.
func nfkRFDetrRefineBboxes(reference: MLXArray, deltas: MLXArray) -> MLXArray {
    let refXY = reference[.ellipsis, 0 ..< 2]
    let refWH = reference[.ellipsis, 2 ..< 4]
    let cxcy = deltas[.ellipsis, 0 ..< 2] * refWH + refXY
    let wh = exp(deltas[.ellipsis, 2 ..< 4]) * refWH
    return concatenated([cxcy, wh], axis: -1)
}

// MARK: - RfDetrModel (two-stage query selection)

/// The bare model: backbone + projector, query selection, mixed queries, and the decoder. The
/// per-group heads (`enc_output` / `enc_output_norm` / `enc_out_class_embed` / `enc_out_bbox_embed`) are
/// ModuleLists over `groupDetr`; only group 0 runs at inference.
final class NFKRFDetrModel: Module {
    @ModuleInfo(key: "backbone") var backbone: NFKRFDetrConvEncoder
    @ModuleInfo(key: "reference_point_embed") var referencePointEmbed: Embedding
    @ModuleInfo(key: "query_feat") var queryFeat: Embedding
    @ModuleInfo(key: "decoder") var decoder: NFKRFDetrDecoder
    @ModuleInfo(key: "enc_output") var encOutput: [Linear]
    @ModuleInfo(key: "enc_output_norm") var encOutputNorm: [LayerNorm]
    @ModuleInfo(key: "enc_out_bbox_embed") var encOutBboxEmbed: [NFKRFDetrMLP]
    @ModuleInfo(key: "enc_out_class_embed") var encOutClassEmbed: [Linear]
    let config: NFKMLXRFDetrConfiguration

    init(_ config: NFKMLXRFDetrConfiguration) {
        self.config = config
        _backbone.wrappedValue = NFKRFDetrConvEncoder(config)
        _referencePointEmbed.wrappedValue = Embedding(embeddingCount: config.numQueries * config.groupDetr, dimensions: 4)
        _queryFeat.wrappedValue = Embedding(embeddingCount: config.numQueries * config.groupDetr, dimensions: config.dModel)
        _decoder.wrappedValue = NFKRFDetrDecoder(config)
        _encOutput.wrappedValue = (0 ..< config.groupDetr).map { _ in Linear(config.dModel, config.dModel) }
        _encOutputNorm.wrappedValue = (0 ..< config.groupDetr).map { _ in
            LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        }
        _encOutBboxEmbed.wrappedValue = (0 ..< config.groupDetr).map { _ in
            NFKRFDetrMLP(inputDim: config.dModel, hiddenDim: config.dModel, outputDim: 4, numLayers: 3)
        }
        _encOutClassEmbed.wrappedValue = (0 ..< config.groupDetr).map { _ in Linear(config.dModel, config.numLabels) }
    }

    /// The single-feature-level anchor proposals (`gen_encoder_output_proposals`) for a `[h, w]` grid,
    /// assuming full validity (no padding). Returns `(proposals [S, 4], invalidMask [S, 1])` in DIRECT
    /// box space. Proposals outside (0.01, 0.99) are invalid; wh = 0.05 * 2^level (level 0 -> 0.05).
    private func proposals(height: Int, width: Int, level: Int) -> (MLXArray, MLXArray) {
        var grid = [Float]()
        for i in 0 ..< height {
            for j in 0 ..< width {
                grid.append((Float(j) + 0.5) / Float(width))
                grid.append((Float(i) + 0.5) / Float(height))
            }
        }
        let gridArray = MLXArray(grid).reshaped([height * width, 2])
        let wh = MLXArray.ones([height * width, 2]) * (0.05 * pow(2.0, Float(level)))
        let proposal = concatenated([gridArray, wh], axis: 1)           // [S, 4]
        let valid = logicalAnd(proposal .> Float(0.01), proposal .< Float(0.99)).all(axis: -1, keepDims: true)
        let invalid = logicalNot(valid)                                 // [S, 1]
        let masked = MLX.where(invalid, MLXArray(Float(0)), proposal)
        return (masked, invalid)
    }

    /// The staged outputs of one forward, for parity localization and the public detection.
    struct Selection {
        let backboneMaps: [MLXArray]                  // per selected stage, NHWC (pre-projector)
        let projected: MLXArray                       // [1, Hp, Wp, d_model]
        let sourceFlatten: MLXArray                   // [1, S, d_model] the deformable value
        let encClassFull: MLXArray                    // [S, num_labels] first-stage class head, pre-mask
        let encCoordTopk: MLXArray                    // [Q, 4]
        let encClassTopk: MLXArray                    // [Q, num_labels]
        let initReference: MLXArray                   // [Q, 4] the mixed decoder-input refpoints
        let topkIndices: MLXArray                     // [Q]
        let shapes: [(Int, Int)]
    }

    /// `pixels` `[1, H, W, 3]` NHWC -> the query selection (up to and including the mixed init refpoints).
    /// `topkOverride` (the reference's `topk_ind`) isolates the decoder from the float-tie top-k.
    func select(_ pixels: MLXArray, topkOverride: MLXArray? = nil) -> Selection {
        let height = pixels.dim(1), width = pixels.dim(2)
        let backboneMaps = backbone.backbone(pixels, height: height, width: width)
        let projected = backbone.projector(backboneMaps)               // [1, Hp, Wp, d_model]
        let (hp, wp) = (projected.dim(1), projected.dim(2))
        let shapes = [(hp, wp)]
        let sourceFlatten = projected.reshaped([1, hp * wp, config.dModel])

        let (proposalsArray, invalidMask) = proposals(height: hp, width: wp, level: 0)   // [S,4], [S,1]
        let source = sourceFlatten[0]                                  // [S, d]
        let objectQueryEmbedding = MLX.where(broadcast(invalidMask, to: source.shape), MLXArray(Float(0)), source)

        // Group 0 (inference uses one group).
        var objectQuery = encOutput[0](objectQueryEmbedding)
        objectQuery = encOutputNorm[0](objectQuery)                    // [S, d]
        let encClassFull = encOutClassEmbed[0](objectQuery)           // [S, num_labels] pre-mask
        let deltaBbox = encOutBboxEmbed[0](objectQuery)               // [S, 4]
        let encCoord = nfkRFDetrRefineBboxes(reference: proposalsArray, deltas: deltaBbox)   // [S, 4]
        let maskedClass = MLX.where(broadcast(invalidMask, to: encClassFull.shape),
                                    MLXArray(-Float.greatestFiniteMagnitude), encClassFull)
        let scores = maskedClass.max(axis: -1)                        // [S]
        let order = argSort(-scores, axis: 0)
        let topk = topkOverride ?? order[0 ..< config.numQueries]     // [Q]
        let encCoordTopk = encCoord[topk]                             // [Q, 4]
        let objectQueryTopk = objectQuery[topk]                      // [Q, d]
        let encClassTopk = encOutClassEmbed[0](objectQueryTopk)      // [Q, num_labels]

        // Mixed queries: the learned reference points refined by the top-k coords (two_stage_len == Q,
        // so every learned refpoint is refined). reference_point_embed is refined; query_feat is the
        // decoder target (built in `decode`).
        let learnedRef = referencePointEmbed.weight[0 ..< config.numQueries]   // [Q, 4]
        let initReference = nfkRFDetrRefineBboxes(reference: encCoordTopk, deltas: learnedRef)

        return Selection(backboneMaps: backboneMaps, projected: projected, sourceFlatten: sourceFlatten,
                         encClassFull: encClassFull, encCoordTopk: encCoordTopk, encClassTopk: encClassTopk,
                         initReference: initReference, topkIndices: topk, shapes: shapes)
    }

    /// Runs the decoder over a selection, returning the last hidden state `[1, Q, d_model]`.
    func decode(_ selection: Selection, referenceOverride: MLXArray? = nil) -> MLXArray {
        let reference = (referenceOverride ?? selection.initReference).expandedDimensions(axis: 0)   // [1,Q,4]
        let target = queryFeat.weight[0 ..< config.numQueries].expandedDimensions(axis: 0)           // [1,Q,d]
        return decoder(target, value: selection.sourceFlatten, referencePoints: reference, shapes: selection.shapes)
    }
}

// MARK: - RfDetrForObjectDetection

public final class NFKMLXRFDetrNet: Module {
    @ModuleInfo(key: "model") var model: NFKRFDetrModel
    @ModuleInfo(key: "class_embed") var classEmbed: Linear
    @ModuleInfo(key: "bbox_embed") var bboxEmbed: NFKRFDetrMLP
    public let config: NFKMLXRFDetrConfiguration

    public init(_ config: NFKMLXRFDetrConfiguration) {
        self.config = config
        _model.wrappedValue = NFKRFDetrModel(config)
        _classEmbed.wrappedValue = Linear(config.dModel, config.numLabels)
        _bboxEmbed.wrappedValue = NFKRFDetrMLP(inputDim: config.dModel, hiddenDim: config.dModel, outputDim: 4, numLayers: 3)
    }

    /// The staged detection, for parity localization and post-processing.
    public struct Detection {
        public let backboneMaps: [MLXArray]           // per selected stage NHWC
        public let projected: MLXArray                // [1, Hp, Wp, d_model]
        public let encClassFull: MLXArray             // [S, num_labels]
        public let topkIndices: MLXArray              // [Q]
        public let initReference: MLXArray            // [Q, 4]
        public let decoderLast: MLXArray              // [1, Q, d_model]
        public let logits: MLXArray                   // [Q, num_labels]
        public let boxes: MLXArray                    // [Q, 4] direct cxcywh
    }

    /// `pixels` `[1, H, W, 3]` NHWC -> the full staged detection.
    public func callAsFunction(_ pixels: MLXArray) -> Detection { detection(pixels) }

    /// The staged detection. `topkOverride` (the reference's `topk_ind`) isolates the decoder and heads
    /// from the float-tie top-k selection, for released-weights parity.
    public func detection(_ pixels: MLXArray, topkOverride: MLXArray? = nil) -> Detection {
        let selection = model.select(pixels, topkOverride: topkOverride)
        let last = model.decode(selection)                            // [1, Q, d_model]
        let logits = classEmbed(last[0])                              // [Q, num_labels]
        let delta = bboxEmbed(last[0])                                // [Q, 4]
        let boxes = nfkRFDetrRefineBboxes(reference: selection.initReference, deltas: delta)
        return Detection(backboneMaps: selection.backboneMaps, projected: selection.projected,
                         encClassFull: selection.encClassFull, topkIndices: selection.topkIndices,
                         initReference: selection.initReference, decoderLast: last, logits: logits, boxes: boxes)
    }
}

// MARK: - Detection (post-processing) and backend

extension NFKMLXRFDetrNet {
    /// Detects objects in a bridged image `[H, W, 3]` (`0...1`): resize to the square input, normalize with
    /// ImageNet mean/std, forward, sigmoid the logits, threshold per query, map center boxes to normalized
    /// corners. DETR-family, so NO non-max suppression. The image processor squashes to the square input
    /// (`do_pad` is a no-op for a square resize), so a normalized box maps to the original frame unchanged.
    /// The resize is `resizeBilinear` where the reference uses PIL bilinear — a documented approximation;
    /// the network is at parity on the reference's own pixel values.
    public func detect(_ image: MLXArray, labels: [String]?) -> [NFKDetection] {
        let batched = image.ndim == 3
            ? image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]) : image
        let resized = NFKMLXResample.resizeBilinear(batched, height: config.inputResolution, width: config.inputResolution)
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 1, 3])
        let std = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 1, 3])
        let detection = self((resized - mean) / std)
        let scores = sigmoid(detection.logits); eval(scores)
        let boxes = detection.boxes; eval(boxes)
        let scoreValues = scores.asArray(Float.self)
        let boxValues = boxes.asArray(Float.self)
        let queries = detection.logits.dim(0)
        let classes = config.numLabels

        var results = [NFKDetection]()
        for query in 0 ..< queries {
            var bestClass = 0
            var bestScore: Float = 0
            for k in 0 ..< classes {
                let score = scoreValues[query * classes + k]
                if score > bestScore { bestScore = score; bestClass = k }
            }
            if bestScore < config.confidenceThreshold { continue }
            let cx = boxValues[query * 4], cy = boxValues[query * 4 + 1]
            let w = boxValues[query * 4 + 2], h = boxValues[query * 4 + 3]
            let minX = min(max(cx - w / 2, 0), 1), maxX = min(max(cx + w / 2, 0), 1)
            let minY = min(max(cy - h / 2, 0), 1), maxY = min(max(cy + h / 2, 0), 1)
            let rect = CGRect(x: CGFloat(minX), y: CGFloat(minY),
                              width: CGFloat(maxX - minX), height: CGFloat(maxY - minY))
            let label = labels.flatMap { bestClass < $0.count ? $0[bestClass] : nil }
            results.append(NFKDetection(label: label, classIndex: bestClass,
                                        confidence: Double(bestScore), boundingBox: rect))
        }
        return results
    }
}

/// Holds the network and labels for capture in the backend's `@Sendable` closure.
private final class NFKRFDetrHolder: @unchecked Sendable {
    let net: NFKMLXRFDetrNet
    let labels: [String]?
    init(_ net: NFKMLXRFDetrNet, labels: [String]?) { self.net = net; self.labels = labels }
}

/// An object-detection backend: an image under `NFKInputImage` produces detections under
/// `NFKOutputDetections`. At reference parity on the released weights (see the file header).
@objc(NFKMLXRFDetrBackend)
public final class NFKMLXRFDetrBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKRFDetrHolder
    private let identifier: String

    init(net: NFKMLXRFDetrNet, identifier: String, labels: [String]?) {
        holder = NFKRFDetrHolder(net, labels: labels)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result { return result }
        if let error = job.error { throw error }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = self.holder
        Task.detached(priority: .userInitiated) {
            do {
                guard let value = request.input(forKey: NFKInputImage) else {
                    throw NFKMLXError.unsupportedInput
                }
                let image = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
                let detections = holder.net.detect(image, labels: holder.labels)
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputDetections: detections]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

/// RF-DETR object detection as an InferKit backend, and its loader.
///
/// At reference parity on both the tiny config and the released `Roboflow/rf-detr-base` weights (see the
/// file header). The released file uses the ORIGINAL Roboflow naming (`backbone.0.encoder.encoder.*`,
/// `transformer.*`, `refpoint_embed`, a fused `self_attn.in_proj_*`); `remapReferenceKey` converts it to
/// the module names on device, so no offline conversion is needed — the same conversion transformers'
/// `_checkpoint_conversion_prefix_free` does at load, derived here by tensor-identity matching against
/// the converted state_dict.
@objc(NFKMLXRFDetr)
public final class NFKMLXRFDetr: NSObject {
    /// The registry name the model builds under.
    @objc public static let modelName = "rf-detr"

    /// Builds a detection backend directly from optional local weights. A nil `weightsURL` builds random
    /// weights (`isReady` is true). Run inference off the render thread.
    @objc(backendWithWeightsURL:labels:error:)
    public static func backend(weightsURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        let net = NFKMLXRFDetrNet(.base)
        if let weightsURL { try loadWeights(into: net, from: weightsURL) }
        net.train(false)
        return NFKMLXRFDetrBackend(net: net, identifier: modelName, labels: labels)
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend. Blocking on the network; run
    /// off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:labels:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url, labels: labels)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:labels:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, labels: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0, labels: labels) },
                               completionHandler: completionHandler)
    }

    /// Registers RF-DETR (`rf-detr`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:labels:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL, labels: nil)
        }
    }

    /// Converts one released Roboflow key to the module's name, or nil to drop / handle specially. The
    /// released file is prefix-free with the original naming: the DINOv2 backbone under
    /// `backbone.0.encoder.encoder.`, the projector under `backbone.0.projector.stages.0.` (a C2f at
    /// `stages.0.0` with `cv1`/`cv2`/`bn` and `m.N` bottlenecks, its channels-first LayerNorm at
    /// `stages.0.1`), the decoder under `transformer.decoder.` (`norm1`/`norm2`/`norm3` are the self-attn/
    /// cross-attn/final LayerNorms, `linear1`/`linear2` the MLP, `out_proj` the self-attn output), the
    /// two-stage heads under `transformer.enc_*`, and the top-level `class_embed`/`bbox_embed`/`query_feat`/
    /// `refpoint_embed`. The fused `self_attn.in_proj_*` is returned nil here and split in `loadWeights`.
    static func remapReferenceKey(_ key: String) -> String? {
        func replacePrefix(_ prefix: String, _ replacement: String) -> String {
            replacement + key.dropFirst(prefix.count)
        }
        if key.hasSuffix("num_batches_tracked") { return nil }
        if key.hasPrefix("class_embed.") || key.hasPrefix("bbox_embed.") { return key }
        if key == "query_feat.weight" { return "model.query_feat.weight" }
        if key == "refpoint_embed.weight" { return "model.reference_point_embed.weight" }
        if key.hasPrefix("backbone.0.encoder.encoder.") {
            return replacePrefix("backbone.0.encoder.encoder.", "model.backbone.backbone.")
        }
        if key.hasPrefix("backbone.0.projector.stages.0.") {
            let rest = String(key.dropFirst("backbone.0.projector.stages.0.".count))   // "1.*" or "0.*"
            if rest.hasPrefix("1.") {
                return "model.backbone.projector.layer_norm." + rest.dropFirst("1.".count)
            }
            var body = String(rest.dropFirst("0.".count))                              // "cv1.conv.*", "m.N.cv2.bn.*"
            if body.hasPrefix("m.") { body = "bottlenecks." + body.dropFirst("m.".count) }
            body = body.replacingOccurrences(of: "cv1", with: "conv1")
                       .replacingOccurrences(of: "cv2", with: "conv2")
                       .replacingOccurrences(of: ".bn.", with: ".norm.")
            return "model.backbone.projector.projector_layer." + body
        }
        if key.hasPrefix("transformer.decoder.norm.") {
            return replacePrefix("transformer.decoder.norm.", "model.decoder.layernorm.")
        }
        if key.hasPrefix("transformer.decoder.ref_point_head.") {
            return replacePrefix("transformer.decoder.ref_point_head.", "model.decoder.ref_point_head.")
        }
        if key.hasPrefix("transformer.decoder.layers.") {
            let tail = String(key.dropFirst("transformer.decoder.layers.".count))       // "N.<rest>"
            guard let dot = tail.firstIndex(of: ".") else { return nil }
            let layer = String(tail[..<dot])
            var rest = String(tail[tail.index(after: dot)...])
            if rest.hasPrefix("self_attn.in_proj_") { return nil }                       // split in loadWeights
            for (from, to) in [("norm1.", "self_attn_layer_norm."), ("norm2.", "cross_attn_layer_norm."),
                               ("norm3.", "layer_norm."), ("linear1.", "mlp.fc1."), ("linear2.", "mlp.fc2."),
                               ("self_attn.out_proj.", "self_attn.o_proj.")] where rest.hasPrefix(from) {
                rest = to + rest.dropFirst(from.count)
                break
            }
            return "model.decoder.layers.\(layer).\(rest)"                               // cross_attn.* passes through
        }
        if key.hasPrefix("transformer.enc_") {
            return replacePrefix("transformer.", "model.")
        }
        return nil
    }

    /// Loads the released safetensors into `net`, converting the original Roboflow naming with
    /// `remapReferenceKey`, splitting the fused `self_attn.in_proj_*` into q/k/v (packed `[q; k; v]`), and
    /// transposing 4-D convolution weights from PyTorch `[out, in, kH, kW]` to MLX `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXRFDetrNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var mapped = [(String, MLXArray)]()
        for (key, value) in checkpoint.arrays {
            if let range = key.range(of: ".self_attn.in_proj_") {
                // transformer.decoder.layers.N.self_attn.in_proj_(weight|bias) -> q/k/v_proj, packed [q;k;v].
                let layer = String(key[..<range.lowerBound]).split(separator: ".").last.map(String.init) ?? ""
                let suffix = String(key[range.upperBound...])                            // "weight" or "bias"
                let third = value.dim(0) / 3
                for (index, projection) in ["q", "k", "v"].enumerated() {
                    let slice = value[(index * third) ..< ((index + 1) * third)]
                    mapped.append(("model.decoder.layers.\(layer).self_attn.\(projection)_proj.\(suffix)", slice))
                }
                continue
            }
            guard let name = remapReferenceKey(key) else { continue }
            mapped.append((name, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
