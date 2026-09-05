//
//  NFKMLXBiRefNet.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// BiRefNet (ZhengPeng7, MIT), the high-resolution dichotomous-segmentation / background-removal model:
// a Swin-v1 backbone, a multi-scale context neck, and an ASPPDeformable decoder producing a single-
// channel matte. It is the MIT best-in-class cutout model, stronger than the shipped U2Net/MODNet.
//
// STATUS. COMPLETE and at REFERENCE PARITY on the RELEASED weights, validated seam by seam in
// `NFKMLXReferenceParityTests` against BiRefNet's own reference: the modulated deformable convolution vs
// torchvision (`testDeformableConv2dMatchesTorchvision`), the Swin-v1-L backbone
// (`testSwinBackboneMatchesTheReference`), the neck — mul_scl_ipt + cxt + the ASPPDeformable squeeze —
// (`testBiRefNetNeckMatchesTheReference`), the decoder (`testBiRefNetDecoderMatchesTheReference`), and
// the assembled model end to end (`testBiRefNetEndToEndMatchesTheReference`), every seam ~1e-12.
//
// The eval forward is: backbone -> multi-scale-input concat (`mul_scl_ipt`) -> context concat (`cxt`) ->
// squeeze -> four decoder stages (BasicDecBlk with ASPPDeformable) + lateral skips + dec_ipt injection +
// the eval-active gdt attention gating -> bilinear upsample -> a 1-channel logit, sigmoided at the
// consumer. Only the gdt prediction/label and the multi-scale-supervision heads are training-only.
// `NFKMLXDeformableConv2d` is the one novel op: a bilinear gather at the learned per-tap offsets, the
// RAFT/RT-DETR `take` idiom, since MLX has no deformable-convolution primitive.

/// A modulated deformable convolution (deform-conv-v2), the op BiRefNet's `ASPPDeformable` branches use.
///
/// torchvision's `deform_conv2d` samples the input bilinearly at a learned per-tap offset, scales each
/// sample by a learned per-tap modulator mask, and sums the samples through the regular convolution
/// weight. MLX has no deformable-convolution primitive, but the sampling is the same bilinear gather at
/// per-pixel coordinates that RAFT/RT-DETR express with `take` (a `grid_sample(padding_mode: "zeros")`),
/// so the whole op reduces to: for each kernel tap, gather the offset-shifted feature map, multiply by
/// that tap's mask, and accumulate a 1x1 matmul with the tap's slice of the weight.
final class NFKMLXDeformableConv2d: Module {
    @ModuleInfo(key: "regular_conv") var regularConv: Conv2d
    @ModuleInfo(key: "offset_conv") var offsetConv: Conv2d
    @ModuleInfo(key: "modulator_conv") var modulatorConv: Conv2d
    let kernel: (Int, Int)
    let stride: (Int, Int)
    let padding: (Int, Int)
    let dilation: (Int, Int)

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int, dilation: Int = 1, bias: Bool = false) {
        kernel = (kernelSize, kernelSize)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        // The offset and modulator convolutions share the regular conv's geometry so their fields land at
        // the output resolution, which is where torchvision reads them. The reference zero-inits both, so
        // an untrained module starts as an ordinary convolution; a released checkpoint carries the trained
        // weights. In `ASPPDeformable` the regular convolution is bias-free (`bias=False`).
        let k = IntOrPair(kernelSize), s = IntOrPair(stride), p = IntOrPair(padding), d = IntOrPair(dilation)
        _regularConv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: k,
                                           stride: s, padding: p, dilation: d, bias: bias)
        _offsetConv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: 2 * kernelSize * kernelSize, kernelSize: k,
                                          stride: s, padding: p, dilation: d, bias: true)
        _modulatorConv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: kernelSize * kernelSize, kernelSize: k,
                                             stride: s, padding: p, dilation: d, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let offset = offsetConv(x)
        let mask = 2 * sigmoid(modulatorConv(x))                        // the reference's 2 * sigmoid modulator, in [0, 4]
        return Self.deform(x, offset: offset, mask: mask, weight: regularConv.weight, bias: regularConv.bias,
                           kernel: kernel, stride: stride, padding: padding, dilation: dilation)
    }

    /// The deformable-convolution core. `x` is NHWC `[1, H, W, C]`, `offset` is `[1, OH, OW, 2*kH*kW]`
    /// with `(Δy, Δx)` interleaved per tap (torchvision's layout), `mask` is `[1, OH, OW, kH*kW]`, and
    /// `weight` is MLX-layout `[OC, kH, kW, C]`.
    static func deform(_ x: MLXArray, offset: MLXArray, mask: MLXArray, weight: MLXArray, bias: MLXArray?,
                       kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int), dilation: (Int, Int)) -> MLXArray {
        let (height, width, channels) = (x.shape[1], x.shape[2], x.shape[3])
        let (kh, kw) = kernel
        let outChannels = weight.shape[0]
        let outHeight = offset.shape[1]
        let outWidth = offset.shape[2]
        let outPixels = outHeight * outWidth

        let xFlat = x.reshaped([height * width, channels])
        let offsetFlat = offset.reshaped([outPixels, 2 * kh * kw])
        let maskFlat = mask.reshaped([outPixels, kh * kw])

        // Base sampling grid in input space: output pixel (oy, ox) maps to (oy*stride - pad), flattened
        // in row-major order to match the [outPixels, ...] offset/mask layout.
        var baseYValues = [Float]()
        var baseXValues = [Float]()
        baseYValues.reserveCapacity(outPixels)
        baseXValues.reserveCapacity(outPixels)
        for oy in 0 ..< outHeight {
            for ox in 0 ..< outWidth {
                baseYValues.append(Float(oy * stride.0 - padding.0))
                baseXValues.append(Float(ox * stride.1 - padding.1))
            }
        }
        let baseY = MLXArray(baseYValues)
        let baseX = MLXArray(baseXValues)

        var contributions = [MLXArray]()
        for ki in 0 ..< kh {
            for kj in 0 ..< kw {
                let tap = ki * kw + kj
                let deltaY = offsetFlat[0..., 2 * tap]                  // [outPixels]
                let deltaX = offsetFlat[0..., 2 * tap + 1]
                let sampleY = baseY + Float(ki * dilation.0) + deltaY
                let sampleX = baseX + Float(kj * dilation.1) + deltaX
                let sampled = bilinearSample(xFlat, sampleY: sampleY, sampleX: sampleX, height: height, width: width)
                let modulator = maskFlat[0..., tap].reshaped([outPixels, 1])
                let tapWeight = weight[0..., ki, kj, 0...]             // [OC, C]
                contributions.append(matmul(sampled * modulator, tapWeight.transposed(1, 0)))   // [outPixels, OC]
            }
        }
        var accumulated = contributions.dropFirst().reduce(contributions[0], +)
        if let bias {
            accumulated = accumulated + bias
        }
        return accumulated.reshaped([1, outHeight, outWidth, outChannels])
    }

    /// Bilinear sample of the NHWC-flattened `[H*W, C]` feature map at per-pixel `(sampleY, sampleX)`.
    /// A corner whose integer index lies outside the map contributes nothing, matching torchvision's
    /// `deform_conv2d` (and RAFT's `grid_sample(padding_mode: "zeros")`) — the two are numerically
    /// identical because both zero each out-of-range corner and weight it by the unclamped fraction.
    private static func bilinearSample(_ xFlat: MLXArray, sampleY: MLXArray, sampleX: MLXArray,
                                       height: Int, width: Int) -> MLXArray {
        let y0 = floor(sampleY), x0 = floor(sampleX)
        let y1 = y0 + 1, x1 = x0 + 1
        let weightX = (sampleX - x0).reshaped([-1, 1])
        let weightY = (sampleY - y0).reshaped([-1, 1])
        let rowStride = Int32(width)

        func inside(_ value: MLXArray, _ size: Int) -> MLXArray {
            let low: MLXArray = (value .>= MLXArray(Float(0))).asType(.float32)
            let high: MLXArray = (value .<= MLXArray(Float(size - 1))).asType(.float32)
            return low * high
        }
        func gather(_ yy: MLXArray, _ xx: MLXArray) -> MLXArray {
            let clampedRows: MLXArray = clip(yy, min: 0, max: Float(height - 1)).asType(.int32)
            let clampedCols: MLXArray = clip(xx, min: 0, max: Float(width - 1)).asType(.int32)
            let index: MLXArray = clampedRows * rowStride + clampedCols
            let valid: MLXArray = (inside(yy, height) * inside(xx, width)).reshaped([-1, 1])
            return xFlat.take(index, axis: 0) * valid
        }
        let top = gather(y0, x0) * (1 - weightX) + gather(y0, x1) * weightX
        let bottom = gather(y1, x0) * (1 - weightX) + gather(y1, x1) * weightX
        return top * (1 - weightY) + bottom * weightY
    }
}

// MARK: - Swin-v1 backbone

// BiRefNet's `swin_v1_l` backbone: a hierarchical Swin Transformer. The window-attention core
// (partition/reverse, cyclic shift, the relative-position index, and the shifted-window mask) is the
// SwinIR machinery — `NFKSwinOps` and `NFKSwinWindowAttention` — reused directly. What the backbone
// adds over SwinIR is per-block padding to the window multiple (the feature sizes are not window-
// divisible), the patch embedding, patch merging between stages, and a per-stage output norm.

/// The Swin block's MLP (`mlp.fc1` -> GELU -> `mlp.fc2`), a nested module so the keys nest as the
/// checkpoint's do (SwinIR flattens them to `mlp_fc1`/`mlp_fc2`; the backbone keeps `mlp.fc1`).
final class NFKSwinMlp: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dimensions: Int, hidden: Int) {
        _fc1.wrappedValue = Linear(dimensions, hidden)
        _fc2.wrappedValue = Linear(hidden, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

/// PatchEmbed: a 4x4 stride-4 convolution to `embedDimensions`, then a LayerNorm over the flattened
/// token sequence (`patch_norm=True`).
final class NFKSwinPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    private let embedDimensions: Int

    init(patchSize: Int, inChannels: Int, embedDimensions: Int) {
        self.embedDimensions = embedDimensions
        _proj.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: embedDimensions,
                                    kernelSize: IntOrPair(patchSize), stride: IntOrPair(patchSize))
        _norm.wrappedValue = LayerNorm(dimensions: embedDimensions)
    }

    /// `x`: `[1, H, W, 3]` NHWC. Returns the token sequence `[1, Wh*Ww, C]` and its grid `(Wh, Ww)`.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, Int, Int) {
        let projected = proj(x)                                          // [1, Wh, Ww, C]
        let (wh, ww) = (projected.shape[1], projected.shape[2])
        let tokens = norm(projected.reshaped([1, wh * ww, embedDimensions]))
        return (tokens, wh, ww)
    }
}

/// PatchMerging: concatenate the four 2x2 sub-grids (even/odd rows x even/odd columns, in the
/// reference's order), LayerNorm the `4C` result, and reduce to `2C` with a bias-free linear.
final class NFKSwinPatchMerging: Module {
    @ModuleInfo(key: "reduction") var reduction: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(dimensions: Int) {
        _reduction.wrappedValue = Linear(4 * dimensions, 2 * dimensions, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: 4 * dimensions)
    }

    /// `x`: `[1, H*W, C]`. Returns `[1, (H/2)*(W/2), 2C]`.
    func callAsFunction(_ x: MLXArray, height h: Int, width w: Int) -> MLXArray {
        let channels = x.shape[2]
        var feature = x.reshaped([1, h, w, channels])
        // Pad an odd side by one on the bottom/right, as the reference does before the 2x2 stride.
        let padB = h % 2, padR = w % 2
        if padB > 0 || padR > 0 {
            feature = MLX.padded(feature, widths: [IntOrPair(0), IntOrPair((0, padB)), IntOrPair((0, padR)), IntOrPair(0)])
        }
        let (hs, ws) = (h + padB, w + padR)
        // Reshaping [.., hs/2, 2, ws/2, 2, C] makes the even index of each size-2 axis the 0::2 slice and
        // the odd index the 1::2 slice, so the reference's x0..x3 are Int-indexed sub-grids with no
        // strided slicing. Concatenation order is x0(row-even,col-even), x1(row-odd,col-even),
        // x2(row-even,col-odd), x3(row-odd,col-odd).
        let grid = feature.reshaped([1, hs / 2, 2, ws / 2, 2, channels])
        let x0 = grid[0..., 0..., 0, 0..., 0, 0...]
        let x1 = grid[0..., 0..., 1, 0..., 0, 0...]
        let x2 = grid[0..., 0..., 0, 0..., 1, 0...]
        let x3 = grid[0..., 0..., 1, 0..., 1, 0...]
        let merged = concatenated([x0, x1, x2, x3], axis: 3).reshaped([1, (hs / 2) * (ws / 2), 4 * channels])
        return reduction(norm(merged))
    }
}

/// One Swin block: pre-norm shifted-window attention (padding to the window multiple, cyclic shift,
/// partition, attend, reverse, unshift, crop) and a pre-norm MLP.
final class NFKSwinBackboneBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NFKSwinWindowAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKSwinMlp

    private let windowSize: Int
    private let shift: Int

    init(dimensions: Int, heads: Int, windowSize: Int, shift: Int, mlpRatio: Int) {
        self.windowSize = windowSize
        self.shift = shift
        _norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        _attn.wrappedValue = NFKSwinWindowAttention(dimensions: dimensions, heads: heads, windowSize: windowSize)
        _norm2.wrappedValue = LayerNorm(dimensions: dimensions)
        _mlp.wrappedValue = NFKSwinMlp(dimensions: dimensions, hidden: dimensions * mlpRatio)
    }

    /// `x`: `[1, H*W, C]`. `mask` is the stage's shifted-window mask at the padded resolution, used only
    /// by the shifted (odd) blocks.
    func callAsFunction(_ x: MLXArray, height h: Int, width w: Int, mask: MLXArray) -> MLXArray {
        let channels = x.shape[2]
        let shortcut = x
        var feature = norm1(x).reshaped([1, h, w, channels])

        // Pad to a multiple of the window size, as the reference does before partitioning.
        let padB = (windowSize - h % windowSize) % windowSize
        let padR = (windowSize - w % windowSize) % windowSize
        if padB > 0 || padR > 0 {
            feature = MLX.padded(feature, widths: [IntOrPair(0), IntOrPair((0, padB)), IntOrPair((0, padR)), IntOrPair(0)])
        }
        let (hp, wp) = (h + padB, w + padR)

        feature = NFKSwinOps.roll(feature, shift: shift)
        let windows = NFKSwinOps.partition(feature, windowSize: windowSize)
        let attended = attn(windows, mask: shift > 0 ? mask : nil)
        var merged = NFKSwinOps.reverse(attended, windowSize: windowSize, height: hp, width: wp)
        merged = NFKSwinOps.unroll(merged, shift: shift)
        if padB > 0 || padR > 0 {
            merged = merged[0..., 0 ..< h, 0 ..< w, 0...]
        }

        let afterAttention = shortcut + merged.reshaped([1, h * w, channels])
        return afterAttention + mlp(norm2(afterAttention))
    }
}

/// One Swin stage (`BasicLayer`): a run of blocks at a fixed resolution, then an optional PatchMerging
/// downsample. Returns the stage output (pre-downsample) and the downsampled tokens for the next stage.
final class NFKSwinBackboneLayer: Module {
    @ModuleInfo(key: "blocks") var blocks: [NFKSwinBackboneBlock]
    @ModuleInfo(key: "downsample") var downsample: NFKSwinPatchMerging?

    private let windowSize: Int

    init(dimensions: Int, depth: Int, heads: Int, windowSize: Int, mlpRatio: Int, hasDownsample: Bool) {
        self.windowSize = windowSize
        _blocks.wrappedValue = (0 ..< depth).map { index in
            NFKSwinBackboneBlock(dimensions: dimensions, heads: heads, windowSize: windowSize,
                                 shift: index % 2 == 0 ? 0 : windowSize / 2, mlpRatio: mlpRatio)
        }
        _downsample.wrappedValue = hasDownsample ? NFKSwinPatchMerging(dimensions: dimensions) : nil
    }

    /// Returns `(stageOutput, nextTokens, nextHeight, nextWidth)`. `stageOutput` is `[1, H*W, C]`.
    func callAsFunction(_ x: MLXArray, height h: Int, width w: Int) -> (MLXArray, MLXArray, Int, Int) {
        let hp = ((h + windowSize - 1) / windowSize) * windowSize
        let wp = ((w + windowSize - 1) / windowSize) * windowSize
        let mask = NFKSwinOps.shiftMask(height: hp, width: wp, windowSize: windowSize, shift: windowSize / 2)

        var tokens = x
        for block in blocks {
            tokens = block(tokens, height: h, width: w, mask: mask)
        }
        if let downsample {
            let down = downsample(tokens, height: h, width: w)
            return (tokens, down, (h + 1) / 2, (w + 1) / 2)
        }
        return (tokens, tokens, h, w)
    }
}

/// BiRefNet's `swin_v1_l` backbone. Returns the four stage feature maps as NHWC `[1, H_i, W_i, C_i]`
/// with `C = [192, 384, 768, 1536]` at strides 4/8/16/32.
final class NFKMLXSwinBackbone: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: NFKSwinPatchEmbed
    @ModuleInfo(key: "layers") var layers: [NFKSwinBackboneLayer]
    @ModuleInfo(key: "norm0") var norm0: LayerNorm
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    let numFeatures: [Int]

    init(embedDimensions: Int = 192, depths: [Int] = [2, 2, 18, 2], heads: [Int] = [6, 12, 24, 48],
         windowSize: Int = 12, mlpRatio: Int = 4) {
        let features = (0 ..< depths.count).map { embedDimensions * (1 << $0) }
        numFeatures = features
        _patchEmbed.wrappedValue = NFKSwinPatchEmbed(patchSize: 4, inChannels: 3, embedDimensions: embedDimensions)
        _layers.wrappedValue = (0 ..< depths.count).map { stage in
            NFKSwinBackboneLayer(dimensions: features[stage], depth: depths[stage], heads: heads[stage],
                                 windowSize: windowSize, mlpRatio: mlpRatio, hasDownsample: stage < depths.count - 1)
        }
        _norm0.wrappedValue = LayerNorm(dimensions: numFeatures[0])
        _norm1.wrappedValue = LayerNorm(dimensions: numFeatures[1])
        _norm2.wrappedValue = LayerNorm(dimensions: numFeatures[2])
        _norm3.wrappedValue = LayerNorm(dimensions: numFeatures[3])
    }

    /// `x`: `[1, H, W, 3]` NHWC in the backbone's normalized input space.
    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var (tokens, h, w) = patchEmbed(x)
        let norms = [norm0, norm1, norm2, norm3]
        var outputs = [MLXArray]()
        for (stage, layer) in layers.enumerated() {
            let (stageOutput, nextTokens, nextHeight, nextWidth) = layer(tokens, height: h, width: w)
            let normed = norms[stage](stageOutput)
            outputs.append(normed.reshaped([1, h, w, numFeatures[stage]]))
            tokens = nextTokens
            h = nextHeight
            w = nextWidth
        }
        return outputs
    }
}

// MARK: - ASPPDeformable, BasicDecBlk, and the neck

/// A no-parameter placeholder occupying an `nn.Sequential` index so the surrounding module keys line up
/// with the checkpoint (`global_avg_pool.1`/`.2` are the conv and BatchNorm; 0 and 3 are the pool and
/// ReLU, which carry no weights).
private final class NFKBiRefNetMarker: Module {}

/// One ASPPDeformable branch: a deformable convolution, BatchNorm, and ReLU (`_ASPPModuleDeformable`).
final class NFKASPPModuleDeformable: Module {
    @ModuleInfo(key: "atrous_conv") var atrousConv: NFKMLXDeformableConv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm

    init(inChannels: Int, planes: Int, kernel: Int, padding: Int) {
        _atrousConv.wrappedValue = NFKMLXDeformableConv2d(inChannels: inChannels, outChannels: planes,
                                                          kernelSize: kernel, stride: 1, padding: padding, bias: false)
        _bn.wrappedValue = BatchNorm(featureCount: planes)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(bn(atrousConv(x)))
    }
}

/// ASPPDeformable: a 1x1 deformable branch, three deformable branches (kernels 1/3/7), and a global
/// average-pool branch, concatenated (`256 * 5 = 1280`) and projected back to `in_channels`.
final class NFKASPPDeformable: Module {
    @ModuleInfo(key: "aspp1") var aspp1: NFKASPPModuleDeformable
    @ModuleInfo(key: "aspp_deforms") var asppDeforms: [NFKASPPModuleDeformable]
    @ModuleInfo(key: "global_avg_pool") var globalAvgPool: [Module]
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm

    init(inChannels: Int) {
        let inter = 256
        _aspp1.wrappedValue = NFKASPPModuleDeformable(inChannels: inChannels, planes: inter, kernel: 1, padding: 0)
        _asppDeforms.wrappedValue = [1, 3, 7].map {
            NFKASPPModuleDeformable(inChannels: inChannels, planes: inter, kernel: $0, padding: $0 / 2)
        }
        _globalAvgPool.wrappedValue = [NFKBiRefNetMarker(),
                                       Conv2d(inputChannels: inChannels, outputChannels: inter, kernelSize: 1, bias: false),
                                       BatchNorm(featureCount: inter),
                                       NFKBiRefNetMarker()]
        _conv1.wrappedValue = Conv2d(inputChannels: inter * (2 + 3), outputChannels: inChannels, kernelSize: 1, bias: false)
        _bn1.wrappedValue = BatchNorm(featureCount: inChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let branch1 = aspp1(x)
        let deforms = asppDeforms.map { $0(x) }
        guard let poolConv = globalAvgPool[1] as? Conv2d, let poolBN = globalAvgPool[2] as? BatchNorm else {
            return branch1
        }
        var pooled = relu(poolBN(poolConv(x.mean(axes: [1, 2], keepDims: true))))   // AdaptiveAvgPool2d(1) -> conv -> bn -> relu
        pooled = NFKMLXResample.resizeBilinearAlignCorners(pooled, height: branch1.shape[1], width: branch1.shape[2])
        let concat = concatenated([branch1] + deforms + [pooled], axis: 3)
        return relu(bn1(conv1(concat)))                                            // Dropout(0.5) is an eval no-op
    }
}

/// BasicDecBlk: conv -> BatchNorm -> ReLU -> ASPPDeformable -> conv -> BatchNorm, with a fixed 64-channel
/// interior. Used as the squeeze module and every decoder-stage block.
final class NFKBasicDecBlk: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "dec_att") var decAtt: NFKASPPDeformable
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    @ModuleInfo(key: "bn_in") var bnIn: BatchNorm
    @ModuleInfo(key: "bn_out") var bnOut: BatchNorm

    init(inChannels: Int, outChannels: Int) {
        let inter = 64
        _convIn.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: inter, kernelSize: 3, padding: 1)
        _decAtt.wrappedValue = NFKASPPDeformable(inChannels: inter)
        _convOut.wrappedValue = Conv2d(inputChannels: inter, outputChannels: outChannels, kernelSize: 3, padding: 1)
        _bnIn.wrappedValue = BatchNorm(featureCount: inter)
        _bnOut.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var feature = relu(bnIn(convIn(x)))
        feature = decAtt(feature)
        return bnOut(convOut(feature))
    }
}

/// The BiRefNet encoder: the Swin-v1-L backbone, the multi-scale-input concatenation (`mul_scl_ipt`),
/// the context concatenation (`cxt`), and the squeeze module.
final class NFKMLXBiRefNetEncoder: Module {
    @ModuleInfo(key: "bb") var backbone: NFKMLXSwinBackbone
    @ModuleInfo(key: "squeeze_module") var squeezeModule: [NFKBasicDecBlk]

    override init() {
        _backbone.wrappedValue = NFKMLXSwinBackbone()
        // channels[0] + sum(cxt) = 3072 + (384+768+1536) = 5760 -> channels[0] = 3072.
        _squeezeModule.wrappedValue = [NFKBasicDecBlk(inChannels: 5760, outChannels: 3072)]
    }

    /// `mul_scl_ipt == 'cat'`: run the backbone at full and half resolution and concatenate each stage's
    /// features, doubling the channels. Returns the four doubled stage features NHWC.
    func multiScale(_ pixels: MLXArray) -> [MLXArray] {
        let full = backbone(pixels)
        let (h, w) = (pixels.shape[1], pixels.shape[2])
        let half = NFKMLXResample.resizeBilinearAlignCorners(pixels, height: h / 2, width: w / 2)
        let low = backbone(half)
        return (0 ..< 4).map { level in
            let up = NFKMLXResample.resizeBilinearAlignCorners(low[level], height: full[level].shape[1], width: full[level].shape[2])
            return concatenated([full[level], up], axis: 3)
        }
    }

    /// The `cxt` context concat: x1/x2/x3 interpolated to x4's resolution and concatenated with x4.
    func contextConcat(_ features: [MLXArray]) -> MLXArray {
        let (h4, w4) = (features[3].shape[1], features[3].shape[2])
        let upsampled = (0 ..< 3).map { NFKMLXResample.resizeBilinearAlignCorners(features[$0], height: h4, width: w4) }
        return concatenated(upsampled + [features[3]], axis: 3)
    }

    /// Applies the squeeze BasicDecBlk to the context-concatenated x4.
    func squeeze(_ contextX4: MLXArray) -> MLXArray {
        squeezeModule[0](contextX4)
    }
}

// MARK: - Decoder

/// BasicLatBlk: a bare 1x1 convolution used for the lateral skip connections.
final class NFKBasicLatBlk: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(inChannels: Int, outChannels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// SimpleConvs: two 3x3 convolutions with no activation between (`dec_ipt` injection block).
final class NFKSimpleConvs: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    init(inChannels: Int, outChannels: Int, interChannels: Int = 64) {
        _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: interChannels, kernelSize: 3, padding: 1)
        _convOut.wrappedValue = Conv2d(inputChannels: interChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { convOut(conv1(x)) }
}

/// BiRefNet's Decoder: four coarse-to-fine BasicDecBlk stages with lateral skips, dec_ipt image-patch
/// injection, the gdt attention gating (eval-active), bilinear upsampling, and a 1-channel output head.
/// The training-only paths (gdt pred/label, ms-supervision) are omitted.
final class NFKMLXBiRefNetDecoder: Module {
    @ModuleInfo(key: "decoder_block4") var decoderBlock4: NFKBasicDecBlk
    @ModuleInfo(key: "decoder_block3") var decoderBlock3: NFKBasicDecBlk
    @ModuleInfo(key: "decoder_block2") var decoderBlock2: NFKBasicDecBlk
    @ModuleInfo(key: "decoder_block1") var decoderBlock1: NFKBasicDecBlk
    @ModuleInfo(key: "lateral_block4") var lateralBlock4: NFKBasicLatBlk
    @ModuleInfo(key: "lateral_block3") var lateralBlock3: NFKBasicLatBlk
    @ModuleInfo(key: "lateral_block2") var lateralBlock2: NFKBasicLatBlk
    @ModuleInfo(key: "ipt_blk5") var iptBlk5: NFKSimpleConvs
    @ModuleInfo(key: "ipt_blk4") var iptBlk4: NFKSimpleConvs
    @ModuleInfo(key: "ipt_blk3") var iptBlk3: NFKSimpleConvs
    @ModuleInfo(key: "ipt_blk2") var iptBlk2: NFKSimpleConvs
    @ModuleInfo(key: "ipt_blk1") var iptBlk1: NFKSimpleConvs
    @ModuleInfo(key: "gdt_convs_4") var gdtConvs4: [Module]
    @ModuleInfo(key: "gdt_convs_3") var gdtConvs3: [Module]
    @ModuleInfo(key: "gdt_convs_2") var gdtConvs2: [Module]
    @ModuleInfo(key: "gdt_convs_attn_4") var gdtConvsAttn4: [Module]
    @ModuleInfo(key: "gdt_convs_attn_3") var gdtConvsAttn3: [Module]
    @ModuleInfo(key: "gdt_convs_attn_2") var gdtConvsAttn2: [Module]
    @ModuleInfo(key: "conv_out1") var convOut1: [Module]

    override init() {
        // channels = lateral_channels_in_collection doubled by mul_scl_ipt = [3072, 1536, 768, 384].
        let channels = [3072, 1536, 768, 384]
        _decoderBlock4.wrappedValue = NFKBasicDecBlk(inChannels: channels[0] + channels[0] / 8, outChannels: channels[1])
        _decoderBlock3.wrappedValue = NFKBasicDecBlk(inChannels: channels[1] + channels[0] / 8, outChannels: channels[2])
        _decoderBlock2.wrappedValue = NFKBasicDecBlk(inChannels: channels[2] + channels[1] / 8, outChannels: channels[3])
        _decoderBlock1.wrappedValue = NFKBasicDecBlk(inChannels: channels[3] + channels[2] / 8, outChannels: channels[3] / 2)
        _lateralBlock4.wrappedValue = NFKBasicLatBlk(inChannels: channels[1], outChannels: channels[1])
        _lateralBlock3.wrappedValue = NFKBasicLatBlk(inChannels: channels[2], outChannels: channels[2])
        _lateralBlock2.wrappedValue = NFKBasicLatBlk(inChannels: channels[3], outChannels: channels[3])
        // dec_ipt patch channels are 3 * grid^2 where grid = the stride ratio: 32/16/8/4/1 -> 3072/768/192/48/3.
        _iptBlk5.wrappedValue = NFKSimpleConvs(inChannels: 3 * 32 * 32, outChannels: channels[0] / 8)
        _iptBlk4.wrappedValue = NFKSimpleConvs(inChannels: 3 * 16 * 16, outChannels: channels[0] / 8)
        _iptBlk3.wrappedValue = NFKSimpleConvs(inChannels: 3 * 8 * 8, outChannels: channels[1] / 8)
        _iptBlk2.wrappedValue = NFKSimpleConvs(inChannels: 3 * 4 * 4, outChannels: channels[2] / 8)
        _iptBlk1.wrappedValue = NFKSimpleConvs(inChannels: 3, outChannels: channels[3] / 8)
        func gdtConvs(_ inChannels: Int) -> [Module] {
            [Conv2d(inputChannels: inChannels, outputChannels: 16, kernelSize: 3, padding: 1), BatchNorm(featureCount: 16), NFKBiRefNetMarker()]
        }
        _gdtConvs4.wrappedValue = gdtConvs(channels[1])
        _gdtConvs3.wrappedValue = gdtConvs(channels[2])
        _gdtConvs2.wrappedValue = gdtConvs(channels[3])
        _gdtConvsAttn4.wrappedValue = [Conv2d(inputChannels: 16, outputChannels: 1, kernelSize: 1)]
        _gdtConvsAttn3.wrappedValue = [Conv2d(inputChannels: 16, outputChannels: 1, kernelSize: 1)]
        _gdtConvsAttn2.wrappedValue = [Conv2d(inputChannels: 16, outputChannels: 1, kernelSize: 1)]
        _convOut1.wrappedValue = [Conv2d(inputChannels: channels[3] / 2 + channels[3] / 8, outputChannels: 1, kernelSize: 1)]
    }

    /// image2patches with `b c (hg h) (wg w) -> b (c hg wg) h w` in NHWC: split H,W by the stride-ratio
    /// grid and fold the grid into the channel axis, then resample to the reference resolution (an
    /// identity when the sizes divide evenly). `image` and `reference` are NHWC.
    private func imagePatches(_ image: MLXArray, reference: MLXArray) -> MLXArray {
        let (h, w, c) = (image.shape[1], image.shape[2], image.shape[3])
        let (rh, rw) = (reference.shape[1], reference.shape[2])
        let (hg, wg) = (h / rh, w / rw)
        let (ph, pw) = (h / hg, w / wg)
        let patches = image.reshaped([1, hg, ph, wg, pw, c]).transposed(0, 2, 4, 5, 1, 3).reshaped([1, ph, pw, c * hg * wg])
        return NFKMLXResample.resizeBilinearAlignCorners(patches, height: rh, width: rw)
    }

    private func inject(_ image: MLXArray, into feature: MLXArray, block: NFKSimpleConvs) -> MLXArray {
        concatenated([feature, block(imagePatches(image, reference: feature))], axis: 3)
    }

    /// The gdt attention gate: gdt_convs (conv -> BN -> ReLU) then gdt_convs_attn (conv) -> sigmoid.
    private func gate(_ convs: [Module], _ attn: [Module], _ p: MLXArray) -> MLXArray {
        guard let conv = convs[0] as? Conv2d, let bn = convs[1] as? BatchNorm, let attnConv = attn[0] as? Conv2d else {
            return p
        }
        let features = relu(bn(conv(p)))
        return p * sigmoid(attnConv(features))
    }

    private func resize(_ x: MLXArray, like reference: MLXArray) -> MLXArray {
        NFKMLXResample.resizeBilinearAlignCorners(x, height: reference.shape[1], width: reference.shape[2])
    }

    /// Returns the full-resolution 1-channel logit and the four pre-gate decoder-block outputs.
    /// `image` is the normalized input NHWC; `x1/x2/x3` are the doubled stage features; `x4` is squeezed.
    func callAsFunction(_ image: MLXArray, _ x1: MLXArray, _ x2: MLXArray, _ x3: MLXArray, _ x4: MLXArray)
        -> (logit: MLXArray, blocks: [MLXArray]) {
        var p4 = decoderBlock4(inject(image, into: x4, block: iptBlk5))
        let block4 = p4
        p4 = gate(gdtConvs4, gdtConvsAttn4, p4)
        var lateral = resize(p4, like: x3) + lateralBlock4(x3)

        var p3 = decoderBlock3(inject(image, into: lateral, block: iptBlk4))
        let block3 = p3
        p3 = gate(gdtConvs3, gdtConvsAttn3, p3)
        lateral = resize(p3, like: x2) + lateralBlock3(x2)

        var p2 = decoderBlock2(inject(image, into: lateral, block: iptBlk3))
        let block2 = p2
        p2 = gate(gdtConvs2, gdtConvsAttn2, p2)
        lateral = resize(p2, like: x1) + lateralBlock2(x1)

        let p1 = decoderBlock1(inject(image, into: lateral, block: iptBlk2))
        var upsampled = resize(p1, like: image)
        upsampled = inject(image, into: upsampled, block: iptBlk1)
        guard let head = convOut1[0] as? Conv2d else { return (upsampled, [block4, block3, block2, p1]) }
        return (head(upsampled), [block4, block3, block2, p1])
    }
}

// MARK: - Full model and background-removal backend

/// The assembled BiRefNet: the encoder (backbone + neck + squeeze) and the decoder, run end to end.
/// Kept as two composed modules rather than one so the released `bb.*`/`squeeze_module.*` and `decoder.*`
/// key namespaces load without renaming, and the neck and decoder parity tests exercise each directly.
final class NFKMLXBiRefNetModel {
    let encoder: NFKMLXBiRefNetEncoder
    let decoder: NFKMLXBiRefNetDecoder

    init() {
        encoder = NFKMLXBiRefNetEncoder()
        decoder = NFKMLXBiRefNetDecoder()
    }

    /// Normalized input `[1, H, W, 3]` -> the full-resolution 1-channel logit `[1, H, W, 1]`.
    func logit(_ pixels: MLXArray) -> MLXArray {
        let features = encoder.multiScale(pixels)
        let contextX4 = encoder.contextConcat(features)
        let squeezed = encoder.squeeze(contextX4)
        return decoder(pixels, features[0], features[1], features[2], squeezed).logit
    }
}

private final class NFKMLXBiRefNetHolder: @unchecked Sendable {
    let model: NFKMLXBiRefNetModel
    init(_ model: NFKMLXBiRefNetModel) { self.model = model }
}

/// BiRefNet (ZhengPeng7, MIT): high-resolution dichotomous segmentation for background removal, through
/// `NFKMLXMattingBackend`. A plate in, a straight (original) foreground with the predicted alpha out.
@objc(NFKMLXBiRefNet)
public final class NFKMLXBiRefNet: NSObject {

    /// The registry name and the resolution the model is trained at.
    @objc public static let modelName = "birefnet"
    @objc public static let inputSize = 1024

    /// Builds a BiRefNet background-removal backend from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true). Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let model = NFKMLXBiRefNetModel()
        if let weightsURL {
            try loadWeights(into: model, from: weightsURL)
        }
        let holder = NFKMLXBiRefNetHolder(model)
        var configuration = NFKMattingConfiguration()
        configuration.emitsMatte = true
        configuration.plateChannels = 3
        return NFKMLXMattingBackend(identifier: modelName, isReady: true, configuration: configuration) { plate, _ in
            // BiRefNet resizes to 1024x1024, scales to [0,1], and ImageNet-normalizes. The resize is
            // CoreGraphics-bilinear here rather than the reference's PIL resample, so a consumer alpha is
            // a documented approximation; the network itself is at reference parity on identical pixels.
            let (height, width) = (plate.shape[0], plate.shape[1])
            let batched = plate.reshaped([1, height, width, 3])
            let resized = NFKMLXResample.resizeBilinear(batched, height: inputSize, width: inputSize)
            let mean = MLXArray([Float(0.485), 0.456, 0.406])
            let std = MLXArray([Float(0.229), 0.224, 0.225])
            let logit = holder.model.logit((resized - mean) / std)
            var alpha = sigmoid(logit)
            alpha = NFKMLXResample.resizeBilinear(alpha, height: height, width: width).reshaped([height, width, 1])
            return concatenated([plate, alpha], axis: 2)               // [H, W, 4]: straight foreground + matte
        }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required. Blocking on the
    /// network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers `birefnet` with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Loads the released `model.safetensors` (or a raw checkpoint), routing `bb.*`/`squeeze_module.*`
    /// to the encoder and `decoder.*` to the decoder. The relative-position-index buffers and the
    /// training-only decoder heads (gdt prediction, ms-supervision) are dropped; 4-D convolution weights
    /// transpose to MLX's layout, and the fp16 release upcasts to float32.
    static func loadWeights(into model: NFKMLXBiRefNetModel, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        func prepared(_ value: MLXArray) -> MLXArray {
            (checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value).asType(.float32)
        }
        let encoderWeights = raw.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("bb.") || key.hasPrefix("squeeze_module.") else { return nil }
            guard !key.hasSuffix("relative_position_index"), !key.hasSuffix("attn_mask"), !key.hasSuffix("num_batches_tracked") else { return nil }
            return (key, prepared(value))
        }
        try NFKMLXWeights.apply(encoderWeights, to: model.encoder)
        let decoderWeights = raw.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("decoder.") else { return nil }
            let name = String(key.dropFirst("decoder.".count))
            guard !name.hasPrefix("gdt_convs_pred"), !name.hasPrefix("conv_ms_spvn"), !name.hasSuffix("num_batches_tracked") else { return nil }
            return (name, prepared(value))
        }
        try NFKMLXWeights.apply(decoderWeights, to: model.decoder)
        // The BatchNorm layers in the squeeze and decoder must read their running statistics, not the
        // batch's — without eval mode a batch of one normalizes to noise.
        model.encoder.train(false)
        model.decoder.train(false)
    }
}
