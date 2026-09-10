//
//  NFKMLXYOLOBlocks.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXNN

// The building blocks the YOLO generations after v8 introduce. Each mirrors one class in
// ultralytics' `nn/modules`, keeping the reference's own submodule names so a released checkpoint's
// `model.N.…` keys land without a remap. `NFKYOLOConv`, `NFKYOLOSPPF` and `NFKYOLODFL` are shared
// with the v8 port. Tensors flow NHWC, so the reference's channel axis is 3 here.

/// The reference `Bottleneck`, in the general form the later stages need: a `k[0]` convolution to a
/// hidden width and a `k[1]` convolution back, with a residual when the widths allow it. The v8 port's
/// `NFKYOLOBottleneck` is the fixed 3×3/3×3, hidden-equals-output case.
final class NFKYOLOWideBottleneck: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    private let usesResidual: Bool

    init(inChannels: Int, outChannels: Int, shortcut: Bool, kernels: (Int, Int) = (3, 3),
         expansion: Float = 0.5) {
        let hidden = Int(Float(outChannels) * expansion)
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: hidden, kernel: kernels.0)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: hidden, outChannels: outChannels, kernel: kernels.1)
        usesResidual = shortcut && inChannels == outChannels
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = cv2(cv1(x))
        return usesResidual ? x + out : out
    }
}

/// The reference `C3`: two 1×1 branches, `n` bottlenecks down one of them, and a 1×1 fuse.
final class NFKYOLOC3: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "cv3") var cv3: NFKYOLOConv
    @ModuleInfo(key: "m") var m: [NFKYOLOWideBottleneck]

    /// `kernels` is `((1, 1), (3, 3))` for `C3` and `(3, 3)` for the `C3k` subclass.
    init(inChannels: Int, outChannels: Int, repeats: Int, shortcut: Bool,
         expansion: Float = 0.5, kernels: (Int, Int) = (1, 3)) {
        let hidden = Int(Float(outChannels) * expansion)
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: hidden)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: hidden)
        _cv3.wrappedValue = NFKYOLOConv(inChannels: hidden * 2, outChannels: outChannels)
        _m.wrappedValue = (0 ..< repeats).map { _ in
            NFKYOLOWideBottleneck(inChannels: hidden, outChannels: hidden, shortcut: shortcut,
                                  kernels: kernels, expansion: 1.0)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var deep = cv1(x)
        for bottleneck in m {
            deep = bottleneck(deep)
        }
        return cv3(concatenated([deep, cv2(x)], axis: 3))
    }
}

/// The attention the PSA blocks run: a fused qkv convolution, a scaled dot product over the flattened
/// map, and a depthwise positional encoding added to the values.
final class NFKYOLOAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: NFKYOLOConv
    @ModuleInfo(key: "proj") var proj: NFKYOLOConv
    @ModuleInfo(key: "pe") var pe: NFKYOLOConv

    private let heads: Int
    private let headDimensions: Int
    private let keyDimensions: Int

    init(dimensions: Int, heads: Int, attentionRatio: Float = 0.5) {
        self.heads = heads
        headDimensions = dimensions / heads
        keyDimensions = Int(Float(headDimensions) * attentionRatio)
        let width = dimensions + keyDimensions * heads * 2
        _qkv.wrappedValue = NFKYOLOConv(inChannels: dimensions, outChannels: width, activates: false)
        _proj.wrappedValue = NFKYOLOConv(inChannels: dimensions, outChannels: dimensions, activates: false)
        _pe.wrappedValue = NFKYOLOConv(inChannels: dimensions, outChannels: dimensions, kernel: 3,
                                       groups: dimensions, activates: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (height, width, channels) = (x.shape[1], x.shape[2], x.shape[3])
        let positions = height * width
        // The reference reads `[B, heads, key·2 + head, N]`, so the head axis is the outer one and the
        // three parts are consecutive spans within each head.
        let projected = qkv(x).reshaped([1, positions, heads, keyDimensions * 2 + headDimensions])
            .transposed(0, 2, 3, 1)
        let queries = projected[0..., 0..., 0 ..< keyDimensions, 0...]
        let keys = projected[0..., 0..., keyDimensions ..< (keyDimensions * 2), 0...]
        let values = projected[0..., 0..., (keyDimensions * 2)..., 0...]

        let scale = 1.0 / sqrtf(Float(keyDimensions))
        let attention = softmax((queries * scale).transposed(0, 1, 3, 2).matmul(keys), axis: -1)
        let attended = values.matmul(attention.transposed(0, 1, 3, 2))
            .reshaped([1, channels, height, width]).transposed(0, 2, 3, 1)
        let encoded = values.reshaped([1, channels, height, width]).transposed(0, 2, 3, 1)
        return proj(attended + pe(encoded))
    }
}

/// The reference `PSABlock`: pre-residual attention, then a pre-residual two-convolution feed-forward.
final class NFKYOLOPSABlock: Module {
    @ModuleInfo(key: "attn") var attn: NFKYOLOAttention
    @ModuleInfo(key: "ffn") var ffn: [NFKYOLOConv]
    private let usesResidual: Bool

    init(channels: Int, heads: Int, shortcut: Bool = true) {
        _attn.wrappedValue = NFKYOLOAttention(dimensions: channels, heads: heads)
        _ffn.wrappedValue = [NFKYOLOConv(inChannels: channels, outChannels: channels * 2),
                             NFKYOLOConv(inChannels: channels * 2, outChannels: channels, activates: false)]
        usesResidual = shortcut
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = usesResidual ? x + attn(x) : attn(x)
        let projected = ffn[1](ffn[0](out))
        out = usesResidual ? out + projected : projected
        return out
    }
}

/// The reference `C2PSA`: a 1×1 split, `n` PSA blocks down the second half, and a 1×1 fuse.
final class NFKYOLOC2PSA: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "m") var m: [NFKYOLOPSABlock]
    private let hidden: Int

    init(channels: Int, repeats: Int, expansion: Float = 0.5) {
        let half = Int(Float(channels) * expansion)
        hidden = half
        _cv1.wrappedValue = NFKYOLOConv(inChannels: channels, outChannels: half * 2)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: half * 2, outChannels: channels)
        _m.wrappedValue = (0 ..< repeats).map { _ in
            NFKYOLOPSABlock(channels: half, heads: max(half / 64, 1))
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let split = cv1(x)
        let first = split[0..., 0..., 0..., 0 ..< hidden]
        var second = split[0..., 0..., 0..., hidden ..< hidden * 2]
        for block in m {
            second = block(second)
        }
        return cv2(concatenated([first, second], axis: 3))
    }
}

/// The reference `PSA` (YOLOv10): one attention and one feed-forward on the second half, without the
/// `C2PSA` block list.
final class NFKYOLOPSA: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "attn") var attn: NFKYOLOAttention
    @ModuleInfo(key: "ffn") var ffn: [NFKYOLOConv]
    private let hidden: Int

    init(channels: Int, expansion: Float = 0.5) {
        hidden = Int(Float(channels) * expansion)
        _cv1.wrappedValue = NFKYOLOConv(inChannels: channels, outChannels: hidden * 2)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: hidden * 2, outChannels: channels)
        _attn.wrappedValue = NFKYOLOAttention(dimensions: hidden, heads: max(hidden / 64, 1))
        _ffn.wrappedValue = [NFKYOLOConv(inChannels: hidden, outChannels: hidden * 2),
                             NFKYOLOConv(inChannels: hidden * 2, outChannels: hidden, activates: false)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let split = cv1(x)
        let first = split[0..., 0..., 0..., 0 ..< hidden]
        var second = split[0..., 0..., 0..., hidden ..< hidden * 2]
        second = second + attn(second)
        second = second + ffn[1](ffn[0](second))
        return cv2(concatenated([first, second], axis: 3))
    }
}

/// The reference `C3k2`: `C2f` whose repeated unit is a plain bottleneck, a `C3k`, or a
/// bottleneck-plus-PSA pair, chosen by the release's scale and the stage's flags.
final class NFKYOLOC3k2: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "m") var m: [Module]
    private let hidden: Int

    /// `c3k` swaps each unit for a `C3k` rather than a plain bottleneck.
    init(inChannels: Int, outChannels: Int, repeats: Int, c3k: Bool, expansion: Float = 0.5,
         shortcut: Bool = true) {
        let half = Int(Float(outChannels) * expansion)
        hidden = half
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: half * 2)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: half * (2 + repeats), outChannels: outChannels)
        _m.wrappedValue = (0 ..< repeats).map { _ -> Module in
            c3k ? NFKYOLOC3(inChannels: half, outChannels: half, repeats: 2, shortcut: shortcut,
                            kernels: (3, 3))
                : NFKYOLOWideBottleneck(inChannels: half, outChannels: half, shortcut: shortcut)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let split = cv1(x)
        var parts = [split[0..., 0..., 0..., 0 ..< hidden],
                     split[0..., 0..., 0..., hidden ..< hidden * 2]]
        for unit in m {
            parts.append(NFKYOLOC3k2.run(unit, parts[parts.count - 1]))
        }
        return cv2(concatenated(parts, axis: 3))
    }

    static func run(_ unit: Module, _ x: MLXArray) -> MLXArray {
        switch unit {
        case let block as NFKYOLOC3: return block(x)
        case let block as NFKYOLOWideBottleneck: return block(x)
        case let block as NFKYOLOPSABlock: return block(x)
        default: return x
        }
    }
}

/// YOLO26's attention `C3k2`: each repeated unit is a bottleneck followed by a PSA block, which the
/// reference holds in a Sequential. The unit list is an array of arrays, so the two keep the
/// reference's positional keys (`m.<unit>.0`, `m.<unit>.1`). A numeric `@ModuleInfo` key would be read
/// as an array index and abort the process, which is why the nesting is a real array.
final class NFKYOLOC3k2Attention: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "m") var m: [[Module]]
    private let hidden: Int

    init(inChannels: Int, outChannels: Int, repeats: Int, expansion: Float = 0.5, shortcut: Bool = true) {
        let half = Int(Float(outChannels) * expansion)
        hidden = half
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: half * 2)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: half * (2 + repeats), outChannels: outChannels)
        _m.wrappedValue = (0 ..< repeats).map { _ in
            [NFKYOLOWideBottleneck(inChannels: half, outChannels: half, shortcut: shortcut),
             NFKYOLOPSABlock(channels: half, heads: max(half / 64, 1))]
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let split = cv1(x)
        var parts = [split[0..., 0..., 0..., 0 ..< hidden],
                     split[0..., 0..., 0..., hidden ..< hidden * 2]]
        for unit in m {
            var out = parts[parts.count - 1]
            for step in unit {
                out = NFKYOLOC3k2.run(step, out)
            }
            parts.append(out)
        }
        return cv2(concatenated(parts, axis: 3))
    }
}

/// The reference `A2C2f` (YOLOv12): a 1×1 narrowing, `n` units that are each a pair of area-attention
/// blocks, a 1×1 fuse over every intermediate, and an optional learned residual gate.
final class NFKYOLOA2C2f: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "m") var m: [[NFKYOLOABlock]]
    @ParameterInfo(key: "gamma") var gamma: MLXArray?

    init(inChannels: Int, outChannels: Int, repeats: Int, areaAttention: Bool, area: Int,
         residual: Bool = false, mlpRatio: Float = 2.0, expansion: Float = 0.5) {
        let hidden = Int(Float(outChannels) * expansion)
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: hidden)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: (1 + repeats) * hidden, outChannels: outChannels)
        _m.wrappedValue = (0 ..< repeats).map { _ in
            (0 ..< 2).map { _ in
                NFKYOLOABlock(dimensions: hidden, heads: hidden / 32, mlpRatio: mlpRatio, area: area)
            }
        }
        // The reference builds `gamma` only when the stage is both attention-based and residual, and
        // its absence is what tells the forward to skip the gate.
        _gamma.wrappedValue = areaAttention && residual ? MLXArray.zeros([outChannels]) + 0.01 : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var parts = [cv1(x)]
        for unit in m {
            var out = parts[parts.count - 1]
            for block in unit {
                out = block(out)
            }
            parts.append(out)
        }
        let fused = cv2(concatenated(parts, axis: 3))
        guard let gamma else { return fused }
        return x + gamma.reshaped([1, 1, 1, gamma.shape[0]]) * fused
    }
}

/// The `A2C2f` form the reference builds when the stage asks for no area attention: the repeated unit
/// is a `C3k`, so it carries no inner Sequential and no residual gate.
final class NFKYOLOA2C2fC3k: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "m") var m: [NFKYOLOC3]

    init(inChannels: Int, outChannels: Int, repeats: Int, expansion: Float = 0.5, shortcut: Bool = true) {
        let hidden = Int(Float(outChannels) * expansion)
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: hidden)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: (1 + repeats) * hidden, outChannels: outChannels)
        _m.wrappedValue = (0 ..< repeats).map { _ in
            NFKYOLOC3(inChannels: hidden, outChannels: hidden, repeats: 2, shortcut: shortcut,
                      kernels: (3, 3))
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var parts = [cv1(x)]
        for unit in m {
            parts.append(unit(parts[parts.count - 1]))
        }
        return cv2(concatenated(parts, axis: 3))
    }
}

/// One YOLOv12 attention block: pre-residual area attention, then a pre-residual two-convolution MLP.
final class NFKYOLOABlock: Module {
    @ModuleInfo(key: "attn") var attn: NFKYOLOAreaAttention
    @ModuleInfo(key: "mlp") var mlp: [NFKYOLOConv]

    init(dimensions: Int, heads: Int, mlpRatio: Float, area: Int) {
        _attn.wrappedValue = NFKYOLOAreaAttention(dimensions: dimensions, heads: heads, area: area)
        let hidden = Int(Float(dimensions) * mlpRatio)
        _mlp.wrappedValue = [NFKYOLOConv(inChannels: dimensions, outChannels: hidden),
                             NFKYOLOConv(inChannels: hidden, outChannels: dimensions, activates: false)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attended = x + attn(x)
        return attended + mlp[1](mlp[0](attended))
    }
}

/// YOLOv12's area attention: the map is cut into `area` equal slices of the flattened positions and
/// attention runs inside each slice, which is what bounds the cost at high resolution.
final class NFKYOLOAreaAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: NFKYOLOConv
    @ModuleInfo(key: "proj") var proj: NFKYOLOConv
    @ModuleInfo(key: "pe") var pe: NFKYOLOConv

    private let heads: Int
    private let headDimensions: Int
    private let area: Int

    init(dimensions: Int, heads: Int, area: Int) {
        self.heads = heads
        self.area = max(area, 1)
        headDimensions = dimensions / heads
        let width = headDimensions * heads
        _qkv.wrappedValue = NFKYOLOConv(inChannels: dimensions, outChannels: width * 3, activates: false)
        _proj.wrappedValue = NFKYOLOConv(inChannels: width, outChannels: dimensions, activates: false)
        // The released YOLOv12 checkpoints carry a bias on this convolution beside its batch norm,
        // which the other generations' positional encodings do not.
        _pe.wrappedValue = NFKYOLOConv(inChannels: width, outChannels: width, kernel: 7,
                                       groups: width, activates: false, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (height, width, channels) = (x.shape[1], x.shape[2], x.shape[3])
        let positions = height * width
        var flattened = qkv(x).reshaped([1, positions, headDimensions * heads * 3])
        // The reference reshapes the batch axis by `area`, so each slice attends on its own.
        flattened = flattened.reshaped([area, positions / area, headDimensions * heads * 3])
        let slices = flattened.shape[0]
        let count = flattened.shape[1]

        let projected = flattened.reshaped([slices, count, heads, headDimensions * 3])
            .transposed(0, 2, 3, 1)
        let queries = projected[0..., 0..., 0 ..< headDimensions, 0...]
        let keys = projected[0..., 0..., headDimensions ..< (headDimensions * 2), 0...]
        let values = projected[0..., 0..., (headDimensions * 2)..., 0...]

        let scale = 1.0 / sqrtf(Float(headDimensions))
        let attention = softmax((queries * scale).transposed(0, 1, 3, 2).matmul(keys), axis: -1)
        var attended = values.matmul(attention.transposed(0, 1, 3, 2)).transposed(0, 3, 1, 2)
        var carried = values.transposed(0, 3, 1, 2)
        attended = attended.reshaped([1, positions, headDimensions * heads])
        carried = carried.reshaped([1, positions, headDimensions * heads])

        let map = attended.reshaped([1, height, width, channels])
        let encoded = carried.reshaped([1, height, width, channels])
        return proj(map + pe(encoded))
    }
}

/// The reference `SCDown` (YOLOv10): a 1×1 that changes the width, then a depthwise strided
/// convolution that changes the resolution.
final class NFKYOLOSCDown: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv

    init(inChannels: Int, outChannels: Int, kernel: Int, stride: Int) {
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: outChannels)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: outChannels, outChannels: outChannels, kernel: kernel,
                                        stride: stride, groups: outChannels, activates: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { cv2(cv1(x)) }
}

/// The reference `RepVGGDW`: a 7×7 and a 3×3 depthwise convolution summed under one activation. The
/// released checkpoints keep both branches, so the port runs them rather than the fused form.
final class NFKYOLORepVGGDW: Module {
    @ModuleInfo(key: "conv") var conv: NFKYOLOConv
    @ModuleInfo(key: "conv1") var conv1: NFKYOLOConv

    init(channels: Int) {
        _conv.wrappedValue = NFKYOLOConv(inChannels: channels, outChannels: channels, kernel: 7,
                                         groups: channels, activates: false)
        _conv1.wrappedValue = NFKYOLOConv(inChannels: channels, outChannels: channels, kernel: 3,
                                          groups: channels, activates: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { silu(conv(x) + conv1(x)) }
}

/// The reference `CIB` (YOLOv10): a five-slot Sequential of depthwise and pointwise convolutions,
/// whose middle slot is a `RepVGGDW` when the stage asks for the large kernel.
final class NFKYOLOCIB: Module {
    @ModuleInfo(key: "cv1") var cv1: [Module]
    private let usesResidual: Bool

    init(inChannels: Int, outChannels: Int, shortcut: Bool, largeKernel: Bool) {
        let hidden = outChannels
        let middle: Module = largeKernel
            ? NFKYOLORepVGGDW(channels: 2 * hidden)
            : NFKYOLOConv(inChannels: 2 * hidden, outChannels: 2 * hidden, kernel: 3, groups: 2 * hidden)
        _cv1.wrappedValue = [
            NFKYOLOConv(inChannels: inChannels, outChannels: inChannels, kernel: 3, groups: inChannels),
            NFKYOLOConv(inChannels: inChannels, outChannels: 2 * hidden),
            middle,
            NFKYOLOConv(inChannels: 2 * hidden, outChannels: outChannels),
            NFKYOLOConv(inChannels: outChannels, outChannels: outChannels, kernel: 3, groups: outChannels),
        ]
        usesResidual = shortcut && inChannels == outChannels
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for step in cv1 {
            switch step {
            case let conv as NFKYOLOConv: out = conv(out)
            case let rep as NFKYOLORepVGGDW: out = rep(out)
            default: break
            }
        }
        return usesResidual ? x + out : out
    }
}

/// The reference `C2fCIB` (YOLOv10): `C2f` whose repeated unit is a `CIB`.
final class NFKYOLOC2fCIB: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "m") var m: [NFKYOLOCIB]
    private let hidden: Int

    init(inChannels: Int, outChannels: Int, repeats: Int, shortcut: Bool, largeKernel: Bool,
         expansion: Float = 0.5) {
        let half = Int(Float(outChannels) * expansion)
        hidden = half
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: half * 2)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: half * (2 + repeats), outChannels: outChannels)
        _m.wrappedValue = (0 ..< repeats).map { _ in
            NFKYOLOCIB(inChannels: half, outChannels: half, shortcut: shortcut, largeKernel: largeKernel)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let split = cv1(x)
        var parts = [split[0..., 0..., 0..., 0 ..< hidden],
                     split[0..., 0..., 0..., hidden ..< hidden * 2]]
        for unit in m {
            parts.append(unit(parts[parts.count - 1]))
        }
        return cv2(concatenated(parts, axis: 3))
    }
}

/// The reference `RepConv`: a 3×3 and a 1×1 convolution summed under one activation. The released
/// YOLOv9 checkpoints keep both branches unfused.
final class NFKYOLORepConv: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKYOLOConv
    @ModuleInfo(key: "conv2") var conv2: NFKYOLOConv

    init(inChannels: Int, outChannels: Int, groups: Int = 1) {
        _conv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: outChannels, kernel: 3,
                                          groups: groups, activates: false)
        _conv2.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: outChannels, kernel: 1,
                                          groups: groups, activates: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { silu(conv1(x) + conv2(x)) }
}

/// The reference `RepBottleneck`: a bottleneck whose first convolution is a `RepConv`.
final class NFKYOLORepBottleneck: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLORepConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    private let usesResidual: Bool

    init(inChannels: Int, outChannels: Int, shortcut: Bool) {
        _cv1.wrappedValue = NFKYOLORepConv(inChannels: inChannels, outChannels: outChannels)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: outChannels, outChannels: outChannels, kernel: 3)
        usesResidual = shortcut && inChannels == outChannels
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = cv2(cv1(x))
        return usesResidual ? x + out : out
    }
}

/// The reference `RepCSP`: `C3` whose repeated unit is a `RepBottleneck`.
final class NFKYOLORepCSP: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "cv3") var cv3: NFKYOLOConv
    @ModuleInfo(key: "m") var m: [NFKYOLORepBottleneck]

    init(inChannels: Int, outChannels: Int, repeats: Int, shortcut: Bool = true, expansion: Float = 0.5) {
        let hidden = Int(Float(outChannels) * expansion)
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: hidden)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: hidden)
        _cv3.wrappedValue = NFKYOLOConv(inChannels: hidden * 2, outChannels: outChannels)
        _m.wrappedValue = (0 ..< repeats).map { _ in
            NFKYOLORepBottleneck(inChannels: hidden, outChannels: hidden, shortcut: shortcut)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var deep = cv1(x)
        for unit in m {
            deep = unit(deep)
        }
        return cv3(concatenated([deep, cv2(x)], axis: 3))
    }
}

/// The reference `RepNCSPELAN4` (YOLOv9's GELAN stage): a 1×1 split in two, two chained branches that
/// each run a `RepCSP` then a 3×3, and a 1×1 fuse over all four tensors. `ELAN1` is the same shape with
/// plain convolutions in place of the two branches, which is why one class covers both.
final class NFKYOLORepNCSPELAN4: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: [Module]
    @ModuleInfo(key: "cv3") var cv3: [Module]
    @ModuleInfo(key: "cv4") var cv4: NFKYOLOConv
    private let hidden: Int

    /// `simple` builds the `ELAN1` form, whose branches are one 3×3 convolution rather than a
    /// `RepCSP` and a 3×3.
    init(inChannels: Int, outChannels: Int, midChannels: Int, branchChannels: Int, repeats: Int,
         simple: Bool = false) {
        hidden = midChannels / 2
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: midChannels)
        if simple {
            _cv2.wrappedValue = [NFKYOLOConv(inChannels: midChannels / 2, outChannels: branchChannels, kernel: 3)]
            _cv3.wrappedValue = [NFKYOLOConv(inChannels: branchChannels, outChannels: branchChannels, kernel: 3)]
        } else {
            _cv2.wrappedValue = [
                NFKYOLORepCSP(inChannels: midChannels / 2, outChannels: branchChannels, repeats: repeats),
                NFKYOLOConv(inChannels: branchChannels, outChannels: branchChannels, kernel: 3),
            ]
            _cv3.wrappedValue = [
                NFKYOLORepCSP(inChannels: branchChannels, outChannels: branchChannels, repeats: repeats),
                NFKYOLOConv(inChannels: branchChannels, outChannels: branchChannels, kernel: 3),
            ]
        }
        _cv4.wrappedValue = NFKYOLOConv(inChannels: midChannels + 2 * branchChannels, outChannels: outChannels)
    }

    private func run(_ branch: [Module], _ x: MLXArray) -> MLXArray {
        var out = x
        for step in branch {
            switch step {
            case let csp as NFKYOLORepCSP: out = csp(out)
            case let conv as NFKYOLOConv: out = conv(out)
            default: break
            }
        }
        return out
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let split = cv1(x)
        var parts = [split[0..., 0..., 0..., 0 ..< hidden],
                     split[0..., 0..., 0..., hidden ..< hidden * 2]]
        parts.append(run(cv2, parts[parts.count - 1]))
        parts.append(run(cv3, parts[parts.count - 1]))
        return cv4(concatenated(parts, axis: 3))
    }
}

/// The reference `ELAN1`: `RepNCSPELAN4`'s shape with a single 3×3 convolution in each branch rather
/// than a `RepCSP` and a 3×3, so the two branches are plain modules rather than Sequentials.
final class NFKYOLOELAN1: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "cv3") var cv3: NFKYOLOConv
    @ModuleInfo(key: "cv4") var cv4: NFKYOLOConv
    private let hidden: Int

    init(inChannels: Int, outChannels: Int, midChannels: Int, branchChannels: Int) {
        hidden = midChannels / 2
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: midChannels)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: midChannels / 2, outChannels: branchChannels, kernel: 3)
        _cv3.wrappedValue = NFKYOLOConv(inChannels: branchChannels, outChannels: branchChannels, kernel: 3)
        _cv4.wrappedValue = NFKYOLOConv(inChannels: midChannels + 2 * branchChannels, outChannels: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let split = cv1(x)
        var parts = [split[0..., 0..., 0..., 0 ..< hidden],
                     split[0..., 0..., 0..., hidden ..< hidden * 2]]
        parts.append(cv2(parts[parts.count - 1]))
        parts.append(cv3(parts[parts.count - 1]))
        return cv4(concatenated(parts, axis: 3))
    }
}

/// The reference `SPPELAN`: a 1×1 narrowing and three chained 5×5 stride-1 max pools, fused by a 1×1.
/// The pools are parameter-free, so only the two convolutions carry weights.
final class NFKYOLOSPPELAN: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv5") var cv5: NFKYOLOConv
    private let kernel: Int

    init(inChannels: Int, outChannels: Int, midChannels: Int, kernel: Int = 5) {
        self.kernel = kernel
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: midChannels)
        _cv5.wrappedValue = NFKYOLOConv(inChannels: 4 * midChannels, outChannels: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var parts = [cv1(x)]
        for _ in 0 ..< 3 {
            parts.append(NFKMLXResample.maxPooled(parts[parts.count - 1], kernel: IntOrPair(kernel),
                                                  stride: 1, padding: IntOrPair(kernel / 2)))
        }
        return cv5(concatenated(parts, axis: 3))
    }
}

/// The reference `AConv` (YOLOv9): a 2×2 average pool at stride 1 that keeps the borders, then a
/// strided 3×3.
final class NFKYOLOAConv: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv

    init(inChannels: Int, outChannels: Int) {
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: outChannels, kernel: 3, stride: 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        cv1(NFKYOLOBlockOps.averagePooledCountIncludingPad(x))
    }
}

/// The reference `ADown` (YOLOv9): the same average pool, then the halves take different routes — a
/// strided 3×3 on one and a strided max pool with a 1×1 on the other.
final class NFKYOLOADown: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv

    init(inChannels: Int, outChannels: Int) {
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels / 2, outChannels: outChannels / 2,
                                        kernel: 3, stride: 2)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: inChannels / 2, outChannels: outChannels / 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pooled = NFKYOLOBlockOps.averagePooledCountIncludingPad(x)
        let half = pooled.shape[3] / 2
        let first = cv1(pooled[0..., 0..., 0..., 0 ..< half])
        let second = cv2(NFKMLXResample.maxPooled(pooled[0..., 0..., 0..., half...],
                                                  kernel: IntOrPair(3), stride: IntOrPair(2),
                                                  padding: IntOrPair(1)))
        return concatenated([first, second], axis: 3)
    }
}

/// The reference `CBLinear` (YOLOv9e): one 1×1 convolution with a bias whose output is split into the
/// widths the graph states, feeding the programmable-gradient branch.
final class NFKYOLOCBLinear: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    let widths: [Int]

    init(inChannels: Int, widths: [Int]) {
        self.widths = widths
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: widths.reduce(0, +),
                                    kernelSize: 1, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let projected = conv(x)
        var parts = [MLXArray]()
        var offset = 0
        for width in widths {
            parts.append(projected[0..., 0..., 0..., offset ..< (offset + width)])
            offset += width
        }
        return parts
    }
}

/// Operations the GELAN stages need that have no `MLXNN` equivalent.
enum NFKYOLOBlockOps {

    /// `F.avg_pool2d(x, 2, 1, 0, ceil_mode=False, count_include_pad=True)`: a 2×2 window at stride 1
    /// with no padding, which shrinks each side by one.
    static func averagePooledCountIncludingPad(_ x: MLXArray) -> MLXArray {
        let (height, width) = (x.shape[1], x.shape[2])
        let a = x[0..., 0 ..< (height - 1), 0 ..< (width - 1), 0...]
        let b = x[0..., 1 ..< height, 0 ..< (width - 1), 0...]
        let c = x[0..., 0 ..< (height - 1), 1 ..< width, 0...]
        let d = x[0..., 1 ..< height, 1 ..< width, 0...]
        return (a + b + c + d) / 4
    }
}
