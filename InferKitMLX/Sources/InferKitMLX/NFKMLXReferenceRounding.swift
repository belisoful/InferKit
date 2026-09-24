//
//  NFKMLXReferenceRounding.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXFast
import MLXNN

// The arithmetic of a PyTorch reference that runs in half precision, with its roundings where the
// reference places them. In float32 the placement of a rounding is invisible, so MLX's fused kernels
// (which widen internally and round once) and a chain of elementwise ops agree to the last bit with the
// reference. In bf16 they do not: a fused RMS norm rounds once where transformers' Gemma norm also
// rounds once but a bf16 elementwise chain rounds at every op, and the fused attention keeps its
// softmax in float32 where eager torch rounds the scores and the probabilities to bf16. Each difference
// moves roundings on most elements of a hidden state.
//
// Every function here takes the fused or plain path when its input is float32, so a float32 forward is
// byte-identical to the code it replaces, and the reference's own placement otherwise.

enum NFKReferenceRounding {

    /// Whether `x` is a reduced-precision tensor the reference's rounding placement applies to.
    static func isReduced(_ x: MLXArray) -> Bool { x.dtype == .bfloat16 || x.dtype == .float16 }

    /// `x · factor` as torch multiplies a tensor by a Python float: the factor stays float32 and the
    /// product rounds once. MLX would round a `Float` literal to `x`'s type first, which changes the
    /// product whenever the factor is not exact in bf16 (`1536^-0.5`, `2^-0.5`, `144^-0.5`).
    static func scaled(_ x: MLXArray, by factor: Float) -> MLXArray {
        guard isReduced(x) else { return x * factor }
        return (x.asType(.float32) * factor).asType(x.dtype)
    }

    /// `x / divisor` as torch divides a tensor by a Python float: in float32, rounded once.
    static func divided(_ x: MLXArray, by divisor: Float) -> MLXArray {
        guard isReduced(x) else { return x / divisor }
        return (x.asType(.float32) / divisor).asType(x.dtype)
    }

    /// transformers' Gemma RMS norm: `x · rsqrt(mean(x²) + eps) · (1 + w)` in float32 throughout,
    /// rounded once to `x`'s type.
    static func gemmaNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
        guard isReduced(x) else {
            return x * rsqrt((x * x).mean(axis: -1, keepDims: true) + eps) * (1 + weight)
        }
        let wide = x.asType(.float32)
        let normalized = wide * rsqrt((wide * wide).mean(axis: -1, keepDims: true) + eps)
        return (normalized * (1 + weight.asType(.float32))).asType(x.dtype)
    }

    /// transformers' Gemma 3n / Gemma 4 RMS norm: `x · (mean(x²) + eps)^-0.5 · w` (no weight when
    /// `weight` is nil) in float32 throughout, rounded once to `x`'s type. The reference raises to the
    /// power `-0.5` rather than taking `rsqrt`, which differs in the last float32 bit.
    static func scaledNorm(_ x: MLXArray, weight: MLXArray?, eps: Float) -> MLXArray {
        guard isReduced(x) else {
            let normalized = x * rsqrt((x * x).mean(axis: -1, keepDims: true) + eps)
            return weight.map { normalized * $0 } ?? normalized
        }
        let wide = x.asType(.float32)
        let normalized = wide * pow((wide * wide).mean(axis: -1, keepDims: true) + eps, Float(-0.5))
        return (weight.map { normalized * $0.asType(.float32) } ?? normalized).asType(x.dtype)
    }

    /// torch's `gelu(approximate="tanh")`, which widens a half-precision input and rounds its result once.
    static func geluTanh(_ x: MLXArray) -> MLXArray {
        guard isReduced(x) else { return geluApproximate(x) }
        return geluApproximate(x.asType(.float32)).asType(x.dtype)
    }

    /// torch's `silu`, which widens a half-precision input and rounds `x · sigmoid(x)` once; MLX's
    /// composed `silu` rounds the sigmoid and then the product.
    static func silu(_ x: MLXArray) -> MLXArray {
        guard isReduced(x) else { return MLXNN.silu(x) }
        let wide = x.asType(.float32)
        return (wide * sigmoid(wide)).asType(x.dtype)
    }

    /// A reduction mean as torch takes it over a half-precision tensor: accumulated in float32, rounded
    /// once. MLX's half-precision reductions do not accumulate the same way.
    static func mean(_ x: MLXArray, axis: Int, keepDims: Bool = true) -> MLXArray {
        guard isReduced(x) else { return x.mean(axis: axis, keepDims: keepDims) }
        return x.asType(.float32).mean(axis: axis, keepDims: keepDims).asType(x.dtype)
    }

    /// An elementwise function applied as torch applies it to a half-precision input: widened,
    /// computed, rounded once. MLX's half-precision transcendentals do not round the same way.
    static func wide(_ x: MLXArray, _ function: (MLXArray) -> MLXArray) -> MLXArray {
        guard isReduced(x) else { return function(x) }
        return function(x.asType(.float32)).asType(x.dtype)
    }

    /// torch's `rsqrt` on a half-precision input, which is not correctly rounded: it rounds the square
    /// root to the input's type and then rounds its reciprocal, `1 / round(sqrt(x))`.
    static func rsqrt(_ x: MLXArray) -> MLXArray {
        guard isReduced(x) else { return MLX.rsqrt(x) }
        let root = MLX.sqrt(x.asType(.float32)).asType(x.dtype).asType(.float32)
        return (1 / root).asType(x.dtype)
    }

    /// torch's `sigmoid` on a half-precision input: computed wide, rounded once.
    static func sigmoid(_ x: MLXArray) -> MLXArray {
        guard isReduced(x) else { return MLX.sigmoid(x) }
        return MLX.sigmoid(x.asType(.float32)).asType(x.dtype)
    }

    /// The floating type a module's parameters carry. A diffusers pipeline casts its latents and its
    /// conditioning to the transformer's type, so a pipeline reads it here.
    static func parameterType(of module: Module) -> DType {
        module.parameters().flattened().first { $0.1.dtype.isFloatingPoint }?.1.dtype ?? .float32
    }

    /// A flow-matching timestep on the schedule's thousand scale as the fraction diffusers passes the
    /// transformer: the timestep in the latents' type, divided by 1000 there. In float32 it is the
    /// plain quotient.
    static func flowFraction(_ timestep: Float, dtype: DType) -> MLXArray {
        guard dtype == .bfloat16 || dtype == .float16 else { return MLXArray([timestep / 1000]) }
        return MLXArray([timestep]).asType(dtype) / 1000
    }

    /// One Euler step of the flow, `sample + dt · velocity`, as diffusers' flow-matching scheduler takes
    /// it on a half-precision velocity: the product rounds to the velocity's type, the sum forms in
    /// float32, and the result rounds back to the velocity's type.
    static func eulerStep(_ sample: MLXArray, velocity: MLXArray, dt: Float) -> MLXArray {
        guard isReduced(velocity) else { return sample + dt * velocity }
        let increment = scaled(velocity, by: dt)
        return (sample.asType(.float32) + increment.asType(.float32)).asType(velocity.dtype)
    }

    /// A Swish module written as `x * torch.sigmoid(x)`, which in half precision rounds the sigmoid and
    /// then the product; torch's fused `silu` rounds once.
    static func swish(_ x: MLXArray) -> MLXArray {
        guard isReduced(x) else { return MLXNN.silu(x) }
        return x * sigmoid(x)
    }

    /// torch's `softplus` on a half-precision input: computed wide, rounded once.
    static func softplus(_ x: MLXArray) -> MLXArray {
        guard isReduced(x) else { return MLXNN.softplus(x) }
        return MLXNN.softplus(x.asType(.float32)).asType(x.dtype)
    }

    /// torch's MATH `scaled_dot_product_attention` on half-precision operands: widened to float32, the
    /// scale split as its square root on the queries and on the keys, the additive `mask` (already in
    /// the operands' type) added, and only the output rounded. A float32 input takes the explicit
    /// float32 form it names.
    static func mathAttention(queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
                              mask: MLXArray?) -> MLXArray {
        guard isReduced(queries) else {
            var scores = matmul(queries, keys.swappedAxes(-1, -2)) * scale
            if let mask { scores = scores + mask }
            return matmul(softmax(scores, axis: -1, precise: true), values)
        }
        let root = scale.squareRoot()
        var scores = matmul(queries.asType(.float32) * root, (keys.asType(.float32) * root).swappedAxes(-1, -2))
        if let mask { scores = scores + mask.asType(.float32) }
        return matmul(softmax(scores, axis: -1), values.asType(.float32)).asType(queries.dtype)
    }

    /// torch's CPU flash `scaled_dot_product_attention` on half-precision operands over one key block
    /// (up to 512 keys): the scores formed and scaled in float32, the mask added, the row max subtracted
    /// and exponentiated in float32, the exponentials rounded to the operands' type for the product with
    /// the values while their float32 sum normalizes, and the output rounded once. A float32 input takes
    /// the fused kernel.
    static func flashAttention(queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
                               mask: MLXArray?) -> MLXArray {
        guard isReduced(queries) else {
            return MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values,
                                                     scale: scale, mask: mask)
        }
        var scores = matmul(queries.asType(.float32), keys.asType(.float32).swappedAxes(-1, -2)) * scale
        if let mask { scores = scores + mask.asType(.float32) }
        let exponentials = exp(scores - scores.max(axis: -1, keepDims: true))
        let weighted = matmul(exponentials.asType(queries.dtype).asType(.float32), values.asType(.float32))
        return (weighted / exponentials.sum(axis: -1, keepDims: true)).asType(queries.dtype)
    }

    /// A top-k router as transformers runs it: the softmax over every expert in float32, the kept
    /// weights renormalized in float32 when `normalize`, then rounded to the logits' type. Returns the
    /// weights and the chosen expert indices, `[..., active]` each.
    static func routed(_ logits: MLXArray, active: Int, normalize: Bool) -> (weights: MLXArray, chosen: MLXArray) {
        let scores = softmax(logits.asType(.float32), axis: -1, precise: true)
        let chosen = argPartition(-scores, kth: active - 1, axis: -1)[.ellipsis, 0 ..< active]
        var weights = takeAlong(scores, chosen, axis: -1)
        if normalize {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (weights.asType(logits.dtype), chosen)
    }

    /// The routed experts' outputs `[..., active, hidden]` combined by their weights `[..., active]`.
    /// In half precision it is transformers' loop: each expert's output times its weight, added into a
    /// zeroed buffer in ascending expert order, every product and every add rounding.
    static func combined(_ outputs: MLXArray, weights: MLXArray, chosen: MLXArray) -> MLXArray {
        guard isReduced(outputs) else { return (outputs * weights.expandedDimensions(axis: -1)).sum(axis: -2) }
        // Each output times its weight rounds once, in whatever type the weights carry.
        let scaled = (outputs.asType(.float32) * weights.expandedDimensions(axis: -1).asType(.float32))
            .asType(outputs.dtype)
        let order = argSort(chosen, axis: -1)
        let ordered = takeAlong(scaled, order.expandedDimensions(axis: -1), axis: -2)
        var total = ordered[.ellipsis, 0, 0...]
        for slot in 1 ..< ordered.dim(-2) {
            total = total + ordered[.ellipsis, slot, 0...]
        }
        return total
    }

    /// The rotate-half rotary over `[batch, heads, length, headDim]`, as transformers applies it: the
    /// cosine and sine tables are built in float32 and rounded to the input's type, and
    /// `x · cos + rotate_half(x) · sin` then rounds at each op. `scale` multiplies the positions (a
    /// linearly scaled rotary is `1 / factor`).
    static func rotary(_ x: MLXArray, dimensions: Int, base: Float, scale: Float = 1, offset: Int) -> MLXArray {
        guard isReduced(x) else {
            return MLXFast.RoPE(x, dimensions: dimensions, traditional: false, base: base, scale: scale, offset: offset)
        }
        let exponents = MLXArray(stride(from: 0, to: dimensions, by: 2).map { Float($0) / Float(dimensions) })
        return rotary(x, inverseFrequencies: 1 / pow(MLXArray(base), exponents) * scale, offset: offset)
    }

    /// The rotate-half rotary at explicit float32 inverse frequencies `[dimensions / 2]` over the
    /// leading `dimensions` channels, the tables rounded to `x`'s type as transformers rounds them. A
    /// zero frequency is the identity rotation, which is how a proportional rotary leaves a pair unturned.
    /// `factor` is the attention scaling a YaRN or LongRoPE rotary multiplies the float32 tables by.
    static func rotary(_ x: MLXArray, inverseFrequencies inverse: MLXArray, offset: Int,
                       factor: Float = 1) -> MLXArray {
        let dimensions = inverse.dim(0) * 2
        let length = x.dim(-2)
        let positions = MLXArray((offset ..< offset + length).map(Float.init))
        let angles = positions.expandedDimensions(axis: 1) * inverse.expandedDimensions(axis: 0)
        let table = concatenated([angles, angles], axis: -1)
        let cos = (MLX.cos(table) * factor).asType(x.dtype), sin = (MLX.sin(table) * factor).asType(x.dtype)
        let rotated = x[.ellipsis, 0 ..< dimensions]
        let half = dimensions / 2
        let turned = concatenated([-rotated[.ellipsis, half...], rotated[.ellipsis, 0 ..< half]], axis: -1)
        let result = rotated * cos + turned * sin
        guard dimensions < x.dim(-1) else { return result }
        return concatenated([result, x[.ellipsis, dimensions...]], axis: -1)
    }

    /// Scaled dot-product attention over `[batch, heads, length, headDim]` queries and grouped keys and
    /// values. In half precision it is transformers' eager path: the scores round to the input's type,
    /// scale and mask round again, the softmax runs in float32 and rounds, and the weighted sum rounds.
    static func attention(queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
                          mask: MLXArray?, softcap: Float = 0) -> MLXArray {
        guard isReduced(queries) || softcap > 0 else {
            return MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values,
                                                     scale: scale, mask: mask)
        }
        let groups = queries.dim(1) / keys.dim(1)
        func spread(_ x: MLXArray) -> MLXArray {
            guard groups > 1 else { return x }
            let (batch, heads, count, width) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            return broadcast(x.expandedDimensions(axis: 2), to: [batch, heads, groups, count, width])
                .reshaped([batch, heads * groups, count, width])
        }
        var scores = scaled(matmul(queries, spread(keys).transposed(0, 1, 3, 2)), by: scale)
        if softcap > 0 {
            let cap = MLXArray(softcap).asType(queries.dtype)
            scores = wide(scores / cap) { tanh($0) } * cap
        }
        if let mask { scores = scores + mask.asType(scores.dtype) }
        return matmul(softmax(scores, axis: -1, precise: true), spread(values))
    }
}

/// MLXNN's `LayerNorm`, rounding as torch's `layer_norm` does in half precision: normalized and
/// scaled in float32, rounded once. MLX's fused kernel rounds a half-precision input differently. A
/// float32 input takes the fused kernel unchanged.
final class NFKLayerNorm: LayerNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard NFKReferenceRounding.isReduced(x) else { return super.callAsFunction(x) }
        return MLXFast.layerNorm(x.asType(.float32), weight: weight?.asType(.float32),
                                 bias: bias?.asType(.float32), eps: eps).asType(x.dtype)
    }
}

/// MLXNN's `GroupNorm`, computed in float32 and rounded once on a half-precision input, as torch's
/// `group_norm` is.
final class NFKGroupNorm: GroupNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard NFKReferenceRounding.isReduced(x) else { return super.callAsFunction(x) }
        return super.callAsFunction(x.asType(.float32)).asType(x.dtype)
    }
}

/// MLXNN's `BatchNorm` in inference, rounding as torch's `batch_norm` does in half precision: the
/// per-channel scale `weight / sqrt(running_var + eps)` and shift `bias - running_mean · scale` are
/// formed in float32 from the stored statistics, applied in float32, and rounded once. MLXNN adds `eps`
/// to the half-precision variance and takes its `rsqrt` in that precision, which widening the input
/// alone does not reach. Training and a float32 input take MLXNN's path unchanged.
final class NFKBatchNorm: BatchNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let statistics = parameters()
        guard NFKReferenceRounding.isReduced(x), !training,
              case .value(let mean)? = statistics["running_mean"],
              case .value(let variance)? = statistics["running_var"]
        else { return super.callAsFunction(x) }
        let scale = (weight?.asType(.float32) ?? MLXArray(Float(1))) / sqrt(variance.asType(.float32) + eps)
        let shift = (bias?.asType(.float32) ?? MLXArray(Float(0))) - mean.asType(.float32) * scale
        return (x.asType(.float32) * scale + shift).asType(x.dtype)
    }
}

/// MLXNN's `Conv1d`, which on a half-precision input rounds the convolution and then the bias; torch's
/// `conv1d` accumulates both in float32 and rounds once.
final class NFKConv1d: Conv1d {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard NFKReferenceRounding.isReduced(x) else { return super.callAsFunction(x) }
        var y = conv1d(x.asType(.float32), weight.asType(.float32), stride: stride, padding: padding,
                       dilation: dilation, groups: groups)
        if let bias { y = y + bias.asType(.float32) }
        return y.asType(x.dtype)
    }
}

/// MLXNN's `Conv2d`, rounded once on a half-precision input as torch's `conv2d` is.
final class NFKConv2d: Conv2d {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard NFKReferenceRounding.isReduced(x) else { return super.callAsFunction(x) }
        var y = conv2d(x.asType(.float32), weight.asType(.float32), stride: .init(stride), padding: .init(padding),
                       dilation: .init(dilation), groups: groups)
        if let bias { y = y + bias.asType(.float32) }
        return y.asType(x.dtype)
    }
}
