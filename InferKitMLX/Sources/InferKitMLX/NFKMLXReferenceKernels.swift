//
//  NFKMLXReferenceKernels.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXFast

// PyTorch's CPU arithmetic for the operations ``NFKMLXArithmetic/reference`` selects, as Metal kernels that
// match torch bit for bit: each computes its operation in torch's order, with a fused multiply-add where
// torch's compiled code fuses one and a separate rounding where it does not. A row-wise kernel gives a row a
// threadgroup of 32 threads: 4 carry torch's 4 vector lanes, which stay independent until torch joins them,
// and all 32 write the outputs. A Metal kernel cannot dispatch on the CPU stream, so the CPU stream takes MLX
// operations. A kernel's gradient is the gradient of the MLX operations it stands for.

extension NFKReferenceRounding {

    /// The float32 exponentials of a half-precision attention's softmax, as ``NFKMLXHalfPrecision/arithmetic``
    /// selects: torch's (``torchExp(_:)``) or MLX's `exp`.
    static func softmaxExponential(_ x: MLXArray) -> MLXArray {
        NFKMLXHalfPrecision.arithmetic == .reference ? torchExp(x) : exp(x)
    }

    /// The softmax over the last axis of `scores`, computed in float32 and returned in `scores`' type, as
    /// ``NFKMLXHalfPrecision/arithmetic`` selects: torch's (``torchSoftmax(_:)``) or MLX's.
    static func softmaxProbabilities(_ scores: MLXArray) -> MLXArray {
        guard NFKMLXHalfPrecision.arithmetic == .reference else { return softmax(scores, axis: -1, precise: true) }
        return torchSoftmax(scores.asType(.float32)).asType(scores.dtype)
    }

    /// The softmax over the last axis of half-precision `scores` that the reference takes without widening them, as
    /// ``NFKMLXHalfPrecision/arithmetic`` selects: torch's reduced-type kernel (``torchHalfSoftmax(_:)``) or `fast`.
    /// A float32 `scores` takes `fast`.
    static func halfSoftmax(_ scores: MLXArray, fast: @autoclosure () -> MLXArray) -> MLXArray {
        guard isReduced(scores), NFKMLXHalfPrecision.arithmetic == .reference else { return fast() }
        return torchHalfSoftmax(scores)
    }

    /// The float32 sum over the last axis of a router's kept weights, kept as one value per row, as
    /// ``NFKMLXHalfPrecision/arithmetic`` selects: torch's order (``torchRowSum(_:)``) or MLX's sum.
    static func routingSum(_ weights: MLXArray) -> MLXArray {
        NFKMLXHalfPrecision.arithmetic == .reference ? torchRowSum(weights) : weights.sum(axis: -1, keepDims: true)
    }

    /// torch's CPU float32 `sum` over the last axis of a row of up to 19 values, kept as one value per row:
    /// whole 4-lane vectors added lane by lane, the values past them summed in order and added into the first
    /// lane, then the lanes summed in order. A longer row takes MLX's sum.
    static func torchRowSum(_ x: MLXArray) -> MLXArray {
        let width = x.dim(-1)
        guard width <= 19 else { return x.sum(axis: -1, keepDims: true) }
        func column(_ index: Int) -> MLXArray { x[.ellipsis, index ..< index + 1] }
        func ordered(_ range: Range<Int>) -> MLXArray { range.dropFirst().reduce(column(range.lowerBound)) { $0 + column($1) } }
        guard width >= 4 else { return ordered(0 ..< width) }
        var lanes = x[.ellipsis, 0 ..< 4]
        var start = 4
        while start + 4 <= width {
            lanes = lanes + x[.ellipsis, start ..< start + 4]
            start += 4
        }
        var first = lanes[.ellipsis, 0 ..< 1]
        if start < width { first = first + ordered(start ..< width) }
        return ((first + lanes[.ellipsis, 1 ..< 2]) + lanes[.ellipsis, 2 ..< 3]) + lanes[.ellipsis, 3 ..< 4]
    }

    /// torch's `layer_norm` over the last axis of a half-precision or float32 `x`, rounded to `x`'s type, as
    /// ``NFKMLXHalfPrecision/arithmetic`` selects: torch's arithmetic (``torchLayerNorm(_:weight:bias:eps:)``)
    /// or MLX's fused kernel in float32, rounded once.
    static func layerNorm(_ x: MLXArray, weight: MLXArray?, bias: MLXArray?, eps: Float) -> MLXArray {
        if NFKMLXHalfPrecision.arithmetic == .reference {
            return torchLayerNorm(x, weight: weight, bias: bias, eps: eps)
        }
        return MLXFast.layerNorm(x.asType(.float32), weight: weight?.asType(.float32),
                                 bias: bias?.asType(.float32), eps: eps).asType(x.dtype)
    }

    /// Sleef's `expf` (`Sleef_expf4_u10`), the float32 exponential of torch's vectorized CPU kernels, of `x`
    /// widened to float32. `x` is reduced by `n = rint(x / ln 2)` in two parts, a polynomial in the remainder
    /// forms the mantissa with fused multiply-adds, and `2ⁿ` applies in two halves. An argument below −104
    /// gives 0 and one above 88.72 gives infinity. On the CPU stream MLX operations round each multiply-add,
    /// and 1.1% of results sit one unit from torch's.
    static func torchExp(_ x: MLXArray) -> MLXArray {
        let wide = x.asType(.float32)
        guard NFKMLXDevice.currentType == .gpu, wide.size > 0 else { return torchExpOperations(wide) }
        return torchExpFunction([wide])[0]
    }

    /// `exp(x)` in `x`'s type, as ``NFKMLXHalfPrecision/arithmetic`` selects: torch's vectorized exponential
    /// (``torchExp(_:)``, rounded once to `x`'s type) or MLX's `exp`.
    static func exponential(_ x: MLXArray) -> MLXArray {
        NFKMLXHalfPrecision.arithmetic == .reference ? torchExp(x).asType(x.dtype) : MLX.exp(x)
    }

    /// torch's exact `gelu` of a bfloat16 or float16 `x`, `x · 0.5 · (1 + erf(x · √½))` formed in float32 and
    /// rounded once, with the error function of torch's vectorized kernels: the Abramowitz–Stegun 7.1.26
    /// polynomial with its multiply-adds fused and Sleef's exponential. On the CPU stream it takes MLX
    /// operations, which round each multiply-add.
    static func torchGELU(_ x: MLXArray) -> MLXArray {
        guard NFKMLXDevice.currentType == .gpu, x.size > 0 else { return geluOperations([x])[0] }
        return torchGELUFunction([x])[0]
    }

    /// torch's `silu` of a bfloat16 or float16 `x`, `x / (1 + exp(−x))` formed in float32 with Sleef's
    /// exponential and rounded once, in one kernel that keeps no float32 copy of `x`. On the CPU stream it
    /// takes MLX operations.
    static func torchSiLU(_ x: MLXArray) -> MLXArray {
        guard NFKMLXDevice.currentType == .gpu, x.size > 0 else { return siluOperations([x])[0] }
        return torchSiLUFunction([x])[0]
    }

    /// torch's CPU float32 `softmax` over the last axis of a float32 `x`: the row's maximum, each
    /// `exp(x − maximum)` by Sleef's `expf`, their sum over 4 vector lanes (each lane's elements in order, the
    /// lanes joined as `(a₀ + a₂) + (a₁ + a₃)`, a row shorter than 4 in order), and each exponential times
    /// `1 / sum`. On the CPU stream it takes MLX's softmax.
    static func torchSoftmax(_ x: MLXArray) -> MLXArray {
        guard NFKMLXDevice.currentType == .gpu, x.size > 0 else { return softmax(x, axis: -1, precise: true) }
        return torchSoftmaxFunction([x])[0]
    }

    /// torch's CPU `softmax` over the last axis of a bfloat16 or float16 `x`, in `x`'s type: the row widened to
    /// float32, its maximum, Sleef's `expf` of each `x − maximum`, the exponentials of whole 4-value vectors summed
    /// over 4 lanes and the lanes joined as `(a₀ + a₂) + (a₁ + a₃)`, the remaining exponentials added in order, and
    /// each exponential times `1 / sum`, rounded once. torch takes the remaining exponentials from the C library's
    /// `expf`, which Sleef's stands in for: 0 to 3 outputs in a million round apart. On the CPU stream it takes
    /// MLX's softmax in float32, rounded once.
    static func torchHalfSoftmax(_ x: MLXArray) -> MLXArray {
        guard NFKMLXDevice.currentType == .gpu, x.size > 0 else {
            return softmax(x.asType(.float32), axis: -1, precise: true).asType(x.dtype)
        }
        return torchHalfSoftmaxFunction([x])[0]
    }

    /// torch's `softplus` of `x` with its default β of 1 and threshold of 20, in `x`'s type: `x` above 20, and
    /// otherwise `log1p(exp(x))` with Sleef's `expf` and `log1pf`, formed in float32 and rounded once. A result
    /// below the least normal float32 is 0, where torch keeps the subnormal. On the CPU stream it takes MLX
    /// operations.
    static func torchSoftplus(_ x: MLXArray) -> MLXArray {
        guard NFKMLXDevice.currentType == .gpu, x.size > 0 else { return softplusOperations([x])[0] }
        return torchSoftplusFunction([x])[0]
    }

    /// `(values · softmax(scores)).sum()` over the second-to-last axis of float32 `values` and `scores`, as torch
    /// computes it along a non-trailing axis: per column of the trailing axis, the maximum, Sleef's `expf` of each
    /// `score − maximum`, their sum in order, each exponential divided by the sum, each product with its value,
    /// and the products summed by torch's cascade (blocks of up to 16 rows summed in order and the block sums
    /// joined in order). The trailing axis is a whole number of 4-value vectors, which torch vectorizes. On the
    /// CPU stream it takes MLX operations.
    static func torchWindowPool(values: MLXArray, scores: MLXArray) -> MLXArray {
        guard NFKMLXDevice.currentType == .gpu, values.size > 0 else { return windowPoolOperations([values, scores])[0] }
        return torchWindowPoolFunction([values, scores])[0]
    }

    /// The probabilities of torch's float32 `softmax` over one query frame's row of a packed key buffer that holds
    /// only part of its keys, as eager flex attention computes them over Wan Animate's generation buffer.
    /// `generationScores` `[rows, frames · area]` are the chunk's keys, which sit at `f · videoArea + i` of the
    /// buffer's first `generationLimit` positions; `referenceScores` `[rows, referenceCount]` sit from
    /// `referenceStart` within the `referenceLength` positions the query's reference slot spans. Every other
    /// position of those spans is a zero key that scores 0, and the rest of the buffer is masked. Each row's
    /// exponentials sum over 4 lanes by their position in the buffer, as ``torchSoftmax(_:)`` sums a row.
    /// The result is `[rows, frames · area + referenceCount]`. On the CPU stream it takes MLX operations, which sum
    /// in MLX's order.
    static func torchPackedSoftmax(generationScores: MLXArray, referenceScores: MLXArray?, area: Int, frames: Int,
                                   videoArea: Int, generationLimit: Int, referenceStart: Int,
                                   referenceLength: Int) -> MLXArray {
        let rows = generationScores.dim(0), generation = generationScores.dim(1)
        let referenceCount = referenceScores?.dim(1) ?? 0
        let zeroKeys = (generationLimit - generation) + (referenceLength - referenceCount)
        guard NFKMLXDevice.currentType == .gpu, generationScores.size > 0 else {
            var scores = generationScores
            if let referenceScores { scores = concatenated([scores, referenceScores], axis: -1) }
            var peak = scores.max(axis: -1, keepDims: true)
            if zeroKeys > 0 { peak = maximum(peak, MLXArray(Float(0))) }
            let exponentials = exp(scores - peak)
            var total = exponentials.sum(axis: -1, keepDims: true)
            if zeroKeys > 0 { total = total + Float(zeroKeys) * exp(-peak) }
            return exponentials / total
        }
        let layout = MLXArray([rows, generation, referenceCount, videoArea, area, frames, generationLimit,
                               referenceStart, referenceLength].map(Int32.init))
        return torchPackedSoftmaxKernel(
            [generationScores, referenceScores ?? MLXArray.zeros([8]), layout],
            grid: (rows * 32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[rows, generation + referenceCount]], outputDTypes: [.float32])[0]
    }

    /// torch's CPU `layer_norm` of a bfloat16, float16, or float32 `x` over its last axis, rounded to `x`'s
    /// type. A missing `weight` is 1 and a missing `bias` 0.
    ///
    /// @discussion torch reads a half-precision row in vectors of 8 values, each widened to two float32
    /// vectors of 4 lanes, and a float32 row in vectors of 4. It sums the moments by Welford's update over
    /// chunks of 16 vectors, combines the chunks in a binary cascade, takes the elements past the last whole
    /// vector one at a time, and joins the 4 lanes. `rstd = 1 / sqrt(max(variance, 0) + eps)`. A
    /// half-precision row's output is `(x · rstd + (−rstd · mean)) · weight + bias`, and a float32 row's
    /// `((x − mean) · rstd) · weight + bias`. The vector steps round each operation, and the scalar steps (the
    /// lanes' join, each remaining element's second moment, and a half-precision row narrower than 8) fuse
    /// their multiply-adds. The kernel computes a row's chunks in parallel and replays torch's cascade over
    /// them, for rows of up to 32 chunks (4,096 half-precision values, 2,048 float32 ones); a wider row's lanes
    /// compute their chunks in order. On the CPU stream it takes MLX's fused kernel in float32, rounded once.
    static func torchLayerNorm(_ x: MLXArray, weight: MLXArray?, bias: MLXArray?, eps: Float) -> MLXArray {
        guard NFKMLXDevice.currentType == .gpu, x.size > 0 else {
            return MLXFast.layerNorm(x.asType(.float32), weight: weight?.asType(.float32),
                                     bias: bias?.asType(.float32), eps: eps).asType(x.dtype)
        }
        let width = x.dim(-1)
        return torchLayerNormFunction([x, weight ?? MLXArray.ones([width], dtype: x.dtype),
                                       bias ?? MLXArray.zeros([width], dtype: x.dtype), MLXArray([eps])])[0]
    }

    /// Sleef's `expf` from MLX operations, each rounding.
    static func torchExpOperations(_ x: MLXArray) -> MLXArray {
        let overflow = Float(88.72283935546875), underflow = Float(-104)
        let bounded = clip(x, min: underflow, max: overflow)
        let n = stopGradient(round(bounded * Float(1.442695040888963407359924681001892137426645954152985934135449406931)))
        var s = n * Float(-0.693145751953125) + bounded
        s = n * Float(-1.428606765330187045e-06) + s
        var u = MLXArray(Float(0.000198527617612853646278381))
        for coefficient: Float in [0.00139304355252534151077271, 0.00833336077630519866943359,
                                   0.0416664853692054748535156, 0.166666671633720397949219, 0.5] {
            u = u * s + coefficient
        }
        u = 1 + ((s * s) * u + s)
        let exponent = n.asType(.int32)
        let half = exponent >> Int32(1)
        func power(_ e: MLXArray) -> MLXArray { stopGradient(((e + Int32(127)) << Int32(23)).view(dtype: .float32)) }
        u = u * power(half) * power(exponent - half)
        u = which(x .> overflow, MLXArray(Float.infinity), u)
        return which(x .< underflow, MLXArray(Float(0)), u)
    }

    /// torch's exact `gelu` of `[x]` from MLX operations: the gradient of the GELU kernel.
    private static func geluOperations(_ inputs: [MLXArray]) -> [MLXArray] {
        let wide = inputs[0].asType(.float32)
        let y = wide * Float(0.5.squareRoot())
        let t = 1 / (Float(0.3275911) * abs(y) + 1)
        let polynomial = (((Float(1.061405429) * t - Float(1.453152027)) * t + Float(1.421413741)) * t
                          - Float(0.284496736)) * t + Float(0.254829592)
        let magnitude = (t * -torchExpOperations(-(y * y))) * polynomial + 1
        return [(wide * 0.5 * (1 + which(y .< 0, -magnitude, magnitude))).asType(inputs[0].dtype)]
    }

    /// torch's `silu` of `[x]` from MLX operations: the gradient of the SiLU kernel.
    private static func siluOperations(_ inputs: [MLXArray]) -> [MLXArray] {
        let wide = inputs[0].asType(.float32)
        return [(wide / (1 + torchExpOperations(-wide))).asType(inputs[0].dtype)]
    }

    /// torch's `softplus` of `[x]` from MLX operations, formed in float32 and rounded once: the gradient of the
    /// softplus kernel. The bound keeps the unselected branch's gradient finite.
    private static func softplusOperations(_ inputs: [MLXArray]) -> [MLXArray] {
        let wide = inputs[0].asType(.float32)
        return [which(wide .> 20, wide, log1p(exp(minimum(wide, 20)))).asType(inputs[0].dtype)]
    }

    /// The window pool of `[values, scores]` from MLX operations: the gradient of the window-pool kernel.
    private static func windowPoolOperations(_ inputs: [MLXArray]) -> [MLXArray] {
        [(inputs[0] * softmax(inputs[1], axis: -2, precise: true)).sum(axis: -2)]
    }

    /// The layer norm of `[x, weight, bias, eps]` from MLX operations in float32, rounded to `x`'s type: the
    /// gradient of the layer-norm kernel.
    private static func layerNormOperations(_ inputs: [MLXArray]) -> [MLXArray] {
        let x = inputs[0].asType(.float32)
        let mean = x.mean(axis: -1, keepDims: true)
        let centered = x - mean
        let normalized = centered * rsqrt(centered.square().mean(axis: -1, keepDims: true) + inputs[3])
        return [(normalized * inputs[1].asType(.float32) + inputs[2].asType(.float32)).asType(inputs[0].dtype)]
    }

    private static let torchExpFunction = CustomFunction {
        Forward { inputs in
            torchExpKernel([inputs[0]], grid: (inputs[0].size, 1, 1), threadGroup: (256, 1, 1),
                           outputShapes: [inputs[0].shape], outputDTypes: [.float32])
        }
        VJP { primals, cotangents in [cotangents[0] * exp(primals[0])] }
    }

    private static let torchGELUFunction = CustomFunction {
        Forward { inputs in
            torchGELUKernel([inputs[0]], template: [("T", inputs[0].dtype)], grid: (inputs[0].size, 1, 1),
                            threadGroup: (256, 1, 1), outputShapes: [inputs[0].shape], outputDTypes: [inputs[0].dtype])
        }
        VJP { primals, cotangents in vjp(geluOperations, primals: primals, cotangents: cotangents).1 }
    }

    private static let torchSiLUFunction = CustomFunction {
        Forward { inputs in
            torchSiLUKernel([inputs[0]], template: [("T", inputs[0].dtype)], grid: (inputs[0].size, 1, 1),
                            threadGroup: (256, 1, 1), outputShapes: [inputs[0].shape], outputDTypes: [inputs[0].dtype])
        }
        VJP { primals, cotangents in vjp(siluOperations, primals: primals, cotangents: cotangents).1 }
    }

    private static let torchSoftmaxFunction = CustomFunction {
        Forward { inputs in
            let x = inputs[0]
            let size = x.dim(-1), rows = x.size / size
            let probabilities = torchSoftmaxKernel(
                [x.reshaped([rows, size]), MLXArray([Int32(rows), Int32(size)])],
                grid: (rows * 32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[rows, size]], outputDTypes: [.float32])[0]
            return [probabilities.reshaped(x.shape)]
        }
        VJP { primals, cotangents in
            vjp({ [softmax($0[0], axis: -1, precise: true)] }, primals: primals, cotangents: cotangents).1
        }
    }

    private static let torchHalfSoftmaxFunction = CustomFunction {
        Forward { inputs in
            let x = inputs[0]
            let size = x.dim(-1), rows = x.size / size
            let probabilities = torchHalfSoftmaxKernel(
                [x.reshaped([rows, size]), MLXArray([Int32(rows), Int32(size)])],
                template: [("T", x.dtype)], grid: (rows * 32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[rows, size]], outputDTypes: [x.dtype])[0]
            return [probabilities.reshaped(x.shape)]
        }
        VJP { primals, cotangents in
            vjp({ [softmax($0[0].asType(.float32), axis: -1, precise: true).asType($0[0].dtype)] },
                primals: primals, cotangents: cotangents).1
        }
    }

    private static let torchSoftplusFunction = CustomFunction {
        Forward { inputs in
            torchSoftplusKernel([inputs[0]], template: [("T", inputs[0].dtype)], grid: (inputs[0].size, 1, 1),
                                threadGroup: (256, 1, 1), outputShapes: [inputs[0].shape],
                                outputDTypes: [inputs[0].dtype])
        }
        VJP { primals, cotangents in vjp(softplusOperations, primals: primals, cotangents: cotangents).1 }
    }

    private static let torchWindowPoolFunction = CustomFunction {
        Forward { inputs in
            let shape = inputs[0].shape
            let window = shape[shape.count - 2], width = shape[shape.count - 1]
            let outer = inputs[0].size / max(window * width, 1)
            let pooled = torchWindowPoolKernel(
                [inputs[0].reshaped([outer, window, width]), inputs[1].reshaped([outer, window, width]),
                 MLXArray([Int32(outer), Int32(window), Int32(width)])],
                grid: (outer * width, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[outer, width]], outputDTypes: [.float32])[0]
            return [pooled.reshaped(Array(shape.dropLast(2)) + [width])]
        }
        VJP { primals, cotangents in vjp(windowPoolOperations, primals: primals, cotangents: cotangents).1 }
    }

    private static let torchLayerNormFunction = CustomFunction {
        Forward { inputs in
            let x = inputs[0]
            let width = x.dim(-1), rows = x.size / width
            let normalized = torchLayerNormKernel(
                [x.reshaped([rows, width]), inputs[1], inputs[2], inputs[3], MLXArray([Int32(rows), Int32(width)])],
                template: [("T", x.dtype), ("H", x.dtype == .float32 ? 1 : 2)], grid: (rows * 32, 1, 1),
                threadGroup: (32, 1, 1), outputShapes: [[rows, width]], outputDTypes: [x.dtype])[0]
            return [normalized.reshaped(x.shape)]
        }
        VJP { primals, cotangents in vjp(layerNormOperations, primals: primals, cotangents: cotangents).1 }
    }

    /// Sleef's `expf` in Metal, shared by the kernels that exponentiate.
    private static let torchExpHeader = """
        inline float nfk_torch_expf(const float value) {
            const float n = rint(value * 1.442695040888963407359924681001892137426645954152985934135449406931f);
            float s = fma(n, -0.693145751953125f, value);
            s = fma(n, -1.428606765330187045e-06f, s);
            float u = 0.000198527617612853646278381f;
            u = fma(u, s, 0.00139304355252534151077271f);
            u = fma(u, s, 0.00833336077630519866943359f);
            u = fma(u, s, 0.0416664853692054748535156f);
            u = fma(u, s, 0.166666671633720397949219f);
            u = fma(u, s, 0.5f);
            const float square = s * s;
            u = 1.0f + fma(square, u, s);
            const int exponent = int(clamp(n, -150.0f, 150.0f));
            const int half_exponent = exponent >> 1;
            u = u * as_type<float>((half_exponent + 127) << 23);
            u = u * as_type<float>((exponent - half_exponent + 127) << 23);
            if (value > 88.72283935546875f) { u = INFINITY; }
            if (value < -104.0f) { u = 0.0f; }
            return u;
        }
        """

    /// Sleef's `log1pf` in Metal: `1 + d` scaled by a power of two `2ᵉ` into [0.75, 1.5), the logarithm of the
    /// scaled value formed from `m / (2 + m)` in double-float arithmetic with a short odd polynomial, and `e · ln 2`
    /// added in two parts. It matches torch's `log1p` below 10³⁸; softplus reads it up to `exp(20)`.
    private static let torchLog1pHeader = """
        inline float nfk_torch_log1pf(const float d) {
            float dp1 = d + 1.0f;
            const bool tiny = dp1 < 1.17549435e-38f;
            if (tiny) { dp1 = dp1 * 18446744073709551616.0f; }
            const float scaled = dp1 * (1.0f / 0.75f);
            int e = ((as_type<int>(scaled) >> 23) & 0xff) - 0x7f;
            const float t = as_type<float>(as_type<int>(1.0f) + ((-e) << 23));
            const float m = fma(d, t, t - 1.0f);
            if (tiny) { e = e - 64; }
            const float fe = float(e);
            const float sh = 0.69314718246459960938f * fe;
            const float sl = fma(-1.904654323148236017e-09f, fe, fma(0.69314718246459960938f, fe, -sh));
            const float ah = 2.0f + m;
            const float al = (2.0f - ah) + m;
            const float rt = 1.0f / ah;
            const float qs = m * rt;
            const float qu = fma(rt, m, -qs);
            const float qv = -fma(al, rt, fma(ah, rt, -1.0f));
            const float xl = fma(qs, qv, fma(0.0f, rt, qu));
            const float x2 = qs * qs;
            float p = 0.3027294874f;
            p = fma(p, x2, 0.3996108174f);
            p = fma(p, x2, 0.6666694880f);
            const float yh = qs * 2.0f;
            const float yl = xl * 2.0f;
            const float s1 = sh + yh;
            const float s1l = (((sh - s1) + yh) + sl) + yl;
            const float w = (x2 * qs) * p;
            const float s2 = s1 + w;
            const float s2l = ((s1 - s2) + w) + s1l;
            float r = s2 + s2l;
            if (-1.0f > d) { r = NAN; }
            if (d == -1.0f) { r = -INFINITY; }
            if (as_type<uint>(d) == 0x80000000u) { r = -0.0f; }
            return r;
        }
        """

    private static let torchExpKernel = MLXFast.metalKernel(
        name: "nfk_torch_exp",
        inputNames: ["x"],
        outputNames: ["y"],
        source: """
            uint index = thread_position_in_grid.x;
            y[index] = nfk_torch_expf(x[index]);
        """,
        header: torchExpHeader)

    // torch's `Vectorized<float>::erf` on NEON: `fmadd` is a fused multiply-add, and every other step rounds.
    private static let torchGELUKernel = MLXFast.metalKernel(
        name: "nfk_torch_gelu",
        inputNames: ["x"],
        outputNames: ["out"],
        source: """
            uint index = thread_position_in_grid.x;
            const float value = float(x[index]);
            const float y = value * 0.70710678118654752440f;
            const float magnitude_y = metal::abs(y);
            const float t = metal::precise::divide(1.0f, fma(0.3275911f, magnitude_y, 1.0f));
            float r = fma(1.061405429f, t, -1.453152027f);
            r = fma(r, t, 1.421413741f);
            r = fma(r, t, -0.284496736f);
            r = fma(r, t, 0.254829592f);
            const float square = y * y;
            const float decay = -nfk_torch_expf(-square);
            const float scaled = t * decay;
            const float magnitude = fma(scaled, r, 1.0f);
            const float erf_value = as_type<float>(as_type<uint>(magnitude) ^ (as_type<uint>(y) & 0x80000000u));
            const float half_value = value * 0.5f;
            const float sum = 1.0f + erf_value;
            out[index] = T(half_value * sum);
        """,
        header: torchExpHeader)

    private static let torchSiLUKernel = MLXFast.metalKernel(
        name: "nfk_torch_silu",
        inputNames: ["x"],
        outputNames: ["y"],
        source: """
            uint index = thread_position_in_grid.x;
            const float value = float(x[index]);
            const float denominator = 1.0f + nfk_torch_expf(-value);
            y[index] = T(metal::precise::divide(value, denominator));
        """,
        header: torchExpHeader)

    private static let torchSoftmaxKernel = MLXFast.metalKernel(
        name: "nfk_torch_softmax",
        inputNames: ["x", "shape"],
        outputNames: ["y"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint t = thread_position_in_threadgroup.x;
            const int rows = shape[0];
            const int size = shape[1];
            if (int(row) >= rows) { return; }
            device const float* X = x + row * size;
            device float* Y = y + row * size;
            threadgroup float lanes[4];
            threadgroup float shared[2];
            float peak = -INFINITY;
            for (int k = int(t); k < size; k += 32) { peak = max(peak, X[k]); }
            peak = simd_max(peak);
            for (int k = int(t); k < size; k += 32) { Y[k] = nfk_torch_expf(X[k] - peak); }
            threadgroup_barrier(mem_flags::mem_device);
            if (size < 4) {
                if (t == 0) {
                    float total = Y[0];
                    for (int k = 1; k < size; ++k) { total = total + Y[k]; }
                    shared[0] = metal::precise::divide(1.0f, total);
                }
            } else {
                if (t < 4) {
                    float total = Y[t];
                    int d = 4;
                    const int whole = size - size % 4;
                    for (; d < whole; d += 4) { total = total + Y[d + int(t)]; }
                    if (int(t) < size - d) { total = total + Y[d + int(t)]; }
                    lanes[t] = total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (t == 0) {
                    const float first = lanes[0] + lanes[2];
                    const float second = lanes[1] + lanes[3];
                    shared[0] = metal::precise::divide(1.0f, first + second);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float reciprocal = shared[0];
            for (int k = int(t); k < size; k += 32) { Y[k] = Y[k] * reciprocal; }
        """,
        header: torchExpHeader)

    private static let torchSoftplusKernel = MLXFast.metalKernel(
        name: "nfk_torch_softplus",
        inputNames: ["x"],
        outputNames: ["y"],
        source: """
            uint index = thread_position_in_grid.x;
            const float value = float(x[index]);
            y[index] = T(value > 20.0f ? value : nfk_torch_log1pf(nfk_torch_expf(value)));
        """,
        header: torchExpHeader + torchLog1pHeader)

    // Lanes 0 to 3 sum the exponentials of whole 4-value vectors, and the first thread adds the rest in order.
    private static let torchHalfSoftmaxKernel = MLXFast.metalKernel(
        name: "nfk_torch_half_softmax",
        inputNames: ["x", "shape"],
        outputNames: ["y"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint t = thread_position_in_threadgroup.x;
            const int rows = shape[0];
            const int size = shape[1];
            if (int(row) >= rows) { return; }
            device const T* X = x + row * size;
            device T* Y = y + row * size;
            threadgroup float lanes[4];
            threadgroup float shared[1];
            float peak = -INFINITY;
            for (int k = int(t); k < size; k += 32) { peak = max(peak, float(X[k])); }
            peak = simd_max(peak);
            const int whole = size - size % 4;
            if (t < 4) {
                float total = 0.0f;
                for (int k = int(t); k < whole; k += 4) { total = total + nfk_torch_expf(float(X[k]) - peak); }
                lanes[t] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t == 0) {
                float total = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
                for (int k = whole; k < size; ++k) { total = total + nfk_torch_expf(float(X[k]) - peak); }
                shared[0] = metal::precise::divide(1.0f, total);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float reciprocal = shared[0];
            for (int k = int(t); k < size; k += 32) { Y[k] = T(nfk_torch_expf(float(X[k]) - peak) * reciprocal); }
        """,
        header: torchExpHeader)

    // A thread per column. torch's cascade sums up to 4 levels, each level's step 2^max(4, ⌈log₂ window⌉ / 4) rows.
    private static let torchWindowPoolKernel = MLXFast.metalKernel(
        name: "nfk_torch_window_pool",
        inputNames: ["v", "s", "shape"],
        outputNames: ["y"],
        source: """
            const uint index = thread_position_in_grid.x;
            const int outer = shape[0];
            const int window = shape[1];
            const int width = shape[2];
            if (int(index) >= outer * width) { return; }
            const int o = int(index) / width;
            const int c = int(index) % width;
            device const float* S = s + o * window * width + c;
            device const float* V = v + o * window * width + c;
            float peak = S[0];
            for (int k = 1; k < window; ++k) { peak = max(peak, S[k * width]); }
            float total = nfk_torch_expf(S[0] - peak);
            for (int k = 1; k < window; ++k) { total = total + nfk_torch_expf(S[k * width] - peak); }
            int bits = 0;
            while ((1 << bits) < window) { ++bits; }
            const int power = max(4, bits / 4);
            const int step = 1 << power;
            const int mask = step - 1;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int i = 0;
            while (i + step <= window) {
                for (int j = 0; j < step; ++j, ++i) {
                    const float weight = metal::precise::divide(nfk_torch_expf(S[i * width] - peak), total);
                    const float product = V[i * width] * weight;
                    acc[0] = acc[0] + product;
                }
                for (int j = 1; j < 4; ++j) {
                    acc[j] = acc[j] + acc[j - 1];
                    acc[j - 1] = 0.0f;
                    if ((i & (mask << (j * power))) != 0) { break; }
                }
            }
            for (; i < window; ++i) {
                const float weight = metal::precise::divide(nfk_torch_expf(S[i * width] - peak), total);
                const float product = V[i * width] * weight;
                acc[0] = acc[0] + product;
            }
            for (int j = 1; j < 4; ++j) { acc[0] = acc[0] + acc[j]; }
            y[index] = acc[0];
        """,
        header: torchExpHeader)

    // Lanes 0 to 3 walk the buffer positions of their lane in order: the generation span, then the reference slot.
    // A masked position's exponential is 0 and leaves a lane's sum unchanged, so the walk skips it.
    private static let torchPackedSoftmaxKernel = MLXFast.metalKernel(
        name: "nfk_torch_packed_softmax",
        inputNames: ["generation", "reference", "layout"],
        outputNames: ["probabilities"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint t = thread_position_in_threadgroup.x;
            const int rows = layout[0];
            const int count = layout[1];
            const int reference_count = layout[2];
            const int video_area = layout[3];
            const int area = layout[4];
            const int frames = layout[5];
            const int generation_limit = layout[6];
            const int reference_start = layout[7];
            const int reference_length = layout[8];
            if (int(row) >= rows) { return; }
            const auto G = generation + row * count;
            const auto R = reference + row * reference_count;
            device float* P = probabilities + row * (count + reference_count);
            threadgroup float lanes[4];
            threadgroup float shared[1];
            float peak = -INFINITY;
            for (int k = int(t); k < count; k += 32) { peak = max(peak, G[k]); }
            for (int k = int(t); k < reference_count; k += 32) { peak = max(peak, R[k]); }
            peak = simd_max(peak);
            if (generation_limit > count || reference_length > reference_count) { peak = max(peak, 0.0f); }
            if (t < 4) {
                float total = 0.0f;
                for (int p = int(t); p < generation_limit; p += 4) {
                    const int f = p / video_area;
                    const int i = p % video_area;
                    const float score = (f < frames && i < area) ? G[f * area + i] : 0.0f;
                    total = total + nfk_torch_expf(score - peak);
                }
                if (reference_length > 0) {
                    int p = reference_start + (int(t) - reference_start % 4 + 4) % 4;
                    for (; p < reference_start + reference_length; p += 4) {
                        const int i = p - reference_start;
                        const float score = i < reference_count ? R[i] : 0.0f;
                        total = total + nfk_torch_expf(score - peak);
                    }
                }
                lanes[t] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t == 0) {
                const float first = lanes[0] + lanes[2];
                const float second = lanes[1] + lanes[3];
                shared[0] = metal::precise::divide(1.0f, first + second);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float reciprocal = shared[0];
            for (int k = int(t); k < count; k += 32) { P[k] = nfk_torch_expf(G[k] - peak) * reciprocal; }
            for (int k = int(t); k < reference_count; k += 32) {
                P[count + k] = nfk_torch_expf(R[k] - peak) * reciprocal;
            }
        """,
        header: torchExpHeader)

    // Each vector step is its own statement, so the compiler fuses no multiply-add torch's vector code
    // rounds; `fma` marks each one torch's scalar code fuses.
    private static let torchLayerNormKernel = MLXFast.metalKernel(
        name: "nfk_torch_layer_norm",
        inputNames: ["x", "gamma", "beta", "epsilon", "shape"],
        outputNames: ["y"],
        source: """
            const uint row = threadgroup_position_in_grid.x;
            const uint t = thread_position_in_threadgroup.x;
            const int rows = shape[0];
            const int N = shape[1];
            if (int(row) >= rows) { return; }
            device const T* X = x + row * N;
            const int n = N / (4 * H);
            const int m = (n + 15) / 16;
            const bool stored = m <= 32;
            threadgroup float chunk_m1[32][2][4];
            threadgroup float chunk_m2[32][2][4];
            threadgroup float lane_m1[4];
            threadgroup float lane_m2[4];
            threadgroup float stats[3];
            float c[16];
            for (int j = 0; j < 16; ++j) { c[j] = metal::precise::divide(1.0f, float(j + 1)); }
            const int lane = int(t) % 4;
            if (stored) {
                for (int i = int(t) / 4; i < m; i += 8) {
                    const int m0 = min(16, n - i * 16);
                    for (int half_index = 0; half_index < H; ++half_index) {
                        float m1v = 0.0f;
                        float m2v = 0.0f;
                        for (int j = 0; j < m0; ++j) {
                            const float value = float(X[(i * 16 + j) * 4 * H + half_index * 4 + lane]);
                            const float delta = value - m1v;
                            const float step = delta * c[j];
                            m1v = m1v + step;
                            const float residual = value - m1v;
                            const float product = delta * residual;
                            m2v = m2v + product;
                        }
                        chunk_m1[i][half_index][lane] = m1v;
                        chunk_m2[i][half_index][lane] = m2v;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t < 4) {
                const int depth = m <= 2 ? 1 : 32 - clz(uint(m - 1));
                int m0s[32];
                float m1s[32];
                float m2s[32];
                for (int d = 0; d < 32; ++d) { m0s[d] = 0; m1s[d] = 0.0f; m2s[d] = 0.0f; }
                for (int i = 0; i < m; ++i) {
                    const int m0 = min(16, n - i * 16);
                    for (int half_index = 0; half_index < H; ++half_index) {
                        float m1v = 0.0f;
                        float m2v = 0.0f;
                        if (stored) {
                            m1v = chunk_m1[i][half_index][lane];
                            m2v = chunk_m2[i][half_index][lane];
                        } else {
                            for (int j = 0; j < m0; ++j) {
                                const float value = float(X[(i * 16 + j) * 4 * H + half_index * 4 + lane]);
                                const float delta = value - m1v;
                                const float step = delta * c[j];
                                m1v = m1v + step;
                                const float residual = value - m1v;
                                const float product = delta * residual;
                                m2v = m2v + product;
                            }
                        }
                        const int total = m0s[0] + m0;
                        const float weight = total == 0 ? 0.0f : metal::precise::divide(float(m0), float(total));
                        const float delta = m1v - m1s[0];
                        const float step = weight * delta;
                        m1s[0] = m1s[0] + step;
                        const float square = delta * delta;
                        const float scaled = square * weight;
                        const float spread = scaled * float(m0s[0]);
                        const float inner = m2v + spread;
                        m2s[0] = m2s[0] + inner;
                        m0s[0] = total;
                    }
                    int mask = i + 1;
                    for (int j = 1; j < depth && (mask & 1) == 0; ++j) {
                        const int total = m0s[j] + m0s[j - 1];
                        const float weight = total == 0 ? 0.0f : metal::precise::divide(float(m0s[j - 1]), float(total));
                        const float delta = m1s[j - 1] - m1s[j];
                        const float step = weight * delta;
                        m1s[j] = m1s[j] + step;
                        const float square = delta * delta;
                        const float scaled = square * weight;
                        const float spread = scaled * float(m0s[j]);
                        const float inner = m2s[j - 1] + spread;
                        m2s[j] = m2s[j] + inner;
                        m1s[j - 1] = 0.0f;
                        m2s[j - 1] = 0.0f;
                        m0s[j] = total;
                        m0s[j - 1] = 0;
                        mask >>= 1;
                    }
                }
                for (int d = 1; d < depth; ++d) {
                    const int total = m0s[0] + m0s[d];
                    const float weight = total == 0 ? 0.0f : metal::precise::divide(float(m0s[d]), float(total));
                    const float delta = m1s[d] - m1s[0];
                    const float step = weight * delta;
                    m1s[0] = m1s[0] + step;
                    const float square = delta * delta;
                    const float scaled = square * weight;
                    const float spread = scaled * float(m0s[0]);
                    const float inner = m2s[d] + spread;
                    m2s[0] = m2s[0] + inner;
                    m0s[0] = total;
                }
                lane_m1[t] = m1s[0];
                lane_m2[t] = m2s[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t == 0) {
                int count = 0;
                float mean = 0.0f;
                float m2 = 0.0f;
                for (int k = n * 4 * H; k < N; ++k) {
                    const float value = float(X[k]);
                    const float delta = value - mean;
                    count += 1;
                    mean = mean + metal::precise::divide(delta, float(count));
                    const float residual = value - mean;
                    m2 = fma(delta, residual, m2);
                }
                const int lane_count = n * H;
                for (int l = 0; l < 4; ++l) {
                    const int total = count + lane_count;
                    const float weight = total == 0 ? 0.0f : metal::precise::divide(float(lane_count), float(total));
                    const float delta = lane_m1[l] - mean;
                    mean = fma(weight, delta, mean);
                    const float square = delta * delta;
                    const float scaled = square * weight;
                    const float inner = fma(scaled, float(count), lane_m2[l]);
                    m2 = m2 + inner;
                    count = total;
                }
                const float variance = metal::precise::divide(m2, float(N));
                const float rstd = metal::precise::divide(1.0f, metal::precise::sqrt(max(variance, 0.0f) + epsilon[0]));
                stats[0] = rstd;
                stats[1] = -rstd * mean;
                stats[2] = mean;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float rstd = stats[0];
            const float shift = stats[1];
            const float mean = stats[2];
            device T* Y = y + row * N;
            for (int k = int(t); k < N; k += 32) {
                float value;
                if (H == 1) {
                    const float centered = float(X[k]) - mean;
                    const float scaled = centered * rstd;
                    const float weighted = scaled * float(gamma[k]);
                    value = weighted + float(beta[k]);
                } else if (N < 8) {
                    value = fma(fma(float(X[k]), rstd, shift), float(gamma[k]), float(beta[k]));
                } else {
                    const float scaled = float(X[k]) * rstd;
                    const float shifted = scaled + shift;
                    const float weighted = shifted * float(gamma[k]);
                    value = weighted + float(beta[k]);
                }
                Y[k] = T(value);
            }
        """)
}
