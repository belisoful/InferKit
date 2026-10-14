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

    /// torch's `layer_norm` over the last axis of a half-precision `x`, rounded to `x`'s type, as
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

    /// torch's CPU `layer_norm` of a bfloat16 or float16 `x` over its last axis, rounded to `x`'s type. A
    /// missing `weight` is 1 and a missing `bias` 0.
    ///
    /// @discussion torch reads a row in vectors of 8 half-precision values, each widened to two float32
    /// vectors of 4 lanes. It sums the moments by Welford's update over chunks of 16 vectors, combines the
    /// chunks in a binary cascade, takes the elements past the last whole vector one at a time, and joins
    /// the 4 lanes. `rstd = 1 / sqrt(max(variance, 0) + eps)`, and `y = (x · rstd + (−rstd · mean)) · weight
    /// + bias`. The vector steps round each operation, and the scalar steps (the lanes' join, each remaining
    /// element's second moment, and a row narrower than 8) fuse their multiply-adds. The kernel computes a
    /// row's chunks in parallel and replays torch's cascade over them, for rows of up to 32 chunks (4,096
    /// values); a wider row's lanes compute their chunks in order. On the CPU stream it takes MLX's fused
    /// kernel in float32, rounded once.
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

    /// torch's `silu` of `[x]` from MLX operations: the gradient of the SiLU kernel.
    private static func siluOperations(_ inputs: [MLXArray]) -> [MLXArray] {
        let wide = inputs[0].asType(.float32)
        return [(wide / (1 + torchExpOperations(-wide))).asType(inputs[0].dtype)]
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

    private static let torchLayerNormFunction = CustomFunction {
        Forward { inputs in
            let x = inputs[0]
            let width = x.dim(-1), rows = x.size / width
            let normalized = torchLayerNormKernel(
                [x.reshaped([rows, width]), inputs[1], inputs[2], inputs[3], MLXArray([Int32(rows), Int32(width)])],
                template: [("T", x.dtype)], grid: (rows * 32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[rows, width]], outputDTypes: [x.dtype])[0]
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

    private static let torchExpKernel = MLXFast.metalKernel(
        name: "nfk_torch_exp",
        inputNames: ["x"],
        outputNames: ["y"],
        source: """
            uint index = thread_position_in_grid.x;
            y[index] = nfk_torch_expf(x[index]);
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
            const int n = N / 8;
            const int m = (n + 15) / 16;
            const bool stored = m <= 32;
            threadgroup float chunk_m1[32][2][4];
            threadgroup float chunk_m2[32][2][4];
            threadgroup float lane_m1[4];
            threadgroup float lane_m2[4];
            threadgroup float stats[2];
            float c[16];
            for (int j = 0; j < 16; ++j) { c[j] = metal::precise::divide(1.0f, float(j + 1)); }
            const int lane = int(t) % 4;
            if (stored) {
                for (int i = int(t) / 4; i < m; i += 8) {
                    const int m0 = min(16, n - i * 16);
                    for (int half_index = 0; half_index < 2; ++half_index) {
                        float m1v = 0.0f;
                        float m2v = 0.0f;
                        for (int j = 0; j < m0; ++j) {
                            const float value = float(X[(i * 16 + j) * 8 + half_index * 4 + lane]);
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
                    for (int half_index = 0; half_index < 2; ++half_index) {
                        float m1v = 0.0f;
                        float m2v = 0.0f;
                        if (stored) {
                            m1v = chunk_m1[i][half_index][lane];
                            m2v = chunk_m2[i][half_index][lane];
                        } else {
                            for (int j = 0; j < m0; ++j) {
                                const float value = float(X[(i * 16 + j) * 8 + half_index * 4 + lane]);
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
                for (int k = n * 8; k < N; ++k) {
                    const float value = float(X[k]);
                    const float delta = value - mean;
                    count += 1;
                    mean = mean + metal::precise::divide(delta, float(count));
                    const float residual = value - mean;
                    m2 = fma(delta, residual, m2);
                }
                const int lane_count = n * 2;
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
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float rstd = stats[0];
            const float shift = stats[1];
            device T* Y = y + row * N;
            for (int k = int(t); k < N; k += 32) {
                float value;
                if (N < 8) {
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
