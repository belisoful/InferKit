//
//  NFKMLXReferenceKernels.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXFast

// PyTorch's CPU arithmetic for the operations ``NFKMLXArithmetic/reference`` selects, as Metal kernels that
// match torch bit for bit: each computes its operation in torch's order, with a fused multiply-add where
// torch's compiled code fuses one and a separate rounding where it does not. A Metal kernel cannot dispatch
// on the CPU stream, so the CPU stream takes MLX operations. A kernel's gradient is the gradient of the MLX
// operations it stands for.

extension NFKReferenceRounding {

    /// The float32 exponentials of a half-precision attention's softmax, as ``NFKMLXHalfPrecision/arithmetic``
    /// selects: torch's (``torchExp(_:)``) or MLX's `exp`.
    static func softmaxExponential(_ x: MLXArray) -> MLXArray {
        NFKMLXHalfPrecision.arithmetic == .reference ? torchExp(x) : exp(x)
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

    /// torch's CPU `layer_norm` of a bfloat16 or float16 `x` over its last axis, rounded to `x`'s type. A
    /// missing `weight` is 1 and a missing `bias` 0.
    ///
    /// @discussion torch reads a row in vectors of 8 half-precision values, each widened to two float32
    /// vectors of 4 lanes. It sums the moments by Welford's update over chunks of 16 vectors, combines the
    /// chunks in a binary cascade, takes the elements past the last whole vector one at a time, and joins
    /// the 4 lanes. `rstd = 1 / sqrt(max(variance, 0) + eps)`, and `y = (x · rstd + (−rstd · mean)) · weight
    /// + bias`. The vector steps round each operation, and the scalar steps (the lanes' join, each remaining
    /// element's second moment, and a row narrower than 8) fuse their multiply-adds. On the CPU stream it
    /// takes MLX's fused kernel in float32, rounded once.
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

    private static let torchLayerNormFunction = CustomFunction {
        Forward { inputs in
            let x = inputs[0]
            let width = x.dim(-1), rows = x.size / width
            let normalized = torchLayerNormKernel(
                [x.reshaped([rows, width]), inputs[1], inputs[2], inputs[3], MLXArray([Int32(rows), Int32(width)])],
                template: [("T", x.dtype)], grid: (rows, 1, 1), threadGroup: (min(256, rows), 1, 1),
                outputShapes: [[rows, width]], outputDTypes: [x.dtype])[0]
            return [normalized.reshaped(x.shape)]
        }
        VJP { primals, cotangents in vjp(layerNormOperations, primals: primals, cotangents: cotangents).1 }
    }

    private static let torchExpKernel = MLXFast.metalKernel(
        name: "nfk_torch_exp",
        inputNames: ["x"],
        outputNames: ["y"],
        source: """
            uint index = thread_position_in_grid.x;
            const float value = x[index];
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
            y[index] = u;
        """)

    // Each vector step is its own statement, so the compiler fuses no multiply-add torch's vector code
    // rounds; `fma` marks each one torch's scalar code fuses.
    private static let torchLayerNormKernel = MLXFast.metalKernel(
        name: "nfk_torch_layer_norm",
        inputNames: ["x", "gamma", "beta", "epsilon", "shape"],
        outputNames: ["y"],
        source: """
            const uint row = thread_position_in_grid.x;
            const int rows = shape[0];
            const int N = shape[1];
            if (int(row) >= rows) { return; }
            device const T* X = x + row * N;
            const int n = N / 8;
            const int m = (n + 15) / 16;
            const int depth = m <= 2 ? 1 : 32 - clz(uint(m - 1));
            float c[16];
            for (int j = 0; j < 16; ++j) { c[j] = metal::precise::divide(1.0f, float(j + 1)); }
            int m0s[16];
            float m1s[16][4];
            float m2s[16][4];
            for (int d = 0; d < 16; ++d) {
                m0s[d] = 0;
                for (int l = 0; l < 4; ++l) { m1s[d][l] = 0.0f; m2s[d][l] = 0.0f; }
            }
            for (int i = 0; i < m; ++i) {
                const int m0 = min(16, n - i * 16);
                for (int half_index = 0; half_index < 2; ++half_index) {
                    float m1v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    float m2v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int j = 0; j < m0; ++j) {
                        for (int l = 0; l < 4; ++l) {
                            const float value = float(X[(i * 16 + j) * 8 + half_index * 4 + l]);
                            const float delta = value - m1v[l];
                            const float step = delta * c[j];
                            m1v[l] = m1v[l] + step;
                            const float residual = value - m1v[l];
                            const float product = delta * residual;
                            m2v[l] = m2v[l] + product;
                        }
                    }
                    const int total = m0s[0] + m0;
                    const float weight = total == 0 ? 0.0f : metal::precise::divide(float(m0), float(total));
                    for (int l = 0; l < 4; ++l) {
                        const float delta = m1v[l] - m1s[0][l];
                        const float step = weight * delta;
                        m1s[0][l] = m1s[0][l] + step;
                        const float square = delta * delta;
                        const float scaled = square * weight;
                        const float spread = scaled * float(m0s[0]);
                        const float inner = m2v[l] + spread;
                        m2s[0][l] = m2s[0][l] + inner;
                    }
                    m0s[0] = total;
                }
                int mask = i + 1;
                for (int j = 1; j < depth && (mask & 1) == 0; ++j) {
                    const int total = m0s[j] + m0s[j - 1];
                    const float weight = total == 0 ? 0.0f : metal::precise::divide(float(m0s[j - 1]), float(total));
                    for (int l = 0; l < 4; ++l) {
                        const float delta = m1s[j - 1][l] - m1s[j][l];
                        const float step = weight * delta;
                        m1s[j][l] = m1s[j][l] + step;
                        const float square = delta * delta;
                        const float scaled = square * weight;
                        const float spread = scaled * float(m0s[j]);
                        const float inner = m2s[j - 1][l] + spread;
                        m2s[j][l] = m2s[j][l] + inner;
                        m1s[j - 1][l] = 0.0f;
                        m2s[j - 1][l] = 0.0f;
                    }
                    m0s[j] = total;
                    m0s[j - 1] = 0;
                    mask >>= 1;
                }
            }
            for (int d = 1; d < depth; ++d) {
                const int total = m0s[0] + m0s[d];
                const float weight = total == 0 ? 0.0f : metal::precise::divide(float(m0s[d]), float(total));
                for (int l = 0; l < 4; ++l) {
                    const float delta = m1s[d][l] - m1s[0][l];
                    const float step = weight * delta;
                    m1s[0][l] = m1s[0][l] + step;
                    const float square = delta * delta;
                    const float scaled = square * weight;
                    const float spread = scaled * float(m0s[0]);
                    const float inner = m2s[d][l] + spread;
                    m2s[0][l] = m2s[0][l] + inner;
                }
                m0s[0] = total;
            }
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
                const float delta = m1s[0][l] - mean;
                mean = fma(weight, delta, mean);
                const float square = delta * delta;
                const float scaled = square * weight;
                const float inner = fma(scaled, float(count), m2s[0][l]);
                m2 = m2 + inner;
                count = total;
            }
            const float variance = metal::precise::divide(m2, float(N));
            const float rstd = metal::precise::divide(1.0f, metal::precise::sqrt(max(variance, 0.0f) + epsilon[0]));
            const float shift = -rstd * mean;
            device T* Y = y + row * N;
            for (int k = 0; k < N; ++k) {
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
