//
//  NFKMLXReferenceScans.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXFast

// The chunked scans of transformers' CPU paths as ``NFKMLXArithmetic/reference`` computes them: Mamba-2's
// `mamba2_chunk_scan` and the gated delta rule's `torch_chunk_gated_delta_rule` (transformers 5.17), each equal bit
// for bit to torch's float32 result. torch's CPU float32 `matmul` and `linalg.solve_triangular` call Accelerate's
// `sgemm` and `strsm`, which MLX's CPU stream and a direct `cblas_strsm` reproduce. torch's `cumsum` accumulates in
// float64, which MLX's CPU stream reproduces. Its exponentials are Sleef's (``NFKReferenceRounding/torchExp(_:)``),
// and its sums along an axis take the orders the kernels here replay. A scan's gradient is its sequential
// recurrence's.

extension NFKReferenceRounding {

    /// torch's CPU `cumsum` of float32 `x` along `axis`: each running sum formed in float64 and rounded to float32.
    static func torchCumsum(_ x: MLXArray, axis: Int) -> MLXArray {
        cumsum(x.asType(.float64, stream: .cpu), axis: axis, stream: .cpu).asType(.float32, stream: .cpu)
    }

    /// transformers' `segment_sum` of float32 `a` `[..., T]` as torch computes it: `[..., T, T]` whose entry `(i, j)`
    /// is the float64 running sum of `a[j + 1 ... i]` rounded to float32 where `i ≥ j`, and −∞ above the diagonal.
    static func torchSegmentSum(_ a: MLXArray) -> MLXArray {
        let length = a.dim(-1)
        let wide = a.asType(.float64, stream: .cpu).expandedDimensions(axis: -1, stream: .cpu)
        let rows = broadcast(wide, to: a.shape + [length], stream: .cpu)
        let strictlyLower = tri(length, k: -1, type: Bool.self, stream: .cpu)
        let masked = which(strictlyLower, rows, MLXArray(Double(0)), stream: .cpu)
        let sums = cumsum(masked, axis: -2, stream: .cpu).asType(.float32, stream: .cpu)
        return which(tri(length, k: 0, type: Bool.self, stream: .cpu), sums, MLXArray(-Float.infinity), stream: .cpu)
    }

    /// torch's CPU float32 `sum` over the last axis of `x`, kept as one value per row: 4-lane vectors in groups of 4
    /// summed by torch's cascade, the vectors past the last group added to the first lanes, the four partials joined,
    /// and the values past the last whole vector added before the lanes.
    static func torchLastAxisSum(_ x: MLXArray) -> MLXArray {
        let width = x.dim(-1), rows = x.size / max(width, 1)
        return torchLastAxisSumKernel(
            [x.asType(.float32).reshaped([rows, width]), MLXArray([Int32(rows), Int32(width)])],
            grid: (rows, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[rows]], outputDTypes: [.float32])[0]
            .reshaped(Array(x.shape.dropLast()) + [1])
    }

    /// Whether the reference arithmetic computes a chunked scan from the GPU: the kernels cannot dispatch on the CPU
    /// stream, where the scans take their sequential recurrences.
    static var computesReferenceScans: Bool {
        NFKMLXHalfPrecision.arithmetic == .reference && NFKMLXDevice.currentType == .gpu && strsm != nil
    }

    private typealias TriangularSolve = @convention(c) (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Float,
                                                         UnsafePointer<Float>?, Int32, UnsafeMutablePointer<Float>?,
                                                         Int32) -> Void

    /// Accelerate's `cblas_strsm`, read through its C symbol: the Swift declaration is deprecated in favor of an
    /// interface a C macro selects, which a Swift package cannot define for its importer.
    private static let strsm: TriangularSolve? = {
        guard let library = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_LAZY),
              let symbol = dlsym(library, "cblas_strsm") else { return nil }
        return unsafeBitCast(symbol, to: TriangularSolve.self)
    }()

    /// transformers' `mamba2_chunk_scan` over float32 `x` `[B, L, H, P]`, the discretization step `dt` `[B, L, H]`
    /// after its softplus and clamp, `a` `[H]`, `b` and `c` `[B, L, H, N]` repeated to the heads, and `d` `[H]`:
    /// `y + d · x` `[B, L, H, P]` in float32, in chunks of `chunkSize` positions. Each product and sum takes torch's
    /// order, which holds from 8 state values a head; torch sums a head width times head count under 16 another way.
    static func torchMamba2ChunkScan(x: MLXArray, dt: MLXArray, a: MLXArray, b: MLXArray, c: MLXArray, d: MLXArray,
                                     chunkSize: Int) -> MLXArray {
        let function = CustomFunction {
            Forward { inputs in
                [mamba2ChunkScan(x: inputs[0], dt: inputs[1], a: inputs[2], b: inputs[3], c: inputs[4], d: inputs[5],
                                 chunkSize: chunkSize)]
            }
            VJP { primals, cotangents in vjp(mamba2SequentialScan, primals: primals, cotangents: cotangents).1 }
        }
        return function([x, dt, a, b, c, d])[0]
    }

    /// The Mamba-2 scan as a sequential recurrence from MLX operations, `[x, dt, a, b, c, d]` → `[y + d · x]`: the
    /// gradient of ``torchMamba2ChunkScan(x:dt:a:b:c:d:chunkSize:)``.
    static func mamba2SequentialScan(_ inputs: [MLXArray]) -> [MLXArray] {
        let (x, dt, a, b, c, d) = (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])
        let (batch, length, heads, width) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let decay = exp(dt * a.reshaped([1, 1, heads]))
        var state = MLXArray.zeros([batch, heads, width, b.dim(3)], dtype: x.dtype)
        var outputs = [MLXArray]()
        for t in 0 ..< length {
            let written = dt[0..., t].reshaped([batch, heads, 1, 1]) * x[0..., t].expandedDimensions(axis: -1)
                * b[0..., t].expandedDimensions(axis: 2)
            state = decay[0..., t].reshaped([batch, heads, 1, 1]) * state + written
            outputs.append((state * c[0..., t].expandedDimensions(axis: 2)).sum(axis: -1))
        }
        return [stacked(outputs, axis: 1) + d.reshaped([1, 1, heads, 1]) * x]
    }

    private static func mamba2ChunkScan(x: MLXArray, dt: MLXArray, a: MLXArray, b: MLXArray, c: MLXArray,
                                        d: MLXArray, chunkSize: Int) -> MLXArray {
        let (batch, length, heads, width) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let state = b.dim(3)
        let pad = (chunkSize - length % chunkSize) % chunkSize
        let chunks = (length + pad) / chunkSize
        func padded(_ t: MLXArray) -> MLXArray {
            guard pad > 0 else { return t }
            return MLX.padded(t, widths: [IntOrPair(integerLiteral: 0), IntOrPair((0, pad))] + Array(repeating: IntOrPair(integerLiteral: 0), count: t.ndim - 2))
        }
        let residual = d.reshaped([heads, 1]) * padded(x)
        let steps = padded(x * dt.expandedDimensions(axis: -1)).reshaped([batch, chunks, chunkSize, heads, width])
        let bs = padded(b).reshaped([batch, chunks, chunkSize, heads, state])
        let cs = padded(c).reshaped([batch, chunks, chunkSize, heads, state])
        let rates = padded(a * dt).reshaped([batch, chunks, chunkSize, heads]).transposed(0, 3, 1, 2)

        let cumulative = torchCumsum(rates, axis: -1)                                   // [B, H, C, l]
        let within = torchExp(torchSegmentSum(rates))                                   // [B, H, C, l, l]
        let gram = scanKernel(ssdGramKernel, [cs, bs], [batch, chunks, chunkSize, heads, state],
                              [batch, chunks, chunkSize, chunkSize, heads])
        let mixed = gram * within.transposed(0, 2, 3, 4, 1)
        let diagonal = scanKernel(ssdDiagonalKernel, [mixed, steps], [batch, chunks, chunkSize, heads, width],
                                  [batch, chunks, chunkSize, heads, width])
        let toEnd = torchExp(cumulative[.ellipsis, (chunkSize - 1)...] - cumulative)
        let decayedB = bs * toEnd.transposed(0, 2, 3, 1).expandedDimensions(axis: -1)
        let states = scanKernel(ssdStatesKernel, [decayedB, steps], [batch, chunks, chunkSize, heads, width, state],
                                [batch, chunks, heads, width, state])
        let initial = concatenated([MLXArray.zeros([batch, 1, heads, width, state]), states], axis: 1)
        let ends = MLX.padded(cumulative[.ellipsis, chunkSize - 1], widths: [IntOrPair(integerLiteral: 0), IntOrPair(integerLiteral: 0), IntOrPair((1, 0))])
        let between = torchExp(torchSegmentSum(ends)).transposed(0, 3, 2, 1)           // [B, j, i, H]
        let passed = scanKernel(ssdPassingKernel, [between, initial], [batch, chunks + 1, heads, width, state],
                                [batch, chunks + 1, heads, width, state])
        let readout = scanKernel(ssdReadoutKernel, [cs, passed[0..., 0 ..< chunks]],
                                 [batch, chunks, chunkSize, heads, width, state], [batch, chunks, chunkSize, heads, width])
        let offDiagonal = readout * torchExp(cumulative).transposed(0, 2, 3, 1).expandedDimensions(axis: -1)
        let y = (diagonal + offDiagonal).reshaped([batch, -1, heads, width]) + residual
        return y[0..., 0 ..< length]
    }

    /// transformers 5.17's `torch_chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel`, in chunks of `chunkSize`:
    /// `queries` and `keys` `[B, L, H, K]` repeated to the value heads, `values` `[B, L, H, V]`, the log decay
    /// `logDecay` and write strength `beta` `[B, L, H]`, from a zero state. The queries and keys are normalized in
    /// float32 (`x · 1 / sqrt(Σx² + 1e-6)`) and the queries scaled by `K^-0.5`; the matrix products run on the CPU
    /// stream and the unit lower-triangular solves through `cblas_strsm`. Returns the float32 read `[B, L, H, V]`.
    static func torchChunkGatedDeltaRule(queries: MLXArray, keys: MLXArray, values: MLXArray, logDecay: MLXArray,
                                         beta: MLXArray, chunkSize: Int = 64) -> MLXArray {
        let function = CustomFunction {
            Forward { inputs in
                [chunkGatedDeltaRule(queries: inputs[0], keys: inputs[1], values: inputs[2], logDecay: inputs[3],
                                     beta: inputs[4], chunkSize: chunkSize)]
            }
            VJP { primals, cotangents in vjp(gatedDeltaRecurrence, primals: primals, cotangents: cotangents).1 }
        }
        return function([queries, keys, values, logDecay, beta])[0]
    }

    /// The gated delta rule as a sequential recurrence from MLX operations, `[queries, keys, values, logDecay, beta]`
    /// → `[read]`: the gradient of ``torchChunkGatedDeltaRule(queries:keys:values:logDecay:beta:chunkSize:)``.
    static func gatedDeltaRecurrence(_ inputs: [MLXArray]) -> [MLXArray] {
        func normalized(_ x: MLXArray) -> MLXArray {
            let wide = x.asType(.float32)
            return wide * (1 / sqrt((wide * wide).sum(axis: -1, keepDims: true) + Float(1e-6)))
        }
        let queries = normalized(inputs[0]) * Float(pow(Double(inputs[0].dim(3)), -0.5)), keys = normalized(inputs[1])
        let values = inputs[2].asType(.float32), decay = exp(inputs[3].asType(.float32)), beta = inputs[4].asType(.float32)
        let (batch, length, heads) = (queries.dim(0), queries.dim(1), queries.dim(2))
        var state = MLXArray.zeros([batch, heads, queries.dim(3), values.dim(3)])
        var outputs = [MLXArray]()
        for step in 0 ..< length {
            let k = keys[0..., step].expandedDimensions(axis: -1)
            state = state * decay[0..., step].reshaped([batch, heads, 1, 1])
            let correction = (values[0..., step] - (state * k).sum(axis: 2)) * beta[0..., step].reshaped([batch, heads, 1])
            state = state + k * correction.expandedDimensions(axis: 2)
            outputs.append((state * queries[0..., step].expandedDimensions(axis: -1)).sum(axis: 2))
        }
        return [stacked(outputs, axis: 1)]
    }

    private static func chunkGatedDeltaRule(queries: MLXArray, keys: MLXArray, values: MLXArray, logDecay: MLXArray,
                                            beta: MLXArray, chunkSize: Int) -> MLXArray {
        let (batch, length, heads, keyWidth) = (keys.dim(0), keys.dim(1), keys.dim(2), keys.dim(3))
        let valueWidth = values.dim(3)
        func normalized(_ x: MLXArray) -> MLXArray {
            let wide = x.asType(.float32).transposed(0, 2, 1, 3)
            return wide * (1 / sqrt(torchLastAxisSum(wide * wide) + Float(1e-6)))
        }
        var q = normalized(queries) * Float(pow(Double(keyWidth), -0.5))
        var k = normalized(keys)
        var v = values.asType(.float32).transposed(0, 2, 1, 3)
        var b = beta.asType(.float32).transposed(0, 2, 1), g = logDecay.asType(.float32).transposed(0, 2, 1)
        let pad = (chunkSize - length % chunkSize) % chunkSize
        if pad > 0 {
            let rows = [IntOrPair(integerLiteral: 0), IntOrPair(integerLiteral: 0), IntOrPair((0, pad))]
            (q, k, v) = (MLX.padded(q, widths: rows + [IntOrPair(integerLiteral: 0)]), MLX.padded(k, widths: rows + [IntOrPair(integerLiteral: 0)]),
                         MLX.padded(v, widths: rows + [IntOrPair(integerLiteral: 0)]))
            (b, g) = (MLX.padded(b, widths: rows), MLX.padded(g, widths: rows))
        }
        let chunks = (length + pad) / chunkSize
        func chunked(_ t: MLXArray) -> MLXArray { t.reshaped([batch, heads, chunks, chunkSize, t.dim(-1)]) }
        let weightedValues = chunked(v * b.expandedDimensions(axis: -1))
        let weightedKeys = chunked(k * b.expandedDimensions(axis: -1))
        q = chunked(q)
        k = chunked(k)
        let cumulative = torchCumsum(g.reshaped([batch, heads, chunks, chunkSize]), axis: -1)
        let upper = triu(MLXArray.ones([chunkSize, chunkSize], type: Bool.self), k: 1)
        let pairwise = torchExp(which(upper, MLXArray(-Float.infinity),
                                      cumulative.expandedDimensions(axis: -1) - cumulative.expandedDimensions(axis: -2)))
        let system = matmul(weightedKeys, k.swappedAxes(-1, -2), stream: .cpu) * pairwise
        let intra = matmul(q, k.swappedAxes(-1, -2), stream: .cpu) * pairwise
        let growth = torchExp(cumulative)
        let newValues = solveUnitLowerTriangular(system, weightedValues)
        let readDecay = solveUnitLowerTriangular(system, weightedKeys * growth.expandedDimensions(axis: -1))

        let decayedQueries = q * growth.expandedDimensions(axis: -1)
        let decayedKeys = k * torchExp(cumulative[.ellipsis, (chunkSize - 1)...] - cumulative).expandedDimensions(axis: -1)
        let chunkDecay = torchExp(cumulative[.ellipsis, chunkSize - 1]).reshaped([batch, heads, chunks, 1, 1])
        var state = MLXArray.zeros([batch, heads, keyWidth, valueWidth])
        var outputs = [MLXArray]()
        for chunk in 0 ..< chunks {
            let correction = newValues[0..., 0..., chunk] - matmul(readDecay[0..., 0..., chunk], state, stream: .cpu)
            outputs.append(matmul(decayedQueries[0..., 0..., chunk], state, stream: .cpu)
                           + matmul(intra[0..., 0..., chunk], correction, stream: .cpu))
            state = state * chunkDecay[0..., 0..., chunk]
                + matmul(decayedKeys[0..., 0..., chunk].swappedAxes(-1, -2), correction, stream: .cpu)
        }
        let read = stacked(outputs, axis: 2).reshaped([batch, heads, -1, valueWidth])[0..., 0..., 0 ..< length]
        return read.transposed(0, 2, 1, 3)
    }

    /// `X` with `A X = B` for each unit lower-triangular `A` `[..., n, n]` and `B` `[..., n, m]`, by Accelerate's
    /// `cblas_strsm`, which torch's CPU `linalg.solve_triangular` calls.
    private static func solveUnitLowerTriangular(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let (n, m) = (b.dim(-2), b.dim(-1))
        let matrices = a.asType(.float32).asArray(Float.self)
        var solutions = b.asType(.float32).asArray(Float.self)
        let count = solutions.count / (n * m)
        // CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit.
        let (rowMajor, left, lower, noTranspose, unit): (Int32, Int32, Int32, Int32, Int32) = (101, 141, 122, 111, 132)
        guard let strsm else { return b }
        matrices.withUnsafeBufferPointer { matrix in
            solutions.withUnsafeMutableBufferPointer { solution in
                for index in 0 ..< count {
                    strsm(rowMajor, left, lower, noTranspose, unit, Int32(n), Int32(m), 1,
                          matrix.baseAddress! + index * n * n, Int32(n), solution.baseAddress! + index * n * m, Int32(m))
                }
            }
        }
        return MLXArray(solutions, b.shape)
    }

    private static func scanKernel(_ kernel: MLXFast.MLXFastKernel, _ inputs: [MLXArray], _ dims: [Int],
                                   _ shape: [Int]) -> MLXArray {
        kernel(inputs.map { $0.asType(.float32) } + [MLXArray(dims.map(Int32.init))],
               grid: (shape.reduce(1, *), 1, 1), threadGroup: (256, 1, 1), outputShapes: [shape],
               outputDTypes: [.float32])[0]
    }

    /// torch's float32 sum orders: the outer-axis cascade (`multi_row_sum`), `row_sum`, the outer sum that picks
    /// between them by column, and the inner-axis sum (`vectorized_inner_sum`). Each term is its own statement, so
    /// the compiler fuses no multiply into an add torch rounds.
    private static let torchSumHeader = NFKReferenceRounding.metalContractOff + """
        template <typename Term>
        inline float nfk_outer_cascade(const int size, thread const Term& term) {
            int bits = 0;
            while ((1 << bits) < size) { ++bits; }
            const int power = max(4, bits / 4);
            const int step = 1 << power;
            const int mask = step - 1;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int i = 0;
            while (i + step <= size) {
                for (int j = 0; j < step; ++j, ++i) {
                    const float value = term(i);
                    acc[0] = acc[0] + value;
                }
                for (int j = 1; j < 4; ++j) {
                    acc[j] = acc[j] + acc[j - 1];
                    acc[j - 1] = 0.0f;
                    if ((i & (mask << (j * power))) != 0) { break; }
                }
            }
            for (; i < size; ++i) {
                const float value = term(i);
                acc[0] = acc[0] + value;
            }
            for (int j = 1; j < 4; ++j) { acc[0] = acc[0] + acc[j]; }
            return acc[0];
        }

        template <typename Term>
        inline float nfk_row_sum(const int size, thread const Term& term) {
            const int groups = size / 4;
            int bits = 0;
            while ((1 << bits) < groups) { ++bits; }
            const int power = max(4, bits / 4);
            const int step = 1 << power;
            const int mask = step - 1;
            float acc[4][4];
            for (int l = 0; l < 4; ++l) { for (int k = 0; k < 4; ++k) { acc[l][k] = 0.0f; } }
            int g = 0;
            while (g + step <= groups) {
                for (int j = 0; j < step; ++j, ++g) {
                    for (int k = 0; k < 4; ++k) {
                        const float value = term(g * 4 + k);
                        acc[0][k] = acc[0][k] + value;
                    }
                }
                for (int j = 1; j < 4; ++j) {
                    for (int k = 0; k < 4; ++k) { acc[j][k] = acc[j][k] + acc[j - 1][k]; acc[j - 1][k] = 0.0f; }
                    if ((g & (mask << (j * power))) != 0) { break; }
                }
            }
            for (; g < groups; ++g) {
                for (int k = 0; k < 4; ++k) {
                    const float value = term(g * 4 + k);
                    acc[0][k] = acc[0][k] + value;
                }
            }
            for (int j = 1; j < 4; ++j) { for (int k = 0; k < 4; ++k) { acc[0][k] = acc[0][k] + acc[j][k]; } }
            for (int r = groups * 4; r < size; ++r) {
                const float value = term(r);
                acc[0][0] = acc[0][0] + value;
            }
            for (int k = 1; k < 4; ++k) { acc[0][0] = acc[0][0] + acc[0][k]; }
            return acc[0][0];
        }

        template <typename Term>
        inline float nfk_outer_sum(const int size, const int column, const int columns, thread const Term& term) {
            const int cascaded = columns >= 4 ? (columns / 16) * 16 : 0;
            return column < cascaded ? nfk_outer_cascade(size, term) : nfk_row_sum(size, term);
        }

        template <typename Term>
        inline float nfk_inner_cascade(const int n, thread const Term& term) {
            const int vectors = n / 4;
            const int groups = vectors / 4;
            if (groups < 16) {
                float lanes[16];
                for (int k = 0; k < 16; ++k) { lanes[k] = 0.0f; }
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < 16; ++k) {
                        const float value = term(g * 16 + k);
                        lanes[k] = lanes[k] + value;
                    }
                }
                for (int v = groups * 4; v < vectors; ++v) {
                    for (int lane = 0; lane < 4; ++lane) {
                        const float value = term(v * 4 + lane);
                        lanes[lane] = lanes[lane] + value;
                    }
                }
                for (int k = 1; k < 4; ++k) {
                    for (int lane = 0; lane < 4; ++lane) { lanes[lane] = lanes[lane] + lanes[k * 4 + lane]; }
                }
                float total = 0.0f;
                for (int e = vectors * 4; e < n; ++e) {
                    const float value = term(e);
                    total = total + value;
                }
                for (int lane = 0; lane < 4; ++lane) { total = total + lanes[lane]; }
                return total;
            }
            int bits = 0;
            while ((1 << bits) < groups) { ++bits; }
            const int power = max(4, bits / 4);
            const int step = 1 << power;
            const int mask = step - 1;
            float acc[4][16];
            for (int l = 0; l < 4; ++l) { for (int k = 0; k < 16; ++k) { acc[l][k] = 0.0f; } }
            int g = 0;
            while (g + step <= groups) {
                for (int j = 0; j < step; ++j, ++g) {
                    for (int k = 0; k < 16; ++k) {
                        const float value = term(g * 16 + k);
                        acc[0][k] = acc[0][k] + value;
                    }
                }
                for (int j = 1; j < 4; ++j) {
                    for (int k = 0; k < 16; ++k) { acc[j][k] = acc[j][k] + acc[j - 1][k]; acc[j - 1][k] = 0.0f; }
                    if ((g & (mask << (j * power))) != 0) { break; }
                }
            }
            for (; g < groups; ++g) {
                for (int k = 0; k < 16; ++k) {
                    const float value = term(g * 16 + k);
                    acc[0][k] = acc[0][k] + value;
                }
            }
            for (int j = 1; j < 4; ++j) { for (int k = 0; k < 16; ++k) { acc[0][k] = acc[0][k] + acc[j][k]; } }
            for (int v = groups * 4; v < vectors; ++v) {
                for (int lane = 0; lane < 4; ++lane) {
                    const float value = term(v * 4 + lane);
                    acc[0][lane] = acc[0][lane] + value;
                }
            }
            for (int k = 1; k < 4; ++k) {
                for (int lane = 0; lane < 4; ++lane) { acc[0][lane] = acc[0][lane] + acc[0][k * 4 + lane]; }
            }
            float total = 0.0f;
            for (int e = vectors * 4; e < n; ++e) {
                const float value = term(e);
                total = total + value;
            }
            for (int lane = 0; lane < 4; ++lane) { total = total + acc[0][lane]; }
            return total;
        }

        struct nfk_product {
            const device float* a;
            const device float* b;
            int stride_a;
            int stride_b;
            float operator()(const int r) const {
                const float product = a[r * stride_a] * b[r * stride_b];
                return product;
            }
        };

        struct nfk_value {
            const device float* a;
            float operator()(const int r) const { return a[r]; }
        };
        """

    private static let torchLastAxisSumKernel = MLXFast.metalKernel(
        name: "nfk_torch_last_axis_sum",
        inputNames: ["x", "dims"],
        outputNames: ["y"],
        source: """
            const uint index = thread_position_in_grid.x;
            const int rows = dims[0], n = dims[1];
            if (int(index) >= rows) { return; }
            nfk_value term = {x + index * n};
            y[index] = nfk_inner_cascade(n, term);
        """,
        header: torchSumHeader)

    // G[b, c, i, j, h] = Σₙ C[b, c, i, h, n] · B[b, c, j, h, n].
    private static let ssdGramKernel = MLXFast.metalKernel(
        name: "nfk_ssd_gram",
        inputNames: ["cm", "bm", "dims"],
        outputNames: ["g"],
        source: """
            const uint index = thread_position_in_grid.x;
            const int batches = dims[0], chunks = dims[1], l = dims[2], h = dims[3], n = dims[4];
            if (int(index) >= batches * chunks * l * l * h) { return; }
            int r = int(index);
            const int head = r % h; r /= h;
            const int j = r % l; r /= l;
            const int i = r % l; r /= l;
            const int bc = r;
            nfk_product term = {cm + ((bc * l + i) * h + head) * n, bm + ((bc * l + j) * h + head) * n, 1, 1};
            g[index] = nfk_inner_cascade(n, term);
        """,
        header: torchSumHeader)

    // Y[b, c, i, h, p] = Σⱼ M[b, c, i, j, h] · X[b, c, j, h, p].
    private static let ssdDiagonalKernel = MLXFast.metalKernel(
        name: "nfk_ssd_diagonal",
        inputNames: ["m", "x", "dims"],
        outputNames: ["y"],
        source: """
            const uint index = thread_position_in_grid.x;
            const int batches = dims[0], chunks = dims[1], l = dims[2], h = dims[3], p = dims[4];
            if (int(index) >= batches * chunks * l * h * p) { return; }
            int r = int(index);
            const int d = r % p; r /= p;
            const int head = r % h; r /= h;
            const int i = r % l; r /= l;
            const int bc = r;
            nfk_product term = {m + ((bc * l + i) * l) * h + head, x + (bc * l * h + head) * p + d, h, h * p};
            y[index] = nfk_outer_sum(l, head * p + d, h * p, term);
        """,
        header: torchSumHeader)

    // S[b, c, h, p, n] = Σₜ Bd[b, c, t, h, n] · X[b, c, t, h, p].
    private static let ssdStatesKernel = MLXFast.metalKernel(
        name: "nfk_ssd_states",
        inputNames: ["bd", "x", "dims"],
        outputNames: ["s"],
        source: """
            const uint index = thread_position_in_grid.x;
            const int batches = dims[0], chunks = dims[1], l = dims[2], h = dims[3], p = dims[4], n = dims[5];
            if (int(index) >= batches * chunks * h * p * n) { return; }
            int r = int(index);
            const int e = r % n; r /= n;
            const int d = r % p; r /= p;
            const int head = r % h; r /= h;
            const int bc = r;
            nfk_product term = {bd + (bc * l * h + head) * n + e, x + (bc * l * h + head) * p + d, h * n, h * p};
            s[index] = nfk_outer_sum(l, (head * p + d) * n + e, h * p * n, term);
        """,
        header: torchSumHeader)

    // O[b, i, h, p, n] = Σⱼ D[b, j, i, h] · S[b, j, h, p, n].
    private static let ssdPassingKernel = MLXFast.metalKernel(
        name: "nfk_ssd_passing",
        inputNames: ["dc", "st", "dims"],
        outputNames: ["o"],
        source: """
            const uint index = thread_position_in_grid.x;
            const int batches = dims[0], k = dims[1], h = dims[2], p = dims[3], n = dims[4];
            if (int(index) >= batches * k * h * p * n) { return; }
            int r = int(index);
            const int e = r % n; r /= n;
            const int d = r % p; r /= p;
            const int head = r % h; r /= h;
            const int i = r % k; r /= k;
            const int b = r;
            nfk_product term = {dc + ((b * k) * k + i) * h + head, st + ((b * k * h + head) * p + d) * n + e,
                                k * h, h * p * n};
            o[index] = nfk_outer_sum(k, ((i * h + head) * p + d) * n + e, k * h * p * n, term);
        """,
        header: torchSumHeader)

    // R[b, c, t, h, p] = Σₙ C[b, c, t, h, n] · S[b, c, h, p, n].
    private static let ssdReadoutKernel = MLXFast.metalKernel(
        name: "nfk_ssd_readout",
        inputNames: ["cm", "st", "dims"],
        outputNames: ["o"],
        source: """
            const uint index = thread_position_in_grid.x;
            const int batches = dims[0], chunks = dims[1], l = dims[2], h = dims[3], p = dims[4], n = dims[5];
            if (int(index) >= batches * chunks * l * h * p) { return; }
            int r = int(index);
            const int d = r % p; r /= p;
            const int head = r % h; r /= h;
            const int i = r % l; r /= l;
            const int bc = r;
            nfk_product term = {cm + ((bc * l + i) * h + head) * n, st + ((bc * h + head) * p + d) * n, 1, 1};
            o[index] = nfk_inner_cascade(n, term);
        """,
        header: torchSumHeader)
}
