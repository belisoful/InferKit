//
//  NFKMLXStableFast3DReference.swift
//  InferKitMLX
//

import Darwin
import Foundation
import MLX

// MARK: - Mapped weights

/// A safetensors file of float32 tensors mapped into memory and read in place.
final class NFKSF3DMappedTensors {
    private struct Entry {
        let offset: Int
        let shape: [Int]
    }

    private let base: UnsafeMutableRawPointer
    private let length: Int
    private let entries: [String: Entry]

    init(url: URL) throws {
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else {
            throw NFKMLXError.unsupportedConfiguration("cannot open \(url.path)")
        }
        defer { close(descriptor) }
        var status = stat()
        guard fstat(descriptor, &status) == 0, status.st_size > 8 else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a safetensors file")
        }
        let length = Int(status.st_size)
        guard let mapped = mmap(nil, length, PROT_READ, MAP_PRIVATE, descriptor, 0), mapped != MAP_FAILED else {
            throw NFKMLXError.unsupportedConfiguration("cannot map \(url.path)")
        }
        let headerLength = Int(UInt64(littleEndian: mapped.loadUnaligned(as: UInt64.self)))
        let header = Data(bytes: mapped + 8, count: min(headerLength, length - 8))
        guard let object = try? JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            munmap(mapped, length)
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) has no safetensors header")
        }
        var entries = [String: Entry]()
        for (key, value) in object where key != "__metadata__" {
            guard let fields = value as? [String: Any], fields["dtype"] as? String == "F32",
                  let shape = fields["shape"] as? [Int], let offsets = fields["data_offsets"] as? [Int], offsets.count == 2,
                  (8 + headerLength + offsets[0]) % MemoryLayout<Float>.alignment == 0 else {
                continue
            }
            entries[key] = Entry(offset: 8 + headerLength + offsets[0], shape: shape)
        }
        self.base = mapped
        self.length = length
        self.entries = entries
    }

    deinit {
        munmap(base, length)
    }

    func contains(_ key: String) -> Bool { entries[key] != nil }

    func shape(_ key: String) throws -> [Int] {
        guard let entry = entries[key] else { throw NFKMLXError.unsupportedConfiguration("the checkpoint has no float32 \(key)") }
        return entry.shape
    }

    func floats(_ key: String) throws -> UnsafePointer<Float> {
        guard let entry = entries[key] else { throw NFKMLXError.unsupportedConfiguration("the checkpoint has no float32 \(key)") }
        return UnsafePointer((base + entry.offset).assumingMemoryBound(to: Float.self))
    }
}

// MARK: - Elementwise functions

/// The float32 elementwise functions torch 2.14.1 runs on Apple silicon (NEON, SLEEF): Sleef's u10
/// `expf`, `tanhf`, `sinf`, and `cosf`, the Arm optimized-routines `expf` torch calls `exp_u20`, and torch's
/// own vector `erf`. Each reproduces its source bit for bit.
enum NFKSF3DTorchMath {

    private static let ln2Upper: Float = 0.693145751953125
    private static let ln2Lower: Float = 1.428606765330187045e-06
    private static let inverseLn2: Float = 1.442695040888963407359924681001892137426645954152985934135449406931

    /// NEON's float-to-int32 conversion, which saturates.
    private static func int32(_ value: Float) -> Int32 {
        if value.isNaN { return 0 }
        if value >= 2147483648 { return .max }
        if value < -2147483648 { return .min }
        return Int32(value)
    }

    private static func pow2i(_ q: Int32) -> Float {
        Float(bitPattern: UInt32(bitPattern: (q &+ 0x7f) &<< 23))
    }

    private static func ldexp2(_ d: Float, _ e: Int32) -> Float {
        let half = e >> 1
        return d * pow2i(half) * pow2i(e &- half)
    }

    /// `Sleef_expf4_u10`.
    static func exp(_ d: Float) -> Float {
        let qf = (d * inverseLn2).rounded(.toNearestOrEven)
        let q = int32(qf)
        let u0 = Float(q)
        var s = d.addingProduct(u0, -ln2Upper)
        s = s.addingProduct(u0, -ln2Lower)
        var u: Float = 0.000198527617612853646278381
        u = Float(0.00139304355252534151077271).addingProduct(u, s)
        u = Float(0.00833336077630519866943359).addingProduct(u, s)
        u = Float(0.0416664853692054748535156).addingProduct(u, s)
        u = Float(0.166666671633720397949219).addingProduct(u, s)
        u = Float(0.5).addingProduct(u, s)
        u = 1 + s.addingProduct(s * s, u)
        u = ldexp2(u, q)
        if d < -104 { u = 0 }
        if 100 < d { u = .infinity }
        return u
    }

    private static func expU20Fast(_ x: Float) -> Float {
        let n = (x * 0x1.715476p+0).rounded(.toNearestOrAwayFromZero)
        var r = x.addingProduct(-n, 0x1.62e4p-1)
        r = r.addingProduct(-n, 0x1.7f7d1cp-20)
        let e = UInt32(bitPattern: Int32(n)) &<< 23
        let scale = Float(bitPattern: e &+ 0x3f800000)
        let r2 = r * r
        var p = Float(0x1.573e2ep-5).addingProduct(r, 0x1.0e4020p-7)
        var q = Float(0x1.fffdb6p-2).addingProduct(r, 0x1.555e66p-3)
        q = q.addingProduct(p, r2)
        p = Float(0x1.ffffecp-1) * r
        let poly = p.addingProduct(q, r2)
        return scale.addingProduct(poly, scale)
    }

    /// `Vectorized<float>::exp_u20` over one four-lane group: the group takes Sleef's `exp` when any lane's
    /// magnitude passes 0x1.5d5e2ap+6.
    static func expU20(_ group: inout (Float, Float, Float, Float)) {
        let bound: Float = 0x1.5d5e2ap+6
        if abs(group.0) > bound || abs(group.1) > bound || abs(group.2) > bound || abs(group.3) > bound {
            group = (exp(group.0), exp(group.1), exp(group.2), exp(group.3))
        } else {
            group = (expU20Fast(group.0), expU20Fast(group.1), expU20Fast(group.2), expU20Fast(group.3))
        }
    }

    /// `Vectorized<float>::erf` over one four-lane group.
    static func erf(_ x: (Float, Float, Float, Float)) -> (Float, Float, Float, Float) {
        var t = (Float(0), Float(0), Float(0), Float(0)), r = t, e = t
        func lane(_ value: Float, _ tOut: inout Float, _ rOut: inout Float, _ eOut: inout Float) {
            let a = abs(value)
            tOut = 1 / Float(1).addingProduct(0.3275911, a)
            var p = Float(-1.453152027).addingProduct(1.061405429, tOut)
            p = Float(1.421413741).addingProduct(p, tOut)
            p = Float(-0.284496736).addingProduct(p, tOut)
            rOut = Float(0.254829592).addingProduct(p, tOut)
            eOut = -(value * value)
        }
        lane(x.0, &t.0, &r.0, &e.0); lane(x.1, &t.1, &r.1, &e.1); lane(x.2, &t.2, &r.2, &e.2); lane(x.3, &t.3, &r.3, &e.3)
        expU20(&e)
        func finish(_ value: Float, _ tValue: Float, _ rValue: Float, _ eValue: Float) -> Float {
            let y = Float(1).addingProduct(tValue * -eValue, rValue)
            return value.sign == .minus ? -y : y
        }
        return (finish(x.0, t.0, r.0, e.0), finish(x.1, t.1, r.1, e.1), finish(x.2, t.2, r.2, e.2), finish(x.3, t.3, r.3, e.3))
    }

    // Sleef's double-float arithmetic, with the fused variants its Arm build selects.
    private typealias Pair = (hi: Float, lo: Float)

    private static func add2(_ x: Pair, _ y: Float) -> Pair {
        let s = x.hi + y, v = s - x.hi
        return (s, ((x.hi - (s - v)) + (y - v)) + x.lo)
    }
    private static func add2(_ x: Pair, _ y: Pair) -> Pair {
        let s = x.hi + y.hi, v = s - x.hi
        return (s, ((x.hi - (s - v)) + (y.hi - v)) + (x.lo + y.lo))
    }
    private static func add2(_ x: Float, _ y: Float) -> Pair {
        let s = x + y, v = s - x
        return (s, (x - (s - v)) + (y - v))
    }
    private static func add(_ x: Float, _ y: Pair) -> Pair {
        let s = x + y.hi
        return (s, ((x - s) + y.hi) + y.lo)
    }
    private static func add(_ x: Pair, _ y: Pair) -> Pair {
        let s = x.hi + y.hi
        return (s, (((x.hi - s) + y.hi) + x.lo) + y.lo)
    }
    private static func add(_ x: Pair, _ y: Float) -> Pair {
        let s = x.hi + y
        return (s, ((x.hi - s) + y) + x.lo)
    }
    private static func add(_ x: Float, _ y: Float) -> Pair {
        let s = x + y
        return (s, (x - s) + y)
    }
    private static func mul(_ x: Pair, _ y: Float) -> Pair {
        let s = x.hi * y
        return (s, (-s).addingProduct(x.hi, y).addingProduct(x.lo, y))
    }
    private static func mul(_ x: Pair, _ y: Pair) -> Pair {
        let s = x.hi * y.hi
        return (s, (-s).addingProduct(x.hi, y.hi).addingProduct(x.lo, y.hi).addingProduct(x.hi, y.lo))
    }
    private static func mulToFloat(_ x: Pair, _ y: Pair) -> Float {
        (x.hi * y.lo).addingProduct(x.lo, y.hi).addingProduct(x.hi, y.hi)
    }
    private static func square(_ x: Pair) -> Pair {
        let s = x.hi * x.hi
        return (s, (-s).addingProduct(x.hi, x.hi).addingProduct(x.hi + x.hi, x.lo))
    }
    private static func reciprocal(_ d: Pair) -> Pair {
        let s = 1 / d.hi
        return (s, s * Float(1).addingProduct(-d.hi, s).addingProduct(-d.lo, s))
    }
    private static func divide(_ n: Pair, _ d: Pair) -> Pair {
        let t = 1 / d.hi
        let s = n.hi * t
        let u = (-s).addingProduct(t, n.hi)
        let v = Float(1).addingProduct(-d.hi, t).addingProduct(-d.lo, t)
        return (s, u.addingProduct(n.lo, t).addingProduct(s, v))
    }

    private static func expk2(_ d: Pair) -> Pair {
        let q = int32(((d.hi + d.lo) * inverseLn2).rounded(.toNearestOrEven))
        let qf = Float(q)
        var s = add2(d, qf * -ln2Upper)
        s = add2(s, qf * -ln2Lower)
        var u: Float = 0.1980960224e-3
        u = Float(0.1394256484e-2).addingProduct(u, s.hi)
        u = Float(0.8333456703e-2).addingProduct(u, s.hi)
        u = Float(0.4166637361e-1).addingProduct(u, s.hi)
        var t = add2(mul(s, u), 0.166666659414234244790680580464e+0)
        t = add2(mul(s, t), 0.5)
        t = add2(s, mul(square(s), t))
        t = add(Float(1), t)
        var result = (hi: ldexp2(t.hi, q), lo: ldexp2(t.lo, q))
        if d.hi < -104 { result = (0, 0) }
        return result
    }

    /// `Sleef_tanhf4_u10`.
    static func tanh(_ x: Float) -> Float {
        let y = abs(x)
        let d = expk2((y, 0))
        let e = reciprocal(d)
        let quotient = divide(add(d, (-e.hi, -e.lo)), add(d, e))
        var result = quotient.hi + quotient.lo
        if abs(x) > 8.664339742 || result.isNaN { result = 1 }
        result = x.sign == .minus ? -abs(result) : abs(result)
        return x.isNaN ? x : result
    }

    private static let inversePi = Float(0.318309886183790671537767526745028724)
    private static let piA2: Float = 3.1414794921875
    private static let piB2: Float = 0.00011315941810607910156
    private static let piC2: Float = 1.9841872589410058936e-09

    private static func trigonometricTail(_ reduced: Pair) -> Float {
        let t = reduced
        let s = square(reduced)
        var u: Float = 2.6083159809786593541503e-06
        u = Float(-0.0001981069071916863322258).addingProduct(u, s.hi)
        u = Float(0.00833307858556509017944336).addingProduct(u, s.hi)
        let x = add(Float(1), mul(add(Float(-0.166666597127914428710938), u * s.hi), s))
        return mulToFloat(t, x)
    }

    /// `Sleef_sinf4_u10` for arguments below 125 in magnitude.
    static func sin(_ d: Float) -> Float {
        precondition(abs(d) < 125)
        let u = (d * inversePi).rounded(.toNearestOrEven)
        let q = Int32(u)
        let v = d.addingProduct(u, -piA2)
        var s = add2(v, u * -piB2)
        s = add(s, u * -piC2)
        var result = trigonometricTail(s)
        if q & 1 == 1 { result = -result }
        return d == 0 && d.sign == .minus ? d : result
    }

    /// `Sleef_cosf4_u10` for arguments below 125 in magnitude.
    static func cos(_ d: Float) -> Float {
        precondition(abs(d) < 125)
        let dq = Float(1).addingProduct(Float(-0.5).addingProduct(d, inversePi).rounded(.toNearestOrEven), 2)
        let q = Int32(dq)
        var s = add2(d, dq * (-piA2 * 0.5))
        s = add2(s, dq * (-piB2 * 0.5))
        s = add2(s, dq * (-piC2 * 0.5))
        let result = trigonometricTail(s)
        return q & 2 == 0 ? -result : result
    }

    /// `cpu_kernel_vec` over `count` contiguous floats: eight-float groups through the vector form, the rest
    /// through the scalar form.
    static func map(_ values: inout [Float], vector: (Float) -> Float, scalar: (Float) -> Float) {
        let grouped = values.count / 8 * 8
        values.withUnsafeMutableBufferPointer { buffer in
            DispatchQueue.concurrentPerform(iterations: max(1, (grouped + 65535) / 65536)) { chunk in
                let end = min(grouped, (chunk + 1) * 65536)
                var index = chunk * 65536
                while index < end {
                    buffer[index] = vector(buffer[index])
                    index += 1
                }
            }
            for index in grouped ..< buffer.count {
                buffer[index] = scalar(buffer[index])
            }
        }
    }

    /// `F.gelu` (the exact form).
    static func gelu(_ values: inout [Float]) {
        let alpha = Float(0.70710678118654752440)
        let vectorEnd = values.count / 8 * 8
        values.withUnsafeMutableBufferPointer { buffer in
            DispatchQueue.concurrentPerform(iterations: max(1, (vectorEnd + 65535) / 65536)) { chunk in
                var index = chunk * 65536
                let end = min(vectorEnd, (chunk + 1) * 65536)
                while index < end {
                    let x = (buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3])
                    let e = erf((x.0 * alpha, x.1 * alpha, x.2 * alpha, x.3 * alpha))
                    buffer[index] = x.0 * 0.5 * (1 + e.0)
                    buffer[index + 1] = x.1 * 0.5 * (1 + e.1)
                    buffer[index + 2] = x.2 * 0.5 * (1 + e.2)
                    buffer[index + 3] = x.3 * 0.5 * (1 + e.3)
                    index += 4
                }
            }
            for index in vectorEnd ..< buffer.count {
                let x = buffer[index]
                buffer[index] = x * 0.5 * (1 + erff(x * alpha))
            }
        }
    }

    /// `F.silu`.
    static func silu(_ values: inout [Float]) {
        map(&values, vector: { $0 / (1 + exp(-$0)) }, scalar: { $0 / (1 + expf(-$0)) })
    }

    /// `torch.sigmoid`.
    static func sigmoid(_ values: inout [Float]) {
        map(&values, vector: { 1 / (1 + exp(-$0)) }, scalar: { 1 / (1 + expf(-$0)) })
    }
}

// MARK: - Kernels

/// The torch 2.14.1 CPU float32 kernels the Stable Fast 3D reference runs on Apple silicon, each with its
/// operation order, its fused multiply-adds, and its BLAS call: Accelerate's `sgemm_` with the transposes and
/// leading dimensions `addmm_impl_cpu_` and `cpu_flash_attention` pass it, the `RowwiseMoments` Welford sums of
/// LayerNorm and GroupNorm, the non-separable bicubic resize, and the bilinear `grid_sample`.
struct NFKSF3DTorchKernels {
    typealias SGEMM = @convention(c) (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<Int32>, UnsafePointer<Int32>,
                                      UnsafePointer<Int32>, UnsafePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>,
                                      UnsafePointer<Float>, UnsafePointer<Int32>, UnsafePointer<Float>, UnsafeMutablePointer<Float>,
                                      UnsafePointer<Int32>) -> Void

    private let sgemmFunction: SGEMM

    init() throws {
        sgemmFunction = try NFKSF3DLAPACK.symbol("sgemm_", as: SGEMM.self)
    }

    /// A strided float32 matrix.
    struct Matrix {
        let base: UnsafePointer<Float>
        let rows: Int
        let columns: Int
        let rowStride: Int
        let columnStride: Int

        var sizes: [Int] { [rows, columns] }
        var strides: [Int] { [rowStride, columnStride] }

        static func rowMajor(_ base: UnsafePointer<Float>, rows: Int, columns: Int) -> Matrix {
            Matrix(base: base, rows: rows, columns: columns, rowStride: columns, columnStride: 1)
        }

        /// `x.contiguous()` as torch judges it: a size-one dimension's stride does not count.
        var isContiguous: Bool {
            (columns == 1 || columnStride == 1) && (rows == 1 || rowStride == columns)
        }
    }

    /// `cpublas::gemm` for float: `normalize_last_dims`, then BLAS `sgemm_` (column-major).
    func gemm(transposeA: Bool, transposeB: Bool, m: Int, n: Int, k: Int, a: UnsafePointer<Float>, lda: Int,
              b: UnsafePointer<Float>, ldb: Int, beta: Float, c: UnsafeMutablePointer<Float>, ldc: Int) {
        var lda = lda, ldb = ldb, ldc = ldc
        if n == 1 { ldc = m }
        if transposeA {
            if m == 1 { lda = k }
        } else if k == 1 {
            lda = m
        }
        if transposeB {
            if k == 1 { ldb = n }
        } else if n == 1 {
            ldb = k
        }
        var ta: CChar = transposeA ? 84 : 78, tb: CChar = transposeB ? 84 : 78
        var m32 = Int32(m), n32 = Int32(n), k32 = Int32(k), lda32 = Int32(lda), ldb32 = Int32(ldb), ldc32 = Int32(ldc)
        var alpha: Float = 1, beta = beta
        let aExtent = lda * ((transposeA ? m : k) - 1) + (transposeA ? k : m)
        let bExtent = ldb * ((transposeB ? k : n) - 1) + (transposeB ? n : k)
        Self.lineAligned(a, count: aExtent) { a in
            Self.lineAligned(b, count: bExtent) { b in
                sgemmFunction(&ta, &tb, &m32, &n32, &k32, &alpha, a, &lda32, b, &ldb32, &beta, c, &ldc32)
            }
        }
    }

    /// The cache line torch's operands start on.
    static let lineBytes = 128

    /// Accelerate's products with a dimension of one read their matrix in an order that depends on its
    /// address modulo the 128-byte cache line. Every operand torch passes it here starts on a line (each
    /// tensor's storage is page-aligned, and each view's offset is a multiple of 256 bytes), so an operand
    /// that does not is copied to one that does.
    static func lineAligned(_ pointer: UnsafePointer<Float>, count: Int, _ body: (UnsafePointer<Float>) -> Void) {
        guard Int(bitPattern: pointer) % lineBytes != 0 else {
            body(pointer)
            return
        }
        let copy = UnsafeMutablePointer<Float>(OpaquePointer(UnsafeMutableRawPointer.allocate(byteCount: count * 4, alignment: lineBytes)))
        defer { copy.deallocate() }
        copy.update(from: pointer, count: count)
        body(copy)
    }

    /// A line-aligned copy of `values`, which the caller deallocates.
    static func lineAlignedCopy(_ values: [Float]) -> UnsafeMutableBufferPointer<Float> {
        let copy = UnsafeMutablePointer<Float>(OpaquePointer(UnsafeMutableRawPointer.allocate(byteCount: max(1, values.count) * 4,
                                                                                              alignment: lineBytes)))
        values.withUnsafeBufferPointer { copy.update(from: $0.baseAddress!, count: values.count) }
        return UnsafeMutableBufferPointer(start: copy, count: values.count)
    }

    /// `addmm_impl_cpu_` into a contiguous row-major `[m1.rows, m2.columns]` result that already holds
    /// `self` when `beta` is 1.
    func addmm(into result: UnsafeMutablePointer<Float>, _ first: Matrix, _ second: Matrix, beta: Float) {
        let resultSizes = [first.rows, second.columns], resultStrides = [second.columns, 1]
        var m1 = first, m2 = second
        let transposeC: Bool
        if resultStrides[0] == 1 && (resultSizes[1] == 1 || resultStrides[1] >= max(1, resultSizes[0])) {
            transposeC = false
        } else {
            swap(&m1, &m2)
            transposeC = true
        }
        let m = resultSizes[transposeC ? 1 : 0], n = resultSizes[transposeC ? 0 : 1], k = m1.sizes[transposeC ? 0 : 1]
        let near = transposeC ? 1 : 0, far = transposeC ? 0 : 1
        var copies = [[Float]]()
        func contiguousCopy(_ x: Matrix) -> Matrix {
            var values = [Float](repeating: 0, count: x.rows * x.columns)
            for row in 0 ..< x.rows {
                for column in 0 ..< x.columns { values[row * x.columns + column] = x.base[row * x.rowStride + column * x.columnStride] }
            }
            copies.append(values)
            return copies.last!.withUnsafeBufferPointer { .rowMajor($0.baseAddress!, rows: x.rows, columns: x.columns) }
        }
        let a: Matrix, transposeA: Bool
        if m1.strides[near] == 1 && m1.strides[far] >= max(1, m) {
            a = m1; transposeA = false
        } else if m1.strides[far] == 1 && m1.strides[near] >= max(1, k) {
            a = m1; transposeA = true
        } else {
            a = contiguousCopy(m1); transposeA = !transposeC
        }
        let b: Matrix, transposeB: Bool
        if m2.strides[near] == 1 && m2.strides[far] >= max(1, k) {
            b = m2; transposeB = false
        } else if m2.strides[far] == 1 && m2.strides[near] >= max(1, n) {
            b = m2; transposeB = true
        } else {
            b = contiguousCopy(m2); transposeB = !transposeC
        }
        let lda = a.strides[transposeA == transposeC ? 1 : 0]
        let ldb = b.strides[transposeB == transposeC ? 1 : 0]
        let ldc = resultStrides[transposeC ? 0 : 1]
        withExtendedLifetime(copies) {
            gemm(transposeA: transposeA, transposeB: transposeB, m: m, n: n, k: k, a: a.base, lda: lda,
                 b: b.base, ldb: ldb, beta: beta, c: result, ldc: ldc)
        }
    }

    /// `F.linear(x, weight, bias)` for `x` `[rows, inputs]` and a `[outputs, inputs]` weight: `addmm` onto the
    /// bias for a two-dimensional input or a contiguous one, `matmul` then `add_` for a strided view of a
    /// batch-of-one tensor.
    func linear(_ x: Matrix, weight: UnsafePointer<Float>, outputs: Int, bias: UnsafePointer<Float>?,
                twoDimensional: Bool = false) -> [Float] {
        let transposedWeight = Matrix(base: weight, rows: x.columns, columns: outputs, rowStride: 1, columnStride: x.columns)
        var out = [Float](repeating: 0, count: x.rows * outputs)
        out.withUnsafeMutableBufferPointer { buffer in
            let o = buffer.baseAddress!
            if let bias, x.isContiguous || twoDimensional {
                for row in 0 ..< x.rows { (o + row * outputs).update(from: bias, count: outputs) }
                addmm(into: o, x, transposedWeight, beta: 1)
            } else {
                addmm(into: o, x, transposedWeight, beta: 0)
                if let bias {
                    for row in 0 ..< x.rows {
                        for column in 0 ..< outputs { o[row * outputs + column] = o[row * outputs + column] + bias[column] }
                    }
                }
            }
        }
        return out
    }

    func linear(_ x: [Float], rows: Int, weight: UnsafePointer<Float>, outputs: Int, bias: UnsafePointer<Float>?) -> [Float] {
        x.withUnsafeBufferPointer {
            linear(.rowMajor($0.baseAddress!, rows: rows, columns: x.count / rows), weight: weight, outputs: outputs, bias: bias)
        }
    }

    // MARK: Normalization

    private typealias Lanes = (Float, Float, Float, Float)

    /// `AddMomentsVec`.
    private static func addMoments(_ countAdd: Int, _ meanAdd: [Float], _ squaresAdd: [Float],
                                   _ count: inout Int, _ mean: inout [Float], _ squares: inout [Float]) {
        let total = count + countAdd
        let c: Float = total == 0 ? 0 : Float(countAdd) / Float(total)
        for lane in 0 ..< 4 {
            let delta = meanAdd[lane] - mean[lane]
            let squaresSum = squares[lane] + squaresAdd[lane]
            let cDelta = c * delta
            let countDelta = delta * Float(count)
            mean[lane] = mean[lane] + cDelta
            squares[lane] = squaresSum.addingProduct(countDelta, cDelta)
        }
        count = total
    }

    /// `RowwiseMoments` of `count` contiguous floats: the mean and the biased variance.
    static func rowwiseMoments(_ x: UnsafePointer<Float>, count total: Int) -> (mean: Float, variance: Float) {
        let n = total / 4
        let chunks = (n + 15) / 16
        var depth = 0
        while (1 << depth) < chunks { depth += 1 }
        var counts = [Int](repeating: 0, count: max(depth, 1))
        var means = [[Float]](repeating: [0, 0, 0, 0], count: max(depth, 1))
        var squares = means
        let reciprocals = (0 ..< 16).map { 1 / Float($0 + 1) }
        for chunk in 0 ..< chunks {
            let size = min(16, n - chunk * 16)
            var mean: [Float] = [0, 0, 0, 0], square: [Float] = [0, 0, 0, 0]
            for j in 0 ..< size {
                let row = x + (chunk * 16 + j) * 4
                for lane in 0 ..< 4 {
                    let delta = row[lane] - mean[lane]
                    mean[lane] = mean[lane].addingProduct(reciprocals[j], delta)
                    square[lane] = square[lane].addingProduct(delta, row[lane] - mean[lane])
                }
            }
            addMoments(size, mean, square, &counts[0], &means[0], &squares[0])
            var mask = chunk + 1, level = 1
            while level < depth && mask & 1 == 0 {
                addMoments(counts[level - 1], means[level - 1], squares[level - 1], &counts[level], &means[level], &squares[level])
                counts[level - 1] = 0; means[level - 1] = [0, 0, 0, 0]; squares[level - 1] = [0, 0, 0, 0]
                mask >>= 1
                level += 1
            }
        }
        for level in stride(from: 1, to: depth, by: 1) {
            addMoments(counts[level], means[level], squares[level], &counts[0], &means[0], &squares[0])
        }
        var count = 0, mean: Float = 0, square: Float = 0
        for index in (n * 4) ..< total {
            let delta = x[index] - mean
            count += 1
            mean = mean + delta / Float(count)
            square = square.addingProduct(delta, x[index] - mean)
        }
        for lane in 0 ..< 4 {
            let sum = count + n
            let c: Float = sum == 0 ? 0 : Float(n) / Float(sum)
            let delta = means[0][lane] - mean
            mean = mean.addingProduct(c, delta)
            square = square + squares[0][lane].addingProduct(delta * delta * c, Float(count))
            count = sum
        }
        return (mean, square / Float(total))
    }

    /// `F.layer_norm` over the last axis of `[rows, width]`.
    func layerNorm(_ x: [Float], width: Int, weight: UnsafePointer<Float>, bias: UnsafePointer<Float>, epsilon: Float) -> [Float] {
        let rows = x.count / width
        var out = [Float](repeating: 0, count: x.count)
        x.withUnsafeBufferPointer { input in
            out.withUnsafeMutableBufferPointer { output in
                DispatchQueue.concurrentPerform(iterations: rows) { row in
                    let source = input.baseAddress! + row * width
                    let (mean, variance) = Self.rowwiseMoments(source, count: width)
                    let rstd = 1 / (max(variance, 0) + epsilon).squareRoot()
                    let target = output.baseAddress! + row * width
                    for index in 0 ..< width {
                        target[index] = (source[index] + -mean) * rstd * weight[index] + bias[index]
                    }
                }
            }
        }
        return out
    }

    /// `F.group_norm` of a contiguous `[channels, spatial]` tensor.
    func groupNorm(_ x: [Float], channels: Int, groups: Int, weight: UnsafePointer<Float>, bias: UnsafePointer<Float>,
                   epsilon: Double) -> [Float] {
        let spatial = x.count / channels, perGroup = channels / groups
        var out = [Float](repeating: 0, count: x.count)
        x.withUnsafeBufferPointer { input in
            out.withUnsafeMutableBufferPointer { output in
                DispatchQueue.concurrentPerform(iterations: groups) { group in
                    let start = group * perGroup * spatial
                    let (mean, variance) = Self.rowwiseMoments(input.baseAddress! + start, count: perGroup * spatial)
                    let rstd = Float(1 / (Double(max(variance, 0)) + epsilon).squareRoot())
                    for local in 0 ..< perGroup {
                        let channel = group * perGroup + local
                        let scale = rstd * weight[channel]
                        let shift = bias[channel].addingProduct(-scale, mean)
                        let offset = start + local * spatial
                        for index in offset ..< offset + spatial {
                            output[index] = shift.addingProduct(scale, input[index])
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: Convolution

    /// `slow_conv2d` of one `[channels, height, width]` frame: `im2col` with rows `(channel, ki, kj)`, then
    /// `sgemm('N', 'N')` onto the bias.
    func conv2d(_ x: UnsafePointer<Float>, channels: Int, height: Int, width: Int, weight: UnsafePointer<Float>,
                outputs: Int, kernel: Int, stride: Int, padding: Int, bias: UnsafePointer<Float>) -> [Float] {
        let outHeight = (height + 2 * padding - kernel) / stride + 1, outWidth = (width + 2 * padding - kernel) / stride + 1
        let pixels = outHeight * outWidth, depth = channels * kernel * kernel
        var columns = [Float](repeating: 0, count: depth * pixels)
        columns.withUnsafeMutableBufferPointer { buffer in
            DispatchQueue.concurrentPerform(iterations: channels) { channel in
                for ki in 0 ..< kernel {
                    for kj in 0 ..< kernel {
                        let row = buffer.baseAddress! + ((channel * kernel + ki) * kernel + kj) * pixels
                        for oy in 0 ..< outHeight {
                            let y = oy * stride + ki - padding
                            guard y >= 0 && y < height else { continue }
                            for ox in 0 ..< outWidth {
                                let xIndex = ox * stride + kj - padding
                                if xIndex >= 0 && xIndex < width {
                                    row[oy * outWidth + ox] = x[(channel * height + y) * width + xIndex]
                                }
                            }
                        }
                    }
                }
            }
        }
        var out = [Float](repeating: 0, count: outputs * pixels)
        out.withUnsafeMutableBufferPointer { buffer in
            for channel in 0 ..< outputs {
                (buffer.baseAddress! + channel * pixels).update(repeating: bias[channel], count: pixels)
            }
            columns.withUnsafeBufferPointer { source in
                gemm(transposeA: false, transposeB: false, m: pixels, n: outputs, k: depth, a: source.baseAddress!, lda: pixels,
                     b: weight, ldb: depth, beta: 1, c: buffer.baseAddress!, ldc: pixels)
            }
        }
        return out
    }

    // MARK: Attention

    /// `F.scaled_dot_product_attention` as `cpu_flash_attention` runs it in float32 without a mask, on
    /// projections laid out `[tokens, heads · width]` and read in place. Returns `[queries, heads · width]`.
    func attention(q: [Float], k: [Float], v: [Float], queries: Int, keys: Int, heads: Int, scale: Float) -> [Float] {
        let rowWidth = q.count / queries, width = rowWidth / heads
        let qSplit = min(queries >= 768 ? 256 : queries >= 192 ? 64 : 32, queries)
        let kvSplit = min(512, keys)
        let qSlices = (queries + qSplit - 1) / qSplit
        var out = [Float](repeating: 0, count: queries * rowWidth)
        let qBuffer = Self.lineAlignedCopy(q), kBuffer = Self.lineAlignedCopy(k), vBuffer = Self.lineAlignedCopy(v)
        defer {
            qBuffer.deallocate(); kBuffer.deallocate(); vBuffer.deallocate()
        }
        do {
            do {
                do {
                    out.withUnsafeMutableBufferPointer { outBuffer in
                        DispatchQueue.concurrentPerform(iterations: heads * qSlices) { task in
                            let head = task / qSlices, start = (task % qSlices) * qSplit
                            let block = min(qSplit, queries - start)
                            var rowMax = [Float](repeating: -.infinity, count: block)
                            var rowSum = [Float](repeating: 0, count: block)
                            var accumulated = [Float](repeating: 0, count: block * width)
                            let scores = Self.lineAlignedCopy([Float](repeating: 0, count: block * kvSplit))
                            defer { scores.deallocate() }
                            var keyStart = 0
                            while keyStart < keys {
                                let keyBlock = min(kvSplit, keys - keyStart)
                                do {
                                    let qk = scores
                                    gemm(transposeA: true, transposeB: false, m: keyBlock, n: block, k: width,
                                         a: kBuffer.baseAddress! + keyStart * rowWidth + head * width, lda: rowWidth,
                                         b: qBuffer.baseAddress! + start * rowWidth + head * width, ldb: rowWidth,
                                         beta: 0, c: qk.baseAddress!, ldc: keyBlock)
                                    for row in 0 ..< block {
                                        let scoreRow = qk.baseAddress! + row * keyBlock
                                        var localMax = -Float.infinity
                                        for index in 0 ..< keyBlock {
                                            scoreRow[index] = scoreRow[index] * scale
                                            localMax = max(localMax, scoreRow[index])
                                        }
                                        let newMax = rowMax[row] > localMax ? rowMax[row] : localMax
                                        let sum = Self.exponentiate(scoreRow, count: keyBlock, maximum: newMax)
                                        let correction = expf(rowMax[row] - newMax)
                                        rowSum[row] = sum.addingProduct(correction, rowSum[row])
                                        rowMax[row] = newMax
                                        if keyStart > 0 {
                                            for index in 0 ..< width { accumulated[row * width + index] = accumulated[row * width + index] * correction }
                                        }
                                    }
                                    accumulated.withUnsafeMutableBufferPointer { destination in
                                        gemm(transposeA: false, transposeB: false, m: width, n: block, k: keyBlock,
                                             a: vBuffer.baseAddress! + keyStart * rowWidth + head * width, lda: rowWidth,
                                             b: qk.baseAddress!, ldb: keyBlock, beta: keyStart == 0 ? 0 : 1,
                                             c: destination.baseAddress!, ldc: width)
                                    }
                                }
                                keyStart += keyBlock
                            }
                            for row in 0 ..< block {
                                let reciprocal = 1 / (rowSum[row] == 0 ? 1 : rowSum[row])
                                let target = outBuffer.baseAddress! + (start + row) * rowWidth + head * width
                                for index in 0 ..< width { target[index] = accumulated[row * width + index] * reciprocal }
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// `_exp_reduce_sum_fusion_kernel`: each score replaced by `exp(score − maximum)` and their sum, four
    /// lanes summed apart, folded as `(l0 + l2) + (l1 + l3)`, then the tail added in order.
    private static func exponentiate(_ row: UnsafeMutablePointer<Float>, count: Int, maximum: Float) -> Float {
        let vectorEnd = count / 4 * 4
        var lanes: Lanes = (0, 0, 0, 0)
        var index = 0
        while index < vectorEnd {
            var group: Lanes = (row[index] - maximum, row[index + 1] - maximum, row[index + 2] - maximum, row[index + 3] - maximum)
            NFKSF3DTorchMath.expU20(&group)
            row[index] = group.0; row[index + 1] = group.1; row[index + 2] = group.2; row[index + 3] = group.3
            lanes = (lanes.0 + group.0, lanes.1 + group.1, lanes.2 + group.2, lanes.3 + group.3)
            index += 4
        }
        var total = (lanes.0 + lanes.2) + (lanes.1 + lanes.3)
        while index < count {
            row[index] = expf(row[index] - maximum)
            total += row[index]
            index += 1
        }
        return total
    }

    // MARK: Resampling

    /// `F.interpolate(mode: "bicubic", align_corners: false, scale_factor:)` of `[channels, height, width]`
    /// (`upsample_non_separable_Nd_kernel_impl`, A = −0.75).
    static func bicubic(_ x: [Float], channels: Int, height: Int, width: Int, outHeight: Int, outWidth: Int,
                        scaleFactors: (Double, Double)) -> [Float] {
        func taps(input: Int, output: Int, factor: Double) -> (indices: [[Int]], weights: [[Float]]) {
            let scale = Float(1.0 / factor)
            let a: Float = -0.75
            // `cubic_convolution1` and `cubic_convolution2` as clang contracts them.
            func near(_ x: Float) -> Float { Float(1).addingProduct((-(a + 3)).addingProduct(a + 2, x) * x, x) }
            func far(_ x: Float) -> Float { (-4 * a).addingProduct((8 * a).addingProduct((-5 * a).addingProduct(a, x), x), x) }
            var indices = [[Int]](), weights = [[Float]]()
            for index in 0 ..< output {
                let real = Float(-0.5).addingProduct(scale, Float(index) + 0.5)
                let floored = min(Int(floorf(real)), input - 1)
                let t = min(max(real - Float(floored), 0), 1)
                let x2 = 1 - t
                weights.append([far(t + 1), near(t), near(x2), far(x2 + 1)])
                indices.append((0 ..< 4).map { min(max(floored + $0 - 1, 0), input - 1) })
            }
            return (indices, weights)
        }
        let rows = taps(input: height, output: outHeight, factor: scaleFactors.0)
        let columns = taps(input: width, output: outWidth, factor: scaleFactors.1)
        var out = [Float](repeating: 0, count: channels * outHeight * outWidth)
        for channel in 0 ..< channels {
            let plane = channel * height * width
            for oy in 0 ..< outHeight {
                for ox in 0 ..< outWidth {
                    func line(_ tap: Int) -> Float {
                        let base = plane + rows.indices[oy][tap] * width
                        var sum = x[base + columns.indices[ox][0]] * columns.weights[ox][0]
                        for j in 1 ..< 4 { sum = sum.addingProduct(x[base + columns.indices[ox][j]], columns.weights[ox][j]) }
                        return sum
                    }
                    var value = line(0) * rows.weights[oy][0]
                    for j in 1 ..< 4 { value = value.addingProduct(line(j), rows.weights[oy][j]) }
                    out[(channel * outHeight + oy) * outWidth + ox] = value
                }
            }
        }
        return out
    }

    /// `F.grid_sample(mode: "bilinear", padding_mode: "zeros", align_corners: true)` of one plane `[channels,
    /// side, side]` at `count` points `(x, y)`, written channel-major into `out` `[channels, count]`.
    static func gridSample(_ plane: UnsafePointer<Float>, channels: Int, side: Int, x: [Float], y: [Float],
                           into out: UnsafeMutablePointer<Float>) {
        let count = x.count
        let scale = Float(side - 1) / 2
        DispatchQueue.concurrentPerform(iterations: (count + 4095) / 4096) { chunk in
            for point in (chunk * 4096) ..< min(count, (chunk + 1) * 4096) {
                let px = (x[point] + 1) * scale, py = (y[point] + 1) * scale
                let west = floorf(px), north = floorf(py)
                let w = px - west, e = 1 - w, n = py - north, s = 1 - n
                let nw = s * e, ne = s * w, sw = n * e, se = n * w
                let ix = Int(west), iy = Int(north)
                let westInside = ix > -1 && ix < side, eastInside = ix + 1 > -1 && ix + 1 < side
                let northInside = iy > -1 && iy < side, southInside = iy + 1 > -1 && iy + 1 < side
                for channel in 0 ..< channels {
                    let base = plane + channel * side * side
                    let vnw: Float = westInside && northInside ? base[iy * side + ix] : 0
                    let vne: Float = eastInside && northInside ? base[iy * side + ix + 1] : 0
                    let vsw: Float = westInside && southInside ? base[(iy + 1) * side + ix] : 0
                    let vse: Float = eastInside && southInside ? base[(iy + 1) * side + ix + 1] : 0
                    out[channel * count + point] = vnw * nw + vne * ne + vsw * sw + vse * se
                }
            }
        }
    }
}

// MARK: - Network

/// Stable Fast 3D's image-to-scene-code network and its field decoder on the CPU in float32, each operation
/// as torch 2.14.1 runs it on Apple silicon, so a run reproduces the reference's scene code and decoded fields
/// bit for bit (`NFKMLXStableFast3DParameterKey.referenceArithmetic`).
final class NFKMLXStableFast3DReferenceNetwork: @unchecked Sendable {
    let configuration: NFKMLXStableFast3DConfiguration
    private let weights: NFKSF3DMappedTensors
    private let kernels: NFKSF3DTorchKernels

    init(weightsURL: URL, configuration: NFKMLXStableFast3DConfiguration = .released) throws {
        self.configuration = configuration
        weights = try NFKSF3DMappedTensors(url: weightsURL)
        kernels = try NFKSF3DTorchKernels()
    }

    private func w(_ key: String) throws -> UnsafePointer<Float> { try weights.floats(key) }

    private func linear(_ x: [Float], rows: Int, _ prefix: String, bias: Bool = true) throws -> [Float] {
        let shape = try weights.shape(prefix + ".weight")
        return kernels.linear(x, rows: rows, weight: try w(prefix + ".weight"), outputs: shape[0],
                              bias: bias ? try w(prefix + ".bias") : nil)
    }

    private func layerNorm(_ x: [Float], _ prefix: String, epsilon: Float = 1e-5) throws -> [Float] {
        let width = try weights.shape(prefix + ".weight")[0]
        return kernels.layerNorm(x, width: width, weight: try w(prefix + ".weight"), bias: try w(prefix + ".bias"), epsilon: epsilon)
    }

    private static func add(_ a: [Float], _ b: [Float]) -> [Float] {
        var out = a
        for index in out.indices { out[index] = a[index] + b[index] }
        return out
    }

    // MARK: Camera and image tokens

    /// The camera embedding `[modulationWidth]` from `cameraCondition`'s 25 floats.
    func cameraEmbedding(condition: [Float]) throws -> [Float] {
        try linear(condition, rows: 1, "camera_embedder.linear")
    }

    /// The DINOv2 position embeddings resized to the conditioning image's patch grid, `[grid² + 1, dinoWidth]`.
    func resizedPositions() throws -> [Float] {
        let key = "image_tokenizer.model.embeddings.position_embeddings"
        let shape = try weights.shape(key)
        let count = shape[1] - 1, width = shape[2]
        let grid = Int(Double(count).squareRoot())
        let target = Double(configuration.imageSize / configuration.patchSize) + 0.1
        let source = try w(key)
        var planes = [Float](repeating: 0, count: count * width)
        for token in 0 ..< count {
            for channel in 0 ..< width { planes[channel * count + token] = source[(token + 1) * width + channel] }
        }
        let side = Int(target)
        let factor = target / Double(count).squareRoot()
        let resized = NFKSF3DTorchKernels.bicubic(planes, channels: width, height: grid, width: grid, outHeight: side, outWidth: side,
                                                  scaleFactors: (factor, factor))
        var out = Array(UnsafeBufferPointer(start: source, count: width))
        out.reserveCapacity((side * side + 1) * width)
        for token in 0 ..< side * side {
            for channel in 0 ..< width { out.append(resized[channel * side * side + token]) }
        }
        return out
    }

    /// The DINOv2 tokenizer's last hidden state `[grid² + 1, dinoWidth]` for a conditioning image
    /// `[imageSize, imageSize, 3]` in 0...1 and the camera embedding.
    func imageTokens(_ color: [Float], camera: [Float]) throws -> [Float] {
        let prefix = "image_tokenizer.model."
        let side = configuration.imageSize
        let mean: [Float] = [0.485, 0.456, 0.406], deviation: [Float] = [0.229, 0.224, 0.225]
        var image = [Float](repeating: 0, count: 3 * side * side)
        for pixel in 0 ..< side * side {
            for channel in 0 ..< 3 { image[channel * side * side + pixel] = (color[pixel * 3 + channel] - mean[channel]) / deviation[channel] }
        }
        let projection = prefix + "embeddings.patch_embeddings.projection"
        let projectionShape = try weights.shape(projection + ".weight")
        let width = projectionShape[0], patch = projectionShape[2]
        let projectionWeight = try w(projection + ".weight"), projectionBias = try w(projection + ".bias")
        let patches = image.withUnsafeBufferPointer {
            kernels.conv2d($0.baseAddress!, channels: 3, height: side, width: side, weight: projectionWeight,
                           outputs: width, kernel: patch, stride: patch, padding: 0, bias: projectionBias)
        }
        let grid = side / patch, tokens = grid * grid + 1
        var x = Array(UnsafeBufferPointer(start: try w(prefix + "embeddings.cls_token"), count: width))
        x.reserveCapacity(tokens * width)
        for token in 0 ..< grid * grid {
            for channel in 0 ..< width { x.append(patches[channel * grid * grid + token]) }
        }
        x = Self.add(x, try resizedPositions())
        var condition = camera
        NFKSF3DTorchMath.silu(&condition)
        let heads = configuration.dinoHeads
        var layer = 0
        while weights.contains(prefix + "encoder.layer.\(layer).norm1.weight") {
            let l = prefix + "encoder.layer.\(layer)."
            var h = try modulated(try layerNorm(x, l + "norm1", epsilon: 1e-6), l + "norm1_modulation", condition: condition)
            let q = try linear(h, rows: tokens, l + "attention.attention.query")
            let k = try linear(h, rows: tokens, l + "attention.attention.key")
            let v = try linear(h, rows: tokens, l + "attention.attention.value")
            let attended = kernels.attention(q: q, k: k, v: v, queries: tokens, keys: tokens, heads: heads,
                                             scale: Float(1.0 / Double(width / heads).squareRoot()))
            var a = try linear(attended, rows: tokens, l + "attention.output.dense")
            try scale(&a, by: l + "layer_scale1.lambda1")
            x = Self.add(a, x)
            h = try modulated(try layerNorm(x, l + "norm2", epsilon: 1e-6), l + "norm2_modulation", condition: condition)
            var hidden = try linear(h, rows: tokens, l + "mlp.fc1")
            NFKSF3DTorchMath.gelu(&hidden)
            var m = try linear(hidden, rows: tokens, l + "mlp.fc2")
            try scale(&m, by: l + "layer_scale2.lambda1")
            x = Self.add(m, x)
            layer += 1
        }
        return try layerNorm(x, prefix + "layernorm", epsilon: 1e-6)
    }

    private func scale(_ x: inout [Float], by key: String) throws {
        let lambda = try w(key), width = try weights.shape(key)[0]
        for index in x.indices { x[index] = x[index] * lambda[index % width] }
    }

    /// `Modulation`: `x · (1 + scale) + shift` from the activated condition's `linear2`.
    private func modulated(_ x: [Float], _ prefix: String, condition: [Float]) throws -> [Float] {
        let embedding = try linear(condition, rows: 1, prefix + ".linear2")
        let width = embedding.count / 2
        var out = x
        for index in out.indices {
            let channel = index % width
            out[index] = x[index] * (1 + embedding[channel]) + embedding[width + channel]
        }
        return out
    }

    // MARK: Backbone

    private func attention(_ query: [Float], queries: Int, _ context: [Float], keys: Int, _ prefix: String) throws -> [Float] {
        let q = try linear(query, rows: queries, prefix + ".wq", bias: false)
        let k = try linear(context, rows: keys, prefix + ".wk", bias: false)
        let v = try linear(context, rows: keys, prefix + ".wv", bias: false)
        let heads = configuration.heads
        let width = q.count / queries / heads
        let attended = kernels.attention(q: q, k: k, v: v, queries: queries, keys: keys, heads: heads,
                                         scale: Float(pow(Double(width), -0.5)))
        return try linear(attended, rows: queries, prefix + ".proj")
    }

    private func feedForward(_ x: [Float], rows: Int, _ prefix: String) throws -> [Float] {
        let projected = try linear(x, rows: rows, prefix + ".net.0.proj")
        let inner = projected.count / rows / 2
        var gate = [Float](repeating: 0, count: rows * inner)
        for row in 0 ..< rows {
            for index in 0 ..< inner { gate[row * inner + index] = projected[row * 2 * inner + inner + index] }
        }
        NFKSF3DTorchMath.gelu(&gate)
        for row in 0 ..< rows {
            for index in 0 ..< inner { gate[row * inner + index] = projected[row * 2 * inner + index] * gate[row * inner + index] }
        }
        return try linear(gate, rows: rows, prefix + ".net.2")
    }

    private func fuse(_ z: [Float], rows: Int, _ x: [Float], keys: Int, _ prefix: String) throws -> [Float] {
        var z = Self.add(z, try attention(try layerNorm(z, prefix + ".norm_z1"), queries: rows, x, keys: keys, prefix + ".attn"))
        z = Self.add(z, try feedForward(try layerNorm(z, prefix + ".norm_z2"), rows: rows, prefix + ".ff"))
        return z
    }

    private func basic(_ z: [Float], rows: Int, _ x: [Float], keys: Int, _ prefix: String) throws -> [Float] {
        var normed = try layerNorm(z, prefix + ".norm1")
        var z = Self.add(z, try attention(normed, queries: rows, normed, keys: rows, prefix + ".attn1"))
        normed = try layerNorm(z, prefix + ".norm2")
        z = Self.add(z, try attention(normed, queries: rows, x, keys: keys, prefix + ".attn2"))
        return Self.add(z, try feedForward(try layerNorm(z, prefix + ".norm3"), rows: rows, prefix + ".ff"))
    }

    /// The backbone's triplane tokens `[tokenWidth, 3 · planeSize²]` from the image tokens.
    func triplaneTokens(_ imageTokens: [Float]) throws -> [Float] {
        let prefix = "backbone."
        let embeddingShape = try weights.shape("tokenizer.embeddings")                        // [3, C, side, side]
        let channels = embeddingShape[1], planeArea = embeddingShape[2] * embeddingShape[3], count = 3 * planeArea
        let embeddings = try w("tokenizer.embeddings")
        var hidden = [Float](repeating: 0, count: channels * count)
        hidden.withUnsafeMutableBufferPointer { buffer in
            for plane in 0 ..< 3 {
                for channel in 0 ..< channels {
                    (buffer.baseAddress! + channel * count + plane * planeArea)
                        .update(from: embeddings + (plane * channels + channel) * planeArea, count: planeArea)
                }
            }
        }
        let normed = kernels.groupNorm(hidden, channels: channels, groups: configuration.normGroups,
                                       weight: try w(prefix + "norm_triplane.weight"), bias: try w(prefix + "norm_triplane.bias"),
                                       epsilon: 1e-6)
        var triplane = try normed.withUnsafeBufferPointer { buffer -> [Float] in
            let view = NFKSF3DTorchKernels.Matrix(base: buffer.baseAddress!, rows: count, columns: channels, rowStride: 1, columnStride: count)
            return kernels.linear(view, weight: try w(prefix + "proj_triplane.weight"), outputs: try weights.shape(prefix + "proj_triplane.weight")[0],
                                  bias: try w(prefix + "proj_triplane.bias"))
        }
        let imageCount = imageTokens.count / channels
        let image = try linear(try layerNorm(imageTokens, prefix + "norm_image"), rows: imageCount, prefix + "proj_image")
        let latentShape = try weights.shape(prefix + "latent_init")
        let latentCount = latentShape[1]
        let initial = Array(UnsafeBufferPointer(start: try w(prefix + "latent_init"), count: latentCount * latentShape[2]))
        let latents = try linear(try layerNorm(initial, prefix + "norm_latent"), rows: latentCount, prefix + "proj_latent")
        var latent = image + latents
        let rows = imageCount + latentCount
        var block = 0
        while weights.contains(prefix + "main_blocks.\(block).fuse_block_in.norm_z1.weight") {
            let b = prefix + "main_blocks.\(block)."
            latent = try fuse(latent, rows: rows, triplane, keys: count, b + "fuse_block_in")
            var inner = 0
            while weights.contains(b + "transformer_block.\(inner).norm1.weight") {
                latent = try basic(latent, rows: rows, imageTokens, keys: imageCount, b + "transformer_block.\(inner)")
                inner += 1
            }
            triplane = try fuse(triplane, rows: count, latent, keys: rows, b + "fuse_block_out")
            block += 1
        }
        let projected = try linear(triplane, rows: count, prefix + "proj_out")
        let width = projected.count / count
        var out = [Float](repeating: 0, count: width * count)
        for token in 0 ..< count {
            for channel in 0 ..< width { out[channel * count + token] = projected[token * width + channel] + hidden[channel * count + token] }
        }
        return out
    }

    // MARK: Upsampler and decoder

    /// The scene code `[3, planeChannels, side, side]` from the triplane tokens: the upsampler's
    /// convolutions and pixel shuffle.
    func sceneCode(_ triplaneTokens: [Float]) throws -> [Float] {
        let planeSide = configuration.planeSize, planeArea = planeSide * planeSide, factor = configuration.upsampleFactor
        let channels = triplaneTokens.count / (3 * planeArea)
        var convolutions = [Int]()
        while weights.contains("post_processor.upsample.\(convolutions.count * 2).weight") { convolutions.append(convolutions.count * 2) }
        var planes = [Float]()
        for plane in 0 ..< 3 {
            var x = [Float](repeating: 0, count: channels * planeArea)
            for channel in 0 ..< channels {
                for pixel in 0 ..< planeArea { x[channel * planeArea + pixel] = triplaneTokens[channel * 3 * planeArea + plane * planeArea + pixel] }
            }
            var current = channels
            for (position, index) in convolutions.enumerated() {
                let key = "post_processor.upsample.\(index)"
                let shape = try weights.shape(key + ".weight")
                let weight = try w(key + ".weight"), bias = try w(key + ".bias"), input = current
                x = x.withUnsafeBufferPointer {
                    kernels.conv2d($0.baseAddress!, channels: input, height: planeSide, width: planeSide, weight: weight,
                                   outputs: shape[0], kernel: shape[2], stride: 1, padding: shape[2] / 2, bias: bias)
                }
                current = shape[0]
                if position != convolutions.count - 1 {
                    for i in x.indices where x[i] < 0 { x[i] = 0 }
                }
            }
            let out = current / (factor * factor), side = planeSide * factor
            var shuffled = [Float](repeating: 0, count: out * side * side)
            for channel in 0 ..< out {
                for i in 0 ..< factor {
                    for j in 0 ..< factor {
                        let source = (channel * factor * factor + i * factor + j) * planeArea
                        for y in 0 ..< planeSide {
                            for xIndex in 0 ..< planeSide {
                                shuffled[(channel * side + y * factor + i) * side + xIndex * factor + j] = x[source + y * planeSide + xIndex]
                            }
                        }
                    }
                }
            }
            planes += shuffled
        }
        return planes
    }

    /// `query_triplane`'s features `[3 · planeChannels, count]` (channel-major, as the decoder's first
    /// layer reads them) at `points` `[count, 3]` in world units.
    func features(_ points: [Float], sceneCode: [Float]) -> [Float] {
        let count = points.count / 3, radius = configuration.radius
        let channels = configuration.planeChannels, side = configuration.planeSize * configuration.upsampleFactor
        var scaled = points
        for index in scaled.indices { scaled[index] = (points[index] - -radius) / (radius - -radius) * 2 + -1 }
        var out = [Float](repeating: 0, count: 3 * channels * count)
        let pairs = [(0, 1), (0, 2), (1, 2)]
        sceneCode.withUnsafeBufferPointer { code in
            out.withUnsafeMutableBufferPointer { buffer in
                for (plane, pair) in pairs.enumerated() {
                    let x = (0 ..< count).map { scaled[$0 * 3 + pair.0] }, y = (0 ..< count).map { scaled[$0 * 3 + pair.1] }
                    NFKSF3DTorchKernels.gridSample(code.baseAddress! + plane * channels * side * side, channels: channels, side: side,
                                                   x: x, y: y, into: buffer.baseAddress! + plane * channels * count)
                }
            }
        }
        return out
    }

    /// One `MaterialMLP` head on features `[inputs, count]` channel-major: the hidden layers with SiLU, then
    /// the output layer, `[count, outputs]`. The surface's features are batched `[1, count, inputs]`; the
    /// bake indexes its batch away (`query_triplane(…)[0]`), which sends its first layer through `addmm`.
    private func head(_ name: String, features: [Float], count: Int, twoDimensional: Bool = false) throws -> [Float] {
        let prefix = "decoder.heads.\(name)."
        var x = try features.withUnsafeBufferPointer { buffer -> [Float] in
            let view = NFKSF3DTorchKernels.Matrix(base: buffer.baseAddress!, rows: count, columns: features.count / count,
                                                  rowStride: 1, columnStride: count)
            return kernels.linear(view, weight: try w(prefix + "0.weight"), outputs: try weights.shape(prefix + "0.weight")[0],
                                  bias: try w(prefix + "0.bias"), twoDimensional: twoDimensional)
        }
        var layer = 2
        while weights.contains(prefix + "\(layer).weight") {
            NFKSF3DTorchMath.silu(&x)
            x = try linear(x, rows: count, prefix + "\(layer)")
            layer += 2
        }
        return x
    }

    /// The density (after its exponential) and the vertex offset at `points` `[count, 3]`.
    func geometry(_ points: [Float], sceneCode: [Float]) throws -> (density: [Float], vertexOffset: [Float]) {
        let count = points.count / 3
        let f = features(points, sceneCode: sceneCode)
        var density = try head("density", features: f, count: count)
        for index in density.indices { density[index] = NFKSF3DTorchMath.exp(density[index] + -1) }
        var offset = try head("vertex_offset", features: f, count: count)
        for index in offset.indices { offset[index] = offset[index] + 0 }
        return (density, offset)
    }

    /// The albedo (after its sigmoid) and the perturbed normal, normalized with an epsilon of 1e-7, at
    /// `points` `[count, 3]`.
    func surface(_ points: [Float], sceneCode: [Float]) throws -> (albedo: [Float], normal: [Float]) {
        let count = points.count / 3
        let f = features(points, sceneCode: sceneCode)
        var albedo = try head("features", features: f, count: count, twoDimensional: true)
        NFKSF3DTorchMath.sigmoid(&albedo)
        let raw = try head("perturb_normal", features: f, count: count, twoDimensional: true)
        var normal = raw
        for point in 0 ..< count {
            let unit = NFKSF3DVector.at(raw, point).torchNormalized(epsilon: 1e-7)
            normal[point * 3] = unit.x; normal[point * 3 + 1] = unit.y; normal[point * 3 + 2] = unit.z
        }
        return (albedo, normal)
    }

    /// The scene code for a conditioning image `[imageSize, imageSize, 3]` in 0...1 and the camera condition.
    func sceneCode(color: [Float], cameraCondition: [Float]) throws -> [Float] {
        let camera = try cameraEmbedding(condition: cameraCondition)
        return try sceneCode(try triplaneTokens(try imageTokens(color, camera: camera)))
    }
}

extension NFKMLXStableFast3DReferenceNetwork {

    /// The whole run with the network and its fields in the reference's arithmetic: the release's
    /// conditioning, the scene code, and the mesh. The material estimate and, when asked, the illumination
    /// estimate come from `net` on the GPU, from the same conditioning and triplane tokens.
    func reconstruct(rgba: [UInt8], width: Int, height: Int, grid: NFKMLXTetrahedralGrid, net: NFKMLXStableFast3DNet,
                     options: NFKMLXStableFast3DOptions) throws -> NFKMLXStableFast3DReconstruction {
        let side = configuration.imageSize
        let (color, mask) = NFKMLXStableFast3D.referenceConditioning(rgba: rgba, width: width, height: height, side: side,
                                                                      foregroundRatio: options.foregroundRatio,
                                                                      background: configuration.background)
        let camera = try cameraEmbedding(condition: configuration.cameraCondition)
        let tokens = try triplaneTokens(try imageTokens(color, camera: camera))
        let code = try sceneCode(tokens)
        let images = MLXArray(color).reshaped([1, side, side, 3])
        let materials = net.materials(images * MLXArray(mask).reshaped([1, side, side, 1]))
        var amplitudes: [Float]? = nil
        if options.estimateIllumination {
            let channels = configuration.tokenWidth, count = tokens.count / channels
            var tokenMajor = [Float](repeating: 0, count: tokens.count)
            for token in 0 ..< count {
                for channel in 0 ..< channels { tokenMajor[token * channels + channel] = tokens[channel * count + token] }
            }
            let estimate = net.illumination(MLXArray(tokenMajor).reshaped([1, count, channels]))[0]
            eval(estimate)
            amplitudes = estimate.asArray(Float.self)
        }
        eval(materials.roughness, materials.metallic)
        let mesh = try NFKMLXStableFast3DGeometry.mesh(fields: NFKSF3DReferenceFields(network: self, sceneCode: code),
                                                       configuration: configuration, grid: grid,
                                                       roughness: materials.roughness[0].item(Float.self),
                                                       metallic: materials.metallic[0].item(Float.self), options: options)
        return NFKMLXStableFast3DReconstruction(mesh: mesh, illuminationAmplitudes: amplitudes)
    }
}
