//
//  NFKMLXStableFast3DGeometry.swift
//  InferKitMLX
//

import Darwin
import Foundation
import InferKit
import MLX

// The surface and texture stages of Stable Fast 3D, on the CPU in float32 as the release runs them:
// marching tetrahedra over the deformed grid (`MarchingTetrahedraHelper`), the box-projection unwrapper
// (`uv_unwrapper`, its overlap test ported from the extension's C++), the UV-space rasterizer and
// interpolator (`texture_baker`'s CPU kernels), the island padding (`dilate_fill`), and the 8-bit
// quantization. Swift's float arithmetic is unfused, as the reference extensions are when built
// without multiply-add contraction (the x86 build's arithmetic). The stages the release runs in torch
// follow torch's CPU kernels: their contracted cross products and norms, their reduction orders, Sleef's
// `tanhf`, `sinf`, and `cosf`, and BLAS for their matrix products (`NFKSF3DTorchKernels`).

// MARK: - The tetrahedral grid

/// The release's tetrahedral grid (`160_tets.npz`): vertices in 0...1 and four vertex indices a
/// tetrahedron. The file is a NumPy archive of `vertices` (float64 `[N, 3]`) and `indices` (int64
/// `[M, 4]`); the reference reads the vertices as float32. Introduced in InferKit 0.4.0.
@objc(NFKMLXTetrahedralGrid)
public final class NFKMLXTetrahedralGrid: NSObject, Sendable {
    /// Three float32 a vertex, in 0...1.
    public let vertices: [Float]
    /// Four vertex indices a tetrahedron.
    public let tetrahedra: [Int32]

    @objc public var vertexCount: Int { vertices.count / 3 }
    @objc public var tetrahedronCount: Int { tetrahedra.count / 4 }

    public init(vertices: [Float], tetrahedra: [Int32]) throws {
        guard vertices.count % 3 == 0, tetrahedra.count % 4 == 0 else {
            throw NFKMLXError.weightsMismatch("a tetrahedral grid holds vertex triples and index quadruples")
        }
        let count = Int32(vertices.count / 3)
        guard tetrahedra.allSatisfy({ $0 >= 0 && $0 < count }) else {
            throw NFKMLXError.weightsMismatch("a tetrahedron names a vertex past the grid's \(count)")
        }
        self.vertices = vertices
        self.tetrahedra = tetrahedra
        super.init()
    }

    /// Reads the release's NumPy archive.
    @objc(initWithContentsOfURL:error:)
    public convenience init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let entries = try NFKMLXZipArchive.entries(in: data)
        func array(_ name: String) throws -> NFKSF3DNumPyArray {
            guard let entry = entries.first(where: { $0.name == name }) else {
                throw NFKMLXError.weightsMismatch("\(url.lastPathComponent) holds no \(name)")
            }
            return try NFKSF3DNumPyArray(try NFKMLXZipArchive.contents(of: entry, in: data))
        }
        let vertices = try array("vertices.npy")
        let indices = try array("indices.npy")
        guard vertices.shape.count == 2, vertices.shape[1] == 3, indices.shape.count == 2, indices.shape[1] == 4 else {
            throw NFKMLXError.weightsMismatch("\(url.lastPathComponent) is not a grid of [N, 3] vertices and [M, 4] tetrahedra")
        }
        try self.init(vertices: vertices.floats(), tetrahedra: try indices.integers())
    }

    /// A regular grid of `divisions` cubes a side over 0...1, each cube split into the six tetrahedra
    /// around its main diagonal and every tetrahedron positively oriented, as every one of the release's
    /// is (the case table assumes it): a stand-in for the release's grid in a test or an example, not the
    /// grid the release was trained on.
    @objc(latticeWithDivisions:)
    public static func lattice(divisions: Int) -> NFKMLXTetrahedralGrid {
        let n = max(1, divisions)
        var vertices = [Float]()
        func index(_ x: Int, _ y: Int, _ z: Int) -> Int32 { Int32((x * (n + 1) + y) * (n + 1) + z) }
        for x in 0 ... n {
            for y in 0 ... n {
                for z in 0 ... n { vertices += [Float(x) / Float(n), Float(y) / Float(n), Float(z) / Float(n)] }
            }
        }
        let paths = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        var tetrahedra = [Int32]()
        for x in 0 ..< n {
            for y in 0 ..< n {
                for z in 0 ..< n {
                    for path in paths {
                        var corner = [x, y, z]
                        var tetrahedron = [index(x, y, z)]
                        for axis in path {
                            corner[axis] += 1
                            tetrahedron.append(index(corner[0], corner[1], corner[2]))
                        }
                        // An odd permutation of the axes walks the tetrahedron the other way round.
                        let odd = [[0, 2, 1], [1, 0, 2], [2, 1, 0]].contains(path)
                        if odd {
                            tetrahedron.swapAt(2, 3)
                        }
                        tetrahedra += tetrahedron
                    }
                }
            }
        }
        // Every index lies inside the lattice, which the checked initializer would confirm again.
        return try! NFKMLXTetrahedralGrid(vertices: vertices, tetrahedra: tetrahedra)
    }
}

/// One `.npy` array: its little-endian element type, shape, and bytes, C order.
struct NFKSF3DNumPyArray {
    let descriptor: String
    let shape: [Int]
    let payload: Data

    init(_ data: Data) throws {
        let reader = NFKMLXByteReader(data)
        guard data.count >= 10, try reader.u8(0) == 0x93, try reader.string(1, count: 5) == "NUMPY" else {
            throw NFKMLXError.malformedCheckpoint("a .npy entry has no NumPy magic")
        }
        let major = try reader.u8(6)
        let headerLength = major == 1 ? try reader.u16(8) : try reader.u32(8)
        let headerStart = major == 1 ? 10 : 12
        let header = try reader.string(headerStart, count: headerLength)
        func field(_ name: String) -> String? {
            guard let range = header.range(of: "'\(name)':") else { return nil }
            return String(header[range.upperBound...]).trimmingCharacters(in: .whitespaces)
        }
        guard let descr = field("descr"), let quoted = descr.split(separator: "'").first,
              let order = field("fortran_order"), let shapeText = field("shape") else {
            throw NFKMLXError.malformedCheckpoint("a .npy header lacks its descr, order, or shape")
        }
        guard order.hasPrefix("False") else {
            throw NFKMLXError.malformedCheckpoint("a .npy array is in Fortran order")
        }
        descriptor = String(quoted)
        let inside = shapeText.drop(while: { $0 != "(" }).dropFirst().prefix(while: { $0 != ")" })
        shape = inside.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let start = data.startIndex + headerStart + headerLength
        payload = data[start ..< data.endIndex]
    }

    var count: Int { shape.reduce(1, *) }

    func floats() throws -> [Float] {
        switch descriptor {
        case "<f8": return elements(Double.self).map { Float($0) }
        case "<f4": return elements(Float.self)
        default: throw NFKMLXError.malformedCheckpoint("a grid's vertices are \(descriptor), not float")
        }
    }

    func integers() throws -> [Int32] {
        switch descriptor {
        case "<i8": return elements(Int64.self).map { Int32(truncatingIfNeeded: $0) }
        case "<i4": return elements(Int32.self)
        default: throw NFKMLXError.malformedCheckpoint("a grid's tetrahedra are \(descriptor), not integers")
        }
    }

    private func elements<T>(_ type: T.Type) -> [T] {
        let size = MemoryLayout<T>.size
        precondition(payload.count >= count * size)
        return payload.withUnsafeBytes { raw in
            (0 ..< count).map { raw.loadUnaligned(fromByteOffset: $0 * size, as: T.self) }
        }
    }
}

// MARK: - Marching tetrahedra

/// `MarchingTetrahedraHelper._forward`: one vertex on every grid edge whose ends straddle zero, at the
/// edge's linear zero crossing, numbered in the edges' sorted order; then each straddling tetrahedron's
/// triangles from the case table, the one-triangle tetrahedra first and the two-triangle ones after, each
/// group in tetrahedron order.
enum NFKMarchingTetrahedra {
    static let table: [[Int]] = [
        [], [1, 0, 2], [4, 0, 3], [1, 4, 2, 1, 3, 4], [3, 1, 5], [2, 3, 0, 2, 5, 3], [1, 4, 0, 1, 5, 4], [4, 2, 5],
        [4, 5, 2], [4, 1, 0, 4, 5, 1], [3, 2, 0, 3, 5, 2], [1, 3, 5], [4, 1, 2, 4, 3, 1], [3, 0, 4], [2, 0, 1], [],
    ]
    static let edges: [(Int, Int)] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    /// - Parameters:
    ///   - positions: three float32 a grid vertex.
    ///   - level: the signed level a vertex, inside above zero.
    ///   - tetrahedra: four vertex indices a tetrahedron.
    static func surface(positions: [Float], level: [Float], tetrahedra: [Int32]) -> (vertices: [Float], faces: [UInt32]) {
        let vertexCount = level.count
        let inside = level.map { $0 > 0 }
        var straddling = [Int]()
        var crossings = [UInt64]()
        for tetrahedron in 0 ..< tetrahedra.count / 4 {
            var occupied = 0
            for corner in 0 ..< 4 where inside[Int(tetrahedra[tetrahedron * 4 + corner])] {
                occupied += 1
            }
            guard occupied > 0 && occupied < 4 else { continue }
            straddling.append(tetrahedron)
            for (a, b) in edges {
                let first = Int(tetrahedra[tetrahedron * 4 + a]), second = Int(tetrahedra[tetrahedron * 4 + b])
                if inside[first] != inside[second] {
                    crossings.append(UInt64(min(first, second)) * UInt64(vertexCount) + UInt64(max(first, second)))
                }
            }
        }
        crossings.sort()
        var unique = [UInt64]()
        unique.reserveCapacity(crossings.count / 2)
        for key in crossings where unique.last != key {
            unique.append(key)
        }
        var vertices = [Float](repeating: 0, count: unique.count * 3)
        for (index, key) in unique.enumerated() {
            let a = Int(key / UInt64(vertexCount)), b = Int(key % UInt64(vertexCount))
            let levelA = level[a], negatedB = -level[b]
            let denominator = levelA + negatedB
            let weightA = negatedB / denominator, weightB = levelA / denominator
            for axis in 0 ..< 3 {
                vertices[index * 3 + axis] = positions[a * 3 + axis] * weightA + positions[b * 3 + axis] * weightB
            }
        }
        func vertex(_ first: Int, _ second: Int) -> UInt32 {
            let key = UInt64(min(first, second)) * UInt64(vertexCount) + UInt64(max(first, second))
            var low = 0, high = unique.count
            while low < high {
                let middle = (low + high) / 2
                if unique[middle] < key { low = middle + 1 } else { high = middle }
            }
            return UInt32(low)
        }
        var faces = [UInt32]()
        for pass in 1 ... 2 {
            for tetrahedron in straddling {
                var code = 0
                for corner in 0 ..< 4 where inside[Int(tetrahedra[tetrahedron * 4 + corner])] {
                    code |= 1 << corner
                }
                let entry = table[code]
                guard entry.count == pass * 3 else { continue }
                for slot in entry {
                    let (a, b) = edges[slot]
                    faces.append(vertex(Int(tetrahedra[tetrahedron * 4 + a]), Int(tetrahedra[tetrahedron * 4 + b])))
                }
            }
        }
        return (vertices, faces)
    }
}

// MARK: - torch's float32 reductions

/// torch's CPU float32 sum over the leading axis of rows narrower than a vector (`cascade_sum` takes
/// `scalar_outer_sum`, which sums each column by `row_sum`): the column read as rows of four interleaved
/// lanes, the lanes summed together by `multi_row_sum`, the tail added to the first lane, and the other
/// three lanes added to it in order. `multi_row_sum` adds rows in order into a first accumulator `2^p` at
/// a time (`p = max(4, ⌈log₂ n⌉ / 4)`), cascades each full group up four levels, adds the tail to the first
/// level, and folds the levels into it in order. Bit-exact against torch 2.14 over sizes 5 to 100,000.
enum NFKSF3DTorchSum {

    /// `multi_row_sum` of `count` rows of `width` lanes, `value(row, lane)`.
    static func multiRow(count: Int, width: Int, _ value: (Int, Int) -> Float) -> [Float] {
        var ceilLog = 0
        while (1 << ceilLog) < count { ceilLog += 1 }
        let power = max(4, ceilLog / 4)
        let step = 1 << power, mask = step - 1
        var levels = [[Float]](repeating: [Float](repeating: 0, count: width), count: 4)
        var i = 0
        while i + step <= count {
            for _ in 0 ..< step {
                for k in 0 ..< width { levels[0][k] += value(i, k) }
                i += 1
            }
            for j in 1 ..< 4 {
                for k in 0 ..< width {
                    levels[j][k] += levels[j - 1][k]
                    levels[j - 1][k] = 0
                }
                if i & (mask << (j * power)) != 0 { break }
            }
        }
        while i < count {
            for k in 0 ..< width { levels[0][k] += value(i, k) }
            i += 1
        }
        for j in 1 ..< 4 {
            for k in 0 ..< width { levels[0][k] += levels[j][k] }
        }
        return levels[0]
    }

    /// The sum over the leading axis of `rows`, `width` floats a row.
    static func outer(_ rows: [Float], width: Int) -> [Float] {
        let count = rows.count / width
        return (0 ..< width).map { column -> Float in
            let groups = count / 4
            var partial = multiRow(count: groups, width: 4) { row, lane in rows[(row * 4 + lane) * width + column] }
            for row in (groups * 4) ..< count { partial[0] += rows[row * width + column] }
            for lane in 1 ..< 4 { partial[0] += partial[lane] }
            return partial[0]
        }
    }

    /// `x.mean(dim)` over the leading axis: the sum divided by the count in float32.
    static func outerMean(_ rows: [Float], width: Int) -> [Float] {
        let count = Float(rows.count / width)
        return outer(rows, width: width).map { $0 / count }
    }
}

// MARK: - Vector helpers

struct NFKSF3DVector: Equatable {
    var x: Float
    var y: Float
    var z: Float

    static func - (a: Self, b: Self) -> Self { Self(x: a.x - b.x, y: a.y - b.y, z: a.z - b.z) }
    static func + (a: Self, b: Self) -> Self { Self(x: a.x + b.x, y: a.y + b.y, z: a.z + b.z) }
    static func * (a: Self, s: Float) -> Self { Self(x: a.x * s, y: a.y * s, z: a.z * s) }
    func dot(_ b: Self) -> Float { x * b.x + y * b.y + z * b.z }
    /// The extensions' `cross`, built without contraction.
    func cross(_ b: Self) -> Self { Self(x: y * b.z - z * b.y, y: z * b.x - x * b.z, z: x * b.y - y * b.x) }
    /// The extensions' normalization: `v / max(‖v‖, epsilon)` without contraction.
    func normalized(epsilon: Float = 1e-12) -> Self {
        let length = max((x * x + y * y + z * z).squareRoot(), epsilon)
        return Self(x: x / length, y: y / length, z: z / length)
    }
    /// `torch.cross`: each component `a·b − c·d` with the first product fused, as clang contracts the kernel.
    func torchCross(_ b: Self) -> Self {
        Self(x: (-(z * b.y)).addingProduct(y, b.z), y: (-(x * b.z)).addingProduct(z, b.x), z: (-(y * b.x)).addingProduct(x, b.y))
    }
    /// `F.normalize` over three components: the squares accumulated in order with fused multiply-adds, the
    /// root floored at `epsilon`.
    func torchNormalized(epsilon: Float = 1e-12) -> Self {
        let length = max((x * x).addingProduct(y, y).addingProduct(z, z).squareRoot(), epsilon)
        return Self(x: x / length, y: y / length, z: z / length)
    }
    subscript(axis: Int) -> Float { axis == 0 ? x : axis == 1 ? y : z }

    static func at(_ values: [Float], _ index: Int) -> Self {
        Self(x: values[index * 3], y: values[index * 3 + 1], z: values[index * 3 + 2])
    }
}

/// `Mesh._compute_vertex_normal`: each face's unnormalized normal added to its three vertices (first
/// corners, then second, then third, each in face order), a near-zero sum replaced by +z, then normalized.
func NFKSF3DVertexNormals(positions: [Float], faces: [UInt32]) -> [Float] {
    let count = positions.count / 3
    var sums = [NFKSF3DVector](repeating: NFKSF3DVector(x: 0, y: 0, z: 0), count: count)
    var faceNormals = [NFKSF3DVector]()
    faceNormals.reserveCapacity(faces.count / 3)
    for face in 0 ..< faces.count / 3 {
        let v0 = NFKSF3DVector.at(positions, Int(faces[face * 3]))
        let v1 = NFKSF3DVector.at(positions, Int(faces[face * 3 + 1]))
        let v2 = NFKSF3DVector.at(positions, Int(faces[face * 3 + 2]))
        faceNormals.append((v1 - v0).torchCross(v2 - v0))
    }
    for corner in 0 ..< 3 {
        for face in 0 ..< faces.count / 3 {
            let index = Int(faces[face * 3 + corner])
            sums[index] = sums[index] + faceNormals[face]
        }
    }
    var normals = [Float](repeating: 0, count: count * 3)
    for index in 0 ..< count {
        var sum = sums[index]
        if !(sum.dot(sum) > 1e-20) {
            sum = NFKSF3DVector(x: 0, y: 0, z: 1)
        }
        let unit = sum.torchNormalized()
        normals[index * 3] = unit.x; normals[index * 3 + 1] = unit.y; normals[index * 3 + 2] = unit.z
    }
    return normals
}

/// The per-vertex tangents of `Mesh._compute_vertex_tangent` and `Unwrapper._calculate_tangents`: each
/// face's `dpos1 · Δv₂ − dpos2 · Δv₁` over its texture-space determinant clipped below at 1e-6, added
/// to its corners, averaged, normalized, made perpendicular to the normal, and normalized again.
/// `cornerUV` holds two floats a face corner.
func NFKSF3DVertexTangents(positions: [Float], normals: [Float], faces: [UInt32], cornerUV: [Float]) -> [Float] {
    let count = positions.count / 3
    let faceCount = faces.count / 3
    var tangents = [NFKSF3DVector](repeating: NFKSF3DVector(x: 0, y: 0, z: 0), count: count)
    var tally = [Float](repeating: 0, count: count)
    var perFace = [NFKSF3DVector]()
    perFace.reserveCapacity(faceCount)
    for face in 0 ..< faceCount {
        let p0 = NFKSF3DVector.at(positions, Int(faces[face * 3]))
        let p1 = NFKSF3DVector.at(positions, Int(faces[face * 3 + 1]))
        let p2 = NFKSF3DVector.at(positions, Int(faces[face * 3 + 2]))
        let u0 = cornerUV[face * 6], v0 = cornerUV[face * 6 + 1]
        let duv1 = (cornerUV[face * 6 + 2] - u0, cornerUV[face * 6 + 3] - v0)
        let duv2 = (cornerUV[face * 6 + 4] - u0, cornerUV[face * 6 + 5] - v0)
        let dpos1 = p1 - p0, dpos2 = p2 - p0
        let numerator = dpos1 * duv2.1 - dpos2 * duv1.1
        let denominator = max(duv1.0 * duv2.1 - duv1.1 * duv2.0, 1e-6)
        perFace.append(NFKSF3DVector(x: numerator.x / denominator, y: numerator.y / denominator, z: numerator.z / denominator))
    }
    for corner in 0 ..< 3 {
        for face in 0 ..< faceCount {
            let index = Int(faces[face * 3 + corner])
            tangents[index] = tangents[index] + perFace[face]
            tally[index] += 1
        }
    }
    var result = [Float](repeating: 0, count: count * 3)
    for index in 0 ..< count {
        let mean = NFKSF3DVector(x: tangents[index].x / tally[index], y: tangents[index].y / tally[index],
                                 z: tangents[index].z / tally[index]).torchNormalized()
        let normal = NFKSF3DVector.at(normals, index)
        let unit = (mean - normal * mean.dot(normal)).torchNormalized()
        result[index * 3] = unit.x; result[index * 3 + 1] = unit.y; result[index * 3 + 2] = unit.z
    }
    return result
}

// MARK: - The box unwrapper

/// The unwrapper of `uv_unwrapper.Unwrapper`: the mesh aligned to its principal axes, each face
/// box-projected onto the cube face its averaged normal points to, each cube face's charts rotated to a
/// common tangent direction, overlapping faces moved to second and third slots, and every slot packed
/// into a 3 × 2 atlas with a third row for the overlaps.
enum NFKSF3DUnwrapper {

    /// The texture coordinates of every face corner, two floats a corner, for the welded mesh `positions`
    /// `[N·3]` with unit `normals` and `faces`. `padding` is the islands' margin (the release's 0.02).
    static func unwrap(positions: [Float], normals: [Float], faces: [UInt32], padding: Double) throws -> [Float] {
        let (aligned, alignedNormals) = try alignedToPrincipalAxes(positions: positions, normals: normals)
        let (boxUV, index) = boxProjection(positions: aligned, normals: alignedNormals, faces: faces)
        let rotated = try rotatedSlices(positions: aligned, normals: alignedNormals, faces: faces, uv: boxUV, index: index)
        let atlas = atlasIndices(positions: aligned, faces: faces, uv: rotated, index: index)
        return placed(uv: rotated, atlas: atlas, padding: padding)
    }

    /// torch's seed-0 `randn(3, 2)`: `_align_mesh_with_main_axis` reseeds before `pca_lowrank` draws its
    /// sketch, so the sketch is this constant (torch 2.14's CPU generator).
    static let sketch: [Float] = [0x1.8a7eb8p+0, -0x1.2c78a0p-2, -0x1.16e292p+1, 0x1.23096cp-1, -0x1.15a342p+0, -0x1.660a5ap+0]

    /// `_align_mesh_with_main_axis`: `torch.pca_lowrank(positions, q: 2)` (two power iterations of
    /// orthonormalized sketches by LAPACK's QR, then LAPACK's SVD of the projection), the two principal
    /// axes and their cross product sent to the canonical axes they lie nearest, and the positions and
    /// normals rotated by them.
    static func alignedToPrincipalAxes(positions: [Float], normals: [Float]) throws -> (positions: [Float], normals: [Float]) {
        let count = positions.count / 3
        let center = NFKSF3DTorchSum.outerMean(positions, width: 3)
        var centered = positions
        for index in 0 ..< count {
            for axis in 0 ..< 3 { centered[index * 3 + axis] -= center[axis] }
        }
        let kernels = try NFKSF3DTorchKernels()
        typealias Matrix = NFKSF3DTorchKernels.Matrix
        // `torch.linalg.qr(x).Q` is column-major, which decides the transposes the next products pass BLAS.
        func orthonormal(_ x: [Float], rows: Int) throws -> [Float] {
            let q = try NFKSF3DLAPACK.reducedQ(rowMajor: x, rows: rows, columns: 2)
            var columnMajor = [Float](repeating: 0, count: q.count)
            for row in 0 ..< rows {
                columnMajor[row] = q[row * 2]
                columnMajor[rows + row] = q[row * 2 + 1]
            }
            return columnMajor
        }
        func product(_ first: Matrix, _ second: Matrix) -> [Float] {
            var out = [Float](repeating: 0, count: first.rows * second.columns)
            out.withUnsafeMutableBufferPointer { kernels.addmm(into: $0.baseAddress!, first, second, beta: 0) }
            return out
        }
        var q = try centered.withUnsafeBufferPointer { a in
            try sketch.withUnsafeBufferPointer { r in
                try orthonormal(product(.rowMajor(a.baseAddress!, rows: count, columns: 3), .rowMajor(r.baseAddress!, rows: 3, columns: 2)),
                                rows: count)
            }
        }
        for _ in 0 ..< 2 {
            let inner = centered.withUnsafeBufferPointer { a in
                q.withUnsafeBufferPointer { qb in
                    product(Matrix(base: a.baseAddress!, rows: 3, columns: count, rowStride: 1, columnStride: 3),
                            Matrix(base: qb.baseAddress!, rows: count, columns: 2, rowStride: 1, columnStride: count))
                }
            }
            let small = try orthonormal(inner, rows: 3)
            let outer = centered.withUnsafeBufferPointer { a in
                small.withUnsafeBufferPointer { qb in
                    product(.rowMajor(a.baseAddress!, rows: count, columns: 3),
                            Matrix(base: qb.baseAddress!, rows: 3, columns: 2, rowStride: 1, columnStride: 3))
                }
            }
            q = try orthonormal(outer, rows: count)
        }
        let b = centered.withUnsafeBufferPointer { a in
            q.withUnsafeBufferPointer { qb in
                product(.rowMajor(qb.baseAddress!, rows: 2, columns: count), .rowMajor(a.baseAddress!, rows: count, columns: 3))
            }
        }                                                                                       // [2, 3]
        let v = try NFKSF3DLAPACK.rightSingularVectors(rowMajor: b, rows: 2, columns: 3)       // [3, 2]
        let main = NFKSF3DVector(x: v[0], y: v[2], z: v[4]).torchNormalized(epsilon: 1e-6)
        let second = NFKSF3DVector(x: v[1], y: v[3], z: v[5])
        let secondary = (second - main * second.dot(main)).torchNormalized(epsilon: 1e-6)
        let third = main.torchCross(secondary).torchNormalized(epsilon: 1e-6)
        func nearest(_ axis: NFKSF3DVector) -> Int {
            let magnitudes = [abs(axis.x), abs(axis.y), abs(axis.z)]
            var best = 0
            for candidate in 1 ..< 3 where magnitudes[candidate] > magnitudes[best] {
                best = candidate
            }
            return best
        }
        var slots = [nearest(main), nearest(secondary), nearest(third)]
        var pass = 1
        while Set(slots).count != 3 {
            let missing = Set([0, 1, 2]).subtracting(slots).sorted().first!
            if pass == 1 {
                slots[2] = missing
            } else if pass == 2 {
                slots[1] = missing
            } else {
                throw NFKMLXError.unsupportedConfiguration("the mesh's principal axes do not reach three canonical axes")
            }
            pass += 1
        }
        var rows = [NFKSF3DVector](repeating: NFKSF3DVector(x: 0, y: 0, z: 0), count: 3)
        rows[slots[0]] = main; rows[slots[1]] = secondary; rows[slots[2]] = third
        let rotation = rows.flatMap { [$0.x, $0.y, $0.z] }
        // `torch.einsum("ij,nj->ni", rotation, values)`: one BLAS product.
        func rotate(_ values: [Float]) -> [Float] {
            values.withUnsafeBufferPointer { p in
                rotation.withUnsafeBufferPointer { r in
                    product(.rowMajor(p.baseAddress!, rows: values.count / 3, columns: 3),
                            Matrix(base: r.baseAddress!, rows: 3, columns: 3, rowStride: 1, columnStride: 3))
                }
            }
        }
        return (rotate(positions), rotate(normals))
    }

    /// `_box_assign_vertex_to_cube_face`: positions to −1...1 over their box, each face sent to the cube
    /// face its averaged vertex normal points to most, and its corners projected onto that face, each
    /// corner slot scaled by the largest coordinate the slot reaches along the projection axis over all
    /// faces. Returns six floats a face and the cube-face index.
    static func boxProjection(positions: [Float], normals: [Float], faces: [UInt32]) -> (uv: [Float], index: [Int]) {
        let count = positions.count / 3
        var low = [Float](repeating: .greatestFiniteMagnitude, count: 3), high = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        for index in 0 ..< count {
            for axis in 0 ..< 3 {
                low[axis] = min(low[axis], positions[index * 3 + axis])
                high[axis] = max(high[axis], positions[index * 3 + axis])
            }
        }
        var scaled = positions
        for index in 0 ..< count {
            for axis in 0 ..< 3 {
                let unit = (positions[index * 3 + axis] - low[axis]) / (high[axis] - low[axis])
                scaled[index * 3 + axis] = 2 * unit - 1
            }
        }
        let faceCount = faces.count / 3
        var u = [Float](repeating: 0, count: faceCount * 3), v = u, reach = [Float](repeating: 1, count: faceCount * 3)
        var cubeFace = [Int](repeating: 0, count: faceCount)
        for face in 0 ..< faceCount {
            let corners = (0 ..< 3).map { NFKSF3DVector.at(scaled, Int(faces[face * 3 + $0])) }
            let n = (0 ..< 3).map { NFKSF3DVector.at(normals, Int(faces[face * 3 + $0])) }
            let averaged = (n[0] + n[1] + n[2]).torchNormalized(epsilon: 1e-6)
            let scores = [averaged.x, -averaged.x, averaged.y, -averaged.y, averaged.z, -averaged.z]
            var best = 0
            for candidate in 1 ..< 6 where scores[candidate] > scores[best] {
                best = candidate
            }
            cubeFace[face] = best
            for (slot, p) in corners.enumerated() {
                let at = face * 3 + slot
                switch best {
                case 0, 1: reach[at] = abs(p.x); u[at] = p.y; v[at] = -p.z
                case 2, 3: reach[at] = abs(p.y); u[at] = p.x; v[at] = -p.z
                case 4: reach[at] = abs(p.z); u[at] = p.x; v[at] = p.y
                default: reach[at] = abs(p.z); u[at] = p.x; v[at] = -p.y
                }
            }
        }
        var divisor = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        for face in 0 ..< faceCount {
            for slot in 0 ..< 3 { divisor[slot] = max(divisor[slot], reach[face * 3 + slot]) }
        }
        var uv = [Float](repeating: 0, count: faceCount * 6)
        for face in 0 ..< faceCount {
            for slot in 0 ..< 3 {
                let at = face * 3 + slot
                uv[at * 2] = min(max((u[at] / divisor[slot] + 1) * 0.5, 0), 1)
                uv[at * 2 + 1] = min(max((v[at] / divisor[slot] + 1) * 0.5, 0), 1)
            }
        }
        return (uv, cubeFace)
    }

    /// `_rotate_uv_slices_consistent_space`: each cube face's charts rotated about the atlas center so
    /// their mean tangent meets the mean of a reference tangent field (the normal's cross with
    /// `(−y, x, 0)`, crossed back), then rescaled to fill 0...1. The release normalizes that field with
    /// `F.normalize(field, -1)`, whose second argument is the norm's order: each vector is divided by
    /// its p = −1 norm, `1 / Σ 1/|xᵢ|`, floored at 1e-12, so a vector with a zero component is scaled by
    /// 1e12 and outweighs the rest of its cube face's mean.
    static func rotatedSlices(positions: [Float], normals: [Float], faces: [UInt32], uv: [Float], index: [Int]) throws -> [Float] {
        let kernels = try NFKSF3DTorchKernels()
        let tangents = NFKSF3DVertexTangents(positions: positions, normals: normals, faces: faces, cornerUV: uv)
        let count = positions.count / 3
        var expected = [Float](repeating: 0, count: count * 3)
        for vertex in 0 ..< count {
            let p = NFKSF3DVector.at(positions, vertex), n = NFKSF3DVector.at(normals, vertex)
            let field = NFKSF3DVector(x: -p.y, y: p.x, z: 0)
            let raw = n.torchCross(field.torchCross(n))
            // The p = −1 norm's powers are libm's `powf`, which is not always the reciprocal.
            let norm = powf(powf(abs(raw.x), -1) + powf(abs(raw.y), -1) + powf(abs(raw.z), -1), -1)
            let divisor = max(norm, 1e-12)
            expected[vertex * 3] = raw.x / divisor; expected[vertex * 3 + 1] = raw.y / divisor; expected[vertex * 3 + 2] = raw.z / divisor
        }
        var result = uv
        let faceCount = faces.count / 3
        for cube in 0 ..< 6 {
            let members = (0 ..< faceCount).filter { index[$0] % 6 == cube }
            guard !members.isEmpty else { continue }
            // `tangents[faces][mask].mean(dim: (0, 1))`: the masked faces' corners as rows, face-major.
            var actualRows = [Float](), expectedRows = [Float]()
            actualRows.reserveCapacity(members.count * 9)
            expectedRows.reserveCapacity(members.count * 9)
            for face in members {
                for corner in 0 ..< 3 {
                    let vertex = Int(faces[face * 3 + corner])
                    for axis in 0 ..< 3 {
                        actualRows.append(tangents[vertex * 3 + axis])
                        expectedRows.append(expected[vertex * 3 + axis])
                    }
                }
            }
            let actual = NFKSF3DTorchSum.outerMean(actualRows, width: 3)
            let wanted = NFKSF3DTorchSum.outerMean(expectedRows, width: 3)
            // `torch.dot` of 3-vectors adds the first and last products, then the middle one.
            let dot = (actual[0] * wanted[0] + actual[2] * wanted[2]) + actual[1] * wanted[1]
            let cross = actual[0] * wanted[1] - actual[1] * wanted[0]
            let angle = atan2(cross, dot)
            let (c, s) = (NFKSF3DTorchMath.cos(angle), NFKSF3DTorchMath.sin(angle))
            var centered = [Float]()
            centered.reserveCapacity(members.count * 6)
            for face in members {
                for at in (face * 6) ..< (face * 6 + 6) { centered.append(uv[at] * 2 - 1) }
            }
            // `torch.einsum("ij,nfj->nfi", rotation, centered)`: one BLAS product.
            let rotation: [Float] = [c, -s, s, c]
            var turned = [Float](repeating: 0, count: centered.count)
            centered.withUnsafeBufferPointer { x in
                rotation.withUnsafeBufferPointer { r in
                    turned.withUnsafeMutableBufferPointer { out in
                        kernels.addmm(into: out.baseAddress!, .rowMajor(x.baseAddress!, rows: centered.count / 2, columns: 2),
                                      NFKSF3DTorchKernels.Matrix(base: r.baseAddress!, rows: 2, columns: 2, rowStride: 1, columnStride: 2),
                                      beta: 0)
                    }
                }
            }
            var low = Float.greatestFiniteMagnitude, high = -Float.greatestFiniteMagnitude
            for (position, face) in members.enumerated() {
                for offset in 0 ..< 6 {
                    let value = turned[position * 6 + offset]
                    result[face * 6 + offset] = value
                    low = min(low, value)
                    high = max(high, value)
                }
            }
            for face in members {
                for at in (face * 6) ..< (face * 6 + 6) {
                    result[at] = (result[at] - low) / (high - low)
                }
            }
        }
        return result
    }

    /// `assign_faces_uv_to_atlas_index`: twice, every pair of faces in one slot whose charts overlap with
    /// positive area (the extension's coplanar triangle test, ignoring an identical chart) moves the face
    /// that lies behind the other along the slot's axis to the slot six further on, clamped at 12.
    static func atlasIndices(positions: [Float], faces: [UInt32], uv: [Float], index: [Int]) -> [Int] {
        let faceCount = faces.count / 3
        var atlas = index
        let centroids: [NFKSF3DVector] = (0 ..< faceCount).map { face in
            let a = NFKSF3DVector.at(positions, Int(faces[face * 3]))
            let b = NFKSF3DVector.at(positions, Int(faces[face * 3 + 1]))
            let c = NFKSF3DVector.at(positions, Int(faces[face * 3 + 2]))
            return NFKSF3DVector(x: (a.x + b.x + c.x) / 3, y: (a.y + b.y + c.y) / 3, z: (a.z + b.z + c.z) / 3)
        }
        let charts: [NFKSF3DChart] = (0 ..< faceCount).map { face in
            NFKSF3DChart(a: (uv[face * 6], uv[face * 6 + 1]), b: (uv[face * 6 + 2], uv[face * 6 + 3]),
                         c: (uv[face * 6 + 4], uv[face * 6 + 5]))
        }
        for offset in [0, 6] {
            var occluded = Set<Int>()
            for slot in offset ..< offset + 6 {
                let members = (0 ..< faceCount).filter { atlas[$0] == slot }
                for (first, second) in NFKSF3DChart.overlappingPairs(members, charts: charts) {
                    let slotIndex = atlas[first]
                    let normalIndex = slotIndex % 6
                    let axis = normalIndex < 2 ? 0 : normalIndex < 4 ? 1 : 2
                    let useMax = slotIndex % 2 == 1
                    let (a, b) = (centroids[first][axis], centroids[second][axis])
                    let behind = useMax ? (a < b ? first : second) : (a > b ? first : second)
                    occluded.insert(behind)
                }
            }
            for face in occluded {
                atlas[face] = min(max(atlas[face] + 6, 0), 12)
            }
        }
        return atlas
    }

    /// `_find_slice_offset_and_scale` and `_distribute_individual_uvs_in_atlas`: the first slots fill a
    /// 3 × 2 grid of thirds, the second slots a 3 × 2 grid of sixths in the lower third's left half, and the
    /// third slot's faces each a cell of a grid over the lower third's right half.
    static func placed(uv: [Float], atlas: [Int], padding: Double) -> [Float] {
        // Each coefficient folds the Python float padding in float64 before the float32 tensor op.
        let slotScale = Float(1 - 2 * padding), slotShift = Float(padding)
        let cellScale = Float(1 - 2 * padding * 0.5), cellShift = Float(padding * 0.5)
        let faceCount = atlas.count
        var u = (0 ..< faceCount * 3).map { uv[$0 * 2] }, v = (0 ..< faceCount * 3).map { uv[$0 * 2 + 1] }
        // `_handle_slice_uvs`: each second slot stretched to fill its cell, by up to a factor of two.
        for slot in 6 ..< 12 {
            let members = (0 ..< faceCount).filter { atlas[$0] == slot }
            guard !members.isEmpty else { continue }
            for values in [0, 1] {
                var low = Float.greatestFiniteMagnitude, high = -Float.greatestFiniteMagnitude
                for face in members {
                    for corner in 0 ..< 3 {
                        let value = values == 0 ? u[face * 3 + corner] : v[face * 3 + corner]
                        low = min(low, value); high = max(high, value)
                    }
                }
                let span = max(high - low, 0.5)
                for face in members {
                    for corner in 0 ..< 3 {
                        if values == 0 { u[face * 3 + corner] = (u[face * 3 + corner] - low) / span }
                        else { v[face * 3 + corner] = (v[face * 3 + corner] - low) / span }
                    }
                }
            }
        }
        for at in 0 ..< faceCount * 3 {
            u[at] = min(max(u[at] * slotScale + slotShift, 0), 1)
            v[at] = min(max(v[at] * slotScale + slotShift, 0), 1)
        }
        // `_handle_remaining_uvs`: the third slot's faces, each in its own cell.
        let remaining = (0 ..< faceCount).filter { atlas[$0] >= 12 }
        if !remaining.isEmpty {
            let ratio = 0.5 * (1.0 / 3.0)
            let multiplier = (Double(remaining.count) / ratio).squareRoot()
            let columns = Int((0.5 * multiplier).rounded(.up))
            let rowCount = Int((Double(remaining.count) / Double(columns)).rounded(.up))
            let width = 1 / Double(columns), height = 1 / Double(rowCount)
            let clip = Float(min(width, height) * 1.5)
            let columnScale = Float(1 - padding * Double(columns) * 0.5), columnShift = Float(padding * Double(columns) * 0.25)
            let rowScale = Float(1 - padding * Double(rowCount) * 0.5), rowShift = Float(padding * Double(rowCount) * 0.25)
            for (cell, face) in remaining.enumerated() {
                let us = (0 ..< 3).map { u[face * 3 + $0] }, vs = (0 ..< 3).map { v[face * 3 + $0] }
                let uLow = us.min()!, uSpan = max(us.max()! - uLow, clip)
                let vLow = vs.min()!, vSpan = max(vs.max()! - vLow, clip)
                let x = Float(cell % columns), y = Float(cell / columns)
                for corner in 0 ..< 3 {
                    var cu = (us[corner] - uLow) / uSpan, cv = (vs[corner] - vLow) / vSpan
                    cu = min(max(cu * columnScale + columnShift, 0), 1)
                    cv = min(max(cv * rowScale + rowShift, 0), 1)
                    cu = cu * Float(width) + x * Float(width)
                    cv = cv * Float(height) + y * Float(height)
                    u[face * 3 + corner] = min(max(cu * cellScale + cellShift, 0), 1)
                    v[face * 3 + corner] = min(max(cv * cellScale + cellShift, 0), 1)
                }
            }
        }
        // The offsets are Python floats stored into float32 tensors.
        let third = 1.0 / 3.0, sixth = 1.0 / 6.0
        let columnOf = [0, 1, 2, 0, 1, 2], rowOf = [0, 0, 0, 1, 1, 1]
        var result = [Float](repeating: 0, count: faceCount * 6)
        for face in 0 ..< faceCount {
            let slot = atlas[face]
            let tier = slot / 6
            let column = Double(columnOf[slot % 6]), row = Double(rowOf[slot % 6])
            let offsetX = Float(tier == 0 ? third * column : sixth * column + Double(min(tier - 1, 1)) * 0.5)
            let offsetY = Float(tier == 0 ? third * row : sixth * row + third * 2)
            let divideX: Float = slot >= 12 ? 2 : slot >= 6 ? 6 : 3
            let divideY: Float = slot >= 12 ? 3 : slot >= 6 ? 6 : 3
            for corner in 0 ..< 3 {
                result[(face * 3 + corner) * 2] = u[face * 3 + corner] / divideX + offsetX
                result[(face * 3 + corner) * 2 + 1] = v[face * 3 + corner] / divideY + offsetY
            }
        }
        return result
    }
}

// MARK: - The chart overlap test

/// One face's chart in the atlas, with the extension's `triangle_triangle_intersection`: degenerate
/// charts never overlap; two charts in one plane overlap when, in the first chart's own frame, the
/// orientation cases of Guigue and Devillers find them crossing and the polygon `clipTriangle` cuts from
/// them has an area of at least 1e-10. Every point lies at z = 0, so the extension's 3-D orientations are
/// all zero and only its coplanar branch runs.
struct NFKSF3DChart {
    typealias Point = (Float, Float)
    let a: Point
    let b: Point
    let c: Point

    static let epsilon: Float = 1e-7

    var low: Point { (min(a.0, b.0, c.0), min(a.1, b.1, c.1)) }
    var high: Point { (max(a.0, b.0, c.0), max(a.1, b.1, c.1)) }

    func same(_ other: NFKSF3DChart) -> Bool {
        a == other.a && b == other.b && c == other.c
    }

    /// The pairs `(first, second)`, first < second, of faces among `members` whose charts the extension's
    /// per-face queries report overlapping: each face queries the slot's BVH (`NFKSF3DChartTree`), which
    /// reaches a chart only through the node boxes its triangle-box test admits, so a pair is found only
    /// in a direction whose box tests pass and whose chart test does.
    static func overlappingPairs(_ members: [Int], charts: [NFKSF3DChart]) -> [(Int, Int)] {
        guard members.count > 1 else { return [] }
        let tree = NFKSF3DChartTree(members: members, charts: charts)
        var pairs = Set<UInt64>()
        for face in members {
            for other in tree.intersecting(charts[face]) where other != face {
                pairs.insert(UInt64(min(face, other)) << 32 | UInt64(max(face, other)))
            }
        }
        return pairs.sorted().map { (Int($0 >> 32), Int($0 & 0xFFFF_FFFF)) }
    }

    /// `triangle_aabb_intersection`: whether the chart touches the box `low ... high`, by the
    /// extension's vertex, edge-crossing, and box-in-triangle tests.
    func touches(low: Point, high: Point) -> Bool {
        let (boxLeft, boxRight, boxTop, boxBottom) = (low.0, high.0, low.1, high.1)
        func code(_ p: Point) -> Int {
            (p.0 > boxLeft ? 1 : 0) | (p.1 > boxTop ? 2 : 0) | (p.0 > boxRight ? 4 : 0) | (p.1 > boxBottom ? 8 : 0)
        }
        let codes = [code(a), code(b), code(c)]
        if codes.contains(3) {
            return true
        }
        func edge(_ p: Point, _ q: Point, _ crossing: Int) -> Bool {
            guard crossing != 0 else { return false }
            if q.0 != p.0 {
                let m = (q.1 - p.1) / (q.0 - p.0)
                let intercept = p.1 - (m * p.0)
                if crossing & 1 != 0 { let s = m * boxLeft + intercept; if s >= boxTop && s <= boxBottom { return true } }
                if crossing & 2 != 0 { let s = (boxTop - intercept) / m; if s >= boxLeft && s <= boxRight { return true } }
                if crossing & 4 != 0 { let s = m * boxRight + intercept; if s >= boxTop && s <= boxBottom { return true } }
                if crossing & 8 != 0 { let s = (boxBottom - intercept) / m; if s >= boxLeft && s <= boxRight { return true } }
                return false
            }
            return boxLeft == p.0 || boxRight == p.0 || (p.0 > boxLeft && p.0 < boxRight)
        }
        if edge(a, b, codes[0] ^ codes[1]) || edge(b, c, codes[1] ^ codes[2]) || edge(a, c, codes[0] ^ codes[2]) {
            return true
        }
        let left = min(a.0, min(b.0, c.0)), top = min(a.1, min(b.1, c.1))
        let right = max(a.0, max(b.0, c.0)), bottom = max(a.1, max(b.1, c.1))
        guard left <= boxLeft && right >= boxRight && top <= boxTop && bottom >= boxBottom else { return false }
        let v0x = c.0 - a.0, v0y = c.1 - a.1, v1x = b.0 - a.0, v1y = b.1 - a.1
        let dot00 = v0x * v0x + v0y * v0y, dot01 = v0x * v1x + v0y * v1y, dot11 = v1x * v1x + v1y * v1y
        let inverse = 1 / (dot00 * dot11 - dot01 * dot01)
        for corner in [(boxLeft, boxTop), (boxLeft, boxBottom), (boxRight, boxBottom), (boxRight, boxTop)] {
            let v2x = corner.0 - a.0, v2y = corner.1 - a.1
            let dot02 = v0x * v2x + v0y * v2y, dot12 = v1x * v2x + v1y * v2y
            let u = (dot11 * dot02 - dot01 * dot12) * inverse
            let v = (dot00 * dot12 - dot01 * dot02) * inverse
            if u >= 0 && v >= 0 && u + v <= 1 {
                return true
            }
        }
        return false
    }

    func overlaps(_ other: NFKSF3DChart) -> Bool {
        let first = NFKSF3DTriangle(self), second = NFKSF3DTriangle(other)
        if first.degenerate || second.degenerate {
            return false
        }
        guard let polygon = NFKSF3DTriangle.coplanarIntersection(first, second) else { return false }
        let area = NFKSF3DTriangle.polygonArea(polygon)
        return area >= 1e-10 && area.isFinite
    }
}

/// The extensions' axis-aligned box: empty as `{FLT_MAX, FLT_MAX}`–`{FLT_MIN, FLT_MIN}`, so an empty box's
/// area overflows to infinity and a bin with no triangles costs NaN, which no plane cost beats.
struct NFKSF3DBox {
    var low: (Float, Float) = (.greatestFiniteMagnitude, .greatestFiniteMagnitude)
    var high: (Float, Float) = (Float.leastNormalMagnitude, Float.leastNormalMagnitude)

    mutating func grow(_ p: (Float, Float)) {
        low = (Swift.min(low.0, p.0), Swift.min(low.1, p.1))
        high = (Swift.max(high.0, p.0), Swift.max(high.1, p.1))
    }

    mutating func grow(_ other: NFKSF3DBox) {
        if other.low.0 != .greatestFiniteMagnitude {
            grow(other.low)
            grow(other.high)
        }
    }

    var area: Float { (high.0 - low.0) * (high.1 - low.1) }

    func contains(_ p: (Float, Float)) -> Bool {
        p.0 >= low.0 && p.0 <= high.0 && p.1 >= low.1 && p.1 <= high.1
    }
}

/// The binned surface-area-heuristic BVH both extensions build over 2-D triangles, breadth first: eight
/// bins along each axis of the node's centroid box, the cheapest of the seven planes, a node kept a leaf
/// when no plane beats its own cost or the partition leaves a side empty. The two differ only where noted:
/// the unwrapper skips an axis whose centroid span is below 1e-8, and the baker takes the centroid as
/// `(a + b + c) · 0.3333`.
struct NFKSF3DTree {
    struct Node {
        var box = NFKSF3DBox()
        var start = 0
        var end = 0
        var left = -1
        var right = -1
        var isLeaf: Bool { left == -1 && right == -1 }
    }

    var nodes = [Node]()
    var order: [Int]
    let corners: [[(Float, Float)]]
    let centroids: [(Float, Float)]

    init(corners: [[(Float, Float)]], centroids: [(Float, Float)], skipsNarrowAxes: Bool) {
        self.corners = corners
        self.centroids = centroids
        order = Array(0 ..< corners.count)
        nodes.append(Node(start: 0, end: corners.count))
        var queue = [0]
        var head = 0
        while head < queue.count {
            let index = queue[head]
            head += 1
            let centroidBox = bounds(index)
            var axis = 0, position = 0
            let cost = bestSplit(index, centroidBox: centroidBox, skipsNarrowAxes: skipsNarrowAxes, axis: &axis, position: &position)
            let node = nodes[index]
            if cost >= Float(node.end - node.start) * node.box.area {
                continue
            }
            let low = axis == 0 ? centroidBox.low.0 : centroidBox.low.1
            let high = axis == 0 ? centroidBox.high.0 : centroidBox.high.1
            let scale = Float(8) / (high - low)
            var i = node.start, j = node.end - 1
            while i <= j {
                let centroid = centroids[order[i]]
                let bin = Swift.min(7, Int(((axis == 0 ? centroid.0 : centroid.1) - low) * scale))
                if bin < position {
                    i += 1
                } else {
                    order.swapAt(i, j)
                    j -= 1
                }
            }
            let leftCount = i - node.start
            if leftCount == 0 || leftCount == node.end - node.start {
                continue
            }
            let leftIndex = nodes.count
            nodes.append(Node(start: node.start, end: i))
            nodes.append(Node(start: i, end: node.end))
            nodes[index].left = leftIndex
            nodes[index].right = leftIndex + 1
            queue.append(leftIndex)
            queue.append(leftIndex + 1)
        }
    }

    /// `update_node_bounds`: the node's box over its triangles' corners, and its centroid box.
    private mutating func bounds(_ index: Int) -> NFKSF3DBox {
        var box = NFKSF3DBox(), centroidBox = NFKSF3DBox()
        for i in nodes[index].start ..< nodes[index].end {
            let triangle = order[i]
            for corner in corners[triangle] { box.grow(corner) }
            centroidBox.grow(centroids[triangle])
        }
        nodes[index].box = box
        return centroidBox
    }

    private func bestSplit(_ index: Int, centroidBox: NFKSF3DBox, skipsNarrowAxes: Bool, axis bestAxis: inout Int,
                           position bestPosition: inout Int) -> Float {
        var best = Float.greatestFiniteMagnitude
        let node = nodes[index]
        for axis in 0 ..< 2 {
            let low = axis == 0 ? centroidBox.low.0 : centroidBox.low.1
            let high = axis == 0 ? centroidBox.high.0 : centroidBox.high.1
            if low == high || (skipsNarrowAxes && high - low < 1e-8) {
                continue
            }
            let scale = Float(8) / (high - low)
            var boxes = [NFKSF3DBox](repeating: NFKSF3DBox(), count: 8)
            var counts = [Int](repeating: 0, count: 8)
            for i in node.start ..< node.end {
                let triangle = order[i]
                let centroid = centroids[triangle]
                let bin = Swift.min(7, Int(((axis == 0 ? centroid.0 : centroid.1) - low) * scale))
                counts[bin] += 1
                for corner in corners[triangle] { boxes[bin].grow(corner) }
            }
            var leftArea = [Float](repeating: 0, count: 7), rightArea = [Float](repeating: 0, count: 7)
            var leftBox = NFKSF3DBox(), rightBox = NFKSF3DBox()
            var leftSum = 0, rightSum = 0
            for i in 0 ..< 7 {
                leftSum += counts[i]
                leftBox.grow(boxes[i])
                leftArea[i] = Float(leftSum) * leftBox.area
                rightSum += counts[7 - i]
                rightBox.grow(boxes[7 - i])
                rightArea[6 - i] = Float(rightSum) * rightBox.area
            }
            for i in 0 ..< 7 {
                let cost = leftArea[i] + rightArea[i]
                if cost < best {
                    bestAxis = axis
                    bestPosition = i + 1
                    best = cost
                }
            }
        }
        return best
    }
}

/// The unwrapper's per-slot BVH and its `Intersect`: the root's triangles are always reached; a child
/// is entered when the query chart touches its box (`triangle_aabb_intersection`), and a leaf's chart
/// is reported when it differs from the query and the query overlaps it.
struct NFKSF3DChartTree {
    let tree: NFKSF3DTree
    let members: [Int]
    let charts: [NFKSF3DChart]

    init(members: [Int], charts: [NFKSF3DChart]) {
        self.members = members
        self.charts = charts
        let corners = members.map { [charts[$0].a, charts[$0].b, charts[$0].c] }
        let centroids = members.map { face -> (Float, Float) in
            let chart = charts[face]
            return ((chart.a.0 + chart.b.0 + chart.c.0) / 3, (chart.a.1 + chart.b.1 + chart.c.1) / 3)
        }
        tree = NFKSF3DTree(corners: corners, centroids: centroids, skipsNarrowAxes: true)
    }

    func intersecting(_ query: NFKSF3DChart) -> [Int] {
        var found = [Int]()
        var stack = [0]
        while let index = stack.popLast() {
            let node = tree.nodes[index]
            if node.isLeaf {
                for i in node.start ..< node.end {
                    let face = members[tree.order[i]]
                    let chart = charts[face]
                    if chart.same(query) { continue }
                    if query.overlaps(chart) { found.append(face) }
                }
                continue
            }
            let right = tree.nodes[node.right].box, left = tree.nodes[node.left].box
            if query.touches(low: right.low, high: right.high) { stack.append(node.right) }
            if query.touches(low: left.low, high: left.high) { stack.append(node.left) }
        }
        return found
    }
}

/// The extension's `Triangle` of three `uv_float3` points and the geometry its coplanar test runs.
struct NFKSF3DTriangle {
    var a: NFKSF3DVector
    var b: NFKSF3DVector
    var c: NFKSF3DVector

    init(_ chart: NFKSF3DChart) {
        a = NFKSF3DVector(x: chart.a.0, y: chart.a.1, z: 0)
        b = NFKSF3DVector(x: chart.b.0, y: chart.b.1, z: 0)
        c = NFKSF3DVector(x: chart.c.0, y: chart.c.1, z: 0)
    }

    init(a: NFKSF3DVector, b: NFKSF3DVector, c: NFKSF3DVector) {
        self.a = a; self.b = b; self.c = c
    }

    var degenerate: Bool {
        let cross = (a - b).cross(a - c)
        return abs(cross.x) < NFKSF3DChart.epsilon && abs(cross.y) < NFKSF3DChart.epsilon && abs(cross.z) < NFKSF3DChart.epsilon
    }

    /// The extension's `normalize`: `v / ‖v‖`, with no floor.
    static func unit(_ v: NFKSF3DVector) -> NFKSF3DVector {
        let length = (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
        return NFKSF3DVector(x: v.x / length, y: v.y / length, z: v.z / length)
    }

    static func orient(_ p: NFKSF3DVector, _ q: NFKSF3DVector, _ r: NFKSF3DVector) -> Int {
        let determinant = p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y)
        if determinant < -NFKSF3DChart.epsilon { return -1 }
        if determinant > NFKSF3DChart.epsilon { return 1 }
        return 0
    }

    mutating func permuteLeft() { (a, b, c) = (b, c, a) }
    mutating func permuteRight() { (a, b, c) = (c, a, b) }
    mutating func counterClockwise() {
        if Self.orient(a, b, c) < 0 { swap(&b, &c) }
    }

    static func intersectionTypeR1(_ t1: Self, _ t2: Self) -> Bool {
        let (p1, q1, r1, p2, r2) = (t1.a, t1.b, t1.c, t2.a, t2.c)
        if orient(r2, p2, q1) >= 0 {
            if orient(r2, p1, q1) >= 0 {
                if orient(p1, p2, q1) >= 0 { return true }
                if orient(p1, p2, r1) >= 0 && orient(q1, r1, p2) >= 0 { return true }
            }
        } else if orient(r2, p2, r1) >= 0 && orient(q1, r1, r2) >= 0 && orient(p1, p2, r1) >= 0 {
            return true
        }
        return false
    }

    static func intersectionTypeR2(_ t1: Self, _ t2: Self) -> Bool {
        let (p1, q1, r1, p2, q2, r2) = (t1.a, t1.b, t1.c, t2.a, t2.b, t2.c)
        if orient(r2, p2, q1) >= 0 {
            if orient(q2, r2, q1) >= 0 {
                if orient(p1, p2, q1) >= 0 {
                    if orient(p1, q2, q1) <= 0 { return true }
                } else if orient(p1, p2, r1) >= 0 && orient(r2, p2, r1) <= 0 {
                    return true
                }
            } else if orient(p1, q2, q1) <= 0 && orient(q2, r2, r1) >= 0 && orient(q1, r1, q2) >= 0 {
                return true
            }
        } else if orient(r2, p2, r1) >= 0 {
            if orient(q1, r1, r2) >= 0 {
                if orient(r1, p1, p2) >= 0 { return true }
            } else if orient(q1, r1, q2) >= 0 && orient(q2, r2, r1) >= 0 {
                return true
            }
        }
        return false
    }

    /// `coplanarIntersect` with a target: both triangles moved into the first's own frame, the
    /// orientation cases, and on a crossing the clipped polygon moved back. Nil when they do not cross.
    static func coplanarIntersection(_ first: Self, _ second: Self) -> [NFKSF3DVector]? {
        var t1 = first, t2 = second
        let normal = unit((t1.b - t1.a).cross(t1.c - t1.a))
        let u = unit(t1.a - t1.b)
        let v = normal.cross(u)
        let ua = u + t1.a, va = v + t1.a, na = normal + t1.a
        var matrix = NFKSF3DMatrix4(rows: [[t1.a.x, ua.x, va.x, na.x], [t1.a.y, ua.y, va.y, na.y],
                                           [t1.a.z, ua.z, va.z, na.z], [1, 1, 1, 1]])
        let affine = NFKSF3DMatrix4(rows: [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]])
        _ = matrix.invert()
        matrix = affine * matrix
        t1 = Self(a: matrix.apply(t1.a), b: matrix.apply(t1.b), c: matrix.apply(t1.c))
        t2 = Self(a: matrix.apply(t2.a), b: matrix.apply(t2.b), c: matrix.apply(t2.c))
        t1.counterClockwise()
        t2.counterClockwise()
        let p1 = t1.a
        let (forward, middle, back) = (orient(t2.a, t2.b, p1), orient(t2.b, t2.c, p1), orient(t2.c, t2.a, p1))
        var crossing: Bool
        switch (forward >= 0, middle >= 0, back >= 0) {
        case (true, true, true): crossing = true
        case (true, true, false): crossing = intersectionTypeR1(t1, t2)
        case (true, false, true): t2.permuteRight(); crossing = intersectionTypeR1(t1, t2)
        case (true, false, false): crossing = intersectionTypeR2(t1, t2)
        case (false, true, true): t2.permuteLeft(); crossing = intersectionTypeR1(t1, t2)
        case (false, true, false): t2.permuteLeft(); crossing = intersectionTypeR2(t1, t2)
        case (false, false, true): t2.permuteRight(); crossing = intersectionTypeR2(t1, t2)
        case (false, false, false): return nil
        }
        guard crossing else { return nil }
        var polygon = clipped(t1, t2)
        _ = matrix.invert()
        polygon = polygon.map { matrix.apply($0) }
        return polygon
    }

    /// `linesIntersect2d`.
    static func lineIntersection(_ a1: NFKSF3DVector, _ b1: NFKSF3DVector, _ a2: NFKSF3DVector, _ b2: NFKSF3DVector) -> NFKSF3DVector {
        let dx1 = a1.x - b1.x, dx2 = a2.x - b2.x, dy1 = a1.y - b1.y, dy2 = a2.y - b2.y
        let d = dx1 * dy2 - dx2 * dy1
        let n1 = a1.x * b1.y - a1.y * b1.x, n2 = a2.x * b2.y - a2.y * b2.x
        return NFKSF3DVector(x: (n1 * dx2 - n2 * dx1) / d, y: (n1 * dy2 - n2 * dy1) / d, z: 0)
    }

    /// `clipTriangle`: the second triangle clipped by the first's three edges in turn, then points within
    /// 1e-6 of an earlier one dropped.
    static func clipped(_ t1: Self, _ t2: Self) -> [NFKSF3DVector] {
        let clip = [t1.a, t1.b, t1.c]
        var output = [t2.a, t2.b, t2.c]
        for i in 0 ..< 3 {
            let previous = (i + 2) % 3
            let input = output
            output = []
            let orients = input.map { orient(clip[previous], clip[i], $0) }
            for j in input.indices {
                let before = (j - 1 + input.count) % input.count
                if orients[j] >= 0 {
                    if orients[before] < 0 {
                        output.append(lineIntersection(clip[previous], clip[i], input[before], input[j]))
                    }
                    output.append(input[j])
                } else if orients[before] >= 0 {
                    output.append(lineIntersection(clip[previous], clip[i], input[before], input[j]))
                }
            }
        }
        var kept = [NFKSF3DVector]()
        for point in output {
            let near = kept.contains { other in
                let dx = Double(point.x - other.x), dy = Double(point.y - other.y), dz = Double(point.z - other.z)
                return Float((dx * dx + dy * dy + dz * dz).squareRoot()) <= 1e-6
            }
            if !near { kept.append(point) }
        }
        return kept
    }

    /// `polygon_area`: half the magnitude of the summed cross products of consecutive points.
    static func polygonArea(_ polygon: [NFKSF3DVector]) -> Float {
        guard polygon.count >= 3 else { return 0 }
        var normal = NFKSF3DVector(x: 0, y: 0, z: 0)
        for index in polygon.indices {
            normal = normal + polygon[index].cross(polygon[(index + 1) % polygon.count])
        }
        return (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).squareRoot() / 2
    }
}

/// The extension's row-major `Matrix4`: float32 storage, a float32 product, an inverse by cofactors in
/// float64, and a projective point transform.
struct NFKSF3DMatrix4 {
    var m: [[Float]]

    init(rows: [[Float]]) { m = rows }

    static func * (lhs: Self, rhs: Self) -> Self {
        var out = [[Float]](repeating: [Float](repeating: 0, count: 4), count: 4)
        for row in 0 ..< 4 {
            for column in 0 ..< 4 {
                out[row][column] = lhs.m[row][0] * rhs.m[0][column] + lhs.m[row][1] * rhs.m[1][column]
                    + lhs.m[row][2] * rhs.m[2][column] + lhs.m[row][3] * rhs.m[3][column]
            }
        }
        return Self(rows: out)
    }

    /// Inverts in place; leaves the matrix as it was and returns false when the determinant's magnitude
    /// is below 1e-6.
    mutating func invert() -> Bool {
        let a = (0 ..< 16).map { Double(m[$0 / 4][$0 % 4]) }
        var inv = [Double](repeating: 0, count: 16)
        inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10]
        inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10]
        inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9]
        inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9]
        inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10]
        inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10]
        inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9]
        inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9]
        inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6]
        inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6]
        inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5]
        inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5]
        inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6]
        inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6]
        inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5]
        inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5]
        var determinant = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12]
        if abs(determinant) < 1e-6 {
            return false
        }
        determinant = 1.0 / determinant
        for index in 0 ..< 16 {
            m[index / 4][index % 4] = Float(inv[index] * determinant)
        }
        return true
    }

    /// `apply_matrix4`: the affine part, divided by w when |w| exceeds 1e-7.
    func apply(_ v: NFKSF3DVector) -> NFKSF3DVector {
        var x = v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2] + m[0][3]
        var y = v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2] + m[1][3]
        var z = v.x * m[2][0] + v.y * m[2][1] + v.z * m[2][2] + m[2][3]
        let w = v.x * m[3][0] + v.y * m[3][1] + v.z * m[3][2] + m[3][3]
        if abs(w) > NFKSF3DChart.epsilon {
            x /= w; y /= w; z /= w
        }
        return NFKSF3DVector(x: x, y: y, z: z)
    }
}

// MARK: - LAPACK

/// The LAPACK routines torch's CPU `linalg.qr` and `linalg.svd` call (Accelerate's `sgeqrf_`,
/// `sorgqr_`, and `sgesdd_`), resolved at run time so the principal axes take torch's signs.
enum NFKSF3DLAPACK {
    typealias GEQRF = @convention(c) (UnsafePointer<Int32>, UnsafePointer<Int32>, UnsafeMutablePointer<Float>, UnsafePointer<Int32>,
                                      UnsafeMutablePointer<Float>, UnsafeMutablePointer<Float>, UnsafePointer<Int32>,
                                      UnsafeMutablePointer<Int32>) -> Int32
    typealias ORGQR = @convention(c) (UnsafePointer<Int32>, UnsafePointer<Int32>, UnsafePointer<Int32>, UnsafeMutablePointer<Float>,
                                      UnsafePointer<Int32>, UnsafePointer<Float>, UnsafeMutablePointer<Float>, UnsafePointer<Int32>,
                                      UnsafeMutablePointer<Int32>) -> Int32
    typealias GESDD = @convention(c) (UnsafePointer<CChar>, UnsafePointer<Int32>, UnsafePointer<Int32>, UnsafeMutablePointer<Float>,
                                      UnsafePointer<Int32>, UnsafeMutablePointer<Float>, UnsafeMutablePointer<Float>, UnsafePointer<Int32>,
                                      UnsafeMutablePointer<Float>, UnsafePointer<Int32>, UnsafeMutablePointer<Float>, UnsafePointer<Int32>,
                                      UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>) -> Int32

    static func symbol<T>(_ name: String, as type: T.Type) throws -> T {
        guard let address = dlsym(UnsafeMutableRawPointer(bitPattern: -2), name) else {
            throw NFKMLXError.unsupportedConfiguration("Accelerate does not export \(name)")
        }
        return unsafeBitCast(address, to: type)
    }

    /// The reduced Q `[rows, columns]` of a row-major matrix, as `torch.linalg.qr(x).Q`.
    static func reducedQ(rowMajor values: [Float], rows: Int, columns: Int) throws -> [Float] {
        let geqrf = try symbol("sgeqrf_", as: GEQRF.self)
        let orgqr = try symbol("sorgqr_", as: ORGQR.self)
        var a = [Float](repeating: 0, count: rows * columns)
        for row in 0 ..< rows {
            for column in 0 ..< columns { a[column * rows + row] = values[row * columns + column] }
        }
        var m = Int32(rows), n = Int32(columns), k = Int32(min(rows, columns)), lda = Int32(max(1, rows)), info: Int32 = 0
        var tau = [Float](repeating: 0, count: Int(k))
        var query: Float = 0, lwork: Int32 = -1
        _ = geqrf(&m, &n, &a, &lda, &tau, &query, &lwork, &info)
        lwork = max(1, Int32(query))
        var work = [Float](repeating: 0, count: Int(lwork))
        _ = geqrf(&m, &n, &a, &lda, &tau, &work, &lwork, &info)
        guard info == 0 else { throw NFKMLXError.unsupportedConfiguration("sgeqrf failed: \(info)") }
        lwork = -1
        _ = orgqr(&m, &k, &k, &a, &lda, &tau, &query, &lwork, &info)
        lwork = max(1, Int32(query))
        work = [Float](repeating: 0, count: Int(lwork))
        _ = orgqr(&m, &k, &k, &a, &lda, &tau, &work, &lwork, &info)
        guard info == 0 else { throw NFKMLXError.unsupportedConfiguration("sorgqr failed: \(info)") }
        var q = [Float](repeating: 0, count: rows * Int(k))
        for row in 0 ..< rows {
            for column in 0 ..< Int(k) { q[row * Int(k) + column] = a[column * rows + row] }
        }
        return q
    }

    /// V `[columns, rank]` row-major of a row-major matrix, as `torch.linalg.svd(x, full_matrices: false)`'s
    /// `Vh.mH`.
    static func rightSingularVectors(rowMajor values: [Float], rows: Int, columns: Int) throws -> [Float] {
        let gesdd = try symbol("sgesdd_", as: GESDD.self)
        var a = [Float](repeating: 0, count: rows * columns)
        for row in 0 ..< rows {
            for column in 0 ..< columns { a[column * rows + row] = values[row * columns + column] }
        }
        let rank = min(rows, columns)
        var job = CChar(UInt8(ascii: "S"))
        var m = Int32(rows), n = Int32(columns), lda = Int32(max(1, rows)), ldu = Int32(max(1, rows)), ldvt = Int32(max(1, rank))
        var s = [Float](repeating: 0, count: rank)
        var u = [Float](repeating: 0, count: rows * rank)
        var vt = [Float](repeating: 0, count: rank * columns)
        var iwork = [Int32](repeating: 0, count: 8 * rank)
        var info: Int32 = 0, query: Float = 0, lwork: Int32 = -1
        _ = gesdd(&job, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &query, &lwork, &iwork, &info)
        lwork = max(1, Int32(query))
        var work = [Float](repeating: 0, count: Int(lwork))
        _ = gesdd(&job, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &iwork, &info)
        guard info == 0 else { throw NFKMLXError.unsupportedConfiguration("sgesdd failed: \(info)") }
        var v = [Float](repeating: 0, count: columns * rank)
        for row in 0 ..< rank {
            for column in 0 ..< columns { v[column * rank + row] = vt[column * rank + row] }
        }
        return v
    }
}

// MARK: - Baking

/// `texture_baker`'s CPU rasterizer and interpolator, and the island padding.
enum NFKSF3DBaker {

    /// Each texel's barycentric weights and triangle, or −1 where no triangle covers it. Texel `(row, column)`
    /// samples the atlas at `(column / side, 1 − row / side)`, its corner, through the extension's BVH: the
    /// root's triangles are always tried, a child is entered when its box holds the point (left before
    /// right), and the first triangle whose `barycentric_coordinates` test passes is taken.
    static func rasterize(uv: [Float], faces: [UInt32], side: Int) -> (weights: [Float], triangle: [Int32]) {
        let faceCount = faces.count / 3
        let corners: [[(Float, Float)]] = (0 ..< faceCount).map { face in
            (0 ..< 3).map { corner in
                let vertex = Int(faces[face * 3 + corner])
                return (uv[vertex * 2], uv[vertex * 2 + 1])
            }
        }
        let centroids = corners.map { c in ((c[0].0 + c[1].0 + c[2].0) * Float(0.3333), (c[0].1 + c[1].1 + c[2].1) * Float(0.3333)) }
        let tree = NFKSF3DTree(corners: corners, centroids: centroids, skipsNarrowAxes: false)
        var weights = [Float](repeating: 0, count: side * side * 3)
        var triangle = [Int32](repeating: -1, count: side * side)
        var stack = [Int]()
        stack.reserveCapacity(64)
        for row in 0 ..< side {
            let y = 1 - min(max(Float(row) / Float(side), 0), 1)
            for column in 0 ..< side {
                let x = min(max(Float(column) / Float(side), 0), 1)
                stack.removeAll(keepingCapacity: true)
                stack.append(0)
                search: while let index = stack.popLast() {
                    let node = tree.nodes[index]
                    if node.isLeaf {
                        for i in node.start ..< node.end {
                            let face = tree.order[i]
                            let c = corners[face]
                            let e12 = (c[1].0 - c[0].0, c[1].1 - c[0].1), e13 = (c[2].0 - c[0].0, c[2].1 - c[0].1)
                            let p = (x - c[0].0, y - c[0].1)
                            let d00 = e12.0 * e12.0 + e12.1 * e12.1
                            let d01 = e12.0 * e13.0 + e12.1 * e13.1
                            let d11 = e13.0 * e13.0 + e13.1 * e13.1
                            let d20 = p.0 * e12.0 + p.1 * e12.1
                            let d21 = p.0 * e13.0 + p.1 * e13.1
                            let denominator = d00 * d11 - d01 * d01
                            let v = (d11 * d20 - d01 * d21) / denominator
                            let w = (d00 * d21 - d01 * d20) / denominator
                            let u = 1 - v - w
                            if v >= 0 && w >= 0 && v + w <= 1 {
                                let texel = row * side + column
                                triangle[texel] = Int32(face)
                                weights[texel * 3] = u; weights[texel * 3 + 1] = v; weights[texel * 3 + 2] = w
                                break search
                            }
                        }
                        continue
                    }
                    if tree.nodes[node.right].box.contains((x, y)) { stack.append(node.right) }
                    if tree.nodes[node.left].box.contains((x, y)) { stack.append(node.left) }
                }
            }
        }
        return (weights, triangle)
    }

    /// `interpolate`: the three-float attribute of each covered texel, `v1 · u + v2 · v + v3 · w`, three
    /// floats a covered texel in row-major texel order.
    static func interpolate(_ attribute: [Float], faces: [UInt32], weights: [Float], triangle: [Int32]) -> [Float] {
        var out = [Float]()
        out.reserveCapacity(triangle.count / 2 * 3)
        for texel in triangle.indices where triangle[texel] >= 0 {
            let face = Int(triangle[texel])
            let (u, v, w) = (weights[texel * 3], weights[texel * 3 + 1], weights[texel * 3 + 2])
            let i0 = Int(faces[face * 3]), i1 = Int(faces[face * 3 + 1]), i2 = Int(faces[face * 3 + 2])
            for axis in 0 ..< 3 {
                out.append(attribute[i0 * 3 + axis] * u + attribute[i1 * 3 + axis] * v + attribute[i2 * 3 + axis] * w)
            }
        }
        return out
    }

    /// `dilate_fill` of a three-channel texture `[side, side, 3]` whose texels outside `mask` are zero:
    /// `iterations` times, the mask grows by a 3 × 3 max, and each new texel takes the sum, over the 3 × 3
    /// windows centered inside the image that hold it, of the window's mean over its masked texels,
    /// divided by the count of masked texels around it.
    static func dilated(_ texture: [Float], mask: [Bool], side: Int, iterations: Int) -> [Float] {
        var image = texture
        var current = mask.map { $0 ? Float(1) : 0 }
        for _ in 0 ..< iterations {
            var grown = current
            for row in 0 ..< side {
                for column in 0 ..< side where current[row * side + column] == 0 {
                    var any = false
                    for dy in -1 ... 1 where !any {
                        for dx in -1 ... 1 {
                            let y = row + dy, x = column + dx
                            if y >= 0 && y < side && x >= 0 && x < side && current[y * side + x] > 0 { any = true; break }
                        }
                    }
                    if any { grown[row * side + column] = 1 }
                }
            }
            // The windows of `F.unfold` without padding: centers 1 ... side − 2 on each axis.
            let inner = side - 2
            var means = [Float](repeating: 0, count: max(0, inner * inner) * 3)
            if inner > 0 {
                for cy in 1 ... inner {
                    for cx in 1 ... inner {
                        var sums: (Float, Float, Float) = (0, 0, 0)
                        var count: Float = 0
                        for ky in -1 ... 1 {
                            for kx in -1 ... 1 {
                                let at = (cy + ky) * side + cx + kx
                                sums.0 += image[at * 3]; sums.1 += image[at * 3 + 1]; sums.2 += image[at * 3 + 2]
                                count += current[at]
                            }
                        }
                        let divisor = max(count, 1)
                        let index = ((cy - 1) * inner + cx - 1) * 3
                        means[index] = sums.0 / divisor; means[index + 1] = sums.1 / divisor; means[index + 2] = sums.2 / divisor
                    }
                }
            }
            // `F.fold`: each kernel offset in turn adds its windows' contributions.
            var folded = [Float](repeating: 0, count: side * side * 3)
            if inner > 0 {
                for ky in -1 ... 1 {
                    for kx in -1 ... 1 {
                        for cy in 1 ... inner {
                            for cx in 1 ... inner {
                                let at = (cy + ky) * side + cx + kx
                                let index = ((cy - 1) * inner + cx - 1) * 3
                                let weight = grown[at]
                                folded[at * 3] += means[index] * weight
                                folded[at * 3 + 1] += means[index + 1] * weight
                                folded[at * 3 + 2] += means[index + 2] * weight
                            }
                        }
                    }
                }
            }
            for row in 0 ..< side {
                for column in 0 ..< side {
                    let at = row * side + column
                    guard grown[at] - current[at] == 1 else { continue }
                    var neighbors: Float = 0
                    for dy in -1 ... 1 {
                        for dx in -1 ... 1 {
                            let y = row + dy, x = column + dx
                            if y >= 0 && y < side && x >= 0 && x < side { neighbors += grown[y * side + x] }
                        }
                    }
                    let divisor = max(neighbors, 1)
                    image[at * 3] = folded[at * 3] / divisor
                    image[at * 3 + 1] = folded[at * 3 + 1] / divisor
                    image[at * 3 + 2] = folded[at * 3 + 2] / divisor
                }
            }
            current = grown
        }
        return image
    }

    /// `float32_to_uint8_np`: `floor(256 · x + d)` clipped to 0...255, `d` a uniform draw in −0.5...0.5 a
    /// texel shared by its channels, times `ditherMask` where one is given.
    static func quantized(_ texture: [Float], ditherMask: [Bool]?, generator: inout NFKSF3DDither) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: texture.count)
        for texel in 0 ..< texture.count / 3 {
            var dither = generator.next() - 0.5
            if let ditherMask, !ditherMask[texel] {
                dither = 0
            }
            for channel in 0 ..< 3 {
                let value = (256 * texture[texel * 3 + channel] + dither).rounded(.down)
                out[texel * 3 + channel] = UInt8(min(max(value, 0), 255))
            }
        }
        return out
    }
}

/// A seeded uniform generator for the dither (SplitMix64). The release draws from NumPy's unseeded global
/// generator; a seed makes the texture reproducible.
struct NFKSF3DDither {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> Float {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Float(z >> 40) / Float(1 << 24)
    }
}

// MARK: - Fields

/// The fields the surface stage reads at world positions `[count, 3]`: the density (after its exponential)
/// and vertex offset of the surface, and the albedo (after its sigmoid) and unit perturbed normal of the bake.
protocol NFKSF3DFields {
    func geometry(_ points: [Float]) throws -> (density: [Float], vertexOffset: [Float])
    func surface(_ points: [Float]) throws -> (albedo: [Float], normal: [Float])
}

/// The MLX network's fields at one scene code `[3, side, side, planeChannels]`, decoded on the GPU in
/// chunks.
struct NFKSF3DNetworkFields: NFKSF3DFields {
    let net: NFKMLXStableFast3DNet
    let sceneCode: MLXArray

    private func decoded(_ points: [Float], _ decode: (MLXArray) -> (MLXArray, MLXArray)) -> ([Float], [Float]) {
        let count = points.count / 3
        var first = [Float](), second = [Float]()
        var start = 0
        while start < count {
            let end = min(start + (1 << 17), count)
            let (a, b) = decode(MLXArray(Array(points[(start * 3) ..< (end * 3)])).reshaped([end - start, 3]))
            eval(a, b)
            first += a.asArray(Float.self)
            second += b.asArray(Float.self)
            start = end
        }
        return (first, second)
    }

    func geometry(_ points: [Float]) throws -> (density: [Float], vertexOffset: [Float]) {
        decoded(points) { net.geometry($0, sceneCode: sceneCode) }
    }

    func surface(_ points: [Float]) throws -> (albedo: [Float], normal: [Float]) {
        decoded(points) { net.surface($0, sceneCode: sceneCode) }
    }
}

/// The reference-arithmetic network's fields at one scene code `[3, planeChannels, side, side]`, every
/// point decoded in one pass as the release decodes them.
struct NFKSF3DReferenceFields: NFKSF3DFields {
    let network: NFKMLXStableFast3DReferenceNetwork
    let sceneCode: [Float]

    func geometry(_ points: [Float]) throws -> (density: [Float], vertexOffset: [Float]) {
        try network.geometry(points, sceneCode: sceneCode)
    }

    func surface(_ points: [Float]) throws -> (albedo: [Float], normal: [Float]) {
        try network.surface(points, sceneCode: sceneCode)
    }
}

// MARK: - The whole surface stage

/// The mesh of one scene code, as `SF3D.generate_mesh` builds it with no remeshing.
public enum NFKMLXStableFast3DGeometry {

    /// One scene code's textured mesh: the grid deformed by the vertex offsets, marching tetrahedra at
    /// `options.threshold`, the box unwrap, the bake of the albedo and of the perturbed normal in tangent
    /// space at `options.textureResolution`, the island padding, and the 8-bit textures. The mesh is in
    /// glTF's frame (`(x, y, z) → (y, z, x)` from the reference's), its vertices welded where the release's
    /// trimesh export welds them, with angle-weighted normals. Introduced in InferKit 0.4.0.
    public static func mesh(sceneCode: MLXArray, net: NFKMLXStableFast3DNet, grid: NFKMLXTetrahedralGrid,
                            roughness: Float, metallic: Float, options: NFKMLXStableFast3DOptions) throws -> NFKMesh {
        try mesh(fields: NFKSF3DNetworkFields(net: net, sceneCode: sceneCode), configuration: net.configuration, grid: grid,
                 roughness: roughness, metallic: metallic, options: options)
    }

    static func mesh(fields: any NFKSF3DFields, configuration: NFKMLXStableFast3DConfiguration, grid: NFKMLXTetrahedralGrid,
                     roughness: Float, metallic: Float, options: NFKMLXStableFast3DOptions) throws -> NFKMesh {
        let surface = try extractedSurface(fields: fields, configuration: configuration, grid: grid, threshold: options.threshold)
        guard !surface.faces.isEmpty else {
            return try NFKMesh(positions: Data(), normals: nil, colors: nil, textureCoordinates: nil, indices: Data())
        }
        let normals = NFKSF3DVertexNormals(positions: surface.positions, faces: surface.faces)
        let cornerUV = try NFKSF3DUnwrapper.unwrap(positions: surface.positions, normals: normals, faces: surface.faces, padding: 0.02)
        let split = splitMesh(positions: surface.positions, faces: surface.faces, cornerUV: cornerUV)
        let textures = try bakedTextures(split: split, fields: fields, options: options)
        let material = try NFKMeshMaterial(baseColorTexture: textures.albedo, normalTexture: textures.normal,
                                           metallicRoughnessTexture: nil, baseColorFactor: nil,
                                           metallicFactor: min(max(metallic, 0), 1), roughnessFactor: min(max(roughness, 0), 1))
        return try exported(split: split, material: material)
    }

    /// The deformed grid's marching-tetrahedra surface in the reference's frame: positions three floats a
    /// vertex in world units and faces in the extractor's winding.
    static func extractedSurface(fields: any NFKSF3DFields, configuration: NFKMLXStableFast3DConfiguration,
                                 grid: NFKMLXTetrahedralGrid, threshold: Float) throws -> (positions: [Float], faces: [UInt32]) {
        let radius = configuration.radius
        let span = radius - (-radius)
        let (density, offsets) = try fields.geometry(grid.vertices.map { $0 * span + (-radius) })
        return surface(grid: grid, density: density, vertexOffsets: offsets, threshold: threshold,
                       resolution: configuration.gridResolution, radius: radius)
    }

    /// `triplane_to_meshes` from the decoder's density and vertex offset at every grid vertex.
    static func surface(grid: NFKMLXTetrahedralGrid, density: [Float], vertexOffsets: [Float], threshold: Float,
                        resolution: Int, radius: Float) -> (positions: [Float], faces: [UInt32]) {
        let scale = Float(1.0 / Double(resolution))
        var deformed = grid.vertices
        for index in deformed.indices {
            deformed[index] += scale * NFKSF3DTorchMath.tanh(vertexOffsets[index])
        }
        let level = density.map { $0 - threshold }
        var (vertices, faces) = NFKMarchingTetrahedra.surface(positions: deformed, level: level, tetrahedra: grid.tetrahedra)
        let span = radius - (-radius)
        for index in vertices.indices {
            vertices[index] = vertices[index] * span + (-radius)
        }
        return (vertices, faces)
    }

    /// `Mesh.unwrap_uv`'s split: every face corner its own vertex, with the face's normal and the
    /// tangent its chart gives.
    struct Split {
        let positions: [Float]
        let faces: [UInt32]
        let uv: [Float]
        let normals: [Float]
        let tangents: [Float]
    }

    static func splitMesh(positions: [Float], faces: [UInt32], cornerUV: [Float]) -> Split {
        var corners = [Float]()
        corners.reserveCapacity(faces.count * 3)
        for index in faces {
            corners += [positions[Int(index) * 3], positions[Int(index) * 3 + 1], positions[Int(index) * 3 + 2]]
        }
        let sequence = (0 ..< UInt32(faces.count)).map { $0 }
        let normals = NFKSF3DVertexNormals(positions: corners, faces: sequence)
        let tangents = NFKSF3DVertexTangents(positions: corners, normals: normals, faces: sequence, cornerUV: cornerUV)
        return Split(positions: corners, faces: sequence, uv: cornerUV, normals: normals, tangents: tangents)
    }

    /// The albedo and the tangent-space normal textures, padded and quantized, as files.
    static func bakedTextures(split: Split, fields: any NFKSF3DFields,
                              options: NFKMLXStableFast3DOptions) throws -> (albedo: NFKMeshTexture, normal: NFKMeshTexture) {
        let side = options.textureResolution
        let raster = NFKSF3DBaker.rasterize(uv: split.uv, faces: split.faces, side: side)
        let covered = raster.triangle.map { $0 >= 0 }
        let positions = NFKSF3DBaker.interpolate(split.positions, faces: split.faces, weights: raster.weights, triangle: raster.triangle)
        let normals = NFKSF3DBaker.interpolate(split.normals, faces: split.faces, weights: raster.weights, triangle: raster.triangle)
        let tangents = NFKSF3DBaker.interpolate(split.tangents, faces: split.faces, weights: raster.weights, triangle: raster.triangle)
        let (albedo, perturbed) = try fields.surface(positions)
        let tangentSpace = tangentSpaceNormals(perturbed: perturbed, normals: normals, tangents: tangents)
        func texture(_ values: [Float]) -> [Float] {
            var image = [Float](repeating: 0, count: side * side * 3)
            var next = 0
            for texel in covered.indices where covered[texel] {
                image[texel * 3] = values[next * 3]; image[texel * 3 + 1] = values[next * 3 + 1]; image[texel * 3 + 2] = values[next * 3 + 2]
                next += 1
            }
            return NFKSF3DBaker.dilated(image, mask: covered, side: side, iterations: side / 150)
        }
        let albedoTexture = texture(albedo)
        let normalTexture = texture(tangentSpace)
        var generator = NFKSF3DDither(seed: options.seed)
        let albedoBytes = NFKSF3DBaker.quantized(albedoTexture, ditherMask: nil, generator: &generator)
        // The release passes the flat texels as the dither mask, so only they are dithered.
        let flat = (0 ..< side * side).map { normalTexture[$0 * 3] == 0.5 && normalTexture[$0 * 3 + 1] == 0.5 && normalTexture[$0 * 3 + 2] == 1 }
        let normalBytes = NFKSF3DBaker.quantized(normalTexture, ditherMask: flat, generator: &generator)
        return (try NFKMLXStableFast3D.encodedImage(albedoBytes, width: side, height: side, type: options.textureFormat),
                try NFKMLXStableFast3D.encodedImage(normalBytes, width: side, height: side, type: options.textureFormat))
    }

    /// The perturbed normal in each texel's tangent frame (tangent, bitangent, normal), mapped to 0...1.
    static func tangentSpaceNormals(perturbed: [Float], normals: [Float], tangents: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: perturbed.count)
        for texel in 0 ..< perturbed.count / 3 {
            let n = NFKSF3DVector.at(normals, texel).torchNormalized()
            let t = NFKSF3DVector.at(tangents, texel).torchNormalized()
            let bitangent = n.torchCross(t).torchNormalized()
            let target = NFKSF3DVector.at(perturbed, texel).torchNormalized(epsilon: 1e-7).torchNormalized()
            // `torch.bmm` of 3 × 3 by 3 runs its naive kernel, whose sums clang contracts.
            func row(_ r: NFKSF3DVector) -> Float { (r.x * target.x).addingProduct(r.y, target.y).addingProduct(r.z, target.z) }
            let components = [row(t), row(bitangent), row(n)]
            for axis in 0 ..< 3 {
                out[texel * 3 + axis] = min(max(components[axis] * 0.5 + 0.5, 0), 1)
            }
        }
        return out
    }

    /// The split mesh as the release's trimesh export holds it: vertices welded where their positions
    /// agree to 1e-8 and their texture coordinates to 1e-4 (the first of each kept, in order), the winding
    /// reversed (`invert`), positions turned into glTF's frame, and angle-weighted vertex normals in float64.
    static func exported(split: Split, material: NFKMeshMaterial) throws -> NFKMesh {
        struct Key: Hashable { let p: [Int64]; let t: [Int64] }
        var remap = [UInt32](repeating: 0, count: split.positions.count / 3)
        var seen = [Key: UInt32]()
        var positions = [Float](), uv = [Float]()
        for vertex in 0 ..< split.positions.count / 3 {
            let p = (0 ..< 3).map { Int64((Double(split.positions[vertex * 3 + $0]) * 1e8).rounded(.toNearestOrEven)) }
            let t = (0 ..< 2).map { Int64((Double(split.uv[vertex * 2 + $0]) * 1e4).rounded(.toNearestOrEven)) }
            let key = Key(p: p, t: t)
            if let existing = seen[key] {
                remap[vertex] = existing
                continue
            }
            let index = UInt32(positions.count / 3)
            seen[key] = index
            remap[vertex] = index
            positions += [split.positions[vertex * 3 + 1], split.positions[vertex * 3 + 2], split.positions[vertex * 3]]
            uv += [split.uv[vertex * 2], split.uv[vertex * 2 + 1]]
        }
        var faces = [UInt32](repeating: 0, count: split.faces.count)
        for face in 0 ..< split.faces.count / 3 {
            faces[face * 3] = remap[Int(split.faces[face * 3 + 2])]
            faces[face * 3 + 1] = remap[Int(split.faces[face * 3 + 1])]
            faces[face * 3 + 2] = remap[Int(split.faces[face * 3])]
        }
        let normals = angleWeightedNormals(positions: positions, faces: faces)
        func data<T>(_ values: [T]) -> Data { values.withUnsafeBufferPointer { Data(buffer: $0) } }
        return try NFKMesh(positions: data(positions), normals: data(normals), colors: nil, textureCoordinates: data(uv),
                           indices: data(faces), material: material)
    }

    /// trimesh's `vertex_normals`: each unit face normal weighted by the corner's angle, summed at the
    /// vertex, and normalized, in float64. A degenerate face contributes nothing.
    static func angleWeightedNormals(positions: [Float], faces: [UInt32]) -> [Float] {
        let count = positions.count / 3
        var sums = [Double](repeating: 0, count: count * 3)
        func point(_ index: UInt32) -> (Double, Double, Double) {
            let i = Int(index) * 3
            return (Double(positions[i]), Double(positions[i + 1]), Double(positions[i + 2]))
        }
        func unit(_ v: (Double, Double, Double)) -> (Double, Double, Double) {
            let length = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).squareRoot()
            return length > 0 ? (v.0 / length, v.1 / length, v.2 / length) : (0, 0, 0)
        }
        for face in 0 ..< faces.count / 3 {
            let a = point(faces[face * 3]), b = point(faces[face * 3 + 1]), c = point(faces[face * 3 + 2])
            let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2), ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2)
            let normal = unit((ab.1 * ac.2 - ab.2 * ac.1, ab.2 * ac.0 - ab.0 * ac.2, ab.0 * ac.1 - ab.1 * ac.0))
            guard normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2 > 0.5 else { continue }
            let u = unit(ab), v = unit(ac), w = unit((c.0 - b.0, c.1 - b.1, c.2 - b.2))
            let first = acos(min(max(u.0 * v.0 + u.1 * v.1 + u.2 * v.2, -1), 1))
            let second = acos(min(max(-u.0 * w.0 - u.1 * w.1 - u.2 * w.2, -1), 1))
            var angles = [first, second, Double.pi - first - second]
            if angles.contains(where: { $0 < 1e-8 }) { angles = [0, 0, 0] }
            for corner in 0 ..< 3 {
                let index = Int(faces[face * 3 + corner]) * 3
                sums[index] += normal.0 * angles[corner]
                sums[index + 1] += normal.1 * angles[corner]
                sums[index + 2] += normal.2 * angles[corner]
            }
        }
        var normals = [Float](repeating: 0, count: count * 3)
        for vertex in 0 ..< count {
            let n = unit((sums[vertex * 3], sums[vertex * 3 + 1], sums[vertex * 3 + 2]))
            normals[vertex * 3] = Float(n.0); normals[vertex * 3 + 1] = Float(n.1); normals[vertex * 3 + 2] = Float(n.2)
        }
        return normals
    }
}
