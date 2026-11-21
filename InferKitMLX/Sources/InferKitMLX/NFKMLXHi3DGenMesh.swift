//
//  NFKMLXHi3DGenMesh.swift
//  InferKitMLX
//
//  Hi3DGen's surface extraction from the TRELLIS mesh decoder's features, on the CPU in float32. The
//  corner attributes average over the voxels sharing each grid point, marching cubes runs on the SDF at
//  level 0, and each vertex moves by the trilinear interpolation of the deformed grid and takes that of
//  the color grid. Every stage sweeps the grid one x plane at a time, so no dense grid is held.
//

import Foundation

/// The grid points the voxels of a sparse set touch, ordered by x, then y, then z, each with the mean of
/// the attributes its voxels give it: an SDF, a deformation (3), and a color (6).
struct NFKHi3DGenCorners {
    static let attributeCount = 10

    let resolution: Int
    /// The first point of each x plane, `resolution + 2` entries: plane `x` holds the points
    /// `planeStarts[x] ..< planeStarts[x + 1]`.
    var planeStarts: [Int]
    /// Each point's `y · (resolution + 1) + z`, ascending within its plane.
    var cells: [Int32]
    /// `attributeCount` floats a point.
    var attributes: [Float]

    var count: Int { cells.count }
    var side: Int { resolution + 1 }

    init(resolution: Int, planeStarts: [Int], cells: [Int32], attributes: [Float]) {
        self.resolution = resolution
        self.planeStarts = planeStarts
        self.cells = cells
        self.attributes = attributes
    }

    /// Points given as x, y, z triples in ascending order, with `attributeCount` attributes each.
    init(resolution: Int, coordinates: [Int32], attributes: [Float]) {
        let side = resolution + 1
        var starts = [Int](repeating: 0, count: side + 1)
        var cells = [Int32]()
        cells.reserveCapacity(coordinates.count / 3)
        for point in 0 ..< coordinates.count / 3 {
            starts[Int(coordinates[3 * point]) + 1] += 1
            cells.append(coordinates[3 * point + 1] * Int32(side) + coordinates[3 * point + 2])
        }
        for x in 0 ..< side {
            starts[x + 1] += starts[x]
        }
        self.init(resolution: resolution, planeStarts: starts, cells: cells, attributes: attributes)
    }

    /// The points as x, y, z triples.
    var coordinates: [Int32] {
        var triples = [Int32]()
        triples.reserveCapacity(count * 3)
        for x in 0 ..< side {
            for point in planeStarts[x] ..< planeStarts[x + 1] {
                let cell = Int(cells[point])
                triples += [Int32(x), Int32(cell / side), Int32(cell % side)]
            }
        }
        return triples
    }

    /// The corners of the voxels at `coordinates` (x, y, z triples on a `resolution`³ grid) whose decoder
    /// features are `features` (`NFKHi3DGenSurface.featureChannels` a voxel). A point's attribute is the
    /// float32 sum over the voxels sharing it, taken in voxel order, divided by their number; the SDF
    /// takes the bias −1 / `resolution` first.
    static func averaged(coordinates: [Int32], features: [Float], resolution: Int) -> NFKHi3DGenCorners {
        let voxels = coordinates.count / 3
        let side = resolution + 1
        let channels = NFKHi3DGenSurface.featureChannels
        let width = attributeCount
        let sdfBias = Float(-1 / Double(resolution))

        // The voxels by x, each x in voxel order.
        var voxelStarts = [Int](repeating: 0, count: resolution + 2)
        for voxel in 0 ..< voxels {
            voxelStarts[Int(coordinates[3 * voxel]) + 1] += 1
        }
        for x in 0 ... resolution {
            voxelStarts[x + 1] += voxelStarts[x]
        }
        var byX = [Int32](repeating: 0, count: voxels)
        var next = voxelStarts
        for voxel in 0 ..< voxels {
            let x = Int(coordinates[3 * voxel])
            byX[next[x]] = Int32(voxel)
            next[x] += 1
        }

        var planeStarts = [Int](repeating: 0, count: side + 1)
        var cells = [Int32]()
        var attributes = [Float]()
        let slotOfCell = UnsafeMutableBufferPointer<Int32>.allocate(capacity: side * side)
        let sums = UnsafeMutableBufferPointer<Float>.allocate(capacity: side * side * width)
        let counts = UnsafeMutableBufferPointer<Float>.allocate(capacity: side * side)
        slotOfCell.initialize(repeating: -1)
        sums.initialize(repeating: 0)
        counts.initialize(repeating: 0)
        defer {
            slotOfCell.deallocate()
            sums.deallocate()
            counts.deallocate()
        }
        var slotCells = [Int32]()
        features.withUnsafeBufferPointer { feature in
            coordinates.withUnsafeBufferPointer { coordinate in
                for x in 0 ... resolution {
                    planeStarts[x] = cells.count
                    // Plane x gathers the far corners of the voxels at x − 1 and the near corners of those at x.
                    let below = x > 0 ? voxelStarts[x - 1] ..< voxelStarts[x] : 0 ..< 0
                    let above = x < resolution ? voxelStarts[x] ..< voxelStarts[x + 1] : 0 ..< 0
                    var i = below.lowerBound, j = above.lowerBound
                    while i < below.upperBound || j < above.upperBound {
                        let fromBelow = j == above.upperBound || (i < below.upperBound && byX[i] < byX[j])
                        let voxel = Int(fromBelow ? byX[i] : byX[j])
                        if fromBelow {
                            i += 1
                        } else {
                            j += 1
                        }
                        let y = Int(coordinate[3 * voxel + 1]), z = Int(coordinate[3 * voxel + 2])
                        let row = voxel * channels
                        var corner = fromBelow ? 1 : 0
                        while corner < 8 {
                            let cell = (y + ((corner >> 1) & 1)) * side + z + ((corner >> 2) & 1)
                            var slot = Int(slotOfCell[cell])
                            if slot < 0 {
                                slot = slotCells.count
                                slotOfCell[cell] = Int32(slot)
                                slotCells.append(Int32(cell))
                            }
                            counts[slot] += 1
                            let sum = sums.baseAddress! + slot * width
                            let sdf = row + corner, deformation = row + 8 + corner * 3, color = row + 53 + corner * 6
                            sum[0] += feature[sdf] + sdfBias
                            sum[1] += feature[deformation]
                            sum[2] += feature[deformation + 1]
                            sum[3] += feature[deformation + 2]
                            sum[4] += feature[color]
                            sum[5] += feature[color + 1]
                            sum[6] += feature[color + 2]
                            sum[7] += feature[color + 3]
                            sum[8] += feature[color + 4]
                            sum[9] += feature[color + 5]
                            corner += 2
                        }
                    }
                    for slot in slotCells.indices.sorted(by: { slotCells[$0] < slotCells[$1] }) {
                        cells.append(slotCells[slot])
                        for k in 0 ..< width {
                            attributes.append(sums[slot * width + k] / counts[slot])
                            sums[slot * width + k] = 0
                        }
                        counts[slot] = 0
                        slotOfCell[Int(slotCells[slot])] = -1
                    }
                    slotCells.removeAll(keepingCapacity: true)
                }
            }
        }
        planeStarts[side] = cells.count
        return NFKHi3DGenCorners(resolution: resolution, planeStarts: planeStarts, cells: cells, attributes: attributes)
    }
}

/// Marching cubes at level 0 over the SDF of a corner set, a point the set lacks reading 1 (outside), with
/// the topology and triangles scikit-image's `measure.marching_cubes` (Lewiner et al. 2003) gives with
/// `gradient_direction='ascent'`, in its order.
///
/// Corner `c` of a cube sits at (c & 1, (c >> 1) & 1, (c >> 2) & 1); a corner is inside where its value is
/// at most 0, and a value of exactly 0 counts as −ε (double precision). The case is the inside pattern
/// plus decisions:
/// - A face whose diagonal corners share a sign joins its inside pair where the product of the inside
///   values exceeds that of the outside values (the face's bilinear saddle is inside).
/// - Two isolated diagonal corners, and two opposite ambiguous faces that both join, are decided inside
///   the cube on the slice where the face test peaks (`interiorJoins`).
/// The triangles filling each decided case are recorded from the reference
/// (`Tools/reference-parity/generate_marching_cubes_table.py`): its triangulations follow its own table,
/// which no rule derives. Cubes run x-major; a vertex is numbered at its first use, an edge's vertex shared
/// by the cubes around the edge, and a vertex inside a cube made once for that cube.
enum NFKHi3DGenMarchingCubes {

    struct Surface {
        /// Three floats a vertex, in grid units.
        var vertices: [Float]
        /// Three vertex indices a triangle, wound with its normal toward the inside.
        var triangles: [UInt32]
    }

    /// The twelve edges as corner pairs, the lower corner first; edge 12 names the vertex inside the cube.
    static let edges: [(Int, Int)] = (0 ..< 8).flatMap { a in
        (0 ..< 3).compactMap { axis in a & (1 << axis) == 0 ? (a, a | (1 << axis)) : nil }
    }.sorted { ($0.0, $0.1) < ($1.0, $1.1) }

    /// Each face's corners in cycle order: x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
    static let faces: [[Int]] = [[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]]

    /// The faces of each case whose diagonal corners share a sign, in face order.
    static let ambiguousFaces: [[Int]] = (0 ..< 256).map { mask in
        faces.indices.filter { index in
            let signs = faces[index].map { (mask >> $0) & 1 }
            return signs[0] == signs[2] && signs[1] == signs[3] && signs[0] != signs[1]
        }
    }

    /// How a decided case fills its cube: edge indices, three a triangle.
    enum Fill: Equatable {
        /// A face-decision combination no trilinear cube produces.
        case unreachable
        case triangles([UInt8])
        /// Triangles without and with the interior connection.
        case split([UInt8], [UInt8])
    }

    /// `fills[mask][joins]`, where bit `k` of `joins` is set when the case's `k`-th ambiguous face joins
    /// its inside corners.
    static let fills: [[Fill]] = {
        var table = [[Fill]](repeating: [.triangles([])], count: 256)
        var mask = 0
        func indices(_ text: Substring) -> [UInt8] { text.map { UInt8($0.hexDigitValue!) } }
        for token in recordedTable.split(whereSeparator: { $0 == " " || $0 == "\n" }) {
            if token.hasSuffix(":") {
                mask = Int(token.dropLast(), radix: 16)!
                table[mask] = []
            } else if token == "." {
                table[mask].append(.unreachable)
            } else if let bar = token.firstIndex(of: "|") {
                table[mask].append(.split(indices(token[..<bar]), indices(token[token.index(after: bar)...])))
            } else {
                table[mask].append(.triangles(indices(token)))
            }
        }
        return table
    }()

    /// The four values on the slice across `axis` where the face test a·c − b·d is stationary, the
    /// corners named by their position on the slice's 0 side (`a`, `c` one diagonal), or nil where that
    /// slice lies outside the cube.
    static func peakSlice(_ v: [Double], axis: Int, _ a: Int, _ b: Int, _ c: Int, _ d: Int)
        -> (a: Double, b: Double, c: Double, d: Double)? {
        let step = 1 << axis
        let (a0, a1, b0, b1) = (v[a], v[a | step], v[b], v[b | step])
        let (c0, c1, d0, d1) = (v[c], v[c | step], v[d], v[d | step])
        let quadratic = (a1 - a0) * (c1 - c0) - (b1 - b0) * (d1 - d0)
        let linear = c0 * (a1 - a0) + a0 * (c1 - c0) - d0 * (b1 - b0) - b0 * (d1 - d0)
        guard quadratic != 0 else { return nil }
        let t = -linear / (2 * quadratic)
        guard t >= 0, t <= 1 else { return nil }
        return (a0 + (a1 - a0) * t, b0 + (b1 - b0) * t, c0 + (c1 - c0) * t, d0 + (d1 - d0) * t)
    }

    /// Whether the cube's interior joins what its faces keep apart, for the two cases that split on it:
    /// - Two isolated diagonal corners, tested on the slice across x through both. Two outside corners
    ///   join where that slice holds them outside on one diagonal, the others inside, and the outside
    ///   product is the larger. For two inside corners the reference's test reads inverted: they take the
    ///   connected triangles unless the slice lies inside the cube and shows one of them outside.
    /// - Two opposite ambiguous faces that both join their inside corners, tested on the slice between
    ///   them: the outside corners join where the slice holds them on one diagonal and their product is
    ///   the larger.
    static func interiorJoins(_ v: [Double], mask: Int) -> Bool {
        let inside = (0 ..< 8).filter { (mask >> $0) & 1 == 1 }
        let outside = (0 ..< 8).filter { (mask >> $0) & 1 == 0 }
        for pair in [inside, outside] where pair.count == 2 && pair[0] ^ pair[1] == 7 {
            let (a, c) = (pair[0] & ~1, pair[1] & ~1)
            let others = [0, 2, 4, 6].filter { $0 != a && $0 != c }
            let slice = peakSlice(v, axis: 0, a, others[0], c, others[1])
            if pair == inside {
                guard let s = slice else { return true }
                return s.a < 0 && s.c < 0
            }
            guard let s = slice else { return false }
            return s.a >= 0 && s.c >= 0 && s.b < 0 && s.d < 0 && s.a * s.c > s.b * s.d
        }
        let axis = ambiguousFaces[mask][0] / 2
        let c = 7 ^ (1 << axis)
        let others = (1 ..< 8).filter { (($0 >> axis) & 1) == 0 && $0 != c }
        guard let s = peakSlice(v, axis: axis, 0, others[0], c, others[1]) else { return false }
        if s.a >= 0 && s.c >= 0 && s.b < 0 && s.d < 0 {
            return s.a * s.c > s.b * s.d
        }
        if s.b >= 0 && s.d >= 0 && s.a < 0 && s.c < 0 {
            return s.b * s.d > s.a * s.c
        }
        return false
    }

    /// The triangles of the cube whose corner values are `v`, as edge indices, and its inside pattern.
    static func cubeTriangles(_ v: [Double]) -> (mask: Int, edges: [UInt8]) {
        var mask = 0
        for corner in 0 ..< 8 where v[corner] < 0 {
            mask |= 1 << corner
        }
        var joins = 0
        for (bit, face) in ambiguousFaces[mask].enumerated() {
            let w = faces[face].map { v[$0] }
            let (inside, outside) = w[0] < 0 ? (w[0] * w[2], w[1] * w[3]) : (w[1] * w[3], w[0] * w[2])
            if inside > outside {
                joins |= 1 << bit
            }
        }
        switch fills[mask][joins] {
        case .unreachable:
            return (mask, [])
        case .triangles(let edges):
            return (mask, edges)
        case .split(let without, let with):
            return (mask, interiorJoins(v, mask: mask) ? with : without)
        }
    }

    /// The surface of `corners`' SDF (the first attribute) at level 0, in grid units.
    static func surface(of corners: NFKHi3DGenCorners) -> Surface {
        let r = corners.resolution, side = r + 1
        let width = NFKHi3DGenCorners.attributeCount
        var vertices = [Float](), triangles = [UInt32]()
        guard r > 0 else { return Surface(vertices: vertices, triangles: triangles) }
        let epsilon = Double.ulpOfOne
        let attributes = corners.attributes, cells = corners.cells, starts = corners.planeStarts

        // The points of planes x (near) and x + 1 (far) by cell, and the vertices of their edges: two per
        // cell along y and z in each plane, one per cell along x between them.
        var near = [Int32](repeating: -1, count: side * side), far = near
        var nearEdges = [Int32](repeating: -1, count: side * side * 2), farEdges = nearEdges
        var nearEdgeSlots = [Int](), farEdgeSlots = [Int]()
        var crossEdges = [Int32](repeating: -1, count: side * side), crossEdgeSlots = [Int]()
        var marked = [Bool](repeating: false, count: r * r)
        var candidates = [Int]()
        var v = [Double](repeating: 0, count: 8)

        func place(_ map: inout [Int32], plane: Int, _ value: (Int) -> Int32) {
            for point in starts[plane] ..< starts[plane + 1] {
                map[Int(cells[point])] = value(point)
            }
        }
        place(&near, plane: 0) { Int32($0) }
        place(&far, plane: 1) { Int32($0) }

        for x in 0 ..< r {
            if x > 0 {
                swap(&near, &far)
                place(&far, plane: x - 1) { _ in -1 }
                place(&far, plane: x + 1) { Int32($0) }
                swap(&nearEdges, &farEdges)
                swap(&nearEdgeSlots, &farEdgeSlots)
                for slot in farEdgeSlots {
                    farEdges[slot] = -1
                }
                farEdgeSlots.removeAll(keepingCapacity: true)
                for slot in crossEdgeSlots {
                    crossEdges[slot] = -1
                }
                crossEdgeSlots.removeAll(keepingCapacity: true)
            }

            // Surface cubes have an inside corner, and only points of the set can be inside.
            for plane in x ... x + 1 {
                for point in starts[plane] ..< starts[plane + 1] where attributes[point * width] <= 0 {
                    let cell = Int(cells[point])
                    let (y, z) = (cell / side, cell % side)
                    for (cy, cz) in [(y - 1, z - 1), (y - 1, z), (y, z - 1), (y, z)]
                    where cy >= 0 && cz >= 0 && cy < r && cz < r && !marked[cy * r + cz] {
                        marked[cy * r + cz] = true
                        candidates.append(cy * r + cz)
                    }
                }
            }
            candidates.sort()

            for cube in candidates {
                marked[cube] = false
                let (y, z) = (cube / r, cube % r)
                for corner in 0 ..< 8 {
                    let cell = (y + ((corner >> 1) & 1)) * side + z + ((corner >> 2) & 1)
                    let point = corner & 1 == 0 ? near[cell] : far[cell]
                    let value = point >= 0 ? attributes[Int(point) * width] : 1
                    v[corner] = value == 0 ? -epsilon : Double(value)
                }
                let (mask, edgeList) = cubeTriangles(v)
                if mask == 255 || edgeList.isEmpty {
                    continue
                }
                var center: UInt32?
                for e in edgeList {
                    if e == 12 {
                        if center == nil {
                            // The corners weighted by the reciprocal of their distance from the level.
                            var total = 0.0, local = (0.0, 0.0, 0.0)
                            for corner in 0 ..< 8 {
                                let weight = 1 / abs(v[corner])
                                total += weight
                                local.0 += corner & 1 == 1 ? weight : 0
                                local.1 += (corner >> 1) & 1 == 1 ? weight : 0
                                local.2 += (corner >> 2) & 1 == 1 ? weight : 0
                            }
                            center = UInt32(vertices.count / 3)
                            vertices += [Float(Double(x) + local.0 / total), Float(Double(y) + local.1 / total),
                                         Float(Double(z) + local.2 / total)]
                        }
                        triangles.append(center!)
                        continue
                    }
                    let (a, b) = edges[Int(e)]
                    let axis = (a ^ b).trailingZeroBitCount
                    let lower = (x + (a & 1), y + ((a >> 1) & 1), z + ((a >> 2) & 1))
                    let cell = lower.1 * side + lower.2
                    let cached: Int32
                    if axis == 0 {
                        cached = crossEdges[cell]
                    } else {
                        cached = a & 1 == 0 ? nearEdges[cell * 2 + axis - 1] : farEdges[cell * 2 + axis - 1]
                    }
                    if cached >= 0 {
                        triangles.append(UInt32(cached))
                        continue
                    }
                    let index = Int32(vertices.count / 3)
                    let t = v[a] / (v[a] - v[b])
                    vertices += [axis == 0 ? Float(Double(lower.0) + t) : Float(lower.0),
                                 axis == 1 ? Float(Double(lower.1) + t) : Float(lower.1),
                                 axis == 2 ? Float(Double(lower.2) + t) : Float(lower.2)]
                    if axis == 0 {
                        crossEdges[cell] = index
                        crossEdgeSlots.append(cell)
                    } else if a & 1 == 0 {
                        nearEdges[cell * 2 + axis - 1] = index
                        nearEdgeSlots.append(cell * 2 + axis - 1)
                    } else {
                        farEdges[cell * 2 + axis - 1] = index
                        farEdgeSlots.append(cell * 2 + axis - 1)
                    }
                    triangles.append(UInt32(index))
                }
            }
            candidates.removeAll(keepingCapacity: true)
        }
        return Surface(vertices: vertices, triangles: triangles)
    }

    /// The triangulations recorded from the reference, by `generate_marching_cubes_table.py`: per case
    /// `mask:` in hex, then one token a face-decision combination (edge indices in hex, three a
    /// triangle), `.` where none is reachable, `without|with` where the interior decides.
    static let recordedTable = """
        01: 210
        02: 403
        03: 421341
        04: 165
        05: 265025
        06: 340651 536346016064
        07: 534546642
        08: 357
        09: 021735 307027157172
        0a: 740570
        0b: 157172274
        0c: 316736
        0d: 302327726
        0e: 016064467
        0f: 264467
        10: 289
        11: 910890
        12: 892340 483893203239
        13: 489493391
        14: 289516 125285965958
        15: 965958850
        16: 340928165 340281518958596 516320239493948 c34c48c89c96c65c51c12c20c03 892536463641014
        34c53c65c96c89c28c12c01c40c 89c48c34c53c65c16c01c20c92c 536863689483120
        17: 453854865896
        18: 892735|978387832523259795
        19: 735918081 159795978387830
        1a: 928745054 487978795259520
        1b: 874817891571
        1c: 892317671 123832387978796
        1d: 967970908073
        1e: 892601064746 c12c28c89c96c67c74c40c01 74c67c16c01c20c92c89c48c 012689864746
        1f: 948964674
        20: 84a
        21: 4a8102 0414a182181a
        22: 803a83
        23: 82181aa13
        24: 4a8516|145a545a686a681418
        25: 84a526250 68286a5a6a54045
        26: 51683a380 a535a686a681018
        27: 83a863826653
        28: a84573 7a5a85435458
        29: 02184a573 02138483585a7a5 735041a141a282a 21c82ca8c7ac57c35c43c04c10c 4a8715172327230
        c21c15c57c7aca8c84c43c30c02 ca8c82c21c15c57c73c30c04c4a 715178a87182043
        2a: 7a8785580
        2b: 15721727a2a8
        2c: 4a8631367 14341868186a7a6
        2d: 4a8307270726 43c84ca8c7ac67c26c02c30c c26c67c73c30c04c4aca8c82 430a87278726
        2e: a80a06a67160
        2f: 87a827267
        30: 24a92a
        31: 04a0a11a9
        32: 20323993a
        33: 91a13a
        34: 516a94249 96a5a6a54145412
        35: 6a960a6504a0
        36: 16520939093a 12c51c65c96ca9c3ac03c20c c3aca9c92c20c01c16c65c53 12065939593a
        37: 5965393a9
        38: 7352a9a24 97a795259523432
        39: 573a040a191a 91ca9c7ac57c35c43c04c10c c73c30c04c4aca9c91c15c57 043a575a191a
        3a: 20525a2a9a57
        3b: 7157a1a91
        3c: 16373649294a 4ac24c12c31c73c67c96ca9c c92c24c43c31c16c67c7aca9 96a7a6412143|37a3a424a92a296261316736
        3d: 73606360949094a 043679a97
        3e: 716170a070a292a 120a676a9
        3f: 97a679
        40: 9b6
        41: 102b69 61b10b29b2b0
        42: b69340|63b4b3b49094906360
        43: b69423132 294b494b363b361
        44: 19b51b
        45: 29b2b00b5
        46: 034b191b5 901094b494b353b
        47: 9b5954942345
        48: 69b357 563693b73b39
        49: 73521069b 735106b602b0b29 021b79397936563 73cb7c9bc29c02c10c61c56c35c b69715172327230
        c73c30c02c29c9bcb6c61c15c57 02c30c73cb7c9bc69c56c15c21c b79097902307615
        4a: b69074705 4b7b49094906560
        4b: 69b152725274 c74c42c29c9bcb6c61c15c57 69c56c15c21c42c74cb7c9bc 6159b272b274
        4c: b73b39931
        4d: b739b3930902
        4e: 01909707479b
        4f: b29b72742
        50: 8b6286
        51: 61060bb08
        52: 34062b8b2 20636063b4b3b48
        53: 68b638613348
        54: 12818558b
        55: 8b0b50
        56: 03412585258b 03c20c12c51cb5c8bc48c34c c8bcb5c53c34c40c01c12c28 01234585458b
        57: 4534858b5
        58: 57386268b 256523832387b78
        59: 57361b0b1b08 56c35c73cb7c8bc08c10c61c c08c8bcb6c61c15c57c73c30 56173b0b3b08
        5a: b282b6407570 c57c74c48c8bcb6c62c20c05 40c74cb7c8bc28c62c56c05c 520256487b78|56b75b57454020482428b2b6
        5b: 457541814186b68 56187478b
        5c: 2312b328b73b
        5d: 3b730b08b
        5e: 40717071b2b1b28 12074b8b4
        5f: b4874b
        60: b694a8 ab4b64984946
        61: 4a869b210 4a89b202b60b061 b69041a141a282a 4ac04c10c61cb6c9bc29c82ca8c 021498946a646ab
        c4acabcb6c61c10c02c29c98c84 b6cabc4ac04c10c21c82c98c69c 041b141b6ab4298
        62: 69b3a080a ab363b360906098
        63: b6918281a3a1 3ac13c61cb6c9bc29c82ca8c c69c98c82c21c13c3acabcb6 8291b6b1a3a1
        64: 4a8195b59 981418145a545ab
        65: 84a290b090b5 82ca8c4ac04c50cb5c9bc29c cb5c50c02c29c98c84c4acab 8294a0b0a0b5
        66: a8303859b951 ca8c80c01c19c9bcb5c53c3a 03c80c98c19c51cb5cabc3ac ab353b098901|153130803a838ab8b919b51b
        67: a8323832595259b 2983a5b5a
        68: b6957384a b69a575a8583438 4a8b79397936563 69c56c35c43c84ca8c7acb7c9bc 735498946a646ab
        c69c98c84c43c35c57c7acabcb6 c35c56c69c98c84c4acabcb7c73 498945354956b7a
        69: 735b692104a8 7354a8106b602b0b29 021b69a575a8583438 . 73569b041a141a282a 7354ac04c10c61cb6c9bc29c82ca8c
        b6921c82ca8c7ac57c35c43c04c10c . 0214a8b79397936563 4a873cb7c9bc29c02c10c61c56c35c
        02169c56c35c43c84ca8c7acb7c9bc . . . . . 69b4a8715172327230 4a8c73c30c02c29c9bcb6c61c15c57
        69bc21c15c57c7aca8c84c43c30c02 . 69bca8c82c21c15c57c73c30c04c4a c15c57c73c30c04c4aca8c82c29c9bcb6c61
        43069b521257278a87 043c15c57c7aca8c82c29c9bcb6c61 4a802c30c73cb7c9bc69c56c15c21c 6154a8290309b393b7
        02c30c43c84ca8c7acb7c9bc69c56c15c21c 615b7c9bc29c02c30c43c84ca8c7ac . 615c29c9bcb7c73c30c04c4aca8c82
        043c7aca8c82c21c15c56c69c9bcb7 6150432a8a2797279b 735021498946a646ab 735c4acabcb6c61c10c02c29c98c84
        021c69c98c84c43c35c57c7acabcb6 . 573c10c04c4acabcb6c69c98c82c21 29873561bab10a1a04
        b6cabc7ac57c35c43c04c10c21c82c98c69c 29804c10c61cb6cabc7ac57c35c43c 21084c98c69c56c35c73cb7cabc4ac
        c98c84c4acabcb7c73c35c56c61c10c02c29 7ab210869684645354 7abc98c84c43c35c56c61c10c02c29 .
        829c61c10c04c4acabcb7c73c35c56 7abc43c35c56c69c98c82c21c10c04 7ab829635364146410 . . . . .
        82961cb6cabc4ac04c30c73c57c15c 0437ac57c15c21c82c98c69cb6cabc 829043b6a1a6a17571 .
        61529c02c30c73cb7cabc4ac84c98c 7ab43c84c98c69c56c15c21c02c30c 7ab615023932394849 . 615298ab404b70b073
        7ab043956598581218 043615829b7a
        6a: 69b87a785058 c9bcb7c7aca8c80c05c56c69 05c80c98c69cb6cabc7ac57c 7ab869685058
        6b: 2a8a25152a57b69 c15c57c7aca8c82c29c9bcb6c61 c7aca8c82c21c15c56c69c9bcb7 6152a8a2797279b
        7ac57c15c21c82c98c69cb6cabc 298b6a1a6a17571 7ab956598581218 b7a561829
        6c: 84a3b7b39193 19c31c43c84ca8c7acb7c9bc c4acabcb7c73c31c19c98c84 b7a384839193
        6d: 023b323b79b24a8 b7c9bc29c02c30c43c84ca8c7ac c29c9bcb7c73c30c04c4aca8c82 0432a8a2797279b
        29c02c30c73cb7cabc4ac84c98c 7ab023932394849 298ab404b70b073 2987ab430
        6e: 9b171b170a070a8 7ab190809
        6f: 79b72982787a 829ab7
        70: ab6a64462
        71: 6ab1a614a104
        72: 0620a603ab6a
        73: 6ab61a13a
        74: a24a52ab5512
        75: a04ab0b50
        76: a030a2b2a2b151b 012b3a3b5
        77: 5ab3a5
        78: 735ab464b462 7ac57c35c43c24c62cb6cabc c62c24c4acabcb7c73c35c56 7ab354645462
        79: b6a0a6a04106735 04c10c61cb6cabc7ac57c35c43c c61c10c04c4acabcb7c73c35c56 7ab635364146410
        61cb6cabc4ac04c30c73c57c15c 043b6a1a6a17571 615307b704b0b4a 6154307ab
        7a: 2b6b2a0a2a07570 b7a062605
        7b: a57a15ab61a6 615b7a
        7c: 4a2b2a2b171b173 b7a241314
        7d: 0730b704ab0a ab7430
        7e: 201ab7|0a2a0717071b2b1b2a
        7f: b7a
        80: ba7
        81: 1027ba|2a070a071b171b2a2b
        82: 7ba034 3707b0a40a0b
        83: a7b142413 2a4a2b1b2b17371
        84: 516a7b 75a51a6ba6a1
        85: a7b260506 6b2a2b2a070a075
        86: 516034ba7 51670307b0b4a4b 340a6b6a171a175 c16c6bcbaca4c40c03c37c75c51 ba7536463641014
        16c01c40ca4cbac7bc37c53c65c c40c01c16c6bcbaca7c75c53c34 a6b6a040a601537
        87: ba7453546264 26c42ca4cbac7bc37c53c65c ca7c75c53c34c42c26c6bcba 5374bab46264
        88: 5ba35a
        89: 210a3b5b3 30a2a0a2b1b2b15
        8a: a40a0bb05
        8b: a42a25a5b521
        8c: 6ba6a11a3
        8d: 02606a0a3ba6
        8e: 6ba16a1a4140
        8f: a6ba46426
        90: 9287ba b97927728a78
        91: ba7091908 1b9b1707170a8a0
        92: 892ba7034 8924bab40b07370 ba7320239493948 c92c20c03c37c7bcbaca4c48c89 3408a272a279b97
        92cb9c7bc37c03c40ca4c8ac28c c7bcb9c92c20c03c34c48c8aca7 32023b7b32b98a4
        93: a7b483938391 a4cbac7bc37c13c91c89c48c c91c13c34c48c8aca7c7bcb9 a487b393b391
        94: a7b165928 ba7659895185812 892a6b6a171a175 ca7c75c51c12c28c89c96c6bcba 5168a272a279b97
        a7c8ac28c12c51c65c96cb9c7bc 51c75ca7c8ac28c92cb9c6bc16c 8a252a25175a96b
        95: ba7968586850 c50c08c89c96c6bcbaca7c75 b9c7bca7c8ac08c50c65c96c b96a78587850
        96: 340516928a7b ba7340659895185812 5168924bab40b07370 . ba7516320239493948 ba7c34c48c89c96c65c51c12c20c03
        516c92c20c03c37c7bcbaca4c48c89 . 928340a6b6a171a175 340ca7c75c51c12c28c89c96c6bcba
        928c16c6bcbaca4c40c03c37c75c51 . . . . . ba7928536463641014 ba734c53c65c96c89c28c12c01c40c
        89216c01c40ca4cbac7bc37c53c65c . a7bc65c53c34c48c89c92c20c01c16 120a7b968486546453
        89c48ca4cbac7bc37c53c65c16c01c20c92c 01253c65c96c89c48ca4cbac7bc37c 928c40c01c16c6bcbaca7c75c53c34
        c6bcbaca7c75c53c34c40c01c12c28c89c96 375928b161ba1a040a 537c6bcbaca4c40c01c12c28c89c96 .
        01296c89c48c34c53c75ca7cbac6bc 537a4cbac6bc16c01c20c92c89c48c 01253789464946aba6 5163408a272a279b97
        340a7c8ac28c12c51c65c96cb9c7bc 51692cb9c7bc37c03c40ca4c8ac28c . 16503c20c92cb9c7bca7c8ac48c34c
        c20c03c34c48c8aca7c7bcb9c96c65c51c12 a4816509290393b7b3 8a4c20c03c37c7bcb9c96c65c51c12
        34051c75ca7c8ac28c92cb9c6bc16c 96b34012575287278a 51c75c37c03c40ca4c8ac28c92cb9c6bc16c
        96b8ac28c12c51c75c37c03c40ca4c . 96b12c51c75ca7c8ac48c34c03c20c 8a437c03c20c92cb9c6bc16c51c75c
        8a496b517271723032 . . . . . 012c96c65c53c34c48c8aca7c7bcb9 8a4c37c7bcb9c92c20c01c16c65c53 8a401297b793639365 .
        96bc12c28c8aca7c75c53c34c40c01 537ca4c40c01c16c6bcb9c92c28c8a 96b53714041a2a1a28 . 96b120483538a585a7
        8a45372b9b20b06160 53796b0128a4
        97: 894549453659a7b 53c65c96c89c48ca4cbac7bc37c 96c89c48c34c53c75ca7cbac6bc 53789464946aba6
        c96c65c53c34c48c8aca7c7bcb9 8a497b793639365 96b75a8a5385834 96b375a48
        98: 8925a3a5b 38a832523259b95
        99: 3bab35890109 bac5bc15c91c89c08c30ca3c c35c5bcb9c91c10c08c8aca3 38a830b95159|30153135b3ba8ab98b891810
        9a: 9280a4a0b5b0 5bc05c20c92c89c48ca4cbac c28c8aca4c40c05c5bcb9c92 a4809290b5b0
        9b: 18981454145aba5 8a459195b
        9c: 9286b1a1b1a3 96c89c28c12c31ca3cbac6bc ca3c31c16c6bcb9c92c28c8a 96b281a181a3
        9d: 3bab36063609890 b960a3a08
        9e: 14041b6b14ba892 c6bcbaca4c40c01c12c28c89c96 a4cbac6bc16c01c20c92c89c48c 12089464946aba6
        ca4c40c01c16c6bcb9c92c28c8a 96b14041a2a1a28 8a42b9b20b06160 8a4b96012
        9f: 4ba46b489649 96b8a4
        a0: b847b4
        a1: 021b474b8 704071b171b282b
        a2: 37b3b00b8
        a3: 21323b2b87b3
        a4: 165478b87 75414541868186b
        a5: 02565278487b 7bc47c04c50c65c26c82cb8c c84c47c75c50c02c26c6bcb8 82b6b2704075|56b5b747b84b482420502652
        a6: 165b373b080b 80cb8c6bc16c51c75c37c03c c65c53c37c7bcb8c80c01c16 375b161b080b
        a7: 65232523878387b 375268b86
        a8: 43545885b
        a9: 210543458b85 b8c5bc35c43c04c10c21c82c c30c02c21c15c5bcb8c84c43 430521258b85
        aa: 80bb05
        ab: 1821585b8
        ac: 6b8683631384
        ad: 602063b363b484b 043b262b8
        ae: 6016b0b80
        af: 86b268
        b0: b92b27724
        b1: 0910790477b9
        b2: b3793b903920
        b3: b37b93913
        b4: 5162b9b27472 c96c65c51c12c24c47c7bcb9 47c24c92cb9c6bc16c51c75c b96251527472
        b5: 7b494b490609065 96b470507
        b6: 97b790209703516 c20c03c37c7bcb9c96c65c51c12 37c03c20c92cb9c6bc16c51c75c 96b517271723032
        c37c7bcb9c92c20c01c16c65c53 12097b793639365 53760106b0b292b 537012b96
        b7: 36539637b93b b96753
        b8: 95b945924354
        b9: 10949094b3b4b35 43091b5b1
        ba: 2b920b05b
        bb: 1b95b1
        bc: 49294b3b4b36163 96b324231
        bd: 96b043|b363b494b490609063
        be: b16b01b920b2 20196b
        bf: 96b
        c0: 9a7697
        c1: 02176a9a6 61707170a2a0a29
        c2: 34097679a 637360906094a49
        c3: a696a7214341 c34c42c29c9aca7c76c61c13 21c42ca4c9ac69c76c37c13c 361637294a49|37a43a34232161296269a6a7
        c4: 75171aa19
        c5: 2502a529aa75
        c6: 34075a1a5a19 37c03c40ca4c9ac19c51c75c c19c9aca7c75c53c34c40c01 37540a1a0a19
        c7: a79597952325234 5379a242a
        c8: 56959339a
        c9: 021956593a39 a3c9ac29c02c10c61c56c35c c21c15c56c69c9aca3c30c02 561902093a39
        ca: 69a6a060540a
        cb: a696a545a541214 61549a942
        cc: 9a11a3
        cd: 2302939a3
        ce: 0a401a19a
        cf: 2a49a2
        d0: 8a7872276
        d1: a08a60a76106
        d2: 03478a872627 62c76c37c03c40ca4c8ac28c c34c48c8aca7c76c62c20c03 8a4703072627
        d3: 341814186a686a7 8a4136763
        d4: 1752712a728a
        d5: 78a758508
        d6: 51787178a281340 8ac28c12c51c75c37c03c40ca4c 12c51c75ca7c8ac48c34c03c20c 8a4517271723032
        c12c28c8aca7c75c53c34c40c01 53714041a2a1a28 120483538a585a7 120a48375
        d7: 5a758a534854 48a375
        d8: 8a3836862635
        d9: 35a6a5a68186810 615a38083
        da: 286a686a545a540 a48625052
        db: 8a4615|54145a6a5a68186814
        dc: 8128a1a31
        dd: 830a38
        de: 1401a4128a18 8a4201
        df: 8a4
        e0: 984946647
        e1: 210986468647 c47c76c61c10c02c29c98c84 21c82c98c69c76c47c04c10c 298106460647
        e2: 976907980037
        e3: 321238783879697 298713176
        e4: 847871819517
        e5: 784879597952025 829547450
        e6: 951597879783038 537819180
        e7: 298537|879783238325952597
        e8: 435845856869
        e9: 635368986384021 c98c84c43c35c56c61c10c02c29 c43c35c56c69c98c82c21c10c04 298635364146410
        43c84c98c69c56c15c21c02c30c 615023932394849 043182815859695 043829561
        ea: 956985805
        eb: 521582569859 982615
        ec: 498439319
        ed: 384398302932 298043
        ee: 901809
        ef: 298
        f0: 246476
        f1: 061046476
        f2: 320372762
        f3: 361763
        f4: 175127247
        f5: 704507
        f6: 703720751271 120537
        f7: 375
        f8: 543564624
        f9: 635643610460 043156
        fa: 256052
        fb: 156
        fc: 412314
        fd: 430
        fe: 201
        """
}

/// Hi3DGen's mesh from the TRELLIS mesh decoder's sparse features (`SparseFeatures2Mesh` with
/// `EnhancedMarchingCubes`). A voxel's 101 channels are eight corner SDFs, eight corner deformations (3),
/// 21 FlexiCubes weights this extraction does not read, and eight corner colors (6).
enum NFKHi3DGenSurface {
    static let featureChannels = 101

    /// The surface of the voxels at `coordinates` (x, y, z triples on a `resolution`³ grid, 256 for the
    /// release) with decoder `features` (`featureChannels` a voxel, row-major): positions in the decoder's
    /// frame (−0.5...0.5, z up), colors as the sigmoid of the six interpolated color channels, and
    /// triangles wound counter-clockwise seen from outside.
    static func surface(coordinates: [Int32], features: [Float], resolution: Int) -> NFKTrellisSurface {
        let corners = NFKHi3DGenCorners.averaged(coordinates: coordinates, features: features, resolution: resolution)
        return surface(NFKHi3DGenMarchingCubes.surface(of: corners), corners: corners)
    }

    /// The marching-cubes surface of `corners` moved and colored. A vertex at grid position p reads the
    /// eight grid points from ⌊p⌋, the upper ones clamped to the grid, trilinearly along x, then y, then z;
    /// a grid point g sits at g / resolution − 0.5 plus (1 − 1e-8) / (2 · resolution) times the tanh of its
    /// deformation, and a point the set lacks has no deformation and color 0. The triangles reverse.
    static func surface(_ marched: NFKHi3DGenMarchingCubes.Surface, corners: NFKHi3DGenCorners) -> NFKTrellisSurface {
        let r = corners.resolution, side = r + 1
        let width = NFKHi3DGenCorners.attributeCount
        let count = marched.vertices.count / 3
        let scale = Float((1 - 1e-8) / Double(2 * r))
        let attributes = corners.attributes, cells = corners.cells, starts = corners.planeStarts

        // The vertices by the plane of their lower grid point.
        var vertexStarts = [Int](repeating: 0, count: side + 1)
        for vertex in 0 ..< count {
            vertexStarts[min(Int(marched.vertices[vertex * 3]), r) + 1] += 1
        }
        for x in 0 ..< side {
            vertexStarts[x + 1] += vertexStarts[x]
        }
        var byPlane = [Int](repeating: 0, count: count)
        var next = vertexStarts
        for vertex in 0 ..< count {
            let x = min(Int(marched.vertices[vertex * 3]), r)
            byPlane[next[x]] = vertex
            next[x] += 1
        }

        var positions = [Float](repeating: 0, count: count * 3)
        var colors = [Float](repeating: 0, count: count * 6)
        var lowerPlane = [Int32](repeating: -1, count: side * side), upperPlane = lowerPlane
        let values = UnsafeMutableBufferPointer<Float>.allocate(capacity: 8 * 9)
        values.initialize(repeating: 0)
        defer { values.deallocate() }
        func place(_ map: inout [Int32], plane: Int, _ value: (Int) -> Int32) {
            for point in starts[plane] ..< starts[plane + 1] {
                map[Int(cells[point])] = value(point)
            }
        }
        for x in 0 ... r where vertexStarts[x] < vertexStarts[x + 1] {
            let x1 = min(x + 1, r)
            place(&lowerPlane, plane: x) { Int32($0) }
            place(&upperPlane, plane: x1) { Int32($0) }
            for vertex in byPlane[vertexStarts[x] ..< vertexStarts[x + 1]] {
                let p = (marched.vertices[vertex * 3], marched.vertices[vertex * 3 + 1], marched.vertices[vertex * 3 + 2])
                let (gy, gz) = (min(Int(p.1), r), min(Int(p.2), r))
                let local = (p.0 - Float(Int(p.0)), p.1 - Float(Int(p.1)), p.2 - Float(Int(p.2)))
                // Corner (dx, dy, dz) at dx · 4 + dy · 2 + dz: three deformed coordinates, then six colors.
                for dx in 0 ... 1 {
                    for dy in 0 ... 1 {
                        for dz in 0 ... 1 {
                            let (px, py, pz) = (dx == 0 ? x : x1, dy == 0 ? gy : min(gy + 1, r), dz == 0 ? gz : min(gz + 1, r))
                            let cell = py * side + pz
                            let point = Int(dx == 0 ? lowerPlane[cell] : upperPlane[cell])
                            let base = (dx * 4 + dy * 2 + dz) * 9
                            values[base] = Float(px) / Float(r) - 0.5
                            values[base + 1] = Float(py) / Float(r) - 0.5
                            values[base + 2] = Float(pz) / Float(r) - 0.5
                            for k in 0 ..< 6 {
                                values[base + 3 + k] = 0
                            }
                            guard point >= 0 else { continue }
                            for k in 0 ..< 3 {
                                values[base + k] += scale * tanh(attributes[point * width + 1 + k])
                            }
                            for k in 0 ..< 6 {
                                values[base + 3 + k] = attributes[point * width + 4 + k]
                            }
                        }
                    }
                }
                for channel in 0 ..< 9 {
                    func c(_ corner: Int) -> Float { values[corner * 9 + channel] }
                    let c00 = c(0) * (1 - local.0) + c(4) * local.0
                    let c01 = c(1) * (1 - local.0) + c(5) * local.0
                    let c10 = c(2) * (1 - local.0) + c(6) * local.0
                    let c11 = c(3) * (1 - local.0) + c(7) * local.0
                    let c0 = c00 * (1 - local.1) + c10 * local.1
                    let c1 = c01 * (1 - local.1) + c11 * local.1
                    let blended = c0 * (1 - local.2) + c1 * local.2
                    if channel < 3 {
                        positions[vertex * 3 + channel] = blended
                    } else {
                        colors[vertex * 6 + channel - 3] = 1 / (1 + exp(-blended))
                    }
                }
            }
            place(&lowerPlane, plane: x) { _ in -1 }
            place(&upperPlane, plane: x1) { _ in -1 }
        }

        var triangles = marched.triangles
        for face in 0 ..< triangles.count / 3 {
            triangles.swapAt(face * 3, face * 3 + 2)
        }
        return NFKTrellisSurface(positions: positions, colors: colors, triangles: triangles)
    }
}
