//
//  NFKMLXTrellisMesh.swift
//  InferKitMLX
//
//  FlexiCubes surface extraction from TRELLIS's mesh features, on the CPU in float32. The dual
//  marching cubes table is derived at first use from the cube's geometry: one dual vertex per
//  surface patch, an ambiguous face keeping its two inside corners apart.
//

import Foundation

/// The dual marching cubes cases. Corner `i` of a cube sits at (i & 1, (i >> 1) & 1, (i >> 2) & 1); a
/// case sets bit `i` when corner `i` is inside (its SDF negative).
enum NFKTrellisDualCubes {
    /// The twelve edges as corner pairs, each in the direction every cube sharing it lists it.
    static let edges: [(Int, Int)] = [(0, 1), (1, 5), (4, 5), (0, 4), (2, 3), (3, 7), (6, 7), (2, 6),
                                      (2, 0), (3, 1), (7, 5), (6, 4)]

    struct Case {
        /// One crossing-edge group per dual vertex, edges ascending; groups ordered by the sorted
        /// minority-sign corners each encloses (the inside corners when four or fewer are inside).
        let groups: [[Int]]
        /// For a case with one ambiguous face and a single dual vertex: that face's outward normal. Where
        /// the cube across that face is such a case too, both take their complement's grouping, so the
        /// shared face reads the same in both.
        let ambiguousNormal: (Int, Int, Int)?
    }

    static let cases: [Case] = (0 ..< 256).map(derive)

    static func corner(_ index: Int) -> (Int, Int, Int) { (index & 1, (index >> 1) & 1, (index >> 2) & 1) }

    /// Each face's corners as a cycle, with its axis and side.
    static let faces: [(cycle: [Int], axis: Int, side: Int)] = (0 ..< 3).flatMap { axis in
        (0 ... 1).map { side -> (cycle: [Int], axis: Int, side: Int) in
            let others = (0 ..< 3).filter { $0 != axis }
            func coordinate(_ c: Int, _ a: Int) -> Int { [corner(c).0, corner(c).1, corner(c).2][a] }
            let corners = (0 ..< 8).filter { coordinate($0, axis) == side }
                .sorted { (coordinate($0, others[0]), coordinate($0, others[1])) < (coordinate($1, others[0]), coordinate($1, others[1])) }
            return ([corners[0], corners[1], corners[3], corners[2]], axis, side)
        }
    }

    static func edge(_ a: Int, _ b: Int) -> Int {
        edges.firstIndex { ($0.0 == a && $0.1 == b) || ($0.0 == b && $0.1 == a) }!
    }

    static func derive(_ code: Int) -> Case {
        let inside = (0 ..< 8).map { (code >> $0) & 1 == 1 }
        let crossing = (0 ..< 12).filter { inside[edges[$0].0] != inside[edges[$0].1] }
        var parent = Array(0 ..< 12)
        func find(_ e: Int) -> Int {
            var e = e
            while parent[e] != e {
                parent[e] = parent[parent[e]]
                e = parent[e]
            }
            return e
        }
        func union(_ a: Int, _ b: Int) { parent[find(a)] = find(b) }
        var ambiguous = [(Int, Int, Int)]()
        for face in faces {
            let cycle = face.cycle
            let sides = (0 ..< 4).map { edge(cycle[$0], cycle[($0 + 1) % 4]) }
            let crossed = sides.filter { crossing.contains($0) }
            if crossed.count == 2 {
                union(crossed[0], crossed[1])
            } else if crossed.count == 4 {
                var normal = [0, 0, 0]
                normal[face.axis] = face.side == 1 ? 1 : -1
                ambiguous.append((normal[0], normal[1], normal[2]))
                for i in 0 ..< 4 where inside[cycle[i]] {
                    union(sides[(i + 3) % 4], sides[i])
                }
            }
        }
        var members = [Int: [Int]]()
        for e in crossing {
            members[find(e), default: []].append(e)
        }
        let minorityInside = inside.filter { $0 }.count <= 4
        func key(_ group: [Int]) -> [Int] {
            Set(group.map { e -> Int in
                let (a, b) = edges[e]
                return inside[a] == minorityInside ? a : b
            }).sorted()
        }
        let groups = members.values.map { $0.sorted() }.sorted { key($0).lexicographicallyPrecedes(key($1)) }
        let flagged = ambiguous.count == 1 && groups.count == 1
        return Case(groups: groups, ambiguousNormal: flagged ? ambiguous[0] : nil)
    }
}

/// A FlexiCubes surface: positions `[V, 3]`, colors `[V, 6]` (red, green, blue in 0...1, then the
/// decoder's three normal-map channels), and counter-clockwise triangles `[F, 3]`.
struct NFKTrellisSurface {
    var positions: [Float]
    var colors: [Float]
    var triangles: [UInt32]
    var vertexCount: Int { positions.count / 3 }
}

/// FlexiCubes over sparse voxels, at inference (no regularizer, no training-time centroid split).
enum NFKTrellisFlexiCubes {
    static let featureChannels = 101
    static let weightScale: Float = 0.99

    /// The surface of the voxels at `coordinates` (x, y, z triples on a `resolution`³ grid) whose
    /// decoder features are `features` (`featureChannels` per voxel): per corner an SDF, a deformation,
    /// and a color; per cube twelve edge weights β, eight corner weights α, and a split weight γ. A
    /// vertex shared by several voxels takes the mean of their values; a vertex no voxel touches reads
    /// outside. Positions span −0.5...0.5 with the grid's z up.
    static func surface(coordinates: [Int32], features: [Float], resolution: Int) -> NFKTrellisSurface {
        let count = coordinates.count / 3
        let side = resolution + 1
        let sdfBias = -1 / Float(resolution)
        func vertexID(_ x: Int, _ y: Int, _ z: Int) -> Int { (x * side + y) * side + z }

        // Vertex attributes: SDF, deformation (3), color (6), each the mean over the voxels sharing it.
        var slots = [Int: Int](minimumCapacity: count * 3)
        var sums = [Float]()
        var counts = [Float]()
        sums.reserveCapacity(count * 30)
        for voxel in 0 ..< count {
            let (x, y, z) = (Int(coordinates[3 * voxel]), Int(coordinates[3 * voxel + 1]), Int(coordinates[3 * voxel + 2]))
            let row = voxel * featureChannels
            for c in 0 ..< 8 {
                let (cx, cy, cz) = NFKTrellisDualCubes.corner(c)
                let id = vertexID(x + cx, y + cy, z + cz)
                let slot: Int
                if let existing = slots[id] {
                    slot = existing
                } else {
                    slot = counts.count
                    slots[id] = slot
                    counts.append(0)
                    sums += [Float](repeating: 0, count: 10)
                }
                counts[slot] += 1
                sums[slot * 10] += features[row + c] + sdfBias
                for k in 0 ..< 3 {
                    sums[slot * 10 + 1 + k] += features[row + 8 + c * 3 + k]
                }
                for k in 0 ..< 6 {
                    sums[slot * 10 + 4 + k] += features[row + 53 + c * 6 + k]
                }
            }
        }
        for slot in 0 ..< counts.count {
            for k in 0 ..< 10 {
                sums[slot * 10 + k] /= counts[slot]
            }
        }
        func sdf(_ id: Int) -> Float { slots[id].map { sums[$0 * 10] } ?? 1 }
        let scale = Float((1 - 1e-8) / Double(resolution * 2))
        func position(_ id: Int) -> (Float, Float, Float) {
            let (x, y, z) = (id / (side * side), id / side % side, id % side)
            var p = (Float(x) / Float(resolution) - 0.5, Float(y) / Float(resolution) - 0.5, Float(z) / Float(resolution) - 0.5)
            if let slot = slots[id] {
                p.0 += scale * tanh(sums[slot * 10 + 1])
                p.1 += scale * tanh(sums[slot * 10 + 2])
                p.2 += scale * tanh(sums[slot * 10 + 3])
            }
            return p
        }
        func color(_ id: Int, _ k: Int) -> Float {
            let value = slots[id].map { sums[$0 * 10 + 4 + k] } ?? 0
            return 1 / (1 + exp(-value))
        }

        // Surface cubes: cubes of the full grid with inside and outside corners, in x-major order.
        var candidates = Set<Int>()
        for (id, slot) in slots where sums[slot * 10] < 0 {
            let (x, y, z) = (id / (side * side), id / side % side, id % side)
            for dx in 0 ... 1 {
                for dy in 0 ... 1 {
                    for dz in 0 ... 1 {
                        let (cx, cy, cz) = (x - dx, y - dy, z - dz)
                        if cx >= 0, cy >= 0, cz >= 0, cx < resolution, cy < resolution, cz < resolution {
                            candidates.insert((cx * resolution + cy) * resolution + cz)
                        }
                    }
                }
            }
        }
        var cubes = [Int]()
        var cases = [Int]()
        var cornerIDs = [Int]()
        for cube in candidates.sorted() {
            let (x, y, z) = (cube / (resolution * resolution), cube / resolution % resolution, cube % resolution)
            var code = 0
            var ids = [Int]()
            for c in 0 ..< 8 {
                let (cx, cy, cz) = NFKTrellisDualCubes.corner(c)
                let id = vertexID(x + cx, y + cy, z + cz)
                ids.append(id)
                if sdf(id) < 0 {
                    code |= 1 << c
                }
            }
            if code != 0 && code != 255 {
                cubes.append(cube)
                cases.append(code)
                cornerIDs += ids
            }
        }
        let surfaceCount = cubes.count
        guard surfaceCount > 0 else { return NFKTrellisSurface(positions: [], colors: [], triangles: []) }
        var surfaceIndex = [Int: Int](minimumCapacity: surfaceCount)
        for (index, cube) in cubes.enumerated() {
            surfaceIndex[cube] = index
        }

        // A one-vertex case across an ambiguous face from another such case takes its complement.
        let original = cases
        for index in 0 ..< surfaceCount {
            guard let normal = NFKTrellisDualCubes.cases[original[index]].ambiguousNormal else { continue }
            let cube = cubes[index]
            let (x, y, z) = (cube / (resolution * resolution) + normal.0, cube / resolution % resolution + normal.1,
                             cube % resolution + normal.2)
            guard x >= 0, y >= 0, z >= 0, x < resolution, y < resolution, z < resolution else { continue }
            if let neighbor = surfaceIndex[(x * resolution + y) * resolution + z],
               NFKTrellisDualCubes.cases[original[neighbor]].ambiguousNormal != nil {
                cases[index] = 255 - original[index]
            }
        }

        // Weights: β per edge and α per corner as 1 + 0.99 tanh, γ as 0.005 + 0.99 sigmoid; a cube no
        // voxel covers reads zeros.
        var voxelOf = [Int: Int](minimumCapacity: count)
        for voxel in 0 ..< count {
            voxelOf[(Int(coordinates[3 * voxel]) * resolution + Int(coordinates[3 * voxel + 1])) * resolution
                    + Int(coordinates[3 * voxel + 2])] = voxel
        }
        let gammaOffset = Float((1 - Double(weightScale)) / 2)
        var beta = [Float](repeating: 0, count: surfaceCount * 12)
        var alpha = [Float](repeating: 0, count: surfaceCount * 8)
        var gamma = [Float](repeating: 0, count: surfaceCount)
        for index in 0 ..< surfaceCount {
            let row = voxelOf[cubes[index]].map { $0 * featureChannels + 32 }
            func weight(_ k: Int) -> Float { row.map { features[$0 + k] } ?? 0 }
            for k in 0 ..< 12 {
                beta[index * 12 + k] = tanh(weight(k)) * weightScale + 1
            }
            for k in 0 ..< 8 {
                alpha[index * 8 + k] = tanh(weight(12 + k)) * weightScale + 1
            }
            gamma[index] = 1 / (1 + exp(-weight(20))) * weightScale + gammaOffset
        }

        // Edges: the surface cubes' edges merged and sorted; a surface edge has one inside end.
        let pairs = (0 ..< surfaceCount * 12).map { slot -> (Int, Int) in
            let (a, b) = NFKTrellisDualCubes.edges[slot % 12]
            return (cornerIDs[(slot / 12) * 8 + a], cornerIDs[(slot / 12) * 8 + b])
        }
        let order = (0 ..< pairs.count).sorted { pairs[$0] < pairs[$1] }
        var edgeOfSlot = [Int](repeating: -1, count: pairs.count)
        var sharing = [Int](repeating: 0, count: pairs.count)
        var surfaceEdges = [(Int, Int)]()
        var start = 0
        while start < order.count {
            var end = start + 1
            while end < order.count && pairs[order[end]] == pairs[order[start]] {
                end += 1
            }
            let (a, b) = pairs[order[start]]
            let isSurface = (sdf(a) < 0) != (sdf(b) < 0)
            if isSurface {
                surfaceEdges.append((a, b))
            }
            for k in start ..< end {
                sharing[order[k]] = end - start
                edgeOfSlot[order[k]] = isSurface ? surfaceEdges.count - 1 : -1
            }
            start = end
        }
        let edgePositions = surfaceEdges.map { (position($0.0), position($0.1)) }
        let edgeSDF = surfaceEdges.map { (sdf($0.0), sdf($0.1)) }

        // Dual vertices: per group, the β-weighted mean of its edges' crossings, each crossing placed by
        // the α-weighted SDFs. Cubes go in order of their vertex count, then in cube order.
        var positions = [Float]()
        var colors = [Float]()
        var vertexGamma = [Float]()
        var vertexOfSlot = [Int](repeating: 0, count: surfaceCount * 12)
        let groupCounts = cases.map { NFKTrellisDualCubes.cases[$0].groups.count }
        for vertexCount in Set(groupCounts).sorted() {
            for index in 0 ..< surfaceCount where groupCounts[index] == vertexCount {
                for group in NFKTrellisDualCubes.cases[cases[index]].groups {
                    var sum: (Float, Float, Float) = (0, 0, 0)
                    var colorSum = [Float](repeating: 0, count: 6)
                    var betaSum: Float = 0
                    let vertex = vertexGamma.count
                    for e in group {
                        let slot = index * 12 + e
                        let surfaceEdge = edgeOfSlot[slot]
                        let (ca, cb) = NFKTrellisDualCubes.edges[e]
                        let w0 = edgeSDF[surfaceEdge].0 * alpha[index * 8 + ca]
                        let w1 = edgeSDF[surfaceEdge].1 * alpha[index * 8 + cb]
                        let denominator = w1 + -w0
                        let (pa, pb) = edgePositions[surfaceEdge]
                        let b = beta[slot]
                        sum.0 += (pa.0 * w1 + pb.0 * -w0) / denominator * b
                        sum.1 += (pa.1 * w1 + pb.1 * -w0) / denominator * b
                        sum.2 += (pa.2 * w1 + pb.2 * -w0) / denominator * b
                        let (ia, ib) = surfaceEdges[surfaceEdge]
                        for k in 0 ..< 6 {
                            colorSum[k] += (color(ia, k) * w1 + color(ib, k) * -w0) / denominator * b
                        }
                        betaSum += b
                        vertexOfSlot[slot] = vertex
                    }
                    positions += [sum.0 / betaSum, sum.1 / betaSum, sum.2 / betaSum]
                    colors += colorSum.map { $0 / betaSum }
                    vertexGamma.append(gamma[index])
                }
            }
        }

        // Quads: each surface edge shared by four surface cubes joins their four dual vertices, wound by
        // the edge's sign; the quad splits along the diagonal whose γ product is larger.
        var quadSlots = [(edge: Int, vertex: Int)]()
        for slot in 0 ..< pairs.count where sharing[slot] == 4 && edgeOfSlot[slot] >= 0 {
            quadSlots.append((edgeOfSlot[slot], vertexOfSlot[slot]))
        }
        let sortedSlots = quadSlots.enumerated().sorted { ($0.element.edge, $0.offset) < ($1.element.edge, $1.offset) }.map(\.element)
        var flipped = [[Int]]()
        var unflipped = [[Int]]()
        var index = 0
        while index + 3 < sortedSlots.count {
            let quad = sortedSlots[index ..< index + 4].map(\.vertex)
            if edgeSDF[sortedSlots[index].edge].0 > 0 {
                flipped.append([quad[0], quad[1], quad[3], quad[2]])
            } else {
                unflipped.append([quad[2], quad[3], quad[1], quad[0]])
            }
            index += 4
        }
        var triangles = [UInt32]()
        triangles.reserveCapacity((flipped.count + unflipped.count) * 6)
        for quad in flipped + unflipped {
            let first = vertexGamma[quad[0]] * vertexGamma[quad[2]]
            let second = vertexGamma[quad[1]] * vertexGamma[quad[3]]
            let split = first > second ? [0, 1, 2, 0, 2, 3] : [0, 1, 3, 3, 1, 2]
            triangles += split.map { UInt32(quad[$0]) }
        }
        return NFKTrellisSurface(positions: positions, colors: colors, triangles: triangles)
    }
}
