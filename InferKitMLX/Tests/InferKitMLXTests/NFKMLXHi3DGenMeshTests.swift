//
//  NFKMLXHi3DGenMeshTests.swift
//  InferKitMLXTests
//
//  The table, crafted-cube, and sphere tests run without weights or MLX. The parity tests read the oracle
//  record (IK_PARITY_HI3DGEN, from `run_reference.py hi3dgen`) and skip without it. The expected figures of
//  the weight-free tests are scikit-image 0.26's `measure.marching_cubes(level=0, gradient_direction='ascent')`
//  on the same float32 grids (2026-11-20).
//

import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXHi3DGenMeshTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func envPath(_ key: String) throws -> URL {
        guard let path = NFKMLXValidationConfig.environment[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key)")
        }
        return URL(fileURLWithPath: path)
    }

    private func record() throws -> [String: MLXArray] {
        try MLX.loadArrays(url: try envPath("IK_PARITY_HI3DGEN"))
    }

    /// Writes a measured figure to standard error, unbuffered, so a later crash in the process keeps it.
    private func report(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    /// A dense `side`³ grid of SDF values (x-major) as a corner set holding every point.
    private func corners(sdf: [Float], side: Int) -> NFKHi3DGenCorners {
        var coordinates = [Int32]()
        var attributes = [Float]()
        for x in 0 ..< side {
            for y in 0 ..< side {
                for z in 0 ..< side {
                    coordinates += [Int32(x), Int32(y), Int32(z)]
                    attributes += [sdf[(x * side + y) * side + z]] + [Float](repeating: 0, count: 9)
                }
            }
        }
        return NFKHi3DGenCorners(resolution: side - 1, coordinates: coordinates, attributes: attributes)
    }

    /// One cube's triangles (corner `c` holding `values[c]`), each vertex named by its edge, 12 inside.
    private func cubeEdges(_ values: [Float]) -> [[Int]] {
        var grid = [Float](repeating: 0, count: 8)
        for c in 0 ..< 8 {
            grid[((c & 1) * 2 + ((c >> 1) & 1)) * 2 + ((c >> 2) & 1)] = values[c]
        }
        let surface = NFKHi3DGenMarchingCubes.surface(of: corners(sdf: grid, side: 2))
        let labels = (0 ..< surface.vertices.count / 3).map { vertex -> Int in
            let p = (0 ..< 3).map { surface.vertices[vertex * 3 + $0] }
            let fractional = (0 ..< 3).filter { p[$0] > 0 && p[$0] < 1 }
            if fractional.count == 3 {
                return 12
            }
            let axis = fractional[0]
            let corner = (0 ..< 3).filter { $0 != axis }.reduce(0) { $0 | (Int(p[$1]) << $1) }
            return NFKHi3DGenMarchingCubes.edges.firstIndex { $0 == (corner, corner | (1 << axis)) }!
        }
        return (0 ..< surface.triangles.count / 3).map { face in (0 ..< 3).map { labels[Int(surface.triangles[face * 3 + $0])] } }
    }

    private func ulps(_ a: Float, _ b: Float) -> Int {
        abs(Int(Int32(bitPattern: a.bitPattern)) - Int(Int32(bitPattern: b.bitPattern)))
    }

    // MARK: - Table

    func testTheTableCoversEveryCaseWithBoundariesFromTheFaceDecisions() {
        let cube = NFKHi3DGenMarchingCubes.self
        func edge(_ a: Int, _ b: Int) -> Int { cube.edges.firstIndex { $0 == (min(a, b), max(a, b)) }! }
        var decided = 0, unreachableMasks = Set<Int>(), unreachable = 0, splits = 0
        for mask in 1 ..< 255 {
            let ambiguous = cube.ambiguousFaces[mask]
            XCTAssertEqual(cube.fills[mask].count, 1 << ambiguous.count, "case \(mask)")
            for (joins, fill) in cube.fills[mask].enumerated() {
                // The surface meets each face along segments joining its crossed edges; an ambiguous face cuts
                // off the corners its decision keeps apart.
                var segments = Set<Set<Int>>()
                for (index, face) in cube.faces.enumerated() {
                    let sides = (0 ..< 4).map { edge(face[$0], face[($0 + 1) % 4]) }
                    let crossed = sides.filter { ((mask >> cube.edges[$0].0) & 1) != ((mask >> cube.edges[$0].1) & 1) }
                    if crossed.count == 2 {
                        segments.insert(Set(crossed))
                    } else if crossed.count == 4 {
                        let join = (joins >> ambiguous.firstIndex(of: index)!) & 1 == 1
                        for k in 0 ..< 4 where ((mask >> face[k]) & 1 == 1) != join {
                            segments.insert([sides[(k + 3) % 4], sides[k]])
                        }
                    }
                }
                var triangulations = [[UInt8]]()
                switch fill {
                case .unreachable:
                    unreachable += 1
                    unreachableMasks.insert(mask)
                case .triangles(let edges):
                    triangulations = [edges]
                case .split(let without, let with):
                    splits += 1
                    triangulations = [without, with]
                }
                for edges in triangulations {
                    decided += 1
                    XCTAssertEqual(edges.count % 3, 0)
                    var directed = [[Int]: Int]()
                    for t in 0 ..< edges.count / 3 {
                        for k in 0 ..< 3 {
                            directed[[Int(edges[t * 3 + k]), Int(edges[t * 3 + (k + 1) % 3])], default: 0] += 1
                        }
                    }
                    XCTAssertTrue(directed.values.allSatisfy { $0 == 1 }, "case \(mask) joins \(joins) repeats a side")
                    let boundary = Set(directed.keys.filter { directed[[$0[1], $0[0]]] == nil }.map { Set($0) })
                    XCTAssertEqual(boundary, segments, "case \(mask) joins \(joins)")
                }
            }
        }
        XCTAssertEqual(decided, 632)
        XCTAssertEqual(splits, 14)
        XCTAssertEqual(unreachable, 36)
        XCTAssertEqual(unreachableMasks, [0x69, 0x96])
    }

    // MARK: - Crafted cubes

    func testAmbiguousFacesFollowTheDecider() {
        // The face z = 0 holds corners 0 and 3 inside: joined where their product exceeds the outside one's.
        XCTAssertEqual(cubeEdges([-0.5, 0.5, 0.5, -0.50001, 0.7, 0.7, 0.7, 0.7]), [[3, 0, 7], [0, 2, 7], [1, 5, 7], [1, 7, 2]])
        XCTAssertEqual(cubeEdges([-0.5, 0.5, 0.50001, -0.5, 0.7, 0.7, 0.7, 0.7]), [[0, 2, 1], [7, 3, 5]])
        XCTAssertEqual(cubeEdges([-0.5, 0.5, 0.5, -0.5, 0.7, 0.7, 0.7, 0.7]), [[0, 2, 1], [7, 3, 5]])
    }

    func testTheInteriorDecidesIsolatedCornersAndOpposedFaces() {
        // Two inside diagonal corners, connected and apart.
        XCTAssertEqual(cubeEdges([-0.8, 0.1, 0.2, 0.25, 0.2, 0.8, 0.85, -0.6]),
                       [[2, 10, 0], [7, 0, 10], [0, 7, 1], [11, 1, 7], [1, 11, 2], [10, 2, 11]])
        XCTAssertEqual(cubeEdges([-0.9, 0.05, 0.1, 0.95, 0.9, 0.3, 0.15, -0.3]), [[1, 0, 2], [7, 11, 10]])
        XCTAssertEqual(cubeEdges([0.05, 0.1, 0.35, -0.45, -0.6, 0.5, 0.3, 0.2]),
                       [[9, 7, 8], [3, 8, 7], [8, 3, 2], [5, 2, 3], [2, 5, 9], [7, 9, 5]])
        XCTAssertEqual(cubeEdges([0.05, 0.85, 0.65, -0.6, -0.25, 0.45, 0.2, 0.75]), [[8, 9, 2], [7, 3, 5]])
        // Two outside diagonal corners.
        XCTAssertEqual(cubeEdges([0.75, -0.05, -0.05, -0.15, -0.15, -0.5, -0.65, 0.8]),
                       [[0, 10, 2], [10, 0, 7], [1, 7, 0], [7, 1, 11], [2, 11, 1], [11, 2, 10]])
        XCTAssertEqual(cubeEdges([0.4, -0.45, -0.65, -0.6, -0.2, -0.75, -0.75, 0.95]), [[2, 0, 1], [10, 11, 7]])
        // Two opposite ambiguous faces (x = 0 and x = 1) joining their inside corners.
        XCTAssertEqual(cubeEdges([0.9, 0.05, -0.5, -0.25, -0.3, -0.2, 0.05, 0.35]),
                       [[3, 7, 10], [3, 10, 4], [2, 4, 10], [9, 2, 10], [2, 9, 6], [2, 6, 1], [3, 1, 6], [7, 3, 6]])
        XCTAssertEqual(cubeEdges([0.6, 0.85, -0.65, -0.4, -0.9, -0.5, 0.4, 0.15]), [[9, 6, 10], [7, 10, 6], [4, 1, 2], [1, 4, 3]])
    }

    func testTheVertexInsideACubeIsTheReciprocalDistanceMean() {
        let values: [Float] = [0.5, 0.05, -0.25, -0.7, -0.5, -0.4, 0.25, 0.1]
        XCTAssertEqual(cubeEdges(values), [[12, 9, 2], [12, 2, 4], [12, 4, 3], [12, 3, 1], [12, 1, 6], [12, 6, 7], [12, 7, 10], [12, 10, 9]])
        var grid = [Float](repeating: 0, count: 8)
        for c in 0 ..< 8 {
            grid[((c & 1) * 2 + ((c >> 1) & 1)) * 2 + ((c >> 2) & 1)] = values[c]
        }
        let surface = NFKHi3DGenMarchingCubes.surface(of: corners(sdf: grid, side: 2))
        XCTAssertEqual(surface.vertices[0 ..< 3].map(\.bitPattern), [0x3f3d1d10, 0x3ed895b3, 0x3ece3bb8])
    }

    func testASphereClosesAsTheReferenceExtractsIt() {
        let side = 13
        let center: (Float, Float, Float) = (6.2, 5.9, 6.1)
        var sdf = [Float]()
        for x in 0 ..< side {
            for y in 0 ..< side {
                for z in 0 ..< side {
                    let (dx, dy, dz) = (Float(x) - center.0, Float(y) - center.1, Float(z) - center.2)
                    sdf.append((dx * dx + dy * dy + dz * dz).squareRoot() - 4.3)
                }
            }
        }
        let grid = corners(sdf: sdf, side: side)
        let marched = NFKHi3DGenMarchingCubes.surface(of: grid)
        let vertexCount = marched.vertices.count / 3, faceCount = marched.triangles.count / 3
        XCTAssertEqual(vertexCount, 360)
        XCTAssertEqual(faceCount, 716)
        XCTAssertEqual(marched.vertices.reduce(0) { $0 + Double($1) }, 6568.0253251791, accuracy: 1e-9)

        let surface = NFKHi3DGenSurface.surface(marched, corners: grid)
        var directed = Set<[UInt32]>()
        var volume: Double = 0
        for face in 0 ..< faceCount {
            let ids = (0 ..< 3).map { surface.triangles[face * 3 + $0] }
            for k in 0 ..< 3 {
                directed.insert([ids[k], ids[(k + 1) % 3]])
            }
            let p = ids.map { i in (0 ..< 3).map { Double(surface.positions[Int(i) * 3 + $0]) } }
            volume += (p[0][1] * p[1][2] - p[0][2] * p[1][1]) * p[2][0] + (p[0][2] * p[1][0] - p[0][0] * p[1][2]) * p[2][1]
                + (p[0][0] * p[1][1] - p[0][1] * p[1][0]) * p[2][2]
        }
        XCTAssertEqual(directed.count, faceCount * 3)
        XCTAssertTrue(directed.allSatisfy { directed.contains([$0[1], $0[0]]) }, "every edge meets its reverse")
        XCTAssertEqual(vertexCount - directed.count / 2 + faceCount, 2)
        // Wound counter-clockwise from outside: positive volume, about (4.3 / 12)³ · 4π / 3.
        XCTAssertEqual(volume / 6, 4 / 3 * Double.pi * pow(4.3 / 12, 3), accuracy: 0.01)
        XCTAssertTrue(surface.colors.allSatisfy { $0 == 0.5 })
    }

    // MARK: - Parity

    func testCornerAveragingMatchesTheRecordBitForBit() throws {
        try requireMLXRuntime()
        let reference = try record()
        let coordinates = reference["mesh_decoder_out.coords"]![0..., 1...].asType(.int32).reshaped([-1]).asArray(Int32.self)
        let corners = NFKHi3DGenCorners.averaged(coordinates: coordinates, features: reference["mesh_decoder_out"]!.asArray(Float.self),
                                                 resolution: 256)
        let expected = reference["mesh.corner_attrs"]!.asArray(Float.self)
        let sameCoordinates = corners.coordinates == reference["mesh.corner_coords"]!.asType(.int32).reshaped([-1]).asArray(Int32.self)
        let worst = corners.attributes.count == expected.count ? zip(corners.attributes, expected).map(ulps).max() ?? 0 : -1
        report("Hi3DGen corners: \(corners.count) against \(expected.count / 10), coordinates identical \(sameCoordinates), max ulp \(worst)")
        XCTAssertTrue(sameCoordinates)
        XCTAssertEqual(worst, 0)
    }

    func testMarchingCubesMatchesTheRecordInOrder() throws {
        try requireMLXRuntime()
        let reference = try record()
        let corners = NFKHi3DGenCorners(resolution: 256,
                                        coordinates: reference["mesh.corner_coords"]!.asType(.int32).reshaped([-1]).asArray(Int32.self),
                                        attributes: reference["mesh.corner_attrs"]!.asArray(Float.self))
        let surface = NFKHi3DGenMarchingCubes.surface(of: corners)
        let vertices = reference["mesh.marching_vertices"]!.asArray(Float.self)
        let faces = reference["mesh.marching_faces"]!.asType(.int32).asArray(Int32.self)
        let sameVertices = surface.vertices.map(\.bitPattern) == vertices.map(\.bitPattern)
        let sameFaces = surface.triangles.count == faces.count && zip(surface.triangles, faces).allSatisfy { Int64($0) == Int64($1) }
        report("Hi3DGen marching cubes: \(surface.vertices.count / 3) vertices against \(vertices.count / 3), "
               + "\(surface.triangles.count / 3) faces against \(faces.count / 3); vertices bitwise \(sameVertices), faces identical \(sameFaces)")
        XCTAssertTrue(sameVertices)
        XCTAssertTrue(sameFaces)
    }

    func testTheSurfaceMatchesTheRecord() throws {
        try requireMLXRuntime()
        let reference = try record()
        let coordinates = reference["mesh_decoder_out.coords"]![0..., 1...].asType(.int32).reshaped([-1]).asArray(Int32.self)
        let surface = NFKHi3DGenSurface.surface(coordinates: coordinates, features: reference["mesh_decoder_out"]!.asArray(Float.self),
                                                resolution: 256)
        let positions = reference["mesh_vertices"]!.asArray(Float.self)
        let colors = reference["mesh_vertex_attrs"]!.asArray(Float.self)
        let faces = reference["mesh_faces"]!.asType(.int32).asArray(Int32.self)
        let sameFaces = surface.triangles.count == faces.count && zip(surface.triangles, faces).allSatisfy { Int64($0) == Int64($1) }
        XCTAssertEqual(surface.positions.count, positions.count)
        XCTAssertEqual(surface.colors.count, colors.count)
        guard surface.positions.count == positions.count, surface.colors.count == colors.count else { return }
        let positionDifference = zip(surface.positions, positions).map { abs($0 - $1) }.max() ?? 0
        let exactPositions = zip(surface.positions, positions).filter { $0.bitPattern == $1.bitPattern }.count
        let colorULPs = zip(surface.colors, colors).map(ulps)
        report("Hi3DGen surface: faces identical \(sameFaces); positions max difference \(positionDifference), "
               + "\(exactPositions) of \(positions.count) exact; colors max \(colorULPs.max() ?? 0) ulp, "
               + "\(colorULPs.filter { $0 == 0 }.count) of \(colors.count) exact")
        XCTAssertTrue(sameFaces)
        // The reference's vectorized tanh and exp differ from libm's by an ulp: a position moves by at most
        // one ulp of 0.5, a color by two ulps.
        XCTAssertLessThanOrEqual(positionDifference, 0x1p-24)
        XCTAssertLessThanOrEqual(colorULPs.max() ?? 0, 2)
    }
}
