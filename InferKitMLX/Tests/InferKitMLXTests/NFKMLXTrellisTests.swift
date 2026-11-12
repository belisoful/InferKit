//
//  NFKMLXTrellisTests.swift
//  InferKitMLXTests
//
//  The structural tests run tiny configurations with random weights. The parity tests read the release
//  directory (IK_VAL_TRELLIS_IMAGE_LARGE), the DINOv2 checkpoint (IK_VAL_DINOV2_VITL14_REG), and the oracle
//  record (IK_PARITY_TRELLIS, from `run_reference.py trellis`), and skip without them.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXTrellisTests: XCTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// A 28-pixel image of 2 × 2 patches into a 16-wide encoder, a 4³ latent of 2 channels through one
    /// 32-wide flow block, and a decoder of 8 then 4 channels.
    private func tinyConfiguration() -> NFKMLXTrellisConfiguration {
        var configuration = NFKMLXTrellisConfiguration()
        configuration.imageSize = 28
        configuration.patchSize = 14
        configuration.encoderPositionGrid = 2
        configuration.encoderWidth = 16
        configuration.encoderDepth = 1
        configuration.encoderHeads = 2
        configuration.encoderHidden = 32
        configuration.registerTokens = 2
        configuration.conditionWidth = 16
        configuration.structureResolution = 4
        configuration.structureChannels = 2
        configuration.flowWidth = 32
        configuration.flowBlocks = 1
        configuration.flowHeads = 2
        configuration.flowHidden = 64
        configuration.frequencyWidth = 16
        configuration.decoderChannels = [8, 4]
        configuration.decoderResidualBlocks = 1
        configuration.decoderMiddleBlocks = 1
        return configuration
    }

    private func envPath(_ key: String) throws -> URL {
        guard let path = NFKMLXValidationConfig.environment[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key)")
        }
        return URL(fileURLWithPath: path)
    }

    private func checkpoint(_ name: String) throws -> URL {
        try envPath("IK_VAL_TRELLIS_IMAGE_LARGE").appendingPathComponent("ckpts/\(name).safetensors")
    }

    private func record() throws -> [String: MLXArray] {
        try MLX.loadArrays(url: try envPath("IK_PARITY_TRELLIS"))
    }

    // The float32 cosine bars are thresholds, the convention the float32 ports hold (TripoSR, Stable Fast 3D):
    // 0.9999999999 for each network's forward, 0.999999999 for the 25-step guided sampler chains, which compound
    // each forward's float32 differences (text-large's structure chain reads 0.99999999988), and 0.9999999 for
    // CLIP's condition. The reference's own float32-against-float64 spread is not measured, so no bar is a floor.
    // Bytes, voxels, faces, and the sphere's counts are exact.
    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).reshaped([-1]).asArray(Float.self), y = b.asType(.float32).reshaped([-1]).asArray(Float.self)
        XCTAssertEqual(x.count, y.count)
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(x.count, y.count) {
            dot += Double(x[i]) * Double(y[i]); na += Double(x[i]) * Double(x[i]); nb += Double(y[i]) * Double(y[i])
        }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Writes a measured figure to standard error, unbuffered, so a later crash in the process keeps it.
    private func report(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    /// A channels-first volume `[N, C, X, Y, Z]` channels last.
    private func channelsLast(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 4, 1) }

    // MARK: - Structure

    func testImageEncoderTokenLayout() throws {
        let configuration = tinyConfiguration()
        let encoder = try NFKMLXTrellis.imageEncoder(weightsURL: nil, configuration: configuration)
        let tokens = encoder(MLXRandom.uniform(0 ..< 1, [2, 28, 28, 3]))
        XCTAssertEqual(tokens.shape, [2, 1 + 2 + 4, 16])
        let mean = tokens.mean(axis: -1), variance = tokens.variance(axis: -1)
        XCTAssertLessThan(abs(mean).max().item(Float.self), 1e-4)
        XCTAssertLessThan(abs(variance - 1).max().item(Float.self), 1e-2)
    }

    func testPixelShuffleDoublesEachSideInChannelOrder() {
        let upsample = NFKTrellisUpsample3d(inputChannels: 1, outputChannels: 1)
        let shuffled = NFKTrellisUpsample3d.shuffle(MLXArray(0 ..< 16).asType(.float32).reshaped([1, 1, 1, 1, 16]), channels: 2)
        // Output voxel (a, b, d), channel c, reads input channel c·8 + a·4 + b·2 + d.
        let values = shuffled.reshaped([-1]).asArray(Float.self)
        XCTAssertEqual(values, (0 ..< 8).flatMap { voxel in [Float(voxel), Float(8 + voxel)] })
        XCTAssertEqual(upsample(MLXArray.zeros([1, 3, 3, 3, 1])).shape, [1, 6, 6, 6, 1])
    }

    func testStructureFlowAndDecoderShapes() throws {
        let configuration = tinyConfiguration()
        let flow = try NFKMLXTrellis.structureFlow(weightsURL: nil, configuration: configuration)
        let decoder = try NFKMLXTrellis.structureDecoder(weightsURL: nil, configuration: configuration)
        let latent = MLXRandom.normal([1, 4, 4, 4, 2])
        let velocity = flow(latent, t: MLXArray([Float(500)]), condition: MLXRandom.normal([1, 7, 16]))
        XCTAssertEqual(velocity.shape, [1, 4, 4, 4, 2])
        let logits = decoder(latent)
        XCTAssertEqual(logits.shape, [1, 8, 8, 8, 1])
        XCTAssertEqual(configuration.occupancyResolution, 8)
    }

    func testAbsolutePositionsMatchTheSinusoidLayout() {
        let embedding = NFKTrellisPositions.absolute(MLXArray([Int32(1), 2, 3]).reshaped([1, 3]), width: 16)
        // 16 / 3 / 2 = 2 frequencies per axis (1 and 1/100), 12 values, 4 zeros.
        let values = embedding.reshaped([-1]).asArray(Float.self)
        let expected: [Float] = [sin(1), sin(0.01), cos(1), cos(0.01), sin(2), sin(0.02), cos(2), cos(0.02),
                                 sin(3), sin(0.03), cos(3), cos(0.03), 0, 0, 0, 0]
        for (value, target) in zip(values, expected) {
            XCTAssertEqual(value, target, accuracy: 1e-6)
        }
    }

    func testOccupiedVoxelsAreXMajor() {
        var flags = [Float](repeating: -1, count: 27)
        flags[1 * 9 + 2 * 3 + 0] = 1
        flags[0 * 9 + 0 * 3 + 2] = 1
        let voxels = NFKMLXTrellisStructureDecoder.occupiedVoxels(MLXArray(flags).reshaped([3, 3, 3, 1]))
        XCTAssertEqual(voxels.asArray(Int32.self), [0, 0, 2, 1, 2, 0])
    }

    // MARK: - Sparse layers

    /// Voxels at the positions of `mask`, a `side`³ x-major occupancy.
    private func grid(side: Int, occupied: (Int, Int, Int) -> Bool) -> NFKTrellisVoxelGrid {
        var coordinates = [Int32]()
        for x in 0 ..< side {
            for y in 0 ..< side {
                for z in 0 ..< side where occupied(x, y, z) {
                    coordinates += [Int32(x), Int32(y), Int32(z)]
                }
            }
        }
        return NFKTrellisVoxelGrid(coordinates: coordinates, resolution: side)
    }

    /// Features on `grid` scattered into a dense `[1, side, side, side, C]` volume, zeros elsewhere.
    private func dense(_ features: MLXArray, grid: NFKTrellisVoxelGrid, side: Int) -> MLXArray {
        let c = grid.coordinates
        let flat = (0 ..< grid.count).map { Int32((Int(c[3 * $0]) * side + Int(c[3 * $0 + 1])) * side + Int(c[3 * $0 + 2])) }
        return MLXArray.zeros([side * side * side, features.dim(1)]).at[MLXArray(flat)].add(features)
            .reshaped([1, side, side, side, features.dim(1)])
    }

    func testSparseConvolutionMatchesADenseConvolutionOnTheOccupiedVoxels() {
        let voxels = grid(side: 5) { x, y, z in (x * 7 + y * 3 + z * 5) % 3 != 0 }
        let convolution = NFKTrellisSparseConvolution(inputChannels: 4, outputChannels: 3, kernel: 3)
        convolution.update(parameters: ModuleParameters.unflattened(["bias": MLXRandom.normal([3])]))
        let features = MLXRandom.normal([voxels.count, 4])
        let reference = Conv3d(inputChannels: 4, outputChannels: 3, kernelSize: 3, padding: 1)
        reference.update(parameters: ModuleParameters.unflattened(["weight": convolution.weight, "bias": convolution.bias]))
        let denseOut = reference(dense(features, grid: voxels, side: 5)).reshaped([-1, 3])
        let c = voxels.coordinates
        let rows = MLXArray((0 ..< voxels.count).map { Int32((Int(c[3 * $0]) * 5 + Int(c[3 * $0 + 1])) * 5 + Int(c[3 * $0 + 2])) })
        let expected = take(denseOut, rows, axis: 0)
        for limit in [NFKTrellisSparseConvolution.fusedLimit, 0] {
            let out = convolution(features, grid: voxels, fusedLimit: limit)
            XCTAssertLessThan(maxDifference(out, expected), 1e-4, "limit \(limit)")
        }
    }

    func testHalvingCountsTheZeroTorchStartsFrom() {
        let voxels = NFKTrellisVoxelGrid(coordinates: [0, 0, 0, 0, 0, 1, 3, 3, 3], resolution: 4)
        let (coarse, parent) = voxels.halved()
        XCTAssertEqual(coarse.coordinates, [0, 0, 0, 1, 1, 1])
        XCTAssertEqual(parent, [0, 0, 1])
        let halved = NFKTrellisResampling.halve(MLXArray([Float(2), 4, 6]).reshaped([3, 1]), parent: parent, parents: 2)
        // (0 + 2 + 4) / 3 and (0 + 6) / 2, as `scatter_reduce(..., "mean")` with the zero included.
        XCTAssertEqual(halved.reshaped([-1]).asArray(Float.self), [2, 3])
        XCTAssertEqual(NFKTrellisResampling.restore(halved, parent: parent).reshaped([-1]).asArray(Float.self), [2, 2, 3])
    }

    func testSubdivisionSplitsEachVoxelXMajor() {
        let children = NFKTrellisVoxelGrid(coordinates: [1, 0, 2], resolution: 4).subdivided()
        XCTAssertEqual(children.resolution, 8)
        XCTAssertEqual(children.coordinates, [2, 0, 4, 2, 0, 5, 2, 1, 4, 2, 1, 5, 3, 0, 4, 3, 0, 5, 3, 1, 4, 3, 1, 5])
    }

    func testWindowedAttentionAttendsOnlyWithinEachShiftedWindow() {
        let voxels = grid(side: 6) { x, y, z in (x + 2 * y + z) % 2 == 0 }
        let attention = NFKTrellisWindowedAttention(width: 8, heads: 2, size: 4, shift: 2)
        let x = MLXRandom.normal([voxels.count, 8])
        let out = attention(x, grid: voxels)
        // Each voxel attends exactly to the voxels whose shifted coordinates share its window.
        let qkv = attention.toQKV(x).reshaped([voxels.count, 3, 2, 4])
        let c = voxels.coordinates
        let window = (0 ..< voxels.count).map { v in (0 ..< 3).map { (Int(c[3 * v + $0]) + 2) / 4 } }
        var rows = [MLXArray]()
        for voxel in 0 ..< voxels.count {
            let members = MLXArray((0 ..< voxels.count).filter { window[$0] == window[voxel] }.map(Int32.init))
            let q = qkv[voxel ..< voxel + 1, 0], k = take(qkv[0..., 1], members, axis: 0), v = take(qkv[0..., 2], members, axis: 0)
            let scores = softmax(einsum("qhd,khd->hqk", q, k) / Float(2), axis: -1)
            rows.append(einsum("hqk,khd->qhd", scores, v).reshaped([1, 8]))
        }
        XCTAssertLessThan(maxDifference(out, attention.toOut(concatenated(rows, axis: 0))), 1e-5)
    }

    func testSparseNetworksRunOnATinyConfiguration() throws {
        var configuration = tinyConfiguration()
        configuration.latentChannels = 2
        configuration.latentFlowIOWidth = 32
        configuration.decoderWidth = 256
        configuration.decoderBlocks = 2
        configuration.decoderHeads = 2
        configuration.decoderWindow = 4
        configuration.meshFeatureChannels = 5
        configuration.gaussiansPerVoxel = 2
        let voxels = grid(side: 8) { x, y, z in x + y + z < 6 }
        let latent = MLXRandom.normal([voxels.count, 2])
        let flow = try NFKMLXTrellis.latentFlow(weightsURL: nil, configuration: configuration)
        XCTAssertEqual(flow.velocity(latent, grid: voxels, t: MLXArray([Float(500)]), condition: MLXRandom.normal([1, 7, 16])).shape,
                       [voxels.count, 2])
        let mesh = try NFKMLXTrellis.meshDecoder(weightsURL: nil, configuration: configuration).features(latent, grid: voxels)
        XCTAssertEqual(mesh.features.shape, [voxels.count * 64, 5])
        XCTAssertEqual(mesh.grid.resolution, 32)
        let decoder = try NFKMLXTrellis.gaussianDecoder(weightsURL: nil, configuration: configuration)
        let gaussians = decoder.gaussians(decoder.features(latent, grid: voxels), grid: voxels)
        XCTAssertEqual(gaussians.positions.shape, [voxels.count * 2, 3])
        XCTAssertLessThan(abs(sqrt(sum(gaussians.rotations * gaussians.rotations, axis: -1)) - 1).max().item(Float.self), 1e-5)
    }

    // MARK: - Parity

    /// A sparse record entry's features and its voxel grid (`.coords` without the batch column).
    private func sparse(_ reference: [String: MLXArray], _ name: String, resolution: Int) -> (MLXArray, NFKTrellisVoxelGrid) {
        (reference[name]!, NFKTrellisVoxelGrid(reference[name + ".coords"]![0..., 1...], resolution: resolution))
    }

    func testLatentFlowMatchesReference() throws {
        let weights = try checkpoint("slat_flow_img_dit_L_64l8p2_fp16")
        let reference = try record()
        let flow = try NFKMLXTrellis.latentFlow(weightsURL: weights)
        for call in ["call0", "call1", "call2", "last"] {
            let prefix = "slat_flow.\(call)."
            let (x, voxels) = sparse(reference, prefix + "x", resolution: 64)
            let velocity = flow.velocity(x, grid: voxels, t: reference[prefix + "t"]!, condition: reference[prefix + "cond"]!)
            let expected = reference[prefix + "out"]!
            let similarity = cosine(velocity, expected)
            report("TRELLIS latent flow \(call) cosine \(similarity), max difference \(maxDifference(velocity, expected))")
            XCTAssertGreaterThan(similarity, 0.9999999999, call)
        }
    }

    func testGaussianDecoderMatchesReference() throws {
        let weights = try checkpoint("slat_dec_gs_swin8_B_64l8gs32_fp16")
        let reference = try record()
        let decoder = try NFKMLXTrellis.gaussianDecoder(weightsURL: weights)
        let (latent, voxels) = sparse(reference, "slat", resolution: 64)
        let features = decoder.features(latent, grid: voxels)
        let gaussians = decoder.gaussians(features, grid: voxels)
        let pairs: [(String, MLXArray)] = [
            ("gaussian_decoder_out", features), ("gaussian_xyz", gaussians.storedPositions),
            ("gaussian_features_dc", gaussians.colors), ("gaussian_scaling", gaussians.storedScales),
            ("gaussian_rotation", gaussians.storedRotations), ("gaussian_opacity", gaussians.storedOpacities),
            ("gaussian.get_xyz", gaussians.positions), ("gaussian.get_scaling", gaussians.scales),
            ("gaussian.get_rotation", gaussians.rotations), ("gaussian.get_opacity", gaussians.opacities),
        ]
        for (name, value) in pairs {
            let expected = reference[name]!
            XCTAssertEqual(value.shape, expected.shape, name)
            let similarity = cosine(value, expected)
            report("TRELLIS \(name) cosine \(similarity), max difference \(maxDifference(value, expected))")
            XCTAssertGreaterThan(similarity, 0.9999999999, name)
        }
    }

    func testMeshDecoderFeaturesMatchReference() throws {
        let weights = try checkpoint("slat_dec_mesh_swin8_B_64l8m256c_fp16")
        let reference = try record()
        let decoder = try NFKMLXTrellis.meshDecoder(weightsURL: weights)
        let (latent, voxels) = sparse(reference, "slat", resolution: 64)
        let (features, fine) = decoder.features(latent, grid: voxels)
        let expectedCoordinates = reference["mesh_decoder_out.coords"]![0..., 1...].asType(.int32).reshaped([-1]).asArray(Int32.self)
        XCTAssertEqual(fine.coordinates, expectedCoordinates)
        let expected = reference["mesh_decoder_out"]!
        let similarity = cosine(features, expected)
        report("TRELLIS mesh decoder features cosine \(similarity), max difference \(maxDifference(features, expected))")
        XCTAssertGreaterThan(similarity, 0.9999999999)
    }

    func testImageEncoderMatchesReference() throws {
        let weights = try envPath("IK_VAL_DINOV2_VITL14_REG")
        let reference = try record()
        let encoder = try NFKMLXTrellis.imageEncoder(weightsURL: weights)
        let image = expandedDimensions(reference["prepared_image"]!.asType(.float32) / 255, axis: 0)
        let tokens = encoder(image)
        let expected = reference["cond"]!
        XCTAssertEqual(tokens.shape, expected.shape)
        let similarity = cosine(tokens, expected)
        report("TRELLIS condition cosine \(similarity), max difference \(maxDifference(tokens, expected))")
        XCTAssertGreaterThan(similarity, 0.9999999999)
    }

    func testStructureFlowMatchesReference() throws {
        let weights = try checkpoint("ss_flow_img_dit_L_16l8_fp16")
        let reference = try record()
        let flow = try NFKMLXTrellis.structureFlow(weightsURL: weights)
        for call in ["call0", "call1", "call2", "last"] {
            let prefix = "ss_flow.\(call)."
            let velocity = flow(channelsLast(reference[prefix + "x"]!), t: reference[prefix + "t"]!, condition: reference[prefix + "cond"]!)
            let expected = channelsLast(reference[prefix + "out"]!)
            let similarity = cosine(velocity, expected)
            report("TRELLIS structure flow \(call) cosine \(similarity), max difference \(maxDifference(velocity, expected))")
            XCTAssertGreaterThan(similarity, 0.9999999999, call)
        }
    }

    func testStructureDecoderMatchesReference() throws {
        let weights = try checkpoint("ss_dec_conv3d_16l8_fp16")
        let reference = try record()
        let decoder = try NFKMLXTrellis.structureDecoder(weightsURL: weights)
        let logits = decoder(channelsLast(reference["ss_latent"]!))
        let expected = channelsLast(reference["ss_logits"]!)
        let similarity = cosine(logits, expected)
        let voxels = NFKMLXTrellisStructureDecoder.occupiedVoxels(logits[0])
        let expectedVoxels = reference["coords"]![0..., 1...].asType(.int32)
        let same = voxels.shape == expectedVoxels.shape && voxels.asArray(Int32.self) == expectedVoxels.asArray(Int32.self)
        report("TRELLIS structure decoder cosine \(similarity), max difference \(maxDifference(logits, expected)), " +
              "occupied \(voxels.dim(0)) against \(expectedVoxels.dim(0)), identical \(same)")
        XCTAssertGreaterThan(similarity, 0.9999999999)
        XCTAssertTrue(same)
    }
    // MARK: - Mesh extraction

    func testTheDerivedDualTableHasTheReleasesCaseCounts() {
        let cases = NFKTrellisDualCubes.cases
        var vertexCounts = [Int: Int]()
        for entry in cases {
            vertexCounts[entry.groups.count, default: 0] += 1
        }
        XCTAssertEqual(vertexCounts, [0: 2, 1: 162, 2: 82, 3: 8, 4: 2])
        XCTAssertEqual(cases.filter { $0.ambiguousNormal != nil }.count, 36)
        XCTAssertEqual(cases[6].groups, [[0, 1, 9], [4, 7, 8]])
        XCTAssertEqual(cases[129].groups, [[0, 3, 8], [5, 6, 10]])
        XCTAssertEqual(cases[20].groups, [[4, 7, 8], [2, 3, 11]])
        XCTAssertEqual(cases[28].groups, [[5, 7, 8, 9], [2, 3, 11]])
        XCTAssertTrue(cases[61].ambiguousNormal! == (1, 0, 0))
        XCTAssertTrue(cases[62].ambiguousNormal! == (-1, 0, 0))
        XCTAssertNil(cases[3].ambiguousNormal)
    }

    func testFlexiCubesClosesASphere() {
        let side = 12
        var coordinates = [Int32]()
        var features = [Float]()
        for x in 0 ..< side {
            for y in 0 ..< side {
                for z in 0 ..< side {
                    coordinates += [Int32(x), Int32(y), Int32(z)]
                    var row = [Float](repeating: 0, count: NFKTrellisFlexiCubes.featureChannels)
                    for c in 0 ..< 8 {
                        let (cx, cy, cz) = NFKTrellisDualCubes.corner(c)
                        let p = [Float(x + cx), Float(y + cy), Float(z + cz)].map { $0 - Float(side) / 2 }
                        row[c] = ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).squareRoot() - 4) / Float(side)
                    }
                    features += row
                }
            }
        }
        let surface = NFKTrellisFlexiCubes.surface(coordinates: coordinates, features: features, resolution: side)
        XCTAssertGreaterThan(surface.vertexCount, 50)
        XCTAssertEqual(surface.triangles.count % 3, 0)
        // The release's FlexiCubes on this grid (`SparseFeatures2Mesh(res=12)` over the same float32 SDFs, run
        // 2026-11-11) gives 416 vertices and 828 faces, 2,304 of whose 2,394 directed edges meet their reverse.
        // The port gives the same.
        var edges = [String: Int]()
        var volume: Float = 0
        for face in 0 ..< surface.triangles.count / 3 {
            let ids = (0 ..< 3).map { Int(surface.triangles[face * 3 + $0]) }
            for k in 0 ..< 3 {
                edges["\(ids[k])>\(ids[(k + 1) % 3])", default: 0] += 1
            }
            let p = ids.map { i in (0 ..< 3).map { surface.positions[i * 3 + $0] } }
            volume += (p[0][1] * p[1][2] - p[0][2] * p[1][1]) * p[2][0] + (p[0][2] * p[1][0] - p[0][0] * p[1][2]) * p[2][1]
                + (p[0][0] * p[1][1] - p[0][1] * p[1][0]) * p[2][2]
        }
        let paired = edges.keys.filter { key in
            let parts = key.split(separator: ">")
            return edges["\(parts[1])>\(parts[0])"] == edges[key]
        }.count
        report("TRELLIS FlexiCubes sphere: \(surface.vertexCount) vertices, \(surface.triangles.count / 3) faces, "
               + "\(paired) of \(edges.count) directed edges paired")
        XCTAssertEqual(surface.vertexCount, 416)
        XCTAssertEqual(surface.triangles.count / 3, 828)
        XCTAssertEqual(edges.count, 2394)
        XCTAssertEqual(paired, 2304)
        XCTAssertGreaterThan(volume, 0)
    }

    func testFlexiCubesReproducesTheReferenceMesh() throws {
        let reference = try record()
        let coordinates = reference["mesh_decoder_out.coords"]![0..., 1...].asType(.int32).reshaped([-1]).asArray(Int32.self)
        let surface = NFKTrellisFlexiCubes.surface(coordinates: coordinates, features: reference["mesh_decoder_out"]!.asArray(Float.self),
                                                   resolution: 256)
        let vertices = reference["mesh_vertices"]!, faces = reference["mesh_faces"]!.asType(.int32)
        let sameFaces = surface.triangles.count == faces.size && zip(surface.triangles, faces.reshaped([-1]).asArray(Int32.self))
            .allSatisfy { Int64($0) == Int64($1) }
        let positions = MLXArray(surface.positions).reshaped([-1, 3]), colors = MLXArray(surface.colors).reshaped([-1, 6])
        let sameCount = positions.dim(0) == vertices.dim(0)
        report("TRELLIS FlexiCubes vertices \(positions.dim(0)) against \(vertices.dim(0)), faces identical \(sameFaces)"
               + (sameCount ? ", position max difference \(maxDifference(positions, vertices)), color max difference \(maxDifference(colors, reference["mesh_vertex_attrs"]!))" : ""))
        XCTAssertTrue(sameCount)
        XCTAssertTrue(sameFaces)
        if sameCount {
            XCTAssertLessThan(maxDifference(positions, vertices), 1e-5)
            XCTAssertLessThan(maxDifference(colors, reference["mesh_vertex_attrs"]!), 1e-5)
        }
    }

    // MARK: - Pipeline

    func testTheTimetableIsTheReleasesRescaledSchedule() {
        let pairs = NFKMLXTrellisSampling.image.timetable
        XCTAssertEqual(pairs.count, 25)
        XCTAssertEqual(pairs[0].0, 1)
        XCTAssertEqual(pairs[1].0, 3 * 0.96 / (1 + 2 * 0.96), accuracy: 1e-15)
        XCTAssertEqual(pairs[24].0, 3 * 0.04 / (1 + 2 * 0.04), accuracy: 1e-15)
        XCTAssertEqual(pairs[24].1, 0)
        XCTAssertTrue(NFKMLXTrellisSampling.image.guidanceInterval.contains(0.5))
        XCTAssertFalse(NFKMLXTrellisSampling.text.guidanceInterval.contains(0.96))
    }

    func testPreparationReproducesThePreparedImage() throws {
        let reference = try record()
        let rgba = reference["input_rgba"]!
        let prepared = NFKTrellisImagePreparation.prepare(rgba: rgba.asArray(UInt8.self), width: rgba.dim(1), height: rgba.dim(0))
        let expected = reference["prepared_image"]!.asArray(UInt8.self)
        let mismatches = zip(prepared, expected).filter { $0 != $1 }.count
        report("TRELLIS prepared image mismatched bytes \(mismatches) of \(expected.count)")
        XCTAssertEqual(prepared.count, expected.count)
        XCTAssertEqual(mismatches, 0)
    }

    private func imagePipeline(residency: NFKMLXResidency = .staged) throws -> NFKMLXTrellisPipeline {
        try NFKMLXTrellisPipeline.image(directoryURL: try envPath("IK_VAL_TRELLIS_IMAGE_LARGE"),
                                        encoderWeightsURL: try envPath("IK_VAL_DINOV2_VITL14_REG"), residency: residency)
    }

    func testBothSamplersMatchReferenceFromTheRecordedNoise() throws {
        let reference = try record()
        let pipeline = try imagePipeline()
        let condition = reference["cond"]!
        let structure = try pipeline.sampleStructure(noise: channelsLast(reference["ss_noise"]!), condition: condition,
                                                     unconditional: MLXArray.zeros(like: condition))
        let structureCosine = cosine(structure, channelsLast(reference["ss_latent"]!))
        let voxels = NFKMLXTrellisStructureDecoder.occupiedVoxels(try pipeline.occupancy(structure)[0])
        let expectedVoxels = reference["coords"]![0..., 1...].asType(.int32)
        let latent = try pipeline.sampleLatent(noise: reference["slat_noise"]!, voxels: expectedVoxels, condition: condition,
                                               unconditional: MLXArray.zeros(like: condition))
        let latentCosine = cosine(latent, reference["slat_normalized"]!)
        let denormalizedCosine = cosine(pipeline.denormalized(latent), reference["slat"]!)
        report("TRELLIS structure sampling cosine \(structureCosine), occupied \(voxels.dim(0)) against \(expectedVoxels.dim(0)); "
               + "latent sampling cosine \(latentCosine), denormalized \(denormalizedCosine)")
        XCTAssertGreaterThan(structureCosine, 0.999999999)
        XCTAssertGreaterThan(latentCosine, 0.999999999)
        XCTAssertGreaterThan(denormalizedCosine, 0.999999999)
        XCTAssertEqual(voxels.dim(0), expectedVoxels.dim(0))
    }

    func testTheBackendGeneratesAMeshAndGaussiansFromItsFactory() throws {
        let reference = try record()
        let backend = try NFKMLXTrellisFactory.backend(directoryURL: try envPath("IK_VAL_TRELLIS_IMAGE_LARGE"),
                                                       encoderWeightsURL: try envPath("IK_VAL_DINOV2_VITL14_REG"), residency: .staged)
        XCTAssertEqual(backend.supportedOutputKeys, [NFKOutputMesh, NFKOutputGaussianSplats])
        let rgba = reference["input_rgba"]!
        let (height, width) = (rgba.dim(0), rgba.dim(1))
        let bytes = rgba.asArray(UInt8.self)
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let request = NFKInferenceRequest(inputs: [NFKInputImage: image],
                                          parameters: [NFKMLXTrellisParameterKey.structureSteps: 4, NFKMLXTrellisParameterKey.latentSteps: 4,
                                                       NFKParameterSeed: 7])
        let result = try backend.runInference(for: request)
        let mesh = try XCTUnwrap(result.mesh)
        let splats = try XCTUnwrap(result.gaussianSplats)
        report("TRELLIS backend at 4 steps: \(mesh.vertexCount) vertices, \(mesh.faceCount) faces, \(splats.count) Gaussians")
        XCTAssertGreaterThan(mesh.faceCount, 1000)
        XCTAssertEqual(splats.count % 32, 0)
        XCTAssertNotNil(mesh.colors)
        // The object faces +z with y up: its height runs along y.
        let low = mesh.boundingBoxMinimum, high = mesh.boundingBoxMaximum
        XCTAssertGreaterThan(high.y - low.y, high.z - low.z)
    }

    func testTheExportTurnsZUpIntoGLTFsFrame() throws {
        let gaussians = NFKTrellisGaussians(storedPositions: MLXArray([Float(0.5), 0.6, 0.7]).reshaped([1, 3]),
                                            colors: MLXArray.zeros([1, 1, 3]), storedScales: MLXArray.zeros([1, 3]),
                                            storedRotations: MLXArray([Float(0), 0, 0, 0]).reshaped([1, 4]),
                                            storedOpacities: MLXArray.zeros([1, 1]), minimumScale: 0, scaleBias: 0, opacityBias: 0)
        let splats = try NFKTrellisExport.splats(gaussians)
        let p = splats.position(at: 0)
        XCTAssertEqual(p.x, 0, accuracy: 1e-6)
        XCTAssertEqual(p.y, 0.2, accuracy: 1e-6)
        XCTAssertEqual(p.z, -0.1, accuracy: 1e-6)
        // The identity orientation becomes the quarter turn about x.
        let q = splats.rotation(at: 0)
        XCTAssertEqual(q.w, Float(0.5).squareRoot(), accuracy: 1e-6)
        XCTAssertEqual(q.x, -Float(0.5).squareRoot(), accuracy: 1e-6)
    }

    // MARK: - Text releases

    private func textCase(_ variant: NFKMLXTrellisVariant) throws -> (pipeline: NFKMLXTrellisPipeline, record: [String: MLXArray]) {
        let suffix = ["", "BASE", "LARGE", "XLARGE"][variant.rawValue]
        let reference = try MLX.loadArrays(url: try envPath("IK_PARITY_TRELLIS_TEXT_\(suffix)"))
        let pipeline = try NFKMLXTrellisPipeline.text(directoryURL: try envPath("IK_VAL_TRELLIS_TEXT_\(suffix)"),
                                                      imageReleaseDirectoryURL: try envPath("IK_VAL_TRELLIS_IMAGE_LARGE"),
                                                      textEncoderDirectoryURL: try envPath("IK_VAL_CLIP_L14_HF"), residency: .staged)
        return (pipeline, reference)
    }

    private func checkText(_ variant: NFKMLXTrellisVariant) throws {
        let (pipeline, reference) = try textCase(variant)
        let name = NFKMLXTrellisFactory.modelName(for: variant)
        let tokens = pipeline.tokenizer!.tokens(for: "A wooden chair with a red cushion", contextLength: 77)
        XCTAssertEqual(tokens.map(Int32.init), reference["prompt_tokens"]!.reshaped([-1]).asArray(Int32.self))
        let (condition, unconditional) = try pipeline.condition(prompt: "A wooden chair with a red cushion")
        let conditionCosine = cosine(condition, reference["cond"]!), unconditionalCosine = cosine(unconditional, reference["neg_cond"]!)
        var flowCosines = [String]()
        try pipeline.structureFlow.use { flow in
            for call in ["call0", "call1", "last"] {
                let prefix = "ss_flow.\(call)."
                let out = flow(channelsLast(reference[prefix + "x"]!), t: reference[prefix + "t"]!, condition: reference[prefix + "cond"]!)
                let value = cosine(out, channelsLast(reference[prefix + "out"]!))
                flowCosines.append("structure \(call) \(value)")
                XCTAssertGreaterThan(value, 0.9999999999, "\(name) structure \(call)")
            }
        }
        try pipeline.latentFlow.use { flow in
            for call in ["call0", "call1", "last"] {
                let prefix = "slat_flow.\(call)."
                let grid = NFKTrellisVoxelGrid(reference[prefix + "x.coords"]![0..., 1...], resolution: 64)
                let out = flow.velocity(reference[prefix + "x"]!, grid: grid, t: reference[prefix + "t"]!, condition: reference[prefix + "cond"]!)
                let value = cosine(out, reference[prefix + "out"]!)
                flowCosines.append("latent \(call) \(value)")
                XCTAssertGreaterThan(value, 0.9999999999, "\(name) latent \(call)")
            }
        }
        let structure = try pipeline.sampleStructure(noise: channelsLast(reference["ss_noise"]!), condition: reference["cond"]!,
                                                     unconditional: reference["neg_cond"]!)
        let structureCosine = cosine(structure, channelsLast(reference["ss_latent"]!))
        let latent = try pipeline.sampleLatent(noise: reference["slat_noise"]!, voxels: reference["coords"]![0..., 1...].asType(.int32),
                                               condition: reference["cond"]!, unconditional: reference["neg_cond"]!)
        let latentCosine = cosine(latent, reference["slat_normalized"]!)
        report("TRELLIS \(name) condition cosine \(conditionCosine), empty prompt \(unconditionalCosine), "
               + flowCosines.joined(separator: ", ") + ", structure sampling \(structureCosine), latent sampling \(latentCosine)")
        XCTAssertGreaterThan(conditionCosine, 0.9999999)
        XCTAssertGreaterThan(unconditionalCosine, 0.9999999)
        XCTAssertGreaterThan(structureCosine, 0.999999999)
        XCTAssertGreaterThan(latentCosine, 0.999999999)
    }

    func testTextBaseMatchesReference() throws { try checkText(.textBase) }
    func testTextLargeMatchesReference() throws { try checkText(.textLarge) }
    func testTextXLargeMatchesReference() throws { try checkText(.textXLarge) }

    func testTheTextPresetsMatchTheReleasesConfigurations() throws {
        for (variant, preset) in [(NFKMLXTrellisVariant.textBase, NFKMLXTrellisConfiguration.textBase), (.textLarge, .textLarge),
                                  (.textXLarge, .textXLarge)] {
            let suffix = ["", "BASE", "LARGE", "XLARGE"][variant.rawValue]
            let release = try NFKMLXTrellisRelease(directoryURL: try envPath("IK_VAL_TRELLIS_TEXT_\(suffix)"),
                                                   imageReleaseDirectoryURL: try envPath("IK_VAL_TRELLIS_IMAGE_LARGE"))
            let read = release.configuration
            XCTAssertEqual([read.flowWidth, read.flowBlocks, read.flowHeads, read.conditionWidth, read.latentFlowIOWidth, read.latentFlowIOBlocks],
                           [preset.flowWidth, preset.flowBlocks, preset.flowHeads, preset.conditionWidth, preset.latentFlowIOWidth,
                            preset.latentFlowIOBlocks], "\(variant)")
            XCTAssertEqual(read.crossAttentionHeadNorm, preset.crossAttentionHeadNorm)
            XCTAssertEqual(read.latentCrossAttentionHeadNorm, preset.latentCrossAttentionHeadNorm)
            XCTAssertEqual(release.structureSampling, .text)
        }
    }

    // MARK: - Customization

    private func objectives() throws -> [String: MLXArray] {
        try MLX.loadArrays(url: try envPath("IK_PARITY_TRELLIS_OBJECTIVES"))
    }

    func testStructureEncoderMatchesReference() throws {
        let reference = try objectives()
        let encoder = try NFKMLXTrellis.structureEncoder(weightsURL: try checkpoint("ss_enc_conv3d_16l8_fp16"))
        let (mean, logVariance) = encoder(channelsLast(reference["ss_occupancy"]!))
        let meanCosine = cosine(mean, channelsLast(reference["ss_encoded"]!))
        let varianceCosine = cosine(logVariance, channelsLast(reference["ss_encoded_logvar"]!))
        report("TRELLIS structure encoder mean cosine \(meanCosine), log-variance \(varianceCosine)")
        XCTAssertGreaterThan(meanCosine, 0.9999999999)
        XCTAssertGreaterThan(varianceCosine, 0.9999999999)
    }

    func testTheObjectivesMatchTheReleasesTrainers() throws {
        let reference = try objectives()
        let structure = try NFKMLXTrellis.structureFlow(weightsURL: try checkpoint("ss_flow_img_dit_L_16l8_fp16"))
        let latent = try NFKMLXTrellis.latentFlow(weightsURL: try checkpoint("slat_flow_img_dit_L_64l8p2_fp16"))
        var lines = [String]()
        for index in 0 ..< 2 {
            let p = "dense\(index)."
            let x0 = channelsLast(reference["ss_encoded"]!), noise = channelsLast(reference[p + "noise"]!), t = reference[p + "t"]!
            let xt = NFKMLXTrellisObjective.diffuse(x0, noise: noise, t: t)
            let loss = NFKMLXTrellisObjective.structureLoss(structure, x0: x0, condition: reference[p + "cond"]!, noise: noise, t: t)
            let xtCosine = cosine(xt, channelsLast(reference[p + "x_t"]!))
            let target = cosine(NFKMLXTrellisObjective.target(x0, noise: noise), channelsLast(reference[p + "target"]!))
            let expected = reference[p + "mse"]!.item(Float.self), measured = loss.item(Float.self)
            lines.append("dense\(index) x_t \(xtCosine) target \(target) mse \(measured) against \(expected)")
            XCTAssertGreaterThan(xtCosine, 0.9999999999)
            XCTAssertEqual(measured, expected, accuracy: expected * 1e-5)
        }
        for index in 0 ..< 2 {
            let p = "sparse\(index)."
            let x0 = reference["slat_x0"]!, voxels = reference["slat_x0.coords"]![0..., 1...].asType(.int32)
            let loss = NFKMLXTrellisObjective.latentLoss(latent, x0: x0, voxels: voxels, condition: reference[p + "cond"]!,
                                                         noise: reference[p + "noise"]!, t: reference[p + "t"]!)
            let expected = reference[p + "mse"]!.item(Float.self), measured = loss.item(Float.self)
            lines.append("sparse\(index) mse \(measured) against \(expected)")
            XCTAssertEqual(measured, expected, accuracy: expected * 1e-5)
        }
        report("TRELLIS objectives " + lines.joined(separator: "; "))
    }

    func testAStructureLoRAFineTuneLowersTheLossAndMovesOnlyTheAdapters() throws {
        var configuration = tinyConfiguration()
        configuration.conditionWidth = 16
        MLXRandom.seed(11)
        let flow = try NFKMLXTrellis.structureFlow(weightsURL: nil, configuration: configuration)
        let frozenBefore = flow.inputLayer.weight
        eval(frozenBefore)
        let example = NFKMLXTrellisStructureExample(condition: MLXRandom.normal([1, 7, 16], key: MLXRandom.key(1)),
                                                    latent: MLXRandom.normal([1, 4, 4, 4, 2], key: MLXRandom.key(2)))
        // Each training step draws its own time and noise, so single-step losses swing with the draw. The
        // objective is scored instead on eight fixed draws spanning the times, before and after training.
        let draws = (0 ..< 8).map { index in
            (noise: MLXRandom.normal([1, 4, 4, 4, 2], key: MLXRandom.key(UInt64(100 + index))),
             t: MLXArray([Float(index) / 8 + 1 / 16]))
        }
        func heldOut() -> Float {
            draws.map { draw in
                NFKMLXTrellisObjective.structureLoss(flow, x0: example.latent, condition: example.condition, noise: draw.noise,
                                                     t: draw.t).item(Float.self)
            }.reduce(0, +) / Float(draws.count)
        }
        let before = heldOut()
        let losses = try NFKMLXTrellisTraining.fineTuneStructureFlow(flow, examples: [example],
                                                                     trainable: .attentionLoRA(rank: 4, alpha: 4),
                                                                     optimizer: NFKMLXAdam(learningRate: 1e-2), steps: 30, seed: 3)
        let after = heldOut()
        report("TRELLIS LoRA fine-tune: held-out objective \(before) before, \(after) after 30 steps")
        XCTAssertEqual(losses.count, 30)
        XCTAssertLessThan(after, before)
        XCTAssertEqual(flow.inputLayer.weight.asArray(Float.self), frozenBefore.asArray(Float.self))
        XCTAssertGreaterThan(NFKMLXLoRA.trainableParameterCount(of: flow), 0)
        XCTAssertTrue(flow.blocks[0].selfAttention.toQKV is NFKMLXLoRAAdapter)
    }

    func testAFineTunedStructureFlowRoundTripsThroughItsLoader() throws {
        var configuration = tinyConfiguration()
        configuration.conditionWidth = 16
        let flow = try NFKMLXTrellis.structureFlow(weightsURL: nil, configuration: configuration)
        let example = NFKMLXTrellisStructureExample(condition: MLXRandom.normal([1, 7, 16]), latent: MLXRandom.normal([1, 4, 4, 4, 2]))
        try NFKMLXTrellisTraining.fineTuneStructureFlow(flow, examples: [example], trainable: .all, steps: 2)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("trellis-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(flow, to: url)
        let reloaded = try NFKMLXTrellis.structureFlow(weightsURL: url, configuration: configuration)
        let x = MLXRandom.normal([1, 4, 4, 4, 2]), t = MLXArray([Float(300)])
        XCTAssertEqual(maxDifference(flow(x, t: t, condition: example.condition), reloaded(x, t: t, condition: example.condition)), 0)
    }

    func testTheOccupancyOfAMeshMarksItsSurfaceInTheDecodersFrame() throws {
        // A thin box twice as tall (y, glTF up) as it is wide.
        var positions = [Float]()
        for (x, y, z) in [(0, 0, 0), (1, 0, 0), (0, 2, 0), (1, 2, 0)] as [(Float, Float, Float)] {
            positions += [x, y, z]
        }
        let mesh = try NFKMesh(positions: positions.withUnsafeBufferPointer { Data(buffer: $0) },
                               normals: nil, colors: nil, textureCoordinates: nil,
                               indices: [UInt32(0), 1, 2, 1, 3, 2].withUnsafeBufferPointer { Data(buffer: $0) })
        let flags = NFKMLXTrellisTraining.occupancy(of: mesh, resolution: 16).reshaped([-1]).asArray(Float.self)
        var low = [16, 16, 16], high = [-1, -1, -1], count = 0
        for (index, flag) in flags.enumerated() where flag > 0 {
            let voxel = [index / 256, index / 16 % 16, index % 16]
            for axis in 0 ..< 3 {
                low[axis] = min(low[axis], voxel[axis])
                high[axis] = max(high[axis], voxel[axis])
            }
            count += 1
        }
        XCTAssertGreaterThan(count, 100)
        // glTF's y is the decoders' z. Cells are half-open and the grid's far face clamps into its last cell:
        // the height spans voxels 0...15, the width (−0.25...0.25 → 4.0...12.0) voxels 4...12, and the plane
        // sits on the face at 8.0, which is voxel 8 alone.
        XCTAssertEqual([low[2], high[2]], [0, 15])
        XCTAssertEqual([low[0], high[0]], [4, 12])
        XCTAssertEqual([low[1], high[1]], [8, 8])
    }
}
