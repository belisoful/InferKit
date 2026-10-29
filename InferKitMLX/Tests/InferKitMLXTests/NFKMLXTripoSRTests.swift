//
//  NFKMLXTripoSRTests.swift
//  InferKitMLXTests
//
//  The structural tests run a tiny configuration. The parity tests read the released checkpoint
//  (IK_VAL_TRIPOSR, the release directory or its model.ckpt) and the oracle record (IK_PARITY_TRIPOSR,
//  from `run_reference.py triposr`), and skip without them.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXNN
import simd
@testable import InferKitMLX

final class NFKMLXTripoSRTests: XCTestCase {

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

    /// A 32-pixel image of 2×2 patches, one ViT layer, one backbone block, 4×4 planes, a two-layer decoder.
    private func tinyConfiguration() -> NFKMLXTripoSRConfiguration {
        var configuration = NFKMLXTripoSRConfiguration()
        configuration.imageSize = 32
        configuration.pretrainedPositionGrid = 2
        configuration.vitWidth = 16
        configuration.vitDepth = 1
        configuration.vitHeads = 2
        configuration.vitHidden = 32
        configuration.planeSize = 4
        configuration.tokenWidth = 32
        configuration.backboneDepth = 1
        configuration.backboneHeads = 2
        configuration.headDimensions = 16
        configuration.normGroups = 4
        configuration.planeChannels = 8
        configuration.decoderNeurons = 8
        configuration.decoderHiddenLayers = 2
        return configuration
    }

    private func envPath(_ key: String) throws -> URL {
        guard let path = NFKMLXValidationConfig.environment[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key)")
        }
        return URL(fileURLWithPath: path)
    }

    /// The release's `model.ckpt`, from the directory or the file IK_VAL_TRIPOSR names.
    private func releaseWeights() throws -> URL {
        let url = try envPath("IK_VAL_TRIPOSR")
        var isDirectory: ObjCBool = false
        FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
        return isDirectory.boolValue ? url.appendingPathComponent("model.ckpt") : url
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        var dot = 0.0, na = 0.0, nb = 0.0
        var i = 0
        while i < min(x.count, y.count) {
            dot += x[i] * y[i]; na += x[i] * x[i]; nb += y[i] * y[i]
            i += 1
        }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
    }

    // MARK: - Structure

    func testTheForwardProducesSceneCodesAndQueriesDensityAndColor() throws {
        let configuration = tinyConfiguration()
        let net = NFKMLXTripoSR.makeNet(configuration)
        let image = MLXRandom.uniform(0 ..< 1, [1, 32, 32, 3], key: MLXRandom.key(3))
        let tokens = net.imageTokens(image)
        XCTAssertEqual(tokens.shape, [1, 5, 16], "four patches and the class token")
        let codes = net(image)
        XCTAssertEqual(codes.shape, [1, 3, 8, 8, 8], "three planes doubled to 8×8 at the plane channel count")

        let points = MLXArray([Float(0), 0, 0, 0.5, -0.2, 0.1]).reshaped([2, 3])
        let (density, color) = net.query(points, sceneCode: codes[0])
        eval(density, color)
        XCTAssertEqual(density.shape, [2])
        XCTAssertEqual(color.shape, [2, 3])
        XCTAssertTrue(density.asArray(Float.self).allSatisfy { $0 > 0 }, "the density is an exponential")
        XCTAssertTrue(color.asArray(Float.self).allSatisfy { $0 > 0 && $0 < 1 }, "the color is a sigmoid")
    }

    func testTheReferenceKeyLayoutIsHeld() throws {
        let net = NFKMLXTripoSR.makeNet(tinyConfiguration())
        let keys = Set(net.parameters().flattened().map(\.0))
        for expected in ["image_tokenizer.model.embeddings.cls_token", "image_tokenizer.model.embeddings.position_embeddings",
                         "image_tokenizer.model.embeddings.patch_embeddings.projection.weight",
                         "image_tokenizer.model.encoder.layer.0.attention.attention.query.weight",
                         "image_tokenizer.model.encoder.layer.0.attention.output.dense.bias",
                         "image_tokenizer.model.encoder.layer.0.intermediate.dense.weight",
                         "image_tokenizer.model.encoder.layer.0.layernorm_before.weight", "image_tokenizer.model.layernorm.bias",
                         "tokenizer.embeddings", "backbone.norm.weight", "backbone.proj_in.weight",
                         "backbone.transformer_blocks.0.attn1.to_q.weight", "backbone.transformer_blocks.0.attn1.to_out.0.bias",
                         "backbone.transformer_blocks.0.attn2.to_k.weight", "backbone.transformer_blocks.0.ff.net.0.proj.weight",
                         "backbone.transformer_blocks.0.ff.net.2.weight", "backbone.proj_out.bias",
                         "post_processor.upsample.weight", "decoder.layers.0.weight", "decoder.layers.2.bias", "decoder.layers.4.weight"] {
            XCTAssertTrue(keys.contains(expected), expected)
        }
        XCTAssertFalse(keys.contains { $0.contains("to_q.bias") }, "the attention projections carry no bias")
        XCTAssertEqual(net.parameters().flattened().first { $0.0 == "tokenizer.embeddings" }?.1.shape, [3, 32, 4, 4],
                       "the token bank keeps the checkpoint's [planes, channels, rows, columns]")
    }

    func testASavedCheckpointReloadsThroughTheFactoryAndReproducesTheForward() throws {
        let configuration = tinyConfiguration()
        let net = NFKMLXTripoSR.makeNet(configuration)
        let image = MLXRandom.uniform(0 ..< 1, [1, 32, 32, 3], key: MLXRandom.key(5))
        let before = net(image)
        eval(before)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("triposr-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXTripoSR.network(weightsURL: url, configuration: configuration)
        let after = reloaded(image)
        eval(after)
        XCTAssertEqual(cosine(before, after), 1, accuracy: 1e-7)
    }

    func testABFloat16CastAndAPackedLoadReproduceTheForward() throws {
        let configuration = tinyConfiguration()
        let net = NFKMLXTripoSR.makeNet(configuration)
        let image = MLXRandom.uniform(0 ..< 1, [1, 32, 32, 3], key: MLXRandom.key(6))
        let reference = net(image)
        eval(reference)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("triposr-ladder-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)

        let half = try NFKMLXTripoSR.network(weightsURL: url, configuration: configuration, castingTo: .bfloat16)
        XCTAssertEqual(half.backbone.projIn.weight.dtype, .bfloat16)
        let halfCodes = half(image.asType(.bfloat16))
        eval(halfCodes)
        XCTAssertGreaterThan(cosine(reference, halfCodes), 0.99)

        // The tiny backbone is 32 wide, the smallest group MLX packs; the 16-wide ViT and the 8-wide decoder stay as loaded.
        let packed = try NFKMLXTripoSR.network(weightsURL: url, configuration: configuration, quantizationBits: 8, groupSize: 32)
        XCTAssertTrue(packed.backbone.projIn is QuantizedLinear, "the backbone's projections pack")
        let packedCodes = packed(image)
        eval(packedCodes)
        XCTAssertGreaterThan(cosine(reference, packedCodes), 0.99)
        XCTAssertThrowsError(try NFKMLXTripoSR.network(weightsURL: url, configuration: configuration, quantizationBits: 12))
        XCTAssertThrowsError(try NFKMLXTripoSR.network(weightsURL: url, configuration: configuration, quantizationBits: 4, groupSize: 16),
                             "MLX aborts the process on a group size it does not pack, so the builder refuses it first")
    }

    func testTheBackendReturnsAMeshUnderTheMeshKey() throws {
        let configuration = tinyConfiguration()

        // A random decoder's densities spread narrowly around e^-1, so the threshold is their median,
        // which puts a surface inside the grid.
        let image = try solidImage(width: 20, height: 24)
        let rgba = try NFKMLXImageBridge.tensor(from: image, channels: 4, colorSpace: CGColorSpaceCreateDeviceRGB())
        let conditioning = NFKMLXTripoSR.resizeAntialiased(NFKMLXTripoSR.prepare(rgba: rgba, foregroundRatio: 0.85),
                                                           side: configuration.imageSize)
        let net = NFKMLXTripoSR.makeNet(configuration)
        let densities = net.densityGrid(sceneCode: net(conditioning.reshaped([1] + conditioning.shape))[0], resolution: 12).sorted()
        let threshold = densities[densities.count / 2]
        let backend = NFKMLXTripoSRBackend(net: net, identifier: "tiny")
        let request = NFKInferenceRequest(inputs: [NFKInputImage: image],
                                          parameters: [NFKMLXTripoSRParameterKey.resolution: 12,
                                                       NFKMLXTripoSRParameterKey.threshold: threshold])
        XCTAssertEqual(backend.supportedOutputKeys, [NFKOutputMesh])
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputImage])
        XCTAssertTrue(backend.supportedParameterKeys.contains(NFKMLXTripoSRParameterKey.resolution))
        let result = try backend.runInference(for: request)
        let mesh = try XCTUnwrap(result.mesh)
        XCTAssertGreaterThan(mesh.faceCount, 0)
        XCTAssertEqual(mesh.colors?.count, mesh.positions.count, "every vertex is colored")
        for index in 0 ..< mesh.vertexCount {
            let position = mesh.position(at: UInt(index))
            XCTAssertLessThanOrEqual(abs(position.x), configuration.radius + 1e-5)
            XCTAssertLessThanOrEqual(abs(position.z), configuration.radius + 1e-5)
        }
        let plain = NFKInferenceRequest(inputs: [NFKInputImage: image],
                                        parameters: [NFKMLXTripoSRParameterKey.resolution: 12,
                                                     NFKMLXTripoSRParameterKey.threshold: threshold,
                                                     NFKMLXTripoSRParameterKey.vertexColors: false])
        XCTAssertNil(try backend.runInference(for: plain).mesh?.colors)
    }

    private func solidImage(width: Int, height: Int) throws -> CGImage {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0 ..< (width * height) {
            bytes[pixel * 4] = UInt8(pixel % 251)
            bytes[pixel * 4 + 1] = 90
            bytes[pixel * 4 + 2] = UInt8((pixel * 7) % 255)
        }
        let provider = try XCTUnwrap(CGDataProvider(data: Data(bytes) as CFData))
        return try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                     space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                     provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }

    // MARK: - Preparation and resampling

    func testAnOpaqueImageIsUsedAsItIsAndATransparentOneIsFramedOnGray() throws {
        let opaque = concatenated([MLXRandom.uniform(0 ..< 1, [6, 5, 3], key: MLXRandom.key(1)), MLXArray.ones([6, 5, 1])], axis: -1)
        let kept = NFKMLXTripoSR.prepare(rgba: opaque, foregroundRatio: 0.85)
        XCTAssertEqual(kept.shape, [6, 5, 3])
        XCTAssertEqual(cosine(kept, opaque[0..., 0..., 0 ..< 3]), 1, accuracy: 1e-7)

        // A 4×6 opaque block at rows 2..<6, columns 3..<9 of a 10×12 transparent image: the half-open crop
        // keeps 3 rows × 5 columns, the square is 5, and 5 / 0.5 = 10 is the framed side.
        let rgba = MLXArray.zeros([10, 12, 4])
        rgba[2 ..< 6, 3 ..< 9, 0 ..< 3] = MLXArray(Float(0.2))
        rgba[2 ..< 6, 3 ..< 9, 3] = MLXArray(Float(1))
        let framed = NFKMLXTripoSR.prepare(rgba: rgba, foregroundRatio: 0.5)
        XCTAssertEqual(framed.shape, [10, 10, 3])
        let values = framed.asArray(Float.self)
        XCTAssertEqual(values[0], 0.5, accuracy: 1e-6, "the padding is mid-gray")
        // The 3 × 5 crop is padded to a 5 × 5 square (one row above, one below, no columns) and the square is
        // centered in 10: the object fills rows 3..<6 and columns 2..<7, and every other pixel is gray.
        XCTAssertEqual(framed[4, 5, 0].item(Float.self), 0.2, accuracy: 1e-6)
        XCTAssertEqual(framed[2, 5, 0].item(Float.self), 0.5, accuracy: 1e-6, "the square's own padding row is transparent, so gray")
        XCTAssertEqual(framed[3, 5, 0].item(Float.self), 0.2, accuracy: 1e-6, "the first row of the object")
        XCTAssertEqual(framed[5, 5, 0].item(Float.self), 0.2, accuracy: 1e-6, "the last row of the object")
        XCTAssertEqual(framed[6, 5, 0].item(Float.self), 0.5, accuracy: 1e-6, "the half-open crop dropped the object's last row")
        XCTAssertEqual(framed[4, 2, 0].item(Float.self), 0.2, accuracy: 1e-6, "the first column of the object")
        XCTAssertEqual(framed[4, 1, 0].item(Float.self), 0.5, accuracy: 1e-6, "the outer padding to the left")
        XCTAssertEqual(framed[4, 7, 0].item(Float.self), 0.5, accuracy: 1e-6, "the half-open crop dropped the object's last column")
    }

    func testTheAntialiasedResizeMatrixIsNormalizedAndWidensWhenShrinking() throws {
        let shrink = NFKMLXTripoSR.antialiasBilinearMatrix(from: 8, to: 2)
        XCTAssertEqual(shrink.shape, [2, 8])
        let rows = shrink.asArray(Float.self)
        for row in 0 ..< 2 {
            XCTAssertEqual((0 ..< 8).reduce(Float(0)) { $0 + rows[row * 8 + $1] }, 1, accuracy: 1e-6)
        }
        XCTAssertGreaterThan((0 ..< 8).filter { rows[$0] > 0 }.count, 2, "the triangle spans the source pixels it covers")
        // The triangle of support 4 centered at 2 weights pixels 0...5 by 0.625, 0.875, 0.875, 0.625, 0.375,
        // 0.125, whose normalized mean of the index is 6.625 / 3.5; the second output mirrors it.
        let values = matmul(shrink, MLXArray((0 ..< 8).map { Float($0) }).reshaped([8, 1])).asArray(Float.self)
        XCTAssertEqual(values[0], 6.625 / 3.5, accuracy: 1e-5)
        XCTAssertEqual(values[1], 7 - 6.625 / 3.5, accuracy: 1e-5)

        let grow = NFKMLXTripoSR.antialiasBilinearMatrix(from: 2, to: 4)
        // Plain bilinear with half-pixel centers and clamped ends: 0, 0.25, 0.75, 1 of the way.
        let grown = matmul(grow, MLXArray([Float(0), 1]).reshaped([2, 1])).asArray(Float.self)
        XCTAssertEqual(grown, [0, 0.25, 0.75, 1].map { Float($0) })
    }

    func testThePositionEmbeddingsResizeToTheConditioningGrid() throws {
        let embeddings = MLXRandom.normal([1, 2 * 2 + 1, 6], key: MLXRandom.key(9))
        let resized = NFKMLXTripoSR.resizedPositionEmbeddings(embeddings, grid: 4)
        XCTAssertEqual(resized.shape, [1, 17, 6])
        XCTAssertEqual(cosine(resized[0..., 0, 0...], embeddings[0..., 0, 0...]), 1, accuracy: 1e-7, "the class position is kept")
        let same = NFKMLXTripoSR.resizedPositionEmbeddings(embeddings, grid: 2)
        XCTAssertEqual(cosine(same, embeddings), 1, accuracy: 1e-6, "a resize to the same grid is near the identity")
    }

    // MARK: - Marching cubes

    /// The reference's up (+z) becomes glTF's (+y), its front (+x) glTF's (+z), and its right (+y) glTF's
    /// (+x); the map keeps handedness, so right × up stays the front.
    func testTheMeshTurnsIntoGLTFsFrame() {
        let turned = NFKMLXTripoSR.gltfFrame([0, 0, 1, 1, 0, 0, 0, 1, 0, 0.25, -0.5, 0.75])
        XCTAssertEqual(turned, [0, 1, 0, 0, 0, 1, 1, 0, 0, -0.5, 0.75, 0.25])
        let up = simd_float3(turned[0], turned[1], turned[2]), front = simd_float3(turned[3], turned[4], turned[5])
        let right = simd_float3(turned[6], turned[7], turned[8])
        XCTAssertEqual(simd_cross(right, up), front)
    }

    func testMarchingCubesDrawsAClosedSphereAtTheRightRadius() throws {
        let resolution = 24
        var values = [Float](repeating: 0, count: resolution * resolution * resolution)
        let center = Float(resolution - 1) / 2, radius: Float = 7.5
        for x in 0 ..< resolution {
            for y in 0 ..< resolution {
                for z in 0 ..< resolution {
                    let dx = Float(x) - center, dy = Float(y) - center, dz = Float(z) - center
                    // Density falls with the distance, so the sphere is the region above the threshold.
                    values[(x * resolution + y) * resolution + z] = 100 - (dx * dx + dy * dy + dz * dz).squareRoot() * 10
                }
            }
        }
        let surface = NFKMarchingCubes.surface(of: values, resolution: resolution, threshold: 100 - radius * 10)
        let vertexCount = surface.vertices.count / 3
        let faceCount = surface.indices.count / 3
        XCTAssertGreaterThan(faceCount, 100)
        for index in 0 ..< vertexCount {
            let dx = surface.vertices[index * 3] - center, dy = surface.vertices[index * 3 + 1] - center, dz = surface.vertices[index * 3 + 2] - center
            XCTAssertEqual((dx * dx + dy * dy + dz * dz).squareRoot(), radius, accuracy: 0.12, "vertex \(index)")
        }
        // Closed and consistently wound: every directed edge appears once and its reverse once.
        var directed = Set<UInt64>()
        var faces = 0
        for face in stride(from: 0, to: surface.indices.count, by: 3) {
            let a = surface.indices[face], b = surface.indices[face + 1], c = surface.indices[face + 2]
            XCTAssertTrue(a != b && b != c && a != c, "a degenerate triangle")
            for (from, to) in [(a, b), (b, c), (c, a)] {
                XCTAssertTrue(directed.insert(UInt64(from) << 32 | UInt64(to)).inserted, "an edge used twice in the same direction")
            }
            faces += 1
        }
        for key in directed {
            XCTAssertTrue(directed.contains(key << 32 | key >> 32), "every edge has its reverse")
        }
        XCTAssertEqual(vertexCount - directed.count / 2 + faces, 2, "the Euler characteristic of a sphere")
        // Outward winding: each face's normal points away from the center.
        for face in stride(from: 0, to: surface.indices.count, by: 3) {
            func point(_ i: UInt32) -> SIMD3<Float> {
                SIMD3(surface.vertices[Int(i) * 3], surface.vertices[Int(i) * 3 + 1], surface.vertices[Int(i) * 3 + 2])
            }
            let a = point(surface.indices[face]), b = point(surface.indices[face + 1]), c = point(surface.indices[face + 2])
            let normal = simd_cross(b - a, c - a)
            let outward = (a + b + c) / 3 - SIMD3(repeating: center)
            XCTAssertGreaterThan(simd_dot(normal, outward), 0, "face \(face / 3) faces inward")
        }
    }

    func testMarchingCubesIsEmptyOffTheThresholdAndHandlesEveryCornerPattern() throws {
        let flat = [Float](repeating: 1, count: 27)
        XCTAssertTrue(NFKMarchingCubes.surface(of: flat, resolution: 3, threshold: 2).indices.isEmpty)
        XCTAssertTrue(NFKMarchingCubes.surface(of: flat, resolution: 3, threshold: 0).indices.isEmpty)
        // Every one of the 256 single-cube patterns yields whole triangles with distinct corners.
        for mask in 1 ..< 255 {
            var values = [Float](repeating: 0, count: 8)
            for corner in 0 ..< 8 where mask & (1 << corner) != 0 {
                let x = corner == 1 || corner == 2 || corner == 5 || corner == 6 ? 1 : 0
                let y = corner == 2 || corner == 3 || corner == 6 || corner == 7 ? 1 : 0
                let z = corner >= 4 ? 1 : 0
                values[(x * 2 + y) * 2 + z] = 1
            }
            let surface = NFKMarchingCubes.surface(of: values, resolution: 2, threshold: 0.5)
            XCTAssertEqual(surface.indices.count % 3, 0, "case \(mask)")
            XCTAssertGreaterThan(surface.indices.count, 0, "case \(mask)")
            for face in stride(from: 0, to: surface.indices.count, by: 3) {
                let a = surface.indices[face], b = surface.indices[face + 1], c = surface.indices[face + 2]
                XCTAssertTrue(a != b && b != c && a != c, "case \(mask): a degenerate triangle")
            }
        }
    }

    // MARK: - Parity with the release

    func testTheReleaseMatchesTheReferenceSeamsAndMesh() throws {
        let record = try loadArrays(url: try envPath("IK_PARITY_TRIPOSR"))
        let net = try NFKMLXTripoSR.network(weightsURL: try releaseWeights())
        func array(_ key: String) throws -> MLXArray {
            try XCTUnwrap(record[key], key)
        }

        // The preparation (crop, frame, composite) and the antialiased resize. The reference demo reads an
        // 8-bit image, so the plate is truncated to 8 bits the way `astype(np.uint8)` truncates it.
        let plate = floor(try array("input_image") * 255) / 255
        let rgba = concatenated([plate, try array("input_alpha").reshaped([-1, plate.dim(1), 1])], axis: -1)
        let prepared = NFKMLXTripoSR.prepare(rgba: rgba, foregroundRatio: 0.85)
        let referencePrepared = try array("prepared")
        XCTAssertEqual(prepared.shape, referencePrepared.shape)
        let preparedCosine = cosine(prepared, referencePrepared)
        let conditioning = NFKMLXTripoSR.resizeAntialiased(prepared, side: 512)
        let conditioningCosine = cosine(conditioning, try array("conditioning"))

        // The position embeddings after the bicubic resize.
        let positions = try XCTUnwrap(net.parameters().flattened().first { $0.0 == "image_tokenizer.model.embeddings.position_embeddings" }?.1)
        let positionCosine = cosine(positions[0], try array("position_embeddings"))

        // The network seams from the reference's own conditioning.
        let referenceConditioning = try array("conditioning").reshaped([1, 512, 512, 3])
        let tokens = net.imageTokens(referenceConditioning)
        eval(tokens)
        let tokenCosine = cosine(tokens[0], try array("image_tokens"))
        let codes = net.sceneCodes(try array("image_tokens").reshaped([1, -1, net.configuration.vitWidth]))
        eval(codes)
        let codeCosine = cosine(codes[0], try array("output"))

        // Density and color on the reference's 32³ grid, from the reference's scene code.
        let sceneCode = try array("output")
        let density = net.densityGrid(sceneCode: sceneCode, resolution: 32)
        let densityCosine = cosine(MLXArray(density), try array("density_32"))
        let radius = net.configuration.radius
        let axis = (0 ..< 32).map { -radius + 2 * radius * Float($0) / 31 }
        var grid = [Float]()
        for x in axis { for y in axis { for z in axis { grid += [x, y, z] } } }
        let color = net.query(MLXArray(grid).reshaped([-1, 3]), sceneCode: sceneCode).color
        eval(color)
        let colorCosine = cosine(color, try array("color_32"))

        print("TripoSR parity: prepared \(preparedCosine), conditioning \(conditioningCosine), positions \(positionCosine), "
              + "tokens \(tokenCosine), codes \(codeCosine), density \(densityCosine), color \(colorCosine)")

        // The ladder: the bfloat16 cast and the 4-bit pack of the same release, against the float32 record.
        let halfNet = try NFKMLXTripoSR.network(weightsURL: try releaseWeights(), castingTo: .bfloat16)
        let halfCodes = halfNet(referenceConditioning.asType(.bfloat16))
        eval(halfCodes)
        let halfCosine = cosine(halfCodes[0], try array("output"))
        let packedNet = try NFKMLXTripoSR.network(weightsURL: try releaseWeights(), quantizationBits: 4)
        let packedCodes = packedNet(referenceConditioning)
        eval(packedCodes)
        let packedCosine = cosine(packedCodes[0], try array("output"))
        print("TripoSR ladder: bfloat16 codes \(halfCosine), 4-bit codes \(packedCosine)")
        XCTAssertGreaterThan(halfCosine, 0.99)
        XCTAssertGreaterThan(packedCosine, 0.99)
        XCTAssertGreaterThan(preparedCosine, 0.99999)
        XCTAssertGreaterThan(conditioningCosine, 0.99999)
        XCTAssertGreaterThan(positionCosine, 0.99999)
        XCTAssertGreaterThan(tokenCosine, 0.9999)
        XCTAssertGreaterThan(codeCosine, 0.9999)
        XCTAssertGreaterThan(densityCosine, 0.9999)
        XCTAssertGreaterThan(colorCosine, 0.9999)

        // The mesh at the reference's resolution: vertex sets within a fraction of a cell of each other.
        guard let referenceVertices = record["mesh_vertices"], let referenceFaces = record["mesh_faces"] else {
            return
        }
        var options = NFKMLXTripoSROptions()
        options.resolution = Int(try array("mesh_resolution").item(Float.self))
        options.vertexColors = false
        let mesh = try net.mesh(sceneCode: sceneCode, options: options)
        let cell = 2 * radius / Float(options.resolution - 1)
        let ours = (0 ..< mesh.vertexCount).map { index -> simd_float3 in
            let p = mesh.position(at: UInt(index))
            return simd_make_float3(p.x, p.y, p.z)
        }
        // The reference writes its own frame, which the backend turns into glTF's.
        let referenceValues = NFKMLXTripoSR.gltfFrame(referenceVertices.reshaped([-1]).asArray(Float.self))
        let theirs = stride(from: 0, to: referenceVertices.dim(0) * 3, by: 3).map { index -> simd_float3 in
            simd_make_float3(referenceValues[index], referenceValues[index + 1], referenceValues[index + 2])
        }
        let forward = meanNearest(from: ours, to: theirs), backward = meanNearest(from: theirs, to: ours)
        print("TripoSR mesh: \(mesh.faceCount) faces against \(referenceFaces.dim(0)); nearest-vertex distance "
              + "\(forward / cell) and \(backward / cell) cells")
        XCTAssertLessThan(forward, 0.25 * cell)
        XCTAssertLessThan(backward, 0.25 * cell)
        XCTAssertEqual(Double(mesh.faceCount), Double(referenceFaces.dim(0)), accuracy: 0.1 * Double(referenceFaces.dim(0)))
    }

    /// The mean distance from each point of `from` to its nearest point of `to`, through a coarse hash grid.
    private func meanNearest(from: [simd_float3], to: [simd_float3]) -> Float {
        guard !from.isEmpty, !to.isEmpty else { return .infinity }
        let cell: Float = 0.05
        var buckets = [SIMD3<Int32>: [simd_float3]]()
        func key(_ p: simd_float3) -> SIMD3<Int32> { SIMD3<Int32>(Int32((p.x / cell).rounded(.down)), Int32((p.y / cell).rounded(.down)), Int32((p.z / cell).rounded(.down))) }
        for point in to {
            buckets[key(point), default: []].append(point)
        }
        var total: Float = 0
        for point in from {
            let center = key(point)
            var best = Float.infinity
            var ring = 0
            while best == .infinity || Float(ring - 1) * cell < best {
                for dx in -ring ... ring { for dy in -ring ... ring { for dz in -ring ... ring where max(abs(dx), abs(dy), abs(dz)) == ring {
                    for candidate in buckets[center &+ SIMD3<Int32>(Int32(dx), Int32(dy), Int32(dz))] ?? [] {
                        best = min(best, simd_distance(point, candidate))
                    }
                } } }
                ring += 1
                if ring > 64 { break }
            }
            total += best
        }
        return total / Float(from.count)
    }
}
