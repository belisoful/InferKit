//
//  NFKMLXStableFast3DTests.swift
//  InferKitMLXTests
//
//  The structural tests run a tiny configuration and synthetic grids. The parity tests read the release
//  directory (IK_VAL_SF3D: model.safetensors beside 160_tets.npz) and the oracle record
//  (IK_PARITY_SF3D, from `run_reference.py sf3d`), and skip without them.
//

import XCTest
import CoreGraphics
import ImageIO
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXStableFast3DTests: XCTestCase {

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

    /// A 28-pixel image of 2×2 patches, one tokenizer layer, one two-stream block of one basic block, 8×8
    /// planes upsampled twice, an 8-wide image tower, and an 8-wide illumination head.
    private func tinyConfiguration() -> NFKMLXStableFast3DConfiguration {
        var configuration = NFKMLXStableFast3DConfiguration()
        configuration.imageSize = 28
        configuration.pretrainedPositionGrid = 3
        configuration.dinoWidth = 16
        configuration.dinoDepth = 1
        configuration.dinoHeads = 2
        configuration.dinoHidden = 32
        configuration.modulationWidth = 8
        configuration.planeSize = 8
        configuration.tokenWidth = 32
        configuration.latentCount = 3
        configuration.backboneBlocks = 1
        configuration.basicBlocks = 1
        configuration.heads = 2
        configuration.headDimensions = 16
        configuration.normGroups = 4
        configuration.upsampleLayers = 2
        configuration.upsampleFactor = 2
        configuration.planeChannels = 4
        configuration.decoderNeurons = 8
        configuration.clip = .tiny
        configuration.estimatorWidth = NFKMLXCLIPConfiguration.tiny.embedDimensions
        configuration.illuminationWidth = 8
        return configuration
    }

    private func envPath(_ key: String) throws -> URL {
        guard let path = NFKMLXValidationConfig.environment[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key)")
        }
        return URL(fileURLWithPath: path)
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        cosine(a.asType(.float32).reshaped([-1]).asArray(Float.self), b.asType(.float32).reshaped([-1]).asArray(Float.self))
    }

    private func cosine(_ x: [Float], _ y: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(x.count, y.count) {
            dot += Double(x[i]) * Double(y[i]); na += Double(x[i]) * Double(x[i]); nb += Double(y[i]) * Double(y[i])
        }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
    }

    private func maxDifference(_ x: [Float], _ y: [Float]) -> Float {
        zip(x, y).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }

    /// The values whose bits differ; a length mismatch counts every value.
    private func bitMismatches(_ x: [Float], _ y: [Float]) -> Int {
        guard x.count == y.count else { return max(x.count, y.count) }
        return zip(x, y).filter { $0.0.bitPattern != $0.1.bitPattern }.count
    }

    /// `[rows, columns]` row-major to `[columns, rows]`.
    private func transposed(_ x: [Float], rows: Int) -> [Float] {
        let columns = x.count / rows
        var out = [Float](repeating: 0, count: x.count)
        for row in 0 ..< rows {
            for column in 0 ..< columns { out[column * rows + row] = x[row * columns + column] }
        }
        return out
    }

    /// The record's plate and its alpha as the oracle writes them (`astype(np.uint8)` truncates).
    private func recordRGBA(_ record: [String: MLXArray]) throws -> (bytes: [UInt8], side: Int) {
        let plate = try XCTUnwrap(record["input_image"]).reshaped([-1]).asArray(Float.self)
        let alpha = try XCTUnwrap(record["input_alpha"]).reshaped([-1]).asArray(Float.self)
        let side = Int(Double(alpha.count).squareRoot())
        var rgba = [UInt8](repeating: 0, count: side * side * 4)
        for pixel in 0 ..< side * side {
            for channel in 0 ..< 3 { rgba[pixel * 4 + channel] = UInt8(plate[pixel * 3 + channel] * 255) }
            rgba[pixel * 4 + 3] = UInt8(alpha[pixel] * 255)
        }
        return (rgba, side)
    }

    private func latticeGrid(_ n: Int) throws -> NFKMLXTetrahedralGrid {
        NFKMLXTetrahedralGrid.lattice(divisions: n)
    }

    // MARK: - Structure

    func testTheForwardProducesSceneCodesAndTheHeadsRead() throws {
        let net = NFKMLXStableFast3D.makeNet(tinyConfiguration())
        let image = MLXRandom.uniform(0 ..< 1, [1, 28, 28, 3], key: MLXRandom.key(3))
        let tokens = net.imageTokens(image)
        XCTAssertEqual(tokens.shape, [1, 5, 16], "the class token and four patches")
        let triplane = net.triplaneTokens(tokens)
        XCTAssertEqual(triplane.shape, [1, 3 * 64, 32])
        let codes = net.sceneCodes(triplane)
        XCTAssertEqual(codes.shape, [1, 3, 16, 16, 4])
        XCTAssertEqual(cosine(net(image), codes), 1, accuracy: 1e-6)
        let points = MLXRandom.uniform(-0.8 ..< 0.8, [64, 3], key: MLXRandom.key(4))
        let geometry = net.geometry(points, sceneCode: codes[0])
        let surface = net.surface(points, sceneCode: codes[0])
        eval(geometry.density, geometry.vertexOffset, surface.albedo, surface.normal)
        XCTAssertEqual(geometry.density.shape, [64])
        XCTAssertEqual(geometry.vertexOffset.shape, [64, 3])
        XCTAssertTrue(geometry.density.asArray(Float.self).allSatisfy { $0 > 0 }, "density is an exponential")
        XCTAssertTrue(surface.albedo.asArray(Float.self).allSatisfy { $0 > 0 && $0 < 1 })
        let lengths = sqrt((surface.normal * surface.normal).sum(axis: -1)).asArray(Float.self)
        XCTAssertTrue(lengths.allSatisfy { abs($0 - 1) < 1e-5 }, "the perturbed normals are unit")
        let materials = net.materials(image)
        eval(materials.roughness, materials.metallic)
        for value in [materials.roughness[0].item(Float.self), materials.metallic[0].item(Float.self)] {
            XCTAssertTrue(value >= 0 && value <= 1, "a beta mode lies in 0...1")
        }
        let amplitudes = net.illumination(triplane)
        eval(amplitudes)
        XCTAssertEqual(amplitudes.shape, [1, 24])
        XCTAssertTrue(amplitudes.asArray(Float.self).allSatisfy { $0 > 0 }, "the amplitudes are a softplus")
    }

    func testTheReferenceKeyLayoutIsHeld() throws {
        let net = NFKMLXStableFast3D.makeNet(tinyConfiguration())
        let keys = Set(net.parameters().flattened().map(\.0))
        for expected in [
            "camera_embedder.linear.weight", "image_tokenizer.model.embeddings.cls_token",
            "image_tokenizer.model.embeddings.position_embeddings",
            "image_tokenizer.model.embeddings.patch_embeddings.projection.weight",
            "image_tokenizer.model.encoder.layer.0.attention.attention.query.weight",
            "image_tokenizer.model.encoder.layer.0.attention.output.dense.bias",
            "image_tokenizer.model.encoder.layer.0.layer_scale1.lambda1", "image_tokenizer.model.encoder.layer.0.mlp.fc2.weight",
            "image_tokenizer.model.encoder.layer.0.norm1_modulation.linear2.weight",
            "image_tokenizer.model.encoder.layer.0.norm2_modulation.linear2.bias", "image_tokenizer.model.layernorm.weight",
            "tokenizer.embeddings", "backbone.latent_init", "backbone.norm_triplane.weight", "backbone.proj_triplane.weight",
            "backbone.norm_image.bias", "backbone.proj_image.weight", "backbone.norm_latent.weight", "backbone.proj_latent.bias",
            "backbone.main_blocks.0.fuse_block_in.attn.wq.weight", "backbone.main_blocks.0.fuse_block_in.attn.proj.bias",
            "backbone.main_blocks.0.fuse_block_in.norm_z1.weight", "backbone.main_blocks.0.fuse_block_in.ff.net.0.proj.weight",
            "backbone.main_blocks.0.fuse_block_in.ff.net.2.weight", "backbone.main_blocks.0.transformer_block.0.attn1.wk.weight",
            "backbone.main_blocks.0.transformer_block.0.attn2.wv.weight", "backbone.main_blocks.0.transformer_block.0.norm3.bias",
            "backbone.main_blocks.0.fuse_block_out.attn.wq.weight", "backbone.proj_out.weight",
            "post_processor.upsample.0.weight", "post_processor.upsample.2.bias",
            "decoder.heads.density.0.weight", "decoder.heads.density.4.bias", "decoder.heads.features.6.weight",
            "decoder.heads.perturb_normal.0.bias", "decoder.heads.vertex_offset.4.weight",
            "image_estimator.model.visual.conv1.weight", "image_estimator.model.visual.class_embedding",
            "image_estimator.model.visual.transformer.resblocks.0.attn.in_proj_weight", "image_estimator.model.visual.proj",
            "image_estimator.heads.roughness.shared.4.weight", "image_estimator.heads.roughness.alpha.2.bias",
            "image_estimator.heads.metallic.beta.0.weight",
            "global_estimator.layers.0.weight", "global_estimator.layers.2.bias", "global_estimator.heads.sg_amplitudes.6.weight",
        ] {
            XCTAssertTrue(keys.contains(expected), expected)
        }
        XCTAssertFalse(keys.contains { $0.contains("wq.bias") }, "the attention projections carry no bias")
        XCTAssertFalse(keys.contains { $0.contains("norm_x") }, "the release does not normalize the attended stream")
        XCTAssertFalse(keys.contains { $0.hasPrefix("image_estimator.model.") && !$0.hasPrefix("image_estimator.model.visual.") },
                       "the text tower is not held")
        XCTAssertEqual(net.parameters().flattened().first { $0.0 == "tokenizer.embeddings" }?.1.shape, [3, 32, 8, 8])
        XCTAssertTrue(NFKMLXStableFast3D.droppedKey("image_estimator.model.token_embedding.weight"))
        XCTAssertTrue(NFKMLXStableFast3D.droppedKey("image_tokenizer.model.embeddings.mask_token"))
        XCTAssertFalse(NFKMLXStableFast3D.droppedKey("image_estimator.model.visual.proj"))
        XCTAssertEqual(NFKSF3DBetaHead.moduleKey("image_estimator.heads.roughness.0.4.weight"), "image_estimator.heads.roughness.shared.4.weight")
        XCTAssertEqual(NFKSF3DBetaHead.moduleKey("image_estimator.heads.metallic.2.0.bias"), "image_estimator.heads.metallic.beta.0.bias")
        XCTAssertEqual(NFKSF3DBetaHead.moduleKey("image_estimator.heads.metallic.shared.0.bias"), "image_estimator.heads.metallic.shared.0.bias",
                       "a module key passes through, so a saved checkpoint reloads")
    }

    func testASavedCheckpointReloadsThroughTheFactoryAndReproducesTheForward() throws {
        let configuration = tinyConfiguration()
        let net = NFKMLXStableFast3D.makeNet(configuration)
        let image = MLXRandom.uniform(0 ..< 1, [1, 28, 28, 3], key: MLXRandom.key(5))
        let before = net(image)
        eval(before)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sf3d-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXStableFast3D.network(weightsURL: url, configuration: configuration)
        let after = reloaded(image)
        eval(after)
        XCTAssertEqual(cosine(before, after), 1, accuracy: 1e-7)
    }

    func testABFloat16CastAndAPackedLoadReproduceTheForward() throws {
        let configuration = tinyConfiguration()
        let net = NFKMLXStableFast3D.makeNet(configuration)
        let image = MLXRandom.uniform(0 ..< 1, [1, 28, 28, 3], key: MLXRandom.key(6))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sf3d-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reference = net(image)
        let half = try NFKMLXStableFast3D.network(weightsURL: url, configuration: configuration, castingTo: .bfloat16)
        let halfCodes = half(image.asType(.bfloat16))
        let packed = try NFKMLXStableFast3D.network(weightsURL: url, configuration: configuration, quantizationBits: 8, groupSize: 32)
        let packedCodes = packed(image)
        eval(reference, halfCodes, packedCodes)
        XCTAssertGreaterThan(cosine(reference, halfCodes), 0.99)
        XCTAssertGreaterThan(cosine(reference, packedCodes), 0.99)
        XCTAssertThrowsError(try NFKMLXStableFast3D.network(weightsURL: url, configuration: configuration, quantizationBits: 4, groupSize: 16))
        XCTAssertThrowsError(try NFKMLXStableFast3D.network(weightsURL: url, configuration: configuration, quantizationBits: 12))
    }

    func testTheReconstructionIsATexturedMeshThatRoundTripsThroughGLB() throws {
        let configuration = tinyConfiguration()
        let net = NFKMLXStableFast3D.makeNet(configuration)
        let grid = try latticeGrid(12)
        let side = 40
        var rgba = [UInt8](repeating: 0, count: side * side * 4)
        for y in 0 ..< side {
            for x in 0 ..< side where (x - 20) * (x - 20) + (y - 20) * (y - 20) < 120 {
                let at = (y * side + x) * 4
                rgba[at] = UInt8(100 + x); rgba[at + 1] = UInt8(60 + y); rgba[at + 2] = 90; rgba[at + 3] = 255
            }
        }
        // The random network's density has no fixed scale; draw the surface at its median over the grid.
        let (color, _) = NFKMLXStableFast3D.conditioning(rgba: rgba, width: side, height: side, side: configuration.imageSize,
                                                          foregroundRatio: 0.85, background: configuration.background)
        let sceneCode = net(color.expandedDimensions(axis: 0))[0]
        let gridPoints: [Float] = grid.vertices.map { $0 * 1.74 - 0.87 }
        let points = MLXArray(gridPoints).reshaped([grid.vertexCount, 3])
        let density = net.geometry(points, sceneCode: sceneCode).density.asArray(Float.self).sorted()
        var options = NFKMLXStableFast3DOptions()
        options.threshold = density[density.count / 2]
        options.textureResolution = 64
        options.estimateIllumination = true
        let reconstruction = try net.reconstruct(rgba: rgba, width: side, height: side, grid: grid, options: options)
        let mesh = reconstruction.mesh
        XCTAssertGreaterThan(mesh.faceCount, 0)
        XCTAssertNotNil(mesh.normals)
        XCTAssertNotNil(mesh.textureCoordinates)
        XCTAssertEqual(reconstruction.illuminationAmplitudes?.count, 24)
        let material = try XCTUnwrap(mesh.material)
        XCTAssertEqual(material.baseColorTexture?.mimeType, NFKMeshTextureMIMETypePNG)
        let texture = try XCTUnwrap(material.baseColorTexture?.data)
        let source = try XCTUnwrap(CGImageSourceCreateWithData(texture as CFData, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual(image.width, 64)
        XCTAssertNotNil(material.normalTexture)
        for index in 0 ..< mesh.vertexCount {
            let uv = mesh.textureCoordinate(at: index)
            XCTAssertTrue(uv.u >= 0 && uv.u <= 1 && uv.v >= 0 && uv.v <= 1)
        }
        let reread = try NFKMesh(glbData: mesh.glbData())
        XCTAssertEqual(reread.material, material)
        XCTAssertEqual(reread.faceCount, mesh.faceCount)
    }

    func testTheBackendTakesTheTextureFormatAndRefusesAnotherAndReportsItsKeys() throws {
        let configuration = tinyConfiguration()
        let backend = NFKMLXStableFast3DBackend(net: NFKMLXStableFast3D.makeNet(configuration), grid: try latticeGrid(4),
                                                weightsURL: nil, identifier: NFKMLXStableFast3D.modelName)
        XCTAssertTrue(backend.supportedOutputKeys.contains(NFKOutputMesh))
        XCTAssertTrue(backend.supportedParameterKeys.contains(NFKMLXStableFast3DParameterKey.textureResolution))
        XCTAssertTrue(backend.supportedParameterKeys.contains(NFKMLXStableFast3DParameterKey.referenceArithmetic))
        let request = NFKInferenceRequest(inputs: [:], parameters: [NFKMLXStableFast3DParameterKey.textureFormat: NFKMeshTextureMIMETypeJPEG,
                                                                   NFKMLXStableFast3DParameterKey.textureResolution: 2048,
                                                                   NFKMLXStableFast3DParameterKey.foregroundRatio: 0.5,
                                                                   NFKMLXStableFast3DParameterKey.referenceArithmetic: true])
        let options = try NFKMLXStableFast3DBackend.options(from: request)
        XCTAssertTrue(options.referenceArithmetic)
        XCTAssertEqual(options.textureFormat, NFKMeshTextureMIMETypeJPEG)
        XCTAssertEqual(options.textureResolution, 2048)
        XCTAssertEqual(options.foregroundRatio, 0.5)
        let refused = NFKInferenceRequest(inputs: [:], parameters: [NFKMLXStableFast3DParameterKey.textureFormat: "image/webp"])
        XCTAssertThrowsError(try NFKMLXStableFast3DBackend.options(from: refused))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])), "no image")
    }

    // MARK: - Preparation

    func testFramingCropsASquareAroundTheAlphaBoxAndKeepsAnOpaqueImage() {
        var rgba = [UInt8](repeating: 0, count: 20 * 10 * 4)
        for y in 2 ... 6 {
            for x in 3 ... 11 { rgba[(y * 20 + x) * 4 + 3] = 255; rgba[(y * 20 + x) * 4] = 200 }
        }
        let framed = NFKMLXStableFast3D.frame(rgba: rgba, width: 20, height: 10, foregroundRatio: 0.85)
        // The box is 8 wide (11 − 3) and 4 high; the side is Int(8 / 0.85) = 9 around the center (7, 4).
        XCTAssertEqual(framed.width, 9)
        XCTAssertEqual(framed.height, 9)
        let top = Int(4.0 - (8.0 / 0.85) / 2), left = Int(7.0 - (8.0 / 0.85) / 2)
        XCTAssertEqual(framed.pixels[((2 - top) * 9 + (3 - left)) * 4 + 3], 255, "the box's corner lands inside the crop")
        let opaque = [UInt8](repeating: 255, count: 6 * 4 * 4)
        let kept = NFKMLXStableFast3D.frame(rgba: opaque, width: 6, height: 4, foregroundRatio: 0.85)
        XCTAssertEqual(kept.width, 6)
        XCTAssertEqual(kept.height, 4)
        XCTAssertEqual(kept.pixels, opaque)
    }

    func testThePillowResizePremultipliesAndRestoresTheColors() {
        // An opaque image resizes as its colors do; a transparent texel carries no color into its neighbors.
        var rgba = [UInt8](repeating: 255, count: 4 * 4 * 4)
        for pixel in 0 ..< 16 { rgba[pixel * 4] = UInt8(pixel * 16) }
        let opaque = NFKMLXStableFast3D.pilResized(rgba, width: 4, height: 4, to: 2)
        var rgb = [UInt8]()
        for pixel in 0 ..< 16 { rgb += [rgba[pixel * 4], 255, 255] }
        let plain = NFKMLXPILResample.resampled(rgb, width: 4, height: 4, toWidth: 2, toHeight: 2, filter: .bicubic)
        for pixel in 0 ..< 4 {
            XCTAssertEqual(opaque[pixel * 4], plain[pixel * 3])
            XCTAssertEqual(opaque[pixel * 4 + 3], 255)
        }
        var cutout = [UInt8](repeating: 0, count: 2 * 1 * 4)
        cutout[0] = 255; cutout[3] = 255                          // red and opaque, then transparent white-less black
        cutout[4] = 0; cutout[5] = 255; cutout[7] = 0             // green, transparent
        let wide = NFKMLXStableFast3D.pilResized(cutout, width: 2, height: 1, to: 4)
        for pixel in 0 ..< 4 where wide[pixel * 4 + 3] > 0 {
            XCTAssertEqual(wide[pixel * 4 + 1], 0, "the transparent texel's green does not bleed")
        }
    }

    // MARK: - Reference arithmetic

    /// Inputs both this test and the digest script build exactly: small integers over a divisor.
    private func values(_ count: Int, seed: Int, divisor: Float = 97) -> [Float] {
        (0 ..< count).map { Float(($0 * 7919 + seed * 104729) % 1000 - 500) / divisor }
    }

    /// FNV-1a over the little-endian bytes of each float.
    private func digest(_ values: [Float]) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for value in values {
            let bits = value.bitPattern
            for byte in 0 ..< 4 {
                hash ^= UInt64((bits >> (8 * UInt32(byte))) & 0xff)
                hash = hash &* 0x100000001b3
            }
        }
        return hash
    }

    func testTheReferenceKernelsReproduceTorchBitForBit() throws {
        // Digests of torch 2.14.1's CPU float32 results on Apple silicon for the same inputs.
        let expected: [String: UInt64] = [
            "exp": 0x919d21ff9cebb9a2, "tanh": 0x33eee2eb4e826bd5, "sin": 0xd3a1165b2797de96, "cos": 0x52b3b074fff239ec,
            "gelu": 0x190229e89305d95b, "silu": 0xa9c46a69dda2bb21, "sigmoid": 0x68ce9a22b4c429a8,
            "linear": 0x8767065a6c10c579, "linearView": 0xec19843158489d0c, "layerNorm": 0xbbd2de86576d33d3,
            "groupNorm": 0x0fa4eec163c863dc, "conv": 0xdef98b8656bef2c5, "attention": 0x75695bd50d06e31f,
            "attentionBlocks": 0x8ce3fab8d28c1e61, "bicubic": 0xe6f17e4fc65f5441, "gridSample": 0x5360d3ccd573f2d8,
            "linearColumn": 0x043da42e43bda933, "linearRow": 0x68e4e1c09facd544,
        ]
        var measured = [String: UInt64]()
        let kernels = try NFKSF3DTorchKernels()
        let x = values(43, seed: 1, divisor: 37)
        measured["exp"] = digest(x.map(NFKSF3DTorchMath.exp))
        measured["tanh"] = digest(x.map { NFKSF3DTorchMath.tanh($0 / 4) })
        measured["sin"] = digest(x.map(NFKSF3DTorchMath.sin))
        measured["cos"] = digest(x.map(NFKSF3DTorchMath.cos))
        var gelu = x, silu = x, sigmoid = x
        NFKSF3DTorchMath.gelu(&gelu); NFKSF3DTorchMath.silu(&silu); NFKSF3DTorchMath.sigmoid(&sigmoid)
        measured["gelu"] = digest(gelu); measured["silu"] = digest(silu); measured["sigmoid"] = digest(sigmoid)

        let weight = values(7 * 12, seed: 3), bias = values(7, seed: 4)
        measured["linear"] = digest(kernels.linear(values(5 * 12, seed: 2), rows: 5, weight: weight, outputs: 7, bias: bias))
        let transposed = values(12 * 5, seed: 5)
        measured["linearView"] = digest(transposed.withUnsafeBufferPointer {
            kernels.linear(.init(base: $0.baseAddress!, rows: 5, columns: 12, rowStride: 1, columnStride: 5), weight: weight, outputs: 7, bias: bias)
        })
        measured["layerNorm"] = digest(kernels.layerNorm(values(3 * 37, seed: 6), width: 37, weight: values(37, seed: 7),
                                                         bias: values(37, seed: 8), epsilon: 1e-5))
        measured["groupNorm"] = digest(kernels.groupNorm(values(8 * 30, seed: 9), channels: 8, groups: 4, weight: values(8, seed: 10),
                                                         bias: values(8, seed: 11), epsilon: 1e-6))
        measured["conv"] = digest(values(3 * 10 * 10, seed: 12).withUnsafeBufferPointer {
            kernels.conv2d($0.baseAddress!, channels: 3, height: 10, width: 10, weight: values(4 * 3 * 3 * 3, seed: 13), outputs: 4,
                           kernel: 3, stride: 1, padding: 1, bias: values(4, seed: 14))
        })
        func attention(queries: Int, keys: Int, heads: Int, width: Int, seed: Int, scale: Double) -> UInt64 {
            digest(kernels.attention(q: values(queries * heads * width, seed: seed), k: values(keys * heads * width, seed: seed + 1),
                                     v: values(keys * heads * width, seed: seed + 2), queries: queries, keys: keys, heads: heads,
                                     scale: Float(scale)))
        }
        measured["attention"] = attention(queries: 9, keys: 11, heads: 2, width: 4, seed: 15, scale: 1 / Double(4).squareRoot())
        measured["attentionBlocks"] = attention(queries: 300, keys: 600, heads: 2, width: 8, seed: 18, scale: pow(8, -0.5))
        measured["bicubic"] = digest(NFKSF3DTorchKernels.bicubic(values(2 * 5 * 5, seed: 21), channels: 2, height: 5, width: 5,
                                                                 outHeight: 4, outWidth: 4, scaleFactors: (4.1 / 5, 4.1 / 5)))
        var sampled = [Float](repeating: 0, count: 3 * 20)
        values(3 * 6 * 6, seed: 22).withUnsafeBufferPointer { plane in
            sampled.withUnsafeMutableBufferPointer { out in
                NFKSF3DTorchKernels.gridSample(plane.baseAddress!, channels: 3, side: 6, x: values(20, seed: 23, divisor: 400),
                                               y: values(20, seed: 24, divisor: 400), into: out.baseAddress!)
            }
        }
        measured["gridSample"] = digest(sampled)
        // Products with a dimension of one, each given an operand off the cache line; torch's start on one.
        let column = [0] + values(5001 * 64, seed: 25)
        measured["linearColumn"] = digest(column.withUnsafeBufferPointer {
            kernels.linear(.rowMajor($0.baseAddress! + 1, rows: 5001, columns: 64), weight: values(64, seed: 26), outputs: 1,
                           bias: values(1, seed: 27))
        })
        let rowWeight = [0] + values(2048 * 768, seed: 29)
        measured["linearRow"] = digest(rowWeight.withUnsafeBufferPointer {
            kernels.linear(values(768, seed: 28), rows: 1, weight: $0.baseAddress! + 1, outputs: 2048, bias: values(2048, seed: 30))
        })
        for (name, value) in expected.sorted(by: { $0.key < $1.key }) {
            XCTAssertEqual(measured[name], value, name)
        }
    }

    func testTheBetaModeAndTheSoftplusFollowTorch() {
        let alpha = MLXArray([Float(3), 0.5, 1.5, 0.5]), beta = MLXArray([Float(2), 0.5, 0.5, 3])
        let mode = NFKMLXStableFast3D.betaMode(alpha: alpha, beta: beta).asArray(Float.self)
        XCTAssertEqual(mode[0], 2.0 / 3.0, accuracy: 1e-6)
        XCTAssertEqual(mode[1], 1, "both below one: the first one-hot")
        XCTAssertEqual(mode[2], 1, "beta below one: all the weight at one")
        XCTAssertEqual(mode[3], 0, "alpha below one: all the weight at zero")
        let soft = NFKMLXStableFast3D.softplus(MLXArray([Float(-3), 0, 25])).asArray(Float.self)
        XCTAssertEqual(soft[0], log1p(exp(-3)), accuracy: 1e-6)
        XCTAssertEqual(soft[1], log(2), accuracy: 1e-6)
        XCTAssertEqual(soft[2], 25, "above the threshold the input passes")
    }

    func testThePositionEmbeddingsResizeByTheScaleFactor() {
        let embeddings = MLXRandom.normal([1, 10, 4], key: MLXRandom.key(8))
        let resized = NFKMLXStableFast3D.resizedPositionEmbeddings(embeddings, grid: 2)
        XCTAssertEqual(resized.shape, [1, 5, 4])
        XCTAssertEqual(cosine(resized[0..., 0, 0...], embeddings[0..., 0, 0...]), 1, accuracy: 1e-7, "the class position is kept")
    }

    // MARK: - Geometry

    func testMarchingTetrahedraDrawsAClosedSphereAndTheExportFacesOutward() throws {
        let grid = try latticeGrid(16)
        let level = (0 ..< grid.vertexCount).map { index -> Float in
            let p = NFKSF3DVector.at(grid.vertices, index)
            let d = ((p.x - 0.5) * (p.x - 0.5) + (p.y - 0.5) * (p.y - 0.5) + (p.z - 0.5) * (p.z - 0.5)).squareRoot()
            return 0.3 - d
        }
        for tetrahedron in 0 ..< grid.tetrahedronCount {
            let corners = (0 ..< 4).map { NFKSF3DVector.at(grid.vertices, Int(grid.tetrahedra[tetrahedron * 4 + $0])) }
            XCTAssertGreaterThan((corners[1] - corners[0]).dot((corners[2] - corners[0]).cross(corners[3] - corners[0])), 0,
                                 "every tetrahedron is positively oriented, as the release's are")
        }
        let (vertices, faces) = NFKMarchingTetrahedra.surface(positions: grid.vertices, level: level, tetrahedra: grid.tetrahedra)
        XCTAssertGreaterThan(faces.count, 0)
        for index in 0 ..< vertices.count / 3 {
            let p = NFKSF3DVector.at(vertices, index)
            let d = ((p.x - 0.5) * (p.x - 0.5) + (p.y - 0.5) * (p.y - 0.5) + (p.z - 0.5) * (p.z - 0.5)).squareRoot()
            XCTAssertEqual(d, 0.3, accuracy: 0.02)
        }
        var edges = [UInt64: Int]()
        for face in 0 ..< faces.count / 3 {
            for corner in 0 ..< 3 {
                let a = UInt64(faces[face * 3 + corner]), b = UInt64(faces[face * 3 + (corner + 1) % 3])
                edges[min(a, b) << 32 | max(a, b), default: 0] += 1
            }
        }
        XCTAssertTrue(edges.values.allSatisfy { $0 == 2 }, "every edge joins two triangles")
        // The extractor winds its triangles inward; the export reverses them, as the release's `invert` does.
        func volume(_ positions: [Float], _ faces: [UInt32]) -> Double {
            var total = 0.0
            for face in 0 ..< faces.count / 3 {
                let a = NFKSF3DVector.at(positions, Int(faces[face * 3])), b = NFKSF3DVector.at(positions, Int(faces[face * 3 + 1]))
                let c = NFKSF3DVector.at(positions, Int(faces[face * 3 + 2]))
                total += Double(a.dot(b.cross(c)))
            }
            return total / 6
        }
        XCTAssertLessThan(volume(vertices, faces), 0)
        let normals = NFKSF3DVertexNormals(positions: vertices, faces: faces)
        let uv = try NFKSF3DUnwrapper.unwrap(positions: vertices, normals: normals, faces: faces, padding: 0.02)
        let split = NFKMLXStableFast3DGeometry.splitMesh(positions: vertices, faces: faces, cornerUV: uv)
        let material = try NFKMeshMaterial(baseColorTexture: nil, normalTexture: nil, metallicRoughnessTexture: nil,
                                           baseColorFactor: nil, metallicFactor: 0, roughnessFactor: 1)
        let mesh = try NFKMLXStableFast3DGeometry.exported(split: split, material: material)
        let positions: [Float] = mesh.positions.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let indices: [UInt32] = mesh.indices.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        XCTAssertGreaterThan(volume(positions, indices), 0, "the exported mesh winds counter-clockwise seen from outside")
        XCTAssertLessThan(Int(mesh.vertexCount), split.positions.count / 3, "the export welds the vertices inside each island")
    }

    func testTheChartTestFindsAnOverlapAndIgnoresASharedEdgeAndADisjointChart() {
        let a = NFKSF3DChart(a: (0, 0), b: (0.4, 0), c: (0, 0.4))
        let overlapping = NFKSF3DChart(a: (0.1, 0.1), b: (0.5, 0.1), c: (0.1, 0.5))
        let neighbor = NFKSF3DChart(a: (0.4, 0), b: (0.4, 0.4), c: (0, 0.4))
        let far = NFKSF3DChart(a: (0.6, 0.6), b: (0.9, 0.6), c: (0.6, 0.9))
        XCTAssertTrue(a.overlaps(overlapping) || overlapping.overlaps(a))
        XCTAssertFalse(a.overlaps(neighbor) || neighbor.overlaps(a), "a shared edge has no area")
        XCTAssertFalse(a.overlaps(far))
        let charts = [a, overlapping, neighbor, far, a]
        let pairs = NFKSF3DChart.overlappingPairs([0, 1, 2, 3, 4], charts: charts).map { [$0.0, $0.1] }
        XCTAssertTrue(pairs.contains([0, 1]))
        XCTAssertTrue(pairs.contains([1, 4]))
        XCTAssertFalse(pairs.contains([0, 4]), "an identical chart is skipped, as the extension's equality skips it")
        XCTAssertFalse(pairs.contains([0, 2]))
    }

    func testTheAtlasMovesTheOccludedFaceToTheNextSlot() {
        // Two faces box-projected onto slot 0 whose charts overlap; slot 0 keeps the one nearer along −x.
        let positions: [Float] = [0, 0, 0, 0, 1, 0, 0, 0, 1,   2, 0, 0, 2, 1, 0, 2, 0, 1]
        let faces: [UInt32] = [0, 1, 2, 3, 4, 5]
        let uv: [Float] = [0.1, 0.1, 0.5, 0.1, 0.1, 0.5,   0.2, 0.2, 0.6, 0.2, 0.2, 0.6]
        let atlas = NFKSF3DUnwrapper.atlasIndices(positions: positions, faces: faces, uv: uv, index: [0, 0])
        XCTAssertEqual(atlas, [0, 6])
        let placed = NFKSF3DUnwrapper.placed(uv: uv, atlas: atlas, padding: 0.02)
        for value in placed[0 ..< 6] {
            XCTAssertLessThanOrEqual(value, Float(1.0 / 3.0) + 1e-6, "slot 0 fills the first third")
        }
        for corner in 3 ..< 6 {
            XCTAssertLessThanOrEqual(placed[corner * 2], 0.5, "slot 6 sits in the left half")
            XCTAssertGreaterThanOrEqual(placed[corner * 2 + 1], Float(2.0 / 3.0) - 1e-6, "slot 6 sits in the lower third")
        }
    }

    func testTheRasterizerCoversATriangleAndInterpolatesItsCorners() {
        let uv: [Float] = [0, 0, 1, 0, 0, 1]
        let raster = NFKSF3DBaker.rasterize(uv: uv, faces: [0, 1, 2], side: 8)
        // Texel (row, column) samples (column / 8, 1 − row / 8): covered when column + (8 − row) ≤ 8.
        for row in 0 ..< 8 {
            for column in 0 ..< 8 {
                XCTAssertEqual(raster.triangle[row * 8 + column] >= 0, column <= row, "texel \(row), \(column)")
            }
        }
        let attribute: [Float] = [0, 0, 0, 1, 0, 0, 0, 1, 0]
        let values = NFKSF3DBaker.interpolate(attribute, faces: [0, 1, 2], weights: raster.weights, triangle: raster.triangle)
        var next = 0
        for row in 0 ..< 8 {
            for column in 0 ..< 8 where raster.triangle[row * 8 + column] >= 0 {
                XCTAssertEqual(values[next * 3], Float(column) / 8, accuracy: 1e-6)
                XCTAssertEqual(values[next * 3 + 1], 1 - Float(row) / 8, accuracy: 1e-6)
                next += 1
            }
        }
    }

    func testTheIslandPaddingGrowsOneTexelAnIterationWithTheNeighborsMean() {
        let side = 9
        var texture = [Float](repeating: 0, count: side * side * 3)
        var mask = [Bool](repeating: false, count: side * side)
        mask[4 * side + 4] = true
        texture[(4 * side + 4) * 3] = 0.8
        let once = NFKSF3DBaker.dilated(texture, mask: mask, side: side, iterations: 1)
        XCTAssertEqual(once[(4 * side + 4) * 3], 0.8, "a masked texel keeps its value")
        XCTAssertGreaterThan(once[(3 * side + 3) * 3], 0)
        XCTAssertEqual(once[(2 * side + 2) * 3], 0, "one iteration reaches one texel")
        let twice = NFKSF3DBaker.dilated(texture, mask: mask, side: side, iterations: 2)
        XCTAssertGreaterThan(twice[(2 * side + 2) * 3], 0)
    }

    // MARK: - Parity

    func testTheReleaseReconstructsTheReferenceMeshEndToEnd() throws {
        let directory = try envPath("IK_VAL_SF3D")
        let record = try loadArrays(url: try envPath("IK_PARITY_SF3D"))
        func floats(_ key: String) throws -> [Float] { try XCTUnwrap(record[key], key).asType(.float32).reshaped([-1]).asArray(Float.self) }
        let weightsURL = directory.appendingPathComponent(NFKMLXStableFast3D.weightsFileName)
        let net = try NFKMLXStableFast3D.network(weightsURL: weightsURL)
        let reference = try NFKMLXStableFast3DReferenceNetwork(weightsURL: weightsURL)
        let grid = try NFKMLXTetrahedralGrid(contentsOf: directory.appendingPathComponent(NFKMLXStableFast3D.gridFileName))
        let (rgba, side) = try recordRGBA(record)
        var options = NFKMLXStableFast3DOptions.default
        options.referenceArithmetic = true
        let mesh = try reference.reconstruct(rgba: rgba, width: side, height: side, grid: grid, net: net, options: options).mesh
        let releasedPositions = try floats("released_positions"), releasedNormals = try floats("released_normals")
        let releasedFaces = try XCTUnwrap(record["released_faces"]).asType(.int64).reshaped([-1]).asArray(Int64.self)
        XCTAssertEqual(Int(mesh.faceCount) * 3, releasedFaces.count, "the surface has the release's triangles")
        var positionDifference: Float = .infinity, normalCosine = 0.0, facesEqual = false
        if Int(mesh.vertexCount) * 3 == releasedPositions.count {
            let ours: [Float] = mesh.positions.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let normals: [Float] = try XCTUnwrap(mesh.normals).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            var turned = releasedPositions, turnedNormals = releasedNormals
            for vertex in 0 ..< releasedPositions.count / 3 {
                turned[vertex * 3] = -releasedPositions[vertex * 3]; turned[vertex * 3 + 2] = -releasedPositions[vertex * 3 + 2]
                turnedNormals[vertex * 3] = -releasedNormals[vertex * 3]; turnedNormals[vertex * 3 + 2] = -releasedNormals[vertex * 3 + 2]
            }
            positionDifference = maxDifference(ours, turned)
            normalCosine = cosine(normals, turnedNormals)
            let faces: [UInt32] = mesh.indices.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
            facesEqual = faces.map { Int64($0) } == releasedFaces
        }
        // A dithered texel lies within one 8-bit level of floor(256 · x), and the release dithers the
        // whole base color and the flat texels of the normal map.
        func levelsApart(_ texture: NFKMeshTexture?, _ key: String) throws -> Int {
            let data = try XCTUnwrap(texture?.data)
            let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
            let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
            // The image's own color space, so the read converts nothing.
            let bytes = NFKMLXImageBridge.rgbaBytes(from: image, colorSpace: image.colorSpace ?? CGColorSpaceCreateDeviceRGB()).bytes
            let reference = try floats(key)
            var apart = 0
            for texel in 0 ..< reference.count / 3 {
                for channel in 0 ..< 3 {
                    let expected = Int(min(max((256 * reference[texel * 3 + channel]).rounded(.down), 0), 255))
                    if abs(Int(bytes[texel * 4 + channel]) - expected) > 1 { apart += 1 }
                }
            }
            return apart
        }
        let material = try XCTUnwrap(mesh.material)
        let albedoApart = try levelsApart(material.baseColorTexture, "albedo_padded")
        let normalApart = try levelsApart(material.normalTexture, "bump_padded")
        print("Stable Fast 3D end to end: vertices \(mesh.vertexCount) against \(releasedPositions.count / 3), faces equal \(facesEqual), "
              + "positions \(positionDifference), normals \(normalCosine), base color channels more than one level apart \(albedoApart), "
              + "normal-map channels \(normalApart), roughness \(material.roughnessFactor), metallic \(material.metallicFactor)")
        // The reference arithmetic reproduces the release's scene code bit for bit, and the surface stages
        // follow it, so the mesh is the release's own.
        XCTAssertEqual(Int(mesh.vertexCount) * 3, releasedPositions.count, "the export welds the release's vertices")
        XCTAssertTrue(facesEqual)
        XCTAssertEqual(positionDifference, 0)
        XCTAssertGreaterThan(normalCosine, 0.9999)
        XCTAssertEqual(albedoApart, 0)
        XCTAssertEqual(normalApart, 0)
    }

    func testTheReferenceArithmeticReproducesTheNetworkBitForBit() throws {
        let directory = try envPath("IK_VAL_SF3D")
        let record = try loadArrays(url: try envPath("IK_PARITY_SF3D"))
        func floats(_ key: String) throws -> [Float] { try XCTUnwrap(record[key], key).asType(.float32).reshaped([-1]).asArray(Float.self) }
        let configuration = NFKMLXStableFast3DConfiguration.released
        let network = try NFKMLXStableFast3DReferenceNetwork(weightsURL: directory.appendingPathComponent(NFKMLXStableFast3D.weightsFileName))
        var mismatches = [String: Int]()

        // Each seam from the reference's own input.
        let (rgba, side) = try recordRGBA(record)
        let (color, mask) = NFKMLXStableFast3D.referenceConditioning(rgba: rgba, width: side, height: side, side: configuration.imageSize,
                                                                      foregroundRatio: 0.85, background: configuration.background)
        mismatches["conditioning"] = bitMismatches(color, try floats("conditioning"))
        mismatches["mask"] = bitMismatches(mask, try floats("conditioning_mask"))
        mismatches["camera"] = bitMismatches(try network.cameraEmbedding(condition: configuration.cameraCondition), try floats("camera_embedding"))
        mismatches["positions"] = bitMismatches(try network.resizedPositions(), try floats("position_embeddings"))
        mismatches["imageTokens"] = bitMismatches(try network.imageTokens(try floats("conditioning"), camera: try floats("camera_embedding")),
                                                  try floats("image_tokens"))
        let referenceTokens = try floats("triplane_tokens")                                     // [count, width]
        let tokenCount = referenceTokens.count / configuration.tokenWidth
        mismatches["triplaneTokens"] = bitMismatches(transposed(try network.triplaneTokens(try floats("image_tokens")),
                                                                rows: configuration.tokenWidth), referenceTokens)
        let planeSide = configuration.planeSize * configuration.upsampleFactor, planeArea = planeSide * planeSide
        let referenceCode = try floats("output")                                                // [3, side, side, channels]
        var code = [Float]()                                                                    // [3, channels, side, side]
        for plane in 0 ..< 3 {
            code += transposed(Array(referenceCode[(plane * planeArea * configuration.planeChannels) ..< ((plane + 1) * planeArea * configuration.planeChannels)]),
                               rows: planeArea)
        }
        mismatches["sceneCode"] = bitMismatches(try network.sceneCode(transposed(referenceTokens, rows: tokenCount)), code)
        let grid = try NFKMLXTetrahedralGrid(contentsOf: directory.appendingPathComponent(NFKMLXStableFast3D.gridFileName))
        let span = configuration.radius - (-configuration.radius)
        let decoded = try network.geometry(grid.vertices.map { $0 * span + (-configuration.radius) }, sceneCode: code)
        mismatches["gridDensity"] = bitMismatches(decoded.density, try floats("grid_density"))
        mismatches["gridOffsets"] = bitMismatches(decoded.vertexOffset, try floats("grid_vertex_offset"))
        let surface = try network.surface(try floats("baked_positions"), sceneCode: code)
        mismatches["albedoTexels"] = bitMismatches(surface.albedo, try floats("albedo_texels"))
        mismatches["perturbedNormals"] = bitMismatches(surface.normal, try floats("perturbed_normals"))
        print("Stable Fast 3D reference arithmetic: "
              + mismatches.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value) mismatches" }.joined(separator: ", "))
        for (seam, count) in mismatches.sorted(by: { $0.key < $1.key }) {
            XCTAssertEqual(count, 0, seam)
        }
    }

    func testTheLadderHoldsTheReleaseNearTheRecord() throws {
        let directory = try envPath("IK_VAL_SF3D")
        let record = try loadArrays(url: try envPath("IK_PARITY_SF3D"))
        let weights = directory.appendingPathComponent(NFKMLXStableFast3D.weightsFileName)
        let conditioning = try XCTUnwrap(record["conditioning"]).reshaped([1, 512, 512, 3])
        let reference = try XCTUnwrap(record["output"])
        func codes(_ net: NFKMLXStableFast3DNet, _ images: MLXArray) -> MLXArray {
            let result = net(images)[0]
            eval(result)
            NFKMLXGPU.clearCache()
            return result
        }
        let half = codes(try NFKMLXStableFast3D.network(weightsURL: weights, castingTo: .bfloat16), conditioning.asType(.bfloat16))
        let halfCosine = cosine(half, reference)
        let packed = codes(try NFKMLXStableFast3D.network(weightsURL: weights, quantizationBits: 4), conditioning)
        let packedCosine = cosine(packed, reference)
        print("Stable Fast 3D ladder: bfloat16 scene codes \(halfCosine), 4-bit scene codes \(packedCosine)")
        XCTAssertGreaterThan(halfCosine, 0.99)
        XCTAssertGreaterThan(packedCosine, 0.99)
    }

    func testTheReleaseMatchesTheReferenceSeams() throws {
        let directory = try envPath("IK_VAL_SF3D")
        let record = try loadArrays(url: try envPath("IK_PARITY_SF3D"))
        func array(_ key: String) throws -> MLXArray { try XCTUnwrap(record[key], key) }
        func floats(_ key: String) throws -> [Float] { try array(key).asType(.float32).reshaped([-1]).asArray(Float.self) }
        func integers(_ key: String) throws -> [Int] { try array(key).asType(.int64).reshaped([-1]).asArray(Int64.self).map(Int.init) }
        let net = try NFKMLXStableFast3D.network(weightsURL: directory.appendingPathComponent(NFKMLXStableFast3D.weightsFileName))
        let grid = try NFKMLXTetrahedralGrid(contentsOf: directory.appendingPathComponent(NFKMLXStableFast3D.gridFileName))
        var figures = [String: Double]()

        // Preparation: the plate and its alpha as the oracle writes them (`astype(np.uint8)` truncates).
        let plate = try floats("input_image"), alpha = try floats("input_alpha")
        let side = Int(Double(alpha.count).squareRoot())
        var rgba = [UInt8](repeating: 0, count: side * side * 4)
        for pixel in 0 ..< side * side {
            for channel in 0 ..< 3 { rgba[pixel * 4 + channel] = UInt8(plate[pixel * 3 + channel] * 255) }
            rgba[pixel * 4 + 3] = UInt8(alpha[pixel] * 255)
        }
        let framed = NFKMLXStableFast3D.frame(rgba: rgba, width: side, height: side, foregroundRatio: 0.85)
        let framedReference = try floats("framed")
        XCTAssertEqual(framed.pixels.count, framedReference.count)
        let framedMismatch = zip(framed.pixels, framedReference).filter { Float($0.0) != ($0.1 * 255).rounded() }.count
        let resized = NFKMLXStableFast3D.pilResized(framed.pixels, width: framed.width, height: framed.height, to: 512)
        let resizedReference = try floats("resized")
        let resizedMismatch = zip(resized, resizedReference).filter { Float($0.0) != ($0.1 * 255).rounded() }.count
        let (color, mask) = NFKMLXStableFast3D.conditioning(rgba: rgba, width: side, height: side, side: 512, foregroundRatio: 0.85,
                                                             background: 0.5)
        figures["conditioning"] = cosine(color, try array("conditioning"))
        figures["mask"] = cosine(mask, try array("conditioning_mask"))

        // The network, each seam from the reference's own input.
        figures["camera"] = cosine(net.cameraEmbedding(), try array("camera_embedding"))
        let positions = try XCTUnwrap(net.parameters().flattened().first { $0.0 == "image_tokenizer.model.embeddings.position_embeddings" }?.1)
        figures["positions"] = cosine(positions[0], try array("position_embeddings"))
        let conditioning = try array("conditioning").reshaped([1, 512, 512, 3])
        let tokens = net.imageTokens(conditioning)
        eval(tokens)
        figures["imageTokens"] = cosine(tokens[0], try array("image_tokens"))
        let triplane = net.triplaneTokens(try array("image_tokens").reshaped([1, -1, net.configuration.dinoWidth]))
        eval(triplane)
        figures["triplaneTokens"] = cosine(triplane[0], try array("triplane_tokens"))
        let referenceTriplane = try array("triplane_tokens").reshaped([1, -1, net.configuration.tokenWidth])
        let codes = net.sceneCodes(referenceTriplane)
        eval(codes)
        figures["sceneCodes"] = cosine(codes[0], try array("output"))
        let conditioningMask = try array("conditioning_mask").reshaped([1, 512, 512, 1])
        let masked = conditioning * conditioningMask
        let materials = net.materials(masked)
        eval(materials.roughness, materials.metallic)
        let roughness = materials.roughness[0].item(Float.self), metallic = materials.metallic[0].item(Float.self)
        let referenceRoughness = try floats("roughness")[0], referenceMetallic = try floats("metallic")[0]
        figures["roughness"] = Double(abs(roughness - referenceRoughness))
        figures["metallic"] = Double(abs(metallic - referenceMetallic))
        let amplitudes = net.illumination(referenceTriplane)
        eval(amplitudes)
        figures["illumination"] = cosine(amplitudes, try array("sg_amplitudes"))

        // The decoder on the grid, from the reference's scene code.
        let sceneCode = try array("output")
        let span: Float = 0.87 - -0.87
        let gridPoints: [Float] = grid.vertices.map { $0 * span + -0.87 }
        let points = MLXArray(gridPoints).reshaped([grid.vertexCount, 3])
        var density = [Float](), offsets = [Float]()
        var start = 0
        while start < grid.vertexCount {
            let end = min(start + (1 << 17), grid.vertexCount)
            let (d, o) = net.geometry(points[start ..< end], sceneCode: sceneCode)
            eval(d, o)
            density += d.asArray(Float.self); offsets += o.asArray(Float.self)
            start = end
        }
        figures["gridDensity"] = cosine(density, try floats("grid_density"))
        figures["gridOffsets"] = cosine(offsets, try floats("grid_vertex_offset"))

        // The surface from the reference's grid values.
        let surface = NFKMLXStableFast3DGeometry.surface(grid: grid, density: try floats("grid_density"),
                                                         vertexOffsets: try floats("grid_vertex_offset"), threshold: 10,
                                                         resolution: 160, radius: 0.87)
        let referenceFaces = try integers("extracted_faces").map(UInt32.init)
        XCTAssertEqual(surface.faces, referenceFaces, "the extractor draws the reference's triangles in its order")
        let referencePositions = try floats("extracted_positions")
        figures["surfacePositions"] = Double(maxDifference(surface.positions, referencePositions))
        let smooth = NFKSF3DVertexNormals(positions: referencePositions, faces: referenceFaces)
        figures["smoothNormals"] = Double(bitMismatches(smooth, try floats("smooth_normals")))

        // The unwrapper, stage by stage from the reference's own inputs.
        let referenceSmooth = try floats("smooth_normals")
        let aligned = try NFKSF3DUnwrapper.alignedToPrincipalAxes(positions: referencePositions, normals: referenceSmooth)
        figures["aligned"] = Double(maxDifference(aligned.positions, try floats("aligned_positions")))
        let referenceAligned = try floats("aligned_positions")
        let box = NFKSF3DUnwrapper.boxProjection(positions: referenceAligned, normals: aligned.normals, faces: referenceFaces)
        let boxIndexMismatch = zip(box.index, try integers("box_index")).filter { $0.0 != $0.1 }.count
        figures["boxUV"] = Double(maxDifference(box.uv, try floats("box_uv")))
        let rotated = try NFKSF3DUnwrapper.rotatedSlices(positions: referenceAligned, normals: aligned.normals, faces: referenceFaces,
                                                     uv: try floats("box_uv"), index: try integers("box_index"))
        figures["rotatedUV"] = Double(maxDifference(rotated, try floats("rotated_uv")))
        let atlas = NFKSF3DUnwrapper.atlasIndices(positions: referenceAligned, faces: referenceFaces, uv: try floats("rotated_uv"),
                                                  index: try integers("box_index"))
        let atlasMismatch = zip(atlas, try integers("atlas_index")).filter { $0.0 != $0.1 }.count
        let placed = NFKSF3DUnwrapper.placed(uv: try floats("rotated_uv"), atlas: try integers("atlas_index"), padding: 0.02)
        figures["placedUV"] = Double(maxDifference(placed, try floats("placed_uv")))

        // The split mesh's normals and tangents, from the reference's surface and atlas.
        let split = NFKMLXStableFast3DGeometry.splitMesh(positions: referencePositions, faces: referenceFaces,
                                                         cornerUV: try floats("texture_coordinates"))
        figures["splitNormals"] = Double(bitMismatches(split.normals, try floats("split_normals")))
        figures["splitTangents"] = Double(bitMismatches(split.tangents, try floats("split_tangents")))

        // The bake, from the reference's atlas and split mesh.
        let referenceSplit = try floats("split_positions")
        let raster = NFKSF3DBaker.rasterize(uv: try floats("texture_coordinates"), faces: split.faces, side: 1024)
        let referenceRaster = try floats("rasterized")
        var coverageMismatch = 0, triangleMismatch = 0
        for texel in 0 ..< 1024 * 1024 {
            let ours = raster.triangle[texel], theirs = Int32(referenceRaster[texel * 4 + 3])
            if (ours >= 0) != (theirs >= 0) { coverageMismatch += 1 } else if ours != theirs { triangleMismatch += 1 }
        }
        let baked = NFKSF3DBaker.interpolate(referenceSplit, faces: split.faces, weights: raster.weights, triangle: raster.triangle)
        if coverageMismatch == 0 {
            figures["bakedPositions"] = Double(maxDifference(baked, try floats("baked_positions")))
        }
        let bakedReference = try array("baked_positions").reshaped([-1, 3])
        let heads = net.surface(bakedReference, sceneCode: sceneCode)
        eval(heads.albedo, heads.normal)
        figures["albedoTexels"] = cosine(heads.albedo, try array("albedo_texels"))
        figures["perturbedNormals"] = cosine(heads.normal, try array("perturbed_normals"))
        var referenceMask = [Bool](repeating: false, count: 1024 * 1024)
        for texel in referenceMask.indices {
            referenceMask[texel] = referenceRaster[texel * 4 + 3] >= 0
        }
        let bakedNormals = NFKSF3DBaker.interpolate(try floats("split_normals"), faces: split.faces, weights: raster.weights, triangle: raster.triangle)
        let bakedTangents = NFKSF3DBaker.interpolate(try floats("split_tangents"), faces: split.faces, weights: raster.weights, triangle: raster.triangle)
        let tangentNormals = NFKMLXStableFast3DGeometry.tangentSpaceNormals(perturbed: try floats("perturbed_normals"), normals: bakedNormals,
                                                                            tangents: bakedTangents)
        figures["tangentNormals"] = Double(bitMismatches(tangentNormals, try floats("tangent_normals")))
        let albedoPadded = NFKSF3DBaker.dilated(try floats("albedo"), mask: referenceMask, side: 1024, iterations: 1024 / 150)
        figures["albedoPadded"] = Double(maxDifference(albedoPadded, try floats("albedo_padded")))

        // The export, from the reference's split mesh: the welds and normals of the release's trimesh, in
        // glTF's frame. The release's export turns (x, y, z) into (−y, z, −x); this port's into (y, z, x).
        let material = try NFKMeshMaterial(baseColorTexture: nil, normalTexture: nil, metallicRoughnessTexture: nil,
                                           baseColorFactor: nil, metallicFactor: 0, roughnessFactor: 1)
        let exported = try NFKMLXStableFast3DGeometry.exported(
            split: .init(positions: referenceSplit, faces: split.faces, uv: try floats("texture_coordinates"),
                         normals: split.normals, tangents: split.tangents), material: material)
        let releasedPositions = try floats("released_positions"), releasedNormals = try floats("released_normals")
        XCTAssertEqual(Int(exported.vertexCount), releasedPositions.count / 3, "the export welds the release's vertices")
        let exportedFaces: [UInt32] = exported.indices.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        let releasedFaces = try integers("released_faces")
        XCTAssertEqual(exportedFaces.map { Int($0) }, releasedFaces)
        let ours: [Float] = exported.positions.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let ourNormals: [Float] = try XCTUnwrap(exported.normals).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        func released(_ values: [Float]) -> [Float] {
            var out = values
            for vertex in 0 ..< values.count / 3 {
                out[vertex * 3] = -values[vertex * 3]
                out[vertex * 3 + 2] = -values[vertex * 3 + 2]
            }
            return out
        }
        figures["exportedPositions"] = Double(maxDifference(ours, released(releasedPositions)))
        figures["exportedNormals"] = cosine(ourNormals, released(releasedNormals))

        print("Stable Fast 3D parity: framed mismatches \(framedMismatch), resized mismatches \(resizedMismatch), "
              + "box index mismatches \(boxIndexMismatch), atlas mismatches \(atlasMismatch), "
              + "raster coverage mismatches \(coverageMismatch), triangle mismatches \(triangleMismatch), "
              + figures.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))
        XCTAssertEqual(framedMismatch, 0)
        XCTAssertEqual(resizedMismatch, 0)
        XCTAssertEqual(boxIndexMismatch, 0)
        XCTAssertEqual(atlasMismatch, 0)
        XCTAssertEqual(coverageMismatch, 0)
        XCTAssertEqual(triangleMismatch, 0)
        // The surface and texture stages read the reference's own float32 inputs and run its float32
        // arithmetic, so each matches bit for bit (differences and mismatch counts of zero). The MLX network
        // runs on the GPU and is held by cosine.
        for key in ["boxUV", "placedUV", "surfacePositions", "aligned", "rotatedUV", "smoothNormals", "splitNormals", "splitTangents",
                    "bakedPositions", "tangentNormals", "albedoPadded", "exportedPositions"] {
            XCTAssertEqual(figures[key] ?? 1, 0, key)
        }
        XCTAssertLessThan(figures["roughness"] ?? 1, 1e-5)
        XCTAssertEqual(figures["metallic"] ?? 1, 0)
        for key in ["conditioning", "mask", "camera", "positions", "imageTokens", "triplaneTokens", "sceneCodes", "illumination",
                    "gridDensity", "gridOffsets", "albedoTexels", "perturbedNormals", "exportedNormals"] {
            XCTAssertGreaterThan(figures[key] ?? 0, 0.9999, key)
        }
    }
}
