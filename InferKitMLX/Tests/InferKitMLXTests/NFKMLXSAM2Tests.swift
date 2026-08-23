//
//  NFKMLXSAM2Tests.swift
//  InferKitMLXTests
//
//  SAM 2's Hiera encoder, mask decoder, and video memory path at sizes a test can run, plus the pure
//  key remaps and the bicubic resampler. The parity records prove the numbers; these prove the
//  geometry and the remaps with no external files, so a clean checkout still exercises the module.
//  The MLX forwards skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSAM2Tests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    // MARK: Remaps (pure)

    func testTheRemapTranslatesTheEncoderKeys() {
        XCTAssertEqual(NFKMLXSAM2.remapReferenceKey("image_encoder.trunk.patch_embed.proj.weight"),
                       "trunk.patch_embed.weight")
        XCTAssertEqual(NFKMLXSAM2.remapReferenceKey("image_encoder.neck.convs.2.conv.weight"),
                       "neck.convs.2.weight")
        XCTAssertEqual(NFKMLXSAM2.remapReferenceKey("image_encoder.trunk.blocks.10.attn.qkv.weight"),
                       "trunk.blocks.10.attn.qkv.weight")
        XCTAssertNil(NFKMLXSAM2.remapReferenceKey("sam_mask_decoder.iou_token.weight"),
                     "everything outside the image encoder is another module's")
    }

    func testTheRemapTranslatesTheDecoderKeys() {
        XCTAssertEqual(NFKMLXSAM2.remapDecoderKey("sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.q_proj.weight"),
                       "transformer_layers.1.cross_token_image.q_proj.weight")
        XCTAssertEqual(NFKMLXSAM2.remapDecoderKey("sam_mask_decoder.output_upscaling.3.bias"),
                       "upscale2.bias")
        XCTAssertEqual(NFKMLXSAM2.remapDecoderKey("sam_mask_decoder.obj_score_token.weight"),
                       "obj_score_token")
        XCTAssertEqual(NFKMLXSAM2.remapDecoderKey("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"),
                       "position_encoding.gaussian")
        XCTAssertNil(NFKMLXSAM2.remapDecoderKey("sam_prompt_encoder.mask_downscaling.0.weight"),
                     "the mask-prompt downscaler is not part of the point-prompt path")
        XCTAssertNil(NFKMLXSAM2.remapDecoderKey("memory_encoder.out_proj.weight"))
    }

    func testTheRemapTranslatesTheMemoryKeys() {
        XCTAssertEqual(NFKMLXSAM2.remapMemoryKey("memory_attention.layers.3.cross_attn_image.k_proj.weight"),
                       "layers.3.cross_attn_image.k_proj.weight")
        // The downsampler's Sequential runs convolution, normalization, activation per stage, so the
        // parameterized slots land at 0/1, 3/4, 6/7, 9/10, with the projection at 12.
        XCTAssertEqual(NFKMLXSAM2.remapMemoryKey("memory_encoder.mask_downsampler.encoder.6.weight"),
                       "mask_convs.2.weight")
        XCTAssertEqual(NFKMLXSAM2.remapMemoryKey("memory_encoder.mask_downsampler.encoder.10.bias"),
                       "mask_norms.3.bias")
        XCTAssertEqual(NFKMLXSAM2.remapMemoryKey("memory_encoder.mask_downsampler.encoder.12.weight"),
                       "mask_out.weight")
        XCTAssertEqual(NFKMLXSAM2.remapMemoryKey("memory_encoder.fuser.layers.1.gamma"),
                       "fuser.1.gamma")
        XCTAssertNil(NFKMLXSAM2.remapMemoryKey("image_encoder.trunk.patch_embed.proj.weight"))
    }

    // MARK: Bicubic (needs MLX)

    func testBicubicKeepsAConstantGridConstant() throws {
        try requireMLXRuntime()
        // The Keys kernel's four taps sum to one, so a constant field resizes to itself; a sign or
        // normalization slip in the weights shows up here immediately.
        let constant = MLXArray.ones([1, 5, 5, 3]) * 0.4
        let resized = NFKMLXBicubic.resize(constant, height: 12, width: 9)
        eval(resized)
        XCTAssertEqual(resized.shape, [1, 12, 9, 3])
        for value in resized.reshaped([-1]).asArray(Float.self) {
            XCTAssertEqual(value, 0.4, accuracy: 1e-5)
        }
    }

    func testBicubicAtTheSameSizeIsTheIdentity() throws {
        try requireMLXRuntime()
        // With half-pixel centers, every output center lands exactly on an input sample.
        let values = (0 ..< 32).map { Float($0) / 31 }
        let grid = values.withUnsafeBufferPointer { MLXArray($0, [1, 4, 8, 1]) }
        let resized = NFKMLXBicubic.resize(grid, height: 4, width: 8)
        eval(resized)
        for (a, b) in zip(grid.reshaped([-1]).asArray(Float.self),
                          resized.reshaped([-1]).asArray(Float.self)) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
    }

    // MARK: Geometry (needs MLX)

    /// The released structure at a size a test can run: four one-block stages over a narrow width,
    /// every window equal so the stage-transition window rule has nothing to hide behind.
    static func tinyConfiguration() -> NFKMLXSAM2Configuration {
        NFKMLXSAM2Configuration(embedDimensions: 8, heads: 1, stages: [1, 1, 1, 1],
                                windowSpec: [4, 4, 4, 4], globalAttentionBlocks: [3],
                                backgroundWindow: 4, neckChannels: 16)
    }

    func testEncoderEmitsFPNLevelsFinestFirstAndScalpsTheCoarsest() throws {
        try requireMLXRuntime()
        let encoder = NFKMLXSAM2EncoderNet(Self.tinyConfiguration(), scalp: 1)
        let image = MLXRandom.uniform(low: 0, high: 1, [1, 64, 64, 3])
        let levels = encoder.features(image)
        eval(levels)
        // Patch stride 4, halving between stages: grids 16/8/4/2, the coarsest dropped by the scalp.
        XCTAssertEqual(levels.map(\.shape), [[1, 16, 16, 16], [1, 8, 8, 16], [1, 4, 4, 16]])
        XCTAssertEqual(encoder.visionFeatures(image).shape, [1, 4, 4, 16],
                       "the vision features are the last kept level")
    }

    func testDecoderPredictsMasksAtFourTimesTheFeatureGrid() throws {
        try requireMLXRuntime()
        let dimensions = 32
        let decoder = NFKMLXSAM2Decoder(dimensions: dimensions, heads: 2, maskCount: 4, depth: 2)
        decoder.train(false)
        let features = MLXRandom.uniform(low: -1, high: 1, [1, 8, 8, dimensions])
        let (masks, iou, objectScore) = decoder(
            features: features,
            positional: MLXRandom.uniform(low: -1, high: 1, [1, 64, dimensions]),
            sparse: MLXArray.zeros([1, 2, dimensions]),
            dense: MLXArray.zeros([1, 8, 8, dimensions]),
            highResolution: [MLXRandom.uniform(low: -1, high: 1, [1, 32, 32, dimensions]),
                             MLXRandom.uniform(low: -1, high: 1, [1, 16, 16, dimensions])])
        eval(masks, iou, objectScore)
        XCTAssertEqual(masks.shape, [1, 4, 32, 32], "two ×2 upscales over the 8×8 grid")
        XCTAssertEqual(iou.shape, [1, 4])
        XCTAssertEqual(objectScore.shape, [1, 1])
    }

    func testMemoryEncoderFoldsAFullResolutionMaskOntoTheFeatureGrid() throws {
        try requireMLXRuntime()
        let encoder = NFKMLXSAM2MemoryEncoderNet(dimensions: 32, outDimensions: 8, fuserDepth: 1)
        encoder.train(false)
        let features = MLXRandom.uniform(low: -1, high: 1, [1, 4, 4, 32])
        // The mask arrives at the FULL frame resolution: the downsampler's total stride of 16 is what
        // lands it on the feature grid, so a 4×4 grid takes a 64×64 mask.
        let mask = MLXRandom.uniform(low: -4, high: 4, [1, 64, 64, 1])
        let memory = encoder(features: features, maskLogits: mask)
        eval(memory)
        XCTAssertEqual(memory.shape, [1, 4, 4, 8])
    }

    func testMemoryAttentionConditionsTheCurrentFrameOnTheMemory() throws {
        try requireMLXRuntime()
        let attention = NFKMLXSAM2MemoryAttentionNet(dimensions: 32, depth: 1)
        attention.train(false)
        // 16 tokens = a 4×4 grid, which is what the axial rotary embedding expects to factor.
        let current = MLXRandom.uniform(low: -1, high: 1, [1, 16, 32])
        let memory = MLXRandom.uniform(low: -1, high: 1, [1, 16, 64])
        let out = attention(current: current, memory: memory,
                            currentPosition: MLXRandom.uniform(low: -1, high: 1, [1, 16, 32]),
                            memoryPosition: MLXRandom.uniform(low: -1, high: 1, [1, 16, 64]))
        eval(out)
        XCTAssertEqual(out.shape, current.shape)
        XCTAssertNotEqual(out.reshaped([-1]).asArray(Float.self),
                          current.reshaped([-1]).asArray(Float.self),
                          "the memory changes the frame's tokens")
    }

    // MARK: Round trip (needs MLX)

    func testACheckpointRoundTripReproducesTheEncoderFeatures() throws {
        try requireMLXRuntime()
        let trained = NFKMLXSAM2EncoderNet(Self.tinyConfiguration(), scalp: 1)
        // Write the reference's own layout: an `image_encoder.` prefix, the two Sequential wrappers
        // the remap unwraps, and 4-D tensors rotated back to `[out, in, kH, kW]`.
        let referenceLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened()
            .map { key, value -> (String, MLXArray) in
                var name = key
                name = name.replacingOccurrences(of: "trunk.patch_embed.", with: "trunk.patch_embed.proj.")
                if name.hasPrefix("neck.convs.") {
                    name = name.replacingOccurrences(of: ".weight", with: ".conv.weight")
                    name = name.replacingOccurrences(of: ".bias", with: ".conv.bias")
                }
                let tensor = value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value
                return ("image_encoder." + name, tensor)
            })
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sam2-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: referenceLayout, url: url)

        let loaded = NFKMLXSAM2EncoderNet(Self.tinyConfiguration(), scalp: 1)
        try NFKMLXSAM2.loadWeights(into: loaded, from: url)

        let image = MLXRandom.uniform(low: 0, high: 1, [1, 64, 64, 3], key: MLXRandom.key(3))
        let expected = trained.visionFeatures(image)
        let actual = loaded.visionFeatures(image)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }
}
