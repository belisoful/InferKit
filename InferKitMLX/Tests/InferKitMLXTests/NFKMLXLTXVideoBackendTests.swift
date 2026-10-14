//
//  NFKMLXLTXVideoBackendTests.swift
//  InferKitMLXTests
//
//  LTX-Video behind the InferKit contract: the request's prompt and parameters reach the generator, the
//  clip comes back as a movie under NFKOutputVideo, and a request's settings do not outlive it.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLTXVideoBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyGenerator() throws -> NFKMLXLTXVideoGenerator {
        guard let tokenizer = NFKMLXValidationConfig.environment["IK_VAL_T5_TOKENIZER"] else {
            throw XCTSkip("set IK_VAL_T5_TOKENIZER (a T5 SentencePiece tokenizer directory)")
        }
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: URL(fileURLWithPath: tokenizer).appendingPathComponent("spiece.model")))
        let t5 = NFKMLXT5Configuration(dModel: 32, layers: 2, heads: 2, keyDim: 16, ffDim: 64,
                                       vocabularySize: 32128, relativeBuckets: 16, relativeMaxDistance: 32)
        let transformer = NFKMLXLTXTransformerConfiguration(
            inChannels: 16, heads: 2, headDim: 8, layers: 2, crossAttentionDim: 16, captionChannels: 32)
        let vae = NFKMLXLTXVAEConfiguration(latentChannels: 16, blockOutChannels: [8, 16, 16, 16],
                                            layersPerBlock: [1, 1, 1, 1, 1])
        MLXRandom.seed(5)
        let generator = NFKMLXLTXVideoGenerator(
            resident: true, segmenter: segmenter,
            loadTextEncoder: { NFKMLXT5Encoder.makeNet(t5) },
            loadPipeline: {
                NFKMLXLTXPipeline(transformer: NFKMLXLTXTransformer.makeNet(transformer),
                                  vae: NFKMLXLTXVideoVAE.makeNet(vae))
            })
        try generator.loadResident()
        return generator
    }

    func testARequestBecomesAMovieAtTheRequestedSize() throws {
        try requireMLXRuntime()
        let generator = try tinyGenerator()
        let backend = generator.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXLTXVideoGenerator.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputNegativePrompt])

        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox"],
                                          parameters: [NFKParameterFrameCount: 9, NFKParameterWidth: 64,
                                                       NFKParameterHeight: 64, NFKParameterSteps: 1,
                                                       NFKParameterFramesPerSecond: 12, NFKParameterSeed: 3])
        let result = try backend.runInference(for: request)
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let url = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(asset.framesPerSecond, 12)
        XCTAssertEqual(asset.dimensions, CGSize(width: 64, height: 64))
        XCTAssertEqual(try NFKMLXVideoFile.read(url).frames.count, 9)
        XCTAssertEqual(generator.steps, 50, "the request's steps apply to that request only")
        XCTAssertEqual(generator.frameRate, 25)
    }

    func testARequestWithoutAPromptIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyGenerator().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }
}
