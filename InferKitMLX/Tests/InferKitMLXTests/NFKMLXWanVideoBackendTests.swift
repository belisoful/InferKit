//
//  NFKMLXWanVideoBackendTests.swift
//  InferKitMLXTests
//
//  Wan behind the InferKit contract: the request's prompt and parameters reach the generator, the clip
//  comes back as a movie under NFKOutputVideo, and a request's settings do not outlive it.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXWanVideoBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyGenerator(vae: NFKMLXWanVAEConfiguration = .tiny21) throws -> NFKMLXWanVideoGenerator {
        let umt5 = NFKMLXT5Configuration(dModel: 32, layers: 2, heads: 2, keyDim: 16, ffDim: 64,
                                         vocabularySize: 32128, relativeBuckets: 16, relativeMaxDistance: 32,
                                         perLayerBias: true)
        let transformer = NFKMLXWanConfiguration(inChannels: 4, heads: 2, headDim: 16, layers: 2, ffnDim: 48,
                                                 textDim: 32, patchSize: [1, 2, 2])
        MLXRandom.seed(7)
        let generator = NFKMLXWanVideoGenerator(
            resident: true, segmenter: try NFKMLXGeneratorBackendTestSupport.t5Segmenter(),
            compression: NFKMLXWanVideoGenerator.compression(of: vae),
            framesPerSecond: NFKMLXWanVideoGenerator.framesPerSecond(of: vae), sequenceLength: 16,
            loadTextEncoder: { NFKMLXT5Encoder.makeNet(umt5) },
            loadPipeline: {
                NFKMLXWanPipeline(transformer: NFKMLXWanTransformerNet(transformer), vae: NFKMLXWanVideoVAENet(vae),
                                  latentsMean: MLXArray([Float(0.1), -0.2, 0.3, -0.4]),
                                  latentsStd: MLXArray([Float(1.1), 0.9, 1.3, 0.8]))
            })
        try generator.loadResident()
        return generator
    }

    func testARequestBecomesAMovieAtTheRequestedSize() throws {
        try requireMLXRuntime()
        let generator = try tinyGenerator()
        let backend = generator.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXWanVideoGenerator.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputNegativePrompt])

        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox", NFKInputNegativePrompt: "a fox"],
                                          parameters: [NFKParameterFrameCount: 5, NFKParameterWidth: 16,
                                                       NFKParameterHeight: 16, NFKParameterSteps: 1,
                                                       NFKParameterGuidanceScale: 3,
                                                       NFKParameterFramesPerSecond: 12, NFKParameterSeed: 3])
        let result = try backend.runInference(for: request)
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let url = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(asset.framesPerSecond, 12)
        XCTAssertEqual(asset.dimensions, CGSize(width: 16, height: 16))
        XCTAssertEqual(try NFKMLXVideoFile.read(url).frames.count, 5)
        XCTAssertEqual(generator.steps, 50, "the request's steps apply to that request only")
        XCTAssertEqual(generator.guidance, 5)
    }

    private func framesPerSecondWithoutARequestedRate(_ generator: NFKMLXWanVideoGenerator) throws -> Double {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox"],
                                          parameters: [NFKParameterFrameCount: 1, NFKParameterWidth: 16,
                                                       NFKParameterHeight: 16, NFKParameterSteps: 1])
        let result = try generator.inferenceBackend().runInference(for: request)
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        defer { asset.fileURL.map { try? FileManager.default.removeItem(at: $0) } }
        return asset.framesPerSecond
    }

    func testTheReleaseAutoencoderSetsTheTrainedFrameRate() throws {
        try requireMLXRuntime()
        XCTAssertEqual(NFKMLXWanVideoGenerator.framesPerSecond(of: .wan21), 16)
        XCTAssertEqual(NFKMLXWanVideoGenerator.framesPerSecond(of: .wan22), 24)
        XCTAssertEqual(NFKMLXWanVideoGenerator.framesPerSecond(of: .tiny21), 16)
        XCTAssertEqual(NFKMLXWanVideoGenerator.framesPerSecond(of: .tiny), 24)
    }

    func testAWan21RequestWithoutAFrameRateIsWrittenAtSixteenFramesPerSecond() throws {
        try requireMLXRuntime()
        let generator = try tinyGenerator(vae: .tiny21)
        XCTAssertEqual(generator.framesPerSecond, 16)
        XCTAssertEqual(try framesPerSecondWithoutARequestedRate(generator), 16)
    }

    func testATI2VRequestWithoutAFrameRateIsWrittenAtTwentyFourFramesPerSecond() throws {
        try requireMLXRuntime()
        let generator = try tinyGenerator(vae: .tiny)
        XCTAssertEqual(generator.framesPerSecond, 24)
        XCTAssertEqual(try framesPerSecondWithoutARequestedRate(generator), 24)
    }

    func testARequestWithoutAPromptIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyGenerator().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }
}
