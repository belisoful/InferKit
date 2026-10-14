//
//  NFKMLXChatterboxParameterKeyTests.swift
//  InferKitMLXTests
//
//  The Chatterbox options a request sets: the exaggeration key replaces the voice's for that request
//  only, the core sampling keys reach T3's options, and the speech backend hands its closure the request
//  and declares the keys the closure reads.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXChatterboxParameterKeyTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func voice(exaggeration: Float) -> NFKMLXChatterboxConditionals {
        NFKMLXChatterboxConditionals(
            t3: NFKMLXT3Condition(speakerEmbedding: MLXArray.zeros([4]), promptTokens: [1, 2, 3],
                                  exaggeration: exaggeration),
            s3gen: NFKMLXS3GenPrompt(tokens: [4, 5], mel: MLXArray.zeros([2, 3]), xVector: MLXArray.zeros([4])))
    }

    func testTheExaggerationKeyReplacesTheVoicesForTheRequest() throws {
        try requireMLXRuntime()
        let base = voice(exaggeration: 0.5)
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"],
                                          parameters: [NFKMLXChatterboxParameterKey.exaggeration: 0.9])
        let adjusted = NFKMLXChatterbox.conditionals(base, for: request)
        XCTAssertEqual(adjusted.t3.exaggeration, 0.9, accuracy: 1e-6)
        XCTAssertEqual(adjusted.t3.promptTokens, [1, 2, 3])
        XCTAssertEqual(base.t3.exaggeration, 0.5, "the backend's voice keeps its own value")

        let plain = NFKMLXChatterbox.conditionals(base, for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"]))
        XCTAssertEqual(plain.t3.exaggeration, 0.5)
    }

    func testTheCoreSamplingKeysReachT3() {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"],
                                          parameters: [NFKParameterTemperature: 0, NFKParameterGuidanceScale: 0.3,
                                                       NFKParameterSeed: 7])
        let options = NFKMLXChatterbox.samplingOptions(for: request)
        XCTAssertEqual(options.temperature, 0)
        XCTAssertEqual(options.guidance, 0.3, accuracy: 1e-6)
        XCTAssertEqual(options.seed, 7)

        let defaults = NFKMLXChatterbox.samplingOptions(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"]))
        let reference = NFKMLXT3SamplingOptions()
        XCTAssertEqual(defaults.temperature, reference.temperature)
        XCTAssertEqual(defaults.guidance, reference.guidance)
        XCTAssertEqual(defaults.seed, reference.seed)
    }

    func testTheSpeechBackendHandsItsClosureTheRequest() throws {
        try requireMLXRuntime()
        let key = NFKMLXChatterboxParameterKey.exaggeration
        let backend = NFKMLXSpeechBackend(identifier: "request-speech",
                                          configuration: NFKMLXSpeechConfiguration(sampleRate: 1000),
                                          parameterKeys: [key]) { _, sampleRate, request in
            let tenths = (request.parameter(forKey: key) as? NSNumber)?.intValue ?? 1
            return MLXArray.zeros([sampleRate * tenths / 10])
        }
        XCTAssertEqual(backend.supportedParameterKeys, [key, NFKParameterSampleRate])
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"],
                                                                       parameters: [key: 3]))
        let audio = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        let url = try XCTUnwrap(audio.fileURL)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(audio.durationSeconds, 0.3, accuracy: 1e-6)
    }

    func testAPlainSpeechBackendDeclaresOnlyTheSampleRate() throws {
        let backend = NFKMLXSpeechBackend(identifier: "plain-speech") { _, sampleRate in MLXArray.zeros([sampleRate]) }
        XCTAssertEqual(backend.supportedParameterKeys, [NFKParameterSampleRate])
    }
}
