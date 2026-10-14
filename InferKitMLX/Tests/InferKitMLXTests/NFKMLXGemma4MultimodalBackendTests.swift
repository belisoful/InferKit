//
//  NFKMLXGemma4MultimodalBackendTests.swift
//  InferKitMLXTests
//
//  Gemma 4 with a picture in the prompt behind the InferKit contract, at a tiny random geometry: the
//  image becomes one placeholder per real soft token inside Gemma 4's turn format, the reply comes back
//  as text, and a clip for a release without an audio tower is refused.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXGemma4MultimodalBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private static let imageToken = 100

    /// A Gemma tokenizer over the lowercase letters and Gemma 4's turn, channel, and media markers.
    private func tokenizer() throws -> NFKMLXGemmaTokenizer {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("gemma4mm-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let added: [String: Int] = ["<pad>": 0, "<eos>": 1, "<bos>": 2, "<|turn>": 105, "<turn|>": 106,
                                    "<|image|>": Self.imageToken, "<|audio|>": 101, "<|image>": 150,
                                    "<image|>": 151, "<|audio>": 152, "<audio|>": 153, "<|channel>": 154,
                                    "<channel|>": 155]
        var vocabulary = added
        vocabulary["<unk>"] = 3
        vocabulary["\u{2581}"] = 4
        for (index, letter) in "abcdefghijklmnopqrstuvwxyz".enumerated() {
            vocabulary[String(letter)] = 5 + index
        }
        let json: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [] as [Any], "unk_token": "<unk>"],
            "added_tokens": added.map { ["content": $0.key, "id": $0.value, "special": true] as [String: Any] },
        ]
        let url = directory.appendingPathComponent("tokenizer.json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url))
    }

    /// The tiny decoder and a vision tower whose pooling is 1, so an 8×8 image is four soft tokens.
    private func model() throws -> NFKGemma4MultimodalModel {
        MLXRandom.seed(59)
        let vision = NFKMLXGemmaLanguage.makeVisionNet(
            NFKMLXGemma4VisionConfiguration(hiddenSize: 32, layerCount: 1, headCount: 4, keyValueHeadCount: 4,
                                            headDimensions: 8, intermediateSize: 48, patchSize: 4,
                                            positionEmbeddingSize: 16, poolingKernelSize: 1))
        let decoder = NFKMLXGemmaLanguage.makeNet(
            NFKMLXGemmaConfiguration(hiddenSize: 16, layerCount: 2, intermediateSize: 32, vocabularySize: 200,
                                     headCount: 2, keyValueHeadCount: 1, headDimensions: 8,
                                     globalHeadDimensions: 8, slidingWindow: 16, perLayerInputSize: 8,
                                     sharedKeyValueLayers: 0, finalLogitSoftcap: 0,
                                     layerTypes: [.sliding, .sliding], perLayerVocabularySize: 200))
        return NFKGemma4MultimodalModel(
            decoder: decoder, vision: (vision, NFKMLXGemma4MultimodalEmbedder(multimodalHidden: 32, textHidden: 16)),
            audio: nil, imageProcessor: NFKMLXGemma4ImageProcessor(patchSize: 4, poolingKernelSize: 1, maxSoftTokens: 8),
            tokenizer: try tokenizer(), imageTokenId: Self.imageToken, audioTokenId: 101)
    }

    func testAnImageTakesOnePlaceholderPerRealSoftToken() throws {
        try requireMLXRuntime()
        let model = try model()
        let image = NFKMLXFluxControlNetBackendTests.plate(width: 8, height: 8)
        let soft = try model.visionSoftTokens(image)
        XCTAssertEqual(soft.shape, [4, 16], "the budget's padding cells are dropped")
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "what is it"], parameters: [:])
        let text = try model.promptText(request, imageTokens: soft.dim(0), audioTokens: nil)
        XCTAssertTrue(text.hasPrefix("<bos><|turn>user\n<|image>"))
        XCTAssertTrue(text.hasSuffix("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
                      "the release's generation prompt opens an empty thinking channel")
        XCTAssertEqual(model.tokenizer.encode(text).filter { $0 == Self.imageToken }.count, 4)
    }

    func testAPromptAndAnImageBecomeAReply() throws {
        try requireMLXRuntime()
        let backend = NFKMLXGemmaLanguage.multimodalBackend(try model())
        XCTAssertEqual(backend.backendIdentifier, NFKMLXGemmaLanguage.gemmaMultimodalModelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputMessages, NFKInputImage, NFKInputAudio])
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "what is it", NFKInputImage: NFKMLXFluxControlNetBackendTests.plate(width: 8, height: 8)],
            parameters: [NFKParameterMaxTokens: 3])
        let result = try backend.runInference(for: request)
        XCTAssertNotNil(result.output(forKey: NFKOutputText) as? String)
        XCTAssertNotNil(result.output(forKey: NFKOutputUsage))
    }

    func testAClipForAReleaseWithoutAnAudioTowerIsRefused() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gemma4mm-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWaveFile.write(samples: [Float](repeating: 0.1, count: 1600), sampleRate: 16_000, to: url)
        let clip = NFKAudioAsset(fileURL: url, durationSeconds: 0.1, sampleRate: 16_000, channelCount: 1)
        let backend = NFKMLXGemmaLanguage.multimodalBackend(try model())
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "what is it", NFKInputAudio: clip], parameters: [NFKParameterMaxTokens: 1])))
    }

    func testARequestWithoutAPromptIsRefused() throws {
        try requireMLXRuntime()
        let backend = NFKMLXGemmaLanguage.multimodalBackend(try model())
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: NFKMLXFluxControlNetBackendTests.plate(width: 8, height: 8)], parameters: [:])))
    }
}
