//
//  NFKMLXSANABackendTests.swift
//  InferKitMLXTests
//
//  SANA behind the InferKit contract: the prompt is encoded as diffusers' SanaPipeline encodes it, the
//  image comes back under NFKOutputImage at the requested size, and a request's settings do not outlive
//  it.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSANABackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// A Gemma tokenizer over the lowercase letters, the metaspace, and the markers; anything else falls
    /// back to bytes the vocabulary lacks, which read as the unknown id.
    private func tokenizer() throws -> NFKMLXGemmaTokenizer {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("sana-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        var vocabulary: [String: Int] = ["<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "\u{2581}": 4]
        for (index, letter) in "abcdefghijklmnopqrstuvwxyz".enumerated() {
            vocabulary[String(letter)] = 5 + index
        }
        let json: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [] as [Any], "unk_token": "<unk>"],
            "added_tokens": [["content": "<bos>", "id": 2, "special": true],
                             ["content": "<eos>", "id": 1, "special": true]],
        ]
        let url = directory.appendingPathComponent("tokenizer.json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url))
    }

    static func cgImage(_ value: Any?) throws -> CGImage {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.noOutput
        }
        return (value as! CGImage)
    }

    private func tinyGenerator() throws -> NFKMLXSANAGenerator {
        let transformer = NFKMLXSANAConfiguration.tiny                        // caption channels 12
        let text = NFKMLXGemma2Configuration(hiddenSize: 12, layerCount: 2, headCount: 2, kvHeadCount: 1, headDim: 4,
                                             intermediateSize: 24, vocabularySize: 64, queryPreAttnScalar: 4,
                                             slidingWindow: 8)
        MLXRandom.seed(9)
        let generator = NFKMLXSANAGenerator(
            resident: true, tokenizer: try tokenizer(),
            pixelsPerLatent: 1 << (NFKMLXDCAEConfiguration.tiny.blockOutChannels.count - 1),
            loadTextEncoder: { NFKMLXGemma2Net(text) },
            loadPipeline: {
                NFKMLXSANAPipeline(transformer: NFKMLXSANATransformerNet(transformer),
                                   vae: NFKMLXDCAutoencoderNet(.tiny))
            })
        try generator.loadResident()
        return generator
    }

    func testAPositivePromptKeepsTheBeginMarkerAndTheStatesFromTheInstructionsLastToken() throws {
        let tokenizer = try tokenizer()
        let count = NFKMLXSANAPrompt.instructionTokenCount(tokenizer: tokenizer)
        let encoding = NFKMLXSANAPrompt.encoding("A Red Fox ", tokenizer: tokenizer, instructionTokens: count)
        XCTAssertEqual(encoding.ids.first, 2, "the begin-of-sequence marker leads")
        XCTAssertEqual(encoding.kept.first, 0)
        XCTAssertEqual(encoding.kept.dropFirst().first, count - 1, "the instruction's last token is kept")
        XCTAssertEqual(encoding.kept.last, encoding.ids.count - 1)
        let negative = NFKMLXSANAPrompt.encoding("", tokenizer: tokenizer, instructionTokens: nil)
        XCTAssertEqual(negative.ids, [2], "an empty negative prompt is the begin marker alone")
        let lowered = NFKMLXSANAPrompt.encoding("RED", tokenizer: tokenizer, instructionTokens: nil)
        XCTAssertEqual(lowered.ids, [2, 5 + 17, 5 + 4, 5 + 3], "the prompt is lowercased as the reference does")
    }

    func testARequestBecomesAnImageAtTheRequestedSize() throws {
        try requireMLXRuntime()
        let generator = try tinyGenerator()
        let backend = generator.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXSANAGenerator.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputNegativePrompt])

        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a red fox"],
                                          parameters: [NFKParameterWidth: 8, NFKParameterHeight: 12,
                                                       NFKParameterSteps: 2, NFKParameterGuidanceScale: 3,
                                                       NFKParameterSeed: 4])
        let result = try backend.runInference(for: request)
        let picture = try Self.cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(picture.width, 8)
        XCTAssertEqual(picture.height, 12)
        XCTAssertEqual(generator.steps, 20, "the request's steps apply to that request only")
        XCTAssertEqual(generator.guidance, 4.5)
    }

    func testARequestWithoutAPromptIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyGenerator().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }

    func testASideOffTheLatentGridIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyGenerator().inferenceBackend()
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a fox"],
                                          parameters: [NFKParameterWidth: 9, NFKParameterHeight: 8])
        XCTAssertThrowsError(try backend.runInference(for: request))
    }
}
