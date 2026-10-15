//
//  NFKMLXTokenCountTests.swift
//  InferKitMLXTests
//
//  tokenCount(for:) on the language decoders, with random weights and hand-written vocabularies: the
//  count of a request equals the input tokens a run of the same request reports, through a chat
//  template that renders the tools. Runs where MLX has a Metal library. Mamba, Gemma 3n, and
//  Phi-4-multimodal have no small build in the suite (a Mistral tokenizer, the audio and vision
//  towers), so their counts have no equality test here.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXTokenCountTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// A template made of the vocabulary's own pieces: each message's content, then one piece per tool.
    private static let template = "{% for message in messages %}{{ message.content }} {% endfor %}"
        + "{% if tools %}{% for tool in tools %}hello {% endfor %}{% endif %}"

    private static let tools: [[String: Any]] = [
        ["name": "lookup", "description": "Looks a term up.", "parameters": ["type": "object"]],
        ["name": "fetch", "description": "Fetches a page.", "parameters": ["type": "object"]],
    ]

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXTokenCountTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    /// A byte-level vocabulary over the letters of "hello", with their merges.
    private func byteLevelTokenizer() throws -> NFKTokenizer {
        let directory = try temporaryDirectory()
        let vocabulary: [String: Int] = ["h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6, "Ġ": 7]
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\nh e\nl l\nhe ll\nhell o\n".write(
            to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        return try NFKTokenizer(forManifest: ["tokenizer": ["type": "bpe-bytelevel"]], directory: directory)
    }

    /// The input tokens a run of the request reports.
    private func runInputTokens(_ backend: any NFKInferenceBackend, _ request: NFKInferenceRequest) throws -> Int {
        let usage = try XCTUnwrap(try backend.runInference(for: request).output(forKey: NFKOutputUsage) as? [String: Any])
        return try XCTUnwrap((usage[NFKUsageInputTokens] as? NSNumber)?.intValue)
    }

    private func toolRequests(templateParameter: Bool) -> (plain: NFKInferenceRequest, withTools: NFKInferenceRequest) {
        let messages: [[String: Any]] = [["role": "user", "content": "hello hello"]]
        var parameters: [String: Any] = [NFKParameterTemperature: 0, NFKParameterMaxTokens: 1]
        if templateParameter {
            parameters[NFKMLXGenerationParameterKey.chatTemplate] = Self.template
        }
        var toolParameters = parameters
        toolParameters[NFKParameterTools] = Self.tools
        return (NFKInferenceRequest(inputs: [NFKInputMessages: messages], parameters: parameters),
                NFKInferenceRequest(inputs: [NFKInputMessages: messages], parameters: toolParameters))
    }

    func testTheLanguageBackendCountsWhatARunTokenizes() throws {
        try requireMLXRuntime()
        MLXRandom.seed(7)
        let backend = NFKMLXLanguageBackend(net: NFKMLXLanguage.makeNet(.tiny), tokenizer: try byteLevelTokenizer(),
                                            identifier: "tiny")
        let prompt = NFKInferenceRequest(inputs: [NFKInputPrompt: "hello hello hello"],
                                         parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 1])
        XCTAssertEqual(try backend.tokenCount(for: prompt).intValue, try runInputTokens(backend, prompt))

        let (plain, withTools) = toolRequests(templateParameter: true)
        let plainCount = try backend.tokenCount(for: plain).intValue
        let toolCount = try backend.tokenCount(for: withTools).intValue
        XCTAssertEqual(plainCount, try runInputTokens(backend, plain))
        XCTAssertEqual(toolCount, try runInputTokens(backend, withTools))
        XCTAssertGreaterThan(toolCount, plainCount, "the template renders the tools, and they count")
    }

    func testTheDecoderBackendCountsWhatARunTokenizes() throws {
        try requireMLXRuntime()
        MLXRandom.seed(11)
        let net = NFKMLXLanguage.makeNet(.tiny)
        let backend = NFKMLXDecoderBackend(logits: { net($0) }, tokenizer: try byteLevelTokenizer(), identifier: "decoder",
                                           chatTemplate: Self.template, stopTokens: [])
        let (plain, withTools) = toolRequests(templateParameter: false)
        let plainCount = try backend.tokenCount(for: plain).intValue
        let toolCount = try backend.tokenCount(for: withTools).intValue
        XCTAssertEqual(plainCount, try runInputTokens(backend, plain))
        XCTAssertEqual(toolCount, try runInputTokens(backend, withTools))
        XCTAssertGreaterThan(toolCount, plainCount, "the template renders the tools, and they count")
    }

    func testTheGraniteAndNemotronBackendsCountWhatARunTokenizes() throws {
        try requireMLXRuntime()
        MLXRandom.seed(13)
        let net = NFKMLXLanguage.makeNet(.tiny)
        let release = try temporaryDirectory()
        try Self.template.write(to: release.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
        let granite = NFKMLXGraniteBackend(logits: { net($0) }, tokenizer: try byteLevelTokenizer(), identifier: "granite",
                                           releaseDirectoryURL: release)
        let nemotron = NFKMLXNemotronBackend(logits: { net($0) }, tokenizer: try byteLevelTokenizer(), identifier: "nemotron",
                                             releaseDirectoryURL: release)
        let (plain, withTools) = toolRequests(templateParameter: false)
        XCTAssertEqual(try granite.tokenCount(for: plain).intValue, try runInputTokens(granite, plain), "a plain chat is joined")
        XCTAssertEqual(try granite.tokenCount(for: withTools).intValue, try runInputTokens(granite, withTools),
                       "a tool request renders through the release's template")
        XCTAssertEqual(try nemotron.tokenCount(for: plain).intValue, try runInputTokens(nemotron, plain), "a plain chat is joined")
        XCTAssertEqual(try nemotron.tokenCount(for: withTools).intValue, try runInputTokens(nemotron, withTools),
                       "a tool request renders through the release's template")
    }

    func testTheDeepSeekBackendCountsWhatARunTokenizes() throws {
        try requireMLXRuntime()
        MLXRandom.seed(17)
        let configuration = NFKMLXDeepSeekConfiguration(hiddenSize: 64, layerCount: 4, vocabularySize: 128,
                                                        headCount: 4, headDimensions: 16, ropeHeadDimensions: 4,
                                                        queryLoRARank: 16, outputLoRARank: 16, outputGroups: 2,
                                                        slidingWindow: 8, routedExpertCount: 8, activatedExpertCount: 2,
                                                        expertIntermediateSize: 32, hashLayerCount: 1)
        let backend = NFKMLXDeepSeekBackend(net: try NFKMLXDeepSeek.makeNet(configuration), tokenizer: try byteLevelTokenizer(),
                                            identifier: "deepseek-small")
        let (plain, withTools) = toolRequests(templateParameter: true)
        let plainCount = try backend.tokenCount(for: plain).intValue
        let toolCount = try backend.tokenCount(for: withTools).intValue
        XCTAssertEqual(plainCount, try runInputTokens(backend, plain))
        XCTAssertEqual(toolCount, try runInputTokens(backend, withTools))
        XCTAssertGreaterThan(toolCount, plainCount, "the template renders the tools, and they count")
    }

    func testTheGemmaBackendCountsWhatARunTokenizes() throws {
        try requireMLXRuntime()
        MLXRandom.seed(61)
        var vocabulary: [String: Int] = ["<pad>": 0, "<eos>": 1, "<bos>": 2, "<|turn>": 105, "<turn|>": 106,
                                         "<unk>": 3, "\u{2581}": 4, "<0x0A>": 31]
        for (index, letter) in "abcdefghijklmnopqrstuvwxyz".enumerated() {
            vocabulary[String(letter)] = 5 + index
        }
        let markers = ["<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>"]
        let json: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [] as [Any], "unk_token": "<unk>"],
            "added_tokens": markers.map { ["content": $0, "id": vocabulary[$0]!, "special": true] as [String: Any] },
        ]
        let url = try temporaryDirectory().appendingPathComponent("tokenizer.json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        let tokenizer = try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url))
        let net = NFKMLXGemmaLanguage.makeNet(
            NFKMLXGemmaConfiguration(hiddenSize: 16, layerCount: 2, intermediateSize: 32, vocabularySize: 200,
                                     headCount: 2, keyValueHeadCount: 1, headDimensions: 8,
                                     globalHeadDimensions: 8, slidingWindow: 16, perLayerInputSize: 8,
                                     sharedKeyValueLayers: 0, finalLogitSoftcap: 0,
                                     layerTypes: [.sliding, .sliding], perLayerVocabularySize: 200))
        let backend = NFKMLXGemmaBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: "gemma4")
        let messages = NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "system", "content": "be brief"],
                                                                       ["role": "user", "content": "hi there"]]],
                                           parameters: [NFKParameterMaxTokens: 1])
        let prompt = NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"], parameters: [NFKParameterMaxTokens: 1])
        XCTAssertEqual(try backend.tokenCount(for: messages).intValue, try runInputTokens(backend, messages))
        XCTAssertEqual(try backend.tokenCount(for: prompt).intValue, try runInputTokens(backend, prompt))
    }

    func testTheGemma3BackendCountsWhatARunTokenizes() throws {
        try requireMLXRuntime()
        MLXRandom.seed(5)
        var vocabulary: [String: Int] = ["<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3]
        for (index, piece) in ["\u{2581}", "H", "i", "\n", "u", "s", "e", "r", "m", "o", "d", "l", "\n\n"].enumerated() {
            vocabulary[piece] = 10 + index
        }
        let added: [[String: Any]] = [
            ["id": 2, "content": "<bos>", "special": true],
            ["id": 105, "content": "<start_of_turn>", "special": true],
            ["id": 106, "content": "<end_of_turn>", "special": true],
        ]
        let json: [String: Any] = ["model": ["vocab": vocabulary, "merges": [["\n", "\n"]], "unk_token": "<unk>"],
                                   "added_tokens": added]
        let url = try temporaryDirectory().appendingPathComponent("tokenizer.json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        let model = NFKMLXGemma3Model(decoder: NFKMLXGemma3Net(.tiny), vision: nil, projector: nil,
                                      tokenizer: try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url)),
                                      tokens: NFKMLXGemma3Tokens(tokensPerImage: 3), chatTemplate: nil)
        let backend = NFKMLXGemma3Backend(model: model, identifier: "tiny")
        let messages = NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "user", "content": "Hi Hi"]]],
                                           parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 1])
        let prompt = NFKInferenceRequest(inputs: [NFKInputPrompt: "Hi"],
                                         parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 1])
        XCTAssertEqual(try backend.tokenCount(for: messages).intValue, try runInputTokens(backend, messages))
        XCTAssertEqual(try backend.tokenCount(for: prompt).intValue, try runInputTokens(backend, prompt))
    }
}
