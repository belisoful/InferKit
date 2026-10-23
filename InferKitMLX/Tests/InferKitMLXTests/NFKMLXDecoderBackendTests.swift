//
//  NFKMLXDecoderBackendTests.swift
//  InferKitMLXTests
//
//  The prefill-only decoder backend over a release directory's tokenizer, chat template, and
//  generation config, with a scripted forward standing in for the decoder. Runs where MLX has a Metal
//  library (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDecoderBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private let vocabularySize = 12
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let vocabulary: [String: Int] = ["h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6,
                                         "Ġ": 7, "Ċ": 8, "<eos>": 9]
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\nh e\nl l\nhe ll\nhell o\n".write(
            to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        let tokenizerConfig: [String: Any] = [
            "tokenizer_class": "Qwen2Tokenizer",
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": ["10": ["content": "<|im_start|>", "special": true],
                                     "11": ["content": "<|im_end|>", "special": true]],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerConfig)
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        // The turn ends on <|im_end|>; the generation config adds <eos>, which only it names.
        try JSONSerialization.data(withJSONObject: ["eos_token_id": [11, 9]])
            .write(to: directory.appendingPathComponent("generation_config.json"))
        let template = "{% for m in messages %}<|im_start|>{{ m.content }}<|im_end|>{% endfor %}"
            + "{% if add_generation_prompt %}<|im_start|>{% endif %}"
        try template.write(to: directory.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    /// A forward that predicts `script` in turn at the last position, recording each input it reads.
    private func backend(script: [Int], inputs: Box) throws -> NFKMLXDecoderBackend {
        let size = vocabularySize
        return try NFKMLXDecoderBackend.release(directoryURL: directory, identifier: "scripted") { tokens in
            inputs.calls.append(tokens.asArray(Int32.self).map(Int.init))
            let length = tokens.dim(1)
            var logits = [Float](repeating: 0, count: length * size)
            logits[(length - 1) * size + script[min(inputs.calls.count - 1, script.count - 1)]] = 10
            return MLXArray(logits, [1, length, size])
        }
    }

    final class Box { var calls = [[Int]]() }

    func testAMessageListRendersThroughTheChatTemplateAndStopsAtAGenerationConfigID() throws {
        try requireMLXRuntime()
        let inputs = Box()
        let backend = try backend(script: [4, 6, 9], inputs: inputs)
        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputMessages: [["role": "user", "content": "hello"]]], parameters: nil))
        XCTAssertEqual(inputs.calls.first, [10, 6, 11, 10], "<|im_start|>hello<|im_end|><|im_start|>")
        XCTAssertEqual(result.output(forKey: NFKOutputText) as? String, "hehello",
                       "the reply stops at <eos>, which only the generation config names")
        XCTAssertEqual(inputs.calls.count, 3)
        XCTAssertEqual(result.output(forKey: NFKOutputFinishReason) as? String, NFKFinishReasonStop)
    }

    /// The release directory rewritten so three added tokens spell a whole Hermes call: the opening
    /// tag, the call's JSON, and the closing tag. A scripted forward then writes a real call.
    private func writeToolRelease() throws {
        let tokenizerConfig: [String: Any] = [
            "tokenizer_class": "Qwen2Tokenizer",
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": ["10": ["content": "<|im_start|>", "special": true],
                                     "11": ["content": "<|im_end|>", "special": true],
                                     "12": ["content": "<tool_call>", "special": false],
                                     "13": ["content": #"{"name": "get_weather", "arguments": {"city": "Paris"}}"#, "special": false],
                                     "14": ["content": "</tool_call>", "special": false]],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerConfig)
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        let template = "{% for m in messages %}<|im_start|>{{ m.content if m.content is string else '' }}"
            + "{% for c in m.tool_calls or [] %}<tool_call>{{ c.function.arguments | tojson }}</tool_call>{% endfor %}<|im_end|>{% endfor %}"
            + "{% if add_generation_prompt %}<|im_start|>{% endif %}"
        try template.write(to: directory.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
    }

    func testAToolLoopRunsOnTheDecoderBackendAndReadsItsCall() throws {
        try requireMLXRuntime()
        try writeToolRelease()
        let inputs = Box()
        // Turn one writes the call and ends the turn; turn two answers "hello".
        let script = [12, 13, 14, 11, 6, 11]
        let size = 15
        let model = try NFKMLXDecoderBackend.release(directoryURL: directory, identifier: "scripted") { tokens in
            inputs.calls.append(tokens.asArray(Int32.self).map(Int.init))
            let length = tokens.dim(1)
            var logits = [Float](repeating: 0, count: length * size)
            logits[(length - 1) * size + script[min(inputs.calls.count - 1, script.count - 1)]] = 10
            return MLXArray(logits, [1, length, size])
        }
        XCTAssertTrue(model.supportedParameterKeys.contains(NFKParameterTools))
        var cities: [String] = []
        let tool = NFKInferenceTool(name: "get_weather", description: "Current weather in a city.",
                                    parameters: ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]],
                                    handler: { arguments, _ in
            cities.append(arguments["city"] as? String ?? "")
            return NFKToolResult(text: "18 C")
        })
        let loop = NFKToolLoop(backend: model, tools: [tool])
        let result = try loop.runInference(for: NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "user", "content": "hello"]]],
                                                                    parameters: nil, outputModality: .text))
        XCTAssertEqual(cities, ["Paris"], "the call the model wrote ran with its arguments")
        XCTAssertEqual(result.text, "hello")
        let conversation = try XCTUnwrap(result.output(forKey: NFKOutputMessages) as? [[String: Any]])
        XCTAssertEqual(conversation.count, 4)
        XCTAssertEqual(conversation[2]["content"] as? String, "18 C")
        XCTAssertEqual(inputs.calls.count, 6)
        // The second turn's prompt renders the call back with the model's own arguments.
        let secondPrompt = inputs.calls[4]
        XCTAssertTrue(secondPrompt.contains(12) && secondPrompt.contains(14), "the call's tags are in the next prompt")
    }

    // DeepSeek builds without a template from a directory that ships one, so a tool request takes the
    // release's template, its generation config's stops, and the tool outputs. Random weights write
    // no call; the run is the path, not the answer.
    func testADeepSeekToolRequestTakesTheReleasesTemplateAndOutputs() throws {
        try requireMLXRuntime()
        try writeToolRelease()
        let tokenizer = try XCTUnwrap(NFKMLXLanguage.releaseTokenizer(inDirectory: directory))
        let configuration = NFKMLXDeepSeekConfiguration(hiddenSize: 64, layerCount: 4, vocabularySize: 128,
                                                        headCount: 4, headDimensions: 16, ropeHeadDimensions: 4,
                                                        queryLoRARank: 16, outputLoRARank: 16, outputGroups: 2,
                                                        slidingWindow: 8, routedExpertCount: 8, activatedExpertCount: 2,
                                                        expertIntermediateSize: 32, hashLayerCount: 1)
        let backend = NFKMLXDeepSeekBackend(net: try NFKMLXDeepSeek.makeNet(configuration), tokenizer: tokenizer,
                                            identifier: "deepseek-small", releaseDirectoryURL: directory)
        XCTAssertTrue(backend.supportedParameterKeys.contains(NFKParameterTools))
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "user", "content": "hello"]]],
                                          parameters: [NFKParameterTools: [["name": "get_weather", "parameters": ["type": "object"]]],
                                                       NFKParameterMaxTokens: 2, NFKParameterTemperature: 0],
                                          outputModality: .text)
        let result = try backend.runInference(for: request)
        XCTAssertNotNil(result.output(forKey: NFKOutputText))
        XCTAssertNotNil(result.output(forKey: NFKOutputAssistantMessage), "the reply comes back as the turn a tool loop appends")
    }

    func testARawPromptIsEncodedAsGivenAndStopsAtTheTokenizersEnd() throws {
        try requireMLXRuntime()
        let inputs = Box()
        let backend = try backend(script: [5, 11], inputs: inputs)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"],
                                                                       parameters: nil))
        XCTAssertEqual(inputs.calls.first, [6], "no chat template and no begin-of-sequence marker")
        XCTAssertEqual(result.output(forKey: NFKOutputText) as? String, "ll")
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputMessages])
        XCTAssertEqual(result.output(forKey: NFKOutputFinishReason) as? String, NFKFinishReasonStop)

        let limited = try self.backend(script: [5, 11], inputs: Box()).runInference(
            for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"], parameters: [NFKParameterMaxTokens: 1]))
        XCTAssertEqual(limited.output(forKey: NFKOutputFinishReason) as? String, NFKFinishReasonLength,
                       "a reply that reaches its limit says so")
    }
}
