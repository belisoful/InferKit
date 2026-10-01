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
    }
}
