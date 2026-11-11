//
//  NFKMLXGemmaConversationTests.swift
//  InferKitMLXTests
//
//  A Gemma 4 message request on `NFKMLXGemmaBackend` renders the release's `<|turn>` format and runs,
//  a tokenizer with `<start_of_turn>` keeps its turn ids, and a Gemma 4 image yields only its real soft
//  tokens, at a tiny random geometry.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXGemmaConversationTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private static let gemma4Markers: [String: Int] = [
        "<pad>": 0, "<eos>": 1, "<bos>": 2, "<|turn>": 105, "<turn|>": 106, "<|channel>": 154, "<channel|>": 155,
    ]
    private static let gemma3Markers: [String: Int] = [
        "<pad>": 0, "<eos>": 1, "<bos>": 2, "<start_of_turn>": 105, "<end_of_turn>": 106,
    ]

    /// The conversation the release's template renders below, verified with Python's jinja2 against the
    /// Gemma 4 `chat_template.jinja` (the 12B and 26B-A4B releases ship the same file).
    private static let messages: [[String: Any]] = [
        ["role": "system", "content": "be brief"],
        ["role": "user", "content": " hi "],
        ["role": "assistant", "content": "<|channel>thought\nmm<channel|>yo"],
        ["role": "user", "content": "what"],
    ]
    private static let rendered = "<bos><|turn>system\nbe brief<turn|>\n<|turn>user\nhi<turn|>\n"
        + "<|turn>model\nyo<turn|>\n<|turn>user\nwhat<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("gemma-chat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    /// A Gemma tokenizer over the lowercase letters (ids 5 … 30), the newline byte (31), and `markers`.
    private func tokenizer(_ markers: [String: Int]) throws -> NFKMLXGemmaTokenizer {
        var vocabulary = markers
        vocabulary["<unk>"] = 3
        vocabulary["\u{2581}"] = 4
        for (index, letter) in "abcdefghijklmnopqrstuvwxyz".enumerated() {
            vocabulary[String(letter)] = 5 + index
        }
        vocabulary["<0x0A>"] = 31
        let json: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [] as [Any], "unk_token": "<unk>"],
            "added_tokens": markers.map { ["content": $0.key, "id": $0.value, "special": true] as [String: Any] },
        ]
        let url = try temporaryDirectory().appendingPathComponent("tokenizer.json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url))
    }

    private func decoder() -> NFKMLXGemmaNet {
        MLXRandom.seed(61)
        return NFKMLXGemmaLanguage.makeNet(
            NFKMLXGemmaConfiguration(hiddenSize: 16, layerCount: 2, intermediateSize: 32, vocabularySize: 200,
                                     headCount: 2, keyValueHeadCount: 1, headDimensions: 8,
                                     globalHeadDimensions: 8, slidingWindow: 16, perLayerInputSize: 8,
                                     sharedKeyValueLayers: 0, finalLogitSoftcap: 0,
                                     layerTypes: [.sliding, .sliding], perLayerVocabularySize: 200))
    }

    private func backend(_ markers: [String: Int], releaseDirectoryURL: URL? = nil) throws -> NFKMLXGemmaBackend {
        let net = decoder()
        return NFKMLXGemmaBackend(logits: { net($0) }, tokenizer: try tokenizer(markers), identifier: "gemma4",
                                  releaseDirectoryURL: releaseDirectoryURL)
    }

    private func messagesRequest() -> NFKInferenceRequest {
        NFKInferenceRequest(inputs: [NFKInputMessages: Self.messages], parameters: [NFKParameterMaxTokens: 2])
    }

    /// A PEFT adapter directory holding `layers`' `[rank, inputs]` and `[outputs, rank]` halves at rank 4.
    private func peftAdapter(_ layers: [String: (inputs: Int, outputs: Int)]) throws -> URL {
        let directory = try temporaryDirectory()
        var arrays = [String: MLXArray]()
        for (index, (path, shape)) in layers.sorted(by: { $0.key < $1.key }).enumerated() {
            arrays["base_model.model.\(path).lora_A.weight"] = MLXRandom.normal([4, shape.inputs], key: MLXRandom.key(UInt64(index))) * 0.3
            arrays["base_model.model.\(path).lora_B.weight"] = MLXRandom.normal([shape.outputs, 4], key: MLXRandom.key(UInt64(index + 50))) * 0.3
        }
        let file = directory.appendingPathComponent("adapter_model.safetensors")
        try MLX.save(arrays: arrays, url: file)
        try JSONSerialization.data(withJSONObject: ["peft_type": "LORA", "r": 4, "lora_alpha": 32])
            .write(to: directory.appendingPathComponent("adapter_config.json"))
        return file
    }

    func testAGemma4BackendRunsHeldAdaptersAtEachRequestsStrength() throws {
        try requireMLXRuntime()
        let plain = decoder()
        eval(plain)
        let adapted = decoder()
        adapted.update(parameters: plain.parameters())
        // A conditional release's adapter names `model.language_model.`; one trained on the bare text model, `layers.`.
        let file = try peftAdapter(["model.language_model.layers.0.self_attn.q_proj": (16, 16),
                                    "layers.1.mlp.down_proj": (32, 16)])
        let strengths = try NFKMLXGemmaLanguage.holdAdapters([(file, 1)], in: adapted, layerStream: nil)
        XCTAssertEqual(strengths, [1])
        XCTAssertTrue(adapted.layers[0].attention.queryProjection is NFKMLXAdaptedLinear)
        XCTAssertTrue(adapted.layers[1].feedForward.down is NFKMLXAdaptedLinear)

        let markers = Self.gemma3Markers
        let backend = NFKMLXGemmaBackend(logits: { adapted($0) }, tokenizer: try tokenizer(markers), identifier: "gemma4")
        backend.adapterStrengths = strengths
        let base = NFKMLXGemmaBackend(logits: { plain($0) }, tokenizer: try tokenizer(markers), identifier: "gemma4")
        XCTAssertTrue(backend.supportedParameterKeys.contains(NFKParameterAdapterStrengths))
        XCTAssertFalse(base.supportedParameterKeys.contains(NFKParameterAdapterStrengths))
        func run(_ backend: NFKMLXGemmaBackend, _ strengths: [Float]?) throws -> String {
            var parameters: [String: Any] = [NFKParameterTemperature: 0, NFKParameterMaxTokens: 4]
            parameters[NFKParameterAdapterStrengths] = strengths?.map { NSNumber(value: $0) }
            let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"], parameters: parameters))
            return try XCTUnwrap(result.output(forKey: NFKOutputText) as? String)
        }
        XCTAssertEqual(try run(backend, [0]), try run(base, nil), "strength 0 runs the base decoder")
        XCTAssertThrowsError(try run(backend, [1, 1]), "one strength per held adapter")

        let towers = try peftAdapter(["model.vision_tower.encoder.layers.0.self_attn.q_proj": (16, 16)])
        XCTAssertThrowsError(try NFKMLXGemmaLanguage.holdAdapters([(towers, 1)], in: decoder(), layerStream: nil),
                             "the text decoder holds no tower's adapter")
    }

    func testAGemma4ConversationWithoutATemplateTakesTheReleaseTurnFormat() throws {
        XCTAssertEqual(try NFKMLXGemmaLanguage.conversationText(Self.messages, template: nil), Self.rendered)
        XCTAssertThrowsError(try NFKMLXGemmaLanguage.conversationText(
            [["role": "system", "content": "be brief"]], template: nil, mediaPrefix: "<|image>"),
            "media needs a user turn to open")
    }

    func testAGemma4MessagesRequestRendersTheTurnsAndRuns() throws {
        try requireMLXRuntime()
        let backend = try backend(Self.gemma4Markers)
        let tokenizer = try tokenizer(Self.gemma4Markers)
        let request = messagesRequest()
        XCTAssertEqual(try backend.tokens(for: request), tokenizer.encode(Self.rendered))
        let result = try backend.runInference(for: request)
        XCTAssertNotNil(result.output(forKey: NFKOutputText) as? String)
        let usage = try XCTUnwrap(result.output(forKey: NFKOutputUsage) as? [String: Any])
        XCTAssertEqual((usage[NFKUsageInputTokens] as? NSNumber)?.intValue, tokenizer.encode(Self.rendered).count)

        let raw = NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"], parameters: [:])
        XCTAssertEqual(try backend.tokens(for: raw), [2, 12, 13], "a raw prompt is the begin marker and its text")
    }

    func testAGemma4MessagesRequestRendersTheReleaseTemplate() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        let release = ["IK_VAL_GEMMA4_26B_CUT6", "IK_VAL_GEMMA4_12B_CUT6"].lazy
            .compactMap { environment[$0] }
            .map { URL(fileURLWithPath: $0).appendingPathComponent("chat_template.jinja") }
            .first { FileManager.default.fileExists(atPath: $0.path) }
        guard let templateURL = release else {
            throw XCTSkip("set IK_VAL_GEMMA4_26B_CUT6 or IK_VAL_GEMMA4_12B_CUT6 (a release with chat_template.jinja)")
        }
        let template = try String(contentsOf: templateURL, encoding: .utf8)
        XCTAssertEqual(try NFKMLXGemmaLanguage.conversationText(Self.messages, template: template), Self.rendered)
        XCTAssertEqual(try NFKMLXGemmaLanguage.conversationText([["role": "user", "content": "hi"]], template: template,
                                                                mediaPrefix: "<|image><|image|><image|>"),
                       "<bos><|turn>user\n<|image><|image|><image|>hi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")

        let directory = try temporaryDirectory()
        try template.write(to: directory.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
        let backend = try backend(Self.gemma4Markers, releaseDirectoryURL: directory)
        let request = messagesRequest()
        XCTAssertEqual(try backend.tokens(for: request), try tokenizer(Self.gemma4Markers).encode(Self.rendered))
        XCTAssertNotNil(try backend.runInference(for: request).output(forKey: NFKOutputText) as? String)
    }

    func testAGemma3MessagesRequestKeepsItsStartOfTurnIds() throws {
        try requireMLXRuntime()
        // A release template next to a `<start_of_turn>` tokenizer leaves its hand-built turns in place.
        let directory = try temporaryDirectory()
        try "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
            .write(to: directory.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
        let backend = try backend(Self.gemma3Markers, releaseDirectoryURL: directory)
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "user", "content": "hi"]]],
                                          parameters: [NFKParameterMaxTokens: 2])
        // <bos> <start_of_turn> "user\nhi" <end_of_turn> "\n" <start_of_turn> "model\n"
        XCTAssertEqual(try backend.tokens(for: request),
                       [2, 105, 25, 23, 9, 22, 31, 12, 13, 106, 31, 105, 17, 19, 8, 9, 16, 31])
        XCTAssertNotNil(try backend.runInference(for: request).output(forKey: NFKOutputText) as? String)
    }

    func testPooledCellsFollowTheReferenceMask() throws {
        try requireMLXRuntime()
        // A 4-column, 2-row patch grid padded to 16 patches, pooled 2×2 into a 4-cell budget.
        var grid = [Int32]()
        for y in 0 ..< 2 {
            for x in 0 ..< 4 { grid += [Int32(x), Int32(y)] }
        }
        let padded = MLXArray(grid + [Int32](repeating: -1, count: 16)).reshaped([16, 2])
        XCTAssertEqual(NFKMLXGemma4ConditionalGeneration.pooledCells(positionIds: padded, kernel: 2, cellCount: 4), [0, 1])

        var full = [Int32]()
        for y in 0 ..< 4 {
            for x in 0 ..< 4 { full += [Int32(x), Int32(y)] }
        }
        XCTAssertEqual(NFKMLXGemma4ConditionalGeneration.pooledCells(
            positionIds: MLXArray(full).reshaped([16, 2]), kernel: 2, cellCount: 4), [0, 1, 2, 3])
    }

    func testAPaddedImageKeepsOnlyItsRealSoftTokens() throws {
        try requireMLXRuntime()
        // A 16×8 picture at 4-pixel patches resizes to 16×8: a 4×2 patch grid, pooled 2×2 into 2 of the
        // 4-token budget's cells.
        let vision = NFKMLXGemmaLanguage.makeVisionNet(
            NFKMLXGemma4VisionConfiguration(hiddenSize: 32, layerCount: 1, headCount: 4, keyValueHeadCount: 4,
                                            headDimensions: 8, intermediateSize: 48, patchSize: 4,
                                            positionEmbeddingSize: 16, poolingKernelSize: 2))
        let processor = NFKMLXGemma4ImageProcessor(patchSize: 4, poolingKernelSize: 2, maxSoftTokens: 4)
        let model = NFKMLXGemma4ConditionalGeneration(
            decoder: decoder(), visionTower: vision,
            visionEmbedder: NFKMLXGemma4MultimodalEmbedder(multimodalHidden: 32, textHidden: 16),
            imageProcessor: processor, imageTokenId: 100, audioTokenId: 101, padTokenId: 0)
        let picture = NFKMLXFluxControlNetBackendTests.plate(width: 16, height: 8)
        let size = processor.resizedSize(width: 16, height: 8)
        XCTAssertEqual(size.height, 8)
        XCTAssertEqual(size.width, 16)
        let soft = try XCTUnwrap(model.visionSoftTokens(for: picture))
        XCTAssertEqual(soft.shape, [2, 16], "the reference keeps (2 / 2) · (4 / 2) soft tokens")
        let prompt = [2, 100, 100, 5, 6]
        XCTAssertEqual(model.generate(promptTokens: prompt, image: picture, maxTokens: 2).count, 2)
    }
}
