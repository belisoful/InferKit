//
//  NFKMLXLayerStreamingTests.swift
//  InferKitMLXTests
//
//  Weight-free tests for streaming a dense decoder's layers: a random Gemma 3 is written out as a
//  bfloat16 release, and the same release loaded with every layer held and with its layers streamed
//  must compute the same values, element for element, at both precisions.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLayerStreamingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private let tokens: [Int32] = [3, 17, 42, 99, 7, 61, 12, 5, 88]

    /// A six-layer random Gemma 3 written as a bfloat16 release, as a multimodal one names its decoder.
    private func release() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("layer-stream-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let config: [String: Any] = [
            "model_type": "gemma3_text", "hidden_size": 64, "num_hidden_layers": 6, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 96, "vocab_size": 131,
            "rope_scaling": ["rope_type": "linear", "factor": 8], "sliding_window": 4, "sliding_window_pattern": 3,
            "query_pre_attn_scalar": 16, "attn_logit_softcapping": 50,
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        var configuration = NFKMLXGemma3Configuration.tiny
        configuration.layerCount = 6
        configuration.layerTypes = NFKMLXGemma3Configuration.layerTypes(count: 6, pattern: 3)
        NFKMLXRandom.seed(11)
        let net = NFKMLXGemma3Net(configuration)
        var stored = [String: MLXArray]()
        for (key, value) in net.parameters().flattened() {
            // Gemma's norm weights start at zero; random ones make every layer's norms count.
            let filled = key.hasSuffix("norm.weight") ? MLXRandom.normal(value.shape) * 0.1 : value
            stored["language_model.model.\(key)"] = filled.asType(.bfloat16)
        }
        try save(arrays: stored, url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    /// A load of `directory` holding only its first layer, every other layer streamed.
    private func streamedNet(_ directory: URL, precision: NFKMLXWeightPrecision) throws -> NFKMLXGemma3Net {
        let footprint = try NFKMLXGemma3.decoderFootprint(directory: directory, precision: precision, includesVision: false)
        let budget = NFKMLXResidencyBudget.reserve + footprint.streamedMinimumBytes + footprint.layerBytes[0]
        return try NFKMLXGemma3Language.network(directoryURL: directory, precision: precision, residency: .streamed,
                                                budget: budget)
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    func testAStreamedDecoderComputesWhatAHeldOneDoes() throws {
        try requireMLXRuntime()
        let directory = try release()
        for precision in [NFKMLXWeightPrecision.float32, .checkpoint] {
            let held = try NFKMLXGemma3Language.network(directoryURL: directory, precision: precision,
                                                        residency: .resident, budget: 0)
            let streamed = try streamedNet(directory, precision: precision)
            XCTAssertNil(held.layerStream)
            XCTAssertEqual(streamed.layerStream?.layers, [1, 2, 3, 4, 5], "the first layer is held, the rest stream")

            let input = MLXArray(tokens).reshaped([1, tokens.count])
            XCTAssertEqual(maxDifference(held(input), streamed(input)), 0, "\(precision): the prefill logits")
            for (index, (a, b)) in zip(held.layerStates(input), streamed.layerStates(input)).enumerated() {
                XCTAssertEqual(maxDifference(a, b), 0, "\(precision): the state after layer \(index)")
            }

            // Cached decoding past the 4-position window, so the sliding cache trims between reads.
            let heldCache = NFKMLXGemma3Cache(layerCount: 6, slidingWindow: 4)
            let streamedCache = NFKMLXGemma3Cache(layerCount: 6, slidingWindow: 4)
            let prompt = MLXArray(Array(tokens[0 ..< 5])).reshaped([1, 5])
            XCTAssertEqual(maxDifference(held(prompt, cache: heldCache), streamed(prompt, cache: streamedCache)), 0)
            for token in tokens[5...] {
                let step = MLXArray([token]).reshaped([1, 1])
                XCTAssertEqual(maxDifference(held(step, cache: heldCache), streamed(step, cache: streamedCache)), 0,
                               "\(precision): the cached step at \(token)")
            }
            try streamed.verifyStream()

            let stream = try XCTUnwrap(streamed.layerStream)
            let passes = 2 + 1 + tokens.count - 5
            XCTAssertGreaterThanOrEqual(stream.readStatistics.bytes, passes * stream.bytesPerPass,
                                        "every pass reads every streamed layer")
            for index in stream.layers {
                XCTAssertTrue(streamed.layers[index].parameters().flattened().allSatisfy { $0.1.size == 0 },
                              "layer \(index) holds nothing between its turns")
            }
            XCTAssertFalse(streamed.layers[0].parameters().flattened().contains { $0.1.size == 0 })
        }
    }

    func testAutomaticHoldsWhatFitsAndOnlyStreamsAgainstAShortfall() throws {
        try requireMLXRuntime()
        let directory = try release()
        XCTAssertNil(try NFKMLXGemma3Language.network(directoryURL: directory, precision: .checkpoint).layerStream,
                     "a release that fits loads whole, as it always has")
        let footprint = try NFKMLXGemma3.decoderFootprint(directory: directory, precision: .float32, includesVision: false)
        XCTAssertEqual(footprint.layerBytes.count, 6)
        XCTAssertEqual(footprint.bytes, footprint.unstreamableBytes + footprint.layerBytes.reduce(0, +))
        let checkpoint = try NFKMLXGemma3.decoderFootprint(directory: directory, precision: .checkpoint, includesVision: false)
        XCTAssertEqual(footprint.bytes, 2 * checkpoint.bytes, "a bfloat16 release doubles at float32")
        XCTAssertEqual(try NFKMLXGemma3.streamedLayers(directory: directory, precision: .float32, includesVision: false,
                                                       residency: .streamed, budget: 1 << 40), [],
                       "streamed where everything fits, nothing streams")
    }

    func testAFailedReadIsReportedRatherThanComputedAround() throws {
        try requireMLXRuntime()
        let directory = try release()
        let streamed = try streamedNet(directory, precision: .checkpoint)
        let file = directory.appendingPathComponent("model.safetensors")
        let header = try XCTUnwrap(NFKMLXSafetensors.entries(inFile: file).values.map(\.start).min())
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(header))
        try handle.close()

        _ = streamed(MLXArray(tokens).reshaped([1, tokens.count]))
        XCTAssertThrowsError(try streamed.verifyStream(), "the truncated release cannot be read")
    }

    /// A Gemma 3 model around `decoder`, its tokenizer a stand-in: generation reads token ids.
    private func model(_ decoder: NFKMLXGemma3Net) throws -> NFKMLXGemma3Model {
        let json: [String: Any] = ["model": ["vocab": ["<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3], "merges": [],
                                             "unk_token": "<unk>"],
                                   "added_tokens": [["id": 2, "content": "<bos>", "special": true]]]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("layer-stream-tok-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return NFKMLXGemma3Model(decoder: decoder, vision: nil, projector: nil,
                                 tokenizer: try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url)),
                                 tokens: NFKMLXGemma3Tokens(tokensPerImage: 3), chatTemplate: nil)
    }

    // A streamed decoder's pass verifies several of a held draft's proposals, so a drafted run reads
    // the release fewer times than a plain one and produces the held decoder's greedy output.
    func testAStreamedDecoderVerifiesAHeldDraftsProposals() throws {
        try requireMLXRuntime()
        let directory = try release()
        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = 16
        options.stopTokens = [-1]
        let prompt = tokens.map { Int($0) }
        let held = try model(try NFKMLXGemma3Language.network(directoryURL: directory, precision: .float32,
                                                              residency: .resident, budget: 0))
        let plain = try held.generate(tokens: prompt, options: options)

        let streamed = try model(try streamedNet(directory, precision: .float32))
        try streamed.useDraft(held)
        XCTAssertEqual(try streamed.generate(tokens: prompt, options: options), plain)
        let report = streamed.lastSpeculativeReport
        XCTAssertEqual(report.accepted, report.proposed, "the draft is the same release held")
        XCTAssertLessThan(report.rounds + 1, plain.count, "fewer streamed passes than tokens")
    }

    // The drafted factory holds the draft's decoder alone, quantized, and the output is still the
    // release's own greedy output.
    func testTheDraftedFactoryHoldsAQuantizedDecoderAndKeepsTheOutput() throws {
        try requireMLXRuntime()
        let directory = try release()
        let tokenizer: [String: Any] = ["model": ["vocab": ["<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3], "merges": [],
                                                  "unk_token": "<unk>"],
                                        "added_tokens": [["id": 2, "content": "<bos>", "special": true]]]
        try JSONSerialization.data(withJSONObject: tokenizer).write(to: directory.appendingPathComponent("tokenizer.json"))
        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = 12
        options.stopTokens = [-1]
        let prompt = tokens.map { Int($0) }
        let plain = try NFKMLXGemma3.model(directoryURL: directory, precision: .checkpoint, residency: .resident)
            .generate(tokens: prompt, options: options)

        let drafted = try NFKMLXGemma3.model(directoryURL: directory, draftDirectoryURL: directory, precision: .checkpoint,
                                             residency: .automatic)
        let draft = try XCTUnwrap(drafted.draft)
        XCTAssertNil(draft.vision)
        XCTAssertTrue(draft.decoder.leafModules().flattened().contains { $0.1 is QuantizedLinear },
                      "the draft's projections are quantized")
        XCTAssertEqual(try drafted.generate(tokens: prompt, options: options), plain)
        XCTAssertGreaterThan(drafted.lastSpeculativeReport.rounds, 0)
    }

    func testAStreamedDecoderDoesNotTrain() throws {
        try requireMLXRuntime()
        let streamed = try streamedNet(try release(), precision: .float32)
        XCTAssertThrowsError(try NFKMLXGemma3Language.fineTune(
            streamed, examples: { _ in (MLXArray(self.tokens).reshaped([1, self.tokens.count]), 0) }, steps: 1))
    }

    /// The released TranslateGemma 4B at its own bfloat16, held and then with half its layers streamed:
    /// the same values, element for element, through the prefill logits, every layer's state, and a
    /// greedy continuation. The two loads run one after the other, about 9 GB each.
    func testAReleasedDecoderStreamsToTheValuesItsHeldLoadComputes() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_TRANSLATEGEMMA"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_TRANSLATEGEMMA to the translategemma-4b-it release directory")
        }
        let directory = URL(fileURLWithPath: path)
        func run(_ translator: NFKMLXTranslateGemmaTranslator) throws -> (logits: MLXArray, states: [MLXArray], produced: [Int]) {
            let ids = translator.promptTokens(text: "The quick brown fox jumps over the lazy dog.", sourceCode: "en", targetCode: "de")
            let states = translator.model.decoder.layerStates(MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count]))
                .map { $0[0, ids.count - 1] }
            let logits = translator.model.logits(tokens: ids, softTokens: nil)[0, (ids.count - 16)...]
            eval(states + [logits])
            return (logits, states, try translator.generate(promptTokens: ids, maxTokens: 24))
        }
        let held = try run(NFKMLXTranslateGemma.translator(directoryURL: directory, precision: .checkpoint, residency: .resident))
        NFKMLXGPU.clearCache()

        let footprint = try NFKMLXGemma3.decoderFootprint(directory: directory, precision: .checkpoint, includesVision: true)
        let half = footprint.layerBytes.prefix(footprint.layerBytes.count / 2).reduce(0, +)
        let parts = try NFKMLXGemma3.load(directory: directory, precision: .checkpoint, decoder: true, residency: .streamed,
                                          budget: NFKMLXResidencyBudget.reserve + footprint.streamedMinimumBytes + half)
        let decoder = try XCTUnwrap(parts.decoder)
        let model = NFKMLXGemma3Model(decoder: decoder, vision: parts.vision, projector: parts.projector,
                                      tokenizer: parts.tokenizer, tokens: parts.tokens,
                                      chatTemplate: NFKMLXGemma3.chatTemplate(inDirectory: directory))
        let stream = try XCTUnwrap(decoder.layerStream)
        XCTAssertEqual(stream.layers.count, decoder.configuration.layerCount - decoder.configuration.layerCount / 2)
        let streamed = try run(NFKMLXTranslateGemma.translator(model: model, directoryURL: directory))

        XCTAssertEqual(maxDifference(held.logits, streamed.logits), 0, "the prefill logits")
        for (index, (a, b)) in zip(held.states, streamed.states).enumerated() {
            XCTAssertEqual(maxDifference(a, b), 0, "the state after layer \(index)")
        }
        XCTAssertEqual(held.produced, streamed.produced, "the greedy continuation")
        let (bytes, seconds) = stream.readStatistics
        print("translategemma-4b streamed: \(stream.layers.count) of \(decoder.configuration.layerCount) layers, "
              + String(format: "%.1f GB read at %.2f GB/s", Double(bytes) / 1e9, Double(bytes) / 1e9 / max(seconds, 1e-9)))
    }
}
