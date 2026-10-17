//
//  NFKMLXLayerStreamingTests.swift
//  InferKitMLXTests
//
//  Weight-free tests for streaming a dense model's layers: a random Gemma 3 is written out as a
//  bfloat16 release and a random MADLAD-geometry T5 as a float32 one, and each release loaded with
//  every layer held and with its layers streamed must compute the same values, element for element,
//  at both precisions.
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
        let budget = NFKMLXResidencyBudget.leastBudget(holding: footprint.streamedMinimumBytes + footprint.layerBytes[0])
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
                                          budget: NFKMLXResidencyBudget.leastBudget(holding: footprint.streamedMinimumBytes + half))
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

    // MARK: T5 encoder-decoder

    private let sourceIds: [Int32] = [5, 17, 42, 99, 7, 61, 2]

    /// A random MADLAD-geometry T5, four blocks a stack and an untied head, written as a float32 release.
    private func t5Release() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("t5-stream-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let config: [String: Any] = [
            "model_type": "t5", "d_model": 32, "num_layers": 4, "num_decoder_layers": 4, "num_heads": 2, "d_kv": 16,
            "d_ff": 64, "vocab_size": 128, "relative_attention_num_buckets": 16, "relative_attention_max_distance": 32,
            "tie_word_embeddings": false, "pad_token_id": 1, "eos_token_id": 2, "decoder_start_token_id": 0,
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        NFKMLXRandom.seed(13)
        let net = NFKMLXT5Seq2SeqNet(try NFKMLXMADLADConfiguration(huggingFaceConfigURL: directory.appendingPathComponent("config.json")))
        var stored = [String: MLXArray]()
        for (key, value) in net.parameters().flattened() {
            // T5's norm weights start at one; random ones make every norm count.
            stored[key] = key.hasSuffix("layer_norm.weight") ? 1 + MLXRandom.normal(value.shape) * 0.1 : value
        }
        try save(arrays: stored, url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    /// A load of `directory` holding the first `count` blocks a streamed load holds (the decoder's from
    /// block 1), every other block but each stack's block 0 streamed.
    private func streamedT5(_ directory: URL, half: Bool, holding count: Int) throws -> NFKMLXT5Seq2SeqNet {
        let stage = try NFKMLXMADLAD.footprint(directory: directory, half: half).stage
        let budget = NFKMLXResidencyBudget.leastBudget(holding: stage.streamedMinimumBytes
            + stage.layerBytes.prefix(count).reduce(0, +))
        return try NFKMLXMADLAD.network(directoryURL: directory, configuration: nil, half: half, residency: .streamed,
                                        budget: budget)
    }

    func testAStreamedT5ComputesWhatAHeldOneDoes() throws {
        try requireMLXRuntime()
        let directory = try t5Release()
        defer { try? FileManager.default.removeItem(at: directory) }
        for half in [false, true] {
            let held = try NFKMLXMADLAD.network(directoryURL: directory, configuration: nil, half: half,
                                                residency: .resident, budget: 0)
            let streamed = try streamedT5(directory, half: half, holding: 2)
            XCTAssertNil(held.encoderStream)
            XCTAssertNil(held.decoderStream)
            XCTAssertEqual(streamed.decoderStream?.layers, [3], "decoder blocks 1 and 2 are held before any encoder block")
            XCTAssertEqual(streamed.encoderStream?.layers, [1, 2, 3], "each stack's block 0 is held")

            let source = MLXArray(sourceIds).reshaped([1, sourceIds.count])
            let target = MLXArray([Int32(0), 9, 10, 11, 12]).reshaped([1, 5])
            let memory = held.encode(source)
            XCTAssertEqual(maxDifference(memory, streamed.encode(source)), 0, "half \(half): the encoder output")
            XCTAssertEqual(maxDifference(held(source: source, target: target), streamed(source: source, target: target)), 0,
                           "half \(half): the teacher-forced logits")

            let heldCache = held.makeCache(), streamedCache = streamed.makeCache()
            for t in 0 ..< 5 {
                let step = target[0..., t ..< t + 1]
                XCTAssertEqual(maxDifference(held.decode(step, memory: memory, cache: heldCache),
                                             streamed.decode(step, memory: memory, cache: streamedCache)), 0,
                               "half \(half): the cached step \(t)")
            }

            let ids = sourceIds.map { Int($0) }
            var decoding = NFKMLXSeq2SeqDecoding(beams: 1, maxTokens: 8, startToken: 0, endToken: 2)
            decoding.suppressedTokens = [2]
            XCTAssertEqual(NFKMLXSeq2SeqDecoder.generate(streamed, source: ids, decoding: decoding),
                           NFKMLXSeq2SeqDecoder.generate(held, source: ids, decoding: decoding), "half \(half): greedy")
            decoding.beams = 3
            XCTAssertEqual(NFKMLXSeq2SeqDecoder.generate(streamed, source: ids, decoding: decoding),
                           NFKMLXSeq2SeqDecoder.generate(held, source: ids, decoding: decoding), "half \(half): beams")
            try streamed.verifyStream()

            let decoderStream = try XCTUnwrap(streamed.decoderStream)
            XCTAssertGreaterThan(decoderStream.readStatistics.bytes, 5 * decoderStream.bytesPerPass,
                                 "every decoder pass reads every streamed decoder block")
            for index in [1, 2, 3] {
                XCTAssertTrue(streamed.encoder.block[index].parameters().flattened().allSatisfy { $0.1.size == 0 },
                              "encoder block \(index) holds nothing between its turns")
            }
            XCTAssertFalse(streamed.encoder.block[0].parameters().flattened().contains { $0.1.size == 0 })
            XCTAssertFalse(streamed.decoder.block[2].parameters().flattened().contains { $0.1.size == 0 })
        }
    }

    func testAutomaticHoldsAT5ThatFitsAndOrdersTheDecoderFirst() throws {
        try requireMLXRuntime()
        let directory = try t5Release()
        defer { try? FileManager.default.removeItem(at: directory) }
        let net = try NFKMLXMADLAD.network(directoryURL: directory)
        XCTAssertNil(net.encoderStream, "a release that fits loads whole, as it always has")
        XCTAssertNil(net.decoderStream)

        let full = try NFKMLXMADLAD.footprint(directory: directory, half: false)
        let half = try NFKMLXMADLAD.footprint(directory: directory, half: true)
        XCTAssertEqual(full.whole, 2 * half.whole, "a float32 release halves at bfloat16")
        XCTAssertEqual(full.stage.layerBytes.count, 6, "blocks 1 to 3 of each stack")
        XCTAssertGreaterThan(full.stage.layerBytes[0], full.stage.layerBytes[5],
                             "decoder blocks, which carry a cross-attention, come first")
        XCTAssertEqual(try NFKMLXMADLAD.streamedBlocks(directory: directory, half: false, residency: .streamed,
                                                       budget: 1 << 40).decoder, [],
                       "streamed where everything fits, nothing streams")
        XCTAssertThrowsError(try NFKMLXMADLAD.verifyFits(full.whole, directory: directory, half: false, budget: 1024),
                             "a held load refuses a release known not to fit")
    }

    func testAFailedT5ReadIsReportedRatherThanComputedAround() throws {
        try requireMLXRuntime()
        let directory = try t5Release()
        defer { try? FileManager.default.removeItem(at: directory) }
        let streamed = try streamedT5(directory, half: false, holding: 0)
        let file = directory.appendingPathComponent("model.safetensors")
        let header = try XCTUnwrap(NFKMLXSafetensors.entries(inFile: file).values.map(\.start).min())
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(header))
        try handle.close()

        let decoding = NFKMLXSeq2SeqDecoding(beams: 2, maxTokens: 6, startToken: 0, endToken: 2)
        _ = NFKMLXSeq2SeqDecoder.generate(streamed, source: sourceIds.map { Int($0) }, decoding: decoding)
        XCTAssertThrowsError(try streamed.verifyStream(), "the truncated release cannot be read")
    }

    func testAStreamedT5DoesNotTrain() throws {
        try requireMLXRuntime()
        let directory = try t5Release()
        defer { try? FileManager.default.removeItem(at: directory) }
        let streamed = try streamedT5(directory, half: false, holding: 2)
        let source = MLXArray(sourceIds).reshaped([1, sourceIds.count])
        let target = MLXArray([Int32(9), 10, 11, 2]).reshaped([1, 4])
        XCTAssertThrowsError(try NFKMLXMADLAD.fineTune(streamed, examples: { _ in (source, target) }, steps: 1))
    }

    /// The released MADLAD-400 3B at float32, held and then with every encoder block and half the
    /// decoder's streamed: the same values, element for element, through the encoder output, the
    /// teacher-forced logits, and a greedy translation. The two loads run one after the other, about
    /// 12 GB each.
    func testAReleasedT5StreamsToTheValuesItsHeldLoadComputes() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_MADLAD"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_MADLAD to the madlad400-3b-mt release directory")
        }
        let directory = URL(fileURLWithPath: path)
        func run(_ translator: NFKMLXMADLADTranslator) throws -> (memory: MLXArray, logits: MLXArray, produced: [Int]) {
            let ids = try XCTUnwrap(translator.sourceIds(for: "The quick brown fox jumps over the lazy dog.", target: "de"))
            let source = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
            let memory = translator.net.encode(source)
            let logits = translator.net(source: source, target: MLXArray([Int32(0), 1000, 2000, 3000]).reshaped([1, 4]))
            eval(memory, logits)
            var decoding = translator.defaultDecoding
            decoding.maxTokens = 24
            let produced = NFKMLXSeq2SeqDecoder.generate(translator.net, source: ids, decoding: decoding)
            try translator.net.verifyStream()
            return (memory, logits, produced)
        }
        let held = try run(NFKMLXMADLAD.translator(directoryURL: directory, half: false, residency: .resident))
        NFKMLXGPU.clearCache()

        let footprint = try NFKMLXMADLAD.footprint(directory: directory, half: false)
        let decoderHalf = (footprint.decoderBlocks - 1) / 2
        let budget = NFKMLXResidencyBudget.leastBudget(holding: footprint.stage.streamedMinimumBytes
            + footprint.stage.layerBytes.prefix(decoderHalf).reduce(0, +))
        let net = try NFKMLXMADLAD.network(directoryURL: directory, configuration: nil, half: false, residency: .streamed,
                                           budget: budget)
        let encoderStream = try XCTUnwrap(net.encoderStream), decoderStream = try XCTUnwrap(net.decoderStream)
        XCTAssertEqual(encoderStream.layers.count, footprint.encoderBlocks - 1)
        XCTAssertEqual(decoderStream.layers.count, footprint.decoderBlocks - 1 - decoderHalf)
        let translator = try NFKMLXMADLAD.translator(net: net, directoryURL: directory)
        let backend = NFKMLXTranslationBackend(translator: translator)
        XCTAssertEqual(backend.streamedLayerCount, encoderStream.layers.count + decoderStream.layers.count)
        XCTAssertEqual(backend.streamedBytesPerPass, encoderStream.bytesPerPass + decoderStream.bytesPerPass)
        let streamed = try run(translator)

        XCTAssertEqual(maxDifference(held.memory, streamed.memory), 0, "the encoder output")
        XCTAssertEqual(maxDifference(held.logits, streamed.logits), 0, "the teacher-forced logits")
        XCTAssertEqual(held.produced, streamed.produced, "the greedy translation")
        for (name, stream) in [("encoder", encoderStream), ("decoder", decoderStream)] {
            let (bytes, seconds) = stream.readStatistics
            print("madlad-3b streamed \(name): \(stream.layers.count) blocks, "
                  + String(format: "%.1f GB read at %.2f GB/s", Double(bytes) / 1e9, Double(bytes) / 1e9 / max(seconds, 1e-9)))
        }
    }
}
