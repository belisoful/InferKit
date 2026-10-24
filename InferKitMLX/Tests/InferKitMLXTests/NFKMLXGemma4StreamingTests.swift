//
//  NFKMLXGemma4StreamingTests.swift
//  InferKitMLXTests
//
//  Weight-free tests for streaming the dense Gemma 4 decoders: the E-series stack (per-layer inputs,
//  key-value sharing, keys that are values) that the 31B also runs, and the 12B's unified stack. A random
//  decoder is written out as a sharded bfloat16 release under `model.language_model.`, and the same
//  release loaded with every layer held and with its layers streamed must compute the same values,
//  element for element.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXGemma4StreamingTests: XCTestCase {

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

    private let tokens: [Int] = [3, 17, 42, 99, 7, 61, 12, 5, 88]

    /// Six layers of both kinds over a window shorter than the prompt, with per-layer inputs, full layers
    /// whose keys are their values, and the last two layers sharing the keys and values of their kind.
    private let eSeries = NFKMLXGemmaConfiguration(
        hiddenSize: 64, layerCount: 6, intermediateSize: 96, vocabularySize: 131, headCount: 4,
        keyValueHeadCount: 2, globalKeyValueHeadCount: 1, attentionKeyEqualsValue: true,
        headDimensions: 16, globalHeadDimensions: 32, slidingWindow: 4, globalPartialRotaryFactor: 0.25,
        perLayerInputSize: 16, sharedKeyValueLayers: 2, finalLogitSoftcap: 30,
        layerTypes: [.sliding, .sliding, .full, .sliding, .sliding, .full], perLayerVocabularySize: 131)

    /// The unified stack at six layers, keys that are values on its full layers.
    private let unified = NFKMLXGemmaConfiguration(
        hiddenSize: 64, layerCount: 6, intermediateSize: 96, vocabularySize: 131, headCount: 4,
        keyValueHeadCount: 2, globalKeyValueHeadCount: 1, attentionKeyEqualsValue: true,
        headDimensions: 16, globalHeadDimensions: 32, slidingWindow: 4, globalPartialRotaryFactor: 0.25,
        perLayerInputSize: 0, sharedKeyValueLayers: 0, finalLogitSoftcap: 0,
        layerTypes: [.sliding, .sliding, .full, .sliding, .sliding, .full])

    /// `net`'s parameters written as a two-shard bfloat16 release under `model.language_model.`, with a
    /// vision tower tensor beside them that the decoder does not read.
    private func release(_ net: Module) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-stream-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        MLXRandom.seed(5)
        var stored = net.parameters().flattened().map { key, value in
            // Random norm weights and layer scalars make every one of them count.
            let varied = key.hasSuffix("norm.weight") || key.hasSuffix("layer_scalar")
            return ("model.language_model." + key, (varied ? 1 + MLXRandom.normal(value.shape) * 0.1 : value).asType(.bfloat16))
        }
        stored.append(("model.vision_tower.patch_embedder.weight", MLXRandom.normal([8, 8]).asType(.bfloat16)))
        stored.sort { $0.0 < $1.0 }
        var weightMap = [String: String]()
        for (index, half) in [stored.prefix(stored.count / 2), stored.dropFirst(stored.count / 2)].enumerated() {
            let name = "model-0000\(index + 1)-of-00002.safetensors"
            try save(arrays: Dictionary(half.map { $0 }, uniquingKeysWith: { first, _ in first }),
                     url: directory.appendingPathComponent(name))
            half.forEach { weightMap[$0.0] = name }
        }
        try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    /// The least budget that holds `directory`'s decoder with only its first `held` layers.
    private func budget(_ directory: URL, precision: NFKMLXWeightPrecision, holding held: Int) throws -> Int {
        let stage = try NFKMLXLayerStream.footprint(
            directory: directory, precision: precision,
            includes: { NFKMLXGemmaLanguage.decoderKey(of: $0) != nil },
            layerOf: { NFKMLXGemmaLanguage.decoderKey(of: $0).flatMap { NFKMLXLayerStream.layer(ofDecoderName: $0) }?.layer })
        return NFKMLXResidencyBudget.leastBudget(holding: stage.streamedMinimumBytes + stage.layerBytes.prefix(held).reduce(0, +))
    }

    private func eSeriesNet(_ directory: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                            streams: Bool = true, holding held: Int? = nil) throws -> NFKMLXGemmaNet {
        let net = NFKMLXGemmaLanguage.makeNet(eSeries)
        try NFKMLXGemmaLanguage.loadWeights(into: net, fromDirectory: directory, precision: precision, residency: residency,
                                            streams: streams,
                                            budget: try held.map { try budget(directory, precision: precision, holding: $0) }
                                                ?? NFKMLXResidencyBudget.current())
        return net
    }

    private func unifiedNet(_ directory: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                            holding held: Int? = nil) throws -> NFKMLXGemma4UnifiedNet {
        let net = NFKMLXGemmaLanguage.makeUnifiedNet(unified)
        try NFKMLXGemmaLanguage.loadUnifiedWeights(into: net, fromDirectory: directory, precision: precision,
                                                   residency: residency, streams: true,
                                                   budget: try held.map { try budget(directory, precision: precision, holding: $0) }
                                                       ?? NFKMLXResidencyBudget.current())
        return net
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private var input: MLXArray { MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count]) }

    private func assertHoldsNothing(_ layers: [Module], _ stream: NFKMLXLayerStream?, _ label: String) throws {
        for index in try XCTUnwrap(stream, label).layers {
            XCTAssertTrue(layers[index].parameters().flattened().allSatisfy { $0.1.size == 0 },
                          "\(label): layer \(index) holds nothing between its turns")
        }
    }

    func testAStreamedGemma4DecoderComputesWhatAHeldOneDoes() throws {
        try requireMLXRuntime()
        let directory = try release(NFKMLXGemmaLanguage.makeNet(eSeries))
        for precision in [NFKMLXWeightPrecision.float32, .checkpoint] {
            let label = "\(precision)"
            let held = try eSeriesNet(directory, precision: precision, residency: .resident)
            let streamed = try eSeriesNet(directory, precision: precision, residency: .streamed, holding: 2)
            XCTAssertNil(held.layerStream)
            XCTAssertEqual(streamed.layerStream?.layers, [2, 3, 4, 5], "\(label): the first two layers are held")

            XCTAssertEqual(maxDifference(held(input), streamed(input)), 0, "\(label): the logits")
            for (index, (a, b)) in zip(held.hiddenStates(input), streamed.hiddenStates(input)).enumerated() {
                XCTAssertEqual(maxDifference(a, b), 0, "\(label): the state after layer \(index)")
            }
            XCTAssertEqual(maxDifference(held.logits(fromEmbeddings: held.embed(input), tokens: input),
                                         streamed.logits(fromEmbeddings: streamed.embed(input), tokens: input)), 0,
                           "\(label): the logits over supplied embeddings, the multimodal path")
            try streamed.verifyStream()
            try assertHoldsNothing(streamed.layers, streamed.layerStream, label)
        }
    }

    func testAStreamedUnifiedDecoderComputesWhatAHeldOneDoes() throws {
        try requireMLXRuntime()
        let directory = try release(NFKMLXGemmaLanguage.makeUnifiedNet(unified))
        for precision in [NFKMLXWeightPrecision.float32, .checkpoint] {
            let label = "\(precision)"
            let held = try unifiedNet(directory, precision: precision, residency: .resident)
            let streamed = try unifiedNet(directory, precision: precision, residency: .streamed, holding: 3)
            XCTAssertNil(held.layerStream)
            XCTAssertEqual(streamed.layerStream?.layers, [3, 4, 5], "\(label): the first three layers are held")
            XCTAssertEqual(maxDifference(held(input), streamed(input)), 0, "\(label): the logits")
            for (index, (a, b)) in zip(held.hiddenStates(input), streamed.hiddenStates(input)).enumerated() {
                XCTAssertEqual(maxDifference(a, b), 0, "\(label): the state after layer \(index)")
            }
            try streamed.verifyStream()
            try assertHoldsNothing(streamed.layers, streamed.layerStream, label)
        }
    }

    func testOnlyACallerThatChecksTheStreamStreamsAndAMixtureNeverDoes() throws {
        try requireMLXRuntime()
        let directory = try release(NFKMLXGemmaLanguage.makeNet(eSeries))
        XCTAssertNil(try eSeriesNet(directory, precision: .float32, residency: .streamed, streams: false, holding: 1).layerStream,
                     "a caller that never checks a stream holds the decoder")
        XCTAssertNil(try eSeriesNet(directory, precision: .float32, residency: .automatic).layerStream,
                     "a release that fits loads whole, as it always has")

        let mixture = NFKMLXGemmaLanguage.makeNet(.tinyMixture)
        let mixtureDirectory = try release(mixture)
        let reloaded = NFKMLXGemmaLanguage.makeNet(.tinyMixture)
        try NFKMLXGemmaLanguage.loadWeights(into: reloaded, fromDirectory: mixtureDirectory, precision: .float32,
                                            residency: .streamed, streams: true,
                                            budget: try budget(mixtureDirectory, precision: .float32, holding: 0))
        XCTAssertNil(reloaded.layerStream, "a mixture of experts pages its experts; its layers never stream")
    }

    func testAStreamPlanCountsTheTowersHeldBesideTheDecoder() throws {
        try requireMLXRuntime()
        let directory = try release(NFKMLXGemmaLanguage.makeNet(eSeries))
        let stage = try NFKMLXLayerStream.footprint(
            directory: directory, precision: .float32, includes: { NFKMLXGemmaLanguage.decoderKey(of: $0) != nil },
            layerOf: { NFKMLXGemmaLanguage.decoderKey(of: $0).flatMap { NFKMLXLayerStream.layer(ofDecoderName: $0) }?.layer })
        let budget = try budget(directory, precision: .float32, holding: 3)
        XCTAssertEqual(try NFKMLXGemmaLanguage.streamedLayers(directory: directory, precision: .float32,
                                                              residency: .streamed, budget: budget), [3, 4, 5])
        XCTAssertEqual(try NFKMLXGemmaLanguage.streamedLayers(directory: directory, precision: .float32, residency: .streamed,
                                                              budget: budget, besides: stage.layerBytes[0]),
                       [2, 3, 4, 5], "a tower the size of a layer leaves room for one layer fewer")
    }

    /// A Gemma tokenizer over the lowercase letters (ids 5 … 30), a distinct piece for every other id the
    /// decoders' 131-entry vocabulary can produce, and the end marker.
    private func tokenizer() throws -> NFKMLXGemmaTokenizer {
        try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: try tokenizerURL()))
    }

    /// The `tokenizer.json` of ``tokenizer()``.
    private func tokenizerURL() throws -> URL {
        var vocabulary: [String: Int] = ["<eos>": 1, "<bos>": 2, "<unk>": 3, "\u{2581}": 4]
        for (index, letter) in "abcdefghijklmnopqrstuvwxyz".enumerated() {
            vocabulary[String(letter)] = 5 + index
        }
        for id in [0] + Array(31 ..< 131) {
            vocabulary["[\(id)]"] = id
        }
        let json: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [] as [Any], "unk_token": "<unk>"],
            "added_tokens": [["content": "<eos>", "id": 1, "special": true], ["content": "<bos>", "id": 2, "special": true]],
        ]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gemma4-stream-\(UUID().uuidString).json")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return url
    }

    private func reply(_ backend: NFKMLXGemmaBackend) throws -> String? {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "abc"],
                                          parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 6])
        return try backend.runInference(for: request).output(forKey: NFKOutputText) as? String
    }

    func testAFailedReadFailsTheRequestAndTheNextRequestReadsTheReleaseAgain() throws {
        try requireMLXRuntime()
        let directory = try release(NFKMLXGemmaLanguage.makeNet(eSeries))
        let heldNet = try eSeriesNet(directory, precision: .checkpoint, residency: .resident)
        let streamedNet = try eSeriesNet(directory, precision: .checkpoint, residency: .streamed, holding: 1)
        let held = NFKMLXGemmaBackend(logits: { heldNet($0) }, tokenizer: try tokenizer(), identifier: "gemma4")
        let streamed = NFKMLXGemmaBackend(logits: { streamedNet($0) }, tokenizer: try tokenizer(), identifier: "gemma4",
                                          layerStream: streamedNet.layerStream, modules: [streamedNet])
        XCTAssertEqual(streamed.streamedLayerCount, 5)
        XCTAssertEqual(streamed.streamedBytesPerPass, streamedNet.layerStream?.bytesPerPass)
        XCTAssertEqual(streamed.modelInfo[NFKModelInfoResidency] as? String, NFKModelResidencyStreamed)
        XCTAssertEqual(try reply(streamed), try reply(held), "the streamed reply")

        let shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            .map { directory.appendingPathComponent($0) }
        let originals = try shards.map { try Data(contentsOf: $0) }
        for file in shards {
            let start = try XCTUnwrap(NFKMLXSafetensors.entries(inFile: file).values.map(\.start).min())
            let handle = try FileHandle(forWritingTo: file)
            try handle.truncate(atOffset: UInt64(start))
            try handle.close()
        }
        XCTAssertThrowsError(try reply(streamed), "the truncated release cannot be read")
        for (file, original) in zip(shards, originals) {
            try original.write(to: file)
        }
        XCTAssertEqual(try reply(streamed), try reply(held), "the next request reads every layer again")
        try streamedNet.verifyStream()
    }

    // MARK: Drafting

    /// `eSeries` as a release's `config.json` states it.
    private var eSeriesConfig: [String: Any] {
        let c = eSeries
        return ["model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"], "text_config": [
            "model_type": "gemma4_text", "hidden_size": c.hiddenSize, "num_hidden_layers": c.layerCount,
            "intermediate_size": c.intermediateSize, "vocab_size": c.vocabularySize, "rms_norm_eps": c.rmsEpsilon,
            "num_attention_heads": c.headCount, "num_key_value_heads": c.keyValueHeadCount,
            "num_global_key_value_heads": c.globalKeyValueHeadCount ?? c.keyValueHeadCount,
            "attention_k_eq_v": c.attentionKeyEqualsValue, "head_dim": c.headDimensions,
            "global_head_dim": c.globalHeadDimensions, "sliding_window": c.slidingWindow,
            "rope_parameters": ["full_attention": ["partial_rotary_factor": c.globalPartialRotaryFactor]],
            "hidden_size_per_layer_input": c.perLayerInputSize, "num_kv_shared_layers": c.sharedKeyValueLayers,
            "final_logit_softcapping": c.finalLogitSoftcap, "layer_types": c.layerTypes.map(\.rawValue),
            "vocab_size_per_layer_input": c.perLayerVocabularySize,
        ] as [String: Any]]
    }

    /// A release of a random `eSeries` decoder seeded with `seed`, with its `config.json` and a tokenizer.
    private func eSeriesRelease(seed: UInt64) throws -> URL {
        MLXRandom.seed(seed)
        let directory = try release(NFKMLXGemmaLanguage.makeNet(eSeries))
        try JSONSerialization.data(withJSONObject: eSeriesConfig).write(to: directory.appendingPathComponent("config.json"))
        try FileManager.default.copyItem(at: try tokenizerURL(), to: directory.appendingPathComponent("tokenizer.json"))
        return directory
    }

    private func reply(_ backend: NFKMLXGemmaBackend, draftTokens: Int?, temperature: Float = 0) throws -> String? {
        var parameters: [String: Any] = [NFKParameterTemperature: temperature, NFKParameterMaxTokens: 12, NFKParameterSeed: 3]
        parameters[NFKMLXGenerationParameterKey.draftTokens] = draftTokens
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "abc"], parameters: parameters)
        return try backend.runInference(for: request).output(forKey: NFKOutputText) as? String
    }

    func testADraftedReplyIsTheStreamedDecodersGreedyReplyInFewerPasses() throws {
        try requireMLXRuntime()
        let directory = try release(NFKMLXGemmaLanguage.makeNet(eSeries))
        let target = try eSeriesNet(directory, precision: .float32, residency: .streamed, holding: 1)
        let twin = try eSeriesNet(directory, precision: .float32, residency: .resident)
        MLXRandom.seed(77)
        let stranger = NFKMLXGemmaLanguage.makeNet(eSeries)
        let plain = NFKMLXGemmaBackend(logits: { target($0) }, tokenizer: try tokenizer(), identifier: "gemma4",
                                       layerStream: target.layerStream)
        let greedy = try reply(plain, draftTokens: nil)
        XCTAssertGreaterThan(greedy?.count ?? 0, 0, "the reply has text to compare")
        XCTAssertFalse(plain.hasDraftModel)
        XCTAssertFalse(plain.supportedParameterKeys.contains(NFKMLXGenerationParameterKey.draftTokens))

        let agreeing = NFKMLXGemmaBackend(logits: { target($0) }, tokenizer: try tokenizer(), identifier: "gemma4",
                                          layerStream: target.layerStream, draft: { twin($0) })
        XCTAssertTrue(agreeing.hasDraftModel)
        XCTAssertTrue(agreeing.supportedParameterKeys.contains(NFKMLXGenerationParameterKey.draftTokens))
        XCTAssertEqual(try reply(agreeing, draftTokens: 3), greedy, "a draft that agrees keeps the greedy reply")
        let report = agreeing.lastSpeculativeReport
        XCTAssertEqual(report.accepted, report.proposed, "the decoder keeps every proposal of its own twin")
        XCTAssertLessThan(report.rounds, 12, "four tokens a round take fewer passes of the release than tokens")

        let disagreeing = NFKMLXGemmaBackend(logits: { target($0) }, tokenizer: try tokenizer(), identifier: "gemma4",
                                             layerStream: target.layerStream, draft: { stranger($0) })
        XCTAssertEqual(try reply(disagreeing, draftTokens: 3), greedy, "rejected proposals leave the greedy reply")
        XCTAssertLessThan(disagreeing.lastSpeculativeReport.accepted, disagreeing.lastSpeculativeReport.proposed)
        XCTAssertEqual(try reply(disagreeing, draftTokens: 0), greedy, "0 draft tokens decodes without the draft")
        XCTAssertNotNil(try reply(disagreeing, draftTokens: 3, temperature: 0.8), "a sampled drafted reply")
        try target.verifyStream()
    }

    func testTheDraftedFactoryQuantizesTheDraftAndKeepsTheGreedyReply() throws {
        try requireMLXRuntime()
        let directory = try eSeriesRelease(seed: 5)
        let draftDirectory = try eSeriesRelease(seed: 91)
        let plain = try XCTUnwrap(try NFKMLXGemmaLanguage.backend(directoryURL: directory, residency: .resident)
                                  as? NFKMLXGemmaBackend)
        let drafted = try NFKMLXGemmaLanguage.backend(directoryURL: directory, draftDirectoryURL: draftDirectory,
                                                      residency: .resident, draftBits: 4)
        XCTAssertTrue(drafted.hasDraftModel)
        XCTAssertEqual(try reply(drafted, draftTokens: 3), try reply(plain, draftTokens: nil),
                       "a 4-bit draft of another release leaves the greedy reply")
        XCTAssertGreaterThan(drafted.lastSpeculativeReport.rounds, 0)
    }

    // MARK: Released weights

    private func release(_ key: String) throws -> URL {
        guard let path = NFKMLXValidationConfig.environment[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key) to the release directory")
        }
        return URL(fileURLWithPath: path)
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).reshaped([-1]), y = b.asType(.float32).reshaped([-1])
        return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))).item(Float.self)
    }

    /// The released Gemma 4 E4B at its own bfloat16, held and then with half its layers streamed: the same
    /// logits and every layer's state, element for element. The two loads run one after the other, about
    /// 16 GB each.
    func testAReleasedGemma4E4BStreamsToTheValuesItsHeldLoadComputes() throws {
        try requireMLXRuntime()
        let directory = try release("IK_VAL_GEMMA4_E4B")
        let geometry = try NFKMLXGemmaLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let prompt = MLXArray([2, 818, 5279, 529, 7001, 563].map { Int32($0) }).reshaped([1, 6])
        func run(_ net: NFKMLXGemmaNet) -> (states: [MLXArray], logits: MLXArray) {
            let (states, logits) = net.pass(prompt)
            eval(states + [logits])
            return (states, logits)
        }
        let heldNet = NFKMLXGemmaLanguage.makeNet(geometry)
        try NFKMLXGemmaLanguage.loadWeights(into: heldNet, fromDirectory: directory, precision: .checkpoint, residency: .resident)
        let held = run(heldNet)
        NFKMLXGPU.clearCache()

        let net = NFKMLXGemmaLanguage.makeNet(geometry)
        try NFKMLXGemmaLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint, residency: .streamed,
                                            streams: true, budget: try budget(directory, precision: .checkpoint,
                                                                              holding: geometry.layerCount / 2))
        let stream = try XCTUnwrap(net.layerStream)
        XCTAssertEqual(stream.layers.count, geometry.layerCount - geometry.layerCount / 2)
        let streamed = run(net)
        try net.verifyStream()
        XCTAssertEqual(maxDifference(held.logits, streamed.logits), 0, "the logits")
        for (index, (a, b)) in zip(held.states, streamed.states).enumerated() {
            XCTAssertEqual(maxDifference(a, b), 0, "the state after layer \(index)")
        }
        let (bytes, seconds) = stream.readStatistics
        print("gemma-4-e4b streamed: \(stream.layers.count) of \(geometry.layerCount) layers, "
              + String(format: "%.1f GB read at %.2f GB/s", Double(bytes) / 1e9, Double(bytes) / 1e9 / max(seconds, 1e-9)))
    }

    /// A released Gemma 4 decoder streamed at its own bfloat16 against the reference's record of the same
    /// release run one layer at a time (`run_reference.py dense_layerwise`), in one pass over the prompt and
    /// the reference's greedy continuation: the last prompt position's state after every layer but the last,
    /// the logits at every prompt position, and the argmax at every position that predicts a continuation
    /// token. Where each of those argmaxes is the next recorded token, greedy decoding reproduces the
    /// continuation token for token.
    private func checkStreamedRelease(release releaseKey: String, record recordKey: String,
                                      pass: (URL, [Int]) throws -> (states: [MLXArray], logits: MLXArray, stream: NFKMLXLayerStream?)) throws {
        try requireMLXRuntime()
        let directory = try release(releaseKey)
        guard let path = NFKMLXValidationConfig.environment[recordKey] else {
            throw XCTSkip("set \(recordKey) (run_reference.py dense_layerwise)")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self).map(Int.init)
        let continuation = try XCTUnwrap(record["continuation"]).asArray(Int32.self).map(Int.init)
        let reference = try XCTUnwrap(record["output"])

        let run = try pass(directory, tokens + continuation)
        let stream = try XCTUnwrap(run.stream, "the release does not fit whole, so its layers stream")
        var worst = (cosine: Float(1), layer: 0)
        for index in 0 ..< run.states.count - 1 {
            let expected = try XCTUnwrap(record["hidden_last.\(index)"], "the record's state after layer \(index)")
            let measured = cosine(run.states[index][0, tokens.count - 1], expected)
            if measured < worst.cosine { worst = (measured, index) }
        }
        let logits = run.logits[0]
        let promptLogits = logits[0 ..< tokens.count]
        let mine = promptLogits.argMax(axis: -1).asArray(Int32.self)
        let theirs = reference.argMax(axis: -1).asArray(Int32.self)
        let agree = zip(mine, theirs).filter { $0 == $1 }.count
        let logitCosine = cosine(promptLogits, reference)
        let predicted = logits[(tokens.count - 1) ..< (tokens.count + continuation.count - 1)]
            .argMax(axis: -1).asArray(Int32.self).map(Int.init)
        let (bytes, seconds) = stream.readStatistics
        print("\(releaseKey) streamed: \(stream.layers.count) of \(run.states.count - 1) layers, logit cosine \(logitCosine), "
              + "argmax \(agree)/\(tokens.count), worst layer \(worst.cosine) at \(worst.layer), greedy "
              + "\(predicted == continuation ? "exact" : "\(predicted) vs \(continuation)"), "
              + String(format: "%.1f GB read at %.2f GB/s", Double(bytes) / 1e9, Double(bytes) / 1e9 / max(seconds, 1e-9)))
        XCTAssertGreaterThan(worst.cosine, 0.999, "the state after layer \(worst.layer)")
        XCTAssertGreaterThan(logitCosine, 0.999, "the prompt's logits")
        XCTAssertGreaterThanOrEqual(agree, tokens.count - 1, "the prompt's argmax")
        XCTAssertEqual(predicted, continuation, "each continuation token is the argmax after the ones before it")
    }

    private func input(_ sequence: [Int]) -> MLXArray {
        MLXArray(sequence.map { Int32($0) }).reshaped([1, sequence.count])
    }

    func testTheStreamedGemma4_31BMatchesItsReference() throws {
        try checkStreamedRelease(release: "IK_VAL_GEMMA4_31B", record: "IK_PARITY_GEMMA4_31B") { directory, sequence in
            let net = NFKMLXGemmaLanguage.makeNet(
                try NFKMLXGemmaLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json")))
            try NFKMLXGemmaLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint,
                                                residency: .streamed, streams: true)
            let (states, logits) = net.pass(self.input(sequence))
            eval(states + [logits])
            try net.verifyStream()
            return (states, logits, net.layerStream)
        }
    }

    /// The 12B at its released bfloat16 is about 24 GB, which a large working set holds whole, so the plan
    /// holds half its layers and streams the rest wherever the test runs.
    func testTheStreamedGemma4_12BMatchesItsReference() throws {
        try checkStreamedRelease(release: "IK_VAL_GEMMA4_12B", record: "IK_PARITY_GEMMA4_12B") { directory, sequence in
            let geometry = try NFKMLXGemmaLanguage.unifiedConfiguration(fromHuggingFace: directory.appendingPathComponent("config.json"))
            let net = NFKMLXGemmaLanguage.makeUnifiedNet(geometry)
            try NFKMLXGemmaLanguage.loadUnifiedWeights(into: net, fromDirectory: directory, precision: .checkpoint,
                                                       residency: .streamed, streams: true,
                                                       budget: try self.budget(directory, precision: .checkpoint,
                                                                               holding: geometry.layerCount / 2))
            let (states, logits) = net.pass(self.input(sequence))
            eval(states + [logits])
            try net.verifyStream()
            return (states, logits, net.layerStream)
        }
    }

    /// The streamed 31B decoding the reference's greedy continuation with the E2B drafting at 4 bits: the
    /// drafted output is the 31B's own greedy output, in fewer passes of the release than tokens.
    func testTheStreamedGemma4_31BDecodesItsReferenceContinuationWithADraft() throws {
        try requireMLXRuntime()
        let directory = try release("IK_VAL_GEMMA4_31B")
        let draftDirectory = try release("IK_VAL_GEMMA4")
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_GEMMA4_31B"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_31B (run_reference.py dense_layerwise)")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self).map(Int.init)
        let continuation = try XCTUnwrap(record["continuation"]).asArray(Int32.self).map(Int.init)
        let backend = try NFKMLXGemmaLanguage.backend(directoryURL: directory, draftDirectoryURL: draftDirectory,
                                                      precision: .checkpoint, residency: .streamed)
        XCTAssertGreaterThan(backend.streamedLayerCount, 0, "the 31B streams beside the held draft")
        let produced = try backend.generate(tokens, sampling: NFKMLXStepSampling(release: nil), maxTokens: continuation.count,
                                            seed: nil, stops: [], draftTokens: 4).tokens
        let report = backend.lastSpeculativeReport
        print("IK_VAL_GEMMA4_31B drafted: \(produced.count) tokens in \(report.rounds) rounds, \(report.accepted)/"
              + "\(report.proposed) proposals kept, \(backend.streamedLayerCount) layers streamed, "
              + String(format: "%.1f GB a pass", Double(backend.streamedBytesPerPass) / 1e9))
        XCTAssertEqual(produced, continuation, "the drafted output is the reference's greedy continuation")
        XCTAssertLessThan(report.rounds, produced.count, "the draft saves passes")
    }
}
