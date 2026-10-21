//
//  NFKMLXDenseStreamingTests.swift
//  InferKitMLXTests
//
//  Weight-free tests for streaming the dense decoder Qwen3, Mistral, Qwen3-VL, and Pixtral share: a random
//  decoder is written out as a sharded bfloat16 release, and the same release loaded with every layer
//  held and with its layers streamed must compute the same values, element for element. The released
//  tests hold a streamed load to the held one on weights that fit both ways; the streamed Pixtral 12B is
//  measured against its reference in NFKMLXPixtralTests.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXDenseStreamingTests: XCTestCase {

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

    /// A six-layer random decoder, tied or with its own output projection, at `configuration`'s geometry.
    private func geometry(tied: Bool) -> NFKMLXLanguageConfiguration {
        var configuration = NFKMLXLanguageConfiguration.tiny
        configuration.layerCount = 6
        // MLX quantizes a cache in groups of 32 or more, so a quantized cache needs heads that wide.
        configuration.headDimensions = 32
        configuration.tiesWordEmbeddings = tied
        return configuration
    }

    /// `geometry` written as a two-shard bfloat16 release with its index and `config.json`, each module key
    /// renamed by `releaseKey` as a multimodal release nests its decoder.
    private func release(_ geometry: NFKMLXLanguageConfiguration,
                         releaseKey: (String) -> String = { $0 }) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dense-stream-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let config: [String: Any] = [
            "model_type": "qwen3", "hidden_size": geometry.hiddenSize, "num_hidden_layers": geometry.layerCount,
            "num_attention_heads": geometry.headCount, "num_key_value_heads": geometry.keyValueHeadCount,
            "head_dim": geometry.headDimensions, "intermediate_size": geometry.intermediateSize,
            "vocab_size": geometry.vocabularySize, "rope_theta": geometry.ropeTheta,
            "tie_word_embeddings": geometry.tiesWordEmbeddings,
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        MLXRandom.seed(5)
        let net = NFKMLXLanguage.makeNet(geometry)
        let stored = net.parameters().flattened().map { key, value in
            // Random norm weights make every layer's norms count.
            (releaseKey(key), (key.hasSuffix("norm.weight") ? 1 + MLXRandom.normal(value.shape) * 0.1 : value).asType(.bfloat16))
        }.sorted { $0.0 < $1.0 }
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
    private func budget(_ directory: URL, precision: NFKMLXWeightPrecision, holding held: Int,
                        moduleKey: @escaping (String) -> String? = { $0 }) throws -> Int {
        let stage = try NFKMLXLayerStream.footprint(directory: directory, precision: precision,
                                                    includes: { moduleKey($0) != nil },
                                                    layerOf: { moduleKey($0).flatMap(NFKMLXLanguage.streamedLayer(ofModuleKey:))?.layer })
        return NFKMLXResidencyBudget.leastBudget(holding: stage.streamedMinimumBytes + stage.layerBytes.prefix(held).reduce(0, +))
    }

    private func loaded(_ directory: URL, _ geometry: NFKMLXLanguageConfiguration, precision: NFKMLXWeightPrecision,
                        residency: NFKMLXResidency, holding held: Int? = nil) throws -> NFKMLXLanguageNet {
        let net = NFKMLXLanguage.makeNet(geometry)
        try NFKMLXLanguage.loadWeights(into: net, fromDirectory: directory, precision: precision, residency: residency,
                                       budget: try held.map { try budget(directory, precision: precision, holding: $0) }
                                           ?? NFKMLXResidencyBudget.current())
        return net
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func greedy(_ configure: (inout NFKMLXGenerationOptions) -> Void = { _ in }) -> NFKMLXGenerationOptions {
        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = 8
        configure(&options)
        return options
    }

    func testAStreamedDenseDecoderComputesWhatAHeldOneDoes() throws {
        try requireMLXRuntime()
        for tied in [true, false] {
            let geometry = geometry(tied: tied)
            let directory = try release(geometry)
            for precision in [NFKMLXWeightPrecision.float32, .checkpoint] {
                let label = "\(tied ? "tied" : "untied") \(precision)"
                let held = try loaded(directory, geometry, precision: precision, residency: .resident)
                let streamed = try loaded(directory, geometry, precision: precision, residency: .streamed, holding: 2)
                XCTAssertNil(held.layerStream)
                XCTAssertEqual(streamed.layerStream?.layers, [2, 3, 4, 5], "\(label): the first two layers are held")

                let input = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
                XCTAssertEqual(maxDifference(held(input), streamed(input)), 0, "\(label): the prefill logits")
                for (index, (a, b)) in zip(held.layerStates(input), streamed.layerStates(input)).enumerated() {
                    XCTAssertEqual(maxDifference(a, b), 0, "\(label): the state after layer \(index)")
                }
                // A plain cache, a quantized one, and a window that trims between reads.
                for options in [greedy(), greedy { $0.cacheQuantization = .init(bits: 8, groupSize: 32, keyAxis: .headDimension) },
                                greedy { $0.contextWindow = 6 }] {
                    XCTAssertEqual(streamed.generate(prompt: tokens, options: options),
                                   held.generate(prompt: tokens, options: options), "\(label): the greedy continuation")
                }
                try streamed.verifyStream()

                let stream = try XCTUnwrap(streamed.layerStream)
                for index in stream.layers {
                    XCTAssertTrue(streamed.model.layers[index].parameters().flattened().allSatisfy { $0.1.size == 0 },
                                  "\(label): layer \(index) holds nothing between its turns")
                }
            }
        }
    }

    func testAStreamedDecoderVerifiesADraftsProposals() throws {
        try requireMLXRuntime()
        let geometry = geometry(tied: true)
        let directory = try release(geometry)
        let held = try loaded(directory, geometry, precision: .float32, residency: .resident)
        let streamed = try loaded(directory, geometry, precision: .float32, residency: .streamed, holding: 1)
        MLXRandom.seed(9)
        let draft = NFKMLXLanguage.makeNet(.tiny)
        let options = greedy { $0.draftTokens = 3 }
        XCTAssertEqual(streamed.generate(prompt: tokens, options: options, draft: draft),
                       held.generate(prompt: tokens, options: greedy()), "drafting keeps the streamed output")
        try streamed.verifyStream()
    }

    func testAutomaticHoldsADenseReleaseThatFitsAndAMixtureIsNeverStreamed() throws {
        try requireMLXRuntime()
        let geometry = geometry(tied: true)
        let directory = try release(geometry)
        XCTAssertNil(try loaded(directory, geometry, precision: .float32, residency: .automatic).layerStream,
                     "a release that fits loads whole, as it always has")
        XCTAssertEqual(try NFKMLXLanguage.streamedLayers(directory: directory, precision: .float32, residency: .streamed,
                                                         budget: 1 << 40, moduleKey: { $0 }), [],
                       "streamed where everything fits, nothing streams")

        let mixture = NFKMLXLanguage.makeNet(.tinyMixture)
        let mixtureDirectory = try release(.tinyMixture)
        try NFKMLXLanguage.loadWeights(into: mixture, fromDirectory: mixtureDirectory, precision: .float32,
                                       residency: .streamed, budget: try budget(mixtureDirectory, precision: .float32, holding: 0))
        XCTAssertNil(mixture.layerStream, "a mixture of experts pages its experts; its layers never stream")
    }

    func testAFailedReadStopsDecodingAndARetryReadsTheReleaseAgain() throws {
        try requireMLXRuntime()
        let geometry = geometry(tied: true)
        let directory = try release(geometry)
        let held = try loaded(directory, geometry, precision: .checkpoint, residency: .resident)
        let streamed = try loaded(directory, geometry, precision: .checkpoint, residency: .streamed, holding: 1)
        let shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            .map { directory.appendingPathComponent($0) }
        let originals = try shards.map { try Data(contentsOf: $0) }
        for file in shards {
            let start = try XCTUnwrap(NFKMLXSafetensors.entries(inFile: file).values.map(\.start).min())
            let handle = try FileHandle(forWritingTo: file)
            try handle.truncate(atOffset: UInt64(start))
            try handle.close()
        }
        XCTAssertEqual(streamed.generate(prompt: tokens, options: greedy()), [], "decoding stops at the failed read")
        XCTAssertThrowsError(try streamed.verifyStream(), "the truncated release cannot be read")
        XCTAssertEqual(streamed.generate(prompt: tokens, options: greedy { $0.draftTokens = 3 },
                                         draft: NFKMLXLanguage.makeNet(.tiny)), [],
                       "drafted decoding stops too")

        for (file, original) in zip(shards, originals) {
            try original.write(to: file)
        }
        try XCTUnwrap(streamed.layerStream).clearFailure()
        let input = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
        XCTAssertEqual(maxDifference(held(input), streamed(input)), 0, "the retried pass reads every layer again")
        try streamed.verifyStream()
    }

    func testOnlyACallerThatChecksTheStreamStreams() throws {
        try requireMLXRuntime()
        let geometry = geometry(tied: false)
        // Qwen3-VL nests the stack under `model.language_model.` and keeps the head at the top.
        let directory = try release(geometry, releaseKey: { key in
            key.hasPrefix("model.") ? "model.language_model." + key.dropFirst("model.".count) : key
        })
        let moduleKey: (String) -> String? = { key in
            if key.hasPrefix("model.language_model.") { return "model." + String(key.dropFirst("model.language_model.".count)) }
            return key == "lm_head.weight" ? key : nil
        }
        let small = try budget(directory, precision: .float32, holding: 1, moduleKey: moduleKey)
        let held = NFKMLXLanguageNet(geometry)
        try NFKMLXQwen3VL.loadDecoderWeights(into: held, fromDirectory: directory, precision: .float32,
                                             residency: .streamed, budget: small)
        XCTAssertNil(held.layerStream, "an embedder or text encoder never checks a stream, so it holds the decoder")
        let streamed = NFKMLXLanguageNet(geometry)
        try NFKMLXQwen3VL.loadDecoderWeights(into: streamed, fromDirectory: directory, precision: .float32,
                                             residency: .streamed, streams: true, budget: small)
        XCTAssertEqual(streamed.layerStream?.layers, [1, 2, 3, 4, 5])

        let plain = try release(geometry)
        let language = NFKMLXLanguage.makeNet(geometry)
        try NFKMLXLanguage.loadWeights(into: language, fromDirectory: plain, precision: .float32, residency: .streamed,
                                       streams: false, budget: try budget(plain, precision: .float32, holding: 1))
        XCTAssertNil(language.layerStream)
    }

    func testAStreamPlanCountsWhatTheCallerHoldsBesideTheDecoder() throws {
        try requireMLXRuntime()
        let geometry = geometry(tied: true)
        let directory = try release(geometry)
        let stage = try NFKMLXLayerStream.footprint(directory: directory, precision: .float32, includes: { _ in true },
                                                    layerOf: { NFKMLXLanguage.streamedLayer(ofModuleKey: $0)?.layer })
        let budget = try budget(directory, precision: .float32, holding: 3)
        XCTAssertEqual(try NFKMLXLanguage.streamedLayers(directory: directory, precision: .float32, residency: .streamed,
                                                         budget: budget, moduleKey: { $0 }), [3, 4, 5])
        XCTAssertEqual(try NFKMLXLanguage.streamedLayers(directory: directory, precision: .float32, residency: .streamed,
                                                         budget: budget, besides: stage.layerBytes[0], moduleKey: { $0 }),
                       [2, 3, 4, 5], "a tower the size of a layer leaves room for one layer fewer")
    }

    func testAStreamedDenseDecoderNeitherTrainsNorQuantizes() throws {
        try requireMLXRuntime()
        let geometry = geometry(tied: true)
        let streamed = try loaded(try release(geometry), geometry, precision: .float32, residency: .streamed, holding: 1)
        XCTAssertThrowsError(try NFKMLXLanguage.fineTune(
            streamed, examples: { _ in (MLXArray(self.tokens.map { Int32($0) }), 0) }, steps: 1))
        XCTAssertThrowsError(try streamed.requireHeldLayers(for: "quantize at load"))
    }

    /// A Pixtral-style release: the decoder nests under `language_model.` beside a tower it does not read.
    func testAStreamedNestedDecoderComputesWhatAHeldOneDoes() throws {
        try requireMLXRuntime()
        let geometry = geometry(tied: false)
        let directory = try release(geometry, releaseKey: { "language_model." + $0 })
        let moduleKey: (String) -> String? = { key in
            key.hasPrefix("language_model.") ? String(key.dropFirst("language_model.".count)) : nil
        }
        let held = NFKMLXLanguage.makeNet(geometry)
        try NFKMLXWeights.apply(try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, remap: moduleKey), to: held)
        let streamed = NFKMLXLanguage.makeNet(geometry)
        let layers = try NFKMLXLanguage.streamedLayers(directory: directory, precision: .float32, residency: .streamed,
                                                       budget: try budget(directory, precision: .float32, holding: 3,
                                                                          moduleKey: moduleKey),
                                                       moduleKey: moduleKey)
        XCTAssertEqual(layers, [3, 4, 5])
        try NFKMLXLanguage.loadStreamedWeights(into: streamed, fromDirectory: directory, precision: .float32,
                                               streamed: layers, moduleKey: moduleKey)
        let input = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
        XCTAssertEqual(maxDifference(held(input), streamed(input)), 0)
        try streamed.verifyStream()
    }

    // MARK: Released weights

    /// The released Qwen3 4B at its own bfloat16, held and then with half its layers streamed: the same
    /// prefill logits, every layer's state, and greedy continuation, element for element. The two loads
    /// run one after the other, about 8 GB each.
    func testAReleasedQwen3StreamsToTheValuesItsHeldLoadComputes() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_QWEN3_4B"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_QWEN3_4B to the Qwen3-4B release directory")
        }
        let directory = URL(fileURLWithPath: path)
        let geometry = try NFKMLXLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let prompt = [785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]
        func run(_ net: NFKMLXLanguageNet) -> (logits: MLXArray, states: [MLXArray], produced: [Int]) {
            let input = MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])
            let states = net.layerStates(input).map { $0[0, prompt.count - 1] }
            let logits = net(input)[0]
            eval(states + [logits])
            return (logits, states, net.generate(prompt: prompt, options: greedy { $0.maxTokens = 16 }))
        }
        let held = run(try loaded(directory, geometry, precision: .checkpoint, residency: .resident))
        NFKMLXGPU.clearCache()

        let net = try loaded(directory, geometry, precision: .checkpoint, residency: .streamed,
                             holding: geometry.layerCount / 2)
        let stream = try XCTUnwrap(net.layerStream)
        XCTAssertEqual(stream.layers.count, geometry.layerCount - geometry.layerCount / 2)
        let streamed = run(net)
        try net.verifyStream()

        XCTAssertEqual(maxDifference(held.logits, streamed.logits), 0, "the prefill logits")
        for (index, (a, b)) in zip(held.states, streamed.states).enumerated() {
            XCTAssertEqual(maxDifference(a, b), 0, "the state after layer \(index)")
        }
        XCTAssertEqual(held.produced, streamed.produced, "the greedy continuation")
        let (bytes, seconds) = stream.readStatistics
        print("qwen3-4b streamed: \(stream.layers.count) of \(geometry.layerCount) layers, "
              + String(format: "%.1f GB read at %.2f GB/s", Double(bytes) / 1e9, Double(bytes) / 1e9 / max(seconds, 1e-9)))
    }

    /// The released Qwen3-VL 2B decoder at its own bfloat16, held and with half its layers streamed,
    /// over a text-only sequence through the M-RoPE path: the same hidden states and greedy continuation.
    func testAReleasedQwen3VLDecoderStreamsToTheValuesItsHeldLoadComputes() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_QWEN3_VL"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_QWEN3_VL to the Qwen3-VL-2B-Instruct release directory")
        }
        let directory = URL(fileURLWithPath: path)
        let prompt = [151_644, 872, 198, 785, 3974, 13876, 38835, 151_645, 198, 151_644, 77091, 198]
        func run(_ decoder: NFKMLXLanguageNet) -> MLXArray {
            let hidden = NFKMLXQwen3VL.hiddenStates(decoder: decoder, inputIds: prompt, visionFeatures: nil,
                                                    deepstack: [], gridT: 1, gridH: 1, gridW: 1)
            eval(hidden)
            return hidden
        }
        let held = run(try NFKMLXQwen3VL.decoder(directoryURL: directory, precision: .checkpoint, residency: .resident))
        NFKMLXGPU.clearCache()

        let geometry = try NFKMLXQwen3VL.decoderConfiguration(directoryURL: directory)
        let decoder = NFKMLXLanguageNet(geometry)
        let moduleKey: (String) -> String? = { key in
            if key.hasPrefix("model.language_model.") { return "model." + String(key.dropFirst("model.language_model.".count)) }
            return key == "lm_head.weight" ? key : nil
        }
        try NFKMLXQwen3VL.loadDecoderWeights(into: decoder, fromDirectory: directory, precision: .checkpoint,
                                             residency: .streamed, streams: true,
                                             budget: try budget(directory, precision: .checkpoint,
                                                                holding: geometry.layerCount / 2, moduleKey: moduleKey))
        XCTAssertEqual(decoder.layerStream?.layers.count, geometry.layerCount - geometry.layerCount / 2)
        XCTAssertEqual(maxDifference(held, run(decoder)), 0, "the decoder's hidden states")
        try decoder.verifyStream()
    }

    // MARK: Releases larger than the working set

    /// The record a `run_reference.py dense_layerwise` run wrote for a release: the prompt's ids, the
    /// greedy continuation, the logits at every prompt position, and the last prompt position's state
    /// after every layer but the last.
    private func denseRecord(_ key: String) throws -> (tokens: [Int], continuation: [Int], logits: MLXArray,
                                                       arrays: [String: MLXArray]) {
        guard let path = NFKMLXValidationConfig.environment[key] else {
            throw XCTSkip("set \(key) (run_reference.py dense_layerwise)")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        return (try XCTUnwrap(record["tokens"]).asArray(Int32.self).map(Int.init),
                try XCTUnwrap(record["continuation"]).asArray(Int32.self).map(Int.init),
                try XCTUnwrap(record["output"]), record)
    }

    private func release(_ key: String) throws -> URL {
        guard let path = NFKMLXValidationConfig.environment[key] else {
            throw XCTSkip("set \(key) to the release directory")
        }
        return URL(fileURLWithPath: path)
    }

    /// A released dense decoder streamed at its own bfloat16 against the reference's record of the same
    /// release run one layer at a time (`run_reference.py dense_layerwise`), in one pass over the prompt
    /// and the reference's greedy continuation: the last prompt position's state after every layer but
    /// the last, the logits at every prompt position, and the argmax at every position that predicts a
    /// continuation token. Where each of those argmaxes is the next recorded token, greedy decoding
    /// reproduces the continuation token for token, so one read of the release checks what decoding it
    /// would check in one read per token.
    private func checkStreamedRelease(release releaseKey: String, record recordKey: String,
                                      load: (URL) throws -> NFKMLXLanguageNet) throws {
        try requireMLXRuntime()
        let directory = try release(releaseKey)
        let record = try denseRecord(recordKey)
        let tokens = record.tokens

        let net = try load(directory)
        let stream = try XCTUnwrap(net.layerStream, "the release does not fit whole, so its layers stream")
        let sequence = tokens + record.continuation
        let states = net.layerStates(MLXArray(sequence.map { Int32($0) }).reshaped([1, sequence.count]))
        let logits = net.logits(fromHidden: states[states.count - 1])[0]
        eval(logits)
        try net.verifyStream()

        var worst = (cosine: Float(1), layer: 0)
        for index in 0 ..< states.count - 1 {
            let expected = try XCTUnwrap(record.arrays["hidden_last.\(index)"], "the record's state after layer \(index)")
            let measured = cosine(states[index][0, tokens.count - 1], expected)
            if measured < worst.cosine { worst = (measured, index) }
        }
        let promptLogits = logits[0 ..< tokens.count]
        let mine = promptLogits.argMax(axis: -1).asArray(Int32.self)
        let theirs = record.logits.argMax(axis: -1).asArray(Int32.self)
        let agree = zip(mine, theirs).filter { $0 == $1 }.count
        let logitCosine = cosine(promptLogits, record.logits)
        let predicted = logits[(tokens.count - 1) ..< (sequence.count - 1)].argMax(axis: -1).asArray(Int32.self).map(Int.init)

        let (bytes, seconds) = stream.readStatistics
        print("\(releaseKey) streamed: \(stream.layers.count) of \(net.configuration.layerCount) layers, logit cosine "
              + "\(logitCosine), argmax \(agree)/\(tokens.count), worst layer \(worst.cosine) at \(worst.layer), greedy "
              + "\(predicted == record.continuation ? "exact" : "\(predicted) vs \(record.continuation)"), "
              + String(format: "%.1f GB read at %.2f GB/s", Double(bytes) / 1e9, Double(bytes) / 1e9 / max(seconds, 1e-9)))
        XCTAssertGreaterThan(worst.cosine, 0.999, "the state after layer \(worst.layer)")
        XCTAssertGreaterThan(logitCosine, 0.999, "the prompt's logits")
        XCTAssertGreaterThanOrEqual(agree, tokens.count - 1, "the prompt's argmax")
        XCTAssertEqual(predicted, record.continuation, "each continuation token is the argmax after the ones before it")
    }

    private func streamedLanguageRelease(_ directory: URL) throws -> NFKMLXLanguageNet {
        try NFKMLXLanguage.loadedRelease(at: directory, precision: .checkpoint, residency: .streamed, streams: true).0
    }

    func testTheStreamedQwen3_14BMatchesItsReference() throws {
        try checkStreamedRelease(release: "IK_VAL_QWEN3_14B", record: "IK_PARITY_QWEN3_14B", load: streamedLanguageRelease)
    }

    /// The streamed Qwen3 14B decoding the reference's greedy continuation with Qwen3 0.6B drafting,
    /// through the cache: the drafted output is the 14B's own greedy output, in fewer passes than tokens.
    func testTheStreamedQwen3_14BDecodesItsReferenceContinuationWithADraft() throws {
        try requireMLXRuntime()
        let directory = try release("IK_VAL_QWEN3_14B")
        let draftDirectory = try release("IK_VAL_QWEN3")
        let record = try denseRecord("IK_PARITY_QWEN3_14B")
        let draft = try NFKMLXLanguage.loadedRelease(at: draftDirectory, precision: .checkpoint).0
        let net = try NFKMLXLanguage.loadedRelease(at: directory, precision: .checkpoint, residency: .streamed,
                                                   streams: true,
                                                   reserving: NFKMLXLanguage.heldBytes(of: draft)).0
        let stream = try XCTUnwrap(net.layerStream)
        var report = NFKMLXSpeculativeReport()
        let produced = net.generate(prompt: record.tokens, options: greedy {
            $0.maxTokens = record.continuation.count
            $0.draftTokens = 4
        }, draft: draft, report: &report)
        try net.verifyStream()
        let (bytes, seconds) = stream.readStatistics
        print("IK_VAL_QWEN3_14B drafted: \(produced.count) tokens in \(report.rounds) rounds, \(report.accepted)/"
              + "\(report.proposed) proposals kept, "
              + String(format: "%.1f GB read at %.2f GB/s", Double(bytes) / 1e9, Double(bytes) / 1e9 / max(seconds, 1e-9)))
        XCTAssertEqual(produced, record.continuation, "the drafted output is the reference's greedy continuation")
        XCTAssertLessThan(report.rounds, produced.count, "the draft saves passes")
    }

    /// Pixtral's Mistral NeMo decoder text-only, streamed, against the reference's text-only record of the
    /// release, which locates a difference in the fused pipeline to the decoder or to the fusion.
    func testTheStreamedPixtralDecoderMatchesItsTextReference() throws {
        try checkStreamedRelease(release: "IK_VAL_PIXTRAL", record: "IK_PARITY_PIXTRAL_TEXT") {
            try NFKMLXPixtral.decoder(directoryURL: $0, precision: .checkpoint, residency: .streamed)
        }
    }

    func testTheStreamedQwen3_32BMatchesItsReference() throws {
        try checkStreamedRelease(release: "IK_VAL_QWEN3_32B", record: "IK_PARITY_QWEN3_32B", load: streamedLanguageRelease)
    }

    func testTheStreamedMistralSmallMatchesItsReference() throws {
        try checkStreamedRelease(release: "IK_VAL_MISTRAL_SMALL", record: "IK_PARITY_MISTRAL_SMALL",
                                 load: streamedLanguageRelease)
    }

    func testTheStreamedQwen3VL8BDecoderMatchesItsReference() throws {
        try checkStreamedRelease(release: "IK_VAL_QWEN3_VL_8B", record: "IK_PARITY_QWEN3_VL_8B") {
            try NFKMLXQwen3VL.decoder(directoryURL: $0, precision: .checkpoint, residency: .streamed)
        }
    }

    func testTheStreamedQwen3VL32BDecoderMatchesItsReference() throws {
        try checkStreamedRelease(release: "IK_VAL_QWEN3_VL_32B", record: "IK_PARITY_QWEN3_VL_32B") {
            try NFKMLXQwen3VL.decoder(directoryURL: $0, precision: .checkpoint, residency: .streamed)
        }
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).reshaped([-1])
        let y = b.asType(.float32).reshaped([-1])
        return ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))).item(Float.self)
    }
}
