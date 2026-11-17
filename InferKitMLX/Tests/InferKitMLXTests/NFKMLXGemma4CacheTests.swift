//
//  NFKMLXGemma4CacheTests.swift
//  InferKitMLXTests
//
//  Gemma 4's cached generation: a prompt prefilled in passes and tokens run one at a time through the
//  full-attention layers' whole keys and values and the sliding layers' window give the logits the full
//  sequence gives, across the window, the shared key-value tail, the mixture, and the 12B unified decoder.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXGemma4CacheTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The E-series' structure at a size a test runs, with a window far shorter than the prompts: both
    /// kinds at their own head widths, a full layer that takes its key projection as its value, per-layer
    /// inputs, and a shared tail of each kind.
    private var eSeries: NFKMLXGemmaConfiguration {
        NFKMLXGemmaConfiguration(
            hiddenSize: 64, layerCount: 6, intermediateSize: 96, vocabularySize: 131, headCount: 4,
            keyValueHeadCount: 2, globalKeyValueHeadCount: 1, attentionKeyEqualsValue: true,
            headDimensions: 16, globalHeadDimensions: 32, slidingWindow: 6, globalPartialRotaryFactor: 0.25,
            perLayerInputSize: 8, sharedKeyValueLayers: 2, finalLogitSoftcap: 30,
            layerTypes: [.sliding, .sliding, .full, .sliding, .sliding, .full], perLayerVocabularySize: 131)
    }

    private var mixture: NFKMLXGemmaConfiguration {
        var c = NFKMLXGemmaConfiguration.tinyMixture
        c.slidingWindow = 6
        return c
    }

    private var unified: NFKMLXGemmaConfiguration {
        var c = NFKMLXGemmaConfiguration.unifiedTiny
        c.slidingWindow = 6
        return c
    }

    private func tokens(_ count: Int, seed: Int) -> [Int] {
        (0 ..< count).map { 3 + ($0 * 37 + seed * 11) % 120 }
    }

    private func worst(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    /// Prefills `prompt` in passes of 8 through `cached`, then runs `steps` one at a time, comparing each
    /// position's logits with `full`'s pass over the whole sequence.
    private func checkCached(_ cached: NFKGemmaCachedGeneration, full: (MLXArray) -> MLXArray, name: String) {
        let prompt = tokens(30, seed: 1), steps = tokens(5, seed: 2)
        let cache = cached.makeCache()
        var logits = cached.prefill(prompt, cache: cache, limit: 8)
        var sequence = prompt
        func expected() -> MLXArray {
            let all = full(MLXArray(sequence.map(Int32.init)).reshaped([1, sequence.count]))[0, sequence.count - 1]
            eval(all)
            return all
        }
        eval(logits)
        XCTAssertLessThan(worst(logits, expected()), 1e-4, "\(name): the prefill's last row")
        for token in steps {
            logits = cached.step(MLXArray(Int32(token)), cache: cache)
            eval(logits)
            sequence.append(token)
            XCTAssertLessThan(worst(logits, expected()), 1e-4, "\(name): after \(sequence.count) tokens")
        }
        XCTAssertEqual(cache.offset, sequence.count)
    }

    // The E-series' sliding window, shared tail, and per-layer inputs give the full sequence's logits.
    func testTheESeriesCacheGivesTheFullSequencesLogits() throws {
        try requireMLXRuntime()
        MLXRandom.seed(41)
        let net = NFKMLXGemmaLanguage.makeNet(eSeries)
        eval(net)
        checkCached(NFKMLXGemmaLanguage.cachedGeneration(net), full: { net($0) }, name: "e-series")
    }

    // The mixture's routed experts run per position, so its cache gives what its full pass gives.
    func testTheMixtureCacheGivesTheFullSequencesLogits() throws {
        try requireMLXRuntime()
        MLXRandom.seed(43)
        let net = NFKMLXGemmaLanguage.makeNet(mixture)
        eval(net)
        checkCached(NFKMLXGemmaLanguage.cachedGeneration(net), full: { net($0) }, name: "mixture")
    }

    func testTheUnifiedCacheGivesTheFullSequencesLogits() throws {
        try requireMLXRuntime()
        MLXRandom.seed(47)
        let net = NFKMLXGemmaLanguage.makeUnifiedNet(unified)
        eval(net)
        checkCached(NFKMLXGemmaLanguage.cachedGeneration(net), full: { net($0) }, name: "unified")
    }

    // Supplied embeddings take the token embedding's place in a cached pass as in a full one, which is the
    // multimodal path's seam.
    func testSuppliedEmbeddingsPrefillTheCacheAsTheFullPassReadsThem() throws {
        try requireMLXRuntime()
        MLXRandom.seed(53)
        let net = NFKMLXGemmaLanguage.makeNet(eSeries)
        eval(net)
        let prompt = tokens(20, seed: 3)
        let ids = MLXArray(prompt.map(Int32.init)).reshaped([1, prompt.count])
        let embeddings = net.embed(ids) + MLXRandom.normal(net.embed(ids).shape, key: MLXRandom.key(5)) * 0.1
        let cached = NFKMLXGemmaLanguage.cachedGeneration(net)
        let logits = cached.prefill(prompt, embeddings: embeddings, cache: cached.makeCache(), limit: 7)
        let full = net.logits(fromEmbeddings: embeddings, tokens: ids)[0, prompt.count - 1]
        eval(logits, full)
        XCTAssertLessThan(worst(logits, full), 1e-4)
    }

    // MARK: The released decoder

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).asArray(Float.self).map(Double.init)
        let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        let norms = sqrt(x.reduce(0) { $0 + $1 * $1 }) * sqrt(y.reduce(0) { $0 + $1 * $1 })
        return dot / norms
    }

    // The E2B decoder at float32 generates a chat's first turn through its cache and continues the second from
    // what it kept, the template having dropped the reply's thinking, and gives the logits and greedy tokens
    // transformers gives running the second turn's whole prompt through its own cache.
    func testTheReleasedDecoderContinuesATurnAsAFullRerunDoes() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let recordPath = environment["IK_PARITY_GEMMA4_CONVERSATION"], let release = environment["IK_VAL_GEMMA4"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_CONVERSATION and IK_VAL_GEMMA4")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        func ids(_ key: String) throws -> [Int] { try XCTUnwrap(record[key]).asArray(Int32.self).map(Int.init) }
        let reference = try XCTUnwrap(record["output"])                      // [steps, vocabulary]
        let directory = URL(fileURLWithPath: release)
        let configuration = try NFKMLXGemmaLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXGemmaLanguage.makeNet(configuration)
        try NFKMLXGemmaLanguage.loadWeights(into: net, fromDirectory: directory)
        let cached = NFKMLXGemmaLanguage.cachedGeneration(net)
        let kept = NFKMLXGemma4Conversation(layerTypes: configuration.layerTypes, slidingWindow: configuration.slidingWindow,
                                            storingLayers: configuration.firstSharedLayer)

        _ = turn(cached, kept, prompt: try ids("turn1.tokens"), turnStart: try ids("turn1.start")[0],
                 reply: Array(try ids("turn1.continuation").dropLast()))
        let prompt = try ids("turn2.tokens"), continuation = try ids("continuation")
        let second = turn(cached, kept, prompt: prompt, turnStart: try ids("turn2.start")[0],
                          reply: Array(continuation.dropLast()))
        XCTAssertGreaterThan(second.start, 0, "the second turn resumed from the first")

        let fresh = cached.makeCache()
        var rerun = [cached.prefill(prompt, cache: fresh)]
        for token in continuation.dropLast() {
            rerun.append(cached.step(MLXArray(Int32(token)), cache: fresh))
        }
        eval(rerun)
        // The reference generates a fixed count with `min_new_tokens`, which suppresses the end ids before it
        // chooses; the greedy token is the raw logits' argmax, and both runs are fed the chosen tokens.
        var worstCosine = 1.0, worstRerun = 1.0, worstAgreement = 1.0
        for step in continuation.indices {
            worstCosine = min(worstCosine, cosine(second.logits[step], reference[step]))
            worstRerun = min(worstRerun, cosine(rerun[step], reference[step]))
            worstAgreement = min(worstAgreement, cosine(second.logits[step], rerun[step]))
            XCTAssertEqual(second.logits[step].argMax().item(Int.self), reference[step].argMax().item(Int.self),
                           "the greedy token at step \(step)")
        }
        print("VALIDATION PARITY gemma4-e2b-conversation: turn 2 resumed at \(second.start) of \(prompt.count), "
              + "\(continuation.count) steps, worst logit cosine \(worstCosine) (a fresh cached run of the whole "
              + "prompt \(worstRerun); resumed against fresh \(worstAgreement))")
        // The released decoder's own logits measure 0.9999999999994 against the reference.
        XCTAssertGreaterThan(worstCosine, 0.9999999999)
        XCTAssertGreaterThan(worstRerun, 0.9999999999)
        XCTAssertGreaterThan(worstAgreement, 0.9999999999, "resuming adds nothing to a fresh run's error")
    }

    // MARK: The backend

    private func tokenizer() throws -> NFKMLXGemmaTokenizer {
        let markers = ["<eos>": 1, "<bos>": 2]
        var vocabulary = markers
        vocabulary["<unk>"] = 3
        vocabulary["\u{2581}"] = 4
        for (index, letter) in "abcdefghijklmnopqrstuvwxyz".enumerated() {
            vocabulary[String(letter)] = 5 + index
        }
        let json: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [] as [Any], "unk_token": "<unk>"],
            "added_tokens": markers.map { ["content": $0.key, "id": $0.value, "special": true] as [String: Any] },
        ]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-cache-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tokenizer.json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url))
    }

    // The backend's cached loop writes the reply its loop over the whole growing sequence writes.
    func testTheCachedBackendRepliesAsTheUncachedOneDoes() throws {
        try requireMLXRuntime()
        MLXRandom.seed(59)
        let net = NFKMLXGemmaLanguage.makeNet(eSeries)
        eval(net)
        let tokenizer = try tokenizer()
        let cached = NFKMLXGemmaBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: "gemma",
                                        cached: NFKMLXGemmaLanguage.cachedGeneration(net))
        let uncached = NFKMLXGemmaBackend(logits: { net($0) }, tokenizer: tokenizer, identifier: "gemma")
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "the quiet harbour town sleeps under the hills"],
                                          parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 12])
        let reply = try cached.runInference(for: request).output(forKey: NFKOutputText) as? String
        XCTAssertEqual(reply, try uncached.runInference(for: request).output(forKey: NFKOutputText) as? String)
    }

    // MARK: Conversations

    /// Runs `prompt` through `kept` as the backend does, then feeds `reply`; answers where the prefill started
    /// and the logits after the prompt and after each fed token.
    private func turn(_ cached: NFKGemmaCachedGeneration, _ kept: NFKMLXGemma4Conversation, prompt: [Int],
                      turnStart: Int?, reply: [Int]) -> (start: Int, logits: [MLXArray]) {
        let start = kept.resume(for: prompt)
        let stops = NFKMLXGemma4Conversation.snapshotPositions(start: start, end: prompt.count, turnStart: turnStart)
        var logits = [cached.prefill(Array(prompt[start...]), cache: kept.cache, limit: 8, stops: stops) { _ in
            kept.snapshot()
        }]
        kept.record(Array(prompt[start...]))
        for token in reply {
            logits.append(cached.step(MLXArray(Int32(token)), cache: kept.cache))
            kept.record([token])
        }
        eval(logits)
        return (start, logits)
    }

    private func fullLast(_ net: NFKMLXGemmaNet, _ sequence: [Int]) -> MLXArray {
        let logits = net(MLXArray(sequence.map(Int32.init)).reshaped([1, sequence.count]))[0, sequence.count - 1]
        eval(logits)
        return logits
    }

    // A follow-up parting further back than the window reaches resumes from the turn-start snapshot, one that
    // extends every held token continues the live cache, and one parting inside the window rolls back; each
    // gives the whole sequence's logits.
    func testAFollowUpResumesFromTheTurnSnapshotTheLiveCacheOrTheWindow() throws {
        try requireMLXRuntime()
        MLXRandom.seed(61)
        let net = NFKMLXGemmaLanguage.makeNet(eSeries)
        eval(net)
        let cached = NFKMLXGemmaLanguage.cachedGeneration(net)
        let kept = NFKMLXGemma4Conversation(layerTypes: eSeries.layerTypes, slidingWindow: eSeries.slidingWindow,
                                            storingLayers: eSeries.firstSharedLayer)
        let first = tokens(30, seed: 1)
        XCTAssertEqual(turn(cached, kept, prompt: first, turnStart: 26, reply: tokens(4, seed: 2)).start, 0)
        XCTAssertEqual(kept.snapshots.map(\.position), [26, 30], "the turn start and the prompt's end")

        var second = Array(first[..<28]) + tokens(9, seed: 3)
        var resumed = turn(cached, kept, prompt: second, turnStart: 33, reply: tokens(3, seed: 4))
        XCTAssertEqual(resumed.start, 26, "six positions back is past what a six-wide window retains: the snapshot")
        XCTAssertLessThan(worst(resumed.logits[0], fullLast(net, second)), 1e-4)
        for (index, token) in tokens(3, seed: 4).enumerated() {
            second.append(token)
            XCTAssertLessThan(worst(resumed.logits[index + 1], fullLast(net, second)), 1e-4, "step \(index)")
        }

        let third = kept.tokens + tokens(5, seed: 5)
        resumed = turn(cached, kept, prompt: third, turnStart: nil, reply: [])
        XCTAssertEqual(resumed.start, second.count, "the live cache")
        XCTAssertLessThan(worst(resumed.logits[0], fullLast(net, third)), 1e-4)

        let fourth = Array(kept.tokens.dropLast(2)) + tokens(4, seed: 6)
        resumed = turn(cached, kept, prompt: fourth, turnStart: nil, reply: [])
        XCTAssertEqual(resumed.start, kept.tokens.count - 4, "two positions back, inside the window")
        XCTAssertLessThan(worst(resumed.logits[0], fullLast(net, fourth)), 1e-4)
    }

    // A conversation written to disk and read back resumes from its snapshot as the held one does.
    func testASavedConversationResumesAsTheHeldOne() throws {
        try requireMLXRuntime()
        MLXRandom.seed(67)
        let net = NFKMLXGemmaLanguage.makeNet(eSeries)
        eval(net)
        let cached = NFKMLXGemmaLanguage.cachedGeneration(net)
        let kept = NFKMLXGemma4Conversation(layerTypes: eSeries.layerTypes, slidingWindow: eSeries.slidingWindow,
                                            storingLayers: eSeries.firstSharedLayer)
        let first = tokens(24, seed: 1)
        _ = turn(cached, kept, prompt: first, turnStart: 20, reply: tokens(6, seed: 2))
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-spill-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let spill = NFKMLXConversationSpill(directory: directory, byteBudget: 1 << 30)
        try spill.write(kept, name: "a")
        let configuration = eSeries
        let restored = try XCTUnwrap(spill.take("a") { url, record in
            try NFKMLXGemma4Conversation.load(from: url, tokens: record.tokens, layerTypes: configuration.layerTypes,
                                              slidingWindow: configuration.slidingWindow,
                                              storingLayers: configuration.firstSharedLayer)
        })
        XCTAssertEqual(restored.snapshots.map(\.position), [20, 24])
        let second = Array(first[..<22]) + tokens(6, seed: 3)
        let resumed = turn(cached, restored, prompt: second, turnStart: nil, reply: [])
        XCTAssertEqual(resumed.start, 20)
        XCTAssertLessThan(worst(resumed.logits[0], fullLast(net, second)), 1e-4)
    }

    // The backend continues a conversation with or without a key, answers as a run that keeps nothing, and a
    // released conversation continues from the spill directory.
    func testTheBackendContinuesConversationsKeyedOrNotAndFromDisk() throws {
        try requireMLXRuntime()
        MLXRandom.seed(71)
        let net = NFKMLXGemmaLanguage.makeNet(eSeries)
        eval(net)
        let served = NFKMLXGemmaBackend(logits: { net($0) }, tokenizer: try tokenizer(), identifier: "gemma",
                                        cached: NFKMLXGemmaLanguage.cachedGeneration(net))
        XCTAssertTrue(served.supportedParameterKeys.contains(NFKParameterConversationKey))
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-backend-spill-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        served.conversationSpillDirectoryURL = directory
        func run(_ prompt: String, _ parameters: [String: Any]) throws -> (text: String, cached: Int) {
            var all: [String: Any] = [NFKParameterTemperature: 0, NFKParameterMaxTokens: 4]
            all.merge(parameters) { $1 }
            let result = try served.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: prompt], parameters: all))
            let usage = try XCTUnwrap(result.output(forKey: NFKOutputUsage) as? [String: Int])
            return (result.output(forKey: NFKOutputText) as? String ?? "", usage[NFKUsageCachedTokens] ?? -1)
        }
        let opening = "tell me about the quiet harbour town"
        let followUp = opening + " and the boats in it"
        for key in ["a", nil] as [String?] {
            let parameters: [String: Any] = key.map { [NFKParameterConversationKey: $0] } ?? [:]
            XCTAssertEqual(try run(opening, parameters).cached, 0)
            let continued = try run(followUp, parameters)
            XCTAssertGreaterThan(continued.cached, 0, "\(key ?? "keyless") continues its prompt")
            XCTAssertEqual(continued.text, try run(followUp, [NFKMLXGenerationParameterKey.reusesPromptCache: false]).text)
            XCTAssertGreaterThan(served.releaseIdleConversationCaches(), 0)
            XCTAssertEqual(served.conversationCacheCount, 0)
            XCTAssertGreaterThan(try run(followUp + " today", parameters).cached, 0, "\(key ?? "keyless") continues from disk")
            served.resetPromptCache()
        }
    }
}
