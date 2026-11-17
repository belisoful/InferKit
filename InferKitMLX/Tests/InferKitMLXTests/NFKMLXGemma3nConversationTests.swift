//
//  NFKMLXGemma3nConversationTests.swift
//  InferKitMLXTests
//
//  Gemma 3n's kept conversations: a follow-up prompt rolls the cache back to the prefix it shares, past the
//  sliding window and through the shared key-value tail, and gives the logits and greedy tokens a run of the
//  whole prompt gives.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXGemma3nConversationTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private let configuration = NFKMLXGemma3nConfiguration.tiny

    private func net() -> NFKMLXGemma3nNet {
        MLXRandom.seed(37)
        let net = NFKMLXGemma3nNet(configuration)
        eval(net)
        return net
    }

    private func tokens(_ count: Int, seed: Int) -> [Int] {
        (0 ..< count).map { 5 + ($0 * 7 + seed * 13) % 26 }
    }

    private func lastLogits(_ net: NFKMLXGemma3nNet, _ sequence: [Int], cache: NFKMLXGemma3nCache? = nil) -> MLXArray {
        let logits = net(MLXArray(sequence.map(Int32.init)).reshaped([1, sequence.count]), cache: cache)[0, -1]
        eval(logits)
        return logits
    }

    private func worst(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    /// Runs `prompt` through the kept cache as generation does, then feeds `reply`; answers where the prefill
    /// started and the logits after the prompt and after each fed token.
    private func turn(_ net: NFKMLXGemma3nNet, _ kept: NFKMLXGemma3nPromptCache, prompt: [Int],
                      reply: [Int]) -> (start: Int, logits: [MLXArray]) {
        let start = kept.align(to: prompt)
        var logits = [lastLogits(net, Array(prompt[start...]), cache: kept.cache)]
        kept.record(Array(prompt[start...]))
        for token in reply {
            logits.append(lastLogits(net, [token], cache: kept.cache))
            kept.record([token])
        }
        return (start, logits)
    }

    // A follow-up that parts in the reply resumes at the end of the last prompt, and one that parts further
    // back than the sliding window resumes there; each gives the whole sequence's logits at every position.
    func testAFollowUpRollsBackPastTheWindowAndGivesTheFullLogits() throws {
        try requireMLXRuntime()
        let net = net()
        let kept = NFKMLXGemma3nPromptCache(layerCount: configuration.layerCount)
        let first = tokens(20, seed: 1)
        XCTAssertEqual(turn(net, kept, prompt: first, reply: tokens(6, seed: 2)).start, 0)

        var second = first + [9, 11] + tokens(5, seed: 3)
        let resumed = turn(net, kept, prompt: second, reply: tokens(3, seed: 4))
        XCTAssertEqual(resumed.start, first.count, "the reply was rendered otherwise, so the prompt resumes")
        XCTAssertLessThan(worst(resumed.logits[0], lastLogits(net, second)), 1e-4)
        for (index, token) in tokens(3, seed: 4).enumerated() {
            second.append(token)
            XCTAssertLessThan(worst(resumed.logits[index + 1], lastLogits(net, second)), 1e-4, "step \(index)")
        }

        let parted = Array(first[..<12]) + tokens(8, seed: 5)
        let deep = turn(net, kept, prompt: parted, reply: [])
        XCTAssertEqual(deep.start, 12, "twelve positions back, three windows deep")
        XCTAssertLessThan(worst(deep.logits[0], lastLogits(net, parted)), 1e-4)
        XCTAssertEqual(kept.sharedPrefixLength, 12)
    }

    // A conversation written to disk and read back continues as the held one does.
    func testASavedConversationContinuesAsTheHeldOne() throws {
        try requireMLXRuntime()
        let net = net()
        let kept = NFKMLXGemma3nPromptCache(layerCount: configuration.layerCount)
        let first = tokens(14, seed: 1)
        _ = turn(net, kept, prompt: first, reply: tokens(4, seed: 2))
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma3n-spill-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let spill = NFKMLXConversationSpill(directory: directory, byteBudget: 1 << 30)
        try spill.write(kept, name: "a")
        let restored = try XCTUnwrap(spill.take("a") { url, record in
            try NFKMLXGemma3nPromptCache.load(from: url, tokens: record.tokens, layerCount: self.configuration.layerCount)
        })
        XCTAssertEqual(restored.tokens, kept.tokens)
        let second = first + tokens(6, seed: 3)
        let resumed = turn(net, restored, prompt: second, reply: [])
        XCTAssertEqual(resumed.start, first.count)
        XCTAssertLessThan(worst(resumed.logits[0], lastLogits(net, second)), 1e-4)
    }

    // MARK: The released decoder

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).asArray(Float.self).map(Double.init)
        let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        let norms = sqrt(x.reduce(0) { $0 + $1 * $1 }) * sqrt(y.reduce(0) { $0 + $1 * $1 })
        return dot / norms
    }

    // The E2B decoder at float32 continues a chat's second turn from what it kept of the first and gives the
    // logits and greedy tokens transformers 5.16 gives running the second turn's whole prompt through its cache.
    func testTheReleasedDecoderContinuesATurnAsAFullRerunDoes() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let recordPath = environment["IK_PARITY_GEMMA3N_CONVERSATION"], let release = environment["IK_VAL_GEMMA3N_E2B"] else {
            throw XCTSkip("set IK_PARITY_GEMMA3N_CONVERSATION and IK_VAL_GEMMA3N_E2B")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        func ids(_ key: String) throws -> [Int] { try XCTUnwrap(record[key]).asArray(Int32.self).map(Int.init) }
        let reference = try XCTUnwrap(record["output"])                      // [steps, vocabulary]
        let directory = URL(fileURLWithPath: release)
        let released = try NFKMLXGemma3nLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXGemma3nNet(released)
        try NFKMLXGemma3nLanguage.loadWeights(into: net, fromDirectory: directory)
        let kept = NFKMLXGemma3nPromptCache(layerCount: released.layerCount)

        _ = turn(net, kept, prompt: try ids("turn1.tokens"), reply: Array(try ids("turn1.continuation").dropLast()))
        let prompt = try ids("turn2.tokens"), continuation = try ids("continuation")
        let second = turn(net, kept, prompt: prompt, reply: Array(continuation.dropLast()))
        XCTAssertGreaterThan(second.start, 0, "the second turn resumed from the first")

        let fresh = NFKMLXGemma3nCache(layerCount: released.layerCount)
        var rerun = [lastLogits(net, prompt, cache: fresh)]
        for token in continuation.dropLast() {
            rerun.append(lastLogits(net, [token], cache: fresh))
        }
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
        print("VALIDATION PARITY gemma3n-e2b-conversation: turn 2 resumed at \(second.start) of \(prompt.count), "
              + "\(continuation.count) steps, worst logit cosine \(worstCosine) (a fresh run of the whole prompt "
              + "\(worstRerun); resumed against fresh \(worstAgreement))")
        // The released decoder's own logits measure 0.99999999999 against the reference.
        XCTAssertGreaterThan(worstCosine, 0.9999999999)
        XCTAssertGreaterThan(worstAgreement, 0.9999999999, "resuming adds nothing to a fresh run's error")
    }

    // MARK: The backend

    /// A Gemma tokenizer over the lowercase letters (ids 5 … 30), the newline byte (31), and Gemma's markers.
    private func tokenizer() throws -> NFKMLXGemmaTokenizer {
        let markers = ["<eos>": 1, "<bos>": 2, "<start_of_turn>": 100, "<end_of_turn>": 101]
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
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma3n-chat-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tokenizer.json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url))
    }

    private func backend() throws -> NFKMLXGemma3nBackend {
        let model = NFKMLXGemma3nModel(decoder: net(), vision: nil, audio: nil, visionEmbedder: nil, audioEmbedder: nil)
        return NFKMLXGemma3nBackend(model: NFKMLXGemma3n(model: model, tokenizer: try tokenizer()), identifier: "gemma3n")
    }

    private func run(_ backend: NFKMLXGemma3nBackend, _ messages: [[String: Any]],
                     _ parameters: [String: Any] = [:]) throws -> (text: String, cached: Int) {
        var all: [String: Any] = [NFKParameterTemperature: 0, NFKParameterMaxTokens: 4]
        all.merge(parameters) { $1 }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputMessages: messages],
                                                                       parameters: all))
        let usage = try XCTUnwrap(result.output(forKey: NFKOutputUsage) as? [String: Int])
        return (result.output(forKey: NFKOutputText) as? String ?? "", usage[NFKUsageCachedTokens] ?? -1)
    }

    // A follow-up with or without a key continues its conversation, answers as a run that keeps nothing, and
    // a released conversation continues from the spill directory.
    func testTheBackendContinuesConversationsKeyedOrNotAndFromDisk() throws {
        try requireMLXRuntime()
        let served = try backend()
        XCTAssertTrue(served.supportedParameterKeys.contains(NFKParameterConversationKey))
        let opening: [[String: Any]] = [["role": "user", "content": "tell me about the quiet harbour town"]]
        let followUp = opening + [["role": "assistant", "content": "it is calm"], ["role": "user", "content": "and the boats"]]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma3n-backend-spill-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        served.conversationSpillDirectoryURL = directory

        for key in ["a", nil] as [String?] {
            let parameters: [String: Any] = key.map { [NFKParameterConversationKey: $0] } ?? [:]
            XCTAssertEqual(try run(served, opening, parameters).cached, 0)
            let continued = try run(served, followUp, parameters)
            XCTAssertGreaterThan(continued.cached, 0, "\(key ?? "keyless") continues its prompt")
            XCTAssertEqual(continued.text, try run(served, followUp, [NFKMLXGenerationParameterKey.reusesPromptCache: false]).text)

            XCTAssertGreaterThan(served.releaseIdleConversationCaches(), 0)
            XCTAssertEqual(served.conversationCacheCount, 0)
            let more = followUp + [["role": "assistant", "content": continued.text], ["role": "user", "content": "thanks"]]
            XCTAssertGreaterThan(try run(served, more, parameters).cached, 0, "\(key ?? "keyless") continues from disk")
            served.resetPromptCache()
        }
        XCTAssertEqual(try run(served, opening, [NFKMLXGenerationParameterKey.reusesPromptCache: false]).cached, 0)
        XCTAssertEqual(served.conversationCacheCount, 0, "a request that turns reuse off keeps nothing")
    }
}
