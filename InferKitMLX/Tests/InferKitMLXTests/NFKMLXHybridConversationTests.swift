//
//  NFKMLXHybridConversationTests.swift
//  InferKitMLXTests
//
//  The hybrid decoder's kept conversations: a follow-up prompt resumes from the live cache or from a
//  snapshot of the recurrent state, and gives the logits and greedy tokens a run of the whole prompt gives.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXHybridConversationTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private var small: NFKMLXHybridConfiguration {
        NFKMLXHybridConfiguration(hiddenSize: 64, layerCount: 8, intermediateSize: 128,
                                  vocabularySize: 256, headCount: 4, keyValueHeadCount: 2,
                                  headDimensions: 16, ropeTheta: 10_000, partialRotaryFactor: 0.25,
                                  gatesAttentionOutput: true,
                                  linearKeyHeadCount: 2, linearKeyHeadDimensions: 8,
                                  linearValueHeadCount: 4, linearValueHeadDimensions: 8,
                                  linearConvolutionKernel: 4, fullAttentionInterval: 4)
    }

    private func tokens(_ count: Int, seed: Int) -> [Int] {
        (0 ..< count).map { ($0 * 37 + seed * 11 + 5) % 256 }
    }

    private func fullLastLogits(_ net: NFKMLXHybridLanguageNet, _ sequence: [Int]) -> MLXArray {
        let logits = net(MLXArray(sequence.map(Int32.init)).reshaped([1, sequence.count]))[0, sequence.count - 1]
        eval(logits)
        return logits
    }

    private func worst(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).asArray(Float.self).map(Double.init)
        let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        let norms = sqrt(x.reduce(0) { $0 + $1 * $1 }) * sqrt(y.reduce(0) { $0 + $1 * $1 })
        return dot / norms
    }

    private func conversation(quantization: NFKMLXKeyValueCache.Quantization? = nil) -> NFKMLXHybridConversation {
        NFKMLXHybridConversation(configuration: small, window: nil, quantization: quantization, adapterStrengths: nil)
    }

    /// Runs `prompt` through the conversation as a backend does: resume, prefill with snapshots, then
    /// `steps` tokens of `reply` one at a time. Answers where the prefill started and the logits after the
    /// prompt and after each fed reply token.
    private func turn(_ net: NFKMLXHybridLanguageNet, _ kept: NFKMLXHybridConversation, prompt: [Int],
                      turnStart: Int?, reply: [Int]) -> (start: Int, logits: [MLXArray]) {
        let start = kept.resume(for: prompt)
        let stops = NFKMLXHybridConversation.snapshotPositions(start: start, end: prompt.count, turnStart: turnStart)
        var logits = [net.prefill(Array(prompt[start...]), cache: kept.cache, limit: 64, stops: stops) { _ in
            kept.snapshot()
        }]
        kept.record(Array(prompt[start...]))
        for token in reply {
            logits.append(net.lastLogits(MLXArray([Int32(token)]).reshaped([1, 1]), cache: kept.cache)[0])
        }
        kept.record(reply)
        kept.snapshot()
        eval(logits)
        return (start, logits)
    }

    // The snapshot positions are the chunk boundary before the generation prompt and each multiple of the
    // interval inside the prefill, never its ends.
    func testTheSnapshotPositionsAreTheTurnStartChunkAndTheInterval() {
        XCTAssertEqual(NFKMLXHybridConversation.snapshotPositions(start: 0, end: 150, turnStart: 140), [128])
        XCTAssertEqual(NFKMLXHybridConversation.snapshotPositions(start: 0, end: 9000, turnStart: 8200), [4096, 8192])
        XCTAssertEqual(NFKMLXHybridConversation.snapshotPositions(start: 4096, end: 9000, turnStart: 8250), [8192])
        XCTAssertEqual(NFKMLXHybridConversation.snapshotPositions(start: 130, end: 300, turnStart: 140), [])
        XCTAssertEqual(NFKMLXHybridConversation.snapshotPositions(start: 0, end: 40, turnStart: nil), [])
    }

    // A follow-up that re-tokenizes the end of the last prompt resumes from the turn-start snapshot, one that
    // extends every held token continues the live cache, and either gives the logits of the whole sequence at
    // every position, across the snapshot's chunk boundary and the convolution's history.
    func testAFollowUpResumesFromTheSnapshotOrTheLiveCacheAndGivesTheFullLogits() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        eval(net)
        let kept = conversation()
        let first = tokens(150, seed: 1), reply = tokens(5, seed: 2)
        XCTAssertEqual(turn(net, kept, prompt: first, turnStart: 140, reply: reply).start, 0)
        XCTAssertEqual(kept.snapshots.map(\.position), [128, 155])

        // The opening's last tokens render differently on the next request, as `<think>\n` does.
        var second = Array(first[..<145]) + [7, 9] + tokens(40, seed: 3)
        var resumed = turn(net, kept, prompt: second, turnStart: 160, reply: tokens(4, seed: 4))
        XCTAssertEqual(resumed.start, 128, "the turn-start snapshot")
        XCTAssertLessThan(worst(resumed.logits[0], fullLastLogits(net, second)), 1e-4, "the prefill's last row")
        for (index, token) in tokens(4, seed: 4).enumerated() {
            second.append(token)
            XCTAssertLessThan(worst(resumed.logits[index + 1], fullLastLogits(net, second)), 1e-4, "step \(index)")
        }

        let third = kept.tokens + tokens(30, seed: 5)
        resumed = turn(net, kept, prompt: third, turnStart: nil, reply: [])
        XCTAssertEqual(resumed.start, second.count, "the live cache")
        XCTAssertLessThan(worst(resumed.logits[0], fullLastLogits(net, third)), 1e-4)

        XCTAssertEqual(kept.resume(for: tokens(60, seed: 9)), 0, "nothing shared starts over")
        XCTAssertEqual(kept.tokens, [])
        XCTAssertEqual(kept.snapshots.count, 0)
    }

    // A quantized cache rolls its packed rows back to the snapshot, and the resumed run gives what the same
    // prompt gives through a fresh quantized cache.
    func testAQuantizedCacheResumesAsAFreshOneRuns() throws {
        try requireMLXRuntime()
        var configuration = small
        configuration.headDimensions = 32
        let net = NFKMLXHybridLanguage.makeNet(configuration)
        eval(net)
        let quantization = NFKMLXKeyValueCache.Quantization(bits: 8, groupSize: 32)
        let kept = NFKMLXHybridConversation(configuration: configuration, window: nil, quantization: quantization,
                                            adapterStrengths: nil)
        let first = tokens(140, seed: 1)
        _ = turn(net, kept, prompt: first, turnStart: 130, reply: tokens(3, seed: 2))
        let second = Array(first[..<135]) + tokens(20, seed: 3)
        let resumed = turn(net, kept, prompt: second, turnStart: nil, reply: [])
        XCTAssertEqual(resumed.start, 128)
        let fresh = NFKMLXHybridCache(configuration: configuration, quantization: quantization)
        let expected = net.prefill(second, cache: fresh, limit: 64)
        eval(expected)
        XCTAssertLessThan(worst(resumed.logits[0], expected), 1e-4)
    }

    // MARK: The released decoder

    // Qwen3.5-2B at float32 continues a chat's second turn from what it kept of the first, the template
    // having rewritten the reply's opening, and gives the logits and greedy tokens transformers 5.17 gives
    // running the second turn's whole prompt through `Qwen3_5DynamicCache`.
    func testTheReleasedDecoderContinuesATurnAsAFullRerunDoes() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let recordPath = environment["IK_PARITY_QWEN3_5_CONVERSATION"],
              let release = environment["IK_VAL_QWEN3_5_2B"] else {
            throw XCTSkip("set IK_PARITY_QWEN3_5_CONVERSATION and IK_VAL_QWEN3_5_2B")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        func ids(_ key: String) throws -> [Int] { try XCTUnwrap(record[key]).asArray(Int32.self).map(Int.init) }
        let reference = try XCTUnwrap(record["output"])                      // [steps, vocabulary]
        let net = try NFKMLXHybridLanguage.network(directoryURL: URL(fileURLWithPath: release), precision: .float32)
        let kept = NFKMLXHybridConversation(configuration: net.configuration, window: nil, quantization: nil,
                                            adapterStrengths: nil)

        func turn(_ prompt: [Int], start turnStart: Int, continuation: [Int]) -> (start: Int, logits: [MLXArray]) {
            let start = kept.resume(for: prompt)
            let stops = NFKMLXHybridConversation.snapshotPositions(start: start, end: prompt.count, turnStart: turnStart)
            var logits = [net.prefill(Array(prompt[start...]), cache: kept.cache, limit: 128, stops: stops) { _ in
                kept.snapshot()
            }]
            kept.record(Array(prompt[start...]))
            for token in continuation.dropLast() {
                logits.append(net.lastLogits(MLXArray([Int32(token)]).reshaped([1, 1]), cache: kept.cache)[0])
            }
            kept.record(Array(continuation.dropLast()))
            kept.snapshot()
            eval(logits)
            return (start, logits)
        }

        let firstPrompt = try ids("turn1.tokens"), secondPrompt = try ids("turn2.tokens")
        _ = turn(firstPrompt, start: try ids("turn1.start")[0], continuation: try ids("turn1.continuation"))
        let continuation = try ids("continuation")
        let second = turn(secondPrompt, start: try ids("turn2.start")[0], continuation: continuation)
        let turnStart = try ids("turn1.start")[0]
        XCTAssertEqual(second.start, turnStart - turnStart % 64, "the turn-start snapshot")
        let fresh = NFKMLXHybridCache(configuration: net.configuration)
        var rerun = [net.prefill(secondPrompt, cache: fresh, limit: 128)]
        for token in continuation.dropLast() {
            rerun.append(net.lastLogits(MLXArray([Int32(token)]).reshaped([1, 1]), cache: fresh)[0])
        }
        eval(rerun)
        var worstCosine = 1.0, worstRerun = 1.0, worstAgreement = 1.0
        // The reference generates a fixed count with `min_new_tokens`, which suppresses the end id before it
        // chooses; the greedy token is the raw logits' argmax, and both runs are fed the chosen tokens.
        for step in continuation.indices {
            worstCosine = min(worstCosine, cosine(second.logits[step], reference[step]))
            worstRerun = min(worstRerun, cosine(rerun[step], reference[step]))
            worstAgreement = min(worstAgreement, cosine(second.logits[step], rerun[step]))
            XCTAssertEqual(second.logits[step].argMax().item(Int.self), reference[step].argMax().item(Int.self),
                           "the greedy token at step \(step)")
        }
        print("VALIDATION PARITY qwen3.5-2b-conversation: turn 2 resumed at \(second.start) of \(secondPrompt.count), "
              + "\(continuation.count) steps, worst logit cosine \(worstCosine) (a fresh run of the whole prompt "
              + "\(worstRerun); resumed against fresh \(worstAgreement))")
        XCTAssertGreaterThan(worstCosine, 0.99999999999)
    }

    // A timing of a served follow-up, run where `IK_TIME_QWEN3_8_27B_FOLLOWUP` is set: the catalog's qwen3.8-27b
    // hosted by NFKInferenceServer with 4-bit weights, a 64K window, and an 8-bit cache, asked through the
    // streamed chat route at temperature 0 with no conversation key. Turn 1 is a long prompt; turn 2 sends it back
    // with the reply, its reasoning under `reasoning_content`, and a short question, as an agent's next turn does.
    // Each turn prints its first delta, the cached prompt tokens the usage reports, and the memory.
    func testTheQwen3_8_27BServedFollowUpPrefillsOnlyItsTurn() async throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard environment["IK_TIME_QWEN3_8_27B_FOLLOWUP"] != nil, let release = environment["IK_VAL_QWEN3_8_27B"] else {
            throw XCTSkip("set IK_TIME_QWEN3_8_27B_FOLLOWUP and IK_VAL_QWEN3_8_27B to time the served follow-up")
        }
        func gb(_ bytes: Int) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }
        func memory(_ stage: String) {
            var info = rusage_info_v4()
            _ = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
            }
            print("VALIDATION TIMING qwen3.8-27b follow-up \(stage): footprint \(gb(Int(info.ri_phys_footprint))), lifetime "
                  + "peak \(gb(Int(info.ri_lifetime_max_phys_footprint))); MLX active \(gb(MLX.Memory.activeMemory)), cache "
                  + "\(gb(MLX.Memory.cacheMemory)), peak \(gb(MLX.Memory.peakMemory))")
        }
        NFKMLXGPU.applyStandingLimits()
        NFKHFHub.defaultModelFolderURLs = [URL(fileURLWithPath: release).deletingLastPathComponent()]
        let server = NFKInferenceServer()
        server.port = 0
        server.loopbackOnly = true
        try server.addModel(identifier: "qwen3.8-27b", options: [
            NFKModelLoadOptionQuantizationBits: 4, NFKModelLoadOptionContextLength: 65536,
            NFKModelLoadOptionCacheBits: 8, NFKModelLoadOptionResidency: NFKModelResidencyResident,
        ], catalog: nil, forModelName: "agent")
        try server.start()
        defer { server.stop() }

        func ask(_ messages: [[String: Any]], label: String) async throws -> (content: String, reasoning: String) {
            var request = URLRequest(url: try XCTUnwrap(server.localBaseURL).appendingPathComponent("chat/completions"))
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = 7200
            request.httpBody = try JSONSerialization.data(withJSONObject: [
                "model": "agent", "messages": messages, "max_tokens": 64, "temperature": 0, "stream": true,
                "stream_options": ["include_usage": true],
            ] as [String: Any])
            let start = Date()
            var first: Date?
            var content = "", reasoning = ""
            var usage: [String: Any] = [:]
            let (bytes, _) = try await URLSession.shared.bytes(for: request)
            for try await line in bytes.lines where line.hasPrefix("data: ") && line != "data: [DONE]" {
                guard let event = try? JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any] else {
                    continue
                }
                if let reported = event["usage"] as? [String: Any] {
                    usage = reported
                }
                for choice in event["choices"] as? [[String: Any]] ?? [] {
                    let delta = choice["delta"] as? [String: Any] ?? [:]
                    if let text = delta["reasoning_content"] as? String, !text.isEmpty {
                        reasoning += text
                        first = first ?? Date()
                    }
                    if let text = delta["content"] as? String, !text.isEmpty {
                        content += text
                        first = first ?? Date()
                    }
                }
            }
            let cached = (usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] ?? 0
            let firstDelta = first.map { String(format: "%.1f s", $0.timeIntervalSince(start)) } ?? "none"
            print("VALIDATION TIMING qwen3.8-27b follow-up \(label): prompt \(usage["prompt_tokens"] ?? "?") tokens, "
                  + "\(cached) cached; first delta \(firstDelta); whole reply \(String(format: "%.1f", Date().timeIntervalSince(start))) s")
            memory("after \(label)")
            return (content, reasoning)
        }
        _ = try await ask([["role": "user", "content": "Say hello."]], label: "the warm-up")
        let docs = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Docs")
        var text = ""
        for name in try FileManager.default.contentsOfDirectory(atPath: docs.path).sorted() where name.hasSuffix(".md") {
            text += (try? String(contentsOf: docs.appendingPathComponent(name), encoding: .utf8)) ?? ""
            if text.count > 88_000 {
                break
            }
        }
        var messages: [[String: Any]] = [["role": "user", "content": String(text.prefix(88_000))
            + "\n\nSummarize the documents above in five sentences."]]
        let first = try await ask(messages, label: "turn 1")
        messages.append(["role": "assistant", "content": first.content, "reasoning_content": first.reasoning])
        messages.append(["role": "user", "content": "Now say it in one sentence."])
        let second = try await ask(messages, label: "turn 2")
        messages.append(["role": "assistant", "content": second.content, "reasoning_content": second.reasoning])
        messages.append(["role": "user", "content": "Which document is the longest?"])
        _ = try await ask(messages, label: "turn 3")
    }

    // MARK: The backend

    private let template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}"
        + "{% if add_generation_prompt %}<assistant>{% endif %}"

    private func backend(_ net: NFKMLXHybridLanguageNet) -> NFKMLXDecoderBackend {
        NFKMLXDecoderBackend(logits: { net($0) }, tokenizer: NFKMLXScalarTestTokenizer(vocabularySize: 256),
                             identifier: "qwen3.5", chatTemplate: template, stopTokens: [], modules: [net],
                             cached: net.cachedGeneration)
    }

    private func run(_ backend: NFKMLXDecoderBackend, _ messages: [[String: Any]],
                     _ parameters: [String: Any] = [:]) throws -> (text: String, cached: Int) {
        var all: [String: Any] = [NFKParameterTemperature: 0, NFKParameterMaxTokens: 6]
        all.merge(parameters) { $1 }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputMessages: messages],
                                                                       parameters: all))
        let usage = result.output(forKey: NFKOutputUsage) as? [String: Any]
        return (result.output(forKey: NFKOutputText) as? String ?? "", (usage?[NFKUsageCachedTokens] as? Int) ?? -1)
    }

    // A request that names no conversation continues the one its prompt extends; its reply is the reply of a
    // run that keeps nothing, and its usage counts the reused positions.
    func testAnUnnamedFollowUpContinuesTheConversationItsPromptExtends() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        eval(net)
        let served = backend(net)
        let system = String(repeating: "You answer briefly. ", count: 6)
        let opening: [[String: Any]] = [["role": "system", "content": system], ["role": "user", "content": "hi"]]
        let first = try run(served, opening)
        XCTAssertEqual(first.cached, 0)
        XCTAssertEqual(served.conversationCacheCount, 1)

        let followUp = opening + [["role": "assistant", "content": first.text], ["role": "user", "content": "more"]]
        let continued = try run(served, followUp)
        let rerun = try run(served, followUp, [NFKMLXGenerationParameterKey.reusesPromptCache: false])
        XCTAssertEqual(continued.text, rerun.text)
        let firstPrompt = system.count + "<system>\n<user>hi\n<assistant>".count
        // The last reply token was sampled but never run, so the cache holds one fewer.
        if first.text.unicodeScalars.count == 6 {
            XCTAssertEqual(continued.cached, firstPrompt + 5, "the prompt and the fed reply")
        }
        XCTAssertGreaterThanOrEqual(continued.cached, (firstPrompt - "<assistant>".count) / 64 * 64)
        XCTAssertEqual(rerun.cached, 0)
        XCTAssertEqual(served.conversationCacheCount, 1, "reuse off keeps nothing")

        // A client that rewrites the reply resumes from the snapshot before the first generation prompt.
        let rewritten = opening + [["role": "assistant", "content": "ok"], ["role": "user", "content": "more"]]
        let fromSnapshot = try run(served, rewritten)
        XCTAssertEqual(fromSnapshot.cached, (firstPrompt - "<assistant>".count) / 64 * 64)
        XCTAssertEqual(fromSnapshot.text, try run(served, rewritten, [NFKMLXGenerationParameterKey.reusesPromptCache: false]).text)
    }

    // A named conversation continues its own cache, beside an unnamed one; the idle release frees both and
    // reports their bytes.
    func testNamedConversationsKeepTheirOwnCachesUntilReleased() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        eval(net)
        let served = backend(net)
        let opening: [[String: Any]] = [["role": "user", "content": String(repeating: "abc ", count: 30)]]
        _ = try run(served, opening, [NFKParameterConversationKey: "a"])
        _ = try run(served, opening)
        XCTAssertEqual(served.conversationCacheCount, 2)
        let again = try run(served, opening, [NFKParameterConversationKey: "a"])
        XCTAssertGreaterThan(again.cached, 0)
        XCTAssertEqual(served.backendStatus["conversation_caches"] as? Int, 2)
        let held = served.conversationCacheBytes
        XCTAssertGreaterThan(held, 0)
        XCTAssertEqual(served.releaseIdleConversationCaches(), held)
        XCTAssertEqual(served.conversationCacheCount, 0)
    }

    // Past the budget the least recently used conversation goes, never the one that just ran.
    func testTheBudgetKeepsTheConversationThatRan() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        eval(net)
        let served = backend(net)
        served.conversationCacheByteBudget = 1
        for key in ["a", "b"] {
            _ = try run(served, [["role": "user", "content": "hello \(key)"]], [NFKParameterConversationKey: key])
        }
        XCTAssertEqual(served.conversationCacheCount, 1)
        XCTAssertEqual(served.conversationCacheByteBudget, 1)
    }

    // MARK: Spilling to disk

    private func spillDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("hybrid-spill-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    // A conversation written to disk and read back continues as the one held in memory does.
    func testASavedConversationContinuesAsTheHeldOne() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        eval(net)
        let kept = conversation()
        let first = tokens(150, seed: 1)
        _ = turn(net, kept, prompt: first, turnStart: 140, reply: tokens(5, seed: 2))
        let directory = try spillDirectory()
        let spill = NFKMLXConversationSpill(directory: directory, byteBudget: 1 << 30)
        try spill.write(kept, name: "a")
        let restored = try XCTUnwrap(spill.take("a", configuration: small, window: nil, quantization: nil, adapterStrengths: nil))
        XCTAssertEqual(restored.tokens, kept.tokens)
        XCTAssertEqual(restored.snapshots.map(\.position), [128, 155])
        XCTAssertTrue(spill.records().isEmpty, "a read conversation leaves the disk")

        let second = Array(first[..<145]) + tokens(30, seed: 3)
        let resumed = turn(net, restored, prompt: second, turnStart: nil, reply: [])
        XCTAssertEqual(resumed.start, 128)
        XCTAssertLessThan(worst(resumed.logits[0], fullLastLogits(net, second)), 1e-4)
        let live = turn(net, kept, prompt: kept.tokens + tokens(10, seed: 4), turnStart: nil, reply: [])
        XCTAssertEqual(live.start, 155)
        try spill.write(restored, name: "b")
        XCTAssertNil(spill.take("b", configuration: small, window: 64, quantization: nil, adapterStrengths: nil),
                     "another window reads nothing")
    }

    // A released conversation, named or not, is written to disk and continued from it by the next request.
    func testAReleasedConversationContinuesFromTheDisk() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        eval(net)
        let served = backend(net)
        served.conversationSpillDirectoryURL = try spillDirectory()
        let opening: [[String: Any]] = [["role": "user", "content": String(repeating: "abc ", count: 75)]]
        let followUp = opening + [["role": "assistant", "content": "ok"], ["role": "user", "content": "more"]]
        for parameters in [[NFKParameterConversationKey: "a"], [:]] as [[String: Any]] {
            _ = try run(served, opening, parameters)
            XCTAssertGreaterThan(served.releaseIdleConversationCaches(), 0)
            XCTAssertEqual(served.conversationCacheCount, 0)
            let continued = try run(served, followUp, parameters)
            XCTAssertEqual(continued.cached, 256, "the turn-start snapshot read from disk")
            XCTAssertEqual(continued.text, try run(served, followUp, [NFKMLXGenerationParameterKey.reusesPromptCache: false]).text)
            served.resetPromptCache()
        }
    }

    // Past the disk budget the least recently written conversation goes.
    func testTheSpillKeepsItsByteBudget() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        eval(net)
        let spill = NFKMLXConversationSpill(directory: try spillDirectory(), byteBudget: 1 << 30)
        for name in ["a", "b"] {
            let kept = conversation()
            _ = turn(net, kept, prompt: tokens(100, seed: name == "a" ? 1 : 2), turnStart: 90, reply: [])
            try spill.write(kept, name: name)
        }
        XCTAssertEqual(Set(spill.records().map(\.name)), ["a", "b"])
        spill.byteBudget = 1
        let kept = conversation()
        _ = turn(net, kept, prompt: tokens(100, seed: 3), turnStart: 90, reply: [])
        try spill.write(kept, name: "c")
        XCTAssertTrue(spill.records().map(\.name).allSatisfy { $0 == "c" })
    }
}
