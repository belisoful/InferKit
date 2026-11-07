//
//  NFKMLXHybridCacheTests.swift
//  InferKitMLXTests
//
//  The hybrid decoder's cache: a prompt prefilled in passes and tokens run one at a time through the
//  full-attention layers' keys and values and the linear-attention layers' convolution inputs and state
//  give the logits the full sequence gives. The released Qwen3.5-2B's cached generation is measured
//  against transformers' in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXHybridCacheTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The release's structure at a size a test runs: three recurrence layers, then full attention.
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

    // The Metal kernel computes the recurrence the MLX operations compute, from a cached state, and its read's
    // gradient is the recurrence's.
    func testTheKernelComputesTheRecurrence() throws {
        try requireMLXRuntime()
        let layer = NFKHybridLinearAttention(small)
        let (batch, length, heads, width) = (2, 37, 4, 8)
        func unit(_ x: MLXArray) -> MLXArray { x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6) }
        let queries = unit(MLXRandom.normal([batch, length, heads, width], key: MLXRandom.key(1)))
        let keys = unit(MLXRandom.normal([batch, length, heads, width], key: MLXRandom.key(2)))
        let values = MLXRandom.normal([batch, length, heads, width], key: MLXRandom.key(3))
        let decay = MLXRandom.uniform(0.5 ..< 1, [batch, length, heads], key: MLXRandom.key(4))
        let write = MLXRandom.uniform(0 ..< 1, [batch, length, heads], key: MLXRandom.key(5))
        let state = MLXRandom.normal([batch, heads, width, width], key: MLXRandom.key(6)) * 0.1
        let expected = layer.recurrence(queries: queries, keys: keys, values: values, decay: decay, write: write,
                                        initialState: state)
        let actual = NFKMLXGatedDeltaKernel.scan(queries: queries, keys: keys, values: values, decay: decay, write: write,
                                                 initialState: state)
        eval(expected.read, expected.state, actual.read, actual.state)
        XCTAssertLessThan(worst(actual.read, expected.read), 1e-5)
        XCTAssertLessThan(worst(actual.state, expected.state), 1e-5)

        let reference = grad({ (inputs: [MLXArray]) -> MLXArray in
            layer.recurrence(queries: queries, keys: inputs[1], values: inputs[0], decay: decay, write: write).square().sum()
        })([values, keys])
        let kernel = grad({ (inputs: [MLXArray]) -> MLXArray in
            NFKMLXGatedDeltaKernel.read(queries: queries, keys: inputs[1], values: inputs[0], decay: decay, write: write)
                .square().sum()
        })([values, keys])
        eval(reference, kernel)
        XCTAssertLessThan(worst(kernel[0], reference[0]), 1e-4)
        XCTAssertLessThan(worst(kernel[1], reference[1]), 1e-4)
    }

    // A prompt prefilled in passes of 64 and then four tokens run one at a time give, at every step, the
    // last row the whole sequence gives: across the delta rule's chunk boundaries, the passes, the
    // convolution's history, and the attention layers' rotary offsets.
    func testCachedPassesGiveTheFullSequencesLogits() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        eval(net)
        let prompt = tokens(150, seed: 1)
        let cache = NFKMLXHybridCache(configuration: small)
        var logits = net.prefill(prompt, cache: cache, limit: 64)
        eval(logits)
        XCTAssertEqual(cache.offset, prompt.count)
        XCTAssertLessThan(worst(logits, fullLastLogits(net, prompt)), 1e-4, "the prefill's last row")
        var sequence = prompt
        for token in tokens(4, seed: 2) {
            logits = net.lastLogits(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)[0]
            eval(logits)
            sequence.append(token)
            XCTAssertLessThan(worst(logits, fullLastLogits(net, sequence)), 1e-4, "after \(sequence.count) tokens")
        }
        XCTAssertEqual(cache.offset, sequence.count)
        XCTAssertGreaterThan(cache.allocatedBytes, 0)
    }

    // Running several tokens through the cache at once is a pass like the prefill's: it gives what one
    // token at a time gives.
    func testAMultiTokenPassContinuesTheCache() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        eval(net)
        let prompt = tokens(40, seed: 3), continuation = tokens(6, seed: 4)
        let together = NFKMLXHybridCache(configuration: small)
        _ = net.prefill(prompt, cache: together)
        let joined = net.lastLogits(MLXArray(continuation.map(Int32.init)).reshaped([1, continuation.count]),
                                    cache: together)[0]
        let apart = NFKMLXHybridCache(configuration: small)
        var stepped = net.prefill(prompt, cache: apart)
        for token in continuation {
            stepped = net.lastLogits(MLXArray([Int32(token)]).reshaped([1, 1]), cache: apart)[0]
        }
        eval(joined, stepped)
        XCTAssertLessThan(worst(joined, stepped), 1e-4)
    }

    // The pass length keeps the score budget and a whole number of the delta rule's chunks.
    func testThePrefillPassKeepsTheScoreBudgetAndWholeChunks() {
        let net = NFKMLXHybridLanguage.makeNet(small)
        XCTAssertEqual(net.prefillPassLength(cached: 0, remaining: 150, limit: 100), 64)
        XCTAssertEqual(net.prefillPassLength(cached: 0, remaining: 50, limit: 100), 50)
        XCTAssertEqual(net.prefillPassLength(cached: 0, remaining: 5000, limit: 2048, budget: 4 * 4 * 1000 * 1000), 512,
                       "halved twice past the budget")
        XCTAssertEqual(net.prefillPassLength(cached: 0, remaining: 5000, limit: 1000), 960, "whole chunks short of the prompt")
    }

    // A quantized cache stores the attention layers' keys and values packed, and a window bounds them; the
    // recurrence keeps its whole state either way.
    func testAQuantizedOrWindowedCacheRuns() throws {
        try requireMLXRuntime()
        // MLX packs in groups of 32 or more, so the heads are 32 wide here.
        var configuration = small
        configuration.headDimensions = 32
        let net = NFKMLXHybridLanguage.makeNet(configuration)
        eval(net)
        let prompt = tokens(96, seed: 5)
        let reference = net.prefill(prompt, cache: NFKMLXHybridCache(configuration: configuration))
        let quantized = NFKMLXHybridCache(configuration: configuration,
                                          quantization: NFKMLXKeyValueCache.Quantization(bits: 8, groupSize: 32))
        let packed = net.prefill(prompt, cache: quantized)
        let windowed = NFKMLXHybridCache(configuration: configuration, window: 32)
        var bounded = net.prefill(prompt, cache: windowed)
        bounded = net.lastLogits(MLXArray([Int32(7)]).reshaped([1, 1]), cache: windowed)[0]
        eval(reference, packed, bounded)
        XCTAssertLessThan(worst(packed, reference), 0.05, "an 8-bit cache approximates the float one")
        XCTAssertTrue(bounded.sum().item(Float.self).isFinite)
        XCTAssertLessThanOrEqual(windowed.attention.retainedLength(layer: 3), 32)
    }

    // MARK: The released decoder

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).asArray(Float.self).map(Double.init)
        let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        let norms = sqrt(x.reduce(0) { $0 + $1 * $1 }) * sqrt(y.reduce(0) { $0 + $1 * $1 })
        return dot / norms
    }

    // Qwen3.5-2B at float32 generates through its cache as transformers 5.17 does through
    // `Qwen3_5DynamicCache`: the prompt prefilled in passes of 128, then each greedy token alone, the logits
    // before every token compared with the reference's, and the greedy token kept.
    func testTheReleasedDecoderGeneratesThroughItsCacheAsTheReferenceDoes() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let recordPath = environment["IK_PARITY_QWEN3_5_CACHED"], let release = environment["IK_VAL_QWEN3_5_2B"] else {
            throw XCTSkip("set IK_PARITY_QWEN3_5_CACHED and IK_VAL_QWEN3_5_2B")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        let prompt = try XCTUnwrap(record["tokens"]).asArray(Int32.self).map(Int.init)
        let continuation = try XCTUnwrap(record["continuation"]).asArray(Int32.self).map(Int.init)
        let reference = try XCTUnwrap(record["output"])                      // [steps, vocabulary]
        let net = try NFKMLXHybridLanguage.network(directoryURL: URL(fileURLWithPath: release), precision: .float32)
        let cache = NFKMLXHybridCache(configuration: net.configuration)
        var logits = net.prefill(prompt, cache: cache, limit: 128)
        var worstCosine = 1.0
        for (step, token) in continuation.enumerated() {
            eval(logits)
            let similarity = cosine(logits, reference[step])
            worstCosine = min(worstCosine, similarity)
            XCTAssertEqual(logits.argMax().item(Int.self), token, "the greedy token at step \(step)")
            logits = net.lastLogits(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)[0]
        }
        print("VALIDATION PARITY qwen3.5-2b-cached: \(prompt.count)-token prompt, \(continuation.count) cached steps, worst logit cosine \(worstCosine)")
        XCTAssertGreaterThan(worstCosine, 0.99999999999)
    }

    // MARK: The catalog

    // Qwen3.8-27B declares the quantization, the context, and the cache options, and the estimate reads them:
    // the window's keys and values count on the attention layers alone, an 8-bit cache halves them, and a
    // quantization packs the token embedding too, so 4-bit weights hold under 16.5 GB at a 4K window.
    func testTheQwen3_8_27BEstimateReadsTheQuantizationAndTheWindow() throws {
        let descriptor = try XCTUnwrap(NFKMLXModelDescriptors.modelDescriptors().first { $0.identifier == "qwen3.8-27b" })
        for key in [NFKModelLoadOptionQuantizationBits, NFKModelLoadOptionContextLength, NFKModelLoadOptionCacheBits] {
            XCTAssertEqual(descriptor.option(withKey: key)?.stage, NFKModelOptionStageLoad, key)
        }
        func resident(_ options: [String: Any]) throws -> Int {
            try XCTUnwrap(descriptor.estimatedMemory(forOptions: options)).residentByteCount
        }
        let packed = [NFKModelLoadOptionQuantizationBits: 4, NFKModelLoadOptionResidency: NFKModelResidencyResident] as [String: Any]
        let short = try resident(packed.merging([NFKModelLoadOptionContextLength: 4096]) { $1 })
        let long = try resident(packed.merging([NFKModelLoadOptionContextLength: 65536]) { $1 })
        let longPacked = try resident(packed.merging([NFKModelLoadOptionContextLength: 65536, NFKModelLoadOptionCacheBits: 8]) { $1 })
        print("VALIDATION ESTIMATE qwen3.8-27b 4-bit: 4K \(short), 64K \(long), 64K with an 8-bit cache \(longPacked) bytes")
        // 16 attention layers × 4 key-value heads × 256 wide × keys and values × 2 bytes a position.
        XCTAssertEqual(long - short, 16 * 4 * 256 * 2 * 2 * (65536 - 4096))
        // An 8-bit element with a 16-bit scale and bias for each group of 64 takes 1.0625 bytes where 2 did.
        XCTAssertEqual(long - longPacked, 16 * 4 * 256 * 2 * 65536 / 16 * 15, "the 8-bit cache's saving")
        XCTAssertLessThan(short, 16_500_000_000)
        XCTAssertLessThan(longPacked, 18_500_000_000)
        XCTAssertGreaterThan(try resident([NFKModelLoadOptionResidency: NFKModelResidencyResident]), 50_000_000_000)
    }

    // A timing, run where `IK_TIME_QWEN3_8_27B` is set in the environment: Qwen3.8-27B packed to 4 bits as the
    // catalog loads it, 32 tokens decoded after a 64-token prompt, then a 23 000-token prompt prefilled through
    // the cache and 32 tokens decoded after it, with the memory MLX holds at each stage.
    func testTheQwen3_8_27BPrefillAndDecodeRates() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard environment["IK_TIME_QWEN3_8_27B"] != nil, let release = environment["IK_VAL_QWEN3_8_27B"] else {
            throw XCTSkip("set IK_TIME_QWEN3_8_27B and IK_VAL_QWEN3_8_27B to time the 27B")
        }
        func gib(_ bytes: Int) -> String { String(format: "%.2f GiB", Double(bytes) / Double(1 << 30)) }
        func memory(_ stage: String) {
            print("VALIDATION TIMING qwen3.8-27b memory \(stage): active \(gib(MLX.Memory.activeMemory)), cache "
                  + "\(gib(MLX.Memory.cacheMemory)), peak \(gib(MLX.Memory.peakMemory))")
        }
        NFKMLXGPU.applyStandingLimits()
        let directory = URL(fileURLWithPath: release)
        let loading = Date()
        let net = NFKMLXHybridLanguage.makeNet(
            try NFKMLXHybridLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json")))
        try NFKMLXHybridLanguage.loadQuantizedWeights(into: net, fromDirectory: directory, bits: 4, includeEmbeddings: true)
        try NFKMLXQuantization.quantize(module: net, bits: 4, groupSize: 64)
        eval(net)
        let loaded = Date().timeIntervalSince(loading)
        memory("after load")
        MLX.Memory.clearCache()
        MLX.Memory.peakMemory = 0
        memory("after clearing")

        func decodeRate(after prompt: [Int]) -> (prefill: Double, decode: Double, cache: Int) {
            let cache = NFKMLXHybridCache(configuration: net.configuration, quantization: .init(bits: 8, groupSize: 64))
            let prefilling = Date()
            let logits = net.prefill(prompt, cache: cache)
            eval(logits)
            let prefill = Date().timeIntervalSince(prefilling)
            // Pipelined as the backend's loop runs: each pass is built on the previous token before it is read.
            let decoding = Date()
            var token = logits.argMax()
            asyncEval(token)
            for _ in 0 ..< 32 {
                let following = net.lastLogits(token.reshaped([1, 1]), cache: cache)[0].argMax()
                asyncEval(following)
                _ = token.item(Int.self)
                token = following
            }
            _ = token.item(Int.self)
            return (prefill, Date().timeIntervalSince(decoding), cache.allocatedBytes)
        }
        let short = decodeRate(after: (0 ..< 64).map { ($0 * 7919) % 150_000 + 1000 })
        memory("after the short run")
        let long = decodeRate(after: (0 ..< 23_000).map { ($0 * 7919) % 150_000 + 1000 })
        memory("after the long run")
        print("VALIDATION TIMING qwen3.8-27b 4-bit, 8-bit cache: load \(Int(loaded)) s; after 64 tokens decode "
              + "\(String(format: "%.2f", 32 / short.decode)) tokens/s; prefill 23000 tokens in "
              + "\(String(format: "%.1f", long.prefill)) s (\(String(format: "%.1f", 23_000 / long.prefill)) tokens/s), "
              + "then decode \(String(format: "%.2f", 32 / long.decode)) tokens/s, cache \(gib(long.cache))")
    }

    // A timing of the served path, run where `IK_TIME_QWEN3_8_27B_SERVED` is set: the catalog's qwen3.8-27b
    // hosted by NFKInferenceServer with 4-bit weights, a 64K window, and an 8-bit cache, asked through the
    // streamed chat route at temperature 0 as a client asks it. Each stage prints the process footprint and the
    // memory MLX holds; each reply prints when its first reasoning and first content delta arrived and its rate.
    func testTheQwen3_8_27BServedThroughTheChatRoute() async throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard environment["IK_TIME_QWEN3_8_27B_SERVED"] != nil, let release = environment["IK_VAL_QWEN3_8_27B"] else {
            throw XCTSkip("set IK_TIME_QWEN3_8_27B_SERVED and IK_VAL_QWEN3_8_27B to time the served 27B")
        }
        func gb(_ bytes: Int) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }
        func memory(_ stage: String) {
            var info = rusage_info_v4()
            _ = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
            }
            print("VALIDATION TIMING qwen3.8-27b served \(stage): footprint \(gb(Int(info.ri_phys_footprint))), lifetime peak "
                  + "\(gb(Int(info.ri_lifetime_max_phys_footprint))); MLX active \(gb(MLX.Memory.activeMemory)), cache "
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
        memory("before load")

        func ask(_ prompt: String, maxTokens: Int, label: String) async throws {
            var request = URLRequest(url: try XCTUnwrap(server.localBaseURL).appendingPathComponent("chat/completions"))
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = 7200
            request.httpBody = try JSONSerialization.data(withJSONObject: [
                "model": "agent", "messages": [["role": "user", "content": prompt]], "max_tokens": maxTokens,
                "temperature": 0, "stream": true, "stream_options": ["include_usage": true],
            ] as [String: Any])
            let start = Date()
            var firstReasoning: Date?, firstContent: Date?, last = start
            var reasoningDeltas = 0, contentDeltas = 0
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
                    if let text = delta["reasoning_content"] as? String ?? delta["reasoning"] as? String, !text.isEmpty {
                        reasoningDeltas += 1
                        firstReasoning = firstReasoning ?? Date()
                        last = Date()
                    }
                    if let text = delta["content"] as? String, !text.isEmpty {
                        contentDeltas += 1
                        firstContent = firstContent ?? Date()
                        last = Date()
                    }
                }
            }
            let first = [firstReasoning, firstContent].compactMap { $0 }.min()
            let deltas = reasoningDeltas + contentDeltas
            let rate = first.map { deltas > 1 ? Double(deltas - 1) / last.timeIntervalSince($0) : 0 } ?? 0
            func seconds(_ date: Date?) -> String { date.map { String(format: "%.1f s", $0.timeIntervalSince(start)) } ?? "none" }
            print("VALIDATION TIMING qwen3.8-27b served \(label): prompt \(usage["prompt_tokens"] ?? "?") tokens, "
                  + "completion \(usage["completion_tokens"] ?? "?"); first reasoning \(seconds(firstReasoning)), first content "
                  + "\(seconds(firstContent)); \(reasoningDeltas) reasoning and \(contentDeltas) content deltas at "
                  + "\(String(format: "%.2f", rate)) deltas/s; whole reply \(String(format: "%.1f", last.timeIntervalSince(start))) s")
            memory("after \(label)")
        }
        let loading = Date()
        try await ask("Say hello.", maxTokens: 8, label: "the warm-up")
        print("VALIDATION TIMING qwen3.8-27b served load and warm-up \(Int(Date().timeIntervalSince(loading))) s")
        try await ask("Write a short paragraph about the sea.", maxTokens: 200, label: "the short prompt")
        let docs = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Docs")
        var text = ""
        for name in try FileManager.default.contentsOfDirectory(atPath: docs.path).sorted() where name.hasSuffix(".md") {
            text += (try? String(contentsOf: docs.appendingPathComponent(name), encoding: .utf8)) ?? ""
            if text.count > 88_000 {
                break
            }
        }
        try await ask(String(text.prefix(88_000)) + "\n\nSummarize the documents above in five sentences.", maxTokens: 200,
                      label: "the long prompt")
    }
}
