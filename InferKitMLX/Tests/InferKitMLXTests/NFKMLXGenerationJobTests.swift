//
//  NFKMLXGenerationJobTests.swift
//  InferKitMLXTests
//
//  The job form of the text-generation backends: the reply reported as it generates, the chain
//  before the answer, and a cancellation that ends generation early. The decoder backend runs a
//  scripted decoder, and the language backend a tiny net with random weights. Runs where MLX has a
//  Metal library.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXGenerationJobTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXGenerationJobTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    private func tokenizer(_ vocabulary: [String: Int]) throws -> NFKTokenizer {
        let directory = try temporaryDirectory()
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        return try NFKTokenizer(forManifest: ["tokenizer": ["type": "bpe-bytelevel"]], directory: directory)
    }

    /// What a job reported, collected from its progress handler on the generation thread.
    private final class Reports: @unchecked Sendable {
        private let lock = NSLock()
        private var collected: [(reasoning: String, text: String)] = []
        var all: [(reasoning: String, text: String)] { lock.withLock { collected } }
        func add(_ partial: NFKInferenceResult?) {
            guard let partial else { return }
            let entry = (partial.output(forKey: NFKOutputReasoning) as? String ?? "", partial.text ?? "")
            lock.withLock { collected.append(entry) }
        }
    }

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        var value: Int { lock.withLock { count } }
        func increment() { lock.withLock { count += 1 } }
    }

    // MARK: The decoder backend

    /// The pieces of the scripted decoder's vocabulary: the prompt's letters, then whole words.
    private static let vocabulary: [String: Int] = ["<": 0, "t": 1, "h": 2, "i": 3, "n": 4, "k": 5, ">": 6,
                                                    "Ġweigh": 10, "Ġit": 11, "</think>": 12, "Ġfour": 13,
                                                    "Ġdone": 14, "<|end|>": 15]

    /// A decoder that answers `script[step]` at each step, past the prompt's `promptLength` tokens, and
    /// waits for `start` before its first step.
    private func scriptedBackend(_ script: @escaping (Int) -> Int, promptLength: Int, start: DispatchSemaphore,
                                 steps: Counter) throws -> NFKMLXDecoderBackend {
        let size = 16
        let backend = NFKMLXDecoderBackend(logits: { input in
            let length = input.dim(1)
            if length == promptLength { start.wait() }
            steps.increment()
            var row = [Float](repeating: 0, count: size)
            row[script(length - promptLength)] = 10
            return MLXArray(Array([[Float]](repeating: row, count: length).joined())).reshaped([1, length, size])
        }, tokenizer: try tokenizer(Self.vocabulary), identifier: "scripted",
           chatTemplate: "{% for message in messages %}{{ message.content }}{% endfor %}<think>", stopTokens: [15])
        backend.partialResultInterval = 0
        return backend
    }

    private func chat(maxTokens: Int) -> NFKInferenceRequest {
        NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "user", "content": "hi"]]],
                            parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: maxTokens])
    }

    func testADecoderJobReportsTheChainAsItGrowsThenTheAnswer() throws {
        try requireMLXRuntime()
        let script = [10, 11, 10, 12, 13, 14, 15]
        let start = DispatchSemaphore(value: 0)
        let request = chat(maxTokens: 32)
        let promptLength = "hi<think>".count
        let backend = try scriptedBackend({ script[$0] }, promptLength: promptLength, start: start, steps: Counter())
        XCTAssertEqual(try backend.tokenCount(for: request).intValue, promptLength)

        let reports = Reports()
        let finished = expectation(description: "finished")
        let job = backend.submitInferenceJob(for: request)
        job.progressHandler = { reports.add($0.partialResult) }
        job.completionHandler = { _ in finished.fulfill() }
        start.signal()
        wait(for: [finished], timeout: 30)

        XCTAssertEqual(reports.all.map(\.reasoning), ["weigh", "weigh it", "weigh it weigh", "weigh it weigh",
                                                      "weigh it weigh", "weigh it weigh"])
        XCTAssertEqual(reports.all.map(\.text), ["", "", "", "", "four", "four done"])
        let result = try XCTUnwrap(job.result)
        XCTAssertEqual(result.output(forKey: NFKOutputReasoning) as? String, "weigh it weigh")
        XCTAssertEqual(result.text, "four done")
    }

    func testCancellingADecoderJobEndsGenerationAtTheNextToken() throws {
        try requireMLXRuntime()
        let start = DispatchSemaphore(value: 0)
        let steps = Counter()
        let backend = try scriptedBackend({ _ in 10 }, promptLength: "hi<think>".count, start: start, steps: steps)
        let finished = expectation(description: "finished")
        let job = backend.submitInferenceJob(for: chat(maxTokens: 200))
        job.progressHandler = { job in
            if job.progress * 200 >= 2.5 { job.cancel() }
        }
        job.completionHandler = { _ in finished.fulfill() }
        start.signal()
        wait(for: [finished], timeout: 30)
        XCTAssertEqual(job.status, .cancelled)
        Thread.sleep(forTimeInterval: 0.3)
        XCTAssertEqual(steps.value, 3, "the run stopped at the token the job was cancelled on")
    }

    // MARK: The language backend

    /// A tiny net whose every token decodes to its own bracketed id, so a run's text names its tokens.
    private func bracketedBackend() throws -> (backend: NFKMLXLanguageBackend, net: NFKMLXLanguageNet, tokenizer: NFKTokenizer) {
        MLXRandom.seed(29)
        let net = NFKMLXLanguage.makeNet(.tiny)
        var vocabulary: [String: Int] = ["h": 0, "e": 1, "l": 2, "o": 3]
        for id in 4 ..< NFKMLXLanguageConfiguration.tiny.vocabularySize {
            vocabulary["[\(id)]"] = id
        }
        let tokenizer = try tokenizer(vocabulary)
        let backend = NFKMLXLanguageBackend(net: net, tokenizer: tokenizer, identifier: "tiny")
        backend.partialResultInterval = 0
        return (backend, net, tokenizer)
    }

    func testALanguageJobReportsTheChainAsItGrowsThenTheAnswer() throws {
        try requireMLXRuntime()
        let (backend, net, tokenizer) = try bracketedBackend()
        // A tiny net with random weights repeats one token under greedy decoding; a seeded sample varies.
        var options = NFKMLXGenerationOptions()
        options.temperature = 1
        options.seed = 5
        options.maxTokens = 16
        let produced = net.generate(prompt: tokenizer.encode("hello").map(\.intValue), options: options)
        let piece = { (token: Int) in tokenizer.decode([NSNumber(value: token)]) }
        // The chain closes on the first token from the fourth on that the run has not produced before.
        let closing = try XCTUnwrap((3 ..< produced.count - 1).first { !produced[..<$0].contains(produced[$0]) })
        let format = [piece(produced[0]), piece(produced[closing])]

        let reports = Reports()
        let finished = expectation(description: "finished")
        let job = backend.submitInferenceJob(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "hello"],
            parameters: [NFKParameterTemperature: 1, NFKParameterSeed: 5, NFKParameterMaxTokens: 16,
                         NFKMLXGenerationParameterKey.reasoningFormat: format]))
        job.progressHandler = { reports.add($0.partialResult) }
        job.completionHandler = { _ in finished.fulfill() }
        wait(for: [finished], timeout: 60)

        let result = try XCTUnwrap(job.result)
        let reasoning = try XCTUnwrap(result.output(forKey: NFKOutputReasoning) as? String)
        let answer = try XCTUnwrap(result.text)
        XCTAssertFalse(answer.isEmpty)
        for (earlier, later) in zip(reports.all, reports.all.dropFirst()) {
            XCTAssertTrue(later.reasoning.hasPrefix(earlier.reasoning))
            XCTAssertTrue(later.text.hasPrefix(earlier.text))
        }
        XCTAssertTrue(reports.all.contains { !$0.reasoning.isEmpty && $0.text.isEmpty }, "the chain comes first")
        XCTAssertEqual(reports.all.last?.reasoning, reasoning)
        XCTAssertEqual(reports.all.last?.text, answer)
    }

    func testCancellingALanguageJobEndsGenerationEarly() throws {
        try requireMLXRuntime()
        let (backend, _, _) = try bracketedBackend()
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"],
                                          parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 200,
                                                       NFKMLXGenerationParameterKey.reusesPromptCache: true])
        let promptLength = try backend.tokenCount(for: request).intValue
        let finished = expectation(description: "finished")
        let job = backend.submitInferenceJob(for: request)
        job.progressHandler = { job in
            if job.progress * 200 >= 2.5 { job.cancel() }
        }
        job.completionHandler = { _ in finished.fulfill() }
        wait(for: [finished], timeout: 60)
        XCTAssertEqual(job.status, .cancelled)
        XCTAssertNil(job.result)
        // The retained cache holds the prompt and every token fed back, and is read once the run ends.
        XCTAssertLessThan(backend.promptCacheLength, promptLength + 10, "the run stopped soon after the cancellation")
    }
}
