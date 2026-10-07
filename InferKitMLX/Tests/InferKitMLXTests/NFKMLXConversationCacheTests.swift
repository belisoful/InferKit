//
//  NFKMLXConversationCacheTests.swift
//  InferKitMLXTests
//
//  The language backend's prompt caches for named conversations, on the tiny configuration with
//  random weights and a hand-written byte-level vocabulary. What is asserted is the cached share each
//  run reports, so the tests need no release. Runs where MLX has a Metal library.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXConversationCacheTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// Two prompts where the second extends the first, and one that shares no prefix with either.
    private let opening = "hello hello"
    private let continuation = "hello hello hello hello"
    private let unrelated = "he he he"

    private func backend() throws -> NFKMLXLanguageBackend {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXConversationCacheTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let vocabulary: [String: Int] = ["h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6, "Ġ": 7]
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\nh e\nl l\nhe ll\nhell o\n".write(
            to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        let tokenizer = try NFKTokenizer(forManifest: ["tokenizer": ["type": "bpe-bytelevel"]], directory: directory)
        return NFKMLXLanguageBackend(net: NFKMLXLanguage.makeNet(.tiny), tokenizer: tokenizer, identifier: "tiny")
    }

    /// Runs `prompt` and answers the cached share of its input the run reports.
    private func cachedTokens(_ backend: NFKMLXLanguageBackend, _ prompt: String,
                              conversation: String? = nil, reuse: Bool? = nil) throws -> Int {
        var parameters: [String: Any] = [NFKParameterTemperature: 0, NFKParameterMaxTokens: 2]
        if let conversation {
            parameters[NFKParameterConversationKey] = conversation
        }
        if let reuse {
            parameters[NFKMLXGenerationParameterKey.reusesPromptCache] = reuse
        }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: prompt],
                                                                       parameters: parameters))
        let usage = try XCTUnwrap(result.output(forKey: NFKOutputUsage) as? [String: Int])
        return try XCTUnwrap(usage[NFKUsageCachedTokens])
    }

    func testInterleavedConversationsEachContinueTheirOwnCache() throws {
        try requireMLXRuntime()
        let single = try backend()
        XCTAssertEqual(try cachedTokens(single, opening, reuse: true), 0)
        XCTAssertEqual(try cachedTokens(single, unrelated, reuse: true), 0)
        XCTAssertEqual(try cachedTokens(single, continuation, reuse: true), 0,
                       "one retained cache holds only the last prompt, which shares nothing")

        let named = try backend()
        XCTAssertEqual(try cachedTokens(named, opening, conversation: "a"), 0)
        XCTAssertEqual(try cachedTokens(named, unrelated, conversation: "b"), 0)
        XCTAssertGreaterThan(try cachedTokens(named, continuation, conversation: "a"), 0,
                             "conversation a continues from its own cache")
        XCTAssertEqual(named.conversationCacheCount, 2)
        XCTAssertGreaterThan(named.conversationCacheBytes, 0)
    }

    func testARequestWithoutAConversationLeavesTheConversationsAlone() throws {
        try requireMLXRuntime()
        let backend = try backend()
        XCTAssertEqual(try cachedTokens(backend, opening, conversation: "a"), 0)
        XCTAssertEqual(try cachedTokens(backend, unrelated), 0)
        XCTAssertEqual(backend.promptCacheLength, 0, "a request that does not ask for reuse keeps no cache")
        XCTAssertGreaterThan(try cachedTokens(backend, continuation, conversation: "a"), 0)
    }

    func testAConversationThatTurnsReuseOffKeepsNoCache() throws {
        try requireMLXRuntime()
        let backend = try backend()
        XCTAssertEqual(try cachedTokens(backend, opening, conversation: "a", reuse: false), 0)
        XCTAssertEqual(try cachedTokens(backend, continuation, conversation: "a", reuse: false), 0)
        XCTAssertEqual(backend.conversationCacheCount, 0)
    }

    func testTheBudgetDropsTheLeastRecentConversationFirst() throws {
        try requireMLXRuntime()
        let backend = try backend()
        XCTAssertEqual(try cachedTokens(backend, opening, conversation: "a"), 0)
        XCTAssertEqual(try cachedTokens(backend, unrelated, conversation: "b"), 0)
        backend.conversationCacheByteBudget = backend.conversationCacheBytes - 1
        XCTAssertEqual(backend.conversationCacheCount, 1, "lowering the budget drops down to it at once")
        XCTAssertEqual(try cachedTokens(backend, continuation, conversation: "a"), 0,
                       "a was the least recently used, so it went")
        XCTAssertEqual(backend.conversationCacheCount, 1, "running a again dropped b to make room")
    }

    func testTheRunningConversationKeepsItsCacheEvenPastTheBudget() throws {
        try requireMLXRuntime()
        let backend = try backend()
        backend.conversationCacheByteBudget = 0
        XCTAssertEqual(try cachedTokens(backend, opening, conversation: "a"), 0)
        XCTAssertEqual(backend.conversationCacheCount, 1)
        XCTAssertGreaterThan(try cachedTokens(backend, continuation, conversation: "a"), 0)
        XCTAssertEqual(try cachedTokens(backend, unrelated, conversation: "b"), 0)
        XCTAssertEqual(backend.conversationCacheCount, 1, "b's run dropped a, the one not running")
    }

    func testTheBackendStatusReportsTheCaches() throws {
        try requireMLXRuntime()
        let backend = try backend()
        XCTAssertEqual(backend.backendStatus["conversation_caches"] as? Int, 0)
        XCTAssertEqual(backend.backendStatus["conversation_cache_byte_budget"] as? Int, 2 << 30)
        XCTAssertEqual(try cachedTokens(backend, opening, conversation: "a"), 0)
        XCTAssertEqual(try cachedTokens(backend, opening, reuse: true), 0)
        let status = backend.backendStatus
        XCTAssertEqual(status["conversation_caches"] as? Int, 1)
        XCTAssertEqual(status["conversation_cache_bytes"] as? Int, backend.conversationCacheBytes)
        XCTAssertGreaterThan(status["prompt_cache_length"] as? Int ?? 0, 0, "the unnamed retained cache")
        backend.conversationCacheByteBudget = 0
        XCTAssertEqual(backend.backendStatus["conversation_caches"] as? Int, 0, "lowering the budget refreshes it")
        XCTAssertEqual(backend.backendStatus["conversation_cache_byte_budget"] as? Int, 0)
    }

    func testResettingDropsOneConversationOrAll() throws {
        try requireMLXRuntime()
        let backend = try backend()
        XCTAssertEqual(try cachedTokens(backend, opening, conversation: "a"), 0)
        XCTAssertEqual(try cachedTokens(backend, unrelated, conversation: "b"), 0)
        backend.resetPromptCache(forConversation: "a")
        XCTAssertEqual(backend.conversationCacheCount, 1)
        XCTAssertEqual(try cachedTokens(backend, continuation, conversation: "a"), 0)
        backend.resetPromptCache()
        XCTAssertEqual(backend.conversationCacheCount, 0)
    }
}
