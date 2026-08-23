//
//  NFKFoundationModelsBackendTests.swift
//  InferKitFoundationModelsTests
//
//  Contract and request-mapping tests always run; generation tests run only where the system
//  language model is available (Apple Intelligence enabled on supported hardware) and are skipped
//  elsewhere, so CI stays green without the model.
//

import XCTest
import FoundationModels
import InferKit
@testable import InferKitFoundationModels

final class NFKFoundationModelsBackendTests: XCTestCase {

    private func requireModel() throws {
        try XCTSkipUnless(SystemLanguageModel.default.availability == .available,
                          "system language model unavailable on this host")
    }

    // MARK: Contract

    func testTheBackendReportsItsIdentifier() {
        let backend = NFKFoundationModelsBackend()
        XCTAssertEqual(backend.backendIdentifier, "foundation-models")
    }

    func testReadinessMatchesSystemAvailability() {
        let backend = NFKFoundationModelsBackend()
        XCTAssertEqual(backend.isReady, SystemLanguageModel.default.availability == .available)
    }

    // MARK: Request mapping

    func testAPlainPromptMapsToASinglePromptPlan() {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"], parameters: nil)
        let plan = NFKFoundationModelsBackend.plan(for: request)
        XCTAssertNil(plan.instructions)
        XCTAssertTrue(plan.history.isEmpty)
        XCTAssertEqual(plan.prompt, "hello")
    }

    func testASystemMessageBecomesInstructions() {
        let messages: [[String: Any]] = [
            ["role": "system", "content": "Be terse."],
            ["role": "user", "content": "Name one color."],
        ]
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: messages], parameters: nil)
        let plan = NFKFoundationModelsBackend.plan(for: request)
        XCTAssertEqual(plan.instructions, "Be terse.")
        XCTAssertTrue(plan.history.isEmpty)
        XCTAssertEqual(plan.prompt, "Name one color.")
    }

    func testMultiTurnHistorySplitsIntoHistoryAndPrompt() {
        let messages: [[String: Any]] = [
            ["role": "user", "content": "Hi"],
            ["role": "assistant", "content": "Hello!"],
            ["role": "user", "content": "Name one color."],
        ]
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: messages], parameters: nil)
        let plan = NFKFoundationModelsBackend.plan(for: request)
        XCTAssertNil(plan.instructions)
        XCTAssertEqual(plan.history, [
            .init(role: "user", content: "Hi"),
            .init(role: "assistant", content: "Hello!"),
        ])
        XCTAssertEqual(plan.prompt, "Name one color.")
    }

    // MARK: Tools

    func testToolAdapterBuildsFromARuntimeTool() throws {
        let tool = NFKFoundationTool(
            name: "get_temperature",
            description: "Get the temperature for a city.",
            parameters: [
                NFKFoundationToolParameter(name: "city", description: "the city", type: .string, required: true),
                NFKFoundationToolParameter(name: "fahrenheit", description: "use Fahrenheit", type: .boolean, required: false),
            ],
            handler: { _ in "20" })
        let adapter = try NFKToolAdapter(tool: tool)
        XCTAssertEqual(adapter.name, "get_temperature")
        XCTAssertEqual(adapter.description, "Get the temperature for a city.")
    }

    func testStructuredSchemaBuildsFromFields() throws {
        let fields = [
            NFKFoundationToolParameter(name: "name", description: "the name", type: .string, required: true),
            NFKFoundationToolParameter(name: "age", description: "the age", type: .integer, required: true),
        ]
        _ = try NFKSchema.generationSchema(name: "Response", description: "a response", properties: fields)
    }

    // MARK: Generation (host-gated)

    func testGeneratesTextFromAPrompt() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "Reply with exactly one word: the color of a clear daytime sky."],
            parameters: [NFKParameterMaxTokens: 16])
        let result = try backend.runInference(for: request)
        let text = result.text                              // Swift-bridged convenience accessor
        XCTAssertNotNil(text)
        XCTAssertFalse(text!.isEmpty)
        print("[live] foundation-models reply: \(text!)")
    }

    func testMessagesCarryInstructionsIntoTheSession() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let messages: [[String: Any]] = [
            ["role": "system", "content": "Answer with a single word."],
            ["role": "user", "content": "Name any one primary color."],
        ]
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: messages],
                                          parameters: [NFKParameterMaxTokens: 16])
        let result = try backend.runInference(for: request)
        let text = result.output(forKey: NFKOutputText) as? String
        XCTAssertNotNil(text)
        print("[live] foundation-models chat reply: \(text!)")
    }

    func testMultiTurnHistorySeedsTheConversation() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        // The answer to the final turn depends on remembering the earlier turn.
        let messages: [[String: Any]] = [
            ["role": "user", "content": "My favorite color is teal. Remember it."],
            ["role": "assistant", "content": "Got it, teal."],
            ["role": "user", "content": "In one word, what is my favorite color?"],
        ]
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: messages],
                                          parameters: [NFKParameterMaxTokens: 16])
        let result = try backend.runInference(for: request)
        let text = (result.output(forKey: NFKOutputText) as? String) ?? ""
        print("[live] foundation-models multi-turn reply: \(text)")
        XCTAssertTrue(text.lowercased().contains("teal"), "expected the model to recall the seeded history, got: \(text)")
    }

    func testTheModelCallsARegisteredTool() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let toolCalled = NFKFlag()
        backend.tools = [
            NFKFoundationTool(
                name: "get_vault_code",
                description: "Returns the secret numeric access code for a named vault. Call this whenever asked for a vault code.",
                parameters: [NFKFoundationToolParameter(name: "vault", description: "the vault name", type: .string, required: true)],
                handler: { arguments in
                    toolCalled.set()
                    let vault = (arguments["vault"] as? String) ?? "?"
                    return "The access code for vault \(vault) is 7391."
                })
        ]
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "What is the access code for the vault named Orion? Use your tools to find out."],
            parameters: [NFKParameterMaxTokens: 64, NFKParameterTemperature: 0])
        let result = try backend.runInference(for: request)
        let text = (result.output(forKey: NFKOutputText) as? String) ?? ""
        print("[live] tool-calling reply: \(text)")
        XCTAssertTrue(toolCalled.value, "expected the model to call the registered tool")
        XCTAssertTrue(text.contains("7391"), "expected the tool result in the reply, got: \(text)")
    }

    func testGeneratesStructuredOutput() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        backend.responseSchema = [
            NFKFoundationToolParameter(name: "name", description: "the character's full name", type: .string, required: true),
            NFKFoundationToolParameter(name: "age", description: "the character's age in whole years", type: .integer, required: true),
        ]
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "Invent a fictional character with a full name and an age between 20 and 40."],
            parameters: [NFKParameterMaxTokens: 64])
        let result = try backend.runInference(for: request)
        let structured = result.output(forKey: NFKOutputStructured) as? [String: Any]
        XCTAssertNotNil(structured, "expected a structured result")
        XCTAssertTrue(structured?["name"] is String, "expected a string name, got: \(String(describing: structured?["name"]))")
        XCTAssertTrue(structured?["age"] is Int, "expected an integer age, got: \(String(describing: structured?["age"]))")
        XCTAssertNotNil(result.output(forKey: NFKOutputText) as? String, "expected the JSON under NFKOutputText")
        print("[live] structured output: \(structured ?? [:])")
    }

    func testStreamingEmitsGrowingPartialText() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "Count from one to five as words."],
            parameters: [NFKParameterMaxTokens: 64])

        let finished = expectation(description: "job finished")
        let counter = NFKStreamCounter()
        let job = backend.submitInferenceJob(for: request)
        job.progressHandler = { reporting in
            if let partial = reporting.partialResult?.output(forKey: NFKOutputText) as? String {
                counter.record(length: partial.count)
            }
        }
        job.completionHandler = { _ in finished.fulfill() }
        wait(for: [finished], timeout: 120)

        XCTAssertEqual(job.status, .succeeded)
        XCTAssertGreaterThan(counter.updates, 0, "expected streamed partial results")
        XCTAssertTrue(counter.monotonic, "streamed text should only grow")
    }
}

/// A thread-safe boolean, set from a tool handler that runs on the generation task.
private final class NFKFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _value = false
    func set() { lock.lock(); _value = true; lock.unlock() }
    var value: Bool { lock.lock(); defer { lock.unlock() }; return _value }
}

/// Progress handlers arrive on the streaming task's thread; the counter serializes the bookkeeping.
private final class NFKStreamCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var _updates = 0
    private var _lastLength = 0
    private var _monotonic = true

    func record(length: Int) {
        lock.lock()
        _updates += 1
        if length < _lastLength { _monotonic = false }
        _lastLength = length
        lock.unlock()
    }

    var updates: Int { lock.lock(); defer { lock.unlock() }; return _updates }
    var monotonic: Bool { lock.lock(); defer { lock.unlock() }; return _monotonic }
}
