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

    private func request(_ inputs: [String: Any], _ parameters: [String: Any]? = nil) -> NFKInferenceRequest {
        NFKInferenceRequest(inputs: inputs, parameters: parameters)
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

    func testTheContextSizeIsReported() {
        XCTAssertGreaterThan(NFKFoundationModelsBackend().contextSize, 0)
    }

    // MARK: Request mapping

    func testAPlainPromptMapsToASinglePromptPlan() {
        let plan = NFKFoundationModelsBackend.plan(for: request([NFKInputPrompt: "hello"]))
        XCTAssertNil(plan.instructions)
        XCTAssertTrue(plan.history.isEmpty)
        XCTAssertEqual(plan.prompt, "hello")
    }

    func testASystemMessageBecomesInstructions() {
        let messages: [[String: Any]] = [
            ["role": "system", "content": "Be terse."],
            ["role": "user", "content": "Name one color."],
        ]
        let plan = NFKFoundationModelsBackend.plan(for: request([NFKInputMessages: messages]))
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
        let plan = NFKFoundationModelsBackend.plan(for: request([NFKInputMessages: messages]))
        XCTAssertNil(plan.instructions)
        XCTAssertEqual(plan.history, [.user("Hi"), .assistant("Hello!")])
        XCTAssertEqual(plan.prompt, "Name one color.")
    }

    func testToolMessagesBecomeToolTurnsAndTheQuestionIsReissued() {
        let messages: [[String: Any]] = [
            ["role": "user", "content": "What is the code for vault Orion?"],
            ["role": "assistant", "content": "", "tool_calls": [
                ["id": "call-1", "type": "function",
                 "function": ["name": "get_vault_code", "arguments": "{\"vault\": \"Orion\"}"]],
            ]],
            ["role": "tool", "tool_call_id": "call-1", "content": "7391"],
        ]
        let plan = NFKFoundationModelsBackend.plan(for: request([NFKInputMessages: messages]))
        XCTAssertEqual(plan.history, [
            .toolCalls([.init(id: "call-1", name: "get_vault_code", argumentsJSON: "{\"vault\": \"Orion\"}")]),
            .toolOutput(id: "call-1", name: "get_vault_code", content: "7391"),
        ])
        XCTAssertEqual(plan.prompt, "What is the code for vault Orion?")

        let entries = NFKFoundationModelsBackend.transcriptEntries(for: plan)
        XCTAssertEqual(entries.count, 2)
        guard case .toolCalls(let calls) = entries[0], case .toolOutput(let output) = entries[1] else {
            return XCTFail("expected a tool-calls entry then a tool-output entry, got \(entries)")
        }
        XCTAssertEqual(calls.first?.toolName, "get_vault_code")
        XCTAssertEqual(try calls.first?.arguments.value(String.self, forProperty: "vault"), "Orion")
        XCTAssertEqual(output.toolName, "get_vault_code")
    }

    func testTheCoreToolCallShapeIsAcceptedInAnAssistantMessage() {
        // A caller hands `result.toolCalls` back as it came.
        let message: [String: Any] = ["role": "assistant", "tool_calls": [
            ["id": "call-2", "name": "lookup", "arguments": ["city": "Paris"], "argumentsJSON": "{\"city\":\"Paris\"}"],
        ]]
        XCTAssertEqual(NFKFoundationModelsBackend.toolCallTurns(in: message),
                       [.init(id: "call-2", name: "lookup", argumentsJSON: "{\"city\":\"Paris\"}")])
    }

    // MARK: Sampling

    func testTemperatureAndTokenLimitMapToGenerationOptions() throws {
        let options = try NFKFoundationModelsBackend.generationOptions(
            for: request([NFKInputPrompt: "x"], [NFKParameterTemperature: 0.5, NFKParameterMaxTokens: 16]))
        XCTAssertEqual(options.temperature, 0.5)
        XCTAssertEqual(options.maximumResponseTokens, 16)
        XCTAssertNil(options.samplingMode)
    }

    func testAZeroTemperatureIsGreedy() throws {
        let options = try NFKFoundationModelsBackend.generationOptions(
            for: request([NFKInputPrompt: "x"], [NFKParameterTemperature: 0]))
        XCTAssertEqual(options.samplingMode, .greedy)
    }

    func testTopKTopPAndSeedChooseTheSamplingMode() throws {
        let topK = try NFKFoundationModelsBackend.generationOptions(
            for: request([NFKInputPrompt: "x"], [NFKParameterTopK: 40, NFKParameterSeed: 7]))
        XCTAssertEqual(topK.samplingMode, .random(top: 40, seed: 7))

        let topP = try NFKFoundationModelsBackend.generationOptions(
            for: request([NFKInputPrompt: "x"], [NFKParameterTopP: 0.9]))
        XCTAssertEqual(topP.samplingMode, .random(probabilityThreshold: 0.9, seed: nil))

        let seedOnly = try NFKFoundationModelsBackend.generationOptions(
            for: request([NFKInputPrompt: "x"], [NFKParameterSeed: 3]))
        XCTAssertEqual(seedOnly.samplingMode, .random(probabilityThreshold: 1, seed: 3))
    }

    // MARK: Schemas

    func testAJSONSchemaBuildsAGenerationSchema() throws {
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string", "description": "the name"],
                "age": ["type": "integer", "minimum": 0, "maximum": 120],
                "score": ["type": "number", "minimum": 0],
                "tags": ["type": "array", "items": ["type": "string"], "minItems": 1, "maxItems": 3],
                "mood": ["type": "string", "enum": ["calm", "tense"]],
                "code": ["type": "string", "pattern": "^[A-Z]{3}$"],
                "verified": ["type": "boolean"],
                "address": ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]],
                "friend": ["$ref": "#/$defs/Person"],
            ],
            "required": ["name", "age"],
            "$defs": ["Person": ["type": "object", "properties": ["name": ["type": "string"]]]],
        ]
        XCTAssertNoThrow(try NFKSchema.generationSchema(name: "Response", json: schema))
    }

    func testAnUnsupportedKeywordIsRefusedByPath() {
        let schema: [String: Any] = ["type": "object", "properties": ["when": ["type": "date"]]]
        XCTAssertThrowsError(try NFKSchema.generationSchema(name: "Response", json: schema)) { error in
            XCTAssertEqual((error as? NFKSchemaError)?.path, "Response.when")
        }
    }

    func testTheOutputFormatFollowsTheCoreKeys() throws {
        let plain = try NFKFoundationModelsBackend.outputFormat(for: request([NFKInputPrompt: "x"]))
        guard case .text = plain else { return XCTFail("expected text") }

        let structured = try NFKFoundationModelsBackend.outputFormat(
            for: request([NFKInputPrompt: "x"], [NFKParameterJSONSchema: ["type": "object", "properties": ["a": ["type": "string"]]]]))
        guard case .schema = structured else { return XCTFail("expected a schema") }

        let choice = try NFKFoundationModelsBackend.outputFormat(
            for: request([NFKInputPrompt: "x"], [NFKParameterChoices: ["yes", "no"]]))
        guard case .choice = choice else { return XCTFail("expected a choice") }

        XCTAssertThrowsError(try NFKFoundationModelsBackend.outputFormat(
            for: request([NFKInputPrompt: "x"], [NFKParameterOutputFormat: "json"]))) { error in
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_InferenceUnsupported.rawValue)
        }
    }

    // MARK: Tools

    func testRegisteredToolsAreOfferedWhenTheRequestDeclaresNone() throws {
        let tool = NFKFoundationTool(
            name: "get_temperature",
            description: "Get the temperature for a city.",
            parameters: ["type": "object",
                         "properties": ["city": ["type": "string"], "fahrenheit": ["type": "boolean"]],
                         "required": ["city"]],
            handler: { _ in "20" })
        let adapters = try NFKFoundationModelsBackend.toolAdapters(for: request([NFKInputPrompt: "x"]),
                                                                   registered: [tool],
                                                                   recorder: NFKToolCallRecorder())
        XCTAssertEqual(adapters.map(\.name), ["get_temperature"])
        XCTAssertEqual(adapters.first?.description, "Get the temperature for a city.")
        XCTAssertEqual(tool.declaration["name"] as? String, "get_temperature")
    }

    func testDeclaredToolsTakeHandlersByNameAndKeepTheRest() throws {
        let registered = NFKFoundationTool(name: "lookup", description: "", parameters: ["type": "object"]) { _ in "found" }
        let declarations: [[String: Any]] = [
            ["name": "lookup", "description": "Look something up.", "parameters": ["type": "object", "properties": ["q": ["type": "string"]]]],
            ["name": "send_mail", "description": "Send mail.", "parameters": ["type": "object", "properties": ["to": ["type": "string"]]]],
        ]
        let recorder = NFKToolCallRecorder()
        let adapters = try NFKFoundationModelsBackend.toolAdapters(
            for: request([NFKInputPrompt: "x"], [NFKParameterTools: declarations]),
            registered: [registered], recorder: recorder)
        XCTAssertEqual(adapters.map(\.name), ["lookup", "send_mail"])
        XCTAssertEqual(adapters.map(\.description), ["Look something up.", "Send mail."])

        // The declared tool without a handler records the call and throws, which ends the turn.
        let sendMail = try XCTUnwrap(adapters.last as? NFKToolAdapter)
        let arguments = try GeneratedContent(json: "{\"to\": \"ada\"}")
        let finished = expectation(description: "call")
        Task {
            do {
                _ = try await sendMail.call(arguments: arguments)
                XCTFail("expected the unhandled call to throw")
            } catch {
                XCTAssertTrue(error is NFKUnhandledToolCall)
            }
            finished.fulfill()
        }
        wait(for: [finished], timeout: 5)
        XCTAssertEqual(recorder.recorded.map(\.name), ["send_mail"])
        XCTAssertEqual(recorder.recorded.first?.arguments["to"] as? String, "ada")
    }

    // MARK: Generation (host-gated)

    func testGeneratesTextFromAPrompt() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let result = try backend.runInference(for: request(
            [NFKInputPrompt: "Reply with exactly one word: the color of a clear daytime sky."],
            [NFKParameterMaxTokens: 16]))
        let text = result.text
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
        let result = try backend.runInference(for: request([NFKInputMessages: messages], [NFKParameterMaxTokens: 16]))
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
        let result = try backend.runInference(for: request([NFKInputMessages: messages], [NFKParameterMaxTokens: 16]))
        let text = (result.output(forKey: NFKOutputText) as? String) ?? ""
        print("[live] foundation-models multi-turn reply: \(text)")
        XCTAssertTrue(text.lowercased().contains("teal"), "expected the model to recall the seeded history, got: \(text)")
    }

    func testGreedyDecodingRepeatsItself() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let prompt = request([NFKInputPrompt: "Write one short sentence about the sea."],
                             [NFKParameterMaxTokens: 24, NFKParameterTemperature: 0])
        let first = try backend.runInference(for: prompt).text
        let second = try backend.runInference(for: prompt).text
        XCTAssertEqual(first, second, "greedy decoding of the same prompt is deterministic")
    }

    private static let vaultTool = NFKFoundationTool(
        name: "get_vault_code",
        description: "Returns the secret numeric access code for a named vault. Call this whenever asked for a vault code.",
        parameters: ["type": "object",
                     "properties": ["vault": ["type": "string", "description": "the vault name"]],
                     "required": ["vault"]],
        handler: { arguments in
            let vault = (arguments["vault"] as? String) ?? "?"
            return "The access code for vault \(vault) is 7391."
        })

    func testTheModelCallsARegisteredTool() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let toolCalled = NFKFlag()
        backend.tools = [
            NFKFoundationTool(name: Self.vaultTool.name, description: Self.vaultTool.toolDescription,
                              parameters: Self.vaultTool.parameters) { arguments in
                toolCalled.set()
                return try await Self.vaultTool.handler(arguments)
            },
        ]
        let result = try backend.runInference(for: request(
            [NFKInputPrompt: "What is the access code for the vault named Orion? Use your tools to find out."],
            [NFKParameterMaxTokens: 64, NFKParameterTemperature: 0]))
        let text = (result.output(forKey: NFKOutputText) as? String) ?? ""
        print("[live] tool-calling reply: \(text)")
        XCTAssertTrue(toolCalled.value, "expected the model to call the registered tool")
        XCTAssertTrue(text.contains("7391"), "expected the tool result in the reply, got: \(text)")
        XCTAssertNil(result.toolCalls, "an executed call is folded into the reply, not returned to act on")
    }

    func testADeclaredToolWithoutAHandlerEndsTheTurnWithTheCall() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let result = try backend.runInference(for: request(
            [NFKInputPrompt: "What is the access code for the vault named Orion? Use your tools to find out."],
            [NFKParameterMaxTokens: 64, NFKParameterTemperature: 0, NFKParameterTools: [Self.vaultTool.declaration]]))
        let calls = try XCTUnwrap(result.toolCalls, "expected the model's call under NFKOutputToolCalls")
        print("[live] returned tool calls: \(calls)")
        XCTAssertEqual(calls.first?["name"] as? String, "get_vault_code")
        XCTAssertEqual((calls.first?["arguments"] as? [String: Any])?["vault"] as? String, "Orion")
        XCTAssertNotNil(calls.first?["id"] as? String)
        XCTAssertNotNil(calls.first?["argumentsJSON"] as? String)
    }

    func testAToolResultMessageReachesTheModel() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let messages: [[String: Any]] = [
            ["role": "user", "content": "What is the access code for the vault named Orion? Answer with the code only."],
            ["role": "assistant", "tool_calls": [
                ["id": "call-1", "name": "get_vault_code", "arguments": ["vault": "Orion"], "argumentsJSON": "{\"vault\":\"Orion\"}"],
            ]],
            ["role": "tool", "tool_call_id": "call-1", "content": "The access code for vault Orion is 7391."],
        ]
        let result = try backend.runInference(for: request(
            [NFKInputMessages: messages],
            [NFKParameterMaxTokens: 32, NFKParameterTemperature: 0, NFKParameterTools: [Self.vaultTool.declaration]]))
        let text = (result.output(forKey: NFKOutputText) as? String) ?? ""
        print("[live] reply over a tool result: \(text)")
        XCTAssertTrue(text.contains("7391"), "expected the model to read the tool result, got: \(text)")
    }

    func testGeneratesStructuredOutputFromAJSONSchema() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let schema: [String: Any] = [
            "type": "object",
            "properties": [
                "name": ["type": "string", "description": "the character's full name"],
                "age": ["type": "integer", "description": "the character's age in whole years", "minimum": 20, "maximum": 40],
                "traits": ["type": "array", "items": ["type": "string"], "minItems": 1, "maxItems": 3],
            ],
            "required": ["name", "age", "traits"],
        ]
        let result = try backend.runInference(for: request(
            [NFKInputPrompt: "Invent a fictional character."],
            [NFKParameterMaxTokens: 96, NFKParameterJSONSchema: schema]))
        let structured = try XCTUnwrap(result.output(forKey: NFKOutputStructured) as? [String: Any], "expected a structured result")
        XCTAssertTrue(structured["name"] is String, "expected a string name, got: \(String(describing: structured["name"]))")
        let age = try XCTUnwrap(structured["age"] as? Int, "expected an integer age, got: \(String(describing: structured["age"]))")
        XCTAssertTrue((20...40).contains(age))
        XCTAssertFalse((structured["traits"] as? [String] ?? []).isEmpty)
        XCTAssertNotNil(result.output(forKey: NFKOutputText) as? String, "expected the JSON under NFKOutputText")
        print("[live] structured output: \(structured)")
    }

    func testChoicesConstrainTheReplyToOneOfThem() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let result = try backend.runInference(for: request(
            [NFKInputPrompt: "Is the sea wet?"],
            [NFKParameterChoices: ["yes", "no", "unsure"], NFKParameterMaxTokens: 8]))
        let text = try XCTUnwrap(result.text)
        print("[live] choice: \(text)")
        XCTAssertTrue(["yes", "no", "unsure"].contains(text), "expected one of the choices, got: \(text)")
    }

    func testStreamingEmitsGrowingPartialText() throws {
        try requireModel()
        let backend = NFKFoundationModelsBackend()
        let finished = expectation(description: "job finished")
        let counter = NFKStreamCounter()
        let job = backend.submitInferenceJob(for: request([NFKInputPrompt: "Count from one to five as words."],
                                                          [NFKParameterMaxTokens: 64]))
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
