//
//  NFKFoundationToolBridgeTests.swift
//  InferKitFoundationModelsTests
//
//  Tools crossing between Apple's tool subsystem and the core's: a core tool on Apple's model, a
//  Foundation tool and an app's own Apple `Tool` as core tools, served over MCP, and the system
//  model's tool runs reported back. The conversions always run; the generation test runs where the
//  system language model is available.
//

import XCTest
import FoundationModels
import InferKit
@testable import InferKitFoundationModels

/// An app's own Apple tool, written the way Apple's documentation writes one.
private struct NFKForecastTool: Tool {
    let name = "get_forecast"
    let description = "Returns tomorrow's forecast for a city."

    @Generable
    struct Arguments {
        @Guide(description: "The city to forecast.")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        "Rain in \(arguments.city) tomorrow."
    }
}

final class NFKFoundationToolBridgeTests: XCTestCase {

    private let schema: [String: Any] = ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]]

    private func coreTool(counting calls: NFKCallCount) -> NFKInferenceTool {
        NFKInferenceTool(name: "get_weather", description: "Current weather in a city.", parameters: schema) { arguments, _ in
            calls.increment()
            return NFKToolResult(text: "18 C in \(arguments["city"] as? String ?? "")", outputs: ["celsius": 18], imagePNGs: nil)
        }
    }

    // MARK: A core tool on Apple's model

    func testACoreToolKeepsItsDeclarationAndRunsThroughItsOwnCheck() async throws {
        let calls = NFKCallCount()
        let tool = NFKFoundationTool(tool: coreTool(counting: calls))
        XCTAssertEqual(tool.name, "get_weather")
        XCTAssertEqual(tool.toolDescription, "Current weather in a city.")
        XCTAssertEqual(tool.parameters["required"] as? [String], ["city"])
        let result = try await tool.resultHandler(["city": "Paris"])
        XCTAssertEqual(result.text, "18 C in Paris")
        XCTAssertEqual(result.outputs["celsius"] as? Int, 18, "the outputs reach the application through the run's invocation")
        let refused = try await tool.resultHandler([:])
        XCTAssertTrue(refused.isError, "the core tool checks the arguments before it runs")
        XCTAssertEqual(calls.value, 1)
    }

    // MARK: Apple's tools as core tools

    func testAFoundationToolIsACoreToolWithABoundedWait() {
        let quick = NFKFoundationTool(name: "echo", description: "Echoes.", parameters: ["type": "object"]) { arguments in
            "echo \(arguments["word"] as? String ?? "")"
        }
        XCTAssertEqual(quick.inferenceTool.invoke(withArguments: ["word": "hi"]).text, "echo hi")
        XCTAssertEqual(quick.inferenceTool.declaration["name"] as? String, "echo")

        struct Broken: Error, LocalizedError { var errorDescription: String? { "the disk is full" } }
        let failing = NFKFoundationTool(name: "save", description: "Saves.", parameters: ["type": "object"]) { _ in throw Broken() }
        let failed = failing.inferenceTool.invoke(withArguments: [:])
        XCTAssertTrue(failed.isError)
        XCTAssertEqual(failed.text, "save failed: the disk is full")

        let slow = NFKFoundationTool(name: "slow", description: "Waits.", parameters: ["type": "object"]) { _ in
            try await Task.sleep(nanoseconds: 5_000_000_000)
            return "late"
        }
        slow.timeout = 0.2
        let started = Date()
        let timedOut = slow.inferenceTool.invoke(withArguments: [:])
        XCTAssertTrue(timedOut.isError)
        XCTAssertTrue(timedOut.text.contains("did not answer"), timedOut.text)
        XCTAssertLessThan(Date().timeIntervalSince(started), 3, "the wait ends at the deadline")
    }

    func testAnAppsOwnAppleToolReachesEveryModel() throws {
        let tool = try NFKFoundationTool(appleTool: NFKForecastTool())
        XCTAssertEqual(tool.name, "get_forecast")
        XCTAssertEqual(tool.toolDescription, "Returns tomorrow's forecast for a city.")
        let properties = try XCTUnwrap(tool.parameters["properties"] as? [String: Any], "\(tool.parameters)")
        XCTAssertNotNil(properties["city"], "the GenerationSchema encodes as a JSON Schema")
        XCTAssertEqual(tool.parameters["required"] as? [String], ["city"])

        let core = tool.inferenceTool
        XCTAssertEqual(core.invoke(withArguments: ["city": "Bergen"]).text, "Rain in Bergen tomorrow.")
        XCTAssertTrue(core.invoke(withArguments: [:]).isError, "the schema's required city is checked")
    }

    func testAFoundationToolIsServedOverMCP() throws {
        let server = NFKInferenceServer()
        server.port = 0
        server.loopbackOnly = true
        server.addTool(try NFKFoundationTool(appleTool: NFKForecastTool()).inferenceTool)
        try server.start()
        defer { server.stop() }
        let client = NFKMCPClient(endpointURL: try XCTUnwrap(server.localMCPURL), apiKey: nil)
        XCTAssertEqual(try client.tools().map(\.name), ["get_forecast"])
        XCTAssertEqual(try client.resultOfTool(named: "get_forecast", arguments: ["city": "Lyon"]).text, "Rain in Lyon tomorrow.")
    }

    // MARK: The conversation

    func testTheTranscriptReadsTheCoresToolTurn() {
        let calling = NFKToolMessages.assistantMessage(withText: nil,
                                                       toolCalls: [["id": "c1", "name": "get_weather", "argumentsJSON": "{\"city\":\"Oslo\"}"]],
                                                       record: nil, format: nil, source: nil)
        let failed = NFKToolMessages.message(forToolCallIdentifier: "c1", name: "get_weather",
                                             result: NFKToolResult(text: "no data", outputs: nil, imagePNGs: nil, isError: true))
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "user", "content": "Weather in Oslo?"], calling, failed]],
                                          parameters: nil)
        let plan = NFKFoundationModelsBackend.plan(for: request)
        XCTAssertEqual(plan.prompt, "Weather in Oslo?")
        XCTAssertEqual(plan.history.last, .toolOutput(id: "c1", name: "get_weather", content: "Error: no data"),
                       "a failed call says so, since the transcript has no error flag")
    }

    // MARK: The system model

    func testTheSystemModelCallsACoreToolAndReportsTheRun() throws {
        try XCTSkipUnless(SystemLanguageModel.default.availability == .available, "system language model unavailable on this host")
        let calls = NFKCallCount()
        let backend = NFKFoundationModelsBackend()
        backend.tools = [NFKFoundationTool(tool: NFKInferenceTool(
            name: "get_vault_code",
            description: "Returns the secret numeric access code for a named vault. Call this whenever asked for a vault code.",
            parameters: ["type": "object", "properties": ["vault": ["type": "string", "description": "the vault name"]], "required": ["vault"]]
        ) { arguments, _ in
            calls.increment()
            return NFKToolResult(text: "The access code for vault \(arguments["vault"] as? String ?? "?") is 7391.",
                                 outputs: ["code": 7391], imagePNGs: nil)
        })]
        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "What is the access code for the vault named Orion? Use your tools to find out."],
            parameters: [NFKParameterMaxTokens: 64, NFKParameterTemperature: 0]))
        let ran = result.output(forKey: NFKOutputToolInvocations) as? [NFKToolInvocation] ?? []
        XCTAssertGreaterThan(calls.value, 0, "the model called the core tool")
        XCTAssertEqual(ran.first?.toolName, "get_vault_code")
        XCTAssertEqual(ran.first?.result.outputs["code"] as? Int, 7391, "the tool's outputs reach the caller")
        XCTAssertTrue(result.text?.contains("7391") ?? false, result.text ?? "")
    }
}

/// A count a `@Sendable` handler can bump.
private final class NFKCallCount: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var value: Int { lock.withLock { count } }
    func increment() { lock.withLock { count += 1 } }
}
