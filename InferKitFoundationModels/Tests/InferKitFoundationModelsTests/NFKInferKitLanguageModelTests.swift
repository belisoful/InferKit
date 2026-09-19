//
//  NFKInferKitLanguageModelTests.swift
//  InferKitFoundationModelsTests
//
//  The capability reading and the request mapping run everywhere, because a transcript, generation
//  options, and a schema are macOS 26 types. The model and its executor need macOS 27, so their
//  tests skip below it.
//

import XCTest
import FoundationModels
import InferKit
@testable import InferKitFoundationModels

final class NFKInferKitLanguageModelTests: XCTestCase {

    // MARK: Capabilities

    func testADeclaredSchemaAndToolsBecomeCapabilities() {
        let capabilities = NFKInferKitLanguageModelCapabilities(backend: NFKFoundationModelsBackend())
        XCTAssertTrue(capabilities.guidedGeneration)
        XCTAssertTrue(capabilities.toolCalling)
        XCTAssertFalse(capabilities.vision, "the system language model takes no image")
    }

    func testABackendThatDeclaresNothingReportsNoCapabilities() {
        let capabilities = NFKInferKitLanguageModelCapabilities(backend: NFKEchoBackend())
        XCTAssertFalse(capabilities.guidedGeneration)
        XCTAssertFalse(capabilities.toolCalling)
        XCTAssertFalse(capabilities.vision)
    }

    func testAnImageInputIsVision() {
        let capabilities = NFKInferKitLanguageModelCapabilities(backend: NFKRemoteBackend(endpointURL: nil))
        XCTAssertTrue(capabilities.vision)
        XCTAssertTrue(capabilities.guidedGeneration)
        XCTAssertTrue(capabilities.toolCalling)
    }

    func testStatedCapabilitiesNeedNoDeclaration() {
        let capabilities = NFKInferKitLanguageModelCapabilities(guidedGeneration: false,
                                                                toolCalling: true,
                                                                vision: true)
        XCTAssertFalse(capabilities.guidedGeneration)
        XCTAssertTrue(capabilities.toolCalling)
        XCTAssertTrue(capabilities.vision)
    }

    // MARK: Transcript mapping

    func testATranscriptBecomesTheCoreMessages() {
        let transcript = Transcript(entries: [
            .instructions(Transcript.Instructions(segments: [.text(Transcript.TextSegment(content: "Be brief."))],
                                                  toolDefinitions: [])),
            .prompt(Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "Which bird?"))])),
            .response(Transcript.Response(assetIDs: [],
                                          segments: [.text(Transcript.TextSegment(content: "A gannet."))]))
        ])
        let messages = NFKInferKitLanguageModelRequest.messages(for: transcript)
        XCTAssertEqual(messages.count, 3)
        XCTAssertEqual(messages[0]["role"] as? String, "system")
        XCTAssertEqual(messages[0]["content"] as? String, "Be brief.")
        XCTAssertEqual(messages[1]["role"] as? String, "user")
        XCTAssertEqual(messages[1]["content"] as? String, "Which bird?")
        XCTAssertEqual(messages[2]["role"] as? String, "assistant")
        XCTAssertEqual(messages[2]["content"] as? String, "A gannet.")
    }

    func testAToolCallAndItsOutputBecomeTheWireShapes() throws {
        let arguments = try GeneratedContent(json: "{\"city\":\"Oslo\"}")
        let transcript = Transcript(entries: [
            .toolCalls(Transcript.ToolCalls([Transcript.ToolCall(id: "call-1",
                                                                 toolName: "weather",
                                                                 arguments: arguments)])),
            .toolOutput(Transcript.ToolOutput(id: "call-1",
                                              toolName: "weather",
                                              segments: [.text(Transcript.TextSegment(content: "rain"))]))
        ])
        let messages = NFKInferKitLanguageModelRequest.messages(for: transcript)
        XCTAssertEqual(messages[0]["role"] as? String, "assistant")
        let calls = try XCTUnwrap(messages[0]["tool_calls"] as? [[String: Any]])
        XCTAssertEqual(calls.first?["id"] as? String, "call-1")
        XCTAssertEqual(calls.first?["type"] as? String, "function")
        let function = try XCTUnwrap(calls.first?["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "weather")
        XCTAssertTrue((function["arguments"] as? String ?? "").contains("Oslo"))
        XCTAssertEqual(messages[1]["role"] as? String, "tool")
        XCTAssertEqual(messages[1]["tool_call_id"] as? String, "call-1")
        XCTAssertEqual(messages[1]["name"] as? String, "weather")
        XCTAssertEqual(messages[1]["content"] as? String, "rain")
    }

    // MARK: Option mapping

    func testTemperatureAndTheTokenLimitMapDirectly() {
        let options = GenerationOptions(temperature: 0.4, maximumResponseTokens: 64)
        let parameters = NFKInferKitLanguageModelRequest.parameters(for: options)
        XCTAssertEqual(parameters[NFKParameterTemperature] as? Double, 0.4)
        XCTAssertEqual(parameters[NFKParameterMaxTokens] as? Int, 64)
    }

    func testTheSamplingModeMapsToTheCoreKeys() throws {
        guard #available(macOS 27, iOS 27, *) else {
            throw XCTSkip("the sampling mode is opaque below macOS 27")
        }
        #if compiler(>=6.4)
        var options = GenerationOptions()
        NFKFoundationModelsBackend.setSamplingMode(.random(top: 12, seed: 7), on: &options)
        var parameters = NFKInferKitLanguageModelRequest.parameters(for: options)
        XCTAssertEqual(parameters[NFKParameterTopK] as? Int, 12)
        XCTAssertEqual((parameters[NFKParameterSeed] as? UInt64), 7)

        NFKFoundationModelsBackend.setSamplingMode(.greedy, on: &options)
        parameters = NFKInferKitLanguageModelRequest.parameters(for: options)
        XCTAssertEqual(parameters[NFKParameterTemperature] as? Int, 0)
        #endif
    }

    // MARK: Schema mapping

    func testASchemaBecomesAJSONSchemaObject() throws {
        let json: [String: Any] = ["type": "object",
                                   "properties": ["city": ["type": "string"]],
                                   "required": ["city"]]
        let schema = try NFKSchema.generationSchema(name: "Place", json: json)
        let encoded = try NFKInferKitLanguageModelRequest.schemaJSON(for: schema)
        XCTAssertEqual(encoded["type"] as? String, "object")
        let properties = try XCTUnwrap(encoded["properties"] as? [String: Any])
        XCTAssertNotNil(properties["city"])
    }

    // MARK: The model

    func testTheModelReportsTheBackendsCapabilities() throws {
        #if compiler(>=6.4)
        guard #available(macOS 27, iOS 27, *) else {
            throw XCTSkip("the provider protocols need macOS 27")
        }
        let model = NFKInferKitLanguageModel(backend: NFKFoundationModelsBackend())
        XCTAssertTrue(model.capabilities.contains(.toolCalling))
        XCTAssertTrue(model.capabilities.contains(.guidedGeneration))
        XCTAssertFalse(model.capabilities.contains(.vision))
        #else
        throw XCTSkip("built with an SDK before macOS 27")
        #endif
    }

    func testTheConfigurationIsTheBackendsIdentity() throws {
        #if compiler(>=6.4)
        guard #available(macOS 27, iOS 27, *) else {
            throw XCTSkip("the provider protocols need macOS 27")
        }
        let backend = NFKEchoBackend()
        let model = NFKInferKitLanguageModel(backend: backend)
        XCTAssertEqual(model.executorConfiguration, NFKInferKitLanguageModel.Configuration(backend: backend))
        XCTAssertNotEqual(model.executorConfiguration,
                          NFKInferKitLanguageModel.Configuration(backend: NFKEchoBackend()))
        #else
        throw XCTSkip("built with an SDK before macOS 27")
        #endif
    }

    func testOnlyNewTextIsAppendedToTheChannel() throws {
        #if compiler(>=6.4)
        guard #available(macOS 27, iOS 27, *) else {
            throw XCTSkip("the provider protocols need macOS 27")
        }
        typealias Executor = NFKInferKitLanguageModelExecutor
        XCTAssertEqual(Executor.appendix(sent: "A gan", text: "A gannet."), "net.")
        XCTAssertEqual(Executor.appendix(sent: "A gannet.", text: "A gannet."), "")
        XCTAssertEqual(Executor.appendix(sent: "A gannet.", text: "A shag."), "",
                       "a rewrite is not an append")
        #else
        throw XCTSkip("built with an SDK before macOS 27")
        #endif
    }
}

/// A backend that answers with its prompt and declares no keys.
private final class NFKEchoBackend: NSObject, NFKInferenceBackend {

    var isReady: Bool { true }

    var backendIdentifier: String { "echo" }

    func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        NFKInferenceResult(outputs: [NFKOutputText: request.prompt ?? ""])
    }
}
