//
//  FoundationModelsExamples.swift
//  InferKitFoundationModelsExamples
//
//  Compiled and run by CI so the Foundation Models snippets in Docs/examples.md cannot silently
//  drift. Each method mirrors a section there — change an example here and update the matching
//  snippet, and vice versa. These construct the backend and set up tools / schema / messages without
//  invoking the model, so they run without Apple Intelligence; live generation is covered by
//  NFKFoundationModelsBackendTests (which skips where the model is unavailable).
//

import XCTest
import FoundationModels
import InferKit
@testable import InferKitFoundationModels

final class FoundationModelsExamples: XCTestCase {

    // Docs/examples.md: Text → text (Apple on-device)
    func testExampleBackendConstructs() {
        let backend = NFKFoundationModelsBackend()
        XCTAssertEqual(backend.backendIdentifier, "foundation-models")
    }

    // Docs/examples.md: Dynamic backend discovery — linking this package activates on-device LLM in the
    // core's text-generation capability, with no build dependency on it.
    func testExampleDynamicTextGenerationCapability() throws {
        XCTAssertTrue(NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityTextGeneration),
                      "NFKFoundationModelsProvider ships here, so the capability resolves")
        let backend = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTextGeneration)
        XCTAssertEqual(backend.backendIdentifier, "foundation-models")
    }

    // Docs/examples.md: Multi-turn conversation
    func testExampleMultiTurnPlan() {
        let request = NFKInferenceRequest(
            inputs: [NFKInputMessages: [["role": "system", "content": "Be terse."],
                                        ["role": "user", "content": "Name one color."]]],
            parameters: nil)
        let plan = NFKFoundationModelsBackend.plan(for: request)
        XCTAssertEqual(plan.instructions, "Be terse.")
        XCTAssertEqual(plan.prompt, "Name one color.")
    }

    // Docs/examples.md: Tool calling
    func testExampleRegisterATool() {
        let backend = NFKFoundationModelsBackend()
        backend.tools = [
            NFKFoundationTool(
                name: "get_temperature",
                description: "Get the current temperature for a city.",
                parameters: [NFKFoundationToolParameter(name: "city", description: "the city", type: .string, required: true)],
                handler: { arguments in "It is 21°C in \(arguments["city"] as? String ?? "")" })
        ]
        XCTAssertEqual(backend.tools.count, 1)
    }

    // Docs/examples.md: Structured output
    func testExampleSetResponseSchema() {
        let backend = NFKFoundationModelsBackend()
        backend.responseSchema = [
            NFKFoundationToolParameter(name: "name", description: "the character's full name", type: .string, required: true),
            NFKFoundationToolParameter(name: "age", description: "the character's age in years", type: .integer, required: true),
        ]
        XCTAssertEqual(backend.responseSchema?.count, 2)
    }
}
