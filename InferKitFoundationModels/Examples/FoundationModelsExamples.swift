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

    // Docs/examples.md: Choosing the model — on-device by default; Private Cloud Compute on macOS 27 /
    // iOS 27, with the quota read before switching to it.
    func testExampleChooseTheModel() {
        let backend = NFKFoundationModelsBackend()
        backend.useCase = .contentTagging                    // the on-device tagging specialization
        backend.guardrails = .permissiveContentTransformations
        if #available(macOS 27, iOS 27, *) {
            if let quota = backend.privateCloudComputeQuota, !quota.isLimitReached {
                backend.model = .privateCloudCompute         // Apple's larger model; leaves the device
            }
        }
        XCTAssertEqual(backend.useCase, .contentTagging)
        if #available(macOS 27, iOS 27, *) {} else {
            XCTAssertEqual(backend.model, .onDevice)
        }
    }

    // Docs/examples.md: Sampling — the core keys choose Apple's sampling mode.
    func testExampleSamplingKeys() throws {
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "Name one color."],
            parameters: [NFKParameterTopK: 40, NFKParameterSeed: 7, NFKParameterMaxTokens: 16])
        let options = try NFKFoundationModelsBackend.generationOptions(for: request)
        XCTAssertEqual(NFKFoundationModelsBackend.samplingMode(of: options), .random(top: 40, seed: 7))
    }

    // Docs/examples.md: Tool calling — a registered tool carries its handler; the declaration is the
    // same {name, description, parameters} dictionary a remote backend takes under NFKParameterTools.
    func testExampleRegisterATool() {
        let backend = NFKFoundationModelsBackend()
        backend.tools = [
            NFKFoundationTool(
                name: "get_temperature",
                description: "Get the current temperature for a city.",
                parameters: ["type": "object",
                             "properties": ["city": ["type": "string", "description": "the city"]],
                             "required": ["city"]],
                handler: { arguments in "It is 21°C in \(arguments["city"] as? String ?? "")" })
        ]
        XCTAssertEqual(backend.tools.count, 1)
        XCTAssertEqual(backend.tools[0].declaration["name"] as? String, "get_temperature")
    }

    // Docs/examples.md: Structured output — the core's JSON Schema key, as for a remote or MLX backend.
    func testExampleStructuredOutputSchema() throws {
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "Invent a fictional character."],
            parameters: [NFKParameterJSONSchema: [
                "type": "object",
                "properties": ["name": ["type": "string", "description": "the character's full name"],
                               "age": ["type": "integer", "description": "the character's age in years"]],
                "required": ["name", "age"],
            ]])
        let format = try NFKFoundationModelsBackend.outputFormat(for: request)
        guard case .schema = format else { return XCTFail("expected the schema path") }
    }
}
