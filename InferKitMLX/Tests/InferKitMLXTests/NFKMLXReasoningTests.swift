//
//  NFKMLXReasoningTests.swift
//  InferKitMLXTests
//
//  Splitting a chain out of an answer, the token counts a run reports, and how a reasoning effort
//  reaches a release's template. All of it is pure text work, so it runs without the MLX runtime.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXReasoningTests: XCTestCase {

    // MARK: The markers

    func testQwenStyleTagsSplitTheChainFromTheAnswer() {
        let (reasoning, answer) = NFKMLXReasoningFormat.thinkTags.split(
            "<think>\nTwo plus two.\n</think>\n\nFour.")
        XCTAssertEqual(reasoning, "Two plus two.")
        XCTAssertEqual(answer, "Four.")
    }

    // A template that pre-closes the block leaves the model nothing to open, so the opening marker
    // is absent from what it generates.
    func testAChainWithNoOpeningMarkerStillSplits() {
        let (reasoning, answer) = NFKMLXReasoningFormat.thinkTags.split("Two plus two.</think>Four.")
        XCTAssertEqual(reasoning, "Two plus two.")
        XCTAssertEqual(answer, "Four.")
    }

    func testTextWithNoChainIsAllAnswer() {
        let (reasoning, answer) = NFKMLXReasoningFormat.thinkTags.split("Four.")
        XCTAssertEqual(reasoning, "")
        XCTAssertEqual(answer, "Four.")
    }

    // A run that hit the token limit inside the chain never reached an answer.
    func testAnUnclosedChainLeavesNoAnswer() {
        let (reasoning, answer) = NFKMLXReasoningFormat.thinkTags.split("<think>\nStill weighing it")
        XCTAssertEqual(reasoning, "Still weighing it")
        XCTAssertEqual(answer, "")
    }

    func testTheHarmonyChannelsDropTheScaffoldingAroundTheAnswer() {
        let generated = "<|channel|>analysis<|message|>Two plus two.<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Four."
        let (reasoning, answer) = NFKMLXReasoningFormat.harmonyChannels.split(generated)
        XCTAssertEqual(reasoning, "Two plus two.")
        XCTAssertEqual(answer, "Four.", "the final channel's markers are not part of the answer")
    }

    func testTheFormatComesFromTheReleasesOwnTemplate() {
        XCTAssertEqual(NFKMLXReasoningFormat.detected(inChatTemplate: "{{ '<think>\\n' }}"),
                       NFKMLXReasoningFormat.thinkTags)
        XCTAssertEqual(NFKMLXReasoningFormat.detected(
            inChatTemplate: "{{ '<|start|>assistant<|channel|>analysis<|message|>' }}"),
                       NFKMLXReasoningFormat.harmonyChannels)
        XCTAssertNil(NFKMLXReasoningFormat.detected(inChatTemplate: "{{ '<start_of_turn>model\\n' }}"),
                     "a release that shows no chain has no markers to read")
    }

    func testARequestNamesAFormatByNameOrByItsMarkers() {
        XCTAssertEqual(NFKMLXReasoningFormat.named("think") ?? nil, NFKMLXReasoningFormat.thinkTags)
        XCTAssertEqual(NFKMLXReasoningFormat.named("HARMONY") ?? nil, NFKMLXReasoningFormat.harmonyChannels)
        XCTAssertNil(NFKMLXReasoningFormat.named("none") ?? nil, "none states that there is no chain")
        // The result is doubly optional: an unknown name is the outer nil, while "none" is the outer
        // some wrapping nil. Comparing the binding keeps the outer meaning and coerces nothing.
        let unknown: NFKMLXReasoningFormat?? = NFKMLXReasoningFormat.named("gibberish")
        XCTAssertTrue(unknown == nil, "an unknown name names nothing")

        let stated = NFKMLXReasoningFormat.named(["<r>", "</r>"]) ?? nil
        XCTAssertEqual(stated?.opening, "<r>")
        XCTAssertEqual(stated?.closing, "</r>")
        XCTAssertEqual(stated?.answerPrefix, "")
    }

    // MARK: The counts

    func testAnUnknownCountIsAbsentRatherThanZero() {
        let known = NFKMLXUsage.outputs(inputTokens: 9, cachedTokens: 0, outputTokens: 4, reasoningTokens: 3)
        XCTAssertEqual(known[NFKUsageInputTokens], 9)
        XCTAssertEqual(known[NFKUsageCachedTokens], 0)
        XCTAssertEqual(known[NFKUsageOutputTokens], 4)
        XCTAssertEqual(known[NFKUsageReasoningTokens], 3)

        let unsplit = NFKMLXUsage.outputs(inputTokens: 9, cachedTokens: 0, outputTokens: 4, reasoningTokens: nil)
        XCTAssertNil(unsplit[NFKUsageReasoningTokens],
                     "without a format the backend cannot say whether the text holds a chain")
        XCTAssertEqual(unsplit[NFKUsageOutputTokens], 4)
    }

    // Each token decodes to one character here, so the answer is the character count itself.
    func testTheChainsShareOfTheReplyIsFoundByBisection() {
        let produced = Array(0 ..< 20)
        let decode: ([Int]) -> String = { String(repeating: "x", count: $0.count) }
        XCTAssertEqual(NFKMLXUsage.tokenCount(inPrefixOf: produced, characters: 0, decode: decode), 0)
        XCTAssertEqual(NFKMLXUsage.tokenCount(inPrefixOf: produced, characters: 1, decode: decode), 1)
        XCTAssertEqual(NFKMLXUsage.tokenCount(inPrefixOf: produced, characters: 7, decode: decode), 7)
        XCTAssertEqual(NFKMLXUsage.tokenCount(inPrefixOf: produced, characters: 20, decode: decode), 20)
        XCTAssertEqual(NFKMLXUsage.tokenCount(inPrefixOf: produced, characters: 99, decode: decode), 20,
                       "a chain that runs past the reply is the whole reply")
        XCTAssertEqual(NFKMLXUsage.tokenCount(inPrefixOf: [], characters: 5, decode: decode), 0)
    }

    // A token is several characters here, so the count is the first token that covers the chain.
    func testAChainEndingInsideATokenCountsThatToken() {
        let produced = Array(0 ..< 6)
        let decode: ([Int]) -> String = { String(repeating: "abcd", count: $0.count) }
        XCTAssertEqual(NFKMLXUsage.tokenCount(inPrefixOf: produced, characters: 1, decode: decode), 1)
        XCTAssertEqual(NFKMLXUsage.tokenCount(inPrefixOf: produced, characters: 4, decode: decode), 1)
        XCTAssertEqual(NFKMLXUsage.tokenCount(inPrefixOf: produced, characters: 5, decode: decode), 2)
    }

    // MARK: The effort

    func testTheEffortBindsBothSpellingsTheReleasesUse() throws {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "why"],
                                          parameters: [NFKParameterReasoningEffort: NFKReasoningEffortDeep])
        let variables = try NFKMLXLanguageBackend.templateVariables(
            for: request, template: .jinja(template: "{{ '' }}"))
        XCTAssertEqual(variables["enable_thinking"] as? Bool, true)
        XCTAssertEqual(variables["reasoning_effort"] as? String, "high")
    }

    // Qwen3's template closes the block when the flag is false, which is the only control it takes.
    func testTheLightestLevelTurnsThinkingOff() throws {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "why"],
                                          parameters: [NFKParameterReasoningEffort: NFKReasoningEffortLight])
        let variables = try NFKMLXLanguageBackend.templateVariables(
            for: request, template: .jinja(template: "{{ '' }}"))
        XCTAssertEqual(variables["enable_thinking"] as? Bool, false)
        XCTAssertEqual(variables["reasoning_effort"] as? String, "low")
    }

    func testALevelTheContractDoesNotNameGoesOutAsWritten() {
        XCTAssertEqual(NFKMLXLanguageBackend.harmonyLevel(for: "minimal"), "minimal")
        XCTAssertEqual(NFKMLXLanguageBackend.harmonyLevel(for: NFKReasoningEffortModerate), "medium")
    }

    // The template is the only thing that takes the level, so without one the request is refused
    // rather than answered as though it had never asked.
    func testAnEffortWithoutATemplateIsRefused() {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "why"],
                                          parameters: [NFKParameterReasoningEffort: NFKReasoningEffortDeep])
        XCTAssertThrowsError(try NFKMLXLanguageBackend.templateVariables(for: request, template: .chatML))
        XCTAssertThrowsError(try NFKMLXLanguageBackend.templateVariables(for: request, template: .none))
    }

    func testARequestWithNoEffortBindsNothing() throws {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "why"])
        let variables = try NFKMLXLanguageBackend.templateVariables(
            for: request, template: .jinja(template: "{{ '' }}"))
        XCTAssertTrue(variables.isEmpty, "the release's own default stands when the request asks for none")
    }

    func testAModelThatDoesNotReasonRefusesTheEffort() {
        let asked = NFKInferenceRequest(inputs: [NFKInputPrompt: "why"],
                                        parameters: [NFKParameterReasoningEffort: NFKReasoningEffortDeep])
        XCTAssertThrowsError(try NFKMLXUsage.refuseReasoningEffort(in: asked, model: "gemma3"))
        XCTAssertNoThrow(try NFKMLXUsage.refuseReasoningEffort(
            in: NFKInferenceRequest(inputs: [NFKInputPrompt: "why"]), model: "gemma3"))
    }

    // MARK: The template's own reading of the bindings

    // The bound flag has to reach the release's own Jinja, which is where the behavior lives.
    func testTheFlagReachesTheReleasesTemplate() throws {
        let template = "{%- if enable_thinking is defined and enable_thinking is false %}"
            + "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
        let closed = try NFKMLXChatTemplateRenderer.render(template, messages: [],
                                                          variables: ["enable_thinking": false])
        XCTAssertEqual(closed, "<think>\n\n</think>\n\n")

        let open = try NFKMLXChatTemplateRenderer.render(template, messages: [],
                                                        variables: ["enable_thinking": true])
        XCTAssertEqual(open, "", "thinking stays on, which is the release's own default")

        let unbound = try NFKMLXChatTemplateRenderer.render(template, messages: [])
        XCTAssertEqual(unbound, "", "an unbound variable leaves the template's default in place")
    }

    func testTheLevelReachesAHarmonyStyleTemplate() throws {
        let template = "{%- if reasoning_effort is not defined %}{%- set reasoning_effort = 'medium' %}"
            + "{%- endif %}{{- 'Reasoning: ' + reasoning_effort }}"
        XCTAssertEqual(try NFKMLXChatTemplateRenderer.render(template, messages: [],
                                                             variables: ["reasoning_effort": "high"]),
                       "Reasoning: high")
        XCTAssertEqual(try NFKMLXChatTemplateRenderer.render(template, messages: []),
                       "Reasoning: medium")
    }
}
