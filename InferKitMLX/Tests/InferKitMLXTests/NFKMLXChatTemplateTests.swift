//
//  NFKMLXChatTemplateTests.swift
//  InferKitMLXTests
//
//  The chat-template renderer is pure Foundation, so these run under `swift test` with no MLX runtime.
//  The unit tests exercise the interpreter's constructs directly; the parity test compares the
//  rendered output against transformers' own `apply_chat_template` over the shipped chat cases.
//

import XCTest
@testable import InferKitMLX

final class NFKMLXChatTemplateTests: XCTestCase {

    private func render(_ template: String, _ messages: [[String: Any]],
                        addGenerationPrompt: Bool = true, bos: String = "",
                        tools: [[String: Any]]? = nil) throws -> String {
        try NFKMLXChatTemplateRenderer.render(template, messages: messages,
                                              addGenerationPrompt: addGenerationPrompt,
                                              bosToken: bos, tools: tools)
    }

    // MARK: Interpreter constructs

    func testAForLoopOverMessagesEmitsEachContent() throws {
        let template = "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"
        let out = try render(template, [["role": "user", "content": "hi"],
                                        ["role": "assistant", "content": "yo"]],
                             addGenerationPrompt: false)
        XCTAssertEqual(out, "user:hi\nassistant:yo\n")
    }

    func testConditionalBranchesSelectByRole() throws {
        let template = "{% for m in messages %}{% if m.role == 'user' %}U{% elif m.role == 'system' %}S{% else %}A{% endif %}{% endfor %}"
        let out = try render(template, [["role": "system", "content": ""],
                                        ["role": "user", "content": ""],
                                        ["role": "assistant", "content": ""]],
                             addGenerationPrompt: false)
        XCTAssertEqual(out, "SUA")
    }

    func testNamespaceMutationPersistsAcrossLoopIterations() throws {
        let template = "{% set ns = namespace(count=0) %}{% for m in messages %}{% set ns.count = ns.count + 1 %}{% endfor %}{{ ns.count }}"
        let out = try render(template, [["role": "user", "content": "a"],
                                        ["role": "user", "content": "b"],
                                        ["role": "user", "content": "c"]],
                             addGenerationPrompt: false)
        XCTAssertEqual(out, "3")
    }

    func testReversedSliceAndLoopIndex() throws {
        let template = "{% for m in messages[::-1] %}{{ loop.index0 }}:{{ m.content }} {% endfor %}"
        let out = try render(template, [["role": "user", "content": "first"],
                                        ["role": "user", "content": "second"]],
                             addGenerationPrompt: false)
        XCTAssertEqual(out, "0:second 1:first ")
    }

    func testFilterBindsTighterThanConcatenation() throws {
        // `a + b | trim + c` groups as `a + (b | trim) + c`, so only the middle is trimmed.
        let template = "{{ '[' + messages[0].content | trim + ']' }}"
        let out = try render(template, [["role": "user", "content": "  x  "]], addGenerationPrompt: false)
        XCTAssertEqual(out, "[x]")
    }

    func testStringMethodsAndMembership() throws {
        let template = "{% set c = messages[0].content %}{{ c.split('</think>')[-1].lstrip('\\n') }}|{{ '</think>' in c }}"
        let out = try render(template, [["role": "assistant", "content": "reason</think>\nanswer"]],
                             addGenerationPrompt: false)
        XCTAssertEqual(out, "answer|True")
    }

    func testDefinedTestOnAnAbsentVariable() throws {
        let template = "{% if enable_thinking is defined %}D{% else %}U{% endif %}"
        XCTAssertEqual(try render(template, [], addGenerationPrompt: false), "U")
    }

    // MARK: Reference parity

    private var config: [String: String] {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }

    /// Every case in the reference must render byte-for-byte as transformers' `apply_chat_template`.
    /// Regenerate the reference with `Tools/reference-parity/generate_chat_templates.py`.
    func testRenderedTemplatesMatchTheReference() throws {
        guard let path = config["IK_CHAT_TEMPLATE_REF"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_CHAT_TEMPLATE_REF to the JSON from Tools/reference-parity/generate_chat_templates.py")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for name in root.keys.sorted() {
            let entry = try XCTUnwrap(root[name] as? [String: Any], "case \(name)")
            let template = try XCTUnwrap(entry["template"] as? String)
            let messages = try XCTUnwrap(entry["messages"] as? [[String: Any]])
            let addGenerationPrompt = (entry["add_generation_prompt"] as? Bool) ?? true
            let bos = (entry["bos_token"] as? String) ?? ""
            let tools = (entry["tools"] as? [Any])?.compactMap { $0 as? [String: Any] }
            let expected = try XCTUnwrap(entry["expected"] as? String)
            let got = try render(template, messages, addGenerationPrompt: addGenerationPrompt,
                                 bos: bos, tools: tools)
            XCTAssertEqual(got, expected, "case \(name)")
        }
    }
}
