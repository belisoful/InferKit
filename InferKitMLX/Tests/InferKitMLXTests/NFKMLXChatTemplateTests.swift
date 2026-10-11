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
                        tools: [[String: Any]]? = nil, variables: [String: Any] = [:]) throws -> String {
        try NFKMLXChatTemplateRenderer.render(template, messages: messages,
                                              addGenerationPrompt: addGenerationPrompt,
                                              bosToken: bos, tools: tools, variables: variables)
    }

    /// A template over no messages, with `variables` bound.
    private func evaluate(_ template: String, _ variables: [String: Any] = [:]) throws -> String {
        try render(template, [], addGenerationPrompt: false, variables: variables)
    }

    private func ordered(_ json: String) throws -> Any {
        try XCTUnwrap(NFKJinjaOrderedJSON.parse(json), json)
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

    // MARK: Jinja's semantics, as jinja2 3.1 under transformers renders each case

    func testAnAssignmentInALoopStaysInItsIteration() throws {
        XCTAssertEqual(try evaluate("{% set x = 1 %}{% for i in [1,2,3] %}{{ x }}{% set x = x + 10 %}{{ x }},{% endfor %}|{{ x }}"),
                       "111,111,111,|1")
    }

    func testAMacroReadsTheTopLevelAndNotTheCallersLoop() throws {
        let template = "{% set top = 'T' %}{% macro m(a, b=2) %}[{{ a }}{{ b }}{{ top }}{{ g }}{{ caller_var }}]{% set top = 'M' %}{% endmacro %}"
            + "{% for caller_var in ['C'] %}{{ m(1) }}{{ m(1, b=5) }}{% endfor %}{{ top }}"
        XCTAssertEqual(try evaluate(template, ["g": "G"]), "[12TG][15TG]T")
        XCTAssertEqual(try evaluate("{% macro m() %}{{ late }}{% endmacro %}{% set late = 'L' %}{{ m() }}"), "L",
                       "a macro reads what the template set before the call")
        XCTAssertEqual(try evaluate("{% set ns = namespace(v=0) %}{% macro m() %}{% set ns.v = ns.v + 1 %}{% endmacro %}{{ m() }}{{ m() }}{{ ns.v }}"),
                       "2", "a namespace is shared by reference")
    }

    func testARecursiveMacroWithKeywordArguments() throws {
        let template = "{% macro walk(v, depth=0) %}{% if v is mapping %}{% for k, x in v|items %}{{ k }}:{{ walk(x, depth=depth + 1) }};{% endfor %}"
            + "{% else %}{{ v }}@{{ depth }}{% endif %}{% endmacro %}{{ walk(d) }}"
        XCTAssertEqual(try evaluate(template, ["d": try ordered(#"{"z": {"y": 1}, "a": 2}"#)]), "z:y:1@2;;a:2@1;")
    }

    func testItemsKeepTheOrderTheSourceWrote() throws {
        XCTAssertEqual(try evaluate("{% for a, b in d|items %}{{a}}={{b}};{% endfor %}", ["d": try ordered(#"{"z": 1, "a": 2}"#)]),
                       "z=1;a=2;")
        XCTAssertEqual(try evaluate("{% for a, b in d|dictsort %}{{a}}={{b}};{% endfor %}", ["d": try ordered(#"{"b": 1, "A": 2, "a": 3}"#)]),
                       "A=2;a=3;b=1;", "dictsort ignores case and keeps equal keys in order")
        XCTAssertEqual(try evaluate("{{ x | string }}{{ y | string }}{{ z | string }}",
                                    ["x": try ordered(#"{"b": 1, "a": [true, null]}"#), "y": NSNull(), "z": true]),
                       "{'b': 1, 'a': [True, None]}NoneTrue")
    }

    func testAFoundationDictionaryOrdersAToolDeclarationAsItIsWritten() throws {
        let declaration: [String: Any] = ["function": ["parameters": ["required": ["city"], "type": "object",
                                                                       "properties": ["city": ["description": "The city.", "type": "string"]]],
                                                       "description": "Weather.", "name": "get_weather"],
                                          "type": "function"]
        XCTAssertEqual(try evaluate("{{ t | tojson }}", ["t": declaration]),
                       #"{"type": "function", "function": {"name": "get_weather", "description": "Weather.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city."}}, "required": ["city"]}}}"#)
    }

    func testTheTestsFollowPython() throws {
        XCTAssertEqual(try evaluate("{{ 'x' is sequence }}{{ [1] is sequence }}{{ {} is sequence }}{{ 1 is sequence }}{{ true is boolean }}"
                                    + "{{ 1 is boolean }}{{ 1 is integer }}{{ 1.5 is integer }}{{ 1.5 is number }}{{ true is number }}{{ none is mapping }}"),
                       "TrueTrueTrueFalseTrueFalseTrueFalseTrueTrueFalse")
    }

    func testTheLoopNamesItsNeighbors() throws {
        XCTAssertEqual(try evaluate("{% for m in ms %}{{ loop.previtem }}|{{ loop.nextitem }}|{{ loop.revindex }};{% endfor %}", ["ms": [1, 2, 3]]),
                       "|2|3;1|3|2;2||1;")
        XCTAssertEqual(try evaluate("{% for x in [1,2,3] if x > 1 %}{{ x }}{{ loop.index }}{% endfor %}"), "2132",
                       "a filtered loop counts what it iterates")
        XCTAssertEqual(try evaluate("{% for i in [1,2,3,4] %}{% if i == 2 %}{% continue %}{% endif %}{% if i == 4 %}{% break %}{% endif %}{{ i }}{% endfor %}"),
                       "13", "transformers enables the loop controls")
        XCTAssertEqual(try evaluate("{% for x in [] %}a{% else %}empty{% endfor %}"), "empty")
    }

    func testValuesPrintAsPythonPrintsThem() throws {
        XCTAssertEqual(try evaluate("{{ [1,2] }}{{ {'a': 1} }}{{ true }}{{ none }}{{ 1.0 }}{{ 2.50 }}{{ 'a' ~ 1 }}{{ [1,'a'] | join('-') }}"
                                    + "{{ ['a','b'] | map('upper') | list }}{{ d.get('x', 'dflt') }}{{ d.get('k') }}", ["d": ["k": "v"]]),
                       "[1, 2]{'a': 1}TrueNone1.02.5a11-a['A', 'B']dfltv")
        XCTAssertEqual(try evaluate(#"{{ "it's" }}{{ ['it\'s', 'q"uote'] }}"#), #"it's["it's", 'q"uote']"#)
        XCTAssertEqual(try evaluate("{{ 'a' if false }}|{{ 'a' if true }}|{{ {'k': [1, {'n': none}]} }}"), "|a|{'k': [1, {'n': None}]}")
    }

    func testArithmeticFollowsPython() throws {
        XCTAssertEqual(try evaluate("{{ 7 / 2 }}{{ 7 // 2 }}{{ 2 * 'ab' }}{{ -3 % 2 }}{{ 1 < 2.5 }}"), "3.53abab1True")
        XCTAssertEqual(try evaluate("{{ 10 // 3 }}|{{ -7 // 2 }}|{{ 7.5 // 2 }}|{{ 2 ** 10 }}|{{ 1 + 2.5 }}|{{ true + 1 }}|{{ 'x' * 3 }}"),
                       "3|-4|3.0|1024|3.5|2|xxx")
    }

    func testBlockSetRangeAndTheStringMethods() throws {
        XCTAssertEqual(try evaluate("{% set s %}a{{ 1 }}b{% endset %}[{{ s }}]{{ range(3)|list }}{{ range(1,3)|list }}"), "[a1b][0, 1, 2][1, 2]")
        XCTAssertEqual(try evaluate("{{ 'a,b,,c'.split(',') }}{{ ' x y '.split() }}{{ 'a-b-c'.split('-', 1) }}"),
                       "['a', 'b', '', 'c']['x', 'y']['a', 'b-c']")
    }

    func testTheFiltersTakeTheirKeywordArguments() throws {
        let x = try ordered(#"{"b": 1, "a": [true, null, "é"]}"#)
        XCTAssertEqual(try evaluate("{{ x | tojson(indent=2) }}|{{ x | tojson }}|{{ [] | tojson(indent=2) }}", ["x": x]),
                       "{\n  \"b\": 1,\n  \"a\": [\n    true,\n    null,\n    \"é\"\n  ]\n}|{\"b\": 1, \"a\": [true, null, \"é\"]}|[]")
        let y = try ordered(#"{"b": 1, "a": 2}"#)
        XCTAssertEqual(try evaluate("{{ x | tojson(separators=(',', ':')) }}|{{ x | tojson(sort_keys=true) }}", ["x": y]),
                       #"{"b":1,"a":2}|{"a": 2, "b": 1}"#)
        XCTAssertEqual(try evaluate("{{ none | default('d') }}|{{ '' | default('d') }}|{{ '' | default('d', true) }}|{{ undefined_name | default('u') }}"),
                       "None||d|u")
        XCTAssertEqual(try evaluate("{{ 'Hello World' | replace('o', '0', 1) }}|{{ ' a ' | trim }}|{{ 'abc' | capitalize }}|{{ ['b','a'] | sort | join(',') }}"),
                       "Hell0 World|a|Abc|a,b")
        XCTAssertEqual(try evaluate("{{ (msgs | selectattr('role', 'equalto', 'user') | list | length) }}|{{ msgs | map(attribute='role') | join(',') }}",
                                    ["msgs": [["role": "user"], ["role": "tool"], ["role": "user"]]]),
                       "2|user,tool,user")
    }

    func testAnUnknownFilterIsAnError() throws {
        XCTAssertThrowsError(try evaluate("{{ 'x' | no_such_filter }}"), "a silently skipped filter would change the prompt unseen")
    }

    func testAFoundationNumberKeepsItsKind() throws {
        XCTAssertEqual(try evaluate("{{ one }} {{ yes }} {{ half }} {{ one is boolean }}",
                                    ["one": NSNumber(value: 1), "yes": NSNumber(value: true), "half": NSNumber(value: 0.5)]),
                       "1 True 0.5 False", "an integer 1 is not the boolean true")
    }

    func testStrftimeNowFormatsTheGivenDate() throws {
        var components = DateComponents()
        (components.year, components.month, components.day, components.hour, components.minute) = (2026, 1, 15, 9, 30)
        let date = try XCTUnwrap(Calendar(identifier: .gregorian).date(from: components))
        XCTAssertEqual(try NFKMLXChatTemplateRenderer.render("{{ strftime_now('%Y-%m-%d %H:%M %B %d') }}", messages: [], date: date),
                       "2026-01-15 09:30 January 15")
    }

    func testOrderedJSONKeepsKeysEscapesAndKinds() throws {
        let value = try XCTUnwrap(NFKJinjaOrderedJSON.parse(#"{"z": [1, 2.5, true, null], "a": "é\n😀"}"#) as? NFKJinjaOrderedObject)
        XCTAssertEqual(value.pairs.map(\.0), ["z", "a"])
        XCTAssertEqual(value.pairs[1].1 as? String, "é\n😀")
        let list = try XCTUnwrap(value.pairs[0].1 as? [Any])
        XCTAssertEqual((list[0] as? NSNumber)?.intValue, 1)
        XCTAssertEqual((list[1] as? NSNumber)?.doubleValue, 2.5)
        XCTAssertEqual(CFGetTypeID(list[2] as CFTypeRef), CFBooleanGetTypeID())
        XCTAssertTrue(list[3] is NSNull)
        XCTAssertNil(NFKJinjaOrderedJSON.parse(#"{"a": 1"#))
        XCTAssertNil(NFKJinjaOrderedJSON.parse("{} trailing"))
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

    /// A JSON object read by `NFKJinjaOrderedJSON` as a dictionary whose values keep their order.
    private func members(_ value: Any?) -> [String: Any] {
        if let object = value as? NFKJinjaOrderedObject {
            return Dictionary(object.pairs, uniquingKeysWith: { $1 })
        }
        return value as? [String: Any] ?? [:]
    }

    private func renderCase(_ entry: [String: Any], date: Date) throws -> String {
        let template = entry["template"] as? String ?? ""
        let messages = (entry["messages"] as? [Any] ?? []).map(members)
        let tools = (entry["tools"] as? [Any])?.map(members)
        var variables = members(entry["variables"])
        let bos = variables.removeValue(forKey: "bos_token") as? String ?? ""
        let eos = variables.removeValue(forKey: "eos_token") as? String ?? ""
        return try NFKMLXChatTemplateRenderer.render(template, messages: messages, bosToken: bos, eosToken: eos,
                                                     tools: tools, variables: variables, date: date)
    }

    /// Every release template the validation store holds, over a chat, declared tools, and a tool
    /// conversation, renders byte-for-byte as transformers renders it; a case transformers refuses is
    /// refused here too. The record is read with its key order, so a call's arguments render in the
    /// order the model wrote them; a case written in the toolkit's declaration order also renders
    /// from Foundation dictionaries. Regenerate the record with
    /// `Tools/reference-parity/generate_tool_chat_templates.py`.
    func testReleaseTemplatesRenderEveryCaseAsTransformersDoes() throws {
        guard let path = config["IK_CHAT_TEMPLATE_TOOLS_REF"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_CHAT_TEMPLATE_TOOLS_REF to the JSON from Tools/reference-parity/generate_tool_chat_templates.py")
        }
        let text = try String(contentsOfFile: path, encoding: .utf8)
        let ordered = try XCTUnwrap(NFKJinjaOrderedJSON.parse(text) as? NFKJinjaOrderedObject)
        let unordered = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        var components = DateComponents()
        (components.year, components.month, components.day, components.hour, components.minute) = (2026, 1, 15, 9, 30)
        let date = try XCTUnwrap(Calendar(identifier: .gregorian).date(from: components))
        var matched = 0
        for (name, value) in ordered.pairs {
            let entry = members(value)
            if let refusal = entry["error"] as? String {
                XCTAssertThrowsError(try renderCase(entry, date: date), "\(name): transformers refused it with \(refusal)")
                continue
            }
            let expected = try XCTUnwrap(entry["expected"] as? String, name)
            let rendered: String
            do {
                rendered = try renderCase(entry, date: date)
            } catch {
                XCTFail("\(name): \(error)")
                continue
            }
            XCTAssertEqual(rendered, expected, name)
            if rendered == expected { matched += 1 }
            if (entry["foundation_order"] as? NSNumber)?.boolValue == true,
               let foundation = unordered[name] as? [String: Any] {
                XCTAssertEqual(try? renderCase(foundation, date: date), expected, "\(name) from Foundation dictionaries")
            }
        }
        print("VALIDATION chat templates: \(matched) of \(ordered.pairs.count) cases byte-exact")
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
