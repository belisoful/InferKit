//
//  NFKMLXToolCallingTests.swift
//  InferKitMLXTests
//
//  Tool calling on device without weights: each family's call syntax read out of a reply, the format
//  detected from the template that writes it, and a tool conversation prepared for and rendered
//  through a chat template. Pure Foundation, so these run under `swift test` with no MLX runtime.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXToolCallingTests: XCTestCase {

    private let weather: [[String: Any]] = [[
        "type": "function",
        "function": ["name": "get_weather", "description": "Current weather.",
                     "parameters": ["type": "object",
                                    "properties": ["city": ["type": "string"], "days": ["type": "integer"],
                                                   "metric": ["type": "boolean"], "where": ["type": "object"]],
                                    "required": ["city"]]],
    ]]

    private func arguments(_ call: NFKMLXToolCallFormat.Call) -> [String: Any] {
        NFKMLXToolCalling.foundation(call.arguments) as? [String: Any] ?? [:]
    }

    // MARK: Detection

    func testTheFormatIsReadFromTheTemplateThatWritesIt() {
        XCTAssertEqual(NFKMLXToolCallFormat.detected(inChatTemplate: "{{- '<tool_call>\\n{\"name\": \"' }}"), .hermes)
        XCTAssertEqual(NFKMLXToolCallFormat.detected(inChatTemplate: "<tool_call>\\n<function=x>\\n<parameter=y>"), .qwenXML,
                       "Qwen3.5 wraps its XML in <tool_call>, so the XML is tested first")
        XCTAssertEqual(NFKMLXToolCallFormat.detected(inChatTemplate: "{{- '<|tool_call>call:' + function['name'] }}"), .gemma)
        XCTAssertEqual(NFKMLXToolCallFormat.detected(inChatTemplate: "{{- \"<|start|>assistant to=\" }}{{- \"functions.\" + tool_call.name + \"<|channel|>commentary \" }}"),
                       .harmony, "gpt-oss writes the address in two pieces")
        XCTAssertEqual(NFKMLXToolCallFormat.detected(inChatTemplate: "<｜tool▁call▁begin｜>"), .deepSeek)
        XCTAssertEqual(NFKMLXToolCallFormat.detected(inChatTemplate: "[TOOL_CALLS]"), .mistral)
        XCTAssertEqual(NFKMLXToolCallFormat.detected(inChatTemplate: "<|tool_call|>"), .granite)
        XCTAssertEqual(NFKMLXToolCallFormat.detected(inChatTemplate: "{{- '<TOOLCALL>[' -}}"), .nemotron)
        XCTAssertNil(NFKMLXToolCallFormat.detected(inChatTemplate: "{% for m in messages %}{{ m.content }}{% endfor %}"))
        XCTAssertEqual(NFKMLXToolCallFormat.named("qwen-xml"), .some(.qwenXML))
        XCTAssertEqual(NFKMLXToolCallFormat.named("none"), .some(nil))
        XCTAssertTrue(NFKMLXToolCallFormat.named("xml?") == nil, "a name the toolkit does not know")
    }

    /// Every tool-calling template the validation store holds is read as the syntax its family
    /// writes, from the record `Tools/reference-parity/generate_tool_chat_templates.py` writes.
    func testEveryReleaseTemplateIsReadAsItsFamilysSyntax() throws {
        var config = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url), let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { config[$0.key] = $0.value }
        }
        guard let path = config["IK_CHAT_TEMPLATE_TOOLS_REF"], let data = FileManager.default.contents(atPath: path),
              let record = try JSONSerialization.jsonObject(with: data) as? [String: [String: Any]] else {
            throw XCTSkip("set IK_CHAT_TEMPLATE_TOOLS_REF to the JSON from Tools/reference-parity/generate_tool_chat_templates.py")
        }
        let families: [(String, NFKMLXToolCallFormat)] = [("QWEN3_5", .qwenXML), ("QWEN4", .qwenXML), ("GPT_OSS", .harmony),
                                                          ("GEMMA4", .gemma), ("NEMOTRON", .nemotron), ("GRANITE_SPEECH", .granite),
                                                          ("QWEN3", .hermes), ("GRANITE_TINY", .hermes), ("SA2VA", .hermes)]
        var checked = 0
        for (name, entry) in record where name.hasSuffix("/tool_conversation") {
            guard let template = entry["template"] as? String,
                  let family = families.first(where: { name.contains($0.0) }) else { continue }
            XCTAssertEqual(NFKMLXToolCallFormat.detected(inChatTemplate: template), family.1, name)
            checked += 1
        }
        XCTAssertGreaterThan(checked, 5, "the record holds the tool-calling releases")
    }

    // MARK: Reading each syntax

    func testHermesCallsAndTheTextAroundThem() {
        let reply = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 2}}\n</tool_call>\n"
            + "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \"{\\\"city\\\": \\\"Rome\\\"}\"}\n</tool_call>"
        let (content, calls) = NFKMLXToolCallFormat.hermes.read(reply, tools: weather)
        XCTAssertEqual(content, "Let me check.")
        XCTAssertEqual(calls.map(\.name), ["get_weather", "get_weather"])
        XCTAssertEqual(calls[0].arguments.pairs.map(\.0), ["city", "days"], "the order the model wrote")
        XCTAssertEqual(arguments(calls[1])["city"] as? String, "Rome", "arguments written as a JSON string are read too")
    }

    func testAHermesCallCutOffAtItsStopTokenIsStillRead() {
        let (content, calls) = NFKMLXToolCallFormat.hermes.read("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}", tools: weather)
        XCTAssertEqual(content, "")
        XCTAssertEqual(arguments(calls[0])["city"] as? String, "Oslo")
    }

    func testAMalformedCallStaysInTheText() {
        let reply = "<tool_call>{not json}</tool_call> done"
        let (content, calls) = NFKMLXToolCallFormat.hermes.read(reply, tools: weather)
        XCTAssertTrue(calls.isEmpty)
        XCTAssertEqual(content, reply)
    }

    func testQwenXMLValuesAreReadByTheirSchemaType() {
        let reply = "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
            + "<parameter=metric>\nfalse\n</parameter>\n<parameter=where>\n{\"lat\": 40.7}\n</parameter>\n<parameter=note>\n42\n</parameter>\n</function>\n</tool_call>"
        let (content, calls) = NFKMLXToolCallFormat.qwenXML.read(reply, tools: weather)
        XCTAssertEqual(content, "")
        let read = arguments(try! XCTUnwrap(calls.first))
        XCTAssertEqual(read["city"] as? String, "New York")
        XCTAssertEqual((read["days"] as? NSNumber)?.intValue, 3)
        XCTAssertEqual(read["metric"] as? Bool, false)
        XCTAssertEqual((read["where"] as? [String: Any])?["lat"] as? Double, 40.7)
        XCTAssertEqual(read["note"] as? String, "42", "a parameter the schema does not name stays the text")
        XCTAssertEqual(calls[0].arguments.pairs.map(\.0), ["city", "days", "metric", "where", "note"])
    }

    func testQwenXMLWithoutItsWrapperIsStillACall() {
        let (content, calls) = NFKMLXToolCallFormat.qwenXML.read("Sure.\n<function=get_weather>\n<parameter=city>\nLima\n</parameter>\n</function>", tools: weather)
        XCTAssertEqual(content, "Sure.")
        XCTAssertEqual(arguments(calls[0])["city"] as? String, "Lima")
    }

    func testGemmaArgumentsWithMarkedStringsAndBareKeys() {
        let reply = "<|tool_call>call:get_weather{city:<|\"|>São Paulo, BR<|\"|>,days:2,metric:true,where:{lat:-23.5,tags:[<|\"|>a<|\"|>,null]}}<tool_call|>"
        let (content, calls) = NFKMLXToolCallFormat.gemma.read(reply, tools: weather)
        XCTAssertEqual(content, "")
        let read = arguments(try! XCTUnwrap(calls.first))
        XCTAssertEqual(calls[0].name, "get_weather")
        XCTAssertEqual(read["city"] as? String, "São Paulo, BR", "a comma inside a marked string is the string's")
        XCTAssertEqual((read["days"] as? NSNumber)?.intValue, 2)
        XCTAssertEqual(read["metric"] as? Bool, true)
        let place = try! XCTUnwrap(read["where"] as? [String: Any])
        XCTAssertEqual(place["lat"] as? Double, -23.5)
        XCTAssertEqual((place["tags"] as? [Any])?.count, 2)
    }

    func testHarmonyCommentaryAddressedToAFunction() {
        let reply = "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Tokyo\"}<|call|>"
        let (content, calls) = NFKMLXToolCallFormat.harmony.read(reply, tools: weather)
        XCTAssertEqual(content, "")
        XCTAssertEqual(calls.first?.name, "get_weather")
        XCTAssertEqual(arguments(calls[0])["city"] as? String, "Tokyo")
        let final = NFKMLXToolCallFormat.harmony.read("<|start|>assistant<|channel|>final<|message|>It is sunny.", tools: weather)
        XCTAssertEqual(final.content, "It is sunny.")
        XCTAssertTrue(final.calls.isEmpty)
    }

    func testDeepSeekBlocksInBothVersions() {
        let current = "Checking.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"Seoul\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        let (content, calls) = NFKMLXToolCallFormat.deepSeek.read(current, tools: weather)
        XCTAssertEqual(content, "Checking.")
        XCTAssertEqual(arguments(calls[0])["city"] as? String, "Seoul")
        let v3 = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Hanoi\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        let older = NFKMLXToolCallFormat.deepSeek.read(v3, tools: weather)
        XCTAssertEqual(older.calls.first?.name, "get_weather")
        XCTAssertEqual(arguments(older.calls[0])["city"] as? String, "Hanoi")
    }

    func testMistralListAndArgsForms() {
        let list = NFKMLXToolCallFormat.mistral.read("[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Nice\"}}]", tools: weather)
        XCTAssertEqual(arguments(list.calls[0])["city"] as? String, "Nice")
        let args = NFKMLXToolCallFormat.mistral.read("Ok[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Lyon\"}[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Metz\"}", tools: weather)
        XCTAssertEqual(args.content, "Ok")
        XCTAssertEqual(args.calls.map { self.arguments($0)["city"] as? String }, ["Lyon", "Metz"])
    }

    func testGraniteNemotronAndLlamaJSON() {
        let granite = NFKMLXToolCallFormat.granite.read("<|tool_call|>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}]", tools: weather)
        XCTAssertEqual(arguments(granite.calls[0])["city"] as? String, "Oslo")
        let nemotron = NFKMLXToolCallFormat.nemotron.read("<think></think>Sure.<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Bern\"}}]</TOOLCALL>", tools: weather)
        XCTAssertEqual(nemotron.content, "<think></think>Sure.")
        XCTAssertEqual(arguments(nemotron.calls[0])["city"] as? String, "Bern")
        let llama = NFKMLXToolCallFormat.llama.read("<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Cork\"}}", tools: weather)
        XCTAssertEqual(llama.content, "")
        XCTAssertEqual(arguments(llama.calls[0])["city"] as? String, "Cork")
        XCTAssertTrue(NFKMLXToolCallFormat.llama.read("Plain words.", tools: weather).calls.isEmpty)
    }

    // MARK: The conversation and the outputs

    func testTheConversationIsPreparedInTheShapeTemplatesRead() throws {
        let calling = NFKToolMessages.assistantMessage(withText: nil,
                                                       toolCalls: [["id": "c1", "name": "get_weather", "argumentsJSON": "{\"days\": 2, \"city\": \"Paris\"}"],
                                                                   ["id": "c2", "name": "get_weather", "argumentsJSON": "{\"city\": \"Rome\"}"]],
                                                       record: ["reasoning": "Two cities."], format: NFKMLXToolCalling.recordFormat, source: "qwen")
        let failed = NFKToolMessages.message(forToolCallIdentifier: "c1", name: "get_weather",
                                             result: NFKToolResult(text: "no data", outputs: nil, imagePNGs: nil, isError: true))
        let answered: [String: Any] = ["role": "tool", "tool_call_id": "c2", "content": "sunny"]
        let conversation: [[String: Any]] = [["role": "user", "content": "Weather?"], calling, failed, answered]

        let prepared = NFKMLXToolCalling.templateMessages(conversation, format: .hermes, source: "qwen")
        XCTAssertEqual(prepared.count, 4)
        XCTAssertNil(prepared[1][NFKMessageMetadataKey])
        XCTAssertEqual(prepared[1]["reasoning_content"] as? String, "Two cities.", "this backend's own reasoning comes back")
        XCTAssertEqual(prepared[1]["thinking"] as? String, "Two cities.", "gpt-oss reads it as thinking")
        let calls = try XCTUnwrap(prepared[1]["tool_calls"] as? [[String: Any]])
        let function = try XCTUnwrap(calls[0]["function"] as? [String: Any])
        XCTAssertEqual((function["arguments"] as? NFKJinjaOrderedObject)?.pairs.map(\.0), ["days", "city"])
        XCTAssertEqual(prepared[2]["content"] as? String, "Error: no data")
        XCTAssertEqual(prepared[2]["name"] as? String, "get_weather")
        XCTAssertEqual(prepared[3]["name"] as? String, "get_weather", "an answer's name is found from its call")

        let other = NFKMLXToolCalling.templateMessages(conversation, format: .hermes, source: "another-model")
        XCTAssertNil(other[1]["reasoning_content"], "another backend's record is not this one's")

        let harmony = NFKMLXToolCalling.templateMessages(conversation, format: .harmony, source: "qwen")
        XCTAssertEqual(harmony.map { $0["role"] as? String ?? "" }, ["user", "assistant", "tool", "assistant", "tool"],
                       "harmony renders one call per assistant message, each followed by its answer")
        XCTAssertEqual((harmony[1]["tool_calls"] as? [[String: Any]])?.first?["id"] as? String, "c1")
        XCTAssertEqual(harmony[2]["tool_call_id"] as? String, "c1")
        XCTAssertEqual((harmony[3]["tool_calls"] as? [[String: Any]])?.first?["id"] as? String, "c2")
        XCTAssertEqual(harmony[4]["tool_call_id"] as? String, "c2")
    }

    func testAToolRequestStopsWhereTheGenerationConfigSays() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: ["eos_token_id": [200002, 199999, 200012]])
            .write(to: directory.appendingPathComponent("generation_config.json"))
        let tooled = NFKInferenceRequest(inputs: [NFKInputPrompt: "x"],
                                         parameters: [NFKParameterTools: [["name": "get_weather", "parameters": ["type": "object"]]]])
        XCTAssertEqual(NFKMLXToolCalling.toolStops(for: tooled, chosen: [], endOfSequence: 200002, releaseDirectoryURL: directory),
                       [200002, 199999, 200012], "gpt-oss ends a call's turn at <|call|>, which only its generation config names")
        XCTAssertNil(NFKMLXToolCalling.toolStops(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "x"]), chosen: [],
                                                 endOfSequence: 200002, releaseDirectoryURL: directory), "a plain chat keeps its stop")
        XCTAssertNil(NFKMLXToolCalling.toolStops(for: tooled, chosen: [7], endOfSequence: 200002, releaseDirectoryURL: directory),
                     "a caller's own stops stand")
    }

    func testTheOutputsCarryTheCallsAndTheTurnToAppend() throws {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Weather?"],
                                          parameters: [NFKParameterTools: [["name": "get_weather", "parameters": ["type": "object"]]]])
        let outputs = NFKMLXToolCalling.outputs(answer: "One moment.<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 1}}</tool_call>",
                                                reasoning: "Need the tool.", request: request, format: .hermes, source: "qwen")
        XCTAssertEqual(outputs[NFKOutputText] as? String, "One moment.")
        let calls = try XCTUnwrap(outputs[NFKOutputToolCalls] as? [[String: Any]])
        XCTAssertTrue((calls[0]["id"] as? String)?.hasPrefix("call_") ?? false)
        XCTAssertEqual(calls[0]["argumentsJSON"] as? String, "{\"city\": \"Paris\", \"days\": 1}")
        XCTAssertEqual((calls[0]["arguments"] as? [String: Any])?["city"] as? String, "Paris")
        let turn = try XCTUnwrap(outputs[NFKOutputAssistantMessage] as? [String: Any])
        XCTAssertEqual(NFKToolMessages.toolCalls(ofAssistantMessage: turn).first?["id"] as? String, calls[0]["id"] as? String)
        XCTAssertEqual((NFKToolMessages.record(ofMessage: turn, format: NFKMLXToolCalling.recordFormat, source: "qwen") as? [String: Any])?["reasoning"] as? String,
                       "Need the tool.")

        let untooled = NFKMLXToolCalling.outputs(answer: "<tool_call>{\"name\": \"x\"}</tool_call>", reasoning: "",
                                                 request: NFKInferenceRequest(inputs: [NFKInputPrompt: "x"]), format: .hermes, source: "qwen")
        XCTAssertNil(untooled[NFKOutputToolCalls], "calls are read only where tools were offered")
    }

    func testAToolConversationRendersThroughTheTemplateWithItsDeclarations() throws {
        let template = "{% if tools %}<tools>{% for t in tools %}{{ t | tojson }}{% endfor %}</tools>{% endif %}"
            + "{% for m in messages %}<|{{ m.role }}|>{{ m.content if m.content is string else '' }}"
            + "{% for c in m.tool_calls or [] %}<call {{ c.function.name }}>{% for k, v in c.function.arguments|items %}{{ k }}={{ v }};{% endfor %}{% endfor %}"
            + "{% if m.role == 'tool' %}[{{ m.name }}]{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
        let calling = NFKToolMessages.assistantMessage(withText: nil, toolCalls: [["id": "c1", "name": "get_weather", "argumentsJSON": "{\"days\": 2, \"city\": \"Oslo\"}"]],
                                                       record: nil, format: nil, source: nil)
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "user", "content": "Oslo?"], calling,
                                                                      ["role": "tool", "tool_call_id": "c1", "content": "cold"]]],
                                          parameters: [NFKParameterTools: [["name": "get_weather", "description": "Weather.",
                                                                            "parameters": ["type": "object"]]]])
        let prompt = try XCTUnwrap(NFKMLXLanguageBackend.prompt(from: request, template: .jinja(template: template),
                                                                source: "qwen", toolCallFormat: .hermes))
        XCTAssertEqual(prompt, #"<tools>{"type": "function", "function": {"name": "get_weather", "description": "Weather.", "parameters": {"type": "object"}}}</tools>"#
                       + "<|user|>Oslo?<|assistant|><call get_weather>days=2;city=Oslo;<|tool|>cold[get_weather]<|assistant|>")
    }
}
