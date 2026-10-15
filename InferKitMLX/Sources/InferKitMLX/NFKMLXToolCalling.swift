//
//  NFKMLXToolCalling.swift
//  InferKitMLX
//
//  Tool calling on device: the declarations a request offers rendered into the release's chat
//  template, a tool conversation in the shape the templates read, and the calls read back out of the
//  reply in the syntax the release writes them in.
//

import Foundation
import InferKit

/// The syntax a release writes a tool call in, so the backend reads the calls out of its reply.
///
/// @discussion The families disagree, so this is a value the backend takes from the release's own
/// chat template through ``detected(inChatTemplate:)``: the template that renders a past call into
/// the prompt shows the syntax the model was trained to write. A caller that renders its own prompt
/// names the format under ``NFKMLXGenerationParameterKey/toolCallFormat``.
///
/// - ``hermes``: `<tool_call>{"name": …, "arguments": {…}}</tool_call>`, Qwen2.5, Qwen3, Qwen3-VL,
///   and Granite 4.
/// - ``qwenXML``: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`,
///   Qwen3.5, Qwen4, and Qwen3-Coder. A value is read by its parameter's schema type.
/// - ``gemma``: `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`, Gemma 4.
/// - ``harmony``: a `commentary` message addressed `to=functions.name`, gpt-oss.
/// - ``deepSeek``: the `<｜tool▁call▁begin｜>` blocks of DeepSeek V3 and later.
/// - ``mistral``: `[TOOL_CALLS]` followed by a JSON list, or by `name[ARGS]{…}`.
/// - ``granite``: `<|tool_call|>` followed by a JSON list, Granite 3.
/// - ``llama``: a bare `{"name": …, "parameters": {…}}` reply, Llama 3.1 and later.
/// - ``nemotron``: `<TOOLCALL>[{"name": …, "arguments": {…}}]</TOOLCALL>`, Nemotron-H.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXToolCallFormat)
public final class NFKMLXToolCallFormat: NSObject, @unchecked Sendable {

    /// The format's name, which ``NFKMLXGenerationParameterKey/toolCallFormat`` takes.
    @objc public let name: String

    private init(_ name: String) {
        self.name = name
        super.init()
    }

    @objc public static let hermes = NFKMLXToolCallFormat("hermes")
    @objc public static let qwenXML = NFKMLXToolCallFormat("qwen-xml")
    @objc public static let gemma = NFKMLXToolCallFormat("gemma")
    @objc public static let harmony = NFKMLXToolCallFormat("harmony")
    @objc public static let deepSeek = NFKMLXToolCallFormat("deepseek")
    @objc public static let mistral = NFKMLXToolCallFormat("mistral")
    @objc public static let granite = NFKMLXToolCallFormat("granite")
    @objc public static let llama = NFKMLXToolCallFormat("llama")
    @objc public static let nemotron = NFKMLXToolCallFormat("nemotron")

    static let all = [hermes, qwenXML, gemma, harmony, deepSeek, mistral, granite, llama, nemotron]

    /// The format a release's chat template writes past calls in, or nil where it renders none.
    ///
    /// @discussion The more specific syntaxes are tested first: Qwen3.5's template writes
    /// `<tool_call>` around its XML, which alone would read as Hermes.
    @objc public static func detected(inChatTemplate template: String) -> NFKMLXToolCallFormat? {
        if template.contains("<function=") && template.contains("<parameter=") { return qwenXML }
        if template.contains("<|tool_call>") { return gemma }
        // gpt-oss writes the address in two pieces, `to=` and then `functions.` with the name.
        if template.contains("<|channel|>") && template.contains("functions.") { return harmony }
        if template.contains("tool▁call▁begin") { return deepSeek }
        if template.contains("[TOOL_CALLS]") { return mistral }
        if template.contains("<|tool_call|>") { return granite }
        if template.contains("<TOOLCALL>") { return nemotron }
        if template.contains("<tool_call>") { return hermes }
        if template.contains("<|python_tag|>") { return llama }
        return nil
    }

    /// The format a request names, `.some(nil)` for `"none"`, or nil where the value names nothing.
    static func named(_ value: Any) -> NFKMLXToolCallFormat?? {
        guard let name = (value as? String)?.lowercased() else { return nil }
        if name == "none" { return .some(nil) }
        return all.first { $0.name == name }.map { .some($0) }
    }

    public override func isEqual(_ object: Any?) -> Bool {
        (object as? NFKMLXToolCallFormat)?.name == name
    }

    public override var hash: Int { name.hash }

    // MARK: Reading calls

    /// One call read from a reply: the tool's name and its arguments, an `NFKJinjaOrderedObject` in
    /// the order the model wrote them.
    struct Call {
        let name: String
        let arguments: NFKJinjaOrderedObject
    }

    /// The reply's text with the calls taken out, and the calls. `tools` are the declarations the
    /// request offered, whose schemas type the values of a syntax that writes them as text.
    func read(_ reply: String, tools: [[String: Any]]) -> (content: String, calls: [Call]) {
        switch self {
        case .qwenXML: return readQwenXML(reply, tools: tools)
        case .gemma: return readGemma(reply)
        case .harmony: return readHarmony(reply)
        case .deepSeek: return readDeepSeek(reply)
        case .mistral: return readMistral(reply)
        case .granite: return readJSONList(reply, after: "<|tool_call|>")
        case .llama: return readLlama(reply)
        case .nemotron:
            return readSpans(reply, open: "<TOOLCALL>", close: "</TOOLCALL>") { body in
                (NFKJinjaOrderedJSON.parse(body.trimmingCharacters(in: .whitespacesAndNewlines)) as? [Any])?
                    .compactMap { call(fromJSON: $0) }.nilIfEmpty
            }
        default: return readHermes(reply)
        }
    }

    /// The call a JSON object names: `name` with `arguments` (or `parameters`), as an object or as
    /// the JSON text of one.
    private func call(fromJSON value: Any?) -> Call? {
        guard let object = value as? NFKJinjaOrderedObject,
              let name = object.pairs.first(where: { $0.0 == "name" })?.1 as? String else { return nil }
        let raw = object.pairs.first(where: { $0.0 == "arguments" || $0.0 == "parameters" })?.1
        if let arguments = raw as? NFKJinjaOrderedObject { return Call(name: name, arguments: arguments) }
        if let text = raw as? String, let parsed = NFKJinjaOrderedJSON.parse(text) as? NFKJinjaOrderedObject {
            return Call(name: name, arguments: parsed)
        }
        return Call(name: name, arguments: NFKJinjaOrderedObject(pairs: []))
    }

    /// Each `open` … `close` span, the last one allowed to run to the end of a reply cut off at its
    /// stop token. `handle` reads a span's body; a span it cannot read stays in the content.
    private func readSpans(_ reply: String, open: String, close: String,
                           _ handle: (String) -> [Call]?) -> (content: String, calls: [Call]) {
        var content = ""
        var calls: [Call] = []
        var rest = reply[...]
        while let start = rest.range(of: open) {
            content += rest[..<start.lowerBound]
            let afterOpen = rest[start.upperBound...]
            let end = afterOpen.range(of: close)
            let body = String(end.map { afterOpen[..<$0.lowerBound] } ?? afterOpen)
            if let read = handle(body) {
                calls += read
            } else {
                content += rest[start.lowerBound..<(end?.upperBound ?? rest.endIndex)]
            }
            rest = end.map { afterOpen[$0.upperBound...] } ?? ""
        }
        content += rest
        return (content.trimmingCharacters(in: .whitespacesAndNewlines), calls)
    }

    private func readHermes(_ reply: String) -> (content: String, calls: [Call]) {
        readSpans(reply, open: "<tool_call>", close: "</tool_call>") { body in
            call(fromJSON: NFKJinjaOrderedJSON.parse(body.trimmingCharacters(in: .whitespacesAndNewlines))).map { [$0] }
        }
    }

    private func readQwenXML(_ reply: String, tools: [[String: Any]]) -> (content: String, calls: [Call]) {
        let wrapped = readSpans(reply, open: "<tool_call>", close: "</tool_call>") { body in
            readFunctions(body, tools: tools).nilIfEmpty
        }
        guard wrapped.calls.isEmpty else { return wrapped }
        // A model that drops the wrapper still writes the function block.
        let bare = readFunctions(reply, tools: tools)
        guard !bare.isEmpty, let first = reply.range(of: "<function=") else { return wrapped }
        return (String(reply[..<first.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines), bare)
    }

    private func readFunctions(_ text: String, tools: [[String: Any]]) -> [Call] {
        var calls: [Call] = []
        var rest = text[...]
        while let open = rest.range(of: "<function=") {
            let afterName = rest[open.upperBound...]
            guard let nameEnd = afterName.range(of: ">") else { break }
            let name = String(afterName[..<nameEnd.lowerBound]).trimmingCharacters(in: .whitespaces)
            let body = afterName[nameEnd.upperBound...]
            let end = body.range(of: "</function>")
            let block = end.map { body[..<$0.lowerBound] } ?? body
            let properties = schemaProperties(of: name, in: tools)
            var pairs: [(String, Any)] = []
            var parameters = block
            while let start = parameters.range(of: "<parameter=") {
                let afterKey = parameters[start.upperBound...]
                guard let keyEnd = afterKey.range(of: ">") else { break }
                let key = String(afterKey[..<keyEnd.lowerBound]).trimmingCharacters(in: .whitespaces)
                let valueText = afterKey[keyEnd.upperBound...]
                let valueEnd = valueText.range(of: "</parameter>")
                var value = String(valueEnd.map { valueText[..<$0.lowerBound] } ?? valueText)
                if value.hasPrefix("\n") { value.removeFirst() }
                if value.hasSuffix("\n") { value.removeLast() }
                pairs.append((key, typed(value, schema: properties[key] as? [String: Any])))
                parameters = valueEnd.map { valueText[$0.upperBound...] } ?? ""
            }
            calls.append(Call(name: name, arguments: NFKJinjaOrderedObject(pairs: pairs)))
            rest = end.map { body[$0.upperBound...] } ?? ""
        }
        return calls
    }

    private func schemaProperties(of tool: String, in tools: [[String: Any]]) -> [String: Any] {
        for declaration in tools {
            let function = declaration["function"] as? [String: Any] ?? declaration
            guard function["name"] as? String == tool else { continue }
            let parameters = function["parameters"] as? [String: Any]
            return parameters?["properties"] as? [String: Any] ?? [:]
        }
        return [:]
    }

    /// A parameter written as text, read as its schema's type. A string, or a parameter the schema
    /// does not name, stays the text the model wrote.
    private func typed(_ text: String, schema: [String: Any]?) -> Any {
        guard let type = schema?["type"] as? String, type != "string" else { return text }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        switch type {
        case "integer":
            if let integer = Int(trimmed) { return NSNumber(value: integer) }
        case "number":
            if let integer = Int(trimmed) { return NSNumber(value: integer) }
            if let double = Double(trimmed) { return NSNumber(value: double) }
        case "boolean":
            if trimmed.lowercased() == "true" { return NSNumber(value: true) }
            if trimmed.lowercased() == "false" { return NSNumber(value: false) }
        default:
            if let parsed = NFKJinjaOrderedJSON.parse(trimmed) { return parsed }
        }
        return text
    }

    private func readGemma(_ reply: String) -> (content: String, calls: [Call]) {
        readSpans(reply, open: "<|tool_call>", close: "<tool_call|>") { body in
            guard body.hasPrefix("call:"), let brace = body.firstIndex(of: "{") else { return nil }
            let name = String(body[body.index(body.startIndex, offsetBy: 5)..<brace])
            var reader = NFKGemmaArgumentReader(String(body[brace...]))
            guard let arguments = reader.object() else { return nil }
            return [Call(name: name, arguments: arguments)]
        }
    }

    /// gpt-oss addresses a call `to=functions.name` in a commentary message whose body, after
    /// `<|message|>`, is the arguments' JSON, ended by `<|call|>`. The final message is the content.
    private func readHarmony(_ reply: String) -> (content: String, calls: [Call]) {
        var calls: [Call] = []
        var content = reply
        var rest = reply[...]
        while let address = rest.range(of: "to=functions.") {
            let afterAddress = rest[address.upperBound...]
            let nameEnd = afterAddress.firstIndex { $0 == " " || $0 == "<" } ?? afterAddress.endIndex
            let name = String(afterAddress[..<nameEnd])
            guard let message = afterAddress.range(of: "<|message|>") else { break }
            let body = afterAddress[message.upperBound...]
            let end = body.range(of: "<|call|>") ?? body.range(of: "<|end|>")
            let argumentsText = String(end.map { body[..<$0.lowerBound] } ?? body)
            let arguments = NFKJinjaOrderedJSON.parse(argumentsText.trimmingCharacters(in: .whitespacesAndNewlines))
                as? NFKJinjaOrderedObject ?? NFKJinjaOrderedObject(pairs: [])
            calls.append(Call(name: name, arguments: arguments))
            let spanStart = rest[..<address.lowerBound].range(of: "<|start|>", options: .backwards)?.lowerBound
                ?? rest[..<address.lowerBound].range(of: "<|channel|>", options: .backwards)?.lowerBound
                ?? address.lowerBound
            content = content.replacingOccurrences(of: String(rest[spanStart..<(end?.upperBound ?? rest.endIndex)]), with: "")
            rest = end.map { body[$0.upperBound...] } ?? ""
        }
        for marker in ["<|start|>assistant", "<|channel|>final<|message|>", "<|channel|>commentary<|message|>",
                       "<|end|>", "<|return|>", "<|call|>"] {
            content = content.replacingOccurrences(of: marker, with: "")
        }
        return (content.trimmingCharacters(in: .whitespacesAndNewlines), calls)
    }

    private func readDeepSeek(_ reply: String) -> (content: String, calls: [Call]) {
        let begin = "<｜tool▁call▁begin｜>", separator = "<｜tool▁sep｜>", end = "<｜tool▁call▁end｜>"
        let calls = readSpans(reply, open: begin, close: end) { body in
            guard let split = body.range(of: separator) else { return nil }
            var name = String(body[..<split.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            var argumentsText = String(body[split.upperBound...])
            // V3 writes `function<｜tool▁sep｜>name` and the arguments in a fenced JSON block.
            if name == "function", let newline = argumentsText.firstIndex(of: "\n") {
                name = String(argumentsText[..<newline]).trimmingCharacters(in: .whitespaces)
                argumentsText = String(argumentsText[newline...])
                    .replacingOccurrences(of: "```json", with: "").replacingOccurrences(of: "```", with: "")
            }
            let arguments = NFKJinjaOrderedJSON.parse(argumentsText.trimmingCharacters(in: .whitespacesAndNewlines))
                as? NFKJinjaOrderedObject ?? NFKJinjaOrderedObject(pairs: [])
            return [Call(name: name, arguments: arguments)]
        }.calls
        guard !calls.isEmpty else { return (reply.trimmingCharacters(in: .whitespacesAndNewlines), []) }
        let firstMarker = ["<｜tool▁calls▁begin｜>", begin].compactMap { reply.range(of: $0)?.lowerBound }.min()!
        return (String(reply[..<firstMarker]).trimmingCharacters(in: .whitespacesAndNewlines), calls)
    }

    private func readMistral(_ reply: String) -> (content: String, calls: [Call]) {
        guard let first = reply.range(of: "[TOOL_CALLS]") else { return (reply.trimmingCharacters(in: .whitespacesAndNewlines), []) }
        let content = String(reply[..<first.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        var calls: [Call] = []
        for segment in reply[first.upperBound...].components(separatedBy: "[TOOL_CALLS]") {
            let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("[") {
                calls += (NFKJinjaOrderedJSON.parse(trimmed) as? [Any] ?? []).compactMap { call(fromJSON: $0) }
            } else if let args = trimmed.range(of: "[ARGS]") {
                let name = String(trimmed[..<args.lowerBound])
                let arguments = NFKJinjaOrderedJSON.parse(String(trimmed[args.upperBound...])) as? NFKJinjaOrderedObject
                calls.append(Call(name: name, arguments: arguments ?? NFKJinjaOrderedObject(pairs: [])))
            }
        }
        return (content, calls)
    }

    private func readJSONList(_ reply: String, after marker: String) -> (content: String, calls: [Call]) {
        guard let start = reply.range(of: marker) else { return (reply.trimmingCharacters(in: .whitespacesAndNewlines), []) }
        let listText = String(reply[start.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        let parsed = NFKJinjaOrderedJSON.parse(listText)
        let entries = parsed as? [Any] ?? (parsed.map { [$0] } ?? [])
        let calls = entries.compactMap { call(fromJSON: $0) }
        guard !calls.isEmpty else { return (reply.trimmingCharacters(in: .whitespacesAndNewlines), []) }
        return (String(reply[..<start.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines), calls)
    }

    private func readLlama(_ reply: String) -> (content: String, calls: [Call]) {
        let body = reply.replacingOccurrences(of: "<|python_tag|>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard body.hasPrefix("{"), let call = call(fromJSON: NFKJinjaOrderedJSON.parse(body)) else {
            return (reply.trimmingCharacters(in: .whitespacesAndNewlines), [])
        }
        return ("", [call])
    }
}

/// Gemma 4's argument syntax: bare keys, strings between `<|"|>` marks, and JSON's other values.
private struct NFKGemmaArgumentReader {
    private let text: [Character]
    private var i = 0
    private static let quote = Array("<|\"|>")

    init(_ text: String) { self.text = Array(text) }

    mutating func object() -> NFKJinjaOrderedObject? {
        guard peek() == "{" else { return nil }
        i += 1
        var pairs: [(String, Any)] = []
        skipSpace()
        if peek() == "}" { i += 1; return NFKJinjaOrderedObject(pairs: pairs) }
        while i < text.count {
            skipSpace()
            let key: String
            if startsQuote() {
                guard let quoted = string() else { return nil }
                key = quoted
            } else {
                var bare = ""
                while i < text.count, text[i] != ":" { bare.append(text[i]); i += 1 }
                key = bare.trimmingCharacters(in: .whitespaces)
            }
            guard peek() == ":" else { return nil }
            i += 1
            guard let member = value() else { return nil }
            pairs.append((key, member))
            skipSpace()
            if peek() == "," { i += 1; continue }
            if peek() == "}" { i += 1; return NFKJinjaOrderedObject(pairs: pairs) }
            return nil
        }
        return nil
    }

    private mutating func value() -> Any? {
        skipSpace()
        if startsQuote() { return string() }
        switch peek() {
        case "{": return object()
        case "[":
            i += 1
            var items: [Any] = []
            skipSpace()
            if peek() == "]" { i += 1; return items }
            while i < text.count {
                guard let item = value() else { return nil }
                items.append(item)
                skipSpace()
                if peek() == "," { i += 1; continue }
                if peek() == "]" { i += 1; return items }
                return nil
            }
            return nil
        default:
            var token = ""
            while i < text.count, !",}]".contains(text[i]) { token.append(text[i]); i += 1 }
            token = token.trimmingCharacters(in: .whitespaces)
            switch token {
            case "true": return NSNumber(value: true)
            case "false": return NSNumber(value: false)
            case "null": return NSNull()
            default:
                if let integer = Int(token) { return NSNumber(value: integer) }
                if let double = Double(token) { return NSNumber(value: double) }
                return token
            }
        }
    }

    private func peek() -> Character? { i < text.count ? text[i] : nil }

    private mutating func skipSpace() {
        while i < text.count, text[i].isWhitespace { i += 1 }
    }

    private func startsQuote() -> Bool {
        i + Self.quote.count <= text.count && Array(text[i..<i + Self.quote.count]) == Self.quote
    }

    private mutating func string() -> String? {
        i += Self.quote.count
        var out = ""
        while i < text.count {
            if startsQuote() { i += Self.quote.count; return out }
            out.append(text[i])
            i += 1
        }
        return nil
    }
}

private extension Array {
    var nilIfEmpty: Self? { isEmpty ? nil : self }
}

/// Tool calling for the on-device language backends.
enum NFKMLXToolCalling {

    /// The record format a backend keeps its own assistant turns in: the reasoning the turn showed.
    static let recordFormat = "mlx-chat-template"

    /// Whether a request needs the release's chat template to carry its tools: it offers tools, or
    /// its conversation holds a call or an answer.
    static func carriesTools(_ request: NFKInferenceRequest) -> Bool {
        if templateTools(for: request) != nil { return true }
        return request.messages?.contains { $0["role"] as? String == "tool" || $0["tool_calls"] != nil } ?? false
    }

    /// The template a request renders through: the one the options name, or, for a request that
    /// carries tools on a backend whose options name none, the release's own `chat_template`, since
    /// only that template writes the release's tool format.
    static func template(_ chosen: NFKMLXChatTemplate, for request: NFKInferenceRequest,
                         releaseDirectoryURL: URL?) -> NFKMLXChatTemplate {
        guard case .none = chosen, carriesTools(request),
              let template = releaseDirectoryURL.flatMap(NFKMLXLanguage.chatTemplate(inDirectory:)) else { return chosen }
        return .jinja(template: template)
    }

    /// The format a request names, or the one `template` shows.
    static func format(for request: NFKInferenceRequest, template: String?) -> NFKMLXToolCallFormat? {
        if let value = request.parameter(forKey: NFKMLXGenerationParameterKey.toolCallFormat),
           let named = NFKMLXToolCallFormat.named(value) {
            return named
        }
        return template.flatMap(NFKMLXToolCallFormat.detected(inChatTemplate:))
    }

    /// The prompt a tool request renders on a backend that builds its plain chats without the
    /// release's template: that template over the prepared conversation and the tools, with the
    /// syntax it shows. Nil when the request carries no tools or the release ships no template, so the
    /// backend's own prompt stands.
    static func toolPrompt(for request: NFKInferenceRequest, template: String?, source: String,
                           bosToken: String = "", eosToken: String = "") throws -> (text: String, format: NFKMLXToolCallFormat?)? {
        guard carriesTools(request), let template, let messages = request.messages else { return nil }
        let format = format(for: request, template: template)
        let text = try NFKMLXChatTemplateRenderer.render(template, messages: templateMessages(messages, format: format, source: source),
                                                         bosToken: bosToken, eosToken: eosToken,
                                                         tools: templateTools(for: request))
        return (text, format)
    }

    /// The stops a tool request generates to: the release's generation-config ids and its
    /// end-of-sequence id. A tool call ends the turn at a marker of its own (gpt-oss's `<|call|>`),
    /// which only the generation config names. Nil leaves the backend's own stop.
    static func toolStops(for request: NFKInferenceRequest, chosen: Set<Int>, endOfSequence: Int,
                          releaseDirectoryURL: URL?) -> Set<Int>? {
        guard chosen.isEmpty, carriesTools(request) else { return nil }
        var stops = generationConfigStops(directoryURL: releaseDirectoryURL)
        guard !stops.isEmpty else { return nil }
        if endOfSequence >= 0 { stops.insert(endOfSequence) }
        return stops
    }

    /// The ids a release's `generation_config.json` lists as ending generation. An instruct release
    /// ends a turn on its own marker, which only the generation config names.
    static func generationConfigStops(directoryURL: URL?) -> Set<Int> {
        guard let url = directoryURL?.appendingPathComponent("generation_config.json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [] }
        if let single = json["eos_token_id"] as? NSNumber { return [single.intValue] }
        return Set((json["eos_token_id"] as? [NSNumber] ?? []).map(\.intValue))
    }

    /// The declarations a request offers, in the wire shape chat templates render: each
    /// `{"type": "function", "function": {name, description, parameters}}`.
    static func templateTools(for request: NFKInferenceRequest) -> [[String: Any]]? {
        guard let declared = request.parameter(forKey: NFKParameterTools) as? [[String: Any]], !declared.isEmpty else {
            return nil
        }
        return declared.map { $0["type"] != nil ? $0 : ["type": "function", "function": $0] }
    }

    /// A conversation in the shape chat templates read. Each call's arguments become an object in
    /// the order the model wrote them; each answer carries its tool's name and states a failure in
    /// its text; an assistant turn this backend wrote gets its reasoning back under
    /// `reasoning_content` and, without content, `thinking`, which gpt-oss reads. harmony renders
    /// one call per assistant message, so a turn with several becomes one message per call, each
    /// followed by its answer.
    static func templateMessages(_ messages: [[String: Any]], format: NFKMLXToolCallFormat?,
                                 source: String) -> [[String: Any]] {
        var prepared: [[String: Any]] = []
        var placed = Set<String>()
        for (index, message) in messages.enumerated() {
            switch message["role"] as? String {
            case "tool":
                let identifier = message["tool_call_id"] as? String ?? ""
                if !placed.contains(identifier) {
                    prepared.append(answer(message, in: messages))
                }
            case "assistant":
                let turns = assistantMessages(message, format: format, source: source)
                guard turns.count > 1 else {
                    prepared += turns
                    continue
                }
                let answers = messages[(index + 1)...].prefix { $0["role"] as? String == "tool" }
                for turn in turns {
                    prepared.append(turn)
                    let identifier = ((turn["tool_calls"] as? [[String: Any]])?.first?["id"] as? String) ?? ""
                    if let reply = answers.first(where: { $0["tool_call_id"] as? String == identifier }) {
                        prepared.append(answer(reply, in: messages))
                        placed.insert(identifier)
                    }
                }
            default:
                prepared.append(NFKToolMessages.messagesWithoutMetadata([message]).first ?? message)
            }
        }
        return prepared
    }

    private static func answer(_ message: [String: Any], in conversation: [[String: Any]]) -> [String: Any] {
        var text = NFKToolMessages.text(ofToolMessage: message)
        if NFKToolMessages.isErrorToolMessage(message) { text = "Error: " + text }
        var answer: [String: Any] = ["role": "tool", "tool_call_id": message["tool_call_id"] ?? "", "content": text]
        answer["name"] = NFKToolMessages.toolName(ofMessage: message, inConversation: conversation)
        return answer
    }

    private static func assistantMessages(_ message: [String: Any], format: NFKMLXToolCallFormat?,
                                          source: String) -> [[String: Any]] {
        var turn = NFKToolMessages.messagesWithoutMetadata([message]).first ?? message
        let record = NFKToolMessages.record(ofMessage: message, format: recordFormat, source: source) as? [String: Any]
        let content = turn["content"] as? String ?? ""
        if let reasoning = record?["reasoning"] as? String, !reasoning.isEmpty {
            turn["reasoning_content"] = reasoning
            if content.isEmpty { turn["thinking"] = reasoning }
        }
        let calls = NFKToolMessages.toolCalls(ofAssistantMessage: message)
        guard !calls.isEmpty else { return [turn] }
        let wireCalls: [[String: Any]] = calls.map { call in
            let text = call["argumentsJSON"] as? String ?? "{}"
            let arguments: Any = NFKJinjaOrderedJSON.parse(text) as? NFKJinjaOrderedObject ?? call["arguments"] ?? [:]
            return ["id": call["id"] ?? "", "type": "function", "function": ["name": call["name"] ?? "", "arguments": arguments]]
        }
        guard format == .harmony, wireCalls.count > 1 else {
            turn["tool_calls"] = wireCalls
            return [turn]
        }
        return wireCalls.enumerated().map { index, call in
            var single = index == 0 ? turn : ["role": "assistant", "content": ""]
            single["tool_calls"] = [call]
            return single
        }
    }

    /// The result keys ``outputs(answer:reasoning:request:format:source:)`` can write.
    static let outputKeys: Set<String> = [NFKOutputText, NFKOutputToolCalls, NFKOutputAssistantMessage]

    /// The outputs a reply becomes: its content under `NFKOutputText`, its calls under
    /// `NFKOutputToolCalls` when the request offered tools, and the turn to append next under
    /// `NFKOutputAssistantMessage`, which keeps the reasoning for this backend to render back.
    static func outputs(answer: String, reasoning: String, request: NFKInferenceRequest,
                        format: NFKMLXToolCallFormat?, source: String) -> [String: Any] {
        var content = answer
        var calls: [[String: Any]] = []
        if let tools = templateTools(for: request), let format {
            let read = format.read(answer, tools: tools)
            if !read.calls.isEmpty {
                content = read.content
                calls = read.calls.map { call in
                    let arguments = NFKMLXChatTemplateRenderer.value(from: call.arguments)
                    let identifier = "call_" + String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24))
                    return ["id": identifier,
                            "name": call.name,
                            "arguments": foundation(call.arguments),
                            "argumentsJSON": NFKJinjaJSON.encode(arguments)]
                }
            }
        }
        var outputs: [String: Any] = [NFKOutputText: content]
        if !calls.isEmpty {
            outputs[NFKOutputToolCalls] = calls
        }
        outputs[NFKOutputAssistantMessage] = NFKToolMessages.assistantMessage(
            withText: content, toolCalls: calls.isEmpty ? nil : calls,
            record: reasoning.isEmpty ? nil : ["reasoning": reasoning],
            format: reasoning.isEmpty ? nil : recordFormat, source: source)
        return outputs
    }

    /// A value read by `NFKJinjaOrderedJSON` as Foundation objects.
    static func foundation(_ value: Any) -> Any {
        switch value {
        case let object as NFKJinjaOrderedObject:
            var dictionary: [String: Any] = [:]
            for (key, member) in object.pairs { dictionary[key] = foundation(member) }
            return dictionary
        case let array as [Any]: return array.map(foundation)
        default: return value
        }
    }
}
