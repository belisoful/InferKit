//
//  NFKMLXChatTemplate.swift
//  InferKitMLX
//
//  A native renderer for the Jinja `chat_template` an instruct release ships. Rendering it wrong
//  silently changes the model's input — the same failure class as the qwen2 pre-tokenization defect —
//  so the language backend should render the release's own template rather than a hand-coded format.
//
//  This is a compact Jinja interpreter for the subset chat templates use: text with `{{ … }}` output
//  and `{% … %}` control (for, if, set and block set, macro, break and continue, transformers'
//  `generation` block), the whitespace controls transformers compiles a chat template with
//  (`trim_blocks`, `lstrip_blocks`, and the explicit `-` markers), and the expression language the
//  templates read: attribute and index access, slicing, `namespace`, the `loop` variable, macros,
//  string and dictionary methods, the filters and `is` tests the shipped templates use, and Python's
//  arithmetic and printing. It is held to transformers' own `apply_chat_template` output over the
//  shipped chat cases and the tool conversations of each tool-calling release.
//
//  Key order: transformers emits a dictionary in insertion order, and a Foundation dictionary keeps
//  none. Iteration and `tojson` order a dictionary's keys by `NFKJinjaKeyOrder`: the keys of a tool
//  declaration and a JSON Schema in the order they are written (type, function, name, description,
//  parameters, properties, required), then the rest alphabetically. A tool declared the way the core
//  writes one renders byte-exact; a schema whose properties a caller listed out of alphabetical order
//  renders them alphabetically.
//

import Foundation

/// A value in the template's expression language.
indirect enum NFKJinjaValue {
    case undefined
    case none
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case list([NFKJinjaValue])
    case dictionary(NFKJinjaMap)
    /// A mutable `namespace(...)`, whose attributes a `set ns.x = …` writes.
    case namespace(NFKJinjaNamespace)
    /// A `{% macro %}`, called like a function.
    case macro(NFKJinjaMacro)

    var isTruthy: Bool {
        switch self {
        case .undefined, .none: return false
        case .bool(let value): return value
        case .int(let value): return value != 0
        case .double(let value): return value != 0
        case .string(let value): return !value.isEmpty
        case .list(let value): return !value.isEmpty
        case .dictionary(let value): return !value.isEmpty
        case .namespace, .macro: return true
        }
    }

    /// The value as Jinja prints it: Python's `str()`, so a list or dictionary prints as its repr.
    var asString: String {
        switch self {
        case .undefined: return ""
        case .string(let value): return value
        default: return NFKJinjaPython.repr(self)
        }
    }
}

/// A dictionary that keeps its keys in order: the order its source wrote them where the source
/// keeps one (a template's dictionary literal, JSON read by `NFKJinjaOrderedJSON`), otherwise
/// `NFKJinjaKeyOrder`'s.
struct NFKJinjaMap {
    private(set) var keys: [String]
    private var storage: [String: NFKJinjaValue]

    init(_ dictionary: [String: NFKJinjaValue] = [:]) {
        storage = dictionary
        keys = NFKJinjaKeyOrder.keys(of: dictionary)
    }

    /// A later pair with an earlier pair's key replaces its value and keeps its place, as a Python
    /// dictionary does.
    init(pairs: [(String, NFKJinjaValue)]) {
        keys = []
        storage = [:]
        for (key, value) in pairs { self[key] = value }
    }

    subscript(key: String) -> NFKJinjaValue? {
        get { storage[key] }
        set {
            if let newValue {
                if storage[key] == nil { keys.append(key) }
                storage[key] = newValue
            } else if storage.removeValue(forKey: key) != nil {
                keys.removeAll { $0 == key }
            }
        }
    }

    var count: Int { keys.count }
    var isEmpty: Bool { keys.isEmpty }
    var values: [NFKJinjaValue] { keys.map { storage[$0]! } }
    var pairs: [(String, NFKJinjaValue)] { keys.map { ($0, storage[$0]!) } }
    var dictionary: [String: NFKJinjaValue] { storage }
}

/// A JSON object as it was written, key order included; `NFKJinjaOrderedJSON` reads one and
/// `NFKMLXChatTemplateRenderer.value(from:)` keeps its order.
struct NFKJinjaOrderedObject {
    let pairs: [(String, Any)]
}

/// A mutable namespace object.
final class NFKJinjaNamespace {
    var attributes: [String: NFKJinjaValue]
    init(_ attributes: [String: NFKJinjaValue]) { self.attributes = attributes }
}

/// A macro's parameters, their defaults, and its body.
final class NFKJinjaMacro {
    let name: String
    let parameters: [(String, NFKJinjaExpr?)]
    let body: [NFKJinjaNode]
    init(name: String, parameters: [(String, NFKJinjaExpr?)], body: [NFKJinjaNode]) {
        self.name = name
        self.parameters = parameters
        self.body = body
    }
}

enum NFKJinjaError: Error { case syntax(String) }

/// The order a dictionary's keys iterate and serialize in. See the file header.
enum NFKJinjaKeyOrder {
    static let written = ["type", "function", "name", "description", "parameters", "properties", "items",
                          "enum", "required", "additionalProperties", "default"]

    static func keys(of map: [String: NFKJinjaValue]) -> [String] {
        map.keys.sorted { left, right in
            let l = written.firstIndex(of: left) ?? written.count
            let r = written.firstIndex(of: right) ?? written.count
            return l != r ? l < r : left < right
        }
    }
}

/// Python's printing of the values a template holds.
enum NFKJinjaPython {
    static func repr(_ value: NFKJinjaValue) -> String {
        switch value {
        case .undefined: return ""
        case .none: return "None"
        case .bool(let value): return value ? "True" : "False"
        case .int(let value): return String(value)
        case .double(let value): return float(value)
        case .string(let value): return stringRepr(value)
        case .list(let items): return "[" + items.map(repr).joined(separator: ", ") + "]"
        case .dictionary(let map):
            return "{" + map.pairs.map { "\(stringRepr($0.0)): \(repr($0.1))" }.joined(separator: ", ") + "}"
        case .namespace: return "<Namespace>"
        case .macro(let macro): return "<Macro '\(macro.name)'>"
        }
    }

    /// Python's float repr: the shortest round-tripping digits, with `.0` on an integral value and
    /// exponents from 1e16 and below 1e-4, as Swift's description already writes them.
    static func float(_ value: Double) -> String {
        if value.isNaN { return "nan" }
        if value.isInfinite { return value < 0 ? "-inf" : "inf" }
        return String(value)
    }

    /// Python quotes a string in single quotes unless it holds a single quote and no double one.
    static func stringRepr(_ string: String) -> String {
        let quote: Character = string.contains("'") && !string.contains("\"") ? "\"" : "'"
        var out = String(quote)
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if Character(scalar) == quote {
                    out += "\\" + String(quote)
                } else if scalar.value < 0x20 || scalar.value == 0x7f {
                    out += String(format: "\\x%02x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + String(quote)
    }
}

/// Renders a Jinja chat template.
public enum NFKMLXChatTemplateRenderer {

    /// Renders `template` over `messages` and the standard chat variables.
    ///
    /// - Parameters:
    ///   - template: the Jinja source, the release's `chat_template`.
    ///   - messages: each a `["role": ..., "content": ...]` dictionary; a content that is a list of
    ///     typed parts (a multimodal message) is passed through as a list.
    ///   - addGenerationPrompt: whether to append the assistant turn opener.
    ///   - bosToken: the release's beginning-of-sequence marker, for templates that reference it.
    ///   - eosToken: the release's end-of-sequence marker, for templates that reference it.
    ///   - tools: tool definitions in the provider's wire shape, bound to the template's `tools`.
    ///   - variables: further bindings the release's template reads, such as the `enable_thinking`
    ///     flag Qwen3 tests or the `reasoning_effort` level gpt-oss writes into its system message.
    ///     A template that reads none of them renders unchanged, which is what an unused binding
    ///     means in Jinja.
    ///   - date: the date `strftime_now` formats, which gpt-oss and Granite write into their system
    ///     message. Introduced in InferKit 0.4.0.
    public static func render(_ template: String, messages: [[String: Any]],
                              addGenerationPrompt: Bool = true,
                              bosToken: String = "", eosToken: String = "",
                              tools: [[String: Any]]? = nil,
                              variables: [String: Any] = [:],
                              date: Date = Date()) throws -> String {
        var context: [String: NFKJinjaValue] = [
            "messages": .list(messages.map(value(from:))),
            "add_generation_prompt": .bool(addGenerationPrompt),
            "bos_token": .string(bosToken),
            "eos_token": .string(eosToken),
        ]
        // transformers binds both to None when absent, so `is defined` holds and `is none` tests them.
        context["tools"] = tools.map { .list($0.map(value(from:))) } ?? NFKJinjaValue.none
        context["documents"] = NFKJinjaValue.none
        for (name, binding) in variables {
            context[name] = value(from: binding)
        }
        let nodes = try NFKJinjaParser(template).parse()
        var output = ""
        let evaluator = NFKJinjaEvaluator(globals: context, date: date)
        try evaluator.run(nodes, into: &output)
        return output
    }

    /// Converts a Foundation JSON-like value (from a message dictionary) into a template value. A
    /// Foundation number keeps its kind: a boolean stays a boolean, and an integral number an int.
    static func value(from any: Any) -> NFKJinjaValue {
        switch any {
        case let string as String: return .string(string)
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return .bool(number.boolValue) }
            if CFNumberIsFloatType(number) { return .double(number.doubleValue) }
            return .int(number.intValue)
        case let array as [Any]: return .list(array.map(value(from:)))
        case let ordered as NFKJinjaOrderedObject:
            return .dictionary(NFKJinjaMap(pairs: ordered.pairs.map { ($0.0, value(from: $0.1)) }))
        case let dictionary as [String: Any]:
            return .dictionary(NFKJinjaMap(dictionary.mapValues(value(from:))))
        default: return .none
        }
    }
}
