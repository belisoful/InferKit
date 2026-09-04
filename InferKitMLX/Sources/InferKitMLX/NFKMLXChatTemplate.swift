//
//  NFKMLXChatTemplate.swift
//  InferKitMLX
//
//  A native renderer for the Jinja `chat_template` an instruct release ships. Rendering it wrong
//  silently changes the model's input — the same failure class as the qwen2 pre-tokenization defect —
//  so the language backend should render the release's own template rather than a hand-coded format.
//
//  This is a compact Jinja interpreter for the subset chat templates use: text with `{{ … }}` output
//  and `{% … %}` control (for / if / set), the whitespace controls transformers compiles a chat
//  template with (`trim_blocks`, `lstrip_blocks`, and the explicit `-` markers), and the expression
//  language the templates read — attribute and index access, slicing, `namespace`, the `loop` variable,
//  string methods, the `tojson`/`trim` filters, and the operators and `is` tests. It is held to
//  transformers' own `apply_chat_template` output over the shipped chat cases.
//
//  One documented divergence: `tojson` emits an object's keys in SORTED order, where transformers
//  emits them in insertion order. Foundation dictionaries do not preserve insertion order, so a
//  faithful whole-object serialization would need an ordered-map pipeline end to end. The rendering
//  is otherwise byte-exact; only the key order inside a serialized multi-key object differs, which
//  affects a tool schema's presentation, not a plain or multi-turn chat.
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
    case dictionary([String: NFKJinjaValue])
    /// A mutable `namespace(...)`, whose attributes a `set ns.x = …` writes.
    case namespace(NFKJinjaNamespace)

    var isTruthy: Bool {
        switch self {
        case .undefined, .none: return false
        case .bool(let value): return value
        case .int(let value): return value != 0
        case .double(let value): return value != 0
        case .string(let value): return !value.isEmpty
        case .list(let value): return !value.isEmpty
        case .dictionary(let value): return !value.isEmpty
        case .namespace: return true
        }
    }

    var asString: String {
        switch self {
        case .undefined, .none: return ""
        case .bool(let value): return value ? "True" : "False"
        case .int(let value): return String(value)
        case .double(let value): return String(value)
        case .string(let value): return value
        case .list, .dictionary, .namespace: return NFKJinjaJSON.encode(self)
        }
    }
}

/// A mutable namespace object.
final class NFKJinjaNamespace {
    var attributes: [String: NFKJinjaValue]
    init(_ attributes: [String: NFKJinjaValue]) { self.attributes = attributes }
}

enum NFKJinjaError: Error { case syntax(String) }

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
    public static func render(_ template: String, messages: [[String: Any]],
                              addGenerationPrompt: Bool = true,
                              bosToken: String = "", eosToken: String = "",
                              tools: [[String: Any]]? = nil) throws -> String {
        var context: [String: NFKJinjaValue] = [
            "messages": .list(messages.map(value(from:))),
            "add_generation_prompt": .bool(addGenerationPrompt),
            "bos_token": .string(bosToken),
            "eos_token": .string(eosToken),
        ]
        if let tools = tools {
            context["tools"] = .list(tools.map(value(from:)))
        }
        let nodes = try NFKJinjaParser(template).parse()
        var output = ""
        var evaluator = NFKJinjaEvaluator(context: context)
        try evaluator.run(nodes, into: &output)
        return output
    }

    /// Converts a Foundation JSON-like value (from a message dictionary) into a template value.
    static func value(from any: Any) -> NFKJinjaValue {
        switch any {
        case let string as String: return .string(string)
        case let bool as Bool: return .bool(bool)
        case let int as Int: return .int(int)
        case let double as Double: return .double(double)
        case let array as [Any]: return .list(array.map(value(from:)))
        case let dictionary as [String: Any]:
            return .dictionary(dictionary.mapValues(value(from:)))
        default: return .none
        }
    }
}
