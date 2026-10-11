//
//  NFKMLXChatTemplateEngine.swift
//  InferKitMLX
//
//  The lexer, parser, and evaluator behind `NFKMLXChatTemplate`. See that file for the rationale.
//  The whitespace model is the one transformers compiles a chat template with: `trim_blocks` and
//  `lstrip_blocks` on, plus the explicit `{%-`/`-%}` and `{{-`/`-}}` markers.
//

import Foundation

// MARK: - JSON encoding (the `tojson` filter)

/// `json.dumps` as transformers' `tojson` calls it: `ensure_ascii=False`, the default separators, and
/// an optional indent.
enum NFKJinjaJSON {
    struct Style {
        let indent: Int?
        let itemSeparator: String
        let keySeparator: String
        let sortsKeys: Bool
    }

    /// `separators` defaults as `json.dumps` does: `(", ", ": ")`, or `(",", ": ")` with an indent.
    static func encode(_ value: NFKJinjaValue, indent: Int? = nil, separators: (String, String)? = nil,
                       sortsKeys: Bool = false) -> String {
        let style = Style(indent: indent, itemSeparator: separators?.0 ?? (indent == nil ? ", " : ","),
                          keySeparator: separators?.1 ?? ": ", sortsKeys: sortsKeys)
        return encode(value, style: style, depth: 0)
    }

    private static func encode(_ value: NFKJinjaValue, style: Style, depth: Int) -> String {
        switch value {
        case .undefined, .none: return "null"
        case .bool(let value): return value ? "true" : "false"
        case .int(let value): return String(value)
        case .double(let value):
            if value.isNaN { return "NaN" }
            if value.isInfinite { return value < 0 ? "-Infinity" : "Infinity" }
            return String(value)
        case .string(let value): return encodeString(value)
        case .list(let items):
            return container("[", "]", items.map { encode($0, style: style, depth: depth + 1) }, style: style, depth: depth)
        case .dictionary(let map):
            return container("{", "}", members(map, style: style, depth: depth), style: style, depth: depth)
        case .namespace(let object):
            return container("{", "}", members(NFKJinjaMap(object.attributes), style: style, depth: depth), style: style, depth: depth)
        case .macro: return "null"
        }
    }

    private static func members(_ map: NFKJinjaMap, style: Style, depth: Int) -> [String] {
        let keys = style.sortsKeys ? map.keys.sorted() : map.keys
        return keys.map { encodeString($0) + style.keySeparator + encode(map[$0]!, style: style, depth: depth + 1) }
    }

    private static func container(_ open: String, _ close: String, _ members: [String], style: Style, depth: Int) -> String {
        guard let indent = style.indent, !members.isEmpty else { return open + members.joined(separator: style.itemSeparator) + close }
        let inner = "\n" + String(repeating: " ", count: indent * (depth + 1))
        let outer = "\n" + String(repeating: " ", count: indent * depth)
        return open + inner + members.joined(separator: style.itemSeparator + inner) + outer + close
    }

    private static func encodeString(_ string: String) -> String {
        var out = "\""
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}

// MARK: - AST

indirect enum NFKJinjaExpr {
    case stringLiteral(String)
    case intLiteral(Int)
    case doubleLiteral(Double)
    case boolLiteral(Bool)
    case noneLiteral
    case listLiteral([NFKJinjaExpr])
    case dictLiteral([(NFKJinjaExpr, NFKJinjaExpr)])
    case variable(String)
    case attribute(NFKJinjaExpr, String)
    case index(NFKJinjaExpr, NFKJinjaExpr)
    case slice(NFKJinjaExpr, NFKJinjaExpr?, NFKJinjaExpr?, NFKJinjaExpr?)
    case call(NFKJinjaExpr, [NFKJinjaExpr], [(String, NFKJinjaExpr)])
    case filter(NFKJinjaExpr, String, [NFKJinjaExpr], [(String, NFKJinjaExpr)])
    case unary(String, NFKJinjaExpr)
    case binary(String, NFKJinjaExpr, NFKJinjaExpr)
    case test(NFKJinjaExpr, String, Bool, [NFKJinjaExpr])
    case ternary(NFKJinjaExpr, NFKJinjaExpr, NFKJinjaExpr?)
}

enum NFKJinjaSetTarget {
    case name(String)
    case attribute(String, String)
}

/// A `for` loop: its targets (one name, or several unpacked from each element), the sequence, the
/// optional `if` that filters it, the body, and the `else` body run when nothing was iterated.
struct NFKJinjaLoop {
    let targets: [String]
    let sequence: NFKJinjaExpr
    let condition: NFKJinjaExpr?
    let body: [NFKJinjaNode]
    let otherwise: [NFKJinjaNode]
}

indirect enum NFKJinjaNode {
    case text(String)
    case output(NFKJinjaExpr)
    case forLoop(NFKJinjaLoop)
    case conditional([(NFKJinjaExpr?, [NFKJinjaNode])])
    case set(NFKJinjaSetTarget, NFKJinjaExpr)
    case setBlock(String, [NFKJinjaNode])
    case macro(NFKJinjaMacro)
    case breakLoop
    case continueLoop
}

// MARK: - Template lexer + parser

/// One raw tag boundary, before whitespace trimming is applied to the literals around it.
private struct NFKJinjaRawTag {
    enum Kind { case output, block }
    let kind: Kind
    let source: String
    let trimLeft: Bool
    let trimRight: Bool
}

private enum NFKJinjaToken {
    case text(String)
    case output(String)
    case block(String)
}

final class NFKJinjaParser {
    private let tokens: [NFKJinjaToken]
    private var position = 0

    init(_ template: String) throws {
        self.tokens = try NFKJinjaParser.tokenize(template)
    }

    // MARK: Tokenization + whitespace

    private static func tokenize(_ template: String) throws -> [NFKJinjaToken] {
        let scalars = Array(template.unicodeScalars)
        var literals: [String] = []
        var tags: [NFKJinjaRawTag] = []
        // literals[i] is the text before tags[i]; a trailing literal follows the last tag.
        var current = String.UnicodeScalarView()
        var i = 0
        func openMatches(_ start: Int, _ a: Character, _ b: Character) -> Bool {
            start + 1 < scalars.count && Character(scalars[start]) == a && Character(scalars[start + 1]) == b
        }
        while i < scalars.count {
            if openMatches(i, "{", "{") || openMatches(i, "{", "%") || openMatches(i, "{", "#") {
                let marker = Character(scalars[i + 1])
                var j = i + 2
                let trimLeft = j < scalars.count && Character(scalars[j]) == "-"
                if trimLeft { j += 1 }
                let closeA: Character = marker == "{" ? "}" : (marker == "%" ? "%" : "#")
                var body = String.UnicodeScalarView()
                var trimRight = false
                var quote: Character?
                while j < scalars.count {
                    let character = Character(scalars[j])
                    // A closing marker inside a string literal belongs to the string.
                    if marker != "#" {
                        if let open = quote {
                            if character == "\\" && j + 1 < scalars.count {
                                body.append(scalars[j])
                                body.append(scalars[j + 1])
                                j += 2
                                continue
                            }
                            if character == open { quote = nil }
                            body.append(scalars[j])
                            j += 1
                            continue
                        }
                        if character == "'" || character == "\"" {
                            quote = character
                            body.append(scalars[j])
                            j += 1
                            continue
                        }
                    }
                    if character == "-" && j + 2 < scalars.count
                        && Character(scalars[j + 1]) == closeA && Character(scalars[j + 2]) == "}" {
                        trimRight = true
                        j += 3
                        break
                    }
                    if j + 1 < scalars.count && character == closeA && Character(scalars[j + 1]) == "}" {
                        j += 2
                        break
                    }
                    body.append(scalars[j])
                    j += 1
                }
                literals.append(String(current))
                current = String.UnicodeScalarView()
                let source = String(body).trimmingCharacters(in: .whitespacesAndNewlines)
                if marker == "#" {
                    // A comment emits nothing, but its tag boundary still trims the literals around it.
                    tags.append(NFKJinjaRawTag(kind: .block, source: "#", trimLeft: trimLeft, trimRight: trimRight))
                } else {
                    tags.append(NFKJinjaRawTag(kind: marker == "{" ? .output : .block,
                                               source: source, trimLeft: trimLeft, trimRight: trimRight))
                }
                i = j
            } else {
                current.append(scalars[i])
                i += 1
            }
        }
        literals.append(String(current))

        // Apply whitespace trimming to each literal from its neighboring tags.
        var trimmed = literals
        for index in trimmed.indices {
            let precedingTag = index > 0 ? tags[index - 1] : nil
            let followingTag = index < tags.count ? tags[index] : nil
            var text = trimmed[index]
            if let tag = precedingTag {
                if tag.trimRight {
                    text = stripLeadingWhitespace(text)
                } else if tag.kind == .block {
                    text = stripOneLeadingNewline(text)
                }
            }
            if let tag = followingTag {
                if tag.trimLeft {
                    text = stripTrailingWhitespace(text)
                } else if tag.kind == .block {
                    text = stripTrailingLineIndent(text, isFirstLiteral: index == 0)
                }
            }
            trimmed[index] = text
        }

        var out: [NFKJinjaToken] = []
        for index in trimmed.indices {
            if !trimmed[index].isEmpty { out.append(.text(trimmed[index])) }
            if index < tags.count {
                let tag = tags[index]
                if tag.source == "#" { continue }
                out.append(tag.kind == .output ? .output(tag.source) : .block(tag.source))
            }
        }
        return out
    }

    private static func stripLeadingWhitespace(_ text: String) -> String {
        var view = Substring(text)
        while let first = view.first, first == " " || first == "\t" || first == "\n" || first == "\r" {
            view = view.dropFirst()
        }
        return String(view)
    }

    private static func stripTrailingWhitespace(_ text: String) -> String {
        var view = Substring(text)
        while let last = view.last, last == " " || last == "\t" || last == "\n" || last == "\r" {
            view = view.dropLast()
        }
        return String(view)
    }

    private static func stripOneLeadingNewline(_ text: String) -> String {
        if text.hasPrefix("\r\n") { return String(text.dropFirst(2)) }
        if text.hasPrefix("\n") { return String(text.dropFirst()) }
        return text
    }

    /// `lstrip_blocks` strips a block tag's line indentation — the spaces and tabs between the tag and
    /// the preceding newline. It acts only when the tag begins a source line, so a trailing whitespace
    /// run that follows content on the same line (no newline before it) is left intact.
    private static func stripTrailingLineIndent(_ text: String, isFirstLiteral: Bool) -> String {
        let characters = Array(text)
        var start = characters.count
        while start > 0, characters[start - 1] == " " || characters[start - 1] == "\t" {
            start -= 1
        }
        if start == 0 {
            // The whole literal is spaces/tabs: only a template-leading indent counts as a line start.
            return isFirstLiteral ? "" : text
        }
        if characters[start - 1] == "\n" || characters[start - 1] == "\r" {
            return String(characters[0 ..< start])
        }
        return text
    }

    // MARK: Node parsing

    func parse() throws -> [NFKJinjaNode] {
        let (nodes, terminator) = try parseNodes(until: [])
        if terminator != nil {
            throw NFKJinjaError.syntax("unexpected \(terminator!)")
        }
        return nodes
    }

    private func parseNodes(until terminators: Set<String>) throws -> ([NFKJinjaNode], String?) {
        var nodes: [NFKJinjaNode] = []
        while position < tokens.count {
            let token = tokens[position]
            switch token {
            case .text(let value):
                nodes.append(.text(value))
                position += 1
            case .output(let source):
                position += 1
                nodes.append(.output(try parseExpressionString(source)))
            case .block(let source):
                let keyword = source.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" }).first.map(String.init) ?? source
                if terminators.contains(keyword) {
                    return (nodes, keyword)
                }
                position += 1
                switch keyword {
                case "for": nodes.append(try parseFor(source))
                case "if": nodes.append(try parseIf(source))
                case "set": nodes.append(try parseSet(source))
                case "macro": nodes.append(try parseMacro(source))
                case "break": nodes.append(.breakLoop)
                case "continue": nodes.append(.continueLoop)
                case "generation":
                    // transformers' assistant-mask marker renders its body unchanged.
                    let (body, terminator) = try parseNodes(until: ["endgeneration"])
                    guard terminator == "endgeneration" else { throw NFKJinjaError.syntax("generation without endgeneration") }
                    position += 1
                    nodes.append(contentsOf: body)
                default: throw NFKJinjaError.syntax("unknown block '\(keyword)'")
                }
            }
        }
        return (nodes, nil)
    }

    private func parseFor(_ source: String) throws -> NFKJinjaNode {
        // for <target>[, <target>…] in <expr> [if <condition>]
        let rest = source.dropFirst("for".count)
        guard let inRange = rest.range(of: " in ") else { throw NFKJinjaError.syntax("malformed for") }
        let targets = rest[rest.startIndex..<inRange.lowerBound]
            .trimmingCharacters(in: CharacterSet(charactersIn: " ()"))
            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        var lexer = NFKJinjaExprLexer(String(rest[inRange.upperBound...]))
        var parser = NFKJinjaExprParser(try lexer.all())
        let sequence = try parser.parseExpression(allowingConditional: false)
        let condition = try parser.consumeKeywordIf("if") ? parser.parseExpression() : nil
        try parser.expectEnd()
        let (body, terminator) = try parseNodes(until: ["endfor", "else"])
        var otherwise: [NFKJinjaNode] = []
        if terminator == "else" {
            position += 1
            let (elseBody, elseTerminator) = try parseNodes(until: ["endfor"])
            guard elseTerminator == "endfor" else { throw NFKJinjaError.syntax("for else without endfor") }
            otherwise = elseBody
        } else if terminator != "endfor" {
            throw NFKJinjaError.syntax("for without endfor")
        }
        position += 1
        return .forLoop(NFKJinjaLoop(targets: targets, sequence: sequence, condition: condition, body: body, otherwise: otherwise))
    }

    private func parseIf(_ source: String) throws -> NFKJinjaNode {
        var branches: [(NFKJinjaExpr?, [NFKJinjaNode])] = []
        var condSource = String(source.dropFirst("if".count))
        while true {
            let cond = try parseExpressionString(condSource)
            let (body, terminator) = try parseNodes(until: ["elif", "else", "endif"])
            branches.append((cond, body))
            guard let terminator = terminator else { throw NFKJinjaError.syntax("if without endif") }
            let tagSource = currentBlockSource()
            position += 1
            if terminator == "endif" { break }
            if terminator == "elif" {
                condSource = String(tagSource.dropFirst("elif".count))
                continue
            }
            // else
            let (elseBody, elseTerminator) = try parseNodes(until: ["endif"])
            branches.append((nil, elseBody))
            guard elseTerminator == "endif" else { throw NFKJinjaError.syntax("else without endif") }
            position += 1
            break
        }
        return .conditional(branches)
    }

    private func currentBlockSource() -> String {
        if position < tokens.count, case .block(let source) = tokens[position] { return source }
        return ""
    }

    private func parseSet(_ source: String) throws -> NFKJinjaNode {
        let rest = source.dropFirst("set".count)
        guard let equals = rest.firstIndex(of: "=") else {
            // {% set name %}…{% endset %} captures its rendered body.
            let name = rest.trimmingCharacters(in: .whitespaces)
            let (body, terminator) = try parseNodes(until: ["endset"])
            guard terminator == "endset" else { throw NFKJinjaError.syntax("set without endset") }
            position += 1
            return .setBlock(name, body)
        }
        let target = rest[rest.startIndex..<equals].trimmingCharacters(in: .whitespaces)
        let valueSource = String(rest[rest.index(after: equals)...])
        let value = try parseExpressionString(valueSource)
        if let dot = target.firstIndex(of: ".") {
            let base = String(target[target.startIndex..<dot])
            let attr = String(target[target.index(after: dot)...])
            return .set(.attribute(base, attr), value)
        }
        return .set(.name(target), value)
    }

    private func parseMacro(_ source: String) throws -> NFKJinjaNode {
        // macro <name>(<parameter>[=<default>], …)
        let rest = source.dropFirst("macro".count).trimmingCharacters(in: .whitespaces)
        guard let open = rest.firstIndex(of: "("), rest.hasSuffix(")") else { throw NFKJinjaError.syntax("malformed macro") }
        let name = rest[rest.startIndex..<open].trimmingCharacters(in: .whitespaces)
        var lexer = NFKJinjaExprLexer(String(rest[open...]))
        var parser = NFKJinjaExprParser(try lexer.all())
        let parameters = try parser.parseParameterList()
        try parser.expectEnd()
        let (body, terminator) = try parseNodes(until: ["endmacro"])
        guard terminator == "endmacro" else { throw NFKJinjaError.syntax("macro without endmacro") }
        position += 1
        return .macro(NFKJinjaMacro(name: name, parameters: parameters, body: body))
    }

    // MARK: Expression parsing

    private func parseExpressionString(_ source: String) throws -> NFKJinjaExpr {
        var lexer = NFKJinjaExprLexer(source)
        let tokens = try lexer.all()
        var parser = NFKJinjaExprParser(tokens)
        let expr = try parser.parseExpression()
        try parser.expectEnd()
        return expr
    }
}
