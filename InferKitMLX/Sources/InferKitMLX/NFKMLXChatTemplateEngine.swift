//
//  NFKMLXChatTemplateEngine.swift
//  InferKitMLX
//
//  The lexer, parser, and evaluator behind `NFKMLXChatTemplate`. See that file for the rationale.
//  The whitespace model is the one transformers compiles a chat template with: `trim_blocks` and
//  `lstrip_blocks` on, plus the explicit `{%-`/`-%}` and `{{-`/`-}}` markers.
//

import Foundation

// MARK: - JSON encoding (the `tojson` filter and list/dict stringification)

enum NFKJinjaJSON {
    static func encode(_ value: NFKJinjaValue) -> String {
        switch value {
        case .undefined, .none: return "null"
        case .bool(let value): return value ? "true" : "false"
        case .int(let value): return String(value)
        case .double(let value): return String(value)
        case .string(let value): return encodeString(value)
        case .list(let items): return "[" + items.map(encode).joined(separator: ", ") + "]"
        case .dictionary(let map):
            let body = map.keys.sorted().map { "\(encodeString($0)): \(encode(map[$0]!))" }
            return "{" + body.joined(separator: ", ") + "}"
        case .namespace(let object):
            let map = object.attributes
            let body = map.keys.sorted().map { "\(encodeString($0)): \(encode(map[$0]!))" }
            return "{" + body.joined(separator: ", ") + "}"
        }
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
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out + "\""
    }
}

// MARK: - AST

indirect enum NFKJinjaExpr {
    case stringLiteral(String)
    case intLiteral(Int)
    case boolLiteral(Bool)
    case noneLiteral
    case listLiteral([NFKJinjaExpr])
    case variable(String)
    case attribute(NFKJinjaExpr, String)
    case index(NFKJinjaExpr, NFKJinjaExpr)
    case slice(NFKJinjaExpr, NFKJinjaExpr?, NFKJinjaExpr?, NFKJinjaExpr?)
    case call(NFKJinjaExpr, [NFKJinjaExpr], [(String, NFKJinjaExpr)])
    case filter(NFKJinjaExpr, String, [NFKJinjaExpr])
    case unary(String, NFKJinjaExpr)
    case binary(String, NFKJinjaExpr, NFKJinjaExpr)
    case test(NFKJinjaExpr, String, Bool, NFKJinjaExpr?)
    case ternary(NFKJinjaExpr, NFKJinjaExpr, NFKJinjaExpr)
}

enum NFKJinjaSetTarget {
    case name(String)
    case attribute(String, String)
}

indirect enum NFKJinjaNode {
    case text(String)
    case output(NFKJinjaExpr)
    case forLoop(String, NFKJinjaExpr, [NFKJinjaNode])
    case conditional([(NFKJinjaExpr?, [NFKJinjaNode])])
    case set(NFKJinjaSetTarget, NFKJinjaExpr)
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
                while j < scalars.count {
                    if Character(scalars[j]) == "-" && j + 2 < scalars.count
                        && Character(scalars[j + 1]) == closeA && Character(scalars[j + 2]) == "}" {
                        trimRight = true
                        j += 3
                        break
                    }
                    if j + 1 < scalars.count && Character(scalars[j]) == closeA && Character(scalars[j + 1]) == "}" {
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
                    // A comment: contributes no tag; splice its surrounding literals back together
                    // by leaving a marker the trimming pass treats as a block boundary that emits
                    // nothing. Simpler: append an empty block that the parser ignores.
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
                let keyword = source.split(separator: " ", maxSplits: 1).first.map(String.init) ?? source
                if terminators.contains(keyword) {
                    return (nodes, keyword)
                }
                position += 1
                switch keyword {
                case "for": nodes.append(try parseFor(source))
                case "if": nodes.append(try parseIf(source))
                case "set": nodes.append(try parseSet(source))
                default: throw NFKJinjaError.syntax("unknown block '\(keyword)'")
                }
            }
        }
        return (nodes, nil)
    }

    private func parseFor(_ source: String) throws -> NFKJinjaNode {
        // for <var> in <expr>
        let rest = source.dropFirst("for".count)
        guard let inRange = rest.range(of: " in ") else { throw NFKJinjaError.syntax("malformed for") }
        let varName = rest[rest.startIndex..<inRange.lowerBound].trimmingCharacters(in: .whitespaces)
        let seqSource = String(rest[inRange.upperBound...])
        let seq = try parseExpressionString(seqSource)
        let (body, terminator) = try parseNodes(until: ["endfor"])
        guard terminator == "endfor" else { throw NFKJinjaError.syntax("for without endfor") }
        position += 1
        return .forLoop(varName, seq, body)
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
        guard let equals = rest.firstIndex(of: "=") else { throw NFKJinjaError.syntax("malformed set") }
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
