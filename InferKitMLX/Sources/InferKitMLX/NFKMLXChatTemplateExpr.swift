//
//  NFKMLXChatTemplateExpr.swift
//  InferKitMLX
//
//  The expression sublanguage for `NFKMLXChatTemplate`: a tokenizer, a precedence-climbing parser,
//  and the evaluator. See `NFKMLXChatTemplate.swift` for the rationale.
//

import Foundation

// MARK: - Expression tokens

enum NFKJinjaExprToken: Equatable {
    case identifier(String)
    case integer(Int)
    case string(String)
    case op(String)
    case end
}

struct NFKJinjaExprLexer {
    private let scalars: [Unicode.Scalar]
    private var i = 0
    init(_ source: String) { self.scalars = Array(source.unicodeScalars) }

    mutating func all() throws -> [NFKJinjaExprToken] {
        var tokens: [NFKJinjaExprToken] = []
        while let token = try next() { tokens.append(token) }
        tokens.append(.end)
        return tokens
    }

    private mutating func next() throws -> NFKJinjaExprToken? {
        while i < scalars.count, isSpace(scalars[i]) { i += 1 }
        guard i < scalars.count else { return nil }
        let c = scalars[i]
        if c == "'" || c == "\"" { return .string(try readString(quote: c)) }
        if isDigit(c) { return .integer(readInteger()) }
        if isIdentStart(c) { return .identifier(readIdentifier()) }
        return .op(readOperator())
    }

    private func isSpace(_ c: Unicode.Scalar) -> Bool { c == " " || c == "\t" || c == "\n" || c == "\r" }
    private func isDigit(_ c: Unicode.Scalar) -> Bool { c >= "0" && c <= "9" }
    private func isIdentStart(_ c: Unicode.Scalar) -> Bool {
        (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || c == "_"
    }
    private func isIdentPart(_ c: Unicode.Scalar) -> Bool { isIdentStart(c) || isDigit(c) }

    private mutating func readIdentifier() -> String {
        var out = String.UnicodeScalarView()
        while i < scalars.count, isIdentPart(scalars[i]) { out.append(scalars[i]); i += 1 }
        return String(out)
    }

    private mutating func readInteger() -> Int {
        var out = ""
        while i < scalars.count, isDigit(scalars[i]) { out.unicodeScalars.append(scalars[i]); i += 1 }
        return Int(out) ?? 0
    }

    private mutating func readString(quote: Unicode.Scalar) throws -> String {
        i += 1
        var out = String.UnicodeScalarView()
        while i < scalars.count {
            let c = scalars[i]
            if c == quote { i += 1; return String(out) }
            if c == "\\" && i + 1 < scalars.count {
                let e = scalars[i + 1]
                switch e {
                case "n": out.append("\n")
                case "t": out.append("\t")
                case "r": out.append("\r")
                case "\\": out.append("\\")
                case "'": out.append("'")
                case "\"": out.append("\"")
                default: out.append(e)
                }
                i += 2
                continue
            }
            out.append(c)
            i += 1
        }
        throw NFKJinjaError.syntax("unterminated string")
    }

    private mutating func readOperator() -> String {
        let two: Set<String> = ["==", "!=", "<=", ">="]
        if i + 1 < scalars.count {
            let pair = String(String.UnicodeScalarView([scalars[i], scalars[i + 1]]))
            if two.contains(pair) { i += 2; return pair }
        }
        let single = String(scalars[i]); i += 1; return single
    }
}

// MARK: - Precedence-climbing parser

struct NFKJinjaExprParser {
    private let tokens: [NFKJinjaExprToken]
    private var pos = 0
    init(_ tokens: [NFKJinjaExprToken]) { self.tokens = tokens }

    private var current: NFKJinjaExprToken { tokens[pos] }
    private mutating func advance() -> NFKJinjaExprToken { defer { pos += 1 }; return tokens[pos] }

    private func isKeyword(_ word: String) -> Bool {
        if case .identifier(let name) = current { return name == word }
        return false
    }
    private func isOp(_ symbol: String) -> Bool {
        if case .op(let value) = current { return value == symbol }
        return false
    }
    private mutating func consumeKeyword(_ word: String) -> Bool {
        if isKeyword(word) { pos += 1; return true }
        return false
    }
    private mutating func expectOp(_ symbol: String) throws {
        guard isOp(symbol) else { throw NFKJinjaError.syntax("expected '\(symbol)'") }
        pos += 1
    }

    mutating func expectEnd() throws {
        guard case .end = current else { throw NFKJinjaError.syntax("trailing tokens in expression") }
    }

    mutating func parseExpression() throws -> NFKJinjaExpr { try parseTernary() }

    private mutating func parseTernary() throws -> NFKJinjaExpr {
        let value = try parseOr()
        if consumeKeyword("if") {
            let condition = try parseOr()
            guard consumeKeyword("else") else { throw NFKJinjaError.syntax("ternary without else") }
            let otherwise = try parseTernary()
            return .ternary(value, condition, otherwise)
        }
        return value
    }

    private mutating func parseOr() throws -> NFKJinjaExpr {
        var left = try parseAnd()
        while consumeKeyword("or") { left = .binary("or", left, try parseAnd()) }
        return left
    }

    private mutating func parseAnd() throws -> NFKJinjaExpr {
        var left = try parseNot()
        while consumeKeyword("and") { left = .binary("and", left, try parseNot()) }
        return left
    }

    private mutating func parseNot() throws -> NFKJinjaExpr {
        if consumeKeyword("not") { return .unary("not", try parseNot()) }
        return try parseComparison()
    }

    private mutating func parseComparison() throws -> NFKJinjaExpr {
        var left = try parseAdditive()
        while true {
            if isOp("==") || isOp("!=") || isOp("<") || isOp(">") || isOp("<=") || isOp(">=") {
                guard case .op(let symbol) = advance() else { break }
                left = .binary(symbol, left, try parseAdditive())
            } else if isKeyword("in") {
                pos += 1
                left = .binary("in", left, try parseAdditive())
            } else if isKeyword("not") && peekKeyword(1, "in") {
                pos += 2
                left = .binary("not in", left, try parseAdditive())
            } else if consumeKeyword("is") {
                let negated = consumeKeyword("not")
                guard case .identifier(let name) = advance() else {
                    throw NFKJinjaError.syntax("expected test name after 'is'")
                }
                var argument: NFKJinjaExpr? = nil
                if isOp("(") { pos += 1; argument = try parseExpression(); try expectOp(")") }
                left = .test(left, name, negated, argument)
            } else {
                break
            }
        }
        return left
    }

    private func peekKeyword(_ ahead: Int, _ word: String) -> Bool {
        let index = pos + ahead
        guard index < tokens.count, case .identifier(let name) = tokens[index] else { return false }
        return name == word
    }

    private mutating func parseAdditive() throws -> NFKJinjaExpr {
        var left = try parseMultiplicative()
        while isOp("+") || isOp("-") || isOp("~") {
            guard case .op(let symbol) = advance() else { break }
            left = .binary(symbol, left, try parseMultiplicative())
        }
        return left
    }

    private mutating func parseMultiplicative() throws -> NFKJinjaExpr {
        var left = try parseFilter()
        while isOp("*") || isOp("/") || isOp("%") {
            guard case .op(let symbol) = advance() else { break }
            left = .binary(symbol, left, try parseFilter())
        }
        return left
    }

    private mutating func parseFilter() throws -> NFKJinjaExpr {
        var left = try parseUnary()
        while isOp("|") {
            pos += 1
            guard case .identifier(let name) = advance() else {
                throw NFKJinjaError.syntax("expected filter name")
            }
            var arguments: [NFKJinjaExpr] = []
            if isOp("(") {
                pos += 1
                if !isOp(")") {
                    repeat { arguments.append(try parseExpression()) } while consumeOp(",")
                }
                try expectOp(")")
            }
            left = .filter(left, name, arguments)
        }
        return left
    }

    private mutating func consumeOp(_ symbol: String) -> Bool {
        if isOp(symbol) { pos += 1; return true }
        return false
    }

    private mutating func parseUnary() throws -> NFKJinjaExpr {
        if isOp("-") { pos += 1; return .unary("-", try parseUnary()) }
        return try parsePostfix()
    }

    private mutating func parsePostfix() throws -> NFKJinjaExpr {
        var expr = try parsePrimary()
        while true {
            if isOp(".") {
                pos += 1
                guard case .identifier(let name) = advance() else {
                    throw NFKJinjaError.syntax("expected attribute name")
                }
                expr = .attribute(expr, name)
            } else if isOp("[") {
                pos += 1
                expr = try parseSubscript(expr)
            } else if isOp("(") {
                pos += 1
                expr = try parseCall(expr)
            } else {
                break
            }
        }
        return expr
    }

    private mutating func parseSubscript(_ base: NFKJinjaExpr) throws -> NFKJinjaExpr {
        // Either base[expr] or a slice base[start?:stop?:step?].
        var start: NFKJinjaExpr? = nil
        var stop: NFKJinjaExpr? = nil
        var step: NFKJinjaExpr? = nil
        var isSlice = false
        if !isOp(":") && !isOp("]") { start = try parseExpression() }
        if isOp(":") {
            isSlice = true
            pos += 1
            if !isOp(":") && !isOp("]") { stop = try parseExpression() }
            if isOp(":") {
                pos += 1
                if !isOp("]") { step = try parseExpression() }
            }
        }
        try expectOp("]")
        if isSlice { return .slice(base, start, stop, step) }
        guard let index = start else { throw NFKJinjaError.syntax("empty subscript") }
        return .index(base, index)
    }

    private mutating func parseCall(_ callee: NFKJinjaExpr) throws -> NFKJinjaExpr {
        var positional: [NFKJinjaExpr] = []
        var keyword: [(String, NFKJinjaExpr)] = []
        if !isOp(")") {
            repeat {
                if case .identifier(let name) = current, case .op("=") = tokens[pos + 1] {
                    pos += 2
                    keyword.append((name, try parseExpression()))
                } else {
                    positional.append(try parseExpression())
                }
            } while consumeOp(",")
        }
        try expectOp(")")
        return .call(callee, positional, keyword)
    }

    private mutating func parsePrimary() throws -> NFKJinjaExpr {
        switch current {
        case .integer(let value): pos += 1; return .intLiteral(value)
        case .string(let value): pos += 1; return .stringLiteral(value)
        case .identifier(let name):
            pos += 1
            switch name {
            case "true", "True": return .boolLiteral(true)
            case "false", "False": return .boolLiteral(false)
            case "none", "None": return .noneLiteral
            default: return .variable(name)
            }
        case .op("("):
            pos += 1
            let expr = try parseExpression()
            try expectOp(")")
            return expr
        case .op("["):
            pos += 1
            var items: [NFKJinjaExpr] = []
            if !isOp("]") {
                repeat { items.append(try parseExpression()) } while consumeOp(",")
            }
            try expectOp("]")
            return .listLiteral(items)
        default:
            throw NFKJinjaError.syntax("unexpected token in expression")
        }
    }
}

// MARK: - Evaluator

struct NFKJinjaEvaluator {
    private var context: [String: NFKJinjaValue]

    init(context: [String: NFKJinjaValue]) { self.context = context }

    mutating func run(_ nodes: [NFKJinjaNode], into output: inout String) throws {
        for node in nodes { try run(node, into: &output) }
    }

    private mutating func run(_ node: NFKJinjaNode, into output: inout String) throws {
        switch node {
        case .text(let value):
            output += value
        case .output(let expr):
            output += try evaluate(expr).asString
        case .set(let target, let valueExpr):
            let value = try evaluate(valueExpr)
            switch target {
            case .name(let name):
                context[name] = value
            case .attribute(let base, let attr):
                if case .namespace(let object)? = context[base] {
                    object.attributes[attr] = value
                }
            }
        case .conditional(let branches):
            for (condition, body) in branches {
                if let condition = condition {
                    if try evaluate(condition).isTruthy {
                        try run(body, into: &output)
                        return
                    }
                } else {
                    try run(body, into: &output)
                    return
                }
            }
        case .forLoop(let varName, let seqExpr, let body):
            let sequence = try iterable(evaluate(seqExpr))
            let savedItem = context[varName]
            let savedLoop = context["loop"]
            let count = sequence.count
            for (position, element) in sequence.enumerated() {
                context[varName] = element
                context["loop"] = .dictionary([
                    "index0": .int(position),
                    "index": .int(position + 1),
                    "first": .bool(position == 0),
                    "last": .bool(position == count - 1),
                    "length": .int(count),
                ])
                try run(body, into: &output)
            }
            context[varName] = savedItem
            context["loop"] = savedLoop
        }
    }

    private func iterable(_ value: NFKJinjaValue) throws -> [NFKJinjaValue] {
        switch value {
        case .list(let items): return items
        case .string(let string): return string.map { .string(String($0)) }
        case .dictionary(let map): return map.keys.sorted().map { .string($0) }
        case .undefined, .none: return []
        default: throw NFKJinjaError.syntax("value is not iterable")
        }
    }

    // MARK: Expression evaluation

    private func evaluate(_ expr: NFKJinjaExpr) throws -> NFKJinjaValue {
        switch expr {
        case .stringLiteral(let value): return .string(value)
        case .intLiteral(let value): return .int(value)
        case .boolLiteral(let value): return .bool(value)
        case .noneLiteral: return .none
        case .listLiteral(let items): return .list(try items.map(evaluate))
        case .variable(let name): return context[name] ?? .undefined
        case .attribute(let base, let name): return try attribute(evaluate(base), name)
        case .index(let base, let indexExpr): return try index(evaluate(base), evaluate(indexExpr))
        case .slice(let base, let start, let stop, let step):
            return try slice(evaluate(base),
                             start.map(evaluate), stop.map(evaluate), step.map(evaluate))
        case .call(let callee, let args, let kwargs): return try call(callee, args, kwargs)
        case .filter(let base, let name, let args): return try applyFilter(evaluate(base), name, args)
        case .unary(let op, let operand): return try unary(op, evaluate(operand))
        case .binary(let op, let left, let right): return try binary(op, left, right)
        case .test(let base, let name, let negated, let arg): return try applyTest(base, name, negated, arg)
        case .ternary(let value, let condition, let otherwise):
            return try evaluate(condition).isTruthy ? evaluate(value) : evaluate(otherwise)
        }
    }

    private func attribute(_ value: NFKJinjaValue, _ name: String) throws -> NFKJinjaValue {
        switch value {
        case .dictionary(let map): return map[name] ?? .undefined
        case .namespace(let object): return object.attributes[name] ?? .undefined
        default: return .undefined
        }
    }

    private func index(_ value: NFKJinjaValue, _ indexValue: NFKJinjaValue) throws -> NFKJinjaValue {
        switch value {
        case .list(let items):
            guard case .int(let raw) = indexValue else { return .undefined }
            let idx = raw < 0 ? items.count + raw : raw
            return (idx >= 0 && idx < items.count) ? items[idx] : .undefined
        case .dictionary(let map):
            if case .string(let key) = indexValue { return map[key] ?? .undefined }
            return .undefined
        case .string(let string):
            let chars = Array(string)
            guard case .int(let raw) = indexValue else { return .undefined }
            let idx = raw < 0 ? chars.count + raw : raw
            return (idx >= 0 && idx < chars.count) ? .string(String(chars[idx])) : .undefined
        default:
            return .undefined
        }
    }

    private func slice(_ value: NFKJinjaValue, _ start: NFKJinjaValue?, _ stop: NFKJinjaValue?,
                       _ step: NFKJinjaValue?) throws -> NFKJinjaValue {
        func asInt(_ v: NFKJinjaValue?) -> Int? {
            if case .int(let value)? = v { return value }
            return nil
        }
        let stepValue = asInt(step) ?? 1
        switch value {
        case .list(let items):
            return .list(pythonSlice(items, asInt(start), asInt(stop), stepValue))
        case .string(let string):
            let chars = string.map { String($0) }
            return .string(pythonSlice(chars, asInt(start), asInt(stop), stepValue).joined())
        default:
            return .undefined
        }
    }

    private func pythonSlice<T>(_ items: [T], _ start: Int?, _ stop: Int?, _ step: Int) -> [T] {
        let n = items.count
        if step == 0 { return [] }
        func clampStart(_ value: Int?) -> Int {
            if step > 0 {
                guard var s = value else { return 0 }
                if s < 0 { s += n }
                return min(max(s, 0), n)
            } else {
                guard var s = value else { return n - 1 }
                if s < 0 { s += n }
                return min(max(s, -1), n - 1)
            }
        }
        func clampStop(_ value: Int?) -> Int {
            if step > 0 {
                guard var s = value else { return n }
                if s < 0 { s += n }
                return min(max(s, 0), n)
            } else {
                guard var s = value else { return -1 }
                if s < 0 { s += n }
                return min(max(s, -1), n - 1)
            }
        }
        var result: [T] = []
        var i = clampStart(start)
        let end = clampStop(stop)
        if step > 0 {
            while i < end { result.append(items[i]); i += step }
        } else {
            while i > end { result.append(items[i]); i += step }
        }
        return result
    }

    private func unary(_ op: String, _ value: NFKJinjaValue) throws -> NFKJinjaValue {
        switch op {
        case "not": return .bool(!value.isTruthy)
        case "-":
            if case .int(let i) = value { return .int(-i) }
            return .undefined
        default: return .undefined
        }
    }

    private func binary(_ op: String, _ leftExpr: NFKJinjaExpr, _ rightExpr: NFKJinjaExpr) throws -> NFKJinjaValue {
        if op == "and" {
            let left = try evaluate(leftExpr)
            return left.isTruthy ? try evaluate(rightExpr) : left
        }
        if op == "or" {
            let left = try evaluate(leftExpr)
            return left.isTruthy ? left : try evaluate(rightExpr)
        }
        let left = try evaluate(leftExpr)
        let right = try evaluate(rightExpr)
        switch op {
        case "+":
            if case .int(let a) = left, case .int(let b) = right { return .int(a + b) }
            return .string(left.asString + right.asString)
        case "-":
            if case .int(let a) = left, case .int(let b) = right { return .int(a - b) }
            return .undefined
        case "*":
            if case .int(let a) = left, case .int(let b) = right { return .int(a * b) }
            return .undefined
        case "/":
            if case .int(let a) = left, case .int(let b) = right, b != 0 { return .int(a / b) }
            return .undefined
        case "%":
            if case .int(let a) = left, case .int(let b) = right, b != 0 { return .int(a % b) }
            return .undefined
        case "~":
            return .string(left.asString + right.asString)
        case "==": return .bool(equals(left, right))
        case "!=": return .bool(!equals(left, right))
        case "<", ">", "<=", ">=": return .bool(compare(op, left, right))
        case "in": return .bool(contains(right, left))
        case "not in": return .bool(!contains(right, left))
        default: return .undefined
        }
    }

    private func equals(_ left: NFKJinjaValue, _ right: NFKJinjaValue) -> Bool {
        switch (left, right) {
        case (.string(let a), .string(let b)): return a == b
        case (.int(let a), .int(let b)): return a == b
        case (.bool(let a), .bool(let b)): return a == b
        case (.none, .none), (.undefined, .undefined): return true
        case (.none, .undefined), (.undefined, .none): return true
        default: return false
        }
    }

    private func compare(_ op: String, _ left: NFKJinjaValue, _ right: NFKJinjaValue) -> Bool {
        guard case .int(let a) = left, case .int(let b) = right else { return false }
        switch op {
        case "<": return a < b
        case ">": return a > b
        case "<=": return a <= b
        case ">=": return a >= b
        default: return false
        }
    }

    private func contains(_ container: NFKJinjaValue, _ needle: NFKJinjaValue) -> Bool {
        switch container {
        case .string(let haystack):
            if case .string(let sub) = needle { return sub.isEmpty || haystack.contains(sub) }
            return false
        case .list(let items): return items.contains { equals($0, needle) }
        case .dictionary(let map):
            if case .string(let key) = needle { return map[key] != nil }
            return false
        default: return false
        }
    }

    private func applyTest(_ baseExpr: NFKJinjaExpr, _ name: String, _ negated: Bool,
                           _ argExpr: NFKJinjaExpr?) throws -> NFKJinjaValue {
        // `defined` must not error on an undefined base, so evaluate here rather than eagerly.
        let base = try evaluate(baseExpr)
        var result: Bool
        switch name {
        case "defined":
            if case .undefined = base { result = false } else { result = true }
        case "undefined":
            if case .undefined = base { result = true } else { result = false }
        case "none", "null":
            if case .none = base { result = true } else { result = false }
        case "string":
            if case .string = base { result = true } else { result = false }
        case "number", "integer":
            if case .int = base { result = true } else { result = false }
        case "mapping":
            if case .dictionary = base { result = true } else { result = false }
        case "iterable":
            switch base { case .list, .string, .dictionary: result = true; default: result = false }
        case "true":
            result = base.isTruthy && { if case .bool(true) = base { return true }; return false }()
        case "false":
            if case .bool(false) = base { result = true } else { result = false }
        case "equalto", "eq":
            if let argExpr = argExpr { result = equals(base, try evaluate(argExpr)) } else { result = false }
        default:
            result = false
        }
        return .bool(negated ? !result : result)
    }

    private func applyFilter(_ value: NFKJinjaValue, _ name: String,
                             _ argExprs: [NFKJinjaExpr]) throws -> NFKJinjaValue {
        switch name {
        case "trim":
            return .string(value.asString.trimmingCharacters(in: .whitespacesAndNewlines))
        case "tojson":
            return .string(NFKJinjaJSON.encode(value))
        case "length", "count":
            switch value {
            case .list(let items): return .int(items.count)
            case .string(let string): return .int(string.count)
            case .dictionary(let map): return .int(map.count)
            default: return .int(0)
            }
        case "upper": return .string(value.asString.uppercased())
        case "lower": return .string(value.asString.lowercased())
        case "first":
            if case .list(let items) = value { return items.first ?? .undefined }
            return .undefined
        case "last":
            if case .list(let items) = value { return items.last ?? .undefined }
            return .undefined
        case "default":
            if case .undefined = value { return try argExprs.first.map(evaluate) ?? .undefined }
            return value
        case "string":
            return .string(value.asString)
        default:
            return value
        }
    }

    private func call(_ calleeExpr: NFKJinjaExpr, _ argExprs: [NFKJinjaExpr],
                      _ kwargExprs: [(String, NFKJinjaExpr)]) throws -> NFKJinjaValue {
        // A bare `namespace(...)` builds a mutable object from keyword arguments.
        if case .variable("namespace") = calleeExpr {
            var attributes: [String: NFKJinjaValue] = [:]
            for (name, expr) in kwargExprs { attributes[name] = try evaluate(expr) }
            return .namespace(NFKJinjaNamespace(attributes))
        }
        // A template guard raises on an unsupported conversation; surface it as an error.
        if case .variable(let name) = calleeExpr, name == "raise_exception" || name == "raise" {
            let message = try argExprs.first.map { try evaluate($0).asString } ?? "template raised"
            throw NFKJinjaError.syntax(message)
        }
        // Everything else is a method call on a string.
        guard case .attribute(let receiverExpr, let method) = calleeExpr else {
            throw NFKJinjaError.syntax("unsupported call")
        }
        let receiver = try evaluate(receiverExpr)
        let args = try argExprs.map(evaluate)
        return try stringMethod(receiver.asString, method, args)
    }

    private func stringMethod(_ string: String, _ method: String,
                              _ args: [NFKJinjaValue]) throws -> NFKJinjaValue {
        func arg(_ i: Int) -> String? {
            guard i < args.count, case .string(let value) = args[i] else { return nil }
            return value
        }
        switch method {
        case "startswith": return .bool(arg(0).map(string.hasPrefix) ?? false)
        case "endswith": return .bool(arg(0).map(string.hasSuffix) ?? false)
        case "strip": return .string(stripCharacters(string, arg(0), leading: true, trailing: true))
        case "lstrip": return .string(stripCharacters(string, arg(0), leading: true, trailing: false))
        case "rstrip": return .string(stripCharacters(string, arg(0), leading: false, trailing: true))
        case "split":
            if let separator = arg(0) {
                return .list(string.components(separatedBy: separator).map { .string($0) })
            }
            return .list(string.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" })
                .map { .string(String($0)) })
        case "upper": return .string(string.uppercased())
        case "lower": return .string(string.lowercased())
        case "replace":
            if let target = arg(0), let replacement = arg(1) {
                return .string(string.replacingOccurrences(of: target, with: replacement))
            }
            return .string(string)
        default:
            throw NFKJinjaError.syntax("unknown string method '\(method)'")
        }
    }

    private func stripCharacters(_ string: String, _ characters: String?,
                                 leading: Bool, trailing: Bool) -> String {
        let set: (Character) -> Bool
        if let characters = characters {
            let scalarSet = Set(characters)
            set = { scalarSet.contains($0) }
        } else {
            set = { $0 == " " || $0 == "\n" || $0 == "\t" || $0 == "\r" }
        }
        var view = Substring(string)
        if leading { while let first = view.first, set(first) { view = view.dropFirst() } }
        if trailing { while let last = view.last, set(last) { view = view.dropLast() } }
        return String(view)
    }
}
