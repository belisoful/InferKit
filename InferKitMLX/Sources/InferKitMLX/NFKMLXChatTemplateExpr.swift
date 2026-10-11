//
//  NFKMLXChatTemplateExpr.swift
//  InferKitMLX
//
//  The expression sublanguage for `NFKMLXChatTemplate`: a tokenizer, a precedence-climbing parser,
//  and the evaluator. See `NFKMLXChatTemplate.swift` for the rationale. Values follow Python, which
//  transformers' Jinja runs on: `/` divides exactly, `%` takes the divisor's sign, and a printed list
//  or dictionary is its repr.
//

import Foundation

// MARK: - Expression tokens

enum NFKJinjaExprToken: Equatable {
    case identifier(String)
    case integer(Int)
    case double(Double)
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
        if isDigit(c) { return readNumber() }
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

    private mutating func readNumber() -> NFKJinjaExprToken {
        var out = ""
        while i < scalars.count, isDigit(scalars[i]) || scalars[i] == "_" {
            if scalars[i] != "_" { out.unicodeScalars.append(scalars[i]) }
            i += 1
        }
        if i + 1 < scalars.count, scalars[i] == ".", isDigit(scalars[i + 1]) {
            out += "."
            i += 1
            while i < scalars.count, isDigit(scalars[i]) { out.unicodeScalars.append(scalars[i]); i += 1 }
            return .double(Double(out) ?? 0)
        }
        return .integer(Int(out) ?? 0)
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
                default: out.append("\\"); out.append(e)
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
        let two: Set<String> = ["==", "!=", "<=", ">=", "//", "**"]
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
    mutating func consumeKeywordIf(_ word: String) throws -> Bool { consumeKeyword(word) }
    private mutating func expectOp(_ symbol: String) throws {
        guard isOp(symbol) else { throw NFKJinjaError.syntax("expected '\(symbol)'") }
        pos += 1
    }

    mutating func expectEnd() throws {
        guard case .end = current else { throw NFKJinjaError.syntax("trailing tokens in expression") }
    }

    /// An expression; a `for` sequence parses without the conditional, whose `if` filters the loop.
    mutating func parseExpression(allowingConditional: Bool = true) throws -> NFKJinjaExpr {
        allowingConditional ? try parseTernary() : try parseOr()
    }

    /// `(name, name=default, …)`, a macro's parameters.
    mutating func parseParameterList() throws -> [(String, NFKJinjaExpr?)] {
        try expectOp("(")
        var parameters: [(String, NFKJinjaExpr?)] = []
        if !isOp(")") {
            repeat {
                guard case .identifier(let name) = advance() else { throw NFKJinjaError.syntax("expected parameter name") }
                parameters.append((name, consumeOp("=") ? try parseExpression() : nil))
            } while consumeOp(",")
        }
        try expectOp(")")
        return parameters
    }

    private mutating func parseTernary() throws -> NFKJinjaExpr {
        let value = try parseOr()
        if consumeKeyword("if") {
            let condition = try parseOr()
            let otherwise = consumeKeyword("else") ? try parseTernary() : nil
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
                var arguments: [NFKJinjaExpr] = []
                if isOp("(") {
                    pos += 1
                    if !isOp(")") {
                        repeat { arguments.append(try parseExpression()) } while consumeOp(",")
                    }
                    try expectOp(")")
                } else if startsOperand {
                    // `is equalto 'x'` and `is divisibleby 3` take one argument without parentheses.
                    arguments.append(try parseAdditive())
                }
                left = .test(left, name, negated, arguments)
            } else {
                break
            }
        }
        return left
    }

    /// Whether the current token begins an operand rather than continuing the expression around it.
    private var startsOperand: Bool {
        switch current {
        case .string, .integer, .double: return true
        case .identifier(let name):
            return !["and", "or", "if", "else", "not", "in", "is"].contains(name)
        default: return false
        }
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
        var left = try parsePower()
        while isOp("*") || isOp("/") || isOp("//") || isOp("%") {
            guard case .op(let symbol) = advance() else { break }
            left = .binary(symbol, left, try parsePower())
        }
        return left
    }

    private mutating func parsePower() throws -> NFKJinjaExpr {
        let left = try parseFilter()
        if consumeOp("**") { return .binary("**", left, try parsePower()) }
        return left
    }

    private mutating func parseFilter() throws -> NFKJinjaExpr {
        var left = try parseUnary()
        while isOp("|") {
            pos += 1
            guard case .identifier(let name) = advance() else {
                throw NFKJinjaError.syntax("expected filter name")
            }
            var positional: [NFKJinjaExpr] = []
            var keyword: [(String, NFKJinjaExpr)] = []
            if isOp("(") {
                pos += 1
                (positional, keyword) = try parseArguments()
            }
            left = .filter(left, name, positional, keyword)
        }
        return left
    }

    private mutating func consumeOp(_ symbol: String) -> Bool {
        if isOp(symbol) { pos += 1; return true }
        return false
    }

    private mutating func parseUnary() throws -> NFKJinjaExpr {
        if isOp("-") { pos += 1; return .unary("-", try parseUnary()) }
        if isOp("+") { pos += 1; return try parseUnary() }
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
                let (positional, keyword) = try parseArguments()
                expr = .call(expr, positional, keyword)
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

    /// The arguments of a call or a filter after its `(`, through the closing `)`.
    private mutating func parseArguments() throws -> ([NFKJinjaExpr], [(String, NFKJinjaExpr)]) {
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
            } while consumeOp(",") && !isOp(")")
        }
        try expectOp(")")
        return (positional, keyword)
    }

    private mutating func parsePrimary() throws -> NFKJinjaExpr {
        switch current {
        case .integer(let value): pos += 1; return .intLiteral(value)
        case .double(let value): pos += 1; return .doubleLiteral(value)
        case .string(var value):
            pos += 1
            // Adjacent string literals concatenate, as in Python.
            while case .string(let next) = current { value += next; pos += 1 }
            return .stringLiteral(value)
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
            if consumeOp(")") { return .listLiteral([]) }
            let first = try parseExpression()
            if consumeOp(",") {
                // A tuple, which a template reads as a list.
                var items = [first]
                while !isOp(")") {
                    items.append(try parseExpression())
                    if !consumeOp(",") { break }
                }
                try expectOp(")")
                return .listLiteral(items)
            }
            try expectOp(")")
            return first
        case .op("["):
            pos += 1
            var items: [NFKJinjaExpr] = []
            if !isOp("]") {
                repeat {
                    if isOp("]") { break }
                    items.append(try parseExpression())
                } while consumeOp(",")
            }
            try expectOp("]")
            return .listLiteral(items)
        case .op("{"):
            pos += 1
            var entries: [(NFKJinjaExpr, NFKJinjaExpr)] = []
            if !isOp("}") {
                repeat {
                    if isOp("}") { break }
                    let key = try parseExpression()
                    try expectOp(":")
                    entries.append((key, try parseExpression()))
                } while consumeOp(",")
            }
            try expectOp("}")
            return .dictLiteral(entries)
        default:
            throw NFKJinjaError.syntax("unexpected token in expression")
        }
    }
}

// MARK: - Evaluator

/// How a `{% break %}` or `{% continue %}` leaves the loop body it runs in.
private enum NFKJinjaLoopSignal: Error {
    case breakLoop
    case continueLoop
}

/// Runs a parsed template. Scopes follow Jinja: a loop iteration and a macro call each open one, an
/// assignment writes the innermost, and nothing set inside a loop or a macro is seen after it. A
/// namespace is shared by reference, which is how a template carries a value out of a loop.
final class NFKJinjaEvaluator {
    private var scopes: [[String: NFKJinjaValue]]
    private let date: Date

    init(globals: [String: NFKJinjaValue], date: Date) {
        self.scopes = [globals]
        self.date = date
    }

    func run(_ nodes: [NFKJinjaNode], into output: inout String) throws {
        for node in nodes { try run(node, into: &output) }
    }

    private func lookup(_ name: String) -> NFKJinjaValue? {
        for scope in scopes.reversed() {
            if let value = scope[name] { return value }
        }
        return nil
    }

    private func assign(_ name: String, _ value: NFKJinjaValue) {
        scopes[scopes.count - 1][name] = value
    }

    private func run(_ node: NFKJinjaNode, into output: inout String) throws {
        switch node {
        case .text(let value):
            output += value
        case .output(let expr):
            output += try evaluate(expr).asString
        case .set(let target, let valueExpr):
            let value = try evaluate(valueExpr)
            switch target {
            case .name(let name):
                assign(name, value)
            case .attribute(let base, let attr):
                if case .namespace(let object)? = lookup(base) {
                    object.attributes[attr] = value
                }
            }
        case .setBlock(let name, let body):
            var captured = ""
            try run(body, into: &captured)
            assign(name, .string(captured))
        case .macro(let macro):
            assign(macro.name, .macro(macro))
        case .breakLoop:
            throw NFKJinjaLoopSignal.breakLoop
        case .continueLoop:
            throw NFKJinjaLoopSignal.continueLoop
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
        case .forLoop(let loop):
            try runLoop(loop, into: &output)
        }
    }

    private func runLoop(_ loop: NFKJinjaLoop, into output: inout String) throws {
        var elements = try iterable(evaluate(loop.sequence))
        if let condition = loop.condition {
            elements = try elements.filter { element in
                scopes.append(try bindings(of: loop.targets, to: element))
                defer { scopes.removeLast() }
                return try evaluate(condition).isTruthy
            }
        }
        guard !elements.isEmpty else {
            try run(loop.otherwise, into: &output)
            return
        }
        let count = elements.count
        for (position, element) in elements.enumerated() {
            var scope = try bindings(of: loop.targets, to: element)
            var state: [String: NFKJinjaValue] = [
                "index0": .int(position), "index": .int(position + 1),
                "revindex0": .int(count - position - 1), "revindex": .int(count - position),
                "first": .bool(position == 0), "last": .bool(position == count - 1),
                "length": .int(count), "depth": .int(1), "depth0": .int(0),
            ]
            if position > 0 { state["previtem"] = elements[position - 1] }
            if position < count - 1 { state["nextitem"] = elements[position + 1] }
            scope["loop"] = .dictionary(NFKJinjaMap(state))
            scopes.append(scope)
            defer { scopes.removeLast() }
            do {
                try run(loop.body, into: &output)
            } catch NFKJinjaLoopSignal.continueLoop {
                continue
            } catch NFKJinjaLoopSignal.breakLoop {
                break
            }
        }
    }

    private func bindings(of targets: [String], to element: NFKJinjaValue) throws -> [String: NFKJinjaValue] {
        guard targets.count > 1 else { return [targets.first ?? "_": element] }
        guard case .list(let parts) = element, parts.count == targets.count else {
            throw NFKJinjaError.syntax("cannot unpack \(NFKJinjaPython.repr(element)) into \(targets.count) names")
        }
        return Dictionary(uniqueKeysWithValues: zip(targets, parts))
    }

    private func iterable(_ value: NFKJinjaValue) throws -> [NFKJinjaValue] {
        switch value {
        case .list(let items): return items
        case .string(let string): return string.map { .string(String($0)) }
        case .dictionary(let map): return map.keys.map { .string($0) }
        case .undefined, .none: return []
        default: throw NFKJinjaError.syntax("value is not iterable")
        }
    }

    // MARK: Expression evaluation

    private func evaluate(_ expr: NFKJinjaExpr) throws -> NFKJinjaValue {
        switch expr {
        case .stringLiteral(let value): return .string(value)
        case .intLiteral(let value): return .int(value)
        case .doubleLiteral(let value): return .double(value)
        case .boolLiteral(let value): return .bool(value)
        case .noneLiteral: return .none
        case .listLiteral(let items): return .list(try items.map(evaluate))
        case .dictLiteral(let entries):
            return .dictionary(NFKJinjaMap(pairs: try entries.map { (try evaluate($0.0).asString, try evaluate($0.1)) }))
        case .variable(let name): return lookup(name) ?? .undefined
        case .attribute(let base, let name): return attribute(try evaluate(base), name)
        case .index(let base, let indexExpr): return index(try evaluate(base), try evaluate(indexExpr))
        case .slice(let base, let start, let stop, let step):
            return slice(try evaluate(base), try start.map(evaluate), try stop.map(evaluate), try step.map(evaluate))
        case .call(let callee, let args, let kwargs): return try call(callee, args, kwargs)
        case .filter(let base, let name, let args, let kwargs):
            return try applyFilter(try evaluate(base), name, try args.map(evaluate), try evaluateKeywords(kwargs))
        case .unary(let op, let operand): return unary(op, try evaluate(operand))
        case .binary(let op, let left, let right): return try binary(op, left, right)
        case .test(let base, let name, let negated, let args):
            let result = try applyTest(try evaluate(base), name, try args.map(evaluate))
            return .bool(negated ? !result : result)
        case .ternary(let value, let condition, let otherwise):
            if try evaluate(condition).isTruthy { return try evaluate(value) }
            return try otherwise.map(evaluate) ?? .undefined
        }
    }

    private func evaluateKeywords(_ kwargs: [(String, NFKJinjaExpr)]) throws -> [String: NFKJinjaValue] {
        var values: [String: NFKJinjaValue] = [:]
        for (name, expr) in kwargs { values[name] = try evaluate(expr) }
        return values
    }

    private func attribute(_ value: NFKJinjaValue, _ name: String) -> NFKJinjaValue {
        switch value {
        case .dictionary(let map): return map[name] ?? .undefined
        case .namespace(let object): return object.attributes[name] ?? .undefined
        default: return .undefined
        }
    }

    private func index(_ value: NFKJinjaValue, _ indexValue: NFKJinjaValue) -> NFKJinjaValue {
        switch value {
        case .list(let items):
            guard let raw = indexValue.integer else { return .undefined }
            let idx = raw < 0 ? items.count + raw : raw
            return (idx >= 0 && idx < items.count) ? items[idx] : .undefined
        case .dictionary(let map):
            if case .string(let key) = indexValue { return map[key] ?? .undefined }
            return map[indexValue.asString] ?? .undefined
        case .namespace(let object):
            if case .string(let key) = indexValue { return object.attributes[key] ?? .undefined }
            return .undefined
        case .string(let string):
            let chars = Array(string)
            guard let raw = indexValue.integer else { return .undefined }
            let idx = raw < 0 ? chars.count + raw : raw
            return (idx >= 0 && idx < chars.count) ? .string(String(chars[idx])) : .undefined
        default:
            return .undefined
        }
    }

    private func slice(_ value: NFKJinjaValue, _ start: NFKJinjaValue?, _ stop: NFKJinjaValue?,
                       _ step: NFKJinjaValue?) -> NFKJinjaValue {
        let stepValue = step?.integer ?? 1
        switch value {
        case .list(let items):
            return .list(pythonSlice(items, start?.integer, stop?.integer, stepValue))
        case .string(let string):
            let chars = string.map { String($0) }
            return .string(pythonSlice(chars, start?.integer, stop?.integer, stepValue).joined())
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

    private func unary(_ op: String, _ value: NFKJinjaValue) -> NFKJinjaValue {
        switch op {
        case "not": return .bool(!value.isTruthy)
        case "-":
            switch value {
            case .int(let i): return .int(-i)
            case .double(let d): return .double(-d)
            case .bool(let b): return .int(b ? -1 : 0)
            default: return .undefined
            }
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
            if let sum = arithmetic(left, right, { $0 &+ $1 }, { $0 + $1 }) { return sum }
            if case .list(let a) = left, case .list(let b) = right { return .list(a + b) }
            return .string(left.asString + right.asString)
        case "-":
            return arithmetic(left, right, { $0 &- $1 }, { $0 - $1 }) ?? .undefined
        case "*":
            if let product = arithmetic(left, right, { $0 &* $1 }, { $0 * $1 }) { return product }
            // A sequence repeats by an integer on either side.
            for (sequence, count) in [(left, right), (right, left)] {
                guard count.isIntegral, let n = count.integer else { continue }
                if case .string(let s) = sequence { return .string(String(repeating: s, count: max(n, 0))) }
                if case .list(let items) = sequence { return .list(Array(repeating: items, count: max(n, 0)).flatMap { $0 }) }
            }
            return .undefined
        case "/":
            guard let a = left.number, let b = right.number, b != 0 else { return .undefined }
            return .double(a / b)
        case "//":
            if let a = left.integer, let b = right.integer, left.isIntegral && right.isIntegral, b != 0 {
                return .int(Int((Double(a) / Double(b)).rounded(.down)))
            }
            guard let a = left.number, let b = right.number, b != 0 else { return .undefined }
            return .double((a / b).rounded(.down))
        case "%":
            if let a = left.integer, let b = right.integer, left.isIntegral && right.isIntegral, b != 0 {
                let remainder = a % b
                return .int(remainder != 0 && (remainder < 0) != (b < 0) ? remainder + b : remainder)
            }
            guard let a = left.number, let b = right.number, b != 0 else { return .undefined }
            let remainder = a.truncatingRemainder(dividingBy: b)
            return .double(remainder != 0 && (remainder < 0) != (b < 0) ? remainder + b : remainder)
        case "**":
            if let a = left.integer, let b = right.integer, left.isIntegral && right.isIntegral, b >= 0 {
                return .int(Int(pow(Double(a), Double(b))))
            }
            guard let a = left.number, let b = right.number else { return .undefined }
            return .double(pow(a, b))
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

    /// Integer arithmetic when both sides are integral (a boolean counts as 0 or 1), otherwise
    /// floating-point when both are numbers; nil when either side is not a number.
    private func arithmetic(_ left: NFKJinjaValue, _ right: NFKJinjaValue,
                            _ integer: (Int, Int) -> Int, _ floating: (Double, Double) -> Double) -> NFKJinjaValue? {
        if left.isIntegral, right.isIntegral, let a = left.integer, let b = right.integer { return .int(integer(a, b)) }
        if let a = left.number, let b = right.number { return .double(floating(a, b)) }
        return nil
    }

    private func equals(_ left: NFKJinjaValue, _ right: NFKJinjaValue) -> Bool {
        if let a = left.number, let b = right.number { return a == b }
        switch (left, right) {
        case (.string(let a), .string(let b)): return a == b
        case (.none, .none), (.undefined, .undefined): return true
        case (.list(let a), .list(let b)):
            return a.count == b.count && zip(a, b).allSatisfy { equals($0, $1) }
        case (.dictionary(let a), .dictionary(let b)):
            return a.count == b.count && a.pairs.allSatisfy { key, value in b[key].map { equals(value, $0) } ?? false }
        case (.namespace(let a), .namespace(let b)): return a === b
        default: return false
        }
    }

    private func compare(_ op: String, _ left: NFKJinjaValue, _ right: NFKJinjaValue) -> Bool {
        let order: ComparisonResult
        if let a = left.number, let b = right.number {
            order = a < b ? .orderedAscending : (a > b ? .orderedDescending : .orderedSame)
        } else if case .string(let a) = left, case .string(let b) = right {
            order = a < b ? .orderedAscending : (a > b ? .orderedDescending : .orderedSame)
        } else {
            return false
        }
        switch op {
        case "<": return order == .orderedAscending
        case ">": return order == .orderedDescending
        case "<=": return order != .orderedDescending
        case ">=": return order != .orderedAscending
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
        case .namespace(let object):
            if case .string(let key) = needle { return object.attributes[key] != nil }
            return false
        default: return false
        }
    }

    // MARK: Tests

    private func applyTest(_ base: NFKJinjaValue, _ name: String, _ args: [NFKJinjaValue]) throws -> Bool {
        switch name {
        case "defined": if case .undefined = base { return false }; return true
        case "undefined": if case .undefined = base { return true }; return false
        case "none": if case .none = base { return true }; return false
        case "string": if case .string = base { return true }; return false
        case "number": return base.number != nil
        case "integer": if case .int = base { return true }; return false
        case "float": if case .double = base { return true }; return false
        case "boolean": if case .bool = base { return true }; return false
        case "mapping": if case .dictionary = base { return true }; return false
        case "iterable", "sequence":
            switch base { case .list, .string, .dictionary: return true; default: return false }
        case "callable": if case .macro = base { return true }; return false
        case "true": if case .bool(true) = base { return true }; return false
        case "false": if case .bool(false) = base { return true }; return false
        case "equalto", "eq", "==", "sameas": return args.first.map { equals(base, $0) } ?? false
        case "ne", "!=": return args.first.map { !equals(base, $0) } ?? false
        case "lt", "<": return args.first.map { compare("<", base, $0) } ?? false
        case "gt", ">": return args.first.map { compare(">", base, $0) } ?? false
        case "le", "<=": return args.first.map { compare("<=", base, $0) } ?? false
        case "ge", ">=": return args.first.map { compare(">=", base, $0) } ?? false
        case "in": return args.first.map { contains($0, base) } ?? false
        case "divisibleby":
            guard let a = base.integer, let b = args.first?.integer, b != 0 else { return false }
            return a % b == 0
        case "even": return base.integer.map { $0 % 2 == 0 } ?? false
        case "odd": return base.integer.map { $0 % 2 != 0 } ?? false
        case "lower": if case .string(let s) = base { return s == s.lowercased() }; return false
        case "upper": if case .string(let s) = base { return s == s.uppercased() }; return false
        case "escaped": return false
        default: throw NFKJinjaError.syntax("unknown test '\(name)'")
        }
    }

    // MARK: Filters

    private func applyFilter(_ value: NFKJinjaValue, _ name: String, _ args: [NFKJinjaValue],
                             _ kwargs: [String: NFKJinjaValue]) throws -> NFKJinjaValue {
        func argument(_ position: Int, _ keyword: String) -> NFKJinjaValue? {
            kwargs[keyword] ?? (position < args.count ? args[position] : nil)
        }
        switch name {
        case "trim":
            if case .string(let characters)? = argument(0, "chars") {
                return .string(strip(value.asString, characters, leading: true, trailing: true))
            }
            return .string(value.asString.trimmingCharacters(in: .whitespacesAndNewlines))
        case "tojson":
            // transformers' signature: tojson(ensure_ascii=False, indent=None, separators=None, sort_keys=False).
            var separators: (String, String)?
            if case .list(let pair)? = argument(2, "separators"), pair.count == 2 {
                separators = (pair[0].asString, pair[1].asString)
            }
            return .string(NFKJinjaJSON.encode(value, indent: argument(1, "indent")?.integer, separators: separators,
                                               sortsKeys: argument(3, "sort_keys")?.isTruthy ?? false))
        case "length", "count":
            switch value {
            case .list(let items): return .int(items.count)
            case .string(let string): return .int(string.count)
            case .dictionary(let map): return .int(map.count)
            default: return .int(0)
            }
        case "upper": return .string(value.asString.uppercased())
        case "lower": return .string(value.asString.lowercased())
        case "capitalize": return .string(capitalize(value.asString))
        case "title":
            return .string(value.asString.split(separator: " ", omittingEmptySubsequences: false)
                .map { capitalize(String($0)) }.joined(separator: " "))
        case "first":
            if case .list(let items) = value { return items.first ?? .undefined }
            if case .string(let s) = value { return s.first.map { .string(String($0)) } ?? .undefined }
            return .undefined
        case "last":
            if case .list(let items) = value { return items.last ?? .undefined }
            if case .string(let s) = value { return s.last.map { .string(String($0)) } ?? .undefined }
            return .undefined
        case "default", "d":
            let fallback = argument(0, "default_value") ?? .string("")
            let replacesFalsy = argument(1, "boolean")?.isTruthy ?? false
            if case .undefined = value { return fallback }
            return replacesFalsy && !value.isTruthy ? fallback : value
        case "string": return .string(value.asString)
        case "safe": return value
        case "e", "escape":
            return .string(value.asString.replacingOccurrences(of: "&", with: "&amp;")
                .replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
                .replacingOccurrences(of: "\"", with: "&#34;").replacingOccurrences(of: "'", with: "&#39;"))
        case "int":
            switch value {
            case .int: return value
            case .bool(let b): return .int(b ? 1 : 0)
            case .double(let d): return .int(Int(d))
            case .string(let s):
                let trimmed = s.trimmingCharacters(in: .whitespaces)
                if let i = Int(trimmed) { return .int(i) }
                if let d = Double(trimmed) { return .int(Int(d)) }
                return argument(0, "default") ?? .int(0)
            default: return argument(0, "default") ?? .int(0)
            }
        case "float":
            if let number = value.number { return .double(number) }
            if case .string(let s) = value, let d = Double(s.trimmingCharacters(in: .whitespaces)) { return .double(d) }
            return argument(0, "default") ?? .double(0)
        case "abs":
            switch value {
            case .int(let i): return .int(abs(i))
            case .double(let d): return .double(abs(d))
            default: return value
            }
        case "round":
            guard let number = value.number else { return value }
            let precision = argument(0, "precision")?.integer ?? 0
            let scale = pow(10, Double(precision))
            return .double((number * scale).rounded(.toNearestOrAwayFromZero) / scale)
        case "items":
            switch value {
            case .dictionary(let map): return .list(map.pairs.map { .list([.string($0.0), $0.1]) })
            case .namespace(let object):
                return .list(NFKJinjaKeyOrder.keys(of: object.attributes).map { .list([.string($0), object.attributes[$0]!]) })
            case .undefined, .none: return .list([])
            default: throw NFKJinjaError.syntax("items applies to a mapping")
            }
        case "dictsort":
            guard case .dictionary(let map) = value else { return .list([]) }
            let caseSensitive = argument(0, "case_sensitive")?.isTruthy ?? false
            let byValue = argument(1, "by")?.asString == "value"
            let reverse = argument(2, "reverse")?.isTruthy ?? false
            let pairs = map.pairs
            let sorted = stableSort(pairs) { a, b in
                byValue ? sortKey(a.1, caseSensitive: caseSensitive) < sortKey(b.1, caseSensitive: caseSensitive)
                        : sortKey(.string(a.0), caseSensitive: caseSensitive) < sortKey(.string(b.0), caseSensitive: caseSensitive)
            }
            return .list((reverse ? sorted.reversed() : sorted).map { .list([.string($0.0), $0.1]) })
        case "list":
            return .list(try iterable(value))
        case "join":
            let separator = argument(0, "d")?.asString ?? ""
            var items = try iterable(value)
            if case .string(let key)? = argument(1, "attribute") { items = items.map { attribute($0, key) } }
            return .string(items.map(\.asString).joined(separator: separator))
        case "replace":
            guard let old = argument(0, "old")?.asString, let new = argument(1, "new")?.asString else { return value }
            if let count = argument(2, "count")?.integer {
                var result = value.asString
                var replaced = 0
                var searchStart = result.startIndex
                while replaced < count, let range = result.range(of: old, range: searchStart..<result.endIndex), !old.isEmpty {
                    result.replaceSubrange(range, with: new)
                    searchStart = result.index(range.lowerBound, offsetBy: new.count)
                    replaced += 1
                }
                return .string(result)
            }
            return .string(value.asString.replacingOccurrences(of: old, with: new))
        case "map":
            let items = try iterable(value)
            if case .string(let key)? = kwargs["attribute"] {
                let fallback = kwargs["default"]
                return .list(items.map { item in
                    let found = attribute(item, key)
                    if case .undefined = found, let fallback { return fallback }
                    return found
                })
            }
            guard case .string(let filterName)? = args.first else { return .list(items) }
            let rest = Array(args.dropFirst())
            return .list(try items.map { try applyFilter($0, filterName, rest, kwargs) })
        case "select", "reject":
            let keep = name == "select"
            return .list(try iterable(value).filter { item in
                let passes = try args.first.map { try applyTest(item, $0.asString, Array(args.dropFirst())) } ?? item.isTruthy
                return passes == keep
            })
        case "selectattr", "rejectattr":
            let keep = name == "selectattr"
            guard let key = args.first?.asString else { return value }
            return .list(try iterable(value).filter { item in
                let field = attribute(item, key)
                let passes = args.count > 1 ? try applyTest(field, args[1].asString, Array(args.dropFirst(2))) : field.isTruthy
                return passes == keep
            })
        case "unique":
            var seen: [NFKJinjaValue] = []
            for item in try iterable(value) where !seen.contains(where: { equals($0, item) }) {
                seen.append(item)
            }
            return .list(seen)
        case "reverse":
            if case .string(let s) = value { return .string(String(s.reversed())) }
            return .list(try iterable(value).reversed())
        case "sort":
            let reverse = argument(0, "reverse")?.isTruthy ?? false
            let caseSensitive = argument(1, "case_sensitive")?.isTruthy ?? false
            let key = argument(2, "attribute")?.asString
            let sorted = stableSort(try iterable(value)) { a, b in
                let left = key.map { attribute(a, $0) } ?? a
                let right = key.map { attribute(b, $0) } ?? b
                return sortKey(left, caseSensitive: caseSensitive) < sortKey(right, caseSensitive: caseSensitive)
            }
            return .list(reverse ? sorted.reversed() : sorted)
        case "sum":
            var total: NFKJinjaValue = argument(1, "start") ?? .int(0)
            for item in try iterable(value) {
                total = arithmetic(total, item, { $0 &+ $1 }, { $0 + $1 }) ?? total
            }
            return total
        case "min", "max":
            let items = try iterable(value)
            let sorted = stableSort(items) { sortKey($0, caseSensitive: false) < sortKey($1, caseSensitive: false) }
            return (name == "min" ? sorted.first : sorted.last) ?? .undefined
        case "indent":
            let width = argument(0, "width")?.integer ?? 4
            let indentsFirst = argument(1, "first")?.isTruthy ?? false
            let pad = String(repeating: " ", count: width)
            let lines = value.asString.components(separatedBy: "\n")
            return .string(lines.enumerated().map { index, line in
                (index == 0 && !indentsFirst) || line.isEmpty ? line : pad + line
            }.joined(separator: "\n"))
        default:
            throw NFKJinjaError.syntax("unknown filter '\(name)'")
        }
    }

    private struct NFKJinjaSortKey: Comparable {
        let rank: Int
        let number: Double
        let text: String
        static func < (a: Self, b: Self) -> Bool {
            if a.rank != b.rank { return a.rank < b.rank }
            if a.number != b.number { return a.number < b.number }
            return a.text < b.text
        }
    }

    private func sortKey(_ value: NFKJinjaValue, caseSensitive: Bool) -> NFKJinjaSortKey {
        if let number = value.number { return NFKJinjaSortKey(rank: 0, number: number, text: "") }
        let text = value.asString
        return NFKJinjaSortKey(rank: 1, number: 0, text: caseSensitive ? text : text.lowercased())
    }

    private func stableSort<T>(_ items: [T], by less: (T, T) throws -> Bool) rethrows -> [T] {
        try items.enumerated().sorted { a, b in
            if try less(a.element, b.element) { return true }
            if try less(b.element, a.element) { return false }
            return a.offset < b.offset
        }.map(\.element)
    }

    private func capitalize(_ string: String) -> String {
        guard let first = string.first else { return string }
        return String(first).uppercased() + string.dropFirst().lowercased()
    }

    // MARK: Calls

    private func call(_ calleeExpr: NFKJinjaExpr, _ argExprs: [NFKJinjaExpr],
                      _ kwargExprs: [(String, NFKJinjaExpr)]) throws -> NFKJinjaValue {
        if case .attribute(let receiverExpr, let method) = calleeExpr {
            let receiver = try evaluate(receiverExpr)
            if case .macro(let macro) = attribute(receiver, method) {
                return try invoke(macro, try argExprs.map(evaluate), try evaluateKeywords(kwargExprs))
            }
            return try callMethod(receiver, method, try argExprs.map(evaluate), try evaluateKeywords(kwargExprs))
        }
        guard case .variable(let name) = calleeExpr else { throw NFKJinjaError.syntax("unsupported call") }
        if case .macro(let macro)? = lookup(name) {
            return try invoke(macro, try argExprs.map(evaluate), try evaluateKeywords(kwargExprs))
        }
        let args = try argExprs.map(evaluate)
        let kwargs = try evaluateKeywords(kwargExprs)
        switch name {
        case "namespace", "dict":
            var map = NFKJinjaMap()
            if case .dictionary(let initial)? = args.first { map = initial }
            for (key, value) in kwargs.sorted(by: { $0.key < $1.key }) { map[key] = value }
            return name == "dict" ? .dictionary(map) : .namespace(NFKJinjaNamespace(map.dictionary))
        case "raise_exception", "raise":
            // A template guard raises on an unsupported conversation; surface it as an error.
            throw NFKJinjaError.syntax(args.first?.asString ?? "template raised")
        case "range":
            let numbers = args.compactMap(\.integer)
            let (start, stop, step) = numbers.count == 1 ? (0, numbers[0], 1)
                : (numbers.first ?? 0, numbers.count > 1 ? numbers[1] : 0, numbers.count > 2 ? numbers[2] : 1)
            guard step != 0 else { throw NFKJinjaError.syntax("range step is zero") }
            return .list(stride(from: start, to: stop, by: step).map { .int($0) })
        case "strftime_now":
            return .string(NFKJinjaDate.format(date, args.first?.asString ?? ""))
        default:
            throw NFKJinjaError.syntax("unknown function '\(name)'")
        }
    }

    /// A macro runs in its own scope over the template's top level, so it reads what the template
    /// set before the call and nothing of the caller's loop.
    private func invoke(_ macro: NFKJinjaMacro, _ args: [NFKJinjaValue], _ kwargs: [String: NFKJinjaValue]) throws -> NFKJinjaValue {
        let inner = NFKJinjaEvaluator(globals: scopes[0], date: date)
        var scope: [String: NFKJinjaValue] = [macro.name: .macro(macro)]
        inner.scopes.append(scope)
        var extraKeywords = kwargs
        for (position, (name, defaultExpr)) in macro.parameters.enumerated() {
            if position < args.count {
                scope[name] = args[position]
            } else if let value = kwargs[name] {
                scope[name] = value
            } else if let defaultExpr {
                inner.scopes[1] = scope
                scope[name] = try inner.evaluate(defaultExpr)
            } else {
                scope[name] = .undefined
            }
            extraKeywords[name] = nil
        }
        scope["varargs"] = .list(Array(args.dropFirst(macro.parameters.count)))
        scope["kwargs"] = .dictionary(NFKJinjaMap(extraKeywords))
        inner.scopes[1] = scope
        var output = ""
        try inner.run(macro.body, into: &output)
        return .string(output)
    }

    private func callMethod(_ receiver: NFKJinjaValue, _ method: String, _ args: [NFKJinjaValue],
                            _ kwargs: [String: NFKJinjaValue]) throws -> NFKJinjaValue {
        switch receiver {
        case .dictionary(let map):
            switch method {
            case "get":
                guard let key = args.first?.asString else { return .none }
                return map[key] ?? (args.count > 1 ? args[1] : kwargs["default"] ?? .none)
            case "items": return .list(map.pairs.map { .list([.string($0.0), $0.1]) })
            case "keys": return .list(map.keys.map { .string($0) })
            case "values": return .list(map.values)
            default: throw NFKJinjaError.syntax("unknown mapping method '\(method)'")
            }
        case .string(let string):
            return try stringMethod(string, method, args)
        default:
            throw NFKJinjaError.syntax("unknown method '\(method)'")
        }
    }

    private func stringMethod(_ string: String, _ method: String,
                              _ args: [NFKJinjaValue]) throws -> NFKJinjaValue {
        func arg(_ i: Int) -> String? {
            guard i < args.count, case .string(let value) = args[i] else { return nil }
            return value
        }
        /// `startswith` and `endswith` take one string or a tuple of them.
        func candidates() -> [String] {
            if case .list(let items)? = args.first { return items.map(\.asString) }
            return arg(0).map { [$0] } ?? []
        }
        switch method {
        case "startswith": return .bool(candidates().contains(where: string.hasPrefix))
        case "endswith": return .bool(candidates().contains(where: string.hasSuffix))
        case "strip": return .string(strip(string, arg(0), leading: true, trailing: true))
        case "lstrip": return .string(strip(string, arg(0), leading: true, trailing: false))
        case "rstrip": return .string(strip(string, arg(0), leading: false, trailing: true))
        case "split":
            let limit = args.count > 1 ? args[1].integer ?? -1 : -1
            if let separator = arg(0) {
                var parts = string.components(separatedBy: separator)
                if limit >= 0, parts.count > limit + 1 {
                    parts = Array(parts[0..<limit]) + [parts[limit...].joined(separator: separator)]
                }
                return .list(parts.map { .string($0) })
            }
            return .list(string.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" || $0 == "\r" })
                .map { .string(String($0)) })
        case "splitlines": return .list(string.components(separatedBy: .newlines).map { .string($0) })
        case "upper": return .string(string.uppercased())
        case "lower": return .string(string.lowercased())
        case "title":
            return .string(string.split(separator: " ", omittingEmptySubsequences: false)
                .map { capitalize(String($0)) }.joined(separator: " "))
        case "capitalize": return .string(capitalize(string))
        case "replace":
            if let target = arg(0), let replacement = arg(1) {
                return .string(string.replacingOccurrences(of: target, with: replacement))
            }
            return .string(string)
        case "join":
            guard case .list(let items)? = args.first else { return .string("") }
            return .string(items.map(\.asString).joined(separator: string))
        case "count": return .int(arg(0).map { $0.isEmpty ? string.count + 1 : string.components(separatedBy: $0).count - 1 } ?? 0)
        case "find":
            guard let needle = arg(0), let range = string.range(of: needle) else { return .int(-1) }
            return .int(string.distance(from: string.startIndex, to: range.lowerBound))
        case "isdigit": return .bool(!string.isEmpty && string.allSatisfy(\.isNumber))
        case "isalpha": return .bool(!string.isEmpty && string.allSatisfy(\.isLetter))
        case "isspace": return .bool(!string.isEmpty && string.allSatisfy(\.isWhitespace))
        default:
            throw NFKJinjaError.syntax("unknown string method '\(method)'")
        }
    }

    private func strip(_ string: String, _ characters: String?, leading: Bool, trailing: Bool) -> String {
        let set: (Character) -> Bool
        if let characters = characters {
            let scalarSet = Set(characters)
            set = { scalarSet.contains($0) }
        } else {
            set = { $0.isWhitespace }
        }
        var view = Substring(string)
        if leading { while let first = view.first, set(first) { view = view.dropFirst() } }
        if trailing { while let last = view.last, set(last) { view = view.dropLast() } }
        return String(view)
    }
}

private extension NFKJinjaValue {
    /// The value as a number, a boolean as 0 or 1, as Python treats it in arithmetic.
    var number: Double? {
        switch self {
        case .int(let i): return Double(i)
        case .double(let d): return d
        case .bool(let b): return b ? 1 : 0
        default: return nil
        }
    }

    var integer: Int? {
        switch self {
        case .int(let i): return i
        case .bool(let b): return b ? 1 : 0
        case .double(let d) where d == d.rounded(): return Int(d)
        default: return nil
        }
    }

    var isIntegral: Bool {
        switch self {
        case .int, .bool: return true
        default: return false
        }
    }
}

/// `strftime` for the codes chat templates write.
enum NFKJinjaDate {
    static func format(_ date: Date, _ pattern: String) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .current
        let parts = calendar.dateComponents([.year, .month, .day, .hour, .minute, .second, .weekday], from: date)
        let posix = Locale(identifier: "en_US_POSIX")
        func named(_ format: String) -> String {
            let formatter = DateFormatter()
            formatter.locale = posix
            formatter.timeZone = .current
            formatter.dateFormat = format
            return formatter.string(from: date)
        }
        var out = ""
        var characters = Array(pattern)[...]
        while let character = characters.popFirst() {
            guard character == "%", let code = characters.popFirst() else {
                out.append(character)
                continue
            }
            switch code {
            case "Y": out += String(format: "%04d", parts.year ?? 0)
            case "y": out += String(format: "%02d", (parts.year ?? 0) % 100)
            case "m": out += String(format: "%02d", parts.month ?? 0)
            case "d": out += String(format: "%02d", parts.day ?? 0)
            case "e": out += String(format: "%2d", parts.day ?? 0)
            case "H": out += String(format: "%02d", parts.hour ?? 0)
            case "I": out += String(format: "%02d", ((parts.hour ?? 0) + 11) % 12 + 1)
            case "M": out += String(format: "%02d", parts.minute ?? 0)
            case "S": out += String(format: "%02d", parts.second ?? 0)
            case "p": out += (parts.hour ?? 0) < 12 ? "AM" : "PM"
            case "B": out += named("MMMM")
            case "b": out += named("MMM")
            case "A": out += named("EEEE")
            case "a": out += named("EEE")
            case "j": out += String(format: "%03d", calendar.ordinality(of: .day, in: .year, for: date) ?? 0)
            case "%": out += "%"
            default: out += "%" + String(code)
            }
        }
        return out
    }
}
