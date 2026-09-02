//
//  NFKMLXConstrainedDecoding.swift
//  InferKitMLX
//
//  Grammar-constrained sampling: a mask over the vocabulary, recomputed as the output grows, that
//  admits only the tokens a grammar can accept next.
//

import Foundation
import InferKit
import MLX

/// The bytes every token id decodes to, which is what a grammar reasons over.
///
/// @discussion A grammar is defined over text, a model emits token ids, and a byte-level vocabulary
/// is the bridge: a token is admissible when appending its bytes keeps the output inside the
/// grammar. The table is built once per tokenizer and shared. `size` is the model's logit width,
/// which can exceed the tokenizer's vocabulary; an id with no bytes is never admitted.
public final class NFKMLXVocabulary: @unchecked Sendable {
    /// Each token's bytes, indexed by id; empty for an id the tokenizer does not hold.
    public let tokens: [[UInt8]]
    /// The end-of-sequence id, which a grammar admits only when the output is complete, or nil.
    public let endToken: Int?

    public init(tokens: [[UInt8]], endToken: Int?) {
        self.tokens = tokens
        self.endToken = endToken
    }

    /// Reads every id below `size` from `tokenizer`.
    public convenience init(tokenizer: NFKTokenizer, size: Int, endToken: Int? = nil) {
        let bytes = (0 ..< size).map { id -> [UInt8] in
            guard let data = tokenizer.bytes(forTokenId: id) else { return [] }
            return [UInt8](data)
        }
        let end = endToken ?? (tokenizer.eosTokenId >= 0 ? tokenizer.eosTokenId : nil)
        self.init(tokens: bytes, endToken: end)
    }

    public var count: Int { tokens.count }
}

/// Something that narrows what a model may emit next.
///
/// @discussion A constraint is shared and immutable; ``makeCursor()`` returns the per-run state
/// that walks the output. The backend applies the cursor's mask to the logits before sampling and
/// tells it each token as it is chosen.
public protocol NFKMLXTokenConstraint: AnyObject, Sendable {
    func makeCursor() -> any NFKMLXConstraintCursor
}

/// One run's position inside a constraint.
public protocol NFKMLXConstraintCursor: AnyObject {
    /// An additive `[vocabulary]` mask: 0 for a token the grammar admits next, `-inf` otherwise.
    func allowedTokenMask() -> MLXArray
    /// Records the token that was emitted.
    func accept(_ token: Int)
    /// The id that ends the output, admitted only once the grammar is satisfied.
    var endToken: Int? { get }
}

/// A constraint defined byte by byte over a hashable state: the grammar advances one byte at a
/// time, so a token is admissible exactly when every one of its bytes advances.
///
/// @discussion Computing the admissible set means walking every token's bytes from the current
/// state, which is the whole vocabulary times a few bytes. The set depends only on the state, and
/// a run revisits a handful of states (inside a string, between a key and its value), so masks
/// are cached per state and most steps cost a dictionary lookup.
open class NFKMLXByteConstraint<State: Hashable>: NFKMLXTokenConstraint, @unchecked Sendable {
    public let vocabulary: NFKMLXVocabulary
    private var masks = [State: MLXArray]()
    private let lock = NSLock()

    public init(vocabulary: NFKMLXVocabulary) {
        self.vocabulary = vocabulary
    }

    /// The state before any output.
    open func initialState() -> State { fatalError("a subclass supplies the grammar") }
    /// The state after `byte`, or nil when the grammar rejects it.
    open func advance(_ state: State, byte: UInt8) -> State? { fatalError("a subclass supplies the grammar") }
    /// Whether the output may end in `state`.
    open func isComplete(_ state: State) -> Bool { fatalError("a subclass supplies the grammar") }

    /// The state after all of `bytes`, or nil when any byte is rejected.
    public func advance(_ state: State, bytes: [UInt8]) -> State? {
        var current = state
        for byte in bytes {
            guard let next = advance(current, byte: byte) else { return nil }
            current = next
        }
        return current
    }

    /// Whether `text` is a prefix the grammar can still complete.
    public func accepts(_ text: String) -> Bool {
        advance(initialState(), bytes: Array(text.utf8)) != nil
    }

    /// Whether `text` is a complete output.
    public func isComplete(_ text: String) -> Bool {
        advance(initialState(), bytes: Array(text.utf8)).map(isComplete) ?? false
    }

    /// The ids admissible from `state`.
    public func allowedTokens(from state: State) -> [Int] {
        var allowed = [Int]()
        for (id, bytes) in vocabulary.tokens.enumerated() {
            if id == vocabulary.endToken {
                if isComplete(state) { allowed.append(id) }
                continue
            }
            if !bytes.isEmpty, advance(state, bytes: bytes) != nil {
                allowed.append(id)
            }
        }
        return allowed
    }

    func mask(for state: State) -> MLXArray {
        lock.lock()
        if let cached = masks[state] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        var values = [Float](repeating: -.infinity, count: vocabulary.count)
        let allowed = allowedTokens(from: state)
        for id in allowed { values[id] = 0 }
        // No way forward at all: only the end can follow, so the run stops rather than emitting a
        // token the grammar would refuse.
        if allowed.isEmpty, let end = vocabulary.endToken { values[end] = 0 }
        let mask = MLXArray(values)
        lock.lock()
        masks[state] = mask
        lock.unlock()
        return mask
    }

    public func makeCursor() -> any NFKMLXConstraintCursor { Cursor(constraint: self) }

    final class Cursor: NFKMLXConstraintCursor {
        let constraint: NFKMLXByteConstraint<State>
        var state: State
        init(constraint: NFKMLXByteConstraint<State>) {
            self.constraint = constraint
            state = constraint.initialState()
        }
        var endToken: Int? { constraint.vocabulary.endToken }
        func allowedTokenMask() -> MLXArray { constraint.mask(for: state) }
        func accept(_ token: Int) {
            guard token != constraint.vocabulary.endToken,
                  token >= 0, token < constraint.vocabulary.count else { return }
            if let next = constraint.advance(state, bytes: constraint.vocabulary.tokens[token]) {
                state = next
            }
        }
    }
}

// MARK: - JSON

/// Where a JSON output stands: the open containers, the scalar being written, and whether the root
/// value has closed.
public struct NFKMLXJSONState: Hashable, Sendable {
    enum ObjectPhase: Hashable { case keyOrEnd, key, colon, value, commaOrEnd }
    enum ArrayPhase: Hashable { case valueOrEnd, value, commaOrEnd }
    enum Frame: Hashable { case object(ObjectPhase), array(ArrayPhase) }
    enum StringPhase: Hashable { case body, escape, unicode(Int) }
    enum NumberPhase: Hashable { case minus, zero, integer, dot, fraction, exponent, exponentSign, exponentDigits }
    enum Literal: Hashable { case `true`, `false`, null }
    enum Scalar: Hashable { case none, string(StringPhase), number(NumberPhase), literal(Literal, Int) }

    var stack: [Frame] = []
    var scalar: Scalar = .none
    var inKey = false
    var complete = false
    /// Consecutive whitespace bytes emitted between tokens.
    var whitespaceRun: UInt8 = 0
}

/// Constrains the output to well-formed JSON: an object or an array (or, when allowed, any value),
/// terminated when it closes.
///
/// @discussion This is syntax, not schema: the keys and types are the model's choice, the shape is
/// guaranteed. Whitespace between tokens is admitted as JSON admits it, strings take any UTF-8
/// and the standard escapes, and a number is refused a leading zero or a trailing dot exactly as
/// the specification refuses them. Once the root value closes, only whitespace and the end token
/// remain admissible, so a run ends where the document does.
public final class NFKMLXJSONConstraint: NFKMLXByteConstraint<NFKMLXJSONState>, @unchecked Sendable {
    /// What the document's root may be.
    public enum Root: Sendable {
        /// An object or an array.
        case container
        /// An object only.
        case object
        /// An array only.
        case array
        /// Any JSON value, scalars included.
        case any
    }

    /// What the root value may be.
    public let root: Root
    /// Whether the root may be a string, number, or literal rather than an object or array.
    public var allowsScalarRoot: Bool { root == .any }
    /// The most consecutive whitespace bytes admitted between tokens.
    ///
    /// @discussion JSON admits unbounded whitespace, and a model whose preferred next token is
    /// forbidden takes the whitespace the grammar offers instead — indefinitely, since every
    /// newline is admissible after the last. Measured on Qwen3-0.6B: with no cap the first
    /// constrained request produced 96 tokens of blank lines and no document. The cap bounds that
    /// detour to a few bytes, after which the grammar's real alternatives are all that remain.
    /// Eight bytes leave room for a newline and two levels of four-space indentation.
    public let maximumWhitespaceRun: Int

    public init(vocabulary: NFKMLXVocabulary, root: Root = .container, maximumWhitespaceRun: Int = 8) {
        self.root = root
        self.maximumWhitespaceRun = maximumWhitespaceRun
        super.init(vocabulary: vocabulary)
    }

    public override func initialState() -> NFKMLXJSONState { NFKMLXJSONState() }

    public override func isComplete(_ state: NFKMLXJSONState) -> Bool {
        if state.complete { return true }
        // A root number cannot know it has ended until something follows; at the end, a terminal
        // digit run is a complete value.
        guard allowsScalarRoot, state.stack.isEmpty, !state.inKey else { return false }
        if case .number(let phase) = state.scalar { return Self.isTerminal(phase) }
        return false
    }

    public override func advance(_ state: NFKMLXJSONState, byte: UInt8) -> NFKMLXJSONState? {
        var s = state
        // Whitespace between tokens is counted and capped; anything else resets the count. Inside
        // a string the bytes are content, not spacing, and are not counted.
        if case .string = s.scalar {
            return advanceStringOrValue(s, byte: byte)
        }
        if Self.isWhitespace(byte) {
            guard Int(s.whitespaceRun) < maximumWhitespaceRun else { return nil }
            s.whitespaceRun += 1
        } else {
            s.whitespaceRun = 0
        }
        return advanceStringOrValue(s, byte: byte)
    }

    private func advanceStringOrValue(_ state: NFKMLXJSONState, byte: UInt8) -> NFKMLXJSONState? {
        var s = state
        if s.complete { return Self.isWhitespace(byte) ? s : nil }

        switch s.scalar {
        case .string(let phase):
            return advanceString(s, phase: phase, byte: byte)
        case .number(let phase):
            if let next = Self.advanceNumber(phase, byte: byte) {
                s.scalar = .number(next)
                return s
            }
            guard Self.isTerminal(phase) else { return nil }
            s.scalar = .none
            s = Self.finished(s)
            return advanceStringOrValue(s, byte: byte)
        case .literal(let literal, let index):
            let word = Self.word(literal)
            guard byte == word[index] else { return nil }
            if index + 1 == word.count {
                s.scalar = .none
                return Self.finished(s)
            }
            s.scalar = .literal(literal, index + 1)
            return s
        case .none:
            break
        }

        if Self.isWhitespace(byte) { return s }
        guard let top = s.stack.last else {
            switch root {
            case .object where byte != UInt8(ascii: "{"), .array where byte != UInt8(ascii: "["):
                return nil
            default:
                return Self.startValue(s, byte: byte, allowScalar: allowsScalarRoot)
            }
        }
        switch top {
        case .object(let phase):
            switch (phase, byte) {
            case (.keyOrEnd, UInt8(ascii: "}")), (.commaOrEnd, UInt8(ascii: "}")):
                s.stack.removeLast()
                return Self.finished(s)
            case (.keyOrEnd, UInt8(ascii: "\"")), (.key, UInt8(ascii: "\"")):
                s.stack[s.stack.count - 1] = .object(.colon)
                s.scalar = .string(.body)
                s.inKey = true
                return s
            case (.colon, UInt8(ascii: ":")):
                s.stack[s.stack.count - 1] = .object(.value)
                return s
            case (.value, _):
                s.stack[s.stack.count - 1] = .object(.commaOrEnd)
                return Self.startValue(s, byte: byte, allowScalar: true)
            case (.commaOrEnd, UInt8(ascii: ",")):
                s.stack[s.stack.count - 1] = .object(.key)
                return s
            default:
                return nil
            }
        case .array(let phase):
            switch (phase, byte) {
            case (.valueOrEnd, UInt8(ascii: "]")), (.commaOrEnd, UInt8(ascii: "]")):
                s.stack.removeLast()
                return Self.finished(s)
            case (.valueOrEnd, _), (.value, _):
                s.stack[s.stack.count - 1] = .array(.commaOrEnd)
                return Self.startValue(s, byte: byte, allowScalar: true)
            case (.commaOrEnd, UInt8(ascii: ",")):
                s.stack[s.stack.count - 1] = .array(.value)
                return s
            default:
                return nil
            }
        }
    }

    private func advanceString(_ state: NFKMLXJSONState, phase: NFKMLXJSONState.StringPhase,
                               byte: UInt8) -> NFKMLXJSONState? {
        var s = state
        switch phase {
        case .body:
            switch byte {
            case UInt8(ascii: "\""):
                s.scalar = .none
                return Self.finished(s)
            case UInt8(ascii: "\\"):
                s.scalar = .string(.escape)
                return s
            case 0 ..< 0x20:
                return nil
            default:
                return s
            }
        case .escape:
            if byte == UInt8(ascii: "u") {
                s.scalar = .string(.unicode(0))
                return s
            }
            guard "\"\\/bfnrt".utf8.contains(byte) else { return nil }
            s.scalar = .string(.body)
            return s
        case .unicode(let digits):
            guard Self.isHexDigit(byte) else { return nil }
            s.scalar = digits + 1 == 4 ? .string(.body) : .string(.unicode(digits + 1))
            return s
        }
    }

    /// Opens the value `byte` begins, with the enclosing frame already advanced past it.
    private static func startValue(_ state: NFKMLXJSONState, byte: UInt8,
                                   allowScalar: Bool) -> NFKMLXJSONState? {
        var s = state
        switch byte {
        case UInt8(ascii: "{"):
            s.stack.append(.object(.keyOrEnd))
        case UInt8(ascii: "["):
            s.stack.append(.array(.valueOrEnd))
        case UInt8(ascii: "\"") where allowScalar:
            s.scalar = .string(.body)
        case UInt8(ascii: "-") where allowScalar:
            s.scalar = .number(.minus)
        case UInt8(ascii: "0") where allowScalar:
            s.scalar = .number(.zero)
        case UInt8(ascii: "1") ... UInt8(ascii: "9") where allowScalar:
            s.scalar = .number(.integer)
        case UInt8(ascii: "t") where allowScalar:
            s.scalar = .literal(.true, 1)
        case UInt8(ascii: "f") where allowScalar:
            s.scalar = .literal(.false, 1)
        case UInt8(ascii: "n") where allowScalar:
            s.scalar = .literal(.null, 1)
        default:
            return nil
        }
        return s
    }

    /// A value (or a key) has closed: a key waits for its colon, a root value completes the document.
    private static func finished(_ state: NFKMLXJSONState) -> NFKMLXJSONState {
        var s = state
        if s.inKey {
            s.inKey = false
            return s
        }
        if s.stack.isEmpty { s.complete = true }
        return s
    }

    private static func advanceNumber(_ phase: NFKMLXJSONState.NumberPhase,
                                      byte: UInt8) -> NFKMLXJSONState.NumberPhase? {
        let digit = isDigit(byte)
        switch phase {
        case .minus:
            if byte == UInt8(ascii: "0") { return .zero }
            return digit ? .integer : nil
        case .zero:
            if byte == UInt8(ascii: ".") { return .dot }
            return isExponent(byte) ? .exponent : nil
        case .integer:
            if digit { return .integer }
            if byte == UInt8(ascii: ".") { return .dot }
            return isExponent(byte) ? .exponent : nil
        case .dot:
            return digit ? .fraction : nil
        case .fraction:
            if digit { return .fraction }
            return isExponent(byte) ? .exponent : nil
        case .exponent:
            if byte == UInt8(ascii: "+") || byte == UInt8(ascii: "-") { return .exponentSign }
            return digit ? .exponentDigits : nil
        case .exponentSign, .exponentDigits:
            return digit ? .exponentDigits : nil
        }
    }

    private static func isTerminal(_ phase: NFKMLXJSONState.NumberPhase) -> Bool {
        switch phase {
        case .zero, .integer, .fraction, .exponentDigits: return true
        default: return false
        }
    }

    private static func word(_ literal: NFKMLXJSONState.Literal) -> [UInt8] {
        switch literal {
        case .true: return Array("true".utf8)
        case .false: return Array("false".utf8)
        case .null: return Array("null".utf8)
        }
    }

    private static func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09
    }
    private static func isDigit(_ byte: UInt8) -> Bool { (UInt8(ascii: "0") ... UInt8(ascii: "9")).contains(byte) }
    private static func isExponent(_ byte: UInt8) -> Bool { byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E") }
    private static func isHexDigit(_ byte: UInt8) -> Bool {
        isDigit(byte) || (UInt8(ascii: "a") ... UInt8(ascii: "f")).contains(byte)
            || (UInt8(ascii: "A") ... UInt8(ascii: "F")).contains(byte)
    }
}

// MARK: - Choices

/// Constrains the output to exactly one of a fixed set of strings.
///
/// @discussion A classification or a menu: the model picks among the choices and nothing else,
/// ending as soon as a choice is spelled out in full. The state is the bytes emitted so far, and a
/// token is admissible while some choice still begins with the output plus the token. A choice
/// that is a prefix of another (`"yes"` and `"yes, please"`) ends only when the model emits the
/// end token, which the grammar admits at any complete choice.
public final class NFKMLXChoiceConstraint: NFKMLXByteConstraint<[UInt8]>, @unchecked Sendable {
    public let choices: [[UInt8]]

    public init(choices: [String], vocabulary: NFKMLXVocabulary) {
        self.choices = choices.map { Array($0.utf8) }
        super.init(vocabulary: vocabulary)
    }

    public override func initialState() -> [UInt8] { [] }

    public override func advance(_ state: [UInt8], byte: UInt8) -> [UInt8]? {
        let next = state + [byte]
        return choices.contains { $0.count >= next.count && $0.prefix(next.count).elementsEqual(next) } ? next : nil
    }

    public override func isComplete(_ state: [UInt8]) -> Bool { choices.contains(state) }
}
