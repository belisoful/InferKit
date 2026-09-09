//
//  NFKMLXJSONSchemaConstraint.swift
//  InferKitMLX
//
//  Schema-constrained sampling: the JSON grammar narrowed to a JSON Schema, so the keys, the types,
//  the enumerations, and the array bounds are guaranteed as well as the syntax.
//

import Foundation
import InferKit
import MLX

/// A JSON Schema compiled into the nodes a byte-level grammar walks.
///
/// @discussion The subset a constrained decoder can enforce one byte at a time: `type` (a name or a
/// list of names), `properties` / `required` / `additionalProperties`, `items` / `minItems` /
/// `maxItems`, `enum`, `const`, `anyOf` / `oneOf`, and `$ref` into `$defs` or `definitions`
/// (recursion included). An empty schema, `true`, or an omitted `additionalProperties` admits any
/// JSON value at that point. A keyword the grammar cannot check byte by byte is ignored when it only
/// narrows a value's content (`pattern`, `format`, `minimum`, `maximum`, `minLength`, `maxLength`)
/// and refused when it changes which structures are admissible (`allOf`, `not`, `if`). A schema that
/// compiles is one whose structure the output is guaranteed to have; what the model chooses inside
/// that structure stays its own.
///
/// Keys are spelled in any order; an object may close only once every `required` key has appeared,
/// and a key is admitted only while it is the prefix of an unwritten property (or, where
/// `additionalProperties` allows it, any key). A property name is matched as raw bytes, so a name that
/// needs a JSON escape cannot be constrained to. Introduced in InferKit 0.4.0.
public struct NFKMLXJSONSchema: Hashable, Sendable {
    /// A property of an object node: the bytes its key is spelled with, its value's node, and
    /// whether the object may close without it.
    struct Property: Hashable, Sendable {
        let name: [UInt8]
        let node: Int
        let required: Bool
    }

    enum Node: Hashable, Sendable {
        /// `properties` in name order; `additional` is the node an unlisted key's value takes, or nil
        /// when unlisted keys are forbidden.
        case object(properties: [Property], additional: Int?)
        case array(items: Int, minItems: Int, maxItems: Int?)
        case string
        case number
        case integer
        case boolean
        case null
        /// The compact JSON serialization of each admitted value, matched byte for byte.
        case choices([[UInt8]])
        case anyOf([Int])
        case any
    }

    let nodes: [Node]
    let root: Int

    /// The most properties one object may declare, so the set of written keys fits a bit mask.
    public static let maximumProperties = 64

    /// Compiles a schema given as the dictionary `JSONSerialization` produces, which is also what the
    /// core's `NFKParameterJSONSchema` carries.
    public init(json: [String: Any]) throws {
        var compiler = Compiler(definitions: Self.definitions(in: json))
        let root = try compiler.compile(json, path: "#")
        self.nodes = compiler.nodes
        self.root = root
    }

    /// Compiles a schema from its JSON text.
    public init(jsonText: String) throws {
        try self.init(data: Data(jsonText.utf8))
    }

    /// Compiles a schema from JSON data.
    public init(data: Data) throws {
        let object = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        if let dictionary = object as? [String: Any] {
            try self.init(json: dictionary)
        } else if object as? Bool == true {
            try self.init(json: [:])
        } else {
            throw NFKMLXError.unsupportedConfiguration("a JSON Schema is an object")
        }
    }

    private static func definitions(in json: [String: Any]) -> [String: Any] {
        var found = [String: Any]()
        for key in ["$defs", "definitions"] {
            if let table = json[key] as? [String: Any] {
                for (name, schema) in table { found["#/\(key)/\(name)"] = schema }
            }
        }
        return found
    }

    /// Walks a schema dictionary into `nodes`. A `$ref` reserves its node before compiling the
    /// referenced schema, which is what lets a recursive schema compile.
    private struct Compiler {
        let definitions: [String: Any]
        var nodes = [Node]()
        var referenced = [String: Int]()

        init(definitions: [String: Any]) { self.definitions = definitions }

        mutating func compile(_ schema: Any, path: String) throws -> Int {
            if let flag = schema as? Bool {
                if flag { return append(.any) }
                throw NFKMLXError.unsupportedConfiguration("\(path): a `false` schema admits nothing")
            }
            guard let dictionary = schema as? [String: Any] else {
                throw NFKMLXError.unsupportedConfiguration("\(path): a schema is an object or a boolean")
            }
            if let reference = dictionary["$ref"] as? String {
                return try resolve(reference, path: path)
            }
            for refused in ["allOf", "not", "if", "then", "else", "patternProperties", "dependentSchemas"]
                where dictionary[refused] != nil {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(path): `\(refused)` cannot be enforced one byte at a time")
            }
            if let values = dictionary["enum"] as? [Any] {
                return append(.choices(try values.map { try Self.serialized($0, path: path) }))
            }
            if dictionary.keys.contains("const") {
                return append(.choices([try Self.serialized(dictionary["const"] ?? NSNull(), path: path)]))
            }
            for key in ["anyOf", "oneOf"] {
                if let alternatives = dictionary[key] as? [Any] {
                    guard !alternatives.isEmpty else {
                        throw NFKMLXError.unsupportedConfiguration("\(path): `\(key)` lists no alternatives")
                    }
                    let index = append(.any)
                    var compiled = [Int]()
                    for (position, alternative) in alternatives.enumerated() {
                        compiled.append(try compile(alternative, path: "\(path)/\(key)/\(position)"))
                    }
                    nodes[index] = .anyOf(compiled)
                    return index
                }
            }
            if let names = dictionary["type"] as? [String] {
                guard !names.isEmpty else {
                    throw NFKMLXError.unsupportedConfiguration("\(path): `type` lists no names")
                }
                if names.count == 1 { return try compile(type: names[0], dictionary, path: path) }
                let index = append(.any)
                var compiled = [Int]()
                for name in names { compiled.append(try compile(type: name, dictionary, path: path)) }
                nodes[index] = .anyOf(compiled)
                return index
            }
            if let name = dictionary["type"] as? String {
                return try compile(type: name, dictionary, path: path)
            }
            if dictionary["properties"] != nil || dictionary["required"] != nil {
                return try compile(type: "object", dictionary, path: path)
            }
            if dictionary["items"] != nil {
                return try compile(type: "array", dictionary, path: path)
            }
            return append(.any)
        }

        private mutating func compile(type name: String, _ dictionary: [String: Any], path: String) throws -> Int {
            switch name {
            case "object":
                let index = append(.any)
                let declared = dictionary["properties"] as? [String: Any] ?? [:]
                guard declared.count <= NFKMLXJSONSchema.maximumProperties else {
                    throw NFKMLXError.unsupportedConfiguration(
                        "\(path): an object declares at most \(NFKMLXJSONSchema.maximumProperties) properties")
                }
                let required = Set(dictionary["required"] as? [String] ?? [])
                var properties = [Property]()
                for name in declared.keys.sorted() {
                    let node = try compile(declared[name] ?? [:], path: "\(path)/properties/\(name)")
                    properties.append(Property(name: Array(name.utf8), node: node, required: required.contains(name)))
                }
                for name in required where declared[name] == nil {
                    throw NFKMLXError.unsupportedConfiguration(
                        "\(path): required property `\(name)` is not declared under `properties`")
                }
                let additional: Int?
                switch dictionary["additionalProperties"] {
                case nil: additional = append(.any)
                case let flag as Bool: additional = flag ? append(.any) : nil
                case let schema?: additional = try compile(schema, path: "\(path)/additionalProperties")
                }
                nodes[index] = .object(properties: properties, additional: additional)
                return index
            case "array":
                let index = append(.any)
                let items = try dictionary["items"].map { try compile($0, path: "\(path)/items") } ?? append(.any)
                let minimum = (dictionary["minItems"] as? NSNumber)?.intValue ?? 0
                let maximum = (dictionary["maxItems"] as? NSNumber)?.intValue
                if let maximum, maximum < minimum {
                    throw NFKMLXError.unsupportedConfiguration("\(path): `maxItems` is below `minItems`")
                }
                nodes[index] = .array(items: items, minItems: minimum, maxItems: maximum)
                return index
            case "string": return append(.string)
            case "number": return append(.number)
            case "integer": return append(.integer)
            case "boolean": return append(.boolean)
            case "null": return append(.null)
            default:
                throw NFKMLXError.unsupportedConfiguration("\(path): unknown type `\(name)`")
            }
        }

        private mutating func resolve(_ reference: String, path: String) throws -> Int {
            if let known = referenced[reference] { return known }
            guard let schema = definitions[reference] else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(path): `$ref` \(reference) is not under `$defs` or `definitions`")
            }
            let index = append(.any)
            referenced[reference] = index
            let compiled = try compile(schema, path: reference)
            nodes[index] = nodes[compiled]
            return index
        }

        private mutating func append(_ node: Node) -> Int {
            nodes.append(node)
            return nodes.count - 1
        }

        /// The compact JSON text of a literal value, with object keys sorted so the serialization is
        /// the one deterministic spelling the grammar can match.
        private static func serialized(_ value: Any, path: String) throws -> [UInt8] {
            let data = try JSONSerialization.data(withJSONObject: value,
                                                  options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes])
            return [UInt8](data)
        }
    }
}

/// Where a schema-constrained output stands: the machines still alive, each a deterministic walk of
/// one reading of the schema. More than one is alive only while an `anyOf` has not yet told its
/// alternatives apart.
public struct NFKMLXJSONSchemaState: Hashable, Sendable {
    var machines: [Machine]

    struct Machine: Hashable, Sendable {
        enum ObjectPhase: Hashable {
            case keyOrEnd
            case keyRequired
            /// Inside a key that is still the prefix of an unwritten property.
            case key([UInt8])
            /// Inside a key that no property matches, admitted because unlisted keys are allowed.
            case freeKey(NFKMLXJSONState.StringPhase)
            case colon
            case value
            case commaOrEnd
        }
        enum ArrayPhase: Hashable { case valueOrEnd, value, commaOrEnd }
        enum Frame: Hashable {
            case root(node: Int)
            /// `seen` is a bit per property; `current` is the property whose value is being written,
            /// or -1 for an unlisted key.
            case object(node: Int, phase: ObjectPhase, seen: UInt64, current: Int)
            case array(node: Int, phase: ArrayPhase, count: Int)
            /// An `any` value, walked by the free JSON grammar.
            case free(NFKMLXJSONState)
        }
        enum Scalar: Hashable {
            case none
            case string(NFKMLXJSONState.StringPhase)
            case number(NFKMLXJSONState.NumberPhase, integer: Bool)
            case literal(NFKMLXJSONState.Literal, Int)
            /// An `enum`/`const` value spelled so far.
            case choice(node: Int, prefix: [UInt8])
        }

        var frames: [Frame]
        var scalar: Scalar = .none
        var complete = false
        var whitespaceRun: UInt8 = 0
    }
}

/// Constrains the output to JSON that conforms to an ``NFKMLXJSONSchema``.
///
/// @discussion The same byte-level engine as ``NFKMLXJSONConstraint``, walking the compiled schema
/// instead of the free grammar: an object admits only its declared keys (unless it allows unlisted
/// ones), closes only once its required keys are written, and types each value; an array honors its
/// item type and count bounds; an `enum` or `const` admits only its listed spellings; an `anyOf`
/// keeps every alternative alive until the bytes decide. Whitespace between tokens is capped as the
/// free grammar caps it, and for the same reason. Introduced in InferKit 0.4.0.
public final class NFKMLXJSONSchemaConstraint: NFKMLXByteConstraint<NFKMLXJSONSchemaState>, @unchecked Sendable {
    public let schema: NFKMLXJSONSchema
    /// The most consecutive whitespace bytes admitted between tokens. See
    /// ``NFKMLXJSONConstraint/maximumWhitespaceRun``.
    public let maximumWhitespaceRun: Int
    /// The free grammar an `any` value runs under.
    private let free: NFKMLXJSONConstraint

    private typealias Machine = NFKMLXJSONSchemaState.Machine

    public init(schema: NFKMLXJSONSchema, vocabulary: NFKMLXVocabulary, maximumWhitespaceRun: Int = 8) {
        self.schema = schema
        self.maximumWhitespaceRun = maximumWhitespaceRun
        free = NFKMLXJSONConstraint(vocabulary: vocabulary, root: .any, maximumWhitespaceRun: maximumWhitespaceRun)
        super.init(vocabulary: vocabulary)
    }

    public override func initialState() -> NFKMLXJSONSchemaState {
        NFKMLXJSONSchemaState(machines: [Machine(frames: [.root(node: schema.root)])])
    }

    public override func isComplete(_ state: NFKMLXJSONSchemaState) -> Bool {
        state.machines.contains { isComplete($0) }
    }

    public override func advance(_ state: NFKMLXJSONSchemaState, byte: UInt8) -> NFKMLXJSONSchemaState? {
        var next = [Machine]()
        var seen = Set<Machine>()
        for machine in state.machines {
            for advanced in advance(machine, byte: byte) where seen.insert(advanced).inserted {
                next.append(advanced)
            }
        }
        return next.isEmpty ? nil : NFKMLXJSONSchemaState(machines: next)
    }

    // MARK: One machine

    private func isComplete(_ m: Machine) -> Bool {
        if m.complete { return true }
        // A root scalar that cannot know it has ended until something follows.
        guard m.frames.count >= 1, case .root = m.frames[0] else { return false }
        if m.frames.count == 2, case .free(let state) = m.frames[1] { return free.isComplete(state) }
        guard m.frames.count == 1 else { return false }
        switch m.scalar {
        case .number(let phase, _): return NFKMLXJSONConstraint.isTerminal(phase)
        case .choice(let node, let prefix): return choices(node).contains(prefix)
        default: return false
        }
    }

    private func advance(_ machine: Machine, byte: UInt8) -> [Machine] {
        var m = machine
        if m.complete { return NFKMLXJSONConstraint.isWhitespace(byte) ? [m] : [] }

        switch m.scalar {
        case .string(let phase):
            switch Self.advanceString(phase, byte: byte) {
            case .inside(let next): m.scalar = .string(next); return [m]
            case .closed: m.scalar = .none; return [finished(m)]
            case .rejected: return []
            }
        case .number(let phase, let integer):
            if let next = NFKMLXJSONConstraint.advanceNumber(phase, byte: byte) {
                if integer, [.dot, .exponent].contains(next) { return [] }
                m.scalar = .number(next, integer: integer)
                return [m]
            }
            guard NFKMLXJSONConstraint.isTerminal(phase) else { return [] }
            m.scalar = .none
            return advance(finished(m), byte: byte)
        case .literal(let literal, let index):
            let word = NFKMLXJSONConstraint.word(literal)
            guard byte == word[index] else { return [] }
            if index + 1 == word.count {
                m.scalar = .none
                return [finished(m)]
            }
            m.scalar = .literal(literal, index + 1)
            return [m]
        case .choice(let node, let prefix):
            let extended = prefix + [byte]
            let candidates = choices(node)
            if candidates.contains(where: { $0.starts(with: extended) }) {
                m.scalar = .choice(node: node, prefix: extended)
                return [m]
            }
            guard candidates.contains(prefix) else { return [] }
            m.scalar = .none
            return advance(finished(m), byte: byte)
        case .none:
            break
        }

        guard let top = m.frames.last else { return [] }

        // A key in progress is string content, so it is handled before whitespace is counted.
        if case .object(let node, let phase, let seen, _) = top {
            switch phase {
            case .key(let prefix):
                return advanceKey(m, node: node, prefix: prefix, seen: seen, byte: byte)
            case .freeKey(let stringPhase):
                switch Self.advanceString(stringPhase, byte: byte) {
                case .inside(let next):
                    m.frames[m.frames.count - 1] = .object(node: node, phase: .freeKey(next), seen: seen, current: -1)
                    return [m]
                case .closed:
                    m.frames[m.frames.count - 1] = .object(node: node, phase: .colon, seen: seen, current: -1)
                    return [m]
                case .rejected:
                    return []
                }
            default:
                break
            }
        }
        if case .free(let state) = top {
            if let next = free.advance(state, byte: byte) {
                m.frames[m.frames.count - 1] = .free(next)
                return [m]
            }
            guard free.isComplete(state) else { return [] }
            m.frames.removeLast()
            return advance(finished(m), byte: byte)
        }

        if NFKMLXJSONConstraint.isWhitespace(byte) {
            guard Int(m.whitespaceRun) < maximumWhitespaceRun else { return [] }
            m.whitespaceRun += 1
            return [m]
        }
        m.whitespaceRun = 0

        switch top {
        case .root(let node):
            return startValue(m, node: node, byte: byte)
        case .free:
            return []
        case .object(let node, let phase, let seen, let current):
            guard case .object(let properties, let additional) = schema.nodes[node] else { return [] }
            let last = m.frames.count - 1
            switch (phase, byte) {
            case (.keyOrEnd, UInt8(ascii: "}")), (.commaOrEnd, UInt8(ascii: "}")):
                guard Self.requiredSatisfied(properties, seen: seen) else { return [] }
                m.frames.removeLast()
                return [finished(m)]
            case (.keyOrEnd, UInt8(ascii: "\"")), (.keyRequired, UInt8(ascii: "\"")):
                m.frames[last] = .object(node: node, phase: .key([]), seen: seen, current: current)
                return [m]
            case (.colon, UInt8(ascii: ":")):
                m.frames[last] = .object(node: node, phase: .value, seen: seen, current: current)
                return [m]
            case (.value, _):
                m.frames[last] = .object(node: node, phase: .commaOrEnd, seen: seen, current: current)
                guard let valueNode = current >= 0 ? properties[current].node : additional else { return [] }
                return startValue(m, node: valueNode, byte: byte)
            case (.commaOrEnd, UInt8(ascii: ",")):
                // A comma promises a key: refused when no property is left and unlisted keys are forbidden.
                let unwritten = properties.indices.contains { seen & (1 << UInt64($0)) == 0 }
                guard unwritten || additional != nil else { return [] }
                m.frames[last] = .object(node: node, phase: .keyRequired, seen: seen, current: current)
                return [m]
            default:
                return []
            }
        case .array(let node, let phase, let count):
            guard case .array(let items, let minimum, let maximum) = schema.nodes[node] else { return [] }
            let last = m.frames.count - 1
            let roomForAnother = maximum.map { count < $0 } ?? true
            switch (phase, byte) {
            case (.valueOrEnd, UInt8(ascii: "]")), (.commaOrEnd, UInt8(ascii: "]")):
                guard count >= minimum else { return [] }
                m.frames.removeLast()
                return [finished(m)]
            case (.valueOrEnd, _), (.value, _):
                guard roomForAnother else { return [] }
                m.frames[last] = .array(node: node, phase: .commaOrEnd, count: count + 1)
                return startValue(m, node: items, byte: byte)
            case (.commaOrEnd, UInt8(ascii: ",")):
                guard roomForAnother else { return [] }
                m.frames[last] = .array(node: node, phase: .value, count: count)
                return [m]
            default:
                return []
            }
        }
    }

    /// Advances a key that is still the prefix of an unwritten property. The closing quote selects
    /// the property spelled exactly; a byte that leaves every property behind turns the key into an
    /// unlisted one where the object allows those.
    private func advanceKey(_ machine: Machine, node: Int, prefix: [UInt8], seen: UInt64, byte: UInt8) -> [Machine] {
        var m = machine
        guard case .object(let properties, let additional) = schema.nodes[node] else { return [] }
        let last = m.frames.count - 1
        let unwritten = properties.indices.filter { seen & (1 << UInt64($0)) == 0 }
        switch byte {
        case UInt8(ascii: "\""):
            if let index = unwritten.first(where: { properties[$0].name == prefix }) {
                m.frames[last] = .object(node: node, phase: .colon, seen: seen | (1 << UInt64(index)), current: index)
                return [m]
            }
            guard additional != nil else { return [] }
            m.frames[last] = .object(node: node, phase: .colon, seen: seen, current: -1)
            return [m]
        case UInt8(ascii: "\\"):
            guard additional != nil else { return [] }
            m.frames[last] = .object(node: node, phase: .freeKey(.escape), seen: seen, current: -1)
            return [m]
        case 0 ..< 0x20:
            return []
        default:
            let extended = prefix + [byte]
            if unwritten.contains(where: { properties[$0].name.starts(with: extended) }) {
                m.frames[last] = .object(node: node, phase: .key(extended), seen: seen, current: -1)
                return [m]
            }
            guard additional != nil else { return [] }
            m.frames[last] = .object(node: node, phase: .freeKey(.body), seen: seen, current: -1)
            return [m]
        }
    }

    /// Opens the value `byte` begins under `node`, with the enclosing frame already advanced past it.
    /// An `anyOf` forks into one machine per alternative the byte can open.
    private func startValue(_ machine: Machine, node: Int, byte: UInt8) -> [Machine] {
        var m = machine
        switch schema.nodes[node] {
        case .anyOf(let alternatives):
            return alternatives.flatMap { startValue(m, node: $0, byte: byte) }
        case .any:
            guard let state = free.advance(NFKMLXJSONState(), byte: byte) else { return [] }
            m.frames.append(.free(state))
            return [m]
        case .object:
            guard byte == UInt8(ascii: "{") else { return [] }
            m.frames.append(.object(node: node, phase: .keyOrEnd, seen: 0, current: -1))
            return [m]
        case .array:
            guard byte == UInt8(ascii: "[") else { return [] }
            m.frames.append(.array(node: node, phase: .valueOrEnd, count: 0))
            return [m]
        case .string:
            guard byte == UInt8(ascii: "\"") else { return [] }
            m.scalar = .string(.body)
            return [m]
        case .number, .integer:
            let integer = schema.nodes[node] == .integer
            switch byte {
            case UInt8(ascii: "-"): m.scalar = .number(.minus, integer: integer)
            case UInt8(ascii: "0"): m.scalar = .number(.zero, integer: integer)
            case UInt8(ascii: "1") ... UInt8(ascii: "9"): m.scalar = .number(.integer, integer: integer)
            default: return []
            }
            return [m]
        case .boolean:
            switch byte {
            case UInt8(ascii: "t"): m.scalar = .literal(.true, 1)
            case UInt8(ascii: "f"): m.scalar = .literal(.false, 1)
            default: return []
            }
            return [m]
        case .null:
            guard byte == UInt8(ascii: "n") else { return [] }
            m.scalar = .literal(.null, 1)
            return [m]
        case .choices(let candidates):
            guard candidates.contains(where: { $0.first == byte }) else { return [] }
            m.scalar = .choice(node: node, prefix: [byte])
            return [m]
        }
    }

    /// A value has closed. Inside a container the frame already waits for a comma or the end; at the
    /// root the document is complete.
    private func finished(_ machine: Machine) -> Machine {
        var m = machine
        if let top = m.frames.last, case .root = top {
            m.frames.removeLast()
        }
        if m.frames.isEmpty { m.complete = true }
        return m
    }

    private func choices(_ node: Int) -> [[UInt8]] {
        if case .choices(let candidates) = schema.nodes[node] { return candidates }
        return []
    }

    private static func requiredSatisfied(_ properties: [NFKMLXJSONSchema.Property], seen: UInt64) -> Bool {
        properties.indices.allSatisfy { !properties[$0].required || seen & (1 << UInt64($0)) != 0 }
    }

    private enum StringStep { case inside(NFKMLXJSONState.StringPhase), closed, rejected }

    /// One byte of a JSON string after its opening quote: the free grammar's rules for escapes,
    /// `\u` digits, and the control characters a string may not carry raw.
    private static func advanceString(_ phase: NFKMLXJSONState.StringPhase, byte: UInt8) -> StringStep {
        switch phase {
        case .body:
            switch byte {
            case UInt8(ascii: "\""): return .closed
            case UInt8(ascii: "\\"): return .inside(.escape)
            case 0 ..< 0x20: return .rejected
            default: return .inside(.body)
            }
        case .escape:
            if byte == UInt8(ascii: "u") { return .inside(.unicode(0)) }
            return "\"\\/bfnrt".utf8.contains(byte) ? .inside(.body) : .rejected
        case .unicode(let digits):
            guard NFKMLXJSONConstraint.isHexDigit(byte) else { return .rejected }
            return .inside(digits + 1 == 4 ? .body : .unicode(digits + 1))
        }
    }
}
