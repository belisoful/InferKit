//
//  NFKMLXPickle.swift
//  InferKitMLX
//
//  A restricted pickle deserializer for PyTorch checkpoints. It interprets the container and
//  dense-tensor opcode surface a checkpoint uses and nothing else: globals resolve to symbols, a
//  constructed object it does not recognize becomes an inert node holding its arguments, and no
//  code from the stream ever executes. Foundation only, so it is tested directly under `swift test`.
//

import Foundation

/// A value the pickle machine produces. Containers are reference types because the stream mutates
/// them after memoization: an `APPENDS` following a `BINPUT` must be visible through a later
/// `BINGET` of the same object.
enum NFKMLXPickleValue {
    case none
    case bool(Bool)
    case int(Int64)
    /// An integer wider than 64 bits, kept as its little-endian two's-complement payload. The
    /// legacy container's magic number is a 10-byte integer, so this is not a theoretical case.
    case bigint(Data)
    case float(Double)
    case string(String)
    case bytes(Data)
    case tuple([NFKMLXPickleValue])
    case list(NFKMLXPickleList)
    case dict(NFKMLXPickleDict)
    case global(module: String, name: String)
    case opaque(NFKMLXPickleOpaque)
    /// A value the caller's `persistentLoad` produced. The machine never inspects it.
    case external(Any)
}

final class NFKMLXPickleList {
    var items: [NFKMLXPickleValue] = []
}

/// A dictionary preserving insertion order, with fast lookup for string keys. Non-string keys are
/// legal pickle and stay reachable through `entries`.
final class NFKMLXPickleDict {
    private(set) var entries: [(key: NFKMLXPickleValue, value: NFKMLXPickleValue)] = []
    private var stringIndex: [String: Int] = [:]
    /// The object state a `BUILD` applied, such as a state_dict's `_metadata` attribute. Kept so the
    /// stream parses; nothing reads it.
    var attributes: NFKMLXPickleValue?

    func setItem(_ key: NFKMLXPickleValue, _ value: NFKMLXPickleValue) {
        if case .string(let name) = key, let existing = stringIndex[name] {
            entries[existing].value = value
            return
        }
        if case .string(let name) = key {
            stringIndex[name] = entries.count
        }
        entries.append((key, value))
    }

    subscript(key: String) -> NFKMLXPickleValue? {
        guard let index = stringIndex[key] else { return nil }
        return entries[index].value
    }
}

/// A constructed object the machine does not interpret: the global that would have built it, its
/// arguments, and whatever state or items the stream attached. Traversal reads the ones it
/// recognizes and drops the rest.
final class NFKMLXPickleOpaque {
    let module: String
    let name: String
    let args: [NFKMLXPickleValue]
    var state: NFKMLXPickleValue?
    let items = NFKMLXPickleDict()

    init(module: String, name: String, args: [NFKMLXPickleValue]) {
        self.module = module
        self.name = name
        self.args = args
    }

    var qualifiedName: String { "\(module).\(name)" }
}

enum NFKMLXPickle {

    /// Runs one pickle starting at `offset` and returns its root together with the offset past the
    /// `STOP` opcode; the legacy checkpoint container reads five pickles back to back.
    /// `persistentLoad` receives each `BINPERSID` argument, which is how the container layer turns a
    /// storage reference into a byte range without the machine knowing what a storage is.
    static func load(_ data: Data, from offset: Int = 0,
                     persistentLoad: (NFKMLXPickleValue) throws -> NFKMLXPickleValue)
        throws -> (root: NFKMLXPickleValue, end: Int) {
        try withoutActuallyEscaping(persistentLoad) { handler in
            try Machine(data: data, cursor: offset, persistentLoad: handler).run()
        }
    }

    private final class Machine {
        let reader: NFKMLXByteReader
        var cursor: Int
        let persistentLoad: (NFKMLXPickleValue) throws -> NFKMLXPickleValue
        var stack: [NFKMLXPickleValue] = []
        var marks: [Int] = []
        var memo: [Int: NFKMLXPickleValue] = [:]

        init(data: Data, cursor: Int, persistentLoad: @escaping (NFKMLXPickleValue) throws -> NFKMLXPickleValue) {
            self.reader = NFKMLXByteReader(data)
            self.cursor = cursor
            self.persistentLoad = persistentLoad
        }

        func run() throws -> (root: NFKMLXPickleValue, end: Int) {
            while true {
                let opcodeOffset = cursor
                let opcode = try byte()
                switch opcode {
                case 0x80:  // PROTO
                    let version = try byte()
                    guard (2 ... 5).contains(version) else {
                        throw NFKMLXError.unsupportedConfiguration(
                            "pickle protocol \(version) is not supported; PyTorch writes protocol 2")
                    }
                case 0x95:  // FRAME
                    cursor += 8
                case 0x2e:  // STOP
                    return (try pop("STOP"), cursor)
                case 0x28:  // MARK
                    marks.append(stack.count)

                case 0x71:  // BINPUT
                    memo[try byte()] = try top("BINPUT")
                case 0x72:  // LONG_BINPUT
                    memo[try u32()] = try top("LONG_BINPUT")
                case 0x94:  // MEMOIZE
                    memo[memo.count] = try top("MEMOIZE")
                case 0x68:  // BINGET
                    stack.append(try memoized(try byte()))
                case 0x6a:  // LONG_BINGET
                    stack.append(try memoized(try u32()))

                case 0x4e:  // NONE
                    stack.append(.none)
                case 0x88:  // NEWTRUE
                    stack.append(.bool(true))
                case 0x89:  // NEWFALSE
                    stack.append(.bool(false))
                case 0x4a:  // BININT
                    let raw = try u32()
                    stack.append(.int(Int64(Int32(truncatingIfNeeded: raw))))
                case 0x4b:  // BININT1
                    stack.append(.int(Int64(try byte())))
                case 0x4d:  // BININT2
                    stack.append(.int(Int64(try u16())))
                case 0x8a:  // LONG1
                    stack.append(try long(count: try byte()))
                case 0x8b:  // LONG4
                    stack.append(try long(count: try u32()))
                case 0x47:  // BINFLOAT
                    var pattern: UInt64 = 0
                    for _ in 0 ..< 8 {
                        pattern = (pattern << 8) | UInt64(try byte())
                    }
                    stack.append(.float(Double(bitPattern: pattern)))

                case 0x58:  // BINUNICODE
                    stack.append(.string(try text(count: try u32(), latin1Fallback: false)))
                case 0x8c:  // SHORT_BINUNICODE
                    stack.append(.string(try text(count: try byte(), latin1Fallback: false)))
                case 0x8d:  // BINUNICODE8
                    stack.append(.string(try text(count: try u64(), latin1Fallback: false)))
                case 0x55:  // SHORT_BINSTRING
                    stack.append(.string(try text(count: try byte(), latin1Fallback: true)))
                case 0x54:  // BINSTRING
                    let count = try u32()
                    guard count <= Int(Int32.max) else {
                        throw malformed("a BINSTRING length overflows", at: opcodeOffset)
                    }
                    stack.append(.string(try text(count: count, latin1Fallback: true)))
                case 0x43:  // SHORT_BINBYTES
                    stack.append(.bytes(try raw(count: try byte())))
                case 0x42:  // BINBYTES
                    stack.append(.bytes(try raw(count: try u32())))
                case 0x8e:  // BINBYTES8
                    stack.append(.bytes(try raw(count: try u64())))

                case 0x7d:  // EMPTY_DICT
                    stack.append(.dict(NFKMLXPickleDict()))
                case 0x5d, 0x8f:  // EMPTY_LIST, EMPTY_SET (a set's element order carries no meaning here)
                    stack.append(.list(NFKMLXPickleList()))
                case 0x29:  // EMPTY_TUPLE
                    stack.append(.tuple([]))
                case 0x74:  // TUPLE
                    stack.append(.tuple(try popToMark("TUPLE")))
                case 0x85:  // TUPLE1
                    stack.append(.tuple([try pop("TUPLE1")]))
                case 0x86:  // TUPLE2
                    let second = try pop("TUPLE2"); let first = try pop("TUPLE2")
                    stack.append(.tuple([first, second]))
                case 0x87:  // TUPLE3
                    let third = try pop("TUPLE3"); let second = try pop("TUPLE3"); let first = try pop("TUPLE3")
                    stack.append(.tuple([first, second, third]))

                case 0x61:  // APPEND
                    let item = try pop("APPEND")
                    try list(at: opcodeOffset, "APPEND").items.append(item)
                case 0x65, 0x90:  // APPENDS, ADDITEMS
                    let items = try popToMark("APPENDS")
                    try list(at: opcodeOffset, "APPENDS").items.append(contentsOf: items)
                case 0x73:  // SETITEM
                    let value = try pop("SETITEM"); let key = try pop("SETITEM")
                    try setItem(key, value, at: opcodeOffset)
                case 0x75:  // SETITEMS
                    let items = try popToMark("SETITEMS")
                    guard items.count.isMultiple(of: 2) else {
                        throw malformed("SETITEMS holds an odd item count", at: opcodeOffset)
                    }
                    for index in stride(from: 0, to: items.count, by: 2) {
                        try setItem(items[index], items[index + 1], at: opcodeOffset)
                    }

                case 0x63:  // GLOBAL
                    let module = try line()
                    let name = try line()
                    stack.append(.global(module: module, name: name))
                case 0x93:  // STACK_GLOBAL
                    guard case .string(let name) = try pop("STACK_GLOBAL"),
                          case .string(let module) = try pop("STACK_GLOBAL") else {
                        throw malformed("STACK_GLOBAL expects two strings", at: opcodeOffset)
                    }
                    stack.append(.global(module: module, name: name))
                case 0x52:  // REDUCE
                    let args = try tupleArguments("REDUCE", at: opcodeOffset)
                    stack.append(try constructed(try pop("REDUCE"), args, at: opcodeOffset))
                case 0x81:  // NEWOBJ
                    let args = try tupleArguments("NEWOBJ", at: opcodeOffset)
                    stack.append(try constructed(try pop("NEWOBJ"), args, at: opcodeOffset))
                case 0x92:  // NEWOBJ_EX
                    _ = try pop("NEWOBJ_EX")  // keyword arguments; nothing on this surface takes them
                    let args = try tupleArguments("NEWOBJ_EX", at: opcodeOffset)
                    stack.append(try constructed(try pop("NEWOBJ_EX"), args, at: opcodeOffset))
                case 0x62:  // BUILD
                    let state = try pop("BUILD")
                    switch try top("BUILD") {
                    case .dict(let dict): dict.attributes = state
                    case .opaque(let node): node.state = state
                    default: throw malformed("BUILD applies to a dict or constructed object", at: opcodeOffset)
                    }
                case 0x51:  // BINPERSID
                    stack.append(try persistentLoad(try pop("BINPERSID")))

                default:
                    throw malformed(refusalDescription(opcode), at: opcodeOffset)
                }
            }
        }

        // MARK: Stream reads

        func byte() throws -> Int {
            let value = try reader.u8(cursor)
            cursor += 1
            return value
        }

        func u16() throws -> Int {
            let value = try reader.u16(cursor)
            cursor += 2
            return value
        }

        func u32() throws -> Int {
            let value = try reader.u32(cursor)
            cursor += 4
            return value
        }

        func u64() throws -> Int {
            let value = try reader.u64(cursor)
            cursor += 8
            return value
        }

        func raw(count: Int) throws -> Data {
            let value = try reader.bytes(cursor, count: count)
            cursor += count
            return value
        }

        /// Decodes text as UTF-8, falling back to Latin-1 for the py2-era string opcodes, which is
        /// how PyTorch itself reads them.
        func text(count: Int, latin1Fallback: Bool) throws -> String {
            let data = try raw(count: count)
            if let utf8 = String(data: data, encoding: .utf8) {
                return utf8
            }
            if latin1Fallback, let latin1 = String(data: data, encoding: .isoLatin1) {
                return latin1
            }
            throw malformed("a string does not decode as UTF-8", at: cursor - count)
        }

        func line() throws -> String {
            let start = cursor
            while try reader.u8(cursor) != 0x0a {
                cursor += 1
            }
            let data = try reader.bytes(start, count: cursor - start)
            cursor += 1
            return String(decoding: data, as: UTF8.self)
        }

        func long(count: Int) throws -> NFKMLXPickleValue {
            let payload = try raw(count: count)
            guard count > 0 else { return .int(0) }
            guard count <= 8 else { return .bigint(payload) }
            var value: UInt64 = 0
            for (index, byte) in payload.enumerated() {
                value |= UInt64(byte) << (8 * UInt64(index))
            }
            if count < 8, payload.last! & 0x80 != 0 {
                value |= ~UInt64(0) << (8 * UInt64(count))  // sign-extend
            }
            return .int(Int64(bitPattern: value))
        }

        // MARK: Stack

        func pop(_ opcode: String) throws -> NFKMLXPickleValue {
            guard let value = stack.popLast() else {
                throw NFKMLXError.malformedCheckpoint("pickle \(opcode) underflows the stack at offset \(cursor)")
            }
            return value
        }

        func top(_ opcode: String) throws -> NFKMLXPickleValue {
            guard let value = stack.last else {
                throw NFKMLXError.malformedCheckpoint("pickle \(opcode) reads an empty stack at offset \(cursor)")
            }
            return value
        }

        func popToMark(_ opcode: String) throws -> [NFKMLXPickleValue] {
            guard let mark = marks.popLast(), mark <= stack.count else {
                throw NFKMLXError.malformedCheckpoint("pickle \(opcode) finds no mark at offset \(cursor)")
            }
            let items = Array(stack[mark...])
            stack.removeSubrange(mark...)
            return items
        }

        func memoized(_ index: Int) throws -> NFKMLXPickleValue {
            guard let value = memo[index] else {
                throw NFKMLXError.malformedCheckpoint("pickle memo slot \(index) is read before being set")
            }
            return value
        }

        func tupleArguments(_ opcode: String, at offset: Int) throws -> [NFKMLXPickleValue] {
            guard case .tuple(let args) = try pop(opcode) else {
                throw malformed("\(opcode) expects a tuple of arguments", at: offset)
            }
            return args
        }

        // MARK: Semantics

        /// The only global the machine itself interprets is `collections.OrderedDict`, because the
        /// stream sets items into it. Every other construction becomes an inert node.
        func constructed(_ callable: NFKMLXPickleValue, _ args: [NFKMLXPickleValue],
                         at offset: Int) throws -> NFKMLXPickleValue {
            switch callable {
            case .global(let module, let name):
                if module == "collections", name == "OrderedDict" {
                    let dict = NFKMLXPickleDict()
                    // An older pickler passes the items as a constructor argument — a list of
                    // (key, value) pairs — where a newer one uses SETITEMS on the empty dict.
                    if case .list(let pairs)? = args.first {
                        for pair in pairs.items {
                            switch pair {
                            case .list(let item) where item.items.count == 2:
                                dict.setItem(item.items[0], item.items[1])
                            case .tuple(let item) where item.count == 2:
                                dict.setItem(item[0], item[1])
                            default:
                                break
                            }
                        }
                    }
                    return .dict(dict)
                }
                return .opaque(NFKMLXPickleOpaque(module: module, name: name, args: args))
            case .opaque(let node):
                return .opaque(NFKMLXPickleOpaque(module: node.module, name: node.name, args: args))
            default:
                throw malformed("a construction's callable is neither a global nor a constructed object", at: offset)
            }
        }

        func list(at offset: Int, _ opcode: String) throws -> NFKMLXPickleList {
            guard case .list(let target) = try top(opcode) else {
                throw malformed("\(opcode) applies to a list", at: offset)
            }
            return target
        }

        func setItem(_ key: NFKMLXPickleValue, _ value: NFKMLXPickleValue, at offset: Int) throws {
            switch try top("SETITEM") {
            case .dict(let dict): dict.setItem(key, value)
            case .opaque(let node): node.items.setItem(key, value)
            default: throw malformed("SETITEM applies to a dict or constructed object", at: offset)
            }
        }

        func malformed(_ detail: String, at offset: Int) -> NFKMLXError {
            .malformedCheckpoint("pickle: \(detail) (offset \(offset))")
        }

        func refusalDescription(_ opcode: Int) -> String {
            let named: [Int: String] = [
                0x50: "PERSID", 0x69: "INST", 0x6f: "OBJ",
                0x82: "EXT1", 0x83: "EXT2", 0x84: "EXT4",
                0x96: "BYTEARRAY8", 0x97: "NEXT_BUFFER", 0x98: "READONLY_BUFFER",
                0x30: "POP", 0x31: "POP_MARK", 0x32: "DUP",
                0x49: "INT", 0x4c: "LONG", 0x46: "FLOAT", 0x53: "STRING", 0x56: "UNICODE",
                0x70: "PUT", 0x67: "GET", 0x64: "DICT", 0x6c: "LIST",
            ]
            let hex = String(format: "0x%02x", opcode)
            if let name = named[opcode] {
                return "opcode \(name) (\(hex)) is outside the checkpoint surface this reader supports"
            }
            return "unknown opcode \(hex)"
        }
    }
}
