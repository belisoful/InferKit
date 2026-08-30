//
//  NFKMLXPickleTests.swift
//  InferKitMLXTests
//
//  The pickle machine is exercised against hand-built opcode streams, because the format is fixed
//  bytes and a constructed stream pins each opcode's semantics without a checkpoint download. The
//  machine is pure Foundation, so these run under `swift test`.
//

import XCTest
@testable import InferKitMLX

final class NFKMLXPickleTests: XCTestCase {

    private func load(_ body: [UInt8],
                      persistentLoad: (NFKMLXPickleValue) throws -> NFKMLXPickleValue = { $0 })
        throws -> NFKMLXPickleValue {
        let stream = Data([0x80, 0x02] + body + [0x2e])
        return try NFKMLXPickle.load(stream, persistentLoad: persistentLoad).root
    }

    private func unicode(_ text: String) -> [UInt8] {
        let bytes = Array(text.utf8)
        return [0x8c, UInt8(bytes.count)] + bytes
    }

    func testTheScalarOpcodesDecode() throws {
        let root = try load([0x28,                                     // MARK
                             0x4e,                                     // NONE
                             0x88, 0x89,                               // NEWTRUE, NEWFALSE
                             0x4a, 0xfe, 0xff, 0xff, 0xff,             // BININT -2
                             0x4b, 0x07,                               // BININT1 7
                             0x4d, 0x01, 0x02,                         // BININT2 513
                             0x8a, 0x01, 0x85,                         // LONG1 -123
                             0x8a, 0x00,                               // LONG1 0
                             0x47, 0x3f, 0xf8, 0, 0, 0, 0, 0, 0]      // BINFLOAT 1.5 (big-endian)
                            + unicode("héllo")
                            + [0x43, 0x02, 0xab, 0xcd,                 // SHORT_BINBYTES
                               0x74])                                  // TUPLE
        guard case .tuple(let items) = root else { return XCTFail("expected a tuple, got \(root)") }
        XCTAssertEqual(items.count, 11)
        guard case .none = items[0] else { return XCTFail("expected none") }
        guard case .bool(true) = items[1], case .bool(false) = items[2] else { return XCTFail("expected bools") }
        guard case .int(-2) = items[3], case .int(7) = items[4], case .int(513) = items[5] else {
            return XCTFail("expected the three BININT forms")
        }
        guard case .int(-123) = items[6], case .int(0) = items[7] else { return XCTFail("expected LONG1 values") }
        guard case .float(1.5) = items[8] else { return XCTFail("expected 1.5") }
        guard case .string("héllo") = items[9] else { return XCTFail("expected the unicode string") }
        guard case .bytes(let raw) = items[10], [UInt8](raw) == [0xab, 0xcd] else {
            return XCTFail("expected the two raw bytes")
        }
    }

    func testALongWiderThanInt64BecomesABigint() throws {
        // The legacy container's magic number: LONG1 with a 10-byte payload.
        let magic: [UInt8] = [0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19]
        let root = try load([0x8a, 0x0a] + magic)
        guard case .bigint(let payload) = root else { return XCTFail("expected a bigint, got \(root)") }
        XCTAssertEqual([UInt8](payload), magic)
    }

    func testAnEightBytePositiveLongStaysAnInt() throws {
        let root = try load([0x8a, 0x08, 0, 0, 0, 0, 0, 0, 0, 0x10])
        guard case .int(let value) = root else { return XCTFail("expected an int, got \(root)") }
        XCTAssertEqual(value, 0x1000_0000_0000_0000)
    }

    func testThePy2StringOpcodeFallsBackToLatin1() throws {
        let root = try load([0x55, 0x01, 0xe9])  // SHORT_BINSTRING, one byte, é in Latin-1
        guard case .string("é") = root else { return XCTFail("expected é, got \(root)") }
    }

    func testMemoAliasingSurvivesMutation() throws {
        // A list is memoized, appended to, and fetched again: both tuple slots must be the same
        // object holding the appended item, which is why the containers are reference types.
        let root = try load([0x5d,             // EMPTY_LIST
                             0x71, 0x00,       // BINPUT 0
                             0x4b, 0x01,       // BININT1 1
                             0x61,             // APPEND
                             0x68, 0x00,       // BINGET 0
                             0x86])            // TUPLE2
        guard case .tuple(let items) = root, items.count == 2,
              case .list(let first) = items[0], case .list(let second) = items[1] else {
            return XCTFail("expected a tuple of two lists, got \(root)")
        }
        XCTAssertTrue(first === second)
        guard case .int(1) = first.items[0] else { return XCTFail("expected the appended item") }
    }

    func testMemoizeUsesTheRunningCount() throws {
        let root = try load([0x4b, 0x05, 0x94,   // BININT1 5, MEMOIZE (slot 0)
                             0x4b, 0x06, 0x94,   // BININT1 6, MEMOIZE (slot 1)
                             0x68, 0x01,         // BINGET 1
                             0x87])              // TUPLE3
        guard case .tuple(let items) = root, items.count == 3,
              case .int(5) = items[0], case .int(6) = items[1], case .int(6) = items[2] else {
            return XCTFail("expected (5, 6, 6), got \(root)")
        }
    }

    func testDictItemsSetThroughSetitems() throws {
        let root = try load([0x7d, 0x28]          // EMPTY_DICT, MARK
                            + unicode("a") + [0x4b, 0x01]
                            + unicode("b") + [0x4b, 0x02]
                            + [0x75])             // SETITEMS
        guard case .dict(let dict) = root else { return XCTFail("expected a dict, got \(root)") }
        XCTAssertEqual(dict.entries.count, 2)
        guard case .int(1)? = dict["a"], case .int(2)? = dict["b"] else {
            return XCTFail("expected the two entries")
        }
    }

    func testAFrameOpcodeIsSkipped() throws {
        let root = try load([0x95, 2, 0, 0, 0, 0, 0, 0, 0, 0x4b, 0x2a])
        guard case .int(42) = root else { return XCTFail("expected 42, got \(root)") }
    }

    func testStackGlobalAndGlobalAgree() throws {
        let viaStack = try load(unicode("torch") + unicode("FloatStorage") + [0x93])
        let viaLines = try load(Array("ctorch\nFloatStorage\n".utf8))
        for root in [viaStack, viaLines] {
            guard case .global(module: "torch", name: "FloatStorage") = root else {
                return XCTFail("expected the torch.FloatStorage global, got \(root)")
            }
        }
    }

    func testAnOrderedDictConstructionBehavesAsADict() throws {
        let root = try load(Array("ccollections\nOrderedDict\n".utf8)
                            + [0x29, 0x52,        // EMPTY_TUPLE, REDUCE
                               0x28]              // MARK
                            + unicode("weight") + [0x4b, 0x03]
                            + [0x75,              // SETITEMS
                               0x7d, 0x62])       // EMPTY_DICT, BUILD (a state_dict's _metadata)
        guard case .dict(let dict) = root else { return XCTFail("expected a dict, got \(root)") }
        guard case .int(3)? = dict["weight"] else { return XCTFail("expected the entry") }
        XCTAssertNotNil(dict.attributes)
    }

    func testAnOrderedDictBuiltFromConstructorPairsIsPopulated() throws {
        // An older pickler passes the items as a constructor argument, a list of [key, value]
        // pairs, where a newer one uses SETITEMS; the eccv16 colorizer checkpoint is this form.
        let root = try load(Array("ccollections\nOrderedDict\n".utf8)
                            + [0x5d, 0x28,                       // EMPTY_LIST, MARK
                               0x5d, 0x28]                       // the pair [key, value]
                            + unicode("weight") + [0x4b, 0x09,
                               0x65,                             // APPENDS (the pair)
                               0x65,                             // APPENDS (the outer list)
                               0x85, 0x52])                      // TUPLE1, REDUCE
        guard case .dict(let dict) = root, case .int(9)? = dict["weight"] else {
            return XCTFail("expected the constructor pairs to populate the dict, got \(root)")
        }
    }

    func testAnUnknownGlobalConstructsAnInertNode() throws {
        let root = try load(Array("cmmengine.logging\nHistoryBuffer\n".utf8)
                            + [0x4b, 0x09, 0x85, 0x52,   // (9,), REDUCE
                               0x7d, 0x62])              // EMPTY_DICT, BUILD
        guard case .opaque(let node) = root else { return XCTFail("expected an opaque node, got \(root)") }
        XCTAssertEqual(node.qualifiedName, "mmengine.logging.HistoryBuffer")
        XCTAssertEqual(node.args.count, 1)
        XCTAssertNotNil(node.state)
    }

    func testNewobjConstructsLikeReduce() throws {
        let root = try load(Array("cbuiltins\nset\n".utf8) + [0x29, 0x81])
        guard case .opaque(let node) = root else { return XCTFail("expected an opaque node, got \(root)") }
        XCTAssertEqual(node.qualifiedName, "builtins.set")
    }

    func testBinpersidRoutesThroughPersistentLoad() throws {
        var received: NFKMLXPickleValue?
        let root = try load(unicode("storage") + [0x85, 0x51]) { pid in
            received = pid
            return .int(99)
        }
        guard case .int(99) = root else { return XCTFail("expected the persistent value, got \(root)") }
        guard case .tuple(let pid)? = received, case .string("storage") = pid[0] else {
            return XCTFail("expected the persistent tuple to reach the handler")
        }
    }

    func testARefusedOpcodeNamesItself() throws {
        XCTAssertThrowsError(try load([0x50])) { error in
            let description = (error as? NFKMLXError)?.errorDescription ?? ""
            XCTAssertTrue(description.contains("PERSID"), "expected the refusal to name PERSID: \(description)")
        }
    }

    func testAStackUnderflowThrowsRatherThanCrashing() throws {
        XCTAssertThrowsError(try load([0x61]))            // APPEND with nothing beneath
        XCTAssertThrowsError(try load([0x86]))            // TUPLE2 with one item missing
        XCTAssertThrowsError(try load([0x74]))            // TUPLE with no mark
        XCTAssertThrowsError(try load([0x68, 0x00]))      // BINGET of an unset memo slot
    }

    func testBackToBackPicklesReturnTheirEndOffsets() throws {
        let first = Data([0x80, 0x02, 0x4b, 0x01, 0x2e])
        let second = Data([0x80, 0x02, 0x4b, 0x02, 0x2e])
        let stream = first + second
        let (firstRoot, firstEnd) = try NFKMLXPickle.load(stream) { $0 }
        XCTAssertEqual(firstEnd, first.count)
        let (secondRoot, secondEnd) = try NFKMLXPickle.load(stream, from: firstEnd) { $0 }
        XCTAssertEqual(secondEnd, stream.count)
        guard case .int(1) = firstRoot, case .int(2) = secondRoot else {
            return XCTFail("expected the two roots in order")
        }
    }

    func testAPickleInADataSliceReadsTheSame() throws {
        // Zip entry contents arrive as slices whose indices do not start at zero; offsets stay logical.
        let padded = Data([0xff, 0xff, 0xff]) + Data([0x80, 0x02, 0x4b, 0x2a, 0x2e])
        let slice = padded[3...]
        let (root, end) = try NFKMLXPickle.load(slice) { $0 }
        guard case .int(42) = root else { return XCTFail("expected 42, got \(root)") }
        XCTAssertEqual(end, 5)
    }
}
