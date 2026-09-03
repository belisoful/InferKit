//
//  NFKMLXGGUFTests.swift
//  InferKitMLXTests
//
//  Structural tests for the native GGUF reader over a hand-built minimal file: the header, typed
//  metadata, tensor descriptors, shape reversal, F32 dequantization, and the tolerant handling of a type
//  the reader does not implement. Bit-exact block-quant parity against the `gguf` package on a real
//  model lives in NFKMLXReferenceParityTests.
//

import XCTest
@testable import InferKitMLX

final class NFKMLXGGUFTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "reads an MLXArray; run via xcodebuild")
    }

    /// Builds a tiny GGUF: two F32 tensors (a vector and a 2×3 matrix) and one tensor of an unsupported
    /// type, plus a string and an integer metadata value.
    private func writeMinimalGGUF() throws -> URL {
        var bytes = [UInt8]()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
        func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
        func str(_ s: String) { let b = Array(s.utf8); u64(UInt64(b.count)); bytes.append(contentsOf: b) }
        func f32(_ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { bytes.append(contentsOf: $0) } }

        u32(0x4655_4747)                               // "GGUF"
        u32(3)                                         // version
        u64(3)                                         // tensor count
        u64(2)                                         // metadata count
        // metadata
        str("general.architecture"); u32(8); str("test")           // string
        str("test.count"); u32(4); u32(7)                          // uint32

        // tensor descriptors: name, n_dims, dims (fastest first), type, offset
        str("vec"); u32(1); u64(3); u32(0); u64(0)                 // F32 [3]  → shape [3]
        str("mat"); u32(2); u64(2); u64(3); u32(0); u64(32)        // F32 ne=[2,3] → shape [3, 2]
        str("unknown"); u32(1); u64(4); u32(99); u64(64)          // unsupported type

        // align the data region to 32
        while bytes.count % 32 != 0 { bytes.append(0) }
        let dataStart = bytes.count
        // vec at offset 0
        f32(1); f32(2); f32(3)
        while bytes.count - dataStart < 32 { bytes.append(0) }
        // mat at offset 32 (row-major over shape [3, 2]): 0,1,2,3,4,5
        for value in 0 ..< 6 { f32(Float(value)) }
        while bytes.count - dataStart < 64 { bytes.append(0) }
        // unknown at offset 64: four bytes, never read
        for _ in 0 ..< 16 { bytes.append(0) }

        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".gguf")
        try Data(bytes).write(to: url)
        return url
    }

    func testTheHeaderMetadataAndTensorsParse() throws {
        let url = try writeMinimalGGUF()
        defer { try? FileManager.default.removeItem(at: url) }
        let gguf = try NFKMLXGGUF.gguf(contentsOf: url)

        XCTAssertEqual(gguf.metadataString(forKey: "general.architecture"), "test")
        XCTAssertEqual(gguf.metadataInteger(forKey: "test.count", defaultValue: 0), 7)
        XCTAssertEqual(Set(gguf.tensorNames), ["vec", "mat", "unknown"])
        XCTAssertEqual(gguf.info(forTensor: "vec")?.typeName, "F32")
        XCTAssertEqual(gguf.info(forTensor: "vec")?.shape, [3])
        // GGUF stores the fastest dimension first, so a stored ne=[2,3] is a row-major shape [3, 2].
        XCTAssertEqual(gguf.info(forTensor: "mat")?.shape, [3, 2])
    }

    func testF32TensorsDequantizeToTheirValues() throws {
        try requireMLXRuntime()
        let url = try writeMinimalGGUF()
        defer { try? FileManager.default.removeItem(at: url) }
        let gguf = try NFKMLXGGUF.gguf(contentsOf: url)

        let vec = try XCTUnwrap(gguf.array(forTensor: "vec"))
        XCTAssertEqual(vec.shape, [3])
        XCTAssertEqual(vec.asArray(Float.self), [1, 2, 3])

        let mat = try XCTUnwrap(gguf.array(forTensor: "mat"))
        XCTAssertEqual(mat.shape, [3, 2])
        XCTAssertEqual(mat.asArray(Float.self), [0, 1, 2, 3, 4, 5])
    }

    func testAnUnsupportedTypeIsListedButRefusedOnRead() throws {
        let url = try writeMinimalGGUF()
        defer { try? FileManager.default.removeItem(at: url) }
        let gguf = try NFKMLXGGUF.gguf(contentsOf: url)

        // The tensor is visible, so a consumer sees the whole model.
        XCTAssertEqual(gguf.info(forTensor: "unknown")?.typeName, "TYPE_99")
        // Reading it throws rather than misreading.
        XCTAssertThrowsError(try gguf.array(forTensor: "unknown"))
    }
}
