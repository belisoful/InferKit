//
//  NFKMLXZipArchiveTests.swift
//  InferKitMLXTests
//
//  The ZIP reader is exercised against archives built byte by byte in the test, which pins the
//  layout facts a checkpoint depends on: central-directory authority, local extra fields whose
//  length differs from the central copy (PyTorch pads local headers to align storages), zip64
//  sentinels, and deflated entries. Foundation only, so these run under `swift test`.
//

import XCTest
import Compression
@testable import InferKitMLX

final class NFKMLXZipArchiveTests: XCTestCase {

    func testAStoredArchiveRoundTrips() throws {
        var builder = ZipBuilder()
        builder.add(name: "archive/data.pkl", contents: Data([0x80, 0x02, 0x4e, 0x2e]))
        builder.add(name: "archive/data/0", contents: Data([1, 2, 3, 4, 5]), localExtra: Data(repeating: 0, count: 28))
        let archive = builder.finished()

        let entries = try NFKMLXZipArchive.entries(in: archive)
        XCTAssertEqual(entries.map(\.name), ["archive/data.pkl", "archive/data/0"])
        XCTAssertEqual(entries[1].method, .stored)
        XCTAssertEqual(entries[1].uncompressedSize, 5)
        let contents = try NFKMLXZipArchive.contents(of: entries[1], in: archive)
        XCTAssertEqual([UInt8](contents), [1, 2, 3, 4, 5])
    }

    func testAStoredEntryIsASliceNotACopy() throws {
        var builder = ZipBuilder()
        builder.add(name: "a", contents: Data(repeating: 7, count: 64))
        let archive = builder.finished()
        let entry = try NFKMLXZipArchive.entries(in: archive)[0]
        let contents = try NFKMLXZipArchive.contents(of: entry, in: archive)
        // A slice keeps the parent's indices; equality of the bytes plus a non-zero start index is
        // what shows no copy happened.
        XCTAssertEqual(contents.count, 64)
        XCTAssertGreaterThan(contents.startIndex, 0)
    }

    func testADeflatedEntryInflates() throws {
        let payload = Data(Array(repeating: UInt8(0x41), count: 400))
        var compressed = Data(count: payload.count + 64)
        let written = compressed.withUnsafeMutableBytes { destination in
            payload.withUnsafeBytes { source in
                compression_encode_buffer(
                    destination.bindMemory(to: UInt8.self).baseAddress!, payload.count + 64,
                    source.bindMemory(to: UInt8.self).baseAddress!, payload.count,
                    nil, COMPRESSION_ZLIB)
            }
        }
        XCTAssertGreaterThan(written, 0)
        compressed.removeSubrange(written...)

        var builder = ZipBuilder()
        builder.add(name: "packed", contents: payload, method: 8, storedBytes: compressed)
        let archive = builder.finished()
        let entry = try NFKMLXZipArchive.entries(in: archive)[0]
        XCTAssertEqual(entry.method, .deflated)
        let contents = try NFKMLXZipArchive.contents(of: entry, in: archive)
        XCTAssertEqual(contents, payload)
    }

    func testZip64SentinelsResolveThroughTheExtraFieldAndLocator() throws {
        var builder = ZipBuilder()
        builder.add(name: "big", contents: Data([9, 8, 7]), zip64: true)
        let archive = builder.finished(zip64: true)
        let entries = try NFKMLXZipArchive.entries(in: archive)
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries[0].uncompressedSize, 3)
        let contents = try NFKMLXZipArchive.contents(of: entries[0], in: archive)
        XCTAssertEqual([UInt8](contents), [9, 8, 7])
    }

    func testAMissingEndRecordIsMalformed() {
        XCTAssertThrowsError(try NFKMLXZipArchive.entries(in: Data(repeating: 0, count: 100))) { error in
            guard case NFKMLXError.malformedCheckpoint = error else {
                return XCTFail("expected malformedCheckpoint, got \(error)")
            }
        }
    }

    func testATruncatedEntryIsMalformed() throws {
        var builder = ZipBuilder()
        builder.add(name: "a", contents: Data([1, 2, 3]))
        var archive = builder.finished()
        // Corrupt the central entry's compressed size so it claims bytes past the end.
        let signature: [UInt8] = [0x50, 0x4b, 0x01, 0x02]
        let central = archive.range(of: Data(signature))!
        archive.replaceSubrange(central.lowerBound + 20 ..< central.lowerBound + 24,
                                with: [0xff, 0xff, 0x00, 0x00])
        XCTAssertThrowsError(try NFKMLXZipArchive.entries(in: archive))
    }

    func testAnUnknownCompressionMethodIsRefusedAtRead() throws {
        var builder = ZipBuilder()
        builder.add(name: "odd", contents: Data([1]), method: 12)
        let archive = builder.finished()
        let entry = try NFKMLXZipArchive.entries(in: archive)[0]
        XCTAssertEqual(entry.method, .other(12))
        XCTAssertThrowsError(try NFKMLXZipArchive.contents(of: entry, in: archive)) { error in
            guard case NFKMLXError.unsupportedConfiguration = error else {
                return XCTFail("expected unsupportedConfiguration, got \(error)")
            }
        }
    }
}
