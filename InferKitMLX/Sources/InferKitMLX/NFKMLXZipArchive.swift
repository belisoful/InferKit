//
//  NFKMLXZipArchive.swift
//  InferKitMLX
//
//  A central-directory ZIP reader for PyTorch checkpoint archives. Foundation and Compression only,
//  so it holds no MLX and is tested directly under `swift test`.
//

import Foundation
import Compression

/// Reads a ZIP archive's directory and entry contents from an in-memory (typically memory-mapped)
/// `Data`. Sizes and offsets come from the central directory, which stays authoritative when an
/// entry's local header defers them to a trailing data descriptor. A stored entry's contents return
/// as a slice of the backing data, so a multi-gigabyte storage is never copied to be located.
enum NFKMLXZipArchive {

    enum Method: Equatable {
        case stored
        case deflated
        case other(Int)
    }

    struct Entry {
        let name: String
        let method: Method
        /// The entry's raw bytes within the backing data, past the local header's own name and
        /// extra fields. For a deflated entry this is the compressed range.
        let dataRange: Range<Int>
        let uncompressedSize: Int
    }

    /// Parses the archive's central directory. Local headers are read only to skip their own
    /// variable-length name and extra fields, whose lengths may differ from the central copies.
    static func entries(in data: Data) throws -> [Entry] {
        let reader = NFKMLXByteReader(data)
        let eocd = try endOfCentralDirectory(reader)
        var entries: [Entry] = []
        entries.reserveCapacity(eocd.entryCount)
        var offset = eocd.directoryOffset
        for _ in 0 ..< eocd.entryCount {
            guard try reader.u32(offset) == 0x02014b50 else {
                throw NFKMLXError.malformedCheckpoint(
                    "the ZIP central directory entry at offset \(offset) has a bad signature")
            }
            let methodCode = try reader.u16(offset + 10)
            var compressedSize = try reader.u32(offset + 20)
            var uncompressedSize = try reader.u32(offset + 24)
            let nameLength = try reader.u16(offset + 28)
            let extraLength = try reader.u16(offset + 30)
            let commentLength = try reader.u16(offset + 32)
            var localOffset = try reader.u32(offset + 42)
            let name = try reader.string(offset + 46, count: nameLength)

            // The 0x0001 zip64 extra field carries, in order, only the fields whose 32-bit central
            // values are sentinels.
            var extraOffset = offset + 46 + nameLength
            let extraEnd = extraOffset + extraLength
            while extraOffset + 4 <= extraEnd {
                let fieldID = try reader.u16(extraOffset)
                let fieldSize = try reader.u16(extraOffset + 2)
                if fieldID == 0x0001 {
                    var fieldCursor = extraOffset + 4
                    if uncompressedSize == 0xffffffff {
                        uncompressedSize = try reader.u64(fieldCursor); fieldCursor += 8
                    }
                    if compressedSize == 0xffffffff {
                        compressedSize = try reader.u64(fieldCursor); fieldCursor += 8
                    }
                    if localOffset == 0xffffffff {
                        localOffset = try reader.u64(fieldCursor); fieldCursor += 8
                    }
                }
                extraOffset += 4 + fieldSize
            }

            guard try reader.u32(localOffset) == 0x04034b50 else {
                throw NFKMLXError.malformedCheckpoint(
                    "the ZIP local header for \"\(name)\" at offset \(localOffset) has a bad signature")
            }
            let localNameLength = try reader.u16(localOffset + 26)
            let localExtraLength = try reader.u16(localOffset + 28)
            let dataStart = localOffset + 30 + localNameLength + localExtraLength
            guard dataStart + compressedSize <= reader.count else {
                throw NFKMLXError.malformedCheckpoint(
                    "the ZIP entry \"\(name)\" claims \(compressedSize) bytes past the end of the archive")
            }

            let method: Method
            switch methodCode {
            case 0: method = .stored
            case 8: method = .deflated
            default: method = .other(methodCode)
            }
            entries.append(Entry(name: name, method: method,
                                 dataRange: dataStart ..< dataStart + compressedSize,
                                 uncompressedSize: uncompressedSize))
            offset = extraEnd + commentLength
        }
        return entries
    }

    /// Returns an entry's uncompressed contents: a zero-copy slice of the backing data for a stored
    /// entry, an inflated copy for a deflated one.
    static func contents(of entry: Entry, in data: Data) throws -> Data {
        let base = data.startIndex
        let raw = data[base + entry.dataRange.lowerBound ..< base + entry.dataRange.upperBound]
        switch entry.method {
        case .stored:
            return raw
        case .deflated:
            return try inflated(raw, uncompressedSize: entry.uncompressedSize, name: entry.name)
        case .other(let code):
            throw NFKMLXError.unsupportedConfiguration(
                "the ZIP entry \"\(entry.name)\" uses compression method \(code); only stored and deflated entries are supported")
        }
    }

    private struct EndOfCentralDirectory {
        let entryCount: Int
        let directoryOffset: Int
    }

    private static func endOfCentralDirectory(_ reader: NFKMLXByteReader) throws -> EndOfCentralDirectory {
        // The record is 22 bytes plus a comment of up to 64 KB, so it sits in the trailing window.
        let windowStart = max(0, reader.count - 22 - 0xffff)
        var recordOffset = -1
        var offset = reader.count - 22
        while offset >= windowStart {
            if try reader.u32(offset) == 0x06054b50 {
                recordOffset = offset
                break
            }
            offset -= 1
        }
        guard recordOffset >= 0 else {
            throw NFKMLXError.malformedCheckpoint("the data holds no ZIP end-of-central-directory record")
        }

        var entryCount = try reader.u16(recordOffset + 10)
        var directoryOffset = try reader.u32(recordOffset + 16)
        let directorySize = try reader.u32(recordOffset + 12)
        if entryCount == 0xffff || directoryOffset == 0xffffffff || directorySize == 0xffffffff {
            let locatorOffset = recordOffset - 20
            guard locatorOffset >= 0, try reader.u32(locatorOffset) == 0x07064b50 else {
                throw NFKMLXError.malformedCheckpoint(
                    "the ZIP end-of-central-directory record defers to zip64 but the zip64 locator is missing")
            }
            let zip64Offset = try reader.u64(locatorOffset + 8)
            guard try reader.u32(zip64Offset) == 0x06064b50 else {
                throw NFKMLXError.malformedCheckpoint(
                    "the zip64 end-of-central-directory record at offset \(zip64Offset) has a bad signature")
            }
            entryCount = try reader.u64(zip64Offset + 32)
            directoryOffset = try reader.u64(zip64Offset + 48)
        }
        return EndOfCentralDirectory(entryCount: entryCount, directoryOffset: directoryOffset)
    }

    private static func inflated(_ raw: Data, uncompressedSize: Int, name: String) throws -> Data {
        guard uncompressedSize > 0 else { return Data() }
        var destination = Data(count: uncompressedSize)
        let written = destination.withUnsafeMutableBytes { destinationBuffer in
            raw.withUnsafeBytes { sourceBuffer in
                compression_decode_buffer(
                    destinationBuffer.bindMemory(to: UInt8.self).baseAddress!, uncompressedSize,
                    sourceBuffer.bindMemory(to: UInt8.self).baseAddress!, raw.count,
                    nil, COMPRESSION_ZLIB)
            }
        }
        guard written == uncompressedSize else {
            throw NFKMLXError.malformedCheckpoint(
                "the ZIP entry \"\(name)\" inflated to \(written) bytes where its directory entry claims \(uncompressedSize)")
        }
        return destination
    }
}

/// Bounds-checked little-endian reads over a `Data` at absolute offsets, independent of the data's
/// own start index so a slice reads the same as a copy.
struct NFKMLXByteReader {
    private let data: Data
    private let base: Data.Index
    let count: Int

    init(_ data: Data) {
        self.data = data
        self.base = data.startIndex
        self.count = data.count
    }

    func u8(_ offset: Int) throws -> Int {
        guard offset >= 0, offset < count else {
            throw NFKMLXError.malformedCheckpoint("a read at offset \(offset) falls outside the \(count)-byte data")
        }
        return Int(data[base + offset])
    }

    func u16(_ offset: Int) throws -> Int {
        try u8(offset) | (u8(offset + 1) << 8)
    }

    func u32(_ offset: Int) throws -> Int {
        try u16(offset) | (u16(offset + 2) << 16)
    }

    func u64(_ offset: Int) throws -> Int {
        let low = try u32(offset)
        let high = try u32(offset + 4)
        guard high < 0x8000_0000 else {
            throw NFKMLXError.malformedCheckpoint("a 64-bit field at offset \(offset) exceeds Int.max")
        }
        return low | (high << 32)
    }

    func bytes(_ offset: Int, count byteCount: Int) throws -> Data {
        guard offset >= 0, byteCount >= 0, offset + byteCount <= count else {
            throw NFKMLXError.malformedCheckpoint(
                "a read of \(byteCount) bytes at offset \(offset) falls outside the \(count)-byte data")
        }
        return data[base + offset ..< base + offset + byteCount]
    }

    func string(_ offset: Int, count byteCount: Int) throws -> String {
        let raw = try bytes(offset, count: byteCount)
        if let utf8 = String(data: raw, encoding: .utf8) {
            return utf8
        }
        return String(decoding: raw, as: UTF8.self)
    }
}
