//
//  NFKMLXZipFixtures.swift
//  InferKitMLXTests
//
//  A byte-level ZIP writer shared by the archive and checkpoint fixture tests.
//

import Foundation

/// Writes a minimal ustar tar, for the `.nemo` unwrap fixture: one 512-byte header per member
/// (name, octal size, typeflag '0', ustar magic at offset 257), the body padded to 512, and two
/// trailing zero blocks. No checksum — the reader does not verify it.
struct TarBuilder {
    private var body = Data()

    mutating func add(name: String, contents: Data) {
        var header = [UInt8](repeating: 0, count: 512)
        for (index, byte) in Array(name.utf8).prefix(100).enumerated() {
            header[index] = byte
        }
        let size = String(format: "%011o", contents.count)          // 11 octal digits + NUL
        for (index, byte) in Array(size.utf8).enumerated() {
            header[124 + index] = byte
        }
        header[156] = 0x30                                          // typeflag '0' (regular file)
        for (index, byte) in Array("ustar\u{0}".utf8).enumerated() {
            header[257 + index] = byte
        }
        header[263] = 0x30; header[264] = 0x30                      // version "00"
        body.append(contentsOf: header)
        body.append(contents)
        let padding = (512 - contents.count % 512) % 512
        body.append(Data(repeating: 0, count: padding))
    }

    func finished() -> Data {
        body + Data(repeating: 0, count: 1024)                     // two zero blocks end the archive
    }
}

/// Writes a ZIP archive byte by byte. `storedBytes` is what lands in the file (the compressed form
/// for a deflated entry); `contents` is the uncompressed truth the directory reports. `localExtra`
/// pads the local header the way PyTorch aligns its storages, so the local and central extra
/// lengths differ. `zip64: true` writes the entry's local offset through the 0x0001 extra field and
/// the archive's directory through the zip64 end record.
struct ZipBuilder {
    private var body = Data()
    private var central = Data()
    private var entryCount = 0

    mutating func add(name: String, contents: Data, method: UInt16 = 0,
                      storedBytes: Data? = nil, localExtra: Data = Data(), zip64: Bool = false) {
        let stored = storedBytes ?? contents
        let nameBytes = Array(name.utf8)
        let localOffset = body.count

        body.append(le32: 0x04034b50)
        body.append(le16: 20); body.append(le16: 0)
        body.append(le16: method)
        body.append(le16: 0); body.append(le16: 0)          // time, date
        body.append(le32: 0)                                // CRC (unchecked by design)
        body.append(le32: UInt32(stored.count))
        body.append(le32: UInt32(contents.count))
        body.append(le16: UInt16(nameBytes.count))
        body.append(le16: UInt16(localExtra.count))
        body.append(contentsOf: nameBytes)
        body.append(localExtra)
        body.append(stored)

        var extra = Data()
        if zip64 {
            extra.append(le16: 0x0001)
            extra.append(le16: 8)
            extra.append(le64: UInt64(localOffset))
        }
        central.append(le32: 0x02014b50)
        central.append(le16: 20); central.append(le16: 20); central.append(le16: 0)
        central.append(le16: method)
        central.append(le16: 0); central.append(le16: 0)    // time, date
        central.append(le32: 0)                             // CRC
        central.append(le32: UInt32(stored.count))
        central.append(le32: UInt32(contents.count))
        central.append(le16: UInt16(nameBytes.count))
        central.append(le16: UInt16(extra.count))
        central.append(le16: 0)                             // comment
        central.append(le16: 0); central.append(le16: 0)    // disk, internal attributes
        central.append(le32: 0)                             // external attributes
        central.append(le32: zip64 ? 0xffffffff : UInt32(localOffset))
        central.append(contentsOf: nameBytes)
        central.append(extra)
        entryCount += 1
    }

    func finished(zip64: Bool = false) -> Data {
        var archive = body
        let directoryOffset = archive.count
        archive.append(central)
        if zip64 {
            let zip64Offset = archive.count
            archive.append(le32: 0x06064b50)
            archive.append(le64: 44)                        // record size past this field
            archive.append(le16: 45); archive.append(le16: 45)
            archive.append(le32: 0); archive.append(le32: 0)
            archive.append(le64: UInt64(entryCount)); archive.append(le64: UInt64(entryCount))
            archive.append(le64: UInt64(central.count))
            archive.append(le64: UInt64(directoryOffset))
            archive.append(le32: 0x07064b50)
            archive.append(le32: 0)
            archive.append(le64: UInt64(zip64Offset))
            archive.append(le32: 1)
        }
        archive.append(le32: 0x06054b50)
        archive.append(le16: 0); archive.append(le16: 0)
        archive.append(le16: zip64 ? 0xffff : UInt16(entryCount))
        archive.append(le16: zip64 ? 0xffff : UInt16(entryCount))
        archive.append(le32: zip64 ? 0xffffffff : UInt32(central.count))
        archive.append(le32: zip64 ? 0xffffffff : UInt32(directoryOffset))
        archive.append(le16: 0)
        return archive
    }
}

extension Data {
    mutating func append(le16 value: UInt16) {
        append(UInt8(value & 0xff)); append(UInt8(value >> 8))
    }
    mutating func append(le32 value: UInt32) {
        for shift in stride(from: 0, to: 32, by: 8) {
            append(UInt8((value >> UInt32(shift)) & 0xff))
        }
    }
    mutating func append(le64 value: UInt64) {
        for shift in stride(from: 0, to: 64, by: 8) {
            append(UInt8((value >> UInt64(shift)) & 0xff))
        }
    }
}
