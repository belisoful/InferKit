//
//  NFKMLXPrecompiledCharsmap.swift
//  InferKitMLX
//
//  SentencePiece's precompiled character map, the `Precompiled` normalizer of a converted `tokenizer.json`.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation

/// SentencePiece's precompiled normalization map, read from a `tokenizer.json` `Precompiled` normalizer's
/// `precompiled_charsmap`, which holds the SentencePiece model's own.
///
/// @discussion The map is a darts-clone double-array trie over UTF-8 keys, followed by the
/// zero-terminated replacement strings its leaf values index. SentencePiece's normalizer walks the text's
/// bytes: at each position the longest key of the trie that the rest begins with is replaced and skipped,
/// and where no key matches, one character is kept as it is. This type reproduces that walk.
struct NFKMLXPrecompiledCharsmap: Sendable {

    private let units: [UInt32]
    private let normalized: [UInt8]

    /// The map a `precompiled_charsmap` value holds, or nil when it is not base64 or its trie overruns it.
    init?(base64 encoded: String) {
        guard let data = Data(base64Encoded: encoded) else {
            return nil
        }
        self.init(bytes: [UInt8](data))
    }

    /// The map in its stored layout: a little-endian 32-bit trie size in bytes, the trie's 32-bit units,
    /// then the replacement strings.
    init?(bytes: [UInt8]) {
        guard bytes.count >= 4 else {
            return nil
        }
        let size = Int(Self.littleEndianWord(in: bytes, at: 0))
        guard size % 4 == 0, 4 + size <= bytes.count else {
            return nil
        }
        units = stride(from: 4, to: 4 + size, by: 4).map { start in
            Self.littleEndianWord(in: bytes, at: start)
        }
        normalized = Array(bytes[(4 + size)...])
    }

    private static func littleEndianWord(in bytes: [UInt8], at offset: Int) -> UInt32 {
        var word: UInt32 = 0
        for index in 0..<4 {
            word |= UInt32(bytes[offset + index]) << (8 * index)
        }
        return word
    }

    /// `text` normalized as SentencePiece's normalizer does.
    func normalize(_ text: String) -> String {
        let bytes = Array(text.utf8)
        var result: [UInt8] = []
        result.reserveCapacity(bytes.count)
        var position = 0
        while position < bytes.count {
            if let (length, value) = longestPrefix(bytes[position...]) {
                result += replacement(at: value)
                position += length
                continue
            }
            let length = Self.characterLength(leading: bytes[position])
            result += bytes[position ..< min(position + length, bytes.count)]
            position += length
        }
        return String(decoding: result, as: UTF8.self)
    }

    /// The zero-terminated replacement string at `start`.
    private func replacement(at start: Int) -> [UInt8] {
        var end = start
        while end < normalized.count, normalized[end] != 0 {
            end += 1
        }
        return Array(normalized[min(start, end) ..< end])
    }

    /// The byte length and value of the longest trie key that prefixes `key`, the last result of darts-clone's
    /// common prefix search.
    private func longestPrefix(_ key: ArraySlice<UInt8>) -> (length: Int, value: Int)? {
        guard !units.isEmpty else {
            return nil
        }
        var found: (length: Int, value: Int)?
        var position = Int(Self.offset(units[0]))
        for (index, byte) in key.enumerated() {
            position ^= Int(byte)
            guard position < units.count else {
                break
            }
            let unit = units[position]
            guard Self.label(unit) == UInt32(byte) else {
                break
            }
            position ^= Int(Self.offset(unit))
            guard position < units.count else {
                break
            }
            if Self.hasLeaf(unit) {
                found = (index + 1, Int(Self.value(units[position])))
            }
        }
        return found
    }

    /// The length of the UTF-8 sequence `leading` begins.
    private static func characterLength(leading: UInt8) -> Int {
        switch leading {
        case 0xF0...: return 4
        case 0xE0...: return 3
        case 0xC0...: return 2
        default: return 1
        }
    }

    private static func hasLeaf(_ unit: UInt32) -> Bool { (unit >> 8) & 1 == 1 }
    private static func value(_ unit: UInt32) -> UInt32 { unit & ((1 << 31) - 1) }
    private static func label(_ unit: UInt32) -> UInt32 { unit & ((1 << 31) | 0xFF) }
    private static func offset(_ unit: UInt32) -> UInt32 { (unit >> 10) << ((unit & (1 << 9)) >> 6) }
}
