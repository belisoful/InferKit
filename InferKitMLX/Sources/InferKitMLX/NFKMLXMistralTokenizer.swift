//
//  NFKMLXMistralTokenizer.swift
//  InferKitMLX
//
//  The Mistral-family byte-fallback BPE tokenizer, read directly from a release's `tokenizer.json`.
//  Codestral-Mamba and the other Mistral releases ship this scheme: a Metaspace pre-tokenizer (space →
//  `▁`) with `prepend_scheme: "first"` (one leading `▁`), a byte-fallback BPE model, and a decoder that
//  turns `▁` back into a space, reassembles `<0xHH>` byte pieces, and strips a single leading space.
//
//  It is the same metaspace-BPE family the core's tokenizer cluster does not cover (its SentencePiece
//  reader is unigram, its BPE reader byte-level). The Gemma reader solved the family for Gemma; this is
//  the Mistral member, differing in the leading-`▁` prepend on encode and the leading-space strip on
//  decode.
//

import Foundation
import InferKit

/// Reads a Mistral-family `tokenizer.json` and encodes/decodes with it.
final class NFKMLXMistralTokenizer {
    // The vocabulary and the merges are keyed by exact scalar sequences: a `String` key compares by
    // canonical equivalence and merges pieces a release keeps apart (Seed-X holds U+0301 and U+0341,
    // which decomposes to it, as two pieces).
    private let vocabulary: [[UInt32]: Int]
    private let pieces: [Int: String]
    /// A merge `left + separator + right` mapped to its rank; a lower rank is a higher merge priority.
    private let ranks: [[UInt32]: Int]
    /// Separates the two sides of a merge key; no scalar takes this value.
    private static let mergeSeparator: UInt32 = 0x11_0000
    private static func key(_ text: String) -> [UInt32] { text.unicodeScalars.map(\.value) }
    private let unknownId: Int
    /// The added tokens, matched as literals in the text before the merge.
    private let addedTokens: [String: Int]
    /// The ids of the added tokens flagged `special`, which a display decode leaves out.
    let specialIds: Set<Int>
    /// Whether every run of text between added tokens takes a leading `▁`, as a `Prepend` normalizer
    /// writes it, in place of the metaspace pre-tokenizer's one `▁` before a run that opens the text.
    let prependsEveryRun: Bool

    private static let metaspace = "\u{2581}"

    /// The id of a token literal (`<s>`, `</s>`, `[INST]`), or nil when neither the vocabulary nor the
    /// added tokens carry it.
    func id(forToken content: String) -> Int? { addedTokens[content] ?? vocabulary[Self.key(content)] }

    convenience init?(directoryURL: URL) {
        self.init(tokenizerJSON: directoryURL.appendingPathComponent("tokenizer.json"))
    }

    convenience init?(tokenizerJSON url: URL) {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        self.init(tokenizerJSON: json)
    }

    init?(tokenizerJSON json: [String: Any], prependsEveryRun: Bool = false) {
        self.prependsEveryRun = prependsEveryRun
        guard let model = json["model"] as? [String: Any],
              let table = model["vocab"] as? NSDictionary,
              let merges = model["merges"] as? [Any] else { return nil }
        // Read through NSDictionary, whose keys compare literally, so both spellings of a piece survive.
        var vocabulary = [[UInt32]: Int](minimumCapacity: table.count)
        var pieces = [Int: String](minimumCapacity: table.count)
        for (key, value) in table {
            guard let piece = key as? String, let id = (value as? NSNumber)?.intValue else { return nil }
            vocabulary[Self.key(piece)] = id
            pieces[id] = piece
        }
        self.vocabulary = vocabulary
        var added = [String: Int]()
        var specials = Set<Int>()
        for entry in (json["added_tokens"] as? [[String: Any]]) ?? [] {
            guard let content = entry["content"] as? String,
                  let id = (entry["id"] as? NSNumber)?.intValue else { continue }
            added[content] = id
            pieces[id] = content
            if (entry["special"] as? NSNumber)?.boolValue ?? false { specials.insert(id) }
        }
        self.pieces = pieces
        addedTokens = added
        specialIds = specials
        var ranks = [[UInt32]: Int](minimumCapacity: merges.count)
        for (index, entry) in merges.enumerated() {
            // A merge is `["left", "right"]` in a recent tokenizer.json and `"left right"` in an older one.
            if let pair = entry as? [String], pair.count == 2 {
                ranks[Self.key(pair[0]) + [Self.mergeSeparator] + Self.key(pair[1])] = index
            } else if let text = entry as? String, let space = text.firstIndex(of: " ") {
                ranks[Self.key(String(text[..<space])) + [Self.mergeSeparator]
                      + Self.key(String(text[text.index(after: space)...]))] = index
            }
        }
        self.ranks = ranks
        unknownId = (model["unk_token"] as? String).flatMap { vocabulary[Self.key($0)] } ?? 0
    }

    /// The token ids for `text`, with no start or end marker added (the backend prepends `<s>`). An
    /// added token written literally encodes to its id. The metaspace pre-tokenizer's `prepend_scheme:
    /// "first"` puts one leading `▁` before a plain run that opens the text, and none before a run that
    /// follows an added token: `<s>[INST]` encodes `[` bare. Under ``prependsEveryRun`` every plain run
    /// takes one, even a run that already opens with a space, as the Llama-style normalizer writes it.
    func encode(_ text: String) -> [Int] {
        var ids = [Int]()
        for (index, segment) in segments(of: text).enumerated() {
            switch segment {
            case .special(let id):
                ids.append(id)
            case .text(let plain) where prependsEveryRun:
                ids += encodePlain(Self.metaspace + plain, prependMetaspace: false)
            case .text(let plain):
                ids += encodePlain(plain, prependMetaspace: index == 0)
            }
        }
        return ids
    }

    /// The text a token-id sequence decodes to: each id's piece, `▁` back to a space, `<0xHH>` pieces
    /// reassembled into bytes, and a single leading space stripped (the release's decoder `Strip`).
    func decode(_ ids: [Int], skipSpecial: Bool = false) -> String {
        var bytes = [UInt8]()
        for id in ids {
            if skipSpecial, specialIds.contains(id) { continue }
            guard let piece = pieces[id] else { continue }
            if piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
               let byte = UInt8(piece.dropFirst(3).dropLast(), radix: 16) {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: piece.replacingOccurrences(of: Self.metaspace, with: " ").utf8)
            }
        }
        var text = String(decoding: bytes, as: UTF8.self)
        if text.hasPrefix(" ") { text.removeFirst() }
        return text
    }

    /// The bytes one id spells: a `<0xHH>` piece is its byte, and any other piece is its text with `▁`
    /// as a space. A piece in no table spells nothing.
    func bytes(forId id: Int) -> Data? {
        guard let piece = pieces[id] else { return nil }
        if piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
           let byte = UInt8(piece.dropFirst(3).dropLast(), radix: 16) {
            return Data([byte])
        }
        return Data(piece.replacingOccurrences(of: Self.metaspace, with: " ").utf8)
    }

    private enum Segment {
        case special(Int)
        case text(String)
    }

    /// Splits the text at every added token written literally in it. Every added token is spelled
    /// `<…>` or `[…]`, so a candidate runs from `<`/`[` to the next `>`/`]`, which keeps the scan linear.
    private func segments(of text: String) -> [Segment] {
        guard !addedTokens.isEmpty else { return [.text(text)] }
        var result = [Segment]()
        var plain = ""
        var index = text.startIndex
        func closer(for open: Character) -> Character? {
            open == "<" ? ">" : (open == "[" ? "]" : nil)
        }
        while index < text.endIndex {
            let character = text[index]
            if let close = closer(for: character),
               let closeIndex = text[index...].firstIndex(of: close),
               let id = addedTokens[String(text[index ... closeIndex])] {
                if !plain.isEmpty { result.append(.text(plain)); plain = "" }
                result.append(.special(id))
                index = text.index(after: closeIndex)
            } else {
                plain.append(character)
                index = text.index(after: index)
            }
        }
        if !plain.isEmpty { result.append(.text(plain)) }
        return result
    }

    private func encodePlain(_ text: String, prependMetaspace: Bool) -> [Int] {
        var normalized = text.replacingOccurrences(of: " ", with: Self.metaspace)
        if prependMetaspace, !normalized.hasPrefix(Self.metaspace) {
            normalized = Self.metaspace + normalized
        }
        // The space split the pre-tokenizer would do is a no-op after normalization, so the whole
        // run is one pre-token. Each character stands alone, or falls back to its UTF-8 bytes.
        var symbols = [[UInt32]]()
        for scalar in normalized.unicodeScalars {
            if vocabulary[[scalar.value]] != nil {
                symbols.append([scalar.value])
            } else {
                for byte in Array(String(scalar).utf8) { symbols.append(Self.key(String(format: "<0x%02X>", byte))) }
            }
        }
        merge(&symbols)
        return symbols.map { vocabulary[$0] ?? unknownId }
    }

    /// The BPE merge loop: repeatedly merge the adjacent pair of highest priority (lowest rank).
    private func merge(_ symbols: inout [[UInt32]]) {
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0 ..< (symbols.count - 1) {
                if let rank = ranks[symbols[index] + [Self.mergeSeparator] + symbols[index + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            if bestIndex < 0 { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }
    }
}

/// A Mistral-family `tokenizer.json` as the core's `NFKTokenizer`, which is how the language loader
/// hands a Metaspace release (Mixtral) to the backends that take one.
final class NFKMLXMetaspaceTokenizer: NFKTokenizer {
    private let reader: NFKMLXMistralTokenizer
    private let endId: Int
    private let startId: Int

    /// - Parameters:
    ///   - tokenizerJSON: the parsed `tokenizer.json`.
    ///   - endToken: the release's end id, or nil to take the `</s>` the file declares.
    ///   - prependsEveryRun: as ``NFKMLXMistralTokenizer/prependsEveryRun``.
    init?(tokenizerJSON: [String: Any], endToken: Int?, prependsEveryRun: Bool = false) {
        guard let reader = NFKMLXMistralTokenizer(tokenizerJSON: tokenizerJSON, prependsEveryRun: prependsEveryRun) else {
            return nil
        }
        self.reader = reader
        endId = endToken ?? reader.id(forToken: "</s>") ?? -1
        startId = reader.id(forToken: "<s>") ?? -1
        super.init()
    }

    override var eosTokenId: Int { endId }
    override var bosTokenId: Int { startId }

    override func encode(_ text: String) -> [NSNumber] {
        reader.encode(text).map { NSNumber(value: $0) }
    }

    override func decode(_ tokenIds: [NSNumber]) -> String {
        reader.decode(tokenIds.map(\.intValue))
    }

    /// The text of `ids` with the added tokens flagged `special` left out, as `tokenizers` decodes under
    /// `skip_special_tokens`.
    func displayText(_ ids: [Int]) -> String {
        reader.decode(ids, skipSpecial: true)
    }

    override func bytes(forTokenId tokenId: Int) -> Data? {
        reader.bytes(forId: tokenId)
    }
}
