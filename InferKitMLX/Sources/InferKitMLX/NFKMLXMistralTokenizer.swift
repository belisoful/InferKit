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

/// Reads a Mistral-family `tokenizer.json` and encodes/decodes with it.
final class NFKMLXMistralTokenizer {
    private let vocabulary: [String: Int]
    private let pieces: [Int: String]
    /// A merge `"left\u{0}right"` mapped to its rank; a lower rank is a higher merge priority.
    private let ranks: [String: Int]
    private let unknownId: Int
    /// The added tokens, matched as literals in the text before the merge.
    private let addedTokens: [String: Int]
    /// The ids of the added tokens flagged `special`, which a display decode leaves out.
    let specialIds: Set<Int>

    private static let metaspace = "\u{2581}"

    /// The id of a token literal (`<s>`, `</s>`, `[INST]`), or nil when neither the vocabulary nor the
    /// added tokens carry it.
    func id(forToken content: String) -> Int? { addedTokens[content] ?? vocabulary[content] }

    convenience init?(directoryURL: URL) {
        self.init(tokenizerJSON: directoryURL.appendingPathComponent("tokenizer.json"))
    }

    init?(tokenizerJSON url: URL) {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any],
              let vocabulary = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any] else { return nil }
        self.vocabulary = vocabulary
        var pieces = [Int: String](minimumCapacity: vocabulary.count)
        for (piece, id) in vocabulary { pieces[id] = piece }
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
        var ranks = [String: Int](minimumCapacity: merges.count)
        for (index, entry) in merges.enumerated() {
            // A merge is `["left", "right"]` in a recent tokenizer.json and `"left right"` in an older one.
            if let pair = entry as? [String], pair.count == 2 {
                ranks[pair[0] + "\u{0}" + pair[1]] = index
            } else if let text = entry as? String, let space = text.firstIndex(of: " ") {
                ranks[String(text[..<space]) + "\u{0}" + String(text[text.index(after: space)...])] = index
            }
        }
        self.ranks = ranks
        unknownId = (model["unk_token"] as? String).flatMap { vocabulary[$0] } ?? 0
    }

    /// The token ids for `text`, with no start or end marker added (the backend prepends `<s>`). An
    /// added token written literally encodes to its id. The metaspace pre-tokenizer's `prepend_scheme:
    /// "first"` puts one leading `▁` before the first plain run.
    func encode(_ text: String) -> [Int] {
        var ids = [Int]()
        var seenText = false
        for segment in segments(of: text) {
            switch segment {
            case .special(let id):
                ids.append(id)
            case .text(let plain):
                ids += encodePlain(plain, prependMetaspace: !seenText)
                seenText = true
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
        var symbols = [String]()
        for scalar in normalized.unicodeScalars {
            let piece = String(scalar)
            if vocabulary[piece] != nil {
                symbols.append(piece)
            } else {
                for byte in Array(piece.utf8) { symbols.append(String(format: "<0x%02X>", byte)) }
            }
        }
        merge(&symbols)
        return symbols.map { vocabulary[$0] ?? unknownId }
    }

    /// The BPE merge loop: repeatedly merge the adjacent pair of highest priority (lowest rank).
    private func merge(_ symbols: inout [String]) {
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0 ..< (symbols.count - 1) {
                if let rank = ranks[symbols[index] + "\u{0}" + symbols[index + 1]], rank < bestRank {
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
