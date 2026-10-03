//
//  NFKMLXSentencePiece.swift
//  InferKitMLX
//

import Foundation
import InferKit

// A SentencePiece model file (`.model` / `.spm`) read straight from its protobuf, with the two
// segmenters the translation models use: the unigram Viterbi pass (Marian, MADLAD) and the merge-by-
// score BPE (M2M100). The core's NFKUnigramTokenizer takes a converted JSON vocabulary whose ids are the
// piece indices; the translation releases map pieces to ids through their own vocab.json, keep two
// models (Marian's source and target), or add language tokens outside the model, which is why this
// reader keeps the pieces and lets each model own the id mapping.

/// A SentencePiece model: its pieces, scores, and the normalization it expects.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXSentencePieceModel: Sendable {

    /// The segmentation algorithm the model was trained for.
    public enum Kind: Sendable {
        case unigram
        case bpe
    }

    /// The role of a piece, as the model file marks it.
    public enum PieceType: Int, Sendable {
        case normal = 1
        case unknown = 2
        case control = 3
        case userDefined = 4
        case unused = 5
        case byte = 6
    }

    public struct Piece: Sendable {
        public var text: String
        public var score: Float
        public var type: PieceType
    }

    public var pieces: [Piece]
    public var kind: Kind
    /// Whether the normalizer spec names a normalization (`nmt_nfkc`) rather than `identity`.
    public var appliesNFKC: Bool
    /// The spec's precompiled character map, which ``NFKMLXSentencePieceNormalizer`` runs; empty for
    /// `identity`.
    ///
    /// Introduced in InferKit 0.4.0.
    public var precompiledCharsMap: Data
    public var addDummyPrefix: Bool
    public var removeExtraWhitespace: Bool
    /// Whether a space is written as `▁` in the normalized text.
    ///
    /// Introduced in InferKit 0.4.0.
    public var escapeWhitespaces: Bool
    public var byteFallback: Bool
    public var unknownId: Int

    /// Reads a `.model` / `.spm` file.
    public init(contentsOf url: URL) throws {
        try self.init(data: try Data(contentsOf: url))
    }

    /// Parses the serialized `ModelProto`.
    public init(data: Data) throws {
        var pieces = [Piece]()
        var kind = Kind.unigram
        var normalizer = "nmt_nfkc"
        var charsMap = Data()
        var addDummyPrefix = true
        var removeExtraWhitespace = true
        var escapeWhitespaces = true
        var byteFallback = false
        var unknownId = 0
        for (field, value) in try NFKProtobuf.fields(of: data) {
            switch (field, value) {
            case (1, .bytes(let message)):
                var text = ""
                var score: Float = 0
                var type = PieceType.normal
                for (subfield, subvalue) in try NFKProtobuf.fields(of: message) {
                    switch (subfield, subvalue) {
                    case (1, .bytes(let raw)): text = String(decoding: raw, as: UTF8.self)
                    case (2, .fixed32(let bits)): score = Float(bitPattern: bits)
                    case (3, .varint(let raw)): type = PieceType(rawValue: Int(raw)) ?? .normal
                    default: break
                    }
                }
                pieces.append(Piece(text: text, score: score, type: type))
            case (2, .bytes(let trainer)):
                for (subfield, subvalue) in try NFKProtobuf.fields(of: trainer) {
                    switch (subfield, subvalue) {
                    case (3, .varint(let raw)): kind = raw == 2 ? .bpe : .unigram
                    case (35, .varint(let raw)): byteFallback = raw != 0
                    case (40, .varint(let raw)): unknownId = Int(raw)
                    default: break
                    }
                }
            case (3, .bytes(let spec)):
                for (subfield, subvalue) in try NFKProtobuf.fields(of: spec) {
                    switch (subfield, subvalue) {
                    case (1, .bytes(let raw)): normalizer = String(decoding: raw, as: UTF8.self)
                    case (2, .bytes(let raw)): charsMap = raw
                    case (3, .varint(let raw)): addDummyPrefix = raw != 0
                    case (4, .varint(let raw)): removeExtraWhitespace = raw != 0
                    case (5, .varint(let raw)): escapeWhitespaces = raw != 0
                    default: break
                    }
                }
            default:
                break
            }
        }
        guard !pieces.isEmpty else {
            throw NFKMLXError.malformedCheckpoint("the SentencePiece model holds no pieces")
        }
        self.pieces = pieces
        self.kind = kind
        self.appliesNFKC = normalizer != "identity"
        self.precompiledCharsMap = charsMap
        self.addDummyPrefix = addDummyPrefix
        self.removeExtraWhitespace = removeExtraWhitespace
        self.escapeWhitespaces = escapeWhitespaces
        self.byteFallback = byteFallback
        self.unknownId = unknownId
    }
}

/// A minimal protobuf wire-format reader: enough to walk `ModelProto`.
enum NFKProtobuf {
    enum Value {
        case varint(UInt64)
        case fixed32(UInt32)
        case fixed64(UInt64)
        case bytes(Data)
    }

    static func fields(of data: Data) throws -> [(Int, Value)] {
        var result = [(Int, Value)]()
        var cursor = data.startIndex
        while cursor < data.endIndex {
            let key = try varint(data, &cursor)
            let field = Int(key >> 3)
            switch key & 7 {
            case 0:
                result.append((field, .varint(try varint(data, &cursor))))
            case 1:
                guard cursor + 8 <= data.endIndex else { throw truncated }
                let value = data[cursor ..< cursor + 8].withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
                cursor += 8
                result.append((field, .fixed64(UInt64(littleEndian: value))))
            case 2:
                let length = Int(try varint(data, &cursor))
                guard cursor + length <= data.endIndex else { throw truncated }
                result.append((field, .bytes(data.subdata(in: cursor ..< cursor + length))))
                cursor += length
            case 5:
                guard cursor + 4 <= data.endIndex else { throw truncated }
                let value = data[cursor ..< cursor + 4].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
                cursor += 4
                result.append((field, .fixed32(UInt32(littleEndian: value))))
            default:
                throw NFKMLXError.malformedCheckpoint("unsupported protobuf wire type \(key & 7)")
            }
        }
        return result
    }

    private static var truncated: NFKMLXError { .malformedCheckpoint("the protobuf message is truncated") }

    private static func varint(_ data: Data, _ cursor: inout Data.Index) throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while true {
            guard cursor < data.endIndex, shift < 64 else { throw truncated }
            let byte = data[cursor]
            cursor += 1
            result |= UInt64(byte & 0x7f) << shift
            if byte & 0x80 == 0 { return result }
            shift += 7
        }
    }
}

/// Text to SentencePiece pieces and back, over one ``NFKMLXSentencePieceModel``.
///
/// @discussion The ids are the piece indices of the model file. A release that numbers its vocabulary
/// differently (Marian, M2M100) maps the pieces through its own table on top of this. Encoding
/// normalizes the way the model's `normalizer_spec` asks (NFKC or identity, extra-whitespace removal,
/// the dummy prefix) and then segments: a unigram model by the Viterbi path of maximal summed piece
/// score, a BPE model by merging the adjacent pair whose joined piece scores highest until no merge is
/// left. A character no piece covers becomes the unknown piece, or its UTF-8 bytes when the model
/// defines byte fallback.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXSentencePieceSegmenter: @unchecked Sendable {
    public let model: NFKMLXSentencePieceModel
    /// Whether the unigram path sums piece scores in double precision. SentencePiece sums in float;
    /// the `tokenizers` library behind a release's `tokenizer.json` sums in double. The two agree
    /// except where two segmentations tie to within float rounding, and then each keeps a different one.
    public let accumulatesInDoublePrecision: Bool
    /// The model's normalizer, from its precompiled character map.
    ///
    /// Introduced in InferKit 0.4.0.
    public let normalizer: NFKMLXSentencePieceNormalizer
    // Pieces are keyed by their exact scalar sequence. A `String` key compares by canonical
    // equivalence, which merges pieces that differ only in combining-mark order or composition;
    // SentencePiece matches bytes, and its vocabularies carry both spellings as distinct pieces.
    private let pieceIds: [[UInt32]: Int]
    private let maxPieceScalars: Int
    private let minScore: Float
    private let byteIds: [Int]

    static let space: Unicode.Scalar = "\u{2581}"

    /// The exact-match key of a piece: its scalar values in order.
    static func key(_ piece: String) -> [UInt32] { piece.unicodeScalars.map(\.value) }

    public init(model: NFKMLXSentencePieceModel, accumulatesInDoublePrecision: Bool = false) {
        self.model = model
        self.accumulatesInDoublePrecision = accumulatesInDoublePrecision
        normalizer = NFKMLXSentencePieceNormalizer(model: model)
        var ids = [[UInt32]: Int]()
        var longest = 1
        var lowest = Float.greatestFiniteMagnitude
        var bytes = [Int](repeating: -1, count: 256)
        for (index, piece) in model.pieces.enumerated() {
            let key = Self.key(piece.text)
            if ids[key] == nil { ids[key] = index }
            longest = max(longest, piece.text.unicodeScalars.count)
            if piece.type == .normal { lowest = min(lowest, piece.score) }
            if piece.type == .byte, let value = Self.byteValue(of: piece.text) { bytes[value] = index }
        }
        pieceIds = ids
        maxPieceScalars = longest
        minScore = lowest == .greatestFiniteMagnitude ? 0 : lowest
        byteIds = bytes
    }

    public convenience init(contentsOf url: URL, accumulatesInDoublePrecision: Bool = false) throws {
        self.init(model: try NFKMLXSentencePieceModel(contentsOf: url),
                  accumulatesInDoublePrecision: accumulatesInDoublePrecision)
    }

    public var pieceCount: Int { model.pieces.count }

    /// The index of `piece` (matched scalar for scalar), or nil when the model has no such piece.
    public func id(of piece: String) -> Int? { pieceIds[Self.key(piece)] }

    /// The piece at `id`, or nil outside the vocabulary.
    public func piece(at id: Int) -> String? {
        guard id >= 0, id < model.pieces.count else { return nil }
        return model.pieces[id].text
    }

    // MARK: Encoding

    /// The model's normalization of `text`: its character map applied, whitespace collapsed and
    /// trimmed where the spec asks for that, every space rewritten as the `▁` marker, and the dummy
    /// prefix prepended when `dummyPrefix` (or, when nil, the spec) asks for it.
    public func normalize(_ text: String, dummyPrefix: Bool? = nil) -> String {
        normalizer.normalize(text, dummyPrefix: dummyPrefix)
    }

    /// Segments `text` into piece ids (the model's own indices).
    public func encode(_ text: String, dummyPrefix: Bool? = nil) -> [Int] {
        let normalized = normalize(text, dummyPrefix: dummyPrefix)
        return encodeNormalized(normalized)
    }

    /// Segments already-normalized text (spaces as `▁`).
    public func encodeNormalized(_ normalized: String) -> [Int] {
        segmentsNormalized(normalized).map(\.id)
    }

    /// One entry per piece of `text`: its model id and the text it covers. An unknown run's id is
    /// the unknown id and its surface the characters it fused, which is what SentencePiece reports
    /// and what a release table may still name.
    ///
    /// Introduced in InferKit 0.4.0.
    public func segments(of text: String, dummyPrefix: Bool? = nil) -> [(id: Int, surface: String)] {
        segmentsNormalized(normalize(text, dummyPrefix: dummyPrefix))
    }

    func segmentsNormalized(_ normalized: String) -> [(id: Int, surface: String)] {
        let scalars = Array(normalized.unicodeScalars)
        guard !scalars.isEmpty else { return [] }
        switch model.kind {
        case .unigram: return viterbi(scalars)
        case .bpe: return mergeByScore(scalars)
        }
    }

    /// The pieces for `text`, as strings.
    public func pieces(for text: String, dummyPrefix: Bool? = nil) -> [String] {
        encode(text, dummyPrefix: dummyPrefix).compactMap { piece(at: $0) }
    }

    private func viterbi(_ scalars: [Unicode.Scalar]) -> [(id: Int, surface: String)] {
        let count = scalars.count
        let wide = accumulatesInDoublePrecision
        // In float mode every sum is rounded to float, so a stored best is exactly the float it was.
        func add(_ base: Double, _ score: Float) -> Double { wide ? base + Double(score) : Double(Float(base) + score) }
        let unknownScore = minScore - 10           // sentencepiece's kUnkPenalty
        var best = [Double](repeating: -.greatestFiniteMagnitude, count: count + 1)
        var backLength = [Int](repeating: 0, count: count + 1)
        var backId = [Int](repeating: model.unknownId, count: count + 1)
        best[0] = 0
        for start in 0 ..< count {
            let base = best[start]
            guard base > -.greatestFiniteMagnitude else { continue }
            var covered = false
            var candidate = [UInt32]()
            candidate.reserveCapacity(maxPieceScalars)
            for length in 1 ... min(maxPieceScalars, count - start) {
                candidate.append(scalars[start + length - 1].value)
                guard let id = pieceIds[candidate] else { continue }
                let piece = model.pieces[id]
                guard piece.type == .normal || piece.type == .userDefined else { continue }
                if length == 1 { covered = true }
                let score = add(base, piece.score)
                if score > best[start + length] {
                    best[start + length] = score
                    backLength[start + length] = length
                    backId[start + length] = id
                }
            }
            if !covered {
                let score = add(base, unknownScore)
                if score > best[start + 1] {
                    best[start + 1] = score
                    backLength[start + 1] = 1
                    backId[start + 1] = -1
                }
            }
        }
        var segments = [(id: Int, start: Int, length: Int)]()
        var cursor = count
        while cursor > 0 {
            let length = backLength[cursor]
            segments.append((backId[cursor], cursor - length, length))
            cursor -= length
        }
        segments.reverse()
        return resolveUnknowns(segments, scalars)
    }

    private func mergeByScore(_ scalars: [Unicode.Scalar]) -> [(id: Int, surface: String)] {
        var symbols = scalars.map { [$0.value] }
        while symbols.count > 1 {
            var bestScore = -Float.greatestFiniteMagnitude
            var bestIndex = -1
            for index in 0 ..< symbols.count - 1 {
                guard let id = pieceIds[symbols[index] + symbols[index + 1]],
                      model.pieces[id].type == .normal else { continue }
                let score = model.pieces[id].score
                if score > bestScore {
                    bestScore = score
                    bestIndex = index
                }
            }
            guard bestIndex >= 0 else { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }
        var segments = [(id: Int, start: Int, length: Int)]()
        var position = 0
        for symbol in symbols {
            let length = symbol.count
            let id = pieceIds[symbol].flatMap { model.pieces[$0].type == .normal ? $0 : nil } ?? -1
            segments.append((id, position, length))
            position += length
        }
        return resolveUnknowns(segments, scalars)
    }

    /// Turns the unknown markers of a segmentation into the unknown id, or into byte pieces where
    /// the model falls back to bytes. Consecutive unknown characters merge into one unknown piece
    /// whose surface is the text they cover.
    private func resolveUnknowns(_ segments: [(id: Int, start: Int, length: Int)],
                                 _ scalars: [Unicode.Scalar]) -> [(id: Int, surface: String)] {
        var result = [(id: Int, surface: String)]()
        func text(_ segment: (id: Int, start: Int, length: Int)) -> String {
            var view = String.UnicodeScalarView()
            view.append(contentsOf: scalars[segment.start ..< segment.start + segment.length])
            return String(view)
        }
        var previousUnknown = false
        for segment in segments {
            if segment.id >= 0 {
                result.append((segment.id, text(segment)))
                previousUnknown = false
                continue
            }
            if model.byteFallback {
                for byte in text(segment).utf8 {
                    let id = byteIds[Int(byte)] >= 0 ? byteIds[Int(byte)] : model.unknownId
                    result.append((id, model.pieces[id].text))
                }
                previousUnknown = false
            } else if previousUnknown {
                result[result.count - 1].surface += text(segment)
            } else {
                result.append((model.unknownId, text(segment)))
                previousUnknown = true
            }
        }
        return result
    }

    // MARK: Decoding

    /// The text a piece sequence spells: `▁` back to a space, byte pieces reassembled, control and
    /// user-defined pieces dropped, and the dummy prefix's leading space removed.
    public func decode(_ ids: [Int]) -> String {
        decode(pieces: ids.compactMap { piece(at: $0) })
    }

    /// The text pieces given by name spell, as SentencePiece's `DecodePieces`: a piece the model
    /// lacks is written as itself, which is how a release table's extra pieces (another model's
    /// vocabulary merged into one `vocab.json`) come back out.
    ///
    /// Introduced in InferKit 0.4.0.
    public func decode(pieces names: [String]) -> String {
        var output = String.UnicodeScalarView()
        var pendingBytes = [UInt8]()
        func flushBytes() {
            guard !pendingBytes.isEmpty else { return }
            output.append(contentsOf: String(decoding: pendingBytes, as: UTF8.self).unicodeScalars)
            pendingBytes.removeAll()
        }
        for name in names {
            guard let id = id(of: name) else {
                flushBytes()
                output.append(contentsOf: name.unicodeScalars)
                continue
            }
            let piece = model.pieces[id]
            switch piece.type {
            case .byte:
                if let value = Self.byteValue(of: piece.text) { pendingBytes.append(UInt8(value)) }
            case .control, .userDefined, .unused:
                flushBytes()
            case .unknown:
                flushBytes()
                output.append(contentsOf: " ⁇ ".unicodeScalars)
            case .normal:
                flushBytes()
                output.append(contentsOf: piece.text.unicodeScalars)
            }
        }
        flushBytes()
        var text = String(output).replacingOccurrences(of: String(Self.space), with: " ")
        if model.addDummyPrefix, text.hasPrefix(" ") {
            text.removeFirst()
        }
        return text
    }

    // MARK: Normalization

    /// An approximation of SentencePiece's `nmt_nfkc` for a model that names it but carries no
    /// character map (a hand-assembled proto): NFKC, with the characters the NMT rules drop or turn
    /// into spaces. A released model carries its map, and ``normalizer`` runs that instead.
    static func nmtNFKC(_ text: String) -> String {
        var scalars = String.UnicodeScalarView()
        for scalar in text.precomposedStringWithCompatibilityMapping.unicodeScalars {
            switch scalar.value {
            case 0x0000 ... 0x0008, 0x000B, 0x000E ... 0x001F, 0x007F, 0x008F, 0x009F, 0xFEFF, 0xFFF9 ... 0xFFFB, 0x200B ... 0x200E, 0x202A ... 0x202E, 0x2060 ... 0x2064:
                continue
            case 0x0009, 0x000A, 0x000C, 0x000D, 0x00A0, 0x1680, 0x2000 ... 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
                scalars.append(" ")
            default:
                scalars.append(scalar)
            }
        }
        return String(scalars)
    }

    private static func byteValue(of piece: String) -> Int? {
        guard piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">") else { return nil }
        return Int(piece.dropFirst(3).dropLast(), radix: 16)
    }
}

/// SentencePiece's normalizer, run from the model's own `NormalizerSpec`.
///
/// @discussion The spec's precompiled character map is a Darts double-array trie over UTF-8 prefixes,
/// each leaf an offset into a table of NUL-terminated replacements. Normalization takes the longest
/// matching prefix at each position and writes its replacement, copies one character where nothing
/// matches, and passes a user-defined piece through untouched. The map composes the sequences NFKC
/// composes and folds compatibility characters, and it does not reorder combining marks, which is why
/// Foundation's NFKC cannot stand in for it. The spec's whitespace rules then apply: leading and
/// trailing spaces dropped and runs collapsed when `remove_extra_whitespaces` is set, a dummy prefix
/// when `add_dummy_prefix` is, and each space written as `▁` when `escape_whitespaces` is.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXSentencePieceNormalizer: Sendable {
    private let units: [UInt32]
    private let replacements: [UInt8]
    /// User-defined pieces by first byte, longest first; a match copies through unnormalized.
    private let userDefined: [UInt8: [[UInt8]]]
    private let fallsBackToNFKC: Bool
    public let addsDummyPrefix: Bool
    public let removesExtraWhitespace: Bool
    public let escapesWhitespace: Bool

    /// Whether the model carries a character map (every released `nmt_nfkc` model does).
    public var hasCharacterMap: Bool { !units.isEmpty }

    public init(model: NFKMLXSentencePieceModel) {
        let bytes = [UInt8](model.precompiledCharsMap)
        var units = [UInt32]()
        var replacements = [UInt8]()
        if bytes.count >= 4 {
            let trieBytes = Int(bytes[0]) | Int(bytes[1]) << 8 | Int(bytes[2]) << 16 | Int(bytes[3]) << 24
            if trieBytes > 0, trieBytes % 4 == 0, 4 + trieBytes <= bytes.count {
                units.reserveCapacity(trieBytes / 4)
                for unit in 0 ..< trieBytes / 4 {
                    let base = 4 + unit * 4
                    units.append(UInt32(bytes[base]) | UInt32(bytes[base + 1]) << 8
                                 | UInt32(bytes[base + 2]) << 16 | UInt32(bytes[base + 3]) << 24)
                }
                replacements = Array(bytes[(4 + trieBytes)...])
            }
        }
        self.units = units
        self.replacements = replacements
        var table = [UInt8: [[UInt8]]]()
        for piece in model.pieces where piece.type == .userDefined {
            let encoded = Array(piece.text.utf8)
            if let first = encoded.first { table[first, default: []].append(encoded) }
        }
        for key in table.keys { table[key]?.sort { $0.count > $1.count } }
        userDefined = table
        fallsBackToNFKC = units.isEmpty && model.appliesNFKC
        addsDummyPrefix = model.addDummyPrefix
        removesExtraWhitespace = model.removeExtraWhitespace
        escapesWhitespace = model.escapeWhitespaces
    }

    /// The normalized text, `dummyPrefix` overriding the spec's `add_dummy_prefix` when given.
    public func normalize(_ text: String, dummyPrefix: Bool? = nil) -> String {
        let source = fallsBackToNFKC ? NFKMLXSentencePieceSegmenter.nmtNFKC(text) : text
        let input = Array(source.utf8)
        let spaceSymbol: [UInt8] = escapesWhitespace ? [0xE2, 0x96, 0x81] : [0x20]
        var cursor = 0
        if removesExtraWhitespace {
            while cursor < input.count {
                let (piece, consumed) = normalizePrefix(input, at: cursor)
                guard piece == [0x20] else { break }
                cursor += consumed
            }
        }
        guard cursor < input.count else { return "" }
        var output = [UInt8]()
        output.reserveCapacity(input.count + 3)
        if dummyPrefix ?? addsDummyPrefix { output += spaceSymbol }
        var previousWasSpace = removesExtraWhitespace
        while cursor < input.count {
            let (piece, consumed) = normalizePrefix(input, at: cursor)
            var start = 0
            while previousWasSpace, start < piece.count, piece[start] == 0x20 { start += 1 }
            if start < piece.count {
                for byte in piece[start...] {
                    if escapesWhitespace, byte == 0x20 { output += spaceSymbol } else { output.append(byte) }
                }
                previousWasSpace = piece[piece.count - 1] == 0x20
            }
            cursor += consumed
            // Without remove_extra_whitespaces every space survives, so no run is ever collapsed.
            if !removesExtraWhitespace { previousWasSpace = false }
        }
        if removesExtraWhitespace {
            while output.count >= spaceSymbol.count, output[(output.count - spaceSymbol.count)...].elementsEqual(spaceSymbol) {
                output.removeLast(spaceSymbol.count)
            }
        }
        return String(decoding: output, as: UTF8.self)
    }

    /// The replacement for the longest mapped prefix at `start` and the bytes it consumes; one
    /// character copied through when nothing maps; U+FFFD for a malformed byte.
    private func normalizePrefix(_ input: [UInt8], at start: Int) -> (piece: [UInt8], consumed: Int) {
        if let candidates = userDefined[input[start]] {
            for candidate in candidates where start + candidate.count <= input.count
                && input[start ..< start + candidate.count].elementsEqual(candidate) {
                return (candidate, candidate.count)
            }
        }
        var longestLength = 0
        var longestValue = 0
        if !units.isEmpty {
            var position = 0
            var unit = units[0]
            position ^= Int(Self.offset(unit))
            var index = start
            while index < input.count, position < units.count {
                let byte = input[index]
                position ^= Int(byte)
                guard position < units.count else { break }
                unit = units[position]
                guard Self.label(unit) == UInt32(byte) else { break }
                position ^= Int(Self.offset(unit))
                if Self.hasLeaf(unit), position < units.count {
                    longestLength = index - start + 1
                    longestValue = Int(Self.value(units[position]))
                }
                index += 1
            }
        }
        if longestLength == 0 {
            guard let length = Self.utf8Length(input, at: start) else { return ([0xEF, 0xBF, 0xBD], 1) }
            return (Array(input[start ..< start + length]), length)
        }
        var end = longestValue
        while end < replacements.count, replacements[end] != 0 { end += 1 }
        return (Array(replacements[min(longestValue, replacements.count) ..< end]), longestLength)
    }

    // Darts double-array unit fields.
    private static func hasLeaf(_ unit: UInt32) -> Bool { (unit >> 8) & 1 == 1 }
    private static func value(_ unit: UInt32) -> UInt32 { unit & 0x7FFF_FFFF }
    private static func label(_ unit: UInt32) -> UInt32 { unit & (0x8000_0000 | 0xFF) }
    private static func offset(_ unit: UInt32) -> UInt32 { (unit >> 10) << ((unit & (1 << 9)) >> 6) }

    /// The byte length of the well-formed UTF-8 character at `start`, or nil.
    private static func utf8Length(_ input: [UInt8], at start: Int) -> Int? {
        let lead = input[start]
        let length: Int
        if lead < 0x80 { return 1 }
        else if lead & 0xE0 == 0xC0 { length = 2 }
        else if lead & 0xF0 == 0xE0 { length = 3 }
        else if lead & 0xF8 == 0xF0 { length = 4 }
        else { return nil }
        guard start + length <= input.count else { return nil }
        for offset in 1 ..< length where input[start + offset] & 0xC0 != 0x80 { return nil }
        return length
    }
}

/// A SentencePiece tokenizer as the core's `NFKTokenizer`, with an optional piece-to-id table for a
/// release that numbers its vocabulary apart from the model file.
///
/// @discussion `encode:` returns the release ids of the segmented text with no special tokens
/// appended; a model wraps them with its own language and end markers. `decode:` maps release ids back
/// to pieces and spells them out. Ids the table does not name decode to nothing.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXSentencePieceTokenizer)
public final class NFKMLXSentencePieceTokenizer: NFKTokenizer {
    public let segmenter: NFKMLXSentencePieceSegmenter
    // Keyed by exact scalar sequence, as the segmenter's table is; see the note there.
    private let releaseIds: [[UInt32]: Int]?
    private let releasePieces: [Int: String]?
    private let unknownReleaseId: Int
    private let endId: Int
    private let startId: Int

    /// - Parameters:
    ///   - segmenter: the model.
    ///   - vocabulary: the release's piece-to-id table (its `vocab.json`), or nil to use the model's
    ///     own indices.
    ///   - unknownToken: the piece unknown characters map to in the table.
    ///   - eosTokenId: the end id the tokenizer reports.
    ///   - bosTokenId: the start id the tokenizer reports, or -1.
    public convenience init(segmenter: NFKMLXSentencePieceSegmenter, vocabulary: [String: Int]? = nil,
                            unknownToken: String = "<unk>", eosTokenId: Int, bosTokenId: Int = -1) {
        self.init(segmenter: segmenter, vocabularyEntries: vocabulary.map { $0.map { ($0.key, $0.value) } },
                  unknownToken: unknownToken, eosTokenId: eosTokenId, bosTokenId: bosTokenId)
    }

    /// The designated initializer, taking the table as pairs so two pieces that are canonically
    /// equivalent (a composed and a decomposed spelling, two orders of the same combining marks)
    /// stay distinct; a `[String: Int]` has already merged them.
    ///
    /// Introduced in InferKit 0.4.0.
    public init(segmenter: NFKMLXSentencePieceSegmenter, vocabularyEntries: [(piece: String, id: Int)]?,
                unknownToken: String = "<unk>", eosTokenId: Int, bosTokenId: Int = -1) {
        self.segmenter = segmenter
        if let vocabularyEntries {
            var ids = [[UInt32]: Int](minimumCapacity: vocabularyEntries.count)
            var pieces = [Int: String](minimumCapacity: vocabularyEntries.count)
            for (piece, id) in vocabularyEntries {
                let key = NFKMLXSentencePieceSegmenter.key(piece)
                if ids[key] == nil { ids[key] = id }
                if pieces[id] == nil { pieces[id] = piece }
            }
            releaseIds = ids
            releasePieces = pieces
            unknownReleaseId = ids[NFKMLXSentencePieceSegmenter.key(unknownToken)] ?? segmenter.model.unknownId
        } else {
            releaseIds = nil
            releasePieces = nil
            unknownReleaseId = segmenter.model.unknownId
        }
        endId = eosTokenId
        startId = bosTokenId
        super.init()
    }

    /// Reads a model file and an optional `vocab.json`.
    public convenience init(modelURL: URL, vocabularyURL: URL? = nil, unknownToken: String = "<unk>",
                            eosTokenId: Int, bosTokenId: Int = -1) throws {
        var entries: [(piece: String, id: Int)]?
        if let vocabularyURL {
            entries = try Self.vocabularyEntries(contentsOf: vocabularyURL)
        }
        self.init(segmenter: try NFKMLXSentencePieceSegmenter(contentsOf: modelURL), vocabularyEntries: entries,
                  unknownToken: unknownToken, eosTokenId: eosTokenId, bosTokenId: bosTokenId)
    }

    /// A `vocab.json` as pairs, every key kept: read through `NSDictionary`, whose keys compare
    /// literally, so a table that spells one piece two canonically equivalent ways keeps both.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func vocabularyEntries(contentsOf url: URL) throws -> [(piece: String, id: Int)] {
        let data = try Data(contentsOf: url)
        guard let table = try JSONSerialization.jsonObject(with: data) as? NSDictionary else {
            throw NFKMLXError.malformedCheckpoint("\(url.lastPathComponent) is not a piece-to-id table")
        }
        var entries = [(piece: String, id: Int)]()
        entries.reserveCapacity(table.count)
        for (key, value) in table {
            guard let piece = key as? String, let id = (value as? NSNumber)?.intValue else {
                throw NFKMLXError.malformedCheckpoint("\(url.lastPathComponent) is not a piece-to-id table")
            }
            entries.append((piece, id))
        }
        return entries
    }

    public override var eosTokenId: Int { endId }
    public override var bosTokenId: Int { startId }

    /// The release id of a piece, or nil when neither the table nor the model names it.
    public func id(ofPiece piece: String) -> Int? {
        if let releaseIds { return releaseIds[NFKMLXSentencePieceSegmenter.key(piece)] }
        return segmenter.id(of: piece)
    }

    /// The piece a release id names, or nil.
    public func piece(ofId id: Int) -> String? {
        if let releasePieces { return releasePieces[id] }
        return segmenter.piece(at: id)
    }

    /// The release ids of `text`'s pieces. A run the model does not know is looked up by the text it
    /// covers, which a table built from two models' vocabularies (OPUS-MT's `vocab.json`) may name;
    /// otherwise it is the unknown id.
    public func encode(_ text: String, dummyPrefix: Bool?) -> [Int] {
        let segments = segmenter.segments(of: text, dummyPrefix: dummyPrefix)
        guard let releaseIds else { return segments.map(\.id) }
        let unknownId = segmenter.model.unknownId
        return segments.map { segment in
            if segment.id == unknownId {
                return releaseIds[NFKMLXSentencePieceSegmenter.key(segment.surface)] ?? unknownReleaseId
            }
            guard let piece = segmenter.piece(at: segment.id),
                  let mapped = releaseIds[NFKMLXSentencePieceSegmenter.key(piece)] else { return unknownReleaseId }
            return mapped
        }
    }

    public override func encode(_ text: String) -> [NSNumber] {
        encode(text, dummyPrefix: nil).map { NSNumber(value: $0) }
    }

    /// The text of release ids; a release piece the model lacks is written as itself.
    public func decode(ids: [Int]) -> String {
        guard releasePieces != nil else { return segmenter.decode(ids) }
        return segmenter.decode(pieces: ids.compactMap { piece(ofId: $0) })
    }

    public override func decode(_ tokenIds: [NSNumber]) -> String {
        decode(ids: tokenIds.map(\.intValue))
    }

    public override func bytes(forTokenId tokenId: Int) -> Data? {
        guard let piece = piece(ofId: tokenId), let modelId = segmenter.id(of: piece),
              segmenter.model.pieces[modelId].type == .normal else { return nil }
        return Data(piece.replacingOccurrences(of: String(NFKMLXSentencePieceSegmenter.space), with: " ").utf8)
    }
}
