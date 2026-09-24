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
    /// `nmt_nfkc` normalization (NFKC plus whitespace canonicalization); `identity` leaves text alone.
    public var appliesNFKC: Bool
    public var addDummyPrefix: Bool
    public var removeExtraWhitespace: Bool
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
        var addDummyPrefix = true
        var removeExtraWhitespace = true
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
                    case (3, .varint(let raw)): addDummyPrefix = raw != 0
                    case (4, .varint(let raw)): removeExtraWhitespace = raw != 0
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
        self.addDummyPrefix = addDummyPrefix
        self.removeExtraWhitespace = removeExtraWhitespace
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
    private let pieceIds: [String: Int]
    private let maxPieceScalars: Int
    private let minScore: Float
    private let byteIds: [Int]

    static let space: Unicode.Scalar = "\u{2581}"

    public init(model: NFKMLXSentencePieceModel, accumulatesInDoublePrecision: Bool = false) {
        self.model = model
        self.accumulatesInDoublePrecision = accumulatesInDoublePrecision
        var ids = [String: Int]()
        var longest = 1
        var lowest = Float.greatestFiniteMagnitude
        var bytes = [Int](repeating: -1, count: 256)
        for (index, piece) in model.pieces.enumerated() {
            if ids[piece.text] == nil { ids[piece.text] = index }
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

    /// The index of `piece`, or nil when the model has no such piece.
    public func id(of piece: String) -> Int? { pieceIds[piece] }

    /// The piece at `id`, or nil outside the vocabulary.
    public func piece(at id: Int) -> String? {
        guard id >= 0, id < model.pieces.count else { return nil }
        return model.pieces[id].text
    }

    // MARK: Encoding

    /// The model's normalization of `text`: NFKC where the model asks for it, whitespace collapsed and
    /// trimmed where it asks for that, every space rewritten as the `▁` marker, and the dummy prefix
    /// prepended when `dummyPrefix` is set.
    public func normalize(_ text: String, dummyPrefix: Bool? = nil) -> String {
        var normalized = model.appliesNFKC ? Self.nmtNFKC(text) : text
        if model.removeExtraWhitespace {
            normalized = Self.collapseWhitespace(normalized)
        }
        if dummyPrefix ?? model.addDummyPrefix, !normalized.isEmpty {
            normalized = " " + normalized
        }
        return normalized.replacingOccurrences(of: " ", with: String(Self.space))
    }

    /// Segments `text` into piece ids (the model's own indices).
    public func encode(_ text: String, dummyPrefix: Bool? = nil) -> [Int] {
        let normalized = normalize(text, dummyPrefix: dummyPrefix)
        return encodeNormalized(normalized)
    }

    /// Segments already-normalized text (spaces as `▁`).
    public func encodeNormalized(_ normalized: String) -> [Int] {
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

    private func viterbi(_ scalars: [Unicode.Scalar]) -> [Int] {
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
            for length in 1 ... min(maxPieceScalars, count - start) {
                var candidate = String.UnicodeScalarView()
                candidate.append(contentsOf: scalars[start ..< start + length])
                guard let id = pieceIds[String(candidate)] else { continue }
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

    private func mergeByScore(_ scalars: [Unicode.Scalar]) -> [Int] {
        var symbols = scalars.map { String($0) }
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
            let length = symbol.unicodeScalars.count
            let id = pieceIds[symbol].flatMap { model.pieces[$0].type == .normal ? $0 : nil } ?? -1
            segments.append((id, position, length))
            position += length
        }
        return resolveUnknowns(segments, scalars)
    }

    /// Turns the unknown markers of a segmentation into the unknown id, or into byte pieces where
    /// the model falls back to bytes. Consecutive unknown characters merge into one unknown piece.
    private func resolveUnknowns(_ segments: [(id: Int, start: Int, length: Int)],
                                 _ scalars: [Unicode.Scalar]) -> [Int] {
        var ids = [Int]()
        var previousUnknown = false
        for segment in segments {
            if segment.id >= 0 {
                ids.append(segment.id)
                previousUnknown = false
                continue
            }
            if model.byteFallback {
                var view = String.UnicodeScalarView()
                view.append(contentsOf: scalars[segment.start ..< segment.start + segment.length])
                for byte in String(view).utf8 {
                    ids.append(byteIds[Int(byte)] >= 0 ? byteIds[Int(byte)] : model.unknownId)
                }
                previousUnknown = false
            } else if !previousUnknown {
                ids.append(model.unknownId)
                previousUnknown = true
            }
        }
        return ids
    }

    // MARK: Decoding

    /// The text a piece sequence spells: `▁` back to a space, byte pieces reassembled, control and
    /// user-defined pieces dropped, and the dummy prefix's leading space removed.
    public func decode(_ ids: [Int]) -> String {
        var output = String.UnicodeScalarView()
        var pendingBytes = [UInt8]()
        func flushBytes() {
            guard !pendingBytes.isEmpty else { return }
            output.append(contentsOf: String(decoding: pendingBytes, as: UTF8.self).unicodeScalars)
            pendingBytes.removeAll()
        }
        for id in ids {
            guard id >= 0, id < model.pieces.count else { continue }
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

    /// SentencePiece's `nmt_nfkc`: NFKC, with the characters the NMT rules drop or canonicalize.
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

    /// Trims leading and trailing spaces and collapses runs of spaces to one.
    static func collapseWhitespace(_ text: String) -> String {
        var result = String.UnicodeScalarView()
        var pendingSpace = false
        for scalar in text.unicodeScalars {
            if scalar == " " {
                pendingSpace = !result.isEmpty
                continue
            }
            if pendingSpace {
                result.append(" ")
                pendingSpace = false
            }
            result.append(scalar)
        }
        return String(result)
    }

    private static func byteValue(of piece: String) -> Int? {
        guard piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">") else { return nil }
        return Int(piece.dropFirst(3).dropLast(), radix: 16)
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
    private let releaseIds: [String: Int]?
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
    public init(segmenter: NFKMLXSentencePieceSegmenter, vocabulary: [String: Int]? = nil,
                unknownToken: String = "<unk>", eosTokenId: Int, bosTokenId: Int = -1) {
        self.segmenter = segmenter
        releaseIds = vocabulary
        releasePieces = vocabulary.map { Dictionary($0.map { ($1, $0) }, uniquingKeysWith: { first, _ in first }) }
        unknownReleaseId = vocabulary?[unknownToken] ?? segmenter.model.unknownId
        endId = eosTokenId
        startId = bosTokenId
        super.init()
    }

    /// Reads a model file and an optional `vocab.json`.
    public convenience init(modelURL: URL, vocabularyURL: URL? = nil, unknownToken: String = "<unk>",
                            eosTokenId: Int, bosTokenId: Int = -1) throws {
        var vocabulary: [String: Int]?
        if let vocabularyURL {
            let data = try Data(contentsOf: vocabularyURL)
            guard let table = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
                throw NFKMLXError.malformedCheckpoint("\(vocabularyURL.lastPathComponent) is not a piece-to-id table")
            }
            vocabulary = table
        }
        self.init(segmenter: try NFKMLXSentencePieceSegmenter(contentsOf: modelURL), vocabulary: vocabulary,
                  unknownToken: unknownToken, eosTokenId: eosTokenId, bosTokenId: bosTokenId)
    }

    public override var eosTokenId: Int { endId }
    public override var bosTokenId: Int { startId }

    /// The release id of a piece, or nil when neither the table nor the model names it.
    public func id(ofPiece piece: String) -> Int? {
        if let releaseIds { return releaseIds[piece] }
        return segmenter.id(of: piece)
    }

    /// The piece a release id names, or nil.
    public func piece(ofId id: Int) -> String? {
        if let releasePieces { return releasePieces[id] }
        return segmenter.piece(at: id)
    }

    /// The release ids of `text`'s pieces, unknown pieces mapped to the unknown id.
    public func encode(_ text: String, dummyPrefix: Bool?) -> [Int] {
        let modelIds = segmenter.encode(text, dummyPrefix: dummyPrefix)
        guard let releaseIds else { return modelIds }
        return modelIds.map { id in
            guard let piece = segmenter.piece(at: id), let mapped = releaseIds[piece] else { return unknownReleaseId }
            return mapped
        }
    }

    public override func encode(_ text: String) -> [NSNumber] {
        encode(text, dummyPrefix: nil).map { NSNumber(value: $0) }
    }

    /// The text of release ids.
    public func decode(ids: [Int]) -> String {
        guard releasePieces != nil else { return segmenter.decode(ids) }
        return segmenter.decode(ids.compactMap { piece(ofId: $0).flatMap { segmenter.id(of: $0) } })
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
