//
//  NFKMLXGeneratorBackendTestSupport.swift
//  InferKitMLXTests
//
//  Tokenizers small enough to build in a test, so the generator backends run at a tiny random geometry
//  without a tokenizer from the validation store.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

/// A tokenizer that maps each Unicode scalar to an id below `vocabularySize`.
final class NFKMLXScalarTestTokenizer: NFKTokenizer {
    private let vocabularySize: Int

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        super.init()
    }

    override func encode(_ text: String) -> [NSNumber] {
        text.unicodeScalars.map { NSNumber(value: Int($0.value) % vocabularySize) }
    }

    override func decode(_ tokenIds: [NSNumber]) -> String {
        String(String.UnicodeScalarView(tokenIds.compactMap { Unicode.Scalar($0.uint32Value) }))
    }
}

enum NFKMLXGeneratorBackendTestSupport {

    /// A CLIP vocabulary small enough to write, with its two markers, in a directory removed at teardown.
    static func clipTokenizer(_ testCase: XCTestCase) throws -> NFKMLXSDPromptTokenizer {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("generator-backend-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        testCase.addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let vocabulary: [String: Int] = ["a</w>": 1, "red</w>": 2, "fox</w>": 3,
                                         "<|startoftext|>": 10, "<|endoftext|>": 11]
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        return try NFKMLXSDPromptTokenizer(directoryURL: directory)
    }

    /// A unigram SentencePiece model assembled as its protobuf: the unknown marker and three pieces.
    static func t5Segmenter() throws -> NFKMLXSentencePieceSegmenter {
        func varint(_ value: Int) -> [UInt8] {
            var v = value, out = [UInt8]()
            repeat {
                var byte = UInt8(v & 0x7f)
                v >>= 7
                if v != 0 { byte |= 0x80 }
                out.append(byte)
            } while v != 0
            return out
        }
        func lengthDelimited(field: Int, _ bytes: [UInt8]) -> [UInt8] { varint(field << 3 | 2) + varint(bytes.count) + bytes }
        var body = [UInt8]()
        let pieces: [(String, Float, Int)] = [("<unk>", 0, 2), ("\u{2581}a", -1, 1), ("\u{2581}red", -1, 1),
                                              ("\u{2581}fox", -1, 1)]
        for (text, score, type) in pieces {
            var piece = lengthDelimited(field: 1, Array(text.utf8))
            piece += varint(2 << 3 | 5) + withUnsafeBytes(of: score.bitPattern.littleEndian) { Array($0) }
            piece += varint(3 << 3 | 0) + varint(type)
            body += lengthDelimited(field: 1, piece)
        }
        body += lengthDelimited(field: 2, varint(3 << 3 | 0) + varint(1))
        body += lengthDelimited(field: 3, lengthDelimited(field: 1, Array("nmt_nfkc".utf8)))
        return NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(data: Data(body)))
    }

    /// A `width` × `height` RGB image of random pixels.
    static func image(width: Int, height: Int) throws -> CGImage {
        try NFKMLXImageBridge.cgImage(from: MLXRandom.uniform(low: 0, high: 1, [height, width, 3]),
                                      options: NFKMLXImageOptions())
    }
}
