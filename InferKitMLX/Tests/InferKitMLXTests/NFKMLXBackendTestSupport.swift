//
//  NFKMLXBackendTestSupport.swift
//  InferKitMLXTests
//
//  Fixtures the backend tests of the vision-language models and the rerankers share: a byte-level BPE
//  tokenizer whose every id decodes to ASCII text, a decoder whose every step emits token 0, and a solid
//  test image.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
@testable import InferKitMLX

enum NFKMLXBackendTestSupport {

    /// A byte-level BPE tokenizer over `size` ids. Ids 0 … 93 are the printable ASCII characters "!" …
    /// "~", then the space and the newline, then two-character tokens; the ids in `specials` are skipped
    /// there and belong to their literals. Every id decodes to ASCII text, so a generated sequence of any
    /// ids is non-empty text.
    static func tokenizer(size: Int = 500, specials: [String: Int] = [:]) throws -> NFKTokenizer {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXBackendTestSupport-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let printable = (33 ... 126).map { String(UnicodeScalar(UInt8($0))) }
        var tokens = printable + ["Ġ", "Ċ"]
        for first in printable where tokens.count < size {
            for second in printable where tokens.count < size {
                tokens.append(first + second)
            }
        }
        let reserved = Set(specials.values)
        var vocabulary = [String: Int]()
        var id = 0
        for token in tokens {
            while reserved.contains(id) { id += 1 }
            if id >= size { break }
            vocabulary[token] = id
            id += 1
        }
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        return try NFKTokenizer(forManifest: ["tokenizer": ["type": "bpe-bytelevel", "specialTokens": specials]],
                                directory: directory)
    }

    /// A tied decoder whose final norm is zero, so every logit is zero and greedy decoding emits token 0
    /// ("!") at every step until its token limit.
    static func constantDecoder(_ configuration: NFKMLXLanguageConfiguration) throws -> NFKMLXLanguageNet {
        let decoder = NFKMLXLanguage.makeNet(configuration)
        try NFKMLXWeights.apply([("model.norm.weight", MLXArray.zeros([configuration.hiddenSize]))],
                                to: decoder, strict: false)
        return decoder
    }

    /// A `width × height` mid-gray image.
    static func solidImage(width: Int, height: Int, gray: UInt8 = 128) -> CGImage {
        var bytes = [UInt8](repeating: gray, count: width * height * 4)
        let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return context.makeImage()!
    }
}
