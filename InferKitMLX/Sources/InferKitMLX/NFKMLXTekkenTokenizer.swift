//
//  NFKMLXTekkenTokenizer.swift
//  InferKitMLX
//
//  Mistral's "tekken" tokenizer (a tiktoken byte-pair encoder), read from a release's `tekken.json`.
//  The Voxtral speech language model ships one. tekken stores each token as base64 `token_bytes` with a
//  merge `rank`, a set of control/special tokens, and a tiktoken split regex. The token ids the model
//  reads place the special tokens first (`[0, numSpecial)`) and the regular tokens after
//  (`id = numSpecial + rank`).
//
//  Encoding splits text on the regex, then byte-pair merges each piece by rank (the tiktoken algorithm:
//  repeatedly merge the adjacent pair with the lowest rank). Decoding concatenates each id's bytes.
//

import Foundation
import InferKit

/// A tekken (tiktoken) byte-pair tokenizer read from `tekken.json`.
public final class NFKMLXTekkenTokenizer: NFKTokenizer {
    private let rankForBytes: [Data: Int]
    private let bytesForRank: [Int: Data]
    private let numSpecial: Int
    private let specialForString: [String: Int]
    private let regex: NSRegularExpression
    private let end: Int
    private let begin: Int

    /// Reads a tekken tokenizer from a `tekken.json` file.
    public init?(tekkenURL: URL) {
        guard let data = try? Data(contentsOf: tekkenURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let config = json["config"] as? [String: Any],
              let pattern = config["pattern"] as? String,
              let vocab = json["vocab"] as? [[String: Any]],
              let specials = json["special_tokens"] as? [[String: Any]],
              let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let vocabularySize = (config["default_vocab_size"] as? Int) ?? 131_072
        let special = (config["default_num_special_tokens"] as? Int) ?? 1000
        numSpecial = special
        let regularCount = vocabularySize - special

        var bytesRank = [Data: Int](minimumCapacity: regularCount)
        var rankBytes = [Int: Data](minimumCapacity: regularCount)
        for entry in vocab {
            guard let rank = entry["rank"] as? Int, rank < regularCount,
                  let base64 = entry["token_bytes"] as? String,
                  let bytes = Data(base64Encoded: base64) else { continue }
            bytesRank[bytes] = rank
            rankBytes[rank] = bytes
        }
        rankForBytes = bytesRank
        bytesForRank = rankBytes

        var specialMap = [String: Int](minimumCapacity: specials.count)
        for entry in specials {
            guard let rank = entry["rank"] as? Int, let str = entry["token_str"] as? String else { continue }
            specialMap[str] = rank
        }
        specialForString = specialMap
        end = specialMap["</s>"] ?? -1
        begin = specialMap["<s>"] ?? -1
        self.regex = regex
        super.init()
    }

    public override var eosTokenId: Int { end }
    public override var bosTokenId: Int { begin }

    /// The id of a control token by its string, e.g. `[AUDIO]` or `[TRANSCRIBE]`.
    public func specialTokenId(_ string: String) -> Int? { specialForString[string] }

    public override func bytes(forTokenId tokenId: Int) -> Data? {
        tokenId >= numSpecial ? bytesForRank[tokenId - numSpecial] : nil
    }

    public override func encode(_ text: String) -> [NSNumber] {
        var ids = [NSNumber]()
        let ns = text as NSString
        regex.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
            guard let match else { return }
            let piece = ns.substring(with: match.range)
            for rank in bytePairEncode(Array(piece.utf8)) {
                ids.append(NSNumber(value: numSpecial + rank))
            }
        }
        return ids
    }

    /// The tiktoken byte-pair merge over one regex piece: start from single bytes, then repeatedly merge
    /// the adjacent pair whose combined bytes have the lowest rank.
    private func bytePairEncode(_ bytes: [UInt8]) -> [Int] {
        if bytes.isEmpty { return [] }
        if let rank = rankForBytes[Data(bytes)] { return [rank] }
        var parts = bytes.map { Data([$0]) }
        while parts.count > 1 {
            var bestRank = Int.max, bestIndex = -1
            for index in 0 ..< parts.count - 1 {
                if let rank = rankForBytes[parts[index] + parts[index + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            if bestIndex < 0 { break }
            parts[bestIndex].append(parts[bestIndex + 1])
            parts.remove(at: bestIndex + 1)
        }
        return parts.map { rankForBytes[$0] ?? 0 }
    }

    public override func decode(_ tokenIds: [NSNumber]) -> String {
        var bytes = Data()
        for id in tokenIds.map(\.intValue) where id >= numSpecial {
            if let piece = bytesForRank[id - numSpecial] { bytes.append(piece) }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
