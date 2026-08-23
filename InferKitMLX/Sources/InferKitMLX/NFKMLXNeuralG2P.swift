//
//  NFKMLXNeuralG2P.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// A neural grapheme-to-phoneme model: a compact encoder-decoder transformer (reusing the Whisper
// transformer block) that maps a grapheme sequence to a phoneme sequence, trained on a permissive
// dictionary such as CMUdict. It needs no external dependency, so it is the in-toolkit phonemizer path
// for text-to-speech (the espeak path is `NFKMLXEspeakPhonemizer`). Tensors flow channels-last.
//
// The scaffold ships the architecture, a greedy decoder, and a weight-load path; the grapheme/phoneme
// vocabularies are load-time artifacts (a real model supplies its symbol tables). Exact vocab and
// checkpoint key alignment are sweep items.

/// Neural G2P dimensions and special tokens.
public struct NFKMLXG2PConfiguration: Sendable {
    public var graphemeVocab: Int = 40
    public var phonemeVocab: Int = 80
    public var state: Int = 128
    public var heads: Int = 4
    public var encoderLayers: Int = 2
    public var decoderLayers: Int = 2
    public var maxLength: Int = 64
    public var startToken: Int = 0
    public var endToken: Int = 1
    public init() {}
}

/// The encoder-decoder G2P network.
final class NFKMLXG2PNet: Module {
    @ModuleInfo(key: "grapheme_embedding") var graphemeEmbedding: Embedding
    @ModuleInfo(key: "phoneme_embedding") var phonemeEmbedding: Embedding
    @ModuleInfo(key: "encoder") var encoder: [NFKWhisperBlock]
    @ModuleInfo(key: "decoder") var decoder: [NFKWhisperBlock]
    @ModuleInfo(key: "ln") var ln: LayerNorm

    let configuration: NFKMLXG2PConfiguration

    init(_ configuration: NFKMLXG2PConfiguration) {
        self.configuration = configuration
        _graphemeEmbedding.wrappedValue = Embedding(embeddingCount: configuration.graphemeVocab, dimensions: configuration.state)
        _phonemeEmbedding.wrappedValue = Embedding(embeddingCount: configuration.phonemeVocab, dimensions: configuration.state)
        _encoder.wrappedValue = (0 ..< configuration.encoderLayers).map { _ in NFKWhisperBlock(state: configuration.state, heads: configuration.heads, cross: false) }
        _decoder.wrappedValue = (0 ..< configuration.decoderLayers).map { _ in NFKWhisperBlock(state: configuration.state, heads: configuration.heads, cross: true) }
        _ln.wrappedValue = LayerNorm(dimensions: configuration.state)
    }

    func encode(_ graphemes: MLXArray) -> MLXArray {
        var x = graphemeEmbedding(graphemes) + NFKMLXWhisperNet.sinusoids(length: graphemes.shape[1], channels: configuration.state)
        for block in encoder { x = block(x, audio: nil, mask: nil) }
        return x
    }

    /// Greedy grapheme-id sequence → phoneme-id sequence (start token excluded).
    func phonemeIDs(for graphemeIDs: [Int]) -> [Int] {
        guard !graphemeIDs.isEmpty else { return [] }
        let graphemes = MLXArray(graphemeIDs.map { Int32($0) }).reshaped([1, graphemeIDs.count])
        let memory = encode(graphemes)

        var tokens = [configuration.startToken]
        for _ in 0 ..< configuration.maxLength {
            let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
            var x = phonemeEmbedding(tokenArray) + NFKMLXWhisperNet.sinusoids(length: tokens.count, channels: configuration.state)
            let mask = NFKMLXWhisperNet.causalMask(tokens.count)
            for block in decoder { x = block(x, audio: memory, mask: mask) }
            x = ln(x)
            let logits = x.matmul(phonemeEmbedding.weight.transposed(1, 0))
            let next = logits[0, tokens.count - 1].argMax().item(Int.self)
            if next == configuration.endToken { break }
            tokens.append(next)
        }
        return Array(tokens.dropFirst())
    }
}

/// The in-toolkit neural phonemizer: text → grapheme ids → the G2P model → phoneme symbols.
public final class NFKMLXNeuralG2P: NFKMLXPhonemizer, @unchecked Sendable {

    private let net: NFKMLXG2PNet
    private let phonemeSymbols: [String]?

    /// - Parameters:
    ///   - configuration: model dimensions and special tokens.
    ///   - phonemeSymbols: maps a phoneme id to its symbol; when nil, ids are returned as strings.
    public init(configuration: NFKMLXG2PConfiguration = NFKMLXG2PConfiguration(), phonemeSymbols: [String]? = nil) {
        self.net = NFKMLXG2PNet(configuration)
        self.phonemeSymbols = phonemeSymbols
    }

    /// Loads a safetensors checkpoint into the model.
    public func loadWeights(from url: URL, remap: (String) -> String = { $0 }) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in (remap(key), value) }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    public func phonemes(for text: String) -> [String] {
        let graphemeIDs = text.lowercased().unicodeScalars.map(Self.graphemeID)
        return net.phonemeIDs(for: graphemeIDs).map { id in
            if let phonemeSymbols, id >= 0, id < phonemeSymbols.count { return phonemeSymbols[id] }
            return String(id)
        }
    }

    /// The G2P model, for tests and weight round-trips.
    var model: NFKMLXG2PNet { net }

    /// Grapheme id: `a`–`z` → 1–26, space → 27, everything else → 0.
    static func graphemeID(_ scalar: Unicode.Scalar) -> Int {
        if scalar >= "a", scalar <= "z" { return Int(scalar.value - Unicode.Scalar("a").value) + 1 }
        if scalar == " " { return 27 }
        return 0
    }
}
