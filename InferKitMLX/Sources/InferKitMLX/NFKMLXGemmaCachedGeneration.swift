//
//  NFKMLXGemmaCachedGeneration.swift
//  InferKitMLX
//
//  Generation through a Gemma 4 decoder's key-value cache: the prompt prefills the cache in passes, and
//  each new token then runs alone.
//

import Foundation
import MLX

/// The masks of one pass over a Gemma 4 decoder.
enum NFKMLXGemmaCachedPass {
    struct Masks {
        let full: MLXArray?
        let sliding: MLXArray?

        func mask(for kind: NFKMLXGemmaAttentionKind) -> MLXArray? { kind == .full ? full : sliding }
    }

    /// Without a cache, the causal mask for both kinds, a sliding layer narrowing it to its window itself.
    /// With one, each kind's mask over the rows its cache retains: every earlier position for a full layer,
    /// the last `window - 1` for a sliding one (``NFKMLXGemma3Masks``).
    static func masks(length: Int, cache: NFKMLXGemma3Cache?) -> Masks {
        guard let cache else {
            let causal = length > 1 ? NFKMLXLanguageNet.causalMask(length, offset: 0) : nil
            return Masks(full: causal, sliding: causal)
        }
        let made = NFKMLXGemma3Masks.make(length: length, offset: cache.offset, window: cache.window,
                                          blockIds: nil, bidirectional: false)
        return Masks(full: made.full, sliding: made.sliding)
    }
}

/// Generation through a Gemma 4 decoder's cache, which keeps the full-attention layers' keys and values
/// whole and the sliding layers' to their window, as the reference's hybrid cache does.
///
/// @discussion `pass` runs tokens, with their embeddings where a caller supplies them, after the positions
/// the cache holds, writes them to it, and returns the logits `[batch, vocabulary]` after the last.
struct NFKGemmaCachedGeneration {
    let configuration: NFKMLXGemmaConfiguration
    let pass: (_ tokens: MLXArray, _ embeddings: MLXArray?, _ cache: NFKMLXGemma3Cache) -> MLXArray

    /// An empty cache for this decoder.
    func makeCache() -> NFKMLXGemma3Cache {
        NFKMLXGemma3Cache(layerCount: configuration.layerCount, slidingWindow: configuration.slidingWindow)
    }

    /// Fills `cache` with `tokens` and returns the logits `[vocabulary]` after the last.
    ///
    /// @discussion The tokens run in passes of at most `limit`, halved while a pass's attention scores, four
    /// bytes for each head, query, and key it sees, would pass `budget` bytes. A pass also ends at each of
    /// `stops`, positions of the sequence, where `atStop` sees the cache. `embeddings` `[1, count, hidden]`,
    /// where given, are sliced with the tokens. An MLX error ends the passes.
    func prefill(_ tokens: [Int], embeddings: MLXArray? = nil, cache: NFKMLXGemma3Cache,
                 limit: Int = NFKMLXLanguageNet.defaultPrefillChunk, budget: Int = NFKMLXLanguageNet.prefillScoreBudget,
                 stops: [Int] = [], atStop: (NFKMLXGemma3Cache) -> Void = { _ in }) -> MLXArray {
        var start = 0
        var stops = stops.drop { $0 <= cache.offset }
        var logits: MLXArray
        repeat {
            var length = Swift.min(tokens.count - start, limit)
            while length > 1 && configuration.headCount * length * (cache.offset + length) * 4 > budget {
                length = (length + 1) / 2
            }
            if let stop = stops.first {
                length = Swift.min(length, stop - cache.offset)
            }
            let ids = MLXArray(tokens[start ..< start + length].map(Int32.init)).reshaped([1, length])
            logits = pass(ids, embeddings.map { $0[0..., start ..< (start + length)] }, cache)
            start += length
            if cache.offset == stops.first {
                eval(logits)
                atStop(cache)
                stops = stops.dropFirst()
            } else if start < tokens.count {
                eval(logits)                                    // free a pass's activations before the next
            }
        } while start < tokens.count && !NFKMLXErrorScope.hasFailed
        return logits[0]
    }

    /// The logits `[vocabulary]` after one more token, a scalar that may be unevaluated, through `cache`.
    func step(_ token: MLXArray, cache: NFKMLXGemma3Cache) -> MLXArray {
        pass(token.reshaped([1, 1]), nil, cache)[0]
    }
}
