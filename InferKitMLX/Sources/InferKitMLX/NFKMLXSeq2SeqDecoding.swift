//
//  NFKMLXSeq2SeqDecoding.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXNN

// Greedy and beam-search decoding over any encoder-decoder that exposes a cached one-step decode. The
// beam search follows transformers' `BeamSearchScorer`: `2 × beams` candidates per step, a finished
// hypothesis bank scored by summed log-probability over length^penalty, and the two stopping rules
// (`early_stopping` on, or no open beam able to beat the worst kept hypothesis).

/// An encoder-decoder a ``NFKMLXSeq2SeqDecoder`` can drive.
///
/// Introduced in InferKit 0.4.0.
public protocol NFKMLXSeq2SeqDecodable: AnyObject {
    associatedtype Cache: AnyObject
    /// Encodes source ids `[1, S]`.
    func encodeSource(_ tokens: MLXArray) -> MLXArray
    func makeDecodingCache() -> Cache
    /// Decoder ids `[B, T]` → logits `[B, T, vocabulary]`, extending the cache by `T` positions.
    func decodeStep(_ tokens: MLXArray, memory: MLXArray, cache: Cache) -> MLXArray
    /// Keeps the batch rows `rows` of the cache, in that order.
    func reorderCache(_ cache: Cache, rows: MLXArray)
}

/// The knobs of a decode.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXSeq2SeqDecoding: Sendable {
    /// Beams to keep; 1 is greedy.
    public var beams: Int
    /// The most tokens to generate after the start token.
    public var maxTokens: Int
    /// Divides a finished hypothesis's summed log-probability by `length^lengthPenalty`.
    public var lengthPenalty: Float
    /// Stops as soon as `beams` hypotheses have finished, as M2M100's generation config asks.
    public var earlyStopping: Bool
    /// The id the decoder starts from.
    public var startToken: Int
    /// The id that ends a hypothesis.
    public var endToken: Int
    /// A token forced as the first generated one (M2M100's target-language marker).
    public var forcedFirstToken: Int?
    /// A token forced as the last one `maxTokens` allows (transformers' `forced_eos_token_id`).
    public var forcedLastToken: Int?
    /// Never repeats an n-gram of this length within a hypothesis, the start token included; 0 allows
    /// any (transformers' `no_repeat_ngram_size`).
    public var noRepeatNgramSize: Int
    /// Ids never generated (Marian's pad).
    public var suppressedTokens: [Int]
    /// Log-softmaxes the scores again after suppression, as Marian's `renormalize_logits` asks.
    public var renormalizes: Bool

    public init(beams: Int = 1, maxTokens: Int = 256, lengthPenalty: Float = 1, earlyStopping: Bool = false,
                startToken: Int, endToken: Int, forcedFirstToken: Int? = nil, forcedLastToken: Int? = nil,
                noRepeatNgramSize: Int = 0, suppressedTokens: [Int] = [], renormalizes: Bool = false) {
        self.beams = beams
        self.maxTokens = maxTokens
        self.lengthPenalty = lengthPenalty
        self.earlyStopping = earlyStopping
        self.startToken = startToken
        self.endToken = endToken
        self.forcedFirstToken = forcedFirstToken
        self.forcedLastToken = forcedLastToken
        self.noRepeatNgramSize = noRepeatNgramSize
        self.suppressedTokens = suppressedTokens
        self.renormalizes = renormalizes
    }
}

/// Runs greedy or beam-search generation.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXSeq2SeqDecoder {

    /// Generates a target sequence for `source` (`[S]` ids). The result excludes the start token and
    /// the end token.
    public static func generate<Model: NFKMLXSeq2SeqDecodable>(_ model: Model, source: [Int],
                                                              decoding: NFKMLXSeq2SeqDecoding) -> [Int] {
        let sourceArray = MLXArray(source.map { Int32($0) }).reshaped([1, source.count])
        let memory = model.encodeSource(sourceArray)
        if decoding.beams <= 1 {
            return greedy(model, memory: memory, decoding: decoding)
        }
        return beamSearch(model, memory: memory, decoding: decoding)
    }

    /// Log-probabilities `[B, V]` for the last position, with the decode's constraints applied.
    /// `sequences` holds each row's tokens so far, the start token first.
    private static func constrainedLogProbabilities(_ logits: MLXArray, sequences: [[Int]], step: Int,
                                                    decoding: NFKMLXSeq2SeqDecoding) -> MLXArray {
        var scores = logSoftmax(logits[0..., -1, 0...], axis: -1)
        let vocabulary = scores.dim(-1)
        let forced = step == 0 ? decoding.forcedFirstToken
            : step == decoding.maxTokens - 1 ? decoding.forcedLastToken : nil
        if let forced, forced >= 0, forced < vocabulary {
            // transformers' forced-token processors replace the scores: the forced token scores 0, so its
            // own log-probability never reaches the hypothesis's sum.
            var only = [Float](repeating: -Float.infinity, count: vocabulary)
            only[forced] = 0
            scores = broadcast(MLXArray(only), to: scores.shape)
        } else {
            let suppressed = decoding.suppressedTokens.filter { $0 >= 0 && $0 < vocabulary }
            let banned = sequences.map { repeatedNgramCompletions($0, size: decoding.noRepeatNgramSize) }
            if !suppressed.isEmpty || banned.contains(where: { !$0.isEmpty }) {
                var penalty = [Float](repeating: 0, count: sequences.count * vocabulary)
                for (row, ids) in banned.enumerated() {
                    for id in suppressed + ids where id < vocabulary { penalty[row * vocabulary + id] = -Float.infinity }
                }
                scores = scores + MLXArray(penalty).reshaped([sequences.count, vocabulary])
            }
        }
        if decoding.renormalizes {
            scores = logSoftmax(scores, axis: -1)
        }
        return scores
    }

    /// The tokens that would complete an n-gram `sequence` already holds: every token that followed an
    /// earlier occurrence of its last `size - 1` tokens.
    static func repeatedNgramCompletions(_ sequence: [Int], size: Int) -> [Int] {
        guard size > 0, sequence.count >= size else { return [] }
        let prefix = sequence.suffix(size - 1)
        var completions = [Int]()
        for start in 0 ... sequence.count - size where sequence[start ..< start + size - 1].elementsEqual(prefix) {
            completions.append(sequence[start + size - 1])
        }
        return completions
    }

    private static func greedy<Model: NFKMLXSeq2SeqDecodable>(_ model: Model, memory: MLXArray,
                                                             decoding: NFKMLXSeq2SeqDecoding) -> [Int] {
        let cache = model.makeDecodingCache()
        var sequence = [decoding.startToken]
        for step in 0 ..< decoding.maxTokens {
            let logits = model.decodeStep(MLXArray([Int32(sequence.last!)]).reshaped([1, 1]), memory: memory, cache: cache)
            let scores = constrainedLogProbabilities(logits, sequences: [sequence], step: step, decoding: decoding)
            let next = scores.argMax(axis: -1).item(Int.self)
            if next == decoding.endToken { break }
            sequence.append(next)
        }
        return Array(sequence.dropFirst())
    }

    private struct Hypothesis {
        var tokens: [Int]
        var score: Float
    }

    private static func beamSearch<Model: NFKMLXSeq2SeqDecodable>(_ model: Model, memory: MLXArray,
                                                                 decoding: NFKMLXSeq2SeqDecoding) -> [Int] {
        let beams = decoding.beams
        let cache = model.makeDecodingCache()
        let repeatedMemory = broadcast(memory, to: [beams, memory.dim(1), memory.dim(2)])
        var sequences = [[Int]](repeating: [decoding.startToken], count: beams)
        var beamScores = [Float](repeating: -1e9, count: beams)
        beamScores[0] = 0
        var finished = [Hypothesis]()
        var lastTokens = MLXArray([Int32](repeating: Int32(decoding.startToken), count: beams)).reshaped([beams, 1])

        func hypothesisScore(sum: Float, generatedLength: Int) -> Float {
            sum / pow(Float(max(generatedLength, 1)), decoding.lengthPenalty)
        }
        func isDone(bestOpenSum: Float, generatedLength: Int) -> Bool {
            guard finished.count >= beams else { return false }
            if decoding.earlyStopping { return true }
            let worstKept = finished.map(\.score).min() ?? -Float.infinity
            return worstKept >= hypothesisScore(sum: bestOpenSum, generatedLength: generatedLength)
        }

        for step in 0 ..< decoding.maxTokens {
            let logits = model.decodeStep(lastTokens, memory: repeatedMemory, cache: cache)
            let scores = constrainedLogProbabilities(logits, sequences: sequences, step: step, decoding: decoding) // [beams, V]
            let vocabulary = scores.dim(-1)
            let joint = (scores + MLXArray(beamScores).reshaped([beams, 1])).reshaped([beams * vocabulary])
            let candidateCount = min(2 * beams, beams * vocabulary)
            let order = argPartition(-joint, kth: candidateCount - 1)[0 ..< candidateCount]
            let candidateScores = joint[order]
            eval(order, candidateScores)
            let flat = order.asArray(Int32.self)
            let flatScores = candidateScores.asArray(Float.self)
            let ranked = zip(flat, flatScores).sorted { $0.1 > $1.1 }

            var nextSequences = [[Int]]()
            var nextScores = [Float]()
            var nextRows = [Int32]()
            var nextTokens = [Int32]()
            for (rank, (index, score)) in ranked.enumerated() {
                let beam = Int(index) / vocabulary
                let token = Int(index) % vocabulary
                if token == decoding.endToken {
                    // A hypothesis ending outside the top `beams` ranks is not kept, as the reference.
                    guard rank < beams else { continue }
                    finished.append(Hypothesis(tokens: Array(sequences[beam].dropFirst()),
                                               score: hypothesisScore(sum: score, generatedLength: step + 1)))
                    continue
                }
                nextSequences.append(sequences[beam] + [token])
                nextScores.append(score)
                nextRows.append(Int32(beam))
                nextTokens.append(Int32(token))
                if nextSequences.count == beams { break }
            }
            // Keep the bank to the best `beams` finished hypotheses.
            if finished.count > beams {
                finished.sort { $0.score > $1.score }
                finished.removeLast(finished.count - beams)
            }
            guard !nextSequences.isEmpty else { break }
            if isDone(bestOpenSum: nextScores.max() ?? -Float.infinity, generatedLength: step + 1) {
                break
            }
            sequences = nextSequences
            beamScores = nextScores
            model.reorderCache(cache, rows: MLXArray(nextRows))
            lastTokens = MLXArray(nextTokens).reshaped([beams, 1])
            if step == decoding.maxTokens - 1 {
                for (sequence, score) in zip(sequences, beamScores) {
                    finished.append(Hypothesis(tokens: Array(sequence.dropFirst()),
                                               score: hypothesisScore(sum: score, generatedLength: step + 1)))
                }
            }
        }
        if finished.isEmpty {
            for (sequence, score) in zip(sequences, beamScores) {
                finished.append(Hypothesis(tokens: Array(sequence.dropFirst()),
                                           score: hypothesisScore(sum: score, generatedLength: max(sequence.count - 1, 1))))
            }
        }
        return finished.max { $0.score < $1.score }?.tokens ?? []
    }
}

extension NFKMLXSeq2SeqNet: NFKMLXSeq2SeqDecodable {
    public func encodeSource(_ tokens: MLXArray) -> MLXArray { encode(tokens) }
    public func makeDecodingCache() -> NFKMLXSeq2SeqCache { makeCache() }
    public func decodeStep(_ tokens: MLXArray, memory: MLXArray, cache: NFKMLXSeq2SeqCache) -> MLXArray {
        decode(tokens, memory: memory, cache: cache)
    }
    public func reorderCache(_ cache: NFKMLXSeq2SeqCache, rows: MLXArray) { cache.reorder(rows) }
}
