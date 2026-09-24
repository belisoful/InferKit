//
//  NFKMLXDeepSeekCache.swift
//  InferKitMLX
//
//  The state a DeepSeek V4.1 decode step carries and a prefill does not.
//

import Foundation
import MLX

/// What one generation carries between steps.
///
/// @discussion Prefill sees a whole sequence at once, so every position it needs is present and the
/// mechanisms reduce to tensor work over a chunk. A decode step sees one token, and five separate
/// pieces of the architecture have to remember what came before it:
///
/// - the sliding window of each layer's own key-value,
/// - the compressed key-value the four source layers publish and every other compressed layer reads,
/// - the index keys those same sources publish for the indexer to score against,
/// - the partial group a compressor has pooled but not yet emitted, which at a ratio above one
///   spans steps, and
/// - the n-gram memory's history of collapsed token ids, so a look-back can cross the boundary
///   between the prompt and what follows it.
///
/// Every one of them is indexed by ABSOLUTE position. `offset` is the number of positions already
/// consumed, which is what the rotary, the window and the group arithmetic are all measured from.
/// Introduced in 0.3.0.
public final class NFKMLXDeepSeekCache {

    /// How many positions the cache already holds.
    public private(set) var offset = 0

    /// Per layer, the most recent `slidingWindow` key-value latents, `[batch, 1, kept, head]`.
    var window: [MLXArray?]
    /// Per source layer, every compressed latent it has published, `[batch, 1, groups, head]`.
    var compressed: [Int: MLXArray] = [:]
    /// Per source layer, the index keys it has published, `[batch, groups, index]`.
    var indexKeys: [Int: MLXArray] = [:]
    /// Per compressor that pools more than one position, the group it has started and not finished.
    /// The reference carries these as `kv_state` and `score_state`: `ratio` slots, or `2 · ratio`
    /// where the groups overlap and the previous window stays parked ahead of the one being filled,
    /// with the unwritten ones scored at negative infinity so they pool to nothing. A V4 indexer's
    /// own compressor parks under ``indexerState(_:)``.
    var pendingValues: [Int: MLXArray] = [:]
    var pendingScores: [Int: MLXArray] = [:]
    /// The collapsed ids the n-gram look-back may still reach back into.
    var engramHistory: MLXArray?
    /// Which compressed positions each layer kept on the step just run, as a mask over positions.
    /// State and CHOICE can disagree — every buffer can match while a different set is selected —
    /// so the choice is recorded rather than inferred.
    public internal(set) var selected: [Int: MLXArray] = [:]
    /// The combined per-position score that choice was ranked from, published for the same reason:
    /// a choice that differs where the scores agree is a degenerate ranking, and one that differs
    /// where the scores do not is a defect. Only the score tells those apart.
    public internal(set) var selectionScores: [Int: MLXArray] = [:]

    /// The target-layer states the draft stack reads, one row per committed position.
    ///
    /// @discussion Collected only when a draft stack is going to read them, because building them
    /// costs a concatenation per chunk that a run without speculation would never look at.
    var draftStates: MLXArray?
    var collectsDraftStates = false

    let configuration: NFKMLXDeepSeekConfiguration

    public init(_ c: NFKMLXDeepSeekConfiguration) {
        configuration = c
        window = Array(repeating: nil, count: c.layerCount)
    }

    /// Everything a step carries, captured so a rejected block can be undone.
    ///
    /// @discussion Speculation needs the cache put back as it was, and putting it back is not the
    /// same as trimming the tail off. The sliding window is a RING: appending the speculative
    /// positions evicted the oldest ones, and no amount of dropping from the end brings those back.
    /// Every buffer here is replaced rather than written through on an append, so capturing them is
    /// a handful of reference copies and restoring them is exact.
    struct Snapshot {
        let offset: Int
        let window: [MLXArray?]
        let compressed: [Int: MLXArray]
        let indexKeys: [Int: MLXArray]
        let pendingValues: [Int: MLXArray]
        let pendingScores: [Int: MLXArray]
        let engramHistory: MLXArray?
        let draftStates: MLXArray?
    }

    func snapshot() -> Snapshot {
        Snapshot(offset: offset, window: window, compressed: compressed, indexKeys: indexKeys,
                 pendingValues: Self.copied(pendingValues),
                 pendingScores: Self.copied(pendingScores),
                 engramHistory: engramHistory, draftStates: draftStates)
    }

    func restore(_ snapshot: Snapshot) {
        offset = snapshot.offset
        window = snapshot.window
        compressed = snapshot.compressed
        indexKeys = snapshot.indexKeys
        // Copied on the way back out too, so one snapshot can be restored more than once: what is
        // put back is written through again by the very next step.
        pendingValues = Self.copied(snapshot.pendingValues)
        pendingScores = Self.copied(snapshot.pendingScores)
        engramHistory = snapshot.engramHistory
        draftStates = snapshot.draftStates
    }

    /// The compressor's parked group, copied.
    ///
    /// @discussion Every other buffer a step carries is REPLACED on an append — a concatenation
    /// binds a new array and leaves the old one alone — so capturing it is a reference copy. These
    /// two are the exception: the compressor parks its group by writing a slot at a time, straight
    /// into the buffer, so a captured reference is a view of the thing it was meant to preserve.
    /// The copy is evaluated here because a lazy one would read that same buffer later and find it
    /// already overwritten.
    private static func copied(_ arrays: [Int: MLXArray]) -> [Int: MLXArray] {
        guard !arrays.isEmpty else { return arrays }
        let copies = arrays.mapValues { ($0 * 1).asType($0.dtype) }
        eval(Array(copies.values))
        return copies
    }

    /// The key a V4 indexer's own compressor parks under and publishes its keys under, which is
    /// the layer's negated so it cannot meet a layer key, and so a snapshot copies it with the rest.
    static func indexerState(_ layer: Int) -> Int { -1 - layer }

    /// Keeps this chunk's target-layer states, which the draft stack reads as its context.
    func rememberDraftStates(_ chunk: MLXArray) {
        draftStates = draftStates.map { concatenated([$0, chunk], axis: 1) } ?? chunk
    }

    /// Advances the cache past a chunk of `length` positions, after the layers have read it.
    func advance(by length: Int) { offset += length }

    /// Appends this step's key-value to a layer's window and returns what the layer may attend to.
    ///
    /// Those are two different things and conflating them is wrong in one direction only. What the
    /// CACHE carries forward is the last `slidingWindow` positions, which is the reference's ring.
    /// What this CHUNK may attend to is everything now available — the ring plus the chunk itself —
    /// because a query early in a multi-token chunk still needs the keys of positions the ring would
    /// have dropped. Trimming before returning silently removed a prompt's own first keys from its
    /// own first queries. The sliding rule is then enforced by the mask, per query, on absolute
    /// positions.
    func window(layer: Int, appending latent: MLXArray) -> MLXArray {
        let available = window[layer].map { concatenated([$0, latent], axis: 2) } ?? latent
        let extent = available.dim(2)
        window[layer] = extent > configuration.slidingWindow
            ? available[0..., 0..., (extent - configuration.slidingWindow)...] : available
        return available
    }

    /// Appends a source layer's newly emitted compressed latents, and returns everything it holds.
    func compressed(layer: Int, appending latents: MLXArray?) -> MLXArray? {
        if let latents {
            compressed[layer] = compressed[layer].map { concatenated([$0, latents], axis: 2) }
                ?? latents
        }
        return compressed[layer]
    }

    /// The same for the index keys a source publishes.
    ///
    /// Published whether or not this step emitted a latent, which is what the reference's own
    /// `compress_kv` does and what its `index_k` does not — see the decode oracle, which corrects
    /// that asymmetry rather than reproducing it.
    func indexKeys(layer: Int, appending keys: MLXArray?) -> MLXArray? {
        if let keys {
            indexKeys[layer] = indexKeys[layer].map { concatenated([$0, keys], axis: 1) } ?? keys
        }
        return indexKeys[layer]
    }

    /// The collapsed ids a look-back of `engramMaxNgramSize - 1` may reach, before this chunk.
    func engramPrefix() -> MLXArray? { engramHistory }

    /// Keeps the tail of the chunk just hashed, which is all a later look-back can reach.
    func rememberEngram(_ ids: MLXArray) {
        let needed = max(configuration.engramMaxNgramSize - 1, 0)
        guard needed > 0 else { return }
        let joined = engramHistory.map { concatenated([$0, ids], axis: 1) } ?? ids
        let extent = joined.dim(1)
        engramHistory = extent > needed ? joined[0..., (extent - needed)...] : joined
    }
}
