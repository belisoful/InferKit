//
//  NFKMLXRecurrentConversation.swift
//  InferKitMLX
//
//  Generation through the cache of a decoder whose layers carry a Mamba-2 recurrent state (Mamba-2 and
//  Nemotron-H), and what such a decoder keeps for a conversation between requests.
//

import Foundation
import InferKit
import MLX

/// The caches of one generation over a Mamba-2 or Nemotron-H decoder: each Mamba layer's convolution inputs
/// and scan state, and the attention layers' keys and values in an ``NFKMLXKeyValueCache`` whose slots are
/// those layers' indices.
final class NFKMLXRecurrentCache {
    /// One slot per layer, nil where the layer is not a Mamba layer.
    let mamba: [NFKMLXMamba2Cache?]
    /// The attention layers' keys and values, or nil for a decoder without attention layers.
    let attention: NFKMLXKeyValueCache?
    /// The attention layers' indices.
    let attentionLayers: [Int]
    /// The positions the cache has seen.
    fileprivate(set) var offset = 0

    init(mambaLayers: [Bool], attentionLayers: [Int]) {
        mamba = mambaLayers.map { $0 ? NFKMLXMamba2Cache() : nil }
        self.attentionLayers = attentionLayers
        attention = attentionLayers.isEmpty ? nil : NFKMLXKeyValueCache(layerCount: mambaLayers.count)
    }

    /// Advances the position count once every layer has run a pass of `count` positions.
    func advance(by count: Int) {
        offset += count
        attention?.advance(by: count)
    }

    /// Moves the position count back by `count` and the attention layers' rows with it; the caller restores
    /// the Mamba layers' state for the position it reaches.
    func rollback(by count: Int) -> Bool {
        guard attention?.rollback(by: count, layers: attentionLayers) ?? true else { return false }
        offset -= count
        return true
    }

    /// Whether ``rollback(by:)`` can move the attention layers back `count` positions.
    func canRollback(by count: Int) -> Bool {
        attention?.canRollback(by: count, layers: attentionLayers) ?? true
    }

    /// The bytes the cached arrays occupy.
    var allocatedBytes: Int {
        (attention?.allocatedBytes ?? 0)
            + mamba.reduce(0) { $0 + ($1?.convolutionInputs?.nbytes ?? 0) + ($1?.state?.nbytes ?? 0) }
    }
}

/// Generation through an ``NFKMLXRecurrentCache``: `pass` runs tokens after the positions the cache holds,
/// writes them to it, and returns the logits `[batch, vocabulary]` after the last. `attentionHeads` bounds a
/// prefill pass's attention scores, 0 for a decoder without attention.
struct NFKRecurrentCachedGeneration {
    let mambaLayers: [Bool]
    let attentionLayers: [Int]
    let attentionHeads: Int
    let pass: (_ tokens: MLXArray, _ cache: NFKMLXRecurrentCache) -> MLXArray

    func makeCache() -> NFKMLXRecurrentCache {
        NFKMLXRecurrentCache(mambaLayers: mambaLayers, attentionLayers: attentionLayers)
    }

    /// Fills `cache` with `tokens` and returns the logits `[vocabulary]` after the last.
    ///
    /// @discussion The tokens run in passes of at most `limit`, halved while a pass's attention scores, four
    /// bytes for each head, query, and key it sees, would pass `budget` bytes. A pass also ends at each of
    /// `stops`, positions of the sequence, where `atStop` sees the cache. An MLX error ends the passes.
    func prefill(_ tokens: [Int], cache: NFKMLXRecurrentCache, limit: Int = NFKMLXLanguageNet.defaultPrefillChunk,
                 budget: Int = NFKMLXLanguageNet.prefillScoreBudget, stops: [Int] = [],
                 atStop: (NFKMLXRecurrentCache) -> Void = { _ in }) -> MLXArray {
        var start = 0
        var stops = stops.drop { $0 <= cache.offset }
        var logits: MLXArray
        repeat {
            var length = Swift.min(tokens.count - start, limit)
            while length > 1 && attentionHeads * length * (cache.offset + length) * 4 > budget {
                length = (length + 1) / 2
            }
            if let stop = stops.first {
                length = Swift.min(length, stop - cache.offset)
            }
            logits = pass(MLXArray(tokens[start ..< start + length].map(Int32.init)).reshaped([1, length]), cache)
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
    func step(_ token: MLXArray, cache: NFKMLXRecurrentCache) -> MLXArray {
        pass(token.reshaped([1, 1]), cache)[0]
    }
}

/// A Mamba-2 or Nemotron-H cache kept for one conversation, with snapshots of its Mamba layers' state.
///
/// @discussion The attention layers' keys and values roll back to any position they hold by moving a cursor.
/// A Mamba layer's state holds no earlier position, so the conversation keeps copies of the Mamba layers'
/// convolution inputs and states at chosen positions: the start of the prompt's generation prompt, the
/// prompt's end, where the reply begins, and each multiple of ``snapshotInterval``. A prompt that extends
/// every held token continues the live cache, and otherwise resumes from the newest snapshot inside the
/// shared prefix. A cached pass runs the scan's sequential form, so a resumed prefill computes what a
/// prefill from the start computes, whatever position it resumes at.
final class NFKMLXRecurrentConversation: NFKMLXConversationState {
    static let snapshotInterval = 4096
    static let keptSnapshots = 4

    let mambaLayers: [Bool]
    let attentionLayers: [Int]
    private(set) var cache: NFKMLXRecurrentCache
    private(set) var tokens: [Int] = []
    private(set) var snapshots: [(position: Int, convolutionInputs: [MLXArray?], states: [MLXArray?])] = []
    private(set) var sharedPrefixLength = 0
    private var lastPromptLength = 0
    /// The strengths of the held adapters the cached rows were computed at, or nil for a model that holds none.
    var adapterStrengths: [Float]?

    init(mambaLayers: [Bool], attentionLayers: [Int]) {
        self.mambaLayers = mambaLayers
        self.attentionLayers = attentionLayers
        cache = NFKMLXRecurrentCache(mambaLayers: mambaLayers, attentionLayers: attentionLayers)
    }

    var allocatedBytes: Int {
        cache.allocatedBytes + snapshots.reduce(0) { total, snapshot in
            total + zip(snapshot.convolutionInputs, snapshot.states).reduce(0) { $0 + ($1.0?.nbytes ?? 0) + ($1.1?.nbytes ?? 0) }
        }
    }

    private func sharedLength(with prompt: [Int]) -> Int {
        let limit = Swift.min(tokens.count, Swift.max(prompt.count - 1, 0))
        return zip(tokens[..<limit], prompt).prefix { $0.0 == $0.1 }.count
    }

    /// The position ``resume(for:)`` would continue `prompt` from, and the snapshot it reads.
    private func resumption(for prompt: [Int]) -> (position: Int, snapshot: Int?) {
        let shared = sharedLength(with: prompt)
        if shared == tokens.count {
            return (shared, nil)
        }
        guard let index = snapshots.lastIndex(where: { $0.position <= shared }),
              cache.canRollback(by: tokens.count - snapshots[index].position) else {
            return (0, nil)
        }
        return (snapshots[index].position, index)
    }

    /// Moves the cache to where `prompt` continues and returns that position; the caller prefills the prompt
    /// from there. Where nothing held can be continued, the conversation starts over and returns 0.
    func resume(for prompt: [Int]) -> Int {
        defer { lastPromptLength = prompt.count }
        let (position, snapshot) = resumption(for: prompt)
        guard position > 0 else {
            reset()
            return 0
        }
        if let snapshot {
            _ = cache.rollback(by: tokens.count - position)
            for (index, slot) in cache.mamba.enumerated() {
                slot?.convolutionInputs = snapshots[snapshot].convolutionInputs[index]
                slot?.state = snapshots[snapshot].states[index]
            }
            tokens.removeLast(tokens.count - position)
            snapshots.removeAll { $0.position > position }
        }
        sharedPrefixLength = position
        return position
    }

    func record(_ fed: [Int]) { tokens.append(contentsOf: fed) }

    /// Keeps evaluated copies of the Mamba layers' convolution inputs and states at the position the cache has
    /// reached. An MLX slice shares its source's buffer, and a convolution's inputs are a slice of its whole
    /// pass, so each array is copied rather than referenced.
    func snapshot() {
        guard snapshots.last?.position != cache.offset else { return }
        func copied(_ array: MLXArray?) -> MLXArray? { array.map { ($0 * 1).asType($0.dtype) } }
        let inputs = cache.mamba.map { copied($0?.convolutionInputs) }
        let states = cache.mamba.map { copied($0?.state) }
        eval(inputs.compactMap { $0 } + states.compactMap { $0 })
        snapshots.append((cache.offset, inputs, states))
        if snapshots.count > Self.keptSnapshots {
            snapshots.removeFirst(snapshots.count - Self.keptSnapshots)
        }
    }

    func reset() {
        cache = NFKMLXRecurrentCache(mambaLayers: mambaLayers, attentionLayers: attentionLayers)
        tokens = []
        snapshots = []
        sharedPrefixLength = 0
    }

    /// The positions a prefill from `start` to `end` stops at to snapshot: each multiple of
    /// ``snapshotInterval`` it passes, `turnStart`, where the prompt's generation prompt begins, and `end`.
    static func snapshotPositions(start: Int, end: Int, turnStart: Int?) -> [Int] {
        var positions = Set(stride(from: (start / snapshotInterval + 1) * snapshotInterval, to: end,
                                   by: snapshotInterval))
        positions.insert(end)
        if let turnStart {
            positions.insert(turnStart)
        }
        return positions.filter { $0 > start && $0 <= end }.sorted()
    }
}

extension NFKMLXRecurrentConversation: NFKMLXSpillableConversation {
    var resumablePositions: [Int]? { snapshots.map(\.position) }
    var optionsSignature: String { "recurrent" }
    var promptLength: Int { lastPromptLength > 0 ? lastPromptLength : tokens.count }

    func resumablePosition(for prompt: [Int]) -> Int { resumption(for: prompt).position }

    /// Writes the attention rows, the Mamba layers' live state, and every snapshot to a safetensors file.
    func save(to url: URL) throws {
        var arrays = ["token_count": MLXArray(Int32(tokens.count))]
        for (key, value) in cache.attention?.exportedArrays() ?? [:] { arrays["attention." + key] = value }
        for (index, slot) in cache.mamba.enumerated() {
            if let inputs = slot?.convolutionInputs { arrays["mamba.\(index).convolution"] = inputs }
            if let state = slot?.state { arrays["mamba.\(index).state"] = state }
        }
        for (number, snapshot) in snapshots.enumerated() {
            for index in snapshot.states.indices {
                if let inputs = snapshot.convolutionInputs[index] { arrays["snapshot.\(number).\(index).convolution"] = inputs }
                if let state = snapshot.states[index] { arrays["snapshot.\(number).\(index).state"] = state }
            }
        }
        let metadata = ["dtype": NFKMLXConversationSpill.name(of: cache.attention?.storedDTypeForExport ?? .float32),
                        "snapshots": snapshots.map { String($0.position) }.joined(separator: ",")]
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
    }

    /// Reads what ``save(to:)`` wrote for `tokens`, or nil where the file holds another count.
    static func load(from url: URL, tokens: [Int], mambaLayers: [Bool],
                     attentionLayers: [Int]) throws -> NFKMLXRecurrentConversation? {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        guard arrays["token_count"].map({ Int($0.item(Int32.self)) }) == tokens.count else { return nil }
        let restored = NFKMLXRecurrentConversation(mambaLayers: mambaLayers, attentionLayers: attentionLayers)
        let attention = Dictionary(uniqueKeysWithValues: arrays.filter { $0.key.hasPrefix("attention.") }
            .map { (String($0.key.dropFirst("attention.".count)), $0.value) })
        restored.cache.attention?.restore(arrays: attention, offset: tokens.count,
                                          storedDType: NFKMLXConversationSpill.dtype(named: metadata["dtype"] ?? "float32"))
        restored.cache.offset = tokens.count
        for (index, slot) in restored.cache.mamba.enumerated() {
            slot?.convolutionInputs = arrays["mamba.\(index).convolution"]
            slot?.state = arrays["mamba.\(index).state"]
        }
        let positions = (metadata["snapshots"] ?? "").split(separator: ",").compactMap { Int($0) }
        restored.snapshots = positions.enumerated().map { number, position in
            (position, mambaLayers.indices.map { arrays["snapshot.\(number).\($0).convolution"] },
             mambaLayers.indices.map { arrays["snapshot.\(number).\($0).state"] })
        }
        restored.tokens = tokens
        return restored
    }
}

/// The cached generation run the Mamba-2 and Nemotron-H backends share.
enum NFKMLXRecurrentGeneration {
    /// Generation through `cached` against the conversation `request` names or its prompt continues, or a cache
    /// kept for no later request where it turns reuse off: the prompt resumes from the furthest position the
    /// conversation can, the rest prefills with snapshots at `turnStart`, the prompt's end, and every 4096
    /// positions, and each new token then runs alone. A run that fails discards the conversation. Runs under
    /// the backend's generation lock. Answers the tokens produced and the position the prefill started at.
    static func run(_ promptTokens: [Int], turnStart: Int?, cached: NFKRecurrentCachedGeneration,
                    conversations: NFKMLXConversationKeeper<NFKMLXRecurrentConversation>, request: NFKInferenceRequest,
                    sampling: NFKMLXStepSampling, limit: Int, seed: UInt64?, stops: Set<Int>,
                    afforded: () -> Int, spills: Bool, verify: () throws -> Void) throws -> (produced: [Int], start: Int) {
        let strengths = NFKMLXAdapterRun.strengths
        // A conversation's cache computed at other adapter strengths starts over.
        let (name, state) = conversations.conversation(
            named: NFKMLXConversation.name(of: request), reuses: NFKMLXConversation.reuses(request), prompt: promptTokens,
            signature: "recurrent", usable: { $0.adapterStrengths == strengths },
            make: {
                let made = NFKMLXRecurrentConversation(mambaLayers: cached.mambaLayers, attentionLayers: cached.attentionLayers)
                made.adapterStrengths = strengths
                return made
            },
            read: { url, record in
                try NFKMLXRecurrentConversation.load(from: url, tokens: record.tokens, mambaLayers: cached.mambaLayers,
                                                     attentionLayers: cached.attentionLayers)
            })
        var failed = true
        defer { conversations.finish(name, failed: failed, afforded: afforded(), spills: spills) }
        let start = state.resume(for: promptTokens)
        let fresh = Array(promptTokens[start...])
        let snapshotAt = name == nil ? [] : NFKMLXRecurrentConversation.snapshotPositions(
            start: start, end: promptTokens.count, turnStart: turnStart)
        let logits = cached.prefill(fresh, cache: state.cache, stops: snapshotAt) { _ in state.snapshot() }
        try verify()
        var produced = [Int]()
        var fed = [Int]()
        // Each pass is built on the previous token before that token is read, so the graph of one pass is
        // built while the GPU runs the one before it. A run that stops leaves one pass unread.
        var token = sampling.token(logits, seed: seed, step: 0)
        asyncEval(token)
        for index in 0 ..< limit {
            var following: MLXArray?
            if index + 1 < limit {
                following = sampling.token(cached.step(token, cache: state.cache), seed: seed, step: index + 1)
                asyncEval(following!)
            }
            let next = token.item(Int.self)
            if following != nil {
                fed.append(next)
            }
            try verify()
            if stops.contains(next) || NFKMLXErrorScope.hasFailed { break }
            produced.append(next)
            guard let following else { break }
            token = following
        }
        guard !NFKMLXErrorScope.hasFailed else { return (produced, start) }
        state.record(fresh + fed)
        failed = false
        return (produced, start)
    }

    /// The bytes one run's cache takes over `positions` positions: the attention layers' keys and values, and
    /// the Mamba layers' state and convolution inputs for the live cache and every kept snapshot.
    static func bytes(positions: Int, attentionLayers: Int, keyValueHeads: Int, headDimensions: Int,
                      mamba: NFKMLXMamba2Configuration, mambaLayers: Int, elementBytes: Int) -> Int {
        let keyValues = 2 * attentionLayers * keyValueHeads * headDimensions * positions * elementBytes
        let state = mamba.headCount * mamba.headDimensions * mamba.stateSize * 4
            + (mamba.convolutionKernel - 1) * mamba.convolutionWidth * elementBytes
        return keyValues + (NFKMLXRecurrentConversation.keptSnapshots + 1) * mambaLayers * state
    }
}
