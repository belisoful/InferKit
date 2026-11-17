//
//  NFKMLXGemma4Conversation.swift
//  InferKitMLX
//
//  What the Gemma 4 text backend keeps for a conversation between requests, so a follow-up turn prefills
//  only what it adds.
//

import Foundation
import MLX

/// A Gemma 4 decoder's cache kept for one conversation, with snapshots of its sliding layers' windows.
///
/// @discussion The full-attention layers keep every position, so they roll back to any held position by
/// moving a cursor. The sliding layers keep only their window, so a rollback longer than what they retain
/// past it cannot be served from them. The conversation keeps evaluated copies of the sliding layers' rows
/// at chosen positions: the start of each turn's generation prompt, the prompt's end, where the reply begins,
/// and each multiple of ``snapshotInterval``. A prompt that shares a prefix with the held tokens continues from the live cache
/// where it extends every held token, else rolls the live cache back where the window still reaches, else
/// resumes from the newest snapshot inside the shared prefix. Gemma 4's chat template drops an earlier
/// turn's thinking, so a follow-up parts from the held tokens at the last generation prompt, which a reply
/// shorter than the window rolls back over and a longer one resumes from that turn's snapshot.
final class NFKMLXGemma4Conversation: NFKMLXConversationState {
    /// The spacing of the snapshots a long prefill takes.
    static let snapshotInterval = 4096
    /// The most snapshots a conversation keeps; the oldest go first.
    static let keptSnapshots = 4

    let layerTypes: [NFKMLXGemmaAttentionKind]
    let slidingWindow: Int
    /// The kinds of the layers that store rows: those before the shared tail, which reads its donors' and
    /// stores none, so a rollback moves these alone.
    private let storingKinds: [NFKMLXGemmaAttentionKind]
    private(set) var cache: NFKMLXGemma3Cache
    /// The token ids the cache holds positions for, in order.
    private(set) var tokens: [Int] = []
    /// The sliding layers' rows at earlier positions, oldest first.
    private(set) var snapshots: [(position: Int, rows: [String: MLXArray])] = []
    /// How many of the last resumed prompt's tokens the cache already held.
    private(set) var sharedPrefixLength = 0
    /// The length of the prompt the last run resumed for, or 0 for a conversation read back from disk.
    private var lastPromptLength = 0
    /// The strengths of the held adapters the cached rows were computed at, or nil for a model that holds none.
    var adapterStrengths: [Float]?

    init(layerTypes: [NFKMLXGemmaAttentionKind], slidingWindow: Int, storingLayers: Int) {
        self.layerTypes = layerTypes
        self.slidingWindow = slidingWindow
        storingKinds = Array(layerTypes.prefix(storingLayers))
        cache = NFKMLXGemma3Cache(layerCount: layerTypes.count, slidingWindow: slidingWindow)
    }

    private var fullLayers: [Int] { storingKinds.indices.filter { storingKinds[$0] == .full } }

    var allocatedBytes: Int {
        cache.allocatedBytes + snapshots.reduce(0) { $0 + $1.rows.values.reduce(0) { $0 + $1.nbytes } }
    }

    /// How many leading tokens `prompt` shares with the held ones, at most one short of the prompt.
    private func sharedLength(with prompt: [Int]) -> Int {
        let limit = Swift.min(tokens.count, Swift.max(prompt.count - 1, 0))
        return zip(tokens[..<limit], prompt).prefix { $0.0 == $0.1 }.count
    }

    /// The position ``resume(for:)`` would continue `prompt` from, and whether it reads a snapshot.
    private func resumption(for prompt: [Int]) -> (position: Int, snapshot: Int?) {
        let shared = sharedLength(with: prompt)
        if shared == tokens.count || cache.canRollback(by: tokens.count - shared, kinds: storingKinds) {
            return (shared, nil)
        }
        guard let index = snapshots.lastIndex(where: { $0.position <= shared }),
              cache.full.canRollback(by: tokens.count - snapshots[index].position, layers: fullLayers) else {
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
        let discarded = tokens.count - position
        if let snapshot {
            _ = cache.full.rollback(by: discarded, layers: fullLayers)
            cache.sliding.restore(arrays: snapshots[snapshot].rows, offset: position,
                                  storedDType: cache.sliding.storedDTypeForExport)
        } else {
            _ = cache.rollback(by: discarded, kinds: storingKinds)
        }
        tokens.removeLast(discarded)
        snapshots.removeAll { $0.position > position }
        sharedPrefixLength = position
        return position
    }

    /// Records tokens whose positions a pass has just written.
    func record(_ fed: [Int]) { tokens.append(contentsOf: fed) }

    /// Keeps evaluated copies of the sliding layers' rows at the position the cache has reached. A cached row
    /// is a slice of its layer's buffer, which a later pass writes over, so each is copied.
    func snapshot() {
        guard snapshots.last?.position != cache.offset else { return }
        let rows = cache.sliding.exportedArrays().mapValues { ($0 * 1).asType($0.dtype) }
        eval(Array(rows.values))
        snapshots.append((cache.offset, rows))
        if snapshots.count > Self.keptSnapshots {
            snapshots.removeFirst(snapshots.count - Self.keptSnapshots)
        }
    }

    /// Empties the cache, the tokens, and the snapshots.
    func reset() {
        cache = NFKMLXGemma3Cache(layerCount: layerTypes.count, slidingWindow: slidingWindow)
        tokens = []
        snapshots = []
        sharedPrefixLength = 0
    }

    /// The positions a prefill from `start` to `end` stops at to snapshot: each multiple of
    /// ``snapshotInterval`` it passes, `turnStart`, the position the prompt's generation prompt begins at, and
    /// `end`, where a raw prompt's reply begins.
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

extension NFKMLXGemma4Conversation: NFKMLXSpillableConversation {
    var resumablePositions: [Int]? { snapshots.map(\.position) }
    var optionsSignature: String { "gemma4 sliding \(slidingWindow)" }
    var promptLength: Int { lastPromptLength > 0 ? lastPromptLength : tokens.count }

    /// Without moving anything, the position a prompt continues from: the shared prefix where the live cache
    /// reaches it, else the newest snapshot inside it.
    func resumablePosition(for prompt: [Int]) -> Int { resumption(for: prompt).position }

    /// Writes both caches' rows and every snapshot to a safetensors file.
    func save(to url: URL) throws {
        var arrays = ["token_count": MLXArray(Int32(tokens.count))]
        for (key, value) in cache.full.exportedArrays() { arrays["full." + key] = value }
        for (key, value) in cache.sliding.exportedArrays() { arrays["sliding." + key] = value }
        for (number, snapshot) in snapshots.enumerated() {
            for (key, value) in snapshot.rows { arrays["snapshot.\(number)." + key] = value }
        }
        let metadata = ["dtype": NFKMLXConversationSpill.name(of: cache.full.storedDTypeForExport),
                        "snapshots": snapshots.map { String($0.position) }.joined(separator: ",")]
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
    }

    /// Reads what ``save(to:)`` wrote for `tokens`, or nil where the file holds another count.
    static func load(from url: URL, tokens: [Int], layerTypes: [NFKMLXGemmaAttentionKind],
                     slidingWindow: Int, storingLayers: Int) throws -> NFKMLXGemma4Conversation? {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        guard arrays["token_count"].map({ Int($0.item(Int32.self)) }) == tokens.count else { return nil }
        let restored = NFKMLXGemma4Conversation(layerTypes: layerTypes, slidingWindow: slidingWindow,
                                                storingLayers: storingLayers)
        let dtype = NFKMLXConversationSpill.dtype(named: metadata["dtype"] ?? "float32")
        func part(_ prefix: String) -> [String: MLXArray] {
            Dictionary(uniqueKeysWithValues: arrays.filter { $0.key.hasPrefix(prefix) }
                .map { (String($0.key.dropFirst(prefix.count)), $0.value) })
        }
        restored.cache.full.restore(arrays: part("full."), offset: tokens.count, storedDType: dtype)
        restored.cache.sliding.restore(arrays: part("sliding."), offset: tokens.count, storedDType: dtype)
        let positions = (metadata["snapshots"] ?? "").split(separator: ",").compactMap { Int($0) }
        restored.snapshots = positions.enumerated().map { ($1, part("snapshot.\($0).")) }
        restored.tokens = tokens
        return restored
    }
}
