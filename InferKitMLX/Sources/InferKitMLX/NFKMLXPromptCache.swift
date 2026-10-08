//
//  NFKMLXPromptCache.swift
//  InferKitMLX
//
//  A key-value cache that outlives one generation, so a prompt sharing a prefix with the previous
//  one re-runs only its tail.
//

import Foundation
import InferKit
import MLX

/// A key-value cache kept between generations.
///
/// @discussion A chat turn's prompt is the previous turn's prompt plus the reply plus the new
/// message, and a system prompt is the same on every request. Prefilling that prefix again is the
/// largest cost in a conversation, and it buys nothing: the cache rows it produces are the rows
/// already held. This keeps the cache and the token ids it was built from, and `align(to:)`
/// rolls it back to the last position the new prompt shares, so generation prefills only what is
/// new. The result is exact — every retained row was written by an ordinary forward pass — and the
/// rollback moves cursors rather than copying.
///
/// A window makes the reuse conditional: once positions have been dropped, a prompt that diverges
/// inside the dropped span cannot be rolled back to, and the cache rebuilds from the start instead.
/// ``save(to:)`` and ``load(from:)`` persist a prefilled prompt, which is how a long system prompt
/// costs its prefill once per install rather than once per launch.
public final class NFKMLXPromptCache {

    /// The cache bound, or nil for an unbounded cache. See ``NFKMLXKeyValueCache/window``.
    public let window: Int?
    /// The storage quantization, or nil for full precision. See ``NFKMLXKeyValueCache/quantization``.
    public let quantization: NFKMLXKeyValueCache.Quantization?
    let layerCount: Int

    /// The token ids the cache currently holds rows for, in order.
    public private(set) var tokens: [Int] = []

    /// How many of the last aligned prompt's tokens the cache already held, which is the cached
    /// share of the input a backend reports under `NFKUsageCachedTokens`. Introduced in InferKit
    /// 0.4.0.
    public private(set) var sharedPrefixLength = 0
    private(set) var cache: NFKMLXKeyValueCache

    public init(layerCount: Int, window: Int? = nil,
                quantization: NFKMLXKeyValueCache.Quantization? = nil) {
        self.layerCount = layerCount
        self.window = window
        self.quantization = quantization
        cache = NFKMLXKeyValueCache(layerCount: layerCount, window: window, quantization: quantization)
    }

    /// How many positions the cache holds.
    public var count: Int { tokens.count }

    /// The bytes the cache occupies, its unused capacity included.
    var allocatedBytes: Int { cache.allocatedBytes }

    /// Whether this cache was built for the same geometry and options as `options` asks for, which
    /// is what decides whether a backend keeps it or starts another.
    func matches(layerCount: Int, options: NFKMLXGenerationOptions) -> Bool {
        self.layerCount == layerCount && window == options.contextWindow
            && quantization == options.cacheQuantization
    }

    /// Rolls the cache back to the longest prefix it shares with `prompt` and returns that prefix's
    /// length. The caller prefills `prompt` from there.
    ///
    /// @discussion The shared prefix is capped one short of the prompt, so at least one token runs
    /// through the model and produces the logits generation starts from. When the rollback reaches
    /// past what a window retains, the cache starts over and the whole prompt prefills.
    func align(to prompt: [Int]) -> Int {
        var shared = 0
        let limit = Swift.min(tokens.count, Swift.max(prompt.count - 1, 0))
        while shared < limit && tokens[shared] == prompt[shared] {
            shared += 1
        }
        let discarded = tokens.count - shared
        guard cache.rollback(by: discarded) else {
            reset()
            sharedPrefixLength = 0
            return 0
        }
        tokens.removeLast(discarded)
        sharedPrefixLength = shared
        return shared
    }

    /// Records tokens whose rows a forward pass has just written.
    func record(_ fed: [Int]) { tokens.append(contentsOf: fed) }

    /// Discards the newest `count` positions from the cache and the record alike.
    @discardableResult
    func rollback(by count: Int) -> Bool {
        guard cache.rollback(by: count) else { return false }
        tokens.removeLast(count)
        return true
    }

    /// Empties the cache.
    public func reset() {
        tokens = []
        sharedPrefixLength = 0
        cache = NFKMLXKeyValueCache(layerCount: layerCount, window: window, quantization: quantization)
    }

    // MARK: Persistence

    private static let formatKey = "inferkit.prompt_cache"
    private static let formatVersion = "1"

    /// Writes the cache to a safetensors file: every retained row, the token ids, and the geometry.
    public func save(to url: URL) throws {
        var metadata = [Self.formatKey: Self.formatVersion,
                        "tokens": tokens.map(String.init).joined(separator: ","),
                        "layer_count": String(layerCount),
                        "dtype": Self.name(of: cache.storedDTypeForExport)]
        if let window { metadata["window"] = String(window) }
        if let quantization {
            metadata["quantization"] = "\(quantization.bits):\(quantization.groupSize)"
                + (quantization.groupsKeysAlongTheSequence ? ":sequence" : "")
        }
        var arrays = cache.exportedArrays()
        // A safetensors file needs at least one tensor; an empty cache writes its token count.
        arrays["token_count"] = MLXArray(Int32(tokens.count))
        let scratch = url.deletingLastPathComponent()
            .appendingPathComponent(".\(UUID().uuidString).safetensors")
        try MLX.save(arrays: arrays, metadata: metadata, url: scratch)
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                _ = try FileManager.default.replaceItemAt(url, withItemAt: scratch)
            } else {
                try FileManager.default.moveItem(at: scratch, to: url)
            }
        } catch {
            try? FileManager.default.removeItem(at: scratch)
            throw error
        }
    }

    /// Reads a cache ``save(to:)`` wrote.
    public static func load(from url: URL) throws -> NFKMLXPromptCache {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        guard metadata[formatKey] == formatVersion, let layerText = metadata["layer_count"],
              let layerCount = Int(layerText) else {
            throw NFKMLXError.malformedCheckpoint("\(url.lastPathComponent) is not a prompt cache")
        }
        let window = metadata["window"].flatMap(Int.init)
        let quantization = metadata["quantization"].flatMap { text -> NFKMLXKeyValueCache.Quantization? in
            let parts = text.split(separator: ":")
            guard parts.count >= 2, let bits = Int(parts[0]), let groupSize = Int(parts[1]) else { return nil }
            return .init(bits: bits, groupSize: groupSize,
                         keyAxis: parts.count > 2 && parts[2] == "sequence" ? .sequence : .headDimension)
        }
        let tokens = (metadata["tokens"] ?? "").split(separator: ",").compactMap { Int($0) }
        let restored = NFKMLXPromptCache(layerCount: layerCount, window: window, quantization: quantization)
        restored.tokens = tokens
        restored.cache.restore(arrays: arrays, offset: tokens.count,
                               storedDType: dtype(named: metadata["dtype"] ?? "float32"))
        return restored
    }

    private static func name(of dtype: DType) -> String {
        switch dtype {
        case .float16: return "float16"
        case .bfloat16: return "bfloat16"
        default: return "float32"
        }
    }

    private static func dtype(named name: String) -> DType {
        switch name {
        case "float16": return .float16
        case "bfloat16": return .bfloat16
        default: return .float32
        }
    }
}

/// What a backend keeps for one named conversation between requests.
protocol NFKMLXConversationState: AnyObject {
    /// The bytes it occupies, the unused capacity of its buffers included.
    var allocatedBytes: Int { get }
}

extension NFKMLXPromptCache: NFKMLXConversationState {}

/// The name of the conversation a request continues: its `NFKParameterConversationKey`, unless the
/// request turns prompt reuse off with `NFKMLXGenerationParameterKey.reusesPromptCache`.
enum NFKMLXConversation {
    static func name(of request: NFKInferenceRequest) -> String? {
        guard let conversation = request.parameter(forKey: NFKParameterConversationKey) as? String,
              !conversation.isEmpty else {
            return nil
        }
        if let reuse = request.parameter(forKey: NFKMLXGenerationParameterKey.reusesPromptCache) as? NSNumber,
           !reuse.boolValue {
            return nil
        }
        return conversation
    }
}

/// What a backend keeps for its named conversations, under one byte budget.
///
/// @discussion Each conversation continues from its own state, so turns of several chats arriving
/// interleaved each prefill only what they add. Past the budget the least recently used
/// conversations go first. The conversation a request is running is never evicted for it, so one
/// conversation larger than the budget keeps its state until another request needs the room. The
/// caller serializes access.
final class NFKMLXConversationStates<State: NFKMLXConversationState> {
    private var states: [String: State] = [:]
    /// The conversations, least recently used first.
    private var order: [String] = []

    /// The bytes the states may hold together.
    var byteBudget: Int

    init(byteBudget: Int) {
        self.byteBudget = byteBudget
    }

    var count: Int { states.count }

    var heldBytes: Int { states.values.reduce(0) { $0 + $1.allocatedBytes } }

    /// The conversation's state, marked most recently used. A conversation without one, or whose
    /// state `isUsable` refuses, starts a new one from `make`.
    func state(for conversation: String, isUsable: (State) -> Bool = { _ in true },
               make: () -> State) -> State {
        order.removeAll { $0 == conversation }
        order.append(conversation)
        if let kept = states[conversation], isUsable(kept) {
            return kept
        }
        let made = make()
        states[conversation] = made
        return made
    }

    /// Evicts the least recently used conversations other than `kept` until the states fit the budget.
    func evict(keeping kept: String? = nil) {
        var held = heldBytes
        for conversation in order where held > byteBudget && conversation != kept {
            held -= states[conversation]?.allocatedBytes ?? 0
            remove(conversation)
        }
    }

    func remove(_ conversation: String) {
        states[conversation] = nil
        order.removeAll { $0 == conversation }
    }

    func removeAll() {
        states = [:]
        order = []
    }
}

/// The language backend's conversation prompt caches.
typealias NFKMLXConversationCaches = NFKMLXConversationStates<NFKMLXPromptCache>

extension NFKMLXConversationStates where State == NFKMLXPromptCache {
    /// The conversation's cache, marked most recently used. A conversation without one, or whose
    /// cache was built for another window or quantization, starts a new one.
    func cache(for conversation: String, layerCount: Int, options: NFKMLXGenerationOptions) -> NFKMLXPromptCache {
        state(for: conversation, isUsable: { $0.matches(layerCount: layerCount, options: options) }) {
            NFKMLXPromptCache(layerCount: layerCount, window: options.contextWindow,
                              quantization: options.cacheQuantization)
        }
    }
}

/// A backend's conversation states and the figures its `backendStatus` reports.
///
/// @discussion The states are read and written under the backend's generation lock. The figures are
/// a snapshot ``refresh()`` takes there and ``status`` reads under a lock of its own, so a status
/// request never waits on a run in progress.
final class NFKMLXConversationKeeper<State: NFKMLXConversationState>: @unchecked Sendable {
    let states: NFKMLXConversationStates<State>
    private var figures: [String: Any]
    private let figuresLock = NSLock()

    init(byteBudget: Int) {
        states = NFKMLXConversationStates(byteBudget: byteBudget)
        figures = ["conversation_caches": 0, "conversation_cache_bytes": 0,
                   "conversation_cache_byte_budget": byteBudget]
    }

    /// Records the figures ``status`` reports. Runs under the backend's generation lock.
    func refresh() {
        let taken: [String: Any] = ["conversation_caches": states.count,
                                    "conversation_cache_bytes": states.heldBytes,
                                    "conversation_cache_byte_budget": states.byteBudget]
        figuresLock.lock(); defer { figuresLock.unlock() }
        figures = taken
    }

    /// The figures as of the last run or change.
    var status: [String: Any] {
        figuresLock.lock(); defer { figuresLock.unlock() }
        return figures
    }
}
