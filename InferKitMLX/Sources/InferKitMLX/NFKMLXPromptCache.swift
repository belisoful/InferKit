//
//  NFKMLXPromptCache.swift
//  InferKitMLX
//
//  A key-value cache that outlives one generation, so a prompt sharing a prefix with the previous
//  one re-runs only its tail.
//

import Foundation
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
            return 0
        }
        tokens.removeLast(discarded)
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
            let parts = text.split(separator: ":").compactMap { Int($0) }
            return parts.count == 2 ? .init(bits: parts[0], groupSize: parts[1]) : nil
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
