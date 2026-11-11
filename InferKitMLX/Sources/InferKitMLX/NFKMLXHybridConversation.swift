//
//  NFKMLXHybridConversation.swift
//  InferKitMLX
//
//  What the hybrid decoder keeps for a conversation between requests, so a follow-up turn prefills only
//  what it adds.
//

import CryptoKit
import Foundation
import InferKit
import MLX

/// The recurrence layers' convolution inputs and states at one position of a conversation.
struct NFKMLXHybridSnapshot {
    let position: Int
    let convolutionInputs: [MLXArray?]
    let states: [MLXArray?]

    var bytes: Int {
        zip(convolutionInputs, states).reduce(0) { $0 + ($1.0?.nbytes ?? 0) + ($1.1?.nbytes ?? 0) }
    }
}

/// A hybrid decoder's cache kept for one conversation, with snapshots of its recurrent state.
///
/// @discussion The full-attention layers' keys and values roll back to any position they hold by moving a
/// cursor. A recurrence layer's state holds no earlier position, so the conversation keeps copies of the
/// states at chosen positions: the last 64-position chunk boundary before each turn's generation prompt,
/// each multiple of ``snapshotInterval``, and the end of each reply. A prompt that shares a prefix with the
/// held tokens continues from the live cache where it extends every held token, and otherwise from the
/// newest snapshot inside the shared prefix. A turn's prompt ends inside the assistant's opening, which a
/// later rendering of the same turn tokenizes differently, so the turn-start snapshot is what a follow-up
/// usually resumes from. Snapshots sit on chunk boundaries, so a resumed prefill splits the delta rule's
/// chunks where a prefill from the start splits them.
final class NFKMLXHybridConversation: NFKMLXConversationState {
    /// The spacing of the snapshots a long prefill takes.
    static let snapshotInterval = 4096
    /// The delta rule's chunk, which every snapshot position is a multiple of, except a reply's end.
    static let chunk = 64
    /// The most snapshots a conversation keeps; the oldest go first.
    static let keptSnapshots = 4

    let configuration: NFKMLXHybridConfiguration
    let window: Int?
    let quantization: NFKMLXKeyValueCache.Quantization?
    let adapterStrengths: [Float]?
    private(set) var cache: NFKMLXHybridCache
    /// The token ids the cache holds positions for, in order.
    private(set) var tokens: [Int] = []
    /// The snapshots, oldest position first.
    private(set) var snapshots: [NFKMLXHybridSnapshot] = []
    /// The length of the prompt the last run resumed for, or 0 for a conversation read back from disk.
    private var lastPromptLength = 0
    private let attentionLayers: [Int]

    init(configuration: NFKMLXHybridConfiguration, window: Int?, quantization: NFKMLXKeyValueCache.Quantization?,
         adapterStrengths: [Float]?) {
        self.configuration = configuration
        self.window = window
        self.quantization = quantization
        self.adapterStrengths = adapterStrengths
        cache = NFKMLXHybridCache(configuration: configuration, window: window, quantization: quantization)
        attentionLayers = configuration.layerTypes.indices.filter { configuration.layerTypes[$0] == .fullAttention }
    }

    /// The bytes the live cache and the snapshots occupy.
    var allocatedBytes: Int { cache.allocatedBytes + snapshots.reduce(0) { $0 + $1.bytes } }

    /// Whether the conversation was built for a run's window, cache quantization, and adapter strengths.
    func matches(window: Int?, quantization: NFKMLXKeyValueCache.Quantization?, adapterStrengths: [Float]?) -> Bool {
        self.window == window && self.quantization == quantization && self.adapterStrengths == adapterStrengths
    }

    /// How many leading tokens `prompt` shares with the held ones, at most one short of the prompt, so at
    /// least one token runs and gives the logits generation starts from.
    func sharedLength(with prompt: [Int]) -> Int {
        let limit = Swift.min(tokens.count, Swift.max(prompt.count - 1, 0))
        var shared = 0
        while shared < limit && tokens[shared] == prompt[shared] {
            shared += 1
        }
        return shared
    }

    /// The position ``resume(for:)`` would continue `prompt` from, without moving anything.
    func resumablePosition(for prompt: [Int]) -> Int {
        let shared = sharedLength(with: prompt)
        if shared == tokens.count {
            return shared
        }
        guard !cache.attention.hasDroppedPositions,
              let snapshot = snapshots.last(where: { $0.position <= shared }),
              cache.attention.canRollback(by: tokens.count - snapshot.position, layers: attentionLayers) else {
            return 0
        }
        return snapshot.position
    }

    /// Moves the cache to where `prompt` continues and returns that position; the caller prefills the prompt
    /// from there. Where nothing held can be continued, the conversation starts over and returns 0.
    func resume(for prompt: [Int]) -> Int {
        defer { lastPromptLength = prompt.count }
        let position = resumablePosition(for: prompt)
        guard position > 0 else {
            reset()
            return 0
        }
        guard position < tokens.count else { return position }
        let snapshot = snapshots.last { $0.position == position }!
        _ = cache.attention.rollback(by: tokens.count - position, layers: attentionLayers)
        for (index, slot) in cache.linear.enumerated() {
            slot?.convolutionInputs = snapshot.convolutionInputs[index]
            slot?.state = snapshot.states[index]
        }
        tokens.removeLast(tokens.count - position)
        snapshots.removeAll { $0.position > position }
        return position
    }

    /// Records tokens whose positions a pass has just written.
    func record(_ fed: [Int]) { tokens.append(contentsOf: fed) }

    /// Keeps evaluated copies of the recurrence layers' current convolution inputs and states, at the
    /// position the cache has reached. An MLX slice shares its source's buffer, and a convolution's inputs
    /// are a slice of its whole pass, so each array is copied rather than referenced.
    func snapshot() {
        guard snapshots.last?.position != cache.offset else { return }
        func copied(_ array: MLXArray?) -> MLXArray? { array.map { ($0 * 1).asType($0.dtype) } }
        let convolutionInputs = cache.linear.map { copied($0?.convolutionInputs) }
        let states = cache.linear.map { copied($0?.state) }
        eval(convolutionInputs.compactMap { $0 } + states.compactMap { $0 })
        snapshots.append(NFKMLXHybridSnapshot(position: cache.offset, convolutionInputs: convolutionInputs,
                                              states: states))
        if snapshots.count > Self.keptSnapshots {
            snapshots.removeFirst(snapshots.count - Self.keptSnapshots)
        }
    }

    /// Empties the cache, the tokens, and the snapshots.
    func reset() {
        cache = NFKMLXHybridCache(configuration: configuration, window: window, quantization: quantization)
        tokens = []
        snapshots = []
    }

    /// The positions a prefill of `count` tokens from `start` stops at to snapshot: each multiple of
    /// ``snapshotInterval`` it passes, and the last chunk boundary at or before `turnStart`, the position the
    /// prompt's generation prompt begins at.
    static func snapshotPositions(start: Int, end: Int, turnStart: Int?) -> [Int] {
        var positions = Set(stride(from: (start / snapshotInterval + 1) * snapshotInterval, to: end,
                                   by: snapshotInterval))
        if let turnStart {
            positions.insert(turnStart - turnStart % chunk)
        }
        return positions.filter { $0 > start && $0 < end }.sorted()
    }

    /// The bytes one conversation of `positions` positions takes at most: the keys and values of the
    /// full-attention layers, stored at `elementBytes` each or packed under `quantization`, and the
    /// snapshots it keeps beside the live state.
    static func bytes(positions: Int, configuration c: NFKMLXHybridConfiguration,
                      quantization: NFKMLXKeyValueCache.Quantization?, elementBytes: Int) -> Int {
        let attentionLayers = c.layerTypes.filter { $0 == .fullAttention }.count
        let elements = attentionLayers * c.keyValueHeadCount * c.headDimensions * 2 * positions
        let keyValues = quantization.map { q in elements * q.bits / 8 + elements / q.groupSize * 2 * elementBytes }
            ?? elements * elementBytes
        return keyValues + (keptSnapshots + 1) * stateBytes(configuration: c, elementBytes: elementBytes)
    }

    /// The bytes of one copy of the recurrence layers' states and convolution inputs.
    static func stateBytes(configuration c: NFKMLXHybridConfiguration, elementBytes: Int) -> Int {
        let linearLayers = c.layerTypes.filter { $0 == .linearAttention }.count
        let state = c.linearValueHeadCount * c.linearKeyHeadDimensions * c.linearValueHeadDimensions * 4
        let width = 2 * c.linearKeyHeadCount * c.linearKeyHeadDimensions + c.linearValueHeadCount * c.linearValueHeadDimensions
        return linearLayers * (state + (c.linearConvolutionKernel - 1) * width * elementBytes)
    }

    // MARK: Spilling to disk

    private static let formatKey = "inferkit.hybrid_conversation"
    private static let formatVersion = "1"

    /// The options a conversation was built for, as ``NFKMLXConversationSpill`` compares them before reading
    /// a file.
    var optionsSignature: String {
        Self.optionsSignature(window: window, quantization: quantization, adapterStrengths: adapterStrengths)
    }

    static func optionsSignature(window: Int?, quantization: NFKMLXKeyValueCache.Quantization?,
                                 adapterStrengths: [Float]?) -> String {
        let span = window.map { String($0) } ?? "-"
        let packing = quantization.map { "\($0.bits):\($0.groupSize):\($0.groupsKeysAlongTheSequence)" } ?? "-"
        let strengths = adapterStrengths.map { values in values.map { String($0) }.joined(separator: ",") } ?? "-"
        return "\(span)|\(packing)|\(strengths)"
    }

    /// Writes the live cache and the snapshots to a safetensors file. The token ids are written beside it by
    /// ``NFKMLXConversationSpill``.
    func save(to url: URL) throws {
        var arrays = cache.attention.exportedArrays()
        for (index, slot) in cache.linear.enumerated() {
            if let inputs = slot?.convolutionInputs { arrays["linear.\(index).convolution"] = inputs }
            if let state = slot?.state { arrays["linear.\(index).state"] = state }
        }
        for (number, snapshot) in snapshots.enumerated() {
            for index in snapshot.states.indices {
                if let inputs = snapshot.convolutionInputs[index] { arrays["snapshot.\(number).\(index).convolution"] = inputs }
                if let state = snapshot.states[index] { arrays["snapshot.\(number).\(index).state"] = state }
            }
        }
        arrays["token_count"] = MLXArray(Int32(tokens.count))
        let metadata = [Self.formatKey: Self.formatVersion, "options": optionsSignature,
                        "dtype": NFKMLXConversationSpill.name(of: cache.attention.storedDTypeForExport),
                        "snapshots": snapshots.map { String($0.position) }.joined(separator: ",")]
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
    }

    /// Reads a conversation ``save(to:)`` wrote with `tokens`, or nil where the file holds another format or
    /// was built for other options.
    static func load(from url: URL, tokens: [Int], configuration: NFKMLXHybridConfiguration, window: Int?,
                     quantization: NFKMLXKeyValueCache.Quantization?, adapterStrengths: [Float]?) throws -> NFKMLXHybridConversation? {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        guard metadata[formatKey] == formatVersion,
              metadata["options"] == optionsSignature(window: window, quantization: quantization, adapterStrengths: adapterStrengths),
              arrays["token_count"].map({ Int($0.item(Int32.self)) }) == tokens.count else {
            return nil
        }
        let restored = NFKMLXHybridConversation(configuration: configuration, window: window, quantization: quantization,
                                                adapterStrengths: adapterStrengths)
        restored.cache.attention.restore(arrays: arrays, offset: tokens.count,
                                         storedDType: NFKMLXConversationSpill.dtype(named: metadata["dtype"] ?? "float32"))
        for (index, slot) in restored.cache.linear.enumerated() {
            slot?.convolutionInputs = arrays["linear.\(index).convolution"]
            slot?.state = arrays["linear.\(index).state"]
        }
        let positions = (metadata["snapshots"] ?? "").split(separator: ",").compactMap { Int($0) }
        restored.snapshots = positions.enumerated().map { number, position in
            NFKMLXHybridSnapshot(position: position,
                                 convolutionInputs: configuration.layerTypes.indices.map { arrays["snapshot.\(number).\($0).convolution"] },
                                 states: configuration.layerTypes.indices.map { arrays["snapshot.\(number).\($0).state"] })
        }
        restored.tokens = tokens
        return restored
    }
}

/// A conversation's state a backend can write to disk and read back: its token ids, the positions a prompt
/// can resume it at, and the options it was built for.
protocol NFKMLXSpillableConversation: NFKMLXConversationState {
    var tokens: [Int] { get }
    /// The positions short of the held tokens a prompt can resume at, or nil where every position can.
    var resumablePositions: [Int]? { get }
    var optionsSignature: String { get }
    /// The length of the last run's prompt, which a keyless prompt is weighed against; the reply after it is
    /// not, since a follow-up often renders a reply in other tokens than the model wrote.
    var promptLength: Int { get }
    func save(to url: URL) throws
    /// The position a prompt continues the conversation from, without moving anything.
    func resumablePosition(for prompt: [Int]) -> Int
}

extension NFKMLXConversationKeeper where State: NFKMLXSpillableConversation {
    /// The conversation a run continues and the name it is kept under, or a conversation kept under no name
    /// where `reuses` is false. A named run continues its own conversation, held or read back from the spill
    /// directory by `read`. A run that names none continues the unnamed conversation, held or on disk, its
    /// prompt resumes furthest into, where the resumed prefix is at least as long as what the rollback
    /// discards of that conversation's last prompt, so one client's prompt never discards most of another's
    /// conversation; otherwise it starts one. A held conversation `usable` refuses starts over. Runs under the
    /// backend's generation lock.
    func conversation(named name: String?, reuses: Bool, prompt: [Int], signature: String,
                      usable: (State) -> Bool = { _ in true }, make: () -> State,
                      read: (URL, NFKMLXConversationSpill.Record) throws -> State?) -> (name: String?, state: State) {
        guard reuses else { return (nil, make()) }
        let chosen: String
        if let name {
            chosen = name
        } else {
            let prefix = NFKMLXConversation.unnamedPrefix
            let inMemory = states.held
                .filter { $0.name.hasPrefix(prefix) && usable($0.state) }
                .map { (name: $0.name, kept: $0.state.resumablePosition(for: prompt), held: $0.state.promptLength) }
                .filter { $0.kept > 0 && $0.kept >= $0.held - $0.kept }
                .max { $0.kept < $1.kept }
            let onDisk = spill?.resumable(for: prompt, prefix: prefix, options: signature)
            if let onDisk, onDisk.kept > inMemory?.kept ?? 0 {
                chosen = onDisk.name
            } else {
                chosen = inMemory?.name ?? prefix + UUID().uuidString
            }
        }
        let held = states.held.contains { $0.name == chosen }
        let spilled = held ? nil : spill?.take(chosen) { url, record in
            record.options == signature ? try read(url, record) : nil
        }
        return (chosen, states.state(for: chosen, isUsable: usable) { spilled ?? make() })
    }

    /// Ends a run of the conversation `name`: drops it where the run failed, sets the budget to the one a
    /// caller fixed or else `afforded`, and lets go of the least recently used conversations past it, writing
    /// each to the spill directory where `spills`. Runs under the backend's generation lock.
    func finish(_ name: String?, failed: Bool, afforded: Int, spills: Bool) {
        if let name {
            if failed {
                states.remove(name)
            }
            states.byteBudget = fixedByteBudget ?? afforded
            states.evict(keeping: name) { name, state in
                if spills { try? spill?.write(state, name: name) }
            }
        }
        refresh()
    }

    /// Defaults the spill directory, once, to the one ``NFKMLXConversationSpill/defaultDirectory(release:identifier:weights:)``
    /// names for `release`; a backend without a release writes none until a caller sets one. Runs under the
    /// backend's generation lock.
    func resolveSpill(release: URL?, identifier: String, info: () -> [String: Any]) {
        guard !spillResolved else { return }
        spillResolved = true
        spill = release.map {
            NFKMLXConversationSpill(directory: NFKMLXConversationSpill.defaultDirectory(
                release: $0, identifier: identifier, weights: NFKMLXConversationSpill.weightsDescription(info())),
                                    byteBudget: NFKMLXConversationSpill.defaultByteBudget)
        }
    }

    /// Fixes the budget at `bytes` and lets go of the least recently used conversations past it, writing each
    /// to the spill directory. Runs under the backend's generation lock.
    func fixBudget(_ bytes: Int) {
        fixedByteBudget = bytes
        states.byteBudget = bytes
        states.evict { name, state in try? spill?.write(state, name: name) }
        refresh()
    }

    /// Sets the spill directory, or none for nil, keeping the disk budget. Runs under the backend's generation lock.
    func setSpillDirectory(_ directory: URL?) {
        spill = directory.map { NFKMLXConversationSpill(directory: $0, byteBudget: spill?.byteBudget
                                                        ?? NFKMLXConversationSpill.defaultByteBudget) }
        spillResolved = true
    }

    /// Writes every conversation to the spill directory where `spills`, drops them all, and returns the bytes
    /// they held. Runs under the backend's generation lock.
    func releaseAll(spills: Bool) -> Int {
        let held = states.heldBytes
        if spills {
            for (name, state) in states.held {
                try? spill?.write(state, name: name)
            }
        }
        states.removeAll()
        refresh()
        return held
    }
}

/// The bytes a backend's conversations may hold where no budget is set: the planned working set less the
/// weights the model holds, the reserve beside them, and `runBytes`, what one run's cache takes.
enum NFKMLXConversationBudget {
    static func afforded(info: [String: Any], runBytes: Int) -> Int {
        let budget = NFKMLXResidencyBudget.current()
        let weights = info[NFKModelInfoWeightBytes] as? Int ?? 0
        return Swift.max(0, budget - weights - NFKMLXResidencyBudget.reserve(forBudget: budget) - runBytes)
    }

    /// One run's keys and values from a model's description: `NFKModelInfoKeyValueBytesPerToken` over
    /// `positions`, or over the context length the description states where `positions` is nil.
    static func runBytes(info: [String: Any], positions: Int?) -> Int {
        (info[NFKModelInfoKeyValueBytesPerToken] as? Int ?? 0) * (positions ?? info[NFKModelInfoContextLength] as? Int ?? 0)
    }
}

extension NFKMLXHybridConversation: NFKMLXSpillableConversation {
    var resumablePositions: [Int]? { snapshots.map(\.position) }
    var promptLength: Int { lastPromptLength > 0 ? lastPromptLength : tokens.count }
}

/// Conversations written to disk where the memory budget lets them go, so a later request, after an
/// eviction, an unload, or a restart, continues one without its prefill.
///
/// @discussion Each conversation is two files named by a digest of its name: the arrays the state writes
/// (``NFKMLXSpillableConversation/save(to:)``) and a JSON record of its name, token ids, resumable positions,
/// and options. Reading a conversation removes its files, since the memory then holds it. Past the byte
/// budget the least recently written files go first. The caller serializes access.
final class NFKMLXConversationSpill {
    let directory: URL
    var byteBudget: Int

    init(directory: URL, byteBudget: Int) {
        self.directory = directory
        self.byteBudget = byteBudget
    }

    /// The bytes the spilled conversations may take on disk where no budget is set.
    static let defaultByteBudget = 8 << 30

    /// A directory under the user's caches named by a digest of the release's path, its configuration's
    /// modification date, the model's identifier, and `weights`, a description of the weights' precision and
    /// quantization, so another release, the same one changed, or other weights find none of these
    /// conversations.
    static func defaultDirectory(release: URL, identifier: String, weights: String) -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let configured = (try? release.appendingPathComponent("config.json")
            .resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)?.timeIntervalSince1970 ?? 0
        let identity = "\(release.standardizedFileURL.path)|\(configured)|\(identifier)|\(weights)"
        return caches.appendingPathComponent("InferKit/Conversations/\(digest(of: identity))", isDirectory: true)
    }

    /// The precision and quantization ``defaultDirectory(release:identifier:weights:)`` reads from a model's
    /// `NFKModelInfo*` description.
    static func weightsDescription(_ info: [String: Any]) -> String {
        "\(info[NFKModelInfoPrecision] ?? "")|\(info[NFKModelInfoQuantizationBits] ?? "")|"
            + "\(info[NFKModelInfoQuantizationGroupSize] ?? "")"
    }

    private static func digest(of text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).prefix(16).map { String(format: "%02x", $0) }.joined()
    }

    /// A conversation's record: what a prefix search reads without loading the arrays.
    struct Record: Codable {
        let name: String
        let tokens: [Int]
        let snapshots: [Int]?
        let options: String
        let promptLength: Int

        /// The position a conversation holding this record's tokens continues `prompt` from.
        func resumablePosition(for prompt: [Int]) -> Int {
            let limit = Swift.min(tokens.count, Swift.max(prompt.count - 1, 0))
            let shared = zip(tokens[..<limit], prompt).prefix { $0.0 == $0.1 }.count
            guard shared < tokens.count, let snapshots else { return shared }
            return snapshots.last { $0 <= shared } ?? 0
        }
    }

    private func arraysURL(_ stem: String) -> URL { directory.appendingPathComponent(stem + ".safetensors") }
    private func recordURL(_ stem: String) -> URL { directory.appendingPathComponent(stem + ".json") }

    /// Writes `conversation` under `name`, replacing what the name held, then trims the directory to the
    /// budget. The record is written last, so a write cut short leaves no record to read.
    func write<State: NFKMLXSpillableConversation>(_ conversation: State, name: String) throws {
        guard byteBudget > 0, !conversation.tokens.isEmpty else { return }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let stem = Self.digest(of: name)
        remove(stem)
        try conversation.save(to: arraysURL(stem))
        let record = Record(name: name, tokens: conversation.tokens, snapshots: conversation.resumablePositions,
                            options: conversation.optionsSignature, promptLength: conversation.promptLength)
        try JSONEncoder().encode(record).write(to: recordURL(stem), options: .atomic)
        trim()
    }

    /// Every record on disk.
    func records() -> [Record] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.filter { $0.hasSuffix(".json") }.compactMap { file in
            (try? Data(contentsOf: directory.appendingPathComponent(file)))
                .flatMap { try? JSONDecoder().decode(Record.self, from: $0) }
        }
    }

    /// The unnamed conversation on disk, its name starting with `prefix`, that `prompt` resumes furthest into,
    /// among those built for `options` whose resumed prefix is at least what a rollback discards of its last
    /// prompt.
    func resumable(for prompt: [Int], prefix: String, options: String) -> (name: String, kept: Int)? {
        records()
            .filter { $0.name.hasPrefix(prefix) && $0.options == options }
            .map { (name: $0.name, kept: $0.resumablePosition(for: prompt), held: $0.promptLength) }
            .filter { $0.kept > 0 && $0.kept >= $0.held - $0.kept }
            .max { $0.kept < $1.kept }
            .map { ($0.name, $0.kept) }
    }

    /// The conversation written under `name`, read by `read` from the arrays file and the record, its files
    /// removed; nil where none was written or `read` refuses it.
    func take<State>(_ name: String, read: (URL, Record) throws -> State?) -> State? {
        let stem = Self.digest(of: name)
        defer { remove(stem) }
        guard let data = try? Data(contentsOf: recordURL(stem)),
              let record = try? JSONDecoder().decode(Record.self, from: data), record.name == name else {
            return nil
        }
        return try? read(arraysURL(stem), record)
    }

    /// Removes the files written under `name`.
    func remove(named name: String) { remove(Self.digest(of: name)) }

    private func remove(_ stem: String) {
        try? FileManager.default.removeItem(at: recordURL(stem))
        try? FileManager.default.removeItem(at: arraysURL(stem))
    }

    /// Removes the least recently written conversations until the directory fits the budget.
    private func trim() {
        let keys: [URLResourceKey] = [.contentModificationDateKey, .fileAllocatedSizeKey]
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: keys)) ?? []
        let written = files.filter { $0.pathExtension == "safetensors" }.map { url -> (stem: String, date: Date, bytes: Int) in
            let values = try? url.resourceValues(forKeys: Set(keys))
            return (url.deletingPathExtension().lastPathComponent, values?.contentModificationDate ?? .distantPast,
                    values?.fileAllocatedSize ?? 0)
        }.sorted { $0.date < $1.date }
        var held = written.reduce(0) { $0 + $1.bytes }
        for file in written where held > byteBudget {
            remove(file.stem)
            held -= file.bytes
        }
    }

    static func name(of dtype: DType) -> String {
        switch dtype {
        case .float16: return "float16"
        case .bfloat16: return "bfloat16"
        default: return "float32"
        }
    }

    static func dtype(named name: String) -> DType {
        switch name {
        case "float16": return .float16
        case "bfloat16": return .bfloat16
        default: return .float32
        }
    }
}
