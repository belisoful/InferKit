//
//  NFKMLXDeepSeekEngram.swift
//  InferKitMLX
//
//  The n-gram memory DeepSeek V4.1 carries on a few of its layers.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// NumPy's `default_rng(seed).integers(low:high:size:)` for 64-bit bounds, reproduced.
///
/// The n-gram hash multipliers are drawn from this stream and from no other. A port that invents its
/// own stream hashes every n-gram to a different bucket, which reads the wrong row of a table with
/// 384 million of them, so the draw is reproduced rather than approximated. Three pieces have to
/// agree: `SeedSequence`'s entropy mixing, PCG64's seeding and its XSL-RR output, and the bounded
/// draw — `Generator.integers` uses Lemire's rejection scheme, where the legacy `RandomState` uses a
/// masked one and gives entirely different numbers.
struct NFKDeepSeekNumPyGenerator {

    /// 128 bits as a pair, because `UInt128` lands after this package's deployment floor.
    private struct Wide {
        var high: UInt64
        var low: UInt64

        static func * (lhs: Wide, rhs: Wide) -> Wide {
            let product = lhs.low.multipliedFullWidth(by: rhs.low)
            return Wide(high: product.high &+ lhs.low &* rhs.high &+ lhs.high &* rhs.low,
                        low: product.low)
        }

        static func + (lhs: Wide, rhs: Wide) -> Wide {
            let (low, carried) = lhs.low.addingReportingOverflow(rhs.low)
            return Wide(high: lhs.high &+ rhs.high &+ (carried ? 1 : 0), low: low)
        }

        var doubled: Wide { Wide(high: (high << 1) | (low >> 63), low: low << 1) }
    }

    private static let multiplier = Wide(high: 0x2360_ed05_1fc6_5da4, low: 0x4385_df64_9fcc_f645)

    private var state: Wide
    private let increment: Wide

    init(seed: Int) {
        let words = Self.seedWords(UInt64(seed))
        increment = Wide(high: words[2], low: words[3]).doubled + Wide(high: 0, low: 1)
        state = Wide(high: 0, low: 0)
        step()
        state = state + Wide(high: words[0], low: words[1])
        step()
    }

    private mutating func step() {
        state = state * Self.multiplier + increment
    }

    /// One draw of PCG64's XSL-RR output function.
    mutating func next() -> UInt64 {
        step()
        let value = state.high ^ state.low
        let rotation = (state.high >> 58) & 63
        return (value >> rotation) | (value << ((64 &- rotation) & 63))
    }

    /// A uniform draw in `0 ... inclusiveMaximum`, by Lemire's multiply-and-reject.
    mutating func bounded(by inclusiveMaximum: UInt64) -> UInt64 {
        let span = inclusiveMaximum &+ 1
        var product = next().multipliedFullWidth(by: span)
        if product.low < span {
            let threshold = (UInt64.max &- inclusiveMaximum) % span
            while product.low < threshold {
                product = next().multipliedFullWidth(by: span)
            }
        }
        return product.high
    }

    // MARK: SeedSequence

    private static let initialA: UInt32 = 0x43b0_d7e5
    private static let multiplyA: UInt32 = 0x931e_8875
    private static let initialB: UInt32 = 0x8b51_f9dd
    private static let multiplyB: UInt32 = 0x58f3_8ded
    private static let mixLeft: UInt32 = 0xca01_f9dd
    private static let mixRight: UInt32 = 0x4973_f715

    /// `SeedSequence(seed).generate_state(4, uint64)`: eight mixed 32-bit words read as four 64-bit
    /// ones, little-endian, which is how NumPy views the buffer.
    private static func seedWords(_ seed: UInt64) -> [UInt64] {
        var entropy = [UInt32]()
        var remaining = seed
        repeat {
            entropy.append(UInt32(truncatingIfNeeded: remaining))
            remaining >>= 32
        } while remaining != 0

        var constant = initialA
        func hashMix(_ value: UInt32) -> UInt32 {
            var mixed = value ^ constant
            constant = constant &* multiplyA
            mixed = mixed &* constant
            return mixed ^ (mixed >> 16)
        }
        func mix(_ x: UInt32, _ y: UInt32) -> UInt32 {
            let result = mixLeft &* x &- mixRight &* y
            return result ^ (result >> 16)
        }

        var pool = (0 ..< 4).map { hashMix($0 < entropy.count ? entropy[$0] : 0) }
        for source in 0 ..< 4 {
            for destination in 0 ..< 4 where source != destination {
                pool[destination] = mix(pool[destination], hashMix(pool[source]))
            }
        }
        // Entropy beyond the pool, folded in one word at a time. A seed that fits in a single
        // 32-bit word has none, and `4 ..< entropy.count` would be a reversed range rather than an
        // empty one, so the sequence is dropped into rather than counted over.
        for source in entropy.indices.dropFirst(4) {
            for destination in 0 ..< 4 {
                pool[destination] = mix(pool[destination], hashMix(entropy[source]))
            }
        }

        constant = initialB
        var words = [UInt32]()
        for index in 0 ..< 8 {
            var value = pool[index % pool.count] ^ constant
            constant = constant &* multiplyB
            value = value &* constant
            words.append(value ^ (value >> 16))
        }
        return (0 ..< 4).map { UInt64(words[2 * $0 + 1]) << 32 | UInt64(words[2 * $0]) }
    }
}

/// Where each `(layer, n-gram size, head)` hashes into its layer's table, and what it multiplies by.
///
/// Every `(n-gram size, head)` pair owns a prime-sized bucket range, drawn in order above
/// `engramVocabularySize - 1` and never reused, so the ranges are disjoint and the layer's table is
/// exactly as many rows as its primes sum to. That sum is not a free parameter: it reproduces the
/// released 384,006,168 and 384,016,682 row counts, which is what checks the derivation without a
/// checkpoint.
struct NFKDeepSeekEngramLayout {
    /// `[layer][n-gram size][head]`, the bucket count each hashed column is reduced modulo.
    let primes: [[[Int]]]
    /// `[layer][column]`, where that column's bucket range starts inside the layer's table.
    let offsets: [[Int]]
    /// `[layer][lookback]`, the odd multiplier each looked-back token id is scaled by.
    let multipliers: [[Int]]
    let maximumNgramSize: Int
    let headCount: Int

    /// The row count each layer's table needs, which a release states and this reproduces.
    var rowCounts: [Int] { primes.map { $0.flatMap { $0 }.reduce(0, +) } }

    init(_ c: NFKMLXDeepSeekConfiguration) {
        maximumNgramSize = c.engramMaxNgramSize
        headCount = c.engramHeadCount

        // The search restarts at the same floor for every pair and skips what it has already handed
        // out, so the primes come out as the consecutive primes above that floor, in order.
        var candidate = c.engramVocabularySize - 1
        var drawn = [[[Int]]]()
        for _ in c.engramLayerIDs {
            var perSize = [[Int]]()
            for _ in 0 ..< max(c.engramMaxNgramSize - 1, 0) {
                var sizes = [Int]()
                for _ in 0 ..< c.engramHeadCount {
                    candidate = Self.nextPrime(above: candidate)
                    sizes.append(candidate)
                }
                perSize.append(sizes)
            }
            drawn.append(perSize)
        }
        primes = drawn
        offsets = drawn.map { layer in
            var running = 0
            return layer.flatMap { $0 }.map { size in
                defer { running += size }
                return running
            }
        }

        // `token_id * multiplier` must not overflow a signed 64-bit integer, which is what bounds
        // the draw; the doubling and the +1 keep every multiplier odd.
        let bound = max(1, (Int(Int64.max) / max(c.engramCompressedVocabularySize, 1)) / 2)
        multipliers = c.engramLayerIDs.map { layer in
            var generator = NFKDeepSeekNumPyGenerator(seed: 10_007 * layer)
            return (0 ..< c.engramMaxNgramSize).map { _ in
                Int(generator.bounded(by: UInt64(bound - 1))) * 2 + 1
            }
        }
    }

    private static func nextPrime(above value: Int) -> Int {
        var candidate = value + 1
        while !isPrime(candidate) { candidate += 1 }
        return candidate
    }

    private static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor = 3
        while divisor * divisor <= value {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }
}

/// Turns token ids into the table rows the n-gram memory looks up.
///
/// A position hashes the 2-gram through the `engramMaxNgramSize`-gram ending at it, each split over
/// `engramHeadCount` independently bucketed columns. The look-back stops at the start of the
/// sequence and at any position that takes no part in an n-gram, so a hash never spans one.
final class NFKDeepSeekNgramHash {
    let layout: NFKDeepSeekEngramLayout
    /// Token ids collapse into a smaller space first, so that " The", "the" and "THE" hash alike.
    /// Nil leaves ids as they are, which is what a configuration whose ids already normalize apart
    /// wants; a release states the collapsed size its multipliers were derived from.
    let compressedTokens: MLXArray?
    let padToken: Int

    private let multipliers: MLXArray
    private let primes: MLXArray
    private let offsets: MLXArray

    init(_ c: NFKMLXDeepSeekConfiguration, compressedTokens map: [Int]? = nil) {
        layout = NFKDeepSeekEngramLayout(c)
        padToken = map.map { $0[c.engramPadToken] } ?? c.engramPadToken
        compressedTokens = map.map { MLXArray($0.map(Int64.init)) }
        multipliers = MLXArray(layout.multipliers.flatMap { $0 }.map(Int64.init))
            .reshaped([layout.multipliers.count, c.engramMaxNgramSize])
        primes = MLXArray(layout.primes.flatMap { $0.flatMap { $0 } }.map(Int64.init))
            .reshaped([layout.primes.count, c.engramMaxNgramSize - 1, c.engramHeadCount])
        offsets = MLXArray(layout.offsets.flatMap { $0 }.map(Int64.init))
            .reshaped([layout.offsets.count, c.engramHashColumns])
    }

    /// - Parameter tokens: `[batch, length]` raw token ids.
    /// - Parameter alive: `[batch, length]`, false where a position takes no part in an n-gram.
    /// - Returns: `[batch, length, engram layers, columns]` rows into each layer's table.
    func callAsFunction(_ tokens: MLXArray, alive: MLXArray? = nil,
                        cache: NFKMLXDeepSeekCache? = nil) -> MLXArray {
        let (batch, length) = (tokens.shape[0], tokens.shape[1])
        var ids = tokens.asType(.int64)
        if let compressedTokens { ids = compressedTokens[ids] }
        // A dead position is marked rather than dropped, so the look-back can stop at it.
        let dead = MLXArray(Int64(-1))
        if let alive { ids = MLX.where(alive, ids, dead) }

        // A one-token step can look back only into what came before it. Without the history every
        // such step hashes `(pad, pad, token)` and addresses rows no n-gram in the prompt ever did.
        let offset = cache?.offset ?? 0
        let history = cache?.engramPrefix()
        cache?.rememberEngram(ids)
        let carried = history?.dim(1) ?? 0
        let reach = history.map { concatenated([$0, ids], axis: 1) } ?? ids
        let span = carried + length
        // Absolute, so a look-back is blocked by the START of the sequence and not by the start of
        // what this call happens to hold.
        let positions = broadcast(
            MLXArray((0 ..< span).map { Int32(offset - carried + $0) }).reshaped([1, span]),
            to: [batch, span])
        ids = reach
        var blocked = positions .< 0
        var lookback = [MLXArray]()
        // Indexed within what this call holds, while the blocking is judged on absolute position.
        let local = MLXArray((0 ..< span).map(Int32.init)).reshaped([1, span])
        for shift in 0 ..< layout.maximumNgramSize {
            let source = takeAlong(ids, broadcast(maximum(local - shift, 0), to: [batch, span]),
                                   axis: 1)
            blocked = blocked .|| (positions .< shift) .|| (source .== dead)
            lookback.append(MLX.where(blocked, MLXArray(Int64(padToken)), source))
        }
        // [batch, length, engram layers, lookback]: one multiplied id per layer per look-back step.
        let products = stacked(lookback, axis: -1).expandedDimensions(axis: 2) * multipliers

        // The running value after step i is the hash of the (i+1)-gram, and each lands in its own
        // prime-sized range. Every head of one n-gram size shares that value and differs only in
        // the modulus it is reduced by.
        var rolling = products[.ellipsis, 0]
        var columns = [MLXArray]()
        for step in stride(from: 1, to: layout.maximumNgramSize, by: 1) {
            rolling = rolling ^ products[.ellipsis, step]
            columns.append(rolling.expandedDimensions(axis: -1) % primes[0..., step - 1])
        }
        // Only the rows this call was asked about; the carried prefix was context for their
        // look-back, not positions of their own.
        let hashed = concatenated(columns, axis: -1) + offsets
        return carried > 0 ? hashed[0..., carried...] : hashed
    }
}

/// Writes an n-gram lookup into the residual stream, gated by how well it matches that stream.
///
/// The hashed columns fetch one table row each; `wkv` turns them into one key per hyper-connection
/// copy plus a value the copies share. The gate is a normalized dot product of stream against key,
/// so a lookup that does not match the position contributes nothing.
final class NFKDeepSeekEngram: Module {
    /// Absent where the table is held in the form the release stores it. A released table is 384
    /// million rows of 256 channels, 366 GiB decoded, so a decoder that pages it must not build the
    /// float embedding at all.
    @ModuleInfo(key: "embed") var table: Embedding?
    @ModuleInfo(key: "wkv") var keyValue: Linear
    @ParameterInfo(key: "q_weight") var queryWeight: MLXArray
    @ParameterInfo(key: "k_weight") var keyWeight: MLXArray

    let copies: Int
    let hiddenSize: Int
    let columns: Int
    let headDimensions: Int
    let epsilon: Float
    /// The floor the gate's magnitude is held above before the square root, which the reference
    /// takes from the training kernel rather than from the norm epsilon.
    private let clamp: Float = 1e-6
    /// Set where the table is held stored, in which case `table` is absent. The loader fills this
    /// after the module is built, because what it holds is the release's own bytes.
    var storedTable: NFKDeepSeekStoredTable?
    /// Whether looked-up rows are cast to bf16 as the release's lookup casts them.
    let quantizesActivations: Bool
    let fp8BlockSize: Int
    let computeType: DType

    /// - Parameter pagingTable: builds no float embedding, leaving the table to `storedTable`.
    init(_ c: NFKMLXDeepSeekConfiguration, rows: Int, pagingTable: Bool = false) {
        quantizesActivations = c.quantizesActivations
        fp8BlockSize = c.fp8BlockSize
        computeType = c.computeType
        copies = c.hyperConnectionCopies
        hiddenSize = c.hiddenSize
        columns = c.engramHashColumns
        headDimensions = c.engramHeadDimensions
        epsilon = c.rmsEpsilon
        _table.wrappedValue = pagingTable
            ? nil : Embedding(embeddingCount: rows, dimensions: c.engramHeadDimensions)
        _keyValue.wrappedValue = Linear(c.engramHashColumns * c.engramHeadDimensions,
                                        c.hiddenSize * (c.hyperConnectionCopies + 1), bias: false)
        _queryWeight.wrappedValue = MLXArray.ones([c.hyperConnectionCopies, c.hiddenSize])
        _keyWeight.wrappedValue = MLXArray.ones([c.hyperConnectionCopies, c.hiddenSize])
        super.init()
    }

    /// - Parameter x: `[batch, length, copies, hidden]`, the hyper-connected residual stream.
    /// - Parameter hashes: `[batch, length, columns]`, this layer's rows.
    /// - Parameter alive: `[batch, length]`, false where the gate is shut and the position passes
    ///   through untouched.
    func callAsFunction(_ x: MLXArray, hashes: MLXArray, alive: MLXArray? = nil) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        // A stored table gathers the rows it was asked for and decodes those; a resident one is the
        // same lookup against an embedding whose scales were folded in at load. The two read the
        // same bytes against the same scales, so they produce the same floats.
        var looked = storedTable.map { $0(hashes) } ?? table!(hashes)
        // The release's lookup casts the dequantized rows to bf16 on the way out. A float32 decoder
        // that reproduces the release brings them back to float32; a bf16 one keeps them.
        if quantizesActivations || computeType == .bfloat16 {
            looked = looked.asType(.bfloat16).asType(computeType)
        }
        let rows = looked.reshaped([batch, length, columns * headDimensions])
        // `wkv` is built fp8 in the release, so its input is rounded as `linear()` rounds it.
        let mixed = keyValue(quantizesActivations
            ? NFKMLXDeepSeekQuantization.roundTripFP8(rows, blockSize: fp8BlockSize) : rows)
        let key = mixed[.ellipsis, 0 ..< (copies * hiddenSize)]
            .reshaped([batch, length, copies, hiddenSize])
        let value = mixed[.ellipsis, (copies * hiddenSize)...]

        // The gate runs in float32 throughout, and its result is handed back in the stream's dtype.
        let stream = x.asType(.float32)
        let key32 = key.asType(.float32)
        // The two learned weights are only ever used as a product, which is why neither is a norm.
        let weight = queryWeight.asType(.float32) * keyWeight.asType(.float32)
        // Normalized per (position, copy) over the hidden width, NOT jointly across the copies.
        let scale = rsqrt((stream * stream).mean(axis: -1) + epsilon)
            * rsqrt((key32 * key32).mean(axis: -1) + epsilon)
        let dot = (stream * weight * key32).sum(axis: -1) * scale / sqrt(Float(hiddenSize))

        // A signed square root before the sigmoid, matching the training kernel. `sign` is written
        // out because it answers zero at zero, where the reference's `copysign` answers positive.
        let magnitude = sqrt(maximum(abs(dot), clamp))
        var gate = sigmoid(MLX.where(dot .< 0, -magnitude, magnitude))
        if let alive { gate = MLX.where(alive.expandedDimensions(axis: -1), gate, MLXArray(0)) }
        return (stream + gate.expandedDimensions(axis: -1)
                * value.asType(.float32).expandedDimensions(axis: -2)).asType(x.dtype)
    }
}

extension NFKMLXDeepSeek {

    /// The collapsed id space DeepSeek V4.1's n-gram memory hashes over, from a release's tokenizer.
    ///
    /// @discussion Token ids that normalize alike share a hash bucket, so `" The"`, `"the"` and
    /// `"THE"` address the same rows of the table. The SIZE of this space is what every hash
    /// multiplier is derived from, which is why a release states it and why the derivation is held
    /// to that statement: a collapse that drifts becomes a load-time error instead of a forward pass
    /// reading the wrong rows of a table with 384 million of them. Introduced in 0.3.0.
    public static func compressedTokens(fromTokenizerJSON url: URL,
                                        in c: NFKMLXDeepSeekConfiguration) throws -> [Int] {
        // Parsed once and freed before the collapse runs. A release's `tokenizer.json` is several
        // megabytes of vocabulary and merges, and `JSONSerialization` turns it into an object graph
        // far larger than the file, so reading it once per accessor held three of those at a time.
        // Measured, this changes no timing here; it is the smaller peak that justifies it.
        var built: NFKTokenizer?
        var spellings = [Int: String]()
        autoreleasepool {
            guard let json = tokenizerJSON(at: url) else { return }
            built = tokenizer(fromTokenizerJSON: url, json: json)
            spellings = tokenSpellings(json)
        }
        guard let tokenizer = built else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(url.lastPathComponent) does not read as the byte-level BPE this release ships")
        }
        let space = collapsedTokenSpace(for: tokenizer, spellings: spellings,
                                        vocabularySize: c.vocabularySize)
        guard c.engramCompressedVocabularySize == 0
                || space.size == c.engramCompressedVocabularySize else {
            throw NFKMLXError.unsupportedConfiguration(
                "this tokenizer collapses \(c.vocabularySize) ids to \(space.size) where the "
                + "release states \(c.engramCompressedVocabularySize); every n-gram hash multiplier "
                + "is derived from that number, so the two have to agree")
        }
        return space.map
    }

    /// The same derivation for a caller with a vocabulary size and the size to check it against
    /// rather than a configuration. Passing zero for `collapsingTo` skips the check.
    @objc public static func compressedTokens(fromTokenizerJSON url: URL,
                                              vocabularySize: Int,
                                              collapsingTo expected: Int) throws -> [NSNumber] {
        var geometry = NFKMLXDeepSeekConfiguration.v41Flash
        geometry.vocabularySize = vocabularySize
        geometry.engramCompressedVocabularySize = expected
        return try compressedTokens(fromTokenizerJSON: url, in: geometry).map(NSNumber.init(value:))
    }

    /// The release's byte-level BPE tokenizer, read from its `tokenizer.json`.
    ///
    /// @discussion The release ships only `tokenizer.json`, so its vocabulary and merges are
    /// extracted into the `vocab.json` and `merges.txt` the core reader takes, and its added tokens
    /// are registered so that an id above the vocabulary decodes to its own literal rather than to
    /// nothing. Introduced in 0.3.0.
    public static func tokenizer(fromTokenizerJSON url: URL) -> NFKTokenizer? {
        guard let json = tokenizerJSON(at: url) else { return nil }
        return tokenizer(fromTokenizerJSON: url, json: json)
    }

    private static func tokenizer(fromTokenizerJSON url: URL,
                                  json: [String: Any]) -> NFKTokenizer? {
        guard let files = NFKMLXLanguage.byteLevelFiles(fromTokenizerJSON: url) else { return nil }
        var added = [String: Int]()
        for entry in (json["added_tokens"] as? [[String: Any]]) ?? [] {
            if let content = entry["content"] as? String, let id = entry["id"] as? Int {
                added[content] = id
            }
        }
        let manifest: [String: Any] = ["tokenizer": ["type": "bpe-bytelevel",
                                                     "pretokenizer": "gpt2",
                                                     "specialTokens": added]]
        return try? NFKTokenizer(forManifest: manifest, directory: files)
    }

    /// Each id's own spelling in the vocabulary, which is the byte-level encoded form for a token
    /// the model merged and the literal itself for one the release added.
    ///
    /// A token that is a fragment of a multi-byte character is keyed by this spelling rather than by
    /// its bytes, and that is not a formality: byte `0xA1` spells `"¡"`, which the vocabulary also
    /// holds as a real token, so the two share a bucket. Keying the fragment by anything the real
    /// token cannot equal would split a group the reference merges.
    static func tokenSpellings(fromTokenizerJSON url: URL) -> [Int: String] {
        guard let json = tokenizerJSON(at: url) else { return [:] }
        return tokenSpellings(json)
    }

    static func tokenSpellings(_ json: [String: Any]) -> [Int: String] {
        var spellings = [Int: String]()
        if let model = json["model"] as? [String: Any],
           let vocabulary = model["vocab"] as? [String: Int] {
            for (spelling, id) in vocabulary { spellings[id] = spelling }
        }
        for entry in (json["added_tokens"] as? [[String: Any]]) ?? [] {
            if let content = entry["content"] as? String, let id = entry["id"] as? Int {
                spellings[id] = content
            }
        }
        return spellings
    }

    private static func tokenizerJSON(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    /// Every token id's bucket in the collapsed space, and how many buckets there are.
    ///
    /// Buckets are numbered in the order their first member is met, which is what makes the
    /// numbering reproducible from the tokenizer alone.
    static func collapsedTokenSpace(for tokenizer: NFKTokenizer, spellings: [Int: String],
                                    vocabularySize: Int) -> (map: [Int], size: Int) {
        // Buckets are keyed by the key's CODE POINTS, not by the string. Swift compares and hashes
        // strings under canonical equivalence, so two Arabic marks in either order are one key to a
        // `[String: Int]` and two keys to the reference's dictionary, which merges groups the
        // reference keeps apart.
        var buckets = [Data: Int](minimumCapacity: vocabularySize)
        var map = [Int](repeating: 0, count: vocabularySize)
        // Drained per id. `bytes(forTokenId:)` hands back an autoreleased `NSData` and the
        // normalization bridges several more temporaries, so without a pool a whole vocabulary's
        // worth stays live until this returns. Measured, this changes no timing either; it bounds
        // the peak rather than the total.
        //
        // What neither of those explains, and what is still open: running this derivation makes
        // LATER tests in the same process slow, most of a minute for one that takes 0.054 s on its
        // own. Skipping the derivation returns them to normal, so it is the cause; parsing once and
        // pooling the temporaries both left it unchanged, so the mechanism is not simply the size
        // of what it allocates. Do not attribute it to either of those without measuring again.
        for id in 0 ..< vocabularySize { autoreleasepool {
            let bytes = tokenizer.bytes(forTokenId: id) ?? Data()
            // The stdlib's decoder, not `String(data:encoding:)`: that one strips a leading
            // byte-order mark, which silently turns a token beginning U+FEFF into the token without
            // it. It also substitutes the replacement character for an invalid sequence, which is
            // what the reference's own decode does and what the branch below tests for.
            let text = String(decoding: bytes, as: UTF8.self)
            let key: String
            // A token that is a fragment of a multi-byte character has nothing to normalize, so it
            // is keyed by its spelling. A token that spells the replacement character takes the same
            // branch, because a decode cannot tell it from a failure either.
            // SCALARS, not `contains`: a replacement character followed by a combining mark is one
            // grapheme cluster, and `String.contains` matches clusters, so it does not find the bare
            // one. Missing the branch sends a byte fragment through the collapse, where stripping
            // its marks leaves every such token as a lone replacement character in one bucket.
            if text.unicodeScalars.contains("\u{FFFD}") {
                key = spellings[id] ?? text
            } else {
                let normalized = collapsedForm(of: text)
                key = normalized.isEmpty ? text : normalized
            }
            let identity = Data(key.utf8)
            if let existing = buckets[identity] {
                map[id] = existing
            } else {
                map[id] = buckets.count
                buckets[identity] = map[id]
            }
        } }
        return (map, buckets.count)
    }

    /// The release's normalizer chain: compatibility composition, canonical decomposition, the
    /// combining marks dropped, lowercased, runs of space, tab, carriage return and newline
    /// collapsed to one space, then trimmed.
    ///
    /// A token that is exactly one space is parked behind a private-use sentinel across the trim, so
    /// that it survives as a space instead of collapsing to the empty string and joining tokens it
    /// has nothing to do with.
    static func collapsedForm(of text: String) -> String {
        let sentinel = "\u{E000}"
        let decomposed = text.precomposedStringWithCompatibilityMapping
            .decomposedStringWithCanonicalMapping
        // EVERY mark category goes, not the nonspacing ones alone: the reference's `StripAccents`
        // drops spacing and enclosing marks too, which is what collapses an Indic vowel sign onto
        // the consonant it sits beside.
        var value = String(String.UnicodeScalarView(decomposed.unicodeScalars.filter {
            switch $0.properties.generalCategory {
            case .nonspacingMark, .spacingMark, .enclosingMark: return false
            default: return true
            }
        })).lowercased()
        // Written out rather than left to `replacingOccurrences(options: .regularExpression)`, which
        // compiles its pattern on every call, and this runs once per token id. Measured against that
        // form the difference is within the noise, so what this buys is the absence of a per-call
        // compile rather than a speed-up anyone would notice.
        var collapsed = String.UnicodeScalarView()
        var inRun = false
        for scalar in value.unicodeScalars {
            if scalar == " " || scalar == "\t" || scalar == "\r" || scalar == "\n" {
                if !inRun { collapsed.append(" ") }
                inRun = true
            } else {
                collapsed.append(scalar)
                inRun = false
            }
        }
        value = String(collapsed)
        if value == " " { value = sentinel }
        value = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return value.replacingOccurrences(of: sentinel, with: " ")
    }
}
