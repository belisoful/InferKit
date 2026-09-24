//
//  NFKMLXQwen4Exp.swift
//  InferKitMLX
//
//  The Qwen4-Exp decoder (`Qwen4ExpForConditionalGeneration`), which Qwen3.8-Flash-Next is the
//  released 180B instance of. It shares the hybrid family's skeleton — a gated delta-rule recurrence
//  in three quarters of the layers, gated full attention in the rest — and adds four mechanisms that
//  no other model in this package uses:
//
//  - **Hyper-connections.** The residual stream is carried `hc_count` times over, and each block
//    reads a learned mixture of the streams and writes back a learned share to each. The plain
//    pre-normalization of a transformer block is gone: the mixer's own norm does that work.
//  - **Per-layer embeddings over hashed n-grams.** One layer injects features looked up by a hash of
//    each token's 2- and 3-gram, from a table with a distinct prime vocabulary per head.
//  - **A query-sparse-attention indexer.** Every full-attention layer first scores compressed blocks
//    of keys with a small side network, keeps the best `indexer_budget` tokens, and attends to those
//    alone.
//  - **A mixture of 512 experts** with a shared expert beside them.
//
//  Scope: the text decoder. A release also carries a vision tower (`model.visual`, the Qwen3-VL ViT
//  without deepstack, which `NFKMLXQwen3VL` runs) and a multi-token-prediction head (`mtp`), which
//  are separate features rather than parts of the decoder.
//
//  The released weights are 360 GB, so the numerics are measured at a small configuration against
//  transformers' own `Qwen4ExpTextModel`, and the released checkpoint is covered structurally.
//

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// Which kind of layer sits at a given depth.
public enum NFKMLXQwen4ExpLayerKind: String, Sendable {
    /// A gated delta-rule recurrence, linear in sequence length.
    case linearAttention = "linear_attention"
    /// Attention over the tokens an indexer selects, with a gated output. The released `config.json`
    /// spells this `full_attention`; the reference rewrites every such entry to the sparse kind,
    /// because the layer carries an indexer and never attends to the whole prefix.
    case sparseAttention = "qwen_sparse_attention"
}

/// The activation the linear branch's output gate applies.
public enum NFKMLXQwen4ExpGateActivation: String, Sendable {
    case sigmoid
    case silu
}

/// The geometry of a Qwen4-Exp decoder.
public struct NFKMLXQwen4ExpConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var vocabularySize: Int
    public var rmsEpsilon: Float

    // Full-attention layers.
    public var headCount: Int
    public var keyValueHeadCount: Int
    public var headDimensions: Int
    public var ropeTheta: Float
    /// The fraction of each head's channels the rotary embedding turns; the rest pass through.
    public var partialRotaryFactor: Float
    /// How the 3-D positions divide the turned channels, in the interleaved M-RoPE layout.
    public var mropeSection: [Int]

    // The indexer on every full-attention layer.
    public var indexerHeadCount: Int
    public var indexerKeyValueHeadCount: Int
    public var indexerHeadDimensions: Int
    /// How many tokens of complete blocks a query keeps.
    public var indexerBudget: Int
    /// How many consecutive keys average into one index block.
    public var indexerCompressRatio: Int

    // Linear-attention layers.
    public var linearKeyHeadCount: Int
    public var linearKeyHeadDimensions: Int
    public var linearValueHeadCount: Int
    public var linearValueHeadDimensions: Int
    public var linearConvolutionKernel: Int
    public var outputGateActivation: NFKMLXQwen4ExpGateActivation

    // Hyper-connections.
    public var hyperConnectionCount: Int
    public var hyperConnectionRank: Int

    // Mixture of experts.
    public var expertCount: Int
    public var activeExpertCount: Int
    public var expertIntermediateSize: Int
    public var sharedExpertIntermediateSize: Int
    public var normalizesExpertWeights: Bool

    // Per-layer embeddings.
    /// ONE-INDEXED layer ids that carry a PLE module, as the release states them.
    public var pleLayerIDs: [Int]
    public var pleEmbedDimensions: Int
    public var pleConvolutionKernel: Int
    public var ngramSize: Int
    public var headsPerNgram: Int
    public var ngramVocabularyBase: Int
    public var ngramVocabularyDivisor: Int
    /// Seeds the per-layer hash multipliers. The release's tables were built with this value.
    public var hashSeed: Int
    /// The token the n-gram context is padded with, and which segments its history.
    public var endOfSequenceToken: Int

    public var tiesWordEmbeddings: Bool
    public var layerTypes: [NFKMLXQwen4ExpLayerKind]

    public init(hiddenSize: Int = 2560, layerCount: Int = 48, vocabularySize: Int = 248_320,
                rmsEpsilon: Float = 1e-6,
                headCount: Int = 24, keyValueHeadCount: Int = 2, headDimensions: Int = 256,
                ropeTheta: Float = 10_000_000, partialRotaryFactor: Float = 0.25,
                mropeSection: [Int] = [11, 11, 10],
                indexerHeadCount: Int = 4, indexerKeyValueHeadCount: Int = 1,
                indexerHeadDimensions: Int = 128, indexerBudget: Int = 2048,
                indexerCompressRatio: Int = 4,
                linearKeyHeadCount: Int = 16, linearKeyHeadDimensions: Int = 128,
                linearValueHeadCount: Int = 48, linearValueHeadDimensions: Int = 128,
                linearConvolutionKernel: Int = 4,
                outputGateActivation: NFKMLXQwen4ExpGateActivation = .sigmoid,
                hyperConnectionCount: Int = 4, hyperConnectionRank: Int = 320,
                expertCount: Int = 512, activeExpertCount: Int = 10,
                expertIntermediateSize: Int = 640, sharedExpertIntermediateSize: Int = 640,
                normalizesExpertWeights: Bool = true,
                pleLayerIDs: [Int] = [2], pleEmbedDimensions: Int? = nil,
                pleConvolutionKernel: Int = 4, ngramSize: Int = 3, headsPerNgram: Int = 8,
                ngramVocabularyBase: Int = 20_000_000, ngramVocabularyDivisor: Int = 128,
                hashSeed: Int = 1234, endOfSequenceToken: Int = 248_044,
                fullAttentionInterval: Int = 4,
                tiesWordEmbeddings: Bool = false,
                layerTypes: [NFKMLXQwen4ExpLayerKind]? = nil) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.vocabularySize = vocabularySize
        self.rmsEpsilon = rmsEpsilon
        self.headCount = headCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimensions = headDimensions
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.mropeSection = mropeSection
        self.indexerHeadCount = indexerHeadCount
        self.indexerKeyValueHeadCount = indexerKeyValueHeadCount
        self.indexerHeadDimensions = indexerHeadDimensions
        self.indexerBudget = indexerBudget
        self.indexerCompressRatio = indexerCompressRatio
        self.linearKeyHeadCount = linearKeyHeadCount
        self.linearKeyHeadDimensions = linearKeyHeadDimensions
        self.linearValueHeadCount = linearValueHeadCount
        self.linearValueHeadDimensions = linearValueHeadDimensions
        self.linearConvolutionKernel = linearConvolutionKernel
        self.outputGateActivation = outputGateActivation
        self.hyperConnectionCount = hyperConnectionCount
        self.hyperConnectionRank = hyperConnectionRank
        self.expertCount = expertCount
        self.activeExpertCount = activeExpertCount
        self.expertIntermediateSize = expertIntermediateSize
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.normalizesExpertWeights = normalizesExpertWeights
        self.pleLayerIDs = pleLayerIDs.sorted()
        self.pleEmbedDimensions = pleEmbedDimensions ?? hiddenSize
        self.pleConvolutionKernel = pleConvolutionKernel
        self.ngramSize = ngramSize
        self.headsPerNgram = headsPerNgram
        self.ngramVocabularyBase = ngramVocabularyBase
        self.ngramVocabularyDivisor = ngramVocabularyDivisor
        self.hashSeed = hashSeed
        self.endOfSequenceToken = endOfSequenceToken
        self.tiesWordEmbeddings = tiesWordEmbeddings
        self.layerTypes = layerTypes ?? (0 ..< layerCount).map {
            ($0 + 1) % fullAttentionInterval == 0 ? .sparseAttention : .linearAttention
        }
    }

    /// The width every hyper-connection module works over.
    var hyperWidth: Int { hyperConnectionCount * hiddenSize }

    /// The width the linear branch's fused q/k/v projection emits.
    var linearQKVWidth: Int {
        2 * linearKeyHeadCount * linearKeyHeadDimensions
            + linearValueHeadCount * linearValueHeadDimensions
    }

    /// The width of the linear branch's value stream, which its gate and output projection match.
    var linearValueWidth: Int { linearValueHeadCount * linearValueHeadDimensions }

    /// How many of a head's channels the rotary embedding turns.
    var rotaryDimensions: Int { Int(Float(headDimensions) * partialRotaryFactor) }

    /// How many hashed heads the n-gram table carries: one set per order from 2 up.
    var ngramHeadCount: Int { (ngramSize - 1) * headsPerNgram }

    /// How many complete blocks a query keeps.
    var indexerBlockBudget: Int { indexerBudget / indexerCompressRatio }

    /// The released `Qwen/Qwen3.8-Flash-Next`.
    public static let qwen3_8FlashNext = NFKMLXQwen4ExpConfiguration()

    /// The size the oracle measures, chosen so that every added mechanism bites: the indexer's budget
    /// is smaller than the block count a late query sees, the sequence fills no whole number of
    /// blocks, and one layer carries a per-layer embedding.
    public static let tiny = NFKMLXQwen4ExpConfiguration(
        hiddenSize: 32, layerCount: 4, vocabularySize: 128,
        headCount: 4, keyValueHeadCount: 2, headDimensions: 32,
        ropeTheta: 10_000, partialRotaryFactor: 0.25, mropeSection: [2, 1, 1],
        indexerHeadCount: 2, indexerKeyValueHeadCount: 1, indexerHeadDimensions: 8,
        indexerBudget: 4, indexerCompressRatio: 2,
        linearKeyHeadCount: 2, linearKeyHeadDimensions: 8,
        linearValueHeadCount: 4, linearValueHeadDimensions: 8,
        hyperConnectionCount: 3, hyperConnectionRank: 8,
        expertCount: 8, activeExpertCount: 2,
        expertIntermediateSize: 16, sharedExpertIntermediateSize: 16,
        pleLayerIDs: [1], pleEmbedDimensions: 32,
        ngramSize: 3, headsPerNgram: 2, ngramVocabularyBase: 1000,
        ngramVocabularyDivisor: 8, endOfSequenceToken: 2)
}

/// The family's normalization, `x · (1 + w)`, optionally over fixed-width groups of the last axis.
///
/// A grouped norm normalizes each hyper-connection stream on its own and then scales the whole width
/// by one weight vector. Normalizing the concatenated streams together is a different function.
final class NFKQwen4ExpNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float
    let groupSize: Int?

    init(dimensions: Int, groupSize: Int? = nil, eps: Float) {
        precondition(groupSize.map { dimensions % $0 == 0 } ?? true,
                     "a grouped norm divides its width into whole groups")
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        epsilon = eps
        self.groupSize = groupSize
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let promoted = x.asType(.float32)
        var normalized: MLXArray
        if let groupSize {
            let grouped = promoted.reshaped(promoted.shape.dropLast() + [-1, groupSize])
            let scaled = grouped * rsqrt((grouped * grouped).mean(axis: -1, keepDims: true) + epsilon)
            normalized = scaled.reshaped(promoted.shape)
        } else {
            normalized = promoted * rsqrt((promoted * promoted).mean(axis: -1, keepDims: true) + epsilon)
        }
        return (normalized * (1 + weight.asType(.float32))).asType(x.dtype)
    }
}

/// The gated normalization inside the recurrence: a PLAIN `w · x̂`, times an activation of the gate.
///
/// The activation is the configuration's `output_gate_type`. The released Qwen3.8-Flash-Next sets it
/// to `sigmoid`, where the rest of the hybrid family uses `silu`, and the two differ in every number.
final class NFKQwen4ExpGatedNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let epsilon: Float
    let activation: NFKMLXQwen4ExpGateActivation

    init(dimensions: Int, eps: Float, activation: NFKMLXQwen4ExpGateActivation) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        epsilon = eps
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let promoted = x.asType(.float32)
        let normalized = promoted * rsqrt((promoted * promoted).mean(axis: -1, keepDims: true) + epsilon)
        let scaled = weight * normalized.asType(x.dtype)
        let g = gate.asType(.float32)
        let activated = activation == .sigmoid ? sigmoid(g) : silu(g)
        return (scaled * activated).asType(x.dtype)
    }
}

/// A hyper-connection: the learned read from, and write back to, the several residual streams.
///
/// @discussion The streams are carried concatenated on the last axis. Reading normalizes them, mixes
/// them with a low-rank gate, and averages; writing scales the block's output per stream. The mixer
/// at the end of the stack reads without writing, which is what `combines` false means.
final class NFKQwen4ExpGatedResidual: Module {
    @ModuleInfo(key: "hc_norm") var norm: NFKQwen4ExpNorm
    @ModuleInfo(key: "input_mix_weight_down") var mixDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var mixUp: Linear
    @ModuleInfo(key: "block_inject_weight") var inject: Linear?

    let streams: Int
    let hiddenSize: Int

    init(_ c: NFKMLXQwen4ExpConfiguration, combines: Bool = true) {
        streams = c.hyperConnectionCount
        hiddenSize = c.hiddenSize
        _norm.wrappedValue = NFKQwen4ExpNorm(dimensions: c.hyperWidth, groupSize: c.hiddenSize,
                                             eps: c.rmsEpsilon)
        _mixDown.wrappedValue = Linear(c.hyperWidth, c.hyperConnectionRank, bias: false)
        _mixUp.wrappedValue = Linear(c.hyperConnectionRank, c.hyperWidth, bias: false)
        _inject.wrappedValue = combines ? Linear(c.hyperWidth, c.hyperConnectionCount, bias: false) : nil
        super.init()
    }

    /// `[batch, length, streams · hidden]` → the mixed read and the per-stream write shares. The
    /// shares are nil where this module only reads. Both the mixture and the shares are derived from
    /// the NORMALIZED streams; the residual the caller adds back to is the raw input.
    func callAsFunction(_ hyper: MLXArray) -> (read: MLXArray, shares: MLXArray?) {
        let (batch, length) = (hyper.dim(0), hyper.dim(1))
        let normalized = norm(hyper)
        var mix = NFKReferenceRounding.silu(NFKReferenceRounding.divided(mixDown(normalized), by: Float(streams)))
        mix = NFKReferenceRounding.sigmoid(mixUp(mix)).reshaped([batch, length, streams, hiddenSize])
        let read = NFKReferenceRounding.mean(mix * normalized.reshaped([batch, length, streams, hiddenSize]), axis: -2,
                                           keepDims: false)
        guard let inject else { return (read, nil) }
        return (read, 2 * NFKReferenceRounding.sigmoid(NFKReferenceRounding.divided(inject(normalized), by: Float(streams))))
    }
}

/// The hashed n-gram vocabulary a per-layer embedding is looked up in.
///
/// @discussion Nothing about the table is learned except its rows. Each head's vocabulary size is a
/// distinct prime above `ngram_vocab_size_base`, taken in order across the layers, and each layer's
/// hash multipliers come from a SplitMix64 stream seeded by the layer's index. The release ships both
/// as buffers; this type derives them, and the derivation is checked against the shipped values.
enum NFKQwen4ExpNGramHash {
    private static let gamma: UInt64 = 0x9E37_79B9_7F4A_7C15
    private static let mix1: UInt64 = 0xBF58_476D_1CE4_E5B9
    private static let mix2: UInt64 = 0x94D0_49BB_1331_11EB
    private static let seedStride = 10007

    static func splitMix64(_ value: UInt64) -> UInt64 {
        var v = value &+ gamma
        v = (v ^ (v >> 30)) &* mix1
        v = (v ^ (v >> 27)) &* mix2
        return v ^ (v >> 31)
    }

    /// One multiplier per n-gram position, odd so that it is invertible modulo a power of two.
    static func multipliers(vocabularySize: Int, ngramSize: Int, layerIndex: Int, seed: Int) -> [Int64] {
        let maximum = Int64.max / Int64(max(vocabularySize, 1))
        let bound = UInt64(max(1, maximum / 2))
        let base = UInt64(bitPattern: Int64(seed + seedStride * layerIndex))
        return (0 ..< ngramSize).map { index in
            let value = base &+ gamma &* UInt64(index + 1)
            return Int64(2 * (splitMix64(value) % bound) + 1)
        }
    }

    static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor = 3
        while divisor * divisor <= value {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }

    /// The `count`-th prime strictly greater than `start`.
    static func nthPrimeAfter(_ start: Int, count: Int) -> Int {
        var prime = start
        for _ in 0 ..< count {
            prime += 1
            while !isPrime(prime) { prime += 1 }
        }
        return prime
    }

    /// Each head's vocabulary size, taken in order across every PLE layer's heads.
    static func headVocabularySizes(_ c: NFKMLXQwen4ExpConfiguration, layerIndex: Int) -> [Int] {
        (0 ..< c.ngramHeadCount).map { head in
            nthPrimeAfter(c.ngramVocabularyBase - 1, count: layerIndex * c.ngramHeadCount + head + 1)
        }
    }
}

/// The per-layer embedding's lookup table over hashed n-grams.
///
/// One row per hashed id, shared by every head through per-head offsets into one table. The hashing
/// runs on token ids in `Int64`, which MLX has no bitwise-xor over, and it is index arithmetic rather
/// than differentiable math, so it runs on the host and the result is one gather.
final class NFKQwen4ExpNGramEmbedding: Module {
    @ModuleInfo(key: "ngram_embedding") var table: Embedding

    let configuration: NFKMLXQwen4ExpConfiguration
    let layerIndex: Int
    let headVocabularySizes: [Int]
    let headOffsets: [Int]
    let multipliers: [Int64]
    /// How many earlier tokens each position's n-grams reach back over.
    var contextLength: Int { configuration.ngramSize - 1 }

    init(_ c: NFKMLXQwen4ExpConfiguration, layerIndex: Int) {
        configuration = c
        self.layerIndex = layerIndex
        headVocabularySizes = NFKQwen4ExpNGramHash.headVocabularySizes(c, layerIndex: layerIndex)
        var offsets = [Int]()
        var total = 0
        for size in headVocabularySizes {
            offsets.append(total)
            total += size
        }
        headOffsets = offsets
        multipliers = NFKQwen4ExpNGramHash.multipliers(vocabularySize: c.vocabularySize,
                                                       ngramSize: c.ngramSize,
                                                       layerIndex: layerIndex, seed: c.hashSeed)
        let padded = (total + c.ngramVocabularyDivisor - 1) / c.ngramVocabularyDivisor
            * c.ngramVocabularyDivisor
        _table.wrappedValue = Embedding(embeddingCount: padded,
                                        dimensions: c.pleEmbedDimensions / c.ngramHeadCount)
        super.init()
    }

    /// Shifts a row right by `shift`, refusing to read across an end-of-sequence token: a position
    /// whose segment is shorter than the shift reads the end token instead.
    private func shiftedRight(_ tokens: [Int], by shift: Int) -> [Int] {
        guard shift > 0 else { return tokens }
        let end = configuration.endOfSequenceToken
        var segmentStart = 0
        var previousEnd = -1
        var result = [Int]()
        result.reserveCapacity(tokens.count)
        for position in 0 ..< tokens.count {
            segmentStart = previousEnd + 1
            let source = position - shift
            let valid = (position - segmentStart) >= shift && source >= 0
            result.append(valid ? tokens[source] : end)
            if tokens[position] == end { previousEnd = position }
        }
        return result
    }

    /// `tokens` `[batch][length]` → the hashed row index per head, `[batch, length, heads]`.
    func indices(tokens: [[Int]], context: [[Int]]? = nil) -> MLXArray {
        let c = configuration
        let heads = c.ngramHeadCount
        let length = tokens.first?.count ?? 0
        var flat = [Int32]()
        flat.reserveCapacity(tokens.count * length * heads)

        for (row, ids) in tokens.enumerated() {
            let previous = context?[row] ?? [Int](repeating: c.endOfSequenceToken, count: contextLength)
            let history = previous + ids
            let shifted = (0 ..< c.ngramSize).map { shiftedRight(history, by: $0) }
            // Only the tail that corresponds to the current tokens is kept; the context exists to give
            // the first of them a full n-gram.
            let offset = history.count - length
            for position in offset ..< history.count {
                for order in 2 ... c.ngramSize {
                    var mixed = Int64(shifted[0][position]) &* multipliers[0]
                    for slot in 1 ..< order {
                        mixed ^= Int64(shifted[slot][position]) &* multipliers[slot]
                    }
                    let start = (order - 2) * c.headsPerNgram
                    for head in start ..< (start + c.headsPerNgram) {
                        let size = Int64(headVocabularySizes[head])
                        flat.append(Int32(mixed % size + Int64(headOffsets[head])))
                    }
                }
            }
        }
        return MLXArray(flat).reshaped([tokens.count, length, heads])
    }

    /// `[batch, length, heads]` row indices → `[batch, length, ple embed]`.
    func callAsFunction(_ indices: MLXArray) -> MLXArray {
        let looked = table(indices)
        return looked.reshaped([indices.dim(0), indices.dim(1), -1])
    }
}

/// The per-layer embedding: hashed n-gram features, gated by the streams they are injected into.
///
/// @discussion The n-gram embedding projects to one value and one key per residual stream. Each
/// stream's normalized activation scores its key, and the score — square-rooted with its sign kept,
/// then squashed — gates the shared value. A dilated depthwise convolution over the gated values adds
/// local lexical context. The convolution's dilation is the n-gram size, so each tap reads one
/// n-gram further back rather than one token.
final class NFKQwen4ExpPLE: Module {
    @ModuleInfo(key: "ple_embedding") var embedding: NFKQwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProjection: Linear
    @ModuleInfo(key: "value_proj") var valueProjection: Linear
    @ModuleInfo(key: "norm_key") var keyNorm: NFKQwen4ExpNorm
    @ModuleInfo(key: "norm_query") var queryNorm: NFKQwen4ExpNorm
    @ModuleInfo(key: "norm_conv") var convolutionNorm: NFKQwen4ExpNorm
    @ModuleInfo(key: "conv1d") var convolution: Conv1d

    let configuration: NFKMLXQwen4ExpConfiguration
    /// How many earlier positions the dilated convolution reaches.
    let convolutionState: Int

    init(_ c: NFKMLXQwen4ExpConfiguration, layerIndex: Int) {
        configuration = c
        convolutionState = (c.pleConvolutionKernel - 1) * c.ngramSize
        _embedding.wrappedValue = NFKQwen4ExpNGramEmbedding(c, layerIndex: layerIndex)
        _keyProjection.wrappedValue = Linear(c.pleEmbedDimensions, c.hyperWidth, bias: false)
        _valueProjection.wrappedValue = Linear(c.pleEmbedDimensions, c.hiddenSize, bias: false)
        _keyNorm.wrappedValue = NFKQwen4ExpNorm(dimensions: c.hyperWidth, groupSize: c.hiddenSize,
                                                eps: c.rmsEpsilon)
        _queryNorm.wrappedValue = NFKQwen4ExpNorm(dimensions: c.hyperWidth, groupSize: c.hiddenSize,
                                                  eps: c.rmsEpsilon)
        _convolutionNorm.wrappedValue = NFKQwen4ExpNorm(dimensions: c.hyperWidth, groupSize: c.hiddenSize,
                                                        eps: c.rmsEpsilon)
        _convolution.wrappedValue = Conv1d(inputChannels: c.hyperWidth, outputChannels: c.hyperWidth,
                                           kernelSize: c.pleConvolutionKernel,
                                           dilation: c.ngramSize, groups: c.hyperWidth, bias: false)
        super.init()
    }

    private func shortConvolution(_ x: MLXArray) -> MLXArray {
        let padded = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((convolutionState, 0)),
                                        IntOrPair((0, 0))])
        return NFKReferenceRounding.silu(convolution(padded))
    }

    /// `hyper` is the layer's incoming streams `[batch, length, streams · hidden]`; `indices` are the
    /// hashed n-gram rows for the same positions.
    func callAsFunction(_ hyper: MLXArray, indices: MLXArray) -> MLXArray {
        let c = configuration
        let (batch, length) = (hyper.dim(0), hyper.dim(1))
        let features = embedding(indices).asType(hyper.dtype)
        let shape = [batch, length, c.hyperConnectionCount, c.hiddenSize]
        let keys = keyNorm(keyProjection(features)).reshaped(shape)
        let value = valueProjection(features)
        let queries = queryNorm(hyper).reshaped(shape)

        // The products sum in float32 and round, as torch reduces a half-precision tensor; the divide
        // by a Python float and the square root are formed wide and round once.
        let products = (keys * queries).asType(.float32).sum(axis: -1, keepDims: true).asType(keys.dtype)
        var gate = NFKReferenceRounding.divided(products, by: sqrt(Float(c.hiddenSize)))
        gate = sign(gate) * NFKReferenceRounding.wide(maximum(abs(gate), 1e-6)) { sqrt($0) }
        let gated = (NFKReferenceRounding.sigmoid(gate) * value.expandedDimensions(axis: -2)).reshaped([batch, length, -1])
        return gated + shortConvolution(convolutionNorm(gated))
    }
}

/// The query-sparse-attention indexer: which earlier tokens a query is allowed to attend to.
///
/// @discussion A small side network projects one query per indexer head and one key per token. The
/// keys of each complete run of `compress_ratio` tokens average into a block key, positioned at the
/// block's FIRST token. Every block scores against the query as a sum of rectified head dot products,
/// the best `indexer_budget / compress_ratio` blocks are kept, and the trailing tokens that do not
/// fill a block are always kept. The result is a mask, so the attention itself is unchanged.
///
/// Prefill only: the visible set of a query is its own causal prefix, which makes every block's
/// membership the same for every query and its pooled key computable once. A cached decode would
/// have to pool against the cache instead.
final class NFKQwen4ExpIndexer: Module {
    @ModuleInfo(key: "index_qk_proj") var projection: Linear
    @ModuleInfo(key: "q_layernorm") var queryNorm: NFKQwen4ExpNorm
    @ModuleInfo(key: "k_layernorm") var keyNorm: NFKQwen4ExpNorm

    let configuration: NFKMLXQwen4ExpConfiguration

    init(_ c: NFKMLXQwen4ExpConfiguration) {
        configuration = c
        _projection.wrappedValue = Linear(
            c.hiddenSize,
            (c.indexerHeadCount + c.indexerKeyValueHeadCount) * c.indexerHeadDimensions, bias: false)
        _queryNorm.wrappedValue = NFKQwen4ExpNorm(dimensions: c.indexerHeadDimensions, eps: c.rmsEpsilon)
        _keyNorm.wrappedValue = NFKQwen4ExpNorm(dimensions: c.indexerHeadDimensions, eps: c.rmsEpsilon)
        super.init()
    }

    /// `[batch, length, kv length]` — true where the query may attend.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.dim(0), x.dim(1))
        let ratio = c.indexerCompressRatio
        let dimensions = c.indexerHeadDimensions

        let projected = projection(x)
        let queryWidth = c.indexerHeadCount * dimensions
        var queries = projected[.ellipsis, 0 ..< queryWidth]
            .reshaped([batch, length, c.indexerHeadCount, dimensions])
        let rawKeys = projected[.ellipsis, queryWidth...].reshaped([batch, length, dimensions])

        queries = queryNorm(queries)
        queries = NFKQwen4ExpAttention.rotated(queries, cos: cos.expandedDimensions(axis: 2),
                                               sin: sin.expandedDimensions(axis: 2))

        let positions = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, length, 1])
        let tokenIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, 1, length])
        let causal = broadcast(tokenIndex .<= positions, to: [batch, length, length])

        // Shorter than one block: every query's whole prefix is tail, and nothing is discarded.
        let blocks = length / ratio
        if blocks == 0 { return causal }

        // Every block pools the same tokens for every query, so the pooling runs once. The reference
        // averages in float32 before normalizing, which matters at the checkpoint's own precision.
        let grouped = rawKeys[0..., 0 ..< (blocks * ratio), 0...]
            .reshaped([batch, blocks, ratio, dimensions])
        var blockKeys = keyNorm(grouped.asType(.float32).mean(axis: 2).asType(rawKeys.dtype))
        let starts = MLXArray((0 ..< blocks).map { Int32($0 * ratio) })
        blockKeys = NFKQwen4ExpAttention.rotated(blockKeys,
                                                 cos: take(cos, starts, axis: 1),
                                                 sin: take(sin, starts, axis: 1))

        // [batch, length, heads, blocks] → rectified, summed over the heads.
        let paired = matmul(queries.asType(.float32),
                            blockKeys.asType(.float32).expandedDimensions(axis: 1).swappedAxes(-1, -2))
        let scores = relu(paired).sum(axis: 2) / sqrt(Float(dimensions))

        // A query sees the blocks that end at or before it. Where it sees no more than the budget,
        // it keeps them all, so one rule covers both cases: rank among the visible, under the budget.
        // The block count is a FLOOR division — MLX's `/` on integers promotes to float, which would
        // let a query see the block it sits halfway through.
        let blockIndex = MLXArray((0 ..< blocks).map { Int32($0) }).reshaped([1, 1, blocks])
        let complete = (positions + 1).floorDivide(Int32(ratio))
        let visible = blockIndex .< complete
        let masked = MLX.where(visible, scores, MLXArray(-Float.infinity))
        let kept = min(c.indexerBlockBudget, blocks)
        let order = broadcast(argSort(-masked, axis: -1)[.ellipsis, 0 ..< kept],
                              to: [batch, length, kept])
        // MLX scatters no boolean, so the selection is accumulated as counts and thresholded.
        let counts = putAlong(MLXArray.zeros([batch, length, blocks], type: Int32.self),
                              order, values: MLXArray.ones([batch, length, kept], type: Int32.self),
                              axis: -1)
        let selected = (counts .> 0) .&& visible

        // A block's decision covers each of its tokens. The tokens past the last complete block are
        // kept as well, up to the query itself: they are what the query sees and no block holds.
        let owner = MLXArray((0 ..< length).map { Int32(min($0 / ratio, blocks - 1)) })
        let tokens = takeAlong(selected, broadcast(owner.reshaped([1, 1, length]),
                                                   to: [batch, length, length]), axis: -1)
        let tail = broadcast(tokenIndex .>= (complete * Int32(ratio)), to: [batch, length, length])
        return (tokens .|| tail) .&& causal
    }
}

/// Attention over the tokens the indexer selects, with a gated output.
///
/// The query projection emits twice the width a plain one would: viewed as `[.., heads, 2 · headDim]`
/// and split on the LAST axis, so each head's queries and gate are adjacent. Only the leading
/// `rotaryDimensions` channels of each head turn.
final class NFKQwen4ExpAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: NFKQwen4ExpNorm
    @ModuleInfo(key: "k_norm") var keyNorm: NFKQwen4ExpNorm
    @ModuleInfo(key: "indexer") var indexer: NFKQwen4ExpIndexer

    let configuration: NFKMLXQwen4ExpConfiguration

    init(_ c: NFKMLXQwen4ExpConfiguration) {
        configuration = c
        let queries = c.headCount * c.headDimensions
        _queryProjection.wrappedValue = Linear(c.hiddenSize, queries * 2, bias: false)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions,
                                             bias: false)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.keyValueHeadCount * c.headDimensions,
                                               bias: false)
        _outputProjection.wrappedValue = Linear(queries, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = NFKQwen4ExpNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        _keyNorm.wrappedValue = NFKQwen4ExpNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        _indexer.wrappedValue = NFKQwen4ExpIndexer(c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.dim(0), x.dim(1))
        let width = c.headCount * c.headDimensions

        // The indexer reads the same hidden states the projections do, before them.
        let selected = indexer(x, cos: cos, sin: sin).expandedDimensions(axis: 1)
        let allowed = mask .&& selected

        let paired = queryProjection(x).reshaped([batch, length, c.headCount, c.headDimensions * 2])
        var queries = queryNorm(paired[.ellipsis, 0 ..< c.headDimensions])
        let gate = paired[.ellipsis, c.headDimensions...].reshaped([batch, length, width])
        var keys = keyNorm(keyProjection(x)
            .reshaped([batch, length, c.keyValueHeadCount, c.headDimensions]))
        var values = valueProjection(x)
            .reshaped([batch, length, c.keyValueHeadCount, c.headDimensions])

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let table = (cos: cos.expandedDimensions(axis: 1), sin: sin.expandedDimensions(axis: 1))
        queries = NFKQwen4ExpAttention.rotated(queries, cos: table.cos, sin: table.sin)
        keys = NFKQwen4ExpAttention.rotated(keys, cos: table.cos, sin: table.sin)

        let additive = MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity))
        let attended = NFKReferenceRounding.attention(
            queries: queries, keys: keys, values: values,
            scale: 1 / sqrt(Float(c.headDimensions)), mask: additive.asType(queries.dtype))
        let result = attended.transposed(0, 2, 1, 3).reshaped([batch, length, width])
        return outputProjection(result * NFKReferenceRounding.sigmoid(gate))
    }

    /// Rotates the leading `cos.dim(-1)` channels and carries the rest through.
    static func rotated(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let turned = cos.dim(-1)
        guard turned > 0 else { return x }
        let head = x[.ellipsis, 0 ..< turned]
        // The tables take the input's type, as transformers rounds them; float32 tables would promote a
        // half-precision layer to float32 from here on.
        let (cos, sin) = (cos.asType(x.dtype), sin.asType(x.dtype))
        let turnedPart = head * cos + NFKLMAttention.rotateHalf(head) * sin
        guard turned < x.dim(-1) else { return turnedPart }
        return concatenated([turnedPart, x[.ellipsis, turned...]], axis: -1)
    }
}

/// The gated delta-rule recurrence that replaces attention in three quarters of the layers.
///
/// One fused projection produces queries, keys, and values; a depthwise convolution mixes each over a
/// short window; a per-head decay and write strength drive a fixed-size state, which the queries then
/// read. The state is `[value heads, key dimensions, value dimensions]` and does not grow with the
/// sequence, which is what makes the layer linear.
final class NFKQwen4ExpLinearAttention: Module {
    @ModuleInfo(key: "in_proj_qkv") var qkvProjection: Linear
    @ModuleInfo(key: "in_proj_z") var gateProjection: Linear
    @ModuleInfo(key: "in_proj_a") var decayProjection: Linear
    @ModuleInfo(key: "in_proj_b") var writeProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear
    @ModuleInfo(key: "norm") var norm: NFKQwen4ExpGatedNorm
    @ModuleInfo(key: "conv1d") var convolution: Conv1d

    @ParameterInfo(key: "A_log") var decayLog: MLXArray
    @ParameterInfo(key: "dt_bias") var stepBias: MLXArray

    let configuration: NFKMLXQwen4ExpConfiguration

    init(_ c: NFKMLXQwen4ExpConfiguration) {
        configuration = c
        _qkvProjection.wrappedValue = Linear(c.hiddenSize, c.linearQKVWidth, bias: false)
        _gateProjection.wrappedValue = Linear(c.hiddenSize, c.linearValueWidth, bias: false)
        _decayProjection.wrappedValue = Linear(c.hiddenSize, c.linearValueHeadCount, bias: false)
        _writeProjection.wrappedValue = Linear(c.hiddenSize, c.linearValueHeadCount, bias: false)
        _outputProjection.wrappedValue = Linear(c.linearValueWidth, c.hiddenSize, bias: false)
        _norm.wrappedValue = NFKQwen4ExpGatedNorm(dimensions: c.linearValueHeadDimensions,
                                                  eps: c.rmsEpsilon,
                                                  activation: c.outputGateActivation)
        _convolution.wrappedValue = Conv1d(inputChannels: c.linearQKVWidth,
                                           outputChannels: c.linearQKVWidth,
                                           kernelSize: c.linearConvolutionKernel,
                                           groups: c.linearQKVWidth, bias: false)
        _decayLog.wrappedValue = MLXArray.zeros([c.linearValueHeadCount])
        _stepBias.wrappedValue = MLXArray.zeros([c.linearValueHeadCount])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.dim(0), x.dim(1))

        // Causal short convolution: pad on the left so a position sees only itself and its history.
        let projected = qkvProjection(x)
        let padded = padded(projected, widths: [IntOrPair((0, 0)),
                                                IntOrPair((c.linearConvolutionKernel - 1, 0)),
                                                IntOrPair((0, 0))])
        let mixed = NFKReferenceRounding.silu(convolution(padded))

        let keyWidth = c.linearKeyHeadCount * c.linearKeyHeadDimensions
        let group = c.linearValueHeadCount / c.linearKeyHeadCount
        // Key heads are shared across a group of value heads, as grouped-query attention shares them.
        // The delta rule reads and writes a unit-norm key space, normalized in the input's type and
        // then widened: the recurrence runs in float32, as the reference's does.
        let queries = NFKHybridLinearAttention.unitNorm(repeated(mixed[0..., 0..., 0 ..< keyWidth]
            .reshaped([batch, length, c.linearKeyHeadCount, c.linearKeyHeadDimensions]), count: group, axis: 2))
            .asType(.float32)
        let keys = NFKHybridLinearAttention.unitNorm(repeated(mixed[0..., 0..., keyWidth ..< (2 * keyWidth)]
            .reshaped([batch, length, c.linearKeyHeadCount, c.linearKeyHeadDimensions]), count: group, axis: 2))
            .asType(.float32)
        let values = mixed[0..., 0..., (2 * keyWidth)...]
            .reshaped([batch, length, c.linearValueHeadCount, c.linearValueHeadDimensions]).asType(.float32)

        // `A_log` is stored as a log so the rate stays positive; the reference keeps the decay as a
        // LOG rate, formed in float32, and exponentiates it inside the recurrence.
        let decay = exp(-exp(decayLog.asType(.float32))
                        * softplus(decayProjection(x).asType(.float32) + stepBias.asType(.float32)))
        let write = NFKReferenceRounding.sigmoid(writeProjection(x)).asType(.float32)

        var state = MLXArray.zeros([batch, c.linearValueHeadCount,
                                    c.linearKeyHeadDimensions, c.linearValueHeadDimensions])
        var outputs = [MLXArray]()
        outputs.reserveCapacity(length)
        let scale = 1 / sqrt(Float(c.linearKeyHeadDimensions))

        for step in 0 ..< length {
            let q = queries[0..., step] * scale
            let k = keys[0..., step]
            let v = values[0..., step]
            let g = decay[0..., step].reshaped([batch, c.linearValueHeadCount, 1, 1])
            let b = write[0..., step].reshaped([batch, c.linearValueHeadCount, 1])

            // Decay FIRST, then read what the decayed state holds for this key, then write the
            // correction toward v. Reading before decaying is a different recurrence entirely.
            state = state * g
            let held = (state * k.expandedDimensions(axis: -1)).sum(axis: 2)
            let correction = (v - held) * b
            state = state + k.expandedDimensions(axis: -1) * correction.expandedDimensions(axis: 2)
            outputs.append((state * q.expandedDimensions(axis: -1)).sum(axis: 2))
        }

        let read = stacked(outputs, axis: 1).asType(x.dtype)
        let gate = gateProjection(x).reshaped([batch, length, c.linearValueHeadCount,
                                               c.linearValueHeadDimensions])
        let gated = norm(read, gate: gate)
        return outputProjection(gated.reshaped([batch, length, c.linearValueWidth]))
    }
}

/// The experts, stored as one weight per projection across all of them.
///
/// @discussion The release names these `gate_up_proj` and `down_proj` with no `.weight` suffix,
/// because they are bare parameters rather than layers, and this module matches that naming so the
/// loader needs no rewrite. `gate_up_proj` is `[experts, 2 · width, hidden]` with the gate in the
/// FIRST half of its rows and the lift in the second; the gpt-oss form interleaves them instead.
final class NFKQwen4ExpExperts: Module {
    /// Nil where the experts are paged: each projection is then read from ``pagers``.
    @ParameterInfo(key: "gate_up_proj") var gateUp: MLXArray?
    @ParameterInfo(key: "down_proj") var down: MLXArray?

    let intermediateSize: Int
    private(set) var pagers: (gateUp: NFKMLXExpertPager, down: NFKMLXExpertPager)?

    init(_ c: NFKMLXQwen4ExpConfiguration) {
        intermediateSize = c.expertIntermediateSize
        let scale = sqrt(1 / Float(c.hiddenSize))
        _gateUp.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [c.expertCount, 2 * c.expertIntermediateSize, c.hiddenSize])
        _down.wrappedValue = MLXRandom.uniform(
            low: -sqrt(1 / Float(c.expertIntermediateSize)),
            high: sqrt(1 / Float(c.expertIntermediateSize)),
            [c.expertCount, c.hiddenSize, c.expertIntermediateSize])
        super.init()
    }

    /// The paged form: no parameters, both projections read from `store` under `path`, this module's
    /// own key path.
    init(pagedFrom store: NFKMLXExpertStore, path: String, intermediateSize: Int) {
        pagers = (NFKMLXExpertPager(store: store, group: path + ".gate_up_proj"),
                  NFKMLXExpertPager(store: store, group: path + ".down_proj"))
        self.intermediateSize = intermediateSize
        super.init()
    }

    /// `x` `[tokens, hidden]` and `experts` `[tokens, k]` → `[tokens, k, hidden]`.
    func callAsFunction(_ x: MLXArray, experts: MLXArray) -> MLXArray {
        let expanded = x.expandedDimensions(axes: [-2, -3])
        let fused = pagers?.gateUp.gatherMM(expanded, experts: experts)
            ?? gatherMM(expanded, gateUp!.swappedAxes(-1, -2), rhsIndices: experts)
        let gate = fused[.ellipsis, 0 ..< intermediateSize]
        let lift = fused[.ellipsis, intermediateSize...]
        let hidden = NFKReferenceRounding.silu(gate) * lift
        let projected = pagers?.down.gatherMM(hidden, experts: experts)
            ?? gatherMM(hidden, down!.swappedAxes(-1, -2), rhsIndices: experts)
        return projected.squeezed(axis: -2)
    }
}

/// The mixture of experts, with one dense expert every token also runs.
final class NFKQwen4ExpMixture: Module {
    @ModuleInfo(key: "gate") var router: Linear
    @ModuleInfo(key: "experts") var experts: NFKQwen4ExpExperts
    @ModuleInfo(key: "shared_expert") var sharedExpert: NFKLMFeedForward
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    let activeExpertCount: Int
    let normalizesWeights: Bool

    init(_ c: NFKMLXQwen4ExpConfiguration) {
        precondition(c.activeExpertCount > 0 && c.activeExpertCount <= c.expertCount,
                     "a mixture routes each token to between one and every expert")
        activeExpertCount = c.activeExpertCount
        normalizesWeights = c.normalizesExpertWeights
        _router.wrappedValue = Linear(c.hiddenSize, c.expertCount, bias: false)
        _experts.wrappedValue = NFKQwen4ExpExperts(c)
        _sharedExpert.wrappedValue = NFKLMFeedForward(hiddenSize: c.hiddenSize,
                                                      intermediateSize: c.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(c.hiddenSize, 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (weights, chosen) = NFKReferenceRounding.routed(router(x), active: activeExpertCount, normalize: normalizesWeights)
        let routed = NFKReferenceRounding.combined(experts(x, experts: chosen), weights: weights, chosen: chosen)
        return routed + NFKReferenceRounding.sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }
}

/// One layer. Which branch it carries is fixed by its depth, and only the layers the configuration
/// names carry a per-layer embedding.
final class NFKQwen4ExpBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKQwen4ExpAttention?
    @ModuleInfo(key: "linear_attn") var linearAttention: NFKQwen4ExpLinearAttention?
    @ModuleInfo(key: "ple") var ple: NFKQwen4ExpPLE?
    @ModuleInfo(key: "mlp") var feedForward: NFKQwen4ExpMixture
    @ModuleInfo(key: "attn_hyper_connection") var attentionResidual: NFKQwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var feedForwardResidual: NFKQwen4ExpGatedResidual

    let kind: NFKMLXQwen4ExpLayerKind

    init(_ c: NFKMLXQwen4ExpConfiguration, kind: NFKMLXQwen4ExpLayerKind, pleIndex: Int?) {
        self.kind = kind
        _attention.wrappedValue = kind == .sparseAttention ? NFKQwen4ExpAttention(c) : nil
        _linearAttention.wrappedValue = kind == .linearAttention ? NFKQwen4ExpLinearAttention(c) : nil
        _ple.wrappedValue = pleIndex.map { NFKQwen4ExpPLE(c, layerIndex: $0) }
        _feedForward.wrappedValue = NFKQwen4ExpMixture(c)
        _attentionResidual.wrappedValue = NFKQwen4ExpGatedResidual(c)
        _feedForwardResidual.wrappedValue = NFKQwen4ExpGatedResidual(c)
        super.init()
    }

    /// Writes a branch's output back into every stream, by the shares the hyper-connection chose.
    private func injected(_ hyper: MLXArray, branch: MLXArray, shares: MLXArray?) -> MLXArray {
        guard let shares else { return hyper }
        let spread = branch.expandedDimensions(axis: -2) * shares.expandedDimensions(axis: -1)
        return hyper + spread.reshaped([hyper.dim(0), hyper.dim(1), -1])
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray,
                        ngramIndices: MLXArray?) -> MLXArray {
        var hyper = x
        if let ple, let ngramIndices {
            hyper = hyper + ple(hyper, indices: ngramIndices)
        }

        let (read, shares) = attentionResidual(hyper)
        let branch: MLXArray
        if let linearAttention {
            branch = linearAttention(read)
        } else {
            branch = attention!(read, cos: cos, sin: sin, mask: mask)
        }
        hyper = injected(hyper, branch: branch, shares: shares)

        let (mlpRead, mlpShares) = feedForwardResidual(hyper)
        return injected(hyper, branch: feedForward(mlpRead), shares: mlpShares)
    }
}

/// The decoder itself, under the `model.language_model.` prefix the release uses. There is no final
/// normalization: the hyper-connection mixer that collapses the streams carries one of its own.
final class NFKQwen4ExpCore: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKQwen4ExpBlock]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: NFKQwen4ExpGatedResidual

    init(_ c: NFKMLXQwen4ExpConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layerCount).map { layer in
            NFKQwen4ExpBlock(c, kind: c.layerTypes[layer],
                             pleIndex: c.pleLayerIDs.firstIndex(of: layer + 1))
        }
        _mixer.wrappedValue = NFKQwen4ExpGatedResidual(c, combines: false)
        super.init()
    }
}

/// The multimodal wrapper's `model`, which a text-only run reaches straight through.
final class NFKQwen4ExpModel: Module {
    @ModuleInfo(key: "language_model") var languageModel: NFKQwen4ExpCore

    init(_ c: NFKMLXQwen4ExpConfiguration) {
        _languageModel.wrappedValue = NFKQwen4ExpCore(c)
        super.init()
    }
}

/// A Qwen4-Exp causal language model.
public final class NFKMLXQwen4ExpNet: Module {
    @ModuleInfo(key: "model") var model: NFKQwen4ExpModel

    /// The routed experts of a paged load, which the mixture layers read in place of parameters; nil
    /// where every expert is resident. Introduced in InferKit 0.4.0.
    public internal(set) var expertStore: NFKMLXExpertStore?
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: NFKMLXQwen4ExpConfiguration

    init(_ c: NFKMLXQwen4ExpConfiguration) {
        configuration = c
        _model.wrappedValue = NFKQwen4ExpModel(c)
        _lmHead.wrappedValue = c.tiesWordEmbeddings
            ? nil : Linear(c.hiddenSize, c.vocabularySize, bias: false)
        super.init()
    }

    /// The interleaved M-RoPE table for `positions` `[3, batch, length]`, one row per grid axis.
    ///
    /// @discussion The turned channels are dealt out to the three axes in a repeating T, H, W cycle
    /// rather than in three contiguous runs, which is what `mrope_interleaved` means. A text-only run
    /// passes the same row three times, where the interleave is an identity.
    func rotaryTable(positions: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let c = configuration
        let turned = c.rotaryDimensions
        let half = turned / 2
        let exponents = (0 ..< half).map { Float($0 * 2) / Float(turned) }
        let inverse = MLXArray(exponents.map { 1 / pow(c.ropeTheta, $0) })
        // [3, batch, length, half]
        let frequencies = positions.asType(.float32).expandedDimensions(axis: -1) * inverse

        var recomposed = frequencies[0]
        for (axis, offset) in [(1, 1), (2, 2)] {
            let limit = c.mropeSection[axis] * 3
            let channels = MLXArray(stride(from: offset, to: min(limit, half), by: 3).map { Int32($0) })
            guard channels.size > 0 else { continue }
            recomposed = putAlong(recomposed.swappedAxes(0, -1),
                                  channels.reshaped([channels.size] + Array(repeating: 1,
                                                                            count: recomposed.ndim - 1)),
                                  values: take(frequencies[axis].swappedAxes(0, -1), channels, axis: 0),
                                  axis: 0).swappedAxes(0, -1)
        }
        let full = concatenated([recomposed, recomposed], axis: -1)
        return (cos(full), sin(full))
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var trace = [MLXArray]()
        return forward(tokens, trace: &trace)
    }

    /// The state entering the stack, then each layer's output, with the mixer folded into the last
    /// entry — the convention the reference's `output_hidden_states` uses.
    func hiddenStates(_ tokens: MLXArray) -> [MLXArray] {
        var trace = [MLXArray]()
        _ = forward(tokens, trace: &trace)
        return trace
    }

    private func forward(_ tokens: MLXArray, trace: inout [MLXArray]) -> MLXArray {
        let c = configuration
        let (batch, length) = (tokens.dim(0), tokens.dim(1))
        let embedded = model.languageModel.embedTokens(tokens)

        let rows = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, 1, length])
        let (cosTable, sinTable) = rotaryTable(positions: broadcast(rows, to: [3, batch, length]))

        // A token may attend to itself and to what came before it; the indexer narrows that further.
        let queryIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([length, 1])
        let keyIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, length])
        let causal = (keyIndex .<= queryIndex).reshaped([1, 1, length, length])

        // Every PLE layer hashes the same token ids; the tables differ, so each holds its own.
        let ids = tokens.asType(.int32).asArray(Int32.self)
        let rowsOfIDs = (0 ..< batch).map { row in
            (0 ..< length).map { Int(ids[row * length + $0]) }
        }

        // The trace records what enters each layer, so its first entry is the expanded streams
        // rather than the embedding — which is the shape the reference's `hidden_states` carries.
        var hidden = concatenated(Array(repeating: embedded, count: c.hyperConnectionCount), axis: -1)
        trace.append(hidden)
        for layer in model.languageModel.layers {
            let indices = layer.ple.map { $0.embedding.indices(tokens: rowsOfIDs) }
            hidden = layer(hidden, cos: cosTable, sin: sinTable, mask: causal, ngramIndices: indices)
            trace.append(hidden)
        }
        let collapsed = model.languageModel.mixer(hidden).read
        if !trace.isEmpty { trace[trace.count - 1] = collapsed }
        return lmHead?(collapsed) ?? model.languageModel.embedTokens.asLinear(collapsed)
    }
}

/// Building the Qwen4-Exp decoder and reading a release's configuration.
@objc(NFKMLXQwen4Exp)
public final class NFKMLXQwen4Exp: NSObject {

    public static func makeNet(_ configuration: NFKMLXQwen4ExpConfiguration = .qwen3_8FlashNext)
        -> NFKMLXQwen4ExpNet {
        NFKMLXQwen4ExpNet(configuration)
    }

    /// Reads a `Qwen4ExpForConditionalGeneration` config, whose decoder lives under `text_config`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXQwen4ExpConfiguration {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("the configuration is not a JSON object")
        }
        let text = (root["text_config"] as? [String: Any]) ?? root
        if let architectures = root["architectures"] as? [String],
           !architectures.contains(where: { $0.hasPrefix("Qwen4Exp") }) {
            throw NFKMLXError.unsupportedConfiguration(
                "\(architectures.first ?? "the release") is not a Qwen4-Exp model")
        }

        func integer(_ key: String, _ fallback: Int) -> Int { (text[key] as? Int) ?? fallback }
        func number(_ key: String, _ fallback: Float) -> Float {
            (text[key] as? NSNumber).map { $0.floatValue } ?? fallback
        }
        let rope = (text["rope_parameters"] as? [String: Any]) ?? [:]
        let ends = text["eos_token_id"]
        let endToken = (ends as? Int) ?? (ends as? [Int])?.first ?? 248_044

        // The release states `full_attention` for layers that carry an indexer and never attend to
        // the whole prefix. The reference rewrites them, and so does this.
        let kinds = (text["layer_types"] as? [String])?.map {
            $0 == "linear_attention" ? NFKMLXQwen4ExpLayerKind.linearAttention : .sparseAttention
        }

        return NFKMLXQwen4ExpConfiguration(
            hiddenSize: integer("hidden_size", 2560),
            layerCount: integer("num_hidden_layers", 48),
            vocabularySize: integer("vocab_size", 248_320),
            rmsEpsilon: number("rms_norm_eps", 1e-6),
            headCount: integer("num_attention_heads", 24),
            keyValueHeadCount: integer("num_key_value_heads", 2),
            headDimensions: integer("head_dim", 256),
            ropeTheta: (rope["rope_theta"] as? NSNumber).map { $0.floatValue } ?? 10_000_000,
            partialRotaryFactor: (rope["partial_rotary_factor"] as? NSNumber).map { $0.floatValue }
                ?? number("partial_rotary_factor", 0.25),
            mropeSection: (rope["mrope_section"] as? [Int]) ?? [11, 11, 10],
            indexerHeadCount: integer("indexer_n_heads", 4),
            indexerKeyValueHeadCount: integer("indexer_kv_heads", 1),
            indexerHeadDimensions: integer("indexer_head_dim", 128),
            indexerBudget: integer("indexer_budget", 2048),
            indexerCompressRatio: integer("indexer_compress_ratio", 4),
            linearKeyHeadCount: integer("linear_num_key_heads", 16),
            linearKeyHeadDimensions: integer("linear_key_head_dim", 128),
            linearValueHeadCount: integer("linear_num_value_heads", 48),
            linearValueHeadDimensions: integer("linear_value_head_dim", 128),
            linearConvolutionKernel: integer("linear_conv_kernel_dim", 4),
            outputGateActivation: (text["output_gate_type"] as? String) == "silu" ? .silu : .sigmoid,
            hyperConnectionCount: integer("hc_count", 4),
            hyperConnectionRank: integer("hc_lowrank", 320),
            expertCount: integer("num_experts", 512),
            activeExpertCount: integer("num_experts_per_tok", 10),
            expertIntermediateSize: integer("moe_intermediate_size", 640),
            sharedExpertIntermediateSize: integer("shared_expert_intermediate_size", 640),
            normalizesExpertWeights: (text["norm_topk_prob"] as? Bool) ?? true,
            pleLayerIDs: (text["ple_layer_ids"] as? [Int]) ?? [],
            pleEmbedDimensions: text["ple_embed_dim"] as? Int,
            pleConvolutionKernel: integer("ple_conv_kernel_size", 4),
            ngramSize: integer("ngram_size", 3),
            headsPerNgram: integer("heads_per_ngram", 8),
            ngramVocabularyBase: integer("ngram_vocab_size_base", 20_000_000),
            ngramVocabularyDivisor: integer("make_ngram_vocab_size_divisible_by", 128),
            hashSeed: integer("seed", 1234),
            endOfSequenceToken: endToken,
            fullAttentionInterval: integer("full_attention_interval", 4),
            tiesWordEmbeddings: (text["tie_word_embeddings"] as? Bool) ?? false,
            layerTypes: kinds)
    }

    /// The names the loader drops on purpose: the n-gram hash tables, which this module derives from
    /// the configuration rather than reading, and the multi-token-prediction head and vision tower,
    /// which are separate features.
    public static func isDropped(key: String) -> Bool {
        key.hasPrefix("mtp.") || key.hasPrefix("model.visual.")
            || key.hasSuffix(".ple_embedding.layer_multipliers")
            || key.hasSuffix(".ple_embedding.ngram_heads_vocab_sizes")
            || key.hasSuffix(".ple_embedding.ngram_heads_offsets")
    }

    /// Converts one released tensor to this module's name and layout.
    ///
    /// @discussion Two rewrites. A depthwise convolution is stored `[channels, 1, kernel]` by PyTorch
    /// and `[channels, kernel, 1]` by MLX. The n-gram table ships in `split_ngram_parts` shards, so a
    /// release can be written back in the layout it was trained under; the shards concatenate, in
    /// index order, into the one table this module looks up.
    public static func adapted(key: String, value: MLXArray) -> (String, MLXArray) {
        if key.hasSuffix("conv1d.weight"), value.ndim == 3 {
            return (key, value.transposed(0, 2, 1))
        }
        return (key, value)
    }

    /// The shard index a `…ngram_embedding.shard_<n>.weight` key names, and the key it belongs to.
    static func ngramShard(key: String) -> (table: String, index: Int)? {
        guard let range = key.range(of: ".ngram_embedding.shard_") else { return nil }
        let tail = key[range.upperBound...]
        guard let dot = tail.firstIndex(of: "."), let index = Int(tail[..<dot]) else { return nil }
        return (String(key[..<range.lowerBound]) + ".ngram_embedding.weight", index)
    }
}

extension NFKMLXQwen4Exp {

    /// Loads a released Qwen4-Exp decoder from its directory, following the shard index.
    ///
    /// @discussion The release nests the decoder under `model.language_model.` beside a vision tower
    /// and a multi-token-prediction head, and splits each n-gram table across `split_ngram_parts`
    /// tensors so that the saved layout matches the one it was trained under. Only the decoder's
    /// tensors are taken, and each table's parts are concatenated in index order.
    public static func loadWeights(into net: NFKMLXQwen4ExpNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        try loadWeights(into: net, fromDirectory: directory, precision: precision, skipping: { _ in false })
    }

    /// Loads a released decoder with its routed experts held as `residency` plans them. The release
    /// stores each projection stacked in the module's own layout, so an expert pages as the release
    /// stores it. Introduced in InferKit 0.4.0.
    public static func loadWeights(into net: NFKMLXQwen4ExpNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision, residency: NFKMLXResidency) throws {
        net.expertStore = try NFKMLXExpertInventory.load(
            directory: directory, precision: precision, residency: residency,
            classify: NFKMLXExpertInventory.stacked(projections: ["gate_up_proj", "down_proj"]) {
                isDropped(key: $0) ? nil : $0
            },
            install: { store in
                try net.model.languageModel.layers.enumerated().flatMap { index, block in
                    let mixture = block.feedForward
                    let count = mixture.experts.gateUp?.dim(0) ?? 0
                    let path = "model.language_model.layers.\(index).mlp.experts"
                    let paged = NFKQwen4ExpExperts(pagedFrom: store, path: path,
                                                   intermediateSize: mixture.experts.intermediateSize)
                    try mixture.update(modules: ModuleChildren.unflattened([("experts", paged)]), verify: .noUnusedKeys)
                    return [(path + ".gate_up_proj", count, ["weight"]), (path + ".down_proj", count, ["weight"])]
                }
            },
            load: { try loadWeights(into: net, fromDirectory: directory, precision: precision, skipping: $0) })
    }

    static func loadWeights(into net: NFKMLXQwen4ExpNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision, skipping skipped: (String) -> Bool) throws {
        let tied = net.lmHead == nil
        var shards = [String: [(index: Int, value: MLXArray)]]()
        var direct = [(String, MLXArray)]()

        let read = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) {
            key -> String? in
            guard !isDropped(key: key), !skipped(key) else { return nil }
            return tied && key.hasPrefix("lm_head.") ? nil : key
        }
        for (key, value) in read {
            if let shard = ngramShard(key: key) {
                shards[shard.table, default: []].append((shard.index, value))
            } else {
                direct.append(adapted(key: key, value: value))
            }
        }
        for (table, parts) in shards {
            direct.append((table, concatenated(parts.sorted { $0.index < $1.index }.map(\.value),
                                               axis: 0)))
        }
        try NFKMLXWeights.apply(direct, to: net)
    }

    /// A decoder built from a release directory, reading its own `config.json`, with its routed
    /// experts held as `residency` plans them.
    public static func backend(directoryURL: URL,
                               precision: NFKMLXWeightPrecision = .float32,
                               residency: NFKMLXResidency = .automatic) throws
        -> NFKMLXQwen4ExpNet {
        let geometry = try configuration(fromHuggingFace: directoryURL
            .appendingPathComponent("config.json"))
        let net = makeNet(geometry)
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision, residency: residency)
        return net
    }
}
