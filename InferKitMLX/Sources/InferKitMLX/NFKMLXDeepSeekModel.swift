//
//  NFKMLXDeepSeekModel.swift
//  InferKitMLX
//
//  The DeepSeek V4 decoder (`DeepseekV4ForCausalLM`): Multi-head Latent Attention over a
//  mixture-of-experts feed-forward. A third architecture family beside the dense stack
//  (`NFKMLXLanguageNet`) and the gated-recurrence hybrid (`NFKMLXHybridLanguageNet`).
//
//  BUILT, NOT MEASURED — and verified more weakly than the hybrid, for a reason worth stating.
//  The released checkpoint is QUANTIZED: attention weights are fp8 with 128×128 block scales, and the
//  experts are 4-bit packed two per int8 byte with their own scales. A float module's parameters
//  therefore do NOT correspond one-to-one with the checkpoint's tensors, so the structural check has to
//  derive what each float parameter would look like quantized. That derivation is itself an assumption,
//  which is why the test asserts it reproduces the observed shapes rather than trusting it.
//
//  Implemented: the attention (low-rank queries, a shared latent key-value, grouped low-rank output,
//  the learned attention sink, rotary applied to the trailing channels only) and the mixture of experts
//  (square-root-softplus scoring, hash routing on the first layers, bias-shifted top-k after them, the
//  clamped SwiGLU, the shared expert).
//
//  NOT implemented, and named so they are known rather than overlooked: the DSpark speculative blocks
//  and the multi-token prediction layers. YaRN rope scaling IS implemented now — V4 Pro extends its
//  window with it, and because it carries no parameters the structural check cannot see whether it is
//  there, so a Pro run without it would have been silently wrong past the trained length. Attention here is dense over the sliding window, which is what the uncompressed
//  layers do; a compressed layer needs the indexer to be correct.
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// The geometry of a DeepSeek V4 decoder.
public struct NFKMLXDeepSeekConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var vocabularySize: Int
    public var rmsEpsilon: Float

    // Multi-head latent attention.
    public var headCount: Int
    public var headDimensions: Int
    /// The channels of each head the rotary embedding turns; they sit at the END of the head.
    public var ropeHeadDimensions: Int
    public var queryLoRARank: Int
    public var outputLoRARank: Int
    /// The output projection is applied per group of heads.
    public var outputGroups: Int
    public var slidingWindow: Int
    public var ropeTheta: Float
    /// The rotary base the COMPRESSED positions use, which the config names separately.
    public var compressRopeTheta: Float
    /// How many parallel copies the hyper-connected residual stream carries.
    public var hyperConnectionCopies: Int
    /// Sinkhorn iterations used to make the copy-mixing matrix near doubly stochastic.
    public var sinkhornIterations: Int
    /// The floor added inside that normalization.
    public var hyperConnectionEpsilon: Float

    /// The sparse indexer's own head count, width, and how many compressed positions it keeps.
    public var indexHeadCount: Int
    public var indexHeadDimensions: Int
    public var indexTopK: Int

    // Mixture of experts.
    public var routedExpertCount: Int
    public var sharedExpertCount: Int
    public var activatedExpertCount: Int
    public var expertIntermediateSize: Int
    public var routeScale: Float
    /// Both branches of the SwiGLU are clamped to this before multiplying.
    public var swigluLimit: Float
    /// The first layers route by a precomputed token-to-expert table rather than by score.
    public var hashLayerCount: Int
    /// Whether the router carries a second bias for the tokens of an image span. A release with a
    /// vision tower has one; a text-only configuration of the same architecture does not.
    public var routerHasVisionBias: Bool

    /// Per-layer key-value compression ratio; 0 means a layer attends over the window alone.
    public var compressRatios: [Int]
    /// Whether the compressor learns a per-slot position bias. V4 does; V4.1 pools without one.
    public var compressorHasPositionBias: Bool
    /// The layers that own a compressor. Every other compressed layer reads the compressed
    /// key-value from one of these, so the whole stack pays for four rather than for thirty-eight.
    /// Nil means each compressed layer owns its own, which is V4's arrangement.
    public var keyValueSourceLayers: [Int]?
    /// The layers that carry an indexer. Nil means every layer at the light ratio, which is V4's.
    public var indexSourceLayers: [Int]?
    /// The layer whose compressed keys pick candidate blocks before the indexer ranks within them.
    public var candidateSourceLayer: Int?
    public var candidateTopKBlocks: Int
    public var candidateBlockSize: Int
    /// Whether the copies collapse through a learned head. V4 learns one; V4.1 collapses by identity.
    public var collapsesThroughLearnedHead: Bool
    /// Whether a sub-block reads the copies with the read weight the PREVIOUS sub-block predicted.
    ///
    /// V4 lets each hyper-connection collapse with its own; V4.1 pipelines, so attention reads with
    /// what the previous layer's feed-forward predicted and the feed-forward reads with what this
    /// layer's attention did. The parameters are identical either way, which is why a structural
    /// check cannot see this and the oracle can.
    public var pipelinesHyperConnectionRead: Bool
    /// Whether an indexer derives its keys from the layer compressor's latent rather than running a
    /// compressor of its own over the hidden state. V4.1 shares; V4 compresses twice.
    public var indexerDerivesKeysFromCompressor: Bool
    /// Whether each query head is RMS-normalized after the up-projection, on top of the norm the
    /// low-rank query already carries. V4 does; V4.1 rotates the projected heads directly.
    public var normalizesQueryHeads: Bool
    /// Whether a compressed layer builds its whole rotary at the compressed base.
    ///
    /// The reference gives a layer ONE frequency table, chosen by whether that layer compresses: a
    /// compressed layer builds it at `compressRopeTheta` with the extrapolation on, an uncompressed
    /// one at `ropeTheta` with it off. The queries, the window key-value, the compressed positions
    /// and the indexer all read that one table, so the base belongs to the layer rather than to the
    /// compressed positions alone.
    public var compressedLayersRotateAtCompressedBase: Bool
    /// Whether every RMSNorm holds its weight float32 in a bf16 decoder. V4's code builds its norm
    /// with a float32 weight; V4.1's builds it in the default dtype.
    public var holdsNormWeightsInFloat32: Bool
    /// Whether the draft stack reads the target layers' OUTPUTS. V4 Pro 0813 averages the copies
    /// each target layer produced; V4.1 averages the stream entering each one's attention.
    public var draftReadsTargetLayerOutputs: Bool
    /// The release's names for the Markov head's two tables, which the module holds as `embed` and
    /// `head`: V4.1 names them that way, V4 Pro 0813 `markov_w1` and `markov_w2`.
    public var draftMarkovTableNames: [String]

    // The n-gram memory, on the layers `engramLayerIDs` names.
    /// ONE table per engram layer, each with its own row count.
    public var engramLayerIDs: [Int]
    public var engramEmbeddingCounts: [Int]
    /// The largest n-gram represented. A position hashes the 2-gram through this size.
    public var engramMaxNgramSize: Int
    /// How many independently hashed heads each n-gram size carries.
    public var engramHeadCount: Int
    public var engramHeadDimensions: Int
    /// The floor each head's prime bucket count is drawn above.
    public var engramVocabularySize: Int
    /// The size of the id space token ids collapse into before hashing, which every hash multiplier
    /// is derived from. The release states it, and the derivation is checked against it.
    public var engramCompressedVocabularySize: Int
    public var engramPadToken: Int

    /// How many multi-token-prediction blocks the release carries beside the decoder.
    public var nextTokenPredictionLayers: Int

    /// The image tower's geometry, where the release carries one.
    public var vision: NFKMLXDeepSeekVisionConfiguration?

    // DSpark, the speculative-decoding draft stack stored under `mtp.`.
    /// How many tokens a draft block proposes at once. Zero means the release carries no draft head.
    public var dsparkBlockSize: Int
    /// The decoder layers whose attention INPUT the draft stack reads, concatenated.
    public var dsparkTargetLayers: [Int]
    /// The width of the Markov head's own embedding, which biases the draft logits.
    public var dsparkMarkovRank: Int
    /// A draft block may route over its own, smaller set of experts. Zero means it routes over the
    /// decoder's, which is what the reference's `get_moe_config` falls back to and what V4 does;
    /// V4.1 names 128 of its own.
    public var dsparkExpertCount: Int
    public var dsparkActiveExpertCount: Int
    /// The id a draft position holds before it is proposed.
    public var dsparkNoiseToken: Int

    /// The square block one fp8 scale covers, from `quantization_config.weight_block_size`. V4
    /// states 128 and V4.1 states 32, so this is read rather than assumed: decoding a V4.1 weight
    /// at 128 reads a quarter of the scale grid and is wrong with no error.
    public var fp8BlockSize: Int

    /// Whether the decoder rounds its activations the way the release's own `inference/model.py`
    /// does. Off by default.
    ///
    /// @discussion That code quantizes and immediately dequantizes at six places, each in place
    /// so everything downstream reads the rounded value: the sliding-window key-value to fp8, the
    /// compressed latent to fp4 with E4M3 scales, the indexer's keys and queries to fp4, and the
    /// draft stack's two key-values to fp8. It also casts the n-gram rows to bf16, and rounds the
    /// input of every GEMM whose weight the release stores fp8 or fp4 to fp8 in blocks of
    /// `fp8BlockSize`, which is what its `linear` does before the fp8 GEMM. Off, the port is the
    /// model those round trips approximate, which is what it is measured against by default; on,
    /// it is measured against the release's code with its quantizers running. With
    /// `computesInBFloat16` as well, every layer's stream is that code's, bit for bit.
    ///
    /// Not read from `config.json`, because the release does not state it: it is the release's
    /// code, and a choice about which of two targets to reproduce.
    public var quantizesActivations: Bool

    /// Whether the decoder computes in bf16, as the release does.
    ///
    /// @discussion A configuration read from a release, and every preset, compute in bf16, the
    /// dtype the release declares and its own inference code runs in; a hand-built configuration
    /// defaults to float32. Parameters are held bf16 except the ones the reference's
    /// own constructor makes float32 (the hyper-connection coefficients, the attention sink, the
    /// router biases, the heads, the draft stack's confidence and Markov heads, a pooling
    /// compressor, and the image tower's norms). Activations are bf16 between the places the reference computes in float32:
    /// the norms, the rotary, the attention's scores and softmax, the hyper-connection mixing, a
    /// pooling compressor, the router, the experts' activation and the mixture's sum, the n-gram
    /// gate, and the head. Each of those upcasts, computes, and casts back to the dtype it was
    /// handed, which is the reference's `.to(x.dtype)`; in float32 every such cast is an identity,
    /// which is what leaves the float32 decoder exactly as it was.
    ///
    /// It halves the bytes every step reads for the parameters that are not already stored
    /// narrow, and decoding is bound by those bytes. Measured against the release's code run in
    /// bf16, every stream is identical to it, bit for bit: every decoder layer at prefill, every
    /// buffer a decode step carries, the draft stack, the image tower and the image router. The
    /// tower's attention is held to torch's own definition of `scaled_dot_product_attention` (its
    /// MATH backend), because the release delegates that step to whichever backend runs it.
    public var computesInBFloat16: Bool

    /// The dtype activations are held in between the places the reference computes in float32.
    public var computeType: DType { computesInBFloat16 ? .bfloat16 : .float32 }

    /// The rotary scaling the release declares. V4 Pro extends its window with YaRN; Flash does not.
    public var ropeScaling: NFKMLXRoPEScaling?

    public init(hiddenSize: Int = 4096, layerCount: Int = 43, vocabularySize: Int = 129_280,
                rmsEpsilon: Float = 1e-6, headCount: Int = 64, headDimensions: Int = 512,
                ropeHeadDimensions: Int = 64, queryLoRARank: Int = 1024, outputLoRARank: Int = 1024,
                outputGroups: Int = 8, slidingWindow: Int = 128, ropeTheta: Float = 10_000,
                compressRopeTheta: Float = 160_000, hyperConnectionCopies: Int = 4,
                sinkhornIterations: Int = 20, hyperConnectionEpsilon: Float = 1e-6,
                indexHeadCount: Int = 64,
                indexHeadDimensions: Int = 128, indexTopK: Int = 512,
                routedExpertCount: Int = 256, sharedExpertCount: Int = 1,
                activatedExpertCount: Int = 6, expertIntermediateSize: Int = 2048,
                routeScale: Float = 1.5, swigluLimit: Float = 10, hashLayerCount: Int = 3,
                routerHasVisionBias: Bool = false,
                compressRatios: [Int]? = nil, compressorHasPositionBias: Bool = true,
                keyValueSourceLayers: [Int]? = nil, indexSourceLayers: [Int]? = nil,
                candidateSourceLayer: Int? = nil, candidateTopKBlocks: Int = 0,
                candidateBlockSize: Int = 0, collapsesThroughLearnedHead: Bool = true,
                indexerDerivesKeysFromCompressor: Bool = false,
                normalizesQueryHeads: Bool = true,
                compressedLayersRotateAtCompressedBase: Bool = true,
                pipelinesHyperConnectionRead: Bool = false,
                engramLayerIDs: [Int] = [], engramEmbeddingCounts: [Int] = [],
                engramMaxNgramSize: Int = 4, engramHeadCount: Int = 8,
                engramHeadDimensions: Int = 256, engramVocabularySize: Int = 16_000_000,
                engramCompressedVocabularySize: Int = 0, engramPadToken: Int = 2,
                nextTokenPredictionLayers: Int = 1,
                vision: NFKMLXDeepSeekVisionConfiguration? = nil,
                dsparkBlockSize: Int = 0, dsparkTargetLayers: [Int] = [],
                dsparkMarkovRank: Int = 256, dsparkExpertCount: Int = 0,
                dsparkActiveExpertCount: Int = 0, dsparkNoiseToken: Int = 0,
                fp8BlockSize: Int = NFKMLXDeepSeekQuantization.fp8BlockSize,
                quantizesActivations: Bool = false, computesInBFloat16: Bool = false,
                holdsNormWeightsInFloat32: Bool = false, draftReadsTargetLayerOutputs: Bool = false,
                draftMarkovTableNames: [String] = ["embed", "head"]) {
        self.quantizesActivations = quantizesActivations
        self.holdsNormWeightsInFloat32 = holdsNormWeightsInFloat32
        self.draftReadsTargetLayerOutputs = draftReadsTargetLayerOutputs
        self.draftMarkovTableNames = draftMarkovTableNames
        self.computesInBFloat16 = computesInBFloat16
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.vocabularySize = vocabularySize
        self.rmsEpsilon = rmsEpsilon
        self.headCount = headCount
        self.headDimensions = headDimensions
        self.ropeHeadDimensions = ropeHeadDimensions
        self.queryLoRARank = queryLoRARank
        self.outputLoRARank = outputLoRARank
        self.outputGroups = outputGroups
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.compressRopeTheta = compressRopeTheta
        self.hyperConnectionCopies = hyperConnectionCopies
        self.sinkhornIterations = sinkhornIterations
        self.hyperConnectionEpsilon = hyperConnectionEpsilon
        self.indexHeadCount = indexHeadCount
        self.indexHeadDimensions = indexHeadDimensions
        self.indexTopK = indexTopK
        self.routedExpertCount = routedExpertCount
        self.sharedExpertCount = sharedExpertCount
        self.activatedExpertCount = activatedExpertCount
        self.expertIntermediateSize = expertIntermediateSize
        self.routeScale = routeScale
        self.swigluLimit = swigluLimit
        self.hashLayerCount = hashLayerCount
        self.routerHasVisionBias = routerHasVisionBias
        self.compressorHasPositionBias = compressorHasPositionBias
        self.keyValueSourceLayers = keyValueSourceLayers
        self.indexSourceLayers = indexSourceLayers
        self.candidateSourceLayer = candidateSourceLayer
        self.candidateTopKBlocks = candidateTopKBlocks
        self.candidateBlockSize = candidateBlockSize
        self.collapsesThroughLearnedHead = collapsesThroughLearnedHead
        self.indexerDerivesKeysFromCompressor = indexerDerivesKeysFromCompressor
        self.normalizesQueryHeads = normalizesQueryHeads
        self.compressedLayersRotateAtCompressedBase = compressedLayersRotateAtCompressedBase
        self.pipelinesHyperConnectionRead = pipelinesHyperConnectionRead
        self.engramLayerIDs = engramLayerIDs
        self.engramEmbeddingCounts = engramEmbeddingCounts
        self.engramMaxNgramSize = engramMaxNgramSize
        self.engramHeadCount = engramHeadCount
        self.engramHeadDimensions = engramHeadDimensions
        self.engramVocabularySize = engramVocabularySize
        self.engramCompressedVocabularySize = engramCompressedVocabularySize
        self.engramPadToken = engramPadToken
        self.nextTokenPredictionLayers = nextTokenPredictionLayers
        self.vision = vision
        self.dsparkBlockSize = dsparkBlockSize
        self.dsparkTargetLayers = dsparkTargetLayers
        self.dsparkMarkovRank = dsparkMarkovRank
        self.dsparkExpertCount = dsparkExpertCount
        self.dsparkActiveExpertCount = dsparkActiveExpertCount
        self.dsparkNoiseToken = dsparkNoiseToken
        self.fp8BlockSize = fp8BlockSize
        // The released pattern alternates a light and a heavy ratio after two uncompressed layers.
        self.compressRatios = compressRatios ?? (0 ..< layerCount).map { index in
            index < 2 ? 0 : (index % 2 == 0 ? 4 : 128)
        }
    }

    /// The released `deepseek-ai/DeepSeek-V4-Flash-0731` decoder.
    public static let v4Flash: NFKMLXDeepSeekConfiguration = {
        var flash = NFKMLXDeepSeekConfiguration(
            compressRatios: v4CompressRatios(first: 0, layers: 43), nextTokenPredictionLayers: 3,
            dsparkBlockSize: 5, dsparkTargetLayers: [40, 41, 42], dsparkNoiseToken: 128_799,
            computesInBFloat16: true, holdsNormWeightsInFloat32: true,
            draftReadsTargetLayerOutputs: true, draftMarkovTableNames: ["markov_w1", "markov_w2"])
        flash.ropeScaling = releasedYaRN
        return flash
    }()

    /// The released `deepseek-ai/DeepSeek-V4-Pro-0813` decoder.
    public static let v4Pro: NFKMLXDeepSeekConfiguration = {
        var pro = NFKMLXDeepSeekConfiguration(
            hiddenSize: 7168, layerCount: 61, headCount: 128, queryLoRARank: 1536,
            outputGroups: 16, indexTopK: 1024, routedExpertCount: 384,
            expertIntermediateSize: 3072, routeScale: 2.5,
            compressRatios: v4CompressRatios(first: 128, layers: 61), nextTokenPredictionLayers: 3,
            dsparkBlockSize: 5, dsparkTargetLayers: [58, 59, 60], dsparkMarkovRank: 512,
            dsparkNoiseToken: 128_799, computesInBFloat16: true, holdsNormWeightsInFloat32: true,
            draftReadsTargetLayerOutputs: true, draftMarkovTableNames: ["markov_w1", "markov_w2"])
        pro.ropeScaling = releasedYaRN
        return pro
    }()

    /// V4's layout: the first two layers at `first`, then ratio 4 and ratio 128 alternating, and
    /// nothing past the decoder, where the release's list also covers the draft layers.
    private static func v4CompressRatios(first: Int, layers: Int) -> [Int] {
        (0 ..< layers + 3).map { $0 < 2 ? first : ($0 < layers ? ($0 % 2 == 0 ? 4 : 128) : 0) }
    }

    /// The window extension every V4 and V4.1 release declares.
    private static let releasedYaRN = NFKMLXRoPEScaling(
        kind: .yarn, factor: 16, originalMaxPositionEmbeddings: 65_536, betaFast: 32, betaSlow: 1)

    private var withReleasedYaRN: NFKMLXDeepSeekConfiguration {
        var extended = self
        extended.ropeScaling = Self.releasedYaRN
        return extended
    }

    /// This configuration with activation rounding on.
    public var withActivationRounding: NFKMLXDeepSeekConfiguration {
        var rounded = self
        rounded.quantizesActivations = true
        return rounded
    }

    /// The released `deepseek-ai/DeepSeek-V4.1-Flash` decoder (`DeepseekV41ForCausalLM`).
    ///
    /// The same architecture as V4 with four changes the shapes make visible: the compression ratios
    /// drop to 1 and 2 and only four layers own a compressor, the hash-routed layers are gone, the
    /// copies collapse without a learned head, and two layers carry an n-gram memory whose tables
    /// hold 384 million rows each.
    public static let v41Flash = NFKMLXDeepSeekConfiguration(
        hiddenSize: 5120, layerCount: 40, rmsEpsilon: 1e-20, queryLoRARank: 1280,
        indexHeadCount: 32, routedExpertCount: 384, expertIntermediateSize: 2304,
        hashLayerCount: 0, routerHasVisionBias: true,
        compressRatios: (0 ..< 43).map { $0 < 2 || $0 >= 40 ? 0 : ($0 < 20 ? 2 : 1) },
        compressorHasPositionBias: false,
        keyValueSourceLayers: [2, 8, 14, 20],
        indexSourceLayers: [2, 8, 14, 20, 24, 28, 32, 36],
        candidateSourceLayer: 20, candidateTopKBlocks: 2048, candidateBlockSize: 8,
        collapsesThroughLearnedHead: false, indexerDerivesKeysFromCompressor: true,
        normalizesQueryHeads: false, compressedLayersRotateAtCompressedBase: true,
        pipelinesHyperConnectionRead: true,
        engramLayerIDs: [1, 14], engramEmbeddingCounts: [384_006_168, 384_016_682],
        engramCompressedVocabularySize: 99_092, nextTokenPredictionLayers: 3,
        vision: .v41Flash,
        dsparkBlockSize: 5, dsparkTargetLayers: [37, 38, 39], dsparkMarkovRank: 256,
        dsparkExpertCount: 128, dsparkActiveExpertCount: 3, dsparkNoiseToken: 128_799,
        fp8BlockSize: 32, computesInBFloat16: true)
        .withReleasedYaRN

    /// Whether a layer routes by the precomputed table rather than by score.
    func routesByHash(_ layer: Int) -> Bool { layer < hashLayerCount }

    /// Whether a layer owns a compressor, rather than reading one layer's compressed key-value.
    func ownsCompressor(_ layer: Int) -> Bool {
        if let keyValueSourceLayers { return keyValueSourceLayers.contains(layer) }
        return compressRatio(of: layer) > 0
    }

    /// The shortest prefill chunk that can complete a compressed group.
    ///
    /// @discussion A chunk shorter than a compressor's ratio finishes no group, so it emits no
    /// compressed position at all and its queries attend over a compressed cache that a single pass
    /// would already have filled. Chunking is exact at and above this and is not below it: measured
    /// on the oracle configuration, whose largest ratio is 2, a chunk of one diverges from a single
    /// pass by 0.31 at the first layer that owns a compressor, while every size from two up agrees
    /// to 1e-6.
    var minimumPrefillChunk: Int {
        Swift.max(1, (0 ..< layerCount).filter(ownsCompressor).map(compressRatio(of:)).max() ?? 1)
    }

    /// Whether a layer carries an indexer of its own.
    func ownsIndexer(_ layer: Int) -> Bool {
        if let indexSourceLayers { return indexSourceLayers.contains(layer) }
        return compressRatio(of: layer) == 4
    }

    /// Whether a layer picks the candidate blocks every indexer after it scores inside.
    func selectsCandidates(_ layer: Int) -> Bool {
        candidateSourceLayer.map { $0 >= 0 && $0 == layer } ?? false
    }

    /// Whether a layer scores only inside the candidate blocks a layer before it picked.
    func readsCandidates(_ layer: Int) -> Bool {
        candidateSourceLayer.map { $0 >= 0 && $0 < layer } ?? false
    }

    /// Whether the compressed key-value is produced by a few named layers and read by every other
    /// compressed layer.
    ///
    /// That is the arrangement the compressed attention path here implements. V4 gives each
    /// compressed layer its own compressor on a schedule driven by when a window closes, and ships
    /// as the dense approximation its entry records.
    var sharesOneCompressedCache: Bool { keyValueSourceLayers != nil }

    /// The compression ratio a layer pools at; 0 where it does not compress.
    func compressRatio(of layer: Int) -> Int {
        layer < compressRatios.count ? compressRatios[layer] : 0
    }

    /// The geometry a draft block is built from: the decoder's, with the draft stack's own expert
    /// counts, no compression and no n-gram memory. A draft stage keeps the router's vision bias,
    /// because the reference builds that from the release carrying a tower at all rather than from
    /// anything about the layer.
    var draftShaped: NFKMLXDeepSeekConfiguration {
        var draft = self
        draft.layerCount = nextTokenPredictionLayers
        draft.routedExpertCount = draftExpertCount
        draft.activatedExpertCount = draftActiveExpertCount
        draft.compressRatios = [Int](repeating: 0, count: nextTokenPredictionLayers)
        draft.keyValueSourceLayers = nil
        draft.indexSourceLayers = []
        draft.candidateSourceLayer = nil
        draft.hashLayerCount = 0
        draft.engramLayerIDs = []
        return draft
    }

    /// The expert counts a draft block routes over, after the fallback to the decoder's.
    var draftExpertCount: Int { dsparkExpertCount > 0 ? dsparkExpertCount : routedExpertCount }
    var draftActiveExpertCount: Int {
        dsparkActiveExpertCount > 0 ? dsparkActiveExpertCount : activatedExpertCount
    }

    /// How many hashed columns each engram layer looks up per position: one set of heads per n-gram
    /// size from two up.
    var engramHashColumns: Int { (engramMaxNgramSize - 1) * engramHeadCount }
}

/// DeepSeek's rotary: ADJACENT channel pairs rotated as complex numbers.
///
/// The reference builds it with `view_as_complex`, so a pair is `(2i, 2i+1)` — the interleaved
/// convention. The Qwen and Gemma decoders here use rotate-half, where a pair is `(i, i + width/2)`.
/// Nothing in a tensor's shape distinguishes them and every value differs, so the convention is
/// written out rather than selected by a flag. `inverse` applies the conjugate, which the attention
/// output needs: the values share their latent with the keys, so the rotation has to be undone.
final class NFKDeepSeekRotary {
    /// The angles are computed per call rather than taken from a precomputed table. A table has a
    /// length, `take` past its end is an unchecked gather, and a decode run that outlives it would
    /// return whatever memory follows it — noise, with no error, at whatever position the table
    /// happened to stop.
    private let frequencies: [Float]
    let width: Int
    /// The most recent tables, keyed by where they start, how many rows, and the stride. A step
    /// rotates its queries, its keys and its output at the same positions, so the table is built
    /// once per step rather than three times.
    private var tables: [[Int]: (cos: MLXArray, sin: MLXArray)] = [:]

    init(width: Int, theta: Float, maximumPositions: Int = 4096,
         scaling: NFKMLXRoPEScaling? = nil) {
        self.width = width
        let pairs = width / 2
        // A release that extended its window supplies the frequencies; otherwise they come from the
        // base alone. The blend is `NFKMLXRoPEScaling`'s. YaRN's attention factor is deliberately
        // not applied: the release's `precompute_freqs_cis` interpolates the frequencies and builds
        // unit-magnitude rotations, with no scalar on the rotated channels.
        frequencies = scaling.map { $0.inverseFrequencies(dimensions: width, base: theta) }
            ?? (0 ..< pairs).map { 1 / powf(theta, Float(2 * $0) / Float(width)) }
    }

    /// `x` is `[batch, heads, length, width]`; only its leading `width` channels are rotated.
    func callAsFunction(_ x: MLXArray, offset: Int = 0, inverse: Bool = false,
                        stride: Int = 1) -> MLXArray {
        let length = x.shape[2]
        let pairs = width / 2
        let table = self.table(offset: offset, length: length, stride: stride)
        let c = table.cos.reshaped([1, 1, length, pairs])
        let s = (inverse ? -table.sin : table.sin).reshaped([1, 1, length, pairs])

        let paired = x.reshaped(x.shape.dropLast() + [pairs, 2])
        let even = paired[.ellipsis, 0]
        let odd = paired[.ellipsis, 1]
        let rotated = stacked([even * c - odd * s, even * s + odd * c], axis: -1)
        // The tables are float32, so the turn is too; the reference writes it back into the tensor
        // it was handed, which is that tensor's dtype.
        return rotated.reshaped(x.shape).asType(x.dtype)
    }

    /// The rotation's cosines and sines at `length` positions from `offset`, `stride` apart.
    ///
    /// @discussion Built as the release's `torch.polar` builds them: each angle is the float32
    /// product of the position and the frequency, and its cosine and sine are correctly rounded
    /// to float32, computed here in double precision. The GPU's float32 `cos` and `sin` are not
    /// correctly rounded, and a table one unit off in its last place moves a bf16 rounding in the
    /// output de-rotation.
    private func table(offset: Int, length: Int, stride: Int) -> (cos: MLXArray, sin: MLXArray) {
        let key = [offset, length, stride]
        if let cached = tables[key] { return cached }
        var cosines = [Float](), sines = [Float]()
        cosines.reserveCapacity(length * frequencies.count)
        sines.reserveCapacity(length * frequencies.count)
        for row in 0 ..< length {
            // A compressed position advances by a whole window, so its angle steps by `stride`.
            let position = Float(offset + row * stride)
            for frequency in frequencies {
                let angle = Double(position * frequency)
                cosines.append(Float(Foundation.cos(angle)))
                sines.append(Float(Foundation.sin(angle)))
            }
        }
        let built = (MLXArray(cosines).reshaped([length, frequencies.count]),
                     MLXArray(sines).reshaped([length, frequencies.count]))
        if tables.count > 8 { tables.removeAll() }
        tables[key] = built
        return built
    }
}

/// V4's indexer rotation: a Hadamard transform over the last axis, scaled to be orthogonal.
///
/// @discussion The release rotates its index queries and keys through `fast_hadamard_transform`
/// before simulating fp4, to spread information across channels. The rotation is shared by both
/// sides of the dot product, so in float32 it cancels and is skipped. In bf16 the release rounds the
/// rotated values, which changes the scores at the last bit, so it is applied as the library's
/// kernel computes it: accumulated in float32 and rounded once.
func nfkDeepSeekHadamardRotated(_ x: MLXArray) -> MLXArray {
    guard x.dtype != .float32 else { return x }
    let width = x.dim(-1)
    precondition(width > 0 && width & (width - 1) == 0, "a Hadamard rotation needs a power-of-two width")
    var matrix: [[Float]] = [[1]]
    while matrix.count < width {
        matrix = matrix.map { row in row + row } + matrix.map { row in row + row.map { -$0 } }
    }
    let hadamard = MLXArray(matrix.flatMap { $0 }).reshaped([width, width])
    return (matmul(x.asType(.float32), hadamard) * (1 / Float(width).squareRoot())).asType(x.dtype)
}

/// The factor V4 normalizes each query head by, rounded where the release rounds it.
///
/// @discussion The release writes `q *= rsqrt(q.square().mean(-1) + eps)` on a bf16 tensor, so each
/// operation rounds to bf16: the square, the mean (which torch sums in float32 and rounds once), the
/// sum with `eps` (applied at float precision), and the reciprocal root, which torch computes on
/// bf16 as a square root rounded to bf16 and then a reciprocal rounded again. MLX's `mean` rounds
/// twice and would cast `eps` to bf16 first, so each step is written as torch computes it. Float32
/// keeps the single `rsqrt` its parity was measured with.
func nfkDeepSeekHeadNormScale(_ queries: MLXArray, epsilon: Float) -> MLXArray {
    let dtype = queries.dtype
    guard dtype != .float32 else {
        return rsqrt((queries * queries).mean(axis: -1, keepDims: true) + epsilon)
    }
    let meanSquare = (queries * queries).asType(.float32).mean(axis: -1, keepDims: true).asType(dtype)
    let shifted = (meanSquare.asType(.float32) + epsilon).asType(dtype)
    let root = sqrt(shifted.asType(.float32)).asType(dtype)
    return (1 / root.asType(.float32)).asType(dtype)
}

/// RMSNorm as the release computes it: in float32, rounded once to the input's dtype.
///
/// @discussion MLX's fused kernel normalizes in float32, rounds to the input's dtype, and THEN
/// multiplies by the weight in that dtype, which rounds a second time. The reference promotes the
/// weight and multiplies in float32, `(weight * x.float()).to(dtype)`, which rounds once. In
/// float32 the two are the same arithmetic, which is why no float32 measurement could see it; in
/// bf16 they disagree on most elements by one step. Running the fused kernel on float32 operands
/// and casting back is the reference's single rounding, and in float32 both casts are identities.
final class NFKDeepSeekRMSNorm: RMSNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps)
            .asType(x.dtype)
    }
}

/// One expert: a SwiGLU whose branches are clamped before they multiply.
final class NFKDeepSeekExpert: Module {
    @ModuleInfo(key: "w1") var gate: Linear
    @ModuleInfo(key: "w2") var down: Linear
    @ModuleInfo(key: "w3") var up: Linear
    let limit: Float
    /// The fp8 block each matrix's input is rounded in, where the configuration serves its GEMMs;
    /// nil where it does not.
    let servedBlock: Int?

    init(hiddenSize: Int, intermediateSize: Int, limit: Float, servedBlock: Int? = nil) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self.limit = limit
        self.servedBlock = servedBlock
        super.init()
    }

    /// The same expert built from weights already decoded, rather than from zeros a load overwrites.
    /// A paged mixture decodes an expert as it is routed to, and allocating the zeros first would
    /// cost the decoded weight's full size a second time.
    init(gate: MLXArray, down: MLXArray, up: MLXArray, limit: Float, servedBlock: Int? = nil) {
        _gate.wrappedValue = Linear(weight: gate)
        _down.wrappedValue = Linear(weight: down)
        _up.wrappedValue = Linear(weight: up)
        self.limit = limit
        self.servedBlock = servedBlock
        super.init()
    }

    /// - Parameter weight: the routing weight, applied to the intermediate BEFORE `w2` as the
    ///   reference applies it. Linear arithmetic would not care which side it went on; a rounded
    ///   `w2` input does, because the block's scale is the next power of two of its maximum and a
    ///   weight that is not a power of two moves which values round where.
    func callAsFunction(_ x: MLXArray, weight: Float? = nil) -> MLXArray {
        let input = served(x)
        var gated = gate(input).asType(.float32)
        var lifted = up(input).asType(.float32)
        if limit > 0 {
            // The reference clamps the gate from above only and the up branch from both sides.
            gated = clip(gated, max: limit)
            lifted = clip(lifted, min: -limit, max: limit)
        }
        var intermediate = silu(gated) * lifted
        if let weight { intermediate = intermediate * weight }
        return down(served(intermediate.asType(x.dtype)))
    }

    /// A matrix's input as the release's `linear()` hands it to a narrow GEMM: rounded to fp8 in
    /// blocks, where the configuration serves its GEMMs.
    private func served(_ x: MLXArray) -> MLXArray {
        guard let servedBlock else { return x }
        return NFKMLXDeepSeekQuantization.roundTripFP8(x, blockSize: servedBlock)
    }
}

/// Expert routing.
///
/// The first layers take their experts from a table indexed by the TOKEN ID, so routing there does not
/// depend on the hidden state at all. The remaining layers score with a square-root softplus, shift by
/// a learned bias for selection only, and renormalize the unshifted scores of whatever was selected.
final class NFKDeepSeekGate: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?
    /// The second correction bias, for the tokens of an image span. A release with a vision tower
    /// carries one; routing those tokens with the text bias picks different experts for them.
    @ParameterInfo(key: "bias_vl") var visionBias: MLXArray?
    @ParameterInfo(key: "tid2eid") var tokenToExpert: MLXArray?

    let configuration: NFKMLXDeepSeekConfiguration
    let routesByHash: Bool

    init(_ c: NFKMLXDeepSeekConfiguration, layer: Int) {
        configuration = c
        routesByHash = c.routesByHash(layer)
        _weight.wrappedValue = MLXArray.zeros([c.routedExpertCount, c.hiddenSize])
        _bias.wrappedValue = routesByHash ? nil : MLXArray.zeros([c.routedExpertCount])
        _visionBias.wrappedValue = !routesByHash && c.routerHasVisionBias
            ? MLXArray.zeros([c.routedExpertCount]) : nil
        _tokenToExpert.wrappedValue = routesByHash
            ? MLXArray.zeros([c.vocabularySize, c.activatedExpertCount], type: Int32.self) : nil
        super.init()
    }

    /// - Parameter images: `[batch · length]`, true for a token inside an image span.
    func callAsFunction(_ x: MLXArray, tokens: MLXArray?,
                        images: MLXArray? = nil) -> (weights: MLXArray, indices: MLXArray) {
        let scores = sqrt(softplus(matmul(x.asType(.float32),
                                          weight.asType(.float32).transposed())))
        var indices: MLXArray
        if routesByHash, let tokenToExpert, let tokens {
            indices = take(tokenToExpert, tokens.reshaped([-1]), axis: 0)
        } else {
            // The bias steers selection; the weights come from the unshifted scores.
            var correction = bias
            if let images, let visionBias, let text = bias {
                correction = MLX.where(images.reshaped([-1, 1]), visionBias, text)
            }
            let shifted = correction.map { scores + $0 } ?? scores
            indices = argPartition(-shifted, kth: configuration.activatedExpertCount - 1, axis: -1)[
                0..., 0 ..< configuration.activatedExpertCount]
        }
        var chosen = takeAlong(scores, indices, axis: -1)
        // One active expert carries its own score unchanged; renormalizing it would force every
        // token's weight to one. The floor is the reference's own, and not the norm epsilon.
        if configuration.activatedExpertCount > 1 {
            chosen = chosen / (chosen.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (chosen * configuration.routeScale, indices)
    }
}

/// The additive floor a masked score carries.
///
/// Large enough that `exp` underflows to zero against any real score, small enough that a sum of
/// two of them stays finite — an actual infinity here turns into a NaN the moment a row is entirely
/// masked, and a score that is merely very negative does not.
let nfkDeepSeekMaskedScore = -Float.greatestFiniteMagnitude / 4

/// What an attention layer hands down the stack rather than recomputing.
///
/// Layers run in order and every source writes before its readers, so one slot each is enough and
/// nothing needs clearing between passes. The compressed key-value and the index keys come from the
/// layers that own a compressor, the chosen positions from the layers that own an indexer, and the
/// candidate blocks from the one layer that picks them.
final class NFKDeepSeekSharedAttention {
    var compressedKeyValue: MLXArray?
    var indexKeys: MLXArray?
    var selection: MLXArray?
    var candidates: MLXArray?
}

/// Latent attention over two key-value sources at once: a sliding window of this layer's own
/// key-value, and — where the layer compresses — the compressed positions the indexer picked, which
/// reach further back than the window does. One latent serves every head as both key and value.
///
/// A compression ratio above zero does not mean the layer compresses its own key-value. Only the
/// layers `keyValueSourceLayers` names do; the rest read what those published.
final class NFKDeepSeekAttention: Module {
    @ModuleInfo(key: "wq_a") var queryDown: Linear
    @ModuleInfo(key: "wq_b") var queryUp: Linear
    @ModuleInfo(key: "wkv") var latentKeyValue: Linear
    @ModuleInfo(key: "wo_a") var outputDown: Linear
    @ModuleInfo(key: "wo_b") var outputUp: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "kv_norm") var latentNorm: RMSNorm
    @ParameterInfo(key: "attn_sink") var sink: MLXArray
    @ModuleInfo(key: "compressor") var compressor: NFKDeepSeekCompressor?
    @ModuleInfo(key: "indexer") var indexer: NFKDeepSeekIndexer?

    let configuration: NFKMLXDeepSeekConfiguration
    let compressRatio: Int
    /// Which layer this is, which is how it finds its own lines in a decode cache.
    let layer: Int
    let rope: NFKDeepSeekRotary

    convenience init(_ c: NFKMLXDeepSeekConfiguration, compressRatio: Int) {
        self.init(c, layer: c.compressRatios.firstIndex(of: compressRatio) ?? 0,
                  ratio: compressRatio)
    }

    init(_ c: NFKMLXDeepSeekConfiguration, layer: Int, ratio: Int) {
        configuration = c
        compressRatio = ratio
        self.layer = layer
        // One frequency table per layer, chosen by whether the layer compresses: its queries, its
        // window, its compressed positions and its indexer all read that same table.
        let compressed = ratio > 0 && c.compressedLayersRotateAtCompressedBase
        rope = NFKDeepSeekRotary(width: c.ropeHeadDimensions,
                                 theta: compressed ? c.compressRopeTheta : c.ropeTheta,
                                 scaling: compressed || !c.compressedLayersRotateAtCompressedBase
                                     ? c.ropeScaling : nil)
        _queryDown.wrappedValue = Linear(c.hiddenSize, c.queryLoRARank, bias: false)
        _queryUp.wrappedValue = Linear(c.queryLoRARank, c.headCount * c.headDimensions, bias: false)
        _latentKeyValue.wrappedValue = Linear(c.hiddenSize, c.headDimensions, bias: false)
        _outputDown.wrappedValue = Linear(c.headCount * c.headDimensions / c.outputGroups,
                                          c.outputGroups * c.outputLoRARank, bias: false)
        _outputUp.wrappedValue = Linear(c.outputGroups * c.outputLoRARank, c.hiddenSize, bias: false)
        _queryNorm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.queryLoRARank, eps: c.rmsEpsilon)
        _latentNorm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.headDimensions, eps: c.rmsEpsilon)
        _sink.wrappedValue = MLXArray.zeros([c.headCount])
        // A layer carries a compressor only where it OWNS one. V4 gives one to every compressed
        // layer; V4.1 names four sources and every other compressed layer reads what they publish.
        _compressor.wrappedValue = c.ownsCompressor(layer)
            ? NFKDeepSeekCompressor(c, ratio: max(ratio, 1),
                                    headDimensions: c.headDimensions,
                                    appliesRotary: false) : nil
        _indexer.wrappedValue = c.ownsIndexer(layer)
            ? NFKDeepSeekIndexer(c, layer: layer, ratio: max(ratio, 1),
                                 ownsKeys: c.ownsCompressor(layer),
                                 derivesKeysFromCompressor: c.indexerDerivesKeysFromCompressor,
                                 rope: rope)
            : nil
        super.init()
    }

    /// - Parameter mask: the additive mask a stack that does not window its attention supplies.
    ///   Where the layers share one compressed cache the window is computed here instead, because
    ///   it is then a real constraint rather than a causal mask under another name.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?,
                        shared: NFKDeepSeekSharedAttention? = nil,
                        cache: NFKMLXDeepSeekCache? = nil) -> MLXArray {
        let c = configuration
        let (batch, length) = (x.shape[0], x.shape[1])
        // Every rotation, every window bound and every group boundary is measured from here. A
        // prefill starts at zero, which is why the chunk-relative form was right until now.
        let offset = cache?.offset ?? 0

        let lowRank = queryNorm(queryDown(served(x)))
        var queries = queryUp(served(lowRank))
            .reshaped([batch, length, c.headCount, c.headDimensions])
        if c.normalizesQueryHeads {
            queries = queries * nfkDeepSeekHeadNormScale(queries, epsilon: c.rmsEpsilon)
        }
        // One latent vector per position serves every head as both key and value.
        let latent = latentNorm(latentKeyValue(served(x)))
            .reshaped([batch, length, 1, c.headDimensions])
        queries = turnedTail(queries.transposed(0, 2, 1, 3), offset: offset)
        // Rounded where the release rounds it: in place, so the window and this step's own keys
        // both read the rounded value.
        let written = roundedKeyValue(turnedTail(latent.transposed(0, 2, 1, 3), offset: offset))
        var keys = written

        var bias = mask
        if let cache {
            // The window is whatever the ring holds after this step. It is never longer than the
            // window itself, and every position in it precedes every query in this chunk, so a
            // single-token step needs no mask at all; a chunk longer than one still does.
            keys = cache.window(layer: layer, appending: written)
            // Always masked, even for a single token: the ring plus this step is one position
            // longer than the window, so the oldest of them is already out of reach.
            bias = broadcast(windowMask(length, offset: offset, keys: keys.dim(2)),
                             to: [batch, 1, length, keys.dim(2)])
        } else if shared != nil {
            bias = broadcast(windowMask(length), to: [batch, 1, length, length])
        }
        if let shared, c.sharesOneCompressedCache || compressor != nil {
            let complete = compressRatio > 0 ? (offset + length) / compressRatio : 0
            if complete > 0 {
                let (compressed, chosen) = compressedPositions(x, lowRankQuery: lowRank,
                                                               complete: complete, shared: shared,
                                                               cache: cache)
                keys = concatenated([keys, compressed], axis: 2)
                let window = bias ?? MLXArray.zeros([batch, 1, length, keys.dim(2) - compressed.dim(2)])
                bias = concatenated([window, chosen], axis: -1)
            }
        }

        // Attention with a learned per-head SINK: an extra logit that competes with the keys and
        // carries no value, so it drains probability mass without contributing. Written out rather
        // than delegated, because the fused call has nowhere to put the extra column.
        let (query32, keys32) = (queries.asType(.float32), keys.asType(.float32))
        var scores = matmul(query32, keys32.transposed(0, 1, 3, 2)) / sqrt(Float(c.headDimensions))
        if let bias { scores = scores + bias }
        let sinkColumn = sink.reshaped([1, c.headCount, 1, 1])
        let peak = maximum(scores.max(axis: -1, keepDims: true), sinkColumn)
        let weights = exp(scores - peak)
        let total = weights.sum(axis: -1, keepDims: true) + exp(sinkColumn - peak)
        var attended = matmul(weights / total, keys32).asType(queries.dtype)

        // The values share the latent with the keys, so their rotary component has to be UNDONE on
        // the way out: the reference applies the conjugate rotation to the output's trailing channels.
        attended = turnedTail(attended, offset: offset, inverse: true)

        // The output projection runs per group of heads: each group has its own slice of `wo_a`.
        let grouped = attended.transposed(0, 2, 1, 3)
            .reshaped([batch, length, c.outputGroups, c.headCount * c.headDimensions / c.outputGroups])
        let perGroup = outputDown.weight.reshaped([c.outputGroups, c.outputLoRARank, -1])
        var reduced = [MLXArray]()
        for group in 0 ..< c.outputGroups {
            reduced.append(matmul(grouped[0..., 0..., group], perGroup[group].transposed()))
        }
        return outputUp(served(concatenated(reduced, axis: -1)))
    }

    /// Rotates the TRAILING channels of each head and leaves the rest alone, which is where this
    /// family puts its rotary. `x` is `[batch, heads, length, width]`.
    func turnedTail(_ x: MLXArray, offset: Int = 0, inverse: Bool = false,
                    stride: Int = 1) -> MLXArray {
        let width = x.shape[x.shape.count - 1]
        let turned = min(configuration.ropeHeadDimensions, width)
        return concatenated([x[.ellipsis, 0 ..< (width - turned)],
                             rope(x[.ellipsis, (width - turned)...], offset: offset,
                                  inverse: inverse, stride: stride)],
                            axis: -1)
    }

    /// The draft stack's attention.
    ///
    /// A drafted position attends to the main stack's key-value over the sliding window, and to
    /// every position of the drafted block INCLUDING the ones after it. There is no causal mask
    /// between drafts: the block is one parallel proposal, and the order among its tokens is
    /// imposed afterwards by the Markov head rather than here.
    ///
    /// Each stage runs the main states through its OWN `wkv`, so the window is this stage's
    /// key-value for those positions and not a cache handed down from another.
    func draft(_ x: MLXArray, context: NFKDeepSeekDraftContext) -> MLXArray {
        let c = configuration
        let (batch, block) = (x.shape[0], x.shape[1])
        let positions = context.mainState.shape[1]
        // The reference keeps a ring of `slidingWindow` slots, seeded across the prefill and then
        // written one slot a step. Its contents are the most recent positions, which is this slice.
        let kept = min(c.slidingWindow, positions)
        let offset = context.offset

        let main = latentNorm(latentKeyValue(served(context.mainState)))
            .reshaped([batch, positions, 1, c.headDimensions]).transposed(0, 2, 1, 3)
        var keys = roundedKeyValue(turnedTail(main))[0..., 0..., (positions - kept)...]

        var queries = queryUp(served(queryNorm(queryDown(served(x)))))
            .reshaped([batch, block, c.headCount, c.headDimensions])
        if c.normalizesQueryHeads {
            queries = queries * nfkDeepSeekHeadNormScale(queries, epsilon: c.rmsEpsilon)
        }
        queries = turnedTail(queries.transposed(0, 2, 1, 3), offset: offset)
        let drafted = latentNorm(latentKeyValue(served(x)))
            .reshaped([batch, block, 1, c.headDimensions]).transposed(0, 2, 1, 3)
        keys = concatenated([keys, roundedKeyValue(turnedTail(drafted, offset: offset))], axis: 2)

        let (query32, keys32) = (queries.asType(.float32), keys.asType(.float32))
        let scores = matmul(query32, keys32.transposed(0, 1, 3, 2)) / sqrt(Float(c.headDimensions))
        let sinkColumn = sink.reshaped([1, c.headCount, 1, 1])
        let peak = maximum(scores.max(axis: -1, keepDims: true), sinkColumn)
        let weights = exp(scores - peak)
        let total = weights.sum(axis: -1, keepDims: true) + exp(sinkColumn - peak)
        var attended = matmul(weights / total, keys32).asType(queries.dtype)
        attended = turnedTail(attended, offset: offset, inverse: true)

        let grouped = attended.transposed(0, 2, 1, 3)
            .reshaped([batch, block, c.outputGroups, c.headCount * c.headDimensions / c.outputGroups])
        let perGroup = outputDown.weight.reshaped([c.outputGroups, c.outputLoRARank, -1])
        var reduced = [MLXArray]()
        for group in 0 ..< c.outputGroups {
            reduced.append(matmul(grouped[0..., 0..., group], perGroup[group].transposed()))
        }
        return outputUp(served(concatenated(reduced, axis: -1)))
    }

    /// A GEMM's input as the release's `linear()` hands it to a narrow GEMM: rounded to fp8 in
    /// blocks where the configuration serves its GEMMs. `wo_a` is applied through an einsum and
    /// never passes through here, which is the release's behaviour and not an omission.
    private func served(_ x: MLXArray) -> MLXArray {
        guard configuration.quantizesActivations else { return x }
        return NFKMLXDeepSeekQuantization.roundTripFP8(x, blockSize: configuration.fp8BlockSize)
    }

    /// A key-value as the release's code leaves it: rounded to fp8 in blocks when the
    /// configuration asks, which it does at the window and at both of the draft stack's key-values.
    private func roundedKeyValue(_ keyValue: MLXArray) -> MLXArray {
        guard configuration.quantizesActivations else { return keyValue }
        return NFKMLXDeepSeekQuantization.roundTripFP8(keyValue,
                                                       blockSize: configuration.fp8BlockSize)
    }

    /// The sliding window as an additive mask: a query sees its own position and the
    /// `slidingWindow - 1` before it, and nothing further back through this source.
    private func windowMask(_ length: Int, offset: Int = 0, keys: Int? = nil) -> MLXArray {
        // Queries sit at absolute `offset ..< offset + length`. Keys sit at the positions the ring
        // holds, which end at the newest query and run back `count` places.
        let count = keys ?? length
        let last = offset + length - 1
        let rows = MLXArray((0 ..< length).map { Int32(offset + $0) }).reshaped([length, 1])
        let columns = MLXArray((0 ..< count).map { Int32(last - count + 1 + $0) })
            .reshaped([1, count])
        let inside = (columns .<= rows) .&& (columns .> rows - configuration.slidingWindow)
        return MLX.where(inside, MLXArray(Float(0)), MLXArray(nfkDeepSeekMaskedScore))
            .reshaped([1, 1, length, count])
    }

    /// The compressed key-value each query may reach past the window, and which of those positions
    /// it is allowed to see.
    ///
    /// A layer that owns a compressor produces the latent and publishes it; every other compressed
    /// layer reads what its source left. The indexer runs on the latent BEFORE the rotary, which is
    /// why the compressor does not apply one and this does.
    private func compressedPositions(_ x: MLXArray, lowRankQuery: MLXArray, complete: Int,
                                     shared: NFKDeepSeekSharedAttention,
                                     cache: NFKMLXDeepSeekCache? = nil)
        -> (keys: MLXArray, mask: MLXArray) {
        let offset = cache?.offset ?? 0
        // Which group this chunk's first emitted latent stands for, so its rotary starts there.
        let already = offset / max(compressRatio, 1)
        let latent = compressor.flatMap { $0(x, offset: offset, cache: cache, layer: layer) }
        if let indexer {
            shared.selection = indexer(x, lowRankQuery: lowRankQuery, latent: latent,
                                       positions: complete, shared: shared, cache: cache,
                                       layer: layer, offset: offset)
        } else if !configuration.sharesOneCompressedCache {
            // A V4 layer without an indexer attends to every group already complete at the
            // querying position, which is what the release's `get_compress_topk_idxs` lists.
            let length = x.dim(1)
            let reached = MLXArray((0 ..< length).map { Int32((offset + $0 + 1) / compressRatio) })
                .reshaped([1, length, 1])
            shared.selection = MLXArray((0 ..< complete).map(Int32.init)).reshaped([1, 1, complete])
                .< reached
        }
        cache?.selected[layer] = shared.selection
        if let latent {
            // A latent stands for the first token of its group, so group j sits at position
            // j · ratio: the same rotary, stepped by the whole group.
            var rotated = turnedTail(latent.expandedDimensions(axis: 1),
                                     offset: already * compressRatio, stride: compressRatio)
            // The release rounds it after the rotary and after the indexer has read the unrotated
            // latent, which is why this comes here and not in the compressor.
            if configuration.quantizesActivations {
                rotated = NFKMLXDeepSeekQuantization.roundTripFP4WithE4M3Scale(rotated,
                                                                              blockSize: 16)
            }
            shared.compressedKeyValue = cache.map { $0.compressed(layer: layer, appending: rotated) }
                ?? rotated
        } else if let cache, compressor != nil {
            // A source that emitted nothing this step still publishes what it holds, or the layers
            // reading it would fall back to whichever source published last.
            shared.compressedKeyValue = cache.compressed(layer: layer, appending: nil)
        }
        let published = shared.compressedKeyValue![0..., 0..., 0 ..< complete]
        // The head axis the attention carries, which the chosen positions do not: every head of a
        // query attends to the same compressed positions.
        let chosen = MLX.where(shared.selection!.expandedDimensions(axis: 1), MLXArray(Float(0)),
                               MLXArray(nfkDeepSeekMaskedScore))
        return (published, chosen)
    }
}

/// The mixture of experts: routed experts plus one shared expert every token passes through.
final class NFKDeepSeekMoE: Module {
    @ModuleInfo(key: "gate") var gate: NFKDeepSeekGate
    @ModuleInfo(key: "experts") var experts: [NFKDeepSeekExpert]
    @ModuleInfo(key: "shared_experts") var shared: NFKDeepSeekExpert

    let configuration: NFKMLXDeepSeekConfiguration
    /// Set where the routed experts are held in the form the release stores them rather than as
    /// parameters of this module. `experts` is then empty and the router's choice is decoded as it
    /// is made. The shared expert stays resident, because every token passes through it.
    let paging: NFKDeepSeekExpertPager?

    init(_ c: NFKMLXDeepSeekConfiguration, layer: Int,
         expertStore: NFKMLXDeepSeekExpertStore? = nil) {
        configuration = c
        paging = expertStore.map { NFKDeepSeekExpertPager(store: $0, layer: layer) }
        _gate.wrappedValue = NFKDeepSeekGate(c, layer: layer)
        let served = c.quantizesActivations ? c.fp8BlockSize : nil
        _experts.wrappedValue = expertStore != nil ? [] : (0 ..< c.routedExpertCount).map { _ in
            NFKDeepSeekExpert(hiddenSize: c.hiddenSize, intermediateSize: c.expertIntermediateSize,
                              limit: c.swigluLimit, servedBlock: served)
        }
        _shared.wrappedValue = NFKDeepSeekExpert(hiddenSize: c.hiddenSize,
                                                 intermediateSize: c.expertIntermediateSize,
                                                 limit: c.swigluLimit, servedBlock: served)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, tokens: MLXArray?, images: MLXArray? = nil) -> MLXArray {
        let c = configuration
        let shape = x.shape
        let flat = x.reshaped([-1, c.hiddenSize])
        let (weights, indices) = gate(flat, tokens: tokens, images: images)
        eval(indices)

        // Every token passes through the shared expert; the routed ones add to it. The subscript
        // writes below go through MLXArray's own storage, so the binding itself never changes.
        // Summed in float32 and rounded once, as the reference sums into a float32 buffer.
        let result = shared(flat).asType(.float32)
        let chosen = indices.asArray(Int32.self)
        let scale = weights.asArray(Float.self)
        let slots = c.activatedExpertCount
        let paged = paging.map { pagedContributions(of: flat, chosen: chosen, scale: scale,
                                                    slots: slots, paging: $0) }

        for token in 0 ..< flat.shape[0] {
            for slot in 0 ..< slots {
                let position = token * slots + slot
                let expert = Int(chosen[position])
                let routed: MLXArray?
                if let paged {
                    routed = paged[position]
                } else if expert >= 0 && expert < experts.count {
                    routed = experts[expert](flat[token ..< (token + 1)], weight: scale[position])
                } else {
                    routed = nil
                }
                // Already weighted: the expert applies its routing weight before `w2`.
                guard let contribution = routed else { continue }
                result[token ..< (token + 1)] = result[token ..< (token + 1)] + contribution
            }
        }
        return result.reshaped(shape).asType(x.dtype)
    }

    /// Every routed token's contribution, computed expert by expert.
    ///
    /// @discussion Grouping by expert is what makes paging affordable: an expert decodes once for a
    /// whole chunk rather than once for each token that reaches it. The arithmetic stays the
    /// resident path's, one token at a time against the same matrices, so the two paths produce the
    /// same floats rather than close ones. Passing an expert's tokens through as one batch would be
    /// fewer and larger matrix multiplies, and it would change the reduction each multiply performs;
    /// that is the trade a maintainer who wants the speed would be making.
    ///
    /// The contributions are held until the accumulation reads them, which is
    /// `tokens x activatedExpertCount x hiddenSize` floats. Each expert's are evaluated before the
    /// next expert is decoded, so one decoded expert is live at a time.
    private func pagedContributions(of flat: MLXArray, chosen: [Int32], scale: [Float],
                                    slots: Int,
                                    paging: NFKDeepSeekExpertPager) -> [Int: MLXArray] {
        var routed = [Int: [Int]]()
        for position in 0 ..< (flat.shape[0] * slots) {
            let expert = Int(chosen[position])
            guard expert >= 0, expert < configuration.routedExpertCount else { continue }
            routed[expert, default: []].append(position)
        }
        var contributions = [Int: MLXArray]()
        for expert in routed.keys.sorted() {
            guard let module = paging.store.expert(layer: paging.layer, index: expert) else {
                continue
            }
            var produced = [MLXArray]()
            for position in routed[expert] ?? [] {
                let token = position / slots
                let contribution = module(flat[token ..< (token + 1)], weight: scale[position])
                contributions[position] = contribution
                produced.append(contribution)
            }
            eval(produced)
        }
        return contributions
    }
}

/// What a draft stage reads in place of a mask: the main stack's own states, already projected,
/// and the position the first drafted token sits at.
struct NFKDeepSeekDraftContext {
    let mainState: MLXArray
    let offset: Int
}

/// One decoder layer. A draft stage is this block with a different attention path, which is why it
/// subclasses rather than duplicates: the checkpoint keys `mtp.<stage>.attn` the same way.
class NFKDeepSeekBlock: Module {
    @ModuleInfo(key: "attn") var attention: NFKDeepSeekAttention
    @ModuleInfo(key: "ffn") var feedForward: NFKDeepSeekMoE
    @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "hc_attn") var attentionConnection: NFKDeepSeekHyperConnection
    @ModuleInfo(key: "hc_ffn") var feedForwardConnection: NFKDeepSeekHyperConnection
    /// The n-gram memory, on the layers a release names. It belongs to the block because that is
    /// where the checkpoint keys it, and it runs before the block rather than inside it.
    @ModuleInfo(key: "engram") var engram: NFKDeepSeekEngram?

    let configuration: NFKMLXDeepSeekConfiguration

    init(_ c: NFKMLXDeepSeekConfiguration, layer: Int,
         expertStore: NFKMLXDeepSeekExpertStore? = nil, pagingNgramTable: Bool = false) {
        configuration = c
        _attention.wrappedValue = NFKDeepSeekAttention(c, layer: layer,
                                                       ratio: c.compressRatio(of: layer))
        _feedForward.wrappedValue = NFKDeepSeekMoE(c, layer: layer, expertStore: expertStore)
        _attentionNorm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _feedForwardNorm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _attentionConnection.wrappedValue = NFKDeepSeekHyperConnection(
            copies: c.hyperConnectionCopies, hiddenSize: c.hiddenSize,
            iterations: c.sinkhornIterations, epsilon: c.hyperConnectionEpsilon,
            normEpsilon: c.rmsEpsilon)
        _feedForwardConnection.wrappedValue = NFKDeepSeekHyperConnection(
            copies: c.hyperConnectionCopies, hiddenSize: c.hiddenSize,
            iterations: c.sinkhornIterations, epsilon: c.hyperConnectionEpsilon,
            normEpsilon: c.rmsEpsilon)
        _engram.wrappedValue = c.engramLayerIDs.firstIndex(of: layer).map {
            NFKDeepSeekEngram(c, rows: c.engramEmbeddingCounts[$0], pagingTable: pagingNgramTable)
        }
        super.init()
    }

    /// `x` is `[batch, length, copies, hidden]` — the hyper-connected residual stream. `incoming` is
    /// the read weight the previous sub-block predicted, where the family pipelines them; the
    /// returned weight is what the next block reads with.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, tokens: MLXArray?,
                        incoming: MLXArray? = nil,
                        shared: NFKDeepSeekSharedAttention? = nil,
                        drafting: NFKDeepSeekDraftContext? = nil,
                        images: MLXArray? = nil,
                        cache: NFKMLXDeepSeekCache? = nil) -> (state: MLXArray, read: MLXArray) {
        let pipelined = configuration.pipelinesHyperConnectionRead
        let (readA, writeA, combineA) = attentionConnection.weights(x)
        let reducedA = attentionConnection.reduce(x, read: pipelined ? incoming! : readA)
        let normed = attentionNorm(reducedA)
        let attended = drafting.map { attention.draft(normed, context: $0) }
            ?? attention(normed, mask: mask, shared: shared, cache: cache)
        let afterAttention = attentionConnection.expand(attended, residual: x,
                                                        write: writeA, combine: combineA)

        let (readF, writeF, combineF) = feedForwardConnection.weights(afterAttention)
        let reducedF = feedForwardConnection.reduce(afterAttention, read: pipelined ? readA : readF)
        let lifted = feedForward(feedForwardNorm(reducedF), tokens: tokens, images: images)
        let state = feedForwardConnection.expand(lifted, residual: afterAttention,
                                                 write: writeF, combine: combineF)
        return (state, readF)
    }
}

/// A DeepSeek V4 decoder. See the file comment for what is and is not implemented.
public final class NFKMLXDeepSeekNet: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "layers") var layers: [NFKDeepSeekBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "head") var head: Linear
    /// Absent where the copies collapse by identity, which is what V4.1 does.
    @ModuleInfo(key: "hc_head") var headConnection: NFKDeepSeekHyperHead?

    let configuration: NFKMLXDeepSeekConfiguration
    /// What this decoder holds in the form the release stores it rather than as float parameters.
    public let paging: NFKMLXDeepSeekPaging
    /// The routed experts, where they are held in their stored form instead of as parameters.
    public let expertStore: NFKMLXDeepSeekExpertStore?
    /// Turns token ids into the rows the n-gram memory looks up, where a release carries one.
    let ngramHash: NFKDeepSeekNgramHash?

    /// - Parameter compressedTokens: the collapsed id space the n-gram memory hashes over, one
    ///   entry per token id. A release states the size it was trained with, and every hash
    ///   multiplier is derived from that size, so a configuration that names one and is built
    ///   without it hashes over raw ids and reads the wrong rows.
    /// - Parameter paging: what this decoder holds in the form the release stores it rather than as
    ///   float parameters. A release whose decoded weights exceed the machine is built with one of
    ///   these and loaded through the same `loadWeights(into:fromDirectory:)`.
    init(_ c: NFKMLXDeepSeekConfiguration, compressedTokens: [Int]? = nil,
         paging: NFKMLXDeepSeekPaging = .none) {
        configuration = c
        self.paging = paging
        let store = paging.routedExperts
            ? NFKMLXDeepSeekExpertStore(configuration: c, cacheByteBudget: paging.expertCacheBytes)
            : nil
        expertStore = store
        ngramHash = c.engramLayerIDs.isEmpty
            ? nil : NFKDeepSeekNgramHash(c, compressedTokens: compressedTokens)
        _embed.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layerCount).map {
            NFKDeepSeekBlock(c, layer: $0, expertStore: store, pagingNgramTable: paging.ngramTables)
        }
        _norm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _head.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
        _headConnection.wrappedValue = c.collapsesThroughLearnedHead
            ? NFKDeepSeekHyperHead(copies: c.hyperConnectionCopies, hiddenSize: c.hiddenSize,
                                   epsilon: c.hyperConnectionEpsilon, normEpsilon: c.rmsEpsilon)
            : nil
        super.init()
    }

    /// The logits for `tokens`. With a cache this is one step of a generation, and the cache
    /// carries everything the step needs from the ones before it.
    public func callAsFunction(_ tokens: MLXArray,
                               images: MLXArray? = nil,
                               embeddings: MLXArray? = nil,
                               cache: NFKMLXDeepSeekCache? = nil) -> MLXArray {
        var read: MLXArray?
        let states = hiddenStates(tokens, finalRead: &read, images: images,
                                  embeddings: embeddings, cache: cache)
        // The head is held float32 and reads in float32, so the logits come out float32.
        return head(finalState(states.last!, read: read).asType(.float32))
    }

    /// The collapsed, normed stream the head reads — what the reference reports as its last state.
    ///
    /// A learned head predicts its own read weights. Collapsing by identity instead means reading
    /// with the weight the last block produced, which is the pipeline's final link rather than a
    /// separate mechanism.
    func finalState(_ hidden: MLXArray, read: MLXArray? = nil) -> MLXArray {
        if let headConnection { return norm(headConnection.reduce(hidden)) }
        return norm((read!.expandedDimensions(axis: -1) * hidden.asType(.float32)).sum(axis: 2)
            .asType(hidden.dtype))
    }

    /// The main-stack states the draft stack reads.
    ///
    /// DSpark reads the stream ENTERING each `dsparkTargetLayers` attention, not what those layers
    /// produced, and reads it collapsed to one copy by averaging the hyper-connection copies. A
    /// trace from `hiddenStates(_:)` holds exactly those, one entry per layer, so this selects and
    /// concatenates rather than running the stack a second time.
    public func draftStates(from states: [MLXArray]) -> MLXArray? {
        // V4.1 reads the stream entering a target layer; V4 Pro 0813 reads what the layer produced,
        // which the trace holds one entry later.
        let shift = configuration.draftReadsTargetLayerOutputs ? 1 : 0
        let targets = configuration.dsparkTargetLayers.map { $0 + shift }
        guard let last = targets.max(), last < states.count else { return nil }
        // Summed in float32 and rounded once, as torch's mean over a bf16 tensor is.
        return concatenated(targets.map {
            states[$0].asType(.float32).mean(axis: 2).asType(states[$0].dtype)
        }, axis: -1)
    }

    /// The same, over a whole committed sequence. The draft stack proposes what follows it, so the
    /// last token here is the one the decoder has just committed.
    public func draftStates(forTokens tokens: MLXArray) -> MLXArray? {
        draftStates(from: hiddenStates(tokens))
    }

    /// The residual stream entering the stack and after every layer, for the isolation harness.
    ///
    /// Each state is the full `[batch, length, copies, hidden]` stream, because that is what the
    /// reference's `output_hidden_states` records — the copies are the residual here, not a detail.
    func hiddenStates(_ tokens: MLXArray) -> [MLXArray] {
        var read: MLXArray?
        return hiddenStates(tokens, finalRead: &read)
    }

    /// The same trace, handing back the read weight the last block produced, which a stack that
    /// collapses by identity needs and one with a learned head ignores.
    /// - Parameter embeddings: `[batch, length, hidden]` in place of what `embed` would look up,
    ///   which is how an image span reaches the decoder. The token ids are still required and are
    ///   still the release's image placeholder at those positions: the n-gram memory hashes over
    ///   ids, and the router's image bias reads `images`, neither of which an embedding carries.
    func hiddenStates(_ tokens: MLXArray, finalRead: inout MLXArray?,
                      images: MLXArray? = nil,
                      embeddings: MLXArray? = nil,
                      cache: NFKMLXDeepSeekCache? = nil) -> [MLXArray] {
        let c = configuration
        // The residual stream is `copies` parallel copies of the embedding from the outset.
        let embedded = (embeddings ?? embed(tokens)).expandedDimensions(axis: 2)
        var hidden = repeated(embedded, count: c.hyperConnectionCopies, axis: 2)
        var states = [MLXArray]()

        let (batch, length) = (tokens.shape[0], tokens.shape[1])
        let mask: MLXArray? = length > 1 ? NFKMLXLanguageNet.causalMask(length, offset: 0) : nil
        // The first block has no predecessor to read from, so it reads the first copy alone.
        var read = concatenated(
            [MLXArray.ones([batch, length, 1]),
             MLXArray.zeros([batch, length, c.hyperConnectionCopies - 1])], axis: -1)
        let shared = NFKDeepSeekSharedAttention()
        let hashes = ngramHash.map { $0(tokens, cache: cache) }
        for (index, layer) in layers.enumerated() {
            // The n-gram memory writes into the stream BEFORE the block reads it, which is why it
            // is a sibling of the block's sublayers rather than one of them. Its contribution is
            // part of the state the next block sees, so the trace records it after the write.
            if let engram = layer.engram, let hashes,
               let column = c.engramLayerIDs.firstIndex(of: index) {
                hidden = engram(hidden, hashes: hashes[0..., 0..., column])
            }
            states.append(hidden)
            (hidden, read) = layer(hidden, mask: mask, tokens: tokens, incoming: read,
                                   shared: shared, images: images, cache: cache)
        }
        states.append(hidden)
        finalRead = read
        // The draft stack reads the target layers' states at every committed position, which a
        // cached run has to accumulate: a chunk sees only its own.
        if let cache, cache.collectsDraftStates, let drafted = draftStates(from: states) {
            cache.rememberDraftStates(drafted)
        }
        // After the layers have read it, so every one of them saw the same starting position.
        cache?.advance(by: length)
        return states
    }
}

/// Building the decoder, reading a release's configuration, and describing what its checkpoint holds.
@objc(NFKMLXDeepSeek)
public final class NFKMLXDeepSeek: NSObject {

    /// Whether this configuration names a shape the runtime modules do not build.
    ///
    /// @discussion V4.1's parameters are enumerated analytically and checked against its released
    /// headers, but `NFKMLXDeepSeekNet` still builds V4's arrangement: a compressor on every
    /// compressed layer, a learned collapse at the top, and no n-gram memory. Building it anyway
    /// would produce a module whose shapes are wrong in ways a load hides — MLX's
    /// `update(parameters:)` adopts a checkpoint's shapes wholesale, so the failure would surface in
    /// the forward pass rather than at load. The refusal names the gap instead.
    static func unbuiltMechanism(in c: NFKMLXDeepSeekConfiguration,
                                 compressedTokens: [Int]? = nil) -> String? {
        // Every n-gram hash multiplier is derived from the size of the collapsed id space, and the
        // collapse itself decides which token ids share a bucket. A release that states a size and
        // is handed no map would hash over raw ids and read the wrong row of a table with 384
        // million of them — wrong in the forward pass only, which no shape check and no load sees.
        if !c.engramLayerIDs.isEmpty && c.engramCompressedVocabularySize > 0
            && compressedTokens == nil {
            return "the n-gram memory's collapsed token map"
        }
        return nil
    }

    /// The first row the release's activation quantizers could not divide into their blocks.
    ///
    /// @discussion The kernels assert it, and the round trips here reshape by it, so a
    /// configuration that quantizes and fails it is refused at construction rather than trapping
    /// in the middle of a forward pass. The released V4.1 Flash passes: its heads are 512 wide and
    /// its index heads 128.
    static func unroundableWidth(in c: NFKMLXDeepSeekConfiguration) -> String? {
        if c.headDimensions % c.fp8BlockSize != 0 {
            return "a head of \(c.headDimensions) is not a whole number of fp8 blocks of "
                + "\(c.fp8BlockSize)"
        }
        if c.headDimensions % 16 != 0 {
            return "a head of \(c.headDimensions) is not a whole number of the compressed "
                + "latent's groups of 16"
        }
        let fourBit = NFKMLXDeepSeekQuantization.fp4BlockSize
        if c.indexHeadDimensions % fourBit != 0 {
            return "an index head of \(c.indexHeadDimensions) is not a whole number of fp4 blocks "
                + "of \(fourBit)"
        }
        return nil
    }

    static func makeNet(_ configuration: NFKMLXDeepSeekConfiguration,
                        compressedTokens: [Int]? = nil,
                        paging: NFKMLXDeepSeekPaging = .none) throws -> NFKMLXDeepSeekNet {
        if let missing = unbuiltMechanism(in: configuration, compressedTokens: compressedTokens) {
            throw NFKMLXError.unsupportedConfiguration(
                "this decoder's mechanisms are implemented and measured, but \(missing) is not "
                + "supplied, so a module built here would not be the model the release describes")
        }
        if configuration.quantizesActivations,
           let width = unroundableWidth(in: configuration) {
            throw NFKMLXError.unsupportedConfiguration(
                "activation rounding was asked for, and \(width); the release's quantizers "
                + "assert that every row divides into its blocks")
        }
        return NFKMLXDeepSeekNet(configuration, compressedTokens: compressedTokens, paging: paging)
    }

    /// The draft stages a checkpoint index beside `config.json` holds, or nil where there is none.
    static func draftStageCount(besideConfig url: URL) -> Int? {
        let index = url.deletingLastPathComponent().appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: index),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let map = root["weight_map"] as? [String: Any] else { return nil }
        let stages = Set(map.keys.compactMap { key -> Int? in
            let parts = key.split(separator: ".")
            return parts.count > 1 && parts[0] == "mtp" ? Int(parts[1]) : nil
        })
        return stages.count
    }

    /// Reads a released `config.json`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXDeepSeekConfiguration {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        // V4 states the decoder at the top level; V4.1 is multimodal and nests it under `text_config`.
        let json = (root["text_config"] as? [String: Any]) ?? root
        let kind = (json["model_type"] as? String) ?? ""
        guard kind == "deepseek_v4" || kind == "deepseek_v41_text" else {
            throw NFKMLXError.unsupportedConfiguration("this reads the deepseek_v4 and v4.1 decoders")
        }
        let isV41 = kind == "deepseek_v41_text"
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (json[key] as? NSNumber)?.floatValue ?? fallback }

        var resolved = NFKMLXDeepSeekConfiguration(
            hiddenSize: integer("hidden_size", 4096),
            layerCount: integer("num_hidden_layers", 43),
            vocabularySize: integer("vocab_size", 129_280),
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            headCount: integer("num_attention_heads", 64),
            headDimensions: integer("head_dim", 512),
            ropeHeadDimensions: integer("qk_rope_head_dim", 64),
            queryLoRARank: integer("q_lora_rank", 1024),
            outputLoRARank: integer("o_lora_rank", 1024),
            outputGroups: integer("o_groups", 8),
            slidingWindow: integer("sliding_window", 128),
            ropeTheta: real("rope_theta", 10_000),
            compressRopeTheta: real("compress_rope_theta", 160_000),
            hyperConnectionCopies: integer("hc_mult", 4),
            sinkhornIterations: integer("hc_sinkhorn_iters", 20),
            hyperConnectionEpsilon: real("hc_eps", 1e-6),
            indexHeadCount: integer("index_n_heads", 64),
            indexHeadDimensions: integer("index_head_dim", 128),
            indexTopK: integer("index_topk", 512),
            routedExpertCount: integer("n_routed_experts", 256),
            sharedExpertCount: integer("n_shared_experts", 1),
            activatedExpertCount: integer("num_experts_per_tok", 6),
            expertIntermediateSize: integer("moe_intermediate_size", 2048),
            routeScale: real("routed_scaling_factor", 1.5),
            swigluLimit: real("swiglu_limit", 10),
            hashLayerCount: integer("num_hash_layers", isV41 ? 0 : 3),
            // The second router bias belongs to the vision path, not to the version: the reference
            // builds `bias_vl` only where a vision tower is configured.
            routerHasVisionBias: root["vision_config"] != nil,
            compressRatios: json["compress_ratios"] as? [Int],
            // V4.1 pools without a learned per-slot bias, collapses the copies by identity, and its
            // indexer reads the layer compressor's latent instead of compressing a second time.
            compressorHasPositionBias: !isV41,
            keyValueSourceLayers: json["kv_source_layer_ids"] as? [Int],
            indexSourceLayers: json["index_source_layer_ids"] as? [Int],
            candidateSourceLayer: json["candidate_source_layer_id"] as? Int,
            candidateTopKBlocks: integer("candidate_topk_blocks", 0),
            candidateBlockSize: integer("candidate_block_size", 0),
            collapsesThroughLearnedHead: !isV41,
            indexerDerivesKeysFromCompressor: isV41,
            // V4 normalizes each query head after the up-projection; V4.1 does not. Both releases'
            // code builds a compressed layer's rotary at the compressed base with YaRN on and an
            // uncompressed layer's at `rope_theta` with it off.
            normalizesQueryHeads: !isV41,
            compressedLayersRotateAtCompressedBase: true,
            pipelinesHyperConnectionRead: isV41,
            engramLayerIDs: (json["engram_layer_ids"] as? [Int]) ?? [],
            engramEmbeddingCounts: (json["engram_num_embeddings"] as? [Int]) ?? [],
            engramMaxNgramSize: integer("engram_max_ngram_size", 4),
            engramHeadCount: integer("engram_n_heads", 8),
            engramHeadDimensions: integer("engram_head_dim", 256),
            engramVocabularySize: integer("engram_vocab_size", 16_000_000),
            engramCompressedVocabularySize: integer("engram_compressed_vocab_size", 0),
            engramPadToken: integer("engram_pad_token_id", 2),
            nextTokenPredictionLayers: integer("num_nextn_predict_layers", 1),
            vision: (root["vision_config"] as? [String: Any]).map { tower in
                func size(_ key: String, _ fallback: Int) -> Int {
                    (tower[key] as? NSNumber)?.intValue ?? fallback
                }
                return NFKMLXDeepSeekVisionConfiguration(
                    layerCount: size("num_hidden_layers", 32),
                    hiddenSize: size("hidden_size", 1024),
                    headCount: size("num_attention_heads", 16),
                    intermediateSize: size("intermediate_size", 2816),
                    patchSize: size("patch_size", 14),
                    ropeTheta: (tower["rope_theta"] as? NSNumber)?.floatValue ?? 10_000,
                    downsampleRatio: size("downsample_ratio", 3),
                    outputSize: integer("hidden_size", 5120),
                    // The release's own `ModelArgs` names these at the top level with a `vision_`
                    // prefix, and a Hugging Face config carries the tower's fields inside
                    // `vision_config` with the prefix dropped. Both spellings are read, the tower's
                    // first, because which one a release writes is not knowable from here.
                    maximumTokenCount: size("max_n_token", integer("vision_max_n_token", 1024)),
                    minimumPixels: size("min_pixels", integer("vision_min_pixels", 544 * 544)),
                    maximumWidthHeightRatio: (tower["max_wh_ratio"] as? NSNumber)?.intValue
                        ?? (json["vision_max_wh_ratio"] as? NSNumber)?.intValue,
                    imageTokenID: integer("image_token_id", 129_264))
            },
            dsparkBlockSize: integer("dspark_block_size", 0),
            dsparkTargetLayers: (json["dspark_target_layer_ids"] as? [Int]) ?? [],
            dsparkMarkovRank: integer("dspark_markov_rank", 256),
            dsparkExpertCount: integer("dspark_n_routed_experts", 0),
            dsparkActiveExpertCount: integer("dspark_num_experts_per_tok", 0),
            dsparkNoiseToken: integer("dspark_noise_token_id", 0))
        // V4 Pro extends its window with YaRN, which carries no parameters and so is invisible to the
        // structural check — a run of Pro without it would be silently wrong past the trained length.
        resolved.ropeScaling = try NFKMLXRoPEScaling.read(
            json["rope_scaling"],
            maximumPositions: integer("max_position_embeddings", 4096))
        // The block an fp8 scale covers is the release's to state, and the two versions disagree:
        // V4 blocks at 128, V4.1 at 32. It sits beside the decoder rather than inside `text_config`,
        // so it is read from the root.
        if let quantization = (root["quantization_config"] as? [String: Any])
            ?? (json["quantization_config"] as? [String: Any]),
           let block = (quantization["weight_block_size"] as? [Any])?.first as? NSNumber {
            resolved.fp8BlockSize = block.intValue
        }
        // V4's code builds its norms with float32 weights; V4.1's builds them in the default dtype.
        resolved.holdsNormWeightsInFloat32 = !isV41
        resolved.draftReadsTargetLayerOutputs = !isV41
        resolved.draftMarkovTableNames = isV41 ? ["embed", "head"] : ["markov_w1", "markov_w2"]
        // How many draft stages the release carries is the checkpoint's to say. V4 Pro 0813's
        // config.json says one while its checkpoint, and its own inference config, carry three, so
        // an index beside the config is read where there is one.
        if let stages = draftStageCount(besideConfig: url), stages > 0 {
            resolved.nextTokenPredictionLayers = stages
        }
        // The release computes in the dtype it declares, and its own inference code sets bf16 as
        // the default unconditionally.
        let declared = (root["dtype"] ?? root["torch_dtype"] ?? json["dtype"] ?? json["torch_dtype"]) as? String
        resolved.computesInBFloat16 = declared == "bfloat16"
        return resolved
    }

    /// The module key a release tensor name maps to.
    ///
    /// @discussion The module's keys ARE the release's names except where MLX's key rules force a
    /// nesting: the release flattens the hyper-connection parameters (`hc_attn_fn`, `hc_head_scale`)
    /// where the module holds them as children of a connection module, and the release calls the
    /// latent norm `attn.norm` where the module says `kv_norm` — `norm` alone would collide with the
    /// block pattern MLX uses for the final norm.
    static func moduleKey(forRelease name: String) -> String {
        var key = name
        for site in ["attn", "ffn", "head"] {
            key = key.replacingOccurrences(of: "hc_\(site)_fn", with: "hc_\(site).fn")
            key = key.replacingOccurrences(of: "hc_\(site)_base", with: "hc_\(site).base")
            key = key.replacingOccurrences(of: "hc_\(site)_scale", with: "hc_\(site).scale")
        }
        key = key.replacingOccurrences(of: "markov_head.markov_w1.", with: "markov_head.embed.")
        key = key.replacingOccurrences(of: "markov_head.markov_w2.", with: "markov_head.head.")
        return key.replacingOccurrences(of: ".attn.norm.", with: ".attn.kv_norm.")
    }

    /// Decodes a released shard's quantized tensors into the float parameters a module holds.
    ///
    /// @discussion The release stores a weight as bytes plus a companion `.scale` tensor, so a
    /// checkpoint's arrays are not parameters until they are decoded. A weight whose last axis is
    /// half its declared width is 4-bit packed; anything else with a scale is fp8. A tensor with no
    /// scale — the norms, the biases, the routing tables — passes through as it is.
    ///
    /// Loading a whole release is out of reach on one machine, so this takes the arrays a caller has
    /// already read rather than a directory: it decodes a shard, or one tensor, at whatever scale the
    /// caller can hold.
    ///
    /// - Parameters:
    ///   - arrays: a shard's contents, weights and `.scale` companions together.
    ///   - shapes: the float shape each parameter takes, which is what separates 4-bit from fp8.
    ///     `expectedParameters(for:)` produces it.
    ///
    /// Each entry is evaluated as it is decoded, so the returned arrays are already materialized and
    /// only one decode is ever in flight. Returning lazy graphs instead would hold every entry's
    /// sources and intermediates until the caller evaluated them, which is where the peak memory of a
    /// shard would otherwise land.
    ///
    /// Introduced in InferKit 0.1.0.
    /// - Parameter fp8BlockSize: the release's `weight_block_size`. Passing the wrong one decodes
    ///   every fp8 weight against the wrong part of its scale grid, silently.
    public static func dequantized(_ arrays: [String: MLXArray], shapes: [String: [Int]],
                                   fp8BlockSize: Int = NFKMLXDeepSeekQuantization.fp8BlockSize)
        -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (name, value) in arrays where !name.hasSuffix(".scale") {
            // A companion is named for the weight it decodes, so only a `.weight` has one. Deriving
            // the name by substitution instead made every OTHER parameter look up ITSELF — the
            // hyper-connection weights (`hc_attn_fn`), the engram's `q_weight`, a router bias: none
            // contains `.weight`, the substitution changed nothing, and the tensor was decoded
            // through the fp8 table against its own bytes. Wrong in the forward pass only, and a
            // load that covered every parameter reported nothing.
            guard name.hasSuffix(".weight"),
                  let scale = arrays[String(name.dropLast(".weight".count)) + ".scale"] else {
                result[name] = value
                continue
            }
            let packedFourBit = shapes[name].map { $0.count == 2 && value.shape.last == $0[1] / 2 }
                ?? false
            let decoded = packedFourBit
                ? NFKMLXDeepSeekQuantization.dequantizeFP4(packedBytes: value, scaleBytes: scale)
                : NFKMLXDeepSeekQuantization.dequantizeFP8(bytes: value, scaleBytes: scale,
                                                          blockSize: fp8BlockSize)
            // A decode is a lazy graph, and an unevaluated graph pins its sources and its
            // intermediates. Deferring every entry to one evaluation at the end therefore holds the
            // whole checkpoint's decode live at once — and the expanded scale array alone is the
            // weight's full size, so the intermediates dominate. Evaluating here keeps exactly one
            // decode in flight.
            eval(decoded)
            result[name] = decoded
        }
        return result
    }

    /// Every parameter the decoder declares, as `name → shape`, WITHOUT building it.
    ///
    /// @discussion A released configuration has 43 layers of 257 experts; instantiating that at float
    /// precision would need hundreds of gigabytes, so the structural check enumerates the architecture
    /// analytically instead. The names are the checkpoint's own.
    static func expectedParameters(for c: NFKMLXDeepSeekConfiguration) -> [String: [Int]] {
        var shapes = [String: [Int]]()
        shapes["embed.weight"] = [c.vocabularySize, c.hiddenSize]
        if c.collapsesThroughLearnedHead {
            shapes["hc_head_fn"] = [c.hyperConnectionCopies, c.hyperConnectionCopies * c.hiddenSize]
            shapes["hc_head_base"] = [c.hyperConnectionCopies]
            shapes["hc_head_scale"] = [1]
        }
        shapes["norm.weight"] = [c.hiddenSize]
        shapes["head.weight"] = [c.vocabularySize, c.hiddenSize]

        for layer in 0 ..< c.layerCount {
            let a = "layers.\(layer).attn."
            shapes[a + "wq_a.weight"] = [c.queryLoRARank, c.hiddenSize]
            shapes[a + "wq_b.weight"] = [c.headCount * c.headDimensions, c.queryLoRARank]
            shapes[a + "wkv.weight"] = [c.headDimensions, c.hiddenSize]
            shapes[a + "wo_a.weight"] = [c.outputGroups * c.outputLoRARank,
                                         c.headCount * c.headDimensions / c.outputGroups]
            shapes[a + "wo_b.weight"] = [c.hiddenSize, c.outputGroups * c.outputLoRARank]
            shapes[a + "q_norm.weight"] = [c.queryLoRARank]
            shapes[a + "kv_norm.weight"] = [c.headDimensions]
            shapes[a + "attn_sink"] = [c.headCount]

            // A layer that owns a compressor carries one. Pooling more than one position per group
            // needs the gate that makes them compete; a ratio of one is a plain projection.
            let ratio = c.compressRatio(of: layer)
            if c.ownsCompressor(layer) {
                let width = (ratio == 4 ? 2 : 1) * c.headDimensions
                shapes[a + "compressor.wkv.weight"] = [width, c.hiddenSize]
                shapes[a + "compressor.norm.weight"] = [c.headDimensions]
                if ratio > 1 {
                    shapes[a + "compressor.wgate.weight"] = [width, c.hiddenSize]
                }
                if c.compressorHasPositionBias {
                    shapes[a + "compressor.ape"] = [ratio, width]
                }
            }
            if c.ownsIndexer(layer) {
                let i = a + "indexer."
                shapes[i + "wq_b.weight"] = [c.indexHeadCount * c.indexHeadDimensions, c.queryLoRARank]
                shapes[i + "weights_proj.weight"] = [c.indexHeadCount, c.hiddenSize]
                if c.indexerDerivesKeysFromCompressor {
                    // The keys come from the layer's own compressed latent, so only a layer that
                    // compresses its own key-value can make them; the rest read that layer's cache.
                    if c.ownsCompressor(layer) {
                        shapes[i + "wk.weight"] = [c.indexHeadDimensions, c.headDimensions]
                        shapes[i + "k_norm.weight"] = [c.indexHeadDimensions]
                    }
                } else {
                    let indexWidth = 2 * c.indexHeadDimensions
                    shapes[i + "compressor.wkv.weight"] = [indexWidth, c.hiddenSize]
                    shapes[i + "compressor.wgate.weight"] = [indexWidth, c.hiddenSize]
                    shapes[i + "compressor.norm.weight"] = [c.indexHeadDimensions]
                    shapes[i + "compressor.ape"] = [4, indexWidth]
                }
            }

            // The n-gram memory, on the layers the release names.
            if let engram = c.engramLayerIDs.firstIndex(of: layer) {
                let e = "layers.\(layer).engram."
                shapes[e + "embed.weight"] = [c.engramEmbeddingCounts[engram], c.engramHeadDimensions]
                shapes[e + "wkv.weight"] = [c.hiddenSize * (c.hyperConnectionCopies + 1),
                                            c.engramHashColumns * c.engramHeadDimensions]
                shapes[e + "q_weight"] = [c.hyperConnectionCopies, c.hiddenSize]
                shapes[e + "k_weight"] = [c.hyperConnectionCopies, c.hiddenSize]
            }

            shapes["layers.\(layer).attn_norm.weight"] = [c.hiddenSize]
            shapes["layers.\(layer).ffn_norm.weight"] = [c.hiddenSize]

            // Hyper-connections: two per block, plus the head's own at the top level.
            let mix = (2 + c.hyperConnectionCopies) * c.hyperConnectionCopies
            let mixWidth = c.hyperConnectionCopies * c.hiddenSize
            for part in ["attn", "ffn"] {
                shapes["layers.\(layer).hc_\(part)_fn"] = [mix, mixWidth]
                shapes["layers.\(layer).hc_\(part)_base"] = [mix]
                shapes["layers.\(layer).hc_\(part)_scale"] = [3]
            }

            let f = "layers.\(layer).ffn."
            shapes[f + "gate.weight"] = [c.routedExpertCount, c.hiddenSize]
            if c.routesByHash(layer) {
                shapes[f + "gate.tid2eid"] = [c.vocabularySize, c.activatedExpertCount]
            } else {
                shapes[f + "gate.bias"] = [c.routedExpertCount]
            }
            if c.routerHasVisionBias {
                shapes[f + "gate.bias_vl"] = [c.routedExpertCount]
            }
            for expert in 0 ..< c.routedExpertCount {
                let e = f + "experts.\(expert)."
                shapes[e + "w1.weight"] = [c.expertIntermediateSize, c.hiddenSize]
                shapes[e + "w2.weight"] = [c.hiddenSize, c.expertIntermediateSize]
                shapes[e + "w3.weight"] = [c.expertIntermediateSize, c.hiddenSize]
            }
            shapes[f + "shared_experts.w1.weight"] = [c.expertIntermediateSize, c.hiddenSize]
            shapes[f + "shared_experts.w2.weight"] = [c.hiddenSize, c.expertIntermediateSize]
            shapes[f + "shared_experts.w3.weight"] = [c.expertIntermediateSize, c.hiddenSize]
        }
        // The DSpark draft stack, under `mtp.`. Each block is a decoder layer that never compresses,
        // routing over its own smaller set of experts; the first reads the main stack's hidden
        // states and the last carries the heads that turn a draft into tokens and a confidence.
        // A release that names no draft expert count routes its stages over the decoder's experts,
        // which is how V4 Pro 0813 reads.
        for stage in 0 ..< (c.dsparkBlockSize > 0 ? c.nextTokenPredictionLayers : 0) {
            let m = "mtp.\(stage)."
            let a = m + "attn."
            shapes[a + "wq_a.weight"] = [c.queryLoRARank, c.hiddenSize]
            shapes[a + "wq_b.weight"] = [c.headCount * c.headDimensions, c.queryLoRARank]
            shapes[a + "wkv.weight"] = [c.headDimensions, c.hiddenSize]
            shapes[a + "wo_a.weight"] = [c.outputGroups * c.outputLoRARank,
                                         c.headCount * c.headDimensions / c.outputGroups]
            shapes[a + "wo_b.weight"] = [c.hiddenSize, c.outputGroups * c.outputLoRARank]
            shapes[a + "q_norm.weight"] = [c.queryLoRARank]
            shapes[a + "kv_norm.weight"] = [c.headDimensions]
            shapes[a + "attn_sink"] = [c.headCount]
            shapes[m + "attn_norm.weight"] = [c.hiddenSize]
            shapes[m + "ffn_norm.weight"] = [c.hiddenSize]

            let mix = (2 + c.hyperConnectionCopies) * c.hyperConnectionCopies
            let mixWidth = c.hyperConnectionCopies * c.hiddenSize
            for part in ["attn", "ffn"] {
                shapes[m + "hc_\(part)_fn"] = [mix, mixWidth]
                shapes[m + "hc_\(part)_base"] = [mix]
                shapes[m + "hc_\(part)_scale"] = [3]
            }

            let f = m + "ffn."
            shapes[f + "gate.weight"] = [c.draftExpertCount, c.hiddenSize]
            shapes[f + "gate.bias"] = [c.draftExpertCount]
            if c.routerHasVisionBias { shapes[f + "gate.bias_vl"] = [c.draftExpertCount] }
            for expert in 0 ..< c.draftExpertCount {
                let e = f + "experts.\(expert)."
                shapes[e + "w1.weight"] = [c.expertIntermediateSize, c.hiddenSize]
                shapes[e + "w2.weight"] = [c.hiddenSize, c.expertIntermediateSize]
                shapes[e + "w3.weight"] = [c.expertIntermediateSize, c.hiddenSize]
            }
            shapes[f + "shared_experts.w1.weight"] = [c.expertIntermediateSize, c.hiddenSize]
            shapes[f + "shared_experts.w2.weight"] = [c.hiddenSize, c.expertIntermediateSize]
            shapes[f + "shared_experts.w3.weight"] = [c.expertIntermediateSize, c.hiddenSize]

            if stage == 0 {
                shapes[m + "main_proj.weight"] = [c.hiddenSize,
                                                  c.hiddenSize * c.dsparkTargetLayers.count]
                shapes[m + "main_norm.weight"] = [c.hiddenSize]
            }
            if stage == c.nextTokenPredictionLayers - 1 {
                shapes[m + "norm.weight"] = [c.hiddenSize]
                shapes[m + "markov_head.\(c.draftMarkovTableNames[0]).weight"] =
                    [c.vocabularySize, c.dsparkMarkovRank]
                shapes[m + "markov_head.\(c.draftMarkovTableNames[1]).weight"] =
                    [c.vocabularySize, c.dsparkMarkovRank]
                shapes[m + "confidence_head.proj.weight"] = [1, c.hiddenSize + c.dsparkMarkovRank]
                // Where the decoder collapses its copies through a learned head, so does the last
                // stage, with a head of its own.
                if c.collapsesThroughLearnedHead {
                    shapes[m + "hc_head_fn"] = [c.hyperConnectionCopies,
                                                c.hyperConnectionCopies * c.hiddenSize]
                    shapes[m + "hc_head_base"] = [c.hyperConnectionCopies]
                    shapes[m + "hc_head_scale"] = [1]
                }
            }
        }

        if let tower = c.vision {
            shapes["vision.patch_embed.proj.weight"] =
                [tower.hiddenSize, 3 * tower.patchSize * tower.patchSize]
            shapes["vision.patch_embed.proj.bias"] = [tower.hiddenSize]
            shapes["vision.norm.weight"] = [tower.hiddenSize]
            for block in 0 ..< tower.layerCount {
                let b = "vision.blocks.\(block)."
                shapes[b + "attn.wqkv.weight"] = [3 * tower.hiddenSize, tower.hiddenSize]
                shapes[b + "attn.wqkv.bias"] = [3 * tower.hiddenSize]
                shapes[b + "attn.wo.weight"] = [tower.hiddenSize, tower.hiddenSize]
                shapes[b + "attn.wo.bias"] = [tower.hiddenSize]
                shapes[b + "mlp.w1.weight"] = [2 * tower.intermediateSize, tower.hiddenSize]
                shapes[b + "mlp.w2.weight"] = [tower.hiddenSize, tower.intermediateSize]
                shapes[b + "norm1.weight"] = [tower.hiddenSize]
                shapes[b + "norm2.weight"] = [tower.hiddenSize]
            }
            let folded = tower.hiddenSize * tower.downsampleRatio * tower.downsampleRatio
            shapes["aligner.w1.weight"] = [tower.outputSize, folded]
            shapes["aligner.w1.bias"] = [tower.outputSize]
            shapes["aligner.w2.weight"] = [tower.outputSize, tower.outputSize]
            shapes["aligner.w2.bias"] = [tower.outputSize]
            // The span delimiters live in the DECODER's width, which is why the release keeps them
            // at the top level rather than inside the tower.
            for delimiter in ["image_start", "image_end", "image_newline"] {
                shapes[delimiter] = [tower.outputSize]
            }
        }
        return shapes
    }

    /// The shape a float parameter takes in the released checkpoint.
    ///
    /// @discussion The release is quantized, so a float weight is not stored as itself. Attention and
    /// shared-expert weights are fp8, one byte a value, keeping their shape. A routed expert is 4-bit
    /// packed two to a byte, so its last axis halves. This derivation is an assumption about the
    /// release's layout, and the structural test checks it against the observed shapes rather than
    /// trusting it.
    static func quantizedShape(of name: String, float: [Int]) -> [Int] {
        let packedFourBit = name.contains(".experts.") && !name.contains("shared_experts")
        guard packedFourBit, var shape = Optional(float), shape.count == 2 else { return float }
        shape[1] /= 2
        return shape
    }
}

/// The final collapse of the hyper-connection copies, before the shared norm and the head.
///
/// This is NOT the block connection at a smaller size: it predicts only the read weights — one
/// sigmoid gate per copy — with no write weights, no combination matrix, and no Sinkhorn, so its
/// `fn` is `[copies, copies * hidden]` where a block connection's is `[(2 + copies) * copies, ...]`.
/// The released `hc_head_fn` has exactly that shape, which is what caught the head being built as
/// the wrong class: the structural check compared DECLARED shapes against the release and passed,
/// while the module quietly built the bigger one.
final class NFKDeepSeekHyperHead: Module {
    @ParameterInfo(key: "fn") var projection: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    let epsilon: Float
    let normEpsilon: Float

    init(copies: Int, hiddenSize: Int, epsilon: Float, normEpsilon: Float) {
        self.epsilon = epsilon
        self.normEpsilon = normEpsilon
        _projection.wrappedValue = MLXArray.zeros([copies, copies * hiddenSize])
        _base.wrappedValue = MLXArray.zeros([copies])
        _scale.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    /// Collapses the copies into one stream, each weighted by its predicted gate.
    func reduce(_ x: MLXArray) -> MLXArray {
        let shape = x.shape                                  // [batch, length, copies, hidden]
        let flat = x.asType(.float32).reshaped(shape[0], shape[1], shape[2] * shape[3])
        let inverse = rsqrt((flat * flat).mean(axis: -1, keepDims: true) + normEpsilon)
        let mixes = matmul(flat, projection.transposed()) * inverse
        let read = sigmoid(mixes * scale + base) + epsilon
        return (read.expandedDimensions(axis: -1) * x.asType(.float32)).sum(axis: 2).asType(x.dtype)
    }
}

/// Compresses the key-value stream by pooling `compressRatio` consecutive positions into one.
///
/// Each position contributes a value (`wkv`) and a score (`wgate`); the scores are softmaxed across
/// the window and used to pool the values. `ape` is a learned per-slot bias, so a position's weight
/// depends on WHERE it sits in the window as well as on its content.
///
/// At ratio 4 the reference also compresses an OVERLAPPING window: the projections are twice as wide,
/// the second half pooling the window as given and the first half pooling it shifted back by one
/// window, so a boundary does not fall between two positions that belong together.
///
/// Prefill only. The reference's incremental decode keeps rolling state buffers, which a single
/// forward pass over a prompt never enters.
final class NFKDeepSeekCompressor: Module {
    @ModuleInfo(key: "wkv") var valueProjection: Linear
    /// Absent where a group holds one position: there is nothing for a gate to choose between, and
    /// the reference makes that case a plain projection.
    @ModuleInfo(key: "wgate") var scoreProjection: Linear?
    @ModuleInfo(key: "norm") var norm: RMSNorm
    /// Absent in V4.1, which pools without a learned per-slot bias.
    @ParameterInfo(key: "ape") var positionBias: MLXArray?

    let ratio: Int
    let headDimensions: Int
    let ropeHeadDimensions: Int
    let overlaps: Bool
    /// Whether the compressed positions are rotated here. The reference rotates them in the
    /// attention, after the indexer has read the unrotated form; V4's port folds it in instead.
    let appliesRotary: Bool
    let rope: NFKDeepSeekRotary

    init(_ c: NFKMLXDeepSeekConfiguration, ratio: Int, headDimensions: Int,
         appliesRotary: Bool = true) {
        self.ratio = ratio
        self.headDimensions = headDimensions
        self.appliesRotary = appliesRotary
        ropeHeadDimensions = c.ropeHeadDimensions
        overlaps = ratio == 4
        let width = (overlaps ? 2 : 1) * headDimensions
        _valueProjection.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        _scoreProjection.wrappedValue = ratio > 1 ? Linear(c.hiddenSize, width, bias: false) : nil
        _norm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: headDimensions, eps: c.rmsEpsilon)
        _positionBias.wrappedValue = c.compressorHasPositionBias
            ? MLXArray.zeros([ratio, width]) : nil
        rope = NFKDeepSeekRotary(width: c.ropeHeadDimensions, theta: c.compressRopeTheta,
                                 scaling: c.ropeScaling)
        super.init()
    }

    /// Pools `x` `[batch, length, hidden]` into `[batch, length / ratio, headDimensions]`.
    ///
    /// With a cache, a group may SPAN steps: the reference parks an unfinished one in `kv_state`
    /// and `score_state` with the unwritten slots scored at negative infinity, so they pool to
    /// nothing, and emits only on the step that completes the group.
    func callAsFunction(_ x: MLXArray, offset: Int = 0, cache: NFKMLXDeepSeekCache? = nil,
                        layer: Int = 0) -> MLXArray? {
        if let cache, ratio > 1 {
            return pooled(x, offset: offset, cache: cache, layer: layer)
        }
        let (batch, length) = (x.shape[0], x.shape[1])
        // One position a group is a projection, not a pooling: nothing competes and nothing is
        // dropped, so every position becomes its own compressed latent.
        guard let scoreProjection else { return norm(valueProjection(x)) }
        let windows = length / ratio
        guard windows > 0 else { return nil }
        let kept = windows * ratio

        let width = (overlaps ? 2 : 1) * headDimensions
        var values = valueProjection(x.asType(.float32))[0..., 0 ..< kept]
            .reshaped([batch, windows, ratio, width])
        var scores = scoreProjection(x.asType(.float32))[0..., 0 ..< kept]
            .reshaped([batch, windows, ratio, width])
        if let positionBias { scores = scores + positionBias }

        if overlaps {
            // The second half of the width pools this window; the first half pools the PREVIOUS one,
            // so a window boundary is covered from both sides. The first window has no predecessor,
            // which the reference fills with zero values and −inf scores so they contribute nothing.
            values = shifted(values, taking: 0 ..< headDimensions, fill: 0)
            scores = shifted(scores, taking: 0 ..< headDimensions, fill: -Float.greatestFiniteMagnitude / 4)
        }

        // Softmax runs ACROSS the window, so the positions in a window compete with one another.
        let pooled = (values * softmax(scores, axis: 2)).sum(axis: 2)
        let compressed = norm(pooled.asType(x.dtype))

        // The compressed positions carry a rotary too, at the window's own stride.
        return appliesRotary ? rotated(compressed, firstGroup: 0) : compressed
    }

    /// The pooling a cache carries across steps, one position at a time.
    ///
    /// A whole prompt still pools by the vectorized path above; what this adds is the tail. The
    /// slot a position occupies is `absolute % ratio`, and a group emits on the position where
    /// `(absolute + 1) % ratio` reaches zero, which is what makes an odd prompt leave one behind.
    private func pooled(_ x: MLXArray, offset: Int, cache: NFKMLXDeepSeekCache,
                        layer: Int) -> MLXArray? {
        guard let scoreProjection else { return norm(valueProjection(x)) }
        let (batch, length) = (x.shape[0], x.shape[1])
        let width = (overlaps ? 2 : 1) * headDimensions
        // An overlapping compressor parks TWO windows, as the release's `kv_state` does: the one
        // just pooled, whose first half the next group reads, ahead of the one being filled.
        let slots = (overlaps ? 2 : 1) * ratio
        let filling = overlaps ? ratio : 0
        // A pooling compressor runs in float32 against float32 weights, as the reference's does.
        let values = valueProjection(x.asType(.float32))
        let scores = scoreProjection(x.asType(.float32))

        // Written THROUGH, a slot at a time: `MLXArray` is a class and its subscript setter goes
        // into the existing buffer rather than rebinding. Anything holding this array holds what
        // the next step overwrites, which is why `NFKMLXDeepSeekCache.Snapshot` copies these two
        // and every other buffer it captures needs no copy.
        let parked = cache.pendingValues[layer] ?? MLXArray.zeros([batch, slots, width])
        // Negative infinity is what the reference initializes the scores to, so a slot nothing has
        // written contributes nothing to the softmax rather than an equal share.
        let parkedScores = cache.pendingScores[layer]
            ?? (MLXArray.zeros([batch, slots, width]) + nfkDeepSeekMaskedScore)
        var emitted = [MLXArray]()
        var firstGroup: Int?
        for step in 0 ..< length {
            let absolute = offset + step
            let slot = absolute % ratio
            parked[0..., filling + slot] = values[0..., step]
            // A position's bias is its slot's, so it is added as the position is parked.
            parkedScores[0..., filling + slot] = positionBias.map { scores[0..., step] + $0[slot] }
                ?? scores[0..., step]
            guard (absolute + 1) % ratio == 0 else { continue }
            firstGroup = firstGroup ?? (absolute + 1) / ratio - 1
            guard overlaps else {
                emitted.append((parked * softmax(parkedScores, axis: 1)).sum(axis: 1, keepDims: true))
                continue
            }
            // The previous window's first half and this window's second half pool together, and
            // this window then becomes the previous one.
            let d = headDimensions
            let pool = concatenated([parked[0..., 0 ..< ratio, 0 ..< d], parked[0..., ratio..., d...]],
                                    axis: 1)
            let weight = concatenated([parkedScores[0..., 0 ..< ratio, 0 ..< d],
                                       parkedScores[0..., ratio..., d...]], axis: 1)
            emitted.append((pool * softmax(weight, axis: 1)).sum(axis: 1, keepDims: true))
            parked[0..., 0 ..< ratio] = parked[0..., ratio...]
            parkedScores[0..., 0 ..< ratio] = parkedScores[0..., ratio...]
        }
        cache.pendingValues[layer] = parked
        cache.pendingScores[layer] = parkedScores
        guard !emitted.isEmpty, let firstGroup else { return nil }
        // Back to the stream's dtype before the norm, which is where the reference casts.
        let compressed = norm(concatenated(emitted, axis: 1).asType(x.dtype))
        return appliesRotary ? rotated(compressed, firstGroup: firstGroup) : compressed
    }

    /// Turns the trailing channels of consecutive groups starting at `firstGroup`, each at the
    /// position of its group's first token.
    private func rotated(_ compressed: MLXArray, firstGroup: Int) -> MLXArray {
        let (batch, groups) = (compressed.dim(0), compressed.dim(1))
        let turned = ropeHeadDimensions
        let head = compressed[.ellipsis, 0 ..< (headDimensions - turned)]
        let tail = compressed[.ellipsis, (headDimensions - turned)...]
            .reshaped([batch, 1, groups, turned])
        return concatenated([head, rope(tail, offset: firstGroup * ratio, stride: ratio)
                                .reshaped([batch, groups, turned])], axis: -1)
    }

    /// Builds the overlapping arrangement: window `w` takes its own second half and window `w-1`'s
    /// first half, giving `2 · ratio` contributors of `headDimensions` each.
    private func shifted(_ tensor: MLXArray, taking range: Range<Int>, fill: Float) -> MLXArray {
        let (batch, windows) = (tensor.shape[0], tensor.shape[1])
        let own = tensor[.ellipsis, headDimensions...]
        let previous = tensor[.ellipsis, range]
        let padding = MLXArray.zeros([batch, 1, ratio, headDimensions]) + fill
        let lagged = concatenated([padding, previous[0..., 0 ..< (windows - 1)]], axis: 1)
        return concatenated([lagged, own], axis: 2)
    }
}

/// Chooses which compressed positions a query should attend to.
///
/// A small side attention: the layer's low-rank query is projected into an index space and scored
/// against one shared key per compressed position, the scores rectified and combined by a learned
/// per-head weight. The highest `indexTopK` positions per query survive.
///
/// Where a release names a candidate source this is the SECOND of two levels. One layer scores with
/// its own weights and keeps the best blocks of positions; every indexer after it scores only
/// inside those blocks, so the whole stack pays for one coarse pass instead of one per layer.
///
/// The reference additionally applies a **Hadamard rotation** to both the query and the compressed
/// keys before simulating fp4 quantization. That rotation is orthogonal and SHARED by both sides of
/// the dot product, so it cancels: it exists to spread information across channels for quantization,
/// not to change the score. Omitting it and the quantization simulation together yields the
/// unquantized scores, which is the same ranking the reference approximates.
final class NFKDeepSeekIndexer: Module {
    @ModuleInfo(key: "wq_b") var queryUp: Linear
    @ModuleInfo(key: "weights_proj") var headWeights: Linear
    /// V4 compresses the hidden state a second time for its index keys.
    @ModuleInfo(key: "compressor") var compressor: NFKDeepSeekCompressor?
    /// V4.1 instead derives them from the layer compressor's latent, so only a layer that owns that
    /// compressor carries these; the others read the keys it publishes.
    @ModuleInfo(key: "wk") var keyProjection: Linear?
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm?

    let heads: Int
    let headDimensions: Int
    let ropeHeadDimensions: Int
    let topK: Int
    let ratio: Int
    let selectsCandidates: Bool
    let readsCandidates: Bool
    let candidateBlockSize: Int
    let candidateTopKBlocks: Int
    let rope: NFKDeepSeekRotary
    /// Whether keys and queries are rounded to fp4 as the release's code rounds them.
    let quantizesActivations: Bool
    let fp8BlockSize: Int

    convenience init(_ c: NFKMLXDeepSeekConfiguration, ratio: Int) {
        self.init(c, layer: c.compressRatios.firstIndex(of: ratio) ?? 0, ratio: ratio,
                  ownsKeys: true, derivesKeysFromCompressor: false, rope: nil)
    }

    init(_ c: NFKMLXDeepSeekConfiguration, layer: Int, ratio: Int, ownsKeys: Bool,
         derivesKeysFromCompressor: Bool, rope shared: NFKDeepSeekRotary? = nil) {
        heads = c.indexHeadCount
        headDimensions = c.indexHeadDimensions
        ropeHeadDimensions = c.ropeHeadDimensions
        topK = c.indexTopK
        self.ratio = ratio
        selectsCandidates = c.selectsCandidates(layer)
        readsCandidates = c.readsCandidates(layer)
        candidateBlockSize = c.candidateBlockSize
        candidateTopKBlocks = c.candidateTopKBlocks
        // The layer's own frequency table, where it has one: the indexer scores against positions
        // the attention rotated, so the two cannot disagree about the base.
        rope = shared ?? NFKDeepSeekRotary(width: c.ropeHeadDimensions, theta: c.ropeTheta,
                                           scaling: c.ropeScaling)
        _queryUp.wrappedValue = Linear(c.queryLoRARank, c.indexHeadCount * c.indexHeadDimensions,
                                       bias: false)
        _headWeights.wrappedValue = Linear(c.hiddenSize, c.indexHeadCount, bias: false)
        if derivesKeysFromCompressor {
            _compressor.wrappedValue = nil
            _keyProjection.wrappedValue = ownsKeys
                ? Linear(c.headDimensions, c.indexHeadDimensions, bias: false) : nil
            _keyNorm.wrappedValue = ownsKeys
                ? NFKDeepSeekRMSNorm(dimensions: c.indexHeadDimensions, eps: c.rmsEpsilon) : nil
        } else {
            _compressor.wrappedValue = NFKDeepSeekCompressor(c, ratio: ratio,
                                                             headDimensions: c.indexHeadDimensions)
        }
        quantizesActivations = c.quantizesActivations
        fp8BlockSize = c.fp8BlockSize
        super.init()
    }

    /// - Parameter lowRankQuery: the layer's `wq_a` output, which this shares rather than recomputing.
    /// - Parameter latent: the layer's compressed latent before the rotary, where it owns one.
    /// - Parameter positions: how many compressed positions the whole chunk completes.
    /// - Returns: `[batch, length, positions]`, true where a query may attend to that position.
    /// The low-rank query as the release's `linear()` hands it to `wq_b`'s fp8 GEMM.
    private func servedQuery(_ x: MLXArray) -> MLXArray {
        guard quantizesActivations else { return x }
        return NFKMLXDeepSeekQuantization.roundTripFP8(x, blockSize: fp8BlockSize)
    }

    /// Index keys and queries as the release's code leaves them: rounded to fp4 in blocks of
    /// `fp4BlockSize` with power-of-two scales, when the configuration asks.
    private func roundedIndex(_ x: MLXArray) -> MLXArray {
        guard quantizesActivations else { return x }
        return NFKMLXDeepSeekQuantization.roundTripFP4(
            x, blockSize: NFKMLXDeepSeekQuantization.fp4BlockSize)
    }

    func callAsFunction(_ x: MLXArray, lowRankQuery: MLXArray, latent: MLXArray?, positions: Int,
                        shared: NFKDeepSeekSharedAttention, cache: NFKMLXDeepSeekCache? = nil,
                        layer: Int = 0, offset: Int = 0) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        let already = offset / max(ratio, 1)
        // V4 compresses again for its keys; V4.1 reads the latent its layer already produced, which
        // is why the key projection only exists on a layer that owns that compressor.
        if let compressor {
            // V4's own compressor, which parks its groups under a key of its own in a cache.
            let state = NFKMLXDeepSeekCache.indexerState(layer)
            if let compressed = compressor(x, offset: offset, cache: cache, layer: state) {
                let keys = nfkDeepSeekHadamardRotated(compressed)
                shared.indexKeys = cache.map { $0.indexKeys(layer: state, appending: keys) } ?? keys
            } else if let cache {
                shared.indexKeys = cache.indexKeys(layer: state, appending: nil)
            }
        } else if let latent, let keyProjection, let keyNorm {
            let keys = keyNorm(keyProjection(latent)).expandedDimensions(axis: 1)
            let rotated = roundedIndex(
                turned(keys, offset: already * ratio, stride: ratio).squeezed(axis: 1))
            shared.indexKeys = cache.map { $0.indexKeys(layer: layer, appending: rotated) }
                ?? rotated
        } else if let cache, keyProjection != nil {
            // An owner that emitted nothing this step publishes what it holds anyway. The reference
            // does not, and the decode oracle corrects that; see its note.
            shared.indexKeys = cache.indexKeys(layer: layer, appending: nil)
        }
        let keys = shared.indexKeys![0..., 0 ..< positions]

        var queries = queryUp(servedQuery(lowRankQuery))
            .reshaped([batch, length, heads, headDimensions])
        queries = turned(queries.transposed(0, 2, 1, 3), offset: offset).transposed(0, 2, 1, 3)
        // V4's indexer rotates its query as it rotates its keys; V4.1's rotates neither.
        if compressor != nil { queries = nfkDeepSeekHadamardRotated(queries) }
        queries = roundedIndex(queries)

        // Every head scores every compressed position; a learned weight then combines the heads.
        let scores = einsum("bshd,btd->bsht", queries, keys)
        // The scale applies at float precision, as torch applies a Python scalar to a bf16 tensor.
        let weights = (headWeights(x).asType(.float32)
                       * (1 / sqrt(Float(headDimensions)) / sqrt(Float(heads)))).asType(x.dtype)
        // Summed in float32 and rounded once, as torch sums a bf16 tensor.
        var combined = (relu(scores) * weights.expandedDimensions(axis: -1)).asType(.float32)
            .sum(axis: 2).asType(scores.dtype)

        // A query may only see groups that are already complete at ITS OWN ABSOLUTE position.
        let reached = MLXArray((0 ..< length).map { Int32((offset + $0 + 1) / ratio) })
            .reshaped([length, 1])
        let index = MLXArray((0 ..< positions).map(Int32.init)).reshaped([1, positions])
        combined = MLX.where(index .< reached, combined,
                             MLXArray(nfkDeepSeekMaskedScore).asType(combined.dtype))

        cache?.selectionScores[layer] = combined
        if selectsCandidates {
            shared.candidates = candidateBlocks(combined, offset: offset)
        } else if readsCandidates, let candidates = shared.candidates {
            combined = MLX.where(candidates, combined,
                                 MLXArray(nfkDeepSeekMaskedScore).asType(combined.dtype))
        }
        return best(combined, count: min(topK, positions))
    }

    /// The `count` highest-scoring entries of each row, as a mask.
    ///
    /// The ranking is scattered back rather than turned into a threshold, because these scores TIE:
    /// a position every head rectifies to zero scores exactly zero, and thresholding at the k-th
    /// value would keep every one of the tied entries instead of k of them. The masked floor is
    /// excluded outright, so a row with fewer reachable entries than `count` keeps only those.
    private func best(_ scores: MLXArray, count: Int) -> MLXArray {
        let live = scores .> MLXArray(nfkDeepSeekMaskedScore / 2)
        guard count > 0 else { return MLXArray.zeros(scores.shape, type: Bool.self) }
        let ranked = argPartition(-scores, kth: count - 1, axis: -1)[.ellipsis, 0 ..< count]
        let kept = putAlong(MLXArray.zeros(scores.shape, type: Int32.self), ranked,
                            values: MLXArray.ones(ranked.shape, type: Int32.self), axis: -1)
        return (kept .> 0) .&& live
    }

    /// Level one of the two-level selection: the best blocks of compressed positions per query.
    ///
    /// The block holding a query's newest position is pinned in whatever it scores. It is only
    /// partly filled, so it holds the most recent tokens and would otherwise be outscored by an
    /// older, complete block.
    private func candidateBlocks(_ scores: MLXArray, offset: Int = 0) -> MLXArray {
        let (batch, length, positions) = (scores.shape[0], scores.shape[1], scores.shape[2])
        let size = candidateBlockSize
        let blocks = (positions + size - 1) / size
        var wide = scores
        if blocks * size > positions {
            let pad = MLXArray.full([batch, length, blocks * size - positions],
                                    values: MLXArray(nfkDeepSeekMaskedScore))
            wide = concatenated([scores, pad], axis: -1)
        }
        // A block scores as its best position.
        var perBlock = wide.reshaped([batch, length, blocks, size]).max(axis: -1)
        // Written out rather than divided in MLX: a query whose newest group is not complete has no
        // block to pin, which is the `-1` that integer division in the reference produces and that
        // MLX's own division would not.
        let newest = MLXArray((0 ..< length).map { position -> Int32 in
            let reached = (offset + position + 1) / ratio
            return reached == 0 ? -1 : Int32((reached - 1) / size)
        }).reshaped([length, 1])
        let index = MLXArray((0 ..< blocks).map(Int32.init)).reshaped([1, blocks])
        perBlock = MLX.where(index .== newest, MLXArray(-nfkDeepSeekMaskedScore), perBlock)

        let kept = best(perBlock, count: min(candidateTopKBlocks, blocks))
        return repeated(kept, count: size, axis: -1)[.ellipsis, 0 ..< positions]
    }

    /// Rotates the trailing channels, at the stride a compressed position advances by.
    private func turned(_ x: MLXArray, offset: Int = 0, stride: Int = 1) -> MLXArray {
        let width = x.shape[x.shape.count - 1]
        let turned = min(ropeHeadDimensions, width)
        return concatenated([x[.ellipsis, 0 ..< (width - turned)],
                             rope(x[.ellipsis, (width - turned)...], offset: offset,
                                  stride: stride)],
                            axis: -1)
    }
}

/// Hyper-Connections: the residual stream is `hcMultiplier` parallel copies rather than one.
///
/// A block reduces the copies to a single stream before its attention or feed-forward (`reduce`) and
/// expands the result back afterwards (`expand`), with the mixing weights PREDICTED per position from
/// the copies themselves. `hc_*_fn` projects the flattened, RMS-normalized copies to
/// `(2 + hc) · hc` numbers, which split into a read weight per copy, a write weight per copy, and a
/// copy-to-copy combination matrix; the matrix is softmaxed and then Sinkhorn-normalized so it is
/// close to doubly stochastic, which keeps the copies from collapsing into one another.
///
/// This is what the top-level `hc_head_*` parameters collapse at the end of the stack. An earlier note
/// here called them a hash-clustering head; that was wrong, and it mattered: the residual stream's
/// rank is different from an ordinary decoder's.
final class NFKDeepSeekHyperConnection: Module {
    @ParameterInfo(key: "fn") var projection: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    let copies: Int
    let epsilon: Float
    let normEpsilon: Float
    let iterations: Int

    init(copies: Int, hiddenSize: Int, iterations: Int, epsilon: Float, normEpsilon: Float) {
        self.copies = copies
        self.iterations = iterations
        self.epsilon = epsilon
        self.normEpsilon = normEpsilon
        let mix = (2 + copies) * copies
        _projection.wrappedValue = MLXArray.zeros([mix, copies * hiddenSize])
        _base.wrappedValue = MLXArray.zeros([mix])
        _scale.wrappedValue = MLXArray.ones([3])
        super.init()
    }

    /// Splits the predicted mixes into read weights, write weights, and the combination matrix.
    func weights(_ x: MLXArray) -> (read: MLXArray, write: MLXArray, combine: MLXArray) {
        let shape = x.shape                                  // [batch, length, copies, hidden]
        let flat = x.asType(.float32).reshaped(shape[0], shape[1], shape[2] * shape[3])
        let inverse = rsqrt((flat * flat).mean(axis: -1, keepDims: true) + normEpsilon)
        let mixes = matmul(flat, projection.transposed()) * inverse

        let read = sigmoid(mixes[0..., 0..., 0 ..< copies] * scale[0]
                           + base[0 ..< copies]) + epsilon
        let write = 2 * sigmoid(mixes[0..., 0..., copies ..< (2 * copies)] * scale[1]
                                + base[copies ..< (2 * copies)])

        var combine = mixes[0..., 0..., (2 * copies)...] * scale[2] + base[(2 * copies)...]
        combine = combine.reshaped(shape[0], shape[1], copies, copies)
        combine = softmax(combine, axis: -1) + epsilon
        combine = combine / (combine.sum(axis: -2, keepDims: true) + epsilon)
        // Sinkhorn: alternate row and column normalization so the matrix approaches doubly stochastic.
        for _ in 1 ..< max(iterations, 1) {
            combine = combine / (combine.sum(axis: -1, keepDims: true) + epsilon)
            combine = combine / (combine.sum(axis: -2, keepDims: true) + epsilon)
        }
        return (read, write, combine)
    }

    /// Collapses the copies into one stream, weighting each by its read weight.
    func reduce(_ x: MLXArray, read: MLXArray) -> MLXArray {
        (read.expandedDimensions(axis: -1) * x.asType(.float32)).sum(axis: 2).asType(x.dtype)
    }

    /// Writes a single stream back across the copies, adding the combined previous copies.
    func expand(_ x: MLXArray, residual: MLXArray, write: MLXArray, combine: MLXArray) -> MLXArray {
        let written = write.expandedDimensions(axis: -1)
            * x.asType(.float32).expandedDimensions(axis: -2)
        // Copy j receives the sum over i of combine[i, j] times residual copy i.
        let mixed = (combine.expandedDimensions(axis: -1)
                     * residual.asType(.float32).expandedDimensions(axis: -2)).sum(axis: 2)
        return (written + mixed).asType(x.dtype)
    }
}
