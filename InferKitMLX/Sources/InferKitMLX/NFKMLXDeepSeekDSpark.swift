//
//  NFKMLXDeepSeekDSpark.swift
//  InferKitMLX
//
//  DeepSeek V4.1's speculative-decoding draft stack, stored under the checkpoint's `mtp.` namespace.
//
//  A draft stage is a decoder block that never compresses, routing over its own smaller set of
//  experts. Two things are the stack's own. Its attention reads the MAIN stack's key-value over the
//  sliding window and the whole drafted block for itself, with no causal mask between the drafts,
//  so the block is one parallel proposal. Its head then walks that block one position at a time, so
//  each drafted token biases the next through a Markov embedding, and scores the whole draft.
//
//  The first stage reads the main stack's hidden states through a projection; the last carries the
//  norm, the Markov head and the confidence head.
//

import Foundation
import MLX
import MLXNN

/// The projection that hands the main stack's hidden states to the first draft stage.
///
/// @discussion `dsparkTargetLayers` names the decoder layers whose ATTENTION INPUT is read, not
/// their output, and the states are concatenated before the projection, so its input is as many
/// times the hidden width as there are target layers.
public final class NFKMLXDeepSeekDraftInput: Module {
    @ModuleInfo(key: "main_proj") var projection: Linear
    @ModuleInfo(key: "main_norm") var norm: RMSNorm

    let blockSize: Int
    let noiseToken: Int

    init(_ c: NFKMLXDeepSeekConfiguration) {
        blockSize = c.dsparkBlockSize
        noiseToken = c.dsparkNoiseToken
        _projection.wrappedValue = Linear(c.hiddenSize * c.dsparkTargetLayers.count, c.hiddenSize,
                                          bias: false)
        _norm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    /// `mainHidden` is the target layers' states concatenated on the last axis.
    public func mainState(_ mainHidden: MLXArray) -> MLXArray { norm(projection(mainHidden)) }

    /// The ids a draft block embeds: the committed token, then a noise token per position it is
    /// about to propose.
    public func draftTokens(continuing token: MLXArray) -> MLXArray {
        NFKMLXDeepSeekDraftStack.draftTokens(continuing: token, blockSize: blockSize,
                                             noiseToken: noiseToken)
    }
}

/// The Markov head: a second, much narrower embedding that biases the logits of each drafted
/// position by the token drafted immediately before it.
public final class NFKMLXDeepSeekMarkovHead: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "head") var head: Linear

    init(vocabularySize: Int, rank: Int) {
        _embed.wrappedValue = Embedding(embeddingCount: vocabularySize, dimensions: rank)
        _head.wrappedValue = Linear(rank, vocabularySize, bias: false)
        super.init()
    }

    /// `tokens` `[batch]` → the logit bias and the embedding the confidence head also reads.
    ///
    /// @discussion The release builds this head as its output head, float32 weight and float32
    /// input, so the bias comes out float32 whatever the decoder computes in.
    public func callAsFunction(_ tokens: MLXArray) -> (bias: MLXArray, embedded: MLXArray) {
        let embedded = embed(tokens)
        return (head(embedded.asType(.float32)), embedded)
    }
}

/// The confidence head, which scores a draft from the stage's hidden state and the Markov embedding.
///
/// @discussion The projection is read at float32 even where the checkpoint stores it in bfloat16,
/// because the score decides whether a draft is accepted and the reference keeps that decision out
/// of the model's own precision.
public final class NFKMLXDeepSeekConfidenceHead: Module {
    @ModuleInfo(key: "proj") var projection: Linear

    init(hiddenSize: Int, rank: Int) {
        _projection.wrappedValue = Linear(hiddenSize + rank, 1, bias: false)
        super.init()
    }

    /// `hidden` `[batch, block, hidden]` with `markov` `[batch, block, rank]` → `[batch, block]`.
    public func callAsFunction(_ hidden: MLXArray, markov: MLXArray) -> MLXArray {
        let joined = concatenated([hidden, markov], axis: -1).asType(.float32)
        return projection(joined).squeezed(axis: -1)
    }
}

/// One stage of the draft stack, stored under `mtp.<stage>`.
///
/// A stage is a decoder block, which is why this subclasses one rather than wrapping it: the
/// checkpoint keys its attention, its feed-forward and its hyper-connections exactly as a decoder
/// layer's. What a stage adds depends on where it sits. The first reads the main stack; the last
/// carries the head that turns the block into tokens.
final class NFKDeepSeekDraftStage: NFKDeepSeekBlock {
    @ModuleInfo(key: "main_proj") var mainProjection: Linear?
    @ModuleInfo(key: "main_norm") var mainNorm: RMSNorm?
    @ModuleInfo(key: "norm") var norm: RMSNorm?
    @ModuleInfo(key: "markov_head") var markov: NFKMLXDeepSeekMarkovHead?
    @ModuleInfo(key: "confidence_head") var confidence: NFKMLXDeepSeekConfidenceHead?
    /// The last stage's own learned collapse, where the decoder collapses through one (V4 Pro 0813).
    @ModuleInfo(key: "hc_head") var headConnection: NFKDeepSeekHyperHead?

    init(_ c: NFKMLXDeepSeekConfiguration, stage: Int) {
        super.init(c.draftShaped, layer: stage)
        if stage == 0 {
            _mainProjection.wrappedValue = Linear(c.hiddenSize * c.dsparkTargetLayers.count,
                                                  c.hiddenSize, bias: false)
            _mainNorm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        }
        if stage == c.nextTokenPredictionLayers - 1 {
            _norm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
            _markov.wrappedValue = NFKMLXDeepSeekMarkovHead(vocabularySize: c.vocabularySize,
                                                            rank: c.dsparkMarkovRank)
            _confidence.wrappedValue = NFKMLXDeepSeekConfidenceHead(hiddenSize: c.hiddenSize,
                                                                    rank: c.dsparkMarkovRank)
            if c.collapsesThroughLearnedHead {
                _headConnection.wrappedValue = NFKDeepSeekHyperHead(
                    copies: c.hyperConnectionCopies, hiddenSize: c.hiddenSize,
                    epsilon: c.hyperConnectionEpsilon, normEpsilon: c.rmsEpsilon)
            }
        }
    }
}

/// DSpark: the draft stack that proposes several tokens at once for the decoder to check.
///
/// @discussion One call proposes `dsparkBlockSize` tokens following a committed one. The stack
/// reads what the decoder already computed — the mean over the hyper-connection copies of the
/// stream entering each `dsparkTargetLayers` attention — so it costs a few narrow layers rather
/// than another pass of the decoder.
///
/// The reference splits the same work over a prefill that seeds each stage's sliding-window cache
/// and a step that drafts, because it decodes one token at a time. This takes the main states for
/// every committed position at once and slices the window out of them, which is the same
/// arithmetic without the ring buffer.
public final class NFKMLXDeepSeekDraftStack: Module {
    @ModuleInfo(key: "mtp") var stages: [NFKDeepSeekDraftStage]

    let configuration: NFKMLXDeepSeekConfiguration

    init(_ c: NFKMLXDeepSeekConfiguration) {
        configuration = c
        _stages.wrappedValue = (0 ..< c.nextTokenPredictionLayers).map {
            NFKDeepSeekDraftStage(c, stage: $0)
        }
        super.init()
    }

    /// The ids a draft block embeds: the committed token, then a noise token per position it is
    /// about to propose.
    public static func draftTokens(continuing token: MLXArray, blockSize: Int,
                                   noiseToken: Int) -> MLXArray {
        let batch = token.dim(0)
        let ids = MLXArray.full([batch, blockSize], values: MLXArray(Int32(noiseToken)))
        ids[0..., 0] = token.reshaped([batch]).asType(ids.dtype)
        return ids
    }

    /// The main states as the first stage reads them: projected and normed.
    ///
    /// @discussion `main_proj` is built fp8 in the release, so where the configuration serves its
    /// GEMMs its input is rounded as `linear()` rounds it. Anything that rebuilds the draft pass by
    /// hand goes through here, or it computes a main state the stack never sees.
    public func projectedMainState(_ mainStates: MLXArray) -> MLXArray {
        let c = configuration
        let first = stages[0]
        let input = c.quantizesActivations
            ? NFKMLXDeepSeekQuantization.roundTripFP8(mainStates, blockSize: c.fp8BlockSize)
            : mainStates
        return first.mainNorm!(first.mainProjection!(input))
    }

    /// Proposes the next `dsparkBlockSize` tokens.
    ///
    /// - Parameter token: `[batch]`, the token the decoder has just committed.
    /// - Parameter mainStates: `[batch, positions, hidden · targets]`, the decoder's target-layer
    ///   states at every committed position, the committed token's included. The drafted positions
    ///   follow them, so `positions` is also where the first draft sits.
    /// - Parameter decoder: the decoder whose embedding and output head the stack shares, which is
    ///   what the reference does by aliasing the modules rather than copying them.
    /// - Parameter sampler: how a drafted position picks its token from its biased logits. The
    ///   default is the argmax the reference's own sampler reduces to at temperature zero.
    /// - Returns: the committed token followed by the proposals, the logits each was drawn from
    ///   with the Markov bias already added, and a confidence per drafted position.
    public func propose(continuing token: MLXArray, mainStates: MLXArray,
                        through decoder: NFKMLXDeepSeekNet,
                        sampler: (MLXArray) -> MLXArray = { argMax($0, axis: -1) })
        -> (tokens: MLXArray, logits: MLXArray, confidence: MLXArray) {
        let c = configuration
        let last = stages[stages.count - 1]
        let mainState = projectedMainState(mainStates)
        let context = NFKDeepSeekDraftContext(mainState: mainState, offset: mainStates.shape[1])

        let ids = Self.draftTokens(continuing: token, blockSize: c.dsparkBlockSize,
                                   noiseToken: c.dsparkNoiseToken)
        let (batch, block) = (ids.shape[0], ids.shape[1])
        var hidden = repeated(decoder.embed(ids).expandedDimensions(axis: 2),
                              count: c.hyperConnectionCopies, axis: 2)
        // Every stage starts from the one-hot read, exactly as the decoder's first block does.
        var read = concatenated(
            [MLXArray.ones([batch, block, 1]),
             MLXArray.zeros([batch, block, c.hyperConnectionCopies - 1])], axis: -1)
        for stage in stages {
            (hidden, read) = stage(hidden, mask: nil, tokens: nil, incoming: read, drafting: context)
        }

        // The last stage collapses its copies as the decoder does: through its own learned head, or
        // by the read weight its last block predicted.
        let collapsed = last.headConnection.map { $0.reduce(hidden) }
            ?? (read.expandedDimensions(axis: -1) * hidden.asType(.float32))
                .sum(axis: 2).asType(hidden.dtype)
        let logits = decoder.head(last.norm!(collapsed))

        // The walk. Each position's logits are biased by the token chosen for the position before
        // it, so the block is drawn in order even though the attention that produced it was not.
        var drafted = [token.reshaped([batch])]
        var embedded = [MLXArray]()
        for position in 0 ..< block {
            let (bias, embedding) = last.markov!(drafted[position])
            let biased = logits[0..., position] + bias
            logits[0..., position] = biased
            embedded.append(embedding)
            drafted.append(sampler(biased).reshaped([batch]))
        }
        let confidence = last.confidence!(collapsed, markov: stacked(embedded, axis: 1))
        return (stacked(drafted, axis: 1), logits, confidence)
    }
}

extension NFKMLXDeepSeek {

    /// The draft stack a release carries, or nil where it carries none.
    ///
    /// @discussion V4.1 names the draft stack's own expert count. V4 Pro 0813 leaves
    /// `dspark_n_routed_experts` out, which its code reads as "route over the decoder's experts",
    /// and its checkpoint carries exactly that: three stages of 384 experts each.
    public static func makeDraftStack(_ c: NFKMLXDeepSeekConfiguration)
        -> NFKMLXDeepSeekDraftStack? {
        guard c.dsparkBlockSize > 0, c.nextTokenPredictionLayers > 0, !c.dsparkTargetLayers.isEmpty else {
            return nil
        }
        return NFKMLXDeepSeekDraftStack(c)
    }

    /// The draft stack's standalone pieces, for a caller that wants one without the blocks.
    public static func makeDraftHeads(_ c: NFKMLXDeepSeekConfiguration)
        -> (input: NFKMLXDeepSeekDraftInput, markov: NFKMLXDeepSeekMarkovHead,
            confidence: NFKMLXDeepSeekConfidenceHead)? {
        guard c.dsparkBlockSize > 0, !c.dsparkTargetLayers.isEmpty else { return nil }
        return (NFKMLXDeepSeekDraftInput(c),
                NFKMLXDeepSeekMarkovHead(vocabularySize: c.vocabularySize, rank: c.dsparkMarkovRank),
                NFKMLXDeepSeekConfidenceHead(hiddenSize: c.hiddenSize, rank: c.dsparkMarkovRank))
    }
}
