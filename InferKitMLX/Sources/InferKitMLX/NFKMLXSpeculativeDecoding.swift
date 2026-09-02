//
//  NFKMLXSpeculativeDecoding.swift
//  InferKitMLX
//
//  Draft-and-verify generation: a small model proposes several tokens, the large model scores them
//  in one cached forward pass, and the rejected tail is rolled back.
//

import Foundation
import MLX

public extension NFKMLXLanguageNet {

    /// Generates with `draft` proposing ``NFKMLXGenerationOptions/draftTokens`` tokens per round,
    /// which this model verifies in a single forward pass.
    ///
    /// @discussion Decoding is bound by memory traffic: every token reads every weight once, so a
    /// step's cost barely depends on how many positions it scores. A small draft model proposes a
    /// run of tokens, this model scores the whole run in one pass, and the leading tokens that agree
    /// are kept. The output is the SAME sequence plain decoding produces. At temperature 0 every kept
    /// token is this model's own argmax given the tokens before it, so greedy speculative decoding
    /// is token-for-token identical to ``generate(prompt:options:promptCache:onToken:)``. Above 0 the
    /// acceptance test is the standard rejection scheme, which leaves the distribution over outputs
    /// exactly this model's own; the samples differ from a plain run under the same seed because the
    /// random stream is consumed differently.
    ///
    /// The draft must share this model's vocabulary — a token id has to mean the same thing to both.
    /// How much time is saved depends on how often the draft agrees: a draft from the same family a
    /// few sizes down typically has most of its proposals accepted.
    func generate(prompt: [Int], options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                  draft: NFKMLXLanguageNet, promptCache: NFKMLXPromptCache? = nil,
                  onToken: ((Int) -> Bool)? = nil) -> [Int] {
        var report = NFKMLXSpeculativeReport()
        return generate(prompt: prompt, options: options, draft: draft, promptCache: promptCache,
                        report: &report, onToken: onToken)
    }

    /// The same run, also filling `report` with how many proposals were accepted.
    func generate(prompt: [Int], options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                  draft: NFKMLXLanguageNet, promptCache: NFKMLXPromptCache? = nil,
                  report: inout NFKMLXSpeculativeReport,
                  onToken: ((Int) -> Bool)? = nil) -> [Int] {
        guard !prompt.isEmpty, options.maxTokens > 0 else { return [] }
        precondition(draft.configuration.vocabularySize == configuration.vocabularySize,
                     "a draft model must share the target's vocabulary")
        if let seed = options.seed { MLXRandom.seed(seed) }
        let proposalsPerRound = Swift.max(options.draftTokens, 1)

        let (cache, tail) = startingCache(for: prompt, options: options, promptCache: promptCache)
        let promptLogits = prefill(tail, cache: cache, chunkSize: options.prefillChunkSize)
        promptCache?.record(tail)

        let draftCache = NFKMLXKeyValueCache(layerCount: draft.configuration.layerCount,
                                             window: options.contextWindow)
        let draftPrompt = draft.prefill(prompt, cache: draftCache, chunkSize: options.prefillChunkSize)
        eval(draftPrompt)                            // the draft's own prompt logits are never read

        var produced = [Int]()
        // Appends a token to the output, or returns false when generation must stop here.
        func emit(_ token: Int) -> Bool {
            guard produced.count < options.maxTokens, !options.stopTokens.contains(token) else {
                return false
            }
            produced.append(token)
            return onToken?(token) ?? true
        }

        // The first token needs no draft: the prompt's own logits decide it.
        var next = NFKMLXLanguageNet.sample(promptLogits[0, -1], options: options)
        guard emit(next) else { return produced }

        while produced.count < options.maxTokens {
            // The draft continues from `next`, feeding each proposal back to itself.
            var proposals = [Int]()
            var proposalDistributions = [MLXArray]()
            var fed = next
            for _ in 0 ..< proposalsPerRound {
                let logits = draft(MLXArray([Int32(fed)]).reshaped([1, 1]), cache: draftCache)[0, -1]
                if options.temperature > 0 {
                    let distribution = NFKMLXLanguageNet.probabilities(of: logits, options: options)
                    proposalDistributions.append(distribution)
                    fed = MLXRandom.categorical(log(distribution)).item(Int.self)
                } else {
                    fed = logits.argMax().item(Int.self)
                }
                proposals.append(fed)
            }

            // One verifying pass over `next` and every proposal. Row j predicts the token that
            // follows the j-th input, so row j is what proposal j is judged against and the last
            // row predicts what follows the final proposal.
            let batch = [next] + proposals
            let rows = self(MLXArray(batch.map { Int32($0) }).reshaped([1, batch.count]), cache: cache)[0]
            promptCache?.record(batch)

            let (accepted, following) = options.temperature > 0
                ? NFKMLXLanguageNet.verifyBySampling(rows: rows, proposals: proposals,
                                                     proposalDistributions: proposalDistributions,
                                                     options: options)
                : NFKMLXLanguageNet.verifyGreedily(rows: rows, proposals: proposals)
            report.rounds += 1
            report.proposed += proposals.count
            report.accepted += accepted

            // Both caches keep `next` and the accepted proposals and drop the rest. The draft fed
            // every proposal but the last to itself, so it is one short when everything was accepted.
            let rejected = proposals.count - accepted
            // A prompt cache owns `cache`, so the rollback goes through it to keep its record in step.
            if let promptCache {
                promptCache.rollback(by: rejected)
            } else {
                cache.rollback(by: rejected)
            }
            if rejected == 0 {
                let last = draft(MLXArray([Int32(proposals[proposals.count - 1])]).reshaped([1, 1]),
                                 cache: draftCache)
                eval(last)
            } else {
                draftCache.rollback(by: rejected - 1)
            }

            for token in proposals.prefix(accepted) {
                guard emit(token) else { return produced }
            }
            guard emit(following) else { return produced }
            next = following
        }
        return produced
    }

    /// Greedy verification: proposals are kept while they equal the target's argmax, and the token
    /// after the kept run is the target's argmax there.
    static func verifyGreedily(rows: MLXArray, proposals: [Int]) -> (accepted: Int, following: Int) {
        let best = rows.argMax(axis: -1).asArray(Int32.self).map { Int($0) }
        var accepted = 0
        while accepted < proposals.count && proposals[accepted] == best[accepted] {
            accepted += 1
        }
        return (accepted, best[accepted])
    }

    /// The rejection-sampling verification: proposal `j` is kept with probability
    /// `min(1, p(x) / q(x))` under the target's distribution `p` and the draft's `q`, and the first
    /// rejected position samples from the normalized positive part of `p − q`, which is what keeps
    /// the outputs distributed exactly as the target alone would produce them.
    static func verifyBySampling(rows: MLXArray, proposals: [Int], proposalDistributions: [MLXArray],
                                 options: NFKMLXGenerationOptions) -> (accepted: Int, following: Int) {
        let target = probabilities(of: rows, options: options)
        eval(target)
        for (index, proposal) in proposals.enumerated() {
            let targetRow = target[index]
            let draftRow = proposalDistributions[index]
            let targetMass = targetRow[proposal].item(Float.self)
            let draftMass = draftRow[proposal].item(Float.self)
            let threshold = draftMass > 0 ? Swift.min(1, targetMass / draftMass) : 1
            let draw = MLXRandom.uniform(0 ..< 1, [1]).item(Float.self)
            if draw < threshold { continue }

            let residual = maximum(targetRow - draftRow, 0)
            let total = residual.sum().item(Float.self)
            let source = total > 0 ? residual / total : targetRow
            return (index, MLXRandom.categorical(log(source)).item(Int.self))
        }
        return (proposals.count, MLXRandom.categorical(log(target[proposals.count])).item(Int.self))
    }
}

/// What a speculative run did: how many rounds it took and how many proposals survived.
public struct NFKMLXSpeculativeReport: Sendable {
    /// Verification passes run.
    public var rounds = 0
    /// Draft tokens proposed in total.
    public var proposed = 0
    /// Draft tokens the target kept.
    public var accepted = 0
    public init() {}

    /// The fraction of proposals kept, which is what the speedup follows.
    public var acceptanceRate: Double { proposed > 0 ? Double(accepted) / Double(proposed) : 0 }
}
