//
//  NFKMLXAudioConversation.swift
//  InferKitMLX
//
//  Generation through a kept conversation's cache for the speech-to-text models whose decoder is the dense
//  Granite decoder (Voxtral, Granite Speech): the clip's projected frames spliced into the prompt, the prompt
//  prefilled from what the conversation already holds, and each new token run alone.
//

import Foundation
import InferKit
import MLX

enum NFKMLXAudioConversation {
    /// Greedy generation of up to `limit` tokens through the conversation `request` names or `tokens` continues,
    /// or a cache kept for no later request where it turns reuse off.
    ///
    /// @discussion The conversation is kept under `tokens` with the clip's positions keyed by `digest` and their
    /// place in the clip, so a prompt about the same clip continues past it and one about another clip
    /// continues nothing past its first position. The cache rolls back to what it shares with that key; the rest
    /// of the prompt prefills with the clip's frames from the first position it holds, and each new token then
    /// runs alone. A run that fails discards the conversation. Runs under the backend's generation lock. Answers
    /// the tokens produced and how many prompt positions the cache already held.
    static func run(decoder: NFKMLXGraniteTextNet, tokens: [Int], audio: MLXArray, audioToken: Int, digest: UInt64,
                    conversations: NFKMLXConversationKeeper<NFKMLXPromptCache>, request: NFKInferenceRequest,
                    limit: Int, stop: Int, afforded: () -> Int) throws -> (produced: [Int], cached: Int) {
        let key = try NFKMLXVisionChat.mediaKey(inputIds: tokens, isMedia: { $0 == audioToken },
                                                media: [(digest, tokens.filter { $0 == audioToken }.count)])
        let layerCount = decoder.config.layerCount
        let (name, state) = conversations.conversation(
            named: NFKMLXConversation.name(of: request), reuses: NFKMLXConversation.reuses(request), prompt: key,
            signature: NFKMLXHybridConversation.optionsSignature(window: nil, quantization: nil, adapterStrengths: nil),
            usable: { $0.layerCount == layerCount && $0.window == nil && $0.quantization == nil },
            make: { NFKMLXPromptCache(layerCount: layerCount) },
            read: { url, record in
                let restored = try NFKMLXPromptCache.load(from: url)
                return restored.tokens == record.tokens && restored.layerCount == layerCount ? restored : nil
            })
        var failed = true
        defer { conversations.finish(name, failed: failed, afforded: afforded(), spills: true) }

        let start = state.align(to: key)
        let fresh = Array(tokens[start...])
        let embeddings = NFKMLXGraniteSpeechNet.fusedEmbeddings(
            decoder, tokens: MLXArray(fresh.map(Int32.init)).reshaped([1, fresh.count]),
            audio: fresh.contains(audioToken) ? audio : nil, audioToken: audioToken,
            audioOffset: tokens[..<start].filter { $0 == audioToken }.count)
        var hidden = decoder.hiddenStates(fromEmbeddings: embeddings, cache: state.cache)
        state.record(Array(key[start...]))
        var produced = [Int]()
        for step in 0 ..< limit {
            let next = decoder.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...]).reshaped([-1])
                .argMax().item(Int.self)
            if let failure = NFKMLXErrorScope.failure { throw failure }
            if next == stop { break }
            produced.append(next)
            guard step + 1 < limit else { break }
            hidden = decoder.hiddenStates(
                fromEmbeddings: decoder.tokenEmbeddings(MLXArray([Int32(next)]).reshaped([1, 1])), cache: state.cache)
            state.record([next])
        }
        if let failure = NFKMLXErrorScope.failure { throw failure }
        failed = false
        return (produced, start)
    }
}
