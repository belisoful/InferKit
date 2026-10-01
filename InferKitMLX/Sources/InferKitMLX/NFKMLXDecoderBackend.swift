//
//  NFKMLXDecoderBackend.swift
//  InferKitMLX
//
//  A text-generation backend over a decoder that carries no key-value cache: generation re-runs the
//  growing sequence each step. Qwen4-Exp and the Qwen3.5 hybrid decoder run through it. A message list
//  renders through the release's own chat template where the release ships one, and a raw prompt is
//  encoded as given.
//

import Foundation
import InferKit
import MLX
import MLXRandom

/// Holds the decoder's forward and the tokenizer across the async job boundary. `MLXArray` and the
/// tokenizer are not `Sendable`, so the crossing is made explicit, as the other language backends do.
final class NFKDecoderBackendHolder: @unchecked Sendable {
    let logits: (MLXArray) -> MLXArray
    let tokenizer: NFKTokenizer
    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer) {
        self.logits = logits
        self.tokenizer = tokenizer
    }
}

/// A prefill-only text-generation backend over a decoder whose forward maps token ids to logits.
///
/// @discussion Reads a prompt under `NFKInputPrompt` or a message list under `NFKInputMessages`, and
/// returns the reply under `NFKOutputText` with its token counts under `NFKOutputUsage`. A message
/// list renders through the release's chat template, which is what an instruct release was trained
/// on; a release without one has its messages joined as plain text. Generation stops at the
/// tokenizer's end-of-sequence marker or at any id the release's `generation_config.json` lists. Each
/// step re-runs the whole sequence, so a reply's cost grows with its length; run inference off the
/// render thread. Introduced in InferKit 0.4.0.
@objc(NFKMLXDecoderBackend)
public final class NFKMLXDecoderBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKDecoderBackendHolder
    private let identifier: String
    private let chatTemplate: String?
    private let beginOfSequence: Int?
    private let stopTokens: Set<Int>

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKTokenizer, identifier: String,
         chatTemplate: String?, stopTokens: Set<Int>) {
        self.holder = NFKDecoderBackendHolder(logits: logits, tokenizer: tokenizer)
        self.identifier = identifier
        self.chatTemplate = chatTemplate
        beginOfSequence = tokenizer.bosTokenId >= 0 ? tokenizer.bosTokenId : nil
        self.stopTokens = stopTokens
        super.init()
    }

    /// The decoder, tokenizer, chat template, and stop ids a release directory supplies.
    static func release(directoryURL: URL, identifier: String,
                        logits: @escaping (MLXArray) -> MLXArray) throws -> NFKMLXDecoderBackend {
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("\(directoryURL.lastPathComponent) has no readable tokenizer")
        }
        return NFKMLXDecoderBackend(logits: logits, tokenizer: tokenizer, identifier: identifier,
                                    chatTemplate: NFKMLXLanguage.chatTemplate(inDirectory: directoryURL),
                                    stopTokens: stopTokens(tokenizer: tokenizer, directoryURL: directoryURL))
    }

    /// The tokenizer's end-of-sequence id and every id `generation_config.json` names as one; an
    /// instruct release ends a turn on its own marker, which only the generation config lists.
    static func stopTokens(tokenizer: NFKTokenizer, directoryURL: URL) -> Set<Int> {
        var stops = Set<Int>()
        if tokenizer.eosTokenId >= 0 { stops.insert(tokenizer.eosTokenId) }
        let url = directoryURL.appendingPathComponent("generation_config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return stops }
        if let single = json["eos_token_id"] as? NSNumber {
            stops.insert(single.intValue)
        } else if let list = json["eos_token_id"] as? [NSNumber] {
            stops.formUnion(list.map(\.intValue))
        }
        return stops
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterTemperature, NFKParameterMaxTokens, NFKParameterSeed]
    }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt, NFKInputMessages] }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        var temperature: Float = 0
        var maximumTokens = 256
        var seed: UInt64?
        if let value = request.parameter(forKey: NFKParameterTemperature) as? NSNumber {
            temperature = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            maximumTokens = value.intValue
        }
        if let value = request.parameter(forKey: NFKParameterSeed) as? NSNumber {
            seed = value.uint64Value
        }

        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        let promptTokens = try tokens(for: request)
        let produced = generate(promptTokens, temperature: temperature, maxTokens: maximumTokens, seed: seed)
        return NFKInferenceResult(outputs: [
            NFKOutputText: holder.tokenizer.decode(produced.map { NSNumber(value: $0) }),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                outputTokens: produced.count, reasoningTokens: nil),
        ])
    }

    /// The prompt as token ids. A raw prompt follows the begin-of-sequence marker where the tokenizer
    /// defines one; a message list renders through the chat template, which places its own markers.
    private func tokens(for request: NFKInferenceRequest) throws -> [Int] {
        let prefix = beginOfSequence.map { [$0] } ?? []
        if let prompt = request.prompt {
            return prefix + holder.tokenizer.encode(prompt).map(\.intValue)
        }
        guard let messages = request.messages else { throw NFKMLXError.unsupportedInput }
        guard let chatTemplate else {
            let joined = messages.compactMap { $0["content"] as? String }.joined(separator: "\n\n")
            return prefix + holder.tokenizer.encode(joined).map(\.intValue)
        }
        let rendered = try NFKMLXChatTemplateRenderer.render(chatTemplate, messages: messages)
        return holder.tokenizer.encode(rendered).map(\.intValue)
    }

    /// Prefill-only generation: the whole growing sequence runs through the decoder each step.
    private func generate(_ promptTokens: [Int], temperature: Float, maxTokens: Int, seed: UInt64?) -> [Int] {
        var tokens = promptTokens
        var produced = [Int]()
        for step in 0 ..< max(maxTokens, 0) {
            let input = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
            let logits = holder.logits(input)[0, tokens.count - 1]
            let next: Int
            if temperature <= 0 {
                next = logits.argMax(axis: -1).item(Int.self)
            } else {
                if let seed { MLXRandom.seed(seed &+ UInt64(step)) }
                next = MLXRandom.categorical(logits * (1 / temperature)).item(Int.self)
            }
            if stopTokens.contains(next) { break }
            produced.append(next)
            tokens.append(next)
        }
        return produced
    }
}
