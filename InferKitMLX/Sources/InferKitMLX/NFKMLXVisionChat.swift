//
//  NFKMLXVisionChat.swift
//  InferKitMLX
//
//  A chat over images for the vision-language models whose decoder is an `NFKMLXLanguageNet` (Qwen3-VL,
//  Pixtral, SmolVLM2): a prompt or a conversation with its images in, a reply out, and each conversation's
//  cache kept between requests so a follow-up prefills only what it adds, its images included.
//

import CoreGraphics
import CryptoKit
import Foundation
import InferKit
import MLX
import MLXNN

/// One image prepared for a vision-language decoder.
struct NFKMLXVisionImage {
    /// What the model's vision tower reads.
    let pixels: MLXArray
    /// The text the image's placeholder expands to in the prompt.
    let expansion: String
    /// How many image-token positions the expansion holds.
    let tokenCount: Int
    /// The patch grid (temporal, height, width) the rotary positions read where they depend on the image.
    let grid: (t: Int, h: Int, w: Int)
    /// A digest of the pixels, which keys the image's positions in a kept conversation.
    let digest: UInt64

    init(pixels: MLXArray, expansion: String, tokenCount: Int, grid: (t: Int, h: Int, w: Int) = (1, 1, 1)) {
        self.pixels = pixels
        self.expansion = expansion
        self.tokenCount = tokenCount
        self.grid = grid
        digest = Self.digest(of: pixels)
    }

    /// The first eight bytes of the SHA-256 of the pixels' values and shape.
    static func digest(of pixels: MLXArray) -> UInt64 {
        var hasher = SHA256()
        hasher.update(data: pixels.asData(access: .copy).data)
        pixels.shape.map(Int64.init).withUnsafeBytes { hasher.update(bufferPointer: $0) }
        return hasher.finalize().withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
    }
}

/// What one vision-language model supplies to ``NFKMLXVisionChatBackend``.
protocol NFKMLXVisionChatModel: AnyObject {
    var chatDecoder: NFKMLXLanguageNet { get }
    var chatTokenizer: NFKTokenizer { get }
    var chatEndTokens: Set<Int> { get }
    /// The id each image-feature position carries in the prompt.
    var chatImageTokenId: Int { get }
    /// The release's chat template, or nil where the release ships none.
    var chatTemplate: NFKMLXVisionTemplate? { get }
    /// The text the release's template writes for one image part, which the image's expansion replaces.
    var chatImageMarker: String { get }
    /// Whether the template reads every message's content as a list of typed parts, a text-only one too.
    var chatTextAsParts: Bool { get }
    /// The modules whose weights the model holds, which its description and memory budget read.
    var chatModules: [Module] { get }
    /// The release directory, which names a conversation's spill directory, or nil for none.
    var chatReleaseDirectoryURL: URL? { get }
    /// The strengths the held adapters were loaded at, in load order; empty for a model that holds none.
    var chatAdapterStrengths: [Float] { get }

    func prepareChatImage(_ image: CGImage) -> NFKMLXVisionImage
    /// The model's own prompt for one question about `images`, the single user turn it has always read.
    func singleTurnPrompt(question: String, images: [NFKMLXVisionImage]) -> String
    /// The image's features `[tokenCount, hidden]` and the maps `[tokenCount, hidden]` the decoder adds to its
    /// first layers' image positions.
    func encodeChatImage(_ image: NFKMLXVisionImage) -> (features: MLXArray, deepstack: [MLXArray])
    /// The rotary positions `[3][sequence]` where they depend on the images, or nil where the decoder counts
    /// positions from its cache.
    func chatPositions(inputIds: [Int], images: [NFKMLXVisionImage]) -> [[Int]]?
    /// The rotary tables for `positions`.
    func chatRope(_ positions: [[Int]]) -> (cos: MLXArray, sin: MLXArray)
}

extension NFKMLXVisionChatModel {
    var chatTextAsParts: Bool { false }
    func chatPositions(inputIds: [Int], images: [NFKMLXVisionImage]) -> [[Int]]? { nil }
    func chatRope(_ positions: [[Int]]) -> (cos: MLXArray, sin: MLXArray) {
        fatalError("a model that answers positions supplies their rotary")
    }
}

/// A release's chat template and the begin-of-sequence marker it writes through `bos_token`.
struct NFKMLXVisionTemplate {
    let source: String
    let beginToken: String

    init(source: String, beginToken: String = "") {
        self.source = source
        self.beginToken = beginToken
    }

    /// The template `directoryURL` ships, or nil where it ships none.
    init?(directoryURL: URL) {
        guard case .jinja(let source, let begin, _) = NFKMLXLanguage.releaseChatTemplate(inDirectory: directoryURL) else {
            return nil
        }
        self.source = source
        beginToken = begin
    }
}

/// How a request's images reach the prompt.
enum NFKMLXVisionPrompt {
    /// The request's images in order: `NFKInputImage`, then `NFKInputImages`.
    static func images(of request: NFKInferenceRequest) throws -> [CGImage] {
        try NFKMLXGenerationBackend.images(request, NFKInputImage) + NFKMLXGenerationBackend.images(request, NFKInputImages)
    }

    /// How many images each message carried, under its `NFKMessageMetadataKey` `imageCount`. Where no message
    /// states a count, every image belongs to the first user turn.
    static func imageCounts(of messages: [[String: Any]], total: Int) throws -> [Int] {
        let stated = messages.map { (($0[NFKMessageMetadataKey] as? [String: Any])?["imageCount"] as? NSNumber)?.intValue }
        if stated.allSatisfy({ $0 == nil }) {
            guard total > 0 else { return messages.map { _ in 0 } }
            guard let first = messages.firstIndex(where: { $0["role"] as? String == "user" }) else {
                throw NFKMLXError.unsupportedConfiguration("a conversation's images need a user turn to belong to")
            }
            return messages.indices.map { $0 == first ? total : 0 }
        }
        let counts = stated.map { $0 ?? 0 }
        guard counts.reduce(0, +) == total else {
            throw NFKMLXError.unsupportedConfiguration(
                "the messages state \(counts.reduce(0, +)) images and the request carries \(total)")
        }
        return counts
    }

    /// The conversation rendered through `model`'s chat template with each message's images as image parts
    /// before its text, then each image's marker replaced by the image's expansion, as the release's
    /// processor does.
    static func render(_ messages: [[String: Any]], images: [NFKMLXVisionImage], model: NFKMLXVisionChatModel) throws -> String {
        guard let template = model.chatTemplate else {
            throw NFKMLXError.unsupportedConfiguration("the release ships no chat template to render a conversation with")
        }
        let counts = try imageCounts(of: messages, total: images.count)
        let parted = zip(messages, counts).map { message, count -> [String: Any] in
            guard count > 0 || model.chatTextAsParts else {
                var plain = message
                plain[NFKMessageMetadataKey] = nil
                return plain
            }
            let text = message["content"] as? String ?? ""
            let parts = Array(repeating: ["type": "image"] as [String: Any], count: count)
                + [["type": "text", "text": text] as [String: Any]]
            return ["role": message["role"] ?? "user", "content": parts]
        }
        let rendered = try NFKMLXChatTemplateRenderer.render(template.source, messages: parted,
                                                             bosToken: template.beginToken)
        let pieces = rendered.components(separatedBy: model.chatImageMarker)
        guard pieces.count == images.count + 1 else {
            throw NFKMLXError.unsupportedConfiguration(
                "the chat template wrote \(pieces.count - 1) image markers for \(images.count) images")
        }
        var prompt = pieces[0]
        for (image, piece) in zip(images, pieces.dropFirst()) {
            prompt += image.expansion + piece
        }
        return prompt
    }
}

/// A generation through a kept conversation's cache over a prompt that splices image features in.
enum NFKMLXVisionChat {
    /// The key a conversation keeps the prompt under: the ids, with each image-token position replaced by a
    /// negative id from the image's digest and the token's place in the image, so a prompt with another image
    /// there shares nothing past it.
    static func key(inputIds: [Int], images: [NFKMLXVisionImage], imageTokenId: Int) throws -> [Int] {
        try mediaKey(inputIds: inputIds, isMedia: { $0 == imageTokenId },
                     media: images.map { (digest: $0.digest, tokenCount: $0.tokenCount) })
    }

    /// ``key(inputIds:images:imageTokenId:)`` for any media whose positions `isMedia` marks: the `n`th run of
    /// `media[n].tokenCount` marked positions is keyed by `media[n].digest`, in prompt order.
    static func mediaKey(inputIds: [Int], isMedia: (Int) -> Bool,
                         media: [(digest: UInt64, tokenCount: Int)]) throws -> [Int] {
        var key = inputIds
        var item = 0, within = 0
        for position in inputIds.indices where isMedia(inputIds[position]) {
            guard item < media.count else {
                throw NFKMLXError.unsupportedConfiguration("the prompt holds more media positions than its media fill")
            }
            let mixed = media[item].digest &+ UInt64(within) &* 0x9E37_79B9_7F4A_7C15
            key[position] = -1 - Int(truncatingIfNeeded: mixed >> 1)
            within += 1
            if within == media[item].tokenCount {
                item += 1
                within = 0
            }
        }
        guard item == media.count, within == 0 else {
            throw NFKMLXError.unsupportedConfiguration("the prompt's media positions do not match its media")
        }
        return key
    }

    /// Generation through the conversation `request` names or its prompt continues, or a cache kept for no
    /// later request where it turns reuse off.
    ///
    /// @discussion The conversation's cache rolls back to what its key shares with this prompt's key. Only the
    /// images with a position past that point run through the vision tower, the rest of the prompt prefills in
    /// passes with their features spliced in (and, for a decoder whose rotary reads the images, its positions
    /// and deepstack), and each new token then runs alone. A run that fails discards the conversation. Runs
    /// under the backend's generation lock. Answers the tokens produced and how many prompt positions the
    /// cache already held.
    static func run(model: NFKMLXVisionChatModel, inputIds: [Int], images: [NFKMLXVisionImage],
                    conversations: NFKMLXConversationKeeper<NFKMLXPromptCache>, request: NFKInferenceRequest,
                    sampling: NFKMLXStepSampling, limit: Int, seed: UInt64?, afforded: () -> Int, spills: Bool,
                    onToken: ((Int) -> Bool)? = nil) throws -> (produced: [Int], cached: Int) {
        let decoder = model.chatDecoder
        let key = try key(inputIds: inputIds, images: images, imageTokenId: model.chatImageTokenId)
        let strengths = NFKMLXAdapterRun.strengths
        let layerCount = decoder.configuration.layerCount
        let signature = NFKMLXHybridConversation.optionsSignature(window: nil, quantization: nil, adapterStrengths: strengths)
        let (name, state) = conversations.conversation(
            named: NFKMLXConversation.name(of: request), reuses: NFKMLXConversation.reuses(request), prompt: key,
            signature: signature,
            usable: { $0.layerCount == layerCount && $0.window == nil && $0.quantization == nil && $0.adapterStrengths == strengths },
            make: {
                let made = NFKMLXPromptCache(layerCount: layerCount)
                made.adapterStrengths = strengths
                return made
            },
            read: { url, record in
                let restored = try NFKMLXPromptCache.load(from: url)
                restored.adapterStrengths = strengths
                return restored.tokens == record.tokens && restored.layerCount == layerCount ? restored : nil
            })
        var failed = true
        defer { conversations.finish(name, failed: failed, afforded: afforded(), spills: spills) }

        let start = state.align(to: key)
        let positions = model.chatPositions(inputIds: inputIds, images: images)
        let input = try freshInput(model: model, inputIds: inputIds, images: images, from: start, positions: positions)
        var hidden = prefill(decoder, embeddings: input.embeddings, multimodal: input.multimodal, cache: state.cache)
        try check(decoder)
        state.record(Array(key[start...]))

        var produced = [Int]()
        var nextPosition = positions.map { ($0.flatMap { $0 }.max() ?? (inputIds.count - 1)) + 1 }
        for step in 0 ..< limit {
            let logits = decoder.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...]).reshaped([-1])
            let next = sampling.next(logits, seed: seed, step: step)
            try check(decoder)
            if model.chatEndTokens.contains(next) { break }
            produced.append(next)
            if let onToken, !onToken(next) { break }
            guard step + 1 < limit else { break }
            var multimodal: NFKLMMultimodal?
            if let position = nextPosition {
                multimodal = textStep(rope: model.chatRope([[position], [position], [position]]))
                nextPosition = position + 1
            }
            hidden = decoder.hiddenStates(fromEmbeddings: decoder.embed(MLXArray([Int32(next)]).reshaped([1, 1])),
                                          cache: state.cache, multimodal: multimodal)
            state.record([next])
        }
        eval(hidden)
        try check(decoder)
        failed = false
        return (produced, start)
    }

    /// Throws a failed evaluation or a failed read of a streamed layer.
    private static func check(_ decoder: NFKMLXLanguageNet) throws {
        if let failure = NFKMLXErrorScope.failure {
            throw failure
        }
        try decoder.verifyStream()
    }

    /// The decoder's input for the prompt from `start` on `[1, length, hidden]`: the ids embedded, with the
    /// features of the images whose positions it reaches spliced into their positions, and, where the rotary
    /// reads the images, the tables and deepstack for those positions.
    static func freshInput(model: NFKMLXVisionChatModel, inputIds: [Int], images: [NFKMLXVisionImage], from start: Int,
                           positions: [[Int]]?) throws -> (embeddings: MLXArray, multimodal: NFKLMMultimodal?) {
        let decoder = model.chatDecoder
        let width = decoder.configuration.hiddenSize
        let fresh = Array(inputIds[start...])
        var embeddings = decoder.embed(MLXArray(fresh.map(Int32.init)).reshaped([1, fresh.count]))[0]

        // The image-token ordinal of the first fresh position, and the first image that ordinal falls in.
        let skipped = inputIds[..<start].filter { $0 == model.chatImageTokenId }.count
        var firstImage = 0, ordinalStart = 0
        while firstImage < images.count && ordinalStart + images[firstImage].tokenCount <= skipped {
            ordinalStart += images[firstImage].tokenCount
            firstImage += 1
        }
        var featureIndex = [Int32](repeating: 0, count: fresh.count)
        var isImage = [Float](repeating: 0, count: fresh.count)
        var ordinal = skipped - ordinalStart
        for position in fresh.indices where fresh[position] == model.chatImageTokenId {
            featureIndex[position] = Int32(ordinal)
            isImage[position] = 1
            ordinal += 1
        }
        let indexArray = MLXArray(featureIndex)
        let mask = MLXArray(isImage).reshaped([fresh.count, 1]) .> 0
        var deepstack = [MLXArray]()
        if ordinal > skipped - ordinalStart {
            let encoded = images[firstImage...].map { model.encodeChatImage($0) }
            let features = concatenated(encoded.map { $0.features.reshaped([-1, width]) }, axis: 0)
            embeddings = MLX.where(mask, features.take(indexArray, axis: 0).asType(embeddings.dtype), embeddings)
            let layers = encoded.map(\.deepstack.count).min() ?? 0
            deepstack = (0 ..< layers).map { layer in
                concatenated(encoded.map { $0.deepstack[layer].reshaped([-1, width]) }, axis: 0)
            }
        }
        let multimodal = positions.map { positions in
            NFKLMMultimodal(rope: model.chatRope(positions.map { Array($0[start...]) }), features: deepstack,
                            featureIndex: indexArray, mask: mask.reshaped([1, fresh.count, 1]))
        }
        return (embeddings.reshaped([1, fresh.count, width]), multimodal)
    }

    /// Runs `embeddings` through `cache` in passes of at most `limit` positions, shortened where their attention
    /// scores would pass the decoder's budget, slicing the rotary tables and the deepstack's positions to each
    /// pass. Answers the last pass's hidden states.
    static func prefill(_ decoder: NFKMLXLanguageNet, embeddings: MLXArray, multimodal: NFKLMMultimodal?,
                        cache: NFKMLXKeyValueCache, limit: Int = NFKMLXLanguageNet.defaultPrefillChunk) -> MLXArray {
        let total = embeddings.dim(1)
        var start = 0
        var hidden: MLXArray
        repeat {
            let length = decoder.prefillPassLength(cached: cache.maskCacheLength, remaining: total - start, limit: limit)
            let range = start ..< start + length
            let pass = multimodal.map { whole in
                NFKLMMultimodal(rope: (whole.rope.cos[0..., range, 0...], whole.rope.sin[0..., range, 0...]),
                                features: whole.features, featureIndex: whole.featureIndex[range],
                                mask: whole.mask[0..., range, 0...])
            }
            hidden = decoder.hiddenStates(fromEmbeddings: embeddings[0..., range, 0...], cache: cache, multimodal: pass)
            start += length
            if start < total {
                eval(hidden)                                    // free a pass's activations before the next
            }
        } while start < total && !NFKMLXErrorScope.hasFailed
        return hidden
    }

    /// One generated token's multimodal input: the rotary at its position, and no image features.
    private static func textStep(rope: (cos: MLXArray, sin: MLXArray)) -> NFKLMMultimodal {
        NFKLMMultimodal(rope: rope, features: [], featureIndex: MLXArray([Int32(0)]),
                        mask: MLXArray([Float(0)]).reshaped([1, 1, 1]) .> 0)
    }
}
