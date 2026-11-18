//
//  NFKMLXPhi4MMBackend.swift
//  InferKitMLX
//
//  Phi-4-multimodal assembled for inference, and its backend. One decoder carries both released LoRA
//  adapters live (the mixture of LoRAs); a request's inputs choose the mode: an image (with or without
//  audio) runs the vision adapter, audio alone runs the speech adapter, and text alone runs the base.
//  A conversation renders through the release's chat template, `<|role|>content<|end|>` per message
//  and a closing `<|assistant|>`, with each image and audio placeholder repeated once per embedding it
//  receives.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// The release's chat template and placeholder layout, as its processor applies them.
enum NFKMLXPhi4MMPrompt {
    static let imagePlaceholder = "<|endoftext10|>"
    static let audioPlaceholder = "<|endoftext11|>"
    static let imageReference = try! NSRegularExpression(pattern: #"<\|image_\d+\|>"#)
    static let audioReference = try! NSRegularExpression(pattern: #"<\|audio_\d+\|>"#)

    /// The markers the release's tokenizer declares `rstrip`: whitespace after one is absorbed into it.
    static let rightStrippedMarkers = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|tool|>", "<|/tool|>"]

    /// The template's text for `messages` (`role` and `content` strings; a system message may carry a
    /// `tools` string), ending with the assistant's turn marker, and with each `<|image_N|>` and
    /// `<|audio_N|>` reference replaced by its placeholder token.
    ///
    /// @discussion The processor consumes pictures and clips in the order their references appear,
    /// whatever their numbers say. A conversation that references none has them placed at the start of
    /// its first user turn, pictures first. A conversation that references some must reference every
    /// one the request carries.
    static func render(messages: [[String: Any]], images: Int, audios: Int) throws -> String {
        var turns = messages.map { message -> (role: String, content: String, tools: String?) in
            ((message["role"] as? String) ?? "user", (message["content"] as? String) ?? "", message["tools"] as? String)
        }
        let joined = turns.map { $0.content + ($0.tools ?? "") }.joined()
        let namedImages = matches(imageReference, in: joined), namedAudios = matches(audioReference, in: joined)
        guard namedImages == 0 || namedImages == images else {
            throw NFKMLXError.unsupportedConfiguration("the conversation references \(namedImages) images; the request carries \(images)")
        }
        guard namedAudios == 0 || namedAudios == audios else {
            throw NFKMLXError.unsupportedConfiguration("the conversation references \(namedAudios) clips; the request carries \(audios)")
        }
        var placed = ""
        if namedImages == 0 { placed += (0 ..< images).map { "<|image_\($0 + 1)|>" }.joined() }
        if namedAudios == 0 { placed += (0 ..< audios).map { "<|audio_\($0 + 1)|>" }.joined() }
        if !placed.isEmpty {
            guard let first = turns.firstIndex(where: { $0.role == "user" }) else {
                throw NFKMLXError.unsupportedConfiguration("a request with media needs a user turn to carry it")
            }
            turns[first].content = placed + turns[first].content
        }
        var text = ""
        for turn in turns {
            text += "<|\(turn.role)|>" + turn.content
            if turn.role == "system", let tools = turn.tools { text += "<|tool|>" + tools + "<|/tool|>" }
            text += "<|end|>"
        }
        text += "<|assistant|>"
        text = replacing(imageReference, in: text, with: imagePlaceholder)
        text = replacing(audioReference, in: text, with: audioPlaceholder)
        return strippingAfterMarkers(text)
    }

    /// `text` with the whitespace following each right-stripped marker removed, which is what the
    /// release's tokenizer does when it splits the markers out.
    static func strippingAfterMarkers(_ text: String) -> String {
        var result = ""
        var rest = Substring(text)
        while let range = rest.firstRange(of: #/<\|[a-z/]+\|>/#) {
            let marker = String(rest[range])
            result += rest[..<range.upperBound]
            rest = rest[range.upperBound...]
            if rightStrippedMarkers.contains(marker) { rest = rest.drop(while: \.isWhitespace) }
        }
        return result + rest
    }

    private static func matches(_ expression: NSRegularExpression, in text: String) -> Int {
        expression.numberOfMatches(in: text, range: NSRange(text.startIndex..., in: text))
    }

    private static func replacing(_ expression: NSRegularExpression, in text: String, with template: String) -> String {
        expression.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text),
                                            withTemplate: NSRegularExpression.escapedTemplate(for: template))
    }
}

/// Phi-4-multimodal ready to answer: the mixture-of-LoRAs decoder, the image and speech towers, and the
/// release tokenizer. Swift-only; Objective-C reaches it through ``NFKMLXPhi4MMBackend``.
///
/// @discussion The active adapter is state on the shared decoder, so one request runs at a time; a
/// second waits for the first.
public final class NFKMLXPhi4MMModel: @unchecked Sendable {
    public let decoder: NFKMLXLanguageNet
    public let imageNet: NFKMLXPhi4MMImageNet
    public let audioNet: NFKMLXPhi4MMAudioNet
    public let tokenizer: NFKTokenizer
    private let lock = NSLock()

    init(decoder: NFKMLXLanguageNet, imageNet: NFKMLXPhi4MMImageNet, audioNet: NFKMLXPhi4MMAudioNet,
         tokenizer: NFKTokenizer) {
        self.decoder = decoder
        self.imageNet = imageNet
        self.audioNet = audioNet
        self.tokenizer = tokenizer
    }

    /// The mode a request's inputs select.
    static func modality(hasImage: Bool, hasAudio: Bool) -> NFKMLXPhi4MMModality {
        hasImage ? .vision : (hasAudio ? .speech : .language)
    }

    /// The prompt token ids for a conversation: the rendered template, tokenized once, with each image
    /// placeholder repeated `imageTokens[i]` times and each audio placeholder `audioTokens[i]` times.
    func promptIds(messages: [[String: Any]], imageTokens: [Int], audioTokens: [Int]) throws -> [Int] {
        let text = try NFKMLXPhi4MMPrompt.render(messages: messages, images: imageTokens.count, audios: audioTokens.count)
        var images = imageTokens.makeIterator(), audios = audioTokens.makeIterator()
        var ids = [Int]()
        for id in tokenizer.encode(text).map(\.intValue) {
            switch id {
            case NFKMLXPhi4MM.imageTokenId: ids += Array(repeating: id, count: images.next() ?? 0)
            case NFKMLXPhi4MM.audioTokenId: ids += Array(repeating: id, count: audios.next() ?? 0)
            default: ids.append(id)
            }
        }
        return ids
    }

    /// Selects the mode the inputs call for and assembles the prompt ids and the placeholder features.
    /// The pictures are embedded one by one; the clips run through the speech tower as one batch, as
    /// the reference processor submits them. The audio projector's head follows the mode. Each tower's
    /// features are evaluated before the decoder runs, so a tower's intermediates are released first
    /// and never share the peak with the prefill.
    func prepare(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput], audios: [NFKMLXPhi4MMAudioInput]) throws
        -> (ids: [Int], features: [(placeholder: Int, features: MLXArray)]) {
        let mode = Self.modality(hasImage: !images.isEmpty, hasAudio: !audios.isEmpty)
        NFKMLXPhi4MM.select(mode, in: decoder)
        var features = [(placeholder: Int, features: MLXArray)]()
        if !images.isEmpty {
            let embedded = images.map {
                imageNet.projected(pixels: $0.pixels, imageSize: $0.imageSize, validPatches: $0.validPatches)
                    .reshaped([-1, decoder.configuration.hiddenSize])
            }
            let imageFeatures = concatenated(embedded, axis: 0)
            eval(imageFeatures)
            features.append((NFKMLXPhi4MM.imageTokenId, imageFeatures))
        }
        let mels = try Self.logMels(audios)
        if !mels.isEmpty {
            let audioFeatures = concatenated(audioNet.projected(clips: mels, mode: mode), axis: 0)
            eval(audioFeatures)
            features.append((NFKMLXPhi4MM.audioTokenId, audioFeatures))
        }
        return (try promptIds(messages: messages, images: images, mels: mels), features)
    }

    /// Each clip's log-mel features.
    static func logMels(_ audios: [NFKMLXPhi4MMAudioInput]) throws -> [MLXArray] {
        try audios.map { try NFKMLXPhi4MMAudioFeatures.logMel($0.samples, sampleRate: $0.sampleRate) }
    }

    /// A conversation's prompt ids, each picture's and each clip's placeholder run at the length its
    /// features take.
    func promptIds(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput], mels: [MLXArray]) throws -> [Int] {
        try promptIds(messages: messages, imageTokens: images.map(\.tokenCount),
                      audioTokens: mels.map { NFKMLXPhi4MMAudioFeatures.tokenCount(frames: $0.dim(0)) })
    }

    /// The input tokens a conversation costs, the ids ``prepare(messages:images:audios:)`` builds,
    /// without running a tower.
    func promptIds(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput], audios: [NFKMLXPhi4MMAudioInput]) throws -> [Int] {
        try promptIds(messages: messages, images: images, mels: Self.logMels(audios))
    }

    func promptTokenCount(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput],
                          audios: [NFKMLXPhi4MMAudioInput]) throws -> Int {
        try promptIds(messages: messages, images: images, mels: Self.logMels(audios)).count
    }

    /// One user turn holding `text`, with at most one picture and one clip.
    func prepare(text: String, image: NFKMLXPhi4MMImageInput?, audio: [Float]?, sampleRate: Int = 16000) throws
        -> (ids: [Int], features: [(placeholder: Int, features: MLXArray)]) {
        try prepare(messages: [["role": "user", "content": text]], images: image.map { [$0] } ?? [],
                    audios: audio.map { [NFKMLXPhi4MMAudioInput(samples: $0, sampleRate: sampleRate)] } ?? [])
    }

    /// The answer's token ids to a conversation, stopping at the turn end. `options` sets the sampling
    /// (greedy by default, as the release's generation config is), the token budget (256 by default), and
    /// the seed; `onToken` sees each id as it is produced, and returning false stops the run.
    public func generate(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput] = [],
                         audios: [NFKMLXPhi4MMAudioInput] = [],
                         options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                         onToken: ((Int) -> Bool)? = nil) throws -> [Int] {
        var limited = options
        limited.maxTokens = options.maxTokens
        return try answer(messages: messages, images: images, audios: audios, options: limited, onToken: onToken).ids
    }

    /// The answer's ids and the prompt's length, which the backend reports as usage.
    /// ``answer(messages:images:audios:options:onToken:)`` continuing `promptCache`.
    ///
    /// @discussion The cache holds the prompt's key: the ids, each image's and clip's placeholder positions keyed by
    /// a digest of its pixels or log-mel features, and the first position keyed by the modality whose adapter the
    /// rows were computed under, so a turn in another modality starts over. The cache rolls back to the key it shares
    /// and further to a span's start where that ends inside a medium; only the media whose spans the rest of the
    /// prompt holds run through their towers. Answers the reply, the prompt's length, and how many of its positions
    /// the cache already held.
    func answer(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput], audios: [NFKMLXPhi4MMAudioInput],
                options: NFKMLXGenerationOptions, promptCache: NFKMLXPromptCache,
                onToken: ((Int) -> Bool)?) throws -> (ids: [Int], promptTokens: Int, cached: Int) {
        lock.lock(); defer { lock.unlock() }
        let mode = Self.modality(hasImage: !images.isEmpty, hasAudio: !audios.isEmpty)
        NFKMLXPhi4MM.select(mode, in: decoder)
        let mels = try Self.logMels(audios)
        let ids = try promptIds(messages: messages, images: images, mels: mels)
        let key = try conversationKey(ids, mode: mode, images: images, mels: mels)
        let placeholders: Set<Int> = [NFKMLXPhi4MM.imageTokenId, NFKMLXPhi4MM.audioTokenId]
        var start = promptCache.align(to: key)
        if start > 0, start < ids.count, placeholders.contains(ids[start]), ids[start - 1] == ids[start] {
            var first = start
            while first > 0, ids[first - 1] == ids[start] { first -= 1 }
            start = promptCache.rollback(by: start - first) ? first : 0
            if start == 0 { promptCache.reset() }
        }
        // The media whose spans the fresh positions hold, from the first such of each kind.
        let skippedImages = ids[..<start].filter { $0 == NFKMLXPhi4MM.imageTokenId }.count
        let skippedClips = ids[..<start].filter { $0 == NFKMLXPhi4MM.audioTokenId }.count
        var features = [(placeholder: Int, features: MLXArray)]()
        var held = 0
        let neededImages = images.filter { image in defer { held += image.tokenCount }; return held >= skippedImages }
        if !neededImages.isEmpty {
            features.append((NFKMLXPhi4MM.imageTokenId, concatenated(neededImages.map {
                imageNet.projected(pixels: $0.pixels, imageSize: $0.imageSize, validPatches: $0.validPatches)
                    .reshaped([-1, decoder.configuration.hiddenSize])
            }, axis: 0)))
        }
        held = 0
        let neededClips = mels.filter { mel in
            defer { held += NFKMLXPhi4MMAudioFeatures.tokenCount(frames: mel.dim(0)) }
            return held >= skippedClips
        }
        if !neededClips.isEmpty {
            features.append((NFKMLXPhi4MM.audioTokenId, concatenated(audioNet.projected(clips: neededClips, mode: mode), axis: 0)))
        }
        var bounded = options
        bounded.maxTokens = try NFKMLXContextRoom.bounded(options.chosenMaxTokens, promptTokens: ids.count,
                                                          contextLength: decoder.configuration.contextLength,
                                                          source: NFKMLXPhi4MM.modelName)
        if let seed = bounded.seed { MLXRandom.seed(seed) }
        let fresh = Array(ids[start...])
        var hidden = NFKMLXPhi4MM.fusedHidden(decoder: decoder, inputIds: fresh, features: features, cache: promptCache.cache)
        promptCache.record(Array(key[start...]))
        var produced = [Int]()
        for step in 0 ..< bounded.maxTokens {
            let logits = decoder.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...]).reshaped([-1])
            let next = NFKMLXLanguageNet.sample(logits, options: bounded)
            if NFKMLXPhi4MM.stopTokens.contains(next) || NFKMLXErrorScope.hasFailed { break }
            produced.append(next)
            if let onToken, !onToken(next) { break }
            guard step + 1 < bounded.maxTokens else { break }
            hidden = decoder.hiddenStates(fromEmbeddings: decoder.embed(MLXArray([Int32(next)]).reshaped([1, 1])),
                                          cache: promptCache.cache)
            promptCache.record([next])
        }
        if let failure = NFKMLXErrorScope.failure {
            promptCache.reset()
            throw failure
        }
        return (produced, ids.count, start)
    }

    /// The key a conversation keeps `ids` under: each image's and clip's placeholder positions keyed by a digest of
    /// its pixels or log-mel features, and the first position by the modality whose adapter computed the rows.
    func conversationKey(_ ids: [Int], mode: NFKMLXPhi4MMModality, images: [NFKMLXPhi4MMImageInput],
                         mels: [MLXArray]) throws -> [Int] {
        var key = try NFKMLXVisionChat.mediaKey(inputIds: ids, isMedia: { $0 == NFKMLXPhi4MM.imageTokenId },
                                                media: images.map { (NFKMLXVisionImage.digest(of: $0.pixels), $0.tokenCount) })
        key = try NFKMLXVisionChat.mediaKey(inputIds: key, isMedia: { $0 == NFKMLXPhi4MM.audioTokenId },
                                            media: mels.map { (NFKMLXVisionImage.digest(of: $0),
                                                               NFKMLXPhi4MMAudioFeatures.tokenCount(frames: $0.dim(0))) })
        if !key.isEmpty {
            let modes: [NFKMLXPhi4MMModality] = [.language, .vision, .speech]
            key[0] = -(modes.firstIndex(of: mode) ?? 0) - 1 - (key[0] << 2)
        }
        return key
    }

    func answer(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput], audios: [NFKMLXPhi4MMAudioInput],
                options: NFKMLXGenerationOptions, onToken: ((Int) -> Bool)?) throws -> (ids: [Int], promptTokens: Int) {
        lock.lock(); defer { lock.unlock() }
        let prepared = try prepare(messages: messages, images: images, audios: audios)
        var bounded = options
        bounded.maxTokens = try NFKMLXContextRoom.bounded(options.chosenMaxTokens, promptTokens: prepared.ids.count,
                                                          contextLength: decoder.configuration.contextLength,
                                                          source: NFKMLXPhi4MM.modelName)
        let ids = NFKMLXPhi4MM.generateFused(decoder: decoder, inputIds: prepared.ids, features: prepared.features,
                                             options: bounded, endTokens: NFKMLXPhi4MM.stopTokens, onToken: onToken)
        return (ids, prepared.ids.count)
    }

    /// Greedy token ids answering one user turn, stopping at the turn end.
    public func generate(text: String, image: NFKMLXPhi4MMImageInput? = nil, audio: [Float]? = nil,
                         sampleRate: Int = 16000, maxTokens: Int = 256) throws -> [Int] {
        var options = NFKMLXGenerationOptions()
        options.maxTokens = maxTokens
        return try generate(messages: [["role": "user", "content": text]], images: image.map { [$0] } ?? [],
                            audios: audio.map { [NFKMLXPhi4MMAudioInput(samples: $0, sampleRate: sampleRate)] } ?? [],
                            options: options)
    }

    /// The first answer token's logits `[vocabulary]` for a conversation.
    func firstLogits(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput] = [],
                     audios: [NFKMLXPhi4MMAudioInput] = []) throws -> MLXArray {
        lock.lock(); defer { lock.unlock() }
        let prepared = try prepare(messages: messages, images: images, audios: audios)
        let hidden = NFKMLXPhi4MM.fusedHidden(decoder: decoder, inputIds: prepared.ids, features: prepared.features)
        return decoder.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...]).reshaped([-1])
    }

    /// The first answer token's logits `[vocabulary]` for one user turn.
    func firstLogits(text: String, image: NFKMLXPhi4MMImageInput? = nil, audio: [Float]? = nil,
                     sampleRate: Int = 16000) throws -> MLXArray {
        try firstLogits(messages: [["role": "user", "content": text]], images: image.map { [$0] } ?? [],
                        audios: audio.map { [NFKMLXPhi4MMAudioInput(samples: $0, sampleRate: sampleRate)] } ?? [])
    }

    /// The text for a run of answer ids.
    public func decode(_ ids: [Int]) -> String { tokenizer.decode(ids.map { NSNumber(value: $0) }) }

    /// The decoded answer to a conversation.
    public func respond(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput] = [],
                        audios: [NFKMLXPhi4MMAudioInput] = [],
                        options: NFKMLXGenerationOptions = NFKMLXGenerationOptions()) throws -> String {
        decode(try generate(messages: messages, images: images, audios: audios, options: options))
    }

    /// The decoded greedy answer to one user turn.
    public func respond(text: String, image: NFKMLXPhi4MMImageInput? = nil, audio: [Float]? = nil,
                        sampleRate: Int = 16000, maxTokens: Int = 256) throws -> String {
        decode(try generate(text: text, image: image, audio: audio, sampleRate: sampleRate, maxTokens: maxTokens))
    }
}

final class NFKPhi4MMCancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
}

/// A multimodal backend over Phi-4-multimodal. Returns the answer under `NFKOutputText` and the token
/// counts under `NFKOutputUsage`.
///
/// @discussion Inputs:
/// - `NFKInputMessages` → a conversation through the release's chat template (`system`, `user`, and
///   `assistant` turns; a system message's `tools` string rides in the template's tool block);
///   otherwise `NFKInputPrompt` → one user turn.
/// - `NFKInputImage`, then each of `NFKInputImages` → the pictures, in order.
/// - `NFKInputAudio`, then each of `NFKInputAudios` → the clips (an `NFKAudioAsset` or WAV bytes), in
///   order, each at its own sample rate.
/// - Text that names `<|image_1|>`… or `<|audio_1|>`… places the media there; otherwise they open the
///   first user turn, pictures first.
/// - Clips with no text and no picture → the release's transcription instruction.
///
/// `NFKParameterTemperature`, `NFKParameterTopP`, `NFKParameterMaxTokens`, and `NFKParameterSeed` set
/// the sampling; the default is greedy, as the release's generation config is. The job form reports
/// the partial answer as it grows and stops at cancellation.
@objc(NFKMLXPhi4MMBackend)
public final class NFKMLXPhi4MMBackend: NSObject, NFKInferenceBackend {
    private let model: NFKMLXPhi4MMModel
    private let identifier: String

    /// The instruction an audio-only request without text runs, the release's transcription prompt.
    static let transcriptionPrompt = "Transcribe the audio clip into text."

    /// The release directory, which names the default spill directory, or nil for a model built without one.
    private let releaseDirectoryURL: URL?

    init(model: NFKMLXPhi4MMModel, identifier: String, releaseDirectoryURL: URL? = nil) {
        self.model = model
        self.identifier = identifier
        self.releaseDirectoryURL = releaseDirectoryURL
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }
    @objc public var supportedInputKeys: Set<String> {
        [NFKInputPrompt, NFKInputMessages, NFKInputImage, NFKInputImages, NFKInputAudio, NFKInputAudios]
    }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputText, NFKOutputUsage] }

    private let modelInfoCache = NFKMLXModelInfoCache()

    /// The decoder and both towers' parameter count, weight bytes, precision, and quantization, and
    /// the decoder's stated positions (`NFKModelInfo*` keys). Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            var info = NFKMLXModelDescription.info(of: [model.decoder, model.imageNet, model.audioNet])
            info[NFKModelInfoContextLength] = model.decoder.configuration.maximumPositions
            return info
        }
    }
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterTemperature, NFKParameterTopP, NFKParameterMaxTokens, NFKParameterSeed, NFKParameterConversationKey]
    }

    /// One generation at a time: a run moves a conversation's cache through the shared decoder.
    private let generationLock = NSLock()
    /// The conversations' caches, read and written under `generationLock`.
    private let conversations = NFKMLXConversationKeeper<NFKMLXPromptCache>(byteBudget: 0)
    /// The model holds no adapters a request sets.
    private var adapterStrengths: [Float] { [] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try run(request, onToken: nil)
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let flag = NFKPhi4MMCancelFlag()
        job.cancellationHandler = { flag.cancel() }
        let maximum = max(Self.options(from: request).maxTokens, 1)
        Task.detached(priority: .userInitiated) { [self] in
            do {
                let result = try run(request) { produced in
                    let partial = NFKInferenceResult(outputs: [NFKOutputText: self.model.decode(produced)])
                    job.reportProgress(min(Double(produced.count) / Double(maximum), 0.99), partialResult: partial)
                    return !flag.isCancelled
                }
                if !flag.isCancelled { job.finish(with: result) }
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    /// The input tokens a run of the request costs: its conversation with each picture's and each
    /// clip's placeholder run, as a run builds it. Runs no tower and no decoder. Introduced in
    /// InferKit 0.4.0.
    @objc(tokenCountForRequest:error:)
    public func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        let (messages, images, audios) = try inputs(for: request)
        return NSNumber(value: try model.promptTokenCount(messages: messages, images: images, audios: audios))
    }

    /// What a run of the request reads: its conversation (a lone prompt as one user turn, a lone clip
    /// as a transcription request), its pictures, and its clips.
    private func inputs(for request: NFKInferenceRequest) throws
        -> (messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput], audios: [NFKMLXPhi4MMAudioInput]) {
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        let images = try Self.images(from: request)
        let audios = try Self.audios(from: request)
        if let conversation = request.messages, !conversation.isEmpty {
            return (conversation, images, audios)
        }
        if let prompt = request.prompt, !prompt.isEmpty {
            return ([["role": "user", "content": prompt]], images, audios)
        }
        if !audios.isEmpty, images.isEmpty {
            return ([["role": "user", "content": Self.transcriptionPrompt]], images, audios)
        }
        if !images.isEmpty {
            return ([["role": "user", "content": ""]], images, audios)
        }
        throw NFKMLXError.unsupportedInput
    }

    private func run(_ request: NFKInferenceRequest, onToken: (([Int]) -> Bool)?) throws -> NFKInferenceResult {
        let options = Self.options(from: request)
        let (messages, images, audios) = try inputs(for: request)

        var produced = [Int]()
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
        let layerCount = model.decoder.configuration.layerCount
        let mode = NFKMLXPhi4MMModel.modality(hasImage: !images.isEmpty, hasAudio: !audios.isEmpty)
        let key = try model.conversationKey(model.promptIds(messages: messages, images: images, audios: audios), mode: mode,
                                            images: images, mels: try NFKMLXPhi4MMModel.logMels(audios))
        let (name, state) = conversations.conversation(
            named: NFKMLXConversation.name(of: request), reuses: NFKMLXConversation.reuses(request), prompt: key,
            signature: NFKMLXHybridConversation.optionsSignature(window: nil, quantization: nil, adapterStrengths: nil),
            usable: { $0.layerCount == layerCount },
            make: { NFKMLXPromptCache(layerCount: layerCount) },
            read: { url, record in
                let restored = try NFKMLXPromptCache.load(from: url)
                return restored.tokens == record.tokens && restored.layerCount == layerCount ? restored : nil
            })
        var failed = true
        defer {
            conversations.finish(name, failed: failed, afforded: NFKMLXConversationBudget.afforded(
                info: modelInfo, runBytes: NFKMLXConversationBudget.runBytes(info: modelInfo, positions: nil)), spills: true)
        }
        let answer = try model.answer(messages: messages, images: images, audios: audios, options: options,
                                      promptCache: state) { token in
            produced.append(token)
            return onToken?(produced) ?? true
        }
        failed = false
        return NFKInferenceResult(outputs: [
            NFKOutputText: model.decode(answer.ids),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: answer.promptTokens, cachedTokens: answer.cached,
                                                outputTokens: answer.ids.count, reasoningTokens: nil),
        ])
    }

    private static func options(from request: NFKInferenceRequest) -> NFKMLXGenerationOptions {
        var options = NFKMLXGenerationOptions()
        if let value = request.parameter(forKey: NFKParameterTemperature) as? NSNumber { options.temperature = value.floatValue }
        if let value = request.parameter(forKey: NFKParameterTopP) as? NSNumber { options.topP = value.floatValue }
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber { options.maxTokens = value.intValue }
        if let value = request.parameter(forKey: NFKParameterSeed) as? NSNumber { options.seed = value.uint64Value }
        return options
    }

    /// `NFKInputImage`, then each of `NFKInputImages`, prepared.
    private static func images(from request: NFKInferenceRequest) throws -> [NFKMLXPhi4MMImageInput] {
        var values = [Any]()
        if let value = request.input(forKey: NFKInputImage) { values.append(value) }
        if let more = request.input(forKey: NFKInputImages) as? [Any] { values += more }
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        return try values.map { value in
            let rgb = try NFKMLXImageBridge.rgbBytes(from: value, colorSpace: colorSpace)
            return NFKMLXPhi4MMImageProcessor.process(rgb: rgb.bytes, width: rgb.width, height: rgb.height)
        }
    }

    /// `NFKInputAudio`, then each of `NFKInputAudios`, at their own rates.
    private static func audios(from request: NFKInferenceRequest) throws -> [NFKMLXPhi4MMAudioInput] {
        var values = [Any]()
        if let value = request.input(forKey: NFKInputAudio) { values.append(value) }
        if let more = request.input(forKey: NFKInputAudios) as? [Any] { values += more }
        return try values.map { value in
            var data = value as? Data
            if let asset = value as? NFKAudioAsset, let url = asset.fileURL { data = try Data(contentsOf: url) }
            guard let data, let clip = NFKMLXWaveFile.read(data) else {
                throw NFKMLXError.unsupportedConfiguration("a Phi-4-multimodal clip is an NFKAudioAsset file or WAV bytes")
            }
            return NFKMLXPhi4MMAudioInput(samples: clip.samples, sampleRate: clip.sampleRate)
        }
    }

    // MARK: Conversation caches

    /// The bytes the conversations' caches may hold together.
    ///
    /// @discussion A request continues the conversation its `NFKParameterConversationKey` names, or, naming
    /// none, the kept conversation its prompt continues furthest, so a follow-up turn prefills only what it
    /// adds. When the caches outgrow the budget, the least recently used conversations are written to
    /// ``conversationSpillDirectoryURL`` and dropped, never the one a request just ran. Until it is set, the
    /// budget is what the memory planned beside the model affords after one run's cache at the context
    /// length, recomputed each run; setting it fixes it, and lowering it drops down to it at once. Introduced
    /// in InferKit 0.4.0.
    @objc public var conversationCacheByteBudget: Int {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            return conversations.states.byteBudget
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.fixBudget(newValue)
        }
    }

    /// How many conversations hold a cache. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheCount: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return conversations.states.count
    }

    /// The bytes the conversations' caches occupy. Introduced in InferKit 0.4.0.
    @objc public var conversationCacheBytes: Int {
        generationLock.lock(); defer { generationLock.unlock() }
        return conversations.states.heldBytes
    }

    /// Drops one named conversation's cache, in memory and on disk. Introduced in InferKit 0.4.0.
    @objc(resetPromptCacheForConversation:)
    public func resetPromptCache(forConversation conversation: String) {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.states.remove(conversation)
        conversations.spill?.remove(named: conversation)
        conversations.refresh()
    }

    /// Drops every conversation's cache in memory; spilled ones stay on disk. Introduced in InferKit 0.4.0.
    @objc public func resetPromptCache() {
        generationLock.lock(); defer { generationLock.unlock() }
        conversations.states.removeAll()
        conversations.refresh()
    }

    /// Writes every conversation's cache to the spill directory and drops it from memory when no run is in
    /// progress, and returns the bytes they held; returns 0 at once while a run holds them.
    /// `NFKInferenceServer` calls it to relieve memory, and the reloadable backend before it unloads the model.
    /// Introduced in InferKit 0.4.0.
    @objc public func releaseIdleConversationCaches() -> Int {
        guard generationLock.try() else { return 0 }
        defer { generationLock.unlock() }
        return conversations.releaseAll(spills: adapterStrengths.isEmpty)
    }

    /// The directory conversations the memory budget lets go are written to, so a later request continues one
    /// without its prefill, after an unload or a restart too; nil writes none. A backend over a release
    /// directory defaults to a directory under the user's caches that only the same release and weights read.
    /// A model that holds adapters writes none. Introduced in InferKit 0.4.0.
    @objc public var conversationSpillDirectoryURL: URL? {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            return conversations.spill?.directory
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.setSpillDirectory(newValue)
        }
    }

    /// The bytes the spilled conversations may take on disk; the least recently written go first past it.
    /// Defaults to 8 GiB. Introduced in InferKit 0.4.0.
    @objc public var conversationSpillByteBudget: Int {
        get {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            return conversations.spill?.byteBudget ?? NFKMLXConversationSpill.defaultByteBudget
        }
        set {
            generationLock.lock(); defer { generationLock.unlock() }
            conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
            conversations.spill?.byteBudget = newValue
        }
    }

    /// The conversation caches as of the last run or change: `conversation_caches`,
    /// `conversation_cache_bytes`, and `conversation_cache_byte_budget`. `NFKInferenceServer` serves it under
    /// the model's "status". It never waits on a run in progress. Introduced in InferKit 0.4.0.
    @objc public var backendStatus: [String: Any] { conversations.status }
}

/// Building Phi-4-multimodal from a release directory.
public extension NFKMLXPhi4MM {

    /// The Hugging Face release the download factories fetch by default.
    @objc static let releaseRepo = "microsoft/Phi-4-multimodal-instruct"

    internal static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    internal static let optionalFiles = ["vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
                                         "generation_config.json"]
    internal static let weightFiles = ["model.safetensors.index.json"]

    /// The model from a release directory: the mixture-of-LoRAs decoder, both towers, and the tokenizer.
    /// `precision` applies to the decoder, which holds most of the parameters: `.checkpoint` keeps the
    /// released bfloat16, which a 32 GB machine holds comfortably, and `.float32` is what the parity tests
    /// measure. The towers (under a billion parameters together) always run at float32.
    static func model(directoryURL: URL, precision: NFKMLXWeightPrecision) throws -> NFKMLXPhi4MMModel {
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Phi-4-multimodal release has no readable tokenizer")
        }
        try NFKMLXReleaseWeights.verifyFits(inDirectory: directoryURL, precision: precision)
        let decoder = try mixtureDecoder(directoryURL: directoryURL, precision: precision)
        return NFKMLXPhi4MMModel(decoder: decoder, imageNet: try imageNet(directoryURL: directoryURL),
                                 audioNet: try audioNet(directoryURL: directoryURL), tokenizer: tokenizer)
    }

    /// A multimodal backend from a release directory at the given precision. Run inference off the render
    /// thread.
    @objc(backendWithDirectoryURL:precision:error:)
    static func backend(directoryURL: URL, precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        NFKMLXPhi4MMBackend(model: try model(directoryURL: directoryURL, precision: precision), identifier: modelName,
                            releaseDirectoryURL: directoryURL)
    }

    /// A multimodal backend from a release directory, at the released bfloat16.
    @objc(backendWithDirectoryURL:error:)
    static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: .checkpoint)
    }

    /// Downloads a release (``releaseRepo`` is the published one, about 11 GB) into the hub cache and
    /// builds the backend. Blocking on the network.
    @objc(backendWithRepo:revision:cacheDirectoryURL:precision:error:)
    static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                        precision: NFKMLXWeightPrecision) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles), precision: precision)
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:precision:)``.
    @objc(backendWithRepo:revision:cacheDirectoryURL:precision:completionHandler:)
    static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?, precision: NFKMLXWeightPrecision,
                        completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, precision: precision)
        }
    }

    /// Registers `phi-4-multimodal` with `NFKMLXModelRegistry`; the registry's URL is the release
    /// directory, loaded at the released bfloat16.
    @objc static func register() {
        NFKMLXModelRegistry.register(name: modelName) { url in
            guard let url else {
                throw NFKMLXError.unsupportedConfiguration("phi-4-multimodal builds from a release directory, not without weights")
            }
            return try backend(directoryURL: url, precision: .checkpoint)
        }
    }
}
