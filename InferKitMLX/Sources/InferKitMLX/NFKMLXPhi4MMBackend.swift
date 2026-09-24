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
    /// the reference processor submits them. The audio projector's head follows the mode.
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
            features.append((NFKMLXPhi4MM.imageTokenId, concatenated(embedded, axis: 0)))
        }
        var audioTokens = [Int]()
        if !audios.isEmpty {
            let mels = try audios.map { try NFKMLXPhi4MMAudioFeatures.logMel($0.samples, sampleRate: $0.sampleRate) }
            audioTokens = mels.map { NFKMLXPhi4MMAudioFeatures.tokenCount(frames: $0.dim(0)) }
            features.append((NFKMLXPhi4MM.audioTokenId,
                             concatenated(audioNet.projected(clips: mels, mode: mode), axis: 0)))
        }
        let ids = try promptIds(messages: messages, imageTokens: images.map(\.tokenCount), audioTokens: audioTokens)
        return (ids, features)
    }

    /// One user turn holding `text`, with at most one picture and one clip.
    func prepare(text: String, image: NFKMLXPhi4MMImageInput?, audio: [Float]?, sampleRate: Int = 16000) throws
        -> (ids: [Int], features: [(placeholder: Int, features: MLXArray)]) {
        try prepare(messages: [["role": "user", "content": text]], images: image.map { [$0] } ?? [],
                    audios: audio.map { [NFKMLXPhi4MMAudioInput(samples: $0, sampleRate: sampleRate)] } ?? [])
    }

    /// The answer's token ids to a conversation, stopping at the turn end. `options` sets the sampling
    /// (greedy by default, as the release's generation config is), the token budget, and the seed;
    /// `onToken` sees each id as it is produced, and returning false stops the run.
    public func generate(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput] = [],
                         audios: [NFKMLXPhi4MMAudioInput] = [],
                         options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                         onToken: ((Int) -> Bool)? = nil) throws -> [Int] {
        try answer(messages: messages, images: images, audios: audios, options: options, onToken: onToken).ids
    }

    /// The answer's ids and the prompt's length, which the backend reports as usage.
    func answer(messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput], audios: [NFKMLXPhi4MMAudioInput],
                options: NFKMLXGenerationOptions, onToken: ((Int) -> Bool)?) throws -> (ids: [Int], promptTokens: Int) {
        lock.lock(); defer { lock.unlock() }
        let prepared = try prepare(messages: messages, images: images, audios: audios)
        let ids = NFKMLXPhi4MM.generateFused(decoder: decoder, inputIds: prepared.ids, features: prepared.features,
                                             options: options, endTokens: NFKMLXPhi4MM.stopTokens, onToken: onToken)
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

    init(model: NFKMLXPhi4MMModel, identifier: String) {
        self.model = model
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }
    @objc public var supportedInputKeys: Set<String> {
        [NFKInputPrompt, NFKInputMessages, NFKInputImage, NFKInputImages, NFKInputAudio, NFKInputAudios]
    }
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterTemperature, NFKParameterTopP, NFKParameterMaxTokens, NFKParameterSeed]
    }

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

    private func run(_ request: NFKInferenceRequest, onToken: (([Int]) -> Bool)?) throws -> NFKInferenceResult {
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        let options = Self.options(from: request)
        let images = try Self.images(from: request)
        let audios = try Self.audios(from: request)
        let messages: [[String: Any]]
        if let conversation = request.messages, !conversation.isEmpty {
            messages = conversation
        } else if let prompt = request.prompt, !prompt.isEmpty {
            messages = [["role": "user", "content": prompt]]
        } else if !audios.isEmpty, images.isEmpty {
            messages = [["role": "user", "content": Self.transcriptionPrompt]]
        } else if !images.isEmpty {
            messages = [["role": "user", "content": ""]]
        } else {
            throw NFKMLXError.unsupportedInput
        }

        var produced = [Int]()
        let answer = try model.answer(messages: messages, images: images, audios: audios, options: options) { token in
            produced.append(token)
            return onToken?(produced) ?? true
        }
        return NFKInferenceResult(outputs: [
            NFKOutputText: model.decode(answer.ids),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: answer.promptTokens, cachedTokens: 0,
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
        NFKMLXPhi4MMBackend(model: try model(directoryURL: directoryURL, precision: precision), identifier: modelName)
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
