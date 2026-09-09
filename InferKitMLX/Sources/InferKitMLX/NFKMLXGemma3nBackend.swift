//
//  NFKMLXGemma3nBackend.swift
//  InferKitMLX
//
//  Text generation from Gemma 3n, with a picture or a clip in the prompt.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN
import MLXRandom

/// The networks cross the async job boundary here; an `MLXArray` is not `Sendable`.
final class NFKGemma3nHolder: @unchecked Sendable {
    let model: NFKMLXGemma3n
    init(_ model: NFKMLXGemma3n) { self.model = model }
}

final class NFKGemma3nCancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
}

// MARK: - Prompting and generation

extension NFKMLXGemma3n {
    /// The text a picture occupies in a prompt: the release processor's own
    /// `\n\n<start_of_image>` + soft-token run + `<end_of_image>\n\n`.
    var imageSequence: String {
        "\n\n<start_of_image>"
            + String(repeating: "<image_soft_token>", count: model.tokens.visionSoftTokens)
            + "<end_of_image>\n\n"
    }

    /// The text a clip occupies in a prompt.
    var audioSequence: String {
        "\n\n<start_of_audio>"
            + String(repeating: "<audio_soft_token>", count: model.tokens.audioSoftTokens)
            + "<end_of_audio>\n\n"
    }

    var endOfTurn: Int { tokenizer.id(forToken: "<end_of_turn>") ?? 106 }
    var endOfSequence: Int { tokenizer.id(forToken: "<eos>") ?? 1 }

    /// One user turn, with the media placed ahead of the question the way the release's processor
    /// places it.
    ///
    /// @discussion The whole turn is built as TEXT and tokenized once, which is load-bearing: the
    /// `\n` closing `user` and the `\n\n` opening the image run merge into ONE id, and a prompt
    /// assembled from separately encoded pieces reads a different sentence to the model. Gemma 3 has
    /// the same trap.
    public func promptTokens(_ prompt: String, withImage image: Bool = false,
                             withAudio audio: Bool = false) -> [Int] {
        var text = "<bos><start_of_turn>user\n"
        if image { text += imageSequence }
        if audio { text += audioSequence }
        text += prompt + "<end_of_turn>\n<start_of_turn>model\n"
        return tokenizer.encode(text)
    }

    /// A conversation rendered into Gemma's turns. A system message opens the first user turn, which
    /// is where Gemma's own template puts it.
    public func chatTokens(messages: [[String: Any]], withImage image: Bool = false,
                           withAudio audio: Bool = false) -> [Int] {
        var text = "<bos>"
        var pending = ""
        var placed = false
        for message in messages {
            let role = (message["role"] as? String) ?? "user"
            let content = (message["content"] as? String) ?? ""
            if role == "system" {
                pending = content + "\n\n"
                continue
            }
            text += "<start_of_turn>\(role == "assistant" ? "model" : "user")\n"
            if role != "assistant", !placed {
                if image { text += imageSequence }
                if audio { text += audioSequence }
                placed = true
            }
            text += pending + content + "<end_of_turn>\n"
            pending = ""
        }
        text += "<start_of_turn>model\n"
        return tokenizer.encode(text)
    }

    /// The text for a run of generated ids, with the markers left out.
    public func decode(_ ids: [Int]) -> String { tokenizer.decode(ids, skipSpecial: true) }

    /// Generates a continuation, calling `onToken` with each id and stopping when it answers false.
    public func generate(tokens: [Int], image: Any? = nil, audio: MLXArray? = nil,
                         options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                         onToken: (Int) -> Bool) throws {
        let pixels = try image.map { try NFKMLXGemma3nImageProcessor.pixelValues(from: $0) }
        let mel = audio.map { NFKMLXGemma3nAudioFeatures()($0) }
        if let seed = options.seed { MLXRandom.seed(seed) }

        let cache = NFKMLXGemma3nCache(layerCount: model.decoder.configuration.layerCount)
        var logits = try model(MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count]),
                               image: pixels, audioMel: mel, cache: cache)[0, -1]
        let stops: Set<Int> = [endOfTurn, endOfSequence]

        for _ in 0 ..< max(options.maxTokens, 1) {
            let next = NFKMLXLanguageNet.sample(logits, options: options)
            if stops.contains(next) { break }
            if !onToken(next) { break }
            logits = try model(MLXArray([Int32(next)]).reshaped([1, 1]), cache: cache)[0, -1]
        }
    }
}

// MARK: - The backend

/// Text generation from Gemma 3n through the InferKit contract.
///
/// @discussion `NFKInputImage` adds a picture and `NFKInputAudio` a clip; either needs a release that
/// carries the matching tower. Inference is synchronous and multi-second, so a caller runs it off the
/// render thread or takes the job form.
@objc(NFKMLXGemma3nBackend)
public final class NFKMLXGemma3nBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKGemma3nHolder
    private let identifier: String
    private let generationLock = NSLock()

    init(model: NFKMLXGemma3n, identifier: String) {
        holder = NFKGemma3nHolder(model)
        self.identifier = identifier
        super.init()
    }

    @objc public var backendIdentifier: String { identifier }
    @objc public var isReady: Bool { true }
    @objc public var acceptsImages: Bool { holder.model.model.vision != nil }
    @objc public var acceptsAudio: Bool { holder.model.model.audio != nil }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try run(request, onToken: nil)
    }

    private func run(_ request: NFKInferenceRequest, onToken: ((Int, [Int]) -> Bool)?) throws
        -> NFKInferenceResult {
        var options = NFKMLXGenerationOptions()
        if let value = request.parameter(forKey: NFKParameterTemperature) as? NSNumber {
            options.temperature = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterTopP) as? NSNumber {
            options.topP = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            options.maxTokens = value.intValue
        }
        if let value = request.parameter(forKey: NFKParameterSeed) as? NSNumber {
            options.seed = value.uint64Value
        }

        let gemma = holder.model
        let image = request.input(forKey: NFKInputImage)
        if image != nil, !acceptsImages {
            throw NFKMLXError.unsupportedConfiguration("\(identifier) carries no vision tower")
        }
        var waveform: MLXArray?
        if let clip = NFKMLXGemma3nBackend.audio(from: request) {
            guard acceptsAudio else {
                throw NFKMLXError.unsupportedConfiguration("\(identifier) carries no audio tower")
            }
            // The encoder is trained at 16 kHz; a clip arriving at another rate is resampled, which
            // preserves its duration.
            waveform = MLXArray(NFKMLXAudioRate.matched(clip.samples, from: clip.sampleRate, to: 16_000))
        }

        let ids: [Int]
        if let prompt = request.prompt {
            ids = gemma.promptTokens(prompt, withImage: image != nil, withAudio: waveform != nil)
        } else if let messages = request.messages {
            ids = gemma.chatTokens(messages: messages, withImage: image != nil, withAudio: waveform != nil)
        } else {
            throw NFKMLXError.unsupportedInput
        }

        generationLock.lock(); defer { generationLock.unlock() }
        var produced = [Int]()
        try gemma.generate(tokens: ids, image: image, audio: waveform, options: options) { token in
            produced.append(token)
            return onToken?(token, produced) ?? true
        }
        return NFKInferenceResult(outputs: [NFKOutputText: gemma.decode(produced)])
    }

    /// The samples an `NFKAudioAsset` or raw WAV bytes carry.
    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL,
           let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data { return NFKMLXWaveFile.read(data) }
        return nil
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let flag = NFKGemma3nCancelFlag()
        job.cancellationHandler = { flag.cancel() }
        let maximum = max((request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber)?.intValue ?? 256, 1)
        Task.detached(priority: .userInitiated) { [self] in
            do {
                let result = try run(request) { _, produced in
                    let partial = NFKInferenceResult(
                        outputs: [NFKOutputText: self.holder.model.decode(produced)])
                    job.reportProgress(min(Double(produced.count) / Double(maximum), 0.99),
                                       partialResult: partial)
                    return !flag.isCancelled
                }
                if !flag.isCancelled { job.finish(with: result) }
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

// MARK: - Building from a release

extension NFKMLXGemma3n {
    /// The registry name the reference models register this under.
    @objc public static let modelName = "gemma3n"

    /// Reads a released `google/gemma-3n-*` directory — the E2B and E4B releases, and the ungated
    /// `unsloth/gemma-3n-*` mirrors of them.
    public static func load(directoryURL directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXGemma3n {
        let networks = try model(directoryURL: directory, precision: precision)
        guard let tokenizer = NFKMLXGemmaTokenizer(directoryURL: directory) else {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) carries no tokenizer.json")
        }
        return NFKMLXGemma3n(model: networks, tokenizer: tokenizer)
    }

    @objc(gemma3nWithDirectoryURL:error:)
    public static func gemma3n(directoryURL: URL) throws -> NFKMLXGemma3n {
        try load(directoryURL: directoryURL)
    }

    /// The model's answer about a picture.
    @objc(answerForImage:question:error:)
    public func answer(image: CGImage, question: String) throws -> String {
        var produced = [Int]()
        try generate(tokens: promptTokens(question, withImage: true), image: image) {
            produced.append($0); return produced.count < 256
        }
        return decode(produced)
    }

    /// The model's answer to a question with no media.
    @objc(answerForQuestion:error:)
    public func answer(question: String) throws -> String {
        var produced = [Int]()
        try generate(tokens: promptTokens(question)) { produced.append($0); return produced.count < 256 }
        return decode(produced)
    }

    /// A text-generation backend around a released directory.
    public static func backend(directoryURL directory: URL,
                               precision: NFKMLXWeightPrecision = .float32) throws
        -> any NFKInferenceBackend {
        NFKMLXGemma3nBackend(model: try load(directoryURL: directory, precision: precision),
                             identifier: modelName)
    }

    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: .float32)
    }

    /// Whether a release directory's `config.json` names Gemma 3n.
    public static func isGemma3n(configURL: URL) -> Bool {
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        let outer = (json["model_type"] as? String) ?? ""
        return outer == "gemma3n" || outer == "gemma3n_text"
    }
}
