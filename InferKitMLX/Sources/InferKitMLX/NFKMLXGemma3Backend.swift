//
//  NFKMLXGemma3Backend.swift
//  InferKitMLX
//
//  A Gemma 3 release as an InferKit backend: text in, text out, and in a multimodal release an image
//  beside the text. The decoder generates through a hybrid key-value cache (one unbounded for the full
//  layers, one bounded to the window for the sliding ones), so a step costs one token's work rather
//  than the whole sequence's.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

/// Holds the model across the async job boundary. `MLXArray` and the modules are not `Sendable`; the
/// backend runs generation on a background task, so the model is carried through an unchecked holder,
/// as the core language backend does.
final class NFKGemma3Holder: @unchecked Sendable {
    let model: NFKMLXGemma3Model
    init(_ model: NFKMLXGemma3Model) { self.model = model }
}

/// A cancellation flag a job's handler sets and the generation loop reads between tokens.
final class NFKGemma3CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
}

/// Text generation, and image understanding, over a Gemma 3 release.
///
/// Reads `NFKInputPrompt` (a raw prompt, encoded after `<bos>`) or `NFKInputMessages` (rendered through
/// the release's own chat template), plus an optional `NFKInputImage` (a `CGImage`, `CVPixelBuffer`, or
/// `MTLTexture`) in a multimodal release — the image's 256 soft tokens are placed before the text, as
/// the reference processor places them. Returns `NFKOutputText`. `NFKParameterTemperature`,
/// `NFKParameterTopP`, `NFKParameterMaxTokens`, and `NFKParameterSeed` override the defaults. A
/// submitted job reports each token through `partialResult` and honors cancellation between tokens.
@objc(NFKMLXGemma3Backend)
public final class NFKMLXGemma3Backend: NSObject, NFKInferenceBackend {
    private let holder: NFKGemma3Holder
    private let identifier: String
    /// Generation is serialized: two runs through one set of networks would interleave their work.
    private let generationLock = NSLock()

    init(model: NFKMLXGemma3Model, identifier: String) {
        holder = NFKGemma3Holder(model)
        self.identifier = identifier
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    /// Whether the release carries a vision tower, so a request may attach `NFKInputImage`.
    @objc public var acceptsImages: Bool { holder.model.acceptsImages }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try run(request, onToken: nil)
    }

    private func run(_ request: NFKInferenceRequest, onToken: ((Int, [Int]) -> Bool)?) throws -> NFKInferenceResult {
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
        let model = holder.model
        let image = request.input(forKey: NFKInputImage)
        if image != nil, !model.acceptsImages {
            throw NFKMLXError.unsupportedConfiguration(
                "\(identifier) is a text-only release; an image needs a multimodal Gemma 3 (4B and up)")
        }
        let ids: [Int]
        if let prompt = request.prompt {
            ids = model.promptTokens(prompt, withImage: image != nil)
        } else if let messages = request.messages {
            ids = model.chatTokens(messages: messages, withImage: image != nil)
        } else {
            throw NFKMLXError.unsupportedInput
        }

        generationLock.lock(); defer { generationLock.unlock() }
        var produced = [Int]()
        try model.generate(tokens: ids, image: image, options: options) { token in
            produced.append(token)
            return onToken?(token, produced) ?? true
        }
        return NFKInferenceResult(outputs: [NFKOutputText: model.decode(produced)])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let flag = NFKGemma3CancelFlag()
        job.cancellationHandler = { flag.cancel() }
        let maximum = max((request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber)?.intValue ?? 256, 1)
        Task.detached(priority: .userInitiated) { [self] in
            do {
                let result = try run(request) { _, produced in
                    let partial = NFKInferenceResult(outputs: [NFKOutputText: self.holder.model.decode(produced)])
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
}

/// Building Gemma 3 from a release directory: the model object, and the backend around it.
///
/// `NFKMLXGemma3` is a released `google/gemma-3-*` directory (through the ungated `unsloth/gemma-3-*`
/// mirrors): the text decoder for every size, and the SigLIP vision tower and projector for the
/// multimodal 4B / 12B / 27B. Run inference off the render thread.
@objc(NFKMLXGemma3)
public final class NFKMLXGemma3: NSObject {

    /// The registry name a Gemma 3 backend reports.
    @objc public static let modelName = "gemma3"

    private let holder: NFKGemma3Holder

    /// The loaded model.
    public var model: NFKMLXGemma3Model { holder.model }

    init(model: NFKMLXGemma3Model) {
        holder = NFKGemma3Holder(model)
        super.init()
    }

    /// Loads a release: `config.json`, the weights (single-file or sharded), `tokenizer.json`, and the
    /// chat template. A multimodal release's vision tower and projector load beside the decoder.
    ///
    /// - Parameter precision: `.float32` (the default, what the parity records were measured at) or
    ///   `.checkpoint` to keep the released bf16, which halves the memory.
    public static func load(directoryURL directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXGemma3 {
        NFKMLXGemma3(model: try model(directoryURL: directory, precision: precision))
    }

    /// The Objective-C entry: loads a release directory.
    @objc(gemma3WithDirectoryURL:error:)
    public static func gemma3(directoryURL: URL) throws -> NFKMLXGemma3 {
        try load(directoryURL: directoryURL)
    }

    /// Answers a question about an image (greedy, through the chat template). The release must be a
    /// multimodal one.
    @objc(answerForImage:question:error:)
    public func answer(image: CGImage, question: String) throws -> String {
        try holder.model.answer(image: image, question: question)
    }

    /// Answers a plain question (greedy, through the chat template).
    @objc(answerForQuestion:error:)
    public func answer(question: String) throws -> String {
        try holder.model.answer(image: nil, question: question)
    }

    /// Builds a backend from a release directory.
    public static func backend(directoryURL directory: URL,
                               precision: NFKMLXWeightPrecision = .float32) throws -> any NFKInferenceBackend {
        NFKMLXGemma3Backend(model: try model(directoryURL: directory, precision: precision), identifier: modelName)
    }

    /// The Objective-C entry: builds a backend from a release directory.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, precision: .float32)
    }

    /// Whether a release's `config.json` names a Gemma 3 (`gemma3` or `gemma3_text`).
    static func isGemma3(configURL: URL) -> Bool {
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        let kind = (json["model_type"] as? String) ?? ""
        return kind == "gemma3" || kind == "gemma3_text"
    }

    /// The model object for a release directory, every part loaded.
    public static func model(directoryURL directory: URL,
                             precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXGemma3Model {
        let parts = try load(directory: directory, precision: precision, decoder: true)
        guard let decoder = parts.decoder else { throw NFKMLXError.noOutput }
        return NFKMLXGemma3Model(decoder: decoder, vision: parts.vision, projector: parts.projector,
                                 tokenizer: parts.tokenizer, tokens: parts.tokens,
                                 chatTemplate: chatTemplate(inDirectory: directory))
    }

    /// The vision tower and projector of a multimodal release alone, the decoder left on disk — what
    /// a test of the vision path loads.
    static func visionParts(directoryURL directory: URL, precision: NFKMLXWeightPrecision = .float32)
        throws -> (vision: NFKMLXGemma3VisionNet, projector: NFKMLXGemma3MultimodalProjector) {
        let parts = try load(directory: directory, precision: precision, decoder: false)
        guard let vision = parts.vision, let projector = parts.projector else {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) carries no vision tower")
        }
        return (vision, projector)
    }

    /// Everything a release directory describes, its weights read once and partitioned by prefix.
    /// `decoder` false leaves the language model unloaded (its weights are most of the file).
    static func load(directory: URL, precision: NFKMLXWeightPrecision, decoder wantsDecoder: Bool) throws
        -> (decoder: NFKMLXGemma3Net?, vision: NFKMLXGemma3VisionNet?, projector: NFKMLXGemma3MultimodalProjector?,
            tokenizer: NFKMLXGemmaTokenizer, tokens: NFKMLXGemma3Tokens) {
        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("config.json is not a JSON object")
        }
        let textConfiguration = try NFKMLXGemma3Language.configuration(fromJSON: json)
        guard let tokenizer = NFKMLXGemmaTokenizer(directoryURL: directory) else {
            throw NFKMLXError.unsupportedConfiguration("the Gemma 3 release has no readable tokenizer.json")
        }
        if wantsDecoder {
            try NFKMLXReleaseWeights.verifyFits(inDirectory: directory, precision: precision)
        }

        let decoder = wantsDecoder ? NFKMLXGemma3Net(textConfiguration) : nil

        var vision: NFKMLXGemma3VisionNet?
        var projector: NFKMLXGemma3MultimodalProjector?
        var tokensPerImage = 256
        if let visionJSON = json["vision_config"] as? [String: Any] {
            func integer(_ key: String, _ fallback: Int) -> Int { (visionJSON[key] as? NSNumber)?.intValue ?? fallback }
            let visionConfiguration = NFKMLXSigLIPConfiguration(
                hiddenSize: integer("hidden_size", 1152), layerCount: integer("num_hidden_layers", 27),
                headCount: integer("num_attention_heads", 16), intermediateSize: integer("intermediate_size", 4304),
                patchSize: integer("patch_size", 14), imageSize: integer("image_size", 896),
                layerNormEpsilon: (visionJSON["layer_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6)
            tokensPerImage = (json["mm_tokens_per_image"] as? NSNumber)?.intValue ?? 256
            vision = NFKMLXGemma3VisionNet(visionConfiguration)
            projector = NFKMLXGemma3MultimodalProjector(
                visionHidden: visionConfiguration.hiddenSize, textHidden: textConfiguration.hiddenSize,
                patchesPerSide: visionConfiguration.grid, tokensPerImage: tokensPerImage,
                eps: visionConfiguration.layerNormEpsilon)
        }

        // Every shard is read once and partitioned by prefix; a 4-D vision convolution weight takes
        // the PyTorch → MLX channel transpose. A tensor nobody wants is skipped before it is converted.
        var decoderWeights = [(String, MLXArray)]()
        var visionWeights = [(String, MLXArray)]()
        var projectorWeights = [(String, MLXArray)]()
        let wanted: (String) -> String? = { key in
            if NFKMLXGemma3Language.decoderName(of: key) != nil { return wantsDecoder ? key : nil }
            if visionName(of: key) != nil || projectorName(of: key) != nil { return vision != nil ? key : nil }
            return nil
        }
        for (key, value) in try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision, remap: wanted) {
            if let name = NFKMLXGemma3Language.decoderName(of: key) {
                decoderWeights.append((name, value))
            } else if let name = visionName(of: key) {
                visionWeights.append((name, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value))
            } else if let name = projectorName(of: key) {
                projectorWeights.append((name, value))
            }
        }
        if let decoder {
            try NFKMLXWeights.apply(decoderWeights, to: decoder)
        }
        if let vision, let projector {
            try NFKMLXWeights.apply(visionWeights, to: vision)
            try NFKMLXWeights.apply(projectorWeights, to: projector)
        }

        let tokens = markerTokens(json: json, tokenizer: tokenizer, tokensPerImage: tokensPerImage)
        return (decoder, vision, projector, tokenizer, tokens)
    }

    /// The vision tower's module key for a checkpoint key, or nil for a tensor that is not the tower's.
    /// A release written by transformers 4.x names it `vision_tower.vision_model.`; 5.x nests it under
    /// `model.`.
    static func visionName(of key: String) -> String? {
        NFKMLXGemma3Language.stripped(key, prefixes: ["model.vision_tower.vision_model.", "vision_tower.vision_model."])
    }

    /// The projector's module key for a checkpoint key, or nil.
    static func projectorName(of key: String) -> String? {
        NFKMLXGemma3Language.stripped(key, prefixes: ["model.multi_modal_projector.", "multi_modal_projector."])
    }

    /// The marker ids: the tokenizer's added tokens first, the config's indices as the fallback.
    static func markerTokens(json: [String: Any], tokenizer: NFKMLXGemmaTokenizer,
                             tokensPerImage: Int) -> NFKMLXGemma3Tokens {
        var tokens = NFKMLXGemma3Tokens(tokensPerImage: tokensPerImage)
        func configured(_ key: String) -> Int? { (json[key] as? NSNumber)?.intValue }
        tokens.beginOfSequence = tokenizer.id(forToken: "<bos>") ?? configured("bos_token_id") ?? tokens.beginOfSequence
        tokens.endOfSequence = tokenizer.id(forToken: "<eos>") ?? tokens.endOfSequence
        tokens.startOfTurn = tokenizer.id(forToken: "<start_of_turn>") ?? tokens.startOfTurn
        tokens.endOfTurn = tokenizer.id(forToken: "<end_of_turn>") ?? tokens.endOfTurn
        tokens.startOfImage = tokenizer.id(forToken: "<start_of_image>") ?? configured("boi_token_index") ?? tokens.startOfImage
        tokens.endOfImage = tokenizer.id(forToken: "<end_of_image>") ?? configured("eoi_token_index") ?? tokens.endOfImage
        tokens.imageSoftToken = tokenizer.id(forToken: "<image_soft_token>") ?? configured("image_token_index") ?? tokens.imageSoftToken
        tokens.padToken = tokenizer.id(forToken: "<pad>") ?? configured("pad_token_id") ?? tokens.padToken
        return tokens
    }

    /// The release's chat template: `chat_template.jinja` beside the weights, else the
    /// `chat_template` string in `tokenizer_config.json`, else nil.
    static func chatTemplate(inDirectory directory: URL) -> String? {
        if let text = try? String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8),
           !text.isEmpty {
            return text
        }
        if let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let template = json["chat_template"] as? String {
            return template
        }
        return nil
    }
}
