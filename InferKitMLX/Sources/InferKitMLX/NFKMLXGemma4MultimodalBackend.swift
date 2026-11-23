//
//  NFKMLXGemma4MultimodalBackend.swift
//  InferKitMLX
//
//  Text generation from a Gemma 4 release with a picture or a clip in the prompt: the release's vision
//  and audio towers, their embedders, and the E-series or mixture decoder, joined through
//  `NFKMLXGemma4ConditionalGeneration`.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

/// A Gemma 4 release's towers and decoder, assembled for the multimodal backend.
final class NFKGemma4MultimodalModel: @unchecked Sendable {
    let chain: NFKMLXGemma4ConditionalGeneration
    let decoder: NFKMLXGemmaNet
    let tokenizer: NFKMLXGemmaTokenizer
    /// The release's chat template, or nil to build Gemma 4's turns without one.
    let chatTemplate: String?
    /// The positions the release's text configuration states, which bound a prompt and its reply.
    let contextLength: Int?
    /// The sampling the release's generation config sets: step 3 of ``NFKMLXGenerationOptions``' sampling order.
    let releaseSampling: NFKMLXReleaseSampling?

    init(decoder: NFKMLXGemmaNet, vision: (tower: NFKMLXGemma4VisionNet, embedder: NFKMLXGemma4MultimodalEmbedder)?,
         audio: (tower: NFKMLXGemma4AudioNet, embedder: NFKMLXGemma4MultimodalEmbedder)?,
         imageProcessor: NFKMLXGemma4ImageProcessor, tokenizer: NFKMLXGemmaTokenizer,
         imageTokenId: Int, audioTokenId: Int, chatTemplate: String? = nil, contextLength: Int? = nil,
         releaseSampling: NFKMLXReleaseSampling? = nil) {
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.chatTemplate = chatTemplate
        self.contextLength = contextLength
        self.releaseSampling = releaseSampling
        chain = NFKMLXGemma4ConditionalGeneration(
            decoder: decoder, visionTower: vision?.tower, visionEmbedder: vision?.embedder,
            audioTower: audio?.tower, audioEmbedder: audio?.embedder, imageProcessor: imageProcessor,
            imageTokenId: imageTokenId, audioTokenId: audioTokenId,
            padTokenId: tokenizer.id(forToken: "<pad>") ?? 0)
    }

    /// The projected soft tokens of `image`, `[count, hidden]`, one per real pooled cell of the resized
    /// image. See ``NFKMLXGemma4ConditionalGeneration/visionSoftTokens(for:)``.
    func visionSoftTokens(_ image: CGImage) throws -> MLXArray {
        guard let soft = chain.visionSoftTokens(for: image) else {
            throw NFKMLXError.unsupportedConfiguration("the release carries no vision tower")
        }
        return soft
    }

    /// The projected soft tokens of a 16 kHz waveform, `[count, hidden]`.
    func audioSoftTokens(_ waveform: [Float]) throws -> MLXArray {
        guard let soft = chain.audioSoftTokens(for: waveform) else {
            throw NFKMLXError.unsupportedConfiguration("the release carries no audio tower")
        }
        return soft
    }

    /// The prompt as Gemma 4's chat template renders it, with the media placed at the start of the
    /// first user turn the way the release's processor expands it: `<|image>`, one `<|image|>` per soft
    /// token, `<image|>`, and the same for audio. A raw prompt is the one user turn.
    /// The modules whose weights the model holds.
    var modules: [Module] { [decoder] + chain.towerModules }

    /// An image prepared for the tower, with its soft-token count and pixel digest.
    func preparedImage(_ image: CGImage) throws -> (pixels: MLXArray, positions: MLXArray, count: Int, digest: UInt64) {
        guard let prepared = chain.preparedImage(image) else {
            throw NFKMLXError.unsupportedConfiguration("the release carries no vision tower")
        }
        return prepared
    }

    /// A clip's soft tokens and the digest of its features.
    func preparedAudio(_ waveform: [Float]) throws -> (soft: MLXArray, digest: UInt64) {
        guard let prepared = chain.preparedAudio(waveform) else {
            throw NFKMLXError.unsupportedConfiguration("the release carries no audio tower")
        }
        return prepared
    }

    /// The key a conversation keeps `tokens` under: the ids, with the image's and the clip's placeholder
    /// positions keyed by their digests.
    func conversationKey(_ tokens: [Int], image: UInt64?, audio: UInt64?) throws -> [Int] {
        var key = tokens
        for (digest, token) in [(image, chain.placeholderIds.image), (audio, chain.placeholderIds.audio)] {
            guard let digest else { continue }
            key = try NFKMLXVisionChat.mediaKey(inputIds: key, isMedia: { $0 == token },
                                                media: [(digest, key.filter { $0 == token }.count)])
        }
        return key
    }

    func promptText(_ request: NFKInferenceRequest, imageTokens: Int?, audioTokens: Int?) throws -> String {
        var media = ""
        if let imageTokens {
            media += "<|image>" + String(repeating: "<|image|>", count: imageTokens) + "<image|>"
        }
        if let audioTokens {
            media += "<|audio>" + String(repeating: "<|audio|>", count: audioTokens) + "<audio|>"
        }
        let messages: [[String: Any]]
        if let conversation = request.messages {
            messages = conversation
        } else if let prompt = request.prompt {
            messages = [["role": "user", "content": prompt]]
        } else {
            throw NFKMLXError.unsupportedInput
        }
        return try NFKMLXGemmaLanguage.conversationText(messages, template: chatTemplate, mediaPrefix: media)
    }
}

extension NFKMLXGemmaLanguage {

    /// The registry name the multimodal Gemma 4 backend builds under. Introduced in InferKit 0.4.0.
    @objc public static let gemmaMultimodalModelName = "gemma4-multimodal"

    /// The request inputs the multimodal backend reads.
    static let multimodalInputKeys: Set<String> = [NFKInputPrompt, NFKInputMessages, NFKInputImage, NFKInputAudio]

    /// The request parameters the multimodal backend reads.
    static let multimodalParameterKeys: Set<String> = [NFKParameterTemperature, NFKParameterTopP, NFKParameterTopK, NFKParameterMaxTokens,
                                                        NFKParameterSeed, NFKParameterConversationKey]

    /// The samples an `NFKAudioAsset` or raw WAV bytes carry, resampled to 16 kHz, the rate the audio
    /// tower is trained at.
    static func waveform(_ request: NFKInferenceRequest) -> [Float]? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        var clip: (samples: [Float], sampleRate: Int)?
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            clip = NFKMLXWaveFile.read(data)
        } else if let data = value as? Data {
            clip = NFKMLXWaveFile.read(data)
        }
        return clip.map { NFKMLXAudioRate.matched($0.samples, from: $0.sampleRate, to: 16_000) }
    }

    /// The multimodal model behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` or `NFKInputMessages`, a `CGImage` or texture under
    /// `NFKInputImage`, and an `NFKAudioAsset` or WAV bytes under `NFKInputAudio`; an image or a clip
    /// needs a release that carries the matching tower. It returns the reply under `NFKOutputText` and
    /// the token counts under `NFKOutputUsage`. Parameters: `NFKParameterTemperature` (0, greedy),
    /// `NFKParameterMaxTokens` (unset, the room the context leaves), and `NFKParameterSeed`. The prompt, its
    /// soft tokens spliced in, prefills the decoder's key-value cache, and each new token runs alone.
    static func multimodalBackend(_ model: NFKGemma4MultimodalModel, releaseDirectoryURL: URL? = nil) -> any NFKInferenceBackend {
        NFKMLXGemma4MultimodalBackend(model: model, releaseDirectoryURL: releaseDirectoryURL)
    }

    static func visionConfiguration(_ json: [String: Any]) -> NFKMLXGemma4VisionConfiguration {
        let base = NFKMLXGemma4VisionConfiguration()
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        let rope = (json["rope_parameters"] as? [String: Any])?["rope_theta"] as? NSNumber
        return NFKMLXGemma4VisionConfiguration(
            hiddenSize: integer("hidden_size", base.hiddenSize), layerCount: integer("num_hidden_layers", base.layerCount),
            headCount: integer("num_attention_heads", base.headCount),
            keyValueHeadCount: integer("num_key_value_heads", base.keyValueHeadCount),
            headDimensions: integer("head_dim", base.headDimensions),
            intermediateSize: integer("intermediate_size", base.intermediateSize),
            patchSize: integer("patch_size", base.patchSize),
            positionEmbeddingSize: integer("position_embedding_size", base.positionEmbeddingSize),
            poolingKernelSize: integer("pooling_kernel_size", base.poolingKernelSize),
            rmsEpsilon: (json["rms_norm_eps"] as? NSNumber)?.floatValue ?? base.rmsEpsilon,
            ropeTheta: rope?.floatValue ?? base.ropeTheta,
            useClippedLinears: (json["use_clipped_linears"] as? NSNumber)?.boolValue ?? false,
            standardize: (json["standardize"] as? NSNumber)?.boolValue ?? false)
    }

    /// The audio tower geometry a release's `audio_config` states.
    static func audioConfiguration(_ json: [String: Any]) -> NFKMLXGemma4AudioConfiguration {
        let base = NFKMLXGemma4AudioConfiguration()
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func float(_ key: String, _ fallback: Float) -> Float { (json[key] as? NSNumber)?.floatValue ?? fallback }
        return NFKMLXGemma4AudioConfiguration(
            hiddenSize: integer("hidden_size", base.hiddenSize), layerCount: integer("num_hidden_layers", base.layerCount),
            headCount: integer("num_attention_heads", base.headCount),
            convKernelSize: integer("conv_kernel_size", base.convKernelSize),
            chunkSize: integer("attention_chunk_size", base.chunkSize),
            contextLeft: integer("attention_context_left", base.contextLeft),
            contextRight: integer("attention_context_right", base.contextRight),
            residualWeight: float("residual_weight", base.residualWeight),
            gradientClipping: float("gradient_clipping", base.gradientClipping),
            attentionLogitCap: float("attention_logit_cap", base.attentionLogitCap),
            rmsEpsilon: float("rms_norm_eps", base.rmsEpsilon),
            subsampleChannels: (json["subsampling_conv_channels"] as? [Int]) ?? base.subsampleChannels,
            outputProjDims: integer("output_proj_dims", base.outputProjDims),
            useClippedLinears: (json["use_clipped_linears"] as? NSNumber)?.boolValue ?? false)
    }

    /// The release's arrays under `prefix`, renamed by `rename`.
    private static func arrays(inDirectory directory: URL, prefix: String,
                               rename: @escaping (String) -> String = { $0 }) throws -> [(String, MLXArray)] {
        try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, remap: { key in
            key.hasPrefix(prefix) ? rename(String(key.dropFirst(prefix.count))) : nil
        })
    }

    /// Builds the multimodal backend from a Gemma 4 release directory: the decoder, and every tower the
    /// release's `config.json` describes (`vision_config`, `audio_config`) with its embedder.
    ///
    /// @discussion A release without a tower answers a request that needs it with an error. The 12B
    /// unified decoder is refused; ``gemmaBackend(directoryURL:)`` runs it for text. The mixture's
    /// routed experts are held as `residency` says. The prompt renders through the release's chat
    /// template where it ships one. The towers and the decoder load at float32. Run
    /// inference off the render thread. Introduced in InferKit 0.4.0.
    @objc(gemmaMultimodalBackendWithDirectoryURL:residency:error:)
    public static func gemmaMultimodalBackend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try gemmaMultimodalBackend(directoryURL: directoryURL, precision: .float32, residency: residency)
    }

    /// ``gemmaMultimodalBackend(directoryURL:residency:)`` with the decoder at `precision`.
    ///
    /// @discussion A dense decoder (the E-series and the 31B) that does not fit beside the towers streams
    /// the layers it cannot hold under ``NFKMLXResidency/automatic`` and ``NFKMLXResidency/streamed``,
    /// read in their turn on every pass; the plan counts the towers, which load first and stay held. A
    /// release larger than memory loads at its own precision (``NFKMLXWeightPrecision/checkpoint``). The
    /// release directory has to stay in place for the life of a paged or streamed backend. Introduced in
    /// InferKit 0.4.0.
    @objc(gemmaMultimodalBackendWithDirectoryURL:precision:residency:error:)
    public static func gemmaMultimodalBackend(directoryURL: URL, precision: NFKMLXWeightPrecision,
                                              residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        let configURL = directoryURL.appendingPathComponent("config.json")
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("config.json is not a JSON object")
        }
        let text = (json["text_config"] as? [String: Any]) ?? json
        if ((text["model_type"] as? String) ?? "").hasPrefix("gemma4_unified") {
            throw NFKMLXError.unsupportedConfiguration("the unified decoder takes no spliced media; use gemmaBackend for text")
        }
        guard let tokenizer = NFKMLXGemmaTokenizer(directoryURL: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("the Gemma release has no readable tokenizer.json")
        }
        let decoder = makeNet(try configuration(fromHuggingFace: configURL))
        let hidden = decoder.configuration.hiddenSize

        var vision: (tower: NFKMLXGemma4VisionNet, embedder: NFKMLXGemma4MultimodalEmbedder)?
        var imageProcessor = NFKMLXGemma4ImageProcessor()
        if let visionJSON = json["vision_config"] as? [String: Any] {
            let geometry = visionConfiguration(visionJSON)
            let tower = makeVisionNet(geometry)
            try NFKMLXWeights.apply(arrays(inDirectory: directoryURL, prefix: "model.vision_tower.") {
                $0.replacingOccurrences(of: "encoder.layers.", with: "encoder_layers.")
            }, to: tower)
            let embedder = NFKMLXGemma4MultimodalEmbedder(multimodalHidden: geometry.hiddenSize, textHidden: hidden,
                                                          eps: geometry.rmsEpsilon)
            try NFKMLXWeights.apply(arrays(inDirectory: directoryURL, prefix: "model.embed_vision."), to: embedder)
            vision = (tower, embedder)
            imageProcessor = NFKMLXGemma4ImageProcessor(
                patchSize: geometry.patchSize, poolingKernelSize: geometry.poolingKernelSize,
                maxSoftTokens: (json["vision_soft_tokens_per_image"] as? NSNumber)?.intValue ?? 280)
        }
        var audio: (tower: NFKMLXGemma4AudioNet, embedder: NFKMLXGemma4MultimodalEmbedder)?
        if let audioJSON = json["audio_config"] as? [String: Any] {
            let geometry = audioConfiguration(audioJSON)
            let tower = makeAudioNet(geometry)
            try loadAudioWeights(arrays(inDirectory: directoryURL, prefix: "model.audio_tower."), into: tower)
            let embedder = NFKMLXGemma4MultimodalEmbedder(multimodalHidden: geometry.outputProjDims, textHidden: hidden,
                                                          eps: geometry.rmsEpsilon)
            try NFKMLXWeights.apply(arrays(inDirectory: directoryURL, prefix: "model.embed_audio."), to: embedder)
            audio = (tower, embedder)
        }
        let towers: [Module] = [vision?.tower, vision?.embedder, audio?.tower, audio?.embedder].compactMap { $0 }
        try loadWeights(into: decoder, fromDirectory: directoryURL, precision: precision, residency: residency,
                        streams: true, besides: towers.reduce(0) { $0 + NFKMLXLanguage.heldBytes(of: $1) })
        let model = NFKGemma4MultimodalModel(
            decoder: decoder, vision: vision, audio: audio, imageProcessor: imageProcessor, tokenizer: tokenizer,
            imageTokenId: (json["image_token_id"] as? NSNumber)?.intValue ?? tokenizer.id(forToken: "<|image|>") ?? 258_880,
            audioTokenId: (json["audio_token_id"] as? NSNumber)?.intValue ?? tokenizer.id(forToken: "<|audio|>") ?? 258_881,
            chatTemplate: NFKMLXLanguage.chatTemplate(inDirectory: directoryURL),
            contextLength: ((json["text_config"] as? [String: Any])?["max_position_embeddings"] as? NSNumber)?.intValue,
            releaseSampling: NFKMLXReleaseSampling(directoryURL: directoryURL))
        return multimodalBackend(model, releaseDirectoryURL: directoryURL)
    }

    /// ``gemmaMultimodalBackend(directoryURL:residency:)`` under ``NFKMLXResidency/automatic``.
    /// Introduced in InferKit 0.4.0.
    @objc(gemmaMultimodalBackendWithDirectoryURL:error:)
    public static func gemmaMultimodalBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try gemmaMultimodalBackend(directoryURL: directoryURL, residency: .automatic)
    }

    /// Downloads a Gemma 4 release, such as `google/gemma-4-E2B-it`, and builds its multimodal backend.
    ///
    /// @discussion The download is the one ``gemmaBackend(repo:revision:cacheDirectoryURL:)`` fetches, at
    /// the same revision; the towers sit in the same weights. A cached file is not fetched again. The call
    /// blocks on the network; call it off the render thread. Introduced in InferKit 0.4.0.
    @objc(gemmaMultimodalBackendWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func gemmaMultimodalBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                              residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try gemmaMultimodalBackend(directoryURL: try NFKMLXReleaseDownload.directory(
                repo: repo, revision: downloadRevision(repo: repo, revision: revision), cacheDirectoryURL: cacheDirectoryURL,
                required: requiredFiles, optional: optionalFiles, weights: weightFiles), residency: residency)
        }
    }

    /// The asynchronous form of ``gemmaMultimodalBackend(repo:revision:cacheDirectoryURL:residency:)``.
    /// The handler runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(gemmaMultimodalBackendWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func gemmaMultimodalBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                              residency: NFKMLXResidency,
                                              completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try gemmaMultimodalBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                                       residency: residency)
        }
    }

    /// Registers the multimodal backend under ``gemmaMultimodalModelName``. The registry's URL is the
    /// release directory; nil is refused. Introduced in InferKit 0.4.0.
    @objc public static func registerMultimodal() {
        NFKMLXGenerationBackend.registerDirectoryModel(gemmaMultimodalModelName) {
            try gemmaMultimodalBackend(directoryURL: $0)
        }
    }
}

/// Text generation from a Gemma 4 release with a picture or a clip in the prompt.
///
/// @discussion The backend reads `NFKInputPrompt` or `NFKInputMessages`, a `CGImage` or texture under
/// `NFKInputImage`, and a recording under `NFKInputAudio`; the media sit at the start of the first user turn.
/// The prompt, its soft tokens spliced in, prefills the decoder's cache, and each new token runs alone. A request
/// continues the conversation its `NFKParameterConversationKey` names, or, naming none, the kept conversation its
/// prompt continues furthest, and prefills only what it adds. The image's and the clip's positions are kept under
/// digests of their pixels and features, so a follow-up about the same picture does not run the vision tower
/// again (a clip's tower runs, since its soft-token count comes from it), and one with other media continues
/// nothing past the first position where they differ. Introduced in InferKit 0.4.0.
public final class NFKMLXGemma4MultimodalBackend: NSObject, NFKInferenceBackend {
    let model: NFKGemma4MultimodalModel
    private let releaseDirectoryURL: URL?
    private let modelInfoCache = NFKMLXModelInfoCache()
    /// One generation at a time: a run moves a conversation's cache through the shared decoder.
    private let generationLock = NSLock()
    /// The conversations' caches, read and written under `generationLock`.
    private let conversations = NFKMLXConversationKeeper<NFKMLXGemma4Conversation>(byteBudget: 0)

    init(model: NFKGemma4MultimodalModel, releaseDirectoryURL: URL? = nil) {
        self.model = model
        self.releaseDirectoryURL = releaseDirectoryURL
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { NFKMLXGemmaLanguage.gemmaMultimodalModelName }
    private var identifier: String { backendIdentifier }
    /// The model holds no adapters.
    private var adapterStrengths: [Float] { [] }

    /// The model's parameter count, weight bytes, precision, and quantization, with the bytes its release
    /// directory occupies on disk, the context length, and the residency (`NFKModelInfo*` keys).
    /// Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] {
        modelInfoCache.value {
            var info = NFKMLXModelDescription.info(of: model.modules, releaseDirectoryURL: releaseDirectoryURL)
            info[NFKModelInfoResidency] = model.decoder.layerStream != nil ? NFKModelResidencyStreamed : NFKModelResidencyResident
            info[NFKModelInfoContextLength] = info[NFKModelInfoContextLength] ?? model.contextLength
            return info
        }
    }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { NFKMLXGemmaLanguage.multimodalParameterKeys }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { NFKMLXGemmaLanguage.multimodalInputKeys }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputText, NFKOutputUsage, NFKOutputFinishReason] }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try NFKMLXUsage.refuseReasoningEffort(in: request, model: identifier)
        var sampling = NFKMLXStepSampling(release: model.releaseSampling)
        sampling.read(request)
        let maximumTokens = NFKMLXGenerationBackend.number(request, NFKParameterMaxTokens)?.intValue
        let seed = NFKMLXGenerationBackend.number(request, NFKParameterSeed)?.uint64Value

        let image = try NFKMLXGenerationBackend.image(request, NFKInputImage).map { try model.preparedImage($0) }
        let waveform = NFKMLXGemmaLanguage.waveform(request)
        if request.input(forKey: NFKInputAudio) != nil, waveform == nil {
            throw NFKMLXError.unsupportedInput
        }
        let audio = try waveform.map { try model.preparedAudio($0) }
        let text = try model.promptText(request, imageTokens: image?.count, audioTokens: audio?.soft.dim(0))
        let promptTokens = model.tokenizer.encode(text)
        let key = try model.conversationKey(promptTokens, image: image?.digest, audio: audio?.digest)
        let stops = Set([model.tokenizer.id(forToken: "<turn|>"), model.tokenizer.id(forToken: "<eos>")].compactMap { $0 })
        let limit = try NFKMLXContextRoom.bounded(maximumTokens, promptTokens: promptTokens.count,
                                                  contextLength: model.contextLength, source: identifier)
        var produced = [Int]()
        var cached = 0
        if limit > 0 {
            generationLock.lock(); defer { generationLock.unlock() }
            (produced, cached) = try generate(promptTokens, key: key, image: image, audio: audio?.soft, request: request,
                                              sampling: sampling, limit: limit, seed: seed, stops: stops)
        }
        return NFKInferenceResult(outputs: [
            NFKOutputText: model.tokenizer.decode(produced, skipSpecial: true),
            NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: cached,
                                                outputTokens: produced.count, reasoningTokens: nil),
            NFKOutputFinishReason: NFKMLXContextRoom.finishReason(produced: produced.count, limit: limit),
        ])
    }

    /// Generation through the conversation `request` names or `key` continues: the cache resumes as far as the
    /// conversation reaches, the rest of the prompt prefills with the soft tokens of the media whose positions it
    /// holds, from the first of them it holds, pausing to snapshot the sliding rows at the last turn's start and
    /// the prompt's end, and each new token runs alone. Runs under `generationLock`. Answers the tokens produced
    /// and how many prompt positions the cache already held.
    private func generate(_ promptTokens: [Int], key: [Int],
                          image: (pixels: MLXArray, positions: MLXArray, count: Int, digest: UInt64)?, audio: MLXArray?,
                          request: NFKInferenceRequest, sampling: NFKMLXStepSampling, limit: Int, seed: UInt64?,
                          stops: Set<Int>) throws -> (produced: [Int], cached: Int) {
        let cached = NFKMLXGemmaLanguage.cachedGeneration(model.decoder)
        let c = cached.configuration
        conversations.resolveSpill(release: releaseDirectoryURL, identifier: identifier) { modelInfo }
        let (name, state) = conversations.conversation(
            named: NFKMLXConversation.name(of: request), reuses: NFKMLXConversation.reuses(request), prompt: key,
            signature: "gemma4 sliding \(c.slidingWindow)",
            make: { NFKMLXGemma4Conversation(layerTypes: c.layerTypes, slidingWindow: c.slidingWindow,
                                             storingLayers: c.firstSharedLayer) },
            read: { url, record in
                try NFKMLXGemma4Conversation.load(from: url, tokens: record.tokens, layerTypes: c.layerTypes,
                                                  slidingWindow: c.slidingWindow, storingLayers: c.firstSharedLayer)
            })
        var failed = true
        defer {
            conversations.finish(name, failed: failed, afforded: NFKMLXConversationBudget.afforded(
                info: modelInfo, runBytes: Self.keyValueBytes(c, positions: model.contextLength)), spills: true)
        }
        let start = state.resume(for: key)
        let fresh = Array(promptTokens[start...])
        // A medium's soft tokens from the first of its positions the fresh part holds; nil where it holds none.
        func soft(_ token: Int, _ all: () -> MLXArray?) -> MLXArray? {
            guard fresh.contains(token), let whole = all() else { return nil }
            let before = promptTokens[..<start].filter { $0 == token }.count
            return before > 0 ? whole[before...] : whole
        }
        let ids = model.chain.placeholderIds
        let visionSoft = soft(ids.image) { image.flatMap { model.chain.visionSoftTokens(prepared: $0) } }
        let audioSoft = soft(ids.audio) { audio }
        let (embeddings, padded) = model.chain.fusedEmbeddings(tokens: fresh, visionSoftTokens: visionSoft,
                                                               audioSoftTokens: audioSoft)
        let turnStart = model.tokenizer.id(forToken: "<|turn>").flatMap { promptTokens.lastIndex(of: $0) }
        let snapshotAt = name == nil ? [] : NFKMLXGemma4Conversation.snapshotPositions(
            start: start, end: promptTokens.count, turnStart: turnStart)
        var logits = cached.prefill(padded.asArray(Int32.self).map(Int.init), embeddings: embeddings, cache: state.cache,
                                    stops: snapshotAt) { _ in state.snapshot() }
        try verify()
        var produced = [Int]()
        var fed = [Int]()
        for step in 0 ..< limit {
            let next = sampling.next(logits, seed: seed, step: step)
            try verify()
            if stops.contains(next) || NFKMLXErrorScope.hasFailed { break }
            produced.append(next)
            guard step + 1 < limit else { break }
            logits = cached.step(MLXArray(Int32(next)), cache: state.cache)
            fed.append(next)
        }
        guard !NFKMLXErrorScope.hasFailed else { return (produced, start) }
        state.record(Array(key[start...]) + fed)
        failed = false
        return (produced, start)
    }

    /// Throws a failed read of a streamed layer.
    private func verify() throws {
        if let stream = model.decoder.layerStream, let failure = stream.readFailure {
            stream.clearFailure()
            throw failure
        }
    }

    /// The bytes one run's keys and values take over `positions` positions: every position for the full
    /// layers, the window for the sliding ones, the shared tail storing none.
    static func keyValueBytes(_ c: NFKMLXGemmaConfiguration, positions: Int?) -> Int {
        (0 ..< c.firstSharedLayer).reduce(0) { total, layer in
            let kind = c.layerTypes[layer]
            let rows = kind == .full ? (positions ?? 0) : Swift.min(positions ?? 0, c.slidingWindow)
            return total + 2 * c.keyValueHeadCount(for: kind) * c.headDimensions(for: kind) * rows * 4
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
