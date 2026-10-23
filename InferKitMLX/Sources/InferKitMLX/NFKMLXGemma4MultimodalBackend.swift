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

    init(decoder: NFKMLXGemmaNet, vision: (tower: NFKMLXGemma4VisionNet, embedder: NFKMLXGemma4MultimodalEmbedder)?,
         audio: (tower: NFKMLXGemma4AudioNet, embedder: NFKMLXGemma4MultimodalEmbedder)?,
         imageProcessor: NFKMLXGemma4ImageProcessor, tokenizer: NFKMLXGemmaTokenizer,
         imageTokenId: Int, audioTokenId: Int, chatTemplate: String? = nil, contextLength: Int? = nil) {
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.chatTemplate = chatTemplate
        self.contextLength = contextLength
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
    func promptText(_ request: NFKInferenceRequest, imageTokens: Int?, audioTokens: Int?) throws -> String {
        var media = ""
        if let imageTokens {
            media += "<|image>" + String(repeating: "<|image|>", count: imageTokens) + "<image|>"
        }
        if let audioTokens {
            media += "<|audio>" + String(repeating: "<|audio|>", count: audioTokens) + "<audio|>"
        }
        let messages: [[String: Any]]
        if let prompt = request.prompt {
            messages = [["role": "user", "content": prompt]]
        } else if let conversation = request.messages {
            messages = conversation
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
    static let multimodalParameterKeys: Set<String> = [NFKParameterTemperature, NFKParameterMaxTokens, NFKParameterSeed]

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
    /// `NFKParameterMaxTokens` (256), and `NFKParameterSeed`. The decoder runs prefill-only, so each step
    /// re-runs the growing sequence with the same soft tokens spliced in.
    static func multimodalBackend(_ model: NFKGemma4MultimodalModel) -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: gemmaMultimodalModelName, inputKeys: multimodalInputKeys,
                                parameterKeys: multimodalParameterKeys,
                                outputKeys: [NFKOutputText, NFKOutputUsage]) { request, job in
            try NFKMLXUsage.refuseReasoningEffort(in: request, model: gemmaMultimodalModelName)
            let temperature = NFKMLXGenerationBackend.number(request, NFKParameterTemperature)?.floatValue ?? 0
            let maximumTokens = NFKMLXGenerationBackend.number(request, NFKParameterMaxTokens)?.intValue ?? 256
            let seed = NFKMLXGenerationBackend.number(request, NFKParameterSeed)?.uint64Value

            let visionSoft = try NFKMLXGenerationBackend.image(request, NFKInputImage).map(model.visionSoftTokens)
            let audioSoft = try waveform(request).map(model.audioSoftTokens)
            if request.input(forKey: NFKInputAudio) != nil, audioSoft == nil {
                throw NFKMLXError.unsupportedInput
            }
            let text = try model.promptText(request, imageTokens: visionSoft?.dim(0), audioTokens: audioSoft?.dim(0))
            let promptTokens = model.tokenizer.encode(text)
            let stops = Set([model.tokenizer.id(forToken: "<turn|>"), model.tokenizer.id(forToken: "<eos>")]
                .compactMap { $0 })

            let limit = try NFKMLXContextRoom.bounded(maximumTokens, promptTokens: promptTokens.count,
                                                      contextLength: model.contextLength, source: gemmaMultimodalModelName)
            var tokens = promptTokens
            var produced = [Int]()
            for step in 0 ..< max(limit, 0) {
                if job.status == .cancelled { return nil }
                let (embeddings, ids) = model.chain.fusedEmbeddings(tokens: tokens, visionSoftTokens: visionSoft,
                                                                     audioSoftTokens: audioSoft)
                let logits = model.decoder.logits(fromEmbeddings: embeddings, tokens: ids)[0, tokens.count - 1]
                let next: Int
                if temperature <= 0 {
                    next = logits.argMax(axis: -1).item(Int.self)
                } else {
                    if let seed { MLXRandom.seed(seed &+ UInt64(step)) }
                    next = MLXRandom.categorical(logits * (1 / temperature)).item(Int.self)
                }
                if stops.contains(next) { break }
                produced.append(next)
                tokens.append(next)
                job.reportProgress(min(Double(produced.count) / Double(max(limit, 1)), 0.99))
            }
            return NFKInferenceResult(outputs: [
                NFKOutputText: model.tokenizer.decode(produced, skipSpecial: true),
                NFKOutputUsage: NFKMLXUsage.outputs(inputTokens: promptTokens.count, cachedTokens: 0,
                                                    outputTokens: produced.count, reasoningTokens: nil),
            ])
        }
    }

    /// The vision tower geometry a release's `vision_config` states.
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
        try loadWeights(into: decoder, fromDirectory: directoryURL, precision: .float32, residency: residency)
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
        let model = NFKGemma4MultimodalModel(
            decoder: decoder, vision: vision, audio: audio, imageProcessor: imageProcessor, tokenizer: tokenizer,
            imageTokenId: (json["image_token_id"] as? NSNumber)?.intValue ?? tokenizer.id(forToken: "<|image|>") ?? 258_880,
            audioTokenId: (json["audio_token_id"] as? NSNumber)?.intValue ?? tokenizer.id(forToken: "<|audio|>") ?? 258_881,
            chatTemplate: NFKMLXLanguage.chatTemplate(inDirectory: directoryURL),
            contextLength: ((json["text_config"] as? [String: Any])?["max_position_embeddings"] as? NSNumber)?.intValue)
        return multimodalBackend(model)
    }

    /// ``gemmaMultimodalBackend(directoryURL:residency:)`` under ``NFKMLXResidency/automatic``.
    /// Introduced in InferKit 0.4.0.
    @objc(gemmaMultimodalBackendWithDirectoryURL:error:)
    public static func gemmaMultimodalBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try gemmaMultimodalBackend(directoryURL: directoryURL, residency: .automatic)
    }

    /// Downloads a Gemma 4 release, such as `google/gemma-4-E2B-it`, and builds its multimodal backend.
    ///
    /// @discussion The download is the one ``gemmaBackend(repo:revision:cacheDirectoryURL:)`` fetches; the
    /// towers sit in the same weights. A cached file is not fetched again. The call blocks on the
    /// network; call it off the render thread. Introduced in InferKit 0.4.0.
    @objc(gemmaMultimodalBackendWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func gemmaMultimodalBackend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                                              residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try gemmaMultimodalBackend(directoryURL: try NFKMLXReleaseDownload.directory(
                repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
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
