//
//  NFKMLXChatterboxPipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Chatterbox stage 6: the pipeline. A voice prompt becomes the T3 condition (speaker embedding, prompt
// codes, exaggeration) and the S3Gen prompt (codes, 24 kHz mel, x-vector); text becomes tokens; T3
// samples speech codes; S3Gen turns them into 24 kHz audio. `ChatterboxTTS.prepare_conditionals` and
// `generate`, with the release's built-in voice (`conds.pt`) available when no prompt is given.

/// The two conditionings a voice prompt yields (`Conditionals`).
public struct NFKMLXChatterboxConditionals {
    public var t3: NFKMLXT3Condition
    public var s3gen: NFKMLXS3GenPrompt
    public init(t3: NFKMLXT3Condition, s3gen: NFKMLXS3GenPrompt) {
        self.t3 = t3
        self.s3gen = s3gen
    }
}

/// Chatterbox end to end: the five networks and the text tokenizer, loaded from a release directory.
public final class NFKMLXChatterboxTTS {
    public let textTokenizer: NFKMLXChatterboxTextTokenizer
    public let voiceEncoder: NFKMLXChatterboxVoiceEncoderNet
    public let speechTokenizer: NFKMLXS3TokenizerNet
    public let t3: NFKMLXT3Net
    public let s3gen: NFKMLXS3GenNet
    /// The synthesized audio's rate.
    public let sampleRate = 24000
    /// How much of the prompt conditions T3 (`ENC_COND_LEN`, at 16 kHz).
    public static let encoderConditionSeconds = 6
    /// How much of the prompt conditions S3Gen (`DEC_COND_LEN`, at 24 kHz).
    public static let decoderConditionSeconds = 10

    /// Loads `ve.safetensors` and, preferring the MULTILINGUAL file of each pair when the release
    /// carries it, `grapheme_mtl_merged_expanded_v1.json` over `tokenizer.json`,
    /// `t3_mtl23ls_v3.safetensors` over `t3_cfg.safetensors`, and `s3gen_v3.safetensors` over
    /// `s3gen.safetensors`.
    ///
    /// The grapheme file is the multilingual tokenizer, and the release's `mtl_tokenizer.json` is NOT:
    /// the grapheme file holds 2454 entries, which is exactly the multilingual text embedding's width,
    /// where `mtl_tokenizer.json` stops at 2352 and pads with placeholders in the positions the
    /// grapheme file gives real characters.
    ///
    /// The two T3 checkpoints hold the same 292 tensors and differ only in the text embedding's width,
    /// so the release's own file sizes the model. `s3gen_v3` drops the one `tokenizer._mel_filters`
    /// buffer, which this package computes rather than reads.
    public init(directoryURL: URL) throws {
        func preferred(_ names: [String]) -> URL {
            let urls = names.map { directoryURL.appendingPathComponent($0) }
            return urls.first { FileManager.default.fileExists(atPath: $0.path) } ?? urls[urls.count - 1]
        }
        textTokenizer = try NFKMLXChatterboxTextTokenizer(
            url: preferred(["grapheme_mtl_merged_expanded_v1.json", "tokenizer.json"]))
        voiceEncoder = NFKMLXChatterboxVoiceEncoderNet(.released)
        try NFKMLXChatterbox.loadVoiceEncoderWeights(into: voiceEncoder, from: directoryURL.appendingPathComponent("ve.safetensors"))
        t3 = try NFKMLXChatterbox.makeT3(from: preferred(["t3_mtl23ls_v3.safetensors", "t3_cfg.safetensors"]))
        let s3genURL = preferred(["s3gen_v3.safetensors", "s3gen.safetensors"])
        speechTokenizer = NFKMLXS3TokenizerNet(.released)
        try NFKMLXChatterbox.loadTokenizerWeights(into: speechTokenizer, from: s3genURL)
        s3gen = NFKMLXS3GenNet()
        try NFKMLXChatterbox.loadS3GenWeights(into: s3gen, from: s3genURL)
    }

    /// `prepare_conditionals`: the voice prompt at any rate is resampled to 24 kHz and from there to
    /// 16 kHz; S3Gen reads its first ten seconds, the T3 prompt codes come from its first six seconds
    /// (at most 150 codes), and the speaker embedding from the whole of it.
    public func conditionals(voice samples: [Float], sampleRate: Int,
                             exaggeration: Float = 0.5) -> NFKMLXChatterboxConditionals {
        let wave24 = NFKMLXAudioRate.matched(samples, from: sampleRate, to: 24000)
        let wave16 = NFKMLXAudioRate.matched(wave24, from: 24000, to: 16000)
        let prompt = s3gen.prompt(samples24k: Array(wave24.prefix(NFKMLXChatterboxTTS.decoderConditionSeconds * 24000)),
                                  samples16k: Array(wave16.prefix(NFKMLXChatterboxTTS.decoderConditionSeconds * 16000)),
                                  tokenizer: speechTokenizer)
        let promptTokens = speechTokenizer.tokenize(Array(wave16.prefix(NFKMLXChatterboxTTS.encoderConditionSeconds * 16000)),
                                                    maximumCodes: t3.configuration.speechPromptLength)
        let speaker = voiceEncoder.embed(samples: wave16)
        return NFKMLXChatterboxConditionals(
            t3: NFKMLXT3Condition(speakerEmbedding: speaker, promptTokens: promptTokens, exaggeration: exaggeration),
            s3gen: prompt)
    }

    /// The release's built-in voice, `conds.pt`, read through the native checkpoint reader.
    public func builtinConditionals(url: URL, exaggeration: Float? = nil) throws -> NFKMLXChatterboxConditionals {
        let arrays = try NFKMLXWeights.loadCheckpoint(url: url).arrays
        func array(_ key: String) throws -> MLXArray {
            guard let value = arrays[key] else {
                throw NFKMLXError.weightsMismatch("\(url.lastPathComponent) carries no \(key)")
            }
            return value
        }
        func tokens(_ key: String) throws -> [Int] {
            let values = try array(key).reshaped([-1])
            return values.dtype == .int64 ? values.asArray(Int64.self).map(Int.init) : values.asArray(Int32.self).map(Int.init)
        }
        let storedExaggeration = try array("t3.emotion_adv").reshaped([-1]).asArray(Float.self).first ?? 0.5
        return NFKMLXChatterboxConditionals(
            t3: NFKMLXT3Condition(speakerEmbedding: try array("t3.speaker_emb").reshaped([-1]),
                                  promptTokens: try tokens("t3.cond_prompt_speech_tokens"),
                                  exaggeration: exaggeration ?? storedExaggeration),
            s3gen: NFKMLXS3GenPrompt(tokens: try tokens("gen.prompt_token"),
                                     mel: try array("gen.prompt_feat")[0],
                                     xVector: try array("gen.embedding").reshaped([-1])))
    }

    /// The S3 speech codes T3 samples for `text` (`punc_norm`, tokenize, start and stop tokens, sample,
    /// drop anything that is not a speech code).
    /// - Parameter text: the text to speak.
    /// - Parameter conditionals: the voice, from ``conditionals(voice:sampleRate:exaggeration:)`` or
    ///   ``builtinConditionals(url:exaggeration:)``.
    /// - Parameter options: T3's sampling settings.
    /// - Parameter language: a `NFKMLXChatterbox.supportedLanguages` id for a MULTILINGUAL release,
    ///   which tags the text and applies that language's rewriting. Nil is the English release's own
    ///   path, which neither lowercases nor normalizes, and is what the English checkpoint was
    ///   measured on.
    /// - Parameter shouldContinue: asked before each sampled token; returning false ends the run with
    ///   the codes produced so far.
    public func speechTokens(text: String, conditionals: NFKMLXChatterboxConditionals,
                             options: NFKMLXT3SamplingOptions = NFKMLXT3SamplingOptions(),
                             language: String? = nil,
                             shouldContinue: () -> Bool = { true }) -> [Int] {
        let normalized = NFKMLXChatterboxTextTokenizer.normalizedPunctuation(text)
        let textTokens = language == nil
            ? textTokenizer.encodeForSynthesis(normalized)
            : textTokenizer.encodeForSynthesis(normalized, language: language)
        return t3.generate(condition: conditionals.t3, textTokens: textTokens, options: options,
                           shouldContinue: shouldContinue)
            .filter { $0 < t3.configuration.startSpeechToken }
    }

    /// Text → 24 kHz samples in the prompt's voice.
    public func synthesize(text: String, conditionals: NFKMLXChatterboxConditionals,
                           t3Options: NFKMLXT3SamplingOptions = NFKMLXT3SamplingOptions(),
                           flowOptions: NFKS3FlowOptions = NFKS3FlowOptions(),
                           language: String? = nil) -> [Float] {
        let codes = speechTokens(text: text, conditionals: conditionals, options: t3Options,
                                 language: language)
        guard !codes.isEmpty else { return [] }
        return s3gen.synthesize(tokens: codes, prompt: conditionals.s3gen, options: flowOptions)
    }
}

private final class NFKChatterboxHolder: @unchecked Sendable {
    let tts: NFKMLXChatterboxTTS
    let conditionals: NFKMLXChatterboxConditionals
    let exaggeration: Float
    private let lock = NSLock()
    private var cloned: (clip: Data, conditionals: NFKMLXChatterboxConditionals)?

    init(_ tts: NFKMLXChatterboxTTS, conditionals: NFKMLXChatterboxConditionals, exaggeration: Float) {
        self.tts = tts
        self.conditionals = conditionals
        self.exaggeration = exaggeration
    }

    /// The voice of the clip the request carries under `NFKInputVoiceReference`, else the build's voice.
    /// The last clip's conditionals are kept, so a run of requests in one cloned voice prepares it once.
    func conditionals(for request: NFKInferenceRequest) throws -> NFKMLXChatterboxConditionals {
        guard let clip = try NFKMLXChatterbox.referenceClip(of: request) else {
            return conditionals
        }
        lock.lock()
        defer { lock.unlock() }
        if let cloned, cloned.clip == clip {
            return cloned.conditionals
        }
        guard let voice = NFKMLXWaveFile.read(clip) else {
            throw NFKMLXError.unsupportedConfiguration("the \(NFKInputVoiceReference) clip is not a readable WAV")
        }
        let prepared = tts.conditionals(voice: voice.samples, sampleRate: voice.sampleRate, exaggeration: exaggeration)
        cloned = (clip, prepared)
        return prepared
    }
}

extension NFKMLXChatterbox {
    /// A text-to-speech backend in one voice: text under `NFKInputPrompt` → a 24 kHz WAV under
    /// `NFKOutputAudio`. `voiceURL` is a WAV of the voice to clone; nil uses the release's built-in
    /// voice (`conds.pt` in the release directory). T3 samples at the reference defaults from a fixed
    /// seed; the flow's initial noise and the vocoder's source noise come from MLX's global random state,
    /// so seed it (`NFKMLXRandom.seed`) when a run must repeat. A request overrides `exaggeration` with
    /// ``NFKMLXChatterboxParameterKey/exaggeration``, and T3's temperature, guidance weight, and seed
    /// with `NFKParameterTemperature`, `NFKParameterGuidanceScale`, and `NFKParameterSeed`, for that
    /// request only. A request that carries a WAV clip under `NFKInputVoiceReference`, an `NFKAudioAsset`
    /// or its data, speaks in the clip's voice instead.
    public static func speechBackend(directoryURL: URL, voiceURL: URL? = nil,
                                     exaggeration: Float = 0.5) throws -> NFKMLXSpeechBackend {
        let tts = try NFKMLXChatterboxTTS(directoryURL: directoryURL)
        let conditionals: NFKMLXChatterboxConditionals
        if let voiceURL {
            guard let voice = NFKMLXWaveFile.read(try Data(contentsOf: voiceURL)) else {
                throw NFKMLXError.unsupportedConfiguration("\(voiceURL.lastPathComponent) is not a readable WAV")
            }
            conditionals = tts.conditionals(voice: voice.samples, sampleRate: voice.sampleRate, exaggeration: exaggeration)
        } else {
            conditionals = try tts.builtinConditionals(url: directoryURL.appendingPathComponent("conds.pt"),
                                                       exaggeration: exaggeration)
        }
        let holder = NFKChatterboxHolder(tts, conditionals: conditionals, exaggeration: exaggeration)
        return NFKMLXSpeechBackend(identifier: "chatterbox",
                                   configuration: NFKMLXSpeechConfiguration(sampleRate: tts.sampleRate),
                                   parameterKeys: requestParameterKeys, inputKeys: [NFKInputVoiceReference]) { text, _, request in
            MLXArray(holder.tts.synthesize(text: text,
                                           conditionals: NFKMLXChatterbox.conditionals(try holder.conditionals(for: request),
                                                                                       for: request),
                                           t3Options: NFKMLXChatterbox.samplingOptions(for: request)))
        }
    }

    /// The Objective-C entry: the release directory and an optional voice WAV, the voice a request speaks
    /// in when it carries no clip under `NFKInputVoiceReference`.
    @objc(chatterboxBackendWithDirectoryURL:voiceURL:error:)
    public static func backend(directoryURL: URL, voiceURL: URL?) throws -> NFKMLXSpeechBackend {
        try speechBackend(directoryURL: directoryURL, voiceURL: voiceURL)
    }

    /// The English release's files. The repo also serves the multilingual set, which a download
    /// leaves alone: the backend runs the English text path, which is what the English checkpoint
    /// was measured on.
    static let requiredFiles = ["ve.safetensors", "tokenizer.json", "s3gen.safetensors"]
    static let optionalFiles: [String] = []
    static let weightFiles = ["t3_cfg.safetensors"]
    /// The release's built-in voice, fetched when no voice WAV is given.
    static let builtinVoiceFile = "conds.pt"

    /// Downloads the English Chatterbox release (`ResembleAI/chatterbox`, public) and builds the
    /// backend in one voice, as ``backend(directoryURL:voiceURL:)`` does.
    ///
    /// @discussion The download is the voice encoder, the text tokenizer, T3, and S3Gen, about
    /// 3.2 GB, plus the built-in voice `conds.pt` when `voiceURL` is nil. A file already in the cache
    /// is not fetched again. The call blocks on the network, so run it off the main and render
    /// threads. Introduced in InferKit 0.4.0.
    @objc(chatterboxBackendWithRepo:revision:cacheDirectoryURL:voiceURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               voiceURL: URL?) throws -> NFKMLXSpeechBackend {
        try NFKMLXBuildHolds.keeping {
            let directory = try NFKMLXReleaseDownload.directory(
                repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                required: requiredFiles + (voiceURL == nil ? [builtinVoiceFile] : []),
                optional: optionalFiles, weights: weightFiles)
            return try backend(directoryURL: directory, voiceURL: voiceURL)
        }
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:voiceURL:)``. The handler
    /// runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(chatterboxBackendWithRepo:revision:cacheDirectoryURL:voiceURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?, voiceURL: URL?,
                               completionHandler: @escaping (NFKMLXSpeechBackend?, Error?) -> Void) {
        NFKMLXReleaseDownload.asyncValue(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                        voiceURL: voiceURL)
        }
    }
}
