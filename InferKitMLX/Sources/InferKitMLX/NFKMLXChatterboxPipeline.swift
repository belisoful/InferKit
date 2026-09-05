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

    /// Loads `tokenizer.json`, `ve.safetensors`, `t3_cfg.safetensors`, and `s3gen.safetensors`.
    public init(directoryURL: URL) throws {
        textTokenizer = try NFKMLXChatterboxTextTokenizer(url: directoryURL.appendingPathComponent("tokenizer.json"))
        voiceEncoder = NFKMLXChatterboxVoiceEncoderNet(.released)
        try NFKMLXChatterbox.loadVoiceEncoderWeights(into: voiceEncoder, from: directoryURL.appendingPathComponent("ve.safetensors"))
        t3 = NFKMLXT3Net(.released)
        try NFKMLXChatterbox.loadT3Weights(into: t3, from: directoryURL.appendingPathComponent("t3_cfg.safetensors"))
        let s3genURL = directoryURL.appendingPathComponent("s3gen.safetensors")
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
    public func speechTokens(text: String, conditionals: NFKMLXChatterboxConditionals,
                             options: NFKMLXT3SamplingOptions = NFKMLXT3SamplingOptions(),
                             shouldContinue: () -> Bool = { true }) -> [Int] {
        let normalized = NFKMLXChatterboxTextTokenizer.normalizedPunctuation(text)
        let textTokens = textTokenizer.encodeForSynthesis(normalized)
        return t3.generate(condition: conditionals.t3, textTokens: textTokens, options: options,
                           shouldContinue: shouldContinue)
            .filter { $0 < t3.configuration.startSpeechToken }
    }

    /// Text → 24 kHz samples in the prompt's voice.
    public func synthesize(text: String, conditionals: NFKMLXChatterboxConditionals,
                           t3Options: NFKMLXT3SamplingOptions = NFKMLXT3SamplingOptions(),
                           flowOptions: NFKS3FlowOptions = NFKS3FlowOptions()) -> [Float] {
        let codes = speechTokens(text: text, conditionals: conditionals, options: t3Options)
        guard !codes.isEmpty else { return [] }
        return s3gen.synthesize(tokens: codes, prompt: conditionals.s3gen, options: flowOptions)
    }
}

private final class NFKChatterboxHolder: @unchecked Sendable {
    let tts: NFKMLXChatterboxTTS
    let conditionals: NFKMLXChatterboxConditionals
    init(_ tts: NFKMLXChatterboxTTS, conditionals: NFKMLXChatterboxConditionals) {
        self.tts = tts
        self.conditionals = conditionals
    }
}

extension NFKMLXChatterbox {
    /// A text-to-speech backend in one voice: text under `NFKInputPrompt` → a 24 kHz WAV under
    /// `NFKOutputAudio`. `voiceURL` is a WAV of the voice to clone; nil uses the release's built-in
    /// voice (`conds.pt` in the release directory). T3 samples at the reference defaults from a fixed
    /// seed; the flow's initial noise and the vocoder's source noise come from MLX's global random state,
    /// so seed it (`NFKMLXRandom.seed`) when a run must repeat.
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
        let holder = NFKChatterboxHolder(tts, conditionals: conditionals)
        return NFKMLXSpeechBackend(identifier: "chatterbox",
                                   configuration: NFKMLXSpeechConfiguration(sampleRate: tts.sampleRate)) { text, _ in
            MLXArray(holder.tts.synthesize(text: text, conditionals: holder.conditionals))
        }
    }

    /// The Objective-C entry: the release directory and an optional voice WAV.
    @objc(chatterboxBackendWithDirectoryURL:voiceURL:error:)
    public static func backend(directoryURL: URL, voiceURL: URL?) throws -> NFKMLXSpeechBackend {
        try speechBackend(directoryURL: directoryURL, voiceURL: voiceURL)
    }
}
