//
//  NFKMLXVoiceBackend.swift
//  InferKitMLX
//
//  The FastSpeech2 + HiFi-GAN voice (`NFKMLXVoice`) and the hand-chained voice (`NFKMLXTTS`) behind the
//  InferKit contract, built from Objective-C: text → an `NFKMLXPhonemizing` object → mel → waveform.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

extension NFKMLXFastSpeech2 {

    /// The registry name the FastSpeech2 + HiFi-GAN voice builds under. Introduced in InferKit 0.4.0.
    @objc public static let voiceModelName = "fastspeech2-voice"

    /// The repository the download factories read when none is named: the espnet LJSpeech FastSpeech2
    /// conformer with its paired HiFi-GAN in one checkpoint. Introduced in InferKit 0.4.0.
    @objc public static let voiceReleaseRepo = "espnet/fastspeech2_conformer_with_hifigan"

    /// The request inputs a voice backend reads.
    static let voiceInputKeys: Set<String> = [NFKInputPrompt, NFKInputMessages]

    /// A result carrying `samples` at `sampleRate` as a WAV file in the temporary directory under
    /// `NFKOutputAudio`.
    static func audioResult(_ samples: [Float], sampleRate: Int, identifier: String) throws -> NFKInferenceResult {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(identifier)-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: samples, sampleRate: sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(samples.count) / Double(sampleRate),
                                  sampleRate: Double(sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    /// `voice` and `phonemizer` behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt`, or the user turns of `NFKInputMessages` joined, and
    /// returns a mono WAV at ``NFKMLXVoice/sampleRate`` as an `NFKAudioAsset` under `NFKOutputAudio`. A
    /// phonemizer that fails fails the request.
    static func voiceBackend(_ voice: NFKMLXVoice, phonemizer: any NFKMLXPhonemizing) -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: voiceModelName, inputKeys: voiceInputKeys, parameterKeys: [],
                                outputKeys: [NFKOutputAudio]) { request, job in
            let text = NFKMLXSpeechBackend.text(from: request)
            guard !text.isEmpty else { throw NFKMLXError.unsupportedInput }
            job.reportProgress(0)
            let samples = voice.speak(phonemes: try phonemizer.phonemes(forText: text))
            if job.status == .cancelled { return nil }
            let result = try audioResult(samples, sampleRate: NFKMLXVoice.sampleRate, identifier: voiceModelName)
            job.reportProgress(1)
            return result
        }
    }

    /// The voice from the acoustic checkpoint, the vocoder checkpoint, and the acoustic model's phoneme
    /// vocabulary (`vocab.json`, symbol → id), speaking the symbols `phonemizer` produces.
    ///
    /// @discussion The checkpoints are the released espnet LJSpeech FastSpeech2 conformer and its paired
    /// HiFi-GAN (`.safetensors` or the raw `pytorch_model.bin`); one combined checkpoint may be named for
    /// both. Introduced in InferKit 0.4.0.
    @objc(voiceBackendWithAcousticURL:vocoderURL:vocabularyURL:phonemizer:error:)
    public static func voiceBackend(acousticURL: URL, vocoderURL: URL, vocabularyURL: URL,
                                    phonemizer: any NFKMLXPhonemizing) throws -> any NFKInferenceBackend {
        voiceBackend(try NFKMLXVoice.voice(acousticURL: acousticURL, vocoderURL: vocoderURL,
                                           vocabularyURL: vocabularyURL), phonemizer: phonemizer)
    }

    /// ``voiceBackend(acousticURL:vocoderURL:vocabularyURL:phonemizer:)`` phonemizing through the
    /// ``NFKMLXNeuralG2P`` checkpoint at `g2pWeightsURL`, whose phoneme ids are the acoustic vocabulary's.
    /// Introduced in InferKit 0.4.0.
    @objc(voiceBackendWithAcousticURL:vocoderURL:vocabularyURL:g2pWeightsURL:error:)
    public static func voiceBackend(acousticURL: URL, vocoderURL: URL, vocabularyURL: URL,
                                    g2pWeightsURL: URL) throws -> any NFKInferenceBackend {
        let voice = try NFKMLXVoice.voice(acousticURL: acousticURL, vocoderURL: vocoderURL, vocabularyURL: vocabularyURL)
        return voiceBackend(voice, phonemizer: try g2p(weightsURL: g2pWeightsURL, voice: voice))
    }

    /// A G2P model whose ids name `voice`'s vocabulary symbols.
    static func g2p(weightsURL: URL, voice: NFKMLXVoice) throws -> NFKMLXNeuralG2P {
        var symbols = [String](repeating: "<unk>", count: (voice.vocabulary.values.max() ?? -1) + 1)
        for (symbol, id) in voice.vocabulary where id >= 0 {
            symbols[id] = symbol
        }
        return try NFKMLXNeuralG2P.g2p(weightsURL: weightsURL, phonemeSymbols: symbols)
    }

    /// The combined checkpoint and vocabulary a voice release directory holds.
    static func voiceFiles(inDirectory directory: URL) throws -> (checkpoint: URL, vocabulary: URL) {
        let candidates = ["model.safetensors", "pytorch_model.bin"].map { directory.appendingPathComponent($0) }
        guard let checkpoint = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) holds neither model.safetensors nor pytorch_model.bin")
        }
        return (checkpoint, directory.appendingPathComponent("vocab.json"))
    }

    /// The voice from a release directory holding one checkpoint with both networks (`model.safetensors`
    /// or `pytorch_model.bin`, the layout of `espnet/fastspeech2_conformer_with_hifigan`) and `vocab.json`.
    /// Introduced in InferKit 0.4.0.
    @objc(voiceBackendWithDirectoryURL:phonemizer:error:)
    public static func voiceBackend(directoryURL: URL, phonemizer: any NFKMLXPhonemizing) throws -> any NFKInferenceBackend {
        let files = try voiceFiles(inDirectory: directoryURL)
        return try voiceBackend(acousticURL: files.checkpoint, vocoderURL: files.checkpoint,
                                vocabularyURL: files.vocabulary, phonemizer: phonemizer)
    }

    /// ``voiceBackend(directoryURL:phonemizer:)`` phonemizing through the ``NFKMLXNeuralG2P`` checkpoint at
    /// `g2pWeightsURL`. Introduced in InferKit 0.4.0.
    @objc(voiceBackendWithDirectoryURL:g2pWeightsURL:error:)
    public static func voiceBackend(directoryURL: URL, g2pWeightsURL: URL) throws -> any NFKInferenceBackend {
        let files = try voiceFiles(inDirectory: directoryURL)
        return try voiceBackend(acousticURL: files.checkpoint, vocoderURL: files.checkpoint,
                                vocabularyURL: files.vocabulary, g2pWeightsURL: g2pWeightsURL)
    }

    /// Downloads the voice release (``voiceReleaseRepo`` when `repo` is nil) and returns it behind the
    /// InferKit contract, speaking the symbols `phonemizer` produces.
    ///
    /// @discussion The download is `config.json`, `vocab.json`, and the combined checkpoint, about 650 MB.
    /// No G2P model is released, so the phonemizer is the caller's. A file already in the cache is not
    /// fetched again. The call blocks on the network, so run it off the main and render threads.
    /// Introduced in InferKit 0.4.0.
    @objc(voiceBackendWithRepo:revision:cacheDirectoryURL:phonemizer:error:)
    public static func voiceBackend(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                    phonemizer: any NFKMLXPhonemizing) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let directory = try NFKMLXReleaseDownload.directory(
                repo: repo ?? voiceReleaseRepo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                required: ["config.json", "vocab.json"], optional: [],
                weights: ["model.safetensors", "pytorch_model.bin"])
            return try voiceBackend(directoryURL: directory, phonemizer: phonemizer)
        }
    }

    /// The asynchronous form of ``voiceBackend(repo:revision:cacheDirectoryURL:phonemizer:)``. The handler
    /// runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(voiceBackendWithRepo:revision:cacheDirectoryURL:phonemizer:completionHandler:)
    public static func voiceBackend(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                    phonemizer: any NFKMLXPhonemizing,
                                    completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try voiceBackend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, phonemizer: phonemizer)
        }
    }

    /// Registers the voice under ``voiceModelName``. The registry's URL is a voice release directory
    /// that also holds the G2P checkpoint as `g2p.safetensors`, built through
    /// ``voiceBackend(directoryURL:g2pWeightsURL:)``; nil is refused. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(voiceModelName) {
            try voiceBackend(directoryURL: $0, g2pWeightsURL: $0.appendingPathComponent("g2p.safetensors"))
        }
    }

    // MARK: The hand-chained voice

    /// The hand-chained voice (``NFKMLXTTS``) at its default geometry behind the InferKit contract.
    ///
    /// @discussion `phonemizer` turns the prompt into symbols, `symbols` maps them to ids by position,
    /// and the acoustic model and the vocoder load from their safetensors checkpoints, or keep random
    /// weights where a URL is nil. The backend reads `NFKInputPrompt` or `NFKInputMessages` and returns a
    /// mono WAV at 22050 Hz under `NFKOutputAudio`. A phonemizer that fails fails the request.
    /// Introduced in InferKit 0.4.0.
    @objc(ttsBackendWithPhonemizer:acousticURL:vocoderURL:symbols:error:)
    public static func ttsBackend(phonemizer: any NFKMLXPhonemizing, acousticURL: URL?, vocoderURL: URL?,
                                  symbols: [String]) throws -> any NFKInferenceBackend {
        let bridge = NFKObjCPhonemizer(phonemizer)
        let tts = NFKMLXTTS(phonemizer: bridge, symbols: symbols)
        try tts.loadWeights(acousticURL: acousticURL, vocoderURL: vocoderURL)
        let sampleRate = 22_050
        return NFKMLXGenerationBackend(identifier: "tts", inputKeys: voiceInputKeys, parameterKeys: [],
                                       outputKeys: [NFKOutputAudio]) { request, job in
            let text = NFKMLXSpeechBackend.text(from: request)
            guard !text.isEmpty else { throw NFKMLXError.unsupportedInput }
            job.reportProgress(0)
            let waveform = tts.synthesize(text)
            try bridge.raiseFailure()
            eval(waveform)
            if job.status == .cancelled { return nil }
            let result = try audioResult(waveform.asType(.float32).asArray(Float.self), sampleRate: sampleRate,
                                         identifier: "tts")
            job.reportProgress(1)
            return result
        }
    }
}
