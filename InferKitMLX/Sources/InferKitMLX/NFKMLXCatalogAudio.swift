//
//  NFKMLXCatalogAudio.swift
//  InferKitMLX
//
//  The catalog entries of the audio models: speech recognition, speech synthesis, speech enhancement, source
//  separation, voice activity, codecs, and music.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The catalog entries of the audio models, and the registrations their registry names need.
///
/// @discussion Each entry describes one registry name. A family of released sizes (Whisper, SNAC, HT Demucs,
/// MuScriptor, Wav2Vec2, and HuBERT) is one entry per name, built from the same release table its
/// registration reads, so the two cannot drift. A source is the publisher's own location: a Hugging Face
/// repository or the original download URL.
enum NFKMLXCatalogAudio {

    /// The audio models' entries, in the order ``NFKMLXModelDescriptors`` lists them.
    static var entries: [NFKMLXCatalogEntry] {
        speechRecognition + voiceActivityAndTagging + separation + enhancement + codecs + music + speechSynthesis
    }

    /// Registers the audio models that have a factory and no registry name, under the names their entries
    /// give. ``NFKMLXReferenceModels/registerAll()`` calls it.
    static func registerAdditionalModels() {
        for release in whisperReleases where release.name != NFKMLXWhisper.modelName {
            let (variant, name, download, repository) = (release.variant, release.name, URL(string: release.url), release.repository)
            NFKMLXModelRegistry.register(name: name) { location in
                try NFKMLXWhisper.catalogBackend(variant: variant, location: location, weightsDownload: download,
                                                 repository: repository, precision: .checkpoint, identifier: name)
            }
        }
        for release in wav2Vec2Releases {
            let name = release.name
            NFKMLXGenerationBackend.registerDirectoryModel(name) {
                try NFKMLXWav2Vec2.backend(directoryURL: $0, identifier: name)
            }
        }
        NFKMLXGenerationBackend.registerDirectoryModel(NFKMLXWav2Vec2Bert.modelName) {
            try NFKMLXWav2Vec2Bert.backend(directoryURL: $0)
        }
        NFKMLXGenerationBackend.registerDirectoryModel(kokoroName) {
            try NFKMLXKokoro.backend(directoryURL: $0, voiceName: kokoroVoice)
        }
        NFKMLXGenerationBackend.registerDirectoryModel(chatterboxName) {
            try NFKMLXChatterbox.backend(directoryURL: $0, voiceURL: nil)
        }
    }

    /// The registry names ``registerAdditionalModels()`` registers.
    static var additionalModelNames: [String] {
        whisperReleases.map(\.name).filter { $0 != NFKMLXWhisper.modelName }
            + wav2Vec2Releases.map(\.name)
            + [NFKMLXWav2Vec2Bert.modelName, kokoroName, chatterboxName]
    }

    // MARK: - Release tables

    /// A released Whisper checkpoint: its name in OpenAI's release, the size it builds as, the URL OpenAI's
    /// `whisper` package downloads it from, and the Hugging Face repository and commit whose tokenizer
    /// files decode it.
    struct WhisperRelease {
        let release: String
        let variant: NFKMLXWhisperVariant
        let url: String
        let repository: String
        let revision: String
        var name: String { "whisper-\(release)" }
    }

    /// The release a Whisper registry name builds, or nil.
    static func whisperRelease(named name: String) -> WhisperRelease? {
        whisperReleases.first { $0.name == name }
    }

    /// Every multilingual Whisper release. large-v1 and large-v2 share the `large` geometry.
    static let whisperReleases = [
        WhisperRelease(release: "tiny", variant: .tiny, url: "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
                       repository: "openai/whisper-tiny", revision: "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"),
        WhisperRelease(release: "base", variant: .base, url: "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
                       repository: "openai/whisper-base", revision: "e37978b90ca9030d5170a5c07aadb050351a65bb"),
        WhisperRelease(release: "small", variant: .small, url: "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
                       repository: "openai/whisper-small", revision: "973afd24965f72e36ca33b3055d56a652f456b4d"),
        WhisperRelease(release: "medium", variant: .medium, url: "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
                       repository: "openai/whisper-medium", revision: "abdf7c39ab9d0397620ccaea8974cc764cd0953e"),
        WhisperRelease(release: "large-v1", variant: .large, url: "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
                       repository: "openai/whisper-large", revision: "4ef9b41f0d4fe232daafdb5f76bb1dd8b23e01d7"),
        WhisperRelease(release: "large-v2", variant: .large, url: "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
                       repository: "openai/whisper-large-v2", revision: "ae4642769ce2ad8fc292556ccea8e901f1530655"),
        WhisperRelease(release: "large-v3", variant: .largeV3, url: "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
                       repository: "openai/whisper-large-v3", revision: "06f233fe06e710322aca913c1bc4249a0d71fce1"),
        WhisperRelease(release: "large-v3-turbo", variant: .largeV3Turbo, url: "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
                       repository: "openai/whisper-large-v3-turbo", revision: "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
    ]

    /// A Wav2Vec2 or HuBERT release `NFKMLXWav2Vec2` reads from its directory. `weights` is the entry of
    /// ``NFKMLXWav2Vec2/weightFiles`` the repository serves; `transcribes` is true for a release with a
    /// CTC head and its vocabulary.
    struct Wav2Vec2Release {
        let repository: String
        let revision: String
        let displayName: String
        let weights: String
        let transcribes: Bool
        var name: String { String(repository.split(separator: "/").last ?? "") }
        var family: String { name.hasPrefix("hubert") ? "hubert" : "wav2vec2" }
    }

    /// The releases the validation manifest pins, each at reference parity against its own record.
    static let wav2Vec2Releases: [Wav2Vec2Release] = {
        let safetensors = NFKMLXWav2Vec2.weightFiles[0]
        let torch = NFKMLXWav2Vec2.weightFiles[1]
        return [
            Wav2Vec2Release(repository: "facebook/wav2vec2-base-960h", revision: "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
                            displayName: "wav2vec 2.0 Base 960h", weights: safetensors, transcribes: true),
            Wav2Vec2Release(repository: "facebook/wav2vec2-base", revision: "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
                            displayName: "wav2vec 2.0 Base", weights: torch, transcribes: false),
            Wav2Vec2Release(repository: "facebook/wav2vec2-large-960h", revision: "bdeaacdf88f7a155f50a2704bc967aa81fbbb2ab",
                            displayName: "wav2vec 2.0 Large 960h", weights: torch, transcribes: true),
            Wav2Vec2Release(repository: "facebook/wav2vec2-large-960h-lv60-self",
                            revision: "54074b1c16f4de6a5ad59affb4caa8f2ea03a119",
                            displayName: "wav2vec 2.0 Large 960h LV-60 self-training", weights: torch, transcribes: true),
            Wav2Vec2Release(repository: "facebook/wav2vec2-large-lv60", revision: "0cde644b64dac88d8416bec1c92a4099b850ba0b",
                            displayName: "wav2vec 2.0 Large LV-60", weights: torch, transcribes: false),
            Wav2Vec2Release(repository: "facebook/wav2vec2-large-xlsr-53", revision: "c3f9d884181a224a6ac87bf8885c84d1cff3384f",
                            displayName: "wav2vec 2.0 XLSR-53", weights: torch, transcribes: false),
            Wav2Vec2Release(repository: "facebook/wav2vec2-xls-r-300m", revision: "1a640f32ac3e39899438a2931f9924c02f080a54",
                            displayName: "XLS-R 300M", weights: torch, transcribes: false),
            Wav2Vec2Release(repository: "facebook/wav2vec2-xls-r-1b", revision: "35eaea9a0ed0f97592277d79208e40ab8917d1e3",
                            displayName: "XLS-R 1B", weights: torch, transcribes: false),
            Wav2Vec2Release(repository: "facebook/wav2vec2-xls-r-2b", revision: "3b6d89d0fabead7da552eaaa07549c7c9c36d303",
                            displayName: "XLS-R 2B", weights: torch, transcribes: false),
            Wav2Vec2Release(repository: "facebook/hubert-base-ls960", revision: "dba3bb02fda4248b6e082697eee756de8fe8aa8a",
                            displayName: "HuBERT Base", weights: torch, transcribes: false),
            Wav2Vec2Release(repository: "facebook/hubert-large-ll60k", revision: "ff022d095678a2995f3c49bab18a96a9e553f782",
                            displayName: "HuBERT Large", weights: torch, transcribes: false),
            Wav2Vec2Release(repository: "facebook/hubert-large-ls960-ft", revision: "ece5fabbf034c1073acae96d5401b25be96709d8",
                            displayName: "HuBERT Large LS-960 fine-tuned", weights: torch, transcribes: true),
            Wav2Vec2Release(repository: "facebook/hubert-xlarge-ll60k", revision: "b0cef767123fe004883915a053f538f1737a1e47",
                            displayName: "HuBERT X-Large", weights: torch, transcribes: false),
            Wav2Vec2Release(repository: "facebook/hubert-xlarge-ls960-ft", revision: "86a09e67e0c8d074533992379242405825516eca",
                            displayName: "HuBERT X-Large LS-960 fine-tuned", weights: safetensors, transcribes: true),
        ]
    }()

    /// The registry name of Kokoro-82M.
    static let kokoroName = "kokoro-82m"

    /// The voicepack the registered Kokoro backend speaks in: the release's `voices/af_heart.pt`.
    static let kokoroVoice = "af_heart"

    /// The registry name of Chatterbox, which speaks in the release's built-in voice.
    static let chatterboxName = "chatterbox"

    /// The Hugging Face repository of a SNAC release.
    static func snacRepository(_ variant: NFKMLXSNACVariant) -> String {
        switch variant {
        case .speech24kHz: return "hubertsiuzdak/snac_24khz"
        case .music32kHz: return "hubertsiuzdak/snac_32khz"
        case .music44kHz: return "hubertsiuzdak/snac_44khz"
        }
    }

    /// The URL Demucs downloads an HT Demucs release from.
    static func htDemucsURL(_ variant: NFKMLXHTDemucsVariant) -> String {
        switch variant {
        case .fourStem: return "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
        case .sixStem: return "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th"
        }
    }
}

// MARK: - Speech recognition

extension NFKMLXCatalogAudio {

    static var speechRecognition: [NFKMLXCatalogEntry] {
        whisper + [parakeet, canary, voxtral, graniteSpeech] + wav2Vec2 + [w2vBert]
    }

    static var whisper: [NFKMLXCatalogEntry] {
        whisperReleases.map { release in
            model(release.name, displayName: "Whisper \(release.release)", family: "whisper",
                  summary: "OpenAI's multilingual speech recognizer, an encoder-decoder transformer over a log-mel spectrogram.",
                  tasks: [NFKModelTaskTranscription],
                  inputs: [audioInput("A WAV recording of speech.")],
                  outputs: [.output(NFKOutputText, required: true, title: "Transcript",
                                    help: "The transcript of the whole recording, window after window."),
                            .output(NFKOutputSegments, title: "Segments",
                                    help: "The transcript's timed spans in seconds from the start of the recording, each with its text, while timestamps are on."),
                            .output(NFKOutputLanguage, title: "Language",
                                    help: "The language the speech is in, as given or as the model detected it from the first 30 seconds.")],
                  parameters: [.option(NFKMLXGenerationParameterKey.timestamps, kind: NFKModelOptionKindBoolean, title: "Timestamps",
                                       help: "Decodes the transcript with its segment times, which the reference's transcribe does by default.",
                                       defaultValue: true),
                               .option(NFKParameterSourceLanguage, kind: NFKModelOptionKindChoice, title: "Language",
                                       help: "The language spoken, by Whisper's code. Absent, the model detects it from the first 30 seconds.",
                                       choices: NFKMLXWhisper.languageCodes(for: release.variant).map { ($0, $0) }),
                               .option(NFKMLXGenerationParameterKey.speechTask, kind: NFKModelOptionKindChoice, title: "Task",
                                       help: "Transcribes the speech in its own language, or translates it into English text.",
                                       defaultValue: NFKMLXWhisper.transcribeTask,
                                       choices: [(NFKMLXWhisper.transcribeTask, "Transcribe"), (NFKMLXWhisper.translateTask, "Translate to English")])],
                  license: .mit(url: "https://github.com/openai/whisper/blob/main/LICENSE",
                                note: "The tokenizer files come from the release's Hugging Face repository, under the same license."),
                  release: release.release,
                  source: .hub(release.repository, revision: release.revision,
                               files: NFKMLXWhisper.tokenizerFiles.map { NFKModelFile(path: $0, role: NFKModelFileRoleTokenizer) }
                                   + [NFKModelFile(path: URL(string: release.url)?.lastPathComponent ?? release.release, byteCount: nil,
                                                   sha256: nil, role: NFKModelFileRoleWeights, optional: false, repository: nil,
                                                   revision: nil, url: URL(string: release.url), extensions: nil)]),
                  precision: "float16")
        }
    }

    static var parakeet: NFKMLXCatalogEntry {
        let name = NFKMLXParakeet.modelName
        func release(_ identifier: String, _ displayName: String, _ archive: String, isDefault: Bool) -> NFKModelVariant {
            .variant(identifier, displayName: displayName, isDefault: isDefault, registryName: name,
                     source: .hub("nvidia/parakeet-tdt-\(identifier)", layout: NFKModelSourceLayoutFile,
                                  files: [NFKModelFile(path: archive, role: NFKModelFileRoleWeights)]))
        }
        return NFKMLXCatalogEntry(
            identifier: name, displayName: "Parakeet-TDT 0.6B", family: "parakeet",
            summary: "NVIDIA's FastConformer speech recognizer with a token-and-duration transducer decoder.",
            registryName: name, tasks: [NFKModelTaskTranscription],
            inputs: [audioInput("A WAV recording of speech. The backend resamples it to 16 kHz.")],
            outputs: [.output(NFKOutputText, required: true, title: "Transcript", help: "The transcript."),
                      .output(NFKOutputSegments, required: true, title: "Tokens",
                              help: "One timed span per decoded token, labeled with its text.")],
            variants: [release("0.6b-v2", "Parakeet-TDT 0.6B v2 (English)", NFKMLXParakeet.weightFiles[0], isDefault: true),
                       release("0.6b-v3", "Parakeet-TDT 0.6B v3 (multilingual)", NFKMLXParakeet.weightFiles[1], isDefault: false)],
            license: ccBy4)
    }

    static var canary: NFKMLXCatalogEntry {
        model(NFKMLXCanary.modelName, displayName: "Canary-1B-v2", family: "canary",
              summary: "NVIDIA's multitask speech model, which transcribes speech and translates it between English and 24 European languages.",
              tasks: [NFKModelTaskTranscription, NFKModelTaskTranslation],
              inputs: [audioInput("A WAV recording of speech. The backend resamples it to 16 kHz."),
                       .input(NFKInputPrompt, title: "Languages",
                              help: "A language code such as en, which transcribes, or a source>target pair such as en>de, which translates. English when absent.")],
              outputs: [.output(NFKOutputText, required: true, title: "Text", help: "The transcript or the translation.")],
              license: ccBy4,
              release: "1b-v2",
              source: .hub("nvidia/canary-1b-v2",
                           files: files(required: NFKMLXCanary.requiredFiles, optional: NFKMLXCanary.optionalFiles,
                                        weights: NFKMLXCanary.weightFiles)))
    }

    static var voxtral: NFKMLXCatalogEntry {
        model(NFKMLXVoxtral.voxtralModelName, displayName: "Voxtral Mini 3B", family: "voxtral",
              summary: "Mistral's speech language model: a Whisper large-v3 audio encoder feeding a Ministral 3B decoder.",
              tasks: [NFKModelTaskTranscription],
              inputs: [audioInput("A WAV recording of speech. The backend reads its first 30 seconds at 16 kHz."),
                       .input(NFKInputPrompt, title: "Language",
                              help: "The language code of the speech, such as en. English when absent.")],
              outputs: [.output(NFKOutputText, required: true, title: "Transcript", help: "The transcript.")],
              parameters: [maximumTokens],
              license: .apache2(),
              release: "2507",
              source: .hub("mistralai/Voxtral-Mini-3B-2507",
                           files: files(required: NFKMLXVoxtral.requiredFiles, optional: NFKMLXVoxtral.optionalFiles,
                                        weights: [NFKMLXVoxtral.weightFiles[1]] + shards("model", count: 2))))
    }

    static var graniteSpeech: NFKMLXCatalogEntry {
        model(NFKMLXGraniteSpeech.graniteSpeechModelName, displayName: "Granite Speech 3.3 2B", family: "granite-speech",
              summary: "IBM's speech language model: a Conformer encoder and a Q-former projector feeding a Granite decoder with an audio LoRA adapter.",
              tasks: [NFKModelTaskTranscription],
              inputs: [audioInput("A WAV recording of speech."),
                       .input(NFKInputPrompt, title: "Instruction",
                              help: "The instruction the model follows. A request to transcribe the speech when absent.")],
              outputs: [.output(NFKOutputText, required: true, title: "Text", help: "The model's answer, a transcript by default.")],
              parameters: [maximumTokens],
              license: .apache2(),
              release: "3.3-2b",
              source: .hub("ibm-granite/granite-speech-3.3-2b",
                           files: files(required: NFKMLXGraniteSpeech.requiredFiles,
                                        optional: NFKMLXGraniteSpeech.optionalFiles,
                                        weights: [NFKMLXGraniteSpeech.weightFiles[1]] + shards("model", count: 4))))
    }

    static var wav2Vec2: [NFKMLXCatalogEntry] {
        wav2Vec2Releases.map { release in
            let optional = NFKMLXWav2Vec2.optionalFiles.filter {
                release.transcribes || role(of: $0) != NFKModelFileRoleTokenizer
            }
            let architecture = release.family == "hubert" ? "Meta's HuBERT" : "Meta's wav2vec 2.0"
            let summary = release.transcribes
                ? "\(architecture) speech encoder fine-tuned with a CTC head, which transcribes English speech and embeds a recording."
                : "\(architecture) self-supervised speech encoder, which embeds a recording as one vector."
            var entry = model(release.name, displayName: release.displayName, family: release.family, summary: summary,
                         tasks: release.transcribes ? [NFKModelTaskTranscription, NFKModelTaskEmbedding] : [NFKModelTaskEmbedding],
                         inputs: [audioInput("A WAV recording of speech. The backend resamples it to 16 kHz.")],
                         outputs: [.output(NFKOutputText, title: "Transcript",
                                           help: "The greedy CTC transcript, written by a release with a CTC head."),
                                   .output(NFKOutputEmbedding, required: true, title: "Embedding",
                                           help: "The encoder's last hidden state averaged over the frames.")],
                         license: .apache2(),
                         release: release.name,
                         source: .hub(release.repository, revision: release.revision,
                                      files: files(required: NFKMLXWav2Vec2.requiredFiles, optional: optional,
                                                   weights: [release.weights])))
            entry.modelTypes = [release.family]
            return entry
        }
    }

    static var w2vBert: NFKMLXCatalogEntry {
        var entry = model(NFKMLXWav2Vec2Bert.modelName, displayName: "W2V-BERT 2.0", family: "w2v-bert",
              summary: "Meta's multilingual Conformer speech encoder behind Seamless, which embeds a recording as one vector.",
              tasks: [NFKModelTaskEmbedding],
              inputs: [audioInput("A WAV recording of speech. The backend resamples it to 16 kHz.")],
              outputs: [.output(NFKOutputText, title: "Transcript",
                                help: "The greedy CTC transcript, written by a fine-tuned directory with a CTC head."),
                        .output(NFKOutputEmbedding, required: true, title: "Embedding",
                                help: "The encoder's output averaged over the real frames.")],
              license: .mit(url: "https://huggingface.co/facebook/w2v-bert-2.0"),
              release: "2.0",
              source: .hub("facebook/w2v-bert-2.0", revision: "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
                           files: files(required: NFKMLXWav2Vec2Bert.requiredFiles,
                                        optional: NFKMLXWav2Vec2Bert.optionalFiles.filter { role(of: $0) == NFKModelFileRoleConfig },
                                        weights: NFKMLXWav2Vec2Bert.weightFiles)))
        entry.modelTypes = ["wav2vec2-bert"]
        return entry
    }
}

// MARK: - Voice activity and tagging

extension NFKMLXCatalogAudio {

    static var voiceActivityAndTagging: [NFKMLXCatalogEntry] {
        [sileroVAD, marbleNetVAD, audioTagger]
    }

    static var sileroVAD: NFKMLXCatalogEntry {
        model(NFKMLXSileroVAD.modelName, displayName: "Silero VAD v6", family: "silero-vad",
              summary: "A streaming voice activity detector: a short-time transform, a convolutional encoder, and an LSTM over 16 kHz audio.",
              tasks: [NFKModelTaskVoiceActivityDetection],
              inputs: [audioInput("A WAV recording.")],
              outputs: [speechSpans],
              license: .mit(url: "https://github.com/snakers4/silero-vad/blob/master/LICENSE"),
              release: "v6",
              source: urlSource("https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.jit",
                                requiresConversion: false))
    }

    static var marbleNetVAD: NFKMLXCatalogEntry {
        model(NFKMLXVAD.modelName, displayName: "Frame VAD Multilingual MarbleNet v2.0", family: "marblenet",
              summary: "NVIDIA's frame-level voice activity detector, a small one-dimensional convolutional network over 16 kHz log-mel frames.",
              tasks: [NFKModelTaskVoiceActivityDetection],
              inputs: [audioInput("A WAV recording.")],
              outputs: [speechSpans],
              license: .license(nil, name: "NVIDIA Open Model License Agreement",
                                url: "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license"),
              release: "v2.0",
              source: .hub("nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0", layout: NFKModelSourceLayoutFile,
                           files: [NFKModelFile(path: "frame_vad_multilingual_marblenet_v2.0.nemo",
                                                role: NFKModelFileRoleWeights)]))
    }

    static var audioTagger: NFKMLXCatalogEntry {
        model(NFKMLXAudioTagger.modelName, displayName: "PANNs Cnn14", family: "panns",
              summary: "A convolutional audio tagger trained on AudioSet, which scores a recording against 527 sound classes.",
              tasks: [NFKModelTaskClassification],
              inputs: [audioInput("A WAV recording.")],
              outputs: [.output(NFKOutputClassifications, required: true, title: "Tags",
                                help: "The sound classes ranked by score. A backend built without labels, as the registry builds it, names each class by its index.")],
              license: .license("CC-BY-4.0", name: "Creative Commons Attribution 4.0 International",
                                url: "https://zenodo.org/records/3987831", note: "Attribution required."),
              release: "cnn14-mAP0.431",
              source: urlSource("https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
                                fileName: "Cnn14_mAP=0.431.pth", requiresConversion: false))
    }
}

// MARK: - Source separation

extension NFKMLXCatalogAudio {

    static var separation: [NFKMLXCatalogEntry] {
        [demucs] + htDemucs + [convTasNet]
    }

    static var demucs: NFKMLXCatalogEntry {
        let configuration = NFKMLXDemucsConfiguration()
        return model(NFKMLXDemucs.modelName, displayName: "Demucs v2", family: "demucs",
                     summary: "Meta's time-domain U-Net music separator, which splits a stereo track into drums, bass, other, and vocals.",
                     tasks: [NFKModelTaskSourceSeparation],
                     inputs: [audioInput("A WAV recording of music.")],
                     outputs: stemOutputs(Array(configuration.stemNames.prefix(configuration.stems))),
                     license: demucsLicense,
                     release: "demucs-e07c671f",
                     source: urlSource("https://dl.fbaipublicfiles.com/demucs/v3.0/demucs-e07c671f.th", requiresConversion: false))
    }

    static var htDemucs: [NFKMLXCatalogEntry] {
        [NFKMLXHTDemucsVariant.fourStem, .sixStem].map { variant in
            let spec = NFKMLXHTDemucs.specs(for: variant)
            let stems = Array(NFKMLXHTDemucsConfiguration.stemNames.prefix(spec.configuration.sources))
            let sixStems = variant == .sixStem
            return model(spec.name, displayName: sixStems ? "HT Demucs 6s" : "HT Demucs", family: "htdemucs",
                         summary: sixStems
                            ? "Meta's hybrid spectrogram and waveform transformer separator, which splits a track into six stems: drums, bass, other, vocals, guitar, and piano."
                            : "Meta's hybrid spectrogram and waveform transformer separator, which splits a track into drums, bass, other, and vocals.",
                         tasks: [NFKModelTaskSourceSeparation],
                         inputs: [audioInput("A WAV recording of music. The backend resamples it to 44.1 kHz.")],
                         outputs: stemOutputs(stems),
                         license: demucsLicense,
                         release: sixStems ? "htdemucs_6s" : "htdemucs",
                         source: urlSource(htDemucsURL(variant), requiresConversion: false))
        }
    }

    static var convTasNet: NFKMLXCatalogEntry {
        let speakers = NFKMLXConvTasNetConfiguration.libri2Mix16k.speakerNames
        return model(NFKMLXConvTasNet.modelName, displayName: "Conv-TasNet Libri2Mix 16k", family: "conv-tasnet",
                     summary: "A time-domain convolutional separator, trained on Libri2Mix, which splits a mixture of two voices.",
                     tasks: [NFKModelTaskSourceSeparation],
                     inputs: [audioInput("A WAV recording of two people speaking at once.")],
                     outputs: speakers.enumerated().map { index, key in
                         .output(key, required: true, valueType: NFKModelValueTypeAudio, title: "Speaker \(index + 1)",
                                 help: "One separated voice.")
                     },
                     license: .license("CC-BY-SA-4.0", name: "Creative Commons Attribution-ShareAlike 4.0 International",
                                       url: "https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k",
                                       note: "Attribution required, and adaptations carry the same license."),
                     release: "libri2mix-sepclean-16k",
                     source: .hub("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k", layout: NFKModelSourceLayoutFile,
                                  files: [NFKModelFile(path: "pytorch_model.bin", role: NFKModelFileRoleWeights)]))
    }
}

// MARK: - Speech enhancement and restoration

extension NFKMLXCatalogAudio {

    static var enhancement: [NFKMLXCatalogEntry] {
        [denoiser, mpSENet, gtcrn, metricGANPlus, cmgan, frcrn, mossFormer2SE, mossFormer2SR, deepFilterNet,
         sgmse, storm, voiceRestore, resembleEnhance, nuWave2, apollo]
    }

    static var denoiser: NFKMLXCatalogEntry {
        model(NFKMLXDenoiser.modelName, displayName: "Denoiser DNS48", family: "denoiser",
              summary: "Meta's causal Demucs speech denoiser, trained on the DNS challenge data.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of noisy speech.")],
              outputs: [enhancedAudio],
              license: .license("CC-BY-NC-4.0", name: "Creative Commons Attribution-NonCommercial 4.0 International",
                                url: "https://github.com/facebookresearch/denoiser/blob/main/LICENSE",
                                restrictsCommercialUse: true, note: "Non-commercial use only."),
              release: "dns48",
              source: urlSource("https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th",
                                requiresConversion: false))
    }

    static var mpSENet: NFKMLXCatalogEntry {
        model(NFKMLXMPSENetFactory.modelName, displayName: "MP-SENet", family: "mp-senet",
              summary: "A speech enhancer that predicts the magnitude and the phase of a 16 kHz spectrogram in parallel.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of noisy speech.")],
              outputs: [enhancedAudio],
              license: .mit(url: "https://github.com/yxlu-0102/MP-SENet/blob/main/LICENSE"),
              release: "dns",
              source: urlSource("https://raw.githubusercontent.com/yxlu-0102/MP-SENet/main/best_ckpt/g_best_dns",
                                requiresConversion: false))
    }

    static var gtcrn: NFKMLXCatalogEntry {
        model(NFKMLXGTCRNFactory.modelName, displayName: "GTCRN", family: "gtcrn",
              summary: "An ultra-light real-time speech denoiser of about 48 thousand parameters, over a 16 kHz spectrogram.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of noisy speech.")],
              outputs: [enhancedAudio],
              license: .mit(url: "https://github.com/Xiaobin-Rong/gtcrn/blob/main/LICENSE"),
              release: "dns3",
              source: urlSource("https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/main/checkpoints/model_trained_on_dns3.tar",
                                requiresConversion: false))
    }

    static var metricGANPlus: NFKMLXCatalogEntry {
        model(NFKMLXMetricGANPlus.modelName, displayName: "MetricGAN+", family: "metricgan",
              summary: "SpeechBrain's spectral-mask speech enhancer, a bidirectional LSTM trained on VoiceBank-DEMAND.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of noisy speech. The backend resamples it to 16 kHz.")],
              outputs: [enhancedAudio],
              license: .apache2(),
              release: "voicebank",
              source: .hub("speechbrain/metricgan-plus-voicebank", layout: NFKModelSourceLayoutFile,
                           files: [NFKModelFile(path: "enhance_model.ckpt", role: NFKModelFileRoleWeights)]))
    }

    static var cmgan: NFKMLXCatalogEntry {
        model(NFKMLXCMGAN.modelName, displayName: "CMGAN", family: "cmgan",
              summary: "A conformer-based speech enhancer with magnitude-mask and complex decoders, trained on VoiceBank-DEMAND.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of noisy speech. The backend resamples it to 16 kHz.")],
              outputs: [enhancedAudio],
              license: .mit(url: "https://github.com/ruizhecao96/CMGAN/blob/main/LICENSE"),
              release: "best-ckpt",
              source: urlSource("https://raw.githubusercontent.com/ruizhecao96/CMGAN/main/src/best_ckpt/ckpt",
                                requiresConversion: false))
    }

    static var frcrn: NFKMLXCatalogEntry {
        model(NFKMLXFRCRN.modelName, displayName: "FRCRN SE 16K", family: "frcrn",
              summary: "ClearerVoice's speech enhancer: two complex U-Nets with frequency-recurrent memories over a 16 kHz spectrogram.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of noisy speech. The backend resamples it to 16 kHz.")],
              outputs: [enhancedAudio],
              license: .apache2(),
              release: "se-16k",
              source: .hub("alibabasglab/FRCRN_SE_16K", layout: NFKModelSourceLayoutFile,
                           files: [NFKModelFile(path: "last_best_checkpoint.pt", role: NFKModelFileRoleWeights)]))
    }

    static var mossFormer2SE: NFKMLXCatalogEntry {
        model(NFKMLXMossFormer2Factory.modelName, displayName: "MossFormer2 SE 48K", family: "mossformer2",
              summary: "ClearerVoice's full-band 48 kHz speech enhancer, gated attention and memory blocks that mask a filterbank.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of noisy speech.")],
              outputs: [enhancedAudio],
              license: .apache2(),
              release: "se-48k",
              source: .hub("alibabasglab/MossFormer2_SE_48K", layout: NFKModelSourceLayoutFile,
                           files: [NFKModelFile(path: "last_best_checkpoint.pt", role: NFKModelFileRoleWeights)]))
    }

    static var mossFormer2SR: NFKMLXCatalogEntry {
        model(NFKMLXMossFormer2SRFactory.modelName, displayName: "MossFormer2 SR 48K", family: "mossformer2",
              summary: "ClearerVoice's speech super-resolution: a mel-to-mel MossFormer2 and a HiFi-GAN generator that extend speech to 48 kHz.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of band-limited speech. The backend resamples it to 48 kHz.")],
              outputs: [.output(NFKOutputAudio, required: true, title: "Audio", help: "The bandwidth-extended clip at 48 kHz.")],
              license: .apache2(),
              release: "sr-48k",
              source: .hub("alibabasglab/MossFormer2_SR_48K",
                           files: files(required: NFKMLXMossFormer2SRFactory.requiredFiles,
                                        optional: NFKMLXMossFormer2SRFactory.optionalFiles,
                                        weights: NFKMLXMossFormer2SRFactory.weightFiles)))
    }

    static var deepFilterNet: NFKMLXCatalogEntry {
        model(NFKMLXDeepFilterNetFactory.modelName, displayName: "DeepFilterNet3", family: "deepfilternet",
              summary: "A real-time 48 kHz speech denoiser that applies an ERB-band gain and a deep filter to the low frequencies.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of noisy speech. The backend resamples it to 48 kHz.")],
              outputs: [enhancedAudio],
              license: .license("MIT OR Apache-2.0", name: "MIT License or Apache License 2.0",
                                url: "https://github.com/Rikorose/DeepFilterNet#license"),
              release: "deepfilternet3",
              source: urlSource("https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3.zip",
                                requiresConversion: true))
    }

    static var sgmse: NFKMLXCatalogEntry {
        model(NFKMLXSGMSE.modelName, displayName: "SGMSE+", family: "sgmse",
              summary: "A score-based generative speech enhancer that samples the clean spectrogram with a reverse diffusion process.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of noisy speech.")],
              outputs: [enhancedAudio],
              license: .mit(url: "https://huggingface.co/sp-uhh/speech-enhancement-sgmse/blob/main/LICENSE"),
              release: "voicebank",
              source: .hub("sp-uhh/speech-enhancement-sgmse", layout: NFKModelSourceLayoutFile,
                           files: [NFKModelFile(path: "train_vb_29nqe0uh_epoch=115.ckpt", role: NFKModelFileRoleWeights)],
                           requiresConversion: true))
    }

    static var storm: NFKMLXCatalogEntry {
        model(NFKMLXStoRM.modelName, displayName: "StoRM", family: "storm",
              summary: "A stochastic-regeneration speech enhancer: a predictive network followed by a few diffusion steps.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of noisy or reverberant speech.")],
              outputs: [enhancedAudio],
              license: .mit(url: "https://github.com/sp-uhh/storm/blob/master/LICENSE"),
              release: "storm",
              source: NFKModelSource(kind: NFKModelSourceKindURL, repository: nil, revision: nil,
                                     url: URL(string: "https://drive.google.com/drive/folders/1ExFm97obaXTYFoBApWjbK_ypxTP-Cgdq"),
                                     layout: NFKModelSourceLayoutFile, files: nil, requiresConversion: true,
                                     extensions: nil))
    }

    static var voiceRestore: NFKMLXCatalogEntry {
        model(NFKMLXVoiceRestoreFactory.modelName, displayName: "VoiceRestore", family: "voicerestore",
              summary: "A flow-matching speech restorer that removes noise, reverberation, clipping, and band limits, voiced by BigVGAN v2.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of degraded speech. The backend resamples it to 24 kHz.")],
              outputs: [enhancedAudio],
              license: .mit(url: "https://huggingface.co/jadechoghari/VoiceRestore/blob/main/LICENSE",
                            note: "The vocoder, bigvgan_generator.pt from nvidia/bigvgan_v2_24khz_100band_256x, is MIT too."),
              release: "voicerestore",
              source: .hub("jadechoghari/VoiceRestore",
                           files: [NFKModelFile(path: NFKMLXVoiceRestoreFactory.weightFiles[1], role: NFKModelFileRoleWeights),
                                   .file(NFKMLXVoiceRestoreFactory.vocoderFile, role: NFKModelFileRoleWeights,
                                         repository: NFKMLXVoiceRestoreFactory.vocoderRepo)]))
    }

    static var resembleEnhance: NFKMLXCatalogEntry {
        model(NFKMLXResembleEnhanceFactory.modelName, displayName: "Resemble Enhance", family: "resemble-enhance",
              summary: "Resemble AI's speech enhancer: a spectrogram denoiser, a latent flow-matching model, and a UnivNet vocoder at 44.1 kHz.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of degraded speech. The backend resamples it to 44.1 kHz.")],
              outputs: [enhancedAudio],
              license: .mit(url: "https://github.com/resemble-ai/resemble-enhance/blob/main/LICENSE"),
              release: "enhancer-stage2",
              source: .hub("ResembleAI/resemble-enhance", layout: NFKModelSourceLayoutFile,
                           files: [NFKModelFile(path: NFKMLXResembleEnhanceFactory.checkpointFile, role: NFKModelFileRoleWeights)]))
    }

    static var nuWave2: NFKMLXCatalogEntry {
        model(NFKMLXNUWave2.modelName, displayName: "NU-Wave 2", family: "nu-wave",
              summary: "A diffusion model that extends narrow-band audio to 48 kHz.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of band-limited audio at its own sample rate.")],
              outputs: [.output(NFKOutputAudio, required: true, title: "Audio", help: "The bandwidth-extended clip at 48 kHz.")],
              parameters: [.option(NFKParameterSeed, kind: NFKModelOptionKindInteger, title: "Seed",
                                   help: "The seed of the diffusion's initial noise.", defaultValue: 0, minimum: 0, step: 1)],
              license: .license("BSD-3-Clause", name: "BSD 3-Clause License",
                                url: "https://github.com/maum-ai/nuwave2/blob/master/LICENSE"),
              release: "official",
              source: NFKModelSource(kind: NFKModelSourceKindURL, repository: nil, revision: nil,
                                     url: URL(string: "https://drive.google.com/file/d/11t0cQYx6ZadKQjmfGnqxUUH2UEk5Yzk7/view"),
                                     layout: NFKModelSourceLayoutFile, files: nil, requiresConversion: false,
                                     extensions: nil))
    }

    static var apollo: NFKMLXCatalogEntry {
        model(NFKMLXApollo.modelName, displayName: "Apollo", family: "apollo",
              summary: "A band-split Roformer that restores the high frequencies of lossy-compressed music at 44.1 kHz.",
              tasks: [NFKModelTaskSpeechEnhancement],
              inputs: [audioInput("A WAV recording of lossy-compressed music. The backend resamples it to 44.1 kHz.")],
              outputs: [.output(NFKOutputAudio, required: true, title: "Audio", help: "The restored clip.")],
              license: .license("CC-BY-SA-4.0", name: "Creative Commons Attribution-ShareAlike 4.0 International",
                                url: "https://huggingface.co/JusperLee/Apollo",
                                note: "Attribution required, and adaptations carry the same license."),
              release: "apollo",
              source: .hub("JusperLee/Apollo", layout: NFKModelSourceLayoutFile,
                           files: [NFKModelFile(path: "pytorch_model.bin", role: NFKModelFileRoleWeights)]))
    }
}

// MARK: - Codecs

extension NFKMLXCatalogAudio {

    static var codecs: [NFKMLXCatalogEntry] {
        [dac] + snac + [mimi, bigVGAN]
    }

    static var dac: NFKMLXCatalogEntry {
        model(NFKMLXDAC.modelName, displayName: "Descript Audio Codec 44.1 kHz", family: "dac",
              summary: "Descript's neural audio codec, which encodes 44.1 kHz audio into nine residual codebooks at 8 kbps.",
              tasks: [NFKModelTaskAudioCodec],
              inputs: [audioInput("A WAV recording.")],
              outputs: [reconstructedAudio],
              license: .mit(url: "https://github.com/descriptinc/descript-audio-codec/blob/main/LICENSE"),
              release: "44khz-8kbps",
              source: urlSource("https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth",
                                requiresConversion: false))
    }

    static var snac: [NFKMLXCatalogEntry] {
        [NFKMLXSNACVariant.speech24kHz, .music32kHz, .music44kHz].map { variant in
            let spec = NFKMLXSNAC.specs(for: variant)
            let (displayName, summary): (String, String) = {
                switch variant {
                case .speech24kHz:
                    return ("SNAC 24 kHz", "A multi-scale neural speech codec with three codebooks at 24 kHz.")
                case .music32kHz:
                    return ("SNAC 32 kHz", "A multi-scale neural music codec with four codebooks at 32 kHz.")
                case .music44kHz:
                    return ("SNAC 44.1 kHz", "A multi-scale neural music codec with four codebooks at 44.1 kHz.")
                }
            }()
            return model(spec.name, displayName: displayName, family: "snac", summary: summary,
                         tasks: [NFKModelTaskAudioCodec],
                         inputs: [audioInput("A WAV recording.")],
                         outputs: [reconstructedAudio],
                         license: .mit(url: "https://github.com/hubertsiuzdak/snac/blob/main/LICENSE"),
                         release: String(snacRepository(variant).split(separator: "/").last ?? ""),
                         source: .hub(snacRepository(variant), layout: NFKModelSourceLayoutFile,
                                      files: [NFKModelFile(path: "pytorch_model.bin", role: NFKModelFileRoleWeights)]))
        }
    }

    static var mimi: NFKMLXCatalogEntry {
        model(NFKMLXMimi.modelName, displayName: "Mimi", family: "mimi",
              summary: "Kyutai's streaming neural codec, which encodes 24 kHz audio into 32 codebooks at 12.5 frames a second.",
              tasks: [NFKModelTaskAudioCodec],
              inputs: [audioInput("A WAV recording.")],
              outputs: [reconstructedAudio],
              license: ccBy4,
              release: "mimi",
              source: .hub("kyutai/mimi", layout: NFKModelSourceLayoutFile,
                           files: [NFKModelFile(path: "model.safetensors", role: NFKModelFileRoleWeights)]))
    }

    static var bigVGAN: NFKMLXCatalogEntry {
        model(NFKMLXBigVGANFactory.modelName, displayName: "BigVGAN v2 24 kHz", family: "bigvgan",
              summary: "NVIDIA's universal neural vocoder for 100-band mel spectrograms at 24 kHz.",
              tasks: [NFKModelTaskAudioCodec],
              inputs: [audioInput("A WAV recording. The backend takes its mel spectrogram.")],
              outputs: [.output(NFKOutputAudio, required: true, title: "Audio",
                                help: "The waveform the vocoder generates from the recording's mel spectrogram.")],
              license: .mit(url: "https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/blob/main/LICENSE"),
              release: "24khz-100band-256x",
              source: .hub(NFKMLXVoiceRestoreFactory.vocoderRepo, layout: NFKModelSourceLayoutFile,
                           files: [NFKModelFile(path: NFKMLXVoiceRestoreFactory.vocoderFile, role: NFKModelFileRoleWeights)]))
    }
}

// MARK: - Music

extension NFKMLXCatalogAudio {

    static var music: [NFKMLXCatalogEntry] {
        [basicPitch, hftTransformer] + muScriptor + [allInOne, music3, stableAudioOpen]
    }

    static var basicPitch: NFKMLXCatalogEntry {
        let defaults = NFKMLXBasicPitchOptions.default
        return model(NFKMLXBasicPitch.modelName, displayName: "Basic Pitch", family: "basic-pitch",
                     summary: "Spotify's lightweight polyphonic note transcriber, which writes notes with pitch bends for any instrument.",
                     tasks: [NFKModelTaskMusicTranscription],
                     inputs: [audioInput("A WAV recording of music. The backend resamples it to 22.05 kHz.")],
                     outputs: [transcribedNotes],
                     parameters: [
                        onsetThreshold(decimal(defaults.onsetThreshold)),
                        .option(NFKMLXTranscriptionParameterKey.frameThreshold, kind: NFKModelOptionKindNumber,
                                title: "Frame threshold", help: "The note activation below which a sounding note ends.",
                                defaultValue: decimal(defaults.frameThreshold), minimum: 0, maximum: 1, step: 0.05),
                        .option(NFKMLXTranscriptionParameterKey.minimumNoteFrames, kind: NFKModelOptionKindInteger,
                                title: "Minimum note frames", help: "The frames a note must last to be kept.",
                                defaultValue: defaults.minimumNoteFrames, minimum: 1, step: 1),
                        .option(NFKMLXTranscriptionParameterKey.pitchBends, kind: NFKModelOptionKindBoolean,
                                title: "Pitch bends", help: "Whether each note carries a pitch-bend curve.",
                                defaultValue: defaults.includesPitchBends),
                        .option(NFKMLXTranscriptionParameterKey.melodiaTrick, kind: NFKModelOptionKindBoolean,
                                title: "Melodia trick", help: "Whether note creation picks up notes the onset head missed.",
                                defaultValue: defaults.melodiaTrick),
                        .option(NFKMLXTranscriptionParameterKey.minimumFrequency, kind: NFKModelOptionKindNumber,
                                title: "Minimum frequency",
                                help: "The lowest frequency to transcribe, in hertz. No floor when absent.", minimum: 0),
                        .option(NFKMLXTranscriptionParameterKey.maximumFrequency, kind: NFKModelOptionKindNumber,
                                title: "Maximum frequency",
                                help: "The highest frequency to transcribe, in hertz. No ceiling when absent.", minimum: 0),
                        tempo(defaults.tempoBPM),
                        program(defaults.program),
                     ],
                     license: .apache2(),
                     release: "icassp-2022",
                     source: urlSource("https://raw.githubusercontent.com/spotify/basic-pitch/main/basic_pitch/saved_models/icassp_2022/nmp.onnx",
                                       requiresConversion: true))
    }

    static var hftTransformer: NFKMLXCatalogEntry {
        let defaults = NFKMLXHFTTransformerOptions.default
        return model(NFKMLXHFTTransformer.modelName, displayName: "hFT-Transformer", family: "hft-transformer",
                     summary: "Sony's hierarchical frequency-time transformer, which transcribes solo piano with onsets, offsets, and velocities.",
                     tasks: [NFKModelTaskMusicTranscription],
                     inputs: [audioInput("A WAV recording of solo piano. The backend resamples it to 16 kHz.")],
                     outputs: [transcribedNotes],
                     parameters: [
                        onsetThreshold(decimal(defaults.onsetThreshold)),
                        .option(NFKMLXTranscriptionParameterKey.frameThreshold, kind: NFKModelOptionKindNumber,
                                title: "Frame threshold",
                                help: "The multi-pitch activation below which a sounding note ends.",
                                defaultValue: decimal(defaults.multiPitchThreshold), minimum: 0, maximum: 1, step: 0.05),
                        tempo(defaults.tempoBPM),
                        program(defaults.program),
                     ],
                     license: .mit(url: "https://github.com/sony/hFT-Transformer/blob/master/LICENSE"),
                     release: "maestro-v3",
                     source: urlSource("https://github.com/sony/hFT-Transformer/releases/download/ismir2023/checkpoint.zip",
                                       requiresConversion: true))
    }

    static var muScriptor: [NFKMLXCatalogEntry] {
        [NFKMLXMuScriptorVariant.small, .medium, .large].map { variant in
            let configuration = variant.configuration
            let name = NFKMLXMuScriptor.registeredName(for: variant)
            return model(name, displayName: "MuScriptor \(variant.name)", family: "muscriptor",
                         summary: "A multi-instrument transcriber: a decoder that writes MT3 note tokens from a 16 kHz mel spectrogram, one track per instrument.",
                         tasks: [NFKModelTaskMusicTranscription],
                         inputs: [audioInput("A WAV recording of music. The backend resamples it to 16 kHz.")],
                         outputs: [transcribedNotes],
                         parameters: [
                            .option(NFKMLXMuScriptorParameterKey.instrumentGroup, kind: NFKModelOptionKindInteger,
                                    title: "Instrument group",
                                    help: "The instrument group the model expects to hear. The model decides when absent.",
                                    minimum: 0, maximum: NSNumber(value: configuration.instrumentClasses - 1), step: 1),
                            .option(NFKMLXMuScriptorParameterKey.datasetClass, kind: NFKModelOptionKindInteger,
                                    title: "Dataset class",
                                    help: "The dataset class the release was conditioned on in training. The backend does not pass it to the model.",
                                    minimum: 0, maximum: NSNumber(value: configuration.datasetClasses - 1), step: 1),
                         ],
                         license: .license("CC-BY-NC-4.0", name: "Creative Commons Attribution-NonCommercial 4.0 International",
                                           url: "https://huggingface.co/\(variant.repository)", gated: true,
                                           restrictsCommercialUse: true,
                                           note: "Gated: accept the terms on Hugging Face. Non-commercial use only."),
                         release: variant.name,
                         source: .hub(variant.repository, layout: NFKModelSourceLayoutFile,
                                      files: [NFKModelFile(path: "model.safetensors", role: NFKModelFileRoleWeights)]))
        }
    }

    static var allInOne: NFKMLXCatalogEntry {
        model(NFKMLXAllInOne.modelName, displayName: "All-In-One (Harmonix)", family: "all-in-one",
              summary: "A music structure analyzer that finds the beats, downbeats, tempo, and functional sections of a track from its four stems.",
              tasks: [NFKModelTaskMusicStructure],
              inputs: [.input(NFKInputAudio, required: true, valueType: NFKModelValueTypeAudios, title: "Stems",
                              help: "The four stems of a track, in the order bass, drums, other, vocals. A backend built with an HT Demucs separator also takes the mixture.")],
              outputs: [.output(NFKOutputSegments, required: true, title: "Sections",
                                help: "The functional sections, each labeled, such as verse or chorus."),
                        .output(NFKOutputBeats, required: true, title: "Beats", help: "The beats, with the downbeats marked."),
                        .output(NFKOutputTempo, title: "Tempo",
                                help: "The tempo in beats per minute, written when the analysis finds one.")],
              license: .mit(url: "https://github.com/mir-aidj/all-in-one/blob/main/LICENSE"),
              release: "harmonix-fold0",
              source: .hub("taejunkim/allinone", layout: NFKModelSourceLayoutFile,
                           files: [NFKModelFile(path: "harmonix-fold0-0vra4ys2.pth", role: NFKModelFileRoleWeights)],
                           requiresConversion: true))
    }

    static var music3: NFKMLXCatalogEntry {
        let components = NFKMLXMusic3.releaseComponents
        let required = components.flatMap(\.required)
        let indexSuffix = ".safetensors.index.json"
        let shardCounts = ["language_model": 4, "transformer": 2]
        let weights = components.flatMap { component -> [String] in
            guard let index = component.weights.first(where: { $0.hasSuffix(indexSuffix) }) else { return [] }
            let directory = (index as NSString).deletingLastPathComponent
            return [index] + shards(String(index.dropLast(indexSuffix.count)), count: shardCounts[directory] ?? 1)
        }
        let maximumSeconds = Double(NFKMusic3Contract.maxAudioFrames) / NFKMusic3Contract.framesPerSecond
        return model(NFKMLXMusic3.modelName, displayName: "MiniMax Music 3", family: "minimax-music",
                     summary: "MiniMax's song generator: a language model, a flow-matching transformer, and a vocoder that sing lyrics in a described style as stereo 44.1 kHz audio.",
                     tasks: [NFKModelTaskMusicGeneration],
                     inputs: [.input(NFKInputPrompt, required: true, title: "Description",
                                     help: "The style of the music: genre, mood, instruments, and voice."),
                              .input(NFKInputLyrics, required: true, title: "Lyrics", help: "The lyrics to sing.")],
                     outputs: [.output(NFKOutputAudio, required: true, title: "Song", help: "A stereo WAV file at 44.1 kHz.")],
                     parameters: [
                        .option(NFKParameterDurationSeconds, kind: NFKModelOptionKindNumber, title: "Duration",
                                help: "The longest the song runs, in seconds. The model may end it earlier.",
                                defaultValue: 10, minimum: 1, maximum: NSNumber(value: maximumSeconds), step: 1),
                        .option(NFKParameterSteps, kind: NFKModelOptionKindInteger, title: "Steps",
                                help: "The flow-matching steps per window.", defaultValue: 30, minimum: 1, step: 1),
                        .option(NFKParameterGuidanceScale, kind: NFKModelOptionKindNumber, title: "Guidance",
                                help: "The flow-matching transformer's classifier-free guidance weight.",
                                defaultValue: 1.7, minimum: 0, step: 0.1),
                        .option(NFKParameterSeed, kind: NFKModelOptionKindInteger, title: "Seed",
                                help: "The seed of the sampling.", defaultValue: 0, minimum: 0, step: 1),
                     ],
                     license: .license(nil, name: "MiniMax-Music3 Community License",
                                       url: "https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE",
                                       restrictsCommercialUse: true,
                                       note: "A commercial product shows \"MiniMax-Music3\" in its interface, and one above USD 20 million in yearly revenue needs MiniMax's written authorization."),
                     release: "music3",
                     source: .hub("MiniMaxAI/MiniMax-Music3", files: files(required: required, weights: weights)),
                     residencies: [NFKModelResidencyResident, NFKModelResidencyStaged])
    }

    static var stableAudioOpen: NFKMLXCatalogEntry {
        let components = NFKMLXStableAudioGenerator.releaseComponents
        let maximumSeconds = NSNumber(value: 1024.0 * 2048 / 44100)
        return model(NFKMLXStableAudioGenerator.modelName, displayName: "Stable Audio Open 1.0", family: "stable-audio",
                     summary: "Stability AI's latent diffusion generator of sound effects, field recordings, and short music samples from a description, as stereo 44.1 kHz audio.",
                     tasks: [NFKModelTaskAudioGeneration, NFKModelTaskMusicGeneration],
                     inputs: [.input(NFKInputPrompt, required: true, title: "Description",
                                     help: "The sound to make: its source, character, and, for music, genre, instruments, and tempo."),
                              .input(NFKInputNegativePrompt, title: "Negative prompt",
                                     help: "What the clip steers away from."),
                              .input(NFKInputAudio, title: "Initial audio",
                                     help: "A 16-bit WAV clip at 44.1 kHz the generation starts from.")],
                     outputs: [.output(NFKOutputAudio, required: true, title: "Audio", help: "A stereo WAV file at 44.1 kHz.")],
                     parameters: [
                        .option(NFKParameterDurationSeconds, kind: NFKModelOptionKindNumber, title: "Duration",
                                help: "The clip's length in seconds.",
                                defaultValue: 10, minimum: 0.5, maximum: maximumSeconds, step: 0.5),
                        .option(NFKMLXStableAudioParameterKey.startSeconds, kind: NFKModelOptionKindNumber, title: "Start",
                                help: "Where the clip starts within the model's window, in seconds. The clip ends within the window.",
                                defaultValue: 0, minimum: 0, maximum: maximumSeconds, step: 0.5),
                        .option(NFKParameterSteps, kind: NFKModelOptionKindInteger, title: "Steps",
                                help: "The sampling steps.", defaultValue: 100, minimum: 1,
                                maximum: NSNumber(value: NFKMLXStableAudioGenerator.maximumSteps), step: 1),
                        .option(NFKParameterGuidanceScale, kind: NFKModelOptionKindNumber, title: "Guidance",
                                help: "The classifier-free guidance weight. At 1 or below the clip is unguided.",
                                defaultValue: 7, minimum: 0, step: 0.5),
                        .option(NFKParameterSeed, kind: NFKModelOptionKindInteger, title: "Seed",
                                help: "The seed of the sampling.", defaultValue: 0, minimum: 0, step: 1),
                     ],
                     license: NFKMLXCatalogImage.gatedStabilityCommunity,
                     release: "1.0",
                     source: .hub(NFKMLXStableAudioGenerator.releaseRepo,
                                  files: files(required: components.flatMap(\.required),
                                               optional: components.flatMap(\.optional),
                                               weights: components.compactMap(\.weights.first))),
                     residencies: [NFKModelResidencyResident, NFKModelResidencyStaged])
    }
}

// MARK: - Speech synthesis

extension NFKMLXCatalogAudio {

    static var speechSynthesis: [NFKMLXCatalogEntry] {
        [fastSpeech2Voice, kokoro, chatterbox]
    }

    static var fastSpeech2Voice: NFKMLXCatalogEntry {
        model(NFKMLXFastSpeech2.voiceModelName, displayName: "FastSpeech2 Conformer LJSpeech", family: "fastspeech2",
              summary: "ESPnet's conformer FastSpeech2 and its paired HiFi-GAN, an English single-speaker voice at 22.05 kHz.",
              tasks: [NFKModelTaskSpeechSynthesis],
              inputs: speechInputs(prompt: "The English text to speak. The G2P checkpoint in the directory phonemizes it."),
              outputs: [synthesizedSpeech("A mono WAV file at 22.05 kHz.")],
              license: .apache2(note: "The registry also reads a G2P checkpoint saved as g2p.safetensors beside the release, which the release does not include."),
              release: "ljspeech",
              source: .hub(NFKMLXFastSpeech2.voiceReleaseRepo,
                           files: files(required: ["config.json", "vocab.json"], weights: ["pytorch_model.bin"])))
    }

    static var kokoro: NFKMLXCatalogEntry {
        let release = NFKMLXKokoro.releaseFiles(voiceName: kokoroVoice)
        return model(kokoroName, displayName: "Kokoro-82M", family: "kokoro",
                     summary: "A small StyleTTS 2 and iSTFTNet voice that speaks a phoneme string at 24 kHz.",
                     tasks: [NFKModelTaskSpeechSynthesis],
                     inputs: speechInputs(prompt: "The phonemes to speak, in the misaki phoneme alphabet Kokoro reads."),
                     outputs: [synthesizedSpeech("A mono WAV file in the af_heart voice.")],
                     parameters: [generatedSampleRate],
                     license: .apache2(),
                     release: "v1.0",
                     source: .hub("hexgrad/Kokoro-82M", files: files(required: release.required, weights: release.weights)))
    }

    static var chatterbox: NFKMLXCatalogEntry {
        let defaults = NFKMLXT3SamplingOptions()
        return model(chatterboxName, displayName: "Chatterbox", family: "chatterbox",
                     summary: "Resemble AI's English voice-cloning speech synthesizer: a Llama text-to-token model and a flow-matching token-to-speech decoder at 24 kHz.",
                     tasks: [NFKModelTaskSpeechSynthesis],
                     inputs: speechInputs(prompt: "The text to speak."),
                     outputs: [synthesizedSpeech("A mono WAV file in the release's built-in voice.")],
                     parameters: [
                        .option(NFKMLXChatterboxParameterKey.exaggeration, kind: NFKModelOptionKindNumber,
                                title: "Exaggeration", help: "How strongly the voice's emotion is exaggerated.",
                                defaultValue: 0.5, minimum: 0, step: 0.05),
                        .option(NFKParameterTemperature, kind: NFKModelOptionKindNumber, title: "Temperature",
                                help: "The sampling temperature of the speech tokens. 0 picks the most likely token at every step.",
                                defaultValue: decimal(defaults.temperature), minimum: 0, step: 0.05),
                        .option(NFKParameterGuidanceScale, kind: NFKModelOptionKindNumber, title: "Guidance",
                                help: "The classifier-free guidance weight of the speech tokens. 0 runs the conditional pass alone.",
                                defaultValue: decimal(defaults.guidance), minimum: 0, step: 0.05),
                        .option(NFKParameterSeed, kind: NFKModelOptionKindInteger, title: "Seed",
                                help: "The seed of the speech-token sampling.",
                                defaultValue: NSNumber(value: defaults.seed), minimum: 0, step: 1),
                        generatedSampleRate,
                     ],
                     license: .mit(url: "https://github.com/resemble-ai/chatterbox/blob/master/LICENSE"),
                     release: "english",
                     source: .hub("ResembleAI/chatterbox",
                                  files: files(required: NFKMLXChatterbox.requiredFiles + [NFKMLXChatterbox.builtinVoiceFile],
                                               optional: NFKMLXChatterbox.optionalFiles,
                                               weights: NFKMLXChatterbox.weightFiles)))
    }
}

// MARK: - Helpers

extension NFKMLXCatalogAudio {

    /// The entry of the model registered under `name`, with one release that downloads from `source`.
    fileprivate static func model(_ name: String, displayName: String, family: String, summary: String,
                                  tasks: [String], inputs: [NFKModelPort], outputs: [NFKModelPort],
                                  parameters: [NFKModelOption] = [], license: NFKModelLicense, release: String,
                                  source: NFKModelSource, precision: String? = nil,
                                  residencies: [String] = [NFKModelResidencyResident]) -> NFKMLXCatalogEntry {
        NFKMLXCatalogEntry(identifier: name, displayName: displayName, family: family, summary: summary,
                           registryName: name, tasks: tasks, inputs: inputs, outputs: outputs, parameters: parameters,
                           variants: [.variant(release, displayName: displayName, isDefault: true, registryName: name,
                                               source: source, precision: precision)],
                           license: license, residencies: residencies)
    }

    /// A single checkpoint downloaded from `url`. The file is named by the URL's last path component
    /// unless `fileName` names it.
    fileprivate static func urlSource(_ url: String, fileName: String? = nil, requiresConversion: Bool) -> NFKModelSource {
        let path = fileName ?? URL(string: url)?.lastPathComponent ?? url
        return NFKModelSource(kind: NFKModelSourceKindURL, repository: nil, revision: nil, url: URL(string: url),
                              layout: NFKModelSourceLayoutFile,
                              files: [NFKModelFile(path: path, role: NFKModelFileRoleWeights)],
                              requiresConversion: requiresConversion, extensions: nil)
    }

    /// The files of a release directory: the required files, the optional ones, and the weights.
    fileprivate static func files(required: [String], optional: [String] = [], weights: [String]) -> [NFKModelFile] {
        required.map { NFKModelFile(path: $0, role: role(of: $0)) }
            + optional.map { NFKModelFile(path: $0, byteCount: nil, sha256: nil, role: role(of: $0), optional: true,
                                          extensions: nil) }
            + weights.map { NFKModelFile(path: $0, role: NFKModelFileRoleWeights) }
    }

    /// The names of a tokenizer's files.
    private static let tokenizerFileNames: Set<String> = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "tekken.json", "added_tokens.json",
        "special_tokens_map.json",
    ]

    /// What a release file holds, judged by its name: a safetensors file holds weights, and a file that is
    /// neither weights, a tokenizer, nor a configuration is `other`.
    static func role(of path: String) -> String {
        let name = (path as NSString).lastPathComponent
        if name.hasSuffix(".safetensors") {
            return NFKModelFileRoleWeights
        }
        if tokenizerFileNames.contains(name) {
            return NFKModelFileRoleTokenizer
        }
        if name.hasSuffix("config.json") || name == "params.json" {
            return NFKModelFileRoleConfig
        }
        return NFKModelFileRoleOther
    }

    /// The shard names a `<stem>.safetensors.index.json` lists: `<stem>-00001-of-0000N.safetensors` onward.
    fileprivate static func shards(_ stem: String, count: Int) -> [String] {
        (1 ... count).map { String(format: "%@-%05d-of-%05d.safetensors", stem, $0, count) }
    }

    fileprivate static var ccBy4: NFKModelLicense {
        .license("CC-BY-4.0", name: "Creative Commons Attribution 4.0 International",
                 url: "https://creativecommons.org/licenses/by/4.0/", note: "Attribution required.")
    }

    fileprivate static var demucsLicense: NFKModelLicense {
        .mit(url: "https://github.com/facebookresearch/demucs/blob/main/LICENSE")
    }

    fileprivate static func audioInput(_ help: String) -> NFKModelPort {
        .input(NFKInputAudio, required: true, title: "Audio", help: help)
    }

    fileprivate static var enhancedAudio: NFKModelPort {
        .output(NFKOutputAudio, required: true, title: "Audio", help: "The enhanced clip.")
    }

    fileprivate static var reconstructedAudio: NFKModelPort {
        .output(NFKOutputAudio, required: true, title: "Audio",
                help: "The recording encoded into the codec's tokens and decoded back.")
    }

    fileprivate static var speechSpans: NFKModelPort {
        .output(NFKOutputSegments, required: true, title: "Speech", help: "The spans of the recording that hold speech.")
    }

    fileprivate static var transcribedNotes: NFKModelPort {
        .output(NFKOutputMIDI, required: true, title: "Notes", help: "The transcribed notes as a MIDI sequence.")
    }

    fileprivate static func stemOutputs(_ stems: [String]) -> [NFKModelPort] {
        stems.map {
            .output($0, required: true, valueType: NFKModelValueTypeAudio, title: $0.capitalized,
                    help: "The \($0) stem.")
        }
    }

    fileprivate static func speechInputs(prompt: String) -> [NFKModelPort] {
        [.input(NFKInputPrompt, title: "Text", help: prompt),
         .input(NFKInputMessages, title: "Conversation", help: "A conversation whose user turns are spoken, joined.")]
    }

    fileprivate static func synthesizedSpeech(_ help: String) -> NFKModelPort {
        .output(NFKOutputAudio, required: true, title: "Audio", help: help)
    }

    /// The sample rate a speech backend tags its WAV file with. The model generates at its own rate.
    fileprivate static var generatedSampleRate: NFKModelOption {
        .option(NFKParameterSampleRate, kind: NFKModelOptionKindInteger, title: "Sample rate",
                help: "The sample rate the WAV file is written at, in hertz. The model generates at 24000 Hz, so another rate changes the pitch and the speed.",
                defaultValue: NFKMLXSpeechConfiguration().sampleRate, minimum: 1, step: 1)
    }

    /// `value` as the shortest decimal that reads back as it, so a manifest writes 0.3 rather than the
    /// float's widened 0.30000001192092896.
    fileprivate static func decimal(_ value: Float) -> Double {
        Double(value.description) ?? Double(value)
    }

    fileprivate static var maximumTokens: NFKModelOption {
        .option(NFKParameterMaxTokens, kind: NFKModelOptionKindInteger, title: "Maximum tokens",
                help: "The most tokens the decoder writes.", defaultValue: 200, minimum: 1, step: 1)
    }

    fileprivate static func onsetThreshold(_ defaultValue: Double) -> NFKModelOption {
        .option(NFKMLXTranscriptionParameterKey.onsetThreshold, kind: NFKModelOptionKindNumber, title: "Onset threshold",
                help: "The onset activation a peak must reach to start a note.",
                defaultValue: defaultValue, minimum: 0, maximum: 1, step: 0.05)
    }

    fileprivate static func tempo(_ defaultValue: Double) -> NFKModelOption {
        .option(NFKMLXTranscriptionParameterKey.tempo, kind: NFKModelOptionKindNumber, title: "Tempo",
                help: "The tempo the MIDI sequence is written at, in beats per minute.",
                defaultValue: defaultValue, minimum: 1, step: 1)
    }

    fileprivate static func program(_ defaultValue: Int) -> NFKModelOption {
        .option(NFKMLXTranscriptionParameterKey.program, kind: NFKModelOptionKindInteger, title: "Program",
                help: "The General MIDI program the notes are written on.",
                defaultValue: defaultValue, minimum: 0, maximum: 127, step: 1)
    }
}
