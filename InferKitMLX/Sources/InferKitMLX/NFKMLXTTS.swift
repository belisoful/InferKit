//
//  NFKMLXTTS.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// A complete text-to-speech voice: text → phonemes (a `NFKMLXPhonemizer`) → mel (an acoustic model) →
// waveform (the HiFi-GAN vocoder). The acoustic model is FastSpeech2-style: a transformer encoder over
// phoneme embeddings, a duration predictor, a length regulator that expands each phoneme by its
// duration, a transformer decoder, and a mel projection. `NFKMLXTTS` chains the three and exposes an
// `NFKMLXSpeechBackend` (text → an audio file).

/// Acoustic-model sizing.
public struct NFKMLXAcousticConfiguration: Sendable {
    public var phonemeVocab: Int = 80
    public var state: Int = 192
    public var heads: Int = 2
    public var encoderLayers: Int = 2
    public var decoderLayers: Int = 2
    public var melBins: Int = 80
    public var maxDuration: Int = 8
    public init() {}
}

/// The FastSpeech2-style acoustic model: phoneme ids → a mel-spectrogram.
final class NFKMLXAcousticNet: Module {
    @ModuleInfo(key: "phoneme_embedding") var phonemeEmbedding: Embedding
    @ModuleInfo(key: "encoder") var encoder: [NFKWhisperBlock]
    @ModuleInfo(key: "duration_conv1") var durationConv1: Conv1d
    @ModuleInfo(key: "duration_conv2") var durationConv2: Conv1d
    @ModuleInfo(key: "duration_linear") var durationLinear: Linear
    @ModuleInfo(key: "decoder") var decoder: [NFKWhisperBlock]
    @ModuleInfo(key: "mel_linear") var melLinear: Linear

    let configuration: NFKMLXAcousticConfiguration

    init(_ configuration: NFKMLXAcousticConfiguration) {
        self.configuration = configuration
        _phonemeEmbedding.wrappedValue = Embedding(embeddingCount: configuration.phonemeVocab, dimensions: configuration.state)
        _encoder.wrappedValue = (0 ..< configuration.encoderLayers).map { _ in NFKWhisperBlock(state: configuration.state, heads: configuration.heads, cross: false) }
        _durationConv1.wrappedValue = Conv1d(inputChannels: configuration.state, outputChannels: configuration.state, kernelSize: 3, padding: 1)
        _durationConv2.wrappedValue = Conv1d(inputChannels: configuration.state, outputChannels: configuration.state, kernelSize: 3, padding: 1)
        _durationLinear.wrappedValue = Linear(configuration.state, 1)
        _decoder.wrappedValue = (0 ..< configuration.decoderLayers).map { _ in NFKWhisperBlock(state: configuration.state, heads: configuration.heads, cross: false) }
        _melLinear.wrappedValue = Linear(configuration.state, configuration.melBins)
    }

    /// Phoneme ids → mel `[1, T, melBins]`.
    func mel(for phonemeIDs: [Int]) -> MLXArray {
        let ids = phonemeIDs.isEmpty ? [0] : phonemeIDs
        let state = configuration.state
        var x = phonemeEmbedding(MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count]))
            + NFKMLXWhisperNet.sinusoids(length: ids.count, channels: state)
        for block in encoder { x = block(x, audio: nil, mask: nil) }

        let logDuration = durationLinear(relu(durationConv2(relu(durationConv1(x)))))   // [1, T, 1]
        let durations = clip(round(exp(logDuration)), min: 1, max: Float(configuration.maxDuration))
        eval(durations)
        let perPhoneme = durations.reshaped([ids.count]).asType(.int32).asArray(Int32.self)

        var index = [Int32]()
        for (phoneme, count) in perPhoneme.enumerated() {
            for _ in 0 ..< Int(count) { index.append(Int32(phoneme)) }
        }
        if index.isEmpty { index = [0] }
        var expanded = x.reshaped([ids.count, state]).take(MLXArray(index), axis: 0).reshaped([1, index.count, state])
        expanded = expanded + NFKMLXWhisperNet.sinusoids(length: index.count, channels: state)
        for block in decoder { expanded = block(expanded, audio: nil, mask: nil) }
        return melLinear(expanded)
    }
}

/// A full text-to-speech voice: phonemizer + acoustic model + HiFi-GAN vocoder.
public final class NFKMLXTTS: @unchecked Sendable {

    private let phonemizer: NFKMLXPhonemizer
    private let acoustic: NFKMLXAcousticNet
    private let vocoder: NFKMLXHiFiGANNet
    private let symbolToID: [String: Int]

    /// - Parameters:
    ///   - phonemizer: text → phoneme symbols (`NFKMLXNeuralG2P` or `NFKMLXEspeakPhonemizer`).
    ///   - acoustic: the acoustic model configuration (phoneme ids → mel-spectrogram).
    ///   - vocoder: the HiFi-GAN vocoder configuration (mel-spectrogram → waveform).
    ///   - symbols: the phoneme symbol table (symbol → id, by position).
    public init(phonemizer: NFKMLXPhonemizer,
                acoustic: NFKMLXAcousticConfiguration = NFKMLXAcousticConfiguration(),
                vocoder: NFKMLXHiFiGANConfiguration = NFKMLXHiFiGANConfiguration(),
                symbols: [String] = []) {
        self.phonemizer = phonemizer
        self.acoustic = NFKMLXAcousticNet(acoustic)
        self.vocoder = NFKMLXHiFiGANNet(vocoder)
        self.symbolToID = Dictionary(uniqueKeysWithValues: symbols.enumerated().map { ($1, $0) })
    }

    /// Loads the acoustic model and vocoder from safetensors checkpoints.
    public func loadWeights(acousticURL: URL?, vocoderURL: URL?) throws {
        if let acousticURL {
            let checkpoint = try NFKMLXWeights.loadCheckpoint(url: acousticURL)
            let raw = checkpoint.arrays
            let mapped = raw.map { key, value -> (String, MLXArray) in
                checkpoint.needsConvTranspose && value.ndim == 3 ? (key, value.transposed(0, 2, 1)) : (key, value)
            }
            try NFKMLXWeights.apply(mapped, to: acoustic)
        }
        if let vocoderURL {
            try NFKMLXHiFiGAN.loadWeights(into: vocoder, from: vocoderURL)
        }
    }

    /// text → a mono waveform `[N]` at the vocoder's implicit rate.
    public func synthesize(_ text: String) -> MLXArray {
        let ids = phonemizer.phonemes(for: text).map { symbolToID[$0] ?? 0 }
        let mel = acoustic.mel(for: ids)
        let waveform = vocoder.waveform(mel)                    // [1, N, 1]
        return waveform.reshaped([waveform.shape[1]])
    }

    /// A speech backend that renders text to a WAV file at `sampleRate`.
    public func makeSpeechBackend(sampleRate: Int = 22050) -> NFKMLXSpeechBackend {
        NFKMLXSpeechBackend(identifier: "tts", configuration: NFKMLXSpeechConfiguration(sampleRate: sampleRate)) { text, _ in
            self.synthesize(text)
        }
    }

    /// The acoustic model, for tests and weight round-trips.
    var acousticModel: NFKMLXAcousticNet { acoustic }
    /// The vocoder, for tests and weight round-trips.
    var vocoderModel: NFKMLXHiFiGANNet { vocoder }
}
