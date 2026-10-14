//
//  NFKMLXVoiceBackendTests.swift
//  InferKitMLXTests
//
//  The FastSpeech2 + HiFi-GAN voice and the hand-chained voice behind the InferKit contract, built the
//  way Objective-C builds them: a prompt and an NFKMLXPhonemizing object become a WAV under
//  NFKOutputAudio, and a phonemizer that fails fails the request.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

/// A phonemizer an application would write: fixed symbols, or a failure.
private final class NFKFixedPhonemizer: NSObject, NFKMLXPhonemizing {
    let fails: Bool
    init(fails: Bool = false) { self.fails = fails }
    func phonemes(forText text: String) throws -> [String] {
        if fails { throw NFKMLXError.unsupportedInput }
        return ["HH", "AH0", "L", "OW1"]
    }
}

final class NFKMLXVoiceBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func smallVoice() -> NFKMLXVoice {
        var acoustic = NFKMLXFastSpeech2Configuration()
        acoustic.hiddenSize = 32
        acoustic.encoderLayers = 1
        acoustic.decoderLayers = 1
        acoustic.linearUnits = 64
        acoustic.durationChannels = 16
        acoustic.pitchChannels = 16
        acoustic.pitchLayers = 2
        acoustic.energyChannels = 16
        acoustic.postnetChannels = 16
        acoustic.postnetLayers = 2
        var vocoder = NFKMLXHiFiGANConfiguration()
        vocoder.initialChannels = 16
        vocoder.upsampleRates = [4, 4]
        vocoder.upsampleKernels = [8, 8]
        vocoder.resblockKernels = [3]
        vocoder.resblockDilations = [[1]]
        MLXRandom.seed(41)
        return NFKMLXVoice(acoustic: NFKMLXFastSpeech2.makeNet(acoustic), vocoder: NFKMLXHiFiGAN.makeNet(vocoder),
                           vocabulary: ["<unk>": 0, "HH": 1, "AH0": 2, "L": 3, "OW1": 4])
    }

    private func audio(_ result: NFKInferenceResult) throws -> NFKAudioAsset {
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        let url = try XCTUnwrap(asset.fileURL)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return asset
    }

    func testAPromptBecomesAWaveThroughTheCallersPhonemizer() throws {
        try requireMLXRuntime()
        let backend = NFKMLXFastSpeech2.voiceBackend(smallVoice(), phonemizer: NFKFixedPhonemizer())
        XCTAssertEqual(backend.backendIdentifier, NFKMLXFastSpeech2.voiceModelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputMessages])
        let asset = try audio(backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"],
                                                                            parameters: [:])))
        XCTAssertEqual(asset.sampleRate, Double(NFKMLXVoice.sampleRate))
        XCTAssertGreaterThan(asset.durationSeconds, 0)
        XCTAssertEqual(NFKMLXWaveFile.read(try Data(contentsOf: try XCTUnwrap(asset.fileURL)))?.sampleRate,
                       NFKMLXVoice.sampleRate)
    }

    func testAFailingPhonemizerOrAMissingPromptFailsTheRequest() throws {
        try requireMLXRuntime()
        let failing = NFKMLXFastSpeech2.voiceBackend(smallVoice(), phonemizer: NFKFixedPhonemizer(fails: true))
        XCTAssertThrowsError(try failing.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"],
                                                                               parameters: [:])))
        let backend = NFKMLXFastSpeech2.voiceBackend(smallVoice(), phonemizer: NFKFixedPhonemizer())
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
    }

    func testAG2PCheckpointLoadsThroughTheObjectiveCFactoryAndNamesTheVoicesSymbols() throws {
        try requireMLXRuntime()
        MLXRandom.seed(43)
        let trained = NFKMLXNeuralG2P()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("g2p-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(trained.model, to: url)

        let voice = smallVoice()
        let g2p = try NFKMLXFastSpeech2.g2p(weightsURL: url, voice: voice)
        let names = ["<unk>", "HH", "AH0", "L", "OW1"]
        let expected = trained.phonemes(for: "hello").map { id -> String in
            guard let index = Int(id), index < names.count else { return id }
            return names[index]
        }
        XCTAssertEqual(try g2p.phonemes(forText: "hello"), expected,
                       "the reloaded model decodes the saved one's ids into the voice's symbols")
    }

    func testTheHandChainedVoiceSpeaksThroughAnObjectiveCPhonemizer() throws {
        try requireMLXRuntime()
        MLXRandom.seed(47)
        let backend = try NFKMLXFastSpeech2.ttsBackend(phonemizer: NFKFixedPhonemizer(), acousticURL: nil, vocoderURL: nil,
                                                       symbols: ["<unk>", "HH", "AH0", "L", "OW1"])
        XCTAssertEqual(backend.backendIdentifier, "tts")
        let asset = try audio(backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"],
                                                                            parameters: [:])))
        XCTAssertEqual(asset.sampleRate, 22_050)
        XCTAssertGreaterThan(asset.durationSeconds, 0)

        let failing = try NFKMLXFastSpeech2.ttsBackend(phonemizer: NFKFixedPhonemizer(fails: true), acousticURL: nil,
                                                       vocoderURL: nil, symbols: ["<unk>"])
        XCTAssertThrowsError(try failing.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"],
                                                                               parameters: [:])))
    }

    func testTheRegistryRefusesAMissingVoiceDirectory() throws {
        NFKMLXFastSpeech2.register()
        XCTAssertThrowsError(try NFKMLXModelRegistry.backend(named: NFKMLXFastSpeech2.voiceModelName, weightsURL: nil))
    }

    // The released espnet voice, both networks read out of its one combined checkpoint.
    func testTheReleasedCombinedCheckpointSpeaks() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let checkpoint = environment["IK_RAW_HIFIGAN_LJSPEECH_ESPNET"],
              let vocabulary = environment["IK_VOCAB_FASTSPEECH2"],
              FileManager.default.fileExists(atPath: checkpoint) else {
            throw XCTSkip("set IK_RAW_HIFIGAN_LJSPEECH_ESPNET (fastspeech2_conformer_with_hifigan) and IK_VOCAB_FASTSPEECH2")
        }
        let url = URL(fileURLWithPath: checkpoint)
        let backend = try NFKMLXFastSpeech2.voiceBackend(acousticURL: url, vocoderURL: url,
                                                         vocabularyURL: URL(fileURLWithPath: vocabulary),
                                                         phonemizer: NFKFixedPhonemizer())
        let asset = try audio(backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"],
                                                                            parameters: [:])))
        XCTAssertGreaterThan(asset.durationSeconds, 0.1)
    }
}
