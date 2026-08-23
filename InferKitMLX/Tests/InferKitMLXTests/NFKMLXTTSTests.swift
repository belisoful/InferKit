//
//  NFKMLXTTSTests.swift
//  InferKitMLXTests
//
//  Acoustic model + HiFi-GAN vocoder + the full text→speech chain. These evaluate MLX arrays, so they
//  skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXTTSTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private let symbols = (0 ..< 40).map { "p\($0)" }

    private func vocoderConfiguration() -> NFKMLXHiFiGANConfiguration {
        var configuration = NFKMLXHiFiGANConfiguration()
        configuration.initialChannels = 16
        configuration.upsampleRates = [4, 4]                    // hop 16
        configuration.upsampleKernels = [8, 8]
        configuration.resblockKernels = [3]
        configuration.resblockDilations = [[1]]
        return configuration
    }

    private func acousticConfiguration() -> NFKMLXAcousticConfiguration {
        var configuration = NFKMLXAcousticConfiguration()
        configuration.phonemeVocab = 40
        configuration.state = 32
        configuration.heads = 2
        configuration.encoderLayers = 1
        configuration.decoderLayers = 1
        configuration.maxDuration = 4
        return configuration
    }

    private func g2pConfiguration() -> NFKMLXG2PConfiguration {
        var configuration = NFKMLXG2PConfiguration()
        configuration.phonemeVocab = 40
        configuration.state = 32
        configuration.heads = 2
        configuration.encoderLayers = 1
        configuration.decoderLayers = 1
        configuration.maxLength = 8
        return configuration
    }

    func testVocoderUpsamplesMelToAWaveform() throws {
        try requireMLXRuntime()
        let net = NFKMLXHiFiGAN.makeNet(vocoderConfiguration())
        let mel = MLXArray.zeros([1, 4, 80]) + 0.1
        let waveform = net.waveform(mel)
        eval(waveform)
        XCTAssertEqual(waveform.shape, [1, 64, 1], "4 mel frames × hop 16")
        let values = waveform.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(values.min()), -1, "tanh output")
        XCTAssertLessThanOrEqual(try XCTUnwrap(values.max()), 1)
    }

    func testAcousticModelProducesAMel() throws {
        try requireMLXRuntime()
        let net = NFKMLXAcousticNet(acousticConfiguration())
        let mel = net.mel(for: [3, 1, 20, 5])
        eval(mel)
        XCTAssertEqual(mel.shape[0], 1)
        XCTAssertEqual(mel.shape[2], 80, "mel bins")
        XCTAssertGreaterThanOrEqual(mel.shape[1], 4, "at least one frame per phoneme")
    }

    func testTheFullChainSynthesizesAWaveformFromText() throws {
        try requireMLXRuntime()
        let g2p = NFKMLXNeuralG2P(configuration: g2pConfiguration(), phonemeSymbols: symbols)
        let tts = NFKMLXTTS(phonemizer: g2p, acoustic: acousticConfiguration(), vocoder: vocoderConfiguration(), symbols: symbols)
        let waveform = tts.synthesize("hello")
        eval(waveform)
        XCTAssertGreaterThan(waveform.shape[0], 0, "text → phonemes → mel → audio")
    }

    func testTheSpeechBackendWritesAWaveFileFromText() throws {
        try requireMLXRuntime()
        let g2p = NFKMLXNeuralG2P(configuration: g2pConfiguration(), phonemeSymbols: symbols)
        let tts = NFKMLXTTS(phonemizer: g2p, acoustic: acousticConfiguration(), vocoder: vocoderConfiguration(), symbols: symbols)
        let backend = tts.makeSpeechBackend(sampleRate: 22050)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hi there"]))
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        let url = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertNotNil(NFKMLXWaveFile.read(try Data(contentsOf: url)), "a playable WAV from text")
    }

    func testVocoderCheckpointRoundTripReproducesTheWaveform() throws {
        try requireMLXRuntime()
        let trained = NFKMLXHiFiGAN.makeNet(vocoderConfiguration())
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value -> (String, MLXArray) in
            if value.ndim == 4 { return (key, value.transposed(0, 3, 1, 2)) }
            if value.ndim == 3 { return (key, value.transposed(0, 2, 1)) }
            return (key, value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("hifigan-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXHiFiGAN.makeNet(vocoderConfiguration())
        try NFKMLXHiFiGAN.loadWeights(into: loaded, from: url)

        let mel = MLXArray.zeros([1, 4, 80]) + 0.2
        let expected = trained.waveform(mel)
        let actual = loaded.waveform(mel)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }
}
