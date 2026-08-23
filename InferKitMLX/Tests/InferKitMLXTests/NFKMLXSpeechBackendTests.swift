//
//  NFKMLXSpeechBackendTests.swift
//  InferKitMLXTests
//
//  The WAV encoder and the backend contract need no GPU. The end-to-end synth evaluates an MLXArray,
//  so it skips under `swift test` and runs under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSpeechBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    // MARK: WAV encoder (no MLX)

    func testWaveDataHasARIFFHeaderAndClampedSamples() {
        let data = NFKMLXWaveFile.data(samples: [0, 0.5, -2.0, 1.0], sampleRate: 24000)
        let bytes = [UInt8](data)
        XCTAssertEqual(String(bytes: bytes[0..<4], encoding: .ascii), "RIFF")
        XCTAssertEqual(String(bytes: bytes[8..<12], encoding: .ascii), "WAVE")
        XCTAssertEqual(String(bytes: bytes[36..<40], encoding: .ascii), "data")
        XCTAssertEqual(data.count, 44 + 4 * 2, "44-byte header + 4 Int16 samples")

        // The out-of-range -2.0 clamps to -1 (Int16 -32767) and 1.0 maps to Int16.max.
        let sampleBytes = Array(bytes[44...])
        let third = Int16(bitPattern: UInt16(sampleBytes[4]) | (UInt16(sampleBytes[5]) << 8))
        let fourth = Int16(bitPattern: UInt16(sampleBytes[6]) | (UInt16(sampleBytes[7]) << 8))
        XCTAssertEqual(third, -Int16.max)
        XCTAssertEqual(fourth, Int16.max)
    }

    func testWaveHeaderCarriesTheFormat() {
        let data = NFKMLXWaveFile.data(samples: [0, 0, 0], sampleRate: 16000, channels: 1)
        let bytes = [UInt8](data)
        let sampleRate = UInt32(bytes[24]) | (UInt32(bytes[25]) << 8) | (UInt32(bytes[26]) << 16) | (UInt32(bytes[27]) << 24)
        XCTAssertEqual(sampleRate, 16000)
        XCTAssertEqual(bytes[22], 1, "one channel")
        XCTAssertEqual(bytes[34], 16, "16 bits per sample")
    }

    // MARK: Contract (no MLX)

    func testTheBackendReportsIdentityAndReadiness() {
        let backend = NFKMLXSpeechBackend(identifier: "tts", isReady: false) { _, _ in MLXArray([Float]()) }
        XCTAssertEqual(backend.backendIdentifier, "tts")
        XCTAssertFalse(backend.isReady)
    }

    func testAnInferenceWithoutTextFails() {
        let backend = NFKMLXSpeechBackend { _, _ in MLXArray([Float(0)]) }
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: "no text"])))
    }

    func testPromptIsReadFromMessagesWhenNoPromptIsSet() {
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "user", "content": "hello"],
                                                                      ["role": "system", "content": "ignored"]]])
        XCTAssertEqual(NFKMLXSpeechBackend.text(from: request), "hello")
    }

    // MARK: End to end (needs MLX)

    func testToneSpeechWritesAPlayableWaveFile() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerToneSpeech()
        let backend = try NFKMLXModelRegistry.backend(named: "tone-speech", weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"]))

        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        XCTAssertEqual(asset.sampleRate, 24000)
        XCTAssertEqual(asset.channelCount, 1)
        XCTAssertGreaterThan(asset.durationSeconds, 0)

        let url = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: url) }
        let bytes = [UInt8](try Data(contentsOf: url))
        XCTAssertEqual(String(bytes: bytes[0..<4], encoding: .ascii), "RIFF", "a real WAV file on disk")
        XCTAssertGreaterThan(bytes.count, 44, "carries samples for 'hi' (2 characters)")
    }

    func testASampleRateParameterOverridesTheConfiguration() throws {
        try requireMLXRuntime()
        nonisolated(unsafe) var capturedRate = 0                 // set synchronously inside runInference, read after
        let backend = NFKMLXSpeechBackend(configuration: NFKMLXSpeechConfiguration(sampleRate: 24000)) { _, sampleRate in
            capturedRate = sampleRate                            // the closure generates at the requested rate
            return MLXArray([Float](repeating: 0, count: sampleRate / 100))
        }
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"],
                                          parameters: [NFKParameterSampleRate: 16000])
        let result = try backend.runInference(for: request)
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        XCTAssertEqual(capturedRate, 16000, "the request rate reaches the closure")
        XCTAssertEqual(asset.sampleRate, 16000, "and is written to the asset")

        let bytes = [UInt8](try Data(contentsOf: try XCTUnwrap(asset.fileURL)))
        defer { try? FileManager.default.removeItem(at: asset.fileURL!) }
        let headerRate = UInt32(bytes[24]) | (UInt32(bytes[25]) << 8) | (UInt32(bytes[26]) << 16) | (UInt32(bytes[27]) << 24)
        XCTAssertEqual(headerRate, 16000, "and to the WAV header")
    }
}
