//
//  NFKMLXSileroVADTests.swift
//  InferKitMLXTests
//
//  Silero VAD v6. The forward, segment decode, chunk assembly, and weight round-trip evaluate MLX arrays,
//  so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSileroVADTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func net() -> NFKMLXSileroVADNet {
        NFKMLXSileroVAD.makeNet(.v6)
    }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(net().parameters().flattened().map(\.0))
        for expected in ["encoder.stft.weight", "encoder.conv1.weight", "encoder.conv4.bias",
                         "decoder.rnn.Wx", "decoder.rnn.Wh", "decoder.rnn.bias",
                         "decoder.final.weight", "decoder.final.bias"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
        // The STFT basis is a stored convolution, not the learned parameters plus a computed constant.
        XCTAssertFalse(names.contains("encoder.stft.bias"), "the STFT convolution carries no bias")
    }

    func testProbabilitiesAreOnePerChunkInRange() throws {
        try requireMLXRuntime()
        // 16000 samples pad up to 32 chunks of 512.
        let probabilities = net().speechProbabilities(Self.floats(samples: 16000), sampleRate: 16000)
        XCTAssertEqual(probabilities.count, 32, "one probability per 512-sample chunk, padded up")
        for probability in probabilities {
            XCTAssertGreaterThanOrEqual(probability, 0)
            XCTAssertLessThanOrEqual(probability, 1)
        }
    }

    func testChunkAssemblyCarriesContextAndReflectsTheRightPad() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXSileroVADConfiguration.v6
        let num = configuration.numSamples, ctx = configuration.contextSamples, pad = configuration.stftPadRight
        let rowIn = ctx + num, rowOut = rowIn + pad
        // A ramp, so every sample equals its own index and the context/reflection layout is checkable.
        let signal = (0 ..< 2 * num).map { Float($0) }
        let flat = NFKMLXSileroVADNet.chunkedInput(signal, configuration).reshaped([-1]).asArray(Float.self)
        XCTAssertEqual(flat.count, 2 * rowOut)

        func value(chunk: Int, position: Int) -> Float { flat[chunk * rowOut + position] }

        // The first chunk's look-back context is zeros.
        for position in 0 ..< ctx {
            XCTAssertEqual(value(chunk: 0, position: position), 0, "the first chunk has no predecessor")
        }
        // The second chunk's context is the first chunk's final `ctx` samples.
        for i in 0 ..< ctx {
            XCTAssertEqual(value(chunk: 1, position: i), Float((num - ctx) + i),
                           "chunk 1's context is chunk 0's tail")
        }
        // The right pad reflects inward without repeating the edge sample.
        for i in 0 ..< pad {
            XCTAssertEqual(value(chunk: 0, position: rowIn + i), value(chunk: 0, position: rowIn - 2 - i),
                           "the right pad is a reflection")
        }
    }

    func testDetectionMergesChunksIntoOrderedSpans() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXSileroVADConfiguration.v6
        configuration.threshold = 0                          // keep every chunk, so the clip is one span
        let net = NFKMLXSileroVADNet(configuration)
        let segments = net.detect(Self.floats(samples: 8192), sampleRate: 16000)
        XCTAssertEqual(segments.count, 1, "a zero threshold merges the clip into a single span")
        let span = try XCTUnwrap(segments.first)
        XCTAssertEqual(span.startSeconds, 0, accuracy: 1e-6)
        XCTAssertGreaterThan(span.endSeconds, span.startSeconds)
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = net()
        // Save in PyTorch Conv1d layout (`[out, in, k]`), which the loader transposes back on load.
        let onDisk = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value -> (String, MLXArray) in
            value.ndim == 3 ? (key, value.transposed(0, 2, 1)) : (key, value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("silero-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: onDisk, url: url)

        let loaded = net()
        try NFKMLXSileroVAD.loadWeights(into: loaded, from: url)

        let samples = Self.floats(samples: 4096)
        XCTAssertEqual(trained.speechProbabilities(samples, sampleRate: 16000),
                       loaded.speechProbabilities(samples, sampleRate: 16000),
                       "loaded weights reproduce the trained forward")
    }

    func testTheBackendReturnsSegments() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXSileroVAD.backend(weightsURL: nil)
        let wave = NFKMLXWaveFile.data(samples: Self.floats(samples: 16000), sampleRate: 16000)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        XCTAssertNotNil(result.segments, "segments output present (possibly empty)")
    }

    func testAClipAtAnotherRateIsResampledRatherThanMisread() throws {
        try requireMLXRuntime()
        let net = net()
        let atModelRate = Self.tone(seconds: 0.5, rate: 16000)
        let atDoubleRate = Self.tone(seconds: 0.5, rate: 32000)

        let expected = net.speechProbabilities(atModelRate, sampleRate: 16000)
        let resampled = net.speechProbabilities(atDoubleRate, sampleRate: 32000)
        XCTAssertEqual(resampled.count, expected.count, "the resampled clip yields the same chunk count")

        let error = zip(expected, resampled).map { abs($0 - $1) }.reduce(0, +) / Float(max(expected.count, 1))
        XCTAssertLessThan(error, 0.05, "and the same speech probabilities, within resampling error")
    }

    func testSegmentTimesAreInSecondsOfTheCallersClip() throws {
        try requireMLXRuntime()
        let net = net()
        for rate in [16000, 32000] {
            let segments = net.detect(Self.tone(seconds: 1.0, rate: rate), sampleRate: rate)
            for segment in segments {
                XCTAssertGreaterThanOrEqual(segment.startSeconds, 0)
                XCTAssertLessThanOrEqual(segment.endSeconds, 1.05, "a one-second clip cannot span longer")
            }
        }
    }

    /// A voiced-like harmonic stack at a given rate, so the same sound exists at two sample rates.
    static func tone(seconds: Double, rate: Int) -> [Float] {
        let count = Int(seconds * Double(rate))
        return (0 ..< count).map { index in
            let t = Double(index) / Double(rate)
            var value = 0.0
            for harmonic in 1 ... 5 {
                value += 0.3 / Double(harmonic) * sin(2 * .pi * 150 * Double(harmonic) * t)
            }
            return Float(value)
        }
    }

    static func floats(samples: Int) -> [Float] {
        (0 ..< samples).map { sinf(Float($0) * 0.06) * 0.4 }
    }
}
