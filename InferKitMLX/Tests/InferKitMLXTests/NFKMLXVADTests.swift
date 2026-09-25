//
//  NFKMLXVADTests.swift
//  InferKitMLXTests
//
//  The MarbleNet-style VAD. The forward, segment decode, and weight round-trip evaluate MLX arrays,
//  so they skip without a Metal library for MLX (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXVADTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyNet() -> NFKMLXVADNet {
        NFKMLXVADNet(.tiny)
    }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["blocks.0.convs.0.depthwise.weight", "blocks.0.convs.0.pointwise.weight",
                         "blocks.0.convs.0.norm.weight", "blocks.1.residual.pointwise.weight", "head.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testProbabilitiesAreOnePerFrameInRange() throws {
        try requireMLXRuntime()
        let probabilities = tinyNet().speechProbabilities(Self.floats(samples: 8000), sampleRate: 16000)
        XCTAssertFalse(probabilities.isEmpty, "one probability per mel frame")
        for probability in probabilities {
            XCTAssertGreaterThanOrEqual(probability, 0)
            XCTAssertLessThanOrEqual(probability, 1)
        }
    }

    func testDetectionMergesFramesIntoOrderedSpans() throws {
        try requireMLXRuntime()
        // Force a deterministic decode: a zero-threshold keeps every frame, so the whole clip is one span.
        var configuration = NFKMLXVADConfiguration.tiny
        configuration.threshold = 0
        let net = NFKMLXVADNet(configuration)
        let segments = net.detect(Self.floats(samples: 8000), sampleRate: 16000)
        XCTAssertEqual(segments.count, 1, "a zero threshold merges the clip into a single span")
        let span = try XCTUnwrap(segments.first)
        XCTAssertEqual(span.startSeconds, 0, accuracy: 1e-6)
        XCTAssertGreaterThan(span.endSeconds, span.startSeconds)
    }

    func testDetectionSkipsTheFramesPastTheEncodersValidLength() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXVADConfiguration.tiny
        configuration.threshold = 0
        let net = NFKMLXVADNet(configuration)
        let samples = Self.floats(samples: 8000)
        let (probabilities, validFrames) = net.speechProbabilitiesAndLength(samples, sampleRate: 16000)
        XCTAssertEqual(probabilities.count, 101, "the forward emits the even-count padding's frame")
        XCTAssertEqual(validFrames, 100, "and the encoder's valid length excludes it")
        let span = try XCTUnwrap(net.detect(samples, sampleRate: 16000).first)
        XCTAssertEqual(span.endSeconds, 0.5, accuracy: 1e-9, "a span ends at the last valid frame, the clip's end")
    }

    func testSpansEndAtTheClipAndDropThoseLeftWithNoLength() {
        let spans = NFKMLXSpeechSpans.merge([0.9, 0.9, 0.1, 0.8, 0.8][...], frameSeconds: 0.02, threshold: 0.5,
                                            duration: 0.09)
        XCTAssertEqual(spans.count, 2)
        XCTAssertEqual(spans[0].startSeconds, 0, accuracy: 1e-9)
        XCTAssertEqual(spans[0].endSeconds, 0.04, accuracy: 1e-9)
        XCTAssertEqual(spans[1].startSeconds, 0.06, accuracy: 1e-9)
        XCTAssertEqual(spans[1].endSeconds, 0.09, accuracy: 1e-9, "the last frame's end is clamped to the clip")
        XCTAssertEqual(spans[1].confidence, 0.8, accuracy: 1e-6)

        let beyond = NFKMLXSpeechSpans.merge([0.1, 0.1, 0.9][...], frameSeconds: 0.02, threshold: 0.5, duration: 0.04)
        XCTAssertTrue(beyond.isEmpty, "a run that starts at the clip's end covers none of it")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let onDisk = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value -> (String, MLXArray) in
            value.ndim == 3 ? (key, value.transposed(0, 2, 1)) : (key, value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("vad-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: onDisk, url: url)

        let loaded = tinyNet()
        try NFKMLXVAD.loadWeights(into: loaded, from: url)

        let samples = Self.floats(samples: 4000)
        XCTAssertEqual(trained.speechProbabilities(samples, sampleRate: 16000),
                       loaded.speechProbabilities(samples, sampleRate: 16000),
                       "loaded weights reproduce the trained forward")
    }

    func testTheBackendReturnsSegments() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXVAD.backend(weightsURL: nil)
        let wave = NFKMLXWaveFile.data(samples: Self.floats(samples: 16000), sampleRate: 16000)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        XCTAssertNotNil(result.segments, "segments output present (possibly empty)")
    }

    func testAClipAtAnotherRateIsResampledRatherThanMisread() throws {
        try requireMLXRuntime()
        // The same sound at two rates has to analyze the same. Without resampling the front end reads
        // the 32 kHz clip as if it were 16 kHz, so every frequency lands an octave off and the frame
        // timing doubles — a silent misreading rather than a visible failure.
        let net = NFKMLXVAD.makeNet(.tiny)
        let modelRate = NFKMLXVADConfiguration.tiny.sampleRate
        let atModelRate = Self.tone(seconds: 0.5, rate: modelRate)
        let atDoubleRate = Self.tone(seconds: 0.5, rate: modelRate * 2)

        let expected = net.speechProbabilities(atModelRate, sampleRate: modelRate)
        let resampled = net.speechProbabilities(atDoubleRate, sampleRate: modelRate * 2)
        XCTAssertEqual(resampled.count, expected.count, "the resampled clip yields the same frame count")

        let error = zip(expected, resampled).map { abs($0 - $1) }.reduce(0, +) / Float(max(expected.count, 1))
        XCTAssertLessThan(error, 0.05, "and the same speech probabilities, within resampling error")
    }

    func testSegmentTimesAreInSecondsOfTheCallersClip() throws {
        try requireMLXRuntime()
        let net = NFKMLXVAD.makeNet(.tiny)
        let modelRate = NFKMLXVADConfiguration.tiny.sampleRate
        // A clip of a known duration must report spans inside that duration whichever rate it arrives
        // at: resampling preserves duration, so seconds are the caller's either way.
        for rate in [modelRate, modelRate * 2] {
            let segments = net.detect(Self.tone(seconds: 1.0, rate: rate), sampleRate: rate)
            for segment in segments {
                XCTAssertGreaterThanOrEqual(segment.startSeconds, 0)
                XCTAssertLessThanOrEqual(segment.endSeconds, 1.0, "a one-second clip cannot span longer")
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
