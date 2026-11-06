//
//  NFKMLXSileroVADLiveStreamTests.swift
//  InferKitMLXTests
//
//  Silero VAD on a live source: a stream scores each chunk from the state the chunks before left, as a
//  clip of the same samples scores it.
//

import AVFoundation
import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSileroVADLiveStreamTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// Splits `samples` at uneven unit lengths, some shorter than a chunk.
    private func units(_ samples: [Float]) -> [[Float]] {
        var units: [[Float]] = []
        var start = 0
        for length in [300, 700, 512, 1000, 60, 2000].cycled(toCover: samples.count) {
            let end = min(start + length, samples.count)
            units.append(Array(samples[start ..< end]))
            start = end
            if start == samples.count {
                break
            }
        }
        return units
    }

    func testAStreamScoresEachChunkAsTheClipDoes() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(7)
        let net = NFKMLXSileroVAD.makeNet(.v6)
        let samples = NFKMLXSileroVADTests.floats(samples: 512 * 24)
        let clip = net.speechProbabilities(samples, sampleRate: 16000)
        let state = NFKSileroStreamState()
        let streamed = units(samples).flatMap { net.streamProbabilities($0, state: state) }
        XCTAssertEqual(streamed.count, clip.count)
        XCTAssertTrue(streamed == clip, "the carried look-back and LSTM state score each chunk exactly as the clip does")
        XCTAssertTrue(state.pending.isEmpty)
    }

    private func buffer(_ samples: [Float], sampleRate: Double = 16000, channels: AVAudioChannelCount = 1) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: channels)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        for channel in 0 ..< Int(channels) {
            for (index, sample) in samples.enumerated() {
                buffer.floatChannelData![channel][index] = sample
            }
        }
        return buffer
    }

    func testTheBackendsStreamReturnsTheClipsSpansChunkByChunk() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(7)
        let net = NFKMLXSileroVAD.makeNet(.v6)
        let samples = (0 ..< 512 * 24).map { index -> Float in
            let envelope: Float = (index / 2048) % 2 == 0 ? 0.6 : 0.02
            return sinf(Float(index) * 0.07) * envelope
        }
        let probabilities = net.speechProbabilities(samples, sampleRate: 16000)
        let threshold = probabilities.sorted()[probabilities.count / 2]
        let spans = NFKMLXSpeechSpans.merge(probabilities[...], frameSeconds: net.configuration.chunkSeconds,
                                            threshold: threshold, duration: Double(samples.count) / 16000)
        XCTAssertFalse(spans.isEmpty)

        let backend = NFKMLXSileroVADBackend(net: net, identifier: "silero-live-test")
        let stream = try backend.openLiveStream(parameters: [NFKMLXVoiceActivityParameterKey.speechThreshold: threshold])
        var segments: [NFKAudioSegment] = []
        for unit in units(samples) {
            let result = try stream.result(for: NFKInferenceRequest(inputs: [NFKInputAudio: buffer(unit)]))
            segments += try XCTUnwrap(result.segments)
        }
        var merged: [(start: Double, end: Double)] = []
        for segment in segments {
            if let last = merged.last, abs(last.end - segment.startSeconds) < 1e-9 {
                merged[merged.count - 1].end = segment.endSeconds
            } else {
                merged.append((segment.startSeconds, segment.endSeconds))
            }
        }
        XCTAssertEqual(merged.map(\.start), spans.map(\.startSeconds))
        XCTAssertEqual(merged.map(\.end), spans.map(\.endSeconds))

        XCTAssertThrowsError(try stream.result(for: NFKInferenceRequest(inputs: [NFKInputAudio: buffer([0, 0], sampleRate: 48000)]))) {
            XCTAssertEqual(($0 as NSError).code, NFKInferenceError.error_InferenceUnsupported.rawValue)
        }
        let opening = NFKInferenceRequest(inputs: [NFKInputAudio: buffer(Array(samples[..<(512 * 16)]))])
        stream.reset()
        let first = try XCTUnwrap(try stream.result(for: opening).segments)
        stream.reset()
        let again = try XCTUnwrap(try stream.result(for: opening).segments)
        XCTAssertFalse(first.isEmpty)
        XCTAssertEqual(again.map(\.startSeconds), first.map(\.startSeconds), "reset times the next chunk from the stream's start")
        XCTAssertEqual(again.map(\.confidence), first.map(\.confidence), "reset clears the carried state")
        XCTAssertEqual(first.map(\.startSeconds), segments.filter { $0.startSeconds < 512 * 16 / 16000.0 }.map(\.startSeconds))
    }
}

private extension Array {
    /// The array repeated until its elements reach `count` items.
    func cycled(toCover count: Int) -> [Element] {
        var out: [Element] = []
        while out.count < count {
            out += self
        }
        return out
    }
}
