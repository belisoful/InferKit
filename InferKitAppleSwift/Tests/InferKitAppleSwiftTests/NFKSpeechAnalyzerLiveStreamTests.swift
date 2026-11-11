//
//  NFKSpeechAnalyzerLiveStreamTests.swift
//  InferKitAppleSwiftTests
//
//  The speech analyzer on a live source: one session transcribes a stream's units as they arrive.
//

import AVFoundation
import XCTest
import InferKit
@testable import InferKitAppleSwift

final class NFKSpeechAnalyzerLiveStreamTests: XCTestCase {

    /// A spoken sentence from the system voice, as 16 kHz mono float32 samples.
    private func spokenSamples(_ text: String) throws -> [Float] {
        let synthesis = NFKSpeechSynthesisBackend()
        let result = try synthesis.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: text],
                                                                         parameters: [NFKParameterSourceLanguage: "en-US"]))
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        let file = try AVAudioFile(forReading: try XCTUnwrap(asset.fileURL))
        let source = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: source)
        let target = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1))
        let converter = try XCTUnwrap(AVAudioConverter(from: source.format, to: target))
        let output = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: target,
                                                    frameCapacity: AVAudioFrameCount(Double(source.frameLength) * 16000 / source.format.sampleRate) + 64))
        var supplied = false
        var conversionError: NSError?
        converter.convert(to: output, error: &conversionError) { _, status in
            if supplied {
                status.pointee = AVAudioConverterInputStatus.endOfStream
                return nil
            }
            supplied = true
            status.pointee = AVAudioConverterInputStatus.haveData
            return source
        }
        if let conversionError {
            throw conversionError
        }
        return Array(UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
    }

    private func unit(_ samples: ArraySlice<Float>) -> NFKInferenceRequest {
        let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(max(samples.count, 1)))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        for (index, sample) in samples.enumerated() {
            buffer.floatChannelData![0][index] = sample
        }
        return NFKInferenceRequest(inputs: [NFKInputAudio: buffer])
    }

    /// Opens a stream on en-US, or skips where this machine has no analyzer or no English assets.
    private func openStream() throws -> any NFKLiveStream {
        let installed = expectation(description: "installed locales")
        var locales: [Locale] = []
        NFKSpeechAnalyzerBackend.installedLocales { found in
            locales = found
            installed.fulfill()
        }
        wait(for: [installed], timeout: 30)
        guard locales.contains(where: { $0.identifier.hasPrefix("en") }) else {
            throw XCTSkip("no English speech assets are installed on this machine")
        }
        let backend = NFKSpeechAnalyzerBackend(locale: Locale(identifier: "en-US"))
        backend.assetInstallTimeout = 60
        backend.transcriptionTimeout = 60
        return try backend.openLiveStream(parameters: nil)
    }

    func testAStreamTranscribesItsUnitsAsTheyArrive() throws {
        let stream = try openStream()
        defer { stream.close() }
        let samples = try spokenSamples("The weather today is sunny with a light breeze from the west.")
        let silence = [Float](repeating: 0, count: 1600)
        var finals: [NFKAudioSegment] = []
        var finalText: [String] = []
        var tentative: [String] = []
        for start in stride(from: 0, to: samples.count, by: 1600) {
            let result = try stream.result(for: unit(samples[start ..< min(start + 1600, samples.count)]))
            finals += try XCTUnwrap(result.output(forKey: NFKOutputSegments) as? [NFKAudioSegment])
            finalText.append(try XCTUnwrap(result.output(forKey: NFKOutputText) as? String))
            tentative.append(try XCTUnwrap(result.output(forKey: NFKOutputTentativeText) as? String))
        }
        let deadline = Date().addingTimeInterval(30)
        while !finals.contains(where: { $0.label?.lowercased().contains("weather") == true }), Date() < deadline {
            let result = try stream.result(for: unit(silence[...]))
            finals += try XCTUnwrap(result.output(forKey: NFKOutputSegments) as? [NFKAudioSegment])
            tentative.append(try XCTUnwrap(result.output(forKey: NFKOutputTentativeText) as? String))
            Thread.sleep(forTimeInterval: 0.1)
        }
        let transcript = finals.compactMap(\.label).joined(separator: " ").lowercased()
        XCTAssertTrue(transcript.contains("weather"), "the stream finalizes the spoken sentence: \(transcript)")
        XCTAssertTrue(tentative.contains { !$0.isEmpty }, "the analyzer reports tentative words before they are final")
        let duration = Double(samples.count) / 16000
        for span in finals {
            XCTAssertGreaterThanOrEqual(span.startSeconds, 0)
            XCTAssertLessThanOrEqual(span.endSeconds, duration + 30, "a span is timed from the stream's first sample")
        }
    }

    func testAUnitWithoutAudioIsRefused() throws {
        let stream = try openStream()
        defer { stream.close() }
        XCTAssertThrowsError(try stream.result(for: NFKInferenceRequest(inputs: [:]))) { error in
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_InferenceMissingInput.rawValue)
        }
    }

    func testAStreamRefusesUnitsAfterClose() throws {
        let stream = try openStream()
        stream.close()
        XCTAssertThrowsError(try stream.result(for: unit([0, 0][...]))) { error in
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_InferenceNotReady.rawValue)
        }
    }
}
