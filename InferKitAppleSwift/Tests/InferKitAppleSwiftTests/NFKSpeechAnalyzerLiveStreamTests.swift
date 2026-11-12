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

    /// The analyzer's throughput through the stream, written as a JSON report to the path
    /// IK_TIME_LIVE_SPEECH names. The analyzer answers asynchronously, so a unit's call returns at once.
    /// Each run therefore feeds a spoken sentence and 1.5 s of silence in 0.1 s chunks as fast as the stream
    /// takes them, then drains with empty units until the transcript, final or tentative, reaches the
    /// sentence's last word: the analyzer has then heard all of it. The time per chunk is that wall time over the chunks fed, and the real-time factor is that wall
    /// time over the audio fed. The median of three runs after a warm-up run. Runs only in a release
    /// build, under the Large Model Coordination lock, on a quiet machine.
    func testTheStreamIsTimedPerChunk() throws {
        guard let reportPath = ProcessInfo.processInfo.environment["IK_TIME_LIVE_SPEECH"], !reportPath.isEmpty else {
            throw XCTSkip("set IK_TIME_LIVE_SPEECH to the report's path to time the speech analyzer's stream")
        }
        #if DEBUG
        let debugBuild = true
        #else
        let debugBuild = false
        #endif
        try XCTSkipIf(debugBuild, "time the stream in a release build (swift test -c release -Xswiftc -enable-testing)")
        let sentence = try spokenSamples("The weather today is sunny with a light breeze from the west.")
        let samples = sentence + [Float](repeating: 0, count: 24000)
        let chunk = 1600
        let chunks = (samples.count + chunk - 1) / chunk
        var runs: [Double] = []
        for run in 0 ..< 4 {
            let stream = try openStream()
            defer { stream.close() }
            var heard = ""
            func hear(_ result: NFKInferenceResult) throws {
                let finals = try XCTUnwrap(result.output(forKey: NFKOutputSegments) as? [NFKAudioSegment])
                heard += finals.compactMap(\.label).joined(separator: " ").lowercased()
                heard += " " + (try XCTUnwrap(result.output(forKey: NFKOutputTentativeText) as? String)).lowercased()
            }
            let started = DispatchTime.now().uptimeNanoseconds
            for start in stride(from: 0, to: samples.count, by: chunk) {
                try hear(try stream.result(for: unit(samples[start ..< min(start + chunk, samples.count)])))
            }
            let deadline = Date().addingTimeInterval(60)
            while !heard.contains("west"), Date() < deadline {
                try hear(try stream.result(for: unit(samples[0 ..< 0])))
                Thread.sleep(forTimeInterval: 0.002)
            }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
            XCTAssertTrue(heard.contains("west"), "run \(run): the stream hears the sentence's last word: \(heard)")
            if run > 0 {
                runs.append(elapsed)
            }
        }
        runs.sort()
        let elapsed = runs[runs.count / 2]
        let milliseconds = elapsed / Double(chunks)
        let realTimeFactor = elapsed / (Double(samples.count) / 16)
        let row: [String: Any] = ["model": "apple-speech-analyzer", "chip": NFKHardwareProfile.current.chipName,
                                  "milliseconds": milliseconds, "realTimeFactor": realTimeFactor,
                                  "measured": ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate]),
                                  "note": "Fed faster than real time; the wall time until the transcript reaches the last word, over the chunks fed."]
        print("LIVE-TIMING apple-speech-analyzer: \(milliseconds) ms per 0.1 s chunk, real-time factor \(realTimeFactor) "
              + "(runs \(runs.map { $0 / 1000 }) s over \(Double(samples.count) / 16000) s of audio)")
        try JSONSerialization.data(withJSONObject: [row], options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: reportPath))
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
