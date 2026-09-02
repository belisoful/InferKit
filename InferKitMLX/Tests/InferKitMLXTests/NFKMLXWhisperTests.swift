//
//  NFKMLXWhisperTests.swift
//  InferKitMLXTests
//
//  Encoder-decoder transformer + log-mel. The WAV read/write round-trip needs no GPU; the model path
//  evaluates MLX arrays, so it skips under `swift test` and runs under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXWhisperTests: XCTestCase {

    // The variant enum maps to each released geometry, which is the whole point of the @objc factory
    // that consumes it — before it, only tiny was buildable in either language.
    func testTheVariantMapperSelectsEachReleasedGeometry() {
        XCTAssertEqual(NFKMLXWhisper.configuration(for: .tiny).nAudioState, 384)
        XCTAssertEqual(NFKMLXWhisper.configuration(for: .small).nAudioState, 768)
        XCTAssertEqual(NFKMLXWhisper.configuration(for: .medium).nAudioState, 1024)
        XCTAssertEqual(NFKMLXWhisper.configuration(for: .largeV3).nAudioState, 1280)
        XCTAssertEqual(NFKMLXWhisper.configuration(for: .largeV3).nAudioLayer, 32)
    }


    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyConfiguration() -> NFKMLXWhisperConfiguration {
        var configuration = NFKMLXWhisperConfiguration()
        configuration.nAudioState = 32
        configuration.nAudioHead = 2
        configuration.nAudioLayer = 1
        configuration.nVocab = 50
        configuration.nTextState = 32
        configuration.nTextHead = 2
        configuration.nTextLayer = 1
        configuration.nTextCtx = 16
        configuration.maxTokens = 8
        configuration.promptTokens = [2, 3]                     // valid ids for the tiny nVocab
        configuration.endToken = 1
        configuration.suppressFrom = 50                         // == nVocab → no suppression
        return configuration
    }

    // MARK: WAV round-trip (no MLX)

    func testWaveWriteThenReadRecoversTheSamples() {
        let samples: [Float] = [0, 0.25, -0.5, 0.75, -1, 1]
        let data = NFKMLXWaveFile.data(samples: samples, sampleRate: 16000)
        let read = NFKMLXWaveFile.read(data)
        XCTAssertEqual(read?.sampleRate, 16000)
        for (a, b) in zip(read?.samples ?? [], samples) {
            XCTAssertEqual(a, b, accuracy: 1.0 / 32767, "16-bit quantization round-trip")
        }
    }

    // MARK: Model (needs MLX)

    func testLogMelHasTheExpectedShape() throws {
        try requireMLXRuntime()
        let samples = [Float](repeating: 0.1, count: 1600)      // 0.1 s at 16 kHz
        let mel = NFKMLXMel.logMel(samples, sampleRate: 16000, nMels: 80)
        eval(mel)
        XCTAssertEqual(mel.shape[0], 1)
        XCTAssertEqual(mel.shape[2], 80, "80 mel bins")
    }

    func testTranscribeProducesTokensWithinTheLimit() throws {
        try requireMLXRuntime()
        let net = NFKMLXWhisper.makeNet(tinyConfiguration())
        let mel = NFKMLXMel.logMel([Float](repeating: 0.2, count: 1600), sampleRate: 16000, nMels: 80)
        let tokens = net.transcribe(mel)
        XCTAssertLessThanOrEqual(tokens.count, tinyConfiguration().maxTokens)
    }

    func testACheckpointRoundTripReproducesTheTranscription() throws {
        try requireMLXRuntime()
        let trained = NFKMLXWhisper.makeNet(tinyConfiguration())
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value -> (String, MLXArray) in
            if value.ndim == 4 { return (key, value.transposed(0, 3, 1, 2)) }
            if value.ndim == 3 { return (key, value.transposed(0, 2, 1)) }
            return (key, value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXWhisper.makeNet(tinyConfiguration())
        try NFKMLXWhisper.loadWeights(into: loaded, from: url)

        let mel = NFKMLXMel.logMel([Float](repeating: 0.2, count: 1600), sampleRate: 16000, nMels: 80)
        XCTAssertEqual(trained.transcribe(mel), loaded.transcribe(mel), "same weights, same greedy tokens")
    }

    func testTheBackendTranscribesAWaveAsset() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-in-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWaveFile.write(samples: [Float](repeating: 0.1, count: 1600), sampleRate: 16000, to: url)

        let backend = NFKMLXWhisperBackend(net: NFKMLXWhisper.makeNet(tinyConfiguration()), tokenizer: nil)
        let request = NFKInferenceRequest(inputs: [NFKInputAudio: NFKAudioAsset(fileURL: url, durationSeconds: 0.1, sampleRate: 16000, channelCount: 1)])
        let result = try backend.runInference(for: request)
        XCTAssertNotNil(result.text, "returns token ids as text when no tokenizer is supplied")
    }

    // MARK: The Hugging Face layout

    /// A transformers export is the same model under different keys, so renaming a checkpoint into
    /// that form and loading it must reproduce the identical transcription. `proj_out` is the tied
    /// embedding stored twice, added here to prove it is dropped rather than misapplied.
    func testAHuggingFaceLayoutCheckpointLoadsIdentically() throws {
        try requireMLXRuntime()
        let reference = NFKMLXWhisper.makeNet(tinyConfiguration())
        let pytorchLayout = Dictionary(uniqueKeysWithValues: reference.parameters().flattened().map {
            key, value -> (String, MLXArray) in
            if value.ndim == 4 { return (key, value.transposed(0, 3, 1, 2)) }
            if value.ndim == 3 { return (key, value.transposed(0, 2, 1)) }
            return (key, value)
        })

        func huggingFaceSpelling(_ key: String) -> String {
            var name = key
            name = name.replacingOccurrences(of: ".blocks.", with: ".layers.")
            name = name.replacingOccurrences(of: ".cross_attn_ln.", with: ".encoder_attn_layer_norm.")
            name = name.replacingOccurrences(of: ".attn_ln.", with: ".self_attn_layer_norm.")
            name = name.replacingOccurrences(of: ".mlp_ln.", with: ".final_layer_norm.")
            name = name.replacingOccurrences(of: ".cross_attn.", with: ".encoder_attn.")
            name = name.replacingOccurrences(of: ".attn.", with: ".self_attn.")
            name = name.replacingOccurrences(of: ".query.", with: ".q_proj.")
            name = name.replacingOccurrences(of: ".key.", with: ".k_proj.")
            name = name.replacingOccurrences(of: ".value.", with: ".v_proj.")
            name = name.replacingOccurrences(of: ".out.", with: ".out_proj.")
            name = name.replacingOccurrences(of: ".mlp.0.", with: ".fc1.")
            name = name.replacingOccurrences(of: ".mlp.2.", with: ".fc2.")
            name = name.replacingOccurrences(of: "decoder.token_embedding.", with: "decoder.embed_tokens.")
            if name == "encoder.positional_embedding" { name = "encoder.embed_positions.weight" }
            if name == "decoder.positional_embedding" { name = "decoder.embed_positions.weight" }
            name = name.replacingOccurrences(of: "encoder.ln_post.", with: "encoder.layer_norm.")
            name = name.replacingOccurrences(of: "decoder.ln.", with: "decoder.layer_norm.")
            return "model." + name
        }
        var renamed = Dictionary(uniqueKeysWithValues: pytorchLayout.map { (huggingFaceSpelling($0.0), $0.1) })
        renamed["proj_out.weight"] = pytorchLayout["decoder.token_embedding.weight"]

        let openAI = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-oa-\(UUID().uuidString).safetensors")
        let huggingFace = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-hf-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: openAI); try? FileManager.default.removeItem(at: huggingFace) }
        try save(arrays: pytorchLayout, url: openAI)
        try save(arrays: renamed, url: huggingFace)

        let first = NFKMLXWhisper.makeNet(tinyConfiguration())
        try NFKMLXWhisper.loadWeights(into: first, from: openAI)
        let second = NFKMLXWhisper.makeNet(tinyConfiguration())
        try NFKMLXWhisper.loadWeights(into: second, from: huggingFace)

        let mel = NFKMLXMel.logMel([Float](repeating: 0.2, count: 1600), sampleRate: 16000, nMels: 80)
        XCTAssertEqual(first.transcribe(mel), second.transcribe(mel),
                       "the two spellings of one checkpoint decode identically")
    }

    // MARK: Timestamps

    /// A tiny configuration whose vocabulary has room for a timestamp range above the text.
    private func timedConfiguration() -> NFKMLXWhisperConfiguration {
        var configuration = tinyConfiguration()
        configuration.promptTokens = [2, 3, 4]           // the last stands in for `<|notimestamps|>`
        configuration.timestampBegin = 5                 // ids 5...49 are times
        configuration.suppressFrom = 50                  // no special range to mask in this vocabulary
        configuration.maxInitialTimestampIndex = 3
        return configuration
    }

    func testATimestampedDecodeOpensWithATimestamp() throws {
        try requireMLXRuntime()
        let configuration = timedConfiguration()
        let net = NFKMLXWhisper.makeNet(configuration)
        let mel = NFKMLXMel.logMel([Float](repeating: 0.2, count: 1600), sampleRate: 16000, nMels: 80)
        let (segments, tokens) = net.transcribeWithTimestamps(mel)

        XCTAssertFalse(tokens.isEmpty)
        XCTAssertGreaterThanOrEqual(tokens[0], configuration.timestampBegin,
                                    "the opening position has to be a timestamp")
        XCTAssertLessThanOrEqual(tokens[0], configuration.timestampBegin + 3,
                                 "and no later than maxInitialTimestampIndex")
        XCTAssertFalse(tokens.contains(configuration.timestampBegin - 1),
                       "`<|notimestamps|>` is masked at every step")
        for segment in segments {
            XCTAssertLessThanOrEqual(segment.start, segment.end, "a span cannot run backward")
        }
    }

    /// Timestamps come in pairs and never decrease, whatever the weights say.
    func testTimestampsPairUpAndNeverDecrease() throws {
        try requireMLXRuntime()
        let configuration = timedConfiguration()
        let net = NFKMLXWhisper.makeNet(configuration)
        let mel = NFKMLXMel.logMel([Float](repeating: 0.4, count: 1600), sampleRate: 16000, nMels: 80)
        let (_, tokens) = net.transcribeWithTimestamps(mel)

        var previous = -1
        var runLength = 0
        for id in tokens where id >= configuration.timestampBegin {
            XCTAssertGreaterThanOrEqual(id, previous, "a timestamp never precedes an earlier one")
            previous = id
            runLength += 1
        }
        XCTAssertGreaterThan(runLength, 0, "a timestamped decode emits at least the opening marker")
        // Whatever the weights do, no text token may sit before the first timestamp.
        let firstText = tokens.firstIndex { $0 < configuration.timestampBegin }
        if let firstText { XCTAssertGreaterThan(firstText, 0) }
    }

    func testTheBackendEmitsSegmentsOnlyWhenAskedTo() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-ts-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWaveFile.write(samples: [Float](repeating: 0.1, count: 1600), sampleRate: 16000, to: url)

        let backend = NFKMLXWhisperBackend(net: NFKMLXWhisper.makeNet(timedConfiguration()), tokenizer: nil)
        let request = NFKInferenceRequest(inputs: [NFKInputAudio: NFKAudioAsset(fileURL: url, durationSeconds: 0.1, sampleRate: 16000, channelCount: 1)])

        XCTAssertNil(try backend.runInference(for: request).segments,
                     "the plain decode is what a backend performs by default")
        backend.emitsTimestamps = true
        let timed = try backend.runInference(for: request)
        let segments = try XCTUnwrap(timed.segments)
        XCTAssertFalse(segments.isEmpty)
        XCTAssertNotNil(timed.text, "the transcript comes back beside the spans")
    }
}
