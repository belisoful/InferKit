//
//  NFKMLXWhisperTests.swift
//  InferKitMLXTests
//
//  Encoder-decoder transformer + log-mel. The WAV read/write round-trip needs no GPU; the model
//  path evaluates MLX arrays, so it skips without a Metal library for MLX (see
//  Tools/mlx-metallib.sh).
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
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
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

    // MARK: Catalog builds

    /// A release directory as the catalog's source lays it out: the tokenizer's vocab.json and merges.txt,
    /// each of the timed configuration's 50 ids a letter, beside a checkpoint of a random net.
    private func releaseDirectory(withCheckpoint: Bool) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-release-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let letters = Array("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX").map(String.init)
        let vocabulary = Dictionary(uniqueKeysWithValues: letters.enumerated().map { ($1, $0) })
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        if withCheckpoint {
            let net = NFKMLXWhisper.makeNet(timedConfiguration())
            let pytorchLayout = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { key, value -> (String, MLXArray) in
                if value.ndim == 4 { return (key, value.transposed(0, 3, 1, 2)) }
                if value.ndim == 3 { return (key, value.transposed(0, 2, 1)) }
                return (key, value)
            })
            try save(arrays: pytorchLayout, url: directory.appendingPathComponent("model.safetensors"))
        }
        return directory
    }

    func testACatalogBuildDecodesTextWithTheTokenizerInItsDirectory() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXWhisper.catalogBackend(configuration: timedConfiguration(), location: try releaseDirectory(withCheckpoint: true),
                                                       weightsDownload: nil, precision: .checkpoint, identifier: "whisper-test")
        XCTAssertNotNil(backend.tokenizer)
        XCTAssertTrue(backend.emitsTimestamps, "a catalog build answers segments by default")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-catalog-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWaveFile.write(samples: [Float](repeating: 0.1, count: 1600), sampleRate: 16000, to: url)
        let audio = NFKAudioAsset(fileURL: url, durationSeconds: 0.1, sampleRate: 16000, channelCount: 1)

        let timed = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: audio]))
        let text = try XCTUnwrap(timed.text)
        XCTAssertTrue(CharacterSet.letters.isSuperset(of: CharacterSet(charactersIn: text)),
                      "every id decodes to its letter, never to its number: \(text)")
        XCTAssertNotNil(timed.segments)
        let plain = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: audio],
                                                                       parameters: [NFKMLXGenerationParameterKey.timestamps: false]))
        XCTAssertNil(plain.segments, "a request turns the timestamps off")
    }

    func testACatalogBuildWithoutItsCheckpointNamesWhatIsMissing() throws {
        let directory = try releaseDirectory(withCheckpoint: false)
        XCTAssertThrowsError(try NFKMLXWhisper.catalogBackend(configuration: timedConfiguration(), location: directory,
                                                              weightsDownload: nil, precision: .checkpoint, identifier: "whisper-test")) { error in
            XCTAssertTrue("\(error)".contains("no Whisper checkpoint"), "\(error)")
        }
        XCTAssertNotNil(NFKMLXWhisper.tokenizer(in: directory))
        XCTAssertNil(NFKMLXWhisper.tokenizer(in: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)))
    }

    func testACatalogBuildDecodesAsLongAsTheReferenceSampleLength() {
        for variant in [NFKMLXWhisperVariant.tiny, .largeV3] {
            XCTAssertEqual(NFKMLXWhisper.catalogConfiguration(for: variant).maxTokens, 224, "the reference's sample_len is n_text_ctx // 2")
        }
    }

    /// The catalog's whisper-tiny as a host builds it, from the default Hugging Face cache: the
    /// tokenizer from openai/whisper-tiny and OpenAI's checkpoint beside it. Skips until the release is
    /// downloaded through the catalog.
    func testTheCatalogsWhisperTinyTranscribesSpeechToTextAndSegments() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerAll()
        let catalog = NFKModelCatalog()
        XCTAssertTrue(catalog.addDescriptors(fromProviderClassName: "NFKMLXModelDescriptors"))
        let source = try XCTUnwrap(catalog.descriptor(withIdentifier: NFKMLXWhisper.modelName)?.defaultVariant?.source)
        guard NFKHFHub(cacheDirectoryURL: NFKHFHub.defaultCacheDirectoryURL()).downloadStatus(source: source).presence == .complete else {
            throw XCTSkip("download the catalog's whisper-tiny into the default Hugging Face cache first")
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-speech-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let spoken = directory.appendingPathComponent("spoken.aiff")
        let wave = directory.appendingPathComponent("spoken.wav")
        for (tool, arguments) in [("/usr/bin/say", ["-o", spoken.path, "Hello from the model catalog. This sentence tests the transcript."]),
                                  ("/usr/bin/afconvert", ["-f", "WAVE", "-d", "LEI16@16000", spoken.path, wave.path])] {
            let process = try Process.run(URL(fileURLWithPath: tool), arguments: arguments)
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus, 0, tool)
        }
        let backend = try catalog.backend(forModelIdentifier: NFKMLXWhisper.modelName, options: nil)
        let audio = NFKAudioAsset(fileURL: wave, durationSeconds: 0, sampleRate: 16000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: audio]))
        let text = try XCTUnwrap(result.text)
        print("VALIDATION whisper-tiny catalog: \(text)")
        XCTAssertTrue(text.lowercased().contains("hello"), "the catalog build answers text, not token ids: \(text)")
        let segments = try XCTUnwrap(result.segments)
        XCTAssertFalse(segments.isEmpty)
        XCTAssertTrue(segments.allSatisfy { $0.endSeconds >= $0.startSeconds && !($0.label ?? "").isEmpty })
    }

    // MARK: Long-form transcription

    /// The reference's reading of a window (timestamps begin at id 5, 20 ms apart): adjacent timestamps
    /// close segments, a window ending on one timestamp is taken whole, one ending on text moves to its last
    /// closed timestamp, and a window with no closed pair is one segment to its last timestamp or its end.
    func testAWindowsTimestampsMakeTheReferencesSegmentsAndSeek() throws {
        try requireMLXRuntime()
        let net = NFKMLXWhisper.makeNet(timedConfiguration())
        let ending = net.windowSegments([5, 0, 0, 15, 15, 0, 30], offset: 10, frames: 3000)
        XCTAssertEqual(ending.segments.map(\.tokens), [[5, 0, 0, 15], [15, 0, 30]])
        XCTAssertEqual(ending.segments[0].start, 10, accuracy: 1e-12)
        XCTAssertEqual(ending.segments[1].start, 10.2, accuracy: 1e-12)
        XCTAssertEqual(ending.segments[0].end, 10.2, accuracy: 1e-12)
        XCTAssertEqual(ending.segments[1].end, 10.5, accuracy: 1e-12)
        XCTAssertEqual(ending.advance, 3000, "a window ending on a single timestamp is taken whole")

        let open = net.windowSegments([5, 0, 15, 15, 0], offset: 0, frames: 3000)
        XCTAssertEqual(open.segments.map(\.tokens), [[5, 0, 15]])
        XCTAssertEqual(open.advance, 20, "the next window starts at the last closed timestamp, two frames a step")

        let untimed = net.windowSegments([5, 0, 0], offset: 4, frames: 1200)
        XCTAssertEqual(untimed.segments.count, 1)
        XCTAssertEqual(untimed.segments[0].end, 16, accuracy: 1e-12, "no closed pair runs to the window's end")
        XCTAssertEqual(untimed.advance, 1200)

        let closedOnce = net.windowSegments([5, 0, 12], offset: 0, frames: 3000)
        XCTAssertEqual(closedOnce.segments[0].end, 0.14, accuracy: 1e-12, "to its last timestamp")
        XCTAssertEqual(closedOnce.advance, 3000)
    }

    func testACatalogBuildTranscribesTheWholeRecording() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXWhisper.catalogBackend(configuration: timedConfiguration(), location: try releaseDirectory(withCheckpoint: true),
                                                       weightsDownload: nil, precision: .checkpoint, identifier: "whisper-test")
        XCTAssertTrue(backend.transcribesWholeRecording)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-long-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let samples = (0 ..< 40 * 16000).map { index in 0.1 * sinf(Float(index) * 0.05) }
        try NFKMLXWaveFile.write(samples: samples, sampleRate: 16000, to: url)
        let audio = NFKAudioAsset(fileURL: url, durationSeconds: 40, sampleRate: 16000, channelCount: 1)

        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: audio]))
        let text = try XCTUnwrap(result.text)
        XCTAssertTrue(CharacterSet.letters.isSuperset(of: CharacterSet(charactersIn: text)), text)
        let segments = try XCTUnwrap(result.segments)
        XCTAssertEqual(segments.map(\.startSeconds), segments.map(\.startSeconds).sorted(), "segments come in file order")
        XCTAssertTrue(segments.allSatisfy { $0.endSeconds <= 70 && $0.startSeconds <= $0.endSeconds })
    }

    /// A decode that feeds the cache one new token a step gives each step the logits the full recompute
    /// gives its last row.
    func testTheCachedDecodeMatchesTheFullRecomputeRowByRow() throws {
        try requireMLXRuntime()
        let net = NFKMLXWhisper.makeNet(timedConfiguration())
        let mel = NFKMLXMel.logMel([Float](repeating: 0.2, count: 16000), sampleRate: 16000, nMels: 80)
        let audio = net.encoder(mel)
        let tokens = [2, 3, 4, 7, 9, 11]
        let full = net.decoder(MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count]), audio: audio)[0]
        let caches = net.decoder.blocks.map { _ in NFKWhisperBlockCache() }
        let opening = net.decoder(MLXArray(tokens.prefix(3).map { Int32($0) }).reshaped([1, 3]), audio: audio, caches: caches)[0]
        XCTAssertLessThan(abs(opening - full[0 ..< 3]).max().item(Float.self), 1e-4)
        for index in 3 ..< tokens.count {
            let step = net.decoder(MLXArray([Int32(tokens[index])]).reshaped([1, 1]), audio: audio, caches: caches)[0, 0]
            XCTAssertLessThan(abs(step - full[index]).max().item(Float.self), 1e-4, "step \(index)")
            XCTAssertEqual(step.argMax().item(Int.self), full[index].argMax().item(Int.self))
        }
    }

    /// The opening prompt names the language and the task, and detection picks among the language tokens.
    func testTheWindowPromptNamesTheLanguageAndTheTask() throws {
        try requireMLXRuntime()
        let net = NFKMLXWhisper.makeNet(NFKMLXWhisper.configuration(for: .tiny))
        XCTAssertEqual(net.promptSequence(language: 50259, translates: false, timestamps: false), [50258, 50259, 50359, 50363])
        XCTAssertEqual(net.promptSequence(language: 50260, translates: true, timestamps: true), [50258, 50260, 50358])
        XCTAssertEqual(net.languageTokens, 50259 ..< 50358, "99 languages before <|translate|>")
        XCTAssertEqual(NFKMLXWhisper.languageTable(for: net)["zh"], 50260)
        let mel = NFKMLXMel.logMel([Float](repeating: 0.1, count: 30 * 16000), sampleRate: 16000, nMels: 80)
        XCTAssertTrue(net.languageTokens.contains(net.detectLanguage(mel)))
    }

    func testAReleasesAddedTokensNameItsLanguages() throws {
        try requireMLXRuntime()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-added-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let added = ["<|endoftext|>": 50257, "<|en|>": 50259, "<|xx|>": 50260, "<|translate|>": 50358]
        try JSONSerialization.data(withJSONObject: added).write(to: directory.appendingPathComponent("added_tokens.json"))
        let net = NFKMLXWhisper.makeNet(NFKMLXWhisper.configuration(for: .tiny))
        XCTAssertEqual(NFKMLXWhisper.languageTable(in: directory, for: net), ["en": 50259, "xx": 50260],
                       "only the tokens in the language range are languages")
    }
}
