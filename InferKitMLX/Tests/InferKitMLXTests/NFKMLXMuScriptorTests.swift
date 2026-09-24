//
//  NFKMLXMuScriptorTests.swift
//  InferKitMLXTests
//
//  MuScriptor multi-instrument music transcription. The vocabulary and the decode state machine are
//  weight-free and run always. The architecture is measured at a TINY random configuration against
//  the reference's own `_build_model` (`IK_PARITY_MUSCRIPTOR`, from `run_reference.py muscriptor`
//  under the `msvenv` interpreter), the way the SD3 and FLUX transformers were: the released weights
//  are CC BY-NC 4.0 behind a gated repository, and the record carries the tiny model's parameters
//  under `w::` so both sides run identical numbers.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXMuScriptorTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private static let tiny = NFKMLXMuScriptorConfiguration(dimension: 64, heads: 4, layers: 2, card: 1395)

    // MARK: Vocabulary

    func testTheVocabularyIsTheReferenceLayout() {
        let vocabulary = NFKMuScriptorVocabulary()
        // Three special tokens, 1001 shifts, 128 pitches, 2 velocities, tie, 130 programs, 128 drums.
        XCTAssertEqual(vocabulary.count, 1393)
        XCTAssertEqual(vocabulary.event(at: 0)?.kind, .pad)
        XCTAssertEqual(vocabulary.event(at: 1)?.kind, .eos)
        XCTAssertEqual(vocabulary.endOfSequence, 1)
        XCTAssertEqual(vocabulary.event(at: 3), NFKMuScriptorEvent(kind: .shift, value: 0))
        XCTAssertEqual(vocabulary.event(at: 1003), NFKMuScriptorEvent(kind: .shift, value: 1000))
        XCTAssertEqual(vocabulary.event(at: 1004), NFKMuScriptorEvent(kind: .pitch, value: 0))
        XCTAssertEqual(vocabulary.token(kind: .tie, value: 0), 1134)
        XCTAssertEqual(vocabulary.token(kind: .program, value: 0), 1135)
        XCTAssertEqual(vocabulary.token(kind: .drum, value: 0), 1265)
    }

    func testTheReleasedHeadIsWiderThanTheVocabulary() {
        // The medium and large heads score 1395 tokens; the two past the vocabulary are masked and
        // can never be sampled.
        XCTAssertEqual(NFKMLXMuScriptorConfiguration.medium.card, 1395)
        XCTAssertEqual(NFKMLXMuScriptorConfiguration.medium.maskedFromToken, NFKMuScriptorVocabulary().count)
        XCTAssertEqual(NFKMLXMuScriptorConfiguration.small.card, NFKMuScriptorVocabulary().count)
    }

    func testTheTieSectionDeclaresEachProgramOnce() {
        let vocabulary = NFKMuScriptorVocabulary()
        let tokens = vocabulary.tieSectionTokens(openNotes: [(program: 0, pitch: 64),
                                                             (program: 0, pitch: 60),
                                                             (program: 40, pitch: 72)])
        let events = tokens.compactMap { vocabulary.event(at: $0) }
        XCTAssertEqual(events.map(\.kind), [.program, .pitch, .pitch, .program, .pitch, .tie])
        XCTAssertEqual(events.map(\.value), [0, 60, 64, 40, 72, 0])
    }

    // MARK: Decode

    private func tokens(_ vocabulary: NFKMuScriptorVocabulary,
                        _ items: [(NFKMuScriptorEvent.Kind, Int)]) -> [Int] {
        items.compactMap { vocabulary.token(kind: $0.0, value: $0.1) }
    }

    func testAChunkDecodesIntoNotes() {
        let vocabulary = NFKMuScriptorVocabulary()
        // An empty tie section, then middle C on program 0 from 0.5 s to 1.2 s.
        let stream = tokens(vocabulary, [(.tie, 0), (.shift, 50), (.program, 0), (.velocity, 1), (.pitch, 60),
                                         (.shift, 120), (.program, 0), (.velocity, 0), (.pitch, 60)])
        let notes = NFKMuScriptorNotes.notes(
            chunks: [(NFKMuScriptorChunkBoundary(seekSeconds: 0, nextSeekSeconds: nil), stream)],
            vocabulary: vocabulary)

        XCTAssertEqual(notes.count, 1)
        XCTAssertEqual(notes.first?.pitch, 60)
        XCTAssertEqual(notes.first?.startSeconds ?? 0, 0.5, accuracy: 1e-9)
        XCTAssertEqual(notes.first?.endSeconds ?? 0, 1.2, accuracy: 1e-9)
        XCTAssertEqual(notes.first?.velocity, 100)
        XCTAssertFalse(notes.first?.isPercussion ?? true)
    }

    func testADrumHitIsPercussionOfTheMinimumLength() {
        let vocabulary = NFKMuScriptorVocabulary()
        let stream = tokens(vocabulary, [(.tie, 0), (.shift, 25), (.drum, 36)])
        let notes = NFKMuScriptorNotes.notes(
            chunks: [(NFKMuScriptorChunkBoundary(seekSeconds: 0, nextSeekSeconds: nil), stream)],
            vocabulary: vocabulary)

        XCTAssertEqual(notes.count, 1)
        XCTAssertTrue(notes.first?.isPercussion ?? false)
        XCTAssertEqual(notes.first?.pitch, 36)
        XCTAssertEqual(notes.first?.durationSeconds ?? 0, 0.01, accuracy: 1e-9)
    }

    func testANoteHeldAcrossAChunkSurvivesItsTieSection() {
        let vocabulary = NFKMuScriptorVocabulary()
        let first = tokens(vocabulary, [(.tie, 0), (.shift, 100), (.program, 0), (.velocity, 1), (.pitch, 67)])
        // The second chunk declares the note as sustained, then releases it a second in.
        let second = tokens(vocabulary, [(.program, 0), (.pitch, 67), (.tie, 0),
                                         (.shift, 100), (.program, 0), (.velocity, 0), (.pitch, 67)])
        let notes = NFKMuScriptorNotes.notes(chunks: [
            (NFKMuScriptorChunkBoundary(seekSeconds: 0, nextSeekSeconds: 5), first),
            (NFKMuScriptorChunkBoundary(seekSeconds: 5, nextSeekSeconds: nil), second),
        ], vocabulary: vocabulary)

        XCTAssertEqual(notes.count, 1, "the note is one note, not one per chunk")
        XCTAssertEqual(notes.first?.startSeconds ?? 0, 1.0, accuracy: 1e-9)
        XCTAssertEqual(notes.first?.endSeconds ?? 0, 6.0, accuracy: 1e-9)
    }

    func testANoteTheTieSectionDropsEndsAtTheBoundary() {
        let vocabulary = NFKMuScriptorVocabulary()
        let first = tokens(vocabulary, [(.tie, 0), (.shift, 100), (.program, 0), (.velocity, 1), (.pitch, 67)])
        let second = tokens(vocabulary, [(.tie, 0), (.shift, 50)])         // declares nothing sustained
        let notes = NFKMuScriptorNotes.notes(chunks: [
            (NFKMuScriptorChunkBoundary(seekSeconds: 0, nextSeekSeconds: 5), first),
            (NFKMuScriptorChunkBoundary(seekSeconds: 5, nextSeekSeconds: nil), second),
        ], vocabulary: vocabulary)

        XCTAssertEqual(notes.count, 1)
        XCTAssertEqual(notes.first?.endSeconds ?? 0, 5.0, accuracy: 1e-9, "it ends where the chunk does")
    }

    func testAChunkWithoutATieTokenIsDropped() {
        let vocabulary = NFKMuScriptorVocabulary()
        // A shift before any tie token: the chunk is malformed, so everything open closes and the
        // rest of the chunk is discarded.
        let first = tokens(vocabulary, [(.tie, 0), (.shift, 100), (.program, 0), (.velocity, 1), (.pitch, 67)])
        let malformed = tokens(vocabulary, [(.shift, 10), (.program, 0), (.velocity, 1), (.pitch, 72)])
        let notes = NFKMuScriptorNotes.notes(chunks: [
            (NFKMuScriptorChunkBoundary(seekSeconds: 0, nextSeekSeconds: 5), first),
            (NFKMuScriptorChunkBoundary(seekSeconds: 5, nextSeekSeconds: nil), malformed),
        ], vocabulary: vocabulary)

        XCTAssertEqual(notes.count, 1, "nothing from the malformed chunk is kept")
        XCTAssertEqual(notes.first?.endSeconds ?? 0, 5.0, accuracy: 1e-9)
    }

    func testANotePastTheNextChunksStartIsLeftToThatChunk() {
        let vocabulary = NFKMuScriptorVocabulary()
        // The shift places the note at 6 s, past this chunk's 5 s window.
        let stream = tokens(vocabulary, [(.tie, 0), (.shift, 600), (.program, 0), (.velocity, 1), (.pitch, 60)])
        let notes = NFKMuScriptorNotes.notes(
            chunks: [(NFKMuScriptorChunkBoundary(seekSeconds: 0, nextSeekSeconds: 5), stream)],
            vocabulary: vocabulary)
        XCTAssertEqual(notes.count, 0)
    }

    // MARK: Module layout

    func testParameterNamesFollowTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        let net = NFKMLXMuScriptor.makeNet(Self.tiny)
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["emb.weight", "out_norm.weight", "linear.weight",
                         "transformer.layers.0.self_attn.in_proj_weight",
                         "transformer.layers.0.self_attn.out_proj.weight",
                         "transformer.layers.0.norm1.weight", "transformer.layers.0.linear1.weight",
                         "condition_provider.conditioners.self_wav.output_proj.weight",
                         "condition_provider.conditioners.self_wav.mel_spec_transform.spectrogram.window",
                         "condition_provider.conditioners.self_wav.mel_spec_transform.mel_scale.fb",
                         "condition_provider.conditioners.instrument_group.embed.weight",
                         "condition_provider.conditioners.dataset_name.embed.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheVariantNamesAreTheRegistryNames() {
        // The medium release is the reference's default, so it takes the bare name. A backend reports
        // whichever name it was registered under; building one here would allocate a 307M-parameter
        // random model to read a string.
        XCTAssertEqual(NFKMLXMuScriptor.registeredName(for: .medium), NFKMLXMuScriptor.modelName)
        XCTAssertEqual(NFKMLXMuScriptor.registeredName(for: .small), "muscriptor-small")
        XCTAssertEqual(NFKMLXMuScriptor.registeredName(for: .large), "muscriptor-large")
    }

    func testTheLegacyModuleListKeysRemap() {
        XCTAssertEqual(NFKMLXMuScriptor.remapReferenceKey("emb.0.weight"), "emb.weight")
        XCTAssertEqual(NFKMLXMuScriptor.remapReferenceKey("linears.0.weight"), "linear.weight")
        XCTAssertEqual(NFKMLXMuScriptor.remapReferenceKey("out_norm.bias"), "out_norm.bias")
    }

    // MARK: Reference parity

    private func cosine(_ mine: [Float], _ reference: [Float]) -> Float {
        let count = min(mine.count, reference.count)
        var dot: Float = 0, left: Float = 0, right: Float = 0
        for index in 0 ..< count {
            dot += mine[index] * reference[index]
            left += mine[index] * mine[index]
            right += reference[index] * reference[index]
        }
        return dot / (sqrtf(left) * sqrtf(right) + 1e-20)
    }

    private func parity() throws -> (net: NFKMLXMuScriptorNet, record: [String: MLXArray]) {
        guard let recordPath = NFKMLXValidationConfig.environment["IK_PARITY_MUSCRIPTOR"] else {
            throw XCTSkip("set IK_PARITY_MUSCRIPTOR (from run_reference.py muscriptor)")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let net = NFKMLXMuScriptor.makeNet(Self.tiny)
        // The record carries the tiny model's own parameters, so both sides run identical numbers.
        let weights = record.filter { $0.key.hasPrefix("w::") }
            .map { (NFKMLXMuScriptor.remapReferenceKey(String($0.key.dropFirst(3))), $0.value) }
        XCTAssertFalse(weights.isEmpty, "the record carries no w:: weights")
        try NFKMLXWeights.apply(weights, to: net)
        return (net, record)
    }

    func testTheMelFrontEndMatchesTheReference() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let audio = record["audio"]!.asArray(Float.self)
        let conditioner = net.conditionProvider.conditioners.selfWav

        let mine = conditioner.melSpectrogram(audio)                        // [frames, mels]
        eval(mine)
        let reference = record["mel"]!.transposed(1, 0)                     // [frames, mels]
        XCTAssertEqual(mine.shape, reference.shape)
        let value = cosine(mine.asArray(Float.self), reference.asArray(Float.self))
        print("VALIDATION PARITY muscriptor: mel cosine \(value)")
        XCTAssertGreaterThan(value, 0.9999999, "the mel front end diverges")

        let embedded = conditioner(audio)
        eval(embedded)
        let embedCosine = cosine(embedded.asArray(Float.self), record["mel_embed"]!.asArray(Float.self))
        print("VALIDATION PARITY muscriptor: mel conditioner cosine \(embedCosine)")
        XCTAssertGreaterThan(embedCosine, 0.9999, "the log and projection diverge")

        // Centered framing yields one frame more than the clip fills, and the reference's length mask
        // zeroes it. Keeping it would feed the transformer a position the reference never sees.
        let frames = embedded.dim(1)
        XCTAssertEqual(conditioner.validFrames(sampleCount: audio.count), frames - 1)
        let last = embedded[0, frames - 1, 0...].asArray(Float.self)
        XCTAssertTrue(last.allSatisfy { $0 == 0 }, "the final mel frame is masked away")
    }

    func testTheClassConditionersIndexTheSameRows() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let conditioners = net.conditionProvider.conditioners

        // An unspecified class reads row 1, not row 0: the reference shifts the index twice.
        let unspecified = conditioners.instrumentGroup(nil)
        eval(unspecified)
        XCTAssertGreaterThan(cosine(unspecified.asArray(Float.self),
                                    record["instrument_unspecified"]!.asArray(Float.self)), 0.99999)

        let specified = conditioners.instrumentGroup(5)
        eval(specified)
        XCTAssertGreaterThan(cosine(specified.asArray(Float.self),
                                    record["instrument_specified"]!.asArray(Float.self)), 0.99999)

        let dataset = conditioners.datasetName(nil)
        eval(dataset)
        XCTAssertGreaterThan(cosine(dataset.asArray(Float.self),
                                    record["dataset_embed"]!.asArray(Float.self)), 0.99999)
    }

    func testTheLogitsMatchTheReference() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let audio = record["audio"]!.asArray(Float.self)

        let prefix = net.conditioningPrefix(samples: audio)
        let caches = (0 ..< Self.tiny.layers).map { _ in NFKMuScriptorLayerCache() }
        let logits = net.logits(prefix: prefix, tokens: [net.initialToken], caches: caches)
        eval(logits)

        let value = cosine(logits.asArray(Float.self), record["output"]!.asArray(Float.self))
        print("VALIDATION PARITY muscriptor: prefill logits cosine \(value)")
        XCTAssertGreaterThan(value, 0.9999, "the transformer diverges")
    }

    func testTheGreedyContinuationMatchesTheReference() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let audio = record["audio"]!.asArray(Float.self)
        let reference = record["tokens"]!.asArray(Int32.self).map(Int.init)

        // The reference recomputes each step from scratch; this decodes through the KV cache, so
        // agreement also says the cache carries what the prefill put in it.
        let mine = net.generate(chunk: audio)
        XCTAssertGreaterThanOrEqual(mine.count, 1)
        let shared = min(mine.count, reference.count)
        XCTAssertEqual(Array(mine.prefix(shared)), Array(reference.prefix(shared)),
                       "the decoded tokens diverge")
    }

    // MARK: Released weights

    /// The released medium model: every tensor lands, and the seams match the reference on the same
    /// clip. Gated on the weights (`IK_VAL_MUSCRIPTOR`) and the record (`IK_PARITY_MUSCRIPTOR_REAL`).
    func testTheReleasedWeightsMatchTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_MUSCRIPTOR"],
              let recordPath = environment["IK_PARITY_MUSCRIPTOR_REAL"] else {
            throw XCTSkip("set IK_VAL_MUSCRIPTOR (the release) and IK_PARITY_MUSCRIPTOR_REAL")
        }
        let net = NFKMLXMuScriptor.makeNet(.medium)
        try NFKMLXMuScriptor.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let audio = record["audio"]!.asArray(Float.self)

        let conditioner = net.conditionProvider.conditioners.selfWav
        let mel = conditioner.melSpectrogram(audio)
        eval(mel)
        let melValue = cosine(mel.asArray(Float.self), record["mel"]!.transposed(1, 0).asArray(Float.self))
        print("VALIDATION PARITY muscriptor_real: mel cosine \(melValue)")
        // Float32 over 501 frames of 512 bins: the measured value sits at the last representable
        // step below one.
        XCTAssertGreaterThan(melValue, 0.999999)

        let prefix = net.conditioningPrefix(samples: audio)
        eval(prefix)
        XCTAssertEqual(prefix.dim(1), record["prefix"]!.dim(0), "the prefix is the reference's length")
        let prefixValue = cosine(prefix.asArray(Float.self), record["prefix"]!.asArray(Float.self))
        print("VALIDATION PARITY muscriptor_real: conditioning prefix cosine \(prefixValue)")
        XCTAssertGreaterThan(prefixValue, 0.9999, "the prefix diverges (order, mask, or projection)")

        let caches = (0 ..< NFKMLXMuScriptorConfiguration.medium.layers).map { _ in NFKMuScriptorLayerCache() }
        let logits = net.logits(prefix: prefix, tokens: [net.initialToken], caches: caches)
        eval(logits)
        let logitValue = cosine(logits.asArray(Float.self), record["output"]!.asArray(Float.self))
        print("VALIDATION PARITY muscriptor_real: prefill logits cosine \(logitValue)")
        XCTAssertGreaterThan(logitValue, 0.9999, "the transformer diverges on the released weights")

        // The decoded events themselves, which is what a transcription is made of.
        let reference = record["tokens"]!.asArray(Int32.self).map(Int.init)
        let mine = net.generate(chunk: audio)
        let expected = reference.last == net.vocabulary.endOfSequence ? Array(reference.dropLast()) : reference
        let shared = min(mine.count, expected.count)
        XCTAssertGreaterThan(shared, 0, "the reference generated nothing to compare against")
        XCTAssertEqual(Array(mine.prefix(shared)), Array(expected.prefix(shared)),
                       "the decoded token stream diverges")
        print("VALIDATION PARITY muscriptor_real: \(shared) tokens match, first events "
              + "\(expected.prefix(6).compactMap { net.vocabulary.event(at: $0) }.map { "\($0.kind.rawValue):\($0.value)" })")

        // The decode state machine over the same tokens, action for action against the reference's.
        let tracker = NFKMuScriptorNoteTracker(vocabulary: net.vocabulary, frameRate: 100)
        var actions = tracker.feed(boundary: NFKMuScriptorChunkBoundary(seekSeconds: 0, nextSeekSeconds: nil))
        for token in expected { actions.append(contentsOf: tracker.feed(token: token)) }
        actions.append(contentsOf: tracker.finish())

        let rows = record["note_actions"]!.asArray(Float.self)
        let rowCount = record["note_actions"]!.dim(0)
        XCTAssertEqual(actions.count, rowCount, "the decode produces the reference's own actions")
        for index in 0 ..< min(actions.count, rowCount) {
            let kind = Int(rows[index * 4])
            let program = Int(rows[index * 4 + 1])
            let pitch = Int(rows[index * 4 + 2])
            let time = Double(rows[index * 4 + 3])
            switch actions[index] {
            case let .start(mineProgram, minePitch, mineTime):
                XCTAssertEqual(kind, 0, "action \(index) should be a start")
                XCTAssertEqual(mineProgram, program)
                XCTAssertEqual(minePitch, pitch)
                XCTAssertEqual(mineTime, time, accuracy: 1e-5)
            case let .end(mineProgram, minePitch, mineTime):
                XCTAssertEqual(kind, 1, "action \(index) should be an end")
                XCTAssertEqual(mineProgram, program)
                XCTAssertEqual(minePitch, pitch)
                XCTAssertEqual(mineTime, time, accuracy: 1e-5)
            case let .drum(minePitch, mineTime):
                XCTAssertEqual(kind, 2, "action \(index) should be a drum hit")
                XCTAssertEqual(minePitch, pitch)
                XCTAssertEqual(mineTime, time, accuracy: 1e-5)
            }
        }

        // And the whole path: audio in, a Standard MIDI File out.
        let sequence = net.transcribe(audio, sampleRate: 16000)
        print("VALIDATION PARITY muscriptor_real: transcribed \(sequence.notes.count) notes end to end")
        XCTAssertEqual(sequence.notes.count, rowCount / 2, "each start/end pair is one note")
        XCTAssertGreaterThan(sequence.standardMIDIFileData().count, 22)
    }

    // MARK: Backend

    func testTheBackendReturnsAMIDISequence() throws {
        try requireMLXRuntime()
        let net = NFKMLXMuScriptor.makeNet(Self.tiny)
        let backend = NFKMLXMuScriptorBackend(net: net, identifier: "muscriptor-test")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("muscriptor-\(UUID()).wav")
        let samples = (0 ..< 16000).map { 0.2 * sinf(2 * .pi * 220 * Float($0) / 16000) }
        try NFKMLXWaveFile.write(samples: samples, sampleRate: 16000, to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let asset = NFKAudioAsset(fileURL: url, durationSeconds: 1, sampleRate: 16000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertNotNil(result.midi)
    }
}
