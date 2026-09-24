//
//  NFKMLXBasicPitchTests.swift
//  InferKitMLXTests
//
//  Basic Pitch music transcription. The weight-free tests cover the module layout, the CQT's shape
//  contract, the windowing, and note creation; the parity tests are gated on the converted weights
//  (`IK_VAL_BASIC_PITCH`) and the recorded oracle (`IK_PARITY_BASIC_PITCH`, from
//  `run_reference.py basic_pitch` under the `bpvenv` interpreter).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXBasicPitchTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private static func chord(samples: Int, sampleRate: Float = 22050) -> [Float] {
        (0 ..< samples).map { index in
            let t = Float(index) / sampleRate
            return 0.3 * (sinf(2 * .pi * 261.63 * t) + sinf(2 * .pi * 329.63 * t) + sinf(2 * .pi * 392.0 * t)) / 3
        }
    }

    // MARK: Module layout

    func testParameterNamesMatchTheConvertersKeys() throws {
        try requireMLXRuntime()
        let net = NFKMLXBasicPitch.makeNet()
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["cqt.kernel_a", "cqt.kernel_b", "cqt.kernel_bias", "cqt.lowpass",
                         "cqt.lowpass_bias", "cqt.scale", "norm_scale", "norm_bias",
                         "contour_conv.weight", "contour_conv.bias", "contour_out.weight",
                         "note_conv.weight", "note_out.weight", "onset_conv.weight", "onset_out.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheHarmonicShiftsAreTheReferences() {
        let configuration = NFKMLXBasicPitchConfiguration.icassp2022
        let shifts = configuration.harmonics.map { configuration.shift(forHarmonic: $0) }
        XCTAssertEqual(shifts, [-36, 0, 36, 57, 72, 84, 93, 101])
    }

    // MARK: Front end

    func testTheTransformLandsEveryOctaveOnTheSameFrameGrid() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXBasicPitchConfiguration.icassp2022
        let net = NFKMLXBasicPitch.makeNet(configuration)
        let audio = MLXArray(Self.chord(samples: configuration.windowSamples))
            .reshaped([1, configuration.windowSamples, 1])
        let magnitude = net.cqt(audio)
        eval(magnitude)
        XCTAssertEqual(magnitude.shape, [1, configuration.frames, configuration.cqtBins])
    }

    func testTheNormalizedLogIsZeroToOneBeforeTheFoldedNormalization() throws {
        try requireMLXRuntime()
        let net = NFKMLXBasicPitch.makeNet()
        // With the identity normalization the released model folds in, the log is normalized to 0...1.
        net.update(parameters: ModuleParameters.unflattened([("norm_scale", MLXArray([Float(1)])),
                                                             ("norm_bias", MLXArray([Float(0)]))]))
        let magnitude = MLXArray((0 ..< (172 * 309)).map { Float($0 % 97) + 0.5 }).reshaped([1, 172, 309])
        let normalized = net.normalizedLog(magnitude)
        eval(normalized)
        let values = normalized.asArray(Float.self)
        XCTAssertEqual(values.min() ?? -1, 0, accuracy: 1e-6)
        XCTAssertEqual(values.max() ?? -1, 1, accuracy: 1e-6)
    }

    func testASilentWindowNormalizesToZeroRatherThanNaN() throws {
        try requireMLXRuntime()
        let net = NFKMLXBasicPitch.makeNet()
        net.update(parameters: ModuleParameters.unflattened([("norm_scale", MLXArray([Float(1)])),
                                                             ("norm_bias", MLXArray([Float(0)]))]))
        let normalized = net.normalizedLog(MLXArray.zeros([1, 8, 309]))
        eval(normalized)
        XCTAssertTrue(normalized.asArray(Float.self).allSatisfy { $0 == 0 })
    }

    func testHarmonicStackingShiftsEachHarmonicIntoItsOwnChannel() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXBasicPitchConfiguration.icassp2022
        let net = NFKMLXBasicPitch.makeNet(configuration)
        // One bin lit, at a bin every harmonic can reach.
        let lit = 150
        var plate = [Float](repeating: 0, count: configuration.cqtBins)
        plate[lit] = 1
        let spectrogram = MLXArray(plate).reshaped([1, 1, configuration.cqtBins, 1])
        let stacked = net.harmonicStack(spectrogram)
        eval(stacked)
        XCTAssertEqual(stacked.shape, [1, 1, configuration.contourBins, configuration.harmonics.count])

        let values = stacked.reshaped([configuration.contourBins, configuration.harmonics.count]).asArray(Float.self)
        for (channel, harmonic) in configuration.harmonics.enumerated() {
            let shift = configuration.shift(forHarmonic: harmonic)
            let bin = lit - shift
            guard bin >= 0, bin < configuration.contourBins else { continue }
            XCTAssertEqual(values[bin * configuration.harmonics.count + channel], 1,
                           "harmonic \(harmonic) should read bin \(lit) at bin \(bin)")
        }
    }

    // MARK: Windowing

    func testTheWindowingCoversTheClipWithTheReferencesOverlap() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXBasicPitchConfiguration.icassp2022
        let net = NFKMLXBasicPitch.makeNet(configuration)
        let windows = net.windows([Float](repeating: 0.1, count: 88200))
        eval(windows)
        XCTAssertEqual(windows.shape, [3, configuration.windowSamples, 1])
        XCTAssertEqual(configuration.windowHop, 36164)
    }

    func testTheFrameTimesStepByOneHopInsideAWindow() {
        let net = NFKMLXBasicPitch.makeNet()
        let times = net.frameTimes(count: 400)
        let hop = 256.0 / 22050.0
        XCTAssertEqual(times[0], 0, accuracy: 1e-12)
        XCTAssertEqual(times[10] - times[9], hop, accuracy: 1e-12)
        // The reference subtracts one window offset per 172 frames, so the seam is not a plain hop.
        XCTAssertLessThan(times[172] - times[171], hop)
    }

    // MARK: Note creation

    func testNoteCreationFindsAHeldNoteWithItsOnsetAndLength() {
        let configuration = NFKMLXBasicPitchConfiguration.icassp2022
        let frames = 200
        let bin = 39                                     // MIDI 60, middle C
        var note = [[Float]](repeating: [Float](repeating: 0.01, count: configuration.noteBins), count: frames)
        var onset = note
        for frame in 40 ..< 120 { note[frame][bin] = 0.9 }
        onset[40][bin] = 0.95

        let events = NFKBasicPitchNoteCreation.notes(note: note, onset: onset,
                                                     configuration: configuration, options: .default)
        XCTAssertEqual(events.count, 1)
        let event = events.first
        XCTAssertEqual(event?.pitch, 60)
        XCTAssertEqual(event?.startFrame, 40)
        // The note ends where its energy dies, which the tracker finds one frame past the last.
        XCTAssertEqual(event?.endFrame ?? 0, 120, accuracy: 1)
        XCTAssertEqual(Double(event?.amplitude ?? 0), 0.9, accuracy: 0.02)
    }

    func testANoteShorterThanTheMinimumIsDropped() {
        let configuration = NFKMLXBasicPitchConfiguration.icassp2022
        let frames = 60
        let bin = 39
        var note = [[Float]](repeating: [Float](repeating: 0.0, count: configuration.noteBins), count: frames)
        var onset = note
        for frame in 20 ..< 25 { note[frame][bin] = 0.9 }
        onset[20][bin] = 0.95

        var options = NFKMLXBasicPitchOptions.default
        options.melodiaTrick = false
        options.infersOnsets = false
        XCTAssertEqual(NFKBasicPitchNoteCreation.notes(note: note, onset: onset,
                                                       configuration: configuration, options: options).count, 0)
    }

    func testAFrequencyFloorSilencesTheBinsBelowIt() {
        let configuration = NFKMLXBasicPitchConfiguration.icassp2022
        let frames = 200
        var note = [[Float]](repeating: [Float](repeating: 0.0, count: configuration.noteBins), count: frames)
        var onset = note
        for frame in 40 ..< 120 { note[frame][10] = 0.9 }      // MIDI 31, about 49 Hz
        onset[40][10] = 0.95

        var options = NFKMLXBasicPitchOptions.default
        options.minimumFrequency = 200
        XCTAssertEqual(NFKBasicPitchNoteCreation.notes(note: note, onset: onset,
                                                       configuration: configuration, options: options).count, 0)
        options.minimumFrequency = nil
        XCTAssertEqual(NFKBasicPitchNoteCreation.notes(note: note, onset: onset,
                                                       configuration: configuration, options: options).count, 1)
    }

    func testOverlappingNotesLoseTheirPitchBends() {
        var first = NFKBasicPitchNoteEvent(startFrame: 0, endFrame: 50, pitch: 60, amplitude: 0.5, bendBins: [0, 1])
        var second = NFKBasicPitchNoteEvent(startFrame: 20, endFrame: 70, pitch: 64, amplitude: 0.5, bendBins: [0, 1])
        let apart = NFKBasicPitchNoteEvent(startFrame: 80, endFrame: 90, pitch: 67, amplitude: 0.5, bendBins: [0, 1])
        let kept = NFKBasicPitchNoteCreation.dropOverlappingBends([first, second, apart])
        XCTAssertNil(kept[0].bendBins)
        XCTAssertNil(kept[1].bendBins)
        XCTAssertNotNil(kept[2].bendBins)
        first.bendBins = nil
        second.bendBins = nil
    }

    // MARK: Backend

    func testTheBackendReturnsAMIDISequence() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXBasicPitch.backend(weightsURL: nil)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("basic-pitch-\(UUID()).wav")
        try NFKMLXWaveFile.write(samples: Self.chord(samples: 44100), sampleRate: 22050, to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let asset = NFKAudioAsset(fileURL: url, durationSeconds: 2, sampleRate: 22050, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        let sequence = try XCTUnwrap(result.midi)
        XCTAssertEqual(sequence.tempoBPM, 120, accuracy: 1e-9)
        XCTAssertGreaterThan(sequence.standardMIDIFileData().count, 22)
    }

    func testTheBackendReadsItsParameterKeys() throws {
        try requireMLXRuntime()
        let request = NFKInferenceRequest(inputs: [:], parameters: [
            NFKMLXTranscriptionParameterKey.onsetThreshold: 0.7,
            NFKMLXTranscriptionParameterKey.frameThreshold: 0.2,
            NFKMLXTranscriptionParameterKey.minimumNoteFrames: 20,
            NFKMLXTranscriptionParameterKey.pitchBends: false,
            NFKMLXTranscriptionParameterKey.tempo: 90.0,
            NFKMLXTranscriptionParameterKey.program: 40,
        ])
        let options = NFKMLXBasicPitchBackend.options(from: request)
        XCTAssertEqual(options.onsetThreshold, 0.7, accuracy: 1e-6)
        XCTAssertEqual(options.frameThreshold, 0.2, accuracy: 1e-6)
        XCTAssertEqual(options.minimumNoteFrames, 20)
        XCTAssertFalse(options.includesPitchBends)
        XCTAssertEqual(options.tempoBPM, 90, accuracy: 1e-9)
        XCTAssertEqual(options.program, 40)
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

    private func parityRecord() throws -> (net: NFKMLXBasicPitchNet, record: [String: MLXArray]) {
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_BASIC_PITCH"],
              let recordPath = environment["IK_PARITY_BASIC_PITCH"] else {
            throw XCTSkip("set IK_VAL_BASIC_PITCH (converted weights) and IK_PARITY_BASIC_PITCH (oracle record)")
        }
        let net = NFKMLXBasicPitch.makeNet()
        try NFKMLXBasicPitch.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        return (net, record)
    }

    /// Localizes a divergence: the front end and the three heads on the reference's own windows.
    func testTheSeamsMatchTheReference() throws {
        try requireMLXRuntime()
        let (net, record) = try parityRecord()
        let windows = record["windows"]!                                   // [W, samples]
        let audio = windows.expandedDimensions(axis: 2)

        let magnitude = net.cqt(audio)
        let cqtCosine = cosine(magnitude.asArray(Float.self), record["cqt"]!.asArray(Float.self))
        print("VALIDATION PARITY basic_pitch: cqt cosine \(cqtCosine)")
        XCTAssertGreaterThan(cqtCosine, 0.9999, "the constant-Q front end diverges")

        let normalized = net.normalizedLog(magnitude)
        let logCosine = cosine(normalized.asArray(Float.self), record["logspec"]!.asArray(Float.self))
        print("VALIDATION PARITY basic_pitch: normalized log cosine \(logCosine)")
        XCTAssertGreaterThan(logCosine, 0.9999, "the normalized log diverges")

        let stacked = net.harmonicStack(normalized)
        let stackCosine = cosine(stacked.asArray(Float.self), record["stack"]!.asArray(Float.self))
        print("VALIDATION PARITY basic_pitch: harmonic stack cosine \(stackCosine)")
        XCTAssertGreaterThan(stackCosine, 0.9999, "harmonic stacking diverges")

        let (contour, note, onset) = net.posteriorgrams(audio)
        eval(contour, note, onset)
        for (name, mine, reference) in [("contour", contour, record["window_contour"]!),
                                        ("note", note, record["window_note"]!),
                                        ("onset", onset, record["window_onset"]!)] {
            let value = cosine(mine.asArray(Float.self), reference.asArray(Float.self))
            print("VALIDATION PARITY basic_pitch: \(name) cosine \(value)")
            XCTAssertGreaterThan(value, 0.9999, "the \(name) head diverges")
        }
    }

    /// The whole path on the reference's own clip: stitching, note creation, and the notes themselves.
    func testTheTranscriptionMatchesTheReference() throws {
        try requireMLXRuntime()
        let (net, record) = try parityRecord()
        let waveform = record["waveform"]!.asArray(Float.self)

        let (contour, note, onset) = net.unwrapped(net.windows(waveform), sampleCount: waveform.count)
        XCTAssertEqual(contour.count, record["contour"]!.dim(0), "the stitched posteriorgram is the reference's length")
        let stitched = cosine(note.flatMap { $0 }, record["note"]!.asArray(Float.self))
        print("VALIDATION PARITY basic_pitch: stitched note cosine \(stitched)")
        XCTAssertGreaterThan(stitched, 0.9999)
        let stitchedOnset = cosine(onset.flatMap { $0 }, record["output"]!.asArray(Float.self))
        XCTAssertGreaterThan(stitchedOnset, 0.9999)

        let sequence = net.transcribe(waveform, sampleRate: 22050)
        let reference = record["notes"]!.asArray(Float.self)               // [N, 4]
        let referenceCount = record["notes"]!.dim(0)
        XCTAssertEqual(sequence.notes.count, referenceCount, "the same notes are found")

        // The reference's note list is ordered by how it found them; the sequence sorts by time.
        var expected = (0 ..< referenceCount).map { index -> (Double, Double, Int) in
            (Double(reference[index * 4]), Double(reference[index * 4 + 1]), Int(reference[index * 4 + 2]))
        }
        expected.sort { ($0.0, Double($0.2)) < ($1.0, Double($1.2)) }
        for (found, want) in zip(sequence.notes, expected) {
            XCTAssertEqual(found.pitch, want.2)
            XCTAssertEqual(found.startSeconds, want.0, accuracy: 1e-6)
            XCTAssertEqual(found.endSeconds, want.1, accuracy: 1e-6)
        }

        // The record holds the bends note creation reads, before the MIDI writer drops the ones that
        // overlap, so the comparison is against that stage: same count, same length, same bins.
        let events = NFKBasicPitchNoteCreation.pitchBends(contour,
                                                          events: NFKBasicPitchNoteCreation.notes(note: note, onset: onset,
                                                                                                  configuration: net.configuration,
                                                                                                  options: .default),
                                                          configuration: net.configuration)
        let times = net.frameTimes(count: contour.count)
        let mine = events.map { event in
            (start: times[event.startFrame], pitch: event.pitch, bends: event.bendBins ?? [])
        }.sorted { ($0.start, Double($0.pitch)) < ($1.start, Double($1.pitch)) }

        let lengths = record["bend_lengths"]!.asArray(Int32.self).map(Int.init)
        let values = record["bend_values"]!.asArray(Int32.self).map(Int.init)
        var offset = 0
        var theirs = (0 ..< referenceCount).map { index -> (start: Double, pitch: Int, bends: [Int]) in
            let length = lengths[index]
            let slice = Array(values[offset ..< (offset + length)])
            offset += length
            return (Double(reference[index * 4]), Int(reference[index * 4 + 2]), slice)
        }
        theirs.sort { ($0.start, Double($0.pitch)) < ($1.start, Double($1.pitch)) }

        XCTAssertEqual(mine.count, theirs.count)
        for (found, want) in zip(mine, theirs) {
            XCTAssertEqual(found.pitch, want.pitch)
            XCTAssertEqual(found.bends, want.bends, "the pitch-bend curve of MIDI \(want.pitch) differs")
        }

        // What the writer keeps: a note whose span no other note touches.
        let written = sequence.notes.filter { $0.pitchBend != nil }.count
        XCTAssertLessThanOrEqual(written, mine.filter { !$0.bends.isEmpty }.count)
    }
}
