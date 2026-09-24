//
//  NFKMLXHFTTransformerTests.swift
//  InferKitMLXTests
//
//  hFT-Transformer piano transcription. The weight-free tests cover the module layout and the
//  geometry; the parity tests are gated on the converted weights (`IK_VAL_HFT`) and the recorded
//  oracle (`IK_PARITY_HFT`, from `run_reference.py hft` under the `hft` oracle environment).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXHFTTransformerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    // MARK: Geometry

    func testTheStemWidthFollowsTheContextWindow() {
        let configuration = NFKMLXHFTTransformerConfiguration.maestro
        // 65 frames of context, a 5-wide kernel, 4 channels.
        XCTAssertEqual(configuration.processFrames, 65)
        XCTAssertEqual(configuration.cnnDimension, 244)
        XCTAssertEqual(configuration.frameSeconds, 0.016, accuracy: 1e-9)
        XCTAssertEqual(Double(configuration.padValue), log(1e-8), accuracy: 1e-5)
    }

    func testParameterNamesFollowTheConvertersKeys() throws {
        try requireMLXRuntime()
        let net = NFKMLXHFTTransformer.makeNet()
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["frontend.window", "frontend.filterbank",
                         "encoder.conv.weight", "encoder.tok_embedding_freq.weight",
                         "encoder.pos_embedding_freq.weight",
                         "encoder.layers_freq.0.layer_norm.weight",
                         "encoder.layers_freq.0.self_attention.fc_q.weight",
                         "encoder.layers_freq.2.positionwise_feedforward.fc_1.weight",
                         "decoder.pos_embedding_freq.weight",
                         "decoder.layer_zero_freq.encoder_attention.fc_k.weight",
                         "decoder.layers_freq.0.self_attention.fc_q.weight",
                         "decoder.layers_freq.1.encoder_attention.fc_o.weight",
                         "decoder.fc_onset_freq.weight", "decoder.fc_velocity_freq.weight",
                         "decoder.pos_embedding_time.weight",
                         "decoder.layers_time.2.layer_norm.weight",
                         "decoder.fc_onset_time.weight", "decoder.fc_velocity_time.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testALayerHoldsOneNormForBothResiduals() throws {
        try requireMLXRuntime()
        let net = NFKMLXHFTTransformer.makeNet()
        let names = net.parameters().flattened().map(\.0).filter { $0.hasPrefix("encoder.layers_freq.0.") }
        // The reference applies one LayerNorm instance after both residuals, so a layer has exactly
        // one norm weight and one norm bias rather than two of each.
        XCTAssertEqual(names.filter { $0.contains("layer_norm") }.count, 2)
    }

    func testTheDecoderHasOneFewerCrossAttentionBlockThanLayers() throws {
        try requireMLXRuntime()
        let net = NFKMLXHFTTransformer.makeNet()
        // Three layers: the first reads frequency with no self-attention, then two full blocks.
        XCTAssertEqual(net.decoder.frequencyLayers.count, 2)
        XCTAssertEqual(net.decoder.timeLayers.count, 3)
        XCTAssertEqual(net.encoder.layers.count, 3)
    }

    // MARK: Forward

    func testTheForwardScoresASegmentAtBothLevels() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXHFTTransformerConfiguration.maestro
        let net = NFKMLXHFTTransformer.makeNet(configuration)
        let frames = configuration.processFrames + configuration.segmentFrames - 1
        let segment = MLXArray.zeros([configuration.melBins, frames]) + configuration.padValue

        let output = net(segment)
        eval(output.onset, output.mpe, output.velocity, output.attention)
        XCTAssertEqual(output.onset.shape, [configuration.segmentFrames, configuration.notes])
        XCTAssertEqual(output.onsetFrequency.shape, [configuration.segmentFrames, configuration.notes])
        XCTAssertEqual(output.velocity.shape, [configuration.segmentFrames, configuration.notes,
                                               configuration.velocities])
        XCTAssertEqual(output.attention.shape, [configuration.segmentFrames, configuration.heads,
                                                configuration.notes, configuration.melBins])
        XCTAssertTrue(output.onset.asArray(Float.self).allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    // MARK: Note detection

    func testAPeakIsFoundAndItsTimeRefinedBetweenFrames() {
        let configuration = NFKMLXHFTTransformerConfiguration.maestro
        var activation = [[Float]](repeating: [Float](repeating: 0, count: 88), count: 20)
        // A peak at frame 10 leaning towards frame 11.
        activation[9][39] = 0.6
        activation[10][39] = 0.9
        activation[11][39] = 0.8
        let found = NFKHFTNoteDetection.peaks(activation, note: 39, threshold: 0.5,
                                              hopSeconds: configuration.frameSeconds)
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found.first?.frame, 10)
        // The interpolation moves the time towards the louder neighbor, by less than half a frame.
        let base = 10 * configuration.frameSeconds
        XCTAssertGreaterThan(found.first?.time ?? 0, base)
        XCTAssertLessThan(found.first?.time ?? 0, base + configuration.frameSeconds * 0.5)
    }

    func testAPlateauSurvivesThePeakScan() {
        var activation = [[Float]](repeating: [Float](repeating: 0, count: 88), count: 20)
        // Two equal frames: the reference walks outward until a different value appears, so both are
        // kept. A strict neighbor comparison would drop them.
        activation[10][39] = 0.9
        activation[11][39] = 0.9
        let found = NFKHFTNoteDetection.peaks(activation, note: 39, threshold: 0.5, hopSeconds: 0.016)
        XCTAssertEqual(found.count, 2)
    }

    func testANoteRunsFromItsOnsetToWhereItStopsSounding() {
        let configuration = NFKMLXHFTTransformerConfiguration.maestro
        let frames = 60
        var onset = [[Float]](repeating: [Float](repeating: 0, count: 88), count: frames)
        var offset = onset
        var mpe = onset
        let velocity = [[Int]](repeating: [Int](repeating: 64, count: 88), count: frames)

        onset[10][39] = 0.9                                       // MIDI 60
        for frame in 10 ..< 30 { mpe[frame][39] = 0.9 }           // sounds for 20 frames
        offset[30][39] = 0.8

        let notes = NFKHFTNoteDetection.notes(onset: onset, offset: offset, mpe: mpe, velocity: velocity,
                                              configuration: configuration, options: .default)
        XCTAssertEqual(notes.count, 1)
        XCTAssertEqual(notes.first?.pitch, 60)
        XCTAssertEqual(notes.first?.velocity, 64)
        XCTAssertEqual(notes.first?.startSeconds ?? 0, 10 * configuration.frameSeconds, accuracy: 1e-9)
        XCTAssertEqual(notes.first?.endSeconds ?? 0, 30 * configuration.frameSeconds, accuracy: 1e-3)
    }

    func testASilentNoteIsDroppedUnlessAskedFor() {
        let configuration = NFKMLXHFTTransformerConfiguration.maestro
        let frames = 40
        var onset = [[Float]](repeating: [Float](repeating: 0, count: 88), count: frames)
        var mpe = onset
        let offset = onset
        let velocity = [[Int]](repeating: [Int](repeating: 0, count: 88), count: frames)
        onset[10][39] = 0.9
        for frame in 10 ..< 20 { mpe[frame][39] = 0.9 }

        var options = NFKMLXHFTTransformerOptions.default
        XCTAssertEqual(NFKHFTNoteDetection.notes(onset: onset, offset: offset, mpe: mpe,
                                                 velocity: velocity, configuration: configuration,
                                                 options: options).count, 0)
        options.dropsSilentNotes = false
        XCTAssertEqual(NFKHFTNoteDetection.notes(onset: onset, offset: offset, mpe: mpe,
                                                 velocity: velocity, configuration: configuration,
                                                 options: options).count, 1)
    }

    // MARK: Backend

    func testTheBackendReturnsAMIDISequence() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXHFTTransformer.backend(weightsURL: nil)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("hft-\(UUID()).wav")
        let samples = (0 ..< 16000).map { 0.2 * sinf(2 * .pi * 261.63 * Float($0) / 16000) }
        try NFKMLXWaveFile.write(samples: samples, sampleRate: 16000, to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let asset = NFKAudioAsset(fileURL: url, durationSeconds: 1, sampleRate: 16000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertNotNil(result.midi)
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

    private func parity() throws -> (net: NFKMLXHFTTransformerNet, record: [String: MLXArray]) {
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_HFT"],
              let recordPath = environment["IK_PARITY_HFT"] else {
            throw XCTSkip("set IK_VAL_HFT (converted weights) and IK_PARITY_HFT (oracle record)")
        }
        let net = NFKMLXHFTTransformer.makeNet()
        try NFKMLXHFTTransformer.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        return (net, record)
    }

    func testTheFrontEndMatchesTorchaudio() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let audio = record["audio"]!.asArray(Float.self)
        let feature = net.frontEnd(audio)
        eval(feature)

        XCTAssertEqual(feature.shape, record["feature"]!.shape)
        let value = cosine(feature.asArray(Float.self), record["feature"]!.asArray(Float.self))
        print("VALIDATION PARITY hft: log-mel cosine \(value)")
        // Float32 over 251 frames of 256 bins: the measured value sits at the last representable
        // step below one.
        XCTAssertGreaterThan(value, 0.999999, "the front end diverges")
    }

    /// Localizes a divergence: the encoder, then both output levels, on the reference's own segment.
    func testTheSeamsMatchTheReference() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let segment = record["segment"]!                                   // [bins, 192]

        let encoded = net.encoder(segment)
        eval(encoded)
        XCTAssertEqual(encoded.shape, record["encoded"]!.shape)
        let encoderValue = cosine(encoded.asArray(Float.self), record["encoded"]!.asArray(Float.self))
        print("VALIDATION PARITY hft: encoder cosine \(encoderValue)")
        XCTAssertGreaterThan(encoderValue, 0.9999, "the frequency encoder diverges")

        let output = net.decoder(encoded)
        eval(output.onset, output.offset, output.mpe, output.velocity, output.attention)
        for (name, mine, reference) in [("onset (frequency)", output.onsetFrequency, record["onset_freq"]!),
                                        ("offset (frequency)", output.offsetFrequency, record["offset_freq"]!),
                                        ("mpe (frequency)", output.mpeFrequency, record["mpe_freq"]!),
                                        ("velocity (frequency)", output.velocityFrequency, record["velocity_freq"]!),
                                        ("attention", output.attention, record["attention"]!),
                                        ("onset (time)", output.onset, record["output"]!),
                                        ("offset (time)", output.offset, record["offset_time"]!),
                                        ("mpe (time)", output.mpe, record["mpe_time"]!),
                                        ("velocity (time)", output.velocity, record["velocity_time"]!)] {
            let value = cosine(mine.asArray(Float.self), reference.asArray(Float.self))
            print("VALIDATION PARITY hft: \(name) cosine \(value)")
            XCTAssertGreaterThan(value, 0.9999, "\(name) diverges")
        }
    }

    /// The whole clip: the stitched posteriorgrams, and the notes read from them.
    func testTheTranscriptionMatchesTheReference() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let audio = record["audio"]!.asArray(Float.self)

        let grams = net.posteriorgrams(audio, sampleRate: 16000)
        XCTAssertEqual(grams.onset.count, record["clip_onset"]!.dim(0), "the same frame count")
        let onsetValue = cosine(grams.onset.flatMap { $0 }, record["clip_onset"]!.asArray(Float.self))
        let mpeValue = cosine(grams.mpe.flatMap { $0 }, record["clip_mpe"]!.asArray(Float.self))
        print("VALIDATION PARITY hft: clip onset cosine \(onsetValue), mpe cosine \(mpeValue)")
        XCTAssertGreaterThan(onsetValue, 0.9999)
        XCTAssertGreaterThan(mpeValue, 0.9999)

        // The velocity head is an argmax, so it agrees exactly or it does not agree at all.
        let referenceVelocity = record["clip_velocity"]!.asArray(Int32.self).map(Int.init)
        let mineVelocity = grams.velocity.flatMap { $0 }
        let agreeing = zip(mineVelocity, referenceVelocity).filter { $0 == $1 }.count
        print("VALIDATION PARITY hft: velocity argmax \(agreeing)/\(referenceVelocity.count)")
        XCTAssertGreaterThan(Double(agreeing) / Double(referenceVelocity.count), 0.999)

        let sequence = net.transcribe(audio, sampleRate: 16000)
        let rows = record["notes"]!.asArray(Float.self)
        let noteCount = record["notes"]!.dim(0)
        XCTAssertEqual(sequence.notes.count, noteCount, "the same notes are found")
        for index in 0 ..< min(sequence.notes.count, noteCount) {
            let note = sequence.notes[index]
            XCTAssertEqual(note.pitch, Int(rows[index * 4]), "note \(index) pitch")
            XCTAssertEqual(note.startSeconds, Double(rows[index * 4 + 1]), accuracy: 1e-4, "note \(index) onset")
            XCTAssertEqual(note.endSeconds, Double(rows[index * 4 + 2]), accuracy: 1e-4, "note \(index) offset")
            XCTAssertEqual(note.velocity, Int(rows[index * 4 + 3]), "note \(index) velocity")
        }
        print("VALIDATION PARITY hft: \(sequence.notes.count) notes match the reference")
    }
}
