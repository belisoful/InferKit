//
//  NFKMLXAllInOneTests.swift
//  InferKitMLXTests
//
//  All-In-One music structure analysis. The weight-free tests cover the module layout, the
//  neighborhood-attention index rules, the front end, the structure post-processing, and the bar
//  tracker; the parity tests are gated on the converted weights (`IK_VAL_ALLIN1`) and the recorded
//  oracle (`IK_PARITY_ALLIN1`, from `run_reference.py allin1` under the `allin1venv` interpreter).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXAllInOneTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private static func stems(seconds: Double, sampleRate: Int = 44100) -> [[Float]] {
        let count = Int(seconds * Double(sampleRate))
        return (0 ..< 4).map { stem in
            (0 ..< count).map { index in
                let t = Float(index) / Float(sampleRate)
                let pulse = (index % (sampleRate / 2)) < 2000 ? Float(1) : Float(0.1)
                return 0.2 * pulse * sinf(2 * .pi * Float(110 * (stem + 1)) * t)
            }
        }
    }

    // MARK: Module layout

    func testParameterNamesFollowTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        let net = NFKMLXAllInOne.makeNet()
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["frontend.filterbank",
                         "embeddings.conv0.weight", "embeddings.conv1.weight", "embeddings.conv2.weight",
                         "embeddings.norm.weight",
                         "encoder.layers.0.timelayer.layernorm_before.weight",
                         "encoder.layers.0.timelayer.attention.self.rpb",
                         "encoder.layers.0.timelayer.attention.self.query.weight",
                         "encoder.layers.0.timelayer.attention.output.dense.weight",
                         "encoder.layers.0.timelayer.attention2.self.rpb",
                         "encoder.layers.0.timelayer.intermediate.dense.weight",
                         "encoder.layers.0.timelayer.output.dense.weight",
                         "encoder.layers.0.instlayer.attention.self.rpb",
                         "encoder.layers.10.timelayer.layernorm_after.weight",
                         "norm.weight", "beat_classifier.classifier.weight",
                         "downbeat_classifier.classifier.weight", "section_classifier.classifier.weight",
                         "function_classifier.classifier.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTheInstrumentAttentionBiasIsTwoDimensional() throws {
        try requireMLXRuntime()
        let net = NFKMLXAllInOne.makeNet()
        let time = net.encoder.layers[0].timeLayer.attention.attention.rpb
        let instrument = net.encoder.layers[0].instrumentLayer.attention.attention.rpb
        XCTAssertEqual(time.shape, [2, 9])
        XCTAssertEqual(instrument.shape, [2, 9, 9])
    }

    func testDilationDoublesPerBlock() {
        let configuration = NFKMLXAllInOneConfiguration.harmonix
        XCTAssertEqual((0 ..< configuration.depth).map { configuration.dilation(atBlock: $0) },
                       [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    }

    // MARK: Neighborhood attention

    func testTheNeighborhoodIsCenteredAndSlidesInwardAtTheEdges() {
        let window = NFKNeighborhoodWindow(length: 10, kernel: 5, dilation: 1)
        // A query in the middle reads the two positions either side of itself.
        XCTAssertEqual(window.start[5], 3)
        XCTAssertEqual(window.biasStart[5], 2)
        // At the edges the window stays inside the sequence and the bias index shifts with it.
        XCTAssertEqual(window.start[0], 0)
        XCTAssertEqual(window.biasStart[0], 4)
        XCTAssertEqual(window.start[9], 5)
        XCTAssertEqual(window.biasStart[9], 0)
    }

    func testEveryQueryReadsExactlyKernelPositionsInsideTheSequence() {
        for length in [12, 20, 41] {
            for dilation in [1, 2, 4] where length >= 5 * dilation {
                let window = NFKNeighborhoodWindow(length: length, kernel: 5, dilation: dilation)
                for index in 0 ..< length {
                    for step in 0 ..< 5 {
                        let neighbor = window.start[index] + step * dilation
                        XCTAssertGreaterThanOrEqual(neighbor, 0, "length \(length) dilation \(dilation) index \(index)")
                        XCTAssertLessThan(neighbor, length, "length \(length) dilation \(dilation) index \(index)")
                        let bias = window.biasStart[index] + step
                        XCTAssertGreaterThanOrEqual(bias, 0)
                        XCTAssertLessThan(bias, 9)
                    }
                }
            }
        }
    }

    func testAUniformValueSurvivesAttention() throws {
        try requireMLXRuntime()
        // Attention is a convex combination, so a constant input comes back constant whatever the
        // scores are: the softmax weights sum to one over exactly the kernel positions.
        let attention = NFKAllInOneAttention(dimension: 8, heads: 2, kernelSize: 5, dilation: 1,
                                             twoDimensional: false)
        attention.update(parameters: ModuleParameters.unflattened([
            ("value.weight", MLXArray.eye(8)), ("value.bias", MLXArray.zeros([8])),
        ]))
        let hidden = MLXArray.ones([1, 20, 8]) * 3.5
        let out = attention(hidden)
        eval(out)
        for value in out.asArray(Float.self) {
            XCTAssertEqual(value, 3.5, accuracy: 1e-4)
        }
    }

    // MARK: Front end

    func testTheFrontEndProducesOneFrameEveryHop() throws {
        try requireMLXRuntime()
        let net = NFKMLXAllInOne.makeNet()
        XCTAssertEqual(net.frontEnd.frameCount(samples: 44100), 100)
        XCTAssertEqual(net.frontEnd.frameCount(samples: 88200), 200)
        XCTAssertEqual(net.frontEnd.frameCount(samples: 88201), 201)
        XCTAssertEqual(net.frontEnd.frameCount(samples: 1000), 3)
    }

    func testTheSpectrogramStackIsOnePlanePerInstrument() throws {
        try requireMLXRuntime()
        let net = NFKMLXAllInOne.makeNet()
        let stacked = net.spectrograms(stems: Self.stems(seconds: 1))
        eval(stacked)
        XCTAssertEqual(stacked.shape, [1, 4, 100, 81])
    }

    // MARK: Forward

    func testTheForwardScoresEveryFrame() throws {
        try requireMLXRuntime()
        let net = NFKMLXAllInOne.makeNet()
        let logits = net(net.spectrograms(stems: Self.stems(seconds: 2)))
        eval(logits.beat, logits.downbeat, logits.section, logits.function)
        XCTAssertEqual(logits.beat.shape, [1, 200])
        XCTAssertEqual(logits.downbeat.shape, [1, 200])
        XCTAssertEqual(logits.section.shape, [1, 200])
        XCTAssertEqual(logits.function.shape, [1, 10, 200])
        XCTAssertTrue(logits.beat.asArray(Float.self).allSatisfy { $0.isFinite })
        XCTAssertTrue(logits.function.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    // MARK: Structure post-processing

    func testPeakPickingKeepsThePeakThatStandsAboveItsNeighborhood() {
        var activation = [Float](repeating: 0, count: 600)
        activation[300] = 0.9
        let peaks = NFKAllInOneStructure.localMaxima(activation, filterSize: 97)
        XCTAssertEqual(peaks[300], 0.9)
        let strength = NFKAllInOneStructure.peakPicking(peaks, past: 200, future: 200)
        XCTAssertGreaterThan(strength[300], 0)
        XCTAssertEqual(strength.filter { $0 > 0 }.count, 1)
    }

    func testTheSectionsCoverTheTrackAndTakeTheirWinningLabel() {
        let configuration = NFKMLXAllInOneConfiguration.harmonix
        let frames = 3000
        // The activation is zero away from the boundary: the reference's peak picking keeps only
        // candidates above zero, and a constant baseline would make every frame tie for its window's
        // maximum.
        var section = [Float](repeating: 0, count: frames)
        section[1500] = 0.95
        var function = [[Float]](repeating: [Float](repeating: 0.01, count: frames), count: configuration.labels.count)
        let verse = configuration.labels.firstIndex(of: "verse")!
        let chorus = configuration.labels.firstIndex(of: "chorus")!
        for frame in 0 ..< 1500 { function[verse][frame] = 0.9 }
        for frame in 1500 ..< frames { function[chorus][frame] = 0.9 }

        let sections = NFKAllInOneStructure.sections(sectionProbabilities: section,
                                                     functionProbabilities: function,
                                                     configuration: configuration)
        XCTAssertEqual(sections.count, 2)
        XCTAssertEqual(sections[0].label, "verse")
        XCTAssertEqual(sections[1].label, "chorus")
        XCTAssertEqual(sections[0].startSeconds, 0, accuracy: 1e-9)
        XCTAssertEqual(sections[0].endSeconds, 15.0, accuracy: 1e-6)
        XCTAssertEqual(sections[1].endSeconds, 30.0, accuracy: 1e-6)
    }

    // MARK: Bar tracking

    func testTheTempoGridSamplesWholeFrameIntervals() {
        let intervals = NFKBarStateSpace.intervals(minimum: 60.0 * 100 / 215, maximum: 60.0 * 100 / 55, count: 60)
        XCTAssertGreaterThanOrEqual(intervals.count, 60)
        XCTAssertEqual(intervals.first, 28)
        XCTAssertEqual(intervals.last, 109)
        XCTAssertEqual(intervals, intervals.sorted())
        XCTAssertEqual(Set(intervals).count, intervals.count)
    }

    func testTheStateSpaceWalksEachBeatOncePerInterval() {
        let space = NFKBarStateSpace(beats: 4, minimumInterval: 40, maximumInterval: 44, tempoCount: 60)
        XCTAssertEqual(space.intervals, [40, 41, 42, 43, 44])
        XCTAssertEqual(space.stateCount, 4 * (40 + 41 + 42 + 43 + 44))
        XCTAssertEqual(space.positions[0], 0)
        XCTAssertEqual(space.positions[space.firstStates[2][0]], 2)
        // A state's position walks from its beat to the next one, never reaching it.
        XCTAssertLessThan(space.positions[space.lastStates[0][0]], 1)
    }

    func testTheTrackerFindsAPulseTrainsBeats() {
        // A beat every 50 frames (120 BPM at 100 frames per second), every fourth one a downbeat.
        let frames = 2000
        var beat = [Float](repeating: 0.01, count: frames)
        var downbeat = [Float](repeating: 0.005, count: frames)
        for frame in stride(from: 100, to: frames - 100, by: 50) {
            beat[frame] = 0.95
            if (frame / 50) % 4 == 0 { downbeat[frame] = 0.9 }
        }

        let tracker = NFKMLXBarTracker()
        let beats = tracker.track(beat: beat, downbeat: downbeat)
        XCTAssertGreaterThan(beats.count, 30)
        let intervals = zip(beats.dropFirst(), beats).map { $0.timeSeconds - $1.timeSeconds }
        let median = intervals.sorted()[intervals.count / 2]
        XCTAssertEqual(median, 0.5, accuracy: 0.02, "the tracker follows the pulse's own period")
        XCTAssertEqual(NFKAllInOneStructure.tempo(beats: beats) ?? 0, 120, accuracy: 1)
        XCTAssertTrue(beats.contains { $0.isDownbeat }, "the meter is tracked, not just the beats")
    }

    // MARK: Backend

    func testTheBackendTakesTheFourStems() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXAllInOne.backend(weightsURL: nil)
        XCTAssertFalse((backend as! NFKMLXAllInOneBackend).separatesMixtures)

        var urls = [URL]()
        var assets = [NFKAudioAsset]()
        for (index, stem) in Self.stems(seconds: 3).enumerated() {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("allin1-\(index)-\(UUID()).wav")
            try NFKMLXWaveFile.write(samples: stem, sampleRate: 44100, to: url)
            urls.append(url)
            assets.append(NFKAudioAsset(fileURL: url, durationSeconds: 3, sampleRate: 44100, channelCount: 1))
        }
        defer { urls.forEach { try? FileManager.default.removeItem(at: $0) } }

        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: assets]))
        XCTAssertNotNil(result.segments)
        XCTAssertNotNil(result.beats)
    }

    func testAMixtureWithoutASeparatorIsRefused() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXAllInOne.backend(weightsURL: nil)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("allin1-mix-\(UUID()).wav")
        try NFKMLXWaveFile.write(samples: Self.stems(seconds: 1)[0], sampleRate: 44100, to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: 1, sampleRate: 44100, channelCount: 1)
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset])))
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

    private func parity() throws -> (net: NFKMLXAllInOneNet, record: [String: MLXArray]) {
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_ALLIN1"],
              let recordPath = environment["IK_PARITY_ALLIN1"] else {
            throw XCTSkip("set IK_VAL_ALLIN1 (converted weights) and IK_PARITY_ALLIN1 (oracle record)")
        }
        let net = NFKMLXAllInOne.makeNet()
        try NFKMLXAllInOne.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        return (net, record)
    }

    /// The front end alone: madmom's filtered logarithmic spectrogram, from the recorded stems.
    func testTheFrontEndMatchesMadmom() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let stems = record["stems"]!                                        // [instruments, samples]
        let mine = net.spectrograms(stems: (0 ..< stems.dim(0)).map { stems[$0].asArray(Float.self) })
        eval(mine)
        XCTAssertEqual(mine.shape, [1] + record["spectrograms"]!.shape)
        let value = cosine(mine.asArray(Float.self), record["spectrograms"]!.asArray(Float.self))
        print("VALIDATION PARITY allin1: spectrogram cosine \(value)")
        XCTAssertGreaterThan(value, 0.9999999, "the madmom front end diverges")
    }

    /// Localizes a divergence: the embedding and every block, on the reference's own spectrograms.
    func testTheSeamsMatchTheReference() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let spectrograms = record["spectrograms"]!                          // [instruments, frames, bands]
        let (instruments, frames, bands) = (spectrograms.dim(0), spectrograms.dim(1), spectrograms.dim(2))

        var hidden = net.embeddings(spectrograms.reshaped([instruments, frames, bands, 1]))
        eval(hidden)
        let embedding = cosine(hidden.asArray(Float.self), record["embeddings"]!.asArray(Float.self))
        print("VALIDATION PARITY allin1: embedding cosine \(embedding)")
        XCTAssertGreaterThan(embedding, 0.9999, "the convolutional embedding diverges")

        for (index, layer) in net.encoder.layers.enumerated() {
            hidden = layer(hidden)
            guard let reference = record["block\(index)"] else { continue }
            eval(hidden)
            let value = cosine(hidden.asArray(Float.self), reference.asArray(Float.self))
            print("VALIDATION PARITY allin1: block \(index) cosine \(value)")
            XCTAssertGreaterThan(value, 0.9999, "block \(index) diverges")
        }
    }

    /// The four heads, end to end from the recorded spectrograms.
    func testTheLogitsMatchTheReference() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let logits = net(record["spectrograms"]!.expandedDimensions(axis: 0))
        eval(logits.beat, logits.downbeat, logits.section, logits.function)

        for (name, mine, reference) in [("beat", logits.beat, record["logits_beat"]!),
                                        ("downbeat", logits.downbeat, record["logits_downbeat"]!),
                                        ("section", logits.section, record["output"]!),
                                        ("function", logits.function, record["logits_function"]!)] {
            let value = cosine(mine.asArray(Float.self), reference.asArray(Float.self))
            print("VALIDATION PARITY allin1: \(name) logits cosine \(value)")
            XCTAssertGreaterThan(value, 0.9999, "the \(name) head diverges")
        }
    }

    /// The analysis the post-processing reads out of the logits: the same sections, with the same
    /// labels and boundaries, and the same beats.
    func testTheAnalysisMatchesTheReference() throws {
        try requireMLXRuntime()
        let (net, record) = try parity()
        let stems = record["stems"]!
        let analysis = net.analyze(stems: (0 ..< stems.dim(0)).map { stems[$0].asArray(Float.self) },
                                   barTracker: NFKMLXBarTracker())

        let sections = record["sections"]!                                  // [count, 3]
        let rows = sections.asArray(Float.self)
        XCTAssertEqual(analysis.sections.count, sections.dim(0), "the same number of sections")
        for index in 0 ..< min(analysis.sections.count, sections.dim(0)) {
            let mine = analysis.sections[index]
            XCTAssertEqual(mine.startSeconds, Double(rows[index * 3]), accuracy: 1e-4)
            XCTAssertEqual(mine.endSeconds, Double(rows[index * 3 + 1]), accuracy: 1e-4)
            XCTAssertEqual(mine.label, net.configuration.labels[Int(rows[index * 3 + 2])],
                           "section \(index) is labeled differently")
        }

        let beats = record["beats"]!                                        // [count, 2]
        let beatRows = beats.asArray(Float.self)
        if analysis.beats.count != beats.dim(0) {
            let mine = analysis.beats.map { String(format: "%.2f/%d", $0.timeSeconds, $0.positionInBar) }
            let theirs = (0 ..< beats.dim(0)).map {
                String(format: "%.2f/%d", beatRows[$0 * 2], Int(beatRows[$0 * 2 + 1]))
            }
            print("VALIDATION allin1 beats mine:   \(mine.joined(separator: " "))")
            print("VALIDATION allin1 beats theirs: \(theirs.joined(separator: " "))")
        }
        XCTAssertEqual(analysis.beats.count, beats.dim(0), "the same number of beats")
        for index in 0 ..< min(analysis.beats.count, beats.dim(0)) {
            XCTAssertEqual(analysis.beats[index].timeSeconds, Double(beatRows[index * 2]), accuracy: 0.011,
                           "beat \(index) falls on a different frame")
            XCTAssertEqual(analysis.beats[index].positionInBar, Int(beatRows[index * 2 + 1]),
                           "beat \(index) is at a different position in the bar")
        }
    }

    // MARK: Released weights

    func testTheReleasedCheckpointLoadsEveryTensor() throws {
        try requireMLXRuntime()
        guard let weightsPath = NFKMLXValidationConfig.environment["IK_VAL_ALLIN1"] else {
            throw XCTSkip("set IK_VAL_ALLIN1 to the converted All-In-One checkpoint")
        }
        let net = NFKMLXAllInOne.makeNet()
        try NFKMLXAllInOne.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))

        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: weightsPath)).arrays
        let module = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        XCTAssertEqual(Set(checkpoint.keys).subtracting(module.keys), [], "tensors the module has no home for")
        XCTAssertEqual(Set(module.keys).subtracting(checkpoint.keys), [], "parameters the checkpoint does not fill")
        for (key, value) in module {
            guard let reference = checkpoint[key] else { continue }
            let expected = value.ndim == 4 ? [reference.dim(0), reference.dim(2), reference.dim(3), reference.dim(1)]
                                           : reference.shape
            XCTAssertEqual(value.shape, expected, "shape of \(key)")
        }
    }
}
