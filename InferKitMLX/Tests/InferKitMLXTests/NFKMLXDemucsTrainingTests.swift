//
//  NFKMLXDemucsTrainingTests.swift
//  InferKitMLXTests
//
//  Training the music separators without released weights: the solver's stem-weighted distance, the
//  augmentations replayed from known draws, the v2 network's unpadded estimate and the geometry read from
//  a checkpoint, and a Demucs v2 fine-tune that lowers the loss and reloads through the public factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

/// Answers each kind of draw from its own queue, in order.
struct NFKDemucsReplayedDraws: NFKDemucsDraws {
    var units: [Double] = []
    var integers: [[Int]] = []
    var uniforms: [[Float]] = []

    mutating func unit() -> Double { units.removeFirst() }
    mutating func integers(below bound: Int, count: Int) -> [Int] { integers.removeFirst() }
    mutating func uniforms(_ count: Int, low: Float, high: Float) -> [Float] { uniforms.removeFirst() }
}

final class NFKMLXDemucsTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_023)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// `[N, stems, channels, L]` whose every value names its position: `1000·n + 100·s + 10·c + t`.
    static func labeled(batch: Int, stems: Int = 4, length: Int) -> MLXArray {
        var values = [Float]()
        for n in 0 ..< batch {
            for s in 0 ..< stems {
                for c in 0 ..< 2 {
                    for t in 0 ..< length { values.append(Float(1000 * n + 100 * s + 10 * c + t)) }
                }
            }
        }
        return MLXArray(values, [batch, stems, 2, length])
    }

    /// Four stems of a few sinusoids each, `[batch, 4, 2, length]`, distinct per example and stem.
    static func stems(batch: Int, length: Int) -> MLXArray {
        var values = [Float]()
        for n in 0 ..< batch {
            for s in 0 ..< 4 {
                for c in 0 ..< 2 {
                    for t in 0 ..< length {
                        let phase = Float(t) * (0.01 + 0.013 * Float(s) + 0.002 * Float(n)) + Float(c)
                        values.append(0.2 * sinf(phase) + 0.05 * sinf(3.1 * phase + Float(n)))
                    }
                }
            }
        }
        return MLXArray(values, [batch, 4, 2, length])
    }

    func testTheObjectiveDefaultsAreTheSolvers() {
        let objective = NFKMLXDemucsObjective()
        XCTAssertEqual(objective.distance, .l1)
        XCTAssertNil(objective.weights, "every stem weighs the same")
        XCTAssertEqual(NFKMLXDemucsObjective.stem(2, of: 4).weights, [0, 0, 1, 0])
    }

    func testEqualWeightsAreTheMeanDistanceOverEveryStem() throws {
        try requireMLXRuntime()
        let sources = Self.stems(batch: 2, length: 64)
        let estimate = sources * 0.5 + 0.01
        let l1 = NFKMLXDemucsObjective().loss(estimate: estimate, sources: sources).item(Float.self)
        XCTAssertEqual(l1, abs(estimate - sources).mean().item(Float.self), accuracy: 1e-7)
        let l2 = NFKMLXDemucsObjective(distance: .l2).loss(estimate: estimate, sources: sources).item(Float.self)
        XCTAssertEqual(l2, square(estimate - sources).mean().item(Float.self), accuracy: 1e-7)
    }

    func testAStemObjectiveScoresThatStemAlone() throws {
        try requireMLXRuntime()
        let sources = Self.stems(batch: 2, length: 64)
        let estimate = sources * 0.5
        let bass = NFKMLXDemucsObjective.stem(1, of: 4).loss(estimate: estimate, sources: sources).item(Float.self)
        XCTAssertEqual(bass, abs(estimate[0..., 1] - sources[0..., 1]).mean().item(Float.self), accuracy: 1e-7)
    }

    func testTheV2ObjectiveScoresTheEstimateAgainstTheMiddleOfTheStems() throws {
        try requireMLXRuntime()
        let net = NFKMLXDemucs.makeNet(NFKMLXDemucsTests.tinyConfiguration())
        let length = net.validLength(600)
        let sources = Self.stems(batch: 2, length: length)
        let estimate = net.estimate(sources.sum(axis: 1).transposed(0, 2, 1))
        XCTAssertLessThan(estimate.dim(1), length, "the decoder's context consumes samples at each stage")

        let shorter = estimate.dim(1), delta = length - shorter
        let middle = sources[0..., 0..., 0..., (delta / 2) ..< (delta / 2 + shorter)]
        let stemsFirst = estimate.reshaped([2, shorter, 4, 2]).transposed(0, 2, 3, 1)
        XCTAssertEqual(NFKMLXDemucsObjective()(net, sources).item(Float.self),
                       abs(stemsFirst - middle).mean().item(Float.self), accuracy: 1e-6)
    }

    func testTheCropTakesEachStemsOwnOffset() throws {
        try requireMLXRuntime()
        let stems = Self.labeled(batch: 2, length: 8)
        let crop = NFKMLXDemucsAugmentation(shift: 3, flip: false, scaleProbability: 0, remixProbability: 0)
        var draws = NFKDemucsReplayedDraws(integers: [[0, 1, 2, 0, 2, 1, 0, 1]])
        let out = crop.apply(stems, draws: &draws)
        XCTAssertEqual(out.shape, [2, 4, 2, 5])
        XCTAssertEqual(out[1, 1, 1, 0].item(Float.self), 1111, "example 1, stem 1 starts at its offset of 1")
        XCTAssertEqual(out[0, 2, 0, 4].item(Float.self), 206, "example 0, stem 2 starts at its offset of 2")

        let shared = NFKMLXDemucsAugmentation(shift: 3, shiftSame: true, flip: false, scaleProbability: 0,
                                              remixProbability: 0)
        var one = NFKDemucsReplayedDraws(integers: [[1, 2]])
        let together = shared.apply(stems, draws: &one)
        XCTAssertEqual(together[1, 3, 0, 0].item(Float.self), 1302, "every stem of example 1 starts at 2")
    }

    func testTheFlipsTakeTheirDrawsInEachBranchsOrder() throws {
        try requireMLXRuntime()
        let stems = Self.labeled(batch: 1, stems: 1, length: 2) + 1
        let bits = [[1], [0]]
        var v4 = NFKDemucsReplayedDraws(integers: bits)
        let flippedV4 = NFKMLXDemucsAugmentation(shift: 0, scaleProbability: 0, remixProbability: 0)
            .apply(stems, draws: &v4)
        XCTAssertEqual(flippedV4[0, 0, 0, 0].item(Float.self), -11, "v4 swaps on the first draw, then flips the sign")
        var v2 = NFKDemucsReplayedDraws(integers: bits)
        let flippedV2 = NFKMLXDemucsAugmentation(shift: 0, signBeforeChannels: true, scaleProbability: 0,
                                                 remixProbability: 0).apply(stems, draws: &v2)
        XCTAssertEqual(flippedV2[0, 0, 0, 0].item(Float.self), 1, "v2 keeps the sign on the first draw and swaps none")
    }

    func testTheGainAndTheRemixFollowTheirDraws() throws {
        try requireMLXRuntime()
        let stems = Self.labeled(batch: 4, stems: 2, length: 2) + 1
        let augmentation = NFKMLXDemucsAugmentation(shift: 0, flip: false, remixGroupSize: 2)
        // The gain draws its probability then eight factors; the remix draws its probability then one
        // key per example and stem, grouped by twos.
        var draws = NFKDemucsReplayedDraws(
            units: [0.5, 0.5],
            uniforms: [[1, 1, 1, 1, 1, 1, 1, 0.5], [0.9, 0.1, 0.2, 0.8, 0.3, 0.6, 0.7, 0.4]])
        let out = augmentation.apply(stems, draws: &draws)
        XCTAssertEqual(out[0, 0, 0, 0].item(Float.self), 1001, "example 0's stem 0 comes from example 1")
        XCTAssertEqual(out[0, 1, 0, 0].item(Float.self), 101, "example 0's stem 1 stays")
        XCTAssertEqual(out[2, 1, 1, 1].item(Float.self), Float(3112) * 0.5, "example 3's halved stem 1 moves to example 2")
        XCTAssertEqual(out[3, 1, 0, 0].item(Float.self), 2101, "example 3's stem 1 comes from example 2")
    }

    func testARemixRefusesABatchItsGroupDoesNotDivide() throws {
        try requireMLXRuntime()
        let net = NFKMLXDemucs.makeNet(NFKMLXDemucsTests.tinyConfiguration())
        let batch = Self.stems(batch: 3, length: net.validLength(400) + 64)
        var augmentation = NFKMLXDemucsAugmentation.demucsV2
        augmentation.shift = 64
        XCTAssertThrowsError(try NFKMLXDemucs.fineTune(net, stems: { _ in batch }, augmentation: augmentation, steps: 1))
    }

    func testTheGeometryIsReadFromTheCheckpoint() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXDemucsTests.tinyConfiguration()
        configuration.lstmLayers = 1
        let trained = NFKMLXDemucs.makeNet(configuration)
        let read = try NFKMLXDemucs.configuration(matching: Dictionary(uniqueKeysWithValues: trained.parameters().flattened()),
                                                  base: NFKMLXDemucsConfiguration())
        XCTAssertEqual(read.baseChannels, 8)
        XCTAssertEqual(read.depth, 2)
        XCTAssertEqual(read.lstmLayers, 1)
        XCTAssertEqual(read.stems, 4)
        XCTAssertEqual(read.audioChannels, 2)

        // The reference's layout: `[out, in, kernel]` convolutions and a `[in, out, kernel]` transposed one.
        let reference: [String: MLXArray] = [
            "encoder.0.0.weight": MLXArray.zeros([64, 2, 8]), "encoder.1.0.weight": MLXArray.zeros([128, 64, 8]),
            "lstm.lstm.weight_ih_l0": MLXArray.zeros([1]), "lstm.lstm.weight_ih_l1": MLXArray.zeros([1]),
            "decoder.1.2.weight": MLXArray.zeros([64, 12, 8]),
        ]
        let release = try NFKMLXDemucs.configuration(matching: reference, base: NFKMLXDemucsConfiguration())
        XCTAssertEqual(release.baseChannels, 64)
        XCTAssertEqual(release.depth, 2)
        XCTAssertEqual(release.lstmLayers, 2)
        XCTAssertEqual(release.stems, 6, "the last decoder writes six stereo stems")
    }

    func testAStepMovesEveryWeight() throws {
        try requireMLXRuntime()
        let net = NFKMLXDemucs.makeNet(NFKMLXDemucsTests.tinyConfiguration())
        let batch = Self.stems(batch: 4, length: net.validLength(400) + 64)
        var augmentation = NFKMLXDemucsAugmentation.demucsV2
        augmentation.shift = 64
        let before = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 * 1) })
        eval(Array(before.values))

        try NFKMLXDemucs.fineTune(net, stems: { _ in batch }, augmentation: augmentation, steps: 1)

        for (name, after) in net.parameters().flattened() {
            XCTAssertGreaterThan(abs(after - before[name]!).max().item(Float.self), 0, "\(name) trains")
        }
    }

    func testFineTuningLowersTheLoss() throws {
        try requireMLXRuntime()
        let net = NFKMLXDemucs.makeNet(NFKMLXDemucsTests.tinyConfiguration())
        let batch = Self.stems(batch: 2, length: net.validLength(400))
        let history = try NFKMLXDemucs.fineTune(net, stems: { _ in batch }, augmentation: nil,
                                                optimizer: Adam(learningRate: 1e-3), steps: 40)
        XCTAssertEqual(history.count, 40)
        XCTAssertLessThan(history.last!, history.first!, "the objective fell: \(history.first!) -> \(history.last!)")
    }

    func testAFineTunedCheckpointReloadsThroughThePublicFactories() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("demucs-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXDemucs.network(weightsURL: nil, configuration: NFKMLXDemucsTests.tinyConfiguration())
        let batch = Self.stems(batch: 2, length: net.validLength(400))
        try NFKMLXDemucs.fineTune(net, stems: { _ in batch }, augmentation: nil, optimizer: Adam(learningRate: 1e-3), steps: 5)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXDemucs.network(weightsURL: url)
        XCTAssertEqual(reloaded.configuration.baseChannels, 8, "the width comes from the file")
        let mix = batch.sum(axis: 1).transposed(0, 2, 1)
        let expected = net(mix), actual = reloaded(mix)
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self), "the fine-tuned checkpoint round-trips exactly")

        let inputURL = FileManager.default.temporaryDirectory.appendingPathComponent("demucs-tuned-in-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: inputURL) }
        try NFKMLXWaveFile.write(samples: (0 ..< 2048).map { sinf(Float($0) * 0.1) }, sampleRate: 44_100, to: inputURL)
        let backend = try NFKMLXDemucs.backend(weightsURL: url)
        let request = NFKInferenceRequest(inputs: [NFKInputAudio: NFKAudioAsset(fileURL: inputURL, durationSeconds: 0,
                                                                                  sampleRate: 44_100, channelCount: 1)])
        let result = try backend.runInference(for: request)
        let vocals = try XCTUnwrap(result.output(forKey: "vocals") as? NFKAudioAsset)
        if let stemURL = vocals.fileURL { try? FileManager.default.removeItem(at: stemURL) }
    }
}
