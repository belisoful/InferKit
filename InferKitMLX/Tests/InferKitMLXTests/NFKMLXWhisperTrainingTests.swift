//
//  NFKMLXWhisperTrainingTests.swift
//  InferKitMLXTests
//
//  Domain-adapting speech recognition. The pieces that carry the recipe: the objective scores next-token
//  prediction the way teacher forcing means it to, LoRA reaches only the decoder's attention, and a run
//  merges back to one ordinary checkpoint.
//

import XCTest
import MLX
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXWhisperTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func setUp() {
        super.setUp()
        NFKMLXRandom.seed(20_260_814)
    }

    /// A tiny Whisper: the real one is far too slow to train in a test.
    private func configuration(textLayers: Int = 1) -> NFKMLXWhisperConfiguration {
        var configuration = NFKMLXWhisperConfiguration()
        configuration.nMels = 8
        configuration.nAudioState = 16
        configuration.nAudioHead = 2
        configuration.nAudioLayer = 1
        configuration.nAudioCtx = 12
        configuration.nVocab = 50
        configuration.nTextState = 16
        configuration.nTextHead = 2
        configuration.nTextLayer = textLayers
        configuration.nTextCtx = 8
        configuration.promptTokens = [1, 2]
        configuration.endToken = 3
        configuration.suppressFrom = 40
        configuration.maxTokens = 4
        return configuration
    }

    private func clip(_ net: NFKMLXWhisperNet) -> (mel: MLXArray, tokens: MLXArray) {
        let samples = (0 ..< 3200).map { sinf(Float($0) * 0.05) * 0.3 }
        let mel = NFKMLXMel.logMel(samples, sampleRate: 16000, nMels: configuration().nMels)
        return (mel, MLXArray([Int32(1), 2, 7, 9, 3]))
    }

    // MARK: - The objective

    func testTheObjectiveScoresNextTokenPrediction() throws {
        try requireMLXRuntime()
        let net = NFKMLXWhisper.makeNet(configuration())
        let sample = clip(net)
        let loss = NFKMLXWhisperObjective()(net, sample.mel, sample.tokens)
        XCTAssertTrue(loss.item(Float.self).isFinite)
        XCTAssertGreaterThan(loss.item(Float.self), 0)
    }

    func testTheObjectiveAlignsPredictionsWithTheFollowingToken() throws {
        try requireMLXRuntime()
        // Logits that put all their mass on the correct NEXT token score near zero. If the objective
        // were misaligned by one position, the same logits would score badly.
        let tokens = MLXArray([Int32(1), 4, 2, 5])
        let vocabulary = 8
        var values = [Float](repeating: 0, count: tokens.shape[0] * vocabulary)
        let targets: [Int] = [4, 2, 5, 0]
        for position in 0 ..< tokens.shape[0] {
            values[position * vocabulary + targets[position]] = 20
        }
        let logits = values.withUnsafeBufferPointer { MLXArray($0, [1, tokens.shape[0], vocabulary]) }

        let loss = NFKMLXWhisperObjective().loss(logits: logits, tokens: tokens)
        XCTAssertLessThan(loss.item(Float.self), 1e-4, "confident, correctly aligned predictions")
    }

    func testASingleTokenSequenceHasNothingToPredict() throws {
        try requireMLXRuntime()
        let loss = NFKMLXWhisperObjective().loss(logits: MLXArray.zeros([1, 1, 8]),
                                                 tokens: MLXArray([Int32(1)]))
        XCTAssertEqual(loss.item(Float.self), 0, "no position has a following token")
    }

    // MARK: - The spectrogram

    func testTheSpectrogramPadsToTheThirtySecondWindow() throws {
        try requireMLXRuntime()
        // Whisper only ever sees 30-second inputs; training on shorter ones was the single biggest
        // accuracy factor when this model was brought to reference parity.
        let short = NFKMLXWhisper.spectrogram(for: [Float](repeating: 0, count: 1600), sampleRate: 16000)
        let long = NFKMLXWhisper.spectrogram(for: [Float](repeating: 0, count: 16000 * 45), sampleRate: 16000)
        XCTAssertEqual(short.shape, long.shape, "short and long clips reach the same window")
    }

    // MARK: - What LoRA reaches

    func testAdaptationTargetsTheDecoderAttentionProjectionsOnly() throws {
        XCTAssertTrue(NFKMLXWhisper.isDecoderAttentionProjection("decoder.blocks.0.attn.query"))
        XCTAssertTrue(NFKMLXWhisper.isDecoderAttentionProjection("decoder.blocks.2.cross_attn.value"))
        XCTAssertFalse(NFKMLXWhisper.isDecoderAttentionProjection("decoder.blocks.0.attn.key"),
                       "query and value are the reference LoRA choice")
        XCTAssertFalse(NFKMLXWhisper.isDecoderAttentionProjection("decoder.blocks.0.attn.out"))
        XCTAssertFalse(NFKMLXWhisper.isDecoderAttentionProjection("encoder.blocks.0.attn.query"),
                       "the encoder's audio features transfer across domains and stay frozen")
    }

    func testAdaptingLeavesTheEncoderFrozen() throws {
        try requireMLXRuntime()
        let net = NFKMLXWhisper.makeNet(configuration())
        try NFKMLXLoRA.apply(to: net, rank: 2) { path, _ in
            NFKMLXWhisper.isDecoderAttentionProjection(path)
        }
        let trainable = net.trainableParameters().flattened().map(\.0)
        XCTAssertFalse(trainable.isEmpty)
        XCTAssertTrue(trainable.allSatisfy { $0.hasPrefix("decoder.") },
                      "nothing in the encoder trains: \(trainable.filter { !$0.hasPrefix("decoder.") })")
    }

    // MARK: - Training

    func testFineTuningDrivesTheLossDown() throws {
        try requireMLXRuntime()
        let net = NFKMLXWhisper.makeNet(configuration())
        let sample = clip(net)

        let history = try NFKMLXWhisper.fineTune(net, examples: { _ in sample }, rank: 4,
                                                 optimizer: AdamW(learningRate: 1e-2), steps: 30)

        XCTAssertEqual(history.count, 30)
        XCTAssertLessThan(history.last!, history.first!,
                          "the adapters learned the clip: \(history.first!) -> \(history.last!)")
    }

    func testTrainingMovesOnlyTheAdapters() throws {
        try requireMLXRuntime()
        let net = NFKMLXWhisper.makeNet(configuration())
        let sample = clip(net)
        let encoderBefore = net.encoder.conv1.weight.asArray(Float.self)

        try NFKMLXWhisper.fineTune(net, examples: { _ in sample }, rank: 4,
                                   optimizer: AdamW(learningRate: 1e-2), steps: 10)

        XCTAssertEqual(net.encoder.conv1.weight.asArray(Float.self), encoderBefore,
                       "the frozen encoder did not move")
    }

    func testAnAdaptedRunMergesBackToAnOrdinaryCheckpoint() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("whisper-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = NFKMLXWhisper.makeNet(configuration())
        let sample = clip(net)
        try NFKMLXWhisper.fineTune(net, examples: { _ in sample }, rank: 4,
                                   optimizer: AdamW(learningRate: 1e-2), steps: 10)

        let adapted = NFKMLXWhisperObjective()(net, sample.mel, sample.tokens).item(Float.self)
        XCTAssertGreaterThan(try NFKMLXLoRA.merge(into: net), 0)
        let merged = NFKMLXWhisperObjective()(net, sample.mel, sample.tokens).item(Float.self)
        XCTAssertEqual(merged, adapted, accuracy: 1e-4, "merging preserved what was learned")

        try NFKMLXWeights.save(net, to: url)
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        XCTAssertFalse(checkpoint.arrays.keys.contains { $0.contains("lora_") },
                       "the saved file is an ordinary Whisper checkpoint")

        let reloaded = NFKMLXWhisper.makeNet(configuration())
        XCTAssertNoThrow(try NFKMLXWeights.apply(Array(checkpoint.arrays), to: reloaded),
                         "and an unmodified model loads every parameter of it")
    }

    func testAPredicateThatMatchesNothingFailsRatherThanTrainingNothing() throws {
        try requireMLXRuntime()
        // A rank with no matching layers would run happily and change nothing, wasting the device's
        // battery for no result.
        let net = NFKMLXWhisper.makeNet(configuration(textLayers: 0))
        XCTAssertThrowsError(try NFKMLXWhisper.fineTune(net, examples: { _ in self.clip(net) },
                                                        rank: 4, steps: 1))
    }
}
