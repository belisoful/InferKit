//
//  NFKMLXSileroVADTrainingTests.swift
//  InferKitMLXTests
//
//  Silero VAD's decoder fine-tune: the chunk targets, the objective, the training-only dropout, the
//  frozen encoder, the folded LSTM bias's doubled rate, the recipe, and the round trip through the
//  model's own loader. Parity against snakers4's `tuning/` lives in NFKMLXReferenceParityTests.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXSileroVADTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_001)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Sixteen chunks at 16 kHz: a voiced stack for the first half, quiet noise after.
    private func example() -> (samples: [Float], labels: [Float], mask: [Float]) {
        let samples = (0 ..< 16 * 512).map { index -> Float in
            let t = Float(index) / 16000
            guard t < 0.256 else { return 0.02 * sinf(Float(index) * 12.9898) }
            return (1 ... 5).reduce(Float(0)) { $0 + 0.3 / Float($1) * sinf(2 * .pi * 150 * Float($1) * t) }
        }
        let targets = NFKMLXSileroVAD.chunkTargets(speech: [(0, 0.256)], sampleCount: samples.count)
        return (samples, targets.labels, targets.mask)
    }

    func testAChunkIsSpeechWhenMoreThanHalfOfItIs() {
        // 1,100 samples pad to three chunks. The first span covers 488 samples of chunk 1 and 12 of
        // chunk 0; the second covers exactly half of chunk 2 and runs past the clip.
        let targets = NFKMLXSileroVAD.chunkTargets(speech: [(0.03125, 0.0625), (0.08, 0.1)], sampleCount: 1100)
        XCTAssertEqual(targets.labels, [0, 1, 0])
        XCTAssertEqual(targets.mask, [0.5, 1, 0.5])
    }

    func testTheObjectiveIsTheMaskedMeanBinaryCrossEntropy() throws {
        try requireMLXRuntime()
        let loss = NFKMLXSileroVADObjective().loss(probabilities: MLXArray([Float(0.9), 0.2, 0.6, 0]),
                                                   labels: MLXArray([Float(1), 0, 1, 1]),
                                                   mask: MLXArray([Float(1), 0.5, 1, 0.5]))
        // The last chunk's log term is floored at −100, as torch floors it.
        let expected = (-log(0.9) - 0.5 * log(0.8) - log(0.6) + 0.5 * 100) / 4
        XCTAssertEqual(Double(loss.item(Float.self)), expected, accuracy: 1e-5)
    }

    func testTheDecoderDropoutActsOnlyInTraining() throws {
        try requireMLXRuntime()
        let net = try NFKMLXSileroVAD.network(weightsURL: nil)
        let input = NFKMLXSileroVADNet.chunkedInput(example().samples, net.configuration)
        let first = net.decoder(net.encoder(input)), second = net.decoder(net.encoder(input))
        XCTAssertEqual(first.asArray(Float.self), second.asArray(Float.self))
        net.train(true)
        let noisy = net.decoder(net.encoder(input))
        net.train(false)
        XCTAssertGreaterThan(abs(noisy - first).max().item(Float.self), 0)
    }

    func testADecoderRunLeavesTheEncoderUntouched() throws {
        try requireMLXRuntime()
        let net = try NFKMLXSileroVAD.network(weightsURL: nil)
        let item = example()
        let encoder = net.encoder.parameters().flattened().map { $0.1.asArray(Float.self) }
        let final = net.decoder.final.weight.asArray(Float.self)

        let history = try NFKMLXSileroVAD.fineTune(net, examples: { _ in item }, steps: 3)

        XCTAssertEqual(history.count, 3)
        for (after, before) in zip(net.encoder.parameters().flattened().map { $0.1.asArray(Float.self) }, encoder) {
            XCTAssertEqual(after, before, "the frozen encoder did not move")
        }
        XCTAssertNotEqual(net.decoder.final.weight.asArray(Float.self), final, "the decoder did")
        XCTAssertFalse(net.training)
    }

    func testTrainingDrivesTheLossDown() throws {
        try requireMLXRuntime()
        let net = try NFKMLXSileroVAD.network(weightsURL: nil)
        let item = example()
        let history = try NFKMLXSileroVAD.fineTune(net, examples: { _ in item }, steps: 30)
        XCTAssertLessThan(history.last!, history.first!, "\(history.first!) -> \(history.last!)")
    }

    func testTheFoldedLSTMBiasStepsAtTwiceTheRate() throws {
        try requireMLXRuntime()
        // The reference's LSTM keeps two biases that both take this port's one gradient, so Adam's
        // first step, about the rate in size wherever the gradient is not tiny, moves their sum twice.
        let net = try NFKMLXSileroVAD.network(weightsURL: nil)
        net.encoder.freeze()
        let item = example()
        let input = NFKMLXSileroVADNet.chunkedInput(item.samples, net.configuration)
        let objective = NFKMLXSileroVADObjective()
        let gradients = valueAndGrad(model: net) { net, _ in
            [objective.loss(probabilities: net.decoder(net.encoder(input)), labels: MLXArray(item.labels),
                            mask: MLXArray(item.mask))]
        }(net, []).1
        let bias = try XCTUnwrap(net.decoder.rnn.bias) * 1, inputWeights = net.decoder.rnn.wx * 1
        eval(bias, inputWeights)

        NFKMLXSileroVAD.referenceOptimizer(for: net).update(model: net, gradients: gradients)

        let biasStep = abs(try XCTUnwrap(net.decoder.rnn.bias) - bias).max().item(Float.self)
        let weightStep = abs(net.decoder.rnn.wx - inputWeights).max().item(Float.self)
        XCTAssertEqual(biasStep, 1e-3, accuracy: 1e-5)
        XCTAssertEqual(weightStep, 5e-4, accuracy: 5e-6)
    }

    func testAFineTunedCheckpointLoadsThroughTheLoader() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("silero-vad-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXSileroVAD.network(weightsURL: nil)
        let item = example()
        try NFKMLXSileroVAD.fineTune(net, examples: { _ in item }, steps: 2)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXSileroVAD.network(weightsURL: url)
        let input = NFKMLXSileroVADNet.chunkedInput(item.samples, net.configuration)
        XCTAssertEqual(reloaded.decoder(reloaded.encoder(input)).asArray(Float.self),
                       net.decoder(net.encoder(input)).asArray(Float.self))
        XCTAssertTrue(try NFKMLXSileroVAD.backend(weightsURL: url).isReady)
    }
}
