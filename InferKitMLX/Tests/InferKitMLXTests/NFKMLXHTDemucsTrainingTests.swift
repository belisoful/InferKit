//
//  NFKMLXHTDemucsTrainingTests.swift
//  InferKitMLXTests
//
//  Training Hybrid Transformer Demucs without released weights: the batched forward the solver trains,
//  the transformer's dropout, the two recipes the released checkpoints record, and a fine-tune that moves
//  every weight through both branches and reloads through the public factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXHTDemucsTrainingTests: XCTestCase {

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

    func testTheRecipesAreTheOnesTheCheckpointsRecord() {
        let base = NFKMLXHTDemucsTrainingRecipe.htdemucs
        XCTAssertEqual(base.learningRate, 3e-4)
        XCTAssertEqual(base.transformerWeightDecay, 0)
        XCTAssertNil(base.clipGradientNorm)
        XCTAssertEqual(base.batchSize, 32)
        XCTAssertEqual(base.augmentation?.remixProbability, 1)
        XCTAssertEqual(base.augmentation?.scaleProbability, 1)
        XCTAssertEqual(base.augmentation?.shiftSame, false)

        let fineTune = NFKMLXHTDemucsTrainingRecipe.htdemucsFineTune
        XCTAssertEqual(fineTune.learningRate, 1e-4)
        XCTAssertEqual(fineTune.transformerWeightDecay, 0.05)
        XCTAssertEqual(fineTune.clipGradientNorm, 5)
        XCTAssertEqual(fineTune.batchSize, 8)
        XCTAssertEqual(fineTune.augmentation?.remixProbability, 0)
        XCTAssertEqual(fineTune.augmentation?.scaleProbability, 0)
        XCTAssertEqual(fineTune.augmentation?.shiftSame, true)
        XCTAssertEqual(NFKMLXHTDemucsConfiguration.htdemucs.transformerDropout, 0.02)
    }

    func testAHalfPrecisionCheckpointLoadsIntoFloat32Parameters() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("htdemucs-half-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let configuration = NFKMLXHTDemucsTests.tinyConfiguration()
        let stored = NFKMLXHTDemucs.makeNet(configuration)
        stored.update(parameters: stored.parameters().mapValues { $0.asType(.float16) })
        try NFKMLXWeights.save(stored, to: url)

        let loaded = try NFKMLXHTDemucs.network(weightsURL: url, configuration: configuration)
        for (name, value) in loaded.parameters().flattened() {
            XCTAssertEqual(value.dtype, .float32, "\(name) trains at float32, as the reference's load_state_dict copies it")
        }
    }

    func testANewNetworksResidualBranchesCompressByEight() {
        let net = NFKMLXHTDemucs.makeNet(.htdemucs)
        XCTAssertEqual(net.encoder[1].dconv.layers[0].convIn.weight.dim(0), 96 / 8, "the releases' dconv_comp")
    }

    func testTheBatchedEstimateIsEachClipSeparatedAlone() throws {
        try requireMLXRuntime()
        let net = NFKMLXHTDemucs.makeNet(NFKMLXHTDemucsTests.tinyConfiguration())
        let stems = NFKMLXDemucsTrainingTests.stems(batch: 2, length: 4096)
        let mix = stems.sum(axis: 1)
        let batched = net.estimate(mix)
        XCTAssertEqual(batched.shape, [2, 4, 2, 4096])
        for index in 0 ..< 2 {
            let alone = net.separate(mix[index], padsToTrainingSegment: false)
            let difference = abs(batched[index] - alone).max().item(Float.self)
            XCTAssertLessThan(difference, 1e-5, "clip \(index) is normalized by its own statistics")
        }
    }

    func testDropoutActsOnlyInTraining() throws {
        try requireMLXRuntime()
        let net = NFKMLXHTDemucs.makeNet(NFKMLXHTDemucsTests.tinyConfiguration())
        let mix = NFKMLXDemucsTrainingTests.stems(batch: 1, length: 4096).sum(axis: 1)

        let evaluated = [net.estimate(mix), net.estimate(mix)]
        eval(evaluated)
        XCTAssertEqual(evaluated[0].asArray(Float.self), evaluated[1].asArray(Float.self), "a built network evaluates")

        net.train(true)
        let trained = [net.estimate(mix), net.estimate(mix)]
        eval(trained)
        XCTAssertGreaterThan(abs(trained[0] - trained[1]).max().item(Float.self), 0, "training drops activations")

        var configuration = NFKMLXHTDemucsTests.tinyConfiguration()
        configuration.transformerDropout = 0
        let plain = NFKMLXHTDemucs.makeNet(configuration)
        plain.update(parameters: net.parameters())
        plain.train(true)
        let undropped = plain.estimate(mix)
        net.train(false)
        let reference = net.estimate(mix)
        eval(undropped, reference)
        XCTAssertEqual(undropped.asArray(Float.self), reference.asArray(Float.self), "no dropout, no difference")
    }

    func testTheFineTuneOptimizerDecaysOnlyTheTransformer() throws {
        try requireMLXRuntime()
        let net = NFKMLXHTDemucs.makeNet(NFKMLXHTDemucsTests.tinyConfiguration())
        let before = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 * 1) })
        eval(Array(before.values))
        let optimizer = NFKMLXHTDemucs.referenceOptimizer(for: net, recipe: .htdemucsFineTune)
        optimizer.update(model: net, gradients: net.trainableParameters().mapValues { MLXArray.zeros(like: $0) })

        for (name, after) in net.parameters().flattened() {
            let factor: Float = name.hasPrefix("crosstransformer.") ? 1 - 1e-4 * 0.05 : 1
            let expected = before[name]! * factor
            XCTAssertLessThan(abs(after - expected).max().item(Float.self), 1e-7, "\(name) decays by \(factor)")
        }
    }

    func testEveryWeightReceivesAGradientThroughBothBranches() throws {
        try requireMLXRuntime()
        // Sixteen channels give each dilated branch a hidden width of 2. At width 1 the bias ahead of the branch's
        // one-group norm is removed whole, and its gradient is zero.
        var configuration = NFKMLXHTDemucsTests.tinyConfiguration()
        configuration.channels = 16
        let net = NFKMLXHTDemucs.makeNet(configuration)
        let batch = NFKMLXDemucsTrainingTests.stems(batch: 2, length: 4096)
        net.train(true)
        // The residual branches start scaled by 1e-3 and 1e-4, so a first Adam step can round away on the
        // parameters behind them; the gradient itself is what must reach every one.
        let gradients = valueAndGrad(model: net) { net, arrays in
            [NFKMLXDemucsObjective()(net, arrays[0])]
        }(net, [batch]).1.flattened()
        XCTAssertEqual(gradients.count, net.trainableParameters().flattened().count)
        for (name, gradient) in gradients {
            XCTAssertGreaterThan(abs(gradient).max().item(Float.self), 0, "\(name) receives a gradient")
        }
    }

    func testFineTuningLowersTheLoss() throws {
        try requireMLXRuntime()
        let net = NFKMLXHTDemucs.makeNet(NFKMLXHTDemucsTests.tinyConfiguration())
        let batch = NFKMLXDemucsTrainingTests.stems(batch: 2, length: 4096)
        var recipe = NFKMLXHTDemucsTrainingRecipe.htdemucs
        recipe.augmentation = nil
        let history = try NFKMLXHTDemucs.fineTune(net, stems: { _ in batch }, recipe: recipe,
                                                  optimizer: Adam(learningRate: 1e-3), steps: 40)
        XCTAssertEqual(history.count, 40)
        XCTAssertLessThan(history.last!, history.first!, "the objective fell: \(history.first!) -> \(history.last!)")
    }

    func testAFineTunedCheckpointReloadsThroughThePublicFactories() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("htdemucs-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        // The released geometry, so the backend factory, which builds that geometry, reads the file.
        let net = try NFKMLXHTDemucs.network(weightsURL: nil)
        let batch = NFKMLXDemucsTrainingTests.stems(batch: 1, length: 8192)
        var recipe = NFKMLXHTDemucsTrainingRecipe.htdemucsFineTune
        recipe.augmentation = nil
        try NFKMLXHTDemucs.fineTune(net, stems: { _ in batch }, objective: .stem(3, of: 4), recipe: recipe, steps: 1)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXHTDemucs.network(weightsURL: url)
        let mix = batch.sum(axis: 1)
        let expected = net.estimate(mix), actual = reloaded.estimate(mix)
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self), "the fine-tuned checkpoint round-trips exactly")

        let inputURL = FileManager.default.temporaryDirectory.appendingPathComponent("htdemucs-tuned-in-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: inputURL) }
        try NFKMLXWaveFile.write(samples: (0 ..< 4096).map { sinf(Float($0) * 0.1) }, sampleRate: 44_100, to: inputURL)
        let backend = try NFKMLXHTDemucs.backend(weightsURL: url)
        let request = NFKInferenceRequest(inputs: [NFKInputAudio: NFKAudioAsset(fileURL: inputURL, durationSeconds: 0,
                                                                                  sampleRate: 44_100, channelCount: 1)])
        let result = try backend.runInference(for: request)
        let vocals = try XCTUnwrap(result.output(forKey: "vocals") as? NFKAudioAsset)
        if let stemURL = vocals.fileURL { try? FileManager.default.removeItem(at: stemURL) }
    }
}
