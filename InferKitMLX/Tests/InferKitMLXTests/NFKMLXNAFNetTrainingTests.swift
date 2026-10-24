//
//  NFKMLXNAFNetTrainingTests.swift
//  InferKitMLXTests
//
//  Training NAFNet without released weights: the option files' recipes, `PSNRLoss`, the reference's padded
//  forward, the paired crop and flips under known draws, and a fine-tune that lowers the loss and reloads
//  through the public factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXNAFNetTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_024)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Two encoder stages of one block each at width 8: the release's structure at a size a test runs.
    static func tinyConfiguration() -> NFKMLXNAFNetConfiguration {
        var configuration = NFKMLXNAFNetConfiguration()
        configuration.width = 8
        configuration.encoderBlocks = [1, 1]
        configuration.middleBlocks = 1
        configuration.decoderBlocks = [1, 1]
        return configuration
    }

    /// Clean images `[batch, size, size, 3]` and noisy copies of them.
    static func pairs(batch: Int, size: Int) -> (degraded: MLXArray, clean: MLXArray) {
        var clean = [Float](), degraded = [Float]()
        for n in 0 ..< batch {
            for y in 0 ..< size {
                for x in 0 ..< size {
                    for c in 0 ..< 3 {
                        let value = 0.5 + 0.3 * sinf(Float(x) * 0.3 + Float(c) + Float(n)) * cosf(Float(y) * 0.2 - Float(c))
                        clean.append(value)
                        degraded.append(value + 0.1 * sinf(Float((x * 31 + y * 17 + c * 7 + n * 3) % 97)))
                    }
                }
            }
        }
        return (MLXArray(degraded, [batch, size, size, 3]), MLXArray(clean, [batch, size, size, 3]))
    }

    func testTheRecipesAreTheOptionFiles() {
        let sidd = NFKMLXNAFNetTrainingRecipe.sidd
        XCTAssertEqual(sidd.learningRate, 1e-3)
        XCTAssertEqual(sidd.weightDecay, 0)
        XCTAssertEqual(sidd.clipGradientNorm, 0.01, "basicsr clips unless an option file turns it off")
        XCTAssertEqual(sidd.cosineSteps, 200_000)
        XCTAssertEqual(sidd.finalLearningRate, 1e-7)
        XCTAssertEqual(sidd.batchSize, 32)
        XCTAssertEqual(sidd.augmentation?.flip, false)
        XCTAssertEqual(sidd.augmentation?.patchSize, 256)

        let goPro = NFKMLXNAFNetTrainingRecipe.goPro
        XCTAssertEqual(goPro.weightDecay, 1e-3)
        XCTAssertEqual(goPro.augmentation?.flip, true)
        XCTAssertEqual(goPro.augmentation?.rotation, true)
        XCTAssertEqual(NFKMLXNAFNetTrainingRecipe.siddWidth64.cosineSteps, 400_000)
        XCTAssertEqual(NFKMLXNAFNetTrainingRecipe.reds.batchSize, 64)
        XCTAssertEqual(NFKMLXNAFNetTrainingRecipe.goProWidth64.weightDecay, 1e-3)
    }

    func testThePSNRLossIsTheBatchMeanOfTheLogError() throws {
        try requireMLXRuntime()
        let (degraded, clean) = Self.pairs(batch: 2, size: 8)
        let loss = NFKMLXNAFNetObjective().loss(restored: degraded, target: clean).item(Float.self)
        let perImage = square(degraded - clean).mean(axes: [1, 2, 3]).asArray(Float.self)
        let expected = perImage.map { 10 / logf(10) * logf($0 + 1e-8) }.reduce(0, +) / 2
        XCTAssertEqual(loss, expected, accuracy: 1e-4)

        let luminance = NFKMLXNAFNetObjective(luminance: true).loss(restored: degraded, target: clean).item(Float.self)
        XCTAssertNotEqual(luminance, loss, "luminance compares one channel of BT.601 weights")
    }

    func testTheForwardPadsWithZerosAndCropsBack() throws {
        try requireMLXRuntime()
        let net = NFKMLXNAFNet.makeNet(Self.tinyConfiguration())
        let image = Self.pairs(batch: 1, size: 24).degraded[0..., 0 ..< 22, 0 ..< 18, 0...]
        let restored = net(image)
        XCTAssertEqual(restored.shape, [1, 22, 18, 3])
        let zeroPadded = padded(image, widths: [IntOrPair((0, 0)), IntOrPair((0, 2)), IntOrPair((0, 2)), IntOrPair((0, 0))])
        let expected = net(zeroPadded)[0..., 0 ..< 22, 0 ..< 18, 0...]
        eval(restored, expected)
        XCTAssertEqual(restored.asArray(Float.self), expected.asArray(Float.self))
    }

    func testTheAugmentationCropsAndFlipsBothImagesAlike() throws {
        try requireMLXRuntime()
        let values = (0 ..< 6 * 8 * 3).map { Float($0) }
        let image = MLXArray(values, [1, 6, 8, 3])
        // Top 1, left 2, then a horizontal flip and a transpose with no vertical flip.
        var draws = NFKDenoiserReplayedDraws(values: [1, 2, 0.1, 0.9, 0.2], permutationValues: [], offsetValues: [])
        let (degraded, clean) = NFKMLXNAFNetAugmentation(patchSize: 4, flip: true, rotation: true)
            .apply(degraded: image, clean: image + 1000, draws: &draws)
        XCTAssertTrue(draws.values.isEmpty, "five draws: the crop's two, then one per flip")
        let crop = image[0, 1 ..< 5, 2 ..< 6, 0...]
        let expected = take(crop, MLXArray([Int32(3), 2, 1, 0]), axis: 1).transposed(1, 0, 2)
        eval(degraded, expected)
        XCTAssertEqual(degraded[0].asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual((clean - degraded).asArray(Float.self), [Float](repeating: 1000, count: 48), "the same crop and flips")

        // An image smaller than the patch mirrors with its edge repeated: rows 0 1 2 continue 2 1.
        var small = NFKDenoiserReplayedDraws(values: [0, 0], permutationValues: [], offsetValues: [])
        let tiny = MLXArray((0 ..< 3 * 5 * 3).map { Float($0) }, [1, 3, 5, 3])
        let (mirrored, _) = NFKMLXNAFNetAugmentation(patchSize: 5).apply(degraded: tiny, clean: tiny, draws: &small)
        XCTAssertEqual(mirrored[0, 3, 0, 0].item(Float.self), tiny[0, 2, 0, 0].item(Float.self))
        XCTAssertEqual(mirrored[0, 4, 0, 0].item(Float.self), tiny[0, 1, 0, 0].item(Float.self))
    }

    func testEveryWeightReceivesAGradientOnceTheBlocksAreScaled() throws {
        try requireMLXRuntime()
        let net = NFKMLXNAFNet.makeNet(Self.tinyConfiguration())
        // `beta` and `gamma` start at zero, as the reference's do, which silences every block's interior.
        net.update(parameters: ModuleParameters.unflattened(net.parameters().flattened().map { key, value in
            (key, key.hasSuffix("beta") || key.hasSuffix("gamma") ? MLXArray.full(value.shape, values: MLXArray(Float(0.1))) : value)
        }))
        let (degraded, clean) = Self.pairs(batch: 2, size: 16)
        let gradients = valueAndGrad(model: net) { net, arrays in
            [NFKMLXNAFNetObjective()(net, arrays[0], arrays[1])]
        }(net, [degraded, clean]).1.flattened()
        XCTAssertEqual(gradients.count, net.trainableParameters().flattened().count)
        for (name, gradient) in gradients {
            XCTAssertGreaterThan(abs(gradient).max().item(Float.self), 0, "\(name) receives a gradient")
        }
    }

    func testFineTuningLowersTheLoss() throws {
        try requireMLXRuntime()
        let net = NFKMLXNAFNet.makeNet(Self.tinyConfiguration())
        let (degraded, clean) = Self.pairs(batch: 2, size: 32)
        var recipe = NFKMLXNAFNetTrainingRecipe.sidd
        recipe.augmentation = nil
        let history = try NFKMLXNAFNet.fineTune(net, examples: { _ in (degraded, clean) }, recipe: recipe,
                                                optimizer: Adam(learningRate: 1e-3), steps: 30)
        XCTAssertEqual(history.count, 30)
        XCTAssertLessThan(history.last!, history.first!, "the objective fell: \(history.first!) -> \(history.last!)")
    }

    func testAFineTunedCheckpointReloadsThroughThePublicFactories() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("nafnet-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        // The released SIDD geometry, so the variant factory, which builds that geometry, reads the file.
        let net = try NFKMLXNAFNet.network(weightsURL: nil, configuration: .sidd)
        let (degraded, clean) = Self.pairs(batch: 1, size: 32)
        var recipe = NFKMLXNAFNetTrainingRecipe.sidd
        recipe.augmentation = nil
        try NFKMLXNAFNet.fineTune(net, examples: { _ in (degraded, clean) }, recipe: recipe, steps: 1)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXNAFNet.network(weightsURL: url, configuration: .sidd)
        let expected = net(degraded), actual = reloaded(degraded)
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self), "the fine-tuned checkpoint round-trips exactly")

        let backend = try NFKMLXNAFNet.backend(variant: .sidd, weightsURL: url)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: NFKMLXZeroDCETrainingTests.darkImage(32)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage), "the variant factory restores with the trained weights")
    }
}
