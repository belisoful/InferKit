//
//  NFKMLXAdaINTrainingTests.swift
//  InferKitMLXTests
//
//  Training AdaIN's decoder without released weights: the reference's arguments and schedule, the
//  encoder's four slices, the random crop under known draws, a fine-tune that moves the decoder alone
//  and lowers the loss, and a reload through the public factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXAdaINTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_027)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    func testTheDefaultsAreTheReferenceArguments() {
        let objective = NFKMLXAdaINObjective()
        XCTAssertEqual(objective.contentWeight, 1)
        XCTAssertEqual(objective.styleWeight, 10)
        XCTAssertEqual(NFKMLXAdaINAugmentation().patchSize, 256)
        let schedule = NFKMLXAdaIN.referenceSchedule
        XCTAssertEqual(schedule.multiplier(0), 1, "update 0 runs at the base rate")
        XCTAssertEqual(schedule.multiplier(20_000), 0.5, accuracy: 1e-7, "1 / (1 + 5e-5 · 20000)")
        XCTAssertEqual(schedule.multiplier(160_000), 1 / 9, accuracy: 1e-7)
    }

    func testTheIntermediatesAreTheReferenceSlices() throws {
        try requireMLXRuntime()
        let net = try NFKMLXAdaIN.network(encoderURL: nil, decoderURL: nil)
        let image = NFKMLXStyleTransferTrainingTests.images(batch: 1, size: 32)
        let intermediates = net.encoder.intermediates(image)
        XCTAssertEqual(intermediates.map(\.shape), [[1, 32, 32, 64], [1, 16, 16, 128], [1, 8, 8, 256], [1, 4, 4, 512]],
                       "relu1_1, relu2_1, relu3_1, relu4_1")
        let features = net.encoder.features(image)
        eval(intermediates[3], features)
        XCTAssertEqual(intermediates[3].asArray(Float.self), features.asArray(Float.self),
                       "the last slice ends where the inference encoder does")
    }

    func testAFreshDecoderDrawsItsBiasesAsTorchDoes() throws {
        try requireMLXRuntime()
        let net = try NFKMLXAdaIN.network(encoderURL: nil, decoderURL: nil)
        let parameters = Dictionary(uniqueKeysWithValues: net.decoder.parameters().flattened())
        let bias = try XCTUnwrap(parameters["layers.28.bias"])
        XCTAssertGreaterThan(abs(bias).max().item(Float.self), 0)
        XCTAssertLessThanOrEqual(abs(bias).max().item(Float.self), (1 / Float(64 * 9)).squareRoot())
    }

    func testTheCropTakesEachImagesOwnDraws() throws {
        try requireMLXRuntime()
        let content = MLXArray((0 ..< 2 * 6 * 7 * 3).map { Float($0) }, [2, 6, 7, 3])
        let style = content + 1000
        // Content image 0 at (1, 2), content 1 at (0, 3), style 0 at (2, 0), style 1 at (1, 1).
        var draws = NFKDenoiserReplayedDraws(values: [1, 2, 0, 3, 2, 0, 1, 1], permutationValues: [], offsetValues: [])
        let (contentCrops, styleCrops) = NFKMLXAdaINAugmentation(patchSize: 4).apply(content: content, style: style, draws: &draws)
        XCTAssertTrue(draws.values.isEmpty, "two draws per image")
        for (crops, source, corners) in [(contentCrops, content, [(1, 2), (0, 3)]), (styleCrops, style, [(2, 0), (1, 1)])] {
            for (index, corner) in corners.enumerated() {
                let expected = source[index, corner.0 ..< (corner.0 + 4), corner.1 ..< (corner.1 + 4), 0...]
                XCTAssertEqual(crops[index].asArray(Float.self), expected.asArray(Float.self))
            }
        }
    }

    func testAFineTuneMovesTheDecoderAndLeavesTheEncoder() throws {
        try requireMLXRuntime()
        let net = try NFKMLXAdaIN.network(encoderURL: nil, decoderURL: nil)
        func snapshot(_ module: Module) -> [String: [Float]] {
            Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { ($0.0, $0.1.asArray(Float.self)) })
        }
        let encoderBefore = snapshot(net.encoder), decoderBefore = snapshot(net.decoder)
        let content = NFKMLXStyleTransferTrainingTests.images(batch: 2, size: 32)
        let style = NFKMLXStyleTransferTrainingTests.images(batch: 2, size: 32, phase: 2)
        let history = try NFKMLXAdaIN.fineTune(net, examples: { _ in (content, style) }, augmentation: nil,
                                               optimizer: Adam(learningRate: 1e-3), steps: 20)
        XCTAssertLessThan(history.last!, history.first!, "the objective fell: \(history.first!) -> \(history.last!)")
        XCTAssertEqual(snapshot(net.encoder), encoderBefore, "the encoder is frozen")
        let decoderAfter = snapshot(net.decoder)
        for (name, before) in decoderBefore {
            XCTAssertNotEqual(decoderAfter[name], before, "\(name) trains")
        }
    }

    func testAFineTunedDecoderReloadsThroughThePublicFactories() throws {
        try requireMLXRuntime()
        let directory = FileManager.default.temporaryDirectory
        let encoderURL = directory.appendingPathComponent("adain-encoder-\(UUID().uuidString).safetensors")
        let decoderURL = directory.appendingPathComponent("adain-decoder-\(UUID().uuidString).safetensors")
        defer {
            try? FileManager.default.removeItem(at: encoderURL)
            try? FileManager.default.removeItem(at: decoderURL)
        }

        let net = try NFKMLXAdaIN.network(encoderURL: nil, decoderURL: nil)
        let content = NFKMLXStyleTransferTrainingTests.images(batch: 1, size: 32)
        let style = NFKMLXStyleTransferTrainingTests.images(batch: 1, size: 32, phase: 2)
        try NFKMLXAdaIN.fineTune(net, examples: { _ in (content, style) }, augmentation: nil, steps: 1)
        try NFKMLXWeights.save(net.encoder, to: encoderURL)
        try NFKMLXWeights.save(net.decoder, to: decoderURL)

        let reloaded = try NFKMLXAdaIN.network(encoderURL: encoderURL, decoderURL: decoderURL)
        let expected = net.stylize(content: content, style: style)
        let actual = reloaded.stylize(content: content, style: style)
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self), "the fine-tuned decoder round-trips exactly")

        let backend = try NFKMLXAdaIN.backend(encoderURL: encoderURL, decoderURL: decoderURL)
        let image = NFKMLXZeroDCETrainingTests.darkImage(32)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image, NFKInputControl: image]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage), "the backend stylizes with the trained decoder")
    }
}
