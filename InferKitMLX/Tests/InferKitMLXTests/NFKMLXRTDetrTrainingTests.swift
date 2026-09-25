//
//  NFKMLXRTDetrTrainingTests.swift
//  InferKitMLXTests
//
//  RT-DETR's training forward on constructed inputs: the denoising group's layout and noise, a batch
//  against its images one at a time, the split between denoising and matching queries, and the
//  gradient paths the original implementation trains through. Parity against transformers lives in
//  NFKMLXReferenceParityTests.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXRTDetrTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_924)
    }

    private let targets = [
        NFKMLXRTDetrTarget(classes: [2, 0], boxes: MLXArray([0.3, 0.3, 0.2, 0.2, 0.7, 0.6, 0.3, 0.4] as [Float], [2, 4])),
        NFKMLXRTDetrTarget(classes: [1], boxes: MLXArray([0.5, 0.5, 0.4, 0.4] as [Float], [1, 4])),
    ]

    /// Draws that keep every class and move every corner outward by `magnitude` of its half-size.
    private func draws(magnitude: Float) -> (Int, Int) -> NFKMLXRTDetrDenoisingGroup.Draws {
        { batch, count in
            NFKMLXRTDetrDenoisingGroup.Draws(
                labelChance: [Float](repeating: 1, count: batch * count),
                newLabels: [Int](repeating: 3, count: batch * count),
                signs: (0 ..< batch * count * 4).map { $0 % 4 < 2 ? -1 : 1 },
                magnitudes: [Float](repeating: magnitude, count: batch * count * 4))
        }
    }

    private func group(magnitude: Float, denoisingQueries: Int = 8, queryCount: Int = 10) throws -> NFKMLXRTDetrDenoisingGroup {
        try XCTUnwrap(NFKMLXRTDetrDenoisingGroup.make(
            targets: targets, classCount: 4, queryCount: queryCount, denoisingQueries: denoisingQueries,
            labelNoiseRatio: 0.5, boxNoiseScale: 1, draw: draws(magnitude: magnitude)))
    }

    func testTheDenoisingGroupLaysOutPositivesPaddingAndTheMask() throws {
        try requireMLXRuntime()
        let group = try group(magnitude: 0)
        XCTAssertEqual(group.groups, 4)
        XCTAssertEqual(group.count, 16)
        XCTAssertEqual(group.positives, [[0, 1, 4, 5, 8, 9, 12, 13], [0, 4, 8, 12]])
        let classes = group.classes.asArray(Int32.self)
        XCTAssertEqual(Array(classes[0 ..< 4]), [2, 0, 2, 0])
        XCTAssertEqual(Array(classes[16 ..< 20]), [1, 4, 1, 4], "the second image pads its missing box")

        let mask = group.mask.asArray(Float.self)
        let total = 26
        XCTAssertEqual(mask[16 * total + 0], -.infinity, "a matching query cannot read a denoising query")
        XCTAssertEqual(mask[16 * total + 16], 0)
        XCTAssertEqual(mask[0 * total + 3], 0, "a group reads itself")
        XCTAssertEqual(mask[0 * total + 4], -.infinity, "a group cannot read the next")
        XCTAssertEqual(mask[5 * total + 3], -.infinity, "or the previous")
        XCTAssertEqual(mask[5 * total + 20], 0, "every denoising query reads the matching queries")
    }

    func testUnmovedCopiesDecodeToTheTruth() throws {
        try requireMLXRuntime()
        let boxes = sigmoid(try group(magnitude: 0).boxesUnactivated).asArray(Float.self)
        for (got, expected) in zip(boxes[0 ..< 4], [Float(0.3), 0.3, 0.2, 0.2]) {
            XCTAssertEqual(got, expected, accuracy: 1e-5)
        }
    }

    func testANegativeCopyMovesFartherThanItsPositive() throws {
        try requireMLXRuntime()
        // Every corner moves outward by 0.5 of a half-size in a positive copy and 1.5 in a negative,
        // so the first box (0.2 wide) grows to 0.3 and to 0.5.
        let boxes = sigmoid(try group(magnitude: 0.5).boxesUnactivated).asArray(Float.self)
        XCTAssertEqual(boxes[2], 0.3, accuracy: 1e-5)
        XCTAssertEqual(boxes[2 * 4 + 2], 0.5, accuracy: 1e-5)
        XCTAssertEqual(boxes[0], 0.3, accuracy: 1e-5, "the center stays put when both sides move outward")
    }

    func testAnImageSetWithoutBoxesHasNoGroup() throws {
        try requireMLXRuntime()
        let empty = NFKMLXRTDetrTarget(classes: [], boxes: MLXArray.zeros([0, 4]))
        XCTAssertNil(NFKMLXRTDetrDenoisingGroup.make(targets: [empty], classCount: 4, queryCount: 10))
    }

    private func network(denoisingQueries: Int = 0) -> NFKMLXRTDetrNet {
        var configuration = NFKMLXRTDetrConfiguration.tiny
        configuration.denoisingQueries = denoisingQueries
        let net = NFKMLXRTDetrNet(configuration)
        net.train(false)
        return net
    }

    func testTheBatchedForwardMatchesEachImageAlone() throws {
        try requireMLXRuntime()
        let net = network()
        let pixels = MLXRandom.uniform(0 ..< 1, [2, 64, 64, 3])
        let batched = net.trainingOutputs(pixels, denoising: nil)
        for image in 0 ..< 2 {
            let alone = net.trainingOutputs(pixels[image ..< image + 1], denoising: nil)
            XCTAssertLessThan(abs(batched.final.logits[image ..< image + 1] - alone.final.logits).max().item(Float.self), 1e-5)
            XCTAssertLessThan(abs(batched.final.boxes[image ..< image + 1] - alone.final.boxes).max().item(Float.self), 1e-5)
        }
        let inference = net(pixels[0 ..< 1])
        XCTAssertLessThan(abs(batched.final.logits[0] - inference.logits).max().item(Float.self), 1e-5,
                          "the training forward selects and decodes as inference does")
    }

    func testTheTrainingForwardSplitsDenoisingFromMatchingQueries() throws {
        try requireMLXRuntime()
        let net = network(denoisingQueries: 8)
        let group = try group(magnitude: 0.3)
        let outputs = net.trainingOutputs(MLXRandom.uniform(0 ..< 1, [2, 64, 64, 3]), denoising: group)
        XCTAssertEqual(outputs.final.logits.shape, [2, 10, 4])
        XCTAssertEqual(outputs.auxiliary.count, 2, "one earlier decoder layer and the encoder's proposals")
        XCTAssertEqual(outputs.denoising.map(\.boxes.shape), [[2, 16, 4], [2, 16, 4]])
        let loss = NFKMLXRTDetrObjective().loss(outputs, denoising: group, targets: targets).item(Float.self)
        XCTAssertTrue(loss.isFinite)
    }

    func testTheDenoisingQueriesCannotReachTheMatchingOnes() throws {
        try requireMLXRuntime()
        let net = network(denoisingQueries: 8)
        let pixels = MLXRandom.uniform(0 ..< 1, [2, 64, 64, 3])
        let plain = net.trainingOutputs(pixels, denoising: nil)
        let withGroup = net.trainingOutputs(pixels, denoising: try group(magnitude: 0.3))
        XCTAssertLessThan(abs(plain.final.logits - withGroup.final.logits).max().item(Float.self), 1e-5)
    }

    func testALayersBoxLossReachesThePreviousBoxHeadAndNotThePaddingClass() throws {
        try requireMLXRuntime()
        let net = network(denoisingQueries: 8)
        let group = try group(magnitude: 0.3)
        let pixels = MLXRandom.uniform(0 ..< 1, [2, 64, 64, 3])
        let gradients = valueAndGrad(model: net) { net, arrays -> [MLXArray] in
            let outputs = net.trainingOutputs(arrays[0], denoising: group)
            return [outputs.final.boxes.sum() + outputs.denoising[0].logits.sum()]
        }(net, [pixels]).1.flattened()
        let byName = Dictionary(gradients, uniquingKeysWith: { first, _ in first })

        let previousHead = try XCTUnwrap(byName["decoder.bbox_embed.0.layers.2.weight"])
        XCTAssertGreaterThan(abs(previousHead).max().item(Float.self), 0,
                             "the final boxes refine the previous layer's undetached boxes")
        let embedding = try XCTUnwrap(byName["denoising_class_embed.weight"])
        XCTAssertEqual(abs(embedding[4]).max().item(Float.self), 0, "the padding class never updates")
        XCTAssertGreaterThan(abs(embedding[2]).max().item(Float.self), 0)
        XCTAssertEqual(abs(try XCTUnwrap(byName["enc_output.0.weight"])).max().item(Float.self), 0,
                       "the selected queries' features are detached")
    }

    // MARK: The recipe

    func testTheOriginalNamesPlaceEachParameterInItsReferenceGroup() {
        let recipe = NFKMLXRTDetr.referenceRecipe(for: .r50vd)
        let expected: [(String, Float, Float)] = [
            ("backbone.model.encoder.stages.0.layers.0.layer.0.convolution.weight", 0.1, 1e-4),
            ("encoder_input_proj.0.0.weight", 1, 1e-4),
            ("encoder_input_proj.0.1.weight", 1, 1e-4),                // the original's `input_proj.0.1` decays
            ("encoder_input_proj.0.1.bias", 1, 0),
            ("encoder.encoder.0.layers.0.fc1.weight", 1, 1e-4),
            ("encoder.encoder.0.layers.0.self_attn.q_proj.bias", 1, 0),
            ("encoder.encoder.0.layers.0.self_attn_layer_norm.weight", 1, 0),
            ("encoder.lateral_convs.0.norm.weight", 1, 0),
            ("encoder.fpn_blocks.0.bottlenecks.0.conv1.conv.weight", 1, 1e-4),
            ("decoder_input_proj.0.0.weight", 1, 1e-4),
            ("decoder_input_proj.0.1.weight", 1, 0),                   // the original's `input_proj.0.norm`
            ("enc_output.0.bias", 1, 0),
            ("enc_output.1.weight", 1, 1e-4),                          // the original's `enc_output.1` decays
            ("enc_score_head.weight", 1, 1e-4),
            ("decoder.layers.0.encoder_attn.sampling_offsets.weight", 1, 1e-4),
            ("decoder.layers.0.encoder_attn.sampling_offsets.bias", 1, 0),
            ("decoder.layers.0.final_layer_norm.weight", 1, 0),
            ("decoder.bbox_embed.0.layers.2.bias", 1, 0),
            ("denoising_class_embed.weight", 1, 1e-4),
        ]
        for (key, rate, decay) in expected {
            let group = NFKMLXRTDetr.referenceGroup(for: key, recipe: recipe)
            XCTAssertEqual(group.rateScale, rate, key)
            XCTAssertEqual(group.weightDecay, decay, key)
        }
    }

    func testRTDetrV2NamesItsProjectionsSoTheirNormalizationsSkipDecay() {
        let recipe = NFKMLXRTDetr.referenceRecipe(for: .v2R50VD)
        XCTAssertEqual(NFKMLXRTDetr.originalParameterName("encoder_input_proj.2.1.weight", version: 2),
                       "encoder.input_proj.2.norm.weight")
        XCTAssertEqual(NFKMLXRTDetr.originalParameterName("enc_output.1.bias", version: 2), "decoder.enc_output.norm.bias")
        XCTAssertEqual(NFKMLXRTDetr.referenceGroup(for: "encoder_input_proj.2.1.weight", recipe: recipe).weightDecay, 0)
        XCTAssertEqual(NFKMLXRTDetr.referenceGroup(for: "enc_output.1.weight", recipe: recipe).weightDecay, 0)
        XCTAssertEqual(NFKMLXRTDetr.referenceGroup(for: "enc_output.0.bias", recipe: recipe).weightDecay, 1e-4,
                       "v2 decays biases")
        XCTAssertEqual(NFKMLXRTDetr.referenceRecipe(for: .v2R18VD).warmupSteps, 2000)
        XCTAssertEqual(NFKMLXRTDetr.referenceRecipe(for: .r18vd).warmupSteps, 0)
    }

    private func tinyNetwork(classCount: Int = 4) -> NFKMLXRTDetrNet {
        var configuration = NFKMLXRTDetrConfiguration.tiny
        configuration.numLabels = classCount
        configuration.denoisingQueries = 12
        let net = NFKMLXRTDetrNet(configuration)
        net.initializeHeads()
        return net
    }

    func testFreezingKeepsTheStemAndTheBackboneNormalizationsFixed() throws {
        try requireMLXRuntime()
        let net = tinyNetwork()
        let recipe = NFKMLXRTDetr.referenceRecipe(for: .r50vd)
        NFKMLXRTDetr.applyFreezing(.everything, recipe: recipe, to: net)
        let trainable = Set(net.trainableParameters().flattened().map(\.0))
        XCTAssertFalse(trainable.contains { $0.hasPrefix("backbone.model.embedder.") }, "the stem")
        XCTAssertFalse(trainable.contains { $0.hasPrefix("backbone.") && $0.contains(".normalization.") })
        XCTAssertTrue(trainable.contains("backbone.model.encoder.stages.0.layers.0.layer.0.convolution.weight"))
        XCTAssertTrue(trainable.contains("encoder.lateral_convs.0.norm.weight"))
        XCTAssertFalse(trainable.contains { $0.hasSuffix("running_mean") || $0.hasSuffix("running_var") })

        NFKMLXRTDetr.applyFreezing(.decoder, recipe: recipe, to: net)
        let decoderOnly = Set(net.trainableParameters().flattened().map(\.0))
        XCTAssertFalse(decoderOnly.contains { $0.hasPrefix("backbone.") || $0.hasPrefix("encoder.") || $0.hasPrefix("encoder_input_proj.") })
        XCTAssertTrue(decoderOnly.contains("decoder_input_proj.0.0.weight"))
        XCTAssertTrue(decoderOnly.contains("denoising_class_embed.weight"))

        NFKMLXRTDetr.applyFreezing(.everything, recipe: NFKMLXRTDetr.referenceRecipe(for: .r18vd), to: net)
        let smallRelease = Set(net.trainableParameters().flattened().map(\.0))
        XCTAssertTrue(smallRelease.contains { $0.hasPrefix("backbone.model.embedder.") }, "r18vd trains its stem")
        XCTAssertTrue(smallRelease.contains { $0.hasPrefix("backbone.") && $0.hasSuffix(".normalization.weight") },
                      "and its backbone's normalizations")
    }

    func testTheHeadsStartAtTheOriginalInitialization() throws {
        try requireMLXRuntime()
        let net = tinyNetwork()
        XCTAssertEqual(net.encScoreHead.bias!.asArray(Float.self)[0], -logf(0.99 / 0.01), accuracy: 1e-5)
        XCTAssertEqual(net.decoder.classEmbed[1].bias!.asArray(Float.self)[3], -logf(0.99 / 0.01), accuracy: 1e-5)
        XCTAssertEqual(abs(net.decoder.bboxEmbed[0].layers[2].weight).max().item(Float.self), 0)
        let embedding = try XCTUnwrap(net.denoisingClassEmbed).weight
        XCTAssertEqual(abs(embedding[4]).max().item(Float.self), 0, "the padding class")
    }

    private func batch() -> (images: MLXArray, targets: [NFKMLXRTDetrTarget]) {
        (MLXRandom.uniform(0 ..< 1, [2, 64, 64, 3]), targets)
    }

    func testAFullRunLowersTheLoss() throws {
        try requireMLXRuntime()
        let net = tinyNetwork()
        let item = batch()
        let history = try NFKMLXRTDetr.fineTune(net, variant: .r50vd, examples: { _ in item }, optimizer: AdamW(learningRate: 1e-3),
                                                steps: 12, clipGradientNorm: nil, learningRateSchedule: .constant,
                                                averagesWeights: false)
        XCTAssertTrue(history.allSatisfy(\.isFinite))
        XCTAssertLessThan(history.suffix(3).reduce(0, +), history.prefix(3).reduce(0, +))
        XCTAssertFalse(net.training)
    }

    func testADecoderRunLeavesTheBackboneAndItsStatisticsUntouched() throws {
        try requireMLXRuntime()
        let net = tinyNetwork()
        eval(net)
        func value(_ key: String) -> [Float] {
            Dictionary(uniqueKeysWithValues: net.parameters().flattened())[key]!.asArray(Float.self)
        }
        let convolution = "backbone.model.encoder.stages.0.layers.0.layer.0.convolution.weight"
        let statistics = "encoder.lateral_convs.0.norm.running_mean"
        let (before, statisticsBefore) = (value(convolution), value(statistics))
        let head = value("decoder.class_embed.0.weight")
        let item = batch()
        try NFKMLXRTDetr.fineTune(net, variant: .r50vd, examples: { _ in item }, trainable: .decoder, optimizer: AdamW(learningRate: 1e-2),
                                  steps: 2, learningRateSchedule: .constant, averagesWeights: false)
        XCTAssertEqual(value(convolution), before)
        XCTAssertEqual(value(statistics), statisticsBefore, "a frozen encoder evaluates")
        XCTAssertNotEqual(value("decoder.class_embed.0.weight"), head)
    }

    func testARetargetTransfersEverythingButTheClassBranches() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rtdetr-coco-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let released = try NFKMLXRTDetr.network(variant: .r18vd, classCount: 80, weightsURL: nil)
        try NFKMLXWeights.save(released, to: url)

        let retargeted = try NFKMLXRTDetr.network(variant: .r18vd, classCount: 3, weightsURL: url)
        XCTAssertEqual(retargeted.encBboxHead.layers[0].weight.asArray(Float.self),
                       released.encBboxHead.layers[0].weight.asArray(Float.self))
        XCTAssertEqual(retargeted.encScoreHead.weight.shape, [3, 256])
        XCTAssertEqual(try XCTUnwrap(retargeted.denoisingClassEmbed).weight.shape, [4, 256])
        XCTAssertEqual(retargeted.encScoreHead.bias!.asArray(Float.self)[0], -logf(0.99 / 0.01), accuracy: 1e-5)
    }

    func testAFineTunedCheckpointLoadsAtItsOwnClassCount() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rtdetr-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXRTDetr.network(variant: .r18vd, classCount: 3, weightsURL: nil)
        // 160 pixels give 525 anchors, enough for the release's 300 queries.
        let item = (images: MLXRandom.uniform(0 ..< 1, [2, 160, 160, 3]),
                    targets: targets.map { NFKMLXRTDetrTarget(classes: $0.classes.map { $0 % 3 }, boxes: $0.boxes) })
        try NFKMLXRTDetr.fineTune(net, variant: .r18vd, examples: { _ in item }, steps: 2)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXRTDetr.network(variant: .r18vd, classCount: 3, weightsURL: url)
        net.train(false)
        reloaded.train(false)
        let image = item.images[0 ..< 1]
        XCTAssertLessThan(abs(reloaded(image).logits - net(image).logits).max().item(Float.self), 1e-4)
        XCTAssertEqual(try NFKMLXRTDetr.classCount(in: url), 3)
        XCTAssertTrue(try NFKMLXRTDetr.backend(variant: .r18vd, weightsURL: url, labels: ["a", "b", "c"]).isReady)
    }
}
