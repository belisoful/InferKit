//
//  NFKMLXSigLIP2ProbeTests.swift
//  InferKitMLXTests
//
//  A consumer's own image classifier over a frozen SigLIP 2 embedding, through the shared linear probe.
//  The claims a user would notice: a few examples per class separate them, the trained probe answers
//  through an ordinary backend, and a saved probe reloads through the model's own factory.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXSigLIP2ProbeTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        // Seeding initializes MLX's runtime, which needs a Metal library it can find; the methods
        // skip without one.
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_924)
    }

    /// A tiny SigLIP 2 whose pooling attention is random. The fused query/key/value projection is
    /// built as zeros for a checkpoint to fill, and at zero every image pools to the same embedding.
    private func tinyModel() throws -> NFKMLXSigLIP2 {
        let model = try NFKMLXSigLIP2.model(configuration: .tiny, weightsURL: nil)
        let hidden = NFKMLXSigLIP2Configuration.tiny.vision.hiddenSize
        model.holder.net.vision.head.attention.update(parameters: ModuleParameters.unflattened(
            ["in_proj_weight": MLXRandom.normal([3 * hidden, hidden]) * 0.2]))
        return model
    }

    private func examples(perClass: Int) -> (images: [CGImage], labels: MLXArray) {
        var images: [CGImage] = []
        var indices: [Int32] = []
        for item in 0 ..< perClass {
            images.append(NFKMLXCLIPProbeTests.patterned(64, seed: item, vertical: true))
            indices.append(0)
            images.append(NFKMLXCLIPProbeTests.patterned(64, seed: item, vertical: false))
            indices.append(1)
        }
        return (images, MLXArray(indices))
    }

    private func trainedProbe(_ model: NFKMLXSigLIP2, steps: Int) throws -> (NFKMLXEmbeddingProbe, MLXArray) {
        let (images, labels) = examples(perClass: 4)
        let cached = try model.imageEmbeddings(for: images)
        let probe = NFKMLXEmbeddingProbe(embedDimensions: model.embeddingDimensions, classCount: 2)
        try NFKMLXEmbeddingProbe.train(probe, embeddings: cached, labels: labels,
                                       optimizer: AdamW(learningRate: 5e-2), steps: steps)
        return (probe, cached)
    }

    func testEncodingCachesOneUnitLengthEmbeddingPerImage() throws {
        try requireMLXRuntime()
        let model = try tinyModel()
        let (images, _) = examples(perClass: 3)
        let cached = try model.imageEmbeddings(for: images)
        XCTAssertEqual(cached.shape, [6, model.embeddingDimensions])
        let norms = sqrt(cached.square().sum(axis: -1)).asArray(Float.self)
        for norm in norms {
            XCTAssertEqual(norm, 1, accuracy: 1e-4)
        }
    }

    func testEncodingNoImagesFails() throws {
        try requireMLXRuntime()
        XCTAssertThrowsError(try tinyModel().imageEmbeddings(for: []))
    }

    func testAProbeSeparatesTwoCategoriesAndLowersItsLoss() throws {
        try requireMLXRuntime()
        let model = try tinyModel()
        let (images, labels) = examples(perClass: 4)
        let cached = try model.imageEmbeddings(for: images)
        // The two categories must embed apart, or no probe could separate them.
        let gap = abs(cached[0] - cached[1]).max().item(Float.self)
        XCTAssertGreaterThan(gap, 1e-3, "the tiny tower embeds the two categories differently")
        let probe = NFKMLXEmbeddingProbe(embedDimensions: model.embeddingDimensions, classCount: 2)
        let history = try NFKMLXEmbeddingProbe.train(probe, embeddings: cached, labels: labels,
                                                     optimizer: AdamW(learningRate: 5e-2), steps: 200)
        XCTAssertLessThan(history.last!, history.first!)
        let predicted = argMax(probe(cached), axis: -1).asArray(Int32.self)
        XCTAssertEqual(predicted, labels.asArray(Int32.self))
    }

    func testTheTrainedProbeAnswersThroughABackend() throws {
        try requireMLXRuntime()
        let model = try tinyModel()
        let (probe, _) = try trainedProbe(model, steps: 200)
        let backend = try model.probeBackend(probe: probe, labels: ["vertical", "horizontal"])
        let result = try backend.runInference(
            for: NFKInferenceRequest(inputs: [NFKInputImage: NFKMLXCLIPProbeTests.patterned(64, seed: 9, vertical: true)]))
        let ranked = try XCTUnwrap(result.classifications)
        XCTAssertEqual(ranked.count, 2)
        XCTAssertEqual(ranked[0].label, "vertical")
        XCTAssertEqual(ranked.reduce(0.0) { $0 + $1.confidence }, 1.0, accuracy: 1e-4)
    }

    func testAProbeOfTheWrongWidthIsRefused() throws {
        try requireMLXRuntime()
        let model = try tinyModel()
        let probe = NFKMLXEmbeddingProbe(embedDimensions: model.embeddingDimensions + 1, classCount: 2)
        XCTAssertThrowsError(try model.probeBackend(probe: probe))
    }

    func testASavedProbeReloadsThroughTheModelsFactory() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("siglip2-probe-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let model = try tinyModel()
        let (probe, _) = try trainedProbe(model, steps: 100)
        try NFKMLXWeights.save(probe, to: url)

        let image = NFKMLXCLIPProbeTests.patterned(64, seed: 5, vertical: false)
        let request = NFKInferenceRequest(inputs: [NFKInputImage: image])
        let expected = try XCTUnwrap(try model.probeBackend(probe: probe).runInference(for: request).classifications)
        let actual = try XCTUnwrap(try model.probeBackend(probeURL: url, labels: nil).runInference(for: request).classifications)
        XCTAssertEqual(actual.map(\.classIndex), expected.map(\.classIndex))
        for (a, e) in zip(actual, expected) {
            XCTAssertEqual(a.confidence, e.confidence, accuracy: 1e-6)
        }
    }
}
