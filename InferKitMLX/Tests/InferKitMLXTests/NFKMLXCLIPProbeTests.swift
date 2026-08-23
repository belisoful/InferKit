//
//  NFKMLXCLIPProbeTests.swift
//  InferKitMLXTests
//
//  A consumer's own image classifier over a frozen CLIP embedding. The claim worth testing is the one
//  a user would notice: a handful of examples per class is enough to separate them, and the trained
//  probe answers through an ordinary backend.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXCLIPProbeTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func setUp() {
        super.setUp()
        NFKMLXRandom.seed(20_260_814)
    }

    /// A small CLIP: the full ViT-B/32 is far too slow to encode in a test.
    private func smallCLIP() -> NFKMLXCLIPNet {
        NFKMLXCLIPNet(NFKMLXCLIPConfiguration(imageResolution: 32, patchSize: 16, visionWidth: 32,
                                              visionLayers: 1, visionHeads: 2, embedDimensions: 16,
                                              vocabularySize: 64, contextLength: 16, textWidth: 32,
                                              textLayers: 1, textHeads: 2))
    }

    /// Two visually distinct groups, standing in for a consumer's two categories.
    private func examples(perClass: Int) -> (images: [CGImage], labels: MLXArray) {
        var images: [CGImage] = []
        var indices: [Int32] = []
        for item in 0 ..< perClass {
            images.append(Self.patterned(32, seed: item, vertical: true))
            indices.append(0)
            images.append(Self.patterned(32, seed: item, vertical: false))
            indices.append(1)
        }
        return (images, MLXArray(indices))
    }

    // MARK: - Encoding

    func testEncodingCachesOneEmbeddingPerImage() throws {
        try requireMLXRuntime()
        let net = smallCLIP()
        let (images, _) = examples(perClass: 3)
        let cached = try NFKMLXCLIP.embeddings(for: images, using: net)
        XCTAssertEqual(cached.shape, [6, 16], "one row per image, at the embedding width")
    }

    func testEncodingIsDeterministicSoTheCacheIsReusable() throws {
        try requireMLXRuntime()
        let net = smallCLIP()
        let (images, _) = examples(perClass: 2)
        let first = try NFKMLXCLIP.embeddings(for: images, using: net)
        let second = try NFKMLXCLIP.embeddings(for: images, using: net)
        eval(first, second)
        // The whole saving is that an embedding never changes while the towers are frozen.
        XCTAssertEqual(first.asArray(Float.self), second.asArray(Float.self))
    }

    func testEncodingNoImagesFails() throws {
        try requireMLXRuntime()
        XCTAssertThrowsError(try NFKMLXCLIP.embeddings(for: [], using: smallCLIP()))
    }

    // MARK: - Training

    func testAProbeSeparatesTwoCategoriesFromAFewExamples() throws {
        try requireMLXRuntime()
        let net = smallCLIP()
        let (images, labels) = examples(perClass: 4)
        let cached = try NFKMLXCLIP.embeddings(for: images, using: net)

        let probe = NFKMLXCLIPProbe(embedDimensions: 16, classCount: 2)
        let history = try NFKMLXCLIP.trainProbe(probe, embeddings: cached, labels: labels,
                                                optimizer: AdamW(learningRate: 5e-2), steps: 200)

        XCTAssertLessThan(history.last!, history.first!)
        let predicted = probe(cached).argMax(axis: -1).asArray(Int32.self)
        XCTAssertEqual(predicted, labels.asArray(Int32.self),
                       "eight examples were enough to learn the consumer's two categories")
    }

    func testMismatchedEmbeddingsAndLabelsFail() throws {
        try requireMLXRuntime()
        let probe = NFKMLXCLIPProbe(embedDimensions: 16, classCount: 2)
        XCTAssertThrowsError(
            try NFKMLXCLIP.trainProbe(probe, embeddings: MLXArray.zeros([4, 16]),
                                      labels: MLXArray([Int32(0), 1]), steps: 1)
        ) { error in
            guard case NFKMLXError.trainingDataMismatch(let detail) = error else {
                return XCTFail("expected trainingDataMismatch, got \(error)")
            }
            XCTAssertTrue(detail.contains("4"), "names both counts: \(detail)")
        }
    }

    func testTheSamplerDrawsMinibatches() throws {
        try requireMLXRuntime()
        let net = smallCLIP()
        let (images, labels) = examples(perClass: 4)
        let cached = try NFKMLXCLIP.embeddings(for: images, using: net)

        let probe = NFKMLXCLIPProbe(embedDimensions: 16, classCount: 2)
        let history = try NFKMLXCLIP.trainProbe(probe, embeddings: cached, labels: labels,
                                                sampler: NFKMLXBatchSampler(count: 8, batchSize: 4, seed: 5),
                                                optimizer: AdamW(learningRate: 5e-2), steps: 100)
        XCTAssertEqual(history.count, 100)
        XCTAssertTrue(history.allSatisfy { $0.isFinite })
    }

    // MARK: - The backend

    func testTheTrainedProbeAnswersThroughABackend() throws {
        try requireMLXRuntime()
        let net = smallCLIP()
        let (images, labels) = examples(perClass: 4)
        let cached = try NFKMLXCLIP.embeddings(for: images, using: net)

        let probe = NFKMLXCLIPProbe(embedDimensions: 16, classCount: 2)
        try NFKMLXCLIP.trainProbe(probe, embeddings: cached, labels: labels,
                                  optimizer: AdamW(learningRate: 5e-2), steps: 200)

        let backend = NFKMLXCLIP.probeBackend(net: net, probe: probe, labels: ["vertical", "horizontal"])
        let result = try backend.runInference(
            for: NFKInferenceRequest(inputs: [NFKInputImage: Self.patterned(32, seed: 9, vertical: true)]))

        let ranked = try XCTUnwrap(result.classifications)
        XCTAssertEqual(ranked.count, 2, "every class is ranked")
        XCTAssertEqual(ranked[0].label, "vertical", "the most confident class is the right one")
        XCTAssertGreaterThan(ranked[0].confidence, ranked[1].confidence, "ranked most confident first")
        let total = ranked.reduce(0.0) { $0 + $1.confidence }
        XCTAssertEqual(total, 1.0, accuracy: 1e-4, "confidences are a distribution over the categories")
    }

    func testTheBackendRejectsARequestWithNoImage() throws {
        try requireMLXRuntime()
        let backend = NFKMLXCLIP.probeBackend(net: smallCLIP(),
                                              probe: NFKMLXCLIPProbe(embedDimensions: 16, classCount: 2))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:])))
    }

    // MARK: - Saving

    func testATrainedProbeSavesAndReloads() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("clip-probe-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = smallCLIP()
        let (images, labels) = examples(perClass: 3)
        let cached = try NFKMLXCLIP.embeddings(for: images, using: net)
        let probe = NFKMLXCLIPProbe(embedDimensions: 16, classCount: 2)
        try NFKMLXCLIP.trainProbe(probe, embeddings: cached, labels: labels,
                                  optimizer: AdamW(learningRate: 5e-2), steps: 50)

        // A probe is a separate small model, so what it saves is a companion file rather than
        // modified CLIP weights.
        try NFKMLXWeights.save(probe, to: url)
        let reloaded = NFKMLXCLIPProbe(embedDimensions: 16, classCount: 2)
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        try NFKMLXWeights.apply(Array(checkpoint.arrays), to: reloaded)

        let expected = probe(cached)
        let actual = reloaded(cached)
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    // MARK: - Helpers

    /// A striped plate. The stripe direction is the visual difference the probe has to pick up.
    static func patterned(_ side: Int, seed: Int, vertical: Bool) -> CGImage {
        var pixels = [UInt8](repeating: 255, count: side * side * 4)
        for row in 0 ..< side {
            for column in 0 ..< side {
                let position = vertical ? column : row
                let level = UInt8(((position + seed) % 8) < 4 ? 210 : 45)
                let offset = (row * side + column) * 4
                pixels[offset] = level
                pixels[offset + 1] = level
                pixels[offset + 2] = level
            }
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)!
    }
}
