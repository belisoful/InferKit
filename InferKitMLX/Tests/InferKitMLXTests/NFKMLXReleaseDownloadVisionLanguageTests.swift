//
//  NFKMLXReleaseDownloadVisionLanguageTests.swift
//  InferKitMLXTests
//
//  The download factories of the vision-language and document models, served from the local store
//  with the network refused. A model whose weights fit builds through its own factory; a release
//  above about 8 GB downloads through the class's lists and loads its configuration and tokenizer.
//

import XCTest
import CoreGraphics
import InferKit
@testable import InferKitMLX

final class NFKMLXReleaseDownloadVisionLanguageTests: XCTestCase {

    private var config: [String: String] { NFKMLXValidationConfig.environment }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func store(_ key: String) throws -> URL {
        guard let path = config[key] else { throw XCTSkip("set \(key)") }
        return URL(fileURLWithPath: path)
    }

    private static func patternImage(width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let offset = (y * width + x) * 4
                bytes[offset] = UInt8((x * 255) / max(width - 1, 1))
                bytes[offset + 1] = UInt8((y * 255) / max(height - 1, 1))
                bytes[offset + 2] = (x / 16 + y / 16) % 2 == 0 ? 220 : 30
                bytes[offset + 3] = 255
            }
        }
        let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return context.makeImage()!
    }

    func testSmolVLM256MBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_SMOLVLM2_256M")
        let repo = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        let model = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                   required: NFKMLXSmolVLM.requiredFiles,
                                                   optional: NFKMLXSmolVLM.optionalFiles,
                                                   weights: NFKMLXSmolVLM.weightFiles) {
            try NFKMLXSmolVLM.load(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        let answer = model.answer(image: Self.patternImage(width: 256, height: 256),
                                  question: "What is in the image?", maxTokens: 3)
        XCTAssertFalse(answer.isEmpty, "the downloaded tokenizer encodes the prompt and decodes the answer")
    }

    func testQwen3VL2BBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_QWEN3_VL")
        let repo = "Qwen/Qwen3-VL-2B-Instruct"
        let model = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                   required: NFKMLXQwen3VL.requiredFiles,
                                                   optional: NFKMLXQwen3VL.optionalFiles,
                                                   weights: NFKMLXQwen3VL.weightFiles) {
            try NFKMLXQwen3VL.model(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        let answer = model.answer(image: Self.patternImage(width: 128, height: 128),
                                  question: "What is in the image?", maxTokens: 3)
        XCTAssertFalse(answer.isEmpty, "the downloaded tokenizer encodes the prompt and decodes the answer")
    }

    // The 12B's weights are about 25 GB, so the download is exercised and the model is not built.
    func testPixtralDownloadsFromItsListsOfflineWithoutBuilding() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_PIXTRAL")
        let repo = "mistral-experimental/pixtral-12b"
        try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                       required: NFKMLXPixtral.requiredFiles,
                                       optional: NFKMLXPixtral.optionalFiles,
                                       weights: NFKMLXPixtral.weightFiles) {
            let directory = try NFKMLXReleaseDownload.directory(
                repo: repo, revision: nil, cacheDirectoryURL: nil, required: NFKMLXPixtral.requiredFiles,
                optional: NFKMLXPixtral.optionalFiles, weights: NFKMLXPixtral.weightFiles)
            let decoder = try NFKMLXPixtral.decoderConfiguration(directoryURL: directory)
            XCTAssertEqual(decoder.hiddenSize, 5120)
            XCTAssertEqual(decoder.headCount, 32)
            let vision = try NFKMLXPixtralVisionConfiguration.configuration(
                fromHuggingFace: directory.appendingPathComponent("config.json"))
            XCTAssertEqual(vision.hiddenSize, 1024)
            XCTAssertNotNil(NFKMLXLanguage.releaseTokenizer(inDirectory: directory))
            XCTAssertEqual(try NFKMLXReleaseWeights.files(inDirectory: directory).count, 6,
                           "every shard the index names is in the cache")
        }
    }

    func testFlorence2BuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_FLORENCE2").deletingLastPathComponent()
        let repo = "microsoft/Florence-2-large"
        let backend = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                     required: NFKMLXFlorence2.requiredFiles,
                                                     optional: NFKMLXFlorence2.optionalFiles,
                                                     weights: NFKMLXFlorence2.weightFiles) {
            try NFKMLXFlorence2.backend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        let maximumTokens = NFKMLXFlorence2.maximumTokens
        NFKMLXFlorence2.maximumTokens = 4
        defer { NFKMLXFlorence2.maximumTokens = maximumTokens }
        let request = NFKInferenceRequest(inputs: [NFKInputImage: Self.patternImage(width: 256, height: 256),
                                                   NFKInputPrompt: "<OD>"])
        let result = try backend.runInference(for: request)
        XCTAssertNotNil(result.output(forKey: NFKOutputText) as? String)
    }

    // Sa2VA's weights are about 15 GB, so the download is exercised and the model is not built.
    func testSa2VADownloadsFromItsListsOfflineWithoutBuilding() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_SA2VA")
        let repo = "ByteDance/Sa2VA-4B"
        try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                       required: NFKMLXSa2VA.requiredFiles,
                                       optional: NFKMLXSa2VA.optionalFiles,
                                       weights: NFKMLXSa2VA.weightFiles) {
            let directory = try NFKMLXReleaseDownload.directory(
                repo: repo, revision: nil, cacheDirectoryURL: nil, required: NFKMLXSa2VA.requiredFiles,
                optional: NFKMLXSa2VA.optionalFiles, weights: NFKMLXSa2VA.weightFiles)
            let configuration = try NFKMLXSa2VANet.configuration(fromDirectory: directory)
            XCTAssertEqual(configuration.imageContextTokenId, 151_667)
            XCTAssertEqual(configuration.segmentationTokenId, 151_674)
            XCTAssertNotNil(NFKMLXLanguage.releaseTokenizer(inDirectory: directory))
            XCTAssertEqual(try NFKMLXReleaseWeights.files(inDirectory: directory).count, 4,
                           "every shard the index names is in the cache")
        }
    }

    func testTrOCRBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_TROCR")
        let repo = "microsoft/trocr-base-handwritten"
        let backend = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                     required: NFKMLXTrOCR.requiredFiles,
                                                     optional: NFKMLXTrOCR.optionalFiles,
                                                     weights: NFKMLXTrOCR.weightFiles) {
            try NFKMLXTrOCR.backend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        let maximumTokens = NFKMLXTrOCR.maximumTokens
        NFKMLXTrOCR.maximumTokens = 4
        defer { NFKMLXTrOCR.maximumTokens = maximumTokens }
        let request = NFKInferenceRequest(inputs: [NFKInputImage: Self.patternImage(width: 384, height: 96)])
        let result = try backend.runInference(for: request)
        XCTAssertNotNil(result.output(forKey: NFKOutputText) as? String)
    }

    func testTableTransformerBuildsFromItsListsOfflineThroughTheCompletionHandler() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_TABLE_TRANSFORMER")
        let repo = "microsoft/table-transformer-structure-recognition"
        let backend = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                     required: NFKMLXTableTransformer.requiredFiles,
                                                     optional: NFKMLXTableTransformer.optionalFiles,
                                                     weights: NFKMLXTableTransformer.weightFiles) {
            () throws -> NFKMLXTableTransformerBackend in
            let built = expectation(description: "the backend is built")
            var backend: NFKMLXTableTransformerBackend?
            var failure: Error?
            NFKMLXTableTransformer.backend(repo: repo, revision: nil, cacheDirectoryURL: nil) { result, error in
                backend = result
                failure = error
                built.fulfill()
            }
            wait(for: [built], timeout: 300)
            if let failure { throw failure }
            return try XCTUnwrap(backend)
        }
        XCTAssertTrue(backend.isReady)
        let request = NFKInferenceRequest(inputs: [NFKInputImage: Self.patternImage(width: 320, height: 240)])
        XCTAssertNoThrow(try backend.runInference(for: request))
    }

    func testVJEPA2BuildsFromItsListsOfflineThroughTheCompletionHandler() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_VJEPA2")
        let repo = "facebook/vjepa2-vitl-fpc64-256"
        let backend = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                     required: NFKMLXVJEPA2.requiredFiles,
                                                     optional: NFKMLXVJEPA2.optionalFiles,
                                                     weights: NFKMLXVJEPA2.weightFiles) {
            () throws -> NFKMLXVJEPA2Backend in
            let built = expectation(description: "the backend is built")
            var backend: NFKMLXVJEPA2Backend?
            var failure: Error?
            NFKMLXVJEPA2.backend(repo: repo, revision: nil, cacheDirectoryURL: nil) { result, error in
                backend = result
                failure = error
                built.fulfill()
            }
            wait(for: [built], timeout: 300)
            if let failure { throw failure }
            return try XCTUnwrap(backend)
        }
        let request = NFKInferenceRequest(inputs: [NFKInputImage: Self.patternImage(width: 256, height: 256)])
        let result = try backend.runInference(for: request)
        let embedding = try XCTUnwrap(result.output(forKey: NFKOutputEmbedding) as? [NSNumber])
        XCTAssertEqual(embedding.count, 1024, "the ViT-L embedding width")
    }
}
