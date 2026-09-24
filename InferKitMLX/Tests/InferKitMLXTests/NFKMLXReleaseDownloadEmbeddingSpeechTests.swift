//
//  NFKMLXReleaseDownloadEmbeddingSpeechTests.swift
//  InferKitMLXTests
//
//  The download factories of the embedding, reranking, and speech models, each served from the local
//  validation store through an offline hub. Only the files a factory's lists name are seeded, so a
//  list that misses a file the model reads fails the build.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXReleaseDownloadEmbeddingSpeechTests: XCTestCase {

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
        guard let path = config[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key)")
        }
        return URL(fileURLWithPath: path)
    }

    private func embedding(_ backend: any NFKInferenceBackend, _ text: String) throws -> [NSNumber] {
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: text]))
        return try XCTUnwrap(result.embedding, "the backend returns an embedding")
    }

    private static let query = "What is the capital of China?"
    private static let relevant = "The capital of China is Beijing."
    private static let irrelevant = "Gravity is a force that attracts two bodies towards each other."

    // MARK: Embedding and reranking

    func testQwen3EmbeddingBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let repo = "Qwen/Qwen3-Embedding-0.6B"
        let backend = try NFKMLXReleaseStore.offline(store: try store("IK_VAL_QWEN3_EMBEDDING"), repo: repo,
                                                     required: NFKMLXQwen3Embedding.requiredFiles,
                                                     optional: NFKMLXQwen3Embedding.optionalFiles,
                                                     weights: NFKMLXQwen3Embedding.weightFiles) {
            try NFKMLXQwen3Embedding.backend(repo: repo, revision: nil, cacheDirectoryURL: nil, outputDimensions: 256)
        }
        XCTAssertEqual(try embedding(backend, Self.relevant).count, 256)
    }

    func testEmbeddingGemmaBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let repo = "unsloth/embeddinggemma-300m"
        let backend = try NFKMLXReleaseStore.offline(store: try store("IK_VAL_EMBEDDINGGEMMA"), repo: repo,
                                                     required: NFKMLXEmbeddingGemma.requiredFiles,
                                                     optional: NFKMLXEmbeddingGemma.optionalFiles,
                                                     weights: NFKMLXEmbeddingGemma.weightFiles) {
            try NFKMLXEmbeddingGemma.backend(repo: repo, revision: nil, cacheDirectoryURL: nil, outputDimensions: 128)
        }
        XCTAssertEqual(try embedding(backend, NFKMLXEmbeddingGemma.document(Self.relevant)).count, 128)
    }

    // The completion-handler form runs the same download on a background queue.
    func testModernBERTRerankerBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let repo = "Alibaba-NLP/gte-reranker-modernbert-base"
        let reranker = try NFKMLXReleaseStore.offline(store: try store("IK_VAL_MODERNBERT_RERANKER"), repo: repo,
                                                      required: NFKMLXModernBERTReranker.requiredFiles,
                                                      optional: NFKMLXModernBERTReranker.optionalFiles,
                                                      weights: NFKMLXModernBERTReranker.weightFiles) {
            () throws -> NFKMLXModernBERTReranker in
            let done = DispatchSemaphore(value: 0)
            var built: NFKMLXModernBERTReranker?
            var failure: Error?
            NFKMLXModernBERTReranker.reranker(repo: repo, revision: nil, cacheDirectoryURL: nil) { reranker, error in
                built = reranker
                failure = error
                done.signal()
            }
            done.wait()
            if let failure {
                throw failure
            }
            return try XCTUnwrap(built)
        }
        let ranked = reranker.rankedIndices(query: Self.query, documents: [Self.irrelevant, Self.relevant])
        XCTAssertEqual(ranked.first?.intValue, 1, "the relevant document ranks first")
    }

    func testQwen3VLEmbedderBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let repo = "Qwen/Qwen3-VL-Embedding-2B"
        let embedder = try NFKMLXReleaseStore.offline(store: try store("IK_VAL_QWEN3_VL_EMBEDDING"), repo: repo,
                                                      required: NFKMLXQwen3VLEmbedder.requiredFiles,
                                                      optional: NFKMLXQwen3VLEmbedder.optionalFiles,
                                                      weights: NFKMLXQwen3VLEmbedder.weightFiles) {
            try NFKMLXQwen3VLEmbedder.embedder(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        let query = embedder.embedding(forText: Self.query).map(\.doubleValue)
        let relevant = embedder.embedding(forText: Self.relevant).map(\.doubleValue)
        let irrelevant = embedder.embedding(forText: Self.irrelevant).map(\.doubleValue)
        XCTAssertEqual(query.count, embedder.embeddingDimensions)
        let similarity = { (a: [Double], b: [Double]) in zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
        XCTAssertGreaterThan(similarity(query, relevant), similarity(query, irrelevant),
                             "the relevant document embeds closer to the query")
    }

    func testQwen3VLRerankerBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let repo = "Qwen/Qwen3-VL-Reranker-2B"
        let reranker = try NFKMLXReleaseStore.offline(store: try store("IK_VAL_QWEN3_VL_RERANKER"), repo: repo,
                                                      required: NFKMLXQwen3VLReranker.requiredFiles,
                                                      optional: NFKMLXQwen3VLReranker.optionalFiles,
                                                      weights: NFKMLXQwen3VLReranker.weightFiles) {
            try NFKMLXQwen3VLReranker.reranker(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        XCTAssertGreaterThan(reranker.score(query: Self.query, document: Self.relevant),
                             reranker.score(query: Self.query, document: Self.irrelevant),
                             "the relevant document scores higher")
    }

    // MARK: Speech

    private func speechAsset() throws -> NFKAudioAsset {
        let path = try store("IK_VAL_AUDIO").path
        return NFKAudioAsset(fileURL: URL(fileURLWithPath: path), durationSeconds: 3.47, sampleRate: 16000, channelCount: 1)
    }

    // The store's unpacked release sits beside the `.nemo` archive the repo serves, which is what is seeded.
    func testParakeetBuildsFromItsArchiveOffline() throws {
        try requireMLXRuntime()
        let asset = try speechAsset()
        let archiveFolder = try store("IK_VAL_PARAKEET").deletingLastPathComponent()
        let repo = "nvidia/parakeet-tdt-0.6b-v2"
        let backend = try NFKMLXReleaseStore.offline(store: archiveFolder, repo: repo,
                                                     required: NFKMLXParakeet.requiredFiles,
                                                     optional: NFKMLXParakeet.optionalFiles,
                                                     weights: NFKMLXParakeet.weightFiles) {
            try NFKMLXParakeet.backend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertEqual(result.text, "The quick brown fox jumps over the lazy dog.")
    }

    // The store keeps the `.nemo` archive and the repo's tokenizer.json in two folders, so a scratch
    // store links them side by side as the repo serves them.
    func testCanaryBuildsFromItsArchiveOffline() throws {
        try requireMLXRuntime()
        let asset = try speechAsset()
        let release = try store("IK_VAL_CANARY")
        let archive = release.deletingLastPathComponent().appendingPathComponent("canary-1b-v2.nemo")
        guard FileManager.default.fileExists(atPath: archive.path) else {
            throw XCTSkip("the Canary store holds no canary-1b-v2.nemo beside \(release.lastPathComponent)")
        }
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("canary-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: scratch.appendingPathComponent("tokenizer.json"),
                                                   withDestinationURL: release.appendingPathComponent("tokenizer.json"))
        try FileManager.default.createSymbolicLink(at: scratch.appendingPathComponent("canary-1b-v2.nemo"),
                                                   withDestinationURL: archive)
        let repo = "nvidia/canary-1b-v2"
        let backend = try NFKMLXReleaseStore.offline(store: scratch, repo: repo,
                                                     required: NFKMLXCanary.requiredFiles,
                                                     optional: NFKMLXCanary.optionalFiles,
                                                     weights: NFKMLXCanary.weightFiles) {
            try NFKMLXCanary.backend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertEqual(result.text, "The quick brown fox jumps over the lazy dog.")
    }

    func testGraniteSpeechBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let asset = try speechAsset()
        let repo = "ibm-granite/granite-speech-3.3-2b"
        let backend = try NFKMLXReleaseStore.offline(store: try store("IK_VAL_GRANITE_SPEECH"), repo: repo,
                                                     required: NFKMLXGraniteSpeech.requiredFiles,
                                                     optional: NFKMLXGraniteSpeech.optionalFiles,
                                                     weights: NFKMLXGraniteSpeech.weightFiles) {
            try NFKMLXGraniteSpeech.graniteSpeechBackend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        let request = NFKInferenceRequest(inputs: [NFKInputAudio: asset], parameters: [NFKParameterMaxTokens: 64])
        let text = try backend.runInference(for: request).text ?? ""
        XCTAssertTrue(text.lowercased().contains("quick brown fox"), "transcribed \(text.debugDescription)")
    }

    // Voxtral's store holds about 8.7 GB of weights, so this downloads the release without building it
    // and reads the configuration and tokenizer from the returned directory.
    func testVoxtralDownloadsItsListsOffline() throws {
        try requireMLXRuntime()
        let repo = "mistralai/Voxtral-Mini-3B-2507"
        let (configuration, tokenizer, shards) = try NFKMLXReleaseStore.offline(
            store: try store("IK_VAL_VOXTRAL"), repo: repo, required: NFKMLXVoxtral.requiredFiles,
            optional: NFKMLXVoxtral.optionalFiles, weights: NFKMLXVoxtral.weightFiles) {
            let directory = try NFKMLXReleaseDownload.directory(repo: repo, revision: nil, cacheDirectoryURL: nil,
                                                                required: NFKMLXVoxtral.requiredFiles,
                                                                optional: NFKMLXVoxtral.optionalFiles,
                                                                weights: NFKMLXVoxtral.weightFiles)
            let json = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any] ?? [:]
            return (try NFKMLXVoxtralConfiguration.configuration(fromHuggingFace: json),
                    NFKMLXTekkenTokenizer(tekkenURL: directory.appendingPathComponent("tekken.json")),
                    try NFKMLXReleaseWeights.files(inDirectory: directory).filter {
                        FileManager.default.fileExists(atPath: $0.path)
                    }.count)
        }
        XCTAssertNotNil(tokenizer, "tekken.json reads as a tokenizer")
        XCTAssertEqual(shards, 2, "every shard the index names is in the directory")
        XCTAssertGreaterThan(configuration.audioTokenId, 0)
    }
}
