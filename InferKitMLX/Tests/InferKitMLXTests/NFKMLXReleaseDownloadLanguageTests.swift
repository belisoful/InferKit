//
//  NFKMLXReleaseDownloadLanguageTests.swift
//  InferKitMLXTests
//
//  The language-model download factories build from their own file lists with the network refused.
//  A store whose weights exceed about 8 GB is fetched through the lists without building the model,
//  and its configuration and tokenizer load from the fetched directory.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXReleaseDownloadLanguageTests: XCTestCase {

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
        guard let root = config[key] else { throw XCTSkip("set \(key)") }
        return URL(fileURLWithPath: root)
    }

    private func firstToken(of backend: any NFKInferenceBackend, prompt: String) throws -> String {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: prompt],
                                          parameters: [NFKParameterMaxTokens: 1, NFKParameterTemperature: 0])
        return try backend.runInference(for: request).text ?? ""
    }

    func testQwen3BuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_QWEN3")
        let repo = "Qwen/Qwen3-0.6B"
        let backend = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                     required: NFKMLXLanguage.requiredFiles,
                                                     optional: NFKMLXLanguage.optionalFiles,
                                                     weights: NFKMLXLanguage.weightFiles) {
            try NFKMLXLanguage.backend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        XCTAssertFalse(try firstToken(of: backend, prompt: "The capital of France is").isEmpty)
    }

    // The same release serves as its own draft, so one seeded repo covers both downloads.
    func testQwen3DraftPairBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_QWEN3")
        let repo = "Qwen/Qwen3-0.6B"
        let backend = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                     required: NFKMLXLanguage.requiredFiles,
                                                     optional: NFKMLXLanguage.optionalFiles,
                                                     weights: NFKMLXLanguage.weightFiles) {
            try NFKMLXLanguage.backend(repo: repo, revision: nil, draftRepo: repo, draftRevision: nil,
                                       cacheDirectoryURL: nil)
        }
        XCTAssertEqual((backend as? NFKMLXLanguageBackend)?.hasDraftModel, true)
    }

    // Gemma 4 E2B is 9.6 GB, so the test fetches through the lists and loads what the build reads first.
    func testGemma4ListsFetchTheConfigurationAndTokenizerOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_GEMMA4")
        let repo = "google/gemma-4-E2B-it"
        let directory = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                       required: NFKMLXGemmaLanguage.requiredFiles,
                                                       optional: NFKMLXGemmaLanguage.optionalFiles,
                                                       weights: NFKMLXGemmaLanguage.weightFiles) {
            let directory = try NFKMLXReleaseDownload.directory(
                repo: repo, revision: nil, cacheDirectoryURL: nil,
                required: NFKMLXGemmaLanguage.requiredFiles, optional: NFKMLXGemmaLanguage.optionalFiles,
                weights: NFKMLXGemmaLanguage.weightFiles)
            let configuration = try NFKMLXGemmaLanguage.configuration(
                fromHuggingFace: directory.appendingPathComponent("config.json"))
            XCTAssertGreaterThan(configuration.layerCount, 0)
            XCTAssertNotNil(NFKMLXGemmaTokenizer(directoryURL: directory)?.id(forToken: "<bos>"))
            XCTAssertNoThrow(try NFKMLXReleaseWeights.files(inDirectory: directory))
            return directory
        }
        XCTAssertTrue(directory.path.hasSuffix("\(repo)/main"))
    }

    func testGemma3BuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_GEMMA3_270M")
        let repo = "unsloth/gemma-3-270m-it"
        let (model, backend) = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                              required: NFKMLXGemma3.requiredFiles,
                                                              optional: NFKMLXGemma3.optionalFiles,
                                                              weights: NFKMLXGemma3.weightFiles) {
            (try NFKMLXGemma3.gemma3(repo: repo, revision: nil, cacheDirectoryURL: nil),
             try NFKMLXGemma3.backend(repo: repo, revision: nil, cacheDirectoryURL: nil))
        }
        XCTAssertNotNil(model.model.chatTemplate, "the chat template was not fetched")
        XCTAssertTrue(backend.isReady)
    }

    // Gemma 3n E2B is 10 GB, so the test fetches through the lists and loads what the build reads first.
    func testGemma3nListsFetchTheConfigurationAndTokenizerOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_GEMMA3N_E2B")
        let repo = "unsloth/gemma-3n-E2B-it"
        try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                       required: NFKMLXGemma3n.requiredFiles,
                                       optional: NFKMLXGemma3n.optionalFiles,
                                       weights: NFKMLXGemma3n.weightFiles) {
            let directory = try NFKMLXReleaseDownload.directory(
                repo: repo, revision: nil, cacheDirectoryURL: nil,
                required: NFKMLXGemma3n.requiredFiles, optional: NFKMLXGemma3n.optionalFiles,
                weights: NFKMLXGemma3n.weightFiles)
            let json = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any] ?? [:]
            XCTAssertGreaterThan(try NFKMLXGemma3nLanguage.configuration(fromJSON: json).layerCount, 0)
            XCTAssertNotNil(NFKMLXGemmaTokenizer(directoryURL: directory)?.id(forToken: "<bos>"))
            XCTAssertEqual(try NFKMLXReleaseWeights.files(inDirectory: directory).count, 3)
        }
    }

    func testGraniteBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_GRANITE")
        let repo = "ibm-granite/granite-4.0-h-1b"
        let backend = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                     required: NFKMLXGraniteHybrid.requiredFiles,
                                                     optional: NFKMLXGraniteHybrid.optionalFiles,
                                                     weights: NFKMLXGraniteHybrid.weightFiles) {
            try NFKMLXGraniteHybrid.graniteBackend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        XCTAssertFalse(try firstToken(of: backend, prompt: "def fibonacci(n):").isEmpty)
    }

    // Nemotron Nano 9B v2 is 17.8 GB, so the test fetches through the lists and loads what the build
    // reads first.
    func testNemotronListsFetchTheConfigurationAndTokenizerOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_NEMOTRON_H")
        let repo = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
        try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                       required: NFKMLXNemotronH.requiredFiles,
                                       optional: NFKMLXNemotronH.optionalFiles,
                                       weights: NFKMLXNemotronH.weightFiles) {
            let directory = try NFKMLXReleaseDownload.directory(
                repo: repo, revision: nil, cacheDirectoryURL: nil,
                required: NFKMLXNemotronH.requiredFiles, optional: NFKMLXNemotronH.optionalFiles,
                weights: NFKMLXNemotronH.weightFiles)
            _ = try NFKMLXNemotronH.configuration(fromDirectory: directory)
            XCTAssertNotNil(NFKMLXLanguage.releaseTokenizer(inDirectory: directory))
            XCTAssertNoThrow(try NFKMLXReleaseWeights.files(inDirectory: directory))
        }
    }

    // Codestral Mamba is 14 GB, so the test fetches through the lists and loads what the build reads first.
    func testMambaListsFetchTheConfigurationAndTokenizerOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_MAMBA2")
        let repo = "mistralai/Mamba-Codestral-7B-v0.1"
        try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                       required: NFKMLXMamba.requiredFiles,
                                       optional: NFKMLXMamba.optionalFiles,
                                       weights: NFKMLXMamba.weightFiles) {
            let directory = try NFKMLXReleaseDownload.directory(
                repo: repo, revision: nil, cacheDirectoryURL: nil,
                required: NFKMLXMamba.requiredFiles, optional: NFKMLXMamba.optionalFiles,
                weights: NFKMLXMamba.weightFiles)
            _ = try NFKMLXMamba.configuration(fromDirectory: directory)
            XCTAssertNotNil(NFKMLXMistralTokenizer(directoryURL: directory))
            XCTAssertEqual(try NFKMLXReleaseWeights.files(inDirectory: directory).count, 3)
        }
    }
}
