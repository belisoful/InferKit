//
//  NFKMLXReleaseDownloadAudioGenerationTests.swift
//  InferKitMLXTests
//
//  The download-and-build factories of the speech, music, and image-generation releases, served
//  offline from the validation store. Only the files a factory's lists name are seeded, so a list
//  that misses a file the model reads fails. A release too large to load here is downloaded offline
//  and its configuration files are read from the returned directory instead of being built.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXReleaseDownloadAudioGenerationTests: XCTestCase {

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
            throw XCTSkip("set \(key) to a present validation store")
        }
        return URL(fileURLWithPath: path)
    }

    /// Runs `body` with every release download going through an offline hub over a fresh cache that
    /// `seed` fills.
    private func offline<T>(seed: (URL) throws -> Void, _ body: () throws -> T) throws -> T {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("release-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        try seed(cache)
        let hub = NFKOfflineHub()
        hub.cacheDirectoryURL = cache
        return try NFKMLXReleaseDownload.using(hub, body)
    }

    /// Seeds each component's files the way a download would place them.
    private func seed(_ components: [NFKMLXReleaseComponent], from store: URL, repo: String, into cache: URL) throws {
        var files = [String]()
        for component in components {
            files += NFKMLXReleaseStore.files(in: store, required: component.required,
                                              optional: component.optional, weights: component.weights)
        }
        try NFKMLXReleaseStore.seed(files, from: store, repo: repo, revision: nil, into: cache)
    }

    /// Links one store file into the cache at `path`, for a store that keeps it under another name.
    private func link(_ source: URL, as path: String, repo: String, into cache: URL) throws {
        let link = cache.appendingPathComponent(repo).appendingPathComponent("main").appendingPathComponent(path)
        try FileManager.default.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source.resolvingSymlinksInPath())
    }

    private func assertWeightFilesPresent(in directory: URL, file: StaticString = #filePath, line: UInt = #line) throws {
        let files = try NFKMLXReleaseWeights.files(inDirectory: directory)
        XCTAssertFalse(files.isEmpty, file: file, line: line)
        for url in files {
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "\(url.lastPathComponent) was not fetched",
                          file: file, line: line)
        }
    }

    // MARK: - Built offline

    func testMossFormer2SRBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_MOSSFORMER2_SR")
        let repo = "alibabasglab/MossFormer2_SR_48K"
        let backend = try NFKMLXReleaseStore.offline(store: root, repo: repo,
                                                     required: NFKMLXMossFormer2SRFactory.requiredFiles,
                                                     optional: NFKMLXMossFormer2SRFactory.optionalFiles,
                                                     weights: NFKMLXMossFormer2SRFactory.weightFiles) {
            try NFKMLXMossFormer2SRFactory.backend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        XCTAssertTrue(backend.isReady)
    }

    // The store keeps the checkpoint without the repo's `enhancer_stage2/` prefix.
    func testResembleEnhanceBuildsFromItsCheckpointOffline() throws {
        try requireMLXRuntime()
        let checkpoint = try store("IK_VAL_REENHANCE")
        let repo = "ResembleAI/resemble-enhance"
        let backend = try offline(seed: { cache in
            try link(checkpoint, as: NFKMLXResembleEnhanceFactory.checkpointFile, repo: repo, into: cache)
        }) {
            try NFKMLXResembleEnhanceFactory.backend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        XCTAssertTrue(backend.isReady)
    }

    // The transformer comes from its own repo as released (`pytorch_model.bin`) and the vocoder from
    // NVIDIA's.
    func testVoiceRestoreBuildsFromBothReposOffline() throws {
        try requireMLXRuntime()
        let transformerStore = try store("IK_VAL_VOICERESTORE").deletingLastPathComponent()
        let vocoder = try store("IK_VAL_BIGVGAN")
        let repo = "jadechoghari/VoiceRestore"
        let backend = try offline(seed: { cache in
            try NFKMLXReleaseStore.seed(
                NFKMLXReleaseStore.files(in: transformerStore, required: [], optional: [],
                                         weights: NFKMLXVoiceRestoreFactory.weightFiles),
                from: transformerStore, repo: repo, revision: nil, into: cache)
            try link(vocoder, as: NFKMLXVoiceRestoreFactory.vocoderFile,
                     repo: NFKMLXVoiceRestoreFactory.vocoderRepo, into: cache)
        }) {
            try NFKMLXVoiceRestoreFactory.backend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        XCTAssertTrue(backend.isReady)
    }

    func testChatterboxBuildsInItsBuiltInVoiceFromItsListsOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_CHATTERBOX")
        let repo = "ResembleAI/chatterbox"
        let backend = try NFKMLXReleaseStore.offline(
            store: root, repo: repo,
            required: NFKMLXChatterbox.requiredFiles + [NFKMLXChatterbox.builtinVoiceFile],
            optional: NFKMLXChatterbox.optionalFiles, weights: NFKMLXChatterbox.weightFiles) {
            try NFKMLXChatterbox.backend(repo: repo, revision: nil, cacheDirectoryURL: nil, voiceURL: nil)
        }
        XCTAssertTrue(backend.isReady)
    }

    // MARK: - Downloaded offline, too large to build here

    // FLUX.1 [schnell] is about 34 GB, so the download runs offline and the configuration files and
    // weight files are read from the returned directory; `flux(directoryURL:)` is the rest of the factory.
    func testFluxDownloadFetchesEveryComponentOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_FLUX_SCHNELL_FULL")
        let repo = "black-forest-labs/FLUX.1-schnell"
        try offline(seed: { cache in
            try seed(NFKMLXFlux.releaseComponents, from: root, repo: repo, into: cache)
        }) {
            let directory = try NFKMLXFlux.releaseDirectory(repo: repo, revision: nil, cacheDirectoryURL: nil)
            _ = try NFKMLXFluxTransformerNet.configuration(
                fromHuggingFace: directory.appendingPathComponent("transformer/config.json"))
            _ = try NFKMLXSDPromptTokenizer(directoryURL: directory.appendingPathComponent("tokenizer"))
            _ = try NFKMLXSentencePieceModel(contentsOf: directory.appendingPathComponent("tokenizer_2/spiece.model"))
            try assertWeightFilesPresent(in: directory.appendingPathComponent("transformer"))
            try assertWeightFilesPresent(in: directory.appendingPathComponent("text_encoder_2"))
            for path in ["vae/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors"] {
                XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent(path).path), path)
            }
        }
    }

    // FLUX.2 [klein] 4B is about 16 GB, so the download runs offline and the configuration files and
    // weight files are read from the returned directory; `flux2(directoryURL:)` is the rest of the factory.
    func testFlux2DownloadFetchesEveryComponentOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_FLUX2_KLEIN_4B_ROOT")
        let repo = "black-forest-labs/FLUX.2-klein-4B"
        try offline(seed: { cache in
            try seed(NFKMLXFlux2.releaseComponents, from: root, repo: repo, into: cache)
        }) {
            let directory = try NFKMLXFlux2.releaseDirectory(repo: repo, revision: nil, cacheDirectoryURL: nil)
            _ = try NFKMLXFlux2TransformerNet.configuration(
                fromHuggingFace: directory.appendingPathComponent("transformer/config.json"))
            _ = try NFKMLXLanguage.configuration(
                fromHuggingFace: directory.appendingPathComponent("text_encoder/config.json"))
            XCTAssertNotNil(NFKMLXLanguage.releaseTokenizer(inDirectory: directory.appendingPathComponent("tokenizer")))
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("tokenizer/chat_template.jinja").path))
            XCTAssertTrue(NFKMLXFlux2.isDistilled(releaseDirectory: directory), "klein 4B's model_index.json marks it distilled")
            try assertWeightFilesPresent(in: directory.appendingPathComponent("transformer"))
            try assertWeightFilesPresent(in: directory.appendingPathComponent("text_encoder"))
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("vae/diffusion_pytorch_model.safetensors").path))
        }
    }

    // MiniMax Music 3 is about 29 GB and its backend loads each stage when it runs, so the download
    // runs offline, the backend is constructed, and the tokenizer and weight files are read from the
    // returned directory.
    func testMusic3DownloadFetchesEveryComponentOffline() throws {
        try requireMLXRuntime()
        let root = try store("IK_VAL_MUSIC3_DIR")
        let repo = "MiniMaxAI/MiniMax-Music3"
        try offline(seed: { cache in
            try seed(NFKMLXMusic3.releaseComponents, from: root, repo: repo, into: cache)
        }) {
            let directory = try NFKMLXMusic3.releaseDirectory(repo: repo, revision: nil, cacheDirectoryURL: nil)
            XCTAssertTrue(try NFKMLXMusic3.backend(directoryURL: directory).isReady)
            _ = try NFKMusic3Prompt.tokenizer(
                directory: directory.appendingPathComponent("qwen_7B/qwen3-8B-tokenizer-music"))
            _ = try NFKMLXLanguage.configuration(
                fromHuggingFace: directory.appendingPathComponent("language_model/config.json"))
            try assertWeightFilesPresent(in: directory.appendingPathComponent("language_model"))
            try assertWeightFilesPresent(in: directory.appendingPathComponent("transformer"))
            for component in ["rvq_depth_decoder", "condition_encoder", "vocoder"] {
                let path = "\(component)/diffusion_pytorch_model.safetensors"
                XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent(path).path), path)
            }
        }
    }
}
