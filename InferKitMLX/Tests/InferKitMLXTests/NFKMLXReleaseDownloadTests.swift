//
//  NFKMLXReleaseDownloadTests.swift
//  InferKitMLXTests
//
//  The shared release download: an offline hub serves only what the store seeded, a list that names
//  everything a model reads builds it, and one that misses a file does not.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXReleaseDownloadTests: XCTestCase {

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

    func testAnOfflineHubServesTheCacheAndRefusesTheRest() throws {
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: store) }
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        for name in ["config.json", "model.safetensors"] {
            try Data("{}".utf8).write(to: store.appendingPathComponent(name))
        }
        let directory = try NFKMLXReleaseStore.offline(store: store, repo: "org/model", required: ["config.json"],
                                                       optional: ["absent.json"], weights: ["missing.bin", "model.safetensors"]) {
            try NFKMLXReleaseDownload.directory(repo: "org/model", revision: nil, cacheDirectoryURL: nil,
                                                required: ["config.json"], optional: ["absent.json"],
                                                weights: ["missing.bin", "model.safetensors"])
        }
        XCTAssertTrue(directory.path.hasSuffix("org/model/main"))
        XCTAssertThrowsError(try NFKMLXReleaseStore.offline(store: store, repo: "org/model", required: ["config.json"],
                                                            optional: [], weights: ["model.safetensors"]) {
            try NFKMLXReleaseDownload.directory(repo: "org/model", revision: nil, cacheDirectoryURL: nil,
                                                required: ["config.json", "tokenizer.json"], optional: [],
                                                weights: ["model.safetensors"])
        }, "a required file the store did not seed is refused, not fetched")
    }

    // An index inside a component folder names its shards relative to that folder; fetching them from
    // the repo root would ask for files the repo does not serve.
    func testAShardIndexInASubfolderFetchesItsShardsBesideIt() throws {
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: store) }
        let folder = store.appendingPathComponent("transformer")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let index = ["weight_map": ["a": "part-1.safetensors", "b": "part-2.safetensors"]]
        try JSONSerialization.data(withJSONObject: index).write(to: folder.appendingPathComponent("model.safetensors.index.json"))
        for name in ["config.json", "part-1.safetensors", "part-2.safetensors"] {
            try Data("{}".utf8).write(to: folder.appendingPathComponent(name))
        }
        let lists = (required: ["transformer/config.json"], weights: ["transformer/model.safetensors.index.json"])
        // The seeded cache lives only for the closure, so the check runs inside it.
        let fetched = try NFKMLXReleaseStore.offline(store: store, repo: "org/model", required: lists.required,
                                                     optional: [], weights: lists.weights) { () -> Bool in
            let directory = try NFKMLXReleaseDownload.directory(repo: "org/model", revision: nil, cacheDirectoryURL: nil,
                                                                required: lists.required, optional: [], weights: lists.weights)
            // The download returns the snapshot root, whatever folder the last required file sits in.
            return directory.path.hasSuffix("org/model/main")
                && FileManager.default.fileExists(atPath: directory.appendingPathComponent("transformer/part-2.safetensors").path)
        }
        XCTAssertTrue(fetched)
    }

    // A released Kokoro voicepack is a bare tensor; read natively it equals the converted file exactly.
    func testAReleasedKokoroVoiceReadsAsItsConvertedFile() throws {
        try requireMLXRuntime()
        guard let root = config["IK_VAL_KOKORO"] else { throw XCTSkip("set IK_VAL_KOKORO") }
        let voices = URL(fileURLWithPath: root).appendingPathComponent("voices")
        let released = try NFKMLXWeights.loadCheckpoint(url: voices.appendingPathComponent("af_heart.pt")).arrays
        XCTAssertEqual(Array(released.keys), ["tensor"], "a bare-tensor root comes back under one key")
        let converted = try NFKMLXWeights.loadCheckpoint(url: voices.appendingPathComponent("af_heart.safetensors")).arrays
        let a = try XCTUnwrap(released["tensor"]), b = try XCTUnwrap(converted["voice"] ?? converted.values.first)
        XCTAssertEqual(a.shape, b.shape)
        XCTAssertEqual(a.asType(.float32).asArray(Float.self), b.asType(.float32).asArray(Float.self))
    }

    // Kokoro downloads its configuration, weights, and one voice, and speaks from them.
    func testKokoroBuildsFromItsReleaseOffline() throws {
        try requireMLXRuntime()
        guard let root = config["IK_VAL_KOKORO"] else { throw XCTSkip("set IK_VAL_KOKORO") }
        let repo = "hexgrad/Kokoro-82M"
        let files = NFKMLXKokoro.releaseFiles(voiceName: "af_heart")
        let backend = try NFKMLXReleaseStore.offline(store: URL(fileURLWithPath: root), repo: repo,
                                                     required: files.required, optional: [], weights: files.weights) {
            try NFKMLXKokoro.backend(repo: repo, revision: nil, cacheDirectoryURL: nil, voiceName: "af_heart")
        }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hˈɛloʊ"]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio))
    }

    // The one release download that reaches the network: Kokoro's factory fetches its configuration,
    // weights, and one voice from Hugging Face into a fresh cache (about 330 MB) and speaks. A second
    // build reads the cache and fetches nothing. Opt-in, like the other live download.
    func testKokoroDownloadsItsReleaseLive() throws {
        try requireMLXRuntime()
        guard ProcessInfo.processInfo.environment["INFERKIT_LIVE_RELEASE_DOWNLOADS"] == "1" else {
            throw XCTSkip("set INFERKIT_LIVE_RELEASE_DOWNLOADS=1 to download hexgrad/Kokoro-82M")
        }
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("kokoro-live-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        let started = Date()
        let backend = try NFKMLXKokoro.backend(repo: "hexgrad/Kokoro-82M", revision: nil, cacheDirectoryURL: cache,
                                               voiceName: "af_heart")
        let fetched = Date().timeIntervalSince(started)
        let snapshot = cache.appendingPathComponent("hexgrad/Kokoro-82M/main")
        for path in ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: snapshot.appendingPathComponent(path).path), path)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("voices/af_bella.pt").path),
                       "only the named voice is fetched")
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hˈɛloʊ wˈɜːld"]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio))

        let restarted = Date()
        _ = try NFKMLXKokoro.backend(repo: "hexgrad/Kokoro-82M", revision: nil, cacheDirectoryURL: cache, voiceName: "af_heart")
        print(String(format: "VALIDATION live kokoro download: first build %.1f s, cached rebuild %.1f s",
                     fetched, Date().timeIntervalSince(restarted)))
    }

    // The existing translator download builds offline from the store with its own lists.
    func testOPUSMTBuildsFromItsListsOffline() throws {
        try requireMLXRuntime()
        guard let root = config["IK_VAL_MARIAN"] else { throw XCTSkip("set IK_VAL_MARIAN") }
        let repo = "Helsinki-NLP/opus-mt-en-de"
        let backend = try NFKMLXReleaseStore.offline(store: URL(fileURLWithPath: root), repo: repo,
                                                     required: NFKMLXMarian.requiredFiles,
                                                     optional: NFKMLXMarian.optionalFiles,
                                                     weights: NFKMLXMarian.weightFiles) {
            try NFKMLXMarian.backend(repo: repo, revision: nil, cacheDirectoryURL: nil)
        }
        XCTAssertTrue(backend.isReady)
    }
}
