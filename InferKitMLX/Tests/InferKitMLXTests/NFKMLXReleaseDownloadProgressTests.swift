//
//  NFKMLXReleaseDownloadProgressTests.swift
//  InferKitMLXTests
//
//  The release-download layer under a current NFKDownload, against an in-process hub served through
//  URLProtocol, so no test reaches the network: the listing's sizes counted before the bytes, the
//  shards an index names, progress that only grows and reaches its total, the completion record,
//  cancellation part-way through a shard, the asynchronous factories carrying the download, and the
//  catalog-level download-and-build call. None of it evaluates MLX.
//

import XCTest
import CryptoKit
import InferKit
@testable import InferKitMLX

/// Answers http://release.test from scripted responses, with Range support and an optional stall.
final class NFKStubReleaseProtocol: URLProtocol {
    struct Response {
        var status = 200
        var body = Data()
        var stalls = false
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var responses = [String: Response]()
    nonisolated(unsafe) private static var served = [URLRequest]()

    static let endpoint = URL(string: "http://release.test")!

    static func reset() {
        lock.lock()
        responses = [:]
        served = []
        lock.unlock()
    }

    static func set(_ response: Response, for path: String) {
        lock.lock()
        responses[path] = response
        lock.unlock()
    }

    static var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return served
    }

    /// Serves a repository's listing and files at revision main.
    static func serve(repo: String, files: [String: Data], stalling: Set<String> = []) {
        var listing = [[String: Any]]()
        for (path, data) in files.sorted(by: { $0.key < $1.key }) {
            let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            listing.append(["type": "file", "path": path, "size": data.count,
                            "lfs": ["oid": sha, "size": data.count]])
            set(Response(body: data, stalls: stalling.contains(path)), for: "/\(repo)/resolve/main/\(path)")
        }
        let tree = try! JSONSerialization.data(withJSONObject: listing)
        set(Response(body: tree), for: "/api/models/\(repo)/tree/main")
    }

    static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [NFKStubReleaseProtocol.self]
        return URLSession(configuration: configuration)
    }

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "release.test" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let path = request.url?.path ?? ""
        Self.lock.lock()
        Self.served.append(request)
        let script = Self.responses[path] ?? Response(status: 404)
        Self.lock.unlock()
        let response = HTTPURLResponse(url: request.url!, statusCode: script.status, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Length": "\(script.body.count)"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        guard request.httpMethod != "HEAD" else {
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        let chunk = 4096
        var offset = 0
        while offset < script.body.count {
            let end = min(offset + chunk, script.body.count)
            client?.urlProtocol(self, didLoad: script.body.subdata(in: offset..<end))
            if script.stalls {
                return
            }
            offset = end
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Collects a download's progress reports.
private final class ReportLog: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = [NFKDownloadProgress]()
    func append(_ progress: NFKDownloadProgress) {
        lock.lock()
        stored.append(progress)
        lock.unlock()
    }
    var reports: [NFKDownloadProgress] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

/// A hub whose transport fails every request, file and listing alike, as a machine with no network does.
final class NFKFailingTransportHub: NFKHFHub {
    private let lock = NSLock()
    private var attempted = [String]()

    var attempts: [String] {
        lock.lock()
        defer { lock.unlock() }
        return attempted
    }

    private func refuse(_ url: URL) -> NSError {
        lock.lock()
        attempted.append(url.path)
        lock.unlock()
        return NSError(domain: NSURLErrorDomain, code: NSURLErrorNotConnectedToInternet)
    }

    override func fetch(_ remoteURL: URL, toFileURL destinationURL: URL) throws {
        throw refuse(remoteURL)
    }

    override func fetchMetadata(from url: URL, httpMethod method: String,
                                response: AutoreleasingUnsafeMutablePointer<HTTPURLResponse?>?) throws -> Data {
        throw refuse(url)
    }
}

final class NFKMLXReleaseDownloadProgressTests: XCTestCase {

    private let repo = "org/release"
    private var cache: URL!
    private var hub: NFKHFHub!
    private var files = [String: Data]()

    override func setUp() {
        super.setUp()
        NFKStubReleaseProtocol.reset()
        cache = FileManager.default.temporaryDirectory.appendingPathComponent("release-progress-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        hub = NFKHFHub(cacheDirectoryURL: cache)
        hub.endpointURL = NFKStubReleaseProtocol.endpoint
        hub.session = NFKStubReleaseProtocol.session()
        hub.accessToken = ""
        hub.excludesCacheFromBackup = false
        let index = try! JSONSerialization.data(withJSONObject: ["weight_map": ["a": "model-00001-of-00002.safetensors",
                                                                                 "b": "model-00002-of-00002.safetensors"]])
        files = ["config.json": Data("{}".utf8),
                 "tokenizer.json": Data(repeating: 7, count: 3000),
                 "model.safetensors.index.json": index,
                 "model-00001-of-00002.safetensors": Data(repeating: 1, count: 40_000),
                 "model-00002-of-00002.safetensors": Data(repeating: 2, count: 24_000),
                 "unrelated.bin": Data(repeating: 3, count: 50_000)]
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: cache)
        NFKStubReleaseProtocol.reset()
        super.tearDown()
    }

    private func fetchRelease() throws -> URL {
        try NFKMLXReleaseDownload.using(hub) {
            try NFKMLXReleaseDownload.directory(repo: repo, revision: nil, cacheDirectoryURL: nil,
                                                required: ["config.json"], optional: ["tokenizer.json", "absent.json"],
                                                weights: ["model.safetensors", "model.safetensors.index.json"])
        }
    }

    private var releaseByteCount: Int64 {
        Int64(files.filter { $0.key != "unrelated.bin" }.values.reduce(0) { $0 + $1.count })
    }

    func testAReleaseReportsProgressThatOnlyGrowsAndReachesItsTotal() throws {
        NFKStubReleaseProtocol.serve(repo: repo, files: files)
        let log = ReportLog()
        let download = NFKDownload(progressHandler: { log.append($0) })
        download.beginActivity()
        var fetchError: Error?
        var root: URL?
        download.performAsCurrent {
            do {
                root = try fetchRelease()
            } catch {
                fetchError = error
            }
        }
        download.endActivity(error: fetchError)
        XCTAssertTrue(download.waitUntilFinished(withTimeout: 10))
        XCTAssertNil(fetchError)
        XCTAssertEqual(download.state.rawValue, NFKDownloadState.completed.rawValue)
        XCTAssertEqual(root?.path, cache.appendingPathComponent("org/release/main").path)

        let progress = download.currentProgress
        XCTAssertEqual(progress.receivedByteCount, releaseByteCount, "every fetched byte, and nothing unrelated")
        XCTAssertEqual(progress.expectedByteCount, releaseByteCount)
        XCTAssertEqual(progress.fileCount, 5)
        XCTAssertEqual(progress.completedFileCount, 5)
        let listed = NFKStubReleaseProtocol.requests.filter { $0.url?.path.contains("/api/models/") == true }
        XCTAssertEqual(listed.count, 1, "the repository is listed once")

        let reports = log.reports
        XCTAssertFalse(reports.isEmpty)
        var previous: Int64 = 0
        for report in reports {
            XCTAssertGreaterThanOrEqual(report.receivedByteCount, previous)
            previous = report.receivedByteCount
        }
        XCTAssertEqual(reports.last?.state.rawValue, NFKDownloadState.completed.rawValue)
        XCTAssertEqual(reports.last?.fractionCompleted, 1)
        XCTAssertEqual(hub.downloadStatus(repo: repo, revision: nil, paths: nil).presence.rawValue,
                       NFKDownloadPresence.complete.rawValue,
                       "the release recorded its completion")
    }

    /// The release is sharded (its preferred `model.safetensors` is not served) and one optional file
    /// is absent, the two cases a cached build used to ask the server about.
    func testACachedReleaseIsNeitherListedNorFetched() throws {
        NFKStubReleaseProtocol.serve(repo: repo, files: files)
        _ = try fetchRelease()
        XCTAssertEqual(hub.recordedAbsentPaths(ofRepo: repo, revision: nil), ["absent.json"],
                       "the refused optional file is recorded; the unserved weights candidate is not optional")
        NFKStubReleaseProtocol.reset()

        let download = NFKDownload(progressHandler: nil)
        download.beginActivity()
        var fetchError: Error?
        download.performAsCurrent {
            do { _ = try fetchRelease() } catch { fetchError = error }
        }
        download.endActivity(error: fetchError)
        XCTAssertNil(fetchError)
        XCTAssertEqual(NFKStubReleaseProtocol.requests.compactMap { $0.url?.path }, [], "a cached release asks nothing")
        XCTAssertEqual(download.currentProgress.fileCount, 5)
        XCTAssertEqual(download.currentProgress.completedFileCount, 5)
        XCTAssertEqual(download.currentProgress.receivedByteCount, 0)
    }

    func testACachedComponentReleaseIsNeitherListedNorFetched() throws {
        let componentRepo = "org/pipeline"
        let index = try JSONSerialization.data(withJSONObject: ["weight_map": ["a": "model-00001-of-00001.safetensors"]])
        NFKStubReleaseProtocol.serve(repo: componentRepo, files: [
            "transformer/config.json": Data("{}".utf8),
            "transformer/model.safetensors.index.json": index,
            "transformer/model-00001-of-00001.safetensors": Data(repeating: 4, count: 9000),
            "vae/config.json": Data("{}".utf8),
            "vae/model.safetensors": Data(repeating: 5, count: 7000),
        ])
        let components = [
            NFKMLXReleaseComponent(required: ["transformer/config.json"], optional: ["transformer/absent.json"],
                                   weights: ["transformer/model.safetensors", "transformer/model.safetensors.index.json"]),
            NFKMLXReleaseComponent(required: ["vae/config.json"], weights: ["vae/model.safetensors"]),
        ]
        let build = {
            try NFKMLXReleaseDownload.using(self.hub) {
                try NFKMLXReleaseDownload.directory(repo: componentRepo, revision: nil, cacheDirectoryURL: nil,
                                                    components: components)
            }
        }
        _ = try build()
        NFKStubReleaseProtocol.reset()

        let root = try build()
        XCTAssertEqual(root.path, cache.appendingPathComponent("org/pipeline/main").path)
        XCTAssertEqual(NFKStubReleaseProtocol.requests.compactMap { $0.url?.path }, [])
    }

    func testForgettingTheRefusedFilesAsksForThemAgain() throws {
        NFKStubReleaseProtocol.serve(repo: repo, files: files)
        _ = try fetchRelease()
        try hub.forgetRecordedAbsentPaths(ofRepo: repo, revision: nil)
        XCTAssertEqual(hub.recordedAbsentPaths(ofRepo: repo, revision: nil), [])
        NFKStubReleaseProtocol.reset()
        NFKStubReleaseProtocol.serve(repo: repo, files: files.merging(["absent.json": Data("{}".utf8)]) { $1 })

        _ = try fetchRelease()
        XCTAssertEqual(NFKStubReleaseProtocol.requests.compactMap { $0.url?.path }, ["/\(repo)/resolve/main/absent.json"],
                       "only the forgotten file is asked for")
        XCTAssertTrue(hub.isCachedRepo(repo, revision: nil, path: "absent.json"))
        XCTAssertEqual(hub.recordedAbsentPaths(ofRepo: repo, revision: nil), [])
    }

    func testACachedReleaseBuildsWithEveryRequestFailing() throws {
        NFKStubReleaseProtocol.serve(repo: repo, files: files)
        _ = try fetchRelease()

        let offline = NFKFailingTransportHub(cacheDirectoryURL: cache)
        offline.endpointURL = NFKStubReleaseProtocol.endpoint
        let download = NFKDownload(progressHandler: nil)
        download.beginActivity()
        var root: URL?
        var fetchError: Error?
        download.performAsCurrent {
            do {
                root = try NFKMLXReleaseDownload.using(offline) {
                    try NFKMLXReleaseDownload.directory(repo: repo, revision: nil, cacheDirectoryURL: nil,
                                                        required: ["config.json"],
                                                        optional: ["tokenizer.json", "absent.json"],
                                                        weights: ["model.safetensors", "model.safetensors.index.json"])
                }
            } catch {
                fetchError = error
            }
        }
        download.endActivity(error: fetchError)
        XCTAssertNil(fetchError)
        XCTAssertEqual(root?.path, cache.appendingPathComponent("org/release/main").path)
        XCTAssertEqual(offline.attempts, [], "neither the listing nor any file was asked for")
        XCTAssertTrue(download.waitUntilFinished(withTimeout: 5))
        XCTAssertEqual(download.state.rawValue, NFKDownloadState.completed.rawValue)

        // A file the transport cannot reach is not recorded as one the repository lacks.
        _ = try NFKMLXReleaseDownload.using(offline) {
            try NFKMLXReleaseDownload.directory(repo: repo, revision: nil, cacheDirectoryURL: nil,
                                                required: ["config.json"], optional: ["unreachable.json"],
                                                weights: ["model.safetensors.index.json"])
        }
        XCTAssertEqual(offline.attempts, ["/\(repo)/resolve/main/unreachable.json"])
        XCTAssertFalse(hub.recordedAbsentPaths(ofRepo: repo, revision: nil).contains("unreachable.json"))
    }

    func testCancellingPartWayThroughAShardStopsTheReleaseAndLeavesNoPartialFile() throws {
        NFKStubReleaseProtocol.serve(repo: repo, files: files, stalling: ["model-00001-of-00002.safetensors"])
        let download = NFKDownload(progressHandler: nil)
        let finished = expectation(description: "the release download ends")
        nonisolated(unsafe) var fetchError: Error?
        let hub = self.hub!
        let repo = self.repo
        download.performAsCurrent {
            NFKMLXReleaseDownload.asyncValue({ (_: URL?, error: Error?) in
                fetchError = error
                finished.fulfill()
            }) {
                try NFKMLXReleaseDownload.using(hub) {
                    try NFKMLXReleaseDownload.directory(repo: repo, revision: nil, cacheDirectoryURL: nil,
                                                        required: ["config.json"], optional: [],
                                                        weights: ["model.safetensors.index.json"])
                }
            }
        }
        let deadline = Date(timeIntervalSinceNow: 10)
        while download.currentProgress.currentFile?.path != "model-00001-of-00002.safetensors"
                || download.currentProgress.currentFile?.receivedByteCount == 0, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        download.cancel()
        wait(for: [finished], timeout: 10)

        XCTAssertEqual((fetchError as NSError?)?.domain, NFKDownloadErrorDomain)
        XCTAssertEqual((fetchError as NSError?)?.code, NFKDownloadError.cancelled.rawValue)
        XCTAssertEqual(download.state.rawValue, NFKDownloadState.cancelled.rawValue, "the factory's activity ended the download")
        let shard = hub.localURL(forRepo: repo, revision: nil, path: "model-00001-of-00002.safetensors")!
        XCTAssertFalse(FileManager.default.fileExists(atPath: shard.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: shard.appendingPathExtension("download").path))
        XCTAssertNotEqual(hub.downloadStatus(repo: repo, revision: nil, paths: nil).presence.rawValue,
                          NFKDownloadPresence.complete.rawValue)
    }

    func testAnAsynchronousFactoryCarriesTheCurrentDownloadToItsQueue() {
        let download = NFKDownload(progressHandler: nil)
        let built = expectation(description: "build")
        nonisolated(unsafe) var sawDownload = false
        download.performAsCurrent {
            NFKMLXReleaseDownload.asyncValue({ (_: Int?, _: Error?) in built.fulfill() }) {
                sawDownload = NFKDownload.current === download
                return 1
            }
        }
        XCTAssertEqual(download.state.rawValue, NFKDownloadState.downloading.rawValue, "the factory's activity began on the calling thread")
        wait(for: [built], timeout: 5)
        XCTAssertTrue(sawDownload)
        XCTAssertTrue(download.waitUntilFinished(withTimeout: 5))
        XCTAssertEqual(download.state.rawValue, NFKDownloadState.completed.rawValue)
    }

    // MARK: Catalog-level call

    private func descriptor(registryName: String, source: NFKModelSource) throws -> NFKModelDescriptor {
        try NFKModelDescriptor(dictionary: [
            "id": "release-model", "family": "release", "engine": "mlx", "registryName": registryName,
            "variants": [["id": "default", "isDefault": true, "source": source.dictionaryRepresentation()]],
        ])
    }

    func testTheCatalogCallDownloadsTheVariantAndBuildsItsRegisteredBackend() throws {
        NFKStubReleaseProtocol.serve(repo: repo, files: files)
        let name = "release-progress-\(UUID().uuidString)"
        nonisolated(unsafe) var builtFrom: URL?
        NFKMLXModelRegistry.register(name: name) { url in
            builtFrom = url
            return NFKPassthroughBackend()
        }
        defer { NFKMLXModelRegistry.unregisterModel(named: name) }
        let source = NFKModelSource(kind: NFKModelSourceKindHuggingFace, repository: repo, revision: nil, url: nil,
                                    layout: NFKModelSourceLayoutDirectory,
                                    files: [NFKModelFile(path: "config.json", role: NFKModelFileRoleConfig),
                                            NFKModelFile(path: "tokenizer.json", role: NFKModelFileRoleTokenizer)],
                                    requiresConversion: false, extensions: nil)
        let model = try descriptor(registryName: name, source: source)

        let log = ReportLog()
        let done = expectation(description: "backend")
        nonisolated(unsafe) var backend: (any NFKInferenceBackend)?
        let download = NFKMLXReleaseDownload.using(hub) {
            NFKMLXHub.backend(for: model, variant: nil, cacheDirectoryURL: nil, progressHandler: { log.append($0) }) { built, error in
                XCTAssertNil(error)
                backend = built
                done.fulfill()
            }
        }
        wait(for: [done], timeout: 10)
        XCTAssertNotNil(backend)
        XCTAssertEqual(download.state.rawValue, NFKDownloadState.completed.rawValue)
        XCTAssertEqual(builtFrom?.path, cache.appendingPathComponent("org/release/main").path)
        XCTAssertEqual(log.reports.last?.receivedByteCount, Int64(2 + 3000))
        XCTAssertEqual(hub.downloadStatus(source: source).presence.rawValue, NFKDownloadPresence.complete.rawValue)
    }

    func testTheCatalogCallFailsAtOnceForAnUnregisteredModel() throws {
        let source = NFKModelSource(kind: NFKModelSourceKindHuggingFace, repository: repo, revision: nil, url: nil,
                                    layout: NFKModelSourceLayoutDirectory, files: nil, requiresConversion: false,
                                    extensions: nil)
        let model = try descriptor(registryName: "never-registered-\(UUID().uuidString)", source: source)
        let done = expectation(description: "refusal")
        let download = NFKMLXReleaseDownload.using(hub) {
            NFKMLXHub.backend(for: model, variant: nil, cacheDirectoryURL: nil, progressHandler: nil) { built, error in
                XCTAssertNil(built)
                XCTAssertNotNil(error)
                done.fulfill()
            }
        }
        wait(for: [done], timeout: 5)
        XCTAssertEqual(download.state.rawValue, NFKDownloadState.failed.rawValue)
        XCTAssertTrue(NFKStubReleaseProtocol.requests.isEmpty, "nothing is fetched")
    }

    func testTheFactoryHubIsTheOneTheFactoriesUse() {
        let folder = cache.appendingPathComponent("factories")
        XCTAssertEqual(NFKMLXHub.hub(cacheDirectoryURL: folder).cacheDirectoryURL, folder)
        XCTAssertEqual(NFKMLXHub.hub(cacheDirectoryURL: nil).cacheDirectoryURL, NFKHFHub.defaultCacheDirectoryURL())
    }
}
