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
        XCTAssertEqual(download.state, .completed)
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
        XCTAssertEqual(reports.last?.state, .completed)
        XCTAssertEqual(reports.last?.fractionCompleted, 1)
        XCTAssertEqual(hub.downloadStatus(repo: repo, revision: nil, paths: nil).presence, .complete,
                       "the release recorded its completion")
    }

    func testACachedReleaseIsNeitherListedNorFetched() throws {
        NFKStubReleaseProtocol.serve(repo: repo, files: files)
        _ = try fetchRelease()
        NFKStubReleaseProtocol.reset()

        let download = NFKDownload(progressHandler: nil)
        download.beginActivity()
        var fetchError: Error?
        download.performAsCurrent {
            do { _ = try fetchRelease() } catch { fetchError = error }
        }
        download.endActivity(error: fetchError)
        XCTAssertNil(fetchError)
        XCTAssertTrue(NFKStubReleaseProtocol.requests.isEmpty)
        XCTAssertEqual(download.currentProgress.completedFileCount, download.currentProgress.fileCount)
        XCTAssertEqual(download.currentProgress.receivedByteCount, 0)
    }

    func testCancellingPartWayThroughAShardStopsTheReleaseAndLeavesNoPartialFile() throws {
        NFKStubReleaseProtocol.serve(repo: repo, files: files, stalling: ["model-00001-of-00002.safetensors"])
        let download = NFKDownload(progressHandler: nil)
        let finished = expectation(description: "the release download ends")
        nonisolated(unsafe) var fetchError: Error?
        let hub = self.hub!
        let repo = self.repo
        download.performAsCurrent {
            NFKMLXReleaseDownload.async({ (_: URL?, error: Error?) in
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
        XCTAssertEqual(download.state, .cancelled, "the factory's activity ended the download")
        let shard = hub.localURL(forRepo: repo, revision: nil, path: "model-00001-of-00002.safetensors")!
        XCTAssertFalse(FileManager.default.fileExists(atPath: shard.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: shard.appendingPathExtension("download").path))
        XCTAssertNotEqual(hub.downloadStatus(repo: repo, revision: nil, paths: nil).presence, .complete)
    }

    func testAnAsynchronousFactoryCarriesTheCurrentDownloadToItsQueue() {
        let download = NFKDownload(progressHandler: nil)
        let built = expectation(description: "build")
        nonisolated(unsafe) var sawDownload = false
        download.performAsCurrent {
            NFKMLXReleaseDownload.async({ (_: Int?, _: Error?) in built.fulfill() }) {
                sawDownload = NFKDownload.current === download
                return 1
            }
        }
        XCTAssertEqual(download.state, .downloading, "the factory's activity began on the calling thread")
        wait(for: [built], timeout: 5)
        XCTAssertTrue(sawDownload)
        XCTAssertTrue(download.waitUntilFinished(withTimeout: 5))
        XCTAssertEqual(download.state, .completed)
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
        XCTAssertEqual(download.state, .completed)
        XCTAssertEqual(builtFrom?.path, cache.appendingPathComponent("org/release/main").path)
        XCTAssertEqual(log.reports.last?.receivedByteCount, Int64(2 + 3000))
        XCTAssertEqual(hub.downloadStatus(source: source).presence, .complete)
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
        XCTAssertEqual(download.state, .failed)
        XCTAssertTrue(NFKStubReleaseProtocol.requests.isEmpty, "nothing is fetched")
    }

    func testTheFactoryHubIsTheOneTheFactoriesUse() {
        let folder = cache.appendingPathComponent("factories")
        XCTAssertEqual(NFKMLXHub.hub(cacheDirectoryURL: folder).cacheDirectoryURL, folder)
        XCTAssertEqual(NFKMLXHub.hub(cacheDirectoryURL: nil).cacheDirectoryURL, NFKHFHub.defaultCacheDirectoryURL())
    }
}
