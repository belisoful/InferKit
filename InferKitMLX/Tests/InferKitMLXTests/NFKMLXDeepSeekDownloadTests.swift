//
//  NFKMLXDeepSeekDownloadTests.swift
//  InferKitMLXTests
//
//  The DeepSeek repo factories and their asynchronous peers fetch the release through the hub. An
//  offline hub over an empty cache refuses every fetch, so each factory reports the refusal of the first
//  file it needs without loading anything.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXDeepSeekDownloadTests: XCTestCase {

    private let repo = "deepseek-ai/DeepSeek-V4.1-Flash"

    private func offlineHub() -> NFKOfflineHub {
        let hub = NFKOfflineHub()
        hub.cacheDirectoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("deepseek-download-\(UUID().uuidString)")
        return hub
    }

    func testEveryRepoFactoryFetchesTheConfigurationThroughTheHub() {
        let repo = self.repo
        let factories: [() throws -> any NFKInferenceBackend] = [
            { try NFKMLXDeepSeek.deepSeekBackend(repo: repo, revision: nil, cacheDirectoryURL: nil) },
            { try NFKMLXDeepSeek.deepSeekBackend(repo: repo, revision: nil, cacheDirectoryURL: nil,
                                                 residency: .paged) },
            { try NFKMLXDeepSeek.deepSeekBackend(repo: repo, revision: nil, cacheDirectoryURL: nil,
                                                 options: NFKMLXDeepSeekLoadOptions()) },
            { try NFKMLXDeepSeek.deepSeekPagedBackend(repo: repo, revision: nil, cacheDirectoryURL: nil,
                                                      expertCacheBytes: 0) },
        ]
        for factory in factories {
            let hub = offlineHub()
            XCTAssertThrowsError(try NFKMLXReleaseDownload.using(hub, factory))
            XCTAssertTrue(hub.refused.contains { $0.hasSuffix("/config.json") },
                          "the release's configuration is fetched first: \(hub.refused)")
        }
    }

    func testEveryAsyncPeerDeliversTheDownloadsError() {
        let repo = self.repo
        typealias Handler = ((any NFKInferenceBackend)?, Error?) -> Void
        let peers: [(@escaping Handler) -> Void] = [
            { NFKMLXDeepSeek.deepSeekBackend(repo: repo, revision: nil, cacheDirectoryURL: nil, completionHandler: $0) },
            { NFKMLXDeepSeek.deepSeekBackend(repo: repo, revision: nil, cacheDirectoryURL: nil, residency: .paged,
                                             completionHandler: $0) },
            { NFKMLXDeepSeek.deepSeekBackend(repo: repo, revision: nil, cacheDirectoryURL: nil,
                                             options: NFKMLXDeepSeekLoadOptions(), completionHandler: $0) },
            { NFKMLXDeepSeek.deepSeekPagedBackend(repo: repo, revision: nil, cacheDirectoryURL: nil,
                                                  expertCacheBytes: 0, completionHandler: $0) },
        ]
        for peer in peers {
            let hub = offlineHub()
            // The override holds only while the body runs, so the body waits for the handler.
            NFKMLXReleaseDownload.using(hub) {
                let done = expectation(description: "the handler runs")
                peer { backend, error in
                    XCTAssertNil(backend)
                    XCTAssertNotNil(error)
                    done.fulfill()
                }
                wait(for: [done], timeout: 30)
            }
            XCTAssertTrue(hub.refused.contains { $0.hasSuffix("/config.json") })
        }
    }
}
