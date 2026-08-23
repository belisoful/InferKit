//
//  NFKMLXReleaseWeightsTests.swift
//  InferKitMLXTests
//

import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXReleaseWeightsTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func makeRelease(sharded: Bool) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("release-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let first: [String: MLXArray] = ["a.weight": MLXArray([Float(1), 2]),
                                         "b.weight": MLXArray([Float(3)])]
        let second: [String: MLXArray] = ["c.weight": MLXArray([Float(4)])]
        if sharded {
            try save(arrays: first, url: directory.appendingPathComponent("model-00001-of-00002.safetensors"))
            try save(arrays: second, url: directory.appendingPathComponent("model-00002-of-00002.safetensors"))
            let index: [String: Any] = ["weight_map": ["a.weight": "model-00001-of-00002.safetensors",
                                                       "b.weight": "model-00001-of-00002.safetensors",
                                                       "c.weight": "model-00002-of-00002.safetensors"]]
            try JSONSerialization.data(withJSONObject: index)
                .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        } else {
            try save(arrays: first.merging(second) { a, _ in a },
                     url: directory.appendingPathComponent("model.safetensors"))
        }
        return directory
    }

    func testASingleFileReleaseReads() throws {
        try requireMLXRuntime()
        let directory = try makeRelease(sharded: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory)
        XCTAssertEqual(Set(arrays.map(\.0)), ["a.weight", "b.weight", "c.weight"])
    }

    func testAShardedReleaseReadsEveryShardOnce() throws {
        try requireMLXRuntime()
        let directory = try makeRelease(sharded: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let files = try NFKMLXReleaseWeights.files(inDirectory: directory)
        XCTAssertEqual(files.count, 2, "two shards, however many tensors the index names")
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory)
        XCTAssertEqual(Set(arrays.map(\.0)), ["a.weight", "b.weight", "c.weight"])
    }

    func testTheRemapSkipsAndRenames() throws {
        try requireMLXRuntime()
        let directory = try makeRelease(sharded: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory) {
            $0 == "b.weight" ? nil : "renamed." + $0
        }
        XCTAssertEqual(Set(arrays.map(\.0)), ["renamed.a.weight", "renamed.c.weight"])
    }

    func testADirectoryWithNeitherFormThrows() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("release-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertThrowsError(try NFKMLXReleaseWeights.files(inDirectory: directory))
    }
}
