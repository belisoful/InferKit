//
//  NFKMLXCheckpointRoundTripTests.swift
//  InferKitMLXTests
//
//  A model's own `loadWeights` has to read back what `NFKMLXWeights.save` writes, because a fine-tuned
//  checkpoint reloads through the model's existing `backendWith…weightsURL:` factory. The file records
//  its layout, and the loader skips its PyTorch transpose for a file already in the module's layout —
//  so a model that ignored the flag would transpose twice and load silently wrong weights.
//
//  The models here are the ones whose transpose is NOT the common `[out, in, kH, kW]` case, which is
//  where a double transpose does the most damage: SAM's `up1`/`up2` and the Colorizer's deconvolutions
//  use `transposed(1, 2, 3, 0)`, and Demucs carries 3-D Conv1d weights.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXCheckpointRoundTripTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
    }

    /// Saves `source`, reloads through `load`, and asserts every parameter survives unchanged.
    private func assertRoundTrips(_ source: Module, into destination: Module,
                                  load: (Module, URL) throws -> Void,
                                  file: StaticString = #filePath, line: UInt = #line) throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try NFKMLXWeights.save(source, to: url)
        XCTAssertFalse(try NFKMLXWeights.loadCheckpoint(url: url).needsConvTranspose,
                       "a saved checkpoint records the module's own layout", file: file, line: line)
        try load(destination, url)

        let before = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let after = Dictionary(uniqueKeysWithValues: destination.parameters().flattened())
        XCTAssertEqual(before.count, after.count, "the same parameters come back", file: file, line: line)
        for (key, original) in before {
            guard let reloaded = after[key] else {
                XCTFail("\(key) is missing after the round trip", file: file, line: line); continue
            }
            XCTAssertEqual(original.shape, reloaded.shape, "\(key) keeps its shape", file: file, line: line)
            eval(original, reloaded)
            let worst = (original - reloaded).abs().max().item(Float.self)
            XCTAssertEqual(worst, 0, accuracy: 0, "\(key) reloads exactly", file: file, line: line)
        }
    }

    func testSAMRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        let source = NFKMLXSAM.makeNet()
        let destination = NFKMLXSAM.makeNet()
        try assertRoundTrips(source, into: destination) { net, url in
            try NFKMLXSAM.loadWeights(into: net as! NFKMLXSAMNet, from: url)
        }
    }

    func testColorizerRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXColorizer.makeNet(), into: NFKMLXColorizer.makeNet()) { net, url in
            try NFKMLXColorizer.loadWeights(into: net as! NFKMLXColorizerNet, from: url)
        }
    }

    func testDemucsRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXDemucs.makeNet(), into: NFKMLXDemucs.makeNet()) { net, url in
            try NFKMLXDemucs.loadWeights(into: net as! NFKMLXDemucsNet, from: url)
        }
    }
}
