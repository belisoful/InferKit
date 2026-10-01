//
//  NFKMLXWanAnimateTests.swift
//  InferKitMLXTests
//
//  The Wan Animate transformer's release loader: a tiny network saved under the release's own names
//  and layout loads back unchanged. Runs where MLX has a Metal library (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXWanAnimateTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    func testAReleaseDirectoryLoadsEveryTensorUnderItsModuleName() throws {
        try requireMLXRuntime()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let trained = NFKMLXWanAnimate.makeNet(.tiny)
        // The release names its blocks `blocks.N.block.` with single-letter projections, and stores the
        // patch convolution `[out, in, kT, kH, kW]`.
        let released = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (NFKMLXWanAnimate.releaseKey(forModule: key), value.ndim == 5 ? value.transposed(0, 4, 1, 2, 3) : value)
        })
        XCTAssertTrue(released.keys.contains { $0.contains(".block.self_attn.q.") }, "the release's own naming")
        try save(arrays: released, url: directory.appendingPathComponent("diffusion_pytorch_model.safetensors"))

        let loaded = NFKMLXWanAnimate.makeNet(.tiny)
        try NFKMLXWanAnimate.loadWeights(into: loaded, fromDirectory: directory)
        let expected = Dictionary(uniqueKeysWithValues: trained.parameters().flattened())
        for (key, value) in loaded.parameters().flattened() {
            let original = try XCTUnwrap(expected[key], key)
            XCTAssertEqual(value.shape, original.shape, key)
            XCTAssertEqual(abs(value - original).max().item(Float.self), 0, key)
        }
    }
}
