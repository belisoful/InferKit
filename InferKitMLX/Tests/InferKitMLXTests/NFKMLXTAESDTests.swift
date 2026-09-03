//
//  NFKMLXTAESDTests.swift
//  InferKitMLXTests
//
//  TAESD. Encode/decode evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild`.
//

import XCTest
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXTAESDTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    func testParameterNamesFollowTheSequentialLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXTAESD.makeNet().parameters().flattened().map(\.0))
        // The [Module] arrays reproduce the checkpoint's numeric Sequential keys directly.
        for expected in ["encoder.0.weight", "encoder.1.conv.0.weight", "encoder.2.weight",
                         "encoder.14.weight", "decoder.1.weight", "decoder.3.conv.0.weight",
                         "decoder.19.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
        // The stride-2 downsample convolutions carry no bias.
        XCTAssertFalse(names.contains("encoder.2.bias"), "downsample convolutions are bias-free")
    }

    func testEncodeDecodeShapesAreEightTimesDown() throws {
        try requireMLXRuntime()
        let net = NFKMLXTAESD.makeNet()
        let image = MLXRandom.uniform(low: 0, high: 1, [1, 128, 128, 3])
        let latent = net.encode(image)
        XCTAssertEqual(latent.shape, [1, 16, 16, 4], "8× spatial reduction to 4 latent channels")
        let reconstruction = net.decode(latent)
        XCTAssertEqual(reconstruction.shape, [1, 128, 128, 3], "8× upsampling back to RGB")
    }

    func testASafetensorsCheckpointRoundTripsThroughTheLoader() throws {
        try requireMLXRuntime()
        let trained = NFKMLXTAESD.makeNet()
        // Save in PyTorch Conv2d layout (`[out, in, kH, kW]`), the inverse of the loader's transpose.
        let onDisk = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value -> (String, MLXArray) in
            value.ndim == 4 ? (key, value.transposed(0, 3, 1, 2)) : (key, value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("taesd-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: onDisk, url: url)

        let loaded = NFKMLXTAESD.makeNet()
        try NFKMLXTAESD.loadWeights(into: loaded, from: url)
        let image = MLXRandom.uniform(low: 0, high: 1, [1, 64, 64, 3])
        eval(trained.encode(image), loaded.encode(image))
        XCTAssertEqual(trained.encode(image).reshaped([-1]).asArray(Float.self),
                       loaded.encode(image).reshaped([-1]).asArray(Float.self),
                       "loaded weights reproduce the encoder")
    }

    func testTheBackendReconstructsAnImage() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXTAESD.backend(weightsURL: nil)
        let image = NFKMLXSigLIP2Tests.solidImage(side: 64)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage))
    }
}
