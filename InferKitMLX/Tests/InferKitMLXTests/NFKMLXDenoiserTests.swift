//
//  NFKMLXDenoiserTests.swift
//  InferKitMLXTests
//
//  The Demucs-U-Net speech denoiser (single output channel). The forward and round-trip evaluate MLX
//  arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDenoiserTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    func testTheNetworkHasASingleOutputChannel() throws {
        try requireMLXRuntime()
        let net = NFKMLXDenoiser.makeNet(baseChannels: 8, depth: 2)
        let cleaned = net.separate(Self.tone(samples: 512))
        eval(cleaned)
        XCTAssertEqual(cleaned.shape, [1, 512, 1], "one cleaned waveform at the input length")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = NFKMLXDenoiser.makeNet(baseChannels: 8, depth: 2)
        let onDisk = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value -> (String, MLXArray) in
            if value.ndim == 4 { return (key, value.transposed(0, 3, 1, 2)) }
            if value.ndim == 3 { return (key, value.transposed(0, 2, 1)) }
            return (key, value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("denoiser-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: onDisk, url: url)

        let loaded = NFKMLXDenoiser.makeNet(baseChannels: 8, depth: 2)
        try NFKMLXDemucs.loadWeights(into: loaded, from: url)

        let waveform = Self.tone(samples: 256)
        let expected = trained.separate(waveform)
        let actual = loaded.separate(waveform)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheBackendReturnsOneCleanedClip() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXDenoiser.backend(weightsURL: nil)
        let wave = NFKMLXWaveFile.data(samples: Self.floats(samples: 8192), sampleRate: 16000)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset, "cleaned clip present")
        XCTAssertNotNil(asset.fileURL)
        XCTAssertEqual(asset.sampleRate, 16000)
    }

    // MARK: Helpers

    static func tone(samples: Int) -> MLXArray {
        floats(samples: samples).withUnsafeBufferPointer { MLXArray($0, [samples]) }
    }

    static func floats(samples: Int) -> [Float] {
        (0 ..< samples).map { sinf(Float($0) * 0.05) * 0.4 + sinf(Float($0) * 0.23) * 0.2 }
    }
}
