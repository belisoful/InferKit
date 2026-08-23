//
//  NFKMLXDemucsTests.swift
//  InferKitMLXTests
//
//  A time-domain separation U-Net. These evaluate MLX arrays, so they skip under `swift test` and run
//  under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDemucsTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    func testSeparateReturnsFourStereoStemsAtInputLength() throws {
        try requireMLXRuntime()
        let net = NFKMLXDemucs.makeNet(Self.tinyConfiguration())
        let stems = net.separate(Self.samples(1024))
        eval(stems)
        XCTAssertEqual(stems.shape, [1, 1024, 8], "four stereo stems at the input length")
    }

    func testTheDefaultConfigurationIsTheReleasedMusicModel() {
        let configuration = NFKMLXDemucsConfiguration()
        XCTAssertEqual(configuration.audioChannels, 2)
        XCTAssertEqual(configuration.depth, 6)
        XCTAssertEqual(configuration.context, 3, "the decoder mixes channels over three time steps")
        XCTAssertTrue(configuration.bidirectional, "the music bottleneck runs both directions")
        XCTAssertFalse(configuration.normalize, "the music model trains without input normalization")
    }

    func testACheckpointRoundTripReproducesTheSeparation() throws {
        try requireMLXRuntime()
        let trained = NFKMLXDemucs.makeNet(Self.tinyConfiguration())
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value -> (String, MLXArray) in
            if value.ndim == 4 { return (key, value.transposed(0, 3, 1, 2)) }
            if value.ndim == 3 { return (key, value.transposed(0, 2, 1)) }
            return (key, value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("demucs-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXDemucs.makeNet(Self.tinyConfiguration())
        try NFKMLXDemucs.loadWeights(into: loaded, from: url)

        let input = Self.samples(512)
        let expected = trained.separate(input)
        let actual = loaded.separate(input)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    func testTheBackendWritesAWaveFilePerStem() throws {
        try requireMLXRuntime()
        let inputURL = FileManager.default.temporaryDirectory.appendingPathComponent("demucs-in-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: inputURL) }
        try NFKMLXWaveFile.write(samples: (0 ..< 1024).map { sinf(Float($0) * 0.1) }, sampleRate: 16000, to: inputURL)

        let backend = NFKMLXDemucsBackend(net: NFKMLXDemucs.makeNet(Self.tinyConfiguration()))
        let request = NFKInferenceRequest(inputs: [NFKInputAudio: NFKAudioAsset(fileURL: inputURL, durationSeconds: 0, sampleRate: 16000, channelCount: 1)])
        let result = try backend.runInference(for: request)
        for name in ["drums", "bass", "other", "vocals"] {
            let stem = try XCTUnwrap(result.output(forKey: name) as? NFKAudioAsset, "\(name) stem")
            XCTAssertEqual(stem.channelCount, 2, "\(name) keeps the model's stereo channels")
            let url = try XCTUnwrap(stem.fileURL)
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertNotNil(NFKMLXWaveFile.read(try Data(contentsOf: url)), "\(name) is a readable WAV")
        }
    }

    /// The released geometry at a size a test can run: same stereo, context, bottleneck, and resampler,
    /// two shallow blocks instead of six wide ones.
    static func tinyConfiguration() -> NFKMLXDemucsConfiguration {
        var configuration = NFKMLXDemucsConfiguration()
        configuration.baseChannels = 8
        configuration.depth = 2
        return configuration
    }

    static func samples(_ count: Int) -> MLXArray {
        let values = (0 ..< count).map { Float(($0 * 13) % 100) / 100 - 0.5 }
        return values.withUnsafeBufferPointer { MLXArray($0, [count]) }
    }
}
