//
//  NFKMLXConvTasNetTests.swift
//  InferKitMLXTests
//
//  The Conv-TasNet speech separator. The forward and the weight round-trip evaluate MLX arrays, so
//  they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXConvTasNetTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXConvTasNetNet {
        NFKMLXConvTasNetNet(.tiny)
    }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["encoder.weight", "input_norm.weight", "bottleneck.weight",
                         "blocks.0.conv1x1.weight", "blocks.0.dconv.weight", "blocks.0.prelu1.weight",
                         "blocks.0.residual.weight", "mask_conv.weight", "decoder.conv.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testSeparationYieldsOneWaveformPerSpeakerAtInputLength() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let waveform = Self.tone(samples: 256)
        let a = net.separate(waveform)
        let b = net.separate(waveform)
        eval(a, b)
        XCTAssertEqual(a.shape, [2, 256], "one waveform per speaker, at the input length")
        let values = a.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0.isFinite }, "finite samples")
        XCTAssertEqual(values, b.asArray(Float.self), "deterministic")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        // Conv1d weights are 3-D [out, k, in] in MLX; save as [out, in, k]; the transposed-conv weight
        // is 4-D. Both invert the loader's transpose.
        let onDisk = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value -> (String, MLXArray) in
            if value.ndim == 4 { return (key, value.transposed(0, 3, 1, 2)) }
            if value.ndim == 3 { return (key, value.transposed(0, 2, 1)) }
            return (key, value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("tasnet-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: onDisk, url: url)

        let loaded = tinyNet()
        try NFKMLXConvTasNet.loadWeights(into: loaded, from: url)

        let waveform = Self.tone(samples: 128)
        let expected = trained.separate(waveform)
        let actual = loaded.separate(waveform)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheBackendReturnsOneAssetPerSpeaker() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXConvTasNet.backend(weightsURL: nil)
        let wave = NFKMLXWaveFile.data(samples: Self.floats(samples: 4096), sampleRate: 8000)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        for name in NFKMLXConvTasNetConfiguration.base.speakerNames {
            let asset = try XCTUnwrap(result.output(forKey: name) as? NFKAudioAsset, "\(name) present")
            XCTAssertNotNil(asset.fileURL)
        }
    }

    // MARK: Helpers

    static func tone(samples: Int) -> MLXArray {
        floats(samples: samples).withUnsafeBufferPointer { MLXArray($0, [samples]) }
    }

    static func floats(samples: Int) -> [Float] {
        (0 ..< samples).map { sinf(Float($0) * 0.1) * 0.5 + sinf(Float($0) * 0.31) * 0.3 }
    }

    // MARK: Per-channel PReLU

    // The released checkpoint carries ONE slope per PReLU, which is asteroid's own shape. The
    // per-channel option is for fine-tuning, and it has to load that checkpoint without changing what
    // the model computes: a shared slope broadcast across channels is the same function.
    func testPerChannelPReLUWidensTheReleasedSlopes() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXConvTasNetConfiguration.tiny
        configuration.perChannelPReLU = true
        let net = NFKMLXConvTasNetNet(configuration)

        let slopes = net.parameters().flattened()
            .filter { $0.0.hasSuffix("prelu1.weight") || $0.0.hasSuffix("prelu2.weight") }
        XCTAssertFalse(slopes.isEmpty)
        for (name, value) in slopes {
            XCTAssertEqual(value.size, configuration.hidden, "\(name) is one slope per channel")
        }

        let shared = NFKMLXConvTasNetNet(NFKMLXConvTasNetConfiguration.tiny)
        for (name, value) in shared.parameters().flattened()
        where name.hasSuffix("prelu1.weight") || name.hasSuffix("prelu2.weight") {
            XCTAssertEqual(value.size, 1, "\(name) defaults to the reference's single slope")
        }
    }

    // Saving a single-slope model and loading it into a per-channel one must not change the output:
    // that equivalence is what lets a fine-tune start from a released checkpoint.
    func testAWidenedCheckpointComputesIdentically() throws {
        try requireMLXRuntime()
        let shared = NFKMLXConvTasNetNet(NFKMLXConvTasNetConfiguration.tiny)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(shared, to: url)

        var configuration = NFKMLXConvTasNetConfiguration.tiny
        configuration.perChannelPReLU = true
        let widened = NFKMLXConvTasNetNet(configuration)
        try NFKMLXConvTasNet.loadWeights(into: widened, from: url)

        // Every widened slope is the shared one repeated.
        let widenedSlopes = Dictionary(uniqueKeysWithValues: widened.parameters().flattened())
        for (name, value) in shared.parameters().flattened()
        where name.hasSuffix("prelu1.weight") || name.hasSuffix("prelu2.weight") {
            let slope = try XCTUnwrap(widenedSlopes[name])
            XCTAssertEqual(slope.size, configuration.hidden)
            eval(slope, value)
            let spread = (slope - value.item(Float.self)).abs().max().item(Float.self)
            XCTAssertEqual(spread, 0, accuracy: 1e-9, "\(name) is the shared slope repeated")
        }

        // And the separation is unchanged, which is the property that matters.
        let waveform = MLXArray((0 ..< 4000).map { sinf(Float($0) * 0.02) })
        let before = shared.separate(waveform)
        let after = widened.separate(waveform)
        eval(before, after)
        XCTAssertEqual(before.shape, after.shape)
        let worst = (before - after).abs().max().item(Float.self)
        XCTAssertEqual(worst, 0, accuracy: 1e-5, "a broadcast slope is the same function")
    }
}
