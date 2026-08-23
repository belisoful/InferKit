//
//  NFKMLXAudioTaggerTests.swift
//  InferKitMLXTests
//
//  The PANNs-style audio tagger. The forward, top-K decode, and weight round-trip evaluate MLX arrays,
//  so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXAudioTaggerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXAudioTaggerNet {
        NFKMLXAudioTagger.makeNet(.tiny)
    }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["bn0.weight", "conv_block.0.conv1.weight", "conv_block.0.bn2.running_mean",
                         "conv_block.1.conv2.weight", "fc1.weight", "fc_audioset.bias"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // The reference numbers its blocks from one, and holds its front end's filterbank in the checkpoint.
    func testTheReferenceBlockNumberingRemapsOntoTheModuleArray() {
        XCTAssertEqual(NFKMLXAudioTagger.remapReferenceKey("conv_block1.conv1.weight"), "conv_block.0.conv1.weight")
        XCTAssertEqual(NFKMLXAudioTagger.remapReferenceKey("conv_block6.bn2.running_var"), "conv_block.5.bn2.running_var")
        XCTAssertEqual(NFKMLXAudioTagger.remapReferenceKey("fc_audioset.weight"), "fc_audioset.weight")
        XCTAssertEqual(NFKMLXAudioTagger.remapReferenceKey("conv_block.0.conv1.weight"), "conv_block.0.conv1.weight",
                       "a module key passes through, so a round-trip reloads")
    }

    func testTaggingReturnsTopKRankedClassesWithProbabilities() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let labels = ["speech", "music", "dog", "siren", "rain", "engine"]
        let tags = net.tag(Self.floats(samples: 8000), sampleRate: 16000, labels: labels)
        XCTAssertEqual(tags.count, NFKMLXAudioTaggerConfiguration.tiny.topK, "top-K tags")
        for tag in tags {
            XCTAssertGreaterThanOrEqual(tag.confidence, 0)
            XCTAssertLessThanOrEqual(tag.confidence, 1)
            XCTAssertLessThan(tag.classIndex, NFKMLXAudioTaggerConfiguration.tiny.classCount)
            XCTAssertEqual(tag.label, labels[tag.classIndex], "labels are attached")
        }
        // Ranked most-confident first.
        for i in 1 ..< tags.count {
            XCTAssertGreaterThanOrEqual(tags[i - 1].confidence, tags[i].confidence)
        }
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()
        let onDisk = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value -> (String, MLXArray) in
            value.ndim == 4 ? (key, value.transposed(0, 3, 1, 2)) : (key, value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("tagger-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: onDisk, url: url)

        let loaded = tinyNet()
        try NFKMLXAudioTagger.loadWeights(into: loaded, from: url)

        let mel = trained.frontEnd.logMel(Self.floats(samples: 4000))
        let expected = trained.logits(mel)
        let actual = loaded.logits(mel)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheBackendReturnsClassifications() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXAudioTagger.backend(weightsURL: nil, labels: nil)
        let wave = NFKMLXWaveFile.data(samples: Self.floats(samples: 16000), sampleRate: 16000)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        let tags = try XCTUnwrap(result.classifications, "classifications present")
        XCTAssertEqual(tags.count, NFKMLXAudioTaggerConfiguration.panns.topK)
    }

    func testAClipAtAnotherRateIsResampledRatherThanMisread() throws {
        try requireMLXRuntime()
        // The filterbank is built for one rate. Feeding a clip recorded at another puts every
        // frequency in the wrong mel bin, which changes the tags without ever looking like an error.
        let net = NFKMLXAudioTagger.makeNet(.tiny)
        let modelRate = NFKMLXAudioTaggerConfiguration.tiny.sampleRate
        let atModelRate = Self.harmonics(seconds: 0.5, rate: modelRate)
        let atDoubleRate = Self.harmonics(seconds: 0.5, rate: modelRate * 2)

        let expected = net.tag(atModelRate, sampleRate: modelRate, labels: nil)
        let resampled = net.tag(atDoubleRate, sampleRate: modelRate * 2, labels: nil)
        XCTAssertEqual(resampled.first?.classIndex, expected.first?.classIndex,
                       "the same sound gets the same top class whichever rate it arrives at")
        let confidence = abs((resampled.first?.confidence ?? 0) - (expected.first?.confidence ?? 0))
        XCTAssertLessThan(confidence, 0.05, "and a comparable score, within resampling error")
    }

    /// A harmonic stack at a given rate, so the same sound exists at two sample rates.
    static func harmonics(seconds: Double, rate: Int) -> [Float] {
        let count = Int(seconds * Double(rate))
        return (0 ..< count).map { index in
            let t = Double(index) / Double(rate)
            var value = 0.0
            for harmonic in 1 ... 6 {
                value += 0.3 / Double(harmonic) * sin(2 * .pi * 220 * Double(harmonic) * t)
            }
            return Float(value)
        }
    }

    static func floats(samples: Int) -> [Float] {
        (0 ..< samples).map { sinf(Float($0) * 0.05) * 0.3 + sinf(Float($0) * 0.2) * 0.2 }
    }
}
