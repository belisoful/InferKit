//
//  NFKMLXReadQuantizationTests.swift
//  InferKitMLXTests
//
//  Weight-free tests for a decoder load that quantizes as it reads (`NFKMLXReadQuantization`): a random
//  decoder is written out as a sharded bfloat16 release, and a load that packs each layer as its group is
//  read must hold the same arrays as a load followed by the quantization, element for element, must hold
//  every layer whatever its budget, and must peak below the release at full width.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXReadQuantizationTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func geometry(tied: Bool) -> NFKMLXLanguageConfiguration {
        var configuration = NFKMLXLanguageConfiguration.tiny
        configuration.layerCount = 4
        configuration.tiesWordEmbeddings = tied
        return configuration
    }

    /// `geometry` written as a two-shard bfloat16 release with its index and `config.json`, each tensor
    /// drawn and written one shard at a time so the release is never held whole.
    private func release(_ geometry: NFKMLXLanguageConfiguration) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("read-quantization-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let config: [String: Any] = [
            "model_type": "qwen3", "hidden_size": geometry.hiddenSize, "num_hidden_layers": geometry.layerCount,
            "num_attention_heads": geometry.headCount, "num_key_value_heads": geometry.keyValueHeadCount,
            "head_dim": geometry.headDimensions, "intermediate_size": geometry.intermediateSize,
            "vocab_size": geometry.vocabularySize, "rope_theta": geometry.ropeTheta,
            "tie_word_embeddings": geometry.tiesWordEmbeddings,
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        MLXRandom.seed(7)
        let shapes = NFKMLXLanguage.makeNet(geometry).parameters().flattened()
            .map { ($0.0, $0.1.shape) }.sorted { $0.0 < $1.0 }
        var weightMap = [String: String]()
        for (index, half) in [shapes.prefix(shapes.count / 2), shapes.dropFirst(shapes.count / 2)].enumerated() {
            let name = "model-0000\(index + 1)-of-00002.safetensors"
            let arrays = Dictionary(half.map { key, shape in
                (key, (key.hasSuffix("norm.weight") ? 1 + MLXRandom.normal(shape) * 0.1 : MLXRandom.normal(shape) * 0.02)
                    .asType(.bfloat16))
            }, uniquingKeysWith: { first, _ in first })
            try save(arrays: arrays, url: directory.appendingPathComponent(name))
            half.forEach { weightMap[$0.0] = name }
        }
        try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    private func loaded(_ directory: URL, _ geometry: NFKMLXLanguageConfiguration, residency: NFKMLXResidency = .resident,
                        budget: Int = NFKMLXResidencyBudget.current(),
                        quantization: (bits: Int, groupSize: Int)?) throws -> NFKMLXLanguageNet {
        let net = NFKMLXLanguage.makeNet(geometry)
        try NFKMLXLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint, residency: residency,
                                       budget: budget, quantization: quantization)
        return net
    }

    func testAPackedReadHoldsWhatALoadThenAQuantizationHolds() throws {
        try requireMLXRuntime()
        for tied in [true, false] {
            let geometry = geometry(tied: tied)
            let directory = try release(geometry)
            for bits in [4, 8] {
                let label = "\(tied ? "tied" : "untied") \(bits)-bit"
                let quantizedAfter = try loaded(directory, geometry, quantization: nil)
                try NFKMLXQuantization.quantize(module: quantizedAfter, bits: bits, groupSize: 64)
                let packedAsRead = try loaded(directory, geometry, quantization: (bits, 64))
                let expected = Dictionary(uniqueKeysWithValues: quantizedAfter.parameters().flattened())
                let actual = Dictionary(uniqueKeysWithValues: packedAsRead.parameters().flattened())
                XCTAssertEqual(Set(actual.keys), Set(expected.keys), label)
                XCTAssertTrue(expected.keys.contains { $0.hasSuffix(".scales") }, "\(label): the layers are packed")
                for (key, value) in expected {
                    guard let other = actual[key] else { continue }
                    XCTAssertEqual(other.dtype, value.dtype, "\(label): \(key)")
                    XCTAssertTrue(arrayEqual(other, value).item(Bool.self), "\(label): \(key) differs")
                }
            }
        }
    }

    // Under a budget that streams the unquantized decoder's later layers, a quantized load asked to stream is
    // refused before it reads a weight, and an automatic one holds every layer.
    func testAQuantizedLoadHoldsEveryLayerUnderABudgetThatStreams() throws {
        try requireMLXRuntime()
        let geometry = geometry(tied: false)
        let directory = try release(geometry)
        let stage = try NFKMLXLayerStream.footprint(directory: directory, precision: .checkpoint, includes: { _ in true },
                                                    layerOf: { NFKMLXLanguage.streamedLayer(ofModuleKey: $0)?.layer })
        let budget = NFKMLXResidencyBudget.leastBudget(holding: stage.streamedMinimumBytes + stage.layerBytes.prefix(1).reduce(0, +))
        let streamed = try loaded(directory, geometry, residency: .streamed, budget: budget, quantization: nil)
        XCTAssertNotNil(streamed.layerStream, "the control streams under this budget")
        XCTAssertThrowsError(try loaded(directory, geometry, residency: .streamed, budget: budget, quantization: (4, 64))) {
            XCTAssertTrue("\($0)".contains("does not quantize at load"), "\($0)")
        }
        let quantized = try loaded(directory, geometry, residency: .automatic, budget: budget, quantization: (4, 64))
        XCTAssertNil(quantized.layerStream)
        XCTAssertTrue(quantized.parameters().flattened().contains { $0.0.hasSuffix(".scales") })
    }

    // A release of about 1 GB at bfloat16: a packed read holds the 4-bit layers and one group of about 256 MB,
    // where a load then a quantization holds the release whole first.
    func testAPackedReadPeaksBelowTheReleaseAtFullWidth() throws {
        try requireMLXRuntime()
        var geometry = NFKMLXLanguageConfiguration.tiny
        geometry.hiddenSize = 2048
        geometry.layerCount = 8
        geometry.headCount = 16
        geometry.keyValueHeadCount = 8
        geometry.headDimensions = 128
        geometry.intermediateSize = 8192
        let directory = try release(geometry)
        let unpacked = try NFKMLXReleaseWeights.weightBytes(inDirectory: directory)
        func peak(_ quantization: (bits: Int, groupSize: Int)?) throws -> Int {
            NFKMLXGPU.clearCache()
            NFKMLXGPU.resetPeakAllocatedByteCount()
            let before = NFKMLXGPU.allocatedByteCount
            var net: NFKMLXLanguageNet? = try loaded(directory, geometry, quantization: quantization)
            if quantization == nil, let held = net {
                try NFKMLXQuantization.quantize(module: held, bits: 4, groupSize: 64)
                eval(held)
            }
            let peak = NFKMLXGPU.peakAllocatedByteCount - before
            net = nil
            return peak
        }
        let afterLoad = try peak(nil)
        let asRead = try peak((4, 64))
        XCTAssertGreaterThanOrEqual(afterLoad, unpacked, "the control holds the release whole")
        XCTAssertLessThan(asRead, unpacked * 4 / 5, "a packed read peaks at \(asRead) bytes against \(unpacked) unpacked")
    }
}
