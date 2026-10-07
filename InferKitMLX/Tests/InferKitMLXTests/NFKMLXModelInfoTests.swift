//
//  NFKMLXModelInfoTests.swift
//  InferKitMLXTests
//
//  What the language backends report under `modelInfo`, and what NFKMLXGPU reports to an
//  NFKInferenceServer's status route. Building a net reaches MLX and skips without a Metal library.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXModelInfoTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    func testAFloatNetReportsTheParametersTheSizingCounts() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXLanguageConfiguration.tiny
        let info = NFKMLXLanguageBackend(net: NFKMLXLanguage.makeNet(configuration), tokenizer: nil,
                                         identifier: "tiny").modelInfo
        let parameters = NFKMLXModelSizing.parameterCount(of: configuration)
        XCTAssertEqual(info[NFKModelInfoParameterCount] as? Int, parameters)
        XCTAssertEqual(info[NFKModelInfoWeightBytes] as? Int, 4 * parameters)
        XCTAssertEqual(info[NFKModelInfoPrecision] as? String, "float32")
        XCTAssertNil(info[NFKModelInfoQuantizationBits])
        XCTAssertEqual(info[NFKModelInfoKeyValueBytesPerToken] as? Int,
                       2 * configuration.layerCount * configuration.keyValueHeadCount * configuration.headDimensions * 4)
        XCTAssertNil(info[NFKModelInfoContextLength], "a configuration built in code states no positions")
    }

    func testAQuantizedNetCountsItsPackedWeightsAtTheirBits() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXLanguageConfiguration.tiny
        let net = NFKMLXLanguage.makeNet(configuration)
        try NFKMLXQuantization.quantize(module: net, bits: 4, groupSize: 64, includeEmbeddings: true)
        let info = NFKMLXLanguageBackend(net: net, tokenizer: nil, identifier: "tiny-q4").modelInfo
        let parameters = NFKMLXModelSizing.parameterCount(of: configuration)
        XCTAssertEqual(info[NFKModelInfoParameterCount] as? Int, parameters, "packed words count at 32 over the bits")
        XCTAssertEqual(info[NFKModelInfoQuantizationBits] as? Int, 4)
        XCTAssertEqual(info[NFKModelInfoQuantizationGroupSize] as? Int, 64)
        XCTAssertLessThan(info[NFKModelInfoWeightBytes] as? Int ?? .max, 4 * parameters)
        XCTAssertEqual(info[NFKModelInfoPrecision] as? String, "float32", "the norms stay float32")
    }

    func testAReleaseConfigurationKeepsItsTypeAndPositions() throws {
        let configuration = try NFKMLXLanguage.configuration(fromJSON: [
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "max_position_embeddings": 40960,
        ])
        XCTAssertEqual(configuration.modelType, "qwen3")
        XCTAssertEqual(configuration.maximumPositions, 40960)
        XCTAssertNil(try NFKMLXLanguage.configuration(fromJSON: [:]).maximumPositions)
    }

    private func releaseDirectory(config: [String: Any]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXModelInfoTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    func testAReleaseDirectoryStatesItsTypeAndPositionsAtEitherLevel() throws {
        let flat = try releaseDirectory(config: ["model_type": "granitemoehybrid", "max_position_embeddings": 131072])
        let flatInfo = NFKMLXModelDescription.info(of: [], releaseDirectoryURL: flat)
        XCTAssertEqual(flatInfo[NFKModelInfoArchitecture] as? String, "granitemoehybrid")
        XCTAssertEqual(flatInfo[NFKModelInfoContextLength] as? Int, 131072)
        XCTAssertNil(flatInfo[NFKModelInfoParameterCount], "no modules, no parameter count")

        let nested = try releaseDirectory(config: ["model_type": "qwen3_5", "text_config": ["max_position_embeddings": 262144]])
        let nestedInfo = NFKMLXModelDescription.info(of: [], releaseDirectoryURL: nested)
        XCTAssertEqual(nestedInfo[NFKModelInfoArchitecture] as? String, "qwen3_5")
        XCTAssertEqual(nestedInfo[NFKModelInfoContextLength] as? Int, 262144)

        let stateless = try releaseDirectory(config: ["model_type": "mamba2"])
        XCTAssertNil(NFKMLXModelDescription.info(of: [], releaseDirectoryURL: stateless)[NFKModelInfoContextLength])
        XCTAssertTrue(NFKMLXModelDescription.info(of: [], releaseDirectoryURL: nil).isEmpty)
    }

    func testABackendOverAForwardDescribesItsNetworkAndRelease() throws {
        try requireMLXRuntime()
        let directory = try releaseDirectory(config: ["model_type": "nemotron_h", "max_position_embeddings": 131072])
        try JSONSerialization.data(withJSONObject: ["a": 0, "b": 1]).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        let tokenizer = try NFKTokenizer(forManifest: ["tokenizer": ["type": "bpe-bytelevel"]], directory: directory)
        let projection = Linear(4, 8)
        let backend = NFKMLXNemotronBackend(logits: { projection($0) }, tokenizer: tokenizer, identifier: "projection",
                                            modules: [projection], releaseDirectoryURL: directory)
        let info = backend.modelInfo
        XCTAssertEqual(info[NFKModelInfoParameterCount] as? Int, 4 * 8 + 8)
        XCTAssertEqual(info[NFKModelInfoWeightBytes] as? Int, 4 * (4 * 8 + 8))
        XCTAssertEqual(info[NFKModelInfoPrecision] as? String, "float32")
        XCTAssertEqual(info[NFKModelInfoArchitecture] as? String, "nemotron_h")
        XCTAssertEqual(info[NFKModelInfoContextLength] as? Int, 131072)
    }

    func testTheDecoderBackendDescribesTheNetworksItIsGiven() throws {
        try requireMLXRuntime()
        let directory = try releaseDirectory(config: ["model_type": "qwen3_5", "text_config": ["max_position_embeddings": 262144]])
        try JSONSerialization.data(withJSONObject: ["a": 0, "b": 1]).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        let projection = Linear(4, 8)
        let backend = try NFKMLXDecoderBackend.release(directoryURL: directory, identifier: "projection",
                                                       modules: [projection]) { projection($0) }
        XCTAssertEqual(backend.modelInfo[NFKModelInfoParameterCount] as? Int, 4 * 8 + 8)
        XCTAssertEqual(backend.modelInfo[NFKModelInfoContextLength] as? Int, 262144)
    }

    func testAServedLanguageModelReportsItsModelAndMLXMemory() throws {
        try requireMLXRuntime()
        let server = NFKInferenceServer()
        server.port = 0
        server.loopbackOnly = true
        server.addBackend(NFKMLXLanguageBackend(net: NFKMLXLanguage.makeNet(.tiny), tokenizer: nil, identifier: "tiny"),
                          forModelName: "tiny")
        try server.start()
        defer { server.stop() }
        let url = try XCTUnwrap(server.localBaseURL).appendingPathComponent("inferkit/status")
        let status = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let model = ((status["models"] as? [[String: Any]])?.first?["model"]) as? [String: Any]
        XCTAssertEqual(model?[NFKModelInfoParameterCount] as? Int, NFKMLXModelSizing.parameterCount(of: .tiny))
        let mlx = ((status["host"] as? [String: Any])?["runtimes"] as? [String: Any])?["mlx"] as? [String: Any]
        XCTAssertEqual(mlx?["metal_library_found"] as? Bool, true)
        XCTAssertNotNil(mlx?["active_memory_bytes"])
        XCTAssertNotNil(mlx?["memory_limit_bytes"])
        XCTAssertNil(mlx?["cache_limit_bytes"], "reading the cache limit would trim the cache")
    }
}
