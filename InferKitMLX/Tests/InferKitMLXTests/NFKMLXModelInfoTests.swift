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
