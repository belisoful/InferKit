//
//  NFKMLXRepositoryMatchingForecastingTests.swift
//  InferKitMLXTests
//
//  TimesFM 2.5's judgment of another repository, from files placed in a hub cache: both layouts read, the
//  1.0 and 2.0 networks and the fields the 2.5 network fixes refused. Nothing here reaches MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXRepositoryMatchingForecastingTests: XCTestCase {

    private var fixture: NFKMLXRepositoryFixture!

    override func setUp() {
        super.setUp()
        fixture = NFKMLXRepositoryFixture()
    }

    override func tearDown() {
        fixture.remove()
        super.tearDown()
    }

    /// The official layout's configuration (`timesfm`), two layers of width 64.
    private static var official: [String: Any] {
        ["model_type": "timesfm", "architectures": ["TimesFmModelForPrediction"], "context_length": 16384,
         "head_dim": 16, "hidden_size": 64, "horizon_length": 128, "intermediate_size": 64, "num_attention_heads": 4,
         "num_hidden_layers": 2, "patch_length": 32, "quantile_horizon_length": 1024,
         "quantiles": [0.1, 0.5, 0.9], "rms_norm_eps": 1e-6]
    }

    /// transformers' layout of the same network (`timesfm2_5`), stating the fields the network fixes.
    private static var transformers: [String: Any] {
        var configuration = official
        configuration["model_type"] = "timesfm2_5"
        configuration["architectures"] = ["TimesFm2_5ModelForPrediction"]
        configuration["quantile_horizon_length"] = nil
        configuration["output_quantile_len"] = 1024
        configuration["activation"] = "swish"
        configuration["attention_bias"] = false
        configuration["use_bias"] = false
        configuration["num_key_value_heads"] = 4
        configuration["rope_parameters"] = ["rope_theta": 10000.0, "rope_type": "default"]
        return configuration
    }

    private func repository(_ configuration: [String: Any]) throws -> NFKModelRepository {
        let weights = NFKMLXRepositoryFixture.safetensors([("stacked_xf.0.ff0.weight", [64, 64]),
                                                           ("stacked_xf.1.ff0.weight", [64, 64])],
                                                          dtype: "F32", elementBytes: 4)
        return try fixture.repository("org/timesfm-tiny",
                                      files: ["config.json": NFKMLXRepositoryFixture.json(configuration),
                                              "model.safetensors": weights],
                                      configuration: configuration, license: nil)
    }

    private func assertRefusal(_ code: NFKModelResolutionError.Code, _ configuration: [String: Any],
                               file: StaticString = #filePath, line: UInt = #line) throws {
        try NFKMLXRepositoryFixture.assertRefusal(code, try repository(configuration), template: NFKMLXTimesFM.modelName,
                                                  file: file, line: line)
    }

    func testBothLayoutsAreReadInTheTimesFMFamily() throws {
        for configuration in [Self.official, Self.transformers] {
            let descriptor = try NFKMLXModelDescriptors.descriptor(
                for: try repository(configuration), matching: try NFKMLXRepositoryFixture.template(NFKMLXTimesFM.modelName))
            XCTAssertEqual(descriptor.family, "timesfm")
            XCTAssertEqual(descriptor.registryName, NFKMLXTimesFM.modelName)
            XCTAssertEqual(descriptor.defaultVariant?.parameterCount, 8192)
        }
    }

    func testTimesFMOneAndTwoAreRefused() throws {
        var earlier = Self.official
        earlier["quantile_horizon_length"] = nil
        earlier["freq_size"] = 3
        earlier["use_positional_embedding"] = false
        try assertRefusal(.unsupportedArchitecture, earlier)
    }

    func testAConfigurationWithoutTheQuantileHeadIsRefused() throws {
        var configuration = Self.official
        configuration["quantile_horizon_length"] = nil
        try assertRefusal(.unsupportedArchitecture, configuration)
    }

    func testFieldsTheNetworkFixesAreRefusedAtOtherValues() throws {
        let changes: [(String, Any)] = [("activation", "gelu"), ("use_bias", true), ("attention_bias", true),
                                        ("num_key_value_heads", 2),
                                        ("rope_parameters", ["rope_theta": 1_000_000.0, "rope_type": "default"])]
        for (key, value) in changes {
            var configuration = Self.transformers
            configuration[key] = value
            try assertRefusal(.unsupportedConfiguration, configuration)
        }
    }

    func testAQuantizedForecasterIsRefused() throws {
        var configuration = Self.official
        configuration["quantization_config"] = ["quant_method": "bitsandbytes"]
        try assertRefusal(.unsupportedQuantization, configuration)
    }
}
