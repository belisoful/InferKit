//
//  NFKMLXTimesFMBackendTests.swift
//  InferKitMLXTests
//
//  TimesFM 2.5 behind the InferKit contract: the series, the horizon, and the forecast options reach the
//  forecaster, the median, mean, quantile rows, and levels come back under NFKOutputStructured as the
//  object computes them, and a request's options do not outlive it.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXTimesFMBackendTests: XCTestCase {

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

    /// The TimesFM tests' tiny geometry, forecasting 64-step output patches.
    private func tinyForecaster() -> NFKMLXTimesFM {
        NFKMLXRandom.seed(5)
        return NFKMLXTimesFM(net: NFKMLXTimesFMNet(NFKMLXTimesFMTests.tiny))
    }

    private let series = (0 ..< 100).map { NSNumber(value: sinf(Float($0) * 0.2) * 3 - 1) }

    private func median(_ result: NFKInferenceResult) throws -> [Float] {
        let fields = try XCTUnwrap(result.output(forKey: NFKOutputStructured) as? [String: Any])
        return try XCTUnwrap(fields[NFKMLXForecastKey.median] as? [NSNumber]).map(\.floatValue)
    }

    func testASeriesBecomesTheForecastAtTheRequestedHorizon() throws {
        try requireMLXRuntime()
        let forecaster = tinyForecaster()
        let backend = forecaster.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXTimesFM.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKMLXForecastKey.series])
        XCTAssertEqual(backend.supportedParameterKeys,
                       NFKMLXTimesFMParameterKey.all.union([NFKMLXForecastKey.horizon]))

        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKMLXForecastKey.series: series], parameters: [NFKMLXForecastKey.horizon: 16]))
        let fields = try XCTUnwrap(result.output(forKey: NFKOutputStructured) as? [String: Any])
        let reference = try forecaster.forecast(context: series, horizon: 16, options: nil)
        XCTAssertEqual(try median(result), reference.pointForecast.map(\.floatValue))
        XCTAssertEqual((fields[NFKMLXForecastKey.mean] as? [NSNumber])?.map(\.floatValue),
                       reference.meanForecast.map(\.floatValue))
        let quantiles = try XCTUnwrap(fields[NFKMLXForecastKey.quantiles] as? [[NSNumber]])
        XCTAssertEqual(quantiles.count, 9)
        XCTAssertEqual(quantiles.map(\.count), Array(repeating: 16, count: 9))
        XCTAssertEqual((fields[NFKMLXForecastKey.quantileLevels] as? [NSNumber])?.count, 9)

        let unbounded = try median(backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKMLXForecastKey.series: series], parameters: [:])))
        XCTAssertEqual(unbounded.count, 64, "a request without a horizon forecasts one output patch")
    }

    func testARequestsOptionsApplyToThatRequestOnly() throws {
        try requireMLXRuntime()
        let forecaster = tinyForecaster()
        let backend = forecaster.inferenceBackend()
        let options = NFKMLXTimesFMForecastOptions()
        options.forcesFlipInvariance = false
        options.normalizesInputs = false

        let adjusted = try median(backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKMLXForecastKey.series: series],
            parameters: [NFKMLXForecastKey.horizon: 16, NFKMLXTimesFMParameterKey.forcesFlipInvariance: false,
                         NFKMLXTimesFMParameterKey.normalizesInputs: false])))
        let expected = try forecaster.forecast(context: series, horizon: 16, options: options).pointForecast
        XCTAssertEqual(adjusted, expected.map(\.floatValue))

        let defaults = try median(backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKMLXForecastKey.series: series], parameters: [NFKMLXForecastKey.horizon: 16])))
        let reference = try forecaster.forecast(context: series, horizon: 16, options: nil).pointForecast
        XCTAssertEqual(defaults, reference.map(\.floatValue), "the next request forecasts at the defaults")
        XCTAssertNotEqual(adjusted, defaults, "the options change the forecast, so the comparison discriminates")
    }

    func testARequestWithoutASeriesOrBeyondTheLimitIsRefused() throws {
        try requireMLXRuntime()
        let backend = tinyForecaster().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKMLXForecastKey.series: series],
            parameters: [NFKMLXForecastKey.horizon: NFKMLXTimesFMTests.tiny.contextLimit])))
    }
}
