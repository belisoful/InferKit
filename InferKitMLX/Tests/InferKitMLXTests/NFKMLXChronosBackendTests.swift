//
//  NFKMLXChronosBackendTests.swift
//  InferKitMLXTests
//
//  Chronos-Bolt behind the InferKit contract: the series and the horizon reach the forecaster, and the
//  median, quantile rows, and levels come back under NFKOutputStructured as the object computes them.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXChronosBackendTests: XCTestCase {

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

    /// The tiny forecaster, predicting 8 steps.
    private func tinyForecaster() -> NFKMLXChronos {
        NFKMLXRandom.seed(3)
        return NFKMLXChronos(net: NFKMLXChronos.makeNet(.tiny))
    }

    private let series = (0 ..< 40).map { sinf(Float($0) * 0.3) * 5 + 10 }

    private func fields(_ result: NFKInferenceResult) throws -> [String: Any] {
        try XCTUnwrap(result.output(forKey: NFKOutputStructured) as? [String: Any])
    }

    func testASeriesBecomesTheForecastAtTheRequestedHorizon() throws {
        try requireMLXRuntime()
        let forecaster = tinyForecaster()
        let backend = forecaster.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXChronos.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKMLXForecastKey.series])
        XCTAssertEqual(backend.supportedParameterKeys, [NFKMLXForecastKey.horizon])

        let result = try fields(backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKMLXForecastKey.series: series], parameters: [NFKMLXForecastKey.horizon: 5])))
        let median = try XCTUnwrap(result[NFKMLXForecastKey.median] as? [NSNumber]).map(\.floatValue)
        let quantiles = try XCTUnwrap(result[NFKMLXForecastKey.quantiles] as? [[NSNumber]])
        let levels = try XCTUnwrap(result[NFKMLXForecastKey.quantileLevels] as? [NSNumber]).map(\.floatValue)
        let rows = forecaster.forecast(context: series, horizon: 5)
        XCTAssertEqual(median, rows[4], "the median is the 0.5 row")
        XCTAssertEqual(quantiles.map { $0.map(\.floatValue) }, rows)
        XCTAssertEqual(quantiles.count, 9)
        XCTAssertEqual(levels, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        XCTAssertNil(result[NFKMLXForecastKey.mean], "Chronos-Bolt predicts no mean")

        let unbounded = try fields(backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKMLXForecastKey.series: series], parameters: [:])))
        XCTAssertEqual((unbounded[NFKMLXForecastKey.median] as? [NSNumber])?.count, 8,
                       "a request without a horizon forecasts the prediction length, not the last request's")
    }

    func testARequestWithoutASeriesIsRefused() throws {
        try requireMLXRuntime()
        let backend = tinyForecaster().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:], parameters: [:])))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKMLXForecastKey.series: [NSNumber]()], parameters: [:])))
    }
}
