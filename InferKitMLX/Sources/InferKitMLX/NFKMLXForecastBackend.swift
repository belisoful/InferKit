//
//  NFKMLXForecastBackend.swift
//  InferKitMLX
//
//  The request and result shape the time-series forecasting backends share.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The names a forecasting backend reads and writes. The core has no time-series keys.
///
/// @discussion A forecasting backend reads the observed series under ``series`` and the number of
/// steps to forecast under ``horizon``. It returns an `NSDictionary` under `NFKOutputStructured`
/// holding ``median``, ``quantiles``, and ``quantileLevels``, plus ``mean`` where the model predicts one.
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXForecastKey)
public final class NFKMLXForecastKey: NSObject {
    /// The request input holding the observed series, oldest first (`NSArray<NSNumber *>`). Introduced in
    /// InferKit 0.4.0.
    @objc public static let series = "NFKMLXInputSeries"
    /// The request parameter holding how many steps to forecast past the series (`NSNumber`). Introduced
    /// in InferKit 0.4.0.
    @objc public static let horizon = "NFKMLXParameterHorizon"
    /// The `NFKOutputStructured` field holding the median forecast, the point forecast
    /// (`NSArray<NSNumber *>`, one value per step). Introduced in InferKit 0.4.0.
    @objc public static let median = "median"
    /// The `NFKOutputStructured` field holding the mean forecast (`NSArray<NSNumber *>`, one value per
    /// step), present where the model predicts one. Introduced in InferKit 0.4.0.
    @objc public static let mean = "mean"
    /// The `NFKOutputStructured` field holding one row per quantile level, each one value per step
    /// (`NSArray<NSArray<NSNumber *> *>`). Introduced in InferKit 0.4.0.
    @objc public static let quantiles = "quantiles"
    /// The `NFKOutputStructured` field holding the level of each ``quantiles`` row (`NSArray<NSNumber *>`).
    /// Introduced in InferKit 0.4.0.
    @objc public static let quantileLevels = "quantileLevels"
}

extension NFKMLXGenerationBackend {

    /// The series under ``NFKMLXForecastKey/series``; a missing or empty one is an unsupported input.
    static func forecastSeries(_ request: NFKInferenceRequest) throws -> [NSNumber] {
        guard let series = request.input(forKey: NFKMLXForecastKey.series) as? [NSNumber], !series.isEmpty else {
            throw NFKMLXError.unsupportedInput
        }
        return series
    }

    /// The horizon under ``NFKMLXForecastKey/horizon``, `fallback` where none is given.
    static func forecastHorizon(_ request: NFKInferenceRequest, fallback: Int) -> Int {
        number(request, NFKMLXForecastKey.horizon)?.intValue ?? fallback
    }

    /// A result carrying the forecast fields under `NFKOutputStructured`.
    static func forecastResult(median: [NSNumber], quantiles: [[NSNumber]], levels: [NSNumber],
                               mean: [NSNumber]? = nil) -> NFKInferenceResult {
        var fields: [String: Any] = [
            NFKMLXForecastKey.median: median,
            NFKMLXForecastKey.quantiles: quantiles,
            NFKMLXForecastKey.quantileLevels: levels,
        ]
        if let mean {
            fields[NFKMLXForecastKey.mean] = mean
        }
        return NFKInferenceResult(outputs: [NFKOutputStructured: fields])
    }
}
