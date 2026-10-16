//
//  NFKMLXTimesFMBackend.swift
//  InferKitMLX
//
//  TimesFM 2.5 behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The request parameters that set ``NFKMLXTimesFMForecastOptions`` for one request. A parameter a
/// request leaves out keeps the option's default.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXTimesFMParameterKey)
public final class NFKMLXTimesFMParameterKey: NSObject {
    /// ``NFKMLXTimesFMForecastOptions/maximumContext`` (`NSNumber` integer). Introduced in InferKit 0.4.0.
    @objc public static let maximumContext = "NFKMLXParameterTimesFMMaximumContext"
    /// ``NFKMLXTimesFMForecastOptions/normalizesInputs`` (`NSNumber` boolean). Introduced in InferKit 0.4.0.
    @objc public static let normalizesInputs = "NFKMLXParameterTimesFMNormalizesInputs"
    /// ``NFKMLXTimesFMForecastOptions/usesContinuousQuantileHead`` (`NSNumber` boolean). Introduced in
    /// InferKit 0.4.0.
    @objc public static let usesContinuousQuantileHead = "NFKMLXParameterTimesFMUsesContinuousQuantileHead"
    /// ``NFKMLXTimesFMForecastOptions/forcesFlipInvariance`` (`NSNumber` boolean). Introduced in InferKit
    /// 0.4.0.
    @objc public static let forcesFlipInvariance = "NFKMLXParameterTimesFMForcesFlipInvariance"
    /// ``NFKMLXTimesFMForecastOptions/infersPositivity`` (`NSNumber` boolean). Introduced in InferKit 0.4.0.
    @objc public static let infersPositivity = "NFKMLXParameterTimesFMInfersPositivity"
    /// ``NFKMLXTimesFMForecastOptions/fixesQuantileCrossing`` (`NSNumber` boolean). Introduced in InferKit
    /// 0.4.0.
    @objc public static let fixesQuantileCrossing = "NFKMLXParameterTimesFMFixesQuantileCrossing"

    /// Every key, which is what the backend lists among its parameters.
    static let all: Set<String> = [maximumContext, normalizesInputs, usesContinuousQuantileHead,
                                   forcesFlipInvariance, infersPositivity, fixesQuantileCrossing]

    /// The options `request` sets, the defaults where it sets none.
    static func options(_ request: NFKInferenceRequest) -> NFKMLXTimesFMForecastOptions {
        let options = NFKMLXTimesFMForecastOptions()
        func flag(_ key: String) -> Bool? { NFKMLXGenerationBackend.number(request, key)?.boolValue }
        options.maximumContext = NFKMLXGenerationBackend.number(request, maximumContext)?.intValue ?? options.maximumContext
        options.normalizesInputs = flag(normalizesInputs) ?? options.normalizesInputs
        options.usesContinuousQuantileHead = flag(usesContinuousQuantileHead) ?? options.usesContinuousQuantileHead
        options.forcesFlipInvariance = flag(forcesFlipInvariance) ?? options.forcesFlipInvariance
        options.infersPositivity = flag(infersPositivity) ?? options.infersPositivity
        options.fixesQuantileCrossing = flag(fixesQuantileCrossing) ?? options.fixesQuantileCrossing
        return options
    }
}

extension NFKMLXTimesFM {

    /// A name for the model the factories produce. Introduced in InferKit 0.4.0.
    @objc public static let modelName = "timesfm-2.5-200m"

    /// The forecaster behind the InferKit contract.
    ///
    /// @discussion The backend reads the series under ``NFKMLXForecastKey/series`` (required) and the
    /// horizon under ``NFKMLXForecastKey/horizon`` (one 128-step output patch for the release where none
    /// is given). The ``NFKMLXTimesFMParameterKey`` parameters set the forecast options for that request
    /// only. It returns the ``NFKMLXForecastKey`` fields under `NFKOutputStructured`: the median, the
    /// mean, the nine quantile rows, and their levels. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.modelName, inputKeys: [NFKMLXForecastKey.series],
                                parameterKeys: NFKMLXTimesFMParameterKey.all.union([NFKMLXForecastKey.horizon]),
                                outputKeys: [NFKOutputStructured]) {
            [self] request, job in
            let series = try NFKMLXGenerationBackend.forecastSeries(request)
            let horizon = NFKMLXGenerationBackend.forecastHorizon(request, fallback: net.configuration.outputPatchLength)
            job.reportProgress(0)
            let result = try forecast(context: series, horizon: horizon,
                                      options: NFKMLXTimesFMParameterKey.options(request))
            if job.status == .cancelled { return nil }
            job.reportProgress(1)
            return NFKMLXGenerationBackend.forecastResult(median: result.pointForecast,
                                                          quantiles: result.quantileForecasts,
                                                          levels: result.levels, mean: result.meanForecast)
        }
    }

    /// Builds the forecaster from a release directory, as ``timesFM(directoryURL:)`` does, and returns it
    /// behind the InferKit contract (``inferenceBackend()``). Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try timesFM(directoryURL: directoryURL).inferenceBackend()
    }

    /// The asynchronous form of ``backend(directoryURL:)``. The handler runs on a background queue.
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:completionHandler:)
    public static func backend(directoryURL: URL,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) { try backend(directoryURL: directoryURL) }
    }

    /// Downloads a release, as ``timesFM(repo:revision:cacheDirectoryURL:)`` does, and returns the
    /// forecaster behind the InferKit contract.
    ///
    /// @discussion A file already in the cache is not fetched again. The call blocks on the network, so
    /// run it off the main and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try timesFM(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL).inferenceBackend()
        }
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``. The handler runs on a
    /// background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers the backend under ``modelName``. The registry's URL is the release directory; nil is
    /// refused. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(modelName) { try backend(directoryURL: $0) }
    }
}
