//
//  NFKMLXChronosBackend.swift
//  InferKitMLX
//
//  Chronos-Bolt behind the InferKit contract.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXChronos {

    /// A name for the model the factories produce. Introduced in InferKit 0.4.0.
    @objc public static let modelName = "chronos-bolt-base"

    /// The forecaster behind the InferKit contract.
    ///
    /// @discussion The backend reads the series under ``NFKMLXForecastKey/series`` (required) and the
    /// horizon under ``NFKMLXForecastKey/horizon``, capped at the model's prediction length (64 for the
    /// release), which is also the horizon where none is given. It returns the ``NFKMLXForecastKey``
    /// fields under `NFKOutputStructured`: the median, the nine quantile rows, and their levels.
    /// Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.modelName, inputKeys: [NFKMLXForecastKey.series],
                                parameterKeys: [NFKMLXForecastKey.horizon],
                                outputKeys: [NFKOutputStructured]) { [self] request, job in
            let series = try NFKMLXGenerationBackend.forecastSeries(request).map(\.floatValue)
            let horizon = NFKMLXGenerationBackend.forecastHorizon(request, fallback: Int.max)
            job.reportProgress(0)
            let rows = forecast(context: series, horizon: horizon).map { $0.map { NSNumber(value: $0) } }
            if job.status == .cancelled { return nil }
            job.reportProgress(1)
            return NFKMLXGenerationBackend.forecastResult(median: rows[rows.count / 2], quantiles: rows,
                                                          levels: quantileLevels)
        }
    }

    /// The asynchronous form of ``chronos(repo:weightsPath:revision:cacheDirectoryURL:)``. The download and
    /// the build run at user-initiated quality of service off the calling thread. Introduced in InferKit
    /// 0.4.0.
    @objc(chronosWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func chronos(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXChronos?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try chronos(repo: repo, weightsPath: weightsPath, revision: revision,
                        cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Builds the forecaster, as ``chronos(weightsURL:)`` does, and returns it behind the InferKit contract
    /// (``inferenceBackend()``). A nil `weightsURL` builds random weights. Introduced in InferKit 0.4.0.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try chronos(weightsURL: weightsURL).inferenceBackend()
    }

    /// Downloads the checkpoint, as ``chronos(repo:weightsPath:revision:cacheDirectoryURL:)`` does, and
    /// returns the forecaster behind the InferKit contract.
    ///
    /// @discussion The release is `amazon/chronos-bolt-base`, weights path `model.safetensors`. A file
    /// already in the cache is not fetched again. The call blocks on the network, so run it off the main
    /// and render threads. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try chronos(repo: repo, weightsPath: weightsPath, revision: revision,
                    cacheDirectoryURL: cacheDirectoryURL).inferenceBackend()
    }

    /// The asynchronous form of ``backend(repo:weightsPath:revision:cacheDirectoryURL:)``. The handler runs
    /// on a background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) }, completionHandler: completionHandler)
    }

    /// Registers the backend under ``modelName``. The registry's URL is the weights file; nil builds random
    /// weights. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }
}
