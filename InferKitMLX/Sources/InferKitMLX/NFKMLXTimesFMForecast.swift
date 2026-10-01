//
//  NFKMLXTimesFMForecast.swift
//  InferKitMLX
//
//  TimesFM 2.5 forecasting: the official module's `decode` (running RevIN statistics per patch, a
//  prefill, and autoregressive 128-step decoding over a key-value cache) and the forecasting flags of
//  `TimesFM_2p5_200M_torch.compile` (global normalization, flip invariance, the continuous quantile head,
//  quantile-crossing repair, and nonnegativity), with the @objc forecaster and its factories.
//

import Foundation
import MLX
import MLXNN

/// The forecasting flags of the official `ForecastConfig`. The defaults are the release card's
/// recommended settings.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXTimesFMForecastOptions)
public final class NFKMLXTimesFMForecastOptions: NSObject {
    /// The context the forecast reads (`max_context`), rounded up to a multiple of the 32-step patch. A
    /// longer series keeps its most recent steps; a shorter one is zero-padded in front, which the global
    /// normalization's statistics count.
    @objc public var maximumContext: Int = 1024
    /// Normalizes the context by its mean and sample standard deviation before the model and restores the
    /// forecast after (`normalize_inputs`).
    @objc public var normalizesInputs = true
    /// Takes every quantile but the median from the 1024-step continuous quantile head, offset to the point
    /// forecast's median (`use_continuous_quantile_head`). Limits the horizon to 1024.
    @objc public var usesContinuousQuantileHead = true
    /// Averages the forecast of the series with the negated forecast of its negation
    /// (`force_flip_invariance`).
    @objc public var forcesFlipInvariance = true
    /// Clamps the forecast at zero when every context value is nonnegative (`infer_is_positive`).
    @objc public var infersPositivity = true
    /// Makes the quantiles monotonic outward from the median (`fix_quantile_crossing`).
    @objc public var fixesQuantileCrossing = true

    @objc public override init() { super.init() }
}

/// One forecast: at every step the mean and the nine quantiles, with the median as the point forecast.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXTimesFMForecast)
public final class NFKMLXTimesFMForecast: NSObject {
    /// `[horizon][10]`: the mean, then the 0.1 … 0.9 quantiles.
    public let values: [[Float]]
    public let quantileLevels: [Float]

    init(values: [[Float]], quantileLevels: [Float]) {
        self.values = values
        self.quantileLevels = quantileLevels
        super.init()
    }

    /// The median at every step, the release's point forecast.
    @objc public var pointForecast: [NSNumber] { values.map { NSNumber(value: $0[5]) } }
    /// The mean head at every step.
    @objc public var meanForecast: [NSNumber] { values.map { NSNumber(value: $0[0]) } }
    /// The levels `quantileForecasts` rows correspond to (0.1 … 0.9).
    @objc public var levels: [NSNumber] { quantileLevels.map { NSNumber(value: $0) } }
    /// One row per quantile level, each `horizon` long.
    @objc public var quantileForecasts: [[NSNumber]] {
        (1 ... quantileLevels.count).map { level in values.map { NSNumber(value: $0[level]) } }
    }
}

/// TimesFM 2.5 time-series forecasting (`google/timesfm-2.5-200m-pytorch`, Apache-2.0). An object like
/// ``NFKMLXChronos``: a numeric series has no core request key.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXTimesFM)
public final class NFKMLXTimesFM: NSObject {
    final class Holder: @unchecked Sendable {
        let net: NFKMLXTimesFMNet
        init(_ net: NFKMLXTimesFMNet) { self.net = net }
    }

    private let holder: Holder
    public var net: NFKMLXTimesFMNet { holder.net }

    public init(net: NFKMLXTimesFMNet) {
        holder = Holder(net)
        super.init()
    }

    // MARK: Factories

    static let requiredFiles = ["config.json"]
    static let weightFiles = ["model.safetensors"]

    /// Builds a forecaster from a release directory (`config.json` and `model.safetensors`, the official
    /// layout or the transformers one).
    @objc(timesFMWithDirectoryURL:error:)
    public static func timesFM(directoryURL: URL) throws -> NFKMLXTimesFM {
        let net = NFKMLXTimesFMNet(try NFKMLXTimesFMConfiguration(configurationURL: directoryURL.appendingPathComponent("config.json")))
        try net.loadWeights(fromDirectory: directoryURL)
        return NFKMLXTimesFM(net: net)
    }

    /// The asynchronous form of the directory factory, at user-initiated quality of service.
    @objc(timesFMWithDirectoryURL:completionHandler:)
    public static func timesFM(directoryURL: URL, completionHandler: @escaping (NFKMLXTimesFM?, Error?) -> Void) {
        Task.detached(priority: .userInitiated) {
            do { completionHandler(try timesFM(directoryURL: directoryURL), nil) }
            catch { completionHandler(nil, error) }
        }
    }

    /// Downloads a release into the hub cache and builds the forecaster.
    ///
    /// @discussion The download fetches `config.json` and `model.safetensors`. The releases are
    /// `google/timesfm-2.5-200m-pytorch` (the official checkpoint) and `google/timesfm-2.5-200m-transformers`
    /// (the same weights in transformers' names); neither is gated. TimesFM 3.0 carries a non-commercial
    /// license and a different architecture, and is not read. The call blocks on the network; call it off
    /// the render thread.
    @objc(timesFMWithRepo:revision:cacheDirectoryURL:error:)
    public static func timesFM(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXTimesFM {
        try timesFM(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: [], weights: weightFiles))
    }

    /// The asynchronous form of ``timesFM(repo:revision:cacheDirectoryURL:)``.
    @objc(timesFMWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func timesFM(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXTimesFM?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do { completionHandler(try timesFM(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil) }
            catch { completionHandler(nil, error) }
        }
    }

    // MARK: Forecasting

    /// Forecasts `horizon` steps past `context`. A NaN value is filled by linear interpolation, and
    /// leading NaNs are dropped, as the reference's `forecast` does. Blocking; call off the render thread.
    public func forecast(context: [Float], horizon: Int,
                         options: NFKMLXTimesFMForecastOptions = NFKMLXTimesFMForecastOptions()) throws -> NFKMLXTimesFMForecast {
        let c = holder.net.configuration
        let series = Self.filled(context)
        guard !series.isEmpty else { throw NFKMLXError.unsupportedInput }
        let maximumContext = (max(options.maximumContext, 1) + c.patchLength - 1) / c.patchLength * c.patchLength
        let maximumHorizon = (horizon + c.outputPatchLength - 1) / c.outputPatchLength * c.outputPatchLength
        guard horizon >= 1, maximumContext + maximumHorizon <= c.contextLimit else {
            throw NFKMLXError.unsupportedConfiguration("context \(maximumContext) + horizon \(maximumHorizon) "
                                                      + "exceeds the \(c.contextLimit)-step limit")
        }
        guard !options.usesContinuousQuantileHead || maximumHorizon <= c.outputQuantileLength else {
            throw NFKMLXError.unsupportedConfiguration("the continuous quantile head reaches \(c.outputQuantileLength) steps")
        }
        let values = Array(series.suffix(maximumContext))
        let paddedLength = max(values.count, maximumContext)
        let isPositive = options.infersPositivity && values.allSatisfy { $0 >= 0 }

        var mean: Float = 0, deviation: Float = 1, normalized = values
        if options.normalizesInputs {
            (mean, deviation) = Self.globalStatistics(values, paddedLength: paddedLength)
            let divisor = deviation < 1e-6 ? 1 : deviation
            normalized = values.map { ($0 - mean) / divisor }
        }

        let steps = (horizon - 1) / c.outputPatchLength
        var (point, spreads, full) = Self.decode(holder.net, normalized, steps: steps)
        if options.forcesFlipInvariance {
            let flipped = Self.decode(holder.net, normalized.map { -$0 }, steps: steps)
            let order = MLXArray([Int32(0)] + (1 ..< c.outputChannels).reversed().map(Int32.init))
            func flip(_ x: MLXArray) -> MLXArray { take(x, order, axis: -1) }
            point = (point - flip(flipped.point)) / 2
            spreads = (spreads - flip(flipped.spreads)) / 2
            full = (full - flip(flipped.full)) / 2
        }
        _ = point
        let length = full.dim(0)
        var rows = full.asType(.float32)
        if options.usesContinuousQuantileHead {
            let median = rows[0..., c.decodeIndex ..< (c.decodeIndex + 1)]
            let spread = spreads[..<length]
            let continuous = spread - spread[0..., c.decodeIndex ..< (c.decodeIndex + 1)] + median
            let keep = MLXArray((0 ..< c.outputChannels).map { $0 == 0 || $0 == c.decodeIndex })
            rows = MLX.where(keep, rows, continuous)
        }
        eval(rows)
        var table = rows[..<horizon].asArray(Float.self)
        let channels = c.outputChannels
        if options.fixesQuantileCrossing {
            for step in 0 ..< horizon {
                let base = step * channels
                for i in stride(from: 4, through: 1, by: -1) where !(table[base + i] < table[base + i + 1]) {
                    table[base + i] = table[base + i + 1]
                }
                for i in 6 ... 9 where !(table[base + i] > table[base + i - 1]) {
                    table[base + i] = table[base + i - 1]
                }
            }
        }
        if options.normalizesInputs {
            table = table.map { $0 * deviation + mean }
        }
        if isPositive {
            table = table.map { max($0, 0) }
        }
        let forecast = (0 ..< horizon).map { Array(table[($0 * channels) ..< (($0 + 1) * channels)]) }
        return NFKMLXTimesFMForecast(values: forecast, quantileLevels: c.quantiles)
    }

    /// The Objective-C form of ``forecast(context:horizon:options:)``; nil options take the defaults.
    @objc(forecastForContext:horizon:options:error:)
    public func forecast(context: [NSNumber], horizon: Int, options: NFKMLXTimesFMForecastOptions?) throws -> NFKMLXTimesFMForecast {
        try forecast(context: context.map(\.floatValue), horizon: horizon, options: options ?? NFKMLXTimesFMForecastOptions())
    }

    /// The median point forecast at the default options, for an Objective-C caller.
    @objc(medianForecastForContext:horizon:error:)
    public func medianForecast(context: [NSNumber], horizon: Int) throws -> [NSNumber] {
        try forecast(context: context, horizon: horizon, options: nil).pointForecast
    }

    // MARK: Decoding

    /// The official module's `decode` on one normalized series: the last patch's 128-step forecast
    /// followed by `steps` autoregressive patches (`full`, `[128 · (steps + 1), 10]`), the last patch's
    /// own forecast (`point`, `[128, 10]`), and its continuous quantile spread (`spreads`, `[1024, 10]`),
    /// each restored to the series' scale by the patch statistics. Leading patches that are padding
    /// throughout are left out: they are excluded as keys and shift no position, so the result is the
    /// reference's.
    static func decode(_ net: NFKMLXTimesFMNet, _ series: [Float], steps: Int) -> (point: MLXArray, spreads: MLXArray, full: MLXArray) {
        let c = net.configuration
        let p = c.patchLength
        let front = (p - series.count % p) % p
        let values = [Float](repeating: 0, count: front) + series
        let padding = [Bool](repeating: true, count: front) + [Bool](repeating: false, count: series.count)
        let patches = values.count / p

        var statistics = NFKTimesFMRunningStatistics()
        var means = [Float](), deviations = [Float]()
        for index in 0 ..< patches {
            statistics.update(Array(values[(index * p) ..< ((index + 1) * p)]),
                              padding: Array(padding[(index * p) ..< ((index + 1) * p)]))
            means.append(statistics.mean)
            deviations.append(statistics.deviation)
        }
        let normed = (0 ..< values.count).map { index -> Float in
            let patch = index / p
            let divisor = deviations[patch] < 1e-6 ? 1 : deviations[patch]
            return padding[index] ? 0 : (values[index] - means[patch]) / divisor
        }
        var caches = [NFKTimesFMLayerCache?](repeating: nil, count: c.numLayers)
        let prefill = net.forward(MLXArray(normed).reshaped([1, patches, p]),
                                         masks: MLXArray(padding).reshaped([1, patches, p]),
                                         positions: MLXArray(0 ..< patches).asType(.float32).reshaped([1, patches]),
                                         caches: &caches)
        let lastMean = means[patches - 1], lastDeviation = deviations[patches - 1]
        let point = (prefill.point[0, patches - 1] * lastDeviation + lastMean).reshaped([c.outputPatchLength, c.outputChannels])
        let spreads = (prefill.quantiles[0, patches - 1] * lastDeviation + lastMean)
            .reshaped([c.outputQuantileLength, c.outputChannels])

        var outputs = [point]
        var last = point[0..., c.decodeIndex]
        let perStep = c.outputPatchLength / p
        var position = patches
        for _ in 0 ..< steps {
            eval(last)
            let continued = last.asArray(Float.self)
            var stepMeans = [Float](), stepDeviations = [Float]()
            var normedStep = [Float]()
            for index in 0 ..< perStep {
                let chunk = Array(continued[(index * p) ..< ((index + 1) * p)])
                statistics.update(chunk, padding: [Bool](repeating: false, count: p))
                stepMeans.append(statistics.mean)
                stepDeviations.append(statistics.deviation)
                let divisor = statistics.deviation < 1e-6 ? 1 : statistics.deviation
                normedStep += chunk.map { ($0 - statistics.mean) / divisor }
            }
            let step = net.forward(MLXArray(normedStep).reshaped([1, perStep, p]),
                                          masks: MLXArray.zeros([1, perStep, p], type: Bool.self),
                                          positions: MLXArray(position ..< (position + perStep)).asType(.float32)
                                              .reshaped([1, perStep]),
                                          caches: &caches)
            position += perStep
            let output = (step.point[0, perStep - 1] * stepDeviations[perStep - 1] + stepMeans[perStep - 1])
                .reshaped([c.outputPatchLength, c.outputChannels])
            outputs.append(output)
            last = output[0..., c.decodeIndex]
        }
        return (point, spreads, concatenated(outputs, axis: 0))
    }

    /// The mean and the sample standard deviation (`torch.std`, `N − 1`) of the context zero-padded in
    /// front to `paddedLength`, as the reference computes them on its padded batch.
    static func globalStatistics(_ values: [Float], paddedLength: Int) -> (mean: Float, deviation: Float) {
        let count = Double(paddedLength)
        let mean = values.reduce(0.0) { $0 + Double($1) } / count
        var squares = values.reduce(0.0) { $0 + (Double($1) - mean) * (Double($1) - mean) }
        squares += Double(paddedLength - values.count) * mean * mean
        let deviation = paddedLength > 1 ? (squares / (count - 1)).squareRoot() : 0
        return (Float(mean), Float(deviation))
    }

    /// Leading NaNs removed and the rest filled by linear interpolation between the nearest valid values,
    /// the nearest one past either end (`strip_leading_nans` then `linear_interpolation`).
    static func filled(_ values: [Float]) -> [Float] {
        guard let first = values.firstIndex(where: { !$0.isNaN }) else { return [] }
        var series = Array(values[first...])
        let valid = series.indices.filter { !series[$0].isNaN }
        guard valid.count < series.count else { return series }
        var cursor = 0
        for index in series.indices where series[index].isNaN {
            while cursor + 1 < valid.count && valid[cursor + 1] < index { cursor += 1 }
            let left = valid[cursor]
            guard let right = valid.first(where: { $0 > index }) else {
                series[index] = series[valid.last!]
                continue
            }
            let fraction = Float(index - left) / Float(right - left)
            series[index] = series[left] + fraction * (series[right] - series[left])
        }
        return series
    }
}

/// `util.update_running_stats`: the count, mean, and population standard deviation of every unpadded
/// value so far, merged patch by patch.
struct NFKTimesFMRunningStatistics {
    var count: Float = 0
    var mean: Float = 0
    var deviation: Float = 0

    mutating func update(_ values: [Float], padding: [Bool]) {
        let legit = zip(values, padding).filter { !$0.1 }.map(\.0)
        let increment = Float(legit.count)
        let incrementMean = increment == 0 ? 0 : legit.reduce(0, +) / increment
        let incrementVariance = increment == 0 ? 0 : legit.reduce(Float(0)) { $0 + ($1 - incrementMean) * ($1 - incrementMean) } / increment
        let total = count + increment
        let safe = total == 0 ? 1 : total
        let newMean = total == 0 ? 0 : (count * mean + incrementMean * increment) / safe
        let variance = total == 0 ? 0 : (count * deviation * deviation + increment * incrementVariance
                                         + count * (mean - newMean) * (mean - newMean)
                                         + increment * (incrementMean - newMean) * (incrementMean - newMean)) / safe
        count = total
        mean = newMean
        deviation = max(variance, 0).squareRoot()
    }
}
