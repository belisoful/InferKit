//
//  NFKMLXModelFit.swift
//  InferKitMLX
//
//  Whether a model fits before it is loaded, and how fast it can possibly decode.
//
//  Loading a model that does not fit does not fail politely: the process is killed, or the system
//  pages until the run is useless. Both are decidable in advance from arithmetic — the weights are a
//  function of the geometry, the cache is a function of the geometry and the length — so the answer
//  costs nothing next to the load it prevents.
//
//  Introduced in InferKit 0.1.0.
//

import Foundation
import InferKit
import MLX

/// What a model's weights and cache would cost, against what the machine has.
public struct NFKMLXModelFit: Sendable, Equatable {

    /// Whether the model fits, and what it would take.
    public enum Verdict: Sendable, Equatable {
        /// Weights and the requested cache both fit inside the budget.
        case fits
        /// The weights fit but the requested cache does not; this many positions do.
        case fitsWithinWindow(Int)
        /// The weights alone exceed the budget, by this many bytes.
        case tooLarge(shortfall: Int)
    }

    /// Bytes the parameters occupy at the given precision.
    public let weightBytes: Int

    /// Bytes the key-value cache adds per position, across every layer.
    public let keyValueBytesPerToken: Int

    /// Positions the caller asked to plan for.
    public let requestedTokens: Int

    /// The bytes this decision was made against.
    public let budget: Int

    public let verdict: Verdict

    /// Whether the model can run at all, at some context length.
    public var fits: Bool {
        if case .tooLarge = verdict { return false }
        return true
    }

    /// The window to bound the cache to, or `nil` when the requested length already fits.
    ///
    /// @discussion This is the number `NFKMLXGenerationOptions.contextWindow` wants. Deriving it is
    /// the point: a hand-picked window is a guess about a machine the author was not using.
    public var recommendedContextWindow: Int? {
        if case .fitsWithinWindow(let window) = verdict { return window }
        return nil
    }

    /// A one-line summary, for a log.
    public var describedFit: String {
        func gigabytes(_ bytes: Int) -> String {
            String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
        }
        let cache = keyValueBytesPerToken * requestedTokens
        let head = "\(gigabytes(weightBytes)) weights + \(gigabytes(cache)) cache "
            + "at \(requestedTokens) tokens, against \(gigabytes(budget))"
        switch verdict {
        case .fits:
            return head + " — fits"
        case .fitsWithinWindow(let window):
            return head + " — fits with the cache bounded to \(window) positions"
        case .tooLarge(let shortfall):
            return head + " — too large by \(gigabytes(shortfall))"
        }
    }
}

/// Sizing a model against the machine, and the ceiling its decode speed is bounded by.
public enum NFKMLXModelSizing {

    // MARK: The budget

    /// The bytes a model may reasonably occupy right now.
    ///
    /// @discussion The smaller of what Metal recommends keeping resident and what is actually free,
    /// less a margin for everything the run itself allocates — activations, the sampler, the caller's
    /// own work. Sizing against physical memory instead is sizing against a number that was never
    /// available.
    ///
    /// - Parameter margin: the share held back, in `0...1`.
    public static func availableBudget(margin: Double = 0.15) -> Int {
        let profile = NFKHardwareProfile.current
        let recommended = profile.recommendedWorkingSetSize
        let free = NFKHardwareProfile.availableMemory()

        var ceiling = recommended > 0 ? recommended : profile.physicalMemory
        if free > 0 { ceiling = Swift.min(ceiling, free) }
        let kept = 1 - Swift.min(Swift.max(margin, 0), 1)
        return Swift.max(Int(Double(ceiling) * kept), 0)
    }

    // MARK: What a model costs

    /// Bytes one parameter occupies at a given precision.
    public static func bytesPerParameter(_ precision: NFKMLXWeightPrecision) -> Int {
        switch precision {
        case .float32: return 4
        // A checkpoint's own precision is whatever it shipped in, and every released decoder here
        // ships 16-bit. A float32 release read this way would be under-counted, which errs toward
        // attempting a load rather than refusing one that would have worked.
        case .checkpoint: return 2
        }
    }

    /// Every parameter the dense decoder declares, counted from the geometry alone.
    ///
    /// @discussion Counted rather than built: the point is to answer before allocating anything, and
    /// a 27B model cannot be instantiated to be measured. `NFKMLXModelFitTests` checks the count
    /// against a module that IS built, so the arithmetic is verified rather than trusted.
    public static func parameterCount(of c: NFKMLXLanguageConfiguration) -> Int {
        let attentionWidth = c.headCount * c.headDimensions
        let keyValueWidth = c.keyValueHeadCount * c.headDimensions

        var perLayer = c.hiddenSize * attentionWidth            // q_proj
            + 2 * c.hiddenSize * keyValueWidth                  // k_proj, v_proj
            + attentionWidth * c.hiddenSize                     // o_proj
            + 3 * c.hiddenSize * c.intermediateSize             // gate, up, down
            + 2 * c.hiddenSize                                  // the two layer norms
        if c.attentionBias {
            perLayer += attentionWidth + 2 * keyValueWidth
        }
        if c.normalizesQueryAndKey {
            perLayer += 2 * c.headDimensions                    // q_norm, k_norm
        }

        var total = c.vocabularySize * c.hiddenSize             // embed_tokens
            + c.layerCount * perLayer
            + c.hiddenSize                                      // the final norm
        if !c.tiesWordEmbeddings {
            total += c.vocabularySize * c.hiddenSize            // lm_head
        }
        return total
    }

    /// Bytes the key-value cache adds per position.
    ///
    /// @discussion Keys and values, every layer, at the key-value head count rather than the query
    /// head count — grouped-query attention is what makes this affordable, and counting it at the
    /// query width would overstate the cache several times over.
    public static func keyValueBytesPerToken(of c: NFKMLXLanguageConfiguration,
                                             precision: NFKMLXWeightPrecision = .float32) -> Int {
        2 * c.layerCount * c.keyValueHeadCount * c.headDimensions * bytesPerParameter(precision)
    }

    // MARK: The verdict

    /// Whether a model fits, and the window it would need.
    ///
    /// - Parameters:
    ///   - configuration: the model's geometry.
    ///   - tokens: the context length to plan for.
    ///   - precision: what the weights and cache are held at.
    ///   - budget: the bytes to fit inside. Defaults to ``availableBudget(margin:)``.
    public static func fit(of configuration: NFKMLXLanguageConfiguration,
                           tokens: Int = 4096,
                           precision: NFKMLXWeightPrecision = .float32,
                           budget: Int? = nil) -> NFKMLXModelFit {
        let ceiling = budget ?? availableBudget()
        let weights = parameterCount(of: configuration) * bytesPerParameter(precision)
        let perToken = keyValueBytesPerToken(of: configuration, precision: precision)
        let requested = Swift.max(tokens, 1)

        let verdict: NFKMLXModelFit.Verdict
        if weights >= ceiling {
            verdict = .tooLarge(shortfall: weights - ceiling)
        } else if weights + perToken * requested <= ceiling {
            verdict = .fits
        } else {
            // Whatever is left after the weights, in positions. A model whose weights only just fit
            // leaves room for none, which is a real answer and not an error.
            let window = perToken > 0 ? (ceiling - weights) / perToken : 0
            verdict = .fitsWithinWindow(Swift.max(window, 0))
        }

        return NFKMLXModelFit(weightBytes: weights, keyValueBytesPerToken: perToken,
                              requestedTokens: requested, budget: ceiling, verdict: verdict)
    }

    /// Generation options sized to this machine: the cache bounded only if it has to be.
    ///
    /// @discussion This is the whole point of the arithmetic above. A hand-picked `contextWindow` is a
    /// guess about a machine the author was not using; this derives it from what is actually free.
    /// When the requested length already fits, the window is left unset and nothing is dropped.
    ///
    /// - Parameters:
    ///   - configuration: the model's geometry.
    ///   - tokens: the context length to plan for.
    ///   - precision: what the weights and cache will be held at.
    ///   - base: options to start from, so the caller's other settings survive.
    ///   - budget: the bytes to fit inside. Defaults to ``availableBudget(margin:)``.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` when the weights alone do not fit, naming the
    ///   shortfall. No window helps in that case, and returning options that cannot work would move
    ///   the failure to the load, where it is a process kill rather than an error.
    public static func options(for configuration: NFKMLXLanguageConfiguration,
                               requesting tokens: Int = 4096,
                               precision: NFKMLXWeightPrecision = .float32,
                               base: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                               budget: Int? = nil) throws -> NFKMLXGenerationOptions {
        let sizing = fit(of: configuration, tokens: tokens, precision: precision, budget: budget)
        guard case .tooLarge(let shortfall) = sizing.verdict else {
            var options = base
            options.contextWindow = sizing.recommendedContextWindow
            return options
        }
        throw NFKMLXError.unsupportedConfiguration(
            "the model does not fit: \(sizing.describedFit). Short by \(shortfall) bytes — no context "
            + "window helps, because the weights alone exceed the budget. Load at a lower precision, "
            + "or use a smaller model.")
    }

    // MARK: The decode ceiling

    /// The memory bandwidth this machine achieves, in bytes per second.
    ///
    /// @discussion MEASURED, not tabulated. No sysctl reports memory bandwidth, and a table of
    /// per-chip figures would be a set of numbers copied from somewhere rather than a property of the
    /// machine the code is running on. This reads a large array with a reduction and times it, which
    /// is what a decode step does to the weights, so it measures the rate that actually bounds
    /// decoding rather than a headline figure.
    ///
    /// The first pass is discarded — it pays for allocation and for the kernel's first compile — and
    /// the best of the rest is taken, since interference from other processes can only slow it down.
    ///
    /// **The array has to be big enough or the answer is wrong**, and wrong in the flattering
    /// direction for a probe that runs fast. Swept on an M1 Max, whose specified bandwidth is
    /// 400 GB/s: 16 MB reads 40 GB/s, 64 MB reads 126, 256 MB reads 158, 512 MB reads 274, and it
    /// settles near 330 from 1 GB up. Below half a gigabyte the launch overhead and the caches are
    /// most of what is being timed. `megabytes` therefore defaults to a share of the machine's own
    /// budget rather than a constant, so a small device measures something it can afford and a large
    /// one measures something true.
    ///
    /// The result is cached for the process. It is not written to disk: where that cache should live
    /// is the consumer's decision, and a stale figure copied from another machine is worse than
    /// measuring again.
    ///
    /// - Parameters:
    ///   - megabytes: the array to read. `nil` derives it from the working-set budget.
    ///   - repetitions: timed passes. The first is discarded, and the best of the rest is taken.
    public static func measuredMemoryBandwidth(megabytes: Int? = nil,
                                               repetitions: Int = 4) -> Double {
        if let cached = bandwidthCache.value { return cached }

        let size = megabytes ?? probeMegabytes()
        let count = Swift.max(size, 1) * 1_048_576 / MemoryLayout<Float>.size
        let scratch = MLXArray.zeros([count], dtype: .float32)
        eval(scratch)
        let bytes = Double(count * MemoryLayout<Float>.size)

        var best: Double = 0
        for pass in 0 ..< Swift.max(repetitions, 2) {
            let started = DispatchTime.now().uptimeNanoseconds
            let total = scratch.sum()
            eval(total)
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
            if pass > 0, elapsed > 0 {
                best = Swift.max(best, bytes / elapsed)
            }
        }
        bandwidthCache.value = best
        return best
    }

    /// The most tokens a second the machine could produce for this model, ignoring compute entirely.
    ///
    /// @discussion Decoding one token reads every weight once, and reads the cache for every position
    /// attended to. At the sizes a language model runs at that traffic is what decides the rate, which
    /// is why a ceiling computed from bandwidth alone is close enough to be useful and is reached by
    /// nothing.
    ///
    /// - Parameters:
    ///   - configuration: the model's geometry.
    ///   - contextLength: the positions attended to, whose cache traffic joins the weight traffic.
    ///   - precision: what the weights and cache are held at.
    ///   - bandwidth: bytes per second. Defaults to ``measuredMemoryBandwidth(megabytes:repetitions:)``.
    ///
    /// - Returns: tokens per second, or 0 when the bandwidth is unknown.
    public static func decodeCeiling(for configuration: NFKMLXLanguageConfiguration,
                                     contextLength: Int = 1024,
                                     precision: NFKMLXWeightPrecision = .float32,
                                     bandwidth: Double? = nil) -> Double {
        let rate = bandwidth ?? measuredMemoryBandwidth()
        guard rate > 0 else { return 0 }
        let perToken = Double(parameterCount(of: configuration) * bytesPerParameter(precision)
                              + keyValueBytesPerToken(of: configuration, precision: precision)
                              * Swift.max(contextLength, 0))
        guard perToken > 0 else { return 0 }
        return rate / perToken
    }

    /// The share of the ceiling a measured rate reached, in `0...1`.
    ///
    /// @discussion Inverting the ceiling is what turns it into a diagnostic. A dense model decoding
    /// well below its ceiling is leaving bandwidth unused; a model decoding ABOVE the ceiling
    /// computed for its full parameter count is not reading all of them, which is what a sparse model
    /// doing its job looks like — and what a sparse model failing to would not.
    public static func achievedFraction(tokensPerSecond: Double,
                                        for configuration: NFKMLXLanguageConfiguration,
                                        contextLength: Int = 1024,
                                        precision: NFKMLXWeightPrecision = .float32,
                                        bandwidth: Double? = nil) -> Double {
        let ceiling = decodeCeiling(for: configuration, contextLength: contextLength,
                                    precision: precision, bandwidth: bandwidth)
        guard ceiling > 0 else { return 0 }
        return tokensPerSecond / ceiling
    }

    /// Discards the cached bandwidth, so the next call measures again.
    public static func resetMeasuredBandwidth() { bandwidthCache.value = nil }

    /// A probe large enough to saturate without being a burden: a sixteenth of the working-set
    /// budget, held between 64 MB and 1 GB.
    private static func probeMegabytes() -> Int {
        let budget = NFKHardwareProfile.current.recommendedWorkingSetSize
        guard budget > 0 else { return 256 }
        return Swift.min(Swift.max(budget / 16 / 1_048_576, 64), 1024)
    }

    private static let bandwidthCache = NFKBandwidthCache()
}

/// A lock-guarded box, because the measurement is cached across whatever thread asks for it.
private final class NFKBandwidthCache: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Double?

    var value: Double? {
        get { lock.lock(); defer { lock.unlock() }; return stored }
        set { lock.lock(); stored = newValue; lock.unlock() }
    }
}
