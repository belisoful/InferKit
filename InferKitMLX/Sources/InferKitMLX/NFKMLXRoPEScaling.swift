//
//  NFKMLXRoPEScaling.swift
//  InferKitMLX
//
//  Rotary frequency scaling: how a release states that its context window was extended past what it
//  was trained on.
//
//  A rotary embedding turns each channel pair at its own frequency, and the highest frequencies
//  complete a full rotation within a few hundred positions. Run such a model past its training length
//  and those channels see angles they never saw, which is why an unscaled model degrades sharply
//  rather than gradually. Scaling changes the frequencies so a longer sequence maps onto the range the
//  model was trained over.
//
//  This is read from a checkpoint's own `rope_scaling`, never chosen here. A model that declares no
//  scaling gets none; a model that declares a kind this does not implement is REJECTED rather than
//  approximated, because a silently wrong rotary produces fluent nonsense rather than an error.
//
//  Introduced in InferKit 0.1.0.
//

import Foundation

/// The rotary scaling a release declares, and the inverse frequencies it implies.
///
/// @discussion Four kinds are implemented, and they work differently. `linear` divides every frequency
/// by the same factor, which is position interpolation: the model sees a longer sequence squeezed into
/// its trained range, at the cost of resolution everywhere.
///
/// `longrope` (Phi-3, Phi-4) carries two explicit per-channel tables instead of a formula: one used
/// while the sequence stays within the trained window, one used past it, chosen by length rather than
/// blended. It also multiplies the rotated queries and keys by a scalar derived from how far the
/// extended window reaches, applied at every length.
///
/// `yarn` divides only the SLOW channels and leaves the fast ones as they were, blending across a band
/// between. The direction is worth stating plainly, because the intuitive guess is the opposite one: a
/// fast channel completes many turns inside the trained window, so it encodes LOCAL offset, and that
/// meaning is unchanged by a longer sequence — interpolating it would only blur short-range position.
/// A slow channel does not complete a turn even at the trained length, so it encodes position across
/// the whole window; run past that length it reaches angles the model has never seen, and it is the
/// one that has to be squeezed.
///
/// `llama3` is the same idea with hard edges instead of a ramp: a channel whose wavelength exceeds the
/// trained window is divided by `factor`, a channel turning at least `high_freq_factor` times inside
/// it is left alone, and the band between is blended linearly in turn count. It carries no attention
/// factor. Chatterbox's T3 decoder declares it, which is what brought it here.
///
/// Both are measured against `transformers`' own `ROPE_INIT_FUNCTIONS`, which is the dispatch every
/// released decoder's config is read by. See `NFKMLXRoPEScalingTests`.
public struct NFKMLXRoPEScaling: Sendable, Equatable {

    /// The scaling kinds this implements.
    public enum Kind: String, Sendable, Equatable, CaseIterable {
        /// Position interpolation: every frequency divided by `factor`.
        case linear
        /// The wavelength-dependent blend of interpolation and extrapolation.
        case yarn
        /// Llama 3.1's three-band rule: channels whose wavelength exceeds the trained window are
        /// interpolated by `factor`, channels turning at least `highFrequencyFactor` times inside it
        /// are left alone, and the band between is blended linearly in wavelength.
        case llama3
        /// Phi-3/Phi-4's per-channel table scaling: two explicit lists of per-pair multipliers, one
        /// for sequences within the trained window (``shortFactor``) and one for longer ones
        /// (``longFactor``), chosen by the sequence length rather than blended. Each factor divides its
        /// pair's frequency. A single scalar multiplies the rotated queries and keys, derived from how
        /// far the extended window reaches past the trained one.
        case longrope
    }

    public var kind: Kind

    /// How far the window is extended, as a multiple of the trained one.
    public var factor: Float

    /// The window the model was trained over, which is what decides which channels have wrapped.
    /// Unused by `linear`.
    public var originalMaxPositionEmbeddings: Int

    /// The rotation count marking the end of the extrapolation band. The paper's default is 32.
    public var betaFast: Float
    /// Whether YaRN's correction band snaps to whole channels (`truncate`, transformers' default).
    public var truncatesCorrectionRange: Bool = true

    /// The rotation count marking the start of the interpolation band. The paper's default is 1.
    public var betaSlow: Float

    /// The config's own `attention_factor`, when it states one. `nil` derives it.
    public var declaredAttentionFactor: Float?
    /// `llama3`'s `low_freq_factor`: a channel completing fewer than this many turns inside the trained
    /// window is interpolated by the full factor. The release default is 1.
    public var lowFrequencyFactor: Float
    /// `llama3`'s `high_freq_factor`: a channel completing at least this many turns is left unscaled.
    /// The release default is 4.
    public var highFrequencyFactor: Float

    /// `longrope`'s per-pair multipliers for a sequence within the trained window. Each divides its
    /// pair's frequency; a table of all ones is the unscaled rotary. Empty for other kinds.
    public var shortFactor: [Float] = []
    /// `longrope`'s per-pair multipliers for a sequence past the trained window. Empty for other kinds.
    public var longFactor: [Float] = []
    /// `longrope`'s extended window, `max_position_embeddings`. The attention scalar is derived from
    /// its ratio to ``originalMaxPositionEmbeddings``.
    public var maximumPositionEmbeddings: Int = 0

    public init(kind: Kind,
                factor: Float,
                originalMaxPositionEmbeddings: Int = 0,
                betaFast: Float = 32,
                betaSlow: Float = 1,
                declaredAttentionFactor: Float? = nil,
                lowFrequencyFactor: Float = 1,
                highFrequencyFactor: Float = 4,
                truncatesCorrectionRange: Bool = true) {
        self.kind = kind
        self.factor = factor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.truncatesCorrectionRange = truncatesCorrectionRange
        self.declaredAttentionFactor = declaredAttentionFactor
        self.lowFrequencyFactor = lowFrequencyFactor
        self.highFrequencyFactor = highFrequencyFactor
    }

    /// The factor the rotated queries and keys are multiplied by.
    ///
    /// @discussion Interpolating the frequencies lowers the average attention logit, and YaRN
    /// compensates with a scalar on the rotary embedding. It multiplies the queries and the keys
    /// alike, so an attention score carries its square.
    ///
    /// The reference derives it as `0.1·ln(factor) + 1` when the config does not state one, and a
    /// config that states one overrides that. `linear` and `llama3` use no such factor.
    public var attentionFactor: Float {
        if kind == .longrope {
            if let declared = declaredAttentionFactor { return declared }
            let scale = Float(maximumPositionEmbeddings) / Float(originalMaxPositionEmbeddings)
            return scale <= 1 ? 1 : sqrt(1 + log(scale) / log(Float(originalMaxPositionEmbeddings)))
        }
        guard kind == .yarn else { return 1 }
        if let declared = declaredAttentionFactor { return declared }
        return factor <= 1 ? 1 : 0.1 * log(factor) + 1
    }

    /// The periods for one `longrope` factor table, which are the per-pair frequency divisors times the
    /// unscaled period. `useLongTable` selects ``longFactor`` (a sequence past the trained window) over
    /// ``shortFactor``. Reads nothing for other kinds.
    public func longRoPEPeriods(dimensions: Int, base: Float, useLongTable: Bool) -> [Float] {
        let pairs = max(dimensions / 2, 0)
        guard pairs > 0 else { return [] }
        let factors = useLongTable ? longFactor : shortFactor
        return (0 ..< pairs).map { index in
            let factor = index < factors.count ? factors[index] : 1
            return factor * powf(base, Float(2 * index) / Float(dimensions))
        }
    }

    /// The scaled inverse frequencies, one per rotary channel pair.
    ///
    /// - Parameters:
    ///   - dimensions: the rotary width in channels, which is twice the pair count. The correction
    ///     band is derived from the full width, and the ramp is evaluated over the pairs, which is
    ///     why both are needed rather than just one.
    ///   - base: the rotary base, `rope_theta`.
    public func inverseFrequencies(dimensions: Int, base: Float) -> [Float] {
        let pairs = max(dimensions / 2, 0)
        guard pairs > 0 else { return [] }
        let unscaled = (0 ..< pairs).map { 1 / powf(base, Float(2 * $0) / Float(dimensions)) }

        switch kind {
        case .linear:
            return unscaled.map { $0 / factor }

        case .yarn:
            // The channel index whose wavelength completes `rotations` turns within the trained
            // window. BELOW it a channel turns often enough to encode local offset and is left
            // alone; ABOVE it a channel has not completed a turn, so it is interpolated.
            func correctionDimension(_ rotations: Float) -> Float {
                Float(dimensions) * log(Float(originalMaxPositionEmbeddings)
                                        / (rotations * 2 * Float.pi)) / (2 * log(base))
            }
            // `truncate` (the default) snaps the band to whole channels; gpt-oss leaves it fractional.
            let low = max(truncatesCorrectionRange ? floor(correctionDimension(betaFast)) : correctionDimension(betaFast), 0)
            var high = min(truncatesCorrectionRange ? ceil(correctionDimension(betaSlow)) : correctionDimension(betaSlow),
                           Float(dimensions - 1))
            // A zero-width band would divide by zero; the reference opens it by a hair instead.
            if low == high { high += 0.001 }

            return (0 ..< pairs).map { index in
                let ramp = min(max((Float(index) - low) / (high - low), 0), 1)
                let extrapolation = 1 - ramp
                return unscaled[index] / factor * (1 - extrapolation) + unscaled[index] * extrapolation
            }
        case .llama3:
            let window = Float(originalMaxPositionEmbeddings)
            let lowWavelength = window / lowFrequencyFactor
            let highWavelength = window / highFrequencyFactor
            return unscaled.map { frequency in
                let wavelength = 2 * Float.pi / frequency
                if wavelength > lowWavelength { return frequency / factor }
                if wavelength < highWavelength { return frequency }
                // The middle band blends the interpolated and the unscaled frequency by how far the
                // channel's turn count sits between the two factors.
                let smooth = (window / wavelength - lowFrequencyFactor) / (highFrequencyFactor - lowFrequencyFactor)
                return (1 - smooth) * frequency / factor + smooth * frequency
            }
        case .longrope:
            // The seam-independent default is the within-window table, which is what a rotary
            // precomputed once uses; a sequence past the trained window switches to the long table
            // through ``longRoPEPeriods``.
            return longRoPEPeriods(dimensions: dimensions, base: base, useLongTable: false).map { 1 / $0 }
        }
    }

    /// The periods `MLXFast.RoPE` takes under `freqs:`, which are the reciprocals of the frequencies.
    public func rotaryPeriods(dimensions: Int, base: Float) -> [Float] {
        inverseFrequencies(dimensions: dimensions, base: base).map { 1 / $0 }
    }

    // MARK: Reading a release

    /// `config` without a dynamic-NTK `rope_scaling` block, for a model whose sequences stay under its
    /// `max_position_embeddings`. Dynamic NTK recomputes the rotary only once a sequence passes that
    /// length; below it the rotary is the base one, which is what the config then declares.
    static func droppingDynamic(_ config: [String: Any]) -> [String: Any] {
        guard let scaling = config["rope_scaling"] as? [String: Any],
              ((scaling["rope_type"] ?? scaling["type"]) as? String)?.lowercased() == "dynamic" else {
            return config
        }
        var adjusted = config
        adjusted["rope_scaling"] = nil
        return adjusted
    }

    /// Reads a checkpoint's `rope_scaling` block.
    ///
    /// - Returns: the scaling, or `nil` when the config declares none or declares the no-op `default`.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` for a kind this does not implement — `dynamic`
    ///   appears in released configs and computes different frequencies. Loading it under a rotary it
    ///   does not match produces a model that runs and is wrong, so it is refused.
    public static func read(_ block: Any?, maximumPositions: Int) throws -> NFKMLXRoPEScaling? {
        guard let scaling = block as? [String: Any] else { return nil }
        // `rope_type` is the current spelling; `type` is what older configs carry.
        let name = ((scaling["rope_type"] ?? scaling["type"]) as? String)?.lowercased() ?? "default"
        if name == "default" { return nil }

        func real(_ key: String, _ fallback: Float) -> Float {
            (scaling[key] as? NSNumber)?.floatValue ?? fallback
        }
        func integer(_ key: String, _ fallback: Int) -> Int {
            (scaling[key] as? NSNumber)?.intValue ?? fallback
        }
        func floatList(_ key: String) -> [Float] {
            (scaling[key] as? [Any])?.compactMap { ($0 as? NSNumber)?.floatValue } ?? []
        }

        if name == "longrope" {
            let short = floatList("short_factor")
            let long = floatList("long_factor")
            guard !short.isEmpty, !long.isEmpty else {
                throw NFKMLXError.unsupportedConfiguration(
                    "longrope rope_scaling needs both short_factor and long_factor tables")
            }
            var scaling = NFKMLXRoPEScaling(
                kind: .longrope, factor: real("factor", 1),
                originalMaxPositionEmbeddings: integer("original_max_position_embeddings", maximumPositions),
                declaredAttentionFactor: (scaling["attention_factor"] as? NSNumber)?.floatValue)
            scaling.shortFactor = short
            scaling.longFactor = long
            scaling.maximumPositionEmbeddings = maximumPositions
            return scaling
        }

        guard let kind = Kind(rawValue: name == "deepseek_yarn" ? "yarn" : name) else {
            throw NFKMLXError.unsupportedConfiguration(
                "rope_scaling type '\(name)' is not implemented; this reads 'linear', 'yarn', and 'llama3'. "
                + "Loading a release under a rotary it was not trained with produces a model that "
                + "runs and is wrong, so it is refused rather than approximated.")
        }
        let factor = real("factor", 1)
        guard factor > 0 else {
            throw NFKMLXError.unsupportedConfiguration("rope_scaling factor must be positive")
        }
        return NFKMLXRoPEScaling(
            kind: kind,
            factor: factor,
            originalMaxPositionEmbeddings: integer("original_max_position_embeddings",
                                                   maximumPositions),
            betaFast: real("beta_fast", 32),
            betaSlow: real("beta_slow", 1),
            declaredAttentionFactor: (scaling["attention_factor"] as? NSNumber)?.floatValue,
            lowFrequencyFactor: real("low_freq_factor", 1),
            highFrequencyFactor: real("high_freq_factor", 4),
            truncatesCorrectionRange: (scaling["truncate"] as? NSNumber)?.boolValue ?? true)
    }
}
