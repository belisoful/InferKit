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
/// @discussion Two kinds are implemented, and they work differently. `linear` divides every frequency
/// by the same factor, which is position interpolation: the model sees a longer sequence squeezed into
/// its trained range, at the cost of resolution everywhere.
///
/// `yarn` divides only the SLOW channels and leaves the fast ones as they were, blending across a band
/// between. The direction is worth stating plainly, because the intuitive guess is the opposite one: a
/// fast channel completes many turns inside the trained window, so it encodes LOCAL offset, and that
/// meaning is unchanged by a longer sequence — interpolating it would only blur short-range position.
/// A slow channel does not complete a turn even at the trained length, so it encodes position across
/// the whole window; run past that length it reaches angles the model has never seen, and it is the
/// one that has to be squeezed.
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
    }

    public var kind: Kind

    /// How far the window is extended, as a multiple of the trained one.
    public var factor: Float

    /// The window the model was trained over, which is what decides which channels have wrapped.
    /// Unused by `linear`.
    public var originalMaxPositionEmbeddings: Int

    /// The rotation count marking the end of the extrapolation band. The paper's default is 32.
    public var betaFast: Float

    /// The rotation count marking the start of the interpolation band. The paper's default is 1.
    public var betaSlow: Float

    /// The config's own `attention_factor`, when it states one. `nil` derives it.
    public var declaredAttentionFactor: Float?

    public init(kind: Kind,
                factor: Float,
                originalMaxPositionEmbeddings: Int = 0,
                betaFast: Float = 32,
                betaSlow: Float = 1,
                declaredAttentionFactor: Float? = nil) {
        self.kind = kind
        self.factor = factor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.declaredAttentionFactor = declaredAttentionFactor
    }

    /// The factor the rotated queries and keys are multiplied by.
    ///
    /// @discussion Interpolating the frequencies lowers the average attention logit, and YaRN
    /// compensates with a scalar on the rotary embedding. It multiplies the queries and the keys
    /// alike, so an attention score carries its square.
    ///
    /// The reference derives it as `0.1·ln(factor) + 1` when the config does not state one, and a
    /// config that states one overrides that. `linear` uses no such factor.
    public var attentionFactor: Float {
        guard kind == .yarn else { return 1 }
        if let declared = declaredAttentionFactor { return declared }
        return factor <= 1 ? 1 : 0.1 * log(factor) + 1
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
            let low = max(floor(correctionDimension(betaFast)), 0)
            var high = min(ceil(correctionDimension(betaSlow)), Float(dimensions - 1))
            // A zero-width band would divide by zero; the reference opens it by a hair instead.
            if low == high { high += 0.001 }

            return (0 ..< pairs).map { index in
                let ramp = min(max((Float(index) - low) / (high - low), 0), 1)
                let extrapolation = 1 - ramp
                return unscaled[index] / factor * (1 - extrapolation) + unscaled[index] * extrapolation
            }
        }
    }

    /// The periods `MLXFast.RoPE` takes under `freqs:`, which are the reciprocals of the frequencies.
    public func rotaryPeriods(dimensions: Int, base: Float) -> [Float] {
        inverseFrequencies(dimensions: dimensions, base: base).map { 1 / $0 }
    }

    // MARK: Reading a release

    /// Reads a checkpoint's `rope_scaling` block.
    ///
    /// - Returns: the scaling, or `nil` when the config declares none or declares the no-op `default`.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` for a kind this does not implement —
    ///   `dynamic`, `llama3` and `longrope` all appear in released configs and all compute different
    ///   frequencies. Loading one of those under a rotary it does not match produces a model that
    ///   runs and is wrong, so it is refused.
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

        guard let kind = Kind(rawValue: name == "deepseek_yarn" ? "yarn" : name) else {
            throw NFKMLXError.unsupportedConfiguration(
                "rope_scaling type '\(name)' is not implemented; this reads 'linear' and 'yarn'. "
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
            declaredAttentionFactor: (scaling["attention_factor"] as? NSNumber)?.floatValue)
    }
}
