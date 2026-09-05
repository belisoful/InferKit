//
//  NFKMLXRoPEScalingTests.swift
//  InferKitMLXTests
//
//  Rotary frequency scaling, against `transformers`' own `ROPE_INIT_FUNCTIONS` — the dispatch every
//  released decoder's `rope_scaling` block is read by, so matching it matches every model that
//  declares one.
//
//  The record carries the PARAMETERS each case used alongside its result, so this reads the
//  configuration from the record rather than repeating literals that could drift out of step with the
//  oracle. Regenerate it with:
//    run_reference.py rope_scaling <out>.safetensors
//

import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXRoPEScalingTests: XCTestCase {

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    // MARK: Reference parity

    /// Every case in the record, against the frequencies transformers computes for the same config.
    func testTheScaledFrequenciesMatchTransformers() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "reads a safetensors record through MLX; run via xcodebuild")
        guard let path = config["IK_PARITY_ROPE_SCALING"] else {
            throw XCTSkip("set IK_PARITY_ROPE_SCALING in ~/.inferkit-validation.json")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let count = try XCTUnwrap(record["case_count"]).item(Int32.self)
        XCTAssertGreaterThan(count, 0, "the record carries cases")

        for index in 0 ..< Int(count) {
            let parameters = try XCTUnwrap(record["case\(index)_params"]).asArray(Float.self)
            let expected = try XCTUnwrap(record["case\(index)_inv_freq"]).asArray(Float.self)
            let expectedFactor = try XCTUnwrap(record["case\(index)_attention_factor"]).item(Float.self)

            // [dimensions, base, maxPositions, factor, originalMax, betaFast, betaSlow, kind,
            //  declaredAttentionFactor or -1, lowFrequencyFactor, highFrequencyFactor]
            let dimensions = Int(parameters[0])
            let base = parameters[1]
            let declared = parameters[8]
            let kinds: [Float: NFKMLXRoPEScaling.Kind] = [0: .linear, 1: .yarn, 2: .llama3]
            let scaling = NFKMLXRoPEScaling(
                kind: try XCTUnwrap(kinds[parameters[7]]),
                factor: parameters[3],
                originalMaxPositionEmbeddings: Int(parameters[4]),
                betaFast: parameters[5],
                betaSlow: parameters[6],
                declaredAttentionFactor: declared < 0 ? nil : declared,
                lowFrequencyFactor: parameters.count > 9 ? parameters[9] : 1,
                highFrequencyFactor: parameters.count > 10 ? parameters[10] : 4)

            let produced = scaling.inverseFrequencies(dimensions: dimensions, base: base)
            XCTAssertEqual(produced.count, expected.count, "case \(index): pair count")

            var worst: Float = 0
            for (a, b) in zip(produced, expected) {
                // Relative, because these span nine orders of magnitude: an absolute tolerance would
                // be vacuous at the low-frequency end and impossible at the high.
                worst = max(worst, abs(a - b) / max(abs(b), 1e-12))
            }
            XCTAssertLessThan(worst, 1e-5,
                              "case \(index) (\(scaling.kind), factor \(scaling.factor)): "
                              + "worst relative difference \(worst)")

            // Case 3 in the record states an attention factor explicitly, which must override the
            // derived one; the others exercise the derivation.
            XCTAssertEqual(scaling.attentionFactor, expectedFactor, accuracy: 1e-5,
                           "case \(index): attention factor")
        }
    }

    /// The declared factor overrides the derived one, which is the case the record's fourth entry
    /// covers from the oracle's side.
    func testADeclaredAttentionFactorOverridesTheDerivation() {
        let derived = NFKMLXRoPEScaling(kind: .yarn, factor: 40,
                                        originalMaxPositionEmbeddings: 4096)
        XCTAssertEqual(derived.attentionFactor, 0.1 * log(Float(40)) + 1, accuracy: 1e-6)

        var declared = derived
        declared.declaredAttentionFactor = 1.25
        XCTAssertEqual(declared.attentionFactor, 1.25)
    }

    // MARK: The shape of the blend

    /// YaRN does not scale every channel alike, and the direction is the opposite of the intuitive
    /// guess: the FAST channels are left as they were and the SLOW ones are interpolated. A fast
    /// channel turns many times inside the trained window, so it encodes local offset and a longer
    /// sequence does not change what it means. `linear` makes no such distinction, and the difference
    /// between the two is what this asserts.
    ///
    /// This test had it backwards until the reference-parity record disagreed with it.
    func testYarnLeavesTheFastChannelsAloneWhereLinearScalesThemAll() {
        let dimensions = 64
        let base: Float = 10_000
        let unscaled = (0 ..< dimensions / 2).map { 1 / powf(base, Float(2 * $0) / Float(dimensions)) }

        let yarn = NFKMLXRoPEScaling(kind: .yarn, factor: 16,
                                     originalMaxPositionEmbeddings: 2048)
            .inverseFrequencies(dimensions: dimensions, base: base)
        let linear = NFKMLXRoPEScaling(kind: .linear, factor: 16)
            .inverseFrequencies(dimensions: dimensions, base: base)

        // The first pair turns fastest, encodes local offset, and is left untouched.
        XCTAssertEqual(yarn[0], unscaled[0], accuracy: unscaled[0] * 1e-4,
                       "the fastest channel is extrapolated, not interpolated")
        // The last pair has not completed a turn within the trained window, so it is interpolated.
        XCTAssertEqual(yarn[yarn.count - 1], unscaled[unscaled.count - 1] / 16,
                       accuracy: unscaled[unscaled.count - 1] * 1e-4,
                       "the slowest channel is interpolated")
        // Linear makes no such distinction.
        XCTAssertEqual(linear[linear.count - 1], unscaled[unscaled.count - 1] / 16,
                       accuracy: unscaled[unscaled.count - 1] * 1e-4)
    }

    /// The blend is monotone in the channel index: it runs from untouched at the fast end to fully
    /// interpolated at the slow end and never turns back. A ramp computed with the boundaries
    /// reversed would still produce plausible-looking numbers and would fail this.
    func testTheBlendMovesInOneDirectionAcrossTheChannels() {
        let dimensions = 128
        let base: Float = 10_000
        let scaling = NFKMLXRoPEScaling(kind: .yarn, factor: 8,
                                        originalMaxPositionEmbeddings: 16384)
        let scaled = scaling.inverseFrequencies(dimensions: dimensions, base: base)
        let unscaled = (0 ..< dimensions / 2).map { 1 / powf(base, Float(2 * $0) / Float(dimensions)) }

        // The ratio to the unscaled frequency runs from 1 down to 1/factor, never back up.
        let ratios = zip(scaled, unscaled).map { $0 / $1 }
        for index in 1 ..< ratios.count {
            XCTAssertLessThanOrEqual(ratios[index], ratios[index - 1] + 1e-6,
                                     "the ramp is monotone at pair \(index)")
        }
        XCTAssertEqual(ratios.first!, 1, accuracy: 1e-3)
        XCTAssertEqual(ratios.last!, 1.0 / 8.0, accuracy: 1e-3)
    }

    // MARK: Reading a release

    func testAConfigWithNoScalingReadsAsNone() throws {
        XCTAssertNil(try NFKMLXRoPEScaling.read(nil, maximumPositions: 4096))
        XCTAssertNil(try NFKMLXRoPEScaling.read(["rope_type": "default"], maximumPositions: 4096))
    }

    func testTheOlderTypeSpellingIsRead() throws {
        let scaling = try XCTUnwrap(NFKMLXRoPEScaling.read(
            ["type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048],
            maximumPositions: 8192))
        XCTAssertEqual(scaling.kind, .yarn)
        XCTAssertEqual(scaling.factor, 4)
        XCTAssertEqual(scaling.originalMaxPositionEmbeddings, 2048)
    }

    func testDeepSeeksOwnSpellingReadsAsYarn() throws {
        let scaling = try XCTUnwrap(NFKMLXRoPEScaling.read(
            ["rope_type": "deepseek_yarn", "factor": 40.0], maximumPositions: 4096))
        XCTAssertEqual(scaling.kind, .yarn)
    }

    /// `dynamic`, `llama3` and `longrope` all appear in released configs and all compute different
    /// frequencies. Loading one under a rotary that does not match produces a model that runs and is
    /// wrong, so it is refused.
    func testAnUnimplementedKindIsRefusedRatherThanApproximated() {
        for kind in ["dynamic", "longrope"] {
            XCTAssertThrowsError(try NFKMLXRoPEScaling.read(["rope_type": kind, "factor": 4.0],
                                                            maximumPositions: 4096)) { error in
                guard case NFKMLXError.unsupportedConfiguration(let message) = error else {
                    return XCTFail("expected unsupportedConfiguration for \(kind), got \(error)")
                }
                XCTAssertTrue(message.contains(kind), "the message names the kind: \(message)")
            }
        }
    }

    /// A config that omits `original_max_position_embeddings` falls back to the model's window, which
    /// is what the field means when it is absent.
    func testTheOriginalWindowFallsBackToTheModels() throws {
        let scaling = try XCTUnwrap(NFKMLXRoPEScaling.read(
            ["rope_type": "yarn", "factor": 2.0], maximumPositions: 8192))
        XCTAssertEqual(scaling.originalMaxPositionEmbeddings, 8192)
    }

    // MARK: Through the decoder

    /// A configuration with no scaling must produce exactly what it produced before this existed, or
    /// every parity record in the package is invalidated.
    func testAnUnscaledConfigurationRotatesExactlyAsBefore() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "evaluates MLX arrays; run via xcodebuild")
        let dimensions = 16
        let base: Float = 10_000
        let plain = NFKLMRotary(dimensions: dimensions, base: base, scaling: nil)
        let x = MLXRandom.normal([1, 2, 5, dimensions], key: MLXRandom.key(3))

        let produced = plain(x, offset: 2)
        let reference = MLXFast.RoPE(x, dimensions: dimensions, traditional: false,
                                     base: base, scale: 1, offset: 2)
        eval(produced, reference)
        let worst = abs(produced - reference).max()
        eval(worst)
        XCTAssertEqual(worst.item(Float.self), 0, "no scaling is the identity on the rotary path")
    }

    /// A scaled rotary differs from an unscaled one — the change actually reaches the tensor.
    func testAScaledRotaryChangesTheResult() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "evaluates MLX arrays; run via xcodebuild")
        let dimensions = 16
        let scaling = NFKMLXRoPEScaling(kind: .yarn, factor: 8,
                                        originalMaxPositionEmbeddings: 512)
        let scaled = NFKLMRotary(dimensions: dimensions, base: 10_000, scaling: scaling)
        let plain = NFKLMRotary(dimensions: dimensions, base: 10_000, scaling: nil)
        let x = MLXRandom.normal([1, 1, 4, dimensions], key: MLXRandom.key(5))

        let difference = abs(scaled(x, offset: 64) - plain(x, offset: 64)).max()
        eval(difference)
        XCTAssertGreaterThan(difference.item(Float.self), 1e-3)
    }
}
