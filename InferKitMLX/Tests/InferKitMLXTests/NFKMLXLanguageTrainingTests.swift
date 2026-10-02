//
//  NFKMLXLanguageTrainingTests.swift
//  InferKitMLXTests
//
//  The customization path of the dense, hybrid, and Gemma 3 decoders: the causal language-model
//  objective agrees with the reference `labels=` loss, prompt positions masked and not, both on the
//  reference's own logits and through this package's forward on the reference's weights; LoRA adapts
//  only the target projections; a fine-tune lowers the loss on the sequence it trains on; and the merged
//  checkpoint reloads through each family's own factory unchanged.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXLanguageTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_001)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// The geometry `run_reference.py qwen3_loss` builds.
    private var denseConfiguration: NFKMLXLanguageConfiguration {
        NFKMLXLanguageConfiguration(hiddenSize: 64, layerCount: 2, headCount: 4, keyValueHeadCount: 2,
                                    headDimensions: 16, intermediateSize: 96, vocabularySize: 128,
                                    ropeTheta: 10_000, rmsEpsilon: 1e-6, tiesWordEmbeddings: false)
    }

    /// The geometry `run_reference.py qwen3_5_loss` builds: three recurrence layers, then attention.
    private var hybridConfiguration: NFKMLXHybridConfiguration {
        NFKMLXHybridConfiguration(hiddenSize: 64, layerCount: 4, intermediateSize: 96, vocabularySize: 128,
                                  headCount: 4, keyValueHeadCount: 2, headDimensions: 32, ropeTheta: 10_000,
                                  linearKeyHeadCount: 2, linearKeyHeadDimensions: 16,
                                  linearValueHeadCount: 4, linearValueHeadDimensions: 16)
    }

    /// The geometry `run_reference.py gemma3_loss` builds, with a window shorter than the sequence.
    private var gemmaConfiguration: NFKMLXGemma3Configuration {
        NFKMLXGemma3Configuration(hiddenSize: 64, layerCount: 4, headCount: 4, keyValueHeadCount: 2,
                                  headDimensions: 16, intermediateSize: 96, vocabularySize: 128,
                                  slidingWindow: 4, layerTypes: [.sliding, .sliding, .sliding, .full],
                                  queryPreAttnScalar: 16)
    }

    private let tokens = MLXArray([Int32(3), 17, 42, 99, 7, 61, 5, 88])

    // MARK: - The objective

    func testObjectiveScoresOnlyTheTokensAfterThePrompt() throws {
        try requireMLXRuntime()
        let vocabulary = 8
        let sequence = MLXArray([Int32(1), 2, 3, 4])
        // Positions 0 and 1 point at the wrong token, position 2 at the right one (4).
        var values = [Float](repeating: 0, count: 4 * vocabulary)
        values[0 * vocabulary + 7] = 30
        values[1 * vocabulary + 7] = 30
        values[2 * vocabulary + 4] = 30
        let logits = MLXArray(values, [1, 4, vocabulary])
        let objective = NFKMLXCausalLanguageObjective()
        XCTAssertLessThan(objective.loss(logits: logits, tokens: sequence, promptLength: 3).item(Float.self), 1e-3,
                          "only the token after the three-token prompt is scored")
        XCTAssertGreaterThan(objective.loss(logits: logits, tokens: sequence).item(Float.self), 10,
                             "without a prompt every following token is scored")
        XCTAssertEqual(objective.loss(logits: logits, tokens: sequence, promptLength: 4).item(Float.self), 0,
                       "a prompt covering the sequence scores nothing")
    }

    /// Holds the objective to a `run_reference.py` loss record: on the reference's logits, and through
    /// `forward` over `net` once the record's weights are applied with `rename`.
    private func checkRecord<Net: Module>(_ variable: String, mode: String, net: Net,
                                          rename: (String, MLXArray) -> (String, MLXArray)?,
                                          forward: (Net, MLXArray) -> MLXArray) throws {
        guard let path = NFKMLXValidationConfig.environment[variable] else {
            throw XCTSkip("set \(variable) (run_reference.py \(mode))")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let tokens = record["tokens"]!.asType(.int32)
        let promptLength = Int(record["prompt_length"]!.asArray(Int32.self)[0])
        let length = tokens.shape[0], vocabulary = record["logits"]!.shape[1]
        let logits = record["logits"]!.reshaped([1, length, vocabulary])
        let masked = record["output"]!.asArray(Float.self)[0]
        let unmasked = record["loss_unmasked"]!.asArray(Float.self)[0]
        let objective = NFKMLXCausalLanguageObjective()

        let onLogits = objective.loss(logits: logits, tokens: tokens, promptLength: promptLength).item(Float.self)
        let onLogitsWhole = objective.loss(logits: logits, tokens: tokens).item(Float.self)
        print("VALIDATION PARITY \(mode) objective: masked \(onLogits) vs \(masked), whole \(onLogitsWhole) vs \(unmasked)")
        XCTAssertEqual(onLogits, masked, accuracy: 1e-5, "the prompt-masked loss diverges from the reference")
        XCTAssertEqual(onLogitsWhole, unmasked, accuracy: 1e-5, "the whole-sequence loss diverges from the reference")

        let weights = record.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("w::") ? rename(String(key.dropFirst(3)), value) : nil
        }
        try NFKMLXWeights.apply(weights, to: net, verifyShapes: true)
        let ours = forward(net, tokens.reshaped([1, length]))
        let difference = abs(ours - logits).max().item(Float.self)
        let throughForward = objective.loss(logits: ours, tokens: tokens, promptLength: promptLength).item(Float.self)
        print("VALIDATION PARITY \(mode) forward: logits max |Δ| \(difference), loss \(throughForward) vs \(masked)")
        XCTAssertLessThan(difference, 1e-4, "the forward on the reference's weights diverges from its logits")
        XCTAssertEqual(throughForward, masked, accuracy: 1e-4, "the loss through the forward diverges from the reference")
    }

    func testDenseObjectiveMatchesTheReferenceLoss() throws {
        try requireMLXRuntime()
        try checkRecord("IK_PARITY_QWEN3_LOSS", mode: "qwen3_loss",
                        net: NFKMLXLanguage.makeNet(denseConfiguration),
                        rename: { ($0, $1) }, forward: { $0($1) })
    }

    func testHybridObjectiveMatchesTheReferenceLoss() throws {
        try requireMLXRuntime()
        try checkRecord("IK_PARITY_QWEN3_5_LOSS", mode: "qwen3_5_loss",
                        net: NFKMLXHybridLanguage.makeNet(hybridConfiguration),
                        rename: { key, value in
                            (key, key.hasSuffix("conv1d.weight") && value.ndim == 3 ? value.transposed(0, 2, 1) : value)
                        },
                        forward: { $0($1) })
    }

    func testGemma3ObjectiveMatchesTheReferenceLoss() throws {
        try requireMLXRuntime()
        try checkRecord("IK_PARITY_GEMMA3_LOSS", mode: "gemma3_loss",
                        net: NFKMLXGemma3Language.makeNet(gemmaConfiguration),
                        rename: { key, value in NFKMLXGemma3Language.decoderName(of: key).map { ($0, value) } },
                        forward: { $0($1) })
    }

    // MARK: - The recipes

    private func adaptedPaths(_ net: Module) -> [String] {
        net.leafModules().flattened().filter { $0.1 is NFKMLXLoRALinear }.map(\.0)
    }

    private func temporaryCheckpoint() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
    }

    func testDenseFineTuningAdaptsAttentionAndRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        let config = denseConfiguration
        let net = try NFKMLXLanguage.network(weightsURL: nil, configuration: config)
        let feedForwardBefore = net.model.layers[0].feedForward.parameters().flattened()[0].1
        eval(feedForwardBefore)
        let objective = NFKMLXCausalLanguageObjective()
        let before = objective(net, tokens, promptLength: 3).item(Float.self)
        let losses = try NFKMLXLanguage.fineTune(net, examples: { _ in (self.tokens, 3) }, rank: 4, steps: 12)
        XCTAssertEqual(losses.count, 12)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the sequence it trains on")
        XCTAssertEqual(abs(net.model.layers[0].feedForward.parameters().flattened()[0].1 - feedForwardBefore).max().item(Float.self), 0,
                       "the feed-forward stays frozen")
        let adapted = adaptedPaths(net)
        XCTAssertEqual(adapted.count, 2 * config.layerCount)
        XCTAssertTrue(adapted.allSatisfy { $0.contains(".self_attn.") && ($0.hasSuffix(".q_proj") || $0.hasSuffix(".v_proj")) })

        XCTAssertEqual(try NFKMLXLoRA.merge(into: net), adapted.count)
        let url = temporaryCheckpoint()
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXLanguage.network(weightsURL: url, configuration: config)
        let input = tokens.reshaped([1, tokens.shape[0]])
        XCTAssertLessThan(abs(net(input) - reloaded(input)).max().item(Float.self), 1e-5,
                          "the merged checkpoint reloads through the factory")
    }

    func testDenseFineTuningRefusesNothingToAdapt() throws {
        try requireMLXRuntime()
        let net = NFKMLXLanguage.makeNet(denseConfiguration)
        try NFKMLXLoRA.apply(to: net, rank: 4) { path, _ in NFKMLXLanguage.isAttentionProjection(path) }
        XCTAssertThrowsError(try NFKMLXLanguage.fineTune(net, examples: { _ in (self.tokens, 0) }, rank: 4, steps: 1),
                             "an already adapted decoder has no plain projection left to adapt")
    }

    func testHybridFineTuningAdaptsTheReleasedTargetsAndRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        let config = hybridConfiguration
        let net = try NFKMLXHybridLanguage.network(weightsURL: nil, configuration: config)
        let decayBefore = net.model.layers[0].linearAttention!.decayLog
        let convolutionBefore = net.model.layers[0].linearAttention!.convolution.weight
        eval(decayBefore, convolutionBefore)
        let objective = NFKMLXCausalLanguageObjective()
        let before = objective(net, tokens, promptLength: 3).item(Float.self)
        let losses = try NFKMLXHybridLanguage.fineTune(net, examples: { _ in (self.tokens, 3) }, rank: 4, steps: 12)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the sequence it trains on")
        XCTAssertEqual(abs(net.model.layers[0].linearAttention!.decayLog - decayBefore).max().item(Float.self), 0,
                       "the recurrence's decay stays frozen")
        XCTAssertEqual(abs(net.model.layers[0].linearAttention!.convolution.weight - convolutionBefore).max().item(Float.self), 0,
                       "the recurrence's convolution stays frozen")
        let adapted = adaptedPaths(net)
        // Three recurrence layers adapt two projections each; the attention layer adapts four.
        XCTAssertEqual(adapted.count, 3 * 2 + 4)
        XCTAssertTrue(adapted.allSatisfy { NFKMLXHybridLanguage.isAdaptedProjection($0) })

        XCTAssertEqual(try NFKMLXLoRA.merge(into: net), adapted.count)
        let url = temporaryCheckpoint()
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXHybridLanguage.network(weightsURL: url, configuration: config)
        let input = tokens.reshaped([1, tokens.shape[0]])
        XCTAssertLessThan(abs(net(input) - reloaded(input)).max().item(Float.self), 1e-5,
                          "the merged checkpoint reloads through the factory")
    }

    func testGemma3FineTuningAdaptsAttentionAndRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        let config = gemmaConfiguration
        let net = try NFKMLXGemma3Language.network(weightsURL: nil, configuration: config)
        let embeddingBefore = net.embedTokens.weight
        eval(embeddingBefore)
        let objective = NFKMLXCausalLanguageObjective()
        let before = objective(net, tokens, promptLength: 3).item(Float.self)
        let losses = try NFKMLXGemma3Language.fineTune(net, examples: { _ in (self.tokens, 3) }, rank: 4, steps: 12)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the sequence it trains on")
        XCTAssertEqual(abs(net.embedTokens.weight - embeddingBefore).max().item(Float.self), 0,
                       "the tied embedding stays frozen")
        let adapted = adaptedPaths(net)
        XCTAssertEqual(adapted.count, 2 * config.layerCount)
        XCTAssertTrue(adapted.allSatisfy { $0.hasSuffix(".self_attn.q_proj") || $0.hasSuffix(".self_attn.v_proj") })

        XCTAssertEqual(try NFKMLXLoRA.merge(into: net), adapted.count)
        let url = temporaryCheckpoint()
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXGemma3Language.network(weightsURL: url, configuration: config)
        let input = tokens.reshaped([1, tokens.shape[0]])
        XCTAssertLessThan(abs(net(input) - reloaded(input)).max().item(Float.self), 1e-5,
                          "the merged checkpoint reloads through the factory")
    }

    func testFineTuningRunsInBFloat16() throws {
        try requireMLXRuntime()
        let net = NFKMLXGemma3Language.makeNet(gemmaConfiguration)
        let losses = try NFKMLXGemma3Language.fineTune(net, examples: { _ in (self.tokens, 0) }, rank: 4, steps: 3,
                                                       precision: .bfloat16)
        XCTAssertEqual(losses.count, 3)
        XCTAssertTrue(losses.allSatisfy(\.isFinite), "a bfloat16 pass produces a finite loss")
        XCTAssertEqual(net.embedTokens.weight.dtype, .float32, "the stored weights stay float32")
    }
}
