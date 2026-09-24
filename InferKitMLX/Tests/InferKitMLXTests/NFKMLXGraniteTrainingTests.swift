//
//  NFKMLXGraniteTrainingTests.swift
//  InferKitMLXTests
//
//  The Granite 4.0-H customization path: the causal language-model objective agrees with the reference
//  loss on identical logits, LoRA adapts only the attention query and value projections (the Mamba
//  layers and everything else stay frozen), a fine-tune lowers the loss on the sequence it trains on,
//  and the merged checkpoint reloads through the model's own factory unchanged.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXGraniteTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// A small dense hybrid with two attention layers, so LoRA has clear targets and a Mamba layer is
    /// present to prove it stays frozen.
    private var tinyConfiguration: NFKMLXGraniteHybridConfiguration {
        NFKMLXGraniteHybridConfiguration(
            hiddenSize: 64, layerCount: 4, vocabularySize: 128, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: false, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            mambaHeadCount: 8, mambaHeadDimensions: 16, mambaGroupCount: 1, mambaStateSize: 16,
            mambaConvolutionKernel: 4, mambaExpand: 2, mambaConvolutionBias: true,
            mambaProjectionBias: false, sharedIntermediateSize: 96, expertCount: 0,
            expertsPerToken: 0, expertIntermediateSize: 0,
            embeddingMultiplier: 2.0, residualMultiplier: 0.5, attentionMultiplier: 0.25,
            logitsScaling: 3.0,
            layerTypes: [.mamba, .attention, .mamba, .attention])
    }

    func testObjectiveMatchesTheReferenceLoss() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let recordPath = env["IK_PARITY_GRANITE_LOSS"] else {
            throw XCTSkip("set IK_PARITY_GRANITE_LOSS (run_reference.py granite_hybrid_loss)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let tokens = rec["tokens"]!.asType(.int32)
        let length = tokens.shape[0], vocabulary = rec["logits"]!.shape[1]
        let logits = rec["logits"]!.reshaped([1, length, vocabulary])
        let loss = NFKMLXGraniteObjective().loss(logits: logits, tokens: tokens)
        let reference = rec["output"]!.asArray(Float.self)[0]
        print("VALIDATION PARITY granite objective loss: \(loss.item(Float.self)) vs reference \(reference)")
        XCTAssertEqual(loss.item(Float.self), reference, accuracy: 1e-3, "the objective loss diverges from the reference")
    }

    func testObjectiveScoresAlignedPredictionsAsCertain() throws {
        try requireMLXRuntime()
        // Each position points overwhelmingly at the next token, so the shifted cross-entropy is ~0.
        let vocabulary = 8
        let tokens = MLXArray([Int32(1), 2, 3, 4])
        var values = [Float](repeating: 0, count: 4 * vocabulary)
        for position in 0 ..< 3 { values[position * vocabulary + Int(2 + position)] = 30 }  // predict 2, 3, 4
        let loss = NFKMLXGraniteObjective().loss(logits: MLXArray(values, [1, 4, vocabulary]), tokens: tokens)
        XCTAssertLessThan(loss.item(Float.self), 1e-3, "aligned predictions score as near-certain")
    }

    func testFineTuningAdaptsAttentionAndRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        let config = tinyConfiguration
        let net = try NFKMLXGraniteHybrid.network(weightsURL: nil, configuration: config)
        let mambaBefore = net.model.layers[0].mamba!.inProjection.weight
        eval(mambaBefore)
        let tokens = MLXArray([Int32(3), 17, 42, 99, 7, 61])
        let objective = NFKMLXGraniteObjective()
        let before = objective(net, tokens).item(Float.self)
        let losses = try NFKMLXGraniteHybrid.fineTune(net, examples: { _ in tokens }, rank: 4, steps: 12)
        XCTAssertEqual(losses.count, 12)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the sequence it trains on")
        XCTAssertEqual(abs(net.model.layers[0].mamba!.inProjection.weight - mambaBefore).max().item(Float.self), 0,
                       "the Mamba layer stays frozen")
        let adapted = net.leafModules().flattened().filter { $0.1 is NFKMLXLoRALinear }.map(\.0)
        XCTAssertFalse(adapted.isEmpty)
        XCTAssertTrue(adapted.allSatisfy { $0.contains(".self_attn.") && ($0.hasSuffix(".q_proj") || $0.hasSuffix(".v_proj")) })

        let merged = try NFKMLXLoRA.merge(into: net)
        XCTAssertEqual(merged, adapted.count)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXGraniteHybrid.network(weightsURL: url, configuration: config)
        let a = net(tokens.reshaped([1, 6]))
        let b = reloaded(tokens.reshaped([1, 6]))
        XCTAssertLessThan(abs(a - b).max().item(Float.self), 1e-5, "the merged checkpoint reloads through the factory")
    }
}
