//
//  NFKMLXNemotronTrainingTests.swift
//  InferKitMLXTests
//
//  The Nemotron Nano 2 customization path: the causal language-model objective agrees with the
//  reference loss on identical logits, LoRA adapts only the attention query and value projections (the
//  Mamba layers, feed-forwards, and everything else stay frozen), a fine-tune lowers the loss on the
//  sequence it trains on, and the merged checkpoint reloads through the model's own factory unchanged.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXNemotronTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// A small hybrid with two attention blocks, so LoRA has clear targets, and a Mamba block present to
    /// prove it stays frozen.
    private var tinyConfiguration: NFKMLXNemotronHConfiguration {
        NFKMLXNemotronHConfiguration(
            hiddenSize: 64, vocabularySize: 128, rmsEpsilon: 1e-5,
            headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            mambaHeadCount: 8, mambaHeadDimensions: 16, mambaGroupCount: 2, mambaStateSize: 16,
            mambaConvolutionKernel: 4, mambaConvolutionBias: true, mambaProjectionBias: false,
            timeStepMinimum: 0.001, intermediateSize: 96, mlpBias: false,
            layerTypes: [.mamba, .attention, .mlp, .attention])
    }

    func testObjectiveMatchesTheReferenceLoss() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let recordPath = env["IK_PARITY_NEMOTRON_H_LOSS"] else {
            throw XCTSkip("set IK_PARITY_NEMOTRON_H_LOSS (run_reference.py nemotron_h_loss)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let tokens = rec["tokens"]!.asType(.int32)
        let length = tokens.shape[0], vocabulary = rec["logits"]!.shape[1]
        let logits = rec["logits"]!.reshaped([1, length, vocabulary])
        let loss = NFKMLXNemotronObjective().loss(logits: logits, tokens: tokens)
        let reference = rec["output"]!.asArray(Float.self)[0]
        print("VALIDATION PARITY nemotron objective loss: \(loss.item(Float.self)) vs reference \(reference)")
        XCTAssertEqual(loss.item(Float.self), reference, accuracy: 1e-3, "the objective loss diverges from the reference")
    }

    func testObjectiveScoresAlignedPredictionsAsCertain() throws {
        try requireMLXRuntime()
        // Each position points overwhelmingly at the next token, so the shifted cross-entropy is ~0.
        let vocabulary = 8
        let tokens = MLXArray([Int32(1), 2, 3, 4])
        var values = [Float](repeating: 0, count: 4 * vocabulary)
        for position in 0 ..< 3 { values[position * vocabulary + Int(2 + position)] = 30 }  // predict 2, 3, 4
        let loss = NFKMLXNemotronObjective().loss(logits: MLXArray(values, [1, 4, vocabulary]), tokens: tokens)
        XCTAssertLessThan(loss.item(Float.self), 1e-3, "aligned predictions score as near-certain")
    }

    func testFineTuningAdaptsAttentionAndRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        let config = tinyConfiguration
        let net = try NFKMLXNemotronH.network(weightsURL: nil, configuration: config)
        let mamba = net.model.layers[0].mixer as! NFKMLXMamba2Mixer
        let mambaBefore = mamba.inProjection.weight
        eval(mambaBefore)
        let tokens = MLXArray([Int32(3), 17, 42, 99, 7, 61])
        let objective = NFKMLXNemotronObjective()
        let before = objective(net, tokens).item(Float.self)
        let losses = try NFKMLXNemotronH.fineTune(net, examples: { _ in tokens }, rank: 4, steps: 12)
        XCTAssertEqual(losses.count, 12)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the sequence it trains on")
        XCTAssertEqual(abs(mamba.inProjection.weight - mambaBefore).max().item(Float.self), 0,
                       "the Mamba layer stays frozen")
        let adapted = net.leafModules().flattened().filter { $0.1 is NFKMLXLoRALinear }.map(\.0)
        XCTAssertFalse(adapted.isEmpty)
        XCTAssertTrue(adapted.allSatisfy { $0.contains(".mixer.") && ($0.hasSuffix(".q_proj") || $0.hasSuffix(".v_proj")) })

        let merged = try NFKMLXLoRA.merge(into: net)
        XCTAssertEqual(merged, adapted.count)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXNemotronH.network(weightsURL: url, configuration: config)
        let a = net(tokens.reshaped([1, 6]))
        let b = reloaded(tokens.reshaped([1, 6]))
        XCTAssertLessThan(abs(a - b).max().item(Float.self), 1e-5, "the merged checkpoint reloads through the factory")
    }
}
