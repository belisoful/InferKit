//
//  NFKMLXNemotronDraftingTests.swift
//  InferKitMLXTests
//
//  Drafted generation for the Nemotron-H decoder. A random decoder drafted by a smaller random decoder of
//  its vocabulary must reply exactly as it does alone at temperature 0, in fewer passes of its own. The
//  factory tests write both as releases with Nemotron Nano 2's tokenizer (`IK_TOK_NEMOTRON_H`). On released
//  weights, Nemotron Nano 9B v2 drafted by Nemotron 3 Nano 4B keeps its greedy reply
//  (`IK_VAL_NEMOTRON_H`, `IK_VAL_NEMOTRON3_NANO_4B`).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXNemotronDraftingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func configuration(vocabularySize: Int, pattern: String) -> NFKMLXNemotronHConfiguration {
        NFKMLXNemotronHConfiguration(
            hiddenSize: 64, vocabularySize: vocabularySize, rmsEpsilon: 1e-5,
            headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            mambaHeadCount: 8, mambaHeadDimensions: 16, mambaGroupCount: 2, mambaStateSize: 16,
            mambaConvolutionKernel: 4, mambaConvolutionBias: true, mambaProjectionBias: false,
            timeStepMinimum: 0.001, intermediateSize: 96, mlpBias: false,
            layerTypes: NFKMLXNemotronHConfiguration.layerKinds(fromPattern: pattern))
    }

    /// A random decoder of `configuration`, its norms varied, seeded with `seed`.
    private func randomNet(_ configuration: NFKMLXNemotronHConfiguration, seed: UInt64) -> NFKMLXNemotronHNet {
        MLXRandom.seed(seed)
        let net = NFKMLXNemotronH.makeNet(configuration)
        let varied = net.parameters().flattened().map { key, value in
            (key, key.hasSuffix("norm.weight") ? 1 + MLXRandom.normal(value.shape) * 0.1 : value)
        }
        net.update(parameters: ModuleParameters.unflattened(varied))
        eval(net)
        return net
    }

    private func reply(_ backend: NFKMLXNemotronBackend, draftTokens: Int? = nil) throws -> String? {
        var parameters: [String: Any] = [NFKParameterTemperature: 0, NFKParameterMaxTokens: 10]
        parameters[NFKMLXGenerationParameterKey.draftTokens] = draftTokens
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "abcde"], parameters: parameters)
        return try backend.runInference(for: request).output(forKey: NFKOutputText) as? String
    }

    func testADraftedReplyIsTheGreedyReplyWhateverTheDraftProposes() throws {
        try requireMLXRuntime()
        let target = randomNet(configuration(vocabularySize: 128, pattern: "M-M*-M-"), seed: 4)
        let draft = randomNet(configuration(vocabularySize: 128, pattern: "M*-"), seed: 5)
        let tokenizer = NFKMLXScalarTestTokenizer(vocabularySize: 128)
        let plain = NFKMLXNemotronBackend(logits: { target($0) }, tokenizer: tokenizer, identifier: "nemotron")
        let drafted = NFKMLXNemotronBackend(logits: { target($0) }, tokenizer: tokenizer, identifier: "nemotron",
                                            draft: { draft($0) })
        XCTAssertFalse(plain.hasDraftModel)
        XCTAssertFalse(plain.supportedParameterKeys.contains(NFKMLXGenerationParameterKey.draftTokens))
        XCTAssertTrue(drafted.hasDraftModel)
        XCTAssertTrue(drafted.supportedParameterKeys.contains(NFKMLXGenerationParameterKey.draftTokens))

        let expected = try XCTUnwrap(try reply(plain))
        XCTAssertEqual(try reply(drafted, draftTokens: 3), expected, "the drafted greedy reply")
        let report = drafted.lastSpeculativeReport
        XCTAssertGreaterThan(report.rounds, 0)
        XCTAssertEqual(try reply(drafted, draftTokens: 0), expected, "no draft tokens decodes without the draft")
        XCTAssertEqual(drafted.lastSpeculativeReport.rounds, report.rounds, "a plain run leaves the last report")
    }

    func testADraftThatAgreesKeepsEveryProposalAndTakesARoundPerFourTokens() throws {
        try requireMLXRuntime()
        let target = randomNet(configuration(vocabularySize: 128, pattern: "M-M*-M-"), seed: 4)
        let tokenizer = NFKMLXScalarTestTokenizer(vocabularySize: 128)
        let plain = NFKMLXNemotronBackend(logits: { target($0) }, tokenizer: tokenizer, identifier: "nemotron")
        let drafted = NFKMLXNemotronBackend(logits: { target($0) }, tokenizer: tokenizer, identifier: "nemotron",
                                            draft: { target($0) })
        XCTAssertEqual(try reply(drafted, draftTokens: 3), try reply(plain))
        let report = drafted.lastSpeculativeReport
        XCTAssertEqual(report.accepted, report.proposed, "the target agrees with itself")
        // Ten tokens at three proposals and the target's own token a round: rounds of 4, 4, and 2.
        XCTAssertEqual(report.rounds, 3)
    }

    // MARK: The drafted factory

    /// A release directory of `net` with `pattern` in its `config.json` and Nemotron Nano 2's tokenizer.
    private func release(_ net: NFKMLXNemotronHNet, pattern: String, tokenizer: URL) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("nemotron-draft-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let c = net.config
        let config: [String: Any] = [
            "model_type": "nemotron_h", "architectures": ["NemotronHForCausalLM"], "hidden_size": c.hiddenSize,
            "vocab_size": c.vocabularySize, "num_attention_heads": c.headCount, "num_key_value_heads": c.keyValueHeadCount,
            "head_dim": c.headDimensions, "mamba_num_heads": c.mambaHeadCount, "mamba_head_dim": c.mambaHeadDimensions,
            "n_groups": c.mambaGroupCount, "ssm_state_size": c.mambaStateSize, "conv_kernel": c.mambaConvolutionKernel,
            "intermediate_size": c.intermediateSize, "hybrid_override_pattern": pattern, "layer_norm_epsilon": 1e-5,
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try FileManager.default.copyItem(at: tokenizer.appendingPathComponent(name), to: directory.appendingPathComponent(name))
        }
        let stored = net.parameters().flattened().map { key, value -> (String, MLXArray) in
            let name = key.hasPrefix("model.") ? "backbone." + key.dropFirst("model.".count) : key
            let shaped = key.hasSuffix("conv1d.weight") ? value.reshaped([value.dim(0), 1, value.dim(1)]) : value
            return (name, shaped.asType(.bfloat16))
        }
        try save(arrays: Dictionary(uniqueKeysWithValues: stored), url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    private func tokenizerDirectory() throws -> URL {
        let environment = NFKMLXValidationConfig.environment
        guard let path = environment["IK_TOK_NEMOTRON_H"] ?? environment["IK_VAL_NEMOTRON_H"] else {
            throw XCTSkip("set IK_TOK_NEMOTRON_H (a directory with Nemotron Nano 2's tokenizer.json)")
        }
        return URL(fileURLWithPath: path)
    }

    func testTheDraftedFactoryQuantizesTheDraftAndKeepsTheGreedyReply() throws {
        try requireMLXRuntime()
        let tokenizer = try tokenizerDirectory()
        let targetPattern = "M-M*-M-", draftPattern = "M*-"
        let target = try release(randomNet(configuration(vocabularySize: 131_072, pattern: targetPattern), seed: 6),
                                 pattern: targetPattern, tokenizer: tokenizer)
        let draft = try release(randomNet(configuration(vocabularySize: 131_072, pattern: draftPattern), seed: 7),
                                pattern: draftPattern, tokenizer: tokenizer)
        let budget = NFKMLXResidencyBudget.current()
        let plain = try NFKMLXNemotronH.backend(directoryURL: target, precision: .checkpoint, residency: .automatic,
                                                budget: budget)
        let drafted = try NFKMLXNemotronH.backend(directoryURL: target, precision: .checkpoint, residency: .automatic,
                                                  budget: budget, draftDirectoryURL: draft, draftBits: 4)
        XCTAssertTrue(drafted.hasDraftModel)
        XCTAssertEqual(drafted.streamedLayerCount, 0)
        XCTAssertEqual(try reply(drafted, draftTokens: 3), try reply(plain), "the drafted greedy reply")
        XCTAssertGreaterThan(drafted.lastSpeculativeReport.rounds, 0)
    }

    func testTheDraftedFactoryRefusesADraftOfAnotherVocabulary() throws {
        try requireMLXRuntime()
        let tokenizer = try tokenizerDirectory()
        let target = try release(randomNet(configuration(vocabularySize: 131_072, pattern: "M-M*-"), seed: 8),
                                 pattern: "M-M*-", tokenizer: tokenizer)
        let draft = try release(randomNet(configuration(vocabularySize: 128, pattern: "M*-"), seed: 9),
                                pattern: "M*-", tokenizer: tokenizer)
        XCTAssertThrowsError(try NFKMLXNemotronH.backend(directoryURL: target, draftDirectoryURL: draft)) { error in
            guard case NFKMLXError.unsupportedConfiguration(let message) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertTrue(message.contains("share the model's vocabulary"), message)
        }
    }

    // MARK: Released weights

    func testNemotronNano9BDraftedByNemotron3Nano4BKeepsItsGreedyReply() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let targetPath = environment["IK_VAL_NEMOTRON_H"], let draftPath = environment["IK_VAL_NEMOTRON3_NANO_4B"] else {
            throw XCTSkip("set IK_VAL_NEMOTRON_H and IK_VAL_NEMOTRON3_NANO_4B")
        }
        let target = URL(fileURLWithPath: targetPath), draft = URL(fileURLWithPath: draftPath)
        // The budget that holds the 9B's first half of its blocks, so the drafted run streams the rest, as on
        // a machine smaller than this one, and its peak leaves the machine room.
        let stage = try NFKMLXLayerStream.footprint(
            directory: target, precision: .checkpoint, includes: { NFKMLXNemotronH.moduleKey(of: $0) != nil },
            layerOf: { NFKMLXNemotronH.block(ofKey: $0)?.layer })
        let budget = NFKMLXResidencyBudget.leastBudget(
            holding: stage.streamedMinimumBytes + stage.layerBytes.prefix(stage.layerBytes.count / 2).reduce(0, +))
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "The capital of France is"],
                                          parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 16])
        let expected = try autoreleasepool { () -> String in
            let plain = try NFKMLXNemotronH.backend(directoryURL: target, precision: .checkpoint, residency: .streamed,
                                                    budget: budget)
            XCTAssertGreaterThan(plain.streamedLayerCount, 0)
            return try XCTUnwrap(try plain.runInference(for: request).output(forKey: NFKOutputText) as? String)
        }
        NFKMLXGPU.clearCache()
        let drafted = try NFKMLXNemotronH.backend(directoryURL: target, precision: .checkpoint, residency: .streamed,
                                                  budget: budget, draftDirectoryURL: draft, draftBits: 4)
        XCTAssertGreaterThan(drafted.streamedLayerCount, 0, "the draft's bytes leave the 9B less to hold")
        let started = Date()
        let reply = try drafted.runInference(for: request).output(forKey: NFKOutputText) as? String
        let report = drafted.lastSpeculativeReport
        print("VALIDATION nemotron-nano-9b-v2 drafted by nemotron-3-nano-4b at 4 bits: \(report.rounds) rounds, "
              + "\(report.accepted) of \(report.proposed) proposals kept, \(drafted.streamedLayerCount) of "
              + "\(stage.layerBytes.count) blocks streamed, \(Int(Date().timeIntervalSince(started))) s; reply \(reply ?? "nil")")
        XCTAssertEqual(reply, expected, "the drafted greedy reply")
        XCTAssertGreaterThan(report.accepted, 0, "the draft's proposals are kept")
    }
}
