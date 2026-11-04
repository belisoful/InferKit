//
//  NFKMLXHybridDraftingTests.swift
//  InferKitMLXTests
//
//  Drafted generation for the Qwen3.5 hybrid decoder through `NFKMLXDecoderBackend`. A random decoder drafted
//  by a smaller random decoder of its vocabulary must reply exactly as it does alone at temperature 0, alone
//  and as a job. The factory tests write both as releases with Qwen3.5's tokenizer (`IK_VAL_QWEN3_5_2B`). On
//  released weights, Qwen3.5 9B drafted by Qwen3.5 2B keeps its greedy reply (`IK_VAL_QWEN3_5_9B`,
//  `IK_VAL_QWEN3_5_2B`).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXHybridDraftingTests: XCTestCase {

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

    private func configuration(vocabularySize: Int, layerCount: Int) -> NFKMLXHybridConfiguration {
        NFKMLXHybridConfiguration(
            hiddenSize: 64, layerCount: layerCount, intermediateSize: 128, vocabularySize: vocabularySize,
            headCount: 4, keyValueHeadCount: 2, headDimensions: 16, ropeTheta: 10_000,
            linearKeyHeadCount: 2, linearKeyHeadDimensions: 8, linearValueHeadCount: 4, linearValueHeadDimensions: 8,
            fullAttentionInterval: 4, tiesWordEmbeddings: true)
    }

    /// A random decoder of `configuration`, its norms varied, seeded with `seed`.
    private func randomNet(_ configuration: NFKMLXHybridConfiguration, seed: UInt64) -> NFKMLXHybridLanguageNet {
        MLXRandom.seed(seed)
        let net = NFKMLXHybridLanguage.makeNet(configuration)
        let varied = net.parameters().flattened().map { key, value in
            (key, key.hasSuffix("norm.weight") ? 1 + MLXRandom.normal(value.shape) * 0.1 : value)
        }
        net.update(parameters: ModuleParameters.unflattened(varied))
        eval(net)
        return net
    }

    private func request(draftTokens: Int?) -> NFKInferenceRequest {
        var parameters: [String: Any] = [NFKParameterTemperature: 0, NFKParameterMaxTokens: 10]
        parameters[NFKMLXGenerationParameterKey.draftTokens] = draftTokens
        return NFKInferenceRequest(inputs: [NFKInputPrompt: "abcde"], parameters: parameters)
    }

    private func reply(_ backend: NFKMLXDecoderBackend, draftTokens: Int? = nil) throws -> String? {
        try backend.runInference(for: request(draftTokens: draftTokens)).output(forKey: NFKOutputText) as? String
    }

    private func backend(_ target: NFKMLXHybridLanguageNet, draft: NFKMLXHybridLanguageNet? = nil) -> NFKMLXDecoderBackend {
        NFKMLXDecoderBackend(logits: { target($0) }, tokenizer: NFKMLXScalarTestTokenizer(vocabularySize: 128),
                             identifier: "qwen3.5", chatTemplate: nil, stopTokens: [],
                             draft: draft.map { draft in { draft($0) } })
    }

    func testADraftedReplyIsTheGreedyReplyWhateverTheDraftProposes() throws {
        try requireMLXRuntime()
        let target = randomNet(configuration(vocabularySize: 128, layerCount: 8), seed: 4)
        let draft = randomNet(configuration(vocabularySize: 128, layerCount: 4), seed: 5)
        let plain = backend(target)
        let drafted = backend(target, draft: draft)
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
        let target = randomNet(configuration(vocabularySize: 128, layerCount: 8), seed: 4)
        let drafted = backend(target, draft: target)
        XCTAssertEqual(try reply(drafted, draftTokens: 3), try reply(backend(target)))
        let report = drafted.lastSpeculativeReport
        XCTAssertEqual(report.accepted, report.proposed, "the target agrees with itself")
        // Ten tokens at three proposals and the target's own token a round: rounds of 4, 4, and 2.
        XCTAssertEqual(report.rounds, 3)
    }

    func testADraftedJobReportsTheReplyAsItGrowsThenTheGreedyReply() throws {
        try requireMLXRuntime()
        let target = randomNet(configuration(vocabularySize: 128, layerCount: 8), seed: 4)
        let draft = randomNet(configuration(vocabularySize: 128, layerCount: 4), seed: 5)
        let expected = try XCTUnwrap(try reply(backend(target)))
        let drafted = backend(target, draft: draft)
        drafted.partialResultInterval = 0
        let finished = expectation(description: "finished")
        let reports = NSMutableArray()
        let job = drafted.submitInferenceJob(for: request(draftTokens: 3))
        job.progressHandler = { job in
            if let text = job.partialResult?.output(forKey: NFKOutputText) as? String {
                reports.add(text)
            }
        }
        job.completionHandler = { _ in finished.fulfill() }
        wait(for: [finished], timeout: 30)
        XCTAssertEqual(try XCTUnwrap(job.result).output(forKey: NFKOutputText) as? String, expected)
        XCTAssertGreaterThan(reports.count, 0, "the job reports the reply as it grows")
        XCTAssertGreaterThan(drafted.lastSpeculativeReport.rounds, 0)
    }

    // MARK: The drafted factory

    /// A release directory of `net` in the released layout, with Qwen3.5's tokenizer.
    private func release(_ net: NFKMLXHybridLanguageNet, tokenizer: URL) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("hybrid-draft-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let c = net.configuration
        let text: [String: Any] = [
            "model_type": "qwen3_5_text", "hidden_size": c.hiddenSize, "num_hidden_layers": c.layerCount,
            "intermediate_size": c.intermediateSize, "vocab_size": c.vocabularySize, "rms_norm_eps": 1e-6,
            "num_attention_heads": c.headCount, "num_key_value_heads": c.keyValueHeadCount, "head_dim": c.headDimensions,
            "linear_num_key_heads": c.linearKeyHeadCount, "linear_key_head_dim": c.linearKeyHeadDimensions,
            "linear_num_value_heads": c.linearValueHeadCount, "linear_value_head_dim": c.linearValueHeadDimensions,
            "linear_conv_kernel_dim": c.linearConvolutionKernel, "tie_word_embeddings": true,
            "layer_types": c.layerTypes.map(\.rawValue),
            "rope_parameters": ["rope_theta": c.ropeTheta, "partial_rotary_factor": c.partialRotaryFactor],
        ]
        try JSONSerialization.data(withJSONObject: ["model_type": "qwen3_5", "text_config": text])
            .write(to: directory.appendingPathComponent("config.json"))
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try FileManager.default.copyItem(at: tokenizer.appendingPathComponent(name), to: directory.appendingPathComponent(name))
        }
        var arrays = [String: MLXArray]()
        for (key, value) in net.parameters().flattened() {
            let shaped = key.hasSuffix("conv1d.weight") ? value.transposed(0, 2, 1) : value
            arrays[NFKMLXHybridLanguage.referenceKey(for: key)] = shaped.asType(.bfloat16)
        }
        try save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    private func tokenizerDirectory() throws -> URL {
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_QWEN3_5_2B"] else {
            throw XCTSkip("set IK_VAL_QWEN3_5_2B (a Qwen3.5 release whose tokenizer the factory tests read)")
        }
        return URL(fileURLWithPath: path)
    }

    func testTheDraftedFactoryQuantizesTheDraftAndKeepsTheGreedyReply() throws {
        try requireMLXRuntime()
        let tokenizer = try tokenizerDirectory()
        let target = try release(randomNet(configuration(vocabularySize: 248_320, layerCount: 8), seed: 6), tokenizer: tokenizer)
        let draft = try release(randomNet(configuration(vocabularySize: 248_320, layerCount: 4), seed: 7), tokenizer: tokenizer)
        let budget = NFKMLXResidencyBudget.current()
        let plain = try NFKMLXHybridLanguage.backend(directoryURL: target, precision: .checkpoint, residency: .automatic,
                                                     budget: budget)
        let drafted = try NFKMLXHybridLanguage.backend(directoryURL: target, precision: .checkpoint, residency: .automatic,
                                                       budget: budget, draftDirectoryURL: draft, draftBits: 4)
        XCTAssertFalse(plain.hasDraftModel)
        XCTAssertTrue(drafted.hasDraftModel)
        XCTAssertEqual(drafted.streamedLayerCount, 0)
        XCTAssertEqual(try reply(drafted, draftTokens: 3), try reply(plain), "the drafted greedy reply")
        XCTAssertGreaterThan(drafted.lastSpeculativeReport.rounds, 0)
    }

    func testTheDraftedFactoryRefusesADraftOfAnotherVocabulary() throws {
        try requireMLXRuntime()
        let tokenizer = try tokenizerDirectory()
        let target = try release(randomNet(configuration(vocabularySize: 248_320, layerCount: 4), seed: 8), tokenizer: tokenizer)
        let draft = try release(randomNet(configuration(vocabularySize: 128, layerCount: 4), seed: 9), tokenizer: tokenizer)
        XCTAssertThrowsError(try NFKMLXHybridLanguage.backend(directoryURL: target, draftDirectoryURL: draft)) { error in
            guard case NFKMLXError.unsupportedConfiguration(let message) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertTrue(message.contains("share the model's vocabulary"), message)
        }
    }

    func testTheCatalogDraftsEachQwen3_5ReleaseWithTheSmallerOnes() throws {
        NFKMLXReferenceModels.registerAll()
        func choices(_ identifier: String) throws -> [String] {
            let descriptor = try XCTUnwrap(NFKMLXModelDescriptors.modelDescriptors().first { $0.identifier == identifier }, identifier)
            return descriptor.option(withKey: NFKModelLoadOptionDraftModel)?.choices.compactMap { $0.value as? String } ?? []
        }
        XCTAssertEqual(try choices("qwen3.5-2b"), [])
        XCTAssertEqual(try choices(NFKMLXHybridLanguage.modelName), ["qwen3.5-2b"])
        XCTAssertEqual(try choices("qwen3.5-9b"), ["qwen3.5-2b", NFKMLXHybridLanguage.modelName])
        XCTAssertEqual(try choices("qwen3.8-27b"), ["qwen3.5-2b", NFKMLXHybridLanguage.modelName, "qwen3.5-9b"])
        let names = Set(NFKMLXModelRegistry.registeredModelNames)
        for name in ["qwen3.5-2b", NFKMLXHybridLanguage.modelName, "qwen3.5-9b", "qwen3.8-27b"] {
            XCTAssertTrue(names.contains(name), name)
            XCTAssertThrowsError(try NFKMLXModelRegistry.backend(named: name, weightsURL: nil), name)
        }
    }

    // MARK: Released weights

    func testQwen3_5_9BDraftedByQwen3_5_2BKeepsItsGreedyReply() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let targetPath = environment["IK_VAL_QWEN3_5_9B"], let draftPath = environment["IK_VAL_QWEN3_5_2B"] else {
            throw XCTSkip("set IK_VAL_QWEN3_5_9B and IK_VAL_QWEN3_5_2B")
        }
        let target = URL(fileURLWithPath: targetPath), draft = URL(fileURLWithPath: draftPath)
        // The budget that holds the 9B's first half of its layers, so the drafted run streams the rest, as on
        // a machine smaller than this one, and its peak leaves the machine room.
        let stage = try NFKMLXLayerStream.footprint(
            directory: target, precision: .checkpoint,
            converting: try NFKMLXHybridLanguage.releasedType(inDirectory: target),
            includes: { NFKMLXHybridLanguage.moduleKey(forReference: $0) != nil },
            layerOf: { NFKMLXHybridLanguage.layer(ofKey: $0)?.layer })
        let budget = NFKMLXResidencyBudget.leastBudget(
            holding: stage.streamedMinimumBytes + stage.layerBytes.prefix(stage.layerBytes.count / 2).reduce(0, +))
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "The capital of France is"],
                                          parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 16])
        let expected = try autoreleasepool { () -> String in
            let plain = try NFKMLXHybridLanguage.backend(directoryURL: target, precision: .checkpoint, residency: .streamed,
                                                         budget: budget)
            XCTAssertGreaterThan(plain.streamedLayerCount, 0)
            return try XCTUnwrap(try plain.runInference(for: request).output(forKey: NFKOutputText) as? String)
        }
        NFKMLXGPU.clearCache()
        let drafted = try NFKMLXHybridLanguage.backend(directoryURL: target, precision: .checkpoint, residency: .streamed,
                                                       budget: budget, identifier: "qwen3.5-9b",
                                                       draftDirectoryURL: draft, draftBits: 4)
        XCTAssertGreaterThan(drafted.streamedLayerCount, 0, "the draft's bytes leave the 9B less to hold")
        let started = Date()
        let reply = try drafted.runInference(for: request).output(forKey: NFKOutputText) as? String
        let report = drafted.lastSpeculativeReport
        print("VALIDATION qwen3.5-9b drafted by qwen3.5-2b at 4 bits: \(report.rounds) rounds, "
              + "\(report.accepted) of \(report.proposed) proposals kept, \(drafted.streamedLayerCount) of "
              + "\(stage.layerBytes.count) layers streamed, \(Int(Date().timeIntervalSince(started))) s; reply \(reply ?? "nil")")
        XCTAssertEqual(reply, expected, "the drafted greedy reply")
        XCTAssertGreaterThan(report.accepted, 0, "the draft's proposals are kept")
    }
}
