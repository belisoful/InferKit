//
//  NFKMLXHybridStreamingTests.swift
//  InferKitMLXTests
//
//  Streaming the Qwen3.5 hybrid decoder's layers. A random decoder of both layer kinds is written out as a
//  sharded release in the released layout (`model.language_model.*`, the convolution stored `[C, 1, K]`,
//  `A_log` and the recurrence's norm stored float32 beside bfloat16, a vision and a multi-token prediction
//  tensor beside them), and the same release loaded with every layer held and with its layers streamed
//  must compute the same values, element for element. On released weights, Qwen3.5 4B streams to its held
//  values, Qwen3.5 9B streamed at float32 and Qwen3.8 27B streamed at bfloat16 match
//  `run_reference.py dense_layerwise` (`IK_PARITY_QWEN3_5_9B_LAYERWISE`, `IK_PARITY_QWEN3_8_27B`).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXHybridStreamingTests: XCTestCase {

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

    /// Eight layers, every fourth attending fully.
    private let configuration = NFKMLXHybridConfiguration(
        hiddenSize: 64, layerCount: 8, intermediateSize: 128, vocabularySize: 256, headCount: 4, keyValueHeadCount: 2,
        headDimensions: 16, ropeTheta: 10_000, partialRotaryFactor: 0.25, gatesAttentionOutput: true,
        linearKeyHeadCount: 2, linearKeyHeadDimensions: 8, linearValueHeadCount: 4, linearValueHeadDimensions: 8,
        linearConvolutionKernel: 4, fullAttentionInterval: 4)

    private let tokens: [Int32] = [3, 17, 42, 99, 7, 61, 12, 5, 88]

    /// A random decoder written as a two-shard release in the released layout.
    private func release() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("hybrid-stream-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        MLXRandom.seed(13)
        let net = NFKMLXHybridLanguage.makeNet(configuration)
        var stored = net.parameters().flattened().map { key, value -> (String, MLXArray) in
            let varied = key.hasSuffix("norm.weight") ? 1 + MLXRandom.normal(value.shape) * 0.1 : value
            let name = key.hasPrefix("model.") ? "model.language_model." + key.dropFirst("model.".count) : key
            let shaped = key.hasSuffix("conv1d.weight") ? varied.transposed(0, 2, 1) : varied
            let keepsFloat32 = key.hasSuffix("A_log") || key.hasSuffix("linear_attn.norm.weight")
            return (name, keepsFloat32 ? shaped.asType(.float32) : shaped.asType(.bfloat16))
        }
        stored.append(("model.visual.patch_embed.proj.weight", MLXRandom.normal([8, 8]).asType(.bfloat16)))
        stored.append(("mtp.layers.0.input_layernorm.weight", MLXArray.ones([64]).asType(.bfloat16)))
        stored.sort { $0.0 < $1.0 }
        var weightMap = [String: String]()
        for (index, half) in [stored.prefix(stored.count / 2), stored.dropFirst(stored.count / 2)].enumerated() {
            let name = "model-0000\(index + 1)-of-00002.safetensors"
            try save(arrays: Dictionary(half.map { $0 }, uniquingKeysWith: { first, _ in first }),
                     url: directory.appendingPathComponent(name))
            half.forEach { weightMap[$0.0] = name }
        }
        try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    /// The least budget that holds `directory`'s decoder with only its first `held` layers.
    private func budget(_ directory: URL, precision: NFKMLXWeightPrecision, holding held: Int) throws -> Int {
        let stage = try NFKMLXLayerStream.footprint(
            directory: directory, precision: precision,
            converting: precision == .checkpoint ? try NFKMLXHybridLanguage.releasedType(inDirectory: directory) : nil,
            includes: { NFKMLXHybridLanguage.moduleKey(forReference: $0) != nil },
            layerOf: { NFKMLXHybridLanguage.layer(ofKey: $0)?.layer })
        return NFKMLXResidencyBudget.leastBudget(holding: stage.streamedMinimumBytes + stage.layerBytes.prefix(held).reduce(0, +))
    }

    private func net(_ directory: URL, configuration: NFKMLXHybridConfiguration? = nil, precision: NFKMLXWeightPrecision,
                     residency: NFKMLXResidency, holding held: Int? = nil) throws -> NFKMLXHybridLanguageNet {
        let net = NFKMLXHybridLanguage.makeNet(configuration ?? self.configuration)
        try NFKMLXHybridLanguage.loadWeights(
            into: net, fromDirectory: directory, precision: precision, residency: residency,
            budget: try held.map { try budget(directory, precision: precision, holding: $0) } ?? NFKMLXResidencyBudget.current())
        return net
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private var input: MLXArray { MLXArray(tokens).reshaped([1, tokens.count]) }

    func testAStreamedHybridDecoderComputesWhatAHeldOneDoes() throws {
        try requireMLXRuntime()
        let directory = try release()
        for precision in [NFKMLXWeightPrecision.float32, .checkpoint] {
            let label = "\(precision)"
            let held = try net(directory, precision: precision, residency: .resident)
            let streamed = try net(directory, precision: precision, residency: .streamed, holding: 3)
            XCTAssertNil(held.layerStream, label)
            XCTAssertEqual(streamed.layerStream?.layers, [3, 4, 5, 6, 7], label)
            let expected = held.hiddenStates(input)
            for pass in 0 ..< 2 {
                let states = streamed.hiddenStates(input)
                XCTAssertEqual(states.count, expected.count, label)
                for (index, state) in states.enumerated() {
                    XCTAssertEqual(maxDifference(state, expected[index]), 0, "\(label) pass \(pass) state \(index)")
                }
                XCTAssertEqual(maxDifference(streamed(input), held(input)), 0, "\(label) pass \(pass) logits")
                for index in try XCTUnwrap(streamed.layerStream).layers {
                    XCTAssertTrue(streamed.model.layers[index].parameters().flattened().allSatisfy { $0.1.size == 0 },
                                  "\(label): layer \(index) holds nothing between its turns")
                }
            }
            try streamed.verifyStream()
        }
    }

    func testAnAutomaticLoadHoldsAReleaseThatFitsAsAWholeLoadDoes() throws {
        try requireMLXRuntime()
        let directory = try release()
        let automatic = try net(directory, precision: .checkpoint, residency: .automatic)
        XCTAssertNil(automatic.layerStream)
        let whole = NFKMLXHybridLanguage.makeNet(configuration)
        try NFKMLXHybridLanguage.loadWeights(into: whole, fromDirectory: directory, precision: .checkpoint)
        XCTAssertEqual(maxDifference(automatic(input), whole(input)), 0)
    }

    private func reply(_ backend: NFKMLXDecoderBackend) throws -> String? {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "abc"],
                                          parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 6])
        return try backend.runInference(for: request).output(forKey: NFKOutputText) as? String
    }

    private func backend(_ net: NFKMLXHybridLanguageNet) -> NFKMLXDecoderBackend {
        NFKMLXDecoderBackend(logits: { net($0) }, tokenizer: NFKMLXScalarTestTokenizer(vocabularySize: 256),
                             identifier: "qwen3.5", chatTemplate: nil, stopTokens: [], modules: [net],
                             layerStream: net.layerStream)
    }

    func testAFailedReadFailsTheRequestAndTheNextRequestReadsTheReleaseAgain() throws {
        try requireMLXRuntime()
        let directory = try release()
        let held = backend(try net(directory, precision: .checkpoint, residency: .resident))
        let streamedNet = try net(directory, precision: .checkpoint, residency: .streamed, holding: 2)
        let streamed = backend(streamedNet)
        XCTAssertEqual(held.streamedLayerCount, 0)
        XCTAssertNil(held.modelInfo[NFKModelInfoResidency])
        XCTAssertEqual(streamed.streamedLayerCount, 6)
        XCTAssertEqual(streamed.streamedBytesPerPass, streamedNet.layerStream?.bytesPerPass)
        XCTAssertEqual(streamed.modelInfo[NFKModelInfoResidency] as? String, NFKModelResidencyStreamed)
        XCTAssertEqual(try reply(streamed), try reply(held), "the streamed reply")

        let shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            .map { directory.appendingPathComponent($0) }
        let originals = try shards.map { try Data(contentsOf: $0) }
        for file in shards {
            let start = try XCTUnwrap(NFKMLXSafetensors.entries(inFile: file).values.map(\.start).min())
            let handle = try FileHandle(forWritingTo: file)
            try handle.truncate(atOffset: UInt64(start))
            try handle.close()
        }
        XCTAssertThrowsError(try reply(streamed), "the truncated release cannot be read")
        for (file, original) in zip(shards, originals) {
            try original.write(to: file)
        }
        XCTAssertEqual(try reply(streamed), try reply(held), "the next request reads every layer again")
        try streamedNet.verifyStream()
    }

    // MARK: Released weights

    private func environmentURL(_ key: String) throws -> URL {
        guard let path = NFKMLXValidationConfig.environment[key] else {
            throw XCTSkip("set \(key)")
        }
        return URL(fileURLWithPath: path)
    }

    func testAReleasedQwen3_5StreamsToTheValuesItsHeldLoadComputes() throws {
        try requireMLXRuntime()
        let directory = try environmentURL("IK_VAL_QWEN3_5")
        let configuration = try NFKMLXHybridLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let prompt = MLXArray([Int32(760), 6511, 314, 9338, 369]).reshaped([1, 5])
        func logits(_ net: NFKMLXHybridLanguageNet) throws -> MLXArray {
            let value = net(prompt)
            eval(value)
            try net.verifyStream()
            return value
        }
        let held = try autoreleasepool {
            try logits(try net(directory, configuration: configuration, precision: .checkpoint, residency: .resident))
        }
        NFKMLXGPU.clearCache()
        let streamedNet = try net(directory, configuration: configuration, precision: .checkpoint, residency: .streamed,
                                  holding: configuration.layerCount / 2)
        XCTAssertEqual(streamedNet.layerStream?.layers.count, configuration.layerCount - configuration.layerCount / 2)
        XCTAssertEqual(maxDifference(try logits(streamedNet), held), 0)
    }

    /// The release `releaseKey` names, loaded under ``NFKMLXResidency/automatic`` at `precision`, against the
    /// `run_reference.py dense_layerwise` record `recordKey` names, in one pass over the prompt and the
    /// record's greedy continuation: the last prompt position's state after every layer but the last, the
    /// prompt's logits and argmax, and the argmax at every position that predicts a continuation token. A bf16
    /// run against a bf16 record may differ at `argmaxSlack` prompt positions, as the dense decoders' may.
    private func checkRelease(_ name: String, release releaseKey: String, record recordKey: String,
                              precision: NFKMLXWeightPrecision, bar: Double, argmaxSlack: Int) throws {
        try requireMLXRuntime()
        let directory = try environmentURL(releaseKey)
        guard let recordPath = NFKMLXValidationConfig.environment[recordKey] else {
            throw XCTSkip("set \(recordKey) (run_reference.py dense_layerwise)")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self)
        let continuation = try XCTUnwrap(record["continuation"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(record["output"])
        let configuration = try NFKMLXHybridLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXHybridLanguage.makeNet(configuration)
        try NFKMLXHybridLanguage.loadWeights(into: net, fromDirectory: directory, precision: precision, residency: .automatic)
        let stream = try XCTUnwrap(net.layerStream, "the release does not fit whole, so its layers stream")

        let sequence = tokens + continuation
        let started = Date()
        let states = net.hiddenStates(MLXArray(sequence).reshaped([1, sequence.count]))
        let normalized = states[states.count - 1]
        let logits = (net.lmHead?(normalized) ?? net.model.embedTokens.asLinear(normalized))[0]
        eval(logits)
        try net.verifyStream()
        let seconds = Date().timeIntervalSince(started)

        var worst = (cosine: 1.0, layer: -1)
        for index in 0 ..< configuration.layerCount {
            let expected = try XCTUnwrap(record["hidden_last.\(index)"], "the record's state entering layer \(index)")
            let measured = NFKMLXWav2Vec2Tests.cosine(states[index][0, tokens.count - 1], expected)
            if measured < worst.cosine { worst = (measured, index) }
        }
        let promptLogits = logits[0 ..< tokens.count]
        let logitCosine = NFKMLXWav2Vec2Tests.cosine(promptLogits, referenceLogits)
        let agree = zip(promptLogits.argMax(axis: -1).asArray(Int32.self), referenceLogits.argMax(axis: -1).asArray(Int32.self))
            .filter { $0 == $1 }.count
        let predicted = logits[(tokens.count - 1) ..< (sequence.count - 1)].argMax(axis: -1).asArray(Int32.self)
        let (bytes, readSeconds) = stream.readStatistics
        print("VALIDATION PARITY \(name) streamed: \(stream.layers.count) of \(configuration.layerCount) layers, "
              + "\(stream.bytesPerPass) bytes per pass, pass \(Int(seconds)) s, read \(bytes) bytes in \(readSeconds) s; "
              + "worst state \(worst.cosine) (entering layer \(worst.layer)), logits \(logitCosine), "
              + "argmax \(agree)/\(tokens.count), greedy \(zip(predicted, continuation).filter { $0 == $1 }.count)/\(continuation.count)")
        XCTAssertGreaterThan(worst.cosine, bar, "the state entering layer \(worst.layer)")
        XCTAssertGreaterThan(logitCosine, bar, "the prompt's logits")
        XCTAssertGreaterThanOrEqual(agree, tokens.count - argmaxSlack, "the prompt's argmax")
        XCTAssertEqual(predicted, continuation, "each continuation token is the argmax after the ones before it")
    }

    func testTheStreamedQwen3_5_9BMatchesItsReferenceAtFloat32() throws {
        try checkRelease("qwen3.5-9b float32", release: "IK_VAL_QWEN3_5_9B", record: "IK_PARITY_QWEN3_5_9B_LAYERWISE",
                         precision: .float32, bar: 0.99999, argmaxSlack: 0)
    }

    func testTheStreamedQwen3_8_27BMatchesItsReference() throws {
        try checkRelease("qwen3.8-27b bf16", release: "IK_VAL_QWEN3_8_27B", record: "IK_PARITY_QWEN3_8_27B",
                         precision: .checkpoint, bar: 0.999, argmaxSlack: 1)
    }
}
