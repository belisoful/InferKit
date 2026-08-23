//
//  NFKMLXLanguageModelTests.swift
//  InferKitMLXTests
//
//  The decoder's contract: shapes, the causal mask, the key-value cache, and sampling. Numeric
//  agreement with a released model is measured separately against real weights — what these cover is
//  everything that must hold before that comparison means anything.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXLanguageModelTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func tinyNet() -> NFKMLXLanguageNet { NFKMLXLanguage.makeNet(.tiny) }

    // MARK: Shapes and the stack

    func testTheModelProducesLogitsForEveryPosition() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let tokens = MLXArray((0 ..< 6).map { Int32($0) }).reshaped([1, 6])
        let logits = net(tokens)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 6, NFKMLXLanguageConfiguration.tiny.vocabularySize])
    }

    // A tied model has no `lm_head`: the embedding matrix is the output projection. Building one
    // anyway would leave it randomly initialized while a strict load still passed.
    func testTiedEmbeddingsCarryNoOutputProjection() {
        var tied = NFKMLXLanguageConfiguration.tiny
        tied.tiesWordEmbeddings = true
        XCTAssertNil(NFKMLXLanguage.makeNet(tied).lmHead)

        var untied = NFKMLXLanguageConfiguration.tiny
        untied.tiesWordEmbeddings = false
        XCTAssertNotNil(NFKMLXLanguage.makeNet(untied).lmHead)
    }

    // Query and key normalization is Qwen3's; a Llama-shaped config carries no such weights, and
    // building them would make a strict load of a Llama checkpoint fail.
    func testQueryKeyNormalizationFollowsTheConfiguration() {
        var qwen = NFKMLXLanguageConfiguration.tiny
        qwen.normalizesQueryAndKey = true
        let withNorm = NFKMLXLanguage.makeNet(qwen).parameters().flattened().map(\.0)
        XCTAssertTrue(withNorm.contains { $0.hasSuffix("q_norm.weight") })

        var llama = NFKMLXLanguageConfiguration.tiny
        llama.normalizesQueryAndKey = false
        let without = NFKMLXLanguage.makeNet(llama).parameters().flattened().map(\.0)
        XCTAssertFalse(without.contains { $0.hasSuffix("q_norm.weight") })
    }

    // The module keys ARE the released checkpoint's names, which is what lets a release load with no
    // remapping. A rename here would break every checkpoint silently.
    func testParameterNamesMatchTheReleasedLayout() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["model.embed_tokens.weight", "model.norm.weight",
                         "model.layers.0.self_attn.q_proj.weight",
                         "model.layers.0.self_attn.k_proj.weight",
                         "model.layers.0.self_attn.v_proj.weight",
                         "model.layers.0.self_attn.o_proj.weight",
                         "model.layers.0.self_attn.q_norm.weight",
                         "model.layers.0.mlp.gate_proj.weight",
                         "model.layers.0.mlp.up_proj.weight",
                         "model.layers.0.mlp.down_proj.weight",
                         "model.layers.0.input_layernorm.weight",
                         "model.layers.0.post_attention_layernorm.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // MARK: The causal mask

    /// A `.checkpoint`-precision load of a bf16 release makes the whole module bf16, and the fused
    /// attention refuses a float32 mask against a bf16 module. Invisible at float32, which is how it
    /// shipped: the mask now takes the queries' dtype, and this holds that open.
    func testABF16ModuleStillForwards() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let halved = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map {
            ($0.0, $0.1.asType(.bfloat16))
        })
        net.update(parameters: ModuleParameters.unflattened(halved))
        let logits = net(MLXArray([Int32(1), 2, 3]).reshaped([1, 3]))
        eval(logits)
        XCTAssertEqual(logits.dtype, .bfloat16)
        XCTAssertTrue(logits.asType(.float32).sum().item(Float.self).isFinite)
    }

    func testTheCausalMaskForbidsAttendingForward() {
        let mask = NFKMLXLanguageNet.causalMask(4, offset: 0)
        eval(mask)
        let values = mask.asArray(Float.self)
        for row in 0 ..< 4 {
            for column in 0 ..< 4 {
                let value = values[row * 4 + column]
                if column <= row {
                    XCTAssertEqual(value, 0, "position \(row) may attend to \(column)")
                } else {
                    XCTAssertLessThan(value, -1e8, "position \(row) may not attend to \(column)")
                }
            }
        }
    }

    // With a cache, a step's row covers the whole prefix, so the mask widens by the offset.
    func testTheMaskWidensWithTheCacheOffset() {
        let mask = NFKMLXLanguageNet.causalMask(2, offset: 3)
        XCTAssertEqual(mask.shape, [2, 5])
        eval(mask)
        let values = mask.asArray(Float.self)
        XCTAssertEqual(values[0 * 5 + 3], 0, "the first new position sees itself")
        XCTAssertLessThan(values[0 * 5 + 4], -1e8, "but not the one after it")
    }

    // MARK: The key-value cache

    // The cache is what makes generation linear rather than quadratic, and it is only correct if a
    // cached step produces the same logits as recomputing the whole prefix. This is the assertion the
    // whole generation path rests on.
    func testACachedStepMatchesRecomputingThePrefix() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let prompt = [3, 17, 42, 8, 91]

        // Whole sequence at once.
        let full = net(MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count]))
        eval(full)
        let expected = full[0, -1].asArray(Float.self)

        // The same sequence one token at a time through the cache.
        let cache = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount)
        var stepped: [Float] = []
        for token in prompt {
            let logits = net(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)
            eval(logits)
            stepped = logits[0, -1].asArray(Float.self)
        }

        XCTAssertEqual(cache.offset, prompt.count, "the cache holds every position")
        XCTAssertEqual(stepped.count, expected.count)
        let worst = zip(stepped, expected).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(worst, 2e-3, "cached decoding agrees with a full forward pass")
    }

    // Prefill then step: the shape generation actually uses.
    func testAPrefillFollowedByAStepAdvancesTheCache() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let cache = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount)
        _ = net(MLXArray([Int32(1), 2, 3]).reshaped([1, 3]), cache: cache)
        XCTAssertEqual(cache.offset, 3)
        let logits = net(MLXArray([Int32(4)]).reshaped([1, 1]), cache: cache)
        eval(logits)
        XCTAssertEqual(cache.offset, 4)
        XCTAssertEqual(logits.shape, [1, 1, net.configuration.vocabularySize])
    }

    // MARK: Sampling

    func testGreedySamplingIsDeterministicAndTakesTheArgmax() throws {
        try requireMLXRuntime()
        var logits = [Float](repeating: 0, count: 8)
        logits[5] = 10
        let array = MLXArray(logits)
        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        XCTAssertEqual(NFKMLXLanguageNet.sample(array, options: options), 5)
        XCTAssertEqual(NFKMLXLanguageNet.sample(array, options: options), 5)
    }

    func testASeedMakesSampledGenerationRepeatable() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        var options = NFKMLXGenerationOptions()
        options.temperature = 0.8
        options.maxTokens = 12
        options.seed = 1234

        let first = net.generate(prompt: [1, 2, 3], options: options)
        let again = net.generate(prompt: [1, 2, 3], options: options)
        XCTAssertEqual(first, again, "the same seed produces the same tokens")
        XCTAssertEqual(first.count, 12)
    }

    func testGenerationStopsAtAStopToken() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = 20

        let unbounded = net.generate(prompt: [7], options: options)
        XCTAssertFalse(unbounded.isEmpty)

        // Stopping on the token greedy decoding produces first must end the run immediately.
        options.stopTokens = [unbounded[0]]
        XCTAssertTrue(net.generate(prompt: [7], options: options).isEmpty)
    }

    func testAHandlerCanStopGeneration() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 50
        var seen = 0
        let produced = net.generate(prompt: [5], options: options) { _ in
            seen += 1
            return seen < 3                       // this is how a job's cancellation reaches generation
        }
        XCTAssertEqual(produced.count, 3)
    }

    // MARK: Configuration parsing

    func testAHybridOrExpertConfigurationIsRejected() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Qwen3.5 and 3.6 interleave linear attention with full attention; loading them into a dense
        // stack would produce confident nonsense rather than an error.
        let hybrid = directory.appendingPathComponent("hybrid.json")
        try #"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}"#
            .write(to: hybrid, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try NFKMLXLanguage.configuration(fromHuggingFace: hybrid))

        let experts = directory.appendingPathComponent("moe.json")
        try #"{"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","num_experts":128}"#
            .write(to: experts, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try NFKMLXLanguage.configuration(fromHuggingFace: experts))
    }

    func testADenseConfigurationIsRead() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("config.json")
        try #"""
        {"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":1024,
         "num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
         "intermediate_size":3072,"vocab_size":151936,"rope_theta":1000000.0,
         "rms_norm_eps":1e-06,"tie_word_embeddings":true}
        """#.write(to: url, atomically: true, encoding: .utf8)

        let configuration = try NFKMLXLanguage.configuration(fromHuggingFace: url)
        XCTAssertEqual(configuration.hiddenSize, 1024)
        XCTAssertEqual(configuration.layerCount, 28)
        XCTAssertEqual(configuration.keyValueHeadCount, 8, "grouped-query attention is read")
        XCTAssertEqual(configuration.headDimensions, 128)
        XCTAssertTrue(configuration.tiesWordEmbeddings)
        XCTAssertTrue(configuration.normalizesQueryAndKey, "qwen3 normalizes queries and keys")
    }

    // MARK: The backend contract

    func testTheBackendReportsAMissingTokenizer() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXLanguage.backend(weightsURL: nil, tokenizer: nil, configuration: .tiny)
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "hello"])))
    }

    func testTheBackendFlattensAMessageList() {
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: [
            ["role": "system", "content": "be brief"],
            ["role": "user", "content": "hello"],
        ]])
        XCTAssertEqual(NFKMLXLanguageBackend.prompt(from: request), "be brief\nhello")
    }
}
