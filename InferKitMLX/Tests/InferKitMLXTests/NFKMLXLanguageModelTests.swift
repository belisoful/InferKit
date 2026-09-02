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

    // The quantized cache stores the same positions in less memory; its dequantized read tracks the
    // full-precision logits closely rather than exactly, which is the one place a tolerance — not the
    // float path's near-equality — is the right claim. The head dimension (64) is divisible by the
    // group size, which the packing requires.
    func testAQuantizedCacheTracksTheFullPrecisionLogits() throws {
        try requireMLXRuntime()
        let config = NFKMLXLanguageConfiguration(hiddenSize: 128, layerCount: 2, headCount: 2,
                                                 keyValueHeadCount: 1, headDimensions: 64,
                                                 intermediateSize: 128, vocabularySize: 512, ropeTheta: 10_000)
        let net = NFKMLXLanguage.makeNet(config)
        let prompt = [3, 17, 42, 8, 91]

        let full = net(MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count]))
        eval(full)
        let expected = full[0, -1].asArray(Float.self)

        let cache = NFKMLXKeyValueCache(layerCount: config.layerCount,
                                        quantization: .init(bits: 8, groupSize: 64))
        var stepped: [Float] = []
        for token in prompt {
            let logits = net(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)
            eval(logits)
            stepped = logits[0, -1].asArray(Float.self)
        }

        XCTAssertEqual(cache.offset, prompt.count, "the quantized cache counts positions the same way")
        XCTAssertEqual(stepped.count, expected.count)
        let dot = zip(stepped, expected).map(*).reduce(0, +)
        let steppedNorm = sqrt(stepped.map { $0 * $0 }.reduce(0, +))
        let expectedNorm = sqrt(expected.map { $0 * $0 }.reduce(0, +))
        XCTAssertGreaterThan(dot / (steppedNorm * expectedNorm), 0.99,
                             "8-bit KV cache decoding tracks the full-precision logits")
    }

    // A bounded quantized cache drops the oldest positions exactly as the float one does.
    func testAQuantizedWindowedCacheAgreesWithTheFloatWindow() throws {
        try requireMLXRuntime()
        let config = NFKMLXLanguageConfiguration(hiddenSize: 128, layerCount: 2, headCount: 2,
                                                 keyValueHeadCount: 1, headDimensions: 64,
                                                 intermediateSize: 128, vocabularySize: 512, ropeTheta: 10_000)
        let net = NFKMLXLanguage.makeNet(config)
        let prompt = (0 ..< 20).map { Int32(($0 * 7 + 3) % 512) }

        func decode(_ cache: NFKMLXKeyValueCache) -> [Float] {
            var last: [Float] = []
            for token in prompt {
                let logits = net(MLXArray([token]).reshaped([1, 1]), cache: cache)
                eval(logits)
                last = logits[0, -1].asArray(Float.self)
            }
            return last
        }
        let float = decode(NFKMLXKeyValueCache(layerCount: config.layerCount, window: 8))
        let quantized = decode(NFKMLXKeyValueCache(layerCount: config.layerCount, window: 8,
                                                   quantization: .init(bits: 8, groupSize: 64)))
        let dot = zip(quantized, float).map(*).reduce(0, +)
        let qn = sqrt(quantized.map { $0 * $0 }.reduce(0, +))
        let fn = sqrt(float.map { $0 * $0 }.reduce(0, +))
        XCTAssertGreaterThan(dot / (qn * fn), 0.99, "the quantized window tracks the float window")
    }

    // The ChatML template renders roles and turn markers and opens an assistant turn, which is the
    // format an instruct release is trained on. The default flattens contents, unchanged.
    func testChatMLTemplateRendersRolesAndTurnMarkers() {
        let messages: [[AnyHashable: Any]] = [
            ["role": "system", "content": "You are terse."],
            ["role": "user", "content": "Hi"],
        ]
        XCTAssertEqual(NFKMLXLanguageBackend.chatMLPrompt(from: messages),
                       "<|im_start|>system\nYou are terse.<|im_end|>\n"
                       + "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
        XCTAssertEqual(NFKMLXLanguageBackend.prompt(from:
            NFKInferenceRequest(inputs: [NFKInputMessages: messages]), template: .none),
                       "You are terse.\nHi")
    }

    // Chunked prefill is exact: feeding the prompt in slices through the cache lands the same logits
    // as one pass, because each chunk attends to the same prefix. Only the peak memory differs.
    func testChunkedPrefillMatchesASinglePass() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let prompt = [3, 17, 42, 8, 91, 5, 60, 22]

        let whole = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount)
        let single = net.prefill(prompt, cache: whole, chunkSize: nil)
        eval(single)
        let expected = single[0, -1].asArray(Float.self)

        let chunked = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount)
        let stepped = net.prefill(prompt, cache: chunked, chunkSize: 3)
        eval(stepped)
        let actual = stepped[0, -1].asArray(Float.self)

        XCTAssertEqual(chunked.offset, prompt.count, "chunked prefill fills the whole cache")
        let worst = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(worst, 2e-3, "chunked prefill agrees with a single pass")
    }

    // The declared-facts check: a checkpoint whose shape disagrees with a config-built module is
    // rejected when verifyShapes is on, and adopted silently when it is off (the default, which a
    // placeholder-width model relies on).
    func testVerifyShapesRejectsAShapeTheConfigDoesNotExpect() throws {
        try requireMLXRuntime()
        var mapped = tinyNet().parameters().flattened().map { ($0.0, $0.1) }
        mapped[0] = (mapped[0].0, MLXArray.zeros(mapped[0].1.shape + [2]))   // one wrong shape

        // Off (the default): MLX adopts the shape, no error — what a placeholder-width model needs.
        XCTAssertNoThrow(try NFKMLXWeights.apply(mapped, to: tinyNet(), verifyShapes: false))

        // On: refused, naming the parameter and both shapes.
        XCTAssertThrowsError(try NFKMLXWeights.apply(mapped, to: tinyNet(), verifyShapes: true)) { error in
            guard case NFKMLXError.weightsMismatch(let detail) = error else {
                return XCTFail("expected weightsMismatch, got \(error)")
            }
            XCTAssertTrue(detail.contains("shape") && detail.contains(mapped[0].0),
                          "expected the mismatch to name the parameter: \(detail)")
        }
    }

    // Every MLX generation option is reachable as a request parameter, which is how an Objective-C
    // caller sets them — parity with the Swift options struct.
    func testMLXGenerationParametersOverrideTheOptions() {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"], parameters: [
            NFKMLXGenerationParameterKey.contextWindow: 128,
            NFKMLXGenerationParameterKey.prefillChunkSize: 64,
            NFKMLXGenerationParameterKey.cacheQuantizationBits: 8,
            NFKMLXGenerationParameterKey.cacheQuantizationGroupSize: 64,
            NFKMLXGenerationParameterKey.chatTemplate: "chatml",
        ])
        var options = NFKMLXGenerationOptions()
        NFKMLXLanguageBackend.applyMLXParameters(from: request, to: &options)
        XCTAssertEqual(options.contextWindow, 128)
        XCTAssertEqual(options.prefillChunkSize, 64)
        XCTAssertEqual(options.cacheQuantization, .init(bits: 8, groupSize: 64))
        if case .chatML = options.chatTemplate {} else { XCTFail("the chatml parameter selects the template") }

        // Absent parameters leave the defaults, so an existing caller is unchanged.
        var untouched = NFKMLXGenerationOptions()
        NFKMLXLanguageBackend.applyMLXParameters(
            from: NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"]), to: &untouched)
        XCTAssertNil(untouched.contextWindow)
        XCTAssertNil(untouched.cacheQuantization)
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

    // MARK: A bounded cache

    // The window is the whole point: memory stops growing with the conversation.
    func testAWindowedCacheStopsGrowing() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let cache = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount, window: 8)
        for token in 0 ..< 40 {
            _ = net(MLXArray([Int32(token % 100)]).reshaped([1, 1]), cache: cache)
        }
        XCTAssertEqual(cache.offset, 40, "the position count is absolute and keeps counting")
        XCTAssertLessThanOrEqual(cache.retainedLength(), 8, "but the rows retained are bounded")
        XCTAssertGreaterThan(cache.retainedLength(), 1)
    }

    // An unbounded cache retains everything, which is what makes the bound a deliberate choice.
    func testAnUnboundedCacheRetainsEveryPosition() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let cache = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount)
        for token in 0 ..< 12 {
            _ = net(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)
        }
        XCTAssertEqual(cache.offset, 12)
        XCTAssertEqual(cache.retainedLength(), 12)
    }

    // The exactness condition. While the sequence fits inside the window, a bounded cache has dropped
    // nothing, so it must agree with an unbounded one to the last bit. Past the window it diverges,
    // and that divergence is the approximation being bought — not a defect.
    func testABoundedCacheIsExactUntilItStartsDropping() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let prompt: [Int32] = [3, 17, 42, 8, 91, 12, 55, 7]
        let window = 6

        let unbounded = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount)
        let bounded = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount, window: window)

        for (index, token) in prompt.enumerated() {
            let a = net(MLXArray([token]).reshaped([1, 1]), cache: unbounded)
            let b = net(MLXArray([token]).reshaped([1, 1]), cache: bounded)
            eval(a, b)
            let worst = zip(a[0, -1].asArray(Float.self), b[0, -1].asArray(Float.self))
                .map { abs($0 - $1) }.max() ?? 0
            if index < window {
                XCTAssertLessThan(worst, 1e-5,
                                  "step \(index) is inside the window, so nothing has been dropped")
            }
        }
        XCTAssertEqual(unbounded.retainedLength(), prompt.count)
        XCTAssertLessThanOrEqual(bounded.retainedLength(), window)
    }

    // A prefill longer than one token must still agree, which is where the mask width matters: it is
    // built against the rows the pass will actually see, not against the absolute position count.
    func testAPrefillIntoABoundedCacheAgreesWithSteppingIntoIt() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let prompt: [Int32] = [5, 9, 11, 2]

        let stepped = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount, window: 16)
        var last = MLXArray.zeros([1])
        for token in prompt {
            last = net(MLXArray([token]).reshaped([1, 1]), cache: stepped)[0, -1]
        }
        let prefilled = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount, window: 16)
        let whole = net(MLXArray(prompt).reshaped([1, prompt.count]), cache: prefilled)[0, -1]
        eval(last, whole)

        XCTAssertEqual(stepped.offset, prefilled.offset)
        let worst = zip(last.asArray(Float.self), whole.asArray(Float.self))
            .map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(worst, 2e-3, "a prefill and a walk reach the same state")
    }

    // A prefill that follows a trim is the case the mask width was wrong for before: the cache holds
    // fewer rows than the offset says, and a mask sized to the offset would not match the keys.
    func testAPrefillAfterTheWindowHasTrimmedStillRuns() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let cache = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount, window: 4)
        for token in 0 ..< 10 {
            _ = net(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)
        }
        XCTAssertGreaterThan(cache.offset, cache.retainedLength(),
                             "the premise: more seen than retained")

        let logits = net(MLXArray([Int32(1), 2, 3]).reshaped([1, 3]), cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, net.configuration.vocabularySize])
        XCTAssertEqual(cache.offset, 13)
    }

    // The rotary offset is the ABSOLUTE position, so a token's angle does not change when older rows
    // are dropped. Reusing the retained count here would silently rewind every remaining position.
    func testTheRotaryOffsetIgnoresWhatTheWindowDropped() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let cache = NFKMLXKeyValueCache(layerCount: net.configuration.layerCount, window: 4)
        for token in 0 ..< 9 {
            _ = net(MLXArray([Int32(token)]).reshaped([1, 1]), cache: cache)
        }
        XCTAssertEqual(cache.offset, 9)
        XCTAssertLessThanOrEqual(cache.retainedLength(), 4)
        XCTAssertEqual(cache.maskCacheLength, 3, "one short of the window, which is what a step reads")
    }
}
