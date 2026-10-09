//
//  NFKMLXGemma3Tests.swift
//  InferKitMLXTests
//
//  Weight-free tests for the Gemma 3 decoder: the configuration reader, the layer-kind pattern, the
//  hybrid masks, the cached decode against a full pass, causality, the projector's pooling, and the
//  prompt expansion. Numeric parity against the reference lives in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXGemma3Tests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        // Clearing the cache reaches MLX's runtime, which needs a Metal library it can find.
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func write(_ json: [String: Any]) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma3-config-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return url
    }

    // MARK: Configuration

    func testTheReaderTakesATextReleaseAndAMultimodalOne() throws {
        let text = try write(["model_type": "gemma3_text", "hidden_size": 640, "num_hidden_layers": 18,
                              "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
                              "intermediate_size": 2048, "sliding_window": 512, "_sliding_window_pattern": 6,
                              "rope_theta": 1000000.0, "rope_local_base_freq": 10000.0, "rope_scaling": NSNull()])
        let small = try NFKMLXGemma3Language.configuration(fromHuggingFace: text)
        XCTAssertEqual(small.hiddenSize, 640)
        XCTAssertEqual(small.ropeScalingFactor, 1)
        XCTAssertEqual(small.layerTypes.count, 18)
        XCTAssertEqual(small.layerTypes.enumerated().filter { $0.element == .full }.map(\.offset), [5, 11, 17],
                       "every sixth layer is full attention")
        XCTAssertFalse(small.isBidirectional)

        let multimodal = try write(["model_type": "gemma3", "vision_config": ["hidden_size": 1152],
                                    "text_config": ["model_type": "gemma3_text", "hidden_size": 2560,
                                                    "num_hidden_layers": 34, "num_attention_heads": 8,
                                                    "num_key_value_heads": 4, "intermediate_size": 10240,
                                                    "sliding_window": 1024, "sliding_window_pattern": 6,
                                                    "vocab_size": 262208,
                                                    "rope_scaling": ["factor": 8.0, "rope_type": "linear"]]])
        let large = try NFKMLXGemma3Language.configuration(fromHuggingFace: multimodal)
        XCTAssertEqual(large.hiddenSize, 2560)
        XCTAssertEqual(large.keyValueHeadCount, 4)
        XCTAssertEqual(large.vocabularySize, 262_208)
        XCTAssertEqual(large.slidingWindow, 1024)
        XCTAssertEqual(large.ropeScalingFactor, 8, "the 4B stretches its global layers' rotary by 8")
        XCTAssertEqual(large.layerTypes.filter { $0 == .full }.count, 5)
    }

    // A bidirectional release states its window as the full span, and the reference turns it into
    // the exclusive bound `span / 2 + 1` on `|q - k|`. EmbeddingGemma's 512 is therefore 257.
    func testABidirectionalReleaseHalvesItsWindow() throws {
        let url = try write(["model_type": "gemma3_text", "use_bidirectional_attention": true, "sliding_window": 512])
        let configuration = try NFKMLXGemma3Language.configuration(fromHuggingFace: url)
        XCTAssertTrue(configuration.isBidirectional)
        XCTAssertEqual(configuration.slidingWindow, 257)
        XCTAssertEqual(NFKMLXGemma3EncoderConfiguration.embeddingGemma300M.geometry.slidingWindow, 257)
    }

    func testTheReaderRefusesTheNamesakes() throws {
        XCTAssertThrowsError(try NFKMLXGemma3Language.configuration(fromHuggingFace: write(["model_type": "gemma3n"])))
        XCTAssertThrowsError(try NFKMLXGemma3Language.configuration(fromHuggingFace: write(["model_type": "gemma4_text"])))
        XCTAssertThrowsError(try NFKMLXGemma3Language.configuration(fromHuggingFace: write(
            ["model_type": "gemma3n", "text_config": ["model_type": "gemma3n_text"]])))
        XCTAssertThrowsError(try NFKMLXGemma3Language.configuration(fromHuggingFace: write(
            ["model_type": "gemma3_text", "rope_scaling": ["rope_type": "yarn", "factor": 4.0]])),
            "a rotary scaling this does not implement is refused, not approximated")
    }

    // MARK: Masks

    func testTheMasksAdmitTheWindowAndTheImageBlock() throws {
        try requireMLXRuntime()
        // Positions 1…3 are one image; everything else is text. Window 3, no cache.
        let masks = NFKMLXGemma3Masks.make(length: 8, offset: 0, window: 3,
                                           blockIds: [-1, 0, 0, 0, -1, -1, -1, -1], bidirectional: false)
        let full = try XCTUnwrap(masks.full)
        let sliding = try XCTUnwrap(masks.sliding)
        eval(full, sliding)
        func admitted(_ mask: MLXArray, _ q: Int, _ k: Int) -> Bool { mask[q, k].item(Float.self) == 0 }

        XCTAssertTrue(admitted(full, 6, 2), "a full layer sees the whole past")
        XCTAssertFalse(admitted(full, 2, 6), "but never the text future")
        XCTAssertTrue(admitted(full, 1, 3), "an image token sees the rest of its image, ahead of it too")
        XCTAssertFalse(admitted(full, 1, 4), "and not the text after the image")

        XCTAssertTrue(admitted(sliding, 6, 4), "a sliding layer sees two back at window 3")
        XCTAssertFalse(admitted(sliding, 6, 3), "and not three back")
        XCTAssertTrue(admitted(sliding, 1, 3), "the image block is bidirectional inside the window too")

        let single = NFKMLXGemma3Masks.make(length: 1, offset: 10, window: 3, blockIds: nil, bidirectional: false)
        XCTAssertNil(single.full, "a single cached step needs no mask")
        XCTAssertNil(single.sliding)
    }

    // The bidirectional window: |q - k| < window, either side.
    func testTheBidirectionalMaskIsSymmetric() throws {
        try requireMLXRuntime()
        let masks = NFKMLXGemma3Masks.make(length: 6, offset: 0, window: 3, blockIds: nil, bidirectional: true)
        let sliding = try XCTUnwrap(masks.sliding)
        eval(sliding)
        XCTAssertEqual(sliding[2, 4].item(Float.self), 0, "two ahead is inside the window")
        XCTAssertEqual(sliding[4, 2].item(Float.self), 0, "and two back")
        XCTAssertLessThan(sliding[2, 5].item(Float.self), -1e8, "three ahead is not")
        XCTAssertEqual(try XCTUnwrap(masks.full).sum().item(Float.self), 0, "a full layer sees everything")
    }

    // MARK: Decoding

    // The whole point of the cache: a step against it reads what a full pass would. The prompt is
    // longer than the tiny 4-position window, so the sliding cache trims while decoding.
    func testCachedDecodingMatchesTheFullPass() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(7)
        let net = NFKMLXGemma3Net(.tiny)
        let tokens: [Int32] = [3, 17, 42, 99, 7, 61, 12, 5, 88, 30, 44, 120]
        let full = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(full)

        let cache = NFKMLXGemma3Cache(layerCount: 3, slidingWindow: 4)
        let prefix = Array(tokens[0 ..< 6])
        var stepped = net(MLXArray(prefix).reshaped([1, prefix.count]), cache: cache)[0, -1]
        XCTAssertEqual(cache.offset, 6)
        for position in 6 ..< tokens.count {
            let reference = full[0, position - 1].asArray(Float.self)
            let mine = stepped.asArray(Float.self)
            let error = zip(reference, mine).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(error, 1e-4, "the cached step at position \(position - 1) reads what the full pass reads")
            stepped = net(MLXArray([tokens[position]]).reshaped([1, 1]), cache: cache)[0, -1]
        }
        XCTAssertEqual(cache.offset, tokens.count)
        XCTAssertEqual(cache.sliding.retainedLength(layer: 0), 4, "the sliding cache keeps the window")
        XCTAssertEqual(cache.full.retainedLength(layer: 2), tokens.count, "the full cache keeps everything")
    }

    func testTheDecoderIsCausalAndTheEncoderIsNot() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(9)
        let decoder = NFKMLXGemma3Net(.tiny)
        let short = decoder(MLXArray([Int32(3), 17, 42, 99]).reshaped([1, 4]))
        let long = decoder(MLXArray([Int32(3), 17, 42, 99, 7, 61]).reshaped([1, 6]))
        eval(short, long)
        let drift = (short - long[0..., 0 ..< 4]).abs().max().item(Float.self)
        XCTAssertLessThan(drift, 1e-4, "appending tokens does not change an earlier position's logits")

        let encoder = NFKMLXGemma3EncoderNet(.tiny)
        let shortHidden = encoder(MLXArray([Int32(3), 17, 42, 99]).reshaped([1, 4]))
        let longHidden = encoder(MLXArray([Int32(3), 17, 42, 99, 7, 61]).reshaped([1, 6]))
        eval(shortHidden, longHidden)
        let change = (shortHidden - longHidden[0..., 0 ..< 4]).abs().max().item(Float.self)
        XCTAssertGreaterThan(change, 1e-4, "the bidirectional encoder reads the tokens after a position")
    }

    // The final soft-cap bounds the logits, and the attention soft-cap runs without a fused kernel.
    func testTheSoftCapsBoundTheLogits() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXGemma3Configuration.tiny
        configuration.finalLogitSoftcap = 30
        let net = NFKMLXGemma3Net(configuration)
        let logits = net(MLXArray([Int32(1), 2, 3, 4, 5]).reshaped([1, 5]))
        eval(logits)
        XCTAssertLessThanOrEqual(Double(logits.abs().max().item(Float.self)), 30 + 1e-3)
    }

    // MARK: Vision

    func testTheProjectorPoolsThePatchGrid() throws {
        try requireMLXRuntime()
        // A 4×4 patch grid pooled 2×2 to four tokens; the projection is the identity so the pooled,
        // normalized cell means come through.
        let projector = NFKMLXGemma3MultimodalProjector(visionHidden: 8, textHidden: 8, patchesPerSide: 4, tokensPerImage: 4)
        projector.update(parameters: ModuleParameters.unflattened([("mm_input_projection_weight", MLXArray.eye(8))]))
        let features = MLXRandom.normal([1, 16, 8])
        let projected = projector(features)
        eval(projected)
        XCTAssertEqual(projected.shape, [1, 4, 8])

        // Token 1 is the top-right cell: patches (0,2), (0,3), (1,2), (1,3) → flat 2, 3, 6, 7.
        let cell = (features[0, 2] + features[0, 3] + features[0, 6] + features[0, 7]) / 4
        let expected = projector.norm(cell.reshaped([1, 8]))[0]
        eval(expected)
        let error = zip(projected[0, 1].asArray(Float.self), expected.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(error, 1e-5, "a soft token is its cell's mean, normalized")
    }

    func testTheVisionTowerProducesPatchFeatures() throws {
        try requireMLXRuntime()
        let tower = NFKMLXGemma3VisionNet(.tiny)
        let features = tower(MLXRandom.uniform(low: -1, high: 1, [1, 64, 64, 3]))
        eval(features)
        XCTAssertEqual(features.shape, [1, 16, 32], "16 patches of a 64-pixel image at patch 16")
    }

    // MARK: Prompts

    /// A tokenizer over a handful of pieces plus the Gemma markers, written as a tokenizer.json.
    private func tinyTokenizer() throws -> NFKMLXGemmaTokenizer {
        var vocabulary: [String: Int] = ["<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3]
        for (index, piece) in ["\u{2581}", "H", "i", "\n", "u", "s", "e", "r", "m", "o", "d", "l", "\n\n"].enumerated() {
            vocabulary[piece] = 10 + index
        }
        let added: [[String: Any]] = [
            ["id": 2, "content": "<bos>", "special": true],
            ["id": 105, "content": "<start_of_turn>", "special": true],
            ["id": 106, "content": "<end_of_turn>", "special": true],
            ["id": 255_999, "content": "<start_of_image>", "special": true],
            ["id": 256_000, "content": "<end_of_image>", "special": true],
            ["id": 262_144, "content": "<image_soft_token>", "special": true],
        ]
        let json: [String: Any] = ["model": ["vocab": vocabulary, "merges": [["\n", "\n"]], "unk_token": "<unk>"],
                                   "added_tokens": added]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gemma3-tok-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        return try XCTUnwrap(NFKMLXGemmaTokenizer(tokenizerJSON: url))
    }

    func testTheTokenizerMatchesAddedTokensLiterally() throws {
        let tokenizer = try tinyTokenizer()
        XCTAssertEqual(tokenizer.encode("<start_of_turn>user\nHi<end_of_turn>"),
                       [105, 14, 15, 16, 17, 13, 11, 12, 106])
        XCTAssertEqual(tokenizer.encode("\n\n<start_of_image><image_soft_token><image_soft_token><end_of_image>\n\n"),
                       [22, 255_999, 262_144, 262_144, 256_000, 22], "the double newline merges into one piece")
        XCTAssertEqual(tokenizer.id(forToken: "<image_soft_token>"), 262_144)
        XCTAssertEqual(tokenizer.decode([105, 14, 15, 16, 17, 106], skipSpecial: true), "user")
        XCTAssertTrue(tokenizer.specialIds.contains(106))
    }

    private func tinyModel() throws -> NFKMLXGemma3Model {
        try requireMLXRuntime()       // constructing the decoder initializes MLX
        return NFKMLXGemma3Model(decoder: NFKMLXGemma3Net(.tiny), vision: nil, projector: nil,
                          tokenizer: try tinyTokenizer(),
                          tokens: NFKMLXGemma3Tokens(tokensPerImage: 3), chatTemplate: nil)
    }

    /// Greedy, never stopping early, so each turn feeds the cache a full reply.
    private var continuingOptions: NFKMLXGenerationOptions {
        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = 3
        options.stopTokens = [-1]
        return options
    }

    private func longestSharedPrefix(_ a: [Int], _ b: [Int]) -> Int {
        zip(a, b).prefix { $0 == $1 }.count
    }

    private func greedy(_ count: Int) -> NFKMLXGenerationOptions {
        var options = continuingOptions
        options.maxTokens = count
        return options
    }

    // The prompt and the run outlast the tiny 4-position window, so the target's rollbacks cross the
    // sliding cache's trim and the draft starts over where its own cache cannot roll back.
    func testDraftedGenerationIsThePlainGreedyOutput() throws {
        let model = try tinyModel()
        let prompt = [2, 11, 12, 13, 14, 15]
        let plain = try model.generate(tokens: prompt, options: greedy(20))

        try model.useDraft(model)
        XCTAssertEqual(try model.generate(tokens: prompt, options: greedy(20)), plain)
        XCTAssertEqual(model.lastSpeculativeReport.accepted, model.lastSpeculativeReport.proposed,
                       "a draft identical to the model is always right")
        XCTAssertLessThan(model.lastSpeculativeReport.rounds, plain.count - 1, "and saves passes")

        NFKMLXRandom.seed(5)
        try model.useDraft(try tinyModel())
        XCTAssertEqual(try model.generate(tokens: prompt, options: greedy(20)), plain,
                       "a wrong draft costs passes, never tokens")
        XCTAssertGreaterThan(model.lastSpeculativeReport.rounds, 0)

        var off = greedy(20)
        off.draftTokens = 0
        let rounds = model.lastSpeculativeReport.rounds
        XCTAssertEqual(try model.generate(tokens: prompt, options: off), plain)
        XCTAssertEqual(model.lastSpeculativeReport.rounds, rounds, "no proposals asked, none made")
    }

    func testDraftedGenerationHonorsStopsTheHandlerAndTheLimit() throws {
        let model = try tinyModel()
        let plain = try model.generate(tokens: [2, 11], options: greedy(12))
        NFKMLXRandom.seed(6)
        try model.useDraft(try tinyModel())
        if let later = plain.indices.dropFirst().first(where: { !plain[..<$0].contains(plain[$0]) }) {
            var stopping = greedy(12)
            stopping.stopTokens = [plain[later]]
            XCTAssertEqual(try model.generate(tokens: [2, 11], options: stopping), Array(plain.prefix(later)))
        }
        var seen = 0
        let cut = try model.generate(tokens: [2, 11], options: greedy(12)) { _ in
            seen += 1
            return seen < 3
        }
        XCTAssertEqual(cut, Array(plain.prefix(3)))
        XCTAssertEqual(try model.generate(tokens: [2, 11], options: greedy(5)), Array(plain.prefix(5)))
    }

    func testADraftedConversationContinuesAsAFreshRun() throws {
        let model = try tinyModel()
        try model.useDraft(model)
        let options = continuingOptions
        let configuration = model.decoder.configuration
        let cache = NFKMLXGemma3PromptCache(kinds: configuration.layerTypes, slidingWindow: configuration.slidingWindow)
        let first = [2, 11, 12]
        let reply = try model.generate(tokens: first, image: nil, options: options, promptCache: cache) { _ in true }
        // A verifying pass keeps every proposal the model agrees with, past the reply's limit too.
        XCTAssertEqual(Array(cache.tokens.prefix(first.count + reply.count)), first + reply,
                       "the cache holds rows for the prompt and the reply")
        let next = first + reply + [13, 11]
        let continued = try model.generate(tokens: next, image: nil, options: options, promptCache: cache) { _ in true }
        XCTAssertEqual(continued, try model.generate(tokens: next, options: options))
    }

    func testADraftMustShareTheVocabulary() throws {
        let model = try tinyModel()
        var configuration = NFKMLXGemma3Configuration.tiny
        configuration.vocabularySize = 140
        let other = NFKMLXGemma3Model(decoder: NFKMLXGemma3Net(configuration), vision: nil, projector: nil,
                                      tokenizer: try tinyTokenizer(), tokens: NFKMLXGemma3Tokens(tokensPerImage: 3),
                                      chatTemplate: nil)
        XCTAssertThrowsError(try model.useDraft(other))
        XCTAssertNil(model.draft)
    }

    // A conversation's next turn continues the rows its last turn left, rolled back to where the
    // prompts part, and answers what a run from nothing answers.
    func testAContinuedConversationAnswersAsAFreshRun() throws {
        let model = try tinyModel()
        let options = continuingOptions
        let configuration = model.decoder.configuration
        let cache = NFKMLXGemma3PromptCache(kinds: configuration.layerTypes, slidingWindow: configuration.slidingWindow)
        let first = [2, 11, 12]
        let reply = try model.generate(tokens: first, image: nil, options: options, promptCache: cache) { _ in true }
        XCTAssertEqual(cache.tokens, first + reply, "the prompt and every token the run fed")

        let next = first + reply + [13, 11]
        let continued = try model.generate(tokens: next, image: nil, options: options, promptCache: cache) { _ in true }
        XCTAssertEqual(cache.sharedPrefixLength, first.count + reply.count, "only the new message prefills")
        XCTAssertEqual(continued, try model.generate(tokens: next, options: options), "the answer of a fresh run")

        let held = cache.tokens
        let branch = Array(held.dropLast(1)) + [15, 16]
        let branched = try model.generate(tokens: branch, image: nil, options: options, promptCache: cache) { _ in true }
        XCTAssertEqual(cache.sharedPrefixLength, longestSharedPrefix(held, branch),
                       "a prompt that parts inside the window rolls back to where it parts")
        XCTAssertEqual(branched, try model.generate(tokens: branch, options: options))
    }

    // The sliding layers keep only their window, so a prompt that parts further back than the window
    // reaches cannot roll back and starts the cache over.
    func testAPromptPartingBeyondTheWindowStartsOver() throws {
        let model = try tinyModel()
        let options = continuingOptions
        let configuration = model.decoder.configuration
        let cache = NFKMLXGemma3PromptCache(kinds: configuration.layerTypes, slidingWindow: configuration.slidingWindow)
        _ = try model.generate(tokens: [2, 11, 12, 13, 11, 12, 13, 11], image: nil, options: options,
                               promptCache: cache) { _ in true }
        let restart = [2, 14, 14, 14]
        let answered = try model.generate(tokens: restart, image: nil, options: options, promptCache: cache) { _ in true }
        XCTAssertEqual(cache.sharedPrefixLength, 0, "the window has dropped the positions it would roll back to")
        XCTAssertEqual(cache.tokens, restart + answered)
        XCTAssertEqual(answered, try model.generate(tokens: restart, options: options))
    }

    // Decoding keeps the sliding layers at their window, so rolling back two positions after a
    // two-token reply leaves fewer rows than the masks read. That rollback is refused and the run
    // starts over; continuing would build masks one key wider than the cache.
    func testARollbackThatWouldLeaveTheWindowShortStartsOver() throws {
        let model = try tinyModel()
        var options = continuingOptions
        options.maxTokens = 2
        let configuration = model.decoder.configuration
        let cache = NFKMLXGemma3PromptCache(kinds: configuration.layerTypes, slidingWindow: configuration.slidingWindow)
        let reply = try model.generate(tokens: [2, 11, 12], image: nil, options: options, promptCache: cache) { _ in true }
        let parted = [2, 11, 12] + [reply[0] == 10 ? 14 : 10, 11, 12]
        let answered = try model.generate(tokens: parted, image: nil, options: options, promptCache: cache) { _ in true }
        XCTAssertEqual(cache.sharedPrefixLength, 0, "two positions back from a full window is past what it keeps")
        XCTAssertEqual(answered, try model.generate(tokens: parted, options: options))
    }

    func testTheBackendKeepsACacheForEachNamedConversation() throws {
        let backend = NFKMLXGemma3Backend(model: try tinyModel(), identifier: "tiny")
        // One reply token, so the next turn rolls back at most one position, which the window keeps.
        func cachedTokens(_ prompt: String, conversation: String?) throws -> Int {
            var parameters: [String: Any] = [NFKParameterTemperature: 0, NFKParameterMaxTokens: 1]
            parameters[NFKParameterConversationKey] = conversation
            let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: prompt],
                                                                           parameters: parameters))
            let usage = try XCTUnwrap(result.output(forKey: NFKOutputUsage) as? [String: Int])
            return try XCTUnwrap(usage[NFKUsageCachedTokens])
        }
        XCTAssertEqual(try cachedTokens("Hi", conversation: "a"), 0)
        XCTAssertGreaterThanOrEqual(try cachedTokens("Hi Hi", conversation: "a"), 3, "the first turn's prompt is reused")
        XCTAssertEqual(try cachedTokens("Hi Hi", conversation: nil), 0, "a request in no conversation keeps nothing")
        XCTAssertTrue(backend.supportedParameterKeys.contains(NFKParameterConversationKey))
        XCTAssertEqual(backend.conversationCacheCount, 1)
        XCTAssertEqual(backend.backendStatus["conversation_caches"] as? Int, 1)
        XCTAssertEqual(backend.backendStatus["conversation_cache_bytes"] as? Int, backend.conversationCacheBytes)
        backend.resetPromptCache(forConversation: "a")
        XCTAssertEqual(backend.conversationCacheCount, 0)
    }

    func testThePromptExpandsAnImageAndTheBlockIdsMarkIt() throws {
        let model = try tinyModel()
        let ids = model.promptTokens("Hi", withImage: true)
        XCTAssertEqual(ids, [2, 22, 255_999, 262_144, 262_144, 262_144, 256_000, 22, 11, 12],
                       "BOS, the image sequence, then the text")
        XCTAssertEqual(model.blockIds(for: ids), [-1, -1, -1, 0, 0, 0, -1, -1, -1, -1])
        XCTAssertNil(model.blockIds(for: [2, 11, 12]), "a text-only prompt has no blocks")
        XCTAssertEqual(model.promptTokens("Hi"), [2, 11, 12])

        // Two images in one prompt are two blocks; a run already expanded is left alone.
        let two = model.expandingImageMarkers(in: "<start_of_image>Hi<start_of_image>")
        XCTAssertEqual(two.components(separatedBy: "<image_soft_token>").count - 1, 6)
        XCTAssertEqual(model.expandingImageMarkers(in: two), two)
        let twoIds = model.tokenizer.encode(two)
        XCTAssertEqual(Set(try XCTUnwrap(model.blockIds(for: twoIds))), [-1, 0, 1])
    }

    func testTheHandRenderedChatOpensTheModelTurn() throws {
        let model = try tinyModel()
        let ids = model.chatTokens(messages: [["role": "user", "content": "Hi"]])
        XCTAssertEqual(ids.first, 2, "BOS leads")
        XCTAssertEqual(ids[1], 105, "then the user turn opens")
        XCTAssertEqual(Array(ids.suffix(9)), [106, 13, 105, 18, 19, 20, 16, 21, 13],
                       "the user turn closes and the model turn opens")
        let text = NFKMLXGemma3Model.handRenderedChat([["role": "system", "content": "Be brief."],
                                                       ["role": "user", "content": "Hi"],
                                                       ["role": "assistant", "content": "Hello."],
                                                       ["role": "user", "content": "Bye"]])
        XCTAssertEqual(text, "<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\nHello.<end_of_turn>\n"
                       + "<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n",
                       "a system message prefixes the first user turn, as the release's template does")
    }
}
