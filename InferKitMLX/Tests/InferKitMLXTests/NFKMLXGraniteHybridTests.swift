//
//  NFKMLXGraniteHybridTests.swift
//  InferKitMLXTests
//
//  Granite 4.0-H (`GraniteMoeHybridForCausalLM`, IBM), the hybrid Mamba-attention decoder built on the
//  reused Codestral SSD mixer. Numeric parity is measured at a tiny random configuration against
//  transformers' own implementation (`run_reference.py granite_hybrid`), seam by seam: the scaled
//  embedding, each hybrid block's output, the final-normed state, and the scaled logits. The tiny
//  config sets Granite's four multipliers to non-default values so a port that hard-codes them diverges.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXGraniteHybridTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        eval(mine)
        let a = mine.reshaped([-1]).asArray(Float.self), b = reference.reshaped([-1]).asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    /// The tiny geometry `run_reference.py granite_hybrid` records: five Mamba layers then one
    /// attention layer, dense (no routed experts), with distinctive multipliers.
    private var tinyConfiguration: NFKMLXGraniteHybridConfiguration {
        NFKMLXGraniteHybridConfiguration(
            hiddenSize: 64, layerCount: 6, vocabularySize: 128, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: false, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            mambaHeadCount: 8, mambaHeadDimensions: 16, mambaGroupCount: 1, mambaStateSize: 16,
            mambaConvolutionKernel: 4, mambaExpand: 2, mambaConvolutionBias: true,
            mambaProjectionBias: false, sharedIntermediateSize: 96, expertCount: 0,
            expertsPerToken: 0, expertIntermediateSize: 0,
            embeddingMultiplier: 2.0, residualMultiplier: 0.5, attentionMultiplier: 0.25,
            logitsScaling: 3.0,
            layerTypes: [.mamba, .mamba, .mamba, .mamba, .mamba, .attention])
    }

    /// The tiny geometry `run_reference.py granite_hybrid_moe` records: the routed mixture of experts
    /// (8 experts, 2 per token) beside the shared MLP on every layer, with a mixed layer stack.
    private var tinyMoEConfiguration: NFKMLXGraniteHybridConfiguration {
        NFKMLXGraniteHybridConfiguration(
            hiddenSize: 64, layerCount: 6, vocabularySize: 128, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: false, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            mambaHeadCount: 8, mambaHeadDimensions: 16, mambaGroupCount: 1, mambaStateSize: 16,
            mambaConvolutionKernel: 4, mambaExpand: 2, mambaConvolutionBias: true,
            mambaProjectionBias: false, sharedIntermediateSize: 96, expertCount: 8,
            expertsPerToken: 2, expertIntermediateSize: 32,
            embeddingMultiplier: 2.0, residualMultiplier: 0.5, attentionMultiplier: 0.25,
            logitsScaling: 3.0,
            layerTypes: [.mamba, .mamba, .attention, .mamba, .mamba, .attention])
    }

    private func loadRecordWeights(into net: NFKMLXGraniteHybridNet,
                                   from record: [String: MLXArray]) throws {
        let weights = record.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = String(key.dropFirst(3))
            let array = name.hasSuffix("conv1d.weight") && value.ndim == 3
                ? value.reshaped([value.dim(0), value.dim(2)]) : value
            return (name, array)
        }
        try NFKMLXWeights.apply(weights, to: net)
    }

    /// A prompt prefilled in two pieces, then one token a step against the caches, gives the logits the
    /// whole sequence gives at every position, on the dense and the mixture geometry.
    func testCachedGenerationStepsMatchTheWholeSequence() throws {
        try requireMLXRuntime()
        for configuration in [tinyConfiguration, tinyMoEConfiguration] {
            MLXRandom.seed(41)
            let net = NFKMLXGraniteHybrid.makeNet(configuration)
            let random = net.parameters().flattened().map { name, value in
                (name, MLXRandom.normal(value.shape) * 0.3)
            }
            try NFKMLXWeights.apply(random, to: net)
            let tokens = (0 ..< 14).map { Int32(($0 * 37 + 5) % configuration.vocabularySize) }
            let whole = net(MLXArray(tokens).reshaped([1, tokens.count]))

            let cache = NFKMLXGraniteCache(layerTypes: configuration.layerTypes)
            var pieces = [net(MLXArray(Array(tokens[0 ..< 5])).reshaped([1, 5]), cache: cache),
                          net(MLXArray(Array(tokens[5 ..< 8])).reshaped([1, 3]), cache: cache)]
            for token in tokens[8...] {
                pieces.append(net(MLXArray([token]).reshaped([1, 1]), cache: cache))
            }
            let stepped = concatenated(pieces, axis: 1)
            XCTAssertEqual(stepped.shape, whole.shape)
            let difference = abs(stepped - whole).max().item(Float.self)
            XCTAssertLessThan(difference, 1e-4, "experts: \(configuration.expertCount)")

            let forgotten = net(MLXArray(Array(tokens[8...])).reshaped([1, tokens.count - 8]),
                                cache: NFKMLXGraniteCache(layerTypes: configuration.layerTypes))
            let control = abs(forgotten - whole[0..., 8..., 0...]).max().item(Float.self)
            XCTAssertGreaterThan(control, 100 * difference, "the cache carries the positions before the steps")
            print("VALIDATION granite cached steps experts=\(configuration.expertCount) max|Δ|=\(difference) control=\(control)")
        }
    }

    /// A byte-level tokenizer over the tiny vocabulary: a few words, and every other id decoding to its
    /// own bracketed number, so a reply's text names each token.
    private func byteLevelTokenizer() throws -> NFKTokenizer {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXGraniteHybridTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        var vocabulary: [String: Int] = ["h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6,
                                         "Ġ": 7, "Ġhello": 8]
        for id in 9 ..< tinyConfiguration.vocabularySize {
            vocabulary["[\(id)]"] = id
        }
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\nh e\nl l\nhe ll\nhell o\nĠ hello\n".write(
            to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        return try NFKTokenizer(forManifest: ["tokenizer": ["type": "bpe-bytelevel"]], directory: directory)
    }

    /// A named conversation's turn whose prompt extends the last one continues from that prompt's
    /// kept prefill: it reports those tokens cached and answers as a fresh run of the same prompt does.
    /// A prompt that does not extend the last one starts again.
    func testAConversationContinuesFromItsLastPromptsPrefill() throws {
        try requireMLXRuntime()
        MLXRandom.seed(43)
        let net = NFKMLXGraniteHybrid.makeNet(tinyConfiguration)
        try NFKMLXWeights.apply(net.parameters().flattened().map { ($0.0, MLXRandom.normal($0.1.shape) * 0.3) },
                                to: net)
        let backend = NFKMLXGraniteBackend(net: net, tokenizer: try byteLevelTokenizer(), identifier: "granite")
        func run(_ prompt: String, conversation: String?) throws -> (text: String, input: Int, cached: Int) {
            var parameters: [String: Any] = [NFKParameterMaxTokens: 6, NFKParameterTemperature: 0]
            parameters[NFKParameterConversationKey] = conversation
            let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: prompt],
                                                                           parameters: parameters))
            let usage = try XCTUnwrap(result.output(forKey: NFKOutputUsage) as? [String: Int])
            return (result.text ?? "", usage[NFKUsageInputTokens] ?? -1, usage[NFKUsageCachedTokens] ?? -1)
        }

        let first = try run("hello hello", conversation: "chat")
        XCTAssertEqual(first.cached, 0)
        XCTAssertEqual(backend.conversationCacheCount, 1)
        XCTAssertGreaterThan(backend.conversationCacheBytes, 0)

        let second = try run("hello hello hello hello", conversation: "chat")
        XCTAssertEqual(second.cached, first.input, "the turn continues from the last prompt's prefill")
        let fresh = try run("hello hello hello hello", conversation: nil)
        XCTAssertEqual(fresh.cached, 0)
        XCTAssertEqual(second.text, fresh.text, "a continued turn answers as a fresh run of its prompt")
        XCTAssertFalse(fresh.text.isEmpty)

        XCTAssertEqual(try run("hello", conversation: "chat").cached, 0,
                       "a prompt that does not extend the last starts again")
        XCTAssertEqual(try run("hello hello", conversation: "other").cached, 0)
        XCTAssertEqual(backend.conversationCacheCount, 2)
        XCTAssertGreaterThan(backend.releaseIdleConversationCaches(), 0)
        XCTAssertEqual(backend.conversationCacheCount, 0)
    }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXGraniteHybrid.makeNet(tinyConfiguration).parameters().flattened().map(\.0))
        for expected in ["model.embed_tokens.weight",
                         "model.layers.0.input_layernorm.weight",
                         "model.layers.0.post_attention_layernorm.weight",
                         "model.layers.0.mamba.in_proj.weight",
                         "model.layers.0.mamba.conv1d.weight",
                         "model.layers.0.mamba.A_log",
                         "model.layers.0.shared_mlp.input_linear.weight",
                         "model.layers.5.self_attn.q_proj.weight",
                         "model.layers.5.self_attn.o_proj.weight",
                         "model.norm.weight", "lm_head.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
        // The attention layer carries no Mamba tensors, and a Mamba layer no attention tensors.
        XCTAssertFalse(names.contains("model.layers.5.mamba.in_proj.weight"))
        XCTAssertFalse(names.contains("model.layers.0.self_attn.q_proj.weight"))
    }

    func testMoEParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXGraniteHybrid.makeNet(tinyMoEConfiguration).parameters().flattened().map(\.0))
        for expected in ["model.layers.0.block_sparse_moe.router.layer.weight",
                         "model.layers.0.block_sparse_moe.input_linear.weight",
                         "model.layers.0.block_sparse_moe.output_linear.weight",
                         "model.layers.0.shared_mlp.input_linear.weight",
                         "model.layers.2.block_sparse_moe.input_linear.weight",
                         "model.layers.2.self_attn.q_proj.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testTinyMoEConfigSeamParityAgainstTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let recordPath = env["IK_PARITY_GRANITE_MOE_TINY"] else {
            throw XCTSkip("set IK_PARITY_GRANITE_MOE_TINY (run_reference.py granite_hybrid_moe)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let net = NFKMLXGraniteHybrid.makeNet(tinyMoEConfiguration)
        try loadRecordWeights(into: net, from: rec)

        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])
        let blocks = net.blockStates(tokens)
        for index in 0 ..< tinyMoEConfiguration.layerCount {
            let similarity = cosine(blocks[index][0], rec["hidden.\(index)"]!)
            print("SEAM granite-moe hidden.\(index): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "seam hidden.\(index) diverges")
        }
        let finalSimilarity = cosine(net.hiddenStates(tokens)[0],
                                     rec["hidden.\(tinyMoEConfiguration.layerCount)"]!)
        print("SEAM granite-moe hidden.\(tinyMoEConfiguration.layerCount) (post norm): cosine \(finalSimilarity)")
        XCTAssertGreaterThan(finalSimilarity, 0.9999, "final normed seam diverges")

        let logitSimilarity = cosine(net(tokens)[0], rec["output"]!)
        print("VALIDATION PARITY granite_hybrid_moe: logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.9999, "logits diverge")
    }

    func testTinyConfigSeamParityAgainstTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let recordPath = env["IK_PARITY_GRANITE_HYBRID_TINY"] else {
            throw XCTSkip("set IK_PARITY_GRANITE_HYBRID_TINY (run_reference.py granite_hybrid)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let net = NFKMLXGraniteHybrid.makeNet(tinyConfiguration)
        try loadRecordWeights(into: net, from: rec)

        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])
        let blocks = net.blockStates(tokens)                                // [embed·mult, L0 … L5]
        for index in 0 ..< tinyConfiguration.layerCount {
            let similarity = cosine(blocks[index][0], rec["hidden.\(index)"]!)
            print("SEAM granite hidden.\(index): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "seam hidden.\(index) diverges")
        }
        let finalSimilarity = cosine(net.hiddenStates(tokens)[0],
                                     rec["hidden.\(tinyConfiguration.layerCount)"]!)
        print("SEAM granite hidden.\(tinyConfiguration.layerCount) (post norm): cosine \(finalSimilarity)")
        XCTAssertGreaterThan(finalSimilarity, 0.9999, "final normed seam diverges")

        let logitSimilarity = cosine(net(tokens)[0], rec["output"]!)
        print("VALIDATION PARITY granite_hybrid: logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.9999, "logits diverge")
    }

    // The released granite-4.0-h-1b (dense hybrid) held to the module by shape: every parameter matches
    // the checkpoint by name and shape (the depthwise convolution squeezed [C, 1, K] → [C, K]), and a
    // tied release ships no lm_head. Lazy MLX arrays, so building the net costs nothing until evaluated.
    func testEveryParameterMatchesTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let shapesPath = env["IK_SHAPES_GRANITE"], let configPath = env["IK_CONFIG_GRANITE"],
              let data = FileManager.default.contents(atPath: shapesPath),
              let released = try JSONSerialization.jsonObject(with: data) as? [String: [Int]]
        else { throw XCTSkip("set IK_SHAPES_GRANITE and IK_CONFIG_GRANITE (shapes.py)") }

        let config = try NFKMLXGraniteHybrid.configuration(
            fromDirectory: URL(fileURLWithPath: configPath).deletingLastPathComponent())
        XCTAssertEqual(config.layerCount, 40)
        XCTAssertEqual(config.expertCount, 0)                                // h-1b is dense
        XCTAssertTrue(config.tiesWordEmbeddings)
        let net = NFKMLXGraniteHybrid.makeNet(config)

        var consumed = Set<String>()
        var missing = [String](), mismatched = [String]()
        for (name, value) in net.parameters().flattened() {
            let expected = name.hasSuffix("conv1d.weight") ? [value.dim(0), 1, value.dim(1)] : value.shape
            guard let shape = released[name] else { missing.append(name); continue }
            consumed.insert(name)
            if shape != expected { mismatched.append("\(name): built \(expected), released \(shape)") }
        }
        let unaccounted = released.keys.filter { !consumed.contains($0) }.sorted()
        print("VALIDATION structure granite-4.0-h-1b: \(consumed.count) consumed, \(missing.count) missing, "
              + "\(mismatched.count) mismatched, \(unaccounted.count) unaccounted")
        XCTAssertTrue(mismatched.isEmpty, "shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(missing.isEmpty, "absent from the release:\n" + missing.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(unaccounted.isEmpty, "released tensors nothing reads:\n" + unaccounted.prefix(8).joined(separator: "\n"))
    }

    // The released granite-4.0-h-1b run numerically at float32 (it fits) against a float32 transformers
    // oracle (`run_reference.py granite_hybrid_real`): each layer's hidden state, the logits, and the
    // greedy continuation.
    func testReleasedWeightsNumericParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_GRANITE"], let recordPath = env["IK_PARITY_GRANITE_REAL"] else {
            throw XCTSkip("set IK_VAL_GRANITE (release dir) and IK_PARITY_GRANITE_REAL (granite_hybrid_real)")
        }
        let dir = URL(fileURLWithPath: directory)
        let config = try NFKMLXGraniteHybrid.configuration(fromDirectory: dir)
        let net = NFKMLXGraniteHybrid.makeNet(config)
        try NFKMLXGraniteHybrid.loadWeights(into: net, fromDirectory: dir, precision: .float32)

        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])
        let blocks = net.blockStates(tokens)
        var worst: Float = 1
        for index in 0 ..< config.layerCount {
            worst = Swift.min(worst, cosine(blocks[index][0], rec["hidden.\(index)"]!))
        }
        let finalSimilarity = cosine(net.hiddenStates(tokens)[0], rec["hidden.\(config.layerCount)"]!)
        let logitSimilarity = cosine(net(tokens)[0], rec["output"]!)
        print("VALIDATION PARITY granite-4.0-h-1b released: worst block seam \(worst), "
              + "final \(finalSimilarity), logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(worst, 0.999, "a block seam diverges")
        XCTAssertGreaterThan(logitSimilarity, 0.999, "released logits diverge")

        let reference = rec["continuation"]!.asType(.int32)
        eval(reference)
        let referenceIds = reference.asArray(Int32.self)
        var sequence = tokens
        var produced = [Int32]()
        for _ in 0 ..< referenceIds.count {
            let next = argMax(net(sequence)[0, -1, 0...], axis: -1)
            eval(next)
            let id = next.item(Int32.self)
            produced.append(id)
            sequence = concatenated([sequence, MLXArray([id]).reshaped([1, 1])], axis: 1)
        }
        let matches = zip(produced, referenceIds).filter { $0 == $1 }.count
        print("VALIDATION PARITY granite-4.0-h-1b released: greedy \(matches)/\(referenceIds.count) tokens match")
        XCTAssertEqual(produced.first, referenceIds.first, "first greedy token differs")
        XCTAssertGreaterThanOrEqual(matches, referenceIds.count - 2, "greedy continuation diverges")
    }

    // Granite's byte-level BPE (the GPT-2 family), read from the release's tokenizer.json through the
    // shared release-tokenizer reader, encodes token-exactly to the ids the reference `transformers`
    // tokenizer produces (no special tokens added, matching the backend's raw-prompt encoding).
    func testTokenizerAgreesWithTheReference() throws {
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_GRANITE"] else {
            throw XCTSkip("set IK_VAL_GRANITE (release directory with tokenizer.json)")
        }
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: URL(fileURLWithPath: directory)) else {
            return XCTFail("the Granite release has no readable tokenizer")
        }
        let ids = tokenizer.encode("def fibonacci(n):").map(\.intValue)
        XCTAssertEqual(ids, [755, 76798, 1471, 1680], "tokenization diverges from the reference")
        XCTAssertEqual(tokenizer.eosTokenId, 100257, "end-of-sequence id differs")
    }

    // The Granite 4.0-H backend runs end to end from a release directory: it builds the decoder, the
    // tokenizer, and generates a non-empty greedy continuation for a raw prompt.
    func testBackendGeneratesText() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_GRANITE"] else {
            throw XCTSkip("set IK_VAL_GRANITE (release directory)")
        }
        let backend = try NFKMLXGraniteHybrid.backend(directoryURL: URL(fileURLWithPath: directory),
                                                      precision: .float32)
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "def fibonacci(n):"],
                                          parameters: [NFKParameterMaxTokens: 8, NFKParameterTemperature: 0])
        let result = try backend.runInference(for: request)
        let text = result.text ?? ""
        print("VALIDATION granite-4.0-h backend produced: \(text.debugDescription)")
        XCTAssertFalse(text.isEmpty, "the backend produced no text")
    }
}
