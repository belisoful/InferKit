//
//  NFKMLXOpenJevTests.swift
//  InferKitMLXTests
//
//  Open-Jev: the candidate prompts the loader writes, the chat turn, the readout's confidence
//  formulas, the PEFT adapter mapping, and the customization round trip through the public factory
//  over a tiny base written to disk; then, gated on the 2B release, its pinned base, and the record
//  (`run_reference.py open_jev`), every candidate's tokens, every hidden state of the adapted text
//  model, the logits, the answers, and the loss against the loader's own code.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXOpenJevTests: XCTestCase {

    private var config: [String: String] { NFKMLXValidationConfig.environment }

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

    private let intent = NFKDecisionQuestion.choiceQuestion(withInstructions: "Choose the customer intent.",
                                                            options: ["billing", "technical", "other"],
                                                            descriptions: ["billing": "A payment or refund issue"])
    private let urgency = NFKDecisionQuestion.scoreQuestion(withInstructions: "How urgent?", levels: ["low", "mid", "high"])
    private let refund = NFKDecisionQuestion.noulQuestion(withInstructions: "The customer asks for a refund.",
                                                          trueMeaning: "money back is requested",
                                                          falseMeaning: "no refund is requested")

    // MARK: - The prompts

    func testTheCandidatePromptsFollowTheLoader() throws {
        let state: [String: Any] = ["b": 2, "a": "é"]
        let choice = try NFKMLXOpenJevPrompt.candidates(state: state, question: intent)
        XCTAssertEqual(choice.count, 3)
        XCTAssertEqual(choice[0], "Context:\n{\"a\": \"é\", \"b\": 2}\n\nQuestion: Choose the customer intent.\n"
                       + "Proposed answer: billing: A payment or refund issue\nIs this proposed answer correct? Answer Yes or No.")
        XCTAssertTrue(choice[2].contains("Proposed answer: other\n"), "an undescribed option is its name")

        let noul = try NFKMLXOpenJevPrompt.candidates(state: "s", question: refund)
        XCTAssertEqual(noul, ["Context:\ns\n\nQuestion: The customer asks for a refund.\nYes means: money back is requested\n"
                              + "No means: no refund is requested\nIs the answer to this question yes? Answer Yes or No."])
        let half = NFKDecisionQuestion.noulQuestion(withInstructions: "x", trueMeaning: "y", falseMeaning: nil)
        XCTAssertThrowsError(try NFKMLXOpenJevPrompt.candidates(state: "s", question: half), "both meanings or neither")

        XCTAssertEqual(NFKMLXOpenJevPrompt.chat("hi"),
                       "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    func testTheReadoutFollowsTheLoadersFormulas() {
        let choice = NFKMLXOpenJev.answer(question: intent, probabilities: [0.7, 0.2, 0.1])
        XCTAssertEqual(choice.choice, "billing")
        XCTAssertEqual(choice.confidence, (0.7 - 1.0 / 3) / (1 - 1.0 / 3), accuracy: 1e-12, "above uniform, scaled to one")
        XCTAssertEqual(NFKMLXOpenJev.answer(question: .choiceQuestion(withInstructions: "t", options: ["only"]),
                                            probabilities: [1]).confidence, 1)

        let score = NFKMLXOpenJev.answer(question: urgency, probabilities: [0.2, 0.5, 0.3])
        XCTAssertEqual(score.score, 1.1, accuracy: 1e-12)
        // Mode 1: expected distance 0.2 + 0.3 = 0.5 against the uniform distance (1 + 0 + 1) / 3.
        XCTAssertEqual(score.confidence, 1 - 0.5 / (2.0 / 3), accuracy: 1e-12)
        XCTAssertEqual(score.legend?["0"], "low")

        let noul = NFKMLXOpenJev.answer(question: refund, probabilities: [0.25, 0.75])
        XCTAssertEqual(noul.probability, 0.75)
        let probabilities = NFKMLXOpenJev.softmax([0, 2], temperature: 2)
        XCTAssertEqual(probabilities[1], 1 / (1 + Foundation.exp(-1.0)), accuracy: 1e-12)
    }

    func testThePEFTAdapterKeysMapOntoTheAdaptedProjections() {
        XCTAssertEqual(NFKMLXOpenJev.adapterKey(forReference: "base_model.model.layers.3.self_attn.q_proj.lora_A.weight"),
                       "decoder.model.layers.3.self_attn.q_proj.lora_a")
        XCTAssertEqual(NFKMLXOpenJev.adapterKey(forReference: "base_model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"),
                       "decoder.model.layers.0.linear_attn.in_proj_qkv.lora_b")
        XCTAssertNil(NFKMLXOpenJev.adapterKey(forReference: "model.layers.0.mlp.gate_proj.weight"))
    }

    // MARK: - The tiny round trip through the factory

    /// Writes a tiny Qwen3.5 base the way a release lays one out: `config.json` with the decoder under
    /// `text_config`, and the weights under `model.language_model.`, the convolution as PyTorch stores it.
    private func writeTinyBase(to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let c = NFKMLXOpenJevConfiguration.tiny.decoder
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
        NFKMLXRandom.seed(21)
        let decoder = NFKMLXHybridLanguage.makeNet(c)
        var arrays = [String: MLXArray]()
        for (key, value) in decoder.parameters().flattened() {
            arrays[NFKMLXHybridLanguage.referenceKey(for: key)] = key.hasSuffix("conv1d.weight") ? value.transposed(0, 2, 1) : value
        }
        try MLX.save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
    }

    /// A release checkpoint for the tiny base: a nonzero random adapter, a head, and the calibration.
    private func writeTinyCheckpoint(to directory: URL) throws -> NFKMLXOpenJevNet {
        let net = NFKMLXOpenJevNet(.tiny)
        try net.adapt()
        NFKMLXRandom.seed(22)
        let detours = net.decoder.leafModules().flattened().compactMap { path, module -> (String, MLXArray)? in
            guard let layer = module as? NFKMLXLoRALinear else { return nil }
            return ("decoder.\(path).lora_b", MLXRandom.normal(layer.loraB.shape) * 0.05)
        }
        try NFKMLXOpenJev.applyPart(detours, to: net) { $0.hasSuffix(".lora_b") }
        try NFKMLXOpenJev(net: net, tokenizer: nil).save(to: directory)
        return net
    }

    func testTheFactoryLoadsTheAdapterAndHeadAndMatchesTheMergedWeights() throws {
        try requireMLXRuntime()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("openjev-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let base = root.appendingPathComponent("base")
        let checkpoint = root.appendingPathComponent("checkpoint")
        try writeTinyBase(to: base)
        let written = try writeTinyCheckpoint(to: checkpoint)

        let loaded = try NFKMLXOpenJev.network(checkpointDirectoryURL: checkpoint, baseDirectoryURL: base, precision: .float32)
        XCTAssertEqual(loaded.configuration.temperature, 1.5)
        let tokens = [5, 17, 90, 3, 44, 12, 7]
        let score = loaded.score(tokens: tokens)

        // The adapter the factory installed equals folding it into the base weights by hand.
        let merged = try NFKMLXOpenJev.network(checkpointDirectoryURL: checkpoint, baseDirectoryURL: base, precision: .float32)
        XCTAssertEqual(try NFKMLXLoRA.merge(into: merged.decoder), 10,
                       "in_proj_qkv and out_proj in each of three recurrent layers, q/k/v/o in the attention layer")
        XCTAssertEqual(merged.score(tokens: tokens).item(Float.self), score.item(Float.self), accuracy: 1e-4)
        XCTAssertEqual(loaded.head.weight.asArray(Float.self), written.head.weight.asArray(Float.self))

        // Removing the adapter changes the score, so the file carried a real detour.
        let bare = NFKMLXOpenJevNet(.tiny)
        try NFKMLXHybridLanguage.loadWeights(into: bare.decoder, fromDirectory: base, precision: .float32)
        bare.head.update(parameters: loaded.head.parameters())
        XCTAssertNotEqual(bare.score(tokens: tokens).item(Float.self), score.item(Float.self))
    }

    func testAFineTuneMovesOnlyTheAdapterAndHeadAndReloadsThroughTheFactory() throws {
        try requireMLXRuntime()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("openjev-tune-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let base = root.appendingPathComponent("base")
        let release = root.appendingPathComponent("release")
        let tuned = root.appendingPathComponent("tuned")
        try writeTinyBase(to: base)
        _ = try writeTinyCheckpoint(to: release)

        let tokenizer = NFKMLXDecisionTokenizer.words
        let model = NFKMLXOpenJev(net: try NFKMLXOpenJev.network(checkpointDirectoryURL: release, baseDirectoryURL: base,
                                                                 precision: .float32), tokenizer: tokenizer)
        let examples = [NFKMLXOpenJevExample(state: "charged twice", question: intent, label: 0),
                        NFKMLXOpenJevExample(state: "a crash", question: intent, label: 1),
                        NFKMLXOpenJevExample(state: "refund me", question: refund, holds: true)]
        let embedding = model.net.decoder.model.embedTokens.weight.asArray(Float.self)
        let losses = try model.fineTune(examples: examples, steps: 6, learningRate: 1e-2, headLearningRate: 1e-2, batchSize: 3)
        XCTAssertEqual(losses.count, 6)
        XCTAssertLessThan(losses.suffix(2).reduce(0, +), losses.prefix(2).reduce(0, +), "the loss falls")
        XCTAssertEqual(model.net.decoder.model.embedTokens.weight.asArray(Float.self), embedding, "the base stays as released")

        try model.save(to: tuned)
        let reloaded = NFKMLXOpenJev(net: try NFKMLXOpenJev.network(checkpointDirectoryURL: tuned, baseDirectoryURL: base,
                                                                    precision: .float32), tokenizer: tokenizer)
        for question in [intent, refund] {
            let before = try model.logits(state: "charged twice", question: question)
            let after = try reloaded.logits(state: "charged twice", question: question)
            for (a, b) in zip(before, after) { XCTAssertEqual(a, b, accuracy: 1e-5) }
        }
        XCTAssertThrowsError(try model.fineTune(examples: [NFKMLXOpenJevExample(state: "s", question: intent, target: [1, 0])],
                                                steps: 1), "a target needs one entry per candidate")
    }

    // A base without a tokenizer could score no candidate, so the model factory refuses it.
    func testABaseWithoutATokenizerIsRefused() throws {
        try requireMLXRuntime()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("openjev-notok-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try writeTinyBase(to: root.appendingPathComponent("base"))
        _ = try writeTinyCheckpoint(to: root.appendingPathComponent("release"))
        XCTAssertThrowsError(try NFKMLXOpenJev.openJev(checkpointDirectoryURL: root.appendingPathComponent("release"),
                                                       baseDirectoryURL: root.appendingPathComponent("base"),
                                                       precision: .float32))
    }

    // MARK: - Downloading

    func testADownloadFetchesTheReleaseThenTheBaseRevisionItNames() throws {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("openjev-hub-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        let hub = NFKOpenJevReleaseHub()
        hub.cacheDirectoryURL = cache
        let variant = NFKMLXOpenJevVariant.twoB
        let release = try NFKMLXOpenJev.download(variant: variant, revision: nil, hub: hub)
        XCTAssertEqual(release.checkpointDirectoryURL.standardizedFileURL.path,
                       cache.appendingPathComponent("\(variant.repository)/\(variant.measuredRevision)/package/checkpoint").standardizedFileURL.path)
        XCTAssertEqual(release.baseDirectoryURL.standardizedFileURL.path,
                       cache.appendingPathComponent("\(variant.baseRepository)/\(variant.baseRevision)").standardizedFileURL.path)
        XCTAssertTrue(hub.fetched.contains("/\(variant.baseRepository)/resolve/\(variant.baseRevision)/model-a.safetensors"),
                      "the shards come from the index")
        XCTAssertEqual(hub.fetched.count, NFKMLXOpenJev.checkpointFiles.count + NFKMLXOpenJev.baseFiles.count + 2)
    }

    // MARK: - The 9B and 27B releases' structure

    /// The tensor shapes a local safetensors file's header names, without reading its data.
    private func headerShapes(_ url: URL) throws -> [String: [Int]] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let length = try XCTUnwrap(handle.readData(ofLength: 8)).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        let header = try JSONSerialization.jsonObject(with: handle.readData(ofLength: Int(length))) as? [String: Any] ?? [:]
        return header.compactMapValues { ($0 as? [String: Any])?["shape"] as? [Int] }
    }

    /// Every adapter tensor of a release lands on a projection of its own geometry at the shape PEFT
    /// stores, and every decoder parameter is among the base's `released` shapes. Nothing is loaded:
    /// the net is built lazily, which is what lets a large geometry be checked on any machine.
    private func assertStructure(name: String, checkpoint checkpointURL: URL, baseConfigurationDirectory: URL,
                                 released: [String: [Int]], hiddenSize: Int, adapterTensors: Int) throws {
        let configuration = try NFKMLXOpenJevConfiguration(checkpointDirectoryURL: checkpointURL,
                                                           baseDirectoryURL: baseConfigurationDirectory)
        XCTAssertEqual(configuration.decoder.hiddenSize, hiddenSize)
        XCTAssertEqual(configuration.modelName, name)
        let net = NFKMLXOpenJevNet(configuration)
        try net.adapt()
        let built = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1.shape) })

        let adapter = try headerShapes(checkpointURL.appendingPathComponent("adapter/adapter_model.safetensors"))
        var mismatched = [String]()
        for (key, shape) in adapter {
            guard let name = NFKMLXOpenJev.adapterKey(forReference: key), let ours = built[name] else {
                mismatched.append("\(key): no adapted projection"); continue
            }
            if Array(ours.reversed()) != shape { mismatched.append("\(key): built \(ours), released \(shape)") }
        }
        let adapted = built.keys.filter { $0.hasSuffix(".lora_a") || $0.hasSuffix(".lora_b") }
        XCTAssertEqual(adapted.count, adapter.count, "every adapted projection has its tensors")
        XCTAssertEqual(adapter.count, adapterTensors, "four in each recurrent layer and eight in each attention layer")

        var checked = 0
        for (name, shape) in built where name.hasPrefix("decoder.") && !name.hasSuffix(".lora_a") && !name.hasSuffix(".lora_b") {
            let key = NFKMLXHybridLanguage.referenceKey(for: String(name.dropFirst("decoder.".count)))
            guard let expected = released[key] else { mismatched.append("\(key): absent from the base"); continue }
            let ours = key.hasSuffix("conv1d.weight") ? [shape[0], shape[2], shape[1]] : shape
            if ours != expected { mismatched.append("\(key): built \(ours), released \(expected)") }
            checked += 1
        }
        print("VALIDATION structure \(name): \(adapter.count) adapter tensors, \(checked) base parameters, \(mismatched.count) mismatched")
        XCTAssertEqual(mismatched, [])
        XCTAssertGreaterThan(checked, 400)
    }

    func testTheNineBillionAdapterAndBaseMatchTheBuiltModel() throws {
        try requireMLXRuntime()
        guard let checkpoint = config["IK_VAL_OPEN_JEV_9B"], let base = config["IK_VAL_QWEN3_5_9B"] else {
            throw XCTSkip("set IK_VAL_OPEN_JEV_9B and IK_VAL_QWEN3_5_9B")
        }
        let baseURL = URL(fileURLWithPath: base)
        let index = try JSONSerialization.jsonObject(with: Data(contentsOf: baseURL.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any]
        var released = [String: [Int]]()
        for shard in Set(((index?["weight_map"] as? [String: String]) ?? [:]).values) {
            released.merge(try headerShapes(baseURL.appendingPathComponent(shard))) { first, _ in first }
        }
        try assertStructure(name: "open-jev-9b", checkpoint: URL(fileURLWithPath: checkpoint),
                            baseConfigurationDirectory: baseURL, released: released, hiddenSize: 4096, adapterTensors: 160)
    }

    // The 27B base (about 54 GB) is not downloaded: its configuration and the shapes its shard headers
    // name, fetched at the revision the release pins, stand in for it.
    func testTheTwentySevenBillionAdapterAndBaseMatchTheBuiltModel() throws {
        try requireMLXRuntime()
        guard let checkpoint = config["IK_VAL_OPEN_JEV_27B"], let shapes = config["IK_SHAPES_QWEN3_8"],
              let baseConfiguration = config["IK_CONFIG_QWEN3_8"] else {
            throw XCTSkip("set IK_VAL_OPEN_JEV_27B, IK_SHAPES_QWEN3_8, and IK_CONFIG_QWEN3_8")
        }
        let released = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: shapes))) as? [String: [Int]])
        try assertStructure(name: "open-jev-27b", checkpoint: URL(fileURLWithPath: checkpoint),
                            baseConfigurationDirectory: URL(fileURLWithPath: baseConfiguration).deletingLastPathComponent(),
                            released: released, hiddenSize: 5120, adapterTensors: 320)
    }

    // MARK: - Reference parity (Open-Jev-2B, float32)

    private static let requests: [(state: Any, questions: [(String, NFKDecisionQuestion)])] = [
        (["customer": "Dana", "message": "I was charged twice for order 88213 and want my money back.", "plan": "Pro", "vip": true] as [String: Any],
         [("intent", .choiceQuestion(withInstructions: "Choose the customer intent.", options: ["billing", "technical", "other"],
                                     descriptions: ["billing": "A payment or refund issue", "technical": "A malfunction or setup issue"])),
          ("urgency", .scoreQuestion(withInstructions: "How urgent is the request?", levels: ["not urgent", "somewhat urgent", "very urgent"])),
          ("refund", .noulQuestion(withInstructions: "The customer asks for a refund.", trueMeaning: "the message requests money back",
                                   falseMeaning: "no refund is requested"))]),
        ("Café au lait costs €4.50 at the 2nd location; the rese\u{0301}rvation for 12 people is at 19:30.",
         [("topic", .choiceQuestion(withInstructions: "What is the text about?", options: ["food", "travel", "sports", "finance"])),
          ("numbers", .noulQuestion(withInstructions: "The text mentions a time of day."))]),
    ]

    private func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) { dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y) }
        return dot / max(sqrt(na * nb), 1e-30)
    }

    // The 9B release does not fit here at float32 (about 36 GB), so it is measured at the precision the
    // loader runs it at: bfloat16 against bfloat16. Both sides round every layer, so the agreement is
    // bounded by bfloat16 rather than by the port; the decisions are what must agree.
    func testTheNineBillionReleaseMatchesTheReferenceAtBFloat16() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_OPEN_JEV_9B_BF16"], let checkpoint = config["IK_VAL_OPEN_JEV_9B"],
              let base = config["IK_VAL_QWEN3_5_9B"] else {
            throw XCTSkip("set IK_PARITY_OPEN_JEV_9B_BF16, IK_VAL_OPEN_JEV_9B, and IK_VAL_QWEN3_5_9B")
        }
        let arrays = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let model = try NFKMLXOpenJev.openJev(checkpointDirectoryURL: URL(fileURLWithPath: checkpoint),
                                              baseDirectoryURL: URL(fileURLWithPath: base))
        var report = [String]()
        var index = 0
        var worstLogit: Float = 0, worstProbability = 0.0
        for request in Self.requests {
            for (_, question) in request.questions {
                let key = "r\(index)"
                let candidates = try model.candidateTokens(state: request.state, question: question)
                for (c, tokens) in candidates.enumerated() {
                    XCTAssertEqual(tokens, arrays["\(key).c\(c).ids"]!.asArray(Int32.self).map(Int.init), "\(key) candidate \(c) tokens")
                }
                let logits = try model.logits(state: request.state, question: question)
                let reference = arrays["\(key).logits"]!.asArray(Float.self)
                let gap = zip(logits, reference).map { abs($0 - $1) }.max() ?? 0
                let probabilities = try model.distribution(state: request.state, question: question)
                let referenceProbabilities = arrays["\(key).probabilities"]!.asArray(Float.self).map(Double.init)
                let probabilityGap = zip(probabilities, referenceProbabilities).map { abs($0 - $1) }.max() ?? 0
                worstLogit = max(worstLogit, gap)
                worstProbability = max(worstProbability, probabilityGap)
                report.append("\(key): logits \(logits) vs \(reference), max gap \(gap), probability gap \(probabilityGap)")
                let mine = probabilities.indices.max { probabilities[$0] < probabilities[$1] }
                let theirs = referenceProbabilities.indices.max { referenceProbabilities[$0] < referenceProbabilities[$1] }
                XCTAssertEqual(mine, theirs, "\(key) decides the same")
                index += 1
            }
        }
        print("VALIDATION PARITY open-jev-9b (bfloat16):\n  " + report.joined(separator: "\n  "))
        // Measured: logits within 0.14 and probabilities within 0.0037 over the five questions.
        XCTAssertLessThan(worstLogit, 0.3)
        XCTAssertLessThan(worstProbability, 0.01)
    }

    func testTheTwoBillionReleaseMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_OPEN_JEV_2B"], let checkpoint = config["IK_VAL_OPEN_JEV_2B"],
              let base = config["IK_VAL_QWEN3_5_2B"] else {
            throw XCTSkip("set IK_PARITY_OPEN_JEV_2B, IK_VAL_OPEN_JEV_2B, and IK_VAL_QWEN3_5_2B")
        }
        let arrays = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let model = try NFKMLXOpenJev.openJev(checkpointDirectoryURL: URL(fileURLWithPath: checkpoint),
                                              baseDirectoryURL: URL(fileURLWithPath: base), precision: .float32)
        XCTAssertEqual(model.configuration.temperature, arrays["temperature"]!.item(Float.self))
        var report = [String]()
        var index = 0
        for request in Self.requests {
            for (_, question) in request.questions {
                let key = "r\(index)"
                let candidates = try model.candidateTokens(state: request.state, question: question)
                for (c, tokens) in candidates.enumerated() {
                    XCTAssertEqual(tokens, arrays["\(key).c\(c).ids"]!.asArray(Int32.self).map(Int.init), "\(key) candidate \(c) tokens")
                }
                if index == 0 {
                    let states = model.net.decoder.hiddenStates(MLXArray(candidates[0].map(Int32.init)).reshaped([1, -1]))
                    var worst = 1.0
                    for (layer, state) in states.enumerated() {
                        let similarity = cosine(state[0].asArray(Float.self), arrays["hidden.\(layer)"]!.asArray(Float.self))
                        worst = min(worst, similarity)
                        XCTAssertGreaterThan(similarity, 0.99999, "hidden state \(layer)")
                    }
                    report.append("worst hidden-state cosine \(worst) over \(states.count)")
                }
                let logits = try model.logits(state: request.state, question: question)
                let reference = arrays["\(key).logits"]!.asArray(Float.self)
                let gap = zip(logits, reference).map { abs($0 - $1) }.max() ?? 0
                report.append("\(key) \(question.type == .noul ? "noul" : question.type == .score ? "score" : "choice"): logits \(logits) vs \(reference), max gap \(gap)")
                XCTAssertLessThan(gap, 5e-3, "\(key) logits")

                let probabilities = try model.distribution(state: request.state, question: question)
                let referenceProbabilities = arrays["\(key).probabilities"]!.asArray(Float.self)
                for (a, b) in zip(probabilities, referenceProbabilities) { XCTAssertEqual(a, Double(b), accuracy: 1e-3) }
                let answer = NFKMLXOpenJev.answer(question: question, probabilities: probabilities)
                switch question.type {
                case .choice:
                    XCTAssertEqual(answer.choice, question.options[Int(arrays["\(key).choice"]!.item(Int32.self))])
                    XCTAssertEqual(answer.confidence, Double(arrays["\(key).confidence"]!.item(Float.self)), accuracy: 1e-3)
                case .score:
                    XCTAssertEqual(answer.score, Double(arrays["\(key).score"]!.item(Float.self)), accuracy: 1e-3)
                    XCTAssertEqual(answer.confidence, Double(arrays["\(key).confidence"]!.item(Float.self)), accuracy: 1e-3)
                default:
                    XCTAssertEqual(answer.probability, Double(arrays["\(key).noul"]!.item(Float.self)), accuracy: 1e-3)
                }

                let loss = NFKMLXOpenJevObjective().loss(logits: MLXArray(reference), target: arrays["loss.\(key).target"]!)
                XCTAssertEqual(loss.item(Float.self), arrays["loss.\(key)"]!.item(Float.self), accuracy: 1e-5, "\(key) loss")
                index += 1
            }
        }
        print("VALIDATION PARITY open-jev-2b:\n  " + report.joined(separator: "\n  "))
    }
}

/// A hub whose transport writes a placeholder, and a two-shard index for the base, recording each path.
final class NFKOpenJevReleaseHub: NFKHFHub {
    private(set) var fetched = [String]()

    override func fetch(_ remoteURL: URL, toFileURL destinationURL: URL) throws {
        fetched.append(remoteURL.path)
        try FileManager.default.createDirectory(at: destinationURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let body: [String: Any]
        switch remoteURL.lastPathComponent {
        case "model.json": body = ["revision": NFKMLXOpenJevVariant.twoB.baseRevision]
        case "model.safetensors.index.json": body = ["weight_map": ["a": "model-a.safetensors", "b": "model-b.safetensors", "c": "model-a.safetensors"]]
        default: body = [:]
        }
        try JSONSerialization.data(withJSONObject: body).write(to: destinationURL)
    }
}

extension NFKMLXDecisionTokenizer {
    /// Maps each whitespace-separated word to an id inside the tiny vocabulary, which keeps a tiny
    /// model's prompts short.
    static let words = NFKMLXDecisionTokenizer { text in
        text.split(whereSeparator: \.isWhitespace).map { word in
            word.unicodeScalars.reduce(7) { ($0 * 31 + Int($1.value)) % 500 } + 5
        }
    }
}
