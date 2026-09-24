//
//  NFKMLXLayaTests.swift
//  InferKitMLXTests
//
//  Weight-free tests for Laya: the prompt the reference builds, the state serialization, the answer
//  shapes, the backend's reading of the contract keys, the release configuration reader, and the
//  customization path (objective, freezing, a fine-tune that moves what it should). Numeric parity
//  against the release lives in NFKMLXReferenceParityTests; the objective's parity is here, gated on
//  its record.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLayaTests: XCTestCase {

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

    /// A tokenizer that maps each character to an id inside the tiny vocabulary and above its special
    /// tokens, so a prompt's layout is readable in the ids.
    private let characterTokenizer = NFKMLXLayaTokenizer { text in
        text.unicodeScalars.map { Int($0.value % 500) + 8 }
    }

    private func tinyLaya() throws -> NFKMLXLaya {
        NFKMLXRandom.seed(9)
        return try NFKMLXLaya.laya(weightsURL: nil, tokenizer: characterTokenizer, configuration: .tiny)
    }

    private var department: NFKDecisionQuestion {
        NFKDecisionQuestion.choiceQuestion(withInstructions: "Which team?", options: ["billing", "technical", "sales"],
                                           descriptions: ["billing": "money", "technical": "bugs"])
    }
    private var severity: NFKDecisionQuestion {
        NFKDecisionQuestion.scoreQuestion(withInstructions: "How severe?", levels: ["low", "medium", "high"])
    }
    private var urgent: NFKDecisionQuestion {
        NFKDecisionQuestion.noulQuestion(withInstructions: "Needs an answer today.")
    }

    // MARK: - The prompt

    func testThePromptFollowsTheReferenceLayout() {
        let c = NFKMLXLayaConfiguration.tiny
        let prompt = NFKMLXLayaPrompt(state: "help", question: department, tokenizer: characterTokenizer, configuration: c)
        XCTAssertEqual(prompt.tokens.first, c.encoder.clsToken)
        XCTAssertEqual(prompt.tokens.last, c.encoder.sepToken)
        XCTAssertEqual(prompt.markers.count, 3)
        for marker in prompt.markers {
            XCTAssertEqual(prompt.tokens[marker], c.maskToken, "every option sits at a mask token")
        }
        XCTAssertEqual(prompt.options, ["billing: money", "technical: bugs", "sales"],
                       "a described option renders as name: description, an undescribed one as its name")
        XCTAssertEqual(prompt.tokens[prompt.markers[0] - 1], c.encoder.sepToken, "the options follow the head's separator")
        XCTAssertLessThanOrEqual(prompt.tokens.count, c.maxLength)

        let noul = NFKMLXLayaPrompt(state: "help", question: urgent, tokenizer: characterTokenizer, configuration: c)
        XCTAssertEqual(noul.markers.count, 2, "a noul is always false then true")
        XCTAssertTrue(noul.options[0].hasPrefix("false: no, the statement"))
        XCTAssertTrue(noul.options[1].hasPrefix("true: yes, the statement"))

        let score = NFKMLXLayaPrompt(state: "help", question: severity, tokenizer: characterTokenizer, configuration: c)
        XCTAssertEqual(score.options, ["level 0: low", "level 1: medium", "level 2: high"])
    }

    func testTheStateIsCutToTheBudgetAndTheOptionsShrinkEvenly() {
        let c = NFKMLXLayaConfiguration.tiny
        let long = String(repeating: "a very long state ", count: 40)
        let prompt = NFKMLXLayaPrompt(state: long, question: urgent, tokenizer: characterTokenizer, configuration: c)
        XCTAssertEqual(prompt.tokens.count, c.maxLength, "the state fills the room and no more")
        XCTAssertEqual(prompt.tokens.last, c.encoder.sepToken, "the closing separator survives the cut")
        XCTAssertEqual(prompt.markers.count, 2)

        // Ten options at the tiny head budget of 32 tokens leave under 16 for the head, so every option
        // shrinks to an even share and the instructions keep at least 8 tokens.
        let many = NFKDecisionQuestion.choiceQuestion(withInstructions: "Pick one of the many options here",
                                                      options: (0 ..< 10).map { "option number \($0)" })
        let crowded = NFKMLXLayaPrompt(state: "s", question: many, tokenizer: characterTokenizer, configuration: c)
        XCTAssertEqual(crowded.markers.count, 10)
        let optionWidths = zip(crowded.markers, crowded.markers.dropFirst()).map { $1 - $0 }
        XCTAssertEqual(Set(optionWidths).count, 1, "the options shrank to one width")
        XCTAssertGreaterThanOrEqual(crowded.markers[0] - 2, 8, "the instructions keep at least 8 tokens")
    }

    func testAStateIsSerializedTheWayPythonWritesJSON() {
        XCTAssertEqual(NFKMLXLayaPrompt.serialize(state: "plain text"), "plain text")
        let record: [String: Any] = ["b": [true, NSNull(), "x\"y"], "a": 1, "c": 2.5, "d": ["z": -3]]
        XCTAssertEqual(NFKMLXLayaPrompt.serialize(state: record),
                       "{\"a\": 1, \"b\": [true, null, \"x\\\"y\"], \"c\": 2.5, \"d\": {\"z\": -3}}")
        XCTAssertEqual(NFKMLXLayaPrompt.serialize(state: ["é", 3.0]), "[\"é\", 3.0]", "non-ASCII stays, a whole float keeps its .0")
    }

    func testTheTemperatureBucketsFollowTheReference() {
        XCTAssertEqual(NFKMLXLayaConfiguration.temperatureBucket(type: .choice, optionCount: 2), "choice:2")
        XCTAssertEqual(NFKMLXLayaConfiguration.temperatureBucket(type: .choice, optionCount: 5), "choice:3-5")
        XCTAssertEqual(NFKMLXLayaConfiguration.temperatureBucket(type: .score, optionCount: 10), "score:6-10")
        XCTAssertEqual(NFKMLXLayaConfiguration.temperatureBucket(type: .noul, optionCount: 2), "noul:2")
        XCTAssertEqual(NFKMLXLayaConfiguration.temperatureBucket(type: .choice, optionCount: 40), "choice:11+")
        let large = NFKMLXLayaConfiguration.large
        XCTAssertEqual(large.temperature(type: .choice, optionCount: 6), 1.0000158548355103, accuracy: 1e-7)
        XCTAssertEqual(large.temperature(type: .score, optionCount: 7), 1.2514300346374512, accuracy: 1e-7,
                       "an unbucketed cardinality falls back to the type's temperature")
        XCTAssertEqual(NFKMLXLayaConfiguration.multilingual.temperature(type: .noul, optionCount: 2), 1)
    }

    // MARK: - The answers

    func testEachAnswerTypeCarriesItsFieldsWithProbabilitiesThatSumToOne() throws {
        try requireMLXRuntime()
        let laya = try tinyLaya()
        let answers = laya.decide(state: "My payouts are failing.", questions: ["department": department, "severity": severity, "urgent": urgent])
        XCTAssertEqual(answers.count, 3)

        let choice = try XCTUnwrap(answers["department"])
        XCTAssertEqual(choice.type, .choice)
        XCTAssertTrue(["billing", "technical", "sales"].contains(choice.choice ?? ""))
        let mass = choice.probabilities!.values.reduce(0) { $0 + $1.doubleValue }
        XCTAssertEqual(mass, 1, accuracy: 1e-5)
        XCTAssertEqual(choice.probabilities?[choice.choice!]?.doubleValue, choice.probabilities!.values.map(\.doubleValue).max())
        XCTAssertTrue((0 ... 1).contains(choice.confidence))
        let act = try XCTUnwrap((choice.raw["rl_agent"] as? [String: Any])?["act_probability"] as? Double)
        XCTAssertTrue((0 ... 1).contains(act), "the act-or-escalate head reports a probability")

        let score = try XCTUnwrap(answers["severity"])
        XCTAssertEqual(score.type, .score)
        XCTAssertTrue((0 ... 2).contains(score.score))
        XCTAssertEqual(score.legend, ["0": "low", "1": "medium", "2": "high"])
        XCTAssertEqual(score.probabilities?.count, 3)

        let noul = try XCTUnwrap(answers["urgent"])
        XCTAssertEqual(noul.type, .noul)
        XCTAssertTrue((0 ... 1).contains(noul.probability))
        XCTAssertNil(noul.probabilities)
    }

    func testWithoutATokenizerEveryAnswerIsUniform() throws {
        try requireMLXRuntime()
        let laya = try NFKMLXLaya.laya(weightsURL: nil, tokenizer: nil, configuration: .tiny)
        let answer = laya.answer(state: "s", question: department)
        for probability in answer.probabilities!.values {
            XCTAssertEqual(probability.doubleValue, 1.0 / 3, accuracy: 1e-6)
        }
    }

    func testTheSameQuestionAnswersTheSameTwice() throws {
        try requireMLXRuntime()
        let laya = try tinyLaya()
        let first = laya.distribution(state: "the state", question: severity)
        let second = laya.distribution(state: "the state", question: severity)
        XCTAssertEqual(first.probabilities, second.probabilities)
        XCTAssertEqual(first.actProbability, second.actProbability)
    }

    // MARK: - The backend

    func testTheBackendReadsTheContractKeysAndAnswersInTheHostedShape() throws {
        try requireMLXRuntime()
        let backend = try tinyLaya().makeBackend()
        XCTAssertEqual(backend.backendIdentifier, "mlx-laya")
        XCTAssertTrue(backend.isReady)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputState, NFKInputQuestions, NFKInputPrompt, NFKInputMessages])
        XCTAssertTrue(backend.supportedParameterKeys.isEmpty)

        // A question already in the wire shape passes through the same reader the core's dictionary
        // initializer provides; an NFKDecisionQuestion beside it is read as it is.
        let wire: [String: Any] = ["type": "noul", "instructions": "Is it urgent?"]
        let request = NFKInferenceRequest(inputs: [NFKInputState: ["subject": "Refund", "tags": ["billing"]],
                                                   NFKInputQuestions: ["urgent": wire, "department": department]])
        let result = try backend.runInference(for: request)
        XCTAssertEqual(result.answers?.count, 2)
        XCTAssertEqual(result.answers?["urgent"]?.type, .noul)
        let reply = try XCTUnwrap(result.structured)
        XCTAssertEqual(reply["model"] as? String, "laya-tiny")
        XCTAssertEqual((reply["answers"] as? [String: Any])?.count, 2)
        let usage = try XCTUnwrap(result.output(forKey: NFKOutputUsage) as? [String: Int])
        XCTAssertGreaterThan(usage[NFKUsageInputTokens] ?? 0, 0)
        XCTAssertEqual(usage[NFKUsageOutputTokens], 0)

        // The prompt stands in for an absent state; a request with neither is refused before a pass.
        let prompted = NFKInferenceRequest(inputs: [NFKInputPrompt: "text", NFKInputQuestions: ["u": urgent]])
        XCTAssertNoThrow(try backend.runInference(for: prompted))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputQuestions: ["u": urgent]]))) { error in
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_InferenceMissingInput.rawValue)
        }
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputState: "s"])))
    }

    // MARK: - The release configuration

    func testAReleaseConfigurationIsReadFromItsThreeFiles() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("laya-config-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("encoder"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("tokenizer"), withIntermediateDirectories: true)
        let encoder: [String: Any] = ["hidden_size": 768, "num_hidden_layers": 22, "num_attention_heads": 12,
                                      "intermediate_size": 1152, "vocab_size": 256000, "global_attn_every_n_layers": 3,
                                      "local_attention": 128, "norm_eps": 1e-5, "cls_token_id": 1, "sep_token_id": 1,
                                      "mask_token_id": 4, "pad_token_id": 0,
                                      "rope_parameters": ["full_attention": ["rope_theta": 160000],
                                                          "sliding_attention": ["rope_theta": 160000]]]
        let agent: [String: Any] = ["head_layers": 2, "max_len": 1024, "head_max_len": 256, "model_name": "rl-agent",
                                    "temperature": [1.5, 1.25, 2.0], "temperature_by_options": ["noul:2": 1.75]]
        try JSONSerialization.data(withJSONObject: encoder).write(to: directory.appendingPathComponent("encoder/config.json"))
        try JSONSerialization.data(withJSONObject: agent).write(to: directory.appendingPathComponent("rl_agent_config.json"))
        try JSONSerialization.data(withJSONObject: ["mask_token": "<mask>", "cls_token": "<bos>", "sep_token": "<eos>", "pad_token": "<pad>"])
            .write(to: directory.appendingPathComponent("tokenizer/tokenizer_config.json"))
        let tokenizerModel: [String: Any] = ["added_tokens": [["id": 0, "content": "<pad>"], ["id": 1, "content": "<eos>"],
                                                              ["id": 2, "content": "<bos>"], ["id": 4, "content": "<mask>"]],
                                             "model": ["vocab": ["a": 10]]]
        try JSONSerialization.data(withJSONObject: tokenizerModel).write(to: directory.appendingPathComponent("tokenizer/tokenizer.json"))

        let c = try NFKMLXLayaConfiguration(directoryURL: directory)
        XCTAssertEqual(c.encoder.hiddenSize, 768)
        XCTAssertEqual(c.encoder.layerCount, 22)
        XCTAssertEqual(c.encoder.localRopeTheta, 160_000, "the sliding base comes from rope_parameters, not the 10000 default")
        XCTAssertEqual(c.encoder.clsToken, 2, "the classifier token is the tokenizer's <bos>, not the encoder config's cls_token_id")
        XCTAssertEqual(c.encoder.sepToken, 1)
        XCTAssertEqual(c.maskToken, 4)
        XCTAssertEqual(c.padToken, 0)
        XCTAssertEqual(c.maxLength, 1024)
        XCTAssertEqual(c.headMaxLength, 256)
        XCTAssertEqual(c.temperatures, [1.5, 1.25, 2.0])
        XCTAssertEqual(c.temperature(type: .noul, optionCount: 2), 1.75)
        XCTAssertEqual(c.temperature(type: .choice, optionCount: 3), 1.5)
        XCTAssertEqual(c.maskLiteral, "<mask>")
        XCTAssertEqual(c.modelName, "rl-agent")
    }

    // MARK: - Downloading a release

    func testEachVariantNamesItsFiveFilesUnderItsFolder() {
        let root = NFKMLXLaya.releaseFiles(for: .root)
        XCTAssertEqual(root, ["rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer_config.json",
                              "tokenizer/tokenizer.json", "model.safetensors"])
        XCTAssertEqual(NFKMLXLaya.releaseFiles(for: .typedDecisions), root.map { "typed-decisions/" + $0 })
        XCTAssertEqual(NFKMLXLaya.releaseFiles(for: .multilingual), root.map { "multilingual/" + $0 })
    }

    func testADownloadFetchesTheVariantOnceAndReturnsItsFolder() throws {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("laya-hub-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        let hub = NFKLayaRecordingHub()
        hub.cacheDirectoryURL = cache
        let revision = NFKMLXLaya.measuredRevision

        let folder = try NFKMLXLaya.download(variant: .multilingual, revision: revision, hub: hub)
        XCTAssertEqual(folder.standardizedFileURL.path,
                       cache.appendingPathComponent("convaiinnovations/laya/\(revision)/multilingual").standardizedFileURL.path)
        XCTAssertEqual(hub.fetched, NFKMLXLaya.releaseFiles(for: .multilingual).map {
            "/convaiinnovations/laya/resolve/\(revision)/\($0)"
        })
        for path in NFKMLXLaya.releaseFiles(for: .root) {
            XCTAssertTrue(FileManager.default.fileExists(atPath: folder.appendingPathComponent(path).path), path)
        }

        _ = try NFKMLXLaya.download(variant: .multilingual, revision: revision, hub: hub)
        XCTAssertEqual(hub.fetched.count, 5, "a cached variant is not fetched again")
    }

    // The released root, placed in a hub cache the way a download lays it out, builds through the
    // variant factory and answers exactly as the directory factory does.
    func testACachedReleaseBuildsThroughTheVariantFactory() throws {
        try requireMLXRuntime()
        guard let root = config["IK_VAL_LAYA"] else { throw XCTSkip("set IK_VAL_LAYA") }
        let store = URL(fileURLWithPath: root)
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("laya-cache-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        let snapshot = cache.appendingPathComponent("\(NFKMLXLaya.repository)/\(NFKMLXLaya.measuredRevision)")
        for path in NFKMLXLaya.releaseFiles(for: .root) {
            let link = snapshot.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: store.appendingPathComponent(path))
        }

        let downloaded = try NFKMLXLaya.laya(variant: .root, revision: NFKMLXLaya.measuredRevision, cacheDirectoryURL: cache)
        let direct = try NFKMLXLaya.laya(directoryURL: store)
        let question = NFKDecisionQuestion.choiceQuestion(withInstructions: "Which team should handle this?",
                                                          options: ["billing", "technical", "sales"])
        let state = "I was charged twice for one order."
        XCTAssertEqual(downloaded.distribution(state: state, question: question).probabilities,
                       direct.distribution(state: state, question: question).probabilities)
    }

    // MARK: - The structure

    func testTheNetworkCarriesTheCheckpointsKeys() throws {
        try requireMLXRuntime()
        let net = NFKMLXLayaNet(.tiny)
        let keys = Set(net.parameters().flattened().map(\.0))
        for expected in ["encoder.embeddings.tok_embeddings.weight", "encoder.layers.1.attn_norm.weight",
                         "encoder.final_norm.weight", "head.layers.0.self_attn.in_proj_weight",
                         "head.layers.1.self_attn.out_proj.bias", "head.layers.0.linear1.weight", "head.layers.0.norm2.bias",
                         "type_emb.weight", "scorer.0.weight", "scorer.1.bias", "scorer.3.weight",
                         "act_head.0.weight", "act_head.2.bias", "temperature"] {
            XCTAssertTrue(keys.contains(expected), "\(expected) is missing")
        }
        XCTAssertFalse(keys.contains("encoder.layers.0.attn_norm.weight"), "layer 0's attention norm is the identity")
        XCTAssertFalse(keys.contains("scorer.2.weight"), "the GELU carries no parameter")
    }

    // MARK: - Customization

    func testTheObjectiveIsLowestAtTheTargetAndPenalizesAnOrdinalMiss() throws {
        try requireMLXRuntime()
        let objective = NFKMLXLayaObjective()
        let target: [Float] = [0, 0, 1]
        let confident = objective.loss(logits: MLXArray([Float(-5), -5, 5]), target: target, type: .choice)
        let wrong = objective.loss(logits: MLXArray([Float(5), -5, -5]), target: target, type: .choice)
        let flat = objective.loss(logits: MLXArray([Float(0), 0, 0]), target: target, type: .choice)
        eval(confident, wrong, flat)
        XCTAssertLessThan(confident.item(Float.self), flat.item(Float.self))
        XCTAssertLessThan(flat.item(Float.self), wrong.item(Float.self))

        // On a score, missing by two levels costs more than missing by one: the ranked probability
        // score reads the cumulative distributions.
        let byOne = objective.loss(logits: MLXArray([Float(-5), 5, -5]), target: target, type: .score)
        let byTwo = objective.loss(logits: MLXArray([Float(5), -5, -5]), target: target, type: .score)
        let sameAsChoice = objective.loss(logits: MLXArray([Float(5), -5, -5]), target: target, type: .choice)
        eval(byOne, byTwo, sameAsChoice)
        XCTAssertLessThan(byOne.item(Float.self), byTwo.item(Float.self))
        XCTAssertGreaterThan(byTwo.item(Float.self), sameAsChoice.item(Float.self), "the ordinal penalty applies to a score only")
    }

    // The objective against the reference's own `rl_common.proper_reward` on identical tensors.
    func testTheObjectiveMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_LAYA_LOSS"] else {
            throw XCTSkip("set IK_PARITY_LAYA_LOSS (run_reference.py laya_loss)")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let objective = NFKMLXLayaObjective()
        let reward = objective.reward(logits: try XCTUnwrap(arrays["logits"]), targets: try XCTUnwrap(arrays["targets"]),
                                      mask: try XCTUnwrap(arrays["mask"]).asType(.bool), types: try XCTUnwrap(arrays["types"]))
        let loss = objective.loss(logits: try XCTUnwrap(arrays["logits"]), targets: try XCTUnwrap(arrays["targets"]),
                                  mask: try XCTUnwrap(arrays["mask"]).asType(.bool), types: try XCTUnwrap(arrays["types"]))
        eval(reward, loss)
        let mine = reward.asArray(Float.self)
        let theirs = try XCTUnwrap(arrays["reward"]).asArray(Float.self)
        print("VALIDATION PARITY laya_loss: rewards \(mine) vs \(theirs), loss \(loss.item(Float.self)) vs \(try XCTUnwrap(arrays["loss"]).item(Float.self))")
        XCTAssertEqual(mine.count, theirs.count)
        for (a, b) in zip(mine, theirs) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
        XCTAssertEqual(loss.item(Float.self), try XCTUnwrap(arrays["loss"]).item(Float.self), accuracy: 1e-5)
    }

    func testAHeadFineTuneLowersTheLossAndLeavesTheEncoderAsReleased() throws {
        try requireMLXRuntime()
        let laya = try tinyLaya()
        let examples = [
            NFKMLXLayaExample(state: "My invoice is wrong.", question: department, label: 0),
            NFKMLXLayaExample(state: "The app crashes on launch.", question: department, label: 1),
            NFKMLXLayaExample(state: "Everything is down!", question: severity, label: 2),
            NFKMLXLayaExample(state: "I need this today.", question: urgent, holds: true),
        ]
        func snapshot(_ key: String) -> [Float] {
            let parameters = Dictionary(uniqueKeysWithValues: laya.net.parameters().flattened())
            let value = parameters[key]!
            eval(value)
            return value.asArray(Float.self)
        }
        let encoderBefore = snapshot("encoder.layers.1.attn.Wqkv.weight")
        let scorerBefore = snapshot("scorer.1.weight")
        let actBefore = snapshot("act_head.0.weight")
        let temperatureBefore = snapshot("temperature")

        let losses = try laya.fineTune(examples: examples, steps: 12, learningRate: 5e-3, trainable: .head)
        XCTAssertEqual(losses.count, 12)
        XCTAssertTrue(losses.allSatisfy(\.isFinite))
        XCTAssertLessThan(losses.suffix(4).reduce(0, +) / 4, losses.prefix(4).reduce(0, +) / 4, "the loss falls")
        XCTAssertEqual(snapshot("encoder.layers.1.attn.Wqkv.weight"), encoderBefore, "a head run leaves the encoder alone")
        XCTAssertNotEqual(snapshot("scorer.1.weight"), scorerBefore, "the scorer moves")
        XCTAssertEqual(snapshot("act_head.0.weight"), actBefore, "the act head stays frozen")
        XCTAssertEqual(snapshot("temperature"), temperatureBefore, "the temperature buffer stays frozen")

        // A full run moves the encoder too.
        _ = try laya.fineTune(examples: examples, steps: 2, learningRate: 5e-3, trainable: .all)
        XCTAssertNotEqual(snapshot("encoder.layers.1.attn.Wqkv.weight"), encoderBefore, "a full run moves the encoder")
    }

    func testAMismatchedTargetAndAMissingTokenizerAreRefused() throws {
        try requireMLXRuntime()
        let laya = try tinyLaya()
        let wrongWidth = NFKMLXLayaExample(state: "s", question: department, target: [1, 0])
        XCTAssertThrowsError(try laya.fineTune(examples: [wrongWidth], steps: 1))
        let bare = try NFKMLXLaya.laya(weightsURL: nil, tokenizer: nil, configuration: .tiny)
        XCTAssertThrowsError(try bare.fineTune(examples: [NFKMLXLayaExample(state: "s", question: urgent, holds: false)], steps: 1))
    }

    // MARK: - Conversation prefixes

    private var resolved: NFKDecisionQuestion {
        NFKDecisionQuestion.noulQuestion(withInstructions: "The customer's problem will be resolved by the end of the conversation.",
                                         trueMeaning: "the issue is on its way to a fix", falseMeaning: "the issue stays open")
    }

    func testThePrefixLengthsFollowTheReference() {
        XCTAssertEqual(NFKMLXLayaPrompt.prefixLengths(turnCount: 4, maxPrefixes: 6), [1, 2, 3, 4])
        XCTAssertEqual(NFKMLXLayaPrompt.prefixLengths(turnCount: 8, maxPrefixes: 6), [1, 2, 4, 5, 7, 8],
                       "linspace(1, 8, 6) rounded: 1, 2.4, 3.8, 5.2, 6.6, 8")
        XCTAssertEqual(NFKMLXLayaPrompt.prefixLengths(turnCount: 20, maxPrefixes: 3), [1, 10, 20])
        XCTAssertEqual(NFKMLXLayaPrompt.prefixLengths(turnCount: 0, maxPrefixes: 6), [])
    }

    func testAPrefixStateListsTheContextThenTheConversation() {
        let text = NFKMLXLayaPrompt.serialize(context: ["plan": "Pro", "account": 1], turns: ["hi", ["role": "agent", "text": "hello"]])
        XCTAssertEqual(text, "{\"account\": 1, \"plan\": \"Pro\", \"conversation\": [\"hi\", {\"role\": \"agent\", \"text\": \"hello\"}]}")
    }

    func testTheTemporalDifferenceTargetsBlendTheOutcomeAndTheNextPrediction() {
        let predictions: [Float] = [0.2, 0.4, 0.6, 0.8]
        let monteCarlo = NFKMLXLayaObjective.temporalDifferenceTargets(outcome: true, nextTrueProbabilities: predictions, lambda: 1)
        XCTAssertEqual(monteCarlo, [[0, 1], [0, 1], [0, 1], [0, 1]], "at λ = 1 every prefix is trained toward the outcome")
        let bootstrapped = NFKMLXLayaObjective.temporalDifferenceTargets(outcome: false, nextTrueProbabilities: predictions, lambda: 0)
        XCTAssertEqual(bootstrapped[3], [1, 0], "the last prefix's target is the outcome")
        XCTAssertEqual(bootstrapped[2][1], 0.8, accuracy: 1e-6, "an earlier prefix is trained toward the next prefix's prediction")
        XCTAssertEqual(bootstrapped[0][1], 0.4, accuracy: 1e-6)
        let half = NFKMLXLayaObjective.temporalDifferenceTargets(outcome: true, nextTrueProbabilities: predictions, lambda: 0.5)
        XCTAssertEqual(half[2][1], 0.5 * 0.8 + 0.5 * 1, accuracy: 1e-6)
        XCTAssertEqual(half[1][1], 0.5 * 0.6 + 0.5 * half[2][1], accuracy: 1e-6)
        XCTAssertEqual(NFKMLXLayaObjective.temporalDifferenceTargets(outcome: true, nextTrueProbabilities: [], lambda: 1), [])
    }

    func testAnEpisodeBuildsOnePromptPerPrefixCutFromTheLeft() throws {
        try requireMLXRuntime()
        let laya = try tinyLaya()
        let turns = (1 ... 8).map { "turn number \($0) of the conversation" }
        let episode = NFKMLXLayaEpisode(context: ["account": 1], turns: turns, question: resolved, holds: true)
        let prefixes = laya.prefixes(of: episode)
        XCTAssertEqual(prefixes.map(\.length), [1, 2, 4, 5, 7, 8])
        for prefix in prefixes {
            XCTAssertEqual(prefix.prompt.markers.count, 2)
            XCTAssertEqual(prefix.prompt.tokens.count, NFKMLXLayaConfiguration.tiny.maxLength, "a prefix state is longer than the tiny budget")
        }
        // The newest turn survives the cut: turn 8's text is in the last prefix and not in the first,
        // which the character tokenizer makes visible as a run of ids.
        func contains(_ tokens: [Int], _ run: [Int]) -> Bool {
            tokens.count >= run.count && (0 ... tokens.count - run.count).contains { Array(tokens[$0 ..< $0 + run.count]) == run }
        }
        // The tiny budget leaves 28 tokens of state, so the run checked is the last few of a turn.
        let turnEight = characterTokenizer.encode("8 of the")
        XCTAssertTrue(contains(prefixes[5].prompt.tokens, turnEight))
        XCTAssertFalse(contains(prefixes[0].prompt.tokens, turnEight))
        XCTAssertTrue(contains(prefixes[0].prompt.tokens, characterTokenizer.encode("1 of the")))
    }

    // The prefix builder and the TD targets against the release's own rl_common on the root tokenizer.
    func testTheEpisodePathMatchesTheReference() throws {
        guard let path = config["IK_PARITY_LAYA_EPISODE"], let root = config["IK_VAL_LAYA"] else {
            throw XCTSkip("set IK_PARITY_LAYA_EPISODE (run_reference.py laya_episode) and IK_VAL_LAYA")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let directory = URL(fileURLWithPath: root)
        let c = try NFKMLXLayaConfiguration(directoryURL: directory)
        let tokenizer = try XCTUnwrap(NFKMLXLayaTokenizer.tokenizer(inDirectory: directory))
        let turns: [Any] = [
            ["role": "customer", "text": "Hi, my payouts have failed three times this week."],
            ["role": "agent", "text": "Sorry to hear that. Which bank is the account with?"],
            ["role": "customer", "text": "Nordbank, and nothing changed on my side."],
            ["role": "agent", "text": "I see two rejections from the receiving bank. Did they contact you?"],
            ["role": "customer", "text": "No. I have staff to pay on Friday, this cannot wait."],
            ["role": "agent", "text": "Understood. I am escalating this to the payments team now."],
            ["role": "customer", "text": "Thank you. Will I hear back today?"],
            ["role": "agent", "text": "Yes, within two hours, and the transfer will be retried."],
        ]
        let prefixes = NFKMLXLayaPrompt.prefixes(context: ["account": 88213, "channel": "chat"], turns: turns,
                                                 question: resolved, tokenizer: tokenizer, configuration: c)
        let referenceLengths = try XCTUnwrap(arrays["prefix_lengths"]).asArray(Int32.self).map(Int.init)
        XCTAssertEqual(prefixes.map(\.length), referenceLengths)
        for (index, prefix) in prefixes.enumerated() {
            let state = NFKMLXLayaPrompt.serialize(context: ["account": 88213, "channel": "chat"], turns: Array(turns.prefix(prefix.length)))
            XCTAssertEqual(Array(state.utf8).map { Int32($0) }, try XCTUnwrap(arrays["p\(index).state_bytes"]).asArray(Int32.self),
                           "prefix \(index) serializes as the reference writes it")
            XCTAssertEqual(prefix.prompt.tokens, try XCTUnwrap(arrays["p\(index).ids"]).asArray(Int32.self).map(Int.init),
                           "prefix \(index) tokenizes to the reference's ids")
            XCTAssertEqual(prefix.prompt.markers, try XCTUnwrap(arrays["p\(index).markers"]).asArray(Int32.self).map(Int.init))
        }

        let predictions = try XCTUnwrap(arrays["p_true"]).asArray(Float.self)
        let outcome = try XCTUnwrap(arrays["outcome"]).asArray(Int32.self)[0] == 1
        for (lambda, key) in [(Float(1), "targets_lambda1"), (Float(0.5), "targets_lambda05")] {
            let mine = NFKMLXLayaObjective.temporalDifferenceTargets(outcome: outcome, nextTrueProbabilities: predictions, lambda: lambda)
            let theirs = try XCTUnwrap(arrays[key])
            XCTAssertEqual(theirs.shape, [mine.count, 2])
            let flat = theirs.asArray(Float.self)
            for (row, target) in mine.enumerated() {
                XCTAssertEqual(target[0], flat[2 * row], accuracy: 1e-6, "\(key) row \(row)")
                XCTAssertEqual(target[1], flat[2 * row + 1], accuracy: 1e-6, "\(key) row \(row)")
            }
        }
        print("VALIDATION PARITY laya_episode: \(prefixes.count) prefixes token-exact; TD targets at λ 1 and 0.5 match")
    }

    func testAnEpisodeFineTuneLowersTheLossAndRefusesTheWrongQuestion() throws {
        try requireMLXRuntime()
        let laya = try tinyLaya()
        let episodes = [
            NFKMLXLayaEpisode(context: ["channel": "chat"], turns: ["it is broken", "we are fixing it", "fixed now"],
                              question: resolved, holds: true),
            NFKMLXLayaEpisode(context: ["channel": "email"], turns: ["it is broken", "we cannot help"],
                              question: resolved, holds: false),
        ]
        let scorerBefore: [Float] = {
            let value = Dictionary(uniqueKeysWithValues: laya.net.parameters().flattened())["scorer.1.weight"]!
            eval(value); return value.asArray(Float.self)
        }()
        let losses = try laya.fineTune(episodes: episodes, steps: 10, learningRate: 5e-3, lambda: 0.5)
        XCTAssertEqual(losses.count, 10)
        XCTAssertTrue(losses.allSatisfy(\.isFinite))
        XCTAssertLessThan(losses.suffix(4).reduce(0, +) / 4, losses.prefix(4).reduce(0, +) / 4, "the loss falls")
        let scorerAfter = Dictionary(uniqueKeysWithValues: laya.net.parameters().flattened())["scorer.1.weight"]!
        eval(scorerAfter)
        XCTAssertNotEqual(scorerAfter.asArray(Float.self), scorerBefore)

        let choice = NFKMLXLayaEpisode(context: [:], turns: ["a"], question: department, holds: true)
        XCTAssertThrowsError(try laya.fineTune(episodes: [choice], steps: 1), "an episode's question is a noul")
        XCTAssertThrowsError(try laya.fineTune(episodes: [], steps: 1))
    }

    // Train, save, reload through the public factory with a fine-tuned file, and get the same answers.
    func testAFineTunedFileReloadsThroughTheFactoryAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let laya = try tinyLaya()
        _ = try laya.fineTune(examples: [NFKMLXLayaExample(state: "s", question: urgent, holds: true)], steps: 3, learningRate: 1e-2)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("laya-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(laya.net, to: url)

        let reloaded = try NFKMLXLaya.laya(weightsURL: url, tokenizer: characterTokenizer, configuration: .tiny)
        let before = laya.distribution(state: "a state", question: department)
        let after = reloaded.distribution(state: "a state", question: department)
        XCTAssertEqual(before.probabilities, after.probabilities)
        XCTAssertEqual(before.actProbability, after.actProbability)
    }
}

/// A hub whose transport writes a placeholder file and records the path it was asked for.
private final class NFKLayaRecordingHub: NFKHFHub {
    private(set) var fetched = [String]()

    override func fetch(_ remoteURL: URL, toFileURL destinationURL: URL) throws {
        fetched.append(remoteURL.path)
        try FileManager.default.createDirectory(at: destinationURL.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: destinationURL)
    }
}
