//
//  NFKMLXOpenJevDeBERTaTests.swift
//  InferKitMLXTests
//
//  open-jev-deberta: the prompt layout and budgets, the relative-position buckets, the readout, the
//  backend's reading of the contract, padding, the objective, and the customization round trip on a
//  tiny random model; then, gated on the release and its record (`run_reference.py open_jev_deberta`),
//  the tokenizer, the buckets, every encoder layer, the logits, the answers, a padded batch, and the
//  objective against the release's own `typed_decisions` code.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXOpenJevDeBERTaTests: XCTestCase {

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

    /// One id per character, above the tiny configuration's special and marker tokens.
    private let characterTokenizer = NFKMLXDecisionTokenizer { text in
        text.unicodeScalars.map { Int($0.value % 400) + 10 }
    }

    private func tinyModel() -> NFKMLXOpenJevDeBERTa {
        NFKMLXRandom.seed(11)
        return NFKMLXOpenJevDeBERTa(net: NFKMLXOpenJevDeBERTaNet(.tiny), tokenizer: characterTokenizer)
    }

    private let team = NFKDecisionQuestion.choiceQuestion(withInstructions: "team?", options: ["ab", "cde", "f"])
    private let level = NFKDecisionQuestion.scoreQuestion(withInstructions: "lvl", levels: ["lo", "mid", "hi"])
    private let refund = NFKDecisionQuestion.noulQuestion(withInstructions: "refund")

    private func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) { dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y) }
        return dot / max(sqrt(na * nb), 1e-30)
    }

    // MARK: - The prompt

    func testThePromptFollowsTheReleaseLayout() throws {
        let c = NFKMLXOpenJevDeBERTaConfiguration.tiny
        let prompt = try NFKMLXOpenJevDeBERTaPrompt(state: "hi", questions: [team, refund],
                                                    tokenizer: characterTokenizer, configuration: c)
        XCTAssertEqual(Array(prompt.tokens.prefix(2)), [c.clsToken, c.stateMarker])
        XCTAssertEqual(prompt.tokens.last, c.sepToken)
        XCTAssertEqual(prompt.questionSpans.count, 2)
        XCTAssertEqual(prompt.optionSpans.map(\.count), [3, 2], "a noul is always no then yes")
        XCTAssertEqual(prompt.tokens[prompt.questionSpans[0].lowerBound - 1], c.questionMarker)
        XCTAssertEqual(prompt.questionSpans[0].count, 5, "the span is the instruction text, without its marker")
        for span in prompt.optionSpans.joined() {
            XCTAssertEqual(prompt.tokens[span.lowerBound - 1], c.optionMarker)
        }
        XCTAssertEqual(prompt.optionSpans[1].map(\.count), [2, 3], "no, yes")
        XCTAssertEqual(prompt.tokens.count, 2 + 2 + (1 + 5 + 3 + 4 + 2) + (1 + 6 + 3 + 4) + 1)
    }

    func testTheStateIsCutToItsBudgetThenToTheQuestionsRoom() throws {
        let c = NFKMLXOpenJevDeBERTaConfiguration.tiny
        let long = String(repeating: "x", count: 100)
        let short = try NFKMLXOpenJevDeBERTaPrompt(state: long, questions: [refund], tokenizer: characterTokenizer, configuration: c)
        XCTAssertEqual(short.questionSpans[0].lowerBound - 1, 2 + c.maxStateTokens, "the state keeps maxStateTokens")

        // A question of 41 tokens leaves the state 64 - 3 - 41 = 20, under its own budget of 24.
        let wide = NFKDecisionQuestion.choiceQuestion(withInstructions: String(repeating: "q", count: 25),
                                                      options: ["aaaaaa", "bbbbbbb"])
        let crowded = try NFKMLXOpenJevDeBERTaPrompt(state: long, questions: [wide], tokenizer: characterTokenizer, configuration: c)
        XCTAssertEqual(crowded.tokens.count, c.maxLength)
        XCTAssertEqual(crowded.questionSpans[0].lowerBound - 1, 2 + 20)

        let huge = NFKDecisionQuestion.choiceQuestion(withInstructions: String(repeating: "q", count: 70), options: ["a", "b"])
        XCTAssertThrowsError(try NFKMLXOpenJevDeBERTaPrompt(state: "s", questions: [huge], tokenizer: characterTokenizer,
                                                           configuration: c), "questions that cannot fit are refused")
    }

    func testDescriptionsAndMeaningsFollowTheirOptionAfterAColon() {
        let described = NFKDecisionQuestion.choiceQuestion(withInstructions: "t", options: ["a", "b"], descriptions: ["a": "first"])
        XCTAssertEqual(described.renderedOptions, ["a: first", "b"])
        let meant = NFKDecisionQuestion.noulQuestion(withInstructions: "s", trueMeaning: "it holds", falseMeaning: nil)
        XCTAssertEqual(meant.renderedOptions, ["no", "yes: it holds"])
    }

    // MARK: - The relative positions

    func testTheBucketsKeepNearOffsetsAndGrowLogarithmically() {
        let c = NFKMLXDeBERTaV2Configuration.tiny          // 8 buckets, scaled to 32
        let table = c.relativePositions(length: 32)
        let row = (0 ..< 32).map { Int(table[31 * 32 + $0]) }       // offsets 31 down to 0
        XCTAssertEqual(Array(row.suffix(5)), [4, 3, 2, 1, 0], "offsets within half the buckets stay exact")
        XCTAssertEqual(row.first, 7, "the largest offset reaches the last bucket, half the buckets past the exact range")
        XCTAssertEqual(row, row.sorted(by: >), "buckets never decrease with distance")
        XCTAssertEqual(Int(table[0 * 32 + 31]), -7, "the table is antisymmetric")
    }

    // MARK: - The answers

    func testTheAnswersCarryTheReleaseReadout() throws {
        try requireMLXRuntime()
        let model = tinyModel()
        let rows = try model.distributions(state: "hello", questions: [team, level, refund])
        XCTAssertEqual(rows.map(\.count), [3, 3, 2])
        for row in rows { XCTAssertEqual(row.reduce(0, +), 1, accuracy: 1e-5) }

        let answers = try model.decide(state: "hello", questions: [team, level, refund])
        let choice = answers[0]
        XCTAssertEqual(choice.type, .choice)
        XCTAssertEqual(choice.confidence, Double(rows[0].max()!), accuracy: 1e-6, "the confidence is the largest probability")
        XCTAssertEqual(choice.choice, ["ab", "cde", "f"][rows[0].firstIndex(of: rows[0].max()!)!])
        let expected = rows[1].enumerated().reduce(0.0) { $0 + Double($1.offset) * Double($1.element) }
        XCTAssertEqual(answers[1].score, expected, accuracy: 1e-6, "a score is the expected level index")
        XCTAssertEqual(answers[1].legend?["2"], "hi")
        XCTAssertEqual(answers[2].probability, Double(rows[2][1]), accuracy: 1e-6, "a noul is the probability of yes")

        let bad = NFKDecisionQuestion.scoreQuestion(withInstructions: "s", levels: ["only"])
        XCTAssertThrowsError(try model.decide(state: "s", questions: [bad]), "a score needs 2 to 10 levels")
    }

    func testPaddingDoesNotChangeALogit() throws {
        try requireMLXRuntime()
        let model = tinyModel()
        let short = try model.prompt(state: "hi", questions: [refund])
        let long = try model.prompt(state: "a longer state here", questions: [team, level])
        func logits(_ prompts: [NFKMLXOpenJevDeBERTaPrompt]) -> MLXArray {
            let inputs = NFKMLXOpenJevDeBERTaNet.inputs(prompts, padToken: 0)
            return model.net.logits(tokens: inputs.tokens, attentionMask: inputs.attentionMask,
                                    pooling: inputs.pooling, optionMask: inputs.optionMask)
        }
        let alone = logits([short])[0, 0, 0 ..< 2].asArray(Float.self)
        let batched = logits([long, short])[1, 0, 0 ..< 2].asArray(Float.self)
        for (a, b) in zip(alone, batched) { XCTAssertEqual(a, b, accuracy: 1e-4) }
    }

    func testTheBackendReadsTheQuestionsInSortedOrder() throws {
        try requireMLXRuntime()
        let model = tinyModel()
        let backend = model.makeBackend()
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [
            NFKInputState: ["ticket": 7, "body": "x"],
            NFKInputQuestions: ["b-refund": refund, "a-team": team.dictionaryRepresentation()],
        ]))
        let direct = try model.decide(state: ["ticket": 7, "body": "x"], questions: [team, refund])
        XCTAssertEqual(result.answers?["a-team"], direct[0])
        XCTAssertEqual(result.answers?["b-refund"], direct[1])
        let structured = result.output(forKey: NFKOutputStructured) as? [String: Any]
        XCTAssertEqual(structured?["model"] as? String, "open-jev-deberta-tiny")
        XCTAssertEqual(backend.backendIdentifier, "mlx-open-jev-deberta-tiny")
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputState: "s"])))
    }

    // MARK: - Customizing

    func testTheObjectiveIsCrossEntropyPlusTheWeightedBrierScore() throws {
        try requireMLXRuntime()
        let logits = MLXArray([Float(2), 0, -Float.infinity, 1, 1, 1], [1, 2, 3])
        let mask = MLXArray([true, true, false, true, true, true], [1, 2, 3])
        let gold = MLXArray([Int32(0), 2], [1, 2])
        let terms = NFKMLXOpenJevDeBERTaObjective(brierWeight: 0.5).terms(logits: logits, gold: gold, optionMask: mask)
        let p = Foundation.exp(2.0) / (Foundation.exp(2.0) + 1)
        let crossEntropy = (-Foundation.log(p) + Foundation.log(3.0)) / 2
        let brier = (2 * (1 - p) * (1 - p) + (1.0 / 9 + 1.0 / 9 + 4.0 / 9)) / 2
        XCTAssertEqual(Double(terms.crossEntropy.item(Float.self)), crossEntropy, accuracy: 1e-5)
        XCTAssertEqual(Double(terms.brier.item(Float.self)), brier, accuracy: 1e-5)
        XCTAssertEqual(Double(terms.total.item(Float.self)), crossEntropy + 0.5 * brier, accuracy: 1e-5)

        let padded = NFKMLXOpenJevDeBERTaObjective().loss(
            logits: MLXArray([Float(2), 0, -Float.infinity, -Float.infinity], [1, 2, 2]),
            gold: MLXArray([Int32(0), -1], [1, 2]), optionMask: MLXArray([true, true, false, false], [1, 2, 2]))
        XCTAssertTrue(padded.item(Float.self).isFinite, "a padded question stays out of the loss")
    }

    // The release's LambdaLR, evaluated by hand: warm = max(1, int(0.06 * steps)); (s + 1) / warm before
    // it, (steps - s) / max(1, steps - warm) from it on.
    func testTheDefaultScheduleIsTheReleasesWarmUpAndLinearDecay() {
        let hundred = NFKMLXLearningRateSchedule.openJevDeBERTa(steps: 100)
        XCTAssertEqual(hundred.multiplier(0), 1.0 / 6, accuracy: 1e-7)
        XCTAssertEqual(hundred.multiplier(5), 1, accuracy: 1e-7)
        XCTAssertEqual(hundred.multiplier(6), 94.0 / 94, accuracy: 1e-7)
        XCTAssertEqual(hundred.multiplier(50), 50.0 / 94, accuracy: 1e-7)
        XCTAssertEqual(hundred.multiplier(99), 1.0 / 94, accuracy: 1e-7)
        let ten = NFKMLXLearningRateSchedule.openJevDeBERTa(steps: 10)
        XCTAssertEqual(ten.multiplier(0), 1, accuracy: 1e-7, "a run too short for 6% warms up in one step")
        XCTAssertEqual(ten.multiplier(9), 1.0 / 9, accuracy: 1e-7)
    }

    func testAHeadFineTuneLowersTheLossAndLeavesTheEncoderAsReleased() throws {
        try requireMLXRuntime()
        let model = tinyModel()
        let examples = [
            NFKMLXOpenJevDeBERTaExample(state: "billing issue", questions: [team, refund], labels: [0, 1]),
            NFKMLXOpenJevDeBERTaExample(state: "a crash", question: team, label: 1),
            NFKMLXOpenJevDeBERTaExample(state: "all fine", question: refund, holds: false),
        ]
        let before = Dictionary(uniqueKeysWithValues: model.net.backbone.parameters().flattened())["encoder.layer.0.attention.self.query_proj.weight"]!
        let beforeValues = before.asArray(Float.self)
        let losses = try model.fineTune(examples: examples, steps: 30, headLearningRate: 1e-2, batchSize: 3)
        XCTAssertEqual(losses.count, 30)
        XCTAssertLessThan(losses.suffix(5).reduce(0, +), losses.prefix(5).reduce(0, +), "the loss falls")
        let after = Dictionary(uniqueKeysWithValues: model.net.backbone.parameters().flattened())["encoder.layer.0.attention.self.query_proj.weight"]!
        XCTAssertEqual(after.asArray(Float.self), beforeValues, "the head policy leaves the encoder alone")

        let wrong = NFKMLXOpenJevDeBERTaExample(state: "s", question: team, label: 3)
        XCTAssertThrowsError(try model.fineTune(examples: [wrong], steps: 1), "a label must index an option")
    }

    func testAFineTunedFileReloadsAndReproducesTheAnswers() throws {
        try requireMLXRuntime()
        let model = tinyModel()
        try model.fineTune(examples: [NFKMLXOpenJevDeBERTaExample(state: "s", question: team, label: 2)], steps: 3,
                           learningRate: 1e-3, headLearningRate: 1e-2, trainable: .all)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ojd-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(model.net, to: url)
        let reloaded = NFKMLXOpenJevDeBERTa(net: try NFKMLXOpenJevDeBERTa.network(weightsURL: url, configuration: .tiny),
                                            tokenizer: characterTokenizer)
        XCTAssertEqual(try model.distributions(state: "a state", questions: [team, level]),
                       try reloaded.distributions(state: "a state", questions: [team, level]))
    }

    // MARK: - Downloading

    func testADownloadFetchesTheReleaseOnceAndReturnsItsFolder() throws {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("ojd-hub-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        let hub = NFKOpenJevRecordingHub()
        hub.cacheDirectoryURL = cache
        let revision = NFKMLXOpenJevDeBERTa.measuredRevision
        let folder = try NFKMLXOpenJevDeBERTa.download(revision: revision, hub: hub)
        XCTAssertEqual(folder.standardizedFileURL.path,
                       cache.appendingPathComponent("\(NFKMLXOpenJevDeBERTa.repository)/\(revision)").standardizedFileURL.path)
        XCTAssertEqual(hub.fetched, NFKMLXOpenJevDeBERTa.releaseFiles.map {
            "/\(NFKMLXOpenJevDeBERTa.repository)/resolve/\(revision)/\($0)"
        })
        _ = try NFKMLXOpenJevDeBERTa.download(revision: revision, hub: hub)
        XCTAssertEqual(hub.fetched.count, NFKMLXOpenJevDeBERTa.releaseFiles.count, "a cached release is not fetched again")
    }

    // MARK: - Reference parity

    private static let cases: [(state: Any, questions: [NFKDecisionQuestion])] = [
        ("Customer: I was charged twice for the same order and nobody answers my emails. I want my money back now.",
         [.choiceQuestion(withInstructions: "Which product area is the message about?",
                          options: ["fees & charges", "pin & security", "refund & dispute", "card", "other"]),
          .scoreQuestion(withInstructions: "How positive is the sentiment of this message?",
                         levels: ["very negative", "negative", "neutral", "positive", "very positive"]),
          .noulQuestion(withInstructions: "The customer is asking for a refund.")]),
        (["ticket": 4471, "channel": "email", "body": "Mon café est froid — rembourser svp", "vip": true,
          "tags": ["billing", NSNull()]] as [String: Any],
         [.noulQuestion(withInstructions: "The message is written in French."),
          .choiceQuestion(withInstructions: "Which team?", options: ["billing", "technical", "sales"])]),
        (String(repeating: "The quarterly report shows revenue of $4.2M, up 12% year over year, while churn fell to 3.1%. "
                 + "Ｆｕｌｌｗｉｄｔｈ text, naïve résumé, 東京 office, and tabs\tand\nnewlines  with   spaces. ", count: 12),
         [.choiceQuestion(withInstructions: "What is the main topic?",
                          options: ["finance", "hiring", "product", "legal", "marketing", "operations", "sales", "support", "security", "other"]),
          .scoreQuestion(withInstructions: "How optimistic is the report?", levels: ["low", "medium", "high"])]),
    ]
    private static let texts = ["Hello world", "  leading and trailing  ", "naïve café résumé", "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３",
                                "東京タワー", "tabs\tand\nnewlines", "$4.2M (12%) -- ok?!", "don't won't I'm", "emoji 🙂 end",
                                "UPPER lower MiXeD", "", String(repeating: "a", count: 40), "x = f(y) + 3.14159e-2", "ﬁ ligature ½ ①"]

    private func release() throws -> (arrays: [String: MLXArray], directory: URL) {
        guard let path = config["IK_PARITY_OPEN_JEV_DEBERTA"], let directory = config["IK_VAL_OPEN_JEV_DEBERTA"] else {
            throw XCTSkip("set IK_PARITY_OPEN_JEV_DEBERTA and IK_VAL_OPEN_JEV_DEBERTA")
        }
        return (try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays, URL(fileURLWithPath: directory))
    }

    private func ints(_ array: MLXArray?) -> [Int] { array.map { $0.asType(.int32).asArray(Int32.self).map(Int.init) } ?? [] }

    func testTheTokenizerMatchesTheReference() throws {
        let (arrays, directory) = try release()
        let tokenizer = try NFKMLXOpenJevDeBERTa.tokenizer(inDirectory: directory)
        var mismatches = [String]()
        for (index, text) in Self.texts.enumerated() {
            let reference = ints(arrays["text\(index)"]).filter { $0 >= 0 }
            let mine = tokenizer.encode(text)
            if mine != reference { mismatches.append("\(text.debugDescription): \(mine) vs \(reference)") }
        }
        print("VALIDATION open-jev-deberta tokenizer: \(Self.texts.count - mismatches.count)/\(Self.texts.count) exact")
        XCTAssertEqual(mismatches, [])
    }

    // The repository's current collator cuts the state so the questions fit, where the release's
    // bundled one refuses the sequence: ten 10-way choices over the long state fill exactly max_len.
    func testTheStateBudgetMatchesTheCurrentCollator() throws {
        guard let path = config["IK_PARITY_OPEN_JEV_DEBERTA_BUDGET"], let directory = config["IK_VAL_OPEN_JEV_DEBERTA"] else {
            throw XCTSkip("set IK_PARITY_OPEN_JEV_DEBERTA_BUDGET and IK_VAL_OPEN_JEV_DEBERTA")
        }
        let reference = ints(try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays["budget.ids"])
        let tokenizer = try NFKMLXOpenJevDeBERTa.tokenizer(inDirectory: URL(fileURLWithPath: directory))
        let options = ["finance", "hiring", "product", "legal", "marketing", "operations", "sales", "support", "security", "other"]
        let questions = (0 ..< 10).map {
            NFKDecisionQuestion.choiceQuestion(withInstructions: "Which department owns item number \($0) in this report?", options: options)
        }
        let prompt = try NFKMLXOpenJevDeBERTaPrompt(state: Self.cases[2].state, questions: questions, tokenizer: tokenizer,
                                                    configuration: .v3Large)
        XCTAssertEqual(prompt.tokens.count, 512)
        XCTAssertEqual(prompt.tokens, reference)
    }

    func testTheBucketsMatchTheReference() throws {
        let (arrays, _) = try release()
        let reference = ints(arrays["buckets"])                      // offsets -600 ... 600
        let table = NFKMLXDeBERTaV2Configuration.v3Large.relativePositions(length: 601)
        let mine = (-600 ... 600).map { offset in
            offset >= 0 ? Int(table[offset * 601]) : Int(table[-offset])
        }
        XCTAssertEqual(mine, reference)
    }

    func testTheReleaseMatchesTheReference() throws {
        try requireMLXRuntime()
        let (arrays, directory) = try release()
        let model = try NFKMLXOpenJevDeBERTa.openJev(directoryURL: directory)
        XCTAssertEqual(model.configuration.temperature, arrays["temperature"]!.item(Float.self))

        var report = [String]()
        for (index, entry) in Self.cases.enumerated() {
            let prompt = try model.prompt(state: entry.state, questions: entry.questions)
            XCTAssertEqual(prompt.tokens, ints(arrays["c\(index).ids"]), "case \(index) tokens")
            if let bytes = arrays["c\(index).state_bytes"] {
                XCTAssertEqual(Array(NFKMLXLayaPrompt.serialize(state: entry.state).utf8).map(Int.init), ints(bytes))
            }
            let inputs = NFKMLXOpenJevDeBERTaNet.inputs([prompt], padToken: model.configuration.padToken)
            if index == 0 {
                let states = model.net.backbone.hiddenStates(inputs.tokens)
                var worst = 1.0
                for (layer, state) in states.enumerated() {
                    let similarity = cosine(state[0].asArray(Float.self), arrays["hidden.\(layer)"]!.asArray(Float.self))
                    worst = min(worst, similarity)
                    XCTAssertGreaterThan(similarity, 0.99999, "layer \(layer)")
                }
                report.append("worst layer cosine \(worst)")
            }
            let logits = model.net.logits(tokens: inputs.tokens, attentionMask: nil, pooling: inputs.pooling,
                                          optionMask: inputs.optionMask)[0]
            let finite = which(inputs.optionMask[0], logits, MLXArray(Float(0))).asArray(Float.self)
            let reference = arrays["c\(index).logits"]!.asArray(Float.self)
            let gap = zip(finite, reference).map { abs($0 - $1) }.max() ?? 0
            report.append("case \(index): logit cosine \(cosine(finite, reference)), max gap \(gap)")
            XCTAssertLessThan(gap, 2e-3, "case \(index) logits")

            let rows = try model.distributions(state: entry.state, questions: entry.questions)
            let referenceRows = arrays["c\(index).probabilities"]!
            let answers = try model.decide(state: entry.state, questions: entry.questions)
            for (q, question) in entry.questions.enumerated() {
                let expected = referenceRows[q].asArray(Float.self).prefix(rows[q].count)
                let probabilityGap = zip(rows[q], expected).map { abs($0 - $1) }.max() ?? 0
                XCTAssertLessThan(probabilityGap, 1e-4, "case \(index) question \(q) probabilities")
                let key = "c\(index).q\(q)"
                switch question.type {
                case .choice:
                    XCTAssertEqual(answers[q].choice, question.options[ints(arrays[key + ".choice"])[0]])
                    XCTAssertEqual(answers[q].confidence, Double(arrays[key + ".confidence"]!.item(Float.self)), accuracy: 1e-4)
                case .score:
                    XCTAssertEqual(answers[q].score, Double(arrays[key + ".score"]!.item(Float.self)), accuracy: 1e-4)
                default:
                    XCTAssertEqual(answers[q].probability, Double(arrays[key + ".noul"]!.item(Float.self)), accuracy: 1e-4)
                }
            }
        }
        print("VALIDATION PARITY open-jev-deberta:\n  " + report.joined(separator: "\n  "))
    }

    func testAPaddedBatchAndTheObjectiveMatchTheReference() throws {
        try requireMLXRuntime()
        let (arrays, directory) = try release()
        let model = try NFKMLXOpenJevDeBERTa.openJev(directoryURL: directory)
        let prompts = try Self.cases.prefix(2).map { try model.prompt(state: $0.state, questions: $0.questions) }
        let inputs = NFKMLXOpenJevDeBERTaNet.inputs(prompts, padToken: model.configuration.padToken)
        XCTAssertEqual(ints(inputs.tokens.reshaped([-1])), ints(arrays["batch.ids"]!.reshaped([-1])))
        let logits = model.net.logits(tokens: inputs.tokens, attentionMask: inputs.attentionMask,
                                      pooling: inputs.pooling, optionMask: inputs.optionMask)
        let finite = which(inputs.optionMask, logits, MLXArray(Float(0))).asArray(Float.self)
        let reference = arrays["batch.logits"]!.asArray(Float.self)
        let gap = zip(finite, reference).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(gap, 2e-3, "the padded batch")

        let gold = arrays["batch.gold"]!.asType(.int32)
        var report = ["padded batch: logit max gap \(gap)"]
        for weight in [Float(1), 0.5] {
            let terms = NFKMLXOpenJevDeBERTaObjective(brierWeight: weight).terms(logits: logits, gold: gold,
                                                                                 optionMask: inputs.optionMask)
            let expected = arrays["batch.loss.\(weight == 1 ? "1.0" : "0.5")"]!.asArray(Float.self)
            let mine = [terms.total.item(Float.self), terms.crossEntropy.item(Float.self), terms.brier.item(Float.self)]
            report.append("brier weight \(weight): loss \(mine[0]) vs \(expected[0]), ce \(mine[1]) vs \(expected[1]), brier \(mine[2]) vs \(expected[2])")
            for (a, b) in zip(mine, expected) { XCTAssertEqual(a, b, accuracy: 1e-4) }
        }
        print("VALIDATION PARITY open-jev-deberta objective:\n  " + report.joined(separator: "\n  "))
    }
}

/// A hub whose transport writes a placeholder file and records the path it was asked for.
final class NFKOpenJevRecordingHub: NFKHFHub {
    private(set) var fetched = [String]()

    override func fetch(_ remoteURL: URL, toFileURL destinationURL: URL) throws {
        fetched.append(remoteURL.path)
        try FileManager.default.createDirectory(at: destinationURL.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: destinationURL)
    }
}
