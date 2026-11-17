//
//  NFKMLXRecurrentConversationTests.swift
//  InferKitMLXTests
//
//  Cached generation for the decoders whose layers carry a Mamba-2 recurrent state: a prompt prefilled in
//  passes and tokens run one at a time through the Mamba layers' state, and Nemotron-H's attention layers'
//  keys and values, give the logits the full sequence gives. A conversation resumes from its live cache or a
//  snapshot, from disk too, and the backends continue conversations keyed or not. On released weights,
//  Nemotron 3 Nano 4B continues a turn as transformers' run of the whole prompt does
//  (`IK_VAL_NEMOTRON3_NANO_4B` / `IK_PARITY_NEMOTRON3_CONVERSATION`).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXRecurrentConversationTests: XCTestCase {

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

    private let mambaConfiguration = NFKMLXMamba2Configuration(
        hiddenSize: 64, layerCount: 4, vocabularySize: 128, rmsEpsilon: 1e-5,
        intermediateSize: 128, headCount: 8, headDimensions: 16, stateSize: 16,
        groupCount: 2, convolutionKernel: 4, useConvolutionBias: true,
        useProjectionBias: false, tiesWordEmbeddings: false)

    private let nemotronConfiguration = NFKMLXNemotronHConfiguration(
        hiddenSize: 64, vocabularySize: 128, rmsEpsilon: 1e-5,
        headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
        mambaHeadCount: 8, mambaHeadDimensions: 16, mambaGroupCount: 2, mambaStateSize: 16,
        mambaConvolutionKernel: 4, mambaConvolutionBias: true, mambaProjectionBias: false,
        timeStepMinimum: 0.001, intermediateSize: 96, mlpBias: false,
        layerTypes: NFKMLXNemotronHConfiguration.layerKinds(fromPattern: "M-M*-M*-"))

    /// `net` with its norms varied, so a norm's weight is not the identity.
    private func varied<Net: Module>(_ net: Net) -> Net {
        let parameters = net.parameters().flattened().map { key, value in
            (key, key.hasSuffix("norm.weight") ? 1 + MLXRandom.normal(value.shape) * 0.1 : value)
        }
        net.update(parameters: ModuleParameters.unflattened(parameters))
        eval(net)
        return net
    }

    private func mambaNet(seed: UInt64) -> NFKMLXMamba2Net {
        MLXRandom.seed(seed)
        return varied(NFKMLXMamba.makeNet(mambaConfiguration))
    }

    private func nemotronNet(seed: UInt64) -> NFKMLXNemotronHNet {
        MLXRandom.seed(seed)
        return varied(NFKMLXNemotronH.makeNet(nemotronConfiguration))
    }

    private func tokens(_ count: Int, seed: Int) -> [Int] {
        (0 ..< count).map { 3 + ($0 * 37 + seed * 11) % 120 }
    }

    private func worst(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    private func fullLast(_ full: (MLXArray) -> MLXArray, _ sequence: [Int]) -> MLXArray {
        let logits = full(MLXArray(sequence.map(Int32.init)).reshaped([1, sequence.count]))[0, sequence.count - 1]
        eval(logits)
        return logits
    }

    /// Prefills `prompt` in passes of 8 through `cached`, then runs five tokens one at a time, comparing each
    /// position's logits with `full`'s pass over the whole sequence.
    private func checkCached(_ cached: NFKRecurrentCachedGeneration, full: (MLXArray) -> MLXArray, name: String) {
        let prompt = tokens(30, seed: 1)
        let cache = cached.makeCache()
        var logits = cached.prefill(prompt, cache: cache, limit: 8)
        var sequence = prompt
        eval(logits)
        XCTAssertLessThan(worst(logits, fullLast(full, sequence)), 1e-4, "\(name): the prefill's last row")
        for token in tokens(5, seed: 2) {
            logits = cached.step(MLXArray(Int32(token)), cache: cache)
            eval(logits)
            sequence.append(token)
            XCTAssertLessThan(worst(logits, fullLast(full, sequence)), 1e-4, "\(name): after \(sequence.count) tokens")
        }
        XCTAssertEqual(cache.offset, sequence.count)
    }

    func testTheMambaCacheGivesTheFullSequencesLogits() throws {
        try requireMLXRuntime()
        let net = mambaNet(seed: 31)
        checkCached(NFKMLXMamba.cachedGeneration(net), full: { net($0) }, name: "mamba")
    }

    func testTheNemotronCacheGivesTheFullSequencesLogits() throws {
        try requireMLXRuntime()
        let net = nemotronNet(seed: 37)
        checkCached(NFKMLXNemotronH.cachedGeneration(net), full: { net($0) }, name: "nemotron")
    }

    // MARK: Conversations

    /// Runs `prompt` through `kept` as the backend does, then feeds `reply`; answers where the prefill started
    /// and the logits after the prompt and after each fed token.
    private func turn(_ cached: NFKRecurrentCachedGeneration, _ kept: NFKMLXRecurrentConversation, prompt: [Int],
                      turnStart: Int?, reply: [Int]) -> (start: Int, logits: [MLXArray]) {
        let start = kept.resume(for: prompt)
        let stops = NFKMLXRecurrentConversation.snapshotPositions(start: start, end: prompt.count, turnStart: turnStart)
        var logits = [cached.prefill(Array(prompt[start...]), cache: kept.cache, limit: 8, stops: stops) { _ in
            kept.snapshot()
        }]
        kept.record(Array(prompt[start...]))
        for token in reply {
            logits.append(cached.step(MLXArray(Int32(token)), cache: kept.cache))
            kept.record([token])
        }
        eval(logits)
        return (start, logits)
    }

    private func conversation(_ cached: NFKRecurrentCachedGeneration) -> NFKMLXRecurrentConversation {
        NFKMLXRecurrentConversation(mambaLayers: cached.mambaLayers, attentionLayers: cached.attentionLayers)
    }

    // A follow-up parting from the held tokens inside the last prompt resumes from the turn-start snapshot,
    // one parting after the prompt resumes from the prompt's end, and one that extends every held token
    // continues the live cache; each gives the whole sequence's logits.
    func testAFollowUpResumesFromASnapshotOrTheLiveCache() throws {
        try requireMLXRuntime()
        let net = nemotronNet(seed: 41)
        let cached = NFKMLXNemotronH.cachedGeneration(net)
        let kept = conversation(cached)
        let first = tokens(30, seed: 1)
        XCTAssertEqual(turn(cached, kept, prompt: first, turnStart: 26, reply: tokens(4, seed: 2)).start, 0)
        XCTAssertEqual(kept.snapshots.map(\.position), [26, 30], "the turn start and the prompt's end")

        var second = Array(first[..<28]) + tokens(9, seed: 3)
        var resumed = turn(cached, kept, prompt: second, turnStart: 33, reply: tokens(3, seed: 4))
        XCTAssertEqual(resumed.start, 26, "the newest snapshot inside the shared prefix")
        XCTAssertLessThan(worst(resumed.logits[0], fullLast({ net($0) }, second)), 1e-4)
        for (index, token) in tokens(3, seed: 4).enumerated() {
            second.append(token)
            XCTAssertLessThan(worst(resumed.logits[index + 1], fullLast({ net($0) }, second)), 1e-4, "step \(index)")
        }

        let held = kept.tokens
        let third = held + tokens(5, seed: 5)
        resumed = turn(cached, kept, prompt: third, turnStart: nil, reply: tokens(2, seed: 6))
        XCTAssertEqual(resumed.start, held.count, "the live cache")
        XCTAssertLessThan(worst(resumed.logits[0], fullLast({ net($0) }, third)), 1e-4)

        let fourth = third + tokens(3, seed: 7)
        resumed = turn(cached, kept, prompt: fourth, turnStart: nil, reply: [])
        XCTAssertEqual(resumed.start, third.count, "the reply's tokens differ: the third prompt's end")
        XCTAssertLessThan(worst(resumed.logits[0], fullLast({ net($0) }, fourth)), 1e-4)
    }

    // A pure Mamba decoder resumes from a snapshot as the hybrid one does.
    func testAMambaConversationResumesFromItsTurnSnapshot() throws {
        try requireMLXRuntime()
        let net = mambaNet(seed: 43)
        let cached = NFKMLXMamba.cachedGeneration(net)
        let kept = conversation(cached)
        let first = tokens(20, seed: 1)
        _ = turn(cached, kept, prompt: first, turnStart: 17, reply: tokens(5, seed: 2))
        let second = Array(first[..<18]) + tokens(6, seed: 3)
        let resumed = turn(cached, kept, prompt: second, turnStart: nil, reply: [])
        XCTAssertEqual(resumed.start, 17)
        XCTAssertLessThan(worst(resumed.logits[0], fullLast({ net($0) }, second)), 1e-4)
    }

    // A conversation written to disk and read back resumes from its live state and its snapshot as the held
    // one does.
    func testASavedConversationResumesAsTheHeldOne() throws {
        try requireMLXRuntime()
        let net = nemotronNet(seed: 47)
        let cached = NFKMLXNemotronH.cachedGeneration(net)
        let kept = conversation(cached)
        let first = tokens(24, seed: 1)
        _ = turn(cached, kept, prompt: first, turnStart: 20, reply: tokens(6, seed: 2))
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("recurrent-spill-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let spill = NFKMLXConversationSpill(directory: directory, byteBudget: 1 << 30)
        func restored() throws -> NFKMLXRecurrentConversation {
            try spill.write(kept, name: "a")
            return try XCTUnwrap(spill.take("a") { url, record in
                try NFKMLXRecurrentConversation.load(from: url, tokens: record.tokens, mambaLayers: cached.mambaLayers,
                                                     attentionLayers: cached.attentionLayers)
            })
        }
        let live = try restored()
        XCTAssertEqual(live.snapshots.map(\.position), [20, 24])
        let extended = kept.tokens + tokens(4, seed: 3)
        var resumed = turn(cached, live, prompt: extended, turnStart: nil, reply: [])
        XCTAssertEqual(resumed.start, kept.tokens.count, "the live state read back")
        XCTAssertLessThan(worst(resumed.logits[0], fullLast({ net($0) }, extended)), 1e-4)

        let parted = Array(first[..<22]) + tokens(6, seed: 4)
        resumed = turn(cached, try restored(), prompt: parted, turnStart: nil, reply: [])
        XCTAssertEqual(resumed.start, 20, "the snapshot read back")
        XCTAssertLessThan(worst(resumed.logits[0], fullLast({ net($0) }, parted)), 1e-4)
    }

    // MARK: The released decoder

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).asArray(Float.self).map(Double.init)
        let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        let norms = sqrt(x.reduce(0) { $0 + $1 * $1 }) * sqrt(y.reduce(0) { $0 + $1 * $1 })
        return dot / norms
    }

    // Nemotron 3 Nano 4B at float32 generates a first turn through its cache and continues the second from the
    // first prompt's end, where its Mamba layers' state was snapshotted, and gives the logits and greedy tokens
    // transformers gives in one uncached pass over the second turn's prompt and its continuation.
    func testTheReleasedNemotronContinuesATurnAsAFullRerunDoes() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let recordPath = environment["IK_PARITY_NEMOTRON3_CONVERSATION"],
              let release = environment["IK_VAL_NEMOTRON3_NANO_4B"] else {
            throw XCTSkip("set IK_PARITY_NEMOTRON3_CONVERSATION and IK_VAL_NEMOTRON3_NANO_4B")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        func ids(_ key: String) throws -> [Int] { try XCTUnwrap(record[key]).asArray(Int32.self).map(Int.init) }
        let reference = try XCTUnwrap(record["output"])                      // [steps, vocabulary]
        let referenceCached = try XCTUnwrap(record["cached"])
        let directory = URL(fileURLWithPath: release)
        let configuration = try NFKMLXNemotronH.configuration(fromDirectory: directory)
        let net = NFKMLXNemotronH.makeNet(configuration)
        try NFKMLXNemotronH.loadWeights(into: net, fromDirectory: directory, precision: .float32, residency: .automatic)
        let cached = NFKMLXNemotronH.cachedGeneration(net)
        let kept = conversation(cached)
        let first = try ids("turn1.tokens")
        _ = turn(cached, kept, prompt: first, turnStart: nil, reply: Array(try ids("turn1.continuation").dropLast()))
        let prompt = try ids("turn2.tokens"), continuation = try ids("continuation")
        let second = turn(cached, kept, prompt: prompt, turnStart: nil, reply: Array(continuation.dropLast()))
        try net.verifyStream()
        XCTAssertEqual(second.start, first.count, "the second turn resumed at the first prompt's end")
        let fresh = cached.makeCache()
        var rerun = [cached.prefill(prompt, cache: fresh)]
        for token in continuation.dropLast() {
            rerun.append(cached.step(MLXArray(Int32(token)), cache: fresh))
        }
        eval(rerun)
        try net.verifyStream()
        // The reference generates a fixed count with `min_new_tokens`, which suppresses the end ids before it
        // chooses; the greedy token is the raw logits' argmax, and both runs are fed the chosen tokens.
        var worstCosine = 1.0, worstRerun = 1.0, worstAgreement = 1.0, referenceDrift = 1.0
        for step in continuation.indices {
            worstCosine = min(worstCosine, cosine(second.logits[step], reference[step]))
            worstRerun = min(worstRerun, cosine(rerun[step], reference[step]))
            worstAgreement = min(worstAgreement, cosine(second.logits[step], rerun[step]))
            referenceDrift = min(referenceDrift, cosine(referenceCached[step], reference[step]))
            XCTAssertEqual(second.logits[step].argMax().item(Int.self), reference[step].argMax().item(Int.self),
                           "the greedy token at step \(step)")
        }
        print("VALIDATION PARITY nemotron-3-nano-4b-conversation: turn 2 resumed at \(second.start) of \(prompt.count), "
              + "\(continuation.count) steps, worst logit cosine \(worstCosine) (a fresh cached run of the whole "
              + "prompt \(worstRerun); resumed against fresh \(worstAgreement)); transformers' own cached steps "
              + "against its uncached pass \(referenceDrift)")
        // The release's own prefill logits measure 0.99999999999437 against the reference.
        XCTAssertGreaterThan(worstCosine, 0.9999999999)
        XCTAssertGreaterThan(worstRerun, 0.9999999999)
        XCTAssertGreaterThan(worstAgreement, 0.9999999999, "resuming adds nothing to a fresh run's error")
    }

    // MARK: The backends

    /// A byte-fallback vocabulary as large as the decoder's, so every id it produces decodes.
    private func mistralTokenizer() throws -> NFKMLXMistralTokenizer {
        var vocabulary: [String: Int] = ["<unk>": 0, "<s>": 1, "</s>": 2, "\u{2581}": 3]
        for (index, letter) in "abcdefghijklmnopqrstuvwxyz".enumerated() {
            vocabulary[String(letter)] = 4 + index
        }
        for id in 30 ..< 128 {
            vocabulary["[\(id)]"] = id
        }
        let json: [String: Any] = ["model": ["type": "BPE", "byte_fallback": true, "merges": [String](), "vocab": vocabulary]]
        return try XCTUnwrap(NFKMLXMistralTokenizer(tokenizerJSON: json, prependsEveryRun: true))
    }

    private func reply(_ backend: NFKInferenceBackend, _ prompt: String, _ parameters: [String: Any] = [:])
        throws -> (text: String, cached: Int) {
        var all: [String: Any] = [NFKParameterTemperature: 0, NFKParameterMaxTokens: 6]
        all.merge(parameters) { $1 }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: prompt], parameters: all))
        let usage = try XCTUnwrap(result.output(forKey: NFKOutputUsage) as? [String: Int])
        return (try XCTUnwrap(result.output(forKey: NFKOutputText) as? String), usage[NFKUsageCachedTokens] ?? -1)
    }

    // Each backend's cached loop writes the reply its loop over the whole growing sequence writes.
    func testTheCachedBackendsReplyAsTheUncachedOnesDo() throws {
        try requireMLXRuntime()
        let prompt = "the quiet harbour town sleeps under the hills"
        let mamba = mambaNet(seed: 53)
        let tokenizer = try mistralTokenizer()
        let cachedMamba = NFKMLXMambaBackend(logits: { mamba($0) }, tokenizer: tokenizer, identifier: "mamba",
                                             cached: NFKMLXMamba.cachedGeneration(mamba))
        let plainMamba = NFKMLXMambaBackend(logits: { mamba($0) }, tokenizer: tokenizer, identifier: "mamba")
        XCTAssertEqual(try reply(cachedMamba, prompt).text, try reply(plainMamba, prompt).text, "mamba")

        let nemotron = nemotronNet(seed: 59)
        let scalar = NFKMLXScalarTestTokenizer(vocabularySize: 128)
        let cachedNemotron = NFKMLXNemotronBackend(logits: { nemotron($0) }, tokenizer: scalar, identifier: "nemotron",
                                                   cached: NFKMLXNemotronH.cachedGeneration(nemotron))
        let plainNemotron = NFKMLXNemotronBackend(logits: { nemotron($0) }, tokenizer: scalar, identifier: "nemotron")
        XCTAssertEqual(try reply(cachedNemotron, prompt).text, try reply(plainNemotron, prompt).text, "nemotron")
    }

    // The backends continue a conversation with or without a key, answer as a run that keeps nothing, and a
    // released conversation continues from the spill directory.
    func testTheBackendsContinueConversationsKeyedOrNotAndFromDisk() throws {
        try requireMLXRuntime()
        let mamba = mambaNet(seed: 61)
        let nemotron = nemotronNet(seed: 67)
        let backends: [(String, NFKMLXConversationServing)] = [
            ("mamba", NFKMLXMambaBackend(logits: { mamba($0) }, tokenizer: try mistralTokenizer(), identifier: "mamba",
                                         cached: NFKMLXMamba.cachedGeneration(mamba))),
            ("nemotron", NFKMLXNemotronBackend(logits: { nemotron($0) }, tokenizer: NFKMLXScalarTestTokenizer(vocabularySize: 128),
                                               identifier: "nemotron", cached: NFKMLXNemotronH.cachedGeneration(nemotron))),
        ]
        let opening = "tell me about the quiet harbour town"
        let followUp = opening + " and the boats in it"
        for (label, served) in backends {
            XCTAssertTrue(served.supportedParameterKeys?.contains(NFKParameterConversationKey) == true, label)
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("recurrent-backend-spill-\(UUID().uuidString)", isDirectory: true)
            addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
            served.conversationSpillDirectoryURL = directory
            for key in ["a", nil] as [String?] {
                let parameters: [String: Any] = key.map { [NFKParameterConversationKey: $0] } ?? [:]
                let name = "\(label) \(key ?? "keyless")"
                XCTAssertEqual(try reply(served, opening, parameters).cached, 0, name)
                let continued = try reply(served, followUp, parameters)
                XCTAssertGreaterThan(continued.cached, 0, "\(name) continues its prompt")
                XCTAssertEqual(continued.text,
                               try reply(served, followUp, [NFKMLXGenerationParameterKey.reusesPromptCache: false]).text, name)
                XCTAssertGreaterThan(served.releaseIdleConversationCaches(), 0, name)
                XCTAssertEqual(served.conversationCacheCount, 0, name)
                XCTAssertGreaterThan(try reply(served, followUp + " today", parameters).cached, 0, "\(name) continues from disk")
                served.resetPromptCache()
            }
        }
    }
}

/// The conversation surface the Mamba-2 and Nemotron-H backends share.
protocol NFKMLXConversationServing: NFKInferenceBackend {
    var conversationSpillDirectoryURL: URL? { get set }
    var conversationCacheCount: Int { get }
    func releaseIdleConversationCaches() -> Int
    func resetPromptCache()
}

extension NFKMLXMambaBackend: NFKMLXConversationServing {}
extension NFKMLXNemotronBackend: NFKMLXConversationServing {}
