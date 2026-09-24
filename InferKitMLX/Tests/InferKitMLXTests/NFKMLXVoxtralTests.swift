//
//  NFKMLXVoxtralTests.swift
//  InferKitMLXTests
//
//  Voxtral-Mini (`VoxtralForConditionalGeneration`, Mistral): the reused Whisper encoder, the two-layer
//  projector, and the reused Llama decoder, measured seam by seam at a tiny random configuration against
//  transformers' own implementation (`run_reference.py voxtral`): the encoder last hidden state, the
//  projected audio embeddings, and the fused logits.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXVoxtralTests: XCTestCase {

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

    /// The tiny geometry `run_reference.py voxtral` records.
    private var tinyNet: NFKMLXVoxtralNet {
        let text = NFKMLXGraniteTextConfiguration(
            hiddenSize: 64, layerCount: 2, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            intermediateSize: 128, vocabularySize: 40, ropeTheta: 10_000, rmsEpsilon: 1e-5,
            embeddingMultiplier: 1, residualMultiplier: 1, attentionMultiplier: 1.0 / Float(16).squareRoot(),
            logitsScaling: 1, tiesWordEmbeddings: false)
        let config = NFKMLXVoxtralConfiguration(
            audioMels: 32, audioState: 64, audioHeads: 4, audioLayers: 2, projectorInputSize: 256,
            audioTokenId: 39, text: text)
        return NFKMLXVoxtral.makeNet(config)
    }

    private func loadRecordWeights(into net: NFKMLXVoxtralNet, from record: [String: MLXArray]) throws {
        let weights = record.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            guard let name = NFKMLXVoxtral.remap(String(key.dropFirst(3))) else { return nil }
            let array = value.ndim == 3 && name.hasSuffix(".weight") ? value.transposed(0, 2, 1) : value
            return (name, array)
        }
        try NFKMLXWeights.apply(weights, to: net)
    }

    func testSeamParityAgainstTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let recordPath = env["IK_PARITY_VOXTRAL_TINY"] else {
            throw XCTSkip("set IK_PARITY_VOXTRAL_TINY (run_reference.py voxtral)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let net = tinyNet
        try loadRecordWeights(into: net, from: rec)

        // The record's features are [mels, frames]; the encoder reads [batch, frames, mels].
        let features = rec["features"]!.reshaped([1, 32, 48]).transposed(0, 2, 1)
        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])

        let encoded = net.audioTower(features)
        let encoderSimilarity = cosine(encoded[0], rec["encoder_out"]!)
        print("SEAM voxtral encoder: cosine \(encoderSimilarity)")
        XCTAssertGreaterThan(encoderSimilarity, 0.9999, "encoder output diverges")

        let audio = net.audioEmbeddings(features)
        let audioSimilarity = cosine(audio, rec["audio_embeds"]!)
        print("SEAM voxtral audio embeds: cosine \(audioSimilarity)")
        XCTAssertGreaterThan(audioSimilarity, 0.9999, "projected audio embeddings diverge")

        let logitSimilarity = cosine(net.logits(tokens: tokens, audioEmbeddings: audio)[0], rec["output"]!)
        print("VALIDATION PARITY voxtral: logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.9999, "fused logits diverge")
    }

    // The released Voxtral-Mini-3B held to the module by shape: every released tensor, remapped from its
    // transformers names to the module's (the audio encoder's openai-Whisper names), matches by shape
    // (Conv1d weights compared in PyTorch's `[out, in, kernel]` against the module's `[out, kernel, in]`).
    // The learned `embed_positions` is the computed sinusoid and is not a module parameter.
    func testEveryParameterMatchesTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let shapesPath = env["IK_SHAPES_VOXTRAL"], let configPath = env["IK_CONFIG_VOXTRAL"],
              let data = FileManager.default.contents(atPath: shapesPath),
              let released = try JSONSerialization.jsonObject(with: data) as? [String: [Int]]
        else { throw XCTSkip("set IK_SHAPES_VOXTRAL and IK_CONFIG_VOXTRAL (shapes.py)") }

        let net = try NFKMLXVoxtral.net(
            fromDirectory: URL(fileURLWithPath: configPath).deletingLastPathComponent())
        let releasedMapped = Dictionary(uniqueKeysWithValues: released.compactMap {
            key, shape -> (String, [Int])? in NFKMLXVoxtral.remap(key).map { ($0, shape) }
        })
        var consumed = Set<String>()
        var missing = [String](), mismatched = [String]()
        for (name, value) in net.parameters().flattened() {
            let expected = value.ndim == 3 && name.hasSuffix(".weight")
                ? [value.dim(0), value.dim(2), value.dim(1)] : value.shape
            guard let shape = releasedMapped[name] else { missing.append(name); continue }
            consumed.insert(name)
            if shape != expected { mismatched.append("\(name): built \(expected), released \(shape)") }
        }
        let unaccounted = releasedMapped.keys.filter { !consumed.contains($0) }.sorted()
        print("VALIDATION structure voxtral-mini-3b: \(consumed.count) consumed, \(missing.count) missing, "
              + "\(mismatched.count) mismatched, \(unaccounted.count) unaccounted")
        XCTAssertTrue(mismatched.isEmpty, "shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(missing.isEmpty, "absent from the release:\n" + missing.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(unaccounted.isEmpty, "released tensors nothing reads:\n" + unaccounted.prefix(8).joined(separator: "\n"))
    }

    // The released Voxtral-Mini-3B run numerically at float32 against a float32 transformers oracle
    // (`run_reference.py voxtral_real`): the encoder and projector seams, the fused logits, and the
    // greedy continuation on fixed features.
    func testReleasedWeightsNumericParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_VOXTRAL"], let recordPath = env["IK_PARITY_VOXTRAL_REAL"] else {
            throw XCTSkip("set IK_VAL_VOXTRAL (release dir) and IK_PARITY_VOXTRAL_REAL (voxtral_real)")
        }
        let dir = URL(fileURLWithPath: directory)
        let net = try NFKMLXVoxtral.net(fromDirectory: dir)
        try NFKMLXVoxtral.loadWeights(into: net, fromDirectory: dir, precision: .float32)

        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let features = rec["features"]!.reshaped([1, net.config.audioMels, 3000]).transposed(0, 2, 1)
        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])

        let encoderSimilarity = cosine(net.audioTower(features)[0], rec["encoder_out"]!)
        let audio = net.audioEmbeddings(features)
        let audioSimilarity = cosine(audio, rec["audio_embeds"]!)
        let logitSimilarity = cosine(net.logits(tokens: tokens, audioEmbeddings: audio)[0], rec["output"]!)
        print("VALIDATION PARITY voxtral-mini-3b released: encoder \(encoderSimilarity), "
              + "audio \(audioSimilarity), logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(encoderSimilarity, 0.999, "encoder diverges")
        XCTAssertGreaterThan(audioSimilarity, 0.999, "audio embeddings diverge")
        XCTAssertGreaterThan(logitSimilarity, 0.999, "released logits diverge")

        let referenceIds = { () -> [Int32] in eval(rec["continuation"]!); return rec["continuation"]!.asType(.int32).asArray(Int32.self) }()
        var sequence = tokens
        var produced = [Int32]()
        for _ in 0 ..< referenceIds.count {
            let next = argMax(net.logits(tokens: sequence, audioEmbeddings: audio)[0, -1, 0...], axis: -1)
            eval(next)
            produced.append(next.item(Int32.self))
            sequence = concatenated([sequence, MLXArray([produced.last!]).reshaped([1, 1])], axis: 1)
        }
        let matches = zip(produced, referenceIds).filter { $0 == $1 }.count
        print("VALIDATION PARITY voxtral-mini-3b released: greedy \(matches)/\(referenceIds.count) tokens match")
        XCTAssertGreaterThanOrEqual(matches, referenceIds.count - 1, "greedy continuation diverges")
    }

    // The tekken tokenizer read from the release's tekken.json encodes token-exactly to the ids the
    // reference mistral-common tokenizer produces, and round-trips.
    func testTekkenTokenizerMatchesTheReference() throws {
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_VOXTRAL"],
              let tokenizer = NFKMLXTekkenTokenizer(
                tekkenURL: URL(fileURLWithPath: directory).appendingPathComponent("tekken.json"))
        else { throw XCTSkip("set IK_VAL_VOXTRAL (release dir with tekken.json)") }
        XCTAssertEqual(tokenizer.encode("The quick brown fox.").map(\.intValue), [1784, 7586, 22980, 94137, 1046])
        XCTAssertEqual(tokenizer.encode("hello world").map(\.intValue), [29706, 4304])
        XCTAssertEqual(tokenizer.encode("lang:en").map(\.intValue), [9909, 1058, 1262])
        XCTAssertEqual(tokenizer.specialTokenId("[AUDIO]"), 24)
        XCTAssertEqual(tokenizer.specialTokenId("[TRANSCRIBE]"), 34)
        XCTAssertEqual(tokenizer.bosTokenId, 1)
        XCTAssertEqual(tokenizer.eosTokenId, 2)
        let sentence = "The quick brown fox jumps over the lazy dog."
        XCTAssertEqual(tokenizer.decode(tokenizer.encode(sentence)), sentence)
    }

    // The backend runs end to end from the release: mel front-end, tekken tokenizer, transcription
    // prompt, and greedy decoding, transcribing a real clip.
    func testBackendTranscribesSpeech() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_VOXTRAL"], let wavPath = env["IK_SPEECH_WAV"],
              let audioData = FileManager.default.contents(atPath: wavPath)
        else { throw XCTSkip("set IK_VAL_VOXTRAL and IK_SPEECH_WAV") }
        let backend = try NFKMLXVoxtral.backend(directoryURL: URL(fileURLWithPath: directory), precision: .float32)
        let request = NFKInferenceRequest(inputs: [NFKInputAudio: audioData],
                                          parameters: [NFKParameterMaxTokens: 64])
        let text = try backend.runInference(for: request).text ?? ""
        print("VALIDATION voxtral backend transcription: \(text.debugDescription)")
        XCTAssertFalse(text.isEmpty, "the backend produced no text")
    }
}
