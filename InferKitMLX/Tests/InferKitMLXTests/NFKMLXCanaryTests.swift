//
//  NFKMLXCanaryTests.swift
//  InferKitMLXTests
//
//  Canary-1B-v2 (NVIDIA NeMo `EncDecMultiTaskModel`): the biased FastConformer encoder (reused from
//  Parakeet) and the attention encoder-decoder, measured on the RELEASED weights against NeMo's own
//  model on the validation speech clip (`run_reference.py canary`), seam by seam — the normalized mel
//  features, the subsampler, the first conformer layer, the encoder output, the decoder's first-step
//  logits at the transcription prompt — and token-exact greedy decoding with the exact transcription.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXCanaryTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Double {
        eval(mine)
        let a = mine.reshaped([-1]).asArray(Float.self).map(Double.init)
        let b = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
        let n = min(a.count, b.count)
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-20)
    }

    // The released canary-1b-v2 on the validation clip, seam by seam against NeMo's own
    // EncDecMultiTaskModel: features, the subsampler, the first conformer layer, the encoder output, the
    // decoder's first-step logits at the transcription prompt, then the greedy token sequence and text.
    func testCanaryMatchesTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let path = env["IK_PARITY_CANARY"], let directory = env["IK_VAL_CANARY"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_CANARY + IK_VAL_CANARY (run_reference.py canary, real weights)")
        }
        let arrays = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let waveform = arrays["waveform"]!.asArray(Float.self)
        let prompt = arrays["prompt"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let referenceTokens = arrays["tokens"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let referenceText = String(decoding: arrays["text"]!.asType(.int32).asArray(Int32.self).map { UInt8($0) }, as: UTF8.self)

        let net = NFKMLXCanaryNet(.v2)
        let releaseURL = URL(fileURLWithPath: directory)
        try NFKMLXCanary.loadWeights(into: net, from: releaseURL.appendingPathComponent("model_weights.ckpt"))

        func check(_ label: String, _ mine: MLXArray, _ key: String, _ threshold: Double = 0.999) {
            let similarity = cosine(mine, arrays[key]!)
            print("VALIDATION PARITY canary: \(label) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, threshold, "canary \(label) matches the reference")
        }
        let features = net.frontEnd.features(waveform)
        check("features", features, "features")
        let pre = net.encoder.preEncode(features)
        check("pre_encode", pre, "pre")
        let posEmb = net.encoder.positionEmbedding(length: pre.dim(1))
        check("layer0", net.encoder.layers[0](pre, posEmb: posEmb), "layer0")
        let encoded = net.encode(features: features)
        check("encoded", encoded, "encoded")

        // The output head is a tied-embedding classifier, so the raw logits carry a large shared
        // component; compared in the reference's own log-softmax space (argmax-invariant), as with
        // Parakeet's joint.
        let promptTensor = MLXArray(prompt.map(Int32.init)).reshaped([1, prompt.count])
        let logits0 = net.logits(tokens: promptTensor, encoder: encoded)[0, prompt.count - 1, 0...]
        let mineLogProb = logits0 - logSumExp(logits0, axis: -1, keepDims: true)
        let refLogits = arrays["logits0"]!
        let refLogProb = refLogits - logSumExp(refLogits, axis: -1, keepDims: true)
        let logitSimilarity = cosine(mineLogProb, refLogProb)
        print("VALIDATION PARITY canary: logits0 (log-softmax) cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.999, "canary logits0 matches the reference")

        let tokens = net.decode(encoded: encoded, prompt: prompt)
        XCTAssertEqual(tokens, referenceTokens, "the greedy decode matches the reference")
        let tokenizer = try NFKMLXCanaryTokenizer(tokenizerURL: releaseURL.appendingPathComponent("tokenizer.json"))
        let text = tokenizer.text(for: tokens)
        print("VALIDATION PARITY canary: transcription \(text.debugDescription)")
        XCTAssertEqual(text, referenceText)
    }

    // Every released tensor, remapped from NeMo's names onto the module, matches a module parameter by
    // shape and nothing is left over (Conv weights compared in PyTorch's `[out, in, k]` against the
    // module's channels-last layout; the fixed position table is computed, not a parameter).
    func testEveryParameterMatchesTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let path = env["IK_PARITY_CANARY"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_CANARY (run_reference.py canary)")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        var releasedMapped = [String: [Int]]()
        for (key, value) in record where key.hasPrefix("w::") {
            guard let name = NFKMLXCanary.remapReferenceKey(String(key.dropFirst(3))) else { continue }
            // The record stores conv weights in PyTorch order; compare against the module's transpose.
            releasedMapped[name] = value.shape
        }
        let net = NFKMLXCanaryNet(.v2)
        var consumed = Set<String>(), missing = [String](), mismatched = [String]()
        for (name, value) in net.parameters().flattened() {
            let expected = value.ndim == 4 && name.hasSuffix(".weight")
                ? [value.dim(0), value.dim(3), value.dim(1), value.dim(2)]
                : (value.ndim == 3 && name.hasSuffix(".weight") ? [value.dim(0), value.dim(2), value.dim(1)] : value.shape)
            guard let shape = releasedMapped[name] else { missing.append(name); continue }
            consumed.insert(name)
            if shape != expected { mismatched.append("\(name): built \(expected), released \(shape)") }
        }
        let unaccounted = releasedMapped.keys.filter { !consumed.contains($0) }.sorted()
        print("VALIDATION structure canary-1b-v2: \(consumed.count) consumed, \(missing.count) missing, "
              + "\(mismatched.count) mismatched, \(unaccounted.count) unaccounted")
        XCTAssertTrue(mismatched.isEmpty, "shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(missing.isEmpty, "absent from the release:\n" + missing.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(unaccounted.isEmpty, "released tensors nothing reads:\n" + unaccounted.prefix(8).joined(separator: "\n"))
    }

    // The public backend path on the released weights: the unpacked `.nemo` directory →
    // `backendWithDirectoryURL:` → the validation WAV under NFKInputAudio → the reference transcription.
    func testCanaryBackendTranscribesTheValidationClip() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_CANARY"], let audio = env["IK_VAL_AUDIO"] ?? env["IK_SPEECH_WAV"],
              FileManager.default.fileExists(atPath: directory) else {
            throw XCTSkip("set IK_VAL_CANARY + IK_VAL_AUDIO")
        }
        let backend = try NFKMLXCanary.backend(directoryURL: URL(fileURLWithPath: directory))
        let asset = NFKAudioAsset(fileURL: URL(fileURLWithPath: audio), durationSeconds: 3.47, sampleRate: 16000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        print("VALIDATION PARITY canary-backend: \(result.text.debugDescription)")
        XCTAssertEqual(result.text, "The quick brown fox jumps over the lazy dog.")
    }
}
