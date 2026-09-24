//
//  NFKMLXGraniteSpeechTests.swift
//  InferKitMLXTests
//
//  Granite Speech 3.3-2B (`GraniteSpeechForConditionalGeneration`, IBM): the Conformer CTC encoder, the
//  BLIP-2 Q-former projector, and the dense Granite decoder, measured seam by seam at a tiny random
//  configuration against transformers' own implementation (`run_reference.py granite_speech`): the
//  encoder output, the projector output, and the fused logits (audio features scattered into the prompt).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXGraniteSpeechTests: XCTestCase {

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

    /// The tiny geometry `run_reference.py granite_speech` records.
    private func tinyNet() -> NFKMLXGraniteSpeechNet {
        let encoder = NFKMLXGraniteSpeechEncoderConfiguration(
            inputDim: 16, hiddenDim: 32, outputDim: 24, layerCount: 4, headCount: 2, headDimensions: 16,
            feedForwardMultiplier: 2, convolutionExpansionFactor: 2, convolutionKernel: 5,
            contextSize: 8, maxPositionEmbeddings: 16)
        let projector = NFKMLXGraniteSpeechProjectorConfiguration(
            hiddenSize: 32, layerCount: 2, headCount: 2, intermediateSize: 64, encoderHiddenSize: 32,
            layerNormEpsilon: 1e-12, windowSize: 4, downsampleRate: 2)
        let text = NFKMLXGraniteTextConfiguration(
            hiddenSize: 32, layerCount: 2, headCount: 4, keyValueHeadCount: 2, headDimensions: 8,
            intermediateSize: 64, vocabularySize: 40, ropeTheta: 1_000_000, rmsEpsilon: 1e-5,
            embeddingMultiplier: 2.0, residualMultiplier: 0.5, attentionMultiplier: 0.25,
            logitsScaling: 3.0, tiesWordEmbeddings: false)
        return NFKMLXGraniteSpeechNet(encoder: encoder, projector: projector, text: text, audioTokenId: 39)
    }

    private func loadRecordWeights(into net: NFKMLXGraniteSpeechNet, from record: [String: MLXArray]) throws {
        let weights = record.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = String(key.dropFirst(3))
            if name.hasSuffix("num_batches_tracked") { return nil }
            // PyTorch Conv1d weights are [out, in, kernel]; MLX `Conv1d` is channels-last [out, kernel, in].
            // Only conv weights are 3-D `.weight` tensors (the learned query is 3-D but ends `.query`).
            let array = value.ndim == 3 && name.hasSuffix(".weight") ? value.transposed(0, 2, 1) : value
            return (name, array)
        }
        try NFKMLXWeights.apply(weights, to: net)
        net.train(false)                                                     // BatchNorm running statistics
    }

    func testSeamParityAgainstTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let recordPath = env["IK_PARITY_GRANITE_SPEECH_TINY"] else {
            throw XCTSkip("set IK_PARITY_GRANITE_SPEECH_TINY (run_reference.py granite_speech)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let net = tinyNet()
        try loadRecordWeights(into: net, from: rec)

        let features = rec["features"]!.reshaped([1, 12, 16])
        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])

        let encoderOut = net.encoder(features)
        let encoderSimilarity = cosine(encoderOut[0], rec["encoder_out"]!)
        print("SEAM granite-speech encoder: cosine \(encoderSimilarity)")
        XCTAssertGreaterThan(encoderSimilarity, 0.9999, "encoder output diverges")

        let projectorOut = net.projector(encoderOut)
        let projectorSimilarity = cosine(projectorOut[0], rec["projector_out"]!)
        print("SEAM granite-speech projector: cosine \(projectorSimilarity)")
        XCTAssertGreaterThan(projectorSimilarity, 0.9999, "projector output diverges")

        let logitSimilarity = cosine(net(tokens, features: features)[0], rec["output"]!)
        print("VALIDATION PARITY granite_speech: logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.9999, "fused logits diverge")
    }

    // The released granite-speech-3.3-2b held to the module by shape: every base tensor matches by name
    // and shape (Conv1d weights compared in PyTorch's `[out, in, kernel]` against the module's
    // channels-last `[out, kernel, in]`; a tied release ships no `lm_head`). The audio adapter is a
    // separate file and is not part of this base-tensor accounting.
    func testEveryParameterMatchesTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let shapesPath = env["IK_SHAPES_GRANITE_SPEECH"], let configPath = env["IK_CONFIG_GRANITE_SPEECH"],
              let data = FileManager.default.contents(atPath: shapesPath),
              let released = try JSONSerialization.jsonObject(with: data) as? [String: [Int]]
        else { throw XCTSkip("set IK_SHAPES_GRANITE_SPEECH and IK_CONFIG_GRANITE_SPEECH (shapes.py)") }

        let net = try NFKMLXGraniteSpeech.net(
            fromDirectory: URL(fileURLWithPath: configPath).deletingLastPathComponent())
        var consumed = Set<String>()
        var missing = [String](), mismatched = [String]()
        for (name, value) in net.parameters().flattened() {
            let expected = value.ndim == 3 && name.hasSuffix(".weight")
                ? [value.dim(0), value.dim(2), value.dim(1)] : value.shape       // conv → [out, in, kernel]
            guard let shape = released[name] else { missing.append(name); continue }
            consumed.insert(name)
            if shape != expected { mismatched.append("\(name): built \(expected), released \(shape)") }
        }
        // `num_batches_tracked` are BatchNorm step counters, not parameters; the loader drops them.
        let unaccounted = released.keys.filter { !consumed.contains($0) && !$0.hasSuffix("num_batches_tracked") }.sorted()
        print("VALIDATION structure granite-speech-3.3-2b: \(consumed.count) consumed, \(missing.count) missing, "
              + "\(mismatched.count) mismatched, \(unaccounted.count) unaccounted")
        XCTAssertTrue(mismatched.isEmpty, "shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(missing.isEmpty, "absent from the release:\n" + missing.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(unaccounted.isEmpty, "released tensors nothing reads:\n" + unaccounted.prefix(8).joined(separator: "\n"))
    }

    // The released granite-speech-3.3-2b run numerically at float32 against a float32 transformers oracle
    // with the audio LoRA adapter folded in (`run_reference.py granite_speech_real`): the encoder and
    // projector seams, the fused logits, and the greedy continuation on fixed features.
    func testReleasedWeightsNumericParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_GRANITE_SPEECH"], let recordPath = env["IK_PARITY_GRANITE_SPEECH_REAL"] else {
            throw XCTSkip("set IK_VAL_GRANITE_SPEECH (release dir) and IK_PARITY_GRANITE_SPEECH_REAL (granite_speech_real)")
        }
        let dir = URL(fileURLWithPath: directory)
        let net = try NFKMLXGraniteSpeech.net(fromDirectory: dir)
        try NFKMLXGraniteSpeech.loadWeights(into: net, fromDirectory: dir, precision: .float32)

        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let frames = rec["features"]!.dim(0)
        let features = rec["features"]!.reshaped([1, frames, net.encoderConfiguration.inputDim])
        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])

        let encoderOut = net.encoder(features)
        let encoderSimilarity = cosine(encoderOut[0], rec["encoder_out"]!)
        let audio = net.projector(encoderOut)
        let projectorSimilarity = cosine(audio[0], rec["projector_out"]!)
        let logitSimilarity = cosine(net.logits(tokens: tokens, audioEmbeddings: audio)[0], rec["output"]!)
        print("VALIDATION PARITY granite-speech-3.3-2b released: encoder \(encoderSimilarity), "
              + "projector \(projectorSimilarity), logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(encoderSimilarity, 0.999, "encoder diverges")
        XCTAssertGreaterThan(projectorSimilarity, 0.999, "projector diverges")
        XCTAssertGreaterThan(logitSimilarity, 0.999, "released logits diverge")

        let reference = rec["continuation"]!.asType(.int32)
        eval(reference)
        let referenceIds = reference.asArray(Int32.self)
        var sequence = tokens
        var produced = [Int32]()
        for _ in 0 ..< referenceIds.count {
            let next = argMax(net.logits(tokens: sequence, audioEmbeddings: audio)[0, -1, 0...], axis: -1)
            eval(next)
            let id = next.item(Int32.self)
            produced.append(id)
            sequence = concatenated([sequence, MLXArray([id]).reshaped([1, 1])], axis: 1)
        }
        let matches = zip(produced, referenceIds).filter { $0 == $1 }.count
        print("VALIDATION PARITY granite-speech-3.3-2b released: greedy \(matches)/\(referenceIds.count) tokens match")
        XCTAssertGreaterThanOrEqual(matches, referenceIds.count - 1, "greedy continuation diverges")
    }

    // The stacked log-mel front-end matches Granite Speech's own feature extractor on a real clip: the
    // torchaudio-style Mel spectrogram, the `log10` normalization, and frame-pair stacking.
    func testMelFeaturesMatchTheProcessor() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let wavPath = env["IK_SPEECH_WAV"], let featPath = env["IK_PARITY_GRANITE_SPEECH_FEATURES"],
              let data = FileManager.default.contents(atPath: wavPath),
              let (samples, rate) = NFKMLXWaveFile.read(data)
        else { throw XCTSkip("set IK_SPEECH_WAV and IK_PARITY_GRANITE_SPEECH_FEATURES") }
        let matched = NFKMLXAudioRate.matched(samples, from: rate, to: 16000)
        let mine = NFKMLXGraniteSpeechFeatures()(matched)
        let reference = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: featPath)).arrays["input_features"]!
        let similarity = cosine(mine, reference)
        print("VALIDATION granite-speech mel: cosine \(similarity) (mine \(mine.shape), reference \(reference.shape))")
        XCTAssertGreaterThan(similarity, 0.999, "mel features diverge from the processor")
    }

    // The backend runs end to end from the release: it builds the decoder, the tokenizer, and the mel
    // front-end, and transcribes a real clip to non-empty text.
    func testBackendTranscribesSpeech() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_GRANITE_SPEECH"], let wavPath = env["IK_SPEECH_WAV"],
              let audioData = FileManager.default.contents(atPath: wavPath)
        else { throw XCTSkip("set IK_VAL_GRANITE_SPEECH and IK_SPEECH_WAV") }
        let backend = try NFKMLXGraniteSpeech.backend(directoryURL: URL(fileURLWithPath: directory),
                                                      precision: .float32)
        let request = NFKInferenceRequest(inputs: [NFKInputAudio: audioData],
                                          parameters: [NFKParameterMaxTokens: 64])
        let text = try backend.runInference(for: request).text ?? ""
        print("VALIDATION granite-speech backend transcription: \(text.debugDescription)")
        XCTAssertFalse(text.isEmpty, "the backend produced no text")
    }
}
