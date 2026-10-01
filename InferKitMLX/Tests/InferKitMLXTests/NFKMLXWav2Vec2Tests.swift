//
//  NFKMLXWav2Vec2Tests.swift
//  InferKitMLXTests
//
//  Wav2Vec2 and HuBERT (facebook/wav2vec2-*, facebook/hubert-*, Apache-2.0). Each release is measured
//  against its own record from `run_reference.py wav2vec2 --checkpoint <release>`, at float32, seam by
//  seam: the feature extractor's normalization, the convolutional feature encoder, the projection, the
//  first, middle, and last encoder layers, the last hidden state, and for a CTC release the logits, the
//  greedy tokens, and the decoded text. A release runs when its `IK_VAL_<NAME>` directory and
//  `IK_PARITY_<NAME>` record are both present.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXWav2Vec2Tests: XCTestCase {

    static let releases = ["WAV2VEC2_BASE_960H", "WAV2VEC2_BASE", "WAV2VEC2_LARGE_960H", "WAV2VEC2_LARGE_960H_LV60_SELF",
                           "WAV2VEC2_LARGE_LV60", "WAV2VEC2_LARGE_XLSR_53", "WAV2VEC2_XLS_R_300M", "WAV2VEC2_XLS_R_1B",
                           "WAV2VEC2_XLS_R_2B", "HUBERT_BASE_LS960", "HUBERT_LARGE_LL60K", "HUBERT_LARGE_LS960_FT",
                           "HUBERT_XLARGE_LL60K", "HUBERT_XLARGE_LS960_FT"]

    override func tearDown() {
        Memory.clearCache()
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    static func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Double {
        let a = mine.asType(.float32).reshaped([-1]), b = reference.asType(.float32).reshaped([-1])
        eval(a, b)
        let x = a.asArray(Float.self), y = b.asArray(Float.self)
        precondition(x.count == y.count, "compared tensors differ in size: \(x.count) and \(y.count)")
        var dot = 0.0, nx = 0.0, ny = 0.0
        for i in 0 ..< x.count {
            dot += Double(x[i]) * Double(y[i])
            nx += Double(x[i]) * Double(x[i])
            ny += Double(y[i]) * Double(y[i])
        }
        return dot / ((nx * ny).squareRoot() + 1e-300)
    }

    /// The module keys mirror the transformers checkpoint below its family prefix, for both the
    /// post-norm base layout and the pre-norm large one.
    func testParameterNamesFollowTheCheckpoint() throws {
        try requireMLXRuntime()
        var large = NFKMLXWav2Vec2Configuration()
        large.featureNorm = .layer
        large.stableLayerNorm = true
        large.convBias = true
        large.numHiddenLayers = 2
        for configuration in [NFKMLXWav2Vec2Configuration.base960h, large] {
            let names = Set(NFKMLXWav2Vec2Net(configuration).parameters().flattened().map(\.0))
            var expected = ["feature_extractor.conv_layers.0.conv.weight", "feature_extractor.conv_layers.0.layer_norm.weight",
                            "feature_projection.layer_norm.weight", "feature_projection.projection.weight",
                            "encoder.pos_conv_embed.conv.weight_g", "encoder.pos_conv_embed.conv.weight_v",
                            "encoder.pos_conv_embed.conv.bias",
                            "encoder.layer_norm.weight", "encoder.layers.1.attention.q_proj.weight",
                            "encoder.layers.1.attention.out_proj.bias", "encoder.layers.1.feed_forward.intermediate_dense.weight",
                            "encoder.layers.1.final_layer_norm.bias", "masked_spec_embed"]
            if configuration.featureNorm == .layer {
                expected += ["feature_extractor.conv_layers.6.layer_norm.bias", "feature_extractor.conv_layers.6.conv.bias"]
            } else {
                expected += ["lm_head.weight", "lm_head.bias"]
                XCTAssertFalse(names.contains("feature_extractor.conv_layers.1.layer_norm.weight"),
                               "the group-norm variant normalizes only its first convolution")
            }
            for name in expected { XCTAssertTrue(names.contains(name), "missing \(name)") }
        }
    }

    /// `weight_norm(dim=2)` resolves to `g · v / ‖v‖` with the norm taken per kernel tap over the output
    /// and input channels, in MLX's `[out, taps, in]` layout.
    func testWeightNormResolvesPerKernelTap() throws {
        try requireMLXRuntime()
        let conv = NFKWav2Vec2WeightNormConv(channels: 6, kernel: 4, groups: 2)
        let direction = MLXRandom.normal([6, 4, 3], key: MLXRandom.key(1))
        let gain = MLXRandom.normal([1, 4, 1], key: MLXRandom.key(2))
        conv.update(parameters: ModuleParameters.unflattened(["weight_v": direction, "weight_g": gain]))
        let w = conv.weight.asArray(Float.self), v = direction.asArray(Float.self), g = gain.asArray(Float.self)
        for tap in 0 ..< 4 {
            var norm: Float = 0
            for o in 0 ..< 6 { for i in 0 ..< 3 { norm += v[(o * 4 + tap) * 3 + i] * v[(o * 4 + tap) * 3 + i] } }
            norm = norm.squareRoot()
            for o in 0 ..< 6 {
                for i in 0 ..< 3 {
                    XCTAssertEqual(w[(o * 4 + tap) * 3 + i], g[tap] * v[(o * 4 + tap) * 3 + i] / norm, accuracy: 1e-5)
                }
            }
        }
    }

    /// Greedy CTC decoding collapses repeats, drops the blank, keeps a repeat the blank separates, and
    /// reads the word delimiter as a space; the label map is its inverse.
    func testCTCDecodingCollapsesRepeatsAndDropsTheBlank() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let vocabulary = ["<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "L": 7, "H": 8]
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        let tokenizer = try NFKMLXWav2Vec2Tokenizer(directoryURL: directory)
        XCTAssertEqual(tokenizer.text(forFrameTokens: [0, 8, 8, 5, 0, 7, 7, 0, 7, 4, 4, 6, 0, 0]), "HELL T")
        XCTAssertEqual(tokenizer.labels(for: "HE LT"), [8, 5, 4, 7, 6])
        XCTAssertEqual(tokenizer.labels(for: "X"), [3])
    }

    /// PARITY: every release against its own record, at float32.
    func testEveryReleaseIsAtParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        let only = env["IK_WAV2VEC2_ONLY"]
        var measured = [String]()
        for name in Self.releases where only == nil || only == name {
            guard let directory = env["IK_VAL_\(name)"], let recordPath = env["IK_PARITY_\(name)"],
                  FileManager.default.fileExists(atPath: recordPath) else { continue }
            try autoreleasepool { try measure(name, directory: URL(fileURLWithPath: directory), recordPath: recordPath) }
            Memory.clearCache()
            measured.append(name)
        }
        if measured.isEmpty {
            throw XCTSkip("set IK_VAL_<RELEASE> (release directory) and IK_PARITY_<RELEASE> (oracle record)")
        }
    }

    private func measure(_ name: String, directory: URL, recordPath: String) throws {
        let net = try NFKMLXWav2Vec2Net(configurationURL: directory.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: directory)
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let waveform = try XCTUnwrap(record["waveform"]).asArray(Float.self)
        let referenceInput = try XCTUnwrap(record["input_values"])

        let input = NFKMLXWav2Vec2Processor.inputValues(waveform, normalize: net.configuration.normalizesInput)
        let inputDifference = abs(input[0] - referenceInput).max().item(Float.self)
        XCTAssertLessThan(inputDifference, 1e-5, "\(name) feature-extractor normalization")

        let seams = net.seams(referenceInput.reshaped([1, -1]))
        var measured: [(String, Double)] = [("features", Self.cosine(seams.features[0], record["features"]!)),
                                            ("projected", Self.cosine(seams.projected[0], record["projected"]!))]
        let indices = try XCTUnwrap(record["layer_indices"]).asArray(Int64.self).map(Int.init)
        for index in indices {
            measured.append(("layer\(index)", Self.cosine(seams.layers[index][0], try XCTUnwrap(record["layer\(index)"]))))
        }
        measured.append(("output", Self.cosine(seams.output[0], record["output"]!)))
        var line = "VALIDATION PARITY wav2vec2 \(name): input max |d| \(inputDifference), "
            + measured.map { "\($0.0) \($0.1)" }.joined(separator: ", ")

        if let referenceLogits = record["logits"] {
            let logits = try XCTUnwrap(seams.logits)[0]
            let similarity = Self.cosine(logits, referenceLogits)
            measured.append(("logits", similarity))
            let maximum = abs(logits - referenceLogits).max().item(Float.self)
            let tokens = argMax(logits, axis: -1).asArray(Int32.self).map(Int.init)
            let referenceTokens = try XCTUnwrap(record["tokens"]).asArray(Int64.self).map(Int.init)
            XCTAssertEqual(tokens, referenceTokens, "\(name) greedy CTC tokens")
            let tokenizer = try NFKMLXWav2Vec2Tokenizer(directoryURL: directory)
            let referenceText = String(decoding: try XCTUnwrap(record["text_bytes"]).asArray(Int64.self).map(UInt8.init),
                                       as: UTF8.self)
            XCTAssertEqual(tokenizer.text(forFrameTokens: tokens), referenceText, "\(name) decoded text")
            line += ", logits \(similarity) (max |d| \(maximum)), tokens \(tokens == referenceTokens ? "exact" : "DIFFER"), "
                + "text \(referenceText.debugDescription)"
        }
        print(line)
        for (seam, similarity) in measured {
            XCTAssertGreaterThan(similarity, 0.99999, "\(name) \(seam) diverges")
        }

        // End to end through the public factory on the same clip, from WAV bytes.
        let backend = try NFKMLXWav2Vec2.backend(directoryURL: directory)
        // The clip the oracle read, byte for byte: re-encoding the samples would move most of them by one
        // 16-bit step, since the writer truncates.
        let wav = try Data(contentsOf: URL(fileURLWithPath: NFKMLXValidationConfig.environment["IK_VAL_AUDIO"]
            ?? NFKMLXValidationConfig.root.appendingPathComponent("inputs/speech.wav").path))
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wav]))
        let embedding = try XCTUnwrap(result.output(forKey: NFKOutputEmbedding) as? [NSNumber])
        XCTAssertEqual(embedding.count, net.configuration.hiddenSize)
        let pooled = MLXArray(embedding.map(\.floatValue))
        XCTAssertGreaterThan(Self.cosine(pooled, record["output"]!.mean(axis: 0)), 0.9999, "\(name) backend embedding")
        if record["text_bytes"] != nil {
            let referenceText = String(decoding: record["text_bytes"]!.asArray(Int64.self).map(UInt8.init), as: UTF8.self)
            XCTAssertEqual(result.output(forKey: NFKOutputText) as? String, referenceText, "\(name) backend transcription")
        } else {
            XCTAssertNil(result.output(forKey: NFKOutputText), "\(name) has no CTC head")
        }
    }
}
