//
//  NFKMLXMambaTests.swift
//  InferKitMLXTests
//
//  The Mamba-2 decoder (`Mamba2ForCausalLM`), the toolkit's first state-space model. The module
//  evaluates MLX arrays, so these skip without a Metal library for MLX (see Tools/mlx-metallib.sh).
//
//  Numeric parity is measured at a tiny random configuration against transformers' own
//  Mamba2ForCausalLM (`run_reference.py mamba2`), seam by seam: the embedding output, each block's
//  output, the final-normed hidden state, and the logits. The record carries the weights under the
//  release's own names, so the same record drives the weight load and the comparison
//  (`IK_PARITY_MAMBA2_TINY`).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXMambaTests: XCTestCase {

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

    /// The tiny geometry `run_reference.py mamba2` records.
    private var tinyConfiguration: NFKMLXMamba2Configuration {
        NFKMLXMamba2Configuration(
            hiddenSize: 64, layerCount: 3, vocabularySize: 128, rmsEpsilon: 1e-5,
            intermediateSize: 128, headCount: 8, headDimensions: 16, stateSize: 16,
            groupCount: 2, convolutionKernel: 4, useConvolutionBias: true,
            useProjectionBias: false, tiesWordEmbeddings: false)
    }

    /// Loads a record's `w::`-prefixed weights into a net, squeezing the depthwise convolution.
    private func loadRecordWeights(into net: NFKMLXMamba2Net, from record: [String: MLXArray]) throws {
        let weights = record.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = String(key.dropFirst(3))
            let array = name.hasSuffix("conv1d.weight") && value.ndim == 3
                ? value.reshaped([value.dim(0), value.dim(2)]) : value
            return (name, array)
        }
        try NFKMLXWeights.apply(weights, to: net)
    }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXMamba.makeNet(tinyConfiguration).parameters().flattened().map(\.0))
        for expected in ["backbone.embeddings.weight",
                         "backbone.layers.0.norm.weight",
                         "backbone.layers.0.mixer.in_proj.weight",
                         "backbone.layers.0.mixer.conv1d.weight",
                         "backbone.layers.0.mixer.conv1d.bias",
                         "backbone.layers.0.mixer.A_log",
                         "backbone.layers.0.mixer.D",
                         "backbone.layers.0.mixer.dt_bias",
                         "backbone.layers.0.mixer.norm.weight",
                         "backbone.layers.0.mixer.out_proj.weight",
                         "backbone.norm_f.weight", "lm_head.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // The released Codestral-Mamba-7B does not fit this machine at float32, so its checkpoint is held
    // to the module by shape: every parameter this module builds exists in the release at the built
    // shape (the depthwise convolution squeezed [C, 1, K] → [C, K]), and every released tensor is one
    // this module consumes. MLX arrays are lazy, so a 7B module costs nothing to build until evaluated.
    func testEveryParameterMatchesTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let shapesPath = env["IK_SHAPES_MAMBA2"], let configPath = env["IK_CONFIG_MAMBA2"],
              let data = FileManager.default.contents(atPath: shapesPath),
              let released = try JSONSerialization.jsonObject(with: data) as? [String: [Int]]
        else { throw XCTSkip("set IK_SHAPES_MAMBA2 and IK_CONFIG_MAMBA2 (Tools/validation-assets/shapes.py)") }

        let config = try NFKMLXMamba.configuration(
            fromDirectory: URL(fileURLWithPath: configPath).deletingLastPathComponent())
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.layerCount, 64)
        XCTAssertEqual(config.headCount, 128)
        XCTAssertFalse(config.tiesWordEmbeddings)
        let net = NFKMLXMamba.makeNet(config)

        var consumed = Set<String>()
        var missing = [String](), mismatched = [String]()
        for (name, value) in net.parameters().flattened() {
            // The depthwise convolution is held [C, K] and released [C, 1, K].
            let expected = name.hasSuffix("conv1d.weight")
                ? [value.dim(0), 1, value.dim(1)] : value.shape
            guard let shape = released[name] else { missing.append(name); continue }
            consumed.insert(name)
            if shape != expected {
                mismatched.append("\(name): built \(expected), released \(shape)")
            }
        }
        let unaccounted = released.keys.filter { !consumed.contains($0) }.sorted()
        print("VALIDATION structure mamba-codestral-7b: \(consumed.count) released tensors consumed, "
              + "\(missing.count) missing, \(mismatched.count) mismatched, \(unaccounted.count) unaccounted")
        XCTAssertTrue(mismatched.isEmpty, "shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(missing.isEmpty, "absent from the release:\n" + missing.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(unaccounted.isEmpty, "released tensors nothing here reads:\n"
                      + unaccounted.prefix(8).joined(separator: "\n"))
        XCTAssertEqual(consumed.count, released.count)
    }

    // The released Codestral-Mamba-7B run at bfloat16 (float32 does not fit 32 GB), against a bf16
    // transformers oracle (`run_reference.py mamba2_real`): the prompt logits by cosine and the greedy
    // continuation token for token. bf16 is lossy, so the logit cosine sits below the tiny-config's 1.0
    // and an occasional near-tie can flip a greedy token, as in the GGUF and dense-decoder checks.
    func testReleasedWeightsBf16Parity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_MAMBA2"], let recordPath = env["IK_PARITY_MAMBA2_REAL"] else {
            throw XCTSkip("set IK_VAL_MAMBA2 (release dir) and IK_PARITY_MAMBA2_REAL (run_reference.py mamba2_real)")
        }
        let dir = URL(fileURLWithPath: directory)
        let config = try NFKMLXMamba.configuration(fromDirectory: dir)
        let net = NFKMLXMamba.makeNet(config)
        try NFKMLXMamba.loadWeights(into: net, fromDirectory: dir, precision: .checkpoint)  // bf16 as stored

        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])

        let logitSimilarity = cosine(net(tokens)[0], rec["output"]!)
        print("VALIDATION PARITY mamba2 released bf16: prompt logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.99, "released prompt logits diverge")

        // Greedy continuation, prefill-only (the module recomputes the prefix each step).
        let reference = rec["continuation"]!.asType(.int32)
        eval(reference)
        let referenceIds = reference.asArray(Int32.self)
        var sequence = tokens
        var produced = [Int32]()
        for _ in 0 ..< referenceIds.count {
            let logits = net(sequence)
            let next = argMax(logits[0, -1, 0...], axis: -1)
            eval(next)
            let id = next.item(Int32.self)
            produced.append(id)
            sequence = concatenated([sequence, MLXArray([id]).reshaped([1, 1])], axis: 1)
        }
        let matches = zip(produced, referenceIds).filter { $0 == $1 }.count
        print("VALIDATION PARITY mamba2 released bf16: greedy \(matches)/\(referenceIds.count) tokens match")
        XCTAssertEqual(produced.first, referenceIds.first, "first greedy token differs")
        XCTAssertGreaterThanOrEqual(matches, referenceIds.count - 2, "greedy continuation diverges beyond near-ties")
    }

    // The Mistral tokenizer reproduces the released tokenizer's ids. The reference ids are the
    // `tokenizers` fast library's over the release's own tokenizer.json (no sentencepiece), which is a
    // pure function of the shipped file. Runs without a Metal library (the tokenizer touches no MLX).
    func testTokenizerAgreesWithTheReference() throws {
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_MAMBA2"] else {
            throw XCTSkip("set IK_VAL_MAMBA2 (release dir with tokenizer.json)")
        }
        guard let tok = NFKMLXMistralTokenizer(directoryURL: URL(fileURLWithPath: directory)) else {
            return XCTFail("no readable tokenizer.json")
        }
        let cases: [(String, [Int])] = [
            ("def fibonacci(n):", [1569, 16950, 1034, 28895, 29500, 29479, 2097]),
            ("Hello, world!", [23325, 29493, 2294, 29576]),
            ("The quick brown fox", [1183, 3704, 9828, 1053, 1910]),
            (" leading space", [6142, 3532]),
            ("café", [29113]),
            ("", []),
        ]
        for (text, expected) in cases {
            XCTAssertEqual(tok.encode(text), expected, "encode mismatch for \(text.debugDescription)")
        }
        XCTAssertEqual(tok.id(forToken: "<s>"), 1)
        XCTAssertEqual(tok.id(forToken: "</s>"), 2)
        XCTAssertEqual(tok.decode(tok.encode("def fibonacci(n):")), "def fibonacci(n):")
    }

    // End-to-end wiring: the backend builds from the release directory (bf16), tokenizes a prompt,
    // generates greedily, and decodes to non-empty text. Numeric correctness is the released-weight
    // parity test above; this proves the tokenizer + generation loop + decode are wired.
    func testBackendGeneratesText() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_MAMBA2"] else { throw XCTSkip("set IK_VAL_MAMBA2 (release dir)") }
        let backend = try NFKMLXMamba.backend(directoryURL: URL(fileURLWithPath: directory))
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "def fibonacci(n):"],
            parameters: [NFKParameterMaxTokens: 24, NFKParameterTemperature: 0])
        let result = try backend.runInference(for: request)
        let text = result.text ?? ""
        print("VALIDATION mamba2 backend greedy: \(text.debugDescription)")
        XCTAssertFalse(text.isEmpty, "backend produced no text")
    }

    func testTinyConfigSeamParityAgainstTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let recordPath = env["IK_PARITY_MAMBA2_TINY"] else {
            throw XCTSkip("set IK_PARITY_MAMBA2_TINY (record from run_reference.py mamba2)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        let net = NFKMLXMamba.makeNet(tinyConfiguration)
        try loadRecordWeights(into: net, from: rec)

        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])

        // The scan runs in float32; the record is float32, so a matching seam is exact to many digits.
        let blocks = net.blockStates(tokens)                                // [embed, block0, block1, block2]
        for index in 0 ..< tinyConfiguration.layerCount {                   // hidden.0 … hidden.{N-1}
            let similarity = cosine(blocks[index][0], rec["hidden.\(index)"]!)
            print("SEAM mamba2 hidden.\(index): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "seam hidden.\(index) diverges")
        }
        // hidden.N is the FINAL-normed state (transformers applies norm_f to its last hidden entry).
        let finalHidden = net.hiddenStates(tokens)
        let finalSimilarity = cosine(finalHidden[0], rec["hidden.\(tinyConfiguration.layerCount)"]!)
        print("SEAM mamba2 hidden.\(tinyConfiguration.layerCount) (post norm_f): cosine \(finalSimilarity)")
        XCTAssertGreaterThan(finalSimilarity, 0.9999, "final normed seam diverges")

        let logitSimilarity = cosine(net(tokens)[0], rec["output"]!)
        print("VALIDATION PARITY mamba2: logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.9999, "logits diverge")
    }
}
