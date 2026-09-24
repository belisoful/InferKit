//
//  NFKMLXNemotronHTests.swift
//  InferKitMLXTests
//
//  Nemotron-H (`NemotronHForCausalLM`, NVIDIA), the hybrid Mamba-attention decoder behind Nemotron
//  Nano 2, built on the reused Codestral SSD mixer. Numeric parity is measured at a tiny random
//  configuration against transformers' own implementation (`run_reference.py nemotron_h`), seam by
//  seam: the embedding, each hybrid block's output, the final-normed state, and the logits. The tiny
//  config sets `n_groups` to 2 so the Mamba mixer's gated output norm exercises its grouped path.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXNemotronHTests: XCTestCase {

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

    /// The tiny geometry `run_reference.py nemotron_h` records: a Mamba, an MLP, a Mamba, an attention,
    /// a Mamba, an MLP layer, with `n_groups` 2 (grouped gated norm) and no scalar multipliers.
    private var tinyConfiguration: NFKMLXNemotronHConfiguration {
        NFKMLXNemotronHConfiguration(
            hiddenSize: 64, vocabularySize: 128, rmsEpsilon: 1e-5,
            headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            mambaHeadCount: 8, mambaHeadDimensions: 16, mambaGroupCount: 2, mambaStateSize: 16,
            mambaConvolutionKernel: 4, mambaConvolutionBias: true, mambaProjectionBias: false,
            timeStepMinimum: 0.001, intermediateSize: 96, mlpBias: false,
            layerTypes: [.mamba, .mlp, .mamba, .attention, .mamba, .mlp])
    }

    private func loadRecordWeights(into net: NFKMLXNemotronHNet,
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

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXNemotronH.makeNet(tinyConfiguration).parameters().flattened().map(\.0))
        for expected in ["model.embeddings.weight",
                         "model.layers.0.norm.weight",
                         "model.layers.0.mixer.in_proj.weight",
                         "model.layers.0.mixer.conv1d.weight",
                         "model.layers.0.mixer.A_log",
                         "model.layers.0.mixer.norm.weight",
                         "model.layers.1.mixer.up_proj.weight",
                         "model.layers.1.mixer.down_proj.weight",
                         "model.layers.3.mixer.q_proj.weight",
                         "model.layers.3.mixer.o_proj.weight",
                         "model.norm_f.weight", "lm_head.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
        // Each block's single mixer carries only its own kind's tensors.
        XCTAssertFalse(names.contains("model.layers.1.mixer.in_proj.weight"))     // mlp layer
        XCTAssertFalse(names.contains("model.layers.0.mixer.q_proj.weight"))      // mamba layer
    }

    func testTinyConfigSeamParityAgainstTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let recordPath = env["IK_PARITY_NEMOTRON_H_TINY"] else {
            throw XCTSkip("set IK_PARITY_NEMOTRON_H_TINY (run_reference.py nemotron_h)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let net = NFKMLXNemotronH.makeNet(tinyConfiguration)
        try loadRecordWeights(into: net, from: rec)

        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])
        let blocks = net.blockStates(tokens)                                // [embed, L0 … L5]
        for index in 0 ..< tinyConfiguration.layerCount {
            let similarity = cosine(blocks[index][0], rec["hidden.\(index)"]!)
            print("SEAM nemotron hidden.\(index): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "seam hidden.\(index) diverges")
        }
        let finalSimilarity = cosine(net.hiddenStates(tokens)[0],
                                     rec["hidden.\(tinyConfiguration.layerCount)"]!)
        print("SEAM nemotron hidden.\(tinyConfiguration.layerCount) (post norm): cosine \(finalSimilarity)")
        XCTAssertGreaterThan(finalSimilarity, 0.9999, "final normed seam diverges")

        let logitSimilarity = cosine(net(tokens)[0], rec["output"]!)
        print("VALIDATION PARITY nemotron_h: logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.9999, "logits diverge")
    }

    // The released Nemotron Nano 2 (9B-v2) held to the module by shape: every parameter matches the
    // checkpoint by name and shape (the depthwise convolution squeezed [C, 1, K] → [C, K]).
    func testEveryParameterMatchesTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let shapesPath = env["IK_SHAPES_NEMOTRON_H"], let configPath = env["IK_CONFIG_NEMOTRON_H"],
              let data = FileManager.default.contents(atPath: shapesPath),
              let released = try JSONSerialization.jsonObject(with: data) as? [String: [Int]]
        else { throw XCTSkip("set IK_SHAPES_NEMOTRON_H and IK_CONFIG_NEMOTRON_H (shapes.py)") }

        let config = try NFKMLXNemotronH.configuration(
            fromDirectory: URL(fileURLWithPath: configPath).deletingLastPathComponent())
        XCTAssertEqual(config.layerCount, 56)
        let net = NFKMLXNemotronH.makeNet(config)

        // The released checkpoint uses the original `backbone.*` naming; the module tree uses the
        // transformers-integrated `model.*` naming, so remap the release before comparing (as the
        // directory loader does), and drop the multi-token-prediction tensors nothing reads.
        let releasedMapped = Dictionary(uniqueKeysWithValues: released.compactMap { key, shape -> (String, [Int])? in
            if key.hasPrefix("mtp.") || key.hasPrefix("model.mtp") { return nil }
            let name = key.hasPrefix("backbone.") ? "model." + key.dropFirst("backbone.".count) : key
            return (name, shape)
        })

        var consumed = Set<String>()
        var missing = [String](), mismatched = [String]()
        for (name, value) in net.parameters().flattened() {
            let expected = name.hasSuffix("conv1d.weight") ? [value.dim(0), 1, value.dim(1)] : value.shape
            guard let shape = releasedMapped[name] else { missing.append(name); continue }
            consumed.insert(name)
            if shape != expected { mismatched.append("\(name): built \(expected), released \(shape)") }
        }
        let unaccounted = releasedMapped.keys.filter { !consumed.contains($0) }.sorted()
        print("VALIDATION structure nemotron-nano-9b-v2: \(consumed.count) consumed, \(missing.count) missing, "
              + "\(mismatched.count) mismatched, \(unaccounted.count) unaccounted")
        XCTAssertTrue(mismatched.isEmpty, "shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(missing.isEmpty, "absent from the release:\n" + missing.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(unaccounted.isEmpty, "released tensors nothing reads:\n" + unaccounted.prefix(8).joined(separator: "\n"))
    }

    // The released Nemotron Nano 2 run numerically at bfloat16 against a bf16 transformers oracle
    // (`run_reference.py nemotron_h_real`): each layer's hidden state, the logits, and the greedy
    // continuation. The 9B does not fit float32 on 32 GB, so both sides run bf16.
    func testReleasedWeightsNumericParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_NEMOTRON_H"], let recordPath = env["IK_PARITY_NEMOTRON_H_REAL"] else {
            throw XCTSkip("set IK_VAL_NEMOTRON_H (release dir) and IK_PARITY_NEMOTRON_H_REAL (nemotron_h_real)")
        }
        let dir = URL(fileURLWithPath: directory)
        let config = try NFKMLXNemotronH.configuration(fromDirectory: dir)
        let net = NFKMLXNemotronH.makeNet(config)
        try NFKMLXNemotronH.loadWeights(into: net, fromDirectory: dir, precision: .checkpoint)

        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let tokens = rec["tokens"]!.asType(.int32).reshaped([1, -1])
        let blocks = net.blockStates(tokens)
        var worst: Float = 1
        for index in 0 ..< config.layerCount {
            worst = Swift.min(worst, cosine(blocks[index][0], rec["hidden.\(index)"]!))
        }
        let logitSimilarity = cosine(net(tokens)[0], rec["output"]!)
        print("VALIDATION PARITY nemotron-nano-9b-v2 released: worst block seam \(worst), "
              + "logit cosine \(logitSimilarity)")
        XCTAssertGreaterThan(worst, 0.99, "a block seam diverges")
        XCTAssertGreaterThan(logitSimilarity, 0.99, "released logits diverge")

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
        print("VALIDATION PARITY nemotron-nano-9b-v2 released: greedy \(matches)/\(referenceIds.count) tokens match")
        XCTAssertEqual(produced.first, referenceIds.first, "first greedy token differs")
        XCTAssertGreaterThanOrEqual(matches, referenceIds.count - 2, "greedy continuation diverges")
    }

    // Nemotron's byte-level BPE (Split + ByteLevel pre-tokenizer), read from the release's tokenizer.json
    // through the shared release-tokenizer reader, encodes token-exactly to the ids the reference
    // `tokenizers` library produces (no special tokens added, matching the backend's raw-prompt
    // encoding). A tokenizer-only directory suffices; the 17.8 GB weights are not needed.
    func testTokenizerAgreesWithTheReference() throws {
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_TOK_NEMOTRON_H"] ?? env["IK_VAL_NEMOTRON_H"] else {
            throw XCTSkip("set IK_TOK_NEMOTRON_H (directory with tokenizer.json)")
        }
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: URL(fileURLWithPath: directory)) else {
            return XCTFail("the Nemotron release has no readable tokenizer")
        }
        XCTAssertEqual(tokenizer.encode("def fibonacci(n):").map(\.intValue),
                       [3149, 9111, 87539, 4990, 4244], "tokenization diverges from the reference")
        XCTAssertEqual(tokenizer.encode("The quick brown fox").map(\.intValue),
                       [1784, 7586, 22980, 94137], "tokenization diverges from the reference")
        print("VALIDATION nemotron tokenizer eos id: \(tokenizer.eosTokenId)")
    }
}
