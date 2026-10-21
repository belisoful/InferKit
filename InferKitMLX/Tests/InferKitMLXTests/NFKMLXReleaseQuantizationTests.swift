//
//  NFKMLXReleaseQuantizationTests.swift
//  InferKitMLXTests
//
//  mlx_lm quantized releases: the packed layers a release carries, read from its tensors, rebuild the
//  module that wrote them, single-file and sharded, dense and mixture, with per-layer overrides; and
//  the quantizations this package does not read are refused by name.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXReleaseQuantizationTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private let tokens = MLXArray([3, 17, 42, 8, 91, 7, 64, 5].map { Int32($0) }).reshaped([1, 8])

    private func scratchDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("release-quantization-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    /// Quantizes `source` as mlx_lm does by default (every eligible linear and the embedding at 4 bits
    /// in groups of 32), with each of `overrides` at 8 bits in groups of 64 first.
    private func quantizeAsMLXLM(_ source: Module, overrides: [String] = []) throws {
        let layers = Dictionary(source.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        let replaced = overrides.compactMap { path in
            layers[path].flatMap { quantizeSingle(layer: $0, groupSize: 64, bits: 8, mode: .affine) }.map { (path, $0 as Module) }
        }
        try NFKMLXModuleReplacement.place(replaced, in: source)
        try NFKMLXQuantization.quantize(module: source, bits: 4, groupSize: 32, includeEmbeddings: true)
        eval(source)
    }

    /// Writes `arrays` as a release with mlx_lm's `quantization` block in its `config.json`: one file,
    /// or two shards and their index.
    private func writeRelease(_ arrays: [(String, MLXArray)], overrides: [String] = [], sharded: Bool,
                              to directory: URL) throws {
        var quantization: [String: Any] = ["group_size": 32, "bits": 4]
        for path in overrides {
            quantization[path] = ["group_size": 64, "bits": 8]
        }
        let config = try JSONSerialization.data(withJSONObject: ["quantization": quantization,
                                                                 "quantization_config": quantization])
        try config.write(to: directory.appendingPathComponent("config.json"))
        let sorted = arrays.sorted { $0.0 < $1.0 }
        guard sharded else {
            try MLX.save(arrays: Dictionary(sorted, uniquingKeysWith: { first, _ in first }),
                         metadata: ["format": "mlx"], url: directory.appendingPathComponent("model.safetensors"))
            return
        }
        let halves = [Array(sorted.prefix(sorted.count / 2)), Array(sorted.dropFirst(sorted.count / 2))]
        var weightMap = [String: String]()
        for (index, half) in halves.enumerated() {
            let name = "model-0000\(index + 1)-of-00002.safetensors"
            try MLX.save(arrays: Dictionary(half, uniquingKeysWith: { first, _ in first }),
                         metadata: ["format": "mlx"], url: directory.appendingPathComponent(name))
            for (key, _) in half {
                weightMap[key] = name
            }
        }
        let index = try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
        try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
    }

    private func assertLoadsAsWritten(_ directory: URL, source: NFKMLXLanguageNet,
                                      geometry: NFKMLXLanguageConfiguration,
                                      residency: NFKMLXResidency = .resident,
                                      file: StaticString = #filePath, line: UInt = #line) throws -> NFKMLXLanguageNet {
        let loaded = NFKMLXLanguage.makeNet(geometry)
        // A float32 request is overridden: the packed words and their scales load as stored.
        try NFKMLXLanguage.loadWeights(into: loaded, fromDirectory: directory, precision: .float32,
                                       residency: residency)
        XCTAssertEqual(loaded(tokens).asArray(Float.self), source(tokens).asArray(Float.self),
                       "the release computes the logits of the module that wrote it", file: file, line: line)
        return loaded
    }

    // MARK: Dense

    func testASingleFileDenseReleaseRebuildsItsPackedLayers() throws {
        try requireMLXRuntime()
        MLXRandom.seed(21)
        let override = "model.layers.0.mlp.down_proj"
        let source = NFKMLXLanguage.makeNet(.tiny)
        try quantizeAsMLXLM(source, overrides: [override])
        let directory = try scratchDirectory()
        try writeRelease(source.parameters().flattened(), overrides: [override], sharded: false, to: directory)
        let loaded = try assertLoadsAsWritten(directory, source: source, geometry: .tiny)

        let layers = Dictionary(loaded.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        let down = try XCTUnwrap(layers[override] as? QuantizedLinear)
        XCTAssertEqual(down.bits, 8, "the override's geometry is read from its tensors")
        XCTAssertEqual(down.groupSize, 64)
        let query = try XCTUnwrap(layers["model.layers.0.self_attn.q_proj"] as? QuantizedLinear)
        XCTAssertEqual(query.bits, 4)
        XCTAssertEqual(query.groupSize, 32)
        XCTAssertTrue(layers["model.embed_tokens"] is QuantizedEmbedding, "the tied head reads the packed embedding")
        let sourceLayers = Dictionary(source.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
        let sourceQuery = try XCTUnwrap(sourceLayers["model.layers.0.self_attn.q_proj"] as? QuantizedLinear)
        XCTAssertEqual(query.scales.dtype, sourceQuery.scales.dtype, "the scales keep the dtype they were stored in")
    }

    func testAShardedDenseReleaseRebuildsItsPackedLayers() throws {
        try requireMLXRuntime()
        MLXRandom.seed(22)
        let source = NFKMLXLanguage.makeNet(.tiny)
        try quantizeAsMLXLM(source)
        let directory = try scratchDirectory()
        try writeRelease(source.parameters().flattened(), sharded: true, to: directory)
        _ = try assertLoadsAsWritten(directory, source: source, geometry: .tiny)
    }

    // MARK: Mixtures

    // mlx_lm stacks a mixture's experts under `switch_mlp`; resident and paged loads both rebuild them
    // packed, with the router at its own 8-bit override.
    func testAStackedMixtureReleaseLoadsResidentAndPaged() throws {
        try requireMLXRuntime()
        MLXRandom.seed(23)
        let router = "model.layers.1.mlp.gate"
        let source = NFKMLXLanguage.makeNet(.tinyMixture)
        try quantizeAsMLXLM(source, overrides: [router])
        let released = source.parameters().flattened().map { name, value in
            (name.replacingOccurrences(of: ".mlp.experts.", with: ".mlp.switch_mlp."), value)
        }
        let directory = try scratchDirectory()
        try writeRelease(released, overrides: [router], sharded: true, to: directory)
        _ = try assertLoadsAsWritten(directory, source: source, geometry: .tinyMixture)
        let paged = try assertLoadsAsWritten(directory, source: source, geometry: .tinyMixture, residency: .paged)
        XCTAssertNotNil(paged.expertStore, "the paged load reads its packed experts from the release")
    }

    // An older mlx_lm Mixtral keeps one packed tensor set per expert under Mixtral's own spelling.
    func testAPerExpertMixtralReleaseStacksItsPackedParts() throws {
        try requireMLXRuntime()
        MLXRandom.seed(24)
        let source = NFKMLXLanguage.makeNet(.tinyMixture)
        try quantizeAsMLXLM(source)
        var released = [(String, MLXArray)]()
        let spelling = ["gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"]
        for (name, value) in source.parameters().flattened() {
            let parts = name.components(separatedBy: ".")
            guard name.contains(".mlp.experts.") else {
                released.append((name.replacingOccurrences(of: ".mlp.gate.", with: ".block_sparse_moe.gate."), value))
                continue
            }
            let layer = parts[2], projection = parts[5], part = parts[6]
            for expert in 0 ..< value.dim(0) {
                released.append(("model.layers.\(layer).block_sparse_moe.experts.\(expert).\(spelling[projection]!).\(part)",
                                 value[expert]))
            }
        }
        let directory = try scratchDirectory()
        try writeRelease(released, sharded: false, to: directory)
        _ = try assertLoadsAsWritten(directory, source: source, geometry: .tinyMixture)
    }

    // MARK: Released weights

    private var config: [String: String] { NFKMLXValidationConfig.environment }

    private func cosine(_ mine: [Float], _ reference: [Float]) -> Double {
        var dot = 0.0, left = 0.0, right = 0.0
        for (a, b) in zip(mine, reference) {
            dot += Double(a) * Double(b)
            left += Double(a) * Double(a)
            right += Double(b) * Double(b)
        }
        return dot / (left.squareRoot() * right.squareRoot())
    }

    /// A release mlx_lm published 4-bit, loaded through the release loader with its tokenizer and held to
    /// two references. At the stored precision it runs with mlx_lm's placement and computes mlx_lm's logits
    /// bit for bit (`mlx_lm_logits.py`), with mlx_lm's greedy tokens. The release's scales and norms are
    /// bfloat16, and on Qwen3-0.6B mlx_lm's run sits at cosine 0.99752 from float32 (0.95991 at position 0)
    /// where torch's bfloat16 run of the same weights sits at 0.99989, so the float32 reference alone
    /// cannot judge it. Widened to float32, it computes the
    /// dequantized network transformers runs (`run_reference.py mlx_quantized_release`), which is where a
    /// misread layer shows.
    private func assertReleaseMatchesItsReferences(release releaseKey: String, record recordKey: String,
                                                   runtimeRecord runtimeKey: String, seamRecord seamKey: String? = nil,
                                                   label: String) throws {
        try requireMLXRuntime()
        guard let directory = config[releaseKey], let path = config[recordKey], let runtimePath = config[runtimeKey] else {
            throw XCTSkip("set \(releaseKey), \(recordKey) (run_reference.py mlx_quantized_release), and \(runtimeKey) "
                          + "(mlx_lm_logits.py)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let runtime = try MLX.loadArrays(url: URL(fileURLWithPath: runtimePath))
        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self).map(Int.init)
        XCTAssertEqual(try XCTUnwrap(runtime["tokens"]).asArray(Int32.self).map(Int.init), tokens,
                       "both references read the same prompt")
        let reference = try XCTUnwrap(record["output"]).reshaped([-1]).asArray(Float.self)
        let half = try XCTUnwrap(record["bf16.output"]).reshaped([-1]).asArray(Float.self)
        let runtimeLogits = try XCTUnwrap(runtime["output"]).reshaped([-1]).asArray(Float.self)
        let runtimeContinuation = try XCTUnwrap(runtime["continuation"]).asArray(Int32.self).map(Int.init)

        let (net, tokenizer) = try NFKMLXLanguage.loadedReleaseWithTokenizer(at: URL(fileURLWithPath: directory))
        let encoded = tokenizer.encode("The capital of France is").map(\.intValue)
        XCTAssertEqual(Array(tokens.suffix(encoded.count)), encoded, "the release tokenizer reads the prompt as the reference's")
        XCTAssertLessThanOrEqual(tokens.count - encoded.count, 1, "the reference adds at most its begin marker")

        let input = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
        let logits = net(input)[0].asType(.float32)
        let mine = logits.reshaped([-1]).asArray(Float.self)
        let againstRuntime = cosine(mine, runtimeLogits)
        let orders = runtime.keys.filter { $0.hasPrefix("orders.") && $0.hasSuffix(".output") }.sorted()
        XCTAssertFalse(orders.isEmpty, "the mlx_lm record carries its runs in other summation orders")
        let orderCosines = try orders.map { key in
            (key, cosine(try XCTUnwrap(runtime[key]).reshaped([-1]).asArray(Float.self), runtimeLogits))
        }
        let vocabulary = logits.dim(1)
        let positions = (0 ..< tokens.count).map { position in
            String(format: "%.6f", cosine(Array(mine[(position * vocabulary) ..< ((position + 1) * vocabulary)]),
                                          Array(runtimeLogits[(position * vocabulary) ..< ((position + 1) * vocabulary)])))
        }
        let floors = orderCosines.map { "\($0.0.dropFirst("orders.".count).dropLast(".output".count)) \($0.1)" }
        print("VALIDATION PARITY \(label): logit cosine \(againstRuntime) against mlx_lm at the stored precision "
              + "(per position \(positions.joined(separator: " "))), whose own runs in other orders reach "
              + "\(floors.joined(separator: ", ")); mlx_lm reaches \(cosine(runtimeLogits, reference)) against the "
              + "dequantized float32 reference, this release \(cosine(mine, reference)), and transformers' bf16 run "
              + "\(cosine(half, reference))")
        for position in 0 ..< tokens.count {
            let row = Array(mine[(position * vocabulary) ..< ((position + 1) * vocabulary)])
            let theirs = Array(runtimeLogits[(position * vocabulary) ..< ((position + 1) * vocabulary)])
            XCTAssertEqual(row.indices.max { row[$0] < row[$1] }, theirs.indices.max { theirs[$0] < theirs[$1] },
                           "position \(position) predicts mlx_lm's token")
        }
        var greedy = NFKMLXGenerationOptions()
        greedy.temperature = 0
        greedy.maxTokens = 8
        XCTAssertEqual(net.generate(prompt: tokens, options: greedy), Array(runtimeContinuation.prefix(8)),
                       "and generates mlx_lm's greedy tokens")

        // A scaled rotary's table comes from a GPU pow the release cannot reproduce (`rotaryTableSteps`); on
        // mlx_lm's own table, read from the seam record, the logits are mlx_lm's bit for bit.
        var exact = mine
        if let seamKey, let seamPath = config[seamKey],
           let table = try MLX.loadArrays(url: URL(fileURLWithPath: seamPath))["rope.freqs"] {
            Self.useRotaryTable(table, in: net)
            exact = net(input)[0].asType(.float32).reshaped([-1]).asArray(Float.self)
        }
        XCTAssertEqual(zip(exact, runtimeLogits).filter { $0 != $1 }.count, 0,
                       "the release computes mlx_lm's logits bit for bit")

        // Every float parameter widened to float32 (the scales, biases, and norms mlx_lm stored at
        // bfloat16) leaves the packed words as read: a loading mistake survives the widening.
        let widened = net.parameters().flattened().map { name, value in
            (name, value.dtype == .bfloat16 || value.dtype == .float16 ? value.asType(.float32) : value)
        }
        net.update(parameters: ModuleParameters.unflattened(widened))
        let wide = cosine(net(input)[0].asType(.float32).reshaped([-1]).asArray(Float.self), reference)
        print("VALIDATION PARITY \(label)-float32: logit cosine \(wide) with the stored scales, biases, and norms "
              + "widened to float32")
        XCTAssertGreaterThan(wide, 0.999999, "every packed layer reads as the dequantized network")
    }

    func testMLXCommunityQwen3FourBitMatchesItsReferences() throws {
        try assertReleaseMatchesItsReferences(release: "IK_VAL_MLX_QWEN3_06B_4BIT", record: "IK_PARITY_MLX_QWEN3_06B_4BIT",
                                              runtimeRecord: "IK_PARITY_MLX_QWEN3_06B_4BIT_MLXLM",
                                              label: "mlx-qwen3-0.6b-4bit")
    }

    func testMLXCommunityLlama32FourBitMatchesItsReferences() throws {
        try assertReleaseMatchesItsReferences(release: "IK_VAL_MLX_LLAMA32_1B_4BIT", record: "IK_PARITY_MLX_LLAMA32_1B_4BIT",
                                              runtimeRecord: "IK_PARITY_MLX_LLAMA32_1B_4BIT_MLXLM",
                                              seamRecord: "IK_PARITY_MLX_LLAMA32_1B_4BIT_SEAMS",
                                              label: "mlx-llama-3.2-1b-4bit")
    }

    /// How far a scaled rotary's period table may sit from mlx_lm's, in float32 steps (USER 2026-10-21). mlx_lm
    /// evaluates `base ** (2i/d)` with the GPU pow its wheel precompiles; mlx-swift compiles its pow at run
    /// time as a float32 `exp2(e · log2(base))`, and no arithmetic here reproduces the wheel's. Measured on
    /// Llama 3.2 1B's llama3 table: 22 of 32 periods apart, at most 14 steps, the smoothed band the widest.
    static let rotaryTableSteps: Int64 = 14

    /// The largest distance between two float32 arrays, in representable steps.
    static func floatSteps(_ a: [Float], _ b: [Float]) -> Int64 {
        zip(a, b).map { abs(Int64($0.bitPattern) - Int64($1.bitPattern)) }.max() ?? 0
    }

    /// Puts `table` in every layer's rotary, as the periods `MLXFast.RoPE` reads.
    static func useRotaryTable(_ table: MLXArray, in net: NFKMLXLanguageNet) {
        for layer in net.model.layers {
            layer.attention.rope.periods = table
        }
    }

    /// Each seam of the release's forward against mlx_lm's (`mlx_lm_seams.py`) at the stored precision, from
    /// mlx_lm's own input to that seam, so a seam that rounds apart shows by itself. Every seam and the
    /// chained forward's layer outputs are bit-equal; each layer's line reports any element that differs.
    private func assertReleaseSeams(release releaseKey: String, record recordKey: String, label: String) throws {
        try requireMLXRuntime()
        guard let directory = config[releaseKey], let path = config[recordKey] else {
            throw XCTSkip("set \(releaseKey) and \(recordKey) (mlx_lm_seams.py)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        func seam(_ key: String) throws -> MLXArray { try XCTUnwrap(record[key], key) }
        let net = try NFKMLXLanguage.loadedRelease(at: URL(fileURLWithPath: directory), precision: .checkpoint).0
        XCTAssertEqual(net.configuration.halfPrecisionPlacement, .mlx, "an mlx_lm release runs with mlx_lm's placement")
        let tokens = try seam("tokens").reshaped([1, -1])
        let length = tokens.dim(1)
        let mask = NFKMLXLanguageNet.causalMask(length, offset: 0)

        func compared(_ name: String, _ mine: MLXArray, _ theirs: MLXArray) -> String {
            XCTAssertEqual(mine.dtype, theirs.dtype, "\(name) is computed in mlx_lm's type")
            let differing = (mine .!= theirs).asType(.int32).sum().item(Int.self)
            XCTAssertEqual(differing, 0, "\(label) \(name) rounds as mlx_lm's")
            guard differing > 0 else { return "\(name) exact" }
            let a = mine.asType(.float32).reshaped([-1]).asArray(Float.self)
            let b = theirs.asType(.float32).reshaped([-1]).asArray(Float.self)
            return "\(name) \(differing)/\(mine.size) \(String(format: "%.9f", cosine(a, b)))"
        }

        print("SEAMS \(label) embedding: \(compared("embedding", net.model.embedTokens(tokens), try seam("embedding")))")
        if let theirs = record["rope.freqs"] {
            let ours = try XCTUnwrap(net.model.layers[0].attention.rope.periods, "the scaled rotary carries a period table")
            let a = ours.asArray(Float.self), b = theirs.asArray(Float.self)
            let apart = zip(a, b).enumerated().filter { $0.element.0 != $0.element.1 }
                .map { "\($0.offset): \($0.element.0) vs \($0.element.1)" }
            let rotary = net.model.layers[0].attention.rope
            let scaling = try XCTUnwrap(net.configuration.ropeScaling)
            let recomputed = NFKLMRotary.mlxLlama3Periods(dimensions: rotary.dimensions, base: rotary.base, scaling: scaling)
            let recomputedApart = zip(recomputed.asArray(Float.self), b).filter { $0 != $1 }.count
            print("SEAMS \(label) rope path: placement \(rotary.placement), kind \(scaling.kind), base \(rotary.base), "
                  + "dimensions \(rotary.dimensions), factor \(scaling.factor), low \(scaling.lowFrequencyFactor), "
                  + "high \(scaling.highFrequencyFactor), original \(scaling.originalMaxPositionEmbeddings); "
                  + "mlx_lm's table recomputed here differs at \(recomputedApart) of \(b.count)")
            print("SEAMS \(label) rope table: \(apart.isEmpty ? "exact" : apart.joined(separator: "; "))")
            let steps = Self.floatSteps(a, b)
            print("SEAMS \(label) rope table: at most \(steps) float32 steps from mlx_lm's")
            XCTAssertLessThanOrEqual(steps, Self.rotaryTableSteps, "\(label)'s rotary periods are mlx_lm's, to its pow")
            // Every seam after the table runs on mlx_lm's own table, where it is bit-exact.
            Self.useRotaryTable(theirs, in: net)
        }
        var chained = net.model.embedTokens(tokens)
        for (index, block) in net.model.layers.enumerated() {
            func ref(_ name: String) throws -> MLXArray { try seam("layers.\(index).\(name)") }
            let attention = block.attention
            let heads = attention.heads, keyValueHeads = attention.keyValueHeads, width = attention.headDimensions
            var rows = [String]()
            rows.append(compared("input_norm", block.attentionNorm(try ref("input")), try ref("input_norm")))
            for (name, projection) in [("q_proj", attention.queryProjection), ("k_proj", attention.keyProjection),
                                       ("v_proj", attention.valueProjection)] {
                rows.append(compared(name, projection(try ref("input_norm")), try ref(name)))
            }
            var queries = try ref("q_proj").reshaped([1, length, heads, width])
            var keys = try ref("k_proj").reshaped([1, length, keyValueHeads, width])
            if let queryNorm = attention.queryNorm, let keyNorm = attention.keyNorm {
                rows.append(compared("q_norm", queryNorm(queries), try ref("q_norm")))
                rows.append(compared("k_norm", keyNorm(keys), try ref("k_norm")))
                (queries, keys) = (try ref("q_norm"), try ref("k_norm"))
            }
            rows.append(compared("q_rope", attention.rope(queries.transposed(0, 2, 1, 3), offset: 0), try ref("q_rope")))
            rows.append(compared("k_rope", attention.rope(keys.transposed(0, 2, 1, 3), offset: 0), try ref("k_rope")))
            let values = try ref("v_proj").reshaped([1, length, keyValueHeads, width]).transposed(0, 2, 1, 3)
            let attended = attention.attended(queries: try ref("q_rope"), keys: try ref("k_rope"), values: values,
                                              mask: mask)
            rows.append(compared("attention", attended.transposed(0, 2, 1, 3).reshaped([1, length, heads * width]),
                                 try ref("attention")))
            rows.append(compared("o_proj", attention.outputProjection(try ref("attention")), try ref("o_proj")))
            rows.append(compared("attended", try ref("input") + (try ref("o_proj")), try ref("attended")))
            rows.append(compared("post_norm", block.feedForwardNorm(try ref("attended")), try ref("post_norm")))
            let feedForward = try XCTUnwrap(block.feedForward as? NFKLMFeedForward)
            rows.append(compared("gate", feedForward.gate(try ref("post_norm")), try ref("gate")))
            rows.append(compared("up", feedForward.up(try ref("post_norm")), try ref("up")))
            rows.append(compared("swiglu", feedForward.activated(gate: try ref("gate"), up: try ref("up")), try ref("swiglu")))
            rows.append(compared("down", feedForward.down(try ref("swiglu")), try ref("down")))
            rows.append(compared("output", try ref("attended") + (try ref("down")), try ref("output")))
            chained = block(chained, mask: mask.asType(chained.dtype), cache: nil, layer: index)
            rows.append(compared("chained", chained, try ref("output")))
            print("SEAMS \(label) layer \(index): \(rows.joined(separator: ", "))")
            NFKMLXGPU.clearCache()
        }
        let lastOutput = try seam("layers.\(net.model.layers.count - 1).output")
        print("SEAMS \(label) head: \(compared("norm", net.model.norm(lastOutput), try seam("norm"))), "
              + compared("logits", net.logits(fromHidden: try seam("norm")).asType(.float32), try seam("output").reshaped([1, length, -1])))
    }

    func testTheQwen3ReleaseSeamsAgainstMLXLM() throws {
        try assertReleaseSeams(release: "IK_VAL_MLX_QWEN3_06B_4BIT", record: "IK_PARITY_MLX_QWEN3_06B_4BIT_SEAMS",
                               label: "mlx-qwen3-0.6b-4bit")
    }

    func testTheLlama32ReleaseSeamsAgainstMLXLM() throws {
        try assertReleaseSeams(release: "IK_VAL_MLX_LLAMA32_1B_4BIT", record: "IK_PARITY_MLX_LLAMA32_1B_4BIT_SEAMS",
                               label: "mlx-llama-3.2-1b-4bit")
    }

    // MARK: What a config states

    func testTheConfigNamesTheQuantizationsThisPackageReads() throws {
        XCTAssertNil(try NFKMLXQuantization.releaseQuantization(configJSON: ["model_type": "qwen3"]))
        XCTAssertEqual(try NFKMLXQuantization.releaseQuantization(configJSON: ["quantization": ["group_size": 64, "bits": 4]]),
                       NFKMLXWeights.Quantization(bits: 4, groupSize: 64, mode: .affine))
        XCTAssertEqual(try NFKMLXQuantization.releaseQuantization(
                           configJSON: ["quantization_config": ["group_size": 32, "bits": 4, "mode": "mxfp4"]]),
                       NFKMLXWeights.Quantization(bits: 4, groupSize: 32, mode: .mxfp4))
        XCTAssertEqual(try NFKMLXQuantization.releaseQuantization(
                           configJSON: ["quantization_config": ["quant_method": "mxfp4"]]),
                       NFKMLXLanguage.mxfp4Quantization)
        XCTAssertEqual(try NFKMLXQuantization.releaseQuantization(
                           configJSON: ["text_config": ["quantization": ["group_size": 64, "bits": 8]]]),
                       NFKMLXWeights.Quantization(bits: 8, groupSize: 64, mode: .affine),
                       "a multimodal release states it for its decoder")
        for method in ["gptq", "awq", "bitsandbytes", "fp8", "compressed-tensors"] {
            XCTAssertThrowsError(try NFKMLXQuantization.releaseQuantization(
                configJSON: ["quantization_config": ["quant_method": method, "bits": 4]]), method)
        }
        XCTAssertThrowsError(try NFKMLXQuantization.releaseQuantization(
            configJSON: ["quantization": ["group_size": 64, "bits": 4, "mode": "int3"]]))
        XCTAssertThrowsError(try NFKMLXQuantization.releaseQuantization(configJSON: ["quantization": ["bits": 4]]))
        XCTAssertTrue(NFKMLXLanguage.isMLXQuantization(["quantization": ["group_size": 64, "bits": 4]]))
        XCTAssertFalse(NFKMLXLanguage.isMLXQuantization(["quantization_config": ["quant_method": "mxfp4"]]))
    }

    func testALayerGeometryIsReadFromItsTensors() throws {
        func entry(_ dtype: String, _ shape: [Int]) -> NFKMLXSafetensorsEntry {
            NFKMLXSafetensorsEntry(dtype: dtype, shape: shape, start: 0, end: 0)
        }
        XCTAssertEqual(try NFKMLXQuantization.releaseLayerQuantization(
                           of: "q", packed: entry("U32", [512, 256]), scales: entry("BF16", [512, 32]), hasBiases: true,
                           inputSize: 2048),
                       NFKMLXWeights.Quantization(bits: 4, groupSize: 64, mode: .affine))
        XCTAssertEqual(try NFKMLXQuantization.releaseLayerQuantization(
                           of: "experts", packed: entry("U32", [32, 2880, 360]), scales: entry("U8", [32, 2880, 90]),
                           hasBiases: false, inputSize: 2880),
                       NFKMLXWeights.Quantization(bits: 4, groupSize: 32, mode: .mxfp4))
        XCTAssertThrowsError(try NFKMLXQuantization.releaseLayerQuantization(
            of: "odd", packed: entry("U32", [8, 7]), scales: entry("F16", [8, 3]), hasBiases: true, inputSize: 64))
        XCTAssertThrowsError(try NFKMLXQuantization.releaseLayerQuantization(
            of: "no mode", packed: entry("U32", [8, 8]), scales: entry("F16", [8, 2]), hasBiases: false, inputSize: 64))
    }
}
