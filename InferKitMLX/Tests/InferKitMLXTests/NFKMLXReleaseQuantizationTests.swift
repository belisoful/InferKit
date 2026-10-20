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

    /// A release mlx_lm published 4-bit, loaded through the release loader with its tokenizer and held
    /// to transformers on the same weights dequantized (Tools/reference-parity `mlx_quantized_release`).
    private func assertReleaseMatchesItsDequantizedReference(release releaseKey: String, record recordKey: String,
                                                             label: String) throws {
        try requireMLXRuntime()
        guard let directory = config[releaseKey], let path = config[recordKey] else {
            throw XCTSkip("set \(releaseKey) and \(recordKey) (run_reference.py mlx_quantized_release)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self).map(Int.init)
        let reference = try XCTUnwrap(record["output"]).reshaped([-1]).asArray(Float.self)
        let half = try XCTUnwrap(record["bf16.output"]).reshaped([-1]).asArray(Float.self)
        let continuation = try XCTUnwrap(record["continuation"]).asArray(Int32.self).map(Int.init)

        let (net, tokenizer) = try NFKMLXLanguage.loadedReleaseWithTokenizer(at: URL(fileURLWithPath: directory))
        let encoded = tokenizer.encode("The capital of France is").map(\.intValue)
        XCTAssertEqual(Array(tokens.suffix(encoded.count)), encoded, "the release tokenizer reads the prompt as the reference's")
        XCTAssertLessThanOrEqual(tokens.count - encoded.count, 1, "the reference adds at most its begin marker")

        let logits = net(MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count]))[0].asType(.float32)
        let mine = logits.reshaped([-1]).asArray(Float.self)
        let similarity = cosine(mine, reference)
        let floor = cosine(half, reference)
        print("VALIDATION PARITY \(label): logit cosine \(similarity) against the dequantized float32 reference "
              + "(its own bf16 run reaches \(floor))")
        XCTAssertGreaterThan(similarity, 0.999, "the packed layers compute the dequantized network")

        let vocabulary = logits.dim(1)
        let lastRow = Array(mine.suffix(vocabulary)), lastReference = Array(reference.suffix(vocabulary))
        XCTAssertEqual(lastRow.indices.max { lastRow[$0] < lastRow[$1] },
                       lastReference.indices.max { lastReference[$0] < lastReference[$1] },
                       "the next token is the reference's")
        var greedy = NFKMLXGenerationOptions()
        greedy.temperature = 0
        greedy.maxTokens = 4
        XCTAssertEqual(net.generate(prompt: tokens, options: greedy), Array(continuation.prefix(4)),
                       "and the first greedy tokens are")
    }

    func testMLXCommunityQwen3FourBitMatchesItsDequantizedReference() throws {
        try assertReleaseMatchesItsDequantizedReference(release: "IK_VAL_MLX_QWEN3_06B_4BIT",
                                                        record: "IK_PARITY_MLX_QWEN3_06B_4BIT",
                                                        label: "mlx-qwen3-0.6b-4bit")
    }

    func testMLXCommunityLlama32FourBitMatchesItsDequantizedReference() throws {
        try assertReleaseMatchesItsDequantizedReference(release: "IK_VAL_MLX_LLAMA32_1B_4BIT",
                                                        record: "IK_PARITY_MLX_LLAMA32_1B_4BIT",
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
