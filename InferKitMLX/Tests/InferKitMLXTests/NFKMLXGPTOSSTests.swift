//
//  NFKMLXGPTOSSTests.swift
//  InferKitMLXTests
//
//  gpt-oss: the MXFP4 expert format the release is stored in, held to the Open Compute layout MLX's
//  `mxfp4` mode must agree with before the released blocks can be fed to it as they are.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXGPTOSSTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The sixteen e2m1 values, in nibble order: sign bit high, then two exponent bits, one mantissa bit.
    private static let e2m1: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]

    private var config: [String: String] { NFKMLXValidationConfig.environment }

    private func tinyGeometry() throws -> NFKMLXLanguageConfiguration {
        var geometry = NFKMLXLanguageConfiguration(
            hiddenSize: 64, layerCount: 2, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            intermediateSize: 32, vocabularySize: 128, ropeTheta: 150_000, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: false, normalizesQueryAndKey: false, attentionBias: true)
        geometry.expertCount = 4
        geometry.activeExpertCount = 2
        geometry.expertIntermediateSize = 32
        geometry.slidingWindows = [4, nil]
        geometry.attentionSinks = true
        geometry.outputProjectionBias = true
        geometry.routerBias = true
        geometry.clampedSwiGLU = NFKMLXClampedSwiGLU()
        return geometry
    }

    // A module whose fused experts hold MXFP4-packed words saves with the mode in its metadata and
    // reloads onto matching packed structure through the ordinary loader, reproducing its logits.
    func testAnMXFP4ExpertModuleRoundTripsThroughTheCheckpoint() throws {
        try requireMLXRuntime()
        MLXRandom.seed(7)
        let net = NFKMLXLanguage.makeNet(try tinyGeometry())
        for block in net.model.layers {
            let fused = try XCTUnwrap((block.feedForward as! NFKLMMixtureFeedForward).experts as? NFKLMFusedSwitchGLU)
            for (name, layer) in [("gate_up_proj", fused.gateUp), ("down_proj", fused.down)] {
                let (packed, scales, biases) = MLX.quantized(layer.weight, groupSize: 32, bits: 4, mode: .mxfp4)
                let quantized = NFKLMQuantizedSwitchLinear(packed: packed, scales: scales, biases: biases,
                                                           groupSize: 32, bits: 4, mode: .mxfp4)
                try fused.update(modules: ModuleChildren.unflattened([(name, quantized)]), verify: .noUnusedKeys)
            }
        }
        let tokens = MLXArray([3, 17, 42, 8, 91].map { Int32($0) }).reshaped([1, 5])
        let expected = net(tokens)
        eval(expected)

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gpt-oss-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        XCTAssertEqual(checkpoint.quantization, .init(bits: 4, groupSize: 32, mode: .mxfp4), "the mode is recorded")

        let loaded = NFKMLXLanguage.makeNet(try tinyGeometry())
        try NFKMLXLanguage.loadWeights(into: loaded, from: url)
        let reloaded = try XCTUnwrap((loaded.model.layers[0].feedForward as! NFKLMMixtureFeedForward).experts as? NFKLMFusedSwitchGLU)
        XCTAssertEqual((reloaded.gateUp as? NFKLMQuantizedSwitchLinear)?.mode, .mxfp4)
        let actual = loaded(tokens)
        eval(actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    // MARK: The released gpt-oss-20b

    /// The `tokenizers` library's ids for these strings over the release's tokenizer.json.
    private static let tokenizerCases: [(String, [Int])] = [
        ("The capital of France is", [976, 9029, 328, 10128, 382]),
        ("Hello world! It's 2026-09-06, and I'd say 1234567 apples cost $12.50.",
         [13225, 2375, 0, 7744, 220, 1323, 21, 12, 3114, 12, 3218, 11, 326, 18754, 2891, 220, 7633, 19354, 22, 57814, 3097, 548, 899, 13, 1434, 13]),
        ("  leading spaces\n\nand newlines\t tabs", [220, 8117, 18608, 279, 427, 620, 10105, 197, 38191]),
        ("naïve café — résumé 日本語テキスト 🚀", [1503, 9954, 737, 30469, 2733, 140184, 17428, 40909, 16056, 18368, 38236, 169883, 222]),
        ("<|start|>user<|message|>Hi<|end|>", [200006, 1428, 200008, 12194, 200007]),
        ("def f(x):\n    return x**2  # square\n", [1314, 285, 4061, 1883, 271, 622, 1215, 410, 17, 220, 1069, 13749, 198]),
        ("CamelCaseWord ALLCAPS lowercase MiXeD 's 'RE 'll",
         [137910, 6187, 12929, 19465, 56928, 50, 90395, 13236, 148218, 35, 461, 82, 461, 1099, 461, 680]),
    ]

    // The release ships only tokenizer.json (o200k_harmony): the extracted vocabulary and merges
    // under the o200k pre-tokenization reproduce the `tokenizers` library's ids token for token,
    // digit runs of three, case-pattern word splits, contractions, and the harmony markers included.
    func testTheReleaseTokenizerAgreesWithTokenizers() throws {
        guard let directory = config["IK_VAL_GPT_OSS"] else { throw XCTSkip("set IK_VAL_GPT_OSS") }
        let tokenizer = try XCTUnwrap(NFKMLXLanguage.releaseTokenizer(inDirectory: URL(fileURLWithPath: directory)))
        XCTAssertEqual(NFKMLXLanguage.pretokenizationName(inDirectory: URL(fileURLWithPath: directory)), "o200k")
        XCTAssertEqual(tokenizer.eosTokenId, 200002, "<|return|> ends a reply")
        for (text, expected) in Self.tokenizerCases {
            XCTAssertEqual(tokenizer.encode(text).map(\.intValue), expected, text)
        }
    }

    /// Every tensor's shape from the release's own shard headers, read locally.
    private func releasedShapes(inDirectory directory: URL) throws -> [String: [Int]] {
        var shapes = [String: [Int]]()
        for file in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            let lengthData = try XCTUnwrap(try handle.read(upToCount: 8))
            let length = lengthData.withUnsafeBytes { $0.load(as: UInt64.self) }
            let header = try XCTUnwrap(try handle.read(upToCount: Int(length)))
            let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: header) as? [String: Any])
            for (name, entry) in json where name != "__metadata__" {
                if let shape = (entry as? [String: Any])?["shape"] as? [Int] { shapes[name] = shape }
            }
        }
        return shapes
    }

    // Every parameter the module builds for gpt-oss-20b exists in the release at the built shape — a
    // fused expert projection `[experts, out, in]` as `_blocks` `[experts, out, in / 32, 16]` and
    // `_scales` `[experts, out, in / 32]` — and every released tensor is one the module consumes.
    func testEveryParameterMatchesTheReleasedGPTOSSCheckpoint() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_VAL_GPT_OSS"] else { throw XCTSkip("set IK_VAL_GPT_OSS") }
        let release = URL(fileURLWithPath: directory)
        let released = try releasedShapes(inDirectory: release)
        let geometry = try NFKMLXLanguage.configuration(fromHuggingFace: release.appendingPathComponent("config.json"))
        XCTAssertEqual(geometry.expertCount, 32)
        XCTAssertEqual(geometry.activeExpertCount, 4)
        XCTAssertEqual(geometry.slidingWindows.count, 24)
        XCTAssertEqual(geometry.slidingWindows[0], 128)
        XCTAssertNil(geometry.slidingWindows[1])
        XCTAssertTrue(geometry.attentionSinks && geometry.routerBias && geometry.outputProjectionBias)
        XCTAssertEqual(geometry.ropeScaling?.truncatesCorrectionRange, false)
        let net = NFKMLXLanguage.makeNet(geometry)

        var consumed = Set<String>(), missing = [String](), mismatched = [String]()
        for (name, value) in net.parameters().flattened() {
            if name.contains(".mlp.experts."), name.hasSuffix("_proj.weight") {
                let base = String(name.dropLast(".weight".count))
                let experts = value.dim(0), out = value.dim(1), input = value.dim(2)
                for (suffix, shape) in [("_blocks", [experts, out, input / 32, 16]), ("_scales", [experts, out, input / 32])] {
                    guard let stored = released[base + suffix] else { missing.append(base + suffix); continue }
                    consumed.insert(base + suffix)
                    if stored != shape { mismatched.append("\(base + suffix): built \(shape), released \(stored)") }
                }
                continue
            }
            let key = name.replacingOccurrences(of: ".mlp.gate.", with: ".mlp.router.")
            guard let stored = released[key] else { missing.append(key); continue }
            consumed.insert(key)
            if stored != value.shape { mismatched.append("\(key): built \(value.shape), released \(stored)") }
        }
        let unaccounted = released.keys.filter { !consumed.contains($0) }.sorted()
        print("VALIDATION structure gpt-oss-20b: \(consumed.count) released tensors consumed, "
              + "\(missing.count) missing, \(mismatched.count) mismatched, \(unaccounted.count) unaccounted")
        XCTAssertTrue(mismatched.isEmpty, mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(missing.isEmpty, missing.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(unaccounted.isEmpty, unaccounted.prefix(8).joined(separator: "\n"))
        XCTAssertEqual(consumed.count, released.count)
    }

    // The released 20B, its experts kept MXFP4-packed and everything else at its bf16, generates
    // through the ordinary backend: a raw completion of "The capital of France is" names Paris.
    func testGPTOSSGeneratesOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_VAL_GPT_OSS"] else { throw XCTSkip("set IK_VAL_GPT_OSS") }
        let backend = try NFKMLXLanguage.backend(directoryURL: URL(fileURLWithPath: directory))
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "The capital of France is"],
                                          parameters: [NFKParameterMaxTokens: 12, NFKParameterTemperature: 0])
        let started = Date()
        let text = try XCTUnwrap(backend.runInference(for: request).text)
        print("VALIDATION gpt-oss-20b: \(text.debugDescription) in \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
        XCTAssertTrue(text.contains("Paris"), text)
        NFKMLXGPU.clearCache()
    }

    // MLX packs an mxfp4 row as uint32 words of eight nibbles, element i in bits 4·(i mod 8) of word
    // i / 8 (little-endian, so a byte holds the earlier value in its low nibble — the Open Compute
    // layout the release stores its `_blocks` in), with one e8m0 scale byte (exponent biased by 127)
    // per group of 32. Decoding the packed words by that rule reproduces MLX's own dequantization,
    // which is what lets the released blocks be viewed as uint32 and used unchanged.
    func testMXFP4PackingIsTheOpenComputeLayout() throws {
        try requireMLXRuntime()
        MLXRandom.seed(3)
        let weight = MLXRandom.normal([8, 128]) * 2
        let (packed, scales, biases) = MLX.quantized(weight, groupSize: 32, bits: 4, mode: .mxfp4)
        eval(packed, scales)
        XCTAssertNil(biases, "mxfp4 carries no bias")
        XCTAssertEqual(packed.dtype, .uint32)
        XCTAssertEqual(packed.shape, [8, 16], "eight nibbles per word")
        XCTAssertEqual(scales.dtype, .uint8, "an e8m0 exponent per group")
        XCTAssertEqual(scales.shape, [8, 4])

        let reference = MLX.dequantized(packed, scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4)
        eval(reference)
        let words = packed.asArray(UInt32.self), exponents = scales.asArray(UInt8.self)
        var decoded = [Float](repeating: 0, count: 8 * 128)
        for row in 0 ..< 8 {
            for column in 0 ..< 128 {
                let word = words[row * 16 + column / 8]
                let nibble = Int((word >> UInt32(4 * (column % 8))) & 0xF)
                let exponent = Int(exponents[row * 4 + column / 32]) - 127
                decoded[row * 128 + column] = Self.e2m1[nibble] * powf(2, Float(exponent))
            }
        }
        let theirs = reference.asArray(Float.self)
        XCTAssertEqual(zip(decoded, theirs).map { abs($0 - $1) }.max() ?? 1, 0, "the hand decode is MLX's decode")
        // The quantization itself is a real approximation of the weight, not garbage.
        let original = weight.asArray(Float.self)
        let squaredError: Float = zip(theirs, original).map { ($0 - $1) * ($0 - $1) }.reduce(0, +)
        let energy: Float = original.map { $0 * $0 }.reduce(0, +)
        XCTAssertLessThan(sqrt(squaredError / energy), 0.2,
                          "4-bit e2m1 with a shared exponent per 32 reconstructs the weight coarsely")
    }

    // The released blocks, viewed as uint32 words and handed to MLX's mxfp4 decode as they are,
    // reproduce transformers' own dequantization of the same bytes exactly. Measured on the first 64
    // rows of gpt-oss-20b's layer-0 `gate_up_proj` (run_reference.py gpt_oss_quant).
    func testTheReleasedMXFP4BlocksDecodeAsTransformersDecodesThem() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_GPT_OSS_MXFP4"] else {
            throw XCTSkip("set IK_PARITY_GPT_OSS_MXFP4 (run_reference.py gpt_oss_quant)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let bytes = try XCTUnwrap(arrays["mxfp4_bytes"])            // [rows, groups, 16] uint8
        let scaleBytes = try XCTUnwrap(arrays["mxfp4_scale_bytes"]) // [rows, groups] uint8
        let expected = try XCTUnwrap(arrays["output"])              // [rows, groups · 32]
        let rows = bytes.dim(0), groups = bytes.dim(1)
        XCTAssertEqual(bytes.dtype, .uint8)
        XCTAssertEqual(bytes.dim(2), 16, "sixteen bytes hold a group of 32 nibbles")

        let words = bytes.view(dtype: .uint32).reshaped([rows, groups * 4])
        let decoded = MLX.dequantized(words, scales: scaleBytes, biases: nil, groupSize: 32, bits: 4,
                                      mode: .mxfp4, dtype: .float32)
        eval(decoded)
        XCTAssertEqual(decoded.shape, expected.shape)
        let worst = abs(decoded - expected).max().item(Float.self)
        print("VALIDATION mxfp4 gpt-oss-20b layer-0 gate_up_proj rows 0..<\(rows): worst |difference| \(worst)")
        XCTAssertEqual(worst, 0, "MLX's mxfp4 decode of the released bytes is transformers' decode")
    }
}
