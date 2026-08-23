//
//  NFKMLXDeepSeekTests.swift
//  InferKitMLXTests
//
//  The DeepSeek V4 decoder. Nothing here loads weights: the released Flash model is hundreds of
//  gigabytes and cannot be instantiated at float precision on any single machine here, let alone run.
//
//  The verification is weaker than the hybrid decoder's, and deliberately says so. That checkpoint is
//  bf16, so a float module's shapes match it exactly. This one is QUANTIZED — attention in fp8, routed
//  experts 4-bit packed two to a byte — so the check has to derive what each float parameter looks like
//  stored. That derivation is an assumption, so it is asserted against the observed shapes rather than
//  trusted.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXDeepSeekTests: XCTestCase {

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func released(_ suffix: String = "") throws
        -> (NFKMLXDeepSeekConfiguration, [String: [String: Any]]) {
        guard let shapesPath = config["IK_SHAPES_DEEPSEEK_V4" + suffix],
              let configPath = config["IK_CONFIG_DEEPSEEK_V4" + suffix],
              let data = FileManager.default.contents(atPath: shapesPath),
              let observed = try JSONSerialization.jsonObject(with: data) as? [String: [String: Any]]
        else { throw XCTSkip("set IK_SHAPES_DEEPSEEK_V4\(suffix) and IK_CONFIG_DEEPSEEK_V4\(suffix)") }
        let geometry = try NFKMLXDeepSeek.configuration(
            fromHuggingFace: URL(fileURLWithPath: configPath))
        return (geometry, observed)
    }

    /// Small enough to build and run while keeping every structural ratio.
    private var small: NFKMLXDeepSeekConfiguration {
        NFKMLXDeepSeekConfiguration(hiddenSize: 64, layerCount: 4, vocabularySize: 128,
                                    headCount: 4, headDimensions: 16, ropeHeadDimensions: 4,
                                    queryLoRARank: 16, outputLoRARank: 16, outputGroups: 2,
                                    slidingWindow: 8, routedExpertCount: 8, activatedExpertCount: 2,
                                    expertIntermediateSize: 32, hashLayerCount: 1)
    }

    // MARK: The configuration

    func testTheReleasedConfigurationIsRead() throws {
        let (geometry, _) = try released()
        XCTAssertEqual(geometry.hiddenSize, 4096)
        XCTAssertEqual(geometry.layerCount, 43)
        XCTAssertEqual(geometry.headCount, 64)
        XCTAssertEqual(geometry.headDimensions, 512)
        XCTAssertEqual(geometry.routedExpertCount, 256)
        XCTAssertEqual(geometry.activatedExpertCount, 6)
        XCTAssertEqual(geometry.outputGroups, 8)
        XCTAssertEqual(geometry.hashLayerCount, 3)
        // The first two layers attend over the window alone; compression starts after them.
        XCTAssertEqual(geometry.compressRatios[0], 0)
        XCTAssertEqual(geometry.compressRatios[1], 0)
        XCTAssertGreaterThan(geometry.compressRatios[2], 0)
    }

    // The first layers route by a table indexed by token id, the rest by score with a learned bias.
    // Which a layer uses decides which parameters it carries, so a wrong boundary is a wrong model.
    func testRoutingChangesAfterTheHashLayers() throws {
        let (geometry, observed) = try released()
        for layer in 0 ..< geometry.hashLayerCount {
            XCTAssertNotNil(observed["layers.\(layer).ffn.gate.tid2eid"], "layer \(layer) routes by table")
            XCTAssertNil(observed["layers.\(layer).ffn.gate.bias"])
        }
        let scored = geometry.hashLayerCount
        XCTAssertNil(observed["layers.\(scored).ffn.gate.tid2eid"])
        XCTAssertNotNil(observed["layers.\(scored).ffn.gate.bias"], "layer \(scored) routes by score")
    }

    // MARK: Structure against the released checkpoint

    // The derivation of a quantized shape is an assumption; this is where it is tested. An attention
    // weight is fp8 and keeps its shape; a routed expert is 4-bit packed two to a byte, so its last
    // axis halves. If either is wrong the comparison below would be meaningless.
    func testTheQuantizedLayoutIsWhatTheReleaseUses() throws {
        let (_, observed) = try released()
        let attention = try XCTUnwrap(observed["layers.0.attn.wq_a.weight"])
        XCTAssertEqual(attention["dtype"] as? String, "F8_E4M3", "attention is fp8")
        XCTAssertEqual(attention["shape"] as? [Int], [1024, 4096], "and keeps its float shape")

        let expert = try XCTUnwrap(observed["layers.0.ffn.experts.0.w1.weight"])
        XCTAssertEqual(expert["dtype"] as? String, "I8", "a routed expert is packed into bytes")
        XCTAssertEqual(expert["shape"] as? [Int], [2048, 2048],
                       "two 4-bit values a byte, so 4096 columns are stored as 2048")

        let shared = try XCTUnwrap(observed["layers.0.ffn.shared_experts.w1.weight"])
        XCTAssertEqual(shared["dtype"] as? String, "F8_E4M3", "the shared expert is not 4-bit")
        XCTAssertEqual(shared["shape"] as? [Int], [2048, 4096])
    }

    // Every parameter the architecture declares, against the checkpoint's own headers. Enumerated
    // analytically: 43 layers of 257 experts cannot be instantiated at float precision.
    func testEveryDeclaredParameterMatchesTheReleasedCheckpoint() throws {
        try assertDeclaredParametersMatch(release: "", label: "deepseek-v4")
    }

    /// Flash's checks, run against another release of the same architecture.
    ///
    /// The Pro sizes everything up — 61 layers, hidden 7168, 384 experts, 128 heads — and adds a
    /// YaRN `rope_scaling`, which carries no parameters and so is invisible to a structural check.
    /// This is the Qwen3.8-27B treatment: the architecture is enumerated analytically and compared
    /// against captured shard headers, never instantiated.
    func testTheProReleaseMatchesStructurally() throws {
        try assertDeclaredParametersMatch(release: "_PRO", label: "deepseek-v4-pro")
    }

    private func assertDeclaredParametersMatch(release suffix: String, label: String) throws {
        let (geometry, observed) = try released(suffix)
        let expected = NFKMLXDeepSeek.expectedParameters(for: geometry)

        // Headers were captured for a subset of shards, so only those layers can be compared. The
        // meaningful coverage assertion is that EVERY expected parameter of a captured layer is
        // checked — an arbitrary count would pass while silently skipping a whole kind of parameter.
        let capturedLayers = Set(observed.keys.compactMap { key -> Int? in
            guard key.hasPrefix("layers.") else { return nil }
            return Int(key.split(separator: ".")[1])
        })
        XCTAssertFalse(capturedLayers.isEmpty, "no layer headers were captured")

        var checked = 0
        var mismatched = [String]()
        var uncomparable = [String]()
        for (name, floatShape) in expected {
            let layer = name.hasPrefix("layers.") ? Int(name.split(separator: ".")[1]) : nil
            let shouldCompare = layer.map(capturedLayers.contains) ?? (observed[name] != nil)
            guard shouldCompare else { continue }
            guard let shape = observed[name]?["shape"] as? [Int] else {
                uncomparable.append(name); continue
            }
            checked += 1
            let stored = NFKMLXDeepSeek.quantizedShape(of: name, float: floatShape)
            if stored != shape {
                mismatched.append("\(name): expected \(stored), released \(shape)")
            }
        }
        print("VALIDATION structure \(label): \(checked) parameters checked across "
              + "\(capturedLayers.count) captured layers, \(mismatched.count) mismatched")
        XCTAssertTrue(mismatched.isEmpty, "shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(uncomparable.isEmpty, "declared but absent from the release:\n"
                      + uncomparable.prefix(8).joined(separator: "\n"))
    }

    // THE CONVERSE, and the assertion that matters most: every tensor in the release is either a
    // parameter this port declares or falls in a named unimplemented group. Without this the check
    // only proves "what I declare exists", which is how an entire mechanism hid here — the
    // hyper-connection weights were in the checkpoint and absent from the port, and the one-directional
    // comparison reported zero problems.
    func testEveryReleasedTensorIsDeclaredOrNamed() throws {
        try assertEveryTensorAccounted(release: "", label: "deepseek-v4")
    }

    func testEveryProTensorIsDeclaredOrNamed() throws {
        try assertEveryTensorAccounted(release: "_PRO", label: "deepseek-v4-pro")
    }

    private func assertEveryTensorAccounted(release suffix: String, label: String) throws {
        guard let indexPath = config["IK_INDEX_DEEPSEEK_V4" + suffix],
              let data = FileManager.default.contents(atPath: indexPath),
              let index = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let map = index["weight_map"] as? [String: String],
              let configPath = config["IK_CONFIG_DEEPSEEK_V4" + suffix]
        else { throw XCTSkip("set IK_INDEX_DEEPSEEK_V4\(suffix) and IK_CONFIG_DEEPSEEK_V4\(suffix)") }

        let geometry = try NFKMLXDeepSeek.configuration(fromHuggingFace: URL(fileURLWithPath: configPath))
        let declared = Set(NFKMLXDeepSeek.expectedParameters(for: geometry).keys)

        // Deliberately outside this port, each named rather than merely absent.
        func unimplemented(_ key: String) -> Bool {
            key.hasPrefix("mtp.")                      // multi-token prediction / DSpark blocks
                || key.contains(".dspark")             // speculative-decoding heads
        }

        // A block scale is not a parameter, it is how the weight beside it is stored, and
        // `dequantized(_:shapes:)` consumes it. Accounted for by the weight it decodes.
        func decodesADeclaredWeight(_ key: String) -> Bool {
            key.hasSuffix(".scale")
                && declared.contains(key.replacingOccurrences(of: ".scale", with: ".weight"))
        }

        var unaccounted = [String]()
        for key in map.keys
        where !declared.contains(key) && !unimplemented(key) && !decodesADeclaredWeight(key) {
            unaccounted.append(key)
        }
        let scales = map.keys.filter(decodesADeclaredWeight).count
        let named = map.keys.filter(unimplemented).count
        print("VALIDATION structure \(label): \(scales) block scales decode a declared weight")
        print("VALIDATION structure \(label): \(declared.count) declared, \(named) named as "
              + "unimplemented, \(unaccounted.count) unaccounted")
        XCTAssertTrue(unaccounted.isEmpty, "released tensors this port neither declares nor names:\n"
                      + unaccounted.sorted().prefix(10).joined(separator: "\n"))
    }

    // MARK: It runs

    func testASmallConfigurationRunsEndToEnd() throws {
        try requireMLXRuntime()
        let net = NFKMLXDeepSeek.makeNet(small)
        let logits = net(MLXArray([Int32(1), 2, 3]).reshaped([1, 3]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, small.vocabularySize])
        XCTAssertTrue(logits.sum().item(Float.self).isFinite)
    }

    // Routing has to select exactly `activatedExpertCount` experts, and the weights it returns must
    // sum to the route scale — the reference renormalizes before scaling.
    func testTheGateSelectsAndNormalizes() throws {
        try requireMLXRuntime()
        let gate = NFKDeepSeekGate(small, layer: small.hashLayerCount)   // a scored layer
        let hidden = MLXArray.zeros([4, small.hiddenSize]) + 0.3
        let (weights, indices) = gate(hidden, tokens: nil)
        eval(weights, indices)
        XCTAssertEqual(indices.shape, [4, small.activatedExpertCount])
        let sums = weights.sum(axis: -1).asArray(Float.self)
        for total in sums {
            XCTAssertEqual(Double(total), Double(small.routeScale), accuracy: 1e-4,
                           "the selected weights renormalize, then scale")
        }
    }

    // MARK: The conventions the isolation harness taught us to pin

    // DeepSeek rotates ADJACENT channel pairs (the reference's `view_as_complex`), where the Qwen and
    // Gemma decoders here rotate halves. Nothing in a shape distinguishes them, and getting it wrong
    // is invisible until the numbers are compared — which cannot be done for this model, so the
    // convention is asserted directly instead.
    func testTheRotaryPairsAdjacentChannels() throws {
        try requireMLXRuntime()
        let rotary = NFKDeepSeekRotary(width: 4, theta: 10_000)
        // At position 1 the first pair turns by 1 radian-equivalent and the second by a smaller angle.
        let x = MLXArray([Float(1), 0, 1, 0]).reshaped([1, 1, 1, 4])
        let turned = rotary(x, offset: 1)
        eval(turned)
        let v = turned.reshaped([-1]).asArray(Float.self)

        // An interleaved rotation maps (1, 0) to (cos, sin) within EACH pair. A rotate-half one would
        // instead mix channel 0 with channel 2, leaving channel 1 untouched.
        XCTAssertEqual(Double(v[0]), Double(cos(Float(1))), accuracy: 1e-5)
        XCTAssertEqual(Double(v[1]), Double(sin(Float(1))), accuracy: 1e-5)
        XCTAssertNotEqual(Double(v[1]), 0, accuracy: 1e-3, "channel 1 is the partner of channel 0")
    }

    // The output's rotary component is undone on the way out, because the values share their latent
    // with the keys. An inverse that is not exactly the conjugate would leave a residual rotation.
    func testTheInverseRotationUndoesTheForwardOne() throws {
        try requireMLXRuntime()
        let rotary = NFKDeepSeekRotary(width: 8, theta: 10_000)
        let x = MLXArray((0 ..< 8).map { Float($0 + 1) }).reshaped([1, 1, 1, 8])
        let round = rotary(rotary(x, offset: 3), offset: 3, inverse: true)
        eval(round)
        let worst = (round - x).abs().max().item(Float.self)
        XCTAssertEqual(Double(worst), 0, accuracy: 1e-4, "rotating then de-rotating is the identity")
    }

    // The learned per-head sink drains probability mass without contributing a value, so raising it
    // must shrink the attended output rather than reorder it.
    //
    // The sink is mutated through `update(parameters:)`: assigning to a `@ParameterInfo` directly
    // aborts the process with "please call update() on the array rather than setting it", the same
    // family of MLX trap as giving a `@ModuleInfo` a numeric key.
    func testTheAttentionSinkDrainsMass() throws {
        try requireMLXRuntime()
        let configuration = small
        let net = NFKMLXDeepSeek.makeNet(configuration)
        let attention = net.layers[0].attention
        let x = MLXArray.zeros([1, 4, configuration.hiddenSize]) + 0.2

        let quiet = attention(x, mask: nil)
        eval(quiet)
        let before = quiet.abs().sum().item(Float.self)

        attention.update(parameters: ModuleParameters.unflattened(
            [("attn_sink", MLXArray.zeros([configuration.headCount]) + 8)]))
        let drained = attention(x, mask: nil)
        eval(drained)
        let after = drained.abs().sum().item(Float.self)

        XCTAssertLessThan(after, before, "a stronger sink takes mass from the keys")
    }

    // MARK: The compressor and indexer

    // Compression pools `ratio` consecutive positions into one, so a sequence shortens by that factor
    // and a sequence too short to fill a window compresses to nothing.
    func testTheCompressorPoolsByItsRatio() throws {
        try requireMLXRuntime()
        let configuration = small
        for ratio in [4, 8] {
            let compressor = NFKDeepSeekCompressor(configuration, ratio: ratio,
                                                   headDimensions: configuration.headDimensions)
            let x = MLXArray.zeros([1, ratio * 3, configuration.hiddenSize]) + 0.1
            let pooled = try XCTUnwrap(compressor(x))
            eval(pooled)
            XCTAssertEqual(pooled.shape, [1, 3, configuration.headDimensions],
                           "ratio \(ratio) pools three windows")
            XCTAssertNil(compressor(MLXArray.zeros([1, ratio - 1, configuration.hiddenSize])),
                         "a partial window compresses to nothing")
        }
    }

    // The indexer selects compressed positions, and may only choose windows that are already complete
    // at the querying position — an early token has nothing to look at, which the reference marks −1.
    func testTheIndexerSelectsOnlyCompletedWindows() throws {
        try requireMLXRuntime()
        var configuration = small
        configuration.indexTopK = 2
        let indexer = NFKDeepSeekIndexer(configuration, ratio: 4)
        let length = 16
        let x = MLXArray.zeros([1, length, configuration.hiddenSize]) + 0.1
        let query = MLXArray.zeros([1, length, configuration.queryLoRARank]) + 0.1

        let chosen = try XCTUnwrap(indexer(x, lowRankQuery: query))
        eval(chosen)
        XCTAssertEqual(chosen.shape, [1, length, 2])
        let values = chosen.asArray(Int32.self)

        // Position 0 sits inside the first window, so no window is complete for it.
        XCTAssertEqual(values[0], -1, "the first token selects nothing")
        // A later position may only pick windows strictly before its own.
        for token in 0 ..< length {
            for slot in 0 ..< 2 {
                let picked = Int(values[token * 2 + slot])
                if picked >= 0 {
                    XCTAssertLessThan(picked, (token + 1) / 4, "token \(token) saw a future window")
                }
            }
        }
    }

    private func worstDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    // MARK: Dequantization

    /// The record holds real bytes from the release beside the float weights they decode to.
    ///
    /// @discussion `output` is the fp8 expectation, which is the record's primary result; the 4-bit
    /// pair travels beside it.
    private func quantizationRecord() throws -> [String: MLXArray] {
        guard let path = config["IK_PARITY_DEEPSEEK_QUANT"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_QUANT (Tools/reference-parity deepseek_quant)")
        }
        return try loadArrays(url: URL(fileURLWithPath: path))
    }

    func testTheEightBitDequantizationMatchesTheReference() throws {
        try requireMLXRuntime()
        let record = try quantizationRecord()
        let decoded = NFKMLXDeepSeekQuantization.dequantizeFP8(
            bytes: record["fp8_bytes"]!, scaleBytes: record["fp8_scale_bytes"]!)
        let expected = record["output"]!
        XCTAssertEqual(decoded.shape, expected.shape)
        XCTAssertEqual(worstDifference(decoded, expected), 0, accuracy: 0,
                       "fp8 decode is exact — every value is representable in float32")
    }

    func testTheFourBitDequantizationMatchesTheReference() throws {
        try requireMLXRuntime()
        let record = try quantizationRecord()
        let decoded = NFKMLXDeepSeekQuantization.dequantizeFP4(
            packedBytes: record["fp4_bytes"]!, scaleBytes: record["fp4_scale_bytes"]!)
        let expected = record["fp4_expected"]!
        XCTAssertEqual(decoded.shape, expected.shape)
        XCTAssertEqual(worstDifference(decoded, expected), 0, accuracy: 0,
                       "fp4 decode is exact — eight magnitudes times a power of two")
    }

    /// The 4-bit blocks run along the LAST axis, which the checkpoint's own values corroborate.
    ///
    /// @discussion The reference quantizer clamps a block to ±6 and rounds its scale to the power of
    /// two that puts the block's largest magnitude in `(3, 6]`. So under the right grouping EVERY
    /// block lands in that range, and under a wrong one some do not. Grouping down the rows instead
    /// leaves about 1% of the blocks outside it, which is what makes this a check rather than a
    /// restatement.
    func testTheFourBitBlocksRunAlongTheLastAxis() throws {
        try requireMLXRuntime()
        let record = try quantizationRecord()
        let decoded = NFKMLXDeepSeekQuantization.dequantizeFP4(
            packedBytes: record["fp4_bytes"]!, scaleBytes: record["fp4_scale_bytes"]!)
        let scales = record["fp4_scale_bytes"]!.asType(.int32)
        let blockSize = NFKMLXDeepSeekQuantization.fp4BlockSize
        let blocks = abs(decoded).reshaped([decoded.shape[0], -1, blockSize]).max(axis: -1)
        let scale = MLXArray(NFKMLXDeepSeekQuantization.scaleValues).take(scales.flattened())
            .reshaped(scales.shape)
        let ratio = blocks / scale
        XCTAssertEqual(ratio.max().item(Float.self), 6, accuracy: 1e-6,
                       "no value exceeds the format's own maximum")
        XCTAssertGreaterThan(ratio.min().item(Float.self), 3,
                             "every block's scale is the tightest power of two that holds it")
    }

    /// A byte holds the earlier value in its LOW nibble.
    ///
    /// @discussion No statistic separates the two orders — a byte's pair decodes to the same values
    /// either way, and both stay inside one block — so the checkpoint cannot settle it. This pins the
    /// format's own convention against a hand-encoded byte instead.
    func testTheFourBitNibbleOrderIsTheFormatsOwn() throws {
        try requireMLXRuntime()
        // 0x1 is 0.5 and 0x7 is 6.0, so 0x71 is the pair (0.5, 6.0) in that order.
        let byte = MLXArray([UInt8(0x71)]).reshaped([1, 1])
        let unitScale = MLXArray([UInt8(127)]).reshaped([1, 1])
        let decoded = NFKMLXDeepSeekQuantization.dequantizeFP4(packedBytes: byte,
                                                              scaleBytes: unitScale)
        XCTAssertEqual(decoded.shape, [1, 2])
        XCTAssertEqual(decoded[0, 0].item(Float.self), 0.5, accuracy: 0)
        XCTAssertEqual(decoded[0, 1].item(Float.self), 6, accuracy: 0)
    }

    func testTheScaleFormatIsAnExponentAlone() throws {
        let values = NFKMLXDeepSeekQuantization.scaleValues
        XCTAssertEqual(values[127], 1, accuracy: 0)
        XCTAssertEqual(values[128], 2, accuracy: 0)
        XCTAssertEqual(values[126], 0.5, accuracy: 0)
        XCTAssertTrue(values[255].isNaN, "the all-ones exponent is the format's only NaN")
    }
}
