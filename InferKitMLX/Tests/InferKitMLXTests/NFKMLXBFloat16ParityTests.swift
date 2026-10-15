//
//  NFKMLXBFloat16ParityTests.swift
//  InferKitMLXTests
//
//  Released decoders at the precision they ship in, against their reference built at that precision.
//  A float32 parity run cannot see a rounding point placed differently from the reference's: a norm
//  that rounds twice where the reference rounds once, or a product formed in bf16 where the reference
//  forms it in float32. In bf16 such a difference moves roundings, and it shows as an element count.
//
//  Each record pair comes from `run_reference.py` on one prompt at two precisions (`IK_*_DTYPE=bfloat16`
//  and float32), so the pair differs in network precision alone. Per hidden state the report gives:
//
//  - `differing`: the fraction of elements where this side's bf16 differs from the reference's bf16.
//  - `ours-vs-ref-bf16`: `1 - cosine` of this side's bf16 against the reference's bf16.
//  - `floor`: `1 - cosine` of the reference's bf16 against its own float32, the precision floor.
//  - `ours-vs-f32`: `1 - cosine` of this side's bf16 against the reference's float32.
//
//  Two checks hold a port to the reference's rounding placement:
//
//  - Isolated, each block runs on the reference's own bf16 input. With every rounding placed as the
//    reference places it, what remains is GEMM summation order (Metal against torch's CPU kernels) and
//    the last float32 bit of a transcendental (Metal's `tanh` against torch's), which flip a few
//    elements in ten thousand and read under a quarter of the floor. A misplaced rounding reads at the
//    floor or above it.
//  - End to end, this side's bf16 sits no farther from float32 than twice the reference's own bf16
//    does. Past the first flipped element the two bf16 streams diverge chaotically, so the end-to-end
//    element count carries no signal; the distance from exact arithmetic does.
//

import CryptoKit
import XCTest
import Foundation
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXBFloat16ParityTests: XCTestCase {

    override func setUp() {
        super.setUp()
        NFKMLXHalfPrecision.arithmetic = .reference
    }

    override func tearDown() {
        NFKMLXHalfPrecision.arithmetic = .fast
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private var config: [String: String] { NFKMLXValidationConfig.environment }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func existing(_ path: String?, _ label: String) throws -> String {
        guard let path, FileManager.default.fileExists(atPath: path) else { throw XCTSkip("no \(label)") }
        return path
    }

    /// `IK_VALIDATION_RECORDS/<name>`, defaulting to `records` under the validation root.
    private func record(_ name: String) throws -> [String: MLXArray] {
        let root = config["IK_VALIDATION_RECORDS"]
            ?? NFKMLXValidationConfig.root.appendingPathComponent("records").path
        let path = try existing(URL(fileURLWithPath: root).appendingPathComponent(name).path, name)
        return try loadArrays(url: URL(fileURLWithPath: path))
    }

    private func floats(_ array: MLXArray) -> [Double] {
        array.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init)
    }

    private func distance(_ a: [Double], _ b: [Double]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(a.count, b.count) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return 1 - dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
    }

    /// One row of the report.
    struct Seam {
        let label: String
        let differing: Double
        let ours: Double
        let floor: Double
        /// `1 - cosine` of this side's bf16 against the reference's float32.
        let exact: Double
        /// `1 - cosine` of the reference's bf16 against the same block run in float32 on the bf16 run's own
        /// input (`<key>.floor`, `IK_PROBE_BLOCK_FLOOR=1`): the rounding that block adds, where recorded.
        var blockFloor: Double? = nil
        /// `1 - cosine` of the reference's half output against the farthest of its own block's reruns, each with its
        /// products, moments, and convolutions summed in another float32 order (`<key>.orderfloor`,
        /// `IK_PROBE_ORDER_FLOOR=1`), where recorded.
        var orderFloor: Double? = nil
        /// Each summation order's distance from the reference's half output (`<key>.order.<order>`, one value).
        var orders: [(String, Double)] = []
        /// Each summation order's whole-model rerun (`<key>.order.<order>`, the seam's shape,
        /// `IK_PROBE_OUTPUT_ORDER_FLOOR=1`) as its `1 - cosine` against the reference's float32.
        var orderRuns: [(String, Double)] = []

        /// `ours` as a fraction of the block floor, or nil where none is recorded or the block adds no rounding
        /// (a layer norm CUDA autocast computes in float32 has a block floor of 0).
        var blockRatio: Double? {
            guard let blockFloor, blockFloor > 1e-13 else { return nil }
            return ours / blockFloor
        }
    }

    /// Compares `states` (this side, bf16) against the `<prefix>i` tensors of both records.
    private func seams(_ states: [(String, MLXArray)], bf16: [String: MLXArray], f32: [String: MLXArray],
                       keys: [String]) throws -> [Seam] {
        try zip(states, keys).map { state, key in
            let ours = floats(state.1)
            let reference = floats(try XCTUnwrap(bf16[key], "no \(key) in the bf16 record"))
            let exact = floats(try XCTUnwrap(f32[key], "no \(key) in the float32 record"))
            XCTAssertEqual(ours.count, reference.count, "\(state.0): element count")
            let differing = zip(ours, reference).filter { $0 != $1 }.count
            return Seam(label: state.0, differing: Double(differing) / Double(max(ours.count, 1)),
                        ours: distance(ours, reference), floor: distance(reference, exact),
                        exact: distance(ours, exact),
                        blockFloor: bf16["\(key).floor"].map { distance(reference, floats($0)) },
                        orderFloor: bf16["\(key).orderfloor"].map { distance(reference, floats($0)) },
                        orders: orderValues(key, bf16).filter { $0.1.count == 1 }.map { ($0.0, $0.1[0]) },
                        orderRuns: orderValues(key, bf16).filter { $0.1.count > 1 }.map { ($0.0, distance($0.1, exact)) })
        }
    }

    /// Each `<key>.order.<order>` of `record`, by order.
    private func orderValues(_ key: String, _ record: [String: MLXArray]) -> [(String, [Double])] {
        record.keys.filter { $0.hasPrefix("\(key).order.") }.sorted().map {
            (String($0.dropFirst("\(key).order.".count)), floats(record[$0]!))
        }
    }

    /// Prints `rows` under `name`. The block floors print only in a report of seams run on the reference's own
    /// input (named `isolated` or `pieces`): an end-to-end state carries every earlier block's drift, so its
    /// distance against one block's floor measures nothing.
    private func report(_ name: String, _ rows: [Seam]) {
        let blockFloors = (name.contains("isolated") || name.contains("pieces")) && rows.contains { $0.blockFloor != nil }
        let orderFloors = rows.contains { $0.orderFloor != nil }
        var lines = ["VALIDATION bf16 \(name):  seam  differing  ours-vs-ref-bf16  floor(ref bf16 vs f32)  ours-vs-f32"
                     + (blockFloors ? "  block-floor  ours/block-floor" : "")
                     + (orderFloors ? "  order-floor  ours/order-floor" : "")]
        for row in rows {
            let block = !blockFloors ? "" : row.blockFloor.map {
                String(format: "  %.3e  %@", $0, row.blockRatio.map { String(format: "%.4f", $0) } ?? "float32")
            } ?? ""
            let order = row.orderFloor.map { String(format: "  %.3e  %.4f", $0, row.ours / max($0, 1e-30)) } ?? ""
            lines.append(String(format: "  %-16s %8.4f%%  %.3e  %.3e  %.3e%@%@%@", (row.label as NSString).utf8String!,
                                row.differing * 100, row.ours, row.floor, row.exact, block, order,
                                row.exact > 2 * row.floor ? "  <- farther from f32 than twice the floor" : ""))
            if !row.orders.isEmpty {
                lines.append("      summation orders: "
                             + row.orders.map { String(format: "%@ %.3e", $0.0, $0.1) }.joined(separator: ", "))
            }
            if !row.orderRuns.isEmpty {
                lines.append("      summation orders vs f32, of the floor: "
                             + row.orderRuns.map { String(format: "%@ %.2f", $0.0, $0.1 / max(row.floor, 1e-30)) }
                                 .joined(separator: ", ")
                             + String(format: "; ours %.2f", row.exact / max(row.floor, 1e-30)))
            }
        }
        print(lines.joined(separator: "\n"))
    }

    // MARK: MLX's half-precision primitives against torch's

    // torch's bf16 `F.linear` with a bias rounds once: `round(x·Wᵀ + b)`, formed wide. This pins what
    // MLX's biased `Linear` (`addMM`) does on the same operands.
    func testABiasedLinearRoundsOnceInBFloat16() throws {
        try requireMLXRuntime()
        MLXRandom.seed(7)
        let x = MLXRandom.normal([6, 64]).asType(.bfloat16)
        let w = (MLXRandom.normal([64, 64]) * 0.1).asType(.bfloat16)
        let b = MLXRandom.normal([64]).asType(.bfloat16)
        let fused = Linear(weight: w, bias: b)(x)
        let once = (matmul(x.asType(.float32), w.asType(.float32).T) + b.asType(.float32)).asType(.bfloat16)
        let twice = matmul(x, w.T) + b
        eval(fused, once, twice)
        let againstOnce = (fused .!= once).sum().item(Int.self), againstTwice = (fused .!= twice).sum().item(Int.self)
        print("VALIDATION bf16 primitives: biased Linear differs from one rounding in \(againstOnce) of 384, "
              + "from matmul-then-add in \(againstTwice)")
        XCTAssertEqual(againstOnce, 0, "a biased Linear rounds once, as torch's does")
    }

    // MLX's `addmm` on one bf16 row rounds the product and then the bias, while torch's `linear` rounds
    // once at every row count (measured: torch 2.8 and 2.14 agree with one rounding at 1 to 64 rows; MLX
    // 0.32.2 differs in 86 of 256 at one row). `NFKLinear`, which every loaded biased Linear becomes,
    // rounds once on one row, in any leading shape.
    func testASingleRowBiasedLinearRoundsOnceInBFloat16() throws {
        try requireMLXRuntime()
        MLXRandom.seed(11)
        let w = (MLXRandom.normal([256, 256]) * 0.05).asType(.bfloat16)
        let b = (MLXRandom.normal([256]) * 0.1).asType(.bfloat16)
        for shape in [[1, 256], [1, 1, 256], [256]] {
            let x = MLXRandom.normal(shape).asType(.bfloat16)
            let once = (matmul(x.asType(.float32).reshaped([1, -1]), w.asType(.float32).T) + b.asType(.float32))
                .asType(.bfloat16).reshaped(Array(shape.dropLast()) + [256])
            let mine = NFKLinear(weight: w, bias: b)(x)
            let plain = Linear(weight: w, bias: b)(x)
            eval(once, mine, plain)
            XCTAssertEqual(mine.shape, once.shape)
            let againstOnce = (mine .!= once).sum().item(Int.self), plainAgainstOnce = (plain .!= once).sum().item(Int.self)
            print("VALIDATION bf16 primitives: single-row Linear \(shape) differs from one rounding in \(againstOnce) of 256 "
                  + "(MLXNN's Linear: \(plainAgainstOnce))")
            XCTAssertEqual(againstOnce, 0, "a single-row biased Linear rounds once, as torch's does")
        }
        final class Holder: Module {
            @ModuleInfo(key: "projection") var projection: Linear
            @ModuleInfo(key: "unbiased") var unbiased: Linear
            @ModuleInfo(key: "stack") var stack: [Module]
            override init() {
                _projection.wrappedValue = Linear(256, 256)
                _unbiased.wrappedValue = Linear(256, 256, bias: false)
                _stack.wrappedValue = [Linear(256, 256), GELU(), Linear(256, 256)]
            }
        }
        let holder = Holder()
        holder.projection.freeze(keys: ["weight"])
        XCTAssertFalse(holder.trainableParameters().flattened().map(\.0).contains("projection.weight"))
        try NFKMLXWeights.apply([("projection.weight", w), ("projection.bias", b), ("unbiased.weight", w),
                                 ("stack.0.weight", w), ("stack.0.bias", b), ("stack.2.weight", w), ("stack.2.bias", b)],
                                to: holder)
        XCTAssertTrue(holder.projection is NFKLinear, "a loaded biased Linear becomes NFKLinear")
        XCTAssertFalse(holder.unbiased is NFKLinear, "an unbiased Linear has no bias to round apart")
        XCTAssertTrue(holder.stack[0] is NFKLinear && holder.stack[1] is GELU && holder.stack[2] is NFKLinear,
                      "a mixed array keeps its order")
        XCTAssertFalse(holder.trainableParameters().flattened().map(\.0).contains("projection.weight"),
                       "the replacement keeps what was frozen frozen")
    }

    // An expert product through `gatherMM` at bf16, against the float32 product rounded once, which is
    // what torch's bf16 matmul returns.
    func testGatherMMRoundsOnceInBFloat16() throws {
        try requireMLXRuntime()
        MLXRandom.seed(11)
        let x = MLXRandom.normal([8, 1, 1, 64]).asType(.bfloat16)
        let w = (MLXRandom.normal([4, 64, 64]) * 0.1).asType(.bfloat16)
        let chosen = MLXArray([0, 2, 1, 3, 3, 0, 2, 1, 1, 2, 0, 3, 2, 3, 0, 1] as [Int32]).reshaped([8, 2])
        let gathered = gatherMM(x, w, rhsIndices: chosen)
        let once = gatherMM(x.asType(.float32), w.asType(.float32), rhsIndices: chosen).asType(.bfloat16)
        eval(gathered, once)
        let differing = (gathered .!= once).sum().item(Int.self)
        print("VALIDATION bf16 primitives: gatherMM differs from one rounding in \(differing) of \(once.size)")
        XCTAssertEqual(differing, 0, "gatherMM rounds once, as torch's bf16 matmul does")
    }

    // The encoders' primitives at bf16 against the float32 computation rounded once, which is what
    // torch's bf16 `layer_norm`, `gelu`, `softmax`, and `conv` return. Reported, not asserted: the
    // encoders' rounding is measured against their own references.
    func testEncoderPrimitivesAgainstOneRoundingInBFloat16() throws {
        try requireMLXRuntime()
        MLXRandom.seed(5)
        let x = MLXRandom.normal([16, 128]).asType(.bfloat16)
        let wide = x.asType(.float32)
        let norm = LayerNorm(dimensions: 128, eps: 1e-6)
        norm.update(parameters: ModuleParameters.unflattened([
            "weight": (MLXRandom.normal([128]) * 0.1 + 1).asType(.bfloat16),
            "bias": (MLXRandom.normal([128]) * 0.1).asType(.bfloat16)]))
        let fusedNorm = norm(x)
        let wideNorm = MLXFast.layerNorm(wide, weight: norm.weight!.asType(.float32), bias: norm.bias!.asType(.float32),
                                         eps: 1e-6).asType(.bfloat16)
        let conv = Conv1d(inputChannels: 8, outputChannels: 8, kernelSize: 3)
        conv.update(parameters: ModuleParameters.unflattened([
            "weight": (MLXRandom.normal([8, 3, 8]) * 0.2).asType(.bfloat16), "bias": MLXRandom.normal([8]).asType(.bfloat16)]))
        let signal = MLXRandom.normal([1, 32, 8]).asType(.bfloat16)
        let wideConv = conv1d(signal.asType(.float32), conv.weight.asType(.float32)) + conv.bias!.asType(.float32)
        let rows: [(String, MLXArray, MLXArray)] = [
            ("LayerNorm", fusedNorm, wideNorm),
            ("gelu", MLXNN.gelu(x), MLXNN.gelu(wide).asType(.bfloat16)),
            ("geluApproximate", MLXNN.geluApproximate(x), MLXNN.geluApproximate(wide).asType(.bfloat16)),
            ("softmax precise", softmax(x, axis: -1, precise: true), softmax(wide, axis: -1).asType(.bfloat16)),
            ("sigmoid", sigmoid(x), sigmoid(wide).asType(.bfloat16)),
            ("exp", exp(x), exp(wide).asType(.bfloat16)),
            ("Conv1d", conv(signal), wideConv.asType(.bfloat16)),
        ]
        var lines = [String]()
        for (name, ours, once) in rows {
            eval(ours, once)
            lines.append("\(name) \((ours .!= once).sum().item(Int.self)) of \(once.size)")
        }
        print("VALIDATION bf16 primitives against one rounding: " + lines.joined(separator: ", "))
    }

    // gpt-oss's clamped SwiGLU expert, step by step, against torch's bf16 steps for expert 0 of the tiny
    // record (`records/gpt_oss_expert0_steps`): each step on torch's own input.
    func testTheClampedSwiGLUStepsMatchTorchInBFloat16() throws {
        try requireMLXRuntime()
        let steps = try record("gpt_oss_expert0_steps.safetensors")
        let bf16 = try record("gpt_oss_tiny_bf16.safetensors")
        let combineRecord = try record("gpt_oss_combine0.safetensors")
        func t(_ key: String) throws -> MLXArray {
            try XCTUnwrap(steps[key] ?? combineRecord[key], "no \(key)").asType(.bfloat16)
        }
        let prefix = "w::model.layers.0.mlp.experts."
        let gateUp = try XCTUnwrap(bf16[prefix + "gate_up_proj"]).asType(.bfloat16)[0]
        let gateUpBias = try XCTUnwrap(bf16[prefix + "gate_up_proj_bias"]).asType(.bfloat16)[0]
        func count(_ ours: MLXArray, _ key: String) throws -> String {
            eval(ours)
            return "\(key) \((ours .!= (try t(key))).sum().item(Int.self))"
        }
        let x = try t("x")
        var lines = [try count(matmul(x, gateUp), "gu"), try count(try t("gu") + gateUpBias, "gub")]
        let gub = try t("gub")
        lines.append(try count(clip(gub[.ellipsis, .stride(from: 0, by: 2)], max: 7.0), "gc"))
        lines.append(try count(clip(gub[.ellipsis, .stride(from: 1, by: 2)], min: -7.0, max: 7.0), "uc"))
        lines.append(try count(NFKReferenceRounding.scaled(try t("gc"), by: 1.702), "ga"))
        lines.append(try count(NFKReferenceRounding.sigmoid(try t("ga")), "s"))
        lines.append(try count(try t("gc") * (try t("s")), "glu"))
        lines.append(try count(try t("uc") + 1, "up1"))
        lines.append(try count(try t("up1") * (try t("glu")), "gated"))
        print("VALIDATION bf16 clamped SwiGLU steps differing: " + lines.joined(separator: ", "))
        XCTAssertTrue(lines.allSatisfy { $0.hasSuffix(" 0") }, "every step rounds as torch's does")

        // The routed layer end to end on the same input, against torch's expert outputs, weights, and sum.
        var geometry = NFKMLXLanguageConfiguration(
            hiddenSize: 64, layerCount: 4, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            intermediateSize: 32, vocabularySize: 128, ropeTheta: 150_000, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: false, normalizesQueryAndKey: false, attentionBias: true)
        geometry.expertCount = 4
        geometry.activeExpertCount = 2
        geometry.expertIntermediateSize = 32
        geometry.normalizesExpertWeights = true
        geometry.slidingWindows = [4, nil, 4, nil]
        geometry.attentionSinks = true
        geometry.outputProjectionBias = true
        geometry.routerBias = true
        geometry.clampedSwiGLU = NFKMLXClampedSwiGLU()
        let net = NFKMLXLanguage.makeNet(geometry)
        let weights = bf16.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value.asType(.bfloat16)) : nil
        }
        try NFKMLXWeights.apply(NFKMLXLanguage.releaseWeights(weights), to: net, verifyShapes: false)
        let mixture = try XCTUnwrap(net.model.layers[0].feedForward as? NFKLMMixtureFeedForward)
        let (routeWeights, chosen) = mixture.route(x)
        let outputs = mixture.experts(x, experts: chosen)          // [tokens, active, hidden]
        eval(routeWeights, chosen, outputs)
        let slots = chosen.asArray(Int32.self)
        var outputLines = [String]()
        for expert in [2, 3] {
            let slot = slots[0] == Int32(expert) ? 0 : 1
            outputLines.append(try count(outputs[0..., slot, 0...], "out\(expert)")
                .replacingOccurrences(of: "out\(expert)", with: "expert \(expert) output"))
            outputLines.append("expert \(expert) weight \((routeWeights[0..., slot] .!= (try t("w\(expert)"))).sum().item(Int.self))")
        }
        outputLines.append(try count(NFKReferenceRounding.combined(outputs, weights: routeWeights, chosen: chosen), "combined"))
        print("VALIDATION bf16 gpt-oss routed layer differing (slots \(slots.prefix(2))): " + outputLines.joined(separator: ", "))
    }

    // MARK: Gemma 3

    private func gemma3(_ name: String, directoryKey: String) throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config[directoryKey], directoryKey))
        let bf16 = try record("gemma3_\(name)_bf16.safetensors")
        let f32 = try record("gemma3_\(name)_f32eager.safetensors")
        let tokens = try XCTUnwrap(bf16["tokens"]).asArray(Int32.self)

        let geometry = try NFKMLXGemma3Language.configuration(
            fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXGemma3Language.makeNet(geometry)
        try NFKMLXGemma3Language.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        let input = MLXArray(tokens).reshaped([1, tokens.count])
        let states = net.layerStates(input)
        let logits = net(input)[0]
        eval(states + [logits])

        let labelled = states.enumerated().map { ($0.offset == 0 ? "embedding" : "layer \($0.offset - 1)", $0.element) }
            + [("logits", logits)]
        let keys = states.indices.map { "hidden.\($0)" } + ["output"]
        let rows = try seams(labelled, bf16: bf16, f32: f32, keys: keys)
        report("gemma3-\(name)", rows)

        // Each block on the reference's own bf16 input, so a layer's count is its own and not what the
        // layers before it passed on.
        let masks = NFKMLXGemma3Masks.make(length: tokens.count, offset: 0, window: geometry.slidingWindow,
                                           blockIds: nil, bidirectional: false)
        var isolated = [(String, MLXArray)]()
        for (index, block) in net.layers.enumerated() {
            let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            var output = block(input, mask: geometry.layerTypes[index] == .full ? masks.full : masks.sliding,
                               cache: nil, layer: index)
            if index == net.layers.count - 1 { output = net.norm(output) }
            isolated.append(("layer \(index)", output))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32,
                                     keys: isolated.indices.map { "hidden.\($0 + 1)" })
        report("gemma3-\(name) isolated", isolatedRows)
        assertRoundingPlacement("gemma3-\(name)", endToEnd: rows, isolated: isolatedRows)
    }

    /// The two checks the file header states, over a report's rows.
    /// `isolatedBar` is the fraction of the floor a layer run alone on the reference's input may reach. A seam
    /// with a recorded order floor is held to that instead: the farthest the reference's own block moves when
    /// its products, moments, and convolutions sum in another float32 order. An end-to-end state whose record
    /// carries whole-model reruns is held to the farthest of them from the float32 run in place of twice the floor.
    private func assertRoundingPlacement(_ name: String, endToEnd: [Seam], isolated: [Seam], isolatedBar: Double = 0.25,
                                         file: StaticString = #filePath, line: UInt = #line) {
        let worst = isolated.max { $0.ours / max($0.floor, 1e-30) < $1.ours / max($1.floor, 1e-30) }!
        let last = endToEnd[endToEnd.count - 1]
        print(String(format: "VALIDATION bf16 %@ summary: worst isolated %@ at %.4f of the floor; %@ ours-vs-f32 %.3e, floor %.3e",
                     name, worst.label, worst.ours / max(worst.floor, 1e-30), last.label, last.exact, last.floor))
        let blocked = isolated.filter { $0.blockRatio != nil }
        if let worstBlock = blocked.max(by: { $0.blockRatio! < $1.blockRatio! }) {
            let ratios = blocked.map { $0.blockRatio! }.sorted()
            let unrounded = isolated.filter { $0.blockFloor != nil && $0.blockRatio == nil }.count
            print(String(format: "VALIDATION bf16 %@ block floors: %d of %d seams%@, median %.4f, worst %@ at %.4f of its block floor",
                         name, blocked.count, isolated.count, unrounded > 0 ? String(format: " (%d in float32)", unrounded) : "",
                         ratios[ratios.count / 2], worstBlock.label, worstBlock.blockRatio!))
        }
        for row in isolated {
            if let orderFloor = row.orderFloor {
                XCTAssertLessThanOrEqual(row.ours, orderFloor,
                                         "\(name) \(row.label) on the reference's input: farther from it than a summation order moves it",
                                         file: file, line: line)
                continue
            }
            XCTAssertLessThan(row.ours, isolatedBar * row.floor,
                              "\(name) \(row.label) on the reference's input: a rounding placed unlike the reference's",
                              file: file, line: line)
        }
        assertWithinBlockFloors(name, isolated, file: file, line: line)
        for row in endToEnd.suffix(2) {
            if let widest = row.orderRuns.map(\.1).max() {
                XCTAssertLessThanOrEqual(row.exact, widest,
                                         "\(name) \(row.label): farther from float32 than the reference's widest summation order",
                                         file: file, line: line)
                continue
            }
            XCTAssertLessThanOrEqual(row.exact, 2 * row.floor,
                                     "\(name) \(row.label): farther from float32 than twice the reference's bf16",
                                     file: file, line: line)
        }
    }

    /// Each seam with a recorded block floor stays inside it: run on the reference's own input, a block may
    /// differ from the reference's bf16 output by no more than the rounding that block adds itself. The widest
    /// measured is SAM 3's mask decoder at 0.87 of it. SigLIP 2's pooling head reaches 0.66 from one
    /// weighted-sum element a float32 accumulation order puts a step from torch's, which its 768-wide out
    /// projection spreads over every output.
    private func assertWithinBlockFloors(_ name: String, _ rows: [Seam], file: StaticString = #filePath,
                                         line: UInt = #line) {
        for row in rows {
            guard let ratio = row.blockRatio else { continue }
            XCTAssertLessThan(ratio, 1, "\(name) \(row.label) on the reference's input: farther from it than its own block's rounding",
                              file: file, line: line)
        }
    }

    // MARK: Gemma 3 vision

    // The Gemma 3 4B vision tower and projector at the released bf16 (`.checkpoint`) against transformers
    // at bf16 with eager attention (`gemma3_vision_probe`, layers 0 and 26 probed), on the release's own
    // pixel values, every layer's state and the soft tokens.
    func testGemma3VisionInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA3_4B"], "IK_VAL_GEMMA3_4B"))
        let bf16 = try record("gemma3_4b_vision_bf16.safetensors"), f32 = try record("gemma3_4b_vision_f32.safetensors")
        let (vision, projector) = try NFKMLXGemma3.visionParts(directoryURL: directory, precision: .checkpoint)
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let pixels = try XCTUnwrap(bf16["pixel_values"]).expandedDimensions(axis: 0).transposed(0, 2, 3, 1)
        let layers = vision.encoder.layers

        var hidden = vision.embeddings(pixels.asType(.bfloat16))
        var states = [("embeddings", hidden[0])]
        var isolated = [(String, MLXArray)]()
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden)
            states.append(("layer \(index)", hidden[0]))
            isolated.append(("layer \(index)", layer(try recorded("vit.hidden.\(index)"))[0]))
        }
        let post = vision.postLayerNorm(hidden)
        let tokens = projector(vision(pixels))
        XCTAssertEqual(tokens.dtype, .bfloat16, "a bf16 load computes the image path in bf16")
        states += [("post norm", post[0]), ("soft tokens", tokens[0])]
        isolated += [("post norm", vision.postLayerNorm(try recorded("vit.hidden.\(layers.count)"))[0]),
                     ("soft-token norm", projector.norm(try recorded("proj.pooled"))[0]),
                     ("projector", projector(try recorded("vit.post"))[0])]
        eval(states.map(\.1) + isolated.map(\.1))
        let rows = try seams(states, bf16: bf16, f32: f32,
                             keys: ["vit.embeddings"] + (1 ... layers.count).map { "vit.hidden.\($0)" } + ["vit.post", "output"])
        report("gemma3-4b-vision", rows)
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32,
                                     keys: (1 ... layers.count).map { "vit.hidden.\($0)" } + ["vit.post", "proj.norm", "output"])
        report("gemma3-4b-vision isolated", isolatedRows)
        assertRoundingPlacement("gemma3-4b-vision", endToEnd: rows, isolated: isolatedRows)
    }

    // MARK: SigLIP 2 and CLIP at half precision

    // SigLIP 2 base (`IK_VAL_SIGLIP2`) with its float32 release rounded to bf16, as a 16-bit file loads, against
    // transformers at bf16 with eager attention (`siglip2_probe`): every vision and text layer, the pooling
    // head, and the normalized embeddings and logits.
    func testSigLIP2InBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let weights = try existing(config["IK_VAL_SIGLIP2"], "IK_VAL_SIGLIP2")
        let bf16 = try record("siglip2_bf16.safetensors"), f32 = try record("siglip2_f32.safetensors")
        let net = NFKMLXSigLIP2.makeNet(.base)
        try NFKMLXSigLIP2.loadWeights(into: net, from: URL(fileURLWithPath: weights))
        net.update(parameters: net.parameters().mapValues { $0.asType(.bfloat16) })
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let pixels = try XCTUnwrap(bf16["pixel_values"]).expandedDimensions(axis: 0).transposed(0, 2, 3, 1)
        let tokens = try XCTUnwrap(bf16["tokens"]).asType(.int32)
        let vision = net.vision, text = net.text

        var hidden = vision.embeddings(pixels.asType(.bfloat16))
        var states = [("embeddings", hidden[0])], keys = ["vit.embeddings"]
        var isolated = [(String, MLXArray)](), isolatedKeys = [String]()
        for (index, layer) in vision.encoder.layers.enumerated() {
            hidden = layer(hidden)
            states.append(("vision layer \(index)", hidden[0]))
            keys.append("vit.hidden.\(index + 1)")
            isolated.append(("vision layer \(index)", layer(try recorded("vit.hidden.\(index)"))[0]))
            isolatedKeys.append("vit.hidden.\(index + 1)")
        }
        let post = vision.postLayerNorm(hidden)
        let pooled = vision.head(post)
        states += [("post norm", post[0]), ("head", pooled[0])]
        keys += ["vit.post", "head.block.out"]
        let layerCount = vision.encoder.layers.count
        isolated += [("post norm", vision.postLayerNorm(try recorded("vit.hidden.\(layerCount)"))[0]),
                     ("head", vision.head(try recorded("vit.post"))[0])]
        isolatedKeys += ["vit.post", "head.block.out"]

        var words = text.embeddings(tokens)
        for (index, layer) in text.encoder.layers.enumerated() {
            words = layer(words)
            states.append(("text layer \(index)", words[0]))
            keys.append("text.hidden.\(index + 1)")
            isolated.append(("text layer \(index)", layer(try recorded("text.hidden.\(index)"))[0]))
            isolatedKeys.append("text.hidden.\(index + 1)")
        }
        let textCount = text.encoder.layers.count
        states.append(("text final norm", text.finalLayerNorm(words)[0]))
        keys.append("text.final")
        isolated.append(("text final norm", text.finalLayerNorm(try recorded("text.hidden.\(textCount)"))[0]))
        isolatedKeys.append("text.final")
        let imageEmbedding = net.imageEmbedding(pixels)
        let textEmbedding = net.textEmbedding(tokens)
        XCTAssertEqual(imageEmbedding.dtype, .bfloat16, "a bf16 load computes the image path in bf16")
        states += [("logits", net.logits(image: pixels, text: tokens).reshaped([-1])), ("text embeddings", textEmbedding),
                   ("image embedding", imageEmbedding[0])]
        keys += ["logits", "text_embeds", "output"]
        eval(states.map(\.1) + isolated.map(\.1))
        let rows = try seams(states, bf16: bf16, f32: f32, keys: keys)
        report("siglip2", rows)
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolatedKeys)
        report("siglip2 isolated", isolatedRows)
        assertRoundingPlacement("siglip2", endToEnd: rows, isolated: isolatedRows)
    }

    // CLIP ViT-B/32 (`IK_VAL_CLIP`) with its weights rounded to float16, the type OpenAI's own `.pt` stores,
    // against transformers' `CLIPModel` at float16 with eager attention (`clip_probe`, `IK_PROBE_FLOOR=float16`
    // for the floor), on four images: every vision and text layer, the pooled class token, and both
    // embeddings. Each image's layers are held alone; the end-to-end bar reads the four images together,
    // since one image's embedding is a single 512-wide vector.
    func testCLIPInFloat16MatchesTheFloat16Reference() throws {
        try requireMLXRuntime()
        let weights = try existing(config["IK_VAL_CLIP"], "IK_VAL_CLIP")
        let net = NFKMLXCLIPNet(.base)
        try NFKMLXCLIP.loadWeights(into: net, from: URL(fileURLWithPath: weights))
        net.update(parameters: net.parameters().mapValues { $0.asType(.float16) })
        var plates = [(states: [(String, MLXArray)], keys: [String], half: [String: MLXArray], f32: [String: MLXArray])]()
        for plate in ["", "_photo", "_face", "_subject"] {
            let half: [String: MLXArray], f32: [String: MLXArray]
            do {
                half = try record("clip_b32\(plate)_f16.safetensors")
                f32 = try record("clip_b32\(plate)_f32.safetensors")
            } catch is XCTSkip {
                continue
            }
            let (states, keys, isolated, isolatedKeys) = try clipHalf(net, half)
            let isolatedRows = try seams(isolated, bf16: half, f32: f32, keys: isolatedKeys)
            report("clip-b32\(plate) f16 isolated", isolatedRows)
            for row in isolatedRows {
                XCTAssertLessThan(row.ours, 0.25 * row.floor, "clip-b32\(plate) \(row.label) on the reference's input")
            }
            assertWithinBlockFloors("clip-b32\(plate)", isolatedRows)
            plates.append((states, keys, half, f32))
        }
        let first = try XCTUnwrap(plates.first, "no clip_b32 record")
        func joined(_ arrays: [MLXArray]) -> MLXArray { concatenated(arrays.map { $0.asType(.float32).reshaped([-1]) }) }
        let states = first.states.indices.map { index in (first.states[index].0, joined(plates.map { $0.states[index].1 })) }
        var half = [String: MLXArray](), f32 = [String: MLXArray]()
        for key in first.keys {
            half[key] = joined(try plates.map { try XCTUnwrap($0.half[key], "no \(key)") })
            f32[key] = joined(try plates.map { try XCTUnwrap($0.f32[key], "no \(key)") })
        }
        let rows = try seams(states, bf16: half, f32: f32, keys: first.keys)
        report("clip-b32 f16, \(plates.count) images", rows)
        for row in rows.suffix(2) {
            XCTAssertLessThanOrEqual(row.exact, 2 * row.floor, "clip-b32 f16 \(row.label): farther from float32 than twice the floor")
        }
    }

    /// One CLIP record's end-to-end states and its layers on the reference's own input, with their keys.
    private func clipHalf(_ net: NFKMLXCLIPNet, _ half: [String: MLXArray]) throws
        -> ([(String, MLXArray)], [String], [(String, MLXArray)], [String]) {
        let visual = net.visual
        let patchWeight = visual.conv1.weight.transposed(0, 3, 1, 2).asType(.float32)
        let recordedPatch = try XCTUnwrap(half["w.patch"])
        XCTAssertEqual((patchWeight .!= recordedPatch).sum().item(Int.self), 0,
                       "the release's weights, rounded as the reference rounds them")
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(half[key], "no \(key)").asType(.float16).expandedDimensions(axis: 0)
        }
        let pixels = try XCTUnwrap(half["pixel_values"]).expandedDimensions(axis: 0).transposed(0, 2, 3, 1)
        let tokenIds = try XCTUnwrap(half["tokens"]).asArray(Int32.self).map(Int.init)

        let patches = visual.conv1(pixels.asType(.float16))
        let width = patches.dim(3)
        var hidden = concatenated([visual.classEmbedding.reshaped([1, 1, width]), patches.reshaped([1, -1, width])], axis: 1)
            + visual.positionalEmbedding
        var states = [("embeddings", hidden[0])], keys = ["vit.embeddings"]
        hidden = visual.lnPre(hidden)
        states.append(("pre norm", hidden[0]))
        keys.append("vit.pre")
        var isolated = [("pre norm", visual.lnPre(try recorded("vit.embeddings"))[0])], isolatedKeys = ["vit.pre"]
        for (index, block) in visual.transformer.resblocks.enumerated() {
            hidden = block(hidden, mask: nil)
            states.append(("vision layer \(index)", hidden[0]))
            keys.append("vit.hidden.\(index + 1)")
            isolated.append(("vision layer \(index)", block(try recorded("vit.hidden.\(index)"), mask: nil)[0]))
            isolatedKeys.append("vit.hidden.\(index + 1)")
        }
        let blocks = visual.transformer.resblocks.count
        let pooled = visual.lnPost(hidden[0..., 0])
        states += [("pooled", pooled[0]), ("image features", pooled.matmul(visual.proj)[0])]
        keys += ["vit.pooled", "image_features"]
        isolated += [("pooled", visual.lnPost(try recorded("vit.hidden.\(blocks)")[0..., 0])[0]),
                     ("image features", try recorded("vit.pooled").matmul(visual.proj)[0])]
        isolatedKeys += ["vit.pooled", "image_features"]

        let length = tokenIds.count
        let mask = NFKCLIPInit.causalMask(length)
        var words = net.tokenEmbedding(MLXArray(tokenIds.map(Int32.init)).reshaped([1, length]))
            + net.positionalEmbedding[0 ..< length]
        for (index, block) in net.transformer.resblocks.enumerated() {
            words = block(words, mask: mask)
            states.append(("text layer \(index)", words[0]))
            keys.append("text.hidden.\(index + 1)")
            isolated.append(("text layer \(index)", block(try recorded("text.hidden.\(index)"), mask: mask)[0]))
            isolatedKeys.append("text.hidden.\(index + 1)")
        }
        let end = tokenIds.firstIndex(of: tokenIds.max() ?? 0) ?? (length - 1)
        let final = net.lnFinal(words)
        states += [("text final norm", final[0]), ("text features", final[0, end].reshaped([1, -1]).matmul(net.textProjection)[0])]
        keys += ["text.final", "text_features"]
        isolated += [("text final norm", net.lnFinal(try recorded("text.hidden.\(net.transformer.resblocks.count)"))[0]),
                     ("text features", try recorded("text.final")[0, end].reshaped([1, -1]).matmul(net.textProjection)[0])]
        isolatedKeys += ["text.final", "text_features"]
        states += [("text embedding", net.encodeText(tokenIds)), ("image embedding", NFKCLIPInit.l2Normalize(pooled.matmul(visual.proj))[0])]
        keys += ["text_embeds", "output"]
        XCTAssertEqual(states.last!.1.dtype, .float16, "a float16 load computes the image path in float16")
        eval(states.map(\.1) + isolated.map(\.1))
        return (states, keys, isolated, isolatedKeys)
    }

    // FLUX.1's CLIP-L text encoder (`IK_VAL_FLUX_SCHNELL_FULL`'s `text_encoder/`, released bf16) as FLUX
    // loads it, against transformers' `CLIPTextModel` at bf16 with eager attention (`clip_text_probe`, the
    // float32 side on the same bf16 weights), over one prompt padded to 77 tokens.
    func testFluxCLIPLInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_FLUX_SCHNELL_FULL"], "IK_VAL_FLUX_SCHNELL_FULL"))
        let net = try NFKMLXSDTextEncoder.net(configuration: .stableDiffusion15,
                                              weightsURL: directory.appendingPathComponent("text_encoder/model.safetensors"),
                                              precision: .checkpoint)
        try clipText("flux-clip-l", net: net, half: try record("flux_clip_l_bf16.safetensors"),
                     f32: try record("flux_clip_l_f32.safetensors"), dtype: .bfloat16)
    }

    // SDXL-Turbo's two text towers (`IK_VAL_SDXL_DIR`'s `text_encoder/` CLIP-L and `text_encoder_2/`
    // OpenCLIP-bigG, released float16) at float16 as the Objective-C factories load them, against
    // transformers' `CLIPTextModel` and `CLIPTextModelWithProjection` at float16 with eager attention
    // (`clip_text_probe`, the float32 side on the same float16 weights), over one prompt padded to 77 tokens
    // by each tower's own tokenizer; bigG's pooled state and its projection as well.
    func testSDXLTextTowersInFloat16MatchTheFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_SDXL_DIR"], "IK_VAL_SDXL_DIR"))
        let towers: [(String, String, NFKMLXSDTextEncoderConfiguration)] = [
            ("sdxl-clip-l", "text_encoder", .sdxlPrimary), ("sdxl-clip-g", "text_encoder_2", .sdxlSecondary)]
        for (name, tower, configuration) in towers {
            let net = try NFKMLXSDTextEncoder.net(
                configuration: configuration,
                weightsURL: directory.appendingPathComponent("\(tower)/model.fp16.safetensors"), precision: .checkpoint)
            let records = name.replacingOccurrences(of: "-", with: "_")
            try clipText(name, net: net, half: try record("\(records)_f16.safetensors"),
                         f32: try record("\(records)_f32.safetensors"), dtype: .float16)
            Memory.clearCache()
        }
    }

    // A CLIP text tower at `dtype` against its `clip_text_probe` records: the embeddings, every layer, the
    // final norm, and the pooled end-of-text state (and a projecting tower's projection), then each piece of
    // the first and last layers on the reference's own input.
    private func clipText(_ name: String, net: NFKMLXSDTextEncoderNet, half: [String: MLXArray],
                          f32: [String: MLXArray], dtype: DType) throws {
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(half[key], "no \(key)").asType(dtype).expandedDimensions(axis: 0)
        }
        let tokenIds = try XCTUnwrap(half["tokens"]).asArray(Int32.self).map(Int.init)
        let length = tokenIds.count
        let mask = NFKCLIPInit.causalMask(length)
        let blocks = net.transformer.resblocks

        var hidden = net.tokenEmbedding(MLXArray(tokenIds.map(Int32.init)).reshaped([1, length]))
            + net.positionalEmbedding[0 ..< length]
        XCTAssertEqual(hidden.dtype, dtype, "a half-precision load embeds at its precision")
        var states = [("embeddings", hidden[0])], keys = ["text.hidden.0"]
        var isolated = [(String, MLXArray)](), isolatedKeys = [String]()
        for (index, block) in blocks.enumerated() {
            hidden = block(hidden, mask: mask)
            states.append(("layer \(index)", hidden[0]))
            keys.append("text.hidden.\(index + 1)")
            isolated.append(("layer \(index)", block(try recorded("text.hidden.\(index)"), mask: mask)[0]))
            isolatedKeys.append("text.hidden.\(index + 1)")
        }
        let final = net.lnFinal(hidden)
        let end = tokenIds.firstIndex(of: tokenIds.max() ?? 0) ?? (length - 1)
        states.append(("final norm", final[0]))
        keys.append("text.final")
        isolated.append(("final norm", net.lnFinal(try recorded("text.hidden.\(blocks.count)"))[0]))
        isolatedKeys.append("text.final")
        if let projection = net.textProjection {
            states += [("pooled", final[0, end]), ("projected", final[0, end].reshaped([1, -1]).matmul(projection)[0])]
            keys += ["pooled", "output"]
            isolated.append(("projection", try recorded("pooled").matmul(projection)[0]))
            isolatedKeys.append("output")
        } else {
            states.append(("pooled", final[0, end]))
            keys.append("output")
        }
        eval(states.map(\.1) + isolated.map(\.1))
        let rows = try seams(states, bf16: half, f32: f32, keys: keys)
        report(name, rows)
        let isolatedRows = try seams(isolated, bf16: half, f32: f32, keys: isolatedKeys)
        report("\(name) isolated", isolatedRows)
        assertRoundingPlacement(name, endToEnd: rows, isolated: isolatedRows)

        var lines = [String]()
        for index in [0, blocks.count - 1] {
            let block = blocks[index], p = "text.\(index)."
            func input(_ name: String) throws -> MLXArray { try recorded(p + name + ".in") }
            lines.append("layer \(index)")
            lines.append(try piece("layer_norm1", block.ln1(input("layer_norm1")), half, p + "layer_norm1.out"))
            lines.append(try piece("self_attn", block.attn(try recorded(p + "layer_norm1.out"), mask: mask)[0], half,
                                   p + "self_attn.out"))
            lines.append(try piece("out_proj", block.attn.outProj(input("self_attn.out_proj")), half, p + "self_attn.out_proj.out"))
            lines.append(try piece("layer_norm2", block.ln2(input("layer_norm2")), half, p + "layer_norm2.out"))
            lines.append(try piece("fc1", block.mlp.cFc(input("mlp.fc1")), half, p + "mlp.fc1.out"))
            lines.append(try piece("activation", block.mlp.activation(input("mlp.activation_fn")), half,
                                   p + "mlp.activation_fn.out"))
            lines.append(try piece("fc2", block.mlp.cProj(input("mlp.fc2")), half, p + "mlp.fc2.out"))
            lines.append(try piece("mlp", block.mlp(input("mlp")), half, p + "mlp.out"))
            lines.append(try piece("block", block(input("block"), mask: mask)[0], half, p + "block.out"))
        }
        print("VALIDATION \(dtype == .float16 ? "f16" : "bf16") \(name) pieces:\n" + lines.joined(separator: "\n"))
    }

    // MADLAD-400 3B (`IK_VAL_MADLAD`, released float32) at bf16 as its registered backend loads it, against
    // transformers' `T5ForConditionalGeneration` at bf16 (`t5_seq2seq_probe`, the float32 side on the same bf16
    // weights) over the `<2de>` source with the German target teacher-forced: every encoder and decoder block,
    // both final norms, and the logits, then each piece of the first and last decoder blocks on the
    // reference's own input.
    func testMADLAD3BInBFloat16MatchesTheBFloat16Reference() throws {
        try madladInBFloat16("madlad-3b", directoryKey: "IK_VAL_MADLAD", records: "madlad_3b")
    }

    // MADLAD-400 7B (`IK_VAL_MADLAD_7B`, released float32, 33 GB) the same way; its float32 side is streamed
    // from a bf16 load one module at a time (`IK_PROBE_STREAM_F32=1`).
    func testMADLAD7BInBFloat16MatchesTheBFloat16Reference() throws {
        try madladInBFloat16("madlad-7b", directoryKey: "IK_VAL_MADLAD_7B", records: "madlad_7b")
    }

    private func madladInBFloat16(_ name: String, directoryKey: String, records: String) throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config[directoryKey], directoryKey))
        let bf16 = try record("\(records)_bf16.safetensors"), f32 = try record("\(records)_f32.safetensors")
        let net = try NFKMLXMADLAD.translator(directoryURL: directory, half: true).net
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let source = try XCTUnwrap(bf16["source_ids"]).asType(.int32).reshaped([1, -1])
        let target = try XCTUnwrap(bf16["decoder_input"]).asType(.int32).reshaped([1, -1])
        let encoderBlocks = net.encoder.block, decoderBlocks = net.decoder.block

        var states = [(String, MLXArray)](), keys = [String]()
        var isolated = [(String, MLXArray)](), isolatedKeys = [String]()
        func measure(_ label: String, _ ours: MLXArray, _ key: String, alone: MLXArray) {
            states.append((label, ours))
            keys.append(key)
            isolated.append((label, alone))
            isolatedKeys.append(key)
        }
        var hidden = net.shared(source)
        XCTAssertEqual(hidden.dtype, .bfloat16, "a bf16 load embeds at bf16")
        states.append(("encoder input", hidden[0]))
        keys.append("enc.hidden.0")
        let encoderBias = encoderBlocks[0].selfAttention.attention.computeBias(source.dim(1))
        for (index, block) in encoderBlocks.enumerated() {
            hidden = block(hidden, bias: encoderBias)
            measure("encoder block \(index)", hidden[0], "enc.hidden.\(index + 1)",
                    alone: block(try recorded("enc.hidden.\(index)"), bias: encoderBias)[0])
        }
        let memory = net.encoder.finalLayerNorm(hidden)
        measure("encoder final norm", memory[0], "enc.final",
                alone: net.encoder.finalLayerNorm(try recorded("enc.hidden.\(encoderBlocks.count)"))[0])
        let recordedMemory = try recorded("enc.final")
        let decoderBias = decoderBlocks[0].selfAttention.attention.causalBias(queryLength: target.dim(1), offset: 0)
        hidden = net.shared(target)
        states.append(("decoder input", hidden[0]))
        keys.append("dec.hidden.0")
        let cache = net.makeCache(), aloneCache = net.makeCache()
        for (index, block) in decoderBlocks.enumerated() {
            hidden = block(hidden, memory: memory, bias: decoderBias, cache: cache, index: index)
            measure("decoder block \(index)", hidden[0], "dec.hidden.\(index + 1)",
                    alone: block(try recorded("dec.hidden.\(index)"), memory: recordedMemory, bias: decoderBias,
                                 cache: aloneCache, index: index)[0])
        }
        let final = net.decoder.finalLayerNorm(hidden)
        measure("decoder final norm", final[0], "dec.final",
                alone: net.decoder.finalLayerNorm(try recorded("dec.hidden.\(decoderBlocks.count)"))[0])
        let head = try XCTUnwrap(net.lmHead, "MADLAD's head is untied")
        measure("logits", head(final)[0], "output", alone: head(try recorded("dec.final"))[0])
        eval(states.map(\.1) + isolated.map(\.1))
        let rows = try seams(states, bf16: bf16, f32: f32, keys: keys)
        report(name, rows)
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolatedKeys)
        report("\(name) isolated", isolatedRows)
        assertRoundingPlacement(name, endToEnd: rows, isolated: isolatedRows)

        var lines = [String]()
        for index in [0, decoderBlocks.count - 1] {
            let block = decoderBlocks[index], p = "dec.\(index).layer."
            func input(_ name: String) throws -> MLXArray { try recorded(p + name + ".in") }
            let selfAttention = block.selfAttention.attention, crossAttention = block.crossAttention.attention
            let ff = block.feedForward.ff
            var keys: MLXArray? = nil, values: MLXArray? = nil
            var crossKeys: MLXArray? = nil, crossValues: MLXArray? = nil
            lines.append("decoder block \(index)")
            lines.append(try piece("self-attention norm", block.selfAttention.layerNorm(input("0.layer_norm")), bf16,
                                   p + "0.layer_norm.out"))
            for (name, projection) in [("q", selfAttention.q), ("k", selfAttention.k), ("v", selfAttention.v), ("o", selfAttention.o)] {
                lines.append(try piece("self-attention \(name)", projection(input("0.SelfAttention.\(name)")), bf16,
                                       p + "0.SelfAttention.\(name).out"))
            }
            lines.append(try piece("self-attention", selfAttention(input("0.SelfAttention"), bias: decoderBias,
                                                                   cachedKeys: &keys, cachedValues: &values),
                                   bf16, p + "0.SelfAttention.out"))
            lines.append(try piece("cross-attention norm", block.crossAttention.layerNorm(input("1.layer_norm")), bf16,
                                   p + "1.layer_norm.out"))
            for (name, projection) in [("q", crossAttention.q), ("k", crossAttention.k), ("v", crossAttention.v), ("o", crossAttention.o)] {
                lines.append(try piece("cross-attention \(name)", projection(input("1.EncDecAttention.\(name)")), bf16,
                                       p + "1.EncDecAttention.\(name).out"))
            }
            lines.append(try piece("cross-attention", crossAttention(input("1.EncDecAttention"), memory: recordedMemory,
                                                                     cachedKeys: &crossKeys, cachedValues: &crossValues),
                                   bf16, p + "1.EncDecAttention.out"))
            lines.append(try piece("feed-forward norm", block.feedForward.layerNorm(input("2.layer_norm")), bf16,
                                   p + "2.layer_norm.out"))
            lines.append(try piece("wi_0", ff.wi0(input("2.DenseReluDense.wi_0")), bf16, p + "2.DenseReluDense.wi_0.out"))
            lines.append(try piece("wi_1", ff.wi1(input("2.DenseReluDense.wi_1")), bf16, p + "2.DenseReluDense.wi_1.out"))
            lines.append(try piece("gelu_new", NFKReferenceRounding.geluNew(input("2.DenseReluDense.act")), bf16,
                                   p + "2.DenseReluDense.act.out"))
            lines.append(try piece("wo", ff.wo(input("2.DenseReluDense.wo")), bf16, p + "2.DenseReluDense.wo.out"))
            lines.append(try piece("feed-forward", ff(input("2.DenseReluDense")), bf16, p + "2.DenseReluDense.out"))
            lines.append(try piece("block", block(try recorded("dec.\(index).block.in"), memory: recordedMemory, bias: decoderBias,
                                                  cache: net.makeCache(), index: index), bf16, "dec.\(index).block.out"))
        }
        print("VALIDATION bf16 \(name) pieces:\n" + lines.joined(separator: "\n"))
    }

    // FLUX.1's autoencoder (`IK_VAL_FLUX_SCHNELL_FULL`'s `vae/`, released bf16) at bf16 as the FLUX pipeline
    // loads it, against diffusers' `AutoencoderKL` at bf16 (`sd_vae_probe`, the float32 side on the same bf16
    // weights) at 256 pixels.
    func testFluxAutoencoderInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_FLUX_SCHNELL_FULL"], "IK_VAL_FLUX_SCHNELL_FULL"))
        let vae = NFKMLXSDAutoencoder(configuration: .flux)
        try NFKMLXStableDiffusionModels.loadVAEWeights(
            into: vae, from: directory.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"), precision: .checkpoint)
        try autoencoder("flux-vae", vae, bf16: try record("flux_vae_bf16.safetensors"),
                        f32: try record("flux_vae_f32.safetensors"))
    }

    // FLUX.2 [klein]'s autoencoder and latent codec (`IK_VAL_FLUX2_KLEIN_4B_VAE`, released bf16) at bf16 as the
    // FLUX.2 pipeline loads them, against diffusers' `AutoencoderKLFlux2` at bf16 (`sd_vae_probe`, the float32
    // side on the same bf16 weights) at 256 pixels, with both quant convolutions, and the pipeline's batch-norm
    // whitening and its inverse over a seeded patched latent.
    func testFlux2AutoencoderInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_FLUX2_KLEIN_4B_VAE"], "IK_VAL_FLUX2_KLEIN_4B_VAE"))
        let vae = NFKMLXSDAutoencoder(configuration: .flux2)
        try NFKMLXStableDiffusionModels.loadVAEWeights(
            into: vae, from: directory.appendingPathComponent("diffusion_pytorch_model.safetensors"), precision: .checkpoint)
        let codec = try NFKMLXFlux2LatentCodec.codec(fromReleaseDirectory: directory, precision: .checkpoint)
        try autoencoder("flux2-vae", vae, bf16: try record("flux2_vae_bf16.safetensors"),
                        f32: try record("flux2_vae_f32.safetensors"),
                        extra: [("codec.whiten", { codec.whiten($0) }), ("codec.unwhiten", { codec.unwhiten($0) })])
    }

    // Stable Diffusion 1.5's autoencoder (`IK_SD15_DIR`'s `vae/`, released float32) at float16, cast as a
    // float16 release loads, against diffusers' `AutoencoderKL` at float16 (`sd_vae_probe`, the float32 side on
    // the same float16-rounded weights) at 256 pixels.
    func testSD15AutoencoderInFloat16MatchesTheFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_SD15_DIR"], "IK_SD15_DIR"))
        let vae = NFKMLXSDAutoencoder(configuration: NFKMLXSDTextToImageConfiguration.stableDiffusion15.vae)
        try NFKMLXStableDiffusionModels.loadVAEWeights(
            into: vae, from: directory.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"), precision: .float32)
        vae.update(parameters: vae.parameters().mapValues { $0.asType(.float16) })
        try autoencoder("sd15-vae", vae, bf16: try record("sd15_vae_f16.safetensors"),
                        f32: try record("sd15_vae_f32.safetensors"), dtype: .float16)
    }

    // An autoencoder at a half precision against its `sd_vae_probe` records: every decoder stage over a seeded latent and
    // every encoder stage over a plate, each `extra` seam on its `<prefix>.in` record, then each piece of the
    // decoder's first mid resnet and of its mid attention, on the reference's input.
    private func autoencoder(_ name: String, _ vae: NFKMLXSDAutoencoder, bf16: [String: MLXArray],
                             f32: [String: MLXArray], extra: [(String, (MLXArray) -> MLXArray)] = [],
                             dtype: DType = .bfloat16) throws {
        vae.train(false)
        /// A recorded channels-first map as the port's channels-last batch.
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(dtype).transposed(1, 2, 0).expandedDimensions(axis: 0)
        }
        func channelsFirst(_ x: MLXArray) -> MLXArray { x[0].transposed(2, 0, 1) }

        var states = [(String, MLXArray)](), keys = [String]()
        var isolated = [(String, MLXArray)](), isolatedKeys = [String]()
        func run(_ prefix: String, _ input: MLXArray, _ steps: [(String, (MLXArray) -> MLXArray)]) throws {
            var hidden = input, previous: MLXArray? = nil
            for (name, step) in steps {
                let key = "\(prefix).\(name)"
                hidden = step(hidden)
                states.append((key, channelsFirst(hidden)))
                keys.append(key)
                isolated.append((key, channelsFirst(step(previous ?? input))))
                isolatedKeys.append(key)
                previous = try recorded(key)
            }
        }
        let decoder = vae.decoder, encoder = vae.encoder
        var decoding: [(String, (MLXArray) -> MLXArray)] = vae.postQuantConv.map { conv in [("post_quant_conv", { conv($0) })] } ?? []
        decoding += [
            ("conv_in", { decoder.convIn($0) }), ("mid.resnet0", { decoder.midBlock.resnets[0]($0) }),
            ("mid.attention", { decoder.midBlock.attentions[0]($0) }), ("mid.resnet1", { decoder.midBlock.resnets[1]($0) })]
        for (index, block) in decoder.upBlocks.enumerated() {
            decoding += block.resnets.enumerated().map { j, resnet in ("up.\(index).resnet.\(j)", { resnet($0) }) }
            decoding += block.upsamplers.map { upsampler in ("up.\(index).upsample", { upsampler($0) }) }
        }
        decoding += [("norm_out", { decoder.normOut($0) }), ("act", { NFKReferenceRounding.silu($0) }),
                     ("out", { decoder.convOut($0) })]
        try run("dec", try recorded("latent"), decoding)
        var encoding: [(String, (MLXArray) -> MLXArray)] = [("conv_in", { encoder.convIn($0) })]
        for (index, block) in encoder.downBlocks.enumerated() {
            encoding += block.resnets.enumerated().map { j, resnet in ("down.\(index).resnet.\(j)", { resnet($0) }) }
            encoding += block.downsamplers.map { downsampler in ("down.\(index).downsample", { downsampler($0) }) }
        }
        encoding += [("mid.resnet0", { encoder.midBlock.resnets[0]($0) }), ("mid.attention", { encoder.midBlock.attentions[0]($0) }),
                     ("mid.resnet1", { encoder.midBlock.resnets[1]($0) }), ("norm_out", { encoder.normOut($0) }),
                     ("act", { NFKReferenceRounding.silu($0) }), ("out", { encoder.convOut($0) })]
        encoding += vae.quantConv.map { conv in [("quant_conv", { conv($0) })] } ?? []
        try run("enc", try recorded("image"), encoding)
        for (key, state) in extra {
            isolated.append((key, channelsFirst(state(try recorded(key.components(separatedBy: ".")[0] + ".in")))))
            isolatedKeys.append(key)
        }
        XCTAssertEqual(states[0].1.dtype, dtype, "a half-precision load decodes at its precision")
        eval(states.map(\.1) + isolated.map(\.1))
        let rows = try seams(states, bf16: bf16, f32: f32, keys: keys)
        report(name, rows)
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolatedKeys)
        report("\(name) isolated", isolatedRows)
        assertRoundingPlacement(name, endToEnd: rows, isolated: isolatedRows)

        let resnet = decoder.midBlock.resnets[0].block, attention = decoder.midBlock.attentions[0]
        func tokens(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(dtype) }
        let r = "dec.mid.resnet0.", a = "dec.mid.attention."
        let normedTokens = try tokens(a + "group_norm.in")                       // [channels, positions]
        let grouped = attention.groupNorm(normedTokens.transposed(1, 0).reshaped([1, 1, normedTokens.dim(1), normedTokens.dim(0)]))
        let lines = [
            try piece("resnet norm1", channelsFirst(resnet.norm1(recorded(r + "norm1.in"))), bf16, r + "norm1.out"),
            try piece("resnet conv1", channelsFirst(resnet.conv1(recorded(r + "conv1.in"))), bf16, r + "conv1.out"),
            try piece("resnet norm2", channelsFirst(resnet.norm2(recorded(r + "norm2.in"))), bf16, r + "norm2.out"),
            try piece("resnet conv2", channelsFirst(resnet.conv2(recorded(r + "conv2.in"))), bf16, r + "conv2.out"),
            try piece("attention group_norm", grouped.reshaped([normedTokens.dim(1), -1]).transposed(1, 0), bf16, a + "group_norm.out"),
            try piece("attention to_q", matmul(tokens(a + "to_q.in"), attention.toQ.weight.T) + attention.toQ.bias!, bf16, a + "to_q.out"),
            try piece("attention to_k", matmul(tokens(a + "to_k.in"), attention.toK.weight.T) + attention.toK.bias!, bf16, a + "to_k.out"),
            try piece("attention to_v", matmul(tokens(a + "to_v.in"), attention.toV.weight.T) + attention.toV.bias!, bf16, a + "to_v.out"),
            try piece("attention to_out", attention.toOut(tokens(a + "to_out.0.in")), bf16, a + "to_out.0.out"),
        ]
        print("VALIDATION \(dtype == .float16 ? "f16" : "bf16") \(name) pieces:\n" + lines.joined(separator: "\n"))
    }

    // The Gemma 3n E2B audio encoder (`IK_VAL_GEMMA3N_E2B`, released bf16) at bf16 against transformers'
    // `Gemma3nAudioEncoder` as `from_pretrained` loads it at bf16 (`gemma3n_audio_probe`, the float32 side on the
    // same bf16 weights), over a seeded mel: the front end, every Conformer block, and the output, then each
    // piece of the first and last blocks on the reference's own input.
    func testGemma3nAudioInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA3N_E2B"], "IK_VAL_GEMMA3N_E2B"))
        let bf16 = try record("gemma3n_audio_bf16.safetensors"), f32 = try record("gemma3n_audio_f32.safetensors")
        let net = NFKMLXGemma3nAudioNet(try NFKMLXGemma3nAudio.configuration(
            fromHuggingFace: directory.appendingPathComponent("config.json")))
        try NFKMLXGemma3nAudio.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let mel = try recorded("mel")
        let blocks = net.conformer

        var hidden = net.subsample(mel)
        XCTAssertEqual(hidden.dtype, .bfloat16, "a bf16 load computes the encoder at bf16")
        var states = [("front end", hidden[0])], keys = ["audio.hidden.0"]
        var isolated = [("front end", hidden[0])], isolatedKeys = ["audio.hidden.0"]
        for (index, block) in blocks.enumerated() {
            hidden = block(hidden, valid: nil)
            states.append(("block \(index)", hidden[0]))
            keys.append("audio.hidden.\(index + 1)")
            isolated.append(("block \(index)", block(try recorded("audio.hidden.\(index)"), valid: nil)[0]))
            isolatedKeys.append("audio.hidden.\(index + 1)")
        }
        let (encoded, _) = net(mel)
        XCTAssertEqual(encoded.dtype, .bfloat16, "the encoder's output stays bf16")
        states.append(("output", encoded[0]))
        keys.append("output")
        eval(states.map(\.1) + isolated.map(\.1))
        let rows = try seams(states, bf16: bf16, f32: f32, keys: keys)
        report("gemma3n-audio", rows)
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolatedKeys)
        report("gemma3n-audio isolated", isolatedRows)
        assertRoundingPlacement("gemma3n-audio", endToEnd: rows, isolated: isolatedRows)

        var lines = [String]()
        for index in [0, blocks.count - 1] {
            let block = blocks[index], p = "audio.\(index)."
            func input(_ name: String) throws -> MLXArray { try recorded(p + name + ".in") }
            let start = block.feedForwardStart, attention = block.attention, light = block.convolution
            lines.append("block \(index)")
            lines.append(try piece("ffw_start", start(input("ffw_layer_start")), bf16, p + "ffw_layer_start.out"))
            lines.append(try piece("ffw_start pre norm", start.preNorm(input("ffw_layer_start.pre_layer_norm")), bf16,
                                   p + "ffw_layer_start.pre_layer_norm.out"))
            lines.append(try piece("ffw_start layer 1", start.expand(input("ffw_layer_start.ffw_layer_1")), bf16,
                                   p + "ffw_layer_start.ffw_layer_1.out"))
            lines.append(try piece("ffw_start layer 2", start.contract(input("ffw_layer_start.ffw_layer_2")), bf16,
                                   p + "ffw_layer_start.ffw_layer_2.out"))
            lines.append(try piece("attention", attention(input("attention"), valid: nil), bf16, p + "attention.out"))
            lines.append(try piece("attention core", attention.attention(input("attention.attn"), valid: nil), bf16,
                                   p + "attention.attn.out"))
            lines.append(try piece("q_proj", attention.attention.queryProjection(input("attention.attn.q_proj")), bf16,
                                   p + "attention.attn.q_proj.out"))
            let timing = attention.attention.relativePosition.timing(dtype: .bfloat16)
            lines.append(try piece("position sinusoids", timing, bf16, p + "attention.attn.relative_position_embedding.pos_proj.in"))
            lines.append(try piece("pos_proj", attention.attention.relativePosition.positionProjection(
                input("attention.attn.relative_position_embedding.pos_proj")), bf16,
                p + "attention.attn.relative_position_embedding.pos_proj.out"))
            lines.append(try piece("attention post", attention.post(input("attention.post")), bf16, p + "attention.post.out"))
            lines.append(try piece("lconv1d", light(input("lconv1d")), bf16, p + "lconv1d.out"))
            lines.append(try piece("linear_start", light.start(input("lconv1d.linear_start")), bf16, p + "lconv1d.linear_start.out"))
            lines.append(try piece("depthwise", light.depthwise(try XCTUnwrap(bf16[p + "lconv1d.depthwise_conv1d.in"], "no depthwise input")
                .asType(.bfloat16).transposed(1, 0).expandedDimensions(axis: 0)).transposed(0, 2, 1), bf16,
                p + "lconv1d.depthwise_conv1d.out"))
            lines.append(try piece("conv_norm", light.convNorm(input("lconv1d.conv_norm")), bf16, p + "lconv1d.conv_norm.out"))
            lines.append(try piece("linear_end", light.end(input("lconv1d.linear_end")), bf16, p + "lconv1d.linear_end.out"))
            lines.append(try piece("ffw_end", block.feedForwardEnd(input("ffw_layer_end")), bf16, p + "ffw_layer_end.out"))
            lines.append(try piece("norm", block.norm(input("norm")), bf16, p + "norm.out"))
        }
        print("VALIDATION bf16 gemma3n-audio pieces:\n" + lines.joined(separator: "\n"))
    }

    // SDXL-Turbo's UNet (`IK_VAL_SDXL_DIR`'s `unet/`, released float16) at float16 as the Objective-C factories
    // load it, against diffusers' `UNet2DConditionModel` at float16 (`sd_unet_probe`) over a seeded 32×32 latent.
    func testSDXLTurboUNetInFloat16MatchesTheFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_SDXL_DIR"], "IK_VAL_SDXL_DIR"))
        let half = try record("sdxl_turbo_unet_f16.safetensors")
        let unet = NFKMLXSDUNet(configuration: NFKMLXSDTextToImageConfiguration.sdxlTurbo.unet)
        try NFKMLXStableDiffusionModels.loadUNetWeights(
            into: unet, from: directory.appendingPathComponent("unet/diffusion_pytorch_model.fp16.safetensors"),
            precision: .checkpoint)
        let added = NFKSDAddedConditioning(
            pooled: try XCTUnwrap(half["pooled"]).asType(.float16).expandedDimensions(axis: 0),
            timeIds: try XCTUnwrap(half["time_ids"]).expandedDimensions(axis: 0))
        try unetInFloat16("sdxl-turbo-unet", unet, half: half, f32: try record("sdxl_turbo_unet_f32.safetensors"),
                          added: added)
    }

    // Stable Diffusion 1.5's UNet (`IK_SD15_DIR`'s `unet/`, released float32) at float16, cast as a float16
    // release loads, with the IP-Adapter (`IK_VAL_IPADAPTER`, released float16) attached at scale 0.6, against
    // diffusers' `UNet2DConditionModel` at float16 with the adapter loaded at its type (`sd_unet_probe`
    // `IK_PROBE_IP_ADAPTER`; the float32 side on the same float16-rounded weights): the image projection and
    // every stage, each cross-attention reading the image tokens. Each transformer block is held to its order
    // floor (`IK_PROBE_ORDER_FLOOR`): MLX's half-precision GPU matmul splits its inner dimension into 1, 2, 4,
    // or 8 parts, each a running float32 sum, which alone takes a block at the first level's 1,024 positions to
    // 0.3 of the accumulated floor.
    func testSD15UNetWithIPAdapterInFloat16MatchesTheFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_SD15_DIR"], "IK_SD15_DIR"))
        let adapterURL = URL(fileURLWithPath: try existing(config["IK_VAL_IPADAPTER"], "IK_VAL_IPADAPTER"))
        var configuration = NFKMLXSDTextToImageConfiguration.stableDiffusion15.unet
        configuration.inputChannels = 4
        let unet = NFKMLXSDUNet(configuration: configuration)
        try NFKMLXStableDiffusionModels.loadUNetWeights(
            into: unet, from: directory.appendingPathComponent("unet/diffusion_pytorch_model.safetensors"), precision: .float32)
        unet.update(parameters: unet.parameters().mapValues { $0.asType(.float16) })
        let adapter = try NFKMLXIPAdapter.load(from: adapterURL, into: unet)
        try unetInFloat16("sd15-unet-ip", unet, half: try record("sd15_unet_ip_f16.safetensors"),
                          f32: try record("sd15_unet_ip_f32.safetensors"), adapter: adapter)
    }

    // Stable Diffusion 2.1's UNet (`IK_VAL_SD21_DIR`'s `unet/`, released float32) at float16, cast as a float16
    // release loads, against diffusers' `UNet2DConditionModel` at float16 (`sd_unet_probe`). Its
    // `upcast_attention` reaches only diffusers' classic attention processor, not the scaled-dot-product one.
    func testSD21UNetInFloat16MatchesTheFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_SD21_DIR"], "IK_VAL_SD21_DIR"))
        let unet = NFKMLXSDUNet(configuration: NFKMLXSDTextToImageConfiguration.stableDiffusion21.unet)
        try NFKMLXStableDiffusionModels.loadUNetWeights(
            into: unet, from: directory.appendingPathComponent("unet/diffusion_pytorch_model.safetensors"), precision: .float32)
        unet.update(parameters: unet.parameters().mapValues { $0.asType(.float16) })
        try unetInFloat16("sd21-unet", unet, half: try record("sd21_unet_f16.safetensors"),
                          f32: try record("sd21_unet_f32.safetensors"))
    }

    // A Stable Diffusion UNet at float16 against its `sd_unet_probe` records: every resnet, sampler, and
    // transformer block, with each transformer's norm and projections, on the reference's own input, the noise
    // end to end, then each piece of the first transformer block.
    private func unetInFloat16(_ name: String, _ unet: NFKMLXSDUNet, half: [String: MLXArray], f32: [String: MLXArray],
                               added: NFKSDAddedConditioning? = nil, adapter: NFKMLXIPAdapter? = nil) throws {
        unet.train(false)
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(half[key], "no \(key)").asType(.float16).expandedDimensions(axis: 0)
        }
        let time = try recorded("temb"), context = try recorded("context")
        let ip = try adapter.map { _ in NFKSDImageConditioning(tokens: try recorded("ip.tokens"), scale: 0.6) }

        var isolated = [(String, MLXArray)](), keys = [String]()
        func stage(_ name: String, _ run: (MLXArray) -> MLXArray) throws {
            isolated.append((name, run(try recorded(name + ".in"))[0]))
            keys.append(name + ".out")
        }
        func transformer(_ name: String, _ t: NFKSDTransformer2D) throws {
            try stage(name + ".norm") { t.norm($0) }
            try stage(name + ".proj_in") { x in t.projInLinear.map { t.projectedIn(x, $0) } ?? t.projInConv!(x) }
            for (k, block) in t.blocks.enumerated() {
                try stage(name + ".transformer_blocks.\(k)") { block($0, context: context, ip: ip) }
            }
            try stage(name + ".proj_out") { x in t.projOutLinear.map { $0(x) } ?? t.projOutConv!(x) }
        }
        try stage("conv_in") { unet.convIn($0) }
        for (i, block) in unet.downBlocks.enumerated() {
            for (j, resnet) in block.resnets.enumerated() {
                try stage("down.\(i).resnets.\(j)") { resnet($0, time: time) }
                if j < block.attentions.count { try transformer("down.\(i).attentions.\(j)", block.attentions[j]) }
            }
            for sampler in block.downsamplers { try stage("down.\(i).sampler") { sampler($0) } }
        }
        let mid = unet.midBlock
        try stage("mid.resnets.0") { mid.resnets[0]($0, time: time) }
        try transformer("mid.attentions.0", mid.attentions[0])
        try stage("mid.resnets.1") { mid.resnets[1]($0, time: time) }
        for (i, block) in unet.upBlocks.enumerated() {
            for (j, resnet) in block.resnets.enumerated() {
                try stage("up.\(i).resnets.\(j)") { resnet($0, time: time) }
                if j < block.attentions.count { try transformer("up.\(i).attentions.\(j)", block.attentions[j]) }
            }
            for sampler in block.upsamplers { try stage("up.\(i).sampler") { sampler($0) } }
        }
        try stage("conv_norm_out") { unet.normOut($0) }
        try stage("conv_act") { NFKReferenceRounding.silu($0) }
        try stage("conv_out") { unet.convOut($0) }

        var conditioning: NFKSDImageConditioning?
        if let adapter {
            let embeds = try XCTUnwrap(half["image_embeds"]).asType(.float16).reshaped([1, -1])
            isolated.append(("ip.tokens", adapter.imageProjection(embeds)[0]))
            keys.append("ip.tokens")
            conditioning = adapter.conditioning(imageEmbedding: embeds, scale: 0.6)
        }
        let noise = unet(try recorded("latent"), timestep: try XCTUnwrap(half["timestep"]), context: context, added: added,
                         imageConditioning: conditioning)
        XCTAssertEqual(noise.dtype, .float16, "a float16 load runs the UNet at float16")
        let states = [("noise", noise[0])]
        eval(isolated.map(\.1) + states.map(\.1))
        let rows = try seams(states, bf16: half, f32: f32, keys: ["output"])
        report("\(name) f16", rows)
        let isolatedRows = try seams(isolated, bf16: half, f32: f32, keys: keys)
        report("\(name) f16 isolated", isolatedRows)
        assertRoundingPlacement("\(name) f16", endToEnd: rows, isolated: isolatedRows)

        let block = try XCTUnwrap(unet.downBlocks.first { !$0.attentions.isEmpty }).attentions[0].blocks[0]
        func input(_ name: String) throws -> MLXArray { try recorded("piece." + name + ".in") }
        let lines = [
            try piece("norm1", block.norm1(input("norm1")), half, "piece.norm1.out"),
            try piece("to_q", block.attn1.toQ(input("attn1.to_q")), half, "piece.attn1.to_q.out"),
            try piece("self-attention", block.attn1(input("attn1")), half, "piece.attn1.out"),
            try piece("to_out", block.attn1.toOut(input("attn1.to_out.0")), half, "piece.attn1.to_out.0.out"),
            try piece("norm2", block.norm2(input("norm2")), half, "piece.norm2.out"),
            try piece("cross-attention", block.attn2(input("attn2"), context: context, ip: ip), half, "piece.attn2.out"),
            try piece("norm3", block.norm3(input("norm3")), half, "piece.norm3.out"),
            try piece("geglu proj", block.ff.proj(input("ff.net.0.proj")), half, "piece.ff.net.0.proj.out"),
            try piece("feed-forward", block.ff(input("ff")), half, "piece.ff.out"),
            try piece("ff out", block.ff.out(input("ff.net.2")), half, "piece.ff.net.2.out"),
        ]
        print("VALIDATION f16 \(name) pieces:\n" + lines.joined(separator: "\n"))
    }

    // Whisper tiny from OpenAI's float16 `.pt` (`IK_RAW_WHISPER`) loaded as published, against transformers'
    // `WhisperForConditionalGeneration` built from the same file at float16 with eager attention
    // (`whisper_f16_probe`, the float32 side on the same float16 weights) over a seeded log-mel and a
    // teacher-forced prompt: both convolutions, every encoder and decoder layer, both final norms, and the
    // logits, then each piece of the first and last layers of each stack.
    func testWhisperTinyInFloat16MatchesTheFloat16Reference() throws {
        try requireMLXRuntime()
        let weights = URL(fileURLWithPath: try existing(config["IK_RAW_WHISPER"], "IK_RAW_WHISPER"))
        let half = try record("whisper_tiny_f16.safetensors"), f32 = try record("whisper_tiny_f32.safetensors")
        let net = NFKMLXWhisperNet(.tiny)
        try NFKMLXWhisper.loadWeights(into: net, from: weights, precision: .checkpoint)
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(half[key], "no \(key)").asType(.float16).expandedDimensions(axis: 0)
        }
        let mel = try XCTUnwrap(half["mel"]).expandedDimensions(axis: 0)
        let tokens = try XCTUnwrap(half["tokens"]).reshaped([1, -1])
        let encoder = net.encoder, decoder = net.decoder

        var states = [(String, MLXArray)](), keys = [String]()
        var isolated = [(String, MLXArray)](), isolatedKeys = [String]()
        func measure(_ label: String, _ ours: MLXArray, _ key: String, alone: MLXArray) {
            states.append((label, ours))
            keys.append(key)
            isolated.append((label, alone)); isolatedKeys.append(key)
        }
        let typedMel = net.typed(mel)
        XCTAssertEqual(typedMel.dtype, .float16, "a float16 file computes at float16")
        let conv1 = encoder.conv1(typedMel)
        measure("conv1", conv1[0], "enc.conv1", alone: conv1[0])
        let conv2 = encoder.conv2(NFKReferenceRounding.wide(conv1) { gelu($0) })
        measure("conv2", conv2[0], "enc.conv2",
                alone: encoder.conv2(NFKReferenceRounding.wide(try recorded("enc.conv1")) { gelu($0) })[0])
        var audio = (NFKReferenceRounding.wide(conv2) { gelu($0) } + NFKMLXWhisperNet.sinusoids(length: conv2.shape[1],
            channels: conv2.shape[2])).asType(conv2.dtype)
        states.append(("encoder input", audio[0]))
        keys.append("enc.hidden.0")
        for (index, block) in encoder.blocks.enumerated() {
            audio = block(audio, audio: nil, mask: nil)
            measure("encoder layer \(index)", audio[0], "enc.hidden.\(index + 1)",
                    alone: block(try recorded("enc.hidden.\(index)"), audio: nil, mask: nil)[0])
        }
        let memory = encoder.lnPost(audio)
        measure("encoder final norm", memory[0], "enc.final",
                alone: encoder.lnPost(try recorded("enc.hidden.\(encoder.blocks.count)"))[0])
        let recordedMemory = try recorded("enc.final")
        let length = tokens.dim(1)
        let mask = NFKMLXWhisperNet.causalMask(length)
        var hidden = decoder.tokenEmbedding(tokens) + decoder.positionalEmbedding[0 ..< length]
        states.append(("decoder input", hidden[0]))
        keys.append("dec.hidden.0")
        for (index, block) in decoder.blocks.enumerated() {
            hidden = block(hidden, audio: memory, mask: mask)
            measure("decoder layer \(index)", hidden[0], "dec.hidden.\(index + 1)",
                    alone: block(try recorded("dec.hidden.\(index)"), audio: recordedMemory, mask: mask)[0])
        }
        let final = decoder.ln(hidden)
        measure("decoder final norm", final[0], "dec.final",
                alone: decoder.ln(try recorded("dec.hidden.\(decoder.blocks.count)"))[0])
        let embedding = decoder.tokenEmbedding.weight.transposed(1, 0)
        measure("logits", final.matmul(embedding)[0], "output", alone: try recorded("dec.final").matmul(embedding)[0])
        eval(states.map(\.1) + isolated.map(\.1))
        let rows = try seams(states, bf16: half, f32: f32, keys: keys)
        report("whisper-tiny f16", rows)
        let isolatedRows = try seams(isolated, bf16: half, f32: f32, keys: isolatedKeys)
        report("whisper-tiny f16 isolated", isolatedRows)
        assertRoundingPlacement("whisper-tiny f16", endToEnd: rows, isolated: isolatedRows)

        var lines = [String]()
        for (stack, blocks) in [("enc", encoder.blocks), ("dec", decoder.blocks)] {
            for index in [0, blocks.count - 1] {
                let block = blocks[index], p = "\(stack).\(index)."
                func input(_ name: String) throws -> MLXArray { try recorded(p + name + ".in") }
                lines.append("\(stack) layer \(index)")
                lines.append(try piece("self-attention norm", block.attnLN(input("self_attn_layer_norm")), half,
                                       p + "self_attn_layer_norm.out"))
                lines.append(try piece("q_proj", block.attn.query(input("self_attn.q_proj")), half, p + "self_attn.q_proj.out"))
                lines.append(try piece("out_proj", block.attn.out(input("self_attn.out_proj")), half, p + "self_attn.out_proj.out"))
                if let crossLN = block.crossAttnLN {
                    lines.append(try piece("cross-attention norm", crossLN(input("encoder_attn_layer_norm")), half,
                                           p + "encoder_attn_layer_norm.out"))
                }
                lines.append(try piece("mlp norm", block.mlpLN(input("final_layer_norm")), half, p + "final_layer_norm.out"))
                lines.append(try piece("fc1", (block.mlp[0] as! Linear)(input("fc1")), half, p + "fc1.out"))
                lines.append(try piece("fc2", (block.mlp[2] as! Linear)(input("fc2")), half, p + "fc2.out"))
            }
        }
        print("VALIDATION f16 whisper-tiny pieces:\n" + lines.joined(separator: "\n"))
    }

    // MARK: Gemma 4

    // The E2B decoder at the released bf16 against transformers built at bf16 (`hf_layer_probe`,
    // `IK_PROBE_DTYPE=bfloat16 IK_PROBE_LAYERS=0,4,15`), with `IK_PARITY_GEMMA4` as the float32 side:
    // the same prompt, every hidden state, the logits. A sharing layer (15 and on) reads the keys and
    // values its donor made from the reference's own input, so each layer is still measured alone.
    func testGemma4E2BInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA4"], "IK_VAL_GEMMA4"))
        let bf16 = try record("gemma4_e2b_bf16_probe.safetensors")
        let f32 = try loadArrays(url: URL(fileURLWithPath: try existing(config["IK_PARITY_GEMMA4"], "IK_PARITY_GEMMA4")))
        let tokens = try XCTUnwrap(bf16["tokens"]).asArray(Int32.self)
        XCTAssertEqual(tokens, try XCTUnwrap(f32["tokens"]).asArray(Int32.self), "one prompt, two precisions")

        let geometry = try NFKMLXGemmaLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXGemmaLanguage.makeNet(geometry)
        try NFKMLXGemmaLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        let input = MLXArray(tokens).reshaped([1, tokens.count])
        let states = net.hiddenStates(input)
        let logits = net(input)[0]
        eval(states + [logits])
        let labelled = states.enumerated().map { ($0.offset == 0 ? "embedding" : "layer \($0.offset - 1)", $0.element) }
            + [("logits", logits)]
        let rows = try seams(labelled, bf16: bf16, f32: f32, keys: states.indices.map { "hidden.\($0)" } + ["output"])
        report("gemma4-e2b", rows)

        let perLayer = net.perLayerInputs(embeddings: net.embed(input), tokens: input)
        let mask = NFKMLXLanguageNet.causalMask(tokens.count, offset: 0)
        var latest = [NFKMLXGemmaAttentionKind: (keys: MLXArray, values: MLXArray)]()
        var isolated = [(String, MLXArray)]()
        for (index, layer) in net.layers.enumerated() {
            let kind = geometry.layerTypes[index]
            let hidden = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            var (output, keys, values) = layer(hidden, perLayerInput: perLayer?[0..., 0..., index], mask: mask,
                                               shared: index >= geometry.firstSharedLayer ? latest[kind] : nil)
            if index < geometry.firstSharedLayer { latest[kind] = (keys, values) }
            if index == net.layers.count - 1 { output = net.norm(output) }
            isolated.append(("layer \(index)", output))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolated.indices.map { "hidden.\($0 + 1)" })
        report("gemma4-e2b isolated", isolatedRows)

        var lines = [String]()
        for index in [0, 4, 15] where bf16["\(index).block.in"] != nil {
            let block = net.layers[index], attention = block.attention
            let p = "\(index)."
            func probe(_ key: String) throws -> MLXArray {
                try XCTUnwrap(bf16[p + key], "no \(p + key)").asType(.bfloat16).expandedDimensions(axis: 0)
            }
            lines.append("layer \(index) (\(geometry.layerTypes[index]))")
            lines.append(try piece("input_layernorm", block.inputNorm(probe("input_layernorm.in")), bf16, p + "input_layernorm.out"))
            lines.append(try piece("q_norm", attention.queryNorm(probe("self_attn.q_norm.in")), bf16, p + "self_attn.q_norm.out"))
            if bf16[p + "self_attn.v_norm.in"] != nil {
                lines.append(try piece("v_norm", NFKReferenceRounding.scaledNorm(probe("self_attn.v_norm.in"), weight: nil,
                                                                                 eps: attention.valueEpsilon),
                                       bf16, p + "self_attn.v_norm.out"))
            }
            lines.append(try piece("attention", NFKReferenceRounding.attention(
                queries: probe("attn.q"), keys: probe("attn.k"), values: probe("attn.v"), scale: 1,
                mask: mask.asType(.bfloat16))[0].transposed(1, 0, 2), bf16, p + "attn.out"))
            lines.append(try piece("mlp", block.feedForward(probe("mlp.in")), bf16, p + "mlp.out"))
            lines.append(try piece("post_feedforward_norm", block.postFeedForwardNorm(probe("post_feedforward_layernorm.in")),
                                   bf16, p + "post_feedforward_layernorm.out"))
            if let gate = block.perLayerGate, bf16[p + "per_layer_input_gate.in"] != nil {
                lines.append(try piece("per_layer_input_gate", gate(probe("per_layer_input_gate.in")), bf16,
                                       p + "per_layer_input_gate.out"))
            }
            if let projection = block.perLayerProjection, bf16[p + "per_layer_projection.in"] != nil {
                lines.append(try piece("per_layer_projection", projection(probe("per_layer_projection.in")), bf16,
                                       p + "per_layer_projection.out"))
            }
        }
        print("VALIDATION bf16 gemma4-e2b pieces:\n" + lines.joined(separator: "\n"))
        assertRoundingPlacement("gemma4-e2b", endToEnd: rows, isolated: isolatedRows)
    }

    // The Gemma 4 E2B vision tower and its embedder at the released bf16 against transformers at bf16 with
    // eager attention (`gemma4_vision_probe`, a 12×12 patch grid pooled to 16 soft tokens): every layer's
    // state, the pooled soft tokens, and their projection.
    func testGemma4VisionInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let release = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA4"], "IK_VAL_GEMMA4"))
        let bf16 = try record("gemma4_e2b_vision_bf16.safetensors"), f32 = try record("gemma4_e2b_vision_f32.safetensors")
        let vision = NFKMLXGemmaLanguage.makeVisionNet(
            NFKMLXGemma4VisionConfiguration(hiddenSize: 768, layerCount: 16, headCount: 12,
                                            keyValueHeadCount: 12, headDimensions: 64, intermediateSize: 3072,
                                            patchSize: 16, positionEmbeddingSize: 10240, poolingKernelSize: 3,
                                            useClippedLinears: true))
        let embedder = NFKMLXGemma4MultimodalEmbedder(multimodalHidden: 768, textHidden: 1536)
        try NFKMLXWeights.apply(NFKMLXReleaseWeights.materializedArrays(inDirectory: release, precision: .checkpoint, remap: { key in
            guard key.hasPrefix("model.vision_tower.") else { return nil }
            return String(key.dropFirst("model.vision_tower.".count))
                .replacingOccurrences(of: "encoder.layers.", with: "encoder_layers.")
        }), to: vision)
        try NFKMLXWeights.apply(NFKMLXReleaseWeights.materializedArrays(inDirectory: release, precision: .checkpoint, remap: { key in
            key.hasPrefix("model.embed_vision.") ? String(key.dropFirst("model.embed_vision.".count)) : nil
        }), to: embedder)
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let pixels = try XCTUnwrap(bf16["pixel_values"]).expandedDimensions(axis: 0)
        let positions = try XCTUnwrap(bf16["position_ids"]).expandedDimensions(axis: 0).asType(.int32)
        let (cosine, sine) = vision.rope(positionIds: positions)

        var hidden = vision.patchEmbedder(pixels, positionIds: positions)
        var states = [("patch embedding", hidden[0])]
        var isolated = [(String, MLXArray)]()
        for (index, layer) in vision.layers.enumerated() {
            hidden = layer(hidden, cosine: cosine, sine: sine)
            states.append(("layer \(index)", hidden[0]))
            isolated.append(("layer \(index)", layer(try recorded("vit.hidden.\(index)"), cosine: cosine, sine: sine)[0]))
        }
        let soft = vision.softTokens(pixels, positionIds: positions)
        XCTAssertEqual(soft.dtype, .bfloat16, "a bf16 load computes the image path in bf16")
        states += [("pooled", soft[0]), ("projected", embedder(soft)[0])]
        isolated.append(("projected", embedder(try recorded("pooled"))[0]))
        eval(states.map(\.1) + isolated.map(\.1))
        let count = vision.layers.count
        let rows = try seams(states, bf16: bf16, f32: f32,
                             keys: ["vit.patch"] + (1 ... count).map { "vit.hidden.\($0)" } + ["pooled", "output"])
        report("gemma4-e2b-vision", rows)
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32,
                                     keys: (1 ... count).map { "vit.hidden.\($0)" } + ["output"])
        report("gemma4-e2b-vision isolated", isolatedRows)
        assertRoundingPlacement("gemma4-e2b-vision", endToEnd: rows, isolated: isolatedRows)
    }

    // MARK: Gemma 3n

    // The Gemma 3n E2B vision tower (timm's MobileNetV5) and its embedder at the released bf16
    // (`.checkpoint`) against timm and transformers at bf16 (`gemma3n_vision_probe`), on a seeded frame:
    // every block, the fusion adapter, the scaled soft tokens, and their projection.
    func testGemma3nVisionInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let release = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA3N_E2B"], "IK_VAL_GEMMA3N_E2B"))
        let bf16 = try record("gemma3n_e2b_vision_bf16.safetensors"), f32 = try record("gemma3n_e2b_vision_f32.safetensors")
        let net = NFKMLXGemma3nVisionNet()
        try NFKMLXGemma3nVision.loadWeights(into: net, fromDirectory: release, precision: .checkpoint)
        let embedder = NFKGemma3nMultimodalEmbedder(hiddenSize: 2048, textHiddenSize: 2048, vocabularySize: 128,
                                                    vocabularyOffset: 262_144)
        try NFKMLXWeights.apply(NFKMLXReleaseWeights.materializedArrays(inDirectory: release, precision: .checkpoint, remap: { key in
            key.hasPrefix("model.embed_vision.") ? String(key.dropFirst("model.embed_vision.".count)) : nil
        }), to: embedder)
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let pixels = try XCTUnwrap(bf16["pixels"]).expandedDimensions(axis: 0)

        var hidden = net.stem(pixels.asType(.bfloat16))
        var states = [("stem", hidden[0])], keys = ["stem"]
        var isolated = [(String, MLXArray)](), isolatedKeys = [String]()
        var captured = [MLXArray](), recordedStages = [MLXArray]()
        for (s, stage) in net.blocks.enumerated() {
            for (b, block) in stage.enumerated() {
                hidden = block(hidden)
                let key = "block.\(s).\(b).out"
                states.append(("block \(s).\(b)", hidden[0]))
                keys.append(key)
                isolated.append(("block \(s).\(b)", block(try recorded("block.\(s).\(b).in"))[0]))
                isolatedKeys.append(key)
            }
            if s >= net.blocks.count - 2 {
                captured.append(hidden)
                recordedStages.append(try recorded("block.\(s).\(stage.count - 1).out"))
            }
        }
        let fused = net.fusion(captured)
        let width = fused.dim(3)
        let tokens = NFKReferenceRounding.scaled(fused.reshaped([1, -1, width]), by: Float(Double(width).squareRoot()))
        XCTAssertEqual(tokens.dtype, .bfloat16, "a bf16 load computes the image path in bf16")
        states += [("fused", fused[0]), ("tokens", tokens[0]), ("projected", embedder(soft: tokens)[0])]
        keys += ["fused", "tokens", "output"]
        isolated += [("fused", net.fusion(recordedStages)[0]), ("projected", embedder(soft: try recorded("tokens"))[0])]
        isolatedKeys += ["fused", "output"]
        eval(states.map(\.1) + isolated.map(\.1))
        let rows = try seams(states, bf16: bf16, f32: f32, keys: keys)
        report("gemma3n-e2b-vision", rows)
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolatedKeys)
        report("gemma3n-e2b-vision isolated", isolatedRows)
        assertRoundingPlacement("gemma3n-e2b-vision", endToEnd: rows, isolated: isolatedRows)
    }


    // The E2B decoder at the released bf16 against transformers at bf16 with eager attention
    // (`IK_GEMMA_DTYPE=bfloat16 run_reference.py gemma3n`), `IK_PARITY_GEMMA3N_E2B` the float32 side.
    // The record keeps only the active AltUp copy, so a later layer cannot be run alone on the
    // reference's input; layer 0 reads the exact embedding and is the isolated measurement.
    func testGemma3nE2BInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA3N_E2B"], "IK_VAL_GEMMA3N_E2B"))
        try gemma3n("gemma3n-e2b", directory: directory, bf16: record("gemma3n_e2b_bf16_eager.safetensors"),
                    f32: loadArrays(url: URL(fileURLWithPath: try existing(config["IK_PARITY_GEMMA3N_E2B"], "IK_PARITY_GEMMA3N_E2B"))))
    }

    // The E4B the same way, at 16 GB of bf16. Its float32 side does not fit this machine, so the
    // reference's float32 run is streamed from a bf16 load (`hf_layer_probe` with `IK_PROBE_STREAM_F32=1`),
    // and both runs probe every layer (`IK_PROBE_LAYERS=0,…,34`) so each block runs alone on the reference's
    // four AltUp copies, each held to a quarter of its floor. A one-step difference anywhere in its 35
    // layers and four copies reaches the logits many times over, so one run of either side is one draw:
    // the reference's own bf16 under one-step nudges at every layer input (`hf_bf16_spread` with
    // `IK_SPREAD_TARGETS`) lands anywhere from 5.6e-5 to 4.5e-4 from float32. So this port runs under the
    // same nudges too, and its median distance from float32 is held to twice the reference's median.
    func testGemma3nE4BInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA3N_E4B"], "IK_VAL_GEMMA3N_E4B"))
        try gemma3n("gemma3n-e4b", directory: directory, bf16: record("gemma3n_e4b_bf16_probeall.safetensors"),
                    f32: record("gemma3n_e4b_f32_probeall.safetensors"),
                    spread: record("gemma3n_e4b_spread_every32b.safetensors"))
    }

    /// A released Gemma 3n decoder at its bf16 against the reference's bf16 record, every state and the
    /// logits, with `f32` the reference's float32 run on the same prompt. `spread` holds the reference's
    /// own bf16 logits under one-step nudges at every layer input (`spread.S`, with `spread.S.reached` the
    /// distance each input was moved); where it is given, the logits are compared as distributions: this
    /// port runs under nudges of those sizes, and its median distance from float32 is held to twice the
    /// reference's median.
    private func gemma3n(_ name: String, directory: URL, bf16: [String: MLXArray], f32: [String: MLXArray],
                         spread: [String: MLXArray]? = nil) throws {
        let tokens = try XCTUnwrap(bf16["tokens"]).asArray(Int32.self)
        XCTAssertEqual(tokens, try XCTUnwrap(f32["tokens"]).asArray(Int32.self), "one prompt, two precisions")

        let configuration = try NFKMLXGemma3nLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXGemma3nNet(configuration)
        try NFKMLXGemma3nLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        let input = MLXArray(tokens).reshaped([1, tokens.count])
        let states = net.layerStates(input)
        let logits = net(input)[0]
        eval(states + [logits])
        for (index, state) in states.enumerated() {
            XCTAssertEqual(state.dtype, .bfloat16, "state \(index) stays in the released precision")
        }
        let labelled = states.enumerated().map { ($0.offset == 0 ? "embedding" : "layer \($0.offset - 1)", $0.element) }
            + [("logits", logits)]
        let rows = try seams(labelled, bf16: bf16, f32: f32, keys: states.indices.map { "hidden.\($0)" } + ["output"])
        report(name, rows)
        let pieces = try gemma3nPieces(net, configuration, tokens: tokens, bf16: bf16, f32: f32)
        if let pieces {
            report("\(name) isolated", pieces)
        }
        guard let spread else {
            assertRoundingPlacement(name, endToEnd: rows, isolated: pieces ?? [rows[1]])
            return
        }
        let exact = floats(try XCTUnwrap(f32["output"]))
        let runs = spread.keys.filter { $0.hasPrefix("spread.") && !$0.hasSuffix(".reached") }.count
        let theirs = try (0 ..< runs).map { distance(floats(try XCTUnwrap(spread["spread.\($0)"])), exact) }.sorted()
        let reached = try (0 ..< runs).map { try XCTUnwrap(spread["spread.\($0).reached"]).asArray(Float.self).map(Double.init) }
        let targets = (0 ..< configuration.layerCount).map { layer in reached.map { $0[layer] }.sorted()[runs / 2] }
        let ours = (0 ..< runs).map { seed in
            distance(floats(gemma3nNudgedLogits(net, configuration, tokens: tokens, targets: targets, seed: UInt64(seed))), exact)
        }.sorted()
        func summary(_ values: [Double]) -> String {
            String(format: "min %.3e median %.3e 95th %.3e max %.3e", values[0], values[values.count / 2],
                   values[Int((0.95 * Double(values.count)).rounded(.up)) - 1], values[values.count - 1])
        }
        print("VALIDATION bf16 \(name) logits from float32 under one-step nudges at every layer, \(runs) runs each: "
              + "reference \(summary(theirs)); this port \(summary(ours))")
        XCTAssertLessThanOrEqual(ours[runs / 2], 2 * theirs[runs / 2],
                                 "\(name) logits: the median nudged run sits farther from float32 than twice the reference's")
        assertRoundingPlacement(name, endToEnd: Array(rows.dropLast()), isolated: pieces ?? [rows[1]])
    }

    /// The logits of `net` composed from its own expansion of `tokens`, each layer's input first nudged by
    /// one bf16 step at random elements until it sits `targets[layer]` (`1 - cosine`) from where it was,
    /// as `hf_bf16_spread` nudges the reference: on the sign-magnitude bits +1 moves away from zero and -1
    /// toward it, a zero moves away, and a step that would leave the finite range is left out. `seed`
    /// fixes the elements and the signs.
    private func gemma3nNudgedLogits(_ net: NFKMLXGemma3nNet, _ configuration: NFKMLXGemma3nConfiguration,
                                     tokens: [Int32], targets: [Double], seed: UInt64) -> MLXArray {
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        func random() -> UInt64 {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        func value(_ bits: UInt16) -> Double { Double(Float(bitPattern: UInt32(bits) << 16)) }
        func nudged(_ hidden: MLXArray, target: Double) -> MLXArray {
            let original = hidden.view(dtype: .uint16).asArray(UInt16.self)
            let values = original.map(value)
            var order = Array(original.indices)
            for i in stride(from: order.count - 1, to: 0, by: -1) {
                order.swapAt(i, Int(random() % UInt64(i + 1)))
            }
            let away = order.map { _ in random() & 1 == 1 }
            var bits = original
            var taken = 0, goal = 1
            while true {
                for slot in taken ..< goal {
                    let index = order[slot], current = original[index]
                    let step: UInt16 = current & 0x7FFF == 0 || away[slot] ? current &+ 1 : current &- 1
                    if step & 0x7F80 != 0x7F80 { bits[index] = step }
                }
                taken = goal
                if distance(bits.map(value), values) >= target || taken == bits.count { break }
                goal = min(goal * 2, bits.count)
            }
            return MLXArray(bits, hidden.shape).view(dtype: .bfloat16)
        }
        let input = MLXArray(tokens).reshaped([1, tokens.count])
        let embeddings = net.embed(input)
        let perLayer = net.projectedPerLayerInputs(embeddings: embeddings, perLayer: net.perLayerEmbeddings(input))
        let masks = NFKMLXGemma3nMasks.make(length: tokens.count, offset: 0, window: configuration.slidingWindow)
        var hidden = net.expanded(embeddings)
        var donated = [Int: (keys: MLXArray, values: MLXArray)]()
        for index in 0 ..< configuration.layerCount {
            eval(hidden)
            hidden = nudged(hidden, target: targets[index])
            let shared = configuration.keyValueDonor(forLayer: index).flatMap { donated[$0] }
            let (next, keys, values) = net.layers[index](
                hidden, perLayerInput: perLayer[0..., 0..., index, 0...],
                mask: configuration.layerTypes[index] == .full ? masks.full : masks.sliding,
                offset: 0, shared: shared, cache: nil, layer: index)
            if configuration.donatesKeyValues(layer: index) {
                donated[index] = (keys, values)
            }
            hidden = next
        }
        let logits = net.logits(fromHidden: net.collapsed(hidden))[0]
        eval(logits)
        return logits
    }

    /// The pieces of a Gemma 3n stack run alone on the reference's own bf16 inputs, when both records probe
    /// every layer (`IK_PROBE_LAYERS=0,…,last`); nil otherwise. The hidden-state record keeps only the active
    /// AltUp copy while the logits read all four, so the blocks are measured on all four:
    /// - the expansion of the embedding into the four copies, against the reference's layer-0 input;
    /// - each block on the reference's four copies (`L.block.in.whole` → `L.block.out.whole`), a sharing
    ///   layer reading the keys and values its donor block produced on the reference's input;
    /// - the output path (the collapse of the copies, the final norm, the head, the softcap) on the
    ///   reference's last four copies.
    private func gemma3nPieces(_ net: NFKMLXGemma3nNet, _ configuration: NFKMLXGemma3nConfiguration, tokens: [Int32],
                               bf16: [String: MLXArray], f32: [String: MLXArray]) throws -> [Seam]? {
        let layers = 0 ..< configuration.layerCount
        guard layers.allSatisfy({ bf16["\($0).block.in.whole"] != nil && f32["\($0).block.out.whole"] != nil }) else {
            return nil
        }
        let input = MLXArray(tokens).reshaped([1, tokens.count])
        let embeddings = net.embed(input)
        let perLayer = net.projectedPerLayerInputs(embeddings: embeddings, perLayer: net.perLayerEmbeddings(input))
        let masks = NFKMLXGemma3nMasks.make(length: tokens.count, offset: 0, window: configuration.slidingWindow)
        var donated = [Int: (keys: MLXArray, values: MLXArray)]()
        var pieces = [(String, MLXArray)]()
        var keys = [String]()
        pieces.append(("altup expansion", net.expanded(embeddings)))
        keys.append("0.block.in.whole")
        for index in layers {
            let whole = try XCTUnwrap(bf16["\(index).block.in.whole"]).asType(.bfloat16)
            let shared = configuration.keyValueDonor(forLayer: index).flatMap { donated[$0] }
            let (output, layerKeys, layerValues) = net.layers[index](
                whole, perLayerInput: perLayer[0..., 0..., index, 0...],
                mask: configuration.layerTypes[index] == .full ? masks.full : masks.sliding,
                offset: 0, shared: shared, cache: nil, layer: index)
            if configuration.donatesKeyValues(layer: index) {
                donated[index] = (layerKeys, layerValues)
            }
            pieces.append(("layer \(index)", output))
            keys.append("\(index).block.out.whole")
        }
        let last = try XCTUnwrap(bf16["\(configuration.layerCount - 1).block.out.whole"]).asType(.bfloat16)
        pieces.append(("output path", net.logits(fromHidden: net.collapsed(last))[0]))
        keys.append("output")
        eval(pieces.map(\.1))
        return try seams(pieces, bf16: bf16, f32: f32, keys: keys)
    }

    // Each piece of a Gemma 3n block on the reference's own bf16 input (`hf_layer_probe`,
    // `IK_PROBE_DTYPE=bfloat16 IK_PROBE_LAYERS=0,1,20`), and a non-sharing block whole on the
    // reference's four AltUp copies, which is the isolation the hidden-state record cannot give.
    func testGemma3nE2BBlockPiecesMatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA3N_E2B"], "IK_VAL_GEMMA3N_E2B"))
        let probe = try record("gemma3n_e2b_bf16_probe.safetensors")
        let configuration = try NFKMLXGemma3nLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXGemma3nNet(configuration)
        try NFKMLXGemma3nLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        let tokens = MLXArray(try XCTUnwrap(probe["tokens"]).asArray(Int32.self)).reshaped([1, -1])
        let embeddings = net.embed(tokens)
        let perLayer = net.projectedPerLayerInputs(embeddings: embeddings, perLayer: net.perLayerEmbeddings(tokens))
        let masks = NFKMLXGemma3nMasks.make(length: tokens.dim(1), offset: 0, window: configuration.slidingWindow)

        var lines = [try piece("altup expansion", net.expanded(embeddings), probe, "0.block.in.whole")]
        for index in [0, 1, 20] where probe["\(index).block.in"] != nil {
            let block = net.layers[index], p = "\(index)."
            func input(_ key: String) throws -> MLXArray {
                try XCTUnwrap(probe[p + key], "no \(p + key)").asType(.bfloat16).expandedDimensions(axis: 0)
            }
            lines.append("layer \(index) (\(configuration.layerTypes[index]))")
            if configuration.keyValueDonor(forLayer: index) == nil {
                let whole = try XCTUnwrap(probe[p + "block.in.whole"]).asType(.bfloat16)
                let mask = configuration.layerTypes[index] == .full ? masks.full : masks.sliding
                let (output, _, _) = block(whole, perLayerInput: perLayer[0..., 0..., index, 0...], mask: mask,
                                           offset: 0, shared: nil, cache: nil, layer: index)
                lines.append(try piece("block (all copies)", output, probe, p + "block.out.whole"))
            }
            lines.append(try piece("router_norm", block.altup.routerNorm(input("altup.router_norm.in")), probe, p + "altup.router_norm.out"))
            lines.append(try piece("modality_router", block.altup.router(input("altup.modality_router.in")), probe,
                                   p + "altup.modality_router.out"))
            lines.append(try piece("input_layernorm", block.inputNorm(input("input_layernorm.in")), probe, p + "input_layernorm.out"))
            lines.append(try piece("laurel", block.laurel(input("laurel.in")), probe, p + "laurel.out"))
            lines.append(try piece("q_norm", block.attention.queryNorm(input("self_attn.q_norm.in")), probe, p + "self_attn.q_norm.out"))
            let mask = (configuration.layerTypes[index] == .full ? masks.full : masks.sliding)?.asType(.bfloat16)
            lines.append(try piece("attention", NFKReferenceRounding.attention(
                queries: input("attn.q"), keys: input("attn.k"), values: input("attn.v"), scale: 1,
                mask: mask)[0].transposed(1, 0, 2), probe, p + "attn.out"))
            lines.append(try piece("post_attention_norm", block.postAttentionNorm(input("post_attention_layernorm.in")),
                                   probe, p + "post_attention_layernorm.out"))
            lines.append(try piece("gate_proj", block.feedForward.gate(input("mlp.gate_proj.in")), probe, p + "mlp.gate_proj.out"))
            lines.append(try piece("mlp", block.feedForward(input("mlp.in")), probe, p + "mlp.out"))
            lines.append(try piece("per_layer_gelu", NFKReferenceRounding.geluTanh(input("act_fn.in")), probe, p + "act_fn.out"))
            lines.append(try piece("per_layer_projection", block.perLayerProjection(input("per_layer_projection.in")),
                                   probe, p + "per_layer_projection.out"))
            lines.append(try piece("post_per_layer_norm", block.postPerLayerInputNorm(input("post_per_layer_input_norm.in")),
                                   probe, p + "post_per_layer_input_norm.out"))
        }
        print("VALIDATION bf16 gemma3n-e2b pieces:\n" + lines.joined(separator: "\n"))
    }

    // MARK: The dense language decoder

    // Qwen3-0.6B through the dense decoder the Qwen3, Llama, Mistral, and Granite Speech releases share,
    // at the released bf16 against transformers at bf16 and float32 (`hf_layer_probe`).
    func testQwen3_06BInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_QWEN3"], "IK_VAL_QWEN3"))
        try languageDecoder("qwen3-0.6b", directory: directory, bf16: record("qwen3_06b_bf16_probe.safetensors"),
                            f32: record("qwen3_06b_f32.safetensors"))
    }

    // The Qwen3 sizes too large for float32 here, cut to their first four layers
    // (`Tools/validation-assets/truncate.py`, `IK_VAL_QWEN3_14B_CUT4` / `IK_VAL_QWEN3_32B_CUT4`): every
    // hidden state at float32 against transformers' float32, then bf16 against bf16.
    func testQwen3LargerSizePrefixesMatchTheReferenceAtBothPrecisions() throws {
        try requireMLXRuntime()
        var measured = 0
        for (name, key) in [("qwen3_14b_cut4", "IK_VAL_QWEN3_14B_CUT4"), ("qwen3_32b_cut4", "IK_VAL_QWEN3_32B_CUT4")] {
            guard let path = config[key], FileManager.default.fileExists(atPath: path),
                  let bf16 = try? record("\(name)_bf16.safetensors"),
                  let f32 = try? record("\(name)_f32.safetensors") else { continue }
            let directory = URL(fileURLWithPath: path)
            // The float32 net is released before the bf16 one loads; the two do not fit together.
            try autoreleasepool {
                let geometry = try NFKMLXLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
                let exact = try loaded(NFKMLXLanguage.makeNet(geometry), onCPU: false) {
                    try NFKMLXLanguage.loadWeights(into: $0, fromDirectory: directory, precision: .float32)
                }
                let tokens = MLXArray(try XCTUnwrap(f32["tokens"]).asArray(Int32.self)).reshaped([1, -1])
                try assertFloat32(name, states: exact.layerStates(tokens) + [exact(tokens)[0]], f32: f32)
            }
            Memory.clearCache()
            try autoreleasepool { try languageDecoder(name, directory: directory, bf16: bf16, f32: f32) }
            Memory.clearCache()
            measured += 1
        }
        if measured == 0 { throw XCTSkip("set IK_VAL_QWEN3_14B_CUT4 or IK_VAL_QWEN3_32B_CUT4") }
    }

    /// The net `load` builds, with every parameter evaluated one array at a time before the first forward,
    /// on the CPU when `onCPU` holds. A release that leaves reads to its first evaluation (a mixture's
    /// per-expert tensors, paged experts) loads on the CPU: those reads and conversions evaluated in a GPU
    /// command buffer end the process at the GPU watchdog when their pages have to come back from swap,
    /// and the CPU has no watchdog. A dense release's loader reads every tensor before it returns, so its
    /// parameters evaluate on the GPU and wait on no file.
    private func loaded<Net: Module>(onCPU: Bool, _ load: () throws -> Net) throws -> Net {
        func evaluated() throws -> Net {
            let net = try load()
            for (_, weight) in net.parameters().flattened() { eval(weight) }
            return net
        }
        let net = try onCPU ? Device.withDefaultDevice(.cpu) { try evaluated() } : evaluated()
        Memory.clearCache()
        return net
    }

    /// `net` after `fill` loads its weights, evaluated as ``loaded(onCPU:_:)`` does when `condition` holds.
    private func loaded<Net: Module>(_ net: Net, onCPU: Bool, if condition: Bool = true,
                                     _ fill: (Net) throws -> Void) throws -> Net {
        guard condition else {
            try fill(net)
            return net
        }
        return try loaded(onCPU: onCPU) { () throws -> Net in
            try fill(net)
            return net
        }
    }

    /// Every state (`hidden.i`, then `output`) of a float32 run against the reference's float32 record.
    private func assertFloat32(_ name: String, states: [MLXArray], f32: [String: MLXArray],
                               file: StaticString = #filePath, line: UInt = #line) throws {
        eval(states)
        let keys = (0 ..< states.count - 1).map { "hidden.\($0)" } + ["output"]
        var worst = 1.0
        for (state, key) in zip(states, keys) {
            worst = min(worst, 1 - distance(floats(state), floats(try XCTUnwrap(f32[key], "no \(key)"))))
        }
        let logits = 1 - distance(floats(states.last!), floats(try XCTUnwrap(f32["output"])))
        print("VALIDATION PARITY \(name) float32: worst of \(states.count) states \(worst), logit cosine \(logits)")
        XCTAssertGreaterThan(worst, 0.99999, "\(name) matches transformers at float32", file: file, line: line)
    }

    /// The end-to-end and isolated report for a decoder release against a record pair. `routes`
    /// (`route.L.index` per mixture layer) fixes each mixture layer's experts to the reference's.
    private func languageDecoder(_ name: String, directory: URL, bf16: [String: MLXArray],
                                 f32: [String: MLXArray], routes: [String: MLXArray]? = nil) throws {
        let geometry = try NFKMLXLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXLanguage.makeNet(geometry)
        try NFKMLXLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        if let routes {
            for (index, layer) in net.model.layers.enumerated() {
                (layer.feedForward as? NFKLMMixtureFeedForward)?.forcedChoice = try recordedChoice(routes, layer: index)
            }
        }
        let tokens = MLXArray(try XCTUnwrap(bf16["tokens"]).asArray(Int32.self)).reshaped([1, -1])
        let states = net.layerStates(tokens)
        let logits = net(tokens)[0]
        eval(states + [logits])
        let labelled = states.enumerated().map { ($0.offset == 0 ? "embedding" : "layer \($0.offset - 1)", $0.element) }
            + [("logits", logits)]
        let rows = try seams(labelled, bf16: bf16, f32: f32, keys: states.indices.map { "hidden.\($0)" } + ["output"])
        report(name, rows)

        let mask = NFKMLXLanguageNet.causalMask(tokens.dim(1), offset: 0)
        var isolated = [(String, MLXArray)]()
        for (index, layer) in net.model.layers.enumerated() {
            let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            var output = layer(input, mask: mask, cache: nil, layer: index)
            if index == net.model.layers.count - 1 { output = net.model.norm(output) }
            isolated.append(("layer \(index)", output))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolated.indices.map { "hidden.\($0 + 1)" })
        report("\(name) isolated", isolatedRows)
        assertRoundingPlacement(name, endToEnd: rows, isolated: isolatedRows)
    }

    /// The experts the reference kept at `layer` (`route.L.index`, `[tokens, k]`) as `[tokens][k]`, or nil
    /// where the record names none (a dense layer).
    private func recordedChoice(_ routes: [String: MLXArray], layer: Int) throws -> [[Int32]]? {
        guard let recorded = routes["route.\(layer).index"] else { return nil }
        let k = recorded.dim(-1)
        let flat = recorded.asArray(Int32.self)
        return stride(from: 0, to: flat.count, by: k).map { Array(flat[$0 ..< $0 + k]) }
    }

    // MARK: Mixture cut releases

    // The mixture families' releases cut to their first layers (`truncate.py`): Qwen3-30B-A3B and
    // Qwen1.5-MoE-A2.7B to four, Mixtral-8x7B and gpt-oss-20b to two (`IK_VAL_QWEN3_MOE_CUT4`,
    // `IK_VAL_QWEN2_MOE_CUT4`, `IK_VAL_MIXTRAL_CUT2`, `IK_VAL_GPT_OSS_CUT2`). Records come from
    // `hf_layer_probe`: float32 streamed from a bf16 load (`IK_PROBE_STREAM_F32=1`), and bf16 with
    // `IK_PROBE_ROUTES=1`, which records each layer's kept experts. Every state at float32, then bf16
    // routed as the reference routed, since `torch.topk` breaks a tie at the `k`-th score without a
    // fixed rule.
    func testMixtureCutPrefixesMatchTheReferenceAtBothPrecisionsRoutedAsTheReference() throws {
        try requireMLXRuntime()
        var measured = 0
        for (name, key) in [("qwen3_moe_cut4", "IK_VAL_QWEN3_MOE_CUT4"), ("qwen2_moe_cut4", "IK_VAL_QWEN2_MOE_CUT4"),
                            ("mixtral_cut2", "IK_VAL_MIXTRAL_CUT2"), ("gpt_oss_cut2", "IK_VAL_GPT_OSS_CUT2")] {
            guard let path = config[key], FileManager.default.fileExists(atPath: path),
                  let bf16 = try? record("\(name)_bf16.safetensors"),
                  let f32 = try? record("\(name)_f32.safetensors") else { continue }
            let directory = URL(fileURLWithPath: path)
            // The float32 net is released before the bf16 one loads; the two do not fit together.
            try autoreleasepool {
                let geometry = try NFKMLXLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
                // gpt-oss's experts stay MXFP4-packed at float32 and dequantize exactly, which is the
                // arithmetic the float32 reference runs on them.
                let exact = try loaded(NFKMLXLanguage.makeNet(geometry), onCPU: true) {
                    try NFKMLXLanguage.loadWeights(into: $0, fromDirectory: directory, precision: .float32)
                }
                let tokens = MLXArray(try XCTUnwrap(f32["tokens"]).asArray(Int32.self)).reshaped([1, -1])
                try assertFloat32(name, states: exact.layerStates(tokens) + [exact(tokens)[0]], f32: f32)
            }
            Memory.clearCache()
            try autoreleasepool { try languageDecoder(name, directory: directory, bf16: bf16, f32: f32, routes: bf16) }
            Memory.clearCache()
            measured += 1
        }
        if measured == 0 {
            throw XCTSkip("set IK_VAL_QWEN3_MOE_CUT4, IK_VAL_QWEN2_MOE_CUT4, IK_VAL_MIXTRAL_CUT2 or IK_VAL_GPT_OSS_CUT2")
        }
    }

    // gpt-oss-20b's first layer piece by piece on the reference's own bf16 inputs: the cut's bf16 record
    // probes layer 0 (`hf_layer_probe`'s default), so a rounding placed differently counts at the one
    // seam that places it. The mixture routes as the reference routed.
    func testGPTOSSFirstLayerPiecesMatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GPT_OSS_CUT2"], "IK_VAL_GPT_OSS_CUT2"))
        let probe = try record("gpt_oss_cut2_bf16.safetensors")
        let geometry = try NFKMLXLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXLanguage.makeNet(geometry)
        try NFKMLXLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        let block = net.model.layers[0]
        func input(_ key: String) throws -> MLXArray {
            try XCTUnwrap(probe["0." + key], "no 0.\(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let mask = NFKMLXLanguageNet.causalMask(try input("block.in").dim(1), offset: 0)
        let mixture = try XCTUnwrap(block.feedForward as? NFKLMMixtureFeedForward)
        mixture.forcedChoice = try recordedChoice(probe, layer: 0)
        let (weights, chosen) = mixture.route(try input("mlp.in"))
        var lines = [String]()
        lines.append(try piece("input_layernorm", block.attentionNorm(input("input_layernorm.in")), probe, "0.input_layernorm.out"))
        lines.append(try piece("q_proj", block.attention.queryProjection(input("self_attn.q_proj.in")), probe, "0.self_attn.q_proj.out"))
        lines.append(try piece("k_proj", block.attention.keyProjection(input("self_attn.k_proj.in")), probe, "0.self_attn.k_proj.out"))
        lines.append(try piece("v_proj", block.attention.valueProjection(input("self_attn.v_proj.in")), probe, "0.self_attn.v_proj.out"))
        lines.append(try piece("o_proj", block.attention.outputProjection(input("self_attn.o_proj.in")), probe, "0.self_attn.o_proj.out"))
        lines.append(try piece("self_attn", block.attention(input("input_layernorm.out"), mask: mask, cache: nil, layer: 0),
                               probe, "0.self_attn.out"))
        lines.append(try piece("post_attention_layernorm", block.feedForwardNorm(input("post_attention_layernorm.in")),
                               probe, "0.post_attention_layernorm.out"))
        lines.append(try piece("experts (routed as recorded)",
                               NFKReferenceRounding.combined(mixture.experts(input("mlp.experts.in"), experts: chosen),
                                                             weights: weights, chosen: chosen),
                               probe, "0.mlp.experts.out"))
        lines.append(try piece("mlp (routed as recorded)", mixture(input("mlp.in")), probe, "0.mlp.out"))
        lines.append(try piece("block", block(input("block.in"), mask: mask, cache: nil, layer: 0), probe, "0.block.out"))
        print("VALIDATION bf16 gpt-oss layer 0 pieces:\n" + lines.joined(separator: "\n"))
    }

    // Granite-4.0-H-Tiny cut to its first six layers (`IK_VAL_GRANITE_TINY_CUT6`), which end on its first
    // attention layer, every layer routing 6 of 64 experts beside the shared MLP. The records are made as
    // the mixture cuts' are; the bf16 run routes as the reference routed.
    func testGraniteMixtureCutMatchesTheReferenceAtBothPrecisionsRoutedAsTheReference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GRANITE_TINY_CUT6"], "IK_VAL_GRANITE_TINY_CUT6"))
        let bf16 = try record("granite_tiny_cut6_bf16.safetensors"), f32 = try record("granite_tiny_cut6_f32.safetensors")
        let configuration = try NFKMLXGraniteHybrid.configuration(fromDirectory: directory)
        let tokens = try XCTUnwrap(f32["tokens"]).asType(.int32).reshaped([1, -1])
        // The state entering each block, the last one normalized, then the logits.
        func states(_ net: NFKMLXGraniteHybridNet) -> [MLXArray] {
            var states = net.blockStates(tokens)
            states[states.count - 1] = net.model.norm(states[states.count - 1])
            return states + [net(tokens)[0]]
        }
        try autoreleasepool {
            let exact = try loaded(NFKMLXGraniteHybrid.makeNet(configuration), onCPU: true) {
                try NFKMLXGraniteHybrid.loadWeights(into: $0, fromDirectory: directory, precision: .float32)
            }
            try assertFloat32("granite-tiny-cut6", states: states(exact), f32: f32)
        }
        Memory.clearCache()

        let net = NFKMLXGraniteHybrid.makeNet(configuration)
        try NFKMLXGraniteHybrid.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        for (index, layer) in net.model.layers.enumerated() {
            layer.moe?.forcedChoice = try recordedChoice(bf16, layer: index)
        }
        let reduced = states(net)
        eval(reduced)
        let labelled = reduced.enumerated().map {
            ($0.offset == 0 ? "embedding" : $0.offset == reduced.count - 1 ? "logits" : "layer \($0.offset - 1)", $0.element)
        }
        let rows = try seams(labelled, bf16: bf16, f32: f32,
                             keys: (0 ..< reduced.count - 1).map { "hidden.\($0)" } + ["output"])
        report("granite-tiny-cut6", rows)
        var isolated = [(String, MLXArray)]()
        for (index, layer) in net.model.layers.enumerated() {
            let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            var output = layer(input)
            if index == net.model.layers.count - 1 { output = net.model.norm(output) }
            isolated.append(("layer \(index) (\(configuration.layerTypes[index]))", output))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolated.indices.map { "hidden.\($0 + 1)" })
        report("granite-tiny-cut6 isolated", isolatedRows)
        assertRoundingPlacement("granite-tiny-cut6", endToEnd: rows, isolated: isolatedRows)
    }

    // Qwen3.8-Flash-Next (Qwen4-Exp) cut to its first two layers (`IK_VAL_QWEN4_EXP_CUT2`): linear attention
    // and 512 experts with ten kept, under hyper-connections. `ple_layer_ids` counts from one, so the release's
    // `[2]` names the cut's second layer; the cut sets it empty and leaves out the 102 GB n-gram table, and
    // both sides run that layer without it. The records are made as the mixture cuts' are, the oracle
    // driving the release's own image-text-to-text class with no cache; the float32 run pages the experts,
    // and the bf16 run routes as the reference routed.
    func testQwen4ExpCutMatchesTheReferenceAtBothPrecisionsRoutedAsTheReference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_QWEN4_EXP_CUT2"], "IK_VAL_QWEN4_EXP_CUT2"))
        let bf16 = try record("qwen4_exp_cut2_bf16.safetensors"), f32 = try record("qwen4_exp_cut2_f32.safetensors")
        let geometry = try NFKMLXQwen4Exp.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let tokens = MLXArray(try XCTUnwrap(f32["tokens"]).asArray(Int32.self)).reshaped([1, -1])
        try autoreleasepool {
            let exact = try loaded(NFKMLXQwen4Exp.makeNet(geometry), onCPU: true) {
                try NFKMLXQwen4Exp.loadWeights(into: $0, fromDirectory: directory, precision: .float32, residency: .paged)
            }
            exact.expertStore?.cacheByteBudget = 4 << 30
            try assertFloat32("qwen4-exp-cut2", states: exact.hiddenStates(tokens) + [exact(tokens)[0]], f32: f32)
        }
        Memory.clearCache()

        let net = NFKMLXQwen4Exp.makeNet(geometry)
        try NFKMLXQwen4Exp.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint, residency: .resident)
        let layers = net.model.languageModel.layers
        for (index, layer) in layers.enumerated() {
            layer.feedForward.forcedChoice = try recordedChoice(bf16, layer: index)
        }
        let states = net.hiddenStates(tokens)
        let logits = net(tokens)[0]
        eval(states + [logits])
        let labelled = states.enumerated().map { ($0.offset == 0 ? "streams" : "layer \($0.offset - 1)", $0.element) }
            + [("logits", logits)]
        let rows = try seams(labelled, bf16: bf16, f32: f32, keys: states.indices.map { "hidden.\($0)" } + ["output"])
        report("qwen4-exp-cut2", rows)
        let length = tokens.dim(1)
        let positions = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, 1, length])
        let (cosTable, sinTable) = net.rotaryTable(positions: broadcast(positions, to: [3, 1, length]))
        let queryIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([length, 1])
        let keyIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, length])
        let causal = (keyIndex .<= queryIndex).reshaped([1, 1, length, length])
        var isolated = [(String, MLXArray)]()
        for (index, layer) in layers.enumerated() {
            let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            var output = layer(input, cos: cosTable, sin: sinTable, mask: causal, ngramIndices: nil)
            if index == layers.count - 1 { output = net.model.languageModel.mixer(output).read }
            isolated.append(("layer \(index)", output))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolated.indices.map { "hidden.\($0 + 1)" })
        report("qwen4-exp-cut2 isolated", isolatedRows)
        assertRoundingPlacement("qwen4-exp-cut2", endToEnd: rows, isolated: isolatedRows)
    }

    // Qwen3.8-Flash-Next cut to its first four layers (`IK_VAL_QWEN4_EXP_CUT4`), the fourth its first
    // full-attention layer, over the first 3,072 tokens of a text: past 2,048 tokens the query-sparse-attention
    // indexer keeps 512 of the blocks a query sees and discards the rest. The records come from
    // `hf_layer_probe` with `IK_PROBE_PROMPT_FILE`, `IK_PROBE_LAYERS=3`, and `IK_PROBE_INDEXER=1` (the
    // indexer's block scores); the float32 run streams from a bf16 load. At each precision the indexer and
    // the sparse-attention layer run on the reference's own input. The cut's mixture layers are the code the
    // two-layer cut holds at both precisions; over 3,072 tokens each reaches all 512 of its experts, a 5 GB
    // bank at bf16 that the store builds beside the experts it stacks. A block either side keeps where the
    // reference does not, or the reverse, must score within a near-tie of the reference's 512th: float32
    // accumulation at float32, one bf16 step at bf16, where the queries and keys the scores multiply are
    // themselves rounded to bf16.
    func testQwen4ExpIndexerCutMatchesTheReferenceAtBothPrecisions() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_QWEN4_EXP_CUT4"], "IK_VAL_QWEN4_EXP_CUT4"))
        let bf16 = try record("qwen4_exp_cut4_bf16.safetensors"), f32 = try record("qwen4_exp_cut4_f32.safetensors")
        let geometry = try NFKMLXQwen4Exp.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let length = try XCTUnwrap(f32["tokens"]).dim(0)
        let full = try XCTUnwrap(geometry.layerTypes.firstIndex(of: .sparseAttention), "the cut keeps a sparse-attention layer")
        let positions = broadcast(MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, 1, length]), to: [3, 1, length])
        let queryIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([length, 1])
        let keyIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, length])
        let causal = (keyIndex .<= queryIndex).reshaped([1, 1, length, length])

        try autoreleasepool {
            let exact = try loaded(NFKMLXQwen4Exp.makeNet(geometry), onCPU: true) {
                try NFKMLXQwen4Exp.loadWeights(into: $0, fromDirectory: directory, precision: .float32, residency: .paged)
            }
            try Device.withDefaultDevice(.cpu) {
                let (cos, sin) = exact.rotaryTable(positions: positions)
                let block = exact.model.languageModel.layers[full]
                try assertIndexerSelection("qwen4-exp-cut4 float32", block, layer: full,
                                           record: f32, cos: cos, sin: sin, type: .float32)
                let attention = try XCTUnwrap(block.attention)
                let input = try XCTUnwrap(f32["\(full).self_attn.in"]).asType(.float32).expandedDimensions(axis: 0)
                let attended = attention(input, cos: cos, sin: sin, mask: causal)[0]
                let similarity = 1 - distance(floats(attended), floats(try XCTUnwrap(f32["\(full).self_attn.out"])))
                print("VALIDATION PARITY qwen4-exp-cut4 float32: sparse attention on the reference's input \(similarity)")
                XCTAssertGreaterThan(similarity, 0.99999, "the sparse-attention layer matches transformers at float32")
            }
        }
        Memory.clearCache()

        let net = NFKMLXQwen4Exp.makeNet(geometry)
        try NFKMLXQwen4Exp.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint, residency: .paged)
        let block = net.model.languageModel.layers[full]
        let (cos, sin) = net.rotaryTable(positions: positions)
        try assertIndexerSelection("qwen4-exp-cut4 bf16", block, layer: full, record: bf16, cos: cos, sin: sin,
                                   type: .bfloat16)
        let input = try XCTUnwrap(bf16["\(full).self_attn.in"]).asType(.bfloat16).expandedDimensions(axis: 0)
        let attended = try XCTUnwrap(block.attention)(input, cos: cos, sin: sin, mask: causal)[0]
        eval(attended)
        let rows = try seams([("sparse attention", attended)], bf16: bf16, f32: f32, keys: ["\(full).self_attn.out"])
        report("qwen4-exp-cut4 isolated", rows)
        for row in rows {
            XCTAssertLessThan(row.ours, 0.25 * row.floor,
                              "qwen4-exp-cut4 \(row.label) on the reference's input: a rounding placed unlike the reference's")
        }
        assertWithinBlockFloors("qwen4-exp-cut4", rows)
        print("VALIDATION MEMORY qwen4-exp-cut4: peak \(NFKMLXGPU.peakMemory >> 20) MB")
    }

    /// The indexer of `layer` on the reference's own input (`L.self_attn.indexer.in`, cast to `type`) against the
    /// reference's selection (`L.self_attn.indexer.out`, 0 where kept). Every block the two sides decide
    /// differently must score within a near-tie of the reference's last kept score (`indexer.L.scores`):
    /// 1e-5 of it at float32, the order a sum is taken in; one bf16 step at bf16, the rounding of the
    /// queries and keys the score multiplies.
    private func assertIndexerSelection(_ name: String, _ block: NFKQwen4ExpBlock, layer: Int,
                                        record: [String: MLXArray], cos: MLXArray, sin: MLXArray, type: DType,
                                        file: StaticString = #filePath, line: UInt = #line) throws {
        let indexer = try XCTUnwrap(block.attention?.indexer, "layer \(layer) carries the indexer")
        let input = try XCTUnwrap(record["\(layer).self_attn.indexer.in"]).asType(type).expandedDimensions(axis: 0)
        let ours = indexer(input, cos: cos, sin: sin)[0]
        let theirs = try XCTUnwrap(record["\(layer).self_attn.indexer.out"]).reshaped(ours.shape) .== 0
        let scores = try XCTUnwrap(record["indexer.\(layer).scores"]).asType(.float32)
        eval(ours, theirs)
        let (length, ratio) = (ours.dim(0), indexer.configuration.indexerCompressRatio)
        let budget = indexer.configuration.indexerBlockBudget
        let mine = ours.asArray(Bool.self), reference = theirs.asArray(Bool.self)
        let nearTie = type == .float32 ? 1e-5 : 1.0 / 256
        let table = scores.asArray(Float.self), width = scores.dim(1)
        var selecting = 0, differingQueries = 0, differingBlocks = 0, widest = 0.0
        for query in 0 ..< length {
            let blocks = (query + 1) / ratio
            if blocks > budget { selecting += 1 }
            var disagreed = [Int]()
            for block in 0 ..< blocks where mine[query * length + block * ratio] != reference[query * length + block * ratio] {
                disagreed.append(block)
            }
            for token in (blocks * ratio) ..< query + 1 where mine[query * length + token] != reference[query * length + token] {
                XCTFail("\(name): query \(query) drops tail token \(token), which no block holds", file: file, line: line)
            }
            guard !disagreed.isEmpty else { continue }
            differingQueries += 1
            differingBlocks += disagreed.count
            let row = Array(table[(query * width) ..< (query * width + blocks)]).map(Double.init)
            let kth = row.sorted(by: >)[min(budget, blocks) - 1]
            for block in disagreed {
                let margin = abs(row[block] - kth) / max(abs(kth), 1e-30)
                widest = max(widest, margin)
                XCTAssertLessThanOrEqual(margin, nearTie, "\(name): query \(query) block \(block) scores \(row[block]) "
                                         + "against the reference's 512th \(kth); a disagreement past a near-tie",
                                         file: file, line: line)
            }
        }
        print(String(format: "VALIDATION indexer %@: %d queries, %d past the budget; %d queries decide %d blocks "
                     + "differently, the widest at %.2e of the reference's last kept score", name, length, selecting,
                     differingQueries, differingBlocks, widest))
        XCTAssertGreaterThan(selecting, 0, "\(name): the prompt reaches past the indexer's budget", file: file, line: line)
    }

    // The Qwen4-Exp cut's first layer piece by piece on the reference's own bf16 inputs (its bf16 record
    // probes layer 0), the mixture routed as the reference routed.
    func testQwen4ExpCutFirstLayerPiecesMatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_QWEN4_EXP_CUT2"], "IK_VAL_QWEN4_EXP_CUT2"))
        let probe = try record("qwen4_exp_cut2_bf16.safetensors")
        let geometry = try NFKMLXQwen4Exp.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXQwen4Exp.makeNet(geometry)
        try NFKMLXQwen4Exp.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint, residency: .resident)
        let block = net.model.languageModel.layers[0]
        func input(_ key: String) throws -> MLXArray {
            try XCTUnwrap(probe["0." + key], "no 0.\(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        block.feedForward.forcedChoice = try recordedChoice(probe, layer: 0)
        let attentionResidual = block.attentionResidual, feedForwardResidual = block.feedForwardResidual
        var lines = [String]()
        for (name, residual) in [("attn_hyper_connection", attentionResidual), ("mlp_hyper_connection", feedForwardResidual)] {
            lines.append(try piece("\(name).hc_norm", residual.norm(input("\(name).hc_norm.in")), probe, "0.\(name).hc_norm.out"))
            lines.append(try piece("\(name).mix_down", residual.mixDown(input("\(name).input_mix_weight_down.in")),
                                   probe, "0.\(name).input_mix_weight_down.out"))
            lines.append(try piece("\(name).mix_up", residual.mixUp(input("\(name).input_mix_weight_up.in")),
                                   probe, "0.\(name).input_mix_weight_up.out"))
            if let inject = residual.inject {
                lines.append(try piece("\(name).inject", inject(input("\(name).block_inject_weight.in")),
                                       probe, "0.\(name).block_inject_weight.out"))
            }
            lines.append(try piece("\(name) read", residual(input("\(name).in")).read, probe, "0.\(name).out"))
        }
        let linear = try XCTUnwrap(block.linearAttention)
        lines.append(try piece("linear_attn", linear(input("linear_attn.in")), probe, "0.linear_attn.out"))
        lines.append(try piece("mlp (routed as recorded)", block.feedForward(input("mlp.in")), probe, "0.mlp.out"))
        let sharedIn = try XCTUnwrap(probe["0.mlp.shared_expert.in"]).asType(.bfloat16).reshaped([1, 1, -1])
        lines.append(try piece("shared_expert (token 0)", block.feedForward.sharedExpert(sharedIn), probe, "0.mlp.shared_expert.out"))
        lines.append(try piece("shared_expert_gate (token 0)", block.feedForward.sharedExpertGate(sharedIn), probe,
                               "0.mlp.shared_expert_gate.out"))
        let length = try input("block.in").dim(1)
        let positions = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, 1, length])
        let (cosTable, sinTable) = net.rotaryTable(positions: broadcast(positions, to: [3, 1, length]))
        let queryIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([length, 1])
        let keyIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, length])
        let causal = (keyIndex .<= queryIndex).reshaped([1, 1, length, length])
        lines.append(try piece("block", block(input("block.in"), cos: cosTable, sin: sinTable, mask: causal, ngramIndices: nil),
                               probe, "0.block.out"))
        print("VALIDATION bf16 qwen4-exp-cut2 layer 0 pieces:\n" + lines.joined(separator: "\n"))
    }

    // The Qwen4-Exp cut's first linear attention step by step on the reference's own bf16 inputs: the
    // record's function probe (`IK_PROBE_FUNCTIONS=causal_conv1d_fn,torch_chunk_gated_delta_rule`) keeps
    // the convolution's and the delta rule's arguments and outputs, the keyword arguments numbered in
    // sorted order (arg3 beta, arg5 g). The delta rule normalizes its queries and keys in float32.
    func testQwen4ExpCutLinearAttentionStepsMatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_QWEN4_EXP_CUT2"], "IK_VAL_QWEN4_EXP_CUT2"))
        let probe = try record("qwen4_exp_cut2_bf16_fn.safetensors")
        let geometry = try NFKMLXQwen4Exp.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXQwen4Exp.makeNet(geometry)
        try NFKMLXQwen4Exp.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint, residency: .resident)
        let linear = try XCTUnwrap(net.model.languageModel.layers[0].linearAttention)
        func tensor(_ key: String) throws -> MLXArray { try XCTUnwrap(probe[key], "no \(key)").asType(.bfloat16) }
        var lines = [String]()
        lines.append(try piece("in_proj_qkv", linear.qkvProjection(tensor("0.linear_attn.in_proj_qkv.in").expandedDimensions(axis: 0)),
                               probe, "0.linear_attn.in_proj_qkv.out"))
        let convIn = try tensor("fn.causal_conv1d_fn.0.arg0").transposed(0, 2, 1)
        let padded = MLX.padded(convIn, widths: [IntOrPair((0, 0)), IntOrPair((geometry.linearConvolutionKernel - 1, 0)), IntOrPair((0, 0))])
        lines.append(try piece("conv1d + silu", NFKReferenceRounding.silu(linear.convolution(padded)).transposed(0, 2, 1),
                               probe, "fn.causal_conv1d_fn.0.out0"))
        let queries = NFKHybridLinearAttention.unitNorm(try tensor("fn.torch_chunk_gated_delta_rule.0.arg0").asType(.float32))
        let keys = NFKHybridLinearAttention.unitNorm(try tensor("fn.torch_chunk_gated_delta_rule.0.arg1").asType(.float32))
        let values = try tensor("fn.torch_chunk_gated_delta_rule.0.arg2").asType(.float32)
        let write = try tensor("fn.torch_chunk_gated_delta_rule.0.arg3").asType(.float32)
        let decay = exp(try XCTUnwrap(probe["fn.torch_chunk_gated_delta_rule.0.arg5"]).asType(.float32))
        let read = NFKQwen4ExpLinearAttention.recurrence(queries: queries, keys: keys, values: values, decay: decay, write: write)
        lines.append(try piece("recurrence (sequential)", read.asType(.bfloat16), probe, "fn.torch_chunk_gated_delta_rule.0.out0"))
        lines.append("  recurrence float32 vs the reference's rounded read: 1-cos " + String(format: "%.3e",
            distance(floats(read), floats(try XCTUnwrap(probe["fn.torch_chunk_gated_delta_rule.0.out0"])))))
        let gate = try tensor("0.linear_attn.in_proj_z.out").reshaped([1, -1, geometry.linearValueHeadCount, geometry.linearValueHeadDimensions])
        let referenceRead = try tensor("fn.torch_chunk_gated_delta_rule.0.out0")
        lines.append(try piece("gated norm", linear.norm(referenceRead, gate: gate).reshaped([1, -1, geometry.linearValueWidth]),
                               probe, "0.linear_attn.out_proj.in"))
        lines.append(try piece("out_proj", linear.outputProjection(tensor("0.linear_attn.out_proj.in").expandedDimensions(axis: 0)),
                               probe, "0.linear_attn.out_proj.out"))
        print("VALIDATION bf16 qwen4-exp-cut2 linear attention steps:\n" + lines.joined(separator: "\n"))
    }

    // MARK: Codestral-Mamba

    // Codestral-Mamba-7B cut to its first four blocks (`Tools/validation-assets/truncate.py`,
    // `IK_VAL_CODESTRAL_CUT4`), which fits float32 where the whole release does not: float32 against
    // float32, then bf16 against bf16 (`hf_layer_probe` at both precisions). transformers' Mamba-2
    // records a state AFTER each block rather than before it, so `hidden.i` is block `i`'s output and
    // the last entry is the final norm's.
    func testCodestralMambaPrefixMatchesTheReferenceAtBothPrecisions() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_CODESTRAL_CUT4"], "IK_VAL_CODESTRAL_CUT4"))
        let bf16 = try record("codestral_cut4_bf16.safetensors")
        let f32 = try record("codestral_cut4_f32.safetensors")
        let configuration = try NFKMLXMamba.configuration(fromDirectory: directory)
        let blocks = configuration.layerCount
        let tokens = MLXArray(try XCTUnwrap(f32["tokens"]).asArray(Int32.self)).reshaped([1, -1])
        let keys = (0 ... blocks).map { "hidden.\($0)" } + ["output"]
        func states(_ precision: NFKMLXWeightPrecision) throws -> (NFKMLXMamba2Net, [(String, MLXArray)]) {
            let net = NFKMLXMamba.makeNet(configuration)
            try NFKMLXMamba.loadWeights(into: net, fromDirectory: directory, precision: precision)
            let outputs = Array(net.blockStates(tokens).dropFirst())
            let labelled = outputs.enumerated().map { ("block \($0.offset)", $0.element) }
                + [("final norm", net.backbone.finalNorm(outputs.last!)), ("logits", net(tokens)[0])]
            eval(labelled.map(\.1))
            return (net, labelled)
        }

        let (_, exact) = try states(.float32)
        var worst = 1.0
        for ((label, state), key) in zip(exact, keys) {
            let similarity = 1 - distance(floats(state), floats(try XCTUnwrap(f32[key])))
            print("VALIDATION PARITY codestral-mamba first \(blocks) blocks float32 \(label): cosine \(similarity)")
            worst = min(worst, similarity)
        }
        XCTAssertGreaterThan(worst, 0.99999, "the released geometry matches transformers at float32")

        let (net, reduced) = try states(.checkpoint)
        let rows = try seams(reduced, bf16: bf16, f32: f32, keys: keys)
        report("codestral-mamba-cut4", rows)
        var isolated = [(String, MLXArray)]()
        for (index, layer) in net.backbone.layers.enumerated() where index > 0 {
            let input = try XCTUnwrap(bf16["hidden.\(index - 1)"]).asType(reduced[index - 1].1.dtype)
            isolated.append(("block \(index)", layer(input.expandedDimensions(axis: 0))))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: (1 ..< blocks).map { "hidden.\($0)" })
        report("codestral-mamba-cut4 isolated", [rows[0]] + isolatedRows)
        assertRoundingPlacement("codestral-mamba-cut4", endToEnd: rows, isolated: [rows[0]] + isolatedRows)
    }

    // MARK: The hybrid decoder

    // Qwen3.5-4B (`IK_VAL_QWEN3_5`): gated delta-rule layers, whose recurrence the reference runs in
    // float32, and gated full attention, at bf16 against bf16 (`hf_layer_probe` under transformers 5.17,
    // layers 0 and 3), whose delta rule normalizes its queries and keys in float32.
    func testQwen35_4BInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_QWEN3_5"], "IK_VAL_QWEN3_5"))
        let bf16 = try record("qwen35_4b_bf16_v517.safetensors"), f32 = try record("qwen35_4b_f32_v517.safetensors")
        let geometry = try NFKMLXHybridLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXHybridLanguage.makeNet(geometry)
        try NFKMLXHybridLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        let tokens = MLXArray(try XCTUnwrap(bf16["tokens"]).asArray(Int32.self)).reshaped([1, -1])
        let states = net.hiddenStates(tokens)
        let logits = net(tokens)[0]
        eval(states + [logits])
        let labelled = states.enumerated().map { ($0.offset == 0 ? "embedding" : "layer \($0.offset - 1)", $0.element) }
            + [("logits", logits)]
        let rows = try seams(labelled, bf16: bf16, f32: f32, keys: states.indices.map { "hidden.\($0)" } + ["output"])
        report("qwen3.5-4b", rows)

        let mask = NFKMLXLanguageNet.causalMask(tokens.dim(1), offset: 0)
        var isolated = [(String, MLXArray)]()
        for (index, layer) in net.model.layers.enumerated() {
            let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            var output = layer(input, mask: mask)
            if index == net.model.layers.count - 1 { output = net.model.norm(output) }
            isolated.append(("layer \(index) (\(layer.kind))", output))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolated.indices.map { "hidden.\($0 + 1)" })
        report("qwen3.5-4b isolated", isolatedRows)
        assertRoundingPlacement("qwen3.5-4b", endToEnd: rows, isolated: isolatedRows)
    }

    // The pieces of Qwen3.5-4B's layer 0 (delta rule) and layer 3 (gated full attention), each on the
    // reference's own bf16 input.
    func testQwen35_4BBlockPiecesMatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_QWEN3_5"], "IK_VAL_QWEN3_5"))
        let probe = try record("qwen35_4b_bf16_v517.safetensors")
        let geometry = try NFKMLXHybridLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXHybridLanguage.makeNet(geometry)
        try NFKMLXHybridLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        func input(_ key: String) throws -> MLXArray {
            try XCTUnwrap(probe[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let length = try XCTUnwrap(probe["tokens"]).shape[0]
        let mask = NFKMLXLanguageNet.causalMask(length, offset: 0).asType(.bfloat16)
        var lines = [String]()
        let linear = net.model.layers[0]
        lines.append("layer 0 (\(linear.kind))")
        lines.append(try piece("input_layernorm", linear.inputNorm(input("0.input_layernorm.in")), probe, "0.input_layernorm.out"))
        lines.append(try piece("linear_attn", try XCTUnwrap(linear.linearAttention)(input("0.input_layernorm.out")), probe, "0.linear_attn.out"))
        lines.append(try piece("in_proj_qkv", try XCTUnwrap(linear.linearAttention).qkvProjection(input("0.linear_attn.in_proj_qkv.in")),
                               probe, "0.linear_attn.in_proj_qkv.out"))
        let delta = try XCTUnwrap(linear.linearAttention)
        func recorded(_ key: String) throws -> MLXArray { try XCTUnwrap(probe[key], "no \(key)").asType(.bfloat16) }
        let convolved = delta.convolved(try recorded("fn.causal_conv1d_fn.0.arg0").transposed(0, 2, 1))
        lines.append(try piece("conv1d + silu", convolved.transposed(0, 2, 1), probe, "fn.causal_conv1d_fn.0.out0"))
        // The reference's `l2norm` runs inside its delta rule on float32 values, so its input is read unrounded.
        lines.append(try piece("unit norm", NFKHybridLinearAttention.unitNorm(try XCTUnwrap(probe["fn.l2norm.0.arg0"])),
                               probe, "fn.l2norm.0.out0"))
        let read = delta.recurrence(
            queries: NFKHybridLinearAttention.unitNorm(try recorded("fn.torch_chunk_gated_delta_rule.0.arg0"), widenedFirst: true),
            keys: NFKHybridLinearAttention.unitNorm(try recorded("fn.torch_chunk_gated_delta_rule.0.arg1"), widenedFirst: true),
            values: try recorded("fn.torch_chunk_gated_delta_rule.0.arg2").asType(.float32),
            decay: exp(try XCTUnwrap(probe["fn.torch_chunk_gated_delta_rule.0.arg5"])),
            write: try XCTUnwrap(probe["fn.torch_chunk_gated_delta_rule.0.arg3"]))
        lines.append(try piece("recurrence", read.asType(.bfloat16), probe, "fn.torch_chunk_gated_delta_rule.0.out0"))
        let recordedRead = try recorded("fn.torch_chunk_gated_delta_rule.0.out0")
        let z = try recorded("0.linear_attn.in_proj_z.out").reshaped(recordedRead.shape)
        let wideZ = z.asType(.float32)
        let gated = (delta.norm(recordedRead).asType(.float32) * (wideZ * MLX.sigmoid(wideZ))).asType(.bfloat16)
        lines.append(try piece("gated norm", gated.reshaped([length, -1]), probe, "0.linear_attn.out_proj.in"))
        lines.append(try piece("gated norm, first row", gated.reshaped([-1, geometry.linearValueHeadDimensions])[0],
                               probe, "0.linear_attn.norm.out"))
        lines.append(try piece("out_proj", try XCTUnwrap(linear.linearAttention).outputProjection(input("0.linear_attn.out_proj.in")),
                               probe, "0.linear_attn.out_proj.out"))
        lines.append(try piece("mlp", linear.feedForward(input("0.mlp.in")), probe, "0.mlp.out"))
        lines.append(try piece("post_attention_layernorm", linear.postNorm(input("0.post_attention_layernorm.in")),
                               probe, "0.post_attention_layernorm.out"))
        let full = net.model.layers[3], attention = try XCTUnwrap(full.attention)
        lines.append("layer 3 (\(full.kind))")
        lines.append(try piece("q_norm", attention.queryNorm(input("3.self_attn.q_norm.in")), probe, "3.self_attn.q_norm.out"))
        lines.append(try piece("k_norm", attention.keyNorm(input("3.self_attn.k_norm.in")), probe, "3.self_attn.k_norm.out"))
        let rotated = NFKReferenceRounding.rotary(try input("3.self_attn.q_norm.out").transposed(0, 2, 1, 3),
                                                  dimensions: geometry.rotaryDimensions, base: geometry.ropeTheta, offset: 0)
        lines.append(try piece("rotary q", rotated[0], probe, "3.attn.q"))
        let attended = NFKReferenceRounding.attention(queries: try input("3.attn.q"), keys: try input("3.attn.k"),
                                                      values: try input("3.attn.v"),
                                                      scale: 1 / sqrt(Float(geometry.headDimensions)), mask: mask)
        lines.append(try piece("attention", attended[0].transposed(1, 0, 2), probe, "3.attn.out"))
        let normed = try input("3.input_layernorm.out")
        let paired = attention.queryProjection(normed).reshaped([1, length, geometry.headCount, geometry.headDimensions * 2])
        let q = NFKReferenceRounding.rotary(attention.queryNorm(paired[.ellipsis, 0 ..< geometry.headDimensions]).transposed(0, 2, 1, 3),
                                            dimensions: geometry.rotaryDimensions, base: geometry.ropeTheta, offset: 0)
        let k = NFKReferenceRounding.rotary(attention.keyNorm(attention.keyProjection(normed)
            .reshaped([1, length, geometry.keyValueHeadCount, geometry.headDimensions])).transposed(0, 2, 1, 3),
                                            dimensions: geometry.rotaryDimensions, base: geometry.ropeTheta, offset: 0)
        let v = attention.valueProjection(normed).reshaped([1, length, geometry.keyValueHeadCount, geometry.headDimensions])
            .transposed(0, 2, 1, 3)
        lines.append(try piece("q from the normed input", q[0], probe, "3.attn.q"))
        lines.append(try piece("k from the normed input", k[0], probe, "3.attn.k"))
        lines.append(try piece("v from the normed input", v[0], probe, "3.attn.v"))
        lines.append(try piece("self_attn", attention(normed, mask: mask), probe, "3.self_attn.out"))
        lines.append(try piece("o_proj", attention.outputProjection(input("3.self_attn.o_proj.in")), probe, "3.self_attn.o_proj.out"))
        print("VALIDATION bf16 qwen3.5-4b pieces:\n" + lines.joined(separator: "\n"))
    }

    // MARK: Qwen4-Exp

    // Qwen4-Exp's tiny configuration with the eight-head indexer at bf16 (`IK_TINY_DTYPE=bfloat16
    // IK_QWEN4_INDEXER_HEADS=8 run_reference.py qwen4_exp`), against float32 arithmetic on the same
    // bf16-rounded weights (`IK_TINY_DTYPE=bfloat16-weights`) as the floor. Twelve tokens over the indexer's
    // compression ratio of 2 against a budget of 4 make the indexer discard blocks at layer 3, and each
    // layer also runs alone on the reference's own input.
    func testQwen4ExpInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let bf16 = try record("qwen4_exp_heads8_bf16.safetensors"), f32 = try record("qwen4_exp_heads8_bf16w.safetensors")
        var configuration = NFKMLXQwen4ExpConfiguration.tiny
        configuration.indexerHeadCount = 8
        let net = NFKMLXQwen4Exp.makeNet(configuration)
        let weights = bf16.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = String(key.dropFirst(3))
            guard !NFKMLXQwen4Exp.isDropped(key: name) else { return nil }
            let (module, tensor) = NFKMLXQwen4Exp.adapted(key: name, value: value)
            return (module, tensor.dtype == .float32 ? tensor.asType(.bfloat16) : tensor)
        }
        try NFKMLXWeights.apply(weights, to: net)
        let tokens = MLXArray(try XCTUnwrap(bf16["tokens"]).asArray(Int32.self)).reshaped([1, -1])
        let states = net.hiddenStates(tokens)
        let logits = net(tokens)[0]
        eval(states + [logits])
        let labelled = states.enumerated().map { ($0.offset == 0 ? "streams" : "layer \($0.offset - 1)", $0.element) }
            + [("logits", logits)]
        let rows = try seams(labelled, bf16: bf16, f32: f32, keys: states.indices.map { "hidden.\($0)" } + ["output"])
        report("qwen4-exp", rows)
        for (index, state) in states.enumerated() {
            XCTAssertEqual(state.dtype, .bfloat16, "state \(index) stays in the released precision")
        }

        let length = tokens.dim(1)
        let positions = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, 1, length])
        let (cosTable, sinTable) = net.rotaryTable(positions: broadcast(positions, to: [3, 1, length]))
        let queryIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([length, 1])
        let keyIndex = MLXArray((0 ..< length).map { Int32($0) }).reshaped([1, length])
        let causal = (keyIndex .<= queryIndex).reshaped([1, 1, length, length])
        let ids = [tokens.asArray(Int32.self).map(Int.init)]
        let layers = net.model.languageModel.layers
        var isolated = [(String, MLXArray)]()
        for (index, layer) in layers.enumerated() {
            let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            var output = layer(input, cos: cosTable, sin: sinTable, mask: causal,
                               ngramIndices: layer.ple.map { $0.embedding.indices(tokens: ids) })
            if index == layers.count - 1 { output = net.model.languageModel.mixer(output).read }
            isolated.append(("layer \(index)", output))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolated.indices.map { "hidden.\($0 + 1)" })
        report("qwen4-exp isolated", isolatedRows)
        assertRoundingPlacement("qwen4-exp", endToEnd: rows, isolated: isolatedRows)
    }

    // MARK: Granite Speech

    // Granite Speech 3.3-2b at the released bf16, the backend's default, against transformers at bf16
    // (`IK_GRANITE_SPEECH_DTYPE=bfloat16 run_reference.py granite_speech_real`), float32 arithmetic on the
    // same bf16 features as the floor (`bfloat16-inputs`). Each Conformer and decoder piece runs on the
    // reference's own bf16 input (`IK_PROBE_ENCODER=1`, `IK_PROBE_LAYERS`), each floor from the matching
    // float32 probe; the whole pipeline is then held to float32.
    func testGraniteSpeechInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GRANITE_SPEECH"], "IK_VAL_GRANITE_SPEECH"))
        let bf16 = try record("granite_speech_bf16.safetensors"), f32 = try record("granite_speech_bf16in.safetensors")
        let folded = try record("granite_speech_bf16_folded.safetensors")
        let probe = try record("granite_speech_probe.safetensors"), exactProbe = try record("granite_speech_probe_f32.safetensors")
        let net = try NFKMLXGraniteSpeech.net(fromDirectory: directory)
        try NFKMLXGraniteSpeech.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        func input(_ arrays: [String: MLXArray], _ key: String) throws -> MLXArray {
            try XCTUnwrap(arrays[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let features = try input(bf16, "features")
        let tokens = try XCTUnwrap(bf16["tokens"]).asType(.int32).reshaped([1, -1])

        // Each Conformer piece of every block on the reference's own input. A whole block is not an
        // isolated seam here: a one-ulp GEMM-order difference in the pointwise up-convolution spreads
        // across its frame through the pointwise down-convolution.
        var pieces = [(String, MLXArray)](), pieceKeys = [String]()
        for (index, block) in net.encoder.layers.enumerated() {
            let p = "enc.\(index)."
            pieces += [("\(index) ff1", block.feedForward1(try input(probe, p + "ff1.in"))),
                       ("\(index) attn", block.attention(try input(probe, p + "attn.in"))),
                       ("\(index) conv", block.convolution(try input(probe, p + "conv.in"))),
                       ("\(index) ff2", block.feedForward2(try input(probe, p + "ff2.in")))]
            pieceKeys += ["ff1", "attn", "conv", "ff2"].map { p + $0 + ".out" }
        }
        eval(pieces.map(\.1))
        let pieceRows = try seams(pieces, bf16: probe, f32: exactProbe, keys: pieceKeys)

        // The decoder's pieces, the release's float32 LoRA folded into the weights at load as the port
        // folds it (`IK_GRANITE_SPEECH_DTYPE=bfloat16-folded IK_PROBE_LAYERS=0,1,39`).
        let decoderProbe = try record("granite_speech_decoder_probe.safetensors")
        let decoderExact = try record("granite_speech_decoder_probe_f32.safetensors")
        var decoderPieces = [(String, MLXArray)](), decoderKeys = [String]()
        for index in [0, 1, 39] {
            let block = net.languageModel.model.layers[index], p = "dec.\(index)."
            decoderPieces += [
                ("dec \(index) input_norm", block.inputNorm(try input(decoderProbe, p + "input_layernorm.in"))),
                ("dec \(index) self_attn", block.attention(try input(decoderProbe, p + "input_layernorm.out"))),
                ("dec \(index) post_norm", block.postNorm(try input(decoderProbe, p + "post_attention_layernorm.in"))),
                ("dec \(index) mlp", block.mlp(try input(decoderProbe, p + "mlp.in")))]
            decoderKeys += ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"].map { p + $0 + ".out" }
        }
        let projected = net.projector(try input(bf16, "encoder_out"))[0]
        eval(decoderPieces.map(\.1), projected)
        let isolated = try seams(decoderPieces, bf16: decoderProbe, f32: decoderExact, keys: decoderKeys)
            + (try seams([("projector", projected)], bf16: bf16, f32: f32, keys: ["projector_out"]))

        // End to end. The decoder amplifies a one-ulp difference about tenfold a layer from layer 1 on,
        // so a GEMM-order difference grows to the size of the floor and ours-vs-reference cannot
        // discriminate there; ours-vs-float32 can.
        let decoded = net.logits(tokens: tokens, audioEmbeddings: try input(folded, "projector_out"))[0]
        let encoded = net.encoder(features)
        let logits = net.logits(tokens: tokens, audioEmbeddings: net.projector(encoded))[0]
        eval(decoded, encoded, logits)
        let endToEnd = try seams([("encoder", encoded[0])], bf16: bf16, f32: f32, keys: ["encoder_out"])
            + (try seams([("decoder (folded)", decoded)], bf16: folded, f32: f32, keys: ["output"]))
            + (try seams([("logits", logits)], bf16: bf16, f32: f32, keys: ["output"]))
        report("granite-speech conformer pieces", pieceRows)
        report("granite-speech isolated", isolated)
        report("granite-speech end to end", endToEnd)
        assertRoundingPlacement("granite-speech", endToEnd: endToEnd, isolated: pieceRows + isolated)
    }

    // MARK: Voxtral

    // Voxtral-Mini-3B at the released bf16, the backend's default, against transformers at bf16 with eager
    // attention (`IK_VOXTRAL_DTYPE=bfloat16 IK_PROBE_ENCODER=1 IK_PROBE_LAYERS=0,1,29 run_reference.py
    // voxtral_real`), float32 arithmetic on the same bf16 features as the floor (`bfloat16-inputs`). Each
    // Whisper-encoder and decoder piece runs on the reference's own bf16 input; the pipeline is then held
    // to float32.
    func testVoxtralInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_VOXTRAL"], "IK_VAL_VOXTRAL"))
        let bf16 = try record("voxtral_bf16.safetensors"), f32 = try record("voxtral_bf16in.safetensors")
        let net = try NFKMLXVoxtral.net(fromDirectory: directory)
        try NFKMLXVoxtral.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        func input(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }

        var pieces = [(String, MLXArray)](), keys = [String]()
        for (index, block) in net.audioTower.blocks.enumerated() {
            let p = "enc.\(index)."
            let activation = block.mlp[1] as! GELU
            let feed = (block.mlp[2] as! Linear)(NFKReferenceRounding.wide((block.mlp[0] as! Linear)(
                try input(p + "final_layer_norm.out"))) { activation($0) })
            pieces += [("enc \(index) attn_ln", block.attnLN(try input(p + "self_attn_layer_norm.in"))),
                       ("enc \(index) attn", block.attn(try input(p + "self_attn_layer_norm.out"), source: nil, mask: nil)),
                       ("enc \(index) mlp_ln", block.mlpLN(try input(p + "final_layer_norm.in"))),
                       ("enc \(index) mlp", feed)]
            keys += ["self_attn_layer_norm", "self_attn", "final_layer_norm", "fc2"].map { p + $0 + ".out" }
        }
        for index in [0, 1, 29] {
            let block = net.languageModel.model.layers[index], p = "dec.\(index)."
            pieces += [("dec \(index) input_norm", block.inputNorm(try input(p + "input_layernorm.in"))),
                       ("dec \(index) self_attn", block.attention(try input(p + "input_layernorm.out"))),
                       ("dec \(index) post_norm", block.postNorm(try input(p + "post_attention_layernorm.in"))),
                       ("dec \(index) mlp", block.mlp(try input(p + "mlp.in")))]
            keys += ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"].map { p + $0 + ".out" }
        }
        let encoderOut = try input("encoder_out")
        pieces.append(("projector", net.projector(encoderOut.reshaped([-1, net.config.projectorInputSize]))))
        keys.append("audio_embeds")
        eval(pieces.map(\.1))
        let isolated = try seams(pieces, bf16: bf16, f32: f32, keys: keys)

        // The record's features are [mels, frames]; the encoder reads [batch, frames, mels].
        let features = try input("features").transposed(0, 2, 1)
        let tokens = try XCTUnwrap(bf16["tokens"]).asType(.int32).reshaped([1, -1])
        let encoded = net.audioTower(features)
        let logits = net.logits(tokens: tokens, audioEmbeddings: net.projector(encoded.reshaped([-1, net.config.projectorInputSize])))[0]
        eval(encoded, logits)
        let endToEnd = try seams([("encoder", encoded[0]), ("logits", logits)], bf16: bf16, f32: f32,
                                 keys: ["encoder_out", "output"])
        report("voxtral isolated", isolated)
        report("voxtral end to end", endToEnd)
        assertRoundingPlacement("voxtral", endToEnd: endToEnd, isolated: isolated)
    }

    // MARK: Phi-4-multimodal

    // Phi-4-multimodal against its release's remote code at bf16 with eager attention
    // (`IK_PHI4MM_DTYPE=bfloat16 run_reference.py phi4mm_bf16`, the `phi4mm` oracle environment), float32
    // arithmetic on the same bf16 inputs as the floor (`bfloat16-inputs`). The backend loads the decoder
    // at the release's bf16 with its mixture of LoRAs beside the base and the two towers at float32, so
    // the decoder's pieces are held to the reference on its own input and each mode's logits, from the
    // float32 towers, to float32. The towers' own half-precision paths are measured piece by piece too,
    // cast to bf16.
    func testPhi4MultimodalInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_PHI4MM"], "IK_VAL_PHI4MM"))
        let bf16 = try record("phi4mm_bf16.safetensors"), f32 = try record("phi4mm_bf16in.safetensors")
        func input(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        func reduced<M: Module>(_ module: M) -> M {
            module.update(parameters: module.parameters().mapValues { $0.asType(.bfloat16) })
            return module
        }
        let model = try NFKMLXPhi4MM.model(directoryURL: directory, precision: .checkpoint)
        let decoder = model.decoder

        var pieces = [(String, MLXArray)](), keys = [String]()
        NFKMLXPhi4MM.select(.language, in: decoder)
        for index in [0, 1, 31] {
            let block = decoder.model.layers[index], p = "dec.\(index)."
            let normed = try input(p + "input_layernorm.out")
            pieces += [
                ("dec \(index) input_norm", block.attentionNorm(try input(p + "input_layernorm.in"))),
                ("dec \(index) self_attn", block.attention(normed, mask: NFKMLXLanguageNet.causalMask(normed.dim(1), offset: 0),
                                                           cache: nil, layer: index)),
                ("dec \(index) post_norm", block.feedForwardNorm(try input(p + "post_attention_layernorm.in"))),
                ("dec \(index) mlp", block.feedForward(try input(p + "mlp.in")))]
            keys += ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"].map { p + $0 + ".out" }
        }

        let audio = reduced(try NFKMLXPhi4MM.audioNet(directoryURL: directory))
        let speechLength = try input("speech.enc.0.block.in").dim(1)
        let bias = audio.encoder.t5Bias(length: speechLength)
        for index in [0, 1, 12, 23] {
            // The release wraps each Conformer layer for activation checkpointing. A whole layer is not an
            // isolated seam: the last layer's final LayerNorm turns its pieces' one-ulp GEMM-order
            // differences into 0.29 of the floor, while every piece stays under 0.01 of it.
            let layer = audio.encoder.layers[index], p = "speech.enc.\(index)._checkpoint_wrapped_module."
            pieces += [("speech \(index) ff_in", layer.feedForwardIn(try input(p + "feed_forward_in.in"))),
                       ("speech \(index) self_attn", layer.attention(try input(p + "self_attn.in"), bias: bias)),
                       ("speech \(index) conv", layer.conv(try input(p + "conv.in"))),
                       ("speech \(index) ff_out", layer.feedForwardOut(try input(p + "feed_forward_out.in"))),
                       ("speech \(index) norm", layer.norm(try input(p + "layer_norm.in")))]
            keys += ["feed_forward_in", "self_attn", "conv", "feed_forward_out", "layer_norm"].map { p + $0 + ".out" }
        }
        let image = reduced(try NFKMLXPhi4MM.imageNet(directoryURL: directory))
        for index in [0, 1, 13, 25] {
            let layer = image.processor.encoder.layers[index], p = "vision.enc.\(index)."
            pieces += [("vision \(index) self_attn", layer.attention(try input(p + "layer_norm1.out"))),
                       ("vision \(index) mlp", layer.mlp(try input(p + "mlp.in"))),
                       ("vision \(index) layer", layer(try input(p + "block.in")))]
            keys += ["self_attn", "mlp", "block"].map { p + $0 + ".out" }
        }
        eval(pieces.map(\.1))
        let isolated = try seams(pieces, bf16: bf16, f32: f32, keys: keys)

        // The backend's own path in each mode: float32 towers, the bf16 decoder with the mode's adapter.
        let textTokens = try XCTUnwrap(bf16["text_tokens"]).asType(.int32).asArray(Int32.self).map(Int.init)
        NFKMLXPhi4MM.select(.language, in: decoder)
        let text = decoder.logits(fromHidden: decoder.hiddenStates(
            fromEmbeddings: decoder.embed(MLXArray(textTokens.map(Int32.init)).reshaped([1, -1]))))[0]
        let float32Audio = try NFKMLXPhi4MM.audioNet(directoryURL: directory)
        let mel = try XCTUnwrap(bf16["speech_input_audio"]).expandedDimensions(axis: 0)
        NFKMLXPhi4MM.select(.speech, in: decoder)
        let speechTokens = try XCTUnwrap(bf16["speech_tokens"]).asType(.int32).asArray(Int32.self).map(Int.init)
        let speech = decoder.logits(fromHidden: NFKMLXPhi4MM.fusedHidden(
            decoder: decoder, inputIds: speechTokens, features: float32Audio.projected(mel, mode: .speech),
            placeholder: NFKMLXPhi4MM.audioTokenId))[0]
        eval(text, speech)
        let endToEnd = try seams([("text logits", text), ("speech logits", speech)], bf16: bf16, f32: f32,
                                 keys: ["text_logits", "speech_logits"])
        report("phi4mm isolated", isolated)
        report("phi4mm end to end", endToEnd)
        assertRoundingPlacement("phi4mm", endToEnd: endToEnd, isolated: isolated)
    }

    // MARK: MiniMax Music 3 depth decoder

    // The released bf16 depth decoder, at the precision the music backend loads it, against diffusers at
    // bf16 with its default attention (`IK_MUSIC_DEPTH_DTYPE=bfloat16 run_reference.py music_depth`),
    // float32 arithmetic on the same bf16 inputs as the floor (`bfloat16-inputs`).
    func testMusic3DepthDecoderInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let url = URL(fileURLWithPath: try existing(config["IK_VAL_MUSIC3_DEPTH"], "IK_VAL_MUSIC3_DEPTH"))
        let bf16 = try record("music_depth_bf16.safetensors"), f32 = try record("music_depth_bf16in.safetensors")
        func input(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let net = NFKMLXMusic3.makeDepthDecoder()
        try NFKMLXMusic3.loadDepthWeights(into: net, from: url, precision: .checkpoint)
        var pieces = [(String, MLXArray)](), keys = [String]()
        for (index, block) in net.layers.enumerated() {
            let p = "enc.\(index)."
            let normed = try input(p + "post_attention_layernorm.out")
            let attended = try input(p + "input_layernorm.out")
            pieces += [
                ("\(index) input_norm", block.inputNorm(try input(p + "input_layernorm.in"))),
                ("\(index) attn", block.attention(attended, mask: NFKMLXLanguageNet.causalMask(attended.dim(1), offset: 0))),
                ("\(index) post_norm", block.postNorm(try input(p + "post_attention_layernorm.in"))),
                ("\(index) mlp", block.down(NFKReferenceRounding.silu(block.gate(normed)) * block.up(normed)))]
            keys += ["input_layernorm", "attn", "post_attention_layernorm", "down_proj"].map { p + $0 + ".out" }
        }
        eval(pieces.map(\.1))
        let isolated = try seams(pieces, bf16: bf16, f32: f32, keys: keys)
        let hidden = net.hiddenStates(try XCTUnwrap(bf16["inputs_embeds"]).asType(.bfloat16))
        eval(hidden)
        let endToEnd = try seams([("hidden", hidden)], bf16: bf16, f32: f32, keys: ["output"])
        report("music3-depth isolated", isolated)
        report("music3-depth end to end", endToEnd)
        assertRoundingPlacement("music3-depth", endToEnd: endToEnd, isolated: isolated)
    }

    // MARK: Diffusion transformers

    // A tiny diffusers transformer's record pair at bf16 (`IK_DIT_DTYPE=bfloat16`) and its float32 floor
    // on the same rounded weights and inputs (`bfloat16-weights`): the `w::` weights and `inputs` read
    // at bf16. Rounding placement depends on the order of operations, not on the size, so the tiny
    // configurations the float32 parity tests use serve here.
    private func tinyDiT(_ name: String) throws -> (bf16: [String: MLXArray], f32: [String: MLXArray],
                                                    weights: [(String, MLXArray)]) {
        let bf16 = try record("\(name)_bf16.safetensors"), f32 = try record("\(name)_bf16in.safetensors")
        let weights = bf16.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value.asType(.bfloat16)) : nil
        }
        return (bf16, f32, weights)
    }

    func testFluxTransformerInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (bf16, f32, weights) = try tinyDiT("flux")
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        let net = NFKMLXFluxTransformerNet(.tiny)
        try NFKMLXWeights.apply(weights, to: net)
        let output = net(try input("hidden"), encoderHidden: try input("encoder"), pooled: try input("pooled"),
                         timestep: try input("timestep"), guidance: try input("guidance"),
                         imageIds: try XCTUnwrap(bf16["img_ids"]))
        eval(output)
        let rows = try seams([("velocity", output)], bf16: bf16, f32: f32, keys: ["output"])
        report("flux", rows)
        assertRoundingPlacement("flux", endToEnd: rows, isolated: rows)
    }

    // The SD3 MMDiT at `run_reference.py sd3`'s tiny configuration (RMS query/key norm, a dual-attention first
    // layer, the last block's `context_pre_only`), at bf16 against diffusers at bf16 (`IK_DIT_DTYPE`).
    func testSD3TransformerInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (bf16, f32, weights) = try tinyDiT("sd3")
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        let net = NFKMLXSD3TransformerNet(.tiny)
        try NFKMLXWeights.apply(weights.map { ($0.0, $0.1.ndim == 4 ? $0.1.transposed(0, 2, 3, 1) : $0.1) }, to: net)
        let output = net(try input("latent"), encoderHidden: try input("encoder"), pooled: try input("pooled"),
                         timestep: try input("timestep"))
        eval(output)
        XCTAssertEqual(output.dtype, .bfloat16, "a bf16 load runs the transformer at bf16")
        let rows = try seams([("velocity", output)], bf16: bf16, f32: f32, keys: ["output"])
        report("sd3", rows)
        assertRoundingPlacement("sd3", endToEnd: rows, isolated: rows)
    }

    // Pipeline glue against torch at bf16 and float16 (`run_reference.py glue_rounding`), element for element:
    // classifier-free guidance at a scale exact in neither type, and Z-Image's image-to-image latent
    // normalization. MLX rounds a `Float` factor to the tensor's type first, which the record tells apart.
    func testPipelineGlueRoundsAsTorchDoes() throws {
        try requireMLXRuntime()
        let record = try self.record("glue_rounding.safetensors")
        for (name, dtype) in [("bf16", DType.bfloat16), ("f16", .float16)] {
            func value(_ key: String) throws -> MLXArray {
                try XCTUnwrap(record["\(name).\(key)"], "no \(name).\(key)").asType(dtype)
            }
            let (u, c, x) = (try value("u"), try value("c"), try value("x"))
            let guided = NFKReferenceRounding.guided(u, c, scale: 3.3)
            let normalized = NFKReferenceRounding.scaled(NFKReferenceRounding.shifted(x, by: -0.1159), by: 0.3611)
            XCTAssertTrue(arrayEqual(guided, try value("guided")).item(Bool.self), "\(name): guidance as torch rounds it")
            XCTAssertTrue(arrayEqual(normalized, try value("normalized")).item(Bool.self),
                          "\(name): the latent normalization as torch rounds it")
            XCTAssertFalse(arrayEqual(u + 3.3 * (c - u), try value("guided")).item(Bool.self),
                           "\(name): the record tells a factor rounded first from torch's")
            let wideShift = NFKReferenceRounding.scaled((x.asType(.float32) - 0.1159).asType(dtype), by: 0.3611)
            XCTAssertFalse(arrayEqual(wideShift, try value("normalized")).item(Bool.self),
                           "\(name): the record tells an offset added in float32 from torch's, which rounds it first")
        }
    }

    // torch's exact GELU at bf16 and float16 over every finite value (`run_reference.py gelu_rounding`) of
    // magnitude at least twice the least normal float32, so `x · 0.5` is normal: the GPU flushes a subnormal
    // product to zero, and torch's CPU keeps it. The kernel fuses torch's multiply-adds and takes Sleef's
    // exponential, and matches every input in either arithmetic. MLX's own error function, faithfully rounded,
    // rounds hundreds apart, which the record tells from torch's.
    func testHalfGELUComputesTorchsErrorFunction() throws {
        try requireMLXRuntime()
        let record = try self.record("gelu_rounding.safetensors")
        for (name, dtype) in [("bf16", DType.bfloat16), ("f16", .float16)] {
            let input = try XCTUnwrap(record["\(name).in"]).asType(dtype)
            let expected = try XCTUnwrap(record["\(name).out"])
            let normal = abs(input.asType(.float32)) .>= 2 * Float.leastNormalMagnitude
            func differing(_ result: MLXArray) -> Int {
                logicalAnd(normal, result.asType(.float32) .!= expected).sum().item(Int.self)
            }
            XCTAssertEqual(differing(NFKReferenceRounding.gelu(input)), 0, "\(name): torch's GELU, reference arithmetic")
            NFKMLXHalfPrecision.arithmetic = .fast
            XCTAssertEqual(differing(NFKReferenceRounding.gelu(input)), 0, "\(name): torch's GELU, fast arithmetic")
            NFKMLXHalfPrecision.arithmetic = .reference
            XCTAssertGreaterThan(differing(MLXNN.gelu(input.asType(.float32)).asType(dtype)), 100,
                                 "\(name): the record tells MLX's error function from torch's")
        }
    }

    // torch's float32 `exp`, Sleef's `expf`, at x = −15i/512 for i in 0..<1024. The kernel matches every one.
    // MLX operations round each multiply-add Sleef fuses and leave 9 one unit apart; MLX's own `exp` leaves 230.
    func testReferenceArithmeticComputesTorchsExp() throws {
        try requireMLXRuntime()
        let input = MLXArray((0 ..< 1024).map { -Float($0) * 15 / 512 })
        let expected = MLXArray(Self.torchExponentialBits).view(dtype: .float32)
        func differing(_ result: MLXArray) -> Int { (result .!= expected).sum().item(Int.self) }
        XCTAssertEqual(differing(NFKReferenceRounding.torchExp(input)), 0, "the kernel")
        XCTAssertEqual(differing(NFKReferenceRounding.torchExpOperations(input)), 9, "MLX operations")
        XCTAssertGreaterThan(differing(exp(input)), 100, "the grid tells MLX's exp from torch's")
        let edges = NFKReferenceRounding.torchExp(MLXArray([-Float.infinity, -104.5, 89, Float.infinity]))
        XCTAssertEqual(edges.asArray(Float.self), [0, 0, .infinity, .infinity])
    }

    // torch's `layer_norm` over float16, bf16, and float32 rows of 320, 1,001, and 7 (whole vectors, a
    // remainder, and a row narrower than a vector), and over rows without the affine at Wan's widths, each
    // input, weight, and bias an exact formula, against the SHA-256 of torch's output. A float32 row is read
    // in 4-wide vectors and normalized as `(x − mean) · rstd`. MLX's fused kernel rounds 4 and 2 of the float16
    // outputs apart, and float32 ones throughout.
    func testReferenceArithmeticComputesTorchsLayerNorm() throws {
        try requireMLXRuntime()
        let cases: [(DType, Int, Int, Bool, String)] = [
            (.float16, 64, 320, true, "7a0dc7af5e43d2734439bc51bb2eabdd581cc1423829aceab2efe7ab36679f91"),
            (.float16, 16, 1001, true, "46d1fa7072d6e4d0922c957f307667fb2de9a90430cf769882e8ada67eace2a3"),
            (.float16, 16, 7, true, "8d7f7ee68c5d41a4f98d3140423f1d978ee1da6c556d1861cb0c315b5221f701"),
            (.bfloat16, 64, 320, true, "33fcce31ee9a6fffcd72a81c77996f73611fda9afc985f88ee9cd279ba6a8a7f"),
            (.bfloat16, 16, 1001, true, "597ba1814b5ba569c62b1fb09e64d83b7d7e2cc9d56fbdd0c705cc226960f7d4"),
            (.bfloat16, 16, 7, true, "4b8ed3d970271f4e9ca026bc2d8c1c9ce9fe771f179481043173e63eda552d1a"),
            (.float16, 16, 1001, false, "15301bf73a734808b5e973ea6949de0de25e42b7d56f1f3dd90dbbc5db31b481"),
            (.bfloat16, 16, 320, false, "3c15acdcbe5a53950cf106c5ff2317cec41b32c4b5bc54238691516873cb8156"),
            (.float32, 64, 320, true, "dda1f7af3620fc73fbd6cdd255451e433eed72fac73a105aee846ebf56b1a92b"),
            (.float32, 16, 1001, true, "e0302a6e097bfc534a032a7d7c748be8811ea055ea5e145ee5077d8b6fdea4b4"),
            (.float32, 16, 7, true, "05e10810a0c4bfd403c99284817fb59c110daf416f4d2e0b82a4bb06eaae00bb"),
            (.float32, 16, 1536, false, "1f723bc7214fef0eb95263fe375095a758749038a32cc462e729e382e1fbf0f1"),
            (.float32, 4, 5120, false, "87dc8d7998ffe530cc67f1515acbf87488396790a777b5196b0e870032790637"),
            (.float32, 16, 7, false, "74cf7103585739cae4f4dda26d68bd7e2deafa42e8d61072970764926efea6ee"),
        ]
        func digest(_ y: MLXArray) -> String {
            let words = y.view(dtype: .uint16).asArray(UInt16.self)
            return SHA256.hash(data: words.withUnsafeBufferPointer { Data(buffer: $0) })
                .map { String(format: "%02x", $0) }.joined()
        }
        for (dtype, rows, width, affine, expected) in cases {
            let (modulus, scale) = dtype == .bfloat16 ? (511, Float(64)) : (2001, Float(256))
            let x = MLXArray((0 ..< rows * width).map { index -> Float in
                let (row, column) = index.quotientAndRemainder(dividingBy: width)
                return Float((row * 131 + column * 37) % modulus - modulus / 2) / scale
            }, [rows, width]).asType(dtype)
            let weight = affine ? MLXArray((0 ..< width).map { 1 + Float(($0 * 29) % 33 - 16) / 64 }).asType(dtype) : nil
            let bias = affine ? MLXArray((0 ..< width).map { Float(($0 * 17) % 41 - 20) / 128 }).asType(dtype) : nil
            let label = "\(dtype) \(rows)×\(width)\(affine ? "" : " without the affine")"
            XCTAssertEqual(digest(NFKReferenceRounding.torchLayerNorm(x, weight: weight, bias: bias, eps: 1e-5)),
                           expected, label)
            if dtype == .float32 || (dtype == .float16 && affine), width > 8 {
                let fused = MLXFast.layerNorm(x.asType(.float32), weight: weight?.asType(.float32),
                                              bias: bias?.asType(.float32), eps: 1e-5).asType(dtype)
                XCTAssertNotEqual(digest(fused), expected, "\(label): the case tells MLX's kernel from torch's")
            }
        }
    }

    func testReferenceKernelsTakeTheGradientsOfTheirOperations() throws {
        try requireMLXRuntime()
        let x = MLXArray((0 ..< 64).map { -Float($0) / 4 })
        XCTAssertEqual(grad({ NFKReferenceRounding.torchExp($0).sum() })(x).asArray(Float.self),
                       exp(x).asArray(Float.self))
        MLXRandom.seed(5)
        let hidden = MLXRandom.normal([8, 320]) * 3, weight = MLXRandom.normal([320]) * 0.2 + 1
        let probe = MLXRandom.normal([8, 320])
        func loss(_ norm: @escaping (MLXArray) -> MLXArray) -> MLXArray {
            grad({ (norm($0.asType(.float16)).asType(.float32) * probe).sum() })(hidden)
        }
        let kernel = loss { NFKReferenceRounding.torchLayerNorm($0, weight: weight.asType(.float16), bias: nil, eps: 1e-5) }
        let fused = loss { MLXFast.layerNorm($0.asType(.float32), weight: weight, bias: nil, eps: 1e-5).asType(.float16) }
        XCTAssertTrue(allClose(kernel, fused, rtol: 1e-2, atol: 1e-3).item(Bool.self))
        let scores = MLXRandom.normal([8, 77]) * 3, weights = MLXRandom.normal([8, 77])
        XCTAssertTrue(allClose(grad({ (NFKReferenceRounding.torchSoftmax($0) * weights).sum() })(scores),
                               grad({ (softmax($0, axis: -1, precise: true) * weights).sum() })(scores)).item(Bool.self))
    }

    // torch's float32 `softmax` over 8 rows of 1, 3, 5, 77 (keys 10 to 29 masked), and 1,025 values, each an
    // exact formula, against the SHA-256 of torch's output. MLX's softmax sums apart from torch's on every
    // row longer than one.
    func testReferenceArithmeticComputesTorchsSoftmax() throws {
        try requireMLXRuntime()
        let cases: [(Int, String)] = [
            (1, "4f05df05f8da9356bd66425351f166a5f7305b7c805e6900d9ec0f50790c9886"),
            (3, "94e3c5c96e7c974d7d3cb675eee28e43b828cc79c02b8eccda49fdb6cc5ffdbf"),
            (5, "c3cca8329789fdca0459ef472ac1862789004d40edcb318647a3f300bfd799de"),
            (77, "91d51ba97edd0e36f6b4f96ca39f36dc3101275579229c39763ba7b36bb3f724"),
            (1025, "6b0dc5a369af6ff3e6ac72c4c38cfcb0cab3a0d4f52e7951e22e6102897e0fff"),
        ]
        func digest(_ y: MLXArray) -> String {
            let values = y.asArray(Float.self)
            return SHA256.hash(data: values.withUnsafeBufferPointer { Data(buffer: $0) })
                .map { String(format: "%02x", $0) }.joined()
        }
        for (size, expected) in cases {
            let values = (0 ..< 8 * size).map { index -> Float in
                let (row, column) = index.quotientAndRemainder(dividingBy: size)
                if size == 77, (10 ..< 30).contains(column) { return -.infinity }
                return Float((row * 131 + column * 37) % 2001 - 1000) / 128
            }
            let x = MLXArray(values, [8, size])
            XCTAssertEqual(digest(NFKReferenceRounding.torchSoftmax(x)), expected, "\(size)")
            if size > 1 {
                XCTAssertNotEqual(digest(softmax(x, axis: -1, precise: true)), expected,
                                  "\(size): the row tells MLX's softmax from torch's")
            }
        }
    }

    // torch's `silu` over every finite float16 and bf16 input whose magnitude is at least twice the least
    // normal float32 (the GPU flushes a subnormal product to zero), against the SHA-256 of torch's output, in
    // either arithmetic. MLX's `x · sigmoid(x)` rounds some of them apart in each type.
    func testReferenceArithmeticComputesTorchsSiLU() throws {
        try requireMLXRuntime()
        let cases: [(DType, String)] = [
            (.float16, "1566bbfe43d1dd25f74708d8780406b0a0b5ed23ae5f7c0a9d183a6b4d071733"),
            (.bfloat16, "85668a9aed34df2a68e6b64c6eb2339347e707779feb7c4f8115a10e4ba15d7b"),
        ]
        for (dtype, expected) in cases {
            let input = MLXArray((0 ..< 1 << 16).map { UInt16($0) }).view(dtype: dtype)
            let magnitudes = abs(input.asType(.float32)).asArray(Float.self)
            let kept = magnitudes.indices.filter {
                magnitudes[$0].isFinite && magnitudes[$0] >= 2 * Float.leastNormalMagnitude
            }
            func digest(_ y: MLXArray) -> String {
                let words = y.view(dtype: .uint16).asArray(UInt16.self)
                let selected = kept.map { words[$0] }
                return SHA256.hash(data: selected.withUnsafeBufferPointer { Data(buffer: $0) })
                    .map { String(format: "%02x", $0) }.joined()
            }
            XCTAssertEqual(digest(NFKReferenceRounding.silu(input)), expected, "\(dtype)")
            NFKMLXHalfPrecision.arithmetic = .fast
            XCTAssertEqual(digest(NFKReferenceRounding.silu(input)), expected, "\(dtype), fast arithmetic")
            NFKMLXHalfPrecision.arithmetic = .reference
            let wide = input.asType(.float32)
            XCTAssertNotEqual(digest((wide * MLX.sigmoid(wide)).asType(dtype)), expected,
                              "\(dtype): the domain tells MLX's SiLU from torch's")
        }
    }

    // Wan 2.1 1.3B's 12 heads over 33 frames at 480×832 (14,040 tokens) attend 1,593 query rows at a time,
    // and SD 1.5's 8 heads over 4,096 positions at batch 2 attend at once.
    func testFlashAttentionBlocksItsQueriesWithinOneGibibyte() {
        XCTAssertEqual(NFKReferenceRounding.automaticQueryBlock(queryShape: [1, 12, 14040, 128], keyLength: 14040), 1593)
        XCTAssertNil(NFKReferenceRounding.automaticQueryBlock(queryShape: [2, 8, 4096, 40], keyLength: 4096))
    }

    // The switch selects flash attention's exponential, a half-precision layer norm's arithmetic, and the
    // shared softmax's.
    func testHalfPrecisionArithmeticSelectsTheReferenceKernels() throws {
        try requireMLXRuntime()
        XCTAssertEqual(NFKMLXHalfPrecision.arithmetic, .reference)
        MLXRandom.seed(3)
        let (queries, keys, values) = ((MLXRandom.normal([1, 4, 1024, 64]) * 2).asType(.float16),
                                       (MLXRandom.normal([1, 4, 1024, 64]) * 2).asType(.float16),
                                       MLXRandom.normal([1, 4, 1024, 64]).asType(.float16))
        let hidden = (MLXRandom.normal([512, 320]) * 3).asType(.float16)
        let norm = NFKLayerNorm(dimensions: 320)
        let scores = MLXRandom.normal([64, 1025]) * 4
        func run() -> [MLXArray] {
            [NFKReferenceRounding.flashAttention(queries: queries, keys: keys, values: values, scale: 0.125, mask: nil),
             norm(hidden), NFKReferenceRounding.softmaxProbabilities(scores), NFKReferenceRounding.exponential(scores),
             NFKReferenceRounding.layerNorm(hidden.asType(.float32), weight: nil, bias: nil, eps: 1e-6)]
        }
        let reference = run()
        NFKMLXHalfPrecision.arithmetic = .fast
        let fast = run()
        for (label, index) in [("flash attention", 0), ("layer norm", 1), ("softmax", 2), ("exponential", 3),
                               ("float32 layer norm", 4)] {
            XCTAssertGreaterThan((fast[index] .!= reference[index]).sum().item(Int.self), 0, label)
        }
    }

    private static let torchExponentialBits: [UInt32] = [
        0x3f800000, 0x3f789bda, 0x3f716e56, 0x3f6a75de, 0x3f63b0ed, 0x3f5d1e04, 0x3f56bbb1, 0x3f50888e,
        0x3f4a833e, 0x3f44aa6e, 0x3f3efcd6, 0x3f397935, 0x3f341e56, 0x3f2eeb0b, 0x3f29de30, 0x3f24f6a9,
        0x3f203362, 0x3f1b934e, 0x3f17156a, 0x3f12b8ba, 0x3f0e7c47, 0x3f0a5f23, 0x3f066068, 0x3f027f34,
        0x3efd7558, 0x3ef623fc, 0x3eef08b5, 0x3ee821f6, 0x3ee16e38, 0x3edaec04, 0x3ed499ec, 0x3ece768b,
        0x3ec88088, 0x3ec2b695, 0x3ebd176a, 0x3eb7a1ce, 0x3eb2548b, 0x3ead2e78, 0x3ea82e73, 0x3ea35362,
        0x3e9e9c36, 0x3e9a07e4, 0x3e95956b, 0x3e9143d1, 0x3e8d1222, 0x3e88ff73, 0x3e850adf, 0x3e813387,
        0x3e7af126, 0x3e73b263, 0x3e6ca92d, 0x3e65d3f7, 0x3e5f3143, 0x3e58bf9a, 0x3e527d92, 0x3e4c69cb,
        0x3e4682ef, 0x3e40c7b2, 0x3e3b36d2, 0x3e35cf14, 0x3e308f4b, 0x3e2b764e, 0x3e2682fe, 0x3e21b446,
        0x3e1d0916, 0x3e188067, 0x3e14193c, 0x3e0fd29b, 0x3e0bab96, 0x3e07a341, 0x3e03b8ba, 0x3dffd64b,
        0x3df8735a, 0x3df14700, 0x3dea4fac, 0x3de38bd5, 0x3ddcf9fe, 0x3dd698b6, 0x3dd06695, 0x3dca6240,
        0x3dc48a64, 0x3dbeddb8, 0x3db95afe, 0x3db400fe, 0x3daece8c, 0x3da9c284, 0x3da4dbc9, 0x3da01948,
        0x3d9b79f6, 0x3d96fccd, 0x3d92a0d3, 0x3d8e6510, 0x3d8a4899, 0x3d864a84, 0x3d8269f1, 0x3d7d4c0e,
        0x3d75fbe2, 0x3d6ee1c4, 0x3d67fc24, 0x3d61497f, 0x3d5ac85a, 0x3d547749, 0x3d4e54e8, 0x3d485fde,
        0x3d4296dc, 0x3d3cf89c, 0x3d3783e4, 0x3d32377e, 0x3d2d1241, 0x3d28130c, 0x3d2338c7, 0x3d1e825f,
        0x3d19eecc, 0x3d157d0d, 0x3d112c26, 0x3d0cfb27, 0x3d08e922, 0x3d04f532, 0x3d011e7b, 0x3cfac845,
        0x3cf38ab0, 0x3cec829f, 0x3ce5ae86, 0x3cdf0ce6, 0x3cd89c4a, 0x3cd25b47, 0x3ccc487e, 0x3cc66298,
        0x3cc0a84a, 0x3cbb1852, 0x3cb5b176, 0x3cb07288, 0x3cab5a60, 0x3ca667de, 0x3ca199ee, 0x3c9cef80,
        0x3c98678f, 0x3c94011b, 0x3c8fbb2d, 0x3c8b94d5, 0x3c878d28, 0x3c83a345, 0x3c7fac9d, 0x3c784ae0,
        0x3c711fb2, 0x3c6a2980, 0x3c6366c3, 0x3c5cd5fe, 0x3c5675c0, 0x3c5044a2, 0x3c4a4148, 0x3c446a60,
        0x3c3ebea0, 0x3c393ccc, 0x3c33e3ab, 0x3c2eb212, 0x3c29a6dc, 0x3c24c0ee, 0x3c1fff33, 0x3c1b60a1,
        0x3c16e434, 0x3c1288f0, 0x3c0e4dde, 0x3c0a3211, 0x3c0634a3, 0x3c0254b2, 0x3bfd22ca, 0x3bf5d3cf,
        0x3beebada, 0x3be7d659, 0x3be124cb, 0x3bdaa4b6, 0x3bd454ac, 0x3bce334b, 0x3bc83f3a, 0x3bc27729,
        0x3bbcd9d4, 0x3bb765fe, 0x3bb21a74, 0x3bacf60f, 0x3ba7f7aa, 0x3ba31e30, 0x3b9e688c, 0x3b99d5b8,
        0x3b9564b2, 0x3b911480, 0x3b8ce42f, 0x3b88d2d4, 0x3b84df89, 0x3b810972, 0x3b7a9f6a, 0x3b736302,
        0x3b6c5c17, 0x3b65891b, 0x3b5ee890, 0x3b587900, 0x3b523902, 0x3b4c2736, 0x3b464246, 0x3b4088e6,
        0x3b3af9d6, 0x3b3593dc, 0x3b3055c9, 0x3b2b3e75, 0x3b264cc2, 0x3b217f9a, 0x3b1cd5ef, 0x3b184ebb,
        0x3b13e8ff, 0x3b0fa3c3, 0x3b0b7e17, 0x3b077713, 0x3b038dd3, 0x3aff82f6, 0x3af8226d, 0x3af0f86a,
        0x3aea035a, 0x3ae341b7, 0x3adcb204, 0x3ad652cf, 0x3ad022b4, 0x3aca2054, 0x3ac44a60, 0x3abe9f8d,
        0x3ab91e9e, 0x3ab3c65c, 0x3aae959c, 0x3aa98b38, 0x3aa4a616, 0x3a9fe522, 0x3a9b4751, 0x3a96cb9f,
        0x3a927110, 0x3a8e36af, 0x3a8a1b8e, 0x3a861ec6, 0x3a823f76, 0x3a7cf98c, 0x3a75abc3, 0x3a6e93f5,
        0x3a67b094, 0x3a61001d, 0x3a5a8117, 0x3a543215, 0x3a4e11b3, 0x3a481e9a, 0x3a42577a, 0x3a3cbb0f,
        0x3a37481c, 0x3a31fd70, 0x3a2cd9e1, 0x3a27dc4e, 0x3a23039d, 0x3a1e4ebe, 0x3a19bca8, 0x3a154c5c,
        0x3a10fcdd, 0x3a0ccd3b, 0x3a08bc89, 0x3a04c9e4, 0x3a00f46c, 0x39fa7695, 0x39f33b5c, 0x39ec3595,
        0x39e563b6, 0x39dec440, 0x39d855bc, 0x39d216c3, 0x39cc05f4, 0x39c621f9, 0x39c06989, 0x39badb60,
        0x39b57648, 0x39b0390e, 0x39ab228f, 0x39a631aa, 0x39a1654a, 0x399cbc62, 0x399835eb, 0x3993d0e6,
        0x398f8c5c, 0x398b675e, 0x39876101, 0x39837864, 0x397f5956, 0x3977fa00, 0x3970d128, 0x3969dd3a,
        0x39631cb1, 0x395c8e0f, 0x39562fe5, 0x395000cb, 0x3949ff66, 0x39442a65, 0x393e807f, 0x39390075,
        0x3933a912, 0x392e792a, 0x39296f99, 0x39248b43, 0x391fcb16, 0x391b2e05, 0x3916b30e, 0x39125935,
        0x390e1f84, 0x390a050e, 0x390608ec, 0x39022a3e, 0x38fcd056, 0x38f583bd, 0x38ee6d17, 0x38e78ad6,
        0x38e0db75, 0x38da5d7e, 0x38d40f83, 0x38cdf021, 0x38c7fe00, 0x38c237d1, 0x38bc9c50, 0x38b72a40,
        0x38b1e072, 0x38acbdb8, 0x38a7c0f5, 0x38a2e90e, 0x389e34f4, 0x3899a39d, 0x38953409, 0x3890e53e,
        0x388cb64b, 0x3888a643, 0x3884b442, 0x3880df6a, 0x387a4dc7, 0x387313bc, 0x386c0f1a, 0x38653e57,
        0x385e9ff5, 0x3858327e, 0x3851f488, 0x384be4b6, 0x384601b2, 0x38404a30, 0x383abcf0, 0x383558b8,
        0x38301c59, 0x382b06ae, 0x38261697, 0x38214b00, 0x381ca2d9, 0x38181d1f, 0x3813b8d1, 0x380f74f9,
        0x380b50a8, 0x38074af3, 0x380362f9, 0x37ff2fbc, 0x37f7d19a, 0x37f0a9ec, 0x37e9b721, 0x37e2f7b1,
        0x37dc6a21, 0x37d60d00, 0x37cfdee8, 0x37c9de7e, 0x37c40a70, 0x37be6176, 0x37b8e252, 0x37b38bce,
        0x37ae5cbe, 0x37a953ff, 0x37a47075, 0x379fb10e, 0x379b14bd, 0x37969a81, 0x3792415d, 0x378e085c,
        0x3789ee92, 0x3785f316, 0x3782150a, 0x377ca726, 0x37755bbd, 0x376e463f, 0x3767651d, 0x3760b6d3,
        0x375a39eb, 0x3753ecf7, 0x374dce94, 0x3747dd6b, 0x3742182d, 0x373c7d96, 0x37370c6a, 0x3731c377,
        0x372ca194, 0x3727a5a0, 0x3722ce84, 0x371e1b2d, 0x37198a95, 0x37151bba, 0x3710cda3, 0x370c9f5e,
        0x37089000, 0x37049ea3, 0x3700ca6b, 0x36fa2500, 0x36f2ec22, 0x36ebe8a5, 0x36e518ff, 0x36de7bb0,
        0x36d80f45, 0x36d1d254, 0x36cbc37f, 0x36c5e170, 0x36c02adc, 0x36ba9e84, 0x36b53b2c, 0x36afffa8,
        0x36aaead1, 0x36a5fb88, 0x36a130b9, 0x369c8955, 0x36980457, 0x3693a0c0, 0x368f5d9a, 0x368b39f5,
        0x368734e8, 0x36834d92, 0x367f062a, 0x3677a93a, 0x367082b7, 0x3669910e, 0x3662d2b7, 0x365c4638,
        0x3655ea21, 0x364fbd0b, 0x3649bd9b, 0x3643ea80, 0x363e4272, 0x3638c433, 0x36336e8e, 0x362e4056,
        0x36293869, 0x362455ab, 0x361f970a, 0x361afb7a, 0x361681f8, 0x36122989, 0x360df139, 0x3609d819,
        0x3605dd44, 0x3601ffd9, 0x35fc7dfd, 0x35f533c5, 0x35ee1f6e, 0x35e73f6a, 0x35e09238, 0x35da165e,
        0x35d3ca70, 0x35cdad0d, 0x35c7bcdc, 0x35c1f88e, 0x35bc5ee0, 0x35b6ee98, 0x35b1a681, 0x35ac8574,
        0x35a78a51, 0x35a2b3fe, 0x359e016b, 0x35997192, 0x35950370, 0x3590b60c, 0x358c8875, 0x358879c0,
        0x35848908, 0x3580b570, 0x3579fc40, 0x3572c48f, 0x356bc236, 0x3564f3ac, 0x355e5771, 0x3557ec12,
        0x3551b026, 0x354ba24c, 0x3545c133, 0x35400b8e, 0x353a801c, 0x35351da6, 0x352fe2fc, 0x352acef9,
        0x3525e07e, 0x35211676, 0x351c6fd4, 0x3517eb93, 0x351388b3, 0x350f463f, 0x350b2347, 0x35071ee2,
        0x3503382d, 0x34fedc9e, 0x34f780e2, 0x34f05b88, 0x34e96b00, 0x34e2adc3, 0x34dc2256, 0x34d5c747,
        0x34cf9b33, 0x34c99cbd, 0x34c3ca95, 0x34be2373, 0x34b8a619, 0x34b35152, 0x34ae23f3, 0x34a91cd7,
        0x34a43ae5, 0x349f7d0a, 0x349ae23a, 0x34966973, 0x349211ba, 0x348dda19, 0x3489c1a4, 0x3485c775,
        0x3481eaab, 0x347c54db, 0x34750bd2, 0x346df8a3, 0x346719be, 0x34606da2, 0x3459f2d6, 0x3453a7ef,
        0x344d8b8b, 0x34479c51, 0x3441d8f4, 0x343c4030, 0x3436d0ca, 0x34318990, 0x342c695a, 0x34276f06,
        0x3422997c, 0x341de7ae, 0x34195892, 0x3414eb29, 0x34109e79, 0x340c7190, 0x34086384, 0x34047371,
        0x3400a078, 0x33f9d386, 0x33f29d02, 0x33eb9bce, 0x33e4ce5f, 0x33de3338, 0x33d7c8e5, 0x33d18dfc,
        0x33cb8120, 0x33c5a0fc, 0x33bfec45, 0x33ba61ba, 0x33b50024, 0x33afc654, 0x33aab325, 0x33a5c578,
        0x33a0fc38, 0x339c5658, 0x3397d2d3, 0x339370aa, 0x338f2ee8, 0x338b0c9c, 0x338708de, 0x338322cd,
        0x337eb318, 0x3377588f, 0x33703460, 0x336944fa, 0x336288d5, 0x335bfe79, 0x3355a474, 0x334f7960,
        0x33497be5, 0x3343aaaf, 0x333e0479, 0x33388804, 0x3333341b, 0x332e0794, 0x3329014a, 0x33242024,
        0x331f630e, 0x331ac8fe, 0x331650f2, 0x3311f9ee, 0x330dc2fd, 0x3309ab33, 0x3305b1a9, 0x3301d581,
        0x32fc2bbf, 0x32f4e3e6, 0x32edd1de, 0x32e6f418, 0x32e04912, 0x32d9cf54, 0x32d38574, 0x32cd6a0e,
        0x32c77bcc, 0x32c1b960, 0x32bc2185, 0x32b6b302, 0x32b16ca4, 0x32ac4d42, 0x32a753be, 0x32a27eff,
        0x329dcdf4, 0x32993f97, 0x3294d2e6, 0x329086e9, 0x328c5aaf, 0x32884d4c, 0x32845ddd, 0x32808b83,
        0x3279aad3, 0x3272757b, 0x326b756b, 0x3264a919, 0x325e0f05, 0x3257a5be, 0x32516bd9, 0x324b5ff9,
        0x324580c9, 0x323fcd00, 0x323a435d, 0x3234e2a8, 0x322fa9b2, 0x322a9756, 0x3225aa76, 0x3220e1fe,
        0x321c3ce0, 0x3217ba17, 0x321358a5, 0x320f1794, 0x320af5f5, 0x3206f2de, 0x32030d70, 0x31fe899a,
        0x31f73043, 0x31f00d3e, 0x31e91ef9, 0x31e263ee, 0x31dbdaa2, 0x31d581a6, 0x31cf5794, 0x31c95b12,
        0x31c38acf, 0x31bde584, 0x31b869f4, 0x31b316ea, 0x31adeb3a, 0x31a8e5c2, 0x31a40567, 0x319f4916,
        0x319aafc7, 0x31963875, 0x3191e226, 0x318dabe5, 0x318994c5, 0x31859be2, 0x3181c05a, 0x317c02aa,
        0x3174bc01, 0x316dab1f, 0x3166ce78, 0x31602488, 0x3159abd8, 0x315362fe, 0x314d4898, 0x31475b4c,
        0x314199d0, 0x313c02df, 0x3136953e, 0x31314fbc, 0x312c3131, 0x3127387c, 0x31226486, 0x311db43f,
        0x3119269f, 0x3114baa7, 0x31106f5e, 0x310c43d1, 0x31083717, 0x3104484c, 0x31007692, 0x30f98226,
        0x30f24dfb, 0x30eb4f0f, 0x30e483d8, 0x30ddead8, 0x30d7829c, 0x30d149ba, 0x30cb3ed7, 0x30c5609c,
        0x30bfadc1, 0x30ba2504, 0x30b4c530, 0x30af8d14, 0x30aa7b8b, 0x30a58f79, 0x30a0c7c8, 0x309c236c,
        0x3097a15f, 0x309340a4, 0x308f0044, 0x308adf51, 0x3086dce2, 0x3082f816, 0x307e6022, 0x307707fe,
        0x306fe623, 0x3068f8fe, 0x30623f0c, 0x305bb6d0, 0x30555edd, 0x304f35cc, 0x30493a44, 0x30436af4,
        0x303dc694, 0x30384be8, 0x3032f9bc, 0x302dcee4, 0x3028ca3e, 0x3023eaae, 0x301f2f23, 0x301a9694,
        0x30161ffc, 0x3011ca61, 0x300d94d0, 0x30097e5b, 0x3005861d, 0x3001ab36, 0x2ffbd99c, 0x2ff49422,
        0x2fed8467, 0x2fe6a8de, 0x2fe00004, 0x2fd98862, 0x2fd3408e, 0x2fcd2726, 0x2fc73ad2, 0x2fc17a46,
        0x2fbbe43e, 0x2fb67780, 0x2fb132d9, 0x2fac1523, 0x2fa71d3e, 0x2fa24a11, 0x2f9d9a8e, 0x2f990dac,
        0x2f94a26c, 0x2f9057d6, 0x2f8c2cf8, 0x2f8820e6, 0x2f8432bf, 0x2f8061a5, 0x2f795980, 0x2f722682,
        0x2f6b28ba, 0x2f645e9e, 0x2f5dc6b1, 0x2f575f80, 0x2f5127a2, 0x2f4b1dba, 0x2f454074, 0x2f3f8e87,
        0x2f3a06b2, 0x2f34a7bc, 0x2f2f707a, 0x2f2a5fc5, 0x2f257480, 0x2f20ad96, 0x2f1c09fc, 0x2f1788ab,
        0x2f1328a7, 0x2f0ee8f8, 0x2f0ac8b2, 0x2f06c6ea, 0x2f02e2c0, 0x2efe36b1, 0x2ef6dfc0, 0x2eefbf0e,
        0x2ee8d30a, 0x2ee21a30, 0x2edb9305, 0x2ed53c1a, 0x2ecf140a, 0x2ec9197c, 0x2ec34b1e, 0x2ebda7aa,
        0x2eb82de2, 0x2eb2dc94, 0x2eadb294, 0x2ea8aebe, 0x2ea3cffa, 0x2e9f1535, 0x2e9a7d64, 0x2e960787,
        0x2e91b2a1, 0x2e8d7dbf, 0x2e8967f5, 0x2e85705c, 0x2e819616, 0x2e7bb094, 0x2e746c4a, 0x2e6d5db5,
        0x2e66834a, 0x2e5fdb86, 0x2e5964f2, 0x2e531e24, 0x2e4d05ba, 0x2e471a5e, 0x2e415ac1, 0x2e3bc5a2,
        0x2e3659c6, 0x2e3115fb, 0x2e2bf91a, 0x2e270204, 0x2e222fa0, 0x2e1d80e1, 0x2e18f4bd, 0x2e148a36,
        0x2e104052, 0x2e0c1621, 0x2e080ab9, 0x2e041d36, 0x2e004cba, 0x2df930e1, 0x2df1ff0f, 0x2deb026a,
        0x2de43969, 0x2ddda290, 0x2dd73c6a, 0x2dd1058f, 0x2dcafca3, 0x2dc52052, 0x2dbf6f52, 0x2db9e863,
        0x2db48a4e, 0x2daf53e5, 0x2daa4404, 0x2da5598c, 0x2da09369, 0x2d9bf090, 0x2d976ffb, 0x2d9310ad,
        0x2d8ed1b0, 0x2d8ab215, 0x2d86b0f5, 0x2d82cd6d, 0x2d7e0d47, 0x2d76b788, 0x2d6f97ff, 0x2d68ad1c,
        0x2d61f55a, 0x2d5b6f40, 0x2d55195d, 0x2d4ef24e, 0x2d48f8b8, 0x2d432b4d, 0x2d3d88c4, 0x2d380fe2,
        0x2d32bf71, 0x2d2d9648, 0x2d289343, 0x2d23b54a, 0x2d1efb4a, 0x2d1a6439, 0x2d15ef16, 0x2d119ae4,
        0x2d0d66b2, 0x2d095192, 0x2d055a9f, 0x2d0180fa, 0x2cfb8793, 0x2cf44478, 0x2ced370a, 0x2ce65dbc,
        0x2cdfb70d, 0x2cd94187, 0x2cd2fbbf, 0x2ccce453, 0x2cc6f9ee, 0x2cc13b41, 0x2cbba70a, 0x2cb63c10,
        0x2cb0f922, 0x2cabdd16, 0x2ca6e6cf, 0x2ca21534, 0x2c9d6738, 0x2c98dbd2, 0x2c947202, 0x2c9028d2,
        0x2c8bff4f, 0x2c87f48f, 0x2c8407b0, 0x2c8037d4, 0x2c790848, 0x2c71d7a2, 0x2c6adc21, 0x2c64143b,
        0x2c5d7e74, 0x2c571959, 0x2c50e382, 0x2c4adb91, 0x2c450035, 0x2c3f5022, 0x2c39ca1a, 0x2c346ce4,
        0x2c2f3755, 0x2c2a2846, 0x2c253e9c, 0x2c207940, 0x2c1bd728, 0x2c17574f, 0x2c12f8b8, 0x2c0eba6c,
        0x2c0a9b7d, 0x2c069b03, 0x2c02b81e, 0x2bfde3e4, 0x2bf68f56, 0x2bef70f6, 0x2be88734, 0x2be1d08a,
        0x2bdb4b80, 0x2bd4f6a6, 0x2bced097, 0x2bc8d7fb, 0x2bc30b81, 0x2bbd69e4, 0x2bb7f1e5, 0x2bb2a252,
        0x2bad7a00, 0x2ba877cc, 0x2ba39a9e, 0x2b9ee163, 0x2b9a4b12, 0x2b95d6a9, 0x2b91832c, 0x2b8d4fa9,
        0x2b893b33, 0x2b8544e6, 0x2b816be1, 0x2b7b5e99, 0x2b741cad, 0x2b6d1065, 0x2b663834, 0x2b5f929b,
        0x2b591e22, 0x2b52d960, 0x2b4cc2f2, 0x2b46d983, 0x2b411bc6, 0x2b3b8878, 0x2b361e60, 0x2b30dc4c,
        0x2b2bc116, 0x2b26cb9e, 0x2b21facc, 0x2b1d4d93, 0x2b18c2ea, 0x2b1459d3, 0x2b101156, 0x2b0be880,
        0x2b07de69, 0x2b03f22d, 0x2b0022f0, 0x2af8dfb6, 0x2af1b03c, 0x2aeab5de, 0x2ae3ef13, 0x2add5a5e,
        0x2ad6f64e, 0x2ad0c17a, 0x2acaba85, 0x2ac4e01c, 0x2abf30f7, 0x2ab9abd5, 0x2ab44f80, 0x2aaf1aca,
        0x2aaa0c8e, 0x2aa523b0, 0x2aa05f1c, 0x2a9bbdc5, 0x2a973ea8, 0x2a92e0c6, 0x2a8ea32b, 0x2a8a84e8,
        0x2a868516, 0x2a82a2d2, 0x2a7dba87, 0x2a76672b, 0x2a6f49f4, 0x2a686152, 0x2a61abc1, 0x2a5b27c6,
        0x2a54d3f4, 0x2a4eaee6, 0x2a48b743, 0x2a42ebba, 0x2a3d4b08, 0x2a37d3ed, 0x2a328538, 0x2a2d5dbd,
        0x2a285c5a, 0x2a237ff7, 0x2a1ec781, 0x2a1a31ef, 0x2a15be40, 0x2a116b77, 0x2a0d38a4, 0x2a0924d8,
        0x2a052f30, 0x2a0156cb, 0x29fb35a5, 0x29f3f4e8, 0x29ece9c6, 0x29e612b3, 0x29df6e2f, 0x29d8fac4
    ]

    // Z-Image-Turbo's transformer (`IK_VAL_Z_IMAGE_TURBO`, released float32) at bf16 as its generator loads it,
    // cut to its refiners and its first two unified layers, against diffusers' `ZImageTransformer2DModel` at
    // bf16 (`z_image_release_probe`; the float32 side widens the same bf16 weights) over a seeded 16×24 latent
    // and 37 caption features padded to 64: the timestep, patch, and caption embedders, every block, and the
    // final layer on the reference's own input, the velocity end to end, then each piece of the first unified
    // layer.
    func testZImageTurboTransformerInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_Z_IMAGE_TURBO"], "IK_VAL_Z_IMAGE_TURBO"))
            .appendingPathComponent("transformer")
        let bf16 = try record("z_image_turbo_cut2_bf16.safetensors")
        let f32 = try record("z_image_turbo_cut2_f32.safetensors")
        var configuration = try NFKMLXZImageRelease.transformerConfiguration(
            fromHuggingFace: directory.appendingPathComponent("config.json"))
        configuration.layers = 2
        let net = NFKMLXZImageTransformerNet(configuration)
        let kept = try NFKMLXReleaseWeights.arrays(inDirectory: directory, converting: .bfloat16) { key in
            key.hasPrefix("layers.") && Int(key.split(separator: ".")[1])! >= configuration.layers ? nil : key
        }
        try NFKMLXWeights.apply(kept, to: net, verifyShapes: true)
        func recorded(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        let adaln = try recorded("t_embedder.out")
        let tables = net.ropeTables(captionLength: 64, grid: (1, 8, 12), imagePad: 0)
        let unified = (cos: concatenated([tables.image.cos, tables.caption.cos], axis: 0),
                       sin: concatenated([tables.image.sin, tables.caption.sin], axis: 0))

        var isolated = [(String, MLXArray)](), keys = [String]()
        func stage(_ name: String, _ run: (MLXArray) -> MLXArray) throws {
            isolated.append((name, run(try recorded(name + ".in"))))
            keys.append(name + ".out")
        }
        isolated.append(("t_embedder", net.tEmbedder(try XCTUnwrap(bf16["t_embedder.in"]))[0]))
        keys.append("t_embedder.out")
        try stage("x_embedder") { net.allXEmbedder.entry($0) }
        try stage("cap_embedder") { (net.capEmbedder[1] as! Linear)((net.capEmbedder[0] as! RMSNorm)($0)) }
        for (i, block) in net.noiseRefiner.enumerated() {
            try stage("noise_refiner.\(i)") { block($0, cos: tables.image.cos, sin: tables.image.sin, adaln: adaln) }
        }
        for (i, block) in net.contextRefiner.enumerated() {
            try stage("context_refiner.\(i)") {
                block($0, cos: tables.caption.cos, sin: tables.caption.sin, adaln: nil)
            }
        }
        for (i, block) in net.layers.enumerated() {
            try stage("layers.\(i)") { block($0, cos: unified.cos, sin: unified.sin, adaln: adaln) }
        }
        try stage("final") { net.allFinalLayer.entry($0, adaln: adaln) }

        let velocity = net(try recorded("latent"), capFeats: try recorded("cap_feats"),
                           t: try XCTUnwrap(bf16["timestep"]))
        XCTAssertEqual(velocity.dtype, .bfloat16, "a bf16 load runs the transformer at bf16")
        let states = [("velocity", velocity)]
        eval(isolated.map(\.1) + states.map(\.1))
        let rows = try seams(states, bf16: bf16, f32: f32, keys: ["output"])
        report("z-image-turbo", rows)
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: keys)
        report("z-image-turbo isolated", isolatedRows)
        assertRoundingPlacement("z-image-turbo", endToEnd: rows, isolated: isolatedRows)

        let block = net.layers[0], attention = block.attention
        func input(_ name: String) throws -> MLXArray { try recorded("piece." + name + ".in") }
        func output(_ name: String) -> String { "piece." + name + ".out" }
        let lines = [
            try piece("adaLN", (block.adaLNModulation![0] as! Linear)(input("adaLN_modulation.0")), bf16,
                      output("adaLN_modulation.0")),
            try piece("attention_norm1", block.attentionNorm1(input("attention_norm1")), bf16,
                      output("attention_norm1")),
            try piece("to_q", attention.toQ(input("attention.to_q")), bf16, output("attention.to_q")),
            try piece("norm_q", attention.normQ(input("attention.norm_q")), bf16, output("attention.norm_q")),
            try piece("norm_k", attention.normK(input("attention.norm_k")), bf16, output("attention.norm_k")),
            try piece("attention", attention(input("attention"), cos: unified.cos, sin: unified.sin), bf16,
                      output("attention")),
            try piece("to_out", (attention.toOut[0] as! Linear)(input("attention.to_out.0")), bf16,
                      output("attention.to_out.0")),
            try piece("attention_norm2", block.attentionNorm2(input("attention_norm2")), bf16,
                      output("attention_norm2")),
            try piece("ffn_norm1", block.ffnNorm1(input("ffn_norm1")), bf16, output("ffn_norm1")),
            try piece("w1", block.feedForward.w1(input("feed_forward.w1")), bf16, output("feed_forward.w1")),
            try piece("feed_forward", block.feedForward(input("feed_forward")), bf16, output("feed_forward")),
            try piece("w2", block.feedForward.w2(input("feed_forward.w2")), bf16, output("feed_forward.w2")),
            try piece("ffn_norm2", block.ffnNorm2(input("ffn_norm2")), bf16, output("ffn_norm2")),
        ]
        print("VALIDATION bf16 z-image-turbo pieces:\n" + lines.joined(separator: "\n"))
    }

    func testFlux2TransformerInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (bf16, f32, weights) = try tinyDiT("flux2")
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        let net = NFKMLXFlux2TransformerNet(.tiny)
        try NFKMLXWeights.apply(weights, to: net)
        let output = net(try input("hidden"), encoderHidden: try input("encoder"), timestep: try input("timestep"),
                         guidance: bf16["guidance"].map { $0.asType(.bfloat16) },
                         imageIds: try XCTUnwrap(bf16["img_ids"]), textIds: try XCTUnwrap(bf16["txt_ids"]))
        eval(output)
        let rows = try seams([("velocity", output)], bf16: bf16, f32: f32, keys: ["output"])
        report("flux2", rows)
        assertRoundingPlacement("flux2", endToEnd: rows, isolated: rows)
    }

    func testQwenImageTransformerInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (bf16, f32, _) = try tinyDiT("qwenimage21")
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        let net = NFKMLXQwenImage.makeNet(.tiny)
        let weights = bf16.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            return NFKMLXQwenImage.remapReferenceKey(String(key.dropFirst(3))).map { ($0, value.asType(.bfloat16)) }
        }
        try NFKMLXWeights.apply(weights, to: net, verifyShapes: true)
        let imageMask = try XCTUnwrap(bf16["img_mask"]).asArray(Int32.self).map { $0 != 0 }
        let encoderMask = try XCTUnwrap(bf16["encoder_mask"]).asArray(Int32.self).map { $0 != 0 }
        let output = net(latents: try input("latents"), encoderHidden: try input("encoder_hidden_states"),
                         timestep: try XCTUnwrap(bf16["timestep"]).item(Float.self),
                         imageShapes: [(frame: 1, height: 2, width: 2), (frame: 1, height: 4, width: 4)],
                         imageMask: imageMask, encoderMask: encoderMask)
        eval(output)
        var observed = [(String, MLXArray)]()
        _ = net(latents: try input("latents"), encoderHidden: try input("encoder_hidden_states"),
                timestep: try XCTUnwrap(bf16["timestep"]).item(Float.self),
                imageShapes: [(frame: 1, height: 2, width: 2), (frame: 1, height: 4, width: 4)],
                imageMask: imageMask, encoderMask: encoderMask) { observed.append(($0, $1)) }
        let probeKeys = ["txt_in": "probe.txt_in.out", "img_in": "probe.img_in.out", "modulation": "probe.modulation.out", "block_0": "probe.transformer_blocks.0.out",
                         "block_1": "probe.transformer_blocks.1.out", "norm_out": "probe.norm_out.out"]
        let probed = observed.filter { probeKeys[$0.0] != nil }
        eval(probed.map(\.1))
        report("qwenimage seams", try seams(probed, bf16: bf16, f32: f32, keys: probed.map { probeKeys[$0.0]! }))
        let rows = try seams([("velocity", output)], bf16: bf16, f32: f32, keys: ["output"])
        report("qwenimage", rows)
        assertRoundingPlacement("qwenimage", endToEnd: rows, isolated: rows)
    }

    func testWanTransformerInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (bf16, f32, _) = try tinyDiT("wan")
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        let net = NFKMLXWanTransformerNet(.tiny)
        let weights = bf16.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            return (String(key.dropFirst(3)), (value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value).asType(.bfloat16))
        }
        try NFKMLXWeights.apply(weights, to: net)
        let output = net(try input("latent"), text: try input("text"), t: try input("timestep"))
        eval(output)
        // Each piece of the first block, and the condition embedder, on the reference's own input. The
        // float32 sinusoid differs from torch's by an ulp (MLX's `exp`, `sin`, and `cos` on either device),
        // which flips a few bf16 roundings of the timestep embedding; every block piece is exact.
        func probed(_ key: String) throws -> MLXArray { try input("probe." + key)[0] }
        let (temb, proj, context) = net.conditionEmbedder(try input("timestep"), text: try input("text"),
                                                          freqDim: net.config.freqDim)
        var positions = [Float]()
        for f in 0 ..< 2 { for h in 0 ..< 2 { for w in 0 ..< 2 { positions += [Float(f), Float(h), Float(w)] } } }
        let (cosT, sinT) = net.rope.table(positions: MLXArray(positions).reshaped([8, 3]))
        let block = net.blocks[0]
        let pieces: [(String, MLXArray)] = [
            ("sinusoid", ltxTimestepEmbedding(try input("timestep"), channels: net.config.freqDim)),
            ("temb", temb), ("time_proj", proj), ("text", context),
            ("attn1", block.attn1(try probed("blocks.0.attn1.in"), context: nil, cos: cosT, sin: sinT)),
            ("attn2", block.attn2(try probed("blocks.0.attn2.in"), context: try probed("condition_embedder.text_embedder.out"),
                                  cos: nil, sin: nil)),
            ("ffn", block.ffn(try probed("blocks.0.ffn.in"))),
            ("block 0", block(try probed("blocks.0.in"), context: try probed("condition_embedder.text_embedder.out"),
                              temb: proj.reshaped([6, -1]), cos: cosT, sin: sinT))]
        eval(pieces.map(\.1))
        report("wan pieces", try seams(pieces, bf16: bf16, f32: f32, keys: [
            "probe.condition_embedder.timesteps_proj.out", "probe.condition_embedder.time_embedder.out", "probe.condition_embedder.time_proj.out",
            "probe.condition_embedder.text_embedder.out", "probe.blocks.0.attn1.out", "probe.blocks.0.attn2.out",
            "probe.blocks.0.ffn.out", "probe.blocks.0.out"]))
        let rows = try seams([("velocity", output)], bf16: bf16, f32: f32, keys: ["output"])
        report("wan", rows)
        assertRoundingPlacement("wan", endToEnd: rows, isolated: rows)
    }

    // MARK: Opt-in half precision at the tiny configurations

    /// A tiny record's weights under `prefix` at bf16, each laid out as its float32 test loads it.
    private func tinyWeights(_ record: [String: MLXArray], prefix: String,
                             _ layout: (String, MLXArray) -> MLXArray = { $1 }) -> [(String, MLXArray)] {
        record.compactMap { key, value in
            guard key.hasPrefix(prefix) else { return nil }
            let name = String(key.dropFirst(prefix.count))
            return (name, layout(name, value).asType(.bfloat16))
        }
    }

    // The FLUX ControlNet at `run_reference.py flux_controlnet`'s tiny configuration with its three-block base,
    // at bf16 against diffusers at bf16 (`IK_DIT_DTYPE`): each double- and single-block residual and the steered
    // velocity end to end; and on the reference's own inputs (its first-call `probe.*` seams) both embedders,
    // the timestep embedding, each double and single block, each zero-initialized projection, and the base on
    // the reference's residuals. A residual carries every block before it, so it is an end-to-end state.
    func testFluxControlNetInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let bf16 = try record("flux_controlnet_bf16.safetensors"), f32 = try record("flux_controlnet_bf16in.safetensors")
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        let control = NFKMLXFluxControlNetNet(.tiny)
        try NFKMLXWeights.apply(tinyWeights(bf16, prefix: "w::"), to: control)
        let base = NFKMLXFluxTransformerNet(NFKMLXFluxConfiguration(
            inChannels: 8, outChannels: 8, numLayers: 3, numSingleLayers: 3, attentionHeadDim: 6,
            numAttentionHeads: 2, jointAttentionDim: 24, pooledProjectionDim: 10, guidanceEmbeds: true,
            axesDimsRope: [2, 2, 2]))
        try NFKMLXWeights.apply(tinyWeights(bf16, prefix: "t::"), to: base)
        let imageIds = try XCTUnwrap(bf16["img_ids"])
        let (double, single) = control(try input("hidden"), controlnetCond: try input("control_cond"),
                                       encoder: try input("encoder"), pooled: try input("pooled"),
                                       timestep: try input("timestep"), guidance: try input("guidance"),
                                       imageIds: imageIds, conditioningScale: 0.6)
        func velocity(_ double: [MLXArray], _ single: [MLXArray]) throws -> MLXArray {
            base(try input("hidden"), encoderHidden: try input("encoder"), pooled: try input("pooled"),
                 timestep: try input("timestep"), guidance: try input("guidance"), imageIds: imageIds,
                 controlnetBlockSamples: double, controlnetSingleBlockSamples: single)
        }
        let residuals = double.enumerated().map { ("double \($0.offset)", $0.element) }
            + single.enumerated().map { ("single \($0.offset)", $0.element) }
        let keys = double.indices.map { "double_\($0)" } + single.indices.map { "single_\($0)" } + ["output"]
        let output = try velocity(double, single)
        let isolated = try velocity(double.indices.map { try input("double_\($0)") },
                                    single.indices.map { try input("single_\($0)") })
        XCTAssertEqual(output.dtype, .bfloat16, "a bf16 load runs the ControlNet and its base at bf16")
        eval(residuals.map(\.1) + [output, isolated])
        let rows = try seams(residuals + [("velocity", output)], bf16: bf16, f32: f32, keys: keys)
        report("flux-controlnet", rows)
        let textLength = try input("encoder").dim(1)
        let (cos, sin) = control.rope.table(ids: concatenated([MLXArray.zeros([textLength, imageIds.dim(1)]), imageIds],
                                                              axis: 0))
        let temb = try input("probe.time_text_embed.out")
        var pieces: [(String, MLXArray)] = [
            ("x embedder", control.xEmbedder(try input("probe.x_embedder.in"))),
            ("control embedder", control.controlnetXEmbedder(try input("probe.controlnet_x_embedder.in"))),
            ("timestep embedding", control.timeTextEmbed(timestep: try input("timestep") * 1000,
                                                         guidance: try input("guidance") * 1000, pooled: try input("pooled"))),
        ]
        var pieceKeys = ["probe.x_embedder.out", "probe.controlnet_x_embedder.out", "probe.time_text_embed.out"]
        for (i, block) in control.transformerBlocks.enumerated() {
            let p = "probe.transformer_blocks.\(i)."
            pieces.append(("double block \(i)", block(try input(p + "norm1.in"), encoder: try input(p + "norm1_context.in"),
                                                      temb: temb, cos: cos, sin: sin).1))
            pieceKeys.append("probe.controlnet_blocks.\(i).in")
            pieces.append(("zero projection \(i)", control.zeroProjected(double: i, try input("probe.controlnet_blocks.\(i).in"))))
            pieceKeys.append("probe.controlnet_blocks.\(i).out")
        }
        for (i, block) in control.singleTransformerBlocks.enumerated() {
            let joined = try input("probe.single_transformer_blocks.\(i).norm.in")
            pieces.append(("single block \(i)", block(joined[0..., textLength...], encoder: joined[0..., 0 ..< textLength],
                                                      temb: temb, cos: cos, sin: sin).1))
            pieceKeys.append("probe.controlnet_single_blocks.\(i).in")
            pieces.append(("single zero projection \(i)",
                           control.zeroProjected(single: i, try input("probe.controlnet_single_blocks.\(i).in"))))
            pieceKeys.append("probe.controlnet_single_blocks.\(i).out")
        }
        pieces.append(("velocity on the reference's residuals", isolated))
        pieceKeys.append("output")
        eval(pieces.map(\.1))
        let isolatedRows = try seams(pieces, bf16: bf16, f32: f32, keys: pieceKeys)
        report("flux-controlnet isolated", isolatedRows)
        assertRoundingPlacement("flux-controlnet", endToEnd: rows, isolated: isolatedRows)
    }

    // The SD3 ControlNet at `run_reference.py sd3_controlnet`'s tiny configuration with its four-block base, at
    // bf16 against diffusers at bf16 (`IK_DIT_DTYPE`): each residual and the steered velocity end to end; and on
    // the reference's own inputs (its first-call `probe.*` seams) both patch embeddings, the timestep embedding,
    // each joint block, each zero-initialized projection, and the base on the reference's residuals. A residual
    // carries every block before it.
    func testSD3ControlNetInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let bf16 = try record("sd3_controlnet_bf16.safetensors"), f32 = try record("sd3_controlnet_bf16in.safetensors")
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        let channelsLast: (String, MLXArray) -> MLXArray = { $1.ndim == 4 ? $1.transposed(0, 2, 3, 1) : $1 }
        let control = NFKMLXSD3ControlNetNet(.tiny)
        try NFKMLXWeights.apply(tinyWeights(bf16, prefix: "w::", channelsLast), to: control)
        let base = NFKMLXSD3TransformerNet(NFKMLXSD3Configuration(
            sampleSize: 16, inChannels: 4, outChannels: 4, numLayers: 4, attentionHeadDim: 8,
            numAttentionHeads: 2, jointAttentionDim: 24, captionProjectionDim: 16, pooledProjectionDim: 20,
            posEmbedMaxSize: 8, dualAttentionLayers: [], qkNorm: true))
        try NFKMLXWeights.apply(tinyWeights(bf16, prefix: "t::", channelsLast), to: base)
        let residuals = control(try input("latent"), controlnetCond: try input("control_cond"), encoder: try input("encoder"),
                                pooled: try input("pooled"), timestep: try input("timestep"), conditioningScale: 0.7)
        func velocity(_ residuals: [MLXArray]) throws -> MLXArray {
            base(try input("latent"), encoderHidden: try input("encoder"), pooled: try input("pooled"),
                 timestep: try input("timestep"), blockControlnetHiddenStates: residuals)
        }
        let named = residuals.enumerated().map { ("residual \($0.offset)", $0.element) }
        let keys = residuals.indices.map { "residual_\($0)" } + ["output"]
        let output = try velocity(residuals)
        let isolated = try velocity(residuals.indices.map { try input("residual_\($0)") })
        XCTAssertEqual(output.dtype, .bfloat16, "a bf16 load runs the ControlNet and its base at bf16")
        eval(named.map(\.1) + [output, isolated])
        let rows = try seams(named + [("velocity", output)], bf16: bf16, f32: f32, keys: keys)
        report("sd3-controlnet", rows)
        let temb = try input("probe.time_text_embed.out")
        var pieces: [(String, MLXArray)] = [
            ("patch embedding", control.posEmbed!(try input("probe.pos_embed.in"))),
            ("control patch embedding", control.posEmbedInput(try input("probe.pos_embed_input.in"))),
            ("timestep embedding", control.timeTextEmbed(timestep: try input("timestep"), pooled: try input("pooled"))),
        ]
        var pieceKeys = ["probe.pos_embed.out", "probe.pos_embed_input.out", "probe.time_text_embed.out"]
        for (i, module) in control.transformerBlocks.enumerated() {
            guard let block = module as? NFKSD3JointBlock else { continue }
            let p = "probe.transformer_blocks.\(i)."
            pieces.append(("joint block \(i)", block(try input(p + "norm1.in"), encoder: try input(p + "norm1_context.in"),
                                                     temb: temb).1))
            pieceKeys.append("probe.controlnet_blocks.\(i).in")
            pieces.append(("zero projection \(i)", control.zeroProjected(i, try input("probe.controlnet_blocks.\(i).in"))))
            pieceKeys.append("probe.controlnet_blocks.\(i).out")
        }
        pieces.append(("velocity on the reference's residuals", isolated))
        pieceKeys.append("output")
        eval(pieces.map(\.1))
        let isolatedRows = try seams(pieces, bf16: bf16, f32: f32, keys: pieceKeys)
        report("sd3-controlnet isolated", isolatedRows)
        assertRoundingPlacement("sd3-controlnet", endToEnd: rows, isolated: isolatedRows)
    }

    // The LTX-2 audio-video transformer at `run_reference.py ltx2`'s tiny configurations (LTX-2.0's caption
    // projections, LTX-2.3, and LTX-2.5's switches), at bf16 against diffusers at bf16 (`IK_DIT_DTYPE`): both
    // velocities of each.
    func testLTX2TransformerInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let bf16 = try record("ltx2_bf16.safetensors"), f32 = try record("ltx2_bf16in.safetensors")
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        var tiny20 = NFKMLXLTX2Configuration.tiny
        tiny20.usesPromptEmbeddings = true
        let audio = try input("audio_hidden")
        for (prefix, configuration) in [("l20", tiny20), ("l23", NFKMLXLTX2Configuration.tiny),
                                        ("l25", NFKMLXLTX2Configuration.tiny25)] {
            let net = NFKMLXLTX2TransformerNet(configuration)
            try NFKMLXWeights.apply(tinyWeights(bf16, prefix: "w::\(prefix)."), to: net)
            let (video, audioVelocity) = net(
                video: try input("hidden"), audio: audio,
                text: try input(prefix == "l20" ? "caption" : "encoder"),
                audioText: try input(prefix == "l20" ? "audio_caption" : "audio_encoder"),
                timestep: try input("timestep"), audioTimestep: try input("audio_timestep"), sigma: try input("sigma"),
                frames: 2, height: 2, width: 3, audioFrames: audio.dim(1),
                keyframes: prefix == "l25" ? try input("keyframes").reshaped([1, -1]) : nil)
            XCTAssertEqual(video.dtype, .bfloat16, "a bf16 load runs LTX-2 at bf16")
            eval(video, audioVelocity)
            let rows = try seams([("video velocity", video), ("audio velocity", audioVelocity)], bf16: bf16, f32: f32,
                                 keys: ["\(prefix).video", "\(prefix).audio"])
            report("ltx2-\(prefix)", rows)
            assertRoundingPlacement("ltx2-\(prefix)", endToEnd: rows, isolated: rows)
        }
    }

    // The Wan Animate DiT at `run_reference.py wan_animate`'s tiny configuration, at bf16 against diffusers at
    // bf16 (`IK_DIT_DTYPE`): the extract pass that fills the reference cache, the generation pass over it, and a
    // shorter chunk.
    func testWanAnimateInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let bf16 = try record("wan_animate_bf16.safetensors"), f32 = try record("wan_animate_bf16in.safetensors")
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        let net = NFKMLXWanAnimateNet(.tiny)
        try NFKMLXWeights.apply(tinyWeights(bf16, prefix: "w::") { $1.ndim == 5 ? $1.transposed(0, 2, 3, 4, 1) : $1 },
                                to: net)
        let text = try input("text"), imageEmbeddings = try input("image_embeds"), timestep = try input("timestep")
        let cache = NFKMLXWanAnimateKVCache(layerCount: 2)
        let extract = try net.extractReference(latent: try input("reference_latent"),
                                               condition: try input("reference_condition"),
                                               text: text, imageEmbeddings: imageEmbeddings, into: cache)
        let generated = try net.generate(latent: try input("latent"), condition: try input("condition"), text: text,
                                         imageEmbeddings: imageEmbeddings, timestep: timestep, cache: cache,
                                         referenceGrid: (2, 4, 4), videoFrames: 2, videoArea: 16)
        let chunk = try net.generate(latent: try input("chunk_latent"), condition: try input("chunk_condition"),
                                     text: text, imageEmbeddings: imageEmbeddings, timestep: timestep, cache: cache,
                                     referenceGrid: (2, 4, 4), videoFrames: 4, videoArea: 16)
        XCTAssertEqual(generated.dtype, .bfloat16, "a bf16 load runs Wan Animate at bf16")
        eval(extract, generated, chunk)
        let rows = try seams([("extract", extract), ("generation", generated), ("chunk", chunk)], bf16: bf16, f32: f32,
                             keys: ["extract", "output", "chunk"])
        report("wan-animate", rows)
        assertRoundingPlacement("wan-animate", endToEnd: rows, isolated: rows)
    }

    // The Wan autoencoders at `run_reference.py wan_vae` (2.2, residual, patchified) and `wan_vae_21`'s tiny
    // configurations, at bf16 against diffusers at bf16 (`IK_DIT_DTYPE`): the encoded latent of a 5-frame clip,
    // the decode of the reference's latent, and the decode of ours. Qwen-Image 2.1's autoencoder is this
    // network specialized to one frame. On the reference's own first-chunk inputs (its first-call `probe.*`
    // seams), each encoder and decoder stage and both quant convolutions, each from an empty cache as the
    // reference's first chunk runs; then each decoder stage of every later latent frame on the reference's
    // input for that frame, over the cache the reference's earlier frames fill.
    func testWanVideoVAEInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        for (name, configuration) in [("wan_vae", NFKMLXWanVAEConfiguration.tiny), ("wan_vae_21", .tiny21)] {
            let bf16 = try record("\(name)_bf16.safetensors"), f32 = try record("\(name)_bf16in.safetensors")
            func input(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
            let net = NFKMLXWanVideoVAENet(configuration)
            try NFKMLXWeights.apply(tinyWeights(bf16, prefix: "w::") { name, value in
                if name.hasSuffix(".gamma") { return value.reshaped([-1]) }
                if value.ndim == 5 { return value.transposed(0, 2, 3, 4, 1) }
                return value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
            }, to: net)
            let latent = net.encode(try input("video").expandedDimensions(axis: 0))
            let decoded = net.decode(latent)
            let isolated = net.decode(try input("latent").expandedDimensions(axis: 0))
            XCTAssertEqual(latent.dtype, .bfloat16, "a bf16 load runs the autoencoder at bf16")
            eval(latent, decoded, isolated)
            let label = name.replacingOccurrences(of: "_", with: "-")
            let rows = try seams([("latent", latent[0]), ("decode", decoded[0])], bf16: bf16, f32: f32,
                                 keys: ["latent", "output"])
            report(label, rows)
            func probe(_ key: String) throws -> MLXArray { try input("probe." + key).transposed(0, 2, 3, 4, 1) }
            func channelsFirst(_ x: MLXArray) -> MLXArray { x.transposed(0, 4, 1, 2, 3) }
            let encoder = net.encoder, decoder = net.decoder
            var stages: [(String, (MLXArray) -> MLXArray)] = [("encoder.conv_in", { encoder.convIn($0, cache: .empty) })]
            for (i, block) in encoder.downBlocks.enumerated() {
                stages.append(("encoder.down_blocks.\(i)", { x in
                    if let b = block as? NFKWanResidualDownBlock { return b(x, NFKWanCache()) }
                    if let b = block as? NFKWanResidualBlock { return b(x, NFKWanCache()) }
                    return (block as! NFKWanResample)(x, NFKWanCache())
                }))
            }
            stages += [("encoder.mid_block", { encoder.midBlock($0, NFKWanCache()) }),
                       ("encoder.norm_out", { encoder.normOut($0) }),
                       ("encoder.conv_out", { encoder.convOut($0, cache: .empty) }),
                       ("quant_conv", { net.quantConv($0, cache: .empty) }),
                       ("post_quant_conv", { net.postQuantConv($0, cache: .empty) }),
                       ("decoder.conv_in", { decoder.convIn($0, cache: .empty) }),
                       ("decoder.mid_block", { decoder.midBlock($0, NFKWanCache()) })]
            for (i, block) in decoder.upBlocks.enumerated() {
                stages.append(("decoder.up_blocks.\(i)", { x in
                    if let b = block as? NFKWanResidualUpBlock { return b(x, NFKWanCache(), firstChunk: true) }
                    return (block as! NFKWanUpBlock)(x, NFKWanCache())
                }))
            }
            stages += [("decoder.norm_out", { decoder.normOut($0) }),
                       ("decoder.conv_out", { decoder.convOut($0, cache: .empty) })]
            var staged = try stages.map { name, run in (name, channelsFirst(run(try probe(name + ".in")))) }
            var stagedKeys = staged.map { "probe.\($0.0).out" }
            // Each later latent frame: the decoder runs the reference's earlier frames to fill its cache as the
            // reference's does, then each stage runs on the reference's own input for this frame over that cache.
            var frame = 1
            while bf16["probe.decoder.call\(frame).in"] != nil {
                let cache = NFKWanCache()
                for earlier in 0 ..< frame {
                    cache.resetIndex()
                    _ = decoder(try probe("decoder.call\(earlier).in"), cache, firstChunk: earlier == 0)
                }
                cache.resetIndex()
                var frameStages: [(String, (MLXArray) -> MLXArray)] = [
                    ("decoder.conv_in", { wanCausal(decoder.convIn, $0, cache) }), ("decoder.mid_block", { decoder.midBlock($0, cache) })]
                for (i, block) in decoder.upBlocks.enumerated() {
                    frameStages.append(("decoder.up_blocks.\(i)", { x in
                        if let b = block as? NFKWanResidualUpBlock { return b(x, cache, firstChunk: false) }
                        return (block as! NFKWanUpBlock)(x, cache)
                    }))
                }
                frameStages += [("decoder.norm_out", { decoder.normOut($0) }),
                                ("decoder.conv_out", { wanCausal(decoder.convOut, $0, cache) })]
                for (name, run) in frameStages {
                    staged.append(("\(name) frame \(frame)", channelsFirst(run(try probe(name + ".call\(frame).in")))))
                    stagedKeys.append("probe.\(name).call\(frame).out")
                }
                frame += 1
            }
            eval(staged.map(\.1))
            let isolatedRows = try seams(staged + [("decode of the reference's latent", isolated[0])], bf16: bf16, f32: f32,
                                         keys: stagedKeys + ["output"])
            report("\(label) isolated", isolatedRows)
            assertRoundingPlacement(label, endToEnd: rows, isolated: isolatedRows)
        }
    }

    // MARK: The T5 text encoders

    // Wan's umT5-XXL (`IK_VAL_WAN21_T2V_1_3B`) at bf16, the precision its generator takes when float32
    // does not fit, against transformers' `UMT5EncoderModel` at bf16 (`t5_layer_probe` under transformers
    // 5.17, eager, blocks 0 and 23), over a prompt and four masked padding positions.
    func testUMT5XXLInBFloat16MatchesTheBFloat16Reference() throws {
        let release = try existing(config["IK_VAL_WAN21_T2V_1_3B"], "IK_VAL_WAN21_T2V_1_3B")
        try t5Encoder("umt5_xxl", directory: URL(fileURLWithPath: release).appendingPathComponent("text_encoder"))
    }

    // SD3.5-medium's T5-XXL (`IK_VAL_SD35_MEDIUM`, stored float16) at bf16, as its generator takes it
    // when float32 does not fit, against transformers' `T5EncoderModel` at bf16, as the umT5 test.
    func testT5XXLInBFloat16MatchesTheBFloat16Reference() throws {
        let release = try existing(config["IK_VAL_SD35_MEDIUM"], "IK_VAL_SD35_MEDIUM")
        try t5Encoder("t5_xxl", directory: URL(fileURLWithPath: release).appendingPathComponent("text_encoder_3"))
    }

    func testUMT5XXLBlockPiecesMatchTheBFloat16Reference() throws {
        let release = try existing(config["IK_VAL_WAN21_T2V_1_3B"], "IK_VAL_WAN21_T2V_1_3B")
        try t5Pieces("umt5_xxl", directory: URL(fileURLWithPath: release).appendingPathComponent("text_encoder"))
    }

    func testT5XXLBlockPiecesMatchTheBFloat16Reference() throws {
        let release = try existing(config["IK_VAL_SD35_MEDIUM"], "IK_VAL_SD35_MEDIUM")
        try t5Pieces("t5_xxl", directory: URL(fileURLWithPath: release).appendingPathComponent("text_encoder_3"))
    }

    /// The encoder at `directory` loaded at bf16 as the generators load it, with the record's tokens and
    /// the per-block bias, padding mask added.
    private func t5Setup(_ directory: URL, probe: [String: MLXArray]) throws
        -> (net: NFKMLXT5EncoderNet, tokens: MLXArray, mask: MLXArray, bias: (Int) -> MLXArray) {
        let configuration = try NFKMLXT5Encoder.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXT5Encoder.makeNet(configuration)
        try NFKMLXT5Encoder.loadWeights(into: net, from: directory, dtype: .bfloat16)
        let tokens = MLXArray(try XCTUnwrap(probe["tokens"]).asArray(Int32.self)).reshaped([1, -1])
        let mask = MLXArray(try XCTUnwrap(probe["mask"]).asArray(Int32.self)).reshaped([1, -1])
        let padding = NFKMLXT5EncoderNet.padding(mask, dtype: .bfloat16)
        let blocks = net.encoder.block
        let shared = configuration.perLayerBias ? nil : blocks[0].selfAttention.attention.computeBias(tokens.dim(1))
        return (net, tokens, mask, { (shared ?? blocks[$0].selfAttention.attention.computeBias(tokens.dim(1))) + padding })
    }

    private func t5Encoder(_ name: String, directory: URL) throws {
        try requireMLXRuntime()
        let bf16 = try record("\(name)_bf16.safetensors"), f32 = try record("\(name)_f32.safetensors")
        let (net, tokens, mask, bias) = try t5Setup(directory, probe: bf16)
        let blocks = net.encoder.block
        var hidden = net.shared(tokens)
        var states = [("embedding", hidden)]
        for (index, block) in blocks.enumerated() {
            hidden = block(hidden, bias: bias(index))
            states.append(("block \(index)", hidden))
        }
        let output = net(tokens, mask: mask)
        states.append(("final norm", output))
        eval(states.map(\.1))
        XCTAssertTrue(arrayEqual(output, net.encoder.finalLayerNorm(hidden)).item(Bool.self),
                      "the encoder's own pass is the block-by-block one")
        let rows = try seams(states, bf16: bf16, f32: f32, keys: (0 ... blocks.count).map { "hidden.\($0)" } + ["output"])
        report(name, rows)

        var isolated = [(String, MLXArray)]()
        for (index, block) in blocks.enumerated() {
            let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            isolated.append(("block \(index)", block(input, bias: bias(index))))
        }
        let last = try XCTUnwrap(bf16["hidden.\(blocks.count)"]).asType(.bfloat16).expandedDimensions(axis: 0)
        isolated.append(("final norm", net.encoder.finalLayerNorm(last)))
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32,
                                     keys: (1 ... blocks.count).map { "hidden.\($0)" } + ["output"])
        report("\(name) isolated", isolatedRows)
        assertRoundingPlacement(name, endToEnd: rows, isolated: isolatedRows)
    }

    // Each piece of blocks 0 and 23 on the reference's own bf16 input, so a rounding placed differently
    // shows as a count at the one seam that places it.
    private func t5Pieces(_ name: String, directory: URL) throws {
        try requireMLXRuntime()
        let probe = try record("\(name)_bf16.safetensors")
        let (net, _, _, bias) = try t5Setup(directory, probe: probe)
        func input(_ key: String) throws -> MLXArray {
            try XCTUnwrap(probe[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        var lines = [String]()
        for index in [0, net.encoder.block.count - 1] {
            let block = net.encoder.block[index]
            let attention = block.selfAttention
            let feedForward = block.layer[1] as! NFKT5FeedForwardLayer
            let p = "\(index)."
            lines.append("block \(index)")
            lines.append(try piece("attention norm", attention.layerNorm(input(p + "layer.0.layer_norm.in")),
                                   probe, p + "layer.0.layer_norm.out"))
            lines.append(try piece("q", attention.attention.q(input(p + "layer.0.SelfAttention.q.in")),
                                   probe, p + "layer.0.SelfAttention.q.out"))
            func heads(_ key: String) throws -> MLXArray {
                try XCTUnwrap(probe[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
            }
            let attended = NFKReferenceRounding.attention(queries: try heads(p + "attn.q"), keys: try heads(p + "attn.k"),
                                                          values: try heads(p + "attn.v"), scale: 1, mask: bias(index))
            lines.append(try piece("attention", attended[0].transposed(1, 0, 2), probe, p + "attn.out"))
            lines.append(try piece("o", attention.attention.o(input(p + "layer.0.SelfAttention.o.in")),
                                   probe, p + "layer.0.SelfAttention.o.out"))
            lines.append(try piece("SelfAttention", attention.attention(input(p + "layer.0.SelfAttention.in"), bias: bias(index)),
                                   probe, p + "layer.0.SelfAttention.out"))
            lines.append(try piece("feed-forward norm", feedForward.layerNorm(input(p + "layer.1.layer_norm.in")),
                                   probe, p + "layer.1.layer_norm.out"))
            lines.append(try piece("gelu_new", NFKReferenceRounding.geluNew(input(p + "layer.1.DenseReluDense.act.in")),
                                   probe, p + "layer.1.DenseReluDense.act.out"))
            lines.append(try piece("wo", feedForward.ff.wo(input(p + "layer.1.DenseReluDense.wo.in")),
                                   probe, p + "layer.1.DenseReluDense.wo.out"))
            lines.append(try piece("DenseReluDense", feedForward.ff(input(p + "layer.1.DenseReluDense.in")),
                                   probe, p + "layer.1.DenseReluDense.out"))
            lines.append(try piece("block", block(input(p + "block.in"), bias: bias(index)), probe, p + "block.out"))
        }
        print("VALIDATION bf16 \(name) pieces:\n" + lines.joined(separator: "\n"))
    }

    // MARK: Sa2VA

    // Sa2VA-1B (`IK_VAL_SA2VA_1B`) loaded at bf16 as its backend loads it: the InternViT tower, the
    // `mlp1` projector, and the Qwen2 decoder over the fused prompt and teacher-forced answer, against the
    // release's own code at bf16 (`sa2va_probe`, transformers 4.57, tower blocks 0 and 23 probed).
    func testSa2VA1BInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (net, bf16, f32) = try sa2vaInternVL("sa2va_1b", directoryKey: "IK_VAL_SA2VA_1B")
        try sa2vaInternVLSeams("sa2va-1b", net, bf16, f32)
    }

    // Each piece of Sa2VA-1B's tower blocks 0 and 23 and of its projector on the reference's own bf16
    // input, so a rounding placed differently shows as a count at the one seam that places it.
    func testSa2VA1BTowerPiecesMatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (net, probe, _) = try sa2vaInternVL("sa2va_1b", directoryKey: "IK_VAL_SA2VA_1B")
        try sa2vaInternVLPieces("sa2va-1b", net, probe)
    }

    private static let internVLReleases = [
        ("sa2va_internvl3_2b", "IK_VAL_SA2VA_INTERNVL3_2B"), ("sa2va_8b_cut4", "IK_VAL_SA2VA_8B_CUT4"),
        ("sa2va_26b_cut4", "IK_VAL_SA2VA_26B_CUT4"), ("sa2va_internvl3_8b_cut4", "IK_VAL_SA2VA_INTERNVL3_8B_CUT4"),
        ("sa2va_internvl3_14b_cut4", "IK_VAL_SA2VA_INTERNVL3_14B_CUT4"),
    ]

    // The other InternVL-based Sa2VA releases as Sa2VA-1B is measured: InternVL3-2B whole, and Sa2VA-8B,
    // Sa2VA-26B (InternViT-6B, with its query and key norms, and InternLM2), InternVL3-8B, and InternVL3-14B
    // cut to their first four decoder layers (`truncate.py`; the 26B's tower to its first four blocks), each
    // with its first and last tower blocks probed. `IK_SA2VA_ONLY=<record name>` measures one release.
    func testSa2VAInternVLReleasesInBFloat16MatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        var measured = 0
        let only = ProcessInfo.processInfo.environment["IK_SA2VA_ONLY"]
        for (name, key) in Self.internVLReleases where only == nil || only == name {
            do {
                try autoreleasepool {
                    let (net, bf16, f32) = try sa2vaInternVL(name, directoryKey: key)
                    let label = name.replacingOccurrences(of: "_", with: "-")
                    try sa2vaInternVLSeams(label, net, bf16, f32)
                    try sa2vaInternVLPieces(label, net, bf16)
                }
            } catch is XCTSkip {
                continue
            }
            Memory.clearCache()
            measured += 1
        }
        if measured == 0 { throw XCTSkip("set IK_VAL_SA2VA_INTERNVL3_2B or a Sa2VA cut (IK_VAL_SA2VA_8B_CUT4, ...)") }
    }

    /// One InternVL release's tower, projector, and decoder seams against its `sa2va_probe` pair.
    private func sa2vaInternVLSeams(_ label: String, _ net: NFKMLXSa2VANet, _ bf16: [String: MLXArray],
                                    _ f32: [String: MLXArray]) throws {
        let pixels = try XCTUnwrap(bf16["pixel_values"])
        let inputIds = try XCTUnwrap(bf16["input_ids"]).reshaped([-1]).asArray(Int32.self).map(Int.init)
        let blocks = net.vision.encoder.layers

        var hidden = net.vision.embeddings(pixels.asType(.bfloat16).transposed(0, 2, 3, 1))
        var states = [("tower embeddings", hidden)]
        for (index, block) in blocks.enumerated() {
            hidden = block(hidden)
            states.append(("tower block \(index)", hidden))
        }
        let features = net.imageFeatures(pixelValues: pixels)
        XCTAssertEqual(features.dtype, .bfloat16, "a bf16 load computes the image path in bf16")
        let fused = net.fusedEmbeddings(inputIds: inputIds, imageFeatures: features)
        XCTAssertEqual(fused.dtype, .bfloat16, "the fused embeddings stay bf16")
        states += [("projector", features), ("fused", fused[0])]
        let decoder = decoderStates(net.language, fused)
        states += decoder.enumerated().map { ("decoder \($0.offset == 0 ? "input" : "layer \($0.offset - 1)")", $0.element[0]) }
        let answer = try XCTUnwrap(bf16["output"]).dim(0)
        states.append(("logits", net.language.logits(fromHidden: decoder.last!)[0, (inputIds.count - answer)...]))
        eval(states.map(\.1))
        let keys = (0 ... blocks.count).map { "vit.hidden.\($0)" } + ["vit_embeds", "fused"]
            + decoder.indices.map { "dec.hidden.\($0)" } + ["output"]
        let rows = try seams(states, bf16: bf16, f32: f32, keys: keys)
        report(label, rows)

        func recorded(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        var isolated = [(String, MLXArray)]()
        for (index, block) in blocks.enumerated() {
            isolated.append(("tower block \(index)", block(try recorded("vit.hidden.\(index)"))))
        }
        isolated.append(("projector", net.projector(try recorded("vit.hidden.\(blocks.count)")[0..., 1...])))
        let mask = NFKMLXLanguageNet.causalMask(inputIds.count, offset: 0)
        let layers = net.language.model.layers
        for (index, layer) in layers.enumerated() {
            var output = layer(try recorded("dec.hidden.\(index)").expandedDimensions(axis: 0), mask: mask, cache: nil, layer: index)
            if index == layers.count - 1 { output = net.language.model.norm(output) }
            isolated.append(("decoder layer \(index)", output[0]))
        }
        let last = try recorded("dec.hidden.\(layers.count)").expandedDimensions(axis: 0)
        isolated.append(("logits", net.language.logits(fromHidden: last)[0, (inputIds.count - answer)...]))
        eval(isolated.map(\.1))
        let isolatedKeys = (1 ... blocks.count).map { "vit.hidden.\($0)" } + ["vit_embeds"]
            + (1 ... layers.count).map { "dec.hidden.\($0)" } + ["output"]
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolatedKeys)
        report("\(label) isolated", isolatedRows)
        assertRoundingPlacement(label, endToEnd: rows, isolated: isolatedRows)
    }

    /// Each piece of one InternVL release's first and last tower blocks and of its projector on the
    /// reference's own bf16 input.
    private func sa2vaInternVLPieces(_ label: String, _ net: NFKMLXSa2VANet, _ probe: [String: MLXArray]) throws {
        func input(_ key: String) throws -> MLXArray {
            try XCTUnwrap(probe[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        var lines = [String]()
        let pixels = try XCTUnwrap(probe["pixel_values"]).asType(.bfloat16)
        lines.append(try piece("embeddings", net.vision.embeddings(pixels.transposed(0, 2, 3, 1)), probe, "vit.hidden.0"))
        let blocks = net.vision.encoder.layers
        for index in [0, blocks.count - 1] {
            let block = blocks[index]
            let p = "\(index)."
            lines.append("tower block \(index)")
            lines.append(try piece("norm1", block.norm1(input(p + "norm1.in")), probe, p + "norm1.out"))
            lines.append(try piece("qkv", block.attention.qkv(input(p + "attn.qkv.in")), probe, p + "attn.qkv.out"))
            if let queryNorm = block.attention.queryNorm, let keyNorm = block.attention.keyNorm {
                lines.append(try piece("q_norm", queryNorm(input(p + "attn.q_norm.in")), probe, p + "attn.q_norm.out"))
                lines.append(try piece("k_norm", keyNorm(input(p + "attn.k_norm.in")), probe, p + "attn.k_norm.out"))
            }
            lines.append(try piece("attn", block.attention(input(p + "attn.in")), probe, p + "attn.out"))
            lines.append(try piece("proj", block.attention.proj(input(p + "attn.proj.in")), probe, p + "attn.proj.out"))
            lines.append(try piece("norm2", block.norm2(input(p + "norm2.in")), probe, p + "norm2.out"))
            lines.append(try piece("fc1", block.mlp.fc1(input(p + "mlp.fc1.in")), probe, p + "mlp.fc1.out"))
            lines.append(try piece("gelu", NFKReferenceRounding.wide(input(p + "mlp.act.in")) { gelu($0) }, probe, p + "mlp.act.out"))
            lines.append(try piece("fc2", block.mlp.fc2(input(p + "mlp.fc2.in")), probe, p + "mlp.fc2.out"))
            lines.append(try piece("mlp", block.mlp(input(p + "mlp.in")), probe, p + "mlp.out"))
            lines.append(try piece("block", block(input(p + "block.in")), probe, p + "block.out"))
        }
        let projector = net.projector
        lines.append("projector")
        lines.append(try piece("norm", projector.norm(input("mlp1.0.in")), probe, "mlp1.0.out"))
        lines.append(try piece("fc1", projector.fc1(input("mlp1.1.in")), probe, "mlp1.1.out"))
        lines.append(try piece("gelu", NFKReferenceRounding.wide(input("mlp1.2.in")) { gelu($0) }, probe, "mlp1.2.out"))
        lines.append(try piece("fc2", projector.fc2(input("mlp1.3.in")), probe, "mlp1.3.out"))
        print("VALIDATION bf16 \(label) pieces:\n" + lines.joined(separator: "\n"))
    }

    // Sa2VA-LLaVA-1.5-7B cut to its first four decoder layers (`IK_VAL_SA2VA_LLAVA_CUT4`) at its default bf16
    // load, against the release's own code at bf16 with eager attention (`sa2va_llava_probe`, the float32 side
    // on the same bf16 weights): every CLIP layer and decoder layer on the reference's own input, the projector,
    // the final norm, and the logits over the answer, then each piece of the first and last CLIP layers.
    func testSa2VALLaVACutInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_SA2VA_LLAVA_CUT4"], "IK_VAL_SA2VA_LLAVA_CUT4"))
        let bf16 = try record("sa2va_llava_cut4_bf16.safetensors"), f32 = try record("sa2va_llava_cut4_f32.safetensors")
        let net = try NFKMLXSa2VALLaVANet.load(directoryURL: directory)
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let pixels = try recorded("pixel_values")
        let ids = try XCTUnwrap(bf16["input_ids"]).asArray(Int32.self).map(Int.init)
        let layers = net.vision.encoder.layers

        var states = [(String, MLXArray)](), keys = [String]()
        var isolated = [(String, MLXArray)](), isolatedKeys = [String]()
        var hidden = net.vision.hiddenState(pixels, afterLayers: 0)
        XCTAssertEqual(hidden.dtype, .bfloat16, "a bf16 load computes the tower at bf16")
        states.append(("tower input", hidden[0]))
        keys.append("vit.hidden.0")
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden)
            states.append(("tower layer \(index)", hidden[0]))
            keys.append("vit.hidden.\(index + 1)")
            isolated.append(("tower layer \(index)", layer(try recorded("vit.hidden.\(index)"))[0]))
            isolatedKeys.append("vit.hidden.\(index + 1)")
        }
        let selected = layers.count - 1
        let features = net.imageFeatures(pixels)
        states.append(("projector", features[0]))
        keys.append("features")
        isolated.append(("projector", net.projector(try recorded("vit.hidden.\(selected)")[0..., 1...])[0]))
        isolatedKeys.append("features")
        let fused = net.fusedEmbeddings(inputIds: ids, imageFeatures: features)
        states.append(("fused", fused[0]))
        keys.append("fused")
        let decoder = net.language.model.layers
        let mask = NFKMLXLanguageNet.causalMask(ids.count, offset: 0)
        var h = fused
        for (index, layer) in decoder.enumerated() {
            h = layer(h, mask: mask, cache: nil, layer: index)
            states.append(("decoder layer \(index)", h[0]))
            keys.append("dec.hidden.\(index + 1)")
            isolated.append(("decoder layer \(index)",
                             layer(try recorded("dec.hidden.\(index)"), mask: mask, cache: nil, layer: index)[0]))
            isolatedKeys.append("dec.hidden.\(index + 1)")
        }
        let normed = net.language.model.norm(h)
        states.append(("final norm", normed[0]))
        keys.append("dec.final")
        isolated.append(("final norm", net.language.model.norm(try recorded("dec.hidden.\(decoder.count)"))[0]))
        isolatedKeys.append("dec.final")
        let answer = try XCTUnwrap(bf16["output"]).dim(0)
        let tail = ids.count - answer
        states.append(("logits", net.language.logits(fromHidden: normed)[0, tail...]))
        keys.append("output")
        isolated.append(("logits", net.language.logits(fromHidden: try recorded("dec.final"))[0, tail...]))
        isolatedKeys.append("output")
        eval(states.map(\.1) + isolated.map(\.1))
        let rows = try seams(states, bf16: bf16, f32: f32, keys: keys)
        report("sa2va-llava-cut4", rows)
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolatedKeys)
        report("sa2va-llava-cut4 isolated", isolatedRows)
        assertRoundingPlacement("sa2va-llava-cut4", endToEnd: rows, isolated: isolatedRows)

        var lines = [String]()
        for index in [0, layers.count - 1] {
            let layer = layers[index], p = "vit.\(index)."
            func input(_ name: String) throws -> MLXArray { try recorded(p + name + ".in") }
            lines.append("tower layer \(index)")
            lines.append(try piece("layer_norm1", layer.norm1(input("layer_norm1")), bf16, p + "layer_norm1.out"))
            lines.append(try piece("q_proj", layer.attention.q(input("self_attn.q_proj")), bf16, p + "self_attn.q_proj.out"))
            lines.append(try piece("out_proj", layer.attention.out(input("self_attn.out_proj")), bf16, p + "self_attn.out_proj.out"))
            lines.append(try piece("layer_norm2", layer.norm2(input("layer_norm2")), bf16, p + "layer_norm2.out"))
            lines.append(try piece("fc1", layer.mlp.fc1(input("mlp.fc1")), bf16, p + "mlp.fc1.out"))
            lines.append(try piece("quick_gelu", NFKCLIPInit.quickGELU(input("mlp.activation_fn")), bf16, p + "mlp.activation_fn.out"))
            lines.append(try piece("fc2", layer.mlp.fc2(input("mlp.fc2")), bf16, p + "mlp.fc2.out"))
            lines.append(try piece("mlp", layer.mlp(input("mlp")), bf16, p + "mlp.out"))
            lines.append(try piece("block", layer(input("block")), bf16, p + "block.out"))
        }
        print("VALIDATION bf16 sa2va-llava-cut4 pieces:\n" + lines.joined(separator: "\n"))
    }

    // Sa2VA-1B's SAM 2 grounding branch at bf16 against the release's own code at bf16
    // (`sa2va_grounding_probe`, Hiera blocks 43 and 44 probed), from the reference's own grounding
    // image and `[SEG]` embedding. The reference keeps the top-down neck level and the conditioned feature
    // in float32 and rounds them once where the decoder casts its inputs to the positional encoding's type.
    func testSa2VA1BGroundingInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (net, bf16, f32) = try sa2va1BGrounding()
        let tracker = net.grounding.tracker
        let encoder = tracker.imageEncoder
        func nchw(_ x: MLXArray) -> MLXArray { x.transposed(0, 3, 1, 2) }
        func nhwc(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16).transposed(0, 2, 3, 1)
        }
        let image = try XCTUnwrap(bf16["g_pixel_values"]).transposed(0, 2, 3, 1)
        let embedding = try XCTUnwrap(bf16["seg_embedding"]).asType(.bfloat16).reshaped([1, 1, -1])
        let ends = encoder.trunk.configuration.stageEnds

        let stages = encoder.trunk(image.asType(.bfloat16))
        let levels = Array(encoder.neck(stages).dropLast(encoder.scalp))
        XCTAssertTrue(arrayEqual(levels[2], net.grounding.imageLevels(image)[2]).item(Bool.self),
                      "the grounding entry point casts and encodes alike")
        let decoder = tracker.maskDecoder
        let conditioned = levels[2] + tracker.noMemoryEmbedding.reshaped([1, 1, 1, -1])
        let decoded = net.grounding.decode(levels: levels, languageEmbedding: embedding)
        var states = zip(ends, stages).map { ("trunk stage end \($0.0)", $0.1) }
        states += [("level 0, conv_s0", nchw(decoder.convS0(levels[0]))), ("level 1, conv_s1", nchw(decoder.convS1(levels[1]))),
                   ("level 2", nchw(levels[2])), ("conditioned", nchw(conditioned)),
                   ("ious", decoded.iou[0..., 1...]), ("masks", decoded.masks[0..., 1...]),
                   ("best mask", net.grounding.segment(image: image, languageEmbedding: embedding).lowResolution)]
        eval(states.map(\.1))
        let keys = ends.map { "sam.trunk.\($0).out" }
            + ["sam.backbone_fpn.0", "sam.backbone_fpn.1", "sam.backbone_fpn.2", "sam.conditioned", "ious", "low_res_multi", "low_res_best"]
        let rows = try seams(states, bf16: bf16, f32: f32, keys: keys)
        report("sa2va-1b grounding", rows)

        var isolated = [(String, MLXArray)]()
        let boundaries = Set(ends + ends.dropLast().map { $0 + 1 } + [0]).sorted()
        for index in boundaries {
            isolated.append(("trunk block \(index)", encoder.trunk.blocks[index](try XCTUnwrap(bf16["sam.trunk.\(index).in"]).asType(.bfloat16))))
        }
        let recordedStages = try ends.map { try XCTUnwrap(bf16["sam.trunk.\($0).out"]).asType(.bfloat16) }
        let neck = Array(encoder.neck(recordedStages).dropLast(encoder.scalp))
        isolated += [("neck level 0, conv_s0", nchw(decoder.convS0(neck[0]))), ("neck level 1, conv_s1", nchw(decoder.convS1(neck[1]))),
                     ("neck level 2", nchw(neck[2]))]
        let recordedLevels = [try nhwc("sam.sam_mask_decoder.conv_s0.in"), try nhwc("sam.sam_mask_decoder.conv_s1.in"),
                              try XCTUnwrap(bf16["sam.conditioned"]).transposed(0, 2, 3, 1)]
        let isolatedDecode = net.grounding.decode(levels: recordedLevels, conditioned: recordedLevels[2],
                                                  languageEmbedding: embedding)
        isolated.append(("decoder masks", isolatedDecode.masks[0..., 1...]))
        eval(isolated.map(\.1))
        let isolatedKeys = boundaries.map { "sam.trunk.\($0).out" }
            + ["sam.backbone_fpn.0", "sam.backbone_fpn.1", "sam.backbone_fpn.2", "low_res_multi"]
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolatedKeys)
        report("sa2va-1b grounding isolated", isolatedRows)
        assertRoundingPlacement("sa2va-1b grounding", endToEnd: rows, isolated: isolatedRows)
    }

    // Each piece of Sa2VA-1B's grounding branch on the reference's own bf16 input: Hiera blocks 43 (global
    // attention over 4,096 keys, in 512-key blocks) and 44 (query pooling into stage 4), the dense
    // positional encoding, both two-way layers, the final attention, and the upscaling path.
    func testSa2VA1BGroundingPiecesMatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (net, probe, _) = try sa2va1BGrounding()
        let tracker = net.grounding.tracker
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(probe[key], "no \(key)").asType(.bfloat16) }
        func nhwc(_ key: String) throws -> MLXArray { try input(key).transposed(0, 2, 3, 1) }
        func nchw(_ x: MLXArray) -> MLXArray { x.transposed(0, 3, 1, 2) }
        var lines = [String]()
        for index in [43, 44] {
            let block = tracker.imageEncoder.trunk.blocks[index]
            let p = "\(index)."
            lines.append("trunk block \(index)")
            lines.append(try piece("norm1", block.norm1(input(p + "norm1.in")), probe, p + "norm1.out"))
            lines.append(try piece("qkv", block.attn.qkv(input(p + "attn.qkv.in")), probe, p + "attn.qkv.out"))
            let qkv = try input(p + "attn.qkv.out")
            let (n, h, w) = (qkv.dim(0), qkv.dim(1), qkv.dim(2))
            let projected = qkv.reshaped([n, h * w, 3, block.attn.heads, -1])
            var queries = projected[0..., 0..., 0]
            var (qh, qw) = (h, w)
            if block.attn.poolsQueries {
                let pooled = NFKMLXResample.maxPooled(queries.reshaped([n, h, w, -1]), kernel: 2, stride: 2)
                (qh, qw) = (pooled.dim(1), pooled.dim(2))
                queries = pooled.reshaped([n, qh * qw, block.attn.heads, -1])
            }
            let attended = NFKReferenceRounding.flashAttention(
                queries: queries.transposed(0, 2, 1, 3), keys: projected[0..., 0..., 1].transposed(0, 2, 1, 3),
                values: projected[0..., 0..., 2].transposed(0, 2, 1, 3), scale: pow(Float(queries.dim(3)), -0.5), mask: nil)
            lines.append(try piece("attention (\(h * w) keys)", attended.transposed(0, 2, 1, 3).reshaped([n, qh, qw, -1]),
                                   probe, p + "attn.proj.in"))
            lines.append(try piece("proj", block.attn.proj(input(p + "attn.proj.in")), probe, p + "attn.proj.out"))
            lines.append(try piece("attn", block.attn(input(p + "attn.in")), probe, p + "attn.out"))
            lines.append(try piece("norm2", block.norm2(input(p + "norm2.in")), probe, p + "norm2.out"))
            lines.append(try piece("gelu", NFKReferenceRounding.wide(input(p + "mlp.act.in")) { gelu($0) }, probe, p + "mlp.act.out"))
            lines.append(try piece("mlp", block.mlp(input(p + "mlp.in")), probe, p + "mlp.out"))
            lines.append(try piece("block", block(input(p + "block.in")), probe, p + "block.out"))
        }
        let grid = tracker.configuration.featureGrid
        lines.append(try piece("dense positional", tracker.promptEncoder.positionEncoding.grid(grid, grid)[0].transposed(1, 0),
                               probe, "sam.sam_prompt_encoder.pe_layer.out"))
        let decoder = tracker.maskDecoder
        let d = "sam.sam_mask_decoder."
        for (index, layer) in decoder.layers.enumerated() {
            let l = d + "transformer.layers.\(index)."
            lines.append("two-way layer \(index)")
            for (name, attention) in [("self_attn", layer.selfAttn), ("cross_attn_token_to_image", layer.crossTokenImage),
                                      ("cross_attn_image_to_token", layer.crossImageToken)] {
                lines.append(try piece(name, attention(input(l + name + ".q_proj.in"), input(l + name + ".k_proj.in"),
                                                       input(l + name + ".v_proj.in")), probe, l + name + ".out"))
            }
            for (name, norm) in [("norm1", layer.norm1), ("norm2", layer.norm2), ("norm3", layer.norm3), ("norm4", layer.norm4)] {
                lines.append(try piece(name, norm(input(l + name + ".in")), probe, l + name + ".out"))
            }
            lines.append(try piece("mlp", layer.mlp(input(l + "mlp.in")), probe, l + "mlp.out"))
        }
        let f = d + "transformer.final_attn_token_to_image."
        lines.append(try piece("final attention", decoder.finalAttention(input(f + "q_proj.in"), input(f + "k_proj.in"),
                                                                        input(f + "v_proj.in")), probe, f + "out"))
        lines.append(try piece("norm_final_attn", decoder.normFinalAttention(input(d + "transformer.norm_final_attn.in")),
                               probe, d + "transformer.norm_final_attn.out"))
        let u = d + "output_upscaling."
        lines.append(try piece("upscale1", nchw(decoder.upscale1(nhwc(u + "0.in"))), probe, u + "0.out"))
        lines.append(try piece("LayerNorm2d", nchw(decoder.upscaleNorm(nhwc(u + "1.in"))), probe, u + "1.out"))
        lines.append(try piece("gelu", NFKReferenceRounding.wide(input(u + "2.in")) { gelu($0) }, probe, u + "2.out"))
        lines.append(try piece("upscale2", nchw(decoder.upscale2(nhwc(u + "3.in"))), probe, u + "3.out"))
        for index in 0 ..< decoder.maskCount {
            let key = d + "output_hypernetworks_mlps.\(index)."
            lines.append(try piece("hypernetwork \(index)", decoder.hyper[index](input(key + "in")), probe, key + "out"))
        }
        let iouInput = try input(d + "iou_prediction_head.in")
        lines.append(try piece("iou head", decoder.iouUsesSigmoid ? NFKReferenceRounding.sigmoid(decoder.iouHead(iouInput))
                                                                   : decoder.iouHead(iouInput),
                               probe, d + "iou_prediction_head.out"))
        print("VALIDATION bf16 sa2va-1b grounding pieces:\n" + lines.joined(separator: "\n"))
    }

    // Sa2VA-Qwen3-VL-4B-SAM3's SAM 3 grounding at bf16 against the release's own code at bf16 as CUDA runs
    // it (`sa2va_sam3_probe`: the tracker's process-wide bf16 autocast, emulated on the CPU), from the
    // reference's own grounding image and `[SEG]` embedding. Under autocast the ViT's residual stream, the
    // two-way tokens, and the keys after the first two-way layer stay float32; each record carries its
    // tensors' types, and an isolated input takes its recorded type.
    func testSa2VAQwenSAM3GroundingInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (grounding, bf16, f32) = try sa2vaSAM3Grounding()
        func recorded(_ key: String) throws -> MLXArray {
            try XCTUnwrap(bf16[key], "no \(key)").asType(bf16["\(key).dtype"]?.item(Int32.self) == 1 ? .bfloat16 : .float32)
        }
        func nchw(_ x: MLXArray) -> MLXArray { x.transposed(0, 3, 1, 2) }
        func nhwc(_ key: String) throws -> MLXArray { try recorded(key).transposed(0, 2, 3, 1) }
        let backbone = grounding.backbone, decoder = grounding.maskDecoder
        let image = try recorded("g_pixel_values").transposed(0, 2, 3, 1)
        let embedding = try recorded("seg_embedding").reshaped([1, 1, -1])
        let global = backbone.layers.indices.filter { backbone.layers[$0].windowSize == 0 }
        let blocks = Set([0, 1] + global + global.map { $0 + 1 }).filter { $0 < backbone.layers.count }.sorted()

        let patches = NFKSAM3Autocast.conv(backbone.embeddings.patchEmbeddings.projection, image)
        var hidden = NFKSAM3Autocast.layerNorm(
            backbone.layerNorm, patches + backbone.embeddings.tiled(height: patches.dim(1), width: patches.dim(2)))
        var states = [("ln_pre", hidden)]
        for (index, layer) in backbone.layers.enumerated() {
            hidden = NFKSAM3Autocast.layer(layer, hidden)
            if blocks.contains(index) { states.append(("trunk block \(index)", hidden)) }
        }
        let levels = grounding.neck.fpnLayers.prefix(3).map { NFKSAM3Autocast.level($0, hidden) }
        XCTAssertTrue(arrayEqual(levels[2], grounding.imageLevels(image)[2]).item(Bool.self),
                      "the grounding entry point encodes alike")
        let conditioned = levels[2] + grounding.noMemoryEmbedding.reshaped([1, 1, 1, -1])
        let decoded = try sam3Decode(grounding, levels: levels, conditioned: conditioned, embedding: embedding)
        states += [("level 0, conv_s0", nchw(NFKSAM3Autocast.conv(decoder.convS0, levels[0]))),
                   ("level 1, conv_s1", nchw(NFKSAM3Autocast.conv(decoder.convS1, levels[1]))),
                   ("level 2", nchw(levels[2])), ("conditioned", nchw(conditioned)),
                   ("ious", decoded.iou[0..., 1...]), ("masks", decoded.masks[0..., 1...]),
                   ("best mask", grounding.segment(image: image, languageEmbedding: embedding).lowResolution)]
        eval(states.map(\.1))
        let keys = ["sam.trunk.ln_pre.out"] + blocks.map { "sam.trunk.\($0).out" }
            + ["sam.backbone_fpn.0", "sam.backbone_fpn.1", "sam.backbone_fpn.2", "sam.conditioned", "ious",
               "low_res_multi", "low_res_best"]
        let rows = try seams(states, bf16: bf16, f32: f32, keys: keys)
        report("sa2va-qwen3-vl-4b-sam3 grounding", rows)

        var isolated = [("ln_pre", NFKSAM3Autocast.layerNorm(backbone.layerNorm, try recorded("sam.trunk.ln_pre.in")))]
        for index in blocks {
            isolated.append(("trunk block \(index)",
                             NFKSAM3Autocast.layer(backbone.layers[index], try recorded("sam.trunk.\(index).in"))))
        }
        let last = try recorded("sam.trunk.\(backbone.layers.count - 1).out")
        let neck = grounding.neck.fpnLayers.prefix(3).map { NFKSAM3Autocast.level($0, last) }
        isolated += [("neck level 0, conv_s0", nchw(NFKSAM3Autocast.conv(decoder.convS0, neck[0]))),
                     ("neck level 1, conv_s1", nchw(NFKSAM3Autocast.conv(decoder.convS1, neck[1]))),
                     ("neck level 2", nchw(neck[2]))]
        let recordedLevels = [try nhwc("sam.sam_mask_decoder.conv_s0.in"), try nhwc("sam.sam_mask_decoder.conv_s1.in")]
        let isolatedDecode = try sam3Decode(grounding, levels: recordedLevels, conditioned: try nhwc("sam.conditioned"),
                                            embedding: embedding)
        isolated.append(("decoder masks", isolatedDecode.masks[0..., 1...]))
        eval(isolated.map(\.1))
        let isolatedKeys = ["sam.trunk.ln_pre.out"] + blocks.map { "sam.trunk.\($0).out" }
            + ["sam.backbone_fpn.0", "sam.backbone_fpn.1", "sam.backbone_fpn.2", "low_res_multi"]
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolatedKeys)
        report("sa2va-qwen3-vl-4b-sam3 grounding isolated", isolatedRows)
        assertRoundingPlacement("sa2va-qwen3-vl-4b-sam3 grounding", endToEnd: rows, isolated: isolatedRows)
    }

    // Each piece of Sa2VA-Qwen3-VL-4B-SAM3's grounding on the reference's own input in its recorded type:
    // ViT blocks 0 (windowed, 576 keys) and 7 (global, 5,184 keys, in 512-key blocks), each tracker neck
    // level, the dense positional encoding, both two-way layers, the final attention, the upscaling path,
    // and the heads.
    func testSa2VAQwenSAM3GroundingPiecesMatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let (grounding, probe, _) = try sa2vaSAM3Grounding()
        func input(_ key: String) throws -> MLXArray {
            try XCTUnwrap(probe[key], "no \(key)").asType(probe["\(key).dtype"]?.item(Int32.self) == 1 ? .bfloat16 : .float32)
        }
        func nhwc(_ key: String) throws -> MLXArray { try input(key).transposed(0, 2, 3, 1) }
        func nchw(_ x: MLXArray) -> MLXArray { x.transposed(0, 3, 1, 2) }
        var lines = [String]()
        for index in [0, 7] {
            let block = grounding.backbone.layers[index]
            let attention = block.attention
            let p = "\(index)."
            lines.append("ViT block \(index)")
            lines.append(try piece("norm1", NFKSAM3Autocast.layerNorm(block.norm1, input(p + "norm1.in")), probe, p + "norm1.out"))
            let x = try input(p + "attn.qkv.in")
            lines.append(try piece("qkv", concatenated([attention.qProj, attention.kProj, attention.vProj].map {
                NFKSAM3Autocast.linear($0, x)
            }, axis: -1), probe, p + "attn.qkv.out"))
            let qkv = try input(p + "attn.qkv.out")
            let (n, h, w) = (qkv.dim(0), qkv.dim(1), qkv.dim(2))
            let heads = qkv.reshaped([n, h * w, 3, attention.heads, attention.headDim]).transposed(2, 0, 3, 1, 4)
            func rotated(_ x: MLXArray) -> MLXArray { block.rotary(x.asType(.float32)).asType(x.dtype) }
            let attended = NFKReferenceRounding.flashAttention(
                queries: rotated(heads[0]), keys: rotated(heads[1]), values: heads[2],
                scale: 1 / sqrt(Float(attention.headDim)), mask: nil)
            lines.append(try piece("attention (\(h * w) keys)", attended.transposed(0, 2, 1, 3).reshaped([n, h, w, -1]),
                                   probe, p + "attn.proj.in"))
            lines.append(try piece("proj", NFKSAM3Autocast.linear(attention.oProj, input(p + "attn.proj.in")), probe,
                                   p + "attn.proj.out"))
            lines.append(try piece("attn", NFKSAM3Autocast.attention(attention, input(p + "attn.in"), rotary: block.rotary),
                                   probe, p + "attn.out"))
            lines.append(try piece("norm2", NFKSAM3Autocast.layerNorm(block.norm2, input(p + "norm2.in")), probe, p + "norm2.out"))
            lines.append(try piece("fc1 + tanh GELU", NFKSAM3Autocast.widened(block.mlp, input(p + "mlp.in")), probe,
                                   p + "mlp.fc2.in"))
            lines.append(try piece("fc2", NFKSAM3Autocast.linear(block.mlp.fc2, input(p + "mlp.fc2.in")), probe, p + "mlp.fc2.out"))
            lines.append(try piece("mlp", NFKSAM3Autocast.mlp(block.mlp, input(p + "mlp.in")), probe, p + "mlp.out"))
            lines.append(try piece("block", NFKSAM3Autocast.layer(block, input(p + "block.in")), probe, p + "block.out"))
        }
        let levels = grounding.neck.fpnLayers
        let up = levels[0].scaleLayers
        let n0 = "sam.neck.0.", n1 = "sam.neck.1.", n2 = "sam.neck.2."
        lines.append("tracker neck")
        lines.append(try piece("level 0 dconv 0", nchw(NFKSAM3Autocast.convTransposed(up[0] as! ConvTransposed2d,
                                                                                        nhwc(n0 + "dconv_2x2_0.in"))),
                               probe, n0 + "dconv_2x2_0.out"))
        lines.append(try piece("level 0 gelu", NFKSAM3Autocast.gelu(input(n0 + "gelu.in")), probe, n0 + "gelu.out"))
        lines.append(try piece("level 0 dconv 1", nchw(NFKSAM3Autocast.convTransposed(up[2] as! ConvTransposed2d,
                                                                                        nhwc(n0 + "dconv_2x2_1.in"))),
                               probe, n0 + "dconv_2x2_1.out"))
        lines.append(try piece("level 1 dconv", nchw(NFKSAM3Autocast.convTransposed(
            levels[1].scaleLayers[0] as! ConvTransposed2d, nhwc(n1 + "dconv_2x2.in"))), probe, n1 + "dconv_2x2.out"))
        for (index, prefix) in [n0, n1, n2].enumerated() {
            lines.append(try piece("level \(index) conv_1x1", nchw(NFKSAM3Autocast.conv(levels[index].proj1, nhwc(prefix + "conv_1x1.in"))),
                                   probe, prefix + "conv_1x1.out"))
            lines.append(try piece("level \(index) conv_3x3", nchw(NFKSAM3Autocast.conv(levels[index].proj2, nhwc(prefix + "conv_3x3.in"))),
                                   probe, prefix + "conv_3x3.out"))
        }
        let grid = grounding.configuration.grid
        lines.append(try piece("dense positional", grounding.promptEncoder.positionEncoding.grid(grid, grid)[0].transposed(1, 0),
                               probe, "sam.sam_prompt_encoder.pe_layer.out"))
        let decoder = grounding.maskDecoder
        let d = "sam.sam_mask_decoder."
        lines.append(try piece("conv_s0", nchw(NFKSAM3Autocast.conv(decoder.convS0, nhwc(d + "conv_s0.in"))), probe, d + "conv_s0.out"))
        lines.append(try piece("conv_s1", nchw(NFKSAM3Autocast.conv(decoder.convS1, nhwc(d + "conv_s1.in"))), probe, d + "conv_s1.out"))
        for (index, layer) in decoder.layers.enumerated() {
            let l = d + "transformer.layers.\(index)."
            lines.append("two-way layer \(index)")
            for (name, attention) in [("self_attn", layer.selfAttn), ("cross_attn_token_to_image", layer.crossTokenImage),
                                      ("cross_attn_image_to_token", layer.crossImageToken)] {
                lines.append(try piece(name, NFKSAM3Autocast.attention(attention, input(l + name + ".q_proj.in"),
                                                                       input(l + name + ".k_proj.in"), input(l + name + ".v_proj.in")),
                                       probe, l + name + ".out"))
            }
            for (name, norm) in [("norm1", layer.norm1), ("norm2", layer.norm2), ("norm3", layer.norm3), ("norm4", layer.norm4)] {
                lines.append(try piece(name, NFKSAM3Autocast.layerNorm(norm, input(l + name + ".in")), probe, l + name + ".out"))
            }
            lines.append(try piece("mlp", NFKSAM3Autocast.perceptron(layer.mlp, input(l + "mlp.in")), probe, l + "mlp.out"))
        }
        let f = d + "transformer.final_attn_token_to_image."
        lines.append(try piece("final attention", NFKSAM3Autocast.attention(decoder.finalAttention, input(f + "q_proj.in"),
                                                                            input(f + "k_proj.in"), input(f + "v_proj.in")),
                               probe, f + "out"))
        lines.append(try piece("norm_final_attn", NFKSAM3Autocast.layerNorm(decoder.normFinalAttention,
                                                                            input(d + "transformer.norm_final_attn.in")),
                               probe, d + "transformer.norm_final_attn.out"))
        let u = d + "output_upscaling."
        lines.append(try piece("upscale1", nchw(NFKSAM3Autocast.convTransposed(decoder.upscale1, nhwc(u + "0.in"))), probe, u + "0.out"))
        lines.append(try piece("LayerNorm2d", nchw(NFKSAM3Autocast.layerNorm2d(decoder.upscaleNorm, nhwc(u + "1.in"))), probe, u + "1.out"))
        lines.append(try piece("gelu", NFKSAM3Autocast.gelu(input(u + "2.in")), probe, u + "2.out"))
        lines.append(try piece("upscale2", nchw(NFKSAM3Autocast.convTransposed(decoder.upscale2, nhwc(u + "3.in"))), probe, u + "3.out"))
        lines.append(try piece("gelu", NFKSAM3Autocast.gelu(input(u + "4.in")), probe, u + "4.out"))
        for index in 0 ..< decoder.maskCount {
            let key = d + "output_hypernetworks_mlps.\(index)."
            lines.append(try piece("hypernetwork \(index)", NFKSAM3Autocast.perceptron(decoder.hyper[index], input(key + "in")),
                                   probe, key + "out"))
        }
        let iou = NFKSAM3Autocast.perceptron(decoder.iouHead, try input(d + "iou_prediction_head.in"))
        lines.append(try piece("iou head", decoder.iouUsesSigmoid ? NFKReferenceRounding.sigmoid(iou) : iou, probe,
                               d + "iou_prediction_head.out"))
        lines.append(try piece("object score head", NFKSAM3Autocast.perceptron(decoder.objScoreHead,
                                                                               input(d + "pred_obj_score_head.in")),
                               probe, d + "pred_obj_score_head.out"))
        print("VALIDATION bf16 sa2va-qwen3-vl-4b-sam3 grounding pieces:\n" + lines.joined(separator: "\n"))
    }

    /// The SAM 3 grounding's decoder under autocast on `levels` (the two finer ones) and `conditioned`.
    private func sam3Decode(_ grounding: NFKSa2VASAM3GroundingEncoder, levels: [MLXArray], conditioned: MLXArray,
                            embedding: MLXArray) throws -> (masks: MLXArray, iou: MLXArray, objectScore: MLXArray) {
        let grid = grounding.configuration.grid
        let encoder = grounding.promptEncoder
        let empty = encoder.sparse(points: [(x: Float(0), y: Float(0), label: -1)])
        return NFKSAM3Autocast.decode(grounding.maskDecoder, features: conditioned,
                                      positional: encoder.positionEncoding.grid(grid, grid),
                                      sparse: concatenated([empty.asType(.float32), embedding.asType(.float32)], axis: 1),
                                      dense: encoder.dense(grid: grid), highResolution: [levels[0], levels[1]])
    }

    /// Sa2VA-Qwen3-VL-4B-SAM3's grounding at bf16, as the backend loads it, with both grounding records.
    private func sa2vaSAM3Grounding() throws
        -> (grounding: NFKSa2VASAM3GroundingEncoder, bf16: [String: MLXArray], f32: [String: MLXArray]) {
        let key = "IK_VAL_SA2VA_QWEN3_VL_4B_SAM3"
        let directory = URL(fileURLWithPath: try existing(config[key], key))
        let bf16 = try record("sa2va_qwen3_vl_4b_sam3_grounding_bf16.safetensors")
        let f32 = try record("sa2va_qwen3_vl_4b_sam3_grounding_f32.safetensors")
        let net = try NFKMLXSa2VAQwenNet.load(directoryURL: directory, parts: .grounding, dtype: .bfloat16)
        return (try XCTUnwrap(net.grounding as? NFKSa2VASAM3GroundingEncoder), bf16, f32)
    }

    /// Sa2VA-1B at bf16 with both grounding records.
    private func sa2va1BGrounding() throws -> (net: NFKMLXSa2VANet, bf16: [String: MLXArray], f32: [String: MLXArray]) {
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_SA2VA_1B"], "IK_VAL_SA2VA_1B"))
        let bf16 = try record("sa2va_1b_sam2_bf16.safetensors")
        let f32 = try record("sa2va_1b_sam2_f32.safetensors")
        let net = NFKMLXSa2VANet(try NFKMLXSa2VANet.configuration(fromDirectory: directory))
        try net.loadWeights(fromDirectory: directory, dtype: .bfloat16)
        return (net, bf16, f32)
    }

    /// Sa2VA-1B at bf16, as ``NFKMLXSa2VA/backend(directoryURL:)`` loads it, with both records.
    /// The releases the Qwen-VL bf16 tests read: Sa2VA's (records `sa2va_<name>_*`) and plain Qwen3-VL
    /// cuts (records `<name>_*`).
    private static let qwenVLReleases = [
        ("sa2va_qwen3_vl_2b", "IK_VAL_SA2VA_QWEN3_VL_2B"), ("sa2va_qwen2_5_vl_3b", "IK_VAL_SA2VA_QWEN2_5_VL_3B"),
        ("sa2va_qwen3_vl_4b", "IK_VAL_SA2VA_QWEN3_VL_4B"), ("sa2va_qwen3_vl_4b_sam3", "IK_VAL_SA2VA_QWEN3_VL_4B_SAM3"),
        ("sa2va_qwen2_5_vl_7b_cut4", "IK_VAL_SA2VA_QWEN2_5_VL_7B_CUT4"), ("qwen3_vl_8b_cut4", "IK_VAL_QWEN3_VL_8B_CUT4"), ("qwen3_vl_32b_cut4", "IK_VAL_QWEN3_VL_32B_CUT4"),
        ("qwen3_vl_30b_a3b_cut2", "IK_VAL_QWEN3_VL_30B_A3B_CUT2"),
        ("qwen3_vl_embedding_8b_cut4", "IK_VAL_QWEN3_VL_EMBEDDING_8B_CUT4"),
    ]

    // Qwen-VL loaded at bf16 as its backends load it: Sa2VA on Qwen3-VL (`IK_VAL_SA2VA_QWEN3_VL_2B`, `_4B`,
    // `_4B_SAM3`) and on Qwen2.5-VL (`IK_VAL_SA2VA_QWEN2_5_VL_3B`), and the larger Qwen3-VL releases cut to their first decoder
    // layers (`truncate.py`, the tower whole). The vision tower, its mergers, and the decoder over the fused
    // prompt and the teacher-forced answer, against the release's own code at bf16 with eager attention
    // (`sa2va_qwen_probe`, transformers 4.57). Qwen2.5-VL's tower states are in window order. A mixture cut's
    // record carries the reference's routing (`IK_PROBE_ROUTES=1`), which the decoder is forced to.
    func testSa2VAQwenInBFloat16MatchesTheBFloat16Reference() throws {
        var measured = 0
        // `IK_QWEN_VL_ONLY=<record name>` measures one release.
        let only = ProcessInfo.processInfo.environment["IK_QWEN_VL_ONLY"]
        for (name, key) in Self.qwenVLReleases where only == nil || only == name {
            do {
                try autoreleasepool { try sa2vaQwen(name, directoryKey: key) }
            } catch is XCTSkip {
                continue
            }
            Memory.clearCache()
            measured += 1
        }
        if measured == 0 { throw XCTSkip("set IK_VAL_SA2VA_QWEN3_VL_2B, IK_VAL_SA2VA_QWEN2_5_VL_3B, or a Qwen3-VL cut") }
    }

    private func sa2vaQwen(_ name: String, directoryKey: String) throws {
        try requireMLXRuntime()
        let (net, bf16, f32) = try sa2vaQwenRelease(name, directoryKey: directoryKey)
        let routed = bf16["route.0.index"] != nil
        if routed {
            for (index, layer) in net.decoder.model.layers.enumerated() {
                (layer.feedForward as? NFKLMMixtureFeedForward)?.forcedChoice = try recordedChoice(bf16, layer: index)
            }
        }
        func recorded(_ key: String) throws -> MLXArray { try XCTUnwrap(bf16[key], "no \(key)").asType(.bfloat16) }
        let pixels = try XCTUnwrap(bf16["pixel_values"])
        let gridValues = try XCTUnwrap(bf16["image_grid_thw"]).reshaped([-1]).asArray(Int32.self).map(Int.init)
        let grid = (t: gridValues[0], h: gridValues[1], w: gridValues[2])
        let inputIds = try XCTUnwrap(bf16["input_ids"]).asArray(Int32.self).map(Int.init)

        var states = [(String, MLXArray)](), keys = [String]()
        var isolated = [(String, MLXArray)](), isolatedKeys = [String]()
        func measure(_ label: String, _ ours: MLXArray, _ key: String, alone: MLXArray? = nil) {
            states.append((label, ours))
            keys.append(key)
            isolated.append((label, alone ?? ours))
            isolatedKeys.append(key)
        }
        switch net.tower {
        case .qwen3(let vision):
            let patch = vision.patchEmbed(pixels.asType(.bfloat16))
            let position = vision.interpolatedPositionEmbedding(grid: grid)
            measure("patch embedding", patch, "vit.patch")
            measure("position", position, "vit.pos")
            var hidden = patch + position
            measure("tower input", hidden, "vit.hidden.0", alone: try recorded("vit.patch") + recorded("vit.pos"))
            let (cos, sin) = vision.rotaryEmbedding(grid: grid)
            for (index, block) in vision.blocks.enumerated() {
                hidden = block(hidden, cos: cos, sin: sin)
                measure("tower block \(index)", hidden, "vit.hidden.\(index + 1)",
                        alone: block(try recorded("vit.hidden.\(index)"), cos: cos, sin: sin))
            }
            for (index, merger) in vision.deepstackMergers.enumerated() {
                let layer = vision.configuration.deepstackLayers[index]
                isolated.append(("deepstack \(index)", merger(try recorded("vit.hidden.\(layer + 1)"))))
                isolatedKeys.append("deepstack.\(index).block.out")
            }
            isolated.append(("merger", vision.merger(try recorded("vit.hidden.\(vision.blocks.count)"))))
            isolatedKeys.append("merger.block.out")
        case .qwen25(let vision):
            let unit = vision.configuration.spatialMergeSize * vision.configuration.spatialMergeSize
            let layout = vision.layout(grid: grid)
            let patch = vision.patchEmbed(pixels.asType(.bfloat16))
            measure("patch embedding", patch, "vit.patch")
            var hidden = patch.reshaped([patch.dim(0) / unit, unit, -1]).take(layout.order, axis: 0)
                .reshaped([patch.dim(0), -1])
            for index in vision.blocks.indices {
                hidden = vision.block(index, hidden, layout: layout)
                measure("tower block \(index)", hidden, "vit.hidden.\(index + 1)",
                        alone: vision.block(index, try recorded("vit.hidden.\(index)"), layout: layout))
            }
            isolated.append(("merger", vision.merger(try recorded("vit.hidden.\(vision.blocks.count)"), mergeUnit: unit)))
            isolatedKeys.append("merger.block.out")
        }
        let features = net.imageFeatures(pixelValues: pixels, grid: grid)
        XCTAssertEqual(features.output.dtype, .bfloat16, "\(name): a bf16 load computes the image path in bf16")
        states.append(("vision output", features.output))
        keys.append("vision_merged")
        for (index, stack) in features.deepstack.enumerated() {
            states.append(("deepstack \(index)", stack))
            keys.append("deepstack.\(index).block.out")
        }

        let decoder = net.decoder
        let input = NFKMLXQwen3VL.decoderInput(decoder: decoder, inputIds: inputIds, visionFeatures: features.output,
                                               deepstack: features.deepstack, gridT: grid.t, gridH: grid.h,
                                               gridW: grid.w, layout: net.layout)
        XCTAssertEqual(input.embeddings.dtype, .bfloat16, "\(name): the fused embeddings stay bf16")
        states.append(("fused", input.embeddings[0]))
        keys.append("fused")
        let multimodal = input.multimodal
        let mask = NFKMLXLanguageNet.causalMask(inputIds.count, offset: 0)
        let layers = decoder.model.layers
        var hidden = input.embeddings
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: nil, layer: index, rope: multimodal.rope)
            let alone = layer(try recorded("dec.layer.\(index).in").expandedDimensions(axis: 0), mask: mask, cache: nil,
                              layer: index, rope: multimodal.rope)
            measure("decoder layer \(index)", hidden[0], "dec.layer.\(index).out", alone: alone[0])
            if index < multimodal.features.count {
                let gathered = multimodal.features[index].take(multimodal.featureIndex, axis: 0)
                    .reshaped([1, inputIds.count, -1]).asType(hidden.dtype)
                hidden = hidden + MLX.where(multimodal.mask, gathered, MLXArray(Float(0)).asType(hidden.dtype))
            }
        }
        let normed = decoder.model.norm(hidden)
        // A cut whose last layer still takes a deepstack map adds the reference's map before the norm.
        var lastState = try recorded("dec.layer.\(layers.count - 1).out").expandedDimensions(axis: 0)
        if layers.count - 1 < multimodal.features.count {
            let gathered = try recorded("deepstack.\(layers.count - 1).block.out")
                .take(multimodal.featureIndex, axis: 0).reshaped([1, inputIds.count, -1])
            lastState = lastState + MLX.where(multimodal.mask, gathered, MLXArray(Float(0)).asType(lastState.dtype))
        }
        let normAlone = decoder.model.norm(lastState)[0]
        measure("final norm", normed[0], "dec.norm.out", alone: normAlone)
        let answer = try XCTUnwrap(bf16["output"]).dim(0)
        let tail = inputIds.count - answer
        // A release without an output head (the embedding model) records its final states there.
        if bf16["no_head"] != nil {
            measure("final states", normed[0, tail...], "output", alone: normAlone[tail...])
        } else {
            measure("logits", decoder.logits(fromHidden: normed)[0, tail...], "output",
                    alone: decoder.logits(fromHidden: try recorded("dec.norm.out").expandedDimensions(axis: 0))[0, tail...])
        }
        eval(states.map(\.1) + isolated.map(\.1))
        let rows = try seams(states, bf16: bf16, f32: f32, keys: keys)
        report(name, rows)
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolatedKeys)
        report("\(name) isolated", isolatedRows)
        assertRoundingPlacement(name, endToEnd: rows, isolated: isolatedRows)
    }

    // Each piece of the probed Qwen-VL tower blocks and of the merger on the reference's own bf16 input,
    // so a rounding placed differently shows as a count at the one seam that places it.
    func testSa2VAQwenTowerPiecesMatchTheBFloat16Reference() throws {
        var measured = 0
        for (name, key) in Self.qwenVLReleases {
            do {
                try autoreleasepool { try sa2vaQwenPieces(name, directoryKey: key) }
            } catch is XCTSkip {
                continue
            }
            Memory.clearCache()
            measured += 1
        }
        if measured == 0 { throw XCTSkip("set IK_VAL_SA2VA_QWEN3_VL_2B, IK_VAL_SA2VA_QWEN2_5_VL_3B, or a Qwen3-VL cut") }
    }

    private func sa2vaQwenPieces(_ name: String, directoryKey: String) throws {
        try requireMLXRuntime()
        let (net, probe, _) = try sa2vaQwenRelease(name, directoryKey: directoryKey)
        func input(_ key: String) throws -> MLXArray { try XCTUnwrap(probe[key], "no \(key)").asType(.bfloat16) }
        let gridValues = try XCTUnwrap(probe["image_grid_thw"]).reshaped([-1]).asArray(Int32.self).map(Int.init)
        let grid = (t: gridValues[0], h: gridValues[1], w: gridValues[2])
        let probed = Set(probe.keys.compactMap { key -> Int? in
            let parts = key.split(separator: ".")
            return parts.count > 2 && parts[0] == "vit" && parts[2] == "block" ? Int(parts[1]) : nil
        }).sorted()
        var lines = [String]()
        switch net.tower {
        case .qwen3(let vision):
            let (cos, sin) = vision.rotaryEmbedding(grid: grid)
            for index in probed {
                let block = vision.blocks[index]
                let p = "vit.\(index)."
                lines.append("tower block \(index)")
                lines.append(try piece("norm1", block.norm1(input(p + "norm1.in")), probe, p + "norm1.out"))
                lines.append(try piece("qkv", block.attention.qkv(input(p + "attn.qkv.in")), probe, p + "attn.qkv.out"))
                lines.append(try piece("attn", block.attention(input(p + "attn.in"), cos: cos, sin: sin), probe, p + "attn.out"))
                lines.append(try piece("proj", block.attention.proj(input(p + "attn.proj.in")), probe, p + "attn.proj.out"))
                lines.append(try piece("norm2", block.norm2(input(p + "norm2.in")), probe, p + "norm2.out"))
                lines.append(try piece("fc1", block.mlp.fc1(input(p + "mlp.linear_fc1.in")), probe, p + "mlp.linear_fc1.out"))
                lines.append(try piece("gelu", NFKReferenceRounding.geluTanh(input(p + "mlp.act_fn.in")), probe, p + "mlp.act_fn.out"))
                lines.append(try piece("fc2", block.mlp.fc2(input(p + "mlp.linear_fc2.in")), probe, p + "mlp.linear_fc2.out"))
                lines.append(try piece("block", block(input(p + "block.in"), cos: cos, sin: sin), probe, p + "block.out"))
            }
            let merger = vision.merger
            lines.append("merger")
            lines.append(try piece("norm", merger.norm(input("merger.norm.in")), probe, "merger.norm.out"))
            lines.append(try piece("fc1", merger.fc1(input("merger.linear_fc1.in")), probe, "merger.linear_fc1.out"))
            lines.append(try piece("gelu", NFKReferenceRounding.wide(input("merger.act_fn.in")) { gelu($0) }, probe, "merger.act_fn.out"))
            lines.append(try piece("fc2", merger.fc2(input("merger.linear_fc2.in")), probe, "merger.linear_fc2.out"))
        case .qwen25(let vision):
            let layout = vision.layout(grid: grid)
            for index in probed {
                let block = vision.blocks[index]
                let p = "vit.\(index)."
                lines.append("tower block \(index)")
                lines.append(try piece("norm1", block.norm1(input(p + "norm1.in")), probe, p + "norm1.out"))
                lines.append(try piece("qkv", block.attention.qkv(input(p + "attn.qkv.in")), probe, p + "attn.qkv.out"))
                lines.append(try piece("proj", block.attention.proj(input(p + "attn.proj.in")), probe, p + "attn.proj.out"))
                lines.append(try piece("norm2", block.norm2(input(p + "norm2.in")), probe, p + "norm2.out"))
                lines.append(try piece("gate", block.mlp.gate(input(p + "mlp.gate_proj.in")), probe, p + "mlp.gate_proj.out"))
                lines.append(try piece("silu", NFKReferenceRounding.silu(input(p + "mlp.act_fn.in")), probe, p + "mlp.act_fn.out"))
                lines.append(try piece("down", block.mlp.down(input(p + "mlp.down_proj.in")), probe, p + "mlp.down_proj.out"))
                lines.append(try piece("mlp", block.mlp(input(p + "mlp.in")), probe, p + "mlp.out"))
                lines.append(try piece("block", vision.block(index, input(p + "block.in"), layout: layout), probe, p + "block.out"))
            }
            let merger = vision.merger
            lines.append("merger")
            lines.append(try piece("norm", merger.norm(input("merger.ln_q.in")), probe, "merger.ln_q.out"))
            lines.append(try piece("fc1", merger.mlp[0](input("merger.mlp.0.in")), probe, "merger.mlp.0.out"))
            lines.append(try piece("gelu", NFKReferenceRounding.wide(input("merger.mlp.1.in")) { gelu($0) }, probe, "merger.mlp.1.out"))
            lines.append(try piece("fc2", merger.mlp[1](input("merger.mlp.2.in")), probe, "merger.mlp.2.out"))
        }
        print("VALIDATION bf16 \(name) pieces:\n" + lines.joined(separator: "\n"))
    }

    /// A Qwen-VL tower, decoder, and M-RoPE layout at bf16, from a Sa2VA release or a plain Qwen3-VL one.
    private struct QwenVLParts {
        let tower: NFKMLXSa2VAQwenNet.Tower
        let decoder: NFKMLXLanguageNet
        let layout: NFKMLXMRoPELayout

        func imageFeatures(pixelValues: MLXArray, grid: (t: Int, h: Int, w: Int)) -> (output: MLXArray, deepstack: [MLXArray]) {
            switch tower {
            case .qwen3(let vision): return vision(pixelValues, grid: grid)
            case .qwen25(let vision): return (vision(pixelValues, grid: grid), [])
            }
        }
    }

    private func sa2vaQwenRelease(_ name: String, directoryKey: String) throws
        -> (net: QwenVLParts, bf16: [String: MLXArray], f32: [String: MLXArray]) {
        let directory = URL(fileURLWithPath: try existing(config[directoryKey], directoryKey))
        let bf16 = try record("\(name)_bf16.safetensors"), f32 = try record("\(name)_f32.safetensors")
        guard !FileManager.default.fileExists(atPath: directory.appendingPathComponent("modeling_sa2va_qwen.py").path) else {
            let net = try NFKMLXSa2VAQwenNet.load(directoryURL: directory, parts: .language, dtype: .bfloat16)
            return (QwenVLParts(tower: net.tower, decoder: net.decoder, layout: net.layout), bf16, f32)
        }
        let vision = NFKMLXQwen3VLVisionNet(try NFKMLXQwen3VLVisionConfiguration.configuration(
            fromHuggingFace: directory.appendingPathComponent("config.json")))
        try NFKMLXQwen3VL.loadVisionWeights(into: vision, directoryURL: directory, dtype: .bfloat16)
        let decoder = try NFKMLXQwen3VL.decoder(directoryURL: directory, precision: .checkpoint, residency: .resident)
        return (QwenVLParts(tower: .qwen3(vision), decoder: decoder, layout: .interleaved), bf16, f32)
    }

    private func sa2vaInternVL(_ name: String, directoryKey: String) throws
        -> (net: NFKMLXSa2VANet, bf16: [String: MLXArray], f32: [String: MLXArray]) {
        let directory = URL(fileURLWithPath: try existing(config[directoryKey], directoryKey))
        let bf16 = try record("\(name)_bf16.safetensors"), f32 = try record("\(name)_f32.safetensors")
        let net = NFKMLXSa2VANet(try NFKMLXSa2VANet.configuration(fromDirectory: directory))
        try net.loadWeights(fromDirectory: directory, dtype: .bfloat16)
        return (net, bf16, f32)
    }

    /// The decoder's input and each layer's output over `embeddings`, the last after the final norm: the
    /// reference's `output_hidden_states` convention.
    private func decoderStates(_ net: NFKMLXLanguageNet, _ embeddings: MLXArray) -> [MLXArray] {
        let mask = NFKMLXLanguageNet.causalMask(embeddings.dim(1), offset: 0)
        var hidden = embeddings
        var states = [hidden]
        for (index, layer) in net.model.layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: nil, layer: index)
            states.append(hidden)
        }
        states[states.count - 1] = net.model.norm(hidden)
        return states
    }

    // MARK: Nemotron-H

    // Nemotron-Nano-9B-v2 cut to its first fifteen layers (`IK_VAL_NEMOTRON_CUT15`), which end on its
    // first attention layer, so the cut reaches Mamba, ReLU-squared feed-forward, and NoPE attention
    // layers: float32 against float32, then bf16 against bf16.
    func testNemotronHPrefixMatchesTheReferenceAtBothPrecisions() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_NEMOTRON_CUT15"], "IK_VAL_NEMOTRON_CUT15"))
        let bf16 = try record("nemotron_cut15_bf16.safetensors"), f32 = try record("nemotron_cut15_f32.safetensors")
        let configuration = try NFKMLXNemotronH.configuration(fromDirectory: directory)
        let tokens = MLXArray(try XCTUnwrap(f32["tokens"]).asArray(Int32.self)).reshaped([1, -1])
        // The state entering each block, the last one normalized, then the logits.
        func states(_ precision: NFKMLXWeightPrecision) throws -> (NFKMLXNemotronHNet, [MLXArray]) {
            let net = NFKMLXNemotronH.makeNet(configuration)
            try NFKMLXNemotronH.loadWeights(into: net, fromDirectory: directory, precision: precision)
            var states = net.blockStates(tokens)
            states[states.count - 1] = net.model.finalNorm(states[states.count - 1])
            states.append(net(tokens)[0])
            eval(states)
            return (net, states)
        }
        let (_, exact) = try states(.float32)
        try assertFloat32("nemotron-cut15", states: exact, f32: f32)

        let (net, reduced) = try states(.checkpoint)
        let labelled = reduced.enumerated().map { index, state in
            (index == 0 ? "embedding" : index == reduced.count - 1 ? "logits" : "layer \(index - 1)", state)
        }
        let keys = (0 ..< reduced.count - 1).map { "hidden.\($0)" } + ["output"]
        let rows = try seams(labelled, bf16: bf16, f32: f32, keys: keys)
        report("nemotron-cut15", rows)
        var isolated = [(String, MLXArray)]()
        for (index, layer) in net.model.layers.enumerated() {
            let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            var output = layer(input)
            if index == net.model.layers.count - 1 { output = net.model.finalNorm(output) }
            isolated.append(("layer \(index) (\(configuration.layerTypes[index]))", output))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolated.indices.map { "hidden.\($0 + 1)" })
        report("nemotron-cut15 isolated", isolatedRows)
        assertRoundingPlacement("nemotron-cut15", endToEnd: rows, isolated: isolatedRows)
    }

    // MARK: Granite 4.0-H

    // The released granite-4.0-h-1b at bf16, the precision its backend loads by default, against
    // transformers at bf16 with eager attention (`IK_GRANITE_DTYPE=bfloat16 run_reference.py
    // granite_hybrid_real`) and at float32 from the same ids.
    func testGraniteH1BInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GRANITE"], "IK_VAL_GRANITE"))
        let bf16 = try record("granite_h1b_bf16.safetensors")
        let f32 = try record("granite_h1b_f32.safetensors")
        let configuration = try NFKMLXGraniteHybrid.configuration(fromDirectory: directory)
        let net = NFKMLXGraniteHybrid.makeNet(configuration)
        try NFKMLXGraniteHybrid.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        let tokens = try XCTUnwrap(bf16["tokens"]).asType(.int32).reshaped([1, -1])

        var states = net.blockStates(tokens)
        states[states.count - 1] = net.model.norm(states[states.count - 1])
        let logits = net(tokens)[0]
        eval(states + [logits])
        let labelled = states.enumerated().map { ($0.offset == 0 ? "embedding" : "layer \($0.offset - 1)", $0.element) }
            + [("logits", logits)]
        let rows = try seams(labelled, bf16: bf16, f32: f32, keys: states.indices.map { "hidden.\($0)" } + ["output"])
        report("granite-h1b", rows)

        var isolated = [(String, MLXArray)]()
        for (index, layer) in net.model.layers.enumerated() {
            let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            var output = layer(input)
            if index == net.model.layers.count - 1 { output = net.model.norm(output) }
            isolated.append(("layer \(index) (\(configuration.layerTypes[index]))", output))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolated.indices.map { "hidden.\($0 + 1)" })
        report("granite-h1b isolated", isolatedRows)
        assertRoundingPlacement("granite-h1b", endToEnd: rows, isolated: isolatedRows)
    }

    // MARK: Gemma 2

    // SANA's caption encoder on its released 2B weights (`IK_VAL_GEMMA2_2B`), against transformers'
    // Gemma2Model: first at float32, where every hidden state must match, then at the released bf16
    // against the bf16 reference (`run_reference.py hf_layer_probe`, `IK_PROBE_DTYPE=bfloat16`).
    func testGemma2_2BMatchesTheReferenceAtBothPrecisions() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA2_2B"], "IK_VAL_GEMMA2_2B"))
        try gemma2("gemma2-2b", directory: directory, bf16: record("gemma2_2b_bf16_probe.safetensors"),
                   f32: record("gemma2_2b_f32.safetensors"))
    }

    // Gemma 2 27B cut to its first four layers (`IK_VAL_GEMMA2_27B_CUT4`): the one released size whose
    // query scale, `144^-0.5`, is not exact in bf16, so the scale must multiply in float32.
    func testGemma2_27BPrefixMatchesTheReferenceAtBothPrecisions() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA2_27B_CUT4"], "IK_VAL_GEMMA2_27B_CUT4"))
        try gemma2("gemma2-27b-cut4", directory: directory, bf16: record("gemma2_27b_cut4_bf16.safetensors"),
                   f32: record("gemma2_27b_cut4_f32.safetensors"))
    }

    /// A released Gemma 2 directory at float32 against the float32 record, then at bf16 end to end and
    /// layer by layer against the bf16 record.
    private func gemma2(_ name: String, directory: URL, bf16: [String: MLXArray], f32: [String: MLXArray]) throws {
        let tokens = MLXArray(try XCTUnwrap(f32["tokens"]).asArray(Int32.self))
        let exact = try loaded(onCPU: false) { try NFKMLXGemma2Net.load(directoryURL: directory, precision: .float32) }
        let exactStates = exact.layerStates(tokens)
        eval(exactStates)
        var worst = 1.0
        for (index, state) in exactStates.enumerated() {
            let similarity = 1 - distance(floats(state), floats(try XCTUnwrap(f32["hidden.\(index)"])))
            worst = min(worst, similarity)
        }
        print("VALIDATION PARITY \(name) float32: worst hidden-state cosine \(worst) over \(exactStates.count)")
        XCTAssertGreaterThan(worst, 0.99999, "\(name) matches transformers at float32")

        let net = try NFKMLXGemma2Net.load(directoryURL: directory, precision: .checkpoint)
        let states = net.layerStates(tokens)
        eval(states)
        let labelled = states.enumerated().map { ($0.offset == 0 ? "embedding" : "layer \($0.offset - 1)", $0.element) }
        let rows = try seams(labelled, bf16: bf16, f32: f32, keys: states.indices.map { "hidden.\($0)" })
        report(name, rows)

        let n = tokens.dim(0)
        let causal = net.causalMask(n, window: nil), sliding = net.causalMask(n, window: net.config.slidingWindow)
        var isolated = [(String, MLXArray)]()
        for (index, layer) in net.layers.enumerated() {
            let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16)
            var output = layer(input, mask: net.config.isSliding(index) ? sliding : causal)
            if index == net.layers.count - 1 { output = net.norm(output) }
            isolated.append(("layer \(index)", output))
        }
        eval(isolated.map(\.1))
        let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolated.indices.map { "hidden.\($0 + 1)" })
        report("\(name) isolated", isolatedRows)
        assertRoundingPlacement(name, endToEnd: rows, isolated: isolatedRows)
    }

    // Gemma 4's mixture block (the 26B-A4B family) and its unified stack (the 12B) at their tiny
    // configurations (`IK_DIT_DTYPE=bfloat16 run_reference.py gemma4_moe` / `gemma4_unified`, eager),
    // the released sizes being out of reach here: every state, then the logits. Two layers keep a
    // chained state as close to its own input as an isolated one.
    func testGemma4MixtureAndUnifiedInBFloat16MatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        for (name, unified) in [("gemma4_moe", false), ("gemma4_unified", true)] {
            let (bf16, f32, _) = try tinyDiT(name)
            let weights = bf16.compactMap { key, value -> (String, MLXArray)? in
                guard key.hasPrefix("w::model.") else { return nil }
                return (String(key.dropFirst("w::model.".count)), value.dtype.isFloatingPoint ? value.asType(.bfloat16) : value)
            }
            let tokens = MLXArray(try XCTUnwrap(bf16["tokens"]).asArray(Int32.self)).reshaped([1, -1])
            let states: [MLXArray], logits: MLXArray
            if unified {
                let net = NFKMLXGemmaLanguage.makeUnifiedNet(.unifiedTiny)
                try NFKMLXWeights.apply(weights, to: net)
                (states, logits) = (net.hiddenStates(tokens), net(tokens)[0])
            } else {
                let net = NFKMLXGemmaLanguage.makeNet(.tinyMixture)
                try NFKMLXWeights.apply(weights, to: net)
                (states, logits) = (net.hiddenStates(tokens), net(tokens)[0])
            }
            eval(states + [logits])
            let labelled = states.enumerated().map { ($0.offset == 0 ? "embedding" : "layer \($0.offset - 1)", $0.element) }
                + [("logits", logits)]
            let rows = try seams(labelled, bf16: bf16, f32: f32, keys: states.indices.map { "hidden.\($0)" } + ["output"])
            report(name, rows)
            assertRoundingPlacement(name, endToEnd: rows, isolated: Array(rows.dropFirst()))
        }
    }

    // MARK: Released sizes, cut to their first layers

    // Gemma 4's dense large sizes cut to their first six layers, which end on the first full-attention
    // layer (`truncate.py`, `IK_VAL_GEMMA4_31B_CUT6` / `IK_VAL_GEMMA4_12B_CUT6`, records from
    // `hf_layer_probe` at float32 and at `IK_PROBE_DTYPE=bfloat16`): every state at float32, then each
    // layer on the reference's own bf16 input. These are releases that set `attention_k_eq_v`; the 12B
    // is the unified stack.
    func testGemma4LargerSizePrefixesMatchTheReferenceAtBothPrecisions() throws {
        try requireMLXRuntime()
        var measured = 0
        for (name, key, unified) in [("gemma4_31b_cut6", "IK_VAL_GEMMA4_31B_CUT6", false),
                                     ("gemma4_12b_cut6", "IK_VAL_GEMMA4_12B_CUT6", true)] {
            guard let path = config[key], FileManager.default.fileExists(atPath: path),
                  let bf16 = try? record("\(name)_bf16.safetensors"),
                  let f32 = try? record("\(name)_f32.safetensors") else { continue }
            try gemma4Prefix(name, directory: URL(fileURLWithPath: path), unified: unified, bf16: bf16, f32: f32)
            measured += 1
        }
        if measured == 0 { throw XCTSkip("set IK_VAL_GEMMA4_31B_CUT6 or IK_VAL_GEMMA4_12B_CUT6") }
    }

    // The 26B-A4B mixture cut the same way (`IK_VAL_GEMMA4_26B_CUT6`), its bf16 run routed as the
    // reference routed: `hf_layer_probe` with `IK_PROBE_ROUTES=1` records each layer's kept experts, since
    // `torch.topk` breaks a tie at the `k`-th score without a fixed rule (at layer 5, token 0, experts 58
    // and 67 tie and it keeps 67). The routing weights stay this port's, computed for those experts.
    func testGemma4MixturePrefixMatchesTheReferenceAtBothPrecisionsRoutedAsTheReference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA4_26B_CUT6"], "IK_VAL_GEMMA4_26B_CUT6"))
        let routes = try record("gemma4_26b_cut6_bf16_routes.safetensors")
        // The block floors come from the routed run, whose float32 copies route as its bf16 layers did.
        let floors = routes.filter { $0.key.hasSuffix(".floor") }
        try gemma4Prefix("gemma4_26b_cut6", directory: directory, unified: false,
                         bf16: record("gemma4_26b_cut6_bf16.safetensors").merging(floors) { $1 },
                         f32: record("gemma4_26b_cut6_f32.safetensors"), routes: routes)
    }

    /// A Gemma 4 cut at float32 against the float32 record, then at bf16 end to end and layer by layer
    /// against the bf16 record. `routes` (`route.L.index` per mixture layer) fixes the bf16 run's experts.
    // Gemma 4 26B-A4B's first layer piece by piece on the reference's own bf16 inputs, in the reference
    // arithmetic and in MLX's: the cut's bf16 record probes layer 0, its router and experts records hold the
    // first token, and the layer routes as the reference routed.
    func testGemma4MixtureFirstLayerPiecesMatchTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA4_26B_CUT6"], "IK_VAL_GEMMA4_26B_CUT6"))
        let probe = try record("gemma4_26b_cut6_bf16.safetensors")
        let routes = try record("gemma4_26b_cut6_bf16_routes.safetensors")
        let net = NFKMLXGemmaLanguage.makeNet(
            try NFKMLXGemmaLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json")))
        try NFKMLXGemmaLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint,
                                            residency: .resident)
        let block = net.layers[0], attention = net.layers[0].attention
        let router = try XCTUnwrap(block.router), experts = try XCTUnwrap(block.experts)
        let choice = try XCTUnwrap(recordedChoice(routes, layer: 0), "no route.0.index")
        func input(_ key: String) throws -> MLXArray {
            let recorded = try XCTUnwrap(probe["0." + key], "no 0.\(key)").asType(.bfloat16)
            return recorded.ndim == 1 ? recorded.reshaped([1, -1]) : recorded.expandedDimensions(axis: 0)
        }
        let mask = NFKMLXLanguageNet.causalMask(try input("block.in").dim(1), offset: 0)
        func pieces() throws -> [String] {
            router.forcedChoice = choice
            var lines = [try piece("block", block(input("block.in"), perLayerInput: nil, mask: mask, shared: nil).output,
                                   probe, "0.block.out")]
            lines.append(try piece("input_layernorm", block.inputNorm(input("input_layernorm.in")), probe,
                                   "0.input_layernorm.out"))
            for (name, projection) in [("q_proj", attention.queryProjection), ("k_proj", attention.keyProjection)] {
                lines.append(try piece(name, projection(input("self_attn.\(name).in")), probe, "0.self_attn.\(name).out"))
            }
            if let valueProjection = attention.valueProjection {
                lines.append(try piece("v_proj", valueProjection(input("self_attn.v_proj.in")), probe, "0.self_attn.v_proj.out"))
            }
            lines.append(try piece("q_norm", attention.queryNorm(input("self_attn.q_norm.in")), probe, "0.self_attn.q_norm.out"))
            lines.append(try piece("k_norm", attention.keyNorm(input("self_attn.k_norm.in")), probe, "0.self_attn.k_norm.out"))
            lines.append(try piece("v_norm", NFKReferenceRounding.scaledNorm(input("self_attn.v_norm.in"), weight: nil,
                                                                             eps: attention.valueEpsilon),
                                   probe, "0.self_attn.v_norm.out"))
            lines.append(try piece("o_proj", attention.outputProjection(input("self_attn.o_proj.in")), probe,
                                   "0.self_attn.o_proj.out"))
            lines.append(try piece("self_attn", attention(input("input_layernorm.out"), mask: mask, shared: nil).output,
                                   probe, "0.self_attn.out"))
            lines.append(try piece("post_attention_layernorm", block.postAttentionNorm(input("post_attention_layernorm.in")),
                                   probe, "0.post_attention_layernorm.out"))
            lines.append(try piece("pre_feedforward_layernorm", block.preFeedForwardNorm(input("pre_feedforward_layernorm.in")),
                                   probe, "0.pre_feedforward_layernorm.out"))
            for (name, projection) in [("gate_proj", block.feedForward.gate), ("up_proj", block.feedForward.up),
                                       ("down_proj", block.feedForward.down)] {
                lines.append(try piece("mlp.\(name)", projection(input("mlp.\(name).in")), probe, "0.mlp.\(name).out"))
            }
            lines.append(try piece("mlp", block.feedForward(input("mlp.in")), probe, "0.mlp.out"))
            lines.append(try piece("post_feedforward_layernorm_1",
                                   try XCTUnwrap(block.postFeedForwardNorm1)(input("post_feedforward_layernorm_1.in")),
                                   probe, "0.post_feedforward_layernorm_1.out"))
            lines.append(try piece("router.norm", NFKReferenceRounding.scaledNorm(input("router.norm.in"), weight: nil,
                                                                                  eps: router.epsilon),
                                   probe, "0.router.norm.out"))
            lines.append(try piece("router.proj", router.proj(input("router.proj.in")), probe, "0.router.proj.out"))
            lines.append(try piece("router (probabilities)", router.probabilities(input("router.in")), probe, "0.router.out"))
            router.forcedChoice = [choice[0]]
            let (weights, indices) = router(try input("router.in"))
            lines.append(try piece("pre_feedforward_layernorm_2",
                                   try XCTUnwrap(block.preFeedForwardNorm2)(input("pre_feedforward_layernorm_2.in")),
                                   probe, "0.pre_feedforward_layernorm_2.out"))
            lines.append(try piece("experts (routed as recorded)",
                                   experts(input("experts.in"), indices: indices, weights: weights), probe, "0.experts.out"))
            lines.append(try piece("post_feedforward_layernorm_2",
                                   try XCTUnwrap(block.postFeedForwardNorm2)(input("post_feedforward_layernorm_2.in")),
                                   probe, "0.post_feedforward_layernorm_2.out"))
            lines.append(try piece("post_feedforward_layernorm", block.postFeedForwardNorm(input("post_feedforward_layernorm.in")),
                                   probe, "0.post_feedforward_layernorm.out"))
            return lines
        }
        let reference = try pieces()
        NFKMLXHalfPrecision.arithmetic = .fast
        let fast = try pieces()
        print("VALIDATION bf16 gemma4_26b_cut6 layer 0 pieces, reference arithmetic:\n" + reference.joined(separator: "\n"))
        print("VALIDATION bf16 gemma4_26b_cut6 layer 0 pieces, MLX's arithmetic:\n" + fast.joined(separator: "\n"))
    }

    private func gemma4Prefix(_ name: String, directory: URL, unified: Bool, bf16: [String: MLXArray],
                              f32: [String: MLXArray], routes: [String: MLXArray]? = nil) throws {
        let configURL = directory.appendingPathComponent("config.json")
        let tokens = MLXArray(try XCTUnwrap(f32["tokens"]).asArray(Int32.self)).reshaped([1, -1])
        let mask = NFKMLXLanguageNet.causalMask(tokens.dim(1), offset: 0)
        // The states, the logits, and one layer run alone on `input` (index, input) → output; the last
        // layer's output takes the final norm, as the reference's last hidden state does.
        func run(_ precision: NFKMLXWeightPrecision)
            throws -> (states: [MLXArray], logits: MLXArray, layer: (Int, MLXArray) -> MLXArray) {
            if unified {
                let net = try loaded(NFKMLXGemmaLanguage.makeUnifiedNet(try NFKMLXGemmaLanguage.unifiedConfiguration(fromHuggingFace: configURL)),
                                     onCPU: false, if: precision == .float32) {
                    try NFKMLXGemmaLanguage.loadUnifiedWeights(into: $0, fromDirectory: directory, precision: precision)
                }
                return (net.hiddenStates(tokens), net(tokens)[0], { index, input in
                    let output = net.layers[index](input, mask: mask, shared: nil).output
                    return index == net.layers.count - 1 ? net.norm(output) : output
                })
            }
            // The 26B-A4B's routed experts are 22 GB at float32, so its float32 run pages them from the
            // release; a paged load computes the resident one's values element for element.
            let net = try loaded(NFKMLXGemmaLanguage.makeNet(try NFKMLXGemmaLanguage.configuration(fromHuggingFace: configURL)),
                                 onCPU: routes != nil, if: precision == .float32) {
                try NFKMLXGemmaLanguage.loadWeights(into: $0, fromDirectory: directory, precision: precision,
                                                    residency: precision == .float32 ? .paged : .resident)
            }
            net.expertStore?.cacheByteBudget = 4 << 30
            if precision != .float32, let routes {
                try route(net, as: routes)
            }
            return (net.hiddenStates(tokens), net(tokens)[0], { index, input in
                let output = net.layers[index](input, perLayerInput: nil, mask: mask, shared: nil).output
                return index == net.layers.count - 1 ? net.norm(output) : output
            })
        }
        // The float32 net is released before the bf16 one loads; the two do not fit together.
        try autoreleasepool {
            let exact = try run(.float32)
            try assertFloat32(name, states: exact.states + [exact.logits], f32: f32)
        }
        Memory.clearCache()

        try autoreleasepool {
            let reduced = try run(.checkpoint)
            eval(reduced.states + [reduced.logits])
            let labelled = reduced.states.enumerated().map { ($0.offset == 0 ? "embedding" : "layer \($0.offset - 1)", $0.element) }
                + [("logits", reduced.logits)]
            let rows = try seams(labelled, bf16: bf16, f32: f32,
                                 keys: reduced.states.indices.map { "hidden.\($0)" } + ["output"])
            report(name, rows)
            var isolated = [(String, MLXArray)]()
            for index in 0 ..< reduced.states.count - 1 {
                let input = try XCTUnwrap(bf16["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
                isolated.append(("layer \(index)", reduced.layer(index, input)))
            }
            eval(isolated.map(\.1))
            let isolatedRows = try seams(isolated, bf16: bf16, f32: f32, keys: isolated.indices.map { "hidden.\($0 + 1)" })
            report("\(name) isolated", isolatedRows)
            // Accumulation order alone moves a mixture layer farther than a dense one: transformers' own
            // layer with its eight experts' matmuls summed in float32 at the same roundings reads 0.27 of
            // the floor at the 26B-A4B's layer 0.
            assertRoundingPlacement(name, endToEnd: rows, isolated: isolatedRows)
        }
        Memory.clearCache()
    }

    /// Fixes every mixture layer of `net` to the experts `routes` recorded (`route.L.index`, `[tokens, k]`),
    /// and reports how many tokens each layer's own router keeps differently and how far its weights for
    /// the recorded experts sit from the reference's (`route.L.weights`).
    private func route(_ net: NFKMLXGemmaNet, as routes: [String: MLXArray]) throws {
        var lines = ["VALIDATION routes: layer  tokens-own-choice-differs  weights-1-cos"]
        for (index, block) in net.layers.enumerated() {
            guard let router = block.router else { continue }
            let recorded = try XCTUnwrap(routes["route.\(index).index"], "no route.\(index).index")
            let k = recorded.dim(-1)
            let flat = recorded.asArray(Int32.self)
            router.forcedChoice = stride(from: 0, to: flat.count, by: k).map { Array(flat[$0 ..< $0 + k]) }
            let input = try XCTUnwrap(routes["hidden.\(index)"]).asType(.bfloat16).expandedDimensions(axis: 0)
            let mask = NFKMLXLanguageNet.causalMask(input.dim(1), offset: 0)
            let attended = input + block.postAttentionNorm(block.attention(block.inputNorm(input), mask: mask, shared: nil).output)
            let flatInput = attended.reshaped([-1, attended.dim(-1)])
            let forced = router(flatInput)
            router.forcedChoice = nil
            let own = router(flatInput).indices.asArray(UInt32.self).map(Int32.init)
            router.forcedChoice = stride(from: 0, to: flat.count, by: k).map { Array(flat[$0 ..< $0 + k]) }
            let differing = stride(from: 0, to: flat.count, by: k).filter {
                Set(own[$0 ..< $0 + k]) != Set(flat[$0 ..< $0 + k])
            }.count
            let weights = distance(floats(forced.weights), floats(try XCTUnwrap(routes["route.\(index).weights"])))
            lines.append(String(format: "  layer %d  %d of %d  %.3e", index, differing, flat.count / k, weights))
        }
        print(lines.joined(separator: "\n"))
    }

    // Gemma 2 9B cut to its first four layers (`IK_VAL_GEMMA2_9B_CUT4`).
    func testGemma2_9BPrefixMatchesTheReferenceAtBothPrecisions() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_GEMMA2_9B_CUT4"], "IK_VAL_GEMMA2_9B_CUT4"))
        try gemma2("gemma2-9b-cut4", directory: directory, bf16: record("gemma2_9b_cut4_bf16.safetensors"),
                   f32: record("gemma2_9b_cut4_f32.safetensors"))
    }

    // Mistral-Small 24B cut to its first four layers (`IK_VAL_MISTRAL_SMALL_CUT4`), the decoder the
    // `.mistralSmall3` preset describes: every state at float32, then bf16.
    func testMistralSmallPrefixMatchesTheReferenceAtBothPrecisions() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_MISTRAL_SMALL_CUT4"], "IK_VAL_MISTRAL_SMALL_CUT4"))
        let bf16 = try record("mistral_small_cut4_bf16.safetensors"), f32 = try record("mistral_small_cut4_f32.safetensors")
        try autoreleasepool {
            let geometry = try NFKMLXLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
            let exact = try loaded(NFKMLXLanguage.makeNet(geometry), onCPU: false) {
                try NFKMLXLanguage.loadWeights(into: $0, fromDirectory: directory, precision: .float32)
            }
            let tokens = MLXArray(try XCTUnwrap(f32["tokens"]).asArray(Int32.self)).reshaped([1, -1])
            try assertFloat32("mistral-small-cut4", states: exact.layerStates(tokens) + [exact(tokens)[0]], f32: f32)
        }
        Memory.clearCache()
        try languageDecoder("mistral-small-cut4", directory: directory, bf16: bf16, f32: f32)
    }

    // MARK: Gemma 3 cut releases

    // Gemma 3 12B and 27B cut to their first four decoder layers (`IK_VAL_GEMMA3_12B_CUT4` /
    // `IK_VAL_GEMMA3_27B_CUT4`, the vision tower kept whole): every hidden state at float32, then bf16.
    func testGemma3LargerSizePrefixesMatchTheReferenceAtBothPrecisions() throws {
        try requireMLXRuntime()
        var measured = 0
        for (name, key) in [("12b-cut4", "IK_VAL_GEMMA3_12B_CUT4"), ("27b-cut4", "IK_VAL_GEMMA3_27B_CUT4")] {
            guard let path = config[key], FileManager.default.fileExists(atPath: path),
                  let f32 = try? record("gemma3_\(name)_f32eager.safetensors"),
                  (try? record("gemma3_\(name)_bf16.safetensors")) != nil else { continue }
            let directory = URL(fileURLWithPath: path)
            // The float32 net is released before the bf16 one loads; the two do not fit together.
            try autoreleasepool {
                let geometry = try NFKMLXGemma3Language.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
                let exact = try loaded(NFKMLXGemma3Language.makeNet(geometry), onCPU: false) {
                    try NFKMLXGemma3Language.loadWeights(into: $0, fromDirectory: directory, precision: .float32)
                }
                let tokens = MLXArray(try XCTUnwrap(f32["tokens"]).asArray(Int32.self)).reshaped([1, -1])
                try assertFloat32("gemma3-\(name)", states: exact.layerStates(tokens) + [exact(tokens)[0]], f32: f32)
            }
            Memory.clearCache()
            try autoreleasepool { try gemma3(name, directoryKey: key) }
            Memory.clearCache()
            measured += 1
        }
        if measured == 0 { throw XCTSkip("set IK_VAL_GEMMA3_12B_CUT4 or IK_VAL_GEMMA3_27B_CUT4") }
    }

    /// Counts the elements where `ours` and the record's `key` differ, as a report line. Where the record holds
    /// a block floor at `<key>.floor`, the piece stays inside it, as `assertWithinBlockFloors` holds a seam.
    private func piece(_ label: String, _ ours: MLXArray, _ arrays: [String: MLXArray], _ key: String,
                       file: StaticString = #filePath, line: UInt = #line) throws -> String {
        eval(ours)
        let mine = floats(ours), theirs = floats(try XCTUnwrap(arrays[key], "no \(key)"))
        guard mine.count == theirs.count else {
            XCTFail("\(label): shape \(ours.shape) vs the record's \(arrays[key]!.shape)", file: file, line: line)
            return "  \(label): shape \(ours.shape) vs \(arrays[key]!.shape)"
        }
        let differing = zip(mine, theirs).filter { $0 != $1 }.count
        let ours = distance(mine, theirs)
        var text = String(format: "  %-22s %6d / %-7d differ  1-cos %.3e", (label as NSString).utf8String!,
                          differing, mine.count, ours)
        let seam = Seam(label: label, differing: 0, ours: ours, floor: 0, exact: 0,
                        blockFloor: arrays["\(key).floor"].map { distance(theirs, floats($0)) })
        if let ratio = seam.blockRatio {
            text += String(format: "  %.4f of its block floor", ratio)
        }
        assertWithinBlockFloors(key, [seam], file: file, line: line)
        return text
    }

    // Each piece of a Gemma 3 block on the reference's own bf16 input, from `run_reference.py
    // hf_layer_probe` (`IK_PROBE_DTYPE=bfloat16 IK_PROBE_LAYERS=…`), so a rounding placed
    // differently shows as a count at the one seam that places it.
    func testGemma3BlockPiecesMatchTheBFloat16Reference() throws {
        for (name, key) in [("270m", "IK_VAL_GEMMA3_270M"), ("1b", "IK_VAL_GEMMA3_1B"), ("4b", "IK_VAL_GEMMA3_4B")] {
            do { try gemma3Pieces(name, directoryKey: key) } catch is XCTSkip { continue }
        }
    }

    private func gemma3Pieces(_ name: String, directoryKey: String) throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config[directoryKey], directoryKey))
        let probe = try record("gemma3_\(name)_bf16_probe.safetensors")
        let geometry = try NFKMLXGemma3Language.configuration(
            fromHuggingFace: directory.appendingPathComponent("config.json"))
        let net = NFKMLXGemma3Language.makeNet(geometry)
        try NFKMLXGemma3Language.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)
        func input(_ key: String) throws -> MLXArray {
            try XCTUnwrap(probe[key], "no \(key)").asType(.bfloat16).expandedDimensions(axis: 0)
        }
        let length = try XCTUnwrap(probe["tokens"]).shape[0]
        let masks = NFKMLXGemma3Masks.make(length: length, offset: 0, window: geometry.slidingWindow,
                                           blockIds: nil, bidirectional: false)
        var lines = [String]()
        for layer in 0 ..< geometry.layerCount where probe["\(layer).block.in"] != nil {
            let block = net.layers[layer], attention = block.attention
            let p = "\(layer)."
            lines.append("layer \(layer) (\(geometry.layerTypes[layer]))")
            lines.append(try piece("input_layernorm", block.inputNorm(input(p + "input_layernorm.in")), probe, p + "input_layernorm.out"))
            for (name, projection) in [("q_proj", attention.queryProjection), ("k_proj", attention.keyProjection),
                                       ("v_proj", attention.valueProjection)] {
                lines.append(try piece(name, projection(input(p + "self_attn.\(name).in")), probe, p + "self_attn.\(name).out"))
            }
            lines.append(try piece("q_norm", attention.queryNorm(input(p + "self_attn.q_norm.in")), probe, p + "self_attn.q_norm.out"))
            lines.append(try piece("k_norm", attention.keyNorm(input(p + "self_attn.k_norm.in")), probe, p + "self_attn.k_norm.out"))
            // transformers transposes the heads ahead of the norms, so these are `[heads, length, width]`.
            let q = try input(p + "self_attn.q_norm.out")
            let k = try input(p + "self_attn.k_norm.out")
            lines.append(try piece("rotary q", NFKReferenceRounding.rotary(q, dimensions: attention.headDimensions,
                base: attention.ropeBase, scale: attention.ropeScale, offset: 0)[0], probe, p + "attn.q"))
            lines.append(try piece("rotary k", NFKReferenceRounding.rotary(k, dimensions: attention.headDimensions,
                base: attention.ropeBase, scale: attention.ropeScale, offset: 0)[0], probe, p + "attn.k"))
            let mask = (geometry.layerTypes[layer] == .full ? masks.full : masks.sliding)?.asType(.bfloat16)
            let attended = NFKReferenceRounding.attention(queries: try input(p + "attn.q"), keys: try input(p + "attn.k"),
                                                          values: try input(p + "attn.v"), scale: attention.scale,
                                                          mask: mask, softcap: attention.softcap)
            lines.append(try piece("attention", attended[0].transposed(1, 0, 2), probe, p + "attn.out"))
            lines.append(try piece("o_proj", attention.outputProjection(input(p + "self_attn.o_proj.in")), probe, p + "self_attn.o_proj.out"))
            lines.append(try piece("post_attention_norm", block.postAttentionNorm(input(p + "post_attention_layernorm.in")),
                                   probe, p + "post_attention_layernorm.out"))
            lines.append(try piece("pre_feedforward_norm", block.preFeedForwardNorm(input(p + "pre_feedforward_layernorm.in")),
                                   probe, p + "pre_feedforward_layernorm.out"))
            lines.append(try piece("gate_proj", block.feedForward.gate(input(p + "mlp.gate_proj.in")), probe, p + "mlp.gate_proj.out"))
            lines.append(try piece("up_proj", block.feedForward.up(input(p + "mlp.up_proj.in")), probe, p + "mlp.up_proj.out"))
            lines.append(try piece("gelu", NFKReferenceRounding.geluTanh(input(p + "mlp.act_fn.in")), probe, p + "mlp.act_fn.out"))
            lines.append(try piece("down_proj", block.feedForward.down(input(p + "mlp.down_proj.in")), probe, p + "mlp.down_proj.out"))
            lines.append(try piece("mlp", block.feedForward(input(p + "mlp.in")), probe, p + "mlp.out"))
            lines.append(try piece("post_feedforward_norm", block.postFeedForwardNorm(input(p + "post_feedforward_layernorm.in")),
                                   probe, p + "post_feedforward_layernorm.out"))
        }
        print("VALIDATION bf16 gemma3-\(name) pieces:\n" + lines.joined(separator: "\n"))
    }

    func testGemma3_270MInBFloat16MatchesTheBFloat16Reference() throws {
        try gemma3("270m", directoryKey: "IK_VAL_GEMMA3_270M")
    }

    func testGemma3_1BInBFloat16MatchesTheBFloat16Reference() throws {
        try gemma3("1b", directoryKey: "IK_VAL_GEMMA3_1B")
    }

    func testGemma3_4BInBFloat16MatchesTheBFloat16Reference() throws {
        try gemma3("4b", directoryKey: "IK_VAL_GEMMA3_4B")
    }
}
