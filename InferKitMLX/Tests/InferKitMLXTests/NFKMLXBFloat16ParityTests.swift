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

import XCTest
import Foundation
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXBFloat16ParityTests: XCTestCase {

    override func tearDown() {
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
                        exact: distance(ours, exact))
        }
    }

    private func report(_ name: String, _ rows: [Seam]) {
        var lines = ["VALIDATION bf16 \(name):  seam  differing  ours-vs-ref-bf16  floor(ref bf16 vs f32)  ours-vs-f32"]
        for row in rows {
            lines.append(String(format: "  %-16s %8.4f%%  %.3e  %.3e  %.3e%@", (row.label as NSString).utf8String!,
                                row.differing * 100, row.ours, row.floor, row.exact,
                                row.exact > 2 * row.floor ? "  <- farther from f32 than twice the floor" : ""))
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
    /// `isolatedBar` is the fraction of the floor a layer run alone on the reference's input may reach.
    private func assertRoundingPlacement(_ name: String, endToEnd: [Seam], isolated: [Seam], isolatedBar: Double = 0.25,
                                         file: StaticString = #filePath, line: UInt = #line) {
        let worst = isolated.max { $0.ours / max($0.floor, 1e-30) < $1.ours / max($1.floor, 1e-30) }!
        let last = endToEnd[endToEnd.count - 1]
        print(String(format: "VALIDATION bf16 %@ summary: worst isolated %@ at %.4f of the floor; %@ ours-vs-f32 %.3e, floor %.3e",
                     name, worst.label, worst.ours / max(worst.floor, 1e-30), last.label, last.exact, last.floor))
        for row in isolated {
            XCTAssertLessThan(row.ours, isolatedBar * row.floor,
                              "\(name) \(row.label) on the reference's input: a rounding placed unlike the reference's",
                              file: file, line: line)
        }
        for row in endToEnd.suffix(2) {
            XCTAssertLessThanOrEqual(row.exact, 2 * row.floor,
                                     "\(name) \(row.label): farther from float32 than twice the reference's bf16",
                                     file: file, line: line)
        }
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

    // MARK: Gemma 3n

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
                let exact = try loadedOnCPU(NFKMLXLanguage.makeNet(geometry)) {
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

    /// The net `load` builds, with every file read and float32 conversion evaluated on the CPU one array at
    /// a time. A cut's conversion evaluated on the GPU in the first forward's command buffer ends the
    /// process at the GPU watchdog when its pages have to come back from swap; the CPU has no watchdog.
    private func loadedOnCPU<Net: Module>(_ load: () throws -> Net) throws -> Net {
        let net = try Device.withDefaultDevice(.cpu) { () throws -> Net in
            let net = try load()
            for (_, weight) in net.parameters().flattened() { eval(weight) }
            return net
        }
        Memory.clearCache()
        return net
    }

    /// `net` after `fill` loads its weights, on the CPU as ``loadedOnCPU(_:)`` does when `condition` holds.
    private func loadedOnCPU<Net: Module>(_ net: Net, if condition: Bool = true,
                                          _ fill: (Net) throws -> Void) throws -> Net {
        guard condition else {
            try fill(net)
            return net
        }
        return try loadedOnCPU { () throws -> Net in
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
        assertRoundingPlacement(name, endToEnd: rows, isolated: isolatedRows, isolatedBar: routes == nil ? 0.25 : 0.5)
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
                let exact = try loadedOnCPU(NFKMLXLanguage.makeNet(geometry)) {
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
            let exact = try loadedOnCPU(NFKMLXGraniteHybrid.makeNet(configuration)) {
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
        assertRoundingPlacement("granite-tiny-cut6", endToEnd: rows, isolated: isolatedRows, isolatedBar: 0.5)
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
            let exact = try loadedOnCPU(NFKMLXQwen4Exp.makeNet(geometry)) {
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
        assertRoundingPlacement("qwen4-exp-cut2", endToEnd: rows, isolated: isolatedRows, isolatedBar: 0.5)
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
    // float32, and gated full attention, at bf16 against bf16 (`hf_layer_probe`, layers 0 and 3).
    func testQwen35_4BInBFloat16MatchesTheBFloat16Reference() throws {
        try requireMLXRuntime()
        let directory = URL(fileURLWithPath: try existing(config["IK_VAL_QWEN3_5"], "IK_VAL_QWEN3_5"))
        let bf16 = try record("qwen35_4b_bf16.safetensors"), f32 = try record("qwen35_4b_f32.safetensors")
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
        let probe = try record("qwen35_4b_bf16.safetensors")
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
        lines.append(try piece("unit norm", NFKHybridLinearAttention.unitNorm(try recorded("fn.l2norm.0.arg0")),
                               probe, "fn.l2norm.0.out0"))
        let read = delta.recurrence(
            queries: NFKHybridLinearAttention.unitNorm(try recorded("fn.torch_chunk_gated_delta_rule.0.arg0")).asType(.float32),
            keys: NFKHybridLinearAttention.unitNorm(try recorded("fn.torch_chunk_gated_delta_rule.0.arg1")).asType(.float32),
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
    // bf16-rounded weights (`IK_TINY_DTYPE=bfloat16-weights`) as the floor.
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
        XCTAssertLessThanOrEqual(rows.last!.exact, 2 * rows.last!.floor, "the logits sit within twice the floor")
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
        let exact = try loadedOnCPU { try NFKMLXGemma2Net.load(directoryURL: directory, precision: .float32) }
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
        try gemma4Prefix("gemma4_26b_cut6", directory: directory, unified: false,
                         bf16: record("gemma4_26b_cut6_bf16.safetensors"),
                         f32: record("gemma4_26b_cut6_f32.safetensors"), routes: routes)
    }

    /// A Gemma 4 cut at float32 against the float32 record, then at bf16 end to end and layer by layer
    /// against the bf16 record. `routes` (`route.L.index` per mixture layer) fixes the bf16 run's experts.
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
                let net = try loadedOnCPU(NFKMLXGemmaLanguage.makeUnifiedNet(try NFKMLXGemmaLanguage.unifiedConfiguration(fromHuggingFace: configURL)),
                                          if: precision == .float32) {
                    try NFKMLXGemmaLanguage.loadUnifiedWeights(into: $0, fromDirectory: directory, precision: precision)
                }
                return (net.hiddenStates(tokens), net(tokens)[0], { index, input in
                    let output = net.layers[index](input, mask: mask, shared: nil).output
                    return index == net.layers.count - 1 ? net.norm(output) : output
                })
            }
            // The 26B-A4B's routed experts are 22 GB at float32, so its float32 run pages them from the
            // release; a paged load computes the resident one's values element for element.
            let net = try loadedOnCPU(NFKMLXGemmaLanguage.makeNet(try NFKMLXGemmaLanguage.configuration(fromHuggingFace: configURL)),
                                      if: precision == .float32) {
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
            assertRoundingPlacement(name, endToEnd: rows, isolated: isolatedRows, isolatedBar: routes == nil ? 0.25 : 0.5)
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
            let exact = try loadedOnCPU(NFKMLXLanguage.makeNet(geometry)) {
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
                let exact = try loadedOnCPU(NFKMLXGemma3Language.makeNet(geometry)) {
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

    /// Counts the elements where `ours` and the record's `key` differ, as a report line.
    private func piece(_ label: String, _ ours: MLXArray, _ arrays: [String: MLXArray], _ key: String) throws -> String {
        eval(ours)
        let mine = floats(ours), theirs = floats(try XCTUnwrap(arrays[key], "no \(key)"))
        guard mine.count == theirs.count else { return "  \(label): shape \(ours.shape) vs \(arrays[key]!.shape)" }
        let differing = zip(mine, theirs).filter { $0 != $1 }.count
        return String(format: "  %-22s %6d / %-7d differ  1-cos %.3e", (label as NSString).utf8String!,
                      differing, mine.count, distance(mine, theirs))
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
