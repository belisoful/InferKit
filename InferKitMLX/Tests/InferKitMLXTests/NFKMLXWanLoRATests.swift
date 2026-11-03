//
//  NFKMLXWanLoRATests.swift
//  InferKitMLXTests
//
//  Released Wan LoRAs: the layer names each layout spells, the alpha folding, the refusals, and, where
//  the validation store holds the record, every merged weight against diffusers' own fuse
//  (`run_reference.py wan_lora`). Runs where MLX has a Metal library (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXWanLoRATests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    // MARK: Names and scales

    func testOriginalNamesMapOntoTheTextToVideoModules() {
        let cases: [(String, String?)] = [
            ("blocks.3.self_attn.q", "blocks.3.attn1.to_q"), ("blocks.3.self_attn.o", "blocks.3.attn1.to_out.0"),
            ("blocks.0.cross_attn.k_img", "blocks.0.attn2.add_k_proj"), ("blocks.7.ffn.0", "blocks.7.ffn.net.0.proj"),
            ("blocks.7.ffn.2", "blocks.7.ffn.net.2"), ("time_projection.1", "condition_embedder.time_proj"),
            ("head.head", "proj_out"), ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
            ("time_embedding.0", "condition_embedder.time_embedder.linear_1"), ("blocks.3.norm3", nil),
        ]
        for (original, module) in cases {
            XCTAssertEqual(NFKMLXWanLoRA.textToVideoPath(original: original), module, original)
        }
        XCTAssertEqual(NFKMLXWanLoRA.animatePath(original: "blocks.2.block.cross_attn.v_img"), "blocks.2.cross_attn.add_v_proj",
                       "Animate keeps the original names but for the attention projections")
        XCTAssertEqual(NFKMLXWanLoRA.musubiOriginal("blocks_12_cross_attn_k_img"), "blocks.12.cross_attn.k_img")
        XCTAssertEqual(NFKMLXWanLoRA.musubiOriginal("blocks_0_ffn_2"), "blocks.0.ffn.2")
        XCTAssertNil(NFKMLXWanLoRA.musubiOriginal("blocks_0_norm3"))
    }

    func testTheAlphaFoldSplitsAPowerOfTwoOntoTheUpWeight() {
        XCTAssertTrue(NFKMLXWanLoRA.foldedScales(2) == (2, 1), "a factor above one stays on the down weight")
        XCTAssertTrue(NFKMLXWanLoRA.foldedScales(0.25) == (0.5, 0.5))
        XCTAssertTrue(NFKMLXWanLoRA.foldedScales(0.0625) == (0.25, 0.25))
        let (down, up) = NFKMLXWanLoRA.foldedScales(0.3)
        XCTAssertEqual(down * up, 0.3, accuracy: 1e-15, "the split keeps the product")
    }

    func testADoRAAndAnUnknownKeyAreRefused() throws {
        try requireMLXRuntime()
        let dora: [String: MLXArray] = ["transformer.proj_out.lora_A.weight": MLXArray.zeros([2, 4]),
                                        "transformer.proj_out.lora_B.weight": MLXArray.zeros([4, 2]),
                                        "transformer.proj_out.lora_magnitude_vector.dora_scale": MLXArray.zeros([4])]
        XCTAssertThrowsError(try NFKMLXWanLoRA.deltas(arrays: dora, metadata: [:],
                                                      naming: NFKMLXWanLoRA.textToVideoPath(original:), scale: 1))
        XCTAssertThrowsError(try NFKMLXWanLoRA.deltas(arrays: ["transformer.proj_out.weight": MLXArray.zeros([4, 4])],
                                                      metadata: [:], naming: NFKMLXWanLoRA.textToVideoPath(original:),
                                                      scale: 1))
        let missingAlpha: [String: MLXArray] = ["lora_unet_blocks_0_self_attn_q.lora_down.weight": MLXArray.zeros([2, 4]),
                                                "lora_unet_blocks_0_self_attn_q.lora_up.weight": MLXArray.zeros([4, 2])]
        XCTAssertThrowsError(try NFKMLXWanLoRA.deltas(arrays: missingAlpha, metadata: [:],
                                                      naming: NFKMLXWanLoRA.textToVideoPath(original:), scale: 1),
                             "a musubi layer carries its alpha")
        let headAlpha: [String: MLXArray] = ["diffusion_model.head.head.lora_down.weight": MLXArray.zeros([2, 4]),
                                             "diffusion_model.head.head.lora_up.weight": MLXArray.zeros([4, 2]),
                                             "diffusion_model.head.head.alpha": MLXArray(Float(2))]
        XCTAssertThrowsError(try NFKMLXWanLoRA.deltas(arrays: headAlpha, metadata: [:],
                                                      naming: NFKMLXWanLoRA.textToVideoPath(original:), scale: 1),
                             "the reference reads an alpha on the blocks' layers alone")
    }

    func testAnAdapterForAnotherGeometryChangesNoWeight() throws {
        try requireMLXRuntime()
        let net = NFKMLXWanTransformerNet(.tiny)
        let before = net.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }
        eval(before.map(\.1))
        // The first target fits; the second is the wrong width, so nothing may be merged.
        let deltas: [String: NFKMLXWanLoRA.Delta] = [
            "proj_out": .init(weight: MLXArray.ones(net.projOut.weight.shape), bias: nil),
            "blocks.0.attn1.to_q": .init(weight: MLXArray.ones([3, 3]), bias: nil),
        ]
        XCTAssertThrowsError(try NFKMLXWanLoRA.apply(deltas, to: net))
        for (key, value) in net.parameters().flattened() {
            let original = try XCTUnwrap(before.first { $0.0 == key }?.1)
            XCTAssertEqual(abs(value.asType(.float32) - original).max().item(Float.self), 0, key)
        }
    }

    // MARK: Against diffusers' fuse

    private let textToVideo = NFKMLXWanConfiguration(inChannels: 4, heads: 2, headDim: 16, layers: 2, ffnDim: 48,
                                                     textDim: 32, patchSize: [1, 2, 2])

    /// The cases whose adapter files the oracle wrote at bfloat16; the record holds every tensor at float32.
    private let bfloat16Files: Set<String> = ["original_alpha", "original_head_diff", "stacked#1"]

    /// Writes the record's adapter `file` (a case, or a case's `#index`) back as a safetensors file.
    private func adapterFile(_ arrays: [String: MLXArray], _ file: String) throws -> URL {
        let prefix = "lora::\(file)::"
        var tensors = [String: MLXArray]()
        for (key, value) in arrays where key.hasPrefix(prefix) {
            tensors[String(key.dropFirst(prefix.count))] = bfloat16Files.contains(file) ? value.asType(.bfloat16) : value
        }
        XCTAssertFalse(tensors.isEmpty, file)
        var metadata = [String: String]()
        if let bytes = arrays["meta::\(file)"] {
            metadata["lora_adapter_metadata"] = String(decoding: bytes.asArray(UInt8.self), as: UTF8.self)
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wan-lora-\(UUID().uuidString).safetensors")
        try MLX.save(arrays: tensors, metadata: metadata, url: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    /// Every parameter of `net` against the fused model: the changed ones to the fuse, the rest unchanged.
    private func assertMatchesTheFuse(_ net: Module, base: [String: MLXArray], arrays: [String: MLXArray], case name: String,
                                      layout: (String, MLXArray) -> MLXArray) -> Float {
        let prefix = "fused::\(name)::"
        var worst: Float = 0, changed = 0
        for (key, value) in net.parameters().flattened() {
            let expected = arrays[prefix + key].map { layout(key, $0) } ?? base[key]!
            if arrays[prefix + key] != nil { changed += 1 }
            let gap = abs(value.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
            worst = max(worst, gap)
            XCTAssertLessThan(gap, 1e-6, "\(name): \(key)")
        }
        XCTAssertEqual(changed, arrays.keys.filter { $0.hasPrefix(prefix) }.count, "\(name): every fused parameter is one of ours")
        return worst
    }

    func testEveryLayoutMergesAsDiffusersFusesIt() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN_LORA"] else {
            throw XCTSkip("set IK_PARITY_WAN_LORA (run_reference.py wan_lora)")
        }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        func weights(_ prefix: String) -> [String: MLXArray] {
            Dictionary(uniqueKeysWithValues: arrays.compactMap { key, value in
                key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil
            })
        }
        let convLayout = { (_: String, value: MLXArray) in value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value }
        let textToVideoBase = weights("w::t2v::").mapValues { $0.ndim == 5 ? $0.transposed(0, 2, 3, 4, 1) : $0 }
        let animateBase = weights("w::animate::").mapValues { $0.ndim == 5 ? $0.transposed(0, 2, 3, 4, 1) : $0 }
        var report = [String]()

        for name in ["original_alpha", "original_head_diff", "musubi", "peft", "peft_metadata", "stacked"] {
            let net = NFKMLXWanTransformerNet(textToVideo)
            try NFKMLXWeights.apply(textToVideoBase.map { ($0.key, $0.value) }, to: net, verifyShapes: true)
            let scale = try XCTUnwrap(arrays["scale::\(name)"]).item(Float.self)
            let files = name == "stacked" ? ["stacked#0", "stacked#1"] : [name]
            for file in files {
                // diffusers' fuse_lora(lora_scale:) scales the adapter loaded last alone; an earlier one fuses at 1.
                let fileScale = file == "stacked#0" ? 1 : scale
                let merged = try NFKMLXWanLoRA.merge(adapterURL: try adapterFile(arrays, file), into: net, scale: fileScale)
                if file == "original_alpha" || file == "stacked#1" {
                    XCTAssertEqual(merged.droppedKeys.count, 2, "the normalization's and the block's .diff are dropped")
                }
                if file == "original_head_diff" {
                    XCTAssertEqual(merged.droppedKeys, ["diffusion_model.time_projection.1.diff_b"],
                                   "a time-projection bias delta without its weights is dropped")
                }
            }
            report.append("\(name) \(assertMatchesTheFuse(net, base: textToVideoBase, arrays: arrays, case: name, layout: convLayout))")
        }

        for (name, file) in [("animate", "animate"), ("animate", "animate_original")] {
            let net = NFKMLXWanAnimateNet(.tiny)
            try NFKMLXWeights.apply(animateBase.map { ($0.key, $0.value) }, to: net, verifyShapes: true)
            let scale = try XCTUnwrap(arrays["scale::animate"]).item(Float.self)
            try NFKMLXWanLoRA.merge(adapterURL: try adapterFile(arrays, file), into: net, scale: scale)
            report.append("\(file) \(assertMatchesTheFuse(net, base: animateBase, arrays: arrays, case: name, layout: convLayout))")
        }
        XCTAssertEqual(try XCTUnwrap(arrays["animate_original_refused"]).item(Int32.self), 2,
                       "diffusers converts the original layout to the text-to-video names, which Animate does not have, "
                       + "so the adapter's PEFT twin is the reference")
        print("VALIDATION PARITY wan-lora (largest weight gap per case): " + report.joined(separator: ", "))
    }
}
