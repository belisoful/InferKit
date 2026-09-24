//
//  NFKMLXReleasedSizesTests.swift
//  InferKitMLXTests
//
//  Every released SIZE of a model family, not only the one it was first ported at. A family whose
//  geometry is a hardcoded configuration (Whisper, SAM, SAM 2, Depth Anything 3, NAFNet, SwinIR, RVM,
//  RT-DETR, RF-DETR, SNAC, CLIP, SigLIP 2) needs a preset per release, and this file is where each
//  preset is held to the release: at reference parity where the checkpoint is a modest download, and
//  otherwise structurally — every parameter's shape against the release's own safetensors headers
//  (`Tools/validation-assets/shapes.py`), which is what a size this machine cannot run still admits.
//
//  The shape checks read `IK_SHAPES_ROOT`, a directory of `<release>/{config.json,shapes.json}`; the
//  parity checks read the usual `IK_VAL_*` / `IK_PARITY_*` keys. Both come through
//  `NFKMLXValidationConfig.environment`, so `~/.inferkit-validation.json` provisions them.
//

import XCTest
import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXReleasedSizesTests: XCTestCase {

    override func tearDown() {
        // Clearing the cache reaches MLX's runtime, which needs a Metal library it can find.
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

    private func path(_ key: String) throws -> String {
        guard let path = config[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key)")
        }
        return path
    }

    private func weights(_ key: String) throws -> URL { URL(fileURLWithPath: try path(key)) }

    private func record(_ key: String) throws -> [String: MLXArray] {
        try loadArrays(url: URL(fileURLWithPath: try path(key)))
    }

    private func cosine(_ a: [Double], _ b: [Double]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(a.count, b.count) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        eval(a)
        return cosine(a.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init),
                      b.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init))
    }

    // MARK: - Structure against the released safetensors headers

    /// A release's `config.json` URL and its `name -> shape` inventory from `IK_SHAPES_ROOT/<name>`.
    private func shapes(_ name: String) throws -> (config: URL, shapes: [String: [Int]]) {
        let root = try path("IK_SHAPES_ROOT")
        let directory = URL(fileURLWithPath: root).appendingPathComponent(name)
        let shapesURL = directory.appendingPathComponent("shapes.json")
        guard let data = FileManager.default.contents(atPath: shapesURL.path),
              let released = try JSONSerialization.jsonObject(with: data) as? [String: [Int]] else {
            throw XCTSkip("no shapes for \(name) under \(root) (Tools/validation-assets/shapes.py)")
        }
        return (directory.appendingPathComponent("config.json"), released)
    }

    /// Holds a module's parameters, in RELEASE names and layouts, to a release inventory: nothing the
    /// module builds may be absent or differently shaped, and nothing the release ships may go unread
    /// unless `named` says the port drops it on purpose.
    private func assertStructure(_ label: String, built: [(String, [Int])], released: [String: [Int]],
                                 named: (String) -> Bool = { _ in false }) {
        var consumed = Set<String>()
        var missing = [String](), mismatched = [String]()
        for (name, shape) in built {
            guard let releasedShape = released[name] else { missing.append(name); continue }
            consumed.insert(name)
            if releasedShape != shape {
                mismatched.append("\(name): built \(shape), released \(releasedShape)")
            }
        }
        let unaccounted = released.keys.filter { !consumed.contains($0) && !named($0) }.sorted()
        let dropped = released.keys.filter { !consumed.contains($0) && named($0) }.count
        print("VALIDATION structure \(label): \(consumed.count) released tensors consumed, "
              + "\(missing.count) missing, \(mismatched.count) mismatched, \(dropped) named as dropped, "
              + "\(unaccounted.count) unaccounted")
        XCTAssertTrue(mismatched.isEmpty, "\(label) shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(missing.isEmpty, "\(label) absent from the release:\n" + missing.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(unaccounted.isEmpty, "\(label) released tensors nothing here reads:\n"
                      + unaccounted.prefix(8).joined(separator: "\n"))
    }

    /// A module's parameters as `(name, shape)`, through `rename` (nil drops one) and `reshape`
    /// (the release's layout for a built shape).
    private func inventory(_ module: Module, rename: (String) -> String? = { $0 },
                           reshape: (String, [Int]) -> [Int] = { $1 }) -> [(String, [Int])] {
        module.parameters().flattened().compactMap { name, value in
            rename(name).map { ($0, reshape($0, value.shape)) }
        }
    }

    /// MLX's channels-last convolution weight `[out, kH, kW, in]` back to PyTorch's `[out, in, kH, kW]`.
    private func convLayout(_ shape: [Int]) -> [Int] {
        shape.count == 4 ? [shape[0], shape[3], shape[1], shape[2]] : shape
    }

    // The two dense Qwen3 sizes above what this machine runs at float32. The config reader is the
    // generic one every Qwen3 release loads through, so the check is that a 14B and a 32B config
    // build the module the release was saved from — head count 40 and 64, untied heads.
    func testQwen3LargerDenseSizesMatchTheReleasedShapes() throws {
        try requireMLXRuntime()
        for (name, layers, heads) in [("qwen3-14b", 40, 40), ("qwen3-32b", 64, 64)] {
            let release = try shapes(name)
            let configuration = try NFKMLXLanguage.configuration(fromHuggingFace: release.config)
            XCTAssertEqual(configuration.layerCount, layers)
            XCTAssertEqual(configuration.headCount, heads)
            XCTAssertFalse(configuration.tiesWordEmbeddings)
            let net = NFKMLXLanguage.makeNet(configuration)
            assertStructure(name, built: inventory(net), released: release.shapes)
        }
    }

    // Mistral-Small 3 is the text stack FLUX.2 [dev] conditions on, inside a multimodal release whose
    // decoder sits under `text_config` and whose tensors carry a `language_model.` prefix. The vision
    // tower and its connector are named as dropped: this reads the decoder only.
    func testMistralSmall3MatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("mistral-small-3.2-24b")
        let configuration = try NFKMLXLanguage.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.hiddenSize, NFKMLXLanguageConfiguration.mistralSmall3.hiddenSize)
        XCTAssertEqual(configuration.layerCount, NFKMLXLanguageConfiguration.mistralSmall3.layerCount)
        XCTAssertEqual(configuration.headCount, NFKMLXLanguageConfiguration.mistralSmall3.headCount)
        // 5120 over 32 heads divides to 160; the release states 128, so a reader that infers the head
        // width from the residual builds projections the checkpoint does not fit.
        XCTAssertEqual(configuration.headDimensions, 128)
        XCTAssertEqual(configuration.keyValueHeadCount, 8)
        XCTAssertFalse(configuration.tiesWordEmbeddings)
        XCTAssertFalse(configuration.normalizesQueryAndKey)
        XCTAssertFalse(configuration.attentionBias)
        let net = NFKMLXLanguage.makeNet(configuration)
        assertStructure("mistral-small-3.2-24b",
                        built: inventory(net, rename: { "language_model." + $0 }),
                        released: release.shapes,
                        named: { $0.hasPrefix("vision_tower.")
                                 || $0.hasPrefix("multi_modal_projector.") })
    }

    // The two Qwen3-Embedding sizes above the ported 0.6B. The release is the base model — no `model.`
    // prefix and no head — so the module's names shed the prefix and the head; the 8B says it is
    // untied, which for an embedder is moot, since the head is never built or read.
    func testQwen3EmbeddingLargerSizesMatchTheReleasedShapes() throws {
        try requireMLXRuntime()
        for name in ["qwen3-embedding-4b", "qwen3-embedding-8b"] {
            let release = try shapes(name)
            var configuration = try NFKMLXLanguage.configuration(fromHuggingFace: release.config)
            configuration.tiesWordEmbeddings = true
            let net = NFKMLXLanguage.makeNet(configuration)
            let prefixed = release.shapes.keys.contains { $0.hasPrefix("model.") }
            let built = inventory(net) { key in prefixed ? key : String(key.dropFirst("model.".count)) }
            assertStructure(name, built: built, released: release.shapes) { $0 == "lm_head.weight" }
        }
    }

    // Gemma 3's 12B and 27B: the same architecture the 4B is at parity on, read by the same
    // configuration reader, with the so400m vision tower and a projector at the wider text width.
    func testGemma3LargerSizesMatchTheReleasedShapes() throws {
        try requireMLXRuntime()
        for (name, layers) in [("gemma-3-12b", 48), ("gemma-3-27b", 62)] {
            let release = try shapes(name)
            let configuration = try NFKMLXGemma3Language.configuration(fromHuggingFace: release.config)
            XCTAssertEqual(configuration.layerCount, layers)
            let decoder = NFKMLXGemma3Net(configuration)
            let vision = NFKMLXGemma3VisionNet(.gemma3)
            let projector = NFKMLXGemma3MultimodalProjector(visionHidden: 1152, textHidden: configuration.hiddenSize,
                                                             patchesPerSide: 896 / 14)
            // The inventory is keyed by module names; the release's names come through the loaders'
            // own mappers, so the comparison runs in module space.
            var released = [String: [Int]]()
            for (key, shape) in release.shapes {
                if key.hasSuffix("lm_head.weight") { continue }
                if let decoderKey = NFKMLXGemma3Language.decoderName(of: key) {
                    released["decoder." + decoderKey] = shape
                } else if let visionKey = NFKMLXGemma3.visionName(of: key) {
                    released["vision." + visionKey] = shape
                } else if let projectorKey = NFKMLXGemma3.projectorName(of: key) {
                    released["projector." + projectorKey] = shape
                } else {
                    released[key] = shape
                }
            }
            let built = inventory(decoder) { "decoder." + $0 }
                + inventory(vision, rename: { "vision." + $0 }, reshape: { _, shape in self.convLayout(shape) })
                + inventory(projector) { "projector." + $0 }
            assertStructure(name, built: built, released: released)
        }
    }

    // Gemma 4's three large sizes, none of which fits this machine. They are the releases that set
    // `attention_k_eq_v`, so they are the only ones whose full-attention layers drop `v_proj`, run
    // `num_global_key_value_heads`, and carry no per-layer input embedding at all. The E-series that
    // the numeric tests measure does none of those things, which is why this check exists: the
    // decoder was previously built for the E-series shape at every size, and nothing held it against
    // a released inventory.
    func testGemma4LargerSizesMatchTheReleasedShapes() throws {
        try requireMLXRuntime()
        // The 12B is `gemma4_unified_text`, a different stack that the dense reader deliberately
        // refuses, so it is read and built through the unified pair.
        for (name, layers, unified) in [("gemma-4-31b", 60, false), ("gemma-4-26b-a4b", 30, false),
                                        ("gemma-4-12b", 48, true)] {
            let release = try shapes(name)
            let configuration = unified
                ? try NFKMLXGemmaLanguage.unifiedConfiguration(fromHuggingFace: release.config)
                : try NFKMLXGemmaLanguage.configuration(fromHuggingFace: release.config)
            XCTAssertEqual(configuration.layerCount, layers, "\(name) layer count")
            XCTAssertTrue(configuration.attentionKeyEqualsValue, "\(name) shares keys and values")
            XCTAssertFalse(configuration.hasPerLayerInput, "\(name) carries no per-layer input")

            let prefix = "model.language_model."
            var released = [String: [Int]]()
            for (key, shape) in release.shapes where key.hasPrefix(prefix) {
                released[String(key.dropFirst(prefix.count))] = shape
            }
            let built = unified
                ? inventory(NFKMLXGemmaLanguage.makeUnifiedNet(configuration))
                : inventory(NFKMLXGemmaLanguage.makeNet(configuration))
            assertStructure(name, built: built, released: released) { $0 == "lm_head.weight" }
        }
    }

    // Gemma 2's 9B and 27B, the sizes SANA's text encoder was not ported at.
    func testGemma2LargerSizesMatchTheReleasedShapes() throws {
        try requireMLXRuntime()
        for (name, configuration) in [("gemma-2-9b", NFKMLXGemma2Configuration.gemma2_9B),
                                      ("gemma-2-27b", NFKMLXGemma2Configuration.gemma2_27B)] {
            let release = try shapes(name)
            let net = NFKMLXGemma2Net(configuration)
            let built = inventory(net) { "model." + $0 }
            assertStructure(name, built: built, released: release.shapes) { $0 == "lm_head.weight" }
        }
    }

    // Gemma 3n's E4B decoder, read by the E2B's configuration reader: 35 layers, fifteen of them
    // sharing keys and values, at the same widths. The vision and audio towers and their embedders are
    // the E2B's and are named rather than read here; the decoder's own set is what the check covers.
    func testGemma3nE4BDecoderMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("gemma-3n-e4b")
        let configuration = try NFKMLXGemma3nLanguage.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.layerCount, 35)
        XCTAssertEqual((0 ..< 35).filter { configuration.sharesKeyValues(layer: $0) }.count, 15,
                       "E4B's last fifteen layers share keys and values")
        let net = NFKMLXGemma3nNet(configuration)
        // The release ships key/value projections for its fifteen sharing layers too; the loader drops
        // them (a sharing layer reads its donor's), which is what `decoderName` returning nil says.
        var released = [String: [Int]]()
        var dropped = Set<String>()
        for (key, shape) in release.shapes {
            if let name = NFKMLXGemma3nLanguage.decoderName(of: key, configuration: configuration) {
                released[name] = shape
            } else {
                released[key] = shape
                if key.hasPrefix("model.language_model.") { dropped.insert(key) }
            }
        }
        assertStructure("gemma-3n-e4b decoder", built: inventory(net), released: released) { key in
            dropped.contains(key)
                || key.hasPrefix("model.vision_tower.") || key.hasPrefix("model.audio_tower.")
                || key.hasPrefix("model.embed_vision.") || key.hasPrefix("model.embed_audio.")
                || key.hasSuffix("lm_head.weight")
        }
    }

    // The E4B decoder on its released weights, at the precision it ships in: 16 GB of bf16 doubles
    // past this machine's RAM at float32, so both sides run bf16 (`IK_GEMMA_DTYPE=bfloat16` for the
    // oracle, `.checkpoint` here), the Gemma 4 E4B treatment. A strict load is itself a check on the
    // fifteen sharing layers, since a wrong donor set fails loudly.
    //
    // The E2B at bf16 on both sides is the control: it is exact at float32, so whatever it loses here
    // is the rounding floor, and the E4B is held to that floor rather than to float32 exactness.
    func testGemma3nE4BMatchesTheReferenceLogits() throws {
        try requireMLXRuntime()
        for (name, valKey, parityKey, layers) in [
            ("gemma3n-e2b-bf16", "IK_VAL_GEMMA3N_E2B", "IK_PARITY_GEMMA3N_E2B_BF16", 30),
            ("gemma3n-e4b", "IK_VAL_GEMMA3N_E4B", "IK_PARITY_GEMMA3N_E4B", 35),
        ] {
            guard let release = config[valKey], let record = config[parityKey] else {
                print("SKIP \(name): set \(valKey) and \(parityKey) (IK_GEMMA_DTYPE=bfloat16 run_reference.py gemma3n)")
                continue
            }
            let arrays = try loadArrays(url: URL(fileURLWithPath: record))
            let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
            let referenceLogits = try XCTUnwrap(arrays["output"])

            let directory = URL(fileURLWithPath: release)
            let configuration = try NFKMLXGemma3nLanguage.configuration(
                fromHuggingFace: directory.appendingPathComponent("config.json"))
            XCTAssertEqual(configuration.layerCount, layers, "\(name): the layer count")
            let net = NFKMLXGemma3nNet(configuration)
            try NFKMLXGemma3nLanguage.loadWeights(into: net, fromDirectory: directory, precision: .checkpoint)

            let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
            eval(logits)
            XCTAssertEqual(logits.shape, [1, referenceLogits.shape[0], referenceLogits.shape[1]])

            let ours = logits[0].reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init)
            let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
            let similarity = cosine(ours, theirs)
            let vocabulary = referenceLogits.shape[1]
            var agreements = 0
            for position in 0 ..< referenceLogits.shape[0] {
                let base = position * vocabulary
                let mine = Array(ours[base ..< base + vocabulary]), reference = Array(theirs[base ..< base + vocabulary])
                let ourBest = (0 ..< vocabulary).max { mine[$0] < mine[$1] }!
                let theirBest = (0 ..< vocabulary).max { reference[$0] < reference[$1] }!
                var note = ""
                if ourBest == theirBest {
                    agreements += 1
                } else {
                    note = "; argmax \(ourBest) vs \(theirBest), reference margin \(reference[theirBest] - reference[ourBest]), ours \(mine[ourBest] - mine[theirBest])"
                }
                print("VALIDATION PARITY \(name): position \(position) cosine \(cosine(mine, reference))\(note)")
            }
            print("VALIDATION PARITY \(name): logit cosine \(similarity), argmax \(agreements)/\(tokens.count)")
            XCTAssertGreaterThan(similarity, 0.999, "\(name): the decoder matches the reference at the released precision")
            XCTAssertGreaterThanOrEqual(agreements, tokens.count - 1,
                                        "\(name): at most one bf16 near-tie flips the argmax")
        }
    }

    // Qwen3-VL's other sizes. The 4B keeps the 2B's vision tower; the 8B, 32B, and 30B-A3B run the
    // deeper 27-block one, and the 30B-A3B's decoder routes fused experts, which the loader splits.
    func testQwen3VLLargerSizesMatchTheReleasedShapes() throws {
        try requireMLXRuntime()
        for (name, depth) in [("qwen3-vl-4b", 24), ("qwen3-vl-8b", 27), ("qwen3-vl-32b", 27), ("qwen3-vl-30b-a3b", 27)] {
            let release = try shapes(name)
            let visionConfiguration = try NFKMLXQwen3VLVisionConfiguration.configuration(fromHuggingFace: release.config)
            XCTAssertEqual(visionConfiguration.depth, depth)
            let vision = NFKMLXQwen3VLVisionNet(visionConfiguration)

            let data = try Data(contentsOf: release.config)
            let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            var decoderConfiguration = try NFKMLXLanguage.configuration(fromJSON: try XCTUnwrap(json["text_config"] as? [String: Any]))
            decoderConfiguration.tiesWordEmbeddings = release.shapes["lm_head.weight"] == nil
            let decoder = NFKMLXLanguageNet(decoderConfiguration)

            var released = [String: [Int]]()
            for (key, shape) in release.shapes {
                if key.hasPrefix("model.visual.") {
                    let name = "vision." + key.dropFirst("model.visual.".count)
                    // The 5-D patch convolution is the module's flattened linear weight.
                    released[name] = shape.count == 5 ? [shape[0], shape[1] * shape[2] * shape[3] * shape[4]] : shape
                } else if key.hasPrefix("model.language_model.") {
                    let name = "model." + key.dropFirst("model.language_model.".count)
                    if name.hasSuffix(".mlp.experts.gate_up_proj") {
                        let base = String(name.dropLast("gate_up_proj".count))
                        released[base + "gate_proj.weight"] = [shape[0], shape[2] / 2, shape[1]]
                        released[base + "up_proj.weight"] = [shape[0], shape[2] / 2, shape[1]]
                    } else if name.hasSuffix(".mlp.experts.down_proj") {
                        released[name + ".weight"] = [shape[0], shape[2], shape[1]]
                    } else {
                        released[name] = shape
                    }
                } else {
                    released[key] = shape
                }
            }
            let built = inventory(vision) { "vision." + $0 } + inventory(decoder)
            assertStructure(name, built: built, released: released)
        }
    }

    // The released Qwen-Image 2.1 transformer, 7.1B over 297 tensors. The arithmetic is measured at a
    // tiny configuration against diffusers; this holds the module the released checkpoint builds to
    // every tensor the release ships, which a tiny oracle cannot catch because it builds the reference
    // from the port's own parameters.
    func testQwenImageTransformerMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("qwen-image-2.1-transformer")
        let configuration = try NFKMLXQwenImage.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.layers, 32)
        XCTAssertEqual(configuration.dimensions, 4096)
        let net = NFKMLXQwenImage.makeNet(configuration)

        var released = [String: [Int]]()
        for (key, shape) in release.shapes {
            guard let name = NFKMLXQwenImage.remapReferenceKey(key) else { continue }
            released[name] = shape
        }
        assertStructure("qwen-image-2.1-transformer", built: inventory(net), released: released)
    }

    // The Qwen3-VL retrieval pair's 8B sizes, which are the same architecture as the measured 2B at
    // the deeper 27-block tower. The 8B reranker ships `lm_head.weight` although its config says the
    // embeddings are tied, which is why the port settles the tie from the weights rather than the
    // config; the 8B embedder ships none, one tensor fewer.
    func testQwen3VLRetrievalSizesMatchTheReleasedShapes() throws {
        try requireMLXRuntime()
        for name in ["qwen3-vl-embedding-8b", "qwen3-vl-reranker-8b"] {
            let release = try shapes(name)
            let visionConfiguration = try NFKMLXQwen3VLVisionConfiguration.configuration(fromHuggingFace: release.config)
            XCTAssertEqual(visionConfiguration.depth, 27)
            let vision = NFKMLXQwen3VLVisionNet(visionConfiguration)

            let data = try Data(contentsOf: release.config)
            let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            var decoderConfiguration = try NFKMLXLanguage.configuration(
                fromJSON: try XCTUnwrap(json["text_config"] as? [String: Any]))
            decoderConfiguration.tiesWordEmbeddings = release.shapes["lm_head.weight"] == nil
            let decoder = NFKMLXLanguageNet(decoderConfiguration)

            var released = [String: [Int]]()
            for (key, shape) in release.shapes {
                if key.hasPrefix("model.visual.") {
                    let name = "vision." + key.dropFirst("model.visual.".count)
                    released[name] = shape.count == 5 ? [shape[0], shape[1] * shape[2] * shape[3] * shape[4]] : shape
                } else if key.hasPrefix("model.language_model.") {
                    released["model." + key.dropFirst("model.language_model.".count)] = shape
                } else {
                    released[key] = shape
                }
            }
            let built = inventory(vision) { "vision." + $0 } + inventory(decoder)
            assertStructure(name, built: built, released: released)
        }
    }

    // Every SigLIP 2 release beyond base-patch16-224: three tower geometries at five patch/resolution
    // pairings, plus giant-opt's text projection to the wider image width.
    func testSigLIP2SizesMatchTheReleasedShapes() throws {
        try requireMLXRuntime()
        for variant in NFKMLXSigLIP2.allVariants where variant != .basePatch16At224 {
            let spec = NFKMLXSigLIP2.specs(for: variant)
            let release = try shapes(spec.name)
            let net = NFKMLXSigLIP2Net(spec.configuration)
            let built = inventory(net, rename: { name in
                if name.hasPrefix("vision.") { return "vision_model." + name.dropFirst("vision.".count) }
                if name.hasPrefix("text.") { return "text_model." + name.dropFirst("text.".count) }
                return name
            }, reshape: { _, shape in self.convLayout(shape) })
            assertStructure(spec.name, built: built, released: release.shapes)
        }
    }

    // MARK: - Parity on the released weights

    // The three Whisper sizes the size test did not cover: base, the pre-v3 large (its v2 checkpoint),
    // and large-v3-turbo, whose four-layer decoder is the one asymmetric geometry.
    func testWhisperRemainingSizesMatchTheReference() throws {
        try requireMLXRuntime()
        let sizes: [(String, String, String, NFKMLXWhisperConfiguration)] = [
            ("base", "IK_VAL_WHISPER_BASE", "IK_PARITY_WHISPER_BASE", .base),
            ("large-v2", "IK_VAL_WHISPER_LARGE_V2", "IK_PARITY_WHISPER_LARGE_V2", .large),
            ("large-v3-turbo", "IK_VAL_WHISPER_LARGE_V3_TURBO", "IK_PARITY_WHISPER_LARGE_V3_TURBO", .largeV3Turbo),
        ]
        for (name, weightsKey, parityKey, geometry) in sizes {
            guard let arrays = try? record(parityKey) else { print("SKIP whisper-\(name): no \(parityKey)"); continue }
            let waveform = try XCTUnwrap(arrays["waveform"]).asArray(Float.self)
            let referenceTokens = try XCTUnwrap(arrays["output"]).asArray(Int32.self)
            if let prompt = arrays["prompt"] {
                XCTAssertEqual(prompt.asArray(Int32.self).map(Int.init), geometry.promptTokens,
                               "\(name): the port's prompt is the reference's")
            }
            var plain = geometry
            plain.suppressesBlankStart = false
            let net = NFKMLXWhisper.makeNet(plain)
            try NFKMLXWhisper.loadWeights(into: net, from: weights(weightsKey))

            var prepared = waveform
            let targetSamples = 30 * 16000
            if prepared.count < targetSamples {
                prepared += [Float](repeating: 0, count: targetSamples - prepared.count)
            }
            let mel = NFKMLXMel.logMel(prepared, sampleRate: 16000, nMels: plain.nMels)
            eval(mel)
            let tokens = net.transcribe(mel).map { Int32($0) }
            print("VALIDATION PARITY whisper-\(name): tokens \(tokens) vs \(Array(referenceTokens))")
            XCTAssertEqual(tokens, Array(referenceTokens), "\(name): greedy decoding matches")
        }
    }

    // SAM's ViT-L and ViT-H encoders, and the mask each predicts through the shared decoder, against
    // the official predictor on a 1024×1024 plate.
    func testSAMLargerEncodersMatchTheReference() throws {
        try requireMLXRuntime()
        let mean = MLXArray([Float(123.675), 116.28, 103.53])
        let deviation = MLXArray([Float(58.395), 57.12, 57.375])
        for (name, weightsKey, encoderKey, maskKey, configuration) in [
            ("vit-l", "IK_VAL_SAM_L", "IK_PARITY_SAM_L_ENCODER", "IK_PARITY_SAM_L_MASK", NFKMLXSAMConfiguration.vitL),
            ("vit-h", "IK_VAL_SAM_H", "IK_PARITY_SAM_H_ENCODER", "IK_PARITY_SAM_H_MASK", NFKMLXSAMConfiguration.vitH),
        ] {
            guard let encoderRecord = try? record(encoderKey) else { print("SKIP sam-\(name): no \(encoderKey)"); continue }
            let inputTensor = try XCTUnwrap(encoderRecord["input_image"])
            let referenceFeatures = try XCTUnwrap(encoderRecord["output"])
            let net = NFKMLXSAM.makeNet(configuration)
            try NFKMLXSAM.loadWeights(into: net, from: weights(weightsKey))

            let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
            let prepared = (inputTensor.reshaped([1, height, width, 3]) * 255 - mean) / deviation
            let features = net.imageEncoder(prepared)
            let similarity = cosine(features, referenceFeatures)
            print("VALIDATION PARITY sam-\(name) encoder: cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.999, "\(name): the ViT image encoder matches the reference")

            guard let maskRecord = try? record(maskKey) else { continue }
            let referenceMask = try XCTUnwrap(maskRecord["output"])
            let mask = net.segment(prepared, pointX: 0.5, pointY: 1.0 / 3.0)
            eval(mask)
            let side = mask.shape[1]
            let reference = NFKMLXResample.resizeBilinear(referenceMask.reshaped([1, height, width, 1]),
                                                          height: side, width: side)
            let ours = mask.reshaped([side * side]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([side * side]).asArray(Float.self).map(Double.init)
            let agreement = Double(zip(ours, theirs).filter { ($0 > 0.5) == ($1 > 0.5) }.count) / Double(ours.count)
            print("VALIDATION PARITY sam-\(name) mask: cosine \(cosine(ours, theirs)), binary agreement \(agreement)")
            XCTAssertGreaterThan(agreement, 0.95, "\(name): the predicted mask matches the reference predictor")
        }
    }

    // SAM 2's small Hiera: tiny's width over a deeper third stage with later global attention, the
    // fourth released size.
    func testSAM2SmallEncoderMatchesTheReference() throws {
        try requireMLXRuntime()
        let arrays = try record("IK_PARITY_SAM2_SMALL")
        let inputTensor = try XCTUnwrap(arrays["input_image"])
        let net = NFKMLXSAM2.makeEncoder(.small)
        try NFKMLXSAM2.loadWeights(into: net, from: weights("IK_VAL_SAM2_SMALL"))
        net.train(false)
        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let levels = net.features(inputTensor.reshaped([1, height, width, 3]))
        eval(levels)
        for (index, name) in ["level0", "level1"].enumerated() {
            let reference = try XCTUnwrap(arrays[name])
            let similarity = cosine(levels[index], reference)
            print("VALIDATION PARITY sam2-small \(name): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "small \(name) matches the reference")
        }
    }

    // Depth Anything 3's base and large, against the authors' package at each size's own geometry:
    // the four hooked backbone features, the exp-depth map (mean-removed), the ray branch, and the
    // camera the decoder reads off the last hook's token.
    func testDepthAnything3LargerSizesMatchTheReference() throws {
        try requireMLXRuntime()
        for (name, weightsKey, parityKey, configuration) in [
            ("base", "IK_VAL_DEPTH3_BASE_WEIGHTS", "IK_PARITY_DEPTH3_BASE", NFKMLXDepth3Configuration.base),
            ("large", "IK_VAL_DEPTH3_LARGE_WEIGHTS", "IK_PARITY_DEPTH3_LARGE", NFKMLXDepth3Configuration.large),
        ] {
            guard let reference = try? record(parityKey) else { print("SKIP da3-\(name): no \(parityKey)"); continue }
            let net = NFKMLXDepthAnything3.makeNet(configuration)
            try NFKMLXDepthAnything3.loadWeights(into: net, from: weights(weightsKey))
            let input = try XCTUnwrap(reference["input_image"]).reshaped([1, 518, 518, 3])
            let features = net.features(input)
            eval(features)
            for i in 0 ..< 4 {
                let similarity = cosine(features[i][0], try XCTUnwrap(reference["hook\(i)"]))
                print("VALIDATION PARITY da3-\(name) hook\(i): cosine \(similarity)")
                XCTAssertGreaterThan(similarity, 0.9999, "\(name) hook\(i) matches the reference backbone feature")
            }
            let depth = net.head(features).depth
            eval(depth)
            let mine = depth[0].reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = try XCTUnwrap(reference["output"]).reshaped([-1]).asArray(Float.self).map(Double.init)
            let meanA = mine.reduce(0, +) / Double(mine.count), meanB = theirs.reduce(0, +) / Double(theirs.count)
            let meanRemoved = cosine(mine.map { $0 - meanA }, theirs.map { $0 - meanB })
            print("VALIDATION PARITY da3-\(name) depth: cosine \(cosine(mine, theirs)), mean-removed \(meanRemoved)")
            XCTAssertGreaterThan(meanRemoved, 0.9999, "\(name): the depth structure matches the reference")

            let prediction = net.head(features)
            eval(prediction.ray, prediction.rayConfidence)
            let ray = cosine(prediction.ray[0], try XCTUnwrap(reference["ray"]))
            let rayConfidence = cosine(prediction.rayConfidence[0], try XCTUnwrap(reference["ray_conf"]))
            print("VALIDATION PARITY da3-\(name) ray: cosine \(ray), confidence \(rayConfidence)")
            XCTAssertGreaterThan(ray, 0.9999, "\(name): the ray map matches the reference")
            XCTAssertGreaterThan(rayConfidence, 0.9999, "\(name): the ray confidence matches the reference")

            let cameras = net.backbone.hooked(input).cameraTokens
            let pose = net.cameraDecoder(cameras[cameras.count - 1]).encoding
            eval(pose)
            let poseSimilarity = cosine(pose, try XCTUnwrap(reference["pose_enc"]))
            print("VALIDATION PARITY da3-\(name) pose_enc: cosine \(poseSimilarity)")
            XCTAssertGreaterThan(poseSimilarity, 0.9999, "\(name): the predicted pose matches the reference")

            let tokens = net.cameraEncoder(try XCTUnwrap(reference["cam_enc_pose_encoding"]).reshaped([1, 1, 9]))
            eval(tokens)
            let tokenSimilarity = cosine(tokens, try XCTUnwrap(reference["cam_enc_tokens"]))
            print("VALIDATION PARITY da3-\(name) cam_enc: cosine \(tokenSimilarity)")
            XCTAssertGreaterThan(tokenSimilarity, 0.9999, "\(name): the camera encoder matches the reference")
        }
    }

    // SmolVLM2's other released sizes, each read from its own config: the 256M keeps the 500M's vision
    // tower under a narrower decoder, and the 2.2B changes the tower, the decoder, the pixel-shuffle
    // factor, and the tiling together, and ships sharded.
    func testSmolVLMOtherSizesMatchTheReference() throws {
        try requireMLXRuntime()
        for (name, directoryKey, parityKey) in [
            ("256m", "IK_VAL_SMOLVLM2_256M", "IK_PARITY_SMOLVLM_256M"),
            ("2.2b", "IK_VAL_SMOLVLM2_2_2B", "IK_PARITY_SMOLVLM_2_2B"),
        ] {
            guard let parityPath = config[parityKey], let directory = config[directoryKey],
                  FileManager.default.fileExists(atPath: parityPath) else {
                print("SKIP smolvlm-\(name): no \(parityKey)")
                continue
            }
            let arrays = try loadArrays(url: URL(fileURLWithPath: parityPath))
            let directoryURL = URL(fileURLWithPath: directory)
            let release = try NFKMLXSmolVLM.release(directoryURL: directoryURL)
            let net = try NFKMLXSmolVLM.model(directoryURL: directoryURL)

            // The vision tower on the reference's own pixel values, so the tiling is out of the
            // comparison.
            let pixels = try XCTUnwrap(arrays["pixel_values"])
            let tiles = pixels.reshaped([-1, 3, release.vision.imageSize, release.vision.imageSize])
            let vision = net.vision(tiles.transposed(0, 2, 3, 1))
            eval(vision)
            let visionSimilarity = cosine(
                vision.reshaped([-1]).asArray(Float.self).map(Double.init),
                try XCTUnwrap(arrays["vision_hidden"]).reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY smolvlm-\(name): vision cosine \(visionSimilarity)")
            XCTAssertGreaterThan(visionSimilarity, 0.9999, "\(name): the vision tower matches the reference")

            let features = net.connector(vision)
            eval(features)
            let connectorSimilarity = cosine(
                features.reshaped([-1]).asArray(Float.self).map(Double.init),
                try XCTUnwrap(arrays["image_features"]).reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY smolvlm-\(name): connector cosine \(connectorSimilarity)")
            XCTAssertGreaterThan(connectorSimilarity, 0.9999, "\(name): the connector matches the reference")
            // The connector folds each tile's patches by the release's own pixel-shuffle factor.
            XCTAssertEqual(features.dim(1), release.tokensPerTile)

            let ids = try XCTUnwrap(arrays["input_ids"]).asArray(Int32.self).map { Int($0) }
            let logits = net.logits(inputIds: ids, pixelValues: tiles)
            eval(logits)
            let reference = try XCTUnwrap(arrays["output"])
            let ourArgmax = logits.argMax(axis: -1).asArray(Int32.self)
            let theirArgmax = reference.argMax(axis: -1).asArray(Int32.self)
            let agreement = zip(ourArgmax, theirArgmax).filter { $0 == $1 }.count
            let logitSimilarity = cosine(
                logits.reshaped([-1]).asArray(Float.self).map(Double.init),
                reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY smolvlm-\(name): logit cosine \(logitSimilarity), argmax agreement \(agreement)/\(ourArgmax.count)")
            XCTAssertGreaterThan(logitSimilarity, 0.9999, "\(name): the language head matches the reference")

            // A position where the two argmaxes differ is only defensible when the reference itself puts
            // the two tokens within float noise of each other; anything wider is a real disagreement.
            for (index, pair) in zip(ourArgmax, theirArgmax).enumerated() where pair.0 != pair.1 {
                let row = reference[index]
                let gap = row[Int(pair.1)].item(Float.self) - row[Int(pair.0)].item(Float.self)
                print("VALIDATION PARITY smolvlm-\(name): position \(index) tie, reference gap \(gap)")
                XCTAssertLessThan(gap, 1e-3, "\(name): position \(index) is a float tie, not a disagreement")
            }
        }
    }

    // NAFNet's width-64 SIDD and GoPro releases, which the width-32 presets could not load.
    func testNAFNetWidth64ReleasesMatchTheReference() throws {
        try requireMLXRuntime()
        for (name, weightsKey, parityKey, configuration) in [
            ("sidd-64", "IK_VAL_NAFNET_SIDD64", "IK_PARITY_NAFNET_SIDD64", NFKMLXNAFNetConfiguration.siddWidth64),
            ("gopro-64", "IK_VAL_NAFNET_GOPRO64", "IK_PARITY_NAFNET_GOPRO64", NFKMLXNAFNetConfiguration.goProWidth64),
        ] {
            guard let arrays = try? record(parityKey) else { print("SKIP nafnet-\(name): no \(parityKey)"); continue }
            let net = NFKMLXNAFNet.makeNet(configuration)
            try NFKMLXNAFNet.loadWeights(into: net, from: weights(weightsKey))
            let restored = net.restore(try XCTUnwrap(arrays["input_image"]))
            let similarity = cosine(restored, try XCTUnwrap(arrays["output"]))
            print("VALIDATION PARITY nafnet-\(name): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.999, "\(name) matches the reference restoration")
        }
    }

    // SwinIR's remaining releases: the classical ×2, the lightweight ×3 and ×4, and the two real-world
    // models — the medium with the nearest-neighbor tail and the large with the three-convolution
    // residual connection as well.
    func testSwinIRRemainingReleasesMatchTheReference() throws {
        try requireMLXRuntime()
        for (name, weightsKey, parityKey, configuration) in [
            ("classical-x2", "IK_VAL_SWINIR_X2", "IK_PARITY_SWINIR_X2", NFKMLXSwinIRConfiguration.classicalSRx2),
            ("light-x3", "IK_VAL_SWINIR_LIGHT_X3", "IK_PARITY_SWINIR_LIGHT_X3", .lightweightSRx3),
            ("light-x4", "IK_VAL_SWINIR_LIGHT_X4", "IK_PARITY_SWINIR_LIGHT_X4", .lightweightSRx4),
            ("real-m-x4", "IK_VAL_SWINIR_REAL_M", "IK_PARITY_SWINIR_REAL_M", .realSRx4Medium),
            ("real-l-x4", "IK_VAL_SWINIR_REAL_L", "IK_PARITY_SWINIR_REAL_L", .realSRx4Large),
        ] {
            guard let arrays = try? record(parityKey) else { print("SKIP swinir-\(name): no \(parityKey)"); continue }
            let net = try NFKMLXSwinIR.makeNet(configuration)
            try NFKMLXSwinIR.loadWeights(into: net, from: weights(weightsKey))
            let upscaled = net.upscale(try XCTUnwrap(arrays["input_image"]))
            let reference = try XCTUnwrap(arrays["output"])
            XCTAssertEqual(upscaled.shape, reference.shape, "\(name): same upscaled volume as the reference")
            let ours = upscaled.reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map { Double(max(0, min(1, $0))) }
            let similarity = cosine(ours, theirs)
            let meanAbsolute = zip(ours, theirs).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(ours.count)
            print("VALIDATION PARITY swinir-\(name): cosine \(similarity), mean |difference| \(meanAbsolute)")
            XCTAssertGreaterThan(similarity, 0.999, "\(name) matches the reference upscale")
            XCTAssertLessThan(meanAbsolute, 0.01, "\(name) agrees pixelwise")
        }
    }

    // Robust Video Matting's ResNet-50 release: torchvision's bottleneck encoder, tapped at the stem
    // and the first two stages, with the last stage dilated, over the wider decoder.
    func testRVMResNet50MatchesTheReference() throws {
        try requireMLXRuntime()
        let arrays = try record("IK_PARITY_RVM_RESNET50")
        let inputTensor = try XCTUnwrap(arrays["input_image"])
        let net = NFKMLXRVMNet(.resNet50)
        try NFKMLXRVM.loadWeights(into: net, from: weights("IK_VAL_RVM_RESNET50"))
        net.train(false)
        let frame = inputTensor.reshaped([1, inputTensor.shape[0], inputTensor.shape[1], 3])
        let (foreground, alpha, _) = net.forward(frame, state: NFKMLXRVMNet.initialState)
        eval(foreground, alpha)
        let alphaSimilarity = cosine(alpha, try XCTUnwrap(arrays["output"]))
        let foregroundSimilarity = cosine(foreground, try XCTUnwrap(arrays["foreground"]))
        print("VALIDATION PARITY rvm-resnet50: alpha cosine \(alphaSimilarity), foreground cosine \(foregroundSimilarity)")
        XCTAssertGreaterThan(alphaSimilarity, 0.999, "the alpha matches the reference")
        XCTAssertGreaterThan(foregroundSimilarity, 0.999, "the composited foreground matches too")
        if let refined = arrays["alpha_downsampled"] {
            let (_, halfAlpha, _) = net.forward(frame, state: NFKMLXRVMNet.initialState, downsampleRatio: 0.5)
            let refinedSimilarity = cosine(halfAlpha, refined)
            print("VALIDATION PARITY rvm-resnet50: refined (downsample 0.5) alpha cosine \(refinedSimilarity)")
            XCTAssertGreaterThan(refinedSimilarity, 0.999, "the guided-filter pass matches as well")
        }
    }

    // SNAC's music codecs: four codebooks at strides [8, 4, 2, 1] and the bottleneck's windowed
    // attention, which the speech release does not have.
    func testSNACMusicCodecsMatchTheReference() throws {
        try requireMLXRuntime()
        for (name, weightsKey, parityKey, configuration) in [
            ("32khz", "IK_VAL_SNAC_32KHZ", "IK_PARITY_SNAC_32KHZ", NFKMLXSNACConfiguration.snac32kHz),
            ("44khz", "IK_VAL_SNAC_44KHZ", "IK_PARITY_SNAC_44KHZ", NFKMLXSNACConfiguration.snac44kHz),
        ] {
            guard let arrays = try? record(parityKey) else { print("SKIP snac-\(name): no \(parityKey)"); continue }
            let waveform = try XCTUnwrap(arrays["waveform"])
            let referenceRecon = try XCTUnwrap(arrays["output"])
            var referenceGrid = [[Int]]()
            var index = 0
            while let stream = arrays["codes\(index)"] {
                referenceGrid.append(stream.asType(.int32).asArray(Int32.self).map(Int.init))
                index += 1
            }
            XCTAssertEqual(referenceGrid.count, 4, "\(name): four codebooks")
            let codec = try NFKMLXSNAC.codec(configuration: configuration, weightsURL: weights(weightsKey))
            let mine = codec.encode(waveform.asArray(Float.self))
            XCTAssertEqual(mine.map(\.count), referenceGrid.map(\.count), "\(name): the per-codebook code counts match")
            var agree = 0, total = 0
            for (mineStream, referenceStream) in zip(mine, referenceGrid) {
                for (a, b) in zip(mineStream, referenceStream) where a == b { agree += 1 }
                total += referenceStream.count
            }
            let myRecon = codec.decode(referenceGrid, deterministic: true)
            let count = min(myRecon.count, referenceRecon.shape[0])
            let similarity = cosine(Array(myRecon.prefix(count)).map(Double.init),
                                    referenceRecon[0 ..< count].asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY snac-\(name): code agreement \(agree)/\(total), reconstruction cosine \(similarity)")
            XCTAssertGreaterThan(Double(agree) / Double(max(total, 1)), 0.99, "\(name): the codes match the reference's")
            XCTAssertGreaterThan(similarity, 0.999, "\(name): the decoder reconstructs the reference's waveform")
        }
    }

    // The six-stem release and the fine-tuned bag, each against demucs' own separation of the clip.
    func testHTDemucsSixStemAndFineTunedReleasesMatchTheReference() throws {
        try requireMLXRuntime()
        if let arrays = try? record("IK_PARITY_HTDEMUCS_6S") {
            let net = NFKMLXHTDemucs.makeNet(.htdemucs6s)
            try NFKMLXHTDemucs.loadWeights(into: net, from: weights("IK_VAL_HTDEMUCS_6S"))
            let stems = net.separate(try XCTUnwrap(arrays["waveform"]), padsToTrainingSegment: false)
            let reference = try XCTUnwrap(arrays["output"])
            XCTAssertEqual(stems.shape, reference.shape, "six stems of the clip")
            let similarity = cosine(stems, reference)
            print("VALIDATION PARITY htdemucs-6s: separated stems cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.999, "the six-stem separation matches the reference")
        } else {
            print("SKIP htdemucs-6s: no IK_PARITY_HTDEMUCS_6S")
        }
        let fineTuned = (1 ... 4).compactMap { config["IK_VAL_HTDEMUCS_FT_\($0)"] }
        if let arrays = try? record("IK_PARITY_HTDEMUCS_FT"), fineTuned.count == 4 {
            let bag = try NFKMLXHTDemucs.fineTunedBag(weightsURLs: fineTuned.map { URL(fileURLWithPath: $0) })
            let stems = bag.separate(try XCTUnwrap(arrays["waveform"]), padsToTrainingSegment: false)
            let reference = try XCTUnwrap(arrays["output"])
            XCTAssertEqual(stems.shape, reference.shape, "four stems of the clip")
            let similarity = cosine(stems, reference)
            print("VALIDATION PARITY htdemucs-ft: separated stems cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.999, "the fine-tuned bag matches the reference's combination")
        } else {
            print("SKIP htdemucs-ft: no IK_PARITY_HTDEMUCS_FT / IK_VAL_HTDEMUCS_FT_1…4")
        }
    }

    // RT-DETR's other releases, on the reference's own pixels and query selection: the basic-block
    // ResNet-18-vd and ResNet-34-vd backbones at half CSP expansion, and ResNet-101-vd with its wider
    // hybrid encoder.
    func testRTDetrRemainingSizesMatchTheReference() throws {
        try requireMLXRuntime()
        for (name, directoryKey, parityKey, configuration) in [
            ("r18vd", "IK_VAL_RTDETR_R18VD", "IK_PARITY_RTDETR_R18VD", NFKMLXRTDetrConfiguration.r18vd),
            ("r34vd", "IK_VAL_RTDETR_R34VD", "IK_PARITY_RTDETR_R34VD", .r34vd),
            ("r101vd", "IK_VAL_RTDETR_R101VD", "IK_PARITY_RTDETR_R101VD", .r101vd),
        ] {
            guard let arrays = try? record(parityKey), let directory = config[directoryKey] else {
                print("SKIP rtdetr-\(name): no \(parityKey)"); continue
            }
            let pixels = try XCTUnwrap(arrays["pixels"]).expandedDimensions(axis: 0)
            let net = NFKMLXRTDetrNet(configuration)
            try NFKMLXRTDetr.loadWeights(into: net, from: URL(fileURLWithPath: directory).appendingPathComponent("model.safetensors"))
            net.train(false)
            let detection = net(pixels)
            let (logits, boxes) = net.decode(indices: try XCTUnwrap(arrays["topk_ind"]), detection: detection)
            let logitSimilarity = cosine(logits, try XCTUnwrap(arrays["output"]))
            let boxSimilarity = cosine(boxes, try XCTUnwrap(arrays["pred_boxes"]))
            print("VALIDATION PARITY rtdetr-\(name): logits cosine \(logitSimilarity), boxes cosine \(boxSimilarity)")
            XCTAssertGreaterThan(logitSimilarity, 0.9999, "\(name): the detection logits match the reference")
            XCTAssertGreaterThan(boxSimilarity, 0.9999, "\(name): the detection boxes match the reference")
        }
    }

    // RF-DETR's four later releases, which share one recipe (16-pixel patches at the run resolution,
    // two windows a side, stages 3/6/9/12) and differ in resolution and decoder depth.
    func testRFDetrRemainingSizesMatchTheReference() throws {
        try requireMLXRuntime()
        for variant in [NFKMLXRFDetrVariant.nano, .small, .medium, .large] {
            let spec = NFKMLXRFDetr.specs(for: variant)
            let suffix = spec.name.dropFirst("rf-detr-".count).uppercased()
            guard let arrays = try? record("IK_PARITY_RF_DETR_\(suffix)"), let directory = config["IK_VAL_RFDETR_\(suffix)"] else {
                print("SKIP \(spec.name): no IK_PARITY_RF_DETR_\(suffix)"); continue
            }
            let pixels = try XCTUnwrap(arrays["pixels"]).expandedDimensions(axis: 0)
            XCTAssertEqual(pixels.shape[1], spec.configuration.inputResolution, "\(spec.name): the reference ran at the preset's resolution")
            let net = NFKMLXRFDetrNet(spec.configuration)
            try NFKMLXRFDetr.loadWeights(into: net, from: URL(fileURLWithPath: directory).appendingPathComponent("model.safetensors"))
            net.train(false)
            let detection = net.detection(pixels, topkOverride: try XCTUnwrap(arrays["topk_ind"]))
            for (label, mine, key) in [("proj", detection.projected, "proj"), ("dec_last", detection.decoderLast, "dec_last"),
                                       ("logits", detection.logits, "output"), ("pred_boxes", detection.boxes, "pred_boxes")] {
                let similarity = cosine(mine, try XCTUnwrap(arrays[key]))
                print("VALIDATION PARITY \(spec.name): \(label) cosine \(similarity)")
                XCTAssertGreaterThan(similarity, 0.9999, "\(spec.name): \(label) matches the reference")
            }
        }
    }

    // CLIP's ViT-B/16 and ViT-L/14 load their released JIT archives strictly through the native reader —
    // the structural proof — and embed an image to a unit vector of the preset's width.
    func testCLIPLargerTowersLoadTheReleasedWeights() throws {
        try requireMLXRuntime()
        for (name, key, configuration) in [("vit-b-16", "IK_VAL_CLIP_B16", NFKMLXCLIPConfiguration.vitB16),
                                           ("vit-l-14", "IK_VAL_CLIP_L14", .vitL14)] {
            guard let url = try? weights(key) else { print("SKIP clip-\(name): no \(key)"); continue }
            let net = NFKMLXCLIPNet(configuration)
            try NFKMLXCLIP.loadWeights(into: net, from: url)
            let pixels = MLXRandom.uniform(low: 0, high: 1, [configuration.imageResolution, configuration.imageResolution, 3])
            let embedding = net.encodeImage(pixels)
            eval(embedding)
            XCTAssertEqual(embedding.shape, [configuration.embedDimensions], "clip-\(name): the embedding width")
            let norm = sqrt(embedding.square().sum()).item(Float.self)
            XCTAssertEqual(norm, 1, accuracy: 1e-3, "clip-\(name): a unit-length embedding")
            print("VALIDATION clip-\(name): loaded strictly, embedding norm \(norm)")
        }
    }

    // SD3.5-large (8B): the same MMDiT the tiny config is at parity on, read by the configuration
    // reader and held to the released transformer's own safetensors headers. 38 layers, 38 heads, RMS
    // qk-norm, standard MMDiT (no dual attention — that is SD3.5-medium's MMDiT-X). The patch-embed
    // convolution is the only 4-D tensor, compared in the release's `[out, in, kH, kW]` layout.
    func testSD35LargeMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("sd35-large")
        let configuration = try NFKMLXSD3TransformerNet.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.numLayers, 38)
        XCTAssertEqual(configuration.numAttentionHeads, 38)
        XCTAssertTrue(configuration.qkNorm)
        XCTAssertTrue(configuration.dualAttentionLayers.isEmpty)
        let net = NFKMLXSD3TransformerNet(configuration)
        assertStructure("sd35-large", built: inventory(net, reshape: { convLayout($1) }), released: release.shapes)
    }

    // SD3.5-medium (2.5B, MMDiT-X): the dual-attention path on the released weights — 24 layers, RMS
    // qk-norm, dual attention (`attn2`) on the first thirteen, and the wider `pos_embed_max_size` 384.
    // The tiny config exercises `attn2`; this holds it to the real shard headers.
    func testSD35MediumMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("sd35-medium")
        let configuration = try NFKMLXSD3TransformerNet.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.numLayers, 24)
        XCTAssertEqual(configuration.posEmbedMaxSize, 384)
        XCTAssertEqual(configuration.dualAttentionLayers, Array(0 ... 12))
        let net = NFKMLXSD3TransformerNet(configuration)
        assertStructure("sd35-medium", built: inventory(net, reshape: { convLayout($1) }), released: release.shapes)
    }

    // Every released SAM 2 and SAM 2.1 size, loaded whole into the tracker. The parity tests measure
    // the networks; this is the coverage claim — a release's checkpoint holds exactly what the module
    // declares, so nothing in it goes unread and nothing the module builds is left random. 2.1 carries
    // three tensors 2.0 does not: the occlusion spatial embedding and the object pointers' temporal
    // projection.
    func testSAM2SizesLoadTheirWholeCheckpoint() throws {
        try requireMLXRuntime()
        let releases: [(String, String, NFKMLXSAM2Variant, NFKMLXSAM2Release, Int)] = [
            ("sam2-tiny", "IK_VAL_SAM2", .tiny, .sam2, 468),
            ("sam2-small", "IK_VAL_SAM2_SMALL", .small, .sam2, 516),
            ("sam2-base-plus", "IK_VAL_SAM2_BASE_PLUS", .basePlus, .sam2, 612),
            ("sam2-large", "IK_VAL_SAM2_LARGE", .large, .sam2, 900),
            ("sam2.1-tiny", "IK_VAL_SAM2_1_TINY", .tiny, .sam21, 471),
        ]
        for (name, key, variant, release, tensors) in releases {
            guard let url = try? weights(key) else { print("SKIP \(name): no \(key)"); continue }
            let net = NFKMLXSAM2.makeTracker(variant: variant, release: release)
            let declared = net.parameters().flattened().count
            XCTAssertEqual(declared, tensors, "\(name): the module declares the release's tensor count")
            try NFKMLXSAM2.loadWeights(into: net, from: url)
            print("VALIDATION sam2 coverage \(name): \(declared) tensors loaded strictly")
        }
    }

    // Wan 2.2 Animate 2 (14B): the DiT the tiny config is at parity on, held to the released
    // `wan_animate_2_bf16.safetensors` header. The release ships the original Wan naming, so the
    // inventory is renamed through `releaseKey(forModule:)`; the patch-embedding convolution is the
    // only 5-D tensor and is compared in the release's `[out, in, kT, kH, kW]` layout. The model is
    // 32.8 GB in bf16 and its pipeline about 50 GB, so the shapes are all this machine can hold it to.
    func testWanAnimateMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("wan-animate-2-14b")
        let net = NFKMLXWanAnimateNet(.base)
        let built = inventory(net, rename: { NFKMLXWanAnimate.releaseKey(forModule: $0) },
                              reshape: { _, shape in
                                  shape.count == 5
                                      ? [shape[0], shape[4], shape[1], shape[2], shape[3]] : shape
                              })
        XCTAssertEqual(release.shapes.count, 1303, "the released inventory")
        assertStructure("wan-animate-2-14b", built: built, released: release.shapes)
    }

    // FLUX.1 [schnell] (12B): the double- and single-stream transformer the tiny config is at parity
    // on, read by the configuration reader and held to the released transformer's headers. Every
    // tensor is at most 2-D, so no layout change is needed. Ungated mirrors of the transformer are
    // used because the official repositories are gated.
    func testFluxSchnellMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("flux-schnell")
        let configuration = try NFKMLXFluxTransformerNet.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.numLayers, 19)
        XCTAssertEqual(configuration.numSingleLayers, 38)
        XCTAssertFalse(configuration.guidanceEmbeds, "schnell carries no guidance embedding")
        let net = NFKMLXFluxTransformerNet(configuration)
        assertStructure("flux-schnell", built: inventory(net), released: release.shapes)
    }

    // FLUX.1 [dev] (12B): the guidance-distilled variant, whose `time_text_embed.guidance_embedder`
    // the schnell size lacks. Held to the released dev transformer's headers.
    func testFluxDevMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("flux-dev")
        let configuration = try NFKMLXFluxTransformerNet.configuration(fromHuggingFace: release.config)
        XCTAssertTrue(configuration.guidanceEmbeds, "dev carries a guidance embedding")
        let net = NFKMLXFluxTransformerNet(configuration)
        assertStructure("flux-dev", built: inventory(net), released: release.shapes)
    }

    // FLUX.2 [klein] base 4B: the un-distilled sibling of klein 4B, also ungated. Its transformer
    // config is identical to klein 4B's, so what this checks is that the same declared geometry reads
    // a SECOND release rather than having been fitted to one.
    func testFlux2KleinBase4BMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("flux2-klein-base-4b")
        let configuration = try NFKMLXFlux2TransformerNet.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.numLayers, 5)
        XCTAssertEqual(configuration.numSingleLayers, 20)
        XCTAssertEqual(release.shapes.count, 169, "the released inventory")
        assertStructure("flux2-klein-base-4b", built: inventory(NFKMLXFlux2TransformerNet(configuration)),
                        released: release.shapes)
    }

    // FLUX.2's small decoder: the same encoder at a narrower decoder, which is the first release here
    // whose autoencoder is ASYMMETRIC. The `bn` buffers belong to the latent codec rather than the
    // autoencoder, so they are named as read elsewhere.
    func testFlux2SmallDecoderMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("flux2-small-decoder")
        XCTAssertEqual(release.shapes.count, 251, "the released inventory")
        // The inverse of `NFKMLXStableDiffusionModels.remapVAEKey`, which turns release names into
        // module names; a structural check needs module names back in the release's spelling.
        func releaseName(_ module: String) -> String {
            var name = module.replacingOccurrences(of: ".to_out.", with: ".to_out.0.")
            for part in ["norm1", "conv1", "norm2", "conv2", "conv_shortcut"] {
                for index in 0 ... 2 {
                    name = name.replacingOccurrences(of: ".resnets.\(index).block.\(part).",
                                                     with: ".resnets.\(index).\(part).")
                }
            }
            return name
        }
        let vae = NFKMLXSDAutoencoder(configuration: .flux2SmallDecoder)
        let built = inventory(vae, rename: { releaseName($0) },
                              reshape: { _, shape in convLayout(shape) })
        assertStructure("flux2-small-decoder", built: built, released: release.shapes,
                        named: { $0.hasPrefix("bn.") })
    }

    // LTX-2.3 22B: the released audio-video transformer this port is held to structurally. LTX-2.5's
    // own repositories are gated, so the inventory comes from the LTX-2.3 release, which is the same
    // `LTX2VideoTransformer3DModel` at the arrangement its `config.json` declares. Every tensor is at
    // most 2-D, so no layout change is needed.
    func testLTX2MatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("ltx2-23-22b")
        let configuration = try NFKMLXLTX2TransformerNet.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.numLayers, 48)
        XCTAssertEqual(configuration.innerDim, 4096)
        XCTAssertEqual(configuration.audioInnerDim, 2048)
        XCTAssertFalse(configuration.usesPromptEmbeddings,
                       "the text is projected by the pipeline's connectors, not the transformer")
        XCTAssertTrue(configuration.usesPromptAdaptiveNorm, "LTX-2.3 modulates the text key and value")
        XCTAssertEqual(release.shapes.count, 4186, "the released inventory")
        let net = NFKMLXLTX2TransformerNet(configuration)
        assertStructure("ltx2-23-22b", built: inventory(net), released: release.shapes)
    }

    // The LTX-2 configuration reader refuses every arrangement it does not build, rather than
    // ignoring the field. A release with the attention gates off ships no `to_gate_logits`, and
    // without this the failure would surface as a missing tensor at load rather than as the
    // configuration that asked for it.
    func testTheLTX2ReaderNamesWhatItDoesNotBuild() throws {
        try requireMLXRuntime()
        let base: [String: Any] = ["_class_name": "LTX2VideoTransformer3DModel", "rope_type": "split"]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx2-reader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        func read(_ overrides: [String: Any]) throws -> NFKMLXLTX2Configuration {
            var json = base
            overrides.forEach { json[$0.key] = $0.value }
            let url = directory.appendingPathComponent("config.json")
            try JSONSerialization.data(withJSONObject: json).write(to: url)
            return try NFKMLXLTX2TransformerNet.configuration(fromHuggingFace: url)
        }

        XCTAssertNoThrow(try read([:]), "the arrangement both releases carry is read")
        for refused: [String: Any] in [["rope_type": "interleaved"], ["cross_attn_mod": false],
                                       ["audio_cross_attn_mod": false], ["gated_attn": false],
                                       ["audio_gated_attn": false], ["qk_norm": "rms_norm"],
                                       ["norm_elementwise_affine": true],
                                       ["activation_fn": "gelu"],
                                       ["_class_name": "LTXVideoTransformer3DModel"]] {
            XCTAssertThrowsError(try read(refused), "\(refused.keys.first!) is refused by name")
        }
    }

    // FLUX.2 [klein] 4B: the released transformer this port is held to structurally. It is the one
    // FLUX.2 release that is NOT gated, so the inventory comes from Black Forest Labs' own repository
    // rather than a mirror. Every tensor is at most 2-D, so no layout change is needed.
    func testFlux2Klein4BMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("flux2-klein-4b")
        let configuration = try NFKMLXFlux2TransformerNet.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.numLayers, 5)
        XCTAssertEqual(configuration.numSingleLayers, 20)
        XCTAssertEqual(configuration.jointAttentionDim, 7680, "three Qwen3 layers wide")
        XCTAssertFalse(configuration.guidanceEmbeds, "the klein releases are step-distilled")
        XCTAssertEqual(release.shapes.count, 169, "the released inventory")
        let net = NFKMLXFlux2TransformerNet(configuration)
        assertStructure("flux2-klein-4b", built: inventory(net), released: release.shapes)
    }

    // FLUX.2 [dev] (32B) is behind Black Forest Labs' gate, so its headers cannot be walked here. Its
    // published parameter total can be, and it pins the geometry exactly: the declared `.dev` shape
    // reproduces 32,223,281,152 parameters to the tensor, the qk-norm vectors included. The same sum
    // over `.klein4B` reproduces that release's total, which is independently confirmed by the shape
    // walk above, so the arithmetic is checked against a measured case before it is trusted here.
    func testFlux2DevParameterTotalMatchesTheRelease() throws {
        try requireMLXRuntime()
        func total(_ configuration: NFKMLXFlux2Configuration) -> Int {
            NFKMLXFlux2TransformerNet(configuration).parameters().flattened()
                .reduce(0) { $0 + $1.1.size }
        }
        XCTAssertEqual(total(.klein4B), 3_875_544_576, "the FLUX.2 [klein] 4B release's own total")
        XCTAssertEqual(total(.dev), 32_223_281_152, "the FLUX.2 [dev] release's own total")
        XCTAssertEqual(total(.klein9B), 9_078_581_248, "the FLUX.2 [klein] 9B release's own total")
    }

    // The 9B split is the one thing the parameter total could not pin, because a double block costs
    // exactly two single blocks. It is read from the release rather than chosen.
    func testFlux2Klein9BMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("flux2-klein-base-9b")
        let configuration = try NFKMLXFlux2TransformerNet.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.numLayers, NFKMLXFlux2Configuration.klein9B.numLayers)
        XCTAssertEqual(configuration.numSingleLayers,
                       NFKMLXFlux2Configuration.klein9B.numSingleLayers)
        XCTAssertEqual(configuration.numAttentionHeads, 32)
        XCTAssertEqual(configuration.jointAttentionDim, 12288)
        XCTAssertFalse(configuration.guidanceEmbeds)
        // Seventeen splits reach the same total; only the release says which one it is.
        XCTAssertEqual(configuration.numLayers * 2 + configuration.numSingleLayers, 40)
        let net = NFKMLXFlux2TransformerNet(.klein9B)
        assertStructure("flux2-klein-base-9b", built: inventory(net), released: release.shapes)
    }

    // FLUX.2 [klein] 9B KV is trained for the reference cache, which is an inference mechanism, not
    // an architecture: its transformer is the base 9B's to the tensor, so `.klein9B` reads it too.
    func testFlux2Klein9BKVMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("flux2-klein-9b-kv")
        let configuration = try NFKMLXFlux2TransformerNet.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.numLayers, NFKMLXFlux2Configuration.klein9B.numLayers)
        XCTAssertEqual(configuration.numSingleLayers, NFKMLXFlux2Configuration.klein9B.numSingleLayers)
        XCTAssertEqual(configuration.jointAttentionDim, NFKMLXFlux2Configuration.klein9B.jointAttentionDim)
        let net = NFKMLXFlux2TransformerNet(.klein9B)
        assertStructure("flux2-klein-9b-kv", built: inventory(net), released: release.shapes)
    }

    // MARK: - Weight-free structure

    // The windowed attention at the bottleneck: a round trip at the tiny geometry, and the reference's
    // Sequential slots shifting by one past it.
    func testSNACAttentionRoundTripsAndShiftsTheSequentialSlots() throws {
        try requireMLXRuntime()
        let codec = try NFKMLXSNAC.codec(configuration: .tinyAttention, weightsURL: nil)
        let samples = (0 ..< 400).map { sinf(Float($0) * 0.05) * 0.5 }
        let codes = codec.encode(samples)
        XCTAssertEqual(codes.count, 2)
        XCTAssertEqual(codes[0].count * 2, codes[1].count, "the coarse codebook emits half the codes")
        let decoded = codec.decode(codes, deterministic: true)
        XCTAssertGreaterThanOrEqual(decoded.count, samples.count)

        let c = NFKMLXSNACConfiguration.tinyAttention
        XCTAssertEqual(NFKMLXSNAC.remapReferenceKey("encoder.block.3.norm.weight", c), "encoder.attention.norm.weight")
        XCTAssertEqual(NFKMLXSNAC.remapReferenceKey("encoder.block.4.weight", c), "encoder.conv_out.weight")
        XCTAssertEqual(NFKMLXSNAC.remapReferenceKey("decoder.model.2.to_qkv.weight", c), "decoder.attention.to_qkv.weight")
        XCTAssertEqual(NFKMLXSNAC.remapReferenceKey("decoder.model.3.block.0.alpha", c), "decoder.blocks.0.snake1.alpha")
        XCTAssertEqual(NFKMLXSNAC.remapReferenceKey("decoder.model.5.alpha", c), "decoder.snake.alpha")
        XCTAssertEqual(NFKMLXSNAC.remapReferenceKey("decoder.model.6.weight", c), "decoder.conv_out.weight")
        // Without attention the same slots keep their old meaning.
        XCTAssertEqual(NFKMLXSNAC.remapReferenceKey("encoder.block.3.weight", .tiny), "encoder.conv_out.weight")
        XCTAssertEqual(NFKMLXSNAC.remapReferenceKey("decoder.model.2.block.0.alpha", .tiny), "decoder.blocks.0.snake1.alpha")
    }

    // The basic-block backbone runs at the tiny geometry and carries the shortcut forms a stage needs:
    // a projecting first block and plain later ones.
    func testRTDetrBasicBlocksRunAtTheTinyGeometry() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXRTDetrConfiguration.tiny
        configuration.basicBlocks = true
        configuration.depths = [2, 2, 2, 2]
        let net = NFKMLXRTDetrNet(configuration)
        net.train(false)
        let keys = Set(net.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("backbone.model.encoder.stages.0.layers.0.shortcut.0.convolution.weight"),
                      "stage one's first block projects through a stride-1 shortcut")
        XCTAssertTrue(keys.contains("backbone.model.encoder.stages.1.layers.0.shortcut.1.convolution.weight"),
                      "a downsampling stage's first block projects after an average pool")
        XCTAssertFalse(keys.contains { $0.contains("stages.1.layers.1.shortcut") }, "later blocks pass the residual through")
        let detection = net(MLXRandom.uniform(low: 0, high: 1, [1, 64, 64, 3]))
        eval(detection.logits)
        XCTAssertEqual(detection.logits.shape, [configuration.numQueries, configuration.numLabels])
    }

    // A bag of one model with unit weights is that model; two copies of one model average to it.
    func testHTDemucsBagOfCopiesReproducesTheModel() throws {
        try requireMLXRuntime()
        var small = NFKMLXHTDemucsConfiguration.htdemucs
        small.channels = 8
        small.nFFT = 1024
        small.bottomChannels = 32
        small.transformerHeads = 2
        small.transformerHidden = 64
        small.trainingSamples = 8192
        let net = NFKMLXHTDemucs.makeNet(small)
        let mix = MLXRandom.uniform(low: -0.5, high: 0.5, [2, 8192])
        let alone = net.separate(mix)
        let bag = try NFKMLXHTDemucsBag(nets: [net, net], weights: [[1, 0, 1, 0], [0, 1, 0, 1]])
        let bagged = bag.separate(mix)
        XCTAssertGreaterThan(cosine(bagged, alone), 0.99999, "one-hot weights over copies of one model give the model")
        XCTAssertThrowsError(try NFKMLXHTDemucsBag(nets: [net], weights: [[1, 1]]), "a weight row must cover every stem")
    }

    // The ResNet encoder threads through the recurrent decoder at a test scale, with the taps at the
    // widths the decoder expects.
    func testRVMResNetEncoderRunsAtATestScale() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXRVMConfiguration(
            stemChannels: 8, lastChannels: 8 * 8 * 4, asppChannels: 16, blocks: [], captures: [],
            decoderChannels: [8, 6, 4, 4], refinerHiddenChannels: 4,
            resNet: NFKMLXResNetConfiguration(blocks: [1, 1, 1, 1], width: 8,
                                              replaceStrideWithDilation: [false, false, true]))
        XCTAssertEqual(configuration.featureChannels, [8, 32, 64])
        let net = NFKMLXRVMNet(configuration)
        net.train(false)
        XCTAssertTrue(net.backbone is NFKMLXResNetBackbone)
        let matte = net.matte(MLXRandom.uniform(low: 0, high: 1, [64, 64, 3]))
        eval(matte)
        XCTAssertEqual(matte.shape, [64, 64, 4])
        XCTAssertEqual(NFKMLXResNetBackbone.remapReferenceKey("backbone.layer1.0.downsample.0.weight"),
                       "backbone.layer1.0.downsample_conv.weight")
    }

    // The real-world tail and the three-convolution residual connection run at the tiny geometry, and
    // the reference's Sequential names land on the module's.
    func testSwinIRRealWorldTailRunsAtTheTinyGeometry() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXSwinIRConfiguration.tiny
        configuration.scale = 4
        configuration.upsampler = .nearestConv
        configuration.residualConnection = .threeConv
        let net = try NFKMLXSwinIR.makeNet(configuration)
        let keys = Set(net.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("layers.0.conv3.0.weight") && keys.contains("layers.0.conv3.4.weight"))
        XCTAssertTrue(keys.contains("conv_after_body_3conv.2.weight") && keys.contains("conv_up2.weight") && keys.contains("conv_hr.weight"))
        XCTAssertFalse(keys.contains("conv_after_body.weight") || keys.contains { $0.hasPrefix("upsample.") })
        let upscaled = net.upscale(MLXRandom.uniform(low: 0, high: 1, [16, 16, 3]))
        eval(upscaled)
        XCTAssertEqual(upscaled.shape, [64, 64, 3])
        XCTAssertEqual(NFKMLXSwinIR.remapReferenceKey("layers.2.conv.4.bias"), "layers.2.conv3.4.bias")
        XCTAssertEqual(NFKMLXSwinIR.remapReferenceKey("conv_after_body.2.weight"), "conv_after_body_3conv.2.weight")
        XCTAssertEqual(NFKMLXSwinIR.remapReferenceKey("layers.2.conv.weight"), "layers.2.conv.weight")
        XCTAssertEqual(NFKMLXSwinIR.remapReferenceKey("conv_before_upsample.0.weight"), "conv_before_upsample.weight")
    }

    // Qwen3.8-Flash-Next is the released Qwen4-Exp: 180B of bfloat16 across 131 shards, 360 GB, so
    // the decoder is held to the release by shape and its arithmetic rests on the oracle's size. The
    // n-gram table is the one parameter with no single counterpart — the release splits it into
    // `split_ngram_parts` shards so the saved layout matches the trained one — so the shards are
    // named as dropped and checked to sum to the one table this module looks up.
    func testQwen4ExpMatchesTheReleasedShapes() throws {
        try requireMLXRuntime()
        let release = try shapes("qwen3.8-flash-next")
        let configuration = try NFKMLXQwen4Exp.configuration(fromHuggingFace: release.config)
        XCTAssertEqual(configuration.layerCount, 48)
        XCTAssertEqual(configuration.expertCount, 512)
        XCTAssertEqual(configuration.hyperConnectionCount, 4)
        XCTAssertEqual(configuration.pleLayerIDs, [2])
        // Every `full_attention` entry the release states is a layer that carries an indexer.
        XCTAssertEqual(configuration.layerTypes.filter { $0 == .sparseAttention }.count, 12)

        let net = NFKMLXQwen4Exp.makeNet(configuration)
        let table = try XCTUnwrap(net.model.languageModel.layers[1].ple).embedding.table.weight
        assertStructure(
            "qwen3.8-flash-next",
            built: inventory(net,
                             rename: { $0.hasSuffix(".ngram_embedding.weight") ? nil : $0 },
                             reshape: { name, shape in
                                 name.hasSuffix("conv1d.weight") && shape.count == 3
                                     ? [shape[0], shape[2], shape[1]] : shape
                             }),
            released: release.shapes,
            named: { NFKMLXQwen4Exp.isDropped(key: $0) || $0.contains(".ngram_embedding.shard_") })

        let sharded = release.shapes.filter { $0.key.contains(".ngram_embedding.shard_") }
        XCTAssertEqual(sharded.count, 128, "the release splits the table into its configured parts")
        XCTAssertEqual(sharded.values.reduce(0) { $0 + $1[0] }, table.dim(0),
                       "the shards' rows sum to the table this module derives")
        XCTAssertEqual(Set(sharded.values.map { $0[1] }), [table.dim(1)],
                       "every shard is one n-gram head's width")
        print("VALIDATION structure qwen3.8-flash-next: n-gram table \(table.dim(0)) rows "
              + "across \(sharded.count) shards")
    }

    // The converse of the shape check: what the port leaves out is named rather than overlooked.
    func testQwen4ExpNamesWhatItDoesNotImplement() throws {
        let release = try shapes("qwen3.8-flash-next")
        let decoder = release.shapes.keys.filter {
            ($0.hasPrefix("model.language_model.") || $0 == "lm_head.weight")
                && !NFKMLXQwen4Exp.isDropped(key: $0)
        }
        let vision = release.shapes.keys.filter { $0.hasPrefix("model.visual.") }
        let prediction = release.shapes.keys.filter { $0.hasPrefix("mtp.") }
        let derived = release.shapes.keys.filter {
            $0.hasPrefix("model.language_model.") && NFKMLXQwen4Exp.isDropped(key: $0)
        }
        // The decoder's count includes the n-gram table's 128 shards, which the shape check names.
        print("VALIDATION structure qwen3.8-flash-next parts: decoder \(decoder.count), "
              + "vision tower \(vision.count), multi-token head \(prediction.count), "
              + "derived hash tables \(derived.count)")
        XCTAssertEqual(vision.count, 333, "the release is multimodal")
        XCTAssertEqual(prediction.count, 31, "and carries a multi-token-prediction head")
        XCTAssertEqual(derived.count, 3, "the hash buffers come from the configuration, not the file")
        XCTAssertEqual(decoder.count + vision.count + prediction.count + derived.count,
                       release.shapes.count,
                       "every tensor is either the decoder's or a named unimplemented part")
    }
}
