//
//  NFKMLXFlorence2Tests.swift
//  InferKitMLXTests
//
//  Florence-2 (microsoft/Florence-2-large, MIT). The DaViT vision tower and the multi-modal projector
//  evaluate MLX arrays, so these skip without a Metal library for MLX (see Tools/mlx-metallib.sh). The
//  parity test is gated on the released weights + the recorded oracle (`IK_VAL_FLORENCE2` model.safetensors
//  + `IK_PARITY_FLORENCE2` record, from `run_reference.py florence2`), compared seam by seam.
//

import XCTest
import InferKit
import CoreGraphics
import ImageIO
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXFlorence2Tests: XCTestCase {

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

    private func visionNet() -> NFKMLXFlorence2VisionNet { NFKMLXFlorence2VisionNet(.large) }
    private func projector() -> NFKMLXFlorence2Projector {
        let c = NFKMLXFlorence2VisionConfiguration.large
        return NFKMLXFlorence2Projector(embedDim: c.embedDim.last!, projectionDim: c.projectionDim,
                                        maxPositionEmbeddings: c.maxPositionEmbeddings)
    }

    /// The module keys the loader targets, including the nested `blocks.N.M` layout and the depthwise
    /// convolution's `.conv` wrapper.
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let vNames = Set(visionNet().parameters().flattened().map(\.0))
        for expected in ["convs.0.proj.weight", "convs.0.norm.weight",
                         "blocks.0.0.spatial.conv1.conv.weight", "blocks.0.0.spatial.norm1.weight",
                         "blocks.0.0.spatial.attn.qkv.weight", "blocks.0.0.spatial.ffn.fc1.weight",
                         "blocks.0.0.channel.attn.qkv.weight", "blocks.2.0.channel.norm2.weight"] {
            XCTAssertTrue(vNames.contains(expected), "vision missing \(expected)")
        }
        let pNames = Set(projector().parameters().flattened().map(\.0))
        for expected in ["row_embeddings.weight", "column_embeddings.weight",
                         "image_projection", "image_proj_norm.weight"] {
            XCTAssertTrue(pNames.contains(expected), "projector missing \(expected)")
        }
    }

    /// PARITY: the released DaViT tower and projector against the recorded oracle, seam by seam
    /// (stage-0 patch embed, stage-0 block, the unpooled vision output, and the projected tokens).
    func testSeamParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let weightsPath = env["IK_VAL_FLORENCE2"], let recordPath = env["IK_PARITY_FLORENCE2"] else {
            throw XCTSkip("set IK_VAL_FLORENCE2 (model.safetensors) and IK_PARITY_FLORENCE2 (oracle record)")
        }
        let vision = visionNet()
        let proj = projector()
        try NFKMLXFlorence2Weights.load(vision: vision, projector: proj, from: URL(fileURLWithPath: weightsPath))
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        let pixels = rec["pixels"]!                                                 // [S, S, 3]
        let side = pixels.dim(0)
        let x = pixels.reshaped([1, side, side, 3]).asType(.float32)

        func report(_ name: String, _ mine: MLXArray, _ reference: MLXArray) {
            XCTAssertGreaterThan(cosine(mine, reference), 0.999, "\(name) diverges")
        }

        let conv0 = vision.convs[0](x)
        report("conv0", conv0.reshaped([-1, conv0.dim(3)]), rec["conv0"]!)

        let block0 = vision.blocks[0][0](conv0)
        report("block0", block0.reshaped([-1, block0.dim(3)]), rec["block0"]!)

        let visionOut = vision(x)
        report("vision", visionOut.reshaped([-1, visionOut.dim(3)]), rec["vision"]!)

        let projected = proj(visionOut)
        report("proj", projected[0], rec["proj"]!)
    }

    /// PARITY: the BART fusion — the projected image tokens concatenated before the prompt embeddings,
    /// the encoder over the joined sequence, and the first decode step's logits (with final_logits_bias).
    func testFusionParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let weightsPath = env["IK_VAL_FLORENCE2"], let recordPath = env["IK_PARITY_FLORENCE2"] else {
            throw XCTSkip("set IK_VAL_FLORENCE2 (model.safetensors) and IK_PARITY_FLORENCE2 (oracle record)")
        }
        let net = NFKMLXFlorence2Net()
        try net.loadWeights(from: URL(fileURLWithPath: weightsPath))
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let pixels = rec["pixels"]!
        let side = pixels.dim(0)
        let x = pixels.reshaped([1, side, side, 3]).asType(.float32)
        let inputIds = rec["input_ids"]!.reshaped([1, -1]).asType(.int32)

        let memory = net.encode(pixels: x, inputIds: inputIds)
        XCTAssertGreaterThan(cosine(memory[0], rec["enc_last"]!), 0.999, "encoder output diverges")

        let cache = net.language.makeCache()
        let start = MLXArray([Int32(net.textConfig.decoderStartTokenId)]).reshaped([1, 1])
        let logits = net.language.decode(start, memory: memory, cache: cache)
        XCTAssertGreaterThan(cosine(logits[0, 0], rec["logits"]!), 0.999, "first-step logits diverge")
    }

    /// PARITY: greedy generation from the fused memory reproduces the reference's greedy tokens (the
    /// generation loop, on the recorded pixels + input_ids so the resize difference is out of the picture).
    func testGeneratedTokensMatchTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let weightsPath = env["IK_VAL_FLORENCE2"], let recordPath = env["IK_PARITY_FLORENCE2"] else {
            throw XCTSkip("set IK_VAL_FLORENCE2 (model.safetensors) and IK_PARITY_FLORENCE2 (oracle record)")
        }
        let net = NFKMLXFlorence2Net()
        try net.loadWeights(from: URL(fileURLWithPath: weightsPath))
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        guard let refGen = rec["generated"] else {
            throw XCTSkip("record has no `generated`; re-run run_reference.py florence2")
        }
        let pixels = rec["pixels"]!
        let side = pixels.dim(0)
        let x = pixels.reshaped([1, side, side, 3]).asType(.float32)
        let inputIds = rec["input_ids"]!.reshaped([1, -1]).asType(.int32)

        let memory = net.encode(pixels: x, inputIds: inputIds)
        let model = NFKFlorence2DecodeModel(net: net, memory: memory)
        let decoding = NFKMLXSeq2SeqDecoding(beams: 1, maxTokens: 64,
                                            startToken: net.textConfig.decoderStartTokenId,
                                            endToken: net.textConfig.eosTokenId)
        let mine = NFKMLXSeq2SeqDecoder.generate(model, source: [net.textConfig.decoderStartTokenId], decoding: decoding)

        func stripEnds(_ ids: [Int]) -> [Int] {
            var out = ids
            if out.first == net.textConfig.decoderStartTokenId { out.removeFirst() }
            while out.last == net.textConfig.eosTokenId { out.removeLast() }
            return out
        }
        let reference = stripEnds(refGen.asArray(Int32.self).map { Int($0) })
        let ours = stripEnds(mine)
        let count = min(reference.count, ours.count)
        XCTAssertGreaterThan(count, 0, "no tokens generated")
        let agree = (0 ..< count).filter { reference[$0] == ours[$0] }.count
        XCTAssertGreaterThan(Double(agree) / Double(count), 0.9, "greedy tokens diverge from the reference")
    }

    /// The byte-level BPE tokenizer plus the BART `<s>…</s>` wrapping reproduce the reference input_ids
    /// for the recorded task (the oracle used `<OD>`), token for token.
    func testTokenizerMatchesTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let weightsPath = env["IK_VAL_FLORENCE2"], let recordPath = env["IK_PARITY_FLORENCE2"] else {
            throw XCTSkip("set IK_VAL_FLORENCE2 (model.safetensors) and IK_PARITY_FLORENCE2 (oracle record)")
        }
        let directory = URL(fileURLWithPath: weightsPath).deletingLastPathComponent()
        guard let tokenizer = NFKMLXFlorence2Processor.tokenizer(inDirectory: directory) else {
            return XCTFail("could not build the Florence-2 tokenizer")
        }
        let reference = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays["input_ids"]!
        let prompt = NFKMLXFlorence2Processor.expandPrompt("<OD>")
        let ids = NFKMLXFlorence2Processor.encodePrompt(prompt, tokenizer: tokenizer, eosTokenId: 2)
        XCTAssertEqual(ids.asArray(Int32.self), reference.asArray(Int32.self),
                       "tokenizer + BART wrapping must match the reference input_ids")
    }

    /// The backend detects objects in a real image end to end (image processor + tokenizer + generation +
    /// location-token box parsing), returning `NFKDetection`s. The output is not token-checked because the
    /// resize differs from the reference's (the network is at parity on the reference's own pixels).
    func testTheBackendDetectsObjectsInAnImage() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let weightsPath = env["IK_VAL_FLORENCE2"] else {
            throw XCTSkip("set IK_VAL_FLORENCE2 (model.safetensors)")
        }
        let directory = URL(fileURLWithPath: weightsPath).deletingLastPathComponent()
        // The oracle records this exact photo (run_reference.py florence2 --image .../face960.jpg), so the
        // image-processor parity below compares the same image on both sides.
        let imagePath = "\(NSHomeDirectory())/.inferkit-validation/inputs/face960.jpg"
        guard let image = Self.loadImage(imagePath) else { throw XCTSkip("missing inputs/face960.jpg") }

        // The image processor matches the reference's preprocessed pixels (the bridge + bilinear resize
        // differs only slightly from the reference's bicubic).
        if let recordPath = env["IK_PARITY_FLORENCE2"] {
            let ref = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays["pixels"]!
            let mine = try NFKMLXFlorence2Processor.pixelValues(image)[0]
            XCTAssertGreaterThan(cosine(mine, ref), 0.95, "the image processor diverges from the reference")
        }

        // End to end on the detection task: the backend runs the image processor, tokenizer, generation
        // under the release's settings, and the <loc_> box post-processing.
        let backend = try NFKMLXFlorence2.backend(directoryURL: directory)
        let request = NFKInferenceRequest(inputs: [NFKInputImage: image, NFKInputPrompt: "<OD>"])
        let result = try backend.runInference(for: request)
        let text = (result.output(forKey: NFKOutputText) as? String) ?? ""
        XCTAssertFalse(text.isEmpty, "the detection text is non-empty")
        let detections = (result.output(forKey: NFKOutputDetections) as? [NFKDetection]) ?? []
        XCTAssertFalse(detections.isEmpty, "at least one object is detected")
    }

    /// Florence-2-base reads its geometry from `config.json`: half the DaViT widths, a 768-wide
    /// projection, and BART-base. The projector output, the encoder, the first-step logits, and greedy
    /// generation match the base release's record, and the directory factory builds it.
    func testTheBaseReleaseBuildsFromItsConfigurationAtParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let root = env["IK_VAL_FLORENCE2_BASE"], let recordPath = env["IK_PARITY_FLORENCE2_BASE"] else {
            throw XCTSkip("set IK_VAL_FLORENCE2_BASE (release directory) and IK_PARITY_FLORENCE2_BASE (oracle record)")
        }
        let directory = URL(fileURLWithPath: root)
        let geometry = try NFKMLXFlorence2Net.configuration(fromConfigURL: directory.appendingPathComponent("config.json"))
        XCTAssertEqual(geometry.vision.embedDim, NFKMLXFlorence2VisionConfiguration.base.embedDim)
        XCTAssertEqual(geometry.vision.projectionDim, 768)
        XCTAssertEqual(geometry.text.dModel, NFKMLXFlorence2Net.bartBase.dModel)
        XCTAssertEqual(geometry.text.encoderLayers, 6)
        XCTAssertEqual(geometry.text.maxPositions, 1024)

        try checkSeams(name: "florence-2-base", directory: directory, recordPath: recordPath)
        XCTAssertTrue(try NFKMLXFlorence2.backend(directoryURL: directory).isReady)
    }

    /// The fine-tuned releases (`Florence-2-base-ft`, `-large-ft`) share their bases' geometry; the
    /// projector, encoder, first-step logits, and greedy generation match their own records.
    func testTheFineTunedReleasesAreAtParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        var measured = 0
        for (name, key) in [("florence-2-base-ft", "FLORENCE2_BASE_FT"), ("florence-2-large-ft", "FLORENCE2_LARGE_FT")] {
            guard let root = env["IK_VAL_\(key)"], let recordPath = env["IK_PARITY_\(key)"],
                  FileManager.default.fileExists(atPath: recordPath) else { continue }
            try checkSeams(name: name, directory: URL(fileURLWithPath: root), recordPath: recordPath)
            measured += 1
        }
        try XCTSkipIf(measured == 0, "set IK_VAL_FLORENCE2_{BASE,LARGE}_FT and IK_PARITY_FLORENCE2_{BASE,LARGE}_FT")
    }

    /// The projector, encoder, first-step logits, and raw greedy generation of one release against its
    /// `run_reference.py florence2` record.
    private func checkSeams(name: String, directory: URL, recordPath: String) throws {
        let geometry = try NFKMLXFlorence2Net.configuration(fromConfigURL: directory.appendingPathComponent("config.json"))
        let net = NFKMLXFlorence2Net(vision: geometry.vision, text: geometry.text)
        try net.loadWeights(from: directory.appendingPathComponent("model.safetensors"))
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let pixels = rec["pixels"]!
        let side = pixels.dim(0)
        let x = pixels.reshaped([1, side, side, 3]).asType(.float32)
        let inputIds = rec["input_ids"]!.reshaped([1, -1]).asType(.int32)

        let projected = net.imageFeatures(x)
        let memory = net.encode(pixels: x, inputIds: inputIds)
        let cache = net.language.makeCache()
        let start = MLXArray([Int32(net.textConfig.decoderStartTokenId)]).reshaped([1, 1])
        let logits = net.language.decode(start, memory: memory, cache: cache)
        let seams = [("proj", cosine(projected[0], rec["proj"]!)), ("encoder", cosine(memory[0], rec["enc_last"]!)),
                     ("logits", cosine(logits[0, 0], rec["logits"]!))]
        print("VALIDATION PARITY \(name): " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        for (seam, similarity) in seams { XCTAssertGreaterThan(similarity, 0.999, "\(name) \(seam) diverges") }

        let decoding = NFKMLXSeq2SeqDecoding(beams: 1, maxTokens: 64, startToken: net.textConfig.decoderStartTokenId,
                                            endToken: net.textConfig.eosTokenId)
        let mine = NFKMLXSeq2SeqDecoder.generate(NFKFlorence2DecodeModel(net: net, memory: memory),
                                                 source: [net.textConfig.decoderStartTokenId], decoding: decoding)
        let ends = Set([net.textConfig.decoderStartTokenId, net.textConfig.eosTokenId])
        XCTAssertEqual(mine.filter { !ends.contains($0) },
                       rec["generated"]!.asArray(Int32.self).map(Int.init).filter { !ends.contains($0) },
                       "\(name) greedy generation is token-exact")
    }

    /// Generation under the release's settings (three beams, early stopping, no repeated 3-gram, `<s>`
    /// forced first and `</s>` at the length limit) is token-exact against Florence-2's own `generate`
    /// on every recorded task of every release, from the reference's preprocessed pixels.
    func testGenerationUnderTheReleaseSettingsMatchesTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        let releases = [("florence-2-base", env["IK_VAL_FLORENCE2_BASE"], env["IK_PARITY_FLORENCE2_BASE_GENERATE"]),
                        ("florence-2-large", env["IK_VAL_FLORENCE2"].map { URL(fileURLWithPath: $0).deletingLastPathComponent().path },
                         env["IK_PARITY_FLORENCE2_GENERATE"]),
                        ("florence-2-base-ft", env["IK_VAL_FLORENCE2_BASE_FT"], env["IK_PARITY_FLORENCE2_BASE_FT_GENERATE"]),
                        ("florence-2-large-ft", env["IK_VAL_FLORENCE2_LARGE_FT"], env["IK_PARITY_FLORENCE2_LARGE_FT_GENERATE"])]
        var measured = 0
        for (name, root, recordPath) in releases {
            guard let root, let recordPath, FileManager.default.fileExists(atPath: recordPath) else { continue }
            let directory = URL(fileURLWithPath: root)
            let net = try Self.net(inDirectory: directory)
            let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
            let side = rec["pixels"]!.dim(0)
            let pixels = rec["pixels"]!.reshaped([1, side, side, 3]).asType(.float32)
            func run(_ inputIds: MLXArray, _ decoding: NFKMLXSeq2SeqDecoding) -> [Int] {
                let memory = net.encode(pixels: pixels, inputIds: inputIds.reshaped([1, -1]).asType(.int32))
                return NFKMLXSeq2SeqDecoder.generate(NFKFlorence2DecodeModel(net: net, memory: memory),
                                                     source: [net.textConfig.decoderStartTokenId], decoding: decoding)
            }
            func reference(_ key: String) -> [Int] { Array(rec[key]!.asArray(Int32.self).map(Int.init).dropFirst().dropLast()) }

            let defaults = try NFKMLXFlorence2.generationDefaults(fromConfigURL: directory.appendingPathComponent("config.json"),
                                                                  text: net.textConfig)
            XCTAssertEqual(defaults.beams, 3)
            XCTAssertEqual(defaults.noRepeatNgramSize, 3)
            XCTAssertEqual(defaults.forcedFirstToken, 0)
            XCTAssertEqual(defaults.forcedLastToken, 2)
            XCTAssertTrue(defaults.earlyStopping)
            var exact = 0, tasks = 0
            while let inputIds = rec["input_ids_\(tasks)"] {
                let mine = run(inputIds, defaults)
                XCTAssertEqual(mine, reference("generated_\(tasks)"), "\(name) task \(tasks) is token-exact")
                exact += mine == reference("generated_\(tasks)") ? 1 : 0
                tasks += 1
            }
            let detailed = rec["input_ids_2"]!, caption = rec["input_ids_0"]!
            var truncated = defaults
            truncated.maxTokens = 16
            XCTAssertEqual(run(detailed, truncated), reference("generated_truncated"), "\(name) ends at the forced </s>")
            var greedy = defaults
            greedy.beams = 1
            XCTAssertEqual(run(caption, greedy), reference("generated_greedy"), "\(name) greedy under the constraints")
            print("VALIDATION PARITY \(name) generation: \(exact)/\(tasks) tasks token-exact")
            measured += 1
        }
        try XCTSkipIf(measured == 0, "set IK_VAL_FLORENCE2[_BASE] and IK_PARITY_FLORENCE2[_BASE]_GENERATE (run_reference.py florence2_generate)")
    }

    private static func net(inDirectory directory: URL) throws -> NFKMLXFlorence2Net {
        let geometry = try NFKMLXFlorence2Net.configuration(fromConfigURL: directory.appendingPathComponent("config.json"))
        let net = NFKMLXFlorence2Net(vision: geometry.vision, text: geometry.text)
        try net.loadWeights(from: directory.appendingPathComponent("model.safetensors"))
        return net
    }

    static func loadImage(_ path: String) -> CGImage? {
        guard FileManager.default.fileExists(atPath: path),
              let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}

/// Florence-2's fine-tuning recipe: the objective against the release's own `labels=` loss
/// (`run_reference.py florence2_loss`, `IK_PARITY_FLORENCE2_BASE_FT_LOSS`), a LoRA run, and the merged
/// network's save and factory reload.
final class NFKMLXFlorence2TrainingTests: XCTestCase {

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func release() throws -> (directory: URL, record: [String: MLXArray]) {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_FLORENCE2_BASE_FT"], let path = env["IK_PARITY_FLORENCE2_BASE_FT_LOSS"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_FLORENCE2_BASE_FT and IK_PARITY_FLORENCE2_BASE_FT_LOSS (run_reference.py florence2_loss)")
        }
        return (URL(fileURLWithPath: directory), try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays)
    }

    /// PARITY: on the release's own caption, the loss on its logits and the port's forward equal the
    /// release's `labels=` loss, held to its float64 value (float32's log-sum-exp over the 51k vocabulary
    /// rounds to about 5e-5).
    func testTheObjectiveIsTheReleasesLabelsLoss() throws {
        let (directory, record) = try release()
        let objective = NFKMLXFlorence2Objective()
        let answer = record["answer_ids"]!
        let reference = record["loss_f64"]!.item(Float.self)
        let onLogits = objective.loss(logits: record["logits"]!.expandedDimensions(axis: 0), target: answer).item(Float.self)
        let net = try NFKMLXFlorence2.network(directoryURL: directory)
        let forward = objective(net, pixels: record["pixels"]!.expandedDimensions(axis: 0), prompt: record["prompt_ids"]!,
                                answer: answer).item(Float.self)
        print("VALIDATION PARITY florence2 BASE_FT loss: float64 \(reference), float32 \(record["output"]!.item(Float.self)), "
              + "on its logits \(onLogits), port forward \(forward)")
        XCTAssertEqual(onLogits, reference, accuracy: 1e-6)
        XCTAssertEqual(forward, reference, accuracy: 1e-4)
    }

    /// A LoRA run adapts only the decoder's query and value adapters and lowers the loss on its example;
    /// merged and saved, it reloads through the factory with the same logits.
    func testALoRARunMergesAndReloadsThroughTheFactory() throws {
        let (directory, record) = try release()
        let net = try NFKMLXFlorence2.network(directoryURL: directory)
        let pixels = record["pixels"]!.expandedDimensions(axis: 0), prompt = record["prompt_ids"]!
        let answer = MLXArray([Int32(0), 250, 7, 2])                                      // "<s>The.</s>"
        let losses = try NFKMLXFlorence2.fineTune(net, examples: { _ in (pixels, prompt, answer) },
                                                  optimizer: NFKMLXReferenceOptimizers.adamW(learningRate: 1e-3, weightDecay: 0),
                                                  steps: 6)
        XCTAssertLessThan(losses.last!, losses.first!, "the loss falls")
        let trainable = net.trainableParameters().flattened().map(\.0)
        XCTAssertFalse(trainable.isEmpty)
        XCTAssertTrue(trainable.allSatisfy { $0.hasPrefix("language_model.decoder.layers.") && $0.contains("lora") },
                      "only the decoder adapters train")

        XCTAssertGreaterThan(try NFKMLXLoRA.merge(into: net), 0)
        let saved = FileManager.default.temporaryDirectory.appendingPathComponent("florence2-tuned-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: saved) }
        try NFKMLXFlorence2.save(net, toDirectoryURL: saved, release: directory)
        let reloaded = try NFKMLXFlorence2.network(directoryURL: saved)
        let objective = NFKMLXFlorence2Objective()
        let before = objective(net, pixels: pixels, prompt: prompt, answer: answer).item(Float.self)
        let after = objective(reloaded, pixels: pixels, prompt: prompt, answer: answer).item(Float.self)
        XCTAssertEqual(after, before, accuracy: 1e-5, "the reloaded network scores the example the same")
        XCTAssertNoThrow(try NFKMLXFlorence2.backend(directoryURL: saved))
    }
}
