//
//  NFKMLXLTXVideoTrainingTests.swift
//  InferKitMLXTests
//
//  The LTX-Video LoRA objective against Lightricks' trainer (`run_reference.py ltxv_loss`, LTX-Video-Trainer at
//  e055182 over diffusers' transformer and peft's LoRA, through the trainer's own `_setup_lora`, `_training_step`,
//  and `_save_checkpoint`), the trainer's saved adapters held beside the transformer, and the fine-tune recipe on a
//  tiny transformer and on the released 0.9.0 transformer.
//

import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXLTXVideoTrainingTests: XCTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func record() throws -> [String: MLXArray] {
        guard let path = config["IK_PARITY_LTXV_LOSS"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_LTXV_LOSS (run_reference.py ltxv_loss)")
        }
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    private static let grid = (frames: 3, height: 2, width: 3)

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Double {
        let a = mine.asType(.float32).reshaped([-1]).asArray(Float.self)
        let b = reference.asType(.float32).reshaped([-1]).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) {
            dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y)
        }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    private func maxDifference(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        (mine.asType(.float32) - reference.asType(.float32)).abs().max().item(Float.self)
    }

    private func required(_ arrays: [String: MLXArray], _ key: String) throws -> MLXArray {
        try XCTUnwrap(arrays[key], "the record has no \(key)")
    }

    /// The record's transformer, without adapters.
    private func baseNet(_ arrays: [String: MLXArray]) throws -> NFKMLXLTXTransformerNet {
        let net = NFKMLXLTXTransformerNet(.tiny)
        try NFKMLXWeights.apply(arrays.compactMap { key, value in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value) : nil
        }, to: net, verifyShapes: true)
        return net
    }

    private func largestDifference(_ a: [String: MLXArray], _ b: [String: MLXArray]) throws -> Float {
        XCTAssertEqual(Set(a.keys), Set(b.keys))
        return try a.reduce(Float(0)) { worst, entry in
            max(worst, maxDifference(try XCTUnwrap(b[entry.key]), entry.value))
        }
    }

    // The sampler's arithmetic from the reference's own draws, either side of the shift's interpolation.
    func testLTXVideoSigmaSamplerMatchesTheTrainer() throws {
        let arrays = try record()
        let sampler = NFKMLXLTXVideoSigmaSampler()
        for tokens in [100, 6000] {
            let mine = sampler.sigmas(normal: try required(arrays, "sampler.\(tokens).normal"), tokens: tokens)
            let difference = maxDifference(mine, try required(arrays, "sampler.\(tokens).sigma"))
            print("VALIDATION PARITY ltxv-sampler-\(tokens): max sigma difference \(difference)")
            XCTAssertLessThan(difference, 1e-6, "\(tokens) tokens")
        }
    }

    // Both sections end to end: the sigmas, the noised latents, the targets, the integer per-token timesteps and
    // the loss mask, the prediction over the padded caption, the loss, and every adapter gradient. The control
    // reads one timestep per clip, which the first-frame section must tell apart from the per-token timesteps.
    func testLTXVideoObjectiveMatchesTheTrainer() throws {
        let arrays = try record()
        let net = try baseNet(arrays)
        let adapted = try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 8) { path, _ in NFKMLXLTXVideoTrainable.targets(path) }
        let adapters = Dictionary(uniqueKeysWithValues: arrays.compactMap { key, value in
            key.hasPrefix("lora::") ? (String(key.dropFirst(6)), value) : nil
        })
        XCTAssertEqual(try NFKMLXLoRA.loadAdapters(adapters, into: net), adapted)
        XCTAssertEqual(adapted, 2 * 2 * 4, "four projections of two attentions in each of two blocks")

        let clean = try required(arrays, "clean"), text = try required(arrays, "text")
        let textMask = try required(arrays, "text_mask")
        let (frames, height, width) = Self.grid
        for prefix in ["plain", "first_frame"] {
            let sigma = try required(arrays, "\(prefix).sigma")
            let sampled = NFKMLXLTXVideoSigmaSampler().sigmas(normal: try required(arrays, "\(prefix).draw.normal"),
                                                              tokens: clean.dim(1))
            XCTAssertLessThan(maxDifference(sampled, sigma), 1e-6, "\(prefix) sigma")
            let conditioned = MLX.broadcast(try required(arrays, "\(prefix).conditioned"), to: [clean.dim(0)])
            let noise = try required(arrays, "\(prefix).draw.noise")
            let inputs = NFKMLXLTXVideoObjective.inputs(latents: clean, sigma: sigma, noise: noise,
                                                        conditioned: conditioned, tokensPerFrame: height * width)
            for (label, mine, key) in [("noisy", inputs.latents, "noisy"), ("targets", inputs.targets, "targets"),
                                       ("timesteps", inputs.timesteps, "timesteps"),
                                       ("loss mask", 1 - inputs.lossMask, "conditioning_mask")] {
                let reference = try required(arrays, "\(prefix).\(key)")
                XCTAssertEqual(mine.shape, reference.shape, "\(prefix) \(label) shape")
                XCTAssertLessThan(maxDifference(mine, reference), 1e-6, "\(prefix) \(label)")
            }

            let frameRate = try required(arrays, "\(prefix).fps").item(Float.self)
            let prediction = NFKMLXLTXVideoObjective.prediction(net, inputs: inputs, text: text, textMask: textMask,
                                                                frames: frames, height: height, width: width,
                                                                frameRate: frameRate)
            let similarity = cosine(prediction, try required(arrays, "\(prefix).prediction"))
            print("SEAM ltxv-loss \(prefix) prediction: cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.99999)
            if prefix == "first_frame" {
                var perClip = inputs
                perClip.timesteps = inputs.timesteps[0..., (inputs.timesteps.dim(1) - 1)...]
                let control = NFKMLXLTXVideoObjective.prediction(net, inputs: perClip, text: text, textMask: textMask,
                                                                 frames: frames, height: height, width: width,
                                                                 frameRate: frameRate)
                let controlSimilarity = cosine(control, try required(arrays, "first_frame.prediction"))
                let controlParity = cosine(control, try required(arrays, "control.per_clip"))
                print("SEAM ltxv-loss control (one timestep per clip): cosine \(controlSimilarity) to the trainer's "
                      + "prediction, \(controlParity) to the reference's own per-clip prediction")
                XCTAssertLessThan(controlSimilarity, 0.9999999, "the record discriminates per-token timesteps")
                XCTAssertGreaterThan(controlParity, 0.99999)
            }

            let step = valueAndGrad(model: net) { (net: NFKMLXLTXTransformerNet, _: [MLXArray]) in
                [NFKMLXLTXVideoObjective.score(net, latents: clean, text: text, textMask: textMask, sigma: sigma,
                                               noise: noise, conditioned: conditioned, frames: frames, height: height,
                                               width: width, frameRate: frameRate)]
            }
            let (values, gradients) = step(net, [])
            let loss = values[0].item(Float.self), referenceLoss = try required(arrays, "\(prefix).loss").item(Float.self)
            let flat = Dictionary(uniqueKeysWithValues: gradients.flattened())
            var worst = 1.0
            for path in adapters.keys.sorted() {
                let mine = try XCTUnwrap(flat[path], "no gradient for \(path)")
                let similarity = cosine(mine, try required(arrays, "\(prefix).grad::\(path)"))
                worst = min(worst, similarity)
                XCTAssertGreaterThan(similarity, 0.99999, "\(prefix) \(path) gradient")
            }
            print("VALIDATION PARITY ltxv-loss-\(prefix): loss \(loss) against \(referenceLoss), "
                  + "worst adapter-gradient cosine \(worst) over \(adapters.count) arrays")
            XCTAssertLessThan(abs(loss - referenceLoss) / referenceLoss, 1e-5, "\(prefix) loss")
        }
    }

    // The file the trainer's `_save_checkpoint` wrote, under its own keys, held beside the base through the public
    // reader the backend's adapter factory uses: the prediction is the trainer's adapted one. The recipe's own
    // file, saved from the same adapters, holds the trainer's keys and values.
    func testTheTrainersSavedAdaptersHoldBesideTheTransformer() throws {
        let arrays = try record()
        let saved = Dictionary(uniqueKeysWithValues: arrays.compactMap { key, value in
            key.hasPrefix("saved::") ? (String(key.dropFirst(7)), value) : nil
        })
        XCTAssertEqual(saved.count, 2 * 2 * 2 * 4)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("ltxv-saved-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let trainers = directory.appendingPathComponent("lora_weights_step_00001.safetensors")
        try MLX.save(arrays: saved, metadata: ["format": "pt"], url: trainers)

        let clean = try required(arrays, "clean"), text = try required(arrays, "text")
        let textMask = try required(arrays, "text_mask")
        let (frames, height, width) = Self.grid
        let inputs = NFKMLXLTXVideoObjective.inputs(
            latents: clean, sigma: try required(arrays, "plain.sigma"), noise: try required(arrays, "plain.draw.noise"),
            conditioned: MLXArray.zeros([clean.dim(0)]), tokensPerFrame: height * width)
        let held = try baseNet(arrays)
        try NFKMLXLTXTransformer.holdAdapters(at: [trainers], strengths: [1], in: held)
        let prediction = NFKMLXLTXVideoObjective.prediction(held, inputs: inputs, text: text, textMask: textMask,
                                                            frames: frames, height: height, width: width, frameRate: 24)
        // The trainer's file carries no alpha, which diffusers and the reader take as alpha = rank; the record's
        // adapters train at alpha 8 over rank 4, so the held prediction is the trainer's at half the detour.
        let net = try baseNet(arrays)
        try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 4) { path, _ in NFKMLXLTXVideoTrainable.targets(path) }
        try NFKMLXLoRA.loadAdapters(Dictionary(uniqueKeysWithValues: arrays.compactMap { key, value in
            key.hasPrefix("lora::") ? (String(key.dropFirst(6)), value) : nil
        }), into: net)
        let unitScale = NFKMLXLTXVideoObjective.prediction(net, inputs: inputs, text: text, textMask: textMask,
                                                           frames: frames, height: height, width: width, frameRate: 24)
        let similarity = cosine(prediction, unitScale)
        print("SEAM ltxv-saved held trainer file: cosine \(similarity) to the alpha-equals-rank prediction")
        XCTAssertGreaterThan(similarity, 0.9999999)

        let trained = try baseNet(arrays)
        try NFKMLXLoRA.apply(to: trained, rank: 4, alpha: 8) { path, _ in NFKMLXLTXVideoTrainable.targets(path) }
        try NFKMLXLoRA.loadAdapters(Dictionary(uniqueKeysWithValues: arrays.compactMap { key, value in
            key.hasPrefix("lora::") ? (String(key.dropFirst(6)), value) : nil
        }), into: trained)
        let mine = directory.appendingPathComponent("recipe.safetensors")
        try NFKMLXLTXVideoGenerator.saveAdapters(of: trained, to: mine)
        let (written, metadata) = try MLX.loadArraysAndMetadata(url: mine)
        XCTAssertEqual(try largestDifference(written, saved), 0, "the recipe writes the trainer's keys and values")
        XCTAssertEqual(metadata["format"], "pt")
        let rehold = try baseNet(arrays)
        try NFKMLXLTXTransformer.holdAdapters(at: [mine], strengths: [1], in: rehold)
        let reference = try required(arrays, "plain.prediction")
        let reheld = NFKMLXLTXVideoObjective.prediction(rehold, inputs: inputs, text: text, textMask: textMask,
                                                        frames: frames, height: height, width: width, frameRate: 24)
        let reheldSimilarity = cosine(reheld, reference)
        print("SEAM ltxv-saved recipe file held: cosine \(reheldSimilarity) to the trainer's prediction")
        XCTAssertGreaterThan(reheldSimilarity, 0.99999, "the recipe's metadata carries alpha 8")
    }

    // The recipe on a tiny transformer: over one clip the loss falls, the adapters move, the base and the layers
    // outside the trainer's targets stay where they were.
    func testLTXVideoFineTuneLowersTheLossAndTrainsOnlyTheAdapters() throws {
        NFKMLXRandom.seed(3)
        let net = NFKMLXLTXTransformerNet(.tiny)
        eval(net)
        let example = NFKMLXLTXVideoTrainingExample(latents: MLXRandom.normal([18, 16]), text: MLXRandom.normal([5, 32]),
                                                    textMask: MLXArray([Float(1), 1, 1, 0, 0]), frames: 3, height: 2,
                                                    width: 3)
        let projection = net.captionProjection.linear1.weight
        let losses = try NFKMLXLTXVideoGenerator.fineTune(net, examples: { _ in [example, example] },
                                                          trainable: .lora(rank: 8, alpha: 8), learningRate: 3e-3,
                                                          steps: 60, seed: 2)
        let first = losses.prefix(10).reduce(0, +) / 10, last = losses.suffix(10).reduce(0, +) / 10
        print("ltxv fine-tune: mean loss \(first) over the first ten steps, \(last) over the last ten")
        XCTAssertLessThan(last, first * 0.9)
        XCTAssertTrue(net.blocks[0].attn1.toQ is NFKMLXLoRALinear)
        XCTAssertTrue(net.blocks[1].attn2.toOut[0] is NFKMLXLoRALinear)
        XCTAssertFalse(net.blocks[0].ff.net[2] is NFKMLXLoRALinear, "the feed-forward is not a target")
        XCTAssertFalse(net.projIn is NFKMLXLoRALinear, "projections outside the blocks are not targets")
        XCTAssertTrue(arrayEqual(net.captionProjection.linear1.weight, projection).item(Bool.self),
                      "a frozen layer stays put")
    }

    // The recipe's seed fixes the adapters' initialization: two runs of one network from different global random
    // states train the same adapters to the same losses.
    func testTheSeedFixesTheAdaptersWhateverTheGlobalRandomState() throws {
        let example = NFKMLXLTXVideoTrainingExample(
            latents: MLXRandom.normal([18, 16], key: MLXRandom.key(1)), text: MLXRandom.normal([5, 32], key: MLXRandom.key(2)),
            textMask: MLXArray([Float(1), 1, 1, 1, 0]), frames: 3, height: 2, width: 3)
        func run(globalSeed: UInt64) throws -> (losses: [Float], adapters: [String: MLXArray]) {
            NFKMLXRandom.seed(3)
            let net = NFKMLXLTXTransformerNet(.tiny)
            eval(net)
            NFKMLXRandom.seed(globalSeed)
            let losses = try NFKMLXLTXVideoGenerator.fineTune(net, examples: { _ in [example] }, trainable: .lora(rank: 4, alpha: 4),
                                                              objective: NFKMLXLTXVideoObjective(firstFrameConditioning: 0.5),
                                                              learningRate: 3e-3, steps: 3, seed: 1)
            let adapters = Dictionary(uniqueKeysWithValues: net.trainableParameters().flattened())
            eval(Array(adapters.values))
            return (losses, adapters)
        }
        let first = try run(globalSeed: 100), second = try run(globalSeed: 200)
        XCTAssertFalse(first.adapters.isEmpty)
        let gap = try largestDifference(first.adapters, second.adapters)
        print("ltxv fine-tune from two global random states: losses \(second.losses) against \(first.losses), "
              + "largest adapter difference \(gap) over \(first.adapters.count) arrays")
        XCTAssertEqual(second.losses, first.losses)
        XCTAssertEqual(gap, 0)
    }

    // Recomputing each block in the backward pass changes no number: the loss and every adapter's gradient equal
    // the plain backward's, over a float base and an 8-bit one, with the first frame conditioned.
    func testRecomputingBlocksInBackwardKeepsTheGradients() throws {
        for bits in [nil, 8] as [Int?] {
            NFKMLXRandom.seed(5)
            let net = NFKMLXLTXTransformerNet(.tiny)
            if let bits { try NFKMLXQuantization.quantize(module: net, bits: bits, groupSize: 32) }
            try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 4) { path, _ in NFKMLXLTXVideoTrainable.targets(path) }
            for (path, layer) in net.leafModules().flattened() {
                guard let adapter = layer as? NFKMLXLoRAAdapter else { continue }
                _ = adapter.update(parameters: ModuleParameters.unflattened(
                    ["lora_b": MLXRandom.normal(adapter.loraB.shape, key: MLXRandom.key(UInt64(path.count))) * 0.1]))
            }
            let latents = MLXRandom.normal([2, 18, 16]), text = MLXRandom.normal([2, 5, 32])
            let textMask = MLXArray([Float(1), 1, 1, 0, 0, 1, 1, 1, 1, 1]).reshaped([2, 5])
            let sigma = MLXArray([Float(0.3), 0.8]), noise = MLXRandom.normal([2, 18, 16])
            let conditioned = MLXArray([Float(1), 1])
            func gradients() -> (Float, [String: MLXArray]) {
                let step = valueAndGrad(model: net) { (net: NFKMLXLTXTransformerNet, _: [MLXArray]) in
                    [NFKMLXLTXVideoObjective.score(net, latents: latents, text: text, textMask: textMask, sigma: sigma,
                                                   noise: noise, conditioned: conditioned, frames: 3, height: 2,
                                                   width: 3, frameRate: 24)]
                }
                let (values, grads) = step(net, [])
                return (values[0].item(Float.self), Dictionary(uniqueKeysWithValues: grads.flattened()))
            }
            let base = bits.map { "\($0)-bit" } ?? "float"
            net.recomputesBlocksInBackward = false
            let (plainLoss, plain) = gradients()
            let (_, again) = gradients()
            net.recomputesBlocksInBackward = true
            let (recomputedLoss, recomputed) = gradients()
            let control = try largestDifference(plain, again), gap = try largestDifference(plain, recomputed)
            print("ltxv recomputed backward (\(base)): loss \(recomputedLoss) against \(plainLoss), largest gradient "
                  + "difference \(gap) over \(plain.count) arrays, plain against itself \(control)")
            XCTAssertFalse(plain.isEmpty)
            XCTAssertEqual(control, 0, "\(base): the plain backward is deterministic")
            XCTAssertEqual(recomputedLoss, plainLoss, "\(base) loss")
            XCTAssertEqual(gap, 0, "\(base): the recomputed backward is the plain one")
        }
    }

    // The data adapter on a tiny generator: the caption pads to the trainer's 256 tokens with its mask, the clip
    // encodes to its latent grid, and the posterior draw follows the seed.
    func testATrainingExampleIsTheTrainersPreprocessing() throws {
        guard let tokenizer = NFKMLXValidationConfig.environment["IK_VAL_T5_TOKENIZER"] else {
            throw XCTSkip("set IK_VAL_T5_TOKENIZER (a T5 SentencePiece tokenizer directory)")
        }
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: URL(fileURLWithPath: tokenizer).appendingPathComponent("spiece.model")))
        let t5 = NFKMLXT5Configuration(dModel: 32, layers: 2, heads: 2, keyDim: 16, ffDim: 64,
                                       vocabularySize: 32128, relativeBuckets: 16, relativeMaxDistance: 32)
        let vae = NFKMLXLTXVAEConfiguration(latentChannels: 16, blockOutChannels: [8, 16, 16, 16],
                                            layersPerBlock: [1, 1, 1, 1, 1])
        MLXRandom.seed(5)
        let generator = NFKMLXLTXVideoGenerator(
            resident: true, segmenter: segmenter, loadTextEncoder: { NFKMLXT5Encoder.makeNet(t5) },
            loadPipeline: {
                NFKMLXLTXPipeline(transformer: NFKMLXLTXTransformerNet(.tiny), vae: NFKMLXLTXVideoVAE.makeNet(vae))
            })
        try generator.loadResident()
        let clip = MLXRandom.uniform(low: 0, high: 1, [9, 64, 96, 3], key: MLXRandom.key(4))
        let example = try generator.trainingExample(video: clip, frameRate: 30, prompt: "a red fox", seed: 1)
        XCTAssertEqual([example.frames, example.height, example.width], [2, 2, 3])
        XCTAssertEqual(example.latents.shape, [12, 16])
        XCTAssertEqual(example.text.shape, [256, 32])
        XCTAssertEqual(example.textMask.shape, [256])
        let tokens = Int(example.textMask.sum().item(Float.self))
        XCTAssertEqual(tokens, NFKMLXT5Prompt("a red fox", segmenter: segmenter, length: 256).mask.reduce(0, +))
        XCTAssertEqual(example.frameRate, 30)
        let again = try generator.trainingExample(video: clip, frameRate: 30, prompt: "a red fox", seed: 1)
        let other = try generator.trainingExample(video: clip, frameRate: 30, prompt: "a red fox", seed: 2)
        XCTAssertEqual(maxDifference(again.latents, example.latents), 0)
        XCTAssertGreaterThan(maxDifference(other.latents, example.latents), 0)
        XCTAssertThrowsError(try generator.trainingExample(video: clip[..<8], frameRate: 30, prompt: "a red fox"))
    }

    // Three steps of the reference recipe on the released 0.9.0 transformer (2B) over a 49-frame 512×320 clip's
    // latent grid and 256 caption tokens, over the float32 base and a 4-bit one, with each build's memory, each
    // step's peak, and each step's time printed.
    func testLTXVideoLoRAStepsOnReleasedWeights() throws {
        try NFKMLXSupportLevels.skipIfRetired("ltx-video-0.9.0")
        guard let path = config["IK_VAL_LTX_TF"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_LTX_TF to LTX-Video 0.9.0's transformer directory")
        }
        // The builder reads a release directory; the store keeps the transformer folder alone.
        let release = FileManager.default.temporaryDirectory.appendingPathComponent("ltxv-release-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: release, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: release) }
        try FileManager.default.createSymbolicLink(at: release.appendingPathComponent("transformer"),
                                                   withDestinationURL: URL(fileURLWithPath: path))
        func gigabytes(_ bytes: Int) -> String { String(format: "%.2f", Double(bytes) / 1e9) }
        let (frames, height, width) = (7, 10, 16)
        let example = NFKMLXLTXVideoTrainingExample(
            latents: MLXRandom.normal([frames * height * width, 128], key: MLXRandom.key(1)),
            text: MLXRandom.normal([256, 4096], key: MLXRandom.key(2)),
            textMask: concatenated([MLXArray.ones([40]), MLXArray.zeros([216])]), frames: frames, height: height,
            width: width, frameRate: 24)
        for bits in [nil, 4] as [Int?] {
            let label = bits.map { "ltxv-qlora-\($0)bit" } ?? "ltxv-lora-float32"
            NFKMLXGPU.clearCache()
            NFKMLXGPU.resetPeakMemory()
            let buildStart = Date()
            let net = try NFKMLXLTXVideoGenerator.network(directoryURL: release, quantizationBits: bits)
            print("MEMORY \(label) build: \(String(format: "%.1f", Date().timeIntervalSince(buildStart))) s, active "
                  + "\(gigabytes(NFKMLXGPU.activeMemory)) GB, peak \(gigabytes(NFKMLXGPU.peakMemory)) GB")
            fflush(stdout)
            NFKMLXGPU.resetPeakMemory()
            var stepStart = Date()
            var stepSeconds = [Double](), stepPeaks = [Int]()
            let losses = try NFKMLXLTXVideoGenerator.fineTune(net, examples: { _ in [example] }, steps: 3, seed: 1) { _ in
                stepSeconds.append(Date().timeIntervalSince(stepStart))
                stepPeaks.append(NFKMLXGPU.peakMemory)
                print("MEMORY \(label) step \(stepPeaks.count): \(String(format: "%.1f", stepSeconds.last!)) s, peak "
                      + "\(gigabytes(stepPeaks.last!)) GB")
                fflush(stdout)
                NFKMLXGPU.resetPeakMemory()
                stepStart = Date()
                return true
            }
            print("MEMORY \(label) steps: losses \(losses), seconds \(stepSeconds.map { String(format: "%.1f", $0) }), "
                  + "peak \(gigabytes(stepPeaks.max() ?? 0)) GB, trainable \(NFKMLXLoRA.trainableParameterCount(of: net)) "
                  + "parameters")
            fflush(stdout)
            XCTAssertEqual(losses.count, 3)
            XCTAssertTrue(losses.allSatisfy(\.isFinite))
            XCTAssertTrue(bits == nil ? net.blocks[0].attn1.toQ is NFKMLXLoRALinear
                                      : net.blocks[0].attn1.toQ is NFKMLXQuantizedLoRALinear)
            XCTAssertEqual(NFKMLXLoRA.trainableParameterCount(of: net), 28 * 2 * 4 * 32 * (2048 + 2048))
        }
    }
}
