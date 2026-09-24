//
//  NFKMLXTrOCRTests.swift
//  InferKitMLXTests
//
//  TrOCR (microsoft/trocr-base-handwritten, MIT). The ViT encoder and the seq2seq decoder evaluate MLX
//  arrays, so these skip without a Metal library for MLX (see Tools/mlx-metallib.sh). The parity tests
//  are gated on the released directory (`IK_VAL_TROCR`) + the recorded oracle (`IK_PARITY_TROCR`, from
//  `run_reference.py trocr`), compared seam by seam.
//

import XCTest
import InferKit
import CoreGraphics
import ImageIO
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXTrOCRTests: XCTestCase {

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

    private func loadedNet(_ directory: String) throws -> NFKMLXTrOCRNet {
        let url = URL(fileURLWithPath: directory)
        let net = try NFKMLXTrOCRNet(configurationURL: url.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: url)
        return net
    }

    /// The module keys the loader targets: the ViT encoder's `google/vit` layout and the decoder-only
    /// seq2seq stack.
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXTrOCRNet(vision: .base, language: try NFKMLXSeq2SeqConfiguration(huggingFaceConfig: [
            "model_type": "trocr", "d_model": 1024, "decoder_layers": 12, "decoder_attention_heads": 16,
            "decoder_ffn_dim": 4096, "vocab_size": 50265, "max_position_embeddings": 512,
            "cross_attention_hidden_size": 768, "pad_token_id": 1, "eos_token_id": 2,
            "decoder_start_token_id": 2, "tie_word_embeddings": true]))
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["encoder.embeddings.cls_token", "encoder.embeddings.position_embeddings",
                         "encoder.embeddings.patch_embeddings.projection.weight",
                         "encoder.encoder.layer.0.attention.attention.query.weight",
                         "encoder.encoder.layer.0.intermediate.dense.weight", "encoder.layernorm.weight",
                         "decoder.shared.weight", "decoder.decoder.embed_positions.weight",
                         "decoder.decoder.layers.0.encoder_attn.k_proj.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
        // The decoder ties its output projection, so there is no lm_head.
        XCTAssertFalse(names.contains("decoder.lm_head.weight"), "trocr-base ties the output projection")
    }

    /// PARITY: the released ViT encoder against the recorded oracle, seam by seam (the embeddings output,
    /// the first encoder block, and the encoder last hidden state that is the decoder memory).
    func testEncoderSeamParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_TROCR"], let recordPath = env["IK_PARITY_TROCR"] else {
            throw XCTSkip("set IK_VAL_TROCR (release directory) and IK_PARITY_TROCR (oracle record)")
        }
        let net = try loadedNet(directory)
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let pixels = rec["pixels"]!                                                 // [384, 384, 3]
        let side = pixels.dim(0)
        let x = pixels.reshaped([1, side, side, 3]).asType(.float32)

        let emb = net.vision.embeddings(x)
        XCTAssertGreaterThan(cosine(emb[0], rec["emb"]!), 0.999, "embeddings diverge")

        let block0 = net.vision.encoder.layer[0](emb)
        XCTAssertGreaterThan(cosine(block0[0], rec["block0"]!), 0.999, "first encoder block diverges")

        let memory = net.imageFeatures(x)
        XCTAssertGreaterThan(cosine(memory[0], rec["enc_last"]!), 0.999, "encoder output diverges")
    }

    /// PARITY: the decoder's first step against the recorded oracle — the cross-attention over the
    /// 768-wide image memory, and the tied output projection.
    func testFirstStepLogitsMatchTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_TROCR"], let recordPath = env["IK_PARITY_TROCR"] else {
            throw XCTSkip("set IK_VAL_TROCR and IK_PARITY_TROCR")
        }
        let net = try loadedNet(directory)
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let pixels = rec["pixels"]!
        let side = pixels.dim(0)
        let memory = net.imageFeatures(pixels.reshaped([1, side, side, 3]).asType(.float32))

        let cache = net.makeCache()
        let start = MLXArray([Int32(net.languageConfiguration.decoderStartTokenId)]).reshaped([1, 1])
        let logits = net.decode(start, memory: memory, cache: cache)
        XCTAssertGreaterThan(cosine(logits[0, 0], rec["logits"]!), 0.999, "first-step logits diverge")
    }

    /// PARITY: greedy generation from the image memory reproduces the reference's greedy transcription ids.
    func testGeneratedTokensMatchTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_TROCR"], let recordPath = env["IK_PARITY_TROCR"] else {
            throw XCTSkip("set IK_VAL_TROCR and IK_PARITY_TROCR")
        }
        let net = try loadedNet(directory)
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        guard let refGen = rec["generated"] else { throw XCTSkip("record has no `generated`") }
        let pixels = rec["pixels"]!
        let side = pixels.dim(0)
        let memory = net.imageFeatures(pixels.reshaped([1, side, side, 3]).asType(.float32))

        let configuration = net.languageConfiguration
        let model = NFKTrOCRDecodeModel(net: net, memory: memory)
        let decoding = NFKMLXSeq2SeqDecoding(beams: 1, maxTokens: 64,
                                            startToken: configuration.decoderStartTokenId,
                                            endToken: configuration.eosTokenId)
        let mine = NFKMLXSeq2SeqDecoder.generate(model, source: [configuration.decoderStartTokenId], decoding: decoding)

        func stripEnds(_ ids: [Int]) -> [Int] {
            var out = ids
            if out.first == configuration.decoderStartTokenId { out.removeFirst() }
            while out.last == configuration.eosTokenId { out.removeLast() }
            return out
        }
        let reference = stripEnds(refGen.asArray(Int32.self).map { Int($0) })
        XCTAssertEqual(mine, reference, "greedy transcription ids must match the reference")
    }

    /// PARITY on every other release, each against its own `run_reference.py trocr` record:
    /// - the processor on the 8-bit image the reference's processor received (PIL bilinear for ViT,
    ///   bicubic for DeiT);
    /// - the embeddings, the first encoder block, and the memory;
    /// - the first-step logits;
    /// - the greedy ids and their decoded text.
    func testEveryOtherReleaseIsAtParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        var measured = [String]()
        for name in ["BASE_PRINTED", "BASE_STR", "BASE_STAGE1", "LARGE_HANDWRITTEN", "LARGE_PRINTED", "LARGE_STR",
                     "LARGE_STAGE1", "SMALL_HANDWRITTEN", "SMALL_PRINTED", "SMALL_STAGE1"] {
            guard let directory = env["IK_VAL_TROCR_\(name)"], let recordPath = env["IK_PARITY_TROCR_\(name)"],
                  FileManager.default.fileExists(atPath: recordPath) else { continue }
            let url = URL(fileURLWithPath: directory)
            let net = try loadedNet(directory)
            let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
            let pixels = rec["pixels"]!
            let side = pixels.dim(0)
            let x = pixels.reshaped([1, side, side, 3]).asType(.float32)

            let rgb = rec["input_rgb"]!.asType(.int32)
            let image = try Self.cgImage(rgb: rgb.asArray(Int32.self).map { UInt8($0) }, width: rgb.dim(1), height: rgb.dim(0))
            let processed = try NFKMLXTrOCRProcessor.pixelValues(image, side: side,
                                                                 bicubic: NFKMLXTrOCRProcessor.resamplesBicubic(inDirectory: url))
            let pixelError = abs(processed[0] - pixels.asType(.float32)).max().item(Float.self)

            let emb = net.vision.embeddings(x)
            let block0 = net.vision.encoder.layer[0](emb)
            let memory = net.imageFeatures(x)
            let configuration = net.languageConfiguration
            let start = MLXArray([Int32(configuration.decoderStartTokenId)]).reshaped([1, 1])
            let logits = net.decode(start, memory: memory, cache: net.makeCache())
            let seams = [("emb", cosine(emb[0], rec["emb"]!)), ("block0", cosine(block0[0], rec["block0"]!)),
                         ("memory", cosine(memory[0], rec["enc_last"]!)), ("logits", cosine(logits[0, 0], rec["logits"]!))]
            print("VALIDATION PARITY trocr \(name): pixels max abs \(pixelError), "
                  + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
            XCTAssertLessThan(pixelError, 1e-5, "\(name) processor")
            for (seam, similarity) in seams { XCTAssertGreaterThan(similarity, 0.999, "\(name) \(seam) diverges") }

            let decoding = NFKMLXSeq2SeqDecoding(beams: 1, maxTokens: 64, startToken: configuration.decoderStartTokenId,
                                                endToken: configuration.eosTokenId)
            let mine = NFKMLXSeq2SeqDecoder.generate(NFKTrOCRDecodeModel(net: net, memory: memory),
                                                     source: [configuration.decoderStartTokenId], decoding: decoding)
            var reference = rec["generated"]!.asArray(Int32.self).map(Int.init)
            if reference.first == configuration.decoderStartTokenId { reference.removeFirst() }
            while reference.last == configuration.eosTokenId { reference.removeLast() }
            XCTAssertEqual(mine, reference, "\(name) greedy ids")

            let tokenizer = try XCTUnwrap(NFKMLXTrOCRProcessor.tokenizer(inDirectory: url), "\(name) tokenizer")
            let text = NFKMLXTrOCRProcessor.cleanText(tokenizer.decode(reference.map { NSNumber(value: $0) }))
            let referenceText = String(decoding: rec["text_utf8"]!.asArray(Int32.self).map { UInt8($0) }, as: UTF8.self)
            XCTAssertEqual(text, referenceText.trimmingCharacters(in: .whitespacesAndNewlines), "\(name) decoded text")

            // End to end through the public factory on the same bytes: processor, encoder, generation,
            // and the release's own tokenizer.
            let backend = try NFKMLXTrOCR.backend(directoryURL: url)
            let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
            XCTAssertEqual(result.output(forKey: NFKOutputText) as? String,
                           referenceText.trimmingCharacters(in: .whitespacesAndNewlines), "\(name) backend transcription")
            measured.append(name)
        }
        try XCTSkipIf(measured.isEmpty, "set IK_VAL_TROCR_<RELEASE> and IK_PARITY_TROCR_<RELEASE>")
    }

    static func cgImage(rgb: [UInt8], width: Int, height: Int) throws -> CGImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0 ..< width * height {
            rgba[pixel * 4] = rgb[pixel * 3]
            rgba[pixel * 4 + 1] = rgb[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = rgb[pixel * 3 + 2]
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) else {
            throw XCTSkip("could not build a CGImage from the recorded bytes")
        }
        return image
    }

    /// The backend reads a line of text end to end (image processor + ViT encoder + generation + byte-level
    /// decode), returning the transcription the reference produces for the same image.
    func testTheBackendReadsTextInAnImage() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_TROCR"] else { throw XCTSkip("set IK_VAL_TROCR") }
        guard let imagePath = env["IK_VAL_TROCR_IMAGE"], let image = Self.loadImage(imagePath) else {
            throw XCTSkip("set IK_VAL_TROCR_IMAGE to a text-line image")
        }
        let backend = try NFKMLXTrOCR.backend(directoryURL: URL(fileURLWithPath: directory))
        let request = NFKInferenceRequest(inputs: [NFKInputImage: image])
        let result = try backend.runInference(for: request)
        let text = (result.output(forKey: NFKOutputText) as? String) ?? ""
        XCTAssertFalse(text.isEmpty, "the transcription is non-empty")
        XCTAssertTrue(text.lowercased().contains("quick brown fox"),
                      "the backend reads the rendered line; got \(text)")
    }

    static func loadImage(_ path: String) -> CGImage? {
        guard FileManager.default.fileExists(atPath: path),
              let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}

/// TrOCR's fine-tuning recipe: the objective and schedule against the authors' fairseq training
/// (`run_reference.py trocr_loss`, `IK_PARITY_TROCR_<RELEASE>_LOSS`), a tiny network's run, and a
/// released network's save and factory reload.
final class NFKMLXTrOCRTrainingTests: XCTestCase {

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
    }

    private func release(_ name: String) throws -> (directory: URL, record: [String: MLXArray]) {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_TROCR_\(name)"], let path = env["IK_PARITY_TROCR_\(name)_LOSS"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_TROCR_\(name) and IK_PARITY_TROCR_\(name)_LOSS (run_reference.py trocr_loss)")
        }
        return (URL(fileURLWithPath: directory), try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays)
    }

    /// PARITY: on each release's own transcription, the target ids are the release tokenizer's, the loss
    /// on the recorded logits is fairseq's, and the port's forward reproduces it.
    func testTheObjectiveIsFairseqsOnTheReleases() throws {
        var measured = 0
        for name in ["SMALL_HANDWRITTEN", "BASE_PRINTED"] {
            guard let (directory, record) = try? release(name) else { continue }
            let text = String(decoding: record["text_utf8"]!.asArray(Int32.self).map { UInt8($0) }, as: UTF8.self)
            let tokenizer = try XCTUnwrap(NFKMLXTrOCRProcessor.tokenizer(inDirectory: directory))
            let target = record["target"]!
            let net = try NFKMLXTrOCR.network(directoryURL: directory)
            let ids = NFKMLXTrOCRProcessor.targetIds(for: text, tokenizer: tokenizer,
                                                     endToken: net.languageConfiguration.eosTokenId)
            XCTAssertEqual(ids.asArray(Int32.self), target.asArray(Int32.self), "\(name) target ids for \(text)")

            let objective = NFKMLXTrOCRObjective()
            // torch's float32 log-sum-exp over the 64k vocabulary rounds to about 2e-5; the float64 value
            // is the one to hold the port to.
            let reference = record["fairseq_loss_f64"]!.item(Float.self)
            let onLogits = objective.loss(logits: record["logits"]!, target: target).item(Float.self)
            let forward = objective(net, record["pixel_values"]!, target).item(Float.self)
            print("VALIDATION PARITY trocr \(name) loss: fairseq float64 \(reference), float32 "
                  + "\(record["output"]!.item(Float.self)), on its logits \(onLogits), port forward \(forward); "
                  + "transformers labels= \(record["hf_loss"]!.item(Float.self))")
            XCTAssertEqual(onLogits, reference, accuracy: 1e-6)
            XCTAssertEqual(forward, reference, accuracy: 1e-5)
            NFKMLXGPU.clearCache()
            measured += 1
        }
        try XCTSkipIf(measured == 0, "no TrOCR loss record is set")
    }

    /// PARITY: `fairseqInverseSquareRoot` against fairseq's own `InverseSquareRootSchedule` under the IAM
    /// recipe's settings, at every update count from 0 to 1199.
    func testTheScheduleIsFairseqsInverseSquareRoot() throws {
        let (_, record) = try release("SMALL_HANDWRITTEN")
        let rates = record["lr_iam"]!.asArray(Float.self)
        let schedule = NFKMLXLearningRateSchedule.fairseqInverseSquareRoot(warmupSteps: 500, initialScale: 1e-8 / 2e-5)
        for (update, rate) in rates.enumerated() {
            XCTAssertEqual(2e-5 * schedule.multiplier(update) / rate, 1, accuracy: 1e-5, "update \(update)")
        }
    }

    private static func tinyNet() -> NFKMLXTrOCRNet {
        let vision = NFKMLXTrOCRVisionConfiguration(imageSize: 32, hiddenSize: 32, layers: 1, heads: 2,
                                                    intermediateSize: 64, qkvBias: true)
        let language = NFKMLXSeq2SeqConfiguration(
            vocabularySize: 40, dModel: 32, encoderLayers: 0, decoderLayers: 1, heads: 2, encoderFFDim: 64,
            decoderFFDim: 64, maxPositions: 64, activation: .relu, positions: .learned, normalizeBefore: false,
            finalLayerNorm: false, layerNormEmbedding: true, scaleEmbedding: true, finalLogitsBias: false,
            padTokenId: 1, eosTokenId: 2, decoderStartTokenId: 2, crossAttentionWidth: 32)
        return NFKMLXTrOCRNet(vision: vision, language: language)
    }

    /// A tiny network's run: the loss falls, and the decoder-only policy leaves the encoder unchanged.
    func testAFineTuneLowersTheLossAndHonorsTheFreeze() throws {
        try requireMLXRuntime()
        MLXRandom.seed(4)
        let net = Self.tinyNet()
        let encoderBefore = net.vision.parameters().flattened().map { $0.1 + 0 }
        eval(encoderBefore)
        let pixels = MLXRandom.normal([1, 32, 32, 3])
        let target = MLXArray([Int32(7), 12, 19, 2])
        let losses = try NFKMLXTrOCR.fineTune(net, examples: { _ in (pixels, target) }, trainable: .decoder,
                                              learningRate: 1e-3, steps: 20, learningRateSchedule: .constant)
        XCTAssertLessThan(losses.last!, losses.first! * 0.5, "the loss falls")
        for (before, after) in zip(encoderBefore, net.vision.parameters().flattened().map(\.1)) {
            XCTAssertEqual(abs(after - before).max().item(Float.self), 0, "the encoder stays frozen")
        }
    }

    /// The round trip on a released network: a fine-tuned directory reloads through the factory with the
    /// same logits, and the backend transcribes from it.
    func testAFineTunedReleaseReloadsThroughTheFactory() throws {
        let (directory, record) = try release("SMALL_HANDWRITTEN")
        let net = try NFKMLXTrOCR.network(directoryURL: directory)
        let pixels = record["pixel_values"]!, target = record["target"]!
        try NFKMLXTrOCR.fineTune(net, examples: { _ in (pixels, target) }, trainable: .decoder, steps: 2)
        let saved = FileManager.default.temporaryDirectory.appendingPathComponent("trocr-tuned-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: saved) }
        try NFKMLXTrOCR.save(net, toDirectoryURL: saved, release: directory)

        let reloaded = try NFKMLXTrOCR.network(directoryURL: saved)
        let input = NFKMLXTranslationObjective.decoderInput(for: target, startToken: 2).reshaped([1, -1])
        let before = net.decode(input, memory: net.imageFeatures(pixels), cache: net.makeCache())
        let after = reloaded.decode(input, memory: reloaded.imageFeatures(pixels), cache: reloaded.makeCache())
        XCTAssertLessThan(abs(after - before).max().item(Float.self), 1e-5, "the reloaded network reproduces the logits")

        let image = try XCTUnwrap(NFKMLXTrOCRTests.loadImage(
            NFKMLXValidationConfig.environment["IK_VAL_TROCR_IMAGE"] ?? ""), "IK_VAL_TROCR_IMAGE")
        let result = try NFKMLXTrOCR.backend(directoryURL: saved)
            .runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
        XCTAssertNotNil(result.output(forKey: NFKOutputText) as? String)
    }
}
