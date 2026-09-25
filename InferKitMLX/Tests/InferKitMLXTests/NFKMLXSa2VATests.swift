//
//  NFKMLXSa2VATests.swift
//  InferKitMLXTests
//
//  Sa2VA-4B (ByteDance/Sa2VA-4B, Apache) — a segmentation VLM: InternViT-300M + pixel-shuffle projector
//  + Qwen2.5-3B, whose `[SEG]` hidden state drives a SAM 2 grounding encoder to a mask. The network
//  evaluates MLX arrays, so these skip without a Metal library for MLX (see Tools/mlx-metallib.sh). The
//  parity tests are gated on the released directory (`IK_VAL_SA2VA`) and the recorded oracle
//  (`IK_PARITY_SA2VA`, from `run_reference.py sa2va`), compared seam by seam from the vision tower
//  through the fused decoder to the final mask.
//

import XCTest
import InferKit
import CoreGraphics
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXSa2VATests: XCTestCase {
    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }


    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        let a = mine.asType(.float32).reshaped([-1]).asArray(Float.self)
        let b = reference.asType(.float32).reshaped([-1]).asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    /// Intersection-over-union of two logit maps thresholded at zero, the reference's mask decision.
    static func maskIoU(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        let a = mine.asType(.float32).reshaped([-1]).asArray(Float.self)
        let b = reference.asType(.float32).reshaped([-1]).asArray(Float.self)
        let n = min(a.count, b.count)
        var intersection = 0, union = 0
        for i in 0 ..< n {
            let p = a[i] > 0, q = b[i] > 0
            if p || q { union += 1 }
            if p && q { intersection += 1 }
        }
        return union == 0 ? 1 : Float(intersection) / Float(union)
    }

    /// The pixels whose mask decision differs from the reference's, and the largest reference logit
    /// magnitude among them: a decision flipped at a logit within float error of zero is rounding, and
    /// one flipped far from zero is a defect.
    static func maskFlips(_ mine: MLXArray, _ reference: MLXArray) -> (count: Int, largestReference: Float) {
        eval(mine)
        let a = mine.reshaped([-1]).asArray(Float.self), b = reference.reshaped([-1]).asArray(Float.self)
        var count = 0, largest: Float = 0
        for i in 0 ..< min(a.count, b.count) where (a[i] > 0) != (b[i] > 0) {
            count += 1
            largest = max(largest, abs(b[i]))
        }
        return (count, largest)
    }

    private func loadedNet(_ directory: String) throws -> NFKMLXSa2VANet {
        let url = URL(fileURLWithPath: directory)
        let net = NFKMLXSa2VANet(try NFKMLXSa2VANet.configuration(fromDirectory: url))
        try net.loadWeights(fromDirectory: url, dtype: .float32)
        return net
    }

    private func record(_ path: String) throws -> [String: MLXArray] {
        try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
    }

    private func gate() throws -> (net: NFKMLXSa2VANet, rec: [String: MLXArray]) {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_SA2VA"], let recordPath = env["IK_PARITY_SA2VA"] else {
            throw XCTSkip("set IK_VAL_SA2VA (release directory) and IK_PARITY_SA2VA (oracle record)")
        }
        return (try loadedNet(directory), try record(recordPath))
    }

    /// The module keys the loader targets mirror the checkpoint across all five subtrees: the InternViT
    /// tower, the projector's numeric sequential, the Qwen2.5 decoder, the `[SEG]` bridge, and the nested
    /// SAM 2 grounding encoder.
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXSa2VAConfiguration(
            decoder: .init(hiddenSize: 2048, layerCount: 36, headCount: 16, keyValueHeadCount: 2,
                           intermediateSize: 11008, vocabularySize: 151_679, ropeTheta: 1_000_000,
                           rmsEpsilon: 1e-6, tiesWordEmbeddings: false, attentionBias: true),
            imageContextTokenId: 151_667, segmentationTokenId: 151_674)
        let names = Set(NFKMLXSa2VANet(configuration).parameters().flattened().map(\.0))
        for expected in ["vision_model.embeddings.patch_embedding.weight",
                         "vision_model.embeddings.class_embedding",
                         "vision_model.embeddings.position_embedding",
                         "vision_model.encoder.layers.0.attn.qkv.weight",
                         "vision_model.encoder.layers.0.ls1",
                         "vision_model.encoder.layers.23.mlp.fc2.weight",
                         "mlp1.norm.weight", "mlp1.fc1.weight", "mlp1.fc2.weight",
                         "language_model.model.layers.0.self_attn.q_proj.weight",
                         "language_model.model.layers.0.self_attn.q_proj.bias",
                         "language_model.lm_head.weight",
                         "text_hidden_fcs.fc1.weight", "text_hidden_fcs.fc2.weight",
                         "grounding_encoder.sam2_model.no_mem_embed",
                         "grounding_encoder.sam2_model.memory_encoder.fuser.0.gamma"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    /// PARITY: the InternViT tower and the pixel-shuffle projector against the recorded oracle. The
    /// pixel values are the reference's own, so the resize is not a variable.
    func testVisionAndProjectorParity() throws {
        let (net, rec) = try gate()
        let pixelValues = rec["pixel_values"]!.asType(.float32)                    // [tiles, 3, 448, 448]
        let visionOut = net.vision(pixelValues.transposed(0, 2, 3, 1))
        let visionCosine = cosine(visionOut, rec["vit_last_hidden"]!)
        let features = net.imageFeatures(pixelValues: pixelValues)
        let projectorCosine = cosine(features, rec["vit_embeds"]!)
        print("VALIDATION PARITY sa2va 4B: vision \(visionCosine), projector \(projectorCosine)")
        XCTAssertGreaterThan(visionCosine, 0.9999, "InternViT diverges")
        XCTAssertGreaterThan(projectorCosine, 0.9999, "projector diverges")
    }

    /// PARITY: the fusion that splices vision tokens into the decoder embeddings, and the decoder hidden
    /// state at the `[SEG]` position mapped through the `[SEG]` bridge — the sparse prompt SAM 2 reads.
    func testFusionAndSegmentationEmbedding() throws {
        let (net, rec) = try gate()
        let features = net.imageFeatures(pixelValues: rec["pixel_values"]!.asType(.float32))
        let inputIds = rec["input_ids"]!.reshaped([-1]).asArray(Int32.self).map(Int.init)
        let fused = net.fusedEmbeddings(inputIds: inputIds, imageFeatures: features)
        let fusionCosine = cosine(fused, rec["fused"]!)

        // The oracle generated from inputs_embeds, so its `sequence` holds the generated tokens only;
        // the teacher-forced hidden state at [SEG] is read from the full prompt+generated sequence.
        let generated = rec["sequence"]!.reshaped([-1]).asArray(Int32.self).map(Int.init)
        let sequence = inputIds + generated
        let hidden = net.language.hiddenStates(
            fromEmbeddings: net.fusedEmbeddings(inputIds: sequence, imageFeatures: features))
        let segPositions = sequence.enumerated().filter { $0.element == net.configuration.segmentationTokenId }.map(\.offset)
        XCTAssertFalse(segPositions.isEmpty, "the recorded sequence carries a [SEG]")
        let embedding = net.textHiddenFCS(hidden[0, segPositions[0]].reshaped([1, -1]))
        let segCosine = cosine(embedding, rec["seg_embedding"]!)
        print("VALIDATION PARITY sa2va 4B: fusion \(fusionCosine), seg \(segCosine)")
        XCTAssertGreaterThan(fusionCosine, 0.9999, "image/text fusion diverges")
        XCTAssertGreaterThan(segCosine, 0.9999, "[SEG] embedding diverges")
    }

    /// PARITY: the SAM 2 grounding branch from the recorded `[SEG]` embedding and grounding image to the
    /// mask logits — the model's segmentation output. The mask is compared by cosine on the logits and
    /// by IoU of the thresholded masks.
    func testMaskParity() throws {
        let (net, rec) = try gate()
        let groundingImage = rec["g_pixel_values"]!.asType(.float32).transposed(0, 2, 3, 1)
        let segEmbedding = rec["seg_embedding"]![0].asType(.float32)               // the first [SEG], [256]
        // The bridge already ran in the oracle: feed the recorded embedding straight into SAM 2.
        let embedding = segEmbedding.reshaped([1, 1, -1])
        let mask = net.grounding.segment(image: groundingImage, languageEmbedding: embedding)
        let maskCosine = cosine(mask.lowResolution, rec["low_res_best"]!)
        let iou = Self.maskIoU(mask.lowResolution, rec["low_res_best"]!)
        print("VALIDATION PARITY sa2va 4B: mask \(maskCosine) IoU \(iou)")
        XCTAssertGreaterThan(maskCosine, 0.9999, "mask logits diverge")
        XCTAssertGreaterThan(iou, 0.999, "mask decision diverges")
    }

    /// PARITY on every other InternVL-based release, each against its own `run_reference.py sa2va`
    /// record: the vision tower and projector, the fusion, the `[SEG]` embedding, the mask, and greedy
    /// generation under the release's own template and end token.
    func testEveryOtherInternVLReleaseIsAtParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        var measured = [String]()
        for name in ["1B", "INTERNVL3_2B"] {
            guard let directory = env["IK_VAL_SA2VA_\(name)"], let recordPath = env["IK_PARITY_SA2VA_\(name)"],
                  FileManager.default.fileExists(atPath: recordPath) else { continue }
            try autoreleasepool {
                let net = try loadedNet(directory)
                let rec = try record(recordPath)
                let configuration = net.configuration
                let pixelValues = rec["pixel_values"]!.asType(.float32)
                let visionCosine = cosine(net.vision(pixelValues.transposed(0, 2, 3, 1)), rec["vit_last_hidden"]!)
                let features = net.imageFeatures(pixelValues: pixelValues)
                let projectorCosine = cosine(features, rec["vit_embeds"]!)
                let inputIds = rec["input_ids"]!.reshaped([-1]).asArray(Int32.self).map(Int.init)
                let fusionCosine = cosine(net.fusedEmbeddings(inputIds: inputIds, imageFeatures: features), rec["fused"]!)

                let referenceGenerated = rec["sequence"]!.reshaped([-1]).asArray(Int32.self).map(Int.init)
                let sequence = inputIds + referenceGenerated
                let hidden = net.language.hiddenStates(
                    fromEmbeddings: net.fusedEmbeddings(inputIds: sequence, imageFeatures: features))
                let seg = try XCTUnwrap(sequence.firstIndex(of: configuration.segmentationTokenId), "\(name) answers with [SEG]")
                let segCosine = cosine(net.textHiddenFCS(hidden[0, seg].reshaped([1, -1])), rec["seg_embedding"]![0])

                let groundingImage = rec["g_pixel_values"]!.asType(.float32).transposed(0, 2, 3, 1)
                let mask = net.grounding.segment(image: groundingImage,
                                                 languageEmbedding: rec["seg_embedding"]![0].asType(.float32).reshaped([1, 1, -1]))
                let maskCosine = cosine(mask.lowResolution, rec["low_res_best"]!)
                let iou = Self.maskIoU(mask.lowResolution, rec["low_res_best"]!)

                let tokenizer = try XCTUnwrap(NFKMLXLanguage.releaseTokenizer(inDirectory: URL(fileURLWithPath: directory)))
                let generated = net.generate(inputIds: inputIds,
                                             imageFeatures: net.imageFeatures(pixelValues: rec["pixel_values"]!.asType(.float32)),
                                             maximumTokens: referenceGenerated.count + 8, endToken: configuration.endTokenId,
                                             stopsAfter: { tokens in
                                                 configuration.template.stops(after: tokens) {
                                                     tokenizer.decode($0.map { NSNumber(value: $0) })
                                                 }
                                             })
                let expected = referenceGenerated.last == configuration.endTokenId
                    ? Array(referenceGenerated.dropLast()) : referenceGenerated
                print("VALIDATION PARITY sa2va \(name): vision \(visionCosine), projector \(projectorCosine), "
                      + "fusion \(fusionCosine), seg \(segCosine), mask \(maskCosine) IoU \(iou), "
                      + "generation \(generated.tokens == expected ? "token-exact" : "DIVERGES")")
                for (seam, similarity) in [("InternViT", visionCosine), ("projector", projectorCosine),
                                           ("fusion", fusionCosine), ("[SEG] embedding", segCosine),
                                           ("mask logits", maskCosine)] {
                    XCTAssertGreaterThan(similarity, 0.9999, "\(name) \(seam)")
                }
                XCTAssertGreaterThan(iou, 0.999, "\(name) mask decision")
                XCTAssertEqual(generated.tokens, expected, "\(name) greedy generation")
            }
            NFKMLXGPU.clearCache()
            try Self.checkBackend(directory: directory, recordPath: recordPath, name: name)
            measured.append(name)
        }
        try XCTSkipIf(measured.isEmpty, "set IK_VAL_SA2VA_<RELEASE> and IK_PARITY_SA2VA_<RELEASE>")
    }

    /// PARITY on the releases too large for a float32 oracle here, each cut to its first decoder layers
    /// (`truncate.py`) and recorded teacher-forced (`run_reference.py sa2va_teacher`): the vision tower
    /// and projector, the fusion, the decoder's final hidden states over the prompt and a fixed answer,
    /// the last logits, the `[SEG]` embedding, and the mask it drives.
    func testEveryCutReleaseIsAtParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        var measured = [String]()
        for name in ["8B_CUT4", "26B_CUT4", "INTERNVL3_8B_CUT4", "INTERNVL3_14B_CUT4"] {
            guard let directory = env["IK_VAL_SA2VA_\(name)"], let recordPath = env["IK_PARITY_SA2VA_\(name)"],
                  FileManager.default.fileExists(atPath: recordPath) else { continue }
            try autoreleasepool {
                let net = try loadedNet(directory)
                let rec = try record(recordPath)
                let pixelValues = rec["pixel_values"]!.asType(.float32)
                let features = net.imageFeatures(pixelValues: pixelValues)
                let ids = rec["input_ids"]!.reshaped([-1]).asArray(Int32.self).map(Int.init)
                let fused = net.fusedEmbeddings(inputIds: ids, imageFeatures: features)
                let hidden = net.language.hiddenStates(fromEmbeddings: fused)
                let logits = net.language.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...])
                let seg = try XCTUnwrap(ids.firstIndex(of: net.configuration.segmentationTokenId))
                let embedding = net.textHiddenFCS(hidden[0, seg].reshaped([1, -1]))
                let mask = net.grounding.segment(image: rec["g_pixel_values"]!.asType(.float32).transposed(0, 2, 3, 1),
                                                 languageEmbedding: embedding.reshaped([1, 1, -1]))
                let seams = [("vision", cosine(net.vision(pixelValues.transposed(0, 2, 3, 1)), rec["vit_last_hidden"]!)),
                             ("projector", cosine(features, rec["vit_embeds"]!)), ("fusion", cosine(fused, rec["fused"]!)),
                             ("decoder", cosine(hidden[0], rec["dec_last"]!)), ("logits", cosine(logits[0, 0], rec["last_logits"]!)),
                             ("seg", cosine(embedding, rec["seg_embedding"]!)), ("mask", cosine(mask.lowResolution, rec["low_res_best"]!))]
                let iou = Self.maskIoU(mask.lowResolution, rec["low_res_best"]!)
                // The tokenizer both ways: the instruction to the recorded prompt ids, the answer's ids to
                // the reference's decode.
                if let promptBytes = rec["prompt_utf8"], let length = rec["prompt_length"],
                   let answerBytes = rec["answer_decoded_utf8"] {
                    let tokenizer = try XCTUnwrap(NFKMLXSa2VA.tokenizer(inDirectory: URL(fileURLWithPath: directory)))
                    let promptLength = Int(length.item(Int32.self))
                    let prompt = String(decoding: promptBytes.asArray(Int32.self).map { UInt8($0) }, as: UTF8.self)
                    XCTAssertEqual(tokenizer.encode(prompt).map(\.intValue), Array(ids.prefix(promptLength)), "\(name) prompt ids")
                    let answer = String(decoding: answerBytes.asArray(Int32.self).map { UInt8($0) }, as: UTF8.self)
                    XCTAssertEqual(tokenizer.decode(ids.dropFirst(promptLength).map { NSNumber(value: $0) }), answer,
                                   "\(name) answer decode")
                }
                print("VALIDATION PARITY sa2va \(name): " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", ") + ", IoU \(iou)")
                for (seam, similarity) in seams { XCTAssertGreaterThan(similarity, 0.9999, "\(name) \(seam) diverges") }
                XCTAssertGreaterThan(iou, 0.999, "\(name) mask decision")
            }
            NFKMLXGPU.clearCache()
            measured.append(name)
        }
        try XCTSkipIf(measured.isEmpty, "set IK_VAL_SA2VA_<RELEASE>_CUT4 and IK_PARITY_SA2VA_<RELEASE>_CUT4")
    }

    /// End to end through the public factory on the oracle's plate and request: the answer is the
    /// reference's generated text, decoded by the release's tokenizer and trimmed, and it drives a mask.
    static func checkBackend(directory: String, recordPath: String, name: String) throws {
        let url = URL(fileURLWithPath: directory)
        // The backend is released inside the closure, so the clearCache below returns its buffers.
        NFKMLXGPU.resetPeakMemory()
        let result = try { () throws -> NFKInferenceResult in
            let backend = try NFKMLXSa2VA.backend(directoryURL: url)
            let loaded = NFKMLXGPU.peakMemory
            let result = try backend.runInference(for: NFKInferenceRequest(
                inputs: [NFKInputImage: plate(), NFKInputPrompt: "<image>Please segment the bright object."]))
            print("VALIDATION MEMORY sa2va \(name) backend peak: after load \(loaded >> 20) MB, "
                  + "after the answer \(NFKMLXGPU.peakMemory >> 20) MB")
            return result
        }()
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let reference = (rec["sequence"] ?? rec["generated"])!.reshaped([-1]).asArray(Int32.self)
        let tokenizer = try XCTUnwrap(NFKMLXSa2VA.tokenizer(inDirectory: url))
        let expected = tokenizer.decode(reference.map { NSNumber(value: $0) }).trimmingCharacters(in: .whitespacesAndNewlines)
        let answer = result.output(forKey: NFKOutputText) as? String
        print("VALIDATION BACKEND sa2va \(name): \(answer ?? "nil")")
        XCTAssertEqual(answer, expected, "\(name) backend answer")
        XCTAssertNotNil(result.output(forKey: NFKOutputMask), "\(name) the [SEG] answer produces a mask")
        NFKMLXGPU.clearCache()
    }

    /// PARITY of every family's preprocessing on a 640×360 picture, against the release's own code with
    /// no weights (`run_reference.py sa2va_processor`, written to the release directory's
    /// `processor.safetensors`): InternVL's tiles and thumbnail, the Qwen-VL patches and grid, LLaVA's
    /// pixels, and the grounding image. The 448 plate resizes nothing on the InternVL path, so the
    /// seams above cannot see these.
    func testEveryFamilysPreprocessingMatchesTheReference() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        let env = NFKMLXValidationConfig.environment
        var measured = [String]()
        for name in ["1B", "QWEN3_VL_2B", "QWEN2_5_VL_3B", "QWEN3_VL_4B_SAM3", "LLAVA_CUT4"] {
            guard let directory = env["IK_VAL_SA2VA_\(name)"] else { continue }
            let recordURL = URL(fileURLWithPath: directory).appendingPathComponent("processor.safetensors")
            guard FileManager.default.fileExists(atPath: recordURL.path) else { continue }
            let rec = try NFKMLXWeights.loadCheckpoint(url: recordURL).arrays
            let rgb = rec["input_rgb"]!
            let image = try NFKMLXTrOCRTests.cgImage(rgb: rgb.asArray(Int32.self).map { UInt8($0) },
                                                     width: rgb.dim(1), height: rgb.dim(0))
            func maxAbs(_ mine: MLXArray, _ reference: MLXArray) -> Float {
                mine.shape == reference.shape ? abs(mine - reference).max().item(Float.self) : .infinity
            }
            var errors = [(String, Float)]()
            if let tiles = rec["tile_pixel_values"] {
                let mine = NFKMLXSa2VAProcessor.tilePixels(try NFKMLXSa2VAProcessor.dynamicTiles(image), side: 448)
                errors.append(("tiles", maxAbs(mine, tiles)))
            }
            if let pixels = rec["pixel_values"], let recordedGrid = rec["image_grid_thw"] {
                let processor = name.hasPrefix("QWEN2_5") ? NFKMLXSa2VAQwen.qwen25ImageProcessor : NFKMLXSa2VAQwen.imageProcessor
                let (mine, grid) = processor.process(image)
                XCTAssertEqual([grid.t, grid.h, grid.w], recordedGrid.reshaped([-1]).asArray(Int32.self).map(Int.init),
                               "\(name) grid")
                errors.append(("pixels", maxAbs(mine, pixels)))
            }
            if let pixels = rec["llava_pixel_values"] {
                errors.append(("pixels", maxAbs(try NFKMLXSa2VALLaVA.pixelValues(image), pixels.transposed(0, 2, 3, 1))))
            }
            let grounding = rec["g_pixel_values"]!
            errors.append(("grounding", maxAbs(try NFKMLXSa2VAProcessor.groundingPixels(image, side: grounding.dim(2)),
                                               grounding)))
            print("VALIDATION PARITY sa2va processor \(name): "
                  + errors.map { "\($0.0) max abs \($0.1)" }.joined(separator: ", "))
            for (seam, error) in errors { XCTAssertLessThan(error, 1e-5, "\(name) \(seam)") }
            measured.append(name)
        }
        try XCTSkipIf(measured.isEmpty, "record run_reference.py sa2va_processor into <release>/processor.safetensors")
    }

    /// The oracle's plate: four quadrants and a bright disk, 448 square.
    static func plate() -> CGImage {
        let size = 448
        var rgba = [UInt8](repeating: 255, count: size * size * 4)
        for y in 0 ..< size {
            for x in 0 ..< size {
                let quadrant: (UInt8, UInt8, UInt8) = y < size / 2
                    ? (x < size / 2 ? (40, 60, 90) : (90, 40, 60))
                    : (x < size / 2 ? (60, 90, 40) : (30, 30, 30))
                let dx = Double(x) - Double(size) * 0.62, dy = Double(y) - Double(size) * 0.40
                let color = dx * dx + dy * dy < pow(Double(size) * 0.16, 2) ? (230, 210, 120) : quadrant
                let offset = (y * size + x) * 4
                (rgba[offset], rgba[offset + 1], rgba[offset + 2]) = color
            }
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        return CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    /// End to end through the public factory on the oracle's plate and request: the answer is the
    /// reference's generated text alone, trimmed, and its `[SEG]` produces a mask.
    func testTheBackendAnswersWithTheReferenceTextAndAMask() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_SA2VA"], let recordPath = env["IK_PARITY_SA2VA"] else {
            throw XCTSkip("set IK_VAL_SA2VA and IK_PARITY_SA2VA")
        }
        let url = URL(fileURLWithPath: directory)
        let backend = try NFKMLXSa2VA.backend(directoryURL: url)
        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputImage: Self.plate(), NFKInputPrompt: "<image>Please segment the bright object."]))
        let tokenizer = try XCTUnwrap(NFKMLXLanguage.releaseTokenizer(inDirectory: url))
        let reference = try record(recordPath)["sequence"]!.reshaped([-1]).asArray(Int32.self)
        let expected = tokenizer.decode(reference.map { NSNumber(value: $0) }).trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertEqual(result.output(forKey: NFKOutputText) as? String, expected)
        XCTAssertNotNil(result.output(forKey: NFKOutputMask), "the [SEG] answer produces a mask")
    }

    /// The decoder's greedy generation reproduces the reference's generated ids token for token.
    func testGeneratedTokensMatchTheReference() throws {
        let (net, rec) = try gate()
        let features = net.imageFeatures(pixelValues: rec["pixel_values"]!.asType(.float32))
        let inputIds = rec["input_ids"]!.reshaped([-1]).asArray(Int32.self).map(Int.init)
        // The oracle's `sequence` is the whole generated run, ending in the phi3 `<|end|>` text tokens; the
        // stop is the backend's own, a test of the decoded text.
        let referenceGenerated = rec["sequence"]!.reshaped([-1]).asArray(Int32.self).map(Int.init)
        let tokenizer = try XCTUnwrap(NFKMLXLanguage.releaseTokenizer(
            inDirectory: URL(fileURLWithPath: NFKMLXValidationConfig.environment["IK_VAL_SA2VA"]!)))
        let generated = net.generate(inputIds: inputIds, imageFeatures: features,
                                     maximumTokens: referenceGenerated.count + 8, endToken: 151_645,
                                     stopsAfter: { tokens in
                                         net.configuration.template.stops(after: tokens) {
                                             tokenizer.decode($0.map { NSNumber(value: $0) })
                                         }
                                     })
        XCTAssertEqual(generated.tokens, referenceGenerated, "greedy diverges from the reference")
    }
}

final class NFKMLXInternLM2Tests: XCTestCase {

    /// The fused projection holds, per key-value head, its query heads, then its key head, then its
    /// value head (modeling_internlm2's `(h gs d)` rearrange); the split recovers each in head order.
    func testTheFusedProjectionSplitsByKeyValueGroup() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        let configuration = NFKMLXLanguageConfiguration(
            hiddenSize: 3, layerCount: 1, headCount: 4, keyValueHeadCount: 2, headDimensions: 1,
            intermediateSize: 4, vocabularySize: 8, ropeTheta: 10_000, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: false, attentionBias: false)
        // Rows: kv head 0 = [q0, q1, k0, v0], kv head 1 = [q2, q3, k1, v1]; each row is its label × 1.
        let labels: [Float] = [10, 11, 20, 30, 12, 13, 21, 31]
        let fused = MLXArray(labels.flatMap { [$0, $0, $0] }).reshaped([8, 3])
        let split = NFKMLXInternLM2.splitQueryKeyValue(fused, configuration: configuration)
        XCTAssertEqual(split.query[0..., 0].asArray(Float.self), [10, 11, 12, 13])
        XCTAssertEqual(split.key[0..., 0].asArray(Float.self), [20, 21])
        XCTAssertEqual(split.value[0..., 0].asArray(Float.self), [30, 31])
    }

    /// The dense decoder, loaded through the InternLM2 remap, against the release's own
    /// `modeling_internlm2.py` at a tiny seeded configuration with grouped-query heads
    /// (`run_reference.py internlm2_tiny`): final hidden states and logits.
    func testTheRemappedDecoderMatchesInternLM2() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_INTERNLM2_TINY"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_INTERNLM2_TINY (run_reference.py internlm2_tiny)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let json: [String: Any] = ["architectures": ["InternLM2ForCausalLM"], "model_type": "internlm2", "vocab_size": 32,
                                   "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
                                   "num_attention_heads": 8, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
                                   "rope_theta": 1_000_000, "bias": false, "tie_word_embeddings": false,
                                   "rope_scaling": ["type": "dynamic", "factor": 2.0], "max_position_embeddings": 64]
        let configuration = try NFKMLXLanguage.configuration(
            fromJSON: NFKMLXInternLM2.denseConfigurationJSON(json, sequencesStayBelowWindow: true))
        let net = NFKMLXLanguageNet(configuration)
        let pairs = rec.filter { !$0.key.hasPrefix("_") && $0.key != "output" && $0.key != "input_image" }
            .map { ("language_model." + $0.key, $0.value) }
        let dense = NFKMLXInternLM2.denseWeights(pairs, prefix: "language_model.", configuration: configuration)
            .map { (String($0.0.dropFirst("language_model.".count)), $0.1) }
        try NFKMLXWeights.apply(dense, to: net, verifyShapes: true)
        let ids = rec["_ids"]!.asArray(Int32.self)
        let hidden = net.hiddenStates(fromEmbeddings: net.embed(MLXArray(ids).reshaped([1, ids.count])))
        let logits = net.logits(fromHidden: hidden)
        func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
            let x = a.reshaped([-1]).asArray(Float.self), y = b.reshaped([-1]).asArray(Float.self)
            var dot: Float = 0, nx: Float = 0, ny: Float = 0
            for i in 0 ..< x.count { dot += x[i] * y[i]; nx += x[i] * x[i]; ny += y[i] * y[i] }
            return dot / (nx.squareRoot() * ny.squareRoot())
        }
        let seams = [("hidden", cosine(hidden[0], rec["output"]!)), ("logits", cosine(logits[0], rec["_logits"]!))]
        print("VALIDATION PARITY internlm2 tiny: " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        for (seam, similarity) in seams { XCTAssertGreaterThan(similarity, 0.99999, "\(seam) diverges") }
    }

    /// The InternLM2 releases' tokenizer both ways against their own slow `InternLM2Tokenizer`
    /// (`run_reference.py internlm2_tokenizer`, under sentencepiece 0.2.0, since 0.2.2 refuses the
    /// release's model for a piece holding a null character): special-token splitting, per-run
    /// SentencePiece encoding with the leading `<s>`, and `_decode`'s spacing.
    func testTheTokenizerMatchesTheRelease() throws {
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_INTERNLM2_TOKENIZER"], let path = env["IK_PARITY_INTERNLM2_TOKENIZER"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_INTERNLM2_TOKENIZER and IK_PARITY_INTERNLM2_TOKENIZER (run_reference.py internlm2_tokenizer)")
        }
        let cases = [
            "<|im_start|>user\n<img><IMG_CONTEXT><IMG_CONTEXT></img>\nPlease segment the bright object.<|im_end|>\n<|im_start|>assistant\n",
            "Sure, it is [SEG].", "Sure, [SEG] and [SEG].<|im_end|>", "  leading spaces, café, 数字 123", "<p>left</p> [SEG]"]
        let tokenizer = try NFKMLXInternLM2Tokenizer(directory: URL(fileURLWithPath: directory))
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        for (index, text) in cases.enumerated() {
            let reference = rec["ids_\(index)"]!.asArray(Int32.self).map(Int.init)
            XCTAssertEqual(tokenizer.encode(text).map(\.intValue), reference, "case \(index) ids")
            let decoded = String(decoding: rec["decoded_\(index)"]!.asArray(Int32.self).map { UInt8($0) }, as: UTF8.self)
            XCTAssertEqual(tokenizer.decodeIds(reference), decoded, "case \(index) decode")
        }
    }

    func testTheNamesMapOntoTheDenseDecoder() {
        let prefix = "language_model."
        let cases = ["model.tok_embeddings.weight": "model.embed_tokens.weight", "output.weight": "lm_head.weight",
                     "model.norm.weight": "model.norm.weight",
                     "model.layers.3.attention.wo.weight": "model.layers.3.self_attn.o_proj.weight",
                     "model.layers.3.feed_forward.w1.weight": "model.layers.3.mlp.gate_proj.weight",
                     "model.layers.3.feed_forward.w3.weight": "model.layers.3.mlp.up_proj.weight",
                     "model.layers.3.feed_forward.w2.weight": "model.layers.3.mlp.down_proj.weight",
                     "model.layers.3.attention_norm.weight": "model.layers.3.input_layernorm.weight",
                     "model.layers.3.ffn_norm.weight": "model.layers.3.post_attention_layernorm.weight"]
        for (release, dense) in cases {
            XCTAssertEqual(NFKMLXInternLM2.denseKey(prefix + release, prefix: prefix), prefix + dense)
        }
        XCTAssertNil(NFKMLXInternLM2.denseKey(prefix + "model.layers.0.attention.wqkv.weight", prefix: prefix))
    }

    func testTheDynamicRotaryIsDroppedOnlyBelowTheWindow() {
        let json: [String: Any] = ["model_type": "internlm2", "bias": false,
                                   "rope_scaling": ["type": "dynamic", "factor": 2.0]]
        XCTAssertNil(NFKMLXInternLM2.denseConfigurationJSON(json, sequencesStayBelowWindow: true)["rope_scaling"])
        XCTAssertNotNil(NFKMLXInternLM2.denseConfigurationJSON(json, sequencesStayBelowWindow: false)["rope_scaling"])
    }
}

final class NFKMLXSa2VAQwenTests: XCTestCase {
    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// STRUCTURE: the SAM 3 grounding against `Sa2VA-Qwen3-VL-4B-SAM3`'s own safetensors headers
    /// (`shapes.py` under `IK_SHAPES_ROOT`): every parameter is supplied, every tensor the map keeps lands
    /// on a parameter of its shape, and the map drops only the detector's own neck and the memory path.
    func testTheSAM3GroundingMatchesTheReleasedShapes() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        guard let root = NFKMLXValidationConfig.environment["IK_SHAPES_ROOT"],
              let data = FileManager.default.contents(
                atPath: root + "/sa2va-qwen3-vl-4b-sam3/shapes.json"),
              let released = try JSONSerialization.jsonObject(with: data) as? [String: [Int]] else {
            throw XCTSkip("run shapes.py ByteDance/Sa2VA-Qwen3-VL-4B-SAM3 into IK_SHAPES_ROOT")
        }
        let prefix = "grounding_encoder.sam2_model."
        let encoder = NFKSa2VASAM3GroundingEncoder()
        let built = Dictionary(uniqueKeysWithValues: encoder.parameters().flattened().map { ($0.0, $0.1.shape) })
        let grounding = released.filter { $0.key.hasPrefix(prefix) }
        let mapped = NFKSa2VASAM3GroundingEncoder.mapped(grounding.map { ($0.key, MLXArray.zeros($0.value)) })

        let supplied = Set(mapped.map(\.0))
        XCTAssertEqual(built.keys.filter { !supplied.contains($0) }.sorted(), [], "parameters no tensor supplies")
        for (name, value) in mapped {
            XCTAssertEqual(value.shape, built[name], "\(name) lands on no parameter of its shape")
        }
        let dropped = grounding.keys.filter { NFKSa2VASAM3GroundingEncoder.moduleKey(String($0.dropFirst(prefix.count))) == nil }
        let families = Set(dropped.map { key -> String in
            let parts = key.dropFirst(prefix.count).split(separator: ".")
            return parts.prefix(parts.first == "backbone" ? 3 : 1).joined(separator: ".")
        })
        print("VALIDATION STRUCTURE sa2va QWEN3_VL_4B_SAM3: \(grounding.count) grounding tensors, "
              + "\(mapped.count) mapped, \(dropped.count) dropped from \(families.sorted())")
        // The detector's own neck (`convs`) and the tracker's memory path (its memory attention is
        // `transformer`), which one conditioning frame never reads.
        XCTAssertTrue(families.isSubset(of: ["backbone.vision_backbone.convs", "transformer", "maskmem_backbone",
                                             "maskmem_tpos_enc", "memory_attention", "memory_encoder",
                                             "no_mem_pos_enc", "obj_ptr_proj", "obj_ptr_tpos_proj",
                                             "no_obj_embed_spatial", "no_obj_ptr", "mask_downsample"]),
                      "unexpected tensors dropped: \(families.sorted())")
    }


    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        let a = mine.asType(.float32).reshaped([-1]).asArray(Float.self)
        let b = reference.asType(.float32).reshaped([-1]).asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    /// PARITY on the Qwen-VL releases (Qwen3-VL and Qwen2.5-VL), each against its own
    /// `run_reference.py sa2va_qwen` record: the
    /// processor's grid at the reference's pixel bounds, the prompt the backend builds, the merged vision
    /// tokens and deepstack, greedy generation, the final hidden states, the `[SEG]` embedding, and the
    /// mask.
    func testEveryQwenVLReleaseIsAtParity() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        let env = NFKMLXValidationConfig.environment
        var measured = [String]()
        // `IK_SA2VA_ONLY=<name>` measures one release, so a large one runs without the others beside it.
        let only = ProcessInfo.processInfo.environment["IK_SA2VA_ONLY"]
        for name in ["QWEN3_VL_2B", "QWEN3_VL_4B", "QWEN3_VL_4B_SAM3", "QWEN2_5_VL_3B", "QWEN2_5_VL_7B_CUT4"]
        where only == nil || only == name {
            guard let directory = env["IK_VAL_SA2VA_\(name)"], let recordPath = env["IK_PARITY_SA2VA_\(name)"],
                  FileManager.default.fileExists(atPath: recordPath) else { continue }
            let (seams, iou, generation) = try measure(name, directory: directory, recordPath: recordPath, dtype: .float32)
            print("VALIDATION PARITY sa2va \(name): " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", ")
                  + ", IoU \(iou), generation \(generation)")
            for (seam, similarity) in seams { XCTAssertGreaterThan(similarity, 0.9999, "\(name) \(seam) diverges") }
            XCTAssertGreaterThan(iou, 0.999, "\(name) mask decision")
            NFKMLXGPU.clearCache()
            print("VALIDATION MEMORY sa2va \(name) released: active \(NFKMLXGPU.activeMemory >> 20) MB, "
                  + "cache \(NFKMLXGPU.cacheMemory >> 20) MB")
            measured.append(name)
        }
        try XCTSkipIf(measured.isEmpty, "set IK_VAL_SA2VA_QWEN3_VL_<2B|4B> and IK_PARITY_SA2VA_QWEN3_VL_<2B|4B>")
    }

    /// The seams of one Qwen-VL release against its record, the network's weights at `dtype`. Two phases,
    /// each holding one part: the vision tower and decoder, then the bridge and the grounding encoder, which
    /// read only the `[SEG]` hidden state. A float32 4B decoder beside SAM 3's 1008-pixel trunk pages on a
    /// 32 GB machine.
    private func measure(_ name: String, directory: String, recordPath: String, dtype: DType) throws
        -> (seams: [(String, Float)], iou: Float, generation: String) {
        let url = URL(fileURLWithPath: directory)
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let (languageSeams, generation, segHidden) = try autoreleasepool {
            () throws -> ([(String, Float)], String, MLXArray) in
            let net = try NFKMLXSa2VAQwenNet.load(directoryURL: url, parts: .language, dtype: dtype)
            let gridValues = rec["image_grid_thw"]!.reshaped([-1]).asArray(Int32.self).map(Int.init)
            let grid = (t: gridValues[0], h: gridValues[1], w: gridValues[2])
            let processor = net.imageProcessor
            let resized = processor.smartResize(height: 448, width: 448)
            XCTAssertEqual([resized.height, resized.width], [grid.h * processor.patchSize, grid.w * processor.patchSize],
                           "\(name) processor grid")

            let inputIds = rec["input_ids"]!.asArray(Int32.self).map(Int.init)
            let tokenizer = try XCTUnwrap(NFKMLXLanguage.releaseTokenizer(inDirectory: url))
            let prompt = net.promptText("Please segment the bright object.", imageTokens: grid.h * grid.w / 4)
            XCTAssertEqual(tokenizer.encode(prompt).map(\.intValue), inputIds, "\(name) prompt ids")

            let features = net.imageFeatures(pixelValues: rec["pixel_values"]!.asType(.float32), grid: grid)
            var seams = [("vision", cosine(features.output, rec["vision_merged"]!))]
            for (index, stack) in features.deepstack.enumerated() {
                seams.append(("deepstack\(index)", cosine(stack, rec["deepstack_\(index)"]!)))
            }
            // A cut release's record teacher-forces a fixed answer (`SA2VA_TEACHER=1`), so its greedy
            // output is not compared.
            let referenceGenerated = rec["generated"]!.asArray(Int32.self).map(Int.init)
            let endTokens: Set<Int> = [151_645, 151_643]
            var generation = "teacher-forced"
            if !name.hasSuffix("_CUT4") {
                let generated = net.generate(inputIds: inputIds, features: features, grid: grid,
                                             maximumTokens: referenceGenerated.count + 4, endTokens: endTokens)
                let expected = endTokens.contains(referenceGenerated.last ?? -1)
                    ? Array(referenceGenerated.dropLast()) : referenceGenerated
                if dtype == .float32 {
                    XCTAssertEqual(generated.tokens, expected, "\(name) greedy generation")
                }
                generation = generated.tokens == expected ? "token-exact" : "DIVERGES"
            }

            let full = inputIds + referenceGenerated
            // The record's `dec_last` is the reference's `hidden_states[-1]`, the states the bridge reads.
            let hidden = net.segmentationStates(inputIds: full, features: features, grid: grid)
            seams.append(("decoder", cosine(hidden[0], rec["dec_last"]!)))
            let seg = try XCTUnwrap(full.firstIndex(of: net.segmentationTokenId), "\(name) answers with [SEG]")
            // Copied out, so nothing the decoder computed outlives it.
            return (seams, generation, MLXArray(hidden[0, seg].asType(.float32).asArray(Float.self)))
        }
        NFKMLXGPU.clearCache()

        let (groundingSeams, iou) = try autoreleasepool { () throws -> ([(String, Float)], Float) in
            let net = try NFKMLXSa2VAQwenNet.load(directoryURL: url, parts: .grounding, dtype: dtype)
            let embedding = net.textHiddenFCS(segHidden.reshaped([1, -1]))
            let mask = net.segment(segHidden: segHidden,
                                   groundingImage: rec["g_pixel_values"]!.asType(.float32).transposed(0, 2, 3, 1))
            return ([("seg", cosine(embedding, rec["seg_embedding"]!)),
                     ("mask", cosine(mask.lowResolution, rec["low_res_best"]!))],
                    NFKMLXSa2VATests.maskIoU(mask.lowResolution, rec["low_res_best"]!))
        }
        return (languageSeams + groundingSeams, iou, generation)
    }

    /// The bfloat16 load the backend runs, against the same float32 records: the precision floor each
    /// release's backend answers at. `IK_SA2VA_ONLY=<name>` selects one release.
    func testTheBFloat16LoadStaysNearTheFloat32Reference() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        let env = NFKMLXValidationConfig.environment
        let only = ProcessInfo.processInfo.environment["IK_SA2VA_ONLY"]
        var measured = [String]()
        for name in ["QWEN3_VL_2B", "QWEN3_VL_4B", "QWEN3_VL_4B_SAM3", "QWEN2_5_VL_3B"] where only == nil || only == name {
            guard let directory = env["IK_VAL_SA2VA_\(name)"], let recordPath = env["IK_PARITY_SA2VA_\(name)"],
                  FileManager.default.fileExists(atPath: recordPath) else { continue }
            let (seams, iou, generation) = try measure(name, directory: directory, recordPath: recordPath, dtype: .bfloat16)
            print("VALIDATION BF16 sa2va \(name): " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", ")
                  + ", IoU \(iou), generation \(generation)")
            for (seam, similarity) in seams { XCTAssertGreaterThan(similarity, 0.99, "\(name) \(seam) at bfloat16") }
            XCTAssertGreaterThan(iou, 0.98, "\(name) mask decision at bfloat16")
            NFKMLXGPU.clearCache()
            measured.append(name)
        }
        try XCTSkipIf(measured.isEmpty, "set IK_VAL_SA2VA_QWEN3_VL_<2B|4B> and IK_PARITY_SA2VA_QWEN3_VL_<2B|4B>")
    }

    /// End to end through the public factory on every recorded full Qwen-VL release: the reference's
    /// decoded answer and a mask. Kept apart from the seams so a large release holds one network per run;
    /// `IK_SA2VA_ONLY=<name>` selects one release.
    func testEveryQwenVLReleaseAnswersThroughTheBackend() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        let env = NFKMLXValidationConfig.environment
        let only = ProcessInfo.processInfo.environment["IK_SA2VA_ONLY"]
        var measured = [String]()
        for name in ["QWEN3_VL_2B", "QWEN3_VL_4B", "QWEN3_VL_4B_SAM3", "QWEN2_5_VL_3B"] where only == nil || only == name {
            guard let directory = env["IK_VAL_SA2VA_\(name)"], let recordPath = env["IK_PARITY_SA2VA_\(name)"],
                  FileManager.default.fileExists(atPath: recordPath) else { continue }
            try NFKMLXSa2VATests.checkBackend(directory: directory, recordPath: recordPath, name: name)
            print("VALIDATION MEMORY sa2va \(name) backend released: active \(NFKMLXGPU.activeMemory >> 20) MB, "
                  + "cache \(NFKMLXGPU.cacheMemory >> 20) MB")
            measured.append(name)
        }
        try XCTSkipIf(measured.isEmpty, "set IK_VAL_SA2VA_QWEN3_VL_<2B|4B> and IK_PARITY_SA2VA_QWEN3_VL_<2B|4B>")
    }
}

final class NFKMLXQwen25VLVisionTests: XCTestCase {

    /// The windowed tower against transformers' own at a tiny seeded configuration
    /// (`run_reference.py qwen25vl_vision_tiny`): the window order and boundaries over a grid the windows
    /// do not divide, the rotary angles, and the merged output in raster order.
    func testTheWindowedTowerMatchesTheReference() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_QWEN25VL_VISION_TINY"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_QWEN25VL_VISION_TINY (run_reference.py qwen25vl_vision_tiny)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let configuration = NFKMLXQwen25VLVisionConfiguration(json: [
            "depth": 4, "hidden_size": 64, "num_heads": 4, "intermediate_size": 96, "out_hidden_size": 48,
            "patch_size": 14, "temporal_patch_size": 2, "spatial_merge_size": 2, "window_size": 112,
            "fullatt_block_indexes": [1, 3]])
        let grid = (t: 1, h: 20, w: 28)
        let windows = NFKMLXQwen25VLVisionNet.windows(grid: grid, configuration: configuration)
        XCTAssertEqual(windows.order, rec["_window_index"]!.asArray(Int32.self).map(Int.init))
        let boundaries = windows.lengths.reduce(into: [0]) { $0.append($0.last! + $1 * 4) }
        XCTAssertEqual(boundaries, Array(Set(rec["_cu_window"]!.asArray(Int32.self).map(Int.init))).sorted())
        let angles = NFKMLXQwen25VLVisionNet.rotaryAngles(grid: grid, configuration: configuration)
        XCTAssertLessThan(abs(angles - rec["_rotary"]!).max().item(Float.self), 1e-4)

        let net = NFKMLXQwen25VLVisionNet(configuration)
        let weights = rec.filter { !$0.key.hasPrefix("_") && $0.key != "output" && $0.key != "input_image" }
            .map { key, value -> (String, MLXArray) in
                let named = key.replacingOccurrences(of: "merger.mlp.2.", with: "merger.mlp.1.")
                return (named, named == "patch_embed.proj.weight" ? value.reshaped([value.dim(0), -1]) : value)
            }
        try NFKMLXWeights.apply(weights, to: net, verifyShapes: true)
        let output = net(rec["_patches"]!, grid: grid)
        let a = output.reshaped([-1]).asArray(Float.self), b = rec["output"]!.reshaped([-1]).asArray(Float.self)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        print("VALIDATION PARITY qwen2.5-vl vision tiny: output \(cosine)")
        XCTAssertGreaterThan(cosine, 0.99999)
    }
}

final class NFKMLXInternViT6BTests: XCTestCase {

    /// InternViT-6B's switches (RMSNorm blocks, query/key RMS normalization across all heads, no qkv
    /// bias) against the release's own `modeling_intern_vit.py` at a tiny seeded configuration
    /// (`run_reference.py internvit_qknorm_tiny`).
    func testTheQueryKeyNormalizedTowerMatchesTheReference() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_INTERNVIT_QKNORM_TINY"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_INTERNVIT_QKNORM_TINY (run_reference.py internvit_qknorm_tiny)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        var configuration = NFKMLXSa2VAConfiguration(
            visionHiddenSize: 64, visionLayers: 2, visionHeads: 4, visionIntermediateSize: 96, patchSize: 14,
            imageSize: 28, qkvBias: false,
            decoder: .init(hiddenSize: 32, layerCount: 1, headCount: 2, keyValueHeadCount: 1, intermediateSize: 32,
                           vocabularySize: 16, ropeTheta: 10_000, rmsEpsilon: 1e-6, tiesWordEmbeddings: false,
                           attentionBias: false),
            imageContextTokenId: 1, segmentationTokenId: 2)
        configuration.visionRMSNorm = true
        configuration.visionQueryKeyNormalization = true
        let net = NFKMLXSa2VAVisionNet(configuration)
        let weights = rec.filter { !$0.key.hasPrefix("_") && $0.key != "output" && $0.key != "input_image" }
            .map { key, value in (key, key == "embeddings.patch_embedding.weight" ? value.transposed(0, 2, 3, 1) : value) }
        try NFKMLXWeights.apply(weights, to: net, verifyShapes: true)
        let output = net(rec["_pixels"]!)[0]
        let a = output.reshaped([-1]).asArray(Float.self), b = rec["output"]!.reshaped([-1]).asArray(Float.self)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        print("VALIDATION PARITY internvit qk-norm tiny: output \(cosine)")
        XCTAssertGreaterThan(cosine, 0.99999)
    }
}

final class NFKMLXSa2VALLaVATests: XCTestCase {
    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }


    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        let a = mine.reshaped([-1]).asArray(Float.self), b = reference.reshaped([-1]).asArray(Float.self)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    /// PARITY on Sa2VA-LLaVA-1.5-7B cut to its first four decoder layers, against the release's own code
    /// (`run_reference.py sa2va_llava_teacher`): the processor on the plate's bytes (PIL bicubic to 336,
    /// CLIP normalization), the prompt ids and the answer's decode, the tower's second-to-last hidden
    /// state, the projected features, the fusion, the decoder over the prompt and a fixed answer, the
    /// last logits, the `[SEG]` embedding, and the mask.
    func testTheCutReleaseIsAtParity() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_SA2VA_LLAVA_CUT4"], let path = env["IK_PARITY_SA2VA_LLAVA_CUT4"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_SA2VA_LLAVA_CUT4 and IK_PARITY_SA2VA_LLAVA_CUT4 (run_reference.py sa2va_llava_teacher)")
        }
        let url = URL(fileURLWithPath: directory)
        let net = try NFKMLXSa2VALLaVANet.load(directoryURL: url)
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays

        let rgb = rec["input_rgb"]!
        let image = try NFKMLXTrOCRTests.cgImage(rgb: rgb.asArray(Int32.self).map { UInt8($0) },
                                                 width: rgb.dim(1), height: rgb.dim(0))
        let processed = try NFKMLXSa2VALLaVA.pixelValues(image)
        let pixelError = abs(processed[0] - rec["pixel_values"]!).max().item(Float.self)
        XCTAssertLessThan(pixelError, 1e-5, "processor")

        let tokenizer = try NFKMLXInternLM2Tokenizer(directory: url, decoding: .llamaFast)
        let ids = rec["input_ids"]!.reshaped([-1]).asArray(Int32.self).map(Int.init)
        let promptLength = Int(rec["prompt_length"]!.item(Int32.self))
        let prompt = String(decoding: rec["prompt_utf8"]!.asArray(Int32.self).map { UInt8($0) }, as: UTF8.self)
        XCTAssertEqual(tokenizer.encode(prompt).map(\.intValue), Array(ids.prefix(promptLength)), "prompt ids")
        let answer = String(decoding: rec["answer_decoded_utf8"]!.asArray(Int32.self).map { UInt8($0) }, as: UTF8.self)
        XCTAssertEqual(tokenizer.decodeIds(Array(ids.dropFirst(promptLength))), answer, "answer decode")

        let pixels = rec["pixel_values"]!.reshaped([1, 336, 336, 3])
        let layers = net.visionConfiguration.layers - 1
        let hidden = net.vision.hiddenState(pixels, afterLayers: layers)
        let features = net.imageFeatures(pixels)
        let fused = net.fusedEmbeddings(inputIds: ids, imageFeatures: features)
        let decoded = net.language.hiddenStates(fromEmbeddings: fused)
        let logits = net.language.logits(fromHidden: decoded[0..., (decoded.dim(1) - 1)...])
        let seg = try XCTUnwrap(ids.firstIndex(of: net.segmentationTokenId))
        let embedding = net.textHiddenFCS(decoded[0, seg].reshaped([1, -1]))
        let mask = net.segment(segHidden: decoded[0, seg],
                               groundingImage: rec["g_pixel_values"]!.transposed(0, 2, 3, 1))
        let seams = [("vision", cosine(hidden[0], rec["vision_hidden"]!)), ("features", cosine(features, rec["features"]!)),
                     ("fusion", cosine(fused, rec["fused"]!)), ("decoder", cosine(decoded[0], rec["dec_last"]!)),
                     ("logits", cosine(logits[0, 0], rec["last_logits"]!)), ("seg", cosine(embedding, rec["seg_embedding"]!)),
                     ("mask", cosine(mask.lowResolution, rec["low_res_best"]!))]
        let iou = NFKMLXSa2VATests.maskIoU(mask.lowResolution, rec["low_res_best"]!)
        let flips = NFKMLXSa2VATests.maskFlips(mask.lowResolution, rec["low_res_best"]!)
        print("VALIDATION PARITY sa2va LLAVA_CUT4: pixels max abs \(pixelError), "
              + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", ")
              + ", IoU \(iou) (\(flips.count) pixels flipped, reference |logit| at most \(flips.largestReference))")
        for (seam, similarity) in seams { XCTAssertGreaterThan(similarity, 0.9999, "\(seam) diverges") }
        XCTAssertGreaterThan(iou, 0.999, "mask decision")
    }

    /// The LLaVA-1.5 layout Sa2VA-LLaVA wraps, against transformers' own at a tiny seeded configuration
    /// (`run_reference.py llava_tiny`): the CLIP tower's second-to-last hidden state, the projected
    /// features without the class token, and the decoder's final hidden states over a sequence whose
    /// four image tokens take the features.
    func testTheLLaVALayoutMatchesTheReference() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_LLAVA_TINY"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_LLAVA_TINY (run_reference.py llava_tiny)")
        }
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let vision = NFKLLaVAVisionConfiguration(json: ["hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                                                        "num_attention_heads": 4, "image_size": 28, "patch_size": 14])
        let tower = NFKLLaVAVisionModel(vision)
        let projector = NFKLLaVAProjector(visionWidth: 32, decoderWidth: 48)
        let decoder = NFKMLXLanguageNet(try NFKMLXLanguage.configuration(fromJSON: [
            "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 48, "intermediate_size": 96,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 64,
            "rms_norm_eps": 1e-6, "rope_theta": 10_000, "tie_word_embeddings": false]))
        func subtree(_ prefix: String, into module: Module, rename: (String) -> String = { $0 }) throws {
            let weights = rec.compactMap { key, value -> (String, MLXArray)? in
                guard key.hasPrefix(prefix) else { return nil }
                let name = rename(String(key.dropFirst(prefix.count)))
                return (name, name == "embeddings.patch_embedding.weight" ? value.transposed(0, 2, 3, 1) : value)
            }
            try NFKMLXWeights.apply(weights, to: module, verifyShapes: true)
        }
        try subtree("model.vision_tower.vision_model.", into: tower)
        try subtree("model.multi_modal_projector.", into: projector)
        try NFKMLXWeights.apply(rec.compactMap { key, value in
            key.hasPrefix("model.language_model.") ? ("model." + key.dropFirst("model.language_model.".count), value)
                : key == "lm_head.weight" ? (key, value) : nil
        }, to: decoder, verifyShapes: true)

        let hidden = tower.hiddenState(rec["_pixels"]!, afterLayers: 1)
        let features = projector(hidden[0..., 1...])
        let ids = rec["_ids"]!.asArray(Int32.self).map(Int.init)
        let text = decoder.embed(MLXArray(ids.map(Int32.init)).reshaped([1, ids.count]))[0]
        var index = [Int32](repeating: 0, count: ids.count), mask = [Float](repeating: 0, count: ids.count)
        var counter: Int32 = 0
        for (position, id) in ids.enumerated() where id == 60 { index[position] = counter; counter += 1; mask[position] = 1 }
        let fused = MLX.where(MLXArray(mask).reshaped([ids.count, 1]) .> 0,
                              features.reshaped([-1, 48]).take(MLXArray(index), axis: 0), text)
        let output = decoder.hiddenStates(fromEmbeddings: fused.reshaped([1, ids.count, 48]))
        let seams = [("vision", cosine(hidden[0], rec["_vision_hidden"]!)), ("features", cosine(features, rec["_features"]!)),
                     ("decoder", cosine(output[0], rec["output"]!))]
        print("VALIDATION PARITY llava tiny: " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        for (seam, similarity) in seams { XCTAssertGreaterThan(similarity, 0.99999, "\(seam) diverges") }
    }
}

/// Sa2VA's fine-tuning recipe against the authors' own training code (`run_reference.py sa2va_loss`,
/// `IK_PARITY_SA2VA_1B_LOSS`): the mask terms and point selection on the reference's masks, mmengine's
/// schedule, the whole objective through the port's forward, the trained set, and the merged network's
/// save and factory reload.
final class NFKMLXSa2VATrainingTests: XCTestCase {

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func release() throws -> (directory: URL, record: [String: MLXArray]) {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_SA2VA_1B"], let path = env["IK_PARITY_SA2VA_1B_LOSS"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_SA2VA_1B and IK_PARITY_SA2VA_1B_LOSS (run_reference.py sa2va_loss)")
        }
        return (URL(fileURLWithPath: directory), try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays)
    }

    private static func example(_ record: [String: MLXArray]) -> NFKMLXSa2VAExample {
        NFKMLXSa2VAExample(pixelValues: record["pixel_values"]!, inputIds: record["input_ids"]!, labels: record["labels"]!,
                           groundingImage: record["g_pixel_values"]!.transposed(0, 2, 3, 1),
                           masks: record["gt_masks"]![0 ..< 1])
    }

    /// PARITY: on the reference's own mask logits, targets, and points, mmdet's point-sampled
    /// cross-entropy and dice; and the selection from the reference's own draws scores the same.
    func testTheMaskTermsAndPointSelectionAreTheReferences() throws {
        let (_, record) = try release()
        let objective = NFKMLXSa2VAObjective()
        let masks = record["pred_masks"]!
        let targets = NFKMLXSa2VAObjective.resizedTargets(record["gt_masks"]!, side: 256)
        let given = objective.maskTerms(masks: masks, targets: targets, points: record["points"]!)
        let selected = objective.uncertainPoints(masks: masks, candidates: record["point_candidates"]!,
                                                 random: record["point_random"]!)
        let reselected = objective.maskTerms(masks: masks, targets: targets, points: selected)
        let (mask, dice) = (record["loss_mask"]!.item(Float.self), record["loss_dice"]!.item(Float.self))
        print("VALIDATION PARITY sa2va 1B mask terms: mask \(given.mask.item(Float.self) * 2) vs \(mask), "
              + "dice \(given.dice.item(Float.self) * 0.5) vs \(dice); reselected mask "
              + "\(reselected.mask.item(Float.self) * 2), dice \(reselected.dice.item(Float.self) * 0.5)")
        XCTAssertEqual(given.mask.item(Float.self) * 2, mask, accuracy: 1e-5)
        XCTAssertEqual(given.dice.item(Float.self) * 0.5, dice, accuracy: 1e-5)
        XCTAssertEqual(reselected.mask.item(Float.self) * 2, mask, accuracy: 1e-5)
        XCTAssertEqual(reselected.dice.item(Float.self) * 0.5, dice, accuracy: 1e-5)
    }

    /// PARITY: `mmengineWarmupCosine` against mmengine's own `LinearParamScheduler` warm-up and
    /// `CosineAnnealingParamScheduler`, over 40- and 200-iteration runs.
    func testTheScheduleIsMMEngines() throws {
        let (_, record) = try release()
        for (steps, key) in [(40, "lr_40"), (200, "lr_200")] {
            let schedule = NFKMLXLearningRateSchedule.mmengineWarmupCosine(steps: steps)
            for (step, rate) in record[key]!.asArray(Float.self).enumerated() {
                XCTAssertEqual(4e-5 * schedule.multiplier(step), rate, accuracy: max(abs(rate) * 1e-5, 1e-12),
                               "\(steps) steps, step \(step)")
            }
        }
    }

    /// PARITY: the whole objective through the port's own forward on the release at float32: the
    /// language loss against the reference's in float64, the masks, and the mask terms on the reference's
    /// points.
    func testTheObjectiveThroughThePortsForward() throws {
        let (directory, record) = try release()
        let net = try NFKMLXSa2VA.network(directoryURL: directory)
        let parts = NFKMLXSa2VAObjective().terms(net, Self.example(record), points: record["points"]!)
        let masks = try XCTUnwrap(parts.masks)
        let a = masks.reshaped([-1]), b = record["pred_masks"]!.reshaped([-1])
        let maskCosine = ((a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))).item(Float.self)
        print("VALIDATION PARITY sa2va 1B objective: language \(parts.language.item(Float.self)) vs float64 "
              + "\(record["llm_loss_f64"]!.item(Float.self)) (float32 \(record["llm_loss"]!.item(Float.self))), masks "
              + "\(maskCosine), mask \(parts.mask.item(Float.self) * 2) vs \(record["loss_mask"]!.item(Float.self)), "
              + "dice \(parts.dice.item(Float.self) * 0.5) vs \(record["loss_dice"]!.item(Float.self))")
        XCTAssertEqual(parts.language.item(Float.self), record["llm_loss_f64"]!.item(Float.self), accuracy: 1e-4)
        XCTAssertGreaterThan(maskCosine, 0.9999)
        XCTAssertEqual(parts.mask.item(Float.self) * 2, record["loss_mask"]!.item(Float.self), accuracy: 1e-4)
        XCTAssertEqual(parts.dice.item(Float.self) * 0.5, record["loss_dice"]!.item(Float.self), accuracy: 1e-4)
    }

    /// The reference's trained set, a short run, and the merged network saved and reloaded through the
    /// factory with the same language loss.
    func testAFineTuneTrainsTheReferencesSetAndReloads() throws {
        let (directory, record) = try release()
        let net = try NFKMLXSa2VA.network(directoryURL: directory)
        let example = Self.example(record)
        let visionBefore = net.vision.parameters().flattened().first!.1 + 0
        eval(visionBefore)
        let losses = try NFKMLXSa2VA.fineTune(net, examples: { _ in example }, rank: 8, alpha: 16,
                                              optimizer: NFKMLXReferenceOptimizers.adamW(learningRate: 1e-5, weightDecay: 0.05),
                                              steps: 2)
        XCTAssertTrue(losses.allSatisfy(\.isFinite))
        let trained = Set(net.trainableParameters().flattened().map(\.0))
        let groups = Set(trained.map { name -> String in
            if name.contains("lora_") { return "lora" }
            return ["language_model.model.embed_tokens", "language_model.lm_head", "mlp1", "text_hidden_fcs",
                    "grounding_encoder.sam2_model.sam_mask_decoder"].first { name.hasPrefix($0) } ?? name
        })
        XCTAssertTrue(groups.isSubset(of: ["lora", "language_model.model.embed_tokens", "language_model.lm_head", "mlp1",
                                           "text_hidden_fcs", "grounding_encoder.sam2_model.sam_mask_decoder"]),
                      "untrained groups: \(groups.sorted())")
        XCTAssertTrue(groups.isSuperset(of: ["lora", "language_model.model.embed_tokens", "mlp1", "text_hidden_fcs",
                                             "grounding_encoder.sam2_model.sam_mask_decoder"]))
        XCTAssertEqual(abs(net.vision.parameters().flattened().first!.1 - visionBefore).max().item(Float.self), 0,
                       "the vision tower stays frozen")

        XCTAssertGreaterThan(try NFKMLXLoRA.merge(into: net), 0)
        let saved = FileManager.default.temporaryDirectory.appendingPathComponent("sa2va-tuned-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: saved) }
        try NFKMLXSa2VA.save(net, toDirectoryURL: saved, release: directory)
        let reloaded = try NFKMLXSa2VA.network(directoryURL: saved)
        let objective = NFKMLXSa2VAObjective()
        let before = objective.terms(net, example, points: record["points"]!)
        let after = objective.terms(reloaded, example, points: record["points"]!)
        XCTAssertEqual(after.language.item(Float.self), before.language.item(Float.self), accuracy: 1e-4)
        XCTAssertEqual(after.mask.item(Float.self), before.mask.item(Float.self), accuracy: 1e-4)
        XCTAssertNoThrow(try NFKMLXSa2VA.backend(directoryURL: saved))
    }
}

/// The Sa2VA recipe on the Qwen-VL and LLaVA releases: an example built through each family's own
/// processor, a short LoRA run, the trained set the reference's wrapper leaves trainable (the merger and
/// LLaVA's projector frozen), and the merged network's save and factory reload.
final class NFKMLXSa2VAFamilyTrainingTests: XCTestCase {

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// A release directory, once its parity record exists: the record is written after the release is
    /// complete, so a directory still downloading is skipped.
    private func directory(_ name: String) throws -> URL {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        let env = NFKMLXValidationConfig.environment
        guard let path = env["IK_VAL_SA2VA_\(name)"], let record = env["IK_PARITY_SA2VA_\(name)"],
              FileManager.default.fileExists(atPath: record) else {
            throw XCTSkip("set IK_VAL_SA2VA_\(name) and record IK_PARITY_SA2VA_\(name)")
        }
        return URL(fileURLWithPath: path)
    }

    /// The plate's bright disk as a mask `[1, 448, 448]`.
    private static var disk: MLXArray {
        let size = 448
        var values = [Float](repeating: 0, count: size * size)
        for y in 0 ..< size {
            for x in 0 ..< size {
                let dx = Double(x) - Double(size) * 0.62, dy = Double(y) - Double(size) * 0.40
                values[y * size + x] = dx * dx + dy * dy < pow(Double(size) * 0.16, 2) ? 1 : 0
            }
        }
        return MLXArray(values).reshaped([1, size, size])
    }

    /// Prompt and answer ids with the prompt masked, from the release's own tokenizer.
    private static func sequence(prompt: String, answer: String, directory: URL) throws -> (ids: MLXArray, labels: MLXArray) {
        let tokenizer = try XCTUnwrap(NFKMLXSa2VA.tokenizer(inDirectory: directory))
        let promptIds = tokenizer.encode(prompt).map(\.int32Value)
        let answerIds = tokenizer.encode(answer).map(\.int32Value)
        return (MLXArray(promptIds + answerIds), MLXArray([Int32](repeating: -100, count: promptIds.count) + answerIds))
    }

    private static func groups(_ names: Set<String>, language: String) -> Set<String> {
        Set(names.map { name -> String in
            if name.contains("lora_") { return "lora" }
            return ["\(language).model.embed_tokens", "\(language).lm_head", "text_hidden_fcs",
                    "grounding_encoder.sam2_model.sam_mask_decoder", "grounding_encoder.sam_mask_decoder"]
                .first { name.hasPrefix($0) } ?? name
        })
    }

    func testAQwenVLFineTuneTrainsTheReferencesSetAndReloads() throws {
        let release = try directory("QWEN3_VL_2B")
        let saved = FileManager.default.temporaryDirectory.appendingPathComponent("sa2va-qwen-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: saved) }
        let objective = NFKMLXSa2VAObjective()
        let points = MLXRandom.uniform(low: 0, high: 1, [5, 12_544, 2])
        eval(points)

        // One float32 network at a time: the trained network is released before the saved one loads,
        // because the two side by side with the optimizer's state exceed a 32 GB machine.
        let (example, before) = try { () throws -> (NFKMLXSa2VAExample, (language: Float, mask: Float)) in
            let net = try NFKMLXSa2VAQwenNet.load(directoryURL: release)
            let image = NFKMLXSa2VATests.plate()
            let (pixels, grid) = net.imageProcessor.process(image)
            let text = net.promptText("Please segment the bright object.", imageTokens: grid.h * grid.w / 4)
            let (ids, labels) = try Self.sequence(prompt: text, answer: "Sure, [SEG].<|im_end|>", directory: release)
            let example = NFKMLXSa2VAExample(
                pixelValues: pixels, inputIds: ids, labels: labels,
                groundingImage: try NFKMLXSa2VAProcessor.groundingPixels(image, side: net.groundingSize).transposed(0, 2, 3, 1),
                masks: Self.disk, grid: MLXArray([Int32(grid.t), Int32(grid.h), Int32(grid.w)]))
            let losses = try NFKMLXSa2VA.fineTune(net, examples: { _ in example }, rank: 8, alpha: 16,
                                                  optimizer: NFKMLXReferenceOptimizers.adamW(learningRate: 1e-5, weightDecay: 0.05),
                                                  steps: 1)
            XCTAssertTrue(losses.allSatisfy(\.isFinite))
            let groups = Self.groups(Set(net.trainableParameters().flattened().map(\.0)), language: "decoder")
            XCTAssertTrue(groups.isSubset(of: ["lora", "decoder.model.embed_tokens", "decoder.lm_head", "text_hidden_fcs",
                                               "grounding_encoder.sam2_model.sam_mask_decoder", "grounding_encoder.sam_mask_decoder"]),
                          "untrained groups: \(groups.sorted())")
            XCTAssertTrue(groups.contains("lora") && groups.contains("decoder.model.embed_tokens") && groups.contains("text_hidden_fcs"))
            XCTAssertTrue(groups.contains("decoder.lm_head"), "the release's untied head trains")

            try NFKMLXLoRA.merge(into: net)
            try NFKMLXSa2VA.save(net, toDirectoryURL: saved, release: release)
            let terms = objective.terms(net, example, points: points)
            return (example, (terms.language.item(Float.self), terms.mask.item(Float.self)))
        }()
        NFKMLXGPU.clearCache()

        try {
            let reloaded = try NFKMLXSa2VAQwenNet.load(directoryURL: saved)
            XCTAssertNotNil(reloaded.decoder.lmHead, "the saved head reloads untied")
            let after = objective.terms(reloaded, example, points: points)
            XCTAssertEqual(after.language.item(Float.self), before.language, accuracy: 1e-4)
            XCTAssertEqual(after.mask.item(Float.self), before.mask, accuracy: 1e-4)
        }()
        NFKMLXGPU.clearCache()
        XCTAssertNoThrow(try NFKMLXSa2VA.backend(directoryURL: saved))
    }

    func testALLaVAFineTuneTrainsTheReferencesSetAndReloads() throws {
        let release = try directory("LLAVA_CUT4")
        let net = try NFKMLXSa2VALLaVANet.load(directoryURL: release)
        let image = NFKMLXSa2VATests.plate()
        let grid = net.visionConfiguration.grid
        let text = NFKMLXSa2VALLaVA.promptText("Please segment the bright object.", imageTokens: grid * grid)
        let (ids, labels) = try Self.sequence(prompt: text, answer: " Sure, [SEG].</s>", directory: release)
        let example = NFKMLXSa2VAExample(
            pixelValues: try NFKMLXSa2VALLaVA.pixelValues(image, side: net.visionConfiguration.imageSize), inputIds: ids,
            labels: labels, groundingImage: try NFKMLXSa2VAProcessor.groundingPixels(image).transposed(0, 2, 3, 1),
            masks: Self.disk)
        let losses = try NFKMLXSa2VA.fineTune(net, examples: { _ in example }, rank: 8, alpha: 16,
                                              optimizer: NFKMLXReferenceOptimizers.adamW(learningRate: 1e-5, weightDecay: 0.05),
                                              steps: 1)
        XCTAssertTrue(losses.allSatisfy(\.isFinite))
        let groups = Self.groups(Set(net.trainableParameters().flattened().map(\.0)), language: "language_model")
        XCTAssertTrue(groups.isSubset(of: ["lora", "language_model.model.embed_tokens", "language_model.lm_head",
                                           "text_hidden_fcs", "grounding_encoder.sam2_model.sam_mask_decoder"]),
                      "untrained groups: \(groups.sorted())")
        XCTAssertFalse(net.trainableParameters().flattened().contains { $0.0.hasPrefix("projector.") }, "the projector stays frozen")

        try NFKMLXLoRA.merge(into: net)
        let saved = FileManager.default.temporaryDirectory.appendingPathComponent("sa2va-llava-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: saved) }
        try NFKMLXSa2VA.save(net, toDirectoryURL: saved, release: release)
        let reloaded = try NFKMLXSa2VALLaVANet.load(directoryURL: saved)
        let objective = NFKMLXSa2VAObjective()
        let points = MLXRandom.uniform(low: 0, high: 1, [5, 12_544, 2])
        let before = objective.terms(net, example, points: points), after = objective.terms(reloaded, example, points: points)
        XCTAssertEqual(after.language.item(Float.self), before.language.item(Float.self), accuracy: 1e-4)
    }
}
