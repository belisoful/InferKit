//
//  NFKMLXPhi4MMTests.swift
//  InferKitMLXTests
//
//  Phi-4-multimodal (`Phi4MMForCausalLM`, Microsoft): the Phi-4-mini decoder shared across three input
//  modes, measured on the RELEASED weights against the model's own `transformers` remote code
//  (`run_reference.py phi4mm`). The text path exercises the partial rotary and LongRoPE scaling and the
//  fused-projection split; the vision and speech paths add each tower, its projector, the placeholder
//  scatter, and the per-modality LoRA folded onto the decoder.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXPhi4MMTests: XCTestCase {

    override func tearDown() {
        // Each test loads multi-gigabyte networks; clearing the cache reaches MLX's runtime, which needs a
        // Metal library it can find.
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Double {
        eval(mine)
        let a = mine.reshaped([-1]).asArray(Float.self).map(Double.init)
        let b = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
        let n = min(a.count, b.count)
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-20)
    }

    // The released Phi-4-multimodal decoder in LANGUAGE mode (no adapter), against the reference's prefill
    // logits and greedy continuation on a text prompt. The logits are compared in log-softmax space
    // because the output head is a tied-embedding classifier whose raw logits carry a large shared
    // component (as with the other tied decoders).
    func testDecoderMatchesTheReferenceTextPath() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let path = env["IK_PARITY_PHI4MM"], let directory = env["IK_VAL_PHI4MM"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_PHI4MM + IK_VAL_PHI4MM (run_reference.py phi4mm, real weights)")
        }
        let arrays = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let tokens = arrays["text_tokens"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let referenceContinuation = arrays["text_continuation"]!.asType(.int32).asArray(Int32.self).map(Int.init)

        let decoder = try NFKMLXPhi4MM.decoder(directoryURL: URL(fileURLWithPath: directory), modality: .language)
        let logits = decoder(MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count]))[0]

        let mineLogProb = logits - logSumExp(logits, axis: -1, keepDims: true)
        let refLogits = arrays["text_logits"]!
        let refLogProb = refLogits - logSumExp(refLogits, axis: -1, keepDims: true)
        let similarity = cosine(mineLogProb, refLogProb)
        print("VALIDATION PARITY phi4mm: text logits (log-softmax) cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.999, "phi4mm text logits match the reference")

        let continuation = NFKMLXPhi4MM.generateText(
            decoder: decoder, inputIds: tokens, maxTokens: referenceContinuation.count, endTokens: [199999])
        XCTAssertEqual(continuation, referenceContinuation, "the greedy text continuation matches the reference")
    }

    // The released speech tower on the validation clip, seam by seam against the reference: the Conformer
    // encoder output, the projected audio embeddings, the fused decoder's first-step logits (with the
    // speech LoRA folded), and the greedy transcription.
    func testSpeechPathMatchesTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let path = env["IK_PARITY_PHI4MM"], let directory = env["IK_VAL_PHI4MM"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_PHI4MM + IK_VAL_PHI4MM (run_reference.py phi4mm, real weights)")
        }
        let arrays = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let url = URL(fileURLWithPath: directory)

        let inputAudio = arrays["speech_input_audio"]!            // [frames, mels]
        let mel = inputAudio.reshaped([1, inputAudio.dim(0), inputAudio.dim(1)])
        let audioNet = try NFKMLXPhi4MM.audioNet(directoryURL: url)

        let encoded = audioNet.encode(mel)[0]
        let encoderSimilarity = cosine(encoded, arrays["speech_audio_encoder"]!)
        print("VALIDATION PARITY phi4mm: speech encoder cosine \(encoderSimilarity)")
        XCTAssertGreaterThan(encoderSimilarity, 0.999, "phi4mm speech encoder matches the reference")

        let projected = audioNet.projected(mel, mode: .speech)[0]
        let projSimilarity = cosine(projected, arrays["speech_audio_proj"]!)
        print("VALIDATION PARITY phi4mm: speech projector cosine \(projSimilarity)")
        XCTAssertGreaterThan(projSimilarity, 0.999, "phi4mm speech projector matches the reference")

        let tokens = arrays["speech_tokens"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let decoder = try NFKMLXPhi4MM.decoder(directoryURL: url, modality: .speech)
        let hidden = NFKMLXPhi4MM.fusedHidden(decoder: decoder, inputIds: tokens,
                                              features: audioNet.projected(mel, mode: .speech),
                                              placeholder: NFKMLXPhi4MM.audioTokenId)
        let logits = decoder.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...]).reshaped([-1])
        let mineLogProb = logits - logSumExp(logits, axis: -1, keepDims: true)
        let refLogits = arrays["speech_logits_last"]!
        let refLogProb = refLogits - logSumExp(refLogits, axis: -1, keepDims: true)
        let logitSimilarity = cosine(mineLogProb, refLogProb)
        print("VALIDATION PARITY phi4mm: speech logits (log-softmax) cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.999, "phi4mm speech fused logits match the reference")

        let referenceContinuation = arrays["speech_continuation"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let continuation = NFKMLXPhi4MM.generateFused(
            decoder: decoder, inputIds: tokens, features: audioNet.projected(mel, mode: .speech),
            placeholder: NFKMLXPhi4MM.audioTokenId, maxTokens: referenceContinuation.count, endTokens: [199999])
        XCTAssertEqual(continuation, referenceContinuation, "the greedy transcription matches the reference")
    }

    // The released image tower on the validation photo, seam by seam against the reference: the SigLIP
    // penultimate patch features, the HD-transformed and projected image embeddings, the fused decoder's
    // first-step logits (with the vision LoRA folded), and the greedy caption.
    func testVisionPathMatchesTheReference() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let path = env["IK_PARITY_PHI4MM"], let directory = env["IK_VAL_PHI4MM"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_PHI4MM + IK_VAL_PHI4MM (run_reference.py phi4mm, real weights)")
        }
        let arrays = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let url = URL(fileURLWithPath: directory)

        // The recorded pixels are `[crops, 3, H, W]`; the SigLIP tower reads channels-last.
        let pixels = arrays["vision_input_image"]!.transposed(0, 2, 3, 1)
        let sizes = arrays["vision_image_sizes"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let imageNet = try NFKMLXPhi4MM.imageNet(directoryURL: url)

        let siglip = imageNet.processor(pixels)
        let siglipSimilarity = cosine(siglip.reshaped([-1, siglip.dim(2)]), arrays["vision_siglip_m2"]!)
        print("VALIDATION PARITY phi4mm: vision SigLIP cosine \(siglipSimilarity)")
        XCTAssertGreaterThan(siglipSimilarity, 0.999, "phi4mm SigLIP features match the reference")

        let projected = imageNet.projected(pixels: pixels, imageSize: (height: sizes[0], width: sizes[1]))
        let projSimilarity = cosine(projected, arrays["vision_img_proj"]!)
        print("VALIDATION PARITY phi4mm: vision projector cosine \(projSimilarity)")
        XCTAssertGreaterThan(projSimilarity, 0.999, "phi4mm image projector matches the reference")

        let tokens = arrays["vision_tokens"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let decoder = try NFKMLXPhi4MM.decoder(directoryURL: url, modality: .vision)
        let features = projected.reshaped([1, projected.dim(0), projected.dim(1)])
        let hidden = NFKMLXPhi4MM.fusedHidden(decoder: decoder, inputIds: tokens, features: features,
                                              placeholder: NFKMLXPhi4MM.imageTokenId)
        let logits = decoder.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...]).reshaped([-1])
        let mineLogProb = logits - logSumExp(logits, axis: -1, keepDims: true)
        let refLogits = arrays["vision_logits_last"]!
        let refLogProb = refLogits - logSumExp(refLogits, axis: -1, keepDims: true)
        let logitSimilarity = cosine(mineLogProb, refLogProb)
        print("VALIDATION PARITY phi4mm: vision logits (log-softmax) cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.999, "phi4mm vision fused logits match the reference")

        let referenceContinuation = arrays["vision_continuation"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let continuation = NFKMLXPhi4MM.generateFused(
            decoder: decoder, inputIds: tokens, features: features, placeholder: NFKMLXPhi4MM.imageTokenId,
            maxTokens: referenceContinuation.count, endTokens: [199999])
        XCTAssertEqual(continuation, referenceContinuation, "the greedy caption matches the reference")
    }

    // Every released tensor has exactly one destination under the rules the loaders use: the decoder
    // (base projections, both adapters, norms, embedding), the image tower, the speech tower, or the
    // SigLIP parts the penultimate-layer feature never reaches. The loaders' strict, shape-verified apply
    // covers each network's side; this covers the checkpoint's side from the shard index alone.
    func testEveryReleasedTensorIsAccountedFor() throws {
        guard let directory = NFKMLXValidationConfig.environment["IK_VAL_PHI4MM"] else {
            throw XCTSkip("set IK_VAL_PHI4MM")
        }
        let index = URL(fileURLWithPath: directory).appendingPathComponent("model.safetensors.index.json")
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: index)) as? [String: Any]
        let names = try XCTUnwrap(json?["weight_map"] as? [String: String]).keys.sorted()
        var decoder = 0, image = 0, audio = 0, unused = [String]()
        for name in names {
            let destinations = [NFKMLXPhi4MM.isDecoderTensor(name),
                                NFKMLXPhi4MM.imageModuleKey(forRelease: name) != nil,
                                NFKMLXPhi4MM.audioModuleKey(forRelease: name) != nil,
                                NFKMLXPhi4MM.isUnusedImageTensor(name)]
            XCTAssertEqual(destinations.filter { $0 }.count, 1, "\(name) has exactly one destination")
            if destinations[0] { decoder += 1 }
            if destinations[1] { image += 1 }
            if destinations[2] { audio += 1 }
            if destinations[3] { unused.append(name) }
        }
        print("VALIDATION PARITY phi4mm: structural \(names.count) tensors — decoder \(decoder), image \(image), "
              + "audio \(audio), unused \(unused.count)")
        // 32 layers of two norms and four projections, each a base weight plus two adapters' pairs.
        XCTAssertEqual(decoder, 2 + 32 * (2 + 4 * 5))
        // SigLIP's 27th layer (16 tensors), its post-layer-norm (2), and its pooling head (11).
        XCTAssertEqual(unused.count, 16 + 2 + 11, "unused: \(unused)")
        XCTAssertEqual(decoder + image + audio + unused.count, names.count)
    }

    private func record() throws -> [String: MLXArray] {
        let env = NFKMLXValidationConfig.environment
        guard let path = env["IK_PARITY_PHI4MM"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_PHI4MM (run_reference.py phi4mm)")
        }
        return try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
    }

    private func bytes(_ array: MLXArray) -> [UInt8] { array.asType(.int32).asArray(Int32.self).map { UInt8($0) } }

    // The SpeechLib filterbank against the reference processor's features on the same samples, and the
    // audio-token count it reserves.
    func testAudioFeaturesMatchTheReferenceProcessor() throws {
        try requireMLXRuntime()
        let arrays = try record()
        let samples = arrays["speech_waveform"]!.asArray(Float.self)
        let features = NFKMLXPhi4MMAudioFeatures.logMel(samples)
        XCTAssertEqual(features.shape, arrays["speech_input_audio"]!.shape)
        let similarity = cosine(features, arrays["speech_input_audio"]!)
        let largest = abs(features - arrays["speech_input_audio"]!).max().item(Float.self)
        print("VALIDATION PARITY phi4mm: audio features cosine \(similarity), max |Δ| \(largest)")
        XCTAssertGreaterThan(similarity, 0.99999, "the log-mel features match the reference processor")
        let reserved = arrays["speech_audio_embed_sizes"]!.asType(.int32).asArray(Int32.self)[0]
        XCTAssertEqual(NFKMLXPhi4MMAudioFeatures.tokenCount(frames: features.dim(0)), Int(reserved))
    }

    // The filterbank at every branch of the reference's sample-rate handling, on the same 16-bit samples:
    // 44.1 kHz (decimated by two, then read as 16 kHz), 48 kHz (decimated by three), 8 kHz (the 8 kHz
    // spectrum zero-filled to 16 kHz), and 11.025 kHz (read as 8 kHz), with the audio-token counts.
    func testAudioFeaturesMatchTheReferenceAtEveryRate() throws {
        try requireMLXRuntime()
        let arrays = try record()
        for rate in [44100, 48000, 8000, 11025] {
            let samples = arrays["rate\(rate)_waveform"]!.asArray(Float.self)
            let reference = arrays["rate\(rate)_features"]!
            let features = try NFKMLXPhi4MMAudioFeatures.logMel(samples, sampleRate: rate)
            XCTAssertEqual(features.shape, reference.shape, "\(rate) Hz frame count")
            let similarity = cosine(features, reference)
            let largest = abs(features - reference).max().item(Float.self)
            print("VALIDATION PARITY phi4mm: \(rate) Hz features cosine \(similarity), max |Δ| \(largest)")
            XCTAssertGreaterThan(similarity, 0.99999, "\(rate) Hz features match the reference processor")
            let reserved = Int(arrays["rate\(rate)_tokens"]!.asType(.int32).asArray(Int32.self)[0])
            XCTAssertEqual(NFKMLXPhi4MMAudioFeatures.tokenCount(frames: features.dim(0)), reserved, "\(rate) Hz tokens")
        }
        XCTAssertThrowsError(try NFKMLXPhi4MMAudioFeatures.logMel([Float](repeating: 0, count: 8000), sampleRate: 6000))
    }

    // The dynamic-HD preprocessor against the reference processor on the same decoded bytes: the square
    // photo (one crop, upscaled) and a 900×500 picture (a 2×3 crop grid with padded bottom rows).
    func testImageProcessorMatchesTheReferenceProcessor() throws {
        try requireMLXRuntime()
        let arrays = try record()
        for (prefix, rgbKey, pixelsKey) in [("vision", "vision_rgb", "vision_input_image"),
                                             ("pad", "pad_rgb", "pad_input_image")] {
            let rgb = arrays[rgbKey]!
            let input = NFKMLXPhi4MMImageProcessor.process(rgb: bytes(rgb), width: rgb.dim(1), height: rgb.dim(0))
            let reference = arrays[pixelsKey]!.transposed(0, 2, 3, 1)
            XCTAssertEqual(input.pixels.shape, reference.shape, "\(prefix) crop layout")
            let largest = abs(input.pixels - reference).max().item(Float.self)
            print("VALIDATION PARITY phi4mm: \(prefix) processor pixels max |Δ| \(largest)")
            XCTAssertLessThan(largest, 1e-4, "\(prefix) pixels match the reference processor")
            let tokens = arrays["\(prefix)_tokens"]!.asType(.int32).asArray(Int32.self)
            XCTAssertEqual(input.tokenCount, tokens.filter { Int($0) == NFKMLXPhi4MM.imageTokenId }.count,
                           "\(prefix) image-token count")
        }
        let padded = arrays["pad_rgb"]!
        let input = NFKMLXPhi4MMImageProcessor.process(rgb: bytes(padded), width: padded.dim(1), height: padded.dim(0))
        let sizes = arrays["pad_image_sizes"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        XCTAssertEqual([input.imageSize.height, input.imageSize.width], sizes)
        let mask = arrays["pad_image_attn_mask"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        XCTAssertEqual(referenceValidPatches(mask: mask, imageSize: input.imageSize).rows, input.validPatches.rows)
        XCTAssertEqual(referenceValidPatches(mask: mask, imageSize: input.imageSize).columns, input.validPatches.columns)
    }

    /// An sRGB `CGImage` holding exactly `rgb` (`[height, width, 3]`), so a backend reads the same bytes
    /// the reference processor did rather than a second JPEG decode.
    private func image(rgb: [UInt8], width: Int, height: Int) -> CGImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0 ..< (width * height) {
            rgba[pixel * 4] = rgb[pixel * 3]
            rgba[pixel * 4 + 1] = rgb[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = rgb[pixel * 3 + 2]
        }
        let context = CGContext(data: &rgba, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return context.makeImage()!
    }

    // The assembled model with both adapters live, from RAW inputs in every mode the release serves
    // (text, speech, vision, and vision with speech): the prompt ids the backend assembles equal the
    // reference processor's, the first answer token's logits match, and the greedy answers match. The
    // reference's own continuations end with the stop token it emits; the model stops before it.
    func testMixtureModelMatchesTheReferenceInEveryMode() throws {
        try requireMLXRuntime()
        let arrays = try record()
        guard let directory = NFKMLXValidationConfig.environment["IK_VAL_PHI4MM"] else {
            throw XCTSkip("set IK_VAL_PHI4MM")
        }
        let model = try NFKMLXPhi4MM.model(directoryURL: URL(fileURLWithPath: directory), precision: .float32)
        func ints(_ key: String) -> [Int] { arrays[key]!.asType(.int32).asArray(Int32.self).map(Int.init) }
        XCTAssertEqual(model.tokenizer.encode("Describe the image in one sentence.").map(\.intValue),
                       ints("tok_text_ids"), "the release tokenizer encodes like the reference")

        let rgb = arrays["vision_rgb"]!
        let photo = NFKMLXPhi4MMImageProcessor.process(rgb: bytes(rgb), width: rgb.dim(1), height: rgb.dim(0))
        let speech = arrays["speech_waveform"]!.asArray(Float.self)
        let cases: [(name: String, text: String, image: NFKMLXPhi4MMImageInput?, audio: [Float]?)] = [
            ("text", "What is the capital of France?", nil, nil),
            ("speech", "Transcribe the audio clip into text.", nil, speech),
            ("vision", "Describe the image in one sentence.", photo, nil),
            ("vs", "", photo, speech),
        ]
        for testCase in cases {
            let prepared = try model.prepare(text: testCase.text, image: testCase.image, audio: testCase.audio)
            XCTAssertEqual(prepared.ids, ints("\(testCase.name)_tokens"), "\(testCase.name) prompt ids")

            let logits = try model.firstLogits(text: testCase.text, image: testCase.image, audio: testCase.audio)
            let referenceKey = testCase.name == "text" ? "text_logits" : "\(testCase.name)_logits_last"
            var reference = arrays[referenceKey]!
            if testCase.name == "text" { reference = reference[reference.dim(0) - 1] }
            let similarity = cosine(logits - logSumExp(logits, axis: -1, keepDims: true),
                                    reference - logSumExp(reference, axis: -1, keepDims: true))
            print("VALIDATION PARITY phi4mm: mixture \(testCase.name) logits (log-softmax) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.999, "\(testCase.name) first-token logits match the reference")

            var expected = ints("\(testCase.name)_continuation")
            if let last = expected.last, NFKMLXPhi4MM.stopTokens.contains(last) { expected.removeLast() }
            let produced = try model.generate(text: testCase.text, image: testCase.image, audio: testCase.audio,
                                          maxTokens: expected.count + 1)
            XCTAssertEqual(Array(produced.prefix(expected.count)), expected, "\(testCase.name) greedy answer")
        }

        // The public backend, end to end: the validation clip as a WAV file transcribes exactly.
        let backend = NFKMLXPhi4MMBackend(model: model, identifier: NFKMLXPhi4MM.modelName)
        let clip = try Data(contentsOf: URL(fileURLWithPath: NSHomeDirectory() + "/.inferkit-validation/inputs/speech.wav"))
        let request = NFKInferenceRequest(inputs: [NFKInputAudio: clip], parameters: [:])
        let transcript = try backend.runInference(for: request).output(forKey: NFKOutputText) as? String
        print("VALIDATION PARITY phi4mm: backend transcription \(transcript.debugDescription)")
        let referenceText = String(decoding: bytes(arrays["speech_text"]!), as: UTF8.self)
        XCTAssertEqual(transcript, referenceText, "the backend transcribes the clip as the reference does")

        // And an image through the backend, from the same bytes the reference processor read.
        let picture = image(rgb: bytes(rgb), width: rgb.dim(1), height: rgb.dim(0))
        let captionRequest = NFKInferenceRequest(inputs: [NFKInputImage: picture,
                                                          NFKInputPrompt: "Describe the image in one sentence."],
                                                 parameters: [:])
        let caption = try backend.runInference(for: captionRequest).output(forKey: NFKOutputText) as? String
        print("VALIDATION PARITY phi4mm: backend caption \(caption.debugDescription)")
        XCTAssertEqual(caption, String(decoding: bytes(arrays["vision_text"]!), as: UTF8.self),
                       "the backend captions the photo as the reference does")
    }

    /// The HD image's unpadded patch extent from the reference's per-crop masks (global crop first).
    private func referenceValidPatches(mask: [Int], imageSize: (height: Int, width: Int)) -> (rows: Int, columns: Int) {
        let side = 32, h = imageSize.height / 448, w = imageSize.width / 448
        func tile(_ r: Int, _ c: Int, _ y: Int, _ x: Int) -> Int { mask[((1 + r * w + c) * side + y) * side + x] }
        let rows = (0 ..< h).reduce(0) { total, r in total + (0 ..< side).filter { tile(r, 0, $0, 0) == 1 }.count }
        let columns = (0 ..< w).reduce(0) { total, c in total + (0 ..< side).filter { tile(0, c, 0, $0) == 1 }.count }
        return (rows, columns)
    }

    // A padded multi-crop image through the released image tower, seam by seam: the NaViT SigLIP features
    // with stretched position ids and masked padding, the trimmed HD layout and projection, the fused
    // decoder's first-step logits, and the greedy caption. Pixels and masks are the reference's own.
    func testPaddedMultiCropImageMatchesTheReference() throws {
        try requireMLXRuntime()
        let arrays = try record()
        guard let directory = NFKMLXValidationConfig.environment["IK_VAL_PHI4MM"] else {
            throw XCTSkip("set IK_VAL_PHI4MM")
        }
        let url = URL(fileURLWithPath: directory)
        let pixels = arrays["pad_input_image"]!.transposed(0, 2, 3, 1)
        let sizes = arrays["pad_image_sizes"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let imageSize = (height: sizes[0], width: sizes[1])
        let mask = arrays["pad_image_attn_mask"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let valid = referenceValidPatches(mask: mask, imageSize: imageSize)
        let imageNet = try NFKMLXPhi4MM.imageNet(directoryURL: url)

        let h = imageSize.height / 448, w = imageSize.width / 448
        var regions: [(rows: Int, columns: Int)] = [(rows: 32, columns: 32)]
        for r in 0 ..< h {
            for c in 0 ..< w {
                regions.append((rows: min(max(valid.rows - r * 32, 0), 32), columns: min(max(valid.columns - c * 32, 0), 32)))
            }
        }
        let siglip = imageNet.processor(pixels, validRegions: regions)
        let siglipSimilarity = cosine(siglip.reshaped([-1, siglip.dim(2)]), arrays["pad_siglip_m2"]!)
        print("VALIDATION PARITY phi4mm: padded SigLIP cosine \(siglipSimilarity)")
        XCTAssertGreaterThan(siglipSimilarity, 0.999, "padded SigLIP features match the reference")

        let projected = imageNet.projected(pixels: pixels, imageSize: imageSize, validPatches: valid)
        XCTAssertEqual(projected.shape, arrays["pad_img_proj"]!.shape, "padded HD token layout")
        let projSimilarity = cosine(projected, arrays["pad_img_proj"]!)
        print("VALIDATION PARITY phi4mm: padded projector cosine \(projSimilarity)")
        XCTAssertGreaterThan(projSimilarity, 0.999, "padded image projector matches the reference")

        let tokens = arrays["pad_tokens"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let decoder = try NFKMLXPhi4MM.decoder(directoryURL: url, modality: .vision)
        let features = projected.reshaped([1, projected.dim(0), projected.dim(1)])
        let hidden = NFKMLXPhi4MM.fusedHidden(decoder: decoder, inputIds: tokens, features: features,
                                              placeholder: NFKMLXPhi4MM.imageTokenId)
        let logits = decoder.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...]).reshaped([-1])
        let mine = logits - logSumExp(logits, axis: -1, keepDims: true)
        let reference = arrays["pad_logits_last"]!
        let logitSimilarity = cosine(mine, reference - logSumExp(reference, axis: -1, keepDims: true))
        print("VALIDATION PARITY phi4mm: padded logits (log-softmax) cosine \(logitSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.999, "padded fused logits match the reference")

        let referenceContinuation = arrays["pad_continuation"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let continuation = NFKMLXPhi4MM.generateFused(
            decoder: decoder, inputIds: tokens, features: features, placeholder: NFKMLXPhi4MM.imageTokenId,
            maxTokens: referenceContinuation.count, endTokens: [199999])
        XCTAssertEqual(continuation, referenceContinuation, "the padded image's greedy caption matches the reference")
    }

    // MARK: Conversations, several pictures and clips, long clips

    // The release's chat template, as its processor applies it: `<|role|>content<|end|>` turns closed by
    // `<|assistant|>`, a system turn's tool block, whitespace absorbed after each `rstrip` marker, media
    // references replaced by their placeholder tokens in the order they appear, and unreferenced media
    // opening the first user turn.
    func testConversationRendersTheReleaseTemplate() throws {
        let conversation: [[String: Any]] = [
            ["role": "system", "content": "Be brief.", "tools": "[]"],
            ["role": "user", "content": "  \n Hello"],
            ["role": "assistant", "content": " Hi."],
            ["role": "user", "content": "And this?"],
        ]
        XCTAssertEqual(try NFKMLXPhi4MMPrompt.render(messages: conversation, images: 0, audios: 0),
                       "<|system|>Be brief.<|tool|>[]<|/tool|><|end|><|user|>Hello<|end|><|assistant|>Hi.<|end|>"
                           + "<|user|>And this?<|end|><|assistant|>")
        XCTAssertEqual(try NFKMLXPhi4MMPrompt.render(messages: conversation, images: 2, audios: 1),
                       "<|system|>Be brief.<|tool|>[]<|/tool|><|end|><|user|><|endoftext10|><|endoftext10|>"
                           + "<|endoftext11|>  \n Hello<|end|><|assistant|>Hi.<|end|><|user|>And this?<|end|><|assistant|>")

        let placed: [[String: Any]] = [["role": "user", "content": "First <|audio_2|> then <|image_7|><|audio_1|>"]]
        XCTAssertEqual(try NFKMLXPhi4MMPrompt.render(messages: placed, images: 1, audios: 2),
                       "<|user|>First <|endoftext11|> then <|endoftext10|><|endoftext11|><|end|><|assistant|>")
        XCTAssertThrowsError(try NFKMLXPhi4MMPrompt.render(messages: placed, images: 2, audios: 2),
                             "a conversation that names its pictures names every one")
        XCTAssertThrowsError(try NFKMLXPhi4MMPrompt.render(messages: [["role": "system", "content": "x"]], images: 1, audios: 0),
                             "media needs a user turn")
    }

    // Clips run through the speech tower as one padded batch, and past the encoder's window each unfolds
    // into windows; a short clip batched beside a long one has windows that are wholly padding. With a
    // window of 8 subsampled frames the tiny tower exercises both, and every output stays finite. A lone
    // clip reads the same through the batched entry as through the single one.
    func testWindowedBatchedSpeechTowerStaysFinite() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXPhi4MMAudioConfiguration.tiny
        configuration.attentionWindow = 8
        let net = NFKMLXPhi4MMAudioNet(configuration)
        let long = MLXRandom.normal([200, 80]), short = MLXRandom.normal([40, 80])
        let clips = net.projected(clips: [long, short], mode: .speech)
        XCTAssertEqual(clips.map { $0.dim(0) }, [25, 5])
        let joined = concatenated(clips, axis: 0)
        XCTAssertFalse((joined .!= joined).any().item(Bool.self), "no NaN from a window with every key masked")

        let single = net.projected(long.expandedDimensions(axis: 0), mode: .speech)[0]
        XCTAssertEqual(cosine(net.projected(clips: [long], mode: .speech)[0], single), 1, accuracy: 1e-9)
    }

    private func conversationRecord() throws -> [String: MLXArray] {
        let env = NFKMLXValidationConfig.environment
        guard let path = env["IK_PARITY_PHI4MM_CONVERSATION"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_PHI4MM_CONVERSATION (run_reference.py phi4mm_conversation)")
        }
        return try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
    }

    /// A mono 16-bit WAV holding `samples`, each an exact multiple of 1/32768 as a 16-bit file reads.
    private func wave(_ samples: [Float], sampleRate: Int = 16000) -> Data {
        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8)); append(UInt32(36 + samples.count * 2))
        data.append(contentsOf: Array("WAVEfmt ".utf8)); append(UInt32(16)); append(UInt16(1)); append(UInt16(1))
        append(UInt32(sampleRate)); append(UInt32(sampleRate * 2)); append(UInt16(2)); append(UInt16(16))
        data.append(contentsOf: Array("data".utf8)); append(UInt32(samples.count * 2))
        for sample in samples { append(Int16((sample * 32768).rounded())) }
        return data
    }

    // The released model on requests past one turn with one picture and one clip, against the
    // reference's processor and model: a multi-turn chat with a system turn and two pictures and two
    // clips, two clips transcribed together (batched under the padding mask, one past the 500-frame
    // window), and a 45 s clip alone (unfolded without a mask). Prompt ids, the speech tower's seams,
    // first-token logits, and greedy answers; then the chat through the backend's message, image, and
    // audio keys, and a seeded sampled run.
    func testConversationsAndClipsMatchTheReference() throws {
        try requireMLXRuntime()
        let arrays = try conversationRecord()
        guard let directory = NFKMLXValidationConfig.environment["IK_VAL_PHI4MM"] else {
            throw XCTSkip("set IK_VAL_PHI4MM")
        }
        let model = try NFKMLXPhi4MM.model(directoryURL: URL(fileURLWithPath: directory), precision: .float32)
        func ints(_ key: String) -> [Int] { arrays[key]!.asType(.int32).asArray(Int32.self).map(Int.init) }
        func picture(_ key: String) -> NFKMLXPhi4MMImageInput {
            let rgb = arrays[key]!
            return NFKMLXPhi4MMImageProcessor.process(rgb: bytes(rgb), width: rgb.dim(1), height: rgb.dim(0))
        }
        let shortSamples = arrays["short_waveform"]!.asArray(Float.self)
        let longSamples = arrays["long_waveform"]!.asArray(Float.self)
        let short = NFKMLXPhi4MMAudioInput(samples: shortSamples), long = NFKMLXPhi4MMAudioInput(samples: longSamples)
        let chat: [[String: Any]] = [
            ["role": "system", "content": "You answer in one short sentence."],
            ["role": "user", "content": "<|image_1|><|image_2|>How many pictures are there?"],
            ["role": "assistant", "content": "There are two pictures."],
            ["role": "user", "content": "\n <|audio_1|><|audio_2|>What does the second clip say?"],
        ]
        let transcribe = "Transcribe the audio clip into text."
        let cases: [(name: String, messages: [[String: Any]], images: [NFKMLXPhi4MMImageInput], audios: [NFKMLXPhi4MMAudioInput])] = [
            ("chat", chat, [picture("photo_rgb"), picture("wide_rgb")], [long, short]),
            ("clips", [["role": "user", "content": "<|audio_1|><|audio_2|>" + transcribe]], [], [long, short]),
            ("long", [["role": "user", "content": "<|audio_1|>" + transcribe]], [], [long]),
        ]
        for testCase in cases {
            let prepared = try model.prepare(messages: testCase.messages, images: testCase.images, audios: testCase.audios)
            XCTAssertEqual(prepared.ids, ints("\(testCase.name)_tokens"), "\(testCase.name) prompt ids")

            let logits = try model.firstLogits(messages: testCase.messages, images: testCase.images, audios: testCase.audios)
            let reference = arrays["\(testCase.name)_logits_last"]!
            let similarity = cosine(logits - logSumExp(logits, axis: -1, keepDims: true),
                                    reference - logSumExp(reference, axis: -1, keepDims: true))
            print("VALIDATION PARITY phi4mm: \(testCase.name) logits (log-softmax) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.999, "\(testCase.name) first-token logits match the reference")

            var expected = ints("\(testCase.name)_continuation")
            if let last = expected.last, NFKMLXPhi4MM.stopTokens.contains(last) { expected.removeLast() }
            var options = NFKMLXGenerationOptions()
            options.maxTokens = expected.count + 1
            let produced = try model.generate(messages: testCase.messages, images: testCase.images,
                                              audios: testCase.audios, options: options)
            XCTAssertEqual(Array(produced.prefix(expected.count)), expected, "\(testCase.name) greedy answer")
        }

        // The speech tower's seams: two clips batched (the short one's second window wholly padding), and
        // the long clip alone, unfolded with its padding visible to attention.
        let mels = try [long, short].map { try NFKMLXPhi4MMAudioFeatures.logMel($0.samples, sampleRate: $0.sampleRate) }
        let batched = model.audioNet.encode(clips: mels)
        let referenceBatched = arrays["clips_audio_encoder"]!
        for (index, mel) in mels.enumerated() {
            let frames = NFKMLXPhi4MMAudioFeatures.tokenCount(frames: mel.dim(0))
            let similarity = cosine(batched[index, 0 ..< frames], referenceBatched[index, 0 ..< frames])
            print("VALIDATION PARITY phi4mm: batched clip \(index) encoder cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "batched clip \(index) encoder output")
        }
        let alone = cosine(model.audioNet.encode(clips: [mels[0]])[0], arrays["long_audio_encoder"]!)
        print("VALIDATION PARITY phi4mm: 45 s clip encoder cosine \(alone)")
        XCTAssertGreaterThan(alone, 0.9999, "the unfolded 45 s clip's encoder output")
        let projected = model.audioNet.projected(clips: mels, mode: .vision)
        let referenceProjected = arrays["chat_audio_proj"]!
        let sizes = ints("chat_audio_embed_sizes")
        for index in 0 ..< 2 {
            let similarity = cosine(projected[index], referenceProjected[index, 0 ..< sizes[index]])
            print("VALIDATION PARITY phi4mm: chat clip \(index) vision-head projection cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "chat clip \(index) projection")
        }

        // The chat through the backend: the conversation under NFKInputMessages, the pictures under
        // NFKInputImage and NFKInputImages, the clips as WAV bytes under NFKInputAudio and NFKInputAudios.
        let backend = NFKMLXPhi4MMBackend(model: model, identifier: NFKMLXPhi4MM.modelName)
        func cgImage(_ key: String) -> CGImage {
            let rgb = arrays[key]!
            return image(rgb: bytes(rgb), width: rgb.dim(1), height: rgb.dim(0))
        }
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: chat, NFKInputImage: cgImage("photo_rgb"),
                                                   NFKInputImages: [cgImage("wide_rgb")],
                                                   NFKInputAudio: wave(longSamples), NFKInputAudios: [wave(shortSamples)]],
                                          parameters: [NFKParameterMaxTokens: 32])
        let result = try backend.runInference(for: request)
        let answer = result.output(forKey: NFKOutputText) as? String
        print("VALIDATION PARITY phi4mm: backend chat answer \(answer.debugDescription)")
        XCTAssertEqual(answer, String(decoding: bytes(arrays["chat_text"]!), as: UTF8.self),
                       "the backend answers the conversation as the reference does")
        let usage = result.output(forKey: NFKOutputUsage) as? [String: Any]
        XCTAssertEqual((usage?[NFKUsageInputTokens] as? NSNumber)?.intValue, ints("chat_tokens").count)

        // Sampling: a seeded run repeats, and temperature 0 is the greedy answer.
        let question: [[String: Any]] = [["role": "user", "content": "Name a color."]]
        var sampled = NFKMLXGenerationOptions()
        sampled.temperature = 0.9
        sampled.topP = 0.95
        sampled.seed = 11
        sampled.maxTokens = 12
        let first = try model.generate(messages: question, options: sampled)
        XCTAssertEqual(try model.generate(messages: question, options: sampled), first, "a seeded sampled run repeats")
        let seeded = NFKInferenceRequest(inputs: [NFKInputPrompt: "Name a color."],
                                         parameters: [NFKParameterTemperature: 0.9, NFKParameterTopP: 0.95,
                                                      NFKParameterSeed: 11, NFKParameterMaxTokens: 12])
        XCTAssertEqual(try backend.runInference(for: seeded).output(forKey: NFKOutputText) as? String, model.decode(first),
                       "the backend's sampling parameters reach the sampler")
        var greedy = NFKMLXGenerationOptions()
        greedy.maxTokens = 12
        XCTAssertEqual(try model.generate(messages: question, options: greedy),
                       try model.generate(text: "Name a color.", maxTokens: 12))
    }
}
