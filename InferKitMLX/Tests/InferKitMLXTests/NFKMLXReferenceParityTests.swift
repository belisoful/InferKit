//
//  NFKMLXReferenceParityTests.swift
//  InferKitMLXTests
//
//  Numerical comparison against each model's reference implementation — the only check that catches a
//  wrong-but-plausible output. Visual inspection passed SAM while it was missing its entire input
//  normalization, and passed Depth Anything for the same reason (its output is min-max normalized, so a
//  wrong input distribution still yields a sensible-looking map).
//
//  `Tools/reference-parity/run_reference.py <model> <file>` writes a safetensors record holding both the
//  raw plate it used (`input_image`, `[H, W, 3]` in 0...1, unpreprocessed) and the reference's result
//  (`output`). These tests feed the SAME plate to the MLX model and compare. Sharing the tensor keeps
//  image decoding out of the comparison while each side still applies its own preprocessing, so a
//  preprocessing bug shows up as a mismatch instead of hiding.
//
//  Point IK_PARITY_<MODEL> at the record in ~/.inferkit-validation.json; each test skips without it.
//

import XCTest
import CoreGraphics
import ImageIO
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXReferenceParityTests: XCTestCase {

    // Each parity test loads a model, several of them by the tens of gigabytes, and MLX's GPU cache
    // survives from one test to the next inside a process. Left alone, the accumulation starves the
    // largest float32 forward (Gemma E2B, ~20 GB) into a Metal command-buffer TIMEOUT — a process
    // kill, not a failure — once enough tests run before it. Measured: the same test passes in 25 s
    // alone and dies mid-suite. Clearing between tests keeps every test's footprint its own.
    override func tearDown() {
        NFKMLXGPU.clearCache()
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

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    /// Loads a reference record, returning the shared plate and the reference result.
    private func record(_ key: String) throws -> (input: MLXArray, output: MLXArray) {
        guard let path = config[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key) to a record from Tools/reference-parity/run_reference.py")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        guard let input = arrays["input_image"], let output = arrays["output"] else {
            throw NFKMLXError.noOutput
        }
        return (input, output)
    }

    /// The shared plate as a `CGImage`, so the model runs through its real public path.
    private func image(from tensor: MLXArray) throws -> CGImage {
        let (height, width) = (tensor.shape[0], tensor.shape[1])
        let values = (tensor * 255).asArray(Float.self)
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0 ..< (width * height) {
            for channel in 0 ..< 3 {
                bytes[pixel * 4 + channel] = UInt8(max(0, min(255, values[pixel * 3 + channel].rounded())))
            }
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    private func grayValues(_ image: CGImage) -> [Double] {
        let width = image.width, height = image.height
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &buffer, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return stride(from: 0, to: buffer.count, by: 4).map { Double(buffer[$0]) / 255.0 }
    }

    private func rgbValues(_ image: CGImage) -> [Double] {
        let width = image.width, height = image.height
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &buffer, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return (0 ..< width * height).flatMap { pixel in
            (0 ..< 3).map { Double(buffer[pixel * 4 + $0]) / 255.0 }
        }
    }

    /// Runs a model's real public path over the record's plate and reads the result back as pixels.
    /// Going through `CGImage` quantizes to eight bits, so the comparison tolerates about 1/255 —
    /// enough to catch a wrong normalization, a transposed weight, or a missing layer.
    private func pixels(from backend: any NFKInferenceBackend, plate: MLXArray, key: String = NFKOutputImage) throws -> CGImage {
        let request = NFKInferenceRequest(inputs: [NFKInputImage: try image(from: plate)])
        let result = try backend.runInference(for: request)
        guard let value = result.output(forKey: key), CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.noOutput
        }
        return (value as! CGImage)
    }

    private func cosine(_ a: [Double], _ b: [Double]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(a.count, b.count) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        return (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 0
    }

    // MARK: CLIP

    // An embedding is only useful if it lands where the reference's lands: unit length and discriminative
    // are necessary but not sufficient, and would both hold with a wrong input normalization.
    func testCLIPMatchesTheReferenceEmbedding() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_CLIP")
        let backend = try NFKMLXCLIP.backend(weightsURL: weights("IK_VAL_CLIP"))
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: try image(from: inputTensor)]))
        let ours = try XCTUnwrap(result.embedding).map(\.doubleValue)
        let reference = referenceOutput.asArray(Float.self).map(Double.init)

        XCTAssertEqual(ours.count, reference.count, "same embedding width as the reference")
        let similarity = cosine(ours, reference)
        print("VALIDATION PARITY clip: cosine against the reference embedding = \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the embedding matches the reference implementation")
    }

    // MARK: Depth Anything

    // The depth map is min-max normalized, so a wrong input distribution still looks plausible. Comparing
    // against the reference map is what distinguishes "sensible" from "correct".
    func testDepthAnythingMatchesTheReferenceMap() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_DEPTH")
        let backend = try NFKMLXDepthAnything.backend(variant: .small, weightsURL: weights("IK_VAL_DEPTH"))
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: try image(from: inputTensor)]))
        let ours = grayValues(try XCTUnwrap(result.output(forKey: NFKOutputImage).map { $0 as! CGImage }))
        let reference = referenceOutput.asArray(Float.self).map(Double.init)

        XCTAssertEqual(ours.count, reference.count, "same map size as the reference")
        let similarity = cosine(ours, reference)
        let meanAbsolute = zip(ours, reference).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(ours.count)
        print("VALIDATION PARITY depth: cosine \(similarity), mean |difference| \(meanAbsolute)")
        XCTAssertGreaterThan(similarity, 0.99, "the depth map matches the reference implementation")
        XCTAssertLessThan(meanAbsolute, 0.05, "and agrees pointwise, not just in shape")
    }

    // Localizes a Depth Anything mismatch: the encoder's hooked features are the seam between the DINOv2
    // backbone and the DPT head. Agreement here with disagreement at the output means the head is at
    // fault; disagreement here means the backbone is.
    func testDepthAnythingEncoderMatchesTheReference() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_DEPTH_ENCODER")
        let net = NFKMLXDepthAnything.makeNet(.small)
        try NFKMLXDepthAnything.loadWeights(into: net, from: weights("IK_VAL_DEPTH"))

        // Feed the encoder exactly what the model's own path feeds it.
        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225])
        let prepared = (inputTensor.reshaped([1, height, width, 3]) - mean) / standardDeviation
        let features = net.pretrained.hookedFeatures(prepared)
        let ours = features[0]                                  // [1, tokens, dim]
        eval(ours)

        // The reference keeps the class token; this port drops it. Orient both to [tokens, dim].
        let dim = ours.shape[2]
        let reference = referenceOutput.shape[0] == dim ? referenceOutput.transposed(1, 0) : referenceOutput
        let referenceTokens = reference.shape[0]
        let trimmed = referenceTokens == ours.shape[1] + 1
            ? reference[1..., 0...]
            : reference
        let a = ours.reshaped([ours.shape[1] * dim]).asArray(Float.self).map(Double.init)
        let b = trimmed.reshaped([trimmed.shape[0] * dim]).asArray(Float.self).map(Double.init)

        XCTAssertEqual(a.count, b.count, "same hooked-feature size as the reference")
        let similarity = cosine(a, b)
        print("VALIDATION PARITY depth-encoder: cosine \(similarity) over \(a.count) values")
        XCTAssertGreaterThan(similarity, 0.99, "the DINOv2 encoder matches the reference")
    }

    // MARK: SegFormer

    // Class logits at the decode head's native quarter resolution, against the HF reference. Comparing
    // logits rather than the argmax label map keeps the check sensitive: a label map agrees wherever the
    // winning class merely survives, hiding sizeable drift underneath.
    func testSegFormerMatchesTheReferenceLogits() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_SEGFORMER")
        let net = NFKMLXSegFormerNet(.mitB0)
        try NFKMLXSegFormer.loadWeights(into: net, from: weights("IK_VAL_SEGFORMER"))
        net.train(false)                                        // the decode head's BatchNorm uses running stats

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225])
        let prepared = (inputTensor.reshaped([1, height, width, 3]) - mean) / standardDeviation
        let ourLogits = net.logits(prepared)                    // [1, h/4, w/4, classes]
        eval(ourLogits)

        let count = referenceOutput.size
        XCTAssertEqual(ourLogits.size, count, "same logit volume as the reference")
        let ours = ourLogits.reshaped([count]).asArray(Float.self).map(Double.init)
        let reference = referenceOutput.reshaped([count]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, reference)

        // Agreement of the predicted class, which is what the backend actually emits.
        let classes = referenceOutput.shape[2]
        var agree = 0, pixels = 0
        for pixel in stride(from: 0, to: count, by: classes) {
            var ourBest = 0, referenceBest = 0
            for c in 1 ..< classes {
                if ours[pixel + c] > ours[pixel + ourBest] { ourBest = c }
                if reference[pixel + c] > reference[pixel + referenceBest] { referenceBest = c }
            }
            if ourBest == referenceBest { agree += 1 }
            pixels += 1
        }
        let agreement = Double(agree) / Double(pixels)
        print("VALIDATION PARITY segformer: logit cosine \(similarity), label agreement \(agreement)")
        XCTAssertGreaterThan(similarity, 0.99, "the class logits match the reference implementation")
        XCTAssertGreaterThan(agreement, 0.99, "and the predicted labels agree")
    }

    // MARK: Conv-TasNet

    // Speech separation against asteroid's own model. Separation is permutation-invariant — the two
    // estimated speakers can come out in either order — so each of our outputs is matched to its best
    // reference partner before scoring, which is how separation quality is measured in the literature.
    func testConvTasNetMatchesTheReferenceSeparation() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_CONVTASNET"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_CONVTASNET to a convtasnet record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"])
        let referenceEstimates = try XCTUnwrap(arrays["output"])          // [speakers, samples]

        let net = NFKMLXConvTasNetNet(.libri2Mix16k)
        try NFKMLXConvTasNet.loadWeights(into: net, from: weights("IK_VAL_CONVTASNET"))
        let ours = net.separate(waveform)                                 // [speakers, samples]
        eval(ours)

        let speakers = referenceEstimates.shape[0]
        let samples = min(ours.shape[1], referenceEstimates.shape[1])
        var best = [Double]()
        for speaker in 0 ..< speakers {
            let mine = ours[speaker, 0 ..< samples].asArray(Float.self).map(Double.init)
            let scores = (0 ..< speakers).map { other -> Double in
                let theirs = referenceEstimates[other, 0 ..< samples].asArray(Float.self).map(Double.init)
                return abs(cosine(mine, theirs))                          // sign is not meaningful here
            }
            best.append(scores.max() ?? 0)
        }
        let worst = best.min() ?? 0
        print("VALIDATION PARITY convtasnet: per-speaker best cosine \(best), worst \(worst)")
        XCTAssertGreaterThan(worst, 0.99, "both separated speakers match the reference implementation")
    }

    // MARK: Demucs

    // Music separation against the released time-domain model. Unlike speech separation this is not
    // permutation-invariant — each output channel pair is a named instrument — so every stem is scored
    // against its own reference stem.
    func testDemucsMatchesTheReferenceSeparation() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEMUCS"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_DEMUCS to a demucs record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"])                  // [channels, samples]
        let reference = try XCTUnwrap(arrays["output"])                   // [stems, channels, samples]

        let net = NFKMLXDemucs.makeNet()
        try NFKMLXDemucs.loadWeights(into: net, from: weights("IK_VAL_DEMUCS"))
        let ours = net.separate(waveform.transposed(1, 0))                // [1, samples, stems·channels]
        eval(ours)

        let (stems, channels) = (reference.shape[0], reference.shape[1])
        let samples = min(ours.shape[1], reference.shape[2])
        var scores = [Double]()
        for stem in 0 ..< stems {
            for channel in 0 ..< channels {
                let mine = ours[0, 0 ..< samples, stem * channels + channel].asArray(Float.self).map(Double.init)
                let theirs = reference[stem, channel, 0 ..< samples].asArray(Float.self).map(Double.init)
                scores.append(cosine(mine, theirs))
            }
        }
        let worst = scores.min() ?? 0
        print("VALIDATION PARITY demucs: per-stem-channel cosine \(scores), worst \(worst)")
        XCTAssertGreaterThan(worst, 0.99, "every stem matches the reference implementation")
    }

    // The denoiser is the same network in its speech configuration, so this scores the shared module
    // against the other released family — a change made for one must not break the other.
    func testDenoiserMatchesTheReferenceEnhancement() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DENOISER"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_DENOISER to a denoiser record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"])                  // [samples]
        let reference = try XCTUnwrap(arrays["output"])                   // [samples]

        let net = NFKMLXDenoiser.makeNet()
        try NFKMLXDemucs.loadWeights(into: net, from: weights("IK_VAL_DENOISER"))
        let ours = net.separate(waveform)                                 // [1, samples, 1]
        eval(ours)

        let samples = min(ours.shape[1], reference.shape[0])
        let mine = ours[0, 0 ..< samples, 0].asArray(Float.self).map(Double.init)
        let theirs = reference[0 ..< samples].asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY denoiser: cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the cleaned waveform matches the reference implementation")
    }

    // MARK: Silero VAD

    // Silero VAD v6 is a stateful streaming model: each 512-sample chunk carries a 64-sample look-back
    // and the LSTM state threads across chunks. Running the whole clip as one LSTM sequence has to
    // reproduce the reference's chunk-by-chunk stream, which a wrong context roll or a dropped state
    // would break silently. The reference is the released snakers4 JIT (silero_vad 6.2.1), loaded into
    // this port through its own `_model.*` weights.
    func testSileroVADMatchesTheReferenceProbabilities() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SILERO_VAD"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_SILERO_VAD to a silero_vad record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"])                  // [samples]
        let reference = try XCTUnwrap(arrays["output"])                   // [chunks]

        let net = NFKMLXSileroVAD.makeNet()
        try NFKMLXSileroVAD.loadWeights(into: net, from: weights("IK_VAL_SILERO_VAD"))
        let ours = net.speechProbabilities(waveform.asArray(Float.self), sampleRate: 16000)
        let theirs = reference.asArray(Float.self)

        XCTAssertEqual(ours.count, theirs.count, "one probability per 512-sample chunk")
        let mine = ours.map(Double.init), refd = theirs.map(Double.init)
        let similarity = cosine(mine, refd)
        let maxAbsolute = zip(mine, refd).map { abs($0 - $1) }.max() ?? 0
        // The spans either implementation would emit are decided at the threshold, so agreement there is
        // what makes the ports interchangeable rather than merely close.
        let agreement = zip(mine, refd).filter { ($0 >= 0.5) == ($1 >= 0.5) }.count
        print("VALIDATION PARITY silero-vad: cosine \(similarity), max |difference| \(maxAbsolute), threshold agreement \(agreement)/\(theirs.count)")
        XCTAssertGreaterThan(similarity, 0.9999, "per-chunk speech probabilities match the reference JIT")
        XCTAssertLessThan(maxAbsolute, 1e-3, "and match pointwise, not only in direction")
        XCTAssertEqual(agreement, theirs.count, "the same chunks cross the speech threshold")
    }

    // MARK: DAC

    // The Descript Audio Codec end to end: the encoder + RVQ must assign the same codebook tokens as the
    // reference, and the decoder must reconstruct the same waveform from a given set of codes. Decoding
    // the REFERENCE codes isolates the decoder from any single code that flips on a codebook near-tie.
    func testDACMatchesTheReferenceCodesAndReconstruction() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DAC"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_DAC to a dac record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"])                  // [samples]
        let referenceCodes = try XCTUnwrap(arrays["codes"])               // [codebooks, frames]
        let referenceRecon = try XCTUnwrap(arrays["output"])              // [samples], the decode of the codes

        let codec = try NFKMLXDAC.codec(configuration: .dac44kHz, weightsURL: weights("IK_VAL_DAC"))

        let (books, frames) = (referenceCodes.shape[0], referenceCodes.shape[1])
        let referenceFlat = referenceCodes.asType(.int32).asArray(Int32.self)
        let referenceGrid = (0 ..< books).map { book in (0 ..< frames).map { Int(referenceFlat[book * frames + $0]) } }

        // Codes: the encoder + RVQ assignment.
        let mine = codec.encode(waveform.asArray(Float.self))
        var agree = 0
        for book in 0 ..< books {
            for frame in 0 ..< frames where mine[book][frame] == referenceGrid[book][frame] { agree += 1 }
        }
        let agreement = Double(agree) / Double(books * frames)

        // Reconstruction: the decoder, fed the reference's codes.
        let myRecon = codec.decode(referenceGrid)
        let count = min(myRecon.count, referenceRecon.shape[0])
        let similarity = cosine(Array(myRecon.prefix(count)).map(Double.init),
                                referenceRecon[0 ..< count].asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY dac: code agreement \(agree)/\(books * frames), reconstruction cosine \(similarity)")
        XCTAssertGreaterThan(agreement, 0.99, "the encoder + RVQ assign the reference's codes")
        XCTAssertGreaterThan(similarity, 0.999, "the decoder reconstructs the reference's waveform from its codes")
    }

    // MARK: SNAC

    // The multi-scale codec: the encoder + RVQ must assign the same tokens at each codebook's own
    // temporal rate (the coarser codebooks emit fewer), and the decoder must reconstruct the same
    // waveform from a given set of codes. The reference's decoder noise block is disabled on both sides
    // (its expected contribution is zero), so the decode is deterministic and comparable.
    func testSNACMatchesTheReferenceCodesAndReconstruction() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SNAC"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_SNAC to a snac record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"])
        let referenceRecon = try XCTUnwrap(arrays["output"])

        // One code stream per codebook, at that codebook's rate.
        var referenceGrid = [[Int]]()
        var index = 0
        while let stream = arrays["codes\(index)"] {
            referenceGrid.append(stream.asType(.int32).asArray(Int32.self).map(Int.init))
            index += 1
        }
        XCTAssertGreaterThan(referenceGrid.count, 1, "SNAC has several codebooks at different rates")

        let codec = try NFKMLXSNAC.codec(configuration: .snac24kHz, weightsURL: weights("IK_VAL_SNAC"))

        let mine = codec.encode(waveform.asArray(Float.self))
        XCTAssertEqual(mine.map(\.count), referenceGrid.map(\.count), "the per-codebook code counts match")
        var agree = 0, total = 0
        for (mineStream, referenceStream) in zip(mine, referenceGrid) {
            for (a, b) in zip(mineStream, referenceStream) where a == b { agree += 1 }
            total += referenceStream.count
        }

        let myRecon = codec.decode(referenceGrid, deterministic: true)
        let count = min(myRecon.count, referenceRecon.shape[0])
        let similarity = cosine(Array(myRecon.prefix(count)).map(Double.init),
                                referenceRecon[0 ..< count].asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY snac: code agreement \(agree)/\(total), reconstruction cosine \(similarity)")
        XCTAssertGreaterThan(Double(agree) / Double(total), 0.99, "the encoder + multi-scale RVQ assign the reference's codes")
        XCTAssertGreaterThan(similarity, 0.999, "the decoder reconstructs the reference's waveform from its codes")
    }

    // MARK: SigLIP 2

    // The image-text model end to end: the image embedding (attention-pooling head), the text embeddings
    // (last-token projection over the 256k-vocab tower), and the sigmoid logits must all match the
    // reference. The float plate is fed directly so the comparison is not blunted by 8-bit image
    // quantization.
    func testSigLIP2MatchesTheReferenceEmbeddingsAndLogits() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SIGLIP2"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_SIGLIP2 to a siglip2 record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let plate = try XCTUnwrap(arrays["input_image"])                  // [224, 224, 3] in 0…1
        let referenceImage = try XCTUnwrap(arrays["output"])              // [768]
        let referenceText = try XCTUnwrap(arrays["text_embeds"])          // [texts, 768]
        let tokens = try XCTUnwrap(arrays["tokens"])                      // [texts, 64] int
        let referenceLogits = try XCTUnwrap(arrays["logits"])             // [texts]

        let net = NFKMLXSigLIP2.makeNet(.base)
        try NFKMLXSigLIP2.loadWeights(into: net, from: weights("IK_VAL_SIGLIP2"))

        let pixels = plate.reshaped([1, plate.shape[0], plate.shape[1], 3]) * 2 - 1
        let ids = tokens.asType(.int32)

        // Localize any vision mismatch: the embeddings seam and the post-layernorm hidden (pre-head).
        if let patchEmbeds = arrays["patch_embeds"], let visionLast = arrays["vision_last"] {
            let myEmbeds = net.vision.embeddings(pixels); eval(myEmbeds)
            let embedCosine = cosine(myEmbeds.reshaped([-1]).asArray(Float.self).map(Double.init),
                                     patchEmbeds.reshaped([-1]).asArray(Float.self).map(Double.init))
            var hidden = myEmbeds
            for layer in net.vision.encoder.layers { hidden = layer(hidden) }
            hidden = net.vision.postLayerNorm(hidden); eval(hidden)
            let lastCosine = cosine(hidden.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    visionLast.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM siglip2: embeddings cosine \(embedCosine), post-layernorm cosine \(lastCosine)")
        }

        let image = net.imageEmbedding(pixels)[0].asArray(Float.self).map(Double.init)
        let imageSimilarity = cosine(image, referenceImage.asArray(Float.self).map(Double.init))

        let text = net.textEmbedding(ids)                                 // [texts, 768]
        eval(text)
        var worstText = 1.0
        for row in 0 ..< referenceText.shape[0] {
            let mine = text[row].asArray(Float.self).map(Double.init)
            let theirs = referenceText[row].asArray(Float.self).map(Double.init)
            worstText = Swift.min(worstText, cosine(mine, theirs))
        }

        let logits = net.logits(image: pixels, text: ids).reshaped([-1]).asArray(Float.self).map(Double.init)
        let logitError = zip(logits, referenceLogits.asArray(Float.self).map(Double.init)).map { abs($0 - $1) }.max() ?? 0

        print("VALIDATION PARITY siglip2: image cosine \(imageSimilarity), worst text cosine \(worstText), max |logit diff| \(logitError)")
        XCTAssertGreaterThan(imageSimilarity, 0.9999, "the image embedding matches the reference")
        XCTAssertGreaterThan(worstText, 0.9999, "every text embedding matches the reference")
        XCTAssertLessThan(logitError, 1e-2, "the sigmoid logits match the reference")
    }

    // MARK: TAESD

    // The tiny autoencoder: the encoder must land the same latent as the reference, and the decoder must
    // reconstruct the same image from a given latent. Decoding the REFERENCE latent isolates the decoder.
    func testTAESDMatchesTheReferenceEncodeAndDecode() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_TAESD"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_TAESD to a taesd record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let plate = try XCTUnwrap(arrays["input_image"])                  // [H, W, 3]
        let referenceLatent = try XCTUnwrap(arrays["latent"])             // [h, w, 4]
        let referenceImage = try XCTUnwrap(arrays["output"])              // [H, W, 3]

        let net = NFKMLXTAESD.makeNet()
        try NFKMLXTAESD.loadWeights(into: net, from: weights("IK_VAL_TAESD"))

        let latent = net.encode(plate.reshaped([1, plate.shape[0], plate.shape[1], 3])); eval(latent)
        let latentCosine = cosine(latent.reshaped([-1]).asArray(Float.self).map(Double.init),
                                  referenceLatent.reshaped([-1]).asArray(Float.self).map(Double.init))

        let decoded = net.decode(referenceLatent.reshaped([1, referenceLatent.shape[0], referenceLatent.shape[1], 4]))
        eval(decoded)
        let mine = decoded.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceImage.reshaped([-1]).asArray(Float.self).map(Double.init)
        let imageCosine = cosine(mine, theirs)
        let meanAbsolute = zip(mine, theirs).map { abs($0 - $1) }.reduce(0, +) / Double(mine.count)
        print("VALIDATION PARITY taesd: latent cosine \(latentCosine), decode cosine \(imageCosine), mean |diff| \(meanAbsolute)")
        XCTAssertGreaterThan(latentCosine, 0.9999, "the encoder lands the reference latent")
        XCTAssertGreaterThan(imageCosine, 0.9999, "the decoder reconstructs the reference image")
        XCTAssertLessThan(meanAbsolute, 1e-3, "and matches pointwise")
    }

    // MARK: LTX-Video VAE

    // The causal 3D video autoencoder end to end: the encoder must land the same latent and the decoder
    // must reconstruct the same video from a given latent. The recorded encoder seams (conv_in, the first
    // down block, the mid block) localize any mismatch in the 3D convolutions / patchify.
    func testLTXVideoVAEMatchesTheReferenceEncodeAndDecode() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_LTX_VAE"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_LTX_VAE to an ltx-vae record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let video = try XCTUnwrap(arrays["input_video"])                 // [T, H, W, 3]
        let referenceLatent = try XCTUnwrap(arrays["latent"])            // [t, h, w, 128]
        let referenceOutput = try XCTUnwrap(arrays["output"])            // [T, H, W, 3]

        let net = NFKMLXLTXVideoVAE.makeNet(.base)
        try NFKMLXLTXVideoVAE.loadWeights(into: net, from: weights("IK_VAL_LTX_VAE"))
        let input = video.reshaped([1, video.shape[0], video.shape[1], video.shape[2], 3])

        func seam(_ key: String, _ produce: () -> MLXArray) {
            guard let reference = arrays[key] else { return }
            let mine = produce(); eval(mine)
            print("SEAM ltx \(key): cosine \(cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init), reference.reshaped([-1]).asArray(Float.self).map(Double.init))), shape \(mine.shape) vs \(reference.shape)")
        }
        seam("enc_conv_in") { net.encoder.convIn(net.encoder.patchify(input)) }
        seam("enc_down0") {
            var h = net.encoder.convIn(net.encoder.patchify(input)); h = net.encoder.downBlocks[0](h); return h
        }
        seam("enc_mid") {
            var h = net.encoder.convIn(net.encoder.patchify(input))
            for block in net.encoder.downBlocks { h = block(h) }
            return net.encoder.midBlock(h)
        }

        let latent = net.encode(input); eval(latent)
        let latentCosine = cosine(latent.reshaped([-1]).asArray(Float.self).map(Double.init),
                                  referenceLatent.reshaped([-1]).asArray(Float.self).map(Double.init))

        let decoded = net.decode(referenceLatent.reshaped([1, referenceLatent.shape[0], referenceLatent.shape[1], referenceLatent.shape[2], 128]))
        eval(decoded)
        let decodeCosine = cosine(decoded.reshaped([-1]).asArray(Float.self).map(Double.init),
                                  referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY ltx-vae: latent cosine \(latentCosine), decode cosine \(decodeCosine)")
        XCTAssertGreaterThan(latentCosine, 0.9999, "the encoder lands the reference latent")
        XCTAssertGreaterThan(decodeCosine, 0.9999, "the decoder reconstructs the reference video")
    }

    // MARK: LTX-Video DiT

    // The LTX denoising transformer, verified in isolation with recorded text embeddings: the 3D rotary,
    // the adaLN self-attention, the cross-attention to text, and the velocity prediction must match the
    // reference. Seams (rope, proj_in, first block) localize any mismatch.
    func testLTXTransformerMatchesTheReferenceVelocity() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_LTX_TF"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_LTX_TF to an ltx-transformer record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let latent = try XCTUnwrap(arrays["latent"]).reshaped([1, 8, 128])
        let text = try XCTUnwrap(arrays["text"]).reshaped([1, 4, 4096])
        let timestep = try XCTUnwrap(arrays["timestep"])
        let referenceOutput = try XCTUnwrap(arrays["output"])

        let net = NFKMLXLTXTransformer.makeNet(.base)
        try NFKMLXLTXTransformer.loadWeights(into: net, from: URL(fileURLWithPath: config["IK_VAL_LTX_TF"]!))
        let grid = (2, 2, 2), scale: (Float, Float, Float) = (1, 1, 1)

        func seam(_ key: String, _ produce: () -> MLXArray) {
            guard let reference = arrays[key] else { return }
            let mine = produce(); eval(mine)
            print("SEAM ltx-tf \(key): cosine \(cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init), reference.reshaped([-1]).asArray(Float.self).map(Double.init)))")
        }
        let rope = net.rotary.embedding(frames: 2, height: 2, width: 2, scale: scale)
        seam("rope_cos") { rope.0 }
        seam("rope_sin") { rope.1 }
        seam("proj_in") { net.projIn(latent) }
        seam("block0") {
            let hidden = net.projIn(latent)
            let temb = net.timeEmbed(timestep).temb.reshaped([1, 1, -1])
            let context = net.captionProjection(text)
            return net.blocks[0](hidden, context: context, temb: temb, rotary: rope)
        }

        let output = net(latent, text: text, timestep: timestep, grid: grid, ropeScale: scale)
        eval(output)
        let similarity = cosine(output.reshaped([-1]).asArray(Float.self).map(Double.init),
                                referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY ltx-transformer: velocity cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the predicted velocity matches the reference DiT")
    }

    // MARK: LTX T5 text encoder

    // The T5-XXL text encoder — the last LTX stage — verified against transformers. Seams (the embedding
    // and the first block) localize any mismatch in the relative-position bias / T5LayerNorm / gated FFN.
    func testLTXT5EncoderMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_LTX_T5"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_LTX_T5 to an ltx-t5 record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asType(.int32)
        let referenceOutput = try XCTUnwrap(arrays["output"])

        let net = NFKMLXT5Encoder.makeNet(.xxl)
        try NFKMLXT5Encoder.loadWeights(into: net, from: URL(fileURLWithPath: config["IK_VAL_LTX_T5"]!))

        func seam(_ key: String, _ produce: () -> MLXArray) {
            guard let reference = arrays[key] else { return }
            let mine = produce(); eval(mine)
            print("SEAM ltx-t5 \(key): cosine \(cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init), reference.reshaped([-1]).asArray(Float.self).map(Double.init)))")
        }
        seam("embed") { net.shared(tokens) }
        seam("block0") {
            let hidden = net.shared(tokens)
            let bias = net.encoder.block[0].selfAttention.attention.computeBias(tokens.shape[1])
            return net.encoder.block[0](hidden, bias: bias)
        }

        let output = net(tokens); eval(output)
        let similarity = cosine(output.reshaped([-1]).asArray(Float.self).map(Double.init),
                                referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY ltx-t5: text embedding cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the text embedding matches the reference T5 encoder")
    }

    // MARK: Z-Image S3-DiT

    // The single-stream Z-Image DiT at a tiny random configuration, against diffusers'
    // ZImageTransformer2DModel. Sequence lengths are not a multiple of 32, so the learned pad tokens and
    // the (0,0,0) pad positions are exercised. The t_embedder seam localizes a timestep-conditioning
    // mismatch; the final velocity covers the refiners, the unified layers, the complex 3-axis rope, and
    // the sandwich adaptive norms.
    func testZImageTransformerMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_Z_IMAGE"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_Z_IMAGE (run_reference.py z_image)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let latent = try XCTUnwrap(arrays["latent"])
        let capFeats = try XCTUnwrap(arrays["cap_feats"])
        let timestep = try XCTUnwrap(arrays["timestep"])
        let referenceOutput = try XCTUnwrap(arrays["output"])

        let net = NFKMLXZImageTransformerNet(.tiny)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value) : nil
        }
        try NFKMLXWeights.apply(weights, to: net)

        if let reference = arrays["t_emb"] {
            let mine = net.tEmbedder(timestep * net.config.timestepScale)[0]; eval(mine)
            print("SEAM z-image t_emb: cosine \(cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init), reference.reshaped([-1]).asArray(Float.self).map(Double.init)))")
        }

        let output = net(latent, capFeats: capFeats, t: timestep); eval(output)
        let similarity = cosine(output.reshaped([-1]).asArray(Float.self).map(Double.init),
                                referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY z-image: velocity cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the predicted velocity matches the reference S3-DiT")
    }

    // MARK: SANA linear-attention DiT

    // The SANA DiT at a tiny random configuration, against diffusers' SanaTransformer2DModel. The
    // patch-embed and block seams localize the linear-attention / GLUMBConv / adaptive-norm path; the
    // final velocity covers the whole stack. The convolution weights load transposed to MLX's NHWC.
    func testSANATransformerMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SANA"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_SANA (run_reference.py sana)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let latent = try XCTUnwrap(arrays["latent"])
        let capFeats = try XCTUnwrap(arrays["cap_feats"])
        let timestep = try XCTUnwrap(arrays["timestep"])
        let referenceOutput = try XCTUnwrap(arrays["output"])

        let net = NFKMLXSANATransformerNet(.tiny)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = String(key.dropFirst(3))
            return (name, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(weights, to: net)

        if let reference = arrays["embedded"] {
            let mine = net.timeEmbed(timestep * net.config.timestepScale).embedded[0]; eval(mine)
            print("SEAM sana embedded: cosine \(cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init), reference.reshaped([-1]).asArray(Float.self).map(Double.init)))")
        }

        let output = net(latent, capFeats: capFeats, t: timestep); eval(output)
        let similarity = cosine(output.reshaped([-1]).asArray(Float.self).map(Double.init),
                                referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY sana: velocity cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the predicted velocity matches the reference SANA DiT")
    }

    // MARK: Wan text-to-video DiT

    // The Wan DiT at a tiny random configuration, against diffusers' WanTransformer3DModel. The final
    // velocity covers the Conv3d patch embed, the 3-axis interleaved rotary, the across-heads q/k norm,
    // self + cross attention, and the adaptive norms. The 5-D convolution weight loads transposed to NHWC.
    func testWanTransformerMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_WAN"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_WAN (run_reference.py wan)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let latent = try XCTUnwrap(arrays["latent"])
        let text = try XCTUnwrap(arrays["text"])
        let timestep = try XCTUnwrap(arrays["timestep"])
        let referenceOutput = try XCTUnwrap(arrays["output"])

        let net = NFKMLXWanTransformerNet(.tiny)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = String(key.dropFirst(3))
            return (name, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value)
        }
        try NFKMLXWeights.apply(weights, to: net)

        let output = net(latent, text: text, t: timestep); eval(output)
        let similarity = cosine(output.reshaped([-1]).asArray(Float.self).map(Double.init),
                                referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY wan: velocity cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the predicted velocity matches the reference Wan DiT")
    }

    // MARK: Flux autoencoder (Z-Image's VAE)

    // The Flux autoencoder Z-Image encodes into, at a tiny random configuration, against diffusers'
    // AutoencoderKL. It is the same class Stable Diffusion uses; the only new path is dropping the quant
    // convolutions (use_quant_conv=False), so the encoder's conv_out is the moments directly. Both the
    // encoded mean and the decode are compared.
    func testFluxAutoencoderMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_FLUX_VAE"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_FLUX_VAE (run_reference.py flux_vae)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let sample = try XCTUnwrap(arrays["sample"]).reshaped([1, 16, 16, 3])
        let referenceMean = try XCTUnwrap(arrays["mean"])
        let referenceDecoded = try XCTUnwrap(arrays["output"])

        var configuration = NFKMLXSDVAEConfiguration()
        configuration.latentChannels = 4
        configuration.blockChannels = [8, 16]
        configuration.layersPerBlock = 1
        configuration.normalizationGroups = 4
        configuration.useQuantConv = false
        let net = NFKMLXSDAutoencoder(configuration: configuration)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = NFKMLXStableDiffusionModels.remapVAEKey(String(key.dropFirst(3)))
            return (name, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(weights, to: net)

        let mean = net.encode(sample).mean; eval(mean)
        let meanSimilarity = cosine(mean.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    referenceMean.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY flux-vae: latent cosine \(meanSimilarity)")
        XCTAssertGreaterThan(meanSimilarity, 0.9999, "the encoded mean matches the reference")

        let decoded = net.decode(mean); eval(decoded)
        let decodeSimilarity = cosine(decoded.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceDecoded.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY flux-vae: decode cosine \(decodeSimilarity)")
        XCTAssertGreaterThan(decodeSimilarity, 0.9999, "the decode matches the reference")
    }

    // MARK: DC-AE (SANA's autoencoder)

    // The Deep-Compression Autoencoder at a tiny random configuration, against diffusers' AutoencoderDC.
    // Deterministic encode/decode. The tiny config exercises a ResBlock stage, an EfficientViTBlock stage
    // (multiscale ReLU linear attention + GLUMBConv), and the Conv-downsample / interpolate-upsample with
    // their channel-average / channel-repeat shortcuts. The reference's `<blocks>.<i>.<j>` Sequential
    // indices map onto the module's `<i>.block.<j>`; the convolution weights load transposed to NHWC.
    func testDCAutoencoderMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DC_AE"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_DC_AE (run_reference.py dc_ae)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let sample = try XCTUnwrap(arrays["sample"]).reshaped([1, 8, 8, 3])
        let referenceLatent = try XCTUnwrap(arrays["latent"])
        let referenceDecoded = try XCTUnwrap(arrays["output"])

        let net = NFKMLXDCAutoencoderNet(.tiny)
        let stageIndex = try! NSRegularExpression(pattern: "(down_blocks|up_blocks)\\.([0-9]+)\\.([0-9]+)\\.")
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            var name = String(key.dropFirst(3))
            name = stageIndex.stringByReplacingMatches(in: name, range: NSRange(name.startIndex..., in: name),
                                                       withTemplate: "$1.$2.block.$3.")
            return (name, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(weights, to: net)

        let latent = net.encode(sample); eval(latent)
        let latentSimilarity = cosine(latent.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceLatent.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY dc-ae: latent cosine \(latentSimilarity)")
        XCTAssertGreaterThan(latentSimilarity, 0.9999, "the encoded latent matches the reference")

        let decoded = net.decode(latent); eval(decoded)
        let decodeSimilarity = cosine(decoded.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceDecoded.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY dc-ae: decode cosine \(decodeSimilarity)")
        XCTAssertGreaterThan(decodeSimilarity, 0.9999, "the decode matches the reference")
    }

    // MARK: IP-Adapter (image conditioning)

    // IP-Adapter's two pieces at a tiny random configuration, against diffusers: the ImageProjection and
    // the decoupled cross-attention (text attention + scale·image attention, shared query). The
    // processor's `to_k_ip`/`to_v_ip` load under the attention module (the reference stores them under a
    // `processor.` prefix, stripped here).
    func testIPAdapterMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_IP_ADAPTER"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_IP_ADAPTER (run_reference.py ip_adapter)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let imageEmbeds = try XCTUnwrap(arrays["image_embeds"]).reshaped([1, -1])
        let referenceTokens = try XCTUnwrap(arrays["ip_tokens"])
        let hidden = try XCTUnwrap(arrays["hidden"]).reshaped([1, 8, 12])
        let text = try XCTUnwrap(arrays["text"]).reshaped([1, 5, 12])
        let referenceOutput = try XCTUnwrap(arrays["output"])

        // Image projection.
        let proj = NFKMLXIPAdapterImageProjection(imageEmbedDim: 16, crossAttentionDim: 12, numTokens: 4)
        try NFKMLXWeights.apply(arrays.compactMap { key, value in
            key.hasPrefix("wp::") ? (String(key.dropFirst(4)), value) : nil }, to: proj)
        let ipTokens = proj(imageEmbeds); eval(ipTokens)
        let tokenSimilarity = cosine(ipTokens.reshaped([-1]).asArray(Float.self).map(Double.init),
                                     referenceTokens.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY ip-adapter: projection cosine \(tokenSimilarity)")
        XCTAssertGreaterThan(tokenSimilarity, 0.9999, "the image projection matches the reference")

        // Decoupled cross-attention.
        let attn = NFKMLXIPAdapterAttention(queryDim: 12, crossAttentionDim: 12, heads: 2, headDim: 6)
        try NFKMLXWeights.apply(arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("wa::") else { return nil }
            return (String(key.dropFirst(4)).replacingOccurrences(of: "processor.", with: ""), value)
        }, to: attn)
        let output = attn(hidden, text: text, ipTokens: ipTokens, scale: 0.7); eval(output)
        let similarity = cosine(output.reshaped([-1]).asArray(Float.self).map(Double.init),
                                referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY ip-adapter: decoupled-attention cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the decoupled cross-attention matches the reference")
    }

    // MARK: DC-AE on the RELEASED weights

    // The Deep-Compression Autoencoder on the ACTUAL released SANA VAE weights, against diffusers'
    // AutoencoderDC — a real-weights end-to-end validation (the released config, the real checkpoint, and
    // the loader), not just the tiny-config architecture check. Loads through the model's own loadWeights.
    func testDCAutoencoderMatchesTheReferenceOnReleasedWeights() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DC_AE_REAL"], let directory = config["IK_VAL_DCAE"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_DC_AE_REAL + IK_VAL_DCAE (run_reference.py dc_ae_real, real weights)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let inputImage = try XCTUnwrap(arrays["input_image"])              // [H, W, 3] in 0...1
        let (h, w) = (inputImage.shape[0], inputImage.shape[1])
        let sample = (inputImage * 2 - 1).reshaped([1, h, w, 3])           // the reference's -1...1 input
        let referenceLatent = try XCTUnwrap(arrays["latent"])
        let referenceDecoded = try XCTUnwrap(arrays["output"])

        let net = NFKMLXDCAutoencoderNet(.sana)
        let weightsURL = URL(fileURLWithPath: directory).appendingPathComponent("diffusion_pytorch_model.safetensors")
        try NFKMLXDCAutoencoderNet.loadWeights(into: net, from: weightsURL)

        let latent = net.encode(sample); eval(latent)
        let latentSimilarity = cosine(latent.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceLatent.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY dc-ae-real: latent cosine \(latentSimilarity)")
        XCTAssertGreaterThan(latentSimilarity, 0.9999, "the released-weights latent matches the reference")

        let decoded = net.decode(latent); eval(decoded)
        let decodeSimilarity = cosine(decoded.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceDecoded.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY dc-ae-real: decode cosine \(decodeSimilarity)")
        XCTAssertGreaterThan(decodeSimilarity, 0.9999, "the released-weights decode matches the reference")
    }

    // MARK: RT-DETR object detection

    // RT-DETR end to end at a tiny random configuration, against transformers' own
    // RTDetrForObjectDetection: the ResNet-D backbone, the hybrid encoder (AIFI + CSP-RepVGG FPN/PAN),
    // anchor query selection, and the deformable-attention decoder with iterative box refinement. The
    // backbone/encoder/query-selection seams localize a divergence; the final logits and boxes cover the
    // whole stack. Weights load with the `model.` prefix stripped (the net root is HF's RTDetrModel), the
    // 4-D convolution weights transposed to NHWC, and BatchNorm run in eval mode.
    func testRTDetrMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_RTDETR"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_RTDETR (run_reference.py rtdetr)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let pixels = try XCTUnwrap(arrays["pixels"]).expandedDimensions(axis: 0)       // [1, H, W, 3] NHWC
        let referenceLogits = try XCTUnwrap(arrays["output"])
        let referenceBoxes = try XCTUnwrap(arrays["pred_boxes"])

        let net = NFKMLXRTDetrNet(.tiny)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::model.") else { return nil }
            let name = String(key.dropFirst("w::model.".count))
            return (name, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(weights, to: net)
        net.train(false)                                                  // BatchNorm running statistics

        func similarity(_ mine: MLXArray, _ reference: MLXArray) -> Double {
            eval(mine)
            return cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                          reference.reshaped([-1]).asArray(Float.self).map(Double.init))
        }
        func seam(_ label: String, _ mine: MLXArray, _ key: String) {
            guard let reference = arrays[key] else { return }
            print("SEAM rtdetr \(label): cosine \(similarity(mine, reference))")
        }

        let detection = net(pixels)
        // The whole architecture, seam by seam: the ResNet-D backbone, the hybrid encoder's PAN
        // outputs, and the query-selection scores (enc_class) and boxes (enc_coord) are exact.
        seam("backbone.0", detection.backboneFeatures[0], "bb.0")
        seam("backbone.1", detection.backboneFeatures[1], "bb.1")
        seam("backbone.2", detection.backboneFeatures[2], "bb.2")
        seam("enc_last", detection.encLastPAN, "enc_last")
        for (label, key, mine) in [("enc_class", "enc_class", detection.encClass),
                                   ("enc_coord", "enc_coord", detection.encCoord)] {
            let reference = try XCTUnwrap(arrays[key])
            let value = similarity(mine, reference)
            print("VALIDATION PARITY rtdetr: \(label) cosine \(value)")
            XCTAssertGreaterThan(value, 0.9999, "the query-selection \(label) matches the reference")
        }

        // The deformable-attention decoder in isolation of the top-k selection: run it over the
        // REFERENCE's selected query indices, so the logits and boxes are exact. This is the decisive
        // decoder parity, independent of the float-sensitive selection below.
        let refTopk = try XCTUnwrap(arrays["topk_ind"])
        let (decLogits, decBoxes) = net.decode(indices: refTopk, detection: detection)
        let decLogitSimilarity = similarity(decLogits, referenceLogits)
        let decBoxSimilarity = similarity(decBoxes, referenceBoxes)
        print("VALIDATION PARITY rtdetr (reference selection): logits cosine \(decLogitSimilarity), boxes cosine \(decBoxSimilarity)")
        XCTAssertGreaterThan(decLogitSimilarity, 0.9999, "the decoder logits match the reference over its selection")
        XCTAssertGreaterThan(decBoxSimilarity, 0.9999, "the decoder boxes match the reference over its selection")

        // The end-to-end run uses the port's OWN top-k selection, which differs from the reference's
        // only where two tokens' selection scores tie to within a float ulp — `torch.topk` and MLX's
        // `argSort` break such ties differently, so one or two of the 10 queries can swap. The logits
        // stay near-exact; the boxes carry the swapped queries, so they are asserted at a tolerance
        // that reflects the selection tie rather than a modeling error.
        let logitSimilarity = similarity(detection.logits, referenceLogits)
        let boxSimilarity = similarity(detection.boxes, referenceBoxes)
        print("VALIDATION PARITY rtdetr (own selection): logits cosine \(logitSimilarity), boxes cosine \(boxSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.9999, "the end-to-end detection logits match the reference")
        XCTAssertGreaterThan(boxSimilarity, 0.97, "the end-to-end boxes match up to the top-k selection tie")
    }

    // RT-DETR on the ACTUAL released PekingU/rtdetr_r50vd weights, against transformers' own
    // RTDetrForObjectDetection — a real-weights end-to-end validation of the r50vd geometry, the
    // released checkpoint, and the loader (including the stage-1 stride-1 shortcut the tiny config does
    // not exercise). Runs on the reference's own preprocessed pixels and its query selection, so the
    // logits and boxes are exact.
    func testRTDetrMatchesTheReferenceOnReleasedWeights() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_RTDETR_REAL"], let directory = config["IK_VAL_RTDETR"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_RTDETR_REAL + IK_VAL_RTDETR (run_reference.py rtdetr_real, real weights)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let pixels = try XCTUnwrap(arrays["pixels"]).expandedDimensions(axis: 0)       // [1, 640, 640, 3] NHWC
        let referenceLogits = try XCTUnwrap(arrays["output"])
        let referenceBoxes = try XCTUnwrap(arrays["pred_boxes"])
        let refTopk = try XCTUnwrap(arrays["topk_ind"])

        let net = NFKMLXRTDetrNet(.r50vd)
        let weightsURL = URL(fileURLWithPath: directory).appendingPathComponent("model.safetensors")
        try NFKMLXRTDetr.loadWeights(into: net, from: weightsURL)
        net.train(false)

        let detection = net(pixels)
        let (decLogits, decBoxes) = net.decode(indices: refTopk, detection: detection)
        eval(decLogits); eval(decBoxes)
        let logitSimilarity = cosine(decLogits.reshaped([-1]).asArray(Float.self).map(Double.init),
                                     referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init))
        let boxSimilarity = cosine(decBoxes.reshaped([-1]).asArray(Float.self).map(Double.init),
                                   referenceBoxes.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY rtdetr-real: logits cosine \(logitSimilarity), boxes cosine \(boxSimilarity)")
        XCTAssertGreaterThan(logitSimilarity, 0.9999, "the released-weights detection logits match the reference")
        XCTAssertGreaterThan(boxSimilarity, 0.9999, "the released-weights detection boxes match the reference")
    }

    // MARK: Parakeet-TDT speech recognition

    // Parakeet-TDT 0.6B v2 on the RELEASED weights, against NeMo's own EncDecRNNTBPEModel on the validation
    // speech clip, seam by seam: the normalized mel features, the depthwise-striding subsampler, the first
    // conformer layer, the full encoder, the joint's first-frame logits at the blank state, and the greedy
    // token-and-duration decode — which must reproduce the reference's tokens and frame timestamps exactly,
    // and the transcription text through the release's SentencePiece pieces.
    func testParakeetMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_PARAKEET"], let directory = config["IK_VAL_PARAKEET"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_PARAKEET + IK_VAL_PARAKEET (run_reference.py parakeet, real weights)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"]).asArray(Float.self)
        let referenceTokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self).map(Int.init)
        let referenceTimestamps = try XCTUnwrap(arrays["timestamps"]).asArray(Int32.self).map(Int.init)
        let referenceText = String(decoding: try XCTUnwrap(arrays["text"]).asArray(Int32.self).map { UInt8($0) }, as: UTF8.self)

        let net = NFKMLXParakeetNet(.tdt06B)
        let releaseURL = URL(fileURLWithPath: directory)
        try NFKMLXParakeet.loadWeights(into: net, from: releaseURL.appendingPathComponent("model_weights.ckpt"))

        func check(_ label: String, _ mine: MLXArray, _ key: String, _ threshold: Double = 0.9999) throws {
            let reference = try XCTUnwrap(arrays[key])
            eval(mine)
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY parakeet: \(label) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, threshold, "parakeet \(label) matches the reference")
        }
        let features = net.frontEnd.features(waveform)
        try check("features", features, "features")
        let pre = net.encoder.preEncode(features)
        try check("pre_encode", pre, "pre")
        let posEmb = net.encoder.positionEmbedding(length: pre.dim(1))
        try check("layer0", net.encoder.layers[0](pre, posEmb: posEmb), "layer0")
        let encoded = net.encode(features: features)
        try check("encoded", encoded, "encoded")
        // NeMo's joint log-softmaxes its output on the CPU when `log_softmax` is null — a constant shift
        // over the whole vector that leaves both argmaxes alone, so the port keeps raw logits and the seam
        // is compared in the reference's own log-softmax space.
        let (g, _) = net.decoder.prediction(token: nil, state: nil)
        let logits = net.joint(frame: encoded[0, 0, 0...], prediction: g)
        try check("joint0", logits - logSumExp(logits, axis: -1, keepDims: true), "joint0")

        let tokens = net.decode(encoded: encoded)
        XCTAssertEqual(tokens.map(\.id), referenceTokens, "the greedy TDT tokens match the reference")
        XCTAssertEqual(tokens.map(\.frame), referenceTimestamps, "the emission frames match the reference")
        let vocabURL = try XCTUnwrap(try FileManager.default.contentsOfDirectory(at: releaseURL, includingPropertiesForKeys: nil)
            .first { $0.lastPathComponent.hasSuffix("_tokenizer.vocab") })
        let text = try NFKMLXParakeetVocabulary(vocabURL: vocabURL).text(for: tokens.map(\.id))
        print("VALIDATION PARITY parakeet: transcription \(text.debugDescription)")
        XCTAssertEqual(text, referenceText)
    }

    // The public backend path on the released weights: the unpacked `.nemo` directory → `backendWithDirectoryURL:`
    // → the validation WAV under NFKInputAudio → the reference transcription under NFKOutputText, with one
    // timestamped segment per token under NFKOutputSegments.
    func testParakeetBackendTranscribesTheValidationClip() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_VAL_PARAKEET"], let audio = config["IK_VAL_AUDIO"],
              FileManager.default.fileExists(atPath: directory) else {
            throw XCTSkip("set IK_VAL_PARAKEET + IK_VAL_AUDIO")
        }
        let backend = try NFKMLXParakeet.backend(directoryURL: URL(fileURLWithPath: directory))
        let asset = NFKAudioAsset(fileURL: URL(fileURLWithPath: audio), durationSeconds: 3.47, sampleRate: 16000, channelCount: 1)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        print("VALIDATION PARITY parakeet-backend: \(result.text.debugDescription)")
        XCTAssertEqual(result.text, "The quick brown fox jumps over the lazy dog.")
        XCTAssertEqual(result.segments?.count, 19)
        XCTAssertEqual(result.segments?.first?.startSeconds ?? -1, 0, accuracy: 1e-9)
    }

    // MARK: Chatterbox — voice encoder + S3 speech tokenizer

    func testChatterboxVoiceEncoderAndTokenizerMatchTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_CHATTERBOX_VOICE"], let directory = config["IK_VAL_CHATTERBOX"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_CHATTERBOX_VOICE + IK_VAL_CHATTERBOX (run_reference.py chatterbox_voice, real weights)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let release = URL(fileURLWithPath: directory)
        let wave16 = try XCTUnwrap(arrays["wav16"]).asArray(Float.self)

        func check(_ label: String, _ mine: MLXArray, _ reference: MLXArray, _ threshold: Double = 0.9999) {
            eval(mine)
            XCTAssertEqual(mine.shape, reference.shape, "chatterbox \(label) shape")
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY chatterbox: \(label) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, threshold, "chatterbox \(label) matches the reference")
        }

        // Voice encoder: the librosa trim is exact, then mel → partials → embedding.
        let voice = NFKMLXChatterboxVoiceEncoderNet(.released)
        try NFKMLXChatterbox.loadVoiceEncoderWeights(into: voice, from: release.appendingPathComponent("ve.safetensors"))
        let trimmed = NFKMLXChatterboxVoiceFrontEnd.trimmed(wave16, topDecibels: 20)
        let referenceTrimmed = try XCTUnwrap(arrays["ve_wav"]).asArray(Float.self)
        XCTAssertEqual(trimmed.count, referenceTrimmed.count, "the trim cuts the same samples")
        XCTAssertEqual(trimmed.first, referenceTrimmed.first)
        let mel = NFKMLXChatterboxVoiceFrontEnd.mel(trimmed, configuration: .released)
        check("ve_mel", mel, try XCTUnwrap(arrays["ve_mel"]))
        let partials = NFKMLXChatterboxVoiceFrontEnd.partials(of: mel, configuration: .released)
        check("ve_partials", voice.embedPartials(partials), try XCTUnwrap(arrays["ve_partials"]))
        check("ve_embed", voice.embed(samples: wave16), try XCTUnwrap(arrays["ve_embed"]))

        // S3 tokenizer: mel, encoder states, and the codes, which must match exactly.
        let tokenizer = NFKMLXS3TokenizerNet(.released)
        try NFKMLXChatterbox.loadTokenizerWeights(into: tokenizer, from: release.appendingPathComponent("s3gen.safetensors"))
        let crop = Array(wave16.prefix(6 * 16000))
        let s3Mel = tokenizer.logMel(crop)
        check("s3_mel", s3Mel, try XCTUnwrap(arrays["s3_mel"]).transposed(1, 0))
        let hidden = tokenizer.hidden(mel: s3Mel)
        check("s3_hidden", hidden, try XCTUnwrap(arrays["s3_hidden"]))
        let referenceCodes = try XCTUnwrap(arrays["s3_codes"]).asArray(Int32.self).map(Int.init)
        let codes = tokenizer.codes(hidden: hidden)
        XCTAssertEqual(codes, referenceCodes, "the FSQ speech codes match the reference exactly")
        XCTAssertEqual(tokenizer.tokenize(crop, maximumCodes: 150), referenceCodes)
        let referenceWave = try XCTUnwrap(arrays["ref_wav16"]).asArray(Float.self)
        let referenceRefCodes = try XCTUnwrap(arrays["s3_ref_codes"]).asArray(Int32.self).map(Int.init)
        let refCodes = tokenizer.tokenize(referenceWave)
        let agreement = zip(refCodes, referenceRefCodes).filter(==).count
        print("VALIDATION PARITY chatterbox: s3_ref_codes agreement \(agreement)/\(referenceRefCodes.count)")
        XCTAssertEqual(refCodes, referenceRefCodes, "the whole-prompt speech codes match the reference exactly")
    }

    // MARK: Chatterbox — T3 text-to-speech-token model

    func testChatterboxT3MatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_CHATTERBOX_T3"], let directory = config["IK_VAL_CHATTERBOX"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_CHATTERBOX_T3 + IK_VAL_CHATTERBOX (run_reference.py chatterbox_t3, real weights)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let release = URL(fileURLWithPath: directory)
        func check(_ label: String, _ mine: MLXArray, _ reference: MLXArray, _ threshold: Double = 0.9999) {
            eval(mine)
            XCTAssertEqual(mine.shape, reference.shape, "chatterbox t3 \(label) shape")
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY chatterbox t3: \(label) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, threshold, "chatterbox t3 \(label) matches the reference")
        }

        // The text tokenizer reproduces the reference ids, start and stop tokens included.
        let text = String(decoding: try XCTUnwrap(arrays["text"]).asArray(Int32.self).map { UInt8($0) }, as: UTF8.self)
        let referenceText = try XCTUnwrap(arrays["text_tokens"]).asArray(Int32.self).map(Int.init)
        let tokenizer = try NFKMLXChatterboxTextTokenizer(url: release.appendingPathComponent("tokenizer.json"))
        XCTAssertEqual(tokenizer.encodeForSynthesis(NFKMLXChatterboxTextTokenizer.normalizedPunctuation(text)),
                       referenceText, "the T3 text tokens match the reference")

        let net = NFKMLXT3Net(.released)
        try NFKMLXChatterbox.loadT3Weights(into: net, from: release.appendingPathComponent("t3_cfg.safetensors"))
        let condition = NFKMLXT3Condition(
            speakerEmbedding: try XCTUnwrap(arrays["speaker_emb"]),
            promptTokens: try XCTUnwrap(arrays["cond_tokens"]).asArray(Int32.self).map(Int.init),
            exaggeration: try XCTUnwrap(arrays["emotion_adv"]).item(Float.self))
        let conditionEmbedding = net.conditionEmbedding(condition)
        check("cond_emb", conditionEmbedding[0], try XCTUnwrap(arrays["cond_emb"]))
        let embeddings = net.inputEmbeddings(condition: conditionEmbedding, textTokens: referenceText, guidance: true)
        check("embeds", embeddings, try XCTUnwrap(arrays["embeds"]))

        // Teacher-forced over the reference's own sampled codes: the second start token at position 0,
        // then each code at its position, both guidance rows.
        let speech = try XCTUnwrap(arrays["speech_tokens"]).asArray(Int32.self).map(Int.init)
        let width = embeddings.dim(2)
        var pieces = [embeddings, broadcast(net.speechTokenEmbedding(net.configuration.startSpeechToken, position: 0),
                                            to: [2, 1, width])]
        for (index, token) in speech.dropLast().enumerated() {
            pieces.append(broadcast(net.speechTokenEmbedding(token, position: index + 1), to: [2, 1, width]))
        }
        let logits = net.speechLogits(embeddings: concatenated(pieces, axis: 1))
        let start = conditionEmbedding.dim(1) + referenceText.count + 1
        let mine = logits[0..., start ..< (start + speech.count)]
        let reference = try XCTUnwrap(arrays["tf_logits"])
        check("tf_logits conditional", mine[0], reference[0])
        check("tf_logits unconditional", mine[1], reference[1])
        let agreement = zip(mine[0].argMax(axis: -1).asArray(Int32.self), reference[0].argMax(axis: -1).asArray(Int32.self))
            .filter(==).count
        print("VALIDATION PARITY chatterbox t3: argmax agreement \(agreement)/\(speech.count)")
        XCTAssertEqual(agreement, speech.count, "every teacher-forced position predicts the reference's top code")

        // The first step's processors (guidance, repetition penalty, temperature, min-p, top-p).
        let step0 = mine[0..., 0]
        let guided = (step0[0] + 0.5 * (step0[0] - step0[1])).asArray(Float.self)
        let options = NFKMLXT3SamplingOptions()
        let processed = NFKMLXT3Sampler.processed(guided, generated: [net.configuration.startSpeechToken], options: options)
        let referenceProcessed = try XCTUnwrap(arrays["step0_processed"]).asArray(Float.self)
        XCTAssertEqual(processed.count, referenceProcessed.count)
        var kept = 0, worst: Float = 0
        for (a, b) in zip(processed, referenceProcessed) {
            XCTAssertEqual(a.isFinite, b.isFinite, "min-p keeps the same codes")
            if a.isFinite && b.isFinite { kept += 1; worst = max(worst, abs(a - b)) }
        }
        print("VALIDATION PARITY chatterbox t3: step0 processed keeps \(kept) codes, worst |difference| \(worst)")
        XCTAssertGreaterThan(kept, 0)
        XCTAssertLessThan(worst, 1e-2)
        XCTAssertTrue(processed[speech[0]].isFinite, "the reference's sampled first code is admissible")

        // Generation runs through the cache: a short greedy run yields valid speech codes.
        var greedy = NFKMLXT3SamplingOptions()
        greedy.temperature = 0
        greedy.maximumTokens = 12
        let codes = net.generate(condition: condition, textTokens: referenceText, options: greedy)
        XCTAssertFalse(codes.isEmpty)
        XCTAssertTrue(codes.allSatisfy { $0 < net.configuration.startSpeechToken }, "greedy codes are speech codes: \(codes)")
    }

    // MARK: Chatterbox — S3Gen (x-vector, flow, vocoder)

    func testChatterboxS3GenMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_CHATTERBOX_S3GEN"], let directory = config["IK_VAL_CHATTERBOX"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_CHATTERBOX_S3GEN + IK_VAL_CHATTERBOX (run_reference.py chatterbox_s3gen, real weights)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let release = URL(fileURLWithPath: directory)
        func reference(_ key: String) throws -> MLXArray { try XCTUnwrap(arrays[key], "record carries \(key)") }
        @discardableResult
        func check(_ label: String, _ mine: MLXArray, _ reference: MLXArray, _ threshold: Double = 0.9999) -> Double {
            eval(mine)
            XCTAssertEqual(mine.shape, reference.shape, "chatterbox s3gen \(label) shape")
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY chatterbox s3gen: \(label) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, threshold, "chatterbox s3gen \(label) matches the reference")
            return similarity
        }

        let net = NFKMLXS3GenNet()
        try NFKMLXChatterbox.loadS3GenWeights(into: net, from: release.appendingPathComponent("s3gen.safetensors"))

        // The x-vector: Kaldi fbank, the 2-D head, the embedding.
        let wave16 = try reference("ref_wav16").asArray(Float.self)
        let fbank = NFKKaldiFbank.features(wave16)
        check("fbank", fbank, try reference("fbank"))
        let referenceFbank = try reference("fbank").expandedDimensions(axis: 0)
        check("xvector_head", net.speakerEncoder.head(referenceFbank)[0].transposed(1, 0), try reference("xvector_head"))
        check("xvector", net.speakerEncoder(referenceFbank)[0], try reference("xvector"))

        // The 24 kHz prompt mel.
        let wave24 = try reference("ref_wav24").asArray(Float.self)
        check("prompt_mel", NFKChatterboxPromptMel.features(wave24), try reference("prompt_mel"))

        // The flow's conditioning: token embedding, the upsampling conformer, mu, the speaker projection.
        let promptTokens = try reference("prompt_tokens").asArray(Int32.self).map(Int.init)
        let speechTokens = try reference("speech_tokens").asArray(Int32.self).map(Int.init)
        let tokenEmbedding = net.flow.tokenEmbedding(promptTokens + speechTokens)
        check("token_emb", tokenEmbedding[0], try reference("token_emb"))
        let encoded = net.flow.encoder(tokenEmbedding)
        check("encoder_h", encoded[0], try reference("encoder_h"))
        check("mu", net.flow.encoderProjection(encoded)[0], try reference("mu"))
        check("spk80", net.flow.speakerEmbedding(xVector: try reference("xvector"))[0], try reference("spk80"))

        // The estimator on the reference's own conditioning and noise, then the whole Euler solve.
        let mu = try reference("mu").expandedDimensions(axis: 0)
        let speaker = try reference("spk80").expandedDimensions(axis: 0)
        let promptMel = try reference("prompt_mel")
        let total = mu.dim(1)
        let condition = concatenated([promptMel, MLXArray.zeros([total - promptMel.dim(0), 80])], axis: 0).expandedDimensions(axis: 0)
        let noise = try reference("flow_noise").transposed(1, 0).expandedDimensions(axis: 0)
        let span = try reference("t_span").asArray(Float.self)
        for (mine, theirs) in zip(NFKS3FlowNet.timeSpan(steps: 10), span) { XCTAssertEqual(mine, theirs, accuracy: 1e-6) }
        var firstVelocity: MLXArray?
        let melFull = net.flow.solve(noise: noise, mu: mu, speaker: speaker, condition: condition) { step, velocity in
            if step == 0 { firstVelocity = velocity }
        }
        check("estimator0", try XCTUnwrap(firstVelocity).transposed(0, 2, 1), try reference("estimator0"))
        check("mel_full", melFull[0].transposed(1, 0), try reference("mel_full"), 0.999)
        let mel = melFull[0, promptMel.dim(0)...]
        check("mel", mel.transposed(1, 0), try reference("mel"), 0.999)

        // The vocoder: F0, the harmonic source, and the generator, each on the reference's own input.
        let referenceMel = try reference("mel").transposed(1, 0).expandedDimensions(axis: 0)
        let f0 = net.vocoder.f0(mel: referenceMel)
        check("f0", f0[0], try reference("f0"))
        let source = net.vocoder.source(f0: try reference("f0").expandedDimensions(axis: 0), deterministic: true)
        check("source", source[0], try reference("source"), 0.99)
        var decoded = net.vocoder.decode(mel: referenceMel, source: try reference("source").expandedDimensions(axis: 0))[0].asArray(Float.self)
        let trim = net.vocoder.sampleRate / 50
        for index in 0 ..< (2 * trim) {
            decoded[index] *= index < trim ? 0 : (cosf(Float.pi * (1 - Float(index - trim) / Float(trim - 1))) + 1) / 2
        }
        check("wav (reference source)", MLXArray(decoded), try reference("wav"), 0.999)
        let waveform = net.vocoder.waveform(mel: referenceMel[0], deterministicSource: true)
        check("wav (own source)", MLXArray(waveform), try reference("wav"), 0.99)
    }

    // MARK: Chatterbox — the whole pipeline on the released weights

    func testChatterboxSynthesizesSpeechParakeetTranscribes() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_VAL_CHATTERBOX"], let audio = config["IK_VAL_AUDIO"],
              let parakeetDirectory = config["IK_VAL_PARAKEET"], FileManager.default.fileExists(atPath: directory) else {
            throw XCTSkip("set IK_VAL_CHATTERBOX + IK_VAL_AUDIO + IK_VAL_PARAKEET")
        }
        let release = URL(fileURLWithPath: directory)
        let tts = try NFKMLXChatterboxTTS(directoryURL: release)
        let voice = try XCTUnwrap(NFKMLXWaveFile.read(try Data(contentsOf: URL(fileURLWithPath: audio))))
        let conditionals = tts.conditionals(voice: voice.samples, sampleRate: voice.sampleRate)
        XCTAssertEqual(conditionals.t3.speakerEmbedding.shape, [256])
        XCTAssertEqual(conditionals.s3gen.mel.dim(0) / 2, conditionals.s3gen.tokens.count, "the prompt codes are trimmed to half the mel frames")

        let text = "The quick brown fox jumps over the lazy dog."
        let samples = tts.synthesize(text: text, conditionals: conditionals)
        let seconds = Double(samples.count) / Double(tts.sampleRate)
        let rms = sqrt(samples.reduce(0) { $0 + Double($1 * $1) } / Double(max(samples.count, 1)))
        print("VALIDATION PARITY chatterbox-e2e: \(samples.count) samples (\(seconds) s), RMS \(rms)")
        XCTAssertGreaterThan(seconds, 1.5)
        XCTAssertLessThan(seconds, 10)
        XCTAssertGreaterThan(rms, 0.01, "the clip is signal, not silence")

        // Closing the loop: the package's own Parakeet, at parity, must hear the sentence.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("chatterbox-e2e-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: samples, sampleRate: tts.sampleRate, to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let parakeet = try NFKMLXParakeet.backend(directoryURL: URL(fileURLWithPath: parakeetDirectory))
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: seconds, sampleRate: Double(tts.sampleRate), channelCount: 1)
        let heard = try parakeet.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset])).text ?? ""
        print("VALIDATION PARITY chatterbox-e2e: Parakeet hears \(heard.debugDescription)")
        let normalized = heard.lowercased().filter { $0.isLetter || $0 == " " }
        XCTAssertTrue(normalized.contains("quick brown fox"), "Parakeet transcribes the synthesized sentence: \(heard)")
        XCTAssertTrue(normalized.contains("lazy dog"), "Parakeet transcribes the synthesized sentence: \(heard)")

        // The release's built-in voice reads through the native checkpoint reader.
        let builtin = try tts.builtinConditionals(url: release.appendingPathComponent("conds.pt"))
        XCTAssertEqual(builtin.t3.speakerEmbedding.shape, [256])
        XCTAssertEqual(builtin.t3.promptTokens.count, 150)
        XCTAssertEqual(builtin.s3gen.mel.shape, [2 * builtin.s3gen.tokens.count, 80])
        XCTAssertEqual(builtin.s3gen.xVector.shape, [192])
        XCTAssertEqual(builtin.t3.exaggeration, 0.5, accuracy: 1e-6)
    }

    func testChatterboxBackendSpeaksTheBuiltinVoice() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_VAL_CHATTERBOX"], FileManager.default.fileExists(atPath: directory) else {
            throw XCTSkip("set IK_VAL_CHATTERBOX")
        }
        let backend = try NFKMLXChatterbox.backend(directoryURL: URL(fileURLWithPath: directory), voiceURL: nil)
        XCTAssertTrue(backend.isReady)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "Hello there."]))
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        XCTAssertEqual(asset.sampleRate, 24000, accuracy: 0.5)
        XCTAssertGreaterThan(asset.durationSeconds, 0.3)
        print("VALIDATION PARITY chatterbox-backend: \(asset.durationSeconds) s at \(asset.sampleRate) Hz")
    }

    // MARK: Kokoro-82M (StyleTTS2 / iSTFTNet) — text path

    // Kokoro's text path on the RELEASED weights, against the vendored KModel, seam by seam: the PL-BERT
    // (Albert) text encoder, the bert_encoder projection, the DurationEncoder, the duration head, the
    // F0/energy predictor, the TextEncoder, and the alignment-expanded asr. The F0/N/asr seams run over
    // the reference's own integer durations, so they are independent of the duration rounding.
    func testKokoroTextPathMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_KOKORO"], let directory = config["IK_VAL_KOKORO"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_KOKORO + IK_VAL_KOKORO (run_reference.py kokoro, real weights)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let ids = try XCTUnwrap(arrays["ids"]).expandedDimensions(axis: 0)             // [1, T]
        let refS = try XCTUnwrap(arrays["ref_s"]).reshaped([1, 256])
        let predDur = try XCTUnwrap(arrays["pred_dur"]).asArray(Int32.self).map(Int.init)

        let net = NFKMLXKokoroNet(.v1)
        let weightsURL = URL(fileURLWithPath: directory).appendingPathComponent("kokoro-v1_0.pth")
        try NFKMLXKokoro.loadTextWeights(into: net, from: weightsURL)

        let seams = net.textPath(ids: ids, refS: refS, durationsOverride: predDur)
        func check(_ label: String, _ mine: MLXArray, _ key: String, _ threshold: Double = 0.9999) {
            guard let reference = arrays[key] else { return }
            eval(mine)
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY kokoro: \(label) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, threshold, "kokoro \(label) matches the reference")
        }
        check("bert", seams.bert, "bert")
        check("d_en", seams.dEn, "d_en")
        check("d", seams.d, "d")
        check("duration", seams.duration, "duration")
        check("f0", seams.f0, "f0")
        check("n", seams.n, "n")
        check("t_en", seams.tEn, "t_en")
        check("asr", seams.asr, "asr")

        // The iSTFTNet decoder, over the reference's F0/N/asr so it is isolated from the text path: the
        // encode output, the generator input, the conv_post output, and the final waveform.
        let f0 = try XCTUnwrap(arrays["f0"]).reshaped([1, -1])
        let energy = try XCTUnwrap(arrays["n"]).reshaped([1, -1])
        let asr = try XCTUnwrap(arrays["asr"]).expandedDimensions(axis: 0)
        // The sine source and the final audio are limited by float32 `sin` precision: the reference
        // multiplies the accumulated phase by the upsample scale (≈18000 rad) before `sin`, where a
        // float32 argument holds ~two fractional digits, so a torch/MLX rounding difference bounds the
        // waveform cosine near 0.99. The encode/gen-in seams are exact; the reconstruction is close.
        check("har_source", net.decoder.generator.sineSource(f0: f0), "dec_har_source", 0.999)
        let (audio, decSeams) = net.decoder.callWithSeams(asr, f0Curve: f0, n: energy, style: refS[0..., 0 ..< 128])
        check("dec_encode", decSeams.encode, "dec_encode")
        check("dec_gen_in", decSeams.genIn, "dec_gen_in")
        check("dec_conv_post", decSeams.convPost, "dec_conv_post", 0.9999)
        check("audio", audio.reshaped([-1]), "output", 0.99)

        // The full consumer path: load the vocabulary and the voicepack, synthesize from the phoneme
        // string (the model's own rounded durations), and reproduce the reference waveform. This
        // additionally exercises loadVocab, loadVoice, and the per-scalar phoneme mapping.
        let vocab = try NFKMLXKokoro.loadVocab(directoryURL: URL(fileURLWithPath: directory))
        let voice = try NFKMLXKokoro.loadVoice(url: URL(fileURLWithPath: directory).appendingPathComponent("voices/af_heart.pt"))
        let phonemeAudio = net.synthesize(phonemes: "həlˈoʊ wˈɜːld", voice: voice, vocab: vocab)
        check("phoneme_audio", phonemeAudio.reshaped([-1]), "output", 0.99)
    }

    // MARK: Wan 3D causal VAE

    // The Wan 3D causal VAE at a tiny random configuration, against diffusers' AutoencoderKLWan (the
    // Wan 2.2 residual path). A 5-frame clip exercises the stateful feat_cache streaming — chunked encode
    // (1 then 4 frames) and per-frame decode, threading the causal convolutions' temporal cache — plus the
    // temporal resample time_convs and the AvgDown3D / DupUp3D shortcuts. The RMS `gamma` loads flattened
    // from its `[C,1,1,1]` layout; the 5-D and 4-D convolution weights load transposed to NDHWC/NHWC.
    func testWanVideoVAEMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_WAN_VAE"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_WAN_VAE (run_reference.py wan_vae)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let video = try XCTUnwrap(arrays["video"]).expandedDimensions(axis: 0)         // [1, T, H, W, 3]
        let referenceLatent = try XCTUnwrap(arrays["latent"])
        let referenceDecoded = try XCTUnwrap(arrays["output"])

        let net = NFKMLXWanVideoVAENet(.tiny)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = String(key.dropFirst(3))
            if name.hasSuffix(".gamma") { return (name, value.reshaped([-1])) }
            if value.ndim == 5 { return (name, value.transposed(0, 2, 3, 4, 1)) }
            if value.ndim == 4 { return (name, value.transposed(0, 2, 3, 1)) }
            return (name, value)
        }
        try NFKMLXWeights.apply(weights, to: net)

        func seam(_ key: String, _ produce: () -> MLXArray) {
            guard let reference = arrays[key] else { return }
            let mine = produce(); eval(mine)
            print("SEAM wan-vae \(key): cosine \(cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init), reference.reshaped([-1]).asArray(Float.self).map(Double.init)))")
        }
        seam("enc_raw") { net.encodeMoments(video) }
        seam("enc_1frame") { net.encodeMoments(video[0..., 0 ..< 1, 0..., 0..., 0...]) }
        let latentF0 = referenceLatent.expandedDimensions(axis: 0)[0..., 0 ..< 1, 0..., 0..., 0...]
        let stages = net.decodeStages(latentF0)
        seam("dec_conv_in") { stages.convIn }
        seam("dec_mid") { stages.mid }
        seam("dec_up0") { stages.up0 }
        seam("decoded_f0") { net.decode(latentF0) }

        let latent = net.encode(video); eval(latent)
        let latentSimilarity = cosine(latent.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceLatent.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY wan-vae: latent cosine \(latentSimilarity)")
        XCTAssertGreaterThan(latentSimilarity, 0.9999, "the encoded latent matches the reference")

        let decoded = net.decode(referenceLatent.expandedDimensions(axis: 0)); eval(decoded)
        let decodeSimilarity = cosine(decoded.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceDecoded.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY wan-vae: decode cosine \(decodeSimilarity)")
        XCTAssertGreaterThan(decodeSimilarity, 0.9999, "the decode matches the reference")
    }

    // The Wan 2.1 autoencoder (the non-residual path) at a tiny random configuration, against diffusers'
    // AutoencoderKLWan. Exercises the flat down-block list, the halving upsampler, and the no-patchify
    // path — the pieces that differ from the 2.2 residual VAE.
    func testWanVideoVAE21MatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_WAN_VAE_21"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_WAN_VAE_21 (run_reference.py wan_vae_21)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let video = try XCTUnwrap(arrays["video"]).expandedDimensions(axis: 0)
        let referenceLatent = try XCTUnwrap(arrays["latent"])
        let referenceDecoded = try XCTUnwrap(arrays["output"])

        let net = NFKMLXWanVideoVAENet(.tiny21)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = String(key.dropFirst(3))
            if name.hasSuffix(".gamma") { return (name, value.reshaped([-1])) }
            if value.ndim == 5 { return (name, value.transposed(0, 2, 3, 4, 1)) }
            if value.ndim == 4 { return (name, value.transposed(0, 2, 3, 1)) }
            return (name, value)
        }
        try NFKMLXWeights.apply(weights, to: net)

        let latent = net.encode(video); eval(latent)
        let latentSimilarity = cosine(latent.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceLatent.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY wan-vae-21: latent cosine \(latentSimilarity)")
        XCTAssertGreaterThan(latentSimilarity, 0.9999, "the 2.1 encoded latent matches the reference")

        let decoded = net.decode(referenceLatent.expandedDimensions(axis: 0)); eval(decoded)
        let decodeSimilarity = cosine(decoded.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceDecoded.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY wan-vae-21: decode cosine \(decodeSimilarity)")
        XCTAssertGreaterThan(decodeSimilarity, 0.9999, "the 2.1 decode matches the reference")
    }

    // MARK: DPM-Solver++ (SANA's sampler)

    // SANA's released DPMSolverMultistepScheduler in its flow-prediction configuration, against diffusers'
    // own, over a fixed velocity sequence (no model). The sigma schedule is checked exactly; the sample
    // trajectory is checked step by step, so the flow→x0 conversion and the first/second-order multistep
    // updates are all covered.
    func testDPMSolverSchedulerMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DPM_SOLVER"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_DPM_SOLVER (run_reference.py dpm_solver)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let referenceSigmas = try XCTUnwrap(arrays["sigmas"]).asArray(Float.self)
        let referenceTimesteps = try XCTUnwrap(arrays["timesteps"]).asArray(Float.self)
        let initial = try XCTUnwrap(arrays["initial"])
        let velocities = try XCTUnwrap(arrays["velocities"])                // [steps, ...]
        let trajectory = try XCTUnwrap(arrays["trajectory"])               // [steps, ...]
        let steps = referenceTimesteps.count

        var scheduler = NFKMLXDPMSolverScheduler(.sana)
        scheduler.setTimesteps(steps)

        let sigmaError = zip(scheduler.sigmas, referenceSigmas).map { abs($0 - $1) }.max() ?? 0
        let timestepError = zip(scheduler.timesteps, referenceTimesteps).map { abs($0 - $1) }.max() ?? 0
        print("VALIDATION PARITY dpm-solver: worst sigma \(sigmaError), worst timestep \(timestepError)")
        XCTAssertLessThan(sigmaError, 1e-4, "the sigma schedule matches the reference")
        XCTAssertLessThan(timestepError, 1e-3, "the timesteps match the reference")

        var sample = initial
        var worst = 0.0
        for i in 0 ..< steps {
            sample = scheduler.step(velocity: velocities[i], sample: sample, index: i)
            eval(sample)
            let reference = trajectory[i].reshaped([-1]).asArray(Float.self).map(Double.init)
            let mine = sample.reshaped([-1]).asArray(Float.self).map(Double.init)
            worst = max(worst, zip(mine, reference).map { abs($0 - $1) }.max() ?? 0)
        }
        print("VALIDATION PARITY dpm-solver: worst trajectory |difference| \(worst)")
        XCTAssertLessThan(worst, 1e-4, "the sample trajectory matches the reference DPM-Solver++")
    }

    // MARK: UniPC (Wan's sampler)

    // Wan's released UniPCMultistepScheduler in its flow-prediction configuration, against diffusers' own,
    // over a fixed velocity sequence. The sigma schedule is exact and the sample trajectory is checked
    // step by step, so the predictor, the corrector, and the order warm-up are all covered.
    func testUniPCSchedulerMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_UNIPC"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_UNIPC (run_reference.py unipc)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let referenceSigmas = try XCTUnwrap(arrays["sigmas"]).asArray(Float.self)
        let initial = try XCTUnwrap(arrays["initial"])
        let velocities = try XCTUnwrap(arrays["velocities"])
        let trajectory = try XCTUnwrap(arrays["trajectory"])
        let steps = try XCTUnwrap(arrays["timesteps"]).asArray(Float.self).count

        var scheduler = NFKMLXUniPCScheduler(.wan)
        scheduler.setTimesteps(steps)
        let sigmaError = zip(scheduler.sigmas, referenceSigmas).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(sigmaError, 1e-4, "the sigma schedule matches the reference")

        var sample = initial
        var worst = 0.0
        for i in 0 ..< steps {
            sample = scheduler.step(velocity: velocities[i], sample: sample, index: i)
            eval(sample)
            let reference = trajectory[i].reshaped([-1]).asArray(Float.self).map(Double.init)
            let mine = sample.reshaped([-1]).asArray(Float.self).map(Double.init)
            worst = max(worst, zip(mine, reference).map { abs($0 - $1) }.max() ?? 0)
        }
        print("VALIDATION PARITY unipc: worst sigma \(sigmaError), worst trajectory |difference| \(worst)")
        XCTAssertLessThan(worst, 1e-4, "the sample trajectory matches the reference UniPC")
    }

    // MARK: Gemma 2 (SANA's text encoder)

    // The Gemma-2 text decoder at a tiny random configuration, against transformers' Gemma2Model. The
    // small sliding window makes the alternating sliding/full layers differ, so the soft-capped GQA, the
    // sandwich norms, and the (1+w) RMS are all covered. Module keys are the checkpoint's, no transpose.
    func testGemma2MatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA2"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_GEMMA2 (run_reference.py gemma2)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asType(.int32)
        let reference = try XCTUnwrap(arrays["output"])

        let net = NFKMLXGemma2Net(.tiny)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value) : nil
        }
        try NFKMLXWeights.apply(weights, to: net)

        let output = net(tokens); eval(output)
        let similarity = cosine(output.reshaped([-1]).asArray(Float.self).map(Double.init),
                                reference.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY gemma2: last hidden cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the last hidden state matches the reference Gemma2Model")
    }

    // MARK: umT5 (Wan's text encoder)

    // umT5 at a tiny random configuration, against transformers' UMT5EncoderModel. umT5 differs from plain
    // T5 in giving every layer its own relative-position bias, which this exercises; the module keys are
    // the checkpoint's (each block carries a bias table), no transpose.
    func testUMT5MatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_UMT5"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_UMT5 (run_reference.py umt5)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asType(.int32).reshaped([1, -1])
        let reference = try XCTUnwrap(arrays["output"])

        let net = NFKMLXT5Encoder.makeNet(.tinyUMT5)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value) : nil
        }
        try NFKMLXWeights.apply(weights, to: net)

        let output = net(tokens); eval(output)
        let similarity = cosine(output.reshaped([-1]).asArray(Float.self).map(Double.init),
                                reference.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY umt5: text embedding cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the text embedding matches the reference umT5 encoder")
    }

    // MARK: The models validated by eye

    // These four ran end to end on real weights and looked right, which is exactly the evidence that
    // passed SAM and Depth Anything while both were missing their input normalization. Each runs through
    // its public backend, so the preprocessing is inside the comparison.

    func testU2NetMatchesTheReferenceSaliency() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_U2NET")
        let backend = try NFKMLXU2Net.backend(variant: .full, weightsURL: weights("IK_VAL_U2NET"))
        let matte = grayValues(try pixels(from: backend, plate: inputTensor, key: NFKOutputMask))
        let reference = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)

        XCTAssertEqual(matte.count, reference.count, "same matte size as the reference")
        let similarity = cosine(matte, reference)
        let meanAbsolute = zip(matte, reference).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(matte.count)
        print("VALIDATION PARITY u2net: cosine \(similarity), mean |difference| \(meanAbsolute)")
        XCTAssertGreaterThan(similarity, 0.99, "the saliency matte matches the reference implementation")
        XCTAssertLessThan(meanAbsolute, 0.02, "and agrees pixelwise")
    }

    // The colorized image goes through two stages that can each be wrong on their own: the network's ab
    // prediction and the Lab round trip that recombines it with the original lightness. The record
    // carries the reference's ab, so a mismatch says which.
    func testColorizerMatchesTheReferenceColorization() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_COLORIZER")
        let arrays = try loadArrays(url: URL(fileURLWithPath: config["IK_PARITY_COLORIZER"]!))
        let net = NFKMLXColorizer.makeNet()
        try NFKMLXColorizer.loadWeights(into: net, from: weights("IK_VAL_COLORIZER"))

        if let referenceAb = arrays["ab"] {
            let ours = net.abPrediction(inputTensor)
            eval(ours)
            let mine = ours.reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = referenceAb.reshaped([-1]).asArray(Float.self).map(Double.init)
            let abSimilarity = cosine(mine, theirs)
            print("VALIDATION PARITY colorizer: ab cosine \(abSimilarity)")
            XCTAssertGreaterThan(abSimilarity, 0.99, "the network's ab prediction matches the reference")
        }

        let backend = try NFKMLXColorizer.backend(weightsURL: weights("IK_VAL_COLORIZER"))
        let ours = rgbValues(try pixels(from: backend, plate: inputTensor))
        let reference = referenceOutput.reshaped([-1]).asArray(Float.self).map { Double(max(0, min(1, $0))) }
        let similarity = cosine(ours, reference)
        let meanAbsolute = zip(ours, reference).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(ours.count)
        print("VALIDATION PARITY colorizer: cosine \(similarity), mean |difference| \(meanAbsolute)")
        XCTAssertGreaterThan(similarity, 0.99, "the colorized image matches the reference implementation")
        XCTAssertLessThan(meanAbsolute, 0.02, "and agrees pixelwise")
    }

    func testZeroDCEMatchesTheReferenceEnhancement() throws {
        try requireMLXRuntime()
        try assertImageParity("IK_PARITY_ZERODCE", model: "zero-dce") {
            try NFKMLXZeroDCE.backend(weightsURL: self.weights("IK_VAL_ZERODCE"))
        }
    }

    func testStyleTransferMatchesTheReferenceStylization() throws {
        try requireMLXRuntime()
        try assertImageParity("IK_PARITY_STYLE", model: "style-transfer") {
            try NFKMLXStyleTransfer.backend(weightsURL: self.weights("IK_VAL_STYLE"))
        }
    }

    func testRealESRGANMatchesTheReferenceUpscale() throws {
        try requireMLXRuntime()
        try assertImageParity("IK_PARITY_REALESRGAN", model: "real-esrgan") {
            try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: self.weights("IK_VAL_REALESRGAN"))
        }
    }

    /// Scores an image-to-image model's public output against its reference record.
    private func assertImageParity(_ key: String, model: String,
                                   build: () throws -> any NFKInferenceBackend) throws {
        let (inputTensor, referenceOutput) = try record(key)
        let ours = rgbValues(try pixels(from: build(), plate: inputTensor))
        let reference = referenceOutput.reshaped([-1]).asArray(Float.self)
            .map { Double(max(0, min(1, $0))) }                       // the port clamps before it writes

        XCTAssertEqual(ours.count, reference.count, "same output size as the reference")
        let similarity = cosine(ours, reference)
        let meanAbsolute = zip(ours, reference).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(ours.count)
        print("VALIDATION PARITY \(model): cosine \(similarity), mean |difference| \(meanAbsolute)")
        XCTAssertGreaterThan(similarity, 0.99, "\(model) matches the reference implementation")
        XCTAssertLessThan(meanAbsolute, 0.02, "and agrees pixelwise")
    }

    // MARK: DeepLab

    // Segmentation logits at the head's native stride-8 resolution, against torchvision's own model.
    // Comparing before the argmax keeps the check sensitive — a label map hides every difference too
    // small to flip a pixel's winner.
    func testDeepLabMatchesTheReferenceLogits() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_DEEPLAB")
        let net = NFKMLXDeepLab.makeNet()
        try NFKMLXDeepLab.loadWeights(into: net, from: weights("IK_VAL_DEEPLAB"))

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let ours = net.logits(NFKMLXDeepLabNet.normalized(inputTensor).reshaped([1, height, width, 3]))
        eval(ours)

        XCTAssertEqual(Array(ours.shape.dropFirst()), referenceOutput.shape, "same logit grid as the reference")
        let mine = ours.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)

        let classes = referenceOutput.shape[2]
        let ourLabels = ours.reshaped([-1, classes]).argMax(axis: -1).asArray(Int32.self)
        let referenceLabels = referenceOutput.reshaped([-1, classes]).argMax(axis: -1).asArray(Int32.self)
        let agreement = Double(zip(ourLabels, referenceLabels).filter(==).count) / Double(ourLabels.count)
        print("VALIDATION PARITY deeplab: logit cosine \(similarity), label agreement \(agreement)")
        XCTAssertGreaterThan(similarity, 0.99, "the class logits match the reference implementation")
        XCTAssertGreaterThan(agreement, 0.99, "and the predicted labels agree")
    }

    // MARK: Robust Video Matting

    // The matte against PeterL1n's own MattingNetwork, on the released rvm_mobilenetv3 weights. The
    // full-resolution pass measures the encoder, LR-ASPP, and recurrent decoder; the downsampled pass
    // routes through the deep-guided-filter refiner the first pass never touches, so both halves of
    // the model are scored rather than one hiding behind the other.
    func testRVMMatchesTheReferenceMatte() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceAlpha) = try record("IK_PARITY_RVM")
        guard let path = config["IK_PARITY_RVM"] else { throw XCTSkip("set IK_PARITY_RVM") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))

        let net = NFKMLXRVMNet()
        try NFKMLXRVM.loadWeights(into: net, from: weights("IK_VAL_RVM"))
        net.train(false)

        let frame = inputTensor.reshaped([1, inputTensor.shape[0], inputTensor.shape[1], 3])
        let (foreground, alpha, _) = net.forward(frame, state: NFKMLXRVMNet.initialState)
        eval(foreground, alpha)

        let mine = alpha.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceAlpha.reshaped([-1]).asArray(Float.self).map(Double.init)
        XCTAssertEqual(mine.count, theirs.count, "same matte size as the reference")
        let similarity = cosine(mine, theirs)
        let error = zip(mine, theirs).map { abs($0 - $1) }.reduce(0, +) / Double(mine.count)
        print("VALIDATION PARITY rvm: alpha cosine \(similarity), mean |difference| \(error)")
        XCTAssertGreaterThan(similarity, 0.99, "the alpha matches the reference implementation")
        XCTAssertLessThan(error, 0.05, "and agrees pointwise, not just in shape")

        if let referenceForeground = arrays["foreground"] {
            let foregroundSimilarity = cosine(foreground.reshaped([-1]).asArray(Float.self).map(Double.init),
                                              referenceForeground.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY rvm: foreground cosine \(foregroundSimilarity)")
            XCTAssertGreaterThan(foregroundSimilarity, 0.99, "the composited foreground matches too")
        }

        if let refinedAlpha = arrays["alpha_downsampled"] {
            let (_, halfAlpha, _) = net.forward(frame, state: NFKMLXRVMNet.initialState, downsampleRatio: 0.5)
            eval(halfAlpha)
            let refinedSimilarity = cosine(halfAlpha.reshaped([-1]).asArray(Float.self).map(Double.init),
                                           refinedAlpha.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY rvm: refined (downsample 0.5) alpha cosine \(refinedSimilarity)")
            XCTAssertGreaterThan(refinedSimilarity, 0.99, "the guided-filter pass matches as well")
        }
    }

    // MARK: CodeFormer

    // The restored face against sczhou/CodeFormer's own architecture on the released codeformer.pth,
    // at the real inference settings (fidelity 0.5, AdaIN on). The code-prediction logits are scored
    // first: they are the continuous seam that localizes a mismatch to encoder+transformer vs
    // generator, and they are robust to argmax near-ties the final image is not.
    func testCodeFormerMatchesTheReferenceRestoration() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_CODEFORMER")
        guard let path = config["IK_PARITY_CODEFORMER"] else { throw XCTSkip("set IK_PARITY_CODEFORMER") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))

        let net = NFKMLXCodeFormerNet(.base)
        try NFKMLXCodeFormer.loadWeights(into: net, from: weights("IK_VAL_CODEFORMER"))
        net.train(false)

        if let referenceLogits = arrays["logits"] {
            var latent = inputTensor.reshaped([1, inputTensor.shape[0], inputTensor.shape[1], 3]) * 2 - 1
            for block in net.encoder.blocks {
                latent = block(latent)
            }
            let logits = net.codeLogits(latent)
            eval(logits)
            let logitSimilarity = cosine(logits.reshaped([-1]).asArray(Float.self).map(Double.init),
                                         referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init))
            let ourIndices = logits.argMax(axis: -1).asArray(Int32.self)
            let referenceIndices = referenceLogits.argMax(axis: -1).asArray(Int32.self)
            let agreement = Double(zip(ourIndices, referenceIndices).filter(==).count) / Double(ourIndices.count)
            print("VALIDATION PARITY codeformer: logit cosine \(logitSimilarity), code agreement \(agreement)")
            XCTAssertGreaterThan(logitSimilarity, 0.99, "the code logits match the reference implementation")
            XCTAssertGreaterThan(agreement, 0.99, "and the predicted codes agree")
        }

        let restored = net.restore(inputTensor)
        eval(restored)
        XCTAssertEqual(restored.shape, referenceOutput.shape, "same restored size as the reference")
        let mine = restored.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        let error = zip(mine, theirs).map { abs($0 - $1) }.reduce(0, +) / Double(mine.count)
        print("VALIDATION PARITY codeformer: restored cosine \(similarity), mean |difference| \(error)")
        XCTAssertGreaterThan(similarity, 0.99, "the restored face matches the reference implementation")
        XCTAssertLessThan(error, 0.05, "and agrees pointwise, not just in shape")

        // The public backend path as a consumer runs it: the CGImage bridge, the factory's fidelity,
        // and the output conversion are all outside the network, so scoring the net alone would leave
        // them unverified.
        try assertImageParity("IK_PARITY_CODEFORMER", model: "codeformer backend") {
            try NFKMLXCodeFormer.backend(weightsURL: weights("IK_VAL_CODEFORMER"))
        }
    }

    // MARK: YOLO

    // The pre-suppression prediction tensor against ultralytics' own YOLOv8n: box centers and sizes in
    // pixels plus class probabilities for all 8400 anchors. Comparing before thresholding and NMS
    // keeps the check sensitive — a wrong DFL decode or anchor grid cannot hide behind a lucky
    // suppression. The boxes and the class probabilities are scored separately because their scales
    // differ by orders of magnitude.
    func testYOLOMatchesTheReferencePredictions() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_YOLO")

        let net = NFKMLXYOLONet(.base)
        try NFKMLXYOLO.loadWeights(into: net, from: weights("IK_VAL_YOLO"))
        net.train(false)

        let ours = net.predictions(inputTensor)
        eval(ours)
        XCTAssertEqual(ours.shape, referenceOutput.shape, "same anchor count and row width as the reference")

        let ourBoxes = ours[0..., 0 ..< 4].reshaped([-1]).asArray(Float.self).map(Double.init)
        let referenceBoxes = referenceOutput[0..., 0 ..< 4].reshaped([-1]).asArray(Float.self).map(Double.init)
        let boxSimilarity = cosine(ourBoxes, referenceBoxes)
        let ourClasses = ours[0..., 4...].reshaped([-1]).asArray(Float.self).map(Double.init)
        let referenceClasses = referenceOutput[0..., 4...].reshaped([-1]).asArray(Float.self).map(Double.init)
        let classSimilarity = cosine(ourClasses, referenceClasses)
        print("VALIDATION PARITY yolo: box cosine \(boxSimilarity), class cosine \(classSimilarity)")
        XCTAssertGreaterThan(boxSimilarity, 0.99, "the decoded boxes match the reference implementation")
        XCTAssertGreaterThan(classSimilarity, 0.99, "and so do the class probabilities")

        // The best anchor's class must agree — the number a consumer actually sees.
        let ourBest = ourClasses.enumerated().max { $0.element < $1.element }!
        let referenceBest = referenceClasses.enumerated().max { $0.element < $1.element }!
        XCTAssertEqual(ourBest.offset % 80, referenceBest.offset % 80, "same top class")
        XCTAssertEqual(ourBest.offset / 80, referenceBest.offset / 80, "at the same anchor")
    }

    // Detections on a NON-SQUARE frame, through the public backend, against ultralytics' own
    // `predict`. The square-plate record above cannot see this path at all: letterboxing is an
    // identity there, so the aspect-preserving scale, the gray padding to a stride multiple, and the
    // mapping back to the caller's own coordinates are only exercised here.
    func testYOLOMatchesTheReferenceDetectionsOnANonSquareFrame() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_YOLO_DETECTIONS"] else { throw XCTSkip("set IK_PARITY_YOLO_DETECTIONS") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let plate = try XCTUnwrap(arrays["plate"])                        // [H, W, 3], 16:9
        let referenceBoxes = try XCTUnwrap(arrays["output"])              // [detections, 4] xyxy normalized
        let referenceClasses = try XCTUnwrap(arrays["classes"]).asArray(Int32.self)

        let backend = try NFKMLXYOLO.backend(weightsURL: weights("IK_VAL_YOLO"), labels: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: try image(from: plate)]))
        let ours = try XCTUnwrap(result.detections)

        XCTAssertEqual(ours.count, referenceClasses.count, "same number of surviving detections")
        let theirs = referenceBoxes.asArray(Float.self)
        var worstOverlap = 1.0
        for (index, detection) in ours.enumerated() where index < referenceClasses.count {
            XCTAssertEqual(Int32(detection.classIndex), referenceClasses[index],
                           "detection \(index) is the same class as the reference's")
            let reference = CGRect(x: CGFloat(theirs[index * 4]), y: CGFloat(theirs[index * 4 + 1]),
                                   width: CGFloat(theirs[index * 4 + 2] - theirs[index * 4]),
                                   height: CGFloat(theirs[index * 4 + 3] - theirs[index * 4 + 1]))
            worstOverlap = min(worstOverlap,
                               Double(NFKMLXYOLONet.intersectionOverUnion(detection.boundingBox, reference)))
        }
        print("VALIDATION PARITY yolo (non-square): \(ours.count) detections, worst box IoU \(worstOverlap)")
        XCTAssertGreaterThan(worstOverlap, 0.98, "every box lands where the reference put it")
    }

    // MARK: RIFE

    // Frame interpolation against the released HDv3 IFNet. The module previously implemented a
    // superseded IFBlock — a flat eight-convolution trunk with one residual and a single combined
    // head — which no released checkpoint fits; this scores the rewrite against the real weights.
    func testRIFEMatchesTheReferenceInterpolation() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_RIFE")
        guard let path = config["IK_PARITY_RIFE"] else { throw XCTSkip("set IK_PARITY_RIFE") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let second = try XCTUnwrap(arrays["frame1"])

        let net = NFKMLXRIFE.makeNet()
        try NFKMLXRIFE.loadWeights(into: net, from: weights("IK_VAL_RIFE"))

        let ours = net.interpolate(inputTensor, second)
        eval(ours)
        XCTAssertEqual(ours.shape, referenceOutput.shape, "same frame size as the reference")
        let mine = ours.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        let error = zip(mine, theirs).map { abs($0 - $1) }.reduce(0, +) / Double(mine.count)
        print("VALIDATION PARITY rife: interpolated cosine \(similarity), mean |difference| \(error)")
        XCTAssertGreaterThan(similarity, 0.99, "the interpolated frame matches the reference implementation")
        XCTAssertLessThan(error, 0.02, "and agrees pixelwise, not just in shape")

        // The public path a consumer actually calls: two CGImages in under the frame keys, one out
        // under NFKOutputImage. The tensor backend's bridging and port wiring sit outside the network,
        // so scoring `interpolate` alone leaves them unmeasured.
        let backend = try NFKMLXRIFE.backend(weightsURL: weights("IK_VAL_RIFE"))
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [
            NFKMLXRIFE.frame0Key: try image(from: inputTensor),
            NFKMLXRIFE.frame1Key: try image(from: second),
        ]))
        let throughBackend = rgbValues(try XCTUnwrap(result.output(forKey: NFKOutputImage).map { $0 as! CGImage }))
        let referencePixels = referenceOutput.reshaped([-1]).asArray(Float.self).map { Double(max(0, min(1, $0))) }
        XCTAssertEqual(throughBackend.count, referencePixels.count, "same frame size through the backend")
        let backendSimilarity = cosine(throughBackend, referencePixels)
        let backendError = zip(throughBackend, referencePixels).map { abs($0 - $1) }.reduce(0, +) / Double(throughBackend.count)
        print("VALIDATION PARITY rife backend: cosine \(backendSimilarity), mean |difference| \(backendError)")
        XCTAssertGreaterThan(backendSimilarity, 0.99, "the backend path matches the reference too")
        XCTAssertLessThan(backendError, 0.02, "within the 8-bit bridge's rounding")
    }

    // A frame whose sides are not a multiple of 32 takes the padding path: RIFE edge-pads up to the
    // coarsest scale's stride and crops back. The parity plate is 256 — a multiple — so the main
    // record cannot speak for an arbitrary frame size.
    func testRIFEMatchesTheReferenceOnAFrameThatNeedsPadding() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_RIFE_ODD"] else { throw XCTSkip("set IK_PARITY_RIFE_ODD") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let first = try XCTUnwrap(arrays["input_image"])
        let second = try XCTUnwrap(arrays["frame1"])
        let referenceOutput = try XCTUnwrap(arrays["output"])

        let net = NFKMLXRIFE.makeNet()
        try NFKMLXRIFE.loadWeights(into: net, from: weights("IK_VAL_RIFE"))
        let ours = net.interpolate(first, second)
        eval(ours)
        XCTAssertEqual(ours.shape, referenceOutput.shape, "cropped back to the frame it was given")
        let similarity = cosine(ours.reshaped([-1]).asArray(Float.self).map(Double.init),
                                referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY rife (needs padding): cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the pad-and-crop path matches the reference")
    }

    // MARK: Shipped variants
    //
    // Each of these shares its code with a variant already at parity and differs only in a preset, so
    // the risk is a wrong preset rather than wrong arithmetic — which is exactly what a checkpoint
    // load plus a reference comparison catches.

    // Depth Anything's Base encoder: the same code as Small at a different width and depth, so what
    // this really checks is that the preset matches the release it claims to.
    func testDepthAnythingBaseMatchesTheReferenceMap() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_DEPTH_BASE")
        let backend = try NFKMLXDepthAnything.backend(variant: .base, weightsURL: weights("IK_VAL_DEPTH_BASE"))
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: try image(from: inputTensor)]))
        let ours = grayValues(try XCTUnwrap(result.output(forKey: NFKOutputImage).map { $0 as! CGImage }))
        let reference = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)

        XCTAssertEqual(ours.count, reference.count, "same map size as the reference")
        let similarity = cosine(ours, reference)
        let meanAbsolute = zip(ours, reference).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(ours.count)
        print("VALIDATION PARITY depth-base: cosine \(similarity), mean |difference| \(meanAbsolute)")
        XCTAssertGreaterThan(similarity, 0.99, "the Base preset matches the released Base model")
        XCTAssertLessThan(meanAbsolute, 0.05, "and agrees pointwise")
    }

    // YOLOv8s: the same depth as the nano model at twice the width. The releases scale by two
    // independent multiples, so a size is only right if both were read correctly.
    func testYOLOSmallMatchesTheReferencePredictions() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_YOLO_SMALL")

        let net = NFKMLXYOLONet(.small)
        try NFKMLXYOLO.loadWeights(into: net, from: weights("IK_VAL_YOLO_SMALL"))
        net.train(false)

        let ours = net.predictions(inputTensor)
        eval(ours)
        XCTAssertEqual(ours.shape, referenceOutput.shape, "same anchor count and row width as the reference")
        let boxSimilarity = cosine(ours[0..., 0 ..< 4].reshaped([-1]).asArray(Float.self).map(Double.init),
                                   referenceOutput[0..., 0 ..< 4].reshaped([-1]).asArray(Float.self).map(Double.init))
        let classSimilarity = cosine(ours[0..., 4...].reshaped([-1]).asArray(Float.self).map(Double.init),
                                     referenceOutput[0..., 4...].reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY yolo-s: box cosine \(boxSimilarity), class cosine \(classSimilarity)")
        XCTAssertGreaterThan(boxSimilarity, 0.99, "the decoded boxes match the released small model")
        XCTAssertGreaterThan(classSimilarity, 0.99, "and so do the class probabilities")
    }

    // NAFNet's REDS release: the GoPro block distribution at twice the width, so it separates a wrong
    // width from a wrong block layout — the two ways a preset can be wrong.
    func testNAFNetREDSMatchesTheReferenceRestoration() throws {
        try requireMLXRuntime()
        try assertImageParity("IK_PARITY_NAFNET_REDS", model: "nafnet reds") {
            try NFKMLXNAFNet.backend(variant: .reds, weightsURL: weights("IK_VAL_NAFNET_REDS"))
        }
    }

    // YOLOv8m is the first size where BOTH multiples change: wider stages and deeper C2f repeats.
    func testYOLOMediumMatchesTheReferencePredictions() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_YOLO_MEDIUM")

        let net = NFKMLXYOLONet(.medium)
        try NFKMLXYOLO.loadWeights(into: net, from: weights("IK_VAL_YOLO_MEDIUM"))
        net.train(false)

        let ours = net.predictions(inputTensor)
        eval(ours)
        XCTAssertEqual(ours.shape, referenceOutput.shape, "same anchor count and row width as the reference")
        let boxSimilarity = cosine(ours[0..., 0 ..< 4].reshaped([-1]).asArray(Float.self).map(Double.init),
                                   referenceOutput[0..., 0 ..< 4].reshaped([-1]).asArray(Float.self).map(Double.init))
        let classSimilarity = cosine(ours[0..., 4...].reshaped([-1]).asArray(Float.self).map(Double.init),
                                     referenceOutput[0..., 4...].reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY yolo-m: box cosine \(boxSimilarity), class cosine \(classSimilarity)")
        XCTAssertGreaterThan(boxSimilarity, 0.99, "the decoded boxes match the released medium model")
        XCTAssertGreaterThan(classSimilarity, 0.99, "and so do the class probabilities")
    }

    // Depth Anything's Large encoder — the deepest released size, and the last unmeasured preset.
    func testDepthAnythingLargeMatchesTheReferenceMap() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_DEPTH_LARGE")
        let backend = try NFKMLXDepthAnything.backend(variant: .large, weightsURL: weights("IK_VAL_DEPTH_LARGE"))
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: try image(from: inputTensor)]))
        let ours = grayValues(try XCTUnwrap(result.output(forKey: NFKOutputImage).map { $0 as! CGImage }))
        let reference = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)

        XCTAssertEqual(ours.count, reference.count, "same map size as the reference")
        let similarity = cosine(ours, reference)
        let meanAbsolute = zip(ours, reference).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(ours.count)
        print("VALIDATION PARITY depth-large: cosine \(similarity), mean |difference| \(meanAbsolute)")
        XCTAssertGreaterThan(similarity, 0.99, "the Large preset matches the released Large model")
        XCTAssertLessThan(meanAbsolute, 0.05, "and agrees pointwise")
    }

    // NAFNet's GoPro deblurrer: the same width as the SIDD denoiser, but twenty-eight of its blocks
    // sit in the last encoder stage rather than the middle. A checkpoint only fits the geometry it was
    // trained as, so this is the check that the preset describes the release.
    func testNAFNetGoProMatchesTheReferenceRestoration() throws {
        try requireMLXRuntime()
        try assertImageParity("IK_PARITY_NAFNET_GOPRO", model: "nafnet gopro") {
            try NFKMLXNAFNet.backend(variant: .goPro, weightsURL: weights("IK_VAL_NAFNET_GOPRO"))
        }
    }

    // Real-ESRGAN's anime release is a six-block generator where the general one has twenty-three.
    func testRealESRGANAnimeMatchesTheReferenceUpscale() throws {
        try requireMLXRuntime()
        try assertImageParity("IK_PARITY_REALESRGAN_ANIME", model: "real-esrgan anime") {
            try NFKMLXRealESRGAN.backend(variant: .anime, weightsURL: weights("IK_VAL_REALESRGAN_ANIME"))
        }
    }

    // `u2netp` is the light saliency network — a separate class in the reference, not a configuration.
    func testU2NetLightMatchesTheReferenceSaliency() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_U2NETP")
        let backend = try NFKMLXU2Net.backend(variant: .light, weightsURL: weights("IK_VAL_U2NETP"))
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: try image(from: inputTensor)]))
        let ours = grayValues(try XCTUnwrap(result.output(forKey: NFKOutputMask).map { $0 as! CGImage }))
        let reference = referenceOutput.reshaped([-1]).asArray(Float.self).map { Double(max(0, min(1, $0))) }

        XCTAssertEqual(ours.count, reference.count, "same saliency map size as the reference")
        let similarity = cosine(ours, reference)
        print("VALIDATION PARITY u2netp: cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the light network matches the reference implementation")
    }

    // MARK: HiFi-GAN vocoder

    // The first REAL weights on the TTS chain: the released UNIVERSAL_V1 generator, whose geometry is
    // this port's default configuration. The reference fuses its weight normalization before running
    // (`remove_weight_norm`), which is the arithmetic the converter bakes in; the input is a
    // deterministic synthetic mel, because a vocoder is a pure function of its mel and nothing about
    // speech needs assuming.
    func testHiFiGANUniversalMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_HIFIGAN"] else {
            throw XCTSkip("set IK_PARITY_HIFIGAN (run_reference.py hifigan)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let mel = try XCTUnwrap(arrays["mel"])                     // [bins, frames]
        let reference = try XCTUnwrap(arrays["output"]).asArray(Float.self).map(Double.init)

        let net = NFKMLXHiFiGANNet(NFKMLXHiFiGANConfiguration())
        try NFKMLXHiFiGAN.loadWeights(into: net, from: weights("IK_VAL_HIFIGAN"))

        let wave = net.waveform(mel.transposed(1, 0).expandedDimensions(axis: 0))
        eval(wave)
        let ours = wave.reshaped([-1]).asArray(Float.self).map(Double.init)
        XCTAssertEqual(ours.count, reference.count, "same sample count: frames × hop")
        let similarity = cosine(ours, reference)
        print("VALIDATION PARITY hifigan-universal: cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the vocoder matches the reference implementation")
    }

    // MARK: MiniMax Music 3 vocoder

    // Stage 1 of the music port: the Flow-VAE decoder against diffusers' own MiniMaxMusic3Vocoder on
    // the released checkpoint. A vocoder is a pure function of its latent, so it is the one stage
    // that reaches measured parity before any other stage exists; the input is the record's own
    // deterministic standard-normal latent. The reference computes its weight norm as a
    // parametrization at run time, which is the fusion the Swift loader bakes in at load.
    func testMusic3VocoderMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_MUSIC3_VOCODER"] else {
            throw XCTSkip("set IK_PARITY_MUSIC3_VOCODER (run_reference.py music_vocoder)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let latents = try XCTUnwrap(arrays["latents"])              // [128, T], the reference's NCL
        let reference = try XCTUnwrap(arrays["output"])             // [2, samples]

        let net = NFKMLXMusic3.makeVocoder()
        try NFKMLXMusic3.loadVocoderWeights(into: net, from: weights("IK_VAL_MUSIC3_VOCODER"))

        let wave = net.waveform(latents.transposed(1, 0).expandedDimensions(axis: 0))
        eval(wave)
        XCTAssertEqual(wave.shape, [1, latents.shape[1] * 512, 2], "hop 512, stereo")

        let ours = wave[0].transposed(1, 0).reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
        XCTAssertEqual(ours.count, theirs.count)
        let similarity = cosine(ours, theirs)
        let worst = zip(ours, theirs).map { abs($0 - $1) }.max() ?? 1
        print("VALIDATION PARITY music3-vocoder: cosine \(similarity), worst |difference| \(worst)")
        XCTAssertGreaterThan(similarity, 0.9999, "the vocoder matches the reference implementation")
    }

    // Stage 2a: the RVQ depth decoder against diffusers' own MiniMaxMusic3RVQDepthDecoder. The
    // record covers every parameter family — the causal transformer forward, all seven codebook
    // heads, the shared projection, and the offset-packed residual embedding table — because the
    // pipeline reads them through different paths and a forward alone touches only the first.
    func testMusic3DepthDecoderMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_MUSIC3_DEPTH"] else {
            throw XCTSkip("set IK_PARITY_MUSIC3_DEPTH (run_reference.py music_depth)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let net = NFKMLXMusic3.makeDepthDecoder()
        try NFKMLXMusic3.loadDepthWeights(into: net, from: weights("IK_VAL_MUSIC3_DEPTH"))

        func check(_ name: String, _ mine: MLXArray, _ referenceKey: String, floor: Double = 0.9999) throws {
            let reference = try XCTUnwrap(arrays[referenceKey])
            XCTAssertEqual(mine.shape, reference.shape, "\(name) shape")
            eval(mine)
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY music3-depth \(name): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, floor, "\(name) matches the reference")
        }

        let hidden = net.hiddenStates(try XCTUnwrap(arrays["inputs_embeds"]))
        try check("hidden", hidden, "output")
        let lastStep = hidden[0..., hidden.shape[1] - 1]
        let heads = stacked(net.audioHeads.map { $0(lastStep) })
        try check("heads", heads, "head_logits")
        try check("projection", net.projection(try XCTUnwrap(arrays["projection_input"])), "projected")
        let ids = try XCTUnwrap(arrays["embedding_ids"])
        try check("embedding", net.audioEmbeddings(ids), "embedded")
    }

    // Stage 2b: the condition encoder against diffusers' own MiniMaxMusic3ConditionEncoder — the
    // learned softmax blend of the 8 per-codebook hidden states, the scalar gain, the 3-wide
    // projection, and the exact nearest-neighbor resample onto the latent timeline (13 frames → 44).
    func testMusic3ConditionEncoderMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_MUSIC3_CONDITION"] else {
            throw XCTSkip("set IK_PARITY_MUSIC3_CONDITION (run_reference.py music_condition)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let hidden = try XCTUnwrap(arrays["hidden_states"])         // [frames, 8·4096]
        let reference = try XCTUnwrap(arrays["output"])             // [latents, 2048]

        let net = NFKMLXMusic3.makeConditionEncoder()
        try NFKMLXMusic3.loadConditionWeights(into: net, from: weights("IK_VAL_MUSIC3_CONDITION"))

        let condition = net.condition(hidden.expandedDimensions(axis: 0))
        eval(condition)
        XCTAssertEqual(condition.shape, [1, reference.shape[0], reference.shape[1]],
                       "the resample lands on the reference's latent count")
        let similarity = cosine(condition.reshaped([-1]).asArray(Float.self).map(Double.init),
                                reference.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY music3-condition: cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the condition encoder matches the reference")
    }

    // Stage 3: the flow-matching DiT against diffusers' own MiniMaxMusic3Transformer1DModel on the
    // released transformer (float32, sharded). Velocities at three timesteps pin the Fourier
    // timestep token, and the zero-condition branch pins the CFG path's unconditional forward.
    func testMusic3DiTMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_MUSIC3_DIT"] else {
            throw XCTSkip("set IK_PARITY_MUSIC3_DIT (run_reference.py music_dit)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let latents = try XCTUnwrap(arrays["latents"]).transposed(1, 0).expandedDimensions(axis: 0)
        let condition = try XCTUnwrap(arrays["condition"]).expandedDimensions(axis: 0)

        let net = NFKMLXMusic3.makeDiT()
        try NFKMLXMusic3.loadDiTWeights(into: net, from: weights("IK_VAL_MUSIC3_DIT"))

        let cases: [(String, Float, MLXArray)] = [
            ("t0", 0, condition), ("tmid", 0.5, condition), ("tlate", 1 - 1.0 / 30, condition),
            ("unconditional", 0.5, MLXArray.zeros(condition.shape)),
        ]
        for (name, timestep, conditioning) in cases {
            let reference = try XCTUnwrap(arrays["velocity_\(name)"])       // [128, L]
            let velocity = net.velocity(latents: latents, timestep: MLXArray([timestep]),
                                        condition: conditioning)
            eval(velocity)
            let similarity = cosine(
                velocity[0].transposed(1, 0).reshaped([-1]).asArray(Float.self).map(Double.init),
                reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY music3-dit \(name): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "the \(name) velocity matches the reference")
        }
    }

    // Stage 4: the autoregressive stage — the released Qwen3-8B and the depth decoder together,
    // teacher-forced with the reference's own sampled codes so the comparison measures the networks
    // rather than two random streams. Both sides run bf16: the model does not fit this machine at
    // float32. Covered seams: the prompt prefill's last hidden, the first step's raw logits (the
    // head) and guided band (the CFG + top-k-gate arithmetic), and the fused per-frame hidden
    // states that condition synthesis — the cache and the depth interleave live in that last one.
    func testMusic3AutoregressiveStageMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_MUSIC3_AR"] else {
            throw XCTSkip("set IK_PARITY_MUSIC3_AR (run_reference.py music_ar)")
        }
        let languageDirectory = try weights("IK_VAL_MUSIC3_LM")
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let textIDs = try XCTUnwrap(arrays["text_ids"])
        let forcedFrames = try XCTUnwrap(arrays["codes"]).asArray(Int32.self)
        let codebooks = 8
        let frames = (0 ..< forcedFrames.count / codebooks).map { frame in
            (0 ..< codebooks).map { Int(forcedFrames[frame * codebooks + $0]) }
        }

        let configuration = try NFKMLXLanguage.configuration(
            fromHuggingFace: languageDirectory.appendingPathComponent("config.json"))
        let languageModel = NFKMLXLanguage.makeNet(configuration)
        try NFKMLXLanguage.loadWeights(into: languageModel, fromDirectory: languageDirectory,
                                       precision: .checkpoint)
        let depthDecoder = NFKMLXMusic3.makeDepthDecoder()
        try NFKMLXMusic3.loadDepthWeights(into: depthDecoder, from: weights("IK_VAL_MUSIC3_DEPTH"),
                                          precision: .checkpoint)
        let stage = NFKMusic3AutoregressiveStage(languageModel: languageModel,
                                                 depthDecoder: depthDecoder)

        func similarity(_ mine: MLXArray, _ theirs: MLXArray) -> Double {
            cosine(mine.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init),
                   theirs.reshaped([-1]).asArray(Float.self).map(Double.init))
        }

        // The prefill and first-step seams, computed the way `generate` computes them.
        let cache = NFKMLXKeyValueCache(layerCount: configuration.layerCount)
        var lastHidden = languageModel.hiddenStates(
            fromEmbeddings: languageModel.embed(textIDs), cache: cache)
        lastHidden = lastHidden[0..., lastHidden.shape[1] - 1]
        eval(lastHidden)
        let prefillSimilarity = similarity(lastHidden, try XCTUnwrap(arrays["prefill_hidden"]))
        print("VALIDATION PARITY music3-ar prefill: cosine \(prefillSimilarity)")
        XCTAssertGreaterThan(prefillSimilarity, 0.999, "the prompt prefill matches")

        let rawLogits = languageModel.logits(fromHidden: lastHidden).asType(.float32)
        eval(rawLogits)
        let referenceLogits = try XCTUnwrap(arrays["first_logits"])
        let logitsSimilarity = similarity(rawLogits, referenceLogits)
        print("VALIDATION PARITY music3-ar first logits: cosine \(logitsSimilarity)")
        XCTAssertGreaterThan(logitsSimilarity, 0.999)

        let offset = NFKMusic3Contract.audioCodeOffset
        let band = offset ..< offset + NFKMusic3Contract.semanticVocabulary
        let guided = stage.guidedSemanticLogits(lastHidden: lastHidden)
        eval(guided)
        let guidedBand = guided[band].asArray(Float.self)
        let referenceBand = try XCTUnwrap(arrays["first_guided"])[band].asArray(Float.self)
        let mineArgmax = guidedBand.indices.max { guidedBand[$0] < guidedBand[$1] }
        let theirsArgmax = referenceBand.indices.max { referenceBand[$0] < referenceBand[$1] }
        XCTAssertEqual(mineArgmax, theirsArgmax, "the guided distribution peaks at the same code")
        // At bf16 the top-50 gate's boundary admits a slightly different tail on each side, so the
        // sets are compared as sets and the values over their intersection — a masked -1e9 against
        // a kept value would otherwise swamp the cosine with a disagreement about one candidate.
        let mineKept = Set(guidedBand.indices.filter { guidedBand[$0] > -1e8 })
        let theirsKept = Set(referenceBand.indices.filter { referenceBand[$0] > -1e8 })
        let shared = mineKept.intersection(theirsKept)
        print("VALIDATION PARITY music3-ar top-k sets: \(shared.count) shared of "
              + "\(mineKept.count)/\(theirsKept.count)")
        XCTAssertGreaterThanOrEqual(shared.count, 40, "the top-50 candidate sets agree")
        let guidedSimilarity = cosine(shared.map { Double(guidedBand[$0]) },
                                      shared.map { Double(referenceBand[$0]) })
        print("VALIDATION PARITY music3-ar guided top-k band: cosine \(guidedSimilarity)")
        XCTAssertGreaterThan(guidedSimilarity, 0.999, "the CFG and top-k gate arithmetic matches")

        // The teacher-forced run: same codes, so the fused hidden states measure prefill, cache,
        // feedback embedding, and the depth interleave end to end.
        let generation = try stage.generate(textIDs: textIDs, maxFrames: 3, forcedFrames: frames)
        let referenceHiddens = try XCTUnwrap(arrays["frame_hiddens"])
        XCTAssertEqual(generation.frameHiddens.shape,
                       [1, referenceHiddens.shape[0], referenceHiddens.shape[1]])
        let hiddensSimilarity = similarity(generation.frameHiddens, referenceHiddens)
        print("VALIDATION PARITY music3-ar frame hiddens: cosine \(hiddensSimilarity)")
        XCTAssertGreaterThan(hiddensSimilarity, 0.99, "the fused per-frame hidden states match")
    }

    // MARK: FastSpeech2 acoustic model

    // The trained acoustic half of the TTS chain: espnet's LJSpeech-trained conformer FastSpeech2,
    // through transformers' implementation as the oracle. Seams first — the encoder output says
    // whether the conformer stack is right, and the DURATIONS gate everything downstream: one frame
    // off and the mel comparison is meaningless, so they are asserted exactly, not by cosine.
    func testFastSpeech2ConformerMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_FASTSPEECH2"] else {
            throw XCTSkip("set IK_PARITY_FASTSPEECH2 (run_reference.py fastspeech2)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceMel = try XCTUnwrap(arrays["output"])
        let referenceEncoded = try XCTUnwrap(arrays["encoder_hidden"])
        let referenceDurations = try XCTUnwrap(arrays["durations"]).asArray(Int32.self).map(Int.init)

        let net = NFKMLXFastSpeech2.makeNet()
        try NFKMLXFastSpeech2.loadWeights(into: net, from: weights("IK_VAL_FASTSPEECH2"))

        let (mel, encoded, durations, pitch, energy) = net.generate(
            MLXArray(tokens).reshaped([1, tokens.count]))
        eval(mel, encoded, pitch, energy)

        let encoderSimilarity = cosine(
            encoded.reshaped([-1]).asArray(Float.self).map(Double.init),
            referenceEncoded.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY fastspeech2 encoder: cosine \(encoderSimilarity)")
        XCTAssertGreaterThan(encoderSimilarity, 0.9999, "the conformer encoder matches")

        print("VALIDATION PARITY fastspeech2 durations: \(durations) vs \(referenceDurations)")
        XCTAssertEqual(durations, referenceDurations, "the predicted durations match exactly")

        for (name, key, mine) in [("pitch", "pitch", pitch), ("energy", "energy", energy)] {
            let reference = try XCTUnwrap(arrays[key])
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY fastspeech2 \(name): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "\(name) prediction matches")
        }

        XCTAssertEqual(mel.shape, [1, referenceMel.shape[0], referenceMel.shape[1]])
        let melSimilarity = cosine(mel.reshaped([-1]).asArray(Float.self).map(Double.init),
                                   referenceMel.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY fastspeech2 mel: cosine \(melSimilarity)")
        XCTAssertGreaterThan(melSimilarity, 0.9999, "the generated mel matches the reference")
    }

    // The COMPLETE trained voice, end to end: real acoustic weights, the release's own phoneme
    // vocabulary, and the real vocoder — text has already become ARPAbet symbols, which is the
    // caller's phonemizer's job. The output is asserted as audio: the right length for the predicted
    // frames, finite, and carrying actual energy.
    func testTheTrainedVoiceSpeaksEndToEnd() throws {
        try requireMLXRuntime()
        // The PAIRED vocoder, not the universal one: espnet's acoustic model emits mels normalized
        // by its training statistics, and the release's own vocoder was trained on exactly those —
        // the universal jik876 generator expects raw log-mels and turns the normalized ones into
        // loud garbage, which Whisper transcribed as "(indistinct)". Measured, not assumed.
        guard let acoustic = config["IK_VAL_FASTSPEECH2"],
              let vocoder = config["IK_VAL_HIFIGAN_LJSPEECH"],
              let vocabulary = config["IK_VOCAB_FASTSPEECH2"] else {
            throw XCTSkip("set IK_VAL_FASTSPEECH2, IK_VAL_HIFIGAN_LJSPEECH, and IK_VOCAB_FASTSPEECH2")
        }
        let voice = try NFKMLXVoice.voice(acousticURL: URL(fileURLWithPath: acoustic),
                                          vocoderURL: URL(fileURLWithPath: vocoder),
                                          vocabularyURL: URL(fileURLWithPath: vocabulary))
        // "Hello world", in the release's ARPAbet-with-stress symbols.
        let samples = voice.speak(phonemes: ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"])

        XCTAssertGreaterThan(samples.count, 2048, "several frames of audio at hop 256")
        XCTAssertEqual(samples.count % 256, 0, "the vocoder emits hop samples per mel frame")
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
        let energy = samples.reduce(Float(0)) { $0 + $1 * $1 } / Float(samples.count)
        print("VALIDATION voice: \(samples.count) samples, mean energy \(energy)")
        XCTAssertGreaterThan(energy, 1e-4, "the voice produced sound, not silence")

        // The definitive check: energy cannot tell speech from loud noise, but the package's own
        // Whisper — real weights, at parity — can. TTS says the words; ASR must hear them.
        guard let whisperWeights = config["IK_VAL_WHISPER"] else { return }
        let configuration = NFKMLXWhisperConfiguration()
        let whisper = NFKMLXWhisper.makeNet(configuration)
        try NFKMLXWhisper.loadWeights(into: whisper, from: URL(fileURLWithPath: whisperWeights))

        var heard = NFKMLXAudioRate.matched(samples, from: NFKMLXVoice.sampleRate, to: 16_000)
        let targetSamples = 30 * 16_000
        if heard.count < targetSamples {
            heard += [Float](repeating: 0, count: targetSamples - heard.count)
        }
        let mel = NFKMLXMel.logMel(heard, sampleRate: 16_000, nMels: configuration.nMels)
        let tokens = whisper.transcribe(mel)
        guard let tokenizerDirectory = config["IK_TOKENIZER_WHISPER"],
              let tokenizer = try? NFKTokenizer(forManifest: ["tokenizer": ["type": "bpe-bytelevel"]],
                                                directory: URL(fileURLWithPath: tokenizerDirectory)) else {
            print("VALIDATION voice: transcribed ids \(tokens) (no tokenizer configured)")
            XCTAssertFalse(tokens.isEmpty, "the recognizer heard something in the synthesized clip")
            return
        }
        let transcript = tokenizer.decode(tokens.map { NSNumber(value: $0) }).lowercased()
        print("VALIDATION voice: whisper heard \"\(transcript)\"")
        XCTAssertTrue(transcript.contains("hello") || transcript.contains("world"),
                      "the recognizer heard the words the voice was asked to speak: \(transcript)")
    }

    // MARK: DeepSeek V4 arithmetic

    // The arithmetic was unmeasurable when this port was written — DeepSeek's own inference code
    // imports GPU-only tilelang kernels — and became measurable when transformers main shipped a
    // complete plain-PyTorch implementation. The oracle runs a tiny configuration whose layers are
    // all sliding attention over a sequence shorter than the window, which degenerates the
    // reference's attention to exactly the dense-with-sink path this port computes, and it saves the
    // weights in the RELEASE naming, so the module loads them strictly with no translation here.
    func testDeepSeekTinyMatchesTheReferenceLayerByLayer() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_TINY"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_TINY (run_reference.py deepseek_v4)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])

        // The oracle's own tiny geometry, spelled here rather than read from a file so a drift in
        // either place fails loudly.
        let geometry = NFKMLXDeepSeekConfiguration(
            hiddenSize: 64, layerCount: 4, vocabularySize: 128,
            headCount: 4, headDimensions: 16, ropeHeadDimensions: 4,
            queryLoRARank: 16, outputLoRARank: 16, outputGroups: 2,
            slidingWindow: 16, ropeTheta: 10_000,
            routedExpertCount: 8, activatedExpertCount: 2,
            expertIntermediateSize: 32, routeScale: 1.5, swigluLimit: 10,
            hashLayerCount: 1, compressRatios: [0, 0, 0, 0])   // every layer sliding, like the oracle
        let net = NFKMLXDeepSeek.makeNet(geometry)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("w::")
                ? (NFKMLXDeepSeek.moduleKey(forRelease: String(key.dropFirst(3))), value) : nil
        }
        try NFKMLXWeights.apply(weights, to: net)

        let ours = net.hiddenStates(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(ours)

        var report = [String]()
        var firstBad: Int?
        for index in 0 ..< ours.count {
            guard let reference = arrays["hidden.\(index)"] else { break }
            // The reference's LAST entry follows its own convention: the stream after the head
            // collapse and the final norm, not the raw copies.
            let state = index == ours.count - 1 ? net.finalState(ours[index]) : ours[index]
            let mine = state.reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            guard mine.count == theirs.count else {
                report.append("\(index): shape \(state.shape) vs \(reference.shape)")
                firstBad = firstBad ?? index
                break
            }
            let similarity = cosine(mine, theirs)
            let label = index == 0 ? "embedding" : "after layer \(index - 1)"
            report.append(String(format: "  %-18s cosine %.10f", (label as NSString).utf8String!, similarity))
            if similarity < 0.9999 && firstBad == nil { firstBad = index }
        }
        print("VALIDATION isolation deepseek-v4-tiny:\n" + report.joined(separator: "\n"))
        XCTAssertNil(firstBad, "first divergence at \(firstBad.map { $0 == 0 ? "the embedding" : "layer \($0 - 1)" } ?? "-")")

        let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(logits)
        let mine = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY deepseek-v4-tiny: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the decoder's arithmetic matches the reference")
    }

    // MARK: SAM 2 image encoder

    // SAM 2's Hiera trunk and FPN neck against facebookresearch's own sources. The FPN levels are
    // scored alongside the vision features: agreeing at the finest level but not the coarsest would
    // point at the neck's top-down fusion rather than the trunk.
    // The largest released Hiera: 48 blocks against tiny's 12, weighted heavily toward the third
    // stage, with a coarser window there and global attention much later. Its config overrides every
    // axis the constructor defaults, so this is the size that says whether the geometry is genuinely
    // configuration-driven or only appears to be.
    func testSAM2LargeEncoderMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SAM2_LARGE"] else { throw XCTSkip("set IK_PARITY_SAM2_LARGE") }
        let (inputTensor, _) = try record("IK_PARITY_SAM2_LARGE")
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))

        let net = NFKMLXSAM2.makeEncoder(.large)
        try NFKMLXSAM2.loadWeights(into: net, from: weights("IK_VAL_SAM2_LARGE"))
        net.train(false)

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let levels = net.features(inputTensor.reshaped([1, height, width, 3]))
        eval(levels)

        for (index, name) in ["level0", "level1"].enumerated() {
            guard let reference = arrays[name] else { continue }
            let ours = levels[index].reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            XCTAssertEqual(ours.count, theirs.count, "large \(name): same feature volume")
            let similarity = cosine(ours, theirs)
            print("VALIDATION PARITY sam2-large \(name): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "large \(name) matches the reference")
        }
    }

    // The middle released Hiera. Its config sets only the width and head count and takes the rest
    // from the constructor's defaults — the opposite of large, which overrides every axis — so this
    // is the size that says whether the DEFAULTS are right, not only the overrides. Until now its
    // numerics rested on the tiny and large measurements; this is its own.
    func testSAM2BasePlusEncoderMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SAM2_BASE_PLUS"] else {
            throw XCTSkip("set IK_PARITY_SAM2_BASE_PLUS")
        }
        let (inputTensor, _) = try record("IK_PARITY_SAM2_BASE_PLUS")
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))

        let net = NFKMLXSAM2.makeEncoder(.basePlus)
        try NFKMLXSAM2.loadWeights(into: net, from: weights("IK_VAL_SAM2_BASE_PLUS"))
        net.train(false)

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let levels = net.features(inputTensor.reshaped([1, height, width, 3]))
        eval(levels)

        for (index, name) in ["level0", "level1"].enumerated() {
            guard let reference = arrays[name] else { continue }
            let ours = levels[index].reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            XCTAssertEqual(ours.count, theirs.count, "base_plus \(name): same feature volume")
            let similarity = cosine(ours, theirs)
            print("VALIDATION PARITY sam2-base-plus \(name): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "base_plus \(name) matches the reference")
        }
    }

    func testSAM2EncoderMatchesTheReference() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_SAM2_ENCODER")
        guard let path = config["IK_PARITY_SAM2_ENCODER"] else { throw XCTSkip("set IK_PARITY_SAM2_ENCODER") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))

        let net = NFKMLXSAM2.makeEncoder()
        try NFKMLXSAM2.loadWeights(into: net, from: weights("IK_VAL_SAM2"))
        net.train(false)

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let levels = net.features(inputTensor.reshaped([1, height, width, 3]))
        eval(levels)

        for (index, name) in ["level0", "level1"].enumerated() {
            guard let reference = arrays[name] else { continue }
            let ours = levels[index].reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            XCTAssertEqual(ours.count, theirs.count, "\(name) has the reference's shape")
            let similarity = cosine(ours, theirs)
            print("VALIDATION PARITY sam2-encoder: \(name) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.99, "\(name) matches the reference implementation")
        }

        let ours = levels.last!.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        XCTAssertEqual(ours.count, theirs.count, "same vision-feature shape as the reference")
        let similarity = cosine(ours, theirs)
        print("VALIDATION PARITY sam2-encoder: vision features cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the vision features match the reference implementation")
    }

    // SAM 2's prompt encoder and mask decoder, from one positive click. The record carries the
    // reference's own sparse tokens, so the prompt encoder is scored before the decoder consumes it —
    // a shifted prompt and a wrong decoder look identical at the mask.
    func testSAM2DecoderMatchesTheReference() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceMasks) = try record("IK_PARITY_SAM2_DECODER")
        guard let path = config["IK_PARITY_SAM2_DECODER"] else { throw XCTSkip("set IK_PARITY_SAM2_DECODER") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))

        let encoder = NFKMLXSAM2.makeEncoder()
        try NFKMLXSAM2.loadWeights(into: encoder, from: weights("IK_VAL_SAM2"))
        encoder.train(false)
        let decoder = NFKMLXSAM2.makeDecoder()
        let prompt = NFKMLXSAM2.makePromptEncoder()
        try NFKMLXSAM2.loadDecoderWeights(into: decoder, prompt: prompt, from: weights("IK_VAL_SAM2"))
        decoder.train(false)

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let levels = encoder.features(inputTensor.reshaped([1, height, width, 3]))
        let features = levels.last!
        let grid = features.shape[1]

        // The reference clicks at (w/2, h/3) in pixels, shifts by half a pixel to the pixel's centre,
        // and only then normalizes — so the port is handed exactly that, not the idealized fractions.
        let sparse = prompt.sparse(pointX: (Float(width) / 2 + 0.5) / Float(width),
                                   pointY: (Float(height) / 3 + 0.5) / Float(height), positive: true)
        if let referenceSparse = arrays["sparse"] {
            eval(sparse)
            let similarity = cosine(sparse.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    referenceSparse.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY sam2-decoder: sparse prompt cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.99, "the prompt encoder matches the reference")
        }

        let positional = prompt.positionEncoding.grid(grid, grid)
        let dense = prompt.dense(grid: grid)
        let (masks, iou, objectScore) = decoder(features: features, positional: positional,
                                                sparse: sparse, dense: dense,
                                                highResolution: [levels[0], levels[1]])
        eval(masks, iou, objectScore)

        // The reference returns the three multimask tokens; the port emits all four, so the
        // comparison drops the single-mask token as `multimask_output=True` does.
        let ours = masks[0, 1...].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceMasks.reshaped([-1]).asArray(Float.self).map(Double.init)
        XCTAssertEqual(ours.count, theirs.count, "same mask logits shape as the reference")
        let similarity = cosine(ours, theirs)
        print("VALIDATION PARITY sam2-decoder: mask logits cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the mask logits match the reference implementation")

        if let referenceScore = arrays["object_score"] {
            let mine = objectScore.reshaped([-1]).asArray(Float.self)[0]
            let theirs = referenceScore.reshaped([-1]).asArray(Float.self)[0]
            print("VALIDATION PARITY sam2-decoder: object score \(mine) vs \(theirs)")
            XCTAssertEqual(Double(mine), Double(theirs), accuracy: 0.01, "the object score matches")
        }
    }

    // SAM 2's video memory path: the encoder that folds a frame and its mask into a memory, and the
    // rotary attention that conditions the next frame on it. Both are driven with the reference's own
    // tensors so the two modules are measured apart from the tracking loop that would sequence them.
    func testSAM2MemoryPathMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SAM2_MEMORY"] else { throw XCTSkip("set IK_PARITY_SAM2_MEMORY") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let features = try XCTUnwrap(arrays["features"])         // [64, 64, 256]
        let mask = try XCTUnwrap(arrays["mask"])                 // [1024, 1024, 1]
        let referenceMemory = try XCTUnwrap(arrays["memory"])    // [64, 64, 64]
        let referenceOutput = try XCTUnwrap(arrays["output"])    // [4096, 256]

        let attention = NFKMLXSAM2.makeMemoryAttention()
        let encoder = NFKMLXSAM2.makeMemoryEncoder()
        try NFKMLXSAM2.loadMemoryWeights(into: attention, encoder: encoder, from: weights("IK_VAL_SAM2MEMORY"))
        attention.train(false)
        encoder.train(false)

        let memory = encoder(features: features.reshaped([1] + features.shape),
                             maskLogits: mask.reshaped([1] + mask.shape))
        eval(memory)
        let mine = memory.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceMemory.reshaped([-1]).asArray(Float.self).map(Double.init)
        XCTAssertEqual(mine.count, theirs.count, "same memory shape as the reference")
        let memorySimilarity = cosine(mine, theirs)
        print("VALIDATION PARITY sam2-memory: encoded memory cosine \(memorySimilarity)")
        XCTAssertGreaterThan(memorySimilarity, 0.99, "the memory encoder matches the reference")

        let current = try XCTUnwrap(arrays["current"])           // [4096, 256]
        let currentPosition = try XCTUnwrap(arrays["current_pos"])
        let memoryPosition = try XCTUnwrap(arrays["memory_pos"])
        let memoryTokens = memory.reshaped([1, memory.shape[1] * memory.shape[2], memory.shape[3]])
        let out = attention(current: current.reshaped([1] + current.shape),
                            memory: memoryTokens,
                            currentPosition: currentPosition.reshaped([1] + currentPosition.shape),
                            memoryPosition: memoryPosition.reshaped([1] + memoryPosition.shape))
        eval(out)
        let ours = out.reshaped([-1]).asArray(Float.self).map(Double.init)
        let reference = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        XCTAssertEqual(ours.count, reference.count, "same attention output shape as the reference")
        let similarity = cosine(ours, reference)
        print("VALIDATION PARITY sam2-memory: attention output cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the rotary memory attention matches the reference")
    }

    // Hybrid Transformer Demucs. The record carries each stage boundary as well as the audio, so a
    // failure names the branch that diverged. The reference runs with `use_train_segment` off, which is
    // a padding policy rather than a shape, so both sides separate the clip at its own length.
    func testHTDemucsMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_HTDEMUCS"] else { throw XCTSkip("set IK_PARITY_HTDEMUCS") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let mix = try XCTUnwrap(arrays["waveform"])              // [channels, samples]
        let referenceOutput = try XCTUnwrap(arrays["output"])    // [sources, channels, samples]

        let net = NFKMLXHTDemucs.makeNet()
        try NFKMLXHTDemucs.loadWeights(into: net, from: weights("IK_VAL_HTDEMUCS"))
        let trace = net.trace(mix, padsToTrainingSegment: false)
        eval(trace.waveform)

        for (key, mine) in [("spectrogram", trace.spectrogram), ("bottleneck_in", trace.bottleneckIn),
                            ("bottleneck_out", trace.bottleneckOut), ("freq_out", trace.frequencyOut),
                            ("time_out", trace.timeOut)] {
            let name = key.replacingOccurrences(of: "_", with: "-")
            let theirs = try XCTUnwrap(arrays[key], "the record carries the \(name) seam")
            XCTAssertEqual(mine.shape, theirs.shape, "\(name) has the reference's shape")
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    theirs.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY htdemucs: \(name) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.99, "the \(name) seam matches the reference")
        }

        let ours = trace.waveform.reshaped([-1]).asArray(Float.self).map(Double.init)
        let reference = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        XCTAssertEqual(ours.count, reference.count, "same separation shape as the reference")
        let similarity = cosine(ours, reference)
        let difference = zip(ours, reference).map { abs($0 - $1) }.reduce(0, +) / Double(ours.count)
        print("VALIDATION PARITY htdemucs: separated stems cosine \(similarity), mean |difference| \(difference)")
        XCTAssertGreaterThan(similarity, 0.99, "the separated stems match the reference")
    }

    // The Stable Diffusion UNet. The text conditioning arrives as a tensor from the record rather than
    // from an encoder, because the port takes a context and the caller brings the tower — the same
    // separation the CLIP text record uses.
    func testSDUNetMatchesTheReference() throws {
        try verifyUNet("sd-unet", .inpainting, record: "IK_PARITY_SD_UNET", weights: "IK_VAL_SDUNET")
    }

    // Marigold is Stable Diffusion 2 geometry: the same blocks at a wider cross-attention, with the
    // transformer's ends projected by a linear layer rather than a 1×1 convolution.
    func testMarigoldUNetMatchesTheReference() throws {
        try verifyUNet("marigold-unet", .marigold, record: "IK_PARITY_MARIGOLD_UNET",
                       weights: "IK_VAL_MARIGOLDUNET")
    }

    // The ×4 upscaler leaves its FIRST level plain where the others leave their last, and folds a noise
    // level into the timestep through a class embedding.
    func testUpscalerUNetMatchesTheReference() throws {
        try verifyUNet("upscaler-unet", .upscaler, record: "IK_PARITY_UPSCALER_UNET",
                       weights: "IK_VAL_UPSCALERUNET")
    }

    // The Stable Diffusion 2.1 UNet: the same geometry Marigold proved, on a text-to-image release.
    func testSD21UNetMatchesTheReference() throws {
        var configuration = NFKMLXSDTextToImageConfiguration.stableDiffusion21.unet
        configuration.inputChannels = 4
        try verifyUNet("sd21-unet", configuration, record: "IK_PARITY_SD21_UNET", weights: "IK_VAL_SD21UNET")
    }

    // SDXL: three levels rather than four, ten transformer blocks deep at the coarsest, and a pooled
    // embedding plus a size descriptor folded into the timestep.
    func testSDXLUNetMatchesTheReference() throws {
        try verifyUNet("sdxl-unet", .sdxl, record: "IK_PARITY_SDXL_UNET", weights: "IK_VAL_SDXLUNET")
    }

    private func verifyUNet(_ name: String, _ configuration: NFKMLXSDUNetConfiguration,
                            record: String, weights key: String) throws {
        try requireMLXRuntime()
        guard let path = config[record] else { throw XCTSkip("set \(record)") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let latent = try XCTUnwrap(arrays["latent"])             // [height, width, inputChannels]
        let context = try XCTUnwrap(arrays["context"])           // [tokens, crossAttentionDimensions]
        let timestep = try XCTUnwrap(arrays["timestep"])
        let reference = try XCTUnwrap(arrays["output"])          // [height, width, 4]

        let net = NFKMLXSDUNet(configuration: configuration)
        try NFKMLXStableDiffusionModels.loadUNetWeights(into: net, from: weights(key))
        net.train(false)
        let added = arrays["pooled"].flatMap { pooled in
            arrays["time_ids"].map {
                NFKSDAddedConditioning(pooled: pooled.reshaped([1] + pooled.shape),
                                       timeIds: $0.reshaped([1] + $0.shape))
            }
        }
        let out = net(latent.reshaped([1] + latent.shape), timestep: timestep,
                      context: context.reshaped([1] + context.shape),
                      classLabel: arrays["class_label"].map { $0.asType(.int32) },
                      added: added)
        eval(out)
        XCTAssertEqual(Array(out.shape.dropFirst()), reference.shape, "the reference's output shape")
        let similarity = cosine(out.reshaped([-1]).asArray(Float.self).map(Double.init),
                                reference.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY \(name): predicted noise cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the \(name) UNet matches the reference")
    }

    // The Stable Diffusion autoencoder, both directions. The latent seam separates a bad encoder from a
    // bad decoder, which a decoded image alone cannot.
    func testSDAutoencoderMatchesTheReference() throws {
        try verifyAutoencoder("sd-vae", .stableDiffusion, record: "IK_PARITY_SD_VAE",
                              weights: "IK_VAL_SDVAE")
    }

    // The upscaler's autoencoder is one resolution level shallower than the others.
    func testUpscalerAutoencoderMatchesTheReference() throws {
        try verifyAutoencoder("upscaler-vae", .upscaler, record: "IK_PARITY_UPSCALER_VAE",
                              weights: "IK_VAL_UPSCALERVAE")
    }

    private func verifyAutoencoder(_ name: String, _ configuration: NFKMLXSDVAEConfiguration,
                                   record: String, weights key: String) throws {
        try requireMLXRuntime()
        guard let path = config[record] else { throw XCTSkip("set \(record)") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let plate = try XCTUnwrap(arrays["input_image"])         // [height, width, 3] in 0...1
        let referenceLatent = try XCTUnwrap(arrays["latent"])
        let reference = try XCTUnwrap(arrays["output"])

        let net = NFKMLXSDAutoencoder(configuration: configuration)
        try NFKMLXStableDiffusionModels.loadVAEWeights(into: net, from: weights(key))
        net.train(false)
        let image = plate.reshaped([1] + plate.shape) * 2 - 1    // the reference works in -1...1
        let (mean, _) = net.encode(image)
        eval(mean)
        let latentSimilarity = cosine(mean.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceLatent.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY \(name): latent cosine \(latentSimilarity)")
        XCTAssertGreaterThan(latentSimilarity, 0.99, "the encoder matches the reference")

        let decoded = net.decode(mean)
        eval(decoded)
        let similarity = cosine(decoded.reshaped([-1]).asArray(Float.self).map(Double.init),
                                reference.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY \(name): decoded image cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the decoder matches the reference")
    }

    // The DDIM sampler itself. The networks are at parity, but the loop that iterates them is its own
    // arithmetic: the schedule it visits, the signal ratio it reads there, and the update it applies.
    // Driving both sides over the same latents and model outputs isolates that from the UNet.
    func testDDIMSchedulerMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SD_SCHEDULER"] else { throw XCTSkip("set IK_PARITY_SD_SCHEDULER") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let referenceTimesteps = try XCTUnwrap(arrays["timesteps"]).asArray(Float.self).map { Int($0) }
        let latentsIn = try XCTUnwrap(arrays["latents_in"])      // [steps, h, w, 4]
        let predictions = try XCTUnwrap(arrays["predictions"])
        let latentsOut = try XCTUnwrap(arrays["output"])

        let scheduler = NFKDDIMScheduler(predictionType: .epsilon)
        let steps = scheduler.steps(referenceTimesteps.count)
        XCTAssertEqual(steps.map(\.train), referenceTimesteps,
                       "the schedule visits the reference's training steps")

        var worst = 1.0
        for (index, step) in steps.enumerated() {
            let next = scheduler.step(prediction: predictions[index], timestep: step,
                                      latent: latentsIn[index])
            eval(next)
            let similarity = cosine(next.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    latentsOut[index].reshaped([-1]).asArray(Float.self).map(Double.init))
            worst = min(worst, similarity)
        }
        print("VALIDATION PARITY sd-scheduler: worst per-step latent cosine \(worst)")
        XCTAssertGreaterThan(worst, 0.9999, "every DDIM step matches the reference")

        let clean = try XCTUnwrap(arrays["clean"])
        let noise = try XCTUnwrap(arrays["noise"])
        let noised = try XCTUnwrap(arrays["noised"])
        let mine = scheduler.addNoise(clean: clean, noise: noise, timestep: steps[0])
        eval(mine)
        let noiseSimilarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                     noised.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY sd-scheduler: add-noise cosine \(noiseSimilarity)")
        XCTAssertGreaterThan(noiseSimilarity, 0.9999, "noising a clean latent matches the reference")
    }

    // MARK: RIFE v4

    // The third IFNet generation, against the released 4.13.2 weights. v4 conditions on a timestep,
    // so unlike HDv3 it can interpolate anywhere between the frames; this scores the midpoint, which
    // is what the earlier generations produce.
    func testRIFEv4MatchesTheReferenceInterpolation() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_RIFEV4")
        guard let path = config["IK_PARITY_RIFEV4"] else { throw XCTSkip("set IK_PARITY_RIFEV4") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let second = try XCTUnwrap(arrays["frame1"])

        let net = NFKMLXRIFEv4.makeNet()
        try NFKMLXRIFEv4.loadWeights(into: net, from: weights("IK_VAL_RIFEV4"))

        let ours = net.interpolate(inputTensor, second)
        eval(ours)
        XCTAssertEqual(ours.shape, referenceOutput.shape, "same frame size as the reference")
        let mine = ours.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        let error = zip(mine, theirs).map { abs($0 - $1) }.reduce(0, +) / Double(mine.count)
        print("VALIDATION PARITY rife-v4: interpolated cosine \(similarity), mean |difference| \(error)")
        XCTAssertGreaterThan(similarity, 0.99, "the interpolated frame matches the reference implementation")
        XCTAssertLessThan(error, 0.02, "and agrees pixelwise, not just in shape")
    }

    // MARK: BiSeNetV2

    // Against CoinCheung's own BiSeNetV2 on the released Cityscapes checkpoint. That checkpoint
    // predates the repository's current head — it emits `classes × upFactor²` channels and
    // pixel-shuffles — so the oracle substitutes the head the weights were trained with.
    func testBiSeNetV2MatchesTheReferenceLogits() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_BISENETV2")

        let net = NFKMLXBiSeNetV2.makeNet()
        try NFKMLXBiSeNetV2.loadWeights(into: net, from: weights("IK_VAL_BISENETV2"))
        net.train(false)

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let ours = net.logits(NFKMLXBiSeNetNet.normalized(inputTensor.reshaped([1, height, width, 3])))
        eval(ours)
        XCTAssertEqual(Array(ours.shape.dropFirst()), referenceOutput.shape, "same logit grid as the reference")

        let similarity = cosine(ours.reshaped([-1]).asArray(Float.self).map(Double.init),
                                referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init))
        let classes = referenceOutput.shape[2]
        let ourLabels = ours.reshaped([-1, classes]).argMax(axis: -1).asArray(Int32.self)
        let referenceLabels = referenceOutput.reshaped([-1, classes]).argMax(axis: -1).asArray(Int32.self)
        let agreement = Double(zip(ourLabels, referenceLabels).filter(==).count) / Double(ourLabels.count)
        print("VALIDATION PARITY bisenetv2: logit cosine \(similarity), label agreement \(agreement)")
        XCTAssertGreaterThan(similarity, 0.99, "the class logits match the reference implementation")
        XCTAssertGreaterThan(agreement, 0.99, "and the predicted labels agree")
    }

    // MARK: siggraph17 colorizer

    // The second released colorizer, against richzhang's own network. The `ab` seam is scored first:
    // the final sRGB passes through a Lab round trip, so comparing only that conflates a network
    // difference with a colour-conversion one — the lesson eccv16 taught when nearest-neighbour
    // resampling scored 0.9993 end to end while its ab prediction was 0.96.
    func testSiggraph17MatchesTheReferenceColorization() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_SIGGRAPH17")
        guard let path = config["IK_PARITY_SIGGRAPH17"] else { throw XCTSkip("set IK_PARITY_SIGGRAPH17") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))

        let net = NFKMLXSiggraphColorizer.makeNet()
        try NFKMLXSiggraphColorizer.loadWeights(into: net, from: weights("IK_VAL_SIGGRAPH17"))
        net.train(false)

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        if let referenceAB = arrays["ab"], let lightness = arrays["lightness"] {
            let ab = net.predictAB(lightness: lightness.reshaped([1, height, width, 1]))
            eval(ab)
            let similarity = cosine(ab.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    referenceAB.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY siggraph17: ab cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.99, "the network's ab prediction matches the reference")
        }

        let colorized = net.colorize(inputTensor)
        eval(colorized)
        let ours = colorized.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, theirs)
        print("VALIDATION PARITY siggraph17: colorized cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "and the colorized image matches end to end")
    }

    // MARK: Whisper

    // Transcription against openai-whisper itself. Whisper was the project's first real-weight
    // success, but its evidence was a hand-run transcript comparison; this is the numeric record every
    // other shipped model has. The mel is scored first: a front-end mismatch and a network mismatch
    // are indistinguishable once both have been reduced to token ids.
    func testWhisperMatchesTheReferenceTranscription() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_WHISPER"] else { throw XCTSkip("set IK_PARITY_WHISPER") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"]).asArray(Float.self)
        let referenceTokens = try XCTUnwrap(arrays["output"]).asArray(Int32.self)

        // This record isolates the NETWORK, so it is decoded under the plain rule the oracle applies
        // beside it: the special and timestamp range masked, nothing else. The reference's own policy
        // is measured separately, below.
        var plain = NFKMLXWhisperConfiguration()
        plain.suppressesBlankStart = false
        let net = NFKMLXWhisper.makeNet(plain)
        try NFKMLXWhisper.loadWeights(into: net, from: weights("IK_VAL_WHISPER"))

        // The same 30-second pad the backend applies: Whisper is trained only on that length, and a
        // short clip fed raw is out of distribution.
        var prepared = waveform
        let targetSamples = 30 * 16000
        if prepared.count < targetSamples {
            prepared += [Float](repeating: 0, count: targetSamples - prepared.count)
        }
        let mel = NFKMLXMel.logMel(prepared, sampleRate: 16000, nMels: net.configuration.nMels)
        eval(mel)
        if let referenceMel = arrays["features"] {
            // The reference stores `[mels, frames]`; the port carries frames first.
            let ours = mel.reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = referenceMel.transposed(1, 0).reshaped([-1]).asArray(Float.self).map(Double.init)
            XCTAssertEqual(ours.count, theirs.count, "same mel grid as the reference")
            let melSimilarity = cosine(ours, theirs)
            print("VALIDATION PARITY whisper: mel cosine \(melSimilarity)")
            XCTAssertGreaterThan(melSimilarity, 0.999, "the log-mel front end matches the reference")
        }

        // The decoder seam, before any suppression policy: a token mismatch alone cannot distinguish a
        // wrong network from a different decoding rule, and this can.
        if let referenceLogits = arrays["first_logits"] {
            let audio = net.encoder(mel)
            let prompt = MLXArray(net.configuration.promptTokens.map { Int32($0) })
                .reshaped([1, net.configuration.promptTokens.count])
            let logits = net.decoder(prompt, audio: audio)[0, net.configuration.promptTokens.count - 1]
            eval(logits)
            let similarity = cosine(logits.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY whisper: first-step logit cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.999, "the decoder matches the reference implementation")
        }

        let tokens = net.transcribe(mel).map { Int32($0) }
        print("VALIDATION PARITY whisper: tokens \(tokens) vs reference \(Array(referenceTokens))")
        XCTAssertEqual(tokens, Array(referenceTokens),
                       "the greedy decode reproduces the reference's tokens under the same suppression")
    }

    // The decoding POLICY, which the network parity test above deliberately holds constant. The
    // reference's `decode` masks its curated non-speech set at every step and, at the first sampled
    // position only, a space and an immediate end of text (`SuppressBlank`). The record carries the
    // tokens those rules produce, so the two policies are compared rather than described — and they
    // differ on this clip, which is what makes the comparison worth making.
    func testWhisperReproducesTheReferenceDecodingRules() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_WHISPER"] else { throw XCTSkip("set IK_PARITY_WHISPER") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        guard let ruled = arrays["ruled_tokens"], let nonSpeech = arrays["non_speech_tokens"] else {
            throw XCTSkip("the record predates the decoding-rule capture; regenerate it")
        }
        let waveform = try XCTUnwrap(arrays["waveform"]).asArray(Float.self)
        let referenceTokens = ruled.asArray(Int32.self)

        // The reference computes these ids from its own tokenizer; the port's
        // `NFKMLXWhisperSuppression.nonSpeechTokens(using:)` does the same from an `NFKTokenizer`.
        // Taking them from the record here keeps this test about the RULES rather than the vocabulary.
        var configuration = NFKMLXWhisperConfiguration()
        configuration.suppressTokens = nonSpeech.asArray(Int32.self).map(Int.init)
        configuration.suppressesBlankStart = true

        let net = NFKMLXWhisper.makeNet(configuration)
        try NFKMLXWhisper.loadWeights(into: net, from: weights("IK_VAL_WHISPER"))

        var prepared = waveform
        let targetSamples = 30 * 16000
        if prepared.count < targetSamples {
            prepared += [Float](repeating: 0, count: targetSamples - prepared.count)
        }
        let mel = NFKMLXMel.logMel(prepared, sampleRate: 16000, nMels: net.configuration.nMels)
        let tokens = net.transcribe(mel).map { Int32($0) }

        print("VALIDATION PARITY whisper-rules: tokens \(tokens) vs reference \(Array(referenceTokens))")
        XCTAssertEqual(tokens, Array(referenceTokens),
                       "the port applies the reference's suppression rules identically")

        // The space id `SuppressBlank` masks is a constant of the byte-level vocabulary, not a guess.
        if let space = arrays["space_token"] {
            XCTAssertEqual(space.asArray(Int32.self).first.map(Int.init), NFKMLXWhisperNet.spaceToken)
        }
    }

    // Timestamped decoding, which is a different decode rather than a different reading of the same
    // one: the times only exist when `<|notimestamps|>` is left out of the prompt and the timestamp
    // range stays unmasked. `ApplyTimestampRules` then orders the result, and the record carries the
    // ids the reference's own rules produce.
    func testWhisperReproducesTheReferenceTimestampRules() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_WHISPER"] else { throw XCTSkip("set IK_PARITY_WHISPER") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        guard let timed = arrays["timed_tokens"], let begin = arrays["timestamp_begin"],
              let nonSpeech = arrays["non_speech_tokens"] else {
            throw XCTSkip("the record predates the timestamp capture; regenerate it")
        }
        let waveform = try XCTUnwrap(arrays["waveform"]).asArray(Float.self)
        let referenceTokens = timed.asArray(Int32.self)

        var configuration = NFKMLXWhisperConfiguration()
        configuration.suppressTokens = nonSpeech.asArray(Int32.self).map(Int.init)
        configuration.timestampBegin = Int(begin.asArray(Int32.self)[0])

        // The reference drops `<|notimestamps|>` from its own prompt, which is what the port does too.
        XCTAssertEqual(try XCTUnwrap(arrays["timestamp_prompt"]).asArray(Int32.self).map(Int.init),
                       configuration.promptTokens.filter { $0 != configuration.timestampBegin - 1 })

        let net = NFKMLXWhisper.makeNet(configuration)
        try NFKMLXWhisper.loadWeights(into: net, from: weights("IK_VAL_WHISPER"))

        var prepared = waveform
        let targetSamples = 30 * 16000
        if prepared.count < targetSamples {
            prepared += [Float](repeating: 0, count: targetSamples - prepared.count)
        }
        let mel = NFKMLXMel.logMel(prepared, sampleRate: 16000, nMels: net.configuration.nMels)
        let (segments, tokens) = net.transcribeWithTimestamps(mel)

        print("VALIDATION PARITY whisper-timestamps: tokens \(tokens) vs reference "
              + "\(Array(referenceTokens)), segments \(segments.map { ($0.start, $0.end) })")
        XCTAssertEqual(tokens.map { Int32($0) }, Array(referenceTokens),
                       "the port applies the reference's timestamp rules identically")

        // The ids pair into spans, and a span's seconds are its distance from `<|0.00|>`.
        XCTAssertFalse(segments.isEmpty, "a timestamped decode produces at least one span")
        for segment in segments {
            XCTAssertLessThanOrEqual(segment.start, segment.end)
            XCTAssertLessThanOrEqual(segment.end, 30, "a span cannot leave the window")
        }
        let opening = try XCTUnwrap(segments.first)
        XCTAssertEqual(opening.start, Float(Int(referenceTokens[0]) - configuration.timestampBegin)
                       * configuration.timestampPrecision, accuracy: 1e-6)
    }

    // The larger Whisper releases. Every size shares one encoder-decoder structure and differs in a
    // configuration — except large-v3, which also produces 128 mel bands instead of 80 and carries one
    // more language token, shifting `<|transcribe|>` and `<|notimestamps|>`. Both come from the
    // model's own tokenizer on each side rather than from the smallest size's constants.
    func testWhisperLargerSizesMatchTheReference() throws {
        try requireMLXRuntime()
        let sizes: [(String, String, String, NFKMLXWhisperConfiguration)] = [
            ("small", "IK_VAL_WHISPER_SMALL", "IK_PARITY_WHISPER_SMALL", .small),
            ("medium", "IK_VAL_WHISPER_MEDIUM", "IK_PARITY_WHISPER_MEDIUM", .medium),
            ("large-v3", "IK_VAL_WHISPER_LARGE_V3", "IK_PARITY_WHISPER_LARGE_V3", .largeV3),
        ]
        for (name, weightsKey, parityKey, geometry) in sizes {
            guard let path = config[parityKey] else { throw XCTSkip("set \(parityKey)") }
            let arrays = try loadArrays(url: URL(fileURLWithPath: path))
            let waveform = try XCTUnwrap(arrays["waveform"]).asArray(Float.self)
            let referenceTokens = try XCTUnwrap(arrays["output"]).asArray(Int32.self)

            // The record carries the prompt the reference actually used, so a shifted special-token id
            // shows up here rather than as a mysterious token mismatch.
            if let prompt = arrays["prompt"] {
                XCTAssertEqual(prompt.asArray(Int32.self).map(Int.init), geometry.promptTokens,
                               "\(name): the port's prompt is the reference's")
            }

            var plain = geometry
            plain.suppressesBlankStart = false          // this record isolates the network
            let net = NFKMLXWhisper.makeNet(plain)
            try NFKMLXWhisper.loadWeights(into: net, from: weights(weightsKey))

            var prepared = waveform
            let targetSamples = 30 * 16000
            if prepared.count < targetSamples {
                prepared += [Float](repeating: 0, count: targetSamples - prepared.count)
            }
            let mel = NFKMLXMel.logMel(prepared, sampleRate: 16000, nMels: plain.nMels)
            eval(mel)
            XCTAssertEqual(mel.shape[2], plain.nMels, "\(name): the front end produces its own band count")

            if let referenceMel = arrays["features"] {
                let ours = mel.reshaped([-1]).asArray(Float.self).map(Double.init)
                let theirs = referenceMel.transposed(1, 0).reshaped([-1]).asArray(Float.self).map(Double.init)
                XCTAssertEqual(ours.count, theirs.count, "\(name): same mel grid")
                let melSimilarity = cosine(ours, theirs)
                print("VALIDATION PARITY whisper-\(name): mel cosine \(melSimilarity)")
                XCTAssertGreaterThan(melSimilarity, 0.999, "\(name): the front end matches")
            }

            let tokens = net.transcribe(mel).map { Int32($0) }
            print("VALIDATION PARITY whisper-\(name): tokens \(tokens) vs \(Array(referenceTokens))")
            XCTAssertEqual(tokens, Array(referenceTokens), "\(name): greedy decoding matches")
        }
    }

    // MARK: CLIP text tower

    // The image tower has had a parity record since the harness existed; the text tower never did.
    // The record carries the reference's own token ids, because the port embeds ids rather than text —
    // a byte-level BPE vocabulary is a load-time artifact, not part of the network.
    func testCLIPTextTowerMatchesTheReferenceEmbedding() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_CLIP_TEXT"] else { throw XCTSkip("set IK_PARITY_CLIP_TEXT") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self).map(Int.init)
        let referenceOutput = try XCTUnwrap(arrays["output"])

        let net = NFKMLXCLIPNet(.base)
        try NFKMLXCLIP.loadWeights(into: net, from: weights("IK_VAL_CLIP"))

        let embedding = net.encodeText(tokens)
        eval(embedding)
        let ours = embedding.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.asArray(Float.self).map(Double.init)
        XCTAssertEqual(ours.count, theirs.count, "same embedding width as the reference")
        let similarity = cosine(ours, theirs)
        print("VALIDATION PARITY clip-text: cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the text tower matches the reference implementation")
    }

    // MARK: BiSeNet

    // Segmentation logits against CoinCheung's own BiSeNetV1 on the released Cityscapes weights.
    // Comparing logits rather than the label map keeps the check sensitive — an argmax hides every
    // difference too small to flip a pixel's winner.
    func testBiSeNetMatchesTheReferenceLogits() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_BISENET")

        let net = NFKMLXBiSeNet.makeNet()
        try NFKMLXBiSeNet.loadWeights(into: net, from: weights("IK_VAL_BISENET"))
        net.train(false)

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let ours = net.logits(NFKMLXBiSeNetNet.normalized(inputTensor.reshaped([1, height, width, 3])))
        eval(ours)
        XCTAssertEqual(Array(ours.shape.dropFirst()), referenceOutput.shape, "same logit grid as the reference")

        let mine = ours.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        let classes = referenceOutput.shape[2]
        let ourLabels = ours.reshaped([-1, classes]).argMax(axis: -1).asArray(Int32.self)
        let referenceLabels = referenceOutput.reshaped([-1, classes]).argMax(axis: -1).asArray(Int32.self)
        let agreement = Double(zip(ourLabels, referenceLabels).filter(==).count) / Double(ourLabels.count)
        print("VALIDATION PARITY bisenet: logit cosine \(similarity), label agreement \(agreement)")
        XCTAssertGreaterThan(similarity, 0.99, "the class logits match the reference implementation")
        XCTAssertGreaterThan(agreement, 0.99, "and the predicted labels agree")
    }

    // MARK: MODNet

    // The portrait matte against ZHKKKe's own MODNet on the released photographic checkpoint. The
    // network splits the work across three branches over one MobileNetV2 encoder, so a single alpha
    // comparison covers the semantic branch, the detail branch, and the fusion that combines them.
    func testMODNetMatchesTheReferenceMatte() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceAlpha) = try record("IK_PARITY_MODNET")

        let net = NFKMLXMODNet.makeNet()
        try NFKMLXMODNet.loadWeights(into: net, from: weights("IK_VAL_MODNET"))
        net.train(false)

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let alpha = net.matte(batched: inputTensor.reshaped([1, height, width, 3]))
        eval(alpha)
        XCTAssertEqual([alpha.shape[1], alpha.shape[2]], referenceAlpha.shape, "same matte size as the reference")

        let mine = alpha.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceAlpha.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        let error = zip(mine, theirs).map { abs($0 - $1) }.reduce(0, +) / Double(mine.count)
        print("VALIDATION PARITY modnet: alpha cosine \(similarity), mean |difference| \(error)")
        XCTAssertGreaterThan(similarity, 0.99, "the matte matches the reference implementation")
        XCTAssertLessThan(error, 0.02, "and agrees pointwise, not just in shape")
    }

    // MARK: NAFNet

    // Restoration against megvii-research's own NAFNet on the released SIDD width-32 denoiser. The
    // model was previously untested against real weights purely because they were thought to be
    // Google-Drive-only; a Hugging Face mirror carries them.
    func testNAFNetMatchesTheReferenceRestoration() throws {
        try requireMLXRuntime()
        try assertImageParity("IK_PARITY_NAFNET", model: "nafnet") {
            try NFKMLXNAFNet.backend(weightsURL: weights("IK_VAL_NAFNET"))
        }
    }

    // MARK: LaMa

    // Inpainting against advimman's own FFCResNetGenerator on the released big-lama weights. This was
    // the sweep's one deliberately unmeasured model: its convolutions zero-padded where the reference
    // reflection-pads, an approximation that cost style transfer a 0.049 mean pixel error. The `raw`
    // seam is the generator's own output before compositing, so a mismatch there is the network while
    // a mismatch only in `output` is the paste-back.
    func testLaMaMatchesTheReferenceInpainting() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_LAMA")
        guard let path = config["IK_PARITY_LAMA"] else { throw XCTSkip("set IK_PARITY_LAMA") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let mask = try XCTUnwrap(arrays["mask"])                          // [H, W], 1 where regenerated

        let net = NFKMLXLaMaNet(NFKMLXLaMaConfiguration())
        try NFKMLXLaMa.loadWeights(into: net, from: weights("IK_VAL_LAMA"))
        net.train(false)

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let ours = net.inpaint(inputTensor, mask: mask.reshaped([height, width, 1]))
        eval(ours)
        XCTAssertEqual(ours.shape, referenceOutput.shape, "same inpainted size as the reference")

        let mine = ours.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        let error = zip(mine, theirs).map { abs($0 - $1) }.reduce(0, +) / Double(mine.count)
        print("VALIDATION PARITY lama: inpainted cosine \(similarity), mean |difference| \(error)")
        XCTAssertGreaterThan(similarity, 0.99, "the inpainted result matches the reference implementation")
        XCTAssertLessThan(error, 0.02, "and agrees pixelwise, not just in shape")
    }

    // MARK: Video super-resolution (BasicVSR)

    // The upscaled clip against mmediting's own BasicVSRNet on the released REDS4 weights. The clip is
    // three genuinely translated crops, so SPyNet's flow, both propagation warps, and the
    // reconstructor all contribute to the number — a wrong flow or padding mode cannot hide.
    func testVideoSRMatchesTheReferenceClip() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_VIDEOSR"] else { throw XCTSkip("set IK_PARITY_VIDEOSR") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let frames = try XCTUnwrap(arrays["frames"])                      // [T, H, W, 3]
        let referenceOutput = try XCTUnwrap(arrays["output"])             // [T, 4H, 4W, 3]

        let net = NFKMLXVideoSRNet(.base)
        try NFKMLXVideoSR.loadWeights(into: net, from: weights("IK_VAL_VIDEOSR"))

        let clip = (0 ..< frames.shape[0]).map { frames[$0] }
        let outputs = net.upscaleSequence(clip)
        let ours = concatenated(outputs.map { $0.reshaped([1] + $0.shape) }, axis: 0)
        eval(ours)
        XCTAssertEqual(ours.shape, referenceOutput.shape, "same upscaled clip shape as the reference")

        let mine = ours.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        let error = zip(mine, theirs).map { abs($0 - $1) }.reduce(0, +) / Double(mine.count)
        print("VALIDATION PARITY videosr: clip cosine \(similarity), mean |difference| \(error)")
        XCTAssertGreaterThan(similarity, 0.99, "the upscaled clip matches the reference implementation")
        XCTAssertLessThan(error, 0.05, "and agrees pointwise, not just in shape")
    }

    // A frame whose sides are not a multiple of the encoder's stride takes a different path: the
    // decoder's source pooling meets an odd dimension (the reference pools with `ceil_mode` and
    // `count_include_pad=False`, so a one-sample remainder averages to itself) and every upsampled
    // feature is cropped back to the skip's size. An even plate exercises neither, so the main parity
    // record cannot speak for a photo of arbitrary size.
    func testRVMMatchesTheReferenceOnAFrameThatIsNotAMultipleOfTheStride() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceAlpha) = try record("IK_PARITY_RVM_ODD")
        let net = NFKMLXRVMNet()
        try NFKMLXRVM.loadWeights(into: net, from: weights("IK_VAL_RVM"))
        net.train(false)

        let frame = inputTensor.reshaped([1, inputTensor.shape[0], inputTensor.shape[1], 3])
        let (_, alpha, _) = net.forward(frame, state: NFKMLXRVMNet.initialState)
        eval(alpha)

        let mine = alpha.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceAlpha.reshaped([-1]).asArray(Float.self).map(Double.init)
        XCTAssertEqual(mine.count, theirs.count, "same matte size as the reference")
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY rvm (odd size): alpha cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99, "the odd-dimension pooling and crops match the reference")
    }

    // MARK: RAFT

    // Optical flow against princeton-vl's own RAFT, on the released raft-things weights. The recurrent
    // update produces the eighth-resolution field; comparing that first separates the network from the
    // convex-mask upsampling that follows it.
    func testRAFTMatchesTheReferenceFlow() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_RAFT")
        guard let path = config["IK_PARITY_RAFT"] else { throw XCTSkip("set IK_PARITY_RAFT") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let second = try XCTUnwrap(arrays["frame1"])

        // The recurrent update is run the same number of times on both sides; a different count is a
        // different answer, so `run_reference.py raft` passes the port's default.
        let net = NFKMLXRAFT.makeNet()
        try NFKMLXRAFT.loadWeights(into: net, from: weights("IK_VAL_RAFT"))

        let ours = net.flowLow(inputTensor, second)
        eval(ours)
        XCTAssertEqual(ours.shape, referenceOutput.shape, "same flow grid as the reference")
        let mine = ours.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        let error = zip(mine, theirs).map { abs($0 - $1) }.reduce(0, +) / Double(mine.count)
        print("VALIDATION PARITY raft: eighth-resolution flow cosine \(similarity), mean |difference| \(error)")
        XCTAssertGreaterThan(similarity, 0.99, "the flow field matches the reference implementation")

        if let referenceUp = arrays["flow_up"] {
            let full = net.flow(inputTensor, second)
            eval(full)
            let upSimilarity = cosine(full.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceUp.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY raft: full-resolution flow cosine \(upSimilarity)")
            XCTAssertGreaterThan(upSimilarity, 0.99, "and so does the upsampled field")
        }
    }

    // MARK: Audio tagger

    // AudioSet tag probabilities against PANNs' own Cnn14. Three seams in one record: the mel front end,
    // the clip embedding, and the probabilities. A sigmoid compresses most of the 527 classes toward
    // zero, so the embedding is where a network difference stays legible.
    func testAudioTaggerMatchesTheReferenceTags() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_AUDIOTAGGER"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_AUDIOTAGGER to an audio_tagger record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"])                  // [samples]
        let reference = try XCTUnwrap(arrays["output"]).asArray(Float.self).map(Double.init)

        let net = NFKMLXAudioTagger.makeNet()
        try NFKMLXAudioTagger.loadWeights(into: net, from: weights("IK_VAL_AUDIOTAGGER"))
        let mel = net.frontEnd.logMel(waveform.asArray(Float.self))
        eval(mel)

        if let referenceFeatures = arrays["features"] {
            XCTAssertEqual(Array(mel.shape.dropFirst()), referenceFeatures.shape, "same spectrogram grid")
            let similarity = cosine(mel.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    referenceFeatures.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY audio-tagger: mel cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.999, "the mel front end matches the reference")
        }
        if let referenceEmbedding = arrays["embedding"] {
            let ours = net.embedding(mel)
            eval(ours)
            let similarity = cosine(ours.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    referenceEmbedding.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY audio-tagger: embedding cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.999, "the clip embedding matches the reference")
        }

        let ours = sigmoid(net.logits(mel))
        eval(ours)
        let mine = ours.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, reference)
        let topReference = reference.enumerated().max { $0.element < $1.element }?.offset
        let topOurs = mine.enumerated().max { $0.element < $1.element }?.offset
        print("VALIDATION PARITY audio-tagger: tag cosine \(similarity), top class \(topOurs ?? -1) vs \(topReference ?? -1)")
        XCTAssertGreaterThan(similarity, 0.99, "the tag probabilities match the reference implementation")
        XCTAssertEqual(topOurs, topReference, "and the highest-scoring class is the same")
    }

    // MARK: Pose

    // Joint heatmaps against microsoft's own SimpleBaseline, run on the released ResNet-50 weights.
    // Comparing the heatmaps rather than the decoded joints keeps the check sensitive: an argmax reports
    // the same cell until a difference is large enough to move the peak.
    func testPoseMatchesTheReferenceHeatmaps() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_POSE")
        let net = NFKMLXPose.makeNet()
        try NFKMLXPose.loadWeights(into: net, from: weights("IK_VAL_POSE"))

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let ours = net.heatmaps(NFKMLXPoseNet.normalized(inputTensor).reshaped([1, height, width, 3]))
        eval(ours)

        XCTAssertEqual(Array(ours.shape.dropFirst()), referenceOutput.shape, "same heatmap grid per joint")
        let mine = ours.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)

        // Every joint's peak cell, which is what the decode reads.
        let joints = referenceOutput.shape[2]
        let ourPeaks = ours.reshaped([-1, joints]).argMax(axis: 0).asArray(Int32.self)
        let referencePeaks = referenceOutput.reshaped([-1, joints]).argMax(axis: 0).asArray(Int32.self)
        let agreement = Double(zip(ourPeaks, referencePeaks).filter(==).count) / Double(joints)
        print("VALIDATION PARITY pose: heatmap cosine \(similarity), peak agreement \(agreement)")
        XCTAssertGreaterThan(similarity, 0.99, "the joint heatmaps match the reference implementation")
        XCTAssertGreaterThan(agreement, 0.99, "and every joint peaks in the same cell")
    }

    // MARK: VAD

    // Frame-level speech probability against NeMo's own MarbleNet. The mel front end is inside the
    // comparison: both sides start from the same waveform and build their own features, so a window,
    // filterbank, or preemphasis difference shows up here.
    func testVADMatchesTheReferenceSpeechProbabilities() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_VAD"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_VAD to a vad record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"])                  // [samples]
        let reference = try XCTUnwrap(arrays["output"]).asArray(Float.self).map(Double.init)

        let net = NFKMLXVAD.makeNet()
        try NFKMLXVAD.loadWeights(into: net, from: weights("IK_VAL_VAD"))
        let samples = waveform.asArray(Float.self)

        // The mel spectrogram first: a front-end mismatch and an encoder mismatch look identical at the
        // output, and only one of them is worth reading the network for.
        if let referenceFeatures = arrays["features"] {
            let features = net.frontEnd.logMel(samples)
            eval(features)
            let mine = features.reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = referenceFeatures.reshaped([-1]).asArray(Float.self).map(Double.init)
            let featureSimilarity = cosine(mine, theirs)
            print("VALIDATION PARITY vad: mel cosine \(featureSimilarity) over \(mine.count) values")
            XCTAssertGreaterThan(featureSimilarity, 0.999, "the mel front end matches the reference")
        }

        let ours = net.speechProbabilities(samples, sampleRate: 16000).map(Double.init)

        XCTAssertEqual(ours.count, reference.count, "one probability per frame, as the reference emits")
        let similarity = cosine(ours, reference)
        let largest = zip(ours, reference).map { abs($0 - $1) }.max() ?? 1
        print("VALIDATION PARITY vad: cosine \(similarity), largest |difference| \(largest)")
        XCTAssertGreaterThan(similarity, 0.99, "the speech probabilities match the reference implementation")
        XCTAssertLessThan(largest, 0.02, "and agree frame by frame, not just in shape")
    }

    // MARK: SwinIR

    // Super-resolution output against the released classical-SR ×4 model. The input side must be a
    // multiple of the window size, so the reference record uses a 64×64 plate.
    func testSwinIRMatchesTheReferenceUpscale() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_SWINIR")
        let net = NFKMLXSwinIRNet(.classicalSRx4)
        try NFKMLXSwinIR.loadWeights(into: net, from: weights("IK_VAL_SWINIR"))

        let upscaled = net.upscale(inputTensor)                 // [4H, 4W, 3]
        eval(upscaled)
        let count = referenceOutput.size
        XCTAssertEqual(upscaled.size, count, "same upscaled volume as the reference")
        let ours = upscaled.reshaped([count]).asArray(Float.self).map(Double.init)
        let reference = referenceOutput.reshaped([count]).asArray(Float.self).map(Double.init)

        let similarity = cosine(ours, reference)
        let meanAbsolute = zip(ours, reference).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(count)
        print("VALIDATION PARITY swinir: cosine \(similarity), mean |difference| \(meanAbsolute)")
        XCTAssertGreaterThan(similarity, 0.999, "the upscale matches the reference implementation")
        XCTAssertLessThan(meanAbsolute, 0.01, "and agrees pixelwise")
    }

    // The ×3 release is a different upsampler, not a different factor: one ×3 pixel-shuffle stage
    // where ×4 runs two ×2 stages. A checkpoint fits only its own scale, so this measures the
    // non-power-of-two path rather than repeating the ×4 one.
    func testSwinIRMatchesTheReferenceUpscaleAtScaleThree() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_SWINIR_X3")
        let net = try NFKMLXSwinIR.makeNet(.classicalSRx3)
        try NFKMLXSwinIR.loadWeights(into: net, from: weights("IK_VAL_SWINIR_X3"))

        let upscaled = net.upscale(inputTensor)                 // [3H, 3W, 3]
        eval(upscaled)
        XCTAssertEqual(upscaled.shape[0], inputTensor.shape[0] * 3, "a ×3 upsampler triples the side")
        let count = referenceOutput.size
        XCTAssertEqual(upscaled.size, count, "same upscaled volume as the reference")
        let ours = upscaled.reshaped([count]).asArray(Float.self).map(Double.init)
        let reference = referenceOutput.reshaped([count]).asArray(Float.self).map(Double.init)

        let similarity = cosine(ours, reference)
        let meanAbsolute = zip(ours, reference).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(count)
        print("VALIDATION PARITY swinir-x3: cosine \(similarity), mean |difference| \(meanAbsolute)")
        XCTAssertGreaterThan(similarity, 0.999, "the ×3 upscale matches the reference implementation")
        XCTAssertLessThan(meanAbsolute, 0.01, "and agrees pixelwise")
    }

    // MARK: RetinaFace

    // The detector the CodeFormer reference pipeline runs. Compared BEFORE suppression, over the whole
    // prediction tensor: a detection-level match alone cannot separate a network mismatch from a
    // threshold or decoding difference, and this can. The anchors are checked too, since a wrong
    // anchor grid decodes plausible boxes in the wrong places.
    func testRetinaFaceMatchesTheReferenceDetector() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_RETINAFACE"] else { throw XCTSkip("set IK_PARITY_RETINAFACE") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let input = try XCTUnwrap(arrays["input_image"])
        let referenceBoxes = try XCTUnwrap(arrays["output"])
        let referenceScores = try XCTUnwrap(arrays["scores"])
        let referenceLandmarks = try XCTUnwrap(arrays["landmark_offsets"])
        let referencePriors = try XCTUnwrap(arrays["priors"])

        let net = NFKMLXRetinaFace.makeNet()
        try NFKMLXRetinaFace.loadWeights(into: net, from: weights("IK_VAL_RETINAFACE"))

        let (height, width) = (input.shape[0], input.shape[1])
        let (boxes, scores, landmarks) = net(NFKMLXRetinaFace.prepared(input))
        eval(boxes, scores, landmarks)

        XCTAssertEqual(boxes.shape, referenceBoxes.shape, "the same anchor count as the reference")

        func similarity(_ ours: MLXArray, _ theirs: MLXArray) -> Double {
            cosine(ours.reshaped([-1]).asArray(Float.self).map(Double.init),
                   theirs.reshaped([-1]).asArray(Float.self).map(Double.init))
        }
        let boxSimilarity = similarity(boxes, referenceBoxes)
        let scoreSimilarity = similarity(scores, referenceScores)
        let landmarkSimilarity = similarity(landmarks, referenceLandmarks)
        print("VALIDATION PARITY retinaface: box cosine \(boxSimilarity), "
              + "score \(scoreSimilarity), landmark \(landmarkSimilarity)")
        XCTAssertGreaterThan(boxSimilarity, 0.9999, "the box head matches the reference")
        XCTAssertGreaterThan(scoreSimilarity, 0.9999, "the class head matches the reference")
        XCTAssertGreaterThan(landmarkSimilarity, 0.9999, "the landmark head matches the reference")

        // The anchor grid is generated, not learned, so it has to agree exactly.
        let ours = NFKMLXRetinaFace.anchors(height: height, width: width, configuration: net.configuration)
        eval(ours)
        XCTAssertEqual(ours.shape, referencePriors.shape)
        let anchorError = (ours - referencePriors).abs().max().item(Float.self)
        XCTAssertEqual(anchorError, 0, accuracy: 1e-6, "the anchors are the reference's")
    }

    // End to end through decoding and suppression, against the reference's own detect_faces.
    func testRetinaFaceDetectionsMatchTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_RETINAFACE"] else { throw XCTSkip("set IK_PARITY_RETINAFACE") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let input = try XCTUnwrap(arrays["input_image"])
        let reference = try XCTUnwrap(arrays["detections"])          // [n, 15]: box, score, 10 landmarks

        let net = NFKMLXRetinaFace.makeNet()
        try NFKMLXRetinaFace.loadWeights(into: net, from: weights("IK_VAL_RETINAFACE"))
        let found = NFKMLXRetinaFace.detect(tensor: input, using: net)

        let expected = reference.shape[0]
        XCTAssertEqual(found.count, expected, "the same faces as the reference")
        guard expected > 0, let first = found.first else { return }

        let values = reference[0].asArray(Float.self)
        let box = CGRect(x: CGFloat(values[0]), y: CGFloat(values[1]),
                         width: CGFloat(values[2] - values[0]), height: CGFloat(values[3] - values[1]))
        let intersection = first.boundingBox.intersection(box)
        let union = first.boundingBox.union(box)
        let iou = (intersection.width * intersection.height) / (union.width * union.height)
        print("VALIDATION PARITY retinaface: detections \(found.count)/\(expected), box IoU \(iou)")
        XCTAssertGreaterThan(iou, 0.99, "the box matches the reference's")
        XCTAssertEqual(Double(first.confidence), Double(values[4]), accuracy: 1e-4)

        for point in 0 ..< 5 {
            XCTAssertEqual(Double(first.landmarks[point].x), Double(values[5 + point * 2]), accuracy: 1,
                           "landmark \(point) x")
            XCTAssertEqual(Double(first.landmarks[point].y), Double(values[5 + point * 2 + 1]), accuracy: 1,
                           "landmark \(point) y")
        }
    }

    // MARK: Qwen3

    // The dense decoder against transformers' own Qwen3ForCausalLM on the released 0.6B weights.
    // Compared at the LOGITS first: a token match alone would hide a small numeric drift, and the
    // prefill covers every position rather than only the last.
    func testQwen3MatchesTheReferenceLogits() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_QWEN3"], let directory = config["IK_VAL_QWEN3"] else {
            throw XCTSkip("set IK_PARITY_QWEN3 and IK_VAL_QWEN3")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])

        let release = URL(fileURLWithPath: directory)
        let configuration = try NFKMLXLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        let net = NFKMLXLanguage.makeNet(configuration)
        try NFKMLXLanguage.loadWeights(into: net,
                                       from: release.appendingPathComponent("model.safetensors"))

        let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, referenceLogits.shape[0], referenceLogits.shape[1]])

        let ours = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, theirs)
        let worst = zip(ours, theirs).map { abs($0 - $1) }.max() ?? 0
        print("VALIDATION PARITY qwen3: logit cosine \(similarity), worst |difference| \(worst)")
        XCTAssertGreaterThan(similarity, 0.9999, "the decoder matches the reference implementation")

        // The argmax at every position is what generation actually consumes.
        let vocabulary = referenceLogits.shape[1]
        for position in 0 ..< referenceLogits.shape[0] {
            let base = position * vocabulary
            let ourBest = (0 ..< vocabulary).max { ours[base + $0] < ours[base + $1] }
            let theirBest = (0 ..< vocabulary).max { theirs[base + $0] < theirs[base + $1] }
            XCTAssertEqual(ourBest, theirBest, "position \(position) predicts the same token")
        }
    }

    // MARK: Qwen3-Embedding

    // The retrieval task the embedding record is measured on, declared identically to the oracle's so
    // the tokenizer-agreement check reproduces the reference's ids from the text.
    private static let embeddingTask =
        "Given a web search query, retrieve relevant passages that answer the query"

    // The embedder against the model card's own transformers recipe on the released 0.6B weights: the
    // base decoder's last hidden state, pooled at the LAST token and L2-normalized. The record carries
    // the reference's ids, so this feeds them and compares the embedding — isolating the network and
    // pooling from the tokenizer, which the next test covers on its own.
    func testQwen3EmbeddingMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_QWEN3_EMBEDDING"],
              let directory = config["IK_VAL_QWEN3_EMBEDDING"] else {
            throw XCTSkip("set IK_PARITY_QWEN3_EMBEDDING and IK_VAL_QWEN3_EMBEDDING")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let queryTokens = try XCTUnwrap(arrays["query_tokens"]).asArray(Int32.self).map(Int.init)
        let documentTokens = try XCTUnwrap(arrays["document_tokens"]).asArray(Int32.self).map(Int.init)
        let referenceQuery = try XCTUnwrap(arrays["output"]).asArray(Float.self).map(Double.init)
        let referenceDocument = try XCTUnwrap(arrays["document_embedding"]).asArray(Float.self).map(Double.init)
        let referenceScore = Double(try XCTUnwrap(arrays["score"]).asArray(Float.self)[0])

        let release = URL(fileURLWithPath: directory)
        let configuration = try NFKMLXLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        let net = NFKMLXLanguage.makeNet(configuration)
        // The released checkpoint is the base model (no `model.` prefix), so the embedder's own loader
        // reads it rather than the causal-LM loader.
        try NFKMLXQwen3Embedding.loadWeights(into: net, fromDirectory: release)
        // The recorded ids already carry the appended end token, so the embedder must not add another.
        let embedder = NFKMLXTextEmbedder(net: net, configuration:
            NFKMLXTextEmbedderConfiguration(pooling: .lastToken, appendedToken: nil, normalizes: true))

        let ourQuery = embedder.embed(tokens: queryTokens)
        let ourDocument = embedder.embed(tokens: documentTokens)
        eval(ourQuery, ourDocument)
        let query = ourQuery.asArray(Float.self).map(Double.init)
        let document = ourDocument.asArray(Float.self).map(Double.init)

        let querySimilarity = cosine(query, referenceQuery)
        let documentSimilarity = cosine(document, referenceDocument)
        let score = zip(query, document).map(*).reduce(0, +)
        print("VALIDATION PARITY qwen3-embedding: query cosine \(querySimilarity), "
              + "document cosine \(documentSimilarity), retrieval score \(score) vs \(referenceScore)")
        XCTAssertGreaterThan(querySimilarity, 0.9999, "the query embedding matches the reference")
        XCTAssertGreaterThan(documentSimilarity, 0.9999, "the document embedding matches the reference")
        XCTAssertEqual(score, referenceScore, accuracy: 1e-3, "the retrieval score matches end to end")
    }

    // Token-for-token agreement with the reference tokenizer, so the ids the network reads are the ids
    // the reference read. The GPT-2 default pre-tokenization would produce different, valid-looking ids
    // for the same text, which no embedding would ever reveal.
    func testQwen3EmbeddingTokenizerMatchesTheReference() throws {
        guard let path = config["IK_PARITY_QWEN3_EMBEDDING"],
              let directory = config["IK_VAL_QWEN3_EMBEDDING"] else {
            throw XCTSkip("set IK_PARITY_QWEN3_EMBEDDING and IK_VAL_QWEN3_EMBEDDING")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let referenceQuery = try XCTUnwrap(arrays["query_tokens"]).asArray(Int32.self).map(Int.init)
        let referenceDocument = try XCTUnwrap(arrays["document_tokens"]).asArray(Int32.self).map(Int.init)

        let release = URL(fileURLWithPath: directory)
        let tokenizer = try XCTUnwrap(NFKMLXLanguage.releaseTokenizer(inDirectory: release))
        let (specials, _) = NFKMLXLanguage.specialTokens(inDirectory: release)
        let endToken = specials["<|endoftext|>"] ?? 151_643

        let query = NFKMLXQwen3Embedding.instruct(task: Self.embeddingTask,
                                                  query: "What is the capital of China?")
        let queryIds = tokenizer.encode(query).map(\.intValue) + [endToken]
        let documentIds = tokenizer.encode("The capital of China is Beijing.").map(\.intValue) + [endToken]

        XCTAssertEqual(queryIds, referenceQuery, "the query tokenizes to the reference's ids")
        XCTAssertEqual(documentIds, referenceDocument, "the document tokenizes to the reference's ids")
    }

    // MARK: EmbeddingGemma

    // The bidirectional Gemma 3 encoder + mean pooling + Dense bottleneck against the
    // sentence-transformers pipeline on the released 300M weights. The record carries the reference's
    // per-layer hidden states, so a divergence is located to a layer rather than guessed from the
    // embedding — the harness that debugged the Gemma 4 decoder here.
    func testEmbeddingGemmaMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_EMBEDDINGGEMMA"],
              let directory = config["IK_VAL_EMBEDDINGGEMMA"] else {
            throw XCTSkip("set IK_PARITY_EMBEDDINGGEMMA and IK_VAL_EMBEDDINGGEMMA")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let queryTokens = try XCTUnwrap(arrays["query_tokens"]).asArray(Int32.self).map(Int.init)
        let documentTokens = try XCTUnwrap(arrays["document_tokens"]).asArray(Int32.self).map(Int.init)
        let referenceQuery = try XCTUnwrap(arrays["output"]).asArray(Float.self).map(Double.init)
        let referenceDocument = try XCTUnwrap(arrays["document_embedding"]).asArray(Float.self).map(Double.init)
        let referenceScore = Double(try XCTUnwrap(arrays["score"]).asArray(Float.self)[0])

        let release = URL(fileURLWithPath: directory)
        let net = NFKMLXGemma3EncoderNet(.embeddingGemma300M)
        let (dense2, dense3) = try NFKMLXEmbeddingGemma.loadWeights(into: net, fromDirectory: release)

        // The recorded ids already carry BOS and EOS, so the embedder must not add its own markers.
        let embedder = NFKMLXEmbeddingGemmaEmbedder(
            net: net, dense2: dense2, dense3: dense3,
            configuration: NFKMLXEmbeddingGemmaConfiguration(prependedToken: nil, appendedToken: nil))

        // Per-layer isolation over the query, which is what locates a Gemma convention mistake.
        let states = net.layerStates(MLXArray(queryTokens.map(Int32.init)).reshaped([1, queryTokens.count]))
        eval(states)
        var firstBad: Int?
        var report = [String]()
        for (index, state) in states.enumerated() {
            guard let reference = arrays["hidden.\(index)"] else { break }
            let mine = state[0].reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            let similarity = cosine(mine, theirs)
            let name = index == 0 ? "embedding" : "after layer \(index - 1)"
            report.append(String(format: "  %-18s cosine %.10f", (name as NSString).utf8String!, similarity))
            if similarity < 0.9999 && firstBad == nil { firstBad = index }
        }
        print("VALIDATION isolation embeddinggemma:\n" + report.joined(separator: "\n"))
        XCTAssertNil(firstBad, "first divergence at \(firstBad.map { $0 == 0 ? "the embedding" : "layer \($0 - 1)" } ?? "-")")

        let ourQuery = embedder.embed(tokens: queryTokens)
        let ourDocument = embedder.embed(tokens: documentTokens)
        eval(ourQuery, ourDocument)
        let query = ourQuery.asArray(Float.self).map(Double.init)
        let document = ourDocument.asArray(Float.self).map(Double.init)
        let querySimilarity = cosine(query, referenceQuery)
        let documentSimilarity = cosine(document, referenceDocument)
        let score = zip(query, document).map(*).reduce(0, +)
        print("VALIDATION PARITY embeddinggemma: query cosine \(querySimilarity), "
              + "document cosine \(documentSimilarity), retrieval score \(score) vs \(referenceScore)")
        XCTAssertGreaterThan(querySimilarity, 0.9999, "the query embedding matches the reference")
        XCTAssertGreaterThan(documentSimilarity, 0.9999, "the document embedding matches the reference")
        XCTAssertEqual(score, referenceScore, accuracy: 1e-3, "the retrieval score matches end to end")
    }

    // Token-for-token agreement with the reference tokenizer. Gemma's tokenizer is a SentencePiece
    // unigram model; NFKUnigramTokenizer reads the converted unigram.json, and the embedder wraps its
    // content ids in BOS and EOS. The whole path reproduces the reference's ids or the network reads a
    // different sentence.
    func testEmbeddingGemmaTokenizerMatchesTheReference() throws {
        guard let path = config["IK_PARITY_EMBEDDINGGEMMA"],
              let directory = config["IK_VAL_EMBEDDINGGEMMA"] else {
            throw XCTSkip("set IK_PARITY_EMBEDDINGGEMMA and IK_VAL_EMBEDDINGGEMMA")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let referenceQuery = try XCTUnwrap(arrays["query_tokens"]).asArray(Int32.self).map(Int.init)
        let referenceDocument = try XCTUnwrap(arrays["document_tokens"]).asArray(Int32.self).map(Int.init)

        let release = URL(fileURLWithPath: directory)
        let tokenizer = try XCTUnwrap(NFKMLXGemmaTokenizer(directoryURL: release))
        // BOS then the content ids then EOS, which is what the embedder builds around the tokenizer.
        func wrapped(_ text: String) -> [Int] {
            [2] + tokenizer.encode(text) + [1]
        }
        XCTAssertEqual(wrapped(NFKMLXEmbeddingGemma.query("What is the capital of China?")),
                       referenceQuery, "the query tokenizes to the reference's ids")
        XCTAssertEqual(wrapped(NFKMLXEmbeddingGemma.document("The capital of China is Beijing.")),
                       referenceDocument, "the document tokenizes to the reference's ids")
    }

    // MARK: ModernBERT reranker

    private static let rerankQuery = "What is the capital of France?"
    private static let rerankRelevant =
        "Paris is the capital and most populous city of France. Situated on the river Seine "
        + "in the north of the country, it has been a major European centre of finance, "
        + "diplomacy, commerce, fashion, and art for centuries. The city is home to landmarks "
        + "such as the Eiffel Tower, the Louvre, and the cathedral of Notre-Dame, and its "
        + "metropolitan area is among the largest in Europe."
    private static let rerankIrrelevant =
        "The Great Barrier Reef is the world's largest coral reef system, off Australia."

    // The cross-encoder against transformers' own ModernBertForSequenceClassification on the released
    // gte-reranker weights. The relevant pair is long enough to engage the local sliding window, and
    // the record carries its per-layer hidden states so a wrong window or convention is located.
    func testModernBertRerankerMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_MODERNBERT_RERANKER"],
              let directory = config["IK_VAL_MODERNBERT_RERANKER"] else {
            throw XCTSkip("set IK_PARITY_MODERNBERT_RERANKER and IK_VAL_MODERNBERT_RERANKER")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let relevantTokens = try XCTUnwrap(arrays["relevant_tokens"]).asArray(Int32.self).map(Int.init)
        let irrelevantTokens = try XCTUnwrap(arrays["irrelevant_tokens"]).asArray(Int32.self).map(Int.init)
        let referenceRelevant = Double(try XCTUnwrap(arrays["output"]).asArray(Float.self)[0])
        let referenceIrrelevant = Double(try XCTUnwrap(arrays["irrelevant_score"]).asArray(Float.self)[0])

        let release = URL(fileURLWithPath: directory)
        let net = NFKMLXModernBertRerankerNet(.gteReranker)
        try NFKMLXModernBERTReranker.loadWeights(into: net, fromDirectory: release)

        // Per-layer isolation over the relevant pair, which is what locates a window or norm mistake.
        let states = net.model.layerStates(MLXArray(relevantTokens.map(Int32.init)).reshaped([1, relevantTokens.count]))
        eval(states)
        var firstBad: Int?
        var report = [String]()
        for (index, state) in states.enumerated() {
            guard let reference = arrays["hidden.\(index)"] else { break }
            let mine = state[0].reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            let similarity = cosine(mine, theirs)
            report.append(String(format: "  %-16s cosine %.10f", ((index == 0 ? "embedding" : "after layer \(index - 1)") as NSString).utf8String!, similarity))
            if similarity < 0.9999 && firstBad == nil { firstBad = index }
        }
        print("VALIDATION isolation modernbert-reranker:\n" + report.joined(separator: "\n"))
        XCTAssertNil(firstBad, "first divergence at \(firstBad.map { $0 == 0 ? "the embedding" : "layer \($0 - 1)" } ?? "-")")

        let relevant = Double(net.score(tokens: relevantTokens).asArray(Float.self)[0])
        let irrelevant = Double(net.score(tokens: irrelevantTokens).asArray(Float.self)[0])
        print("VALIDATION PARITY modernbert-reranker: relevant \(relevant) vs \(referenceRelevant), "
              + "irrelevant \(irrelevant) vs \(referenceIrrelevant)")
        XCTAssertEqual(relevant, referenceRelevant, accuracy: 5e-3, "the relevant score matches")
        XCTAssertEqual(irrelevant, referenceIrrelevant, accuracy: 5e-3, "the irrelevant score matches")
        XCTAssertGreaterThan(relevant, irrelevant, "the reranker ranks the relevant document first")

        // The public ranking API over real weights: the relevant document sorts ahead of the other.
        let reranker = try NFKMLXModernBERTReranker.reranker(directoryURL: release)
        let ranked = reranker.rankedIndices(query: Self.rerankQuery,
                                            documents: [Self.rerankIrrelevant, Self.rerankRelevant])
        XCTAssertEqual(ranked.map(\.intValue), [1, 0], "the reranking puts the relevant document first")
    }

    // Token-for-token agreement: the byte-level BPE tokenizer plus the [CLS] query [SEP] document [SEP]
    // wrapping reproduces the reference's ids.
    func testModernBertRerankerTokenizerMatchesTheReference() throws {
        guard let path = config["IK_PARITY_MODERNBERT_RERANKER"],
              let directory = config["IK_VAL_MODERNBERT_RERANKER"] else {
            throw XCTSkip("set IK_PARITY_MODERNBERT_RERANKER and IK_VAL_MODERNBERT_RERANKER")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let referenceRelevant = try XCTUnwrap(arrays["relevant_tokens"]).asArray(Int32.self).map(Int.init)
        let referenceIrrelevant = try XCTUnwrap(arrays["irrelevant_tokens"]).asArray(Int32.self).map(Int.init)

        let reranker = try NFKMLXModernBERTReranker.reranker(directoryURL: URL(fileURLWithPath: directory))
        XCTAssertEqual(reranker.tokens(query: Self.rerankQuery, document: Self.rerankRelevant),
                       referenceRelevant, "the relevant pair tokenizes to the reference's ids")
        XCTAssertEqual(reranker.tokens(query: Self.rerankQuery, document: Self.rerankIrrelevant),
                       referenceIrrelevant, "the irrelevant pair tokenizes to the reference's ids")
    }

    // MARK: SmolVLM2 (vision-language)

    // The whole VLM against transformers' own SmolVLMForConditionalGeneration on the released 500M
    // weights, staged: the SigLIP vision encoder, the pixel-shuffle connector, the fused decoder logits,
    // and the greedy continuation. The record carries the processor's pixel values, so the image
    // processor is out of the comparison and each network is measured on its own.
    func testSmolVLMMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SMOLVLM"], let directory = config["IK_VAL_SMOLVLM2"] else {
            throw XCTSkip("set IK_PARITY_SMOLVLM and IK_VAL_SMOLVLM2")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let inputIds = try XCTUnwrap(arrays["input_ids"]).asArray(Int32.self).map(Int.init)
        let pixelValues = try XCTUnwrap(arrays["pixel_values"])
        let referenceVision = try XCTUnwrap(arrays["vision_hidden"])
        let referenceFeatures = try XCTUnwrap(arrays["image_features"])
        let referenceLogits = try XCTUnwrap(arrays["output"])
        let continuation = try XCTUnwrap(arrays["continuation"]).asArray(Int32.self).map(Int.init)

        let model = try NFKMLXSmolVLM.model(directoryURL: URL(fileURLWithPath: directory))

        // 0. The SigLIP patch + position embeddings and the first encoder layer, when recorded, so a
        // vision divergence is located to the embeddings, a layer, or the norm.
        if let referenceEmbeddings = arrays["vision_embeddings"], let referenceLayer0 = arrays["vision_layer0"] {
            let embeddings = model.vision.embeddings(pixelValues.transposed(0, 2, 3, 1))
            let layer0 = model.vision.encoder.layers[0](embeddings)
            eval(embeddings, layer0)
            let embeddingSimilarity = cosine(embeddings.reshaped([-1]).asArray(Float.self).map(Double.init),
                                             referenceEmbeddings.reshaped([-1]).asArray(Float.self).map(Double.init))
            let layer0Similarity = cosine(layer0.reshaped([-1]).asArray(Float.self).map(Double.init),
                                          referenceLayer0.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION isolation smolvlm: embeddings cosine \(embeddingSimilarity), layer0 cosine \(layer0Similarity)")
            XCTAssertGreaterThan(embeddingSimilarity, 0.9999, "the patch + position embeddings match")
            XCTAssertGreaterThan(layer0Similarity, 0.9999, "the first encoder layer matches")
        }

        // 1. The SigLIP vision encoder, over the reference's own pixel values.
        let vision = model.vision(pixelValues.transposed(0, 2, 3, 1))
        eval(vision)
        let visionSimilarity = cosine(vision.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      referenceVision.reshaped([-1]).asArray(Float.self).map(Double.init))
        // 2. The connector's projected vision tokens.
        let features = model.connector(vision)
        eval(features)
        let featureSimilarity = cosine(features.reshaped([-1]).asArray(Float.self).map(Double.init),
                                       referenceFeatures.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION isolation smolvlm: vision cosine \(visionSimilarity), connector cosine \(featureSimilarity)")
        XCTAssertGreaterThan(visionSimilarity, 0.9999, "the SigLIP encoder matches the reference")
        XCTAssertGreaterThan(featureSimilarity, 0.9999, "the connector matches the reference")

        // 3. The fused decoder logits: the same next token at every position, and the last position exact.
        let logits = model.logits(inputIds: inputIds, pixelValues: pixelValues)
        eval(logits)
        let ourArgmax = logits[0].argMax(axis: -1).asArray(Int32.self)
        let referenceArgmax = referenceLogits.argMax(axis: -1).asArray(Int32.self)
        let agreement = zip(ourArgmax, referenceArgmax).filter { $0 == $1 }.count
        let last = inputIds.count - 1
        let lastSimilarity = cosine(logits[0, last].asArray(Float.self).map(Double.init),
                                    referenceLogits[last].asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY smolvlm: argmax agreement \(agreement)/\(ourArgmax.count), "
              + "last-position logit cosine \(lastSimilarity)")
        XCTAssertEqual(agreement, ourArgmax.count, "every position predicts the reference's token")
        XCTAssertGreaterThan(lastSimilarity, 0.9999, "the last-position logits match")

        // 4. The greedy continuation, token for token.
        let produced = model.generate(inputIds: inputIds, pixelValues: pixelValues,
                                      maxTokens: continuation.count, endTokens: [])
        print("VALIDATION smolvlm continuation: \(produced)")
        XCTAssertEqual(produced, continuation, "the greedy continuation reproduces the reference")
    }

    // The whole consumer path on a real photograph: the CoreGraphics image processor, the prompt, and
    // greedy generation produce a coherent caption. The resize is not the reference's PIL LANCZOS, so
    // this checks the answer is real language rather than a token-identical match.
    func testSmolVLMAnswersAboutARealImage() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_VAL_SMOLVLM2"], let facePath = config["IK_VAL_FACE"] else {
            throw XCTSkip("set IK_VAL_SMOLVLM2 and IK_VAL_FACE")
        }
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: facePath) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw XCTSkip("could not read the validation image")
        }
        let model = try NFKMLXSmolVLM.load(directoryURL: URL(fileURLWithPath: directory))
        let answer = model.answer(image: image, question: "What is in this image?", maxTokens: 24)
        print("VALIDATION smolvlm answer: \(answer)")
        XCTAssertFalse(answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "produces a caption")
        XCTAssertGreaterThan(answer.split(separator: " ").count, 2, "the caption is a multi-word description")
    }

    // The prompt builder plus the byte-level BPE tokenizer reproduce the processor's input ids: the
    // "User:" opener, the 4×4 tiled image structure with a global thumbnail, the question, and the
    // assistant turn, all token for token.
    func testSmolVLMPromptMatchesTheReference() throws {
        guard let path = config["IK_PARITY_SMOLVLM"], let directory = config["IK_VAL_SMOLVLM2"] else {
            throw XCTSkip("set IK_PARITY_SMOLVLM and IK_VAL_SMOLVLM2")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let referenceIds = try XCTUnwrap(arrays["input_ids"]).asArray(Int32.self).map(Int.init)

        let tokenizer = try XCTUnwrap(NFKMLXSmolVLM.tokenizer(inDirectory: URL(fileURLWithPath: directory)))
        let prompt = NFKMLXSmolVLM.prompt(rows: 4, cols: 4, question: "What is in this image?")
        let ids = tokenizer.encode(prompt).map(\.intValue)
        XCTAssertEqual(ids, referenceIds, "the expanded prompt tokenizes to the reference's ids")
    }

    // MARK: Qwen3-VL vision tower

    // The Qwen3-VL 2D-rotary ViT + merger + deepstack against transformers' own vision model on the
    // released 2B weights, staged: the patch embedding, the interpolated position embedding, the merged
    // output, and the three deepstack feature maps.
    func testQwen3VLVisionMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_QWEN3_VL"], let directory = config["IK_VAL_QWEN3_VL"] else {
            throw XCTSkip("set IK_PARITY_QWEN3_VL and IK_VAL_QWEN3_VL")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let pixelValues = try XCTUnwrap(arrays["pixel_values"])
        let g = try XCTUnwrap(arrays["image_grid_thw"]).asArray(Int32.self).map(Int.init)
        let grid = (t: g[0], h: g[1], w: g[2])

        let net = try NFKMLXQwen3VL.visionNet(directoryURL: URL(fileURLWithPath: directory))

        func compare(_ ours: MLXArray, _ reference: MLXArray, _ label: String, threshold: Double = 0.9999) {
            eval(ours)
            let similarity = cosine(ours.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION isolation qwen3vl \(label): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, threshold, "\(label) matches the reference")
        }

        compare(net.patchEmbed(pixelValues), try XCTUnwrap(arrays["patch_embed"]), "patch_embed")
        compare(net.interpolatedPositionEmbedding(grid: grid), try XCTUnwrap(arrays["pos_embeds"]), "pos_embeds")

        let (output, deepstack) = net(pixelValues, grid: grid)
        compare(output, try XCTUnwrap(arrays["vision_output"]), "vision_output")
        for index in 0 ..< deepstack.count {
            compare(deepstack[index], try XCTUnwrap(arrays["deepstack_\(index)"]), "deepstack_\(index)")
        }
    }

    // MARK: GGUF reader

    // The native GGUF reader's dequantizers against the `gguf` package on a real Q4_K_M model. For the
    // first tensor of each block-quant type (the same one both sides pick in file order), the
    // dequantized values match the reference's.
    func testGGUFDequantizationMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GGUF"], let ggufPath = config["IK_VAL_GGUF"] else {
            throw XCTSkip("set IK_PARITY_GGUF and IK_VAL_GGUF")
        }
        let reference = try loadArrays(url: URL(fileURLWithPath: path))
        let gguf = try NFKMLXGGUF.gguf(contentsOf: URL(fileURLWithPath: ggufPath))

        for type in ["F32", "Q8_0", "Q5_0", "Q4_K", "Q6_K"] {
            guard let recorded = reference[type] else { continue }
            let referenceValues = recorded.asArray(Float.self)
            let name = try XCTUnwrap(gguf.tensorNames.first { gguf.info(forTensor: $0)?.typeName == type },
                                     "the model has a \(type) tensor")
            let ours = try XCTUnwrap(gguf.array(forTensor: name)).reshaped([-1]).asArray(Float.self)
            XCTAssertGreaterThanOrEqual(ours.count, referenceValues.count)

            var worst = Float(0)
            for index in 0 ..< referenceValues.count {
                worst = Swift.max(worst, abs(ours[index] - referenceValues[index]))
            }
            print("VALIDATION PARITY gguf \(type) (\(name)): \(referenceValues.count) values, worst |diff| \(worst)")
            XCTAssertLessThan(worst, 1e-4, "the \(type) dequantization matches the reference")
        }

        // The container's metadata reads too.
        XCTAssertEqual(gguf.metadataString(forKey: "general.architecture"), "llama")
        XCTAssertGreaterThan(gguf.tensorNames.count, 0)
    }

    // MARK: Mixture of experts (tiny-configuration oracles)

    // The released mixture sizes do not fit this machine, so the routed feed-forward is measured
    // at a tiny random configuration against transformers' own implementation, layer by layer, in
    // both released families: Qwen3-MoE (softmax over every expert, top-k, renormalized) and
    // Mixtral (softmax over the selected — the same arithmetic — under different tensor names).
    private func mixtureParity(recordKey: String, mode: String, label: String,
                               geometry: NFKMLXLanguageConfiguration) throws {
        try requireMLXRuntime()
        guard let path = config[recordKey] else {
            throw XCTSkip("set \(recordKey) (run_reference.py \(mode))")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])

        let net = NFKMLXLanguage.makeNet(geometry)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("w::") ? (NFKMLXLanguage.moduleKey(forRelease: String(key.dropFirst(3))), value) : nil
        }
        try NFKMLXWeights.apply(NFKMLXLanguage.stackingExperts(weights), to: net, verifyShapes: true)

        let input = MLXArray(tokens).reshaped([1, tokens.count])
        let states = net.layerStates(input)
        eval(states)
        var report = [String]()
        var firstBad: Int?
        for (index, state) in states.enumerated() {
            guard let reference = arrays["hidden.\(index)"] else { break }
            let mine = state.reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            guard mine.count == theirs.count else {
                report.append("\(index): shape \(state.shape) vs \(reference.shape)")
                firstBad = firstBad ?? index
                break
            }
            let similarity = cosine(mine, theirs)
            let name = index == 0 ? "embedding" : "after layer \(index - 1)"
            report.append(String(format: "  %-18s cosine %.10f", (name as NSString).utf8String!, similarity))
            if similarity < 0.9999 && firstBad == nil { firstBad = index }
        }
        print("VALIDATION isolation \(label):\n" + report.joined(separator: "\n"))
        XCTAssertNil(firstBad, "first divergence at \(firstBad.map { $0 == 0 ? "the embedding" : "layer \($0 - 1)" } ?? "-")")

        let logits = net(input)
        eval(logits)
        let mine = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY \(label): logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the routed decoder's arithmetic matches the reference")
        let vocabulary = referenceLogits.shape[1]
        for position in 0 ..< referenceLogits.shape[0] {
            let base = position * vocabulary
            let ourBest = (0 ..< vocabulary).max { mine[base + $0] < mine[base + $1] }
            let theirBest = (0 ..< vocabulary).max { theirs[base + $0] < theirs[base + $1] }
            XCTAssertEqual(ourBest, theirBest, "position \(position) predicts the same token")
        }
    }

    func testQwen3MoeTinyMatchesTheReferenceLayerByLayer() throws {
        var geometry = NFKMLXLanguageConfiguration(
            hiddenSize: 64, layerCount: 3, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            intermediateSize: 96, vocabularySize: 128, ropeTheta: 10_000, rmsEpsilon: 1e-6,
            tiesWordEmbeddings: false)
        geometry.normalizesQueryAndKey = true
        geometry.expertCount = 8
        geometry.activeExpertCount = 2
        geometry.expertIntermediateSize = 32
        geometry.normalizesExpertWeights = true
        try mixtureParity(recordKey: "IK_PARITY_QWEN3_MOE_TINY", mode: "qwen3_moe",
                          label: "qwen3-moe-tiny", geometry: geometry)
    }

    func testMixtralTinyMatchesTheReferenceLayerByLayer() throws {
        var geometry = NFKMLXLanguageConfiguration(
            hiddenSize: 64, layerCount: 3, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            intermediateSize: 32, vocabularySize: 128, ropeTheta: 10_000, rmsEpsilon: 1e-6,
            tiesWordEmbeddings: false)
        geometry.normalizesQueryAndKey = false
        geometry.expertCount = 4
        geometry.activeExpertCount = 2
        geometry.expertIntermediateSize = 32
        geometry.normalizesExpertWeights = true
        try mixtureParity(recordKey: "IK_PARITY_MIXTRAL_TINY", mode: "mixtral",
                          label: "mixtral-tiny", geometry: geometry)
    }

    // The release's special tokens and end token come from tokenizer_config.json, not vocab.json:
    // a ChatML marker must encode to ONE id, and the end token must be the one the release names.
    func testTheReleaseTokenizerResolvesItsSpecialTokens() throws {
        guard let directory = config["IK_VAL_QWEN3"] else { throw XCTSkip("set IK_VAL_QWEN3") }
        let (specials, endToken) = NFKMLXLanguage.specialTokens(inDirectory: URL(fileURLWithPath: directory))
        XCTAssertEqual(specials["<|im_start|>"], 151644)
        XCTAssertEqual(specials["<|im_end|>"], 151645)
        XCTAssertEqual(endToken, 151645, "Qwen3's eos_token is <|im_end|>")

        var manifest: [String: Any] = ["tokenizer": ["type": "bpe-bytelevel", "pretokenizer": "qwen2",
                                                     "specialTokens": specials]]
        manifest["eosTokenId"] = endToken
        let tokenizer = try NFKTokenizer(forManifest: manifest, directory: URL(fileURLWithPath: directory))
        XCTAssertEqual(tokenizer.eosTokenId, 151645)
        let rendered = NFKMLXLanguageBackend.chatMLPrompt(from: [["role": "user", "content": "Hi"]])
        let ids = tokenizer.encode(rendered).map(\.intValue)
        XCTAssertEqual(ids.first, 151644, "the opening marker is one token: \(ids.prefix(4))")
        XCTAssertTrue(ids.contains(151645))
    }

    // JSON-constrained generation on released weights: whatever the model says, it is a document.
    func testConstrainedGenerationOnQwen3ProducesAJSONObject() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_VAL_QWEN3"] else { throw XCTSkip("set IK_VAL_QWEN3") }
        let backend = try NFKMLXLanguage.backend(directoryURL: URL(fileURLWithPath: directory))
        // Qwen3 opens every answer with a <think> block. The grammar forbids it, and what the model
        // does with the leftover probability mass is not an answer (measured: `{}`), so the prompt
        // closes the block itself, the way the release's own template does for its no-think mode.
        let prompt = NFKMLXLanguageBackend.chatMLPrompt(from: [
            ["role": "user", "content": "Describe Paris as a JSON object with the keys city, country, and population."],
        ]) + "<think>\n\n</think>\n\n"
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: prompt],
            parameters: [NFKParameterMaxTokens: 96, NFKParameterTemperature: 0,
                         NFKMLXGenerationParameterKey.outputFormat: "json-object"])
        let text = try XCTUnwrap(backend.runInference(for: request).text)
        print("VALIDATION constrained qwen3-0.6B: \(text.trimmingCharacters(in: .whitespacesAndNewlines))")
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any],
                                   "the output is a JSON object")
        XCTAssertFalse(object.isEmpty, "the object carries fields")
    }

    // Speculative decoding on released weights: a 0.6B draft proposing for the 1.7B target must
    // reproduce the target's own greedy continuation token for token — that is the whole contract —
    // and the acceptance rate and wall-clock ratio are what say whether it was worth it.
    func testSpeculativeDecodingReproducesTheTargetsGreedyRunOnQwen3() throws {
        try requireMLXRuntime()
        guard let targetDirectory = config["IK_VAL_QWEN3_1_7B"], let draftDirectory = config["IK_VAL_QWEN3"] else {
            throw XCTSkip("set IK_VAL_QWEN3_1_7B and IK_VAL_QWEN3")
        }
        let (target, tokenizer) = try NFKMLXLanguage.loadedRelease(at: URL(fileURLWithPath: targetDirectory))
        let (draft, _) = try NFKMLXLanguage.loadedRelease(at: URL(fileURLWithPath: draftDirectory))
        let prompt = try XCTUnwrap(tokenizer).encode("The capital of France is").map(\.intValue)

        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = 64
        options.draftTokens = 4

        var plainSeconds = Date()
        let plain = target.generate(prompt: prompt, options: options)
        let plainElapsed = Date().timeIntervalSince(plainSeconds)

        var report = NFKMLXSpeculativeReport()
        plainSeconds = Date()
        let speculative = target.generate(prompt: prompt, options: options, draft: draft, report: &report)
        let speculativeElapsed = Date().timeIntervalSince(plainSeconds)

        print(String(format: "VALIDATION speculative qwen3-1.7B←0.6B: %d tokens, %d rounds, acceptance %.3f, "
                     + "plain %.2f s, speculative %.2f s, ratio %.2fx",
                     plain.count, report.rounds, report.acceptanceRate, plainElapsed, speculativeElapsed,
                     plainElapsed / speculativeElapsed))
        XCTAssertEqual(speculative, plain, "the speculative run is the target's own greedy run")
        XCTAssertGreaterThan(report.acceptanceRate, 0.5, "a same-family draft agrees most of the time")
    }

    // Follow-on to the music embedding win: the SMALLER Qwen3 releases are TIED, so their input
    // embedding IS the logit head — `logits(fromHidden:)` routes through `embedTokens.asLinear`, which
    // becomes a QuantizedEmbedding.asLinear once the embedding is packed. So quantizing the embedding
    // here quantizes the output projection too, the cost the untied music LM avoided. This measures
    // that hit against the Qwen3 logit records, at 4-bit and 8-bit, before any tied model packs its
    // embedding by default. Opt-in: it loads each release several times.
    func testTheTiedEmbeddingQuantizationCostAgainstTheRecord() throws {
        try requireMLXRuntime()
        try XCTSkipUnless(config["IK_QWEN_EMB_PROBE"] == "1",
                          "set IK_QWEN_EMB_PROBE=1 to probe quantizing a tied model's embedding")
        let models = [("qwen3-0.6B", "IK_PARITY_QWEN3", "IK_VAL_QWEN3"),
                      ("qwen3-1.7B", "IK_PARITY_QWEN3_1_7B", "IK_VAL_QWEN3_1_7B")]
        for (label, recordKey, dirKey) in models {
            guard let path = config[recordKey], let directory = config[dirKey] else {
                print("EMB PROBE \(label): skipped (records absent)"); continue
            }
            let arrays = try loadArrays(url: URL(fileURLWithPath: path))
            let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
            let reference = try XCTUnwrap(arrays["output"]).reshaped([-1])
                .asArray(Float.self).map(Double.init)
            let release = URL(fileURLWithPath: directory)
            let configuration = try NFKMLXLanguage.configuration(
                fromHuggingFace: release.appendingPathComponent("config.json"))
            XCTAssertTrue(configuration.tiesWordEmbeddings, "\(label) is expected to be tied")

            func measure(_ prepare: (NFKMLXLanguageNet) -> Void) throws -> Double {
                let net = NFKMLXLanguage.makeNet(configuration)
                try NFKMLXLanguage.loadWeights(into: net, fromDirectory: release)
                prepare(net)
                let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
                eval(logits)
                let ours = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
                NFKMLXGPU.clearCache()
                return cosine(ours, reference)
            }

            func packEmbedding(_ net: NFKMLXLanguageNet, bits: Int) {
                MLXNN.quantize(model: net, groupSize: 64, bits: bits) { _, layer in
                    guard let embedding = layer as? Embedding,
                          !(embedding is QuantizedEmbedding) else { return false }
                    return embedding.weight.shape[1] % 64 == 0
                }
            }
            let baseline = try measure { _ in }
            let linear4 = try measure { NFKMLXQuantization.quantize(module: $0, bits: 4, groupSize: 64) }
            let linear4embedding4 = try measure {
                NFKMLXQuantization.quantize(module: $0, bits: 4, groupSize: 64, includeEmbeddings: true)
            }
            let linear8 = try measure { NFKMLXQuantization.quantize(module: $0, bits: 8, groupSize: 64) }
            let linear8embedding8 = try measure {
                NFKMLXQuantization.quantize(module: $0, bits: 8, groupSize: 64, includeEmbeddings: true)
            }
            let linear4embedding8 = try measure { net in
                NFKMLXQuantization.quantize(module: net, bits: 4, groupSize: 64)
                packEmbedding(net, bits: 8)
            }
            print("EMB PROBE \(label): float \(baseline)")
            print("EMB PROBE \(label): 4-bit Linear \(linear4), +4-bit emb \(linear4embedding4), "
                  + "+8-bit emb \(linear4embedding8)")
            print("EMB PROBE \(label): 8-bit Linear \(linear8), +8-bit emb \(linear8embedding8)")
            XCTAssertGreaterThan(baseline, 0.9999, "\(label) baseline reproduces the record")
            // The tied embedding at the SAME width as the Linear layers costs almost nothing beyond
            // the Linear quantization itself, which is the dominant term for these small models.
            XCTAssertGreaterThan(linear8embedding8, linear8 - 0.01,
                                 "\(label): packing the tied embedding at 8-bit is nearly free")
        }
    }

    // End to end: greedy generation through the key-value cache has to reproduce the reference's own
    // continuation exactly. This is what proves the cache, the rotary offsets, and the sampling path
    // together — none of which the single forward pass above exercises.
    func testQwen3ReproducesTheReferenceContinuation() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_QWEN3"], let directory = config["IK_VAL_QWEN3"] else {
            throw XCTSkip("set IK_PARITY_QWEN3 and IK_VAL_QWEN3")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self).map(Int.init)
        let reference = try XCTUnwrap(arrays["continuation"]).asArray(Int32.self).map(Int.init)

        let release = URL(fileURLWithPath: directory)
        let configuration = try NFKMLXLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        let net = NFKMLXLanguage.makeNet(configuration)
        try NFKMLXLanguage.loadWeights(into: net,
                                       from: release.appendingPathComponent("model.safetensors"))

        var options = NFKMLXGenerationOptions()
        options.temperature = 0                       // greedy, as the reference generated
        options.maxTokens = reference.count
        let produced = net.generate(prompt: tokens, options: options)

        print("VALIDATION PARITY qwen3: continuation \(produced) vs reference \(reference)")
        XCTAssertEqual(produced, reference, "greedy decoding reproduces the reference's tokens")
    }

    // The same decoder at a larger size, from a SHARDED release. Every Qwen3 above 0.6B splits its
    // weights across files with an index naming which shard holds each tensor, so this covers the
    // loader's shard path as well as the geometry.
    func testQwen3AtOnePointSevenBillionMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_QWEN3_1_7B"], let directory = config["IK_VAL_QWEN3_1_7B"] else {
            throw XCTSkip("set IK_PARITY_QWEN3_1_7B and IK_VAL_QWEN3_1_7B")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])
        let referenceContinuation = try XCTUnwrap(arrays["continuation"]).asArray(Int32.self).map(Int.init)

        let release = URL(fileURLWithPath: directory)
        let configuration = try NFKMLXLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        XCTAssertEqual(configuration.hiddenSize, 2048, "the 1.7B geometry is read from its own config")
        let net = NFKMLXLanguage.makeNet(configuration)
        try NFKMLXLanguage.loadWeights(into: net, fromDirectory: release)

        let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(logits)
        let ours = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, theirs)
        print("VALIDATION PARITY qwen3-1.7b: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the larger size matches the reference too")

        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = referenceContinuation.count
        let produced = net.generate(prompt: tokens.map(Int.init), options: options)
        XCTAssertEqual(produced, referenceContinuation, "greedy decoding reproduces the reference")
    }

    // The two largest released YOLOv8 sizes. Both run the full depth multiple, so their C2f stages
    // repeat [3, 6, 6, 3] where the smaller sizes repeat less; x is wider still. A wrong width or
    // repeat count fails the strict load with a shape rather than a number, which is what makes these
    // presets self-checking.
    func testYOLOLargeSizesMatchTheReference() throws {
        try requireMLXRuntime()
        let sizes: [(String, String, String, NFKMLXYOLOConfiguration)] = [
            ("l", "IK_VAL_YOLO_LARGE", "IK_PARITY_YOLO_LARGE", .large),
            ("x", "IK_VAL_YOLO_XLARGE", "IK_PARITY_YOLO_XLARGE", .extraLarge),
        ]
        for (name, weightsKey, parityKey, geometry) in sizes {
            let (inputTensor, referenceOutput) = try record(parityKey)
            let net = NFKMLXYOLONet(geometry)
            try NFKMLXYOLO.loadWeights(into: net, from: weights(weightsKey))
            net.train(false)

            let ours = net.predictions(inputTensor)
            eval(ours)
            XCTAssertEqual(ours.shape, referenceOutput.shape, "yolov8\(name): same prediction grid")
            let boxes = cosine(ours[0..., 0 ..< 4].reshaped([-1]).asArray(Float.self).map(Double.init),
                               referenceOutput[0..., 0 ..< 4].reshaped([-1]).asArray(Float.self).map(Double.init))
            let classes = cosine(ours[0..., 4...].reshaped([-1]).asArray(Float.self).map(Double.init),
                                 referenceOutput[0..., 4...].reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY yolov8\(name): box cosine \(boxes), class cosine \(classes)")
            XCTAssertGreaterThan(boxes, 0.99, "yolov8\(name) boxes match the release")
            XCTAssertGreaterThan(classes, 0.99, "and so do its class probabilities")
        }
    }

    // The largest size this machine can hold at float32: 4B parameters is about 16 GB of weights on
    // each side, and the oracle and this run are separate processes. Three shards rather than two, a
    // deeper stack (36 layers), and a wider hidden size than 1.7B — all read from the release's own
    // config rather than a preset, which is what makes the family scale by configuration.
    func testQwen3AtFourBillionMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_QWEN3_4B"], let directory = config["IK_VAL_QWEN3_4B"] else {
            throw XCTSkip("set IK_PARITY_QWEN3_4B and IK_VAL_QWEN3_4B")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])
        let referenceContinuation = try XCTUnwrap(arrays["continuation"]).asArray(Int32.self).map(Int.init)

        let release = URL(fileURLWithPath: directory)
        let configuration = try NFKMLXLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        XCTAssertEqual(configuration.layerCount, 36)
        XCTAssertEqual(configuration.hiddenSize, 2560)
        let net = NFKMLXLanguage.makeNet(configuration)
        try NFKMLXLanguage.loadWeights(into: net, fromDirectory: release)

        let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(logits)
        let ours = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, theirs)
        print("VALIDATION PARITY qwen3-4b: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the 4B size matches the reference")

        var options = NFKMLXGenerationOptions()
        options.temperature = 0
        options.maxTokens = referenceContinuation.count
        XCTAssertEqual(net.generate(prompt: tokens.map(Int.init), options: options),
                       referenceContinuation, "greedy decoding reproduces the reference")
    }

    // The x8 release: the same network as x4 with a THIRD ×2 pixel-shuffle stage, which is what the
    // power-of-two branch of the reference upsampler produces. A checkpoint fits only its own scale,
    // so the strict load is itself a check on the stage count.
    func testSwinIRMatchesTheReferenceUpscaleAtScaleEight() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_SWINIR_X8")
        let net = try NFKMLXSwinIR.makeNet(.classicalSRx8)
        try NFKMLXSwinIR.loadWeights(into: net, from: weights("IK_VAL_SWINIR_X8"))
        XCTAssertEqual(NFKMLXSwinIRConfiguration.classicalSRx8.upsampleStages, 3)

        let upscaled = net.upscale(inputTensor)
        eval(upscaled)
        XCTAssertEqual(upscaled.shape[0], inputTensor.shape[0] * 8, "a ×8 upsampler multiplies the side by eight")
        let count = referenceOutput.size
        let ours = upscaled.reshaped([count]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([count]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, theirs)
        let meanAbsolute = zip(ours, theirs).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(count)
        print("VALIDATION PARITY swinir-x8: cosine \(similarity), mean |difference| \(meanAbsolute)")
        XCTAssertGreaterThan(similarity, 0.999, "the ×8 upscale matches the reference")
        XCTAssertLessThan(meanAbsolute, 0.01, "and agrees pixelwise")
    }

    // The lightweight release, which is not the classical network at a smaller size: it reconstructs
    // through `pixelshuffledirect` — ONE convolution to 3·scale² channels and a single shuffle, with
    // neither the convolution before the upsampler nor the one after it. Its checkpoint carries a
    // single `upsample.0`, so a strict load is itself the check that the tail is right.
    func testSwinIRLightweightMatchesTheReference() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_SWINIR_LIGHT_X2")
        let net = try NFKMLXSwinIR.makeNet(.lightweightSRx2)
        XCTAssertNil(net.convBeforeUpsample, "the direct upsampler has no convolution before it")
        XCTAssertNil(net.convLast, "and none after it")
        try NFKMLXSwinIR.loadWeights(into: net, from: weights("IK_VAL_SWINIR_LIGHT_X2"))

        let upscaled = net.upscale(inputTensor)
        eval(upscaled)
        XCTAssertEqual(upscaled.shape[0], inputTensor.shape[0] * 2)
        let count = referenceOutput.size
        let ours = upscaled.reshaped([count]).asArray(Float.self).map(Double.init)
        let theirs = referenceOutput.reshaped([count]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, theirs)
        let meanAbsolute = zip(ours, theirs).reduce(0.0) { $0 + abs($1.0 - $1.1) } / Double(count)
        print("VALIDATION PARITY swinir-light-x2: cosine \(similarity), mean |difference| \(meanAbsolute)")
        XCTAssertGreaterThan(similarity, 0.999, "the lightweight release matches the reference")
        XCTAssertLessThan(meanAbsolute, 0.01, "and agrees pixelwise")
    }

    // MARK: Gemma 4

    // The Gemma 4 text decoder against transformers' own implementation on the released E2B weights.
    //
    // Reaching this took the per-layer isolation harness below: a structural check passed 600/600
    // while the forward scored 0.0044, and four whole-model guesses moved it to 0.48 without finding
    // the cause. Isolating showed the input to layer 0 was exact and its `input_layernorm` was not,
    // which located the first defect precisely — Gemma 4 normalizes with a PLAIN scale, where Gemma 3
    // used `1 + weight`. Then the MLP (Gemma's activation is gelu, not silu), then the full-attention
    // layers' `proportional` rotary. Each fix was one measurement, not a guess.
    func testGemma4MatchesTheReferenceLogits() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4"], let directory = config["IK_VAL_GEMMA4"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4 and IK_VAL_GEMMA4")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])

        let release = URL(fileURLWithPath: directory)
        let geometry = try NFKMLXGemmaLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        let net = NFKMLXGemmaLanguage.makeNet(geometry)
        try NFKMLXGemmaLanguage.loadWeights(into: net, fromDirectory: release)

        let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, referenceLogits.shape[0], referenceLogits.shape[1]])

        let ours = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, theirs)
        print("VALIDATION PARITY gemma4-e2b: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.999, "the decoder matches the reference implementation")

        let vocabulary = referenceLogits.shape[1]
        for position in 0 ..< referenceLogits.shape[0] {
            let base = position * vocabulary
            let ourBest = (0 ..< vocabulary).max { ours[base + $0] < ours[base + $1] }
            let theirBest = (0 ..< vocabulary).max { theirs[base + $0] < theirs[base + $1] }
            XCTAssertEqual(ourBest, theirBest, "position \(position) predicts the same token")
        }
    }

    // The next released size, measured at the precision it ships in. E4B's 16 GB of bf16 weights
    // would double past this machine's RAM at float32, so BOTH sides run bf16 — the oracle under
    // IK_GEMMA_DTYPE=bfloat16, this side at `.checkpoint` — and the thresholds are half-precision
    // ones. The structural claim (42 layers, 2 KV heads, kv-shared 18, doubled widths) is what this
    // measures; the E2B float32 record is what pins the arithmetic to more digits.
    func testGemma4E4BMatchesTheReferenceLogits() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_E4B"], let directory = config["IK_VAL_GEMMA4_E4B"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_E4B and IK_VAL_GEMMA4_E4B")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])

        let release = URL(fileURLWithPath: directory)
        let geometry = try NFKMLXGemmaLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        let net = NFKMLXGemmaLanguage.makeNet(geometry)
        try NFKMLXGemmaLanguage.loadWeights(into: net, fromDirectory: release, precision: .checkpoint)

        let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, referenceLogits.shape[0], referenceLogits.shape[1]])

        let ours = logits[0].reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, theirs)
        print("VALIDATION PARITY gemma4-e4b: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99,
                             "the decoder matches the reference at the released precision")

        let vocabulary = referenceLogits.shape[1]
        var agreements = 0
        for position in 0 ..< referenceLogits.shape[0] {
            let base = position * vocabulary
            let ourBest = (0 ..< vocabulary).max { ours[base + $0] < ours[base + $1] }
            let theirBest = (0 ..< vocabulary).max { theirs[base + $0] < theirs[base + $1] }
            if ourBest == theirBest { agreements += 1 }
        }
        print("VALIDATION PARITY gemma4-e4b: argmax agreement \(agreements)/\(referenceLogits.shape[0])")
        XCTAssertEqual(agreements, referenceLogits.shape[0],
                       "every position predicts the same token")
    }

    // The isolation harness. A whole-model cosine says a port is wrong; this says WHERE. The oracle
    // records the state entering the stack and the state each layer produced, so walking them together
    // finds the FIRST layer that diverges — everything after it is downstream of that one mistake.
    func testGemma4LayerByLayerAgainstTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4"], let directory = config["IK_VAL_GEMMA4"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4 and IK_VAL_GEMMA4")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        guard arrays["hidden.0"] != nil else {
            throw XCTSkip("the record predates the per-layer capture; regenerate it")
        }
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)

        let release = URL(fileURLWithPath: directory)
        let geometry = try NFKMLXGemmaLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        let net = NFKMLXGemmaLanguage.makeNet(geometry)
        try NFKMLXGemmaLanguage.loadWeights(into: net, fromDirectory: release)

        let ours = net.hiddenStates(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(ours)

        var firstBad: Int?
        var report = [String]()
        for index in 0 ..< ours.count {
            guard let reference = arrays["hidden.\(index)"] else { break }
            let mine = ours[index].reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            guard mine.count == theirs.count else {
                report.append("\(index): shape \(ours[index].shape) vs \(reference.shape)")
                firstBad = firstBad ?? index
                break
            }
            let similarity = cosine(mine, theirs)
            let label = index == 0 ? "embedding" : "after layer \(index - 1)"
            report.append(String(format: "  %-18s cosine %.10f", (label as NSString).utf8String!, similarity))
            if similarity < 0.9999 && firstBad == nil { firstBad = index }
        }
        print("VALIDATION isolation gemma4:\n" + report.joined(separator: "\n"))
        if let firstBad {
            let label = firstBad == 0 ? "the embedding" : "layer \(firstBad - 1)"
            print("VALIDATION isolation gemma4: FIRST DIVERGENCE at \(label)")
        } else {
            print("VALIDATION isolation gemma4: every layer matches")
        }
    }

    // MARK: Gemma 4 mixture of experts (the 26B-A4B family)

    // The routed-expert block — the scale-free router norm, the learned per-channel and per-expert
    // scales, the softmax and top-k, and the fused-projection experts summed BESIDE the dense
    // feed-forward — at a tiny random configuration, since the released mixture does not fit. The
    // oracle saves its weights in the release naming (`model.layers.N.experts.gate_up_proj`), which the
    // net's own key paths match once the `model.` prefix is stripped, so no per-tensor remap is needed.
    // Every hidden state is compared so a divergence lands on a layer.
    func testGemma4MixtureTinyMatchesTheReferenceLayerByLayer() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_MOE"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_MOE (run_reference.py gemma4_moe)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])

        let net = NFKMLXGemmaLanguage.makeNet(.tinyMixture)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::model.") else { return nil }
            return (String(key.dropFirst("w::model.".count)), value)
        }
        try NFKMLXWeights.apply(weights, to: net)

        let ours = net.hiddenStates(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(ours)

        var firstBad: Int?
        var report = [String]()
        for index in 0 ..< ours.count {
            guard let reference = arrays["hidden.\(index)"] else { break }
            let mine = ours[index].reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            guard mine.count == theirs.count else {
                report.append("\(index): shape \(ours[index].shape) vs \(reference.shape)")
                firstBad = firstBad ?? index
                break
            }
            let similarity = cosine(mine, theirs)
            let label = index == 0 ? "embedding" : "after layer \(index - 1)"
            report.append(String(format: "  %-18s cosine %.10f", (label as NSString).utf8String!, similarity))
            if similarity < 0.9999 && firstBad == nil { firstBad = index }
        }
        print("VALIDATION isolation gemma4-moe-tiny:\n" + report.joined(separator: "\n"))
        XCTAssertNil(firstBad, "first divergence at \(firstBad.map { $0 == 0 ? "the embedding" : "layer \($0 - 1)" } ?? "-")")

        let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(logits)
        let mine = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY gemma4-moe-tiny: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the mixture block's arithmetic matches the reference")
    }

    // The Gemma text-generation backend end to end: build it from the released E2B directory and
    // generate greedily. The decoder is already at parity layer by layer, so this exercises the
    // tokenizer and the prefill-only generation loop — the pieces the backend adds — and asserts the
    // output is coherent text rather than empty or garbage.
    func testGemmaBackendGeneratesText() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_VAL_GEMMA4"] else { throw XCTSkip("set IK_VAL_GEMMA4") }
        let backend = try NFKMLXGemmaLanguage.backend(directoryURL: URL(fileURLWithPath: directory))
        XCTAssertTrue(backend.isReady)
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "The capital of France is"],
            parameters: [NFKParameterMaxTokens: 8, NFKParameterTemperature: 0])
        let result = try backend.runInference(for: request)
        let text = try XCTUnwrap(result.output(forKey: NFKOutputText) as? String)
        print("VALIDATION gemma backend generated: \(text.debugDescription)")
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       "the backend produces non-empty text")
    }

    // MARK: Gemma 4 unified text decoder (the 12B)

    // The unified decoder — a different architecture from the E-series: no per-layer input embeddings
    // and no mixture, but the same attention (learned query/key norms, a scale-free value norm, scale
    // 1, per-layer head widths, the proportional rotary on the full layers). Measured at a tiny
    // configuration over a sliding/sliding/full layer mix, since the 12B does not fit. The oracle saves
    // its weights in the release naming, which the net's keys match once `model.` is stripped.
    func testGemma4UnifiedTinyMatchesTheReferenceLayerByLayer() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_UNIFIED"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_UNIFIED (run_reference.py gemma4_unified)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])

        let net = NFKMLXGemmaLanguage.makeUnifiedNet(.unifiedTiny)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::model.") else { return nil }
            return (String(key.dropFirst("w::model.".count)), value)
        }
        try NFKMLXWeights.apply(weights, to: net)

        let ours = net.hiddenStates(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(ours)

        var firstBad: Int?
        var report = [String]()
        for index in 0 ..< ours.count {
            guard let reference = arrays["hidden.\(index)"] else { break }
            let mine = ours[index].reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            guard mine.count == theirs.count else {
                report.append("\(index): shape \(ours[index].shape) vs \(reference.shape)")
                firstBad = firstBad ?? index
                break
            }
            let similarity = cosine(mine, theirs)
            let label = index == 0 ? "embedding" : "after layer \(index - 1)"
            report.append(String(format: "  %-18s cosine %.10f", (label as NSString).utf8String!, similarity))
            if similarity < 0.9999 && firstBad == nil { firstBad = index }
        }
        print("VALIDATION isolation gemma4-unified-tiny:\n" + report.joined(separator: "\n"))
        XCTAssertNil(firstBad, "first divergence at \(firstBad.map { $0 == 0 ? "the embedding" : "layer \($0 - 1)" } ?? "-")")

        let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(logits)
        let mine = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY gemma4-unified-tiny: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the unified decoder's arithmetic matches the reference")
    }

    // MARK: Gemma 4 vision encoder

    // The vision tower's patch embedder and bidirectional transformer, against transformers' own
    // Gemma4VisionModel (its encoder output, before the pooler). The reference feeds flattened patches
    // and their (x, y) grid, so the learned 2-D position embedding and the sandwich blocks are what
    // this measures. The projections load under `.linear` (Gemma4ClippableLinear), and the reference's
    // `encoder.layers.N` names map onto the net's flattened layer array.
    func testGemma4VisionEncoderMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_VISION"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_VISION (run_reference.py gemma4_vision)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let pixelValues = try XCTUnwrap(arrays["pixel_values"])
        let positionIds = try XCTUnwrap(arrays["position_ids"])
        let reference = try XCTUnwrap(arrays["output"])

        let net = NFKMLXGemmaLanguage.makeVisionNet(.tiny)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            let name = String(key.dropFirst(3)).replacingOccurrences(of: "encoder.layers.",
                                                                      with: "encoder_layers.")
            return (name, value)
        }
        try NFKMLXWeights.apply(weights, to: net)

        let patches = pixelValues.shape[0]
        let encoded = net(pixelValues.reshaped([1, patches, pixelValues.shape[1]]),
                          positionIds: positionIds.reshaped([1, patches, 2]).asType(.int32))
        eval(encoded)
        let mine = encoded.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY gemma4-vision-tiny: encoder cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the vision encoder matches the reference")

        // The full tower, through the position-based pooler and the sqrt(hidden) scaling, against the
        // reference's own soft tokens.
        if let pooledReference = arrays["pooled"] {
            let soft = net.softTokens(pixelValues.reshaped([1, patches, pixelValues.shape[1]]),
                                      positionIds: positionIds.reshaped([1, patches, 2]).asType(.int32))
            eval(soft)
            let ours = soft.reshaped([-1]).asArray(Float.self).map(Double.init)
            let refPooled = pooledReference.reshaped([-1]).asArray(Float.self).map(Double.init)
            let pooledSimilarity = cosine(ours, refPooled)
            print("VALIDATION PARITY gemma4-vision-tiny: pooled cosine \(pooledSimilarity)")
            XCTAssertGreaterThan(pooledSimilarity, 0.9999, "the vision pooler matches the reference")
        }
    }

    // The two fixes the released vision weights forced — the 2-D rope and the finite clamp bounds —
    // are each MEASURED to be load-bearing here, not merely asserted. Both were invisible at the tiny
    // random-weight configuration; on the real tower, removing either collapses the encoder cosine.
    func testGemma4VisionRopeAndClampsAreLoadBearingOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_VISION_REAL"], let directory = config["IK_VAL_GEMMA4"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_VISION_REAL and IK_VAL_GEMMA4")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let pixelValues = try XCTUnwrap(arrays["pixel_values"])
        let positionIds = try XCTUnwrap(arrays["position_ids"])
        let reference = try XCTUnwrap(arrays["encoded"])
        let release = URL(fileURLWithPath: directory)
        let patches = pixelValues.shape[0]
        let pixelInput = pixelValues.reshaped([1, patches, pixelValues.shape[1]])
        let positionInput = positionIds.reshaped([1, patches, 2]).asType(.int32)

        func configuration(useClippedLinears: Bool) -> NFKMLXGemma4VisionConfiguration {
            NFKMLXGemma4VisionConfiguration(hiddenSize: 768, layerCount: 16, headCount: 12,
                                            keyValueHeadCount: 12, headDimensions: 64, intermediateSize: 3072,
                                            patchSize: 16, positionEmbeddingSize: 10240, poolingKernelSize: 3,
                                            useClippedLinears: useClippedLinears)
        }
        func load(_ vision: NFKMLXGemma4VisionNet, clamps: Bool) throws {
            try NFKMLXWeights.apply(NFKMLXReleaseWeights.arrays(inDirectory: release) { key in
                guard key.hasPrefix("model.vision_tower.") else { return nil }
                let name = String(key.dropFirst("model.vision_tower.".count))
                    .replacingOccurrences(of: "encoder.layers.", with: "encoder_layers.")
                let isClamp = name.hasSuffix("input_min") || name.hasSuffix("input_max")
                    || name.hasSuffix("output_min") || name.hasSuffix("output_max")
                return (!clamps && isClamp) ? nil : name
            }, to: vision)
        }
        func encoderCosine(_ hidden: MLXArray) -> Double {
            eval(hidden)
            return cosine(hidden.reshaped([-1]).asArray(Float.self).map(Double.init),
                          reference.reshaped([-1]).asArray(Float.self).map(Double.init))
        }

        // Fix 1 removed: run the encoder layers with an identity rope (cos = 1, sin = 0), clamps kept.
        let withoutRope = NFKMLXGemmaLanguage.makeVisionNet(configuration(useClippedLinears: true))
        try load(withoutRope, clamps: true)
        var hidden = withoutRope.patchEmbedder(pixelInput, positionIds: positionInput)
        let identityCosine = MLXArray.ones([1, patches, 64]), identitySine = MLXArray.zeros([1, patches, 64])
        for layer in withoutRope.layers { hidden = layer(hidden, cosine: identityCosine, sine: identitySine) }
        let noRope = encoderCosine(hidden)

        // Fix 2 removed: the release's finite clamps replaced by no clamp at all, rope kept.
        let withoutClamps = NFKMLXGemmaLanguage.makeVisionNet(configuration(useClippedLinears: false))
        try load(withoutClamps, clamps: false)
        let noClamps = encoderCosine(withoutClamps(pixelInput, positionIds: positionInput))

        print("VALIDATION load-bearing gemma4-vision-real: without rope \(noRope), without clamps \(noClamps)")
        // With both, the encoder is 0.9999999999898 (the parity test); dropping either is far from that.
        XCTAssertLessThan(noRope, 0.9, "the 2-D rope is load-bearing on the released weights")
        XCTAssertLessThan(noClamps, 0.99, "the finite clamp bounds are load-bearing on the released weights")
    }

    // MARK: Gemma 4 vision path on the released weights

    // The vision tower, pooler, and multimodal embedder on the RELEASED E2B weights (the release is the
    // full tri-modal Gemma4ForConditionalGeneration, so the vision tower and embedder ship alongside the
    // decoder). Random pixel values over a real grid are fed to both sides, so the tower and the
    // projection into the decoder's space are compared on real weights.
    func testGemma4VisionPathOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_VISION_REAL"], let directory = config["IK_VAL_GEMMA4"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_VISION_REAL and IK_VAL_GEMMA4")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let pixelValues = try XCTUnwrap(arrays["pixel_values"])
        let positionIds = try XCTUnwrap(arrays["position_ids"])
        let pooledReference = try XCTUnwrap(arrays["pooled"])
        let projectedReference = try XCTUnwrap(arrays["output"])
        let release = URL(fileURLWithPath: directory)

        let vision = NFKMLXGemmaLanguage.makeVisionNet(
            NFKMLXGemma4VisionConfiguration(hiddenSize: 768, layerCount: 16, headCount: 12,
                                            keyValueHeadCount: 12, headDimensions: 64, intermediateSize: 3072,
                                            patchSize: 16, positionEmbeddingSize: 10240, poolingKernelSize: 3,
                                            useClippedLinears: true))
        let embedder = NFKMLXGemma4MultimodalEmbedder(multimodalHidden: 768, textHidden: 1536)

        let visionWeights = try NFKMLXReleaseWeights.arrays(inDirectory: release) { key in
            guard key.hasPrefix("model.vision_tower.") else { return nil }
            return String(key.dropFirst("model.vision_tower.".count))
                .replacingOccurrences(of: "encoder.layers.", with: "encoder_layers.")
        }
        try NFKMLXWeights.apply(visionWeights, to: vision)
        let embedWeights = try NFKMLXReleaseWeights.arrays(inDirectory: release) { key in
            key.hasPrefix("model.embed_vision.") ? String(key.dropFirst("model.embed_vision.".count)) : nil
        }
        try NFKMLXWeights.apply(embedWeights, to: embedder)

        let patches = pixelValues.shape[0]
        let pixelInput = pixelValues.reshaped([1, patches, pixelValues.shape[1]])
        let positionInput = positionIds.reshaped([1, patches, 2]).asType(.int32)
        if let encodedReference = arrays["encoded"] {
            let encoded = vision(pixelInput, positionIds: positionInput)
            eval(encoded)
            let encoderSimilarity = cosine(encoded.reshaped([-1]).asArray(Float.self).map(Double.init),
                                           encodedReference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY gemma4-vision-real: encoder \(encoderSimilarity)")
            XCTAssertGreaterThan(encoderSimilarity, 0.9999, "the released vision encoder matches the reference")
        }
        let soft = vision.softTokens(pixelInput, positionIds: positionInput)
        let projected = embedder(soft)
        eval(soft, projected)

        let pooledSimilarity = cosine(soft.reshaped([-1]).asArray(Float.self).map(Double.init),
                                      pooledReference.reshaped([-1]).asArray(Float.self).map(Double.init))
        let projectedSimilarity = cosine(projected.reshaped([-1]).asArray(Float.self).map(Double.init),
                                         projectedReference.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY gemma4-vision-real: pooled \(pooledSimilarity), projected \(projectedSimilarity)")
        XCTAssertGreaterThan(pooledSimilarity, 0.9999, "the released vision tower matches the reference")
        XCTAssertGreaterThan(projectedSimilarity, 0.9999, "the released vision embedder matches the reference")
    }

    // The audio Conformer and its multimodal embedder on the RELEASED E2B weights, fed random mel
    // features. Validates the audio tower and its projection into the decoder's space on real weights.
    func testGemma4AudioPathOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_AUDIO_REAL"], let directory = config["IK_VAL_GEMMA4"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_AUDIO_REAL and IK_VAL_GEMMA4")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let features = try XCTUnwrap(arrays["features"])
        let encodedReference = try XCTUnwrap(arrays["encoded"])
        let projectedReference = try XCTUnwrap(arrays["output"])
        let release = URL(fileURLWithPath: directory)

        let audio = NFKMLXGemmaLanguage.makeAudioNet(
            NFKMLXGemma4AudioConfiguration(hiddenSize: 1024, layerCount: 12, headCount: 8, convKernelSize: 5,
                                           chunkSize: 12, contextLeft: 13, contextRight: 0, rmsEpsilon: 1e-6,
                                           subsampleChannels: [128, 32], outputProjDims: 1536,
                                           useClippedLinears: true))
        let embedder = NFKMLXGemma4MultimodalEmbedder(multimodalHidden: 1536, textHidden: 1536)

        let audioWeights = try NFKMLXReleaseWeights.arrays(inDirectory: release) { key in
            key.hasPrefix("model.audio_tower.") ? String(key.dropFirst("model.audio_tower.".count)) : nil
        }
        try NFKMLXGemmaLanguage.loadAudioWeights(audioWeights, into: audio)
        let embedWeights = try NFKMLXReleaseWeights.arrays(inDirectory: release) { key in
            key.hasPrefix("model.embed_audio.") ? String(key.dropFirst("model.embed_audio.".count)) : nil
        }
        try NFKMLXWeights.apply(embedWeights, to: embedder)

        let encoded = audio(features.reshaped([1, features.shape[0], features.shape[1]]))
        let projected = embedder(encoded)
        eval(encoded, projected)
        let encoderSimilarity = cosine(encoded.reshaped([-1]).asArray(Float.self).map(Double.init),
                                       encodedReference.reshaped([-1]).asArray(Float.self).map(Double.init))
        let projectedSimilarity = cosine(projected.reshaped([-1]).asArray(Float.self).map(Double.init),
                                         projectedReference.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY gemma4-audio-real: encoded \(encoderSimilarity), projected \(projectedSimilarity)")
        XCTAssertGreaterThan(encoderSimilarity, 0.9999, "the released audio tower matches the reference")
        XCTAssertGreaterThan(projectedSimilarity, 0.9999, "the released audio embedder matches the reference")
    }

    // The FULL Gemma4ForConditionalGeneration on the released E2B weights, end to end: an image and a
    // placeholder-carrying prompt through the vision tower, the embedder, the splice, and the decoder.
    // This is the numeric end-to-end run the tiny plumbing test stands in for — on real weights, against
    // transformers' own conditional model.
    func testGemma4ConditionalGenerationOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_CONDITIONAL_REAL"], let directory = config["IK_VAL_GEMMA4"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_CONDITIONAL_REAL and IK_VAL_GEMMA4")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self).map(Int.init)
        let pixelValues = try XCTUnwrap(arrays["pixel_values"])
        let positionIds = try XCTUnwrap(arrays["position_ids"])
        let referenceLogits = try XCTUnwrap(arrays["output"])
        let release = URL(fileURLWithPath: directory)

        // The E-series decoder from the release's text config.
        let decoder = NFKMLXGemmaLanguage.makeNet(
            try NFKMLXGemmaLanguage.configuration(fromHuggingFace: release.appendingPathComponent("config.json")))
        try NFKMLXGemmaLanguage.loadWeights(into: decoder, fromDirectory: release)

        // The vision tower and its embedder from the release.
        let vision = NFKMLXGemmaLanguage.makeVisionNet(
            NFKMLXGemma4VisionConfiguration(hiddenSize: 768, layerCount: 16, headCount: 12,
                                            keyValueHeadCount: 12, headDimensions: 64, intermediateSize: 3072,
                                            patchSize: 16, positionEmbeddingSize: 10240, poolingKernelSize: 3,
                                            useClippedLinears: true))
        let visionEmbedder = NFKMLXGemma4MultimodalEmbedder(multimodalHidden: 768, textHidden: 1536)
        try NFKMLXWeights.apply(NFKMLXReleaseWeights.arrays(inDirectory: release) { key in
            key.hasPrefix("model.vision_tower.")
                ? String(key.dropFirst("model.vision_tower.".count)).replacingOccurrences(of: "encoder.layers.", with: "encoder_layers.")
                : nil
        }, to: vision)
        try NFKMLXWeights.apply(NFKMLXReleaseWeights.arrays(inDirectory: release) { key in
            key.hasPrefix("model.embed_vision.") ? String(key.dropFirst("model.embed_vision.".count)) : nil
        }, to: visionEmbedder)

        let model = NFKMLXGemma4ConditionalGeneration(
            decoder: decoder, visionTower: vision, visionEmbedder: visionEmbedder,
            imageTokenId: 258_880, audioTokenId: 258_881, padTokenId: 0)

        // The four image placeholders read the four soft tokens the vision tower produces.
        let patches = pixelValues.shape[0]
        let soft = visionEmbedder(vision.softTokens(pixelValues.reshaped([1, patches, pixelValues.shape[1]]),
                                                    positionIds: positionIds.reshaped([1, patches, 2]).asType(.int32)))
        let (embeddings, ids) = model.fusedEmbeddings(tokens: tokens,
                                                      visionSoftTokens: soft.reshaped([-1, 1536]),
                                                      audioSoftTokens: nil)
        let logits = decoder.logits(fromEmbeddings: embeddings, tokens: ids)
        eval(logits)

        let mine = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY gemma4-conditional-real: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.999, "the full conditional model matches the reference")

        let vocabulary = referenceLogits.shape[1]
        var agreements = 0
        for position in 0 ..< referenceLogits.shape[0] {
            let base = position * vocabulary
            let ourBest = (0 ..< vocabulary).max { mine[base + $0] < mine[base + $1] }
            let theirBest = (0 ..< vocabulary).max { theirs[base + $0] < theirs[base + $1] }
            if ourBest == theirBest { agreements += 1 }
        }
        print("VALIDATION PARITY gemma4-conditional-real: argmax \(agreements)/\(referenceLogits.shape[0])")
        XCTAssertEqual(agreements, referenceLogits.shape[0], "every position predicts the same token")
    }

    // MARK: Gemma 4 multimodal fusion

    // The multimodal embedder — a scale-free RMS norm and a projection into the language model's space —
    // against transformers' own Gemma4MultimodalEmbedder, plus the splice that fuses soft tokens into a
    // text embedding sequence at the placeholder positions.
    func testGemma4MultimodalFusion() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_EMBEDDER"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_EMBEDDER (run_reference.py gemma4_embedder)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let soft = try XCTUnwrap(arrays["soft"])
        let reference = try XCTUnwrap(arrays["output"])

        let embedder = NFKMLXGemma4MultimodalEmbedder(multimodalHidden: 32, textHidden: 48)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("w::") ? (String(key.dropFirst(3)), value) : nil
        }
        try NFKMLXWeights.apply(weights, to: embedder)
        let projected = embedder(soft)
        eval(projected)
        let mine = projected.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY gemma4-embedder: cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the multimodal embedder matches the reference")

        // The splice: three soft tokens replace the placeholder positions of a five-token sequence.
        let hidden = 4
        let text = MLXArray((0 ..< 5 * hidden).map { Float($0) }).reshaped([1, 5, hidden])
        let softTokens = MLXArray((0 ..< 3 * hidden).map { Float(100 + $0) }).reshaped([3, hidden])
        let fused = NFKMLXGemma4Fusion.fuse(textEmbeddings: text, softTokens: softTokens,
                                            isPlaceholder: [false, true, true, true, false])
        eval(fused)
        let values = fused.reshaped([-1]).asArray(Float.self)
        // Position 0 and 4 keep their text embeddings; 1,2,3 take the three soft tokens in order.
        XCTAssertEqual(values[0], 0)                              // text position 0, channel 0
        XCTAssertEqual(values[1 * hidden], 100)                  // position 1 = soft token 0
        XCTAssertEqual(values[3 * hidden], 108)                  // position 3 = soft token 2
        XCTAssertEqual(values[4 * hidden], 16)                   // text position 4, channel 0
    }

    // The full Gemma4ForConditionalGeneration chain, wired with tiny random-weight towers and decoder.
    // No released tri-modal weights exist locally, so this is a plumbing check: the image flows through
    // the processor, the vision tower, and the embedder into projected soft tokens, those replace the
    // placeholder embeddings, and the prefill-only loop produces text. Every numeric component is at
    // parity in its own test.
    func testGemma4ConditionalGenerationChain() throws {
        try requireMLXRuntime()
        // A vision tower whose pooling is 1, so an 8×8 image's four patches stay four soft tokens.
        let vision = NFKMLXGemmaLanguage.makeVisionNet(
            NFKMLXGemma4VisionConfiguration(hiddenSize: 32, layerCount: 1, headCount: 4, keyValueHeadCount: 4,
                                            headDimensions: 8, intermediateSize: 48, patchSize: 4,
                                            positionEmbeddingSize: 16, poolingKernelSize: 1))
        let decoder = NFKMLXGemmaLanguage.makeNet(
            NFKMLXGemmaConfiguration(hiddenSize: 16, layerCount: 2, intermediateSize: 32, vocabularySize: 200,
                                     headCount: 2, keyValueHeadCount: 1, headDimensions: 8,
                                     globalHeadDimensions: 8, slidingWindow: 16, perLayerInputSize: 8,
                                     sharedKeyValueLayers: 0, finalLogitSoftcap: 0,
                                     layerTypes: [.sliding, .sliding], perLayerVocabularySize: 200))
        let embedder = NFKMLXGemma4MultimodalEmbedder(multimodalHidden: 32, textHidden: 16)
        let imageToken = 100
        let model = NFKMLXGemma4ConditionalGeneration(
            decoder: decoder, visionTower: vision, visionEmbedder: embedder,
            imageProcessor: NFKMLXGemma4ImageProcessor(patchSize: 4, poolingKernelSize: 1, maxSoftTokens: 4),
            imageTokenId: imageToken, audioTokenId: 101, padTokenId: 0)

        var bytes = [UInt8](repeating: 255, count: 8 * 8 * 4)
        for i in stride(from: 0, to: bytes.count, by: 4) { bytes[i] = UInt8((i / 4) % 200) }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = CGImage(width: 8, height: 8, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8 * 4,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!

        let soft = try XCTUnwrap(model.visionSoftTokens(for: image))
        eval(soft)
        XCTAssertEqual(soft.shape, [4, 16])   // four soft tokens projected to the decoder width

        // A prompt with four image placeholders, then two text tokens.
        let prompt = [1, imageToken, imageToken, imageToken, imageToken, 5, 6]
        let (fused, _) = model.fusedEmbeddings(tokens: prompt, visionSoftTokens: soft, audioSoftTokens: nil)
        eval(fused)
        XCTAssertEqual(fused.shape, [1, 7, 16])
        // The placeholder positions carry the soft tokens, not the pad embedding.
        let placeholder = fused[0, 1].asArray(Float.self)
        let text = fused[0, 5].asArray(Float.self)
        XCTAssertNotEqual(placeholder, text)

        let produced = model.generate(promptTokens: prompt, image: image, maxTokens: 3)
        XCTAssertEqual(produced.count, 3)     // the prefill-only loop runs and yields tokens
        print("VALIDATION gemma4-conditional: produced \(produced)")
    }

    // MARK: Gemma 4 audio front end

    // The mel front end — semicausal framing, a Hann window, a magnitude FFT, an HTK triangular mel
    // filterbank, and log — against transformers' own Gemma4AudioFeatureExtractor on a shared waveform.
    // Pure preprocessing, so it matches to floating precision.
    func testGemma4MelFrontEndMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_MEL"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_MEL (run_reference.py gemma4_mel)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let waveform = try XCTUnwrap(arrays["waveform"]).asArray(Float.self)
        let reference = try XCTUnwrap(arrays["output"])

        let features = NFKMLXGemma4AudioFeatureExtractor().features(waveform)
        eval(features)
        XCTAssertEqual(features.shape, [reference.shape[0], reference.shape[1]])
        let mine = features.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY gemma4-mel: cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the mel front end matches the reference")
    }

    // MARK: Gemma 4 image processor

    // The image processor's structure — the aspect-ratio-preserving resize dimensions, the patch
    // flattening order, and the (x, y) position ids — against the reference algorithm. The resize
    // pixels are a documented CoreGraphics approximation, so this checks the layout, which is exact.
    func testGemma4ImageProcessorLayout() throws {
        try requireMLXRuntime()
        // Resized dimensions, largest within the 2520-patch budget and divisible by 48.
        let processor = NFKMLXGemma4ImageProcessor()
        let size = processor.resizedSize(width: 200, height: 100)
        XCTAssertEqual(size.height, 528)
        XCTAssertEqual(size.width, 1104)

        // A synthetic gradient at a patch-aligned size, split by a 1×1-pooling processor so there is no
        // padding: R encodes x, G encodes y. Patch (row, col) covers pixels [col·16, row·16].
        let side = 48
        var bytes = [UInt8](repeating: 255, count: side * side * 4)
        for y in 0 ..< side {
            for x in 0 ..< side {
                let base = (y * side + x) * 4
                bytes[base] = UInt8(x); bytes[base + 1] = UInt8(y); bytes[base + 2] = 128
            }
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                            bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let tiles = NFKMLXGemma4ImageProcessor(patchSize: 16, poolingKernelSize: 1, maxSoftTokens: 9)
        let (pixelValues, positionIds) = tiles.process(image)
        XCTAssertEqual(pixelValues.shape, [9, 16 * 16 * 3])
        eval(pixelValues, positionIds)

        // Position ids run row-major as (x, y) = (column, row).
        let positions = positionIds.reshaped([-1]).asArray(Int32.self)
        let expected: [Int32] = [0,0, 1,0, 2,0, 0,1, 1,1, 2,1, 0,2, 1,2, 2,2]
        XCTAssertEqual(Array(positions), expected)

        // Patch (row 1, col 2) is index 1·3 + 2 = 5; its first pixel is the image pixel at (32, 16),
        // so R ≈ 32/255 and G ≈ 16/255 (within CoreGraphics rounding of the identity resize).
        let values = pixelValues.reshaped([-1]).asArray(Float.self)
        let firstPixel = 5 * (16 * 16 * 3)
        XCTAssertEqual(Double(values[firstPixel]), 32.0 / 255, accuracy: 0.02)
        XCTAssertEqual(Double(values[firstPixel + 1]), 16.0 / 255, accuracy: 0.02)
    }

    // MARK: Gemma 4 audio Conformer

    // The audio subsampler — two stride-2 convolutions, a channel LayerNorm, a ReLU, and a linear
    // projection — against transformers' own Gemma4AudioModel subsampler. The convolution kernels load
    // in PyTorch layout and are transposed to MLX's.
    func testGemma4AudioSubSampleMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_AUDIO"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_AUDIO (run_reference.py gemma4_audio)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let features = try XCTUnwrap(arrays["input_features"])
        let reference = try XCTUnwrap(arrays["subsampled"])

        let subsample = NFKMLXGemmaLanguage.makeAudioSubSample(.tiny)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::subsample_conv_projection.") else { return nil }
            return (String(key.dropFirst("w::subsample_conv_projection.".count)), value)
        }
        try NFKMLXGemmaLanguage.loadAudioWeights(weights, into: subsample)

        let out = subsample(features.reshaped([1, features.shape[0], features.shape[1]]))
        eval(out)
        let mine = out.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY gemma4-audio-subsample: cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the audio subsampler matches the reference")
    }

    // The Conformer layers and output projection — the blocked relative-position attention, the light
    // convolution, the macaron feed-forwards — against transformers' own Gemma4AudioModel. The
    // post-subsample hidden state, the position encoding, and the blocked mask come from the reference,
    // so this isolates the Conformer's arithmetic (the hardest part: block conversion, the rel-shift,
    // the softcap) from the subsampler and the mask construction.
    func testGemma4AudioConformerMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GEMMA4_AUDIO"] else {
            throw XCTSkip("set IK_PARITY_GEMMA4_AUDIO (run_reference.py gemma4_audio)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let subsampled = try XCTUnwrap(arrays["subsampled"])
        let positionEmbeddings = try XCTUnwrap(arrays["position_embeddings"])
        let mask = try XCTUnwrap(arrays["attention_mask"])
        let reference = try XCTUnwrap(arrays["output"])

        let net = NFKMLXGemmaLanguage.makeAudioNet(.tiny)
        let weights = arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("w::") else { return nil }
            return (String(key.dropFirst(3)), value)
        }
        try NFKMLXGemmaLanguage.loadAudioWeights(weights, into: net)

        let hidden = subsampled.reshaped([1, subsampled.shape[0], subsampled.shape[1]])
        let maskF = mask.asType(.float32)

        // Isolation: compare each sub-component of layer 0 to the reference's captured seam.
        func score(_ a: MLXArray, _ key: String) {
            guard let ref = arrays[key] else { return }
            eval(a)
            let m = a.reshaped([-1]).asArray(Float.self).map(Double.init)
            let t = ref.reshaped([-1]).asArray(Float.self).map(Double.init)
            print("VALIDATION seam \(key): cosine \(cosine(m, t))")
        }
        let layer0 = net.layers[0]
        let ff1 = layer0.feedForward1(hidden)
        score(ff1, "seam_ff1")
        let attn = layer0.attention(layer0.normPreAttention(ff1), positionEmbeddings: positionEmbeddings, mask: maskF)
        score(attn, "seam_attn")

        let output = net.conformer(hidden, positionEmbeddings: positionEmbeddings, mask: maskF)
        eval(output)
        let mine = output.reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY gemma4-audio-conformer: cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the audio Conformer matches the reference")

        // The full tower from the mel features, building its own sliding-window mask rather than being
        // handed one — the end-to-end path.
        if let features = arrays["input_features"] {
            let full = net(features.reshaped([1, features.shape[0], features.shape[1]]))
            eval(full)
            let fullMine = full.reshaped([-1]).asArray(Float.self).map(Double.init)
            let fullSimilarity = cosine(fullMine, theirs)
            print("VALIDATION PARITY gemma4-audio-full: cosine \(fullSimilarity)")
            XCTAssertGreaterThan(fullSimilarity, 0.9999, "the full audio tower matches the reference")
        }
    }

    // MARK: GGUF language model, end to end

    // A GGUF release loaded through the native reader and run through the decoder, against transformers
    // loading the SAME GGUF. This is the whole chain: metadata → configuration, the llama.cpp tensor
    // names remapped onto the module's keys, and the reader's dequantization feeding the forward. The
    // reference feeds its own token ids (a quantized model still has one correct answer), so the logits
    // are compared directly and the greedy continuation token for token; a separate check confirms the
    // rebuilt tokenizer reproduces the reference's ids from the shared text.
    func testGGUFLanguageModelMatchesTheReferenceEndToEnd() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_GGUF_LM"], let ggufPath = config["IK_VAL_GGUF"] else {
            throw XCTSkip("set IK_PARITY_GGUF_LM (run_reference.py gguf_lm) and IK_VAL_GGUF")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])
        let referenceContinuation = try XCTUnwrap(arrays["continuation"]).asArray(Int32.self)

        let (net, tokenizer) = try NFKMLXLanguage.loadedGGUF(at: URL(fileURLWithPath: ggufPath))

        // The whole-prompt logits: a quantized model has one correct dequantization, so the two agree
        // closely rather than exactly (the reference and this port each run their own kernels).
        let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, referenceLogits.shape[0], referenceLogits.shape[1]])
        let mine = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(mine, theirs)
        print("VALIDATION PARITY gguf-lm smollm2-135m: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.999, "the GGUF decoder matches transformers loading the same file")

        let vocabulary = referenceLogits.shape[1]
        for position in 0 ..< referenceLogits.shape[0] {
            let base = position * vocabulary
            let ourBest = (0 ..< vocabulary).max { mine[base + $0] < mine[base + $1] }
            let theirBest = (0 ..< vocabulary).max { theirs[base + $0] < theirs[base + $1] }
            XCTAssertEqual(ourBest, theirBest, "position \(position) predicts the same token")
        }

        // Teacher-forced on the reference's own continuation, this port predicts the reference's next
        // token at every step. Feeding the reference sequence keeps the context identical at each
        // position, so this isolates the forward from the compounding a free run would add. The first
        // sampled token from a free run is checked too — the end-to-end greedy step.
        let forced = tokens.map(Int.init) + referenceContinuation.dropLast().map(Int.init)
        let forcedLogits = net(MLXArray(forced).reshaped([1, forced.count]))
        eval(forcedLogits)
        let forcedFlat = forcedLogits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        var agreements = 0
        for step in 0 ..< referenceContinuation.count {
            let position = tokens.count - 1 + step
            let base = position * vocabulary
            let best = (0 ..< vocabulary).max { forcedFlat[base + $0] < forcedFlat[base + $1] }
            if best == Int(referenceContinuation[step]) { agreements += 1 }
        }
        print("VALIDATION gguf-lm teacher-forced continuation: \(agreements)/\(referenceContinuation.count) agree")
        // Quantized weights and two dequantization implementations flip an occasional near-tie, so this
        // is a strong-majority check rather than an exact one; the logit cosine above is the tight bound.
        XCTAssertGreaterThanOrEqual(agreements, referenceContinuation.count - 2,
                                    "teacher-forced, the port predicts the reference's continuation")

        var options = NFKMLXGenerationOptions()
        options.maxTokens = 1
        let firstToken = net.generate(prompt: tokens.map(Int.init), options: options)
        XCTAssertEqual(firstToken.first, Int(referenceContinuation[0]), "the first greedy token matches")

        // The rebuilt tokenizer reproduces the reference's ids from the shared text (the reference
        // prepends its bos token, which the encoder does not add).
        let encoded = try XCTUnwrap(tokenizer).encode("The capital of France is").map(\.intValue)
        XCTAssertEqual(encoded, Array(tokens.dropFirst().map(Int.init)), "the GGUF tokenizer matches the reference")
    }

    // Inside layer 0, sub-step by sub-step. The layer-level harness said layer 0 diverges while its
    // input is exact; this says which part of it.
    func testGemma4Layer0SubSteps() throws {
        try requireMLXRuntime()
        guard let probePath = config["IK_PROBE_GEMMA4_LAYER0"],
              let directory = config["IK_VAL_GEMMA4"] else { throw XCTSkip("set IK_PROBE_GEMMA4_LAYER0") }
        let probe = try loadArrays(url: URL(fileURLWithPath: probePath))
        let tokens = try XCTUnwrap(probe["tokens"]).asArray(Int32.self)

        let release = URL(fileURLWithPath: directory)
        let geometry = try NFKMLXGemmaLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        let net = NFKMLXGemmaLanguage.makeNet(geometry)
        try NFKMLXGemmaLanguage.loadWeights(into: net, fromDirectory: release)

        let hiddenIn = try XCTUnwrap(probe["hidden_in"]).reshaped([1, tokens.count, geometry.hiddenSize])
        let layer = net.layers[0]

        func compare(_ label: String, _ mine: MLXArray, _ key: String) {
            guard let reference = probe[key] else { return }
            eval(mine)
            let a = mine.reshaped([-1]).asArray(Float.self).map(Double.init)
            let b = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            guard a.count == b.count else {
                print(String(format: "  %-16s SHAPE %@ vs %@", (label as NSString).utf8String!,
                             "\(mine.shape)", "\(reference.shape)"))
                return
            }
            print(String(format: "  %-16s cosine %.10f", (label as NSString).utf8String!, cosine(a, b)))
        }

        let normed = layer.inputNorm(hiddenIn)
        compare("input_norm", normed, "input_norm")

        let (attended, _, _) = layer.attention(normed, mask: nil, shared: nil)
        compare("attn_out", attended, "attn_out")

        let postAttn = layer.postAttentionNorm(attended)
        compare("post_attn_norm", postAttn, "post_attn_norm")

        let residual = hiddenIn + postAttn
        let mlp = layer.feedForward(layer.preFeedForwardNorm(residual))
        compare("mlp_out", mlp, "mlp_out")

        let lifted = residual + layer.postFeedForwardNorm(mlp)
        compare("ple_gate", layer.perLayerGate(lifted), "ple_gate")
        print("VALIDATION isolation gemma4-layer0: sub-steps above")
    }

    // MARK: Qwen3.5 hybrid

    // The gated delta-rule recurrence, layer by layer, against transformers' own implementation.
    // Qwen3.5-4B is the smallest release of the family Qwen3.5 / 3.6 / 3.8 share, and the only one
    // that fits here — measuring it is what takes the hybrid decoder from built to verified.
    func testQwen3_5LayerByLayerAgainstTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_QWEN3_5"], let directory = config["IK_VAL_QWEN3_5"] else {
            throw XCTSkip("set IK_PARITY_QWEN3_5 and IK_VAL_QWEN3_5")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)

        let release = URL(fileURLWithPath: directory)
        let geometry = try NFKMLXHybridLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        let net = NFKMLXHybridLanguage.makeNet(geometry)
        try NFKMLXHybridLanguage.loadWeights(into: net, fromDirectory: release)

        let ours = net.hiddenStates(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(ours)

        var firstBad: Int?
        var report = [String]()
        for index in 0 ..< ours.count {
            guard let reference = arrays["hidden.\(index)"] else { break }
            let mine = ours[index].reshaped([-1]).asArray(Float.self).map(Double.init)
            let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            guard mine.count == theirs.count else {
                report.append("  \(index): shape \(ours[index].shape) vs \(reference.shape)")
                firstBad = firstBad ?? index
                break
            }
            let similarity = cosine(mine, theirs)
            let kind = index == 0 ? "embedding"
                : "\(geometry.layerTypes[index - 1] == .fullAttention ? "full  " : "linear") \(index - 1)"
            report.append(String(format: "  %-12s cosine %.10f", (kind as NSString).utf8String!, similarity))
            if similarity < 0.9999 && firstBad == nil { firstBad = index }
        }
        print("VALIDATION isolation qwen3.5:\n" + report.joined(separator: "\n"))
        if let firstBad {
            print("VALIDATION isolation qwen3.5: FIRST DIVERGENCE at index \(firstBad)")
        } else {
            print("VALIDATION isolation qwen3.5: every layer matches")
        }
    }

    // Inside the first gated delta-rule layer, sub-step by sub-step.
    func testQwen3_5Layer0SubSteps() throws {
        try requireMLXRuntime()
        guard let probePath = config["IK_PROBE_QWEN3_5_LAYER0"],
              let directory = config["IK_VAL_QWEN3_5"] else { throw XCTSkip("set IK_PROBE_QWEN3_5_LAYER0") }
        let probe = try loadArrays(url: URL(fileURLWithPath: probePath))
        let tokens = try XCTUnwrap(probe["tokens"]).asArray(Int32.self)

        let release = URL(fileURLWithPath: directory)
        let geometry = try NFKMLXHybridLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        let net = NFKMLXHybridLanguage.makeNet(geometry)
        try NFKMLXHybridLanguage.loadWeights(into: net, fromDirectory: release)

        let hiddenIn = try XCTUnwrap(probe["hidden_in"])
            .reshaped([1, tokens.count, geometry.hiddenSize])
        let block = net.model.layers[0]
        let linear = try XCTUnwrap(block.linearAttention)

        func compare(_ label: String, _ mine: MLXArray, _ key: String) {
            guard let reference = probe[key] else { print("  \(label): (not captured)"); return }
            eval(mine)
            let a = mine.reshaped([-1]).asArray(Float.self).map(Double.init)
            let b = reference.reshaped([-1]).asArray(Float.self).map(Double.init)
            guard a.count == b.count else {
                print("  \(label): SHAPE \(mine.shape) vs \(reference.shape)"); return
            }
            print(String(format: "  %-16s cosine %.10f", (label as NSString).utf8String!, cosine(a, b)))
        }

        let normed = block.inputNorm(hiddenIn)
        compare("input_norm", normed, "input_norm")
        compare("qkv", linear.qkvProjection(normed), "qkv")
        compare("z", linear.gateProjection(normed), "z")
        compare("a", linear.decayProjection(normed), "a")
        compare("b", linear.writeProjection(normed), "b")
        compare("linear_attn_out", linear(normed), "linear_attn_out")
        print("VALIDATION isolation qwen3.5-layer0: sub-steps above")
    }

    // The hybrid decoder end to end, and its greedy continuation. The layer harness proves each stage;
    // this proves the logits and the tied output projection on top of them.
    func testQwen3_5MatchesTheReferenceLogits() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_QWEN3_5"], let directory = config["IK_VAL_QWEN3_5"] else {
            throw XCTSkip("set IK_PARITY_QWEN3_5 and IK_VAL_QWEN3_5")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self)
        let referenceLogits = try XCTUnwrap(arrays["output"])

        let release = URL(fileURLWithPath: directory)
        let geometry = try NFKMLXHybridLanguage.configuration(
            fromHuggingFace: release.appendingPathComponent("config.json"))
        let net = NFKMLXHybridLanguage.makeNet(geometry)
        try NFKMLXHybridLanguage.loadWeights(into: net, fromDirectory: release)

        let logits = net(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(logits)
        let ours = logits[0].reshaped([-1]).asArray(Float.self).map(Double.init)
        let theirs = referenceLogits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, theirs)
        print("VALIDATION PARITY qwen3.5-4b: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the hybrid decoder matches the reference")

        let vocabulary = referenceLogits.shape[1]
        for position in 0 ..< referenceLogits.shape[0] {
            let base = position * vocabulary
            let ourBest = (0 ..< vocabulary).max { ours[base + $0] < ours[base + $1] }
            let theirBest = (0 ..< vocabulary).max { theirs[base + $0] < theirs[base + $1] }
            XCTAssertEqual(ourBest, theirBest, "position \(position) predicts the same token")
        }
    }

    // MARK: SAM

    // The encoder's neck output is the seam between SAM's ViT and its mask decoder. The reference record
    // uses a 1024×1024 plate, so `ResizeLongestSide` and the square padding are identities on both sides
    // and a mismatch here localizes to the ViT itself.
    func testSAMEncoderMatchesTheReference() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_SAM_ENCODER")
        let net = NFKMLXSAM.makeNet(.vitB)
        try NFKMLXSAM.loadWeights(into: net, from: weights("IK_VAL_SAM"))

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let mean = MLXArray([Float(123.675), 116.28, 103.53])
        let standardDeviation = MLXArray([Float(58.395), 57.12, 57.375])
        let prepared = (inputTensor.reshaped([1, height, width, 3]) * 255 - mean) / standardDeviation
        let features = net.imageEncoder(prepared)               // [1, 64, 64, 256]
        eval(features)

        let ours = features.reshaped([64 * 64 * 256]).asArray(Float.self).map(Double.init)
        let reference = referenceOutput.reshaped([64 * 64 * 256]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, reference)
        print("VALIDATION PARITY sam-encoder: cosine \(similarity) over \(ours.count) values")
        XCTAssertGreaterThan(similarity, 0.99, "the ViT image encoder matches the reference")
    }

    // The full pipeline: encoder → prompt encoder → mask decoder, against the official predictor's mask
    // probabilities for the same plate and the same click point (w/2, h/3).
    func testSAMMaskMatchesTheReference() throws {
        try requireMLXRuntime()
        let (inputTensor, referenceOutput) = try record("IK_PARITY_SAM_MASK")
        let net = NFKMLXSAM.makeNet(.vitB)
        try NFKMLXSAM.loadWeights(into: net, from: weights("IK_VAL_SAM"))

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let mean = MLXArray([Float(123.675), 116.28, 103.53])
        let standardDeviation = MLXArray([Float(58.395), 57.12, 57.375])
        let prepared = (inputTensor.reshaped([1, height, width, 3]) * 255 - mean) / standardDeviation
        let mask = net.segment(prepared, pointX: 0.5, pointY: 1.0 / 3.0)   // [1, m, m, 1] probabilities
        eval(mask)

        // The reference mask is full resolution; ours is the decoder's native grid. Compare at ours.
        let side = mask.shape[1]
        let reference = NFKMLXResample.resizeBilinear(
            referenceOutput.reshaped([1, height, width, 1]), height: side, width: side)
        let ours = mask.reshaped([side * side]).asArray(Float.self).map(Double.init)
        let referenceValues = reference.reshaped([side * side]).asArray(Float.self).map(Double.init)
        let similarity = cosine(ours, referenceValues)
        let agreement = zip(ours, referenceValues).filter { ($0 > 0.5) == ($1 > 0.5) }.count
        let agreementFraction = Double(agreement) / Double(ours.count)
        print("VALIDATION PARITY sam-mask: cosine \(similarity), binary agreement \(agreementFraction)")
        XCTAssertGreaterThan(agreementFraction, 0.95, "the predicted mask matches the reference predictor")
    }

    // Diagnostic, not an assertion: prints the pairwise cosine matrix between our four raw mask-token
    // logits and the reference's, plus both IoU vectors. A permutation shows up off-diagonal, an
    // inversion as a negative cosine, a genuine decoder divergence as small values everywhere.
    func testSAMDecoderTokenMatrixAgainstTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SAM_DECODER"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_SAM_DECODER to a sam_decoder record")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let inputTensor = try XCTUnwrap(arrays["input_image"])
        let referenceMasks = try XCTUnwrap(arrays["output"])    // [4, 256, 256] logits
        let referenceScores = try XCTUnwrap(arrays["scores"])   // [4]

        let net = NFKMLXSAM.makeNet(.vitB)
        try NFKMLXSAM.loadWeights(into: net, from: weights("IK_VAL_SAM"))

        let (height, width) = (inputTensor.shape[0], inputTensor.shape[1])
        let mean = MLXArray([Float(123.675), 116.28, 103.53])
        let standardDeviation = MLXArray([Float(58.395), 57.12, 57.375])
        let prepared = (inputTensor.reshaped([1, height, width, 3]) * 255 - mean) / standardDeviation

        let embedding = net.imageEncoder(prepared)
        let grid = embedding.shape[1]
        let imagePE = net.promptEncoder.positionEncoding.grid(grid, grid)
        let sparse = net.promptEncoder.sparse(pointX: 0.5, pointY: 1.0 / 3.0, positive: true)
        let dense = net.promptEncoder.dense(grid: grid)
        let (masks, iou) = net.maskDecoder(embedding + dense, imagePE: imagePE, sparse: sparse)
        eval(masks, iou)

        let side = masks.shape[1]
        print("PARITY sam-decoder: ours [\(masks.shape)] iou \(iou.asArray(Float.self)); "
              + "reference scores \(referenceScores.asArray(Float.self))")
        let referenceSized = referenceMasks.shape[1] == side ? referenceMasks
            : NFKMLXResample.resizeBilinear(referenceMasks.reshaped([4, referenceMasks.shape[1], referenceMasks.shape[2], 1]),
                                            height: side, width: side).reshaped([4, side, side])
        for ourToken in 0 ..< 4 {
            var row = [String]()
            let a = masks[0..., 0..., 0..., ourToken ..< (ourToken + 1)]
                .reshaped([side * side]).asArray(Float.self).map(Double.init)
            for referenceToken in 0 ..< 4 {
                let b = referenceSized[referenceToken].reshaped([side * side]).asArray(Float.self).map(Double.init)
                row.append(String(format: "%+.3f", cosine(a, b)))
            }
            print("PARITY sam-decoder token \(ourToken) vs reference: [\(row.joined(separator: ", "))]")
        }
    }

    // MARK: - Zero-DCE training losses

    /// The four zero-reference losses against the reference implementation.
    ///
    /// The record carries the enhanced image and the curve maps the reference scored, so both sides
    /// score identical tensors and a mismatch can only come from the loss arithmetic. Fine-tuning is
    /// the one path where a wrong loss is invisible in the output: a model trained against a subtly
    /// different objective still produces plausible images, just not the ones the method describes.
    ///
    /// `python run_reference.py zero_dce_losses out/zero-dce-losses.safetensors --size 64`
    func testZeroDCETrainingLossesMatchTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_ZERO_DCE_LOSSES"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_ZERO_DCE_LOSSES to a record from run_reference.py zero_dce_losses")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        guard let original = arrays["input_image"], let enhanced = arrays["enhanced"],
              let curveMaps = arrays["curve_maps"], let reference = arrays["output"] else {
            throw NFKMLXError.noOutput
        }

        let batchedOriginal = original.expandedDimensions(axis: 0)
        let batchedEnhanced = enhanced.expandedDimensions(axis: 0)
        let ours = [
            NFKMLXZeroDCELoss.exposure(batchedEnhanced),
            NFKMLXZeroDCELoss.colorConstancy(batchedEnhanced),
            NFKMLXZeroDCELoss.illuminationSmoothness(curveMaps.expandedDimensions(axis: 0)),
            NFKMLXZeroDCELoss.spatialConsistency(batchedEnhanced, original: batchedOriginal),
        ].map { $0.item(Float.self) }

        let expected = reference.asArray(Float.self)
        let names = ["exposure", "color-constancy", "illumination-smoothness", "spatial-consistency"]
        for (index, name) in names.enumerated() {
            print("PARITY zero-dce \(name): ours \(ours[index]), reference \(expected[index])")
            // Relative tolerance: the losses span four orders of magnitude, so one absolute bound
            // would be vacuous for the largest and unmeetable for the smallest.
            let tolerance = max(abs(expected[index]) * 1e-4, 1e-7)
            XCTAssertEqual(ours[index], expected[index], accuracy: tolerance, name)
        }
    }

    /// SegFormer's segmentation loss against the reference.
    ///
    /// The record carries the logits and labels the reference scored, so the network is factored out
    /// and a mismatch can only be the upsample or the cross-entropy. The upsample is the part worth
    /// pinning: downsampling the labels instead would be cheaper, would still train, and would quietly
    /// discard the thin structures segmentation is judged on.
    ///
    /// `python run_reference.py segformer_loss out/segformer-loss.safetensors --size 64`
    func testSegFormerTrainingLossMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_SEGFORMER_LOSS"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_SEGFORMER_LOSS to a record from run_reference.py segformer_loss")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        guard let logits = arrays["logits"], let labels = arrays["labels"],
              let reference = arrays["output"] else {
            throw NFKMLXError.noOutput
        }

        let ours = NFKMLXSegFormerObjective()
            .loss(logits: logits.expandedDimensions(axis: 0), labels: labels)
            .item(Float.self)
        let expected = reference.item(Float.self)
        print("PARITY segformer-loss: ours \(ours), reference \(expected)")
        XCTAssertEqual(ours, expected, accuracy: max(abs(expected) * 1e-4, 1e-7))
    }

    // MARK: Stable Diffusion text conditioning

    /// The prompts `run_reference.py`'s `sd_tokenizer` and `sd_text_encoder` modes are measured on.
    /// The Python side declares the same list; a change to one without the other shows up as a
    /// mismatch rather than passing quietly.
    private static let sdPrompts = [
        "a photograph of an astronaut riding a horse",
        "A PHOTO, of  a   cat!!! (highly detailed), 8k",
        "",
        "  spaced   out  text  ",
        "2024 was a year; na\u{ef}ve caf\u{e9} \u{2014} r\u{e9}sum\u{e9}",
    ]

    // The tokenizer is its own seam. A prompt that tokenizes differently reaches the model as a
    // different sentence, and comparing embeddings alone cannot tell that from a wrong weight.
    func testCLIPTokenizerMatchesTheReferenceIds() throws {
        guard let path = config["IK_PARITY_SD_TOKENIZER"] else { throw XCTSkip("set IK_PARITY_SD_TOKENIZER") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let markers = try XCTUnwrap(arrays["markers"]).asArray(Int32.self).map(Int.init)
        let padded = try XCTUnwrap(arrays["output"])             // [prompts, 77]

        let tokenizer = try NFKMLXSDPromptTokenizer(directoryURL: try weights("IK_VAL_SD15_TOKENIZER"))
        XCTAssertEqual([tokenizer.startTokenId, tokenizer.endTokenId, tokenizer.paddingTokenId], markers,
                       "the markers the release's own files name")

        for (index, prompt) in Self.sdPrompts.enumerated() {
            let reference = try XCTUnwrap(arrays["prompt_\(index)"]).asArray(Int32.self).map(Int.init)
            XCTAssertEqual(tokenizer.encode(prompt), reference,
                           "prompt \(index) tokenizes as the reference does")
            XCTAssertEqual(tokenizer.tokens(for: prompt, contextLength: 77),
                           padded[index].asArray(Int32.self).map(Int.init),
                           "and the padded model input matches")
        }
    }

    // The text encoder, against `transformers`' own CLIPTextModel on the released weights, driven with
    // the reference's own token ids. Both hidden states are compared: the releases read different ones
    // (SD 1.x and 2.x the last, SDXL the penultimate), so checking only one leaves the other unmeasured.
    func testSD15TextEncoderMatchesTheReference() throws {
        try verifyTextEncoder("sd15-text", .stableDiffusion15, record: "IK_PARITY_SD15_TEXT",
                              weights: "IK_VAL_SD15_TEXT")
    }

    // The whole text-to-image path: prompt, tokenizer, text encoder, UNet, DDIM sampler, autoencoder.
    // Each piece is measured on its own elsewhere; this is what says they compose into the reference's
    // picture. The run starts from the reference's own initial latent — matching a random source across
    // two implementations proves nothing about either.
    func testSD15TextToImageMatchesTheReference() throws {
        try verifyTextToImage("sd15", .stableDiffusion15, record: "IK_PARITY_SD15_T2I",
                              directory: "IK_VAL_SD15_DIR")
    }

    // SDXL-Turbo: two text towers concatenated, a pooled embedding and a size descriptor folded into
    // the timestep, a schedule counted down from the end of the training range, and one step with no
    // classifier-free guidance.
    func testSDXLTurboTextToImageMatchesTheReference() throws {
        try verifyTextToImage("sdxl-turbo", .sdxlTurbo, record: "IK_PARITY_SDXL_T2I",
                              directory: "IK_VAL_SDXL_DIR")
    }

    // The same release with classifier-free guidance on, which is a different path: the conditional and
    // unconditional predictions run as one batch of two, and an empty negative prompt conditions on
    // zeros rather than on the embedding of an empty sentence.
    func testSDXLTurboGuidedTextToImageMatchesTheReference() throws {
        try verifyTextToImage("sdxl-turbo-guided", .sdxlTurbo, record: "IK_PARITY_SDXL_T2I_GUIDED",
                              directory: "IK_VAL_SDXL_DIR")
    }

    // And with no negative prompt at all, which is what `force_zeros_for_empty_prompt` acts on: SDXL
    // then conditions on zeros rather than on the embedding of an empty sentence.
    func testSDXLTurboWithoutANegativePromptMatchesTheReference() throws {
        try verifyTextToImage("sdxl-turbo-zero-negative", .sdxlTurbo,
                              record: "IK_PARITY_SDXL_T2I_ZERONEG", directory: "IK_VAL_SDXL_DIR")
    }

    private func verifyTextToImage(_ name: String, _ configuration: NFKMLXSDTextToImageConfiguration,
                                   record: String, directory key: String) throws {
        try requireMLXRuntime()
        guard let path = config[record] else { throw XCTSkip("set \(record)") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let latent = try XCTUnwrap(arrays["latent"])             // [side, side, 4]
        let settings = try XCTUnwrap(arrays["settings"]).asArray(Float.self)
        let reference = try XCTUnwrap(arrays["output"])          // [height, width, 3] in 0...1
        let (steps, side, guidance) = (Int(settings[0]), Int(settings[1]), settings[2])

        let model = try NFKMLXSDTextToImageModel(
            configuration: configuration,
            files: NFKMLXSDReleaseFiles(directoryURL: try weights(key)))
        model.startLatent = latent

        // The reference distinguishes an absent negative prompt from an empty one, and the record says
        // which it ran with.
        var inputs: [String: Any] = [NFKInputPrompt: Self.sdPrompts[0]]
        if settings.count < 4 || settings[3] == 0 {
            inputs[NFKInputNegativePrompt] = ""
        }
        let request = NFKInferenceRequest(inputs: inputs,
                                          parameters: [NFKParameterSteps: steps,
                                                       NFKParameterGuidanceScale: guidance,
                                                       NFKParameterWidth: side * 8,
                                                       NFKParameterHeight: side * 8])
        if let referenceContext = arrays["context"] {
            let context = try model.encode(request: request, image: nil)
            // The record stores zeros for the unconditional embedding when the reference ran without
            // classifier-free guidance and never built one; comparing against that measures nothing.
            let unconditional = arrays["uncontext"].flatMap {
                $0.abs().max().item(Float.self) > 0 ? $0 : nil
            }
            for (key, expected) in [("context", referenceContext), ("uncontext", unconditional)].compactMap({ pair in
                pair.1.map { (pair.0, $0) } }) {
                let mine = try XCTUnwrap(context.conditioning[key])
                let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                        expected.reshaped([-1]).asArray(Float.self).map(Double.init))
                print("VALIDATION PARITY \(name)-text-to-image: \(key) cosine \(similarity)")
            }
            if let expected = arrays["first_prediction"] {
                let schedule = NFKDDIMScheduler(predictionType: configuration.predictionType,
                                                spacing: configuration.timestepSpacing).steps(steps)
                let mine = model.denoise(latent, schedule[0], context, guidance)
                eval(mine)
                let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                        expected.reshaped([-1]).asArray(Float.self).map(Double.init))
                print("VALIDATION PARITY \(name)-text-to-image: first prediction cosine \(similarity)")
            }
        }

        if let trace = arrays["latents"] {
            // Per-step latents isolate a whole-picture mismatch to the step that first diverged.
            let scheduler = NFKDDIMScheduler(predictionType: configuration.predictionType,
                                             spacing: configuration.timestepSpacing)
            let schedule = scheduler.steps(steps)
            var current = latent
            let context = try model.encode(request: request, image: nil)
            for (index, step) in schedule.enumerated() {
                current = scheduler.step(prediction: model.denoise(current, step, context, guidance),
                                         timestep: step, latent: current)
                eval(current)
                let similarity = cosine(current.reshaped([-1]).asArray(Float.self).map(Double.init),
                                        trace[index].reshaped([-1]).asArray(Float.self).map(Double.init))
                print("VALIDATION PARITY \(name)-text-to-image: latent after step \(index) (training step \(step.train)) cosine \(similarity)")
            }
        }

        let result = try model.makeBackend().runInference(for: request)
        let value = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        let ours = rgbValues(value as! CGImage)
        let theirs = reference.reshaped([-1]).asArray(Float.self).map(Double.init)

        XCTAssertEqual(ours.count, theirs.count, "the reference's picture size")
        let similarity = cosine(ours, theirs)
        let difference = zip(ours, theirs).map { abs($0 - $1) }.reduce(0, +) / Double(ours.count)
        print("VALIDATION PARITY \(name)-text-to-image: image cosine \(similarity), mean |difference| \(difference)")
        XCTAssertGreaterThan(similarity, 0.99, "the generated picture matches the reference pipeline")
    }

    // Stable Diffusion 2.x conditions on the OpenCLIP tower: wider, deeper, and activating with the
    // exact GELU rather than the sigmoid approximation SD 1.x uses.
    func testSD21TextEncoderMatchesTheReference() throws {
        try verifyTextEncoder("sd21-text", .stableDiffusion21, record: "IK_PARITY_SD21_TEXT",
                              weights: "IK_VAL_SD21_TEXT")
    }

    // The same chain on a release whose model predicts the velocity rather than the noise, which is a
    // different recovery inside the sampler.
    func testSD21TextToImageMatchesTheReference() throws {
        try verifyTextToImage("sd21", .stableDiffusion21V, record: "IK_PARITY_SD21_T2I",
                              directory: "IK_VAL_SD21_DIR")
    }

    // SDXL cross-attends to two towers at once. The first is Stable Diffusion 1.x's, read a layer
    // earlier; the second is wider and deeper still, and is the only one carrying a projection — which
    // is where the pooled embedding its UNet conditions on comes from.
    func testSDXLPrimaryTextEncoderMatchesTheReference() throws {
        try verifyTextEncoder("sdxl-text", .sdxlPrimary, record: "IK_PARITY_SDXL_TEXT",
                              weights: "IK_VAL_SDXL_TEXT")
    }

    func testSDXLSecondaryTextEncoderMatchesTheReference() throws {
        try verifyTextEncoder("sdxl-text-2", .sdxlSecondary, record: "IK_PARITY_SDXL_TEXT2",
                              weights: "IK_VAL_SDXL_TEXT2")
    }

    private func verifyTextEncoder(_ name: String, _ configuration: NFKMLXSDTextEncoderConfiguration,
                                   record: String, weights key: String) throws {
        try requireMLXRuntime()
        guard let path = config[record] else { throw XCTSkip("set \(record)") }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self).map(Int.init)

        for (label, output, expected) in [("last-hidden-state", NFKSDTextOutput.lastHiddenState, "output"),
                                          ("penultimate", .penultimateHiddenState, "penultimate")] {
            var variant = configuration
            variant.output = output
            let net = try NFKMLXSDTextEncoder.net(configuration: variant, weightsURL: try weights(key))
            let embedding = net.encode(tokens)
            eval(embedding.hidden)
            let reference = try XCTUnwrap(arrays[expected])
            XCTAssertEqual(Array(embedding.hidden.shape.dropFirst()), reference.shape,
                           "the reference's \(label) shape")
            let similarity = cosine(embedding.hidden.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY \(name): \(label) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.99, "the \(label) matches the reference")

            guard let pooled = embedding.pooled, let referencePooled = arrays["pooled"] else { continue }
            eval(pooled)
            let pooledSimilarity = cosine(pooled.reshaped([-1]).asArray(Float.self).map(Double.init),
                                          referencePooled.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("VALIDATION PARITY \(name): pooled cosine \(pooledSimilarity)")
            XCTAssertGreaterThan(pooledSimilarity, 0.99, "the pooled embedding matches the reference")
        }
    }

    private func weights(_ key: String) throws -> URL {
        guard let path = config[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key) to a converted safetensors checkpoint")
        }
        return URL(fileURLWithPath: path)
    }
}
