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
import InferKit
import MLX
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
        var configuration = NFKMLXWhisperConfiguration()
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
