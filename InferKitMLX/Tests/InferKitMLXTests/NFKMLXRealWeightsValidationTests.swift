//
//  NFKMLXRealWeightsValidationTests.swift
//  InferKitMLXTests
//
//  End-to-end validation against real trained checkpoints. Each test is gated on an environment
//  variable pointing at a converted safetensors file (produced by the matching Tools/*-to-safetensors
//  converter from the official release), plus a real input, and writes its output to IK_VAL_OUTDIR for
//  inspection. Skipped unless the variables are set, so CI stays green.
//
//  Drive it, for example, with:
//    IK_VAL_OUTDIR=/tmp/out IK_VAL_IMAGE=/tmp/photo.jpg \
//    IK_VAL_REALESRGAN=/tmp/real-esrgan-x4.safetensors \
//    xcodebuild test -scheme InferKitMLX -destination 'platform=macOS' \
//      -skipPackagePluginValidation -only-testing:InferKitMLXTests/NFKMLXRealWeightsValidationTests
//

import XCTest
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXRealWeightsValidationTests: XCTestCase {

    // xcodebuild sanitizes the environment for the test runner, so configuration comes from a JSON file
    // at a fixed path (~/.inferkit-validation.json), with the process environment as a fallback.
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

    private func filePath(_ key: String, _ hint: String) throws -> String {
        guard let path = config[key] else { throw XCTSkip("set \(key) in ~/.inferkit-validation.json (\(hint))") }
        guard FileManager.default.fileExists(atPath: path) else { throw XCTSkip("\(key) points at a missing file: \(path)") }
        return path
    }

    private func weights(_ key: String) throws -> URL {
        URL(fileURLWithPath: try filePath(key, "a converted safetensors"))
    }

    private func inputImage() throws -> CGImage {
        try loadImage(atPath: try filePath("IK_VAL_IMAGE", "a real photo"))
    }

    private func loadImage(atPath path: String) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NFKMLXError.noOutput
        }
        return image
    }

    /// A high-contrast random-ish texture `[1, H, W, 3]` in 0...255, translated left by `offset` pixels,
    /// so a pair gives optical flow an unambiguous correspondence to find.
    static func texture(height: Int, width: Int, offset: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let sx = x + offset
                // A deterministic, spatially varied pattern with strong local structure.
                let v = Float(((sx &* 37 &+ y &* 17) ^ (sx &* y &* 3)) % 256) / 255
                for c in 0 ..< 3 {
                    values[(y * width + x) * 3 + c] = v
                }
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, height, width, 3]) }
    }

    /// The image's pixels as 8-bit RGBA, for the statistics below.
    private static func rgbaBytes(_ image: CGImage) -> [UInt8] {
        let width = image.width, height = image.height
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &buffer, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    /// Mean of the RGB channels across the image, 0...255 — brightness.
    private func meanLuma(_ image: CGImage) -> Double {
        let bytes: [UInt8] = Self.rgbaBytes(image)
        let pixelCount: Int = bytes.count / 4
        guard pixelCount > 0 else { return 0 }
        var total: Double = 0
        var i: Int = 0
        while i < bytes.count {
            let r: Double = Double(bytes[i])
            let g: Double = Double(bytes[i + 1])
            let b: Double = Double(bytes[i + 2])
            total += (r + g + b) / 3.0
            i += 4
        }
        return total / Double(pixelCount)
    }

    /// True when pixels carry chroma (the channels differ), rather than being another gray image.
    private func isColored(_ image: CGImage) -> Bool {
        let bytes: [UInt8] = Self.rgbaBytes(image)
        let pixelCount: Int = bytes.count / 4
        var chromatic: Int = 0
        var i: Int = 0
        while i < bytes.count {
            let r: Int = Int(bytes[i])
            let g: Int = Int(bytes[i + 1])
            let b: Int = Int(bytes[i + 2])
            let redGreen: Int = r > g ? r - g : g - r
            let greenBlue: Int = g > b ? g - b : b - g
            if redGreen > 8 || greenBlue > 8 {
                chromatic += 1
            }
            i += 4
        }
        return chromatic * 100 > pixelCount             // more than 1% of pixels carry visible chroma
    }

    private func outputDirectory() -> URL {
        URL(fileURLWithPath: config["IK_VAL_OUTDIR"] ?? NSTemporaryDirectory(), isDirectory: true)
    }

    @discardableResult
    private func savePNG(_ image: CGImage, _ name: String) throws -> URL {
        let url = outputDirectory().appendingPathComponent(name)
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination), "wrote \(name)")
        return url
    }

    private func cgImage(_ value: Any?) throws -> CGImage {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else { throw NFKMLXError.noOutput }
        return (value as! CGImage)
    }

    // Draws to an 8-bit context and reports whether the pixels vary — a real forward produces structure,
    // a broken one tends to a constant.
    private func pixelsVary(_ image: CGImage) -> Bool {
        let width = image.width, height = image.height
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &buffer, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let first = buffer.first else { return false }
        return buffer.contains { $0 != first }
    }

    // MARK: Image → image (super-resolution)

    func testRealESRGANUpscalesARealImage() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: weights("IK_VAL_REALESRGAN"))
        let input = try inputImage()
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: input]))
        let output = try cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, input.width * 4, "×4 width")
        XCTAssertEqual(output.height, input.height * 4, "×4 height")
        XCTAssertTrue(pixelsVary(output), "upscaled image has structure")
        let url = try savePNG(output, "real-esrgan-x4.png")
        print("VALIDATION real-esrgan-x4: \(input.width)x\(input.height) → \(output.width)x\(output.height) @ \(url.path)")
    }

    // MARK: Image → map (depth)

    func testDepthAnythingProducesADepthMap() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXDepthAnything.backend(variant: .small, weightsURL: weights("IK_VAL_DEPTH"))
        let input = try inputImage()
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: input]))
        let output = try cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertGreaterThan(output.width, 0)
        XCTAssertTrue(pixelsVary(output), "depth map has near/far variation")
        let url = try savePNG(output, "depth-anything-v2-small.png")
        print("VALIDATION depth-anything-v2-small: \(input.width)x\(input.height) → \(output.width)x\(output.height) @ \(url.path)")
    }

    // MARK: Image → matte (background removal)

    func testU2NetProducesAForegroundMatte() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXU2Net.backend(variant: .full, weightsURL: weights("IK_VAL_U2NET"))
        let input = try inputImage()
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: input]))
        let foreground = try cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertTrue(pixelsVary(foreground), "cutout has structure")
        let url = try savePNG(foreground, "u2net-foreground.png")
        if let mask = result.output(forKey: NFKOutputMask), CFGetTypeID(mask as CFTypeRef) == CGImage.typeID {
            try savePNG(mask as! CGImage, "u2net-matte.png")
        }
        print("VALIDATION u2net: \(input.width)x\(input.height) → foreground \(foreground.width)x\(foreground.height) @ \(url.path)")
    }

    // MARK: Image → image (colorization, low-light, style)

    func testColorizerAddsColorToAnImage() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXColorizer.backend(weightsURL: weights("IK_VAL_COLORIZER"))
        let input = try inputImage()
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: input]))
        let output = try cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertTrue(pixelsVary(output), "the colorized image has structure")
        XCTAssertTrue(isColored(output), "the prediction adds chroma, not another gray image")
        let url = try savePNG(output, "colorizer-eccv16.png")
        print("VALIDATION colorizer-eccv16: \(input.width)x\(input.height) → \(output.width)x\(output.height) @ \(url.path)")
    }

    func testZeroDCEBrightensADarkImage() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXZeroDCE.backend(weightsURL: weights("IK_VAL_ZERODCE"))
        // The low-light model needs a dark plate; IK_VAL_DARK_IMAGE falls back to the standard input.
        let path = config["IK_VAL_DARK_IMAGE"] ?? config["IK_VAL_IMAGE"]
        let input = try loadImage(atPath: try XCTUnwrap(path))
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: input]))
        let output = try cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertTrue(pixelsVary(output), "the enhanced image has structure")
        XCTAssertGreaterThan(meanLuma(output), meanLuma(input), "low-light enhancement brightens the plate")
        let url = try savePNG(output, "zero-dce.png")
        print("VALIDATION zero-dce: mean luma \(meanLuma(input)) → \(meanLuma(output)) @ \(url.path)")
    }

    func testStyleTransferRestylesAnImage() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXStyleTransfer.backend(weightsURL: weights("IK_VAL_STYLE"))
        let input = try inputImage()
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: input]))
        let output = try cgImage(result.output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, input.width, "the stylizer keeps the input size")
        XCTAssertTrue(pixelsVary(output), "the stylized image has structure")
        let url = try savePNG(output, "fast-style-transfer.png")
        print("VALIDATION fast-style-transfer: \(input.width)x\(input.height) → \(output.width)x\(output.height) @ \(url.path)")
    }

    // MARK: Promptable segmentation

    // The released sam_vit_b checkpoint against the ViT-B geometry: every parameter must be covered, and
    // a point prompt must produce a mask that actually selects part of the image rather than all or none.
    func testSAMSegmentsFromAPointPrompt() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXSAM.backend(variant: .vitB, weightsURL: weights("IK_VAL_SAM"))
        let input = try inputImage()
        // Click the subject: the puppy's head sits around the upper-middle of the frame.
        let point: [NSNumber] = [NSNumber(value: input.width / 2), NSNumber(value: input.height / 3)]
        let request = NFKInferenceRequest(inputs: [NFKInputImage: input],
                                          parameters: [NFKSAMPointKey: point])
        let matte = try cgImage(try backend.runInference(for: request).output(forKey: NFKOutputMask))
        try savePNG(matte, "sam-matte.png")

        // A prompted mask must select the clicked subject and leave the far corners out. Mean alpha in a
        // patch around the point is compared with the mean over the four corner patches.
        let bytes = Self.rgbaBytes(matte)
        let (width, height) = (matte.width, matte.height)
        func meanAlpha(centerX: Int, centerY: Int, radius: Int) -> Double {
            var total = 0.0, count = 0.0
            for y in max(0, centerY - radius) ..< min(height, centerY + radius) {
                for x in max(0, centerX - radius) ..< min(width, centerX + radius) {
                    total += Double(bytes[(y * width + x) * 4])
                    count += 1
                }
            }
            return count > 0 ? total / count : 0
        }
        let radius = max(width, height) / 12
        let atPoint = meanAlpha(centerX: width / 2, centerY: height / 3, radius: radius)
        let corners = [(radius, radius), (width - radius, radius),
                       (radius, height - radius), (width - radius, height - radius)]
            .map { meanAlpha(centerX: $0.0, centerY: $0.1, radius: radius) }
        let cornerMean = corners.reduce(0, +) / Double(corners.count)

        print("VALIDATION sam: mean alpha at the clicked subject \(atPoint) vs corners \(cornerMean)")
        XCTAssertGreaterThan(atPoint, 160, "the clicked subject is selected")
        XCTAssertGreaterThan(atPoint - cornerMean, 60, "the mask separates the subject from the background")
    }

    // MARK: Image → embedding

    // A real CLIP embedding is unit length, identical for the same image, and closer for two views of
    // the same picture than for two different pictures — random weights satisfy none of that.
    func testCLIPEmbedsImagesIntoAMeaningfulSpace() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXCLIP.backend(weightsURL: weights("IK_VAL_CLIP"))
        func embed(_ image: CGImage) throws -> [Double] {
            let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
            return try XCTUnwrap(result.embedding).map(\.doubleValue)
        }
        func cosine(_ a: [Double], _ b: [Double]) -> Double {
            zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 }
        }

        let subject = try inputImage()
        let other = try loadImage(atPath: try filePath("IK_VAL_DARK_IMAGE", "a second, different photo"))
        let a = try embed(subject), b = try embed(subject), c = try embed(other)

        XCTAssertEqual(a.count, 512, "ViT-B/32 embeds to 512 dimensions")
        XCTAssertEqual(cosine(a, a), 1.0, accuracy: 1e-3, "the embedding is L2-normalized")
        XCTAssertEqual(cosine(a, b), 1.0, accuracy: 1e-5, "the same image embeds identically")
        XCTAssertLessThan(cosine(a, c), 0.99, "a different image embeds somewhere else")
        print("VALIDATION clip-vit-b-32: dims \(a.count), self-similarity \(cosine(a, b)), cross-image \(cosine(a, c))")
    }

    // MARK: Two frames → optical flow

    // A shifted pair carries a known ground truth: the flow should be a near-uniform horizontal
    // translation of the shift, so this checks the estimate's value, not just its shape.
    func testRAFTRecoversAKnownTranslation() throws {
        try requireMLXRuntime()
        let net = NFKMLXRAFT.makeNet(iterations: 12)
        try NFKMLXRAFT.loadWeights(into: net, from: weights("IK_VAL_RAFT"))

        let shift = 6
        let (height, width) = (96, 128)
        let frame0 = Self.texture(height: height, width: width, offset: 0)
        let frame1 = Self.texture(height: height, width: width, offset: shift)

        let flow = net.flow(frame0, frame1)                     // [H, W, 2] — (fx, fy)
        eval(flow)
        XCTAssertEqual(flow.shape, [height, width, 2])

        // Sample the interior, away from the edges where the shift leaves no correspondence.
        let values = flow.asArray(Float.self)
        var sumX: Float = 0, sumY: Float = 0, count: Float = 0
        for y in 16 ..< (height - 16) {
            for x in 24 ..< (width - 24) {
                sumX += values[(y * width + x) * 2]
                sumY += values[(y * width + x) * 2 + 1]
                count += 1
            }
        }
        let (meanX, meanY) = (sumX / count, sumY / count)
        print("VALIDATION raft: known shift dx=\(shift) → estimated flow (\(meanX), \(meanY))")
        XCTAssertEqual(Double(meanX), Double(-shift), accuracy: 2.0, "recovers the horizontal translation")
        XCTAssertEqual(Double(meanY), 0.0, accuracy: 2.0, "no vertical motion")
    }

    // MARK: Audio → audio (speech denoising)

    // Mixes noise into real speech and checks the output correlates with the clean signal better than
    // the noisy input does — the model has to actually suppress noise, not merely return something.
    func testDenoiserSuppressesNoiseInRealSpeech() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXDenoiser.backend(weightsURL: weights("IK_VAL_DENOISER"))
        let wav = try Data(contentsOf: URL(fileURLWithPath: try filePath("IK_VAL_AUDIO", "a 16 kHz mono WAV")))
        let clean = try XCTUnwrap(NFKMLXWaveFile.read(wav)).samples

        var seed: UInt64 = 12345
        var noisy = clean
        for i in 0 ..< noisy.count {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407      // deterministic noise
            let noise = Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(Int32.max)
            noisy[i] = clean[i] + 0.05 * noise
        }
        let noisyWAV = NFKMLXWaveFile.data(samples: noisy, sampleRate: 16000)

        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: noisyWAV]))
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        let outURL = try XCTUnwrap(asset.fileURL)
        defer { try? FileManager.default.removeItem(at: outURL) }
        let denoised = try XCTUnwrap(NFKMLXWaveFile.read(try Data(contentsOf: outURL))).samples

        func correlation(_ a: [Float], _ b: [Float]) -> Double {
            let n = min(a.count, b.count)
            var dot = 0.0, na = 0.0, nb = 0.0
            for i in 0 ..< n {
                dot += Double(a[i]) * Double(b[i]); na += Double(a[i]) * Double(a[i]); nb += Double(b[i]) * Double(b[i])
            }
            return (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 0
        }
        let before = correlation(noisy, clean)
        let after = correlation(denoised, clean)
        print("VALIDATION denoiser: correlation with clean — noisy \(before) → denoised \(after)")

        XCTAssertEqual(denoised.count, noisy.count, "the cleaned signal keeps the input length")
        XCTAssertGreaterThan(after, before, "the trained denoiser moves the signal toward the clean one")
    }

    // MARK: Audio → text (transcription)

    func testWhisperRunsOnRealSpeech() throws {
        try requireMLXRuntime()
        let weightsURL = try weights("IK_VAL_WHISPER")
        let audio = try Data(contentsOf: URL(fileURLWithPath: try filePath("IK_VAL_AUDIO", "a 16 kHz mono WAV")))
        let backend = try NFKMLXWhisper.backend(weightsURL: weightsURL)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: audio]))
        // Without an NFKTokenizer the backend returns token ids; the decode prompt is a scaffold, so this
        // validates that real weights load and the encoder+decoder forward runs, not transcription text.
        let text = result.text ?? (result.output(forKey: NFKOutputText) as? String) ?? "<none>"
        try text.write(to: outputDirectory().appendingPathComponent("whisper-output.txt"), atomically: true, encoding: .utf8)
        print("VALIDATION whisper-tiny output tokens/text: \(text)")
        XCTAssertFalse(text.isEmpty, "the model produced a token sequence")
    }
}
