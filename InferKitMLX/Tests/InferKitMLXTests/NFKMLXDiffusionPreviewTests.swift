//
//  NFKMLXDiffusionPreviewTests.swift
//  InferKitMLXTests
//
//  The latent preview: the map itself, the fit that derives one from a decoder, and — against a real
//  Stable Diffusion autoencoder — how closely the shipped coefficients track a full decode. That last
//  one is the only thing that can justify shipping constants, so it is measured rather than asserted.
//

import XCTest
import CoreGraphics
import ImageIO
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDiffusionPreviewTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    /// Pearson correlation: cosine after each side's mean is removed.
    ///
    /// Raw cosine over `0...1` images is dominated by their shared mean — two nearly constant grey
    /// images score above 0.98 — so it cannot tell a preview that shows the picture from one that
    /// shows a flat rectangle of roughly the right brightness.
    private func correlation(_ a: [Float], _ b: [Float]) -> Float {
        let meanA = a.reduce(Float(0), +) / Float(a.count)
        let meanB = b.reduce(Float(0), +) / Float(b.count)
        return cosine(a.map { $0 - meanA }, b.map { $0 - meanB })
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let left = sqrt(a.reduce(Float(0)) { $0 + $1 * $1 })
        let right = sqrt(b.reduce(Float(0)) { $0 + $1 * $1 })
        return (left > 0 && right > 0) ? dot / (left * right) : 0
    }

    // MARK: The map

    func testTheMapAppliesThePerChannelCombinationAndTheBias() throws {
        try requireMLXRuntime()
        // One latent channel contributing only to red, with a half bias everywhere.
        let map = NFKDiffusionLatentPreview(weights: [0.25, 0, 0], biases: [0.5, 0.5, 0.5])
        let latent = MLXArray([Float(1), -1], [1, 2, 1])
        let image = try XCTUnwrap(map.image(from: latent))
        eval(image)

        XCTAssertEqual(image.shape, [1, 2, 3])
        let values = image.asArray(Float.self)
        XCTAssertEqual(values[0], 0.75, accuracy: 1e-6, "0.5 + 0.25·1")
        XCTAssertEqual(values[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(values[3], 0.25, accuracy: 1e-6, "0.5 + 0.25·(−1)")
    }

    func testTheResultIsClampedIntoTheDisplayableRange() throws {
        try requireMLXRuntime()
        let map = NFKDiffusionLatentPreview(weights: [10, 10, 10], biases: [0, 0, 0])
        let latent = MLXArray([Float(5), -5], [1, 2, 1])
        let image = try XCTUnwrap(map.image(from: latent))
        eval(image)
        let values = image.asArray(Float.self)
        XCTAssertEqual(values.max(), 1)
        XCTAssertEqual(values.min(), 0)
    }

    // A map built for four channels cannot describe a three-channel latent, and returning nil is what
    // keeps a mismatched preview from failing the run it is only reporting on.
    func testAMismatchedChannelCountReturnsNilRatherThanThrowing() throws {
        try requireMLXRuntime()
        let latent = MLXArray.zeros([4, 4, 3])
        XCTAssertNil(NFKDiffusionLatentPreview.stableDiffusion.image(from: latent))
        XCTAssertNotNil(NFKDiffusionLatentPreview.passthrough.image(from: latent))
    }

    func testTheShippedMapsDeclareTheChannelCountsTheyAreFor() {
        XCTAssertEqual(NFKDiffusionLatentPreview.passthrough.latentChannels, 3)
        XCTAssertEqual(NFKDiffusionLatentPreview.stableDiffusion.latentChannels, 4)
        XCTAssertEqual(NFKDiffusionLatentPreview.stableDiffusionXL.latentChannels, 4)
    }

    // The passthrough map exists to show a latent that is already an image, so it must be exactly the
    // `-1...1` to `0...1` remap and nothing more.
    func testPassthroughMapsSignedImageRangeOntoDisplayRange() throws {
        try requireMLXRuntime()
        let latent = MLXArray([Float(-1), 0, 1, 1, 0, -1], [1, 2, 3])
        let image = try XCTUnwrap(NFKDiffusionLatentPreview.passthrough.image(from: latent))
        eval(image)
        XCTAssertEqual(image.asArray(Float.self), [0, 0.5, 1, 1, 0.5, 0])
    }

    // MARK: Fitting a map to a decoder

    // The fit has to recover a map that is genuinely linear, or it cannot be trusted on one that is
    // only approximately so.
    func testFittingRecoversAKnownLinearDecoder() throws {
        try requireMLXRuntime()
        let truth = NFKDiffusionLatentPreview(
            weights: [0.3, -0.1, 0.2,
                      0.05, 0.4, -0.2,
                      -0.25, 0.15, 0.35,
                      0.1, 0.2, -0.05],
            biases: [0.5, 0.45, 0.55])

        let fitted = try XCTUnwrap(NFKDiffusionLatentPreview.fitted(
            latentChannels: 4,
            decode: { latent in
                // Unclamped, so the fit sees the whole linear relationship.
                let matrix = MLXArray(truth.weights, [4, 3])
                let flat = latent.reshaped([latent.shape[0] * latent.shape[1], 4])
                return (matmul(flat, matrix) + MLXArray(truth.biases, [1, 3]))
                    .reshaped([latent.shape[0], latent.shape[1], 3])
            },
            sample: { index in MLXRandom.normal([8, 8, 4], key: MLXRandom.key(UInt64(index) &+ 5)) },
            count: 3))

        for (recovered, expected) in zip(fitted.weights, truth.weights) {
            XCTAssertEqual(recovered, expected, accuracy: 1e-3)
        }
        for (recovered, expected) in zip(fitted.biases, truth.biases) {
            XCTAssertEqual(recovered, expected, accuracy: 1e-3)
        }
    }

    // A decoder that upsamples is the real case: the fit compares at the latent's own resolution.
    func testFittingHandlesADecoderThatUpsamples() throws {
        try requireMLXRuntime()
        let fitted = NFKDiffusionLatentPreview.fitted(
            latentChannels: 4,
            decode: { latent in
                let matrix = MLXArray([Float](repeating: 0.25, count: 12), [4, 3])
                let flat = latent.reshaped([latent.shape[0] * latent.shape[1], 4])
                let small = matmul(flat, matrix).reshaped([latent.shape[0], latent.shape[1], 3])
                return repeated(repeated(small, count: 8, axis: 0), count: 8, axis: 1)
            },
            sample: { index in MLXRandom.normal([4, 4, 4], key: MLXRandom.key(UInt64(index) &+ 11)) },
            count: 3)
        let map = try XCTUnwrap(fitted)
        XCTAssertEqual(map.latentChannels, 4)
        for weight in map.weights {
            XCTAssertEqual(weight, 0.25, accuracy: 1e-3, "the fit saw through the upsampling")
        }
    }

    func testFittingReportsFailureRatherThanReturningAMeaninglessMap() throws {
        try requireMLXRuntime()
        // A decode that produces the wrong rank contributes nothing, so there is nothing to solve.
        XCTAssertNil(NFKDiffusionLatentPreview.fitted(
            latentChannels: 4,
            decode: { _ in MLXArray.zeros([4, 4]) },
            sample: { _ in MLXArray.zeros([4, 4, 4]) },
            count: 2))
    }

    // MARK: Through the backend

    // A preview reaches the caller as the job's partial result, which is the mechanism a streaming
    // text backend already uses.
    func testTheBackendReportsAPreviewAsEachStepsPartialResult() throws {
        try requireMLXRuntime()
        var configuration = NFKDiffusionConfiguration(steps: 4, latentChannels: 3, plateChannels: 3)
        configuration.latentPreview = .passthrough

        let backend = NFKMLXDiffusionBackend(
            configuration: configuration,
            encode: { _, _, _ in NFKDiffusionContext(width: 8, height: 8) },
            denoise: { latent, _, _, _ in MLXArray.zeros(latent.shape) },
            decode: { ($0 + 1) / 2 })

        let job = backend.submitInferenceJob(for: NFKInferenceRequest(inputs: [:], parameters: [:]))
        // Two things make counting this fiddly. `partialResult` holds the LAST non-nil value, so a
        // step that reports no preview still reads as one; and an ObjectIdentifier is only unique
        // among LIVE objects, so released results hand their address to their successors. Keeping a
        // strong reference to each distinct result settles both.
        var partials: [NFKInferenceResult] = []
        let finished = expectation(description: "the run finished")
        job.progressHandler = { job in
            if let partial = job.partialResult, partial.output(forKey: NFKOutputImage) != nil,
               partials.last !== partial {
                partials.append(partial)
            }
        }
        job.completionHandler = { _ in finished.fulfill() }
        wait(for: [finished], timeout: 60)

        XCTAssertEqual(job.status, .succeeded)
        XCTAssertEqual(partials.count, 4, "one preview per step")
    }

    func testPreviewEveryStepsThinsTheReports() throws {
        try requireMLXRuntime()
        var configuration = NFKDiffusionConfiguration(steps: 6, latentChannels: 3, plateChannels: 3)
        configuration.latentPreview = .passthrough
        configuration.previewEverySteps = 3

        let backend = NFKMLXDiffusionBackend(
            configuration: configuration,
            encode: { _, _, _ in NFKDiffusionContext(width: 8, height: 8) },
            denoise: { latent, _, _, _ in MLXArray.zeros(latent.shape) },
            decode: { ($0 + 1) / 2 })

        let job = backend.submitInferenceJob(for: NFKInferenceRequest(inputs: [:], parameters: [:]))
        var partials: [NFKInferenceResult] = []
        let finished = expectation(description: "the run finished")
        job.progressHandler = { job in
            if let partial = job.partialResult, partial.output(forKey: NFKOutputImage) != nil,
               partials.last !== partial {
                partials.append(partial)
            }
        }
        job.completionHandler = { _ in finished.fulfill() }
        wait(for: [finished], timeout: 60)

        XCTAssertEqual(partials.count, 2, "steps 0 and 3 of six")
    }

    // Previews are off unless a map is set, so an existing consumer's progress reports are unchanged.
    func testNoPreviewIsReportedWithoutAMap() throws {
        try requireMLXRuntime()
        let backend = NFKMLXDiffusionBackend(
            configuration: NFKDiffusionConfiguration(steps: 3, latentChannels: 3, plateChannels: 3),
            encode: { _, _, _ in NFKDiffusionContext(width: 8, height: 8) },
            denoise: { latent, _, _, _ in MLXArray.zeros(latent.shape) },
            decode: { ($0 + 1) / 2 })

        let job = backend.submitInferenceJob(for: NFKInferenceRequest(inputs: [:], parameters: [:]))
        var sawPartial = false
        var progressReports = 0
        let finished = expectation(description: "the run finished")
        job.progressHandler = { job in
            progressReports += 1
            if job.partialResult != nil { sawPartial = true }
        }
        job.completionHandler = { _ in finished.fulfill() }
        wait(for: [finished], timeout: 60)

        XCTAssertGreaterThan(progressReports, 0, "progress is still reported")
        XCTAssertFalse(sawPartial, "but carries no partial result")
    }

    // MARK: Against a real autoencoder

    /// What the shipped Stable Diffusion coefficients are actually worth.
    ///
    /// Two things make this measurement easy to get wrong, and the first version here got both.
    ///
    /// **The latent convention.** The sampler's latent is the SCALED one — it starts as unit noise
    /// and the pipeline divides by `scaleFactor` before the VAE sees it. A preview map is applied to
    /// the sampler's latent, so it must be measured against one whose magnitude matches.
    ///
    /// **The measure.** Raw cosine over `0...1` images is dominated by their shared mean: two nearly
    /// constant grey images score above 0.98 while sharing no structure at all. What a progress
    /// indicator has to get right is the STRUCTURE, so the number that matters is the correlation
    /// after each side's mean is removed. Both are printed, because the gap between them is the point.
    ///
    /// The latent comes from encoding a real photograph, which is what a latent looks like near the
    /// end of a run. Pure noise would measure the map at the one point where a preview is useless.
    ///
    /// Set `IK_SD15_DIR` and `IK_VAL_IMAGE` in `~/.inferkit-validation.json`.
    func testTheShippedStableDiffusionMapTracksARealDecode() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_SD15_DIR"], let imagePath = config["IK_VAL_IMAGE"] else {
            throw XCTSkip("set IK_SD15_DIR and IK_VAL_IMAGE in ~/.inferkit-validation.json")
        }
        let vaeURL = URL(fileURLWithPath: directory)
            .appendingPathComponent("vae/diffusion_pytorch_model.safetensors")
        guard FileManager.default.fileExists(atPath: vaeURL.path),
              let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: imagePath) as CFURL, nil),
              let photograph = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw XCTSkip("the autoencoder or the photograph is missing")
        }

        let vae = NFKMLXSDAutoencoder(configuration: .stableDiffusion)
        try NFKMLXStableDiffusionModels.loadVAEWeights(into: vae, from: vaeURL, precision: .float32)
        vae.train(false)
        let scaleFactor = NFKMLXSDVAEConfiguration.stableDiffusion.scaleFactor

        // Encode a real photograph, then scale into the sampler's own representation.
        let plate = try NFKMLXImageBridge.tensor(from: photograph, channels: 3,
                                                 colorSpace: CGColorSpaceCreateDeviceRGB())
        let sized = NFKMLXResample.resizeBilinear(plate.expandedDimensions(axis: 0),
                                                  height: 256, width: 256)
        let encoded = vae.encode(sized * 2 - 1).mean.squeezed(axis: 0) * scaleFactor
        let rootMeanSquare = sqrt(encoded.square().mean())
        eval(encoded, rootMeanSquare)

        let decode: (MLXArray) -> MLXArray = { latent in
            let decoded = vae.decode(latent.expandedDimensions(axis: 0) / scaleFactor)
            return clip((decoded.squeezed(axis: 0) + 1) / 2, min: 0, max: 1)
        }

        let truth = decode(encoded)
        let pooled = NFKMLXResample.averagePooled(truth.expandedDimensions(axis: 0),
                                                  kernel: .init(8), stride: .init(8)).squeezed(axis: 0)
        let shipped = try XCTUnwrap(NFKDiffusionLatentPreview.stableDiffusion.image(from: encoded))
        eval(pooled, shipped)

        let target = pooled.asArray(Float.self)
        let shippedValues = shipped.asArray(Float.self)

        // The ceiling: the best linear map there is, fitted on OTHER latents — noise at the sampler's
        // own scale — so it is not scored on its own training data.
        let best = try XCTUnwrap(NFKDiffusionLatentPreview.fitted(
            latentChannels: 4, decode: decode,
            sample: { index in MLXRandom.normal([32, 32, 4], key: MLXRandom.key(UInt64(index) &+ 7)) },
            count: 3))
        let bestImage = try XCTUnwrap(best.image(from: encoded))
        eval(bestImage)
        let bestValues = bestImage.asArray(Float.self)

        print("""
              latent preview against the released SD 1.5 decode (latent RMS \(rootMeanSquare.item(Float.self))):
                shipped  correlation \(correlation(shippedValues, target))  raw cosine \(cosine(shippedValues, target))
                fitted   correlation \(correlation(bestValues, target))  raw cosine \(cosine(bestValues, target))
                fitted weights \(best.weights)
                fitted biases  \(best.biases)
              """)

        // A preview is a progress indicator, so the bar is "recognizably the same picture". The
        // mean-removed correlation carries that; raw cosine would pass on a grey rectangle.
        XCTAssertGreaterThan(correlation(shippedValues, target), 0.7,
                             "the shipped coefficients track the decode's STRUCTURE, not just its mean")
    }
}
