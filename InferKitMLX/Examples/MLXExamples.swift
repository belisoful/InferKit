//
//  MLXExamples.swift
//  InferKitMLXExamples
//
//  Compiled and run by CI so the MLX snippets in Docs/examples.md cannot silently drift. Each method
//  mirrors a section there — change an example here and update the matching snippet, and vice versa.
//  Constructing the backends needs no GPU; running a forward needs the MLX Metal library, which the
//  Xcode build system bundles but a plain `swift test` does not, so evaluation is host-verified via
//  `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class MLXExamples: XCTestCase {

    // Docs/examples.md: Text → image and image → image
    func testExampleStableDiffusionBackend() {
        let backend = NFKMLXBackend(model: .sdxlTurbo)
        XCTAssertEqual(backend.backendIdentifier, "mlx-stable-diffusion")
        XCTAssertFalse(backend.isReady)                     // weights load on prepare / first use
    }

    // Docs/examples.md: Text → image, from a release directory a caller already holds
    func testExampleTextToImageFromAReleaseDirectory() {
        let releaseDirectory = URL(fileURLWithPath: "/nonexistent/stable-diffusion-v1-5")
        XCTAssertThrowsError(try NFKMLXTextToImage.backend(configuration: .stableDiffusion15,
                                                           directoryURL: releaseDirectory,
                                                           precision: .checkpoint),
                             "the release's files are what it needs")
    }

    // Docs/examples.md: Image → image (bring-your-own MLX)
    func testExampleModuleBackend() {
        let backend = NFKMLXModuleBackend(identifier: "sr", isReady: true) { $0 }
        XCTAssertEqual(backend.backendIdentifier, "sr")
    }

    // Docs/examples.md: Image → image + mask (matting)
    func testExampleMattingBackend() {
        var configuration = NFKMattingConfiguration()
        configuration.emitsMatte = true
        configuration.tileSize = 1024
        let backend = NFKMLXMattingBackend(identifier: "keyer", configuration: configuration) { plate, _ in plate }
        XCTAssertEqual(backend.backendIdentifier, "keyer")
    }

    // Docs/examples.md: Many tensors in and out
    func testExampleTensorBackend() {
        let configuration = NFKMLXTensorConfiguration(
            inputs: [NFKMLXTensorPort(key: NFKInputImage, tensorName: "foreground", channels: 4)],
            outputs: [NFKMLXTensorPort(key: NFKOutputImage, tensorName: "composite")])
        let backend = NFKMLXTensorBackend(identifier: "compositor", configuration: configuration) { $0 }
        XCTAssertEqual(backend.backendIdentifier, "compositor")
    }

    // Docs/examples.md: Diffusion — a bring-your-own MLX diffusion model (encode/denoise/decode +
    // scheduler). The forward closures are identity stand-ins here; a real model plugs in its UNet
    // and VAE. Construction needs no GPU.
    func testExampleCustomDiffusionBackend() {
        let backend = NFKMLXDiffusionBackend(
            identifier: "my-diffusion",
            configuration: NFKDiffusionConfiguration(steps: 20, guidanceScale: 7.5),
            scheduler: NFKDDIMScheduler(predictionType: .epsilon),
            encode: { _, _, _ in NFKDiffusionContext(width: 512, height: 512) },
            denoise: { latent, _, _, _ in latent },
            decode: { latent in clip(latent, min: 0, max: 1) })
        XCTAssertEqual(backend.backendIdentifier, "my-diffusion")
        XCTAssertTrue(backend.isReady)
    }

    // Docs/examples.md: Diffusion — the three reference pipelines register by name for the ObjC path.
    func testExampleDiffusionReferencesRegisterByName() throws {
        NFKMLXReferenceModels.registerDiffusionUpscaler()
        NFKMLXReferenceModels.registerDiffusionDepth()
        NFKMLXReferenceModels.registerDiffusionInpainter()
        for name in ["diffusion-upscaler", "diffusion-depth", "diffusion-inpaint"] {
            XCTAssertTrue(NFKMLXModelRegistry.registeredModelNames.contains(name))
            XCTAssertNoThrow(try NFKMLXModelRegistry.backend(named: name, weightsURL: nil))
        }
    }

    // Docs/examples.md: Real-ESRGAN upscaling — a real single-forward MLX model, built by name.
    // Building the generator constructs MLXNN layers (initializes MLX), so this runs under xcodebuild.
    func testExampleRealESRGANRegistersAndBuilds() throws {
        NFKMLXRealESRGAN.register()
        XCTAssertTrue(NFKMLXModelRegistry.registeredModelNames.contains(NFKMLXRealESRGAN.modelName))
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "building the RRDBNet initializes MLX; run via xcodebuild")
        let upscaler = try NFKMLXModelRegistry.backend(named: NFKMLXRealESRGAN.modelName, weightsURL: nil)
        XCTAssertEqual(upscaler.backendIdentifier, "real-esrgan-x4")
    }

    // Docs/examples.md: Depth Anything V2 — a real single-forward depth model, built by name.
    func testExampleDepthAnythingRegistersAndBuilds() throws {
        NFKMLXDepthAnything.register()
        XCTAssertTrue(NFKMLXModelRegistry.registeredModelNames.contains(NFKMLXDepthAnything.modelName))
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "building the DINOv2/DPT net initializes MLX; run via xcodebuild")
        let depth = try NFKMLXModelRegistry.backend(named: NFKMLXDepthAnything.modelName, weightsURL: nil)
        XCTAssertEqual(depth.backendIdentifier, "depth-anything-v2-small")
    }

    // Docs/examples.md: LaMa inpainting — a real single-forward FFC model, built by name.
    func testExampleLaMaRegistersAndBuilds() throws {
        NFKMLXLaMa.register()
        XCTAssertTrue(NFKMLXModelRegistry.registeredModelNames.contains(NFKMLXLaMa.modelName))
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "building the FFC net initializes MLX; run via xcodebuild")
        let inpainter = try NFKMLXModelRegistry.backend(named: NFKMLXLaMa.modelName, weightsURL: nil)
        XCTAssertEqual(inpainter.backendIdentifier, "lama-inpaint")
    }

    // Docs/examples.md: SD inpainting — a latent-diffusion model on the diffusion backend, by name.
    func testExampleSDInpaintRegistersAndBuilds() throws {
        NFKMLXStableDiffusionInpaint.register()
        XCTAssertTrue(NFKMLXModelRegistry.registeredModelNames.contains(NFKMLXStableDiffusionInpaint.modelName))
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "building the VAE/UNet initializes MLX; run via xcodebuild")
        let inpaint = try NFKMLXModelRegistry.backend(named: NFKMLXStableDiffusionInpaint.modelName, weightsURL: nil)
        XCTAssertEqual(inpaint.backendIdentifier, "sd-inpaint")
    }

    // Docs/examples.md: Text → audio — a bring-your-own MLX text-to-speech model, built by name.
    func testExampleSpeechRegistersAndBuilds() throws {
        NFKMLXReferenceModels.registerToneSpeech()
        XCTAssertTrue(NFKMLXModelRegistry.registeredModelNames.contains("tone-speech"))
        let speech = try NFKMLXModelRegistry.backend(named: "tone-speech", weightsURL: nil)
        XCTAssertEqual(speech.backendIdentifier, "tone-speech")
    }

    // Docs/examples.md: Running MLX models from Objective-C — a Swift model registers by name.
    func testExampleRegisterAndBuildByName() throws {
        NFKMLXReferenceModels.registerGreenScreenKeyer()
        XCTAssertTrue(NFKMLXModelRegistry.registeredModelNames.contains("green-screen-keyer"))
        let backend = try NFKMLXModelRegistry.backend(named: "green-screen-keyer", weightsURL: nil)
        XCTAssertEqual(backend.backendIdentifier, "green-screen-keyer")
    }

    // Docs/examples.md: Running MLX models from Objective-C — download from Hugging Face + build.
    // The download hits the network; this verifies the fail-fast path (unregistered model, no download).
    func testExampleHubFailsFastForAnUnregisteredModel() {
        XCTAssertThrowsError(try NFKMLXHub.backend(named: "not-registered",
                                                   repo: "org/model", weightsPath: "model.safetensors",
                                                   revision: nil, cacheDirectoryURL: nil))
    }

    // Docs/examples.md: the asynchronous download peer delivers the result to a completion handler off
    // the render thread. Network-free here: the unregistered model fails fast before any download.
    func testExampleAsyncHubDeliversFailureToTheCompletionHandler() {
        let done = expectation(description: "completion handler")
        NFKMLXHub.backend(named: "not-registered", repo: "org/model", weightsPath: "model.safetensors",
                          revision: nil, cacheDirectoryURL: nil) { backend, error in
            XCTAssertNil(backend)
            XCTAssertNotNil(error)
            done.fulfill()
        }
        wait(for: [done], timeout: 5)
    }

    // End to end: register -> build by name -> run. Needs MLX; runs under xcodebuild, skips under
    // `swift test` (no bundled metallib). Green keys out (alpha low); red is kept (alpha high).
    func testExampleReferenceKeyerRunsThroughTheRegistry() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
        NFKMLXReferenceModels.registerGreenScreenKeyer()
        let backend = try NFKMLXModelRegistry.backend(named: "green-screen-keyer", weightsURL: nil)

        let green = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solidImage(red: 0, green: 255, blue: 0)]))
        XCTAssertLessThan(Int(Self.alpha(of: green.output(forKey: NFKOutputImage))), 16, "green keyed out")

        let red = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solidImage(red: 255, green: 0, blue: 0)]))
        XCTAssertGreaterThan(Int(Self.alpha(of: red.output(forKey: NFKOutputImage))), 240, "red kept")
    }

    private static func solidImage(red: UInt8, green: UInt8, blue: UInt8) -> CGImage {
        let pixels: [UInt8] = [red, green, blue, 255]
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: 1, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    // Docs/examples.md: Customizing a model on a consumer's own data
    func testExampleFineTuningRoundTripsThroughTheShippedFactory() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("zerodce-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tuned) }

        let net = try NFKMLXZeroDCE.network(weightsURL: nil)
        var objective = NFKMLXZeroDCEObjective()
        objective.wellExposedLevel = 0.65

        let myPhotos = [Self.darkPhotos()]
        let history = try NFKMLXZeroDCE.fineTune(net, photos: { myPhotos[$0 % myPhotos.count] },
                                                 objective: objective, steps: 4,
                                                 checkpoint: NFKMLXTrainingCheckpoint(url: tuned,
                                                                                      everySteps: 2)) { _ in
            true                                            // return false to end the run early
        }
        XCTAssertEqual(history.count, 4)

        try NFKMLXWeights.save(net, to: tuned)
        let backend = try NFKMLXZeroDCE.backend(weightsURL: tuned)
        XCTAssertEqual(backend.backendIdentifier, NFKMLXZeroDCE.modelName)
    }

    // Docs/examples.md: Retargeting a segmentation model to your own classes
    func testExampleSegmentationDataAndSamplerFeedTheTrainer() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
        let myFrames = [Self.gray(8, level: 60), Self.gray(8, level: 200)]
        let myMasks = [Self.gray(8, level: 0), Self.gray(8, level: 255)]
        let sampler = NFKMLXBatchSampler(count: myFrames.count, seed: 7)

        let index = sampler.indices(forStep: 0)[0]
        let image = try NFKMLXTrainingData.tensor(myFrames[index])
        let labels = try NFKMLXTrainingData.labels(myMasks[index], classCount: 3)

        XCTAssertEqual(image.shape, [8, 8, 3])
        XCTAssertEqual(labels.shape, [8, 8])
        XCTAssertEqual(labels.dtype, .int32, "cross-entropy takes class indices")
    }

    // Docs/examples.md: LoRA, for models with no small head to train
    func testExampleLoRAAdaptsMergesAndLeavesAnOrdinaryCheckpoint() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
        let net = NFKMLXZeroDCENet(filters: 4)
        let adapted = try NFKMLXLoRA.apply(to: net, rank: 8, alpha: 16) { path, _ in
            path.hasSuffix("q") || path.hasSuffix("v")
        }
        // Zero-DCE is all convolution, so the attention predicate matches nothing — which `apply`
        // reports rather than hiding.
        XCTAssertEqual(adapted, 0)
        XCTAssertEqual(try NFKMLXLoRA.merge(into: net), 0)
    }

    // Docs/examples.md: A custom image classifier from a handful of photos
    func testExampleCLIPProbeClassifiesThroughABackend() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
        var configuration = NFKMLXCLIPConfiguration()
        configuration.imageResolution = 32
        configuration.patchSize = 16
        configuration.visionWidth = 32
        configuration.visionLayers = 1
        configuration.visionHeads = 2
        configuration.embedDimensions = 16
        configuration.textWidth = 32
        configuration.textLayers = 1
        configuration.textHeads = 2
        configuration.contextLength = 16
        configuration.vocabularySize = 64
        let clip = try NFKMLXCLIP.network(weightsURL: nil, configuration: configuration)

        let myPhotos = [Self.gray(32, level: 40), Self.gray(32, level: 220)]
        let cached = try NFKMLXCLIP.embeddings(for: myPhotos, using: clip)      // run once
        let probe = NFKMLXCLIPProbe(embedDimensions: 16, classCount: 2)
        try NFKMLXCLIP.trainProbe(probe, embeddings: cached, labels: MLXArray([Int32(0), 1]), steps: 20)

        let classifier = NFKMLXCLIP.probeBackend(net: clip, probe: probe, labels: ["dark", "bright"])
        let result = try classifier.runInference(
            for: NFKInferenceRequest(inputs: [NFKInputImage: myPhotos[0]]))
        XCTAssertEqual(result.classifications?.count, 2)
    }

    private static func gray(_ side: Int, level: UInt8) -> CGImage {
        var pixels = [UInt8](repeating: 255, count: side * side * 4)
        for pixel in 0 ..< (side * side) {
            pixels[pixel * 4] = level
            pixels[pixel * 4 + 1] = level
            pixels[pixel * 4 + 2] = level
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    private static func darkPhotos() -> MLXArray {
        var values = [Float](repeating: 0, count: 16 * 16 * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 37) % 60) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, 16, 16, 3]) }
    }

    private static func alpha(of value: Any?) -> UInt8 {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else { return 0 }
        return [UInt8]((value as! CGImage).dataProvider!.data! as Data)[3]
    }

    // Docs/examples.md: A live preview of each step. The map is a 1×1 convolution over the channel
    // axis, reported as the job's partialResult — the mechanism a streaming text backend uses.
    func testExampleADiffusionRunPreviewsEveryStep() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "the sampler evaluates MLX arrays; run via xcodebuild")
        var configuration = NFKDiffusionConfiguration(steps: 4, latentChannels: 3, plateChannels: 3)
        configuration.latentPreview = .passthrough      // .stableDiffusion / .stableDiffusionXL for a VAE latent
        configuration.previewEverySteps = 1

        let backend = NFKMLXDiffusionBackend(
            configuration: configuration,
            encode: { _, _, _ in NFKDiffusionContext(width: 8, height: 8) },
            denoise: { latent, _, _, _ in MLXArray.zeros(latent.shape) },
            decode: { ($0 + 1) / 2 })

        let job = backend.submitInferenceJob(for: NFKInferenceRequest(inputs: [:], parameters: [:]))
        // `partialResult` holds the LAST non-nil value, so compare identity rather than presence when
        // the preview rate matters.
        var previews: [NFKInferenceResult] = []
        let finished = expectation(description: "the run finished")
        job.progressHandler = { job in
            if let partial = job.partialResult, partial.output(forKey: NFKOutputImage) != nil,
               previews.last !== partial {
                previews.append(partial)
            }
        }
        job.completionHandler = { _ in finished.fulfill() }
        wait(for: [finished], timeout: 60)

        XCTAssertEqual(job.status, .succeeded)
        XCTAssertEqual(previews.count, 4, "one preview per step")
    }

    // A map for a model with no published factors is derived from its own decoder by least squares.
    func testExampleAPreviewMapIsFittedToADecoder() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "the fit evaluates MLX arrays; run via xcodebuild")
        let map = try XCTUnwrap(NFKDiffusionLatentPreview.fitted(
            latentChannels: 4,
            decode: { latent in
                let matrix = MLXArray([Float](repeating: 0.2, count: 12), [4, 3])
                let flat = latent.reshaped([latent.shape[0] * latent.shape[1], 4])
                return matmul(flat, matrix).reshaped([latent.shape[0], latent.shape[1], 3])
            },
            sample: { index in MLXRandom.normal([8, 8, 4], key: MLXRandom.key(UInt64(index))) }))
        XCTAssertEqual(map.latentChannels, 4)

        // A mismatched channel count returns nil: a preview is a progress indicator, so it never
        // fails the run it is only reporting on.
        XCTAssertNil(map.image(from: MLXArray.zeros([8, 8, 3])))
    }

    // Docs/examples.md: Local, on device through MLX. The cache bound and the rotary scaling are the
    // two knobs that decide whether a long conversation survives.
    func testExampleAContextWindowBoundsTheCache() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "runs the decoder; run via xcodebuild")
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 4
        // Retain at most this many positions; the oldest are dropped as new ones arrive.
        options.contextWindow = 8
        XCTAssertEqual(options.contextWindow, 8)

        let net = NFKMLXLanguage.makeNet(.tiny)
        let produced = net.generate(prompt: [1, 2, 3, 4, 5, 6], options: options)
        XCTAssertLessThanOrEqual(produced.count, options.maxTokens)

        // Unbounded is the default, so an existing caller's results are unchanged.
        XCTAssertNil(NFKMLXGenerationOptions().contextWindow)
    }

    // Docs/examples.md: Extended context. A release states its own scaling; an unimplemented kind is
    // refused rather than loaded under a rotary it was not trained with.
    func testExampleRoPEScalingIsReadFromTheReleaseAndUnknownKindsAreRefused() throws {
        let yarn = try XCTUnwrap(NFKMLXRoPEScaling.read(
            ["rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192],
            maximumPositions: 32768))
        XCTAssertEqual(yarn.kind, .yarn)
        // The attention factor is derived unless the config states one.
        XCTAssertEqual(yarn.attentionFactor, 0.1 * log(Float(4)) + 1, accuracy: 1e-6)

        // A config that declares nothing scales nothing.
        XCTAssertNil(try NFKMLXRoPEScaling.read(nil, maximumPositions: 32768))

        XCTAssertThrowsError(try NFKMLXRoPEScaling.read(
            ["rope_type": "llama3", "factor": 8.0], maximumPositions: 131072))
    }

    // Docs/examples.md: Sizing it to the machine. The window is derived from what is free rather than
    // guessed, and a model that cannot fit says so here instead of at the load.
    func testExampleOptionsAreSizedToTheMachine() throws {
        let options = try NFKMLXModelSizing.options(for: .qwen3_0_6B, requesting: 8192,
                                                    precision: .checkpoint)
        // Left unset when the request already fits; a number when the cache has to be bounded.
        if let window = options.contextWindow {
            XCTAssertGreaterThanOrEqual(window, 0)
        }

        let fit = NFKMLXModelSizing.fit(of: .qwen3_0_6B, tokens: 8192, precision: .checkpoint)
        XCTAssertGreaterThan(fit.weightBytes, 0)
        XCTAssertGreaterThan(fit.describedFit.count, 0)

        // A model far too large for any machine reports that no window helps.
        var enormous = NFKMLXLanguageConfiguration.qwen3_8B
        enormous.layerCount = 4096
        XCTAssertThrowsError(try NFKMLXModelSizing.options(for: enormous, requesting: 1024,
                                                           precision: .float32))
    }

    // The decode ceiling is bounded by memory traffic, so it falls as the model and the context grow.
    func testExampleTheDecodeCeilingFallsWithSizeAndContext() {
        let bandwidth = 300e9      // supplied, so the example does not run a measurement
        let small = NFKMLXModelSizing.decodeCeiling(for: .qwen3_0_6B, contextLength: 4096,
                                                    precision: .checkpoint, bandwidth: bandwidth)
        let large = NFKMLXModelSizing.decodeCeiling(for: .qwen3_8B, contextLength: 4096,
                                                    precision: .checkpoint, bandwidth: bandwidth)
        XCTAssertGreaterThan(small, large)
    }

    // Docs/examples.md: Loading a PyTorch checkpoint directly
    func testExampleAPyTorchCheckpointReadsAndConvertsWithNoPython() throws {
        // A real .pth comes from a release; this embedded one keeps the example runnable offline.
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("example-\(UUID().uuidString).pth")
        try Data(base64Encoded: MLXExamples.tinyTorchCheckpoint)!.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        // Inspect the state dict before deciding to load it.
        let checkpoint = try NFKMLXTorchCheckpoint.checkpoint(contentsOf: url)
        XCTAssertEqual(checkpoint.tensorNames, ["conv.bias", "conv.weight"])
        XCTAssertEqual(checkpoint.info(forTensor: "conv.weight")?.shape, [2, 2])
        XCTAssertEqual(checkpoint.info(forTensor: "conv.weight")?.scalarType, .float32)

        // Convert on device: the output is what the model's Tools converter produces, and every
        // weightsURL: factory reads it. Or skip this step — the loaders sniff a .pth's bytes and
        // read it directly through the same reader.
        let converted = url.deletingPathExtension().appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: converted) }
        try checkpoint.writeSafetensors(to: converted)
        XCTAssertTrue(FileManager.default.fileExists(atPath: converted.path))
    }

    /// `torch.save` of a two-tensor state dict, legacy (pre-1.6) serialization: 488 bytes.
    private static let tinyTorchCheckpoint = """
        gAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAoh\
        YCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAmNjb2xsZWN0aW\
        9ucwpPcmRlcmVkRGljdApxAClScQEoWAsAAABjb252LndlaWdodHECY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKc\
        QMoKFgHAAAAc3RvcmFnZXEEY3RvcmNoCkZsb2F0U3RvcmFnZQpxBVgLAAAAMzkzODAxNzMyODBxBlgDAAAAY3B1cQdLBE50cQhR\
        SwBLAksChnEJSwJLAYZxColoAClScQt0cQxScQ1YCQAAAGNvbnYuYmlhc3EOaAMoKGgEaAVYCwAAADM5MzgwMTczMDg4cQ9oB0s\
        CTnRxEFFLAEsChXERSwGFcRKJaAApUnETdHEUUnEVdS6AAl1xAChYCwAAADM5MzgwMTczMDg4cQFYCwAAADM5MzgwMTczMjgwcQ\
        JlLgIAAAAAAAAAAAAAPwAAAL8EAAAAAAAAAAAAgD8AAABAAABAQAAAgEA=
        """
}
