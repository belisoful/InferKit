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

    // Docs/examples.md: Speculative decoding — a small draft proposes, the model verifies in one pass,
    // and the output is the model's own greedy run. Tiny random nets here; a Qwen3-0.6B drafting for
    // a Qwen3-1.7B is the released pairing.
    func testExampleSpeculativeDecodingReproducesThePlainRun() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "runs the decoder; run via xcodebuild")
        let model = NFKMLXLanguage.makeNet(.tiny)
        let draft = NFKMLXLanguage.makeNet(.tiny)         // any model sharing the vocabulary
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 12
        options.draftTokens = 4                            // proposals per verification round
        var report = NFKMLXSpeculativeReport()
        let produced = model.generate(prompt: [3, 17, 42], options: options, draft: draft, report: &report)
        XCTAssertEqual(produced, model.generate(prompt: [3, 17, 42], options: options))
        XCTAssertGreaterThan(report.rounds, 0)             // report.acceptanceRate says how much it helped
    }

    // Docs/examples.md: A prompt cache — the next turn of a conversation prefills only what is new.
    func testExampleAPromptCacheContinuesAConversation() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "runs the decoder; run via xcodebuild")
        let model = NFKMLXLanguage.makeNet(.tiny)
        let cache = NFKMLXPromptCache(layerCount: model.configuration.layerCount)
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 4
        let turn = [3, 17, 42, 8]
        let reply = model.generate(prompt: turn, options: options, promptCache: cache)
        // The follow-up extends the first turn; only the new tokens run through the model.
        let followUp = turn + reply + [91, 5]
        let continued = model.generate(prompt: followUp, options: options, promptCache: cache)
        XCTAssertEqual(continued, model.generate(prompt: followUp, options: options))
        XCTAssertEqual(cache.count, followUp.count + continued.count)
        // Persist a long system prompt's cache and reload it later.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("system-prompt.safetensors")
        try cache.save(to: url)
        XCTAssertEqual(try NFKMLXPromptCache.load(from: url).tokens, cache.tokens)
        try? FileManager.default.removeItem(at: url)
    }

    // Docs/examples.md: A mixture-of-experts release reads through the same factory; the config
    // says it is one, and the loader stacks the released per-expert tensors.
    func testExampleAMixtureOfExpertsConfiguration() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "runs the decoder; run via xcodebuild")
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("config.json")
        try #"""
        {"architectures":["Qwen3MoeForCausalLM"],"model_type":"qwen3_moe","hidden_size":2048,
         "num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":4,"head_dim":128,
         "moe_intermediate_size":768,"num_experts":128,"num_experts_per_tok":8,"norm_topk_prob":true}
        """#.write(to: url, atomically: true, encoding: .utf8)
        let configuration = try NFKMLXLanguage.configuration(fromHuggingFace: url)   // Qwen3-30B-A3B
        XCTAssertTrue(configuration.isMixtureOfExperts)
        XCTAssertEqual(configuration.activeExpertCount, 8)
        // A tiny mixture runs with random weights, like the dense tiny configuration.
        let produced = NFKMLXLanguage.makeNet(.tinyMixture).generate(prompt: [3, 17], options: .init())
        XCTAssertFalse(produced.isEmpty)
    }

    // Docs/examples.md: Gemma 4's 26B-A4B mixture runs a routed-expert branch beside every layer's
    // dense feed-forward. A released directory whose config.json sets `enable_moe_block` turns it on
    // through the same @objc directory factory; a tiny random geometry exercises the path here.
    func testExampleGemma4MixtureConfiguration() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "runs the decoder; run via xcodebuild")
        XCTAssertTrue(NFKMLXGemmaConfiguration.tinyMixture.isMixtureOfExperts)
        XCTAssertFalse(NFKMLXGemmaConfiguration.e2b.isMixtureOfExperts)   // the dense E-series
        let net = NFKMLXGemmaLanguage.makeNet(.tinyMixture)
        let logits = net(MLXArray([3, 17, 42]).reshaped([1, 3]))
        XCTAssertEqual(logits.shape, [1, 3, NFKMLXGemmaConfiguration.tinyMixture.vocabularySize])
    }

    // Docs/examples.md: A GGUF release generates text end to end. The native reader turns the file's
    // metadata into a configuration, remaps its llama.cpp tensor names onto the module's keys
    // (un-permuting the rotary query/key projections), and rebuilds the embedded tokenizer, so one file
    // is the whole model. A real .gguf is needed to run it; the factory selectors are what this checks.
    func testExampleLoadingAGGUFReleaseGeneratesText() throws {
        // let backend = try NFKMLXLanguage.backend(ggufURL: url)                        // Swift
        // NFKInferenceBackend *llm = [NFKMLXLanguage backendWithGGUFURL:url error:&e];  // Objective-C
        // The config reader maps GGUF metadata (architecture, block_count, embedding_length, …) onto
        // the same NFKMLXLanguageConfiguration a HF config.json produces.
        XCTAssertTrue(NFKMLXLanguage.responds(to: Selector(("backendWithGGUFURL:error:"))))
    }

    // Docs/examples.md: The Gemma 4 decoders (E2B/E4B, the 26B-A4B mixture, the 12B unified) generate
    // text through their own backend, which reads a release directory and dispatches on the config's
    // model type. Gemma runs prefill-only (no key-value cache). A real release is needed to run it.
    func testExampleGemmaTextBackend() throws {
        // let backend = try NFKMLXGemmaLanguage.backend(directoryURL: releaseDirectory)         // Swift
        // NFKInferenceBackend *llm = [NFKMLXGemmaLanguage gemmaBackendWithDirectoryURL:dir error:&e]; // ObjC
        XCTAssertTrue(NFKMLXGemmaLanguage.responds(to: Selector(("gemmaBackendWithDirectoryURL:error:"))))
    }

    // Docs/examples.md: Gemma 4's tri-modal preprocessing. The image processor turns a CGImage into the
    // vision tower's flattened patches and positions; the audio feature extractor turns raw audio into
    // the log-mel features the audio tower reads; the multimodal embedder projects a tower's soft tokens
    // into the decoder's space, and the fusion splices them at the placeholder positions.
    func testExampleGemma4TriModalPreprocessing() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "builds MLX arrays; run via xcodebuild")
        // A tiny 48×48 image → 3×3 patches with a 1×1-pooling processor.
        var bytes = [UInt8](repeating: 255, count: 48 * 48 * 4)
        for i in stride(from: 0, to: bytes.count, by: 4) { bytes[i] = 120 }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let image = CGImage(width: 48, height: 48, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 48 * 4,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let (pixels, positions) = NFKMLXGemma4ImageProcessor(patchSize: 16, poolingKernelSize: 1, maxSoftTokens: 9).process(image)
        XCTAssertEqual(pixels.shape, [9, 768])
        XCTAssertEqual(positions.shape, [9, 2])

        // Raw audio → log-mel features [frames, 128].
        let audio = (0 ..< 4000).map { sinf(Float($0) * 0.05) }
        let features = NFKMLXGemma4AudioFeatureExtractor().features(audio)
        XCTAssertEqual(features.shape[1], 128)

        // A tower's soft tokens → decoder space, then spliced at placeholder positions.
        let embedder = NFKMLXGemma4MultimodalEmbedder(multimodalHidden: 8, textHidden: 16)
        let projected = embedder(MLXArray((0 ..< 2 * 8).map { Float($0) }).reshaped([2, 8]))
        XCTAssertEqual(projected.shape, [2, 16])
        let fused = NFKMLXGemma4Fusion.fuse(textEmbeddings: MLXArray.zeros([1, 3, 16]),
                                            softTokens: projected, isPlaceholder: [true, false, true])
        XCTAssertEqual(fused.shape, [1, 3, 16])
    }

    // Docs/examples.md: Constrained decoding — a grammar mask keeps the output well-formed JSON, or
    // one of a fixed set of answers. The vocabulary's bytes come from the release's tokenizer; a
    // hand-built one stands in here.
    func testExampleConstrainedDecoding() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "runs the decoder; run via xcodebuild")
        let size = NFKMLXLanguageConfiguration.tiny.vocabularySize
        var tokens = (0 ..< 256).map { [UInt8($0)] } + ["{\"", "\":", "yes", "no", "}", "42"].map { Array($0.utf8) }
        while tokens.count < size - 1 { tokens.append(Array("t\(tokens.count)".utf8)) }
        tokens.append([])                                              // the end token has no bytes
        let vocabulary = NFKMLXVocabulary(tokens: tokens, endToken: size - 1)
        // With a release: NFKMLXVocabulary(tokenizer: tokenizer, size: configuration.vocabularySize)

        let model = NFKMLXLanguage.makeNet(.tiny)
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 40
        options.constraint = NFKMLXJSONConstraint(vocabulary: vocabulary, root: .object)
        let json = model.generate(prompt: [3, 17], options: options)
        let text = String(decoding: json.flatMap { vocabulary.tokens[$0] }, as: UTF8.self)
        XCTAssertTrue(text.hasPrefix("{") || text.trimmingCharacters(in: .whitespaces).hasPrefix("{"))

        options.constraint = NFKMLXChoiceConstraint(choices: ["yes", "no"], vocabulary: vocabulary)
        let answer = model.generate(prompt: [3, 17], options: options)
        XCTAssertTrue(["yes", "no"].contains(String(decoding: answer.flatMap { vocabulary.tokens[$0] }, as: UTF8.self)))
    }

    // Docs/examples.md: Text embeddings — the decoder read one layer earlier, pooled and normalized to
    // one vector per text, so a dot product is a cosine similarity. The released Qwen3-Embedding loads
    // from its directory (NFKMLXQwen3Embedding.backend(directoryURL:)); a tiny random backbone here
    // exercises the path. A query is embedded with a task instruction, a document without one.
    func testExampleTextEmbeddingSimilarity() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "runs the embedder; run via xcodebuild")
        let backend = try NFKMLXQwen3Embedding.backend(weightsURL: nil, tokenizer: nil,
                                                       configuration: .tiny) as! NFKMLXTextEmbeddingBackend

        // A real backend tokenizes text through runInference; with no tokenizer, embed token ids.
        func embedding(_ tokens: [Int]) -> [Float] {
            backend.embedding(forTokens: tokens.map { NSNumber(value: $0) }).map(\.floatValue)
        }
        func cosine(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).map(*).reduce(0, +) }

        let query = embedding([12, 45, 7, 88])          // "Instruct: …\nQuery:…" tokenized, in practice
        let related = embedding([12, 45, 7, 90])
        let unrelated = embedding([3, 3, 3, 3])
        // Every embedding is unit length, so the dot product is bounded by 1.
        XCTAssertEqual(cosine(query, query), 1, accuracy: 1e-4)
        XCTAssertLessThanOrEqual(cosine(query, related), 1)
        XCTAssertLessThanOrEqual(cosine(query, unrelated), 1)

        // Format a retrieval query the way the model is trained to read it.
        let prompt = NFKMLXQwen3Embedding.instruct(task: "Retrieve passages that answer the query",
                                                   query: "What is the capital of France?")
        XCTAssertTrue(prompt.hasPrefix("Instruct: "))
    }

    // Docs/examples.md: Local generation — quantize the cache for a longer conversation, chunk a long
    // prompt to bound the prefill peak, and apply a chat template so an instruct release is prompted
    // in its trained format. All three are off by default.
    func testExampleGenerationOptionsForMemoryAndInstructModels() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "runs the decoder; run via xcodebuild")
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 4
        options.cacheQuantization = .init(bits: 8, groupSize: 64)   // 8-bit KV; groupSize divides the head dim
        options.prefillChunkSize = 4                                // run a long prompt through the cache in slices
        options.chatTemplate = .chatML                             // render instruct turns (a message list only)

        // A head dimension the cache group size divides.
        let config = NFKMLXLanguageConfiguration(hiddenSize: 128, layerCount: 2, headCount: 2,
                                                 keyValueHeadCount: 1, headDimensions: 64,
                                                 intermediateSize: 128, vocabularySize: 512, ropeTheta: 10_000)
        let produced = NFKMLXLanguage.makeNet(config).generate(prompt: [3, 17, 42, 8, 91, 5], options: options)
        XCTAssertLessThanOrEqual(produced.count, options.maxTokens)

        // The chat template renders roles and a trailing assistant turn for a message list.
        let rendered = NFKMLXLanguageBackend.chatMLPrompt(from: [["role": "user", "content": "Hi"]])
        XCTAssertTrue(rendered.contains("<|im_start|>user\nHi<|im_end|>"))

        // Every one is off by default, so an existing caller is unchanged.
        let defaults = NFKMLXGenerationOptions()
        XCTAssertNil(defaults.cacheQuantization)
        XCTAssertNil(defaults.prefillChunkSize)
        if case .none = defaults.chatTemplate {} else { XCTFail("the default template flattens contents") }
    }

    // Docs/examples.md: The release's own Jinja chat template renders a message list into the exact
    // input the instruct model was trained on, rather than the ChatML approximation. Pass the template
    // text (from the release's tokenizer_config.json) as a generation option. No runtime needed here —
    // the renderer is pure Foundation.
    func testExampleRenderingTheReleaseChatTemplate() throws {
        let template = """
        {%- for message in messages %}\
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}\
        {%- endfor %}\
        {%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}
        """
        let messages: [[String: Any]] = [["role": "user", "content": "What is 2+2?"]]
        let prompt = try NFKMLXChatTemplateRenderer.render(template, messages: messages,
                                                           addGenerationPrompt: true)
        XCTAssertEqual(prompt, "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n")

        // As a generation option, so the backend renders it before generating.
        var options = NFKMLXGenerationOptions()
        options.chatTemplate = .jinja(template: template)
        XCTAssertEqual(NFKMLXLanguageBackend.prompt(from: NFKInferenceRequest(inputs: [NFKInputMessages: messages]),
                                                    template: options.chatTemplate), prompt)
    }

    // Docs/examples.md: A release too large for the machine is refused before it is materialized.
    func testExampleAReleaseLargerThanMemoryIsRefusedBeforeLoading() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(count: 2_000_000).write(to: directory.appendingPathComponent("model.safetensors"))

        // A tiny budget stands in for a machine the release does not fit; the error names the shortfall.
        XCTAssertThrowsError(try NFKMLXReleaseWeights.verifyFits(inDirectory: directory,
                                                                 precision: .checkpoint, budget: 1_000_000))
        XCTAssertNoThrow(try NFKMLXReleaseWeights.verifyFits(inDirectory: directory,
                                                             precision: .checkpoint, budget: 4_000_000))
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

    // Docs/examples.md: Reading a GGUF model. The reader opens the container, reads its metadata and
    // tensor descriptors, and dequantizes a tensor to an MLXArray. A tiny F32 file stands in for a real
    // quantized model here; the released Q4_K_M dequantizes bit-exactly against the gguf package.
    func testExampleReadsAGGUFModelWithNoPython() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "reads an MLXArray; run via xcodebuild")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".gguf")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(MLXExamples.minimalGGUF()).write(to: url)

        let gguf = try NFKMLXGGUF.gguf(contentsOf: url)
        XCTAssertEqual(gguf.metadataString(forKey: "general.architecture"), "example")
        XCTAssertEqual(gguf.info(forTensor: "weight")?.typeName, "F32")
        let weight = try XCTUnwrap(gguf.array(forTensor: "weight"))
        XCTAssertEqual(weight.asArray(Float.self), [0.5, -0.5, 1.5])
    }

    /// A minimal GGUF file: one F32 tensor `weight` and a string metadata value.
    private static func minimalGGUF() -> [UInt8] {
        var bytes = [UInt8]()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
        func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) } }
        func str(_ s: String) { let b = Array(s.utf8); u64(UInt64(b.count)); bytes.append(contentsOf: b) }
        func f32(_ v: Float) { withUnsafeBytes(of: v.bitPattern.littleEndian) { bytes.append(contentsOf: $0) } }
        u32(0x4655_4747); u32(3); u64(1); u64(1)
        str("general.architecture"); u32(8); str("example")
        str("weight"); u32(1); u64(3); u32(0); u64(0)
        while bytes.count % 32 != 0 { bytes.append(0) }
        f32(0.5); f32(-0.5); f32(1.5)
        return bytes
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
