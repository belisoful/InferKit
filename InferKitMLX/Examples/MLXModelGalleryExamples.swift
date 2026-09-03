//
//  MLXModelGalleryExamples.swift
//  InferKitMLXExamples
//
//  A live example of every shipped MLX model, built through its public `@objc` factory (the primary
//  Objective-C path — no registry). Each group mirrors a "Model gallery" entry in Docs/examples.md.
//  Constructing a real model builds MLXNN layers (initializes MLX) and running evaluates arrays, so the
//  build/run checks skip under `swift test` and run under `xcodebuild test`. Exhaustive per-model
//  forwards live in the individual NFKMLX*Tests; these show the consumer-facing factory call and a
//  representative run per modality.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class MLXModelGalleryExamples: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "building a real model initializes MLX; run via xcodebuild")
    }

    // MARK: Upscaling & restoration (image → image, module backend)

    func testUpscalingAndRestorationModels() throws {
        try requireMLXRuntime()
        // Each factory builds a real network; nil weights → random weights, ready to run.
        let realESRGAN = try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: nil)
        let swinIR = try NFKMLXSwinIR.backend(weightsURL: nil)
        let nafnet = try NFKMLXNAFNet.backend(weightsURL: nil)
        let zeroDCE = try NFKMLXZeroDCE.backend(weightsURL: nil)
        let styleTransfer = try NFKMLXStyleTransfer.backend(weightsURL: nil)
        let colorizer = try NFKMLXColorizer.backend(weightsURL: nil)
        let codeFormer = try NFKMLXCodeFormer.backend(weightsURL: nil)

        XCTAssertEqual(realESRGAN.backendIdentifier, "real-esrgan-x4")
        XCTAssertEqual(swinIR.backendIdentifier, "swinir-x4")
        for backend in [nafnet, zeroDCE, styleTransfer, colorizer, codeFormer] {
            XCTAssertTrue(backend.isReady)
        }

        // Representative run: Zero-DCE brightens a dark frame to a same-size image.
        let result = try zeroDCE.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(32, value: 40)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage))
    }

    // MARK: Depth (image → grayscale depth)

    func testDepthModels() throws {
        try requireMLXRuntime()
        let depthAnything = try NFKMLXDepthAnything.backend(variant: .small, weightsURL: nil)
        XCTAssertEqual(depthAnything.backendIdentifier, "depth-anything-v2-small")
        let result = try depthAnything.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage), "grayscale depth map")
    }

    // MARK: Matting (plate → foreground + alpha, matting backend)

    func testMattingModels() throws {
        try requireMLXRuntime()
        let u2net = try NFKMLXU2Net.backend(variant: .full, weightsURL: nil)
        let rvm = try NFKMLXRVM.backend(weightsURL: nil)
        let modnet = try NFKMLXMODNet.backend(weightsURL: nil)
        for backend in [u2net, rvm, modnet] {
            XCTAssertTrue(backend.isReady)
        }
        // Representative run: MODNet portrait matte → foreground image + a separate matte.
        let result = try modnet.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(32)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage))
        XCTAssertNotNil(result.output(forKey: NFKOutputMask), "the alpha matte comes out on its own")
    }

    // MARK: Semantic segmentation (image → grayscale class-label map)

    func testSegmentationModels() throws {
        try requireMLXRuntime()
        let segformer = try NFKMLXSegFormer.backend(weightsURL: nil)
        let deeplab = try NFKMLXDeepLab.backend(weightsURL: nil)
        let bisenet = try NFKMLXBiSeNet.backend(weightsURL: nil)
        for backend in [segformer, deeplab, bisenet] {
            XCTAssertTrue(backend.isReady)
        }
        // Recover a class index from the grayscale label map as round(gray · (classCount − 1)).
        let result = try bisenet.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage), "label map")
    }

    // MARK: Detection & pose (new core value types)

    func testDetectionAndPoseModels() throws {
        try requireMLXRuntime()
        // YOLO returns NFKDetection boxes under NFKOutputDetections; labels attach class names.
        let yolo = try NFKMLXYOLO.backend(weightsURL: nil, labels: nil)
        let detected = try yolo.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(32)]))
        XCTAssertNotNil(detected.detections, "detections (possibly empty)")

        // Pose returns NFKKeypoint joints under NFKOutputPose; positions are normalized 0…1.
        let pose = try NFKMLXPose.backend(weightsURL: nil, jointNames: nil)
        let estimated = try pose.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(48)]))
        XCTAssertEqual(estimated.pose?.count, 17, "17 COCO joints")
    }

    // MARK: Embeddings (image+text → shared space)

    func testCLIPEmbeddings() throws {
        try requireMLXRuntime()
        let clip = try NFKMLXCLIP.backend(weightsURL: nil)
        let result = try clip.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(32)]))
        let embedding = try XCTUnwrap(result.embedding, "an L2-normalized image embedding")
        XCTAssertEqual(embedding.count, 512, "ViT-B/32 embedding width")

        // SigLIP 2: the CLIP upgrade — image embedding via the attention-pooling head.
        let siglip2 = try NFKMLXSigLIP2.backend(weightsURL: nil)
        let siglip2Result = try siglip2.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(224)]))
        XCTAssertNotNil(siglip2Result.output(forKey: NFKOutputEmbedding), "a SigLIP 2 image embedding")
    }

    func testTextEmbeddings() throws {
        try requireMLXRuntime()
        // The released embedders load from their directories: NFKMLXQwen3Embedding.backend(directoryURL:)
        // (last-token pooled decoder) and NFKMLXEmbeddingGemma.backend(directoryURL:) (mean-pooled
        // bidirectional encoder). Here tiny random backbones exercise the modality without a download;
        // the caller tokenizes and reads the embedding through embedding(forTokens:).
        let qwen3 = try XCTUnwrap(try NFKMLXQwen3Embedding.backend(weightsURL: nil, tokenizer: nil,
                                  configuration: .tiny) as? NFKMLXTextEmbeddingBackend)
        let gemma = try XCTUnwrap(try NFKMLXEmbeddingGemma.backend(weightsURL: nil, dense2URL: nil,
                                  dense3URL: nil, tokenizer: nil, configuration: .tiny) as? NFKMLXTextEmbeddingBackend)
        for embedder in [qwen3, gemma] {
            let embedding = embedder.embedding(forTokens: [3, 17, 42, 5].map { NSNumber(value: $0) })
            XCTAssertEqual(embedder.embeddingDimensions, embedding.count, "one vector per input")
        }
    }

    func testVisionLanguage() throws {
        try requireMLXRuntime()
        // The released SmolVLM2 loads from its directory: NFKMLXSmolVLM.load(directoryURL:), then
        // model.answer(image:question:). Here the vision encoder, connector, and image processor run on
        // tiny random / synthetic inputs to exercise the pipeline without a download.
        NFKMLXRandom.seed(3)
        let vision = NFKMLXSigLIPNet(.tiny)
        let connector = NFKMLXSmolVLMConnector(visionHidden: 32, decoderHidden: 48, scaleFactor: 4)
        let features = connector(vision(MLXRandom.normal([1, 64, 64, 3])))
        XCTAssertEqual(features.shape, [1, 1, 48], "16 patches shuffle to one token at the decoder width")

        let (pixels, rows, cols) = NFKMLXSmolVLMImageProcessor.process(Self.solid(300))
        XCTAssertEqual([rows, cols], [4, 4])
        XCTAssertEqual(pixels.shape, [17, 3, 512, 512])
    }

    func testQwen3VLVisionTower() throws {
        try requireMLXRuntime()
        // The released Qwen3-VL vision tower loads from its directory:
        // NFKMLXQwen3VL.visionNet(directoryURL:). Here a tiny random tower folds a 4×4 patch grid into
        // 4 tokens and produces the deepstack features the decoder injects.
        NFKMLXRandom.seed(1)
        let config = NFKMLXQwen3VLVisionConfiguration(
            hiddenSize: 32, depth: 4, headCount: 2, intermediateSize: 64, patchSize: 2, temporalPatchSize: 2,
            spatialMergeSize: 2, outHiddenSize: 16, positionGridSide: 4, deepstackLayers: [1, 2])
        let net = NFKMLXQwen3VLVisionNet(config)
        let (output, deepstack) = net(MLXRandom.normal([16, 24]), grid: (t: 1, h: 4, w: 4))
        eval(output)
        XCTAssertEqual(output.shape, [4, 16])
        XCTAssertEqual(deepstack.count, 2)
    }

    func testReranking() throws {
        try requireMLXRuntime()
        // The released reranker loads from its directory: NFKMLXModernBERTReranker.reranker(directoryURL:),
        // then reranker.rankedIndices(query:documents:) / scores(query:documents:) score a query against
        // each candidate and order them. Here a tiny random net stands in without a download.
        NFKMLXRandom.seed(2)
        let reranker = try NFKMLXModernBERTReranker.reranker(weightsURL: nil, tokenizer: nil,
                                                             configuration: .tiny)
        XCTAssertEqual(NFKMLXModernBERTReranker.modelName, "gte-reranker-modernbert-base")
        // With no tokenizer the request path returns a neutral 0; a release directory supplies the
        // byte-level BPE tokenizer that makes the score meaningful.
        XCTAssertTrue(reranker.score(query: "a query", document: "a candidate document").isFinite)
    }

    // MARK: Video (frame pair / recurrent, tensor & module backends)

    func testVideoModels() throws {
        try requireMLXRuntime()
        // RIFE / RAFT take two frames under frame0 / frame1; VideoSR upscales single frames or a clip.
        let rife = try NFKMLXRIFE.backend(weightsURL: nil)
        let raft = try NFKMLXRAFT.backend(weightsURL: nil)
        let videoSR = try NFKMLXVideoSR.backend(weightsURL: nil)
        for backend in [rife, raft, videoSR] {
            XCTAssertTrue(backend.isReady)
        }
        let result = try videoSR.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(16)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage), "×4 upscaled frame")
    }

    // MARK: Text → image (built by factory)

    // The tiny configuration keeps the example fast; a release directory is what makes it a picture.
    func testTextToImage() throws {
        try requireMLXRuntime()
        let backend = NFKMLXTextToImage.backend(configuration: .tiny)
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a watercolor lighthouse at dawn"],
                                          parameters: [NFKParameterSteps: 2])
        XCTAssertNotNil(try backend.runInference(for: request).output(forKey: NFKOutputImage))
    }

    // MARK: Inpainting & latent diffusion (built by factory)

    func testInpaintingAndDiffusionModels() throws {
        try requireMLXRuntime()
        let lama = try NFKMLXLaMa.backend(weightsURL: nil)                 // plate + mask → inpainted
        let sdInpaint = try NFKMLXStableDiffusionInpaint.backend(weightsURL: nil)
        let marigold = try NFKMLXMarigold.backend(weightsURL: nil)
        let sdUpscaler = try NFKMLXSDUpscaler.backend(weightsURL: nil)
        for backend in [lama, sdInpaint, marigold, sdUpscaler] {
            XCTAssertTrue(backend.isReady)
        }
    }

    // MARK: LCM few-step sampling & ControlNet conditioning (no SD reimplementation)

    func testLCMSchedulerAndControlNet() throws {
        try requireMLXRuntime()
        // LCM is a drop-in scheduler for the diffusion backend — few steps, no model reimplementation.
        // A real integration supplies its own encode/denoise/decode (or a dynamically linked SD engine).
        let fastDiffusion = NFKMLXDiffusionBackend(
            identifier: "my-lcm",
            configuration: NFKDiffusionConfiguration(steps: 4),
            scheduler: NFKLCMScheduler(predictionType: .epsilon),
            encode: { _, _, _ in NFKDiffusionContext(width: 64, height: 64) },
            denoise: { latent, _, _, _ in latent },
            decode: { clip($0, min: 0, max: 1) })
        XCTAssertTrue(fastDiffusion.isReady)

        // ControlNet conditioning flows through conditioning["control"]; the reference shows the wiring.
        NFKMLXReferenceModels.registerControlNet()
        let controlNet = try NFKMLXModelRegistry.backend(named: "diffusion-controlnet", weightsURL: nil)
        let result = try controlNet.runInference(for: NFKInferenceRequest(inputs: [NFKInputControl: Self.solid(32)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage), "generation guided by the control map")
    }

    // MARK: Promptable segmentation (SAM)

    func testSegmentAnything() throws {
        try requireMLXRuntime()
        let sam = try NFKMLXSAM.backend(weightsURL: nil)
        XCTAssertEqual(sam.backendIdentifier, "sam")                      // plate + point under NFKSAMPointKey → mask
    }

    // MARK: Video (clip → clip)

    // The video modality: a backend that reads an NFKVideoAsset and returns one. The transform sees
    // the whole frame sequence, because interpolation returns more frames than it took and BasicVSR
    // propagates state through time — neither is a per-frame map.
    func testVideoClipBackends() throws {
        try requireMLXRuntime()

        // Frame interpolation: n frames become 2n - 1, written at twice the source rate so the clip
        // plays smoother rather than slower.
        let interpolator = try NFKMLXRIFE.clipBackend(weightsURL: nil)
        XCTAssertEqual(interpolator.backendIdentifier, "rife-clip")

        // Video super-resolution: ×4, same rate, propagation both directions through the clip.
        let upscaler = try NFKMLXVideoSR.clipBackend(weightsURL: nil)
        XCTAssertEqual(upscaler.backendIdentifier, "video-super-resolution-clip")

        // Bring your own: any [MLXArray] -> [MLXArray] over frames in 0...1.
        let custom = NFKMLXVideoBackend(identifier: "half-speed") { frames in
            frames.flatMap { [$0, $0] }
        }
        XCTAssertTrue(custom.isReady)
    }

    // MARK: Faces in a photograph (detect → align → restore → composite)

    func testFaceRestorationOnAPhotograph() throws {
        try requireMLXRuntime()
        // The model restores an ALIGNED 512×512 crop; this backend finds the faces itself. The
        // detector defaults to RetinaFace, the one the reference pipeline runs, so the crop matches
        // the reference's. Pass NFKMLXVisionFaceDetector() instead for a download-free path.
        let photo = try NFKMLXCodeFormer.photoBackend(fidelity: 0.5, weightsURL: nil,
                                                      detectorWeightsURL: nil)
        XCTAssertEqual(photo.backendIdentifier, "codeformer-photo")

        // The alignment is a similarity transform onto the reference's five-point template, so it
        // never shears the face.
        let template = NFKMLXFaceAlignment.template512
        let transform = try XCTUnwrap(NFKMLXFaceAlignment.similarityTransform(from: template, to: template))
        XCTAssertEqual(transform.a, 1, accuracy: 1e-9)
        XCTAssertEqual(transform.b, 0, accuracy: 1e-9)
    }

    // MARK: Audio (audio → text / stems / speakers / clean / segments / tags)

    func testAudioModels() throws {
        try requireMLXRuntime()
        let wave = NFKMLXWaveFile.data(samples: Self.tone(16000), sampleRate: 16000)

        let whisper = try NFKMLXWhisper.backend(weightsURL: nil)
        XCTAssertEqual(whisper.backendIdentifier, "whisper-tiny")

        // Asking for timestamps gives the spans as well as the words, as NFKAudioSegments beside the
        // transcript. It is a different decode, so it is asked for rather than always produced.
        let timedWhisper = try NFKMLXWhisper.backend(weightsURL: nil, tokenizer: nil, timestamps: true)
        let timed = try timedWhisper.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        XCTAssertNotNil(timed.segments)
        XCTAssertNotNil(timed.text)

        // Conv-TasNet separates a mixture into one NFKAudioAsset per speaker.
        let tasnet = try NFKMLXConvTasNet.backend(weightsURL: nil)
        let separated = try tasnet.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        XCTAssertNotNil(separated.output(forKey: "speaker-1"))

        // Denoiser cleans a noisy clip; VAD marks speech spans; the tagger names sounds.
        let denoiser = try NFKMLXDenoiser.backend(weightsURL: nil)
        XCTAssertNotNil(try denoiser.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        let vad = try NFKMLXVAD.backend(weightsURL: nil)
        XCTAssertNotNil(try vad.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).segments)

        // Silero VAD v6: a streaming STFT + conv + LSTM model, one speech probability per 512-sample chunk.
        let silero = try NFKMLXSileroVAD.backend(weightsURL: nil)
        XCTAssertNotNil(try silero.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).segments)

        // DAC: a neural audio codec. The backend reconstructs audio → codes → audio; NFKMLXDAC.encode
        // returns the codebook tokens themselves, which is what a codec-token speech-LLM generates.
        let dac = try NFKMLXDAC.backend(weightsURL: nil)
        XCTAssertNotNil(try dac.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // SNAC: a multi-scale codec — its codebooks emit token streams at different temporal rates.
        let snac = try NFKMLXSNAC.backend(weightsURL: nil)
        XCTAssertNotNil(try snac.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        let tagger = try NFKMLXAudioTagger.backend(weightsURL: nil, labels: nil)
        XCTAssertNotNil(try tagger.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).classifications)

        let demucs = try NFKMLXDemucs.backend(weightsURL: nil)
        XCTAssertEqual(demucs.backendIdentifier, "demucs")

        // Demucs v4: parallel spectrogram and waveform branches joined by a cross-transformer.
        let htdemucs = try NFKMLXHTDemucs.backend(weightsURL: nil)
        XCTAssertEqual(htdemucs.backendIdentifier, "htdemucs")

        // Full text-to-speech chain: phonemizer → acoustic (FastSpeech2-style) → vocoder (HiFi-GAN).
        let tts = NFKMLXTTS(phonemizer: NFKMLXNeuralG2P(), symbols: (0 ..< 40).map { "p\($0)" })
        let speech = tts.makeSpeechBackend()                     // reads NFKInputPrompt, writes a WAV NFKAudioAsset
        XCTAssertEqual(speech.backendIdentifier, "tts")

        // MiniMax Music 3: a music description under NFKInputPrompt and lyrics under NFKInputLyrics
        // become a stereo 44.1 kHz clip. The stack is 27 GB of separately licensed weights, so the
        // factory takes the downloaded release DIRECTORY and there is no random-weights form;
        // isReady reports whether the weights are present rather than failing the build.
        let music = try NFKMLXMusic3.backend(directoryURL:
            FileManager.default.temporaryDirectory.appendingPathComponent("minimax-music3"))
        XCTAssertEqual(music.backendIdentifier, "minimax-music3")
        XCTAssertFalse(music.isReady, "no weights at that path yet — download the release first")
    }

    // MARK: Dynamic discovery (Stable Diffusion / transcription activate when linked)

    func testDynamicCapabilitiesActivateWithInferKitMLXLinked() throws {
        // No MLX needed: pure runtime class lookup. Both providers ship in this package.
        XCTAssertTrue(NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityStableDiffusion))
        XCTAssertTrue(NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityTranscription))
        let sd = try NFKDynamicBackend.stableDiffusionBackend()
        XCTAssertEqual(sd.backendIdentifier, "mlx-stable-diffusion")
    }

    // MARK: Helpers

    private static func solid(_ side: Int, value: UInt8 = 128) -> CGImage {
        let pixels = [UInt8](repeating: value, count: side * side * 4)
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    private static func tone(_ samples: Int) -> [Float] {
        (0 ..< samples).map { sinf(Float($0) * 0.05) * 0.3 }
    }
}
