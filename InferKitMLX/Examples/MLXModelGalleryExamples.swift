//
//  MLXModelGalleryExamples.swift
//  InferKitMLXExamples
//
//  A live example of every shipped MLX model, built through its public `@objc` factory (the primary
//  Objective-C path — no registry). Each group mirrors a "Model gallery" entry in Docs/examples.md.
//  Constructing a real model builds MLXNN layers (initializes MLX) and running evaluates arrays, so
//  the build/run checks skip without a Metal library for MLX (see Tools/mlx-metallib.sh).
//  Exhaustive per-model forwards live in the individual NFKMLX*Tests; these show the
//  consumer-facing factory call and a representative run per modality.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class MLXModelGalleryExamples: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "building a real model initializes MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    // MARK: Upscaling & restoration (image → image, module backend)

    func testUpscalingAndRestorationModels() throws {
        try requireMLXRuntime()
        // Each factory builds a real network; nil weights → random weights, ready to run.
        let realESRGAN = try NFKMLXRealESRGAN.backend(variant: .x4, weightsURL: nil)
        let swinIR = try NFKMLXSwinIR.backend(weightsURL: nil)
        let nafnet = try NFKMLXNAFNet.backend(weightsURL: nil)
        let zeroDCE = try NFKMLXZeroDCE.backend(weightsURL: nil)
        // Zero-DCE++ is the authors' successor: separable convolutions and one shared curve.
        let zeroDCEPlus = try NFKMLXZeroDCEPlus.backend(weightsURL: nil)
        let styleTransfer = try NFKMLXStyleTransfer.backend(weightsURL: nil)
        // AdaIN stylizes with any style image rather than a baked-in style.
        let adain = try NFKMLXAdaIN.backend(encoderURL: nil, decoderURL: nil)
        let colorizer = try NFKMLXColorizer.backend(weightsURL: nil)
        // DDColor is the modern colorizer: a ConvNeXt encoder under learned color queries.
        let ddcolor = try NFKMLXDDColor.backend(variant: .modelscope, weightsURL: nil)
        let codeFormer = try NFKMLXCodeFormer.backend(weightsURL: nil)

        XCTAssertEqual(realESRGAN.backendIdentifier, "real-esrgan-x4")
        XCTAssertEqual(swinIR.backendIdentifier, "swinir-x4")
        for backend in [nafnet, zeroDCE, zeroDCEPlus, styleTransfer, adain, colorizer, codeFormer] {
            XCTAssertTrue(backend.isReady)
        }
        XCTAssertEqual(adain.backendIdentifier, "adain")
        XCTAssertEqual(ddcolor.backendIdentifier, "ddcolor")

        // Every released SwinIR and NAFNet fits its own variant: the lightweight ×3 / ×4, the classical
        // ×2, the real-world ×4 (nearest-neighbor tail; the large one with the 3-conv residual), and
        // NAFNet's width-64 SIDD and GoPro.
        let lightX3 = try NFKMLXSwinIR.backend(variant: .lightweightSRX3, weightsURL: nil)
        let realWorld = try NFKMLXSwinIR.backend(variant: .realWorldX4Medium, weightsURL: nil)
        let wide = try NFKMLXNAFNet.backend(variant: .siddWidth64, weightsURL: nil)
        // Real-ESRGAN's later releases run the compact generator behind the same variant enum.
        let compact = try NFKMLXRealESRGAN.backend(variant: .generalX4V3, weightsURL: nil)
        for backend in [lightX3, realWorld, wide, compact] {
            XCTAssertTrue(backend.isReady)
        }

        // HAT is SwinIR's successor: window attention plus a channel-attention branch in every block
        // and an overlapping cross-attention block closing every group.
        let hat = try NFKMLXHAT.backend(variant: .large, weightsURL: nil)
        XCTAssertEqual(hat.backendIdentifier, "hat-l-x4")

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

        let depth3 = try NFKMLXDepthAnything3.backend(weightsURL: nil)
        XCTAssertEqual(depth3.backendIdentifier, "depth-anything-3-small")
        let result3 = try depth3.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64)]))
        XCTAssertNotNil(result3.output(forKey: NFKOutputImage), "grayscale depth map (DA3)")

        // DA3-BASE and DA3-LARGE are the same network at their own widths, through the variant factory.
        let depth3Base = try NFKMLXDepthAnything3.backend(variant: .base, weightsURL: nil)
        // Depth Anything 3 also predicts the camera and a ray map, which a single-image backend cannot
        // carry; `NFKMLXDepth3Estimator` is the way to them.
        let estimator = try NFKMLXDepth3Estimator.estimator(variant: .small, weightsURL: nil)
        let camera = try estimator.camera(for: Self.solid(64))
        XCTAssertEqual(camera.translation.count, 3)
        XCTAssertEqual(camera.rotation.count, 9)
        XCTAssertGreaterThan(camera.focalLengthX, 0)
        XCTAssertEqual(depth3Base.backendIdentifier, "depth-anything-3-base")
    }

    // MARK: Matting (plate → foreground + alpha, matting backend)

    func testMattingModels() throws {
        try requireMLXRuntime()
        let u2net = try NFKMLXU2Net.backend(variant: .full, weightsURL: nil)
        // IS-Net is U²-Net's successor: the same Residual U-blocks behind a stride-2 stem.
        let isnet = try NFKMLXISNet.backend(weightsURL: nil)
        XCTAssertEqual(isnet.backendIdentifier, "isnet")
        let rvm = try NFKMLXRVM.backend(weightsURL: nil)
        // RVM's heavier release swaps MobileNetV3 for a ResNet-50 encoder under the same contract.
        let rvmResNet = try NFKMLXRVM.backend(variant: .resNet50, weightsURL: nil)
        XCTAssertEqual(rvmResNet.backendIdentifier, "robust-video-matting-resnet50")
        let modnet = try NFKMLXMODNet.backend(weightsURL: nil)
        // BiRefNet: high-resolution background removal (MIT), the same foreground + alpha contract.
        let birefnet = try NFKMLXBiRefNet.backend(weightsURL: nil)
        // SAM 2.1: promptable segmentation from a click, which the matting contract carries as alpha.
        let sam2 = try NFKMLXSAM2.backend(variant: .tiny, release: .sam21, weightsURL: nil)
        XCTAssertEqual(sam2.backendIdentifier, "sam2")
        for backend in [u2net, rvm, rvmResNet, modnet, birefnet, sam2] {
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

        // The generations after v8 share one graph interpreter. YOLO11 keeps the classic head, so it
        // runs suppression; YOLO26 and YOLOv10 predict from a one-to-one branch and need none.
        let yolo11 = try NFKMLXYOLOGenerations.backend(release: .v11Nano, weightsURL: nil, labels: nil)
        XCTAssertEqual(yolo11.backendIdentifier, "yolo11n")
        let yolo26 = try NFKMLXYOLOGenerations.backend(release: .v26Nano, weightsURL: nil, labels: nil)
        XCTAssertEqual(yolo26.backendIdentifier, "yolo26n")
        let latest = try yolo26.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64)]))
        XCTAssertNotNil(latest.detections, "detections (possibly empty)")
        for release in [NFKMLXYOLORelease.v9Tiny, .v9Extended, .v10Nano, .v12Nano] {
            let backend = try NFKMLXYOLOGenerations.backend(release: release, weightsURL: nil, labels: nil)
            XCTAssertTrue(backend.isReady)
        }

        // RT-DETR: the license-clean (Apache-2.0) detector — same NFKOutputDetections contract, no NMS.
        let rtdetr = try NFKMLXRTDetr.backend(weightsURL: nil, labels: nil)
        let rtDetected = try rtdetr.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64)]))
        XCTAssertNotNil(rtDetected.detections, "detections (possibly empty)")

        // RF-DETR: the Roboflow windowed-DINOv2 / Group-DETR detector — same contract, no NMS.
        let rfdetr = try NFKMLXRFDetr.backend(weightsURL: nil, labels: nil)
        let rfDetected = try rfdetr.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64)]))
        XCTAssertNotNil(rfDetected.detections, "detections (possibly empty)")

        // Every released size of both detectors has a variant: RT-DETR's basic-block r18vd / r34vd and
        // r101vd, RF-DETR's nano / small / medium / large.
        let rtdetrSmall = try NFKMLXRTDetr.backend(variant: .r18vd, weightsURL: nil, labels: nil)
        XCTAssertEqual(rtdetrSmall.backendIdentifier, "rtdetr-r18vd")
        let rfdetrNano = try NFKMLXRFDetr.backend(variant: .nano, weightsURL: nil, labels: nil)
        XCTAssertEqual(rfdetrNano.backendIdentifier, "rf-detr-nano")

        // Pose returns NFKKeypoint joints under NFKOutputPose; positions are normalized 0…1.
        let pose = try NFKMLXPose.backend(weightsURL: nil, jointNames: nil)
        let estimated = try pose.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(48)]))
        XCTAssertEqual(estimated.pose?.count, 17, "17 COCO joints")

        // ViTPose is the modern pose model: a ViT backbone under a small decoding head, decoded with
        // DARK rather than the classic quarter-cell shift.
        let vitPose = try NFKMLXVitPose.backend(weightsURL: nil, jointNames: nil)
        XCTAssertEqual(vitPose.backendIdentifier, "vitpose-base-simple")
        let vitEstimated = try vitPose.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(48)]))
        XCTAssertEqual(vitEstimated.pose?.count, 17, "17 COCO joints")
        let vitPoseClassic = try NFKMLXVitPose.backend(variant: .base, weightsURL: nil, jointNames: nil)
        XCTAssertEqual(vitPoseClassic.backendIdentifier, "vitpose-base")
    }

    // MARK: Embeddings (image+text → shared space)

    func testCLIPEmbeddings() throws {
        try requireMLXRuntime()
        let clip = try NFKMLXCLIP.backend(weightsURL: nil)
        let result = try clip.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(32)]))
        let embedding = try XCTUnwrap(result.embedding, "an L2-normalized image embedding")
        XCTAssertEqual(embedding.count, 512, "ViT-B/32 embedding width")

        // The other vision-transformer releases fit their own variants: ViT-B/16, ViT-L/14, and
        // ViT-L/14@336px, the last two embedding at 768.
        let clipL14 = try NFKMLXCLIP.backend(variant: .vitL14, weightsURL: nil)
        XCTAssertEqual(clipL14.backendIdentifier, "clip-vit-l-14")

        // SigLIP 2: the CLIP upgrade — image embedding via the attention-pooling head.
        let siglip2 = try NFKMLXSigLIP2.backend(weightsURL: nil)
        let siglip2Result = try siglip2.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(224)]))
        XCTAssertNotNil(siglip2Result.output(forKey: NFKOutputEmbedding), "a SigLIP 2 image embedding")

        // TAESD: the tiny autoencoder — image → latent → image reconstruction.
        let taesd = try NFKMLXTAESD.backend(weightsURL: nil)
        let taesdResult = try taesd.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64)]))
        XCTAssertNotNil(taesdResult.output(forKey: NFKOutputImage), "a TAESD reconstruction")
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

    func testGemma3() throws {
        try requireMLXRuntime()
        // The released Gemma 3 loads from its directory: NFKMLXGemma3.backend(directoryURL:) for text
        // (NFKInputPrompt / NFKInputMessages, an NFKInputImage beside them on the multimodal 4B), or
        // NFKMLXGemma3.load(directoryURL:) then answer(image:question:). Here the tiny decoder decodes
        // through its hybrid cache and the tiny vision tower and projector produce soft tokens, so the
        // pipeline runs without a download.
        NFKMLXRandom.seed(2)
        let decoder = NFKMLXGemma3Net(.tiny)
        let cache = NFKMLXGemma3Cache(layerCount: 3, slidingWindow: 4)
        let prefill = decoder(MLXArray([Int32(3), 17, 42, 99, 7]).reshaped([1, 5]), cache: cache)
        let step = decoder(MLXArray([Int32(61)]).reshaped([1, 1]), cache: cache)
        XCTAssertEqual(prefill.shape, [1, 5, 131])
        XCTAssertEqual(step.shape, [1, 1, 131], "a cached step reads one token")
        XCTAssertEqual(cache.offset, 6)

        let tower = NFKMLXGemma3VisionNet(.tiny)                                  // 64-pixel image, 16 patches
        let projector = NFKMLXGemma3MultimodalProjector(visionHidden: 32, textHidden: 64, patchesPerSide: 4, tokensPerImage: 4)
        let soft = projector(tower(MLXRandom.uniform(low: -1, high: 1, [1, 64, 64, 3])))
        XCTAssertEqual(soft.shape, [1, 4, 64], "16 patches pool to 4 soft tokens at the decoder width")
    }

    func testGemma3n() throws {
        try requireMLXRuntime()
        // The released Gemma 3n loads from its directory: NFKMLXGemma3n.backend(directoryURL:) for
        // text (NFKInputPrompt / NFKInputMessages, with NFKInputImage or NFKInputAudio beside them),
        // or NFKMLXGemma3n.load(directoryURL:) then answer(image:question:). Here the tiny decoder
        // and the tiny audio encoder run without a download.
        NFKMLXRandom.seed(3)
        let decoder = NFKMLXGemma3nNet(.tiny)
        let tokens = MLXArray([Int32(3), 17, 42, 99, 7]).reshaped([1, 5])
        let cache = NFKMLXGemma3nCache(layerCount: NFKMLXGemma3nConfiguration.tiny.layerCount)
        let prefill = decoder(tokens, cache: cache)
        let step = decoder(MLXArray([Int32(61)]).reshaped([1, 1]), cache: cache)
        XCTAssertEqual(prefill.shape, [1, 5, 140])
        XCTAssertEqual(step.shape, [1, 1, 140], "a cached step reads one token")

        // AltUp holds the residual stream as four parallel copies, which the per-layer states report
        // one of; the last entry is the merged, normalized output.
        XCTAssertEqual(decoder.layerStates(tokens).count, NFKMLXGemma3nConfiguration.tiny.layerCount + 1)

        let audio = NFKMLXGemma3nAudioNet(.tiny)
        let (encoded, _) = audio(MLXRandom.normal([1, 26, 16]))
        XCTAssertEqual(encoded.shape, [1, 4, 32], "26 mel frames subsample by 4 and reduce by 2")
    }

    func testDeepSeekV41() throws {
        try requireMLXRuntime()
        // The released DeepSeek V4.1 Flash loads from its directory:
        // NFKMLXDeepSeek.backend(directoryURL:), which reads config.json and tokenizer.json, derives
        // the collapsed token map its n-gram memory addresses through, and generates one token a step
        // through NFKMLXDeepSeekCache. Nothing here downloads it, because nothing can hold it.
        //
        // The fit check is the part a consumer meets first, and it is pure arithmetic over the
        // configuration: the release stores its weights fp8 and fp4 and a load holds them bf16, so
        // what it needs is two to four times what the directory measures.
        let resident = NFKMLXDeepSeek.residentBytes(for: .v41Flash)
        XCTAssertGreaterThan(Double(resident) / 1_099_511_627_776, 1.25,
                             "763 billion parameters decode to more than 1.25 TiB of bf16")
        XCTAssertThrowsError(try NFKMLXDeepSeek.verifyFits(.v41Flash, budget: 512 << 30),
                             "a 512 GB machine cannot hold the decoded weights either")

        // Paging holds a group as the release stores it and decodes what a step reads: the routed
        // experts an expert at a time, the n-gram tables a row at a time. Each group moves the
        // figure, and the two together move it by most of the release.
        let experts = NFKMLXDeepSeek.residentBytes(for: .v41Flash, paging: .init(routedExperts: true))
        let everything = NFKMLXDeepSeek.residentBytes(for: .v41Flash, paging: .all)
        XCTAssertLessThan(experts, resident / 2, "the experts are more than half the decoder")
        XCTAssertLessThan(everything, experts, "and the n-gram tables are most of what is left")

        // What a load allocates is the decoder, and the decoder is not the release: the enumeration
        // covers the DSpark draft stack for the structural check's sake and nothing builds it.
        // Counting it is the difference between a 512 GiB machine being over budget and under it.
        XCTAssertLessThan(NFKMLXDeepSeek.decoderBytes(for: .v41Flash, paging: .all), 512 << 30,
                          "fully paged, the decoder's parameters fit a 512 GiB machine")
        XCTAssertThrowsError(try NFKMLXDeepSeek.verifyFits(.v41Flash, budget: 512 << 30),
                             "which the same machine cannot do with nothing paged")

        // The release computes in bf16, and so does a decoder loaded from it, matching the
        // release's own code bit for bit. Float32 is the opt-out,
        // NFKMLXDeepSeek.backend(directoryURL:computesInFloat32:), at twice the bytes a step reads.
        XCTAssertTrue(NFKMLXDeepSeekConfiguration.v41Flash.computesInBFloat16)
        var wide = NFKMLXDeepSeekConfiguration.v41Flash
        wide.computesInBFloat16 = false
        XCTAssertLessThan(NFKMLXDeepSeek.decoderBytes(for: .v41Flash, paging: .fullyMapped),
                          NFKMLXDeepSeek.decoderBytes(for: wide, paging: .fullyMapped),
                          "bf16 holds less than float32")

        // A picture becomes a span of positions, and the grid it plans is arithmetic over the
        // release's own settings: the aspect-preserving resize, the pad up to whole patches, and
        // the aligner's 3x3 pooling. Nothing here needs the weights.
        let processor = NFKMLXDeepSeekImageProcessor(.v41Flash)
        let plan = processor.plan(width: 1280, height: 720)
        XCTAssertEqual(plan.pixelWidth % processor.patchSize, 0, "a whole number of patches across")
        XCTAssertEqual(plan.pixelHeight % processor.patchSize, 0, "and down")
        XCTAssertEqual(plan.tokenCount, plan.tokenRows * (plan.tokenColumns + 1) + 2,
                       "a delimiter each side, and a newline ending every row of tokens")
        XCTAssertLessThanOrEqual(plan.tokenCount,
                                 NFKMLXDeepSeekVisionConfiguration.v41Flash.maximumTokenCount,
                                 "a picture costs no more positions than the release allows")

        // A directory that holds no release is refused rather than half-built. The error it carries
        // is the one reading config.json gives, so this asserts that it throws and not which code.
        let absent = URL(fileURLWithPath: "/nonexistent-deepseek-release")
        XCTAssertThrowsError(try NFKMLXDeepSeek.backend(directoryURL: absent))
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

    func testPixtralVisionTower() throws {
        try requireMLXRuntime()
        // The released Pixtral loads from its directory: NFKMLXPixtral.model(directoryURL:), then
        // answer(image:question:maxTokens:). Here a tiny random 2D-rotary tower embeds an 8×6-pixel
        // image (a 4×3 patch grid) and produces one feature per patch.
        NFKMLXRandom.seed(1)
        let config = NFKMLXPixtralVisionConfiguration(
            hiddenSize: 32, depth: 4, headCount: 2, intermediateSize: 64, patchSize: 2, imageSize: 16)
        let net = NFKMLXPixtralVisionNet(config)
        let output = net(MLXRandom.normal([1, 3, 8, 6]))
        eval(output)
        XCTAssertEqual(output.shape, [12, 32])
        let projected = NFKMLXPixtralConnector(visionSize: 32, textSize: 48)(output)
        eval(projected)
        XCTAssertEqual(projected.shape, [12, 48])
    }

    func testStateSpaceLanguageModels() throws {
        try requireMLXRuntime()
        // The released state-space decoders load from their directories: NFKMLXMamba.mambaBackend(directoryURL:)
        // (Codestral Mamba 7B, Mamba-2 blocks), NFKMLXGraniteHybrid.graniteBackend(directoryURL:) (Granite
        // 4.0-H, Mamba-2 and attention layers, dense or with routed experts), and
        // NFKMLXNemotronH.nemotronBackend(directoryURL:) (Nemotron Nano 2, Mamba-2, attention, and MLP
        // layers); each also downloads through its repo factory and registers with register(). Here
        // shrunk random geometries run each decoder over a short prompt.
        NFKMLXRandom.seed(5)
        let prompt = MLXArray([Int32(1), 5, 9, 12, 7]).reshaped([1, 5])
        let mamba = NFKMLXMamba.makeNet(NFKMLXMamba2Configuration(
            hiddenSize: 64, layerCount: 2, vocabularySize: 128, rmsEpsilon: 1e-5,
            intermediateSize: 128, headCount: 8, headDimensions: 16, stateSize: 16,
            groupCount: 2, convolutionKernel: 4, useConvolutionBias: true,
            useProjectionBias: false, tiesWordEmbeddings: false))
        XCTAssertEqual(mamba(prompt).shape, [1, 5, 128])

        let granite = NFKMLXGraniteHybrid.makeNet(NFKMLXGraniteHybridConfiguration(
            hiddenSize: 64, layerCount: 3, vocabularySize: 128, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: false, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            mambaHeadCount: 8, mambaHeadDimensions: 16, mambaGroupCount: 1, mambaStateSize: 16,
            mambaConvolutionKernel: 4, mambaExpand: 2, mambaConvolutionBias: true,
            mambaProjectionBias: false, sharedIntermediateSize: 96, expertCount: 0,
            expertsPerToken: 0, expertIntermediateSize: 0,
            embeddingMultiplier: 2.0, residualMultiplier: 0.5, attentionMultiplier: 0.25,
            logitsScaling: 3.0, layerTypes: [.mamba, .mamba, .attention]))
        XCTAssertEqual(granite(prompt).shape, [1, 5, 128])

        let nemotron = NFKMLXNemotronH.makeNet(NFKMLXNemotronHConfiguration(
            hiddenSize: 64, vocabularySize: 128, rmsEpsilon: 1e-5,
            headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            mambaHeadCount: 8, mambaHeadDimensions: 16, mambaGroupCount: 2, mambaStateSize: 16,
            mambaConvolutionKernel: 4, mambaConvolutionBias: true, mambaProjectionBias: false,
            timeStepMinimum: 0.001, intermediateSize: 96, mlpBias: false,
            layerTypes: [.mamba, .attention, .mlp]))
        XCTAssertEqual(nemotron(prompt).shape, [1, 5, 128])
    }

    func testPhi4Multimodal() throws {
        try requireMLXRuntime()
        // The released Phi-4-multimodal loads from its directory: NFKMLXPhi4MM.backend(directoryURL:precision:)
        // (Objective-C backendWithDirectoryURL:precision:error:), or from the hub through
        // backend(repo:revision:cacheDirectoryURL:precision:). NFKMLXPhi4MM.model(directoryURL:precision:)
        // then answers respond(messages:images:audios:options:): a conversation in the release's chat
        // template, any number of pictures and clips, and sampled or greedy decoding. Here the two
        // preprocessors read synthetic input, tiny random towers project it (two clips as one padded
        // batch), and a tiny partial-rotary LongRoPE decoder fuses the image and the clips into one
        // prompt, as the vision-with-speech mode does.
        NFKMLXRandom.seed(1)
        let rgb = (0 ..< 500 * 300 * 3).map { UInt8(($0 * 37) % 256) }
        let image = NFKMLXPhi4MMImageProcessor.process(rgb: rgb, width: 500, height: 300)
        XCTAssertEqual(image.pixels.dim(0), 3, "a global view and a 1×2 crop grid")
        let imageNet = NFKMLXPhi4MMImageNet(NFKMLXSigLIPConfiguration(
            hiddenSize: 32, layerCount: 2, headCount: 2, intermediateSize: 64, patchSize: 14, imageSize: 448),
            decoderHidden: 48)
        let imageFeatures = imageNet.projected(pixels: image.pixels, imageSize: image.imageSize,
                                               validPatches: image.validPatches)
        XCTAssertEqual(imageFeatures.dim(0), image.tokenCount, "one embedding per reserved image token")

        let mels = try [NFKMLXPhi4MMAudioFeatures.logMel(Self.tone(44100), sampleRate: 44100),
                        NFKMLXPhi4MMAudioFeatures.logMel(Array(Self.tone(16000).prefix(8000)), sampleRate: 16000)]
        let clips = NFKMLXPhi4MMAudioNet(.tiny).projected(clips: mels, mode: .vision)
        XCTAssertEqual(clips.map { $0.dim(0) }, mels.map { NFKMLXPhi4MMAudioFeatures.tokenCount(frames: $0.dim(0)) },
                       "one embedding per reserved audio token, per clip")
        let audioFeatures = concatenated(clips, axis: 0)
        let audioTokens = audioFeatures.dim(0)

        var configuration = NFKMLXLanguageConfiguration(
            hiddenSize: 48, layerCount: 2, headCount: 3, keyValueHeadCount: 1, headDimensions: 16,
            intermediateSize: 96, vocabularySize: 200_064, ropeTheta: 10_000, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: true, normalizesQueryAndKey: false)
        configuration.rotaryDimensions = 12
        var scaling = NFKMLXRoPEScaling(kind: .longrope, factor: 1, originalMaxPositionEmbeddings: 4096)
        scaling.shortFactor = Array(repeating: 1, count: 6)
        scaling.longFactor = [1, 1.5, 2, 3, 4, 6]
        scaling.maximumPositionEmbeddings = 131_072
        configuration.ropeScaling = scaling
        let decoder = NFKMLXLanguageNet(configuration)
        let ids = [NFKMLXPhi4MM.userTokenId]
            + Array(repeating: NFKMLXPhi4MM.imageTokenId, count: image.tokenCount)
            + Array(repeating: NFKMLXPhi4MM.audioTokenId, count: audioTokens)
            + [NFKMLXPhi4MM.endTokenId, NFKMLXPhi4MM.assistantTokenId]
        let hidden = NFKMLXPhi4MM.fusedHidden(decoder: decoder, inputIds: ids,
                                              features: [(NFKMLXPhi4MM.imageTokenId, imageFeatures),
                                                         (NFKMLXPhi4MM.audioTokenId, audioFeatures)])
        eval(hidden)
        XCTAssertEqual(hidden.shape, [1, ids.count, 48])
    }

    func testTypedDecisions() throws {
        try requireMLXRuntime()
        // The released model loads from a variant directory: NFKMLXLaya.laya(directoryURL:) (the root,
        // typed-decisions, or multilingual folder), then laya.decide(state:questions:) answers the same
        // NFKDecisionQuestions the hosted Jev backend takes. Here a tiny random net stands in.
        NFKMLXRandom.seed(3)
        let tokenizer = NFKMLXLayaTokenizer { text in text.unicodeScalars.map { Int($0.value % 500) + 8 } }
        let laya = try NFKMLXLaya.laya(weightsURL: nil, tokenizer: tokenizer, configuration: .tiny)
        let answers = laya.decide(state: "Help! My payouts have been failing for 3 days.", questions: [
            "department": NFKDecisionQuestion.choiceQuestion(withInstructions: "Which team should handle this?",
                                                             options: ["billing", "technical", "sales"]),
            "urgent": NFKDecisionQuestion.noulQuestion(withInstructions: "The customer needs an answer today."),
        ])
        XCTAssertEqual(answers.count, 2)
        XCTAssertTrue(["billing", "technical", "sales"].contains(answers["department"]?.choice ?? ""))
        XCTAssertTrue((0 ... 1).contains(answers["urgent"]?.probability ?? -1))
        // The same through the contract: NFKInputState + NFKInputQuestions in, NFKOutputAnswers out.
        let backend = laya.makeBackend()
        let request = NFKInferenceRequest(inputs: [NFKInputState: "My invoice is wrong.",
                                                   NFKInputQuestions: ["urgent": NFKDecisionQuestion.noulQuestion(withInstructions: "Urgent?")]])
        XCTAssertEqual(try backend.runInference(for: request).answers?["urgent"]?.type, .noul)
    }

    func testTypedDecisionsCommunityReproductions() throws {
        try requireMLXRuntime()
        // open-jev-deberta reads the state and every question in one DeBERTa pass:
        //   NFKMLXOpenJevDeBERTa.openJev(revision: NFKMLXOpenJevDeBERTa.measuredRevision, cacheDirectoryURL: nil)
        // Open-Jev scores each candidate with a Qwen3.5 text model and a LoRA adapter:
        //   NFKMLXOpenJev.openJev(variant: .twoB, revision: nil, cacheDirectoryURL: nil)
        // Both take the NFKDecisionQuestions Laya and the hosted Jev take. Tiny random nets stand in.
        NFKMLXRandom.seed(4)
        let questions: [NFKDecisionQuestion] = [
            .choiceQuestion(withInstructions: "Which team?", options: ["billing", "technical", "sales"]),
            .noulQuestion(withInstructions: "Urgent?"),
        ]
        let characters = NFKMLXDecisionTokenizer { text in text.unicodeScalars.map { Int($0.value % 400) + 10 } }
        let deberta = NFKMLXOpenJevDeBERTa(net: NFKMLXOpenJevDeBERTaNet(.tiny), tokenizer: characters)
        let ordered = try deberta.decide(state: "My invoice is wrong.", questions: questions)
        XCTAssertEqual(ordered.map(\.type), [.choice, .noul])

        let words = NFKMLXDecisionTokenizer { text in
            text.split(whereSeparator: \.isWhitespace).map { $0.unicodeScalars.reduce(7) { ($0 * 31 + Int($1.value)) % 500 } + 5 }
        }
        let openJev = NFKMLXOpenJev(net: NFKMLXOpenJevNet(.tiny), tokenizer: words)
        // Behind the contract, as NFKTypeSafeBackend is: NFKInputState + NFKInputQuestions in.
        let request = NFKInferenceRequest(inputs: [NFKInputState: "My invoice is wrong.",
                                                   NFKInputQuestions: ["urgent": questions[1]]])
        XCTAssertEqual(try openJev.makeBackend().runInference(for: request).answers?["urgent"]?.type, .noul)
    }

    /// A validation-store path from the environment, or else from `~/.inferkit-validation.json`, which
    /// `Tools/validation-assets/fetch.py` writes.
    private func validationPath(_ key: String) -> String? {
        if let value = ProcessInfo.processInfo.environment[key] { return value }
        let file = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        let store = (try? JSONSerialization.jsonObject(with: Data(contentsOf: file))) as? [String: Any]
        return store?[key] as? String
    }

    // Docs/examples.md "Downloading and setting up Laya": one call fetches a variant into the hub cache
    // and builds it; a cached file is not fetched again. Here the cache is seeded from the local
    // validation store (IK_VAL_LAYA), so the call reads it without the network.
    func testTypedDecisionsDownloadAndSetup() throws {
        try requireMLXRuntime()
        guard let root = validationPath("IK_VAL_LAYA") else {
            throw XCTSkip("set IK_VAL_LAYA to the Laya release")
        }
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("laya-example-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: cache) }
        let snapshot = cache.appendingPathComponent("\(NFKMLXLaya.repository)/\(NFKMLXLaya.measuredRevision)")
        for path in NFKMLXLaya.releaseFiles(for: .typedDecisions) {
            let link = snapshot.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: URL(fileURLWithPath: root).appendingPathComponent(path))
        }

        let laya = try NFKMLXLaya.laya(variant: .typedDecisions, revision: NFKMLXLaya.measuredRevision,
                                       cacheDirectoryURL: cache)
        let answer = laya.answer(state: "I was charged twice for one order.",
                                 question: .choiceQuestion(withInstructions: "Which team should handle this?",
                                                           options: ["billing", "technical", "sales"]))
        XCTAssertTrue(["billing", "technical", "sales"].contains(answer.choice ?? ""))

        // The asynchronous form hands the folder to a completion handler on a background queue.
        let done = expectation(description: "download")
        NFKMLXLaya.download(variant: .typedDecisions, revision: NFKMLXLaya.measuredRevision,
                            cacheDirectoryURL: cache) { folder, error in
            XCTAssertNil(error)
            XCTAssertEqual(folder?.lastPathComponent, "typed-decisions")
            done.fulfill()
        }
        wait(for: [done], timeout: 30)
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

    func testMultimodalRetrieval() throws {
        try requireMLXRuntime()
        // The released pair loads from its directory: NFKMLXQwen3VLEmbedder.embedder(directoryURL:)
        // and NFKMLXQwen3VLReranker.reranker(directoryURL:), then embedding(forText:) /
        // embedding(forImage:text:instruction:) and scores(query:documents:). Both are 4.3 GB, so
        // what runs here without a download is the prompt each is trained to read and the probe a
        // consumer trains over the frozen backbone.
        let prompt = NFKMLXQwen3VLEmbedder.prompt(text: "a red bicycle", imageTokens: 4,
                                                  instruction: "Represent the photo")
        XCTAssertTrue(prompt.hasPrefix("<|im_start|>system\nRepresent the photo.<|im_end|>"),
                      "the instruction is punctuated and goes in the system turn")
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n"))
        let pair = NFKMLXQwen3VLReranker.prompt(query: "how tall is it", queryImageTokens: 0,
                                                document: "330 metres", documentImageTokens: 0,
                                                instruction: nil)
        XCTAssertTrue(pair.contains("<Query>:how tall is it\n<Document>:330 metres"))

        let adapter = NFKMLXQwen3VLEmbeddingAdapter(dimensions: 8)
        let embeddings = MLXRandom.normal([2, 8])
        let adapted = adapter(embeddings)
        eval(adapted)
        XCTAssertEqual(adapted.shape, [2, 8])
        let head = NFKMLXQwen3VLRerankerHead(dimensions: 8)
        let logits = head(MLXRandom.normal([3, 8]))
        eval(logits)
        XCTAssertEqual(logits.shape, [3])
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

        // Cosmos Tokenizer: an image or a clip → a continuous latent or discrete tokens → a reconstruction.
        let cosmos = try NFKMLXCosmosTokenizer.backend(variant: .discreteImage8x8, weightsURL: nil)
        let cosmosResult = try cosmos.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(16)]))
        XCTAssertNotNil(cosmosResult.output(forKey: NFKOutputImage), "a Cosmos Tokenizer reconstruction")
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

    func testFlux2TextToImage() throws {
        try requireMLXRuntime()
        // The released path is `NFKMLXFlux2.flux2(directoryURL:)` over a diffusers FLUX.2 [klein]
        // release — transformer, autoencoder, Qwen3 text encoder and tokenizer — then
        // `image(forPrompt:)`, which renders the release's own chat template, reads three layers of
        // the encoder, denoises, and decodes. That is several gigabytes of weights, so what runs here
        // is the same public pipeline at a tiny geometry with the conditioning supplied directly.
        var vae = NFKMLXSDVAEConfiguration.flux2
        vae.latentChannels = 4
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4
        var geometry = NFKMLXFlux2Configuration.tiny
        geometry.inChannels = 16                                       // 4 latent channels, 2×2 patch

        let pipeline = NFKMLXFlux2Pipeline(
            transformer: NFKMLXFlux2TransformerNet(geometry),
            autoencoder: NFKMLXSDAutoencoder(configuration: vae),
            codec: NFKMLXFlux2LatentCodec(patchedChannels: 16, epsilon: 1e-4, patch: 2))
        let conditioning = MLXRandom.normal([1, 5, geometry.jointAttentionDim])
        let image = pipeline.generate(promptEmbeds: conditioning, latentHeight: 2, latentWidth: 3,
                                      steps: 2, seed: 1)
        eval(image)
        XCTAssertEqual(image.shape[3], 3, "the pipeline decodes to RGB")
        XCTAssertEqual(NFKMLXFlux2.modelName, "flux.2-klein-4b")
    }

    func testResidencyPagesAMixture() throws {
        try requireMLXRuntime()
        // A release directory is held as an NFKMLXResidency says. The language, Gemma, and FLUX factories
        // take one: `NFKMLXLanguage.backend(directoryURL:residency:)`,
        // `NFKMLXGemmaLanguage.backend(directoryURL:precision:residency:)`, and
        // `NFKMLXFlux.flux(directoryURL:residency:)`. `.paged` leaves a mixture's routed experts in the
        // release and reads each as the router reaches it; what runs here is the same load over a tiny
        // mixture written as a release.
        let source = NFKMLXLanguage.makeNet(.tinyMixture)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("residency-example-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try NFKMLXWeights.save(source, to: directory.appendingPathComponent("model.safetensors"))

        let paged = NFKMLXLanguage.makeNet(.tinyMixture)
        try NFKMLXLanguage.loadWeights(into: paged, fromDirectory: directory, precision: .float32,
                                       residency: .paged)
        let store = try XCTUnwrap(paged.expertStore, "the routed experts stay in the release")
        store.cacheByteBudget = 1 << 20

        let tokens = MLXArray([3, 17, 42, 8].map { Int32($0) }).reshaped([1, 4])
        XCTAssertEqual(paged(tokens).asArray(Float.self), source(tokens).asArray(Float.self),
                       "a paged model computes the resident model's logits")
        XCTAssertGreaterThan(store.materializeCount, 0, "and read only the experts it routed to")
    }

    func testQwenImageTextToImage() throws {
        try requireMLXRuntime()
        // The released pipeline loads each stage from its own directory:
        // NFKMLXQwenImage.makeNet + loadWeights for the 7.1B transformer,
        // NFKMLXQwenImageVAE.net(directoryURL:) for the autoencoder, and NFKMLXQwen3VL.decoder for
        // the text encoder, then generate(promptEmbeddings:height:width:steps:). That is 31 GB of
        // weights, so what runs here is the same public path at a tiny geometry.
        NFKMLXRandom.seed(4)
        var configuration = NFKMLXQwenImageConfiguration.tiny
        configuration.inChannels = 4
        configuration.outChannels = 4
        let pipeline = NFKMLXQwenImagePipeline(
            transformer: NFKMLXQwenImage.makeNet(configuration),
            vae: NFKMLXQwenImageVAE.makeNet(.qwenImage21Tiny))

        let embeddings = MLXRandom.normal([8, configuration.contextInDimensions])
        let image = pipeline.generate(promptEmbeddings: embeddings, height: 64, width: 64, steps: 2)
        eval(image)
        // The pipeline's latent grid is the request divided by the release's 16, and this tiny
        // autoencoder upsamples by 2 rather than the released 16, so 64 pixels in is 8 pixels out.
        XCTAssertEqual(image.shape, [8, 8, 4])
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
        // The released encoders are `.vitB`, `.vitL`, and `.vitH` (`NFKMLXSAMVariant`); a checkpoint
        // fits only its own size, and each is built the same way with its weights URL.
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
        // Every released size is a variant — tiny, base, small, medium, large (v1/v2), large-v3, and
        // large-v3-turbo, whose decoder is four layers deep.
        let whisperBase = try NFKMLXWhisper.backend(variant: .base, weightsURL: nil)
        XCTAssertTrue(whisperBase.isReady)

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

        // MP-SENet: a time-frequency transformer that denoises magnitude and phase in parallel.
        let mpsenet = try NFKMLXMPSENetFactory.backend(weightsURL: nil)
        XCTAssertNotNil(try mpsenet.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // GTCRN: an ultra-light grouped TCRN for real-time speech enhancement.
        let gtcrn = try NFKMLXGTCRNFactory.backend(weightsURL: nil)
        XCTAssertNotNil(try gtcrn.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // MetricGAN+: a two-layer BLSTM magnitude mask over log1p(|X|) frames, the family's smallest.
        let metricgan = try NFKMLXMetricGANPlus.backend(weightsURL: nil)
        XCTAssertNotNil(try metricgan.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // CMGAN: a dense encoder, four two-stage conformer blocks, and mask + complex decoders over a
        // power-compressed spectrogram.
        let cmgan = try NFKMLXCMGAN.backend(weightsURL: nil)
        XCTAssertNotNil(try cmgan.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // FRCRN: two complex UNets with frequency-recurrent FSMN memories over a conv-STFT.
        let frcrn = try NFKMLXFRCRN.backend(weightsURL: nil)
        XCTAssertNotNil(try frcrn.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // MossFormer2 SR: a mel-to-mel MossFormer2 backbone and a Snake HiFi-GAN generator, the input's
        // own band kept and the generated band above it added.
        let superResolution = try NFKMLXMossFormer2SRFactory.backend(directoryURL: nil)
        XCTAssertNotNil(try superResolution.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // NU-Wave 2: diffusion bandwidth extension (short-time Fourier convolutions, an 8-step logSNR DDIM).
        let nuwave = try NFKMLXNUWave2.backend(weightsURL: nil)
        XCTAssertNotNil(try nuwave.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // Apollo: music codec-artifact restoration (an 80-band split, band Roformer + time convolution layers).
        let apollo = try NFKMLXApollo.backend(weightsURL: nil)
        XCTAssertNotNil(try apollo.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // SGMSE+: score-based generative dereverberation, a reverse-SDE sampler over an NCSN++ score net.
        // The released net is large and the sampler multi-step; this exercises the pipeline with a small,
        // few-step configuration and a short clip (random weights — not the quality).
        let shortClip = NFKMLXWaveFile.data(samples: Self.tone(4000), sampleRate: 16000)
        let sgmse = try NFKMLXSGMSE.backend(weightsURL: nil, seed: 0,
                                            config: NFKMLXSGMSEConfiguration(reverseSteps: 2, baseChannels: 8))
        XCTAssertNotNil(try sgmse.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: shortClip])).output(forKey: NFKOutputAudio))

        // StoRM: a few-step follow-on on SGMSE+ — a discriminative predictor then a conditioned score net
        // regenerates from the estimate. Same reduced config for the smoke test.
        let stormBase = NFKMLXSGMSEConfiguration(reverseSteps: 2, baseChannels: 8)
        let storm = try NFKMLXStoRM.backend(weightsURL: nil, seed: 0,
                                            config: NFKMLXStoRMConfiguration(base: stormBase, condition: .both))
        XCTAssertNotNil(try storm.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: shortClip])).output(forKey: NFKOutputAudio))

        // MossFormer2 SE: full-band 48 kHz enhancement — a Kaldi-fbank mask over the MossFormer2 backbone.
        let mossformer2 = try NFKMLXMossFormer2Factory.backend(weightsURL: nil)
        XCTAssertNotNil(try mossformer2.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // DeepFilterNet3: a ~2.3M-parameter real-time 48 kHz denoiser — an ERB mask plus a deep filter on
        // the lowest bins, over a libdf-reproduced STFT/ERB/norm front end.
        let deepfilternet = try NFKMLXDeepFilterNetFactory.backend(weightsURL: nil)
        XCTAssertNotNil(try deepfilternet.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // VoiceRestore: a flow-matching universal restorer (an E2-TTS transformer with a gateloop mixing
        // layer + a BigVGAN vocoder). Random weights, 2 CFM steps, guidance off — the fast gallery path.
        let voiceRestore = NFKMLXVoiceRestoreBackend(net: NFKMLXVoiceRestoreFactory.makeNet(),
                                                     vocoder: NFKMLXBigVGAN(.init()),
                                                     identifier: "voicerestore", steps: 2, cfgStrength: 0)
        XCTAssertNotNil(try voiceRestore.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // Resemble Enhance: a five-network general restorer (STFT-mask denoiser + IRMAE/CFM latent flow
        // matching + a UnivNet LVC vocoder). Random weights, denoiser off and 2 CFM steps for the fast path.
        let resemble = NFKMLXResembleEnhanceBackend(net: NFKMLXResembleEnhanceFactory.makeNet(),
                                                    identifier: "resemble-enhance", lambd: 0, tau: 0.5, nfe: 2)
        XCTAssertNotNil(try resemble.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

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
        // The music codecs (32 kHz, 44.1 kHz) add a fourth codebook and windowed attention at the bottleneck.
        let snacMusic = try NFKMLXSNAC.backend(variant: .music32kHz, weightsURL: nil)
        XCTAssertEqual(snacMusic.backendIdentifier, "snac-32khz")

        // BigVGAN v2: an anti-aliased SnakeBeta vocoder, shipped standalone. The backend runs
        // copy-synthesis (audio → the released mel front end → generator → waveform).
        let bigvgan = try NFKMLXBigVGANFactory.backend(weightsURL: nil)
        XCTAssertNotNil(try bigvgan.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // Mimi: a transformer-in-codec neural audio codec. The backend reconstructs audio → codes → audio;
        // NFKMLXMimi.encode returns the per-codebook token streams (semantic + acoustic).
        let mimi = try NFKMLXMimi.backend(weightsURL: nil)
        XCTAssertNotNil(try mimi.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).output(forKey: NFKOutputAudio))

        // Basic Pitch: a recording becomes notes. The result is an NFKMIDISequence under NFKOutputMIDI,
        // which writes a Standard MIDI File.
        let basicPitch = try NFKMLXBasicPitch.backend(weightsURL: nil)
        let transcription = try basicPitch.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        let midi = try XCTUnwrap(transcription.midi)
        XCTAssertGreaterThan(midi.standardMIDIFileData().count, 22)

        // hFT-Transformer: piano transcription, attending across frequency and then across time.
        // Four heads at two levels; the time level is the answer.
        let hft = try NFKMLXHFTTransformer.backend(weightsURL: nil)
        let piano = try hft.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        XCTAssertNotNil(piano.midi)

        // MuScriptor: a mixture becomes one MIDI track per instrument. The released weights are CC
        // BY-NC 4.0 behind a gated repository, and the smallest is 103M parameters, so the gallery
        // builds the module at a small configuration rather than allocating a release-sized model
        // with random weights; the factory and the decode path are the same either way.
        let muScriptorNet = NFKMLXMuScriptor.makeNet(
            NFKMLXMuScriptorConfiguration(dimension: 64, heads: 4, layers: 2, card: 1395))
        let muScriptorTokens = muScriptorNet.generate(chunk: Self.tone(16000))
        XCTAssertTrue(muScriptorTokens.allSatisfy { $0 >= 0 && $0 < 1393 },
                      "the decode stays inside the tokenizer's vocabulary")

        // All-In-One: a track's structure. It reads the four HT Demucs stems (bass, drums, other,
        // vocals), and returns labeled sections, beats with their position in the bar, and the tempo.
        let allInOne = try NFKMLXAllInOne.backend(weightsURL: nil)
        let stems = [Data](repeating: wave, count: 4)
        let structure = try allInOne.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: stems]))
        XCTAssertNotNil(structure.segments)

        let tagger = try NFKMLXAudioTagger.backend(weightsURL: nil, labels: nil)
        XCTAssertNotNil(try tagger.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave])).classifications)

        let demucs = try NFKMLXDemucs.backend(weightsURL: nil)
        XCTAssertEqual(demucs.backendIdentifier, "demucs")

        // Demucs v4: parallel spectrogram and waveform branches joined by a cross-transformer.
        let htdemucs = try NFKMLXHTDemucs.backend(weightsURL: nil)
        XCTAssertEqual(htdemucs.backendIdentifier, "htdemucs")
        // The six-stem release adds guitar and piano; the fine-tuned release is four checkpoints
        // combined by `backendWithFineTunedWeightsURLs:`.
        let sixStem = try NFKMLXHTDemucs.backend(variant: .sixStem, weightsURL: nil)
        XCTAssertEqual(sixStem.backendIdentifier, "htdemucs-6s")

        // Full text-to-speech chain: phonemizer → acoustic (FastSpeech2-style) → vocoder (HiFi-GAN).
        let tts = NFKMLXTTS(phonemizer: NFKMLXNeuralG2P(), symbols: (0 ..< 40).map { "p\($0)" })
        let speech = tts.makeSpeechBackend()                     // reads NFKInputPrompt, writes a WAV NFKAudioAsset
        XCTAssertEqual(speech.backendIdentifier, "tts")

        // Parakeet-TDT (NeMo FastConformer + token-and-duration transducer): a second ASR beside Whisper.
        // Random weights at a shrunk geometry run the whole path — mel front end, dw-striding subsampler,
        // rel-pos conformer, LSTM prediction net, joint, greedy TDT decode — on the same clip.
        let parakeet = NFKMLXParakeet.backend(configuration: .tiny)
        let recognized = try parakeet.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        XCTAssertNotNil(recognized.output(forKey: NFKOutputText), "a transcript (token ids without a vocabulary)")

        // Canary-1B-v2 (NeMo FastConformer encoder + attention encoder-decoder): a multitask
        // ASR/translation speech model. Random weights at a shrunk geometry run the whole path — the
        // biased FastConformer encoder (reused from Parakeet) and the Transformer decoder's greedy
        // generation from a task prompt.
        let canary = NFKMLXCanaryNet(.tiny)
        let canaryTokens = canary.recognize(Self.tone(16000), prompt: [4, 5])
        XCTAssertLessThanOrEqual(canaryTokens.count, canary.configuration.maxDecodeTokens,
                                 "greedy decode over the encoder frames")

        // The released speech language models load from their directories:
        // NFKMLXGraniteSpeech.graniteSpeechBackend(directoryURL:) (a Conformer, a BLIP-2 Q-former, and a
        // Granite decoder with its audio LoRA) and NFKMLXVoxtral.voxtralBackend(directoryURL:) (a Whisper
        // encoder, a projector, and a Llama decoder); each also downloads through its repo factory and
        // registers with register(). Here shrunk random geometries fuse a clip's features into the
        // decoder at the audio-token positions, as the transcription prompt does.
        let graniteText = NFKMLXGraniteTextConfiguration(
            hiddenSize: 32, layerCount: 2, headCount: 4, keyValueHeadCount: 2, headDimensions: 8,
            intermediateSize: 64, vocabularySize: 40, ropeTheta: 1_000_000, rmsEpsilon: 1e-5,
            embeddingMultiplier: 2.0, residualMultiplier: 0.5, attentionMultiplier: 0.25,
            logitsScaling: 3.0, tiesWordEmbeddings: false)
        let graniteSpeech = NFKMLXGraniteSpeechNet(
            encoder: NFKMLXGraniteSpeechEncoderConfiguration(
                inputDim: 16, hiddenDim: 32, outputDim: 24, layerCount: 2, headCount: 2, headDimensions: 16,
                feedForwardMultiplier: 2, convolutionExpansionFactor: 2, convolutionKernel: 5,
                contextSize: 8, maxPositionEmbeddings: 16),
            projector: NFKMLXGraniteSpeechProjectorConfiguration(
                hiddenSize: 32, layerCount: 1, headCount: 2, intermediateSize: 64, encoderHiddenSize: 32,
                layerNormEpsilon: 1e-12, windowSize: 4, downsampleRate: 2),
            text: graniteText, audioTokenId: 39)
        graniteSpeech.train(false)
        let graniteFeatures = MLXRandom.normal([1, 12, 16])
        let graniteAudio = graniteSpeech.audioEmbeddings(graniteFeatures)
        let graniteSlots = graniteAudio.size / graniteText.hiddenSize
        let graniteTokens = MLXArray([Int32(1)] + Array(repeating: Int32(39), count: graniteSlots) + [Int32(2)])
            .reshaped([1, -1])
        XCTAssertEqual(graniteSpeech.logits(tokens: graniteTokens, audioEmbeddings: graniteAudio).shape,
                       [1, graniteSlots + 2, 40], "fused logits over the prompt")

        let voxtralText = NFKMLXGraniteTextConfiguration(
            hiddenSize: 64, layerCount: 2, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            intermediateSize: 128, vocabularySize: 40, ropeTheta: 10_000, rmsEpsilon: 1e-5,
            embeddingMultiplier: 1, residualMultiplier: 1, attentionMultiplier: 1.0 / Float(16).squareRoot(),
            logitsScaling: 1, tiesWordEmbeddings: false)
        let voxtral = NFKMLXVoxtral.makeNet(NFKMLXVoxtralConfiguration(
            audioMels: 32, audioState: 64, audioHeads: 4, audioLayers: 2, projectorInputSize: 256,
            audioTokenId: 39, text: voxtralText))
        let voxtralMel = MLXRandom.normal([1, 48, 32])
        let voxtralAudio = voxtral.audioEmbeddings(voxtralMel)
        let voxtralSlots = voxtralAudio.dim(0)
        let voxtralTokens = MLXArray([Int32(1)] + Array(repeating: Int32(39), count: voxtralSlots) + [Int32(2)])
            .reshaped([1, -1])
        XCTAssertEqual(voxtral.logits(tokens: voxtralTokens, audioEmbeddings: voxtralAudio).shape,
                       [1, voxtralSlots + 2, 40], "fused logits over the prompt")

        // Chatterbox (zero-shot voice cloning TTS): a VoiceEncoder speaker embedding and the S3 speech
        // tokenizer read the voice prompt, T3 samples speech codes for the text, and S3Gen (flow matching +
        // HiFT vocoder) renders them. Shrunk random geometries run the prompt side and a few T3 steps.
        let promptSamples = Self.tone(16000)
        let voiceEncoder = NFKMLXChatterboxVoiceEncoderNet(.tiny)
        let speakerEmbedding = voiceEncoder.embed(samples: promptSamples)
        let speechTokenizer = NFKMLXS3TokenizerNet(.tiny)
        let promptCodes = speechTokenizer.tokenize(promptSamples, maximumCodes: 8)
        XCTAssertTrue(promptCodes.allSatisfy { $0 < speechTokenizer.configuration.codebookSize }, "S3 codes are base-3 indices")
        let t3 = NFKMLXT3Net(.tiny)
        var sampling = NFKMLXT3SamplingOptions()
        sampling.temperature = 0
        sampling.maximumTokens = 4
        let condition = NFKMLXT3Condition(speakerEmbedding: speakerEmbedding, promptTokens: promptCodes)
        let speechCodes = t3.generate(condition: condition, textTokens: [255, 3, 4, 5, 0], options: sampling)
        XCTAssertLessThanOrEqual(speechCodes.count, 4, "T3 emits speech codes step by step")

        // Kokoro-82M (StyleTTS2 / iSTFTNet): a phoneme string + a voicepack row → a 24 kHz waveform. Run
        // here with random weights to exercise the whole pipeline (PL-BERT → duration → F0/N → iSTFTNet).
        let kokoro = NFKMLXKokoroNet(.v1)
        let voice = MLXArray.zeros([512, 1, 256]) + 0.01
        var vocab = [String: Int]()
        for scalar in "hɛloʊwɜld ".unicodeScalars { vocab[String(scalar)] = Int(scalar.value) % 170 + 1 }
        let kokoroAudio = kokoro.synthesize(phonemes: "hɛloʊ wɜld", voice: voice, vocab: vocab)
        XCTAssertGreaterThan(kokoroAudio.dim(0), 0, "Kokoro produces a waveform")

        // MiniMax Music 3: a music description under NFKInputPrompt and lyrics under NFKInputLyrics
        // become a stereo 44.1 kHz clip. The stack is 27 GB of separately licensed weights, so the
        // factory takes the downloaded release DIRECTORY and there is no random-weights form;
        // isReady reports whether the weights are present rather than failing the build.
        let music = try NFKMLXMusic3.backend(directoryURL:
            FileManager.default.temporaryDirectory.appendingPathComponent("minimax-music3"))
        XCTAssertEqual(music.backendIdentifier, "minimax-music3")
        XCTAssertFalse(music.isReady, "no weights at that path yet — download the release first")
        // .staged loads each stage for its turn and releases it; .resident holds them between runs.
        let staged = try NFKMLXMusic3.backend(directoryURL:
            FileManager.default.temporaryDirectory.appendingPathComponent("minimax-music3"), residency: .staged)
        XCTAssertEqual((staged as? NFKMLXMusicBackend)?.residency, .staged)
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

    // MARK: Translation (text → text, seq2seq backend)

    func testTranslationModels() throws {
        try requireMLXRuntime()
        // The released translators load from their release directories: NFKMLXMarian.backend(directoryURL:)
        // (one OPUS-MT pair, or backend(sourceLanguage:targetLanguage:cacheDirectoryURL:) to download it),
        // NFKMLXM2M100.backend(variant:directoryURL:) (100 languages, or SMaLL-100), and
        // NFKMLXMADLAD.backend(directoryURL:half:) (400+ languages, T5). Each answers NFKInputPrompt with
        // NFKOutputText for the NFKParameterTargetLanguage asked. Here tiny random networks exercise the
        // two architectures and the shared greedy/beam decoder without a download.
        let marian = try NFKMLXMarian.network(directoryURL: nil, configuration: .tinyMarian)
        let m2m = try NFKMLXM2M100.network(directoryURL: nil, configuration: .tinyM2M100)
        for net in [marian, m2m] {
            let c = net.configuration
            let decoding = NFKMLXSeq2SeqDecoding(beams: 3, maxTokens: 8, startToken: c.decoderStartTokenId, endToken: c.eosTokenId)
            let tokens = NFKMLXSeq2SeqDecoder.generate(net, source: [5, 6, 7, c.eosTokenId], decoding: decoding)
            XCTAssertLessThanOrEqual(tokens.count, 8)
        }
        let madlad = try NFKMLXMADLAD.network(directoryURL: nil, configuration: .tiny)
        let decoding = NFKMLXSeq2SeqDecoding(beams: 1, maxTokens: 8, startToken: 0, endToken: 2)
        XCTAssertLessThanOrEqual(NFKMLXSeq2SeqDecoder.generate(madlad, source: [5, 6, 7, 2], decoding: decoding).count, 8)

        // TranslateGemma is the shipped Gemma 3 behind the release's translation template:
        // NFKMLXTranslateGemma.backend(directoryURL:precision:) loads a gated google/translategemma-*-it
        // release. The template itself renders without weights.
        let languages = NFKMLXTranslateGemmaTranslator.languageTable(inDirectory: URL(fileURLWithPath: "/nonexistent"))
        XCTAssertTrue(languages.isEmpty, "the table comes from the release's chat_template.jinja")
    }

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
