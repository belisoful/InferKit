//
//  NFKMLXHalfPrecisionArithmeticTimingTests.swift
//  InferKitMLXTests
//
//  The whole-forward cost of each `NFKMLXHalfPrecision.arithmetic` mode, for every network whose
//  half-precision path reaches `NFKReferenceRounding.flashAttention`. Each network runs at its released
//  width, head count, and head size, with random weights at the half type its parity test uses, over a
//  typical generation input. Where a release exceeds 6 GB at half precision, only its block count is
//  reduced: each block's attention share is unchanged, so the ratio of the two modes holds.
//
//  The class also times half-precision networks that reach `NFKReferenceRounding.layerNorm` through
//  `NFKLayerNorm` and never reach the flash kernel. In those the layer norm is the only operation the two
//  modes compute differently. The float32 layer norm Wan's blocks take is also timed alone.
//
//  Each test prints one `TIMING arithmetic` line per measured forward, one `ROUNDS arithmetic` line with
//  each round's time in run order, and one `MEMORY arithmetic` line. These are
//  measurements, not gates; they run only with `IK_ARITHMETIC_TIMING` set to "1" (environment or
//  `~/.inferkit-validation.json`).
//

import XCTest
import Foundation
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXHalfPrecisionArithmeticTimingTests: XCTestCase {

    override func tearDown() {
        NFKMLXHalfPrecision.arithmetic = .fast
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func requireOptIn() throws {
        try XCTSkipUnless(NFKMLXValidationConfig.environment["IK_ARITHMETIC_TIMING"] == "1",
                          "builds released-width networks and times them; set IK_ARITHMETIC_TIMING to \"1\" to run")
    }

    /// Applies `NFKMLXGPU.applyStandingLimits()` for the rest of the test and restores MLX's limits after it.
    private func applyStandingLimits() {
        let (cacheLimit, memoryLimit) = (NFKMLXGPU.cacheLimit, NFKMLXGPU.memoryLimit)
        NFKMLXGPU.applyStandingLimits()
        addTeardownBlock {
            NFKMLXGPU.setCacheLimit(cacheLimit)
            NFKMLXGPU.setMemoryLimit(memoryLimit)
        }
    }

    /// Casts every parameter of `module` to `dtype`, evaluates them, and names their size.
    private func castWeights(_ module: Module, to dtype: DType) -> String {
        module.update(parameters: module.parameters().mapValues { $0.asType(dtype) })
        let parameters = module.parameters().flattened().map(\.1)
        eval(parameters)
        let bytes = parameters.reduce(0) { $0 + $1.nbytes }
        return "\(dtype == .float16 ? "f16" : "bf16") weights \(String(format: "%.2f", Double(bytes) / 1e9)) GB"
    }

    /// A standard normal array at `dtype`, evaluated.
    private func normal(_ shape: [Int], _ dtype: DType) -> MLXArray {
        let array = MLXRandom.normal(shape).asType(dtype)
        eval(array)
        return array
    }

    /// Times `forward` in each arithmetic: one warm-up forward per mode, then three rounds alternating
    /// reference and fast, each forward evaluated and the GPU synchronized. Prints the median of each mode
    /// and each round's time in run order.
    private func time(_ family: String, _ note: String, _ forward: () throws -> [MLXArray]) throws {
        defer { NFKMLXHalfPrecision.arithmetic = .fast }
        func run(_ mode: NFKMLXArithmetic) throws -> Double {
            NFKMLXHalfPrecision.arithmetic = mode
            NFKMLXGPU.synchronize()
            let start = DispatchTime.now().uptimeNanoseconds
            eval(try forward())
            NFKMLXGPU.synchronize()
            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        }
        func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }

        NFKMLXGPU.resetPeakMemory()
        _ = try run(.reference)
        _ = try run(.fast)
        var reference = [Double](), fast = [Double]()
        for _ in 0 ..< 3 {
            reference.append(try run(.reference))
            fast.append(try run(.fast))
        }
        let referenceTime = median(reference), fastTime = median(fast)
        XCTAssertGreaterThan(referenceTime, 0)
        XCTAssertGreaterThan(fastTime, 0)
        let ratio = referenceTime / fastTime
        print("TIMING arithmetic \(family) \(note): reference \(String(format: "%.1f", referenceTime)) ms, "
              + "fast \(String(format: "%.1f", fastTime)) ms, reference/fast \(String(format: "%.3f", ratio))x "
              + "(\(String(format: "%.1f", (ratio - 1) * 100))% slower)")
        func milliseconds(_ values: [Double]) -> String {
            values.map { String(format: "%.1f", $0) }.joined(separator: ", ")
        }
        print("ROUNDS arithmetic \(family): reference \(milliseconds(reference)) ms; fast \(milliseconds(fast)) ms")
        print("MEMORY arithmetic \(family): peak \(String(format: "%.2f", Double(NFKMLXGPU.peakMemory) / 1e9)) GB")
    }

    // MARK: Stable Diffusion UNets and autoencoders

    // Stable Diffusion 1.5's UNet, released geometry, float16; a 64x64 latent (512 px) at batch 2 for
    // classifier-free guidance over 77 CLIP tokens.
    func testSD15UNetForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let unet = NFKMLXSDUNet(configuration: NFKMLXSDTextToImageConfiguration.stableDiffusion15.unet)
        let weights = castWeights(unet, to: .float16)
        let latent = normal([2, 64, 64, 4], .float16), context = normal([2, 77, 768], .float16)
        let timestep = MLXArray([Float(500)])
        try time("sd15-unet", "released, \(weights), latent 2x64x64x4, text 2x77") {
            [unet(latent, timestep: timestep, context: context)]
        }
    }

    // SDXL's UNet, released geometry, float16; a 128x128 latent (1024 px) at batch 2 for classifier-free
    // guidance over 77 tokens of both CLIP towers, the pooled embedding, and the size descriptor, under the
    // standing limits. Under MLX's default limits single rounds ran 2 to 6 times the others' time.
    func testSDXLUNetForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        applyStandingLimits()
        MLXRandom.seed(0)
        let unet = NFKMLXSDUNet(configuration: .sdxl)
        let weights = castWeights(unet, to: .float16)
        let latent = normal([2, 128, 128, 4], .float16), context = normal([2, 77, 2048], .float16)
        let sizes: [Float] = [1024, 1024, 0, 0, 1024, 1024]
        let added = NFKSDAddedConditioning(pooled: normal([2, 1280], .float16), timeIds: MLXArray(sizes + sizes, [2, 6]))
        let timestep = MLXArray([Float(500)])
        try time("sdxl-unet", "released, \(weights), latent 2x128x128x4, text 2x77") {
            [unet(latent, timestep: timestep, context: context, added: added)]
        }
    }

    // Stable Diffusion 1.5's autoencoder, released geometry, float16; the decode of a 64x64 latent to 512 px.
    func testSD15AutoencoderDecodeTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let vae = NFKMLXSDAutoencoder(configuration: NFKMLXSDTextToImageConfiguration.stableDiffusion15.vae)
        let weights = castWeights(vae, to: .float16)
        let latent = normal([1, 64, 64, 4], .float16)
        try time("sd15-vae-decode", "released, \(weights), latent 1x64x64x4 to 512x512") {
            [vae.decode(latent)]
        }
    }

    // FLUX.1's autoencoder, released geometry, bfloat16; the decode of a 128x128 latent to 1024 px, under the
    // standing limits. Under MLX's default limits its rounds grow the buffer cache to 25 GB on a 32 GB Mac,
    // where the decode has ended in a GPU command-buffer timeout.
    func testFluxAutoencoderDecodeTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        applyStandingLimits()
        MLXRandom.seed(0)
        let vae = NFKMLXSDAutoencoder(configuration: .flux)
        let weights = castWeights(vae, to: .bfloat16)
        let latent = normal([1, 128, 128, 16], .bfloat16)
        try time("flux-vae-decode", "released, \(weights), latent 1x128x128x16 to 1024x1024") {
            [vae.decode(latent)]
        }
    }

    // MARK: Diffusion transformers

    // SD3.5-medium's MMDiT, released geometry (24 layers, 13 dual-attention), bfloat16; a 1024 px latent
    // (4,096 image tokens) over 333 text tokens (77 CLIP + 256 T5).
    func testSD35MediumTransformerForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let net = NFKMLXSD3TransformerNet(.sd35Medium)
        let weights = castWeights(net, to: .bfloat16)
        let latent = normal([1, 16, 128, 128], .bfloat16), encoder = normal([1, 333, 4096], .bfloat16)
        let pooled = normal([1, 2048], .bfloat16), timestep = MLXArray([Float(500)]).asType(.bfloat16)
        try time("sd35-medium", "released 24 layers, \(weights), 1024x1024 (4096 image + 333 text tokens)") {
            [net(latent, encoderHidden: encoder, pooled: pooled, timestep: timestep)]
        }
    }

    // FLUX.1 [schnell]'s transformer at its released width (24 heads of 128), bfloat16, cut from 19 double
    // and 38 single blocks to 3 and 6; a 1024 px packed latent (4,096 tokens) over 512 T5 tokens.
    func testFluxTransformerForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        var configuration = NFKMLXFluxConfiguration.schnell
        configuration.numLayers = 3
        configuration.numSingleLayers = 6
        let net = NFKMLXFluxTransformerNet(configuration)
        let weights = castWeights(net, to: .bfloat16)
        let hidden = normal([1, 4096, 64], .bfloat16), encoder = normal([1, 512, 4096], .bfloat16)
        let pooled = normal([1, 768], .bfloat16), timestep = MLXArray([Float(0.5)]).asType(.bfloat16)
        let imageIds = NFKMLXFluxTransformerNet.imageIds(height: 64, width: 64)
        try time("flux1-schnell", "3 of 19 double + 6 of 38 single blocks, \(weights), 1024x1024 (4096 image + 512 text tokens)") {
            [net(hidden, encoderHidden: encoder, pooled: pooled, timestep: timestep, guidance: nil, imageIds: imageIds)]
        }
    }

    // FLUX.2 [klein] 4B's transformer at its released width (24 heads of 128), bfloat16, cut from 5 double
    // and 20 single blocks to 3 and 12; a 1024 px packed latent (4,096 tokens) over 512 Qwen3 tokens.
    func testFlux2KleinTransformerForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        var configuration = NFKMLXFlux2Configuration.klein4B
        configuration.numLayers = 3
        configuration.numSingleLayers = 12
        let net = NFKMLXFlux2TransformerNet(configuration)
        let weights = castWeights(net, to: .bfloat16)
        let hidden = normal([1, 4096, 128], .bfloat16), encoder = normal([1, 512, 7680], .bfloat16)
        let timestep = MLXArray([Float(0.5)])
        let imageIds = NFKMLXFlux2TransformerNet.imageIds(height: 64, width: 64)
        let textIds = NFKMLXFlux2TransformerNet.textIds(length: 512)
        try time("flux2-klein-4b", "3 of 5 double + 12 of 20 single blocks, \(weights), 1024x1024 (4096 image + 512 text tokens)") {
            [net(hidden, encoderHidden: encoder, timestep: timestep, guidance: nil, imageIds: imageIds, textIds: textIds)]
        }
    }

    // Qwen-Image 2.1's transformer at its released width (32 heads of 128), bfloat16, cut from 32 layers to
    // 12; a 1024 px latent (4,096 tokens) after 256 caption tokens, under the block-causal mask.
    func testQwenImageTransformerForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        var configuration = NFKMLXQwenImageConfiguration.base
        configuration.layers = 12
        let net = NFKMLXQwenImage.makeNet(configuration)
        let weights = castWeights(net, to: .bfloat16)
        let latents = normal([4096, 64], .bfloat16), encoder = normal([256, 4096], .bfloat16)
        let imageMask = [Bool](repeating: false, count: 256) + [Bool](repeating: true, count: 1024)
        try time("qwen-image-2.1", "12 of 32 layers, \(weights), 1024x1024 (4096 image + 256 text tokens)") {
            [net(latents: latents, encoderHidden: encoder, timestep: 0.5,
                 imageShapes: [(frame: 1, height: 64, width: 64)], imageMask: imageMask)]
        }
    }

    // Z-Image's transformer at its released width (30 heads of 128), bfloat16, cut from 30 unified layers
    // to 10 with both released two-layer refiners; a 1024 px latent (4,096 tokens) over 512 caption tokens.
    func testZImageTransformerForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        var configuration = NFKMLXZImageConfiguration.base
        configuration.layers = 10
        let net = NFKMLXZImageTransformerNet(configuration)
        let weights = castWeights(net, to: .bfloat16)
        let latent = normal([16, 1, 128, 128], .bfloat16), caption = normal([512, 2560], .bfloat16)
        let t = MLXArray([Float(0.5)])
        try time("z-image", "10 of 30 layers + 2+2 refiners, \(weights), 1024x1024 (4096 image + 512 text tokens)") {
            [net(latent, capFeats: caption, t: t)]
        }
    }

    // Wan 2.1 T2V 1.3B's transformer, released geometry (12 heads of 128, 30 layers, feed-forward 8960),
    // bfloat16; 480x832 at 17 frames (latent 5x60x104, 7,800 tokens) over 512 umT5 tokens. At 33 frames
    // the attention's float32 scores alone would take 9.5 GB.
    func testWanTransformerForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let net = NFKMLXWanTransformerNet(NFKMLXWanConfiguration(heads: 12, headDim: 128, layers: 30, ffnDim: 8960))
        let weights = castWeights(net, to: .bfloat16)
        let latent = normal([16, 5, 60, 104], .bfloat16), text = normal([512, 4096], .bfloat16)
        let t = MLXArray([Float(500)])
        try time("wan2.1-t2v-1.3b", "released 30 layers, \(weights), 480x832x17 (latent 5x60x104, 7800 tokens), text 512") {
            [net(latent, text: text, t: t)]
        }
    }

    // Wan 2.2 Animate 14B's transformer at its released width (40 heads of 128), bfloat16, cut from 40
    // layers to 6; at 480x832, the extract pass over one reference latent frame (1,560 tokens), then a
    // generation pass over a two-latent-frame chunk (3,120 tokens), each over 512 text tokens and 257 CLIP
    // image tokens. The generation pass's self-attention is its own masked softmax; only its
    // cross-attentions reach the flash kernel.
    func testWanAnimateForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        var configuration = NFKMLXWanAnimateConfiguration.base
        configuration.layers = 6
        let net = NFKMLXWanAnimateNet(configuration)
        let weights = castWeights(net, to: .bfloat16)
        let text = normal([512, 4096], .bfloat16), image = normal([257, 1280], .bfloat16)
        let referenceLatent = normal([16, 1, 60, 104], .bfloat16)
        let referenceCondition = normal([20, 1, 60, 104], .bfloat16)
        let latent = normal([16, 2, 60, 104], .bfloat16), condition = normal([20, 2, 60, 104], .bfloat16)
        let timestep = MLXArray([Float(500)])
        let cache = NFKMLXWanAnimateKVCache(layerCount: configuration.layers)
        try time("wan2.2-animate-14b extract", "6 of 40 layers, \(weights), 480x832 reference 1 latent frame (1560 tokens)") {
            [try net.extractReference(latent: referenceLatent, condition: referenceCondition, text: text,
                                      imageEmbeddings: image, into: cache)]
        }
        try time("wan2.2-animate-14b generate", "6 of 40 layers, \(weights), 480x832 chunk 2 latent frames (3120 tokens)") {
            [try net.generate(latent: latent, condition: condition, text: text, imageEmbeddings: image,
                              timestep: timestep, cache: cache, referenceGrid: (1, 30, 52), videoFrames: 2,
                              videoArea: 1560)]
        }
    }

    // LTX-2.3's audio-video transformer at its released widths (video 32 heads of 128, audio 32 of 64),
    // bfloat16, cut from 48 layers to 5; 512x768 at 49 frames (latent 7x16x24, 2,688 video tokens) with
    // its 51 audio latent frames, over 1,024 text tokens per stream.
    func testLTX2TransformerForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        var configuration = NFKMLXLTX2Configuration.ltx23
        configuration.numLayers = 5
        let net = NFKMLXLTX2TransformerNet(configuration)
        let weights = castWeights(net, to: .bfloat16)
        let video = normal([1, 2688, 128], .bfloat16), audio = normal([1, 51, 128], .bfloat16)
        let text = normal([1, 1024, 4096], .bfloat16), audioText = normal([1, 1024, 2048], .bfloat16)
        let timestep = (MLXArray.ones([1, 2688]) * 500).asType(.bfloat16)
        let audioTimestep = (MLXArray.ones([1, 51]) * 500).asType(.bfloat16)
        let sigma = MLXArray([Float(0.5)]).asType(.bfloat16)
        try time("ltx-2.3", "5 of 48 layers, \(weights), 512x768x49 (2688 video + 51 audio tokens), text 1024") {
            let (videoVelocity, audioVelocity) = net(
                video: video, audio: audio, text: text, audioText: audioText, timestep: timestep,
                audioTimestep: audioTimestep, sigma: sigma, frames: 7, height: 16, width: 24, audioFrames: 51)
            return [videoVelocity, audioVelocity]
        }
    }

    // MARK: Segmentation

    // SAM ViT-B, released geometry, bfloat16; a 1024 px image segmented around one click, then the mask
    // decoder alone on the cached image embedding, which is what each further click costs. Only the mask
    // decoder's attention reaches the flash kernel.
    func testSAMForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let net = NFKMLXSAM.makeNet(.vitB)
        let weights = castWeights(net, to: .bfloat16)
        let image = normal([1, 1024, 1024, 3], .bfloat16)
        try time("sam-vit-b segment", "released, \(weights), 1024x1024 image and one click") {
            [net.segment(image, pointX: 0.5, pointY: 0.5)]
        }
        let embedding = net.imageEncoder(image)
        eval(embedding)
        let grid = embedding.dim(1)
        try time("sam-vit-b mask decoder", "released, \(weights), 64x64 embedding and one click") {
            let decoded = net.maskDecoder(embedding + net.promptEncoder.dense(grid: grid),
                                          imagePE: net.promptEncoder.positionEncoding.grid(grid, grid),
                                          sparse: net.promptEncoder.sparse(pointX: 0.5, pointY: 0.5, positive: true))
            return [decoded.masks, decoded.iou]
        }
    }

    // SAM 2 hiera-tiny as Sa2VA grounds with it (image encoder, prompt encoder, mask decoder), released
    // geometry, bfloat16; a 1024 px image and one [SEG] embedding.
    func testSAM2GroundingForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let grounding = NFKSa2VAGroundingEncoder(NFKMLXSAM2TrackerConfiguration())
        let weights = castWeights(grounding, to: .bfloat16)
        let image = normal([1, 1024, 1024, 3], .bfloat16), embedding = normal([1, 1, 256], .bfloat16)
        try time("sam2-hiera-tiny grounding", "released, \(weights), 1024x1024 image") {
            let masks = grounding.segment(image: image, languageEmbedding: embedding)
            return [masks.highResolution, masks.lowResolution]
        }
    }

    // SAM 3's tracker as Sa2VA grounds with it (ViT-L 32 layers, the tracker neck, SAM 2's mask decoder),
    // released geometry, bfloat16 weights under the release's autocast; a 1008 px image and one [SEG]
    // embedding.
    func testSAM3GroundingForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let grounding = NFKSa2VASAM3GroundingEncoder()
        let weights = castWeights(grounding, to: .bfloat16)
        let image = normal([1, 1008, 1008, 3], .float32), embedding = normal([1, 1, 256], .bfloat16)
        try time("sam3 grounding", "released, \(weights), 1008x1008 image") {
            let masks = grounding.segment(image: image, languageEmbedding: embedding)
            return [masks.highResolution, masks.lowResolution]
        }
    }

    // MARK: Music

    // MiniMax Music 3's RVQ depth decoder, released geometry (4 layers, 16 heads of 256), bfloat16; one
    // audio frame's seven depth steps over sequences of 2 to 8 positions at batch 2 (the guidance pair),
    // each evaluated before the next as sampling requires.
    func testMusic3DepthDecoderFrameTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let net = NFKMLXMusic3.makeDepthDecoder()
        let weights = castWeights(net, to: .bfloat16)
        let steps = normal([2, 8, 4096], .bfloat16)
        try time("music3 depth frame", "released, \(weights), 7 steps of 2...8 positions at batch 2") {
            var last = [MLXArray]()
            for length in 2 ... 8 {
                let hidden = net.hiddenStates(steps[0..., 0 ..< length, 0...])
                eval(hidden)
                last.append(hidden[0..., length - 1])
            }
            return last
        }
    }

    // MARK: Layer-norm-only networks

    // Stable Diffusion 1.5's CLIP ViT-L/14 text encoder, released geometry (12 layers of 768, 12 heads),
    // float16; 77 tokens at batch 2 for classifier-free guidance, through the final layer norm.
    func testSD15TextEncoderForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let configuration = NFKMLXSDTextEncoderConfiguration.stableDiffusion15
        let net = NFKMLXSDTextEncoderNet(configuration: configuration)
        let weights = castWeights(net, to: .float16)
        let length = configuration.contextLength
        let ids = (0 ..< 2 * length).map { Int32($0 * 7919 % configuration.vocabularySize) }
        let tokens = MLXArray(ids, [2, length])
        let mask = NFKCLIPInit.causalMask(length)
        try time("sd15-clip-text", "released, \(weights), 2x\(length) tokens") {
            let embedded = net.tokenEmbedding(tokens) + net.positionalEmbedding[0 ..< length]
            return [net.lnFinal(net.transformer(embedded, mask: mask))]
        }
    }

    // SigLIP 2 so400m-patch14-384's vision tower, released geometry (27 layers of 1152, 16 heads), bfloat16;
    // one 384x384 image (729 patches) through the post layer norm and the pooling head.
    func testSigLIP2So400mVisionForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let configuration = NFKMLXSigLIP2Configuration.so400mPatch14At384.vision
        let net = NFKSigLIP2VisionNet(configuration)
        let weights = castWeights(net, to: .bfloat16)
        let side = configuration.imageSize
        let image = normal([1, side, side, 3], .bfloat16)
        try time("siglip2-so400m-p14-384 vision", "released, \(weights), \(side)x\(side) image (\(configuration.positionCount) patches)") {
            [net(image)]
        }
    }

    // Whisper large-v3, released geometry (32 encoder and 32 decoder layers of 1280, 20 heads, 128 mel
    // bands), float16; the encoder over 30 s of log-mel (3,000 frames to 1,500 positions), then one greedy
    // decode step over a 100-token prefix cross-attending to that encoding. Decoding reruns the whole
    // prefix each step, so the second line is the cost of one token.
    func testWhisperLargeV3ForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let configuration = NFKMLXWhisperConfiguration.largeV3
        let net = NFKMLXWhisperNet(configuration)
        let weights = castWeights(net, to: .float16)
        let mel = normal([1, 3000, configuration.nMels], .float16)
        try time("whisper-large-v3 encoder", "released, \(weights), 30 s log-mel (1x3000x\(configuration.nMels))") {
            [net.encoder(mel)]
        }
        let audio = net.encoder(mel)
        eval(audio)
        let prefix = 100
        let tokens = MLXArray((0 ..< prefix).map { Int32($0 * 7919 % configuration.endToken) }, [1, prefix])
        try time("whisper-large-v3 decode step", "released, \(weights), \(prefix)-token prefix over 1500 audio positions") {
            [net.decoder(tokens, audio: audio)[0, prefix - 1]]
        }
    }

    // Gemma 3's SigLIP so400m vision tower, released geometry (27 layers of 1152, 16 heads, patch 14),
    // bfloat16; one 896x896 image (4,096 patches) through the post layer norm.
    func testGemma3VisionForwardTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        let configuration = NFKMLXSigLIPConfiguration.gemma3
        let net = NFKMLXGemma3VisionNet(configuration)
        let weights = castWeights(net, to: .bfloat16)
        let side = configuration.imageSize
        let image = normal([1, side, side, 3], .bfloat16)
        try time("gemma3-vision", "released, \(weights), \(side)x\(side) image (\(configuration.positionCount) patches)") {
            [net(image)]
        }
    }

    // MARK: Operations

    // The float32 layer norm without the affine that Wan's blocks take, alone, over Wan 2.1 1.3B's 7,800
    // tokens at 480x832x17 at its width (1,536) and at the 14B's (5,120).
    func testFloat32LayerNormTimeInEachArithmetic() throws {
        try requireMLXRuntime()
        try requireOptIn()
        MLXRandom.seed(0)
        for width in [1536, 5120] {
            let x = normal([1, 7800, width], .float32)
            try time("float32-layer-norm-\(width)", "7800 tokens of \(width), no affine") {
                [NFKReferenceRounding.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)]
            }
        }
    }
}
