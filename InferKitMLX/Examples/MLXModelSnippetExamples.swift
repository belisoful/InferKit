//
//  MLXModelSnippetExamples.swift
//  InferKitMLXExamples
//
//  The per-model snippets from Docs/examples.md that load a released checkpoint or directory. Each
//  method runs the snippet's calls on a tiny random stand-in, so the documented API cannot drift
//  without a download. Constructing a model builds MLXNN layers (initializes MLX), so each method
//  skips without a Metal library for MLX (see Tools/mlx-metallib.sh).
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class MLXModelSnippetExamples: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "building a real model initializes MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    // Docs/examples.md: RF-DETR detection. A nil weightsURL stands in for the released checkpoint.
    func testExampleRFDetrDetection() throws {
        try requireMLXRuntime()
        let detector = try NFKMLXRFDetr.backend(variant: .nano, weightsURL: nil, labels: ["person", "bicycle"])
        let result = try detector.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(64)]))
        XCTAssertNotNil(result.detections, "detections (possibly empty)")
        XCTAssertEqual(detector.backendIdentifier, "rf-detr-nano")
    }

    // Docs/examples.md: IP-Adapter image prompts. A tiny random Stable Diffusion model and an adapter
    // file written in the released layout stand in for the SD 1.5 release and ip-adapter_sd15.
    func testExampleIPAdapterImagePrompt() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(14)
        let adapterURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("ip-adapter-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: adapterURL) }

        let model = NFKMLXSDTextToImageModel(configuration: .tiny)
        let attentions = model.pipeline.unet.crossAttentions
        let width = try XCTUnwrap(attentions.first).contextDimensions
        var arrays: [String: MLXArray] = [
            "image_proj.proj.weight": MLXRandom.normal([4 * width, 1024]) * 0.02,
            "image_proj.proj.bias": MLXArray.zeros([4 * width]),
            "image_proj.norm.weight": MLXArray.ones([width]),
            "image_proj.norm.bias": MLXArray.zeros([width]),
        ]
        for (index, attention) in attentions.enumerated() {
            let shape = [attention.dimensions, attention.contextDimensions]
            arrays["ip_adapter.\(index).to_k_ip.weight"] = MLXRandom.normal(shape) * 0.02
            arrays["ip_adapter.\(index).to_v_ip.weight"] = MLXRandom.normal(shape) * 0.02
        }
        try MLX.save(arrays: arrays, url: adapterURL)
        try model.attachImageAdapter(url: adapterURL, scale: 0.7)

        let referenceEmbedding = (0 ..< 1024).map { NSNumber(value: Float(sin(Double($0) * 0.01))) }
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a watercolor lighthouse at dawn",
                     NFKMLXInputImageEmbedding: referenceEmbedding],
            parameters: [NFKParameterSteps: 2, NFKParameterSeed: 42])
        XCTAssertNotNil(try model.makeBackend().runInference(for: request).output(forKey: NFKOutputImage))
    }

    // Docs/examples.md: SD3 ControlNet. A tiny random MMDiT, ControlNet, and autoencoder stand in for
    // the released stages, and random embeddings for what NFKMLXSD3Generator.promptEmbeddings(for:)
    // returns from a release's text stage (NFKMLXSD3GeneratorTests holds those to the reference).
    func testExampleSD3ControlNet() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(15)
        var autoencoder = NFKMLXSDVAEConfiguration()
        autoencoder.latentChannels = 4
        autoencoder.blockChannels = [8, 16]
        autoencoder.layersPerBlock = 1
        autoencoder.normalizationGroups = 4
        autoencoder.scaleFactor = 1.5305
        autoencoder.shiftFactor = 0.0609
        let controlnet = NFKMLXSD3ControlNetNet(.tiny)
        let pipeline = NFKMLXSD3ControlNetPipeline(transformer: NFKMLXSD3TransformerNet(.tiny), controlnet: controlnet,
                                                   vae: NFKMLXSDAutoencoder(configuration: autoencoder))
        let image = pipeline.generate(promptEmbeds: MLXRandom.normal([7, 24]), pooled: MLXRandom.normal([20]),
                                      negativeEmbeds: MLXRandom.normal([7, 24]), negativePooled: MLXRandom.normal([20]),
                                      controlImage: MLXRandom.uniform(low: -1, high: 1, [1, 8, 8, 3]),
                                      controlnetScale: 0.7, latentHeight: 4, latentWidth: 4, steps: 2)
        eval(image)
        XCTAssertEqual(image.shape, [1, 8, 8, 3])
    }

    // Docs/examples.md: FLUX.1 ControlNet. A tiny random transformer, ControlNet, and autoencoder stand in
    // for the released stages, and random embeddings for the text encoder's.
    func testExampleFluxControlNet() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(16)
        var autoencoder = NFKMLXSDVAEConfiguration()
        autoencoder.latentChannels = 2
        autoencoder.blockChannels = [8, 16]
        autoencoder.layersPerBlock = 1
        autoencoder.normalizationGroups = 4
        autoencoder.useQuantConv = false
        autoencoder.scaleFactor = 0.3611
        autoencoder.shiftFactor = 0.1159
        let controlnet = NFKMLXFluxControlNetNet(.tiny)
        let pipeline = NFKMLXFluxControlNetPipeline(transformer: NFKMLXFluxTransformerNet(.tiny), controlnet: controlnet,
                                                    vae: NFKMLXSDAutoencoder(configuration: autoencoder))
        let image = pipeline.generate(promptEmbeds: MLXRandom.normal([5, 24]), pooled: MLXRandom.normal([10]),
                                      controlImage: MLXRandom.uniform(low: -1, high: 1, [1, 8, 8, 3]),
                                      controlnetScale: 0.7, latentHeight: 4, latentWidth: 4, steps: 2)
        eval(image)
        XCTAssertEqual(image.shape, [1, 8, 8, 3])
    }

    // Docs/examples.md: LTX-2 audio and video. A tiny random transformer stands in for the release.
    func testExampleLTX2Transformer() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(17)
        let ltx2 = NFKMLXLTX2TransformerNet(.tiny)
        let (videoVelocity, audioVelocity) = ltx2(
            video: MLXRandom.normal([1, 12, 8]), audio: MLXRandom.normal([1, 4, 6]),
            text: MLXRandom.normal([1, 5, 32]), audioText: MLXRandom.normal([1, 5, 16]),
            timestep: MLXArray.full([1, 12], values: MLXArray(Float(500))),
            audioTimestep: MLXArray.full([1, 4], values: MLXArray(Float(500))),
            sigma: MLXArray([Float(0.5)]), frames: 2, height: 2, width: 3, audioFrames: 4)
        eval(videoVelocity, audioVelocity)
        XCTAssertEqual(videoVelocity.shape, [1, 12, 8], "one velocity per video token")
        XCTAssertEqual(audioVelocity.shape, [1, 4, 6], "one velocity per audio frame")
    }

    // Docs/examples.md: Wan 2.2 Animate. The tiny configuration stands in for the 14B geometry, whose
    // release NFKMLXWanAnimate.loadWeights(into:fromDirectory:precision:) reads (NFKMLXWanAnimateTests).
    func testExampleWanAnimate() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(18)
        let animate = NFKMLXWanAnimate.makeNet(.tiny)
        let cache = NFKMLXWanAnimate.makeCache(layerCount: 2)
        let textStates = MLXRandom.normal([7, 10])
        let clipFeatures = MLXRandom.normal([5, 1280])
        _ = try animate.extractReference(latent: MLXRandom.normal([4, 2, 8, 8]),
                                         condition: MLXRandom.normal([4, 2, 8, 8]),
                                         text: textStates, imageEmbeddings: clipFeatures, into: cache)
        let velocity = try animate.generate(latent: MLXRandom.normal([4, 3, 8, 8]),
                                            condition: MLXRandom.normal([4, 3, 8, 8]), text: textStates,
                                            imageEmbeddings: clipFeatures, timestep: MLXArray([Float(0.35)]),
                                            cache: cache, referenceGrid: (2, 4, 4), videoFrames: 2, videoArea: 16)
        eval(velocity)
        XCTAssertEqual(velocity.shape, [4, 3, 8, 8], "the chunk's latent shape")
    }

    // Docs/examples.md: Kokoro. The factories need the release; here the released geometry at random
    // weights runs through the same phoneme-in, WAV-out speech backend.
    func testExampleKokoroSpeech() throws {
        try requireMLXRuntime()
        XCTAssertTrue(NFKMLXKokoro.responds(to: NSSelectorFromString("kokoroBackendWithRepo:revision:cacheDirectoryURL:voiceName:error:")))
        XCTAssertThrowsError(try NFKMLXKokoro.backend(directoryURL: URL(fileURLWithPath: "/nonexistent-kokoro"),
                                                      voiceName: "af_heart"))
        NFKMLXRandom.seed(19)
        var vocabulary = [String: Int]()
        for scalar in "həlˈoʊ".unicodeScalars { vocabulary[String(scalar)] = Int(scalar.value) % 170 + 1 }
        let speaker = KokoroSpeaker(net: NFKMLXKokoroNet(.v1), voice: MLXArray.zeros([512, 1, 256]) + 0.01,
                                    vocabulary: vocabulary)
        let kokoro = NFKMLXSpeechBackend(identifier: "kokoro",
                                         configuration: NFKMLXSpeechConfiguration(sampleRate: 24000)) { phonemes, _ in
            speaker.net.synthesize(phonemes: phonemes, voice: speaker.voice, vocab: speaker.vocabulary)
        }
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "həlˈoʊ"])
        XCTAssertNotNil(try kokoro.runInference(for: request).output(forKey: NFKOutputAudio) as? NFKAudioAsset)
    }

    // Docs/examples.md: Chatterbox voice cloning. The factories need the 3.2 GB release; the gallery's
    // testAudioModels runs the voice encoder, the speech tokenizer, and T3 at tiny geometries.
    func testExampleChatterboxSpeech() throws {
        XCTAssertTrue(NFKMLXChatterbox.responds(to: NSSelectorFromString(
            "chatterboxBackendWithRepo:revision:cacheDirectoryURL:voiceURL:error:")))
        XCTAssertThrowsError(try NFKMLXChatterbox.backend(directoryURL: URL(fileURLWithPath: "/nonexistent-chatterbox"),
                                                          voiceURL: nil))
    }

    /// Holds the network, voicepack, and vocabulary for capture in the speech backend's `@Sendable`
    /// closure, as the release factory does.
    private final class KokoroSpeaker: @unchecked Sendable {
        let net: NFKMLXKokoroNet
        let voice: MLXArray
        let vocabulary: [String: Int]
        init(net: NFKMLXKokoroNet, voice: MLXArray, vocabulary: [String: Int]) {
            self.net = net
            self.voice = voice
            self.vocabulary = vocabulary
        }
    }

    private static func solid(_ side: Int, value: UInt8 = 128) -> CGImage {
        let pixels = [UInt8](repeating: value, count: side * side * 4)
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    // Docs/examples.md: Text → text, the Qwen3.5 hybrid and Qwen4-Exp. A tiny release directory (its
    // config.json, weights under the release's names, tokenizer, and chat template) stands in for a
    // downloaded one, so the public factory runs end to end.
    func testExampleQwenHybridTextBackend() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(21)
        let releaseDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: releaseDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: releaseDirectory) }
        let decoder: [String: Any] = [
            "model_type": "qwen3_5_text", "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
            "vocab_size": 12, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "rope_parameters": ["rope_theta": 10_000, "partial_rotary_factor": 0.25],
            "linear_num_key_heads": 2, "linear_key_head_dim": 8, "linear_num_value_heads": 4,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "tie_word_embeddings": false,
        ]
        try JSONSerialization.data(withJSONObject: ["model_type": "qwen3_5", "text_config": decoder])
            .write(to: releaseDirectory.appendingPathComponent("config.json"))
        let trained = NFKMLXHybridLanguage.makeNet(try NFKMLXHybridLanguage.configuration(
            fromHuggingFace: releaseDirectory.appendingPathComponent("config.json")))
        let released = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (NFKMLXHybridLanguage.referenceKey(for: key),
             key.hasSuffix("conv1d.weight") && value.ndim == 3 ? value.transposed(0, 2, 1) : value)
        })
        try save(arrays: released, url: releaseDirectory.appendingPathComponent("model.safetensors"))
        let vocabulary: [String: Int] = ["h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6,
                                         "Ġ": 7, "Ċ": 8, "<eos>": 9]
        try JSONSerialization.data(withJSONObject: vocabulary)
            .write(to: releaseDirectory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\nh e\nl l\nhe ll\nhell o\n".write(
            to: releaseDirectory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        try JSONSerialization.data(withJSONObject: [
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": ["10": ["content": "<|im_start|>"], "11": ["content": "<|im_end|>"]],
        ]).write(to: releaseDirectory.appendingPathComponent("tokenizer_config.json"))
        try ("{% for m in messages %}<|im_start|>{{ m.content }}<|im_end|>{% endfor %}"
             + "{% if add_generation_prompt %}<|im_start|>{% endif %}")
            .write(to: releaseDirectory.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)

        let qwen = try NFKMLXHybridLanguage.backend(directoryURL: releaseDirectory)
        let result = try qwen.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputMessages: [["role": "user", "content": "hello"]]],
            parameters: [NFKParameterMaxTokens: 3]))
        XCTAssertNotNil(result.text, "a reply, however short, from random weights")
        XCTAssertEqual(qwen.backendIdentifier, "qwen3.5")
    }

    // Docs/examples.md: SANA. A tiny transformer directory, read through the same configuration reader
    // and loader pipeline(directoryURL:) uses, and a tiny autoencoder stand in for the release; random
    // captions stand in for Gemma 2's.
    func testExampleSANAPipeline() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(22)
        let transformerDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: transformerDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: transformerDirectory) }
        try JSONSerialization.data(withJSONObject: [
            "in_channels": 8, "num_attention_heads": 2, "attention_head_dim": 8, "num_layers": 2,
            "num_cross_attention_heads": 2, "cross_attention_head_dim": 8, "caption_channels": 12,
            "mlp_ratio": 2.0,
        ]).write(to: transformerDirectory.appendingPathComponent("config.json"))
        let geometry = try NFKMLXSANATransformerNet.configuration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json"))
        let released = Dictionary(uniqueKeysWithValues: NFKMLXSANATransformerNet(geometry).parameters().flattened()
            .map { key, value in (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value) })
        try save(arrays: released, url: transformerDirectory.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        let transformer = NFKMLXSANATransformerNet(geometry)
        try NFKMLXSANATransformerNet.loadWeights(into: transformer, fromDirectory: transformerDirectory)

        let sana = NFKMLXSANAPipeline(transformer: transformer, vae: NFKMLXDCAutoencoderNet(.tiny))
        let image = sana.generate(promptEmbeds: MLXRandom.normal([6, 12]), negativeEmbeds: MLXRandom.normal([6, 12]),
                                  latentHeight: 4, latentWidth: 4, steps: 2, guidance: 4.5)
        eval(image)
        XCTAssertEqual(image.shape[3], 3, "an RGB image")
    }
}
