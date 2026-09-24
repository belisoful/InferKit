//
//  NFKMLXGeneratorStagingTests.swift
//  InferKitMLXTests
//
//  The end-to-end generators for Qwen-Image, LTX-Video and Wan: a staged generator and a resident one
//  over the same weights produce the same output, and a staged one holds no stage between runs. The
//  arithmetic of each stage and of each pipeline's glue is measured in the parity tests.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXGeneratorStagingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// One set of weights, applied to a fresh module on every load, so a staged reload reads the same
    /// network a resident load holds.
    private func captured(_ module: Module) -> [(String, MLXArray)] {
        let parameters = module.parameters().flattened()
        eval(parameters.map(\.1))
        return parameters
    }

    // MARK: Qwen-Image

    private var qwenImageRelease: URL? {
        NFKMLXValidationConfig.environment["IK_VAL_QWEN_IMAGE"].map { URL(fileURLWithPath: $0) }
    }

    func testQwenImageStagingReleasesEachStageAndChangesNothing() throws {
        try requireMLXRuntime()
        guard let release = qwenImageRelease,
              let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: release.appendingPathComponent("processor")) else {
            throw XCTSkip("set IK_VAL_QWEN_IMAGE (the release directory, for its tokenizer)")
        }
        MLXRandom.seed(41)
        // The M-RoPE sections the encoder reads cover 64 frequency pairs, so the head is 128 wide.
        var language = NFKMLXLanguageConfiguration.tiny
        language.vocabularySize = 151_936
        language.headCount = 2
        language.keyValueHeadCount = 1
        language.headDimensions = 128
        var geometry = NFKMLXQwenImageConfiguration.tiny
        geometry.inChannels = 4
        geometry.outChannels = 4
        geometry.contextInDimensions = language.hiddenSize

        let decoderWeights = captured(NFKMLXLanguageNet(language))
        let transformerWeights = captured(NFKMLXQwenImage.makeNet(geometry))
        let vaeWeights = captured(NFKMLXQwenImageVAE.makeNet(.qwenImage21Tiny))
        var loads = (encoder: 0, pipeline: 0)
        func generator(resident: Bool) throws -> NFKMLXQwenImageGenerator {
            let generator = NFKMLXQwenImageGenerator(
                resident: resident, tokenizer: tokenizer,
                loadTextEncoder: {
                    loads.encoder += 1
                    let decoder = NFKMLXLanguageNet(language)
                    try NFKMLXWeights.apply(decoderWeights, to: decoder)
                    return decoder
                },
                loadPipeline: {
                    loads.pipeline += 1
                    let transformer = NFKMLXQwenImage.makeNet(geometry)
                    try NFKMLXWeights.apply(transformerWeights, to: transformer)
                    let vae = NFKMLXQwenImageVAE.makeNet(.qwenImage21Tiny)
                    try NFKMLXWeights.apply(vaeWeights, to: vae)
                    return NFKMLXQwenImagePipeline(transformer: transformer, vae: vae)
                })
            if resident {
                try generator.loadResident()
            }
            generator.steps = 2
            generator.guidance = 3
            return generator
        }

        let resident = try generator(resident: true)
        let residentImages = try (0 ..< 2).map { _ in
            try resident.image(forPrompt: "a red fox", negativePrompt: "blurry", width: 64, height: 64, seed: 5)
        }
        XCTAssertEqual(loads.encoder, 1, "a resident generator loads the encoder once")
        XCTAssertEqual(loads.pipeline, 1)
        XCTAssertTrue(resident.holdsStagesResident)

        loads = (0, 0)
        let staged = try generator(resident: false)
        XCTAssertFalse(staged.isHoldingTextEncoder || staged.isHoldingPipeline, "a staged generator loads nothing up front")
        let stagedImages = try (0 ..< 2).map { _ in
            try staged.image(forPrompt: "a red fox", negativePrompt: "blurry", width: 64, height: 64, seed: 5)
        }
        XCTAssertEqual(loads.encoder, 2, "a staged generator loads the encoder once per image, both prompts together")
        XCTAssertEqual(loads.pipeline, 2)
        XCTAssertFalse(staged.isHoldingTextEncoder || staged.isHoldingPipeline, "and holds neither after")
        XCTAssertEqual(stagedImages[0].dim(-1), 4, "the release decodes RGBA")
        for (residentImage, stagedImage) in zip(residentImages, stagedImages) {
            XCTAssertEqual(stagedImage.reshaped([-1]).asArray(Float.self),
                           residentImage.reshaped([-1]).asArray(Float.self), "staging does not change the image")
        }
    }

    // The released Qwen-Image 2.1 runs end to end through its factory. On a 32 GB machine the plan
    // stages it, which is the only way its 16 GB encoder and 14 GB transformer run there at all.
    func testQwenImageReleaseRunsEndToEnd() throws {
        try requireMLXRuntime()
        guard let release = qwenImageRelease else { throw XCTSkip("set IK_VAL_QWEN_IMAGE") }
        let start = Date()
        let generator = try NFKMLXQwenImageGenerator.generator(directoryURL: release)
        generator.steps = 4
        let image = try generator.image(forPrompt: "A red fox sitting in fresh snow, photograph",
                                        width: 256, height: 256, seed: 0)
        let seconds = Date().timeIntervalSince(start)
        XCTAssertEqual(image.shape, [256, 256, 4])
        let values = image.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        let mean = values.reduce(0, +) / Float(values.count)
        let spread = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(values.count)
        XCTAssertGreaterThan(spread, 1e-3, "the image is not a flat field")
        print("VALIDATION runtime qwenimage21-e2e: resident \(generator.holdsStagesResident), "
              + "256x256 at 4 steps in \(String(format: "%.1f", seconds)) s")
        if let cgImage = try? NFKMLXImageBridge.cgImage(from: image, options: NFKMLXImageOptions()),
           let destination = CGImageDestinationCreateWithURL(
            FileManager.default.temporaryDirectory.appendingPathComponent("qwenimage21-e2e.png") as CFURL,
            "public.png" as CFString, 1, nil) {
            CGImageDestinationAddImage(destination, cgImage, nil)
            CGImageDestinationFinalize(destination)
        }
    }

    // The released Wan 2.1 T2V 1.3B runs end to end through its factory: umT5-XXL (stored float32,
    // 22.7 GB) and the transformer take turns on a 32 GB machine.
    func testWanReleaseRunsEndToEnd() throws {
        try requireMLXRuntime()
        guard let release = NFKMLXValidationConfig.environment["IK_VAL_WAN21_T2V_1_3B"] else {
            throw XCTSkip("set IK_VAL_WAN21_T2V_1_3B (Wan-AI/Wan2.1-T2V-1.3B-Diffusers)")
        }
        // The release may sit on a network share this process is not granted, or be mid-download.
        let last = URL(fileURLWithPath: release)
            .appendingPathComponent("transformer/diffusion_pytorch_model-00002-of-00002.safetensors")
        try XCTSkipUnless(FileManager.default.isReadableFile(atPath: last.path),
                          "the Wan release at \(release) is not readable from this process")
        let start = Date()
        let generator = try NFKMLXWanVideoGenerator.generator(directoryURL: URL(fileURLWithPath: release))
        generator.steps = 20
        let clip = try generator.video(forPrompt: "A red fox walking through fresh snow in a pine forest, "
                                       + "soft morning light, cinematic", frames: 17, width: 832, height: 480, seed: 0)
        let seconds = Date().timeIntervalSince(start)
        XCTAssertEqual(clip.shape, [17, 480, 832, 3])
        let values = clip.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        let mean = values.reduce(0, +) / Float(values.count)
        let spread = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(values.count)
        XCTAssertGreaterThan(spread, 1e-3, "the clip is not a flat field")
        print("VALIDATION runtime wan21-t2v-1.3b-e2e: resident \(generator.holdsStagesResident), umT5 float32 "
              + "\(generator.encodesInFloat32), 17x480x832 at 20 steps in \(String(format: "%.1f", seconds)) s")
        for index in [0, 8, 16] {
            if let image = try? NFKMLXImageBridge.cgImage(from: clip[index], options: NFKMLXImageOptions()),
               let destination = CGImageDestinationCreateWithURL(
                FileManager.default.temporaryDirectory.appendingPathComponent("wan21-e2e-\(index).png") as CFURL,
                "public.png" as CFString, 1, nil) {
                CGImageDestinationAddImage(destination, image, nil)
                CGImageDestinationFinalize(destination)
            }
        }
    }

    // MARK: The releases the generators read, held by shape

    /// A release component's config and its tensors' shapes, fetched from its headers alone
    /// (`Tools/validation-assets/shapes.py`).
    private func shapes(_ name: String) throws -> (config: URL, shapes: [String: [Int]]) {
        guard let root = NFKMLXValidationConfig.environment["IK_SHAPES_ROOT"] else {
            throw XCTSkip("set IK_SHAPES_ROOT")
        }
        let directory = URL(fileURLWithPath: root).appendingPathComponent(name)
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("shapes.json")) else {
            throw XCTSkip("fetch \(name) with Tools/validation-assets/shapes.py")
        }
        let shapes = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: [Int]])
        return (directory.appendingPathComponent("config.json"), shapes)
    }

    /// Every parameter `module` builds, against every released tensor as the loader maps it: none
    /// missing, none left over, none at another shape.
    private func assertStructure(_ name: String, module: Module, released: [String: [Int]],
                                 file: StaticString = #filePath, line: UInt = #line) {
        let built = Dictionary(module.parameters().flattened().map { ($0.0, $0.1.shape) },
                               uniquingKeysWith: { first, _ in first })
        let missing = built.keys.filter { released[$0] == nil }.sorted()
        let unconsumed = released.keys.filter { built[$0] == nil }.sorted()
        let mismatched = built.compactMap { key, shape in
            released[key].flatMap { $0 == shape ? nil : "\(key) \(shape) vs \($0)" }
        }.sorted()
        print("VALIDATION structure \(name): \(built.count) built, \(released.count) released, \(missing.count) "
              + "missing, \(unconsumed.count) unconsumed, \(mismatched.count) mismatched")
        XCTAssertTrue(missing.isEmpty, "\(name) lacks \(missing.prefix(5))", file: file, line: line)
        XCTAssertTrue(unconsumed.isEmpty, "\(name) ships \(unconsumed.prefix(5)) nothing reads", file: file, line: line)
        XCTAssertTrue(mismatched.isEmpty, "\(name): \(mismatched.prefix(5))", file: file, line: line)
    }

    /// A shape as a loader transposes the tensor: `[out, in, t, h, w]` to `[out, t, h, w, in]` and
    /// `[out, in, h, w]` to `[out, h, w, in]`.
    private func channelsLast(_ shape: [Int]) -> [Int] {
        shape.count >= 4 ? [shape[0]] + shape.dropFirst(2) + [shape[1]] : shape
    }

    func testWanReleasesMatchTheirLoaders() throws {
        try requireMLXRuntime()
        for name in ["wan21-t2v-1.3b-transformer", "wan22-ti2v-5b-transformer"] {
            let release = try shapes(name)
            let net = NFKMLXWanTransformerNet(try NFKMLXWanRelease.transformerConfiguration(fromHuggingFace: release.config))
            assertStructure(name, module: net, released: release.shapes.mapValues { $0.count == 5 ? channelsLast($0) : $0 })
        }
        for name in ["wan21-t2v-1.3b-vae", "wan22-ti2v-5b-vae"] {
            let release = try shapes(name)
            let vae = NFKMLXWanVideoVAENet(try NFKMLXWanRelease.vaeConfiguration(fromHuggingFace: release.config).configuration)
            let adapted = release.shapes.map { key, shape -> (String, [Int]) in
                if key.hasSuffix(".gamma") { return (key, [shape.reduce(1, *)]) }
                return (key, channelsLast(shape))
            }
            assertStructure(name, module: vae, released: Dictionary(adapted, uniquingKeysWith: { first, _ in first }))
        }
        let text = try shapes("wan21-t2v-1.3b-text-encoder")
        let configuration = try NFKMLXT5Encoder.configuration(fromHuggingFace: text.config)
        XCTAssertTrue(configuration.perLayerBias, "umT5 gives every layer its own position bias")
        assertStructure("wan21-t2v-1.3b-text-encoder", module: NFKMLXT5Encoder.makeNet(configuration),
                        released: text.shapes)
    }

    func testLTXVideoReleaseMatchesItsLoaders() throws {
        try requireMLXRuntime()
        let transformer = try shapes("ltx-video-transformer")
        assertStructure("ltx-video-transformer", module: NFKMLXLTXTransformer.makeNet(
            try NFKMLXLTXVideoGenerator.transformerConfiguration(fromHuggingFace: transformer.config)),
            released: transformer.shapes)
        let vae = try shapes("ltx-video-vae")
        let (configuration, scalingFactor) = try NFKMLXLTXVideoGenerator.vaeConfiguration(fromHuggingFace: vae.config)
        XCTAssertEqual(scalingFactor, 1)
        assertStructure("ltx-video-vae", module: NFKMLXLTXVideoVAE.makeNet(configuration),
                        released: vae.shapes.mapValues { $0.count == 5 ? self.channelsLast($0) : $0 })
    }
}
