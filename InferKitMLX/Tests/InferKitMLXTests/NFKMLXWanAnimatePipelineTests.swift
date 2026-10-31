//
//  NFKMLXWanAnimatePipelineTests.swift
//  InferKitMLXTests
//
//  The Wan 2.2 Animate 2 pipeline: the segment geometry, letterbox, resampling, and mask arithmetic
//  against values the reference's own functions give; the guided passes; and, where the validation store
//  holds them, the whole pipeline against diffusers' modular `WanAnimate2Blocks` over the test-dev
//  releases (`Tools/wan-animate-testdev/make_testdev.py`), the streamed and packed loads, the backend, and
//  the released inventories. Runs where MLX has a Metal library (see Tools/mlx-metallib.sh).
//

import CoreGraphics
import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXWanAnimatePipelineTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init)
        var dot = 0.0, xx = 0.0, yy = 0.0
        for (p, q) in zip(x, y) {
            dot += p * q
            xx += p * p
            yy += q * q
        }
        return dot / (xx.squareRoot() * yy.squareRoot())
    }

    private func maximumGap(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    // MARK: Geometry, against the reference's own arithmetic

    func testTheSegmentGeometryIsTheReferences() throws {
        // (real frames, segment length, carried) → (padding, segments), from the reference's video preprocess.
        let cases: [((Int, Int, Int), (Int, Int))] = [
            ((13, 9, 1), (4, 2)), ((17, 9, 1), (0, 2)), ((81, 81, 1), (0, 1)), ((100, 81, 1), (61, 2)),
            ((77, 81, 5), (4, 1)), ((5, 9, 4), (4, 1)),
        ]
        for ((real, length, carried), (padding, count)) in cases {
            let segments = try NFKWanAnimateSegments(realFrames: real, segmentLength: length, carriedFrames: carried)
            XCTAssertEqual(segments.paddingFrames, padding, "\(real), \(length), \(carried)")
            XCTAssertEqual(segments.count, count, "\(real), \(length), \(carried)")
            XCTAssertEqual(segments.paddedIndices.count, carried + count * segments.advance)
        }
        let segments = try NFKWanAnimateSegments(realFrames: 13, segmentLength: 9, carriedFrames: 1)
        XCTAssertEqual(segments.paddedIndices, Array(0 ..< 13) + [12, 11, 10, 9], "the tail mirrors")
        XCTAssertThrowsError(try NFKWanAnimateSegments(realFrames: 1, segmentLength: 9, carriedFrames: 1),
                             "no frame past the carried one is no segment")
        XCTAssertThrowsError(try NFKWanAnimateSegments(realFrames: 2, segmentLength: 9, carriedFrames: 1),
                             "a clip shorter than its own mirror is refused")
        XCTAssertThrowsError(try NFKWanAnimateSegments(realFrames: 20, segmentLength: 4, carriedFrames: 4))
    }

    func testTheFrameSizeAndCropFollowTheRoundedAspect() {
        // (image width, height, target width, height) → ((height, width), crop (top, left, height, width)).
        let cases: [((Int, Int, Int, Int), (Int, Int), (Int, Int, Int, Int))] = [
            ((50, 70, 48, 64), (64, 48), (0, 0, 64, 48)), ((1920, 1080, 640, 800), (528, 944), (0, 0, 527, 944)),
            ((720, 1280, 640, 800), (944, 528), (3, 0, 938, 528)), ((1000, 1000, 640, 800), (704, 704), (0, 0, 704, 704)),
            ((37, 91, 640, 800), (1120, 448), (0, 0, 1120, 448)),
        ]
        for ((iw, ih, w, h), size, crop) in cases {
            let frame = NFKWanAnimatePreprocess.frameSize(imageWidth: iw, imageHeight: ih, width: w, height: h)
            XCTAssertEqual(frame.height, size.0, "\(iw)×\(ih)")
            XCTAssertEqual(frame.width, size.1, "\(iw)×\(ih)")
            let region = NFKWanAnimatePreprocess.cropRegion(imageWidth: iw, imageHeight: ih, frameHeight: frame.height,
                                                            frameWidth: frame.width)
            XCTAssertEqual([region.top, region.left, region.height, region.width], [crop.0, crop.1, crop.2, crop.3],
                           "\(iw)×\(ih)")
        }
    }

    func testFrameIndicesTakeTheNearestFrameTiesToEven() {
        XCTAssertEqual(NFKWanAnimatePreprocess.frameIndices(count: 17, sourceRate: 30, targetRate: 24),
                       [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15])
        XCTAssertEqual(NFKWanAnimatePreprocess.frameIndices(count: 10, sourceRate: 24, targetRate: 24), Array(0 ..< 10))
        XCTAssertEqual(NFKWanAnimatePreprocess.frameIndices(count: 7, sourceRate: 15, targetRate: 24),
                       [0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6], "upsampling repeats frames")
        XCTAssertEqual(NFKWanAnimatePreprocess.frameIndices(count: 100, sourceRate: 29.97, targetRate: 24).count, 80)
    }

    func testTheConditionMaskMarksTheCarriedFrames() throws {
        try requireMLXRuntime()
        func mask(_ frames: Int) -> [[Float]] {
            let values = NFKMLXWanAnimatePipeline.conditionMask(latentFrames: 3, height: 1, width: 1, frames: frames)
            return (0 ..< 4).map { channel in values[channel].reshaped([-1]).asArray(Float.self) }
        }
        XCTAssertEqual(mask(5), Array(repeating: [1, 1, 0], count: 4), "pixel frames 0–4 fill two latent frames")
        XCTAssertEqual(mask(1), Array(repeating: [1, 0, 0], count: 4))
        XCTAssertEqual(mask(0), Array(repeating: [0, 0, 0], count: 4))
    }

    func testTheScheduleReaderTakesBothReleasedSamplers() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let scheduler = directory.appendingPathComponent("scheduler")
        try FileManager.default.createDirectory(at: scheduler, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = scheduler.appendingPathComponent("scheduler_config.json")
        try Data("""
            {"_class_name": "DPMSolverMultistepScheduler", "algorithm_type": "dpmsolver++", "flow_shift": 5.0,
             "lambda_min_clipped": -Infinity, "prediction_type": "flow_prediction", "solver_order": 2,
             "solver_type": "midpoint", "use_flow_sigmas": true, "final_sigmas_type": "zero"}
            """.utf8).write(to: url)
        guard case .dpmSolver(let solver) = try NFKMLXWanAnimateSchedule.reading(url) else {
            return XCTFail("the standard release samples with DPM-Solver++")
        }
        XCTAssertEqual(solver.flowShift, 5)
        XCTAssertEqual(try NFKMLXWanAnimateGenerator.variant(ofDirectory: directory), .standard)
        try Data("""
            {"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 5.0, "shift_terminal": null,
             "use_dynamic_shifting": false, "num_train_timesteps": 1000}
            """.utf8).write(to: url)
        guard case .flowMatchEuler(let euler) = try NFKMLXWanAnimateSchedule.reading(url) else {
            return XCTFail("the distilled release samples with Euler")
        }
        XCTAssertEqual(euler.baseShift, 5)
        XCTAssertEqual(try NFKMLXWanAnimateGenerator.variant(ofDirectory: directory), .distilled)
        try Data(#"{"_class_name": "UniPCMultistepScheduler"}"#.utf8).write(to: url)
        XCTAssertThrowsError(try NFKMLXWanAnimateSchedule.reading(url))
    }

    // MARK: The guided passes

    private func smallNet(layers: Int) -> NFKMLXWanAnimateNet {
        MLXRandom.seed(41)
        let net = NFKMLXWanAnimateNet(NFKMLXWanAnimateConfiguration(
            inChannels: 8, outChannels: 4, dim: 32, heads: 2, layers: layers, ffnDim: 48, textDim: 10,
            textLength: 16, patchSize: [1, 2, 2]))
        net.update(parameters: net.parameters().mapValues { MLXRandom.normal($0.shape) * 0.1 })
        return net
    }

    private func generationPasses(_ net: NFKMLXWanAnimateNet, passes: [(MLXArray, Bool)]) throws -> [MLXArray] {
        MLXRandom.seed(42)
        let cache = NFKMLXWanAnimateKVCache(layerCount: net.config.layers)
        let image = MLXRandom.normal([5, 1280])
        _ = try net.extractReference(latent: MLXRandom.normal([4, 2, 8, 8]), condition: MLXRandom.normal([4, 2, 8, 8]),
                                     text: MLXRandom.normal([7, 10]), imageEmbeddings: image, into: cache)
        let latent = MLXRandom.normal([4, 3, 8, 8]), condition = MLXRandom.normal([4, 3, 8, 8])
        return try net.generate(latent: latent, condition: condition, passes: passes.map { ($0.0, $0.1) },
                                imageEmbeddings: image, timestep: MLXArray([Float(0.4)]), cache: cache,
                                referenceGrid: (2, 4, 4), videoFrames: 2, videoArea: 16)
    }

    func testBothGuidedPassesRunOneBlockAtATimeAsEachRunsAlone() throws {
        try requireMLXRuntime()
        let net = smallNet(layers: 11)
        MLXRandom.seed(43)
        let text = MLXRandom.normal([7, 10]), negative = MLXRandom.normal([6, 10])
        let together = try generationPasses(net, passes: [(text, false), (negative, true)])
        let conditioned = try generationPasses(net, passes: [(text, false)])[0]
        let unconditioned = try generationPasses(net, passes: [(negative, true)])[0]
        XCTAssertEqual(maximumGap(together[0], conditioned), 0)
        XCTAssertEqual(maximumGap(together[1], unconditioned), 0)
        let sameTextSkipping = try generationPasses(net, passes: [(text, true)])[0]
        XCTAssertGreaterThan(maximumGap(sameTextSkipping, conditioned), 1e-4,
                             "the unconditional pass skips block 9")
    }

    func testAStackWithoutBlockNineSkipsNothing() throws {
        try requireMLXRuntime()
        let net = smallNet(layers: 3)
        MLXRandom.seed(44)
        let text = MLXRandom.normal([7, 10])
        let outputs = try generationPasses(net, passes: [(text, false), (text, true)])
        XCTAssertEqual(maximumGap(outputs[0], outputs[1]), 0)
    }

    // MARK: Against diffusers' WanAnimate2Blocks over the test-dev releases

    private func devRelease(_ key: String) throws -> URL {
        guard let path = NFKMLXValidationConfig.environment[key] else {
            throw XCTSkip("set \(key) (Tools/wan-animate-testdev/make_testdev.py)")
        }
        return URL(fileURLWithPath: path)
    }

    private func record(_ key: String) throws -> [String: MLXArray] {
        guard let path = NFKMLXValidationConfig.environment[key] else {
            throw XCTSkip("set \(key) (run_reference.py wan_animate_pipeline)")
        }
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    /// The record's reference image and driving frames.
    private func inputs(_ arrays: [String: MLXArray]) throws -> (NFKWanAnimateRGB, [NFKWanAnimateRGB]) {
        let reference = try XCTUnwrap(arrays["reference_rgb"])                // [70, 50, 3] uint8
        let driving = try XCTUnwrap(arrays["driving_rgb"])                    // [17, 30, 40, 3] uint8
        let image = NFKWanAnimateRGB(pixels: reference.asArray(UInt8.self), width: reference.dim(1),
                                     height: reference.dim(0))
        let frames = (0 ..< driving.dim(0)).map { index in
            NFKWanAnimateRGB(pixels: driving[index].asArray(UInt8.self), width: driving.dim(2), height: driving.dim(1))
        }
        return (image, frames)
    }

    static let prompt = "人物外观描述：一只银灰色的小猫，穿着深蓝色的制服。背景描述：背景为纯白色。"
    static let negativePrompt = "色调艳丽，过曝，静态，细节模糊不清"

    /// The test-dev generator at the record's settings, at float32, its noise the record's.
    private func generator(_ directory: URL, _ arrays: [String: MLXArray],
                           residency: NFKMLXResidency = .resident, quantizationBits: Int = 0,
                           budget: Int? = nil, streamedBlocks: Set<Int>? = nil) throws -> NFKMLXWanAnimateGenerator {
        let generator = try budget.map {
            try NFKMLXWanAnimateGenerator.generator(directoryURL: directory, residency: residency, precision: .float32,
                                                    quantizationBits: quantizationBits, budget: $0, groupSize: 32,
                                                    streamedBlocks: streamedBlocks)
        } ?? NFKMLXWanAnimateGenerator.generator(directoryURL: directory, residency: residency, precision: .float32,
                                                 quantizationBits: quantizationBits)
        generator.width = 48
        generator.height = 64
        generator.segmentLength = 9
        generator.carriedFrames = 1
        generator.steps = 3
        generator.noiseSource = { segment, shape in
            let noise = arrays["noise_\(segment)"]!
            precondition(noise.shape == shape, "the record's noise is \(noise.shape), the segment's \(shape)")
            return noise
        }
        return generator
    }

    private func assertPipelineMatchesTheReference(testDevKey: String, recordKey: String, label: String) throws {
        try requireMLXRuntime()
        let directory = try devRelease(testDevKey)
        let arrays = try record(recordKey)
        let distilled = try XCTUnwrap(arrays["distilled"]).item(Int32.self) == 1
        let generator = try generator(directory, arrays)
        XCTAssertEqual(generator.variant, distilled ? .distilled : .standard)
        XCTAssertEqual(generator.guidance, distilled ? 1 : 3, "the release's guidance")
        let (reference, driving) = try inputs(arrays)

        // The tokenizer reads the prompts as the reference's T5TokenizerFast does.
        let ids = { (text: String) in generator.prompt(text).ids.map(Int32.init) }
        XCTAssertEqual(ids(Self.prompt), try XCTUnwrap(arrays["prompt_ids"]).asArray(Int32.self))
        XCTAssertEqual(ids(NFKMLXWanAnimateGenerator.referencePrompt),
                       try XCTUnwrap(arrays["reference_prompt_ids"]).asArray(Int32.self))
        if !distilled {
            XCTAssertEqual(ids(Self.negativePrompt), try XCTUnwrap(arrays["negative_ids"]).asArray(Int32.self))
        }

        // The letterboxed inputs are PIL's to the bit, so their images agree to float32 rounding.
        let frame = NFKWanAnimatePreprocess.frameSize(imageWidth: reference.width, imageHeight: reference.height,
                                                      width: 48, height: 64)
        XCTAssertEqual([frame.height, frame.width], try XCTUnwrap(arrays["frame_size"]).asArray(Int32.self).map(Int.init))
        let crop = NFKWanAnimatePreprocess.cropRegion(imageWidth: reference.width, imageHeight: reference.height,
                                                      frameHeight: frame.height, frameWidth: frame.width)
        XCTAssertEqual([crop.top, crop.left, crop.height, crop.width],
                       try XCTUnwrap(arrays["crop_region"]).asArray(Int32.self).map(Int.init))
        let image = NFKWanAnimatePreprocess.letterboxed(reference.pixels, width: reference.width, height: reference.height,
                                                        toWidth: 48, toHeight: 64, filter: .bicubic)
        XCTAssertLessThan(maximumGap(image, try XCTUnwrap(arrays["image_pixels"])), 1e-6)
        let kept = NFKWanAnimatePreprocess.frameIndices(count: driving.count, sourceRate: 30, targetRate: 24)
        XCTAssertEqual(kept.count, Int(try XCTUnwrap(arrays["real_frames"]).item(Int32.self)))
        let segments = try NFKWanAnimateSegments(realFrames: kept.count, segmentLength: 9, carriedFrames: 1)
        XCTAssertEqual(segments.count, Int(try XCTUnwrap(arrays["segments"]).item(Int32.self)))
        let drivingPixels = stacked(segments.paddedIndices.map { index -> MLXArray in
            let source = driving[kept[index]]
            return NFKWanAnimatePreprocess.letterboxed(source.pixels, width: source.width, height: source.height,
                                                       toWidth: 48, toHeight: 64, filter: .bilinear)
        }, axis: 0)
        XCTAssertLessThan(maximumGap(drivingPixels, try XCTUnwrap(arrays["driving_pixels"])), 1e-6)

        // The encoders and the reference image's latents.
        let (features, imageFeatures, referenceLatent) = try generator.usingStages { encoders, pipeline in
            ([encoders.text.features(generator.prompt(Self.prompt)),
              encoders.text.features(generator.prompt(NFKMLXWanAnimateGenerator.referencePrompt))]
             + (distilled ? [] : [encoders.text.features(generator.prompt(Self.negativePrompt))]),
             [encoders.image.features(image), encoders.image.features(drivingPixels[0])],
             pipeline.encode(image.expandedDimensions(axis: 0)))
        }
        let promptCosine = cosine(features[0], try XCTUnwrap(arrays["prompt_embeds"]))
        let referencePromptCosine = cosine(features[1], try XCTUnwrap(arrays["reference_prompt_embeds"]))
        let imageCosine = cosine(imageFeatures[0], try XCTUnwrap(arrays["image_embeds"]))
        let drivingImageCosine = cosine(imageFeatures[1], try XCTUnwrap(arrays["driving_image_embeds"]))
        let referenceLatentCosine = cosine(referenceLatent, try XCTUnwrap(arrays["reference_image_latents"])[4...])
        XCTAssertGreaterThan(promptCosine, 0.99999)
        XCTAssertGreaterThan(referencePromptCosine, 0.99999)
        if !distilled {
            XCTAssertGreaterThan(cosine(features[2], try XCTUnwrap(arrays["negative_prompt_embeds"])), 0.99999)
        }
        XCTAssertGreaterThan(imageCosine, 0.99999)
        XCTAssertGreaterThan(drivingImageCosine, 0.99999)
        XCTAssertGreaterThan(referenceLatentCosine, 0.99999)

        // The whole clip through the generator.
        let clip = try generator.video(reference: reference, driving: driving, drivingRate: 30, prompt: Self.prompt,
                                       negativePrompt: distilled ? nil : Self.negativePrompt, seed: 0)
        let expected = try XCTUnwrap(arrays["video"])
        XCTAssertEqual(clip.shape, expected.shape)
        let clipCosine = cosine(clip, expected)
        print("VALIDATION PARITY \(label): prompt features \(promptCosine), reference prompt \(referencePromptCosine), "
              + "reference image CLIP \(imageCosine), driving frame CLIP \(drivingImageCosine), reference latents "
              + "\(referenceLatentCosine), clip \(clipCosine) (max |difference| \(maximumGap(clip, expected)))")
        XCTAssertGreaterThan(clipCosine, 0.99999, "the segment loop matches the reference end to end")
    }

    func testThePipelineMatchesTheReference() throws {
        try assertPipelineMatchesTheReference(testDevKey: "IK_VAL_WAN_ANIMATE_TESTDEV",
                                              recordKey: "IK_PARITY_WAN_ANIMATE_PIPELINE", label: "wan-animate-pipeline")
    }

    func testTheDistilledPipelineMatchesTheReference() throws {
        try assertPipelineMatchesTheReference(testDevKey: "IK_VAL_WAN_ANIMATE_DISTILLED_TESTDEV",
                                              recordKey: "IK_PARITY_WAN_ANIMATE_DISTILLED_PIPELINE",
                                              label: "wan-animate-distilled-pipeline")
    }

    // The same clip whether the transformer's blocks are held, streamed, or packed.
    func testStreamedAndPackedTransformersRunTheSameClip() throws {
        try requireMLXRuntime()
        let directory = try devRelease("IK_VAL_WAN_ANIMATE_TESTDEV")
        let arrays = try record("IK_PARITY_WAN_ANIMATE_PIPELINE")
        let (reference, driving) = try inputs(arrays)
        func clip(_ generator: NFKMLXWanAnimateGenerator) throws -> MLXArray {
            try generator.video(reference: reference, driving: driving, drivingRate: 30, prompt: Self.prompt,
                                negativePrompt: Self.negativePrompt, seed: 0)
        }
        let held = try clip(generator(directory, arrays))

        let footprint = try NFKMLXWanAnimateRelease.transformerFootprint(
            directory: directory.appendingPathComponent("transformer"), precision: .float32)
        XCTAssertEqual(footprint.layerBytes.count, 10, "each block is a repeated layer")
        // The test-dev encoders outweigh its transformer, so no budget that holds them streams a block; the
        // streamed blocks are named instead.
        let streamedGenerator = try generator(directory, arrays, residency: .streamed, budget: 1 << 30,
                                              streamedBlocks: Set(3 ..< 10))
        XCTAssertEqual(streamedGenerator.streamedLayerCount, 7)
        XCTAssertEqual(streamedGenerator.streamedBytesPerPass, footprint.layerBytes[3...].reduce(0, +))
        let streamed = try clip(streamedGenerator)
        XCTAssertEqual(maximumGap(streamed, held), 0, "a streamed block computes what a held one does")

        let packedGenerator = try generator(directory, arrays, quantizationBits: 8, budget: 1 << 30)
        XCTAssertEqual(packedGenerator.quantizationBits, 8)
        let packed = try clip(packedGenerator)
        let packedCosine = cosine(packed, held)
        print("VALIDATION wan-animate packed: 8-bit transformer clip cosine \(packedCosine) against float32, "
              + "\(streamedGenerator.streamedLayerCount) blocks streamed bit-exact")
        XCTAssertGreaterThan(packedCosine, 0.99, "8-bit blocks stay close to the float32 clip")
    }

    // The backend reads a movie and writes one, one frame per resampled driving frame.
    func testTheBackendAnimatesAMovie() throws {
        try requireMLXRuntime()
        let directory = try devRelease("IK_VAL_WAN_ANIMATE_TESTDEV")
        let arrays = try record("IK_PARITY_WAN_ANIMATE_PIPELINE")
        let (reference, driving) = try inputs(arrays)
        func cgImage(_ rgb: NFKWanAnimateRGB) throws -> CGImage {
            let tensor = MLXArray(rgb.pixels.map { Float($0) / 255 }).reshaped([rgb.height, rgb.width, 3])
            return try NFKMLXImageBridge.cgImage(from: tensor, options: NFKMLXImageOptions())
        }
        let movie = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: movie) }
        try NFKMLXVideoFile.write(try driving.map(cgImage), framesPerSecond: 30, to: movie)
        let backend = try NFKMLXWanAnimateGenerator.backend(directoryURL: directory, residency: .resident,
                                                            precision: .float32, quantizationBits: 0)
        XCTAssertEqual(backend.backendIdentifier, NFKMLXWanAnimateGenerator.modelName)
        let request = NFKInferenceRequest(inputs: [
            NFKInputImage: try cgImage(reference),
            NFKInputVideo: NFKVideoAsset(fileURL: movie, durationSeconds: 17.0 / 30, framesPerSecond: 30,
                                         dimensions: CGSize(width: 40, height: 30)),
            NFKInputPrompt: Self.prompt,
        ], parameters: [
            NFKParameterWidth: 48, NFKParameterHeight: 64, NFKParameterSteps: 2,
            NFKMLXGenerationParameterKey.segmentLength: 9, NFKMLXGenerationParameterKey.carriedFrames: 1,
        ])
        let result = try backend.runInference(for: request)
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        let written = try NFKMLXVideoFile.read(try XCTUnwrap(asset.fileURL))
        XCTAssertEqual(written.frames.count, 13, "17 frames at 30 fps resample to 13 at 24")
        XCTAssertEqual(written.framesPerSecond, 24, accuracy: 0.01)
        XCTAssertEqual(written.frames[0].width, 48)
        XCTAssertEqual(written.frames[0].height, 64)
    }

    // MARK: The released inventories

    private func shapes(_ name: String) throws -> (config: URL, shapes: [String: [Int]]) {
        guard let root = NFKMLXValidationConfig.environment["IK_SHAPES_ROOT"] else {
            throw XCTSkip("set IK_SHAPES_ROOT")
        }
        let directory = URL(fileURLWithPath: root).appendingPathComponent(name)
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("shapes.json")),
              let shapes = try JSONSerialization.jsonObject(with: data) as? [String: [Int]] else {
            throw XCTSkip("no flat shapes.json for \(name) under \(root)")
        }
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

    private func channelsLast(_ shape: [Int]) -> [Int] {
        shape.count >= 4 ? [shape[0]] + shape.dropFirst(2) + [shape[1]] : shape
    }

    // The released 14B pipeline (about 50 GB) is held to its loaders by shape: every stage's configuration
    // builds the module the release was saved from.
    func testTheReleaseMatchesItsLoaders() throws {
        try requireMLXRuntime()
        let transformer = try shapes("wan-animate-2-14b-diffusers-transformer")
        let configuration = try NFKMLXWanAnimateRelease.transformerConfiguration(fromHuggingFace: transformer.config)
        XCTAssertEqual(configuration.layers, 40)
        XCTAssertEqual(configuration.heads, 40)
        assertStructure("wan-animate-2-14b-diffusers-transformer", module: NFKMLXWanAnimateNet(configuration),
                        released: transformer.shapes.mapValues { $0.count == 5 ? channelsLast($0) : $0 })
        let image = try shapes("wan-animate-2-14b-diffusers-image-encoder")
        assertStructure("wan-animate-2-14b-diffusers-image-encoder",
                        module: NFKMLXWanAnimateImageEncoderNet(try .configuration(fromHuggingFace: image.config)),
                        released: image.shapes.mapValues(channelsLast))
        let text = try shapes("wan-animate-2-14b-diffusers-text-encoder")
        assertStructure("wan-animate-2-14b-diffusers-text-encoder",
                        module: NFKMLXT5Encoder.makeNet(try NFKMLXT5Encoder.configuration(fromHuggingFace: text.config)),
                        released: text.shapes)
        let vae = try shapes("wan-animate-2-14b-diffusers-vae")
        let adapted = vae.shapes.map { key, shape -> (String, [Int]) in
            key.hasSuffix(".gamma") ? (key, [shape.reduce(1, *)]) : (key, channelsLast(shape))
        }
        assertStructure("wan-animate-2-14b-diffusers-vae",
                        module: NFKMLXWanVideoVAENet(try NFKMLXWanRelease.vaeConfiguration(fromHuggingFace: vae.config).configuration),
                        released: Dictionary(adapted, uniquingKeysWith: { first, _ in first }))
    }
}
