//
//  NFKMLXWanExpertsTests.swift
//  InferKitMLXTests
//
//  Wan 2.2 A14B: the schedule handed from the high-noise expert to the low-noise one, the second guidance
//  scale, the image-to-video condition and its Lanczos resize, the block-packed and streamed loads, and,
//  where the validation store holds the test-dev releases and their records, the whole generator against
//  diffusers' own `WanPipeline` and `WanImageToVideoPipeline` (`run_reference.py wan22_a14b_pipeline`).
//  Runs where MLX has a Metal library (see Tools/mlx-metallib.sh).
//

import CoreGraphics
import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXWanExpertsTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).flattened(), y = b.asType(.float32).flattened()
        let dot = Double((x * y).sum().item(Float.self))
        return dot / (Double(sqrt((x * x).sum()).item(Float.self)) * Double(sqrt((y * y).sum()).item(Float.self)))
    }

    // MARK: Weight-free

    func testTheScheduleHandsItsLaterStepsToTheLowNoiseExpert() throws {
        try requireMLXRuntime()
        let schedule = NFKMLXUniPCConfiguration(flowShift: 3, solverOrder: 2, trainTimesteps: 1000)
        for (ratio, high) in [(Float(0.875), 2), (0.9, 2), (0.5, 5)] {
            let denoiser = NFKWanDenoiser(schedule: schedule, steps: 6, text: MLXArray.zeros([1, 1]), negative: nil,
                                          guidance: 1, boundary: ratio * 1000, noise: MLXArray.zeros([1, 1, 1, 1]))
            XCTAssertEqual(denoiser.timesteps, [999, 937, 857, 750, 600, 376])
            XCTAssertEqual(denoiser.timesteps.indices.filter { !denoiser.usesLowNoiseExpert($0) }, Array(0 ..< high),
                           "boundary \(ratio)")
        }
    }

    func testTheImageMaskMarksTheGivenFrames() throws {
        try requireMLXRuntime()
        // diffusers: frame 0 repeated four times, then one channel per pixel frame, viewed [F', 4].
        let first = NFKMLXWanPipeline.imageMask(latentFrames: 3, height: 1, width: 1, frames: 9, marked: [0])
        XCTAssertEqual(first[0..., 0..., 0, 0].asArray(Float.self), [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        let both = NFKMLXWanPipeline.imageMask(latentFrames: 3, height: 1, width: 1, frames: 9, marked: [0, 8])
        XCTAssertEqual(both[0..., 0..., 0, 0].asArray(Float.self), [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1])
    }

    func testAWeightedLanczosKeepsAConstantImage() throws {
        let constant = [UInt8](repeating: 77, count: 40 * 56 * 3)
        let resized = NFKMLXPILResample.resampled(constant, width: 56, height: 40, toWidth: 48, toHeight: 32,
                                                  filter: .lanczos)
        XCTAssertEqual(Set(resized), [77])
    }

    // MARK: Against diffusers

    private func picture(_ pixels: MLXArray) throws -> CGImage {
        let height = pixels.dim(0), width = pixels.dim(1)
        let rgb = pixels.asType(.uint8).asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0 ..< width * height {
            rgba[pixel * 4] = rgb[pixel * 3]
            rgba[pixel * 4 + 1] = rgb[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = rgb[pixel * 3 + 2]
        }
        let provider = try XCTUnwrap(CGDataProvider(data: Data(rgba) as CFData))
        return try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                     bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                     bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                     provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }

    private func release(_ key: String, record recordKey: String, mode: String) throws -> (URL, [String: MLXArray]) {
        guard let directory = NFKMLXValidationConfig.environment[key] else {
            throw XCTSkip("set \(key) (Tools/wan22-a14b-testdev/make_testdev.py)")
        }
        guard let record = NFKMLXValidationConfig.environment[recordKey] else {
            throw XCTSkip("set \(recordKey) (run_reference.py \(mode) --checkpoint <\(key)>)")
        }
        return (URL(fileURLWithPath: directory), try MLX.loadArrays(url: URL(fileURLWithPath: record)))
    }

    private let prompt = "  A red fox   walking through fresh snow, cinematic "
    private let negative = "worst quality, blurry"

    /// The generator's run over the recorded inputs, its final latent and clip against the record's.
    private func assertMatchesTheReference(_ generator: NFKMLXWanVideoGenerator, _ record: [String: MLXArray],
                                           prefix: String = "", image: CGImage? = nil,
                                           lastImage: CGImage? = nil) throws -> (latent: Double, clip: Double) {
        generator.steps = 6
        generator.guidance = 4
        generator.lowNoiseGuidance = 3
        let run = try generator.generate(prompt: prompt, image: image, lastImage: lastImage, negativePrompt: negative,
                                         frames: 9, width: 48, height: 32, seed: 0, noise: record["latents"]!)
        let latent = cosine(run.latent, record[prefix + "final_latents"]!)
        let clip = cosine(run.clip, record[prefix + "video"]!)
        XCTAssertGreaterThan(latent, 0.9999999, "\(prefix)final latents")
        XCTAssertGreaterThan(clip, 0.9999999, "\(prefix)clip")
        return (latent, clip)
    }

    func testTextToVideoExpertsMatchWanPipeline() throws {
        try requireMLXRuntime()
        let (directory, record) = try release("IK_VAL_WAN22_T2V_A14B_TESTDEV", record: "IK_PARITY_WAN22_T2V_A14B_PIPELINE",
                                              mode: "wan22_a14b_pipeline")
        XCTAssertEqual(record["experts"]!.asArray(Int32.self), [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                       "the reference runs two guided steps on each expert's side of the boundary as the test assumes")
        let generator = try NFKMLXWanVideoGenerator.generator(directoryURL: directory, residency: .resident)
        XCTAssertTrue(generator.hasExperts)
        XCTAssertFalse(generator.isImageToVideo)
        XCTAssertThrowsError(try generator.video(forPrompt: prompt, image: try picture(MLXArray.zeros([32, 48, 3])),
                                                 frames: 9, width: 48, height: 32))
        let measured = try assertMatchesTheReference(generator, record)
        print("VALIDATION PARITY wan22-t2v-a14b-pipeline: final latents \(measured.latent), clip \(measured.clip)")
    }

    /// The denoising loop alone, from the recorded prompt features: the two experts, the handover, and both
    /// guidance scales, without umT5.
    private func denoisedFromRecordedFeatures(_ directory: URL, _ record: [String: MLXArray],
                                              condition: MLXArray?) throws -> MLXArray {
        let experts = try ["transformer", "transformer_2"].map { folder -> NFKMLXWanTransformerNet in
            let url = directory.appendingPathComponent(folder)
            return try NFKMLXWanVideoGenerator.loadTransformer(
                try NFKMLXWanRelease.transformerConfiguration(fromHuggingFace: url.appendingPathComponent("config.json")),
                directory: url, transformerType: nil, precision: .float32, bits: nil, groupSize: 64, streamed: [],
                adapters: [])
        }
        let schedule = try NFKMLXWanRelease.schedule(
            fromHuggingFace: directory.appendingPathComponent("scheduler/scheduler_config.json"))
        let ratio = try XCTUnwrap(try NFKMLXWanRelease.boundaryRatio(ofDirectory: directory))
        var denoiser = NFKWanDenoiser(schedule: schedule, steps: 6, text: record["prompt_embeds"]!,
                                      negative: record["negative_embeds"]!, guidance: 4, lowNoiseGuidance: 3,
                                      boundary: ratio * 1000, condition: condition, noise: record["latents"]!)
        for index in denoiser.timesteps.indices {
            denoiser.step(index, with: experts[denoiser.usesLowNoiseExpert(index) ? 1 : 0])
        }
        return denoiser.latent
    }

    func testTheLoopMatchesFromTheRecordedFeatures() throws {
        try requireMLXRuntime()
        let (directory, record) = try release("IK_VAL_WAN22_T2V_A14B_TESTDEV", record: "IK_PARITY_WAN22_T2V_A14B_PIPELINE",
                                              mode: "wan22_a14b_pipeline")
        let latent = cosine(try denoisedFromRecordedFeatures(directory, record, condition: nil), record["final_latents"]!)
        let generator = try NFKMLXWanVideoGenerator.generator(directoryURL: directory, residency: .resident)
        let features = try generator.usingText { stage in stage.features(generator.prompt(prompt)) }
        let featureCosine = cosine(features, record["prompt_embeds"]!)
        print("VALIDATION PARITY wan22-t2v-a14b-loop: final latents from the recorded features \(latent), "
              + "features \(featureCosine)")
        XCTAssertGreaterThan(latent, 0.9999999)
        XCTAssertGreaterThan(featureCosine, 0.9999999)
    }

    func testImageToVideoExpertsMatchWanImageToVideoPipeline() throws {
        try requireMLXRuntime()
        let (directory, record) = try release("IK_VAL_WAN22_I2V_A14B_TESTDEV", record: "IK_PARITY_WAN22_I2V_A14B_PIPELINE",
                                              mode: "wan22_a14b_pipeline")
        let first = try picture(record["picture"]!), last = try picture(record["last_picture"]!)
        let prepared = try NFKMLXWanVideoGenerator.preparedImage(first, width: 48, height: 32)
        let resizeGap = abs(prepared - record["image"]!).max().item(Float.self)
        XCTAssertLessThan(resizeGap, 1e-6, "PIL's Lanczos resize, byte for byte")

        let generator = try NFKMLXWanVideoGenerator.generator(directoryURL: directory, residency: .resident)
        XCTAssertTrue(generator.isImageToVideo)
        XCTAssertThrowsError(try generator.video(forPrompt: prompt, frames: 9, width: 48, height: 32))
        let condition = try generator.usingPipeline { pipeline in
            pipeline.imageCondition(prepared, lastImage: nil, frames: 9)
        }
        let conditionCosine = cosine(condition, record["condition"]!)
        XCTAssertGreaterThan(conditionCosine, 0.999999)
        let firstOnly = try assertMatchesTheReference(generator, record, image: first)
        let both = try assertMatchesTheReference(generator, record, prefix: "last.", image: first, lastImage: last)
        print("VALIDATION PARITY wan22-i2v-a14b-pipeline: resize gap \(resizeGap), condition \(conditionCosine), "
              + "final latents \(firstOnly.latent) / with a last frame \(both.latent), "
              + "clip \(firstOnly.clip) / \(both.clip)")
    }

    func testStreamedAndPackedExpertsHoldTheirOutputs() throws {
        try requireMLXRuntime()
        let (directory, record) = try release("IK_VAL_WAN22_T2V_A14B_TESTDEV", record: "IK_PARITY_WAN22_T2V_A14B_PIPELINE",
                                              mode: "wan22_a14b_pipeline")
        func run(_ generator: NFKMLXWanVideoGenerator) throws -> MLXArray {
            generator.steps = 6
            generator.guidance = 4
            generator.lowNoiseGuidance = 3
            return try generator.generate(prompt: prompt, image: nil, lastImage: nil, negativePrompt: negative,
                                          frames: 9, width: 48, height: 32, seed: 0, noise: record["latents"]!).latent
        }
        let held = try run(try NFKMLXWanVideoGenerator.generator(directoryURL: directory, residency: .resident))
        let streamed = try NFKMLXWanVideoGenerator.generator(directoryURL: directory, residency: .resident,
                                                             transformerType: nil, streamedBlocks: [1])
        XCTAssertEqual(streamed.streamedLayerCount, 2, "one block of each expert")
        XCTAssertEqual(abs(try run(streamed) - held).max().item(Float.self), 0, "a streamed block computes as a held one")
        let staged = try run(try NFKMLXWanVideoGenerator.generator(directoryURL: directory, residency: .staged))
        XCTAssertEqual(abs(staged - held).max().item(Float.self), 0, "experts loaded in turn compute as held ones")
        // Of the tiny blocks' layers only the feed-forward's output projection (96 wide) divides a group of 32.
        let packed = try NFKMLXWanVideoGenerator.generator(directoryURL: directory, residency: .resident,
                                                           transformerType: nil, bits: 8, groupSize: 32)
        XCTAssertEqual(packed.quantizationBits, 8)
        XCTAssertGreaterThan(cosine(try run(packed), held), 0.999)
    }
}
