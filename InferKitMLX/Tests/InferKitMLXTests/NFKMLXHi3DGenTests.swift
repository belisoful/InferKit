//
//  NFKMLXHi3DGenTests.swift
//  InferKitMLXTests
//
//  Hi3DGen's preparation, its TRELLIS stages on the normal map, and the backend. The parity tests read the
//  release directory (IK_VAL_HI3DGEN), DINOv2 (IK_VAL_DINOV2_VITL14_REG), BiRefNet (IK_VAL_BIREFNET),
//  StableNormal-turbo (IK_VAL_YOSO_NORMAL), and the oracle record (IK_PARITY_HI3DGEN, from
//  `run_reference.py hi3dgen`), and skip without them. The normal stage and the mesh extraction have their own
//  test classes.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXHi3DGenTests: XCTestCase {

    // Each bar sits at about three times the gap the first parity run measured (hi3dgen-run1, float32 MLX
    // against the float32 CPU record): condition 0.99999999978, structure flow 0.99999999999324 (last),
    // structure decoder 0.99999999999967, latent flow 0.99999999999924 (last), mesh decoder features
    // 0.99999999998712, structure sampling 0.99999999999865, latent sampling 0.99999999999952 and
    // 0.99999999999948 denormalized.
    private let conditionBar = 0.9999999995
    private let structureFlowBar = 0.99999999998
    private let structureDecoderBar = 0.999999999999
    private let latentFlowBar = 0.999999999998
    private let meshFeaturesBar = 0.99999999996
    private let structureSamplingBar = 0.999999999996
    private let latentSamplingBar = 0.999999999998

    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func envPath(_ key: String) throws -> URL {
        guard let path = NFKMLXValidationConfig.environment[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key)")
        }
        return URL(fileURLWithPath: path)
    }

    private func checkpoint(_ name: String) throws -> URL {
        try envPath("IK_VAL_HI3DGEN").appendingPathComponent("ckpts/\(name).safetensors")
    }

    private func record() throws -> [String: MLXArray] {
        try MLX.loadArrays(url: try envPath("IK_PARITY_HI3DGEN"))
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).reshaped([-1]).asArray(Float.self), y = b.asType(.float32).reshaped([-1]).asArray(Float.self)
        XCTAssertEqual(x.count, y.count)
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(x.count, y.count) {
            dot += Double(x[i]) * Double(y[i]); na += Double(x[i]) * Double(x[i]); nb += Double(y[i]) * Double(y[i])
        }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
    }

    /// Writes a measured figure to standard error, unbuffered, so a later crash in the process keeps it.
    private func report(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    private func channelsLast(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 4, 1) }

    private func sparse(_ reference: [String: MLXArray], _ name: String, resolution: Int) -> (MLXArray, NFKTrellisVoxelGrid) {
        (reference[name]!, NFKTrellisVoxelGrid(reference[name + ".coords"]![0..., 1...], resolution: resolution))
    }

    private func mismatches(_ a: [UInt8], _ b: [UInt8]) -> Int {
        XCTAssertEqual(a.count, b.count)
        return zip(a, b).filter { $0 != $1 }.count
    }

    // MARK: - Preparation

    func testAnImageWithAlphaKeepsItAndCropsToItsBox() throws {
        let (width, height) = (40, 30)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for y in 10 ..< 20 {
            for x in 5 ..< 25 {
                rgba[(y * width + x) * 4] = 200
                rgba[(y * width + x) * 4 + 3] = 255
            }
        }
        let prepared = try NFKHi3DGenImagePreparation.prepare(rgba: rgba, width: width, height: height, side: 64) { _, _, _ in
            XCTFail("an image with alpha runs no mask")
            return []
        }
        XCTAssertEqual(prepared.width, 64)
        XCTAssertEqual(prepared.pixels[(32 * 64 + 32) * 4 + 3], 255)
        XCTAssertEqual(prepared.pixels[3], 0)
    }

    func testPreparationReproducesThePreparedImageFromTheRecordedMask() throws {
        let reference = try record()
        let input = reference["input"]!
        let (height, width) = (input.dim(0), input.dim(1))
        let rgb = input.asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0 ..< width * height {
            for c in 0 ..< 3 { rgba[pixel * 4 + c] = rgb[pixel * 3 + c] }
        }
        let mask = reference["birefnet_mask"]!.asArray(UInt8.self)
        let prepared = try NFKHi3DGenImagePreparation.prepare(rgba: rgba, width: width, height: height) { masked, w, h in
            XCTAssertEqual(masked, reference["birefnet_resized_input"]!.asArray(UInt8.self))
            XCTAssertEqual([w, h], [width, height])
            return mask
        }
        let differing = mismatches(prepared.pixels, reference["prepared_image"]!.asArray(UInt8.self))
        report("Hi3DGen prepared image mismatched bytes \(differing) of \(prepared.pixels.count)")
        XCTAssertEqual(differing, 0)
    }

    func testBiRefNetsMaskMatchesTheRecordedMask() throws {
        let reference = try record()
        let model = NFKMLXBiRefNetModel()
        try NFKMLXBiRefNet.loadWeights(into: model, from: try envPath("IK_VAL_BIREFNET"))
        let input = reference["input"]!
        let mask = NFKHi3DGenImagePreparation.birefnetMask(rgb: input.asArray(UInt8.self), width: input.dim(1), height: input.dim(0),
                                                           model: model)
        let differing = mismatches(mask, reference["birefnet_mask"]!.asArray(UInt8.self))
        report("Hi3DGen BiRefNet mask mismatched pixels \(differing) of \(mask.count)")
        XCTAssertEqual(differing, 0)
    }

    // MARK: - TRELLIS stages on the normal map

    func testTheConditionOfTheNormalMapMatchesReference() throws {
        let reference = try record()
        let encoder = try NFKMLXTrellis.imageEncoder(weightsURL: try envPath("IK_VAL_DINOV2_VITL14_REG"))
        let map = reference["normal_image"]!
        let side = 518
        let resized = NFKMLXPillowLanczos.resize(map.asArray(UInt8.self), width: map.dim(1), height: map.dim(0), outWidth: side,
                                                 outHeight: side)
        let tokens = encoder(MLXArray(resized.map { Float($0) / 255 }).reshaped([1, side, side, 3]))
        let similarity = cosine(tokens, reference["cond"]!)
        report("Hi3DGen condition cosine \(similarity)")
        XCTAssertGreaterThan(similarity, conditionBar)
    }

    func testStructureFlowMatchesReference() throws {
        let reference = try record()
        let flow = try NFKMLXTrellis.structureFlow(weightsURL: try checkpoint("ss_flow_normal_dit_L_16l8_fp16"))
        for call in ["call0", "call1", "call2", "call3", "last"] {
            let prefix = "ss_flow.\(call)."
            let velocity = flow(channelsLast(reference[prefix + "x"]!), t: reference[prefix + "t"]!, condition: reference[prefix + "cond"]!)
            let similarity = cosine(velocity, channelsLast(reference[prefix + "out"]!))
            report("Hi3DGen structure flow \(call) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, structureFlowBar, call)
        }
    }

    func testStructureDecoderMatchesReference() throws {
        let reference = try record()
        let decoder = try NFKMLXTrellis.structureDecoder(weightsURL: try checkpoint("ss_dec_conv3d_16l8_fp16"))
        let logits = decoder(channelsLast(reference["ss_latent"]!))
        let similarity = cosine(logits, channelsLast(reference["ss_logits"]!))
        let voxels = NFKMLXTrellisStructureDecoder.occupiedVoxels(logits[0])
        let expected = reference["coords"]![0..., 1...].asType(.int32)
        let same = voxels.shape == expected.shape && voxels.asArray(Int32.self) == expected.asArray(Int32.self)
        report("Hi3DGen structure decoder cosine \(similarity), occupied \(voxels.dim(0)) against \(expected.dim(0)), identical \(same)")
        XCTAssertGreaterThan(similarity, structureDecoderBar)
        XCTAssertTrue(same)
    }

    func testLatentFlowMatchesReference() throws {
        let reference = try record()
        let flow = try NFKMLXTrellis.latentFlow(weightsURL: try checkpoint("slat_flow_normal_dit_L_64l8p2_fp16"))
        for call in ["call0", "call1", "call2", "call3", "last"] {
            let prefix = "slat_flow.\(call)."
            let (x, voxels) = sparse(reference, prefix + "x", resolution: 64)
            let velocity = flow.velocity(x, grid: voxels, t: reference[prefix + "t"]!, condition: reference[prefix + "cond"]!)
            let similarity = cosine(velocity, reference[prefix + "out"]!)
            report("Hi3DGen latent flow \(call) cosine \(similarity)")
            XCTAssertGreaterThan(similarity, latentFlowBar, call)
        }
    }

    func testMeshDecoderFeaturesMatchReference() throws {
        let reference = try record()
        let decoder = try NFKMLXTrellis.meshDecoder(weightsURL: try checkpoint("slat_dec_mesh_swin8_B_64l8m256c_fp16"))
        let (latent, voxels) = sparse(reference, "slat", resolution: 64)
        let (features, fine) = decoder.features(latent, grid: voxels)
        XCTAssertEqual(fine.coordinates, reference["mesh_decoder_out.coords"]![0..., 1...].asType(.int32).reshaped([-1]).asArray(Int32.self))
        let similarity = cosine(features, reference["mesh_decoder_out"]!)
        report("Hi3DGen mesh decoder features cosine \(similarity), \(fine.coordinates.count / 3) voxels")
        XCTAssertGreaterThan(similarity, meshFeaturesBar)
    }

    private func pipeline() throws -> NFKMLXHi3DGenPipeline {
        try NFKMLXHi3DGenPipeline(directoryURL: try envPath("IK_VAL_HI3DGEN"), normalDirectoryURL: try envPath("IK_VAL_YOSO_NORMAL"),
                                  encoderWeightsURL: try envPath("IK_VAL_DINOV2_VITL14_REG"),
                                  birefnetWeightsURL: try envPath("IK_VAL_BIREFNET"), residency: .staged)
    }

    func testTheReleaseRunsTheAppsSamplerSettings() throws {
        let pipeline = try pipeline()
        XCTAssertEqual([pipeline.structureSampling.steps, pipeline.latentSampling.steps], [50, 6])
        XCTAssertEqual([pipeline.structureSampling.guidance, pipeline.latentSampling.guidance], [3, 3])
        XCTAssertEqual(pipeline.structureSampling.guidanceInterval, 0.5 ... 1)
        XCTAssertEqual(pipeline.structureSampling.rescale, 3)
        XCTAssertNil(pipeline.trellis.gaussianDecoder)
    }

    func testBothSamplersMatchReferenceFromTheRecordedNoise() throws {
        let reference = try record()
        let pipeline = try pipeline()
        let condition = reference["cond"]!
        let structure = try pipeline.trellis.sampleStructure(noise: channelsLast(reference["ss_noise"]!), condition: condition,
                                                             unconditional: MLXArray.zeros(like: condition),
                                                             sampling: pipeline.structureSampling)
        let structureCosine = cosine(structure, channelsLast(reference["ss_latent"]!))
        let voxels = NFKMLXTrellisStructureDecoder.occupiedVoxels(try pipeline.trellis.occupancy(structure)[0])
        let expectedVoxels = reference["coords"]![0..., 1...].asType(.int32)
        let latent = try pipeline.trellis.sampleLatent(noise: reference["slat_noise"]!, voxels: expectedVoxels, condition: condition,
                                                       unconditional: MLXArray.zeros(like: condition), sampling: pipeline.latentSampling)
        let latentCosine = cosine(latent, reference["slat_normalized"]!)
        let denormalizedCosine = cosine(pipeline.trellis.denormalized(latent), reference["slat"]!)
        report("Hi3DGen structure sampling cosine \(structureCosine), occupied \(voxels.dim(0)) against \(expectedVoxels.dim(0)); "
               + "latent sampling cosine \(latentCosine), denormalized \(denormalizedCosine)")
        XCTAssertGreaterThan(structureCosine, structureSamplingBar)
        XCTAssertGreaterThan(latentCosine, latentSamplingBar)
        XCTAssertGreaterThan(denormalizedCosine, latentSamplingBar)
        XCTAssertEqual(voxels.dim(0), expectedVoxels.dim(0))
    }

    // MARK: - Customization

    /// MLX's peak since the last reset and the process's lifetime peak footprint, flushed, and MLX's peak reset.
    private func peak(_ stage: String, since start: Date) -> Int {
        var info = rusage_info_v4()
        _ = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        let peak = NFKMLXGPU.peakMemory
        report(String(format: "MEMORY hi3dgen %@: %.1f s, MLX peak %.3f GB, lifetime peak footprint %.3f GB", stage,
                      Date().timeIntervalSince(start), Double(peak) / 1e9, Double(info.ri_lifetime_max_phys_footprint) / 1e9))
        NFKMLXGPU.resetPeakMemory()
        return peak
    }

    // One LoRA step of the default recipe on the released structure flow, from an image's condition through the
    // public method and the record's sampled structure latent, then merged, saved, and reloaded through the
    // pipeline. Loads the release: run it under LMC.
    func testAStructureFlowFineTunedOnAnImagesConditionReloadsThroughThePipeline() throws {
        let reference = try record()
        let input = reference["input"]!
        let (height, width) = (input.dim(0), input.dim(1))
        let rgb = input.asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0 ..< width * height {
            for c in 0 ..< 3 { rgba[pixel * 4 + c] = rgb[pixel * 3 + c] }
        }
        let condition = try pipeline().condition(rgba: rgba, width: width, height: height).condition
        eval(condition)
        NFKMLXGPU.clearCache()

        let flow = try NFKMLXTrellis.structureFlow(weightsURL: try checkpoint("ss_flow_normal_dit_L_16l8_fp16"))
        let example = NFKMLXTrellisStructureExample(condition: condition, latent: channelsLast(reference["ss_latent"]!))
        NFKMLXGPU.resetPeakMemory()
        let start = Date()
        let losses = try NFKMLXTrellisTraining.fineTuneStructureFlow(flow, examples: [example], steps: 1, seed: 1)
        let stepPeak = peak("structure flow LoRA step (loss \(losses[0]))", since: start)
        XCTAssertTrue(losses[0].isFinite)
        // Measured 7.793 GB (2026-11-21), as for TRELLIS; the bar leaves 1.87 GB of margin.
        XCTAssertLessThan(stepPeak, 9 << 30)
        XCTAssertGreaterThan(try NFKMLXLoRA.merge(into: flow), 0)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("hi3dgen-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(flow, to: url)

        let tuned = try NFKMLXHi3DGenPipeline(directoryURL: try envPath("IK_VAL_HI3DGEN"), normalDirectoryURL: try envPath("IK_VAL_YOSO_NORMAL"),
                                              encoderWeightsURL: try envPath("IK_VAL_DINOV2_VITL14_REG"),
                                              birefnetWeightsURL: try envPath("IK_VAL_BIREFNET"), residency: .staged,
                                              flows: NFKMLXTrellisFlowWeights(structure: url))
        let x = channelsLast(reference["ss_flow.call0.x"]!), t = reference["ss_flow.call0.t"]!
        let expected = flow(x, t: t, condition: condition)
        let reloaded = try tuned.trellis.structureFlow.use { $0(x, t: t, condition: condition) }
        let release = try NFKMLXTrellis.structureFlow(weightsURL: try checkpoint("ss_flow_normal_dit_L_16l8_fp16"))
        let moved = abs(expected - release(x, t: t, condition: condition)).max().item(Float.self)
        report("Hi3DGen structure flow after one LoRA step: the release's velocity moved by up to \(moved)")
        XCTAssertEqual(abs(expected - reloaded).max().item(Float.self), 0)
        XCTAssertGreaterThan(moved, 0)
    }

    // One LoRA step of the default recipe on the released latent flow, on the record's structured latent (24,672
    // voxels, 4,744 at the blocks' half resolution) and condition. Loads the release: run it under LMC.
    func testALoRAStepOnTheReleasedLatentFlowHoldsOneBlockAtATime() throws {
        let reference = try record()
        let example = NFKMLXTrellisLatentExample(condition: reference["cond"]!, voxels: reference["coords"]![0..., 1...].asType(.int32),
                                                 latent: reference["slat_normalized"]!)
        eval(example.condition, example.voxels, example.latent)
        let flow = try NFKMLXTrellis.latentFlow(weightsURL: try checkpoint("slat_flow_normal_dit_L_64l8p2_fp16"))
        eval(flow)
        _ = peak("latent flow build, before the step", since: Date())
        let start = Date()
        let losses = try NFKMLXTrellisTraining.fineTuneLatentFlow(flow, examples: [example], steps: 1, seed: 1)
        let stepPeak = peak("latent flow LoRA step on \(example.voxels.dim(0)) voxels (loss \(losses[0]))", since: start)
        XCTAssertTrue(losses[0].isFinite)
        // Measured 9.381 GB (2026-11-21); the bar leaves 2.43 GB of margin.
        XCTAssertLessThan(stepPeak, 11 << 30)
    }

    func testTheBackendGeneratesAMeshAndItsNormalMapFromItsFactory() throws {
        let reference = try record()
        let backend = try NFKMLXHi3DGenFactory.backend(directoryURL: try envPath("IK_VAL_HI3DGEN"),
                                                       normalDirectoryURL: try envPath("IK_VAL_YOSO_NORMAL"),
                                                       encoderWeightsURL: try envPath("IK_VAL_DINOV2_VITL14_REG"),
                                                       birefnetWeightsURL: try envPath("IK_VAL_BIREFNET"), residency: .staged)
        XCTAssertEqual(backend.supportedOutputKeys, [NFKOutputMesh, NFKOutputImage])
        let input = reference["input"]!
        let (height, width) = (input.dim(0), input.dim(1))
        let rgb = input.asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0 ..< width * height {
            for c in 0 ..< 3 { rgba[pixel * 4 + c] = rgb[pixel * 3 + c] }
        }
        let image = try NFKMLXImageBridge.cgImage(rgba: rgba, width: width, height: height, options: NFKMLXImageOptions())
        let request = NFKInferenceRequest(inputs: [NFKInputImage: image],
                                          parameters: [NFKMLXTrellisParameterKey.structureSteps: 4, NFKMLXTrellisParameterKey.latentSteps: 2,
                                                       NFKParameterSeed: 7])
        let result = try backend.runInference(for: request)
        let mesh = try XCTUnwrap(result.mesh)
        let normalMap = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        report("Hi3DGen backend at 4 and 2 steps: \(mesh.vertexCount) vertices, \(mesh.faceCount) faces")
        XCTAssertGreaterThan(mesh.faceCount, 1000)
        XCTAssertEqual(try NFKMLXImageBridge.cgImage(fromValue: normalMap).width, 1024)
    }
}
