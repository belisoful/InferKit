//
//  MLXCustomizationExamples.swift
//  InferKitMLXExamples
//
//  The customization snippets from Docs/examples.md, compiled against the package's PUBLIC surface.
//
//  This file imports InferKitMLX without `@testable` deliberately. A fine-tuning recipe a consumer
//  cannot call is not shipped, and the whole path here — build a network, train it, write a
//  checkpoint, reload it through the model's own factory — has to hold together for an app that
//  links the package like any other dependency. Compiling it that way is what keeps a customization
//  API from drifting back behind `internal`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXOptimizers
import InferKitMLX

final class MLXCustomizationExamples: XCTestCase {

    // Docs/examples.md: Customizing a model on a consumer's own data
    func testExampleFineTuningRoundTripsThroughTheShippedFactory() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
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
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
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
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
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
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
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

    // MARK: The public surface these recipes rest on

    /// The generic trainer entry, an optimizer chosen by the caller, and both ends of the checkpoint
    /// API, driven the way an app drives them. `Docs/examples.md` shows this shape in the LoRA
    /// section, where the caller supplies the optimizer and the loss.
    func testTheTrainerAndTheCheckpointAPIAreCallableFromOutsideThePackage() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("trainer-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tuned) }

        let net = NFKMLXZeroDCENet(filters: 4)
        let photos = Self.darkPhotos()
        let objective = NFKMLXZeroDCEObjective()
        let history = try NFKMLXTrainer.train(net, optimizer: AdamW(learningRate: 1e-4), steps: 3,
                                              sample: { _ in photos },
                                              loss: objective.callAsFunction,
                                              clipGradientNorm: 0.1)
        XCTAssertEqual(history.count, 3)

        try NFKMLXWeights.save(net, to: tuned)
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: tuned)
        XCTAssertFalse(checkpoint.needsConvTranspose,
                       "a checkpoint written by save is already in the module's own layout")
    }

    /// A call that documents what it throws has to let a caller act on it, so the error type is part
    /// of the customization surface.
    func testAFullyFrozenModelReportsThatNothingWouldTrain() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let net = NFKMLXZeroDCENet(filters: 4)
        net.freeze()
        let photos = Self.darkPhotos()
        let objective = NFKMLXZeroDCEObjective()

        XCTAssertThrowsError(try NFKMLXTrainer.train(net, optimizer: AdamW(learningRate: 1e-4),
                                                     steps: 1, sample: { _ in photos },
                                                     loss: objective.callAsFunction)) { error in
            guard case NFKMLXError.nothingToTrain = error else {
                return XCTFail("expected nothingToTrain, got \(error)")
            }
        }
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
}
