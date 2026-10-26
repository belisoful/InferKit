//
//  NFKMLXSD3StreamingTests.swift
//  InferKitMLXTests
//
//  Streaming the SD3 MMDiT's blocks. A random transformer is written out as a sharded bfloat16 diffusers
//  release, and the same release loaded with every block held and with its blocks streamed must compute
//  the same values, element for element. On released weights, the SD3.5 Large four-block cut streams to
//  its held values, the whole 8B transformer streams at float32 to `run_reference.py sd35_layerwise`
//  (`IK_VAL_SD35_LARGE_RELEASE`, `IK_PARITY_SD35_LARGE_LAYERWISE`), and the generator draws through its
//  streamed transformer.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSD3StreamingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Six blocks, the first dual-attention, the last `context_pre_only`, with RMS query-key norms.
    private let configuration = NFKMLXSD3Configuration(
        sampleSize: 16, inChannels: 4, outChannels: 4, numLayers: 6, attentionHeadDim: 8,
        numAttentionHeads: 2, jointAttentionDim: 24, captionProjectionDim: 16, pooledProjectionDim: 20,
        posEmbedMaxSize: 8, dualAttentionLayers: [0], qkNorm: true)

    /// A random transformer written as a two-shard bfloat16 diffusers release, the patch-embed
    /// convolution stored `[out, in, h, w]` as the releases store it.
    private func release() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("sd3-stream-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        MLXRandom.seed(3)
        let net = NFKMLXSD3TransformerNet(configuration)
        let stored = net.parameters().flattened().map { key, value -> (String, MLXArray) in
            let varied = key.hasSuffix("norm_q.weight") || key.hasSuffix("norm_k.weight")
                || key.hasSuffix("norm_added_q.weight") || key.hasSuffix("norm_added_k.weight")
            let random = varied ? 1 + MLXRandom.normal(value.shape) * 0.1 : MLXRandom.normal(value.shape) * 0.2
            return (key, (value.ndim == 4 ? random.transposed(0, 3, 1, 2) : random).asType(.bfloat16))
        }.sorted { $0.0 < $1.0 }
        var weightMap = [String: String]()
        for (index, half) in [stored.prefix(stored.count / 2), stored.dropFirst(stored.count / 2)].enumerated() {
            let name = "diffusion_pytorch_model-0000\(index + 1)-of-00002.safetensors"
            try save(arrays: Dictionary(half.map { $0 }, uniquingKeysWith: { first, _ in first }),
                     url: directory.appendingPathComponent(name))
            half.forEach { weightMap[$0.0] = name }
        }
        try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
            .write(to: directory.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json"))
        return directory
    }

    private func transformer(_ directory: URL, configuration: NFKMLXSD3Configuration? = nil,
                             precision: NFKMLXWeightPrecision, converting conversion: DType? = nil,
                             streamed: Set<Int> = []) throws -> NFKMLXSD3TransformerNet {
        let net = NFKMLXSD3TransformerNet(configuration ?? self.configuration)
        try NFKMLXSD3TransformerNet.loadWeights(into: net, from: directory, precision: precision, converting: conversion,
                                                streamed: streamed)
        return net
    }

    private struct Inputs {
        let latent: MLXArray
        let conditions: [(encoder: MLXArray, pooled: MLXArray)]
        let timestep: MLXArray
    }

    private func inputs(dtype: DType) -> Inputs {
        MLXRandom.seed(11)
        let latent = MLXRandom.normal([1, 4, 8, 8]).asType(dtype)
        let conditions = (0 ..< 2).map { _ in
            (MLXRandom.normal([1, 7, 24]).asType(dtype), MLXRandom.normal([1, 20]).asType(dtype))
        }
        return Inputs(latent: latent, conditions: conditions, timestep: MLXArray([Float(600)]))
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func assertHoldsNothing(_ net: NFKMLXSD3TransformerNet, _ label: String) throws {
        for index in try XCTUnwrap(net.layerStream, label).layers {
            XCTAssertTrue(net.transformerBlocks[index].parameters().flattened().allSatisfy { $0.1.size == 0 },
                          "\(label): block \(index) holds nothing between its turns")
        }
    }

    func testAStreamedTransformerComputesWhatAHeldOneDoes() throws {
        try requireMLXRuntime()
        let directory = try release()
        for (precision, conversion) in [(NFKMLXWeightPrecision.float32, DType?.none), (.checkpoint, .bfloat16)] {
            let label = "\(precision)"
            let held = try transformer(directory, precision: precision, converting: conversion)
            let streamed = try transformer(directory, precision: precision, converting: conversion, streamed: [2, 3, 4, 5])
            XCTAssertNil(held.layerStream, label)
            XCTAssertEqual(streamed.layerStream?.layers, [2, 3, 4, 5], label)
            let input = inputs(dtype: conversion ?? .float32)
            for pass in 0 ..< 2 {
                let condition = input.conditions[0]
                let expected = held(input.latent, encoderHidden: condition.encoder, pooled: condition.pooled,
                                    timestep: input.timestep)
                let actual = streamed(input.latent, encoderHidden: condition.encoder, pooled: condition.pooled,
                                      timestep: input.timestep)
                XCTAssertEqual(maxDifference(actual, expected), 0, "\(label) pass \(pass)")
                try assertHoldsNothing(streamed, "\(label) pass \(pass)")
            }
            try streamed.verifyStream()
        }
    }

    func testEveryConditionOfOnePassIsTheVelocityItsOwnPassComputes() throws {
        try requireMLXRuntime()
        let directory = try release()
        let held = try transformer(directory, precision: .checkpoint, converting: .bfloat16)
        let streamed = try transformer(directory, precision: .checkpoint, converting: .bfloat16, streamed: [1, 2, 3, 4, 5])
        let input = inputs(dtype: .bfloat16)
        let alone = input.conditions.map {
            held(input.latent, encoderHidden: $0.encoder, pooled: $0.pooled, timestep: input.timestep)
        }
        XCTAssertGreaterThan(maxDifference(alone[0], alone[1]), 0, "the two conditions differ")
        for (label, net) in [("held", held), ("streamed", streamed)] {
            let before = net.layerStream?.readStatistics.bytes ?? 0
            let together = net.velocities(input.latent, conditions: input.conditions, timestep: input.timestep)
            eval(together)
            XCTAssertEqual(together.count, 2, label)
            for (condition, velocity) in together.enumerated() {
                XCTAssertEqual(maxDifference(velocity, alone[condition]), 0, "\(label) condition \(condition)")
            }
            if let stream = net.layerStream {
                XCTAssertEqual(stream.readStatistics.bytes - before, stream.bytesPerPass, "one read serves both conditions")
            }
        }
    }

    func testAFailedReadThrowsOnceAndTheNextPassReadsEveryBlock() throws {
        try requireMLXRuntime()
        let directory = try release()
        let held = try transformer(directory, precision: .float32)
        let streamed = try transformer(directory, precision: .float32, streamed: [3, 4, 5])
        let input = inputs(dtype: .float32)
        let condition = input.conditions[0]
        func velocity(_ net: NFKMLXSD3TransformerNet) -> MLXArray {
            let value = net(input.latent, encoderHidden: condition.encoder, pooled: condition.pooled, timestep: input.timestep)
            eval(value)
            return value
        }
        let expected = velocity(held)

        let shards = ["diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"]
            .map { directory.appendingPathComponent($0) }
        let originals = try shards.map { try Data(contentsOf: $0) }
        // A pass primes the next one's first streamed block, so read past it before cutting the files.
        _ = velocity(streamed)
        for file in shards {
            let start = try XCTUnwrap(NFKMLXSafetensors.entries(inFile: file).values.map(\.start).min())
            let handle = try FileHandle(forWritingTo: file)
            try handle.truncate(atOffset: UInt64(start))
            try handle.close()
        }
        _ = velocity(streamed)
        _ = velocity(streamed)
        XCTAssertThrowsError(try streamed.verifyStream(), "the truncated release cannot be read")
        XCTAssertNoThrow(try streamed.verifyStream(), "a failure is reported once")
        for (file, original) in zip(shards, originals) {
            try original.write(to: file)
        }
        _ = velocity(streamed)
        try streamed.verifyStream()
        XCTAssertEqual(maxDifference(velocity(streamed), expected), 0, "the next pass reads every block again")
        try streamed.verifyStream()
    }

    func testTheFootprintCountsEachBlockAtTheTypeItLoadsAt() throws {
        let directory = try release()
        let stored = try NFKMLXSD3TransformerNet.footprint(directory: directory, precision: .checkpoint)
        let widened = try NFKMLXSD3TransformerNet.footprint(directory: directory, precision: .float32)
        let converted = try NFKMLXSD3TransformerNet.footprint(directory: directory, precision: .float32,
                                                             converting: .bfloat16)
        XCTAssertEqual(stored.layerBytes.count, 6)
        XCTAssertEqual(widened.layerBytes, stored.layerBytes.map { $0 * 2 })
        XCTAssertEqual(widened.bytes, stored.bytes * 2)
        XCTAssertEqual(converted, stored)
        XCTAssertEqual(stored.bytes, try NFKMLXStageWeights.bytes(inDirectory: directory, holding: .bfloat16))
    }

    // MARK: Released weights

    private func environmentURL(_ key: String) -> URL? {
        NFKMLXValidationConfig.environment[key].map { URL(fileURLWithPath: $0) }
    }

    func testTheSD35LargeCutStreamsToTheValuesItsHeldLoadComputes() throws {
        try requireMLXRuntime()
        guard let release = environmentURL("IK_VAL_SD35_LARGE") else {
            throw XCTSkip("set IK_VAL_SD35_LARGE (the four-block SD3.5 Large cut)")
        }
        let directory = release.appendingPathComponent("transformer")
        let configuration = try NFKMLXSD3TransformerNet.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        MLXRandom.seed(35)
        let latent = MLXRandom.normal([1, 16, 32, 32]), encoder = MLXRandom.normal([1, 77, 4096]) * 0.25
        let pooled = MLXRandom.normal([1, 2048]), timestep = MLXArray([Float(750)])
        func velocity(_ streamed: Set<Int>) throws -> MLXArray {
            try autoreleasepool {
                let net = try transformer(directory, configuration: configuration, precision: .float32, streamed: streamed)
                let value = net(latent, encoderHidden: encoder, pooled: pooled, timestep: timestep)
                eval(value)
                try net.verifyStream()
                return value
            }
        }
        let held = try velocity([])
        NFKMLXGPU.clearCache()
        let streamed = try velocity([1, 2, 3])
        XCTAssertEqual(maxDifference(streamed, held), 0)
    }

    func testTheStreamedSD35LargeTransformerMatchesItsReferenceAtFloat32() throws {
        try requireMLXRuntime()
        guard let release = environmentURL("IK_VAL_SD35_LARGE_RELEASE"),
              let recordURL = environmentURL("IK_PARITY_SD35_LARGE_LAYERWISE"),
              FileManager.default.fileExists(atPath: recordURL.path) else {
            throw XCTSkip("set IK_VAL_SD35_LARGE_RELEASE and IK_PARITY_SD35_LARGE_LAYERWISE (run_reference.py sd35_layerwise)")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: recordURL).arrays
        let directory = release.appendingPathComponent("transformer")
        let configuration = try NFKMLXSD3TransformerNet.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let budget = NFKMLXResidencyBudget.current()
        let streamed = try NFKMLXLayerStream.plannedLayers(
            residency: .streamed, budget: budget,
            footprint: { try NFKMLXSD3TransformerNet.footprint(directory: directory, precision: .float32) },
            fitsWhole: { throw NFKMLXError.unsupportedConfiguration("measured streamed") })
        XCTAssertFalse(streamed.isEmpty, "the float32 transformer streams on this machine")
        let net = try transformer(directory, configuration: configuration, precision: .float32, streamed: streamed)
        let stream = try XCTUnwrap(net.layerStream)

        let latent = try XCTUnwrap(record["latent"]), encoder = try XCTUnwrap(record["encoder"])
        let pooled = try XCTUnwrap(record["pooled"]), timestep = try XCTUnwrap(record["timestep"])
        var readings = [(String, Double)]()
        var image = net.posEmbed(latent)
        readings.append(("patch", NFKMLXWav2Vec2Tests.cosine(image[0], try XCTUnwrap(record["patch"]))))
        let temb = net.timeTextEmbed(timestep: timestep, pooled: pooled)
        var context: MLXArray? = net.contextEmbedder(encoder)
        let started = Date()
        for (index, block) in net.transformerBlocks.enumerated() {
            let output = try XCTUnwrap(NFKMLXLayerStream.run(stream, layer: block, index: index, {
                block(image, encoder: context!, temb: temb)
            }, keeping: { [$0.image] + ($0.context.map { [$0] } ?? []) }), "block \(index)")
            image = output.image
            context = output.context
            readings.append(("block\(index) image",
                             NFKMLXWav2Vec2Tests.cosine(image[0], try XCTUnwrap(record["block\(index).image"]))))
            if let context {
                readings.append(("block\(index) text",
                                 NFKMLXWav2Vec2Tests.cosine(context[0], try XCTUnwrap(record["block\(index).text"]))))
            }
        }
        try net.verifyStream()
        let seconds = Date().timeIntervalSince(started)
        let velocity = net(latent, encoderHidden: encoder, pooled: pooled, timestep: timestep)
        readings.append(("velocity", NFKMLXWav2Vec2Tests.cosine(velocity[0], try XCTUnwrap(record["output"]))))
        try net.verifyStream()

        let worst = try XCTUnwrap(readings.min { $0.1 < $1.1 })
        let (bytes, readSeconds) = stream.readStatistics
        print("VALIDATION PARITY sd3.5 large streamed float32: \(stream.layers.count) of \(net.transformerBlocks.count) blocks streamed, "
              + "\(stream.bytesPerPass) bytes per pass, first pass \(Int(seconds)) s, read \(bytes) bytes in \(readSeconds) s; "
              + "worst \(worst.0) \(worst.1); " + readings.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        for (seam, similarity) in readings {
            XCTAssertGreaterThan(similarity, 0.99999, "\(seam) diverges")
        }
    }

    func testTheSD35LargeGeneratorDrawsThroughItsStreamedTransformer() throws {
        try requireMLXRuntime()
        guard let release = environmentURL("IK_VAL_SD35_LARGE_RELEASE") else {
            throw XCTSkip("set IK_VAL_SD35_LARGE_RELEASE (the SD3.5 Large diffusers release)")
        }
        // The budget that holds the transformer's first 20 of 38 blocks beside the autoencoder.
        let transformer = try NFKMLXSD3TransformerNet.footprint(
            directory: release.appendingPathComponent("transformer"), precision: .checkpoint, converting: .bfloat16)
        let pipeline = NFKMLXStageFootprint(
            bytes: transformer.bytes + (try NFKMLXStageWeights.bytes(inDirectory: release.appendingPathComponent("vae"),
                                                                     holding: .float32)),
            layerBytes: transformer.layerBytes)
        let budget = NFKMLXResidencyBudget.leastBudget(
            holding: pipeline.streamedMinimumBytes + transformer.layerBytes.prefix(20).reduce(0, +))
        let generator = try NFKMLXSD3Generator.generator(directoryURL: release, residency: .streamed, budget: budget)
        XCTAssertFalse(generator.holdsStagesResident)
        XCTAssertEqual(generator.streamedLayerCount, 18, "the blocks past the first 20")
        XCTAssertEqual(generator.streamedBytesPerPass, transformer.layerBytes.suffix(18).reduce(0, +))
        let automatic = try NFKMLXSD3Generator.generator(directoryURL: release, residency: .automatic, budget: budget)
        XCTAssertEqual(automatic.streamedLayerCount, 18, "automatic streams the stage it cannot hold whole")
        let roomy = try NFKMLXSD3Generator.generator(directoryURL: release, residency: .automatic,
                                                     budget: NFKMLXResidencyBudget.leastBudget(holding: pipeline.bytes))
        XCTAssertEqual(roomy.streamedLayerCount, 0, "a budget that holds the stage whole streams nothing")
        generator.steps = 2
        let started = Date()
        let image = try generator.image(forPrompt: "a red fox in the snow", width: 256, height: 256, seed: 7)
        eval(image)
        XCTAssertEqual(image.shape, [256, 256, 3])
        let deviation = image.variance().sqrt().item(Float.self)
        XCTAssertTrue(deviation.isFinite)
        XCTAssertGreaterThan(deviation, 0.02, "the image is not flat")
        print("VALIDATION sd3.5 large streamed generator: \(generator.streamedLayerCount) blocks streamed, "
              + "\(generator.streamedBytesPerPass) bytes per pass, two guided steps at 256 in \(Int(Date().timeIntervalSince(started))) s")
    }
}
