//
//  NFKMLXWeightsTests.swift
//  InferKitMLXTests
//
//  The checkpoint key-coverage guard. MLX's `update(parameters:)` ignores unrecognized keys, so a
//  checkpoint whose names do not match leaves parameters randomly initialized and the model produces
//  garbage without any error — the U²-Net failure. These verify the guard catches that and stays out of
//  the way of a matching checkpoint.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXWeightsTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    func testAMatchingCheckpointApplies() throws {
        try requireMLXRuntime()
        let source = Linear(4, 4)
        let target = Linear(4, 4)
        let mapped = source.parameters().flattened()
        try NFKMLXWeights.apply(mapped, to: target)
        eval(target)
        XCTAssertEqual(target.weight.asArray(Float.self), source.weight.asArray(Float.self),
                       "the checkpoint's values reached the module")
    }

    func testAMisnamedCheckpointThrowsInsteadOfLoadingRandomWeights() throws {
        try requireMLXRuntime()
        let source = Linear(4, 4)
        // The U²-Net shape of the bug: every key carries a suffix the module does not know, so nothing
        // matches and `update(parameters:)` alone would leave the module at its random initialization.
        let misnamed = source.parameters().flattened().map { ($0.0 + "_s1", $0.1) }

        XCTAssertThrowsError(try NFKMLXWeights.apply(misnamed, to: Linear(4, 4))) { error in
            guard case NFKMLXError.weightsMismatch(let detail) = error else {
                return XCTFail("expected weightsMismatch, got \(error)")
            }
            XCTAssertTrue(detail.contains("weight"), "names the uncovered parameters: \(detail)")
            XCTAssertTrue(detail.contains("remap"), "points at the fix: \(detail)")
        }
    }

    func testAPartiallyCoveredCheckpointThrows() throws {
        try requireMLXRuntime()
        let source = Linear(4, 4)
        let partial = source.parameters().flattened().filter { $0.0 != "bias" }
        XCTAssertThrowsError(try NFKMLXWeights.apply(partial, to: Linear(4, 4)),
                             "a checkpoint missing one parameter is still a silent-corruption risk")
    }

    func testExtraCheckpointKeysAreHarmless() throws {
        try requireMLXRuntime()
        let source = Linear(4, 4)
        // Real checkpoints carry keys the module does not use (`num_batches_tracked`, a teacher branch).
        let extra = source.parameters().flattened() + [("teacher.weight", MLXArray.zeros([4, 4]))]
        XCTAssertNoThrow(try NFKMLXWeights.apply(extra, to: Linear(4, 4)))
    }

    func testStrictFalseAllowsADeliberatePartialLoad() throws {
        try requireMLXRuntime()
        let partial = Linear(4, 4).parameters().flattened().filter { $0.0 != "bias" }
        XCTAssertNoThrow(try NFKMLXWeights.apply(partial, to: Linear(4, 4), strict: false))
    }

    // MARK: - Saving

    private func temporaryURL(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(name)
    }

    func testASavedCheckpointReportsItsOwnLayoutAndReloadsExactly() throws {
        try requireMLXRuntime()
        let url = temporaryURL("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let source = Conv2d(inputChannels: 3, outputChannels: 4, kernelSize: 3)
        try NFKMLXWeights.save(source, to: url)

        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        XCTAssertFalse(checkpoint.needsConvTranspose,
                       "a checkpoint written by save is already in the module's layout")

        // The layout marker is what lets a model's loadWeights skip its PyTorch transpose, so applying
        // the arrays unchanged must reproduce the source exactly.
        let target = Conv2d(inputChannels: 3, outputChannels: 4, kernelSize: 3)
        try NFKMLXWeights.apply(Array(checkpoint.arrays), to: target)
        XCTAssertEqual(target.weight.asArray(Float.self), source.weight.asArray(Float.self))
        XCTAssertEqual(target.weight.shape, source.weight.shape,
                       "no transpose was applied in either direction")
    }

    func testAConvertedCheckpointStillReportsPyTorchLayout() throws {
        try requireMLXRuntime()
        let url = temporaryURL("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        // What every Tools/*-to-safetensors converter produces: no InferKit layout metadata.
        try MLX.save(arrays: ["weight": MLXArray.zeros([4, 3, 3, 3])], url: url)

        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        XCTAssertTrue(checkpoint.needsConvTranspose,
                      "an unmarked checkpoint keeps the existing PyTorch-layout behavior")
    }

    func testSavingRejectsAURLThatCannotCarryMetadata() throws {
        try requireMLXRuntime()
        let url = temporaryURL("\(UUID().uuidString).npy")
        XCTAssertThrowsError(try NFKMLXWeights.save(Linear(4, 4), to: url)) { error in
            guard case NFKMLXError.checkpointNotWritable = error else {
                return XCTFail("expected checkpointNotWritable, got \(error)")
            }
        }
    }

    func testSavingOverAnExistingCheckpointReplacesItInPlace() throws {
        try requireMLXRuntime()
        let url = temporaryURL("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try NFKMLXWeights.save(Linear(4, 4), to: url)
        let second = Linear(4, 4)
        try NFKMLXWeights.save(second, to: url)

        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let reloaded = Linear(4, 4)
        try NFKMLXWeights.apply(Array(checkpoint.arrays), to: reloaded)
        XCTAssertEqual(reloaded.weight.asArray(Float.self), second.weight.asArray(Float.self),
                       "the later write won, and the atomic replacement left no scratch file behind")

        let strays = try FileManager.default.contentsOfDirectory(
            atPath: url.deletingLastPathComponent().path).filter { $0.hasPrefix(".") && $0.hasSuffix(".safetensors") }
        XCTAssertTrue(strays.isEmpty, "no scratch file survived: \(strays)")
    }

    func testSavingIncludesNonTrainableParameters() throws {
        try requireMLXRuntime()
        let url = temporaryURL("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        // A model carrying batch-normalization running statistics must reload complete, so the file
        // covers every parameter rather than only the trainable ones.
        let source = BatchNorm(featureCount: 4)
        source.freeze()
        try NFKMLXWeights.save(source, to: url)

        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        XCTAssertNoThrow(try NFKMLXWeights.apply(Array(checkpoint.arrays), to: BatchNorm(featureCount: 4)),
                         "the saved file covers every parameter the module expects")
    }
}
