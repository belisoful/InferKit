//
//  NFKMLXConvTasNetTrainingTests.swift
//  InferKitMLXTests
//
//  Conv-TasNet's full fine-tune: the permutation-invariant objective, the geometry the factories read
//  from a checkpoint, the recipe, and the round trip through the model's own factory. Parity of the
//  objective against asteroid's `PITLossWrapper` lives in NFKMLXReferenceParityTests.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXConvTasNetTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_924)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Two speakers and their mixture, 2,000 samples.
    private func example() -> (mixture: [Float], sources: [[Float]]) {
        let first = (0 ..< 2000).map { 0.4 * sinf(Float($0) * 0.07) }
        let second = (0 ..< 2000).map { 0.3 * sinf(Float($0) * 0.013) * cosf(Float($0) * 0.002) }
        return (zip(first, second).map(+), [first, second])
    }

    func testTheObjectiveIgnoresTheSpeakerOrderAndTheScale() throws {
        try requireMLXRuntime()
        let (_, sources) = example()
        let target = MLXArray(sources.flatMap { $0 }).reshaped([2, 2000])
        let swappedAndScaled = concatenated([2 * target[1 ..< 2], 0.5 * target[0 ..< 1]], axis: 0)
        let objective = NFKMLXConvTasNetObjective()
        let exact = objective.loss(estimates: target, sources: target).item(Float.self)
        let crossed = objective.loss(estimates: swappedAndScaled, sources: target).item(Float.self)
        XCTAssertEqual(crossed, exact, accuracy: 1e-3)
        XCTAssertLessThan(exact, -60, "a perfect estimate scores a large negative SI-SDR")
    }

    func testPermutationsCoverEveryAssignment() {
        XCTAssertEqual(Set(NFKMLXConvTasNetObjective.permutations(3).map { $0.map(String.init).joined() }),
                       ["012", "021", "102", "120", "201", "210"])
    }

    func testFineTuningLowersTheLoss() throws {
        try requireMLXRuntime()
        let net = NFKMLXConvTasNetNet(.tiny)
        let item = example()
        let history = try NFKMLXConvTasNet.fineTune(net, examples: { _ in item }, steps: 12)
        XCTAssertEqual(history.count, 12)
        XCTAssertTrue(history.allSatisfy(\.isFinite))
        XCTAssertLessThan(history.suffix(3).reduce(0, +), history.prefix(3).reduce(0, +))
    }

    func testTheFactoriesReadTheGeometryFromTheCheckpoint() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("convtasnet-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = NFKMLXConvTasNetNet(.tiny)
        let item = example()
        try NFKMLXConvTasNet.fineTune(net, examples: { _ in item }, steps: 2)
        try NFKMLXWeights.save(net, to: url)

        let configuration = try NFKMLXConvTasNet.configuration(matching: url)
        XCTAssertEqual(configuration.filters, NFKMLXConvTasNetConfiguration.tiny.filters)
        XCTAssertEqual(configuration.kernel, NFKMLXConvTasNetConfiguration.tiny.kernel)
        XCTAssertEqual(configuration.speakers, 2)
    }

    func testAFineTunedCheckpointReloadsThroughTheFactory() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("convtasnet-reload-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        // Every width the checkpoint records differs from the default; the block layout does not.
        var geometry = NFKMLXConvTasNetConfiguration.base
        geometry.filters = 32
        geometry.kernel = 8
        geometry.bottleneck = 16
        geometry.hidden = 32
        let net = NFKMLXConvTasNetNet(geometry)
        let item = example()
        try NFKMLXConvTasNet.fineTune(net, examples: { _ in item }, steps: 2)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXConvTasNet.network(weightsURL: url)
        let mixture = MLXArray(item.mixture)
        XCTAssertLessThan(abs(reloaded.separate(mixture) - net.separate(mixture)).max().item(Float.self), 1e-5)
        XCTAssertTrue(try NFKMLXConvTasNet.backend(weightsURL: url).isReady)
    }
}
