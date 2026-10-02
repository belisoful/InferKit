//
//  NFKMLXDenoiserTrainingTests.swift
//  InferKitMLXTests
//
//  The speech denoiser's training path without released weights: the waveform distances and the STFT
//  magnitude, the augmentations' shapes and draws, the gradient through the half-sample resampler's wide
//  filter on the GPU against the CPU, and a fine-tune that lowers the loss and reloads through both
//  factories at the width it was built at.
//

import XCTest
import InferKit
import MLX
import MLXFFT
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

/// Replays a reference's recorded draws in order.
struct NFKDenoiserReplayedDraws: NFKDenoiserDraws {
    var values: [Double]
    var permutationValues: [Int]
    var offsetValues: [Int]

    mutating func unit() -> Double { values.removeFirst() }
    mutating func uniform(_ low: Double, _ high: Double) -> Double { values.removeFirst() }
    mutating func integer(_ low: Int, _ high: Int) -> Int { Int(values.removeFirst()) }
    mutating func permutation(_ count: Int) -> [Int] { permutationValues }
    mutating func offsets(below bound: Int, count: Int) -> [Int] { offsetValues }
}

/// Counts the draws an augmentation takes and answers each with a fixed value.
struct NFKDenoiserCountedDraws: NFKDenoiserDraws {
    var count = 0
    mutating func unit() -> Double { count += 1; return 0.5 }
    mutating func uniform(_ low: Double, _ high: Double) -> Double { count += 1; return (low + high) / 2 }
    mutating func integer(_ low: Int, _ high: Int) -> Int { count += 1; return low }
    mutating func permutation(_ count: Int) -> [Int] { self.count += 1; return Array((0 ..< count).reversed()) }
    mutating func offsets(below bound: Int, count: Int) -> [Int] { self.count += 1; return Array(repeating: 0, count: count) }
}

final class NFKMLXDenoiserTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_002)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Two short clips of a harmonic tone under noise, `[2, length]` each.
    private func pair(length: Int) -> (noisy: MLXArray, clean: MLXArray) {
        let clean = (0 ..< 2).flatMap { row in
            (0 ..< length).map { index in 0.3 * sinf(2 * .pi * Float(150 + 40 * row) * Float(index) / 16_000) }
        }
        let noise = MLXRandom.normal([2, length], key: MLXRandom.key(3)) * 0.05
        let cleanArray = MLXArray(clean, [2, length])
        return (cleanArray + noise, cleanArray)
    }

    func testTheDistancesScoreAsTheReferenceDefinesThem() throws {
        try requireMLXRuntime()
        let estimate = MLXArray([Float(0), 0.5, 3], [1, 3]), clean = MLXArray([Float(0), 0, 0], [1, 3])
        XCTAssertEqual(NFKMLXDenoiserObjective(distance: .l1).loss(estimate: estimate, clean: clean).item(Float.self),
                       3.5 / 3, accuracy: 1e-6)
        XCTAssertEqual(NFKMLXDenoiserObjective(distance: .l2).loss(estimate: estimate, clean: clean).item(Float.self),
                       9.25 / 3, accuracy: 1e-6)
        // Huber is quadratic under 1 and linear above: 0.5 · 0.25, then 3 − 0.5.
        XCTAssertEqual(NFKMLXDenoiserObjective(distance: .huber).loss(estimate: estimate, clean: clean).item(Float.self),
                       (0.125 + 2.5) / 3, accuracy: 1e-6)
    }

    func testTheSTFTMagnitudeMatchesTheFFT() throws {
        try requireMLXRuntime()
        let signal = MLXRandom.normal([1, 3000], key: MLXRandom.key(9))
        let ours = NFKMLXDenoiserObjective.magnitude(signal, fft: 512, hop: 50, window: 240)        // [1, frames, bins]
        let (real, imaginary) = NFKMLXComplexSTFT(nFFT: 512, hop: 50, winLength: 240).transformComplex(signal)
        let reference = sqrt(maximum(square(real) + square(imaginary), MLXArray(Float(1e-7)))).transposed(0, 2, 1)
        XCTAssertEqual(ours.shape, reference.shape)
        XCTAssertLessThan(abs(ours - reference).max().item(Float.self), 1e-3 * reference.max().item(Float.self))
    }

    func testTheAugmentationsKeepTheirShapesAndRepeatFromASeed() throws {
        try requireMLXRuntime()
        let (noisy, clean) = pair(length: 20_000)
        var first = NFKDenoiserSeededDraws(seed: 7), second = NFKDenoiserSeededDraws(seed: 7)
        let a = NFKMLXDenoiserAugmentation.dns.apply(noisy: noisy, clean: clean, draws: &first)
        let b = NFKMLXDenoiserAugmentation.dns.apply(noisy: noisy, clean: clean, draws: &second)
        XCTAssertEqual(a.noisy.shape, [2, 4000], "the shift crops 16000 samples")
        XCTAssertEqual(abs(a.noisy - b.noisy).max().item(Float.self), 0, "one seed, one augmentation")
        XCTAssertGreaterThan(abs(a.noisy - a.clean).max().item(Float.self), 0)
        let valentini = NFKMLXDenoiserAugmentation.valentini.apply(noisy: noisy, clean: clean, draws: &first)
        XCTAssertEqual(valentini.clean.shape, [2, 12_000])
    }

    func testAnAugmentationThatIsOffTakesNoDraw() throws {
        try requireMLXRuntime()
        let (noisy, clean) = pair(length: 4000)
        var draws = NFKDenoiserCountedDraws()
        let result = NFKMLXDenoiserAugmentation().apply(noisy: noisy, clean: clean, draws: &draws)
        XCTAssertEqual(draws.count, 0)
        XCTAssertEqual(abs(result.noisy - noisy).max().item(Float.self), 0, accuracy: 1e-6)
        _ = NFKMLXDenoiserAugmentation(revEcho: 1).apply(noisy: noisy, clean: clean, draws: &draws)
        XCTAssertGreaterThan(draws.count, 3, "the reverb's probability draw, its three settings, and its jitters")
    }

    /// One SGD step at rate 1 moves the first encoder weight by its gradient, which reaches it through
    /// the output resampler's 112-tap filter; the GPU step through the trainer is held to the CPU's.
    func testTheGradientThroughTheResamplerMatchesTheCPU() throws {
        try requireMLXRuntime()
        let net = NFKMLXDenoiser.makeNet(baseChannels: 4)
        let (noisy, clean) = pair(length: 4000)
        let objective = NFKMLXDenoiserObjective()
        let reference: MLXArray = Device.withDefaultDevice(.cpu) {
            net.train(true)
            let gradients = valueAndGrad(model: net) { net, arrays in [objective(net, arrays[0], arrays[1])] }(net, [noisy, clean]).1
            let g = Dictionary(uniqueKeysWithValues: gradients.flattened())["encoder.0.conv1.weight"]!
            eval(g)
            return g
        }
        let before = net.encoder[0].parameters().flattened().first { $0.0 == "conv1.weight" }!.1 * 1
        eval(before)
        try NFKMLXTrainer.train(net, optimizer: SGD(learningRate: 1), steps: 1,
                                batch: { _ in (noisy, clean) }, loss: objective.callAsFunction)
        let after = net.encoder[0].parameters().flattened().first { $0.0 == "conv1.weight" }!.1
        let movement = (before - after).reshaped([-1]), expected = reference.reshaped([-1])
        let relative = sqrt(square(movement - expected).sum()).item(Float.self) / sqrt(square(expected).sum()).item(Float.self)
        print("PROBE denoiser resampler gradient: first encoder weight relative error \(relative)")
        XCTAssertLessThan(relative, 1e-4)
    }

    /// The output layer's transposed convolution reads 16,000 positions per second of audio, past the
    /// 8,192 where MLX's GPU weight gradient of the 2-D form goes wrong; in training the layer's weight
    /// gradient is held to the CPU's, and its training forward to its inference forward.
    func testTheTransposedConvolutionsWeightGradientOnALongClipMatchesTheCPU() throws {
        try requireMLXRuntime()
        let layer = NFKDemucsConvT1d(48, 1, kernel: 8, stride: 4, padding: 0)
        let x = MLXRandom.normal([1, 10_000, 48], key: MLXRandom.key(4))
        layer.train(false)
        let inference = layer(x)
        layer.train(true)
        let training = layer(x)
        XCTAssertLessThan(abs(training - inference).max().item(Float.self), 1e-4 * abs(inference).max().item(Float.self))

        let upstream = MLXRandom.normal(training.shape, key: MLXRandom.key(5))
        func gradient() -> MLXArray {
            let gradients = valueAndGrad(model: layer) { layer, arrays in [(layer(arrays[0]) * arrays[1]).sum()] }(layer, [x, upstream]).1
            let weight = Dictionary(uniqueKeysWithValues: gradients.flattened())["conv.weight"]!
            eval(weight)
            return weight
        }
        let cpu = Device.withDefaultDevice(.cpu) { gradient() }
        let gpu = gradient()
        let relative = sqrt(square(gpu - cpu).sum()).item(Float.self) / sqrt(square(cpu).sum()).item(Float.self)
        print("PROBE denoiser transposed convolution at 10000 positions: GPU weight-gradient relative error \(relative)")
        XCTAssertLessThan(relative, 1e-4)
    }

    /// Every network holding `NFKDemucsConvT1d` starts in evaluation mode, so inference never takes the
    /// layer's training form, and a fine-tune hands its network back in evaluation mode.
    func testTheNetworksHoldingTheTransposedConvolutionStartInEvaluation() throws {
        try requireMLXRuntime()
        var music = NFKMLXDemucsConfiguration()
        music.baseChannels = 4
        music.depth = 3
        let networks: [(String, Module)] = [
            ("Demucs", NFKMLXDemucs.makeNet(music)),
            ("denoiser", NFKMLXDenoiser.makeNet(baseChannels: 4)),
            ("Conv-TasNet", NFKMLXConvTasNetNet(.tiny)),
            ("HiFi-GAN", NFKMLXHiFiGANNet(NFKMLXHiFiGANConfiguration())),
            ("Music 3 vocoder", NFKMusic3VocoderNet(.tiny)),
            ("SNAC", NFKMLXSNACNet(.tiny)),
            ("DAC", NFKMLXDACNet(.tiny)),
        ]
        for (name, net) in networks {
            XCTAssertTrue(net.modules().allSatisfy { !$0.training }, "\(name) starts in evaluation mode")
        }

        let net = NFKMLXDenoiser.makeNet(baseChannels: 4)
        let (noisy, clean) = pair(length: 4000)
        try NFKMLXDenoiser.fineTune(net, examples: { _ in (noisy, clean) }, steps: 1)
        XCTAssertTrue(net.modules().allSatisfy { !$0.training }, "a fine-tune restores evaluation mode")
    }

    func testFineTuningLowersTheLossAndReloadsThroughBothFactories() throws {
        try requireMLXRuntime()
        let net = try NFKMLXDenoiser.network(weightsURL: nil, baseChannels: 4)
        let (noisy, clean) = pair(length: 4000)
        let objective = NFKMLXDenoiserObjective()
        let before = objective(net, noisy, clean).item(Float.self)
        let losses = try NFKMLXDenoiser.fineTune(net, examples: { _ in (noisy, clean) },
                                                 optimizer: Adam(learningRate: 3e-3), steps: 8)
        XCTAssertEqual(losses.count, 8)
        XCTAssertLessThan(losses.last!, before, "the loss falls on the batch it trains on")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXDenoiser.network(weightsURL: url)
        XCTAssertEqual(reloaded.configuration.baseChannels, 4, "the width comes from the checkpoint")
        net.train(false)
        let input = noisy.expandedDimensions(axis: -1)
        XCTAssertLessThan(abs(net(input) - reloaded(input)).max().item(Float.self), 1e-5)
        XCTAssertNoThrow(try NFKMLXDenoiser.backend(weightsURL: url))
    }

    func testTheReferenceOptimizerStepsTheFoldedRecurrentBiasAtTwiceTheRate() throws {
        try requireMLXRuntime()
        let net = NFKMLXDenoiser.makeNet(baseChannels: 4)
        let optimizer = NFKMLXDenoiser.referenceOptimizer(for: net)
        let gradients = net.parameters().mapValues { MLXArray.ones(like: $0) }
        let before = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 * 1) })
        optimizer.update(model: net, gradients: gradients)
        eval(net)
        let after = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        let biasStep = abs(after["lstm.lstm.0.bias"]! - before["lstm.lstm.0.bias"]!).max().item(Float.self)
        let weightStep = abs(after["lstm.lstm.0.Wx"]! - before["lstm.lstm.0.Wx"]!).max().item(Float.self)
        XCTAssertEqual(weightStep, 3e-4, accuracy: 1e-6)
        XCTAssertEqual(biasStep, 6e-4, accuracy: 2e-6)
    }
}
