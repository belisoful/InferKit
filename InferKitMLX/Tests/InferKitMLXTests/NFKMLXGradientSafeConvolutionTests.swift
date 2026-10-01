//
//  NFKMLXGradientSafeConvolutionTests.swift
//  InferKitMLXTests
//
//  The sliced convolution that keeps a wide stride-1 convolution's input gradient exact on the GPU, and
//  the trainer's swap that applies it for the length of a run.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXGradientSafeConvolutionTests: XCTestCase {

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

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.reshaped([-1]).asArray(Float.self).map(Double.init)
        let y = b.reshaped([-1]).asArray(Float.self).map(Double.init)
        let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        let (xx, yy) = (x.reduce(0) { $0 + $1 * $1 }, y.reduce(0) { $0 + $1 * $1 })
        guard xx > 0, yy > 0 else { return xx == yy ? 1 : 0 }
        return dot / (xx * yy).squareRoot()
    }

    /// One convolution geometry: the input and kernel shapes and how it pads, dilates, and groups.
    private struct Geometry {
        let name: String
        let input: [Int]
        let kernel: [Int]
        let padding: [Int]
        var dilation: [Int]? = nil
        var groups = 1
    }

    /// Geometries the GPU differentiates wrongly in mlx 0.32.2, measured against the CPU.
    private let failing = [
        Geometry(name: "conv1d 17 taps, 8 → 32", input: [2, 60, 8], kernel: [32, 17, 8], padding: [8]),
        Geometry(name: "conv1d 32 taps, 32 → 32", input: [1, 50, 32], kernel: [32, 32, 32], padding: [16]),
        Geometry(name: "conv1d 17 taps dilated 2", input: [1, 80, 8], kernel: [32, 17, 8], padding: [16], dilation: [2]),
        Geometry(name: "conv1d 128 taps, 16 groups", input: [1, 173, 768], kernel: [768, 128, 48], padding: [64], groups: 16),
        Geometry(name: "conv2d 3×39, 8 → 32", input: [1, 20, 60, 8], kernel: [32, 3, 39, 8], padding: [1, 19]),
        Geometry(name: "conv2d 1×17, 8 → 24", input: [1, 12, 40, 8], kernel: [24, 1, 17, 8], padding: [0, 8]),
        Geometry(name: "conv2d 17×17, 8 → 16", input: [1, 30, 30, 8], kernel: [16, 17, 17, 8], padding: [8, 8]),
        Geometry(name: "conv3d 1×1×17, 8 → 32", input: [1, 3, 4, 40, 8], kernel: [32, 1, 1, 17, 8], padding: [0, 0, 8]),
    ]

    private func mlxConvolution(_ x: MLXArray, _ w: MLXArray, _ geometry: Geometry) -> MLXArray {
        let d = geometry.dilation ?? Array(repeating: 1, count: geometry.padding.count)
        let p = geometry.padding
        switch p.count {
        case 1: return conv1d(x, w, padding: p[0], dilation: d[0], groups: geometry.groups)
        case 2: return conv2d(x, w, padding: .init((p[0], p[1])), dilation: .init((d[0], d[1])), groups: geometry.groups)
        default: return conv3d(x, w, padding: .init((p[0], p[1], p[2])), dilation: .init((d[0], d[1], d[2])),
                               groups: geometry.groups)
        }
    }

    private func slicedConvolution(_ x: MLXArray, _ w: MLXArray, _ geometry: Geometry) -> MLXArray {
        NFKMLXGradientSafeConvolution.convolve(
            x, w, padding: geometry.padding,
            dilation: geometry.dilation ?? Array(repeating: 1, count: geometry.padding.count), groups: geometry.groups)
    }

    private func arrays(_ geometry: Geometry, seed: UInt64) -> (x: MLXArray, w: MLXArray) {
        (MLXRandom.normal(geometry.input, key: MLXRandom.key(seed)),
         MLXRandom.normal(geometry.kernel, key: MLXRandom.key(seed + 1)) * 0.1)
    }

    func testTheSlicedConvolutionComputesMLXsConvolution() throws {
        try requireMLXRuntime()
        for (index, geometry) in failing.enumerated() {
            let (x, w) = arrays(geometry, seed: UInt64(10 * index))
            let expected = mlxConvolution(x, w, geometry)
            let sliced = slicedConvolution(x, w, geometry)
            XCTAssertEqual(sliced.shape, expected.shape, geometry.name)
            XCTAssertLessThan(abs(sliced - expected).max().item(Float.self), 1e-4, geometry.name)
        }
    }

    /// The input gradient of each geometry on the GPU against the CPU: MLX's own convolution's is wrong,
    /// which is the premise, and the sliced form's matches.
    func testTheSlicedInputGradientMatchesTheCPUWhereMLXsDoesNot() throws {
        try requireMLXRuntime()
        for (index, geometry) in failing.enumerated() {
            let (x, w) = arrays(geometry, seed: UInt64(10 * index + 100))
            let upstream = MLXRandom.normal(mlxConvolution(x, w, geometry).shape, key: MLXRandom.key(UInt64(index)))
            func gradient(on device: Device, _ forward: @escaping (MLXArray) -> MLXArray) -> MLXArray {
                Device.withDefaultDevice(device) {
                    let g = valueAndGrad { (inputs: [MLXArray]) in [(forward(inputs[0]) * upstream).sum()] }([x]).1[0]
                    eval(g)
                    return g
                }
            }
            let reference = gradient(on: .cpu) { self.mlxConvolution($0, w, geometry) }
            let sliced = cosine(gradient(on: .gpu) { self.slicedConvolution($0, w, geometry) }, reference)
            let fused = cosine(gradient(on: .gpu) { self.mlxConvolution($0, w, geometry) }, reference)
            print("PROBE gradient-safe convolution \(geometry.name): sliced \(sliced), MLX's own \(fused)")
            XCTAssertGreaterThan(sliced, 0.999999, geometry.name)
        }
    }

    func testOnlyAStrideOneConvolutionWiderThanSixteenTapsIsSliced() {
        XCTAssertTrue(NFKMLXGradientSafeConvolution.needsSlicing(kernel: [17], stride: [1]))
        XCTAssertTrue(NFKMLXGradientSafeConvolution.needsSlicing(kernel: [3, 39], stride: [1, 1]))
        XCTAssertFalse(NFKMLXGradientSafeConvolution.needsSlicing(kernel: [16], stride: [1]))
        XCTAssertFalse(NFKMLXGradientSafeConvolution.needsSlicing(kernel: [11, 11], stride: [1, 1]))
        XCTAssertFalse(NFKMLXGradientSafeConvolution.needsSlicing(kernel: [400], stride: [160]))
        XCTAssertFalse(NFKMLXGradientSafeConvolution.needsSlicing(kernel: [3, 39], stride: [1, 3]))
    }

    // MARK: The trainer's swap

    /// A trainable projection before a wide convolution: the shape in which the GPU's wrong input
    /// gradient reaches a parameter.
    private final class ProjectionBeforeWideConvolution: Module {
        @ModuleInfo(key: "projection") var projection: Linear
        @ModuleInfo(key: "convolution") var convolution: Conv1d
        @ModuleInfo(key: "narrow") var narrow: Conv1d

        override init() {
            _projection.wrappedValue = Linear(8, 8)
            _convolution.wrappedValue = Conv1d(inputChannels: 8, outputChannels: 32, kernelSize: 33, padding: 16)
            _narrow.wrappedValue = Conv1d(inputChannels: 32, outputChannels: 4, kernelSize: 3, padding: 1)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            narrow(convolution(projection(x)))
        }
    }

    private final class WideConvolutionInAPlainProperty: Module {
        let convolution = Conv1d(inputChannels: 8, outputChannels: 32, kernelSize: 17, padding: 8)

        func callAsFunction(_ x: MLXArray) -> MLXArray { convolution(x) }
    }

    /// One SGD step at rate 1 moves each parameter by minus its gradient, so the projection's movement
    /// on the GPU through the trainer is its gradient, held against the same gradient on the CPU.
    func testATrainingStepGetsTheGradientBeforeAWideConvolutionRight() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(5)
        let model = ProjectionBeforeWideConvolution()
        let input = MLXRandom.normal([2, 40, 8], key: MLXRandom.key(21))
        let target = MLXRandom.normal([2, 40, 4], key: MLXRandom.key(22))
        let loss = { (model: ProjectionBeforeWideConvolution, x: MLXArray, y: MLXArray) -> MLXArray in
            (model(x) - y).square().mean()
        }

        let reference: MLXArray = Device.withDefaultDevice(.cpu) {
            let gradients = valueAndGrad(model: model) { model, arrays in [loss(model, arrays[0], arrays[1])] }(model, [input, target]).1
            let g = Dictionary(uniqueKeysWithValues: gradients.flattened())["projection.weight"]!
            eval(g)
            return g
        }
        let before = model.projection.weight + 0
        eval(before)

        try NFKMLXTrainer.train(model, optimizer: SGD(learningRate: 1), steps: 1,
                                batch: { _ in (input, target) }, loss: loss)
        let movement = before - model.projection.weight
        let similarity = cosine(movement, reference)
        print("PROBE gradient-safe convolution through the trainer: projection gradient cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.99999)
        XCTAssertTrue(type(of: model.convolution) == Conv1d.self, "the original convolution is back")
    }

    func testTheSwapCarriesParametersFrozenStateAndModeAndRestoresThem() throws {
        try requireMLXRuntime()
        let model = ProjectionBeforeWideConvolution()
        model.convolution.freeze(keys: ["bias"])
        model.train(false)
        // An evaluated copy: `restore()` writes into the original's own arrays, so the live array would
        // read the restored value.
        let weight = model.convolution.weight + 0
        eval(weight)

        let installation = try NFKMLXGradientSafeConvolution.install(in: model)
        XCTAssertEqual(installation.paths, ["convolution"], "only the wide stride-1 convolution swaps")
        XCTAssertTrue(model.convolution is NFKMLXSlicedConv1d)
        XCTAssertEqual(abs(model.convolution.weight - weight).max().item(Float.self), 0)
        XCTAssertTrue(model.convolution.noGrad().contains("bias"))
        XCTAssertFalse(model.convolution.training)

        model.convolution.update(parameters: ModuleParameters.unflattened([("weight", weight * 2)]))
        installation.restore()
        XCTAssertTrue(type(of: model.convolution) == Conv1d.self)
        XCTAssertEqual(abs(model.convolution.weight - weight * 2).max().item(Float.self), 0,
                       "what the run trained goes back into the original")
    }

    func testTheSwapLeavesTheRandomStreamWhereItWas() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(9)
        let untouched = MLXRandom.normal([4]).asArray(Float.self)
        let model = ProjectionBeforeWideConvolution()
        NFKMLXRandom.seed(9)
        let installation = try NFKMLXGradientSafeConvolution.install(in: model)
        let afterSwap = MLXRandom.normal([4]).asArray(Float.self)
        installation.restore()
        XCTAssertEqual(afterSwap, untouched)
    }

    /// Blocks in an array, each holding a wide convolution, the way a Conformer stack holds its
    /// depthwise convolutions. MLX refuses a path through `blocks.1` alone, so the swap goes through
    /// each block.
    private final class Block: Module {
        @ModuleInfo(key: "convolution") var convolution: Conv1d

        override init() {
            _convolution.wrappedValue = Conv1d(inputChannels: 8, outputChannels: 8, kernelSize: 31, padding: 15,
                                               groups: 8)
            super.init()
        }
    }

    private final class Stack: Module {
        @ModuleInfo(key: "blocks") var blocks: [Block]
        @ModuleInfo(key: "convolutions") var convolutions: [Conv1d]

        override init() {
            _blocks.wrappedValue = [Block(), Block(), Block()]
            _convolutions.wrappedValue = [
                Conv1d(inputChannels: 8, outputChannels: 8, kernelSize: 3, padding: 1),
                Conv1d(inputChannels: 8, outputChannels: 8, kernelSize: 17, padding: 8),
                Conv1d(inputChannels: 8, outputChannels: 8, kernelSize: 17, padding: 8),
            ]
            super.init()
        }
    }

    func testConvolutionsInsideArraysSwapAndRestoreAtEveryIndex() throws {
        try requireMLXRuntime()
        let model = Stack()
        let installation = try NFKMLXGradientSafeConvolution.install(in: model)
        XCTAssertEqual(Set(installation.paths), ["blocks.0.convolution", "blocks.1.convolution", "blocks.2.convolution",
                                                 "convolutions.1", "convolutions.2"])
        XCTAssertTrue(model.blocks.allSatisfy { $0.convolution is NFKMLXSlicedConv1d })
        XCTAssertTrue(type(of: model.convolutions[0]) == Conv1d.self, "a narrow convolution stays as it is")
        XCTAssertTrue(model.convolutions[1] is NFKMLXSlicedConv1d)
        XCTAssertTrue(model.convolutions[2] is NFKMLXSlicedConv1d)

        installation.restore()
        XCTAssertTrue(model.blocks.allSatisfy { type(of: $0.convolution) == Conv1d.self })
        XCTAssertTrue(model.convolutions.allSatisfy { type(of: $0) == Conv1d.self })
    }

    func testAWideConvolutionInAPlainPropertyIsRefused() throws {
        try requireMLXRuntime()
        XCTAssertThrowsError(try NFKMLXGradientSafeConvolution.install(in: WideConvolutionInAPlainProperty())) { error in
            guard case NFKMLXError.unsupportedConfiguration = error else {
                return XCTFail("expected unsupportedConfiguration, got \(error)")
            }
        }
    }
}
