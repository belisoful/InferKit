//
//  NFKMLXStyleTransferTrainingTests.swift
//  InferKitMLXTests
//
//  Training fast style transfer without released weights: the reference's arguments, its Gram matrix
//  and batch normalization, a fresh network's torch biases, gradient reach, a fine-tune that lowers the
//  loss, and a reload through the public factories.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXStyleTransferTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_261_027)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// Images `[batch, size, size, 3]` in `0...1`, a different pattern per image.
    static func images(batch: Int, size: Int, phase: Float = 0) -> MLXArray {
        var values = [Float]()
        for n in 0 ..< batch {
            for y in 0 ..< size {
                for x in 0 ..< size {
                    for c in 0 ..< 3 {
                        values.append(0.5 + 0.4 * sinf(Float(x) * 0.37 + Float(c) * 1.1 + Float(n) + phase)
                                      * cosf(Float(y) * 0.23 - Float(c) + phase))
                    }
                }
            }
        }
        return MLXArray(values, [batch, size, size, 3])
    }

    func testTheObjectiveTakesTheReferenceWeights() throws {
        try requireMLXRuntime()
        let objective = try NFKMLXStyleTransferObjective(vggWeightsURL: nil, styleImage: Self.images(batch: 1, size: 16)[0])
        XCTAssertEqual(objective.contentWeight, 1e5)
        XCTAssertEqual(objective.styleWeight, 1e10)
        XCTAssertEqual(objective.styleGrams.map(\.shape), [[1, 64, 64], [1, 128, 128], [1, 256, 256], [1, 512, 512]],
                       "relu1_2, relu2_2, relu3_3, and relu4_3")
    }

    func testTheGramMatrixIsDividedByChannelsAndPositions() throws {
        try requireMLXRuntime()
        let features = MLXArray((0 ..< 2 * 3 * 2 * 4).map { Float($0 % 7) - 3 }, [2, 3, 2, 4])
        let gram = NFKMLXStyleTransferObjective.gram(features)
        XCTAssertEqual(gram.shape, [2, 4, 4])
        let flat = features.asArray(Float.self)
        for n in 0 ..< 2 {
            for i in 0 ..< 4 {
                for j in 0 ..< 4 {
                    var sum: Float = 0
                    for position in 0 ..< 6 {
                        sum += flat[(n * 6 + position) * 4 + i] * flat[(n * 6 + position) * 4 + j]
                    }
                    XCTAssertEqual(gram[n, i, j].item(Float.self), sum / 24, accuracy: 1e-6)
                }
            }
        }
    }

    func testTheBatchNormalizationDividesBy255ThenStandardizes() throws {
        try requireMLXRuntime()
        let pixels = MLXArray([Float(0), 127.5, 255], [1, 1, 1, 3])
        let normalized = NFKMLXStyleTransferObjective.normalized(pixels).asArray(Float.self)
        XCTAssertEqual(normalized[0], (0 - 0.485) / 0.229, accuracy: 1e-6)
        XCTAssertEqual(normalized[1], (0.5 - 0.456) / 0.224, accuracy: 1e-6)
        XCTAssertEqual(normalized[2], (1 - 0.406) / 0.225, accuracy: 1e-6)
    }

    func testAFloat64ConvolutionMatchesTheLayerOnTheCPU() throws {
        try requireMLXRuntime()
        Device.withDefaultDevice(.cpu) {
            for (kernel, stride, padding) in [(9, 1, 0), (3, 2, 0), (3, 1, 1)] {
                let layer = Conv2d(inputChannels: 4, outputChannels: 6, kernelSize: IntOrPair(kernel),
                                   stride: IntOrPair(stride), padding: IntOrPair(padding))
                let x = MLXRandom.normal([2, 13, 11, 4])
                let expected = layer(x)
                layer.update(parameters: ModuleParameters.unflattened(layer.parameters().flattened().map { ($0.0, $0.1.asType(.float64)) }))
                let widened = NFKFloat64Convolution.apply(layer, x.asType(.float64))
                XCTAssertEqual(widened.dtype, .float64)
                XCTAssertEqual(widened.shape, expected.shape)
                let difference = abs(widened.asType(.float32) - expected).max().item(Float.self)
                XCTAssertLessThan(difference, 1e-5, "kernel \(kernel), stride \(stride), padding \(padding)")
            }
        }
    }

    func testAFreshNetworkDrawsItsBiasesAsTorchDoes() throws {
        try requireMLXRuntime()
        let net = try NFKMLXStyleTransfer.network(weightsURL: nil)
        let parameters = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        for (name, fanIn) in [("conv1.conv2d.bias", 3 * 81), ("res3.conv1.conv2d.bias", 128 * 9),
                              ("deconv3.conv2d.bias", 32 * 81)] {
            let bias = try XCTUnwrap(parameters[name], name)
            let largest = abs(bias).max().item(Float.self)
            XCTAssertGreaterThan(largest, 0, "\(name) is drawn, not zero")
            XCTAssertLessThanOrEqual(largest, (1 / Float(fanIn)).squareRoot(), "\(name) stays within 1/√fan_in")
        }
        XCTAssertEqual(try XCTUnwrap(parameters["in1.weight"]).asArray(Float.self), [Float](repeating: 1, count: 32))
    }

    func testEveryWeightOutsideTheNormalizedBiasesReceivesAGradient() throws {
        try requireMLXRuntime()
        let net = NFKMLXStyleTransferNet(base: 4)
        let objective = try NFKMLXStyleTransferObjective(vggWeightsURL: nil, styleImage: Self.images(batch: 1, size: 24, phase: 2)[0])
        let gradients = valueAndGrad(model: net) { net, arrays in
            [objective(net, arrays[0])]
        }(net, [Self.images(batch: 2, size: 16)]).1.flattened()
        XCTAssertEqual(gradients.count, net.trainableParameters().flattened().count)
        for (name, gradient) in gradients {
            // A constant added where only linear layers stand before the next instance norm is subtracted
            // back out, so these biases' gradients are zero up to rounding, in the reference as here: every
            // convolution's but the last, and each residual block's second norm's.
            if (name.hasSuffix("conv2d.bias") && !name.hasPrefix("deconv3")) || (name.hasSuffix("in2.bias") && name.hasPrefix("res")) {
                continue
            }
            XCTAssertGreaterThan(abs(gradient).max().item(Float.self), 0, "\(name) receives a gradient")
        }
    }

    func testFineTuningLowersTheLoss() throws {
        try requireMLXRuntime()
        let net = NFKMLXStyleTransferNet(base: 4)
        let objective = try NFKMLXStyleTransferObjective(vggWeightsURL: nil, styleImage: Self.images(batch: 1, size: 32, phase: 2)[0])
        let content = Self.images(batch: 2, size: 32)
        let history = try NFKMLXStyleTransfer.fineTune(net, contentImages: { _ in content }, objective: objective, steps: 30)
        XCTAssertEqual(history.count, 30)
        XCTAssertLessThan(history.last!, history.first!, "the objective fell: \(history.first!) -> \(history.last!)")
    }

    func testAFineTunedCheckpointReloadsThroughThePublicFactories() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("style-tuned-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let net = try NFKMLXStyleTransfer.network(weightsURL: nil)
        let objective = try NFKMLXStyleTransferObjective(vggWeightsURL: nil, styleImage: Self.images(batch: 1, size: 32, phase: 2)[0])
        let content = Self.images(batch: 1, size: 32)
        try NFKMLXStyleTransfer.fineTune(net, contentImages: { _ in content }, objective: objective, steps: 1)
        try NFKMLXWeights.save(net, to: url)

        let reloaded = try NFKMLXStyleTransfer.network(weightsURL: url)
        let expected = net(content * 255), actual = reloaded(content * 255)
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self), "the fine-tuned checkpoint round-trips exactly")

        let backend = try NFKMLXStyleTransfer.backend(weightsURL: url)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: NFKMLXZeroDCETrainingTests.darkImage(32)]))
        XCTAssertNotNil(result.output(forKey: NFKOutputImage), "the backend stylizes with the trained weights")
    }
}
