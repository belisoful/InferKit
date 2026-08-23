//
//  NFKMLXStyleTransfer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Fast neural style transfer runs a single feed-forward pass through Johnson et al.'s image
// transformation network (the reference `TransformerNet` from the PyTorch examples): three
// downsampling convolutions, five residual blocks, two nearest-neighbor upsampling convolutions, and a
// final convolution. Each convolution is instance-normalized. The style is baked into the weights, so
// one checkpoint applies one style; loading a different style checkpoint changes the look.
//
// The module structure and parameter names mirror the reference, so a converted checkpoint loads by
// name. Tensors flow in NHWC. The reference pads with reflection before each convolution to suppress
// border artifacts, which `NFKMLXResample.reflectPadded` reproduces. The reference operates on
// `0...255` pixel values, so `stylize` scales the bridged
// `0...1` image up before the forward and back down after.

/// A convolution with the reference's pad-then-convolve structure. The wrapped `Conv2d` is named
/// `conv2d` so a parameter reads `conv1.conv2d.weight`, matching the reference `ConvLayer`.
final class NFKStyleConvLayer: Module {

    @ModuleInfo(key: "conv2d") var conv2d: Conv2d
    private let reflectionPad: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        reflectionPad = kernelSize / 2
        _conv2d.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                      kernelSize: IntOrPair(kernelSize), stride: IntOrPair(stride), padding: 0)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv2d(NFKMLXResample.reflectPadded(x, reflectionPad))
    }
}

/// An upsampling convolution: nearest-neighbor upsample by `upsample`, then a `NFKStyleConvLayer`.
final class NFKStyleUpsampleConvLayer: Module {

    @ModuleInfo(key: "conv2d") var conv2d: Conv2d
    private let reflectionPad: Int
    private let upsample: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, upsample: Int) {
        reflectionPad = kernelSize / 2
        self.upsample = upsample
        _conv2d.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                      kernelSize: IntOrPair(kernelSize), stride: IntOrPair(1), padding: 0)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scaled = upsample > 1 ? NFKMLXResample.upsampleNearest(x, scale: upsample) : x
        return conv2d(NFKMLXResample.reflectPadded(scaled, reflectionPad))
    }
}

/// A residual block: two instance-normalized 3×3 convolutions, the second added back to the input.
final class NFKStyleResidualBlock: Module {

    let conv1: NFKStyleConvLayer
    let in1: InstanceNorm
    let conv2: NFKStyleConvLayer
    let in2: InstanceNorm

    init(channels: Int) {
        conv1 = NFKStyleConvLayer(inChannels: channels, outChannels: channels, kernelSize: 3, stride: 1)
        in1 = InstanceNorm(dimensions: channels, affine: true)
        conv2 = NFKStyleConvLayer(inChannels: channels, outChannels: channels, kernelSize: 3, stride: 1)
        in2 = InstanceNorm(dimensions: channels, affine: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = relu(in1(conv1(x)))
        return in2(conv2(out)) + x
    }
}

/// The image transformation network. `base: 32` matches the reference `TransformerNet`; downsampling
/// runs at `base`, `2·base`, then `4·base`, where the five residual blocks operate. Input `[H, W, 3]`
/// stylizes to `[H, W, 3]` at the same size.
final class NFKStyleTransferNet: Module {

    let conv1: NFKStyleConvLayer
    let in1: InstanceNorm
    let conv2: NFKStyleConvLayer
    let in2: InstanceNorm
    let conv3: NFKStyleConvLayer
    let in3: InstanceNorm

    // The reference names its five residual blocks res1…res5 (not an array), so they are individual
    // properties to keep the checkpoint keys matching without a converter remap.
    let res1: NFKStyleResidualBlock
    let res2: NFKStyleResidualBlock
    let res3: NFKStyleResidualBlock
    let res4: NFKStyleResidualBlock
    let res5: NFKStyleResidualBlock

    let deconv1: NFKStyleUpsampleConvLayer
    let in4: InstanceNorm
    let deconv2: NFKStyleUpsampleConvLayer
    let in5: InstanceNorm
    let deconv3: NFKStyleConvLayer

    init(base: Int = 32) {
        conv1 = NFKStyleConvLayer(inChannels: 3, outChannels: base, kernelSize: 9, stride: 1)
        in1 = InstanceNorm(dimensions: base, affine: true)
        conv2 = NFKStyleConvLayer(inChannels: base, outChannels: base * 2, kernelSize: 3, stride: 2)
        in2 = InstanceNorm(dimensions: base * 2, affine: true)
        conv3 = NFKStyleConvLayer(inChannels: base * 2, outChannels: base * 4, kernelSize: 3, stride: 2)
        in3 = InstanceNorm(dimensions: base * 4, affine: true)

        res1 = NFKStyleResidualBlock(channels: base * 4)
        res2 = NFKStyleResidualBlock(channels: base * 4)
        res3 = NFKStyleResidualBlock(channels: base * 4)
        res4 = NFKStyleResidualBlock(channels: base * 4)
        res5 = NFKStyleResidualBlock(channels: base * 4)

        deconv1 = NFKStyleUpsampleConvLayer(inChannels: base * 4, outChannels: base * 2, kernelSize: 3, upsample: 2)
        in4 = InstanceNorm(dimensions: base * 2, affine: true)
        deconv2 = NFKStyleUpsampleConvLayer(inChannels: base * 2, outChannels: base, kernelSize: 3, upsample: 2)
        in5 = InstanceNorm(dimensions: base, affine: true)
        deconv3 = NFKStyleConvLayer(inChannels: base, outChannels: 3, kernelSize: 9, stride: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = relu(in1(conv1(x)))
        y = relu(in2(conv2(y)))
        y = relu(in3(conv3(y)))
        y = res5(res4(res3(res2(res1(y)))))
        y = relu(in4(deconv1(y)))
        y = relu(in5(deconv2(y)))
        return deconv3(y)
    }

    /// Stylizes a bridged image `[H, W, 3]` (`0...1`). The network is trained on `0...255` pixels, so
    /// the input scales up before the forward and the output scales back down and clips to `0...1`.
    func stylize(_ image: MLXArray) -> MLXArray {
        let batched = (image * 255.0).reshaped([1, image.shape[0], image.shape[1], image.shape[2]])
        let output = clip(callAsFunction(batched) / 255.0, min: 0, max: 1)
        return output.reshaped([output.shape[1], output.shape[2], output.shape[3]])
    }
}

/// Holds the network for capture in the backend's `@Sendable` forward closure. Inference is serialized
/// per job and MLX evaluation is internally synchronized, so shared read access to the weights is safe.
private final class NFKStyleTransferHolder: @unchecked Sendable {
    let net: NFKStyleTransferNet
    init(_ net: NFKStyleTransferNet) { self.net = net }
}

/// Fast neural style transfer as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKStyleTransferNet` is the real image transformation network, not a stand-in. Random weights run
/// (proving the pipeline); a trained checkpoint bakes in a style. Load a **safetensors** checkpoint
/// whose parameter names match the reference `TransformerNet` (`conv1.conv2d.*`, `in1.*`,
/// `res1.conv1.conv2d.*`, `deconv3.conv2d.*`); the loader transposes 4-D PyTorch convolution weights
/// `[out, in, kH, kW]` to MLX's `[out, kH, kW, in]`.
@objc(NFKMLXStyleTransfer)
public final class NFKMLXStyleTransfer: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "fast-style-transfer"

    /// Builds a style-transfer backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true).
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKStyleTransferNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKStyleTransferHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in holder.net.stylize(image) }
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers fast style transfer (`fast-style-transfer`) with `NFKMLXModelRegistry`, delegating to
    /// `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Loads a safetensors checkpoint into `net`, transposing 4-D convolution weights from PyTorch's
    /// `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKStyleTransferNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (key, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
