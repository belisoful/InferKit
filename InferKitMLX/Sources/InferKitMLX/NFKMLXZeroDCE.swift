//
//  NFKMLXZeroDCE.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Zero-DCE brightens a low-light photo by estimating pixel-wise tone curves rather than predicting the
// output image directly. A seven-layer convolutional network (DCE-Net) with symmetric skip
// concatenations produces 24 curve-parameter maps — eight iterations of three channels — and the image
// is enhanced by applying the quadratic curve `x = x + r·(x² − x)` eight times.
//
// Module structure and parameter names mirror the reference DCE-Net (`e_conv1` … `e_conv7`), so a
// converted checkpoint loads by name. Tensors flow in NHWC. Input and output are `0...1`.

/// The DCE-Net curve estimator: seven 3×3 convolutions with U-style skip concatenations, producing 24
/// curve-parameter channels.
///
/// Fine-tuning works on this type directly: build one with ``NFKMLXZeroDCE/network(weightsURL:)``,
/// train it with ``NFKMLXZeroDCE/fineTune(_:photos:objective:optimizer:steps:clipGradientNorm:checkpoint:observer:)``,
/// then save it for ``NFKMLXZeroDCE/backend(weightsURL:)``.
public final class NFKMLXZeroDCENet: Module {
    @ModuleInfo(key: "e_conv1") var conv1: Conv2d
    @ModuleInfo(key: "e_conv2") var conv2: Conv2d
    @ModuleInfo(key: "e_conv3") var conv3: Conv2d
    @ModuleInfo(key: "e_conv4") var conv4: Conv2d
    @ModuleInfo(key: "e_conv5") var conv5: Conv2d
    @ModuleInfo(key: "e_conv6") var conv6: Conv2d
    @ModuleInfo(key: "e_conv7") var conv7: Conv2d

    private let iterations = 8

    public init(filters: Int = 32) {
        _conv1.wrappedValue = Conv2d(inputChannels: 3, outputChannels: filters, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv2d(inputChannels: filters, outputChannels: filters, kernelSize: 3, padding: 1)
        _conv3.wrappedValue = Conv2d(inputChannels: filters, outputChannels: filters, kernelSize: 3, padding: 1)
        _conv4.wrappedValue = Conv2d(inputChannels: filters, outputChannels: filters, kernelSize: 3, padding: 1)
        _conv5.wrappedValue = Conv2d(inputChannels: filters * 2, outputChannels: filters, kernelSize: 3, padding: 1)
        _conv6.wrappedValue = Conv2d(inputChannels: filters * 2, outputChannels: filters, kernelSize: 3, padding: 1)
        _conv7.wrappedValue = Conv2d(inputChannels: filters * 2, outputChannels: 3 * iterations, kernelSize: 3, padding: 1)
    }

    /// Predicts the curve-parameter maps for a batch `[N, H, W, 3]`, returning `[N, H, W, 24]`:
    /// eight iterations of three channels.
    public func curveMaps(_ batch: MLXArray) -> MLXArray {
        let x1 = relu(conv1(batch))
        let x2 = relu(conv2(x1))
        let x3 = relu(conv3(x2))
        let x4 = relu(conv4(x3))
        let x5 = relu(conv5(concatenated([x3, x4], axis: 3)))
        let x6 = relu(conv6(concatenated([x2, x5], axis: 3)))
        return tanh(conv7(concatenated([x1, x6], axis: 3)))
    }

    /// Applies the eight curve iterations `x = x + r·(x² − x)` to a batch `[N, H, W, 3]`.
    ///
    /// The result is unclipped. Training reads it directly, because clipping would flatten the
    /// gradient wherever a pixel saturates, which is where an over-brightening run most needs to be
    /// corrected.
    public func applyCurves(_ maps: MLXArray, to batch: MLXArray) -> MLXArray {
        var enhanced = batch
        for i in 0 ..< iterations {
            let r = maps[0..., 0..., 0..., (i * 3) ..< (i * 3 + 3)]
            enhanced = enhanced + r * (enhanced * enhanced - enhanced)
        }
        return enhanced
    }

    /// Enhances a bridged image `[H, W, 3]` (`0...1`), returning the brightened image `[H, W, 3]`.
    func enhance(_ image: MLXArray) -> MLXArray {
        let batched = image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]])
        let enhanced = clip(applyCurves(curveMaps(batched), to: batched), min: 0, max: 1)
        return enhanced.reshaped([enhanced.shape[1], enhanced.shape[2], enhanced.shape[3]])
    }
}

/// Holds the network for capture in the backend's `@Sendable` forward closure.
private final class NFKZeroDCEHolder: @unchecked Sendable {
    let net: NFKMLXZeroDCENet
    init(_ net: NFKMLXZeroDCENet) { self.net = net }
}

/// Zero-DCE low-light enhancement as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXZeroDCENet` is the real DCE-Net. Random weights run (proving the pipeline); a trained
/// checkpoint brightens correctly. Load a **safetensors** checkpoint whose parameter names match the
/// reference (`e_conv1.*` … `e_conv7.*`); the loader transposes 4-D convolution weights `[out, in, kH,
/// kW]` to MLX's `[out, kH, kW, in]`.
@objc(NFKMLXZeroDCE)
public final class NFKMLXZeroDCE: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "zero-dce"

    /// Builds a low-light-enhancement backend directly from optional local weights — no registry
    /// required. A nil `weightsURL` builds random weights (`isReady` is true).
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXZeroDCENet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKZeroDCEHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in holder.net.enhance(image) }
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

    /// Registers Zero-DCE (`zero-dce`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Builds the curve estimator itself, ready to fine-tune, rather than a backend wrapping it.
    ///
    /// A nil `weightsURL` leaves the network at its random initialization, which is training from
    /// scratch; passing the released checkpoint fine-tunes from it.
    public static func network(weightsURL: URL?) throws -> NFKMLXZeroDCENet {
        let net = NFKMLXZeroDCENet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Loads a safetensors checkpoint into `net`, transposing 4-D convolution weights from PyTorch's
    /// `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`.
    ///
    /// A checkpoint written by fine-tuning is already in the module's layout and skips the transpose,
    /// so a customized model and a converted one load through this same path.
    static func loadWeights(into net: NFKMLXZeroDCENet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { key, value in
            (key, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
