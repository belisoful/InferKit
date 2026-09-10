//
//  NFKMLXISNet.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// IS-Net is the highly-accurate dichotomous image segmentation network from the DIS project, by the
// authors of U²-Net. It keeps U²-Net's nested Residual U-block structure and changes three things:
// a stride-2 stem halves the plate before stage 1, the stage widths grow, and the six side maps stay
// separate (there is no fusion `outconv`). The first side map is the prediction. Tensors flow NHWC.
//
// The RSU blocks are U²-Net's, unchanged, so `NFKU2NetRSU` and `NFKMLXU2Net.remapReferenceKey(_:)`
// carry over and a raw release loads directly.

/// The IS-Net dichotomous-segmentation network. Input `[1, H, W, 3]` → six side maps in `0...1`,
/// each resampled to the input resolution.
final class NFKMLXISNetNet: Module {
    // The reference names the stem `conv_in`: a plain stride-2 convolution with no norm and no
    // activation, which is what separates it from `myrebnconv` in the ground-truth encoder.
    @ModuleInfo(key: "conv_in") var convIn: Conv2d

    let stage1, stage2, stage3, stage4, stage5, stage6: NFKU2NetRSU
    let stage5d, stage4d, stage3d, stage2d, stage1d: NFKU2NetRSU
    let side1, side2, side3, side4, side5, side6: Conv2d
    private let pool = MaxPool2d(kernelSize: 2, stride: 2)

    override init() {
        _convIn.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 64, kernelSize: 3, stride: 2, padding: 1)

        stage1 = NFKU2NetRSU(height: 7, inChannels: 64, midChannels: 32, outChannels: 64)
        stage2 = NFKU2NetRSU(height: 6, inChannels: 64, midChannels: 32, outChannels: 128)
        stage3 = NFKU2NetRSU(height: 5, inChannels: 128, midChannels: 64, outChannels: 256)
        stage4 = NFKU2NetRSU(height: 4, inChannels: 256, midChannels: 128, outChannels: 512)
        stage5 = NFKU2NetRSU(height: 4, inChannels: 512, midChannels: 256, outChannels: 512, dilated: true)
        stage6 = NFKU2NetRSU(height: 4, inChannels: 512, midChannels: 256, outChannels: 512, dilated: true)

        stage5d = NFKU2NetRSU(height: 4, inChannels: 1024, midChannels: 256, outChannels: 512, dilated: true)
        stage4d = NFKU2NetRSU(height: 4, inChannels: 1024, midChannels: 128, outChannels: 256)
        stage3d = NFKU2NetRSU(height: 5, inChannels: 512, midChannels: 64, outChannels: 128)
        stage2d = NFKU2NetRSU(height: 6, inChannels: 256, midChannels: 32, outChannels: 64)
        stage1d = NFKU2NetRSU(height: 7, inChannels: 128, midChannels: 16, outChannels: 64)

        side1 = Conv2d(inputChannels: 64, outputChannels: 1, kernelSize: 3, padding: 1)
        side2 = Conv2d(inputChannels: 64, outputChannels: 1, kernelSize: 3, padding: 1)
        side3 = Conv2d(inputChannels: 128, outputChannels: 1, kernelSize: 3, padding: 1)
        side4 = Conv2d(inputChannels: 256, outputChannels: 1, kernelSize: 3, padding: 1)
        side5 = Conv2d(inputChannels: 512, outputChannels: 1, kernelSize: 3, padding: 1)
        side6 = Conv2d(inputChannels: 512, outputChannels: 1, kernelSize: 3, padding: 1)
    }

    /// The six side maps, coarse-to-fine ordered `d1…d6` as the reference returns them.
    func sides(_ input: MLXArray) -> [MLXArray] {
        let (height, width) = (input.shape[1], input.shape[2])
        let hx1 = stage1(convIn(input))
        let hx2 = stage2(pool(hx1))
        let hx3 = stage3(pool(hx2))
        let hx4 = stage4(pool(hx3))
        let hx5 = stage5(pool(hx4))
        let hx6 = stage6(pool(hx5))

        let hx5d = stage5d(concatenated([Self.align(hx6, to: hx5), hx5], axis: 3))
        let hx4d = stage4d(concatenated([Self.align(hx5d, to: hx4), hx4], axis: 3))
        let hx3d = stage3d(concatenated([Self.align(hx4d, to: hx3), hx3], axis: 3))
        let hx2d = stage2d(concatenated([Self.align(hx3d, to: hx2), hx2], axis: 3))
        let hx1d = stage1d(concatenated([Self.align(hx2d, to: hx1), hx1], axis: 3))

        let logits = [side1(hx1d), side2(hx2d), side3(hx3d), side4(hx4d), side5(hx5d), side6(hx6)]
        return logits.map { sigmoid(NFKMLXResample.resizeBilinear($0, height: height, width: width)) }
    }

    /// The prediction: the finest side map, `[1, H, W, 1]` in `0...1`.
    func saliency(_ input: MLXArray) -> MLXArray { sides(input)[0] }

    private static func align(_ x: MLXArray, to reference: MLXArray) -> MLXArray {
        NFKMLXResample.resizeBilinear(x, height: reference.shape[1], width: reference.shape[2])
    }
}

/// IS-Net background removal as an InferKit backend, and its registration for the Objective-C path.
@objc(NFKMLXISNet)
public final class NFKMLXISNet: NSObject {

    @objc public static let modelName = "isnet"

    /// The square resolution the reference resamples every plate to before the forward.
    @objc public static let inputSize = 1024

    static func makeNet() -> NFKMLXISNetNet { NFKMLXISNetNet() }

    /// Builds an IS-Net background-removal backend directly from optional local weights — no registry
    /// required. A nil `weightsURL` builds random weights (`isReady` is true).
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXISNetNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXISNetHolder(net)
        var configuration = NFKMattingConfiguration()
        configuration.emitsMatte = true
        configuration.plateChannels = 3
        return NFKMLXMattingBackend(identifier: modelName, isReady: true, configuration: configuration) { plate, _ in
            // The reference resizes to 1024x1024, scales to [0,1], normalizes with mean 0.5 and unit
            // standard deviation, then min-max stretches the returned map. The resize is
            // CoreGraphics-bilinear here rather than the reference's `F.upsample`, so a consumer alpha is
            // a documented approximation; the network itself is at reference parity on identical pixels.
            let (height, width) = (plate.shape[0], plate.shape[1])
            let batched = plate.reshaped([1, height, width, 3])
            let resized = NFKMLXResample.resizeBilinear(batched, height: inputSize, width: inputSize)
            let prediction = holder.net.saliency(resized - 0.5)
            var alpha = NFKMLXResample.resizeBilinear(prediction, height: height, width: width)
            let (low, high) = (alpha.min(), alpha.max())
            alpha = ((alpha - low) / maximum(high - low, MLXArray(Float(1e-8)))).reshaped([height, width, 1])
            return concatenated([plate, alpha], axis: 2)               // [H, W, 4]: straight foreground + matte
        }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
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

    /// Registers `isnet` with `NFKMLXModelRegistry`, delegating to ``backend(weightsURL:)``.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Loads a checkpoint (safetensors, or a raw `.pth` through the native reader), transposing 4-D
    /// convolution weights to MLX's layout. The RSU blocks are U²-Net's, so the reference `rebnconvN`
    /// names translate through ``NFKMLXU2Net/remapReferenceKey(_:)``.
    static func loadWeights(into net: NFKMLXISNetNet, from url: URL, remap: (String) -> String = { $0 }) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { key, value in
            (remap(NFKMLXU2Net.remapReferenceKey(key)),
             checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
        net.train(false)                                       // BatchNorm running statistics
    }
}

private final class NFKMLXISNetHolder: @unchecked Sendable {
    let net: NFKMLXISNetNet
    init(_ net: NFKMLXISNetNet) { self.net = net }
}
