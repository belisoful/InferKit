//
//  NFKMLXU2Net.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// U²-Net is a nested U-structure of Residual U-blocks (RSU) for salient-object detection / background
// removal. It runs as a single forward through `NFKMLXMattingBackend`: a plate in, the straight
// foreground plus the saliency map as alpha out (RGBA), so it drops straight into the keyer/matte
// pipeline. Tensors flow NHWC.
//
// Stage and side names match the reference (`stage1…stage6`, `stage1d…stage5d`, `side1…side6`,
// `outconv`); the RSU-internal convolutions are held in arrays, so mapping a real checkpoint's
// `rebnconvN` names is a `remap` (validation-sweep task).

/// REBNCONV: 3×3 convolution → batch norm → ReLU, with optional dilation (padding tracks it).
final class NFKU2NetConv: Module {
    // Names match the reference REBNCONV (`conv_s1`, `bn_s1`) so a converted checkpoint's keys land on
    // these layers; the property names stay `conv`/`bn` for readability.
    @ModuleInfo(key: "conv_s1") var conv: Conv2d
    @ModuleInfo(key: "bn_s1") var bn: BatchNorm

    init(_ inChannels: Int, _ outChannels: Int, dilation: Int = 1) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3,
                                    padding: IntOrPair(dilation), dilation: IntOrPair(dilation))
        _bn.wrappedValue = BatchNorm(featureCount: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { relu(bn(conv(x))) }
}

/// A Residual U-block. `dilated` swaps pooling/upsampling for dilated convolutions (the RSU4F variant).
final class NFKU2NetRSU: Module {
    let rebnconvin: NFKU2NetConv
    @ModuleInfo(key: "enc") var encoders: [NFKU2NetConv]
    @ModuleInfo(key: "dec") var decoders: [NFKU2NetConv]

    private let height: Int
    private let dilated: Bool
    private let pool = MaxPool2d(kernelSize: 2, stride: 2)

    init(height: Int, inChannels: Int, midChannels: Int, outChannels: Int, dilated: Bool = false) {
        self.height = height
        self.dilated = dilated
        rebnconvin = NFKU2NetConv(inChannels, outChannels)

        var encoderConvs = [NFKU2NetConv]()
        for i in 0 ..< height {
            let inC = i == 0 ? outChannels : midChannels
            let dilation = dilated ? (1 << i) : (i == height - 1 ? 2 : 1)
            encoderConvs.append(NFKU2NetConv(inC, midChannels, dilation: dilation))
        }
        _encoders.wrappedValue = encoderConvs

        var decoderConvs = [NFKU2NetConv]()
        for j in 0 ..< (height - 1) {
            let outC = j == height - 2 ? outChannels : midChannels
            let dilation = dilated ? (1 << (height - 2 - j)) : 1
            decoderConvs.append(NFKU2NetConv(midChannels * 2, outC, dilation: dilation))
        }
        _decoders.wrappedValue = decoderConvs
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hxin = rebnconvin(x)
        var encoded = [MLXArray]()
        var hx = hxin
        for i in 0 ..< (height - 1) {
            let e = encoders[i](hx)
            encoded.append(e)
            hx = dilated ? e : (i < height - 2 ? pool(e) : e)
        }
        var d = encoders[height - 1](encoded[height - 2])       // bottleneck
        for j in 0 ..< (height - 1) {
            let skip = encoded[height - 2 - j]
            let aligned = dilated ? d : NFKMLXResample.resizeBilinear(d, height: skip.shape[1], width: skip.shape[2])
            d = decoders[j](concatenated([aligned, skip], axis: 3))
        }
        return d + hxin
    }
}

/// The U²-Net saliency network. `light` builds the small `u2netp` (all mid 16, out 64); otherwise the
/// full `u2net`. Input `[1, H, W, 3]` → a saliency map `[1, H, W, 1]` in `0...1`.
final class NFKMLXU2NetNet: Module {
    let stage1, stage2, stage3, stage4, stage5, stage6: NFKU2NetRSU
    let stage5d, stage4d, stage3d, stage2d, stage1d: NFKU2NetRSU
    let side1, side2, side3, side4, side5, side6: Conv2d
    let outconv: Conv2d
    private let pool = MaxPool2d(kernelSize: 2, stride: 2)

    init(light: Bool) {
        let mids = light ? [16, 16, 16, 16, 16, 16] : [32, 32, 64, 128, 256, 256]
        let outs = light ? [64, 64, 64, 64, 64, 64] : [64, 128, 256, 512, 512, 512]
        let dmids = light ? [16, 16, 16, 16, 16] : [256, 128, 64, 32, 16]
        let douts = light ? [64, 64, 64, 64, 64] : [512, 256, 128, 64, 64]

        stage1 = NFKU2NetRSU(height: 7, inChannels: 3, midChannels: mids[0], outChannels: outs[0])
        stage2 = NFKU2NetRSU(height: 6, inChannels: outs[0], midChannels: mids[1], outChannels: outs[1])
        stage3 = NFKU2NetRSU(height: 5, inChannels: outs[1], midChannels: mids[2], outChannels: outs[2])
        stage4 = NFKU2NetRSU(height: 4, inChannels: outs[2], midChannels: mids[3], outChannels: outs[3])
        stage5 = NFKU2NetRSU(height: 4, inChannels: outs[3], midChannels: mids[4], outChannels: outs[4], dilated: true)
        stage6 = NFKU2NetRSU(height: 4, inChannels: outs[4], midChannels: mids[5], outChannels: outs[5], dilated: true)

        stage5d = NFKU2NetRSU(height: 4, inChannels: outs[5] + outs[4], midChannels: dmids[0], outChannels: douts[0], dilated: true)
        stage4d = NFKU2NetRSU(height: 4, inChannels: douts[0] + outs[3], midChannels: dmids[1], outChannels: douts[1])
        stage3d = NFKU2NetRSU(height: 5, inChannels: douts[1] + outs[2], midChannels: dmids[2], outChannels: douts[2])
        stage2d = NFKU2NetRSU(height: 6, inChannels: douts[2] + outs[1], midChannels: dmids[3], outChannels: douts[3])
        stage1d = NFKU2NetRSU(height: 7, inChannels: douts[3] + outs[0], midChannels: dmids[4], outChannels: douts[4])

        side1 = Conv2d(inputChannels: douts[4], outputChannels: 1, kernelSize: 3, padding: 1)
        side2 = Conv2d(inputChannels: douts[3], outputChannels: 1, kernelSize: 3, padding: 1)
        side3 = Conv2d(inputChannels: douts[2], outputChannels: 1, kernelSize: 3, padding: 1)
        side4 = Conv2d(inputChannels: douts[1], outputChannels: 1, kernelSize: 3, padding: 1)
        side5 = Conv2d(inputChannels: douts[0], outputChannels: 1, kernelSize: 3, padding: 1)
        side6 = Conv2d(inputChannels: outs[5], outputChannels: 1, kernelSize: 3, padding: 1)
        outconv = Conv2d(inputChannels: 6, outputChannels: 1, kernelSize: 1)
    }

    func saliency(_ input: MLXArray) -> MLXArray {
        let (height, width) = (input.shape[1], input.shape[2])
        let hx1 = stage1(input)
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

        let d1 = side1(hx1d)
        let d2 = NFKMLXResample.resizeBilinear(side2(hx2d), height: height, width: width)
        let d3 = NFKMLXResample.resizeBilinear(side3(hx3d), height: height, width: width)
        let d4 = NFKMLXResample.resizeBilinear(side4(hx4d), height: height, width: width)
        let d5 = NFKMLXResample.resizeBilinear(side5(hx5d), height: height, width: width)
        let d6 = NFKMLXResample.resizeBilinear(side6(hx6), height: height, width: width)
        return sigmoid(outconv(concatenated([d1, d2, d3, d4, d5, d6], axis: 3)))
    }

    private static func align(_ x: MLXArray, to reference: MLXArray) -> MLXArray {
        NFKMLXResample.resizeBilinear(x, height: reference.shape[1], width: reference.shape[2])
    }
}

/// U²-Net background removal as an InferKit backend, and its registration for the Objective-C path.
/// The U²-Net variant, for the Objective-C factory.
@objc(NFKMLXU2NetVariant)
public enum NFKMLXU2NetVariant: Int {
    case full       // u2net
    case light      // u2netp
}

@objc(NFKMLXU2Net)
public final class NFKMLXU2Net: NSObject {

    @objc public static let modelName = "u2net"
    @objc public static let lightModelName = "u2netp"

    static func makeNet(light: Bool = false) -> NFKMLXU2NetNet { NFKMLXU2NetNet(light: light) }

    private static func specs(for variant: NFKMLXU2NetVariant) -> (name: String, light: Bool) {
        switch variant {
        case .full: return (modelName, false)
        case .light: return (lightModelName, true)
        }
    }

    /// Builds a U²-Net background-removal backend directly from optional local weights — no registry
    /// required. A nil `weightsURL` builds random weights (`isReady` is true).
    /// Run inference off the render thread.
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXU2NetVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let spec = specs(for: variant)
        let net = NFKMLXU2NetNet(light: spec.light)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXU2NetHolder(net)
        var configuration = NFKMattingConfiguration()
        configuration.emitsMatte = true
        configuration.plateChannels = 3
        return NFKMLXMattingBackend(identifier: spec.name, isReady: true, configuration: configuration) { plate, _ in
            // U²-Net's reference `ToTensorLab` scales the plate to [0,1] by its own maximum, then
            // normalizes with the ImageNet per-channel mean/std. Feeding the raw plate produces noise.
            let mean = MLXArray([Float(0.485), 0.456, 0.406])
            let std = MLXArray([Float(0.229), 0.224, 0.225])
            let scaled = plate / plate.max()
            let normalized = (scaled - mean) / std
            let batched = normalized.reshaped([1, plate.shape[0], plate.shape[1], 3])
            let saliency = holder.net.saliency(batched).reshaped([plate.shape[0], plate.shape[1], 1])
            return concatenated([plate, saliency], axis: 2)    // [H, W, 4]: straight (original) foreground + matte
        }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXU2NetVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXU2NetVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers the full `u2net` and the light `u2netp` with `NFKMLXModelRegistry`, each delegating to
    /// `backend(variant:weightsURL:)`.
    @objc public static func register() {
        for variant in [NFKMLXU2NetVariant.full, .light] {
            NFKMLXModelRegistry.register(name: specs(for: variant).name) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL)
            }
        }
    }

    /// Loads a safetensors checkpoint, transposing 4-D convolution weights to MLX's layout. `remap`
    /// maps the reference RSU `rebnconvN` names to the module's `enc`/`dec` array keys (sweep task).
    static func loadWeights(into net: NFKMLXU2NetNet, from url: URL, remap: (String) -> String = { $0 }) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remap(key), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

private final class NFKMLXU2NetHolder: @unchecked Sendable {
    let net: NFKMLXU2NetNet
    init(_ net: NFKMLXU2NetNet) { self.net = net }
}
