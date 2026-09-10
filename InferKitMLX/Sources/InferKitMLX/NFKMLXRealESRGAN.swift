//
//  NFKMLXRealESRGAN.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Real-ESRGAN's generator is RRDBNet: a norm-free convolutional network (no attention, no VAE), so it
// runs as a single forward through `NFKMLXModuleBackend`. The module structure and parameter names
// mirror the reference PyTorch model (BasicSR `RRDBNet`), so a converted checkpoint loads by name.
// Tensors flow in NHWC (MLX's channels-last convolution layout); concatenation is over the channel
// axis (3).

/// A residual dense block: five 3×3 convolutions with dense (concatenated) connections, scaled and
/// added back to the input.
final class NFKRealESRGANDenseBlock: Module {

    let conv1: Conv2d
    let conv2: Conv2d
    let conv3: Conv2d
    let conv4: Conv2d
    let conv5: Conv2d

    init(features: Int, growth: Int) {
        conv1 = Conv2d(inputChannels: features, outputChannels: growth, kernelSize: 3, padding: 1)
        conv2 = Conv2d(inputChannels: features + growth, outputChannels: growth, kernelSize: 3, padding: 1)
        conv3 = Conv2d(inputChannels: features + 2 * growth, outputChannels: growth, kernelSize: 3, padding: 1)
        conv4 = Conv2d(inputChannels: features + 3 * growth, outputChannels: growth, kernelSize: 3, padding: 1)
        conv5 = Conv2d(inputChannels: features + 4 * growth, outputChannels: features, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x1 = leakyRelu(conv1(x), negativeSlope: 0.2)
        let x2 = leakyRelu(conv2(concatenated([x, x1], axis: 3)), negativeSlope: 0.2)
        let x3 = leakyRelu(conv3(concatenated([x, x1, x2], axis: 3)), negativeSlope: 0.2)
        let x4 = leakyRelu(conv4(concatenated([x, x1, x2, x3], axis: 3)), negativeSlope: 0.2)
        let x5 = conv5(concatenated([x, x1, x2, x3, x4], axis: 3))
        return x5 * 0.2 + x
    }
}

/// A residual-in-residual dense block: three dense blocks in series, scaled and added back.
final class NFKRealESRGANRRDB: Module {

    let rdb1: NFKRealESRGANDenseBlock
    let rdb2: NFKRealESRGANDenseBlock
    let rdb3: NFKRealESRGANDenseBlock

    init(features: Int, growth: Int) {
        rdb1 = NFKRealESRGANDenseBlock(features: features, growth: growth)
        rdb2 = NFKRealESRGANDenseBlock(features: features, growth: growth)
        rdb3 = NFKRealESRGANDenseBlock(features: features, growth: growth)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = rdb3(rdb2(rdb1(x)))
        return out * 0.2 + x
    }
}

/// The RRDBNet generator (×4). `features: 64`, `blocks: 23`, `growth: 32` matches `RealESRGAN_x4plus`;
/// `blocks: 6` matches the anime model. Input `[H, W, 3]` in `0...1` upscales to `[4H, 4W, 3]`.
final class NFKRealESRGANNet: Module {

    @ModuleInfo(key: "conv_first") var convFirst: Conv2d
    let body: [NFKRealESRGANRRDB]
    @ModuleInfo(key: "conv_body") var convBody: Conv2d
    @ModuleInfo(key: "conv_up1") var convUp1: Conv2d
    @ModuleInfo(key: "conv_up2") var convUp2: Conv2d
    @ModuleInfo(key: "conv_hr") var convHR: Conv2d
    @ModuleInfo(key: "conv_last") var convLast: Conv2d

    // The body always upscales ×4; the net scale is that divided by the input pixel-unshuffle factor
    // (scale 4 → unshuffle 1, scale 2 → unshuffle 2, scale 1 → unshuffle 4). `conv_first` reads the
    // unshuffled channels, so a real x2plus / x1 checkpoint's `conv_first` shape matches.
    private let unshuffle: Int

    init(inChannels: Int = 3, outChannels: Int = 3, features: Int = 64, blocks: Int = 23, growth: Int = 32, scale: Int = 4) {
        unshuffle = scale == 4 ? 1 : (scale == 2 ? 2 : 4)
        _convFirst.wrappedValue = Conv2d(inputChannels: inChannels * unshuffle * unshuffle, outputChannels: features, kernelSize: 3, padding: 1)
        body = (0 ..< blocks).map { _ in NFKRealESRGANRRDB(features: features, growth: growth) }
        _convBody.wrappedValue = Conv2d(inputChannels: features, outputChannels: features, kernelSize: 3, padding: 1)
        _convUp1.wrappedValue = Conv2d(inputChannels: features, outputChannels: features, kernelSize: 3, padding: 1)
        _convUp2.wrappedValue = Conv2d(inputChannels: features, outputChannels: features, kernelSize: 3, padding: 1)
        _convHR.wrappedValue = Conv2d(inputChannels: features, outputChannels: features, kernelSize: 3, padding: 1)
        _convLast.wrappedValue = Conv2d(inputChannels: features, outputChannels: outChannels, kernelSize: 3, padding: 1)
    }

    /// Folds a `scale`×`scale` spatial block into channels (channel order `[C, i, j]`), matching
    /// PyTorch `pixel_unshuffle`.
    static func pixelUnshuffle(_ x: MLXArray, scale r: Int) -> MLXArray {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        return x.reshaped([n, h / r, r, w / r, r, c])
            .transposed(0, 1, 3, 5, 2, 4)
            .reshaped([n, h / r, w / r, c * r * r])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let input = unshuffle > 1 ? Self.pixelUnshuffle(x, scale: unshuffle) : x
        let first = convFirst(input)
        var feature = first
        for block in body {
            feature = block(feature)
        }
        feature = first + convBody(feature)
        feature = leakyRelu(convUp1(Self.upsampleNearest2x(feature)), negativeSlope: 0.2)
        feature = leakyRelu(convUp2(Self.upsampleNearest2x(feature)), negativeSlope: 0.2)
        return convLast(leakyRelu(convHR(feature), negativeSlope: 0.2))
    }

    /// Upscales an `[N, H, W, C]` tensor 2× by nearest neighbor.
    static func upsampleNearest2x(_ x: MLXArray) -> MLXArray {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let expanded = x.reshaped([n, h, 1, w, 1, c])
        return broadcast(expanded, to: [n, h, 2, w, 2, c]).reshaped([n, h * 2, w * 2, c])
    }

    /// Upscales a bridged image `[H, W, 3]` (`0...1`) to `[4H, 4W, 3]` (`0...1`).
    func upscale(_ image: MLXArray) -> MLXArray {
        let batched = image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]])
        let output = clip(callAsFunction(batched), min: 0, max: 1)
        return output.reshaped([output.shape[1], output.shape[2], output.shape[3]])
    }
}

/// Holds the network for capture in the backend's `@Sendable` forward closure. Inference is serialized
/// per job and MLX evaluation is internally synchronized, so shared read access to the weights is safe.
private final class NFKRealESRGANHolder: @unchecked Sendable {
    let net: NFKRealESRGANNet
    init(_ net: NFKRealESRGANNet) { self.net = net }
}

/// Real-ESRGAN ×4 upscaling as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKRealESRGANNet` is the real generator, not a stand-in. Random weights run (proving the pipeline);
/// a trained checkpoint makes the output useful. Load a **safetensors** checkpoint whose parameter
/// names match the reference `RRDBNet` (`conv_first.*`, `body.N.rdbM.convK.*`, `conv_last.*`); the
/// loader transposes 4-D PyTorch convolution weights `[out, in, kH, kW]` to MLX's `[out, kH, kW, in]`.
/// A compact VGG-style generator (`SRVGGNetCompact`), which is what Real-ESRGAN's later releases run
/// instead of RRDBNet: a first convolution, a run of convolutions each followed by a per-channel PReLU,
/// a last convolution to `3 · scale²` channels, a pixel shuffle, and a nearest-upsampled residual so
/// the network learns the difference rather than the picture.
///
/// The reference holds the whole stack in one `nn.ModuleList`, so the convolutions and the activations
/// share one index sequence: `body.0` is the first convolution, `body.1` its PReLU, and the last entry
/// is the convolution the shuffle reads.
///
/// - Since: InferKit 0.4.0
final class NFKRealESRGANCompactNet: Module {
    @ModuleInfo(key: "body") var body: [Module]
    private let scale: Int

    init(features: Int, convolutions: Int, scale: Int) {
        self.scale = scale
        var stack: [Module] = [
            Conv2d(inputChannels: 3, outputChannels: features, kernelSize: 3, padding: 1),
            NFKRealESRGANPReLU(features: features),
        ]
        for _ in 0 ..< convolutions {
            stack.append(Conv2d(inputChannels: features, outputChannels: features, kernelSize: 3, padding: 1))
            stack.append(NFKRealESRGANPReLU(features: features))
        }
        stack.append(Conv2d(inputChannels: features, outputChannels: 3 * scale * scale,
                            kernelSize: 3, padding: 1))
        _body.wrappedValue = stack
    }

    /// - Parameter image: `[1, H, W, 3]` in `0...1`.
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        var out = image
        for entry in body {
            switch entry {
            case let conv as Conv2d: out = conv(out)
            case let activation as NFKRealESRGANPReLU: out = activation(out)
            default: break
            }
        }
        let shuffled = NFKMLXPixelShuffle.apply(out, factor: scale)
        return shuffled + NFKMLXResample.upsampleNearest(image, scale: scale)
    }

    /// Upscales a bridged `[H, W, 3]` image in `0...1`.
    func upscale(_ image: MLXArray) -> MLXArray {
        let batched = image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]])
        return clip(self(batched)[0], min: MLXArray(Float(0)), max: MLXArray(Float(1)))
    }
}

/// A per-channel PReLU (`nn.PReLU(num_parameters:)`), whose single parameter the reference names
/// `weight`.
final class NFKRealESRGANPReLU: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    init(features: Int) { _weight.wrappedValue = MLXArray.zeros([features]) + Float(0.25) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { maximum(x, 0) + weight * minimum(x, 0) }
}

/// The Real-ESRGAN model variant, for the Objective-C factory.
@objc(NFKMLXRealESRGANVariant)
public enum NFKMLXRealESRGANVariant: Int {
    case x4
    case anime
    case x2
    /// `realesr-general-x4v3`, the compact general-purpose ×4 release (32 body convolutions).
    ///
    /// - Since: InferKit 0.4.0
    case generalX4V3
    /// `realesr-animevideov3`, the compact anime-video ×4 release (16 body convolutions).
    ///
    /// - Since: InferKit 0.4.0
    case animeVideoV3
}

@objc(NFKMLXRealESRGAN)
public final class NFKMLXRealESRGAN: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "real-esrgan-x4"
    @objc public static let animeModelName = "real-esrgan-x4-anime"
    @objc public static let x2ModelName = "real-esrgan-x2"

    @objc public static let generalV3ModelName = "real-esrgan-general-x4v3"
    @objc public static let animeVideoV3ModelName = "real-esrgan-anime-video-x4v3"

    private static func specs(for variant: NFKMLXRealESRGANVariant) -> (name: String, blocks: Int, scale: Int) {
        switch variant {
        case .x4: return (modelName, 23, 4)
        case .anime: return (animeModelName, 6, 4)
        case .x2: return (x2ModelName, 23, 2)
        case .generalX4V3: return (generalV3ModelName, 32, 4)
        case .animeVideoV3: return (animeVideoV3ModelName, 16, 4)
        }
    }

    /// The body length a variant's release carries: RRDB blocks for the earlier releases, body
    /// convolutions for the compact ones.
    static func bodyConvolutions(for variant: NFKMLXRealESRGANVariant) -> Int {
        specs(for: variant).blocks
    }

    /// Whether a variant runs the later compact generator rather than RRDBNet.
    static func isCompact(_ variant: NFKMLXRealESRGANVariant) -> Bool {
        variant == .generalX4V3 || variant == .animeVideoV3
    }

    /// Builds a Real-ESRGAN backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true).
    /// Run inference off the render thread.
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXRealESRGANVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let spec = specs(for: variant)
        if isCompact(variant) {
            // `blocks` is the body's convolution count for this generator, not an RRDB count.
            let net = NFKRealESRGANCompactNet(features: 64, convolutions: spec.blocks, scale: spec.scale)
            if let weightsURL {
                try loadCompactWeights(into: net, from: weightsURL)
            }
            let holder = NFKRealESRGANCompactHolder(net)
            return NFKMLXModuleBackend(identifier: spec.name, isReady: true) { image in holder.net.upscale(image) }
        }
        let net = NFKRealESRGANNet(blocks: spec.blocks, scale: spec.scale)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKRealESRGANHolder(net)
        return NFKMLXModuleBackend(identifier: spec.name, isReady: true) { image in holder.net.upscale(image) }
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXRealESRGANVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXRealESRGANVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers Real-ESRGAN ×4 (`real-esrgan-x4`, 23 blocks), the lighter anime model
    /// (`real-esrgan-x4-anime`, 6 blocks), and ×2 (`real-esrgan-x2`, pixel-unshuffle front-end) with
    /// `NFKMLXModelRegistry`, each delegating to `backend(variant:weightsURL:)`.
    @objc public static func register() {
        for variant in [NFKMLXRealESRGANVariant.x4, .anime, .x2, .generalX4V3, .animeVideoV3] {
            NFKMLXModelRegistry.register(name: specs(for: variant).name) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL)
            }
        }
    }

    /// Loads a safetensors checkpoint into `net`, transposing 4-D convolution weights from PyTorch's
    /// `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKRealESRGANNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (key, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Loads a compact release. Its `params` subtree carries the body directly, and the PReLU slopes are
    /// one-dimensional, so only the convolutions transpose.
    ///
    /// - Since: InferKit 0.4.0
    static func loadCompactWeights(into net: NFKRealESRGANCompactNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.map { key, value in
            (key, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

private final class NFKRealESRGANCompactHolder: @unchecked Sendable {
    let net: NFKRealESRGANCompactNet
    init(_ net: NFKRealESRGANCompactNet) { self.net = net }
}
