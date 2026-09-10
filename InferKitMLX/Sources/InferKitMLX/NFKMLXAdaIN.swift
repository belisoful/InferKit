//
//  NFKMLXAdaIN.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// AdaIN is arbitrary style transfer: one pair of networks handles any style image, where the shipped
// `NFKMLXStyleTransfer` (TransformerNet) carries one style per checkpoint. A normalized VGG-19 encodes
// both images through relu4_1, adaptive instance normalization moves the content features onto the
// style's per-channel mean and standard deviation, and a mirrored decoder inverts the result.
//
// Both networks are flat `nn.Sequential` stacks in the reference, so they are held as `[Module]`
// arrays whose indices are the reference's own; the parameter-free slots (reflection padding, ReLU,
// pooling, upsampling) keep their index with marker modules. The encoder and the decoder ship as two
// separate files, which is why the loader takes each one on its own. Tensors flow NHWC.

/// The layer kinds a VGG/decoder `nn.Sequential` holds beside its convolutions. Each occupies an
/// index in the reference stack, so each needs a module to keep the numbering aligned.
enum NFKAdaINLayer {
    /// `nn.ReflectionPad2d((1, 1, 1, 1))`.
    final class Pad: Module {}
    /// `nn.ReLU()`.
    final class Activation: Module {}
    /// `nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)`.
    final class Pool: Module {}
    /// `nn.Upsample(scale_factor=2, mode='nearest')`.
    final class Upsample: Module {}

    static func forward(_ stack: [Module], _ x: MLXArray, through limit: Int? = nil) -> MLXArray {
        var out = x
        for (index, entry) in stack.enumerated() {
            if let limit, index > limit { break }
            switch entry {
            case let conv as Conv2d: out = conv(out)
            case is Pad: out = NFKMLXResample.reflectPadded(out, 1)
            case is Pool: out = maxPool(out)
            case is Upsample: out = NFKMLXResample.upsampleNearest(out, scale: 2)
            default: out = relu(out)
            }
        }
        return out
    }

    /// `ceil_mode=True` pooling. An odd extent keeps its last row or column as a partial window rather
    /// than dropping it, which MLX's `MaxPool2d` does not do on its own.
    private static func maxPool(_ x: MLXArray) -> MLXArray {
        var input = x
        let (height, width) = (x.shape[1], x.shape[2])
        if height % 2 == 1 || width % 2 == 1 {
            input = MLX.padded(input,
                               widths: [IntOrPair(0), IntOrPair((0, height % 2)),
                                        IntOrPair((0, width % 2)), IntOrPair(0)],
                               mode: .edge)
        }
        return MaxPool2d(kernelSize: 2, stride: 2)(input)
    }
}

/// The normalized VGG-19 the reference encodes with, truncated after relu4_1 (index 30). The first
/// 1×1 convolution carries the input normalization, which is why the plate reaches it unnormalized.
final class NFKMLXAdaINEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    /// The reference slices the released VGG to `[:31]`, so relu4_1 is the last layer that runs.
    static let outputIndex = 30

    override init() {
        typealias L = NFKAdaINLayer
        func block(_ inChannels: Int, _ outChannels: Int) -> [Module] {
            [L.Pad(), Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3), L.Activation()]
        }
        var stack: [Module] = [Conv2d(inputChannels: 3, outputChannels: 3, kernelSize: 1)]
        stack += block(3, 64) + block(64, 64) + [L.Pool()]
        stack += block(64, 128) + block(128, 128) + [L.Pool()]
        stack += block(128, 256) + block(256, 256) + block(256, 256) + block(256, 256) + [L.Pool()]
        stack += block(256, 512)
        _layers.wrappedValue = stack
    }

    /// The relu4_1 features, `[1, H/8, W/8, 512]`.
    func features(_ x: MLXArray) -> MLXArray {
        NFKAdaINLayer.forward(layers, x, through: Self.outputIndex)
    }
}

/// The mirrored decoder that inverts relu4_1 features back to an image.
final class NFKMLXAdaINDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    override init() {
        typealias L = NFKAdaINLayer
        func block(_ inChannels: Int, _ outChannels: Int, activates: Bool = true) -> [Module] {
            var stack: [Module] = [L.Pad(), Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3)]
            if activates { stack.append(L.Activation()) }
            return stack
        }
        var stack = block(512, 256) + [L.Upsample()]
        stack += block(256, 256) + block(256, 256) + block(256, 256) + block(256, 128) + [L.Upsample()]
        stack += block(128, 128) + block(128, 64) + [L.Upsample()]
        stack += block(64, 64) + block(64, 3, activates: false)
        _layers.wrappedValue = stack
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { NFKAdaINLayer.forward(layers, x) }
}

/// The encoder, the decoder, and the normalization between them.
final class NFKMLXAdaINNet: Module {
    let encoder: NFKMLXAdaINEncoder
    let decoder: NFKMLXAdaINDecoder

    override init() {
        encoder = NFKMLXAdaINEncoder()
        decoder = NFKMLXAdaINDecoder()
    }

    /// Per-channel mean and standard deviation over the spatial extent.
    ///
    /// The reference reads `Tensor.var`, whose default correction is 1, so the estimate is unbiased.
    /// A population variance here shifts every feature by a factor of `sqrt(n / (n - 1))`.
    static func statistics(_ x: MLXArray) -> (mean: MLXArray, deviation: MLXArray) {
        let flattened = x.reshaped([x.shape[0], -1, x.shape[3]])
        let mean = flattened.mean(axis: 1, keepDims: true)
        let deviation = sqrt(flattened.variance(axis: 1, keepDims: true, ddof: 1) + 1e-5)
        return (mean.reshaped([x.shape[0], 1, 1, x.shape[3]]),
                deviation.reshaped([x.shape[0], 1, 1, x.shape[3]]))
    }

    /// Adaptive instance normalization: the content features standardized, then rescaled onto the
    /// style's own per-channel statistics.
    static func adaptiveInstanceNormalization(content: MLXArray, style: MLXArray) -> MLXArray {
        let contentStatistics = statistics(content)
        let styleStatistics = statistics(style)
        let normalized = (content - contentStatistics.mean) / contentStatistics.deviation
        return normalized * styleStatistics.deviation + styleStatistics.mean
    }

    /// Stylizes `content` with `style`, both `[1, H, W, 3]` in `0...1`. `strength` blends the
    /// normalized features against the content's own, so 0 returns the content reconstruction and 1
    /// the full transfer.
    func stylize(content: MLXArray, style: MLXArray, strength: Float = 1.0) -> MLXArray {
        let contentFeatures = encoder.features(content)
        let styleFeatures = encoder.features(style)
        let transferred = Self.adaptiveInstanceNormalization(content: contentFeatures, style: styleFeatures)
        let blended = transferred * strength + contentFeatures * (1 - strength)
        return decoder(blended)
    }
}

/// Arbitrary style transfer as an InferKit backend, and its registration for the Objective-C path.
///
/// The request carries the content image under `NFKInputImage` and the style image under
/// `NFKInputControl`; the core's `NFKParameterStrength` sets the blend.
@objc(NFKMLXAdaIN)
public final class NFKMLXAdaIN: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "adain"

    static func makeNet() -> NFKMLXAdaINNet { NFKMLXAdaINNet() }

    /// Builds an arbitrary-style-transfer backend from optional local weights — no registry required.
    /// The reference publishes the encoder and the decoder as two files, so both are named. A nil URL
    /// leaves that network at its random initialization (`isReady` is true).
    /// Run inference off the render thread.
    @objc(backendWithEncoderURL:decoderURL:error:)
    public static func backend(encoderURL: URL?, decoderURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXAdaINNet()
        if let encoderURL {
            try loadEncoderWeights(into: net.encoder, from: encoderURL)
        }
        if let decoderURL {
            try loadDecoderWeights(into: net.decoder, from: decoderURL)
        }
        let holder = NFKMLXAdaINHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { content, request in
            guard let styleValue = request.input(forKey: NFKInputControl),
                  let style = try? NFKMLXImageBridge.tensor(from: styleValue, channels: 3,
                                                            colorSpace: CGColorSpaceCreateDeviceRGB()) else {
                // With no style image the content passes through the pair unchanged, which is the
                // reconstruction the decoder is trained for.
                let batched = content.reshaped([1, content.shape[0], content.shape[1], 3])
                return clip(holder.net.stylize(content: batched, style: batched)[0], min: 0, max: 1)
            }
            let strength = (request.parameter(forKey: NFKParameterStrength) as? NSNumber)?.floatValue ?? 1.0
            let stylized = holder.net.stylize(
                content: content.reshaped([1, content.shape[0], content.shape[1], 3]),
                style: style.reshaped([1, style.shape[0], style.shape[1], 3]),
                strength: max(0, min(1, strength)))
            return clip(stylized[0], min: 0, max: 1)
        }
    }

    /// Downloads both checkpoints from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:encoderPath:decoderPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, encoderPath: String, decoderPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let encoderURL = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: encoderPath,
                                                       revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        let decoderURL = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: decoderPath,
                                                       revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(encoderURL: encoderURL, decoderURL: decoderURL)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:encoderPath:decoderPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, encoderPath: String, decoderPath: String, revision: String?,
                               cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        Task.detached(priority: .userInitiated) {
            do {
                let built = try backend(repo: repo, encoderPath: encoderPath, decoderPath: decoderPath,
                                        revision: revision, cacheDirectoryURL: cacheDirectoryURL)
                completionHandler(built, nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// Registers `adain` with `NFKMLXModelRegistry`. The registry hands over one URL, which is the
    /// decoder: the encoder is the released normalized VGG and is named through the download factory.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(encoderURL: nil, decoderURL: weightsURL)
        }
    }

    /// Loads the released normalized VGG. Its keys are the flat `nn.Sequential` indices, which the
    /// `layers` array carries.
    static func loadEncoderWeights(into encoder: NFKMLXAdaINEncoder, from url: URL) throws {
        try loadStack(into: encoder, from: url)
    }

    /// Loads the released decoder, whose keys are the same flat indices.
    static func loadDecoderWeights(into decoder: NFKMLXAdaINDecoder, from url: URL) throws {
        try loadStack(into: decoder, from: url)
    }

    private static func loadStack(into module: Module, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let truncatesAt = module is NFKMLXAdaINEncoder ? NFKMLXAdaINEncoder.outputIndex : Int.max
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            // A decoder trained through a wrapper module carries that attribute name on every key.
            let stripped = key.hasPrefix("net.") ? String(key.dropFirst("net.".count)) : key
            // The released VGG carries all 19 layers; only the truncated prefix has a home here.
            guard let index = Int(stripped.split(separator: ".").first.map(String.init) ?? ""),
                  index <= truncatesAt else { return nil }
            return ("layers.\(stripped)",
                    checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: module)
    }
}

private final class NFKMLXAdaINHolder: @unchecked Sendable {
    let net: NFKMLXAdaINNet
    init(_ net: NFKMLXAdaINNet) { self.net = net }
}
